"""The two workflow checks added on 30 Sep 2026: the BM ownership tables and the auction calibration."""

import pandas as pd
import pytest

from scripts.public_ownership import derive_cost_calibration as calib
from scripts.public_ownership import verify_ownership_tables as verify


def _shares(share_b: float = 0.3) -> pd.DataFrame:
    return pd.DataFrame({
        "case_id": ["public_ownership", "public_ownership", "private"],
        "generator": ["a", "b", "a"],
        "capacity_mw": [100.0, 50.0, 100.0],
        "state_capacity_mw": [100.0, 50.0 * share_b, 0.0],
        "state_share": [1.0, share_b, 0.0],
        "state_commissioning_year": [2027.0, float("nan"), float("nan")],
    })


def test_matching_tables_pass_and_nan_years_are_equal():
    report = verify.compare_tables(_shares(), _shares(), verify.SHARE_COLUMNS)
    assert not report.get("failed_columns") and not report["missing"] and not report["extra"]


def test_a_moved_share_fails():
    report = verify.compare_tables(_shares(), _shares(share_b=0.31), verify.SHARE_COLUMNS)
    assert "state_share" in report["failed_columns"]


def test_a_generator_missing_from_the_saved_table_fails():
    report = verify.compare_tables(_shares(), _shares().iloc[[0, 2]], verify.SHARE_COLUMNS)
    assert report["missing"] == [["public_ownership", "b"]]


def test_pairs_group_cases_by_table_and_keep_the_at_sign():
    pairs = verify.parse_pairs(["t.csv::public_ownership", "t.csv::public_ownership@fully_nationalised_market_value",
                                "t.csv::public_ownership"])
    assert pairs == {"t.csv": ["public_ownership", "public_ownership@fully_nationalised_market_value"]}


def _register(factor: float, insurance_scales: bool) -> pd.DataFrame:
    """One private offshore AR7 new-build row whose cost exceeds its strike revenue, at calibration ``factor``."""
    capital, insurance, other_fixed = 900.0, 30.0, 300.0
    scaled_insurance = insurance * factor if insurance_scales else insurance
    return pd.DataFrame({
        "case_id": ["private"], "evidence_status": ["resolved"],
        "carrier": ["wind_offshore" if not insurance_scales else "marine"],
        "cfd_round": ["AR7" if not insurance_scales else "AR5"], "finance_vintage": ["new_build"],
        "capacity_mw": [10.0], "dispatch_mwh": [10.0],
        "market_revenue_gbp": [600.0], "support_revenue_gbp": [300.0], "fuel_carbon_gbp": [0.0],
        "annual_capital_recovery_gbp": [capital * factor], "annual_insurance_gbp": [scaled_insurance],
        "annual_required_revenue_gbp": [capital * factor + scaled_insurance + other_fixed],
        "capex_calibration_factor": [factor],
    })


@pytest.mark.parametrize("insurance_scales", [False, True])
def test_the_calibration_derives_the_same_factor_from_a_calibrated_register(insurance_scales):
    uncalibrated = calib.derive(_register(1.0, insurance_scales), "test")
    k = float(uncalibrated["capex_factor"].iloc[0])
    assert 0.0 < k < 1.0
    recalibrated = calib.derive(_register(k, insurance_scales), "test")
    assert float(recalibrated["capex_factor"].iloc[0]) == pytest.approx(k, abs=1e-6)


def test_the_calibrated_cohort_breaks_even_at_its_strike():
    k = float(calib.derive(_register(1.0, False), "test")["capex_factor"].iloc[0])
    row = _register(k, False).iloc[0]
    revenue = row.market_revenue_gbp + row.support_revenue_gbp - row.fuel_carbon_gbp
    assert row.annual_required_revenue_gbp == pytest.approx(revenue, abs=1e-3)


def test_drift_is_reported_per_cohort():
    fresh = calib.derive(_register(1.0, False), "test")
    committed = fresh.assign(capex_factor=fresh["capex_factor"] + 0.01)
    report = calib.compare(committed, fresh)
    assert report["drifted"] and report["max_abs_diff"] == pytest.approx(0.01, abs=1e-9)


def _cheap_register(carrier: str, factor: float = 1.0) -> pd.DataFrame:
    """One private AR7a new-build row whose cost is BELOW its strike revenue, at calibration ``factor``."""
    capital, insurance, other_fixed = 500.0, 20.0, 200.0
    return pd.DataFrame({
        "case_id": ["private"], "evidence_status": ["resolved"], "carrier": [carrier], "cfd_round": ["AR7a"],
        "finance_vintage": ["new_build"], "capacity_mw": [10.0], "dispatch_mwh": [10.0],
        "market_revenue_gbp": [600.0], "support_revenue_gbp": [300.0], "fuel_carbon_gbp": [0.0],
        "annual_capital_recovery_gbp": [capital * factor], "annual_insurance_gbp": [insurance * factor],
        "annual_required_revenue_gbp": [capital * factor + insurance * factor + other_fixed],
        "capex_calibration_factor": [factor],
    })


@pytest.mark.parametrize("carrier", ["wind_onshore", "solar_pv"])
def test_onshore_and_solar_ar7a_are_calibrated_both_ways(carrier):
    # F19 decision 2: a cohort cheaper than its strike is scaled up (k > 1), reproducibly, and breaks even.
    table = calib.derive(_cheap_register(carrier), "test")
    assert set(table["cfd_round"]) == {"AR7a"}  # AR8 is its own cohort since F18 (27 Oct 2026)
    k = float(table["capex_factor"].iloc[0])
    assert k == pytest.approx((900.0 - 200.0) / 520.0) and k > 1.0
    assert table["status"].iloc[0] == calib.STATUS_BOTH_WAYS
    assert float(calib.derive(_cheap_register(carrier, k), "test")["capex_factor"].iloc[0]) == pytest.approx(k, abs=1e-6)


def test_a_cheap_offshore_cohort_is_still_left_alone():
    cheap = _cheap_register("wind_offshore").assign(cfd_round="AR6")
    assert calib.derive(cheap, "test").empty
