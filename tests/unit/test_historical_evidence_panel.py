"""Guard evidence completeness and prevent held-out target leakage."""

import pandas as pd
import pytest

from scripts.consumer_bill.build_historical_evidence_panel import procurement_diagnostics, validate_periods


def periods():
    return pd.DataFrame({
        "cap_period": ["q1", "q2", "q3", "q4"],
        "delivery_start": ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"],
        "delivery_end": ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"],
        "source_energy_definition": ["energy"] * 4,
        "pypsa_spot_price_gbp_per_mwh": [10., 20., 30., 40.],
        "published_energy_allowance_gbp_per_3100kwh_ex_vat": [62., 124., 186., 248.],
    })


def test_calendar_rejects_missing_day_and_duplicate():
    frame = periods()
    assert validate_periods(frame, 2024).calendar_weight.sum() == pytest.approx(1)
    frame.loc[1, "delivery_start"] = "2024-04-02"
    with pytest.raises(ValueError, match="missing delivery days"):
        validate_periods(frame, 2024)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_periods(pd.concat([periods(), periods().iloc[:1]]), 2024)


def test_held_out_targets_do_not_change_fitted_coefficient():
    frame = periods()
    first, _ = procurement_diagnostics({2024: frame})
    frame.loc[2:, "published_energy_allowance_gbp_per_3100kwh_ex_vat"] *= 20
    second, summary = procurement_diagnostics({2024: frame})
    assert first.diagnostic_multiplier.tolist() == second.diagnostic_multiplier.tolist()
    assert second.diagnostic_multiplier.iloc[0] == pytest.approx(2)
    assert second.loc[second.split.eq("train"), "diagnostic_prediction_gbp"].isna().all()
    assert summary["forecast_validated"] is False


def test_incompatible_target_definitions_block_cross_year_fit():
    earlier = periods()
    for name in ["delivery_start", "delivery_end"]:
        earlier[name] = earlier[name].str.replace("2024", "2020")
    earlier["source_energy_definition"] = "different cost boundary"
    _, summary = procurement_diagnostics({2020: earlier, 2024: periods()})
    assert summary["cross_year_fit_status"] == "blocked_incompatible_target_definitions"


def test_nonfinite_prices_rejected():
    frame = periods()
    frame.loc[0, "pypsa_spot_price_gbp_per_mwh"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        procurement_diagnostics({2024: frame})
