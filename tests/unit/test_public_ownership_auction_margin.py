"""Follow-up F6 (user, 4 Oct 2026): the auction margin of the calibrated CfD cohorts.

CfD auctions pay as clear, so projects cheaper than the marginal one keep a margin above
their required return.  The calibration (trust the auction) solves k(0) so the cohort just
breaks even; lever ``auction_margin`` keeps a share m of contract revenue above full cost
instead: k(m) = k(0) x (1 - m x slope), slope = revenue / ((capital charge + capex-scaled
insurance) x k(0)).  Central 5%, none 0, high 10%, in every case alike.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership.calculate_financing_scenarios import (
    LEVER_PARAMETER_DEFAULTS, apply_cost_calibration, auction_margin_capex_scale, calibration_margin_slope,
    load_cost_calibration,
)
from scripts.public_ownership.derive_cost_calibration import derive
from scripts.public_ownership.levers import (
    ALL_LEVERS, GATED_LEVERS, LEVER_AUCTION_MARGIN, LEVERS, SCOPE_ALL, comparator_for_settings, lever_default,
)
from config.config_loader import load_defaults

TABLE = Path("data/public_ownership/financing/private_cost_calibration.csv")


def test_the_lever_is_a_comparator_lever_with_a_central_default():
    lever = ALL_LEVERS[LEVER_AUCTION_MARGIN]
    # A 2030 lever: in the main catalogue (with explorer labels), not the gated 2050 set.
    assert LEVER_AUCTION_MARGIN in LEVERS and LEVER_AUCTION_MARGIN not in GATED_LEVERS
    assert lever.scope == SCOPE_ALL and lever.register
    assert lever_default(LEVER_AUCTION_MARGIN) == "central"
    assert LEVER_PARAMETER_DEFAULTS["auction_margin_share"] == {"central": 0.05, "none": 0.0, "high": 0.10}
    po = load_defaults()["public_ownership"]
    assert po["lever_parameters"]["auction_margin_share"] == LEVER_PARAMETER_DEFAULTS["auction_margin_share"]
    assert "auction_margin" in po["sensitivities"]["one_at_a_time"]


def test_k_at_a_margin_matches_the_derivation_for_every_cohort():
    table = load_cost_calibration(TABLE)
    for row in table.itertuples(index=False):
        k0, revenue = float(row.capex_factor), float(row.cohort_contract_revenue_gbp)
        capital, insurance = float(row.cohort_capital_charge_private_gbp), float(row.cohort_capex_scaled_insurance_gbp)
        other = float(row.cohort_other_costs_gbp)
        slope = calibration_margin_slope(row)
        for m in (0.05, 0.10):
            direct = ((1.0 - m) * revenue - (other - insurance)) / (capital + insurance)
            assert k0 * float(auction_margin_capex_scale([slope], m)[0]) == pytest.approx(direct, rel=1e-5), (
                row.carrier, row.cfd_round, m)


def test_calibration_records_the_slope_and_which_insurance_scales():
    table = load_cost_calibration(TABLE)
    small = table.loc[((table["carrier"] == "wind_offshore") & (table["cfd_round"] == "AR7"))
                      | ((table["carrier"] == "marine") & (table["cfd_round"] == "AR4"))]
    merged = pd.DataFrame({"carrier": ["wind_offshore", "marine", "solar_pv"], "cfd_round": ["AR7", "AR4", "AR7"],
                           "finance_vintage": ["new_build"] * 3, "capex_power_gbp_per_kw": [2000.0, 5000.0, 600.0]})
    out = apply_cost_calibration(merged, small).set_index("carrier")
    assert out.loc["wind_offshore", "capex_calibration_margin_slope"] > 0
    assert out.loc["solar_pv", "capex_calibration_margin_slope"] == 0.0
    # Offshore insurance is per kW; tidal's is a share of capex, so it scales with the margin.
    assert not out.loc["wind_offshore", "capex_calibration_insurance_scales"]
    assert out.loc["marine", "capex_calibration_insurance_scales"]


def test_a_margin_the_cohort_cannot_carry_is_refused():
    with pytest.raises(ValueError, match="larger than the cohort can carry"):
        auction_margin_capex_scale([2.0], 0.5)
    assert auction_margin_capex_scale([0.0, 1.5], 0.10).tolist() == pytest.approx([1.0, 0.85])


def test_at_the_margin_the_developer_keeps_that_share_and_the_derivation_recovers_k0():
    # One offshore AR7 cohort: uncalibrated capital charge 100, other costs 40, contract revenue 110.
    capital, other, revenue, m = 100.0, 40.0, 110.0, 0.05
    k0 = (revenue - other) / capital
    slope = revenue / (capital * k0)
    scale = float(auction_margin_capex_scale([slope], m)[0])
    capital_cal = capital * k0 * scale
    register = pd.DataFrame([{
        "case_id": "private", "evidence_status": "resolved", "carrier": "wind_offshore", "cfd_round": "AR7",
        "finance_vintage": "new_build", "capacity_mw": 1.0, "dispatch_mwh": 1.0,
        "market_revenue_gbp": 50.0, "support_revenue_gbp": revenue - 50.0, "fuel_carbon_gbp": 0.0,
        "annual_required_revenue_gbp": capital_cal + other, "annual_capital_recovery_gbp": capital_cal,
        "annual_insurance_gbp": 0.0, "capex_calibration_factor": k0, "auction_margin_capex_scale": scale}])
    assert revenue - (capital_cal + other) == pytest.approx(m * revenue)
    table = derive(register, "synthetic")
    assert float(table.loc[table["cfd_round"].eq("AR7"), "capex_factor"].iloc[0]) == pytest.approx(k0, rel=1e-6)


def test_a_margin_case_is_measured_against_private_at_the_same_margin():
    settings = {
        "private": {}, "private@auction_margin_high": {LEVER_AUCTION_MARGIN: "high"},
        "public_control": {"new_build_generation": "state_stake"},
        "public_control@auction_margin_high": {"new_build_generation": "state_stake", LEVER_AUCTION_MARGIN: "high"},
    }
    assert comparator_for_settings("public_control@auction_margin_high", settings) == "private@auction_margin_high"
    assert comparator_for_settings("public_control", settings) == "private"


def test_the_financing_scales_only_calibrated_rows_and_every_case_alike():
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
    from tests.unit.test_public_ownership_cfd_repricing import NO_OVERHEAD, _asset, _costs, _snapshot

    assets = pd.DataFrame([
        _asset("ar7_a", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6),
        _asset("ar6_new", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR6", "pre_policy", 2024, 800.0, 3.0e6),
    ])
    table = pd.DataFrame([{"carrier": "wind_offshore", "cfd_round": "AR7", "finance_vintage": "new_build",
                           "capex_factor": 0.8, "cohort_contract_revenue_gbp": 3.0e8,
                           "cohort_capital_charge_private_gbp": 2.5e8, "cohort_capex_scaled_insurance_gbp": 0.0}])
    slope = 3.0e8 / (2.5e8 * 0.8)

    def run(central_share):
        register, _ = calculate_financing_scenarios(
            assets, _costs(), _snapshot(), price_year=2024, axes={}, state_operating_overhead=NO_OVERHEAD,
            cost_calibration=table,
            lever_parameter_block={"auction_margin_share": {"central": central_share, "none": 0.0, "high": 0.10}})
        return register.set_index(["case_id", "asset_id"])

    with_margin, without = run(0.05), run(0.0)
    capital = "annual_capital_recovery_gbp"
    ratio = (with_margin[capital].astype(float) / without[capital].astype(float)).unstack("asset_id")
    assert ratio["ar7_a"].to_numpy() == pytest.approx(np.full(len(ratio), 1.0 - 0.05 * slope))
    assert ratio["ar6_new"].to_numpy() == pytest.approx(np.ones(len(ratio)))
    assert "auction_margin_capex_scale" in with_margin.columns


# ── follow-up F7: self-insurance on wholly state-owned plant ────────────────


def test_wholly_state_owned_rows_save_the_insurers_loading_and_private_does_not():
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
    from tests.unit.test_public_ownership_cfd_repricing import NO_OVERHEAD, _asset, _costs, _snapshot

    assets = pd.DataFrame([
        _asset("ar7_a", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6),
        _asset("solar_new", "solar_pv", "Solar PV", "new_build", "merchant", "", "not_applicable", np.nan, 300.0, 3.0e5),
    ])
    insurance = {"insurance": {"enabled": True, "gbp_per_kw_year_by_technology": {"Offshore wind": 11.22},
                               "share_of_capex_per_year": 0.004}}

    def run(central):
        register, _ = calculate_financing_scenarios(
            assets, _costs(), _snapshot(), price_year=2024, axes={}, state_operating_overhead=NO_OVERHEAD,
            generator_costs=insurance,
            lever_parameter_block={"self_insurance_share": {"central": central, "none": 0.0, "high": 0.5}})
        return register.set_index(["case_id", "asset_id"])

    saving, full = run(0.35), run(0.0)
    ins = "annual_insurance_gbp"
    ratio = (saving[ins].astype(float) / full[ins].astype(float)).unstack("asset_id")
    # Private keeps commercial cover; P2 owns the post-2025 build outright and saves 35%.
    assert ratio.loc["private"].to_numpy() == pytest.approx([1.0, 1.0])
    assert ratio.loc["public_ownership"].to_numpy() == pytest.approx([0.65, 0.65])
    # P1's 30% stake (public_coordination in this fixture) leaves the project company insuring commercially.
    assert ratio.loc["public_coordination"].to_numpy() == pytest.approx([1.0, 1.0])
    assert "self_insurance_saving_gbp" in saving.columns
    assert "self_insurance_saving_gbp" not in full.columns
