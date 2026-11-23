"""F29 (26 Oct 2026): the financing workbook's methods in the model (v6.2 since 30 Oct 2026).

The legacy valuation's Approach 2 on the model's own rows (after-tax flows at the
workbook's post-tax operating rates, end-of-year discounting, capital allowances on
the workbook's net book values), the acquisition bases, P1's Scenario 2 rate at
another stake, and the social price of the risk the state holds.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.finance import (
    coordination_at_stake, require_snapshot_schema, systematic_premium, yield_analogue,
)
from scripts.public_ownership.legacy_valuation import (
    OWNER_VALUE_WORKBOOK, present_value, settings_from_config, value_legacy_fleet, workbook_bases,
)
from tests.unit.test_public_ownership_legacy_valuation import PUBLIC_RATE, _row, _stations


TAU, WDA = 0.25, 0.18


def _snapshot():
    return {
        "schema_version": 5,
        "assumptions": {
            "corporation_tax_rate": {"value": TAU}, "legacy_capital_allowance_rate": {"value": WDA},
            "equity_risk_premium": {"value": 0.0522}, "total_market_return": {"value": 0.0683},
            "risk_free_real": {"value": 0.026}, "public_coordination_retained_premium": {"value": 0.0},
            "public_coordination_state_equity_share": {"value": 0.30},
        },
        "technologies": [
            {"technology": "Onshore wind", "legacy_owner_rate": 0.0478, "post_tax_wacc": 0.06,
             "gearing": {"value": 0.7}, "equity_beta": 1.5,
             "wacc": {"private": 0.08, "public_coordination": 0.07},
             "wacc_by_basis": {"notional_gearing_shielded": {"private": 0.08, "public_coordination": 0.07},
                               "fixed_tmr_shielded": {"private": 0.075, "public_coordination": 0.066}},
             "corporation_tax_wedge": {"private": 0.004, "public_coordination": 0.003}},
            {"technology": "Large-scale nuclear", "legacy_owner_rate": 0.055, "post_tax_wacc": 0.063},
        ],
        "legacy_cohorts": [
            {"cohort": "Onshore wind - RO", "technology": "Onshore wind", "support_regime": "RO",
             "historic_capex_gbp_per_kw": 1300.0, "accounting_life_years": 25},
        ],
    }


def _register():
    onshore = _row("onshore_roc::legacy", "wind_onshore", "ROC", market=100.0, fixed_om=20.0, roc=60.0,
                   commissioning=2012)
    hpc = _row("Hinkley Point C::legacy", "nuclear", "CfD", market=800.0, fuel=100.0, fixed_om=300.0, cfd=900.0,
               cfd_round="HPC", award=2016)
    rows = pd.DataFrame([onshore, hpc])
    rows["finance_technology"] = ["Onshore wind", "Large-scale nuclear"]
    rows["as_new_capex_gbp"] = [np.nan, 2.0e9]
    rows["economic_life_years"] = [np.nan, 60.0]
    return rows


@pytest.fixture(scope="module")
def valued():
    settings = settings_from_config({"horizon_year": 2070, "capacity_from_year": None})
    valuation, _, summary = value_legacy_fleet(_register(), PUBLIC_RATE, settings, _stations(), _snapshot())
    return valuation.set_index("asset_id"), summary


def test_approach_2_is_after_tax_plus_the_capital_allowances(valued):
    v, summary = valued
    assert summary["owner_value_method"] == OWNER_VALUE_WORKBOOK and summary["owners_value_discounting"] == "end_of_year"
    for asset, row in v.iterrows():
        r = row["owner_discount_rate"]
        allowances = TAU * WDA / (r + WDA) * row["net_book_value_gbp"]
        assert row["capital_allowance_pv_gbp"] == pytest.approx(allowances)
        assert row["pv_market_gbp"] == pytest.approx((1 - TAU) * row["pv_market_pre_tax_gbp"] + allowances), asset


def test_rates_are_the_workbooks_post_tax_rates_and_hinkley_takes_new_build_nuclear(valued):
    v, _ = valued
    assert v.loc["onshore_roc::legacy", "owner_discount_rate"] == pytest.approx(0.0478)
    assert v.loc["onshore_roc::legacy", "owner_discount_rate_source"] == "workbook_operating_post_tax:Onshore wind"
    assert v.loc["Hinkley Point C::legacy", "owner_discount_rate"] == pytest.approx(0.063)
    assert v.loc["Hinkley Point C::legacy", "owner_discount_rate_source"].startswith("workbook_new_build_post_tax")


def test_book_values_are_the_workbooks_historic_capex_written_down(valued):
    v, _ = valued
    # 100 MW of 2012 RO onshore, 18 years into a 25-year accounting life in 2030.
    assert v.loc["onshore_roc::legacy", "net_book_value_gbp"] == pytest.approx(100_000 * 1300.0 * (1 - 18 / 25))
    assert v.loc["onshore_roc::legacy", "net_book_value_source"].startswith("workbook_cohort:")
    # A row the workbook has no cohort for (Hinkley is named) takes the register's as-new capex.
    assert v.loc["Hinkley Point C::legacy", "net_book_value_source"] == "register_as_new_capex"


def test_end_of_year_discounting():
    flows = np.full((1, 10), 1.0)
    assert present_value(flows, 0.05, first_period=1.0)[0] == pytest.approx((1 - 1.05 ** -10) / 0.05)
    assert present_value(flows, 0.05)[0] == pytest.approx((1 - 1.05 ** -10) / 0.05 * 1.05)


def test_the_acquisition_bases_floor_each_row_and_pass_the_ro_through():
    rows = pd.DataFrame({
        "pv_market_gbp": [100.0, 50.0, -10.0], "pv_no_support_gbp": [40.0, 30.0, -20.0],
        "net_book_value_gbp": [70.0, 20.0, 5.0], "support_type": ["ROC", "CfD", "merchant"]})
    bases = workbook_bases(rows)
    assert bases == {"market_value": 150.0, "no_support": 70.0, "book_value": 95.0,
                     "ro_pass_through": 40.0 + 50.0 + 0.0}


def test_p1_at_another_stake_moves_by_the_workbook_formula():
    snapshot = _snapshot()
    unchanged = coordination_at_stake(snapshot, 0.30)
    assert unchanged["technologies"][0]["wacc"]["public_coordination"] == pytest.approx(0.07)
    half = coordination_at_stake(snapshot, 0.15)
    row = half["technologies"][0]
    # (1 - g)(1 - phi) beta_e ERP (sigma0 - sigma) / (1 - tau), on each ERP convention.
    post_tax = 0.3 * 1.0 * 1.5 * 0.0522 * 0.15
    assert row["wacc"]["public_coordination"] == pytest.approx(0.07 + post_tax / (1 - TAU))
    assert row["wacc_by_basis"]["notional_gearing_shielded"]["public_coordination"] == row["wacc"]["public_coordination"]
    tmr_post_tax = 0.3 * 1.5 * (0.0683 - 0.026) * 0.15
    assert row["wacc_by_basis"]["fixed_tmr_shielded"]["public_coordination"] == pytest.approx(
        0.066 + tmr_post_tax / (1 - TAU))
    assert row["corporation_tax_wedge"]["public_coordination"] == pytest.approx(0.003 + post_tax * TAU / (1 - TAU))
    assert half["assumptions"]["public_coordination_state_equity_share"]["value"] == 0.15
    # The input is not edited.
    assert snapshot["technologies"][0]["wacc"]["public_coordination"] == 0.07


def test_the_social_price_of_the_states_risk_and_the_gilt_path_it_follows():
    row = {"economic_rate_by_premium": {"social_flat_19": 0.046, "none": 0.027, "social_beta_20": 0.042,
                                        "market_erp": 0.067},
           "operating_economic_rate_by_premium": {"social_flat_19": 0.046, "none": 0.027, "social_beta_20": 0.040}}
    own = {"new_build_generation": "public", "public_return": "public_rate"}
    # Meeting notes (user, 10 Nov 2026): the central is gilts alone; the 1.9% is the Scenario 4 sensitivity.
    assert systematic_premium(row, own) == 0.0
    flat = dict(own, systematic_premium="social_flat_19")
    assert systematic_premium(row, flat) == pytest.approx(0.019)
    assert systematic_premium(row, flat, operating=True) == pytest.approx(0.019)
    assert systematic_premium(row, dict(own, systematic_premium="social_beta_20"), operating=True) == pytest.approx(0.013)
    assert systematic_premium(row, dict(own, systematic_premium="none")) == 0.0
    assert systematic_premium({}, own) == 0.0  # a fixture without the workbook's economic rates
    with pytest.raises(ValueError, match="retired 30 Oct 2026"):
        systematic_premium(row, dict(own, systematic_premium="social_p30"), operating=True)
    assert yield_analogue({"public_return": "cepa_decomposition"}) == "Sc2"
    assert yield_analogue(own) == "Sc3"
    assert yield_analogue(dict(own, legacy_fleet="acquire_all")) == "Sc4"


def test_an_older_workbooks_snapshot_is_refused():
    require_snapshot_schema({"schema_version": 5})
    require_snapshot_schema({})  # a fixture without a version
    for old in (3, 4):  # v11-2, v2.0
        with pytest.raises(ValueError, match=f"schema {old}"):
            require_snapshot_schema({"schema_version": old})


def _repository_snapshot():
    import yaml
    from pathlib import Path

    path = Path("resources/public_ownership/workbook_v5_assumptions.yaml")
    if not path.exists():
        pytest.skip("snapshot not built")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v6_2_central_rates():
    """v6.2: 55% notional gearing, the 15-year gilt read directly, a flat 1.9% premium on every technology."""
    snap = _repository_snapshot()
    assert snap["schema_version"] == 5
    assert snap["switches"]["gearing_basis"] == "RIIO-3 notional 55%"
    assert snap["switches"]["social_premium_beta_basis"] == "Flat premium (no beta)"
    cash = snap["assumptions"]["public_borrowing_real"]["value"]
    for row in snap["technologies"]:
        assert row["gearing"]["value"] == pytest.approx(0.55), row["technology"]
        for key in ("economic_rate_by_premium", "operating_economic_rate_by_premium"):
            assert row[key]["social_flat_19"] == pytest.approx(cash + 0.019), row["technology"]
            assert row[key]["social_flat_10"] == pytest.approx(cash + 0.010), row["technology"]
            assert row[key]["none"] == pytest.approx(cash), row["technology"]
        assert row["economic_rate_by_premium"]["social_beta_20"] == pytest.approx(cash + 0.02 * row["asset_beta"])


def test_the_contested_gilt_parameters_are_the_workbooks():
    """contested_fiscal_costs.csv carries workbook v6.2's Build-Out section F inputs (user, 30 Oct 2026)."""
    from scripts.public_ownership.exchequer_ledger import load_contested_costs

    snap = _repository_snapshot()
    a = {k: v["value"] for k, v in snap["assumptions"].items()}
    csv = load_contested_costs("data/public_ownership/financing/contested_fiscal_costs.csv")
    assert csv["yield_bp_per_pp_debt_to_gdp"] == a["yield_response_bp_per_pp"]
    assert csv["stress_case_loss_share"] == a["gilt_stress_case_loss_share"]
    assert csv["counterfactual_offset_share"] == a["gilt_counterfactual_offset_share"]
    assert csv["gdp_2024_gbp"] == pytest.approx(a["gdp_2024_gbp_bn"] * 1e9)
    assert csv["real_gdp_growth"] == a["real_gdp_growth"]
    assert csv["marketable_gilt_stock_2024_gbp"] == pytest.approx(a["gilt_marketable_stock_2024_gbp_bn"] * 1e9)
    assert csv["average_gilt_maturity_years"] == a["gilt_average_maturity_years"]


def test_the_repository_snapshot_matches_the_workbook_cohort_values():
    """The extracted cohorts carry the workbook's own A2 numbers: our formula reproduces them."""
    snap = _repository_snapshot()
    tau = snap["assumptions"]["corporation_tax_rate"]["value"]
    wda = snap["assumptions"]["legacy_capital_allowance_rate"]["value"]
    for cohort in snap["legacy_cohorts"]:
        if snap["switches"].get("legacy_gas_valuation_basis") == "Asset register (Lyden)" and \
                cohort["technology"] == "Gas CCGT":
            # v6.2 values the gas cohort from the asset register (Legacy Fleet D99), not by A2.
            assert cohort["market_value_gbp_m"] == pytest.approx(
                snap["assumptions"]["legacy_gas_register_value_gbp_bn"]["value"] * 1e3)
            continue
        r = cohort["owner_rate"]
        assert cohort["capital_allowance_pv_factor"] == pytest.approx(wda / (r + wda)), cohort["cohort"]
        expected = max(0.0, (1 - tau) * cohort["market_value_pre_tax_gbp_m"]
                       + tau * wda / (r + wda) * cohort["net_book_value_gbp_m"])
        assert cohort["market_value_gbp_m"] == pytest.approx(expected, rel=1e-9), cohort["cohort"]


def test_the_register_books_the_states_risk_off_the_bill():
    """socialised_risk_gbp = the state's capital charge x (CRF(r + premium, n) / CRF(r, n) - 1), state developer only."""
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
    from scripts.public_ownership.finance import capital_recovery_factor
    from tests.unit.test_public_ownership_finance import NO_OVERHEAD, _costs
    from tests.unit.test_public_ownership_levers import SCENARIOS

    assets = pd.DataFrame([{
        "model_year": 2030, "component_type": "Generator", "asset_id": "new-solar", "carrier": "solar_pv",
        "finance_technology": "Solar PV", "finance_vintage": "new_build", "capacity_mw": 1.0, "dispatch_mwh": 1_000.0,
        "evidence_status": "resolved", "commissioning_year": 2027, "support_type": "merchant",
    }])
    rates = {"private": 0.072, "public_coordination": 0.058, "public_ownership_new_build": 0.0271,
             "public_ownership_legacy": 0.0271}
    snapshot = {"rate_basis": "real_CPI", "assumptions": {"public_coordination_state_equity_share": {"value": 0.3}},
                "technologies": [{"technology": "Solar PV", "risk_rating": "L-M", "wacc": rates,
                                  "economic_rate_by_premium": {"social_flat_19": 0.0461, "none": 0.0271,
                                                               "market_erp": 0.0589}}]}
    sens = {"structural": [], "one_at_a_time": [("systematic_premium", ["social_flat_19", "market_erp"])], "named": {},
            "bounding_pair": False}
    register, totals = calculate_financing_scenarios(assets, _costs(), snapshot, scenarios=SCENARIOS,
                                                     sensitivities=sens, price_year=2024,
                                                     state_operating_overhead=NO_OVERHEAD)
    by_case = register.set_index("case_id")
    assert by_case.loc["private", "socialised_risk_gbp"] == 0.0
    assert by_case.loc["public_control", "socialised_risk_gbp"] == 0.0  # a stake books none
    # Meeting notes (user, 10 Nov 2026): the central books none; the 1.9% case books the premium.
    assert by_case.loc["public_ownership", "socialised_risk_gbp"] == 0.0
    row = by_case.loc["public_ownership@systematic_premium_flat_19"]
    life = float(row["remaining_life_years"])  # the row's own annuity term: 35 years less 3 built
    assert life == 32.0
    ratio = capital_recovery_factor(0.0461, life) / capital_recovery_factor(0.0271, life) - 1.0
    assert row["socialised_risk_gbp"] == pytest.approx(row["annual_capital_recovery_gbp"] * ratio, rel=1e-6)
    erp = by_case.loc["public_ownership@systematic_premium_market_erp", "socialised_risk_gbp"]
    assert erp > row["socialised_risk_gbp"] > 0.0
    # Never on the bill: the required revenue does not move with the premium.
    t = totals.set_index("case_id")["annual_required_revenue_gbp"]
    assert t["public_ownership@systematic_premium_flat_19"] == pytest.approx(t["public_ownership"])
    assert t["public_ownership@systematic_premium_market_erp"] == pytest.approx(t["public_ownership"])
