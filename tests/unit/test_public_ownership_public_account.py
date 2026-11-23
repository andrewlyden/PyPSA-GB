"""WP-D, WP-E, WP-I, WP-J and WS9 items 4 to 9 of the public-ownership bill link (Release 2)."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.cfd_repricing import (
    ROUTE_CFD_BILL, ROUTE_CFD_PUBLIC_ACCOUNT, ROUTE_MERCHANT, base_case, is_variant, per_row_savings,
    repricing_settings, route_summary, savings_with_sensitivities,
)
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    LINE_CFD_AT_COST, LINE_LDES, LINE_SHORTFALL, LINE_W1, extra_lines,
)
from scripts.public_ownership.public_account import (
    GROUP_CM, GROUP_DPA, GROUP_LDES, GROUP_MERCHANT, GROUP_R1, GROUP_R2, account_settings, build_public_account,
)
from scripts.public_ownership.reference_runs import validate_public_ownership_config
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.settlement import attach_settlement

from tests.unit.test_public_ownership_cfd_repricing import _asset, _costs, _snapshot
from config.config_loader import load_defaults


#: Item OH1: the state's holding-company cost, switched off in fixtures that
#: pin other mechanics.  See tests/unit/test_public_ownership_state_overhead.py.
NO_OVERHEAD = {"enabled": False}

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
PUBLIC = Path("resources/public_ownership")


@pytest.fixture(scope="module")
def defaults() -> dict:
    return load_defaults()


def _axes(**alternatives):
    """An axis declaration with the named axes on the given alternatives."""
    return {name: {"alternatives": list(values)} for name, values in alternatives.items()}


def _central(**values):
    """An axis declaration moving the named levers' central value on every case (item K1 lever settings)."""
    return {name: {"central": value} for name, value in values.items()}


def _settled_register(axes=None):
    """A register with a re-priced CfD (R1), a pre-policy CfD (R2), merchant solar, gas and two storage units."""
    assets = pd.DataFrame([
        _asset("ar7", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6),
        _asset("ar6", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR6", "pre_policy", 2024, 800.0, 3.0e6),
        _asset("solar", "solar_pv", "Solar PV", "new_build", "merchant", "", "not_applicable", np.nan, 500.0, 5.0e5),
        _asset("gas", "CCGT", "Gas generation - unabated", "new_build", "", "", "not_applicable", np.nan, 400.0, 1.0e6),
        _asset("beccs", "CCS Biomass", "Biomass - with CCUS - mature", "new_build", "", "", "not_applicable", np.nan, 100.0, 1.0e5),
        _asset("ldes", "LAES", "Novel long duration energy storage (LDES)", "new_build", "", "", "not_applicable", np.nan,
               100.0, 5.0e4, component_type="StorageUnit", energy_capacity_mwh=400.0),
        _asset("battery", "Battery", "Lithium batteries", "new_build", "", "", "not_applicable", np.nan, 100.0, 5.0e4,
               component_type="StorageUnit", energy_capacity_mwh=200.0),
    ])
    settlement = pd.DataFrame({
        "component_type": ["Generator"] * 5 + ["StorageUnit"] * 2,
        "asset_id": ["ar7", "ar6", "solar", "gas", "beccs", "ldes", "battery"],
        "market_revenue_gbp": [2.0e8, 1.5e8, 4.0e7, 1.2e8, 1.5e7, 5.0e6, 8.0e6],
        "support_revenue_gbp": [1.5e8, 1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0],
        "fuel_carbon_gbp": [0.0, 0.0, 0.0, 1.0e8, 1.0e7, 0.0, 0.0],
    })
    costs = pd.concat([_costs(), pd.DataFrame([{
        "technology": "Biomass - with CCUS - mature", "capex_power_gbp_per_kw": 4423.57, "capex_energy_gbp_per_kwh": 0.0,
        "fixed_om_gbp_per_kw_year": 202.73, "variable_om_gbp_per_mwh": 5.06, "economic_life_years": 25.0,
        "price_year": 2024, "real_or_nominal": "real", "status": "reviewed"}, {
        "technology": "Novel long duration energy storage (LDES)", "capex_power_gbp_per_kw": 977.99,
        "capex_energy_gbp_per_kwh": 0.0, "duration_hours": 4, "fixed_om_gbp_per_kw_year": 32.61,
        "variable_om_gbp_per_mwh": 0.0, "economic_life_years": 20.0, "price_year": 2024, "real_or_nominal": "real",
        "status": "provisional"}])], ignore_index=True)
    snapshot = _snapshot()
    snapshot["technologies"].append({"technology": "Biomass - with CCUS - mature",
                                     "wacc": {"private": 0.095, "public_coordination": 0.09,
                                              "public_ownership_new_build": 0.0425, "public_ownership_legacy": 0.0275}})
    snapshot["technologies"].append({"technology": "Novel long duration energy storage (LDES)",
                                     "wacc": {"private": 0.108, "public_coordination": 0.10,
                                              "public_ownership_new_build": 0.0375, "public_ownership_legacy": 0.0275}})
    # Item OH1 off: this fixture pins the public account's routing rules.  Item
    # B2: the LDES floor the bill pays is booked to the store's rows, on the
    # repository's account settings, so register and account agree.
    block = load_defaults()["public_ownership"]
    register, summary = calculate_financing_scenarios(attach_settlement(assets, settlement), costs, snapshot,
                                                      price_year=2024, axes=axes or {},
                                                      state_operating_overhead=NO_OVERHEAD,
                                                      ldes_account_settings=account_settings(block))
    return register, summary


def test_settlement_attaches_by_cohort_share_and_sums_back_to_the_asset():
    assets = pd.DataFrame([
        {"component_type": "Generator", "asset_id": "farm::legacy", "cohort_legacy_share": 0.25},
        {"component_type": "Generator", "asset_id": "farm::new_build", "cohort_legacy_share": 0.25},
        {"component_type": "Generator", "asset_id": "dated", "cohort_legacy_share": 0.0},
        {"component_type": "StorageUnit", "asset_id": "orphan::new_build", "cohort_legacy_share": 0.0},
    ])
    settlement = pd.DataFrame([
        {"component_type": "Generator", "asset_id": "farm", "market_revenue_gbp": 100.0, "support_revenue_gbp": 40.0, "fuel_carbon_gbp": 8.0},
        {"component_type": "Generator", "asset_id": "dated", "market_revenue_gbp": 10.0, "support_revenue_gbp": 0.0, "fuel_carbon_gbp": 1.0},
    ])
    out = attach_settlement(assets, settlement).set_index("asset_id")
    assert out.loc["farm::legacy", "market_revenue_gbp"] == pytest.approx(25.0)
    assert out.loc["farm::new_build", "market_revenue_gbp"] == pytest.approx(75.0)
    assert out.loc[["farm::legacy", "farm::new_build"], "support_revenue_gbp"].sum() == pytest.approx(40.0)
    assert out.loc["dated", "fuel_carbon_gbp"] == pytest.approx(1.0)
    assert out.loc["orphan::new_build", "settlement_status"] == "no_settlement_record"
    assert out.loc["orphan::new_build", "market_revenue_gbp"] == 0.0


def test_register_surplus_deducts_fuel_and_carbon_and_uses_the_ownership_share():
    register, summary = _settled_register()
    own = register.loc[register.case_id.eq("public_ownership")].set_index("asset_id")
    gas = own.loc["gas"]
    assert gas.producer_rent_gbp == pytest.approx(1.2e8 - 1.0e8 - gas.annual_required_revenue_gbp)
    assert gas.public_surplus_before_recycling_gbp == pytest.approx(gas.producer_rent_gbp)
    coord = register.loc[register.case_id.eq("public_coordination")].set_index("asset_id").loc["gas"]
    assert coord.public_surplus_before_recycling_gbp == pytest.approx(0.3 * coord.producer_rent_gbp)
    assert summary.set_index("case_id").loc["public_ownership", "public_surplus_status"] == "asset_settlement_linked"


def test_public_account_excludes_r1_and_routes_by_rule(defaults):
    register, summary = _settled_register()
    repricing = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    savings = per_row_savings(register, repricing)
    settings = account_settings(defaults["public_ownership"])
    # Item K1: the recycling rule is a lever on the register.  The synthetic register here
    # carries the item-D1 bundles (public_cfd_at_cost); since 30 Sep 2026 the repository's
    # P2 base rebates gross (bill-payers first) and public_cfd_at_cost is its todays_rules.
    assert defaults["public_ownership"]["scenarios"]["public_ownership"]["surplus_recycling"] == "rebate_gross"
    account, report = build_public_account(register, savings, settings)
    assert report["public_ownership"]["recycling_rule"] == "public_cfd_at_cost"
    own = account.loc[account.case_id.eq("public_ownership")].set_index("group")
    assert set(own.index) >= {GROUP_R1, GROUP_R2, GROUP_MERCHANT, GROUP_CM, GROUP_DPA, GROUP_LDES}
    assert not own.loc[GROUP_R1, "in_account"]
    rows = register.loc[register.case_id.eq("public_ownership")].set_index("asset_id")
    private_rows = register.loc[register.case_id.eq("private")].set_index("asset_id")
    # Gas (Capacity Market route) and BECCS (dispatchable CfD route) are outside the contracted portfolio.
    expected_merchant = rows.loc["solar", "public_surplus_before_recycling_gbp"]
    case = report["public_ownership"]
    assert case["merchant_generation_surplus_gbp"] == pytest.approx(expected_merchant)
    assert case["capacity_market_route_surplus_not_recycled_gbp"] == pytest.approx(rows.loc["gas", "public_surplus_before_recycling_gbp"])
    assert case["dispatchable_cfd_route_surplus_not_recycled_gbp"] == pytest.approx(rows.loc["beccs", "public_surplus_before_recycling_gbp"])
    assert case["r2_cfd_surplus_gbp"] == pytest.approx(rows.loc["ar6", "public_surplus_before_recycling_gbp"])
    # public_cfd_at_cost routes the renewable surplus as a policy line; R2 stays in the account.
    assert case["policy_route_gbp"] == pytest.approx(expected_merchant) and case["wholesale_route_gbp"] == 0.0
    # Item B12: the state keeps what is left of its re-priced (R1) row's rent after
    # the strike cut, and it is retained with the R2 surplus, not lost.
    ar7 = savings.set_index(["case_id", "asset_id"]).loc[("public_ownership", "ar7")]
    ar7_rent = rows.loc["ar7", "producer_rent_gbp"]
    expected_r1 = rows.loc["ar7", "ownership_share"] * (ar7_rent - ar7["saving_bill_term_gbp"])
    assert case["r1_retained_rent_gbp"] == pytest.approx(expected_r1)
    assert case["r1_bill_cut_gbp"] == pytest.approx(ar7["saving_bill_term_gbp"])
    assert case["retained_in_public_account_gbp"] == pytest.approx(case["r2_cfd_surplus_gbp"] + expected_r1)
    # A 30% stake on the financing-saving basis: the cut is the whole JV's cost
    # fall, so it splits pro rata and the co-owner keeps the private rent.
    coord = register.loc[register.case_id.eq("public_coordination")].set_index("asset_id").loc["ar7"]
    coord_cut = savings.set_index(["case_id", "asset_id"]).loc[("public_coordination", "ar7"), "saving_bill_term_gbp"]
    assert not bool(savings.set_index(["case_id", "asset_id"]).loc[("public_coordination", "ar7"), "at_cost_applied"])
    assert report["public_coordination"]["r1_cut_borne_by_state_gbp"] == pytest.approx(0.3 * coord_cut)
    assert report["public_coordination"]["r1_retained_rent_gbp"] == pytest.approx(
        0.3 * (coord.producer_rent_gbp - coord_cut))
    # Sensitivities: the state share of the financing saving on the plant the Capacity Market and CCS levy pay.
    def saving(asset):
        return rows.loc[asset, "ownership_share"] * (private_rows.loc[asset, "annual_required_revenue_gbp"] - rows.loc[asset, "annual_required_revenue_gbp"])
    assert case["capacity_market_sensitivity_gbp"] == pytest.approx(saving("gas") + saving("battery"))
    assert case["ccs_levy_sensitivity_gbp"] == pytest.approx(saving("beccs"))
    # LDES floor: re-annuitised at the debt-like rate (F29: workbook v2.0 BBB- cost of debt, 5.674% private),
    # below the floor at the technology WACC.
    private_ldes = private_rows.loc["ldes"]
    at_wacc = max(private_ldes.annual_required_revenue_gbp - 5.0e6, 0.0)
    assert report["private"]["ldes_floor_topup_at_wacc_gbp"] == pytest.approx(at_wacc)
    assert 0 < report["private"]["ldes_floor_topup_gbp"] < at_wacc
    capex = private_ldes.annual_capital_recovery_gbp / capital_recovery_factor(private_ldes.wacc, private_ldes.remaining_life_years)
    expected_floor = (capex * capital_recovery_factor(0.05674, private_ldes.remaining_life_years)
                      + private_ldes.annual_fixed_om_gbp + private_ldes.annual_variable_om_gbp)
    assert report["private"]["ldes_floor_topup_gbp"] == pytest.approx(max(expected_floor - 5.0e6, 0.0))
    # Under ownership the floor is the state rate (null -> case WACC), so it equals the WACC floor.
    assert report["public_ownership"]["ldes_floor_topup_gbp"] == pytest.approx(report["public_ownership"]["ldes_floor_topup_at_wacc_gbp"])
    assert report["public_ownership"]["ldes_floor_topup_gbp"] < report["private"]["ldes_floor_topup_gbp"]
    # Storage is never recycled.  Item B2: the LDES floor the bill pays reaches
    # the store's rent (register and account agree on it).
    assert rows.loc["ldes", "ldes_floor_topup_paid_gbp"] == pytest.approx(report["public_ownership"]["ldes_floor_topup_gbp"])
    assert report["public_ownership"]["ldes_floor_topup_paid_to_owners_gbp"] == pytest.approx(
        report["public_ownership"]["ldes_floor_topup_gbp"])
    assert case["storage_surplus_not_recycled_gbp"] == pytest.approx(
        rows.loc["ldes", "public_surplus_before_recycling_gbp"] + rows.loc["battery", "public_surplus_before_recycling_gbp"])
    # Saving part plus rent part is the surplus.
    total = own.loc[own.in_account]
    assert (total.public_surplus_saving_part_gbp + total.public_surplus_rent_part_gbp).sum() == pytest.approx(total.public_surplus_gbp.sum())


@pytest.mark.parametrize("rule, wholesale, policy", [
    ("no_assumption", 0.0, 0.0), ("retained_by_treasury", 0.0, 0.0),
    ("rebate_per_mwh_eligible_supply", 0.0, "merchant+r2"), ("offset_policy_levies", 0.0, "merchant+r2"),
    ("cost_recovery_contracting", "merchant", 0.0), ("public_cfd_at_cost", 0.0, "merchant"),
])
def test_each_recycling_rule_routes_the_declared_amount(defaults, rule, wholesale, policy):
    register, _ = _settled_register(_central(surplus_recycling=rule))
    savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    settings = account_settings(defaults["public_ownership"])
    _, report = build_public_account(register, savings, settings)
    assert report["public_ownership"]["recycling_rule"] == rule
    case = report["public_ownership"]
    merchant, r2 = case["merchant_generation_surplus_gbp"], case["r2_cfd_surplus_gbp"]
    expected_wholesale = merchant if wholesale == "merchant" else 0.0
    # Item B12: the R1 remainder follows R2, so the rebate rules route it with R2.
    r1_kept = case["r1_retained_rent_gbp"]
    expected_policy = merchant + r2 + r1_kept if policy == "merchant+r2" else (merchant if policy == "merchant" else 0.0)
    assert case["wholesale_route_gbp"] == pytest.approx(expected_wholesale)
    assert case["policy_route_gbp"] == pytest.approx(expected_policy)
    routed = expected_wholesale + expected_policy
    assert case["shortfall_gbp"] == pytest.approx(max(-routed, 0.0))


def test_shortfall_funding_switches_between_bill_and_exchequer(defaults):
    for funding, on_bill in (("levy_on_eligible_supply", True), ("exchequer", False)):
        register, _ = _settled_register(_central(shortfall_funding=funding))
        savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
        # Force a shortfall: strip the state's merchant revenue.
        register = register.copy()
        register.loc[register.asset_id.isin(["solar", "gas"]), "market_revenue_gbp"] = 0.0
        settings = account_settings(defaults["public_ownership"])
        _, report = build_public_account(register, savings, settings)
        case = report["public_ownership"]
        assert case["shortfall_gbp"] > 0
        assert (case["shortfall_on_bill_gbp"] > 0) is on_bill
        assert (case["shortfall_on_exchequer_gbp"] > 0) is not on_bill
        # public_cfd_at_cost: the top-up sits on the CfD-at-cost line itself (positive), or off the bill.
        lines = {r["component_id"]: r for r in extra_lines("public_ownership", case, factor=1.0, final_demand_mwh=100.0,
                                                           eligible_mwh=50.0, mwh=3.1, account_source="test")}
        assert (LINE_CFD_AT_COST in lines) is on_bill and LINE_SHORTFALL not in lines
        if on_bill:
            assert lines[LINE_CFD_AT_COST]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(case["shortfall_on_bill_gbp"] / 50.0 * 3.1)
        # cost_recovery_contracting: a separate shortfall levy line.
        register, _ = _settled_register(_central(shortfall_funding=funding, surplus_recycling="cost_recovery_contracting"))
        register = register.copy()
        register.loc[register.asset_id.isin(["solar", "gas"]), "market_revenue_gbp"] = 0.0
        savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
        _, report = build_public_account(register, savings, settings)
        ids = {r["component_id"] for r in extra_lines("public_ownership", report["public_ownership"], factor=1.0,
                                                       final_demand_mwh=100.0, eligible_mwh=50.0, mwh=3.1, account_source="test")}
        assert (LINE_SHORTFALL in ids) is on_bill and LINE_W1 not in ids


def test_extra_lines_convert_account_amounts_with_the_right_denominators():
    account = {"ldes_cap_and_floor_enabled": True, "ldes_floor_topup_gbp": 1000.0, "recycling_rule": "cost_recovery_contracting",
               "routed_to_bill_gbp": 500.0, "shortfall_on_bill_gbp": 0.0}
    rows = {r["component_id"]: r for r in extra_lines("public_ownership", account, factor=1.1, final_demand_mwh=100.0,
                                                       eligible_mwh=50.0, mwh=3.1, account_source="test")}
    # Item D4(c): the LDES line is the forecast bill's own, never an extra component.
    assert LINE_LDES not in rows
    assert rows[LINE_W1]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-500.0 * 1.1 / 100.0 * 3.1)
    assert rows[LINE_W1]["stack_group"] == "hedged_wholesale"
    private = extra_lines("private", account, factor=1.1, final_demand_mwh=100.0, eligible_mwh=50.0, mwh=3.1, account_source="test")
    assert [r["component_id"] for r in private] == []
    gross_only = extra_lines("public_ownership", account, factor=1.1, final_demand_mwh=100.0, eligible_mwh=50.0, mwh=3.1,
                             account_source="test", gross_only=True)
    assert [r["component_id"] for r in gross_only] == []
    at_cost = dict(account, recycling_rule="public_cfd_at_cost")
    rows = {r["component_id"]: r for r in extra_lines("public_ownership", at_cost, factor=1.1, final_demand_mwh=100.0,
                                                       eligible_mwh=50.0, mwh=3.1, account_source="test")}
    assert rows[LINE_CFD_AT_COST]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-500.0 * 1.1 / 50.0 * 3.1)
    assert rows[LINE_CFD_AT_COST]["stack_group"] == "policy" and LINE_W1 not in rows


def test_ownership_boundaries_are_labelled_variants_with_their_own_account(defaults):
    """Two readings of the boundary: pre-2025-award CfD capacity stays private, or is owned and re-priced."""
    from scripts.public_ownership.calculate_financing_scenarios import BOUNDARY_ALL_CFD_REPRICED
    register, summary = _settled_register(_axes(ownership_boundary=[BOUNDARY_ALL_CFD_REPRICED]))
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    assert settings["sensitivities"] == []
    savings = savings_with_sensitivities(register, settings)
    variants = sorted(c for c in savings.case_id.unique() if is_variant(c))
    assert variants == ["public_coordination@ownership_all_cfd_repriced", "public_ownership@ownership_all_cfd_repriced"]
    assert base_case(variants[1]) == "public_ownership"
    central = savings.loc[savings.case_id.eq("public_ownership")].set_index("asset_id")
    variant = savings.loc[savings.case_id.eq(variants[1])].set_index("asset_id")
    # Central boundary: the AR6 farm keeps its private owner and its strike, so no saving and no re-pricing.
    assert not central.loc["ar6", "state_owned"] and central.loc["ar6", "saving_economic_life_gbp"] == 0.0
    assert central.loc["ar6", "route"] == ROUTE_CFD_PUBLIC_ACCOUNT and central.loc["ar7", "route"] == ROUTE_CFD_BILL
    # Variant boundary: the state owns it and re-prices it.
    assert variant.loc["ar6", "state_owned"] and variant.loc["ar6", "saving_economic_life_gbp"] > 0
    assert variant.loc["ar6", "route"] == ROUTE_CFD_BILL and variant.loc["solar", "route"] == ROUTE_MERCHANT
    assert variant.loc["ar6", "method"].startswith("ownership_all_cfd_repriced;commissioning_vintage")
    routes = route_summary(savings, summary)
    assert routes.identity_within_tolerance.all()
    assert set(routes.case_id) == set(savings.case_id)
    # Neither boundary leaves a re-pricing-free saving in the public account.
    for case_id in ("public_ownership", variants[1]):
        r2 = routes.loc[routes.case_id.eq(case_id) & routes.route.eq(ROUTE_CFD_PUBLIC_ACCOUNT), "saving_economic_life_gbp_2024"].iloc[0]
        assert r2 == pytest.approx(0.0)
    # The variant has its own public account entry, keyed by the base case's rule and floor rate.
    account_settings_ = account_settings(defaults["public_ownership"])
    _, report = build_public_account(register, savings, account_settings_)
    assert report[variants[1]]["recycling_rule"] == report["public_ownership"]["recycling_rule"]
    assert report[variants[1]]["ownership_boundary"] == BOUNDARY_ALL_CFD_REPRICED == "public_repriced"
    assert report["public_ownership"]["ownership_boundary"] == "private"


def test_release_2_decisions_are_validated(defaults):
    config = deepcopy(defaults["public_ownership"])
    forecasts = defaults["consumer_bill"]["forecasts"]["cases"]
    assert validate_public_ownership_config(config, forecasts) is config
    # 30 Sep 2026 (user decision 1): every base is bill-payers first.
    assert config["scenarios"]["public_ownership"]["surplus_recycling"] == "rebate_gross"
    for mutate, message in [
        (lambda c: c.__setitem__("public_shortfall_funding", "magic"), "is retired"),
        (lambda c: c.__setitem__("surplus_recycling", {"public_ownership": "public_cfd_at_cost"}), "is retired"),
        (lambda c: c.__setitem__("public_ownership_asset_boundary", "eligible_post_2025_build_only"), "is retired"),
        (lambda c: c.__setitem__("boundary_sensitivities", ["ownership_excluding_pre_2025_cfd"]), "is retired"),
        (lambda c: c.__setitem__("sensitivity_axes", {"axes": {}}), "is retired"),
        (lambda c: c["scenarios"]["public_ownership"].__setitem__("shortfall_funding", "magic"), "not implemented"),
        (lambda c: c["scenarios"]["public_control"].__setitem__("surplus_recycling", "everything"), "not implemented"),
        (lambda c: c["sensitivities"]["structural"].append("strike_price"), "unknown lever"),
        (lambda c: c["sensitivities"]["named"]["fully_nationalised"].__setitem__("public_return", "free"), "not implemented"),
        (lambda c: c["consumer_bill"]["cfd_repricing"].__setitem__("sensitivities", ["contract_vintage"]), "is retired"),
        (lambda c: c["consumer_bill"]["cfd_repricing"].__setitem__("eligibility", "commissioning_vintage"), "is retired"),
        (lambda c: c["consumer_bill"]["ldes_cap_and_floor"].__setitem__("carriers", []), "carriers"),
        (lambda c: c["consumer_bill"]["ldes_cap_and_floor"].__setitem__("recovery_denominator", "eligible"), "recovery_denominator"),
        (lambda c: c["consumer_bill"]["capacity_market_sensitivity"].__setitem__("enabled", "yes"), "capacity_market_sensitivity"),
        (lambda c: c["consumer_bill"]["ldes_cap_and_floor"]["floor_return"].__setitem__("basis", "equity"), "floor_return.basis"),
        (lambda c: c["consumer_bill"]["ldes_cap_and_floor"]["floor_return"]["real_rate_by_case"].__setitem__("private", 1.5), "real_rate_by_case"),
        (lambda c: c["consumer_bill"]["label_corrections"]["route_classification"].__setitem__("CCGT", "merchant"), "route_classification"),
        (lambda c: (c.__setitem__("consumer_benefit", {"treatment": "routed_via_consumer_bill_link"}), c.pop("consumer_bill")), "needs the consumer_bill block"),
    ]:
        broken = deepcopy(config)
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            validate_public_ownership_config(broken, forecasts)


def test_ws9_items_4_to_9_are_primary_sourced():
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    sourced = assumptions["sourced_externals"]
    for key in ("capacity_market", "sizewell_c_rab", "bics_levy", "nuclear_contract_counterparty",
                "smart_meter_net_cost_change", "aahedc"):
        block = sourced[key]
        assert key not in assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"]
        assert block["source_class"] == "external_primary_source" and block["unit"] == "gbp_per_household_year"
        assert block["source_url"].startswith("https://") and "electricitybills" not in block["source_url"]
        assert block["access_date"] and block["reference_gbp_per_3100kwh_ex_vat"] > 0
        if block.get("source_file"):
            assert Path(block["source_file"]).exists(), block["source_file"]
        if int(block["price_year"]) != 2025:
            rebasing = block["rebasing"]
            assert rebasing["to_price_year"] == 2025
            assert rebasing["factor"] == pytest.approx(rebasing["to_index"] / rebasing["from_index"], rel=1e-6)
    cm = sourced["capacity_market"]["derivation"]
    # Item MR4: the contract book and modelled auctions; the T-4-only figure is kept as the previous basis.
    previous = sourced["capacity_market"]["previous_basis"]
    assert previous["clearing_price_gbp_per_kw_year"] == 27.10 and previous["capacity_agreements_mw"] == pytest.approx(40108.608)
    assert cm["agreement_cost_gbp"] > previous["agreement_cost_gbp"]
    # The value is derived over the recovery file's GSP volume at projection time
    # (items F4, D1); tests/unit/test_sourced_external_denominators.py checks it.
    assert cm["recovery_denominator"] == "gsp_charging_volume_mwh" and "value" not in sourced["capacity_market"]
    # Item F1 sourced the last five placeholders (22 September 2026).
    assert assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"] == {}
    # Item G6: Ofgem lowered its medium TDCV to 2,500 kWh from 1 July 2026.
    assert assumptions["benchmark"]["alternative_consumption_kwh"] == [2500]
    assert "environmental_levies_gbp_bn" in assumptions["source"]["obr_cross_check"]


def test_alternative_consumption_bill_uses_the_declared_charge_basis():
    """Item G6 replaced 'primary-source lines fixed, the rest scaled' with a declared basis."""
    from tests.unit.test_future_consumer_bill import fixture_inputs
    from scripts.consumer_bill.project_future_bill import project_components

    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["benchmark"]["alternative_consumption_kwh"] = [1550]
    assumptions["sourced_externals"]["allocated_ccs"] = {
        "value": 20.0, "unit": "gbp_per_household_year", "consumption_basis_kwh": 3100, "price_year": 2025,
        "stack_group": "policy", "label": "CCS (test)", "source_class": "external_primary_source",
        "reference_gbp_per_3100kwh_ex_vat": 4.0, "method": "test", "source_url": "https://example.org",
        # The repository file declares a calendar bill year (28 Sep 2026).
        "period_basis": "held_flat_real", "source_period": "test"}
    assumptions["household_archetypes"] = {"benchmark_kwh": 3100, "archetypes": [{"id": "low", "kwh": 1550}]}
    assumptions["charge_basis"] = {"standing_share": {"allocated_ccs": 1.0}}
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    alt = summary["diagnostics"]["alternative_consumption_bills"]["1550"]
    total = float(rows["annual_gbp_per_3100kwh_ex_vat"].sum())
    assert alt["ex_vat_gbp"] == pytest.approx((total - 20.0) * 0.5 + 20.0)
    assert alt["fixed_per_household_gbp_ex_vat"] == pytest.approx(20.0)
    bills = {r["archetype"]: r for r in summary["diagnostics"]["household_archetypes"]["bills"]}
    assert bills["benchmark"]["ex_vat_gbp"] == pytest.approx(total)
    assert bills["low"]["ex_vat_gbp"] == pytest.approx(alt["ex_vat_gbp"])


def test_the_pay_as_bid_ceiling_counts_only_non_cfd_wholesale_margin(defaults):
    """Item K3: CfD plant is made whole by its top-up, so only non-CfD generators carry removable margin."""
    register, _ = _settled_register(_central(wholesale_pricing="pay_as_bid_ceiling"))
    register = register.copy()
    register["wholesale_revenue_gbp"] = register["market_revenue_gbp"]
    savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    _, report = build_public_account(register, savings, account_settings(defaults["public_ownership"]))
    private = register.loc[register.case_id.eq("private") & register.component_type.eq("Generator")].set_index("asset_id")
    non_cfd = private.loc[~private.support_type.astype(str).str.lower().eq("cfd")]
    expected = (non_cfd.wholesale_revenue_gbp - non_cfd.fuel_carbon_gbp - non_cfd.annual_variable_om_gbp).clip(lower=0.0).sum()
    assert report["private"]["pay_as_bid_ceiling_gbp"] == pytest.approx(expected)
    assert report["private"]["pay_as_bid_ceiling_applies"] is True
    assert expected < (private.wholesale_revenue_gbp - private.fuel_carbon_gbp - private.annual_variable_om_gbp).clip(lower=0.0).sum()
