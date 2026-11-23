"""WP-A, WP-B and WP-G of the public-ownership bill link (plan sections 22 and 27)."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.market.cfd_strike_table import StrikeTable, load_strike_table
from scripts.public_ownership.build_asset_finance_register import (
    build_asset_finance_register, contract_vintage, load_award_years,
)
from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.cfd_repricing import (
    ROUTE_CAPACITY_MARKET, ROUTE_CFD_BILL, ROUTE_CFD_PUBLIC_ACCOUNT, ROUTE_MERCHANT, ROUTE_STORAGE,
    ROUTES, assert_single_route, build_strike_adjustments, case_strike_table, per_row_savings,
    _term_label, repricing_settings, route_summary,
)
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.consumer_bill.project_future_bill import SUPPLIER_LINE
from scripts.public_ownership.project_public_ownership_consumer_bill import R1_LINES
from scripts.public_ownership.reference_runs import (
    consumer_bill_groups, price_level_twin_groups, validate_public_ownership_config,
)
from config.config_loader import load_defaults


#: Item OH1: the state's holding-company cost, switched off in fixtures that
#: pin other mechanics.  See tests/unit/test_public_ownership_state_overhead.py.
NO_OVERHEAD = {"enabled": False}


AWARD_YEARS = Path("data/public_ownership/financing/cfd_round_award_years.csv")
STRIKE_TABLE = Path("data/market/lccc/cfd_strike_prices_by_round.csv")
ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
PUBLIC = Path("resources/public_ownership")
VALUE = "annual_gbp_per_3100kwh_ex_vat"


@pytest.fixture(scope="module")
def defaults() -> dict:
    return load_defaults()


@pytest.fixture(scope="module")
def settings(defaults) -> dict:
    return repricing_settings(defaults["public_ownership"]["consumer_bill"])


@pytest.fixture(scope="module")
def bill_table() -> StrikeTable:
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    return load_strike_table(STRIKE_TABLE, assumptions["cfd_strike_indexation"], 2024)


# ----------------------------------------------------------------------------- WP-A


def test_award_year_table_covers_every_round_in_the_strike_table(bill_table):
    years = load_award_years(AWARD_YEARS)
    assert set(bill_table.rounds) <= set(years), sorted(set(bill_table.rounds) - set(years))
    assert years["AR6"] == 2024 and years["AR7"] == 2026 and years["AR8_modelled"] >= 2027


def test_contract_vintage_follows_the_award_year_not_the_commissioning_year():
    years = load_award_years(AWARD_YEARS)
    assert contract_vintage("CfD", "AR6", years, 2025) == ("pre_policy", 2024.0)
    assert contract_vintage("CfD", "AR7", years, 2025) == ("post_policy", 2026.0)
    assert contract_vintage("CfD", "AR8_modelled", years, 2025) == ("post_policy", 2027.0)
    assert contract_vintage("CfD", "", years, 2025) == ("unknown_round", None)
    assert contract_vintage("CfD", "AR99", years, 2025) == ("unknown_round", None)
    assert contract_vintage("merchant", "AR7", years, 2025) == ("not_applicable", None)
    # Item MR5: a bridging contract on legacy plant, awarded after the policy
    # start, finances nothing new and is never classed post-policy.
    from scripts.public_ownership.build_asset_finance_register import load_legacy_plant_rounds
    legacy = load_legacy_plant_rounds(AWARD_YEARS)
    assert "DRAX_BRIDGE" in legacy
    assert contract_vintage("CfD", "DRAX_BRIDGE", years, 2025, legacy) == ("legacy_plant_contract", 2025.0)
    assert contract_vintage("CfD", "POWER_BECCS", years, 2025, legacy) == ("post_policy", 2027.0)


def _network(path, rows, storage_rows=(), hours=3):
    import pypsa

    network = pypsa.Network()
    snapshots = pd.date_range("2030-01-01", periods=hours, freq="h")
    network.set_snapshots(snapshots)
    network.add("Bus", "bus")
    for name, attrs in rows.items():
        dispatch = attrs.pop("dispatch")
        network.add("Generator", name, bus="bus", **attrs)
        network.generators_t.p[name] = dispatch
    for name, attrs in storage_rows:
        network.add("StorageUnit", name, bus="bus", **attrs)
        network.storage_units_t.p[name] = 1.0
    for column in ("operational_date", "development_status", "data_source", "support_type", "cfd_round"):
        if column not in network.generators.columns:
            network.generators[column] = ""
    network.export_to_netcdf(path)
    return path


def _crosswalk():
    return pd.DataFrame([
        {"component_type": "Generator", "pypsa_carrier": "wind_offshore", "workbook_technology": "Offshore wind", "finance_eligible": "true"},
        {"component_type": "Generator", "pypsa_carrier": "marine", "workbook_technology": "Tidal stream", "finance_eligible": "true"},
        {"component_type": "Generator", "pypsa_carrier": "CCGT", "workbook_technology": "Gas generation - unabated", "finance_eligible": "true"},
        {"component_type": "StorageUnit", "pypsa_carrier": "Battery", "workbook_technology": "Lithium batteries", "finance_eligible": "true"},
    ])


def test_register_uses_asset_dates_and_splits_only_the_undated_residual(tmp_path):
    pytest.importorskip("pypsa")
    rows = {
        # Dated legacy farm: stays legacy whatever the baseline says.
        "old_farm": dict(carrier="wind_offshore", p_nom=100.0, operational_date="31/03/2015", support_type="CfD",
                         cfd_round="AR1", data_source="REPD_named_fleet", dispatch=50.0),
        # Dated post-policy farm: stays new build.
        "new_farm": dict(carrier="wind_offshore", p_nom=100.0, operational_date="01/06/2028", support_type="CfD",
                         cfd_round="AR7", data_source="REPD_named_fleet", dispatch=40.0),
        # Pipeline project without a date: new build by development status.
        "pipeline": dict(carrier="wind_offshore", p_nom=100.0, development_status="Under Construction",
                         support_type="CfD", cfd_round="AR6", data_source="REPD_named_fleet", dispatch=30.0),
        # Undated proxy capacity: carries the residual of the 2024 fleet.
        "CP30_wind_offshore_1": dict(carrier="wind_offshore", p_nom=200.0, support_type="CfD", cfd_round="AR8_modelled",
                                     data_source="CP30", dispatch=80.0),
        "tidal": dict(carrier="marine", p_nom=10.0, support_type="merchant", data_source="CP30", dispatch=2.0),
        "gas": dict(carrier="CCGT", p_nom=400.0, data_source="CP30", dispatch=100.0),
    }
    storage = [("battery_1", dict(carrier="Battery", p_nom=50.0, max_hours=2.0))]
    network = _network(tmp_path / "model.nc", rows, storage)
    baseline = _network(tmp_path / "baseline.nc", {
        "old_farm": dict(carrier="wind_offshore", p_nom=100.0, dispatch=1.0),
        "other_old_farm": dict(carrier="wind_offshore", p_nom=150.0, dispatch=1.0),
        "gas": dict(carrier="CCGT", p_nom=300.0, dispatch=1.0),
    }, [("battery_0", dict(carrier="Battery", p_nom=10.0, max_hours=2.0))])
    register = build_asset_finance_register(
        [network], _crosswalk(), model_year=2030, policy_start_year=2025, baseline_network_paths=[baseline],
        support_type_overrides={"marine": "CfD"}, award_years=load_award_years(AWARD_YEARS),
        eligible_award_years_from=2025,
    ).set_index("asset_id")

    assert register.loc["old_farm", ["finance_vintage", "vintage_method"]].tolist() == ["legacy", "asset_date"]
    assert register.loc["new_farm", ["finance_vintage", "vintage_method"]].tolist() == ["new_build", "asset_date"]
    assert register.loc["pipeline", ["finance_vintage", "vintage_method"]].tolist() == ["new_build", "development_status"]
    # Baseline 250 MW offshore, 100 MW of it dated legacy: 150 MW of the 200 MW proxy is legacy.
    proxy = register.loc[["CP30_wind_offshore_1::legacy", "CP30_wind_offshore_1::new_build"]]
    assert proxy["capacity_mw"].tolist() == pytest.approx([150.0, 50.0])
    assert proxy["dispatch_mwh"].sum() == pytest.approx(80.0 * 3)
    assert set(proxy["vintage_method"]) == {"2024_to_model_capacity_delta_by_carrier"}
    # Gas: 300 of 400 MW legacy by the carrier delta, no dated assets.
    assert register.loc["gas::legacy", "capacity_mw"] == pytest.approx(300.0)
    assert register.loc["gas::new_build", "capacity_mw"] == pytest.approx(100.0)
    # Label correction keeps the network label beside the corrected one.
    tidal = register.loc[["tidal::new_build"]].iloc[0]
    assert tidal["support_type"] == "CfD" and tidal["network_support_type"] == "merchant"
    assert tidal["contract_vintage"] == "unknown_round"
    # Contract vintages from the award-year table.
    assert register.loc["old_farm", "contract_vintage"] == "pre_policy"
    assert register.loc["pipeline", "contract_vintage"] == "pre_policy"
    assert register.loc["new_farm", ["contract_vintage", "cfd_award_year"]].tolist() == ["post_policy", 2026.0]
    assert register.loc["CP30_wind_offshore_1::new_build", "contract_vintage"] == "post_policy"
    assert register.loc["gas::legacy", "contract_vintage"] == "not_applicable"
    assert register.loc["battery_1::new_build", "capacity_mw"] == pytest.approx(40.0)


# ----------------------------------------------------------------------------- WP-B


def _snapshot():
    return {
        "rate_basis": "real_CPI",
        "assumptions": {"public_coordination_state_equity_share": {"value": 0.30}},
        "technologies": [
            {"technology": "Offshore wind", "wacc": {"private": 0.083, "public_coordination": 0.075,
                                                    "public_ownership_new_build": 0.035, "public_ownership_legacy": 0.0275}},
            {"technology": "Solar PV", "wacc": {"private": 0.0666, "public_coordination": 0.0576,
                                               "public_ownership_new_build": 0.0325, "public_ownership_legacy": 0.0275}},
            {"technology": "Gas generation - unabated", "wacc": {"private": 0.09, "public_coordination": 0.085,
                                                                "public_ownership_new_build": 0.0375, "public_ownership_legacy": 0.0275}},
            {"technology": "Lithium batteries", "wacc": {"private": 0.10, "public_coordination": 0.09,
                                                        "public_ownership_new_build": 0.0375, "public_ownership_legacy": 0.0275}},
        ],
    }


def _costs():
    return pd.DataFrame([
        {"technology": "Offshore wind", "capex_power_gbp_per_kw": 2670.0, "capex_energy_gbp_per_kwh": 0.0,
         "fixed_om_gbp_per_kw_year": 46.0, "variable_om_gbp_per_mwh": 0.0, "economic_life_years": 30.0,
         "price_year": 2024, "real_or_nominal": "real", "status": "reviewed"},
        {"technology": "Solar PV", "capex_power_gbp_per_kw": 530.0, "capex_energy_gbp_per_kwh": 0.0,
         "fixed_om_gbp_per_kw_year": 5.9, "variable_om_gbp_per_mwh": 0.0, "economic_life_years": 38.0,
         "price_year": 2024, "real_or_nominal": "real", "status": "reviewed"},
        {"technology": "Gas generation - unabated", "capex_power_gbp_per_kw": 1020.0, "capex_energy_gbp_per_kwh": 0.0,
         "fixed_om_gbp_per_kw_year": 16.0, "variable_om_gbp_per_mwh": 5.0, "economic_life_years": 25.0,
         "price_year": 2024, "real_or_nominal": "real", "status": "reviewed"},
        {"technology": "Lithium batteries", "capex_power_gbp_per_kw": 732.67, "capex_energy_gbp_per_kwh": 0.0,
         "duration_hours": 4, "fixed_om_gbp_per_kw_year": 11.75, "variable_om_gbp_per_mwh": 0.0,
         "economic_life_years": 15.0, "price_year": 2024, "real_or_nominal": "real", "status": "reviewed"},
    ])


def _asset(asset_id, carrier, technology, vintage, support, cfd_round, contract, award, mw, mwh, **extra):
    row = {
        "model_year": 2030, "component_type": extra.pop("component_type", "Generator"), "asset_id": asset_id,
        "carrier": carrier, "finance_technology": technology, "finance_vintage": vintage, "capacity_mw": mw,
        "dispatch_mwh": mwh, "evidence_status": "resolved", "support_type": support, "cfd_round": cfd_round,
        "contract_vintage": contract, "cfd_award_year": award, "energy_capacity_mwh": 0.0,
    }
    row.update(extra)
    return row


def _synthetic_register(axes=None):
    assets = pd.DataFrame([
        _asset("ar7_a", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6),
        _asset("ar7_b", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 500.0, 2.0e6),
        _asset("ar6_new", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR6", "pre_policy", 2024, 800.0, 3.0e6),
        _asset("ar3_old", "wind_offshore", "Offshore wind", "legacy", "CfD", "AR3", "pre_policy", 2019, 600.0, 2.5e6),
        _asset("solar_future::legacy", "solar_pv", "Solar PV", "legacy", "CfD", "AR8_modelled", "post_policy", 2027, 100.0, 1.0e5),
        _asset("solar_future::new_build", "solar_pv", "Solar PV", "new_build", "CfD", "AR8_modelled", "post_policy", 2027, 300.0, 3.0e5),
        _asset("solar_merchant", "solar_pv", "Solar PV", "new_build", "merchant", "", "not_applicable", np.nan, 200.0, 2.0e5),
        _asset("gas_new", "CCGT", "Gas generation - unabated", "new_build", "", "", "not_applicable", np.nan, 400.0, 1.0e6),
        _asset("battery", "Battery", "Lithium batteries", "new_build", "", "", "not_applicable", np.nan, 100.0, 5.0e4,
               component_type="StorageUnit", energy_capacity_mwh=200.0),
        _asset("unmapped", "mystery", "Missing", "new_build", "CfD", "AR7", "post_policy", 2026, 10.0, 1.0e3,
               evidence_status="unmapped_carrier"),
    ])
    # Item OH1 off: this fixture pins the re-pricing mechanics, so the state's
    # holding-company cost is disabled rather than threaded through every
    # expected number.  It has its own tests.
    return calculate_financing_scenarios(assets, _costs(), _snapshot(), price_year=2024, axes=axes or {},
                                         state_operating_overhead=NO_OVERHEAD)


def test_strike_adjustments_reconcile_exactly_and_zero_delta_reproduces_private(bill_table, defaults):
    register, summary = _synthetic_register()
    block = deepcopy(defaults["public_ownership"]["consumer_bill"])
    block["cfd_repricing"]["amortisation"] = "economic_life"
    settings = repricing_settings(block)
    savings = per_row_savings(register, settings)
    assert_single_route(savings)
    adjustments = build_strike_adjustments(savings, bill_table, settings)

    for case_id in ("public_coordination", "public_ownership"):
        rows = adjustments.loc[adjustments.case_id.eq(case_id)].set_index("strike_key")
        # AR7 offshore: both post-policy farms; delta x all AR7 dispatch == their saving exactly.
        r1 = savings.loc[savings.case_id.eq(case_id) & savings.route.eq(ROUTE_CFD_BILL)]
        ar7 = r1.loc[r1.cfd_round.eq("AR7"), "saving_economic_life_gbp"].sum()
        ar7_row = rows.loc["AR7_offshore"]
        assert ar7_row.delta_gbp_per_mwh_2024 * ar7_row.eligible_dispatch_mwh == pytest.approx(ar7, rel=1e-12)
        assert ar7_row.eligible_dispatch_mwh == pytest.approx(6.0e6)
        assert ar7_row.strike_case_gbp_per_mwh_2024 == pytest.approx(ar7_row.strike_private_gbp_per_mwh_2024 - ar7_row.delta_gbp_per_mwh_2024)
        assert ar7_row.strike_private_gbp_per_mwh_2024 == pytest.approx(bill_table.strikes[("AR7", "offshore")])
        # AR6 offshore was awarded in 2024: no re-pricing under contract vintage, even though the cohort is new build.
        assert rows.loc["AR6_offshore", "delta_gbp_per_mwh_2024"] == 0.0 and not rows.loc["AR6_offshore", "eligible"]
        # The AR8 solar entry spreads the new-build cohort's saving over legacy and new-build dispatch alike.
        future = rows.loc["AR8_modelled_solar"]
        assert future.eligible_dispatch_mwh == pytest.approx(4.0e5)
        assert future.delta_gbp_per_mwh_2024 * future.eligible_dispatch_mwh == pytest.approx(
            r1.loc[r1.cfd_round.eq("AR8_modelled"), "saving_economic_life_gbp"].sum(), rel=1e-12)
        # Sum over entries equals the R1 route total.
        assert rows.r1_saving_gbp_2024.sum() == pytest.approx(r1.saving_economic_life_gbp.sum(), rel=1e-12)
        assert rows.register_saving_gbp_2024.sum() == pytest.approx(rows.r1_saving_gbp_2024.sum(), rel=1e-12)
    private = adjustments.loc[adjustments.case_id.eq("private")]
    assert (private.delta_gbp_per_mwh_2024 == 0.0).all()
    private_table = case_strike_table(bill_table, adjustments, "private")
    assert private_table.strikes == bill_table.strikes

    ownership_table = case_strike_table(bill_table, adjustments, "public_ownership")
    strike, key, fallback = ownership_table.lookup("wind_offshore", "AR7")
    delta = adjustments.set_index(["case_id", "strike_key"]).loc[("public_ownership", "AR7_offshore"), "delta_gbp_per_mwh_2024"]
    assert (strike, key, fallback) == (pytest.approx(bill_table.strikes[("AR7", "offshore")] - delta), "AR7_offshore", False)
    assert ownership_table.lookup("wind_offshore", "AR6") == bill_table.lookup("wind_offshore", "AR6")
    assert ownership_table.adjustment_label == "public_ownership"


def test_contract_term_amortisation_uses_the_term_annuity_and_declares_the_difference(defaults, bill_table):
    register, summary = _synthetic_register()
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    # The term is read from config (15 until AR7's 20-year terms are adopted), never pinned here.
    term = settings["contract_term_years"]
    assert settings["amortisation"] == "contract_term" and "AR7" not in settings["contract_term_years_by_round"]
    savings = per_row_savings(register, settings)
    row = savings.loc[savings.case_id.eq("public_ownership") & savings.asset_id.eq("ar7_a")].iloc[0]
    capex = 1000.0 * 1000.0 * 2670.0
    expected = capex * (capital_recovery_factor(0.083, term) - capital_recovery_factor(0.035, term))
    assert row.saving_bill_term_gbp == pytest.approx(expected, rel=1e-12)
    assert row.saving_economic_life_gbp == pytest.approx(capex * (capital_recovery_factor(0.083, 30) - capital_recovery_factor(0.035, 30)), rel=1e-12)
    # A 15-year annuity is less rate-sensitive than a 30-year one (CRF tends to 1/n as n shrinks),
    # so re-pricing over the contract term takes less to the bill than the life-long resource saving.
    assert row.saving_bill_term_gbp < row.saving_economic_life_gbp
    routes = route_summary(savings, summary)
    r1 = routes.loc[routes.case_id.eq("public_ownership") & routes.route.eq(ROUTE_CFD_BILL)].iloc[0]
    assert r1.term_adjustment_gbp_2024 == pytest.approx(r1.bill_allocated_gbp_2024 - r1.saving_economic_life_gbp_2024)
    adjustments = build_strike_adjustments(savings, bill_table, settings)
    ar7 = adjustments.set_index(["case_id", "strike_key"]).loc[("public_ownership", "AR7_offshore")]
    assert ar7.r1_saving_gbp_2024 < ar7.register_saving_gbp_2024
    assert ar7.method == (f"pre_2025_cfd_private;contract_vintage;contract_term_{_term_label(term, settings['contract_term_years_by_round'])}"
                          ";strike_financing_saving_only")


def test_a_contract_the_network_settles_as_merchant_cannot_be_repriced(defaults):
    """Tidal is relabelled CfD in the register only; the bill pays it no CfD, so it stays in the public account."""
    assets = pd.DataFrame([
        _asset("tidal", "marine", "Tidal stream", "new_build", "CfD", "", "unknown_round", np.nan, 254.0, 7.5e5,
               network_support_type="merchant"),
        _asset("ar7", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6,
               network_support_type="CfD"),
    ])
    costs = pd.concat([_costs(), pd.DataFrame([{
        "technology": "Tidal stream", "capex_power_gbp_per_kw": 5598.98, "capex_energy_gbp_per_kwh": 0.0,
        "fixed_om_gbp_per_kw_year": 159.12, "variable_om_gbp_per_mwh": 0.0, "economic_life_years": 25.0,
        "price_year": 2024, "real_or_nominal": "real", "status": "reviewed"}])], ignore_index=True)
    snapshot = _snapshot()
    snapshot["technologies"].append({"technology": "Tidal stream", "wacc": {
        "private": 0.09, "public_coordination": 0.08, "public_ownership_new_build": 0.0375, "public_ownership_legacy": 0.0275}})
    # Review 29 Sep 2026: the eligibility lever is retired; contract vintage is the only rule.
    register, _ = calculate_financing_scenarios(
        assets, costs, snapshot, price_year=2024, state_operating_overhead=NO_OVERHEAD)
    own = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    own = own.loc[own.case_id.eq("public_ownership")].set_index("asset_id")
    assert own.loc["tidal", "route"] == ROUTE_CFD_PUBLIC_ACCOUNT
    assert own.loc["ar7", "route"] == ROUTE_CFD_BILL


def test_the_retired_eligibility_lever_is_refused_with_a_pointer():
    """User decision 10: cfd_repricing_eligibility is retired (its two cases reproduced P1 and P2)."""
    from scripts.public_ownership.levers import LeverConfigError
    with pytest.raises(LeverConfigError, match="retired"):
        _synthetic_register(axes={"cfd_repricing_eligibility": {"alternatives": ["commissioning_vintage"]}})


# ----------------------------------------------------------------------------- WP-G


def test_every_pound_is_routed_once_and_the_identity_holds_to_one_pound(defaults):
    register, summary = _synthetic_register()
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    savings = per_row_savings(register, settings)
    assert_single_route(savings)
    assert set(savings.route) <= set(ROUTES)
    own = savings.loc[savings.case_id.eq("public_ownership")].set_index("asset_id")
    assert own.loc["ar7_a", "route"] == ROUTE_CFD_BILL
    assert own.loc["ar6_new", "route"] == ROUTE_CFD_PUBLIC_ACCOUNT
    assert own.loc["gas_new", "route"] == ROUTE_CAPACITY_MARKET
    assert own.loc["battery", "route"] == ROUTE_STORAGE
    assert own.loc["solar_merchant", "route"] == ROUTE_MERCHANT
    assert own.loc["unmapped", "route"] == "unresolved_cost_or_finance_mapping"
    assert own.loc["unmapped", "saving_economic_life_gbp"] == 0.0
    routes = route_summary(savings, summary)
    for case_id, frame in routes.groupby("case_id"):
        assert frame.identity_within_tolerance.all()
        assert abs(frame.identity_residual_gbp_2024.iloc[0]) <= 1.0
        assert frame.saving_economic_life_gbp_2024.sum() == pytest.approx(
            float(summary.set_index("case_id").loc[case_id, "annual_financing_saving_vs_private_gbp"]), abs=1e-6)


def test_historical_groups_have_no_new_build_cohort_and_zero_savings(defaults, bill_table):
    assets = pd.DataFrame([
        _asset("old_a", "wind_offshore", "Offshore wind", "legacy", "CfD", "AR2", "pre_policy", 2017, 500.0, 2.0e6),
        _asset("old_b", "solar_pv", "Solar PV", "legacy", "merchant", "", "not_applicable", np.nan, 300.0, 3.0e5),
    ])
    register, summary = calculate_financing_scenarios(assets, _costs(), _snapshot(), price_year=2024,
                                                      state_operating_overhead=NO_OVERHEAD)
    totals = summary.set_index("case_id")["annual_financing_saving_vs_private_gbp"]
    assert totals["public_coordination"] == 0.0 and totals["public_ownership"] == 0.0
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    savings = per_row_savings(register, settings)
    assert (savings.saving_economic_life_gbp == 0.0).all() and (savings.saving_bill_term_gbp == 0.0).all()
    adjustments = build_strike_adjustments(savings, bill_table, settings)
    assert (adjustments.delta_gbp_per_mwh_2024 == 0.0).all()


def test_ordering_and_vat_once_through_the_live_projector(defaults):
    from tests.unit.test_future_consumer_bill import fixture_inputs
    from scripts.consumer_bill.project_future_bill import project_components

    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    totals = []
    for payment in (400.0, 300.0, 100.0):  # private, coordination, ownership
        annual_case = dict(annual, cfd_payment_gbp=payment)
        rows, summary = project_components(annual_case, assumptions, case, balancing, recovery, ro)
        assert summary["vat_gbp"] == pytest.approx(summary["ex_vat_gbp"] * summary["vat_rate"])
        assert summary["total_gbp_inc_vat"] == pytest.approx(summary["ex_vat_gbp"] + summary["vat_gbp"])
        totals.append(summary["total_gbp_inc_vat"])
    assert totals[0] > totals[1] > totals[2]
    # Only the CfD line differs between cases in Release 1.
    base = project_components(dict(annual, cfd_payment_gbp=400.0), assumptions, case, balancing, recovery, ro)[0].set_index("component_id")[VALUE]
    other = project_components(dict(annual, cfd_payment_gbp=100.0), assumptions, case, balancing, recovery, ro)[0].set_index("component_id")[VALUE]
    changed = (base - other).loc[lambda s: ~np.isclose(s, 0.0)].index.tolist()
    assert changed == ["cfd"]


def test_bill_link_decisions_are_validated(defaults):
    config = deepcopy(defaults["public_ownership"])
    forecasts = defaults["consumer_bill"]["forecasts"]["cases"]
    assert validate_public_ownership_config(config, forecasts) is config
    # Item BM1: the P2 and P3 groups are further bill groups, each on its own forecast case.
    # Item F12 (12 Oct 2026): price-level twins (*_cp32) are tested in test_public_ownership_price_level_twins.py.
    twins = price_level_twin_groups(config)
    # P1/P2 under central dispatch (20 Oct 2026): an institution twin is not a model-price role either.
    from scripts.public_ownership.reference_runs import institution_twin_groups

    twins = {**twins, **institution_twin_groups(config)}
    # Network bounds (18-19 Oct 2026): one market, P2 and P3 group per network case.
    # F28: the legislation-year register bound re-accounts the market group's months; not a role of its own.
    model_price = [g for g in consumer_bill_groups(config, forecasts) if g not in twins
                   and config["reference_run_groups"][g].get("register_bound_of") is None]
    for network, base in (("cp30", "cp30_2030_cp30_network"), ("no_acceleration", "cp30_2030_no_acceleration"),
                          ("no_upgrades", "cp30_2030_no_upgrades")):
        assert [g for g in model_price if config["reference_run_groups"][g].get("network_case") == network] == [
            base, f"{base}_public_bids_p2", f"{base}_public_bids_p3"]
    for mutate, message in [
        (lambda c: c["consumer_bill"]["cfd_repricing"].__setitem__("eligibility", "everything"), "eligibility"),
        (lambda c: c["consumer_bill"]["cfd_repricing"].__setitem__("amortisation", "perpetuity"), "amortisation"),
        (lambda c: c["consumer_bill"]["cfd_repricing"].__setitem__("contract_term_years", 0), "contract_term_years"),
        (lambda c: c["consumer_bill"]["cfd_repricing"].__setitem__("strike_basis", "carrier_fallback"), "strike_basis"),
        (lambda c: c["consumer_bill"]["label_corrections"]["support_type"].__setitem__("marine", "ROC"), "support_type"),
        (lambda c: c["consumer_bill"]["label_corrections"].__setitem__("solar_cfd_share", 0.5), "solar_cfd_share"),
        (lambda c: c["consumer_bill"].__setitem__("forecast_case", "Nope"), "forecast_case"),
        # The bill link's forecast case loses its group (and its price-level and institution twins, which share
        # its months).
        (lambda c: [c["reference_run_groups"][g].__setitem__("scenario_pattern", "Other_{month}")
                    for g in ("cp30_2030_cp30_network", "cp30_2030_cp30_network_cp32", "cp30_2030_cp30_network_cd")], "source_scenario_pattern"),
    ]:
        broken = deepcopy(config)
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            validate_public_ownership_config(broken, forecasts)
    # Historical groups produce no bill.
    assert "historical_2024" not in consumer_bill_groups(config, forecasts)


# ----------------------------------------------------------------------------- repository outputs (skipped until built)


@pytest.mark.skipif(not (PUBLIC / "cp30_2030_cp30_network_consumer_bill_by_ownership_case_summary.json").exists(),
                    reason="per-case bill not built")
def test_repository_bill_link_outputs_satisfy_the_acceptance_criteria():
    report = json.loads((PUBLIC / "cp30_2030_cp30_network_consumer_bill_by_ownership_case_summary.json").read_text(encoding="utf-8"))
    assert report["private_reproduces_forecast_case"] is True
    cases = report["cases"]
    new_lines = set(report["new_lines"])
    assert cases["private"]["delta_net_vs_private_gbp_inc_vat"] == 0.0
    # Gross bills (before recycling and shortfall rows) never rise from private to coordination to ownership.
    assert cases["private"]["gross_total_gbp_inc_vat"] >= cases["public_control"]["gross_total_gbp_inc_vat"] >= cases["public_ownership"]["gross_total_gbp_inc_vat"]
    for case_id, entry in cases.items():
        for line, residual in entry["identity_residuals_gbp_2024"].items():
            assert abs(residual) <= 1.0, (case_id, line)
        assert abs(entry["route_identity_residual_gbp_2024"]) <= 1.0, case_id
        assert entry["vat_gbp"] == pytest.approx(entry["ex_vat_gbp"] * 0.05)
        # R1 re-prices the CfD line and the BECCS CfD carved out of it (MR7); the supplier line follows any
        # moved line through the cap's EBIT and headroom rule (F27 item 10).
        assert set(entry["lines_moved"]) <= set(R1_LINES) | new_lines | {SUPPLIER_LINE}
    by_case = pd.read_csv(PUBLIC / "cp30_2030_cp30_network_consumer_bill_by_ownership_case.csv")
    reference = pd.read_csv("resources/analysis/CP30_2030_cp30_network_consumer_bill_future_components.csv")
    private = by_case.loc[by_case.case_id.eq("private")].set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    # Every shared line reproduces the forecast bill; the new lines are added on top of it.
    assert np.allclose(private.reindex(reference.component_id).to_numpy(), reference["annual_gbp_per_3100kwh_ex_vat"].to_numpy())
    assert set(private.index) - set(reference.component_id) <= new_lines
    savings = pd.read_csv(PUBLIC / "cp30_2030_cp30_network_financing_saving_routes.csv")
    assert savings.identity_within_tolerance.all()



# ----------------------------------------------------------------------------- audit 26 Sep 2026: item 5

def _at_cost_register(amortisation):
    """The synthetic register with a settled contract revenue and the at-cost basis on the public case."""
    register, summary = _synthetic_register()
    register = register.copy()
    register["market_revenue_gbp"] = 250.0e6
    register["support_revenue_gbp"] = 0.0
    register["fuel_carbon_gbp"] = 0.0
    public = register["case_id"].eq("public_ownership")
    register.loc[public, "cfd_strike_basis"] = "at_cost"
    register.loc[public, "cfd_amortisation"] = amortisation
    return register


def test_at_cost_amortisation_over_the_term_recovers_capital_faster_and_takes_less(defaults):
    """public_ownership@amortised_over_term used to reproduce the central case: at cost never amortised."""
    block = deepcopy(defaults["public_ownership"]["consumer_bill"])
    block["cfd_repricing"]["contract_term_years_by_round"] = {"AR7": 20}
    settings = repricing_settings(block)
    life = per_row_savings(_at_cost_register("economic_life"), settings).set_index(["case_id", "asset_id"])
    term = per_row_savings(_at_cost_register("contract_term"), settings).set_index(["case_id", "asset_id"])
    a_life, a_term = life.loc[("public_ownership", "ar7_a")], term.loc[("public_ownership", "ar7_a")]
    assert a_life.at_cost_applied and a_term.at_cost_applied
    assert a_term.contract_term_years == 20.0
    register = _at_cost_register("contract_term").set_index(["case_id", "asset_id"])
    case, private = register.loc[("public_ownership", "ar7_a")], register.loc[("private", "ar7_a")]
    life_years = float(private.remaining_life_years)
    capex = float(private.annual_capital_recovery_gbp) / capital_recovery_factor(float(private.wacc), life_years)
    case_rr_term = float(case.annual_required_revenue_gbp) + capex * (
        capital_recovery_factor(float(case.wacc), 20) - capital_recovery_factor(float(case.wacc), life_years))
    assert a_term.saving_bill_term_gbp == pytest.approx(250.0e6 - case_rr_term, rel=1e-9)
    assert a_life.saving_bill_term_gbp == pytest.approx(250.0e6 - float(case.annual_required_revenue_gbp), rel=1e-9)
    # A 20-year annuity carries a higher annual charge than a 30-year one: the strike at cost is higher.
    assert a_term.saving_bill_term_gbp < a_life.saving_bill_term_gbp
    assert a_term.saving_at_cost_economic_life_gbp == pytest.approx(a_life.saving_bill_term_gbp)


def test_round_terms_apply_by_row_and_never_past_the_asset_life(defaults):
    block = deepcopy(defaults["public_ownership"]["consumer_bill"])
    block["cfd_repricing"]["contract_term_years"] = 15
    block["cfd_repricing"]["contract_term_years_by_round"] = {"AR7": 20, "AR8_modelled": 50}
    settings = repricing_settings(block)
    savings = per_row_savings(_at_cost_register("contract_term"), settings).set_index(["case_id", "asset_id"])
    assert savings.loc[("public_ownership", "ar7_a"), "contract_term_years"] == 20.0
    assert savings.loc[("public_ownership", "ar6_new"), "contract_term_years"] == 15.0
    # A 50-year term on a 38-year solar asset is capped at the asset's remaining life.
    solar = savings.loc[("public_ownership", "solar_future::new_build")]
    register = _at_cost_register("contract_term").set_index(["case_id", "asset_id"])
    assert solar.contract_term_years == float(register.loc[("private", "solar_future::new_build"), "remaining_life_years"])
    assert "contract_term_15y_AR720y_AR8_modelled50y" in savings.loc[("public_ownership", "ar7_a"), "method"]


def test_round_terms_are_validated(defaults):
    block = deepcopy(defaults["public_ownership"]["consumer_bill"])
    for bad in ({"AR7": 0}, {"AR7": 20.5}, ["AR7", 20], {"AR7": True}):
        block["cfd_repricing"]["contract_term_years_by_round"] = bad
        with pytest.raises(ValueError, match="contract_term_years_by_round"):
            repricing_settings(block)
