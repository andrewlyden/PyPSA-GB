"""Flow-of-funds fixes in the register, settlement and public account (26 Sep 2026).

B1 fuel and carbon on the resource basis, B2 the LDES floor to its payees, B3 the
booked Capacity Market and CCS amounts out of the state's rent, B4 the ownership
guard always, B5 the state overhead scaled once, B6 the CM booking on the CM
line's peak factor, B10 the hydrogen payee, B11 the DPA to Net Zero Teesside, B12
the R1 remainder the state keeps, D4(a) generator TNUoS and insurance, and the
corporation-tax wedge for the Exchequer ledger.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership.calculate_financing_scenarios import (
    _state_owned, calculate_financing_scenarios, generator_charges, load_generator_cost_tables,
    offshore_local_tariffs,
)
from scripts.public_ownership.finance import (
    TNUOS_CONVENTIONAL_CARBON, TNUOS_CONVENTIONAL_LOW_CARBON, TNUOS_INTERMITTENT, corporation_tax_wedges,
    embedded_corporation_tax, nearest_zone, wider_tnuos_tariff,
)
from scripts.public_ownership.levers import DEFAULT_SETTINGS
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    LINE_CM_BOOKED, LINE_CCS_BOOKED, booked_route_amounts, cm_household_allocation_factor,
)
from scripts.public_ownership.public_account import exchequer_position, state_surplus
from scripts.public_ownership.settlement import (
    FUEL_BASIS_RESOURCE, allocate_dpa_support, attach_settlement, combine_settlement_revenue,
    resource_fuel_carbon_rates,
)
from tests.unit.test_public_ownership_cfd_repricing import _asset, _costs, _snapshot
from config.config_loader import load_defaults

NO_OVERHEAD = {"enabled": False}
CONFIG = load_defaults()["public_ownership"]


# ── B1: fuel and carbon on the resource basis ────────────────────────────────

def test_fuel_and_carbon_come_from_the_resource_breakdown_not_the_balancing_floors():
    breakdown = pd.DataFrame([
        {"generator": "ccgt", "source": "formula+vom", "marginal_cost": 119.53, "fuel_cost_gbp_per_mwh": 65.72,
         "carbon_cost_gbp_per_mwh": 51.81, "variable_om_gbp_per_mwh": 2.0},
        {"generator": "efw", "source": "formula:gate_fee_covers_fuel_and_carbon", "marginal_cost": 0.0,
         "fuel_cost_gbp_per_mwh": 0.0, "carbon_cost_gbp_per_mwh": 0.0, "variable_om_gbp_per_mwh": 0.0},
        {"generator": "chp", "source": "formula-heat_credit(79.0)", "marginal_cost": 66.32,
         "fuel_cost_gbp_per_mwh": 81.24, "carbon_cost_gbp_per_mwh": 64.05, "variable_om_gbp_per_mwh": 0.0},
        {"generator": "roc_wind", "source": "subsidy_ROC", "marginal_cost": -70.24, "fuel_cost_gbp_per_mwh": 0.0,
         "carbon_cost_gbp_per_mwh": 0.0, "variable_om_gbp_per_mwh": 0.0},
        {"generator": "nuke", "source": "nuclear_override", "marginal_cost": 10.0, "fuel_cost_gbp_per_mwh": 0.0,
         "carbon_cost_gbp_per_mwh": 0.0, "variable_om_gbp_per_mwh": 0.0},
    ])
    rates = resource_fuel_carbon_rates(breakdown)
    # Network VOM is not fuel (the register adds DESNZ VOM once).
    assert rates["ccgt"] == pytest.approx(65.72 + 51.81)
    # The balancing floor of 30 GBP/MWh on waste to energy is not fuel: the gate fee covers it.
    assert rates["efw"] == 0.0
    # CHP: fuel and carbon net of the heat credit (the electricity's share).
    assert rates["chp"] == pytest.approx(66.32)
    assert rates["roc_wind"] == 0.0
    assert rates["nuke"] == pytest.approx(10.0)
    assert FUEL_BASIS_RESOURCE == "resource_breakdown"


# ── B11: the DPA line to Net Zero Teesside ───────────────────────────────────

def test_the_dpa_line_is_paid_to_the_dpa_plant_and_counted_as_support_revenue():
    out = pd.DataFrame({
        "component_type": ["Generator", "Generator"], "asset_id": ["nzt", "ccgt"], "support_type": ["DPA", ""],
        "balancing_mwh": [1.35e6, 7.0e6], "wholesale_revenue_gbp": [1.0e8, 5.0e8], "balancing_settlement_gbp": [0.0, 0.0],
        "cfd_support_gbp": [0.0, 0.0], "cfd_curtailment_compensation_gbp": [0.0, 0.0], "roc_support_gbp": [0.0, 0.0],
    })
    out = combine_settlement_revenue(allocate_dpa_support(out, 4.03e8), priced_on_contract_value=False)
    by_id = out.set_index("asset_id")
    assert by_id.loc["nzt", "dpa_support_gbp"] == pytest.approx(4.03e8)
    assert by_id.loc["nzt", "support_revenue_gbp"] == pytest.approx(4.03e8)
    assert by_id.loc["ccgt", "support_revenue_gbp"] == 0.0
    with pytest.raises(ValueError, match="no generator is labelled DPA"):
        allocate_dpa_support(out.assign(support_type=""), 1.0)


def test_descriptive_settlement_columns_are_copied_not_scaled_by_the_cohort_share():
    assets = pd.DataFrame([{"component_type": "Generator", "asset_id": "farm::legacy", "cohort_legacy_share": 0.25}])
    settlement = pd.DataFrame([{"component_type": "Generator", "asset_id": "farm", "market_revenue_gbp": 100.0,
                                "support_revenue_gbp": 0.0, "fuel_carbon_gbp": 0.0, "bus": "ABHA11",
                                "bus_lat": 57.1, "bus_lon": -4.2, "payee_class": "register_asset",
                                "fuel_carbon_basis": "resource_breakdown"}])
    out = attach_settlement(assets, settlement).iloc[0]
    assert out["market_revenue_gbp"] == pytest.approx(25.0)
    assert out["bus_lat"] == pytest.approx(57.1) and out["bus"] == "ABHA11"


# ── D4(a): generator TNUoS and insurance ─────────────────────────────────────

def test_the_wider_tariff_reproduces_the_neso_2030_31_example_tariffs():
    """Five-Year View Table 6, zone 1 North Scotland, at NESO's example ALFs."""
    zone = pd.read_csv("data/public_ownership/financing/neso_tnuos_generation_wider_tariffs_2030_31.csv").iloc[0]
    kwargs = dict(system_peak=zone.system_peak_gbp_per_kw, shared_year_round=zone.shared_year_round_gbp_per_kw,
                  not_shared_year_round=zone.not_shared_year_round_gbp_per_kw, adjustment=zone.adjustment_gbp_per_kw)
    assert wider_tnuos_tariff(tnuos_class=TNUOS_CONVENTIONAL_CARBON, annual_load_factor=0.40, **kwargs) == pytest.approx(41.421469, abs=1e-5)
    assert wider_tnuos_tariff(tnuos_class=TNUOS_CONVENTIONAL_LOW_CARBON, annual_load_factor=0.75, **kwargs) == pytest.approx(80.666363, abs=1e-5)
    assert wider_tnuos_tariff(tnuos_class=TNUOS_INTERMITTENT, annual_load_factor=0.45, **kwargs) == pytest.approx(51.467219, abs=1e-5)
    # Zone 27 West Devon and Cornwall, intermittent: a credit.
    last = pd.read_csv("data/public_ownership/financing/neso_tnuos_generation_wider_tariffs_2030_31.csv").iloc[-1]
    assert wider_tnuos_tariff(tnuos_class=TNUOS_INTERMITTENT, annual_load_factor=0.45,
                              system_peak=last.system_peak_gbp_per_kw, shared_year_round=last.shared_year_round_gbp_per_kw,
                              not_shared_year_round=last.not_shared_year_round_gbp_per_kw,
                              adjustment=last.adjustment_gbp_per_kw) == pytest.approx(-10.250220, abs=1e-5)


def test_zones_and_offshore_local_tariffs_are_found_by_location_and_name():
    zones = pd.read_csv("data/public_ownership/financing/neso_tnuos_generation_wider_tariffs_2030_31.csv")
    # Caithness, Peterhead, London, Cornwall.
    assert nearest_zone([58.5, 57.5, 51.5, 50.3], [-3.3, -1.8, -0.1, -5.0], zones) == [1, 2, 23, 27]
    assert nearest_zone([np.nan], [0.0], zones) == [None]
    table = pd.read_csv("data/public_ownership/financing/neso_tnuos_offshore_local_tariffs_2026_27.csv")
    values, basis = offshore_local_tariffs(pd.Series(["Hornsea_1___Heron_&_Njord", "Walney�1", "Seagreen_Phase_1A",
                                                      "CP30_increment_wind_offshore_1_AR7::new_build"]), table)
    assert values.iloc[0] == pytest.approx(10.046396 + 35.545717)
    assert values.iloc[1] == pytest.approx(31.872687 + 63.721604)
    # Seagreen Phase 1A is not the transferred Seagreen 1; it and the proxy take the median.
    assert list(basis) == ["neso_table", "neso_table", "table_median_untransferred", "table_median_untransferred"]


def _charge_register(**overrides):
    assets = pd.DataFrame([
        _asset("offshore", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6,
               bus_lat=53.7, bus_lon=0.1),
        _asset("small_solar", "solar_pv", "Solar PV", "new_build", "merchant", "", "not_applicable", np.nan, 20.0, 2.0e4,
               bus_lat=51.2, bus_lon=-0.6),
        _asset("gas", "CCGT", "Gas generation - unabated", "new_build", "", "", "not_applicable", np.nan, 400.0, 1.0e6,
               bus_lat=51.5, bus_lon=0.9),
    ])
    block = dict(CONFIG["generator_costs"])
    tables = load_generator_cost_tables(block)
    scenarios = {"private": {}, "public_ownership": {"new_build_generation": "public", "public_return": "public_rate"}}
    return calculate_financing_scenarios(
        assets, _costs(), _snapshot(), scenarios=scenarios,
        sensitivities={"structural": [], "one_at_a_time": [], "named": {}, "bounding_pair": False},
        price_year=2024, state_operating_overhead=NO_OVERHEAD,
        generator_costs=overrides.get("generator_costs", block), generator_cost_tables=tables)


def test_tnuos_and_insurance_are_added_once_to_every_case_and_leave_the_financing_saving_alone():
    register, totals = _charge_register()
    reg = register.set_index(["case_id", "asset_id"])
    off = reg.loc[("private", "offshore")]
    # Offshore wind: BVG insurance per kW, the wider tariff at its zone and ALF,
    # plus the median offshore local tariff, all folded into fixed O&M once.
    assert off.annual_insurance_gbp == pytest.approx(1000.0 * 1000.0 * 11.22)
    assert off.tnuos_transmission_connected and off.tnuos_offshore_local_basis == "table_median_untransferred"
    assert off.annual_generator_tnuos_gbp == pytest.approx(
        (off.tnuos_wider_gbp_per_kw + off.tnuos_offshore_local_gbp_per_kw) * 1.0e6)
    assert off.annual_fixed_om_gbp == pytest.approx(
        off.annual_site_fixed_om_gbp + off.annual_generator_tnuos_gbp + off.annual_insurance_gbp)
    assert off.annual_site_fixed_om_gbp == pytest.approx(46.0 * 1.0e6)
    # Gas: 0.4% of capex insurance; transmission-connected whatever its size.
    gas = reg.loc[("private", "gas")]
    assert gas.annual_insurance_gbp == pytest.approx(0.004 * 1020.0 * 400.0 * 1000.0)
    assert gas.tnuos_class == TNUOS_CONVENTIONAL_CARBON
    # 20 MW of solar in the south is embedded: no generator TNUoS.
    solar = reg.loc[("private", "small_solar")]
    assert not solar.tnuos_transmission_connected and solar.annual_generator_tnuos_gbp == 0.0
    # Identical in every case, so the financing saving does not move; the one declared
    # difference is follow-up F7: plant the state owns outright saves the insurer's loading.
    for asset in ("offshore", "gas", "small_solar"):
        assert reg.loc[("public_ownership", asset), "annual_generator_tnuos_gbp"] == pytest.approx(
            reg.loc[("private", asset), "annual_generator_tnuos_gbp"])
        saved = float(reg.loc[("public_ownership", asset), "self_insurance_saving_gbp"] or 0.0)
        assert reg.loc[("public_ownership", asset), "annual_insurance_gbp"] + saved == pytest.approx(
            reg.loc[("private", asset), "annual_insurance_gbp"])
        assert saved == pytest.approx(0.35 * reg.loc[("private", asset), "annual_insurance_gbp"]) or saved == 0.0
    _, off_totals = _charge_register(generator_costs={})
    t, o = totals.set_index("case_id"), off_totals.set_index("case_id")
    added = (t.loc["private", "annual_generator_tnuos_gbp"] + t.loc["private", "annual_insurance_gbp"])
    assert t.loc["private", "annual_required_revenue_gbp"] - o.loc["private", "annual_required_revenue_gbp"] == pytest.approx(added)
    # The charges leave the financing saving alone apart from F7's self-insurance, which adds its saving.
    self_insured = pd.to_numeric(register.loc[register.case_id.eq("public_ownership"), "self_insurance_saving_gbp"],
                                 errors="coerce").fillna(0.0).sum()
    assert self_insured > 0
    assert t.loc["public_ownership", "annual_financing_saving_vs_private_gbp"] == pytest.approx(
        o.loc["public_ownership", "annual_financing_saving_vs_private_gbp"] + self_insured)


def test_generator_charges_refuse_an_enabled_tariff_without_its_tables():
    frame = pd.DataFrame([_asset("x", "CCGT", "Gas generation - unabated", "new_build", "", "", "", np.nan, 1.0, 1.0)])
    with pytest.raises(ValueError, match="tariff tables"):
        generator_charges(frame, {"tnuos": {"enabled": True}}, {})


# ── B4: the ownership guard always ───────────────────────────────────────────

def test_foreign_and_uncostable_new_build_is_never_state_owned_even_without_an_acquisition_plan():
    settings = dict(DEFAULT_SETTINGS, new_build_generation="public")
    foreign = pd.Series({"finance_vintage": "new_build", "carrier": "EU_market", "evidence_status": "excluded"})
    chp = pd.Series({"finance_vintage": "new_build", "carrier": "CHP", "evidence_status": "excluded"})
    wind = pd.Series({"finance_vintage": "new_build", "carrier": "wind_onshore", "evidence_status": "resolved",
                      "component_type": "Generator", "support_type": "merchant"})
    assert not _state_owned(foreign, settings, None, 0)
    assert not _state_owned(chp, settings, None, 0)
    assert _state_owned(wind, settings, None, 0)


# ── B5: the state overhead scaled once ───────────────────────────────────────

def test_a_stake_carries_its_whole_share_of_the_centre_not_the_share_squared():
    # 30% stake, whole-row rent 100 before the state overhead of 3 (already the state's share).
    rent_with_overhead = pd.Series([100.0 - 3.0])
    assert state_surplus(pd.Series([0.3]), rent_with_overhead, pd.Series([3.0])).iloc[0] == pytest.approx(0.3 * 100.0 - 3.0)


# ── B6: the CM booking on the CM line's peak factor ──────────────────────────

def test_the_capacity_market_booking_uses_the_cm_line_household_factor():
    import json
    assumptions = json.loads(Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json").read_text(encoding="utf-8"))
    factor = cm_household_allocation_factor(assumptions)
    assert factor == pytest.approx(1.348308)
    account = {"sensitivity_axes": dict(DEFAULT_SETTINGS, route_booking="booked"),
               "capacity_market_sensitivity_gbp": 1.0e8, "ccs_levy_sensitivity_gbp": 1.0e7}
    flat = booked_route_amounts(account, factor=1.0, final_demand_mwh=3.0e8, eligible_mwh=2.0e8, mwh=3.1,
                                cm_line=100.0, ccs_line=100.0)
    peak = booked_route_amounts(account, factor=1.0, final_demand_mwh=3.0e8, eligible_mwh=2.0e8, mwh=3.1,
                                cm_line=100.0, ccs_line=100.0, cm_household_factor=factor)
    assert peak[LINE_CM_BOOKED] == pytest.approx(flat[LINE_CM_BOOKED] * factor)
    assert peak[LINE_CCS_BOOKED] == pytest.approx(flat[LINE_CCS_BOOKED])


# ── B3 and D2: what the Exchequer carries ────────────────────────────────────

def test_the_exchequer_position_separates_deficits_from_surpluses():
    case = {"r2_cfd_surplus_gbp": 50.0, "r1_retained_rent_gbp": 10.0, "merchant_generation_surplus_gbp": 30.0,
            "shortfall_on_exchequer_gbp": 5.0}
    rows = pd.DataFrame({"public_surplus_gbp": [-20.0, 4.0, -7.0, 2.0]},
                        index=["storage_battery", "storage_ldes", "capacity_market_route_generation",
                               "dispatchable_cfd_route_generation"])
    retained = exchequer_position(case, rows, "retained_by_treasury")
    assert retained["state_deficit_exchequer_funded_gbp"] == pytest.approx(20.0 + 7.0 + 5.0)
    assert retained["state_surplus_retained_gbp"] == pytest.approx(60.0 + 30.0 + 4.0 + 2.0)
    assert retained["state_retained_net_gbp"] == pytest.approx(96.0 - 27.0 - 5.0)
    # Under the at-cost CfD the merchant surplus goes to the bill, not the Exchequer.
    at_cost = exchequer_position(case, rows, "public_cfd_at_cost")
    assert at_cost["state_surplus_retained_gbp"] == pytest.approx(66.0)
    assert exchequer_position(case, rows, "not_applicable")["state_deficit_exchequer_funded_gbp"] == pytest.approx(5.0)


def test_the_corporation_tax_wedge_is_the_tax_on_equity_returns():
    snapshot = yaml.safe_load(Path("resources/public_ownership/workbook_v5_assumptions.yaml").read_text(encoding="utf-8")) \
        if Path("resources/public_ownership/workbook_v5_assumptions.yaml").exists() else None
    if snapshot is None:
        pytest.skip("workbook snapshot not built")
    wedges = corporation_tax_wedges(snapshot)
    solar = wedges["Solar PV"]
    assert 0.0 < solar["public_coordination"] < solar["private"] < 0.25 * 0.2
    assert solar["tax_rate"] == 0.25
    assert embedded_corporation_tax(1.0e9, 0.07, solar["private"], 30.0) > 0.0
    assert corporation_tax_wedges(_snapshot()) == {}


def test_an_at_cost_cut_on_a_stake_comes_out_of_the_state_share_only():
    """D3 sizes the at-cost cut on the state share alone, so the co-owner is untouched."""
    from scripts.public_ownership.cfd_repricing import per_row_savings, repricing_settings
    from scripts.public_ownership.public_account import account_settings, build_public_account
    from tests.unit.test_public_ownership_public_account import _central, _settled_register

    register, _ = _settled_register(_central(cfd_strike_basis="at_cost"))
    savings = per_row_savings(register, repricing_settings(CONFIG["consumer_bill"]))
    _, report = build_public_account(register, savings, account_settings(CONFIG))
    row = savings.set_index(["case_id", "asset_id"]).loc[("public_coordination", "ar7")]
    assert bool(row["at_cost_applied"])
    coord = register.loc[register.case_id.eq("public_coordination")].set_index("asset_id").loc["ar7"]
    case = report["public_coordination"]
    assert case["r1_cut_borne_by_state_gbp"] == pytest.approx(row["saving_bill_term_gbp"])
    assert case["r1_retained_rent_gbp"] == pytest.approx(0.3 * coord.producer_rent_gbp - row["saving_bill_term_gbp"])
