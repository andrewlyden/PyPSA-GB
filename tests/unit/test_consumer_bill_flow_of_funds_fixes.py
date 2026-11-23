"""Flow-of-funds fixes to the consumer bill (26 Sep 2026): D1, D4(c), D4(d), B7, B8, B9."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill.project_future_bill import (
    project_components, read_balancing_costs, read_hourly_inputs, read_ldes_topup,
    read_recovery_denominators, resolve_sourced_externals,
)
from tests.unit.test_future_consumer_bill import (
    ASSUMPTIONS_PATH, _bid_audit, _toy_month_inputs, fixture_inputs,
)

RECOVERY_PATH = Path("data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json")
VALUE = "annual_gbp_per_3100kwh_ex_vat"


def _line(rows, component_id):
    return float(rows.set_index("component_id").loc[component_id, VALUE])


# ── D1: metered supply ──────────────────────────────────────────────────────

def test_recovery_file_is_on_metered_supply_with_electrolysers_paying():
    document = json.loads(RECOVERY_PATH.read_text())
    basis = document["metered_supply_basis"]
    recovery = read_recovery_denominators(RECOVERY_PATH, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"])
    gb, draw, offtake = (basis[k]["value"] for k in ("gb_electrical_load_mwh", "electrolyser_electricity_draw_mwh",
                                                     "h2_offtake_mwh"))
    # The hydrogen offtake is in the model's load total but pays nothing.
    assert gb + offtake == pytest.approx(document["system_demand_mwh"]["value"])
    # F18 (6 Oct 2026): the load contains losses, so metered supply and the GSP charging volume are
    # derived from it on declared factors; the electrolysers pay on their own draw in both.
    assert recovery["metered_supply_mwh"] == pytest.approx(gb / 1.14 + draw)
    assert recovery["gsp_charging_volume_mwh"] == pytest.approx(gb / 1.14 * 1.085 + draw)
    assert recovery["ro_eligible_metered_supply_mwh"] == pytest.approx(
        recovery["metered_supply_mwh"] * recovery["cfd_eligible_share"])
    assert recovery["cfd_eligible_gsp_volume_mwh"] == pytest.approx(
        recovery["gsp_charging_volume_mwh"] * recovery["cfd_eligible_share"])
    # One household loss factor: the procurement losses line applies the same multiplier.
    assumptions = json.loads(Path(ASSUMPTIONS_PATH).read_text())
    losses = assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["allowances"]["losses"]
    assert recovery["household_loss_factor"] == losses["loss_multiplier_gb_average"] == 1.1313


def test_recovery_file_is_checked_against_the_run_volume_by_volume():
    document = json.loads(RECOVERY_PATH.read_text())
    basis = document["metered_supply_basis"]
    volumes = {"gb_electrical_load_mwh": basis["gb_electrical_load_mwh"]["value"],
               "electrolyser_draw_mwh": basis["electrolyser_electricity_draw_mwh"]["value"],
               "h2_offtake_mwh": basis["h2_offtake_mwh"]["value"]}
    read_recovery_denominators(RECOVERY_PATH, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"], volumes)
    volumes["electrolyser_draw_mwh"] *= 1.1
    with pytest.raises(ValueError, match="electrolyser_draw_mwh"):
        read_recovery_denominators(RECOVERY_PATH, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"], volumes)


def test_inconsistent_metered_supply_is_refused(tmp_path):
    document = json.loads(RECOVERY_PATH.read_text())
    basis = document["metered_supply_basis"]
    basis["metered_supply_mwh"]["value"] = basis["gsp_charging_volume_mwh"]["value"]
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="metered_supply_mwh does not equal"):
        read_recovery_denominators(path, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"])


def test_hourly_reader_drops_the_hydrogen_offtake_and_adds_electrolyser_draw(tmp_path):
    import xarray as xr
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    ds = xr.Dataset(
        {"snapshots_snapshot": ("snapshots", snapshots), "snapshots_generators": ("snapshots", [1., 1.]),
         "buses_carrier": ("buses_i", ["AC", "AC", "H2_gas"]), "buses_country": ("buses_i", ["GB", "GB", ""]),
         "loads_bus": ("loads_i", ["A", "B", "H2"]), "loads_p_set": ("loads_i", [0., 0., 7.]),
         "loads_t_p_set": (("snapshots", "loads_t_p_set_i"), [[10., 20.], [30., 40.]]),
         "links_bus0": ("links_i", ["A", "A"]), "links_bus1": ("links_i", ["H2", "B"]),
         "links_carrier": ("links_i", ["electrolysis", "AC"])},
        coords={"buses_i": ["A", "B", "H2"], "loads_i": ["la", "lb", "h2"], "loads_t_p_set_i": ["la", "lb"],
                "links_i": ["electrolysis_A", "line_AB"]})
    network = tmp_path / "x_wholesale.nc"
    ds.to_netcdf(network)
    price = tmp_path / "price.csv"
    pd.DataFrame({"wholesale_price": [10., 40.]}, index=snapshots).to_csv(price)
    links = tmp_path / "links.csv"
    pd.DataFrame({"electrolysis_A": [5., 0.], "line_AB": [100., -100.]}, index=snapshots).to_csv(links)
    frame = read_hourly_inputs(network, price, links)
    assert frame.total_load_mw.tolist() == [37., 77.]          # the market summary's count
    assert frame.h2_offtake_mw.tolist() == [7., 7.]
    assert frame.electrolyser_draw_mw.tolist() == [5., 0.]     # a flow on an electrical link is not a draw
    assert frame.demand_mw.tolist() == [35., 70.]              # electrical load + electrolyser draw
    with pytest.raises(ValueError, match="electrolyser links"):
        read_hourly_inputs(network, price, None)


def test_gsp_rate_lines_carry_the_loss_uplift_and_the_loss_basis_is_single():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    bridge = assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["allowances"]["losses"]
    bridge["loss_multiplier_gb_average"] = 1.1
    recovery = dict(recovery, gsp_to_meter_loss_factor=1.1, gsp_chargeable_volume_mwh=110.0)
    assumptions["sourced_externals"]["aahedc"]["value"] = 1.0
    other = assumptions["sourced_externals"]["bsuos_other_services"]
    other["value"] = 2.0
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    # OW16 follow-up (3 Oct 2026): BSUoS is charged on metered supply, so its lines carry no loss uplift.
    assert "loss_uplift" not in other
    assert _line(rows, "bsuos_other_services") == pytest.approx(2.0 * other["rebasing"]["factor"])
    assert _line(rows, "aahedc") == pytest.approx(1.0 * assumptions["sourced_externals"]["aahedc"]["rebasing"]["factor"] * 1.1)
    assert summary["recovery_basis"]["gsp_to_meter_loss_factor"] == 1.1
    recovery["gsp_to_meter_loss_factor"] = 1.05
    with pytest.raises(ValueError, match="one loss basis"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_capacity_market_divides_by_the_gsp_volume_not_metered_supply():
    assumptions = json.loads(Path(ASSUMPTIONS_PATH).read_text())
    cm = assumptions["sourced_externals"]["capacity_market"]
    # F18: the GSP charging volume, with no loss factor beyond the household factor, which already carries it.
    assert cm["derivation"]["recovery_denominator"] == "gsp_charging_volume_mwh"
    assert cm["derivation"]["household_losses"] == "in_household_allocation_factor"
    recovery = {"recovery_basis": "loss_inclusive_two_volume", "metered_supply_mwh": 100.0,
                "gsp_charging_volume_mwh": 110.0, "household_loss_factor": 1.1313}
    resolved = resolve_sourced_externals({"capacity_market": cm}, recovery, 3100)["capacity_market"]
    expected = cm["derivation"]["agreement_cost_gbp"] / 110.0 * 3.1 * cm["derivation"]["household_allocation_factor"]
    assert resolved["value"] == pytest.approx(expected)


# ── D4(c): LDES in the forecast bill ─────────────────────────────────────────

def test_ldes_line_is_part_of_the_forecast_bill():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["model_linked_rules"]["ldes_cap_and_floor"]["enabled"] = True
    with pytest.raises(ValueError, match="no LDES floor top-up"):
        project_components(annual, assumptions, case, balancing, recovery, ro)
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro,
                                       ldes={"topup_gbp": 200.0, "case_id": "private", "source": "test"})
    assert _line(rows, "ldes_cap_and_floor") == pytest.approx(200.0 / 100.0 * 3.1)
    assert summary["ldes_cap_and_floor"]["topup_gbp"] == 200.0
    # An extra component with the same id is refused, so no caller can add it twice.
    extra = [{"component_id": "ldes_cap_and_floor", "stack_group": "policy", "label": "x",
              "annual_gbp_per_3100kwh_ex_vat": 1.0, "method": "x", "source": "x"}]
    with pytest.raises(ValueError, match="duplicates a projector line"):
        project_components(annual, assumptions, case, balancing, recovery, ro,
                           ldes={"topup_gbp": 200.0}, extra_components=extra)


def test_ldes_topup_is_read_from_the_private_case_of_the_public_account(tmp_path):
    path = tmp_path / "account.json"
    path.write_text(json.dumps({"cases": {
        "private": {"ldes_cap_and_floor_enabled": True, "ldes_floor_topup_gbp": 1.5e9, "ldes_floor_real_rate": 0.045},
        "off": {"ldes_cap_and_floor_enabled": False, "ldes_floor_topup_gbp": 9.0}}}))
    assert read_ldes_topup(path)["topup_gbp"] == 1.5e9
    assert read_ldes_topup(path, "off")["topup_gbp"] == 0.0
    with pytest.raises(ValueError, match="no case"):
        read_ldes_topup(path, "missing")


def test_ownership_projector_passes_each_case_its_own_ldes_topup():
    from scripts.public_ownership.project_public_ownership_consumer_bill import ldes_topup_for
    assert ldes_topup_for({"ldes_cap_and_floor_enabled": True, "ldes_floor_topup_gbp": 7.0}) == 7.0
    assert ldes_topup_for({"ldes_cap_and_floor_enabled": False, "ldes_floor_topup_gbp": 7.0}) == 0.0
    assert ldes_topup_for(None) is None


# ── D4(d): voltage and stability constraints, supplier imbalance ─────────────

def test_voltage_and_stability_line_is_neso_constraints_less_thermal():
    assumptions = json.loads(Path(ASSUMPTIONS_PATH).read_text())
    block = assumptions["sourced_externals"]["bsuos_voltage_and_stability_constraints"]
    d = block["derivation"]["fy2024_25"]
    breakdown = pd.read_csv(block["source_file"], encoding="utf-8-sig")
    assert breakdown["Thermal constraints cost"].sum() == pytest.approx(d["thermal_constraints_gbp_2024_25"])
    split = json.loads(Path("data/market/neso_balancing_costs/bsuos_constraint_split_2024_25.json").read_text())
    assert d["constraints_category_gbp_2024_25"] == pytest.approx(split["constraint_cost_gbp"])
    # OW16 follow-up (3 Oct 2026): a cost, not a per-MWh GSP rate, and no loss uplift.
    assert block["unit"] == "gbp_per_household_year" and "loss_uplift" not in block
    # The three BSUoS lines still tile NESO's FY2024/25 total at its own volume, to within the
    # internal-cost estimate and the breakdown's GBP 1.2 m gap (about 1%), with the services at their
    # FY2024/25 outturn (the 2030 lines carry them to 2030 on CP30's drivers, user decision 6 Oct 2026).
    volume = split["chargeable_final_demand_mwh"]
    other = assumptions["sourced_externals"]["bsuos_other_services"]["derivation"]
    voltage_outturn = block["derivation"]["cp30_projection"]["previous_value"]["services_gbp"]
    rebuilt = (other["fy2024_25"]["services_outturn_gbp"] + other["demand_scaled_gbp_per_mwh"] * volume
               + voltage_outturn + d["thermal_constraints_gbp_2024_25"])
    assert rebuilt == pytest.approx(split["total_actual_bsuos_cost_gbp"], rel=0.01)


def test_bsuos_service_lines_carry_the_money_the_register_pays_providers():
    """OW16 follow-up (user, 3 Oct 2026), as FES 2050 decision 65: one figure in the bill and the register."""
    from scripts.consumer_bill.build_2050_bill_inputs import bsuos_rebase_2050
    from scripts.public_ownership import support_lines_2050 as sl

    se = json.loads(Path(ASSUMPTIONS_PATH).read_text())["sourced_externals"]
    other, volt = se["bsuos_other_services"]["derivation"], se["bsuos_voltage_and_stability_constraints"]["derivation"]
    held = sl.services_held_by_line(sl.load_ancillary(Path(other["ancillary_file"])), other["ancillary_case"])
    assert other["cost_gbp"] == pytest.approx(held["bsuos_other_services"])
    assert volt["cost_gbp"] == pytest.approx(held["bsuos_voltage_and_stability_constraints"])
    # The part that is not services is held per MWh at the rate 2050 uses.
    assert other["demand_scaled_gbp_per_mwh"] == pytest.approx(bsuos_rebase_2050(Path("."))["other_rate_gbp_per_mwh"])
    # F18 (user, 6 Oct 2026): BSUoS is a GSP-rate line, on the GSP charging volume at the household loss factor.
    assert other["demand_scaled_volume"] == "gsp_charging_volume_mwh" == other["recovery_denominator"]
    assert other["household_losses"] == "household_loss_factor"
    recovery = {"recovery_basis": "loss_inclusive_two_volume", "gsp_charging_volume_mwh": 300e6,
                "household_loss_factor": 1.1313}
    resolved = resolve_sourced_externals({"bsuos_other_services": se["bsuos_other_services"]}, recovery, 3100)
    expected = (other["cost_gbp"] + other["demand_scaled_gbp_per_mwh"] * 300e6) / 300e6 * 3.1 * 1.1313
    assert resolved["bsuos_other_services"]["value"] == pytest.approx(expected)


def test_supplier_imbalance_allowance_is_charged_on_the_domestic_energy_cost():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    allowances = assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["allowances"]
    allowances["imbalance"]["rate"] = 0.02
    rows, _ = project_components(annual, assumptions, case, balancing, recovery, ro)
    # Domestic price 10 GBP/MWh x 3.1 MWh = 31.
    assert _line(rows, "procurement_imbalance") == pytest.approx(0.02 * 31.0)
    repository = json.loads(Path(ASSUMPTIONS_PATH).read_text())
    rate = repository["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["allowances"]["imbalance"]["rate"]
    inputs = pd.read_csv("data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv")
    assert rate == pytest.approx(float(inputs["imbalance_rate"].mean()))


# ── B8: the constraint line excludes slack and penalty turn-down ─────────────

def test_constraint_cost_excludes_voll_slack_and_non_participant_penalty(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import extract_month
    from scripts.market.cfd_strike_table import StrikeTable
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing, price, summary, _ = _toy_month_inputs(tmp_path, snapshots)
    redispatch = tmp_path / "redispatch_b8.csv"
    pd.DataFrame({"carrier": ["wind_offshore", "load_shedding", "solar_pv"],
                  "increase_MWh": [0.0, 2.0, 0.0], "decrease_MWh": [10.0, 0.0, 0.1],
                  "offer_cost": [0.0, 120.0, 0.0], "bid_cost": [3800.0, 0.0, 600.0]},
                 index=["curtailed", "shed", "tiny"]).to_csv(redispatch, index_label="name")
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0}, 2024, {}, {})
    _, month, _ = extract_month("toy", wholesale, balancing, price, summary, table,
                                compensation_treatment="add_post_hoc", redispatch_summary_path=redispatch,
                                contract_value_bids_path=_bid_audit(tmp_path, contract_value=False))
    assert month["feasibility_slack_cost_in_accepted_costs_gbp"] == pytest.approx(120.0)
    assert month["non_participant_turn_down_cost_at_penalty_gbp"] == pytest.approx(600.0)   # 6,000 GBP/MWh
    assert month["constraint_cost_gbp"] == pytest.approx(500.0 + 4000.0 + 4000.0 - 120.0 - 600.0)


def test_bill_refuses_an_extraction_that_still_carries_slack(tmp_path):
    annual = {"model_price_year": 2024, "constraint_cost_gbp": 100.0, "accepted_offer_cost_gbp": 60.0,
              "accepted_bid_cost_gbp": 50.0, "cfd_curtailment_compensation_gbp": 0.0, "cfd_curtailed_mwh": 0.0,
              "feasibility_slack_cost_gbp": 10.0, "feasibility_slack_energy_mwh": 1.0,
              "cfd_curtailment_compensation_treatment": "reconcile_against_lp_bid"}
    path = tmp_path / "balancing.json"
    path.write_text(json.dumps({"annual": annual}))
    with pytest.raises(ValueError, match="predates item B8"):
        read_balancing_costs(path, [], 2024)
    annual.update(constraint_cost_excludes_slack_and_penalty=True, feasibility_slack_cost_in_accepted_costs_gbp=10.0,
                  non_participant_turn_down_cost_at_penalty_gbp=0.0)
    path.write_text(json.dumps({"annual": annual}))
    assert read_balancing_costs(path, [], 2024)["constraint_cost_gbp"] == 100.0
    annual["constraint_cost_gbp"] = 110.0
    path.write_text(json.dumps({"annual": annual}))
    with pytest.raises(ValueError, match="does not equal accepted offer"):
        read_balancing_costs(path, [], 2024)


# ── B7: every pound of the Capacity Market book has a payee ──────────────────

def _cm_fleet_inputs_exist():
    from scripts.market import build_capacity_market as bcm
    return all(Path(p).exists() for p in ("data/market/capacity_market/cm_register_cmu_20260923.csv",
                                          bcm.FLEET_ASSETS, bcm.FLEET_INTERCONNECTORS, bcm.FLEET_ES1))


@pytest.mark.skipif(not _cm_fleet_inputs_exist(), reason="NESO register or the CP30 2030 fleet inputs not available")
def test_cm_revenue_by_technology_adds_up_to_the_bill_total():
    from scripts.market import build_capacity_market as bcm
    from scripts.market import capacity_market as cm

    register = cm.read_register()
    cpi = cm.read_cpi()
    agreements = cm.read_agreements(register)
    fleet = bcm.read_fleet_2030()
    model = bcm.auction_model(register, fleet, agreements, cpi)
    summary, book = bcm.calendar_2030(agreements, cpi, model)
    tech = bcm.revenue_by_technology(book, model, cpi, register)
    assert tech["calendar_2030_revenue_gbp"].sum() == pytest.approx(summary["calendar_2030_cost_gbp"], rel=1e-9)
    # The carrier split (user decision, 6 Oct 2026: the CP30 fleet's rows) reproduces the
    # category supply and the modelled award.
    central = model["t4_2030_31"]["central"]
    for category, by in central["entered_by_carrier_mw"].items():
        assert sum(by.values()) == pytest.approx(central["entered_mw"][category], abs=0.06)
    assert bcm.modelled_t4_by_carrier(model).sum() == pytest.approx(central["awarded_mw"])
    # New LDES on the cap and floor earns exactly its 2030/31 award (T-4 and its T-1 share, half a
    # delivery year), and existing pumped hydro the long-duration contract book (user decision, 6 Oct 2026).
    rates = tech.set_index("carrier")
    ldes = rates.loc[cm.LDES_CAP_AND_FLOOR_RATE_KEY]
    assert ldes["connection_basis"] == "award_2030_31"
    assert ldes["calendar_2030_revenue_gbp"] == pytest.approx(
        0.5 * (ldes["t4_2030_31_modelled_mw"] + ldes["t1_2030_31_mw"]) * 1e3 * central["price"])
    paid = cm.attach_capacity_market_revenue(pd.read_csv(bcm.FLEET_ASSETS, low_memory=False), tech)
    by_carrier = paid.groupby("carrier")["capacity_market_revenue_gbp"].sum()
    assert by_carrier["LAES"] == pytest.approx(ldes["calendar_2030_revenue_gbp"], rel=1e-9)
    assert by_carrier["pumped_hydro"] == pytest.approx(rates.loc["long_duration_storage", "calendar_2030_revenue_gbp"],
                                                       rel=1e-9)


def test_awarded_by_category_scales_to_the_award():
    from scripts.market import capacity_market as cm
    split = cm.awarded_by_category(pd.Series({"a": 100.0, "b": 50.0}), {"a": (1.0, 1.0), "b": (0.0, 1.0)},
                                   price=37.5, awarded_mw=110.0)
    assert split.sum() == pytest.approx(110.0)
    assert split["a"] / split["b"] == pytest.approx(100.0 / 25.0)


# ── B9: exports valued at the GB-side flow ───────────────────────────────────

def test_congestion_rent_values_exports_at_the_gb_side_flow():
    import logging
    from scripts.interconnectors.market_coupling import interconnector_trade_cost, link_price_tables
    from scripts.market.analyze_market import calculate_interconnector_metrics
    from tests.unit.test_signed_interconnector_piecewise import _network
    network, snapshots = _network()
    assert network.links.at["IC_TestCable", "bus1"] == "GB"      # p1 = -0.9 p0 is the GB side
    dispatch = pd.DataFrame({"IC_TestCable": [50.0, -40.0]}, index=snapshots)
    gb = pd.Series([50.0, 50.0], index=snapshots)
    metrics = calculate_interconnector_metrics(network, dispatch, gb, logging.getLogger(__name__))
    trade = float(interconnector_trade_cost(dispatch, *link_price_tables(network))["IC_TestCable"].sum())
    # Import delivers 45 MWh; the export takes 0.9 x 40 = 36 MWh out of GB, not 40.
    assert float(metrics["estimated_congestion_rent_GBP"].iloc[0]) == pytest.approx(50.0 * 45.0 - 50.0 * 36.0 - trade)
    assert float(metrics["gross_export_gb_side_MWh"].iloc[0]) == pytest.approx(36.0)
    assert float(metrics["net_import_MWh"].iloc[0]) == pytest.approx(9.0)


# ── NIRO: GB suppliers carry the GB share of the UK obligation ───────────────

def test_niro_stations_are_flagged_not_dropped():
    from scripts.consumer_bill.build_ro_eligible_fleet import read_eligible_fleet
    fleet = read_eligible_fleet("data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv")
    ni = fleet["Country"].eq("Northern Ireland")
    assert set(fleet["jurisdiction"]) == {"GB", "NI"}
    assert (fleet.loc[ni, "jurisdiction"] == "NI").all() and (fleet.loc[~ni, "jurisdiction"] == "GB").all()
    # Eligibility is unchanged by the flag, so the network's ROC labels are too.
    assert fleet.loc[ni & fleet.eligible, "capacity_mw"].sum() > 1000


def test_gb_share_of_the_uk_obligation_is_the_desnz_split():
    from scripts.consumer_bill.build_ro_eligible_fleet import gb_share_of_uk_obligation
    parameters = json.loads(Path("data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json").read_text())
    split = gb_share_of_uk_obligation(parameters)
    assert split["gb_share_of_uk_obligation"] == pytest.approx((120.1e6 - 7.5e6 * 0.184) / 120.1e6)
    assert "NIRO" in parameters["certificate_evidence"]["schemes"]
    with pytest.raises(ValueError, match="uk_obligation_split"):
        gb_share_of_uk_obligation({})
