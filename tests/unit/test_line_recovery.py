"""F18 (6 Oct 2026): two recovery volumes, per-line recovery, Ofgem's direct-fuel formula, EBIT and
headroom on the bill's own costs, direct-debit levelisation and the BICS sensitivity's scope."""

import copy
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from scripts.consumer_bill import line_recovery as lr
from scripts.consumer_bill import project_future_bill as bill
from scripts.consumer_bill.refresh_recovery_demand import refresh_document
from tests.unit.test_future_consumer_bill import fixture_inputs

RECOVERY_PATH = Path("data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json")
RECOVERY_2050 = Path("data/market/consumer_bill_forecast/2050_recovery_demand_assumptions.json")
ASSUMPTIONS_PATH = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
PO_SCRIPT = Path("scripts/public_ownership/project_public_ownership_consumer_bill.py")


def _recovery_2030():
    document = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    return bill.read_recovery_denominators(RECOVERY_PATH, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"])


# ── the interface ────────────────────────────────────────────────────────────────

def test_household_and_system_pounds_convert_through_the_entry_both_ways():
    entry = lr.make_entry("gsp_charging_volume_mwh", 315e6, household_loss_factor=1.1313,
                          household_allocation_factor=1.2, price_year_factor=1.0384, system_price_year=2024,
                          basis="test")
    household = lr.system_to_household_gbp(entry, 1.0e9)
    assert household == pytest.approx(1.0e9 * 1.0384 * 1.1313 * 1.2 / 315e6 * 3.1)
    assert lr.household_to_system_gbp(entry, household) == pytest.approx(1.0e9)
    assert lr.household_to_system_gbp(entry, household, consumption_kwh=2500.0) == pytest.approx(1.0e9 * 3100 / 2500)


def test_entry_for_reads_a_summary_or_its_block_and_names_a_missing_line():
    entry = lr.make_entry("metered_supply_mwh", 290e6, system_price_year=2024, basis="b")
    assert lr.entry_for({"line_recovery": {"duos": entry}}, "duos") is entry
    assert lr.entry_for({"duos": entry}, "duos") is entry
    with pytest.raises(KeyError, match="tnuos"):
        lr.entry_for({"line_recovery": {"duos": entry}}, "tnuos")


def test_an_inherited_entry_copies_the_line_it_adjusts():
    cfd = lr.make_entry("cfd_eligible_gsp_volume_mwh", 300e6, household_loss_factor=1.1313,
                        system_price_year=2024, basis="cfd")
    entry = lr.inherited_entry({"cfd": cfd}, "public_surplus_rebate")
    assert {k: entry[k] for k in lr.ENTRY_FIELDS if k != "basis"} == {k: cfd[k] for k in lr.ENTRY_FIELDS if k != "basis"}
    assert "inherited from cfd" in entry["basis"]
    with pytest.raises(KeyError, match="not declared"):
        lr.inherited_entry({"cfd": cfd}, "a_new_route_line")
    with pytest.raises(KeyError, match="does not carry"):
        lr.inherited_entry({}, "public_surplus_rebate")


def test_every_line_the_ownership_projector_writes_inherits_a_recovery():
    """Each component id the ownership script can emit, static or built, is declared."""
    from scripts.public_ownership import project_public_ownership_consumer_bill as po

    emitted = set(po.NEW_LINES) | set(po.DOMESTIC_LEVY_LINES.values()) | set(po.NETWORK_LINES) | set(po.BOOKED_ROUTE_SPEC)
    for name in re.findall(r'"component_id":\s*([A-Z_]+)\b', PO_SCRIPT.read_text(encoding="utf-8")):
        emitted.add(getattr(po, name))
    emitted.discard(po.LINE_LDES)            # the forecast bill's own line (item D4(c))
    assert emitted <= set(lr.INHERITED_LINE_RECOVERY), sorted(emitted - set(lr.INHERITED_LINE_RECOVERY))


# ── the recovery file ────────────────────────────────────────────────────────────

def test_the_2030_file_is_on_two_volumes_derived_from_its_own_stored_volumes():
    recovery = _recovery_2030()
    document = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    basis = document["metered_supply_basis"]
    load, draw = basis["gb_electrical_load_mwh"]["value"], basis["electrolyser_electricity_draw_mwh"]["value"]
    assert load == pytest.approx(315_281_271.880, abs=1e-3) and draw == pytest.approx(14_282_708.689, abs=1e-3)  # refreshed on the re-solved year (8 Nov 2026)
    share = recovery["cfd_eligible_share"]
    assert recovery["recovery_basis"] == bill.TWO_VOLUME_BASIS
    assert recovery["metered_supply_mwh"] == pytest.approx(load / 1.14 + draw)
    assert recovery["gsp_charging_volume_mwh"] == pytest.approx(load / 1.14 * 1.085 + draw)
    assert recovery["cfd_eligible_gsp_volume_mwh"] == pytest.approx(recovery["gsp_charging_volume_mwh"] * share)
    assert recovery["ro_eligible_metered_supply_mwh"] == pytest.approx(recovery["metered_supply_mwh"] * share)
    assert recovery["household_loss_factor"] == 1.1313
    assert recovery["metered_supply_mwh"] / 1e6 == pytest.approx(290.85, abs=0.01)
    assert recovery["gsp_charging_volume_mwh"] / 1e6 == pytest.approx(314.35, abs=0.01)
    # The deprecated aliases mean what they alias.
    assert recovery["final_demand_mwh"] == recovery["metered_supply_mwh"]
    assert recovery["gsp_chargeable_volume_mwh"] == recovery["gsp_charging_volume_mwh"]
    assert recovery["cfd_eligible_supplier_sales_mwh"] == recovery["ro_eligible_metered_supply_mwh"]
    assert "gsp_to_meter_loss_factor" not in recovery


@pytest.mark.parametrize("path, value", [
    (("final_demand_mwh", "value"), 300e6),
    (("metered_supply_basis", "gsp_chargeable_volume_mwh", "value"), 330e6),
    (("metered_supply_basis", "metered_supply_mwh", "value"), 300e6),
    (("cfd_eligible_supply", "cfd_eligible_gsp_volume_mwh", "value"), 300e6),
    (("ro_obligation_eligible_supply_mwh", "value"), 280e6),
    (("metered_supply_basis", "system_to_metered_factor", "value"), 1.2),
])
def test_a_two_volume_file_that_breaks_an_identity_is_refused(tmp_path, path, value):
    document = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    block = document
    for key in path[:-1]:
        block = block[key]
    block[path[-1]] = value
    target = tmp_path / "recovery.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        bill.read_recovery_denominators(target, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"])


def test_the_2050_file_stays_on_item_d1():
    document = json.loads(RECOVERY_2050.read_text(encoding="utf-8"))
    assert "recovery_basis" not in document
    recovery = bill.read_recovery_denominators(RECOVERY_2050, "FES25_HT_2050", document["system_demand_mwh"]["value"])
    assert recovery["recovery_basis"] == bill.D1_BASIS
    assert recovery["final_demand_mwh"] == pytest.approx(
        recovery["gsp_chargeable_volume_mwh"] / recovery["gsp_to_meter_loss_factor"])


def test_a_refresh_rederives_the_two_volume_file_and_leaves_a_d1_file_on_d1():
    document = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    refreshed = refresh_document(document, document["refresh"]["volumes_mwh"], "the same months")
    for key in ("metered_supply_mwh", "gsp_charging_volume_mwh", "gsp_chargeable_volume_mwh"):
        assert refreshed["metered_supply_basis"][key]["value"] == pytest.approx(document["metered_supply_basis"][key]["value"])
    assert refreshed["final_demand_mwh"]["value"] == pytest.approx(document["final_demand_mwh"]["value"])
    assert refreshed["ro_obligation_eligible_supply_mwh"]["value"] == pytest.approx(
        document["ro_obligation_eligible_supply_mwh"]["value"])
    assert set(refreshed["refresh"]["derived_mwh"]) >= {"metered_supply", "gsp_charging_volume",
                                                        "cfd_eligible_gsp_volume", "ro_eligible_metered_supply"}
    # Scaling the load moves both volumes on their declared factors.
    volumes = {**document["refresh"]["volumes_mwh"]}
    volumes["gb_electrical_load_mwh"] *= 1.1
    volumes["system_demand_mwh"] = volumes["gb_electrical_load_mwh"] + volumes["h2_offtake_mwh"]
    moved = refresh_document(document, volumes, "scaled")
    assert moved["metered_supply_basis"]["gsp_charging_volume_mwh"]["value"] == pytest.approx(
        volumes["gb_electrical_load_mwh"] / 1.14 * 1.085 + volumes["electrolyser_draw_mwh"])
    d1 = json.loads(RECOVERY_2050.read_text(encoding="utf-8"))
    again = refresh_document(d1, d1["refresh"]["volumes_mwh"], "the same months")
    assert again["final_demand_mwh"]["value"] == pytest.approx(d1["final_demand_mwh"]["value"])


def test_conversion_denominators_match_what_the_summary_reports():
    document = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    recovery = _recovery_2030()
    out = bill.conversion_denominators(document)
    assert out["final_demand_mwh"] == pytest.approx(recovery["metered_supply_mwh"])
    assert out["eligible_mwh"] == pytest.approx(recovery["cfd_eligible_gsp_volume_mwh"] / 1.1313)


# ── the projected bill ───────────────────────────────────────────────────────────

DERIVED = ("capacity_market", "bsuos_other_services", "bsuos_voltage_and_stability_constraints", "feed_in_tariffs",
           "remaining_ccs", "supplier_costs_and_margin")


def _two_volume_inputs():
    """The neutral fixture on the 2030 recovery file, with the derived lines and the supplier line live."""
    annual, assumptions, case, balancing, _, ro = fixture_inputs()
    original = json.loads(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
    for key in DERIVED:
        assumptions["sourced_externals"][key] = copy.deepcopy(original["sourced_externals"][key])
    assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["allowances"]["losses"][
        "loss_multiplier_gb_average"] = 1.1313
    assumptions["model_linked_rules"]["ldes_cap_and_floor"].update(
        enabled=True, recovery_denominator=original["model_linked_rules"]["ldes_cap_and_floor"]["recovery_denominator"])
    case["cfd_recovery"] = {"method": "eligible_supplier_sales"}
    balancing.update(constraint_cost_gbp=1.0e9, accepted_offer_cost_gbp=1.0e9)
    annual["cfd_payment_gbp"] = 5.0e9
    recovery = _recovery_2030()
    ro = {**ro, "gbp_per_3100kwh_ex_vat": 60.0, "obligation_eligible_supply_mwh": recovery["ro_eligible_metered_supply_mwh"]}
    ldes = {"topup_gbp": 1.0e9, "case_id": "private"}
    return annual, assumptions, case, balancing, recovery, ro, ldes


def _project(extra=None, **kw):
    annual, assumptions, case, balancing, recovery, ro, ldes = _two_volume_inputs()
    frame, summary = bill.project_components(annual, assumptions, case, balancing, recovery, ro,
                                             extra_components=extra, ldes=ldes, **kw)
    return frame.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"], summary, assumptions, recovery


def test_every_line_has_a_recovery_entry_on_a_volume_the_bill_carries():
    lines, summary, _, recovery = _project()
    entries = summary["line_recovery"]
    assert set(entries) == set(lines.index)
    for component_id, entry in entries.items():
        assert set(entry) == set(lr.ENTRY_FIELDS), component_id
        assert entry["volume_mwh"] == pytest.approx(recovery[entry["volume_key"]]), component_id
        assert entry["system_price_year"] == 2024
    expected = {
        "cfd": ("cfd_eligible_gsp_volume_mwh", 1.1313, 1.0), "allocated_ccs": ("cfd_eligible_gsp_volume_mwh", 1.1313, 1.0),
        "remaining_ccs": ("cfd_eligible_gsp_volume_mwh", 1.1313, 1.0),
        "bsuos_constraints": ("gsp_charging_volume_mwh", 1.1313, 1.0),
        "bsuos_other_services": ("gsp_charging_volume_mwh", 1.1313, 1.0),
        "bsuos_voltage_and_stability_constraints": ("gsp_charging_volume_mwh", 1.1313, 1.0),
        "ldes_cap_and_floor": ("gsp_charging_volume_mwh", 1.1313, 1.0),
        "aahedc": ("gsp_charging_volume_mwh", 1.1313, 1.0),
        "nuclear_contract_counterparty": ("gsp_charging_volume_mwh", 1.1313, 1.0),
        "capacity_market": ("gsp_charging_volume_mwh", 1.0, 1.348308),
        "renewables_obligation": ("ro_eligible_metered_supply_mwh", 1.0, 1.0),
        "feed_in_tariffs": ("ro_eligible_metered_supply_mwh", 1.0, 1.0),
        "battery_revenue_top_up": ("metered_supply_mwh", 1.0, 1.0),
        "duos": ("metered_supply_mwh", 1.0, 1.0), "tnuos": ("metered_supply_mwh", 1.0, 1.0),
        "supplier_costs_and_margin": ("metered_supply_mwh", 1.0, 1.0),
        "sizewell_c_rab": ("metered_supply_mwh", 1.0, 1.0),
    }
    for component_id, (key, loss, allocation) in expected.items():
        entry = entries[component_id]
        assert (entry["volume_key"], entry["household_loss_factor"]) == (key, loss), component_id
        assert entry["household_allocation_factor"] == pytest.approx(allocation), component_id
    assert summary["exchequer_per_household_key"]["volume_key"] == "metered_supply_mwh"
    assert summary["exchequer_per_household_key"]["volume_mwh"] == pytest.approx(recovery["metered_supply_mwh"])


def test_each_lines_system_cost_converts_back_to_its_household_figure():
    lines, summary, assumptions, _ = _project()
    entries = summary["line_recovery"]
    model_factor = float(assumptions["currency_rebasing"]["factor"])
    known = {"cfd": 5.0e9, "bsuos_constraints": 1.0e9, "ldes_cap_and_floor": 1.0e9}
    for component_id, system in known.items():
        assert lr.system_to_household_gbp(entries[component_id], system) == pytest.approx(lines[component_id])
    # A declared cost in its own price year: the entry converts the model-year equivalent.
    for component_id in ("capacity_market", "bsuos_other_services", "feed_in_tariffs", "remaining_ccs"):
        block = assumptions["sourced_externals"][component_id]
        resolved = bill.resolve_sourced_externals({component_id: block}, _recovery_2030(), 3100,
                                                  calendar_year=2030)[component_id]
        own = 1.0 if int(block["price_year"]) == 2025 else float(block["rebasing"]["factor"])
        cost = float(resolved["derivation"]["cost_gbp_used"]) * own / model_factor
        assert lr.system_to_household_gbp(entries[component_id], cost) == pytest.approx(lines[component_id]), component_id
        assert lr.household_to_system_gbp(entries[component_id], lines[component_id]) == pytest.approx(cost)


def test_gsp_rate_lines_carry_the_household_loss_factor_on_the_gsp_volume():
    lines, summary, _, recovery = _project()
    gsp, cfd_gsp = recovery["gsp_charging_volume_mwh"], recovery["cfd_eligible_gsp_volume_mwh"]
    assert lines["bsuos_constraints"] == pytest.approx(1.0e9 / gsp * 3.1 * 1.1313)
    assert lines["ldes_cap_and_floor"] == pytest.approx(1.0e9 / gsp * 3.1 * 1.1313)
    assert lines["cfd"] == pytest.approx(5.0e9 / cfd_gsp * 3.1 * 1.1313)
    # The deprecated summary denominators keep their conversions exact for old readers.
    assert summary["cfd_recovery_denominator_mwh"] == pytest.approx(cfd_gsp / 1.1313)
    assert summary["final_demand_recovery_denominator_mwh"] == pytest.approx(recovery["metered_supply_mwh"])


def test_lines_the_ownership_projector_adds_copy_the_entry_of_the_line_they_adjust():
    extra = [{"component_id": child, "stack_group": "policy", "label": child, "annual_gbp_per_3100kwh_ex_vat": -1.0,
              "method": "test", "source": "test"} for child in lr.INHERITED_LINE_RECOVERY]
    lines, summary, _, _ = _project(extra=extra)
    entries = summary["line_recovery"]
    assert set(entries) == set(lines.index)
    for child, parent in lr.INHERITED_LINE_RECOVERY.items():
        for field in lr.ENTRY_FIELDS:
            if field != "basis":
                assert entries[child][field] == entries[parent][field], (child, field)
    with pytest.raises(KeyError, match="INHERITED_LINE_RECOVERY"):
        _project(extra=[{"component_id": "public_surplus_rebate", "stack_group": "policy", "label": "x",
                         "annual_gbp_per_3100kwh_ex_vat": -1.0, "method": "m", "source": "s"},
                        {"component_id": "an_undeclared_line", "stack_group": "policy", "label": "x",
                         "annual_gbp_per_3100kwh_ex_vat": -1.0, "method": "m", "source": "s"}])


def test_every_ownership_case_summary_carries_a_recovery_entry_per_bill_line(monkeypatch):
    """The ownership projector's per-case summaries, on the 2030 recovery file, through the live projector."""
    from scripts.public_ownership import project_public_ownership_consumer_bill as po

    annual, assumptions, case, balancing, recovery, ro, ldes = _two_volume_inputs()

    def fake_run_projection(case_, assumptions_path, *args, extra_components=None, ldes_topup_gbp=None,
                            supplier_margin_base=None, **kwargs):
        frame, summary = bill.project_components(
            copy.deepcopy(annual), copy.deepcopy(assumptions), case_, balancing, recovery, ro,
            extra_components=extra_components, supplier_margin_base=supplier_margin_base,
            ldes={"topup_gbp": 1.0e9 if ldes_topup_gbp is None else ldes_topup_gbp})
        return None, frame, summary

    monkeypatch.setattr(po, "run_projection", fake_run_projection)
    monkeypatch.setattr(po, "load_bill_strike_table", lambda *a, **k: None)
    monkeypatch.setattr(po, "case_strike_table", lambda *a, **k: None)
    _, private_frame, _ = fake_run_projection(case, ASSUMPTIONS_PATH)
    cases = ["private", "public_ownership"]
    floor = {"ldes_cap_and_floor_enabled": True, "ldes_floor_topup_gbp": 1.0e9}
    account = {"cases": {"private": dict(floor), "public_ownership": {
        **floor, "recycling_rule": "rebate_gross", "routed_to_bill_gbp": 1.0e8}}}
    routes = pd.DataFrame(columns=["case_id", "route", "bill_allocated_gbp_2024", "saving_economic_life_gbp_2024",
                                   "treatment", "assets", "term_adjustment_gbp_2024", "identity_residual_gbp_2024"])
    by_case, report, _ = po._project_ownership_cases(
        case, cases, str(ASSUMPTIONS_PATH), [], [], [], [], None, None, str(RECOVERY_PATH), None, None,
        pd.DataFrame({"case_id": cases}), routes,
        pd.DataFrame({"case_id": cases, "annual_financing_saving_vs_private_gbp": [0.0, 1.0e8]}),
        private_frame, public_account=account)
    for case_id, entry in report["cases"].items():
        ids = set(by_case.loc[by_case["case_id"].eq(case_id), "component_id"])
        assert set(entry["line_recovery"]) == ids, case_id
    rebate = report["cases"]["public_ownership"]["line_recovery"][po.LINE_REBATE]
    assert rebate["volume_key"] == "cfd_eligible_gsp_volume_mwh" and rebate["household_loss_factor"] == 1.1313
    # The supplier line is the private case's in every case (its margin base is passed on).
    supplier = by_case.loc[by_case["component_id"].eq("supplier_costs_and_margin")].set_index("case_id")
    assert supplier.loc["public_ownership", "annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(
        supplier.loc["private", "annual_gbp_per_3100kwh_ex_vat"])
    # The rebate line carries exactly the account's amount through the CfD line's entry.
    line = float(by_case.loc[by_case["case_id"].eq("public_ownership")
                             & by_case["component_id"].eq(po.LINE_REBATE), "annual_gbp_per_3100kwh_ex_vat"].iloc[0])
    assert -lr.household_to_system_gbp(rebate, line) == pytest.approx(1.0e8)


def test_procurement_losses_must_be_the_household_loss_factor():
    annual, assumptions, case, balancing, recovery, ro, ldes = _two_volume_inputs()
    assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["allowances"]["losses"][
        "loss_multiplier_gb_average"] = 1.0919
    with pytest.raises(ValueError, match="household loss factor"):
        bill.project_components(annual, assumptions, case, balancing, recovery, ro, ldes=ldes)


def test_a_stale_ro_summary_is_refused_on_the_two_volume_basis():
    annual, assumptions, case, balancing, recovery, ro, ldes = _two_volume_inputs()
    ro["obligation_eligible_supply_mwh"] = 289_429_836.8
    with pytest.raises(ValueError, match="rebuild it"):
        bill.project_components(annual, assumptions, case, balancing, recovery, ro, ldes=ldes)


def test_a_gsp_volume_line_must_say_how_it_takes_household_losses():
    recovery = _recovery_2030()
    block = copy.deepcopy(json.loads(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))["sourced_externals"]["remaining_ccs"])
    block["derivation"].pop("household_losses")
    with pytest.raises(ValueError, match="household_losses"):
        bill.resolve_sourced_externals({"remaining_ccs": block}, recovery, 3100, calendar_year=2030)
    block["derivation"]["recovery_denominator"] = "cfd_eligible_supplier_sales_mwh"
    with pytest.raises(ValueError, match="deprecated alias"):
        bill.resolve_sourced_externals({"remaining_ccs": block}, recovery, 3100, calendar_year=2030)


# ── item 2: Ofgem's direct-fuel formula ──────────────────────────────────────────

def test_the_wholesale_lines_follow_ofgems_direct_fuel_formula():
    annual, assumptions, case, balancing, recovery, ro, ldes = _two_volume_inputs()
    bridge = assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]
    rates = {"transaction_cost": 0.004, "day_ahead_rehedging": 0.003, "imbalance": 0.013, "additional_risk_and_basis": 0.01}
    for key, rate in rates.items():
        bridge["allowances"][key]["rate"] = rate
    assert bridge["rate_composition"] == bill.PROCUREMENT_OFGEM_ANNEX2
    annual["domestic_profile_price_gbp_per_mwh"] = 50.0
    frame, summary = bill.project_components(annual, assumptions, case, balancing, recovery, ro, ldes=ldes)
    lines = frame.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    index = 50.0 * 3.1      # the fixture's currency factor is one; the model profile has no shaping rate
    for key, rate in rates.items():
        assert lines[f"procurement_{key}"] == pytest.approx(index * rate)
    assert lines["procurement_losses"] == pytest.approx(index * (1 + sum(rates.values())) * 0.1313)
    wholesale = lines[[c for c in lines.index if c.startswith("procurement_")] + ["direct_energy", "shaping_adjustment"]]
    assert wholesale.sum() == pytest.approx(index * (1 + sum(rates.values())) * 1.1313)
    assert frame.set_index("component_id").loc["procurement_additional_risk_and_basis", "component"] == \
        "Additional risk allowance"
    # Without the declaration the allowances stay on the shaped energy cost (the 2050 bill).
    bridge.pop("rate_composition")
    frame, _ = bill.project_components(annual, assumptions, case, balancing, recovery, ro, ldes=ldes)
    legacy = frame.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    assert legacy["procurement_losses"] == pytest.approx(index * 0.1313)


# ── items 3 and 4: EBIT and headroom on 2030 costs; levelisation ─────────────────

def test_ebit_and_headroom_are_ofgems_percentages_of_the_bills_own_costs():
    lines, summary, assumptions, _ = _project()
    margin = summary["supplier_margin"]
    rule = assumptions["sourced_externals"]["supplier_costs_and_margin"]["derivation"]["ebit_and_headroom"]
    factor = margin["external_factor"]
    levelisation = 3.4819013671345997 * factor
    networks = {"tnuos", "duos", "bsuos_constraints", "bsuos_other_services", "bsuos_voltage_and_stability_constraints",
                "ldes_cap_and_floor", "interconnector_cap_and_floor_pass_through"}
    costs = lines.sum() - margin["ebit_gbp"] - margin["headroom_gbp"] - levelisation
    assert margin["base"]["costs_gbp"] == pytest.approx(costs)
    assert margin["base"]["costs_excluding_networks_gbp"] == pytest.approx(costs - lines[list(networks)].sum())
    ebit = 0.014446 * costs + 22.4611 / 2 * factor
    assert margin["ebit_gbp"] == pytest.approx(ebit)
    assert margin["headroom_gbp"] == pytest.approx(0.014641 * (margin["base"]["costs_excluding_networks_gbp"] + ebit))
    held = (201.8492 - 27.37852 - 10.95044) * factor
    assert lines["supplier_costs_and_margin"] == pytest.approx(held + levelisation + margin["ebit_gbp"]
                                                             + margin["headroom_gbp"])
    assert rule["network_group"] == {"view": "ofgem", "group": "Network costs"}


def test_a_supplied_margin_base_holds_the_supplier_line_across_cases():
    base = {"costs_gbp": 900.0, "costs_excluding_networks_gbp": 600.0, "source": "private"}
    lines, summary, _, _ = _project(supplier_margin_base=base)
    factor = summary["supplier_margin"]["external_factor"]
    ebit = 0.014446 * 900.0 + 22.4611 / 2 * factor
    assert summary["supplier_margin"]["ebit_gbp"] == pytest.approx(ebit)
    assert summary["supplier_margin"]["headroom_gbp"] == pytest.approx(0.014641 * (600.0 + ebit))
    # Another line moving does not move the supplier line.
    more = [{"component_id": "public_surplus_rebate", "stack_group": "policy", "label": "x",
             "annual_gbp_per_3100kwh_ex_vat": -50.0, "method": "m", "source": "s"}]
    again, _, _, _ = _project(extra=more, supplier_margin_base=base)
    assert again["supplier_costs_and_margin"] == pytest.approx(lines["supplier_costs_and_margin"])


def test_the_direct_debit_household_pays_levelisation():
    assumptions = json.loads(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
    block = assumptions["sourced_externals"]["supplier_costs_and_margin"]
    assert "direct-debit household" in assumptions["benchmark"]["payment_method_scope"]
    assert "levelisation" in assumptions["benchmark"]["payment_method_scope"]
    assert block["period_basis"] == "held_flat_real"
    factor = float(block["rebasing"]["factor"])
    value, basis, margin = bill.supplier_line_parts("supplier_costs_and_margin", block, factor, 0.0, "")
    assert margin["levelisation_gbp"] == pytest.approx(3.4819 * factor, abs=1e-4)
    assert margin["levelisation_gbp"] == pytest.approx(3.39, abs=0.01)
    assert value == pytest.approx((201.8492 - 27.37852 - 10.95044 + 3.4819013671345997) * factor)
    assert "levelisation" in basis


# ── item 5: the BICS sensitivity's scope; the levy exemptions on two volumes ─────

def test_the_bics_sensitivity_scales_the_ro_fit_and_capacity_market_only():
    lines, summary, assumptions, _ = _project()
    sensitivity = summary["diagnostics"]["levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat"]
    share = assumptions["levy_exemptions"]["sensitivities"]["bics_2027"]["additional_exempt_share"]
    expected = (lines["renewables_obligation"] + lines["feed_in_tariffs"] + lines["capacity_market"]) * (1 / (1 - share) - 1)
    assert sensitivity["levy_exemption_bics_2027"] == pytest.approx(expected)
    assert set(bill.ELIGIBLE_BASE_LINES) == {"renewables_obligation", "feed_in_tariffs"}


def test_exempt_electrolysers_leave_both_eligible_volumes_and_the_capacity_market_volume():
    recovery = _recovery_2030()
    out, record = bill.apply_levy_exemptions(recovery, {"electrolysers_exempt": True})
    draw, share = recovery["electrolyser_draw_mwh"], recovery["cfd_eligible_share"]
    assert out["cfd_eligible_gsp_volume_mwh"] == pytest.approx(recovery["cfd_eligible_gsp_volume_mwh"] - draw * share)
    assert out["ro_eligible_metered_supply_mwh"] == pytest.approx(recovery["ro_eligible_metered_supply_mwh"] - draw * share)
    assert out["gsp_charging_volume_mwh"] == recovery["gsp_charging_volume_mwh"]   # BSUoS and LDES stay
    assert record["capacity_market_volume_after_mwh"] == pytest.approx(recovery["gsp_charging_volume_mwh"] - draw)
    frame = pd.DataFrame({"component_id": ["cfd", "renewables_obligation", "capacity_market", "bsuos_constraints"],
                          "annual_gbp_per_3100kwh_ex_vat": [100.0, 50.0, 30.0, 40.0]})
    out = bill.levy_and_cm_sensitivities(frame, {"levy_exemptions": {"sensitivities": {"e": {"electrolysers_exempt": True}}}},
                                         recovery, 3100)
    expected = (100.0 * (record["cfd_eligible_scale"] - 1) + 50.0 * (record["ro_eligible_scale"] - 1)
                + 30.0 * (recovery["gsp_charging_volume_mwh"] / record["capacity_market_volume_after_mwh"] - 1))
    assert out["levy_exemption_e"] == pytest.approx(expected)


def test_bics_moves_the_ro_base_and_not_the_cfd_base():
    recovery = _recovery_2030()
    out, record = bill.apply_levy_exemptions(recovery, {"additional_exempt_share": 0.05})
    assert out["cfd_eligible_gsp_volume_mwh"] == recovery["cfd_eligible_gsp_volume_mwh"]
    assert out["ro_eligible_metered_supply_mwh"] == pytest.approx(recovery["ro_eligible_metered_supply_mwh"] * 0.95)
    assert record["eligible_scale"] == pytest.approx(1 / 0.95)


def test_a_central_capacity_market_exemption_is_refused_on_two_volumes():
    annual, assumptions, case, balancing, recovery, ro, ldes = _two_volume_inputs()
    assumptions["levy_exemptions"].update(additional_exempt_share=0.05, additional_exemption_covers_capacity_market=True)
    with pytest.raises(ValueError, match="named sensitivity"):
        bill.project_components(annual, assumptions, case, balancing, recovery, ro, ldes=ldes)
