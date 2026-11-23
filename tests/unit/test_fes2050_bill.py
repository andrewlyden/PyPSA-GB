"""Task Y13: the 2050 bill inputs, the heat-pump plus EV household and the 2050 ledger switches.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill.build_ro_eligible_fleet import RO_SCHEME_END, build_ro_scheme_cost, closed_scheme_cost
from scripts.consumer_bill.household_archetypes import (
    archetype_composition, archetype_consumption, project_archetypes,
)
from scripts.consumer_bill.project_future_bill import (
    PERIOD_BASES, check_period_basis, project_components, read_recovery_denominators, read_ro_scheme_cost,
    tnuos_household_share,
)
from tests.helpers import requires_outputs

A2050 = Path("data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json")
A2030 = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
R2050 = Path("data/market/consumer_bill_forecast/2050_recovery_demand_assumptions.json")
RO2050 = Path("data/market/consumer_bill_forecast/2050_ro_scheme_parameters.json")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@requires_outputs("resources/FES/FES_2025_data.csv")
def test_the_committed_2050_files_are_the_builders_output():
    from scripts.consumer_bill.build_2050_bill_inputs import OUTPUTS, build, keep_committed_refresh, render

    documents = build()
    # The refresh (scripts/consumer_bill/refresh_recovery_demand.py) replaces the pre-run
    # volumes with the solved months' own; everything else is the builder's.
    documents["recovery"] = keep_committed_refresh(documents["recovery"])
    for key, path in OUTPUTS.items():
        assert Path(path).read_text(encoding="utf-8") == render(documents[key]), path


# ── the external lines ────────────────────────────────────────────────────────

def test_every_2030_external_line_has_a_2050_value_and_basis():
    a50, a30 = load(A2050), load(A2030)
    assert a50["benchmark"]["bill_calendar_year"] == 2050
    # FES 2050 decision 58: the life-extension top-up is a 2050-only line (no 2030 counterpart). The
    # battery top-up is in both years since 3 Oct 2026 (user decision).
    assert set(a50["sourced_externals"]) - {"life_extension_operating_top_up"} == set(
        a30["sourced_externals"])
    for key, block in a50["sourced_externals"].items():
        assert check_period_basis(key, block, 2050) in PERIOD_BASES
        assert block.get("source_url"), key
        derived = "recovery_denominator" in (block.get("derivation") or {})
        assert derived or "value" in block, key
        if block.get("value_basis") == "nil":
            cost = (block.get("derivation") or {}).get("agreement_cost_gbp") if derived else block["value"]
            assert cost == 0.0 and block["method"], key
        # No 2030 placeholder is carried as a 2050 reference.
        assert block.get("reference_gbp_per_3100kwh_ex_vat") is None, key


def test_nil_lines_are_the_closed_schemes():
    se = load(A2050)["sourced_externals"]
    nil = {k for k, b in se.items() if b.get("value_basis") == "nil"}
    # The three closed schemes (the Capacity Market clears above zero once V2G bids in: decision 30).
    assert nil == {"feed_in_tariffs", "eco", "remaining_ccs"}
    assert "early 2040s" in se["feed_in_tariffs"]["method"]


def test_extrapolated_declared_needs_its_rule():
    block = {"period_basis": "extrapolated_declared", "extrapolation": {"from_period": "FY2030/31"}}
    with pytest.raises(ValueError, match="extrapolation.rule"):
        check_period_basis("x", block, 2050)
    block["extrapolation"]["rule"] = "held"
    assert check_period_basis("x", block, 2050) == "extrapolated_declared"


def _fy_drivers(derivation):
    d = derivation["drivers"]
    return {y: {"peak": d["peak_gw"][y] / d["peak_gw"]["2030"],
                "customers": d["domestic_customers"][y] / d["domestic_customers"]["2030"],
                "demand": (d["consumer_demand_excluding_electrolysis_gwh"][y]
                           / d["consumer_demand_excluding_electrolysis_gwh"]["2030"])} for y in ("2049", "2050")}


@requires_outputs("resources/analysis/network_price_control_building_blocks.json")
def test_duos_and_tnuos_extrapolations_recompute_from_their_drivers():
    se = load(A2050)["sourced_externals"]
    du, tn = se["duos"]["derivation"], se["tnuos"]["derivation"]
    weights = {"2049": 0.25, "2050": 0.75}
    r = _fy_drivers(du)
    f = du["fixed_share_f"]
    # User, 3 Oct 2026: the revenue drivers are Y12's roll-forward (transmission building blocks,
    # distribution RAV), pinned in the builder and kept equal to the network account below.
    from scripts.consumer_bill.build_2050_bill_inputs import y12_driver

    fy = {"2049": 2049, "2050": 2050}
    duos = sum(w * du["base_fy2030_31_gbp_2026_money"]
               * (f * y12_driver("distribution", fy[y]) / r[y]["customers"]
                  + (1 - f) * y12_driver("distribution", fy[y]) / r[y]["demand"])
               for y, w in weights.items())
    assert se["duos"]["value"] == pytest.approx(duos, rel=1e-12)
    base = tn["base_fy2030_31"]
    tnuos = sum(w * (base["domestic_residual_gbp_per_year"] * y12_driver("transmission", fy[y]) / r[y]["customers"]
                     + base["locational_gbp_per_year"]) for y, w in weights.items())
    assert se["tnuos"]["value"] == pytest.approx(tnuos, rel=1e-12)
    # The interconnector credit's household share reads the 2050 blend.
    share = tnuos_household_share(load(A2050))
    blend = tn["calendar_blend"]
    assert share == pytest.approx(blend["domestic_residual_gbp_per_year"] / (blend["demand_residual_revenue_gbp_m"] * 1e6))
    assert "demand_residual_revenue_gbp_m_2050_51" in tn


@requires_outputs("resources/market/capacity_market/cm_calendar_2050_summary.json")
def test_capacity_market_is_the_2050_auction_model():
    """Task E2E and decision 30: the CM line is Y11's 2050 auction with V2G bidding in as DSR.

    User, 14 Nov 2026: nothing is pinned. The bill builder reads the auction model's output file, which must be
    the model's current result, and the committed bill file must carry it (the build_capacity_market_2050 rule
    rewrites the output; the bill inputs are then rebuilt).
    """
    from scripts.consumer_bill.build_2050_bill_inputs import cm_2050
    from scripts.market.build_capacity_market_2050 import UNPAID_CASE, build

    CM_2050 = cm_2050()
    result = build()["summary"]
    cases = result["calendar_2050_cost_by_case_gbp"]
    assert result["settings"]["v2g"] == "in_auction"
    assert CM_2050["central_gbp"] == pytest.approx(result["calendar_2050_cost_gbp"], abs=1.0)
    assert CM_2050["central_gbp"] > 0.0
    assert CM_2050["low_case"] == UNPAID_CASE and cases[UNPAID_CASE] == 0.0
    assert CM_2050["low_gbp"] == pytest.approx(cases[CM_2050["low_case"]], abs=1.0)
    assert CM_2050["high_gbp"] == pytest.approx(cases[CM_2050["high_case"]], abs=1.0)
    assert not result["cases"][CM_2050["high_case"]]["shortfall"]
    assert CM_2050["target_mw"] == pytest.approx(result["auction"]["target_mw"])
    assert CM_2050["supply_mw"] == pytest.approx(result["auction"]["supply_total_mw"])
    assert CM_2050["price_gbp_per_kw_year_2025"] == pytest.approx(result["price_gbp_per_kw_year_2025"], abs=0.005)
    payees = result["payments_by_payee_gbp"]
    for key, value in CM_2050["payments_by_payee_gbp"].items():
        assert value == pytest.approx(payees[key], abs=1.0)
    assert sum(CM_2050["payments_by_payee_gbp"].values()) == pytest.approx(CM_2050["central_gbp"], abs=1.0)
    line = load(A2050)["sourced_externals"]["capacity_market"]
    cm = line["derivation"]
    assert cm["agreement_cost_gbp"] == pytest.approx(CM_2050["central_gbp"], abs=1.0)
    assert "value_basis" not in line and "nil_reason" not in cm
    assert cm["builder_output"].endswith("cm_calendar_2050_summary.json")
    assert set(cm["sensitivity_cost_gbp"]) == {"clearing_low", "clearing_high"}


def test_capacity_market_2050_unpaid_case_is_y11s_old_central():
    """Decision 30: the 'V2G netted out, unpaid' sensitivity reproduces Y11's central exactly."""
    from scripts.market import capacity_market as cm_module
    from scripts.market import capacity_market_2050 as cm50
    from scripts.market.build_capacity_market import auction_model
    from scripts.market.build_capacity_market_2050 import UNPAID_CASE, build

    old = cm50.run_2050(auction_model(cm_module.read_register()), cm_module.read_cpi())
    new = build(v2g="outside_cm")
    assert new["summary"]["calendar_2050_cost_gbp"] == old["summary"]["calendar_2050_cost_gbp"]
    pd.testing.assert_frame_equal(new["revenue"], old["revenue"])
    assert new["summary"]["cases"][UNPAID_CASE]["calendar_2050_cost_gbp"] == old["summary"]["calendar_2050_cost_gbp"]


def test_a_zero_sourced_line_needs_its_reason():
    from scripts.consumer_bill.project_future_bill import resolve_sourced_externals

    block = {"derivation": {"agreement_cost_gbp": 0.0, "recovery_denominator": "final_demand_mwh"}}
    recovery = {"final_demand_mwh": 1.0e8}
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_sourced_externals({"capacity_market": block}, recovery, 3100)
    block["value_basis"] = "nil"
    assert resolve_sourced_externals({"capacity_market": block}, recovery, 3100)["capacity_market"]["value"] == 0.0


def test_interim_lines_name_their_task():
    se = load(A2050)["sourced_externals"]
    interim = {k: b["interim"]["until"] for k, b in se.items() if b.get("interim")}
    # Decision 41 (c): Sizewell C's line reads the register's RAB allowance; TNUoS and DUoS follow
    # Y12's network roll-forward (user, 3 Oct 2026): nothing is interim.
    assert interim == {}
    assert se["sizewell_c_rab"]["derivation"]["cost_from"]["key"] == "rab_support_gbp"


# ── the household ─────────────────────────────────────────────────────────────

def test_heat_pump_ev_household_is_composed_from_traceable_inputs():
    block = load(A2050)["household_archetypes"]
    kwh = archetype_consumption(block)
    parts = {p["part"]: p for p in archetype_composition(block)["heat_pump_ev_2050"]}
    hp, ev = parts["heat_pump"]["inputs"], parts["electric_vehicle"]["inputs"]
    assert hp["seasonal_performance_factor"] == 2.8 and ev["kwh_per_mile"] == 0.3
    # FES F.05's own method: saving = gain x 2050 demand.
    assert hp["seasonal_performance_gain"] == pytest.approx(22.381269177718828 / 108.224)
    assert ev["efficiency_gain"] == pytest.approx(15.768564668512923 / 126.1125)
    base = (50293.628 + 2090.715) * 1e6 / 34886840.449
    expected = base + 9500 * 0.85 / (2.8 * (1 + hp["seasonal_performance_gain"])) + 9400 * 0.3 / (1 + ev["efficiency_gain"])
    assert kwh["heat_pump_ev_2050"] == pytest.approx(expected)
    assert kwh["heat_pump_ev_2050"] == pytest.approx(6398, abs=1)
    assert kwh["heat_pump_ev_2050_fes_per_unit"] == pytest.approx(5594, abs=1)
    assert sum(p["kwh"] for p in parts.values()) == pytest.approx(kwh["heat_pump_ev_2050"])


def test_the_2030_single_addition_form_is_unchanged():
    block = load(A2030)["household_archetypes"]
    kwh = archetype_consumption(block)
    assert kwh["heat_pump"] == 2500 + 9500 * 0.85 / 2.8
    assert kwh["electric_vehicle"] == 2500 + 9400 * 0.30


def test_composition_is_reported_only_when_asked():
    components = pd.DataFrame({"component_id": ["a", "b"], "annual_gbp_per_3100kwh_ex_vat": [100.0, 200.0]})
    block = {"benchmark_kwh": 3100, "archetypes": [{"id": "base", "kwh": 1500},
                                                   {"id": "hp", "base": "base", "added": [{"part": "heat_pump", "inputs": {"annual_kwh": 2000}}]}]}
    assumptions = {"household_archetypes": block, "benchmark": {}, "charge_basis": {"standing_share": {"a": 1.0}}}
    assert "composition" not in project_archetypes(components, assumptions, vat_rate=0.05)
    block["report_composition"] = True
    out = project_archetypes(components, assumptions, vat_rate=0.05)
    assert out["composition"]["hp"][-1] == {"part": "heat_pump", "kwh": 2000.0, "inputs": {"annual_kwh": 2000.0}}


def test_a_listed_part_must_be_named_once():
    block = {"archetypes": [{"id": "b", "kwh": 1}, {"id": "x", "base": "b", "added": [
        {"part": "heat_pump", "inputs": {"annual_kwh": 1}}, {"part": "heat_pump", "inputs": {"annual_kwh": 2}}]}]}
    with pytest.raises(ValueError, match="named twice"):
        archetype_consumption(block)


# ── the recovery file and the RO ──────────────────────────────────────────────

def test_recovery_file_closes_its_identities():
    doc = load(R2050)
    assert doc["forecast_year"] == 2050
    rec = read_recovery_denominators(R2050, "FES25_HT_2050", doc["system_demand_mwh"]["value"])
    assert rec["final_demand_mwh"] == pytest.approx((rec["gb_electrical_load_mwh"] + rec["electrolyser_draw_mwh"])
                                                   / rec["gsp_to_meter_loss_factor"])
    # Pre-run FES volumes: ED1 system excluding electrolysis + DACCS, networked electrolysis.
    fes_load, fes_draw = (615323.5 + 17828.38) * 1e3, 88775.0e3
    if "volumes_mwh" in (doc.get("refresh") or {}):
        # Refreshed from the solved year (2 Oct 2026): the run's own volumes, with the flexible
        # demand units counted, land within 1% (load) and 10% (electrolysis) of FES.
        vols = doc["refresh"]["volumes_mwh"]
        assert rec["gb_electrical_load_mwh"] == pytest.approx(vols["gb_electrical_load_mwh"])
        assert rec["gb_electrical_load_mwh"] == pytest.approx(fes_load, rel=0.01)
        assert rec["electrolyser_draw_mwh"] == pytest.approx(fes_draw, rel=0.10)
    else:
        assert rec["gb_electrical_load_mwh"] == pytest.approx(fes_load)
        assert rec["electrolyser_draw_mwh"] == pytest.approx(fes_draw)


def _register(tmp_path, effective="01/04/2016"):
    path = tmp_path / "register.csv"
    pd.DataFrame({"Accreditation reference": ["R1"], "Station name": ["S"], "Station DNC in kW": ["1,000"],
                  "Technology": ["Onshore wind"], "Effective date": [effective]}).to_csv(path, index=False)
    return path


def test_closed_ro_scheme_is_a_declared_nil(tmp_path):
    params, recovery = load(RO2050), load(R2050)
    by_carrier, summary = build_ro_scheme_cost(_register(tmp_path), [], params, recovery, None, None)
    assert by_carrier.empty and summary["method_label"] == "scheme_closed"
    assert summary["gross_gbp_per_3100kwh_ex_vat"] == 0.0 and summary["expired_capacity_mw"] == 1.0
    path = tmp_path / "ro.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    ro = read_ro_scheme_cost(path, "scheme_closed", 2025)
    assert ro["gbp_per_3100kwh_ex_vat"] == 0.0 and ro["relief_cases"] == {"scheme_closed": 0.0}
    assert pd.Timestamp(params["scheme_closed"]["closed_on"]) == RO_SCHEME_END


def test_closed_ro_scheme_needs_the_declaration(tmp_path):
    params = {k: v for k, v in load(RO2050).items() if k != "scheme_closed"}
    with pytest.raises(ValueError, match="scheme_closed"):
        closed_scheme_cost(_register(tmp_path), params, load(R2050), 2050)


def test_a_positive_summary_cannot_claim_the_closed_label(tmp_path):
    summary = {"relief_cases": {"scheme_closed": {"exchequer_relief_share": 0.0, "gbp_per_3100kwh_ex_vat": 5.0}},
               "notional_roc_value": {"report_price_year": 2025}, "method_label": "scheme_closed",
               "gross_gbp_per_3100kwh_ex_vat": 5.0}
    path = tmp_path / "ro.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="nil gross"):
        read_ro_scheme_cost(path, "scheme_closed", 2025)


# ── the projector on the 2050 files ───────────────────────────────────────────

def _project_2050(tmp_path):
    assumptions = load(A2050)
    # Decision 58: the top-up line reads the public account's private top-up, which the register writes once it
    # is rebuilt with the support lines; point both register-booked lines at a copy that carries it.
    account_path = Path(assumptions["sourced_externals"]["sizewell_c_rab"]["derivation"]["cost_from"]["public_account"])
    account = json.loads(account_path.read_text(encoding="utf-8"))
    account["cases"]["private"].setdefault("life_extension_top_up_gbp", 1.2e9)
    account["cases"]["private"].setdefault("battery_revenue_top_up_gbp", 0.37e9)
    account["cases"]["private"].setdefault("battery_revenue_top_up_subhourly_high_gbp", 0.31e9)
    copy = tmp_path / "public_account_summary.json"
    copy.write_text(json.dumps(account), encoding="utf-8")
    for line in ("sizewell_c_rab", "life_extension_operating_top_up", "battery_revenue_top_up"):
        assumptions["sourced_externals"][line]["derivation"]["cost_from"]["public_account"] = str(copy)
    assumptions["model_linked_rules"]["direct_fuel"]["shaping"]["method"] = "model_profile"
    doc = load(R2050)
    recovery = read_recovery_denominators(R2050, "FES25_HT_2050", doc["system_demand_mwh"]["value"])
    _, summary = closed_scheme_cost(_register(tmp_path), load(RO2050), doc, 2050)
    path = tmp_path / "ro.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    ro = read_ro_scheme_cost(path, "scheme_closed", 2025)
    case = {"case_id": "FES25_HT_2050", "modelled_year": 2050, "model_price_year": 2024,
            "cfd_recovery": {"method": "eligible_supplier_sales"}}
    annual = {"modelled_year": 2050, "reported_period_hours": 8760, "demand_mwh": doc["system_demand_mwh"]["value"],
              "wholesale_price_gbp_per_mwh": 50.0, "domestic_profile_price_gbp_per_mwh": 55.0,
              "domestic_shaping_gbp_per_mwh": 5.0,
              "domestic_profile": {"peak_share_of_consumption": .55, "peak_share_of_hours": .36},
              "cfd_payment_gbp": 1e9, "cfd_negative_price_rule": "gb_contract_terms", "gross_redispatch_mwh": 0.0,
              "interconnector_congestion_rent_by_cable_gbp": {}}
    balancing = {"constraint_cost_gbp": 1e9, "accepted_offer_cost_gbp": 1e9, "accepted_bid_cost_gbp": 0.0,
                 "cfd_curtailment_compensation_gbp": 0.0, "cfd_curtailed_mwh": 0.0, "feasibility_slack_cost_gbp": 0.0,
                 "feasibility_slack_energy_mwh": 0.0, "path": "test-balancing.json"}
    ldes = {"case_id": "private", "enabled": True, "topup_gbp": 0.0, "source": "test"}
    # Task SETTLE2050: the 2050 bill's DPA line reads the support settlement's total.
    dpa = {"net_support_gbp": 7e9, "path": "test-support.json",
           "summary": {"dpa": {"availability_payment_gbp": 6e9, "variable_payment_gbp": 2e9, "gainshare_gbp": 1e9}}}
    return project_components(annual, assumptions, case, balancing, recovery, ro, ldes=ldes, dpa=dpa)


@requires_outputs("resources/public_ownership/fes_ht_2050_public_account_summary.json")
def test_the_projector_accepts_the_2050_files(tmp_path):
    rows, summary = _project_2050(tmp_path)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    for key in ("feed_in_tariffs", "eco", "remaining_ccs", "renewables_obligation"):
        assert value[key] == 0.0
    # SETTLE2050: the DPA line is the settlement's total on the CfD levy base, like the CfD line.
    assert value["dpa"] == pytest.approx(value["cfd"] * 7.0)
    assert summary["bill_calendar_year"] == 2050
    assert not summary.get("interim_external_lines")          # written only when a line is interim
    assert summary["external_period_bases"]["duos"] == "extrapolated_declared"
    arche = summary["diagnostics"]["household_archetypes"]
    bills = {b["archetype"]: b for b in arche["bills"]}
    assert bills["benchmark"]["ex_vat_gbp"] == pytest.approx(summary["ex_vat_gbp"], rel=1e-9)
    assert bills["heat_pump_ev_2050"]["annual_consumption_kwh"] == pytest.approx(6397.86, abs=0.01)
    # User, 13 Nov 2026: the headline is the electrified household on today's base (Ofgem medium 2,500 kWh), with the
    # high-demand home (Ofgem high, heat pump on 14,000 kWh of gas) beside it; the FES-based households are kept.
    assert arche["headline_archetype"] == "heat_pump_ev_2050_today_base"
    assert bills["heat_pump_ev_2050_today_base"]["annual_consumption_kwh"] == pytest.approx(7396.31, abs=0.01)
    assert bills["heat_pump_ev_2050_high_demand"]["annual_consumption_kwh"] == pytest.approx(9828.28, abs=0.01)
    assert set(arche["composition"]["heat_pump_ev_2050_fes_per_unit"][i]["part"] for i in range(3)) == {
        "base", "heat_pump", "electric_vehicle"}
    # The 2050 CM range and the electrolyser exemption reach the sensitivities the ledger reads.
    levy = summary["diagnostics"].get("levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat") or {}
    assert {"capacity_market_clearing_low", "capacity_market_clearing_high",
            "levy_exemption_electrolysers_2050"} <= set(levy)


# ── the ledger ────────────────────────────────────────────────────────────────

def test_ledger_uncertainties_are_year_keyed():
    from scripts.public_ownership import exchequer_ledger as el

    diagnostics = {"wholesale_procurement": {"sensitivities_gbp_per_3100kwh_ex_vat": {
                       "forward_premium": {"minus_3pct": -1.0, "plus_5pct": 2.0, "stress_plus_10pct": 4.0},
                       "shaping": {"low_1pct": -0.5, "high_7pct": 0.5}}},
                   "levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat": {
                       "capacity_market_t4_2030_31_at_2029_30_clearing_27_10": -3.0, "levy_exemption_bics_2027": 1.0,
                       "capacity_market_clearing_low": -6.0, "capacity_market_clearing_high": 7.0,
                       "levy_exemption_electrolysers_2050": 2.5},
                   "ro_relief_case_values_gbp_per_3100kwh": {"relief_extended": 10.0, "relief_ends_march_2029": 40.0}}
    v2030 = el.uncertainty_values({"modelled_year": 2030, "diagnostics": diagnostics})
    assert v2030["ro_exchequer_relief_continued"] == -30.0 and v2030["bics_levy_exemption_2027"] == 1.0
    assert "capacity_market_2050_clearing_low" not in v2030
    v2050 = el.uncertainty_values({"modelled_year": 2050, "diagnostics": diagnostics})
    assert set(v2050) == set(el.UNCERTAINTIES_2050)
    assert v2050["capacity_market_2050_clearing_high"] == 7.0 and v2050["levy_exemption_electrolysers_2050"] == 2.5
    assert "ro_exchequer_relief_continued" in el.uncertainty_list(2050)[2]
    rows = el.uncertainties_by_case([{"case_id": "private"}], {"modelled_year": 2050, "diagnostics": diagnostics}, None, 0.05)
    assert rows["private"]["levy_exemption_electrolysers_2050"]["private_bill_change_gbp_inc_vat"] == pytest.approx(2.625)


def test_hpc_schedule_switch_is_off_for_2050_only():
    from scripts.public_ownership.exchequer_ledger import _hpc_schedule_for_year, hpc_schedule_switch

    assert hpc_schedule_switch(load(A2030)) == (True, None)
    enabled, reason = hpc_schedule_switch(load(A2050))
    assert enabled is False and "2050" in reason
    assert _hpc_schedule_for_year(load(A2050), "hpc.json", None, 2050) == {"skipped": reason}
    with pytest.raises(ValueError, match="without a reason"):
        hpc_schedule_switch({"exchequer_ledger": {"hpc_schedule_sensitivity": {"enabled": False}}})


# ── cost-to-serve ─────────────────────────────────────────────────────────────

def test_composite_cost_to_serve_prices_each_part_on_its_shape():
    from scripts.consumer_bill.archetype_cost_to_serve import (
        composite_cost_to_serve, heat_pump_parameters, main_composite,
    )

    index = pd.date_range("2050-01-01", periods=48, freq="h")
    price = pd.Series(np.where(index.hour >= 17, 100.0, 20.0), index=index)
    composition = archetype_composition(load(A2050)["household_archetypes"])
    hp = heat_pump_parameters({"x": composition["heat_pump_ev_2050"]})
    assert hp["spf"] == pytest.approx(2.8 * (1 + 22.381269177718828 / 108.224))
    shapes = {"heat_pump": pd.Series(1.0 / 48, index=index),
              "ev_evening": pd.Series(np.where(index.hour >= 17, 1.0, 0.0), index=index),
              "ev_overnight": pd.Series(np.where(index.hour < 5, 1.0, 0.0), index=index)}
    shapes = {k: v / v.sum() for k, v in shapes.items()}
    out = composite_cost_to_serve(price, {"h": composition["heat_pump_ev_2050"]}, shapes, 40.0,
                                  {"evening": "ev_evening", "overnight": "ev_overnight"}).set_index("ev_charging")
    parts = {p["part"]: p["kwh"] for p in composition["heat_pump_ev_2050"]}
    hp_price = float((price * shapes["heat_pump"]).sum())
    expected = (parts["base"] * 40.0 + parts["heat_pump"] * hp_price + parts["electric_vehicle"] * 100.0) / sum(parts.values())
    assert out.loc["evening", "household_profile_price_gbp_per_mwh"] == pytest.approx(expected)
    assert out.loc["overnight", "electric_vehicle_price_gbp_per_mwh"] == pytest.approx(20.0)
    assert out.loc["evening", "premium_gbp_per_household"] == pytest.approx(
        (expected - 40.0) * sum(parts.values()) / 1000.0)
    summary = {"domestic_profile_price_gbp_per_mwh": 40.0, "diagnostics": {"household_archetypes": {
        "bills": [{"archetype": "heat_pump_ev_2050", "ex_vat_gbp": 1.0}], "composition": composition,
        "headline_archetype": "heat_pump_ev_2050"}}}
    result = main_composite(price, summary, pd.Series(5.0, index=index), ["heat_pump_ev_2050"])
    assert len(result["composite_results"]) == 2 and result["headline_archetype"] == "heat_pump_ev_2050"
