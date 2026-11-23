"""The bill chapter's generated tables and values (scripts/analysis/build_bill_docs_values.py, 6 Oct 2026)."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis import build_bill_docs_values as docs
from scripts.consumer_bill import bill_groupings

GSP_MWH = 330e6
FACTOR = 1.35
CM_COST = 2.5e9
REBASE = 1.04


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp: Path) -> dict[str, Path]:
    """A small, self-consistent set of bill outputs."""
    paths = {}
    # hourly prices: two hours a quarter, baseload 10, 20, 30, 40
    snaps = pd.to_datetime(["2030-01-07 08:00", "2030-01-07 20:00", "2030-04-08 08:00", "2030-04-08 20:00",
                            "2030-07-08 08:00", "2030-07-08 20:00", "2030-10-07 08:00", "2030-10-07 20:00"])
    prices = [5.0, 15.0, 0.0, 40.0, 30.0, 30.0, 0.5, 79.5]
    paths["hourly"] = tmp / "hourly.csv"
    pd.DataFrame({"snapshot": snaps, "hours": 1.0, "price_gbp_per_mwh": prices}).to_csv(paths["hourly"], index=False)

    paths["recovery"] = _write_json(tmp / "recovery.json", {"case_id": "fixture"})
    paths["balancing"] = _write_json(tmp / "balancing.json", {"annual": {
        "model_price_year": 2024, "accepted_offer_cost_gbp": 1.0e9, "accepted_bid_cost_gbp": 0.5e9,
        "feasibility_slack_cost_in_accepted_costs_gbp": 1.0e6, "non_participant_turn_down_cost_at_penalty_gbp": 2.0e6,
        "interconnector_retrading_cost_gbp": 0.3e9, "include_interconnector_retrading": True,
        "constraint_cost_gbp": 1.797e9, "constraint_cost_excluding_retrading_gbp": 1.497e9,
        "cfd_curtailment_compensation_gbp": 0.2e9, "cfd_curtailed_mwh": 3.0e6}})

    roc_value, share = 80.0, 0.99
    fleet = pd.DataFrame({
        "carrier": ["wind_offshore", "wind_onshore", "solar_pv", "biomass", "marine"],
        "eligible_capacity_mw": [5000.0, 8000.0, 6000.0, 1000.0, 20.0],
        "estimated_generation_mwh": [20e6, 18e6, 6e6, 4e6, 30e3],
        "roc_volume": [40e6, 18e6, 7.2e6, 6e6, 0.15e6],
        "applied_load_factor": [0.45, 0.25, 0.12, 0.39, 0.17],
        "pypsa_load_factor": [0.45, 0.25, 0.12, 0.6, 0.17],
    })
    paths["ro_fleet"] = tmp / "ro_fleet.csv"
    fleet.to_csv(paths["ro_fleet"], index=False)
    ro_cost = fleet["roc_volume"].sum() * roc_value * share
    ro_line = ro_cost / 290e6 * 3.1
    paths["ro_summary"] = _write_json(tmp / "ro.json", {
        "register_total_capacity_mw": 42000.0, "expired_capacity_mw": 19000.0, "inactive_capacity_mw": 1500.0,
        "unaccredited_part_of_year_capacity_mw": 400.0, "eligible_capacity_mw": 20020.0, "eligible_stations": 24000,
        "roc_volume_by_jurisdiction": {"NI": {"capacity_mw": 1300.0}},
        "notional_roc_value": {"notional_roc_value_gbp_per_roc_report_prices": roc_value, "base_obligation_year": "2020/21",
                               "base_buyout_price_gbp_per_roc": 50.05, "base_recycle_value_gbp_per_roc": 4.42},
        "uk_obligation_split": {"gb_share_of_uk_obligation": share},
        "obligation_eligible_supply_mwh": 290e6, "gross_scheme_cost_gbp": ro_cost,
        "uk_roc_volume": fleet["roc_volume"].sum(), "gross_gbp_per_3100kwh_ex_vat": ro_line,
        "relief_cases": {"relief_ends_march_2029": {"gbp_per_3100kwh_ex_vat": ro_line},
                         "relief_extended": {"gbp_per_3100kwh_ex_vat": ro_line / 4}}})

    paths["cm_calendar"] = _write_json(tmp / "cm_calendar.json", {
        "price_year": 2025, "calendar_2030_cost_gbp": CM_COST,
        "delivery_years": {
            "2029/2030": {"contract_book_gbp": 1.9e9, "t1_gbp": 0.03e9, "total_gbp": 1.93e9, "weight_in_calendar_2030": 0.5},
            "2030/2031": {"contract_book_gbp": 0.9e9, "t4_modelled_gbp": 2.11e9, "t1_gbp": 0.06e9, "total_gbp": 3.07e9,
                          "weight_in_calendar_2030": 0.5}},
        "calendar_2030_cost_by_t4_case_gbp": {"central": CM_COST, "ldes_excluded": 2.9e9},
        "household_peak_factor": {"factor": FACTOR, "ofgem_allowance_gbp_per_3100kwh_nominal": 20.0,
                                  "flat_share_gbp_per_3100kwh": 20.0 / FACTOR}})
    paths["cm_auction"] = _write_json(tmp / "cm_auction.json", {
        "checks": {"T-4 2029/30": {"modelled_price": 26.0, "actual_price": 27.1},
                   "T-4 2028/29": {"modelled_price": 60.03, "actual_price": 60.0}},
        "t4_2030_31": {"central": {"price": 50.0, "entered_total_mw": 43000.0,
                                   "entered_mw": {"existing": 30000.0, "dsr": 3000.0, "interconnector": 10000.0}},
                       "ldes_excluded": {"price": 75.0, "shortfall": True, "entered_total_mw": 39000.0,
                                         "entered_mw": {"existing": 29000.0, "dsr": 3000.0, "interconnector": 7000.0}}}})
    paths["public_account"] = _write_json(tmp / "public_account.json", {"cases": {"private": {
        "ldes_floor_topup_gbp": 1.2e9, "ldes_floor_topup_at_wacc_gbp": 1.8e9, "ldes_floor_real_rate": 0.045,
        "ldes_capacity_market_revenue_netted_gbp": 70e6, "battery_revenue_top_up_gbp": 330e6}}})

    lines = {
        "direct_energy": 150.0, "shaping_adjustment": 6.0, "procurement_losses": 14.0, "procurement_imbalance": 2.0,
        "cfd": 150.0, "capacity_market": CM_COST * FACTOR / GSP_MWH * 3.1, "renewables_obligation": ro_line,
        "supplier_costs_and_margin": 196.45, "smart_meter_net_cost_change": -1.5, "duos": 136.0,
    }
    methods = {k: "" for k in lines}
    methods["supplier_costs_and_margin"] = ("Sum of allowances at 3,100 kWh from the nil and 2,500 kWh benchmark values "
                                            "(standing plus unit): cost to serve 110.43; debt related costs 31.58; "
                                            "industry charges 21.51; ebit 27.38; headroom 10.95 = GBP201.85 "
                                            "(Oct-Dec 2026 money), held flat in real terms.")
    methods["smart_meter_net_cost_change"] = "GB average SMNCC at 3,100 kWh = GBP-1.54. Negative since July 2025."
    paths["components"] = tmp / "components.csv"
    pd.DataFrame({"component_id": list(lines), "component": list(lines), docs.VALUE: list(lines.values()),
                  "method": [methods[k] for k in lines]}).to_csv(paths["components"], index=False)
    ex_vat = sum(lines.values())

    by_strike = [
        {"carrier": "wind_offshore", "strike_key": "AR1_offshore", "strike_price": 160.0, "dispatch_mwh": 4e6, "payment_gbp": 0.6e9},
        {"carrier": "wind_offshore", "strike_key": "AR8_modelled_offshore", "strike_price": 90.0, "dispatch_mwh": 40e6, "payment_gbp": 2.4e9},
        {"carrier": "wind_onshore", "strike_key": "AR5_onshore", "strike_price": 70.0, "dispatch_mwh": 5e6, "payment_gbp": 0.1e9},
        {"carrier": "nuclear", "strike_key": "HPC_nuclear", "strike_price": 125.0, "dispatch_mwh": 18e6, "payment_gbp": 1.4e9},
    ]
    payment = sum(r["payment_gbp"] for r in by_strike)
    quarters = [{"quarter": q, "base_price_gbp_per_mwh": base, "peak_price_gbp_per_mwh": base + 2,
                 "domestic_volume_weight": 0.25} for q, base in zip((1, 2, 3, 4), (10.0, 20.0, 30.0, 40.0))]
    summary = {
        "ex_vat_gbp": ex_vat, "vat_gbp": 0.05 * ex_vat, "total_gbp_inc_vat": 1.05 * ex_vat, "vat_rate": 0.05,
        "model_rebasing_factor": REBASE, "reported_period_hours": 8.0, "wholesale_price_gbp_per_mwh": 24.5,
        "minimum_price_gbp_per_mwh": 0.0, "hours_price_below_minus_0_01_gbp_per_mwh": 0.0,
        "hours_price_at_or_below_1_gbp_per_mwh": 3.0, "cfd_payment_gbp": payment, "cfd_recovery_denominator_mwh": 290e6,
        "cfd_strike_provenance": {"by_strike": by_strike},
        "recovery_denominators": {
            "sha256": _sha(paths["recovery"]), "gsp_chargeable_volume_mwh": GSP_MWH, "metered_supply_mwh": 302e6,
            "gb_electrical_load_mwh": 315e6, "electrolyser_draw_mwh": 15e6, "cfd_eligible_supplier_sales_mwh": 290e6,
            "h2_offtake_mwh": 10e6, "gsp_to_meter_loss_factor": GSP_MWH / 302e6, "cfd_eligible_share": 290 / 302},
        "balancing_costs": {"sha256": _sha(paths["balancing"])},
        "ro_scheme_cost": {"sha256": _sha(paths["ro_summary"]), "gross_gbp_per_3100kwh_ex_vat": ro_line},
        "ldes_cap_and_floor": {"sha256": _sha(paths["public_account"])},
        "levy_exemptions": {"sensitivities": {"bics_2027": {"additional_exempt_share": 0.055}}},
        "diagnostics": {
            "cfd_line_gbp_per_3100kwh": payment * REBASE / 290e6 * 3.1,
            "levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat": {"duos_low": -10.0, "duos_high": 9.0},
            "wholesale_procurement": {
                "shaping": {"blend_gbp_per_mwh": 25.6, "rate": 0.04, "quarters": quarters},
                "sensitivities_gbp_per_3100kwh_ex_vat": {"shaping": {"low_1pct": -5.0},
                                                         "forward_premium": {"plus_5pct": 7.5}},
                "scarcity_markup": {"available": True, "load_weighted_uplift_gbp_per_mwh": 1.6, "hours_scarce": 1349}}},
    }
    paths["summary"] = _write_json(tmp / "summary.json", summary)
    paths["line_sources"] = docs.DEFAULT_INPUTS["line_sources"]
    paths["groupings"] = bill_groupings.GROUPINGS_PATH
    paths["grouping_views"] = bill_groupings.VIEWS_PATH
    paths["cm_supply"] = tmp / "absent_supply.csv"
    return paths


def test_generator_writes_every_declared_file_with_the_run_figures(tmp_path):
    paths = _fixture(tmp_path)
    out = tmp_path / "_generated"
    written = docs.build(paths, out, run_date=date(2026, 10, 6))
    assert sorted(p.name for p in written) == sorted(docs.GENERATED_FILES)

    values = json.loads((out / "values.json").read_text(encoding="utf-8"))
    v = values["values"]
    assert values["_meta"]["bill_run_date"] == "2026-10-06"
    assert v["run_date"] == "6 October 2026"
    assert v["cfd_payment_bn"] == "4.50" and v["cfd_twh"] == "67.0"
    assert v["line_smart_meter_net_cost_change_gbp"] == "-£1.50"
    assert v["cm_t4_central_price"] == "50.00" and v["cm_household_factor"] == "1.3500"
    assert v["ro_lf_biomass"] == "0.39" and v["ro_lf_biomass_model"] == "0.600"
    assert v["supplier_ofgem_total"] == "201.85" and v["smncc_cap_money"] == "-£1.54"
    assert v["peak_block_hours"] == "4"  # the four 08:00 weekday hours
    assert v["cfd_oldest_offshore_payment_pct"] == "13"  # AR1 offshore: 0.6 of 4.5bn

    lines = (out / "bill_lines.md").read_text(encoding="utf-8")
    assert lines.startswith("% GENERATED")
    assert "% Bill run of 2026-10-06." in lines            # provenance, in a hidden comment
    assert "*From `components.csv`" in lines               # what readers see: the files, no run date
    assert "*Bill run of" not in lines
    assert f"| **{docs.num(sum(pd.read_csv(paths['components'])[docs.VALUE]))}** |" in lines

    cases = (out / "cm_cases.md").read_text(encoding="utf-8")
    assert "| **Central** | **50.00** | **£2.50bn** |" in cases
    assert "| New LDES cannot enter | 75.00 (cap, short) | £2.90bn |" in cases

    quarters = (out / "wholesale_quarters.md").read_text(encoding="utf-8")
    assert "```{list-table} Quarterly prices, 2030 (GBP/MWh" in quarters
    assert "  - 25.0%" in quarters

    supplier = (out / "supplier_components.md").read_text(encoding="utf-8")
    assert "| Core operating cost (cost to serve) | 110.43 |" in supplier
    assert "| **Total** | **201.85** |" in supplier

    build_up = (out / "network_constraint_build_up.md").read_text(encoding="utf-8")
    assert "| **Constraint cost** | **1,797.0** |" in build_up


def test_a_missing_input_writes_nothing(tmp_path):
    paths = _fixture(tmp_path)
    paths["balancing"].unlink()
    out = tmp_path / "_generated"
    with pytest.raises(docs.DocsInputError, match="missing bill-docs inputs"):
        docs.build(paths, out)
    assert not out.exists()


def test_an_input_from_another_run_is_refused(tmp_path):
    paths = _fixture(tmp_path)
    _write_json(paths["recovery"], {"case_id": "edited after the bill ran"})
    with pytest.raises(docs.DocsInputError, match="not the file the bill read"):
        docs.build(paths, tmp_path / "_generated")


def test_a_capacity_market_rebuilt_after_the_bill_is_refused(tmp_path):
    paths = _fixture(tmp_path)
    calendar = json.loads(paths["cm_calendar"].read_text(encoding="utf-8"))
    calendar["calendar_2030_cost_gbp"] = 2.4e9
    _write_json(paths["cm_calendar"], calendar)
    with pytest.raises(docs.DocsInputError, match="newer than the bill"):
        docs.build(paths, tmp_path / "_generated")


def test_the_f18_supplier_line_and_applied_quarter_weights(tmp_path):
    """8 Oct 2026: the first F18 bill. EBIT and headroom come from the summary's supplier_margin, the
    allowances from the method's new wording; the quarters carry the applied weight beside the volume."""
    paths = _fixture(tmp_path)
    components = pd.read_csv(paths["components"])
    factor, levelisation, ebit, headroom = 0.97, 3.39, 25.10, 10.16
    allowances = 110.43 + 31.58 + 21.51
    line = allowances * factor + levelisation + ebit + headroom
    supplier = components["component_id"].eq("supplier_costs_and_margin")
    delta = line - float(components.loc[supplier, docs.VALUE].iloc[0])
    components.loc[supplier, docs.VALUE] = line
    components.loc[supplier, "method"] = (
        "Sum of allowances at 3,100 kWh from the nil and 2,500 kWh benchmark values (standing plus unit): cost to "
        "serve 110.43; debt related costs 31.58; industry charges 21.51 (Oct-Dec 2026 money), held flat in real "
        "terms. EBIT 1.4446% x GBP980.80 + GBP10.93 = GBP25.10; headroom 1.4641% x (GBP668.79 excluding "
        "networks + EBIT) = GBP10.16.")
    components.to_csv(paths["components"], index=False)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["supplier_margin"] = {
        "rule": {"ebit_variable_share": 0.014446, "headroom_share": 0.014641},
        "external_factor": factor, "levelisation_gbp": levelisation, "ebit_gbp": ebit, "headroom_gbp": headroom,
        "ebit_fixed_gbp": 10.93, "cap_period_ebit_gbp": 26.65, "cap_period_headroom_gbp": 10.66,
        "change_against_cap_period_cash_gbp": ebit + headroom - 26.65 - 10.66,
        "base": {"costs_gbp": 980.80, "costs_excluding_networks_gbp": 668.79}}
    summary["ex_vat_gbp"] += delta
    summary["vat_gbp"], summary["total_gbp_inc_vat"] = 0.05 * summary["ex_vat_gbp"], 1.05 * summary["ex_vat_gbp"]
    for q in summary["diagnostics"]["wholesale_procurement"]["shaping"]["quarters"]:
        q.update(quarter_weight=0.25, quarter_weight_basis="declared_quarter_weights")
    _write_json(paths["summary"], summary)

    out = tmp_path / "_generated"
    docs.build(paths, out, run_date=date(2026, 10, 8))
    v = json.loads((out / "values.json").read_text(encoding="utf-8"))["values"]
    assert v["supplier_ebit"] == "25.10" and v["supplier_to_report_money"] == "0.9700"
    table = (out / "supplier_components.md").read_text(encoding="utf-8")
    assert "| Core operating cost (cost to serve) | 110.43 | 107.12 |" in table
    assert f"| **Total** |  | **{docs.num(line)}** |" in table
    assert v["quarter_weight_basis"] == "Ofgem PC1 seasonal weights"
