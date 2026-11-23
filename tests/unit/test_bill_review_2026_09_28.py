"""Consumer-bill review fixes, 28 Sep 2026.

Calendar-2030 period bases for the external lines, the levy-exemption switch,
the counter-trade premium in the nationalised tariff's link trade, the replace
flag, the non-money balancing terms in the settlement, the at-cost rent label,
the state-bids Exchequer account and the consumer-plus-Exchequer gain.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill import project_future_bill as bill

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")


# ── calendar 2030 ───────────────────────────────────────────────────────────────

def test_calendar_2030_takes_three_months_of_fy2029_30_and_nine_of_fy2030_31():
    components = [{"period": "FY2029/30"}, {"period": "FY2030/31"}]
    assert bill.blend_weights(components, 2030) == [0.25, 0.75]
    with pytest.raises(ValueError, match="does not cover calendar 2030"):
        bill.blend_weights([{"period": "FY2030/31"}, {"period": "FY2031/32"}], 2030)
    with pytest.raises(ValueError, match="FY2029/30"):
        bill._fiscal_year_start("FY2029/31")


def test_every_external_line_in_the_repository_file_states_its_period():
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    assert assumptions["benchmark"]["bill_calendar_year"] == 2030
    assert assumptions["benchmark"]["period"] == "calendar 2030"
    for key, block in assumptions["sourced_externals"].items():
        assert bill.check_period_basis(key, block, 2030) in bill.PERIOD_BASES, key
    bases = {k: b["period_basis"] for k, b in assumptions["sourced_externals"].items()}
    assert bases["tnuos"] == bases["warm_home_discount"] == bases["feed_in_tariffs"] == "calendar_blend"
    assert bases["capacity_market"] == "calendar_year"
    assert bases["remaining_ccs"] == "period_average"


def test_a_line_without_a_period_basis_is_refused():
    with pytest.raises(ValueError, match="period_basis must be one of"):
        bill.check_period_basis("x", {"unit": "gbp_per_household_year"}, 2030)
    with pytest.raises(ValueError, match="source_period"):
        bill.check_period_basis("x", {"period_basis": "held_flat_real"}, 2030)
    with pytest.raises(ValueError, match="must contain calendar 2030"):
        bill.check_period_basis("x", {"period_basis": "period_average",
                                      "period_span": {"first_fiscal_year_start": 2030,
                                                      "last_fiscal_year_start": 2031}}, 2030)


def test_the_tnuos_blend_deflates_each_fiscal_year_at_its_own_midpoint():
    block = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]["tnuos"]
    value, text = bill.blended_household_value("tnuos", block, 2030, 2025)
    fy29, fy30 = block["period_blend"]
    expected = (0.25 * fy29["value"] * fy29["rebasing"]["factor"]
                + 0.75 * fy30["value"] * fy30["rebasing"]["factor"])
    assert value == pytest.approx(expected)
    # NESO FYV 2027/28-2031/32 v1.0: 0.356025 GBP/site/day and 0.462346 p/kWh in 2029/30.
    assert fy29["value"] == pytest.approx(0.356025 * 365 + 0.00462346 * 630.744, rel=1e-6)
    # Calendar 2030 sits between the two fiscal years in real terms.
    assert fy29["value"] * fy29["rebasing"]["factor"] < value < fy30["value"] * fy30["rebasing"]["factor"] + 1.0
    assert "FY2029/30" in text and "x 0.25" in text


def test_the_fit_blend_uses_both_desnz_years():
    sourced = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]
    # F18: the FiT is levied on eligible metered supply.
    out = bill.resolve_sourced_externals({"feed_in_tariffs": sourced["feed_in_tariffs"]},
                                         {"ro_eligible_metered_supply_mwh": 280e6}, 3100, calendar_year=2030)
    assert out["feed_in_tariffs"]["value"] == pytest.approx((0.25 * 1.8e9 + 0.75 * 1.7e9) / 280e6 * 3.1)
    with pytest.raises(ValueError, match="bill_calendar_year"):
        bill.resolve_sourced_externals({"feed_in_tariffs": sourced["feed_in_tariffs"]},
                                       {"ro_eligible_metered_supply_mwh": 280e6}, 3100)


# ── levy exemptions ─────────────────────────────────────────────────────────────

RECOVERY = {"final_demand_mwh": 300e6, "cfd_eligible_supplier_sales_mwh": 285e6, "cfd_eligible_share": 0.95,
            "gsp_chargeable_volume_mwh": 327.6e6, "gsp_to_meter_loss_factor": 1.092,
            "electrolyser_draw_mwh": 15e6}


def test_the_levy_exemption_defaults_leave_the_bill_untouched():
    out, record = bill.apply_levy_exemptions(RECOVERY, None)
    assert out is RECOVERY and record["eligible_scale"] == 1.0 and not record["applied"]
    ro = {"gbp_per_3100kwh_ex_vat": 50.0, "gross_gbp_per_3100kwh_ex_vat": 50.0}
    assert bill.scale_ro_for_exemptions(ro, record) is ro


def test_exempt_electrolysers_move_their_draw_onto_everyone_else():
    out, record = bill.apply_levy_exemptions(RECOVERY, {"electrolysers_exempt": True})
    assert out["cfd_eligible_supplier_sales_mwh"] == pytest.approx(285e6 - 15e6 / 1.092 * 0.95)
    assert out["gsp_chargeable_volume_mwh"] == pytest.approx(327.6e6 - 15e6)
    assert out["final_demand_mwh"] == RECOVERY["final_demand_mwh"]  # BSUoS and LDES stay on final demand
    ro = bill.scale_ro_for_exemptions({"gbp_per_3100kwh_ex_vat": 50.0, "gross_gbp_per_3100kwh_ex_vat": 50.0}, record)
    assert ro["gbp_per_3100kwh_ex_vat"] == pytest.approx(50.0 * 285e6 / out["cfd_eligible_supplier_sales_mwh"])


def test_a_further_eii_exemption_can_reach_the_capacity_market():
    out, _ = bill.apply_levy_exemptions(RECOVERY, {"additional_exempt_share": 0.05})
    assert out["cfd_eligible_supplier_sales_mwh"] == pytest.approx(285e6 * 0.95)
    assert out["gsp_chargeable_volume_mwh"] == RECOVERY["gsp_chargeable_volume_mwh"]
    out, _ = bill.apply_levy_exemptions(RECOVERY, {"additional_exempt_share": 0.05,
                                                   "additional_exemption_covers_capacity_market": True})
    assert out["gsp_chargeable_volume_mwh"] == pytest.approx(327.6e6 * 0.95)
    with pytest.raises(ValueError, match="unknown keys"):
        bill.apply_levy_exemptions(RECOVERY, {"electrolyser_exempt": True})


# ── the counter-trade premium in the nationalised tariff ───────────────────────

def test_link_trade_adds_the_counter_trade_premium_against_the_stage1_schedule(tmp_path, monkeypatch):
    from scripts.public_ownership import nationalised_institutions as ni

    monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
    index = pd.date_range("2030-01-01", periods=3, freq="h")
    pd.DataFrame({"IFA": [100.0, 100.0, 100.0]}, index=index).to_csv(tmp_path / "M_wholesale_links.csv")
    flow = pd.DataFrame({"IFA": [100.0, 40.0, 130.0]}, index=index)
    links = pd.DataFrame({"market_counter_trade_premium": [100.0]}, index=["IFA"])
    premium = ni.counter_trade_premium("M", flow, links)
    assert premium["IFA"].tolist() == [0.0, 6000.0, 3000.0]
    # A network solved without a premium values at zero and needs no schedule.
    none = ni.counter_trade_premium("absent", flow, pd.DataFrame(index=["IFA"]))
    assert float(none.to_numpy().sum()) == 0.0


def test_the_replace_flag_follows_the_charge_not_the_amount():
    from scripts.public_ownership.nationalised_institutions import replace_flag_from_settlement

    enabled = pd.DataFrame({"YEAR": ["renewable_capacity, NPV reading", 0.0]},
                           index=["forecast_error_charge_basis", "central_realtime_correction_replaced_gbp_m"])
    assert replace_flag_from_settlement(enabled, "YEAR") is True   # replaced amount is exactly zero
    disabled = enabled.copy()
    disabled.loc["forecast_error_charge_basis", "YEAR"] = "disabled"
    assert replace_flag_from_settlement(disabled, "YEAR") is False
    legacy = pd.DataFrame({"YEAR": [-31.8]}, index=["central_realtime_correction_replaced_gbp_m"])
    assert replace_flag_from_settlement(legacy, "YEAR") is True
    assert replace_flag_from_settlement(pd.DataFrame({"YEAR": []}), "YEAR", default=None) is None


# ── the settlement and the compensation check ──────────────────────────────────

def test_penalty_turn_down_and_tie_steps_are_not_balancing_revenue(tmp_path):
    from scripts.public_ownership.settlement import non_money_balancing_gbp

    redispatch = pd.DataFrame({
        "type": ["generator", "generator", "generator"],
        "carrier": ["wind_offshore", "embedded_solar", "wind_onshore"],
        "decrease_MWh": [100.0, 10.0, 50.0],
        "bid_cost": [5000.0, 60000.0, 1000.0],
    }, index=pd.Index(["A", "B", "C"], name="component"))
    audit = tmp_path / "audit.csv"
    pd.DataFrame({"generator": ["A", "C"], "pricing": ["cfd_contract_value"] * 2,
                  "tie_break_step_gbp_per_mwh": [0.5, 0.0]}).to_csv(audit, index=False)
    out = non_money_balancing_gbp(redispatch, audit)
    assert out["A"] == pytest.approx(50.0)       # 0.5 x 100 MWh of tie step
    assert out["B"] == pytest.approx(60000.0)    # 6,000/MWh penalty turn-down, all of it
    assert out["C"] == 0.0


def test_the_compensation_check_covers_only_contract_value_priced_units(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import contract_value_units

    audit = tmp_path / "audit.csv"
    pd.DataFrame({"generator": ["wind1", "wind2", "merch"],
                  "pricing": ["cfd_contract_value", "cfd_contract_value", "merchant_short_run_cost"]}).to_csv(
        audit, index=False)
    assert contract_value_units(audit) == {"wind1", "wind2"}
    assert contract_value_units(tmp_path / "missing.csv") is None


# ── transfers labelled ─────────────────────────────────────────────────────────

def test_at_cost_rent_is_split_out_of_the_term_adjustment():
    from scripts.public_ownership.cfd_repricing import ROUTE_CFD_BILL, route_summary

    savings = pd.DataFrame({
        "case_id": ["p", "p"], "route": [ROUTE_CFD_BILL, ROUTE_CFD_BILL], "asset_id": ["a", "b"],
        "saving_economic_life_gbp": [100.0, 50.0], "saving_financing_term_gbp": [90.0, 45.0],
        "saving_bill_term_gbp": [130.0, 45.0], "at_cost_applied": [True, False],
    })
    financing = pd.DataFrame({"case_id": ["p"], "annual_financing_saving_vs_private_gbp": [150.0]})
    row = route_summary(savings, financing).set_index("route").loc[ROUTE_CFD_BILL]
    assert row["term_adjustment_gbp_2024"] == pytest.approx(175.0 - 150.0)
    assert row["at_cost_rent_transfer_gbp_2024"] == pytest.approx(40.0)   # 130 - 90 on the at-cost row
    assert row["term_adjustment_excluding_rent_gbp_2024"] == pytest.approx(25.0 - 40.0)
    assert row["identity_kind"] == "row_partition"


def test_consumer_and_exchequer_gain_sits_beside_the_resource_saving():
    from scripts.public_ownership.exchequer_ledger import household_to_system, ledger_row, per_household

    conv = {"factor": 1.0384, "final_demand_mwh": 295.8e6, "mwh": 3.1, "vat_rate": 0.05, "domestic_mwh": 97.5e6}
    assert household_to_system(per_household(1.0e9, conv), conv) == pytest.approx(1.0e9)
    base = {"carbon_receipts_gbp": 0.0, "corporation_tax_forgone_gbp": 0.0, "network_tax_forgone_gbp": 0.0,
            "state_deficits_funded_gbp": 0.0, "state_surplus_retained_gbp": 0.0, "exchequer_residual_gbp": 0.0,
            "financing_saving_gbp": 0.0, "financing_saving_tax_transfer_gbp": 0.0}
    private = {**base, "bill_ex_vat": 900.0, "bill_inc_vat": 945.0}
    case = {**base, "bill_ex_vat": 880.0, "bill_inc_vat": 924.0, "financing_saving_gbp": 2.0e9,
            "financing_saving_tax_transfer_gbp": 0.4e9, "corporation_tax_forgone_gbp": 0.4e9}
    row = ledger_row(case, private, conv)
    consumer = -household_to_system(924.0 - 945.0, conv)
    assert row["consumer_gain_vs_private_gbp"] == pytest.approx(consumer)
    assert row["consumer_and_exchequer_gain_vs_private_gbp"] == pytest.approx(
        consumer + row["exchequer_net_vs_private_gbp"])
    assert row["gain_beyond_resource_saving_gbp"] == pytest.approx(
        row["consumer_and_exchequer_gain_vs_private_gbp"] - 1.6e9)


def test_a_state_bids_case_reads_its_own_run_groups_account(tmp_path):
    from scripts.public_ownership.exchequer_ledger import balancing_group_accounts

    market = tmp_path / "cp30_2030_cp30_network_public_account_summary.json"
    market.write_text(json.dumps({"cases": {}}), encoding="utf-8")
    own = tmp_path / "cp30_2030_cp30_network_public_bids_p2_public_account_summary.json"
    own.write_text(json.dumps({"cases": {"x@state_bids_at_cost": {"marker": 1}}}), encoding="utf-8")
    balancing = {"cases": {"x@state_bids_at_cost": {"run_group": "cp30_2030_cp30_network_public_bids_p2"}}}
    accounts = balancing_group_accounts(balancing, market)
    assert accounts["x@state_bids_at_cost"]["cases"]["x@state_bids_at_cost"]["marker"] == 1
    with pytest.raises(FileNotFoundError, match="public account"):
        balancing_group_accounts({"cases": {"y": {"run_group": "nope"}}}, market)


def test_the_strike_basis_premium_is_what_the_strike_bid_pays_above_contract_value(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import compute_cfd_curtailment_compensation
    from scripts.market.cfd_strike_table import StrikeTable
    from tests.unit.test_future_consumer_bill import _toy_networks

    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0}, 2024, {}, {})
    wholesale, balancing = _toy_networks(
        tmp_path, [[80.0, 50.0], [60.0, 50.0]], [[30.0, 50.0], [60.0, 50.0]], snapshots)
    prices = pd.Series([20.0, 40.0], index=snapshots)
    strike, _ = compute_cfd_curtailment_compensation(wholesale, balancing, prices, table, bid_basis="strike")
    value, _ = compute_cfd_curtailment_compensation(wholesale, balancing, prices, table)
    s, v = strike.set_index("generator").loc["curtailed"], value.set_index("generator").loc["curtailed"]
    # 50 MWh turned down at a reference of 20: paid the 100 strike under the strike
    # basis, the 80 it forgoes under contract value; the 20 GBP/MWh gap is the premium.
    assert s["compensation_gbp"] == pytest.approx(5000.0)
    assert v["compensation_gbp"] == pytest.approx(4000.0)
    assert s["strike_basis_premium_over_contract_value_gbp"] == pytest.approx(1000.0)
    assert v["strike_basis_premium_over_contract_value_gbp"] == 0.0


# ── wholesale: the Ofgem cap rule for shaping, forward-premium sensitivities ──

def test_base_and_peak_prices_follow_the_elexon_peak_block_per_quarter():
    index = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")
    peak = bill.peak_mask(index).to_numpy()
    hourly = pd.DataFrame({"price_gbp_per_mwh": np.where(peak, 60.0, 30.0), "hours": 1.0,
                           "domestic_weight": 1.0 / len(index)}, index=index)
    rows = bill.base_peak_prices_by_quarter(hourly)
    assert [r["quarter"] for r in rows] == [1, 2, 3, 4]
    for row in rows:
        assert row["peak_price_gbp_per_mwh"] == pytest.approx(60.0)
        assert 30.0 < row["base_price_gbp_per_mwh"] < 60.0
    price, detail = bill.ofgem_cap_rule_price(rows, 0.7, 0.3)
    assert detail[0]["blend_gbp_per_mwh"] == pytest.approx(0.7 * rows[0]["base_price_gbp_per_mwh"] + 0.3 * 60.0)
    with pytest.raises(ValueError, match="sum to one"):
        bill.ofgem_cap_rule_price(rows, 0.7, 0.4)


def _cap_rule_fixture():
    from tests.unit.test_future_consumer_bill import fixture_inputs

    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["model_linked_rules"]["direct_fuel"]["shaping"]["method"] = "ofgem_cap_rule"
    annual["base_peak_prices_by_quarter"] = [
        {"quarter": q, "base_price_gbp_per_mwh": 8.0, "peak_price_gbp_per_mwh": 20.0, "domestic_volume_weight": 0.25}
        for q in (1, 2, 3, 4)]
    return annual, assumptions, case, balancing, recovery, ro


def test_the_cap_rule_prices_the_blend_and_adds_ofgems_shaping_allowances():
    annual, assumptions, case, balancing, recovery, ro = _cap_rule_fixture()
    rows, summary = bill.project_components(annual, assumptions, case, balancing, recovery, ro)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    blend, rate = 0.7 * 8.0 + 0.3 * 20.0, 0.00232182 + 0.04160818
    assert value["direct_energy"] == pytest.approx(blend * 3.1)          # the fixture's currency factor is 1
    assert value["shaping_adjustment"] == pytest.approx(blend * rate * 3.1)
    record = summary["diagnostics"]["wholesale_procurement"]
    assert record["shaping"]["method"] == "ofgem_cap_rule"
    assert record["shaping"]["model_shaping_over_blend"] == pytest.approx(10.0 / blend - 1.0)
    sens = record["sensitivities_gbp_per_3100kwh_ex_vat"]
    # The fixture zeroes every allowance except losses (multiplier 1), so the uplift is zero.
    assert sens["shaping"]["high_7pct"] == pytest.approx(blend * (0.07 - rate) * 3.1)
    assert sens["shaping"]["model_profile_only"] == pytest.approx((10.0 - blend * (1 + rate)) * 3.1)
    # F18: Ofgem's direct-fuel formula adds every rate on the index (the blend), the premium included.
    index = blend * 3.1
    assert sens["forward_premium"]["plus_5pct"] == pytest.approx(0.05 * index)
    assert sens["forward_premium"]["minus_3pct"] == pytest.approx(-0.03 * index)


def test_the_cap_rule_needs_the_quarterly_prices_and_refuses_an_unknown_method():
    annual, assumptions, case, balancing, recovery, ro = _cap_rule_fixture()
    annual.pop("base_peak_prices_by_quarter")
    with pytest.raises(ValueError, match="base_peak_prices_by_quarter"):
        bill.project_components(annual, assumptions, case, balancing, recovery, ro)
    with pytest.raises(ValueError, match="shaping.method"):
        bill.shaping_settings({"shaping": {"method": "flat"}})
    assert bill.shaping_settings({})["method"] == "model_profile"


def test_the_repository_file_takes_ofgems_current_shaping_rates():
    shaping = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["model_linked_rules"]["direct_fuel"]["shaping"]
    assert shaping["method"] == "ofgem_cap_rule"
    assert (shaping["base_share"], shaping["peak_share"]) == (0.7, 0.3)
    # Annex 2 tab 3a Allowances, unchanged from 2019 to Q4 2026 (v1.30), and the 2020 inputs pinned here.
    inputs = pd.read_csv("data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv")
    assert shaping["seasonal_to_monthly_rate"] == pytest.approx(inputs.seasonal_to_monthly_shaping_rate.iloc[0], abs=1e-6)
    assert shaping["monthly_to_hourly_rate"] == pytest.approx(inputs.monthly_to_hourly_shaping_rate.iloc[0], abs=1e-6)


def test_the_scarcity_summary_is_reported_when_present(tmp_path):
    assert bill.read_scarcity_summary("X", tmp_path)["available"] is False
    (tmp_path / "X_scarcity_summary.json").write_text(json.dumps(
        {"basis": "b", "hours_scarce": 748, "load_weighted_uplift_gbp_per_mwh": 0.88}), encoding="utf-8")
    record = bill.read_scarcity_summary("X", tmp_path)
    assert record["available"] and record["load_weighted_uplift_gbp_per_mwh"] == 0.88


# ── feeding the bill from the new solve: the recovery-demand refresh ─────────

RECOVERY_PATH = Path("data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json")


def test_the_refresh_rewrites_volumes_and_keeps_the_projectors_identities():
    from scripts.consumer_bill.refresh_recovery_demand import refresh_document

    document = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    volumes = {"system_demand_mwh": 315.0e6, "gb_electrical_load_mwh": 305.0e6,
               "electrolyser_draw_mwh": 14.0e6, "h2_offtake_mwh": 10.0e6}
    out = refresh_document(document, volumes, "test months")
    basis = out["metered_supply_basis"]
    share = out["cfd_eligible_supply"]["eligible_share_of_supplier_sales"]["value"]
    # F18: the 2030 file is on two volumes, each derived from the load on its declared factor.
    metered, gsp = 305.0e6 / 1.14 + 14.0e6, 305.0e6 / 1.14 * 1.085 + 14.0e6
    assert basis["metered_supply_mwh"]["value"] == pytest.approx(metered)
    assert basis["gsp_charging_volume_mwh"]["value"] == pytest.approx(gsp)
    assert basis["gsp_chargeable_volume_mwh"]["value"] == pytest.approx(gsp)           # deprecated alias
    assert out["final_demand_mwh"]["value"] == pytest.approx(metered)                   # deprecated alias
    assert out["cfd_eligible_supply"]["cfd_eligible_gsp_volume_mwh"]["value"] == pytest.approx(gsp * share)
    assert out["ro_obligation_eligible_supply_mwh"]["value"] == pytest.approx(metered * share)
    assert out["ro_obligation_eligible_supply_mwh"]["value"] == out["cfd_eligible_supply"]["eligible_supplier_sales_mwh"]["value"]
    assert document["system_demand_mwh"]["value"] != 315.0e6            # the input is not mutated
    # The projector's own reader accepts the refreshed file for a run with these volumes.
    tmp = Path(pytest.importorskip("tempfile").mkdtemp()) / "recovery.json"
    tmp.write_text(json.dumps(out), encoding="utf-8")
    volumes_run = {**volumes, "gsp_chargeable_volume_mwh": 319.0e6}
    recovery = bill.read_recovery_denominators(tmp, out["case_id"], 315.0e6, volumes=volumes_run)
    assert recovery["metered_supply_mwh"] == pytest.approx(metered)
    with pytest.raises(ValueError, match="does not equal system demand"):
        refresh_document(document, {**volumes, "h2_offtake_mwh": 1.0e6}, "bad")


def test_the_refresh_labels_the_months_and_aggregate_it_measured():
    """27 Oct 2026: the label named the reinforced aggregate after the value had moved to cp30_network's months."""
    from scripts.consumer_bill.refresh_recovery_demand import refresh_document, system_demand_source

    document = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    assert "reinforced" not in document["system_demand_mwh"]["source"]
    # The committed label names what the last refresh measured: since 8 Nov 2026 the twelve Stage 1 months directly
    # (refresh stamp), before that the cp30_network annual aggregate.
    measured = document["refresh"]["stamp"]
    if "CP30_FFR_2030_Jan_stage1" in measured:
        assert "12 monthly Stage 1 runs" in document["system_demand_mwh"]["source"]
        assert "CP30_FFR_2030_Jan_stage1" in document["system_demand_mwh"]["source"]
    else:
        assert "CP30_2030_cp30_network_annual_summary.json" in document["system_demand_mwh"]["source"]
    months = [f"CP30_FFR_2030_{m}_stage1" for m in ("Jan", "Feb", "Dec")]
    label = system_demand_source(months, "CP30_2030_cp30_network")
    assert months[0] in label and months[-1] in label and "3 monthly Stage 1 runs" in label
    assert "CP30_2030_cp30_network_annual_summary.json" in label
    assert "annual aggregate" not in system_demand_source(months)
    volumes = {"system_demand_mwh": 315.0e6, "gb_electrical_load_mwh": 305.0e6,
               "electrolyser_draw_mwh": 14.0e6, "h2_offtake_mwh": 10.0e6}
    assert refresh_document(document, volumes, "x", demand_source=label)["system_demand_mwh"]["source"] == label
    # Without one, the label is kept as it is.
    assert (refresh_document(document, volumes, "x")["system_demand_mwh"]["source"]
            == document["system_demand_mwh"]["source"])


def test_the_capacity_market_pin_is_checked_against_the_builder():
    from scripts.consumer_bill.refresh_recovery_demand import check_capacity_market_pin

    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    pinned = assumptions["sourced_externals"]["capacity_market"]["derivation"]["agreement_cost_gbp"]
    assert check_capacity_market_pin(assumptions, {"calendar_2030_cost_gbp": pinned})["agrees"]
    with pytest.raises(ValueError, match="update sourced_externals.capacity_market"):
        check_capacity_market_pin(assumptions, {"calendar_2030_cost_gbp": pinned * 1.1})


# ── the Oct-Dec 2026 cap values and the named sensitivities ────────────────────

def test_the_updated_lines_come_from_the_stored_oct_dec_2026_cap():
    sourced = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]
    for key in ("supplier_costs_and_margin", "smart_meter_net_cost_change", "duos", "bics_levy", "sizewell_c_rab"):
        block = sourced[key]
        assert Path(block["source_file"]).exists(), key
        # DUoS (29 Sep 2026) keeps the cap 17a base and rolls it forward on the RIIO-ED2 path.
        assert block["source_period"].startswith("Ofgem cap period 17a, October-December 2026")
        assert block["rebasing"]["from_price_year"] == 2026 and block["rebasing"]["to_price_year"] == 2025
    assert sourced["smart_meter_net_cost_change"]["value"] < 0 and sourced["smart_meter_net_cost_change"]["value_basis"] == "signed"
    assert sourced["bics_levy"]["value"] == pytest.approx(1.043618419609701 * 3.1)       # NCC, not BICS
    assert sourced["sizewell_c_rab"]["value"] == pytest.approx(4.634096837462586 * 3.1)
    assert "Network Charging Compensation" in sourced["bics_levy"]["label"]


def test_a_signed_line_may_be_negative_and_an_unsigned_one_may_not():
    from tests.unit.test_future_consumer_bill import fixture_inputs

    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    block = assumptions["sourced_externals"]["smart_meter_net_cost_change"]
    block.update(value=-1.5, value_basis="signed", price_year=2025)
    block.pop("rebasing", None)
    rows, _ = bill.project_components(annual, assumptions, case, balancing, recovery, ro)
    assert rows.set_index("component_id").loc["smart_meter_net_cost_change", "annual_gbp_per_3100kwh_ex_vat"] == -1.5
    block.pop("value_basis")
    with pytest.raises(ValueError, match="positive"):
        bill.project_components(annual, assumptions, case, balancing, recovery, ro)


def test_levy_and_capacity_market_sensitivities_are_reported():
    frame = pd.DataFrame({"component_id": ["cfd", "renewables_obligation", "feed_in_tariffs", "capacity_market", "duos"],
                          "annual_gbp_per_3100kwh_ex_vat": [100.0, 50.0, 20.0, 30.0, 120.0]})
    assumptions = {"levy_exemptions": {"sensitivities": {"bics": {"additional_exempt_share": 0.05,
                                                                  "additional_exemption_covers_capacity_market": True}}},
                   "sourced_externals": {"capacity_market": {"derivation": {
                       "agreement_cost_gbp": 2.0e9, "recovery_denominator": "gsp_chargeable_volume_mwh",
                       "household_allocation_factor": 1.5, "sensitivity_cost_gbp": {"low": 1.5e9}}}}}
    out = bill.levy_and_cm_sensitivities(frame, assumptions, {"gsp_chargeable_volume_mwh": 300e6}, 3100)
    # F18 (user, 6 Oct 2026): BICS exempts industry from the RO, FiT and Capacity Market, not the CfD.
    assert out["levy_exemption_bics"] == pytest.approx((50 + 20 + 30) * (1 / 0.95 - 1))
    assert out["capacity_market_low"] == pytest.approx(-0.5e9 / 300e6 * 3.1 * 1.5)
