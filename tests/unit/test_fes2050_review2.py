"""FES 2050 review round 2 (3 Oct 2026) and decisions 68-69: light synthetic checks.

Item 1 (rate-path offsets only on the path's technologies), item 2 / decision 68 (co-owned rows
financed separately), items 3-4 (the LDES floor per owner and on the level annuity), item 5 (the
BSUoS base), item 6 (support income in the valuation), item 7 (a bound's own top-ups), item 8 (HPC
carried capacity, the comparator gate, solar repowered whole, the RAB netting, the merged config),
decision 69 (the risk-adjusted view), and the review's untested paths.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.finance import capital_recovery_factor as crf


# ── item 1: the rate-path offsets ────────────────────────────────────────────

def test_a_rate_path_calibration_overlays_the_flat_one_outside_its_technologies():
    from scripts.public_ownership.contract_book_2050 import (ContractBookError, overlay_rate_path_calibration,
                                                            rate_path_strike_offsets)

    flat = pd.DataFrame({"key": ["PAR7_2031_onshore", "PAR7_2031_nuclear", "DPA_2031"],
                         "technology": ["onshore", "nuclear", None], "calibrated_value": [86.9, 110.0, 50.0]})
    ramp = pd.DataFrame({"key": ["PAR7_2031_onshore"], "technology": ["onshore"], "calibrated_value": [80.0]})
    out = overlay_rate_path_calibration(flat, ramp, ("offshore", "onshore")).set_index("key")["calibrated_value"]
    assert out.to_dict() == {"PAR7_2031_onshore": 80.0, "PAR7_2031_nuclear": 110.0, "DPA_2031": 50.0}
    assert overlay_rate_path_calibration(flat, None, ("onshore",)) is flat
    table = lambda s: SimpleNamespace(strikes=s)  # noqa: E731
    flat_t = table({("PAR7_2031", "onshore"): 86.9, ("PAR7_2031", "nuclear"): 110.0})
    assert rate_path_strike_offsets(flat_t, table({("PAR7_2031", "onshore"): 80.0, ("PAR7_2031", "nuclear"): 110.0}),
                                    ("offshore", "onshore")) == {("PAR7_2031", "onshore"): pytest.approx(6.9)}
    with pytest.raises(ContractBookError, match="does not move"):
        rate_path_strike_offsets(flat_t, table({("PAR7_2031", "onshore"): 80.0, ("PAR7_2031", "nuclear"): 95.3}),
                                 ("offshore", "onshore"))


# ── item 2 / decision 68: co-owned rows ──────────────────────────────────────

def _l4_register(co_ownership):
    from tests.unit.test_fes2050_l4 import NO_OVERHEAD, _assets, _costs, _snapshot, _specs
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios

    assets = _assets().assign(market_revenue_gbp=9e6, support_revenue_gbp=0.0, fuel_carbon_gbp=0.0)
    register, _ = calculate_financing_scenarios(assets, _costs(), _snapshot(), case_specs=_specs(), price_year=2024,
                                                state_operating_overhead=NO_OVERHEAD, co_ownership=co_ownership)
    return register


def test_separate_financing_puts_the_state_on_its_own_cost():
    from scripts.public_ownership.calculate_financing_scenarios import co_ownership_settings

    register = _l4_register(co_ownership_settings({"co_ownership_finance": {"basis": "separate"}}))
    case = register.loc[register["case_id"].eq("public_ownership@deployment_share_25")
                        & register["finance_vintage"].eq("new_build")]
    f = case["ownership_share"].astype(float)
    n = case["remaining_life_years"].astype(float)
    capex = {"Solar PV": 600.0, "Onshore wind": 1400.0}
    gross = case["capacity_mw"].astype(float) * 1000.0 * case["finance_technology"].map(capex)
    cap_s = gross * [crf(0.0325, y) for y in n]
    cap_p = gross * [crf(0.08, y) for y in n]
    assert np.allclose(case["annual_capital_recovery_gbp"].astype(float), f * cap_s + (1 - f) * cap_p, rtol=1e-12)
    om = case["capacity_mw"].astype(float) * 1000.0 * 10.0
    # the state's surplus is its share of the revenue less its own cost (capital at its rate)
    expect = f * 9e6 - f * (cap_s + om)
    assert np.allclose(case["public_surplus_before_recycling_gbp"].astype(float), expect, rtol=1e-10)
    assert (case["co_ownership_basis"] == "separate").all()
    # the co-owner keeps the rest of the rent: (1 - f) of the revenue less its own cost at the private rate
    rent = case["producer_rent_gbp"].astype(float)
    co_owner = rent - case["public_surplus_before_recycling_gbp"].astype(float)
    assert np.allclose(co_owner, (1 - f) * 9e6 - (1 - f) * (cap_p + om), rtol=1e-10)


def test_pooled_financing_is_the_original_treatment_and_the_reduction_is_a_gain():
    from scripts.public_ownership.calculate_financing_scenarios import co_ownership_settings

    pooled = _l4_register(None)
    stress = _l4_register(co_ownership_settings({"co_ownership_finance": {"basis": "pooled"}}))
    cols = ["annual_capital_recovery_gbp", "producer_rent_gbp", "public_surplus_before_recycling_gbp"]
    pd.testing.assert_frame_equal(pooled[cols].astype(float), stress[cols].astype(float))
    reduced = _l4_register(co_ownership_settings({"co_ownership_finance": {"basis": "separate_with_private_reduction"}}))
    separate = _l4_register(co_ownership_settings({"co_ownership_finance": {"basis": "separate"}}))
    pick = lambda r: r.loc[r["case_id"].eq("public_ownership@deployment_share_50")  # noqa: E731
                           & r["finance_vintage"].eq("new_build")]
    a, b = pick(reduced), pick(separate)
    gain = a["co_investment_rate_gain_gbp"].astype(float)
    assert (gain > 0).all()
    # the gain is the co-owner's capital fall, and its rate is the private rate less 0.5 bp x 50 = 25 bp
    assert np.allclose(b["annual_capital_recovery_gbp"].astype(float) - a["annual_capital_recovery_gbp"].astype(float),
                       gain, rtol=1e-9)
    assert np.allclose(a["co_owner_rate"].astype(float), 0.08 - 0.0025)
    assert np.allclose(a["public_surplus_before_recycling_gbp"].astype(float),
                       b["public_surplus_before_recycling_gbp"].astype(float), rtol=1e-12)    # the state is untouched
    with pytest.raises(ValueError):
        co_ownership_settings({"co_ownership_finance": {"basis": "separate", "private_rate_reduction_bp_per_point": 0.5}})
    with pytest.raises(ValueError):
        co_ownership_settings({"co_ownership_finance": {"basis": "separate", "co_investment_genuine_share": 0.6}})
    assert co_ownership_settings({}) is None


def test_the_public_account_and_party_flows_take_the_adjustments():
    from scripts.public_ownership import party_flows as pf
    from scripts.public_ownership.public_account import co_ownership_adjustments

    frame = pd.DataFrame({"co_ownership_state_adjustment_gbp": [5.0, np.nan], "ownership_share": [0.25, 1.0]})
    cost, floor = co_ownership_adjustments(frame)
    assert cost.tolist() == [5.0, 0.0] and floor.tolist() == [0.0, 0.0]
    assert co_ownership_adjustments(pd.DataFrame({"x": [1.0]}))[0].tolist() == [0.0]       # 2030: nothing
    reg = pd.DataFrame({"case_id": ["p", "private"], "asset_id": ["A", "A"], "producer_rent_gbp": [1.0, 0.0],
                        "ownership_share": [0.25, 0.0], "annual_capital_recovery_gbp": [80.0, 100.0],
                        "annual_required_revenue_gbp": [90.0, 110.0],
                        "co_ownership_state_adjustment_gbp": [3.0, 0.0],
                        "co_ownership_state_capital_adjustment_gbp": [2.0, 0.0]})
    with_adj = pf.register_components(reg, "p")
    without = pf.register_components(reg.drop(columns=["co_ownership_state_adjustment_gbp",
                                                       "co_ownership_state_capital_adjustment_gbp"]), "p")
    assert with_adj["state_capital"] == pytest.approx(without["state_capital"] + 2.0)
    assert with_adj["private_capital"] == pytest.approx(without["private_capital"] - 2.0)
    assert with_adj["state_operating"] == pytest.approx(without["state_operating"] + 1.0)
    total = lambda c: sum(c[k] for k in ("state_capital", "private_capital", "state_operating", "private_operating"))  # noqa: E731
    assert total(with_adj) == pytest.approx(total(without))


# ── items 3-4: the LDES floor ────────────────────────────────────────────────

def test_the_floor_follows_the_level_annuity_and_each_owner():
    from scripts.public_ownership.public_account import _floor_parts, ldes_floor_topups

    capex = 1000.0
    life, remaining = 25.0, 10.0
    rows = pd.DataFrame({
        "case_id": ["public_ownership@deployment_share_25", "private", "public_ownership@deployment_share_25"],
        "annual_capital_recovery_gbp": [capex * crf(0.08, life)] * 2 + [0.25 * capex * crf(0.0325, life)
                                                                       + 0.75 * capex * crf(0.08, life)],
        "wacc": [0.08, 0.08, 0.07], "remaining_life_years": [remaining] * 3, "capital_annuity_years": [life] * 3,
        "annual_fixed_om_gbp": [0.0] * 3, "annual_variable_om_gbp": [0.0] * 3,
        "ownership_share": [0.0, 0.0, 0.25], "co_ownership_basis": [None, None, "separate"],
        "capex_state_view_gbp": [np.nan, np.nan, capex], "capex_private_view_gbp": [np.nan, np.nan, capex],
        "state_part_rate": [np.nan, np.nan, 0.0325], "co_owner_rate": [np.nan, np.nan, 0.08]})
    rates = {"public_ownership": 0.02, "private": 0.045}
    parts = _floor_parts(rows, rates, "row_owner")
    # a level-annuity row recovers its capex over its life: 1000 x CRF(4.5%, 25), not over the remaining 10 years
    assert parts["row"].iloc[1] == pytest.approx(capex * crf(0.045, life))
    # a separately financed co-owned row: the state's 25% at its floor, the co-owner's 75% at the private floor
    assert parts["row"].iloc[2] == pytest.approx(0.25 * capex * crf(0.02, life) + 0.75 * capex * crf(0.045, life))
    assert parts["state"].iloc[2] == pytest.approx(capex * crf(0.02, life))
    pooled = rows.assign(co_ownership_basis=[None, None, "pooled"])
    from scripts.public_ownership.finance import co_ownership_rate

    blend = co_ownership_rate(0.02, 0.045, 0.25, life)
    got = _floor_parts(pooled, rates, "row_owner")["row"].iloc[2]
    assert got == pytest.approx(rows["annual_capital_recovery_gbp"].iloc[2] / crf(0.07, life) * crf(blend, life))
    frame = rows.assign(component_type="StorageUnit", carrier="LAES", finance_vintage="new_build",
                        annual_required_revenue_gbp=100.0, market_revenue_gbp=10.0, capacity_market_revenue_gbp=0.0)
    out = ldes_floor_topups(frame, {"ldes_carriers": ["LAES"], "ldes_enabled": True,
                                    "ldes_floor_real_rate_by_case": rates, "ldes_floor_rate_key": "row_owner"})
    state_part = 0.25 * max(capex * crf(0.02, life) - 10.0, 0.0)
    assert out["ldes_floor_topup_state_part_gbp"].iloc[2] == pytest.approx(state_part)
    assert out["ldes_floor_topup_gbp"].iloc[2] == pytest.approx(
        state_part + 0.75 * max(capex * crf(0.045, life) - 10.0, 0.0))
    # a 2030 frame (no co-ownership or annuity columns) is floored exactly as before
    plain = frame.drop(columns=["capital_annuity_years", "co_ownership_basis", "capex_state_view_gbp",
                                "capex_private_view_gbp", "state_part_rate", "co_owner_rate"])
    assert "ldes_floor_topup_state_part_gbp" not in ldes_floor_topups(
        plain, {"ldes_carriers": ["LAES"], "ldes_enabled": True, "ldes_floor_real_rate_by_case": rates}).columns


def test_the_merchant_share_of_a_term_row_keeps_its_economic_life_charge():
    from scripts.public_ownership.calculate_financing_scenarios import _term_basis_capital

    terms = {"term_row": np.array([True]), "expired_row": np.array([False]), "term_years": np.array([15.0]),
             "live_share": np.array([0.7]), "tail_margin_gbp_per_kw_year": np.array([0.0]),
             "life_years": np.array([38.0]), "merchant_share": np.array([0.3])}
    out = _term_basis_capital(terms, np.array([0]), np.array([0.05]), np.ones(1), np.zeros(1), np.full(1, 500.0),
                              np.zeros(1), np.array([1234.0]))
    assert out[0] == pytest.approx(0.7 * 500e3 * crf(0.05, 15) + 0.3 * 1234.0)


# ── item 6: the valuation counts the support income ──────────────────────────

def test_the_valuation_counts_support_income():
    from scripts.public_ownership import legacy_valuation as lv

    rows = pd.DataFrame({"case_id": "private", "asset_id": ["B1"], "component_type": ["StorageUnit"],
                         "carrier": ["Battery"], "finance_vintage": "legacy", "support_type": [None],
                         "cfd_round": [None], "cfd_award_year": [np.nan], "capacity_mw": [10.0],
                         "commissioning_year": [2045.0], "market_revenue_gbp": [100.0], "cfd_support_gbp": 0.0,
                         "cfd_curtailment_compensation_gbp": 0.0, "roc_support_gbp": 0.0, "fuel_carbon_gbp": 0.0,
                         "annual_fixed_om_gbp": 20.0, "wacc": 0.07, "model_year": 2050})
    settings = lv.settings_from_config(None, model_year=2050)
    shares = pd.DataFrame(1.0, index=range(2050, 2101), columns=[lv.RO_SHARE_ALL])
    base, _ = lv.legacy_cash_flow_profile(rows, settings, shares)
    more, _ = lv.legacy_cash_flow_profile(rows.assign(ancillary_services_revenue_gbp=7.0,
                                                      battery_subhourly_revenue_gbp=3.0), settings, shares)
    assert more["net_operating_cash"][0, 0] == pytest.approx(base["net_operating_cash"][0, 0] + 10.0)


# ── decision 69 and the identity ─────────────────────────────────────────────

def test_the_risk_adjusted_view_moves_the_wedge_to_taxpayers_and_closes():
    from scripts.public_ownership import investor_ledger as il

    parties = {"bill_change_system_gbp": -60.0,
               "investor_parts_gbp": {"rent": -10.0},
               "resource_parts_gbp": {il.CAPITAL_REGISTER: 40.0, "O&M": 5.0, il.CO_INVESTMENT_GUARANTEE: -3.0}}
    # consumers +60, investors -10, resource 42: the Exchequer must be -8
    row = il.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=-8.0, case_id="c", risk_adjusted=True)
    assert row["cost_of_capital_saving_gbp"] == pytest.approx(37.0)       # the guarantee is no capital saving
    assert row["taxpayer_risk_bearing_gbp"] == pytest.approx(37.0)
    assert row["resource_saving_risk_adjusted_gbp"] == pytest.approx(5.0)
    assert row["exchequer_change_risk_adjusted_gbp"] == pytest.approx(-45.0)
    assert "taxpayer_risk_bearing_gbp" not in il.three_party_row(parties, vat_change_gbp=0.0,
                                                                 exchequer_change_gbp=-8.0, case_id="c")
    from scripts.public_ownership.exchequer_ledger import ledger_options

    assert ledger_options({}) == {}
    assert ledger_options({"risk_adjusted_view": {"enabled": True},
                           "co_ownership_finance": {"basis": "separate_with_private_reduction",
                                                    "co_investment_genuine_share": 0.25}}) == {
        "risk_adjusted_view": True, "co_investment_genuine_share": 0.25}


def test_a_market_case_with_support_lines_closes_the_identity():
    """End to end on two register rows: support income booked to both cases, the state's share recycled."""
    from scripts.public_ownership import investor_ledger as il
    from scripts.public_ownership import support_lines_2050 as sl

    reg = pd.DataFrame({"case_id": ["private", "p2"], "asset_id": ["B", "B"], "producer_rent_gbp": [-50.0, -20.0],
                        "public_surplus_before_recycling_gbp": [0.0, -20.0], "ownership_share": [0.0, 1.0],
                        "annual_required_revenue_gbp": [150.0, 120.0], "annual_capital_recovery_gbp": [100.0, 70.0],
                        "market_revenue_gbp": [100.0, 100.0], "support_revenue_gbp": 0.0, "fuel_carbon_gbp": 0.0,
                        "corporation_tax_forgone_gbp": [0.0, 0.0]})
    booked = sl.book_support(reg, {sl.COLUMN_ANCILLARY: pd.Series({"B": 10.0}),
                                   sl.COLUMN_BATTERY_TOP_UP: pd.Series({"B": 40.0})})
    pos = {c: il.register_position(booked, c) for c in ("private", "p2")}
    parties = il.market_case_parties(bill_case={}, bill_private={}, account_case={}, position=pos["p2"],
                                     position_private=pos["private"], network=il.network_parties(None, None, None))
    # the bill is the same in both cases; the state's surplus (+30) is the Exchequer's when retained
    exchequer = float(booked.loc[1, "public_surplus_before_recycling_gbp"] - booked.loc[0, "public_surplus_before_recycling_gbp"])
    row = il.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=exchequer, case_id="p2")
    assert row["three_party_identity_residual_gbp"] == pytest.approx(0.0, abs=1e-9)
    assert row["resource_saving_vs_private_gbp"] == pytest.approx(30.0)


# ── items 7 and 8 ────────────────────────────────────────────────────────────

def test_a_group_rereads_its_top_ups_from_its_own_account(tmp_path):
    import json

    from scripts.consumer_bill.project_future_bill import register_booked_cost, register_lines_from
    from scripts.public_ownership.project_public_ownership_consumer_bill import own_account_top_up_lines

    parent, own = tmp_path / "parent.json", tmp_path / "own.json"
    parent.write_text(json.dumps({"cases": {"private": {"battery_revenue_top_up_gbp": 1.0}}}))
    own.write_text(json.dumps({"cases": {"private": {"battery_revenue_top_up_gbp": 2.0}}}))
    assumptions = tmp_path / "a.json"
    assumptions.write_text(json.dumps({"sourced_externals": {"battery_revenue_top_up": {"derivation": {"cost_from": {
        "public_account": str(parent), "case": "private", "key": "battery_revenue_top_up_gbp"}}}}}))
    account = json.loads(own.read_text())
    lines = own_account_top_up_lines(assumptions, own, account)
    assert lines == {"battery_revenue_top_up": str(own)}
    assert own_account_top_up_lines(assumptions, parent, json.loads(parent.read_text())) == {}
    cost_from = {"public_account": str(parent), "case": "private", "key": "battery_revenue_top_up_gbp"}
    assert register_booked_cost("battery_revenue_top_up", cost_from)[0] == 1.0
    with register_lines_from(lines):
        assert register_booked_cost("battery_revenue_top_up", cost_from)[0] == 2.0
    assert register_booked_cost("battery_revenue_top_up", cost_from)[0] == 1.0          # restored


def test_the_carried_capacity_is_what_the_earlier_purchase_bought():
    from scripts.public_ownership.nationalised_tariff import revenue_requirement
    import inspect

    assert "carried_mw" in inspect.signature(revenue_requirement).parameters


def test_the_comparator_revenue_is_its_own_only_on_a_rate_path():
    from scripts.public_ownership.cfd_repricing import SavingsContext

    shared = pd.Series([1.0])
    frame = pd.DataFrame({"market_revenue_gbp": [5.0], "support_revenue_gbp": [0.0], "fuel_carbon_gbp": [0.0]})
    import scripts.public_ownership.cfd_repricing as cr

    # F20 phase 2: the comparator is looked up through the context (memoised), read at call time here.
    fake = SimpleNamespace(settings_by_case={}, has_settlement=True, contract_revenue=shared,
                           private_for=lambda case_id: frame,
                           comparator=lambda case_id: cr.comparator_for_settings(case_id, {}))

    original = cr.comparator_for_settings
    cr.comparator_for_settings = lambda case_id, settings: "private@other"
    try:
        assert SavingsContext.contract_revenue_for(fake, "x") is shared            # no offset: the shared revenue
        frame["cfd_strike_offset_gbp_per_mwh"] = [2.0]
        assert SavingsContext.contract_revenue_for(fake, "x").tolist() == [5.0]
    finally:
        cr.comparator_for_settings = original


def test_support_lines_refuse_an_ambiguous_merged_block():
    from scripts.public_ownership import support_lines_2050 as sl

    with pytest.raises(sl.SupportLinesError, match="both file and table"):
        sl.settings({"ancillary_services": {"file": "a.csv", "table": "b.csv"}})
    with pytest.raises(sl.SupportLinesError, match="name one form"):
        sl.settings({"battery_subhourly_uplift": {"share_by_case": {"central": 0.2, "high": 0.4}, "central": 0.2}})


ROUND3_CONFIG = Path(os.environ.get("FIX53_ROUND3_CONFIG", "config"))


def _round3_po():
    from scripts.public_ownership.study_config import load_studies

    po = load_studies(ROUND3_CONFIG).configs["FES25_HT_2050"]["public_ownership"]
    if "co_ownership_finance" not in po:
        pytest.skip("the HT50 round-3 patch (scratchpad FIX53 ht50_scenarios_round3.patch) is not applied yet")
    return po


def test_the_merged_ht50_config_uses_the_2050_tables():
    from scripts.public_ownership.calculate_financing_scenarios import co_ownership_settings, support_line_settings

    po = _round3_po()
    support = support_line_settings(po)
    assert Path(support["ancillary_services"]["file"]).name == "ancillary_services_2050.csv"
    assert Path(support["efw_gate_fee"]["file"]).name == "efw_gate_fees_2050.json"
    assert support["battery_subhourly_uplift"]["share"] == 0.20 and support["battery_subhourly_uplift"]["high_share"] == 0.40
    assert set(support) == {"ancillary_services", "efw_gate_fee", "life_extension_top_up", "battery_top_up",
                            "battery_subhourly_uplift"}
    assert co_ownership_settings(po)["basis"] == "separate"
    assert po["risk_adjusted_view"] == {"enabled": True}
    groups = po["reference_run_groups"]
    assert groups["fes_ht_2050_coownership_reduction_bound"]["co_ownership_finance"]["basis"] == \
        "separate_with_private_reduction"
    # Decision 55's MW are derived from the final 2030 register's Hinkley row (F19 dates HPC from 2031), so
    # the key is declared and empty until then (review 2's 1,630 + 687 rested on a 31 July 2030 start).
    vc = po["vintage_capital_2050"]
    assert "carried_from_earlier_purchase_mw" in vc
    assert set(vc["carried_from_earlier_purchase_mw"]) <= set(vc["carried_from_earlier_purchase_contracts"])


# ── round 4 (3 Oct 2026) ─────────────────────────────────────────────────────

def test_the_tariff_carries_its_own_electrolyser_exemption():
    from scripts.public_ownership.exchequer_ledger import tariff_policy_ratio, uncertainties_by_case

    tariff = {"cases": {"p3": {"bill": {"replaced_lines_total_gbp_per_3100kwh_ex_vat": 400.0,
                                        "new_lines_total_gbp_per_3100kwh_ex_vat": 350.0}}}}
    assert tariff_policy_ratio(tariff["cases"]["p3"]) == pytest.approx(0.875)
    forecast = {"modelled_year": 2050, "diagnostics": {"levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat": {
        "levy_exemption_electrolysers_2050": 10.0}}}
    out = uncertainties_by_case([{"case_id": "private"}, {"case_id": "p3"}], forecast, tariff, 0.05)
    exemption = out["p3"]["levy_exemption_electrolysers_2050"]
    assert out["private"]["levy_exemption_electrolysers_2050"]["case_bill_change_gbp_inc_vat"] == pytest.approx(10.5)
    assert exemption["case_bill_change_gbp_inc_vat"] == pytest.approx(10.5 * 0.875)
    assert exemption["gap_change_gbp_inc_vat"] == pytest.approx(10.5 * 0.875 - 10.5)


def test_the_post_solve_chain_runs_the_coownership_bounds():
    from scripts.public_ownership import post_solve_2050 as ps

    assert set(ps.COOWNERSHIP_BOUND_GROUPS) <= set(ps.BOUND_GROUPS)
    for group in ps.COOWNERSHIP_BOUND_GROUPS:
        targets = [t for t in ps.BOUND_TARGETS if f"/{group}_" in t]
        assert len(targets) == 2 and not any("nationalised" in t for t in targets)
