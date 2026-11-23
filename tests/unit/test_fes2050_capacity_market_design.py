"""FES 2050 decision 73 (user, 5 Oct 2026): the Capacity Market design under public ownership.

Synthetic fixtures only: a nine-row auction fleet on the MR4 bid shapes, and a 120-row register.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import capacity_market_design as design
from scripts.public_ownership.levers import (
    ALL_LEVERS, CM_DESIGN_MARKET_PARTICIPANT, CM_DESIGN_STATE_PLANNED, CM_DESIGN_STUDY, GATED_LABELS, GATED_LEVERS,
    LABELS, LEVER_CM_DESIGN, LEVERS, expand_cases, lever_default, parse_scenarios, parse_sensitivities,
)
from config.config_loader import load_defaults

SCENARIOS = {
    "private": {},
    "public_control": {"new_build_generation": "state_stake", "public_return": "cepa_decomposition"},
    "public_ownership": {"new_build_generation": "public", "public_return": "public_rate",
                         "public_rate_basis": "real_cpi"},
}
NO_OVERHEAD = {"enabled": False}


def _auction(requirement_mw: float = 34000.0):
    from scripts.market import capacity_market_2050 as cm50

    fleet = pd.DataFrame([
        ("solar_pv", 2040, 4000.0, 0.0, "existing", "solar_pv"),
        ("solar_pv", 2048, 1500.0, 0.0, "new_build", "solar_pv"),
        ("wind_onshore", 2035, 3000.0, 500.0, "existing", "wind_onshore"),
        ("Battery", 2045, 5000.0, 0.0, "existing", "battery"),
        ("V2G", 2050, 6000.0, 0.0, "dsr", "dsr"),
        ("DSR_residential", 2050, 2000.0, 0.0, "dsr", "dsr"),
        ("DSR_industrial_commercial", 2050, 1000.0, 0.0, "dsr", "dsr"),
        ("interconnector", 2030, 4000.0, 0.0, "interconnector", "interconnector"),
        ("nuclear", 2045, 0.0, 9000.0, "existing", "nuclear"),
    ], columns=["carrier", "build_year", "eligible_derated_mw", "non_cm_derated_mw", "category", "rate_key"])
    settings = {k: v for k, v in asdict(cm50.Settings2050()).items() if k != "notes"}
    shape = 5.2135
    distributions = {"existing": {"floor_share": 0.9966, "shape": shape}, "new_build": {"floor_share": 0.4991, "shape": shape},
                     "dsr": {"floor_share": 0.8232, "shape": shape}, "interconnector": {"floor_share": 1.0, "shape": shape}}
    summary = {"model_year": 2050, "settings": settings, "bid_distributions": distributions,
               "requirement": {"requirement_mw": requirement_mw}, "indexation_2025_26_to_2025": 1.0}
    return fleet, summary


def _options(central=CM_DESIGN_STATE_PLANNED, factor=1.04):
    fleet, summary = _auction()
    return design.from_inputs(fleet, summary, central=central, rebasing_factor=factor)


def _specs(one_at_a_time=None, named=None):
    bundles = parse_scenarios(SCENARIOS)
    sens = {"bounding_pair": False, "one_at_a_time": one_at_a_time or [], "named": named or {}}
    return expand_cases(bundles, parse_sensitivities(sens, bundles))


# ── the lever ────────────────────────────────────────────────────────────────


def test_the_lever_is_gated_with_unique_labels_and_a_study_default():
    lever = GATED_LEVERS[LEVER_CM_DESIGN]
    assert lever.values == (CM_DESIGN_STUDY, CM_DESIGN_MARKET_PARTICIPANT, CM_DESIGN_STATE_PLANNED)
    assert lever_default(LEVER_CM_DESIGN) == CM_DESIGN_STUDY and not lever.register
    assert LEVER_CM_DESIGN not in LEVERS and LEVER_CM_DESIGN in ALL_LEVERS
    assert not set(GATED_LABELS) & set(LABELS)
    ids = [s.case_id for s in _specs([{LEVER_CM_DESIGN: [CM_DESIGN_MARKET_PARTICIPANT]}])]
    # P2 only (it requires the state's new build); private and P1's stakes have nothing to move
    assert [c for c in ids if "cm_" in c] == ["public_ownership@cm_market_participant"]


def test_no_2030_config_names_the_design():
    import yaml

    from scripts.public_ownership.calculate_financing_scenarios import register_options_from_config
    from scripts.public_ownership.reference_runs import configured_case_specs

    block = load_defaults()["public_ownership"]
    assert design.CONFIG_KEY not in block
    assert all(LEVER_CM_DESIGN not in spec.settings for spec in configured_case_specs(block))
    assert design.settings(block.get(design.CONFIG_KEY)) is None


def test_resolve_routes_each_case_to_its_design():
    by = {s.case_id: s for s in _specs(named={"p3": {"scenario": "public_ownership", "legacy_fleet": "acquire_all"},
                                               "p3_mp": {"extends": "p3", LEVER_CM_DESIGN: "market_participant"}})}
    options = _options()
    assert design.resolve(by["private"].settings, options) == design.AUCTION_ALL
    assert design.resolve(by["public_control"].settings, options) == design.STATE_OUT
    assert design.resolve(by["public_ownership"].settings, options) == design.STATE_OUT
    assert design.resolve(by["public_ownership@p3"].settings, options) == design.ADEQUACY_PLANNING
    assert design.resolve(by["public_ownership@p3_mp"].settings, options) == design.AUCTION_ALL
    assert design.resolve(by["public_ownership"].settings, None) == design.AUCTION_ALL
    assert design.resolve(by["public_ownership"].settings, _options(CM_DESIGN_MARKET_PARTICIPANT)) == design.AUCTION_ALL


def test_settings_refuse_unknown_keys_and_values():
    with pytest.raises(design.CapacityMarketDesignError, match="unknown keys"):
        design.settings({"central": "state_planned", "fleet": "x", "summary": "y", "price": 1})
    with pytest.raises(design.CapacityMarketDesignError, match="central"):
        design.settings({"central": "auction_all", "fleet": "x", "summary": "y"})
    fleet, summary = _auction()
    summary["calendar_2050_cost_gbp"] = 1.0
    with pytest.raises(design.CapacityMarketDesignError, match="not one build"):
        design.from_inputs(fleet, summary)


# ── the auction ──────────────────────────────────────────────────────────────


def test_no_exit_reproduces_the_full_auction_exactly():
    options = _options()
    fleet = options["fleet"]
    same = design.clear(design.apply_exit(fleet, {"cell": {}, "carrier": {}}), options["summary"])
    assert same["cost_gbp"] == options["full"]["cost_gbp"]
    assert same["price_gbp_per_kw_year_2025"] == options["full"]["price_gbp_per_kw_year_2025"]
    assert abs(sum(options["full"]["payments_by_payee_gbp"].values()) - options["full"]["cost_gbp"]) < 1.0


def test_the_target_falls_by_exactly_what_the_state_takes_out():
    options = _options()
    shares = {"cell": {("Battery", "existing"): 1.0, ("solar_pv", "existing"): 0.5}, "carrier": {}}
    exited = design.apply_exit(options["fleet"], shares)
    state = float(exited["state_planned_derated_mw"].sum())
    assert state == pytest.approx(5000.0 + 2000.0)
    assert float(exited["eligible_derated_mw"].sum() + state) == pytest.approx(options["fleet"]["eligible_derated_mw"].sum())
    case = design.clear(exited, options["summary"])
    t1 = options["summary"]["settings"]["t1_share_of_target"]
    full_target = options["full"]["cleared"]["target_mw"]
    assert full_target - case["cleared"]["target_mw"] == pytest.approx(state / (1.0 + t1))
    assert case["cost_gbp"] < options["full"]["cost_gbp"]
    # V2G and DSR stay in and are paid the residual clearing price
    assert case["payments_by_payee_gbp"]["v2g_owners_households"] > 0


# ── the register booking ─────────────────────────────────────────────────────


def _rows(case_id, own, *, cm=1e6):
    return pd.DataFrame({
        "case_id": case_id, "asset_id": [f"A{i}" for i in range(4)],
        "carrier": ["Battery", "solar_pv", "solar_pv", "nuclear"],
        "support_type": ["merchant", "merchant", "merchant", "CfD"],
        "capacity_mw": [500.0, 800.0, 400.0, 1000.0], "build_year": [2045, 2040, 2048, 2045],
        "ownership_share": own, "capacity_market_revenue_gbp": [cm, cm / 2, cm / 4, 0.0],
        "producer_rent_gbp": [10e6, 5e6, 2e6, 1e6], "public_surplus_before_recycling_gbp": 0.0})


def test_p2_rows_lose_their_cm_revenue_and_co_owned_rows_keep_the_co_owners_part():
    options = _options()
    specs = [s for s in _specs() if s.case_id in ("private", "public_ownership")]
    own = [1.0, 0.5, 0.0, 1.0]
    frame = pd.concat([_rows("private", [0.0] * 4), _rows("public_ownership", own)], ignore_index=True)
    frame["public_surplus_before_recycling_gbp"] = (frame["ownership_share"] * frame["producer_rent_gbp"]).astype(float)
    booked, results = design.book(frame, options, specs)
    private = booked.loc[booked["case_id"].eq("private")]
    pd.testing.assert_frame_equal(private[frame.columns].reset_index(drop=True),
                                  frame.loc[frame["case_id"].eq("private")].reset_index(drop=True), check_dtype=False)
    p2 = booked.loc[booked["case_id"].eq("public_ownership")].reset_index(drop=True)
    before = frame.loc[frame["case_id"].eq("public_ownership")].reset_index(drop=True)
    ratio = results["public_ownership"]["rate_ratio_by_category"]
    assert p2.at[0, "capacity_market_revenue_gbp"] == 0.0                           # whole state row: out
    assert p2.at[1, "capacity_market_revenue_gbp"] == pytest.approx(0.5 * ratio["existing"] * 5e5)  # co-owner's half
    assert p2.at[2, "capacity_market_revenue_gbp"] == pytest.approx(ratio["new_build"] * 2.5e5)     # private row
    # the state keeps none of the CM revenue: its surplus falls by its share of the full revenue
    lost = (before["ownership_share"] * before["capacity_market_revenue_gbp"]).to_numpy()
    assert np.allclose(before["public_surplus_before_recycling_gbp"] - p2["public_surplus_before_recycling_gbp"], lost)
    # the co-owner's position moves by its in-auction part at the residual rate
    private_part = (p2["producer_rent_gbp"] - p2["public_surplus_before_recycling_gbp"]) - (
        before["producer_rent_gbp"] - before["public_surplus_before_recycling_gbp"])
    expect = p2["capacity_market_revenue_gbp"] - (1 - before["ownership_share"]) * before["capacity_market_revenue_gbp"]
    assert np.allclose(private_part, expect)
    assert results["public_ownership"]["design"] == design.STATE_OUT
    assert results["public_ownership"]["state_forgone_register_gbp"] == pytest.approx(lost.sum())


def test_p1_stakes_stay_in_the_auction_so_nothing_moves():
    options = _options()
    specs = [s for s in _specs() if s.case_id == "public_control"]
    frame = _rows("public_control", [0.3, 0.3, 0.3, 0.3])
    booked, results = design.book(frame, options, specs)
    assert results["public_control"]["design"] == design.STATE_OUT
    assert results["public_control"]["state_exit_derated_mw"] == 0.0
    assert results["public_control"]["line_cost_gbp_2025"] == options["full"]["cost_gbp"]
    for column in ("capacity_market_revenue_gbp", "producer_rent_gbp", "public_surplus_before_recycling_gbp"):
        assert np.array_equal(booked[column].to_numpy(dtype=float), frame[column].to_numpy(dtype=float))


def test_summary_columns_and_case_moves_add_up():
    options = _options(factor=1.04)
    specs = [s for s in _specs() if s.case_id in ("private", "public_ownership")]
    frame = pd.concat([_rows("private", [0.0] * 4), _rows("public_ownership", [1.0, 0.5, 0.0, 1.0])], ignore_index=True)
    _, results = design.book(frame, options, specs)
    table = design.summary_columns(results, 1.04)
    p = design.SUMMARY_PREFIX
    assert table.set_index("case_id").at["private", p + "line_cost_gbp"] == pytest.approx(options["full"]["cost_gbp"] / 1.04)
    moves = design.case_moves(table, "public_ownership")
    assert moves["line_change_gbp"] < 0
    assert moves["line_change_gbp"] == pytest.approx(moves["register_change_gbp"] + moves["household_change_gbp"]
                                                     + moves["outside_register_other_change_gbp"])
    assert design.case_moves(table.drop(columns=[p + "line_cost_gbp"]), "public_ownership") is None


# ── the finance step, end to end on a synthetic register ─────────────────────


def _snapshot():
    return {"rate_basis": "real_CPI",
            "assumptions": {"public_coordination_state_equity_share": {"value": 0.30},
                            "public_borrowing_real": {"value": 0.02745}},
            "technologies": [{"technology": t, "risk_rating": "Medium",
                              "wacc": {"private": 0.08, "public_coordination": 0.07,
                                       "public_ownership_new_build": 0.0325, "public_ownership_legacy": 0.02745}}
                             for t in ("Solar PV", "Onshore wind")]}


def _costs():
    return pd.DataFrame([{"technology": t, "capex_power_gbp_per_kw": c, "capex_energy_gbp_per_kwh": 0.0,
                          "fixed_om_gbp_per_kw_year": 10.0, "variable_om_gbp_per_mwh": 0.0,
                          "economic_life_years": 30.0, "price_year": 2024, "real_or_nominal": "real",
                          "status": "reviewed"} for t, c in (("Solar PV", 600.0), ("Onshore wind", 1400.0))])


def _assets():
    rng = np.random.default_rng(11)
    rows = []
    for i in range(120):
        technology = "Solar PV" if i % 3 else "Onshore wind"
        year = 2020 + (i % 31)
        capacity = float(rng.uniform(5, 60))
        rows.append({"model_year": 2050, "component_type": "Generator", "asset_id": f"G{i:03d}::b{year}",
                     "carrier": "solar_pv" if technology == "Solar PV" else "wind_onshore",
                     "finance_technology": technology, "finance_vintage": "legacy" if year < 2025 else "new_build",
                     "build_year": year, "commissioning_year": float(year), "support_type": "merchant",
                     "capacity_mw": capacity, "dispatch_mwh": 1e5, "evidence_status": "resolved",
                     "market_revenue_gbp": capacity * 6e4, "support_revenue_gbp": 0.0, "fuel_carbon_gbp": 0.0,
                     "capacity_market_revenue_gbp": capacity * 1e3 * 3.0})
    return pd.DataFrame(rows)


def _run(options):
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios

    specs = _specs([{LEVER_CM_DESIGN: [CM_DESIGN_MARKET_PARTICIPANT]}])
    return calculate_financing_scenarios(_assets(), _costs(), _snapshot(), case_specs=specs, price_year=2024,
                                         state_operating_overhead=NO_OVERHEAD, capacity_market_design=options)


def test_the_finance_step_books_the_design_and_its_sensitivity_is_todays_register():
    base, base_totals = _run(None)
    register, totals = _run(_options())
    keys = ["case_id", "asset_id"]
    columns = [c for c in base.columns if c != LEVER_CM_DESIGN]
    # private and the market-participant sensitivity: today's register exactly (P2's own rows aside)
    for case_id, base_id in (("private", "private"), ("public_ownership@cm_market_participant", "public_ownership")):
        got = register.loc[register["case_id"].eq(case_id), columns].drop(columns=["case_id"]).reset_index(drop=True)
        want = base.loc[base["case_id"].eq(base_id), columns].drop(columns=["case_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want, check_dtype=False)
    p2 = register.loc[register["case_id"].eq("public_ownership")].set_index("asset_id")
    p2_base = base.loc[base["case_id"].eq("public_ownership")].set_index("asset_id")
    owned = p2["ownership_share"].astype(float) >= 1.0
    assert owned.any() and (p2.loc[owned, "capacity_market_revenue_gbp"] == 0.0).all()
    # required revenue never moves: the design is a CM booking
    assert np.allclose(p2["annual_required_revenue_gbp"].astype(float), p2_base["annual_required_revenue_gbp"].astype(float))
    t = totals.set_index("case_id")
    p = design.SUMMARY_PREFIX
    assert t.at["public_ownership", p + "design"] == design.STATE_OUT
    assert t.at["public_ownership@cm_market_participant", p + "design"] == design.AUCTION_ALL
    assert t.at["public_ownership", p + "line_cost_gbp"] < t.at["private", p + "line_cost_gbp"]
    assert t.at["public_ownership", "public_surplus_before_recycling_gbp"] < t.at[
        "public_ownership@cm_market_participant", "public_surplus_before_recycling_gbp"]
    assert p + "design" not in base_totals.columns and design.COLUMN_STATE_ADJUSTMENT not in base.columns


# ── the identity pieces ──────────────────────────────────────────────────────


def test_household_income_closes_the_three_party_identity():
    from scripts.public_ownership import investor_ledger as il

    # consumers pay 100 less on the CM line; households lose 20 of V2G income; the state 50; investors 30
    parties = {"bill_change_system_gbp": -100.0, "investor_parts_gbp": {"x": -30.0}, "resource_parts_gbp": {},
               "consumer_parts_gbp": {il.CM_DESIGN_HOUSEHOLDS: -20.0}}
    row = il.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=-50.0, case_id="c")
    assert row["consumer_change_vs_private_gbp"] == 80.0 and row["three_party_identity_residual_gbp"] == 0.0
    combined = il.combine(parties, {"bill_change_system_gbp": 0.0, "investor_parts_gbp": {}, "resource_parts_gbp": {}})
    assert combined["consumer_parts_gbp"] == {il.CM_DESIGN_HOUSEHOLDS: -20.0}
    plain = il.three_party_row({k: v for k, v in parties.items() if k != "consumer_parts_gbp"}, vat_change_gbp=0.0,
                               exchequer_change_gbp=-50.0, case_id="c", tolerance=1e9)
    assert "consumer_change_vs_private_parts_gbp" not in plain


def test_bill_line_moves_carry_the_design_only_where_reported():
    from scripts.public_ownership import investor_ledger as il

    assert il.LINE_CM_DESIGN not in il.bill_line_moves({})
    assert il.bill_line_moves({"capacity_market_design_line_gbp_2024": -5.0})[il.LINE_CM_DESIGN] == -5.0


def test_the_party_flows_split_the_line_by_payee_and_refuse_a_gap():
    from scripts.public_ownership import party_flows as pf

    flows = pf.Flows("c", lambda gbp: gbp / 1e6)
    components = {"cm_design_state": -50.0, "cm_design_private": -10.0}
    parties = {"capacity_market_design_split_gbp": {"outside_register_other_change_gbp": -20.0,
                                                    "household_change_gbp": -20.0}}
    pf.cm_design_flows(flows, -100.0, components, parties)
    net = pf.net_by_party(flows.rows)
    assert net["exchequer"] == -50.0 and net["private_investors"] == -30.0 and net["consumers"] == 80.0
    with pytest.raises(ValueError, match="payees"):
        pf.cm_design_flows(pf.Flows("c", lambda g: g), -90.0, components, parties)


def test_register_components_carry_the_cm_move_for_each_owner():
    from scripts.public_ownership import party_flows as pf

    private = _rows("private", [0.0] * 4)
    case = _rows("public_ownership", [1.0, 0.5, 0.0, 1.0])
    case["capacity_market_revenue_gbp"] = [0.0, 0.5 * 0.8 * 5e5, 0.8 * 2.5e5, 0.0]
    case[design.COLUMN_STATE_ADJUSTMENT] = [0.0, -0.5 * 0.5 * 0.8 * 5e5, 0.0, 0.0]
    for frame in (private, case):
        for column in ("annual_capital_recovery_gbp", "annual_required_revenue_gbp"):
            frame[column] = 0.0
    parts = pf.register_components(pd.concat([private, case], ignore_index=True), "public_ownership")
    cm_c, cm_p = case["capacity_market_revenue_gbp"].sum(), private["capacity_market_revenue_gbp"].sum()
    assert parts["cm_design_state"] + parts["cm_design_private"] == pytest.approx(cm_c - cm_p)
    assert parts["cm_design_state"] == pytest.approx(-(1e6 + 0.5 * 5e5))       # the state's full CM revenue
    zero = pf.register_components(pd.concat([private, private.assign(case_id="x")], ignore_index=True), "x")
    assert zero["cm_design_state"] == 0.0 and zero["cm_design_private"] == 0.0


def test_the_flows_parts_are_exactly_the_registers_moves_for_each_owner():
    """The booking and the party flows agree: the state's part is its surplus change, the co-owners' the rest."""
    from scripts.public_ownership import party_flows as pf

    options = _options()
    specs = [s for s in _specs() if s.case_id in ("private", "public_ownership")]
    own = [1.0, 0.5, 0.25, 1.0]
    frame = pd.concat([_rows("private", [0.0] * 4), _rows("public_ownership", own)], ignore_index=True)
    frame["public_surplus_before_recycling_gbp"] = (frame["ownership_share"] * frame["producer_rent_gbp"]).astype(float)
    for column in ("annual_capital_recovery_gbp", "annual_required_revenue_gbp"):
        frame[column] = 0.0
    booked, _ = design.book(frame, options, specs)
    before = frame.loc[frame["case_id"].eq("public_ownership")].reset_index(drop=True)
    after = booked.loc[booked["case_id"].eq("public_ownership")].reset_index(drop=True)
    state = (after["public_surplus_before_recycling_gbp"] - before["public_surplus_before_recycling_gbp"]).sum()
    private = ((after["producer_rent_gbp"] - after["public_surplus_before_recycling_gbp"])
               - (before["producer_rent_gbp"] - before["public_surplus_before_recycling_gbp"])).sum()
    parts = pf.register_components(booked, "public_ownership")
    assert parts["cm_design_state"] == pytest.approx(state)
    assert parts["cm_design_private"] == pytest.approx(private)


# ── the acquisition tariff ───────────────────────────────────────────────────


def test_the_kept_line_is_todays_without_a_contract_and_the_contract_with_one():
    from scripts.public_ownership.nationalised_tariff import VALUE, capacity_market_not_acquired

    components = pd.DataFrame({"case_id": ["private"], "component_id": ["capacity_market"], VALUE: [40.0]})
    conv = {"factor": 1.04, "cm_household_allocation_factor": 1.348, "gsp_chargeable_volume_mwh": 2.5e8,
            "mwh": 3.1, "final_demand_mwh": 3e8, "eligible_mwh": 2.5e8}
    diagnostics = {"capacity_market_revenue_owned_costed_gbp": 1.2e9}
    book, ic = 3.44e9, 0.2e9
    system, detail = capacity_market_not_acquired(components, diagnostics, conv, book, ic)
    assert detail["capacity_market_kept_share"] == max(0.0, 1.0 - (1.2e9 + ic / 1.04) / (book / 1.04))
    _, contract = capacity_market_not_acquired(components, diagnostics, conv, book, ic, contract_cost_gbp=1.65e9)
    assert contract["capacity_market_kept_share"] == pytest.approx((1.65e9 - ic) / book)
    assert contract["capacity_market_owned_costed_model_year_gbp"] == 0.0


# ── the port onto the current tree (25 Oct 2026): scope, the 2030 mechanisms, the HT50 config ─────────


def test_the_design_is_refused_on_a_2030_register():
    with pytest.raises(ValueError, match="vintaged"):
        _run_year(_options(), 2030)


def test_the_design_never_shares_a_register_with_capacity_procurement():
    """F27 item 6 (a 2030 lever) re-prices the same CM receipts: a register takes one or the other."""
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios

    specs = _specs([{"capacity_procurement": ["cost_based_existing"]}])
    assert any(s.settings.get("capacity_procurement") == "cost_based_existing" for s in specs)
    with pytest.raises(ValueError, match="capacity_procurement"):
        calculate_financing_scenarios(_assets(), _costs(), _snapshot(), case_specs=specs, price_year=2024,
                                      state_operating_overhead=NO_OVERHEAD, capacity_market_design=_options())


def _run_year(options, year):
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios

    assets = _assets().assign(model_year=year)
    return calculate_financing_scenarios(assets, _costs(), _snapshot(), case_specs=_specs(), price_year=2024,
                                         state_operating_overhead=NO_OVERHEAD, capacity_market_design=options)


def test_the_ht50_study_names_the_design_and_its_two_sensitivities():
    from scripts.public_ownership.reference_runs import configured_case_specs
    from scripts.public_ownership.study_config import load_studies

    block = load_studies("config", validate=False).configs["FES25_HT_2050"]["public_ownership"]
    assert block[design.CONFIG_KEY] == {
        "central": CM_DESIGN_STATE_PLANNED,
        "fleet": "resources/market/capacity_market/cm_fleet_2050.csv",
        "summary": "resources/market/capacity_market/cm_calendar_2050_summary.json"}
    by = {s.case_id: s for s in configured_case_specs(block, "fes_ht_2050")}
    sensitivity = by["public_ownership@cm_market_participant"].settings
    assert sensitivity[LEVER_CM_DESIGN] == CM_DESIGN_MARKET_PARTICIPANT
    p3_alt = by["public_ownership@fully_nationalised_market_value_cm_market_participant"]
    assert p3_alt.settings[LEVER_CM_DESIGN] == CM_DESIGN_MARKET_PARTICIPANT and p3_alt.acquires_legacy()
    options = {"central": CM_DESIGN_STATE_PLANNED}
    assert design.resolve(by["public_ownership"].settings, options) == design.STATE_OUT
    assert design.resolve(by["public_ownership@fully_nationalised_market_value"].settings, options) \
        == design.ADEQUACY_PLANNING
    assert design.resolve(sensitivity, options) == design.AUCTION_ALL
    assert design.resolve(p3_alt.settings, options) == design.AUCTION_ALL
    # no 2050 case takes the 2030 CM lever, so the financing step's exclusivity check never bites there
    assert all(s.settings.get("capacity_procurement", "auction") == "auction" for s in by.values())


def test_f9_is_off_in_a_vintaged_register_so_it_never_meets_the_design():
    """F9 (the 2030/31 T-4 with state bids at cost) is switched off by the public account for a 2050 register."""
    import inspect

    from scripts.public_ownership import build_public_account

    source = inspect.getsource(build_public_account.write_public_account)
    assert "is_vintaged_model_year" in source and '"enabled": False' in source


def test_the_projector_row_carries_the_design_cost_change_on_the_cm_line():
    from scripts.public_ownership import project_public_ownership_consumer_bill as projector

    options = _options(factor=1.04)
    specs = [s for s in _specs() if s.case_id in ("private", "public_ownership")]
    frame = pd.concat([_rows("private", [0.0] * 4), _rows("public_ownership", [1.0, 0.5, 0.0, 1.0])], ignore_index=True)
    _, results = design.book(frame, options, specs)
    table = design.summary_columns(results, 1.04)
    extra = {"two_volume": False, "final_demand_mwh": 3.0e8, "factor": 1.04, "mwh": 3.1}
    rows = projector.capacity_market_design_row("public_ownership", table, private_extra=extra, cm_factor=1.348,
                                                account_source="x")
    moves = design.case_moves(table, "public_ownership")
    assert len(rows) == 1 and rows[0]["component_id"] == projector.LINE_CM_DESIGN
    back = projector._system(rows[0]["annual_gbp_per_3100kwh_ex_vat"], 3.0e8, 1.04, 3.1) / 1.348
    assert back == pytest.approx(moves["line_change_gbp"])
    assert projector.capacity_market_design_row("private", table, private_extra=extra, cm_factor=1.348,
                                                account_source="x") == []
    assert projector.LINE_CM_DESIGN not in projector.NEW_LINES


def test_the_new_bill_line_is_grouped_and_recovered_as_the_cm_line():
    from scripts.consumer_bill import bill_groupings
    from scripts.consumer_bill.line_recovery import INHERITED_LINE_RECOVERY

    assert INHERITED_LINE_RECOVERY["capacity_market_design_adjustment"] == "capacity_market"
    for view in bill_groupings.VIEWS:
        groups = bill_groupings.group_of(view)
        assert groups["capacity_market_design_adjustment"] == groups["capacity_market"]


def test_a_case_moving_only_the_design_carries_its_parents_earlier_purchase():
    """The P3 market-participant case has no 2030 twin; it carries the purchase it otherwise equals."""
    import pandas as pd
    from scripts.public_ownership.nationalised_tariff import carried_case_id

    settings = pd.DataFrame({
        "case_id": ["p3", "p3_cm", "p3_term"],
        "purchase_price_basis": ["market_value", "market_value", "market_value"],
        "acquisition_term_years": ["30", "30", "15"],
        "capacity_market_design": ["state_planned", "market_participant", "state_planned"],
    })
    carried = {"p3": {}}
    assert carried_case_id("p3", carried, settings) == "p3"
    assert carried_case_id("p3_cm", carried, settings) == "p3"
    assert carried_case_id("p3_term", carried, settings) is None   # a purchase lever moved: no carry
    assert carried_case_id("p3_cm", carried, None) is None
