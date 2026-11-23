"""Named sensitivity axes and the three axes built on them (plan items D1 to D4)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership.cfd_repricing import (
    ROUTE_CFD_BILL, build_strike_adjustments, repricing_settings, route_summary,
    savings_with_sensitivities,
)
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    LINE_CCS_BOOKED, LINE_CM_BOOKED, LINE_NEW_BUILD, NEW_LINES, axis_settings,
    booked_route_amounts, extra_lines,
)
from scripts.public_ownership.public_account import account_settings, build_public_account
from scripts.public_ownership.reference_runs import (
    configured_case_specs, expanded_cases, market_bill_cases, tariff_cases, validate_public_ownership_config,
)
from scripts.public_ownership.levers import (
    LEVERS, STRUCTURAL_LEVERS, parse_scenarios, parse_sensitivities,
)
from scripts.public_ownership.sensitivity_axes import (
    AXES, AXIS_CFD_STRIKE_BASIS, AXIS_NEW_BUILD_CONTRACTS, AXIS_OWNERSHIP_BOUNDARY, AXIS_ROUTE_BOOKING,
    AxisConfigError, CENTRAL_SETTINGS, SCOPE_ALL, comparator_case_id, comparator_for_settings,
    expand_cases, parse_axes,
)

from tests.unit.test_public_ownership_public_account import _axes, _settled_register
from config.config_loader import load_defaults


CASES = ("private", "public_coordination", "public_ownership")
NUMERIC = [
    "annual_required_revenue_gbp", "annual_capital_recovery_gbp", "annual_fixed_om_gbp",
    "annual_variable_om_gbp", "wacc", "ownership_share", "public_surplus_before_recycling_gbp",
]


@pytest.fixture(scope="module")
def defaults() -> dict:
    return load_defaults()


# ------------------------------------------------------------------------ D1


def test_every_declared_lever_value_is_one_the_code_implements(defaults):
    """House rule 11: an unsupported lever value fails at Snakemake parse time (item K1)."""
    config = defaults["public_ownership"]
    validate_public_ownership_config(config, defaults["consumer_bill"]["forecasts"]["cases"])
    bundles = parse_scenarios(config["scenarios"])
    assert list(bundles) == ["private", "public_control", "public_ownership"]
    for scenario, bundle in bundles.items():
        assert set(bundle) == set(LEVERS)
        for name, value in bundle.items():
            assert value in LEVERS[name].values, (scenario, name, value)
    sens = parse_sensitivities(config["sensitivities"], bundles)
    # Decision 5(B), 29 Sep 2026: best and worst bounds replace the bounding pair.
    assert sens["bounding_pair"] is False
    # 30 Sep 2026: consumer_first is every base (retired as a preset); todays_rules keeps
    # the old central incidence; P3's purchase on taxation is the one position the
    # retired preset took on P3.  P2 with state bids at cost has no grid (decision 6).
    assert set(sens["presets"]) == {"as_designed", "todays_rules", "competitive_neutrality", "treasury_first",
                                    "purchase_on_taxation"}
    assert set(sens["bounds"]["cases"]) == {"best", "worst"}
    assert {g for g in sens["toggle_grids"]["grids"]} == {
        "public_control", "public_ownership", "public_ownership@fully_nationalised_market_value"}
    assert [name for name, _ in sens["structural"]] == list(STRUCTURAL_LEVERS)
    for name, values in sens["structural"] + sens["one_at_a_time"]:
        assert set(values) <= set(LEVERS[name].values)
    # F29 (26 Oct 2026): the company model (v11-2's 6.2% commercial benchmark) is retired.
    assert {"fully_nationalised", "fully_nationalised_market_value_central_dispatch"} <= set(sens["named"])
    assert "company_model" not in sens["named"]


@pytest.mark.parametrize("block, message", [
    ({"axes": {"strike_price": {}}}, "unknown sensitivity axes"),
    ({"axes": {"route_booking": {"central": "always"}}}, "not implemented"),
    ({"axes": {"route_booking": {"alternatives": ["always"]}}}, "not implemented"),
    ({"axes": {"route_booking": {"alternatives": ["sensitivity_only"]}}}, "central value"),
    ({"axes": {"route_booking": {"alternatives": ["booked", "booked"]}}}, "repeats"),
    ({"axes": {"route_booking": {"label": "x"}}}, "unknown keys"),
    ({"bounding_pairs": True}, "unknown keys"),
])
def test_parse_axes_refuses_what_is_not_implemented(block, message):
    with pytest.raises(AxisConfigError, match=message):
        parse_axes(block)


def test_undeclared_axes_sit_at_their_central_value():
    axes, _ = parse_axes(None)
    assert {name: spec["central"] for name, spec in axes.items()} == CENTRAL_SETTINGS
    assert expand_cases(CASES, axes) == expand_cases(CASES, {})
    assert [spec.case_id for spec in expand_cases(CASES, axes)] == list(CASES)


def test_the_boundary_axis_alone_reproduces_the_five_cases_by_their_old_labels():
    specs = expand_cases(CASES, _axes(ownership_boundary=["ownership_all_cfd_repriced"]))
    assert [spec.case_id for spec in specs] == [
        "private", "public_coordination", "public_ownership",
        "public_coordination@ownership_all_cfd_repriced", "public_ownership@ownership_all_cfd_repriced",
    ]


def test_expansion_is_scenarios_then_one_at_a_time_then_named_then_a_bounding_pair(defaults):
    """Item K1: three scenarios, sensitivities underneath, one case list for every stage."""
    config = defaults["public_ownership"]
    specs = configured_case_specs(config)
    ids = [spec.case_id for spec in specs]
    assert ids[:3] == ["private", "public_control", "public_ownership"]
    assert ids == expanded_cases(config) and len(set(ids)) == len(ids)
    # A public_cases lever never varies the private baseline; an all_cases lever does.
    assert "private@strike_at_cost" not in ids and "private@routes_booked" not in ids
    assert {"private@new_build_at_cost", "private@ldes_floor_at_wacc", "private@wacc_fixed_tmr",
            "private@pay_as_bid_ceiling"} <= set(ids)
    # A lever is generated only where it bites.  F29 (26 Oct 2026): the lending margins and the
    # price of the state's systematic risk where the state finances at its cash rate (P2), not on
    # P1's stake rate; the construction premium, +1 pp, the 30-year gilt and the guarantee retired.
    # Meeting notes (user, 10 Nov 2026): the premium and gilt responses are Scenario 4 sensitivities only.
    assert {"public_ownership@public_rate_pwlb_standard", "public_ownership@public_rate_gilt_flat",
            "public_ownership@fully_nationalised_market_value_systematic_premium",
            "public_ownership@fully_nationalised_market_value_yield_b"} <= set(ids)
    # User (13 Nov 2026, results review): Scenario 2 is tested at 1% and 1.9% as named cases; still none on S1.
    assert {"public_ownership@systematic_premium_1pc", "public_ownership@systematic_premium_1pc9"} <= set(ids)
    assert not any(i.startswith("public_control@") and "systematic_premium" in i for i in ids)
    assert not any("@yield_b" in i for i in ids)
    assert "public_control@public_rate_pwlb_standard" not in ids and "private@public_rate_pwlb_standard" not in ids
    assert not any(t in i for i in ids for t in ("public_rate_nominal", "public_rate_plus_1pp", "@premium_none",
                                                  "guarantee_cost", "company_model"))
    # The wires are private in every base, so the networks sensitivity is the acquisition, on P1 and
    # P2 alike (P2b is public_ownership@networks_acquired); the guarantee is retired (F29).
    assert "public_control@networks_private" not in ids
    assert {"public_control@networks_acquired", "public_ownership@networks_acquired"} <= set(ids)
    assert not any("networks_guaranteed" in i for i in ids)
    # Named positions and their extensions.
    assert {"public_ownership@fully_nationalised",
            "public_ownership@fully_nationalised_market_value_central_dispatch"} <= set(ids)
    # Since 29 Sep 2026 there is no bounding pair: best and worst bounds per scenario.
    assert not any(i.endswith("@all_alternatives") for i in ids)
    assert {"public_control@best_case", "public_control@worst_case", "public_ownership@best_case",
            "public_ownership@worst_case", "public_ownership@fully_nationalised_market_value_best_case",
            "public_ownership@fully_nationalised_market_value_worst_case"} <= set(ids)
    # One-at-a-time cases move exactly one lever off their scenario; named ones may move several.
    for spec in specs:
        if spec.meta.get("kind") != "one_at_a_time":
            continue
        base = next(s for s in specs if s.case_id == spec.base_case)
        assert sum(spec.settings[n] != base.settings[n] for n in LEVERS) == 1, spec.case_id
    # Each destination is a partition of the list.
    market, tariff = market_bill_cases(config), tariff_cases(config)
    assert set(market).isdisjoint(tariff)
    assert all(c.startswith("public_ownership@") for c in tariff)
    # 61 before the 29 Sep 2026 review (presets, bounds, toggle grids), 210 after it, 972
    # after the 30 Sep 2026 rebase (the exact grids), 590 with P3's market-bill
    # treasury_first (user answer Q5) and no strike axis in P2's grid (497 grid cases); 593
    # with the 25% wire premium and 5,313 with the bound grids (2 Oct 2026); 5,322 with the
    # nine OW9 named cases (3 Oct 2026).
    # 4 Oct 2026: +6 auction margin, +2 self-insurance (F6, F7); 6 Oct 2026 (F19): +2 named P3 cases,
    # one of which the bound grid now lands on; +1 named (P3's owners' value at the new-build hurdle);
    # 7 Oct: +2 GB Energy (F10).  23 Oct: +2 pre-2025 CfD at owners' value (F27 item 1); 24 Oct: +1 one at a
    # time and +1 named, existing capacity on cost-based contracts on P2 and P3 (F27 item 6); +1 named, P3 with
    # legacy offshore wind at observed deal multiples (user decision, 24 Oct 2026).  26 Oct (F29, workbook
    # v2.0): the retired levers leave the catalogue and the grids shrink (P1 48, P2 128, P3 96; bound grids 24,
    # 144, 3,456), 3,967 in all (test_public_ownership_levers.py holds the breakdown).  F28 (27 Oct): +1 named,
    # P3 repaid over one 30-year term (the central repays each asset over its remaining life): 3,968.  Meeting
    # notes (user, 10 Nov 2026): the premium and gilt-response sensitivities on Scenario 4 only: 3,835.
    assert len(ids) == 3999  # 13 Nov 2026: +3 named risk-premium cases, S2's bound grid x 8; +3 named (10 Nov 2026): energy-from-waste bought, Hinkley left private, 2030 prices flat; +1 (11 Nov): today's prices


def test_the_private_bound_exists_only_when_two_all_cases_axes_move_it(defaults):
    """Private's bound is a real third point once more than one all_cases axis moves it.

    With a single all_cases axis the bound would repeat that axis's own variant,
    so it is not added. With two (new-build contracts and the storage floor
    basis) the combination is a point no one-at-a-time variant covers, and every
    case that moves both is measured against it.
    """
    one = {spec.case_id for spec in expand_cases(CASES, _axes(
        ownership_boundary=["ownership_all_cfd_repriced"], new_build_contracts=["at_cost"]))}
    assert "private@all_alternatives" not in one, "one all_cases axis: the bound repeats its variant"

    # Since F29 (26 Oct 2026) each bound moves one comparator lever: the best case the LDES floor
    # basis, the worst case the private WACC basis (workbook v2.0's Fixed TMR, the lower private
    # rate).  Each comparator is the point of a one-at-a-time private case, which the case is
    # measured against.
    specs = configured_case_specs(defaults["public_ownership"])
    ids = {spec.case_id for spec in specs}
    assert "private@all_alternatives" not in ids
    for comparator, lever, twin in (("private@best_case_comparator", "ldes_floor_basis", "private@ldes_floor_at_wacc"),
                                    ("private@worst_case_comparator", "private_wacc_basis", "private@wacc_fixed_tmr")):
        settings = next(s for s in specs if s.case_id == comparator).settings
        moved = [n for n in AXES if settings[n] != CENTRAL_SETTINGS[n]]
        assert moved == [lever] and AXES[lever].scope == SCOPE_ALL
        assert next(s for s in specs if s.case_id == twin).settings == settings

    assert comparator_case_id(
        next(s for s in specs if s.case_id == "public_ownership@best_case"), specs
    ) == "private@ldes_floor_at_wacc"
    assert comparator_case_id(
        next(s for s in specs if s.case_id == "public_ownership@worst_case"), specs
    ) == "private@wacc_fixed_tmr"
    assert comparator_case_id(
        next(s for s in specs if s.case_id == "public_ownership@strike_financing_saving"), specs
    ) == "private"
    assert comparator_case_id(
        next(s for s in specs if s.case_id == "public_ownership@ldes_floor_at_wacc"), specs
    ) == "private@ldes_floor_at_wacc"


def test_adding_an_axis_moves_no_number_on_the_cases_already_there():
    """The D1 invariance: a new axis adds cases, it does not disturb the old ones."""
    before, before_summary = _settled_register(_axes(ownership_boundary=["ownership_all_cfd_repriced"]))
    after, after_summary = _settled_register(_axes(
        ownership_boundary=["ownership_all_cfd_repriced"], cfd_strike_basis=["at_cost"],
        route_booking=["booked"], new_build_contracts=["at_cost"],
    ))
    shared = sorted(set(before.case_id))
    assert shared == sorted(set(before.case_id) & set(after.case_id))
    left = before.set_index(["case_id", "asset_id"]).sort_index()
    right = after.loc[after.case_id.isin(shared)].set_index(["case_id", "asset_id"]).sort_index()
    pd.testing.assert_frame_equal(left[NUMERIC], right[NUMERIC])
    pd.testing.assert_frame_equal(
        before_summary.set_index("case_id").sort_index(),
        after_summary.loc[after_summary.case_id.isin(shared)].set_index("case_id").sort_index(),
    )


def test_the_register_records_the_axis_point_of_every_case():
    register, _ = _settled_register(_axes(cfd_strike_basis=["at_cost"], route_booking=["booked"]))
    by_case = register.drop_duplicates("case_id").set_index("case_id")
    assert by_case.loc["public_ownership@strike_at_cost", AXIS_CFD_STRIKE_BASIS] == "at_cost"
    assert by_case.loc["public_ownership@strike_at_cost", AXIS_ROUTE_BOOKING] == "sensitivity_only"
    assert by_case.loc["public_ownership@routes_booked", AXIS_ROUTE_BOOKING] == "booked"
    assert by_case.loc["public_ownership", AXIS_OWNERSHIP_BOUNDARY] == "private"
    # Every lever is a column, the scenario bundle included.
    assert by_case.loc["public_ownership", "new_build_generation"] == "public"
    assert by_case.loc["public_coordination", "new_build_generation"] == "state_stake"
    assert by_case.loc["private", "new_build_generation"] == "private"


# ------------------------------------------------------------------------ D3


def _at_cost_over_life(register):
    """The @strike_at_cost cases on economic-life amortisation, as the P2 bundle holds them.

    Since 26 Sep 2026 the at-cost strike honours ``cfd_amortisation``; these tests
    pin the life-long rent, so the variant is put on the life basis explicitly.
    """
    register = register.copy()
    register.loc[register["case_id"].astype(str).str.endswith("@strike_at_cost"), "cfd_amortisation"] = "economic_life"
    return register


def test_the_at_cost_strike_basis_returns_the_whole_rent_not_only_the_financing_saving(defaults):
    register, summary = _settled_register(_axes(cfd_strike_basis=["at_cost"]))
    register = _at_cost_over_life(register)
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    savings = savings_with_sensitivities(register, settings)
    central = savings.loc[savings.case_id.eq("public_ownership")].set_index("asset_id")
    at_cost = savings.loc[savings.case_id.eq("public_ownership@strike_at_cost")].set_index("asset_id")

    # The re-priced contract gives up market plus support revenue less its own cost.
    ar7 = register.loc[register.case_id.eq("public_ownership@strike_at_cost")].set_index("asset_id").loc["ar7"]
    expected = (float(ar7.market_revenue_gbp) + float(ar7.support_revenue_gbp) - float(ar7.fuel_carbon_gbp)
                - float(ar7.annual_required_revenue_gbp))
    assert at_cost.loc["ar7", "route"] == ROUTE_CFD_BILL
    assert at_cost.loc["ar7", "saving_bill_term_gbp"] == pytest.approx(expected)
    assert at_cost.loc["ar7", "saving_bill_term_gbp"] > central.loc["ar7", "saving_bill_term_gbp"]
    # The resource saving is unchanged, so the route identity still closes and
    # the whole difference is declared as the term adjustment.
    assert at_cost.loc["ar7", "saving_economic_life_gbp"] == pytest.approx(
        central.loc["ar7", "saving_economic_life_gbp"])
    routes = route_summary(savings, summary)
    assert routes.identity_within_tolerance.all()
    r1 = routes.loc[routes.case_id.eq("public_ownership@strike_at_cost") & routes.route.eq(ROUTE_CFD_BILL)].iloc[0]
    assert r1.term_adjustment_gbp_2024 == pytest.approx(
        r1.bill_allocated_gbp_2024 - r1.saving_economic_life_gbp_2024)
    assert at_cost.loc["ar7", "method"].endswith(";strike_at_cost")

    # A 30% stake re-prices 30% of the contract, not all of it.
    coordination = savings.loc[savings.case_id.eq("public_coordination@strike_at_cost")].set_index("asset_id")
    share = float(register.loc[register.case_id.eq("public_coordination@strike_at_cost")]
                  .set_index("asset_id").loc["ar7", "ownership_share"])
    coordination_ar7 = register.loc[register.case_id.eq("public_coordination@strike_at_cost")].set_index("asset_id").loc["ar7"]
    assert share == pytest.approx(0.30)
    assert coordination.loc["ar7", "saving_bill_term_gbp"] == pytest.approx(
        share * (float(coordination_ar7.market_revenue_gbp) + float(coordination_ar7.support_revenue_gbp)
                 - float(coordination_ar7.fuel_carbon_gbp) - float(coordination_ar7.annual_required_revenue_gbp)))


def test_the_at_cost_strike_is_lower_and_still_settles_through_the_same_table(defaults, bill_table=None):
    from scripts.market.cfd_strike_table import load_strike_table
    import json

    assumptions = json.loads(
        Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json").read_text(encoding="utf-8")
    )
    table = load_strike_table(Path("data/market/lccc/cfd_strike_prices_by_round.csv"),
                              assumptions["cfd_strike_indexation"], 2024)
    register, _ = _settled_register(_axes(cfd_strike_basis=["at_cost"]))
    register = _at_cost_over_life(register)
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    savings = savings_with_sensitivities(register, settings)
    adjustments = build_strike_adjustments(savings, table, settings).set_index(["case_id", "strike_key"])
    central = adjustments.loc[("public_ownership", "AR7_offshore")]
    at_cost = adjustments.loc[("public_ownership@strike_at_cost", "AR7_offshore")]
    assert at_cost.delta_gbp_per_mwh_2024 > central.delta_gbp_per_mwh_2024 > 0
    assert 0 < at_cost.strike_case_gbp_per_mwh_2024 < central.strike_case_gbp_per_mwh_2024


def test_at_cost_refuses_a_register_without_the_settlement(defaults):
    register, _ = _settled_register(_axes(cfd_strike_basis=["at_cost"]))
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    with pytest.raises(ValueError, match="needs the WP-D settlement"):
        savings_with_sensitivities(register.drop(columns=["market_revenue_gbp", "support_revenue_gbp"]), settings)


# ------------------------------------------------------------------------ D2


def _account(defaults, axes):
    register, _ = _settled_register(axes)
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    savings = savings_with_sensitivities(register, settings)
    _, report = build_public_account(register, savings, account_settings(defaults["public_ownership"]))
    return report


def test_route_booking_puts_the_capacity_market_and_ccs_savings_on_the_bill(defaults):
    report = _account(defaults, _axes(route_booking=["booked"]))
    denominators = dict(factor=1.0, final_demand_mwh=100.0, eligible_mwh=50.0, mwh=3.1,
                        cm_line=1e9, ccs_line=1e9)
    central = booked_route_amounts(report["public_ownership"], **denominators)
    booked = booked_route_amounts(report["public_ownership@routes_booked"], **denominators)
    assert central == {}
    assert booked[LINE_CM_BOOKED] > 0 and booked[LINE_CCS_BOOKED] > 0

    rows = {r["component_id"]: r for r in extra_lines(
        "public_ownership@routes_booked", report["public_ownership@routes_booked"],
        account_source="test", **denominators)}
    assert rows[LINE_CM_BOOKED]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-booked[LINE_CM_BOOKED])
    assert rows[LINE_CM_BOOKED]["stack_group"] == "system_balancing"
    assert rows[LINE_CCS_BOOKED]["stack_group"] == "policy"
    # sensitivity_only leaves both lines off the bill entirely.
    unbooked = {r["component_id"] for r in extra_lines(
        "public_ownership", report["public_ownership"], account_source="test", **denominators)}
    assert LINE_CM_BOOKED not in unbooked and LINE_CCS_BOOKED not in unbooked


def test_a_booking_never_takes_more_off_a_line_than_the_line_holds(defaults):
    report = _account(defaults, _axes(route_booking=["booked"]))
    capped = booked_route_amounts(
        report["public_ownership@routes_booked"], factor=1.0, final_demand_mwh=100.0, eligible_mwh=50.0,
        mwh=3.1, cm_line=0.25, ccs_line=0.1,
    )
    assert capped[LINE_CM_BOOKED] == pytest.approx(0.25)
    assert capped[LINE_CCS_BOOKED] == pytest.approx(0.1)


# ------------------------------------------------------------------------ D4


def test_new_build_contracts_at_cost_bills_only_the_privately_owned_share(defaults):
    report = _account(defaults, _axes(new_build_contracts=["at_cost"]))
    for case_id in ("private", "public_coordination", "public_ownership"):
        variant = report[f"{case_id}@new_build_at_cost"]
        assert variant["new_build_contract_shortfall_gbp"] == pytest.approx(
            report[case_id]["new_build_contract_shortfall_gbp"])
    # Private owns none of it, so the whole shortfall is contracted on the bill;
    # under full ownership the surplus-recycling line already carries it.
    private = report["private@new_build_at_cost"]
    ownership = report["public_ownership@new_build_at_cost"]
    assert private["new_build_contract_bill_gbp"] == pytest.approx(private["new_build_contract_shortfall_gbp"])
    assert ownership["new_build_contract_bill_gbp"] == pytest.approx(0.0)
    coordination = report["public_coordination@new_build_at_cost"]
    assert coordination["new_build_contract_bill_gbp"] == pytest.approx(
        0.70 * coordination["new_build_contract_shortfall_gbp"])


def test_the_new_build_line_reaches_the_private_bill_too(defaults):
    report = _account(defaults, _axes(new_build_contracts=["at_cost"]))
    denominators = dict(factor=1.1, final_demand_mwh=100.0, eligible_mwh=50.0, mwh=3.1)
    rows = {r["component_id"]: r for r in extra_lines(
        "private@new_build_at_cost", report["private@new_build_at_cost"], account_source="test", **denominators)}
    expected = report["private@new_build_at_cost"]["new_build_contract_bill_gbp"] * 1.1 / 50.0 * 3.1
    assert rows[LINE_NEW_BUILD]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(expected)
    assert rows[LINE_NEW_BUILD]["stack_group"] == "policy"
    assert LINE_NEW_BUILD in NEW_LINES
    # The central private case carries no such line.
    central = {r["component_id"] for r in extra_lines(
        "private", report["private"], account_source="test", **denominators)}
    assert LINE_NEW_BUILD not in central


def test_the_account_records_the_axis_point_each_case_sits_at(defaults):
    report = _account(defaults, _axes(route_booking=["booked"], new_build_contracts=["at_cost"]))
    structural = {n: axis_settings(report["public_ownership"])[n] for n in STRUCTURAL_LEVERS}
    assert structural == {n: CENTRAL_SETTINGS[n] for n in STRUCTURAL_LEVERS}
    assert axis_settings(report["public_ownership"])["new_build_generation"] == "public"
    assert axis_settings(report["public_ownership@routes_booked"])[AXIS_ROUTE_BOOKING] == "booked"
    assert axis_settings(report["private@new_build_at_cost"])[AXIS_NEW_BUILD_CONTRACTS] == "at_cost"


def test_the_comparator_needs_every_all_cases_axis_to_agree():
    """A private case matching on one all_cases axis but not the other is the wrong column.

    The bill projector used to match on the new-build axis alone. With a second
    all_cases axis that silently compared a case against a baseline it did not
    share, so both it and the expansion now use comparator_for_settings.
    """
    settings = {
        "private": dict(CENTRAL_SETTINGS),
        "private@new_build_at_cost": dict(CENTRAL_SETTINGS, new_build_contracts="at_cost"),
        "private@ldes_floor_at_wacc": dict(CENTRAL_SETTINGS, ldes_floor_basis="technology_wacc"),
        "private@all_alternatives": dict(
            CENTRAL_SETTINGS, new_build_contracts="at_cost", ldes_floor_basis="technology_wacc"),
        "public_ownership": dict(CENTRAL_SETTINGS),
        "public_ownership@strike_at_cost": dict(CENTRAL_SETTINGS, cfd_strike_basis="at_cost"),
        "public_ownership@ldes_floor_at_wacc": dict(
            CENTRAL_SETTINGS, ldes_floor_basis="technology_wacc"),
        "public_ownership@all_alternatives": dict(
            CENTRAL_SETTINGS, new_build_contracts="at_cost", ldes_floor_basis="technology_wacc",
            cfd_strike_basis="at_cost"),
    }
    # A private case is its own comparator, whichever axes it moves.
    for case_id in [c for c in settings if c.startswith("private")]:
        assert comparator_for_settings(case_id, settings) == case_id

    # A public_cases axis does not change the column.
    assert comparator_for_settings("public_ownership@strike_at_cost", settings) == "private"
    # Each all_cases axis selects the private case at that same point.
    assert comparator_for_settings("public_ownership@ldes_floor_at_wacc", settings) \
        == "private@ldes_floor_at_wacc"
    # Both moved: only the private case with both moved will do.
    assert comparator_for_settings("public_ownership@all_alternatives", settings) \
        == "private@all_alternatives"
