"""Item K1: levers, scenario bundles, the sensitivity expansion and the lever-driven treatments."""

from pathlib import Path

import pytest
import yaml

from scripts.public_ownership.finance import (
    ownership_share_for_settings, public_legacy_rate, public_new_build_rate, rate_for_settings,
)
from scripts.public_ownership.levers import (
    DEFAULT_SETTINGS, LABELS, LEVERS, STRUCTURAL_LEVERS, LeverConfigError, expand_axis_cases, expand_cases,
    parse_axes, parse_scenarios, parse_sensitivities,
)
from config.config_loader import load_defaults


SCENARIOS = {
    "private": {},
    "public_control": {"new_build_generation": "state_stake", "public_return": "cepa_decomposition"},
    "public_ownership": {"new_build_generation": "public", "public_return": "public_rate"},
}


def _snapshot():
    """Schema 4 (workbook v2.0, F29) in miniature."""
    return {
        "schema_version": 4,
        "assumptions": {"public_borrowing_real": {"value": 0.02709}},
        "public_rate_by_lending_margin": {"nlf": {"value": 0.02709}, "pwlb_hra": {"value": 0.02999},
                                          "pwlb_certainty": {"value": 0.03399}, "pwlb_standard": {"value": 0.03599},
                                          "gilt_flat": {"value": 0.02599}},
        "sovereign_yield_shift_2030": {"Sc2": {"B": {"value": 0.0005}}, "Sc3": {"B": {"value": 0.0018}},
                                       "Sc4": {"B": {"value": 0.0030}}},
    }


ROW = {
    "technology": "Offshore wind", "risk_rating": "M",
    "wacc": {"private": 0.0848, "public_coordination": 0.0711, "public_ownership_new_build": 0.02709,
             "public_ownership_legacy": 0.02709},
    "wacc_by_basis": {"notional_gearing_shielded": {"private": 0.0848, "public_coordination": 0.0711},
                      "fixed_tmr_shielded": {"private": 0.0761, "public_coordination": 0.0650}},
    "economic_rate_by_premium": {"social_flat_19": 0.04609, "none": 0.02709, "market_erp": 0.0741},
}


def _s(**levers):
    return dict(DEFAULT_SETTINGS, **levers)


# ── the catalogue ────────────────────────────────────────────────────────────


def test_every_label_is_unique_and_every_default_is_the_first_value():
    assert len(LABELS) == sum(len(lever.values) for lever in LEVERS.values())
    for name, lever in LEVERS.items():
        assert DEFAULT_SETTINGS[name] == lever.values[0]
    assert set(STRUCTURAL_LEVERS) <= set(LEVERS)


def test_a_lever_bites_only_where_it_can_move_a_number():
    private, control, ownership = (parse_scenarios(SCENARIOS)[k] for k in ("private", "public_control", "public_ownership"))
    # A public lever never moves the private baseline; an all_cases lever does.
    assert not LEVERS["cfd_strike_basis"].applies("at_cost", private)
    assert LEVERS["cfd_strike_basis"].applies("at_cost", ownership)
    assert LEVERS["new_build_contracts"].applies("at_cost", private)
    # F29: the social price of the state's risk and the lending margin need the public rate;
    # a stake earns the workbook's Sc2 rate, which uses neither.
    # Meeting notes (user, 10 Nov 2026): the central premium is none (gilts alone); the 1.9% is the alternative.
    for lever, value in (("systematic_premium", "social_flat_19"), ("public_rate_basis", "pwlb_standard")):
        assert LEVERS[lever].applies(value, ownership)
        assert not LEVERS[lever].applies(value, control)
    # The wires are bought or left private in every public scenario (F29 retired the guarantee);
    # buying them never touches the private baseline.
    assert LEVERS["networks"].values == ("private", "acquired")
    assert LEVERS["networks"].applies("acquired", control) and LEVERS["networks"].applies("acquired", ownership)
    assert not LEVERS["networks"].applies("acquired", private)
    # The pay-as-bid ceiling is a private-only bound; acquisition terms need an acquiring bundle.
    assert LEVERS["wholesale_pricing"].applies("pay_as_bid_ceiling", private)
    assert not LEVERS["wholesale_pricing"].applies("pay_as_bid_ceiling", ownership)
    assert not LEVERS["purchase_price_basis"].applies("market_value", ownership)
    assert LEVERS["purchase_price_basis"].applies("market_value", dict(ownership, legacy_fleet="acquire_all"))


# ── the declaration ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("block, message", [
    ({}, "at least"),
    ({"public_ownership": {}}, "private"),
    ({"private": {"new_build_generation": "public"}}, "baseline"),
    ({"private": {}, "x@y": {}}, "may not contain"),
    ({"private": {}, "s": {"unknown_lever": "x"}}, "unknown levers"),
    ({"private": {}, "s": {"legacy_fleet": "everything"}}, "not implemented"),
])
def test_parse_scenarios_refuses_what_is_not_implemented(block, message):
    with pytest.raises(LeverConfigError, match=message):
        parse_scenarios(block)


@pytest.mark.parametrize("block, message", [
    ({"grid": True}, "unknown keys"),
    ({"structural": ["gilt"]}, "unknown lever"),
    ({"one_at_a_time": [{"legacy_fleet": ["buy"]}]}, "not implemented"),
    ({"structural": ["route_booking"], "one_at_a_time": ["route_booking"]}, "twice"),
    ({"named": {"x": {"legacy_fleet": "acquire_all"}}}, "declared scenario"),
    ({"named": {"x": {"scenario": "public_ownership"}}}, "moves no lever"),
    ({"named": {"x": {"extends": "y", "legacy_fleet": "acquire_all"}}}, "declared before"),
    ({"named": {"strike_at_cost": {"scenario": "public_ownership", "legacy_fleet": "acquire_all"}}}, "collides"),
])
def test_parse_sensitivities_refuses_what_is_not_implemented(block, message):
    bundles = parse_scenarios(SCENARIOS)
    with pytest.raises(LeverConfigError, match=message):
        parse_sensitivities(block, bundles)


def test_expansion_is_scenarios_then_one_at_a_time_then_named_then_the_bound():
    bundles = parse_scenarios(SCENARIOS)
    sens = parse_sensitivities({
        "structural": ["cfd_strike_basis", "new_build_contracts"],
        "one_at_a_time": ["legacy_fleet", {"networks": ["acquired"]}],
        "named": {
            "fully_nationalised": {"scenario": "public_ownership", "legacy_fleet": "acquire_all", "networks": "acquired"},
            "fully_nationalised_15y": {"extends": "fully_nationalised", "acquisition_term_years": "15"},
        },
    }, bundles)
    ids = [spec.case_id for spec in expand_cases(bundles, sens)]
    assert ids == [
        "private", "public_control", "public_ownership",
        "public_control@strike_at_cost", "public_ownership@strike_at_cost",
        "public_control@strike_auction", "public_ownership@strike_auction",
        "private@new_build_at_cost", "public_control@new_build_at_cost", "public_ownership@new_build_at_cost",
        "public_ownership@legacy_acquired", "public_ownership@legacy_and_thermal_acquired",
        # 30 Sep 2026: the acquired wires bite on the stake too (user decision 3).
        "public_control@networks_acquired", "public_ownership@networks_acquired",
        "public_ownership@fully_nationalised", "public_ownership@fully_nationalised_15y",
        "public_control@all_alternatives", "public_ownership@all_alternatives",
    ]
    specs = {spec.case_id: spec for spec in expand_cases(bundles, sens)}
    # extends inherits the parent's settings and moves one more lever.
    child, parent = specs["public_ownership@fully_nationalised_15y"], specs["public_ownership@fully_nationalised"]
    assert {k for k in LEVERS if child.settings[k] != parent.settings[k]} == {"acquisition_term_years"}
    assert not parent.market_bill() and parent.acquires_legacy()
    assert specs["public_ownership@legacy_acquired"].register_key() != specs["public_ownership"].register_key()
    # A bill-stage lever does not change the register key.
    assert specs["public_ownership@strike_at_cost"].register_key() == specs["public_ownership"].register_key()
    # No private bound: only one structural all_cases lever moves it, so the bound would repeat a variant.
    assert "private@all_alternatives" not in ids


def test_the_item_d1_declaration_still_expands_to_its_twenty_one_cases():
    axes, bounding = parse_axes({"axes": {
        "ownership_boundary": {"alternatives": ["ownership_all_cfd_repriced"]},
        "cfd_strike_basis": {"alternatives": ["at_cost"]}, "route_booking": {"alternatives": ["booked"]},
        "new_build_contracts": {"alternatives": ["at_cost"]}, "ldes_floor_basis": {"alternatives": ["technology_wacc"]},
        "cfd_amortisation": {"alternatives": ["economic_life"]}}})
    specs = expand_axis_cases(["private", "public_coordination", "public_ownership", "public_nationalised"], axes,
                              bounding_pair=bounding)
    ids = [s.case_id for s in specs]
    assert len(ids) == 21 and "public_nationalised" in ids and not any(i.startswith("public_nationalised@") for i in ids)
    assert "public_coordination@ownership_all_cfd_repriced" in ids and "private@all_alternatives" in ids
    nationalised = next(s for s in specs if s.case_id == "public_nationalised")
    assert nationalised.acquires_legacy() and nationalised.settings["pre_2025_cfd_capacity"] == "public_repriced"


# ── the treatments ───────────────────────────────────────────────────────────


def test_rates_follow_the_levers_not_the_case_name():
    from scripts.public_ownership.finance import systematic_premium

    snapshot = _snapshot()
    # Unowned rows are private at the selected ERP convention.
    assert rate_for_settings(ROW, snapshot, _s(), finance_vintage="new_build", owned=False) == 0.0848
    assert rate_for_settings(ROW, snapshot, _s(private_wacc_basis="fixed_tmr_shielded"), finance_vintage="legacy",
                             owned=False) == 0.0761
    # F29: state-owned new build earns the cash rate (gilt + lending margin), no construction premium.
    own = _s(new_build_generation="public", public_return="public_rate")
    assert public_new_build_rate(ROW, snapshot, own) == pytest.approx(0.02709)
    assert public_new_build_rate(ROW, snapshot, dict(own, public_rate_basis="pwlb_standard")) == pytest.approx(0.03599)
    assert public_new_build_rate(ROW, snapshot, dict(own, public_rate_basis="gilt_flat")) == pytest.approx(0.02599)
    for retired in ("nominal", "real_plus_1pp", "gilt_30y_real"):
        with pytest.raises(ValueError, match="retired"):
            public_new_build_rate(ROW, snapshot, dict(own, public_rate_basis=retired))
    # Yield path B on Sc3 for the state developer; C is blank in the workbook and retired.
    assert public_new_build_rate(ROW, snapshot, dict(own, sovereign_yield="B")) == pytest.approx(0.02709 + 0.0018)
    # The risk the state holds is priced off the rate: economic rate less the cash rate.
    # Meeting notes (user, 10 Nov 2026): the central is gilts alone, no premium; the 1.9% is Scenario 4's sensitivity.
    assert systematic_premium(ROW, own) == 0.0
    assert systematic_premium(ROW, dict(own, systematic_premium="social_flat_19")) == pytest.approx(0.04609 - 0.02709)
    assert systematic_premium(ROW, dict(own, systematic_premium="none")) == 0.0
    assert systematic_premium(ROW, dict(own, systematic_premium="market_erp")) == pytest.approx(0.0741 - 0.02709)
    # A stake earns the Scenario 2 rate on the same convention, shifted on the Scenario 2 path.
    stake = _s(new_build_generation="state_stake", public_return="cepa_decomposition")
    assert public_new_build_rate(ROW, snapshot, stake) == 0.0711
    assert public_new_build_rate(ROW, snapshot, dict(stake, private_wacc_basis="fixed_tmr_shielded")) == 0.0650
    assert public_new_build_rate(ROW, snapshot, dict(stake, sovereign_yield="B")) == pytest.approx(0.0711 + 0.0005)
    # The company model (F29) and explicit (OW9, 3 Oct 2026) are retired.
    for retired in ("commercial_benchmark", "explicit"):
        with pytest.raises(ValueError, match="public_return"):
            public_new_build_rate(ROW, snapshot, _s(new_build_generation="public", public_return=retired))
    # Acquired legacy plant is refinanced at the cash rate, on Sc4's gilt path when the fleet is bought.
    acquired = dict(own, legacy_fleet="acquire_all")
    assert public_legacy_rate(snapshot, acquired) == pytest.approx(0.02709)
    assert public_legacy_rate(snapshot, dict(acquired, public_rate_basis="pwlb_standard", sovereign_yield="B")) == (
        pytest.approx(0.03599 + 0.0030))
    assert rate_for_settings(ROW, snapshot, acquired, finance_vintage="legacy", owned=True) == pytest.approx(0.02709)


def test_ownership_share_follows_the_new_build_lever():
    assert ownership_share_for_settings(_s(), finance_vintage="new_build", owned=False) == 0.0
    assert ownership_share_for_settings(_s(new_build_generation="state_stake"), finance_vintage="new_build", owned=True, configured_share=0.3) == 0.3
    assert ownership_share_for_settings(_s(new_build_generation="public"), finance_vintage="new_build", owned=True) == 1.0
    assert ownership_share_for_settings(_s(new_build_generation="public"), finance_vintage="legacy", owned=True) == 1.0
    with pytest.raises(ValueError, match="contradiction"):
        ownership_share_for_settings(_s(), finance_vintage="new_build", owned=True)


def test_repository_config_declares_three_scenarios_and_the_review_case_list():
    """Review 29 Sep 2026: 61 cases became 193 (210 after the second round); 972 on 30 Sep 2026.

    The one-at-a-time list lost the eligibility lever's two cases and the nominal
    rate's two, and gained the +1 pp and gilt rates, the guarantee's cost and
    pass-through, and the two background switches (legacy levies to taxation,
    BECCS removal on bills) on every scenario; three P3 named cases were added;
    the three presets beyond as_designed ran on four parents (12 cases); the best
    and worst bounds replaced @all_alternatives (6 cases and one private
    comparator); and the toggle grids added 2^4 + 2^5 + 2^6 = 112 combinations, of
    which the three parents and six existing cases are aliased rather than repeated.
    """
    from scripts.public_ownership.reference_runs import expanded_cases, institution_cases, market_bill_cases, tariff_cases

    config = load_defaults()["public_ownership"]
    assert list(config["scenarios"]) == ["private", "public_control", "public_ownership"]
    ids = expanded_cases(config)
    # Second round (29 Sep 2026): +2 named (the wires at RAV by statute on P2b and P3) and
    # +15 grid (P2's toggles, less half delivery, on P2's own bid stack): 210.
    # 30 Sep 2026 (the rebase): the exact grids hold 192 + 192 + 128 positions (P2 has no
    # strike axis: a 100% owner has no co-owner, so every strike rule gives the same bill),
    # of which the three parents and twelve existing cases are memberships (497 grid cases);
    # one at a time loses networks_private (now the base), the guarantee's cost and
    # pass-through (no base guarantees the wires) and gains the guarantee on P1 and P2
    # and the acquired wires on P1 (48); named gains the guarantee's cost on P1 (23);
    # presets lose consumer_first (4 cases, now the base) and gain todays_rules on P1, P2
    # and P2 with state bids and purchase_on_taxation on P3 (12; P3's treasury_first is
    # the market-bill tariff since the user's answer Q5).
    # 2 Oct 2026 (user request): +3 named, the wires bought at a 25% premium over RAV on P1,
    # P2 and P3 (26 named, 593 cases); then the bound grids (every best and worst
    # assumption switched one by one): 48 + 96 + 4,608 positions, of which 38 + 85 +
    # 4,597 are new cases (5,313).  3 Oct 2026 (OW9): +9 named, one per lever value no case
    # used (P2's private storage and four recycling rules, P1's guarantee pass-through low
    # and fee on bills, P2b's low premium and retained return): 5,322.  4 Oct 2026 (F6, F7):
    # +6 one at a time for the auction margin (private, P1, P2 at none and high) and +2 for
    # P2's self-insurance (none, high): 5,330.  6 Oct 2026 (F19 P3 decisions 2 and 3): +2 named
    # (P3 with thermal at book value, P3 on the model-price path); the bound grid's thermal switch
    # now goes to book value, and alone it lands on the first of them: 5,331.  F19 (6 Oct 2026):
    # +1 named, P3 with the owners' value at the new-build hurdle rate: 5,332.  F10: +2 named (GB
    # Energy as announced and its partnership twin): 5,334.  F27 item 1 (23 Oct 2026): +2 one at a time, P1 and P2
    # with the pre-2025 CfD plant bought at its owners' value: 5,336.  F27 item 6 (24 Oct 2026): +1 one at a
    # time (P2 with existing capacity on cost-based contracts) and +1 named (the same on P3): 5,338.  User
    # decision (24 Oct 2026): +1 named, P3 with legacy offshore wind at observed deal multiples: 5,339.  F29
    # (26 Oct 2026, workbook v2.0): the guarantee, the +1 pp and gilt rates, the construction premium and yield C
    # go and the grids shrink (P1 48, P2 128, P3 96; bound grids 24, 144 and 3,456): 3,967.  F28 (27 Oct 2026):
    # +1 named, P3 repaid over one 30-year term (the central is each asset over its remaining life): 3,968.
    # Meeting notes (user, 10 Nov 2026): the premium and gilt sensitivities on Scenario 4 only (+2 named, P3 with
    # the premium and with yield path B; -6 one at a time on P1 and P2; the P1 and P2 bound grids lose their
    # premium and yield axes, 24 -> 12 and 144 -> 24): 3,835 (one at a time -6, named +2, grid -129).
    # User decisions (10 Nov 2026): +2 named on Scenario 4, energy-from-waste bought and Hinkley left private: 3,837.
    # +1 named (10 Nov 2026): the fleet valued on 2030 prices held flat: 3,838.
    # +1 named (user, 11 Nov 2026): the fleet valued on today's prices held for its life: 3,839.
    # 13 Nov 2026 (results review): +3 named risk-premium cases, S2's bound grid x 8 (the premium and two delivery
    # switches), less positions other cases already carry.
    assert len(ids) == 3999 == len(market_bill_cases(config)) + len(tariff_cases(config)) + len(institution_cases(config))
    assert institution_cases(config) == ["public_ownership@fully_nationalised_market_value_central_dispatch"]
    g3 = [c for c in ids if "contracts_cancelled" in c]
    assert len(g3) == 4 and set(g3) <= set(tariff_cases(config))
    assert not any(c.endswith("@all_alternatives") or "public_rate_nominal" in c or "commissioning_vintage" in c
                   for c in ids)
    _, specs = _repository_specs()
    kinds = {}
    for spec in specs.values():
        kinds[spec.meta.get("kind")] = kinds.get(spec.meta.get("kind"), 0) + 1
    # 4 Oct 2026 (F6, F7): one at a time 47 -> 55 (auction margin 6, self-insurance 2).  F27 item 1: 57.
    # F27 item 6: one at a time 58, named 42.  Offshore at deal multiples (24 Oct 2026): named 43.
    # F29 (26 Oct 2026): one at a time 53 (the lending margins and the social premium replace the
    # +1 pp, gilt, construction-premium, yield C, CEPA and uniform-gearing and guarantee cases);
    # named 37 (the company model, the four guarantee cases, thermal at book value and the hurdle
    # rate go; P3 without support and with RO pass-through come); the bound comparators are
    # private at Fixed TMR, at the LDES technology rate and at both.
    # Meeting notes (user, 10 Nov 2026): one at a time 47 (P2's four premium values and yield B on P1 and P2 go),
    # named 40 (P3 with the premium and with yield B), grid 3,724 (the P1 and P2 bound grids lose those axes).
    # User (11 Nov 2026): named 44 (P3 with the fleet valued on today's prices).
    # User (13 Nov 2026, results review): named 47 (the premium at 1% on S3 and at 1% and 1.9% on S2), grid 3,881
    # (S2's bound grid gains the premium and the two delivery switches).
    assert kinds == {"scenario": 3, "one_at_a_time": 47, "named": 47, "preset": 12, "bound": 6,
                     "bound_comparator": 3, "grid": 3881}


def test_a_private_side_lever_is_measured_against_the_private_case_at_the_same_point():
    """House rule 3 in the register: private@wacc_cepa_vanilla is the baseline for public_control@wacc_cepa_vanilla."""
    import pandas as pd
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
    from tests.unit.test_public_ownership_finance import _costs

    assets = pd.DataFrame([{
        "model_year": 2030, "component_type": "Generator", "asset_id": "new-solar", "carrier": "solar_pv",
        "finance_technology": "Solar PV", "finance_vintage": "new_build", "capacity_mw": 1.0, "dispatch_mwh": 1_000.0,
        "evidence_status": "resolved", "commissioning_year": 2027, "support_type": "merchant",
    }, {
        "model_year": 2030, "component_type": "Generator", "asset_id": "old-solar", "carrier": "solar_pv",
        "finance_technology": "Solar PV", "finance_vintage": "legacy", "capacity_mw": 1.0, "dispatch_mwh": 1_000.0,
        "evidence_status": "resolved", "commissioning_year": 2015, "support_type": "merchant",
    }])
    snapshot = {"rate_basis": "real_CPI", "assumptions": {"public_coordination_state_equity_share": {"value": 0.3}},
                "technologies": [{**ROW, "technology": "Solar PV", "risk_rating": "L-M"}]}
    sens = {"structural": [], "one_at_a_time": [("private_wacc_basis", ["fixed_tmr_shielded"])], "named": {},
            "bounding_pair": False}
    _, totals = calculate_financing_scenarios(assets, _costs(), snapshot, scenarios=SCENARIOS, sensitivities=sens, price_year=2024)
    totals = totals.set_index("case_id")
    assert totals.loc["public_control@wacc_fixed_tmr", "comparator_case_id"] == "private@wacc_fixed_tmr"
    assert totals.loc["public_control", "comparator_case_id"] == "private"
    # The private variant costs less than central private (Fixed TMR's lower WACC), yet its own saving is zero.
    assert totals.loc["private@wacc_fixed_tmr", "annual_required_revenue_gbp"] < totals.loc["private", "annual_required_revenue_gbp"]
    assert totals.loc["private@wacc_fixed_tmr", "annual_financing_saving_vs_private_gbp"] == 0.0
    # And the control variant's saving is against that cheaper baseline, still positive.
    assert totals.loc["public_control@wacc_fixed_tmr", "annual_financing_saving_vs_private_gbp"] > 0


# ── Policy propositions (22 September 2026) ──────────────────────────────────
#: The bundles that go to a policy audience, and the scenario each sits under.
#: P3 is `fully_nationalised_market_value`, which predates the propositions:
#: the contract-form levers are inert once the state has bought the
#: counterparty, so it declares no bundle of its own.
PROPOSITIONS = {
    # P1 and P2 ARE the public scenarios since item L2, so the headline a policy
    # audience reads is the scenario's own number, not a sensitivity's.
    "public_control": "public_control",                                  # P1  GB Energy as legislated
    "public_control@gbe_value_to_bills": "public_control",               # P1b value returned to bills
    "public_ownership": "public_ownership",                              # P2  State developer
    "public_ownership@networks_acquired": "public_ownership",            # P2b and the wires
    "public_ownership@fully_nationalised_market_value": "public_ownership",  # P3 Public power system
}
#: The propositions still recovered through the market bill, where the CfD
#: contract form and the route booking are live choices rather than dissolved
#: into the acquisition tariff.
MARKET_BILL_PROPOSITIONS = sorted(p for p in PROPOSITIONS if "nationalised" not in p)


def _repository_specs():
    from scripts.public_ownership.reference_runs import configured_case_specs

    config = load_defaults()["public_ownership"]
    return config, {spec.case_id: spec for spec in configured_case_specs(config)}


def test_the_policy_propositions_are_declared():
    _, specs = _repository_specs()
    missing = sorted(set(PROPOSITIONS) - set(specs))
    assert not missing, f"the proposition bundles are not in the config: {missing}"
    for case_id, scenario in PROPOSITIONS.items():
        assert specs[case_id].base_case == scenario


@pytest.mark.parametrize("case_id", sorted(PROPOSITIONS))
def test_a_proposition_pairs_the_surplus_rule_with_who_carries_the_shortfall(case_id):
    """Coherence rule (ii): whoever takes the surplus carries the shortfall.

    ``retained_by_treasury`` with a bill-borne shortfall is heads-the-Treasury-
    wins, tails-the-bill-pays, and is not a policy anyone would legislate.
    """
    _, specs = _repository_specs()
    settings = specs[case_id].settings
    if settings["surplus_recycling"] == "retained_by_treasury":
        assert settings["shortfall_funding"] == "exchequer", case_id
    if settings["surplus_recycling"] == "public_cfd_at_cost":
        assert settings["shortfall_funding"] == "levy_on_eligible_supply", case_id


@pytest.mark.parametrize("case_id", MARKET_BILL_PROPOSITIONS)
def test_a_proposition_takes_the_contract_form_its_ownership_implies(case_id):
    """Coherence rule (i): outright ownership contracts at cost over the asset's life.

    A minority stake keeps the auction-cleared strike over the contract term,
    because there is a private majority owner on a 15-year CfD.  Only the
    market-bill propositions are covered: see the test below for why the rule
    has nothing to say once the state has bought the counterparty.
    """
    _, specs = _repository_specs()
    settings = specs[case_id].settings
    if settings["new_build_generation"] == "public":
        assert settings["cfd_strike_basis"] == "at_cost", case_id
        assert settings["cfd_amortisation"] == "economic_life", case_id
    else:
        assert settings["cfd_strike_basis"] == "financing_saving_only", case_id
        assert settings["cfd_amortisation"] == "contract_term", case_id


@pytest.mark.parametrize("case_id", MARKET_BILL_PROPOSITIONS)
def test_a_proposition_books_the_routes_it_owns(case_id):
    """Coherence rule (iii): the Capacity Market and CCS savings are bill lines."""
    _, specs = _repository_specs()
    assert specs[case_id].settings["route_booking"] == "booked", case_id


def test_the_contract_form_levers_are_inert_once_the_state_buys_the_counterparty():
    """Why P3 declares no bundle of its own (checked 22 September 2026).

    Moving ``cfd_strike_basis``, ``cfd_amortisation`` and ``route_booking`` on
    top of ``fully_nationalised_market_value`` leaves the required revenue and
    the tariff unchanged: with ``legacy_fleet: acquire_all`` there is no strike
    left to re-price and no route left to book, because the acquisition tariff
    replaces the recovery those levers describe.  Declaring them would imply a
    design choice that does not exist.  If this ever starts to bite, the tariff
    has grown a dependency and P3 needs revisiting.
    """
    from scripts.public_ownership.levers import CaseSpec

    _, specs = _repository_specs()
    base = specs["public_ownership@fully_nationalised_market_value"]
    assert base.settings["legacy_fleet"] == "acquire_all"
    # Since item L2 P3 inherits the contract form from P2, so the probe runs the
    # other way: take the three levers back to their private-baseline values and
    # the tariff still does not move.
    assert (base.settings["cfd_strike_basis"], base.settings["cfd_amortisation"],
            base.settings["route_booking"]) == ("at_cost", "economic_life", "booked")
    stripped = CaseSpec(
        "probe", base.base_case,
        dict(base.settings, cfd_strike_basis="financing_saving_only", cfd_amortisation="contract_term",
             route_booking="sensitivity_only"),
    )
    # The register is computed once per register-lever point; these three are
    # not register levers, so the two cases share every required-revenue number.
    assert stripped.register_key() == base.register_key()
    assert not base.market_bill() and not stripped.market_bill()


def test_item_l2_preserves_the_pre_promotion_scenario_bundles():
    """Promoting the propositions to scenarios loses nothing.

    ``@conservative_stake`` and ``@conservative_developer`` are exactly what
    ``public_control`` and ``public_ownership`` were before 22 September 2026, so
    the 21 September numbers (955.58 / 956.74 and 944.66 / 943.99) still have a
    case to sit on.
    """
    _, specs = _repository_specs()
    pre_l2 = {
        "public_control@conservative_stake": {
            "new_build_generation": "state_stake", "public_return": "cepa_decomposition",
            "surplus_recycling": "public_cfd_at_cost", "shortfall_funding": "levy_on_eligible_supply"},
        "public_ownership@conservative_developer": {
            "new_build_generation": "public", "public_return": "public_rate",
            "public_rate_basis": "real_cpi",
            "surplus_recycling": "public_cfd_at_cost", "shortfall_funding": "levy_on_eligible_supply"},
    }
    for case_id, bundle in pre_l2.items():
        assert specs[case_id].settings == _s(**bundle), case_id


def test_the_bounds_move_only_assumptions_and_keep_the_central_incidence():
    """Decision 5(B): best and worst replace ``@all_alternatives``.

    A bound moves assumption levers only (ranges on costs, rates, prices and
    lives), never an incidence or structural lever, so each scenario's best and
    worst answer "how wrong could the central be", not "what else could the
    policy be".  F29: the P3 worst case takes the PWLB standard margin and not yield B.
    """
    from scripts.public_ownership.levers import ASSUMPTION_LEVERS, BOUND_COMPARATOR_LEVERS

    _, specs = _repository_specs()
    movable = set(ASSUMPTION_LEVERS) | set(BOUND_COMPARATOR_LEVERS)
    for parent in ("public_control", "public_ownership", "public_ownership@fully_nationalised_market_value"):
        for side in ("best_case", "worst_case"):
            bound = specs[f"{parent}_{side}" if "@" in parent else f"{parent}@{side}"]
            moved = {n for n in LEVERS if bound.settings[n] != specs[parent].settings[n]}
            assert moved and moved <= movable, (bound.case_id, sorted(moved - movable))
    p3_worst = specs["public_ownership@fully_nationalised_market_value_worst_case"]
    assert p3_worst.settings["sovereign_yield"] == "A" and p3_worst.settings["public_rate_basis"] == "pwlb_standard"
    assert p3_worst.settings["systematic_premium"] == "market_erp"
    # Meeting notes (user, 10 Nov 2026): no gilt-response bound on Scenarios 2 and 3. Since 13 Nov 2026 (user,
    # results review) Scenario 2's worst case prices the risk the state takes on, and the state's delivery risks
    # (finding 7) are in both worst cases.
    s2_worst = specs["public_ownership@worst_case"].settings
    assert s2_worst["sovereign_yield"] == "A" and s2_worst["systematic_premium"] == "market_erp"
    for worst in (s2_worst, p3_worst.settings):
        assert worst["state_capex_overrun"] == "plus_15pct" and worst["state_om_premium"] == "plus_10pct"
    assert specs["public_control@worst_case"].settings["systematic_premium"] == "none"
    # A comparator lever in a bound gets its private comparator at the same point.
    assert "private@best_case_comparator" in specs


def test_each_preset_moves_only_who_pays():
    """Decision 5(A): a preset moves who pays, never what the system costs."""
    from scripts.public_ownership.levers import INCIDENCE_LEVERS, LEVER_CFD_STRIKE_BASIS

    _, specs = _repository_specs()
    presets = [s for s in specs.values() if s.meta.get("kind") == "preset"]
    assert len(presets) == 12
    for spec in presets:
        parent = specs[spec.meta["parent"]]
        moved = {n for n in LEVERS if spec.settings[n] != parent.settings[n]}
        assert moved, spec.case_id
        # The strike basis moves who is paid for a contract, not what the plant costs
        # (the auction strike keeps the cost): the one non-incidence lever treasury_first takes.
        assert moved <= set(INCIDENCE_LEVERS) | {LEVER_CFD_STRIKE_BASIS}, (spec.case_id, moved)
    # P3 takes only the levers that mean something once the fleet is bought: its purchase
    # on taxation moves the one lever the retired consumer_first moved there.
    p3 = specs["public_ownership@fully_nationalised_market_value"]
    taxed = specs["public_ownership@fully_nationalised_market_value_purchase_on_taxation"]
    assert {n for n in LEVERS if taxed.settings[n] != p3.settings[n]} == {"acquisition_financing"}
    # 30 Sep 2026: todays_rules moves nothing on P3 (its old incidence is its base), so it
    # makes no case; P3's treasury_first (user answer Q5) is the tariff at the private
    # comparator's market bill, the one lever it moves there.
    assert "public_ownership@fully_nationalised_market_value_todays_rules" not in specs
    first = specs["public_ownership@fully_nationalised_market_value_treasury_first"]
    assert {n for n in LEVERS if first.settings[n] != p3.settings[n]} == {"tariff_pricing"}
    assert first.settings["tariff_pricing"] == "market_bill"
    # Competitive neutrality (user answer Q2): a company paying dividends to its
    # shareholder keeps its surplus at the Treasury and levies its own deficits on bills.
    for parent in ("public_control", "public_ownership"):
        neutral = specs[f"{parent}@competitive_neutrality"].settings
        assert (neutral["surplus_recycling"], neutral["state_deficit_funding"], neutral["tax_equivalent"]) == (
            "retained_by_treasury", "bill_levy", "on"), parent


def test_a_toggle_grid_is_the_full_factorial_and_carries_its_toggle_vector():
    _, specs = _repository_specs()
    grids = {}
    for spec in specs.values():
        for membership in spec.meta.get("grid_memberships") or []:
            if membership["grid"].endswith("#bounds"):
                continue  # the bound grids (2 Oct 2026) have their own test
            grids.setdefault(membership["grid"], []).append(tuple(sorted(membership["toggles"].items())))
    # 30 Sep 2026: every position of every axis, the parent's own all-base position
    # included (it carries that membership, so the explorer maps every position to one
    # case).  F29 (no guarantee, the PWLB standard margin for the +1 pp rate): P1 2^2 x
    # networks {private, acquired} x pre-2025 x strike 3; P2 2^5 x 2 x 2 (no strike axis,
    # user decision 30 Sep 2026); P3 2^5 x {private, acquired x premium high}.  P2 with
    # state bids at cost has no grid.
    assert {g: len(v) for g, v in grids.items()} == {
        "public_control": 48, "public_ownership": 128, "public_ownership@fully_nationalised_market_value": 96}
    # The P2 grid has no strike axis, so its one-at-a-time strike case carries no membership.
    assert all("strike" not in m["toggles"] for s in specs.values() for m in s.meta.get("grid_memberships") or []
               if m["grid"] == "public_ownership")
    assert not specs["public_ownership@strike_financing_saving"].meta.get("grid_memberships")
    for vectors in grids.values():
        assert len(set(vectors)) == len(vectors)
    # An existing case the grid lands on is carried as a membership, never repeated.
    assert specs["public_ownership@overhead_high"].meta["grid_memberships"][0]["toggles"]["overhead-high"] == 1
    assert specs["public_ownership@networks_acquired"].meta["grid_memberships"][0]["toggles"]["networks"] == "acquired"
    # A conditional switch exists only under its selector option: null elsewhere.
    for spec in specs.values():
        for membership in spec.meta.get("grid_memberships") or []:
            toggles = membership["toggles"]
            if membership["grid"] == "public_ownership@fully_nationalised_market_value":
                assert (toggles["rav-premium-high"] is None) == (toggles["networks"] != "acquired"), spec.case_id


def test_the_retired_levers_and_values_are_refused_with_a_pointer():
    bundles = parse_scenarios(SCENARIOS)
    with pytest.raises(LeverConfigError, match="retired"):
        parse_sensitivities({"one_at_a_time": ["cfd_repricing_eligibility"]}, bundles)
    with pytest.raises(LeverConfigError, match="retired"):
        parse_sensitivities({"one_at_a_time": [{"public_rate_basis": ["nominal"]}]}, bundles)


@pytest.mark.parametrize("case_id", sorted(PROPOSITIONS))
def test_only_a_proposition_that_buys_the_counterparty_re_prices_a_signed_contract(case_id):
    """Coherence rule (iv): nothing retrospective in a headline.

    ``public_repriced`` re-opens CfDs awarded before 2025.  It is coherent only
    where the state has bought the counterparty, so the contract has no other
    side left to re-open.
    """
    _, specs = _repository_specs()
    settings = specs[case_id].settings
    if settings["pre_2025_cfd_capacity"] == "public_repriced":
        assert settings["legacy_fleet"] == "acquire_all", case_id


def test_every_proposition_has_at_least_one_declared_financing_stress():
    """A policy audience gets a design and a range, not a point estimate."""
    _, specs = _repository_specs()
    stresses = {
        "public_ownership": ("public_ownership@public_rate_pwlb_standard",),
        # Meeting notes (user, 10 Nov 2026): Scenario 4 carries the public-risk and gilt-response sensitivities.
        "public_ownership@fully_nationalised_market_value": (
            "public_ownership@fully_nationalised", "public_ownership@fully_nationalised_market_value_term_15y",
            "public_ownership@fully_nationalised_market_value_systematic_premium",
            "public_ownership@fully_nationalised_market_value_yield_b"),
        "public_control": ("public_control@gbe_value_to_bills",),
    }
    for proposition, twins in stresses.items():
        base = specs[proposition].settings
        for twin in twins:
            assert twin in specs, twin
            moved = [n for n in LEVERS if specs[twin].settings[n] != base[n]]
            assert moved, f"{twin} moves nothing off {proposition}"


# ── The rebase (user decisions of 30 Sep 2026) ───────────────────────────────


def test_every_base_is_bill_payers_first_with_private_wires():
    """Decisions 1 and 2: the consumer_first incidence is every base; the wires are private."""
    _, specs = _repository_specs()
    first = {"surplus_recycling": "rebate_gross", "shortfall_funding": "exchequer",
             "state_deficit_funding": "exchequer", "tax_equivalent": "none", "network_incidence": "bills"}
    for case_id in ("public_control", "public_ownership", "public_ownership@fully_nationalised_market_value"):
        settings = specs[case_id].settings
        assert settings["networks"] == "private", case_id
        assert {k: settings[k] for k in first} == first, case_id
    # The exception: P3's purchase stays on the tariff.
    assert specs["public_ownership@fully_nationalised_market_value"].settings["acquisition_financing"] == "tariff"
    # The retired preset is refused by name, with a pointer at where it lives now.
    bundles = parse_scenarios(SCENARIOS)
    with pytest.raises(LeverConfigError, match="retired"):
        parse_sensitivities({"presets": {"consumer_first": {"applies_to": ["public_ownership"], "levers": {}}}}, bundles)


def test_todays_rules_and_the_grid_reproduce_the_old_centrals():
    """Decisions 7 and 11: the old central of each scenario is still a position, never lost.

    Old P1 = todays_rules on P1 with the guarantee on the wires; old P2 = todays_rules
    on P2 (P2's wires were already private); old P3 = P3's grid with the wires acquired.
    The P3 position differs from the old bundle only on the market-bill incidence levers
    it inherits from P2, which have no meaning once the fleet is bought (the tariff and
    the ledger never read them for an acquiring case).
    """
    from scripts.public_ownership.levers import MARKET_ONLY_INCIDENCE_LEVERS

    _, specs = _repository_specs()
    old_p1 = _s(new_build_generation="state_stake", public_return="cepa_decomposition", networks="debt_guarantee",
                route_booking="booked", surplus_recycling="retained_by_treasury", shortfall_funding="exchequer")
    old_p2 = _s(new_build_generation="public", public_return="public_rate", cfd_strike_basis="at_cost",
                cfd_amortisation="economic_life", route_booking="booked")
    assert dict(specs["public_control@todays_rules"].settings, networks="debt_guarantee") == old_p1
    assert specs["public_ownership@todays_rules"].settings == old_p2
    old_p3 = dict(old_p2, pre_2025_cfd_capacity="public_repriced", legacy_fleet="acquire_all", networks="acquired",
                  purchase_price_basis="market_value")
    grid = specs["public_ownership@fully_nationalised_market_value_grid_networks-acquired"].settings
    moved = {k for k in LEVERS if grid[k] != old_p3[k]}
    # 13 Nov 2026 (user, results review): the central purchase carries 2% transaction costs (the old one none).
    assert grid["acquisition_transaction_costs"] == "two_pct"
    moved.discard("acquisition_transaction_costs")
    assert moved <= set(MARKET_ONLY_INCIDENCE_LEVERS), moved
    # The old P1 with the wires private is todays_rules itself (it was @networks_private).
    assert "public_control@networks_private" not in specs


def test_a_selector_and_a_conditional_switch_expand_exactly():
    """The grid framework (decision 5): selectors, conditional switches, readable ids."""
    from scripts.public_ownership.levers import grid_positions

    bundles = parse_scenarios(SCENARIOS)
    sens = parse_sensitivities({"toggle_grids": {"enabled": True, "grids": {"public_control": [
        {"token": "overhead-high", "lever": "state_overhead", "value": "high"},
        {"selector": "networks", "lever": "networks", "options": {"private": "private", "acquired": "acquired"}},
        {"token": "rav-premium-high", "lever": "network_acquisition_premium", "value": "high",
         "when": {"networks": "acquired"}},
        {"selector": "strike", "lever": "cfd_strike_basis",
         "options": {"saving": "financing_saving_only", "cost": "at_cost", "auction": "auction_clearing"}},
    ]}}}, bundles)
    toggles = sens["toggle_grids"]["grids"]["public_control"]
    positions = grid_positions(toggles, bundles["public_control"])
    # 2 x (1 + 2) x 3 = 18 positions, less the parent's own.
    assert len(positions) == 17
    specs = {s.case_id: s for s in expand_cases(bundles, sens)}
    case = specs["public_control@grid_networks-acquired_rav-premium-high_strike-auction"]
    assert (case.settings["networks"], case.settings["network_acquisition_premium"], case.settings["cfd_strike_basis"]) == (
        "acquired", "high", "auction_clearing")
    assert case.meta["toggles"] == {"overhead-high": 0, "networks": "acquired", "rav-premium-high": 1,
                                    "strike": "auction"}
    # The premium never appears outside the acquisition, so no position repeats a case.
    assert not any("rav-premium-high" in c and "networks-acquired" not in c for c in specs)
    assert specs["public_control"].meta["grid_memberships"] == [{"grid": "public_control", "toggles": {
        "overhead-high": 0, "networks": "private", "rav-premium-high": None, "strike": "saving"}}]
    # Ids follow the declared order, one position per case.
    assert "public_control@grid_overhead-high_networks-acquired_strike-cost" in specs
    assert sum(1 for s in specs.values() if s.meta.get("kind") == "grid") == 17
    # Positions are ordered as a binary grid always was: fewer moves first, earlier axes first.
    first = [pieces for _, pieces, _ in positions[:4]]
    assert first == [["overhead-high"], ["networks-acquired"], ["strike-cost"], ["strike-auction"]]


@pytest.mark.parametrize("entry, message", [
    ({"token": "x", "lever": "network_acquisition_premium", "value": "high", "when": {"networks": "acquired"}},
     "declared before"),
    ({"selector": "networks", "lever": "networks", "options": {"private": "private"}}, "two options"),
    ({"selector": "networks", "lever": "networks", "options": {"a": "private", "b": "private"}}, "twice"),
    ({"selector": "tilt", "lever": "surplus_recycling", "options": {"a": "rebate_gross", "b": "public_cfd_at_cost"}},
     "may move only"),
])
def test_a_grid_axis_the_code_cannot_expand_is_refused(entry, message):
    bundles = parse_scenarios(SCENARIOS)
    with pytest.raises(LeverConfigError, match=message):
        parse_sensitivities({"toggle_grids": {"enabled": True, "grids": {"public_control": [entry]}}}, bundles)


def test_a_grid_position_that_would_not_move_its_lever_is_refused():
    """Exactness: a switch whose requirement the position does not meet is an error, not a silent alias."""
    bundles = parse_scenarios(SCENARIOS)
    sens = parse_sensitivities({"toggle_grids": {"enabled": True, "grids": {"public_control": [
        {"token": "rav-premium-high", "lever": "network_acquisition_premium", "value": "high"}]}}}, bundles)
    with pytest.raises(LeverConfigError, match="moves nothing"):
        expand_cases(bundles, sens)


def test_the_market_bill_tariff_bites_only_where_the_fleet_is_bought():
    """User answer Q5: tariff_pricing is an incidence lever of the acquisition tariff alone."""
    from scripts.public_ownership.levers import INCIDENCE_LEVERS, MARKET_ONLY_INCIDENCE_LEVERS

    ownership = parse_scenarios(SCENARIOS)["public_ownership"]
    assert not LEVERS["tariff_pricing"].applies("market_bill", ownership)
    assert LEVERS["tariff_pricing"].applies("market_bill", dict(ownership, legacy_fleet="acquire_all"))
    assert "tariff_pricing" in INCIDENCE_LEVERS and "tariff_pricing" not in MARKET_ONLY_INCIDENCE_LEVERS
    assert not LEVERS["tariff_pricing"].register


def test_a_preset_takes_per_parent_values():
    """Decision 7: todays_rules holds each scenario's own old incidence."""
    bundles = parse_scenarios({
        "private": {},
        "public_control": {**SCENARIOS["public_control"], "surplus_recycling": "rebate_gross",
                           "shortfall_funding": "exchequer"},
        "public_ownership": {**SCENARIOS["public_ownership"], "surplus_recycling": "rebate_gross",
                             "shortfall_funding": "exchequer"},
    })
    sens = parse_sensitivities({"presets": {"todays_rules": {
        "applies_to": ["public_control", "public_ownership"], "levers": {},
        "levers_by_case": {"public_control": {"surplus_recycling": "retained_by_treasury"},
                           "public_ownership": {"surplus_recycling": "public_cfd_at_cost",
                                                "shortfall_funding": "levy_on_eligible_supply"}}}}}, bundles)
    specs = {s.case_id: s for s in expand_cases(bundles, sens)}
    assert specs["public_control@todays_rules"].settings["surplus_recycling"] == "retained_by_treasury"
    assert specs["public_control@todays_rules"].settings["shortfall_funding"] == "exchequer"
    assert specs["public_ownership@todays_rules"].settings["shortfall_funding"] == "levy_on_eligible_supply"
    with pytest.raises(LeverConfigError, match="may move only"):
        parse_sensitivities({"presets": {"x": {"applies_to": ["public_control"],
                                               "levers_by_case": {"public_control": {"networks": "private"}}}}},
                            bundles)


def test_the_bound_grids_cover_every_combination_between_best_and_worst():
    """User request (2 Oct 2026): the best and worst assumptions switched one by one, exactly.

    Each bound grid maps every position to one case (F29: 24, 144 and 3,456), its all-best and
    all-worst corners are the bound cases themselves, and a position that moves the private
    WACC basis or the LDES floor is measured against private at the same point on them.
    """
    import collections
    from scripts.public_ownership.levers import BOUND_GRID_SUFFIX, comparator_case_id
    from scripts.public_ownership.reference_runs import configured_case_specs

    config = load_defaults()["public_ownership"]
    specs = configured_case_specs(config)
    by_id = {spec.case_id: spec for spec in specs}
    positions = collections.defaultdict(set)
    for spec in specs:
        for membership in spec.meta.get("grid_memberships") or []:
            if membership["grid"].endswith(BOUND_GRID_SUFFIX):
                positions[membership["grid"]].add(tuple(sorted((k, str(v)) for k, v in membership["toggles"].items())))
    assert {grid: len(p) for grid, p in positions.items()} == {
        # Meeting notes (user, 10 Nov 2026): no yield axis on Scenarios 2 and 3. 13 Nov 2026 (results review):
        # Scenario 2's grid adds the premium and the two delivery switches (24 x 8).
        "public_control" + BOUND_GRID_SUFFIX: 12, "public_ownership" + BOUND_GRID_SUFFIX: 192,
        "public_ownership@fully_nationalised_market_value" + BOUND_GRID_SUFFIX: 3456}

    def bound_grid_toggles(case_id):
        return [m["toggles"] for m in by_id[case_id].meta.get("grid_memberships") or []
                if m["grid"].endswith(BOUND_GRID_SUFFIX)]

    # The corners: P2's best case is the all-best position, its worst case the all-worst one.
    # Meeting notes (user, 10 Nov 2026): Scenario 3's grid has no premium or yield axis.
    assert {"wacc": "notional", "ldes-wacc": 1, "overhead": "low", "rate-pwlb-standard": 0, "premium-erp": 0,
            "overrun-15": 0, "om-plus-10": 0} in bound_grid_toggles("public_ownership@best_case")
    assert {"wacc": "tmr", "ldes-wacc": 0, "overhead": "high", "rate-pwlb-standard": 1, "premium-erp": 1,
            "overrun-15": 1, "om-plus-10": 1} in bound_grid_toggles("public_ownership@worst_case")
    # House rule 3 on a comparator lever: the Fixed TMR positions sit on private's Fixed TMR twin.
    tmr = next(spec for spec in specs if spec.case_id.startswith("public_control@bgrid_")
               and spec.settings["private_wacc_basis"] == "fixed_tmr_shielded"
               and spec.settings["ldes_floor_basis"] == "cost_of_debt_like")
    assert comparator_case_id(tmr, specs) == "private@wacc_fixed_tmr"


def test_the_gilt_sources_of_f19_are_retired():
    """F29 (26 Oct 2026): workbook v2.0's cash rate replaces config's 30-year gilt (F19 decision 4)."""
    from scripts.public_ownership.levers import RETIRED_VALUES
    from scripts.public_ownership.reference_runs import RETIRED_KEYS

    assert ("public_rate_basis", "gilt_30y_real") in RETIRED_VALUES
    assert ("public_rate_basis", "real_plus_1pp") in RETIRED_VALUES
    assert "public_rate_sources" in RETIRED_KEYS
    po = load_defaults()["public_ownership"]
    assert "public_rate_sources" not in po