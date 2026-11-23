"""Item BM1, P3: a tariff balancing group on the P3 months.

P3 (full nationalisation at market value, market machinery kept) acquires the
legacy fleet, so its bill is the acquisition tariff, not the market bill. Its run
group ``cp30_2030_cp30_network_public_bids_p3`` names that tariff case with ``bid_case`` (the
case its months bid on), projects ``private`` alone on the market bill and
tariffs the bid case alone. It takes no part in the one-group-per-stack count, so
P2's group, cases and outputs are exactly as they were. The checks refuse a bid
case that is not a tariff case, a bid case on a contract-value group, two groups
naming one bid case, a P3 group off its own forecast case, and a month bidding on
another case's shares.
"""

import copy
from pathlib import Path

import pytest
import yaml

from scripts.public_ownership.reference_runs import (
    register_bound_groups,
    check_balancing_group_scenarios, configured_case_specs, consumer_bill_groups, expanded_cases, group_bid_case,
    group_excluded_cases, group_forecast_case_id, group_market_cases, group_tariff_cases,
    institution_twin_groups, is_tariff_balancing_group, price_level_twin_groups, tariff_cases,
    validate_public_ownership_config,
    wholesale_source,
)
from config.config_loader import load_scenario_entries
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
# Network bounds (user decision, 18-19 Oct 2026): the full case list is the CENTRAL network's
# (no_acceleration) family; cp30_2030* is the LOWER network on the headline cases
# (tests/unit/test_network_bound_groups.py).
P3_GROUP, P2_GROUP, MARKET_GROUP = ("cp30_2030_no_acceleration_public_bids_p3",
                                     "cp30_2030_no_acceleration_public_bids_p2", "cp30_2030_no_acceleration")
P3_CASE = "public_ownership@fully_nationalised_market_value"
# The tariff cases owning the same fleet as P3 (they differ only in purchase price
# basis, thermal basis or annuity term), so they bid the same stack: bid case first.
P3_FAMILY = [
    P3_CASE,
    "public_ownership@fully_nationalised",
    "public_ownership@fully_nationalised_book_value",
    # F29 (26 Oct 2026): workbook v2.0's A3 bases, no support and the RO passed through.
    "public_ownership@fully_nationalised_no_support_value",
    "public_ownership@fully_nationalised_ro_pass_through",
    "public_ownership@fully_nationalised_thermal_at_zero",
    # F29: the model's own 2030 price is the central path; CP.32 is the sensitivity.
    "public_ownership@fully_nationalised_market_value_price_path_cp30",
    # 11 Nov 2026 (user): the fleet valued at today's forward prices for its whole life (legacy_price_path: today_level).
    "public_ownership@fully_nationalised_market_value_price_path_today",
    # F27 item 6 (24 Oct 2026): existing capacity on cost-based contracts moves only the purchase price.
    "public_ownership@fully_nationalised_market_value_capacity_cost_based",
    # User decision (24 Oct 2026): legacy offshore wind at deal multiples moves only the purchase price.
    "public_ownership@fully_nationalised_market_value_offshore_deal_multiples",
    "public_ownership@fully_nationalised_term_15y",
    "public_ownership@fully_nationalised_market_value_term_15y",
    # F28 (e): the purchase repaid over one 30-year term (the central is each asset over its remaining life).
    "public_ownership@fully_nationalised_market_value_term_30y",
    # 10-11 Nov 2026 (user): the anchored price path is central; the model's 2030 prices held flat (the old central,
    # legacy_price_path: model_2030) is now this named case. Both price-path cases move only the purchase price.
    # (The 13 Nov systematic_premium_1pc case moves systematic_premium, not an acquisition-term lever, so it is not
    # in the same-fleet family; reference_runs.ACQUISITION_TERM_LEVERS.)
    "public_ownership@fully_nationalised_market_value_prices_held_at_2030",
]
P2_CASE = "public_ownership@state_bids_at_cost"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@pytest.fixture(scope="module")
def defaults():
    return load_defaults()


@pytest.fixture(scope="module")
def forecast_cases(defaults):
    cases = copy.deepcopy(defaults["consumer_bill"]["forecasts"]["cases"])
    for case_id, case in cases.items():
        case["case_id"] = case_id
    return cases


@pytest.fixture()
def config(defaults):
    return copy.deepcopy(defaults["public_ownership"])


@pytest.fixture(scope="module")
def scenario_configs():
    return load_scenario_entries(ROOT / "config")


def test_the_p3_group_is_a_bill_group_on_its_own_forecast_case(config, forecast_cases):
    validate_public_ownership_config(config, forecast_cases)
    # Item F12 (12 Oct 2026): the price-level twins (*_cp32) are tested in test_public_ownership_price_level_twins.py.
    # P1/P2 under central dispatch (20 Oct 2026): the institution twin is covered in
    # tests/unit/test_public_ownership_institution_twins.py.
    twins = {**price_level_twin_groups(config), **institution_twin_groups(config)}
    network = config["reference_run_groups"][MARKET_GROUP]["network_case"]
    # F28: the legislation-year register bound re-accounts the market group's months; not counted here.
    assert [g for g in consumer_bill_groups(config, forecast_cases) if g not in twins
            and config["reference_run_groups"][g].get("network_case") == network
            and g not in register_bound_groups(config)] == [MARKET_GROUP, P2_GROUP, P3_GROUP]
    assert group_forecast_case_id(config, P3_GROUP) == "CP30_2030_no_acceleration_public_bids_p3"
    assert config["reference_run_groups"][P3_GROUP]["balancing_bids"] == "state_at_cost"


def test_the_p3_group_projects_private_and_tariffs_the_p3_fleet_family(config):
    assert is_tariff_balancing_group(config, P3_GROUP)
    assert group_bid_case(config, P3_GROUP) == P3_CASE
    assert group_market_cases(config, P3_GROUP) == ["private"]
    # The acquisition-term family, then (29 Sep 2026) the P3 toggle grid, which changes no fleet.
    tariffed = group_tariff_cases(config, P3_GROUP)
    assert tariffed[:len(P3_FAMILY)] == P3_FAMILY
    specs = {spec.case_id: spec for spec in configured_case_specs(config)}
    # F29 (26 Oct 2026): the bound grid's thermal-book and placeholder-price switches are retired, so no bound-grid
    # position moves acquisition terms alone; every case after the family is a P3 toggle-grid position.
    assert all(any(m["grid"] == P3_CASE for m in specs[c].meta.get("grid_memberships") or [])
               for c in tariffed[len(P3_FAMILY):])
    # The exact P3 grid is 96 positions (F29), less the parent; hpc_completion is a named case the grid lands on.
    assert len(tariffed) == len(P3_FAMILY) + 95
    # Cases that acquire less (or keep pre-2025 CfDs and the wires private) own a
    # different fleet, so they bid a different stack and stay off these months.
    assert "public_ownership@legacy_and_thermal_acquired" not in group_tariff_cases(config, P3_GROUP)
    assert "public_ownership@legacy_acquired" not in group_tariff_cases(config, P3_GROUP)
    assert set(group_excluded_cases(config, P3_GROUP)) == set(expanded_cases(config)) - {"private"}
    # P3 is a tariff case, never a market-bill one, on any group.
    assert P3_CASE in tariff_cases(config)
    assert P3_CASE not in group_market_cases(config, MARKET_GROUP)


def test_the_family_is_every_case_differing_only_in_acquisition_terms(config):
    from scripts.public_ownership.reference_runs import ACQUISITION_TERM_LEVERS, same_fleet_tariff_cases

    specs = {spec.case_id: spec.settings for spec in configured_case_specs(config)}
    grid = {lever for toggle in config["sensitivities"]["toggle_grids"]["grids"][P3_CASE] for lever in [toggle["lever"]]}
    for case in same_fleet_tariff_cases(config, P3_CASE):
        moved = {k for k, v in specs[case].items() if v != specs[P3_CASE][k]}
        # Acquisition terms, or the P3 toggles, none of which changes the fleet.
        assert moved <= set(ACQUISITION_TERM_LEVERS) | grid, (case, moved)
    # A new acquisition-term variant joins the family without a config change.
    config["sensitivities"]["named"]["fully_nationalised_market_value_thermal_at_zero"] = {
        "extends": "fully_nationalised_market_value", "thermal_basis": "zero"}
    assert "public_ownership@fully_nationalised_market_value_thermal_at_zero" in group_tariff_cases(config, P3_GROUP)


def test_p2_and_the_market_group_are_unchanged(config):
    assert not is_tariff_balancing_group(config, P2_GROUP)
    assert not is_tariff_balancing_group(config, MARKET_GROUP)
    assert group_market_cases(config, P2_GROUP)[:2] == ["private", P2_CASE]
    assert group_bid_case(config, P2_GROUP) == "public_ownership"
    assert group_tariff_cases(config, P2_GROUP) == []
    assert group_tariff_cases(config, MARKET_GROUP) == tariff_cases(config)
    without = copy.deepcopy(config)
    del without["reference_run_groups"][P3_GROUP]
    assert group_market_cases(without, P2_GROUP) == group_market_cases(config, P2_GROUP)
    assert group_market_cases(without, MARKET_GROUP) == group_market_cases(config, MARKET_GROUP)


def test_the_p3_group_does_not_count_against_the_p2_stack(config, forecast_cases):
    # Both groups declare state_at_cost; only P2 projects a market case on it.
    validate_public_ownership_config(config, forecast_cases)
    del config["reference_run_groups"][P2_GROUP]
    with pytest.raises(ValueError, match="exactly one bill group"):
        validate_public_ownership_config(config, forecast_cases)


def test_a_bid_case_must_be_a_tariff_case(config, forecast_cases):
    for bad in ("public_ownership", "public_ownership@fully_nationalised_market_value_central_dispatch", "nope"):
        broken = copy.deepcopy(config)
        broken["reference_run_groups"][P3_GROUP]["bid_case"] = bad
        with pytest.raises(ValueError, match="acquisition-tariff case"):
            validate_public_ownership_config(broken, forecast_cases)


def test_a_bid_case_on_a_contract_value_group_is_refused(config, forecast_cases):
    config["reference_run_groups"][P3_GROUP]["balancing_bids"] = "contract_value"
    with pytest.raises(ValueError, match="needs a balancing_bids stack"):
        validate_public_ownership_config(config, forecast_cases)


def test_two_groups_naming_one_bid_case_are_refused(config, forecast_cases):
    config["reference_run_groups"]["cp30_2030_no_acceleration_public_bids_p3_copy"] = copy.deepcopy(
        config["reference_run_groups"][P3_GROUP])
    with pytest.raises(ValueError, match="more than one reference_run_group names bid_case"):
        validate_public_ownership_config(config, forecast_cases)


def test_the_p3_group_off_its_forecast_case_is_refused(config, forecast_cases):
    config["reference_run_groups"][P3_GROUP]["forecast_case"] = "CP30_2030_cp30_network"
    with pytest.raises(ValueError, match="is not a bill group"):
        validate_public_ownership_config(config, forecast_cases)


def test_the_months_bid_on_the_p3_case_shares(config, scenario_configs):
    check_balancing_group_scenarios(config, P3_GROUP, scenario_configs)
    # The months merge one YAML anchor and share its nested mappings, so give
    # June its own copy before changing it.
    month = config["reference_run_groups"][P3_GROUP]["scenario_pattern"].format(month="Jun")
    wrong = {**scenario_configs, month: copy.deepcopy(scenario_configs[month])}
    wrong[month]["support"]["bm_bids"]["contract_value_bids"][
        "public_ownership_bids"]["case_id"] = "public_ownership"
    with pytest.raises(ValueError, match="Jun_no_acceleration_p3_bids bids on the shares of case 'public_ownership'"):
        check_balancing_group_scenarios(config, P3_GROUP, wrong)


def test_every_p3_month_is_the_lifo_ordering_on_its_parent_stage_1(config, scenario_configs):
    pattern = config["reference_run_groups"][P3_GROUP]["scenario_pattern"]
    parent = config["reference_run_groups"][MARKET_GROUP]["scenario_pattern"]
    for month in MONTHS:
        scenario = pattern.format(month=month)
        bids = (scenario_configs[scenario]["support"]["bm_bids"]
                ["contract_value_bids"]["public_ownership_bids"])
        assert bids["tie_break"]["enabled"] is True
        # The months are the headline ordering, never the gate's reverse.
        assert bids["tie_break"]["rule"] == "lifo_commissioning"
        # Both share the market month's Stage 1 (s1_zf since 13 Oct 2026).
        assert wholesale_source(scenario, scenario_configs) == wholesale_source(parent.format(month=month), scenario_configs)
