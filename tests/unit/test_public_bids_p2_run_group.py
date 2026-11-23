"""Item BM1, step 4 of the P2 bill chain: a run group on the P2 months, selected by a lever.

The ``balancing_bids`` lever says which balancing-bid stack a case's bill reads.
``state_at_cost`` is solved by one run group, ``cp30_2030_cp30_network_public_bids_p2``, which
projects its own forecast case for ``private`` and the cases on that stack only.
The market group projects every other market case and never these. The checks
refuse a case whose stack no group solves, a group whose months bid on another
case's shares, and a stack combined with the acquisition tariff, which reads the
market group only.
"""

import copy
from pathlib import Path

import pytest
import yaml

from scripts.public_ownership.levers import LEVER_BALANCING_BIDS
from scripts.public_ownership.reference_runs import (
    check_balancing_group_scenarios, configured_case_specs, consumer_bill_groups, expanded_cases,
    group_bid_case, group_excluded_cases, group_forecast_case_id, group_market_cases,
    institution_twin_groups, price_level_twin_groups, validate_public_ownership_config, wholesale_source,
)
from config.config_loader import load_scenario_entries
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
# Network bounds (user decision, 18-19 Oct 2026): the full case list is the CENTRAL network's
# (no_acceleration) family; cp30_2030* is the LOWER network on the headline cases
# (tests/unit/test_network_bound_groups.py).
P2_GROUP, MARKET_GROUP = "cp30_2030_no_acceleration_public_bids_p2", "cp30_2030_no_acceleration"
P2_CASE = "public_ownership@state_bids_at_cost"


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


def _p2_stack(config):
    """P2 with state bids, its presets and its toggle grid (29 Sep 2026): every case on the P2 stack."""
    specs = configured_case_specs(config)
    return [spec.case_id for spec in specs if spec.settings[LEVER_BALANCING_BIDS] != "contract_value"]


def test_the_lever_generates_one_case_on_the_p2_stack(config):
    moved = _p2_stack(config)
    # The lever's own case, then the presets built on it (todays_rules, competitive
    # neutrality, treasury_first).  30 Sep 2026 (user decision 6): the explorer does not
    # show P2 with state bids, so its toggle grid (15 cases) is gone.
    assert moved[0] == P2_CASE and len(moved) == 1 + 3
    assert all(case.startswith(P2_CASE) for case in moved)


def test_each_bill_group_reads_its_own_forecast_case(config, forecast_cases):
    validate_public_ownership_config(config, forecast_cases)
    # Item BM1, P3: the tariff balancing group follows (tests/unit/test_public_bids_p3_run_group.py).
    # Item F12 (12 Oct 2026): price-level twins (*_cp32) re-account these groups; tests/unit/
    # test_public_ownership_price_level_twins.py covers them.
    # P1/P2 under central dispatch (20 Oct 2026): the institution twin is covered in
    # tests/unit/test_public_ownership_institution_twins.py.
    twins = {**price_level_twin_groups(config), **institution_twin_groups(config)}
    network = config["reference_run_groups"][MARKET_GROUP]["network_case"]
    # F28: the legislation-year register bound re-accounts the market group's months; not counted here.
    assert [g for g in consumer_bill_groups(config, forecast_cases) if g not in twins
            and config["reference_run_groups"][g].get("network_case") == network
            and config["reference_run_groups"][g].get("register_bound_of") is None] == [
        MARKET_GROUP, P2_GROUP, "cp30_2030_no_acceleration_public_bids_p3"]
    assert group_forecast_case_id(config, MARKET_GROUP) == "CP30_2030_no_acceleration"
    assert group_forecast_case_id(config, P2_GROUP) == "CP30_2030_no_acceleration_public_bids_p2"


def test_the_groups_split_the_cases_between_them(config):
    market, p2 = group_market_cases(config, MARKET_GROUP), group_market_cases(config, P2_GROUP)
    assert p2 == ["private", *_p2_stack(config)]
    assert "private" in market and P2_CASE not in market
    assert P2_CASE in group_excluded_cases(config, MARKET_GROUP)
    assert set(group_excluded_cases(config, P2_GROUP)) == set(expanded_cases(config)) - set(p2)
    assert group_bid_case(config, P2_GROUP) == "public_ownership"
    assert group_bid_case(config, MARKET_GROUP) is None


def test_a_stack_no_group_solves_is_refused(config, forecast_cases):
    del config["reference_run_groups"][P2_GROUP]
    with pytest.raises(ValueError, match="exactly one bill group"):
        validate_public_ownership_config(config, forecast_cases)


def test_a_balancing_group_off_its_forecast_case_is_refused(config, forecast_cases):
    config["reference_run_groups"][P2_GROUP]["forecast_case"] = "CP30_2030_cp30_network"
    with pytest.raises(ValueError, match="is not a bill group"):
        validate_public_ownership_config(config, forecast_cases)


def test_an_unknown_stack_or_forecast_case_is_refused(config, forecast_cases):
    bad = copy.deepcopy(config)
    bad["reference_run_groups"][P2_GROUP]["balancing_bids"] = "pay_as_bid"
    with pytest.raises(ValueError, match="balancing_bids"):
        validate_public_ownership_config(bad, forecast_cases)
    config["reference_run_groups"][P2_GROUP]["forecast_case"] = "nope"
    with pytest.raises(ValueError, match=r"not a consumer_bill[_.]forecasts case"):
        validate_public_ownership_config(config, forecast_cases)


def test_a_stack_with_the_acquisition_tariff_is_refused(config, forecast_cases):
    config["sensitivities"]["named"]["acquired_state_bids"] = {
        "scenario": "public_ownership", "legacy_fleet": "acquire_valued_group", "balancing_bids": "state_at_cost"}
    with pytest.raises(ValueError, match="only market-bill cases"):
        validate_public_ownership_config(config, forecast_cases)


def test_the_months_bid_on_the_shares_of_the_group_cases(config, scenario_configs):
    check_balancing_group_scenarios(config, P2_GROUP, scenario_configs)
    check_balancing_group_scenarios(config, MARKET_GROUP, scenario_configs)
    # The months merge one YAML anchor and share its nested mappings, so give
    # March its own copy before changing it.
    month = config["reference_run_groups"][P2_GROUP]["scenario_pattern"].format(month="Mar")
    wrong = {**scenario_configs, month: copy.deepcopy(scenario_configs[month])}
    wrong[month]["support"]["bm_bids"]["contract_value_bids"][
        "public_ownership_bids"]["case_id"] = "public_control"
    with pytest.raises(ValueError, match="Mar_no_acceleration_p2_bids bids on the shares of case 'public_control'"):
        check_balancing_group_scenarios(config, P2_GROUP, wrong)


def test_a_month_reads_its_parent_stage_1(scenario_configs):
    month = "CP30_FFR_2030_Jan_cp30_network_p2_bids"
    assert wholesale_source(month, scenario_configs) == (
        "CP30_FFR_2030_Jan_stage1")
    assert wholesale_source("anything_else", scenario_configs) == "anything_else"
