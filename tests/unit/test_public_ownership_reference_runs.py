from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.public_ownership.reference_runs import (
    MONTHS,
    baseline_scenarios,
    reference_run_scenarios,
    required_scenarios,
    runnable_groups,
    validate_public_ownership_config,
)
from config.config_loader import load_defaults


@pytest.fixture(scope="module")
def defaults_config() -> dict:
    return load_defaults()["public_ownership"]


def test_repository_defaults_are_a_supported_decision_register(defaults_config):
    assert validate_public_ownership_config(defaults_config) is defaults_config


@pytest.mark.parametrize(
    "mutate, message",
    [
        # Item K1: the retired keys point at the levers.
        (lambda c: c.__setitem__("existing_assets", "public_refinanced"), "retired"),
        (lambda c: c.__setitem__("cases", ["private", "public_ownership"]), "retired"),
        (lambda c: c.__setitem__("sensitivity_axes", {}), "retired"),
        (lambda c: c.__setitem__("yield_assumption", "B"), "retired"),
        (lambda c: c.__setitem__("network_ownership", "acquired"), "retired"),
        (lambda c: c["acquisition"].__setitem__("enabled", True), "retired"),
        (lambda c: c["acquisition"].__setitem__("purchase_price_basis", "haggle"), "retired"),
        (lambda c: c["acquisition"].__setitem__("term_years", 0), "retired"),
        # The scenarios and sensitivities are enforced.
        (lambda c: c["scenarios"].pop("private"), "private"),
        (lambda c: c["scenarios"]["private"].__setitem__("new_build_generation", "public"), "baseline"),
        (lambda c: c["scenarios"]["public_ownership"].__setitem__("legacy_fleet", "buy_everything"), "not implemented"),
        (lambda c: c["scenarios"].__setitem__("public_control@x", {}), "may not contain"),
        (lambda c: c["sensitivities"]["one_at_a_time"].append("gilt_yield"), "unknown lever"),
        (lambda c: c["sensitivities"]["named"].__setitem__("orphan", {"legacy_fleet": "acquire_all"}), "declared scenario"),
        (lambda c: c["sensitivities"]["named"].__setitem__("noop", {"scenario": "public_ownership"}), "moves no lever"),
        # OW9 (3 Oct 2026): the three explicit values and their config keys are retired, with a pointer.
        (lambda c: c["sensitivities"]["named"]["fully_nationalised"].__setitem__("purchase_price_basis", "explicit"),
         "retired 3 Oct 2026"),
        (lambda c: c["sensitivities"]["named"]["fully_nationalised"].__setitem__("thermal_basis", "explicit"),
         "retired 3 Oct 2026"),
        (lambda c: c["sensitivities"]["named"]["fully_nationalised"].__setitem__("public_return", "explicit"),
         "retired 3 Oct 2026"),
        (lambda c: c.__setitem__("public_return_explicit_rate", 0.05), "retired"),
        (lambda c: c["acquisition"].__setitem__("purchase_price_gbp", 1.0e11), "retired"),
        (lambda c: c["acquisition"].__setitem__("thermal_price_gbp", 1.0e10), "retired"),
        (lambda c: c.__setitem__("state_stake_share", 1.5), "state_stake_share"),
        (lambda c: c.__setitem__("ownership_scope", ["Line"]), "ownership_scope"),
        (lambda c: c["acquisition"].__setitem__("include_in_net_bill", True), "include_in_net_bill"),
        (lambda c: c.__setitem__("surplus_recycling", {"public_ownership": "full_rebate"}), "retired"),
        (lambda c: c.__setitem__("public_shortfall_funding", "print_money"), "retired"),
        (lambda c: c["consumer_benefit"].__setitem__("treatment", "bill_cut"), "consumer_benefit"),
        (lambda c: c["reference_run_groups"]["cp30_2030_cp30_network"].__setitem__("baseline_run_group", "cp30_2030_cp30_network"), "baseline_run_group"),
        (lambda c: c["reference_run_groups"]["cp30_2030_cp30_network"].__setitem__("scenario_pattern", "NoMonthHere"), "{month}"),
        (lambda c: c.__setitem__("policy_start_year", "2025"), "policy_start_year"),
    ],
)
def test_unsupported_decisions_fail_loudly(defaults_config, mutate, message):
    config = deepcopy(defaults_config)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_public_ownership_config(config)


def test_baseline_must_predate_policy_start_and_model_year(defaults_config):
    config = deepcopy(defaults_config)
    config["reference_run_groups"]["historical_2024"]["model_year"] = 2031
    with pytest.raises(ValueError, match="earlier model year"):
        validate_public_ownership_config(config)
    config["reference_run_groups"]["historical_2024"]["model_year"] = 2026
    with pytest.raises(ValueError, match="predate policy_start_year"):
        validate_public_ownership_config(config)


def test_scenario_expansion_matches_group_mode(defaults_config):
    groups = defaults_config["reference_run_groups"]
    monthly = reference_run_scenarios(groups["cp30_2030_cp30_network"])
    assert len(monthly) == 12 and len(set(monthly)) == 12
    assert all(month in scenario for month, scenario in zip(MONTHS, monthly))
    assert reference_run_scenarios(groups["historical_2024"]) == [groups["historical_2024"]["scenario"]]
    assert baseline_scenarios(defaults_config, "cp30_2030_cp30_network") == reference_run_scenarios(groups["historical_2024"])
    assert baseline_scenarios(defaults_config, "historical_2024") == []
    assert len(required_scenarios(defaults_config, "cp30_2030_cp30_network")) == 13


def test_all_target_only_requests_groups_whose_scenarios_are_active(defaults_config):
    groups = defaults_config["reference_run_groups"]
    active = reference_run_scenarios(groups["cp30_2030_cp30_network"])
    # The group and every group on the same months (its price-level twin, item F12, 12 Oct 2026).
    same_months = [name for name, group in groups.items()
                   if group.get("scenario_pattern") == groups["cp30_2030_cp30_network"]["scenario_pattern"]]
    assert same_months[0] == "cp30_2030_cp30_network" and "cp30_2030_cp30_network_cp32" in same_months
    assert runnable_groups(defaults_config, active) == same_months
    assert runnable_groups(defaults_config, active[:-1]) == []
    active += reference_run_scenarios(groups["historical_2024"])
    assert runnable_groups(defaults_config, active) == ["historical_2024", *same_months]
    assert runnable_groups(defaults_config, []) == []
