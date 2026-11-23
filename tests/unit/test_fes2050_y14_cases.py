"""FES 2050 plan, task Y14: per-group case lists, per-share P2 balancing groups and register bounds.

A run group may declare its own case list (``sensitivities``, the global schema) or
share another group's (``sensitivities_from``); every group without one keeps the
global list, so the 2030 groups are unchanged.  The one-group-per-stack check is
counted per list and per L4 share, so the 2050 list's P2 stack can be solved by one
group per share.  The learning-rate and retirement-rule bounds are register bounds:
second contract-value groups on the parent's months that change one group block.
"""

import copy
from pathlib import Path

import pytest
import yaml

from scripts.public_ownership.reference_runs import (
    case_list_owner, check_balancing_group_scenarios, configured_case_specs, consumer_bill_groups, expanded_cases,
    group_bid_case, group_excluded_cases, group_market_cases, group_tariff_cases, institution_market_twin,
    institution_scenarios, market_bill_cases, register_bound_groups, tariff_cases, validate_public_ownership_config,
)
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
SHARES = ("25", "50", "75", "100")

# The 2050 groups and forecast cases of the proposed config/fes2050.yaml overlay
# (decision 31), in the shape the Y14 report hands the coordinator.
OVERLAY = """
consumer_bill:
  forecasts:
    cases:
      FES25_HT_2050: &fc
        modelled_year: 2050
        model_price_year: 2024
        source_scenario_pattern: "FES25_HT_2050_{month}_market"
        assumptions: "data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json"
        recovery_demand_assumptions: "data/market/consumer_bill_forecast/2050_recovery_demand_assumptions.json"
        annual_aggregate_study: "FES25_HT_2050"
        ro_scheme_parameters: "data/market/consumer_bill_forecast/2050_ro_scheme_parameters.json"
        ro_relief_case: scheme_closed
        cfd_recovery: {method: eligible_supplier_sales}
      FES25_HT_2050_p2_bids_25pct: {<<: *fc, source_scenario_pattern: "FES25_HT_2050_{month}_p2_bids_25pct", recovery_case_id: FES25_HT_2050}
      FES25_HT_2050_p2_bids_50pct: {<<: *fc, source_scenario_pattern: "FES25_HT_2050_{month}_p2_bids_50pct", recovery_case_id: FES25_HT_2050}
      FES25_HT_2050_p2_bids_75pct: {<<: *fc, source_scenario_pattern: "FES25_HT_2050_{month}_p2_bids_75pct", recovery_case_id: FES25_HT_2050}
      FES25_HT_2050_p2_bids_100pct: {<<: *fc, source_scenario_pattern: "FES25_HT_2050_{month}_p2_bids_100pct", recovery_case_id: FES25_HT_2050}
public_ownership:
  network_account:
    model_years:
      2050:
        distribution_roll_forward: peak_demand
        transmission_roll_forward: declared
        roll_forward_inputs: data/market/network_cost_sources/rav_roll_forward_2050.yaml
        acquisition_year: 2030
        guarantee_pass_through_table: data/market/network_cost_sources/network_guarantee_pass_through_2050.csv
  reference_run_groups:
    fes_ht_2050:
      model_year: 2050
      mode: twelve_months
      scenario_pattern: "FES25_HT_2050_{month}_market"
      forecast_case: FES25_HT_2050
      fleet_vintages: &fleet {pathway: Holistic Transition, retirement_rule: economic_life_with_extension}
      vintage_costs: &costs {enabled: true, insurance: exclude, learning: false}
      institution_comparison:
        central_scenario_pattern: "FES25_HT_2050_{month}_central_dispatch"
        marginal_costs: resources/generators/FES25_HT_2050_marginal_costs_breakdown.csv
      sensitivities:
        bounding_pair: false
        one_at_a_time:
          - state_deployment_share: ["0.25", "0.50", "0.75", "1.00"]
          - private_rate_path: [neso_wind_ramp]
        named:
          fully_nationalised_market_value:
            scenario: public_ownership
            pre_2025_cfd_capacity: public_repriced
            legacy_fleet: acquire_all
            purchase_price_basis: market_value
          fully_nationalised_market_value_central_dispatch:
            extends: fully_nationalised_market_value
            dispatch_institution: nationalised_dispatch
          deployment_share_25_state_bids: {scenario: public_ownership, state_deployment_share: "0.25", balancing_bids: state_at_cost}
          deployment_share_50_state_bids: {scenario: public_ownership, state_deployment_share: "0.50", balancing_bids: state_at_cost}
          deployment_share_75_state_bids: {scenario: public_ownership, state_deployment_share: "0.75", balancing_bids: state_at_cost}
          deployment_share_100_state_bids: {scenario: public_ownership, state_deployment_share: "1.00", balancing_bids: state_at_cost}
    fes_ht_2050_public_bids_p2_25: &p2
      model_year: 2050
      mode: twelve_months
      scenario_pattern: "FES25_HT_2050_{month}_p2_bids_25pct"
      forecast_case: FES25_HT_2050_p2_bids_25pct
      balancing_bids: state_at_cost
      state_deployment_share: "0.25"
      sensitivities_from: fes_ht_2050
      fleet_vintages: *fleet
      vintage_costs: *costs
    fes_ht_2050_public_bids_p2_50: {<<: *p2, scenario_pattern: "FES25_HT_2050_{month}_p2_bids_50pct", forecast_case: FES25_HT_2050_p2_bids_50pct, state_deployment_share: "0.50"}
    fes_ht_2050_public_bids_p2_75: {<<: *p2, scenario_pattern: "FES25_HT_2050_{month}_p2_bids_75pct", forecast_case: FES25_HT_2050_p2_bids_75pct, state_deployment_share: "0.75"}
    fes_ht_2050_public_bids_p2_100: {<<: *p2, scenario_pattern: "FES25_HT_2050_{month}_p2_bids_100pct", forecast_case: FES25_HT_2050_p2_bids_100pct, state_deployment_share: "1.00"}
    fes_ht_2050_learning_bound:
      register_bound_of: fes_ht_2050
      model_year: 2050
      mode: twelve_months
      scenario_pattern: "FES25_HT_2050_{month}_market"
      forecast_case: FES25_HT_2050
      fleet_vintages: *fleet
      vintage_costs: {<<: *costs, learning: true}
      sensitivities: &bound_cases
        bounding_pair: false
        named:
          fully_nationalised_market_value:
            scenario: public_ownership
            pre_2025_cfd_capacity: public_repriced
            legacy_fleet: acquire_all
            purchase_price_basis: market_value
    fes_ht_2050_retirement_bound:
      register_bound_of: fes_ht_2050
      model_year: 2050
      mode: twelve_months
      scenario_pattern: "FES25_HT_2050_{month}_market"
      forecast_case: FES25_HT_2050
      fleet_vintages: {<<: *fleet, retirement_rule: strict_economic_life}
      vintage_costs: *costs
      sensitivities: *bound_cases
"""


def _merge(base, over):
    out = copy.deepcopy(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) \
            else copy.deepcopy(value)
    return out


@pytest.fixture(scope="module")
def defaults():
    return load_defaults()


@pytest.fixture(scope="module")
def merged(defaults):
    return _merge(defaults, yaml.safe_load(OVERLAY))


@pytest.fixture()
def po(merged):
    return copy.deepcopy(merged["public_ownership"])


@pytest.fixture(scope="module")
def forecast_cases(merged):
    return copy.deepcopy(merged["consumer_bill"]["forecasts"]["cases"])


@pytest.fixture(scope="module")
def po_2030(defaults):
    return copy.deepcopy(defaults["public_ownership"])


def _2050_groups(config):
    return {name: group for name, group in config["reference_run_groups"].items() if name.startswith("fes_ht_2050")}


# ── 2030 unchanged ───────────────────────────────────────────────────────────


def test_every_2030_group_keeps_the_global_list(po_2030, po):
    from scripts.public_ownership.reference_runs import (
        group_case_selection, institution_parent_group, institution_twin_cases, select_cases,
    )

    for name in po_2030["reference_run_groups"]:
        assert case_list_owner(po_2030, name) is None
        # Network bounds (18-19 Oct 2026): a 2030 group may keep only the headline cases of the global
        # list (case_selection); it never owns a list of its own.
        selection = group_case_selection(po_2030, name)
        expected = (expanded_cases(po_2030) if selection is None else
                    [spec.case_id for spec in select_cases(configured_case_specs(po_2030), selection)])
        if institution_parent_group(po_2030, name) is not None:
            # P1/P2 under central dispatch (20 Oct 2026): an institution twin keeps the market-bill cases only.
            expected = [spec.case_id for spec in institution_twin_cases(
                [s for s in configured_case_specs(po_2030) if s.case_id in expected])]
        assert expanded_cases(po_2030, name) == expected
        # adding the 2050 groups changes nothing a 2030 group or the global list reads
        assert expanded_cases(po, name) == expanded_cases(po_2030, name)
        assert group_market_cases(po, name) == group_market_cases(po_2030, name)
        assert group_excluded_cases(po, name) == group_excluded_cases(po_2030, name)
        assert group_tariff_cases(po, name) == group_tariff_cases(po_2030, name)
        assert group_bid_case(po, name) == group_bid_case(po_2030, name)
    assert expanded_cases(po) == expanded_cases(po_2030)
    assert market_bill_cases(po) == market_bill_cases(po_2030)
    assert tariff_cases(po) == tariff_cases(po_2030)
    assert institution_market_twin(po) == institution_market_twin(po_2030)
    assert institution_scenarios(po) == institution_scenarios(po_2030)
    # F28 (27 Oct 2026): the one 2030 register bound, the legislation year in force since 2025.
    assert register_bound_groups(po_2030) == ["cp30_2030_no_acceleration_legislated_2025"]


def test_the_2030_config_still_validates(po_2030, defaults):
    cases = copy.deepcopy(defaults["consumer_bill"]["forecasts"]["cases"])
    validate_public_ownership_config(po_2030, cases)


# ── the 2050 list ───────────────────────────────────────────────────────────


def test_the_overlay_validates_and_the_2050_group_takes_the_y14_list(po, forecast_cases):
    validate_public_ownership_config(po, forecast_cases)
    cases = expanded_cases(po, "fes_ht_2050")
    assert cases == [
        "private", "public_control", "public_ownership",
        *[f"public_ownership@deployment_share_{s}" for s in SHARES],
        "private@private_rate_neso_wind_ramp", "public_control@private_rate_neso_wind_ramp",
        "public_ownership@private_rate_neso_wind_ramp",
        "public_ownership@fully_nationalised_market_value",
        "public_ownership@fully_nationalised_market_value_central_dispatch",
        *[f"public_ownership@deployment_share_{s}_state_bids" for s in SHARES],
    ]
    # the market group projects the contract-value market cases, tariffs P3 alt, and
    # pairs P3 central with P3 alt on the HT50 central-dispatch year
    assert group_market_cases(po, "fes_ht_2050") == cases[:10]
    assert group_tariff_cases(po, "fes_ht_2050") == ["public_ownership@fully_nationalised_market_value"]
    assert institution_market_twin(po, "fes_ht_2050") == (
        "public_ownership@fully_nationalised_market_value_central_dispatch",
        "public_ownership@fully_nationalised_market_value")
    assert institution_scenarios(po, "fes_ht_2050")[0] == "FES25_HT_2050_Jan_central_dispatch"
    by = {spec.case_id: spec for spec in configured_case_specs(po, "fes_ht_2050")}
    assert by["public_ownership@deployment_share_50_state_bids"].settings["state_deployment_share"] == "0.50"
    assert "state_deployment_share" not in by["public_ownership"].settings


def test_each_p2_share_group_projects_its_own_share_and_bids_on_its_table(po, forecast_cases):
    groups = consumer_bill_groups(po, forecast_cases)
    for pct in SHARES:
        name = f"fes_ht_2050_public_bids_p2_{pct}"
        assert name in groups and case_list_owner(po, name) == "fes_ht_2050"
        assert group_market_cases(po, name) == ["private", f"public_ownership@deployment_share_{pct}_state_bids"]
        assert group_bid_case(po, name) == f"public_ownership@deployment_share_{pct}"
        assert group_tariff_cases(po, name) == []
        excluded = group_excluded_cases(po, name)
        assert "public_ownership@deployment_share_25_state_bids" in excluded or pct == "25"


def test_the_share_groups_months_must_bid_on_their_share(po):
    name = "fes_ht_2050_public_bids_p2_50"
    good = {f"FES25_HT_2050_{m}_p2_bids_50pct": {"support": {"bm_bids": {
        "contract_value_bids": {"public_ownership_bids": {"enabled": True,
                                                          "case_id": "public_ownership@deployment_share_50"}}}}}
        for m in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")}
    check_balancing_group_scenarios(po, name, good)
    bad = copy.deepcopy(good)
    bad["FES25_HT_2050_Jan_p2_bids_50pct"]["support"]["bm_bids"][
        "contract_value_bids"]["public_ownership_bids"]["case_id"] = "public_ownership"
    with pytest.raises(ValueError, match="deployment_share_50"):
        check_balancing_group_scenarios(po, name, bad)


def test_the_bounds_re_account_the_parent_months(po):
    vintaged = [g for g in register_bound_groups(po) if not g.startswith("cp30_2030")]   # F28's 2030 bound aside
    assert vintaged == ["fes_ht_2050_learning_bound", "fes_ht_2050_retirement_bound"]
    for name in vintaged:
        assert expanded_cases(po, name) == ["private", "public_control", "public_ownership",
                                            "public_ownership@fully_nationalised_market_value"]
        assert group_tariff_cases(po, name) == ["public_ownership@fully_nationalised_market_value"]


# ── refusals ────────────────────────────────────────────────────────────────


def _refused(po, forecast_cases, match):
    with pytest.raises(ValueError, match=match):
        validate_public_ownership_config(po, forecast_cases)


def test_two_groups_on_one_share_are_refused(po, forecast_cases):
    po["reference_run_groups"]["fes_ht_2050_public_bids_p2_50"]["state_deployment_share"] = "0.25"
    _refused(po, forecast_cases, "0.25")


def test_a_share_without_a_group_is_refused(po, forecast_cases):
    del po["reference_run_groups"]["fes_ht_2050_public_bids_p2_75"]
    _refused(po, forecast_cases, r"state_deployment_share='0.75', so exactly one bill group")


def test_an_unkeyed_group_beside_keyed_ones_is_ambiguous(po, forecast_cases):
    del po["reference_run_groups"]["fes_ht_2050_public_bids_p2_100"]["state_deployment_share"]
    _refused(po, forecast_cases, "exactly one bill group")


def test_the_share_key_is_refused_on_2030_and_on_contract_value(po, forecast_cases, po_2030, defaults):
    cfg = copy.deepcopy(po_2030)
    cfg["reference_run_groups"]["cp30_2030_cp30_network_public_bids_p2"]["state_deployment_share"] = "0.50"
    _refused(cfg, copy.deepcopy(defaults["consumer_bill"]["forecasts"]["cases"]), "applies from model year 2040")
    po["reference_run_groups"]["fes_ht_2050"]["state_deployment_share"] = "0.50"
    _refused(po, forecast_cases, "keys a P2 balancing group")


def test_a_share_value_outside_l4_is_refused(po, forecast_cases):
    po["reference_run_groups"]["fes_ht_2050_public_bids_p2_25"]["state_deployment_share"] = "0.30"
    _refused(po, forecast_cases, "supported")


def test_a_share_group_needs_its_table_case_in_the_list(po, forecast_cases):
    block = po["reference_run_groups"]["fes_ht_2050"]["sensitivities"]
    block["one_at_a_time"][0] = {"state_deployment_share": ["0.50", "0.75", "1.00"]}
    _refused(po, forecast_cases, "public_ownership@deployment_share_25")


def test_a_group_list_error_names_the_group(po, forecast_cases):
    po["reference_run_groups"]["fes_ht_2050"]["sensitivities"]["one_at_a_time"].append("not_a_lever")
    _refused(po, forecast_cases, "reference_run_groups.fes_ht_2050.sensitivities")


def test_a_gated_lever_in_a_2030_group_list_is_refused(po_2030, defaults):
    cfg = copy.deepcopy(po_2030)
    cfg["reference_run_groups"]["historical_2024"]["sensitivities"] = {
        "one_at_a_time": [{"private_rate_path": ["neso_wind_ramp"]}]}
    _refused(cfg, copy.deepcopy(defaults["consumer_bill"]["forecasts"]["cases"]), "gated lever")


def test_sensitivities_from_must_name_a_list_owner(po, forecast_cases):
    po["reference_run_groups"]["fes_ht_2050_public_bids_p2_25"]["sensitivities_from"] = "cp30_2030_cp30_network"
    _refused(po, forecast_cases, "sensitivities_from")


def test_both_list_keys_are_refused(po, forecast_cases):
    po["reference_run_groups"]["fes_ht_2050_public_bids_p2_25"]["sensitivities"] = {}
    _refused(po, forecast_cases, "not both")


def test_a_group_list_needs_its_institution_pair(po, forecast_cases):
    del po["reference_run_groups"]["fes_ht_2050"]["sensitivities"]["named"]["fully_nationalised_market_value"]
    po["reference_run_groups"]["fes_ht_2050"]["sensitivities"]["named"][
        "fully_nationalised_market_value_central_dispatch"] = {
        "scenario": "public_ownership", "pre_2025_cfd_capacity": "public_repriced", "legacy_fleet": "acquire_all",
        "purchase_price_basis": "market_value", "dispatch_institution": "nationalised_dispatch"}
    _refused(po, forecast_cases, "identical but for dispatch_institution")


def test_a_bound_that_moves_nothing_or_other_months_is_refused(po, forecast_cases):
    bound = po["reference_run_groups"]["fes_ht_2050_learning_bound"]
    bound["vintage_costs"] = dict(bound["vintage_costs"], learning=False)
    _refused(po, forecast_cases, "bounds nothing")


def test_a_bound_on_other_months_is_refused(po, forecast_cases):
    po["reference_run_groups"]["fes_ht_2050_retirement_bound"]["scenario_pattern"] = "FES25_HT_2050_{month}_p2_bids_25pct"
    _refused(po, forecast_cases, "same solved months")


def test_a_bound_of_a_2030_group_is_refused(po, forecast_cases):
    po["reference_run_groups"]["fes_ht_2050_retirement_bound"]["register_bound_of"] = "cp30_2030_cp30_network"
    _refused(po, forecast_cases, "vintaged groups")
