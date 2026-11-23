"""FES 2050 plan, decision 31 (task D31): the 2050 finance settings live on the FES25_HT_2050 scenario entry.

A study is a parent scenario entry (no ``network_from``) carrying ``public_ownership`` and/or
``consumer_bill.forecasts``; its groups and forecast cases resolve as config/defaults/ with
those blocks deep-merged over it.  Every group declared in defaults resolves to defaults itself,
so no 2030 case list, setting or file can move.
"""

import copy
import json
import os
from pathlib import Path

import pytest
import yaml

from scripts.public_ownership import reference_runs as rr
from scripts.public_ownership.study_config import (
    FINANCE_BLOCKS, StudyConfigError, deep_merge, dump_config, finance_studies, load_studies, main,
)
from config.config_loader import load_scenario_entries
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
STUDY = "FES25_HT_2050"
#: User, 10 Nov 2026: the 2025-law register bound (fes_ht_2050_legislated_2025) after the P2 stacks.
GROUPS_2050 = ["fes_ht_2050", "fes_ht_2050_public_bids_p2_25", "fes_ht_2050_public_bids_p2_50",
               "fes_ht_2050_public_bids_p2_75", "fes_ht_2050_public_bids_p2_100", "fes_ht_2050_legislated_2025",
               "fes_ht_2050_learning_bound", "fes_ht_2050_retirement_bound", "fes_ht_2050_repowering_bound"]
#: FES 2040 (11 Nov 2026) is the second finance study.
STUDY_2040 = "FES25_HT_2040"
#: Groups the held HT50 patch adds after the others: decision 68's co-ownership bounds and decision 71's
#: auction-margin bounds.
ROUND3_BOUNDS = ("fes_ht_2050_coownership_reduction_bound", "fes_ht_2050_coownership_pooled_bound",
                 "fes_ht_2050_auction_margin_none_bound", "fes_ht_2050_auction_margin_high_bound")
# Decision 70 (25 Oct 2026): FES25_HT_2050_cd, the private market bill on the central-dispatch year (no run group).
CASES_2050 = ["FES25_HT_2050", "FES25_HT_2050_cd", "FES25_HT_2050_p2_bids_25pct", "FES25_HT_2050_p2_bids_50pct",
              "FES25_HT_2050_p2_bids_75pct", "FES25_HT_2050_p2_bids_100pct"]
SHARES = ("25", "50", "75", "100")
#: The E2E smoke test's merged config (``scratchpad/E2E/cfg/defaults.yaml``), when available.
E2E_CONFIG_ENV = "D31_E2E_CONFIG"
#: The scratch-only additions the E2E harness made on top of the 2050 settings (make_config.py):
#: the annualised gate-week forecast case and run group, and its two register bounds.
E2E_SHIM_CASES = ("HT50_gate_ann",)
E2E_SHIM_GROUP_PREFIX = "fes_ht_2050_gate"


def _load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def defaults():
    return load_defaults(ROOT / "config")


@pytest.fixture(scope="module")
def raw_scenarios():
    return load_scenario_entries(ROOT / "config")


@pytest.fixture(scope="module")
def studies(defaults):
    return load_studies(ROOT / "config", copy.deepcopy(defaults))


# ── 2030: every defaults group resolves to defaults, unchanged ───────────────


def test_every_defaults_group_resolves_to_defaults_itself(studies):
    groups = studies.defaults["public_ownership"]["reference_run_groups"]
    assert groups, "defaults declares the 2030 groups"
    for group in groups:
        assert studies.study_of(group) is None
        assert studies.config(group) is studies.defaults
        assert studies.public_ownership(group) is studies.defaults["public_ownership"]
    # an unknown name (a scratch output's group) also reads defaults, as configured_group does
    assert studies.config("not_a_group") is studies.defaults


def test_2030_case_lists_and_settings_are_byte_identical(studies, defaults):
    po = defaults["public_ownership"]
    cases = defaults["consumer_bill"]["forecasts"]["cases"]
    for group in po["reference_run_groups"]:
        resolved = studies.public_ownership(group)
        assert json.dumps(resolved, sort_keys=True, default=str) == json.dumps(po, sort_keys=True, default=str)
        assert rr.expanded_cases(resolved, group) == rr.expanded_cases(po, group)
        assert rr.market_bill_cases(resolved, group) == rr.market_bill_cases(po, group)
        assert rr.tariff_cases(resolved, group) == rr.tariff_cases(po, group)
        assert rr.institution_block(resolved, group) == rr.institution_block(po, group)
    assert rr.expanded_cases(studies.public_ownership()) == rr.expanded_cases(po)
    assert rr.consumer_bill_groups(po, cases) == rr.consumer_bill_groups(studies.public_ownership(), cases)
    # the study added nothing to the defaults it was given
    assert studies.defaults == defaults


def test_no_study_config_touches_a_defaults_group(studies, defaults):
    for config in studies.configs.values():
        for group, entry in defaults["public_ownership"]["reference_run_groups"].items():
            assert config["public_ownership"]["reference_run_groups"][group] == entry
        for case, entry in defaults["consumer_bill"]["forecasts"]["cases"].items():
            assert config["consumer_bill"]["forecasts"]["cases"][case] == entry


# ── the 2050 study ───────────────────────────────────────────────────────────


def test_ht50_2050_is_the_one_study_and_declares_the_2050_groups(studies):
    # Since 11 Nov 2026 FES 2040 is a second study; each study keeps its own groups.
    assert sorted(studies.configs) == sorted([STUDY, STUDY_2040])   # load order follows config.yaml or PYPSA_GB_STUDIES
    assert not set(studies.groups(STUDY)) & set(studies.groups(STUDY_2040))
    # Round 3 (decision 68): the co-ownership bounds arrive with the HT50 patch; listed after these.
    assert [g for g in studies.groups(STUDY) if g not in ROUND3_BOUNDS] == GROUPS_2050
    assert studies.cases(STUDY) == CASES_2050
    for group in GROUPS_2050:
        assert studies.study_of(group) == STUDY
        assert studies.config(group) is studies.configs[STUDY]


def test_the_2050_group_takes_its_16_case_list_and_institution_override(studies):
    # User, 10 Nov 2026: 2050 runs the 2030 case set (sensitivities.extend_global), so the original 16 cases are
    # now a subset of a much longer list, still first among them the three scenarios; the named 11 Nov case
    # (today's prices) is in it.
    po = studies.public_ownership("fes_ht_2050")
    cases = rr.expanded_cases(po, "fes_ht_2050")
    assert cases[:3] == ["private", "public_control", "public_ownership"] and len(cases) > 400
    assert "public_ownership@fully_nationalised_market_value_price_path_today" in cases
    original = [
        "private", "public_control", "public_ownership",
        *[f"public_ownership@deployment_share_{s}" for s in SHARES],
        "private@private_rate_neso_wind_ramp", "public_control@private_rate_neso_wind_ramp",
        "public_ownership@private_rate_neso_wind_ramp",
        # User, 3 Oct 2026, the public rate +1 pp; since F29 (26 Oct 2026) workbook v2.0's PWLB standard
        # margin, on P2 only (P1's rate is the workbook's stake rate, not the cash rate).
        "public_ownership@public_rate_pwlb_standard",
        # Decision 73's sensitivity on P2 (the state's plant in the full auction), one at a time.
        "public_ownership@cm_market_participant",
        # F28 (c): the state owning and re-pricing pre-2025 CfDs is a switch in 2050 as in 2030.
        "public_control@ownership_all_cfd_repriced", "public_ownership@ownership_all_cfd_repriced",
        "public_ownership@fully_nationalised_market_value",
        "public_ownership@fully_nationalised_market_value_central_dispatch",
        # Decision 73's sensitivity on P3: today's kept CM line instead of adequacy planning.
        "public_ownership@fully_nationalised_market_value_cm_market_participant",
        *[f"public_ownership@deployment_share_{s}_state_bids" for s in SHARES],
    ]
    assert [c for c in original if c not in cases] == []
    block = rr.institution_block(po, "fes_ht_2050")
    assert block["central_scenario_pattern"] == "FES25_HT_2050_{month}_central_dispatch"
    assert block["marginal_costs"] == "resources/generators/FES25_HT_2050_marginal_costs_breakdown.csv"
    # User, 11 Nov 2026: over the benchmark's discount-weighted 2030-2040 fleet (was 2.32, decision 18).
    assert block["forecast_error_charge"]["fleet_ratio"] == 1.714
    assert rr.institution_market_twin(po, "fes_ht_2050") == (
        "public_ownership@fully_nationalised_market_value_central_dispatch",
        "public_ownership@fully_nationalised_market_value")
    # the 2030 global block is untouched for the 2030 groups
    assert rr.institution_block(studies.public_ownership(), "cp30_2030_cp30_network")["central_scenario_pattern"] != \
        block["central_scenario_pattern"]
    year = po["network_account"]["model_years"][2050]
    assert year["roll_forward_inputs"] == "data/market/network_cost_sources/rav_roll_forward_2050.yaml"
    assert year["acquisition_year"] == 2030
    assert "model_years" not in studies.public_ownership()["network_account"]


def test_the_2050_bill_groups_and_bounds(studies):
    po = studies.public_ownership("fes_ht_2050")
    cases = studies.forecast_cases("fes_ht_2050")
    bill = [g for g in rr.consumer_bill_groups(po, cases) if g in GROUPS_2050]
    assert bill == GROUPS_2050
    # The 2025-law group (user, 10 Nov 2026) is a register bound of fes_ht_2050 too.
    assert [g for g in rr.register_bound_groups(po) if g in GROUPS_2050] == [
        "fes_ht_2050_legislated_2025", "fes_ht_2050_learning_bound", "fes_ht_2050_retirement_bound",
        "fes_ht_2050_repowering_bound"]
    assert [g for g in rr.vintaged_groups(po) if g in GROUPS_2050] == GROUPS_2050
    for share in SHARES:
        group = f"fes_ht_2050_public_bids_p2_{share}"
        assert rr.group_forecast_case_id(po, group) == f"FES25_HT_2050_p2_bids_{share}pct"
        assert rr.group_bid_case(po, group) == f"public_ownership@deployment_share_{share}"
    assert cases["FES25_HT_2050"]["modelled_year"] == 2050
    assert cases["FES25_HT_2050_p2_bids_25pct"]["recovery_case_id"] == "FES25_HT_2050"


def _at(entry, path):
    for part in path:
        entry = (entry or {}).get(part)
    return entry


def test_the_months_inherit_the_blocks_and_are_not_studies(raw_scenarios):
    parent = {path: _at(raw_scenarios[STUDY], path) for path in FINANCE_BLOCKS}
    assert all(value is not None for value in parent.values())
    children = [s for s, e in raw_scenarios.items() if isinstance(e, dict) and e.get("network_from") == STUDY]
    assert len(children) >= 24
    for child in children:
        assert {path: _at(raw_scenarios[child], path) for path in FINANCE_BLOCKS} == parent, child


def test_the_cli_writes_the_resolved_config(studies, tmp_path):
    out = tmp_path / "FES25_HT_2050_finance_config.yaml"
    main(["--study", STUDY, "--out", str(out), "--config-dir", str(ROOT / "config")])
    assert out.read_text(encoding="utf-8") == dump_config(studies.configs[STUDY])
    assert _load(out) == studies.configs[STUDY]


@pytest.mark.skipif(not os.environ.get(E2E_CONFIG_ENV), reason=f"set {E2E_CONFIG_ENV} to the E2E harness config")
def test_the_2050_group_resolves_to_what_the_e2e_harness_ran_on(studies):
    """Acceptance (1): the E2E smoke test's config, less its scratch-only gate-week shims."""
    e2e = _load(os.environ[E2E_CONFIG_ENV])
    for case in E2E_SHIM_CASES:
        del e2e["consumer_bill"]["forecasts"]["cases"][case]
    groups = e2e["public_ownership"]["reference_run_groups"]
    for group in [g for g in groups if g.startswith(E2E_SHIM_GROUP_PREFIX)]:
        del groups[group]
    assert studies.config("fes_ht_2050") == e2e


# ── the resolver's rules, on synthetic configs ───────────────────────────────


def _defaults():
    return {"other": {"a": 1},
            "public_ownership": {"enabled": True, "network_account": {"x": 1},
                                 "reference_run_groups": {"cp30_2030_cp30_network": {"mode": "twelve_months"}}},
            "consumer_bill": {"forecasts": {"cases": {"CP30_2030_cp30_network": {"modelled_year": 2030}}}}}


def _study(**po):
    return {"public_ownership": {"reference_run_groups": {"g50": {"mode": "full_year"}}, **po}}


def test_deep_merge_merges_mappings_and_replaces_the_rest():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    merged = deep_merge(base, {"a": {"c": [3], "e": 2}, "d": {"f": 1}})
    assert merged == {"a": {"b": 1, "c": [3], "e": 2}, "d": {"f": 1}}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 1}


def test_a_study_resolves_over_defaults():
    studies = finance_studies(_defaults(), {"S": _study(network_account={"y": 2})}, validate=False)
    config = studies.config("g50")
    assert config["other"] == {"a": 1}
    assert config["public_ownership"]["network_account"] == {"x": 1, "y": 2}
    assert set(config["public_ownership"]["reference_run_groups"]) == {"cp30_2030_cp30_network", "g50"}
    assert studies.config("cp30_2030_cp30_network") is studies.defaults


@pytest.mark.parametrize("scenarios, message", [
    ({"S": {"public_ownership": {"reference_run_groups": {"cp30_2030_cp30_network": {}}}}}, "redefines run group"),
    ({"S": {"consumer_bill": {"forecasts": {"cases": {"CP30_2030_cp30_network": {}}}}}}, "redefines forecast case"),
    ({"S": _study(), "T": _study()}, "declared by S and T"),
    ({"S": _study(enabled=False)}, "a study may not change it"),
    ({"S": _study(), "S_Jan": {"network_from": "S", "public_ownership": {"price_year": 1}}}, "parent entry S"),
    ({"S": {"public_ownership": [1]}}, "must be a mapping"),
])
def test_bad_study_blocks_are_refused(scenarios, message):
    with pytest.raises(StudyConfigError, match=message):
        finance_studies(_defaults(), scenarios, validate=False)


def test_a_child_carrying_its_parents_blocks_is_not_a_second_study():
    parent = _study()
    studies = finance_studies(_defaults(), {"S": parent, "S_Jan": {**copy.deepcopy(parent), "network_from": "S"}},
                              validate=False)
    assert list(studies.configs) == ["S"]
    assert studies.study_of("g50") == "S"


def test_a_bad_resolved_config_fails_validation(defaults):
    bad = {STUDY: {"public_ownership": {"reference_run_groups": {"fes_bad": {"mode": "weekly"}}}}}
    with pytest.raises(StudyConfigError, match="fes_bad"):
        finance_studies(copy.deepcopy(defaults), bad)
