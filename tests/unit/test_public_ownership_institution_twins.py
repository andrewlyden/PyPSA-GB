"""P1 and P2 under central dispatch (user decisions of 14 and 20 Oct 2026): institution twin run groups.

An institution twin (``reference_run_groups.<twin>.institution_of: <market group>``) re-projects
a contract-value market group's market-bill cases on that group's central-dispatch year (I3,
FTI's settlement), under its own forecast case (``institution: central_dispatch``). These tests
hold the config checks, the case list (market-bill cases only: no P3, no state bids), that the
twin takes no role a market group has, and that every model-price group is unchanged.
"""

import copy
from pathlib import Path

import pytest
import yaml

from scripts.public_ownership import reference_runs as rr
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
MARKET, P2 = "cp30_2030_no_acceleration", "cp30_2030_no_acceleration_public_bids_p2"
TWIN = f"{MARKET}_cd"
TWIN_CASE = "CP30_2030_no_acceleration_cd"
#: Decision 5 (20 Oct 2026): scenarios, named cases and presets with their comparators; no grids.
SELECTION = {"kinds": ["scenario", "named", "preset"]}


@pytest.fixture(scope="module")
def defaults():
    return load_defaults()


def with_twin(defaults):
    """The live block and forecast cases with the central network's institution twin (built in memory if absent)."""
    po = copy.deepcopy(defaults["public_ownership"])
    cases = copy.deepcopy(defaults["consumer_bill"]["forecasts"]["cases"])
    groups = po["reference_run_groups"]
    if TWIN not in groups:
        block = rr.institution_block(po, MARKET)
        cases[TWIN_CASE] = {**copy.deepcopy(cases["CP30_2030_no_acceleration"]), "institution": "central_dispatch",
                            "institution_parent": "CP30_2030_no_acceleration",
                            "central_scenario_pattern": block["central_scenario_pattern"],
                            "fti_settlement": block["fti_settlement"]}
        groups[TWIN] = {**copy.deepcopy(groups[MARKET]), "institution_of": MARKET, "forecast_case": TWIN_CASE,
                        "case_selection": copy.deepcopy(SELECTION)}
    for case_id, case in cases.items():
        case["case_id"] = case_id
    return po, cases


@pytest.fixture(scope="module")
def twinned(defaults):
    po, cases = with_twin(defaults)
    rr.validate_public_ownership_config(po, cases)
    return po, cases


def test_the_twin_validates_and_carries_market_bill_cases_only(twinned):
    po, cases = twinned
    # The LOWER and UPPER networks' twins (24 Oct 2026) beside CENTRAL's.
    assert rr.institution_twin_groups(po) == {TWIN: MARKET, "cp30_2030_cp30_network_cd": "cp30_2030_cp30_network",
                                              "cp30_2030_no_upgrades_cd": "cp30_2030_no_upgrades"}
    assert rr.institution_parent_group(po, TWIN) == MARKET and rr.institution_parent_group(po, MARKET) is None
    specs = rr.configured_case_specs(po, TWIN)
    assert specs and all(spec.market_bill() for spec in specs)
    ids = [spec.case_id for spec in specs]
    assert "private" in ids and "public_control" in ids and "public_ownership" in ids
    assert "public_ownership@state_bids_at_cost" not in ids                # decision 7
    assert not any("fully_nationalised" in case for case in ids)          # P3 has its own I3 path
    assert not any("_grid_" in case for case in ids)                      # decision 5: no grids
    assert rr.group_tariff_cases(po, TWIN) == [] and rr.institution_cases(po, TWIN) == []
    assert rr.group_market_cases(po, TWIN) == ids
    # The twin's cases carry exactly the full list's settings.
    full = {spec.case_id: spec.settings for spec in rr.configured_case_specs(po)}
    assert all(dict(spec.settings) == dict(full[spec.case_id]) for spec in specs)
    assert rr.institution_twin_central_scenarios(po, TWIN) == rr.institution_scenarios(po, MARKET)


def test_the_twin_takes_no_market_group_role(twinned):
    po, cases = twinned
    # A balancing group is still measured against the model-price market group alone.
    assert rr.contract_value_group_for(po, P2, cases) == MARKET
    assert TWIN in rr.consumer_bill_groups(po, cases)


def test_the_model_price_groups_are_unchanged_by_the_twin(defaults, twinned):
    po, cases = twinned
    plain = copy.deepcopy(defaults["public_ownership"])
    plain["reference_run_groups"].pop(TWIN, None)
    for group in (MARKET, P2):
        assert rr.expanded_cases(po, group) == rr.expanded_cases(plain, group)


def _check(po, cases):
    rr._validate_institution_twins(po, cases)


def test_a_twin_must_repeat_its_parent_and_name_its_own_case(twinned):
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"][TWIN]["scenario_pattern"] = "Other_{month}2030"
    with pytest.raises(ValueError, match="repeats its parent"):
        _check(po, cases)
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"][TWIN]["forecast_case"] = "CP30_2030_no_acceleration"
    with pytest.raises(ValueError, match="names its own forecast_case"):
        _check(po, cases)


def test_a_twins_case_must_declare_central_dispatch_on_the_parents_months(twinned):
    po, cases = copy.deepcopy(twinned)
    cases[TWIN_CASE].pop("institution")
    with pytest.raises(ValueError, match="institution: central_dispatch"):
        _check(po, cases)
    po, cases = copy.deepcopy(twinned)
    cases[TWIN_CASE]["institution_parent"] = "CP30_2030_cp30_network"
    with pytest.raises(ValueError, match="institution_parent"):
        _check(po, cases)
    po, cases = copy.deepcopy(twinned)
    cases[TWIN_CASE]["central_scenario_pattern"] = "CP30_FFR_2030_{month}_cp30_network_central_dispatch"
    with pytest.raises(ValueError, match="central_scenario_pattern"):
        _check(po, cases)
    po, cases = copy.deepcopy(twinned)
    cases[TWIN_CASE]["fti_settlement"] = "resources/market/other.csv"
    with pytest.raises(ValueError, match="fti_settlement"):
        _check(po, cases)
    po, cases = copy.deepcopy(twinned)
    cases[TWIN_CASE]["price_level"] = "cp32"
    with pytest.raises(ValueError, match="model's own price level"):
        _check(po, cases)


def test_only_a_contract_value_market_group_with_an_i3_year_can_be_twinned(twinned):
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"][TWIN]["institution_of"] = P2
    with pytest.raises(ValueError):
        _check(po, cases)
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"]["second_cd"] = copy.deepcopy(po["reference_run_groups"][TWIN])
    with pytest.raises(ValueError, match="two institution twins"):
        _check(po, cases)
    po, cases = copy.deepcopy(twinned)
    for name in (MARKET, TWIN):
        po["reference_run_groups"][name].pop("institution_comparison")
    po.pop("institution_comparison")
    with pytest.raises(ValueError, match="central-dispatch year"):
        _check(po, cases)


def test_the_workflow_keeps_the_twin_out_of_market_roles_and_meters_it_on_the_central_months():
    """The rule file's wiring (rules/public_ownership.smk, rules/consumer_bill.smk), read as text."""
    po = (ROOT / "rules/public_ownership.smk").read_text(encoding="utf-8")
    bill = (ROOT / "rules/consumer_bill.smk").read_text(encoding="utf-8")

    def block(text, rule):
        start = text.index(f"rule {rule}:")
        end = text.find("\nrule ", start + 1)
        return text[start:end]

    # Solve-side tables, the market settlement and the bid-stack pairing never match an institution twin.
    for name in ("PUBLIC_OWNERSHIP_SOLVED_GROUP_REGEX", "PUBLIC_OWNERSHIP_SOLVED_MARKET_BID_GROUP_REGEX",
                 "PUBLIC_OWNERSHIP_UNVINTAGED_BILL_GROUP_REGEX"):
        definition = po[po.index(f"{name} = "):po.index(" or r\"(?!x)x\"", po.index(f"{name} = "))]
        assert "_PO_INSTITUTION_TWINS" in definition, name
    pairing = po[po.index("def _po_market_bid_group_for"):po.index("def _po_group_months")]
    assert "g not in _PO_INSTITUTION_TWINS" in pairing
    assert "run_group=PUBLIC_OWNERSHIP_INSTITUTION_TWIN_REGEX" in block(po, "build_public_ownership_institution_twin_settlement")
    # The CfD paid energy, the per-case bill and the ledger's carbon on the physical (central) months.
    for rule in ("build_public_ownership_cfd_strike_adjustments", "project_public_ownership_consumer_bill"):
        assert "_po_physical_scenarios(wc)" in block(po, rule), rule
    assert 'market_scenarios=lambda wc: " ".join(_po_physical_scenarios(wc))' in block(
        po, "build_public_ownership_exchequer_ledger")
    # The forecast bill: the I3 extraction for a central-dispatch case only, CfD and RO on the physical months.
    assert "bill_case=_consumer_regex(_consumer_market_institution_cases)" in block(bill, "extract_future_balancing_costs")
    assert "bill_case=_consumer_regex(_consumer_central_dispatch_cases)" in block(bill, "extract_central_dispatch_costs")
    for rule in ("build_future_ro_scheme_cost", "project_future_consumer_bill"):
        assert '["physical_scenarios"]' in block(bill, rule), rule
    # The FTI settlement writes the per-unit transfers the twin settles on.
    assert 'units="{fti_stem}_units.csv"' in po and "--units-out {output.units}" in po
