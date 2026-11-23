"""Network bounds of the 2030 bill and public-ownership layer (user decision, 18-19 Oct 2026).

Three network-delivery cases, each a market group with its P2 and P3 bid stacks and a CP.32 twin
of each: CENTRAL = no_acceleration (every case), LOWER = the recommended CP30 network (the existing
cp30_2030 family) and UPPER = no_upgrades (the headline cases).  The purchase price and the auction
calibration come from CENTRAL and are held across the bounds; TNUoS moves per network.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership import reference_runs as rr
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
NETWORKS = {"cp30": "cp30_2030_cp30_network", "no_acceleration": "cp30_2030_no_acceleration",
            "no_upgrades": "cp30_2030_no_upgrades"}
MARKET_TAGS = {"cp30": "cp30_network", "no_acceleration": "no_acceleration", "no_upgrades": "no_upgrades"}
BID_TAGS = {"cp30": "cp30_network_{p}_bids", "no_acceleration": "no_acceleration_{p}_bids",
            "no_upgrades": "no_upgrades_{p}_bids"}
CENTRAL_TAGS = {"cp30": "cp30_network_central_dispatch", "no_acceleration": "no_acceleration_central_dispatch",
                "no_upgrades": "no_upgrades_central_dispatch"}
PATTERN = "CP30_FFR_2030_{{month}}_{tag}"


@pytest.fixture(scope="module")
def defaults():
    return load_defaults()


@pytest.fixture(scope="module")
def po(defaults):
    return defaults["public_ownership"]


@pytest.fixture(scope="module")
def forecast_cases(defaults):
    return defaults["consumer_bill"]["forecasts"]["cases"]


@pytest.fixture(scope="module")
def full_cases(po):
    return rr.expanded_cases(po)


def _groups_of(network):
    base = NETWORKS[network]
    return {"market": base, "p2": f"{base}_public_bids_p2", "p3": f"{base}_public_bids_p3",
            "market_cp32": f"{base}_cp32", "p2_cp32": f"{base}_public_bids_p2_cp32",
            "p3_cp32": f"{base}_public_bids_p3_cp32"}


@pytest.mark.parametrize("network", sorted(NETWORKS))
def test_each_network_has_its_six_groups_on_its_own_months(po, forecast_cases, network):
    groups = po["reference_run_groups"]
    names = _groups_of(network)
    expected = {"market": MARKET_TAGS[network], "market_cp32": MARKET_TAGS[network],
                "p2": BID_TAGS[network].format(p="p2"), "p2_cp32": BID_TAGS[network].format(p="p2"),
                "p3": BID_TAGS[network].format(p="p3"), "p3_cp32": BID_TAGS[network].format(p="p3")}
    bill_groups = rr.consumer_bill_groups(po, forecast_cases)
    for role, name in names.items():
        group = groups[name]
        assert group["scenario_pattern"] == PATTERN.format(tag=expected[role]), name
        assert rr.group_network_case(group) == network, name
        assert name in bill_groups, name
        case = forecast_cases[rr.group_forecast_case_id(po, name)]
        assert case["source_scenario_pattern"] == group["scenario_pattern"], name
        assert str(case.get("price_level", "model")) == ("cp32" if role.endswith("cp32") else "model"), name
        # Every case reads the demand year's recovery denominators.
        assert case.get("recovery_case_id", "CP30_2030_cp30_network") == "CP30_2030_cp30_network", name
        assert case["tnuos_network_delivery"]["case"] == network, name
    assert groups[names["p3"]]["bid_case"] == "public_ownership@fully_nationalised_market_value"
    assert rr.group_balancing_bids(groups[names["p2"]]) == "state_at_cost"


def test_the_lower_network_groups_name_their_network(po, forecast_cases):
    """The LOWER network's group and its twins name the network, as the other networks' do."""
    groups = po["reference_run_groups"]
    assert groups["cp30_2030_cp30_network"]["scenario_pattern"] == PATTERN.format(tag="cp30_network")
    assert rr.group_forecast_case_id(po, "cp30_2030_cp30_network") == "CP30_2030_cp30_network"
    assert rr.group_forecast_case_id(po, "cp30_2030_cp30_network_public_bids_p2") == "CP30_2030_cp30_network_public_bids_p2"
    assert rr.group_forecast_case_id(po, "cp30_2030_cp30_network_cp32") == "CP30_2030_cp30_network_cp32"
    assert po["consumer_bill"]["forecast_case"] == "CP30_2030_cp30_network"


def test_central_carries_every_case_and_the_bounds_the_headline_cases(po, full_cases):
    groups = po["reference_run_groups"]
    assert rr.expanded_cases(po, "cp30_2030_no_acceleration") == full_cases
    assert rr.group_case_selection(po, "cp30_2030_no_acceleration") is None
    headline = rr.expanded_cases(po, "cp30_2030_cp30_network_cp32")
    assert 50 <= len(headline) < 100 and len(headline) < len(full_cases)
    assert "public_ownership@state_bids_at_cost" in headline
    assert "public_ownership@fully_nationalised_market_value_central_dispatch" in headline
    for network in ("cp30", "no_upgrades"):
        for name in _groups_of(network).values():
            assert rr.expanded_cases(po, name) == headline, name
    for role in ("market_cp32", "p2_cp32", "p3_cp32"):
        assert rr.expanded_cases(po, _groups_of("no_acceleration")[role]) == headline
    assert groups["cp30_2030_cp30_network"]["case_selection"] == groups["cp30_2030_cp30_network_cp32"]["case_selection"]


@pytest.mark.parametrize("network", sorted(NETWORKS))
def test_each_stack_is_measured_against_its_own_networks_market_group(po, forecast_cases, network):
    names = _groups_of(network)
    for role in ("p2", "p3"):
        assert rr.contract_value_group_for(po, names[role], forecast_cases) == names["market"]
        assert rr.contract_value_group_for(po, names[f"{role}_cp32"], forecast_cases) == names["market_cp32"]
    assert rr.contract_value_group_for(po, names["market_cp32"], forecast_cases) == names["market_cp32"]


@pytest.mark.parametrize("network", sorted(NETWORKS))
def test_each_network_has_its_own_institution_year(po, network):
    market = NETWORKS[network]
    months = rr.institution_scenarios(po, market)
    assert months[0] == PATTERN.format(tag=CENTRAL_TAGS[network]).format(month="Jan")
    assert len(months) == 12
    settlement = rr.institution_block(po, market)["fti_settlement"]
    assert f"_year_{CENTRAL_TAGS[network]}_fti_settlement_" in settlement
    assert rr.institution_market_twin(po, market) == (
        "public_ownership@fully_nationalised_market_value_central_dispatch",
        "public_ownership@fully_nationalised_market_value")


def test_purchase_price_is_held_at_the_central_network(po, forecast_cases):
    central = NETWORKS["no_acceleration"]
    assert rr.purchase_price_group(po, central, forecast_cases) is None
    assert rr.purchase_price_group(po, f"{central}_cp32", forecast_cases) is None
    for network in ("cp30", "no_upgrades"):
        names = _groups_of(network)
        assert rr.purchase_price_group(po, names["market"], forecast_cases) == central
        assert rr.purchase_price_group(po, names["market_cp32"], forecast_cases) == f"{central}_cp32"
        # A bid stack's tariff is merged against its market group's, whose purchase is the held one.
        assert rr.purchase_price_group(po, names["p3"], forecast_cases) is None


def test_the_auction_calibration_is_derived_from_the_central_network(po):
    calibration = po["cost_calibration"]
    assert calibration["derived_from_group"] == NETWORKS["no_acceleration"]
    assert calibration["table"] == "data/public_ownership/financing/private_cost_calibration.csv"


def test_the_reports_read_the_central_network(po):
    assert po["report_group"] == NETWORKS["no_acceleration"]


def test_the_forecast_cases_point_the_battery_top_up_at_their_networks_account(forecast_cases):
    for network in ("no_acceleration", "no_upgrades"):
        for suffix in ("", "_cp32", "_public_bids_p2", "_public_bids_p3", "_public_bids_p2_cp32",
                       "_public_bids_p3_cp32"):
            case = forecast_cases[f"CP30_2030_{network}{suffix}"]
            assert case["register_account_group"] == NETWORKS[network]
    assert "register_account_group" not in forecast_cases["CP30_2030_cp30_network"]


# ── validation: the one-group-per-role rules hold per network case ─────────────────────────────

@pytest.fixture(scope="module")
def specs(po):
    return rr.configured_case_specs(po)


def test_two_p2_stacks_on_one_network_are_refused(po, forecast_cases, specs):
    config = copy.deepcopy(po)
    config["reference_run_groups"]["cp30_2030_no_upgrades_public_bids_p2"]["network_case"] = "no_acceleration"
    with pytest.raises(ValueError, match="exactly one bill group on network case 'no_acceleration'"):
        rr._validate_balancing_groups(config, specs, forecast_cases)


def test_the_configured_stacks_pass(po, forecast_cases, specs):
    rr._validate_balancing_groups(po, specs, forecast_cases)
    rr._validate_purchase_price_sources(po, forecast_cases)


def test_a_purchase_source_must_price_every_held_case(po, forecast_cases):
    config = copy.deepcopy(po)
    groups = config["reference_run_groups"]
    groups["cp30_2030_no_upgrades"]["purchase_price_from"] = "cp30_2030_cp30_network"      # a bound, itself holding
    with pytest.raises(ValueError, match="prices its own purchase"):
        rr._validate_purchase_price_sources(config, forecast_cases)
    config = copy.deepcopy(po)
    config["reference_run_groups"]["cp30_2030_cp30_network_public_bids_p3"]["purchase_price_from"] = "cp30_2030_no_acceleration"
    with pytest.raises(ValueError, match="only a contract-value market group"):
        rr._validate_purchase_price_sources(config, forecast_cases)


def test_without_network_cases_the_market_group_lookup_is_unchanged():
    """A config that declares no network case resolves as before (one market group per year and level)."""
    config = {"reference_run_groups": {
        "m": {"model_year": 2030, "mode": "twelve_months", "scenario_pattern": "X_{month}_a"},
        "p2": {"model_year": 2030, "mode": "twelve_months", "scenario_pattern": "X_{month}_b",
               "forecast_case": "B", "balancing_bids": "state_at_cost"}},
        "consumer_bill": {"forecast_case": "A"}}
    cases = {"A": {"modelled_year": 2030, "source_scenario_pattern": "X_{month}_a"},
             "B": {"modelled_year": 2030, "source_scenario_pattern": "X_{month}_b"}}
    assert rr.contract_value_group_for(config, "p2", cases) == "m"
    config["reference_run_groups"]["m2"] = dict(config["reference_run_groups"]["m"])
    with pytest.raises(ValueError, match="exactly one contract-value bill group"):
        rr.contract_value_group_for(config, "p2", {**cases})


# ── the held purchase price (calculate_financing_scenarios.hold_acquisition_prices) ──────────────

def _plan():
    from scripts.public_ownership.legacy_valuation import GROUP_EXCLUDED, GROUP_VALUED

    merged = pd.DataFrame({"asset_id": ["a", "b", "c", "n"],
                           "finance_vintage": ["legacy", "legacy", "legacy", "new_build"]})
    plan = pd.DataFrame({"legacy_scope_group": [GROUP_VALUED, "thermal", GROUP_EXCLUDED, None],
                         "acquirable": [True, True, False, True],
                         "acquisition_price_gbp": [100.0, 50.0, np.nan, np.nan],
                         "cancelled_contract_residual_pv_gbp": [0.0, 0.0, 0.0, 0.0]})
    summary = {"annuity_factor": 0.05, "valued_purchase_price_gbp": 100.0,
               "thermal_and_storage_purchase_price_gbp": 50.0, "annual_acquisition_charge_gbp": 7.5,
               "cancelled_contract_residual_pv_gbp": 0.0, "cancelled_contract_residual_annual_gbp": 0.0}
    return merged, plan, summary


def test_held_prices_replace_each_legacy_rows_price_and_the_totals():
    from scripts.public_ownership.calculate_financing_scenarios import hold_acquisition_prices

    merged, plan, summary = _plan()
    held = pd.DataFrame({"case_id": ["p3"] * 3 + ["other"], "asset_id": ["a", "b", "n", "a"],
                         "finance_vintage": ["legacy", "legacy", "new_build", "legacy"],
                         "acquisition_price_gbp": [120.0, 40.0, np.nan, 999.0],
                         "annual_cancelled_contract_residual_gbp": [1.0, 0.0, 0.0, 0.0]})
    out, held_summary = hold_acquisition_prices(plan, summary, merged, held, "p3", "central.parquet")
    assert out["acquisition_price_gbp"].tolist()[:2] == [120.0, 40.0]
    assert np.isnan(out["acquisition_price_gbp"].iloc[2]) and np.isnan(out["acquisition_price_gbp"].iloc[3])
    assert held_summary["valued_purchase_price_gbp"] == 120.0
    assert held_summary["thermal_and_storage_purchase_price_gbp"] == 40.0
    assert held_summary["annual_acquisition_charge_gbp"] == pytest.approx(160.0 * 0.05)
    assert out["cancelled_contract_residual_pv_gbp"].iloc[0] == pytest.approx(1.0 / 0.05)
    record = held_summary["purchase_price_held_from"]
    assert record["own_valued_purchase_price_gbp"] == 100.0 and record["register"] == "central.parquet"
    assert summary["valued_purchase_price_gbp"] == 100.0          # the input summary is not touched


def test_held_prices_refuse_a_different_fleet():
    from scripts.public_ownership.calculate_financing_scenarios import hold_acquisition_prices

    merged, plan, summary = _plan()
    held = pd.DataFrame({"case_id": ["p3", "p3"], "asset_id": ["a", "z"], "finance_vintage": ["legacy", "legacy"],
                         "acquisition_price_gbp": [1.0, 2.0], "annual_cancelled_contract_residual_gbp": [0.0, 0.0]})
    with pytest.raises(ValueError, match="different legacy fleet"):
        hold_acquisition_prices(plan, summary, merged, held, "p3")
    with pytest.raises(ValueError, match="no legacy rows"):
        hold_acquisition_prices(plan, summary, merged, held, "missing")


def _shared_plan():
    """One plan serving a valued-only case and its acquire_all twin (the 20 Oct 2026 cp30_2030_cp30_network failure).

    Rows: v valued, t thermal, s storage, o an ineligible OCGT the plan prices but the register never
    costs, x excluded, n new build.
    """
    from scripts.public_ownership.legacy_valuation import GROUP_EXCLUDED, GROUP_VALUED

    merged = pd.DataFrame({"asset_id": ["v", "t", "s", "o", "x", "n"],
                           "finance_vintage": ["legacy"] * 5 + ["new_build"],
                           "finance_eligible": [True, True, True, False, True, True],
                           "evidence_status": ["resolved", "resolved", "resolved", "excluded", "resolved", "resolved"]})
    plan = pd.DataFrame({"legacy_scope_group": [GROUP_VALUED, "thermal", "storage", "thermal", GROUP_EXCLUDED, None],
                         "acquirable": [True, True, True, True, False, True],
                         "acquisition_price_gbp": [100.0, 50.0, 20.0, 5.0, np.nan, np.nan],
                         "cancelled_contract_residual_pv_gbp": [0.0] * 6})
    summary = {"annuity_factor": 0.05, "valued_purchase_price_gbp": 100.0,
               "thermal_and_storage_purchase_price_gbp": 75.0, "annual_acquisition_charge_gbp": 8.75,
               "cancelled_contract_residual_pv_gbp": 0.0, "cancelled_contract_residual_annual_gbp": 0.0}
    # The register writes a price only on the eligible rows each case acquires.
    held = pd.DataFrame({"case_id": ["valued_only", "all", "all", "all"], "asset_id": ["v", "v", "t", "s"],
                         "finance_vintage": ["legacy"] * 4,
                         "acquisition_price_gbp": [120.0, 120.0, 40.0, 10.0],
                         "annual_cancelled_contract_residual_gbp": [1.0, 1.0, 0.0, 0.0]})
    return merged, plan, summary, held


def test_held_prices_take_the_union_of_a_valued_only_case_and_its_acquire_all_twin():
    from scripts.public_ownership.calculate_financing_scenarios import hold_acquisition_prices

    merged, plan, summary, held = _shared_plan()
    # The valued-only case alone, read as if it bought everything, is the refusal seen on 20 Oct 2026.
    with pytest.raises(ValueError, match="different legacy fleet"):
        hold_acquisition_prices(plan, summary, merged, held, "valued_only")
    out, held_summary = hold_acquisition_prices(plan, summary, merged, held, ["valued_only", "all"],
                                                "central.parquet", acquires_all=True)
    price = out["acquisition_price_gbp"].tolist()
    assert price[:3] == [120.0, 40.0, 10.0]                         # held from the union
    assert price[3] == 5.0                                          # the ineligible OCGT keeps the plan's own
    assert np.isnan(price[4]) and np.isnan(price[5])
    assert out["cancelled_contract_residual_pv_gbp"].iloc[0] == pytest.approx(1.0 / 0.05)
    assert held_summary["valued_purchase_price_gbp"] == 120.0
    assert held_summary["thermal_and_storage_purchase_price_gbp"] == 55.0
    record = held_summary["purchase_price_held_from"]
    assert record["case_ids"] == ["valued_only", "all"] and record["held_rows"] == 3
    assert record["own_price_rows_outside_scope"] == 1


def test_held_prices_on_a_valued_only_plan_check_the_valued_group_only():
    from scripts.public_ownership.calculate_financing_scenarios import hold_acquisition_prices

    merged, plan, summary, held = _shared_plan()
    out, _ = hold_acquisition_prices(plan, summary, merged, held.loc[held["case_id"].eq("valued_only")],
                                     ["valued_only"], acquires_all=False)
    assert out["acquisition_price_gbp"].tolist()[:4] == [120.0, 50.0, 20.0, 5.0]   # no case buys t, s or o


def test_held_prices_refuse_cases_that_disagree():
    from scripts.public_ownership.calculate_financing_scenarios import hold_acquisition_prices

    merged, plan, summary, held = _shared_plan()
    held.loc[held["case_id"].eq("all") & held["asset_id"].eq("v"), "acquisition_price_gbp"] = 121.0
    with pytest.raises(ValueError, match="disagree on acquisition_price_gbp"):
        hold_acquisition_prices(plan, summary, merged, held, ["valued_only", "all"], acquires_all=True)
    # An eligible row the acquire_all twin should price but does not is still refused.
    _, _, _, held = _shared_plan()
    with pytest.raises(ValueError, match=r"only here: \['s'\]"):
        hold_acquisition_prices(plan, summary, merged, held.loc[~held["asset_id"].eq("s")],
                                ["valued_only", "all"], acquires_all=True)


# ── TNUoS per network (project_future_bill.apply_tnuos_network_delivery) ─────────────────────────

ASSUMPTIONS = ROOT / "data" / "market" / "consumer_bill_benchmarks" / "2030_pypsa_informed_assumptions_cfd_rule_none.json"
DELIVERY = "data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json"


@pytest.fixture(scope="module")
def assumptions():
    return json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))


def _blend(block):
    from scripts.consumer_bill.project_future_bill import blended_household_value

    return blended_household_value("tnuos", block, 2030, 2025)[0]


def test_tnuos_without_a_network_case_is_the_published_line(assumptions):
    from scripts.consumer_bill.project_future_bill import apply_tnuos_network_delivery

    assert apply_tnuos_network_delivery(assumptions, {"modelled_year": 2030}) is assumptions


@pytest.mark.parametrize("network", ["cp30", "no_acceleration"])
def test_tnuos_is_unchanged_on_the_central_and_lower_networks(assumptions, network):
    """NESO's source is the TOs' January 2026 revenue, on the projects' own 2031 dates (the expected network)."""
    from scripts.consumer_bill.project_future_bill import apply_tnuos_network_delivery

    out = apply_tnuos_network_delivery(assumptions, {"tnuos_network_delivery": {"file": DELIVERY, "case": network}})
    before, after = assumptions["sourced_externals"]["tnuos"], out["sourced_externals"]["tnuos"]
    assert _blend(after) == pytest.approx(_blend(before), abs=1e-12)
    assert after["value"] == before["value"]
    assert after["network_delivery_adjustment"]["change_share"] == 0.0
    assert "network_delivery_adjustment" not in before          # the input is not touched


def test_tnuos_on_the_upper_network_takes_the_labelled_ten_percent(assumptions):
    from scripts.consumer_bill.project_future_bill import apply_tnuos_network_delivery

    document = json.loads((ROOT / DELIVERY).read_text(encoding="utf-8"))
    assert document["cases"]["no_upgrades"]["status"] == "labelled_assumption"
    assert "LABELLED ASSUMPTION" in document["cases"]["no_upgrades"]["assumption"]
    out = apply_tnuos_network_delivery(assumptions, {"tnuos_network_delivery": {"file": DELIVERY, "case": "no_upgrades"}})
    derivation = assumptions["sourced_externals"]["tnuos"]["derivation"]
    income = document["to_income_gbp_m"]
    # FY2030/31: 10% of NESO's onshore TO income (Table 23) on the demand residual, at the domestic band's share.
    expected_3031 = -0.10 * income["FY2030/31"] / derivation["demand_residual_revenue_gbp_m_2030_31"] * (
        derivation["domestic_residual_gbp_per_site_day_2030_31"] * 365)
    by_period = out["sourced_externals"]["tnuos"]["network_delivery_adjustment"]["by_period"]
    assert by_period["FY2030/31"]["household_change_gbp_nominal"] == pytest.approx(expected_3031, rel=1e-12)
    assert out["sourced_externals"]["tnuos"]["value"] == pytest.approx(
        assumptions["sourced_externals"]["tnuos"]["value"] + expected_3031)
    change = _blend(out["sourced_externals"]["tnuos"]) - _blend(assumptions["sourced_externals"]["tnuos"])
    # About GBP 12.5 a household in 2025 prices (the calendar-2030 blend of FY2029/30 and FY2030/31).
    assert -14.0 < change < -11.0


def test_an_unknown_network_case_is_refused(assumptions):
    from scripts.consumer_bill.project_future_bill import apply_tnuos_network_delivery

    with pytest.raises(ValueError, match="no network-delivery case"):
        apply_tnuos_network_delivery(assumptions, {"tnuos_network_delivery": {"file": DELIVERY, "case": "seven_late"}})
