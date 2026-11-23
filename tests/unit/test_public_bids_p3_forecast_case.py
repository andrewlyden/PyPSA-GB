"""Item BM1, P3: a forecast case on the P3 months, and the tie steps kept out of its money.

The P3 months change only the balancing stage (every CfD unit state-owned,
bidding at cost with the decision-20 LIFO tie-break); they read their parent
month's Stage 1. The forecast case built on them differs from ``CP30_2030_cp30_network`` in
nothing but the months it reads, and reads the parent's recovery denominators.

The tie steps order a tie the state is indifferent to; they sit inside the LP's
accepted bid cost, and the balancing-cost extraction nets them out of the
constraint line. The bill's own identity check must net them too, or every P3
bill would be refused, and no other bill may change.
"""

import json
from pathlib import Path

import pytest
import yaml

from scripts.consumer_bill.project_future_bill import (
    read_balancing_costs, read_recovery_denominators, recovery_case_id,
)
from config.config_loader import load_scenario_entries
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PATH = ROOT / "data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PARENT = "CP30_2030_cp30_network"
P3 = "CP30_2030_cp30_network_public_bids_p3"
DIFFERING = {"source_scenario_pattern", "annual_aggregate_study", "recovery_case_id"}


@pytest.fixture(scope="module")
def cases():
    defaults = load_defaults()
    return defaults["consumer_bill"]["forecasts"]["cases"]


@pytest.fixture(scope="module")
def aggregates():
    from config.config_loader import study_settings

    return study_settings(ROOT / "config")["monthly_aggregates"]


@pytest.fixture(scope="module")
def scenarios():
    return load_scenario_entries(ROOT / "config")


@pytest.mark.skipif(not RECOVERY_PATH.exists(), reason="recovery denominators not present")
def test_the_p3_case_reads_the_parent_recovery_file(cases):
    document = yaml.safe_load(RECOVERY_PATH.read_text(encoding="utf-8"))
    case = {**cases[P3], "case_id": P3}
    recovery = read_recovery_denominators(RECOVERY_PATH, recovery_case_id(case),
                                          document["system_demand_mwh"]["value"])
    assert recovery["final_demand_mwh"] > 0


def test_the_p3_case_differs_from_its_parent_only_in_the_months_it_reads(cases):
    parent, p3 = cases[PARENT], cases[P3]
    assert p3["recovery_case_id"] == PARENT
    for key in (set(parent) | set(p3)) - DIFFERING:
        assert p3.get(key) == parent.get(key), key
    assert p3["source_scenario_pattern"] == "CP30_FFR_2030_{month}_cp30_network_p3_bids"


def test_the_p3_aggregate_lists_the_twelve_p3_months(cases, aggregates):
    study = aggregates[cases[P3]["annual_aggregate_study"]]
    assert study["scenarios"] == [cases[P3]["source_scenario_pattern"].format(month=m) for m in MONTHS]


def test_every_p3_month_takes_its_stage_1_from_the_parent_case_month(cases, scenarios):
    for month in MONTHS:
        scenario = scenarios[cases[P3]["source_scenario_pattern"].format(month=month)]
        parent = scenarios[cases[PARENT]["source_scenario_pattern"].format(month=month)]
        # Both read the parent month's own Stage 1 (s1_zf since 13 Oct 2026).
        assert scenario["wholesale_from"] == parent.get("wholesale_from", cases[PARENT]["source_scenario_pattern"].format(month=month))


def _annual(**overrides):
    annual = {"model_price_year": 2024, "accepted_offer_cost_gbp": 60.0, "accepted_bid_cost_gbp": 50.0,
              "cfd_curtailment_compensation_gbp": 0.0, "cfd_curtailed_mwh": 0.0,
              "feasibility_slack_cost_gbp": 0.0, "feasibility_slack_energy_mwh": 0.0,
              "feasibility_slack_cost_in_accepted_costs_gbp": 0.0,
              "non_participant_turn_down_cost_at_penalty_gbp": 0.0,
              "constraint_cost_excludes_slack_and_penalty": True,
              "cfd_curtailment_compensation_treatment": "reconcile_against_lp_bid"}
    annual.update(overrides)
    return annual


def test_the_bill_nets_the_tie_steps_the_extraction_netted(tmp_path):
    path = tmp_path / "balancing.json"
    # Accepted bid cost 50 carries 7 of tie steps; the extraction's line is 110 - 7.
    path.write_text(json.dumps({"annual": _annual(tie_break_step_cost_gbp=7.0, constraint_cost_gbp=103.0)}))
    balancing = read_balancing_costs(path, [], 2024)
    assert balancing["constraint_cost_gbp"] == 103.0
    assert balancing["tie_break_step_cost_gbp"] == 7.0
    # Counting the steps as money is refused.
    path.write_text(json.dumps({"annual": _annual(tie_break_step_cost_gbp=7.0, constraint_cost_gbp=110.0)}))
    with pytest.raises(ValueError, match="tie-break steps"):
        read_balancing_costs(path, [], 2024)


def test_a_bill_without_tie_steps_is_unchanged(tmp_path):
    path = tmp_path / "balancing.json"
    for annual in (_annual(constraint_cost_gbp=110.0), _annual(constraint_cost_gbp=110.0, tie_break_step_cost_gbp=0.0)):
        path.write_text(json.dumps({"annual": annual}))
        balancing = read_balancing_costs(path, [], 2024)
        assert balancing["constraint_cost_gbp"] == 110.0
        # No new key on any bill but P3's, so every other summary is byte-identical.
        assert "tie_break_step_cost_gbp" not in balancing


def test_the_reconciliation_can_set_penalty_priced_units_aside(tmp_path):
    import pandas as pd

    from scripts.consumer_bill.extract_future_balancing_costs import reconciliation_net_of_penalty

    # P3 Jan 2030 in miniature: state units bid their tie step; one unit left out of
    # the balancing mechanism is turned down at the 6,000 GBP/MWh penalty.
    redispatch = tmp_path / "redispatch.csv"
    pd.DataFrame({"type": ["generator"] * 3, "carrier": ["wind_offshore", "wind_onshore", "wind_onshore"],
                  "decrease_MWh": [1000.0, 500.0, 0.005], "bid_cost": [100.0, 25.0, 30.0]},
                 index=["off", "on", "embedded"]).to_csv(redispatch, index_label="component")
    reconciled = pd.DataFrame({"generator": ["off", "on", "embedded"], "compensation_gbp": [100.0, 25.0, 0.0005]})
    compensation = float(reconciled.compensation_gbp.sum())
    lp = 155.0
    assert abs((compensation - lp) / compensation) > 0.01          # the gross check fails
    net = reconciliation_net_of_penalty(redispatch, reconciled, compensation, lp)
    assert net["penalty_priced_units"] == 1 and net["lp_penalty_bid_cost_gbp"] == 30.0
    assert abs(net["gap_share"]) < 1e-9                             # and passes on the participants
    # Nothing penalty-priced: nothing to set aside.
    pd.DataFrame({"type": ["generator"], "carrier": ["wind_offshore"], "decrease_MWh": [1000.0],
                  "bid_cost": [100.0]}, index=["off"]).to_csv(redispatch, index_label="component")
    assert reconciliation_net_of_penalty(redispatch, reconciled.iloc[:1], 100.0, 100.0) is None
