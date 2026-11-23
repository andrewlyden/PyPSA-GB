"""Item BM1, step 3 of the P2 bill chain: a forecast case on the P2 months.

The P2 months change only how the state-owned units bid in the balancing
mechanism; they read their parent month's Stage 1 (``wholesale_from``). The
forecast case built on them must differ from ``CP30_2030_cp30_network`` in nothing but the
months it reads, so the bill it gives differs only by what the balancing stage
did. It reads the same recovery denominators, which are a property of the demand
year, not of the balancing bids, and the file says so by naming its own case.
"""

from pathlib import Path

import pytest
import yaml

from scripts.consumer_bill.project_future_bill import read_recovery_denominators, recovery_case_id
from config.config_loader import load_scenario_entries
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PATH = ROOT / "data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PARENT = "CP30_2030_cp30_network"
P2 = "CP30_2030_cp30_network_public_bids_p2"
# The keys the P2 case may set differently from its parent, and nothing else.
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


def test_a_case_reads_its_own_recovery_file_unless_it_names_another():
    assert recovery_case_id({"case_id": "A"}) == "A"
    assert recovery_case_id({"case_id": "A", "recovery_case_id": "B"}) == "B"


def test_the_p2_case_reads_the_parent_recovery_file(cases):
    document = yaml.safe_load(RECOVERY_PATH.read_text(encoding="utf-8"))
    case = {**cases[P2], "case_id": P2}
    recovery = read_recovery_denominators(RECOVERY_PATH, recovery_case_id(case),
                                          document["system_demand_mwh"]["value"])
    assert recovery["final_demand_mwh"] > 0
    with pytest.raises(ValueError, match="declares case"):
        read_recovery_denominators(RECOVERY_PATH, P2, document["system_demand_mwh"]["value"])


def test_the_p2_case_differs_from_its_parent_only_in_the_months_it_reads(cases):
    parent, p2 = cases[PARENT], cases[P2]
    assert p2["recovery_case_id"] == PARENT
    for key in (set(parent) | set(p2)) - DIFFERING:
        assert p2.get(key) == parent.get(key), key
    assert p2["source_scenario_pattern"] == "CP30_FFR_2030_{month}_cp30_network_p2_bids"


def test_the_p2_aggregate_lists_the_twelve_p2_months(cases, aggregates):
    study = aggregates[cases[P2]["annual_aggregate_study"]]
    assert study["scenarios"] == [cases[P2]["source_scenario_pattern"].format(month=m) for m in MONTHS]


def test_every_p2_month_takes_its_stage_1_from_the_parent_case_month(cases, scenarios):
    for month in MONTHS:
        scenario = scenarios[cases[P2]["source_scenario_pattern"].format(month=month)]
        parent = scenarios[cases[PARENT]["source_scenario_pattern"].format(month=month)]
        # Both read the parent month's own Stage 1 (s1_zf since 13 Oct 2026).
        assert scenario["wholesale_from"] == parent.get("wholesale_from", cases[PARENT]["source_scenario_pattern"].format(month=month))
