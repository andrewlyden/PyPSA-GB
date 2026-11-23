import json
from io import StringIO
from unittest.mock import patch

import pytest

from scripts.public_ownership.aggregate_reference_runs import aggregate_market_summaries


def _summary(scenario, hours, demand, price):
    return json.dumps({
        "scenario": scenario,
        "accounting": {
            "reported_period_hours": hours, "demand_MWh": demand,
            "modelled_redispatch_cost_GBP": demand * 2,
            "ordinary_redispatch_cost_excluding_feasibility_slack_GBP": demand,
            "feasibility_slack_energy_MWh": 0, "feasibility_slack_cost_GBP": 0,
            "gross_redispatch_MWh": demand / 10,
        },
        "prices": {"mean_wholesale_price": price, "mean_balancing_nodal_dual": price + 1},
    })


def _mock_summary_files(payloads):
    def open_file(path, *args, **kwargs):
        return StringIO(payloads[path.name])
    return patch("scripts.public_ownership.aggregate_reference_runs.Path.open", open_file)


def test_complete_monthly_bundle_sums_costs_and_demand_weights_prices():
    paths, scenarios, payloads = [], [], {}
    for month in range(12):
        path = f"month-{month}.json"
        payloads[path] = _summary(f"scenario-{month}", 730, 100 + month, 50 + month)
        paths.append(path)
        scenarios.append(f"scenario-{month}")
    # Use a leap-hour distribution while keeping the test fixture intentionally small.
    payloads[paths[-1]] = _summary(scenarios[-1], 754, 111, 61)
    with _mock_summary_files(payloads), patch(
        "scripts.public_ownership.aggregate_reference_runs._sha256", return_value="test-hash"
    ):
        monthly, annual, manifest = aggregate_market_summaries(paths, expected_scenarios=scenarios, run_group="test")
    assert len(monthly) == 12
    assert annual["reported_period_hours"] == 8784
    assert annual["demand_MWh"] == pytest.approx(sum(100 + i for i in range(12)))
    expected_price = sum((100 + i) * (50 + i) for i in range(12)) / sum(100 + i for i in range(12))
    assert annual["mean_wholesale_price_GBP_per_MWh_demand_weighted"] == pytest.approx(expected_price)
    assert len(manifest["sources"]) == 12


def test_aggregator_rejects_partial_year_and_duplicate_month_scenarios():
    paths, payloads = [], {}
    for month in range(12):
        path = f"month-{month}.json"
        payloads[path] = _summary(f"scenario-{month}", 720, 100, 50)
        paths.append(path)
    with _mock_summary_files(payloads):
        with pytest.raises(ValueError, match="complete calendar year"):
            aggregate_market_summaries(paths, expected_scenarios=[f"scenario-{i}" for i in range(12)])
        with pytest.raises(ValueError, match="unique scenario"):
            aggregate_market_summaries(paths, expected_scenarios=["same"] * 12)
