"""Carbon Price Support ends April 2028 (review item M1, user decision 28 Sep 2026).

``marginal_costs.future_costs.carbon_price_support_end`` drops CPS from a future
cost case whose modelled period starts on or after the date; the UK ETS part is
kept as the case gives it.
"""

import sys
from pathlib import Path

import pytest
import yaml

from scripts.generators.apply_marginal_costs import get_carbon_price, get_future_cost_assumptions
from config.config_loader import load_defaults
from tests.helpers import requires_outputs

ROOT = Path(__file__).resolve().parents[2]


def _scenario(year, end=None, start=None):
    future = {
        "enabled": True,
        "case": "central",
        "cases": {"central": {"fuel_prices": {"gas": 30.0}, "carbon": {"uk_ets": 55.0, "carbon_price_support": 18.0}}},
    }
    if end is not None:
        future["carbon_price_support_end"] = end
    scenario = {"modelled_year": year, "marginal_costs": {"future_costs": future}}
    if start:
        scenario["solve_period"] = {"enabled": True, "start": start, "end": start}
    return scenario


def test_default_keeps_cps_and_matches_defaults_yaml():
    defaults = load_defaults()
    assert defaults["marginal_costs"]["future_costs"]["carbon_price_support_end"] is None
    assert get_future_cost_assumptions(_scenario(2030))["carbon"]["carbon_price_support"] == 18.0
    assert get_carbon_price(_scenario(2030)) == pytest.approx(73.0)


def test_2030_drops_cps_and_keeps_uk_ets():
    case = get_future_cost_assumptions(_scenario(2030, end="2028-04-01"))
    assert case["carbon"]["carbon_price_support"] == 0.0
    assert case["carbon"]["uk_ets"] == 55.0
    assert case["carbon_price_support_ended"] == "2028-04-01"
    assert get_carbon_price(_scenario(2030, end="2028-04-01")) == pytest.approx(55.0)


def test_a_period_starting_before_the_end_keeps_cps():
    assert get_future_cost_assumptions(_scenario(2027, end="2028-04-01"))["carbon"]["carbon_price_support"] == 18.0
    early = _scenario(2028, end="2028-04-01", start="2028-01-01 00:00")
    assert get_future_cost_assumptions(early)["carbon"]["carbon_price_support"] == 18.0
    late = _scenario(2028, end="2028-04-01", start="2028-04-01 00:00")
    assert get_future_cost_assumptions(late)["carbon"]["carbon_price_support"] == 0.0


def test_cp30_months_keep_cps():
    # F13 (user, 7 Oct 2026) reversed the April 2028 end for CP30: CP30's own SRMCs (CP.25) carry the full
    # carbon price including the 18 CPS, so CPS continues to 2030; the HM Treasury end date is a sensitivity.
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenario = load_config()["scenarios"]["CP30_FFR_2030_Jan_cp30_network"]
    assert scenario["marginal_costs"]["future_costs"]["carbon_price_support_end"] is None


@requires_outputs("resources/market/neso_clean_power_2030_annex4_costs.json")
def test_cp30_months_price_cps():
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenario = load_config()["scenarios"]["CP30_FFR_2030_Jan_cp30_network"]
    case = get_future_cost_assumptions(scenario)
    assert case["carbon"]["carbon_price_support"] > 0.0
    assert case["carbon"]["uk_ets"] > 0.0
