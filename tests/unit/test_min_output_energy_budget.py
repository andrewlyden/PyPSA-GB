"""Minimum output as an energy budget (user decision 27 Sep 2026).

``thermal.availability_factors.min_output_energy_window_hours`` ({carrier: hours})
turns the hourly floor of the listed carriers into energy per block of a Stage 1
optimisation, timing free up to p_max_pu; the BM drops the hourly floor. CP30
uses it for biomass so the 0.35 floor stops running through surplus hours while
the daily energy (and CP30's 12.2 TWh) is kept.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest
import yaml

from scripts.market.market_utils import (
    _energy_blocks,
    build_min_output_energy_callback,
    drop_min_output_floor_for_energy_budget,
    min_output_energy_delivered,
    prepare_min_output_energy_budget,
    resolve_min_output_energy_windows,
)
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_min_output_energy_budget")
KEY = "min_output_energy_window_hours"
SURPLUS_HOURS = 12        # first half of each day: 200 MW of free wind against 100 MW of load


def _network(hours=48):
    """Windy first half of each day, 150 GBP gas otherwise; biomass floor 0.35, cap 0.53, 128.6 GBP."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-12-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    wind = [1.0 if i % 24 < SURPLUS_HOURS else 0.0 for i in range(hours)]
    n.add("Generator", "wind", bus="el", p_nom=200.0, p_max_pu=pd.Series(wind, index=n.snapshots),
          marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=150.0, carrier="CCGT")
    n.add("Generator", "drax", bus="el", p_nom=100.0, p_min_pu=0.35, p_max_pu=0.53,
          marginal_cost=128.6, carrier="biomass")
    n.add("Generator", "efw", bus="el", p_nom=10.0, p_min_pu=0.55, p_max_pu=0.62,
          marginal_cost=20.0, carrier="waste_to_energy")
    n.add("Load", "load", bus="el", p_set=100.0)
    return n


def _config(windows=None):
    block = {} if windows is None else {KEY: windows}
    return {"market": {}, "thermal": {"availability_factors": block}}


def _solve_rolling(n, config):
    from scripts.market.solve_wholesale import solve_rolling_day_ahead

    floor = prepare_min_output_energy_budget(n, config, LOG)
    gen, _, _, _, objective = solve_rolling_day_ahead(
        n, {"window_hours": 24}, "highs", {}, config, LOG, energy_floor=floor,
    )
    return gen, objective


def _surplus(gen):
    return gen[[i % 24 < SURPLUS_HOURS for i in range(len(gen))]]


# ── Config ──────────────────────────────────────────────────────────────────

def test_default_is_off_and_matches_defaults_yaml():
    defaults = load_defaults()
    assert defaults["thermal"]["availability_factors"][KEY] == {}
    assert resolve_min_output_energy_windows({}) == {}
    n = _network()
    floor = prepare_min_output_energy_budget(n, _config(), LOG)
    assert floor.shape[1] == 0
    assert build_min_output_energy_callback(floor) is None
    assert float(n.generators.at["drax", "p_min_pu"]) == pytest.approx(0.35)
    with pytest.raises(ValueError):
        resolve_min_output_energy_windows(_config({"biomass": 0}))


def test_cp30_scenarios_budget_biomass_daily():
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenario = load_config()["scenarios"]["CP30_FFR_2030_Dec_cp30_network"]
    # Must-run package (28 Sep 2026): EfW, CHP and biogas joined biomass.
    assert resolve_min_output_energy_windows(scenario) == {
        "biomass": 24.0, "waste_to_energy": 24.0, "CHP": 24.0, "biogas": 24.0}
    assert scenario["thermal"]["availability_factors"]["min_output"]["biomass"] == pytest.approx(0.35)


# ── Preparation and the BM side ─────────────────────────────────────────────

def test_prepare_zeroes_only_the_listed_carriers_and_keeps_the_floor_energy():
    n = _network()
    floor = prepare_min_output_energy_budget(n, _config({"biomass": 24}), LOG)
    assert list(floor.columns) == ["drax"]
    assert float(floor["drax"].sum()) == pytest.approx(0.35 * 100.0 * 48)
    assert float(n.generators.at["drax", "p_min_pu"]) == 0.0
    assert float(n.generators.at["efw", "p_min_pu"]) == pytest.approx(0.55)


def test_committable_units_keep_their_floor():
    n = _network()
    n.generators.loc["drax", "committable"] = True
    floor = prepare_min_output_energy_budget(n, _config({"biomass": 24}), LOG)
    assert floor.shape[1] == 0
    assert float(n.generators.at["drax", "p_min_pu"]) == pytest.approx(0.35)


def test_bm_drops_the_hourly_floor_for_the_listed_carriers_only():
    n = _network()
    n.generators_t.p_min_pu["drax"] = 0.35
    names = drop_min_output_floor_for_energy_budget(n, _config({"biomass": 24}), LOG)
    assert list(names) == ["drax"]
    assert float(n.generators.at["drax", "p_min_pu"]) == 0.0
    assert "drax" not in n.generators_t.p_min_pu.columns
    assert float(n.generators.at["efw", "p_min_pu"]) == pytest.approx(0.55)
    assert len(drop_min_output_floor_for_energy_budget(_network(), _config(), LOG)) == 0


def test_blocks_cut_from_the_start_with_a_shorter_last_block():
    sns = pd.date_range("2030-12-01", periods=30, freq="h")
    blocks = _energy_blocks(sns, 24.0, pd.Series(1.0, index=sns))
    assert [len(b) for b in blocks] == [24, 6]


# ── The rolling Stage 1 ─────────────────────────────────────────────────────

def test_budget_moves_biomass_out_of_surplus_hours_and_keeps_the_daily_energy():
    n = _network()
    hourly, _ = _solve_rolling(n, _config())
    n = _network()
    budget, _ = _solve_rolling(n, _config({"biomass": 24}))

    # Hourly floor: 35 MW in every surplus hour.
    assert float(_surplus(hourly)["drax"].min()) == pytest.approx(35.0, abs=1e-3)
    # Budget: the same 840 MWh a day, with the rest of the day at the 53 MW cap first.
    for day in (slice(0, 24), slice(24, 48)):
        assert float(budget["drax"].iloc[day].sum()) == pytest.approx(0.35 * 100.0 * 24, rel=1e-5)
    non_surplus = budget[[i % 24 >= SURPLUS_HOURS for i in range(len(budget))]]
    assert float(non_surplus["drax"].min()) == pytest.approx(53.0, abs=1e-3)
    assert float(_surplus(budget)["drax"].sum()) < float(_surplus(hourly)["drax"].sum()) - 100.0
    # Energy the surplus hours no longer take from biomass goes to wind, not to gas.
    assert float(_surplus(budget)["wind"].sum()) > float(_surplus(hourly)["wind"].sum()) + 100.0
    # Waste (not listed) keeps its hourly floor.
    assert float(budget["efw"].min()) == pytest.approx(5.5, abs=1e-3)


def test_delivered_energy_is_reported_per_window():
    n = _network(hours=24)
    config = _config({"biomass": 24})
    floor = prepare_min_output_energy_budget(n, config, LOG)
    callback = build_min_output_energy_callback(floor)
    status, condition = n.optimize(solver_name="highs", extra_functionality=callback)
    assert status == "ok", condition
    budget, delivered = min_output_energy_delivered(n, callback)
    assert budget == pytest.approx(840.0)
    assert delivered >= budget - 1e-4
    assert min_output_energy_delivered(n, None) == (0.0, 0.0)
