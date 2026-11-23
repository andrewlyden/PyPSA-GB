"""The BM may hold the must-run energy budget Stage 1 carried (OW2, user decision 3 Oct 2026).

``market.balancing.min_output_energy_budget``: ``drop`` (default) zeroes the hourly
floor of the budget carriers in the BM and leaves their energy to the Stage 1
anchor; ``hold`` keeps each unit's energy per block, timing free. The CP30
central-dispatch months hold (the at-cost BM unwound 8.8 TWh of biomass for gas
under drop); the market months keep drop.
"""

import inspect
import logging
import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import (
    build_min_output_energy_callback,
    min_output_energy_delivered,
    prepare_bm_min_output_energy_budget,
    resolve_min_output_energy_budget_mode,
)

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_bm_min_output_energy_hold")
CONFIG = {"market": {}, "thermal": {"availability_factors": {"min_output_energy_window_hours": {"biomass": 24}}}}


def _network(hours=48):
    """Biomass (floor 0.35, cap 0.80) at 128.6 GBP/MWh against cheaper gas at 107: at cost the
    LP would run no biomass at all, the BM failure the budget guards against."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    n.add("Generator", "drax", bus="el", p_nom=100.0, p_min_pu=0.35, p_max_pu=0.80,
          marginal_cost=128.6, carrier="biomass")
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=107.0, carrier="CCGT")
    peak = [150.0 if 17 <= i % 24 <= 20 else 100.0 for i in range(hours)]
    n.add("Load", "load", bus="el", p_set=pd.Series(peak, index=n.snapshots))
    return n


def _solve_windows(n, floor):
    """Daily windows as the rolling BM solves them, the budget callback in each."""
    callback = build_min_output_energy_callback(floor)
    out, budget_total, delivered_total = [], 0.0, 0.0
    for day in range(len(n.snapshots) // 24):
        w = n.copy()
        w.set_snapshots(n.snapshots[day * 24:(day + 1) * 24])
        status, condition = w.optimize(solver_name="highs", extra_functionality=callback)
        assert status == "ok", condition
        out.append(w.generators_t.p["drax"].copy())
        budget, delivered = min_output_energy_delivered(w, callback)
        budget_total += budget
        delivered_total += delivered
    return pd.concat(out), budget_total, delivered_total


@pytest.mark.unit
def test_default_is_drop_and_bad_values_are_refused():
    assert resolve_min_output_energy_budget_mode({}) == "drop"
    assert resolve_min_output_energy_budget_mode(None) == "drop"
    assert resolve_min_output_energy_budget_mode({"min_output_energy_budget": "HOLD"}) == "hold"
    with pytest.raises(ValueError, match="min_output_energy_budget"):
        resolve_min_output_energy_budget_mode({"min_output_energy_budget": "keep"})


@pytest.mark.unit
def test_drop_zeroes_the_floor_and_lets_the_bm_unwind_the_energy():
    n = _network()
    assert prepare_bm_min_output_energy_budget(n, CONFIG, {}, LOG) is None
    assert float(n.generators.at["drax", "p_min_pu"]) == 0.0
    p, _, _ = _solve_windows(n, None)
    assert float(p.sum()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_hold_keeps_the_daily_energy_with_free_timing():
    n = _network()
    floor = prepare_bm_min_output_energy_budget(
        n, CONFIG, {"min_output_energy_budget": "hold", "mode": "rolling", "window_hours": 24}, LOG)
    assert list(floor.columns) == ["drax"]
    assert float(n.generators.at["drax", "p_min_pu"]) == 0.0      # hourly floor still dropped
    p, budget, delivered = _solve_windows(n, floor)
    for day in range(2):
        assert float(p.iloc[day * 24:(day + 1) * 24].sum()) == pytest.approx(0.35 * 100.0 * 24, rel=1e-6)
    assert budget == pytest.approx(2 * 840.0)
    assert delivered == pytest.approx(budget, rel=1e-6)
    # Timing free: some hours below the old 35 MW floor, some above.
    assert float(p.min()) < 35.0 - 1e-6 < 35.0 + 1e-6 < float(p.max())


@pytest.mark.unit
def test_hold_warns_when_bm_windows_are_shorter_than_the_blocks(caplog):
    n = _network()
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        prepare_bm_min_output_energy_budget(
            n, CONFIG, {"min_output_energy_budget": "hold", "mode": "rolling", "window_hours": 1}, LOG)
    assert "shorter than the budget blocks" in caplog.text


@pytest.mark.unit
def test_solve_balancing_passes_the_budget_to_both_bm_modes():
    import scripts.market.solve_balancing as sb

    assert "energy_floor" in inspect.signature(sb.solve_rolling_balancing).parameters
    source = inspect.getsource(sb)
    assert "prepare_bm_min_output_energy_budget(" in source
    assert "energy_floor=bm_energy_floor" in source
    assert "build_min_output_energy_callback(bm_energy_floor)" in source


@pytest.mark.unit
def test_central_dispatch_months_hold_and_market_months_drop():
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenarios = load_config(include_archive=True)["scenarios"]   # the gate-week probe is archived
    for month in ("Jan", "Jul", "Dec"):
        central = scenarios[f"CP30_FFR_2030_{month}_cp30_network_central_dispatch"]["market"]["balancing"]
        assert resolve_min_output_energy_budget_mode(central) == "hold"
        excluded = central["participation"]["generators"]["exclude_carriers"]
        assert "biomass" not in excluded
        assert {"CHP", "waste_to_energy", "biogas", "load_shedding"} <= set(excluded)
        assert central["mode"] == "rolling" and float(central["window_hours"]) == 24.0
        market = scenarios[f"CP30_FFR_2030_{month}_cp30_network"]
        assert resolve_min_output_energy_budget_mode(market["market"]["balancing"]) == "drop"
    gate = scenarios["CP30_FFR_2030_ETYS_market_1week_central_dispatch_hold"]
    assert resolve_min_output_energy_budget_mode(gate["market"]["balancing"]) == "hold"
    assert gate["network_from"] == "CP30_FFR_2030"
