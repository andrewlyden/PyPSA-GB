"""Stage 1 at a coarser temporal resolution (market.wholesale.temporal_resolution_hours, 17 Oct 2026).

NESO's CP.28-30 market run is a 3-hour PLEXOS run; Run B of the NESO comparison solves Stage 1 on
3-hour blocks and writes hourly outputs (power held within each block) for the hourly BM.
Default 1 is the hourly solve exactly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.market.solve_wholesale import solve_rolling_day_ahead
from scripts.market.temporal_resolution import (
    DEFAULT_TEMPORAL_RESOLUTION_HOURS,
    META_KEY,
    aggregate_frame,
    aggregate_to_blocks,
    block_starts,
    check_compatible,
    expand_levels,
    expand_rows,
    record_meta,
    resolve_temporal_resolution_hours,
    restore_hourly,
)
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_wholesale_temporal_resolution")


def _network(days=2):
    """Wind varying hour by hour, a 40 GBP unit with a 0.2/h ramp, gas at 150, a 4 h battery, a varying load."""
    hours = 24 * days
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    rng = np.random.default_rng(3)
    wind = pd.Series(rng.uniform(0.0, 1.0, hours), index=n.snapshots)
    load = pd.Series(100.0 + 40.0 * np.sin(np.arange(hours) / 24 * 2 * np.pi), index=n.snapshots)
    n.add("Generator", "wind", bus="el", p_nom=150.0, p_max_pu=wind, marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "mid", bus="el", p_nom=100.0, marginal_cost=40.0, carrier="biomass",
          ramp_limit_up=0.2, ramp_limit_down=0.2)
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=150.0, carrier="CCGT")
    n.add("Load", "load", bus="el", p_set=load)
    n.add("StorageUnit", "battery", bus="el", p_nom=30.0, max_hours=4.0, efficiency_store=0.9,
          efficiency_dispatch=0.9, marginal_cost=1.0, state_of_charge_initial=20.0,
          cyclic_state_of_charge=False, carrier="Battery")
    return n


# ── Config ──────────────────────────────────────────────────────────────────

def test_default_is_hourly_and_matches_global_defaults():
    defaults = load_defaults()
    assert defaults["market"]["wholesale"]["temporal_resolution_hours"] == DEFAULT_TEMPORAL_RESOLUTION_HOURS == 1
    assert resolve_temporal_resolution_hours({}) == 1
    assert resolve_temporal_resolution_hours({"temporal_resolution_hours": None}) == 1
    assert resolve_temporal_resolution_hours({"temporal_resolution_hours": 3}) == 3
    assert resolve_temporal_resolution_hours({"temporal_resolution_hours": 3.0}) == 3
    for bad in (0, 0.5, 2.5, -3):
        with pytest.raises(ValueError):
            resolve_temporal_resolution_hours({"temporal_resolution_hours": bad})


def test_incompatible_settings_are_refused():
    rolling = {"mode": "rolling_day_ahead", "window_hours": 24, "look_ahead_hours": 24, "look_ahead_resolution_hours": 6}
    check_compatible(3, rolling)                       # the s1_3h settings
    check_compatible(1, dict(rolling, window_hours=25))  # hourly: nothing checked
    with pytest.raises(ValueError):
        check_compatible(3, dict(rolling, window_hours=25))
    with pytest.raises(ValueError):
        check_compatible(3, dict(rolling, look_ahead_hours=10))
    with pytest.raises(ValueError):
        check_compatible(4, dict(rolling, look_ahead_resolution_hours=6))
    with pytest.raises(ValueError):
        check_compatible(3, rolling, commitment=True)


def test_hourly_resolution_changes_nothing():
    n = _network()
    assert aggregate_to_blocks(n, 1, LOG) is None
    assert len(n.snapshots) == 48
    assert restore_hourly(n, None, LOG) == []


# ── Aggregation and restore ─────────────────────────────────────────────────

def test_blocks_start_at_the_period_start():
    idx = pd.date_range("2030-01-01", periods=7, freq="h")
    starts = block_starts(idx, 3)
    assert list(starts.dt.hour) == [0, 0, 0, 3, 3, 3, 6]


def test_aggregation_means_inputs_sums_weights_scales_ramps_and_restores_exactly():
    n = _network()
    wind = n.generators_t.p_max_pu.copy()
    load = n.loads_t.p_set.copy()
    blocks = aggregate_to_blocks(n, 3, LOG)
    assert len(n.snapshots) == 16
    assert (n.snapshot_weightings.objective == 3.0).all() and (n.snapshot_weightings.stores == 3.0).all()
    assert n.generators_t.p_max_pu["wind"].iloc[1] == pytest.approx(wind["wind"].iloc[3:6].mean())
    assert n.loads_t.p_set["load"].sum() * 3 == pytest.approx(load["load"].sum())   # block energy kept
    assert n.generators.at["mid", "ramp_limit_up"] == pytest.approx(0.6)            # 0.2/h x 3 h
    assert np.isnan(n.generators.at["gas", "ramp_limit_up"])                        # unlimited stays unlimited
    restore_hourly(n, blocks, LOG)
    assert len(n.snapshots) == 48 and (n.snapshot_weightings.objective == 1.0).all()
    pd.testing.assert_frame_equal(n.generators_t.p_max_pu, wind)
    pd.testing.assert_frame_equal(n.loads_t.p_set, load)
    assert n.generators.at["mid", "ramp_limit_up"] == pytest.approx(0.2)


def test_expand_rows_and_levels():
    n = _network(days=1)
    blocks = aggregate_to_blocks(n, 3, LOG)
    frame = pd.DataFrame({"x": np.arange(8.0)}, index=blocks.block_index)
    out = expand_rows(frame, blocks)
    assert list(out["x"].iloc[:7]) == [0, 0, 0, 1, 1, 1, 2]
    levels = pd.DataFrame({"s": [30.0, 60.0, 60.0, 0.0, 0.0, 0.0, 0.0, 9.0]}, index=blocks.block_index)
    hourly = expand_levels(levels, blocks, pd.Series({"s": 0.0}))
    assert list(hourly["s"].iloc[:9]) == pytest.approx([10, 20, 30, 40, 50, 60, 60, 60, 60])
    assert hourly["s"].iloc[11] == pytest.approx(0.0) and hourly["s"].iloc[-1] == pytest.approx(9.0)


def test_energy_floor_is_aggregated_to_block_means_with_its_attrs():
    n = _network(days=1)
    idx = n.snapshots
    floor = pd.DataFrame({"g": np.arange(24.0)}, index=idx)
    floor.attrs["hours"] = {"g": 24.0}
    blocks = aggregate_to_blocks(n, 3, LOG)
    agg = aggregate_frame(floor, blocks)
    assert list(agg["g"].iloc[:2]) == [1.0, 4.0]
    assert agg.attrs["hours"] == {"g": 24.0}
    assert (agg["g"] * 3).sum() == pytest.approx(floor["g"].sum())


# ── Solves ──────────────────────────────────────────────────────────────────

def test_block_solve_conserves_energy_prices_per_block_and_expands_outputs():
    n = _network()
    load = n.loads_t.p_set["load"].copy()
    blocks = aggregate_to_blocks(n, 3, LOG)
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    block_price = n.buses_t.marginal_price["el"].copy()
    block_soc = n.storage_units_t.state_of_charge["battery"].copy()
    restore_hourly(n, blocks, LOG)
    gen = n.generators_t.p
    su = n.storage_units_t.p["battery"]
    # Energy balance per block: hourly expanded supply = block-mean load x 3 = hourly load energy.
    supply = gen.sum(axis=1) + su
    grouped = supply.groupby(blocks.group).sum()
    assert np.allclose(grouped.to_numpy(), load.groupby(blocks.group).sum().to_numpy(), atol=1e-4)
    assert supply.sum() == pytest.approx(load.sum(), abs=1e-3)
    # Power held within each block; one price per block, in GBP/MWh (duals divided by the weighting).
    for frame in (gen, n.storage_units_t.p, n.buses_t.marginal_price):
        assert (frame.groupby(blocks.group).nunique() <= 1).all().all()
    price = n.buses_t.marginal_price["el"]
    assert set(np.round(price.unique(), 4)) <= {0.0, 1.0, 40.0, 150.0} | set(np.round(block_price.unique(), 4))
    assert price.max() <= 150.0 + 1e-6
    # The ramp limit acts per block at 3 x the hourly rate.
    assert gen["mid"].diff().abs().max() <= 0.6 * 100.0 + 1e-6
    # State of charge: block ends are Stage 1's, linear inside the block.
    soc = n.storage_units_t.state_of_charge["battery"]
    ends = soc.groupby(blocks.group).last()
    assert np.allclose(ends.to_numpy(), block_soc.to_numpy(), atol=1e-6)
    rebuilt = 20.0 + (su.clip(upper=0.0).abs() * 0.9 - su.clip(lower=0.0) / 0.9).cumsum()
    assert np.allclose(rebuilt.to_numpy(), soc.to_numpy(), atol=1e-4)
    record_meta(n, blocks)
    assert n.meta[META_KEY] == {"resolution_hours": 3, "blocks": 16, "hourly_snapshots": 48}


def test_rolling_block_solve_with_coarse_look_ahead_returns_hourly_outputs():
    n = _network(days=3)
    load = n.loads_t.p_set["load"].copy()
    wholesale = {"window_hours": 24, "carry_soc": True, "look_ahead_hours": 24, "look_ahead_resolution_hours": 6}
    blocks = aggregate_to_blocks(n, 3, LOG)
    gen, su, links, prices, _ = solve_rolling_day_ahead(n, wholesale, "highs", {}, {"market": {}}, LOG)
    assert len(gen) == 24  # 3 days x 8 blocks, kept blocks only
    gen, su, prices = (expand_rows(f, blocks) for f in (gen, su, prices))
    restore_hourly(n, blocks, LOG)
    assert gen.index.equals(n.snapshots) and len(gen) == 72
    supply = gen.sum(axis=1) + su["battery"]
    assert supply.sum() == pytest.approx(load.sum(), abs=1e-3)
    assert (prices.groupby(blocks.group).nunique() <= 1).all().all()
    # The saved network carries the hourly legs, which net to the saved schedule.
    legs = n.storage_units_t.p_dispatch["battery"] - n.storage_units_t.p_store["battery"]
    assert np.allclose(legs.to_numpy(), su["battery"].to_numpy(), atol=1e-6)


def test_bm_state_of_charge_rebuild_matches_the_block_solve():
    """solve_balancing rebuilds Stage 1's SoC hour by hour from the hourly legs: equal at every block end."""
    from scripts.market.solve_balancing import stage1_state_of_charge

    n = _network()
    blocks = aggregate_to_blocks(n, 3, LOG)
    n.optimize(solver_name="highs")
    block_soc = n.storage_units_t.state_of_charge["battery"].copy()
    restore_hourly(n, blocks, LOG)
    legs = (n.storage_units_t.p_store, n.storage_units_t.p_dispatch)
    soc = stage1_state_of_charge(n, n.storage_units_t.p, legs)["battery"]
    assert np.allclose(soc.groupby(blocks.group).last().to_numpy(), block_soc.to_numpy(), atol=1e-6)


# ── The balancing stage at the same resolution (market.balancing.temporal_resolution_hours) ─────────────────

def test_balancing_default_is_hourly_and_matches_global_defaults():
    from scripts.market.temporal_resolution import (
        check_balancing_compatible, resolve_balancing_temporal_resolution_hours,
    )
    defaults = load_defaults()
    assert defaults["market"]["balancing"]["temporal_resolution_hours"] == DEFAULT_TEMPORAL_RESOLUTION_HOURS == 1
    assert resolve_balancing_temporal_resolution_hours({}) == 1
    assert resolve_balancing_temporal_resolution_hours({"temporal_resolution_hours": 3}) == 3
    check_balancing_compatible(3, {"mode": "rolling", "window_hours": 24})
    with pytest.raises(ValueError):
        check_balancing_compatible(3, {"mode": "rolling", "window_hours": 1})
    with pytest.raises(ValueError):
        resolve_balancing_temporal_resolution_hours({"temporal_resolution_hours": 1.5})


def test_flatten_holds_inputs_at_block_means_and_keeps_hourly_snapshots():
    from scripts.market.temporal_resolution import flatten_to_blocks
    n = _network()
    wind = n.generators_t.p_max_pu["wind"].copy()
    load = n.loads_t.p_set["load"].copy()
    assert flatten_to_blocks(n, 1, LOG) == 0
    assert flatten_to_blocks(n, 3, LOG) >= 2
    assert len(n.snapshots) == 48 and (n.snapshot_weightings.objective == 1.0).all()
    grp = np.arange(48) // 3
    assert np.allclose(n.generators_t.p_max_pu["wind"].to_numpy(), wind.groupby(grp).transform("mean").to_numpy())
    assert n.loads_t.p_set["load"].sum() == pytest.approx(load.sum())
    assert (n.loads_t.p_set["load"].groupby(grp).nunique() == 1).all()
    assert n.generators.at["mid", "ramp_limit_up"] == pytest.approx(0.6)


def test_held_block_schedule_is_feasible_hour_by_hour_on_flattened_inputs():
    """The 3-hour Stage 1's hourly-held schedule needs no redispatch on block-mean inputs, unlike on hourly ones."""
    from scripts.market.temporal_resolution import flatten_to_blocks
    n = _network()
    hourly = _network()
    blocks = aggregate_to_blocks(n, 3, LOG)
    n.optimize(solver_name="highs")
    restore_hourly(n, blocks, LOG)
    held = n.generators_t.p["wind"]
    avail_hourly = hourly.generators_t.p_max_pu["wind"] * 150.0
    over_hourly = (held - avail_hourly).clip(lower=0).sum()
    flat = _network()
    flatten_to_blocks(flat, 3, LOG)
    avail_flat = flat.generators_t.p_max_pu["wind"] * 150.0
    over_flat = (held - avail_flat).clip(lower=0).sum()
    assert over_hourly > 1.0          # hourly inputs: wind held above the hour's availability (a forced bid)
    assert over_flat < 1e-6           # block-mean inputs: none
    supply = n.generators_t.p.sum(axis=1) + n.storage_units_t.p["battery"]
    assert np.allclose(supply.to_numpy(), flat.loads_t.p_set["load"].to_numpy(), atol=1e-4)
