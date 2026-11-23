"""Stage 1 rolling day-ahead look-ahead and the re-fitted interconnector hurdle (OW1, 3 Oct 2026).

``market.wholesale.look_ahead_hours`` N optimises each rolling window over its own
hours plus the next N and keeps only its own hours, so storage values what it holds
at the window's end at the next day's prices instead of at nothing. Default 0 is
today's loop exactly. CP30 sets 24 and re-fits the single-price hurdle and offset.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.market.market_utils import final_store_levels, min_output_energy_delivered
from scripts.market.solve_wholesale import (
    coarsen_look_ahead,
    keep_window_rows,
    measure_storage_churn,
    resolve_look_ahead_hours,
    resolve_look_ahead_resolution_hours,
    rolling_window_horizons,
    solve_rolling_day_ahead,
)
from config.config_loader import load_defaults, load_layer

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_wholesale_look_ahead")
CP30_MONTH = "CP30_FFR_2030_Jan_cp30_network"


def _network(days=3):
    """Day 1: wind surplus to noon (price 0), a 40 GBP unit after; days 2+: gas at 150 only. A 200 MWh battery."""
    hours = 24 * days
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    wind = [300.0 if i < 12 else 0.0 for i in range(hours)]
    mid = [1.0 if i < 24 else 0.0 for i in range(hours)]
    n.add("Generator", "wind", bus="el", p_nom=1.0, p_max_pu=pd.Series(wind, index=n.snapshots),
          marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "mid", bus="el", p_nom=500.0, p_max_pu=pd.Series(mid, index=n.snapshots),
          marginal_cost=40.0, carrier="biomass")
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=150.0, carrier="CCGT")
    n.add("Load", "load", bus="el", p_set=100.0)
    n.add("StorageUnit", "battery", bus="el", p_nom=50.0, max_hours=4.0, efficiency_store=0.9,
          efficiency_dispatch=0.9, marginal_cost=1.0, state_of_charge_initial=0.0,
          cyclic_state_of_charge=False, carrier="Battery")
    return n


def _solve(n, wholesale):
    gen, su, links, prices, objective = solve_rolling_day_ahead(
        n, wholesale, "highs", {}, {"market": {}}, LOG,
    )
    return gen, su, prices, objective


def _soc(n, su):
    """State of charge rebuilt from the kept dispatch (start 0, eta 0.9 each way)."""
    p = su["battery"]
    return (p.clip(upper=0.0).abs() * 0.9 - p.clip(lower=0.0) / 0.9).cumsum()


# ── Config ──────────────────────────────────────────────────────────────────

def test_the_base_sets_a_day_at_six_hours_and_the_code_default_equals_it():
    defaults = load_defaults()
    # In the base for every scenario since the config restructure (decision D4, 27 Oct 2026).
    assert defaults["market"]["wholesale"]["look_ahead_hours"] == 24
    assert defaults["market"]["wholesale"]["look_ahead_resolution_hours"] == 6
    assert resolve_look_ahead_resolution_hours({}) == 6.0
    assert resolve_look_ahead_resolution_hours({"look_ahead_resolution_hours": None}) == 1.0
    with pytest.raises(ValueError):
        resolve_look_ahead_resolution_hours({"look_ahead_resolution_hours": 0})
    assert resolve_look_ahead_hours({}) == 24.0
    assert resolve_look_ahead_hours({"look_ahead_hours": None}) == 0.0
    with pytest.raises(ValueError):
        resolve_look_ahead_hours({"look_ahead_hours": -1})


def test_cp30_month_carries_look_ahead_and_refitted_hurdle_and_2050_its_own_spread():
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenarios = load_config(include_archive=True)["scenarios"]   # every study, loaded or not
    cp30 = scenarios[CP30_MONTH]
    assert cp30["market"]["wholesale"]["look_ahead_hours"] == 24
    assert cp30["market"]["wholesale"]["look_ahead_resolution_hours"] == 6
    central = scenarios["CP30_FFR_2030_Jan_cp30_network_central_dispatch"]["market"]["wholesale"]
    assert central["look_ahead_hours"] == 24 and central["look_ahead_resolution_hours"] == 6
    pricing = cp30["interconnectors"]["pricing"]
    assert pricing["single_price_transaction_spread_gbp_per_mwh"] == pytest.approx(20.0)
    # F13 (7 Oct 2026): the foreign level comes from the cp30_basis prices themselves (TYNDP NT2030
    # refitted to 2013 weather on CP30's fuel basis), so the build offsets are 0; the SEM takes 2024
    # ENTSO-E plus its own 7.5 inside cp30_basis. (Before F13: offsets 7.5 on one ENTSO-E 2024 price.)
    assert pricing["foreign_price_source"] == "cp30_basis"
    assert pricing["single_price_level_offsets_gbp_per_mwh"] == {"continent": 0.0, "norway": 0.0, "sem": 0.0}
    assert pricing["cp30_basis"]["ireland_offset_gbp"] == pytest.approx(7.5)
    # Margins on the single basis: import ask = offset + hurdle/2, export bid = offset - hurdle/2.
    assert 0.0 + 20.0 / 2 == pytest.approx(10.0)
    assert 0.0 - 20.0 / 2 == pytest.approx(-10.0)
    ht50 = scenarios["FES25_HT_2050_Jan_market"]
    assert resolve_look_ahead_hours(ht50.get("market", {}).get("wholesale", {})) == 24.0   # the base's, D4
    assert resolve_look_ahead_resolution_hours(ht50.get("market", {}).get("wholesale", {})) == 6.0
    assert ht50["interconnectors"]["pricing"]["single_price_transaction_spread_gbp_per_mwh"] == pytest.approx(30.0)


# ── Window bookkeeping ──────────────────────────────────────────────────────

def test_horizons_extend_by_the_look_ahead_and_stop_at_the_period_end():
    sns = pd.date_range("2030-01-01", periods=60, freq="h")
    windows, horizons = rolling_window_horizons(sns, 24, 0)
    assert [len(w) for w in windows] == [24, 24, 12]
    assert all(w.equals(h) for w, h in zip(windows, horizons))
    windows, horizons = rolling_window_horizons(sns, 24, 24)
    assert [len(w) for w in windows] == [24, 24, 12]
    assert [len(h) for h in horizons] == [48, 36, 12]
    assert all(h[: len(w)].equals(w) for w, h in zip(windows, horizons))
    assert pd.Index(np.concatenate([w.values for w in windows])).equals(pd.Index(sns.values))


def test_keep_window_rows_and_store_level_at_the_kept_end():
    idx = pd.date_range("2030-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    assert keep_window_rows(frame, None) is frame
    assert list(keep_window_rows(frame, idx[:2])["a"]) == [1.0, 2.0]
    n = pypsa.Network()
    n.set_snapshots(idx)
    n.stores_t["e"] = pd.DataFrame({"s": [5.0, 6.0, 7.0, 8.0]}, index=idx)
    assert final_store_levels(n, ["s"]) == {"s": 8.0}
    assert final_store_levels(n, ["s"], at=idx[1]) == {"s": 6.0}


def test_energy_budget_and_churn_count_only_the_kept_hours():
    idx = pd.date_range("2030-01-01", periods=4, freq="h")
    n = pypsa.Network()
    n.set_snapshots(idx)
    n.add("Bus", "b")
    n.add("Generator", "g", bus="b", p_nom=10.0)
    n.add("StorageUnit", "s", bus="b", p_nom=10.0, max_hours=2.0, efficiency_store=0.9, efficiency_dispatch=0.9)
    n.generators_t["p"] = pd.DataFrame({"g": [1.0, 1.0, 5.0, 5.0]}, index=idx)
    callback = lambda: None  # noqa: E731
    callback.last = pd.DataFrame({"generator": ["g", "g"], "block": [0, 1], "start": [idx[0], idx[2]],
                                  "hours": [2.0, 2.0], "budget_mwh": [2.0, 10.0]})
    assert min_output_energy_delivered(n, callback) == (12.0, 12.0)
    assert min_output_energy_delivered(n, callback, snapshots=idx[:2]) == (2.0, 2.0)
    n.storage_units_t["p_store"] = pd.DataFrame({"s": [1.0, 0.0, 2.0, 0.0]}, index=idx)
    n.storage_units_t["p_dispatch"] = pd.DataFrame({"s": [1.0, 0.0, 2.0, 0.0]}, index=idx)
    assert measure_storage_churn(n, LOG)["churned_mwh"] == pytest.approx(3.0)
    assert measure_storage_churn(n, LOG, snapshots=idx[:2])["churned_mwh"] == pytest.approx(1.0)


# ── The rolling loop on a toy network (HiGHS) ───────────────────────────────

def test_zero_look_ahead_is_todays_loop_exactly():
    gen_a, su_a, prices_a, obj_a = _solve(_network(), {"window_hours": 24, "look_ahead_hours": 0})
    gen_b, su_b, prices_b, obj_b = _solve(_network(), {"window_hours": 24, "look_ahead_hours": 0})
    pd.testing.assert_frame_equal(gen_a, gen_b)
    pd.testing.assert_frame_equal(su_a, su_b)
    pd.testing.assert_frame_equal(prices_a, prices_b)
    assert obj_a == pytest.approx(obj_b)


def test_look_ahead_holds_energy_over_midnight_for_the_dear_next_day():
    n0, n24 = _network(), _network()
    gen0, su0, prices0, _ = _solve(n0, {"window_hours": 24, "look_ahead_hours": 0})
    gen24, su24, prices24, _ = _solve(n24, {"window_hours": 24, "look_ahead_hours": 24})
    for frame in (gen24, su24, prices24):
        assert frame.index.equals(n24.snapshots)  # kept hours only, each once
    end_day1 = n24.snapshots[23]
    # Myopic: the day sells its energy at 40 by midnight; look-ahead: keeps it for 150.
    assert _soc(n0, su0).loc[end_day1] == pytest.approx(0.0, abs=1e-3)
    assert _soc(n24, su24).loc[end_day1] == pytest.approx(200.0, rel=1e-4)
    day2 = slice(n24.snapshots[24], n24.snapshots[47])
    assert su24["battery"].loc[day2].clip(lower=0.0).sum() > 150.0
    assert su0["battery"].loc[day2].clip(lower=0.0).sum() == pytest.approx(0.0, abs=1e-3)
    # The true cost of the kept dispatch falls: gas displaced at 150 by energy banked at 0.
    mc = n24.generators["marginal_cost"]
    cost0 = float((gen0 * mc).sum().sum())
    cost24 = float((gen24 * mc).sum().sum())
    assert cost24 < cost0 - 1000.0
    # No simultaneous charge and discharge (the Stage 1 churn diagnostic, kept hours only).
    from scripts.market.solve_wholesale import STORAGE_CHURN_META_KEY

    churn = n24.meta[STORAGE_CHURN_META_KEY]
    assert churn["dissipated_mwh"] == pytest.approx(0.0, abs=1e-3)
    assert len(churn["per_window_dissipated_mwh"]) == 3


def test_carried_hydrogen_store_starts_each_window_at_the_kept_end_and_meets_the_month_end():
    """The store level carried to the next window is the level at the last KEPT hour, not the horizon's end."""
    from scripts.market.market_utils import prepare_carried_stores

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=72, freq="h"))
    n.add("Bus", "el", carrier="AC")
    n.add("Bus", "GB_H2", carrier="H2")
    wind = [200.0 if i < 24 else 0.0 for i in range(72)]
    n.add("Generator", "wind", bus="el", p_nom=1.0, p_max_pu=pd.Series(wind, index=n.snapshots),
          marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=150.0, carrier="CCGT")
    n.add("Load", "load", bus="el", p_set=100.0)
    n.add("Link", "electrolysis", bus0="el", bus1="GB_H2", p_nom=100.0, efficiency=0.7, carrier="electrolysis")
    n.add("Link", "h2_turbine", bus0="GB_H2", bus1="el", p_nom=200.0, efficiency=0.5,
          marginal_cost=5.0, carrier="H2_turbine")
    n.add("Store", "GB_H2_storage", bus="GB_H2", carrier="H2_gas", e_nom=4000.0, e_cyclic=True,
          standing_loss=0.0)
    market = {"hydrogen_store": {"month_end": "at_least_initial", "end_of_window_value_gbp_per_mwh_h2": 41.2}}
    start = prepare_carried_stores(n, market, LOG)
    _, _, flows, _, _ = solve_rolling_day_ahead(
        n, {"window_hours": 24, "look_ahead_hours": 24, "look_ahead_resolution_hours": 1}, "highs", {}, {"market": market}, LOG)
    e = n.stores_t.e["GB_H2_storage"]
    assert e.index.equals(n.snapshots) and flows.index.equals(n.snapshots)
    assert e.iloc[-1] >= float(start.sum()) - 1e-3            # month end held
    assert e.iloc[23] > float(start.sum()) + 100.0            # the windy day banks hydrogen
    # Level continuity across the kept windows: each hour's change is what the
    # electrolyser makes less what the turbine burns (p0 at each link's bus0).
    made = 0.7 * flows["electrolysis"] - flows["h2_turbine"]
    level = float(start.sum()) + made.cumsum()
    np.testing.assert_allclose(level.to_numpy(), e.to_numpy(), atol=1e-2)


# ── Coarse look-ahead (follow-up, 3 Oct 2026) ───────────────────────────────

def _varying_network(hours=48):
    rng = np.random.default_rng(7)
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    n.add("Load", "load", bus="el", p_set=pd.Series(80 + 40 * rng.random(hours), index=n.snapshots))
    n.add("Generator", "wind", bus="el", p_nom=200.0, carrier="wind_onshore",
          p_max_pu=pd.Series(rng.random(hours), index=n.snapshots), marginal_cost=0.0)
    n.add("Generator", "gas", bus="el", p_nom=500.0, carrier="CCGT",
          marginal_cost=pd.Series(100 + 50 * rng.random(hours), index=n.snapshots))
    n.add("StorageUnit", "battery", bus="el", p_nom=50.0, max_hours=4.0, efficiency_store=0.9,
          efficiency_dispatch=0.9, marginal_cost=1.0, state_of_charge_initial=100.0, standing_loss=0.01,
          cyclic_state_of_charge=False, carrier="Battery")
    return n


def test_coarsening_conserves_energy_and_keeps_the_day_hourly():
    n = _varying_network()
    original = n.copy()
    kept, horizon = n.snapshots[:24], n.snapshots
    new = coarsen_look_ahead(n, kept, horizon, 6)
    assert len(new) == 28 and new[:24].equals(kept)
    assert list(n.snapshot_weightings.objective.iloc[24:]) == [6.0] * 4
    assert list(n.snapshot_weightings.stores.iloc[24:]) == [6.0] * 4
    w = n.snapshot_weightings.objective
    for frame, before in ((n.loads_t.p_set, original.loads_t.p_set),
                          (n.generators_t.p_max_pu, original.generators_t.p_max_pu)):
        pd.testing.assert_frame_equal(frame.iloc[:24], before.iloc[:24], check_freq=False)  # kept day untouched
        np.testing.assert_allclose(frame.mul(w, axis=0).sum().to_numpy(), before.sum().to_numpy())
    # a block's value is its hours' mean
    assert n.generators_t.marginal_cost["gas"].iloc[25] == pytest.approx(
        original.generators_t.marginal_cost["gas"].iloc[30:36].mean())
    # resolution 1 or no look-ahead: nothing changes
    m = _varying_network()
    assert coarsen_look_ahead(m, m.snapshots[:24], m.snapshots, 1).equals(m.snapshots)
    assert len(m.snapshots) == 48


def test_weighted_state_of_charge_dynamics_on_blocks():
    n = _varying_network()
    coarsen_look_ahead(n, n.snapshots[:24], n.snapshots, 6)
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    st = n.storage_units_t
    soc = st.state_of_charge["battery"].to_numpy()
    w = n.snapshot_weightings.stores.to_numpy()
    prev = np.concatenate([[100.0], soc[:-1]])
    expected = prev * (1 - 0.01) ** w + w * (0.9 * st.p_store["battery"].to_numpy() - st.p_dispatch["battery"].to_numpy() / 0.9)
    # From the second snapshot on (PyPSA's first step from state_of_charge_initial is its own
    # convention); the blocks are snapshots 24-27, each a 6 h step.
    np.testing.assert_allclose(soc[1:], expected[1:], atol=1e-4)
    # energy balance with weights: load energy = generation + net storage energy
    gen = n.generators_t.p.sum(axis=1) + st.p["battery"]
    np.testing.assert_allclose((gen * n.snapshot_weightings.objective).sum(),
                               (n.loads_t.p_set["load"] * n.snapshot_weightings.objective).sum(), rtol=1e-6)


def test_hourly_resolution_reproduces_the_hourly_look_ahead_exactly():
    gen_a, su_a, prices_a, obj_a = _solve(_network(), {"window_hours": 24, "look_ahead_hours": 24,
                                                    "look_ahead_resolution_hours": 1})
    gen_b, su_b, prices_b, obj_b = _solve(_network(), {"window_hours": 24, "look_ahead_hours": 24,
                                                       "look_ahead_resolution_hours": 1})
    pd.testing.assert_frame_equal(gen_a, gen_b)
    pd.testing.assert_frame_equal(su_a, su_b)
    pd.testing.assert_frame_equal(prices_a, prices_b)
    assert obj_a == pytest.approx(obj_b)


def test_coarse_look_ahead_still_holds_energy_over_midnight():
    n = _network()
    gen, su, prices, _ = _solve(n, {"window_hours": 24, "look_ahead_hours": 24, "look_ahead_resolution_hours": 6})
    for frame in (gen, su, prices):
        assert frame.index.equals(n.snapshots)
    assert _soc(n, su).loc[n.snapshots[23]] == pytest.approx(200.0, rel=1e-4)
    assert su["battery"].loc[n.snapshots[24]:n.snapshots[47]].clip(lower=0.0).sum() > 150.0


def test_energy_floor_on_coarse_blocks_is_the_block_mean():
    from scripts.market.market_utils import _floor_on_snapshots

    idx = pd.date_range("2030-01-01", periods=12, freq="h")
    floor = pd.DataFrame({"g": np.arange(12, dtype=float)}, index=idx)
    uniform = pd.Series(1.0, index=idx[:6])
    pd.testing.assert_frame_equal(_floor_on_snapshots(floor, idx[:6], uniform), floor.iloc[:6])
    sns = idx[:6].append(pd.DatetimeIndex([idx[6], idx[9]]))
    w = pd.Series([1.0] * 6 + [3.0, 3.0], index=sns)
    out = _floor_on_snapshots(floor, sns, w)
    assert list(out["g"]) == [0, 1, 2, 3, 4, 5, 7.0, 10.0]
    assert float((out["g"] * w).sum()) == pytest.approx(floor["g"].sum())
