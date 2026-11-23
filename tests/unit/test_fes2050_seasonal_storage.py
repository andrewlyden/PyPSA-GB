"""Seasonal storage, two-pass approach (plan decision 33, task Y8a).

Pass 1: the annual copperplate pass on an aggregated fleet (scripts/market/seasonal_storage.py).
Pass 2: monthly rolling windows started on its trajectory, valued at its duals, never
pinned at the end (soft floor optional), with storage units on the pass optionally
(scripts/market/market_utils.py, ``market.hydrogen_store.seasonal_trajectory``).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.copperplate import MAINLAND_HUB
from scripts.market.market_utils import (
    apply_initial_storage_soc,
    build_end_of_window_store_value_callback,
    build_month_end_soc_callback,
    build_month_end_store_callback,
    objective_only_adjustment,
    prepare_carried_stores,
    resolve_storage_soc_config,
    resolve_store_level_config,
    store_end_of_window_credit,
)
from scripts.market.seasonal_storage import (
    aggregate_for_seasonal_pass,
    coarsen_network,
    extract_trajectory,
    resolve_seasonal_pass_config,
    resolve_seasonal_trajectory_config,
    run_annual_pass,
    solve_annual_pass,
)

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_fes2050_seasonal_storage")
HOURS = 72


def _toy(hours=HOURS, seed=7):
    """Three mainland buses, an island, a foreign bus, a hydrogen system; multi-unit carriers."""
    rng = np.random.default_rng(seed)
    n = pypsa.Network()
    sns = pd.date_range("2050-01-01", periods=hours, freq="h")
    n.set_snapshots(sns)
    for b in ("A", "B", "C", "ISL"):
        n.add("Bus", b, carrier="AC", v_nom=400.0, x=float(ord(b[0])), y=0.0)
    n.add("Bus", "External_FR", carrier="AC", v_nom=400.0)
    n.add("Bus", "GB_H2", carrier="H2_gas")
    n.add("Bus", "GB_gas", carrier="natural_gas")
    n.add("Line", "AB", bus0="A", bus1="B", x=0.1, s_nom=300.0)
    n.add("Line", "BC", bus0="B", bus1="C", x=0.1, s_nom=300.0)
    n.add("Link", "ISL_HVDC", bus0="ISL", bus1="A", p_nom=5000.0, p_min_pu=-1.0, carrier="DC")
    t = np.arange(hours)
    for i, b in enumerate(("A", "B", "C", "ISL")):
        wind = np.clip(0.5 + 0.4 * np.sin(t / 9.0 + i) + 0.1 * rng.standard_normal(hours), 0, 1)
        n.add("Generator", f"wind_{b}", bus=b, carrier="wind_onshore", p_nom=200.0 + 50 * i,
              p_max_pu=pd.Series(wind, index=sns), marginal_cost=0.0)
    for i, b in enumerate(("A", "C")):
        solar = np.clip(np.sin((t % 24 - 6) / 12 * np.pi), 0, 1) * (0.8 + 0.1 * i)
        n.add("Generator", f"solar_{b}", bus=b, carrier="solar_pv", p_nom=150.0, p_max_pu=pd.Series(solar, index=sns))
    for i, b in enumerate(("A", "B")):
        avail = np.where(t % 30 < 3 + i, 0.6, 1.0)
        n.add("Generator", f"ccs_{b}", bus=b, carrier="CCS Gas", p_nom=300.0, marginal_cost=66.0,
              p_max_pu=pd.Series(avail, index=sns), ramp_limit_up=0.5, ramp_limit_down=0.5)
    n.add("Generator", "nuclear_C", bus="C", carrier="nuclear", p_nom=200.0, p_min_pu=0.5, marginal_cost=10.0)
    n.add("Generator", "biogas_A", bus="A", carrier="biogas", p_nom=40.0, marginal_cost=81.0)
    n.add("Generator", "biogas_B", bus="B", carrier="biogas", p_nom=60.0, marginal_cost=81.0)
    n.add("Generator", "EU_FR", bus="External_FR", carrier="EU_market", p_nom=500.0, p_min_pu=-1.0)
    n.add("Generator", "gas_supply", bus="GB_gas", carrier="gas_supply", p_nom=400.0, marginal_cost=23.6)
    for b in ("A", "B", "C"):
        load = 400.0 + 150.0 * np.sin(t / 24.0 * 2 * np.pi + 1.0)
        n.add("Load", f"load_{b}", bus=b, carrier="electricity", p_set=pd.Series(load, index=sns))
        n.add("Generator", f"shed_{b}", bus=b, carrier="load_shedding", p_nom=1e4, marginal_cost=6000.0)
    for b in ("A", "C"):
        n.add("StorageUnit", f"battery_{b}", bus=b, carrier="Battery", p_nom=100.0, max_hours=2.0,
              efficiency_store=0.92, efficiency_dispatch=0.92, marginal_cost=1.0, p_min_pu=-1.0,
              state_of_charge_initial=100.0)
    # Consumer flexibility: one national profile spread by bus share (proportional members).
    plug = pd.Series(np.where(t % 24 >= 18, 0.9, 0.4), index=sns)
    drain = pd.Series(-np.where((t % 24 >= 7) & (t % 24 < 9), 0.3, 0.05), index=sns)
    for b, share in (("A", 0.6), ("B", 0.4)):
        n.add("StorageUnit", f"ev_{b}", bus=b, carrier="EV_smart_charging", p_nom=200.0 * share, max_hours=5.0,
              efficiency_store=0.9, efficiency_dispatch=1.0, p_max_pu=0.0, p_min_pu=-plug,
              inflow=drain * 200.0 * share, state_of_charge_initial=500.0 * share)
    n.add("Link", "IC_FR", bus0="External_FR", bus1="B", p_nom=300.0, p_min_pu=-1.0, efficiency=0.975,
          carrier="DC", marginal_cost=pd.Series(50.0 + 20.0 * np.sin(t / 7.0), index=sns))
    for b in ("A", "C"):
        n.add("Link", f"electrolysis_{b}", bus0=b, bus1="GB_H2", p_nom=150.0, efficiency=0.815, carrier="electrolysis")
        n.add("Link", f"H2_turbine_{b}", bus0="GB_H2", bus1=b, p_nom=400.0, efficiency=0.381, carrier="H2_turbine",
              marginal_cost=5.0)
    n.add("Link", "reformer", bus0="GB_gas", bus1="GB_H2", p_nom=200.0, efficiency=0.834, carrier="gas_reforming_ccs",
          marginal_cost=7.0)
    n.add("Store", "GB_H2_storage", bus="GB_H2", carrier="H2_gas", e_nom=20000.0, e_cyclic=True)
    n.add("Load", "H2_demand", bus="GB_H2", carrier="H2_demand", p_set=150.0)
    return n


def _reference_copperplate(n):
    """The full network as Stage 1 sees it: lines relaxed, island HVDC non-binding; cyclic storage."""
    ref = n.copy()
    ref.lines["s_nom"] = 1e6
    ref.storage_units["cyclic_state_of_charge"] = True
    ref.stores["e_cyclic"] = True
    return ref


def _pass_cfg(**over):
    return resolve_seasonal_pass_config({"seasonal_pass": over})


# ── Aggregation exactness ────────────────────────────────────────────────

def test_aggregation_preserves_capacity_energy_and_load():
    n = _toy()
    agg, report, _ = aggregate_for_seasonal_pass(n, {}, _pass_cfg(), LOG)
    assert set(agg.buses.index) == {MAINLAND_HUB, "External_FR", "GB_H2", "GB_gas"}
    assert agg.lines.empty
    for carrier in n.generators.carrier.unique():
        g = n.generators[n.generators.carrier == carrier]
        a = agg.generators[agg.generators.carrier == carrier]
        assert a.p_nom.sum() == pytest.approx(g.p_nom.sum())
        full = n.get_switchable_as_dense("Generator", "p_max_pu")[g.index].mul(g.p_nom).sum(axis=1)
        pooled = agg.get_switchable_as_dense("Generator", "p_max_pu")[a.index].mul(a.p_nom).sum(axis=1)
        np.testing.assert_allclose(pooled.to_numpy(), full.to_numpy(), rtol=1e-12, atol=1e-9)
    assert len(agg.generators[agg.generators.carrier == "wind_onshore"]) == 1   # island merged too
    assert agg.loads_t.p_set.sum().sum() == pytest.approx(n.get_switchable_as_dense("Load", "p_set").sum().sum())
    assert report["loads"]["energy_twh"] == pytest.approx(report["loads"]["energy_twh_aggregated"])
    su = agg.storage_units
    assert su.loc[su.carrier == "EV_smart_charging", "p_nom"].sum() == pytest.approx(200.0)
    assert agg.storage_units_t.inflow.sum().sum() == pytest.approx(n.storage_units_t.inflow.sum().sum())
    # Proportional members: the pooled flexibility is exact (zero deviation).
    assert report["storage_units"]["max_member_deviation"]["EV_smart_charging"]["inflow_pu"] == pytest.approx(0.0, abs=1e-12)
    assert "ISL_HVDC" not in agg.links.index and "IC_FR" in agg.links.index
    assert agg.links.at["IC_FR", "bus1"] == MAINLAND_HUB
    assert "IC_FR" in agg.links_t.marginal_cost.columns
    assert agg.links[agg.links.carrier == "electrolysis"].p_nom.sum() == pytest.approx(300.0)
    ramp = agg.generators.loc[agg.generators.carrier == "CCS Gas", "ramp_limit_up"]
    assert float(ramp.iloc[0]) == pytest.approx(0.5)


def test_aggregated_copperplate_has_the_full_copperplate_objective():
    n = _toy()
    n.generators[["ramp_limit_up", "ramp_limit_down"]] = np.nan   # the pooled ramp is a relaxation
    ref = _reference_copperplate(n)
    status, condition = ref.optimize(solver_name="highs")
    assert status == "ok", condition
    agg, _, _ = aggregate_for_seasonal_pass(n, {}, _pass_cfg(), LOG)
    status, condition = agg.optimize(solver_name="highs")
    assert status == "ok", condition
    assert float(agg.objective) == pytest.approx(float(ref.objective), rel=1e-6)
    # Same totals: load shed, hydrogen made and burnt (the optimum's energy balance).
    shed = lambda net: float(net.generators_t.p[net.generators.index[net.generators.carrier == "load_shedding"]].sum().sum())  # noqa: E731
    assert shed(agg) == pytest.approx(shed(ref), abs=1e-3)


def test_coarsening_preserves_energy():
    n = _toy()
    agg, _, _ = aggregate_for_seasonal_pass(n, {}, _pass_cfg(), LOG)
    load = float(agg.loads_t.p_set.mul(agg.snapshot_weightings.generators, axis=0).sum().sum())
    coarse, _ = coarsen_network(agg.copy(), 3)
    assert len(coarse.snapshots) == HOURS // 3
    assert float(coarse.loads_t.p_set.mul(coarse.snapshot_weightings.generators, axis=0).sum().sum()) == pytest.approx(load)
    assert set(coarse.snapshot_weightings.generators) == {3.0}


# ── The annual pass end to end ───────────────────────────────────────────

def test_run_annual_pass_gives_an_hourly_trajectory_and_a_dual():
    n = _toy()
    trajectory, summary = run_annual_pass([n], {}, "highs", {}, LOG)
    assert summary["solver"]["condition"] == "optimal"
    assert len(trajectory) == HOURS
    assert list(trajectory.columns[:4]) == ["e_start_mwh", "e_end_mwh", "e_nom_mwh", "value_gbp_per_mwh_h2"]
    # e_start is the previous hour's end; cyclic: the first start is the last end.
    np.testing.assert_allclose(trajectory["e_start_mwh"].iloc[1:].to_numpy(), trajectory["e_end_mwh"].iloc[:-1].to_numpy())
    assert trajectory["e_start_mwh"].iloc[0] == pytest.approx(trajectory["e_end_mwh"].iloc[-1])
    # The store dual is the hydrogen price whenever the store is strictly inside its bounds.
    inside = (trajectory["e_end_mwh"] > 1.0) & (trajectory["e_end_mwh"] < trajectory["e_nom_mwh"] - 1.0)
    gap = (trajectory["value_gbp_per_mwh_h2"] - trajectory["h2_bus_price_gbp_per_mwh_h2"]).abs()[inside]
    assert inside.any() and float(gap.max()) < 1e-4
    assert summary["annual"]["load_shedding"]["twh"] >= 0.0
    assert summary["aggregation"]["generators"]["aggregated"] < summary["aggregation"]["generators"]["original"]


def test_hydrogen_reserve_holds_the_store_above_the_floor():
    free, _ = run_annual_pass([_toy()], {}, "highs", {}, LOG)
    e_nom = float(free["e_nom_mwh"].iloc[0])
    reserve_mwh = 0.5 * (free["e_end_mwh"].min() + free["e_end_mwh"].max())
    assert free["e_end_mwh"].min() < reserve_mwh - 1.0   # the free pass goes below the reserve
    held, summary = run_annual_pass(
        [_toy()], {"market": {"seasonal_pass": {"hydrogen_reserve_twh": reserve_mwh / 1e6}}}, "highs", {}, LOG,
    )
    assert summary["solver"]["condition"] == "optimal"
    assert held["e_end_mwh"].min() >= reserve_mwh - 1e-3
    assert summary["aggregation"]["hydrogen_reserve"]["e_min_pu"] == pytest.approx(reserve_mwh / e_nom)
    assert summary["pass_config"]["hydrogen_reserve_twh"] == pytest.approx(reserve_mwh / 1e6)


def test_hydrogen_reserve_is_off_by_default_and_validated():
    assert _pass_cfg()["hydrogen_reserve_twh"] == 0.0
    with pytest.raises(ValueError, match="hydrogen_reserve_twh"):
        _pass_cfg(hydrogen_reserve_twh=-1)
    with pytest.raises(ValueError, match="below the stores' e_nom"):
        run_annual_pass([_toy()], {"market": {"seasonal_pass": {"hydrogen_reserve_twh": 1e6}}}, "highs", {}, LOG)


def test_coarse_pass_is_interpolated_back_to_hours():
    n = _toy()
    trajectory, summary = run_annual_pass([n], {"market": {"seasonal_pass": {"resolution_hours": 3}}}, "highs", {}, LOG)
    assert len(trajectory) == HOURS and summary["snapshots"] == HOURS // 3
    assert trajectory["e_end_mwh"].notna().all() and trajectory["value_gbp_per_mwh_h2"].notna().all()


# ── Pass 2: the monthly windows on the trajectory ────────────────────────

def _h2_network(hours=48, windy_day=1):
    """The end-of-window test network: surplus wind one day, 150 GBP gas the other."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    n.add("Bus", "GB_H2", carrier="H2")
    wind = [200.0 if (i < 24) == (windy_day == 1) else 0.0 for i in range(hours)]
    n.add("Generator", "wind", bus="el", p_nom=1.0, p_max_pu=pd.Series(wind, index=n.snapshots), carrier="wind_onshore")
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=150.0, carrier="CCGT")
    n.add("Generator", "shed", bus="el", p_nom=1000.0, marginal_cost=6000.0, carrier="load_shedding")
    n.add("Load", "load", bus="el", p_set=100.0)
    n.add("Link", "electrolysis", bus0="el", bus1="GB_H2", p_nom=100.0, efficiency=0.7, carrier="electrolysis")
    n.add("Link", "h2_turbine", bus0="GB_H2", bus1="el", p_nom=200.0, efficiency=0.5, marginal_cost=5.0,
          carrier="H2_turbine")
    n.add("Store", "GB_H2_storage", bus="GB_H2", carrier="H2_gas", e_nom=4000.0, e_cyclic=True)
    return n


def _trajectory_file(tmp_path, index, start, end, values):
    """A trajectory CSV: level ``start`` before the first hour, ``end`` after the last; values per hour."""
    e_end = np.linspace(start, end, len(index) + 1)[1:]
    frame = pd.DataFrame({"e_start_mwh": np.r_[start, e_end[:-1]], "e_end_mwh": e_end, "e_nom_mwh": 4000.0,
                          "value_gbp_per_mwh_h2": values}, index=pd.DatetimeIndex(index, name="snapshot"))
    path = tmp_path / "trajectory.csv"
    frame.to_csv(path)
    return str(path)


def _market(path, **over):
    block = {"month_end": "free", "end_of_window_value_gbp_per_mwh_h2": 36.75,
             "seasonal_trajectory": {"enabled": True, "file": path, **over}}
    return {"hydrogen_store": block}


def _rolling(n, market):
    from scripts.market.solve_wholesale import solve_rolling_day_ahead

    apply_initial_storage_soc(n, market, LOG)
    prepare_carried_stores(n, market, LOG)
    return solve_rolling_day_ahead(n, {"window_hours": 24, "look_ahead_hours": 0}, "highs", {}, {"market": market}, LOG)


def _dispatch_cost(n, gen, links):
    cost = float((gen * n.generators["marginal_cost"].reindex(gen.columns)).sum().sum())
    return cost + float((links * n.links["marginal_cost"].reindex(links.columns)).sum().sum())


def test_defaults_are_off_and_hard_end_modes_are_refused(tmp_path):
    cfg = resolve_seasonal_trajectory_config({})
    assert cfg["enabled"] is False and cfg["storage_units"] is False
    assert cfg["end_value"] == "dual" and cfg["end_condition"] == "value"
    # The hard end pins of the first draft are retired, with a pointer.
    with pytest.raises(ValueError, match="retired"):
        resolve_store_level_config({"hydrogen_store": {"month_end": "trajectory"}})
    for retired in ("equal", "at_least"):
        with pytest.raises(ValueError, match="retired"):
            resolve_seasonal_trajectory_config({"hydrogen_store": {"seasonal_trajectory": {"end_condition": retired}}})
    with pytest.raises(ValueError, match="retired"):
        resolve_seasonal_trajectory_config({"hydrogen_store": {"seasonal_trajectory": {"window_floor": True}}})
    with pytest.raises(ValueError):
        resolve_seasonal_trajectory_config({"hydrogen_store": {"seasonal_trajectory": {"enabled": True}}})
    with pytest.raises(ValueError):
        resolve_seasonal_trajectory_config({"hydrogen_store": {"seasonal_trajectory": {"end_value": "max"}}})
    with pytest.raises(ValueError, match="shortfall_penalty"):
        resolve_seasonal_trajectory_config({"hydrogen_store": {"seasonal_trajectory": {"end_condition": "soft_floor"}}})
    with pytest.raises(ValueError, match="only with"):
        resolve_seasonal_trajectory_config(
            {"hydrogen_store": {"seasonal_trajectory": {"shortfall_penalty_gbp_per_mwh_h2": 10.0}}})
    with pytest.raises(ValueError, match="unknown"):
        resolve_seasonal_trajectory_config({"hydrogen_store": {"seasonal_trajectory": {"end_pin": True}}})
    # With the trajectory, month_end must be free (at_least_initial would pin the end hard).
    path = _trajectory_file(tmp_path, pd.date_range("2050-01-01", periods=24, freq="h"), 1000.0, 1000.0, 1.0)
    with pytest.raises(ValueError, match="month_end 'free'"):
        resolve_store_level_config({"hydrogen_store": {"month_end": "at_least_initial",
                                                       "seasonal_trajectory": {"enabled": True, "file": path}}})
    # Without the key the flat end value and the existing month-end callback are unchanged.
    n = _h2_network()
    start = prepare_carried_stores(n, {"hydrogen_store": {"end_of_window_value_gbp_per_mwh_h2": 36.75}}, LOG)
    cb = build_end_of_window_store_value_callback({"hydrogen_store": {"end_of_window_value_gbp_per_mwh_h2": 36.75}},
                                                  start, LOG)
    assert cb.value == pytest.approx(36.75) and not getattr(cb, "time_varying", False)
    assert start["GB_H2_storage"] == pytest.approx(2000.0)
    assert objective_only_adjustment(n, cb, None) == 0.0


def test_store_starts_on_the_trajectory_and_the_end_is_not_pinned(tmp_path):
    # Windy day 1 (banks up to e_nom), 150 GBP gas day 2: at a value of 40 the turbines
    # (40 / 0.5 + 5 = 85) beat gas, so the store is burnt well below the trajectory's end.
    n = _h2_network()
    path = _trajectory_file(tmp_path, n.snapshots, 3000.0, 1200.0, 40.0)
    market = _market(path)
    start = prepare_carried_stores(n, market, LOG)
    assert start["GB_H2_storage"] == pytest.approx(3000.0)
    assert build_month_end_store_callback(n, market, start, n.snapshots[-1], LOG) is None
    n = _h2_network()
    gen, su, links, prices, objective = _rolling(n, market)
    e = n.stores_t.e["GB_H2_storage"]
    assert float(e.iloc[0]) <= 3000.0 + 100.0 * 0.7 + 1e-6   # first hour from the pinned start
    assert float(e.iloc[-1]) < 1200.0 - 500.0                  # not held at the trajectory's end
    assert float(gen.filter(like="shed").sum().sum()) == pytest.approx(0.0, abs=1e-6)


def test_soft_floor_holds_the_path_only_when_it_is_worth_it(tmp_path):
    n0 = _h2_network()
    path = _trajectory_file(tmp_path, n0.snapshots, 3000.0, 1200.0, 40.0)
    # P = 100 over V = 40: below the path hydrogen is worth 140, a turbine MWh 140 / 0.5 + 5
    # = 285 > gas 150, so gas runs and the store ends at the trajectory's level.
    n = _h2_network()
    gen, su, links, prices, objective = _rolling(n, _market(path, end_condition="soft_floor",
                                                            shortfall_penalty_gbp_per_mwh_h2=100.0))
    assert float(n.stores_t.e["GB_H2_storage"].iloc[-1]) == pytest.approx(1200.0, abs=1.0)
    # The penalty and the end value are objective only: the reported objective is the dispatch cost.
    assert objective == pytest.approx(_dispatch_cost(n, gen, links), rel=1e-6)
    # P = 20: (40 + 20) / 0.5 + 5 = 125 < 150, so the turbines still burn and the shortfall is paid.
    n = _h2_network()
    gen, su, links, prices, objective = _rolling(n, _market(path, end_condition="soft_floor",
                                                            shortfall_penalty_gbp_per_mwh_h2=20.0))
    assert float(n.stores_t.e["GB_H2_storage"].iloc[-1]) < 1200.0 - 500.0
    assert objective == pytest.approx(_dispatch_cost(n, gen, links), rel=1e-6)


def test_soft_floor_never_sheds_load_to_keep_hydrogen(tmp_path):
    n0 = _h2_network()
    path = _trajectory_file(tmp_path, n0.snapshots, 5000.0, 2500.0, 40.0)

    def no_gas():
        net = _h2_network()
        net.generators.loc["gas", "p_nom"] = 0.0
        net.stores.loc["GB_H2_storage", "e_nom"] = 10000.0   # enough hydrogen for the still day
        return net

    # VOLL 6000 x turbine efficiency 0.5 = 3000: V 40 + P at or above it is refused.
    n = no_gas()
    market = _market(path, end_condition="soft_floor", shortfall_penalty_gbp_per_mwh_h2=2960.0)
    start = prepare_carried_stores(n, market, LOG)
    with pytest.raises(ValueError, match="VOLL-equivalent"):
        build_month_end_store_callback(n, market, start, n.snapshots[-1], LOG)
    # Just below it the turbines still serve the load on the still day (6680 MWh_H2 in
    # store, 4800 burnt): no shedding, the shortfall below the path is paid instead.
    n = no_gas()
    gen, su, links, prices, objective = _rolling(n, _market(path, end_condition="soft_floor",
                                                            shortfall_penalty_gbp_per_mwh_h2=2900.0))
    assert float(gen["shed"].sum()) == pytest.approx(0.0, abs=1e-3)
    assert float(n.stores_t.e["GB_H2_storage"].iloc[-1]) < 2500.0 - 100.0


def test_soft_floor_every_window_vs_period_end(tmp_path):
    # A flat trajectory at 3000 and a still first day: every_window makes day 1 hold the
    # path (gas at 150 < (40 + 100) / 0.5 + 5); period_end lets day 1 burn and day 2 refill.
    n = _h2_network(windy_day=2)
    path = _trajectory_file(tmp_path, n.snapshots, 3000.0, 3000.0, 40.0)
    _rolling(n, _market(path, end_condition="soft_floor", shortfall_penalty_gbp_per_mwh_h2=100.0))
    assert float(n.stores_t.e["GB_H2_storage"].iloc[23]) == pytest.approx(3000.0, abs=1.0)
    n = _h2_network(windy_day=2)
    _rolling(n, _market(path, end_condition="soft_floor", shortfall_penalty_gbp_per_mwh_h2=100.0,
                        soft_floor_windows="period_end"))
    assert float(n.stores_t.e["GB_H2_storage"].iloc[23]) < 3000.0 - 500.0


def test_time_varying_end_value_follows_the_dual(tmp_path):
    # Windy day 1: a high value at its last hour banks hydrogen, zero does not.
    n = _h2_network()
    values = np.where(np.arange(48) == 23, 200.0, 0.0)
    path = _trajectory_file(tmp_path, n.snapshots, 2000.0, 2000.0, values)
    gen, su, links, prices, objective = _rolling(n, _market(path))
    assert float(links["electrolysis"].iloc[:24].sum()) == pytest.approx(2400.0, rel=1e-4)
    path0 = _trajectory_file(tmp_path, n.snapshots, 2000.0, 0.0, 0.0)
    n0 = _h2_network()
    gen0, _, links0, _, _ = _rolling(n0, _market(path0))
    assert float(links0["electrolysis"].iloc[:24].sum()) == pytest.approx(0.0, abs=1e-3)


def test_credit_uses_the_value_applied_last(tmp_path):
    n = _h2_network(hours=24)
    values = np.full(24, 10.0)
    values[-1] = 77.0
    path = _trajectory_file(tmp_path, n.snapshots, 2000.0, 2000.0, values)
    market = {"hydrogen_store": {"month_end": "free", "seasonal_trajectory": {"enabled": True, "file": path}}}
    start = prepare_carried_stores(n, market, LOG)
    cb = build_end_of_window_store_value_callback(market, start, LOG)
    assert cb.time_varying
    status, condition = n.optimize(solver_name="highs", extra_functionality=cb)
    assert status == "ok", condition
    mwh, gbp = store_end_of_window_credit(n, cb)
    assert cb.last_value == pytest.approx(77.0)
    assert gbp == pytest.approx(77.0 * mwh)


def test_reported_objective_strips_the_time_varying_credit(tmp_path):
    n = _h2_network()
    values = np.where(np.arange(48) == 23, 120.0, 0.0)
    path = _trajectory_file(tmp_path, n.snapshots, 2000.0, 2500.0, values)
    gen, su, links, prices, objective = _rolling(n, _market(path))
    assert objective == pytest.approx(_dispatch_cost(n, gen, links), rel=1e-6)


# ── Storage units on the pass ────────────────────────────────────────────

def _su_network(hours=48):
    """``_h2_network`` plus two batteries of different size and a still second day."""
    n = _h2_network(hours=hours)
    for name, p_nom in (("bat_a", 50.0), ("bat_b", 150.0)):
        n.add("StorageUnit", name, bus="el", carrier="Battery", p_nom=p_nom, max_hours=4.0, efficiency_store=0.9,
              efficiency_dispatch=0.9, marginal_cost=1.0, state_of_charge_initial=0.0)
    return n


def _su_trajectory(tmp_path, n, battery_level, battery_value, h2_start=2000.0):
    path = _trajectory_file(tmp_path, n.snapshots, h2_start, h2_start, 40.0)
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame["su_soc_end_mwh|Battery"] = battery_level
    frame["su_value|Battery"] = battery_value
    frame.to_csv(path)
    return path


def test_storage_units_start_on_the_pass_pro_rata_and_capped(tmp_path):
    n = _su_network()
    # The level before the first hour is the last hour's (cyclic pass).
    levels = np.full(48, 300.0)
    levels[-1] = 400.0
    path = _su_trajectory(tmp_path, n, levels, 50.0)
    market = {**_market(path, storage_units=True), "storage_soc": {"month_end": "free"}}
    apply_initial_storage_soc(n, market, LOG)
    soc0 = n.storage_units["state_of_charge_initial"]
    assert soc0["bat_a"] == pytest.approx(100.0) and soc0["bat_b"] == pytest.approx(300.0)
    levels[-1] = 5000.0   # more than the 800 MWh the units hold: each capped at its capacity
    path = _su_trajectory(tmp_path, n, levels, 50.0)
    market = {**_market(path, storage_units=True), "storage_soc": {"month_end": "free"}}
    apply_initial_storage_soc(n, market, LOG)
    assert n.storage_units["state_of_charge_initial"].to_dict() == pytest.approx({"bat_a": 200.0, "bat_b": 600.0})
    # The storage_soc rules that would pin the start or the end hard are refused.
    with pytest.raises(ValueError, match="month_end"):
        resolve_storage_soc_config({**_market(path, storage_units=True)})
    with pytest.raises(ValueError, match="initial_fraction"):
        resolve_storage_soc_config({**_market(path, storage_units=True),
                                    "storage_soc": {"month_end": "free", "initial_fraction": 0.5}})


def test_storage_period_end_is_valued_only_in_the_last_window(tmp_path):
    n = _su_network()
    path = _su_trajectory(tmp_path, n, 400.0, 0.0)
    values = np.zeros(48)
    values[-1] = 500.0    # worth more than gas at the period end: the last window keeps it full
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame["su_value|Battery"] = values
    frame.to_csv(path)
    market = {**_market(path, storage_units=True), "storage_soc": {"month_end": "free"}}
    cb = build_month_end_soc_callback(n, market, n.snapshots[-1], LOG)
    assert hasattr(cb, "objective_only_gbp")
    gen, su, links, prices, objective = _rolling(n, market)
    assert float(n.storage_units["state_of_charge_initial"].sum()) == pytest.approx(400.0)
    dispatch, store = su.clip(lower=0.0), (-su).clip(lower=0.0)
    end = 400.0 + float((0.9 * store - dispatch / 0.9).sum().sum())
    assert end == pytest.approx(800.0, abs=1.0)    # full at the period end
    # The credit is stripped: the reported objective is the dispatch cost (storage at 1 GBP/MWh).
    su_cost = float(dispatch.sum().sum()) * 1.0
    assert objective == pytest.approx(_dispatch_cost(n, gen, links) + su_cost, rel=1e-6)
    # Without the key the month end is the old at_least_initial floor (a plain callback).
    plain = build_month_end_soc_callback(_su_network(), {}, n.snapshots[-1], LOG)
    assert plain is not None and not hasattr(plain, "objective_only_gbp")


def test_annual_pass_writes_storage_states_and_values():
    from scripts.market.seasonal_storage import trajectory_storage_start, trajectory_storage_value

    n = _toy()
    trajectory, summary = run_annual_pass([n], {}, "highs", {}, LOG)
    assert {"su_soc_end_mwh|Battery", "su_value|Battery", "su_soc_end_mwh|EV_smart_charging"} <= set(trajectory.columns)
    start = trajectory_storage_start(trajectory, trajectory.index[0])
    assert start["Battery"] == pytest.approx(float(trajectory["su_soc_end_mwh|Battery"].iloc[-1]))
    assert float(trajectory_storage_start(trajectory, trajectory.index[5])["Battery"]) == pytest.approx(
        float(trajectory["su_soc_end_mwh|Battery"].iloc[4]))
    # Positive values (the store's sign convention), and between the cheapest charge and
    # dearest discharge price a battery could see.
    value = trajectory["su_value|Battery"]
    price = trajectory["gb_price_gbp_per_mwh"]
    assert (value > 0).all()
    assert float(value.max()) <= float(price.max()) / 0.92 + 1e-6
    assert float(trajectory_storage_value(trajectory, trajectory.index[3])["Battery"]) == pytest.approx(float(value.iloc[3]))
    assert "Battery" in summary["annual"]["storage_units_value_gbp_per_mwh"]


# ── Scenario wiring and the 2030 year ────────────────────────────────────

# 11 Nov 2026 (user): the FES 2040 study (config/studies/fes2040, templates ht40_*) runs on 2050's methods,
# so its FES25_HT_2040_* months set the hydrogen-store seasonal trajectory too. Only the long-range FES
# studies (2040, 2050, the prepared 2045, and the archived HT50 ones) may set it; no CP30/2030 scenario does.
LONG_RANGE_FES_PREFIXES = ("HT50_", "FES25_HT_2040", "FES25_HT_2045", "FES25_HT_2050")


def test_only_ht50_scenarios_set_the_trajectory():
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenarios = load_config(include_archive=True)["scenarios"]   # the gate weeks are archived
    on = sorted(s for s, c in scenarios.items()
                if resolve_seasonal_trajectory_config((c or {}).get("market", {}))["enabled"])
    assert on and all(s.startswith(LONG_RANGE_FES_PREFIXES) for s in on)
    assert {"HT50_1week_gate_seasonal", "FES25_HT_2050_Jan_market", "FES25_HT_2050_Dec_market",
            "FES25_HT_2040_Jan_market", "FES25_HT_2040_Dec_market"} <= set(on)
    assert not any(s.startswith("CP30") or (scenarios[s] or {}).get("modelled_year") == 2030 for s in on)
    gate = scenarios["HT50_1week_gate_seasonal"]["market"]
    assert resolve_store_level_config(gate)["month_end"] == "free"
    cfg = resolve_seasonal_trajectory_config(gate)
    assert cfg["end_condition"] == "soft_floor" and cfg["storage_units"]
    assert resolve_storage_soc_config(gate)["month_end"] == "free"
    for sid, c in scenarios.items():
        market = (c or {}).get("market", {}) or {}
        # Every scenario's market block still resolves (no retired keys anywhere).
        resolve_store_level_config(market)
        resolve_storage_soc_config(market)
        if not sid.startswith(LONG_RANGE_FES_PREFIXES):
            assert "seasonal_trajectory" not in (market.get("hydrogen_store", {}) or {})
            assert "seasonal_pass" not in market


def test_surplus_discount_breaks_the_tie_with_a_hydrogen_supplier_at_the_end_value(tmp_path):
    # 9 Nov 2026: a reformer whose hydrogen costs the end value V (2050: 64.0000 against the annual dual
    # 63.99994) is indifferent between filling the store above the path and not. With the discount D the
    # level above the path is worth V - D, so the reformer only holds the path.
    def network():
        n = _h2_network()
        n.add("Generator", "reformer", bus="GB_H2", p_nom=50.0, marginal_cost=40.0, carrier="gas_reforming")
        return n

    # Flat path at 1000; day 1's wind (100 MW x 0.7 x 24 h = 1680) lifts the store above it.
    path = _trajectory_file(tmp_path, network().snapshots, 1000.0, 1000.0, 40.0)
    with pytest.raises(ValueError, match="surplus_discount"):
        resolve_seasonal_trajectory_config({"hydrogen_store": {"seasonal_trajectory": {
            "surplus_discount_gbp_per_mwh_h2": 1.0}}})
    n = network()
    gen, su, links, prices, objective = _rolling(n, _market(
        path, end_condition="soft_floor", shortfall_penalty_gbp_per_mwh_h2=100.0,
        surplus_discount_gbp_per_mwh_h2=1.0))
    e = n.stores_t.e["GB_H2_storage"]
    # Day 1: electrolysis from free wind still fills above the path, the reformer adds nothing.
    assert float(e.iloc[23]) == pytest.approx(1000.0 + 1680.0, abs=1.0)
    assert float(gen["reformer"].iloc[:24].sum()) == pytest.approx(0.0, abs=1e-6)
    # Day 2 (no wind; turbines at 39 / 0.5 + 5 = 83 beat gas at 150 above the path): the surplus is burnt
    # down to the path, then the reformer (40 / 0.5 + 5 = 85) keeps the turbines running on the path.
    assert float(e.iloc[-1]) == pytest.approx(1000.0, abs=1.0)
    # Every hydrogen term is objective only: the reported objective is the dispatch cost.
    assert objective == pytest.approx(_dispatch_cost(n, gen, links), rel=1e-6)
    # The discount changes nothing where no supplier ties with the end value.
    n1, n2 = _h2_network(), _h2_network()
    _rolling(n1, _market(path, end_condition="soft_floor", shortfall_penalty_gbp_per_mwh_h2=100.0))
    _, _, _, _, objective2 = _rolling(n2, _market(path, end_condition="soft_floor",
                                                  shortfall_penalty_gbp_per_mwh_h2=100.0,
                                                  surplus_discount_gbp_per_mwh_h2=1.0))
    assert float(n2.stores_t.e["GB_H2_storage"].iloc[-1]) == pytest.approx(
        float(n1.stores_t.e["GB_H2_storage"].iloc[-1]), abs=1.0)
