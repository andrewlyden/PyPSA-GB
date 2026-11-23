"""Hydrogen store end-of-window value in the Stage 1 rolling windows (user decision 27 Sep 2026).

A myopic 24 h window put no value on hydrogen left at its end, so Stage 1 ran
the store down and electrolysed only the day's own needs even with curtailed
wind. ``market.hydrogen_store.end_of_window_value_gbp_per_mwh_h2`` credits each
carried store's level at the window's last snapshot, objective only.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest
import yaml

from scripts.market.market_utils import (
    build_end_of_window_store_value_callback,
    prepare_carried_stores,
    resolve_store_level_config,
    store_end_of_window_credit,
)
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_hydrogen_end_of_window_value")
KEY = "end_of_window_value_gbp_per_mwh_h2"


def _network(hours=48, windy_day=1):
    """Surplus wind on one day (day 1 by default), 150 GBP gas on the other; electrolyser, H2 turbine, store."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    n.add("Bus", "GB_H2", carrier="H2")
    wind = [200.0 if (i < 24) == (windy_day == 1) else 0.0 for i in range(hours)]
    n.add("Generator", "wind", bus="el", p_nom=1.0, p_max_pu=pd.Series(wind, index=n.snapshots),
          marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=150.0, carrier="CCGT")
    n.add("Load", "load", bus="el", p_set=100.0)
    n.add("Link", "electrolysis", bus0="el", bus1="GB_H2", p_nom=100.0, efficiency=0.7, carrier="electrolysis")
    n.add("Link", "h2_turbine", bus0="GB_H2", bus1="el", p_nom=200.0, efficiency=0.5,
          marginal_cost=5.0, carrier="H2_turbine")
    n.add("Store", "GB_H2_storage", bus="GB_H2", carrier="H2_gas", e_nom=4000.0, e_cyclic=True,
          standing_loss=0.0)
    return n


def _market(value=None, month_end="free"):
    block = {"month_end": month_end}
    if value is not None:
        block[KEY] = value
    return {"hydrogen_store": block}


def _solve_rolling(n, market_config):
    """The real Stage 1 rolling loop on the toy network (HiGHS)."""
    from scripts.market.solve_wholesale import solve_rolling_day_ahead

    prepare_carried_stores(n, market_config, LOG)
    gen, su, links, prices, objective = solve_rolling_day_ahead(
        n, {"window_hours": 24, "look_ahead_hours": 0}, "highs", {}, {"market": market_config}, LOG,
    )
    return gen, links, objective


def _dispatch_cost(n, gen, links):
    """What the dispatch costs: generator and link marginal costs times their flows."""
    cost = float((gen * n.generators["marginal_cost"].reindex(gen.columns)).sum().sum())
    return cost + float((links * n.links["marginal_cost"].reindex(links.columns)).sum().sum())


# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_default_is_off_and_matches_defaults_yaml():
    defaults = load_defaults()
    assert defaults["market"]["hydrogen_store"][KEY] is None
    assert resolve_store_level_config({})[KEY] is None
    start = prepare_carried_stores(_network(), {}, LOG)
    assert build_end_of_window_store_value_callback({}, start, LOG) is None
    with pytest.raises(ValueError):
        resolve_store_level_config({"hydrogen_store": {KEY: -1.0}})


def test_cp30_scenarios_carry_the_value_and_keep_the_store_rules():
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenario = load_config()["scenarios"]["CP30_FFR_2030_Dec_cp30_network"]
    cfg = resolve_store_level_config(scenario["market"])
    # 1.2 x CP30 gas (28 Sep 2026), 41.2 GBP/MWh HHV, in the GB_H2 system's LHV units (OW3, 3 Oct 2026).
    assert cfg[KEY] == pytest.approx(round(41.2 * 141.88 / 119.96, 2)) == pytest.approx(48.73)
    assert cfg["carry_level"] and cfg["month_end"] == "at_least_initial" and cfg["carriers"] == ("H2_gas",)


def test_no_callback_when_nothing_is_carried():
    n = _network()
    market = {"hydrogen_store": {"carry_level": False, KEY: 55.0}}
    start = prepare_carried_stores(n, market, LOG)
    assert start.empty
    assert build_end_of_window_store_value_callback(market, start, LOG) is None


# â”€â”€ The objective term â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_the_credit_sits_on_the_last_snapshot_only():
    n = _network(hours=24)
    start = prepare_carried_stores(n, _market(55.0), LOG)
    callback = build_end_of_window_store_value_callback(_market(55.0), start, LOG)
    model = n.optimize.create_model()

    def store_coefficients():
        expr = model.objective.expression
        terms = pd.Series(expr.coeffs.values.ravel(), index=expr.vars.values.ravel())
        labels = set(int(v) for v in model.variables["Store-e"].labels.values.ravel())
        return terms[terms.index.isin(labels)].groupby(level=0).sum()

    before = store_coefficients()
    callback(n, n.snapshots)
    delta = store_coefficients().sub(before, fill_value=0.0)
    delta = delta[delta.abs() > 0]
    last = int(model.variables["Store-e"].labels.sel(snapshot=n.snapshots[-1]).values.ravel()[0])
    assert list(delta.index) == [last]
    assert delta.iloc[0] == pytest.approx(-55.0)


def test_windy_day_banks_hydrogen_only_with_the_credit_and_gas_never_does():
    n = _network()
    _, links_off, _ = _solve_rolling(n, _market())
    made_off = float(links_off["electrolysis"].iloc[:24].sum())
    assert made_off == pytest.approx(0.0, abs=1e-3)           # myopic day 1 leaves the surplus curtailed

    n = _network()
    _, links_on, _ = _solve_rolling(n, _market(55.0))
    assert float(links_on["electrolysis"].iloc[:24].sum()) == pytest.approx(2400.0, rel=1e-4)   # full 100 MW
    assert float(links_on["electrolysis"].iloc[24:].sum()) == pytest.approx(0.0, abs=1e-3)      # 150 > 0.7 x 55
    # 55 / 0.5 + 5 = 115 < 150: the still day burns the banked hydrogen against gas.
    assert float(links_on["h2_turbine"].iloc[24:].sum()) > 1000.0


def test_reported_objective_is_the_dispatch_cost_with_the_credit_stripped():
    n = _network()
    gen, links, objective = _solve_rolling(n, _market(55.0))
    assert objective == pytest.approx(_dispatch_cost(n, gen, links), rel=1e-6)
    n = _network()
    gen, links, objective = _solve_rolling(n, _market())
    assert objective == pytest.approx(_dispatch_cost(n, gen, links), rel=1e-6)


def test_store_end_of_window_credit_reads_the_last_level():
    n = _network(hours=24)
    start = prepare_carried_stores(n, _market(55.0), LOG)
    callback = build_end_of_window_store_value_callback(_market(55.0), start, LOG)
    status, condition = n.optimize(solver_name="highs", extra_functionality=callback)
    assert status == "ok", condition
    mwh, gbp = store_end_of_window_credit(n, callback)
    assert mwh == pytest.approx(float(n.stores_t.e["GB_H2_storage"].iloc[-1]))
    assert gbp == pytest.approx(55.0 * mwh)
    assert store_end_of_window_credit(n, None) == (0.0, 0.0)


# â”€â”€ Coexistence with the month-end rule â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_month_end_floor_still_binds_under_the_credit():
    # Day 2 (the month's last window) burns banked hydrogen worth more against
    # gas than V; the at_least_initial floor still stops it at the run's start.
    n = _network()
    _, links, _ = _solve_rolling(n, _market(55.0, month_end="at_least_initial"))
    e = n.stores_t.e["GB_H2_storage"]
    assert float(e.iloc[23]) == pytest.approx(2000.0 + 0.7 * 2400.0, rel=1e-4)
    assert float(e.iloc[-1]) == pytest.approx(2000.0, abs=1e-2)


def test_credit_on_a_capped_month_end_window_stays_feasible():
    # Still day 1 burns down, windy last day: the floor is capped at what 1 MW of
    # electrolysis can make; the credit adds on top and the window still solves.
    n = _network(windy_day=2)
    n.links.loc["electrolysis", "p_nom"] = 1.0
    _, links, _ = _solve_rolling(n, _market(55.0, month_end="at_least_initial"))
    e = n.stores_t.e["GB_H2_storage"]
    assert float(e.iloc[-1]) == pytest.approx(float(e.iloc[23]) + 0.7 * 24.0, rel=1e-4)
