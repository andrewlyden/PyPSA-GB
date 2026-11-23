"""Exact branch-constraint screening for a central dispatch Stage 1 (18 Oct 2026).

``market.wholesale.central_constraint_screening`` solves each rolling window with
the thermal limits on a monitored subset of lines and transformers, checks every
branch on every snapshot (look-ahead included) and re-solves with the violated
ones added. The answer must be the full LP's; off (the default) must be the
rolling loop exactly as before.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market import central_constraint_screening as ccs
from scripts.market.central_constraint_screening import (
    CentralConstraintScreen,
    branch_excess_and_loading,
    build_central_constraint_screen,
    mask_unmonitored_branch_limits,
    resolve_screening_config,
)
from scripts.market.market_utils import solve_with_fallback
from scripts.market.solve_wholesale import solve_rolling_day_ahead

LOG = logging.getLogger("test_central_constraint_screening")


def _meshed_network(hours=48):
    """Six-bus meshed AC network plus a transformer; cheap supply in the north, load in the south.

    Line ``n1_s1`` and the transformer bind when the wind is up; ``n2_s2`` binds in
    the evening peak; the other branches never bind.
    """
    n = pypsa.Network()
    sns = pd.date_range("2030-07-01", periods=hours, freq="h")
    n.set_snapshots(sns)
    for bus in ("n1", "n2", "m1", "s1", "s2", "s3"):
        n.add("Bus", bus, carrier="AC", v_nom=400.0)
    n.add("Bus", "s3_lv", carrier="AC", v_nom=132.0)
    lines = [
        ("n1_n2", "n1", "n2", 0.10, 2000.0),
        ("n1_m1", "n1", "m1", 0.20, 2000.0),
        ("n2_m1", "n2", "m1", 0.15, 2000.0),
        ("n1_s1", "n1", "s1", 0.30, 280.0),
        ("n2_s2", "n2", "s2", 0.25, 330.0),
        ("m1_s1", "m1", "s1", 0.20, 2000.0),
        ("m1_s2", "m1", "s2", 0.20, 2000.0),
        ("s1_s2", "s1", "s2", 0.10, 2000.0),
        ("s2_s3", "s2", "s3", 0.10, 2000.0),
    ]
    for name, b0, b1, x, s_nom in lines:
        n.add("Line", name, bus0=b0, bus1=b1, x=x, r=0.0, s_nom=s_nom)
    n.add("Transformer", "s3_t", bus0="s3", bus1="s3_lv", x=0.1, r=0.0, s_nom=180.0)
    hour = np.arange(hours) % 24
    wind = pd.Series(0.35 + 0.6 * (np.sin(np.arange(hours) / 7.0) ** 2), index=sns)
    n.add("Generator", "wind_n1", bus="n1", p_nom=900.0, p_max_pu=wind, marginal_cost=0.0, carrier="wind")
    n.add("Generator", "wind_n2", bus="n2", p_nom=500.0, p_max_pu=wind.shift(5).fillna(0.5),
          marginal_cost=0.5, carrier="wind")
    # Costs rise slightly hour by hour so no two hours tie: the LP has one optimum, and
    # the rolling chain (carried state of charge) cannot drift between equal-cost optima.
    step = pd.Series(np.arange(hours, dtype=float), index=sns)
    n.add("Generator", "ccgt_m1", bus="m1", p_nom=900.0, marginal_cost=60.0 + 0.013 * step, carrier="CCGT")
    n.add("Generator", "ccgt_s2", bus="s2", p_nom=900.0, marginal_cost=85.0 + 0.017 * step, carrier="CCGT")
    n.add("Generator", "ocgt_s3", bus="s3_lv", p_nom=400.0, marginal_cost=140.0 + 0.011 * step,
          carrier="OCGT")
    peak = pd.Series(np.where((hour >= 17) & (hour <= 20), 1.35, 1.0), index=sns)
    for bus, base in (("s1", 350.0), ("s2", 450.0), ("s3", 200.0), ("s3_lv", 260.0), ("m1", 120.0)):
        n.add("Load", f"load_{bus}", bus=bus, p_set=base * peak)
    n.add("StorageUnit", "battery_s1", bus="s1", p_nom=150.0, max_hours=2.0, efficiency_store=0.92,
          efficiency_dispatch=0.92, marginal_cost=1.0, state_of_charge_initial=0.0,
          cyclic_state_of_charge=False, carrier="Battery")
    return n


def _wholesale(screening=None, look_ahead=True):
    config = {"window_hours": 24, "carry_soc": True, "copperplate_mode": "none"}
    if look_ahead:
        config.update({"look_ahead_hours": 24, "look_ahead_resolution_hours": 6})
    if screening is not None:
        config["central_constraint_screening"] = screening
    return config


def _solve(n, wholesale):
    return solve_rolling_day_ahead(n, wholesale, "highs", {}, {"market": {}}, LOG)


def _full_branch_limits_hold(n, tol=1e-3):
    for component in ("Line", "Transformer"):
        excess, _ = branch_excess_and_loading(n, component)
        assert float(excess.max()) <= tol, f"{component} over its limit by {excess.max():.4f} MW"


# ── Config ──────────────────────────────────────────────────────────────────

def test_off_by_default_and_validated():
    assert resolve_screening_config({}) is None
    assert resolve_screening_config({"central_constraint_screening": {"enabled": False}}) is None
    on = resolve_screening_config({"central_constraint_screening": {"enabled": True}})
    assert on["max_iterations"] == ccs.DEFAULT_CONFIG["max_iterations"]
    with pytest.raises(ValueError):
        resolve_screening_config({"central_constraint_screening": {"enabled": True, "typo": 1}})
    with pytest.raises(ValueError):
        resolve_screening_config({"central_constraint_screening": {"enabled": True, "initial_set": "x"}})
    with pytest.raises(ValueError):
        resolve_screening_config({"central_constraint_screening": {"enabled": True, "max_iterations": 0}})
    # Only a central dispatch (copperplate_mode none) is screened.
    n = _meshed_network(24)
    block = {"central_constraint_screening": {"enabled": True}}
    assert build_central_constraint_screen(n, {**block, "copperplate_mode": "collapse"}, LOG) is None
    assert build_central_constraint_screen(n, block, LOG) is None  # default mode is collapse
    assert build_central_constraint_screen(n, {**block, "copperplate_mode": "none"}, LOG) is not None
    assert not ccs.DEFAULT_CONFIG["enabled"]


def test_mask_keeps_only_the_monitored_branch_limits():
    n = _meshed_network(48)
    kept = {}

    def extra(net, sns):
        kept.update(mask_unmonitored_branch_limits(net.model, {"Line": {"n1_s1"}, "Transformer": set()}))

    n.optimize(solver_name="highs", extra_functionality=extra)
    assert kept == {"Line": 1, "Transformer": 0}
    # Every other branch was free, so the relaxed LP overloads at least one of them.
    excess_lines, _ = branch_excess_and_loading(n, "Line")
    excess_trafo, _ = branch_excess_and_loading(n, "Transformer")
    assert excess_lines["n1_s1"] <= 1e-6
    assert max(excess_lines.drop("n1_s1").max(), excess_trafo.max()) > 1.0


# ── Same answer ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("initial_set", ["previous_windows", "empty"])
def test_screened_equals_full_with_look_ahead(initial_set):
    full_net = _meshed_network()
    gen_f, su_f, links_f, prices_f, obj_f = _solve(full_net, _wholesale())
    full_flows = full_net.lines_t.p0.copy()
    full_trafo = full_net.transformers_t.p0.copy()
    # The case only tests something if some branch binds on the full LP.
    _, loading = branch_excess_and_loading(full_net, "Line")
    assert loading.max() > 0.999

    screened_net = _meshed_network()
    gen_s, su_s, links_s, prices_s, obj_s = _solve(
        screened_net, _wholesale({"enabled": True, "initial_set": initial_set, "tolerance_mw": 1e-3}))
    assert obj_s == pytest.approx(obj_f, rel=1e-7, abs=1e-3)
    pd.testing.assert_frame_equal(gen_s, gen_f, atol=1e-3, rtol=0, check_freq=False)
    pd.testing.assert_frame_equal(su_s, su_f, atol=1e-3, rtol=0, check_freq=False)
    pd.testing.assert_frame_equal(screened_net.lines_t.p0, full_flows, atol=1e-3, rtol=0, check_freq=False)
    pd.testing.assert_frame_equal(screened_net.transformers_t.p0, full_trafo, atol=1e-3, rtol=0,
                                  check_freq=False)
    pd.testing.assert_frame_equal(prices_s, prices_f, atol=1e-3, rtol=0, check_freq=False)
    _full_branch_limits_hold(screened_net)
    summary = screened_net.meta["central_constraint_screening"]
    assert summary["windows"] == 2
    assert summary["solves"] >= 3            # at least one window had to add a branch
    assert summary["monitored_final"] < summary["branches"]
    assert summary["full_solves"] == 0


def test_iteration_cap_falls_back_to_every_limit():
    full_net = _meshed_network()
    *_, obj_f = _solve(full_net, _wholesale())
    capped = _meshed_network()
    *_, obj_c = _solve(capped, _wholesale({"enabled": True, "initial_set": "empty", "max_iterations": 1,
                                           "seed_loading_threshold": 1.0, "add_loading_threshold": 1.0}))
    assert obj_c == pytest.approx(obj_f, rel=1e-7, abs=1e-3)
    assert capped.meta["central_constraint_screening"]["full_solves"] >= 1
    _full_branch_limits_hold(capped)


def test_a_violation_only_in_the_look_ahead_is_caught():
    """The check covers every snapshot the window optimises, not only the kept hours."""
    n = _meshed_network(4)
    n.loads_t.p_set.loc[n.snapshots[:3], :] *= 0.2   # nothing binds before the last snapshot
    screen = CentralConstraintScreen(n, {**ccs.DEFAULT_CONFIG, "enabled": True, "tolerance_mw": 1e-3,
                                         "add_loading_threshold": 1.0}, LOG)
    screen.start_window("w")
    optimize = screen.wrap(lambda net, **kw: net.optimize(**kw))
    status, condition = optimize(n, solver_name="highs")
    assert (status, condition) == ("ok", "optimal")
    assert screen.iterations >= 2
    assert screen.sticky["Line"] | screen.sticky["Transformer"]
    _full_branch_limits_hold(n)


# ── Ladder interaction and the default path ─────────────────────────────────

def test_an_unaccepted_iteration_goes_back_to_the_ladder_with_the_branches_found():
    n = _meshed_network(3)
    screen = CentralConstraintScreen(n, {**ccs.DEFAULT_CONFIG, "enabled": True, "initial_set": "empty"},
                                     LOG, accepted_conditions=("optimal",))
    screen.start_window("w")
    calls = []

    def flaky(net, **kwargs):
        calls.append(kwargs["solver_options"].get("rung"))
        if len(calls) == 2:          # the second screened solve fails numerically on rung 1
            return "warning", "other"
        kwargs.pop("solver_options")
        kwargs["solver_name"] = "highs"   # the ladder is Gurobi's; the solve itself is HiGHS
        return net.optimize(**kwargs)

    outcome = solve_with_fallback(n, "gurobi", {"rung": 1}, None, LOG, "test", optimize=screen.wrap(flaky))
    assert outcome.rung == 2
    assert outcome.condition == "optimal"
    _full_branch_limits_hold(n)
    # Rung 2 started from the branches rung 1 had already added.
    assert len(calls) >= 3 and screen.added >= 1


def test_disabled_is_the_existing_code_path(monkeypatch):
    """No key, or enabled false, never imports the screen and gives the same answer as before."""
    called = []
    monkeypatch.setattr(ccs, "build_central_constraint_screen",
                        lambda *a, **k: called.append(1))
    a = _meshed_network()
    out_a = _solve(a, _wholesale())
    b = _meshed_network()
    out_b = _solve(b, _wholesale({"enabled": False, "max_iterations": 3}))
    assert called == []
    assert out_a[4] == out_b[4]
    pd.testing.assert_frame_equal(out_a[0], out_b[0])
    assert "central_constraint_screening" not in (a.meta or {})
    assert "central_constraint_screening" not in (b.meta or {})
