"""Tests for the solver fallback ladder in scripts/market/market_utils.py.

The ladder is: (1) configured options, (2) conservative barrier numerics,
(3) dual simplex with crossover. A rung is accepted only when the status is
``ok`` and the termination condition is ``optimal``. Success on a later rung
logs a WARNING naming the rung; failure on every rung raises SolveFailedError
carrying the last status and condition.
"""

import logging

import pytest

from scripts.market.market_utils import (
    BARRIER_ONLY_SOLVER_KEYS,
    CONSERVATIVE_BARRIER_OVERRIDES,
    SolveFailedError,
    build_solver_fallback_ladder,
    conservative_barrier_options,
    dual_simplex_options,
    log_fallback_summary,
    solve_with_counterflow_lock,
    solve_with_fallback,
)


FAST_OPTIONS = {
    "threads": 2,
    "method": 2,
    "crossover": 0,
    "BarHomogeneous": 0,
    "BarConvTol": 1e-4,
    "FeasibilityTol": 1e-4,
    "OptimalityTol": 1e-4,
    "NumericFocus": 0,
    "ScaleFlag": -1,
    "DualReductions": 1,
    "BarIterLimit": 200,
}


class _ScriptedOptimize:
    """Stand-in for network.optimize returning scripted (status, condition) pairs."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, network, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


class TestLadder:
    def test_gurobi_ladder_has_three_distinct_rungs(self):
        ladder = build_solver_fallback_ladder("gurobi", FAST_OPTIONS)
        names = [name for name, _ in ladder]
        assert names == ["configured", "conservative_barrier", "dual_simplex"]
        assert ladder[0][1] == FAST_OPTIONS
        assert ladder[0][1] is not FAST_OPTIONS  # a copy, the caller's dict is untouched

    def test_rung_two_adds_conservative_numerics_and_keeps_the_rest(self):
        options = conservative_barrier_options(FAST_OPTIONS)
        for key, value in CONSERVATIVE_BARRIER_OVERRIDES.items():
            assert options[key] == value
        assert options["method"] == 2
        assert options["crossover"] == 0
        assert options["BarConvTol"] == 1e-4
        assert options["threads"] == 2

    def test_rung_three_drops_barrier_only_keys_and_uses_dual_simplex(self):
        options = dual_simplex_options(FAST_OPTIONS)
        for key in BARRIER_ONLY_SOLVER_KEYS:
            assert key not in options
        assert options["method"] == 1
        assert options["crossover"] == -1
        assert options["DualReductions"] == 1
        assert options["NumericFocus"] >= 1
        assert options["threads"] == 2
        assert options["FeasibilityTol"] == 1e-4

    def test_option_keys_are_matched_case_insensitively(self):
        options = dual_simplex_options({"Method": 2, "Crossover": 0, "numericfocus": 3})
        assert options["Method"] == 1 and "method" not in options
        assert options["Crossover"] == -1 and "crossover" not in options
        assert options["numericfocus"] == 3

    def test_already_conservative_config_has_two_rungs(self):
        conservative = conservative_barrier_options(FAST_OPTIONS)
        ladder = build_solver_fallback_ladder("gurobi", conservative)
        assert [name for name, _ in ladder] == ["configured", "dual_simplex"]

    def test_non_gurobi_solver_has_the_single_configured_rung(self):
        ladder = build_solver_fallback_ladder("highs", {"threads": 2})
        assert ladder == [("configured", {"threads": 2})]


class TestSolveWithFallback:
    def test_first_rung_optimal_returns_rung_one_without_warning(self, caplog):
        optimize = _ScriptedOptimize([("ok", "optimal")])
        logger = logging.getLogger("test_fallback_ok")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            outcome = solve_with_fallback(
                object(), "gurobi", FAST_OPTIONS, None, logger, "window 1",
                optimize=optimize,
            )
        assert outcome.rung == 1
        assert outcome.rung_name == "configured"
        assert outcome.status == "ok" and outcome.condition == "optimal"
        assert outcome.solver_options == FAST_OPTIONS
        assert len(optimize.calls) == 1
        assert optimize.calls[0]["solver_options"] == FAST_OPTIONS
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_failure_on_rung_one_succeeds_on_rung_two_and_warns(self, caplog):
        optimize = _ScriptedOptimize([("warning", "infeasible_or_unbounded"), ("ok", "optimal")])
        logger = logging.getLogger("test_fallback_rung2")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            outcome = solve_with_fallback(
                object(), "gurobi", FAST_OPTIONS, None, logger, "BM window 3",
                optimize=optimize,
            )
        assert outcome.rung == 2
        assert outcome.rung_name == "conservative_barrier"
        assert optimize.calls[1]["solver_options"]["NumericFocus"] == 3
        assert optimize.calls[1]["solver_options"]["BarHomogeneous"] == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("rung 2" in r.getMessage() and "conservative_barrier" in r.getMessage() for r in warnings)
        assert all("BM window 3" in r.getMessage() for r in warnings)
        assert len(outcome.attempts) == 2

    def test_iteration_limit_with_status_ok_is_not_accepted(self):
        # linopy maps iteration_limit and suboptimal to status "ok"; the old
        # status == "ok" check accepted them silently.
        optimize = _ScriptedOptimize([("ok", "iteration_limit"), ("ok", "suboptimal"), ("ok", "optimal")])
        outcome = solve_with_fallback(
            object(), "gurobi", FAST_OPTIONS, None, logging.getLogger("t"), "w",
            optimize=optimize,
        )
        assert outcome.rung == 3
        assert outcome.rung_name == "dual_simplex"
        assert "BarHomogeneous" not in optimize.calls[2]["solver_options"]
        assert optimize.calls[2]["solver_options"]["method"] == 1

    def test_all_rungs_fail_raises_with_last_condition(self):
        optimize = _ScriptedOptimize([
            ("warning", "infeasible_or_unbounded"),
            ("warning", "infeasible_or_unbounded"),
            ("warning", "infeasible"),
        ])
        with pytest.raises(SolveFailedError) as excinfo:
            solve_with_fallback(
                object(), "gurobi", FAST_OPTIONS, None, logging.getLogger("t"), "BM window 9",
                optimize=optimize,
            )
        err = excinfo.value
        assert err.status == "warning"
        assert err.condition == "infeasible"
        assert len(err.attempts) == 3
        assert "BM window 9" in str(err)
        assert "dual_simplex" in str(err)
        assert len(optimize.calls) == 3

    def test_non_gurobi_failure_raises_after_one_attempt(self):
        optimize = _ScriptedOptimize([("warning", "infeasible")])
        with pytest.raises(SolveFailedError) as excinfo:
            solve_with_fallback(
                object(), "highs", {"threads": 1}, None, logging.getLogger("t"), "w",
                optimize=optimize,
            )
        assert len(optimize.calls) == 1
        assert excinfo.value.condition == "infeasible"

    def test_extra_functionality_and_kwargs_are_forwarded(self):
        optimize = _ScriptedOptimize([("ok", "optimal")])

        def extra(n, sns):
            return None

        solve_with_fallback(
            object(), "gurobi", FAST_OPTIONS, extra, logging.getLogger("t"), "w",
            optimize=optimize, snapshots=[1, 2, 3],
        )
        call = optimize.calls[0]
        assert call["extra_functionality"] is extra
        assert call["snapshots"] == [1, 2, 3]
        assert call["solver_name"] == "gurobi"


class TestCounterflowLock:
    def test_no_locks_means_a_single_solve(self):
        optimize = _ScriptedOptimize([("ok", "optimal")])
        outcome = solve_with_counterflow_lock(
            object(), "gurobi", FAST_OPTIONS, None, logging.getLogger("t"), "w",
            lock_counterflows=lambda n, logger=None: [], optimize=optimize,
        )
        assert len(optimize.calls) == 1
        assert outcome.rung == 1

    def test_locks_trigger_a_second_solve_and_the_worst_rung_is_reported(self):
        optimize = _ScriptedOptimize([
            ("warning", "infeasible_or_unbounded"), ("ok", "optimal"),  # first solve: rung 2
            ("ok", "optimal"),                                            # re-solve: rung 1
        ])
        outcome = solve_with_counterflow_lock(
            object(), "gurobi", FAST_OPTIONS, None, logging.getLogger("t"), "w",
            lock_counterflows=lambda n, logger=None: ["IC1"], optimize=optimize,
        )
        assert len(optimize.calls) == 3
        assert outcome.rung == 2
        assert outcome.rung_name == "conservative_barrier"
        assert len(outcome.attempts) == 3
        assert outcome.elapsed >= 0.0


def test_fallback_summary_warns_only_when_a_rung_was_used(caplog):
    logger = logging.getLogger("test_fallback_summary")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_fallback_summary(logger, "Rolling BM", [], 31)
        log_fallback_summary(logger, "Rolling BM", [(4, "dual_simplex")], 31)
    levels = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert levels[0][0] == logging.INFO and "no fallback" in levels[0][1]
    assert levels[1][0] == logging.WARNING and "4 (dual_simplex)" in levels[1][1]


class TestSolverIoKwargs:
    def test_default_is_direct_for_gurobi_and_highs(self):
        from scripts.market.market_utils import solver_io_kwargs
        assert solver_io_kwargs({}, "gurobi") == {"io_api": "direct"}
        assert solver_io_kwargs({"solve": {"solver": {}}}, "highs") == {"io_api": "direct"}

    def test_file_interfaces_and_unknown_values(self):
        from scripts.market.market_utils import solver_io_kwargs
        assert solver_io_kwargs({"solve": {"solver": {"io_api": "lp"}}}, "gurobi") == {"io_api": "lp"}
        assert solver_io_kwargs({"solve": {"solver": {"io_api": "MPS"}}}, "gurobi") == {"io_api": "mps"}
        assert solver_io_kwargs({"solve": {"solver": {"io_api": "file"}}}, "gurobi") == {}

    def test_direct_is_dropped_for_solvers_without_a_direct_interface(self):
        from scripts.market.market_utils import solver_io_kwargs
        assert solver_io_kwargs({"solve": {"solver": {"io_api": "direct"}}}, "glpk") == {}


class TestPostSolveBoundCheck:
    """Item C11: every accepted rung is measured against its own bounds; > 50 MW fails the rung."""

    @staticmethod
    def _network():
        import pandas as pd
        import pypsa

        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
        n.add("Bus", ["a", "b"])
        n.add("Generator", "g", bus="a", p_nom=100.0, p_max_pu=1.0)
        n.add("Line", "l", bus0="a", bus1="b", s_nom=50.0, x=0.1, r=0.01)
        n.add("StorageUnit", "s", bus="b", p_nom=10.0, max_hours=2.0)
        return n

    @staticmethod
    def _solution(n, gen_mw, line_mw, soc_mwh=0.0):
        import pandas as pd

        n.generators_t.p = pd.DataFrame({"g": [gen_mw, 50.0]}, index=n.snapshots)
        n.lines_t.p0 = pd.DataFrame({"l": [line_mw, -10.0]}, index=n.snapshots)
        n.storage_units_t.state_of_charge = pd.DataFrame({"s": [soc_mwh, 0.0]}, index=n.snapshots)

    def test_measures_the_largest_excess_over_every_component(self):
        from scripts.market.market_utils import max_bound_violation
        n = self._network()
        self._solution(n, 104.2, 70.0)
        value, where = max_bound_violation(n)
        assert value == pytest.approx(20.0) and where == "Line l"
        self._solution(n, 100.0, -50.0, soc_mwh=23.0)
        value, where = max_bound_violation(n)
        assert value == pytest.approx(3.0) and "state_of_charge" in where
        assert max_bound_violation(object()) != max_bound_violation(object())  # nan: not measurable

    def test_a_violating_rung_is_rejected_and_the_next_accepted(self, caplog):
        n = self._network()
        results = iter([(180.0, 0.0), (100.0, 10.0)])

        def optimize(network, **kwargs):
            gen, line = next(results)
            self._solution(network, gen, line)
            return "ok", "optimal"

        logger = logging.getLogger("test_bound_check")
        with caplog.at_level(logging.INFO, logger=logger.name):
            outcome = solve_with_fallback(n, "gurobi", FAST_OPTIONS, None, logger, "window 7", optimize=optimize)
        assert outcome.rung == 2 and outcome.bound_violation == pytest.approx(0.0)
        assert "bound violation 80.0 MW" in outcome.attempts[0][2]
        messages = [r.getMessage() for r in caplog.records]
        assert any("max post-solve bound violation 80 MW (Generator g)" in m for m in messages)
        assert any("max post-solve bound violation 0 MW" in m for m in messages)

    def test_every_rung_violating_fails_the_window(self):
        n = self._network()

        def optimize(network, **kwargs):
            self._solution(network, 100.0, 120.0)
            return "ok", "optimal"

        with pytest.raises(SolveFailedError, match="bound violation 70.0 MW"):
            solve_with_fallback(n, "gurobi", FAST_OPTIONS, None, logging.getLogger("t"), "w", optimize=optimize)
