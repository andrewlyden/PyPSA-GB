"""Large hydro hands back the water it was lent (hydro.large_hydro.month_end).

Each market month started the reservoir at initial_storage_hours x p_nom and
nothing asked for it back, so Stage 1 dispatch ran 26-30% above inflow every
month (~1 TWh/yr of free water). With ``at_least_initial`` the run ends with at
least the store it started with; a rolling solve holds each window at or above
the lowest store from which the rest of the run can still refill it.
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.solve.hydro_constraints import (
    build_hydro_constraints_callback,
    initialise_large_hydro_storage_state,
    large_hydro_end_floor,
    large_hydro_net_inflow,
    resolve_large_hydro_month_end,
    update_large_hydro_storage_state,
)

LOGGER = logging.getLogger("test_large_hydro_month_end")
P_NOM = 100.0
INFLOW_FACTOR = 0.55          # 55 MW of inflow an hour at p_max_pu 1
INITIAL_HOURS = 1.0           # 100 MWh lent at the start


def _config(month_end="at_least_initial"):
    return {"renewables": {"hydro": {"enabled": True, "large_hydro": {
        "reservoir_model": True,
        "inflow_profile_factor": INFLOW_FACTOR,
        "initial_storage_hours": INITIAL_HOURS,
        "reservoir_capacity_hours": 168.0,
        "month_end": month_end,
    }}}}


def _network(hours=4):
    """Free hydro (100 MW) against dear gas serving a flat 100 MW load."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Carrier", "large_hydro")
    n.add("Carrier", "CCGT")
    n.add("Bus", "A")
    n.add("Generator", "hydro", bus="A", p_nom=P_NOM, carrier="large_hydro", marginal_cost=0.0)
    n.add("Generator", "gas", bus="A", p_nom=200.0, carrier="CCGT", marginal_cost=100.0)
    n.add("Load", "load", bus="A", p_set=100.0)
    return n


def _solve(n, config, **kwargs):
    callback = build_hydro_constraints_callback(n, config, logger=LOGGER, **kwargs)
    status, condition = n.optimize(solver_name="highs", extra_functionality=callback)
    assert status == "ok", condition
    return float(n.generators_t.p["hydro"].sum())


@pytest.mark.unit
class TestLargeHydroMonthEnd:

    def test_default_is_at_least_initial_and_bad_values_are_refused(self):
        assert resolve_large_hydro_month_end({}) == "at_least_initial"
        with pytest.raises(ValueError, match="month_end"):
            resolve_large_hydro_month_end({"large_hydro": {"month_end": "sometimes"}})

    def test_free_mode_spends_the_lent_buffer(self):
        n = _network()
        assert _solve(n, _config("free")) == pytest.approx(100.0 + 4 * 55.0)

    def test_single_solve_hands_the_buffer_back(self):
        n, config = _network(), _config()
        initial = initialise_large_hydro_storage_state(n, config["renewables"]["hydro"])
        floor = large_hydro_end_floor(
            large_hydro_net_inflow(n, config["renewables"]["hydro"]), initial, n.snapshots[-1]
        )
        assert floor["hydro"] == pytest.approx(100.0)
        # dispatch is at most the run's inflow
        assert _solve(n, config, large_hydro_end_storage=floor) == pytest.approx(4 * 55.0)

    def test_rolling_windows_use_the_buffer_early_and_return_it(self):
        """Window 1 may draw the buffer (its floor, 100 - 110 of later inflow, is below
        zero); window 2 must refill it, so the run's dispatch equals its inflow and
        neither window is infeasible."""
        full, config = _network(), _config()
        hydro_cfg = config["renewables"]["hydro"]
        state = initialise_large_hydro_storage_state(full, hydro_cfg)
        run_initial = state.copy()
        net_inflow = large_hydro_net_inflow(full, hydro_cfg)
        dispatched = []
        for window in (full.snapshots[:2], full.snapshots[2:]):
            n = full.copy()
            n.set_snapshots(window)
            dispatched.append(_solve(
                n, config, large_hydro_storage_state=state,
                large_hydro_end_storage=large_hydro_end_floor(net_inflow, run_initial, window[-1]),
            ))
            state = update_large_hydro_storage_state(n, window, config, state)
        assert dispatched[0] == pytest.approx(200.0)       # buffer spent on day one
        assert dispatched[1] == pytest.approx(20.0)        # and repaid on day two
        assert sum(dispatched) == pytest.approx(4 * 55.0)
        assert state["hydro"] == pytest.approx(run_initial["hydro"])

    def test_an_unreachable_floor_is_capped_not_infeasible(self):
        """A window that starts below what it could refill ends as full as it can."""
        n, config = _network(hours=2), _config()
        state = pd.Series({"hydro": 0.0})
        dispatched = _solve(
            n, config, large_hydro_storage_state=state,
            large_hydro_end_storage=pd.Series({"hydro": 500.0}),
        )
        assert dispatched == pytest.approx(0.0, abs=1e-6)

    def test_the_floor_respects_minimum_output(self):
        """With a 20% floor the run can only repay inflow net of minimum output."""
        n, config = _network(), _config()
        n.generators.loc["hydro", "p_min_pu"] = 0.2
        net = large_hydro_net_inflow(n, config["renewables"]["hydro"])
        assert net["hydro"].tolist() == pytest.approx([55.0 - 20.0] * 4)
