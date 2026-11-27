"""Thermal ramp limits: per-carrier limits (nuclear included) and the hourly widening beside a floor.

The nuclear cases follow Luke Bennett's tests for load-following nuclear (PR #50), on the model's existing keys.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.generators.thermal_ramping import (
    apply_thermal_ramp_limits,
    widen_ramp_limits_to_availability,
)

LOG = logging.getLogger(__name__)
NUCLEAR_RAMP = {"enabled": True, "carriers": {"nuclear": {"ramp_limit_up": 0.025, "ramp_limit_down": 0.025}}}


@pytest.fixture
def fleet() -> pypsa.Network:
    """Two nuclear units (one with a capitalised carrier) and a gas unit with its own limits."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2035-01-01", periods=6, freq="h"))
    n.add("Bus", "bus")
    n.add("Generator", "nuclear_extendable", bus="bus", carrier="nuclear", p_nom=1000.0,
          p_nom_extendable=True, committable=False, p_min_pu=0.0)
    n.add("Generator", "nuclear_committable", bus="bus", carrier="Nuclear", p_nom=1000.0,
          committable=True, p_min_pu=0.0)
    n.add("Generator", "gas", bus="bus", carrier="gas", p_nom=1000.0, p_min_pu=0.1,
          ramp_limit_up=0.5, ramp_limit_down=0.4)
    return n


def test_nuclear_takes_its_ramp_limit_whatever_the_carrier_case(fleet):
    assert apply_thermal_ramp_limits(fleet, NUCLEAR_RAMP, LOG) == 2
    for name in ("nuclear_extendable", "nuclear_committable"):
        assert fleet.generators.at[name, "ramp_limit_up"] == pytest.approx(0.025)
        assert fleet.generators.at[name, "ramp_limit_down"] == pytest.approx(0.025)


def test_other_generators_and_model_choices_unchanged(fleet):
    gas = fleet.generators.loc["gas"].copy()
    extendable = fleet.generators["p_nom_extendable"].copy()
    committable = fleet.generators["committable"].copy()
    apply_thermal_ramp_limits(fleet, NUCLEAR_RAMP, LOG)
    pd.testing.assert_series_equal(fleet.generators.loc["gas"], gas)
    pd.testing.assert_series_equal(fleet.generators["p_nom_extendable"], extendable)
    pd.testing.assert_series_equal(fleet.generators["committable"], committable)


@pytest.mark.parametrize("up, down", [(0.0, 0.1), (0.1, 0.0), (1.01, 0.1), (0.1, 1.01), (-0.1, 0.1)])
def test_invalid_ramp_limits_refused(fleet, up, down):
    config = {"enabled": True, "carriers": {"nuclear": {"ramp_limit_up": up, "ramp_limit_down": down}}}
    with pytest.raises(ValueError):
        apply_thermal_ramp_limits(fleet, config, LOG)


def test_disabled_ramping_changes_nothing(fleet):
    before = fleet.generators.copy()
    assert apply_thermal_ramp_limits(fleet, {**NUCLEAR_RAMP, "enabled": False}, LOG) == 0
    pd.testing.assert_frame_equal(fleet.generators, before)


def _outage_network(widen: bool) -> pypsa.Network:
    """Nuclear with a 20% floor and a 2.5%/h ramp limit through an outage (availability 1 -> 0 -> 1)."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2035-01-01", periods=8, freq="h"))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=500.0)
    n.add("Generator", "nuclear", bus="bus", carrier="nuclear", p_nom=1000.0, marginal_cost=10.0,
          p_min_pu=0.2, ramp_limit_up=0.025, ramp_limit_down=0.025,
          p_max_pu=pd.Series([1, 1, 1, 0, 0, 1, 1, 1], index=n.snapshots, dtype=float))
    n.add("Generator", "backup", bus="bus", carrier="gas", p_nom=2000.0, marginal_cost=100.0)
    if widen:
        widen_ramp_limits_to_availability(n, LOG)
    return n


def test_widened_limits_follow_the_outage_and_hold_elsewhere():
    n = _outage_network(widen=True)
    down = n.generators_t.ramp_limit_down["nuclear"]
    up = n.generators_t.ramp_limit_up["nuclear"]
    # Availability falls 1 -> 0 into hour 3: the floor clipped to availability falls 0.2 -> 0, then rises at hour 5.
    np.testing.assert_allclose(down.to_numpy(), [0.025, 0.025, 0.025, 1.0, 0.025, 0.025, 0.025, 0.025])
    np.testing.assert_allclose(up.to_numpy(), [0.025, 0.025, 0.025, 0.025, 0.025, 0.2, 0.025, 0.025])


def test_outage_with_floor_and_ramp_limit_solves_only_when_widened():
    status, _ = _outage_network(widen=False).optimize(solver_name="highs")
    assert status != "ok"
    n = _outage_network(widen=True)
    status, condition = n.optimize(solver_name="highs")
    assert (status, condition) == ("ok", "optimal")
    p = n.generators_t.p["nuclear"] / 1000.0
    assert p.iloc[3] == pytest.approx(0.0) and p.iloc[4] == pytest.approx(0.0)
    steps = p.diff().iloc[1:]
    limit = np.maximum(n.generators_t.ramp_limit_up["nuclear"], n.generators_t.ramp_limit_down["nuclear"]).iloc[1:]
    assert (steps.abs() <= limit + 1e-6).all()


def test_units_without_a_floor_or_committable_keep_static_limits(fleet):
    apply_thermal_ramp_limits(fleet, NUCLEAR_RAMP, LOG)
    fleet.generators_t.p_max_pu = pd.DataFrame(
        {name: [1, 1, 0, 0, 1, 1] for name in fleet.generators.index}, index=fleet.snapshots, dtype=float)
    fleet.generators.at["nuclear_committable", "p_min_pu"] = 0.2  # committable: PyPSA's commitment constraints
    assert widen_ramp_limits_to_availability(fleet, LOG) == 1     # gas only: floor 0.1, own limits
    assert "nuclear_extendable" not in fleet.generators_t.ramp_limit_down.columns
    assert "nuclear_committable" not in fleet.generators_t.ramp_limit_down.columns
    assert fleet.generators_t.ramp_limit_down["gas"].max() == pytest.approx(1.0)
    assert fleet.generators_t.ramp_limit_down["gas"].min() == pytest.approx(0.4)
