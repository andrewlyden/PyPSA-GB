"""Thermal ramp limits per carrier, nuclear included.

The nuclear cases follow Luke Bennett's tests for load-following nuclear (PR #50), on the model's existing keys.
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.generators.thermal_ramping import apply_thermal_ramp_limits

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
