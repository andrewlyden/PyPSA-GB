"""market.hydrogen_offtake_scale (Run B, 17 Oct 2026; scripts/market/market_utils.apply_hydrogen_offtake_scale)."""
import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import HYDROGEN_OFFTAKE_CARRIER, apply_hydrogen_offtake_scale

LOG = logging.getLogger("test")


def _network(timed=False):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=3, freq="h"))
    n.add("Bus", "GB"); n.add("Bus", "GB_H2", carrier="H2")
    n.add("Load", "GB_H2_exogenous_demand", bus="GB_H2", carrier=HYDROGEN_OFFTAKE_CARRIER, p_set=1187.65)
    n.add("Load", "demand", bus="GB", carrier="electricity", p_set=pd.Series([10.0, 20.0, 30.0], index=n.snapshots))
    if timed:
        n.loads_t.p_set["GB_H2_exogenous_demand"] = [100.0, 200.0, 300.0]
    return n


def test_absent_is_a_no_op():
    n = _network()
    assert apply_hydrogen_offtake_scale(n, {}, LOG) == 0
    assert apply_hydrogen_offtake_scale(n, {"hydrogen_offtake_scale": None}, LOG) == 0
    assert n.loads.at["GB_H2_exogenous_demand", "p_set"] == pytest.approx(1187.65)


def test_scales_the_offtake_only():
    n = _network(timed=True)
    assert apply_hydrogen_offtake_scale(n, {"hydrogen_offtake_scale": 0.9353}, LOG) == 1
    assert n.loads.at["GB_H2_exogenous_demand", "p_set"] == pytest.approx(1187.65 * 0.9353)
    assert n.loads_t.p_set["GB_H2_exogenous_demand"].tolist() == pytest.approx([93.53, 187.06, 280.59])
    assert n.loads_t.p_set["demand"].tolist() == [10.0, 20.0, 30.0]


def test_refuses_bad_values_and_a_network_without_offtake():
    with pytest.raises(ValueError):
        apply_hydrogen_offtake_scale(_network(), {"hydrogen_offtake_scale": -0.1}, LOG)
    n = _network(); n.remove("Load", "GB_H2_exogenous_demand")
    with pytest.raises(ValueError, match="no hydrogen offtake"):
        apply_hydrogen_offtake_scale(n, {"hydrogen_offtake_scale": 0.9}, LOG)
