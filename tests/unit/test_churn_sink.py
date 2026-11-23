"""market.wholesale.churn_sink (11 Oct 2026): CP30-style surplus absorption without free-cycling storage."""
import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import CHURN_SINK_CARRIER, apply_churn_sink, fix_churn_sink

LOG = logging.getLogger("test_churn_sink")
CFG = {"churn_sink": {"enabled": True, "share_of_storage_power": 0.5},
       "renewable_curtailment_order": {"ar4_plus_bid_gbp_per_mwh": -10.0}}


def _network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-04-01", periods=3, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", "b", carrier="AC")
    # almost no energy capacity, so it cannot carry the surplus into the deficit hour itself
    n.add("StorageUnit", "bat", bus="b", carrier="Battery", p_nom=100.0, max_hours=0.001,
          efficiency_store=0.92, efficiency_dispatch=0.92, marginal_cost=30.0)
    # AR4+ wind bids -10 in the market; the load is too small to take it all
    n.add("Generator", "wind", bus="b", carrier="wind_offshore", p_nom=200.0, marginal_cost=-10.0)
    n.add("Generator", "gas", bus="b", carrier="CCGT", p_nom=200.0, marginal_cost=100.0)
    n.add("Load", "load", bus="b", p_set=pd.Series([150.0, 150.0, 250.0], index=n.snapshots))
    return n


def test_sink_absorbs_only_surplus_that_would_be_curtailed():
    n = _network()
    assert apply_churn_sink(n, CFG, LOG) == 1
    sink = n.generators.loc["churn_sink b"]
    assert sink.carrier == CHURN_SINK_CARRIER and sink.p_nom == pytest.approx(50.0)
    assert sink.marginal_cost == pytest.approx(-9.99)
    n.optimize(solver_name="highs")
    p = n.generators_t.p["churn_sink b"]
    # surplus hours: wind 200 against load 150 -> the sink takes 50 before any wind is curtailed
    assert p.iloc[0] == pytest.approx(-50.0, abs=0.5) and p.iloc[1] == pytest.approx(-50.0, abs=0.5)
    # a deficit hour: the sink is idle
    assert p.iloc[2] == pytest.approx(0.0, abs=1e-6)


def test_disabled_sink_changes_nothing():
    n = _network()
    assert apply_churn_sink(n, {"churn_sink": {"enabled": False, "share_of_storage_power": 0.5}}, LOG) == 0
    assert apply_churn_sink(n, {}, LOG) == 0
    assert CHURN_SINK_CARRIER not in set(n.generators.carrier)


def test_balancing_stage_holds_the_sink_at_stage1():
    n = _network()
    ws = pd.DataFrame({"wind": [200.0, 200.0, 200.0], "gas": [0.0, 0.0, 50.0],
                       "churn_sink b": [-50.0, -20.0, 0.0]}, index=n.snapshots)
    out = fix_churn_sink(n, CFG, ws, LOG)
    assert list(out.columns) == ["wind", "gas"]
    lo = n.generators_t.p_min_pu["churn_sink b"] * n.generators.at["churn_sink b", "p_nom"]
    hi = n.generators_t.p_max_pu["churn_sink b"] * n.generators.at["churn_sink b", "p_nom"]
    assert lo.tolist() == pytest.approx([-50.0, -20.0, 0.0]) and hi.tolist() == pytest.approx([-50.0, -20.0, 0.0])


def test_balancing_stage_can_free_the_sink():
    """market.balancing.churn_sink: free (24 Oct 2026): anchored like any unit, bounded by its Stage 1 peak."""
    n = _network()
    ws = pd.DataFrame({"wind": [200.0, 200.0, 200.0], "gas": [0.0, 0.0, 50.0],
                       "churn_sink b": [-50.0, -20.0, 0.0]}, index=n.snapshots)
    out = fix_churn_sink(n, CFG, ws, LOG, balancing_mode="free")
    assert list(out.columns) == ["wind", "gas", "churn_sink b"]
    sink = n.generators.loc["churn_sink b"]
    assert sink.p_nom == pytest.approx(50.0) and sink.p_min_pu == -1.0 and sink.p_max_pu == 0.0
    assert "churn_sink b" not in n.generators_t.p_min_pu.columns
    with pytest.raises(ValueError, match="must be one of"):
        fix_churn_sink(_network(), CFG, ws, LOG, balancing_mode="loose")


def test_balancing_takes_the_sink_layout_from_stage1():
    """Stage 1 solves on a collapsed network: its sink units, not the full network's storage buses, define the layout."""
    n = _network()
    n.add("Bus", "c", carrier="AC")
    ws = pd.DataFrame({"wind": [200.0] * 3, "churn_sink c": [-10.0, 0.0, 0.0]}, index=n.snapshots)
    out = fix_churn_sink(n, CFG, ws, LOG)
    assert "churn_sink c" in n.generators.index and "churn_sink b" not in n.generators.index
    assert n.generators.at["churn_sink c", "bus"] == "c" and list(out.columns) == ["wind"]
    ws_bad = pd.DataFrame({"wind": [200.0] * 3, "churn_sink nowhere": [-1.0] * 3}, index=n.snapshots)
    with pytest.raises(ValueError, match="not in the balancing network"):
        fix_churn_sink(_network(), CFG, ws_bad, LOG)


def test_balancing_sink_follows_storage_the_delivery_revert_rehomed():
    """A delay case's revert drops a new bus and re-homes its storage (3 Nov 2026): the sink moves with it."""
    import json
    n = _network()
    n.meta = {"network_delivery": json.dumps({"appendix_b": {"rehomed": [
        {"component": "storage_unit", "name": "bat_new", "p_nom": 35.0, "from_bus": "gone", "to_bus": "b"},
        {"component": "generator", "name": "pv", "p_nom": 5.0, "from_bus": "gone_too", "to_bus": "b"}]}})}
    ws = pd.DataFrame({"wind": [200.0] * 3, "churn_sink gone": [-10.0, 0.0, 0.0]}, index=n.snapshots)
    out = fix_churn_sink(n, CFG, ws, LOG)
    assert n.generators.at["churn_sink gone", "bus"] == "b" and list(out.columns) == ["wind"]
    # only storage moves carry a sink; a bus dropped without one still fails
    ws_gen = pd.DataFrame({"wind": [200.0] * 3, "churn_sink gone_too": [-1.0] * 3}, index=n.snapshots)
    m = _network()
    m.meta = n.meta
    with pytest.raises(ValueError, match="not in the balancing network"):
        fix_churn_sink(m, CFG, ws_gen, LOG)


def test_balancing_refuses_a_stage1_without_the_sink():
    n = _network()
    ws = pd.DataFrame({"wind": [200.0] * 3}, index=n.snapshots)
    with pytest.raises(ValueError, match="no output"):
        fix_churn_sink(n, CFG, ws, LOG)
    with pytest.raises(ValueError, match="churn sink but"):
        fix_churn_sink(_network(), {}, ws.assign(**{"churn_sink b": -1.0}), LOG)


def test_round_trip_override_sets_both_legs():
    from scripts.market.market_utils import apply_storage_round_trip_override
    n = _network()
    assert apply_storage_round_trip_override(n, {"storage_round_trip_override": {"Battery": 0.70}}, LOG) == 1
    su = n.storage_units.loc["bat"]
    assert su.efficiency_store * su.efficiency_dispatch == pytest.approx(0.70)
    assert su.efficiency_store == pytest.approx(su.efficiency_dispatch)
    assert apply_storage_round_trip_override(_network(), {}, LOG) == 0
    with pytest.raises(ValueError):
        apply_storage_round_trip_override(_network(), {"storage_round_trip_override": {"Battery": 1.5}}, LOG)
