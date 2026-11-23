"""Commitment proxy: premium on unscheduled units, floor on committed ones, nothing when off."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.commitment_proxy import apply_commitment_proxy


def _network():
    n = pypsa.Network()
    snaps = pd.date_range("2030-01-01", periods=4, freq="h")
    n.set_snapshots(snaps)
    n.add("Bus", "GB")
    n.add("Generator", "gas_on", bus="GB", carrier="CCGT", p_nom=100.0, p_min_pu=0.0, marginal_cost=60.0)
    n.add("Generator", "gas_low", bus="GB", carrier="CCGT", p_nom=100.0, p_min_pu=0.0, marginal_cost=60.0)
    n.add("Generator", "gas_off", bus="GB", carrier="CCGT", p_nom=100.0, p_min_pu=0.0, marginal_cost=60.0)
    n.add("Generator", "bio", bus="GB", carrier="biomass", p_nom=50.0, p_min_pu=0.35, marginal_cost=120.0)
    n.add("Generator", "wind", bus="GB", carrier="wind_onshore", p_nom=200.0, marginal_cost=0.0)
    n.add("Load", "l", bus="GB", p_set=pd.Series(150.0, index=snaps))
    sched = pd.DataFrame({
        "gas_on": [80.0, 80.0, 0.0, 80.0],     # committed above min stable, off in hour 3
        "gas_low": [20.0, 20.0, 20.0, 20.0],   # running below min stable: treated as uncommitted for the floor
        "gas_off": [0.0, 0.0, 0.0, 0.0],
        "bio": [50.0, 50.0, 50.0, 50.0],
        "wind": [100.0, 100.0, 100.0, 100.0],
    }, index=snaps)
    offer = pd.Series({"gas_on": 66.0, "gas_low": 66.0, "gas_off": 66.0, "bio": 130.0, "wind": 0.0})
    bid = pd.Series({"gas_on": 54.0, "gas_low": 54.0, "gas_off": 54.0, "bio": 100.0, "wind": 25.0})
    return n, sched, offer, bid


CFG = {"enabled": True, "min_stable_fraction": 0.4, "start_premium_gbp_mwh": 50.0, "committed_threshold_mw": 1.0}


def test_disabled_returns_inputs_untouched():
    n, sched, offer, bid = _network()
    o, b, s = apply_commitment_proxy(n, sched, offer, bid, None, None, {"enabled": False})
    assert o is None and b is None and s == {}
    assert n.generators_t.p_min_pu.empty


def test_floor_only_on_units_scheduled_at_or_above_min_stable():
    n, sched, offer, bid = _network()
    apply_commitment_proxy(n, sched, offer, bid, None, None, CFG)
    f = n.generators_t.p_min_pu
    assert list(f["gas_on"]) == [0.4, 0.4, 0.0, 0.4]      # off in hour 3 -> no floor
    assert (f["gas_low"] == 0.0).all()                      # below 40 MW -> not floored
    assert (f["gas_off"] == 0.0).all()
    assert (f["bio"] == 0.4).all()                          # 50/50 >= 0.4 and above its 0.35 static floor
    assert "wind" not in f.columns


def test_premium_only_where_unit_is_off_and_frames_are_created_for_all_generators():
    n, sched, offer, bid = _network()
    o, b, s = apply_commitment_proxy(n, sched, offer, bid, None, None, CFG)
    assert s["frames_created"] is True
    assert set(o.columns) == set(n.generators.index) and set(b.columns) == set(n.generators.index)
    assert list(o["gas_off"]) == [116.0] * 4
    assert list(o["gas_on"]) == [66.0, 66.0, 116.0, 66.0]
    assert (o["gas_low"] == 66.0).all()                     # 20 MW counts as committed for the premium
    assert (o["wind"] == 0.0).all() and (b["gas_off"] == 54.0).all()


def test_existing_frames_are_augmented_not_replaced():
    n, sched, offer, bid = _network()
    snaps = n.snapshots
    o0 = pd.DataFrame({"gas_off": [70.0] * 4, "wind": [1.0] * 4}, index=snaps)
    b0 = pd.DataFrame({"gas_off": [50.0] * 4, "wind": [2.0] * 4}, index=snaps)
    o, b, s = apply_commitment_proxy(n, sched, offer, bid, o0, b0, CFG)
    assert s["frames_created"] is False
    assert list(o["gas_off"]) == [120.0] * 4               # 70 existing + 50 premium
    assert (o["wind"] == 1.0).all()                         # untouched
    assert list(o["gas_on"]) == [66.0, 66.0, 116.0, 66.0]   # added from static, then premium in hour 3
    assert (b["gas_on"] == 54.0).all()


def test_static_floor_is_never_lowered():
    n, sched, offer, bid = _network()
    n.generators.at["bio", "p_min_pu"] = 0.55
    apply_commitment_proxy(n, sched, offer, bid, None, None, CFG)
    assert (n.generators_t.p_min_pu["bio"] == 0.55).all()


def test_a_lower_hourly_floor_replaces_the_static_one():
    # Review 26 Sep 2026: seasonal CHP carries an hourly floor below its static 0.3
    # in summer; the proxy must keep the hourly floor, not lift it back to 0.3.
    n, sched, offer, bid = _network()
    n.generators.loc["bio", "p_min_pu"] = 0.3
    n.generators_t.p_min_pu = pd.DataFrame({"bio": [0.18, 0.18, 0.5, 0.18]}, index=n.snapshots)
    sched["bio"] = [9.0, 0.0, 25.0, 9.0]  # 18% of 50 MW: committed but below min stable
    apply_commitment_proxy(n, sched, offer, bid, None, None, CFG)
    f = n.generators_t.p_min_pu
    assert list(f["bio"]) == pytest.approx([0.18, 0.18, 0.5, 0.18])
    # A unit with no hourly floor still keeps its static one.
    assert (f["gas_low"] == 0.0).all()
