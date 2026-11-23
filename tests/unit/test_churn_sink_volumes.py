"""Item F22 (13 Oct 2026): the churn sink is not a balancing action.

``fix_churn_sink`` holds the sink at its Stage 1 output and drops it from the
wholesale table, so aligning that table on the balancing dispatch filled its
position with zero and its absorption (about 4.4 TWh a year) showed as a
zero-cost turn-down. The accounting now leaves it out; costs do not move.
"""
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.market.market_utils import (
    align_redispatch_generators,
    churn_sink_columns,
    compute_redispatch_volumes,
)

LOG = logging.getLogger("test_churn_sink_volumes")


def _frames():
    snaps = pd.date_range("2030-01-01", periods=4, freq="h")
    wholesale = pd.DataFrame({"wind": [100.0, 100.0, 80.0, 50.0], "ccgt": [0.0, 10.0, 20.0, 0.0]}, index=snaps)
    # Balancing dispatch: wind turned down, CCGT turned up, the sink held at its Stage 1 output.
    physical = pd.DataFrame({"wind": [60.0, 100.0, 40.0, 50.0], "ccgt": [40.0, 10.0, 60.0, 0.0],
                             "churn_sink BUS1": [-30.0, -20.0, 0.0, -10.0],
                             "churn_sink BUS2": [-5.0, 0.0, 0.0, 0.0]}, index=snaps)
    network = SimpleNamespace(
        generators=pd.DataFrame({"carrier": ["wind_onshore", "CCGT", "churn_sink", "churn_sink"]},
                                index=["wind", "ccgt", "churn_sink BUS1", "churn_sink BUS2"]),
        storage_units=pd.DataFrame(columns=["carrier"]),
    )
    offer = pd.Series({"wind": 0.0, "ccgt": 120.0, "churn_sink BUS1": 0.0, "churn_sink BUS2": 0.0})
    bid = pd.Series({"wind": 40.0, "ccgt": 50.0, "churn_sink BUS1": 0.0, "churn_sink BUS2": 0.0})
    return snaps, wholesale, physical, network, offer, bid


def _volumes(wholesale, physical, network, offer, bid):
    empty = pd.DataFrame(index=physical.index)
    gen, _ = compute_redispatch_volumes(wholesale, physical, empty, empty, offer, bid,
                                        pd.Series(dtype=float), pd.Series(dtype=float), network, LOG)
    return gen.set_index("component")


def test_sink_is_left_out_of_the_redispatch_volumes_and_costs_do_not_move():
    snaps, wholesale, physical, network, offer, bid = _frames()
    # Before: the sink's dropped Stage 1 column aligned as zero.
    old = _volumes(wholesale.reindex(columns=physical.columns, fill_value=0.0), physical, network, offer, bid)
    assert old.loc[churn_sink_columns(old.index), "decrease_MWh"].sum() == pytest.approx(65.0)

    ws, phys, sink_mwh = align_redispatch_generators(wholesale, physical, snaps, LOG)
    new = _volumes(ws, phys, network, offer, bid)
    assert not churn_sink_columns(new.index)
    assert sink_mwh == pytest.approx(65.0)
    # The real actions and every cost are unchanged.
    for column in ("increase_MWh", "decrease_MWh", "offer_cost", "bid_cost", "net_cost"):
        assert new.loc[["wind", "ccgt"], column].to_numpy() == pytest.approx(old.loc[["wind", "ccgt"], column].to_numpy())
    for column in ("offer_cost", "bid_cost", "net_cost"):
        assert new[column].sum() == pytest.approx(old[column].sum())
    assert new["decrease_MWh"].sum() == pytest.approx(old["decrease_MWh"].sum() - 65.0)


def test_half_hourly_energy_and_no_sink():
    snaps = pd.date_range("2030-01-01", periods=4, freq="30min")
    physical = pd.DataFrame({"g": [1.0, 2.0, 3.0, 4.0], "churn_sink X": [-10.0, -10.0, 0.0, 0.0]}, index=snaps)
    _, phys, sink_mwh = align_redispatch_generators(physical[["g"]], physical, snaps)
    assert sink_mwh == pytest.approx(10.0) and list(phys.columns) == ["g"]
    ws, phys, sink_mwh = align_redispatch_generators(physical[["g"]], physical[["g"]], snaps)
    assert sink_mwh == 0.0 and list(ws.columns) == ["g"]
    assert churn_sink_columns(["churn_sink A", "churn_sinking", "wind"]) == ["churn_sink A"]
    assert np.isfinite(sink_mwh)
