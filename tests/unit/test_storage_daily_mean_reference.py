"""storage_bid_reference stage1_daily_mean (10 Oct 2026): storage offers referenced to the day's mean price."""
import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import _attach_storage_price_frames


def _net(prices):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-05-01", periods=len(prices), freq="h"))
    n.add("Bus", "b")
    n.add("StorageUnit", "bat", bus="b", carrier="Battery", p_nom=100, max_hours=2,
          efficiency_store=0.92, efficiency_dispatch=0.92)
    n._bm_reference_price = pd.Series(prices, index=n.snapshots)
    return n


def _frames(reference, prices):
    n = _net(prices)
    cfg = {"storage_bid_reference": reference,
           "carrier_overrides": {"Battery": {"mode": "round_trip", "bid_margin": 44.0, "bid_price": 40.0,
                                             "offer_premium": 67.0}}}
    rte = 0.92 * 0.92
    offer = pd.Series({"bat": 40.0 / rte + 67.0})   # static offer = paid / eta + premium; premium_eff = offer + bid / eta
    bid = pd.Series({"bat": -40.0})
    _attach_storage_price_frames(n, cfg, offer, bid, logging.getLogger())
    return n._bm_su_offer_tv["bat"], rte


def test_daily_mean_lifts_offers_in_a_negative_hour():
    prices = [60.0] * 12 + [-10.0] * 12
    hourly, rte = _frames("stage1_price", prices)
    daily, _ = _frames("stage1_daily_mean", prices)
    assert hourly.iloc[18] == pytest.approx((-10.0 - 44.0) / rte + 67.0, abs=0.01)   # about 3 GBP/MWh
    assert daily.iloc[18] > hourly.iloc[18] + 20.0
    assert np.isclose(daily.iloc[12], ((pd.Series(prices).rolling(24, center=True, min_periods=1).mean()[12] - 44.0) / rte
                                      + 67.0))
