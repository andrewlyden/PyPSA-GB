"""Storage offer_floor (11 Oct 2026): indexed round_trip offers never fall below the 2024 low-price offer level."""
import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import _attach_storage_price_frames

RTE = 0.92 * 0.92


def _offers(floor, prices):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-05-01", periods=len(prices), freq="h"))
    n.add("Bus", "b")
    n.add("StorageUnit", "bat", bus="b", carrier="Battery", p_nom=100, max_hours=2,
          efficiency_store=0.92, efficiency_dispatch=0.92)
    n._bm_reference_price = pd.Series(prices, index=n.snapshots)
    spec = {"mode": "round_trip", "bid_margin": 44.0, "bid_price": 40.0, "offer_premium": 67.0}
    if floor is not None:
        spec["offer_floor"] = floor
    cfg = {"storage_bid_reference": "stage1_price", "carrier_overrides": {"Battery": spec}}
    offer = pd.Series({"bat": 40.0 / RTE + 67.0})
    bid = pd.Series({"bat": -40.0})
    _attach_storage_price_frames(n, cfg, offer, bid, logging.getLogger())
    return n._bm_su_offer_tv["bat"], n._bm_su_bid_tv["bat"]


def test_floor_binds_only_in_low_price_hours():
    prices = [0.0, 120.0]
    floored, bids = _offers(69.0, prices)
    plain, plain_bids = _offers(None, prices)
    assert plain.iloc[0] == pytest.approx((0.0 - 44.0) / RTE + 67.0)   # about 15 GBP/MWh unfloored
    assert floored.iloc[0] == pytest.approx(69.0)
    assert floored.iloc[1] == pytest.approx(plain.iloc[1])             # above the floor: formula unchanged
    pd.testing.assert_series_equal(bids, plain_bids)                  # bids untouched


def test_no_floor_key_keeps_previous_offers():
    prices = [-10.0, 0.0, 60.0]
    plain, _ = _offers(None, prices)
    zero, _ = _offers(0.0, prices)
    pd.testing.assert_series_equal(plain, zero)
