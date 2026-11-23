"""Carrier-median ELEXON fallback: the cap applies to each side separately (18 Nov 2026)."""

import numpy as np
import pandas as pd

from scripts.market.market_utils import carrier_medians_within_cap


def test_wind_bids_survive_its_do_not_call_offers():
    carriers = pd.Series(["wind_onshore"] * 3 + ["CCGT"] * 2, index=list("abcde"))
    offers = pd.Series([1500.0, 1600.0, np.nan, 60.0, 62.0], index=carriers.index)
    bids = pd.Series([74.0, 73.0, np.nan, -30.0, -32.0], index=carriers.index)
    med_offer, med_bid, rejected = carrier_medians_within_cap(carriers, offers, bids, 500.0)
    assert "wind_onshore" not in med_offer          # its offers stay derived
    assert med_bid["wind_onshore"] == 73.5          # its bids are what wind bid
    assert med_offer["CCGT"] == 61.0 and med_bid["CCGT"] == -31.0
    assert set(rejected) == {"wind_onshore"}
