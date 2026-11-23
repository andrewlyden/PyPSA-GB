"""The anchored legacy price path (user decisions, 10 Nov 2026).

Each technology's capture price runs in a straight line through today's forward level and every solved year,
flat after the last; today's anchor keeps the earliest solved year's capture ratios.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import levers
from scripts.public_ownership.legacy_valuation import PRICE_PATH_ANCHORED, anchored_price_ratios
from scripts.public_ownership.price_anchors import build


def _settlement(price):
    return pd.DataFrame({"carrier": ["wind_offshore", "solar_pv"], "wholesale_mwh": [1e6, 1e6],
                         "wholesale_revenue_gbp": [price * 1e6, 2 * price * 1e6]})


def test_the_anchored_path_is_the_central():
    assert levers.LEVERS["legacy_price_path"].default == PRICE_PATH_ANCHORED == "anchored"


def test_today_keeps_the_first_solved_years_capture_ratios():
    out = build({2030: _settlement(30.0), 2050: _settlement(20.0)},
                {"year": 2026, "level": 100.0, "deflator": 0.9, "reference": 60.0})
    t = out.pivot_table(index="carrier", columns="year", values="wholesale_gbp_per_mwh")
    assert t.loc["wind_offshore", 2026] == pytest.approx(30.0 * 100.0 * 0.9 / 60.0)
    assert t.loc["solar_pv", 2026] / t.loc["wind_offshore", 2026] == pytest.approx(2.0)
    assert t.loc["wind_offshore", 2050] == pytest.approx(20.0)


def test_ratios_run_straight_between_anchors_and_flat_after():
    anchors = pd.DataFrame({"carrier": ["wind_offshore"] * 3, "year": [2026, 2030, 2050],
                            "wholesale_gbp_per_mwh": [50.0, 30.0, 20.0]})
    years = np.arange(2030, 2061)
    r = anchored_price_ratios(["wind_offshore", "CCGT"], years, anchors, 2030)
    assert r[0, 0] == pytest.approx(1.0)
    assert r[0, 10] == pytest.approx(25.0 / 30.0)       # 2040, halfway to 2050
    assert r[0, 20] == pytest.approx(20.0 / 30.0) == r[0, 30]   # 2050 and flat after
    assert (r[1] == 1.0).all()                            # a carrier off the path keeps its price
