"""Item BM1, P3: the acquisition tariff on the P3 months reads the right files and prices the right case.

The P3 months share their parent month's Stage 1 (``wholesale_from``), so no
``*_wholesale_links.csv`` exists under their own names. The counter-trade premium
and the congestion rent are Stage 1 quantities and must be read from the parent;
the physical flows still come from the P3 month's own balancing network. The
tariff on a tariff balancing group prices its bid case alone.
"""

import pandas as pd
import pytest

from scripts.public_ownership import nationalised_institutions as ni
from scripts.public_ownership.nationalised_tariff import select_tariff_cases


def test_the_premium_reads_the_parent_stage_1_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
    index = pd.date_range("2030-01-01", periods=3, freq="h")
    # Only the parent carries a Stage 1 schedule, as on disk for a wholesale_from month.
    pd.DataFrame({"IFA": [100.0, 100.0, 100.0]}, index=index).to_csv(tmp_path / "PARENT_wholesale_links.csv")
    flow = pd.DataFrame({"IFA": [100.0, 40.0, 130.0]}, index=index)
    links = pd.DataFrame({"market_counter_trade_premium": [100.0]}, index=["IFA"])
    premium = ni.counter_trade_premium("P3_MONTH", flow, links, "PARENT")
    assert premium["IFA"].tolist() == [0.0, 6000.0, 3000.0]
    with pytest.raises(FileNotFoundError, match="P3_MONTH_wholesale_links"):
        ni.counter_trade_premium("P3_MONTH", flow, links)


def test_link_trade_pairs_each_month_with_one_stage_1_month():
    with pytest.raises(ValueError, match="pair month by month"):
        ni.measure_link_trade(["A", "B"], schedule_scenarios=["A"])


def test_the_tariff_prices_the_requested_cases_only():
    available = ["public_ownership@legacy_acquired", "public_ownership@fully_nationalised_market_value"]
    assert select_tariff_cases(available, None) == available
    assert select_tariff_cases(available, ["public_ownership@fully_nationalised_market_value"]) == [
        "public_ownership@fully_nationalised_market_value"]
    with pytest.raises(ValueError, match="must name tariff cases"):
        select_tariff_cases(available, ["public_ownership"])
    with pytest.raises(ValueError, match="must name tariff cases"):
        select_tariff_cases(available, [])
