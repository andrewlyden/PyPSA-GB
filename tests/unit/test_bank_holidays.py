"""F18 (27 Oct 2026): England and Wales bank holidays are not in the Elexon peak block.

The calendar is committed data (``england_wales_bank_holidays.csv``): published dates with their
one-off proclamations to 2027, later years projected by the standing rules.
"""

import datetime as dt

import pandas as pd
import pytest

from scripts.consumer_bill import bank_holidays as bh
from scripts.consumer_bill.build_domestic_demand_profile import peak_mask

ONE_OFF_YEARS = {2011, 2012, 2020, 2022, 2023}


@pytest.fixture(scope="module")
def calendar():
    return pd.read_csv(bh.CALENDAR_FILE, parse_dates=["date"])


def test_the_calendar_covers_the_bill_years_and_states_its_basis(calendar):
    dates, first, last = bh.england_wales_bank_holidays()
    assert first <= 2020 and last >= 2050                # the historical bills, 2030 and the 2050 study
    assert set(calendar["basis"]) == {bh.BASIS_PUBLISHED, bh.BASIS_PROJECTED}
    assert calendar["source"].str.len().gt(20).all()
    assert (calendar.loc[calendar["basis"].eq(bh.BASIS_PROJECTED), "date"].dt.year >= 2028).all()
    assert len(dates) == len(calendar)


def test_the_years_without_proclamations_follow_the_standing_rules(calendar):
    by_year = calendar.groupby(calendar["date"].dt.year)["date"].apply(lambda d: set(d.dt.date))
    for year, dates in by_year.items():
        if year in ONE_OFF_YEARS:
            continue
        assert dates == set(bh.standard_dates(int(year))), year
    # The one-offs: VE Day moved the 2020 early May holiday; 2022 had ten.
    assert dt.date(2020, 5, 8) in by_year[2020] and dt.date(2020, 5, 4) not in by_year[2020]
    assert {dt.date(2022, 6, 2), dt.date(2022, 6, 3), dt.date(2022, 9, 19)} <= by_year[2022]
    assert len(by_year[2022]) == 10 and dt.date(2023, 5, 8) in by_year[2023]


def test_substitute_days_and_easter():
    assert bh.easter_sunday(2030) == dt.date(2030, 4, 21) and bh.easter_sunday(2024) == dt.date(2024, 3, 31)
    # 2021: Christmas on a Saturday moves both days; 2022: a Sunday Christmas moves Christmas only.
    assert {dt.date(2021, 12, 27), dt.date(2021, 12, 28)} <= set(bh.standard_dates(2021))
    assert {dt.date(2022, 12, 26), dt.date(2022, 12, 27)} <= set(bh.standard_dates(2022))
    assert dt.date(2026, 12, 28) in bh.standard_dates(2026)          # Friday Christmas: Boxing Day on Monday
    assert dt.date(2033, 1, 3) in bh.standard_dates(2033)            # New Year's Day on a Saturday


def test_the_peak_block_leaves_out_bank_holidays_in_the_model_year():
    index = pd.date_range("2030-01-01", "2031-01-01", freq="h", inclusive="left")
    with_holidays, without = peak_mask(index), peak_mask(index, bank_holidays=False)
    # 2030: 261 weekdays, all eight holidays on weekdays: 253 peak days of twelve local hours.
    assert int(without.sum()) == 261 * 12
    assert int(with_holidays.sum()) == 253 * 12
    christmas = pd.DatetimeIndex(["2030-12-25 10:00", "2030-12-27 10:00"])
    assert peak_mask(christmas).tolist() == [False, True]
    # Local dates: 23:30 UTC on 25 August 2030 is 00:30 BST on the bank holiday (a Monday), off peak anyway;
    # 06:30 UTC on 26 August is 07:30 BST on the holiday, which would be peak on another Monday.
    assert peak_mask(pd.DatetimeIndex(["2030-08-26 06:30", "2030-09-02 06:30"])).tolist() == [False, True]


def test_a_year_outside_the_calendar_is_refused():
    with pytest.raises(ValueError, match="calendar covers"):
        peak_mask(pd.DatetimeIndex(["2075-01-02 10:00"]))
    assert peak_mask(pd.DatetimeIndex([])).empty
