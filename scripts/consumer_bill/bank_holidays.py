"""England and Wales bank holidays, the days the Elexon peak block leaves out (F18, 27 Oct 2026).

Ofgem's cap model (Annex 2, sheet ``3b Demand``) carries a bank-holiday list and treats those
days as non-peak in its base/peak split, so the bill's peak block does too. The dates are a
committed calendar, never fetched at run time:
``data/market/consumer_bill_sources/england_wales_bank_holidays.csv`` (``date``, ``name``,
``basis``, ``source``), England and Wales only (Scotland and Northern Ireland differ; the cap is
one GB figure and Ofgem's list is the England and Wales one).

* ``published``: the dates gov.uk lists for England and Wales (2019-2028, checked against
  https://www.gov.uk/bank-holidays.json on 27 Oct 2026), including the
  one-off proclamations (VE Day 2020, the Platinum Jubilee and the Queen's funeral 2022, the
  Coronation 2023), and the earlier one-offs back to 2010 (the 2011 Royal Wedding, the 2012
  Diamond Jubilee).
* ``projected``: later years by :func:`standard_dates`, the rules of the Banking and
  Financial Dealings Act 1971 (Schedule 1) and the standing proclamations for New Year's Day
  and the early May holiday, with the usual substitute days. A later proclamation (a one-off
  holiday or a moved date) would be added by hand as a ``published`` row.

The bill is calendar 2030 (its snapshots are dated in the model year, on 2013 weather), so the
model year's own dates apply; the file covers 2010-2060 so the historical bills (2020-2024)
and the 2050 study are inside it, and a date outside the coverage is refused.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALENDAR_FILE = PROJECT_ROOT / "data/market/consumer_bill_sources/england_wales_bank_holidays.csv"
BASIS_PUBLISHED, BASIS_PROJECTED = "published", "projected"


def easter_sunday(year: int) -> dt.date:
    """Easter Sunday in the Gregorian calendar (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def _first_monday(year: int, month: int) -> dt.date:
    day = dt.date(year, month, 1)
    return day + dt.timedelta(days=(0 - day.weekday()) % 7)


def _last_monday(year: int, month: int) -> dt.date:
    day = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return day - dt.timedelta(days=day.weekday())


def standard_dates(year: int) -> dict[dt.date, str]:
    """The England and Wales bank holidays of ``year`` under the standing rules (no one-off proclamations)."""
    easter = easter_sunday(year)
    new_year = dt.date(year, 1, 1)
    new_year += dt.timedelta(days={5: 2, 6: 1}.get(new_year.weekday(), 0))
    christmas, boxing = dt.date(year, 12, 25), dt.date(year, 12, 26)
    if christmas.weekday() == 5:          # Saturday: both move to Monday and Tuesday
        christmas, boxing = dt.date(year, 12, 27), dt.date(year, 12, 28)
    elif christmas.weekday() == 6:        # Sunday: Boxing Day on Monday, Christmas substitute on Tuesday
        christmas = dt.date(year, 12, 27)
    elif boxing.weekday() == 5:           # Friday Christmas: Boxing Day substitute on Monday
        boxing = dt.date(year, 12, 28)
    return {
        new_year: "New Year's Day", easter - dt.timedelta(days=2): "Good Friday",
        easter + dt.timedelta(days=1): "Easter Monday", _first_monday(year, 5): "Early May bank holiday",
        _last_monday(year, 5): "Spring bank holiday", _last_monday(year, 8): "Summer bank holiday",
        christmas: "Christmas Day", boxing: "Boxing Day",
    }


@lru_cache(maxsize=None)
def _calendar(path: str) -> tuple[frozenset, int, int]:
    frame = pd.read_csv(path)
    missing = {"date", "name", "basis", "source"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d").dt.date
    if dates.duplicated().any():
        raise ValueError(f"{path} repeats a date")
    return frozenset(dates), int(min(d.year for d in dates)), int(max(d.year for d in dates))


def england_wales_bank_holidays(path: Path | str = CALENDAR_FILE) -> tuple[frozenset, int, int]:
    """``(dates, first year, last year)`` of the committed England and Wales calendar."""
    return _calendar(str(path))


def is_bank_holiday(local_dates, path: Path | str = CALENDAR_FILE) -> pd.Series:
    """Whether each local calendar date is an England and Wales bank holiday; refuses a year outside the file."""
    dates, first, last = england_wales_bank_holidays(path)
    days = pd.Series(pd.to_datetime(pd.Index(local_dates)).date)
    years = {d.year for d in days}
    if years and (min(years) < first or max(years) > last):
        raise ValueError(f"the bank-holiday calendar covers {first}-{last}; the dates span "
                         f"{min(years)}-{max(years)} ({Path(path).name})")
    return days.isin(dates)
