"""Calendar helpers for putting weather-year and source-year series on a modelled year.

A modelled year is a calendar year of 8,760 hours, or 8,784 in a leap year (2040 is one). The
weather year (2013 for the FES long-range studies) and most published hourly sources (TYNDP,
FES profiles) have 8,760, so a leap modelled year needs 29 February from somewhere. The rule
used throughout the model: values are matched by month, day and hour, and a 29 February the
source lacks takes 28 February's values (as ``fes_demand.map_weather_year`` and
``interconnectors.price_shape.align_by_calendar`` already do). A source 29 February that the
modelled year lacks is dropped.
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd


def hours_in_year(year: int) -> int:
    """Hours in calendar ``year``: 8,784 in a leap year, else 8,760."""
    return 8784 if calendar.isleap(int(year)) else 8760


def _is_leap_day(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((index.month == 2) & (index.day == 29))


def fill_leap_day(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Add 29 February, as a copy of 28 February, to a series on a leap year that lacks it.

    ``values`` has a DatetimeIndex on one calendar year (any regular sub-daily step). Unchanged
    when the year is not a leap year, when 29 February is already there, or when 28 February is
    not (a partial year that does not span it).
    """
    index = pd.DatetimeIndex(values.index)
    if len(index) == 0:
        return values
    year = int(index[0].year)
    if not calendar.isleap(year) or _is_leap_day(index).any():
        return values
    feb28 = (index.month == 2) & (index.day == 28)
    if not feb28.any():
        return values
    extra = values.loc[np.asarray(feb28)].copy()
    extra.index = pd.DatetimeIndex(extra.index) + pd.Timedelta(days=1)
    return pd.concat([values, extra]).sort_index()


def on_calendar(values: pd.Series | pd.DataFrame, target_index: pd.Index) -> pd.Series | pd.DataFrame:
    """Values of a source-year series at the same month, day and hour as each target timestamp.

    A 29 February the source lacks takes 28 February's values; a source 29 February the target
    lacks is unused. Raises when a target hour has no source counterpart.
    """
    source = values.copy()
    src_index = pd.DatetimeIndex(source.index)
    keys = pd.MultiIndex.from_arrays([src_index.month, src_index.day, src_index.hour, src_index.minute])
    source.index = keys
    source = source[~keys.duplicated(keep="first")]
    target = pd.DatetimeIndex(target_index)
    day = np.where(_is_leap_day(target) & (not _is_leap_day(src_index).any()), 28, target.day)
    wanted = pd.MultiIndex.from_arrays([target.month, day, target.hour, target.minute])
    out = source.reindex(wanted)
    if out.isna().to_numpy().any():
        missing = int(out.isna().to_numpy().any(axis=-1).sum()) if out.ndim > 1 else int(out.isna().sum())
        raise ValueError(f"{missing} target hours have no calendar match in the source series")
    out.index = pd.Index(target_index)
    return out
