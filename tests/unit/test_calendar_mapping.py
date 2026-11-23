"""Source-year series go on the modelled year by calendar when exactly one of the two is a leap year.

User decision, 27 Oct 2026: a 2020 weather or demand year on a non-leap model year (and a 2013 year on
2040) is matched by month, day and hour. By day of year every day from 1 March read a day off.
"""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.demand.load import _complete_other_leap_year
from scripts.interconnectors.add_to_network import apply_availability_profiles


def _hourly(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")


@pytest.mark.parametrize("source_year, model_year, expected", [
    (2020, 2030, True),    # leap source, common target
    (2013, 2040, True),    # common source, leap target
    (2013, 2030, False),   # both common: day of year is already the calendar
    (2020, 2040, False),   # both leap
])
def test_complete_other_leap_year(source_year, model_year, expected):
    assert _complete_other_leap_year(_hourly(source_year), _hourly(model_year), model_year) is expected


def test_partial_source_year_is_not_mapped_by_calendar():
    assert not _complete_other_leap_year(_hourly(2020)[:-24], _hourly(2030), 2030)


def _network(year: int) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(_hourly(year))
    n.add("Bus", ["GB", "FR"])
    n.add("Link", "IC_Test", bus0="GB", bus1="FR", p_nom=1000, p_min_pu=-1.0, p_max_pu=1.0)
    n.links.loc["IC_Test", "interconnector_name"] = "Test"
    return n


def _availability(year: int, out_days) -> pd.DataFrame:
    """Full availability except whole days out of service, given as (month, day)."""
    times = _hourly(year)
    value = np.ones(len(times))
    for month, day in out_days:
        value[(times.month == month) & (times.day == day)] = 0.0
    return pd.DataFrame({"name": "Test", "time": times, "p_max_pu": value})


def test_leap_source_on_common_year_keeps_the_date():
    # An outage on 1 March 2020 stays on 1 March (by day of year it fell on 2 March); 29 February is unused.
    n = _network(2030)
    apply_availability_profiles(n, _availability(2020, [(3, 1), (2, 29)]))
    p = n.links_t.p_max_pu["IC_Test"]
    assert (p[p.index.normalize() == "2030-03-01"] == 0.0).all()
    assert (p[p.index.normalize() == "2030-03-02"] == 1.0).all()
    assert (p[p.index.normalize() == "2030-02-28"] == 1.0).all()
    assert (p[p.index.normalize() == "2030-12-31"] == 1.0).all()


def test_common_source_on_leap_year_fills_29_february_from_28():
    n = _network(2040)
    apply_availability_profiles(n, _availability(2013, [(2, 28), (12, 31)]))
    p = n.links_t.p_max_pu["IC_Test"]
    assert (p[p.index.normalize() == "2040-02-29"] == 0.0).all()
    assert (p[p.index.normalize() == "2040-12-31"] == 0.0).all()
    assert (p[p.index.normalize() == "2040-03-01"] == 1.0).all()
