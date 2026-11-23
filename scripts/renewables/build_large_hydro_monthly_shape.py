#!/usr/bin/env python3
"""Build the large-hydro monthly inflow shape from ESPENI metered hydro output.

Reads ``ELEC_POWER_ELEX_NPSHYD[MW]`` (Elexon transmission-metered non-pumped
hydro) from ``data/demand/espeni.csv`` and writes, for every complete calendar
year, each month's mean MW divided by that year's mean of the monthly means
(so the twelve values average 1). Used by
``scripts/renewables/large_hydro_inflow.py`` for proxy hydro units with no
site profile (audit 26 Sep 2026).

    python scripts/renewables/build_large_hydro_monthly_shape.py \
        --espeni data/demand/espeni.csv \
        --output data/renewables/hydro/espeni_npshyd_monthly_shape.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TIME_COLUMN = "ELEC_elex_startTime[utc](datetime)"
HYDRO_COLUMN = "ELEC_POWER_ELEX_NPSHYD[MW](float32)"


def build_monthly_shape(espeni: pd.DataFrame) -> pd.DataFrame:
    series = espeni.set_index(pd.to_datetime(espeni[TIME_COLUMN], utc=True))[HYDRO_COLUMN].astype(float)
    counts = series.groupby([series.index.year, series.index.month]).count()
    monthly = series.groupby([series.index.year, series.index.month]).mean()
    table = pd.DataFrame({"mean_mw": monthly, "half_hours": counts})
    table.index = table.index.set_names(["year", "month"])
    table = table.reset_index()
    # A year is complete when all twelve months are present with at least 90%
    # of their half-hours; partial first/last years are dropped.
    days = pd.to_datetime(dict(year=table["year"], month=table["month"], day=1)).dt.days_in_month
    table = table[table["half_hours"] >= 0.9 * 48 * days]
    complete = table.groupby("year")["month"].nunique().eq(12)
    table = table[table["year"].isin(complete[complete].index)].copy()
    table["shape"] = table["mean_mw"] / table.groupby("year")["mean_mw"].transform("mean")
    return table[["year", "month", "mean_mw", "shape"]].round({"mean_mw": 2, "shape": 5})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espeni", default="data/demand/espeni.csv")
    parser.add_argument("--output", default="data/renewables/hydro/espeni_npshyd_monthly_shape.csv")
    args = parser.parse_args()
    espeni = pd.read_csv(args.espeni, usecols=[TIME_COLUMN, HYDRO_COLUMN])
    table = build_monthly_shape(espeni)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} rows for {table['year'].nunique()} complete years to {args.output}")


if __name__ == "__main__":
    main()
