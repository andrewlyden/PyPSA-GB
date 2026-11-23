"""Monthly-shaped inflow profiles for large-hydro units without a site profile.

Why this exists (audit 26 Sep 2026): the CP30 hydro line (1,897 MW, 128 FES
proxy units) matches none of the 15 named-station columns in
``large_hydro_{year}.csv``, so ``add_renewable_generators`` gave every unit a
flat ``p_max_pu`` of 0.4 all year. Two things were wrong with that:

* no seasonality: GB reservoir hydro produces about three times as much in a
  wet winter month as in a dry summer one (ESPENI ``ELEX_NPSHYD`` monthly means,
  2009-2024: December 1.40x, June 0.50x the annual mean);
* an energy level unrelated to CP30: the value was a placeholder, not the
  pathway's 3.49 TWh on 1,897 MW (CF 0.210).

The balancing and wholesale stages treat large hydro as a reservoir
(``scripts/solve/hydro_constraints.py``): the hourly ceiling is ``p_max_pu``
and the cumulative water budget accrues ``p_max_pu x p_nom x
inflow_profile_factor``. A profile for those units therefore has to be written
in the same convention, ``p_max_pu = inflow_cf / inflow_profile_factor``, so
the water budget receives exactly the calibrated inflow and the power ceiling
keeps the ``1 / inflow_profile_factor`` headroom every other large-hydro unit
has. With the reservoir model off, ``p_max_pu`` is the energy itself.

The monthly shape is the observed output of transmission-metered GB
non-pumped hydro (ESPENI ``ELEX_NPSHYD``, Elexon BM data, Wilson et al. 2020,
doi:10.5281/zenodo.3884858) in the renewables weather year, so hydro and wind
see the same year. Monthly output of a reservoir fleet whose storage is a few
weeks is a close proxy for monthly inflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SHAPE_COLUMNS = {"year", "month", "shape"}


def load_monthly_shape(path: str | Path, weather_year: int | None) -> tuple[pd.Series, str]:
    """Return a 12-value monthly shape (mean 1 over the months) and its label.

    Uses the weather year's row when present and complete, otherwise the mean
    shape over all complete years, so a weather year outside the ESPENI record
    still gets a documented seasonal profile.
    """
    table = pd.read_csv(path)
    missing = SHAPE_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Large-hydro monthly shape file {path} missing columns {sorted(missing)}")
    table = table.astype({"year": int, "month": int})
    complete = table.groupby("year")["month"].nunique().eq(12)
    complete_years = complete[complete].index
    if weather_year is not None and int(weather_year) in complete_years:
        shape = table.loc[table["year"].eq(int(weather_year))].set_index("month")["shape"]
        label = f"ESPENI NPSHYD {int(weather_year)}"
    else:
        shape = (table.loc[table["year"].isin(complete_years)]
                 .groupby("month")["shape"].mean())
        label = f"ESPENI NPSHYD mean {complete_years.min()}-{complete_years.max()}"
    shape = shape.reindex(range(1, 13)).astype(float)
    if shape.isna().any() or (shape < 0).any():
        raise ValueError(f"Large-hydro monthly shape from {path} is incomplete or negative")
    return shape / shape.mean(), label


def cp30_hydro_capacity_factor(es1_path: str | Path, pathway: str, modelled_year: int,
                               hours: float = 8760.0) -> float:
    """CP30 ES1 hydro generation / (hydro capacity x hours)."""
    es1 = pd.read_csv(es1_path)
    year_col = str(modelled_year)
    rows = es1[(es1["Pathway"] == pathway)
               & es1["Type"].astype(str).str.strip().eq("Hydro")
               & es1["SubType"].astype(str).str.strip().eq("Hydro")]
    generation_gwh = pd.to_numeric(rows.loc[rows["Variable"].eq("Generation (GWh)"), year_col],
                                   errors="coerce").sum()
    capacity_mw = pd.to_numeric(rows.loc[rows["Variable"].eq("Capacity (MW)"), year_col],
                                errors="coerce").sum()
    if capacity_mw <= 0 or generation_gwh <= 0:
        raise ValueError(f"CP30 ES1 has no hydro generation/capacity for {pathway} {modelled_year}")
    return float(generation_gwh * 1e3 / (capacity_mw * hours))


def large_hydro_proxy_p_max_pu(
    snapshots: pd.DatetimeIndex,
    annual_capacity_factor: float,
    monthly_shape: pd.Series,
    inflow_profile_factor: float | None,
) -> pd.Series:
    """Hourly ``p_max_pu`` whose calendar-year inflow equals the target CF.

    The shape is normalised over the hours of the snapshots' calendar year, not
    over the snapshots themselves, so a network built for one month gets that
    month's share of the year rather than the annual mean. On a full-year
    network the mean inflow CF is exactly ``annual_capacity_factor``.
    ``inflow_profile_factor`` (reservoir model on) converts inflow to the
    ceiling convention of ``hydro_constraints``. Values above 1 are clipped.
    """
    snapshots = pd.DatetimeIndex(snapshots)
    year = int(pd.Series(snapshots.year).mode().iloc[0])
    month_hours = pd.Series(
        [pd.Timestamp(year=year, month=m, day=1).days_in_month * 24.0 for m in range(1, 13)],
        index=range(1, 13),
    )
    shape = monthly_shape.reindex(range(1, 13)).astype(float)
    shape = shape / float((shape * month_hours).sum() / month_hours.sum())
    inflow_cf = float(annual_capacity_factor) * pd.Series(shape.reindex(snapshots.month).to_numpy(), index=snapshots)
    if inflow_profile_factor:
        return (inflow_cf / float(inflow_profile_factor)).clip(upper=1.0)
    return inflow_cf.clip(upper=1.0)


def inflow_capacity_factor(p_max_pu: pd.Series, inflow_profile_factor: float | None,
                           snapshot_weights: pd.Series | None = None) -> float:
    """Weighted mean inflow CF implied by a ceiling profile (inverse of the above)."""
    weights = (pd.Series(1.0, index=p_max_pu.index) if snapshot_weights is None
               else pd.Series(snapshot_weights, index=p_max_pu.index).astype(float))
    return float(np.average(p_max_pu, weights=weights) * (float(inflow_profile_factor) if inflow_profile_factor else 1.0))


ESPENI_TIME_COLUMN = "ELEC_elex_startTime[utc](datetime)"
ESPENI_HYDRO_COLUMN = "ELEC_POWER_ELEX_NPSHYD[MW](float32)"


def metered_hydro_mw(espeni_path: str | Path, year: int) -> pd.Series | None:
    """ESPENI NPSHYD (Elexon transmission-metered non-pumped hydro), half-hourly MW in ``year`` (UTC, naive)."""
    table = pd.read_csv(espeni_path, usecols=[ESPENI_TIME_COLUMN, ESPENI_HYDRO_COLUMN])
    times = pd.to_datetime(table[ESPENI_TIME_COLUMN], utc=True).dt.tz_localize(None)
    series = pd.Series(table[ESPENI_HYDRO_COLUMN].astype(float).to_numpy(), index=times)
    series = series[series.index.year == int(year)].dropna()
    days = 366 if pd.Timestamp(year=int(year), month=12, day=31).dayofyear == 366 else 365
    return series if series.index.normalize().nunique() >= days else None


def apply_historical_metered_hydro(network, scenario_config: dict, weather_year: int | None,
                                   espeni_path: str | None, log) -> bool:
    """Historical years: the large-hydro fleet's inflow is ESPENI's metered hydro output, day by day.

    The named-station profiles (``generate_hydro_profiles.py``) are an availability placeholder: a sine
    that peaks in July, and under the reservoir model a flat inflow CF of about 0.55. In the 2020
    validation (19 Nov 2026) Scottish hydro ran at 780 MW in August against 260 MW metered, and the
    balancing stage turned it down by 1.8 TWh in the year (ELEXON: 0.3).

    Here each unit's inflow (``generators_t.hydro_inflow_pu``, read by the reservoir model in place of
    ``p_max_pu x inflow_profile_factor``) is the day's mean metered output over the fleet's capacity, so
    wet and dry spells inside a month are kept and the fleet's water is the energy GB hydro produced. The
    power ceiling (``p_max_pu``) is the fleet's highest metered half-hour of the year over its capacity.
    Needs the reservoir model. Future years keep their own path (CP30 ES1 or site profiles).
    Returns True when applied.
    """
    from scripts.renewables.large_hydro_reservoir import _set_generator_series
    from scripts.solve.hydro_constraints import LARGE_HYDRO_INFLOW_SERIES
    from scripts.utilities.scenario_detection import target_source

    if target_source(scenario_config) != "historical" or not espeni_path or weather_year is None:
        return False
    hydro_cfg = (scenario_config.get("renewables", {}) or {}).get("hydro", {}) or {}
    if not (hydro_cfg.get("enabled", False) and (hydro_cfg.get("large_hydro", {}) or {}).get("reservoir_model", False)):
        return False
    gens = network.generators.index[network.generators["carrier"] == "large_hydro"]
    if not len(gens):
        return False
    metered = metered_hydro_mw(espeni_path, int(weather_year))
    if metered is None:
        log.warning("Large hydro (historical): ESPENI has no complete %s; site profiles kept", weather_year)
        return False
    fleet = float(network.generators.loc[gens, "p_nom"].sum())
    daily = metered.groupby(metered.index.normalize()).mean()
    snapshots = pd.DatetimeIndex(network.snapshots)
    inflow_pu = pd.Series(daily.reindex(snapshots.normalize()).to_numpy() / fleet, index=snapshots)
    if inflow_pu.isna().any():
        raise ValueError(f"ESPENI metered hydro does not cover every snapshot day of {weather_year}")
    ceiling = min(1.0, float(metered.max()) / fleet)
    _set_generator_series(network, LARGE_HYDRO_INFLOW_SERIES,
                          pd.DataFrame({g: inflow_pu.to_numpy() for g in gens}, index=snapshots))
    p_max_t = network.generators_t.p_max_pu
    network.generators_t["p_max_pu"] = p_max_t.drop(columns=[g for g in gens if g in p_max_t.columns])
    network.generators.loc[gens, "p_max_pu"] = ceiling
    log.info("Large hydro (historical): inflow = ESPENI NPSHYD %s daily metered output on the %.0f MW fleet "
             "(mean %.0f MW over the snapshots, daily %.0f-%.0f MW); ceiling %.3f of p_nom (the year's "
             "metered peak, %.0f MW)", weather_year, fleet, float(inflow_pu.mean() * fleet),
             float(inflow_pu.min() * fleet), float(inflow_pu.max() * fleet), ceiling, float(metered.max()))
    return True
