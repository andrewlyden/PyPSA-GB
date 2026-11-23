"""Technology and resource-region renewable performance-factor registry.

Performance factors describe physical profile bias (weather reanalysis,
turbine modelling, wakes and normal availability).  They must not depend on a
generator's CfD/ROC/merchant support label.  Support labels are retained for
accounting and validation only.

The registry supports an evidence-backed progression from a GB-wide carrier
factor to a carrier-and-resource-region factor without changing the workflow
interface.  The row-selection order is, from most to least specific:

``carrier + resource_region + year + month`` -> ``carrier + GB + year +
month`` -> matching rows with an unspecified year/month.  The caller's
carrier-wide factor remains a backwards-compatible final fallback.

An optional ``weather_year`` column pins a row to the reanalysis year it was
calibrated against.  A factor fitted so that weather-year-2013 profiles
reproduce a target energy is wrong for any other weather year (on a common
687-site set, raw 2013 onshore CF is 3.8% above the mean of the other eight
profile years, audit 26 Sep 2026), so such a row only
matches a build whose ``renewables_year`` equals it; a blank weather_year
matches every weather year.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Calibration scopes whose factor scales a profile to a published gross energy
#: and may therefore exceed 1 (bounded by GROSS_CALIBRATION_MAX).
GROSS_CALIBRATION_SCOPES = frozenset({"cp30_gross_energy_calibration"})
GROSS_CALIBRATION_MAX = 1.25


REQUIRED_COLUMNS = {"carrier", "resource_region", "factor"}


def resource_region_for_site(carrier: str, site: pd.Series | dict[str, Any]) -> str:
    """Assign a broad physical resource region from WGS84 coordinates.

    These regions are deliberately broad.  They are a routing key for future
    evidence-backed factors, not a claim that each named project is perfectly
    represented by a regional average.  Missing or non-WGS84 coordinates use
    the transparent ``GB`` fallback.
    """
    site = dict(site)
    lat = site.get("lat", site.get("y_coord"))
    lon = site.get("lon", site.get("x_coord"))
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return "GB"
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return "GB"

    if carrier == "wind_offshore":
        if lat >= 56.0:
            return "scottish_offshore"
        if lon < -3.0:
            return "irish_sea"
        return "north_sea"
    if carrier in {"wind_onshore", "solar_pv"}:
        if lat >= 55.5:
            return "scotland"
        if lat >= 53.5:
            return "northern_gb"
        return "southern_gb"
    return "GB"


def load_performance_factor_registry(path: str | Path) -> pd.DataFrame:
    """Load and validate a renewable performance-factor registry."""
    registry = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"Performance-factor registry missing columns: {sorted(missing)}")
    registry = registry.copy()
    registry["carrier"] = registry["carrier"].astype(str).str.strip()
    registry["resource_region"] = registry["resource_region"].fillna("GB").astype(str).str.strip()
    registry["factor"] = pd.to_numeric(registry["factor"], errors="raise")
    # Physical factors (bias, wakes, availability) only derate: (0, 1]. A
    # gross-energy calibration scales a weather profile to a published energy
    # total and may exceed 1 where the profile runs low (CP30 solar on 2013
    # ERA5: 1.0243, 27 Sep 2026); it is bounded at GROSS_CALIBRATION_MAX so a
    # units slip cannot pass. p_max_pu is still clipped to 1 per hour.
    scope = registry.get("calibration_scope", pd.Series("", index=registry.index)).fillna("").astype(str)
    upper = np.where(scope.isin(GROSS_CALIBRATION_SCOPES), GROSS_CALIBRATION_MAX, 1.0)
    if registry["factor"].le(0).any() or (registry["factor"] > upper).any():
        raise ValueError(
            "Performance-factor registry values must be in (0, 1], or (0, "
            f"{GROSS_CALIBRATION_MAX}] for scopes {sorted(GROSS_CALIBRATION_SCOPES)}"
        )
    for column in ("calendar_year", "calendar_month", "weather_year"):
        if column not in registry:
            registry[column] = pd.NA
        registry[column] = pd.to_numeric(registry[column], errors="coerce").astype("Int64")
    if not registry["calendar_month"].dropna().between(1, 12).all():
        raise ValueError("calendar_month must be between 1 and 12")
    return registry


def select_performance_factor(
    registry: pd.DataFrame | None,
    carrier: str,
    resource_region: str,
    calendar_year: int | None,
    calendar_month: int | None,
    fallback_factor: float = 1.0,
    weather_year: int | None = None,
) -> tuple[float, str]:
    """Select the most-specific registry factor or return the carrier fallback.

    ``weather_year`` is the build's ``renewables_year``. Rows that declare a
    weather year match only that year; with ``weather_year=None`` they never
    match, so a weather-specific calibration cannot leak into a build whose
    weather year is unknown.
    """
    if registry is None or registry.empty:
        return float(fallback_factor), "carrier_config_fallback"
    candidates = registry.loc[registry["carrier"].eq(str(carrier))].copy()
    if candidates.empty:
        return float(fallback_factor), "carrier_config_fallback"
    if "weather_year" in candidates.columns:
        pinned = candidates["weather_year"]
        if weather_year is None:
            candidates = candidates.loc[pinned.isna()]
        else:
            candidates = candidates.loc[pinned.isna() | pinned.eq(int(weather_year)).fillna(False)]
    else:
        candidates = candidates.assign(weather_year=pd.NA)
    candidates = candidates.loc[candidates["resource_region"].isin([resource_region, "GB"])]
    if calendar_year is not None:
        candidates = candidates.loc[
            candidates["calendar_year"].isna() | candidates["calendar_year"].eq(int(calendar_year))
        ]
    else:
        candidates = candidates.loc[candidates["calendar_year"].isna()]
    if calendar_month is not None:
        candidates = candidates.loc[
            candidates["calendar_month"].isna() | candidates["calendar_month"].eq(int(calendar_month))
        ]
    else:
        candidates = candidates.loc[candidates["calendar_month"].isna()]
    if candidates.empty:
        return float(fallback_factor), "carrier_config_fallback"

    candidates["_specificity"] = (
        candidates["resource_region"].eq(resource_region).astype(int) * 8
        + candidates["weather_year"].notna().astype(int) * 4
        + candidates["calendar_year"].notna().astype(int) * 2
        + candidates["calendar_month"].notna().astype(int)
    )
    best = candidates.loc[candidates["_specificity"].eq(candidates["_specificity"].max())]
    if len(best) != 1:
        details = best[["carrier", "resource_region", "weather_year", "calendar_year", "calendar_month"]]
        raise ValueError(f"Ambiguous performance-factor rows for {carrier}/{resource_region}: {details.to_dict('records')}")
    row = best.iloc[0]
    return float(row["factor"]), "registry"
