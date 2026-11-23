"""Regional weather profiles for proxy renewable units (15 Oct 2026).

A renewable unit with no site profile of its own (a CP30/FES proxy increment)
used to take ONE GB-wide capacity-weighted fleet-average profile per carrier,
so proxy offshore at Bramford, Blackhillock and Heysham produced identical
hourly output. With ``renewables.proxy_profile_basis: regional`` each proxy
unit instead takes the capacity-weighted average profile of the named sites
(units that matched a site profile column) of the same carrier in its own area.
With ``proxy_profile_members: all_sites`` (the FES GSP-block build, which has no
named units) the sites are every site profile column, located by the site's own
coordinates, and a unit without a region takes the region of its coordinates.
The areas:

* onshore wind and solar: the unit's CP30 ES5 region;
* offshore wind: the unit's sea basin (``offshore_basin``, from the unit's own
  coordinates, which for a proxy are the FES proxy geography the landing map
  preserves), then the CP30 region of its landing bus.

A group with fewer than ``min_sites`` named sites or ``min_mw`` of their
capacity falls back to the next level and finally to the GB fleet average
(the old profile), and every fallback is logged.

The weighting is the national average's: summed site output over summed site
capacity (the capacity each column was built on, its peak where unknown).

``preserve_carrier_energy`` (default true) multiplies every regional profile of
a carrier by one scalar so the carrier's proxy fleet keeps the energy it had on
the national profile, weighted by each unit's p_nom and performance factor.
The performance-factor registry is calibrated per carrier and resource region
to CP30's available energy with the national proxy profile, so without this
the calibrated carrier totals would drift with the regional mix. The regional
shape and the regional differences in capacity factor are kept; only the
carrier total is pinned.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROXY_PROFILE_BASES = ("national", "regional")
#: Which site profiles make up an area's group. ``matched_units``: the build's own units that matched a site
#: profile column (the CP30 brownfield build, whose named fleet is placed in the network). ``all_sites``: every
#: site profile column of the carrier, located by its site's coordinates, for builds with no named units of their
#: own (the FES GSP-block build, whose units are all FES capacity blocks; issue #51).
PROXY_PROFILE_MEMBERS = ("matched_units", "all_sites")
#: GSP polygons and the GSP group -> region map that give a point its region under ``all_sites`` (the same
#: eight CP30 ES5 regions the brownfield build uses; ``spatial_allocation.load_gsp_region_map``).
DEFAULT_REGION_POLYGON_FILE = "data/network/GSP/GSP_regions_27700_20250109.geojson"
DEFAULT_REGION_GROUP_FILE = "data/network/ETYS/cp30_es5_region_by_gsp_group.csv"
DEFAULT_PROXY_PROFILE_CARRIERS = ("wind_onshore", "wind_offshore", "solar_pv")
DEFAULT_MIN_SITES = 3
DEFAULT_MIN_MW = 300.0
NATIONAL_KEY = ("national", "GB")

#: Offshore sea basins on WGS84 site coordinates, first match wins. Chosen from
#: where the build's named offshore fleet sits (15 Oct 2026, see the offline
#: check): Moray Firth and Aberdeen (Beatrice, Moray East/West, Hywind,
#: Kincardine, EOWDC) from 56.8 N; the Forth and Tay (Seagreen, Inch Cape,
#: Neart na Gaoithe, Methil) from 55.8 N east of 4 W; the west coast and Irish
#: Sea (Walney, Barrow, Ormonde, West of Duddon Sands, Burbo, Gwynt y Mor,
#: Rhyl Flats, Robin Rigg, Hunterston) everything else west of 2.6 W; Humber,
#: Dogger and Teesside (Hornsea, Dogger Bank, Sofia, Triton Knoll, Westermost
#: Rough, Humber Gateway, Teesside) from 53.3 N; the Wash and East Anglia
#: (Race Bank, Dudgeon, Sheringham Shoal, Lincs, East Anglia ONE/THREE,
#: Galloper, Greater Gabbard) from 51.95 N; the Thames estuary and south coast
#: (London Array, Thanet, Kentish Flats, Gunfleet, Rampion) below that.
OFFSHORE_BASINS: Tuple[Tuple[str, float, float, float, float], ...] = (
    # name, lat_min, lat_max, lon_min, lon_max  (half-open: min <= x < max)
    ("moray_aberdeen", 56.8, 90.0, -4.5, 90.0),
    ("forth_tay", 55.8, 56.8, -4.0, 90.0),
    ("west_coast_irish_sea", -90.0, 90.0, -180.0, -2.6),
    ("humber_dogger_teesside", 53.3, 90.0, -2.6, 90.0),
    ("wash_east_anglia", 51.95, 53.3, -2.6, 90.0),
    ("thames_south_coast", -90.0, 51.95, -2.6, 90.0),
)


def offshore_basin(lat, lon) -> Optional[str]:
    """Sea basin of an offshore site from its WGS84 coordinates (None when unknown)."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lat) and np.isfinite(lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    for name, lat_min, lat_max, lon_min, lon_max in OFFSHORE_BASINS:
        if lat_min <= lat < lat_max and lon_min <= lon < lon_max:
            return name
    return None


def area_keys(site, carrier: str) -> List[Tuple[str, str]]:
    """The (level, key) chain a unit's profile group is chosen from, most local first."""
    keys: List[Tuple[str, str]] = []
    lat = site.get("lat", site.get("y_coord")) if hasattr(site, "get") else None
    lon = site.get("lon", site.get("x_coord")) if hasattr(site, "get") else None
    if carrier == "wind_offshore":
        basin = offshore_basin(lat, lon)
        if basin:
            keys.append(("basin", basin))
    region = site.get("cp30_region") if hasattr(site, "get") else None
    if region is not None and not (isinstance(region, float) and np.isnan(region)) and str(region).strip():
        keys.append(("region", str(region).strip()))
    keys.append(NATIONAL_KEY)
    return keys


def proxy_profile_settings(config: Optional[dict]) -> dict:
    """Validated ``renewables`` settings for the proxy profile basis."""
    config = config or {}
    basis = str(config.get("proxy_profile_basis", "national")).strip()
    if basis not in PROXY_PROFILE_BASES:
        raise ValueError(f"renewables.proxy_profile_basis must be one of {PROXY_PROFILE_BASES}, got {basis!r}")
    members = str(config.get("proxy_profile_members", "matched_units")).strip()
    if members not in PROXY_PROFILE_MEMBERS:
        raise ValueError(f"renewables.proxy_profile_members must be one of {PROXY_PROFILE_MEMBERS}, got {members!r}")
    return {
        "basis": basis,
        "members": members,
        "region_polygon_file": str(config.get("proxy_profile_region_polygon_file") or DEFAULT_REGION_POLYGON_FILE),
        "region_group_file": str(config.get("proxy_profile_region_group_file") or DEFAULT_REGION_GROUP_FILE),
        "carriers": tuple(config.get("proxy_profile_carriers") or DEFAULT_PROXY_PROFILE_CARRIERS),
        "min_sites": int(config.get("proxy_profile_min_sites", DEFAULT_MIN_SITES)),
        "min_mw": float(config.get("proxy_profile_min_mw", DEFAULT_MIN_MW)),
        "preserve_carrier_energy": bool(config.get("proxy_profile_preserve_carrier_energy", True)),
    }


def regions_of_points(lat, lon, polygon_file: str, group_file: str) -> pd.Series:
    """Region of WGS84 points: the GSP polygon each falls in, else the nearest (offshore sites, islands).

    The rule of ``spatial_allocation.regions_of_xy``, on points given in degrees. NaN where a point has no
    coordinates.
    """
    from pyproj import Transformer

    from scripts.CP30.spatial_allocation import load_gsp_region_map, regions_of_xy

    lat = pd.to_numeric(pd.Series(lat), errors="coerce")
    lon = pd.to_numeric(pd.Series(lon, index=lat.index), errors="coerce")
    x = pd.Series(np.nan, index=lat.index)
    y = pd.Series(np.nan, index=lat.index)
    ok = lat.between(-90, 90) & lon.between(-180, 180)
    if ok.any():
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        xs, ys = transformer.transform(lon[ok].to_numpy(), lat[ok].to_numpy())
        x[ok], y[ok] = xs, ys
    polygons, group_region = load_gsp_region_map(polygon_file, group_file)
    return regions_of_xy(x, y, polygons, group_region)["cp30_region"]


def site_members(carrier: str, locations: pd.DataFrame, columns: Iterable[str], regions: pd.Series) -> pd.DataFrame:
    """Every located site profile column as a group member (``proxy_profile_members: all_sites``).

    ``locations`` is indexed by profile column with ``lat``, ``lon`` and ``capacity_mw``; ``regions`` is each
    column's region on the same index. Columns without a location or capacity are left out of the area groups
    (they stay in the national average). Returns the ``members`` frame of ``build_regional_proxy_profiles``.
    """
    present = [c for c in columns if c in locations.index]
    rows = []
    for column in present:
        site = locations.loc[column]
        capacity = float(site["capacity_mw"])
        if not capacity > 0:
            continue
        keys = area_keys({"lat": site["lat"], "lon": site["lon"], "cp30_region": regions.get(column)}, carrier)
        rows.append({"column": column, "capacity": capacity, "keys": keys})
    return pd.DataFrame(rows, columns=["column", "capacity", "keys"])


def _capacity_normalised_mean(profile_df: pd.DataFrame, columns: Iterable[str],
                              capacities: Optional[Dict[str, float]]) -> pd.Series:
    """Summed output over summed site capacity (peak where the capacity is unknown)."""
    columns = list(columns)
    frame = profile_df[columns]
    peaks = frame.max()
    known = pd.Series([(capacities or {}).get(str(c), np.nan) for c in columns], index=columns, dtype=float)
    site_capacity = known.where(known.gt(0), peaks)
    total = float(site_capacity.sum())
    if total <= 0:
        return frame.mean(axis=1)
    return frame.sum(axis=1) / total


def build_regional_proxy_profiles(
    carrier: str,
    profile_df: pd.DataFrame,
    national: pd.Series,
    members: pd.DataFrame,
    proxies: pd.DataFrame,
    capacities: Optional[Dict[str, float]],
    settings: dict,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[object, pd.Series], pd.DataFrame, pd.DataFrame, float]:
    """Choose and build each proxy unit's area profile.

    ``members``: one row per named unit that matched a profile column, with
    ``column``, ``capacity`` (the column's site capacity) and ``keys`` (its
    ``area_keys`` chain). ``proxies``: one row per unmatched unit (index = the
    unit's row id) with ``keys`` and ``weight`` (p_nom x performance factor).
    ``national`` is the carrier's GB fleet-average profile (all columns).

    Returns ({row id: profile}, per-unit assignment table, per-group table,
    energy scalar applied).
    """
    logger = logger or logging.getLogger(__name__)
    min_sites, min_mw = settings["min_sites"], settings["min_mw"]
    # Site columns and their capacity per (level, key); a column shared by
    # several units (support tranches) counts once.
    group_columns: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, row in members.iterrows():
        for key in row["keys"]:
            if key == NATIONAL_KEY:
                continue
            group_columns.setdefault(key, {})[str(row["column"])] = float(row["capacity"])

    cache: Dict[Tuple[str, str], pd.Series] = {NATIONAL_KEY: national}
    stats_rows = {}
    assignment_rows = []
    chosen: Dict[object, Tuple[str, str]] = {}
    logged_fallbacks = set()
    for idx, row in proxies.iterrows():
        keys = list(row["keys"])
        picked = NATIONAL_KEY
        skipped = []
        for key in keys:
            if key == NATIONAL_KEY:
                break
            cols = group_columns.get(key, {})
            mw = float(sum(cols.values()))
            if len(cols) >= min_sites and mw >= min_mw:
                picked = key
                break
            skipped.append(f"{key[0]}:{key[1]} ({len(cols)} sites, {mw:,.0f} MW)")
        if skipped and (carrier, tuple(keys)) not in logged_fallbacks:
            logged_fallbacks.add((carrier, tuple(keys)))
            logger.info("Proxy %s profile: %s too thin (min %d sites, %.0f MW); using %s:%s",
                        carrier, "; ".join(skipped), min_sites, min_mw, picked[0], picked[1])
        if picked not in cache:
            cache[picked] = _capacity_normalised_mean(profile_df, group_columns[picked], capacities)
        chosen[idx] = picked
        assignment_rows.append({"row": idx, "carrier": carrier, "level": picked[0], "group": picked[1],
                                "fallback_from": "; ".join(skipped), "weight": float(row["weight"])})
    assignments = pd.DataFrame(assignment_rows, columns=["row", "carrier", "level", "group", "fallback_from",
                                                         "weight"])

    # One energy scalar per carrier keeps the proxy fleet's energy on the national basis.
    scale = 1.0
    if settings.get("preserve_carrier_energy", True) and len(assignments):
        # Units left on the national profile keep it unscaled; the scalar is set
        # by the units on a regional profile alone.
        mean_cf = {key: float(series.mean()) for key, series in cache.items()}
        regional = [(float(w), chosen[r]) for w, r in zip(assignments["weight"], assignments["row"])
                    if chosen[r] != NATIONAL_KEY]
        regional_energy = sum(w * mean_cf[key] for w, key in regional)
        national_energy = sum(w for w, _ in regional) * mean_cf[NATIONAL_KEY]
        if regional_energy > 0:
            scale = national_energy / regional_energy
        logger.info("Proxy %s profiles: energy scalar %.4f keeps the proxy fleet's energy on the national basis",
                    carrier, scale)

    national_mean = float(national.mean())
    for key, series in cache.items():
        used = assignments.loc[[chosen[r] == key for r in assignments["row"]]]
        cols = group_columns.get(key, {})
        stats_rows[key] = {
            "carrier": carrier, "level": key[0], "group": key[1],
            "named_sites": len(cols) if key != NATIONAL_KEY else int(profile_df.shape[1]),
            "named_mw": float(sum(cols.values())) if key != NATIONAL_KEY else np.nan,
            "proxy_units": int(len(used)), "proxy_weight_mw": float(used["weight"].sum()),
            "cf_raw": float(series.mean()),
            "cf_scaled": float(series.mean()) * (scale if key != NATIONAL_KEY else 1.0),
            "cf_national": national_mean,
            "corr_with_national": float(series.corr(national)) if key != NATIONAL_KEY else 1.0,
            "energy_scalar": scale,
        }
    groups = pd.DataFrame(list(stats_rows.values()))
    profiles = {r: cache[key] * (scale if key != NATIONAL_KEY else 1.0) for r, key in chosen.items()}
    # Regional profiles scaled above 1 are clipped later with the unit's factor; record the headroom.
    return profiles, assignments, groups, scale
