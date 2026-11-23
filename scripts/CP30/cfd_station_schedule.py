"""Named CfD stations carved out of CP30's anonymous proxies (rebuild batch, 14 Oct 2026).

CP30 holds its waste (2,481 MW in FFR 2030: ES1 Waste, Waste CHP and
distributed Waste) and its "Other Renewables" (1,071 MW, mapped to biogas) as
anonymous FES proxy rows. Three live or awarded CfDs sit inside those totals:
K3 CHP at Kemsley (AR1 energy from waste, 45 MW), Energy Works Hull (AR1
advanced conversion, 25 MW) and the AR5 geothermal plants in Cornwall (12 MW).
Without named units the bill's CfD line misses them.

``apply_cfd_station_schedule`` replaces that capacity of the carrier's proxy
with named units at the nearest bus to the site, so each carrier's CP30 total is
unchanged. The units carry ``support_type`` CfD, their round, ``cfd_id`` and
``cfd_technology`` (the strike-table row: energy_from_waste, advanced_conversion,
geothermal), and ``support_label_declared``, so the projected CfD/merchant split
and the RO register share of the waste proxy (applied to the proxy rows only)
leave them alone. The geothermal unit stays on the biogas carrier (CP30 has no
geothermal line); only its strike differs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

CFD_STATION_SOURCE = "CP30_cfd_station_schedule"
REQUIRED_COLUMNS = ("station_name", "carrier", "capacity_mw", "lat", "lon", "support_type", "cfd_round",
                    "cfd_technology", "cfd_ids", "source")


def load_cfd_station_schedule(path) -> pd.DataFrame:
    """The declared CfD station schedule (``data/market/CP30/cfd_station_schedule_2030.csv``)."""
    table = pd.read_csv(path)
    if missing := set(REQUIRED_COLUMNS).difference(table.columns):
        raise ValueError(f"CfD station schedule {path} is missing: {sorted(missing)}")
    table["capacity_mw"] = pd.to_numeric(table["capacity_mw"], errors="raise")
    for column in ("lat", "lon"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table["station_name"].duplicated().any():
        raise ValueError(f"CfD station schedule {path} lists a station twice")
    if table["capacity_mw"].le(0).any():
        raise ValueError(f"CfD station schedule {path} has a non-positive capacity")
    return table


def nearest_buses(schedule: pd.DataFrame, network, land_boundary_file="default") -> pd.Series:
    """Nearest bus to each station's coordinates, avoiding buses no onshore plant may sit on.

    The buses ``spatial_utils.generation_placement_exclusions`` lists (no AC
    branch, load-free AC islands, offshore platforms and OFTO export
    substations, misplaced coordinates) are skipped, as for onshore renewables.
    ``land_boundary_file=None`` drops the offshore test (toy networks).
    """
    from scripts.utilities.spatial_utils import (
        DEFAULT_LAND_BOUNDARY_FILE,
        generation_placement_exclusions,
        map_sites_to_buses,
        reassign_excluded_placements,
    )

    sites = schedule[["station_name", "lat", "lon"]].rename(columns={"station_name": "site_name"}).copy()
    sites["technology"] = schedule["carrier"].to_numpy()
    sites["bus"] = None
    mapped = map_sites_to_buses(network, sites, method="nearest", lat_col="lat", lon_col="lon",
                                max_distance_km=200.0)
    if land_boundary_file == "default":
        land_boundary_file = DEFAULT_LAND_BOUNDARY_FILE
    exclusions = generation_placement_exclusions(network, land_boundary_file=land_boundary_file)
    mapped, _ = reassign_excluded_placements(mapped, network, exclusions)
    return pd.Series(mapped["bus"].to_numpy(), index=schedule.index)


def apply_cfd_station_schedule(scaled: pd.DataFrame, path, network, logger: logging.Logger | None = None,
                               carrier_column: str = "fuel_type",
                               protected_sources=(), land_boundary_file="default") -> pd.DataFrame:
    """Carve the scheduled CfD stations out of their carriers' proxy rows; totals unchanged.

    The proxy rows of each carrier (those not placed by another station schedule,
    ``protected_sources``) are scaled down by the named MW in proportion, and one
    row per station is added at its nearest bus. Refuses a carrier whose proxy is
    smaller than its named stations. ``land_boundary_file`` is passed to
    :func:`nearest_buses`.
    """
    logger = logger or logging.getLogger(__name__)
    schedule = load_cfd_station_schedule(path)
    schedule["bus"] = nearest_buses(schedule, network, land_boundary_file)
    if schedule["bus"].isna().any():
        raise ValueError(f"CfD station schedule: no bus for {schedule.loc[schedule['bus'].isna(), 'station_name'].tolist()}")
    result = scaled.copy()
    sources = result.get("data_source", pd.Series("", index=result.index)).fillna("").astype(str)
    rows = []
    for carrier, stations in schedule.groupby("carrier", sort=False):
        proxy = result[carrier_column].eq(carrier) & ~sources.isin(list(protected_sources)) \
            & pd.to_numeric(result["capacity_mw"], errors="coerce").gt(0)
        proxy_mw = float(result.loc[proxy, "capacity_mw"].sum())
        named_mw = float(stations["capacity_mw"].sum())
        if proxy_mw <= named_mw + 1e-6:
            raise ValueError(f"CfD station schedule: {carrier} proxy holds {proxy_mw:,.1f} MW, less than the "
                             f"{named_mw:,.1f} MW of named stations")
        result.loc[proxy, "capacity_mw"] = result.loc[proxy, "capacity_mw"] * (proxy_mw - named_mw) / proxy_mw
        template = result.loc[proxy].iloc[0].to_dict()
        for _, station in stations.iterrows():
            record = {key: np.nan for key in template}
            for key in ("cp30_pathway", "cp30_target_mw", "cp30_source_type", "cp30_source_subtype"):
                if key in template:
                    record[key] = template[key]
            record.update({
                carrier_column: carrier, "technology": carrier, "capacity_mw": float(station["capacity_mw"]),
                "station_name": station["station_name"], "source_site_name": station["station_name"],
                "bus": station["bus"], "lat": float(station["lat"]), "lon": float(station["lon"]), "gsp": np.nan,
                "data_source": CFD_STATION_SOURCE, "spatial_proxy_source": "nearest bus to the site",
                "support_type": station["support_type"], "cfd_round": station["cfd_round"],
                "cfd_id": station["cfd_ids"], "cfd_technology": station["cfd_technology"],
                "support_label_declared": True,
            })
            rows.append(record)
        logger.info("CfD station schedule %s: %d named stations, %.1f MW carved out of the %.1f MW proxy (%s)",
                    carrier, len(stations), named_mw, proxy_mw,
                    "; ".join(f"{s.station_name} {s.capacity_mw:g} MW at {s.bus}" for s in stations.itertuples()))
    return pd.concat([result, pd.DataFrame(rows)], ignore_index=True, sort=False)
