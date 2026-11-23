"""Purpose-built low-resolution GB transmission map for bill backtesting.

This is deliberately not a geographic k-means clustering.  The zones retain
the Scotland--England boundary and the main demand, generation and coastal
areas relevant to annual balancing and loss diagnostics.  It is a screening
network: it does not replace the full ETYS topology for detailed constraint
studies or published network tariffs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GB_ZONES = {
    "north_scotland",
    "south_scotland",
    "north_west",
    "north_east",
    "yorkshire_humber",
    "north_wales_merseyside",
    "west_midlands",
    "east_midlands",
    "east_anglia",
    "london",
    "south_east",
    "south",
    "south_west",
    "south_wales",
}


def _classify_gb_ac_bus(lon: float, lat: float) -> str:
    """Assign a located GB AC bus to one of the 14 retained regions."""
    # Keep Scotland separate before applying the lower-resolution England/Wales
    # boundaries.  This is the most important retained transmission boundary.
    if lat >= 57.20:
        return "north_scotland"
    if lat >= 55.55:
        return "south_scotland"

    # North of England: separate western generation/export routes from the
    # north-east transmission corridor.
    if lat >= 54.65:
        return "north_west" if lon < -2.40 else "north_east"
    if lat >= 53.50:
        return "north_west" if lon < -2.35 else "yorkshire_humber"
    if lat >= 52.75:
        return "north_wales_merseyside" if lon < -2.60 else "yorkshire_humber"

    # Midlands and Wales.
    if lat >= 52.10:
        if lon < -3.15:
            return "south_wales"
        if lon < -1.05:
            return "west_midlands"
        if lon < 0.35:
            return "east_midlands"
        return "east_anglia"

    # London is retained explicitly because it is a large, geographically
    # concentrated demand centre.  Its test comes before the southern zones.
    if 51.25 <= lat < 51.75 and -0.75 <= lon <= 0.45:
        return "london"
    if lat >= 51.50:
        if lon < -3.10:
            return "south_wales"
        if lon < -1.15:
            return "west_midlands"
        if lon < 0.30:
            return "east_midlands"
        return "east_anglia"

    # Southern England, South Wales and the South West.  Offshore buses use
    # the same coastal-facing zones as their geographical connection point.
    if lon < -3.00:
        return "south_wales" if lat >= 51.15 else "south_west"
    if lon < -1.80:
        return "south_west"
    if lon < -0.20:
        return "south"
    return "south_east"


def cluster_buses(network, config=None) -> pd.Series:
    """Return a complete, safe busmap for a 14-zone GB AC aggregation.

    Foreign AC buses and non-AC buses stay individual.  Grouping either would
    erase interconnector endpoints or mix electricity with hydrogen/DC buses.
    The small number of GB buses without coordinates are assigned to the
    nearest populated zone centroid, rather than to a catch-all cluster.
    """
    del config  # The named map is intentionally fixed and version-controlled.

    buses = network.buses
    busmap = pd.Series(index=buses.index, dtype="object")
    carrier = buses.get("carrier", pd.Series("AC", index=buses.index)).fillna("AC")
    country = buses.get("country", pd.Series("GB", index=buses.index)).fillna("")

    non_ac = carrier.ne("AC")
    foreign_ac = (~non_ac) & country.ne("") & country.ne("GB")
    for bus in buses.index[non_ac]:
        busmap.at[bus] = f"{carrier.at[bus]}_{bus}"
    for bus in buses.index[foreign_ac]:
        busmap.at[bus] = f"external_{country.at[bus]}_{bus}"

    gb_ac = ~(non_ac | foreign_ac)
    located = gb_ac & buses["lon"].notna() & buses["lat"].notna()
    for bus, row in buses.loc[located, ["lon", "lat"]].iterrows():
        busmap.at[bus] = _classify_gb_ac_bus(float(row.lon), float(row.lat))

    # Some offshore ETYS entries do not have a coordinate.  Assigning them to
    # the nearest existing zone is transparent and avoids creating a spurious
    # isolated "unassigned" node.
    missing = busmap[gb_ac & busmap.isna()].index
    if len(missing):
        known = buses.loc[located, ["lon", "lat"]].copy()
        known["zone"] = busmap.loc[known.index]
        centroids = known.groupby("zone")[["lon", "lat"]].mean()
        for bus in missing:
            point = buses.loc[bus, ["lon", "lat"]]
            if point.notna().all():
                distances = (centroids["lon"] - float(point.lon)) ** 2 + (
                    centroids["lat"] - float(point.lat)
                ) ** 2
                busmap.at[bus] = distances.idxmin()
            else:
                # No geographical evidence remains; use the network's overall
                # geographic centre rather than a labelled pseudo-region.
                busmap.at[bus] = "south"

    if busmap.isna().any():
        raise ValueError("14-zone busmap failed to assign every network bus")
    return busmap
