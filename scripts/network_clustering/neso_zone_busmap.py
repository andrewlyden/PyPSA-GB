"""NESO CP.08 17-zone busmap for ``clustering.method: custom`` (draft, 20 Oct 2026).

User decision (20 Oct 2026): the 2050 zonal model uses NESO's 17 CP.08 economic zones
(``data/CP30/validation_references/neso_cp08_zones.geojson``), zoned exactly as the CP30 balancing
check and the pro rata turn-down re-pricing zone a bus (``scripts.market.neso_zones.bus_zone_map``:
polygon, offshore buses through the network to the nearest on-land bus, foreign buses none,
reconciled against the 11 zone-defining ETYS boundaries). GB AC buses collapse to ``Z1``..``Z17``;
foreign and non-AC buses (GB_H2, gas) stay individual, as in ``boundary_busmap``.

Placement is unchanged: every generator, store, load and link is placed on the full ETYS network
before clustering and moves to its bus's zone. Inter-zonal limits are the secured boundaries
(``transmission.outage_schedule.secured_capability.boundaries`` = ``neso_zones.ZONE_BOUNDARIES``);
the crossing test on zone centroids was checked offline (31 of 31 zone pairs on the 1 Oct network).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULTS = {"zones_file": None, "remote_km": 2.0, "cutoff": 6}


def cluster_buses(network, config: Optional[dict] = None) -> pd.Series:
    """Custom clustering entry point: bus -> ``Z1``..``Z17`` (GB AC) or its own special name.

    Args:
        network: PyPSA network, bus x/y in OSGB36 (cluster_network calls custom functions before its
            WGS84 conversion).
        config: the scenario's clustering config; ``neso_zone_busmap`` sub-block (``zones_file``,
            ``remote_km``, ``cutoff``) and ``secured_capability`` (include/exclude/geometry_file, the
            scenario's own corrections, so the B4 EGL2 include applies).

    Returns:
        busmap Series, bus -> cluster name.
    """
    from pyproj import Transformer

    from scripts.market import neso_zones as nz
    from scripts.network_clustering.boundary_busmap import _special_buses

    cfg = {**DEFAULTS, **dict((config or {}).get("neso_zone_busmap") or {})}
    buses = network.buses.copy()
    if float(buses["x"].max() - buses["x"].min()) <= 1000.0:
        raise ValueError("neso_zone_busmap needs bus x/y in OSGB36 metres")
    tr = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    buses["lon"], buses["lat"] = tr.transform(buses["x"].to_numpy(float), buses["y"].to_numpy(float))
    if "country" not in buses:
        buses["country"] = "GB"
    net = {"buses": buses, "lines": network.lines, "transformers": network.transformers, "links": network.links}
    secured = dict((config or {}).get("secured_capability") or {})
    crossings = nz.boundary_crossings(net, nz.ZONE_BOUNDARIES, secured)
    polys = nz.load_zone_polygons(cfg["zones_file"] or nz.ZONES_FILE)
    bz = nz.bus_zone_map(net, polys, crossings, float(cfg["remote_km"]), int(cfg["cutoff"]))

    special = _special_buses(buses)
    busmap = special.copy()
    gb = special.isna()
    zone = bz["neso_zone"].reindex(buses.index)
    unzoned = gb & zone.isna()
    if unzoned.any():
        # A GB AC bus with no coordinates/zone: keep it individual and say so (should not happen).
        logger.warning("neso_zone_busmap: %d GB AC buses without a zone kept individual: %s",
                       int(unzoned.sum()), list(buses.index[unzoned])[:10])
        busmap[unzoned] = buses.index[unzoned]
    busmap[gb & zone.notna()] = zone[gb & zone.notna()]
    counts = bz.loc[gb[gb].index, "mapping"].astype(str).str.split(":").str[0].value_counts().to_dict()
    logger.info("neso_zone_busmap: %d GB AC buses -> %d zones (%s); %d special buses kept individual",
                int(gb.sum()), busmap[gb].nunique(), counts, int((~gb).sum()))
    missing = sorted(set(nz.ZONES) - set(busmap[gb]))
    if missing:
        raise ValueError(f"neso_zone_busmap: zones with no bus: {missing}")
    return busmap.astype(str)
