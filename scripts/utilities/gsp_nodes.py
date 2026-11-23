"""The network node where a location's own GSP takes its demand (15 Oct 2026).

Distribution-connected plant (below its licence area's transmission threshold)
connects inside a GSP's distribution network, so in a nodal model it belongs
at that GSP's node, the bus that also carries the GSP's demand. Nearest-bus
placement instead put rooftop and field solar on whatever substation was
closest at any voltage (400 kV nodes and substations with no demand at all).

``GspNodeLocator`` joins a WGS84 point to NESO's GSP region polygon
(``GSPs``, ``GSPGroup``), maps the polygon's GSP IDs through
``GB_network.xlsx`` ``Dem_per_node`` (the table the demand build uses) and
resolves them on the network with the demand build's own rule
(``scripts.demand.load.resolve_gsp_node_shares``: listed nodes present,
then the same substation, then the nearest demand node), restricted to buses
that carry load. A GSP split over several nodes takes the node with the
largest demand share. A polygon whose GSP IDs have no ``Dem_per_node`` row
falls back to the nearest load bus to the polygon's representative point.

Used by the CP30 brownfield renewables (distribution-size named onshore and
solar), the storage build (distribution-size batteries) and the CP30 locational
validation (which reads the same polygon join and thresholds)::

    locator = GspNodeLocator(network, polygon_file)      # build once per network
    node = locator.node_for(lat, lon)                     # one site
    placed = locator.assign(sites)                        # adds gsp_node, gsp_group, gsp_ids
    place_distributed_at_gsp_nodes(sites, locator)        # move distribution-size rows

``distribution_threshold_mw`` gives the licence-area transmission threshold
of a GSP group (SHEPD ``_P`` 10 MW, SPD ``_N`` 30 MW, England and Wales 100 MW);
``nearest_gsp_polygon`` is the point-to-polygon join every caller shares.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_POLYGON_FILE = "data/network/GSP/GSP_regions_27700_20250109.geojson"
DEFAULT_DEM_PER_NODE_FILE = "data/network/ETYS/GB_network.xlsx"

#: Licence-area transmission thresholds (MW) by GSP group: SHET (North of
#: Scotland, SHEPD group _P) 10 MW, SPT (South of Scotland, SPD group _N)
#: 30 MW, NGET (England and Wales) 100 MW. Plant below its threshold is
#: distribution-connected.
DEFAULT_DISTRIBUTION_THRESHOLDS_MW = {"_P": 10.0, "_N": 30.0, "default": 100.0}

#: ``data_source`` of a row :func:`place_distributed_at_gsp_nodes` put on its GSP's node.
DISTRIBUTED_GSP_NODE_SOURCE = "CP30_distributed_gsp_node"
#: Renewable rows placed on their GSP's demand node by rule (the distributed named fleet, FES
#: small-scale solar held there). Later placement steps keep them at that GSP (16 Oct 2026):
#: placement exclusions do not move them (the node carries the GSP's demand, whatever its
#: coordinates say) and weak-bus relief moves them only to another busbar of the same substation.
GSP_NODE_SOURCES = (DISTRIBUTED_GSP_NODE_SOURCE, "CP30_small_scale_solar_gsp")

logger = logging.getLogger(__name__)


def distribution_threshold_mw(gsp_group, thresholds: Optional[dict] = None) -> float:
    """Transmission threshold (MW) for a GSP group's licence area."""
    table = {**DEFAULT_DISTRIBUTION_THRESHOLDS_MW, **(thresholds or {})}
    return float(table.get(str(gsp_group), table["default"]))


def load_gsp_polygons(polygon_file=DEFAULT_POLYGON_FILE):
    """NESO's GSP region polygons in EPSG:27700, without the unassigned (no ``GSPGroup``) ones."""
    import geopandas as gpd

    path = Path(polygon_file)
    if not path.is_file():
        raise FileNotFoundError(f"GSP polygon file not found: {path}")
    return prepare_gsp_polygons(gpd.read_file(path))


def prepare_gsp_polygons(polygons):
    """Validate GSP polygons and return them in EPSG:27700 with a fresh index (no NaN ``GSPGroup``)."""
    if "GSPGroup" not in polygons.columns:
        raise ValueError("GSP polygons lack the GSPGroup property")
    # NESO's two Off_NETS polygons have no licence area: never a point's polygon.
    polygons = polygons[polygons["GSPGroup"].notna()].reset_index(drop=True)
    if polygons.crs is None:
        polygons = polygons.set_crs("EPSG:27700")
    return polygons.to_crs("EPSG:27700")


def nearest_gsp_polygon(polygons, lat, lon) -> pd.Series:
    """Positional index of the GSP polygon each WGS84 point falls in (the nearest one off the coast).

    ``polygons`` come from :func:`prepare_gsp_polygons`. Points without both
    coordinates get NaN. Returns a Series on the index of ``lat``.
    """
    import geopandas as gpd

    lat = pd.to_numeric(pd.Series(lat), errors="coerce")
    lon = pd.to_numeric(pd.Series(lon, index=lat.index), errors="coerce")
    out = pd.Series(np.nan, index=lat.index, dtype=float)
    ok = lat.notna() & lon.notna()
    if not ok.any():
        return out
    points = gpd.GeoDataFrame({"_row": lat.index[ok]}, geometry=gpd.points_from_xy(lon[ok], lat[ok]),
                              crs="EPSG:4326").to_crs("EPSG:27700")
    joined = gpd.sjoin_nearest(points, polygons[["geometry"]], how="left")
    joined = joined[~joined["_row"].duplicated()].set_index("_row")["index_right"]
    out.loc[joined.index] = joined.to_numpy(dtype=float)
    return out


class GspNodeLocator:
    """Locate the demand node of the GSP polygon a WGS84 point falls in."""

    def __init__(self, network, polygon_file=DEFAULT_POLYGON_FILE,
                 dem_per_node_file=DEFAULT_DEM_PER_NODE_FILE, log: Optional[logging.Logger] = None,
                 polygons=None, dem_per_node: Optional[pd.DataFrame] = None):
        self.log = log or logger
        self.network = network
        self.polygons = (load_gsp_polygons(polygon_file) if polygons is None
                         else prepare_gsp_polygons(polygons))
        if "GSPs" not in self.polygons.columns:
            raise ValueError("GSP polygons lack the GSPs property (the GSP IDs of each polygon)")
        if dem_per_node is None:
            dem_per_node = pd.read_excel(dem_per_node_file, sheet_name="Dem_per_node")
        self.table = self._polygon_nodes(dem_per_node)

    def _load_buses(self) -> list:
        loads = getattr(self.network, "loads", None)
        if loads is not None and len(loads):
            buses = sorted(set(loads["bus"].astype(str)) & set(self.network.buses.index))
            if buses:
                return buses
        return list(self.network.buses.index)

    def _polygon_nodes(self, dem_per_node: pd.DataFrame) -> pd.DataFrame:
        from pyproj import Transformer
        from scripts.demand.load import resolve_gsp_node_shares

        weight_col = next(c for c in dem_per_node.columns if str(c).startswith("Dem as %"))
        rows: Dict[str, list] = {}
        for gsp, node, weight in zip(dem_per_node["GSP Id"].astype(str).str.strip(),
                                     dem_per_node["Node Id"].astype(str).str.strip(),
                                     pd.to_numeric(dem_per_node[weight_col], errors="coerce").fillna(0.0)):
            if gsp and node and weight > 0:
                rows.setdefault(gsp, []).append((node, float(weight)))
        load_buses = self._load_buses()
        to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        bus_xy = self.network.buses.loc[load_buses, ["x", "y"]].astype(float).dropna()
        out = []
        for i, poly in self.polygons.iterrows():
            ids = [g.strip() for g in str(poly["GSPs"]).split("|") if g.strip()]
            mappings = [m for g in ids for m in rows.get(g, [])]
            point = poly.geometry.representative_point()
            lon, lat = to_wgs.transform(point.x, point.y)
            node, method = None, "unresolved"
            if mappings:
                shares, method = resolve_gsp_node_shares(mappings, self.network, gsp_lonlat=(lon, lat),
                                                         candidate_buses=load_buses)
                if shares:
                    node = max(shares, key=lambda s: s[1])[0]
            if node is None and len(bus_xy):
                # No Dem_per_node row: the load bus nearest the polygon (network x/y are OSGB metres).
                d = np.hypot(bus_xy["x"] - point.x, bus_xy["y"] - point.y)
                node, method = str(d.idxmin()), "nearest_load_bus"
            out.append({"polygon": i, "gsp_ids": "|".join(ids), "gsp_group": poly["GSPGroup"],
                        "gsp_node": node, "method": method})
        table = pd.DataFrame(out).set_index("polygon")
        counts = table["method"].value_counts().to_dict()
        self.log.info("GSP node locator: %d GSP polygons resolved to demand nodes (%s)", len(table), counts)
        return table

    def assign(self, sites: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
        """``sites`` with ``gsp_node``, ``gsp_group`` and ``gsp_ids`` (NaN where it has no coordinates)."""
        result = sites.copy()
        for column in ("gsp_node", "gsp_group", "gsp_ids"):
            result[column] = np.nan
            result[column] = result[column].astype(object)
        if lat_col not in result.columns or lon_col not in result.columns:
            return result
        polygon = nearest_gsp_polygon(self.polygons, result[lat_col], result[lon_col]).dropna()
        if polygon.empty:
            return result
        hit = self.table.reindex(polygon.astype(int).to_numpy())
        for column in ("gsp_node", "gsp_group", "gsp_ids"):
            result.loc[polygon.index, column] = hit[column].to_numpy()
        return result

    def node_for(self, lat: float, lon: float) -> Optional[str]:
        """Demand node of the GSP whose polygon holds (lat, lon), WGS84 degrees."""
        frame = self.assign(pd.DataFrame({"lat": [lat], "lon": [lon]}))
        node = frame["gsp_node"].iloc[0]
        return None if pd.isna(node) else str(node)


class GspDemandNodes:
    """The demand node of a GSP named by its GSP id (27 Oct 2026).

    The FES long-range chain's GSP-level building blocks carry their GSP id, so their plant goes
    where that GSP's demand goes: the ``Dem_per_node`` rows of the id, resolved on the network with
    the demand build's own rule (``scripts.demand.load.resolve_gsp_node_shares``) over the buses
    that carry load, the largest share taken. :class:`GspNodeLocator` does the same from a point,
    through the GSP polygon it falls in.
    """

    def __init__(self, network, dem_per_node_file=DEFAULT_DEM_PER_NODE_FILE,
                 dem_per_node: Optional[pd.DataFrame] = None, log: Optional[logging.Logger] = None):
        self.log = log or logger
        self.network = network
        if dem_per_node is None:
            dem_per_node = pd.read_excel(dem_per_node_file, sheet_name="Dem_per_node")
        weight_col = next(c for c in dem_per_node.columns if str(c).startswith("Dem as %"))
        self.rows: Dict[str, list] = {}
        for gsp, node, weight in zip(dem_per_node["GSP Id"].astype(str).str.strip(),
                                     dem_per_node["Node Id"].astype(str).str.strip(),
                                     pd.to_numeric(dem_per_node[weight_col], errors="coerce").fillna(0.0)):
            if gsp and node and weight > 0:
                self.rows.setdefault(gsp, []).append((node, float(weight)))
        loads = getattr(network, "loads", None)
        buses = sorted(set(loads["bus"].astype(str)) & set(network.buses.index)) if loads is not None else []
        self.load_buses = buses or list(network.buses.index)
        self._cache: Dict[str, Tuple[Optional[str], str]] = {}

    def node(self, gsp_id, lat: Optional[float] = None, lon: Optional[float] = None) -> Tuple[Optional[str], str]:
        """``(node, method)`` for one GSP id; ``(None, "no_dem_per_node_row")`` when the id has no row."""
        from scripts.demand.load import resolve_gsp_node_shares

        key = "" if gsp_id is None or (isinstance(gsp_id, float) and np.isnan(gsp_id)) else str(gsp_id).strip()
        if key in self._cache:
            return self._cache[key]
        result: Tuple[Optional[str], str] = (None, "no_dem_per_node_row")
        if key in self.rows:
            lonlat = (lon, lat) if lat is not None and lon is not None and not pd.isna(lat) else None
            shares, method = resolve_gsp_node_shares(self.rows[key], self.network, gsp_lonlat=lonlat,
                                                     candidate_buses=self.load_buses)
            if shares:
                result = (str(max(shares, key=lambda s: s[1])[0]), method)
        self._cache[key] = result
        return result


def place_distributed_at_gsp_nodes(sites: pd.DataFrame, locator: GspNodeLocator, carriers=("wind_onshore", "solar_pv"),
                                   thresholds: Optional[dict] = None, capacity_col: str = "capacity_mw",
                                   carrier_col: str = "technology", log: Optional[logging.Logger] = None,
                                   data_source: str = DISTRIBUTED_GSP_NODE_SOURCE,
                                   keep: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Move distribution-size sites of ``carriers`` to their own GSP's demand node.

    A site is distribution-size when its capacity is below its GSP group's
    licence-area threshold (``distribution_threshold_mw``). Transmission-size
    sites, sites with no coordinates and rows flagged in ``keep`` (a boolean
    Series on the index of ``sites``: documented connection points, named
    storage sites) keep their bus. Only ``locator.assign`` is used. Returns
    (sites, moves).
    """
    log = log or logger
    placed = locator.assign(sites)
    capacity = pd.to_numeric(placed[capacity_col], errors="coerce").fillna(0.0)
    threshold = placed["gsp_group"].map(lambda g: distribution_threshold_mw(g, thresholds) if pd.notna(g) else np.nan)
    eligible = (placed[carrier_col].isin(list(carriers)) & placed["gsp_node"].notna()
                & capacity.lt(threshold.astype(float)))
    if keep is not None:
        eligible &= ~keep.reindex(placed.index).fillna(False).astype(bool)
    moved = eligible & placed["bus"].astype(str).ne(placed["gsp_node"].astype(str))
    moves = placed.loc[moved, [c for c in ("site_name", carrier_col, capacity_col, "bus", "gsp_node", "gsp_group")
                               if c in placed.columns]].rename(columns={"bus": "from_bus", "gsp_node": "to_bus"})
    result = sites.copy()
    result.loc[eligible, "bus"] = placed.loc[eligible, "gsp_node"].to_numpy()
    result.loc[eligible, "data_source"] = data_source
    result.loc[eligible, "gsp_ids"] = placed.loc[eligible, "gsp_ids"].to_numpy()
    if len(moves):
        log.info("Distributed sites at their GSP node: %d of %d eligible sites moved (%.0f of %.0f MW), by carrier %s",
                 len(moves), int(eligible.sum()), float(capacity[moved].sum()), float(capacity[eligible].sum()),
                 moves.groupby(carrier_col)[capacity_col].sum().round(0).to_dict())
    return result, moves
