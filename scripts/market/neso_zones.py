"""NESO's 17 CP.08 economic zones on a saved network (20 Oct 2026).

Bus -> zone, as the CP30 balancing check has done since 18 Oct 2026 (moved here from
``scripts/CP30/balancing_validation.py``, which imports it back, so the check and the pro rata
turn-down re-pricing, ``scripts/market/turn_down_allocation.py``, zone every bus the same way):

1. Polygon (``data/CP30/validation_references/neso_cp08_zones.geojson``, decoded from CP.08's heat
   maps), the nearest polygon for a bus outside them all.
2. A bus more than ``remote_km`` outside every polygon (offshore) takes the zone of the nearest on-land
   bus through the network (lines, transformers, GB links; at most ``cutoff`` branches away).
3. Foreign buses get no zone.
4. A bus whose zone contradicts its side of the zone-defining ETYS boundaries (SSEP methodology) takes
   the nearest allowed zone.

:func:`network_bus_zones` does the whole thing for a saved ``.nc`` (netCDF4 reads, no PyPSA load).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ZONES_FILE = "data/CP30/validation_references/neso_cp08_zones.geojson"
ZONES = tuple(f"Z{i}" for i in range(1, 18))
SCOTLAND = ("Z1", "Z2", "Z3", "Z4")
#: The ETYS boundaries NESO drew its economic zones from (SSEP methodology).
ZONE_BOUNDARIES = ("B1a", "B2", "B4", "B6", "B7a", "B8", "B9", "EC5", "LE1", "SW1", "B13")
#: Defaults of the zoning (``cp30_balancing_validation.remote_bus_km`` / ``graph_cutoff`` carry the same).
DEFAULT_REMOTE_KM = 2.0
DEFAULT_GRAPH_CUTOFF = 6


def is_external_bus(name: str, country: str = "GB") -> bool:
    return "External" in str(name) or str(country) not in ("GB", "", "nan")


def load_zone_polygons(path: str | Path = ZONES_FILE):
    import geopandas as gpd

    polys = gpd.read_file(str(path)).to_crs(27700)
    if set(polys["zone"]) != set(ZONES):
        raise ValueError(f"{path}: zones {sorted(polys['zone'])}, expected Z1-Z17")
    return polys[["zone", "geometry"]]


def polygon_zones(buses: pd.DataFrame, polys) -> pd.DataFrame:
    """``neso_zone`` of every bus with coordinates (nearest polygon outside them) and ``dist_km`` outside."""
    import geopandas as gpd

    ok = buses[np.isfinite(buses["lon"].astype(float)) & np.isfinite(buses["lat"].astype(float))]
    out = pd.DataFrame(index=ok.index, columns=["neso_zone", "dist_km"], dtype=object)
    out["dist_km"] = 0.0
    if ok.empty:
        return out
    pts = gpd.GeoDataFrame(index=ok.index, geometry=gpd.points_from_xy(ok["lon"], ok["lat"]), crs=4326).to_crs(27700)
    inside = gpd.sjoin(pts, polys, how="left", predicate="within")
    inside = inside[~inside.index.duplicated()]
    out["neso_zone"] = inside["zone"].reindex(ok.index)
    miss = out["neso_zone"].isna()
    if miss.any():
        nn = gpd.sjoin_nearest(pts[miss.to_numpy()], polys, how="left", distance_col="dist_m")
        nn = nn[~nn.index.duplicated()]
        out.loc[nn.index, "neso_zone"] = nn["zone"]
        out.loc[nn.index, "dist_km"] = nn["dist_m"] / 1e3
    out["dist_km"] = out["dist_km"].astype(float)
    return out


def boundary_crossings(net: dict, boundaries=ZONE_BOUNDARIES, secured: dict | None = None) -> dict:
    """{boundary: crossings frame} for the zone-defining boundaries, on this network's own branches.

    The secured-boundary code's own crossing rule (``secured_boundaries._boundary_crossings``) with the
    scenario's include/exclude corrections.
    """
    from scripts.solve.secured_boundaries import DEFAULT_GEOMETRY, _boundary_crossings, load_boundary_geometries

    secured = dict(secured or {})
    geoms = load_boundary_geometries(secured.get("geometry_file") or DEFAULT_GEOMETRY)
    cfg = {"include": dict(secured.get("include") or {}), "exclude": dict(secured.get("exclude") or {})}
    view = SimpleNamespace(buses=net["buses"], lines=net["lines"], transformers=net["transformers"],
                           links=net["links"])
    return {b: _boundary_crossings(view, b, geoms[b], cfg) for b in boundaries if b in geoms}


def boundary_sides(net: dict, crossings: dict) -> pd.DataFrame:
    """Each bus's side (+1 / -1; 0 unknown) of each boundary: the component it falls in once the crossings go."""
    import networkx as nx

    links = net["links"]
    internal = links[(links["carrier"] == "DC") & ~links["bus0"].str.contains("External")
                     & ~links["bus1"].str.contains("External")]
    g = nx.MultiGraph()
    g.add_nodes_from(net["buses"].index)
    for tab, df in (("line", net["lines"]), ("transformer", net["transformers"]), ("link", internal)):
        g.add_edges_from((r.bus0, r.bus1, f"{tab}:{n}") for n, r in df[["bus0", "bus1"]].iterrows())
    tables = {"line": net["lines"], "transformer": net["transformers"], "link": links}
    sides = {}
    for b, cross in crossings.items():
        h = g.copy()
        ends = []
        for _, c in cross.iterrows():
            df = tables[c["component"]]
            if c["name"] not in df.index:
                continue
            b0, b1 = df.at[c["name"], "bus0"], df.at[c["name"], "bus1"]
            key = f"{c['component']}:{c['name']}"
            if h.has_edge(b0, b1, key):
                h.remove_edge(b0, b1, key)
            ends += [(b0, int(np.sign(c["sign"]))), (b1, -int(np.sign(c["sign"])))]
        comp_of = {bus: i for i, comp in enumerate(nx.connected_components(h)) for bus in comp}
        side_of_comp = {}
        for bus, sign in ends:
            side_of_comp.setdefault(comp_of[bus], sign)
        sides[b] = pd.Series({bus: side_of_comp.get(c, 0) for bus, c in comp_of.items()}, dtype=int)
    return pd.DataFrame(sides).reindex(net["buses"].index).fillna(0).astype(int)


def allowed_zones(sides: dict) -> list[str]:
    """Zones a bus may be in, given its sides of the zone-defining boundaries (+1 = north or inside)."""
    s = sides
    if s.get("EC5") == 1:
        return ["Z12"]
    if s.get("LE1") == 1:
        return ["Z17"]
    if s.get("SW1") == -1:
        return ["Z13"]
    if s.get("B13") == -1:
        return ["Z15"]
    for b, z in (("B1a", ["Z1"]), ("B2", ["Z2"]), ("B4", ["Z3"]), ("B6", ["Z4"]), ("B7a", ["Z5", "Z6"]),
                 ("B8", ["Z7", "Z8"])):
        if s.get(b) == 1:
            return z
    if s.get("B9") == -1:
        return ["Z9", "Z10", "Z11"]
    return ["Z11", "Z14", "Z16", "Z17", "Z15", "Z10"]


def bus_zone_map(net: dict, polys, crossings: dict | None = None, remote_km: float = DEFAULT_REMOTE_KM,
                 cutoff: int = DEFAULT_GRAPH_CUTOFF) -> pd.DataFrame:
    """bus -> NESO zone with how it was found (``polygon``, ``graph:<bus>``, ``external``, ``reconciled``).

    ``net``: {"buses" (lon, lat, country), "lines", "transformers", "links" (bus0, bus1, carrier)}.
    """
    import networkx as nx

    buses = net["buses"]
    bz = polygon_zones(buses, polys)
    bz["mapping"] = "polygon"
    g = nx.Graph()
    for df in (net["lines"], net["transformers"]):
        g.add_edges_from(zip(df["bus0"], df["bus1"]))
    lk = net["links"]
    lk = lk[~lk["bus0"].str.contains("External") & ~lk["bus1"].str.contains("External")]
    g.add_edges_from(zip(lk["bus0"], lk["bus1"]))
    onland = set(bz.index[bz["dist_km"] <= remote_km])
    for bus in bz.index[bz["dist_km"] > remote_km]:
        if bus not in g:
            continue
        lengths = nx.single_source_shortest_path_length(g, bus, cutoff=cutoff)
        cand = sorted((dist, n) for n, dist in lengths.items() if n in onland)
        if cand:
            near = cand[0][1]
            bz.loc[bus, "neso_zone"] = bz.loc[near, "neso_zone"]
            bz.loc[bus, "mapping"] = f"graph:{near}"
    country = buses["country"] if "country" in buses else pd.Series("GB", index=buses.index)
    external = [b for b in bz.index if is_external_bus(b, country.get(b, "GB"))]
    bz.loc[external, "mapping"] = "external"
    bz.loc[external, "neso_zone"] = np.nan
    bz["polygon_zone"] = bz["neso_zone"]
    if crossings:
        sides = boundary_sides(net, crossings)
        ok = bz.index[bz["mapping"].ne("external") & bz["neso_zone"].notna()].intersection(sides.index)
        if len(ok):
            import geopandas as gpd

            pts = gpd.GeoSeries(gpd.points_from_xy(buses.loc[ok, "lon"], buses.loc[ok, "lat"]), index=ok,
                                crs=4326).to_crs(27700)
            geom = polys.set_index("zone").geometry
            for bus in ok:
                sig = sides.loc[bus]
                if (sig.reindex(list(ZONE_BOUNDARIES)).fillna(0) == 0).any():
                    continue
                allow = allowed_zones(sig.to_dict())
                if bz.at[bus, "neso_zone"] not in allow:
                    bz.at[bus, "neso_zone"] = min(allow, key=lambda z: geom[z].distance(pts[bus]))
                    bz.at[bus, "mapping"] = f"reconciled:{bz.at[bus, 'mapping']}"
    return bz


# ── A saved network ─────────────────────────────────────────────────────────
def _strings(d, name) -> np.ndarray:
    return np.asarray(d.variables[name][:]).astype(str)


def _floats(d, name) -> np.ndarray:
    return np.asarray(d.variables[name][:], dtype=float)


def read_zone_topology(path: str | Path) -> dict:
    """The tables :func:`bus_zone_map` needs, read from a saved network with netCDF4."""
    import netCDF4

    d = netCDF4.Dataset(str(path))
    d.set_auto_mask(False)
    try:
        bus_i = _strings(d, "buses_i")
        buses = pd.DataFrame({"x": _floats(d, "buses_x"), "y": _floats(d, "buses_y"),
                              "lon": _floats(d, "buses_lon") if "buses_lon" in d.variables else np.nan,
                              "lat": _floats(d, "buses_lat") if "buses_lat" in d.variables else np.nan,
                              "country": (_strings(d, "buses_country") if "buses_country" in d.variables
                                          else np.full(len(bus_i), "GB"))}, index=bus_i)

        def table(prefix, cols):
            if f"{prefix}_i" not in d.variables:
                return pd.DataFrame(columns=list(cols))
            out = pd.DataFrame(index=_strings(d, f"{prefix}_i"))
            for col, kind in cols.items():
                name = f"{prefix}_{col}"
                if name in d.variables:
                    out[col] = _strings(d, name) if kind == "s" else _floats(d, name)
                else:
                    out[col] = "" if kind == "s" else np.nan
            return out

        return {"buses": buses,
                "lines": table("lines", {"bus0": "s", "bus1": "s", "s_nom": "f"}),
                "transformers": table("transformers", {"bus0": "s", "bus1": "s", "s_nom": "f"}),
                "links": table("links", {"bus0": "s", "bus1": "s", "carrier": "s", "p_nom": "f"})}
    finally:
        d.close()


def network_bus_zones(path: str | Path, zones_file: str | Path = ZONES_FILE, secured: dict | None = None,
                      remote_km: float = DEFAULT_REMOTE_KM, cutoff: int = DEFAULT_GRAPH_CUTOFF,
                      polys=None) -> pd.DataFrame:
    """bus -> ``neso_zone`` (and ``mapping``) on the saved network at ``path``, boundary-reconciled."""
    net = read_zone_topology(path)
    polys = load_zone_polygons(zones_file) if polys is None else polys
    crossings = boundary_crossings(net, ZONE_BOUNDARIES, secured)
    return bus_zone_map(net, polys, crossings, float(remote_km), int(cutoff))
