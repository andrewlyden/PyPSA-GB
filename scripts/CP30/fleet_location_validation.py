#!/usr/bin/env python3
"""CP30 fleet location check: installed capacity by CP30 ES5 region, NESO CP.08 zone and ETYS boundary side.

Read only (19 Oct 2026). Takes a built (or solved) CP30 network and writes

* a tidy CSV, one row per (check, scope, key, technology): ``ours_mw``, ``cp30_mw``, ``gap_mw``, ``gap_pct``,
  ``status``, ``note`` (plus bus, zone, region and boundary sides for the per-asset rows);
* a JSON summary (status counts, every WARN/FAIL row, the zone x technology, region x technology and
  boundary x technology tables).

Checks
------
1. ``zone_capacity``        capacity by NESO CP.08 zone (Z1-Z17) x technology. INFO: CP30 publishes no zonal
                            capacity. Capacity on GB buses without a zone WARNs.
2. ``es5_region``           capacity by CP30 ES5 region (bus GSP polygon -> GSP group -> region, the build's own
                            rule, ``spatial_allocation.bus_cp30_regions``) against ES5 for the five technologies
                            ES5 publishes (offshore, onshore, solar, batteries, storage). Others are INFO.
3. ``region_boundary``      MW whose ES5 region contradicts its bus's side of B4/B6 (North Scotland north of B4;
                            South Scotland between B4 and B6; England and Wales south of B6), by technology.
4. ``boundary_remote_side`` MW on the exporting (north / inside) side of each boundary by technology; for B4 and B6
                            ES5 implies the remote-side total of the published technologies (North Scotland;
                            North plus South Scotland), which is checked.
5. ``offshore_landing``     each named offshore farm (REPD project) and each proxy landing bus of at least
                            ``offshore_list_mw``: landing bus, zone, region, sides. A named farm is checked against
                            the landing reference (the build's REPD crosswalk + the committed additions); FAIL when
                            the reference landing is across any main boundary.
6. ``interconnector``       each cross-border cable: GB landing bus against the NESO register reference (accepted
                            buses; FAIL when not among them), capacity against the CP30 cable composition; plus the
                            total.
7. ``position``             bus positions against circuit lengths (bus position audit, 19 Oct 2026). A line's
                            straight-line distance between its bus positions against its length (ETYS
                            ``ETYS_<year>_components.csv`` ``length_km``, else the network's own non-zero length,
                            which Appendix B circuits carry as OHL + cable length; inter-site lines without a length
                            are not judged): inconsistent (WARN per circuit) beyond 1.1 x length + 3 km.
                            Transformers and same-site sections without a length: ends more than 2 km apart.
                            FAIL: a bus with most of its (at least two) judged circuits over by more than 10 km; a
                            transformer / section with ends more than 10 km apart; a circuit over by more than
                            5 km that crosses a secured boundary; an end of such a circuit whose position estimated
                            from its circuits lies in another CP30 region or CP.08 zone, or (carrying more than
                            50 MW) across a boundary. Reviewed exemptions (``position_exemptions.csv``: an ETYS
                            section length on a longer route such as Eccles-Stella West, the London Array
                            inter-platform tie) are INFO.

Bus -> NESO zone: the CP.08 zone polygon a bus falls in; a bus outside every polygon (offshore platforms, coastal
nodes) takes the zone of the nearest (by branches) bus inside one, else the nearest polygon; then a bus whose zone
contradicts its side of the zone-defining boundaries (NESO SSEP draws the zones from B1a, B2, B4, B6, B7a, B8, B9,
EC5, LE1, SW1 and B13) moves to the nearest allowed zone. Boundary sides are topological: the branches crossing each
ETYS boundary polyline (``secured_boundaries.find_boundary_crossings`` plus the scenario's
``secured_capability.include``, e.g. EGL2 on B4) are cut, and each connected component takes the side of the
crossing ends it holds (+1 = the polyline's left = north / inside for every boundary used here).

A FAIL row fails the rule after the outputs are written (``fail_on_fail``, as validate_CP30_locational).
Thresholds: :data:`DEFAULT_CONFIG` (scenario block ``validation.fleet_location``; the two must match,
tests/unit/test_cp30_fleet_location_validation.py). Rule ``validate_CP30_fleet_location`` (rules/CP30.smk).

    python scripts/CP30/fleet_location_validation.py --network resources/network/X.nc \
        --out-csv out.csv --out-json out.json [--fail-on-fail]
"""
# No `from __future__` import: Snakemake's script directive prepends a preamble.

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

#: Code defaults of the scenario block ``validation.fleet_location`` (in
#: config/defaults/method/brownfield.yaml; the two must match, tests/unit/test_cp30_fleet_location_validation.py).
DEFAULT_CONFIG: dict = {
    "enabled": True,                 # in rule all's CP30 validation targets
    "fail_on_fail": True,            # a FAIL row fails the rule (after the outputs are written)
    "zones_file": "data/CP30/validation_references/neso_cp08_zones.geojson",
    "landing_additions_file": "data/CP30/validation_references/offshore_landing_additions.csv",
    "ic_reference_file": "data/CP30/validation_references/interconnectors_2030.csv",
    "rel_tol": 0.05,                 # WARN when |gap| exceeds 5% of the CP30 value ...
    "min_mw": 25.0,                  # ... and at least this many MW (keeps rounding noise out)
    "fail_mw": 500.0,                # FAIL when |gap| exceeds this (any row with a CP30 value) ...
    "fail_rel": 0.15,                # ... or exceeds this share and fail_mw / 4
    "offshore_list_mw": 200.0,       # offshore farms / proxy landing buses listed individually
    "unreferenced_warn_mw": 200.0,   # a named offshore farm this big with no landing reference WARNs
    "region_side_warn_mw": 25.0,     # MW on the wrong side of B4/B6 for its ES5 region: WARN above, FAIL above fail_mw
    "graph_cutoff": 6,               # branches searched for the nearest bus inside a zone polygon
    "report_boundaries": ["B0", "B1a", "B2", "B4", "B6", "B7a", "B8", "B9", "EC5", "LE1", "SC1", "SW1", "B13"],
    "main_side_boundaries": ["B1a", "B2", "B4", "B6", "B7a", "B8", "EC5"],
    # 7. Position consistency (bus position audit, 19 Oct 2026). A line whose bus positions are further apart in
    #    a straight line than (1 + slack_rel) x its length + slack_km is inconsistent (WARN), as is a transformer
    #    or a same-site section without a length whose ends are more than transformer_max_km apart. FAIL: a bus
    #    with at least min_circuits judged circuits, more than suspect_share of them over by more than
    #    fail_excess_km; a transformer / section with ends more than fail_transformer_km apart; a circuit over by
    #    more than boundary_excess_km that crosses a secured boundary; an end of such a circuit whose position
    #    estimated from its circuits (L1 trilateration) lies in another CP30 region or CP.08 zone, or, carrying
    #    more than fail_mw, across a boundary. Reviewed exemptions (exemptions_file, bus pairs) are INFO.
    "position_line_slack_km": 3.0,
    "position_line_slack_rel": 0.10,
    "position_transformer_max_km": 2.0,
    "position_min_circuits": 2,
    "position_suspect_share": 0.5,
    "position_fail_excess_km": 10.0,
    "position_fail_transformer_km": 10.0,
    "position_boundary_excess_km": 5.0,
    "position_fail_mw": 50.0,
    "position_exemptions_file": "data/CP30/validation_references/position_exemptions.csv",
}

STATUSES = ("PASS", "WARN", "FAIL", "INFO")
ES5_TECH = {"Offshore Wind": "Offshore wind", "Onshore Wind": "Onshore wind", "Solar": "Solar",
            "Batteries": "Battery", "Storage": "Storage (all)"}
GEN_TECH = {"wind_offshore": "Offshore wind", "wind_onshore": "Onshore wind", "solar_pv": "Solar",
            "nuclear": "Nuclear", "CCGT": "Gas", "OCGT": "Gas", "gas_engine": "Gas", "CCS Gas": "Gas",
            "CHP": "Gas CHP", "biomass": "Biomass", "CCS Biomass": "Biomass", "biogas": "Other thermal",
            "waste_to_energy": "Other thermal", "oil": "Other thermal", "large_hydro": "Hydro",
            "small_hydro": "Hydro", "marine": "Marine", "tidal_stream": "Marine", "shoreline_wave": "Marine"}
TECH_ORDER = ["Offshore wind", "Onshore wind", "Solar", "Hydro", "Marine", "Nuclear", "Gas", "Gas CHP", "Biomass",
              "Other thermal", "Battery", "LDES/PSH", "Electrolysis", "Interconnector"]
#: The ETYS boundaries NESO drew its economic zones from (SSEP methodology).
ZONE_KEYS = ["B1a", "B2", "B4", "B6", "B7a", "B8", "B9", "EC5", "LE1", "SW1", "B13"]
#: Required boundary sides of each ES5 region (+1 = north of the boundary).
SCOTLAND_REGIONS = {"North Scotland": {"B4": 1}, "South Scotland": {"B4": -1, "B6": 1}}
EW_REGIONS = {"East England", "Midlands", "North England", "South East", "South West", "Wales"}
#: ES5 regions whose published capacity sits on the exporting side of a boundary.
ES5_REMOTE = {"B4": ["North Scotland"], "B6": ["North Scotland", "South Scotland"]}


def fleet_location_config(block: dict | None) -> dict:
    """``validation.fleet_location`` over :data:`DEFAULT_CONFIG`, refusing unknown keys."""
    block = dict(block or {})
    if unknown := sorted(set(block) - set(DEFAULT_CONFIG)):
        raise ValueError(f"validation.fleet_location has unknown keys {unknown}; "
                         f"supported: {sorted(DEFAULT_CONFIG)}")
    return {**copy.deepcopy(DEFAULT_CONFIG), **block}


# ── Network reading ─────────────────────────────────────────────────────────
def _s(d, v: str) -> np.ndarray:
    return np.asarray(d.variables[v][:]).astype(str)


def _f(d, v: str) -> np.ndarray:
    return np.asarray(d.variables[v][:], dtype=float)


def _opt(d, v: str, n: int, kind=str) -> np.ndarray:
    if v in d.variables:
        return _s(d, v) if kind is str else _f(d, v)
    return np.full(n, "" if kind is str else np.nan, dtype=object if kind is str else float)


def read_network(path: str | Path) -> dict:
    """Static component tables of a PyPSA netCDF, one array at a time (no time series)."""
    import netCDF4

    d = netCDF4.Dataset(str(path))
    d.set_auto_mask(False)
    try:
        nb = len(d.variables["buses_i"])
        buses = pd.DataFrame({"x": _f(d, "buses_x"), "y": _f(d, "buses_y"), "lon": _opt(d, "buses_lon", nb, float),
                              "lat": _opt(d, "buses_lat", nb, float), "v_nom": _f(d, "buses_v_nom"),
                              "carrier": _opt(d, "buses_carrier", nb), "country": _opt(d, "buses_country", nb)},
                             index=_s(d, "buses_i"))
        ng = len(d.variables["generators_i"])
        gens = pd.DataFrame({"bus": _s(d, "generators_bus"), "carrier": _s(d, "generators_carrier"),
                             "p_nom": _f(d, "generators_p_nom"),
                             "lon": _opt(d, "generators_lon", ng, float), "lat": _opt(d, "generators_lat", ng, float),
                             "site": _opt(d, "generators_source_site_name", ng),
                             "repd_ref_id": _opt(d, "generators_repd_ref_id", ng),
                             "fleet_origin": _opt(d, "generators_fleet_origin", ng)},
                            index=_s(d, "generators_i"))
        sus = (pd.DataFrame({"bus": _s(d, "storage_units_bus"), "carrier": _s(d, "storage_units_carrier"),
                             "p_nom": _f(d, "storage_units_p_nom")}, index=_s(d, "storage_units_i"))
               if "storage_units_i" in d.variables else pd.DataFrame(columns=["bus", "carrier", "p_nom"]))
        nl = len(d.variables["links_i"])
        links = pd.DataFrame({"bus0": _s(d, "links_bus0"), "bus1": _s(d, "links_bus1"),
                              "carrier": _s(d, "links_carrier"), "p_nom": _f(d, "links_p_nom"),
                              "interconnector_name": _opt(d, "links_interconnector_name", nl)},
                             index=_s(d, "links_i"))
        nli = len(d.variables["lines_i"])
        lines = pd.DataFrame({"bus0": _s(d, "lines_bus0"), "bus1": _s(d, "lines_bus1"), "s_nom": _f(d, "lines_s_nom"),
                              "length_km": _opt(d, "lines_length_km", nli, float),
                              "length": _opt(d, "lines_length", nli, float)},
                             index=_s(d, "lines_i"))
        trafos = pd.DataFrame({"bus0": _s(d, "transformers_bus0"), "bus1": _s(d, "transformers_bus1"),
                               "s_nom": _f(d, "transformers_s_nom")}, index=_s(d, "transformers_i"))
    finally:
        d.close()
    for df in (gens, links):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].replace({"nan": "", "None": ""})
    return dict(buses=buses, gens=gens, sus=_storage_plant(sus), links=links, lines=lines, trafos=trafos)


def _storage_plant(sus: pd.DataFrame) -> pd.DataFrame:
    """Storage units without consumer flexibility (demand, F5: ``demand_carriers.DEMAND_FLEX_STORAGE_CARRIERS``)."""
    from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS

    return sus[~sus["carrier"].astype(str).isin(DEMAND_FLEX_STORAGE_CARRIERS)]


def network_tables(network) -> dict:
    """The same tables from an in-memory PyPSA network (tests, notebooks)."""
    def col(df, name, default):
        return df[name] if name in df else pd.Series(default, index=df.index)

    b, g, lk = network.buses, network.generators, network.links
    buses = pd.DataFrame({"x": b["x"].astype(float), "y": b["y"].astype(float),
                          "lon": col(b, "lon", np.nan).astype(float), "lat": col(b, "lat", np.nan).astype(float),
                          "v_nom": b["v_nom"].astype(float), "carrier": col(b, "carrier", "AC").astype(str),
                          "country": col(b, "country", "").astype(str)})
    gens = pd.DataFrame({"bus": g["bus"].astype(str), "carrier": g["carrier"].astype(str),
                         "p_nom": g["p_nom"].astype(float), "lon": col(g, "lon", np.nan).astype(float),
                         "lat": col(g, "lat", np.nan).astype(float),
                         "site": col(g, "source_site_name", "").astype(str),
                         "repd_ref_id": col(g, "repd_ref_id", "").astype(str),
                         "fleet_origin": col(g, "fleet_origin", "").astype(str)})
    sus = _storage_plant(network.storage_units[["bus", "carrier", "p_nom"]].copy())
    links = pd.DataFrame({"bus0": lk["bus0"].astype(str), "bus1": lk["bus1"].astype(str),
                          "carrier": lk["carrier"].astype(str), "p_nom": lk["p_nom"].astype(float),
                          "interconnector_name": col(lk, "interconnector_name", "").astype(str)})
    for df in (gens, links):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].replace({"nan": "", "None": ""})
    ln = network.lines
    lines = pd.DataFrame({"bus0": ln["bus0"].astype(str), "bus1": ln["bus1"].astype(str),
                          "s_nom": ln["s_nom"].astype(float), "length_km": col(ln, "length_km", np.nan).astype(float),
                          "length": col(ln, "length", np.nan).astype(float)})
    return dict(buses=buses, gens=gens, sus=sus, links=links, lines=lines,
                trafos=network.transformers[["bus0", "bus1", "s_nom"]].copy())


def external_mask(buses: pd.DataFrame) -> pd.Series:
    """Foreign buses, buses without coordinates and non-electrical carrier buses (hydrogen, heat)."""
    c = buses["country"].astype(str)
    electrical = buses["carrier"].astype(str).isin(["AC", "DC", "", "nan"])
    return (c.ne("GB") & c.ne("")) | buses.index.str.contains("External") | buses["x"].isna() | ~electrical


# ── Boundary sides ──────────────────────────────────────────────────────────
def boundary_sides(net: dict, geoms: dict, boundaries: list[str], include: dict | None) -> pd.DataFrame:
    """+1 / -1 side of every bus for each boundary (0 when its component holds no crossing end).

    Args:
        net: tables from :func:`read_network`.
        geoms: ``{boundary: LineString}`` (``secured_boundaries.load_boundary_geometries``).
        boundaries: boundaries to resolve (absent from ``geoms``: skipped).
        include: the scenario's ``secured_capability.include`` ({boundary: [{component, name, sign}]}).
    """
    import networkx as nx

    from scripts.solve.secured_boundaries import find_boundary_crossings

    buses, lines, trafos, links = net["buses"], net["lines"], net["trafos"], net["links"]
    ext = links.bus0.str.contains("External") | links.bus1.str.contains("External")
    internal = links[links.carrier.eq("DC") & links.p_nom.gt(0) & ~ext]
    shim = SimpleNamespace(buses=buses, lines=lines, transformers=trafos, links=internal)
    g = nx.MultiGraph()
    g.add_nodes_from(buses.index)
    tables = {"line": lines, "transformer": trafos, "link": internal}
    for tab, df in tables.items():
        g.add_edges_from((b0, b1, f"{tab}:{n}") for n, b0, b1 in zip(df.index, df.bus0, df.bus1))
    out = {}
    for b in boundaries:
        if b not in geoms:
            continue
        cross = find_boundary_crossings(shim, geoms[b])
        for extra in (include or {}).get(b, []) or []:
            if extra["name"] in tables[extra["component"]].index and extra["name"] not in set(cross["name"]):
                cross = pd.concat([cross, pd.DataFrame([{"component": extra["component"], "name": extra["name"],
                                                         "sign": float(extra.get("sign", 1.0)), "rating": np.nan}])],
                                  ignore_index=True)
        h = g.copy()
        ends = []
        for comp, name, sign in zip(cross.component, cross.name, cross.sign):
            df = tables[comp]
            b0, b1 = df.at[name, "bus0"], df.at[name, "bus1"]
            if h.has_edge(b0, b1, f"{comp}:{name}"):
                h.remove_edge(b0, b1, f"{comp}:{name}")
            ends += [(b0, int(np.sign(sign))), (b1, -int(np.sign(sign)))]
        comp_of = {bus: i for i, c in enumerate(nx.connected_components(h)) for bus in c}
        side_of: dict = {}
        for bus, sg in ends:
            side_of.setdefault(comp_of[bus], sg)
        out[b] = pd.Series({bus: side_of.get(c, 0) for bus, c in comp_of.items()}).reindex(buses.index).fillna(0)
    return pd.DataFrame(out, index=buses.index).astype(int)


# ── NESO zones ──────────────────────────────────────────────────────────────
def allowed_zones(sig: dict) -> list[str]:
    """Zones a bus may belong to, given its sides of the zone-defining boundaries (+1 = north / inside)."""
    from scripts.CP30.balancing_validation import allowed_zones as _allowed

    return _allowed(sig)


def bus_zones(net: dict, sides: pd.DataFrame, zones_file: str | Path, cutoff: int = 6) -> pd.DataFrame:
    """bus -> ``neso_zone``, with ``mapping`` (polygon / graph / nearest_polygon) and ``reconciled``."""
    import geopandas as gpd
    import networkx as nx

    buses = net["buses"]
    gb = buses[~external_mask(buses)]
    poly = gpd.read_file(str(zones_file)).to_crs(27700)[["zone", "geometry"]]
    pts = gpd.GeoDataFrame(index=gb.index, geometry=gpd.points_from_xy(gb.x, gb.y), crs=27700)
    j = gpd.sjoin(pts, poly, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = pd.DataFrame({"neso_zone": j["zone"], "mapping": np.where(j["zone"].notna(), "polygon", "")}, index=gb.index)
    miss = out.index[out.neso_zone.isna()]
    if len(miss):
        # Buses outside every polygon: the zone of the nearest (by branches) connected bus inside one.
        g = nx.Graph()
        for df in (net["lines"], net["trafos"]):
            g.add_edges_from(zip(df.bus0, df.bus1))
        lk = net["links"]
        ext = lk.bus0.str.contains("External") | lk.bus1.str.contains("External")
        g.add_edges_from(zip(lk.bus0[~ext], lk.bus1[~ext]))
        inside = set(out.index[out.neso_zone.notna()])
        for bus in miss:
            if bus in g:
                lengths = nx.single_source_shortest_path_length(g, bus, cutoff=cutoff)
                for nb, _ in sorted(lengths.items(), key=lambda t: t[1]):
                    if nb in inside:
                        out.at[bus, "neso_zone"], out.at[bus, "mapping"] = out.at[nb, "neso_zone"], "graph"
                        break
        miss = out.index[out.neso_zone.isna()]
        if len(miss):
            nn = gpd.sjoin_nearest(pts.loc[miss], poly, how="left")
            nn = nn[~nn.index.duplicated()]
            out.loc[miss, "neso_zone"], out.loc[miss, "mapping"] = nn["zone"], "nearest_polygon"
    # Reconcile with the zone-defining boundary sides.
    out["polygon_zone"] = out["neso_zone"]
    npoly = poly.set_index("zone").geometry
    if all(k in sides.columns for k in ZONE_KEYS):
        for bus in out.index:
            sig = {k: int(sides.at[bus, k]) for k in ZONE_KEYS}
            if 0 in sig.values():
                continue
            allow = allowed_zones(sig)
            if out.at[bus, "neso_zone"] not in allow:
                p = pts.geometry[bus]
                out.at[bus, "neso_zone"] = min(allow, key=lambda z: npoly[z].distance(p))
    out["reconciled"] = out.neso_zone.ne(out.polygon_zone)
    return out


# ── CP30 regions and ES5 ────────────────────────────────────────────────────
def bus_regions(buses: pd.DataFrame, gsp_polygons: str | Path, group_region: str | Path) -> pd.Series:
    """CP30 ES5 region of every GB bus, by the build's own rule (``spatial_allocation.bus_cp30_regions``)."""
    from scripts.CP30.spatial_allocation import bus_cp30_regions, load_gsp_region_map

    poly, grp = load_gsp_region_map(gsp_polygons, group_region)
    gb = buses[~external_mask(buses)]
    return bus_cp30_regions(gb, poly, grp)["cp30_region"]


def read_es5(path: str | Path, pathway: str) -> pd.DataFrame:
    """ES5 capacity (MW), technology x region: the CP30 workbook (sheet ES5) or its extracted CSV."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        es5 = pd.read_excel(path, "ES5", header=5).dropna(how="all", axis=1)
    else:
        es5 = pd.read_csv(path)
    es5 = es5[(es5["Scenario"] == pathway) & (es5["Variable"] == "Capacity (MW)")].set_index("Technology")
    return es5.drop(columns=[c for c in ("Scenario", "Variable", "source_sheet") if c in es5]).astype(float)


def _point_region_fn(gsp_polygons, group_region):
    """(x, y) arrays -> CP30 ES5 region, the build's rule; None without the region files."""
    if not gsp_polygons or not group_region or not Path(gsp_polygons).exists() or not Path(group_region).exists():
        return None
    from scripts.CP30.spatial_allocation import bus_cp30_regions, load_gsp_region_map

    poly, grp = load_gsp_region_map(gsp_polygons, group_region)

    def f(x, y):
        pts = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)})
        return bus_cp30_regions(pts, poly, grp, overrides={})["cp30_region"].to_numpy()
    return f


def _point_zone_fn(zones_file):
    """(x, y) arrays -> NESO CP.08 zone polygon (the nearest outside every polygon); None without the file."""
    if not zones_file or not Path(zones_file).exists():
        return None
    import geopandas as gpd

    poly = gpd.read_file(str(zones_file)).to_crs(27700)[["zone", "geometry"]]

    def f(x, y):
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.asarray(x, float), np.asarray(y, float)), crs=27700)
        j = gpd.sjoin_nearest(pts, poly, how="left")
        return j[~j.index.duplicated()]["zone"].reindex(pts.index).to_numpy()
    return f


# ── Rows ────────────────────────────────────────────────────────────────────
def status_of(ours: float, ref: float, cfg: dict, warn_mw: float | None = None) -> str:
    """PASS / WARN / FAIL of a value against a CP30 reference (INFO without one)."""
    if ref is None or not np.isfinite(ref):
        return "INFO"
    warn_mw = cfg["min_mw"] if warn_mw is None else warn_mw
    gap = abs(ours - ref)
    if gap > cfg["fail_mw"] or (ref > 0 and gap / ref > cfg["fail_rel"] and gap > cfg["fail_mw"] / 4):
        return "FAIL"
    if gap > warn_mw and (ref == 0 or gap / ref > cfg["rel_tol"]):
        return "WARN"
    return "PASS"


def make_row(cfg: dict, check: str, scope: str, key, tech: str, ours: float, ref: float = np.nan,
             status: str | None = None, note: str = "", **extra) -> dict:
    gap = ours - ref if np.isfinite(ref) else np.nan
    pct = 100 * gap / ref if np.isfinite(ref) and ref else np.nan
    r = dict(check=check, scope=scope, key=key, technology=tech, ours_mw=round(float(ours), 1),
             cp30_mw=round(float(ref), 1) if np.isfinite(ref) else np.nan,
             gap_mw=round(gap, 1) if np.isfinite(gap) else np.nan, gap_pct=round(pct, 1) if np.isfinite(pct) else np.nan,
             status=status or status_of(ours, ref, cfg), note=note)
    r.update(extra)
    return r


def interconnector_links(net: dict) -> pd.DataFrame:
    """Cross-border links, one row per cable, with its GB bus in ``bus`` (bidirectional links counted once)."""
    lk, ext = net["links"], external_mask(net["buses"])
    e0, e1 = lk.bus0.map(ext).fillna(True).astype(bool), lk.bus1.map(ext).fillna(True).astype(bool)
    x = lk[lk.carrier.eq("DC") & (e0 ^ e1)].copy()
    x["bus"] = np.where(e0[x.index], x.bus1, x.bus0)
    x["name"] = x["interconnector_name"].where(x["interconnector_name"].astype(str).ne(""),
                                               x.index.str.replace("IC_", ""))
    return x.drop_duplicates(["name", "bus"])


def capacity_rows(net: dict) -> pd.DataFrame:
    """Installed capacity rows (bus, tech, p_nom): generators, storage, electrolysers, interconnectors."""
    g, su, lk = net["gens"], net["sus"], net["links"]
    gg = g[g.carrier.isin(GEN_TECH)].assign(tech=lambda d: d.carrier.map(GEN_TECH))
    s = su.assign(tech=np.where(su.carrier.eq("Battery"), "Battery", "LDES/PSH"))
    el = lk[lk.carrier.eq("electrolysis")].assign(bus=lambda d: d.bus0, tech="Electrolysis")
    ic = interconnector_links(net).assign(tech="Interconnector")
    return pd.concat([gg[["bus", "tech", "p_nom"]], s[["bus", "tech", "p_nom"]], el[["bus", "tech", "p_nom"]],
                      ic[["bus", "tech", "p_nom"]]])


def _split(v) -> list[str]:
    return [t for t in str(v).replace(";", "|").split("|") if t and t != "nan"]


def landing_reference(files: list) -> pd.Series:
    """repd_ref_id -> reference landing bus(es) ('|'-joined), the first file that names a project winning."""
    ref = [pd.read_csv(f, dtype=str)[["repd_ref_id", "landing_bus"]] for f in files if f and Path(f).exists()]
    if not ref:
        return pd.Series(dtype=str)
    return pd.concat(ref).drop_duplicates("repd_ref_id").set_index("repd_ref_id")["landing_bus"]


# ── Position consistency ────────────────────────────────────────────────────
def etys_line_lengths(path: str | Path | None) -> pd.Series:
    """Line name -> ETYS ``length_km`` from ``resources/network/ETYS_<year>_components.csv`` (empty without one)."""
    if not path or not Path(path).exists():
        return pd.Series(dtype=float)
    comp = pd.read_csv(path, usecols=["name", "component", "length_km"])
    comp = comp[comp["component"].eq("line")].drop_duplicates("name")
    return pd.to_numeric(comp.set_index("name")["length_km"], errors="coerce")


def _site(bus: pd.Series) -> pd.Series:
    """The ETYS site code of a node name (its first four characters)."""
    return bus.astype(str).str[:4]


def position_exemptions(path: str | Path | None) -> pd.DataFrame:
    """Reviewed position exemptions: ``bus0, bus1, reason, source`` (a bus pair, either order), empty without a file."""
    cols = ["bus0", "bus1", "reason", "source"]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    ex = pd.read_csv(path, dtype=str, comment="#").fillna("")
    if missing := sorted({"bus0", "bus1", "reason"} - set(ex.columns)):
        raise ValueError(f"{path} lacks columns {missing}")
    if (ex["reason"].str.strip() == "").any():
        raise ValueError(f"{path}: every position exemption needs a reason")
    return ex.reindex(columns=cols, fill_value="")


def _pair_keys(b0: pd.Series, b1: pd.Series) -> pd.Series:
    return pd.Series(["|".join(sorted((str(a), str(b)))) for a, b in zip(b0, b1)], index=b0.index)


def circuit_positions(net: dict, etys_lengths: pd.Series | None, cfg: dict,
                      exemptions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every judged line and transformer: straight-line distance between its bus positions against what it allows.

    Columns: component, bus0, bus1, length_km (lines: ETYS, else the network's own non-zero length; transformers
    and same-site sections without a length 0), length_source, straight_km, allowed_km, excess_km, inconsistent,
    exempt, exemption. A line with a length allows (1 + position_line_slack_rel) x length + position_line_slack_km;
    a transformer or same-site section allows position_transformer_max_km. Branches with an external or unplaced
    end are left out, as are inter-site lines without a length.
    """
    buses = net["buses"]
    gb = buses[~external_mask(buses)]
    lines = net["lines"].copy()
    etys = (etys_lengths if etys_lengths is not None else pd.Series(dtype=float)).reindex(lines.index)
    own = lines["length_km"] if "length_km" in lines else pd.Series(np.nan, index=lines.index)
    own = own.where(own > 0)
    if "length" in lines:
        own = own.fillna(lines["length"].where(lines["length"] > 0))
    length = etys.where(etys.notna() & (etys > 0), own)
    source = pd.Series(np.where(etys.notna() & (etys > 0), "etys_components",
                                np.where(own.notna(), "network", "")), index=lines.index)
    # A same-site line without a length is a bus section (judged as co-located) at 132 kV and above; below that
    # it is an offshore array string (33/66 kV collectors span the farm) and is not judged.
    v_nom = buses["v_nom"].astype(float)
    hv = (lines["bus0"].map(v_nom).fillna(0) >= 132.0) & (lines["bus1"].map(v_nom).fillna(0) >= 132.0)
    same_site = _site(lines["bus0"]).eq(_site(lines["bus1"])) & hv
    length = length.where(length.notna() | ~same_site, 0.0)
    source = source.where(~(length.notna() & source.eq("")), "same_site_section")
    lines = lines.assign(component="line", length_km=length, length_source=source)
    judged = lines[lines["length_km"].notna() & ((lines["length_km"] > 0) | same_site)]
    trafos = net["trafos"].assign(component="transformer", length_km=0.0, length_source="transformer")
    cols = ["component", "bus0", "bus1", "length_km", "length_source"]
    x = pd.concat([judged[cols], trafos[cols]])
    x = x[x["bus0"].isin(gb.index) & x["bus1"].isin(gb.index)].copy()
    x0, y0 = gb["x"].reindex(x["bus0"]).to_numpy(float), gb["y"].reindex(x["bus0"]).to_numpy(float)
    x1, y1 = gb["x"].reindex(x["bus1"]).to_numpy(float), gb["y"].reindex(x["bus1"]).to_numpy(float)
    x["straight_km"] = np.hypot(x0 - x1, y0 - y1) / 1e3
    has_length = x["component"].eq("line") & (x["length_km"] > 0)
    x["allowed_km"] = np.where(
        has_length,
        (1.0 + float(cfg["position_line_slack_rel"])) * x["length_km"] + float(cfg["position_line_slack_km"]),
        float(cfg["position_transformer_max_km"]))
    x["excess_km"] = x["straight_km"] - x["allowed_km"]
    x["inconsistent"] = x["excess_km"] > 0
    x["exemption"] = ""
    if exemptions is not None and len(exemptions):
        reason = dict(zip(_pair_keys(exemptions["bus0"], exemptions["bus1"]), exemptions["reason"]))
        x["exemption"] = _pair_keys(x["bus0"], x["bus1"]).map(reason).fillna("")
    x["exempt"] = x["exemption"].ne("")
    return x


def bus_positions(circuits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per bus: judged circuits, inconsistent circuits, and the verdict (misplaced / suspect / unresolved / ok).

    Circuits are counted per neighbouring site (parallel circuits and the busbars of one site are one
    neighbour, at their worst), and exempt circuits are left out. ``misplaced`` (FAIL): at least
    position_min_circuits neighbours, more than position_suspect_share of them over by more than
    position_fail_excess_km. ``suspect`` (WARN): the same share inconsistent at all. ``unresolved``: some
    inconsistent, not most (or one neighbour only, which cannot say which end is wrong).
    """
    c = circuits[~circuits["exempt"]] if "exempt" in circuits else circuits
    fail_km = float(cfg["position_fail_excess_km"])
    ends = pd.concat([c.assign(bus=c["bus0"], other=c["bus1"]), c.assign(bus=c["bus1"], other=c["bus0"])])
    ends = (ends.assign(other_site=_site(ends["other"])).sort_values("excess_km", ascending=False)
            .drop_duplicates(["bus", "other_site"]))
    ends = ends.assign(far=ends["excess_km"] > fail_km)
    out = ends.groupby("bus").agg(n_circuits=("inconsistent", "size"), n_inconsistent=("inconsistent", "sum"),
                                  n_far=("far", "sum"))
    bad = ends[ends["inconsistent"]]
    out["worst_excess_km"] = bad["excess_km"].groupby(bad["bus"]).max().reindex(out.index)
    out["inconsistent_with"] = bad.groupby("bus")["other"].apply(lambda s: "|".join(sorted(set(s)))).reindex(out.index)
    enough = out["n_circuits"] >= int(cfg["position_min_circuits"])
    share = float(cfg["position_suspect_share"])
    out["verdict"] = np.select(
        [enough & (out["n_far"] / out["n_circuits"] > share),
         enough & (out["n_inconsistent"] / out["n_circuits"] > share),
         out["n_inconsistent"] > 0],
        ["misplaced", "suspect", "unresolved"], "ok")
    return out


def _crosses(geom, p0, p1) -> bool:
    """True when the straight segment p0-p1 cuts ``geom`` an odd number of times."""
    from shapely.geometry import LineString

    if tuple(p0) == tuple(p1):
        return False
    inter = LineString([tuple(p0), tuple(p1)]).intersection(geom)
    k = 0 if inter.is_empty else (len(inter.geoms) if hasattr(inter, "geoms") else 1)
    return k % 2 == 1


def estimate_bus_position(bus: str, circuits: pd.DataFrame, xy: pd.DataFrame) -> tuple[float, float, float]:
    """Where ``bus``'s judged circuits put it (OSGB metres), the other ends held where they are.

    Minimises the summed excess over each circuit's allowance (an L1 hinge, so one odd circuit does not drag a bus
    its other circuits hold) plus a small pull to the current position. Returns (x, y, moved_km).
    """
    from scipy.optimize import minimize

    c = circuits[(circuits["bus0"].eq(bus) | circuits["bus1"].eq(bus)) & ~circuits.get("exempt", False)]
    other = np.where(c["bus0"].eq(bus), c["bus1"], c["bus0"])
    q = xy.loc[other, ["x", "y"]].to_numpy(float)
    allow = c["allowed_km"].to_numpy(float) * 1e3
    p0 = xy.loc[bus, ["x", "y"]].to_numpy(float)

    def f(p):
        return np.maximum(np.hypot(*(q - p).T) - allow, 0).sum() + 1e-3 * np.hypot(*(p - p0))

    best = None
    for start in [p0, *q, q.mean(axis=0)]:
        r = minimize(f, start, method="Nelder-Mead", options={"xatol": 10.0, "fatol": 1.0, "maxiter": 2000})
        if best is None or r.fun < best.fun - 1.0:
            best = r
    return float(best.x[0]), float(best.x[1]), float(np.hypot(*(best.x - p0)) / 1e3)


def position_rows(net: dict, etys_lengths: pd.Series | None, cfg: dict, *, geoms: dict | None = None,
                  boundaries: list | None = None, bus_mw: pd.Series | None = None,
                  point_region=None, point_zone=None,
                  exemptions: pd.DataFrame | None = None) -> tuple[list[dict], pd.DataFrame]:
    """Rows of check 7 (see the module docstring) and the per-bus table.

    Args:
        geoms, boundaries: boundary polylines and the boundaries judged (crossings and side switches).
        bus_mw: MW carried per bus (generators, storage, electrolysers, interconnectors).
        point_region, point_zone: callables (x array, y array) -> labels; the region / zone flip test is skipped
            when None.
        exemptions: :func:`position_exemptions`.
    """
    circuits = circuit_positions(net, etys_lengths, cfg, exemptions)
    per_bus = bus_positions(circuits, cfg)
    geoms = {b: geoms[b] for b in (boundaries or []) if geoms and b in geoms}
    xy = net["buses"][["x", "y"]].astype(float)
    bus_mw = bus_mw if bus_mw is not None else pd.Series(dtype=float)
    rows = []
    # circuits
    edge_km, tr_fail_km = float(cfg["position_boundary_excess_km"]), float(cfg["position_fail_transformer_km"])
    bad = circuits[circuits["inconsistent"]].sort_values(["bus0", "bus1"])
    share = (per_bus["n_inconsistent"] / per_bus["n_circuits"]).to_dict()
    far_ends: set = set()
    for name, r in bad.iterrows():
        crossed = sorted(b for b, g in geoms.items()
                         if _crosses(g, xy.loc[r["bus0"]].to_numpy(), xy.loc[r["bus1"]].to_numpy()))
        why = []
        if not r["exempt"]:
            if r["component"] == "transformer" or r["length_source"] == "same_site_section":
                if r["straight_km"] > tr_fail_km:
                    why.append(f"ends more than {tr_fail_km:g} km apart")
            if r["excess_km"] > edge_km:
                # The end its other circuits disagree with more (both on a tie) is the one to place.
                s0, s1 = share.get(r["bus0"], 0.0), share.get(r["bus1"], 0.0)
                far_ends |= {b for b, s in ((r["bus0"], s0), (r["bus1"], s1)) if s >= max(s0, s1)}
                if crossed:
                    why.append(f"over by more than {edge_km:g} km and crosses {','.join(crossed)}")
        status = "INFO" if r["exempt"] else ("FAIL" if why else "WARN")
        note = (f"{r['bus0']}-{r['bus1']}: {r['straight_km']:.1f} km apart, allowed {r['allowed_km']:.1f} km "
                f"({r['length_source']} length {r['length_km']:.2f} km)"
                + (f"; crosses {','.join(crossed)}" if crossed else "")
                + (f"; FAIL: {'; '.join(why)}" if why else "")
                + (f"; exempt: {r['exemption']}" if r["exempt"] else ""))
        rows.append(make_row(cfg, "position", f"circuit_{r['component']}", name, "position", np.nan, status=status,
                             note=note, bus=r["bus0"], bus1=r["bus1"], length_km=round(float(r["length_km"]), 2),
                             straight_km=round(float(r["straight_km"]), 1), excess_km=round(float(r["excess_km"]), 1)))
    # buses: the verdict, plus the estimated position of every end of a circuit over by more than boundary_excess_km
    est = {}
    for bus in sorted(far_ends):
        ex, ey, moved = estimate_bus_position(bus, circuits, xy)
        est[bus] = (ex, ey, moved)
    if est and (point_region is not None or point_zone is not None):
        bl = list(est)
        cur_x, cur_y = xy.loc[bl, "x"].to_numpy(), xy.loc[bl, "y"].to_numpy()
        new_x, new_y = np.array([est[b][0] for b in bl]), np.array([est[b][1] for b in bl])
        reg = (dict(zip(bl, point_region(cur_x, cur_y))), dict(zip(bl, point_region(new_x, new_y)))) \
            if point_region is not None else ({}, {})
        zon = (dict(zip(bl, point_zone(cur_x, cur_y))), dict(zip(bl, point_zone(new_x, new_y)))) \
            if point_zone is not None else ({}, {})
    else:
        reg, zon = ({}, {}), ({}, {})
    for bus in sorted(set(per_bus.index[per_bus["verdict"].ne("ok")]) | set(est)):
        r = per_bus.loc[bus] if bus in per_bus.index else None
        verdict = r["verdict"] if r is not None else "ok"
        why = []
        if verdict == "misplaced":
            why.append(f"{int(r['n_far'])} of {int(r['n_circuits'])} judged circuits over by more than "
                       f"{float(cfg['position_fail_excess_km']):g} km")
        extra = {}
        if bus in est:
            ex, ey, moved = est[bus]
            extra = dict(est_x=round(ex), est_y=round(ey), moved_km=round(moved, 1))
            if moved > edge_km:
                p0, p1 = xy.loc[bus].to_numpy(), np.array([ex, ey])
                if reg[0].get(bus) != reg[1].get(bus):
                    why.append(f"its circuits put it in region {reg[1].get(bus)}, not {reg[0].get(bus)}")
                if zon[0].get(bus) != zon[1].get(bus):
                    why.append(f"its circuits put it in zone {zon[1].get(bus)}, not {zon[0].get(bus)}")
                mw = float(bus_mw.get(bus, 0.0))
                switched = sorted(b for b, g in geoms.items() if _crosses(g, p0, p1))
                if switched and mw > float(cfg["position_fail_mw"]):
                    why.append(f"carries {mw:.0f} MW and its circuits put it across {','.join(switched)}")
                extra["sides_switched"] = ",".join(switched)
        if r is None and not why:
            continue
        # Each inconsistent circuit already WARNs; a bus WARNs when most of its neighbours disagree with it.
        status = "FAIL" if why else ("WARN" if verdict == "suspect" else "INFO")
        label = {"misplaced": "MISPLACED", "suspect": "SUSPECT position",
                 "unresolved": "some circuits inconsistent (not most, or one circuit: which end is wrong cannot be "
                               "told)", "ok": "circuits consistent"}[verdict]
        note = (f"{label}" + (f": {int(r['n_inconsistent'])} of {int(r['n_circuits'])} judged circuits inconsistent; "
                              f"with {r['inconsistent_with']}" if r is not None and r["n_inconsistent"] else "")
                + (f"; FAIL: {'; '.join(why)}" if why else ""))
        rows.append(make_row(cfg, "position", "bus", bus, "position", np.nan, status=status, note=note, bus=bus,
                             n_circuits=int(r["n_circuits"]) if r is not None else 0,
                             n_inconsistent=int(r["n_inconsistent"]) if r is not None else 0,
                             excess_km=(round(float(r["worst_excess_km"]), 1)
                                        if r is not None and pd.notna(r["worst_excess_km"]) else np.nan),
                             mw=round(float(bus_mw.get(bus, 0.0)), 1), **extra))
        per_bus.loc[bus, "status"] = status
    if not len(rows):
        rows.append(make_row(cfg, "position", "bus", "all", "position", np.nan, status="PASS",
                             note=f"all {len(circuits)} judged circuits are consistent with their bus positions"))
    return rows, per_bus


# ── The checks ──────────────────────────────────────────────────────────────
def run_checks(net: dict, *, cfg: dict, es5: pd.DataFrame, geoms: dict, include: dict | None,
               gsp_polygons: str | Path, gsp_group_region: str | Path, landing_files: list,
               ic_reference: pd.DataFrame | None, ic_composition: pd.DataFrame | None, model_year: int,
               bus_zone: pd.DataFrame | None = None, bus_region: pd.Series | None = None,
               etys_lengths: pd.Series | None = None, point_region=None,
               point_zone=None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """All six checks on one network.

    Args:
        net: tables from :func:`read_network` / :func:`network_tables`.
        cfg: :func:`fleet_location_config` output.
        es5: :func:`read_es5` output (technology x region, MW).
        geoms: boundary polylines (``secured_boundaries.load_boundary_geometries``).
        include: the scenario's ``secured_capability.include``.
        gsp_polygons, gsp_group_region: the build's region files (``fleet.spatial_proxy.regional_allocation``).
        landing_files: offshore landing references (the build's crosswalk, then the committed additions).
        ic_reference: NESO register landings (``interconnectors_2030.csv``), or None.
        ic_composition: the CP30 cable composition, or None.
        model_year: the cables in the composition up to this year count.
        bus_zone, bus_region: precomputed bus maps (tests); computed when None.
        etys_lengths: line name -> ETYS length (km) for check 7 (:func:`etys_line_lengths`).
        point_region, point_zone: (x, y) -> CP30 region / CP.08 zone callables for check 7 (tests); built from
            gsp_polygons / gsp_group_region and the zones file when None (skipped when those are missing).

    Returns:
        (table, summary, buses with zone, region and sides).
    """
    row = lambda *a, **k: make_row(cfg, *a, **k)  # noqa: E731
    buses = net["buses"]
    report, main = list(cfg["report_boundaries"]), list(cfg["main_side_boundaries"])
    sides = boundary_sides(net, geoms, sorted(set(report) | set(ZONE_KEYS) | set(main)), include)
    bz = bus_zone if bus_zone is not None else bus_zones(net, sides, cfg["zones_file"], int(cfg["graph_cutoff"]))
    buses = buses.join(bz[["neso_zone"]]).join(sides.add_prefix("side_"))
    buses["region"] = (bus_region.reindex(buses.index) if bus_region is not None
                       else bus_regions(buses, gsp_polygons, gsp_group_region))
    cap = capacity_rows(net)
    cap["zone"], cap["region"] = cap.bus.map(buses.neso_zone), cap.bus.map(buses.region)
    rows = []

    def side_cols(bus, names):
        return {b: int(buses.at[bus, "side_" + b]) for b in names if "side_" + b in buses}

    # 1 zone capacity
    zt = cap.pivot_table(index="zone", columns="tech", values="p_nom", aggfunc="sum").reindex(columns=TECH_ORDER).fillna(0)
    zt = zt.reindex(sorted(zt.index, key=lambda z: int(str(z)[1:])))
    for z, r in zt.iterrows():
        for t, v in r.items():
            if v > 0:
                rows.append(row("zone_capacity", "neso_zone", z, t, v, note="CP30 publishes no zonal capacity"))
    unzoned = cap[cap.zone.isna()].p_nom.sum()
    rows.append(row("zone_capacity", "neso_zone", "unzoned", "all", unzoned, 0.0,
                    status="PASS" if unzoned < cfg["min_mw"] else "WARN", note="GB capacity on buses without a NESO zone"))

    # 2 ES5 regions
    rt = cap.pivot_table(index="region", columns="tech", values="p_nom", aggfunc="sum").reindex(columns=TECH_ORDER).fillna(0)
    rt["Storage (all)"] = rt["Battery"] + rt["LDES/PSH"]
    for t5, ours_t in ES5_TECH.items():
        if t5 not in es5.index:
            continue
        for reg in es5.columns:
            rows.append(row("es5_region", "cp30_region", reg, ours_t, rt[ours_t].get(reg, 0.0), es5.at[t5, reg],
                            note="region = GSP polygon of the bus (spatial_allocation.bus_cp30_regions)"))
    for reg in rt.index:
        for t in TECH_ORDER:
            if t not in ES5_TECH.values() and rt.at[reg, t] > 0:
                rows.append(row("es5_region", "cp30_region", reg, t, rt.at[reg, t], note="not published by ES5"))
    unreg = cap[cap.region.isna()].p_nom.sum()
    rows.append(row("es5_region", "cp30_region", "no region", "all", unreg, 0.0,
                    status="PASS" if unreg < cfg["min_mw"] else "WARN", note="GB capacity on buses without a CP30 region"))

    # 3 region vs boundary side
    for reg in list(SCOTLAND_REGIONS) + sorted(EW_REGIONS):
        want = SCOTLAND_REGIONS.get(reg, {"B6": -1})
        if not all("side_" + b in buses for b in want):
            continue
        sub = cap[cap.region.eq(reg)]
        bad = np.zeros(len(sub), bool)
        for b, sg in want.items():
            bad |= sub.bus.map(buses["side_" + b]).ne(sg).to_numpy()
        wrong = sub[bad].groupby("tech").p_nom.sum()
        for t, v in wrong.items():
            st = "FAIL" if v > cfg["fail_mw"] else ("WARN" if v > cfg["region_side_warn_mw"] else "PASS")
            buses_bad = sub[bad & sub.tech.eq(t).to_numpy()].groupby("bus").p_nom.sum().sort_values(ascending=False)
            rows.append(row("region_boundary", "cp30_region", reg, t, v, 0.0, status=st,
                            note=f"MW of region {reg} not on its side {want}; buses: "
                                 + "; ".join(f"{b} {m:.0f}" for b, m in buses_bad.head(8).items())))
        if not len(wrong):
            rows.append(row("region_boundary", "cp30_region", reg, "all", 0.0, 0.0, status="PASS",
                            note=f"every MW of {reg} is on its side {want}"))

    # 4 remote side per boundary
    remote = {}
    for b in report:
        if "side_" + b in buses:
            m = cap.bus.map(buses["side_" + b]).eq(1)
            remote[b] = cap[m].groupby("tech").p_nom.sum().reindex(TECH_ORDER).fillna(0)
    remote = pd.DataFrame(remote, index=TECH_ORDER).fillna(0)
    remote.loc["Storage (all)"] = remote.loc["Battery"] + remote.loc["LDES/PSH"]
    tech5 = {v_: k for k, v_ in ES5_TECH.items()}
    for b in remote.columns:
        for t, v in remote[b].items():
            ref = np.nan
            if b in ES5_REMOTE and tech5.get(t) in es5.index:
                ref = float(es5.loc[tech5[t], [r for r in ES5_REMOTE[b] if r in es5.columns]].sum())
            if v > 0 or np.isfinite(ref):
                rows.append(row("boundary_remote_side", "boundary", b, t, v, ref,
                                note="+1 side (north / inside)" + (f"; ES5 {' + '.join(ES5_REMOTE[b])}"
                                                                   if np.isfinite(ref) else "")))

    # 5 offshore landings
    g = net["gens"]
    off = g[g.carrier.eq("wind_offshore")].copy()
    ref = landing_reference(landing_files)
    off["rid"] = off.repd_ref_id.astype(str).str.replace(r"\.0$", "", regex=True)
    named = off[off.fleet_origin.eq("named") & off.rid.ne("")]
    for (rid, bus), grp in named.groupby(["rid", "bus"]):
        mw = grp.p_nom.sum()
        refs = _split(ref.get(rid, ""))
        if not refs:
            st, note = ("WARN" if mw >= cfg["unreferenced_warn_mw"] else "INFO"), "no landing reference"
        elif bus in refs:
            st, note = "PASS", "at its reference landing"
        else:
            rb = [r for r in refs if r in buses.index]
            crossed = sorted({b for r in rb for b in main
                              if "side_" + b in buses and buses.at[r, "side_" + b] != buses.at[bus, "side_" + b]})
            st = "FAIL" if crossed else "WARN"
            note = f"reference landing {'|'.join(refs)}" + (f" is across {','.join(crossed)}" if crossed else
                                                           " on the same boundary sides")
        if mw >= cfg["offshore_list_mw"] or st in ("WARN", "FAIL"):
            rows.append(row("offshore_landing", "named_farm", str(grp.site.iloc[0]).replace("\xa0", " ").strip(),
                            "Offshore wind", mw, status=st, note=note, bus=bus, zone=buses.at[bus, "neso_zone"],
                            region=buses.at[bus, "region"], farm_lat=round(grp.lat.mean(), 3),
                            farm_lon=round(grp.lon.mean(), 3), **side_cols(bus, main)))
    prox = off[~off.index.isin(named.index)]
    for bus, grp in prox.groupby("bus"):
        mw = grp.p_nom.sum()
        if mw >= cfg["offshore_list_mw"]:
            rows.append(row("offshore_landing", "proxy_bus", bus, "Offshore wind", mw, status="INFO",
                            note=f"{len(grp)} proxy/refill rows; ES5 regional capacity at a landing bus", bus=bus,
                            zone=buses.at[bus, "neso_zone"], region=buses.at[bus, "region"],
                            farm_lat=round(grp.lat.mean(), 3), farm_lon=round(grp.lon.mean(), 3),
                            **side_cols(bus, main)))

    # 6 interconnectors
    ic = interconnector_links(net)
    icref = ic_reference if ic_reference is not None else pd.DataFrame()
    comp = ic_composition.copy() if ic_composition is not None else pd.DataFrame()
    if len(comp):
        comp["cp30_mw"] = pd.to_numeric(comp["cp30_mw"], errors="coerce")
        fy = pd.to_numeric(comp["first_year"], errors="coerce")
        comp = comp[comp.action.isin(["keep", "add"]) & (fy.isna() | (fy <= model_year))]
    for _, r in ic.iterrows():
        name, bus = r["name"], r["bus"]
        cref = comp[comp.name.str.lower().eq(str(name).lower())] if len(comp) else pd.DataFrame()
        cap_ref = float(cref.cp30_mw.iloc[0]) if len(cref) else np.nan
        lref = icref[icref.composition_name.str.lower().eq(str(name).lower())] if len(icref) else pd.DataFrame()
        acc = _split(lref.accepted_buses.iloc[0]) if len(lref) else []
        land_ok = (bus in acc) if acc else None
        st = status_of(r.p_nom, cap_ref, cfg)
        if land_ok is False:
            st = "FAIL"
        note = (f"landing {bus} {'in' if land_ok else 'NOT in' if land_ok is False else 'no reference for'} "
                f"accepted {('|'.join(acc)) or '-'}; " + (lref.source.iloc[0] if len(lref) else ""))
        rows.append(row("interconnector", "cable", name, "Interconnector", r.p_nom, cap_ref, status=st, note=note,
                        bus=bus, zone=buses.at[bus, "neso_zone"], region=buses.at[bus, "region"],
                        **side_cols(bus, main + ["LE1", "SC1"])))
    if len(comp):
        rows.append(row("interconnector", "total", "GB", "Interconnector", ic.p_nom.sum(), comp.cp30_mw.sum(),
                        note="CP30 composition (equals ES1)"))

    # 7 position consistency
    bus_mw = cap.groupby("bus").p_nom.sum()
    pos_rows, per_bus = position_rows(
        net, etys_lengths, cfg, geoms=geoms, boundaries=sorted(set(report) | set(main) | set(ZONE_KEYS)),
        bus_mw=bus_mw, point_region=point_region if point_region is not None else _point_region_fn(gsp_polygons, gsp_group_region),
        point_zone=point_zone if point_zone is not None else _point_zone_fn(cfg["zones_file"]),
        exemptions=position_exemptions(cfg["position_exemptions_file"]))
    rows += pos_rows

    out = pd.DataFrame(rows)
    summary = {
        "config": cfg,
        "status_counts": out.status.value_counts().to_dict(),
        "status_counts_by_check": out.groupby(["check", "status"]).size().unstack(fill_value=0).to_dict("index"),
        "warn_fail": out[out.status.isin(["WARN", "FAIL"])].replace({np.nan: None}).to_dict("records"),
        "zone_by_technology_mw": zt.round(1).to_dict("index"),
        "region_by_technology_mw": rt.round(1).to_dict("index"),
        "boundary_remote_side_mw": remote.round(1).to_dict(),
        "bus_zone_mapping": bz["mapping"].value_counts().to_dict() if "mapping" in bz else {},
        "buses_reconciled_to_boundary_sides": int(bz["reconciled"].sum()) if "reconciled" in bz else 0,
        "position_misplaced_buses": sorted(per_bus.index[per_bus["verdict"].eq("misplaced")]),
        "position_suspect_buses": sorted(per_bus.index[per_bus["verdict"].eq("suspect")]),
        "position_unresolved_buses": sorted(per_bus.index[per_bus["verdict"].eq("unresolved")]),
        "position_fail_buses": sorted(per_bus.index[per_bus.get("status", pd.Series(dtype=str)).eq("FAIL")]),
    }
    return out, summary, buses


def fleet_location_failures(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["status"].eq("FAIL")]


def run(network_path: str | Path, out_csv: str | Path, out_json: str | Path, *, es5_path: str | Path, pathway: str,
        model_year: int, config: dict | None = None, include: dict | None = None,
        geometry_file: str | Path | None = None, gsp_polygons: str | Path = "data/network/GSP/GSP_regions_27700_20250109.geojson",
        gsp_group_region: str | Path = "data/network/ETYS/cp30_es5_region_by_gsp_group.csv",
        landing_crosswalk: str | Path | None = "data/network/ETYS/repd_offshore_landing_crosswalk.csv",
        ic_composition: str | Path | None = "data/interconnectors/cp30_interconnector_composition.csv",
        etys_components: str | Path | None = None, out_bus_csv: str | Path | None = None, log_path: str | None = None) -> pd.DataFrame:
    """Run the checks on a network file, write the CSV and JSON, and raise on FAIL rows when ``fail_on_fail``."""
    from scripts.solve.secured_boundaries import DEFAULT_GEOMETRY, load_boundary_geometries
    from scripts.utilities.logging_config import setup_logging

    log = setup_logging(log_path or "validate_cp30_fleet_location")
    cfg = fleet_location_config(config)
    out_csv, out_json = Path(out_csv), Path(out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not cfg["enabled"]:
        table = pd.DataFrame([make_row(cfg, "fleet_location", "all", "checks disabled", "all", 0.0, status="INFO")])
        table.to_csv(out_csv, index=False)
        out_json.write_text(json.dumps({"network": str(network_path), "enabled": False}, indent=1), encoding="utf-8")
        return table
    net = read_network(network_path)
    geometry_file = geometry_file or DEFAULT_GEOMETRY
    table, summary, buses = run_checks(
        net, cfg=cfg, es5=read_es5(es5_path, pathway), geoms=load_boundary_geometries(geometry_file),
        include=include, gsp_polygons=gsp_polygons, gsp_group_region=gsp_group_region,
        landing_files=[landing_crosswalk, cfg["landing_additions_file"]],
        ic_reference=(pd.read_csv(cfg["ic_reference_file"], dtype=str)
                      if cfg["ic_reference_file"] and Path(cfg["ic_reference_file"]).exists() else None),
        ic_composition=pd.read_csv(ic_composition, comment="#", dtype=str) if ic_composition else None,
        model_year=int(model_year), etys_lengths=etys_line_lengths(etys_components))
    table.to_csv(out_csv, index=False)
    summary = {"network": str(network_path), "pathway": pathway, "model_year": int(model_year),
               "inputs": {"es5": str(es5_path), "geometry": str(geometry_file), "include": include or {},
                          "gsp_polygons": str(gsp_polygons), "gsp_group_region": str(gsp_group_region),
                          "landing_crosswalk": str(landing_crosswalk), "ic_composition": str(ic_composition),
                          "etys_components": str(etys_components)},
               **summary}
    out_json.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    if out_bus_csv:
        buses.to_csv(out_bus_csv)
    log.info("CP30 fleet location (%s): %s", Path(network_path).name, json.dumps(summary["status_counts"]))
    for r in summary["warn_fail"]:
        log.warning("%-4s %-22s %-40s %-15s ours %s cp30 %s | %s", r["status"], r["check"], r["key"],
                    r["technology"], r["ours_mw"], r["cp30_mw"], str(r["note"])[:160])
    fails = fleet_location_failures(table)
    if len(fails) and cfg["fail_on_fail"]:
        # Snakemake deletes a failed rule's outputs: keep the table beside it for the review.
        kept = out_csv.with_name(out_csv.stem + "_failed_run.csv")
        table.to_csv(kept, index=False)
        raise ValueError(f"CP30 fleet location validation failed on {len(fails)} rows (full table kept at {kept}):\n"
                         + fails[["check", "key", "technology", "ours_mw", "cp30_mw", "note"]].to_string(index=False))
    return table


def scenario_settings(scenario: dict) -> dict:
    """What the check reads from a scenario config (the rule passes only this as params)."""
    secured = ((((scenario.get("network") or {}).get("transmission") or {}).get("outage_schedule") or {})
               .get("secured_capability") or {})
    return {"cp30_fleet_location_validation": (scenario.get("validation") or {}).get("fleet_location") or {},
            "include": dict(secured.get("include") or {}),
            "geometry_file": secured.get("geometry_file"),
            "pathway": scenario.get("CP30_pathway") or "Further Flex and Renewables",
            "modelled_year": int(scenario.get("modelled_year") or 2030)}


if "snakemake" in globals():
    _settings = dict(snakemake.params.settings)

    def _one(value):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return str(value) if value else None

    run(snakemake.input.network, snakemake.output.csv, snakemake.output.json,
        es5_path=_one(snakemake.input.cp30_es5), pathway=_settings["pathway"], model_year=_settings["modelled_year"],
        config=_settings["cp30_fleet_location_validation"], include=_settings["include"],
        geometry_file=_one(snakemake.input.boundaries), gsp_polygons=_one(snakemake.input.gsp_polygons),
        gsp_group_region=_one(snakemake.input.gsp_group_regions),
        landing_crosswalk=_one(snakemake.input.offshore_crosswalk),
        ic_composition=_one(getattr(snakemake.input, "ic_composition", None)),
        etys_components=_one(getattr(snakemake.input, "etys_components", None)),
        log_path=snakemake.log[0] if snakemake.log else None)
elif __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--network", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-bus-csv", default=None, help="optional: bus -> zone, region, sides")
    p.add_argument("--cp30-es5", default="resources/CP30/CP30_ES5_regional.csv",
                   help="ES5: the extracted CSV (source_data.files.es5) or the CP30 workbook")
    p.add_argument("--pathway", default="Further Flex and Renewables")
    p.add_argument("--modelled-year", type=int, default=2030)
    p.add_argument("--gsp-polygons", default="data/network/GSP/GSP_regions_27700_20250109.geojson")
    p.add_argument("--gsp-group-region", default="data/network/ETYS/cp30_es5_region_by_gsp_group.csv")
    p.add_argument("--boundary-geometry", default=None, help="default secured_boundaries.DEFAULT_GEOMETRY")
    p.add_argument("--boundary-include", default=None,
                   help="JSON {boundary: [{component, name, sign}]} (default: the CP30 scenarios' EGL2 on B4)")
    p.add_argument("--ic-composition", default="data/interconnectors/cp30_interconnector_composition.csv")
    p.add_argument("--landing-crosswalk", default="data/network/ETYS/repd_offshore_landing_crosswalk.csv")
    p.add_argument("--etys-components", default="resources/network/ETYS_2024_components.csv",
                   help="the build's ETYS components table (line lengths for the position check); '' for none")
    p.add_argument("--fail-on-fail", action="store_true", help="exit non-zero when any row FAILs")
    p.add_argument("--log", default=None)
    args = p.parse_args()
    _include = (json.loads(args.boundary_include) if args.boundary_include else
                {"B4": [{"component": "link", "name": "DRAX41_PEHE4J_HVDC_2029", "sign": -1.0}]})
    try:
        run(args.network, args.out_csv, args.out_json, es5_path=args.cp30_es5, pathway=args.pathway,
            model_year=args.modelled_year, config={"fail_on_fail": args.fail_on_fail}, include=_include,
            geometry_file=args.boundary_geometry, gsp_polygons=args.gsp_polygons,
            gsp_group_region=args.gsp_group_region, landing_crosswalk=args.landing_crosswalk,
            ic_composition=args.ic_composition or None, etys_components=args.etys_components or None,
            out_bus_csv=args.out_bus_csv, log_path=args.log)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
