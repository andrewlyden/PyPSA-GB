"""Boundary-respecting busmap for the FES 2050 clustered ETYS network (task Y5).

The 2050 network is ETYS clustered to a few hundred buses, with the NESO
secured boundary capabilities as the binding transfer limits (plan decision
5). Those limits are enforced as sums of flow over the branches that cross
each ETYS boundary polyline (``scripts/solve/secured_boundaries.py``). A
cluster that straddles a boundary turns some crossing branches into internal
ones, so the boundary loses them and the sum-of-flow limit no longer sees the
transfer. This module builds a busmap in which no cluster straddles any
boundary.

Method:

1. **GSP regions.** Every GB AC bus is placed in the GSP region that contains
   it (the ``gsp_spatial`` preset's regions,
   ``data/network/GSP/GSP_regions_27700_20250109.geojson``). A bus outside
   every region (offshore platforms, the coast) goes to the nearest region by
   polygon distance; the ``gsp_spatial`` preset uses the nearest region
   centroid, which can send an offshore bus to an inland region.
2. **Boundary sides.** Each boundary polyline is extended at both ends along
   its end chords until it is far outside GB, so that it divides the plane.
   A bus's side of the boundary is the parity of the number of times the
   segment from the bus to a fixed far-away reference point crosses the
   extended polyline. With the ends at infinity, a branch crosses the
   polyline an odd number of times exactly when its two buses have different
   parities, which is the crossing test the solve uses (to within branches
   that pass beyond the end of the published polyline).
3. **Split.** A GSP region whose buses do not all share one side of every
   boundary is split by the side signature. The cluster name is the GSP name,
   with ``__<boundary>-<side>`` for each boundary that cuts that region.
4. **Optional merge** (``max_clusters``): the cluster with fewest buses is
   merged into the neighbouring cluster (by branch) with the same side
   signature and the largest connecting rating, until the count is at or
   under the target. Merging never joins different side signatures, so it
   cannot create a straddle.

Foreign buses and non-AC buses (hydrogen, heat, DC) stay individual, as in
``consumer_bill_14_zone.py``, so interconnector endpoints and carrier buses
are never merged with GB AC buses.

Use through the repository's clustering path (``method: custom``)::

    clustering:
      method: custom
      function_path: "scripts/network_clustering/boundary_busmap.py"
      function_name: "cluster_buses"
      convert_transformers_to_lines: true
      boundary_busmap:              # all optional; defaults shown
        gsp_regions_path: "data/network/GSP/GSP_regions_27700_20250109.geojson"
        gsp_column: "GSPs"
        geometry_file: "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson"
        capability_file: "data/network/etys_boundary_capability.csv"
        boundaries: null            # null = every boundary with a capability row (21)
        max_clusters: 300           # null = no merge step
      strategies:                   # needed on networks built after 27 Sep 2026
        line: {...}                 # see LINE_STRATEGIES below

Networks built after 27 Sep 2026 carry seasonal rating and provenance columns
on lines. pypsa 1.0.7 refuses to aggregate a column whose values differ within
a cluster unless it has a strategy, so clustering any current ETYS network
fails without ``LINE_STRATEGIES`` (the ``cluster_network.py`` defaults do not
cover them). A scenario's ``strategies.line`` replaces the default line dict
as a whole, so it must be the full dict.

The committed busmap ``data/clustering/fes2050_boundary_busmap.csv`` is built
by this module's command line on the 2030 ETYS base network, as the provenance
record and as a regression reference for Y6's 2050 build.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scripts.solve.secured_boundaries import (
        DEFAULT_CAPABILITY,
        DEFAULT_GEOMETRY,
        find_boundary_crossings,
        load_boundary_geometries,
    )
except ImportError:  # loaded by file path from a working directory without the package
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.solve.secured_boundaries import (  # noqa: E402
        DEFAULT_CAPABILITY,
        DEFAULT_GEOMETRY,
        find_boundary_crossings,
        load_boundary_geometries,
    )

logger = logging.getLogger(__name__)

DEFAULT_GSP_REGIONS = "data/network/GSP/GSP_regions_27700_20250109.geojson"
DEFAULT_GSP_COLUMN = "GSPs"
# Ends of each boundary polyline are pushed this far out (metres), and the
# parity reference sits this far from GB, so that no triangle bus-bus-reference
# contains a polyline end (the condition for parity to equal crossing).
EXTENSION_M = 1.0e7
REFERENCE_POINTS = ((-2.0e6, -2.5e6), (3.0e6, 3.5e6))
# Length of the end chord (metres) that sets the direction of each extension.
END_CHORD_M = 20000.0
# Line aggregation strategies for clustering a current ETYS network (copy into
# the scenario's clustering.strategies.line). Seasonal ratings add like s_nom;
# text provenance columns keep the first value. r and x are always combined in
# parallel per unit by cluster_network.py, whatever is given here.
LINE_STRATEGIES = {
    "voltage_kv": "max", "v_nom": "max", "s_nom": "sum", "length": "mean", "length_km": "mean",
    "b": "sum", "s_nom_winter": "sum", "s_nom_spring": "sum", "s_nom_summer": "sum",
    "s_nom_autumn": "sum", "etys_licensee": "first", "etys_licensee_source": "first",
    "etys_topology_source": "first", "cp30_project_code": "first", "cp30_source": "first",
    "seasonal_rating_source": "first",
}
BUSMAP_SOURCE = (
    "scripts/network_clustering/boundary_busmap.py: GSP region "
    "(data/network/GSP/GSP_regions_27700_20250109.geojson) split by side of each ETYS boundary "
    "(data/network/ETYS/etys_boundaries_mar25_osgb36.geojson)"
)


# ── configuration ───────────────────────────────────────────────────────────────

def boundary_busmap_config(config: Optional[dict]) -> dict:
    """Resolve the ``boundary_busmap`` options with their defaults.

    Args:
        config: the clustering method config (may hold a ``boundary_busmap``
            sub-dict) or ``None``.

    Returns:
        dict with every option set.
    """
    cfg = dict(((config or {}).get("boundary_busmap") or {}))
    cfg.setdefault("gsp_regions_path", DEFAULT_GSP_REGIONS)
    cfg.setdefault("gsp_column", DEFAULT_GSP_COLUMN)
    cfg.setdefault("geometry_file", DEFAULT_GEOMETRY)
    cfg.setdefault("capability_file", DEFAULT_CAPABILITY)
    cfg.setdefault("boundaries", None)
    cfg.setdefault("max_clusters", None)
    return cfg


def capability_boundaries(capability_file: str = DEFAULT_CAPABILITY) -> List[str]:
    """Boundaries with a published capability (the ones the solve enforces)."""
    return sorted(set(pd.read_csv(capability_file)["boundary"].astype(str)))


# ── geometry ────────────────────────────────────────────────────────────────────

def extend_polyline(geom, distance_m: float = EXTENSION_M, chord_m: float = END_CHORD_M):
    """Extend an open polyline at both ends along its end chords.

    Args:
        geom: shapely LineString (OSGB36 metres).
        distance_m: how far to push each end out.
        chord_m: length of the end chord that sets each extension's direction.

    Returns:
        shapely LineString. A closed ring is returned unchanged (it already
        divides the plane).
    """
    from shapely.geometry import LineString

    coords = np.asarray(geom.coords, dtype=float)
    if np.allclose(coords[0], coords[-1]):
        return geom
    chord = min(chord_m, geom.length / 2.0)
    start_in = np.asarray(geom.interpolate(chord).coords[0])
    end_in = np.asarray(geom.interpolate(geom.length - chord).coords[0])
    d0 = coords[0] - start_in
    d1 = coords[-1] - end_in
    p0 = coords[0] + d0 / np.linalg.norm(d0) * distance_m
    p1 = coords[-1] + d1 / np.linalg.norm(d1) * distance_m
    return LineString(np.vstack([p0, coords, p1]))


def _parity(points: np.ndarray, extended, reference: Tuple[float, float]) -> np.ndarray:
    """Parity (0/1) of crossings of the segment point -> reference with ``extended``."""
    import shapely

    ref = np.tile(np.asarray(reference, dtype=float), (len(points), 1))
    segs = shapely.linestrings(np.stack([points, ref], axis=1))
    inter = shapely.intersection(segs, extended)
    counts = shapely.get_num_geometries(inter)
    counts = np.where(shapely.is_empty(inter), 0, counts)
    return (counts % 2).astype(int)


def boundary_sides(xy: pd.DataFrame, geoms: Dict[str, object],
                   boundaries: List[str]) -> pd.DataFrame:
    """Side (0/1) of every bus relative to every boundary.

    The side is the crossing parity to the first reference point, taken on the
    extended polyline. Side 0 is the reference side; it has no geographic
    meaning (the sign convention of the solve's crossings is separate).

    Args:
        xy: frame with ``x``, ``y`` in OSGB36 metres, indexed by bus.
        geoms: boundary name -> LineString.
        boundaries: boundary names to evaluate.

    Returns:
        frame indexed like ``xy``, one int column per boundary.

    Raises:
        ValueError: if the two reference points disagree on relative sides
            for any boundary (a degenerate crossing at a polyline vertex).
    """
    pts = xy[["x", "y"]].to_numpy(dtype=float)
    out = {}
    for b in boundaries:
        ext = extend_polyline(geoms[b])
        p1 = _parity(pts, ext, REFERENCE_POINTS[0])
        p2 = _parity(pts, ext, REFERENCE_POINTS[1])
        # The two references may be on opposite sides; relative sides must agree.
        flip = p1 ^ p2
        if flip.min() != flip.max():
            bad = xy.index[flip != np.bincount(flip).argmax()]
            raise ValueError(f"boundary {b}: side parity is ambiguous for buses {list(bad)[:10]}")
        out[b] = p1
    return pd.DataFrame(out, index=xy.index)


# ── GSP assignment ──────────────────────────────────────────────────────────────

def assign_gsp_regions(xy: pd.DataFrame, regions, column: str = DEFAULT_GSP_COLUMN) -> pd.DataFrame:
    """GSP region of every bus: containing region, else nearest by polygon distance.

    Args:
        xy: frame with ``x``, ``y`` in OSGB36 metres.
        regions: GeoDataFrame of GSP regions in EPSG:27700.
        column: region name column.

    Returns:
        frame with ``gsp_region``, ``gsp_assignment`` (``contains`` or
        ``nearest_region``) and ``gsp_distance_km`` (0 when contained).
    """
    import geopandas as gpd

    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy["x"], xy["y"]), index=xy.index,
                           crs="EPSG:27700")
    regions = regions.to_crs("EPSG:27700")[[column, "geometry"]].reset_index(drop=True)
    inside = gpd.sjoin(pts, regions, how="left", predicate="within")
    inside = inside[~inside.index.duplicated(keep="first")]
    out = pd.DataFrame(index=xy.index)
    out["gsp_region"] = inside[column].reindex(xy.index)
    out["gsp_assignment"] = np.where(out["gsp_region"].notna(), "contains", "nearest_region")
    out["gsp_distance_km"] = 0.0
    missing = out.index[out["gsp_region"].isna()]
    if len(missing):
        near = gpd.sjoin_nearest(pts.loc[missing], regions, how="left", distance_col="dist")
        near = near[~near.index.duplicated(keep="first")]
        out.loc[missing, "gsp_region"] = near[column].reindex(missing).values
        out.loc[missing, "gsp_distance_km"] = (near["dist"].reindex(missing) / 1000.0).values
    return out


# ── busmap ──────────────────────────────────────────────────────────────────────

def _special_buses(buses: pd.DataFrame) -> pd.Series:
    """Cluster names for foreign and non-AC buses (kept individual); NaN for GB AC."""
    carrier = buses.get("carrier", pd.Series("AC", index=buses.index)).fillna("AC").astype(str)
    carrier = carrier.replace("", "AC")
    country = buses.get("country", pd.Series("GB", index=buses.index)).fillna("").astype(str)
    out = pd.Series(np.nan, index=buses.index, dtype=object)
    non_ac = carrier.ne("AC")
    foreign = ~non_ac & country.ne("") & country.ne("GB")
    for bus in buses.index[non_ac]:
        out.at[bus] = f"{carrier.at[bus]}_{bus}"
    for bus in buses.index[foreign]:
        out.at[bus] = f"External_{bus}"
    return out


def _branch_table(network) -> pd.DataFrame:
    """bus0, bus1, rating for lines, transformers and links."""
    parts = []
    for comp, df, rating in (("line", network.lines, "s_nom"),
                             ("transformer", network.transformers, "s_nom"),
                             ("link", network.links, "p_nom")):
        if len(df):
            parts.append(pd.DataFrame({"component": comp, "name": df.index,
                                       "bus0": df["bus0"].values, "bus1": df["bus1"].values,
                                       "rating": pd.to_numeric(df[rating], errors="coerce").fillna(0.0).values}))
    if not parts:
        return pd.DataFrame(columns=["component", "name", "bus0", "bus1", "rating"])
    return pd.concat(parts, ignore_index=True)


def _merge_to_target(table: pd.DataFrame, network, max_clusters: int) -> pd.DataFrame:
    """Merge smallest clusters into same-signature neighbours until at most ``max_clusters``."""
    table = table.copy()
    gb = table["special"].isna()
    branches = _branch_table(network)
    frozen = set()
    absorbed: Dict[str, List[str]] = {}
    while table.loc[gb, "cluster"].nunique() > max_clusters:
        counts = table.loc[gb].groupby("cluster").size()
        counts = counts.drop(index=[c for c in frozen if c in counts.index])
        if counts.empty:
            logger.warning("boundary busmap: no further merge possible at %d clusters",
                           table.loc[gb, "cluster"].nunique())
            break
        smallest = counts.sort_values(kind="stable").index[0]
        sig = table.loc[table["cluster"] == smallest, "side_code"].iloc[0]
        cl = table["cluster"]
        c0 = branches["bus0"].map(cl)
        c1 = branches["bus1"].map(cl)
        touch = branches[(c0 == smallest) ^ (c1 == smallest)].copy()
        touch["other"] = np.where(c0.loc[touch.index] == smallest, c1.loc[touch.index],
                                  c0.loc[touch.index])
        sig_of = table.drop_duplicates("cluster").set_index("cluster")["side_code"]
        touch = touch[touch["other"].map(sig_of) == sig]
        touch = touch[touch["other"].isin(table.loc[gb, "cluster"])]
        if touch.empty:
            frozen.add(smallest)
            continue
        target = touch.groupby("other")["rating"].sum().sort_values(ascending=False,
                                                                      kind="stable").index[0]
        table.loc[table["cluster"] == smallest, "cluster"] = target
        absorbed.setdefault(target, []).extend([smallest] + absorbed.pop(smallest, []))
    for target, names in absorbed.items():
        table.loc[table["cluster"] == target, "merged_from"] = ";".join(sorted(names))
    return table


def build_boundary_busmap(network, config: Optional[dict] = None) -> pd.DataFrame:
    """Boundary-respecting busmap with its provenance columns.

    Args:
        network: PyPSA network with bus ``x``/``y`` in OSGB36 metres.
        config: clustering method config; see :func:`boundary_busmap_config`.

    Returns:
        frame indexed by bus with ``cluster`` first, then ``gsp_region``,
        ``gsp_assignment``, ``gsp_distance_km``, ``side_code`` (the bus's side
        of every boundary, in boundary order), ``split_by`` (boundaries that
        cut its GSP region), ``merged_from``, ``v_nom``, ``x``, ``y``,
        ``special``.
    """
    import geopandas as gpd

    cfg = boundary_busmap_config(config)
    boundaries = cfg["boundaries"] or capability_boundaries(cfg["capability_file"])
    geoms = load_boundary_geometries(cfg["geometry_file"])
    missing_geom = [b for b in boundaries if b not in geoms]
    if missing_geom:
        raise ValueError(f"boundary busmap: no geometry for {missing_geom} in {cfg['geometry_file']}")

    buses = network.buses
    x_range = float(buses["x"].max() - buses["x"].min())
    if x_range <= 1000.0:
        raise ValueError("boundary busmap needs bus x/y in OSGB36 metres (EPSG:27700)")
    special = _special_buses(buses)
    gb = special.isna()
    xy = buses.loc[gb, ["x", "y"]].astype(float)
    if xy.isna().any().any():
        raise ValueError(f"boundary busmap: GB AC buses without coordinates: "
                         f"{list(xy.index[xy.isna().any(axis=1)])[:10]}")

    regions = gpd.read_file(cfg["gsp_regions_path"])
    gsp = assign_gsp_regions(xy, regions, cfg["gsp_column"])
    sides = boundary_sides(xy, geoms, boundaries)
    code = "s" + sides.astype(str).agg("".join, axis=1)   # "s" keeps leading zeros in CSV

    table = pd.DataFrame(index=buses.index)
    table["cluster"] = special
    table = table.join(gsp)
    table["side_code"] = code.reindex(buses.index).fillna("")
    table["split_by"] = ""
    for region, idx in table.loc[gb].groupby("gsp_region").groups.items():
        s = sides.loc[idx]
        varying = [b for b in boundaries if s[b].nunique() > 1]
        if not varying:
            table.loc[idx, "cluster"] = str(region)
            continue
        suffix = s[varying].apply(lambda r: "__" + "_".join(f"{b}-{int(r[b])}" for b in varying), axis=1)
        table.loc[idx, "cluster"] = str(region) + suffix
        table.loc[idx, "split_by"] = ";".join(varying)
    table["merged_from"] = ""
    table["special"] = special
    table["v_nom"] = buses.get("v_nom")
    table["x"] = buses["x"]
    table["y"] = buses["y"]
    if cfg["max_clusters"]:
        table = _merge_to_target(table, network, int(cfg["max_clusters"]))
    if table["cluster"].isna().any():
        raise ValueError("boundary busmap failed to assign every bus")
    return table


def cluster_buses(network, config: Optional[dict] = None) -> pd.Series:
    """Custom clustering entry point (``method: custom`` in a scenario's clustering).

    Args:
        network: PyPSA network (bus x/y in OSGB36).
        config: the scenario's resolved clustering config.

    Returns:
        busmap Series, bus -> cluster name.
    """
    table = build_boundary_busmap(network, config)
    n_gb = table.loc[table["special"].isna(), "cluster"].nunique()
    split = table.loc[table["split_by"] != "", "gsp_region"].nunique()
    logger.info("boundary busmap: %d GB AC clusters (%d GSP regions split by a boundary), "
                "%d individual foreign/non-AC buses", n_gb, split, int(table["special"].notna().sum()))
    return table["cluster"].astype(str)


# ── verification ────────────────────────────────────────────────────────────────

def straddle_check(network, busmap: pd.Series, boundaries: List[str],
                   geometry_file: str = DEFAULT_GEOMETRY) -> pd.DataFrame:
    """Per boundary: crossing branches before clustering and how many a cluster swallows.

    Uses the solve's own crossing test (``find_boundary_crossings``) on the
    unclustered network.

    Args:
        network: unclustered network.
        busmap: bus -> cluster.
        boundaries: boundaries to check.
        geometry_file: boundary geometry.

    Returns:
        frame indexed by boundary with ``crossings``, ``internal`` (both ends
        in one cluster) and ``internal_names``.
    """
    geoms = load_boundary_geometries(geometry_file)
    rows = []
    for b in boundaries:
        cr = find_boundary_crossings(network, geoms[b])
        tables = {"line": network.lines, "transformer": network.transformers, "link": network.links}
        internal = []
        for _, r in cr.iterrows():
            t = tables[r["component"]]
            if busmap.get(t.at[r["name"], "bus0"]) == busmap.get(t.at[r["name"], "bus1"]):
                internal.append(r["name"])
        rows.append({"boundary": b, "crossings": len(cr), "internal": len(internal),
                     "internal_names": ";".join(internal)})
    return pd.DataFrame(rows).set_index("boundary")


def clustered_crossing_table(clustered, boundaries: List[str], capability: Optional[pd.Series] = None,
                             geometry_file: str = DEFAULT_GEOMETRY) -> pd.DataFrame:
    """Per boundary on the clustered network: crossing count, AC s_nom, DC p_nom, capability.

    Args:
        clustered: clustered network.
        boundaries: boundaries to report.
        capability: optional boundary -> capability (MW).
        geometry_file: boundary geometry.

    Returns:
        frame indexed by boundary.
    """
    geoms = load_boundary_geometries(geometry_file)
    rows = []
    for b in boundaries:
        cr = find_boundary_crossings(clustered, geoms[b])
        ac = cr.loc[cr["component"] != "link", "rating"]
        rows.append({"boundary": b, "crossings": len(cr),
                     "ac_crossings": int((cr["component"] != "link").sum()),
                     "dc_crossings": int((cr["component"] == "link").sum()),
                     "ac_s_nom_mva": float(ac.sum()),
                     "dc_p_nom_mw": float(cr.loc[cr["component"] == "link", "rating"].sum())})
    out = pd.DataFrame(rows).set_index("boundary")
    if capability is not None:
        out["capability_mw"] = capability.reindex(out.index)
    return out


def isolated_buses(network) -> List[str]:
    """Buses with no line, transformer or link attached."""
    used = set()
    for df in (network.lines, network.transformers, network.links):
        if len(df):
            used |= set(df["bus0"]) | set(df["bus1"])
    return sorted(set(network.buses.index) - used)


# ── command line: build the committed busmap ────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """Build ``data/clustering/fes2050_boundary_busmap.csv`` from a base network."""
    import pypsa

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--network", required=True, help="unclustered ETYS network (.nc), read only")
    ap.add_argument("--output", default="data/clustering/fes2050_boundary_busmap.csv")
    ap.add_argument("--summary", default="data/clustering/fes2050_boundary_busmap_summary.json")
    ap.add_argument("--max-clusters", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    network = pypsa.Network(args.network)
    cfg = {"boundary_busmap": {"max_clusters": args.max_clusters}}
    table = build_boundary_busmap(network, cfg)
    boundaries = capability_boundaries()
    check = straddle_check(network, table["cluster"], boundaries)
    out = table.drop(columns=["special"]).copy()
    out.insert(0, "bus_id", out.index)
    out["base_network"] = Path(args.network).name
    out["source"] = BUSMAP_SOURCE
    out["status"] = "built 30 Sep 2026 (Y5) on the 2030 ETYS base network; regenerate if the base bus set changes"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    gb = table["special"].isna()
    summary = {
        "base_network": str(args.network).replace("\\", "/"),
        "buses": int(len(table)),
        "gb_ac_clusters": int(table.loc[gb, "cluster"].nunique()),
        "gsp_regions_used": int(table.loc[gb, "gsp_region"].nunique()),
        "gsp_regions_split": int(table.loc[gb & (table["split_by"] != ""), "gsp_region"].nunique()),
        "buses_by_nearest_region": int((table["gsp_assignment"] == "nearest_region").sum()),
        "boundaries": boundaries,
        "side_code": "s + one digit per boundary in the order above (0 = reference side, 1 = other side)",
        "crossings_before": check["crossings"].to_dict(),
        "crossings_swallowed_by_a_cluster": check["internal"].to_dict(),
        "max_clusters": args.max_clusters,
        "method": BUSMAP_SOURCE,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
