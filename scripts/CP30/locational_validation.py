#!/usr/bin/env python3
"""Locational and network validation of a built CP30 network (15 Oct 2026).

The CP30 totals validation (``validate_cp30_network``) checks *how much* of each
technology the network carries. This module checks *where* it sits, so the
placement problems found by the 15 Oct 2026 audits are caught every time the
network is built:

1. ES5 regional totals by technology, by the region of the bus each unit sits on
   (GSP polygon), not the stored tag. Hydro and marine (not in ES5) against FES.
2. Offshore landings: named farms and refill rows at their TEC/crosswalk landing;
   FES proxies landing across a boundary from their source (the Blackhillock
   case); the offshore region tag against the landing bus's region.
3. Named stations at their TEC-register buses.
4. Electrolysers behind B4, B6 and B8 against FES Dem_BB009.
5. LDES and pumped hydro behind B2, B4 and B6 against the FES/TEC project list.
6. Interconnectors: the CP30 total and each link's landing bus and capacity.
7. Demand: shares behind B6, B7a, B8, LE1 and SC1 against FES 2024, the total
   against ED1, the peak against ED1's peak.
8. Distribution-size solar, onshore and batteries away from their own GSP node (the build's
   locator and scope: named sites and batteries; project size; island networks).
9. Radial pockets whose generation and storage exceed their connection.
10. Boundary exposure summary (INFO).
11. Demand-side headroom: each node's peak load plus electrolysis against what it can import at
    the BM's ratings (16 Oct 2026; the BUMU3-/ABNE3-/BAIN3- load shedding).

Every check writes rows (check, item, model, reference, difference, tolerance,
units, status, source, note). FAIL fails the rule (``fail_on_fail``), WARN
reports, INFO documents, PASS confirms. Settings live in
``validation.locational`` in config/defaults/method/brownfield.yaml;
``DEFAULT_CONFIG`` here must equal it.

References. Tables derived mechanically from repository sources are built at run
time from the files the rule declares (:func:`derive_references`), so they cannot
drift from their sources: the FES node tables (demand, electrolysis, hydro, marine)
from the FES data of ``fes_reference``; the offshore landings from the REPD landing
crosswalk and the checked CfD contracts; the transmission-connected named loads from
the CP30 demand build's own inputs, energy split and loss factor
(``cp30_demand.named_load_reference``); the FES24 ETYS winter-peak shares (INFO) from
rule ``extract_cp30_demand_inputs``; the interconnector capacities and which cables
CP30 counts from the build's cable composition (interconnectors.cp30_composition).
Only hand judgements are committed, in data/CP30/validation_references/, each row
with its source: the Direct-row locations of FES, offshore landings outside the
crosswalk, named stations, LDES projects, interconnector landings and the DESNZ
demand shares (an external dataset).
"""

# No `from __future__` import: Snakemake's script directive prepends a preamble.

import argparse
import copy
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

STATUSES = ("PASS", "INFO", "WARN", "FAIL")
COLUMNS = ["check", "item", "model", "reference", "difference", "tolerance", "units", "status", "source", "note"]

#: Must equal ``validation.locational`` in config/defaults/method/brownfield.yaml (test_cp30_locational_validation checks it).
DEFAULT_CONFIG: dict = {
    "enabled": True,
    "fail_on_fail": True,
    "reference_dir": "data/CP30/validation_references",
    "boundary_file": "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson",
    # Sources of the reference tables derived at run time (derive_references): FES GSP data
    # (resources/FES/FES_{FES_year}_data.csv, a declared rule input) by building block, located
    # at the FES node's network bus, else its GSP in gsp_info_file; the Direct rows at the points
    # of fes_direct_row_locations.csv (reference_dir).
    "fes_reference": {
        "FES_year": 2024,
        "FES_scenario": "Holistic Transition",
        "gsp_info_file": "data/network/ETYS/fes2024_regional_breakdown_gsp_info.csv",
        "blocks": {"demand": "Dem_BB003", "electrolysis": "Dem_BB009", "hydro": "Gen_BB018",
                   "marine": "Gen_BB017"},
    },
    # A point on the remote side of each boundary (OSGB36 m): north of the B
    # boundaries (Thurso), inside EC5 (Leiston), LE1 (central London), SC1
    # (Brighton) and B13 (Cornwall).
    "boundary_remote_points": {
        "default": [311738.0, 967891.0],
        "EC5": [644682.0, 262905.0],
        "LE1": [530563.0, 180683.0],
        "SC1": [531087.0, 105033.0],
        "B13": [190000.0, 55000.0],
    },
    "es5_regional": {
        "abs_tol_mw": 20.0,
        "rel_tol": 0.01,
        # Built (operational) capacity the build keeps above ES5 in a region and carries
        # nationally (brownfield_fleet.reconcile_with_es5); other regions may fall short by
        # the same total. Upper bounds in MW.
        "declared_overshoots": [
            {"technology": "wind_onshore", "region": "Midlands", "max_mw": 150.0,
             "reason": "operational onshore above ES5 (brownfield operational_excess, about 130 MW)"},
            {"technology": "wind_offshore", "region": "Midlands", "max_mw": 20.0,
             "reason": "operational offshore above ES5 (Lincolnshire farms, about 16 MW)"},
        ],
        "fes_abs_tol_mw": 50.0,
        "fes_rel_tol": 0.25,
        "tag_abs_tol_mw": 20.0,
    },
    "offshore": {
        "proxy_boundaries": ["B1a", "B2", "B4", "B6", "B7a", "B8", "EC5"],
        "crossing_tol_mw": 5.0,
        # A FES source point at sea further than this from the coast has no judged side.
        "coast_km": 30.0,
        # [{boundary, landing_bus, reason}]: a proxy crossing accepted by decision.
        "crossing_exceptions": [
            {"boundary": "B7a", "landing_bus": "CREB41",
             "reason": "Teesside/Blyth FES proxies (LACK21, TODP21, BLYT21) land at Creyke Beck, the Dogger Bank "
                       "South / Birkhill Wood point; about 250 MW, accepted on review"},
        ],
        "region_tol_mw": 5.0,
        # [{tag_region, bus_region, landing_bus, reason}]
        "region_exceptions": [],
        "refill_name_pattern": "_refill_",
    },
    "named_stations": {
        # station -> extra buses accepted by decision (reported INFO).
        "exceptions": {"Enfield": ["TOTT2A"]},
        "large_battery_mw": 100.0,
    },
    "electrolysis": {
        "carriers": ["electrolysis"],
        "boundaries": ["B4", "B6", "B8"],
        "abs_tol_mw": 200.0,
        "rel_tol": 0.10,
    },
    "ldes": {
        "battery_carriers": ["Battery"],
        "boundaries": ["B2", "B4", "B6"],
        "abs_tol_mw": 250.0,
        "rel_tol": 0.15,
    },
    "interconnectors": {
        "abs_tol_mw": 20.0,
        "rel_tol": 0.01,
        "capacity_warn_rel": 0.05,
    },
    "demand": {
        "boundaries": ["B6", "B7a", "B8", "LE1", "SC1"],
        "share_warn_pp": 2.0,
        "share_fail_pp": 4.0,
        "total_fail_rel": 0.001,
        "peak_warn_rel": 0.03,
        "peak_fail_rel": 0.08,
        # Compose the transmission-connected named loads (their share of system demand at their
        # nodes) with the FES GSP shares, as the CP30 demand build does. The named loads are those
        # of the scenario's demand.build (cp30_demand.named_load_reference, from the outputs of rule
        # extract_cp30_demand_inputs); none when the scenario does not enable that build.
        "named_loads": True,
        "ed1_total_item": "GB Total System Demand - Minus electrolysis",
        # The peak is judged against CP30's dispatch-model peak, ES1 "Demand (MW) - Excluding
        # Electrolysis" (56,284.67 MW FFR 2030, the basis of its 315.16 TWh; user decision
        # 15 Oct 2026). ED1's GBFES peak (consumption plus losses, an average-cold-spell
        # planning peak, 62,357 MW) is reported as INFO only. Both are read as the CP30 demand
        # build reads them (cp30_demand.peak_target_mw).
    },
    "distributed": {
        "generator_carriers": ["solar_pv", "wind_onshore"],
        "storage_carriers": ["Battery"],
        "warn_share": 0.10,
        "fail_share": 0.25,
        "no_demand_warn_share": 0.10,
    },
    "stranded": {
        "margin": 0.30,
        "max_rows": 40,
        "excluded_generator_carriers": ["load_shedding", "EU_market"],
    },
    "exposure": {
        "boundaries": ["B0", "B1a", "B2", "B4", "B5", "B6", "B7a", "B8", "B9", "EC5", "LE1", "SC1", "B13"],
    },
    # Check 11: each node's peak load plus electrolysis against its import capacity at the BM's
    # ratings (DemandHeadroom). The electrolyser relief holds every node to 1.0 x, so a WARN is a
    # node whose own load is near its connection; above 1.2 x the held electrolysers' Stage 1
    # schedule cannot be met in the month's peak hours without shedding load. A node whose peak
    # load alone exceeds its capacity by more than its firm local generation (carriers below
    # excluded) FAILs whatever the ratio: only load shedding can serve it.
    "demand_headroom": {
        "warn_ratio": 1.0,
        "fail_ratio": 1.2,
        "max_rows": 40,
        "non_firm_generator_carriers": ["load_shedding", "EU_market", "wind_onshore", "wind_offshore", "solar_pv",
                                        "marine", "tidal_stream", "shoreline_wave", "tidal_lagoon"],
    },
}

RENEWABLE_GROUPS = {
    "hydro": ["large_hydro", "small_hydro"],
    "marine": ["marine", "tidal_stream", "shoreline_wave", "tidal_lagoon"],
}
ES5_GENERATION = {"Onshore Wind": "wind_onshore", "Offshore Wind": "wind_offshore", "Solar": "solar_pv"}
SCOTTISH_REGIONS = {"North Scotland", "South Scotland"}
#: network.transmission.min_bm_constraint_voltage_kv of config/defaults/base/network.yaml, for a run given no scenario.
DEFAULT_BM_FLOOR_KV = 220.0


def resolve_config(config: Optional[dict]) -> dict:
    """Deep-merge a configured block over ``DEFAULT_CONFIG`` (dicts merge, everything else replaces)."""
    def merge(base, over):
        out = copy.deepcopy(base)
        for k, v in (over or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
    return merge(DEFAULT_CONFIG, config or {})


# ── Network tables ─────────────────────────────────────────────────────────────

@dataclass
class NetworkTables:
    """The static tables and load time series the checks read.

    ``load_p`` is snapshots x loads (MW); ``weights`` the snapshot weightings (h);
    ``snapshots`` the timestamps of ``load_p``'s rows where known (check 11 takes
    each month's peak at that month's seasonal line ratings), else None.
    """
    buses: pd.DataFrame
    generators: pd.DataFrame
    storage_units: pd.DataFrame
    links: pd.DataFrame
    loads: pd.DataFrame
    lines: pd.DataFrame
    transformers: pd.DataFrame
    load_p: pd.DataFrame
    weights: pd.Series
    snapshots: Optional[pd.Index] = None

    def __post_init__(self):
        # PyPSA and its netCDF files store a missing string attribute as "": read it as missing.
        for name in ("buses", "generators", "storage_units", "links", "loads", "lines", "transformers"):
            df = getattr(self, name)
            for col in df.columns[df.dtypes.eq(object)]:
                s = df[col]
                df[col] = s.where(~s.astype(str).str.strip().isin(["", "nan", "None"]), np.nan)

    @classmethod
    def from_network(cls, network) -> "NetworkTables":
        loads = network.loads.copy()
        p = network.loads_t.p_set.reindex(columns=loads.index)
        static = loads.get("p_set", pd.Series(0.0, index=loads.index)).fillna(0.0)
        for col in p.columns[p.isna().all()]:
            p[col] = static.get(col, 0.0)
        weights = network.snapshot_weightings["generators"] if "generators" in network.snapshot_weightings else \
            pd.Series(1.0, index=network.snapshots)
        storage, p = _as_built_demand(loads, p.fillna(0.0), network.storage_units.copy(),
                                      getattr(network, "meta", None), pd.Index(network.snapshots))
        return cls(network.buses.copy(), network.generators.copy(), storage,
                   network.links.copy(), loads, network.lines.copy(), network.transformers.copy(),
                   p, weights.astype(float), pd.Index(network.snapshots))

    @classmethod
    def from_netcdf(cls, path) -> "NetworkTables":
        """Read a saved PyPSA network with xarray full-variable reads (no label .sel)."""
        import xarray as xr

        ds = xr.open_dataset(path)
        try:
            def table(comp, dim):
                if dim not in ds.dims:
                    return pd.DataFrame()
                data = {v[len(comp) + 1:]: ds[v].values for v in ds.data_vars
                        if v.startswith(comp + "_") and "_t_" not in v and ds[v].dims == (dim,)}
                return pd.DataFrame(data, index=pd.Index(ds[dim].values.astype(str), name=comp[:-1]))

            buses = table("buses", "buses_i")
            generators = table("generators", "generators_i")
            storage = table("storage_units", "storage_units_i")
            links = table("links", "links_i")
            loads = table("loads", "loads_i")
            lines = table("lines", "lines_i")
            transformers = table("transformers", "transformers_i")
            n_snap = ds.sizes.get("snapshots", 1)
            if "loads_t_p_set" in ds:
                arr = ds["loads_t_p_set"]
                cols = ds[arr.dims[1]].values.astype(str)
                load_p = pd.DataFrame(arr.values, columns=cols)
            else:
                load_p = pd.DataFrame(index=range(n_snap))
            static = loads.get("p_set", pd.Series(0.0, index=loads.index))
            missing = [c for c in loads.index if c not in load_p.columns]
            for c in missing:
                load_p[c] = float(static.get(c, 0.0) or 0.0)
            load_p = load_p.reindex(columns=loads.index).fillna(0.0)
            if "snapshots_generators" in ds:
                weights = pd.Series(ds["snapshots_generators"].values.astype(float))
            else:
                weights = pd.Series(np.ones(len(load_p)))
            # PyPSA's NetCDF indexes snapshots 0..n and keeps the times in snapshots_snapshot.
            snapshots = None
            if "snapshots_snapshot" in ds and len(ds["snapshots_snapshot"]) == len(load_p):
                snapshots = pd.DatetimeIndex(ds["snapshots_snapshot"].values, name="snapshot")
            meta = json.loads(ds.attrs["meta"]) if "meta" in ds.attrs else None
        finally:
            ds.close()
        storage, load_p = _as_built_demand(loads, load_p, storage, meta, snapshots)
        return cls(buses, generators, storage, links, loads, lines, transformers, load_p, weights, snapshots)


def _as_built_demand(loads: pd.DataFrame, load_p: pd.DataFrame, storage: pd.DataFrame, meta, snapshots):
    """Storage plant without consumer flexibility, and the loads with its carved reference added back.

    Consumer flexibility (F5; ``scripts/demand/cp30_flexibility.py``) is demand: the storage checks
    (ES5 split, LDES, named stations, distributed plant, stranded and exposed capacity) leave it
    out, and the demand checks and the electrolyser import relief judge the inflexible demand the
    CP30 build calibrated (fixed loads plus the reference each unit's energy was carved from).
    A no-op on a network without it.
    """
    from scripts.demand import cp30_flexibility

    if storage is None or storage.empty or "carrier" not in storage.columns:
        return storage, load_p
    flex = cp30_flexibility.demand_storage_mask(storage)
    if not flex.any():
        return storage, load_p
    load_p = cp30_flexibility.restore_reference_loads(loads, load_p, storage[flex], meta, snapshots)
    return storage[~flex], load_p


# ── Geography helpers ──────────────────────────────────────────────────────────

def lonlat_to_osgb(lon, lat):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return t.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def assign_bus_regions(buses: pd.DataFrame, polygons, group_region: pd.Series,
                       overrides: Optional[Dict[str, str]] = None) -> pd.Series:
    """CP30 ES5 region of every GB bus: the GSP polygon it falls in, else the nearest one.

    The build's own rule (``spatial_allocation.regions_of_xy``); ``overrides`` maps a
    bus-name prefix to a region and wins over the polygon (by default
    ``spatial_allocation.TRANSMISSION_SITE_REGIONS``, as the storage split uses).
    """
    from scripts.CP30.spatial_allocation import bus_cp30_regions

    gb = _gb_mask(buses)
    found = bus_cp30_regions(buses.loc[gb], polygons, group_region, overrides).reindex(buses.index)
    region = found["cp30_region"].copy()
    region.attrs["gsp_group"] = found["gsp_group"]
    region.attrs["onshore"] = pd.to_numeric(found["distance_m"], errors="coerce").fillna(np.inf) <= 0.0
    return region


def coastal_points(xy: pd.DataFrame, polygons, return_distance: bool = False):
    """Each point, or for a point outside every polygon the nearest point on the nearest polygon.

    With ``return_distance`` also returns the distance moved (m; 0 for a point on land).
    """
    import geopandas as gpd
    from shapely.ops import nearest_points

    poly = polygons[["geometry"]].to_crs("EPSG:27700").reset_index(drop=True)
    pts = gpd.GeoDataFrame({"key": range(len(xy))}, geometry=gpd.points_from_xy(xy["x"], xy["y"]), crs="EPSG:27700")
    j = gpd.sjoin_nearest(pts, poly, how="left", distance_col="dist").drop_duplicates("key").sort_values("key")
    out = xy[["x", "y"]].astype(float).copy()
    moved = pd.Series(0.0, index=xy.index)
    for k, (idx, dist) in enumerate(zip(j["index_right"], j["dist"])):
        if dist > 0 and pd.notna(idx):
            p = nearest_points(poly.geometry.iloc[int(idx)], pts.geometry.iloc[k])[0]
            out.iloc[k] = (p.x, p.y)
            moved.iloc[k] = float(dist)
    return (out, moved) if return_distance else out


def region_sides(ctx, boundary: str) -> Dict[str, bool]:
    """Regions lying wholly on one side of a boundary: region -> remote (True/False). Straddling regions omitted."""
    side = ctx.bus_sides[boundary]
    onshore = ctx.bus_region.attrs.get("onshore")
    if onshore is not None:  # offshore platform buses say nothing about a region's coast
        side = side[onshore.reindex(side.index).fillna(False).astype(bool)]
    reg = ctx.bus_region.reindex(side.index)
    out = {}
    for r, s in side.groupby(reg):
        if s.all():
            out[r] = True
        elif not s.any():
            out[r] = False
    return out


def _gb_mask(buses: pd.DataFrame) -> pd.Series:
    country = buses.get("country", pd.Series("GB", index=buses.index)).fillna("GB").astype(str)
    carrier = buses.get("carrier", pd.Series("AC", index=buses.index)).fillna("AC").astype(str)
    external = buses.index.to_series().astype(str).str.contains("External", case=False)
    return country.eq("GB") & carrier.eq("AC") & ~external


def oriented_sides(xy: pd.DataFrame, geoms: dict, boundaries: Iterable[str],
                   remote_points: dict) -> pd.DataFrame:
    """True where a point is on the remote side of each boundary (the side holding its remote point)."""
    from scripts.network_clustering.boundary_busmap import boundary_sides

    boundaries = [b for b in boundaries if b in geoms]
    pts = xy[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    ok = pts.notna().all(axis=1)
    refs = pd.DataFrame([remote_points.get(b, remote_points["default"]) for b in boundaries],
                        columns=["x", "y"], index=[f"__remote__{b}" for b in boundaries])
    frame = pd.concat([pts[ok], refs])
    raw = boundary_sides(frame, geoms, boundaries)
    out = pd.DataFrame(False, index=xy.index, columns=boundaries)
    for b in boundaries:
        out.loc[ok[ok].index, b] = (raw.loc[ok[ok].index, b] == raw.at[f"__remote__{b}", b]).to_numpy()
    out.attrs["located"] = ok
    return out


# ── Context and row helpers ────────────────────────────────────────────────────

@dataclass
class Context:
    tables: NetworkTables
    cfg: dict
    refs: Dict[str, pd.DataFrame]
    geoms: dict
    polygons: object
    group_region: pd.Series
    es5: pd.DataFrame
    ed1: Optional[pd.DataFrame]
    es1: Optional[pd.DataFrame]
    pathway: str
    year: int
    # The scenario's transmission block (BM voltage floor: min_bm_constraint_voltage_kv, bm_voltage_floor).
    transmission: dict = field(default_factory=dict)
    # Bus-name prefix -> CP30 region overrides (None: spatial_allocation.TRANSMISSION_SITE_REGIONS).
    site_regions: Optional[Dict[str, str]] = None
    # GB_network.xlsx (Dem_per_node), the table the build's GSP-node locator reads (check 8); the
    # Dem_per_node table itself may be given instead of the file.
    dem_per_node_file: object = None
    bus_region: pd.Series = field(default=None)
    bus_sides: pd.DataFrame = field(default=None)

    def __post_init__(self):
        b = self.tables.buses
        self.bus_region = assign_bus_regions(b, self.polygons, self.group_region, self.site_regions)
        gb = _gb_mask(b)
        self.bus_sides = oriented_sides(b[gb], self.geoms, self.all_boundaries(), self.remote_points)

    @property
    def remote_points(self):
        return self.cfg["boundary_remote_points"]

    def all_boundaries(self) -> List[str]:
        c = self.cfg
        names = (c["offshore"]["proxy_boundaries"] + c["electrolysis"]["boundaries"] + c["ldes"]["boundaries"]
                 + c["demand"]["boundaries"] + c["exposure"]["boundaries"])
        return [b for b in dict.fromkeys(names) if b in self.geoms]

    def sides_of_points(self, xy: pd.DataFrame, boundaries=None) -> pd.DataFrame:
        return oriented_sides(xy, self.geoms, boundaries or self.all_boundaries(), self.remote_points)

    def bus_remote(self, buses: pd.Series, boundary: str) -> pd.Series:
        """Remote-side flag for a Series of bus names (False for unknown buses)."""
        side = self.bus_sides[boundary] if boundary in self.bus_sides else pd.Series(dtype=bool)
        return buses.map(side).fillna(False).astype(bool)


def _row(check, item, model=np.nan, reference=np.nan, tolerance=np.nan, status="INFO", source="",
         note="", units="MW", difference=None):
    if difference is None:
        try:
            difference = float(model) - float(reference)
        except (TypeError, ValueError):
            difference = np.nan
    return {"check": check, "item": item, "model": model, "reference": reference, "difference": difference,
            "tolerance": tolerance, "units": units, "status": status, "source": source, "note": note}


def _tol(ref, rel, abs_):
    return max(abs(float(ref)) * float(rel), float(abs_))


def _band(value, lo, hi):
    """Distance of value outside [lo, hi] (0 inside)."""
    if value < lo:
        return value - lo
    if value > hi:
        return value - hi
    return 0.0


def _split(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [p.strip() for p in str(value).split("|") if p.strip()]


def _carrier(df):
    return df.get("carrier", pd.Series("", index=df.index)).astype(str)


def _numeric(df, col, default=0.0):
    return pd.to_numeric(df.get(col, pd.Series(default, index=df.index)), errors="coerce").fillna(default)


# ── Check 1: ES5 regional totals ───────────────────────────────────────────────

def es5_targets(es5: pd.DataFrame, pathway: str) -> pd.DataFrame:
    """ES5 capacity (MW) by technology (rows) and region (columns), with non-battery storage derived.

    Read with the build's own readers (``spatial_allocation.read_es5_targets`` for generation,
    ``storage_regions.es5_storage_targets`` for storage), so the check and the build cannot
    disagree on what ES5 says.
    """
    from scripts.CP30.spatial_allocation import read_es5_targets
    from scripts.CP30.storage_regions import es5_storage_targets

    generation = read_es5_targets(es5, pathway, 0)
    generation = generation[generation["carrier"].isin(ES5_GENERATION.values())]
    storage = es5_storage_targets(es5, pathway).replace(
        {"carrier": {"Battery": "battery", "LAES": "non_battery_storage"}})
    long = pd.concat([generation[["carrier", "cp30_region", "capacity_mw"]], storage], ignore_index=True)
    return long.pivot_table(index="carrier", columns="cp30_region", values="capacity_mw", aggfunc="sum")


def check_es5_regional(ctx: Context) -> List[dict]:
    c = ctx.cfg["es5_regional"]
    rows = []
    t = ctx.tables
    g, s = t.generators, t.storage_units
    targets = es5_targets(ctx.es5, ctx.pathway)
    battery = set(ctx.cfg["ldes"]["battery_carriers"])
    model = {
        "wind_onshore": g[_carrier(g).eq("wind_onshore")],
        "wind_offshore": g[_carrier(g).eq("wind_offshore")],
        "solar_pv": g[_carrier(g).eq("solar_pv")],
        "battery": s[_carrier(s).isin(battery)],
        "non_battery_storage": s[~_carrier(s).isin(battery)],
    }
    declared = {}
    for d in c.get("declared_overshoots") or []:
        declared[(d["technology"], d["region"])] = (float(d["max_mw"]), d.get("reason", ""))
    src = (f"CP30 ES5 ({ctx.pathway}, {ctx.year}); region = GSP polygon of the bus "
           "(spatial_allocation.TRANSMISSION_SITE_REGIONS applied)")
    for tech, df in model.items():
        if tech not in targets.index:
            continue
        by_region = _numeric(df, "p_nom").groupby(df["bus"].map(ctx.bus_region)).sum()
        tech_declared = {r: v for (tt, r), v in declared.items() if tt == tech}
        pool = sum(v[0] for v in tech_declared.values())
        diffs = {r: float(by_region.get(r, 0.0)) - float(targets.at[tech, r]) for r in targets.columns}
        tols = {r: _tol(targets.at[tech, r], c["rel_tol"], c["abs_tol_mw"]) for r in targets.columns}
        shortfall = sum(max(-diffs[r] - tols[r], 0.0) for r in targets.columns if r not in tech_declared)
        for r in targets.columns:
            ref, diff, tol = float(targets.at[tech, r]), diffs[r], tols[r]
            note = ""
            if r in tech_declared:
                cap, reason = tech_declared[r]
                ok = -tol <= diff <= cap + tol
                note = f"declared overshoot up to {cap:,.0f} MW: {reason}"
            elif diff < -tol:
                ok = shortfall <= pool + 1e-6
                if ok:
                    note = f"shortfall within the {pool:,.0f} MW of declared overshoot carried nationally"
            else:
                ok = diff <= tol
            rows.append(_row("es5_regional", f"{tech} | {r}", float(by_region.get(r, 0.0)), ref, tol,
                             "PASS" if ok else "FAIL", src, note))
        unmapped = float(_numeric(df, "p_nom")[df["bus"].map(ctx.bus_region).isna()].sum())
        if unmapped > 0.5:
            rows.append(_row("es5_regional", f"{tech} | no region", unmapped, 0.0, c["abs_tol_mw"],
                             "FAIL" if unmapped > c["abs_tol_mw"] else "WARN", src,
                             "capacity on buses with no CP30 region (external or unlocated)"))
        # Stored tag against the bus region (offshore is judged in check 2).
        if tech in ("wind_onshore", "solar_pv") and "cp30_region" in df:
            tagged = df[df["cp30_region"].notna()]
            mism = tagged[tagged["cp30_region"].astype(str) != tagged["bus"].map(ctx.bus_region).astype(str)]
            mw = float(_numeric(mism, "p_nom").sum())
            rows.append(_row("es5_regional", f"{tech} | stored cp30_region differs from bus region", mw, 0.0,
                             c["tag_abs_tol_mw"], "WARN" if mw > c["tag_abs_tol_mw"] else "PASS", src,
                             _top_pairs(mism, "cp30_region", ctx)))
    rows.extend(_fes_regional(ctx, g))
    return rows


def _top_pairs(df, tag_col, ctx, n=4):
    if df.empty:
        return ""
    pairs = _numeric(df, "p_nom").groupby([df[tag_col].astype(str), df["bus"].map(ctx.bus_region).astype(str)]).sum()
    pairs = pairs.sort_values(ascending=False).head(n)
    return "; ".join(f"{a} -> {b}: {v:,.0f} MW" for (a, b), v in pairs.items())


def _fes_regional(ctx, g) -> List[dict]:
    """Hydro and marine (not in ES5) against FES 2024 HT 2030 by region: WARN only."""
    c = ctx.cfg["es5_regional"]
    ref = ctx.refs.get("fes_regional_hydro_marine")
    rows = []
    if ref is None or ref.empty:
        return rows
    regions = list(ctx.group_region.unique())
    for tech, carriers in RENEWABLE_GROUPS.items():
        r = ref[ref["technology"] == tech]
        if r.empty:
            continue
        df = g[_carrier(g).isin(carriers)]
        by_region = _numeric(df, "p_nom").groupby(df["bus"].map(ctx.bus_region)).sum()
        unlocated = float(r.loc[~r["region"].isin(regions), "mw"].sum())
        src = str(r["source"].iloc[0])
        for reg in sorted(regions):
            fes = float(r.loc[r["region"] == reg, "mw"].sum())
            hi = fes + (unlocated if reg not in SCOTTISH_REGIONS else 0.0)
            m = float(by_region.get(reg, 0.0))
            tol = _tol(fes, c["fes_rel_tol"], c["fes_abs_tol_mw"])
            out = _band(m, fes, hi)
            note = f"FES Direct(NGET) {unlocated:,.0f} MW unlocated, allowed in any E&W region" if hi > fes else ""
            rows.append(_row("fes_regional", f"{tech} | {reg}", m, fes, tol,
                             "WARN" if abs(out) > tol else "PASS", src, note, difference=out))
    return rows


# ── Check 2: offshore landing ──────────────────────────────────────────────────

def check_offshore(ctx: Context) -> List[dict]:
    c = ctx.cfg["offshore"]
    g = ctx.tables.generators
    off = g[_carrier(g).eq("wind_offshore")].copy()
    rows = []
    if off.empty:
        return [_row("offshore", "offshore generators", 0.0, status="INFO", note="no offshore wind in the network")]
    ref = ctx.refs.get("offshore_landing_reference", pd.DataFrame())
    origin = off.get("fleet_origin", pd.Series("", index=off.index)).astype(str)
    is_refill = off.index.to_series().astype(str).str.contains(c["refill_name_pattern"], regex=False)
    repd = pd.to_numeric(off.get("repd_ref_id", pd.Series(np.nan, index=off.index)), errors="coerce")
    named = off[(origin.eq("named") | (repd.notna() & ~origin.eq("increment"))) & ~is_refill]
    proxies = off[~off.index.isin(named.index) & ~is_refill]
    refill = off[is_refill]
    src_ref = ("offshore landing reference: the build's REPD landing crosswalk + offshore_landing_additions.csv; "
               "exclusions from the checked CfD contracts")

    # 2a. Named farms at their landing.
    ref_by_id = ref.set_index(pd.to_numeric(ref["repd_ref_id"], errors="coerce")) if len(ref) else pd.DataFrame()
    named_repd = pd.to_numeric(named.get("repd_ref_id"), errors="coerce")
    ok_mw, ok_n = 0.0, 0
    no_ref = []
    for rid, grp in named.groupby(named_repd):
        mw = float(_numeric(grp, "p_nom").sum())
        buses = sorted(set(grp["bus"].astype(str)))
        site = str(grp.get("source_site_name", pd.Series([rid])).iloc[0])
        if rid not in ref_by_id.index:
            no_ref.append((site, int(rid), mw, "|".join(buses)))
            continue
        r = ref_by_id.loc[rid]
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        accepted = set(_split(r["landing_bus"]))
        if set(buses) <= accepted:
            ok_mw += mw
            ok_n += 1
            continue
        sourced = str(r.get("sourced", True)).lower() not in ("false", "0")
        rows.append(_row("offshore_named", f"{r['site_name']} (REPD {int(rid)})", mw, np.nan, np.nan,
                         "FAIL" if sourced else "WARN", str(r["source"]),
                         f"model bus {'|'.join(buses)}; reference {r['landing_bus']}"
                         + ("" if sourced else " (reference unsourced: GSP-proxy nearest bus)"), units="MW"))
    rows.append(_row("offshore_named", "named farms at their reference landing", ok_mw, np.nan, np.nan, "PASS",
                     src_ref, f"{ok_n} farms"))
    if no_ref:
        no_ref.sort(key=lambda t: -t[2])
        rows.append(_row("offshore_named", "named farms with no landing reference", sum(t[2] for t in no_ref),
                         np.nan, np.nan, "WARN", src_ref,
                         f"{len(no_ref)} farms with no evidenced landing (not in the crosswalk or the additions): "
                         + "; ".join(f"{n} (REPD {i}) {mw:,.0f} MW at {b}" for n, i, mw, b in no_ref)))
    for rid, r in ref_by_id.iterrows() if len(ref_by_id) else []:
        if str(r.get("excluded_not_built", False)).lower() == "true" and rid in set(named_repd.dropna()):
            rows.append(_row("offshore_named", f"{r['site_name']} (REPD {int(rid)})", np.nan, 0.0, np.nan, "FAIL",
                             "data/market/lccc/checked_cfd_contracts.csv",
                             "excluded as not built by 2030 (not_built_refill) but present in the named fleet"))

    # 2b. Refill rows at an excluded project's landing.
    excluded = ref[ref.get("excluded_not_built", pd.Series(False, index=ref.index)).astype(str).str.lower() == "true"] \
        if len(ref) else pd.DataFrame()
    landings = set()
    for v in excluded.get("landing_bus", []):
        landings |= set(_split(v))
    if refill.empty:
        rows.append(_row("offshore_refill", "refill-at-landing rows", 0.0, status="INFO", source=src_ref,
                         note="no refill rows (excluded MW refilled pro rata through the FES proxies)"))
    else:
        for bus, grp in refill.groupby("bus"):
            mw = float(_numeric(grp, "p_nom").sum())
            ok = str(bus) in landings
            rows.append(_row("offshore_refill", f"refill at {bus}", mw, np.nan, np.nan, "PASS" if ok else "FAIL",
                             src_ref, "" if ok else f"not the landing of an excluded project ({'|'.join(sorted(landings))})"))

    # 2c. FES proxies landing across a boundary from their source.
    rows.extend(_proxy_crossings(ctx, proxies, c))

    # 2d. Offshore region tag = landing bus region.
    if "cp30_region" in off:
        tag = off["cp30_region"].astype(str)
        busreg = off["bus"].map(ctx.bus_region).astype(str)
        mism = off[tag.ne(busreg) & off["cp30_region"].notna()]
        exc = c.get("region_exceptions") or []
        if mism.empty:
            rows.append(_row("offshore_region", "offshore region tag equals landing bus region", 0.0, 0.0,
                             c["region_tol_mw"], "PASS", "GSP polygon of the landing bus"))
        for (tg, br, bus), grp in mism.groupby([tag[mism.index], busreg[mism.index], mism["bus"]]):
            mw = float(_numeric(grp, "p_nom").sum())
            excepted = any(e.get("landing_bus") in (None, bus) and e.get("tag_region") in (None, tg)
                           and e.get("bus_region") in (None, br) for e in exc)
            status = "INFO" if excepted else ("FAIL" if mw > c["region_tol_mw"] else "PASS")
            rows.append(_row("offshore_region", f"tag {tg} | landing {bus} in {br}", mw, 0.0, c["region_tol_mw"],
                             status, "GSP polygon of the landing bus",
                             "declared exception" if excepted else "ES5 counts the capacity in one region, the "
                             "network connects it in another"))
    return rows


def _proxy_crossings(ctx, proxies, c) -> List[dict]:
    rows = []
    bnds = [b for b in c["proxy_boundaries"] if b in ctx.geoms]
    p = proxies.copy()
    lat = pd.to_numeric(p.get("lat"), errors="coerce")
    lon = pd.to_numeric(p.get("lon"), errors="coerce")
    p = p[lat.notna() & lon.notna()]
    src = ("FES proxy source = the row's lat/lon (retained FES proxy geography); landing = its bus; "
           "data/network/ETYS/cp30_offshore_landing_point_profile_preserving.csv")
    if p.empty:
        return [_row("offshore_proxy", "proxy rows with source coordinates", 0.0, status="INFO", source=src)]
    x, y = lonlat_to_osgb(lon[p.index], lat[p.index])
    # The source side is the side of the source region's coast: the region tag's side when
    # that region lies wholly on one side of the boundary; otherwise the side of the FES
    # source point (a point at sea is moved to the nearest GSP polygon, its coast, if
    # within coast_km; a platform further out has no meaningful side and is not judged).
    src_xy, dist = coastal_points(pd.DataFrame({"x": x, "y": y}, index=p.index), ctx.polygons, return_distance=True)
    point_sides = ctx.sides_of_points(src_xy, bnds)
    near = dist.to_numpy() <= float(c["coast_km"]) * 1e3
    tag = p.get("cp30_region", pd.Series(np.nan, index=p.index))
    exc = c.get("crossing_exceptions") or []
    for b in bnds:
        land = ctx.bus_remote(p["bus"].astype(str), b)
        whole = region_sides(ctx, b)
        by_tag = tag.map(whole)
        src_side = np.where(by_tag.notna(), by_tag.fillna(False).astype(bool), point_sides[b].to_numpy())
        judged = by_tag.notna().to_numpy() | near
        unjudged = float(_numeric(p, "p_nom")[~judged].sum())
        if unjudged > 0.5:
            rows.append(_row("offshore_proxy", f"{b}: proxies with no judged source side", unjudged, status="INFO",
                             source=src, note=f"source region straddles {b} and the FES source point is more than "
                             f"{c['coast_km']:.0f} km offshore"))
        cross = p[(src_side != land.to_numpy()) & judged]
        if cross.empty:
            rows.append(_row("offshore_proxy", f"{b}: proxies landing across the boundary", 0.0, 0.0,
                             c["crossing_tol_mw"], "PASS", src))
            continue
        direction = np.where(ctx.bus_remote(cross["bus"].astype(str), b), "into remote side", "out of remote side")
        for (bus, d), grp in cross.groupby([cross["bus"].astype(str), direction]):
            mw = float(_numeric(grp, "p_nom").sum())
            excepted = [e for e in exc if e.get("boundary") == b and e.get("landing_bus") == bus]
            status = "INFO" if excepted else ("FAIL" if mw > c["crossing_tol_mw"] else "PASS")
            note = f"{d}; {len(grp)} rows"
            if excepted:
                note += f"; declared exception: {excepted[0].get('reason', '')}"
            rows.append(_row("offshore_proxy", f"{b}: proxy landing at {bus}", mw, 0.0, c["crossing_tol_mw"],
                             status, src, note))
    return rows


# ── Check 3: named stations ────────────────────────────────────────────────────

def check_named_stations(ctx: Context) -> List[dict]:
    c = ctx.cfg["named_stations"]
    ref = ctx.refs.get("named_station_buses", pd.DataFrame())
    rows = []
    tables = {"generator": ctx.tables.generators, "storage_unit": ctx.tables.storage_units}
    src_file = "data/CP30/validation_references/named_station_buses.csv"
    for _, r in ref.iterrows():
        df = tables.get(str(r["component"]), pd.DataFrame())
        if df.empty:
            continue
        pat = re.compile(str(r["match"]), re.IGNORECASE)
        names = df.index.to_series().astype(str)
        site = df.get("source_site_name", pd.Series("", index=df.index)).fillna("").astype(str)
        hit = names.map(lambda s: bool(pat.fullmatch(s.strip()))) | site.map(lambda s: bool(pat.fullmatch(s.strip())))
        if pd.notna(r.get("carrier")) and str(r.get("carrier")):
            hit &= _carrier(df).eq(str(r["carrier"]))
        m = df[hit]
        expected = set(_split(r["expected_buses"]))
        required = str(r.get("required", "false")).lower() == "true"
        if m.empty:
            rows.append(_row("named_station", str(r["station"]), 0.0, np.nan, np.nan, "FAIL" if required else "INFO",
                             str(r["source"]), "not in the network" + (" (required)" if required else "")))
            continue
        mw = float(_numeric(m, "p_nom").sum())
        buses = set(m["bus"].astype(str))
        accepted_exc = set((c.get("exceptions") or {}).get(str(r["station"]), []))
        if buses <= expected:
            status, note = "PASS", f"at {'|'.join(sorted(buses))}"
        elif buses <= expected | accepted_exc:
            status, note = "INFO", f"declared exception: at {'|'.join(sorted(buses))}, TEC {r['tec_connection_site']}"
        else:
            status = "FAIL"
            note = f"model bus {'|'.join(sorted(buses))}; TEC {r['tec_connection_site']} ({'|'.join(sorted(expected))})"
        rows.append(_row("named_station", str(r["station"]), mw, np.nan, np.nan, status, str(r["source"]), note))
    s = ctx.tables.storage_units
    big = s[_carrier(s).isin(ctx.cfg["ldes"]["battery_carriers"]) & (_numeric(s, "p_nom") >= c["large_battery_mw"])]
    proxy = big.index.to_series().astype(str).str.startswith("CP30_")
    rows.append(_row("named_station", f"batteries >= {c['large_battery_mw']:.0f} MW without a named reference",
                     float(_numeric(big, "p_nom").sum()), np.nan, np.nan, "INFO", src_file,
                     f"{len(big)} units ({int(proxy.sum())} FES proxy rows); no named BESS reference to check against"))
    return rows


# ── Check 4: electrolysers ─────────────────────────────────────────────────────

def _reference_range(ctx, ref: pd.DataFrame, value_col: str, boundaries) -> Dict[str, tuple]:
    """[lo, hi] of a node-level reference on the remote side of each boundary.

    A row whose ``known_boundaries`` is set (a Direct row at a representative point)
    counts only on those boundaries; elsewhere it widens the range.
    """
    xy = ref[["x", "y"]]
    sides = ctx.sides_of_points(xy, boundaries)
    known = ref.get("known_boundaries", pd.Series("", index=ref.index)).fillna("").astype(str)
    out = {}
    for b in boundaries:
        if b not in sides:
            continue
        certain = known.eq("") | known.map(lambda k: b in k.split(";"))
        v = pd.to_numeric(ref[value_col], errors="coerce").fillna(0.0)
        lo = float(v[certain & sides[b]].sum())
        out[b] = (lo, lo + float(v[~certain].sum()))
    return out


def check_electrolysis(ctx: Context) -> List[dict]:
    c = ctx.cfg["electrolysis"]
    ref = ctx.refs.get("fes_electrolysis_by_gsp")
    links = ctx.tables.links
    el = links[_carrier(links).isin(c["carriers"])]
    if ref is None or ref.empty:
        return [_row("electrolysis", "reference", status="INFO", note="no FES electrolysis reference")]
    src = str(ref["source"].iloc[0])
    bnds = [b for b in c["boundaries"] if b in ctx.geoms]
    rng = _reference_range(ctx, ref, "mw", bnds)
    rows = [_row("electrolysis", "total", float(_numeric(el, "p_nom").sum()), float(ref["mw"].sum()), np.nan,
                 "INFO", src)]
    for b in bnds:
        model = float(_numeric(el, "p_nom")[ctx.bus_remote(el["bus0"].astype(str), b)].sum())
        lo, hi = rng[b]
        tol = _tol(max(lo, hi), c["rel_tol"], c["abs_tol_mw"])
        out = _band(model, lo, hi)
        rows.append(_row("electrolysis", f"north of {b}", model, lo if lo == hi else f"{lo:,.0f}-{hi:,.0f}", tol,
                         "FAIL" if abs(out) > tol else "PASS", src,
                         "" if lo == hi else "range: FES Direct rows whose side of this boundary is unknown",
                         difference=out))
    return rows


# ── Check 5: LDES and pumped hydro ─────────────────────────────────────────────

def check_ldes(ctx: Context) -> List[dict]:
    c = ctx.cfg["ldes"]
    ref = ctx.refs.get("ldes_projects_2030")
    s = ctx.tables.storage_units
    ld = s[~_carrier(s).isin(c["battery_carriers"])]
    rows = []
    if ref is None or ref.empty:
        return [_row("ldes", "reference", status="INFO", note="no LDES project reference")]
    bnds = [b for b in c["boundaries"] if b in ctx.geoms]
    proj_sides = ctx.sides_of_points(ref[["x", "y"]], bnds)
    targets = es5_targets(ctx.es5, ctx.pathway)
    es5_nb = targets.loc["non_battery_storage"] if "non_battery_storage" in targets.index else pd.Series(dtype=float)
    proj_by_region = pd.to_numeric(ref["mw"], errors="coerce").groupby(ref["region"]).sum()
    residual = {r: max(float(es5_nb.get(r, 0.0)) - float(proj_by_region.get(r, 0.0)), 0.0) for r in es5_nb.index}
    gb = ctx.bus_sides.index
    region_of = ctx.bus_region.reindex(gb)
    src = ("data/CP30/validation_references/ldes_projects_2030.csv (FES 2024 HT Srg_BB003/004, TEC register) "
           "+ ES5 non-battery storage not named by a project, anywhere in its region")
    for b in bnds:
        lo = float(pd.to_numeric(ref["mw"], errors="coerce")[proj_sides[b]].sum())
        hi = lo
        notes = []
        for r, res in residual.items():
            if res <= 0:
                continue
            share = ctx.bus_sides.loc[region_of[region_of == r].index, b]
            if share.empty or not share.any():
                continue
            if share.all():
                lo += res
                hi += res
            else:
                hi += res
                notes.append(f"{r} residual {res:,.0f} MW straddles {b}")
        model = float(_numeric(ld, "p_nom")[ctx.bus_remote(ld["bus"].astype(str), b)].sum())
        tol = _tol(max(lo, hi), c["rel_tol"], c["abs_tol_mw"])
        out = _band(model, lo, hi)
        rows.append(_row("ldes", f"north of {b}", model, lo if lo == hi else f"{lo:,.0f}-{hi:,.0f}", tol,
                         "FAIL" if abs(out) > tol else "PASS", src, "; ".join(notes), difference=out))
    # Per site (projects sharing buses together): what the network holds there.
    site_key = ref["bus"].map(lambda v: "|".join(sorted(_split(v))) or None)
    for key, grp in ref.groupby(site_key.fillna("__no_bus__" + ref["project"]), sort=False):
        buses = [] if key.startswith("__no_bus__") else key.split("|")
        at = float(_numeric(ld, "p_nom")[ld["bus"].astype(str).isin(buses)].sum()) if buses else np.nan
        rows.append(_row("ldes_site", f"{' + '.join(grp['project'])} ({grp['region'].iloc[0]})", at,
                         float(pd.to_numeric(grp["mw"]).sum()), np.nan, "INFO", "; ".join(grp["source"].astype(str)),
                         f"non-battery storage MW at {key if buses else 'no ETYS bus (site coordinates only)'}"))
    return rows


# ── Check 6: interconnectors ───────────────────────────────────────────────────

def _cp30_value(table: Optional[pd.DataFrame], pathway, year, mask_fn) -> Optional[float]:
    if table is None or table.empty:
        return None
    t = table[table["Pathway"].astype(str).eq(pathway)] if "Pathway" in table else table
    t = t[mask_fn(t)]
    col = str(year)
    if t.empty or col not in t:
        return None
    v = pd.to_numeric(t[col], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else None


def check_interconnectors(ctx: Context) -> List[dict]:
    c = ctx.cfg["interconnectors"]
    # interconnectors_reference: the committed landings joined to CP30's cable composition at run time
    # (derive_references); without the composition, the committed table as it is.
    ref = ctx.refs.get("interconnectors_reference", ctx.refs.get("interconnectors_2030", pd.DataFrame()))
    links = ctx.tables.links
    name_col = links.get("interconnector_name", pd.Series(np.nan, index=links.index))
    ic = links[name_col.notna()].copy()
    rows = []
    gb = _gb_mask(ctx.tables.buses)
    ic["gb_bus"] = np.where(ic["bus0"].map(gb).fillna(False), ic["bus0"], ic["bus1"])
    ic["norm"] = ic["interconnector_name"].astype(str).str.strip().str.lower()
    # A directional import/export pair (interconnector_pair_id) is one cable: count it once.
    pair = ic.get("interconnector_pair_id", pd.Series(np.nan, index=ic.index))
    key = pair.where(pair.notna(), ic.index.to_series()).astype(str)
    per_key = _numeric(ic, "p_nom").groupby([ic["norm"], key]).max()
    cap = per_key.groupby(level=0).sum()
    gb_bus = ic.groupby("norm")["gb_bus"].agg(lambda s: "|".join(sorted(set(map(str, s)))))
    matched = set()
    counted_total = 0.0
    for _, r in ref.iterrows():
        pat = re.compile(str(r["match"]), re.IGNORECASE)
        names = [n for n in cap.index if pat.fullmatch(n)]
        required = str(r.get("required", "true")).lower() == "true"
        counted = str(r.get("counted_in_cp30", "true")).lower() == "true"
        excluded = str(r.get("in_cp30_build", "true")).lower() == "false"
        if excluded:
            # Outside CP30's cable set this year (composition: exclude, or first_year later).
            matched |= set(names)
            rows.append(_row("interconnector", f"{r['interconnector']} landing",
                             float(cap[names].sum()) if names else np.nan, np.nan, np.nan,
                             "WARN" if names else "INFO", str(r["source"]),
                             (f"in the network at {'|'.join(sorted(set('|'.join(gb_bus[names]).split('|'))))} although "
                              if names else "not in the network: ") + str(r.get("composition_note", "outside CP30's count"))))
            continue
        if not names:
            rows.append(_row("interconnector", f"{r['interconnector']} landing", np.nan, float(r["capacity_mw"]),
                             np.nan, "FAIL" if required else "WARN", str(r["source"]),
                             f"absent from the network (expected at {r['landing_bus']})"))
            continue
        matched |= set(names)
        mw = float(cap[names].sum())
        if counted:
            counted_total += mw
        buses = set("|".join(gb_bus[names]).split("|"))
        ok = buses <= set(_split(r["accepted_buses"]))
        rows.append(_row("interconnector", f"{r['interconnector']} landing", mw, np.nan, np.nan,
                         "PASS" if ok else "FAIL", str(r["source"]),
                         f"at {'|'.join(sorted(buses))}" + ("" if ok else f"; expected {r['accepted_buses']}")))
        rel = mw / float(r["capacity_mw"]) - 1.0
        rows.append(_row("interconnector", f"{r['interconnector']} capacity", mw, float(r["capacity_mw"]),
                         c["capacity_warn_rel"], "WARN" if abs(rel) > c["capacity_warn_rel"] else "PASS",
                         str(r["source"]), f"{100 * rel:+.1f}% against the reference capacity"))
    for n in sorted(set(cap.index) - matched):
        counted_total += float(cap[n])
        rows.append(_row("interconnector", f"{n} landing", float(cap[n]), np.nan, np.nan, "WARN",
                         "data/CP30/validation_references/interconnectors_2030.csv",
                         f"no reference row; at {gb_bus[n]}"))
    target = _cp30_value(ctx.es1, ctx.pathway, ctx.year,
                         lambda t: t["Variable"].astype(str).eq("Capacity (MW)")
                         & t["Category"].astype(str).eq("Interconnectors"))
    if target is not None:
        tol = _tol(target, c["rel_tol"], c["abs_tol_mw"])
        rows.append(_row("interconnector", "CP30 total (links counted by CP30)", counted_total, target, tol,
                         "FAIL" if abs(counted_total - target) > tol else "PASS",
                         f"CP30 ES1 interconnector capacity, {ctx.pathway} {ctx.year}",
                         "Isle of Man excluded (not in CP30's count)"))
    return rows


# ── Check 7: demand ────────────────────────────────────────────────────────────

def _electricity_loads(ctx) -> pd.DataFrame:
    loads = ctx.tables.loads
    gb = _gb_mask(ctx.tables.buses)
    car = _carrier(loads)
    keep = loads["bus"].map(gb).fillna(False) & ~car.str.contains("H2|hydrogen", case=False)
    return loads[keep]


def check_demand(ctx: Context) -> List[dict]:
    c = ctx.cfg["demand"]
    rows = []
    loads = _electricity_loads(ctx)
    p = ctx.tables.load_p.reindex(columns=loads.index).fillna(0.0)
    w = ctx.tables.weights.to_numpy()[: len(p)]
    energy = pd.Series((p.to_numpy() * w[:, None]).sum(axis=0), index=loads.index)
    total_gwh = float(energy.sum()) / 1e3
    peak_mw = float(p.sum(axis=1).max()) if len(p) else 0.0
    ed1_total = _cp30_value(ctx.ed1, ctx.pathway, ctx.year,
                            lambda t: t["Data item"].astype(str).str.strip().eq(c["ed1_total_item"]))
    # The two peaks as the CP30 demand build reads them (cp30_demand.peak_target_mw).
    from scripts.demand.cp30_demand import peak_target_mw

    def _peak(kind):
        try:
            return peak_target_mw(kind, ctx.pathway, ctx.year, ctx.es1, ctx.ed1)[0]
        except (ValueError, KeyError, TypeError):
            return None
    es1_peak, ed1_peak = _peak("es1"), _peak("ed1")
    src_ed1 = f"CP30 ED1, {ctx.pathway} {ctx.year}"
    if ed1_total is not None:
        tol = abs(ed1_total) * c["total_fail_rel"]
        rows.append(_row("demand", "annual demand excl. electrolysis", total_gwh, ed1_total, tol,
                         "FAIL" if abs(total_gwh - ed1_total) > tol else "PASS", src_ed1, units="GWh"))
    if es1_peak is not None:
        rel = peak_mw / es1_peak - 1.0
        status = "FAIL" if abs(rel) > c["peak_fail_rel"] else ("WARN" if abs(rel) > c["peak_warn_rel"] else "PASS")
        rows.append(_row("demand", "peak demand excl. electrolysis", peak_mw, es1_peak, es1_peak * c["peak_warn_rel"],
                         status, f"CP30 ES1 Demand (MW) - Excluding Electrolysis, {ctx.pathway} {ctx.year} "
                         "(dispatch-model peak)",
                         f"{100 * rel:+.1f}% (WARN at {100 * c['peak_warn_rel']:.0f}%, FAIL at "
                         f"{100 * c['peak_fail_rel']:.0f}%)"))
    if ed1_peak is not None:
        ref = ed1_peak
        rows.append(_row("demand", "peak demand vs ED1 GBFES peak (consumption plus losses)", peak_mw, ref, np.nan,
                         "INFO", src_ed1, f"{100 * (peak_mw / ref - 1.0):+.1f}%; average-cold-spell planning peak, "
                         "not CP30's dispatch basis"))
    # Reference = the CP30 demand build's own definition (cp30_demand, 15 Oct 2026): FES 2024 HT
    # Dem_BB003 GSP shares for everything except the transmission-connected named loads (FES24
    # ETYS type T, ED1 energy), which sit at their own nodes. The named loads come from the
    # build's own function (cp30_demand.named_load_reference); without them (the scenario does
    # not enable the build) the reference is the GSP shares alone.
    ref = ctx.refs.get("fes_demand_by_gsp")
    named = ctx.refs.get("transmission_named_loads") if c.get("named_loads", True) else None
    info = ctx.refs.get("demand_boundary_shares_info")  # DESNZ 2024 meter shares (committed, external source)
    bnds = [b for b in c["boundaries"] if b in ctx.geoms]
    # FES24 ETYS HT 2030/31 winter peak (all types), from rule extract_cp30_demand_inputs: INFO.
    etys = ctx.refs.get("fes24_etys_peak_by_gsp")
    etys_share = {}
    if etys is not None and len(etys):
        loc = etys.dropna(subset=["latitude", "longitude"])
        ex, ey = lonlat_to_osgb(loc["longitude"], loc["latitude"])
        esides = ctx.sides_of_points(pd.DataFrame({"x": ex, "y": ey}, index=loc.index), bnds)
        mw = pd.to_numeric(loc["peak_mw"], errors="coerce").fillna(0.0)
        total_mw = float(pd.to_numeric(etys["peak_mw"], errors="coerce").fillna(0.0).sum())
        etys_share = {b: 100.0 * float(mw[esides[b]].sum()) / max(total_mw, 1e-9) for b in bnds if b in esides}
        etys_src = (f"{str(etys['source'].iloc[0])}, all demand types; GSPs at their GSP info coordinates, "
                    "type-T direct connects at their FLOP-zone GSP centroid")
    rng = _reference_range(ctx, ref, "gwh", bnds) if ref is not None and len(ref) else {}
    ref_total = float(pd.to_numeric(ref["gwh"], errors="coerce").sum()) if ref is not None and len(ref) else np.nan
    f_named, named_remote, src = 0.0, {}, str(ref["source"].iloc[0]) if ref is not None and len(ref) else ""
    if named is not None and len(named):
        sh = pd.to_numeric(named["share_of_system_demand"], errors="coerce").fillna(0.0)
        f_named = float(sh.sum())
        nsides = ctx.sides_of_points(named[["x", "y"]], bnds)
        named_remote = {b: 100.0 * float(sh[nsides[b]].sum()) for b in bnds if b in nsides}
        src += f"; plus {100 * f_named:.2f}% transmission-connected named loads at their nodes ({str(named['source'].iloc[0])[:90]})"
    for b in bnds:
        share = 100.0 * float(energy[ctx.bus_remote(loads["bus"].astype(str), b).to_numpy()].sum()) / max(energy.sum(), 1e-9)
        if b in rng:
            lo, hi = ((1.0 - f_named) * 100.0 * v / ref_total + named_remote.get(b, 0.0) for v in rng[b])
            out = _band(share, lo, hi)
            status = "FAIL" if abs(out) > c["share_fail_pp"] else ("WARN" if abs(out) > c["share_warn_pp"] else "PASS")
            rows.append(_row("demand_share", f"{b} remote side", share, lo if abs(hi - lo) < 1e-9 else f"{lo:.2f}-{hi:.2f}",
                             c["share_warn_pp"], status, src,
                             f"share of GB annual demand (excl. electrolysis); WARN at {c['share_warn_pp']} pp, FAIL at "
                             f"{c['share_fail_pp']} pp", units="pp", difference=out))
        if b in etys_share:
            rows.append(_row("demand_share", f"{b} remote side vs FES24 ETYS winter peak", share, etys_share[b],
                             np.nan, "INFO", etys_src, units="pp"))
        if info is not None and b in set(info["boundary"]):
            r = info[info["boundary"] == b].iloc[0]
            rows.append(_row("demand_share", f"{b} remote side vs DESNZ 2024 meter consumption", share,
                             float(r["desnz_2024_pct"]), np.nan, "INFO", str(r["source"]), units="pp"))
    return rows


# ── Check 8: distributed generation ────────────────────────────────────────────

#: ``data_source`` of renewable rows the build places by the GSP-node rule besides the named fleet:
#: excluded not-built CfD sites refilled where the project would have been, and FES small-scale
#: solar held at its GSP node (``brownfield_fleet``). The named fleet is ``fleet_origin == named``.
GSP_NODE_RULE_SOURCES = ("CP30_refill_at_excluded_site", "CP30_small_scale_solar_gsp", "CP30_distributed_gsp_node")


def _project_capacity(df: pd.DataFrame) -> pd.Series:
    """Each unit's project capacity (MW): the sum over the units of one project.

    The build judges a project's size before splitting it into CfD and merchant parts or keeping
    its REPD phases apart (Cowley Complex is 40 + 80 MW, Brochloch Rig 61 + 29 MW), so the check
    does too: units share a project by REPD id, else by source site name, else they stand alone.
    """
    key = pd.Series(df.index.astype(str), index=df.index)
    for col in ("source_site_name", "repd_ref_id"):
        if col in df.columns:
            value = df[col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            ok = df[col].notna() & ~value.isin(["", "nan", "None"])
            key[ok] = col + ":" + value[ok]
    key = df["carrier"].astype(str) + "|" + key
    return df["p_nom"].groupby(key).transform("sum")


def _island_codes(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Substation codes of the island network a point lies in (``spatial_utils.ISLAND_NETWORKS``)."""
    from scripts.utilities.spatial_utils import ISLAND_NETWORKS

    out = pd.Series([frozenset()] * len(lat), index=lat.index, dtype=object)
    for spec in ISLAND_NETWORKS.values():
        lon_min, lat_min, lon_max, lat_max = spec["bbox"]
        inside = lon.between(lon_min, lon_max) & lat.between(lat_min, lat_max)
        out[inside] = [frozenset(spec["sites"])] * int(inside.sum())
    return out


def check_distributed(ctx: Context) -> List[dict]:
    """Distribution-size plant at its own GSP's node, judged by the build's own rule (16 Oct 2026).

    In scope: every battery (the build moves every distribution-size battery and every FES GSP
    row) and the renewable rows the GSP-node rule places (the named fleet, refills of excluded
    sites, FES small-scale solar). A project is distribution-size when its total is below its
    licence area's threshold. Its home is the substation of the node the build's locator gives
    its GSP polygon (``gsp_nodes.GspNodeLocator`` on this network and Dem_per_node), or any of the
    polygon's GSP substations, so a sibling busbar of that substation passes; a site inside an
    island network (Orkney) is at home on that island's substations. Documented TEC connections
    keep their bus by rule and regional proxy tranches are not sites; both are reported as INFO.
    """
    from types import SimpleNamespace

    from scripts.utilities.gsp_nodes import (DEFAULT_DEM_PER_NODE_FILE, DEFAULT_DISTRIBUTION_THRESHOLDS_MW,
                                             GspNodeLocator, distribution_threshold_mw, prepare_gsp_polygons)
    from scripts.utilities.spatial_utils import documented_connection_mask

    c = ctx.cfg["distributed"]
    rows = []
    g, s = ctx.tables.generators, ctx.tables.storage_units
    src = ("the build's GSP-node rule: site in its NESO GSP polygon; home = the substation of that polygon's demand "
           "node (gsp_nodes.GspNodeLocator, GB_network.xlsx Dem_per_node) or of its GSP IDs; project size against the "
           "licence-area threshold; data/network/GSP/GSP_regions_27700_20250109.geojson")
    if "GSPs" not in ctx.polygons.columns:
        return [_row("distributed", "GSP polygons", status="INFO", note="polygons carry no GSPs column")]
    parts = []
    for df, carriers, kind in ((g, c["generator_carriers"], "generator"), (s, c["storage_carriers"], "storage")):
        sub = df[_carrier(df).isin(carriers)]
        part = pd.DataFrame({"bus": sub["bus"].astype(str), "p_nom": _numeric(sub, "p_nom"),
                             "carrier": _carrier(sub), "lat": pd.to_numeric(sub.get("lat"), errors="coerce"),
                             "lon": pd.to_numeric(sub.get("lon"), errors="coerce"), "kind": kind}, index=sub.index)
        for col in ("source_site_name", "repd_ref_id", "fleet_origin", "data_source"):
            part[col] = sub[col] if col in sub.columns else np.nan
        parts.append(part)
    a = pd.concat(parts)
    a["project_mw"] = _project_capacity(a)
    located = a["lat"].notna() & a["lon"].notna()
    # Scope: batteries all; renewables the rows the GSP-node rule places (all when the network
    # carries no fleet_origin, i.e. was built without the named fleet).
    generator = a["kind"].eq("generator")
    if a.loc[generator, "fleet_origin"].notna().any():
        placed_by_rule = (a["fleet_origin"].astype(str).eq("named")
                          | a["data_source"].astype(str).isin(GSP_NODE_RULE_SOURCES))
    else:
        placed_by_rule = pd.Series(True, index=a.index)
    placed_by_rule |= ~generator
    documented = generator & documented_connection_mask(
        a.assign(site_name=a.index.astype(str)), ctx.tables.buses.index)
    # The build's locator on this network's buses and loads.
    poly = prepare_gsp_polygons(ctx.polygons)
    view = SimpleNamespace(buses=ctx.tables.buses, loads=ctx.tables.loads)
    dem = ctx.dem_per_node_file
    if isinstance(dem, pd.DataFrame):
        locator = GspNodeLocator(view, polygons=poly, dem_per_node=dem, log=logger)
    else:
        locator = GspNodeLocator(view, polygons=poly, dem_per_node_file=dem or DEFAULT_DEM_PER_NODE_FILE, log=logger)
    al = locator.assign(a[located])
    gsp_codes = al["gsp_ids"].map(lambda v: {p.strip()[:4] for p in str(v).split("|") if p.strip()}
                                  if isinstance(v, str) else set())
    node_code = al["gsp_node"].map(lambda v: {str(v)[:4]} if isinstance(v, str) else set())
    island = _island_codes(al["lat"], al["lon"])
    al["home"] = [a_ | b_ | set(c_) for a_, b_, c_ in zip(gsp_codes, node_code, island)]
    al["threshold"] = al["gsp_group"].map(lambda grp: distribution_threshold_mw(grp) if isinstance(grp, str)
                                          else np.nan)
    al["at_home"] = [str(b)[:4] in h for b, h in zip(al["bus"], al["home"])]
    distribution = al["project_mw"] < al["threshold"]
    judged = al[distribution & placed_by_rule.reindex(al.index) & ~documented.reindex(al.index)]
    thr = DEFAULT_DISTRIBUTION_THRESHOLDS_MW
    loads = _electricity_loads(ctx)
    p = ctx.tables.load_p.reindex(columns=loads.index).fillna(0.0)
    has_demand = set(loads["bus"][p.abs().sum(axis=0).to_numpy() > 0].astype(str).str[:4])
    for carrier in [x for x in list(c["generator_carriers"]) + list(c["storage_carriers"]) if x in set(al["carrier"])]:
        grp = judged[judged["carrier"].eq(carrier)]
        total = float(grp["p_nom"].sum())
        away = float(grp.loc[~grp["at_home"], "p_nom"].sum())
        share = away / total if total else 0.0
        status = "FAIL" if share > c["fail_share"] else ("WARN" if share > c["warn_share"] else "PASS")
        rows.append(_row("distributed", f"{carrier}: distribution-size MW not at its GSP node", away, total,
                         c["warn_share"], status, src,
                         f"{100 * share:.1f}% of {total:,.0f} MW (WARN above {100 * c['warn_share']:.0f}%, FAIL above "
                         f"{100 * c['fail_share']:.0f}%); thresholds {thr}", difference=share))
        if carrier == "solar_pv":
            nodem = float(grp.loc[~grp["bus"].str[:4].isin(has_demand), "p_nom"].sum())
            sh = nodem / total if total else 0.0
            rows.append(_row("distributed", "solar_pv: distribution-size MW on substations with no demand", nodem,
                             total, c["no_demand_warn_share"], "WARN" if sh > c["no_demand_warn_share"] else "PASS",
                             src, f"{100 * sh:.1f}%", difference=sh))
        in_carrier = al["carrier"].eq(carrier) & distribution
        doc = al[in_carrier & documented.reindex(al.index)]
        if len(doc):
            rows.append(_row("distributed", f"{carrier}: distribution-size documented TEC connections (not judged)",
                             float(doc["p_nom"].sum()), status="INFO", source=src,
                             note=f"{len(doc)} units kept at their TEC bus (spatial_utils.ONSHORE_RENEWABLE_DIRECT_BUS); "
                                  f"{float(doc.loc[~doc['at_home'], 'p_nom'].sum()):,.1f} MW of it away from the GSP node"))
        proxy = al[in_carrier & ~placed_by_rule.reindex(al.index)]
        if len(proxy):
            off = float(proxy.loc[~proxy["at_home"], "p_nom"].sum())
            tot = float(proxy["p_nom"].sum())
            rows.append(_row("distributed", f"{carrier}: regional proxy tranches below the threshold (not judged)",
                             off, tot, status="INFO", source=src, difference=off / tot if tot else 0.0,
                             note=f"{100 * off / tot if tot else 0:.1f}% of {tot:,.0f} MW not at the GSP node; ES5 "
                                  "residual tranches follow the FES proxy geography and export relief, not the "
                                  "GSP-node rule (which places sites)"))
    unl = float(a.loc[~located, "p_nom"].sum())
    rows.append(_row("distributed", "capacity without site coordinates (not judged)", unl, status="INFO", source=src))
    return rows


# ── Check 9: stranded capacity in radial pockets ───────────────────────────────

def effective_branch_ratings(ctx: Context) -> pd.DataFrame:
    """Lines and transformers with the BM-effective rating (relaxed branches -> inf); see :func:`bm_effective_ratings`."""
    return bm_effective_ratings(ctx.tables, ctx.transmission)


def bm_effective_ratings(tables, transmission: Optional[dict]) -> pd.DataFrame:
    """Lines and transformers with the rating the balancing solve enforces (relaxed branches -> inf).

    The branches relaxed are the balancing solve's own (``solve_network.bm_voltage_floor_branches``
    with the scenario's ``transmission.min_bm_constraint_voltage_kv`` and ``bm_voltage_floor``);
    ``transmission.line_rating_overrides`` then win, as in ``solve_network.apply_line_rating_overrides``
    (ELDE-JOHN is 1 MVA in ETYS and 149 MVA in the BM). ``tables`` needs ``buses``, ``lines`` and
    ``transformers``. Columns bus0, bus1, rating, kind and nominal (the rating before relaxation,
    after any override).
    """
    from types import SimpleNamespace

    from scripts.solve.solve_network import _floor_config, bm_voltage_floor_branches

    transmission = transmission or {}
    min_v = float(transmission.get("min_bm_constraint_voltage_kv", DEFAULT_BM_FLOOR_KV) or 0.0)
    relaxed_lines, relaxed_trafos = pd.Index([]), pd.Index([])
    if min_v > 0:
        scottish, scottish_kv = _floor_config(transmission)
        view = SimpleNamespace(buses=tables.buses, lines=tables.lines, transformers=tables.transformers)
        relaxed_lines, relaxed_trafos = bm_voltage_floor_branches(view, min_v, scottish, scottish_kv)
    overrides = transmission.get("line_rating_overrides", {}) or {}
    overrides = overrides if isinstance(overrides, dict) else {}
    out = []
    for kind, t, relaxed in (("line", tables.lines, relaxed_lines),
                             ("transformer", tables.transformers, relaxed_trafos)):
        if t.empty:
            continue
        s_nom = _numeric(t, "s_nom")
        for name, value in overrides.items():
            if name in s_nom.index:
                s_nom.at[name] = float(value)
        rating = s_nom * _numeric(t, "s_max_pu", 1.0)
        rating[t.index.isin(relaxed) & ~t.index.isin(list(overrides))] = np.inf
        out.append(pd.DataFrame({"bus0": t["bus0"].astype(str), "bus1": t["bus1"].astype(str), "rating": rating,
                                 "kind": kind, "nominal": s_nom}, index=t.index))
    return pd.concat(out) if out else pd.DataFrame(columns=["bus0", "bus1", "rating", "kind", "nominal"])


def internal_dc_links(tables) -> pd.DataFrame:
    """GB-internal HVDC links (carrier DC, both ends GB AC buses, not an interconnector) as branches."""
    links = tables.links
    if links.empty:
        return pd.DataFrame(columns=["bus0", "bus1", "rating", "kind", "nominal"])
    gb = _gb_mask(tables.buses)
    name = links.get("interconnector_name", pd.Series(np.nan, index=links.index))
    dc = links[_carrier(links).eq("DC") & name.isna() & links["bus0"].map(gb).fillna(False).astype(bool)
               & links["bus1"].map(gb).fillna(False).astype(bool)]
    p_nom = _numeric(dc, "p_nom")
    return pd.DataFrame({"bus0": dc["bus0"].astype(str), "bus1": dc["bus1"].astype(str), "rating": p_nom,
                         "kind": "dc_link", "nominal": p_nom}, index=dc.index)


class DemandHeadroom:
    """Each node's peak demand against what its circuits can bring in, month by month, at the BM's ratings (16 Oct 2026).

    The model behind check 11 (:func:`check_demand_headroom`) and the electrolyser import relief
    (``add_hydrogen_system.relieve_electrolyser_import_headroom``), so a build that passes its relief
    passes the check. For each month of ``tables.snapshots`` (one period, at winter ratings, when the
    tables carry no timestamps): the branch ratings the balancing solve enforces
    (:func:`bm_effective_ratings` plus GB-internal DC links, times the ETYS seasonal factor of
    ``transmission.seasonal_ratings`` from ``branch_attributes.seasonal_rating_factors``), the peak of
    each bus's electricity load in that month, and ``spatial_utils.NodeImportModel`` (bus-section nodes,
    the substation's lower-voltage busbars drawing through them, DC sharing over the circuits, the
    first circuit to reach its rating) on that load plus the electrolysis passed to :meth:`evaluate`.

    No credit is taken for local generation (it may be calm or dark at the peak) or storage: storage
    is priced in the BM and can stop charging (at ABNE it discharged 150 MW into the shedding hours),
    whereas CP30's electrolysers are held at their Stage 1 schedule
    (``market.balancing.hydrogen_links.hold_carriers``) and the BM can meet a held load the network
    cannot carry only by shedding.
    """

    def __init__(self, tables, transmission: Optional[dict] = None):
        from types import SimpleNamespace

        from scripts.network_build.branch_attributes import seasonal_rating_factors
        from scripts.utilities.spatial_utils import NodeImportModel, branch_x_pu

        transmission = transmission or {}
        self.tables = tables
        gb = _gb_mask(tables.buses)
        self.gb_buses = tables.buses.index[gb.to_numpy()]
        self.network = SimpleNamespace(buses=tables.buses.loc[self.gb_buses], lines=tables.lines)
        branches = pd.concat([bm_effective_ratings(tables, transmission), internal_dc_links(tables)])
        x_pu = branch_x_pu(tables.buses, tables.lines, tables.transformers)
        branches["x_pu"] = x_pu[~x_pu.index.duplicated()].reindex(branches.index)
        self.branches = branches
        self.model = NodeImportModel(self.network, branches)
        self.group = self.model.group
        loads = _electricity_loads(SimpleNamespace(tables=tables))
        p = tables.load_p.reindex(columns=loads.index).fillna(0.0)
        bus_load = p.T.groupby(loads["bus"].astype(str)).sum().T if len(loads) else pd.DataFrame(index=p.index)
        snapshots = getattr(tables, "snapshots", None)
        seasonal = transmission.get("seasonal_ratings", {}) or {}
        if snapshots is not None and len(snapshots) == len(p) and isinstance(pd.Index(snapshots), pd.DatetimeIndex):
            months = pd.DatetimeIndex(snapshots).month
            self.months = sorted(set(months))
            self.peak = bus_load.groupby(np.asarray(months)).max()
            factors = pd.DataFrame(index=self.months)
            if bool(seasonal.get("enabled", True)) and len(tables.lines):
                mid = pd.DatetimeIndex([pd.Timestamp(int(pd.DatetimeIndex(snapshots).year[0]), m, 15)
                                        for m in self.months])
                factors = seasonal_rating_factors(SimpleNamespace(lines=tables.lines), mid,
                                                  seasonal.get("season_months"))
                factors.index = self.months
        else:
            self.months = ["all"]
            self.peak = pd.DataFrame([bus_load.max()], index=self.months) if len(bus_load.columns) else \
                pd.DataFrame(index=self.months)
            factors = pd.DataFrame(index=self.months)
        self.ratings = {}
        for m in self.months:
            rating = self.branches["rating"].copy()
            if len(factors.columns):
                cols = factors.columns.intersection(rating.index)
                rating.loc[cols] = rating.loc[cols] * factors.loc[m, cols].to_numpy(dtype=float)
            self.ratings[m] = rating

    def evaluate(self, electrolysis_mw: Optional[pd.Series] = None, include_load: bool = True,
                 all_nodes: bool = False) -> pd.DataFrame:
        """``NodeImportModel.evaluate`` for every month: rows (month, node), with the peak load and the
        electrolysis at the node and its substation's lower-voltage busbars (``load_mw``, ``electrolysis_mw``)."""
        el = pd.Series(electrolysis_mw if electrolysis_mw is not None else {}, dtype=float)
        frames = []
        for m in self.months:
            load = self.peak.loc[m] if include_load and len(self.peak.columns) else pd.Series(dtype=float)
            out = self.model.evaluate(load.add(el, fill_value=0.0), ratings=self.ratings[m], all_nodes=all_nodes)
            if out.empty:
                continue
            behind = [self.model.behind[n] for n in out.index]
            out["electrolysis_mw"] = [float(el.reindex(b).fillna(0.0).sum()) for b in behind] if len(el) else 0.0
            out["load_mw"] = out["demand_mw"] - out["electrolysis_mw"]
            out["month"] = m
            frames.append(out.rename_axis("node").reset_index())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def worst(self, electrolysis_mw: Optional[pd.Series] = None, max_loading: float = 1.0,
              include_load: bool = True, all_nodes: bool = False) -> pd.DataFrame:
        """One row per node: its worst month (highest ratio), with ``headroom_mw`` = the least of
        max_loading x capacity - demand over the months and ``excess_mw`` = the most demand exceeds it."""
        rows = self.evaluate(electrolysis_mw, include_load, all_nodes)
        if rows.empty:
            return rows
        with np.errstate(invalid="ignore"):
            rows["room"] = max_loading * rows["capacity_mw"] - rows["demand_mw"]
        headroom = rows.groupby("node")["room"].min()
        worst = rows.sort_values(["ratio", "month"], ascending=[False, True], kind="stable").groupby("node").head(1)
        worst = worst.set_index("node")
        worst["headroom_mw"] = headroom.reindex(worst.index)
        worst["excess_mw"] = (-worst["headroom_mw"]).clip(lower=0.0)
        return worst.drop(columns="room")

    def effect(self, node, binding, binding_share, bus) -> float:
        """Node demand MW cleared per MW removed at ``bus`` (a bus of the node's pocket) on its binding circuit.

        The binding circuit's overload is ``excess x binding_share``; a MW removed at ``bus`` takes its own
        flow on that circuit off (``NodeImportModel.bus_flow``), so moving ``excess / effect`` clears it.
        1 for the node's own buses when the circuit carries the pocket evenly; about 0.5 for a load busbar
        that draws half through it. 0 when the bus does not load the circuit.
        """
        if not isinstance(binding, str) or not binding_share or not np.isfinite(binding_share):
            return 1.0 if self.group.get(bus) == node else 0.0
        flow = self.model.bus_flow(node, bus)
        if binding not in flow.index:
            return 0.0
        return float(flow[binding]) / float(binding_share)


def check_stranded(ctx: Context) -> List[dict]:
    import networkx as nx

    c = ctx.cfg["stranded"]
    rows = []
    branches = effective_branch_ratings(ctx)
    links = ctx.tables.links
    gb = _gb_mask(ctx.tables.buses)
    name = links.get("interconnector_name", pd.Series(np.nan, index=links.index))
    dc = links[_carrier(links).eq("DC") & name.isna() & links["bus0"].map(gb).fillna(False)
               & links["bus1"].map(gb).fillna(False)]
    edges = pd.concat([branches, pd.DataFrame({"bus0": dc["bus0"].astype(str), "bus1": dc["bus1"].astype(str),
                                               "rating": _numeric(dc, "p_nom"), "kind": "dc_link"})])
    H = nx.Graph()
    H.add_nodes_from(ctx.tables.buses.index[gb].astype(str))
    for _, e in edges.iterrows():
        if e.bus0 == e.bus1:
            continue
        if H.has_edge(e.bus0, e.bus1):
            H[e.bus0][e.bus1]["cap"] += e.rating
            H[e.bus0][e.bus1]["n"] += 1
        else:
            H.add_edge(e.bus0, e.bus1, cap=e.rating, n=1)
    g = ctx.tables.generators
    gen = _numeric(g, "p_nom")[~_carrier(g).isin(c["excluded_generator_carriers"])].groupby(g["bus"].astype(str)).sum()
    sto = _numeric(ctx.tables.storage_units, "p_nom").groupby(ctx.tables.storage_units["bus"].astype(str)).sum()
    cap_at = gen.add(sto, fill_value=0.0)
    src = ("bridges of the AC + internal DC graph at BM-effective ratings (sub-220 kV E&W branches relaxed; "
           "Scottish >= 132 kV kept; transmission.min_bm_constraint_voltage_kv / bm_voltage_floor)")
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    if not comps:
        return rows
    main = comps[0]
    for comp in comps[1:]:
        mw = float(cap_at.reindex(list(comp)).fillna(0.0).sum())
        if mw > 1.0:
            rows.append(_row("stranded", f"island {sorted(comp)[0]} ({len(comp)} buses)", mw, 0.0, 0.0, "WARN",
                             src, "generation/storage with no AC or internal DC path to the main grid"))
    Hm = H.subgraph(main).copy()
    bridges = list(nx.bridges(Hm))
    G2 = Hm.copy()
    G2.remove_edges_from(bridges)
    comp_of = {}
    members = []
    for i, cc in enumerate(nx.connected_components(G2)):
        members.append(cc)
        for n in cc:
            comp_of[n] = i
    root = max(range(len(members)), key=lambda i: len(members[i]))
    tree = nx.Graph()
    tree.add_nodes_from(range(len(members)))
    for u, v in bridges:
        tree.add_edge(comp_of[u], comp_of[v], bridge=(u, v), cap=Hm[u][v]["cap"], n=Hm[u][v]["n"])
    own = {i: float(cap_at.reindex(list(cc)).fillna(0.0).sum()) for i, cc in enumerate(members)}
    nbus = {i: len(cc) for i, cc in enumerate(members)}
    order = list(nx.dfs_preorder_nodes(tree, root))
    parent = {root: None}
    for u, v in nx.dfs_edges(tree, root):
        parent[v] = u
    sub, subn = dict(own), dict(nbus)
    for node in reversed(order):
        p = parent.get(node)
        if p is not None:
            sub[p] += sub[node]
            subn[p] += subn[node]
    flagged = []
    for node in order:
        p = parent.get(node)
        if p is None:
            continue
        e = tree[p][node]
        cap = float(e["cap"])
        if not np.isfinite(cap) or sub[node] <= 0:
            continue
        if sub[node] > (1.0 + c["margin"]) * cap:
            u, v = e["bridge"]
            inner = u if comp_of[u] == node else v
            flagged.append((sub[node] / max(cap, 1e-9), inner, u, v, cap, sub[node], subn[node], e["n"]))
    flagged.sort(reverse=True)
    for ratio, inner, u, v, cap, mw, n, npar in flagged[: int(c["max_rows"])]:
        rows.append(_row("stranded", f"pocket at {inner} behind {u}-{v}", mw, cap, c["margin"], "WARN", src,
                         f"{ratio:.2f} x connection; {n} buses; {npar} parallel branch(es)"))
    rows.append(_row("stranded", "radial pockets above connection + margin", float(sum(f[5] for f in flagged)),
                     np.nan, c["margin"], "WARN" if flagged else "PASS", src,
                     f"{len(flagged)} pockets (largest {flagged[0][0]:.2f} x)" if flagged else "none"))
    return rows


# ── Check 11: demand-side headroom ─────────────────────────────────────────────

#: Relative tolerance on the check 11 thresholds.
RATIO_TOL = 1e-6


def _month_label(m) -> str:
    import calendar
    return calendar.month_abbr[int(m)] if isinstance(m, (int, np.integer)) else str(m)


def check_demand_headroom(ctx: Context) -> List[dict]:
    """Every node's peak load plus electrolysis against what it can import, at the BM's ratings (16 Oct 2026).

    :class:`DemandHeadroom`, worst month per node. FAIL above ``fail_ratio``, WARN above
    ``warn_ratio``; and FAIL wherever the peak load alone exceeds the node's capacity by more
    than its firm local generation (carriers not in ``non_firm_generator_carriers``): such a
    node can be served only by load shedding. The calibration solve of 15 Oct 2026 shed
    215 GWh at BUMU3-, ABNE3- and BAIN3-, behind 1.28 GW of electrolysers sited on 132 kV
    busbars fed by 2 x 132 and 2 x 114 MVA circuits.
    """
    c = ctx.cfg["demand_headroom"]
    links = ctx.tables.links
    el = links[_carrier(links).isin(ctx.cfg["electrolysis"]["carriers"])] if len(links) else links
    elec = _numeric(el, "p_nom").groupby(el["bus0"].astype(str)).sum() if len(el) else pd.Series(dtype=float)
    model = DemandHeadroom(ctx.tables, ctx.transmission)
    worst = model.worst(elec)
    load_only = model.worst(None)
    g = ctx.tables.generators
    firm = g[~_carrier(g).isin(c["non_firm_generator_carriers"])]
    firm_by_site = _numeric(firm, "p_nom").groupby(firm["bus"].astype(str).str[:4]).sum()
    src = ("spatial_utils.NodeImportModel: bus-section nodes with their substation's lower-voltage busbars; DC "
           "sharing of the withdrawal over the node's circuits, the first to reach its BM rating (bm_effective_ratings: "
           "voltage floor, line_rating_overrides; x the ETYS seasonal factor of the month); peak load of the month "
           "plus electrolysis p_nom; no credit for local generation or storage")
    rows, flagged, el_buses = [], [], set()
    for node, r in worst.iterrows():
        lo = load_only.loc[node] if node in load_only.index else None
        load_short = 0.0 if lo is None else float(lo["demand_mw"] - lo["capacity_mw"])
        firm_mw = float(firm_by_site.get(str(node)[:4], 0.0))
        shed_only = lo is not None and load_short > max(firm_mw, 0.0) + 1e-6
        ratio = float(r["ratio"])
        # A relative 1e-6: the relief moves exactly the excess, leaving a node at its limit.
        if not (ratio > c["warn_ratio"] * (1.0 + RATIO_TOL) or shed_only):
            continue
        status = "FAIL" if (ratio > c["fail_ratio"] * (1.0 + RATIO_TOL) or shed_only) else "WARN"
        buses_behind = set(map(str, model.model.behind[node]))
        el_buses |= buses_behind
        bind = (f"first to bind {r['binding']} ({100 * r['binding_share']:.0f}% of the withdrawal, "
                f"{r['capacity_mw'] * r['binding_share']:,.1f} MW rating)" if isinstance(r["binding"], str)
                else "no single circuit binds")
        note = (f"{ratio:.2f} x in {_month_label(r['month'])} (WARN above {c['warn_ratio']:.2f} x, FAIL above "
                f"{c['fail_ratio']:.2f} x); peak load {r['load_mw']:,.1f} MW and electrolysis "
                f"{r['electrolysis_mw']:,.1f} MW at the node and its substation's lower-voltage busbars; {bind}; "
                f"constrained circuits sum to {r['import_mw']:,.1f} MW")
        if shed_only:
            note += (f"; peak load alone exceeds the capacity by {load_short:,.1f} MW against {firm_mw:,.1f} MW of "
                     "firm local generation: only load shedding can serve it")
        flagged.append((status == "FAIL", ratio, _row("demand_headroom", f"{node} node ({r['buses']}, {r['v_nom']:g} kV)",
                                                      float(r["demand_mw"]), float(r["capacity_mw"]), c["warn_ratio"],
                                                      status, src, note)))
    flagged.sort(key=lambda f: (f[0], f[1]), reverse=True)
    rows.extend(f[2] for f in flagged[: int(c["max_rows"])])
    n_fail = sum(1 for f in flagged if f[0])
    el_over = float(elec.reindex(sorted(el_buses)).fillna(0.0).sum()) if len(elec) and el_buses else 0.0
    rows.append(_row("demand_headroom", "nodes whose peak demand exceeds their import capacity", float(len(flagged)),
                     0.0, np.nan, "FAIL" if n_fail else ("WARN" if flagged else "PASS"), src,
                     f"{n_fail} FAIL, {len(flagged) - n_fail} WARN; {el_over:,.1f} MW of electrolysis on nodes above "
                     f"{c['warn_ratio']:.2f} x", units="nodes"))
    return rows


# ── Check 10: boundary exposure ────────────────────────────────────────────────

def check_exposure(ctx: Context) -> List[dict]:
    c = ctx.cfg["exposure"]
    rows = []
    g, s, links = ctx.tables.generators, ctx.tables.storage_units, ctx.tables.links
    gen = g[~_carrier(g).isin(ctx.cfg["stranded"]["excluded_generator_carriers"])]
    vre = gen[_carrier(gen).isin(["wind_onshore", "wind_offshore", "solar_pv"])]
    el = links[_carrier(links).isin(ctx.cfg["electrolysis"]["carriers"])]
    loads = _electricity_loads(ctx)
    p = ctx.tables.load_p.reindex(columns=loads.index).fillna(0.0)
    w = ctx.tables.weights.to_numpy()[: len(p)]
    mean_mw = pd.Series((p.to_numpy() * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9), index=loads.index)
    peak_t = int(np.argmax(p.sum(axis=1).to_numpy())) if len(p) else 0
    at_peak = p.iloc[peak_t] if len(p) else pd.Series(0.0, index=loads.index)
    src = "boundary_sides on data/network/ETYS/etys_boundaries_mar25_osgb36.geojson; remote side as configured"
    for b in [b for b in c["boundaries"] if b in ctx.geoms]:
        def side_sum(df, col="bus", values=None):
            v = _numeric(df, "p_nom") if values is None else values
            return float(v[ctx.bus_remote(df[col].astype(str), b).to_numpy()].sum())
        rem = ctx.bus_remote(loads["bus"].astype(str), b).to_numpy()
        for item, val in (("generation MW", side_sum(gen)), ("of which wind and solar MW", side_sum(vre)),
                          ("storage MW", side_sum(s)), ("electrolysis MW", side_sum(el, "bus0")),
                          ("mean demand MW", float(mean_mw[rem].sum())),
                          ("demand at GB peak MW", float(at_peak[rem].sum()))):
            rows.append(_row("exposure", f"{b} remote side: {item}", val, status="INFO", source=src))
    return rows


# ── Runner ─────────────────────────────────────────────────────────────────────

CHECKS: Dict[str, Callable[[Context], List[dict]]] = {
    "es5_regional": check_es5_regional,
    "offshore": check_offshore,
    "named_stations": check_named_stations,
    "electrolysis": check_electrolysis,
    "ldes": check_ldes,
    "interconnectors": check_interconnectors,
    "demand": check_demand,
    "distributed": check_distributed,
    "stranded": check_stranded,
    "exposure": check_exposure,
    "demand_headroom": check_demand_headroom,
}


def load_references(reference_dir) -> Dict[str, pd.DataFrame]:
    """Every CSV in the reference folder, keyed by file stem (the committed hand judgements)."""
    return {p.stem: pd.read_csv(p, comment="#") for p in sorted(Path(reference_dir).glob("*.csv"))}


# ── Reference tables derived at run time ───────────────────────────────────────

#: Keys of the tables :func:`derive_references` builds (never committed).
DERIVED_REFERENCES = ("fes_demand_by_gsp", "fes_electrolysis_by_gsp", "fes_regional_hydro_marine",
                      "transmission_named_loads", "offshore_landing_reference", "fes24_etys_peak_by_gsp",
                      "interconnectors_reference")


def _fes_pathway_column(fes: pd.DataFrame) -> str:
    for col in ("FES Pathway", "FES Scenario", "\ufeffFES Scenario"):
        if col in fes.columns:
            return col
    raise ValueError("FES data has no pathway/scenario column")


def fes_node_table(fes: pd.DataFrame, block: str, pathway: str, year: int, buses: pd.DataFrame,
                   gsp_info: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    """One FES building block by FES node, located (OSGB36 m).

    A node is located at its network bus when the network has it, else at its GSP (by GSP
    ID, then by the ID's first four characters) in the FES GSP info; a Direct(TO) row at its
    representative point from ``direct`` (node_id, licence, x, y, known_boundaries), whose
    side is known only on ``known_boundaries``.
    """
    col = _fes_pathway_column(fes)
    x = fes[(fes["Building Block ID Number"] == block) & (fes[col] == pathway)].copy()
    x["value"] = pd.to_numeric(x[str(year)], errors="coerce").fillna(0.0)
    x = x[x["value"] > 0]
    d = direct.set_index("node_id") if len(direct) else pd.DataFrame(columns=["licence", "x", "y", "known_boundaries"])
    by_id = gsp_info.drop_duplicates("GSP ID").set_index("GSP ID")
    by_prefix = gsp_info.assign(p=gsp_info["GSP ID"].astype(str).str[:4]).drop_duplicates("p").set_index("p")
    bx = pd.to_numeric(buses.get("x"), errors="coerce")
    by = pd.to_numeric(buses.get("y"), errors="coerce")
    out = []
    for _, r in x.iterrows():
        node, gid = str(r.get("Node ID")), str(r.get("GSP ID"))
        lic, known = "", ""
        if node in d.index:
            px, py, basis = float(d.at[node, "x"]), float(d.at[node, "y"]), "direct_representative_point"
            lic = str(d.at[node, "licence"])
            known = "" if pd.isna(d.at[node, "known_boundaries"]) else str(d.at[node, "known_boundaries"])
        elif node in buses.index and np.isfinite(bx.get(node, np.nan)):
            px, py, basis = float(bx[node]), float(by[node]), "network_bus"
        else:
            px = py = np.nan
            basis = "unplaced"
            for table, key in ((by_id, gid), (by_prefix, gid[:4])):
                if key in table.index:
                    px, py = lonlat_to_osgb(table.at[key, "Longitude"], table.at[key, "Latitude"])
                    px, py, basis = float(px), float(py), "fes_gsp_info"
                    break
        out.append({"gsp": r.get("GSP"), "gsp_id": r.get("GSP ID"), "node_id": r.get("Node ID"),
                    "licence_direct": lic, "value": round(float(r["value"]), 4), "x": round(px, 1),
                    "y": round(py, 1), "location_basis": basis, "known_boundaries": known})
    return pd.DataFrame(out, columns=["gsp", "gsp_id", "node_id", "licence_direct", "value", "x", "y",
                                      "location_basis", "known_boundaries"])


def fes_regional_table(node_tables: Dict[str, pd.DataFrame], direct: pd.DataFrame, polygons,
                       group_region: pd.Series, source: str) -> pd.DataFrame:
    """Technology x CP30 region MW: GSP nodes by the GSP polygon they fall in (else the nearest);
    Direct rows in their licence's ``region`` from ``direct``."""
    import geopandas as gpd

    region_of_licence = direct.set_index("licence")["region"].to_dict() if len(direct) else {}
    poly = polygons[["GSPGroup", "geometry"]].to_crs("EPSG:27700")
    rows = []
    for tech, (block, r) in node_tables.items():
        if r.empty:
            continue
        r = r.copy()
        is_direct = r["licence_direct"].ne("")
        pts = gpd.GeoDataFrame(r[~is_direct], geometry=gpd.points_from_xy(r.loc[~is_direct, "x"],
                                                                         r.loc[~is_direct, "y"]), crs="EPSG:27700")
        j = gpd.sjoin_nearest(pts, poly, how="left")
        j = j[~j.index.duplicated()]
        r["region"] = None
        r.loc[~is_direct, "region"] = j["GSPGroup"].map(group_region)
        r.loc[is_direct, "region"] = r.loc[is_direct, "licence_direct"].map(region_of_licence)
        for reg, mw in r.groupby("region")["value"].sum().items():
            rows.append({"technology": tech, "region": reg, "mw": round(float(mw), 3),
                         "source": f"{source}, {block}; GSP nodes placed by GSP polygon, Direct rows by licence "
                                   "(fes_direct_row_locations.csv)"})
    return pd.DataFrame(rows, columns=["technology", "region", "mw", "source"])


def offshore_landing_table(crosswalk: pd.DataFrame, additions: Optional[pd.DataFrame],
                           cfd: Optional[pd.DataFrame], crosswalk_name: str) -> pd.DataFrame:
    """The crosswalk's rows (they win), then the committed additions for farms outside it;
    ``excluded_not_built`` from the checked CfD contracts; ``sourced`` = evidence given."""
    ref = {}
    for _, r in (additions if additions is not None else pd.DataFrame()).iterrows():
        rid = int(r["repd_ref_id"])
        ref[rid] = dict(repd_ref_id=rid, site_name=str(r["site_name"]), landing_bus=str(r["landing_bus"]),
                        evidence=str(r["evidence"]), source=str(r["source"]))
    for _, r in crosswalk.iterrows():
        rid = int(r["repd_ref_id"])
        ref[rid] = dict(repd_ref_id=rid, site_name=str(r["site_name"]), landing_bus=str(r["landing_bus"]),
                        evidence=str(r["evidence"]), source=f"{crosswalk_name} ({r['source']})")
    out = pd.DataFrame(list(ref.values()), columns=["repd_ref_id", "site_name", "landing_bus", "evidence", "source"])
    out = out.sort_values("repd_ref_id").reset_index(drop=True)
    excluded = set()
    if cfd is not None and len(cfd):
        sel = cfd["action"].astype(str).eq("not_built_refill") & cfd["carrier"].astype(str).eq("wind_offshore")
        excluded = set(pd.to_numeric(cfd.loc[sel, "repd_ref_id"], errors="coerce").dropna().astype(int))
    out["excluded_not_built"] = out["repd_ref_id"].isin(excluded)
    out["sourced"] = ~out["evidence"].isin(["none", "nan", ""])
    missing = excluded - set(out["repd_ref_id"])
    if missing:
        logger.warning("Offshore farms excluded as not built have no landing reference: %s", sorted(missing))
    return out


def named_loads_table(demand_config: dict, pathway: str, year: int, ed1_path, es1_path,
                      buses: pd.DataFrame) -> pd.DataFrame:
    """The transmission-connected named loads as the CP30 demand build places them.

    Energy, split and loss factor from ``cp30_demand.named_load_reference`` (the build's own
    function); each (site, node) row at its node's network bus, else at the site's
    coordinates (where the build puts a load whose node is not in the network).
    """
    from scripts.demand import cp30_demand

    t = cp30_demand.named_load_reference(demand_config, pathway, year, ed1_path, es1_path, logger)
    bx = pd.to_numeric(buses.get("x"), errors="coerce")
    by = pd.to_numeric(buses.get("y"), errors="coerce")
    placed = []
    for _, r in t.iterrows():
        node = str(r["node"])
        if node and node in buses.index and np.isfinite(bx.get(node, np.nan)):
            placed.append((float(bx[node]), float(by[node]), "network bus"))
        else:
            x, y = lonlat_to_osgb(r["longitude"], r["latitude"])
            placed.append((float(x), float(y), "site coordinates"))
    out = t.assign(x=[p[0] for p in placed], y=[p[1] for p in placed], location_basis=[p[2] for p in placed])
    out.attrs.update(t.attrs)
    return out


def interconnector_reference(landings: pd.DataFrame, composition_path, year: int) -> pd.DataFrame:
    """The committed landings joined to CP30's cable composition (the build's own file and loader).

    Capacity = the composition's CP30 nominal rating (``cp30_mw``); a cable is in the CP30 build
    and counted when its action is keep or add and its ``first_year`` (blank = always) is not after
    ``year``; such a cable is required. An excluded cable (the Isle of Man link) or one not yet due is
    ``in_cp30_build`` false: absent is INFO, present is WARN. Every landing row must name its
    composition row (``composition_name``) and every composition row must have a landing row.
    Without a composition (the scenario does not declare one) every cable is required at its
    register rating (``register_mw``).
    """
    from scripts.CP30.storage_interconnectors import load_cp30_interconnector_composition

    if not composition_path:
        out = landings.copy()
        out["capacity_mw"] = pd.to_numeric(out["register_mw"], errors="coerce")
        out["in_cp30_build"] = out["counted_in_cp30"] = out["required"] = True
        return out
    comp = load_cp30_interconnector_composition(composition_path).set_index("name")
    names = landings["composition_name"].astype(str)
    unknown = sorted(set(names) - set(comp.index))
    missing = sorted(set(comp.index) - set(names))
    if unknown or missing:
        raise ValueError(f"interconnectors_2030.csv and {composition_path} disagree: landing rows without a "
                         f"composition row {unknown}; composition rows without a landing row {missing}")
    c = comp.reindex(names)
    first = pd.to_numeric(c["first_year"], errors="coerce").fillna(-np.inf).to_numpy()
    in_build = c["action"].isin(["keep", "add"]).to_numpy() & (first <= int(year))
    out = landings.copy()
    out["capacity_mw"] = pd.to_numeric(c["cp30_mw"], errors="coerce").to_numpy()
    out["in_cp30_build"] = in_build
    out["counted_in_cp30"] = in_build
    out["required"] = in_build
    out["composition_note"] = [
        f"composition {act}" + (f" from {int(f)}" if np.isfinite(f) else "") + f": {ev}"
        for act, f, ev in zip(c["action"], first, c.get("evidence", pd.Series("", index=c.index)).fillna(""))]
    out["source"] = [f"{s}; capacity = CP30 nominal rating in {Path(composition_path).name} ({n})"
                     for s, n in zip(out["source"].astype(str), names)]
    return out


def derive_references(sources: dict, cfg: dict, buses: pd.DataFrame, polygons, group_region: pd.Series,
                      committed: Dict[str, pd.DataFrame], pathway: str, year: int) -> tuple:
    """Build the run-time reference tables from their declared sources.

    ``sources`` keys (each optional; a table is built only when its sources are given):
    ``fes_data`` and ``gsp_info`` (FES node tables), ``offshore_crosswalk`` and
    ``cfd_contracts`` (offshore landings), ``cp30_demand`` (the scenario's demand-build
    config, with ``named_loads``/``demand_lines`` file overrides), ``cp30_ed1`` and
    ``cp30_es1`` (the named loads' loss factor), ``etys_peak`` (FES24 ETYS winter peak by
    ETYS id, an output of rule extract_cp30_demand_inputs), ``ic_composition`` (CP30's cable
    composition, the build's interconnectors.cp30_composition file).

    Returns (tables keyed as :data:`DERIVED_REFERENCES`, rows reporting a table that could
    not be built).
    """
    out, problems = {}, []

    def attempt(name, fn):
        try:
            value = fn()
            if value is not None:
                out[name] = value
        except Exception as exc:  # the checks report the missing table; the run continues
            logger.exception("CP30 locational reference %s could not be derived", name)
            problems.append(_row("references", f"{name} could not be derived", status="FAIL",
                                 note=f"{type(exc).__name__}: {exc}"))

    fr = cfg["fes_reference"]
    direct = committed.get("fes_direct_row_locations", pd.DataFrame(columns=["node_id", "licence", "x", "y",
                                                                               "known_boundaries", "region"]))
    if sources.get("fes_data"):
        fes_path = sources["fes_data"]
        fes_holder = {}

        def fes():
            if "fes" not in fes_holder:
                fes_holder["fes"] = pd.read_csv(fes_path, low_memory=False)
                fes_holder["info"] = pd.read_csv(sources.get("gsp_info") or fr["gsp_info_file"])
            return fes_holder["fes"], fes_holder["info"]

        label = f"FES {fr['FES_year']} GSP data ({fes_path}), {fr['FES_scenario']}, {year}"
        blocks = fr["blocks"]

        def node_table(block):
            f, info = fes()
            return fes_node_table(f, block, fr["FES_scenario"], year, buses, info, direct)

        def table(kind, value_col, note):
            t = node_table(blocks[kind]).rename(columns={"value": value_col})
            t["source"] = f"{label}, {blocks[kind]} ({note})"
            return t

        attempt("fes_demand_by_gsp", lambda: table("demand", "gwh", "GSP total consumption incl. distribution losses"))
        attempt("fes_electrolysis_by_gsp", lambda: table("electrolysis", "mw", "grid electrolysis"))
        attempt("fes_regional_hydro_marine", lambda: fes_regional_table(
            {"hydro": (blocks["hydro"], node_table(blocks["hydro"])),
             "marine": (blocks["marine"], node_table(blocks["marine"]))}, direct, polygons, group_region, label))
    if sources.get("offshore_crosswalk"):
        cw_path = sources["offshore_crosswalk"]
        cfd_path = sources.get("cfd_contracts")
        attempt("offshore_landing_reference", lambda: offshore_landing_table(
            pd.read_csv(cw_path), committed.get("offshore_landing_additions"),
            pd.read_csv(cfd_path) if cfd_path else None, Path(cw_path).name))
    if "interconnectors_2030" in committed and "composition_name" in committed["interconnectors_2030"]:
        attempt("interconnectors_reference", lambda: interconnector_reference(
            committed["interconnectors_2030"], sources.get("ic_composition"), year))
    if sources.get("etys_peak"):
        attempt("fes24_etys_peak_by_gsp", lambda: pd.read_csv(sources["etys_peak"]))
    demand_cfg = sources.get("cp30_demand") or {}
    if demand_cfg.get("enabled", False) and cfg["demand"].get("named_loads", True) and sources.get("cp30_ed1"):
        dc = copy.deepcopy(demand_cfg)
        if sources.get("named_loads"):
            dc.setdefault("named_loads", {})["file"] = sources["named_loads"]
        if sources.get("demand_lines"):
            dc["national_lines_file"] = sources["demand_lines"]
        attempt("transmission_named_loads", lambda: named_loads_table(
            dc, pathway, year, sources["cp30_ed1"], sources.get("cp30_es1"), buses))
    return out, problems


def run_locational_checks(tables: NetworkTables, config: Optional[dict], *, es5: pd.DataFrame,
                          ed1: Optional[pd.DataFrame] = None, es1: Optional[pd.DataFrame] = None,
                          pathway: str, year: int, refs: Optional[Dict[str, pd.DataFrame]] = None,
                          geoms: Optional[dict] = None, polygons=None, group_region: Optional[pd.Series] = None,
                          checks: Optional[Iterable[str]] = None, sources: Optional[dict] = None,
                          region_files: Optional[tuple] = None, transmission: Optional[dict] = None,
                          site_regions: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Run the locational checks and return the table (one row per item).

    ``refs`` are the committed reference tables (default: every CSV in ``reference_dir``);
    ``sources`` the declared files the derived references are built from at run time
    (:func:`derive_references`), which are added to ``refs``. Regions come from ``polygons``
    and ``group_region``, else from ``region_files`` (the build's GSP polygon and group map,
    read by ``spatial_allocation.load_gsp_region_map``; default :data:`REGION_FILES`).
    ``transmission`` is the scenario's transmission block (the BM voltage floor of check 9).
    """
    from scripts.CP30.spatial_allocation import load_gsp_region_map
    from scripts.solve.secured_boundaries import load_boundary_geometries

    cfg = resolve_config(config)
    refs = refs if refs is not None else load_references(cfg["reference_dir"])
    geoms = geoms if geoms is not None else load_boundary_geometries(cfg["boundary_file"])
    if polygons is None or group_region is None:
        polygons, group_region = load_gsp_region_map(*(region_files or REGION_FILES))
    rows = []
    if sources:
        refs = dict(refs)
        derived, problems = derive_references(sources, cfg, tables.buses, polygons, group_region, refs, pathway,
                                              int(year))
        refs.update(derived)
        rows.extend(problems)
    ctx = Context(tables, cfg, refs, geoms, polygons, group_region, es5, ed1, es1, pathway, int(year),
                  transmission=transmission or {}, site_regions=site_regions,
                  dem_per_node_file=(sources or {}).get("gsp_dem_per_node"))
    for name in (checks or CHECKS):
        try:
            rows.extend(CHECKS[name](ctx))
        except Exception as exc:  # a broken check must not hide the others
            logger.exception("CP30 locational check %s raised", name)
            rows.append(_row(name, "check raised an exception", status="FAIL", note=f"{type(exc).__name__}: {exc}"))
    out = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("model", "difference", "tolerance"):
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


#: The CP30 builds' GSP polygons and GSP group -> CP30 region map
#: (fleet.spatial_proxy.regional_allocation), for a run that names none.
REGION_FILES = ("data/network/GSP/GSP_regions_27700_20250109.geojson",
                "data/network/ETYS/cp30_es5_region_by_gsp_group.csv")


def locational_failures(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["status"].eq("FAIL")]


def summarise(table: pd.DataFrame) -> str:
    counts = table.groupby(["check", "status"]).size().unstack(fill_value=0).reindex(columns=list(STATUSES), fill_value=0)
    return counts.to_string()


def _run(network_path, es5_path, ed1_path, es1_path, pathway, year, config, output, logger_=None,
         sources: Optional[dict] = None, region_files: Optional[tuple] = None, transmission: Optional[dict] = None):
    from scripts.utilities.logging_config import setup_logging

    log = logger_ or setup_logging("validate_cp30_locational")
    cfg = resolve_config(config)
    if not cfg.get("enabled", True):
        pd.DataFrame([_row("locational", "checks disabled", status="INFO")], columns=COLUMNS).to_csv(output, index=False)
        return
    if str(network_path).endswith(".nc"):
        tables = NetworkTables.from_netcdf(network_path)
    else:
        from scripts.utilities.network_io import load_network
        tables = NetworkTables.from_network(load_network(network_path, custom_logger=log))
    table = run_locational_checks(
        tables, cfg, es5=pd.read_csv(es5_path), ed1=pd.read_csv(ed1_path) if ed1_path else None,
        es1=pd.read_csv(es1_path) if es1_path else None, pathway=pathway, year=year,
        sources={**(sources or {}), "cp30_ed1": ed1_path, "cp30_es1": es1_path},
        region_files=region_files, transmission=transmission)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 2000, "display.max_colwidth", 120):
        log.info("CP30 locational checks (status counts):\n%s", summarise(table))
        bad = table[table["status"].isin(["FAIL", "WARN"])]
        if len(bad):
            log.warning("CP30 locational FAIL/WARN rows:\n%s",
                        bad[["check", "item", "model", "reference", "difference", "status", "note"]].to_string(index=False))
    fails = locational_failures(table)
    if len(fails) and cfg.get("fail_on_fail", True):
        # Snakemake deletes a failed rule's outputs: keep the table beside it for the review.
        kept = Path(output).with_name(Path(output).stem + "_failed_run.csv")
        table.to_csv(kept, index=False)
        raise ValueError(f"CP30 locational validation failed on {len(fails)} rows (full table kept at {kept}):\n"
                         + fails[["check", "item", "model", "reference", "note"]].to_string(index=False))


if "snakemake" in globals():
    from scripts.utilities.logging_config import setup_logging as _setup_logging

    _sc = dict(snakemake.params.scenario_config)
    _log = _setup_logging(snakemake.log[0] if snakemake.log else "validate_cp30_locational")

    def _input(name):
        value = getattr(snakemake.input, name, None)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return str(value) if value else None

    _sources = {k: _input(k) for k in ("fes_data", "gsp_info", "offshore_crosswalk", "cfd_contracts",
                                       "named_loads", "demand_lines", "etys_peak", "ic_composition",
                                       "gsp_dem_per_node")}
    _sources["cp30_demand"] = dict(snakemake.params.cp30_demand or {})
    _run(snakemake.input.network, snakemake.input.cp30_es5, snakemake.input.cp30_ed1, snakemake.input.cp30_es1,
         snakemake.params.cp30_pathway, int(snakemake.params.modelled_year),
         (_sc.get("validation") or {}).get("locational"), snakemake.output.checks, logger_=_log, sources=_sources,
         region_files=(_input("gsp_polygons"), _input("gsp_group_regions")),
         transmission=(_sc.get("network") or {}).get("transmission") or {})
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CP30 locational and network validation")
    parser.add_argument("--network", required=True)
    parser.add_argument("--cp30-es5", default="resources/CP30/CP30_ES5_regional.csv")
    parser.add_argument("--cp30-ed1", default="resources/CP30/CP30_ED1.csv")
    parser.add_argument("--cp30-es1", default="resources/CP30/CP30_ES1.csv")
    parser.add_argument("--pathway", default="Further Flex and Renewables")
    parser.add_argument("--modelled-year", type=int, default=2030)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-only", action="store_true", help="do not raise on FAIL rows")
    parser.add_argument("--fes-data", default="resources/FES/FES_2024_data.csv")
    parser.add_argument("--named-loads", default="resources/demand/cp30_2030_named_large_loads.csv",
                        help="output of rule extract_cp30_demand_inputs")
    parser.add_argument("--demand-lines", default="resources/demand/fes24_ht_2030_demand_lines.csv",
                        help="output of rule extract_cp30_demand_inputs")
    parser.add_argument("--etys-peak", default="resources/demand/fes24_etys_ht_2030_peak_by_gsp.csv",
                        help="output of rule extract_cp30_demand_inputs")
    parser.add_argument("--ic-composition", default="data/interconnectors/cp30_interconnector_composition.csv",
                        help="CP30 cable composition (interconnectors.cp30_composition); '' for none")
    parser.add_argument("--offshore-crosswalk", default="data/network/ETYS/repd_offshore_landing_crosswalk.csv",
                        help="the build's REPD offshore landing crosswalk (fleet.brownfield.offshore_landing_crosswalk)")
    parser.add_argument("--cfd-contracts", default="data/market/lccc/checked_cfd_contracts.csv",
                        help="the build's checked CfD contracts (support.labels.support_registers)")
    parser.add_argument("--gsp-dem-per-node", default="data/network/ETYS/GB_network.xlsx",
                        help="GB_network.xlsx (Dem_per_node), read by the build's GSP-node locator")
    parser.add_argument("--no-named-loads", action="store_true",
                        help="the network was built without the CP30 demand build (demand.build disabled)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    _cfg = DEFAULT_CONFIG
    _run(args.network, args.cp30_es5, args.cp30_ed1, args.cp30_es1, args.pathway, args.modelled_year,
         {"fail_on_fail": not args.report_only}, args.output, logger_=logger,
         sources={"fes_data": args.fes_data, "gsp_info": _cfg["fes_reference"]["gsp_info_file"],
                  "offshore_crosswalk": args.offshore_crosswalk, "cfd_contracts": args.cfd_contracts,
                  "named_loads": args.named_loads, "demand_lines": args.demand_lines, "etys_peak": args.etys_peak,
                  "ic_composition": args.ic_composition or None, "gsp_dem_per_node": args.gsp_dem_per_node,
                  "cp30_demand": {} if args.no_named_loads else {"enabled": True}})
