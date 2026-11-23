"""FES offshore landing table of one pathway and year.

FES 2025 Holistic Transition puts 102.97 GW of its 103.93 GW of 2050 offshore
wind at "Direct" connections with no site (Gen_BB014: Direct(NGET) 72,336 MW,
Direct(SHETL) 23,900 MW, Direct(SPTL) 6,736 MW; the other 959 MW sit at eleven
named GSPs and are placed by the ordinary GSP mapping). So the landing table is
the whole offshore siting job for the year. The declared landing capacity (step 1)
is the same evidence in every year; only the FES totals it is scaled to (step 3)
and the boundary capability year (step 5) move with ``--year``.

Method:

1. **Declared landing capacity by landing bus**, from three sources:
   * REPD offshore projects of at least ``min_repd_mw`` (operational, under
     construction, consented, in planning), landed through
     ``data/network/ETYS/repd_offshore_landing_crosswalk.csv`` (evidence per
     project) or, where the crosswalk has no row, at the nearest surviving
     275/400 kV bus (``nearest_bus_fallback``);
   * the offshore-to-onshore circuits of the NESO Beyond 2030 Update workbook
     (June 2026), sheet ``Baseline - Offshore`` (HND, HNDFUE, Celtic Sea and
     INTOG), each interface point mapped to an ETYS bus in
     ``data/network/ETYS/fes2050_landing_point_bus_map.csv``. Their capacity is
     the cable capacity, which is what can come ashore there. The two Tee1
     onshore legs (Hunterston, Pentir-South) are transmission shared with WCD4,
     not generation, and are left out; MachairWind is split between them;
   * REPD projects that the workbook also carries are taken from the workbook
     only (``REPD_REPLACED_BY_HND``); Berwick Bank keeps its REPD capacity above
     the workbook's 1.8 GW Blyth circuit at Branxton (East Lothian).
2. **FES region of each landing bus** by the ETYS boundary geometry: SHETL
   north of B4, SPTL between B4 and B6, NGET south of B6.
3. **Scale to FES** within each region: capacity = declared x (FES Direct for
   the region / declared total for the region). One factor per region, so the
   regional totals equal FES exactly and the shares within a region follow the
   evidence.
4. **One row per landing bus.** Every landing bus must be in the base network
   the busmap was built on (the buses that survive dead-island pruning) and at
   or above ``min_landing_kv``. Its cluster in the boundary busmap is recorded.
5. **Boundary check.** For each ETYS boundary, the scaled landing capacity on
   the side away from London is set against the year's capability (the ETYS
   table stops at 2045, which is held after it). Offshore alone, before onshore
   generation and local demand; a screening flag, not a flow.

Command line: ``python -m scripts.network_clustering.fes_offshore_landing --year 2050``
(other paths default as below) writes the three ``data/network/ETYS/fes<year>_offshore_landing*``
files (``fes2050_offshore_landing.csv`` etc. for 2050).
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULTS = {
    "fes_data": "resources/FES/FES_2025_data.csv",
    "repd_operational": "resources/renewable/wind_offshore_sites.csv",
    "repd_pipeline": "resources/renewable/wind_offshore_pipeline_sites.csv",
    "crosswalk": "data/network/ETYS/repd_offshore_landing_crosswalk.csv",
    "beyond_2030": "data/fes_2050_sources/beyond_2030_update_june2026_383401.xlsx",
    "point_map": "data/network/ETYS/fes2050_landing_point_bus_map.csv",
    "network": "resources/network/CP30_FFR_2030_network.nc",
    "busmap": "data/clustering/fes2050_boundary_busmap.csv",
    "capability": "data/network/etys_boundary_capability.csv",
    "geometry": "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson",
}
#: Output files of a year; ``{year}`` is the FES year the table is scaled to.
OUTPUTS = {
    "out_landing": "data/network/ETYS/fes{year}_offshore_landing.csv",
    "out_projects": "data/network/ETYS/fes{year}_offshore_landing_projects.csv",
    "out_boundaries": "data/network/ETYS/fes{year}_offshore_landing_boundary_check.csv",
}
FES_PATHWAY = "Holistic Transition"
#: Short code of the pathway in the ETYS boundary capability table.
FES_PATHWAY_CODE = "HT"
FES_BLOCK = "Gen_BB014"
DIRECT_GSP = {"Direct(NGET)": "NGET", "Direct(SHETL)": "SHETL", "Direct(SPTL)": "SPTL"}
MIN_REPD_MW = 100.0   # smaller REPD farms are demonstrators or the GSP-connected FES rows
MIN_LANDING_KV = 275.0
# REPD pipeline projects the Beyond 2030 workbook carries as HND circuits.
REPD_REPLACED_BY_HND = {
    10918: "Mona: HND R4_4 at Bodelwyddan",
    11663: "Outer Dowsing: HND R4_3 at Weston Marsh",
    10772: "West of Orkney: HND SW_N1 at Spittal",
    12307: "Caledonia North: HND SW_NE4 at New Deer",
    17559: "Caledonia South: HND SW_NE4 at New Deer",
}
BERWICK_BANK_REPD = 9873
BERWICK_BANK_HND_MW = 1800.0
BERWICK_BANK_REMAINDER_BUS = "BRNX4-"
# Crosswalk rows with no ETYS bus whose TEC connection is now in the base network.
CROSSWALK_FILL = {2534: ("NEDE4J", "TEC Project a0l4L0000005ip7QAA: Moray East at New Deer 400 kV; "
                                   "NEDE4J is in the 2030 ETYS base network")}
# Exporting side of each boundary. North-to-south boundaries (B0 to B9): the
# side holding Dounreay (Caithness). Every other boundary: the side with fewer
# base-network buses (the peripheral region it encloses), with a note where
# that is not the exporting side.
_NORTH = ("DOUN2-", "north-to-south boundary; exporting side is the north")
EXPORT_SIDE = {
    "B0": _NORTH, "B1a": _NORTH, "B2": _NORTH, "B4": _NORTH, "B5": _NORTH, "B6": _NORTH, "B7a": _NORTH,
    "B8": _NORTH,
    "B9": ("DOUN2-", "exporting side is everything north of the Midlands, which also holds about half "
                     "of GB demand, so the ratio overstates"),
}
SMALLER_SIDE_NOTES = {
    "EC5": "East Anglia export boundary",
    "LE1": ("LE1 encloses London and the South East, which import across it; the enclosed side is "
            "reported, so this is not an export test"),
    "NW1": "North Wales export", "NW2": "North Wales export", "NW3": "North Wales export",
    "NW4": "North Wales export", "SC1": "south coast export", "SC1.5": "south coast export",
    "SC2": "south coast export", "SC3": "south-east coast export", "SW1": "South West export",
    "B13": "South West peninsula export",
}
CHECK_NOTE = ("offshore landing only, before onshore generation, local demand and the DC links the "
              "capability already counts; a screening flag, not a flow")
TEE1_TRANSMISSION = {("Tee1", "Hunterston"), ("Tee1", "Pentir-South")}
B2030_SOURCE = ("NESO Beyond 2030 Update workbook, June 2026 (https://www.neso.energy/document/383401/download), "
                "sheet Baseline - Offshore")
REPD_SOURCE = "REPD Q2 July 2025 (resources/renewable/wind_offshore_*sites.csv)"


def fes_direct_offshore(fes_data: str, year: int, pathway: str = FES_PATHWAY) -> pd.Series:
    """FES Direct offshore capacity (MW) of ``pathway`` in ``year`` by region: NGET, SHETL, SPTL."""
    column = str(int(year))
    d = pd.read_csv(fes_data, usecols=["Building Block ID Number", "FES Pathway", "GSP", column])
    d = d[(d["Building Block ID Number"] == FES_BLOCK) & (d["FES Pathway"] == pathway)
          & d["GSP"].isin(DIRECT_GSP)]
    out = d.groupby(d["GSP"].map(DIRECT_GSP))[column].sum()
    if set(out.index) != set(DIRECT_GSP.values()):
        raise ValueError(f"FES {FES_BLOCK} {pathway} {column}: Direct regions missing, got {dict(out)}")
    return out.astype(float)


def _nearest_bus(buses: pd.DataFrame, lat: float, lon: float) -> str:
    """Nearest bus (OSGB36 distance) to a WGS84 point."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(lon, lat)
    return str(np.hypot(buses["x"] - x, buses["y"] - y).idxmin())


def repd_rows(operational: str, pipeline: str, crosswalk: str, buses: pd.DataFrame,
              min_mw: float = MIN_REPD_MW) -> pd.DataFrame:
    """REPD offshore projects with a landing bus and evidence (one row per project)."""
    op = pd.read_csv(operational).assign(repd_status="Operational")
    pl = pd.read_csv(pipeline).rename(columns={"development_status": "repd_status"})
    cols = ["repd_ref_id", "site_name", "capacity_mw", "lat", "lon", "repd_status"]
    sites = pd.concat([op[cols], pl[cols]], ignore_index=True)
    sites["repd_ref_id"] = pd.to_numeric(sites["repd_ref_id"], errors="coerce").astype("Int64")
    sites = sites.drop_duplicates("repd_ref_id")
    cw = pd.read_csv(crosswalk)
    cw["repd_ref_id"] = pd.to_numeric(cw["repd_ref_id"], errors="coerce").astype("Int64")
    cw = cw.drop_duplicates("repd_ref_id").set_index("repd_ref_id")
    hv = buses[buses["v_nom"] >= MIN_LANDING_KV]
    rows = []
    for _, s in sites.iterrows():
        rid = int(s["repd_ref_id"])
        cap = float(s["capacity_mw"])
        name = str(s["site_name"]).replace("�", " ").replace("\xa0", " ").strip()
        base = {"project": name, "repd_ref_id": rid, "repd_status": s["repd_status"],
                "site_lat": float(s["lat"]), "site_lon": float(s["lon"])}
        if cap < min_mw:
            continue
        if rid in REPD_REPLACED_BY_HND:
            continue
        if rid == BERWICK_BANK_REPD:
            rows.append({**base, "declared_mw": cap - BERWICK_BANK_HND_MW, "source_type": "repd",
                         "landing_bus": BERWICK_BANK_REMAINDER_BUS, "landing_basis": "declared_split",
                         "evidence": ("Berwick Bank 4.1 GW across East Lothian and Blyth (developer, "
                                      "data/network/HND/hnd1_status_reconciliation.csv); 1.8 GW Blyth "
                                      "circuit from the Beyond 2030 workbook (PA_2); remainder at Branxton"),
                         "source": "https://www.berwickbank.com/cambois-onshore; " + REPD_SOURCE})
            continue
        bus, basis, ev, src = None, None, None, None
        if rid in cw.index and isinstance(cw.at[rid, "landing_bus"], str) and cw.at[rid, "landing_bus"].strip():
            bus = cw.at[rid, "landing_bus"].strip()
            basis, ev, src = f"crosswalk:{cw.at[rid, 'evidence']}", str(cw.at[rid, "notes"]), str(cw.at[rid, "source"])
        elif rid in CROSSWALK_FILL:
            bus, ev = CROSSWALK_FILL[rid]
            basis, src = "tec_register", "data/generators/tec-register-02-september-2025.csv"
        else:
            bus = _nearest_bus(hv, float(s["lat"]), float(s["lon"]))
            basis, ev, src = "nearest_bus_fallback", "no landing evidence in the crosswalk; nearest 275/400 kV bus to the REPD site", REPD_SOURCE
        rows.append({**base, "declared_mw": cap, "source_type": "repd", "landing_bus": bus,
                     "landing_basis": basis, "evidence": ev, "source": src})
    return pd.DataFrame(rows)


def beyond_2030_rows(workbook: str, point_map: str) -> pd.DataFrame:
    """Offshore-to-onshore circuits of the Beyond 2030 Update baseline, mapped to buses."""
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    rows = list(wb["Baseline - Offshore"].iter_rows(values_only=True))
    hdr = [str(h).strip() if h is not None else f"c{i}" for i, h in enumerate(rows[5])]
    df = pd.DataFrame(rows[6:], columns=hdr).dropna(how="all")
    df = df[df["Type of Work"].astype(str).str.strip() == "Offshore - Onshore"].copy()
    df["pair"] = list(zip(df["Interface point 1"].astype(str).str.strip(),
                          df["Interface point 2"].astype(str).str.strip()))
    df = df[~df["pair"].isin(TEE1_TRANSMISSION)]
    pm = pd.read_csv(point_map)
    out = []
    for _, r in df.iterrows():
        ip1, ip2 = r["pair"]
        maps = pm[pm["interface_point"] == ip2]
        if maps.empty:
            raise ValueError(f"Beyond 2030 interface point '{ip2}' has no row in {point_map}")
        if abs(maps["share"].sum() - 1.0) > 1e-9:
            raise ValueError(f"shares for interface point '{ip2}' do not sum to 1")
        for _, m in maps.iterrows():
            out.append({"project": f"{r['HND']} {ip1}", "repd_ref_id": pd.NA, "repd_status": "",
                        "site_lat": np.nan, "site_lon": np.nan,
                        "declared_mw": float(r["Capacity"]) * float(m["share"]), "source_type": "beyond_2030",
                        "landing_bus": m["landing_bus"], "landing_basis": m["status"],
                        "evidence": f"{r['Technology type']}; {ip1} to {ip2}; {m['basis']}",
                        "source": f"{B2030_SOURCE}; {m['source']}"})
    return pd.DataFrame(out)


def region_of_buses(buses: pd.DataFrame, geometry: str) -> pd.Series:
    """FES Direct region of each bus from the B4 and B6 sides."""
    from scripts.network_clustering.boundary_busmap import boundary_sides
    from scripts.solve.secured_boundaries import load_boundary_geometries

    geoms = load_boundary_geometries(geometry)
    sides = boundary_sides(buses[["x", "y"]].astype(float), geoms, ["B4", "B6"])
    north = sides.loc["BEAU4-"]                 # Beauly: SHE Transmission
    south = sides.loc[_london_bus(buses)]       # London: NGET
    region = pd.Series("SPTL", index=buses.index)
    region[sides["B4"] == north["B4"]] = "SHETL"
    region[sides["B6"] == south["B6"]] = "NGET"
    return region


def _london_bus(buses: pd.DataFrame) -> str:
    """Bus nearest central London (OSGB36 530000, 180000)."""
    return str(np.hypot(buses["x"] - 530000.0, buses["y"] - 180000.0).idxmin())


def build_landing_table(paths: Optional[Dict[str, str]] = None, min_repd_mw: float = MIN_REPD_MW,
                        year: int = 2050, pathway: str = FES_PATHWAY, pathway_code: str = FES_PATHWAY_CODE):
    """Build the project list, the per-bus landing table and the boundary check of one year.

    Args:
        paths: overrides of :data:`DEFAULTS`.
        min_repd_mw: smallest REPD project counted.
        year: FES year the landing capacity is scaled to (and the boundary capability year).
        pathway: FES pathway name; ``pathway_code`` its code in the capability table.

    Returns:
        (projects, landing, boundary_check) DataFrames.
    """
    import pypsa
    from scripts.network_clustering.boundary_busmap import boundary_sides
    from scripts.solve.secured_boundaries import load_boundary_geometries, load_capabilities

    p = {**DEFAULTS, **(paths or {})}
    network = pypsa.Network(p["network"])
    buses = network.buses[["x", "y", "v_nom", "lat", "lon"]].copy()
    fes = fes_direct_offshore(p["fes_data"], year, pathway)
    projects = pd.concat([repd_rows(p["repd_operational"], p["repd_pipeline"], p["crosswalk"], buses, min_repd_mw),
                          beyond_2030_rows(p["beyond_2030"], p["point_map"])], ignore_index=True)
    missing = sorted(set(projects["landing_bus"]) - set(buses.index))
    if missing:
        raise ValueError(f"landing buses absent from the base network (pruned or renamed): {missing}")
    low = projects[projects["landing_bus"].map(buses["v_nom"]) < MIN_LANDING_KV]
    if len(low):
        raise ValueError(f"landing buses below {MIN_LANDING_KV:g} kV: {sorted(set(low['landing_bus']))}")
    region = region_of_buses(buses, p["geometry"])
    projects["fes_region"] = projects["landing_bus"].map(region)
    declared = projects.groupby("fes_region")["declared_mw"].sum()
    factor = (fes / declared).reindex(fes.index)
    if factor.isna().any():
        raise ValueError(f"no declared landing capacity for FES region(s) {list(factor.index[factor.isna()])}")
    projects["scale_factor"] = projects["fes_region"].map(factor)
    projects["capacity_mw"] = projects["declared_mw"] * projects["scale_factor"]

    busmap = pd.read_csv(p["busmap"], usecols=["bus_id", "cluster"]).set_index("bus_id")["cluster"]
    g = projects.groupby("landing_bus")
    landing = pd.DataFrame({
        "carrier": "wind_offshore",
        "fes_region": g["fes_region"].first(),
        "capacity_mw": g["capacity_mw"].sum(),
        "declared_mw": g["declared_mw"].sum(),
        "operational_mw": projects[projects["repd_status"] == "Operational"].groupby("landing_bus")["declared_mw"].sum(),
        "repd_pipeline_mw": projects[(projects["source_type"] == "repd") & (projects["repd_status"] != "Operational")
                                     ].groupby("landing_bus")["declared_mw"].sum(),
        "beyond_2030_mw": projects[projects["source_type"] == "beyond_2030"].groupby("landing_bus")["declared_mw"].sum(),
        "scale_factor": g["scale_factor"].first(),
        "n_projects": g.size(),
        "projects": g["project"].apply(lambda s: "; ".join(s)),
        "landing_basis": g["landing_basis"].apply(lambda s: "; ".join(sorted(set(s)))),
    }).fillna({"operational_mw": 0.0, "repd_pipeline_mw": 0.0, "beyond_2030_mw": 0.0})
    landing.index.name = "landing_bus"
    landing["v_nom"] = buses["v_nom"].reindex(landing.index)
    landing["lat"] = buses["lat"].reindex(landing.index)
    landing["lon"] = buses["lon"].reindex(landing.index)
    landing["cluster"] = busmap.reindex(landing.index)
    landing["fes_region_direct_mw"] = landing["fes_region"].map(fes)
    landing["source"] = (f"FES 2025 {FES_BLOCK} {pathway} {year} Direct by TO (resources/FES/FES_2025_data.csv); "
                         f"shares from {REPD_SOURCE}, repd_offshore_landing_crosswalk.csv and {B2030_SOURCE}; "
                         f"per-project evidence in fes{year}_offshore_landing_projects.csv")
    has_repd = landing["operational_mw"] + landing["repd_pipeline_mw"] > 0
    has_b2030 = landing["beyond_2030_mw"] > 0
    kind = pd.Series("beyond_2030", index=landing.index)
    kind[has_repd] = "repd"
    kind[has_repd & has_b2030] = "repd_and_beyond_2030"
    landing["status"] = kind + "_scaled_to_fes_region"
    if landing["cluster"].isna().any():
        raise ValueError(f"landing buses without a cluster: {list(landing.index[landing['cluster'].isna()])}")

    # Boundary check: landing capacity on each boundary's exporting side vs the year's capability
    # (the last published year after the table ends: 2045 for 2050).
    geoms = load_boundary_geometries(p["geometry"])
    cap = load_capabilities(p["capability"], year, pathway_code, hold_last_year=True)
    # Named by the latest year read (a boundary whose table ends sooner holds its own last year).
    cap_column = f"capability_{int(cap['year'].max())}_mw"
    sides = boundary_sides(buses[["x", "y"]].astype(float), geoms, list(cap.index))
    land_sides = sides.loc[landing.index]
    rows = []
    for b in cap.index:
        if b in EXPORT_SIDE:
            ref, note = EXPORT_SIDE[b]
            export_value = sides.at[ref, b]
        else:
            counts = sides[b].value_counts()
            export_value = counts.idxmin()
            ref = f"smaller side ({int(counts.min())} of {len(sides)} buses)"
            note = SMALLER_SIDE_NOTES.get(b, "")
        export = land_sides.index[land_sides[b] == export_value]
        mw = float(landing.loc[export, "capacity_mw"].sum())
        other = float(landing["capacity_mw"].sum()) - mw
        c = float(cap.at[b, "capability_mw"])
        ratio = mw / c
        rows.append({"boundary": b, cap_column: c, "export_side": ref,
                     "offshore_landing_export_side_mw": mw, "offshore_landing_other_side_mw": other,
                     "landing_buses_export_side": len(export), "ratio": ratio,
                     "flag": "gross" if ratio > 1.5 else ("exceeds" if ratio > 1.0 else ""),
                     "note": note + ("; " if note else "") + CHECK_NOTE})
    check = pd.DataFrame(rows).set_index("boundary")
    return projects, landing.reset_index(), check


def main(argv=None) -> None:
    """Write the three landing files of one year."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, required=True, help="FES year the landing table is scaled to")
    ap.add_argument("--pathway", default=FES_PATHWAY)
    ap.add_argument("--pathway-code", default=FES_PATHWAY_CODE)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    for k, v in OUTPUTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=None, help=f"default {v}")
    ap.add_argument("--min-repd-mw", type=float, default=MIN_REPD_MW)
    args = vars(ap.parse_args(argv))
    min_mw = args.pop("min_repd_mw")
    year, pathway, code = args.pop("year"), args.pop("pathway"), args.pop("pathway_code")
    for k, v in OUTPUTS.items():
        args[k] = args[k] or v.format(year=year)
    logging.basicConfig(level=logging.INFO)
    projects, landing, check = build_landing_table(args, min_mw, year, pathway, code)
    projects.to_csv(args["out_projects"], index=False)
    landing.to_csv(args["out_landing"], index=False)
    check.to_csv(args["out_boundaries"])
    print(landing.groupby("fes_region")[["declared_mw", "capacity_mw"]].sum().round(1))
    print(landing.groupby("fes_region").agg(buses=("landing_bus", "size"), factor=("scale_factor", "first")))
    print(check.round(2).to_string())


if __name__ == "__main__":
    main()
