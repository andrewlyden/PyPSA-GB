"""Extract the CP30 2030 demand inputs from the FES 2024 workbooks (demand audit, 15 Oct 2026).

Rule ``extract_cp30_demand_inputs`` (``rules/demand.smk``) runs this script; its outputs
are declared inputs of ``build_base_demand`` (read by ``scripts/demand/cp30_demand.py``)
and of ``validate_CP30_locational`` (the named-loads reference of the demand check), so a
change to any source workbook or to the node choices re-runs the chain. It also runs by
hand from the repository root::

    python scripts/demand/extract_cp30_demand_inputs.py \\
        --named-out resources/demand/cp30_2030_named_large_loads.csv \\
        --lines-out resources/demand/fes24_ht_2030_demand_lines.csv

Inputs (all in the repository):

``data/network/ETYS/Regional breakdown of FES24 data.xlsx``
    Sheets Active (demand by ETYS id, type, scenario and year), GSP info (GSP
    coordinates by FLOP zone) and MAIN DATA (ETYS id -> name and FLOP zone).
``data/network/ETYS/GB_network.xlsx``
    Sheet Dir_con_BMUs_to_node: the node(s) of every directly connected BMU with its
    split shares.
``data/demand/cp30_named_load_nodes.csv``
    The node choices for the type-T rows that carry no BMU id (the B_EXTRA rows), each
    with its evidence: a declared hand judgement, not code.
``data/FES/Future Energy Scenarios 2024 Data Workbook_V006_0.xlsx``
    Sheet ED1 (FES 2024 national demand lines).

Outputs:

named large loads (``resources/demand/cp30_2030_named_large_loads.csv``)
    Every transmission direct connect (ETYS demand type "T") with demand at the Holistic
    Transition 2030/31 winter peak. Rows: data centres (the "(Data_centre)" B_EXTRA rows),
    rail (HS2 and Network Rail feeders), offshore electrification (INTOG, St Fergus), the
    Port Talbot electric arc furnace, and the industrial direct connects (T_/M_ BMUs).
    Node: ``Dir_con_BMUs_to_node`` for the BMUs (with its split shares); the node file
    for the B_EXTRA rows.

FES 2024 ED1 lines (``resources/demand/fes24_ht_2030_demand_lines.csv``)
    The FES 2024 ED1 Holistic Transition 2030 lines the build uses: the national energy
    of the data centres (transmission and distribution connected) and of the direct
    transmission customers, and the at-peak figures of EVs, heat pumps and district heat
    the shape calibration reports against. FES 2024 is the vintage CP30 was built on.

FES24 ETYS winter peak by ETYS id (``resources/demand/fes24_etys_ht_2030_peak_by_gsp.csv``)
    Every Active row of HT 2030/31 (all demand types) with its location: the GSP's
    coordinates in GSP info, else (type-T direct connects) its FLOP-zone GSP centroid. The
    CP30 locational validation reports the placed demand's boundary shares against it
    (INFO rows).

The regression fixtures ``tests/fixtures/cp30_demand/*.csv`` hold the reviewed tables of
15 Oct 2026; ``tests/unit/test_cp30_demand_build.py`` checks this extraction reproduces
them byte for byte.
"""

# No `from __future__` import: Snakemake's script directive prepends a preamble.

import argparse
import datetime
import functools
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ETYS_BREAKDOWN = "data/network/ETYS/Regional breakdown of FES24 data.xlsx"
GB_NETWORK = "data/network/ETYS/GB_network.xlsx"
NODE_CHOICES = "data/demand/cp30_named_load_nodes.csv"
FES24_WORKBOOK = "data/FES/Future Energy Scenarios 2024 Data Workbook_V006_0.xlsx"
NAMED_OUT = "resources/demand/cp30_2030_named_large_loads.csv"
LINES_OUT = "resources/demand/fes24_ht_2030_demand_lines.csv"
ETYS_PEAK_OUT = "resources/demand/fes24_etys_ht_2030_peak_by_gsp.csv"

SCENARIO = "HT"
ETYS_YEAR = 30          # 2030/31
PATHWAY = "Holistic Transition"
YEAR = 2030

NodeChoice = Tuple[str, str, Optional[float], Optional[float]]


def read_node_choices(path: str | Path) -> Dict[str, NodeChoice]:
    """``etys_id -> (nodes, evidence, lat, lon)`` from the node-choices file.

    ``nodes`` is ";"-separated ("" = no ETYS substation); lat/lon are None when blank.
    """
    df = pd.read_csv(path, comment="#", dtype={"etys_id": str, "nodes": str, "node_evidence": str})
    for col in ("etys_id", "nodes", "node_evidence", "latitude", "longitude"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col}")
    if df["etys_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated etys_id {sorted(df.loc[df['etys_id'].duplicated(), 'etys_id'])}")
    if df["node_evidence"].isna().any():
        raise ValueError(f"{path}: every row needs node_evidence")
    out = {}
    for _, r in df.iterrows():
        lat = None if pd.isna(r["latitude"]) else float(r["latitude"])
        lon = None if pd.isna(r["longitude"]) else float(r["longitude"])
        if (lat is None) != (lon is None):
            raise ValueError(f"{path}: {r['etys_id']} has only one of latitude/longitude")
        out[str(r["etys_id"]).strip()] = ("" if pd.isna(r["nodes"]) else str(r["nodes"]).strip(),
                                          str(r["node_evidence"]), lat, lon)
    return out


def category_of(etys_id: str, name: str) -> str:
    low = str(name).lower()
    if "data_centre" in low:
        return "data_centre"
    if "(rail)" in low or "(n_rail)" in low or "network rail" in low or low.endswith(" nr") \
            or "rtk" in str(etys_id).lower() or "rlink" in low:
        return "rail"
    if "intog" in low:
        return "offshore_electrification"
    if "port talbot" in low:
        return "steel_eaf"
    return "industrial"


@functools.lru_cache(maxsize=2)
def read_etys_breakdown(etys_breakdown: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sheets Active, GSP info and MAIN DATA of the FES24 regional breakdown (read once)."""
    act = pd.read_excel(etys_breakdown, "Active", header=2).iloc[:, :7]
    info = pd.read_excel(etys_breakdown, "GSP info", header=4).iloc[:, 1:8]
    md = pd.read_excel(etys_breakdown, "MAIN DATA", header=14).iloc[:, :4]
    return act, info, md


def extract_named_loads(etys_breakdown: str | Path = ETYS_BREAKDOWN, gb_network: str | Path = GB_NETWORK,
                        node_choices: str | Path = NODE_CHOICES) -> pd.DataFrame:
    """The type-T direct connects of FES24 ETYS HT 2030/31 with their node(s) and location."""
    choices = read_node_choices(node_choices)
    act, info, md = read_etys_breakdown(str(etys_breakdown))
    t = act[(act["scenario"] == SCENARIO) & (act["year"] == ETYS_YEAR) & (act["type"] == "T")]
    t = t.groupby("GSP", as_index=False)[["DemandPk", "DemandAM", "DemandPM"]].sum()
    t = t[t["DemandPk"] > 0.5]
    names = md.drop_duplicates("Elexon ID").set_index("Elexon ID")
    flop_centroid = info.groupby("Minor FLOP")[["Latitude", "Longitude"]].mean()
    bmu = pd.read_excel(gb_network, "Dir_con_BMUs_to_node")
    bmu_nodes = pd.Series({
        unit: ";".join(f"{n}:{s / g['% of Metered Volume'].sum():.6f}"
                       for n, s in zip(g["Node Id"], g["% of Metered Volume"]))
        for unit, g in bmu.groupby("BM Unit Id")})
    rows = []
    for _, r in t.iterrows():
        gid = str(r["GSP"])
        name = str(names["Name"].get(gid, gid)).replace("�", "")
        flop = names["Minor FLOP"].get(gid)
        if gid in choices:
            nodes, evidence, lat, lon = choices[gid]
            parts = [p for p in nodes.split(";") if p]
            nodes = ";".join(f"{p}:{1 / len(parts):.6f}" for p in parts)
        elif gid in bmu_nodes.index:
            nodes, evidence, lat, lon = bmu_nodes[gid], "GB_network.xlsx Dir_con_BMUs_to_node", None, None
        else:
            raise ValueError(f"No node for named load {gid} ({name}): add a row to {node_choices}")
        if lat is None and isinstance(flop, str) and flop in flop_centroid.index:
            lat, lon = flop_centroid.loc[flop, ["Latitude", "Longitude"]]
            evidence += f"; location = FLOP {flop} GSP centroid"
        rows.append({
            "etys_id": gid, "name": name, "category": category_of(gid, name), "minor_flop": flop,
            "peak_mw_ht_2030_31": round(float(r["DemandPk"]), 3),
            "summer_am_mw": round(float(r["DemandAM"]), 3), "summer_pm_mw": round(float(r["DemandPM"]), 3),
            "nodes": nodes, "latitude": None if lat is None else round(float(lat), 5),
            "longitude": None if lon is None else round(float(lon), 5), "node_evidence": evidence,
            "source": "Regional breakdown of FES24 data.xlsx, Active, HT, 2030/31, type T",
        })
    unused = sorted(set(choices) - set(t["GSP"].astype(str)))
    if unused:
        logger.warning("Node choices with no type-T row at the HT 2030/31 peak (unused): %s", unused)
    return pd.DataFrame(rows).sort_values(["category", "peak_mw_ht_2030_31"], ascending=[True, False])


def extract_etys_peak_by_gsp(etys_breakdown: str | Path = ETYS_BREAKDOWN) -> pd.DataFrame:
    """FES24 ETYS HT 2030/31 winter-peak demand by ETYS id and demand type, located (WGS84).

    Location: the id's row in GSP info; else (the type-T direct connects, which are not
    GSPs) the GSP centroid of its FLOP zone (MAIN DATA). Unlocated rows are kept with blank
    coordinates.
    """
    act, info, md = read_etys_breakdown(str(etys_breakdown))
    a = act[(act["scenario"] == SCENARIO) & (act["year"] == ETYS_YEAR)]
    g = a.groupby(["GSP", "type"], as_index=False)["DemandPk"].sum()
    by_id = info.drop_duplicates("GSP ID").set_index("GSP ID")
    flop_of = md.drop_duplicates("Elexon ID").set_index("Elexon ID")["Minor FLOP"]
    flop_centroid = info.groupby("Minor FLOP")[["Latitude", "Longitude"]].mean()
    rows = []
    for _, r in g.iterrows():
        gid = str(r["GSP"])
        lat = lon = None
        basis = "unlocated"
        if gid in by_id.index:
            lat, lon, basis = by_id.at[gid, "Latitude"], by_id.at[gid, "Longitude"], "GSP info"
        else:
            flop = flop_of.get(gid)
            if isinstance(flop, str) and flop in flop_centroid.index:
                lat, lon = flop_centroid.loc[flop, ["Latitude", "Longitude"]]
                basis = f"FLOP {flop} GSP centroid"
        rows.append({"etys_id": gid, "type": r["type"], "peak_mw": round(float(r["DemandPk"]), 3),
                     "latitude": None if lat is None else round(float(lat), 5),
                     "longitude": None if lon is None else round(float(lon), 5), "location_basis": basis,
                     "source": "Regional breakdown of FES24 data.xlsx, Active, HT, 2030/31 winter peak"})
    return pd.DataFrame(rows)


ED1_LINES = {
    # key: ED1 data item (the unit row: "Peak" for *_gw keys, "Annual..." otherwise)
    "data_centres_transmission_gwh": "GB Customer Demand: Commercial Data Centres - Direct Transmission Customers",
    "data_centres_distribution_gwh": "GB Customer Demand: Commercial Data Centres - Distribution System Customers",
    "direct_transmission_customers_gwh": "GBFES Customer Demand: Direct Transmission Customers",
    "residential_evs_gwh": "Residential Electric Vehicles Demand (EVs)",
    "commercial_evs_gwh": "Commercial Electric Vehicles Demand (EVs)",
    "rail_traction_gwh": "I&C Rail Traction Demand",
    "transmission_losses_gwh": "GBFES System Losses: Transmission",
    "distribution_losses_gwh": "GBFES System Losses: Distribution ",
    "peak_total_gw": "GBFES Peak Customer Demand: Total Consumption plus Losses ",
    "peak_direct_connects_gw": "GB FES System Demand: Direct Connects",
    "residential_evs_at_peak_gw": "Residential EVs at Peak",
    "commercial_evs_at_peak_gw": "Commercial EVs at Peak",
    "residential_heat_pumps_at_peak_gw": "Residential Heat Pumps at Peak",
    "commercial_heat_pumps_at_peak_gw": "Commercial Heat Pumps at Peak",
    "residential_district_heat_at_peak_gw": "Residential District Heat at Peak",
    "commercial_district_heat_at_peak_gw": "Commercial District Heat at Peak",
    "industrial_district_heat_at_peak_gw": "Industrial District Heat at Peak",
}


def extract_ed1_lines(fes24_workbook: str | Path = FES24_WORKBOOK) -> pd.DataFrame:
    """The FES 2024 ED1 Holistic Transition 2030 lines of :data:`ED1_LINES`."""
    import openpyxl

    wb = openpyxl.load_workbook(fes24_workbook, read_only=True, data_only=True)
    try:
        rows = list(wb["ED1"].iter_rows(values_only=True))
    finally:
        wb.close()
    header = rows[4]
    col = next(j for j, h in enumerate(header) if isinstance(h, datetime.datetime) and h.year == YEAR)
    out = []
    for key, item in ED1_LINES.items():
        hits = [(i + 1, r) for i, r in enumerate(rows) if r[3] == item and r[5] == PATHWAY
                and (r[7] == "Peak" if key.endswith("_gw") else str(r[7]).startswith("Annual"))]
        if len(hits) != 1:
            raise ValueError(f"FES 2024 ED1 {item!r}: expected one {PATHWAY} row, found {len(hits)}")
        excel_row, r = hits[0]
        out.append({"key": key, "value": float(r[col]), "unit": r[4], "period": r[7],
                    "data_item": item.strip(), "excel_row": excel_row,
                    "source": f"FES 2024 Data Workbook V006, ED1, {PATHWAY}, {YEAR}"})
    return pd.DataFrame(out)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run(etys_breakdown, gb_network, node_choices, fes24_workbook, named_out, lines_out, etys_peak_out=None,
        log: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log = log or logger
    named = extract_named_loads(etys_breakdown, gb_network, node_choices)
    write_csv(named, named_out)
    log.info("%s: %d rows, %.1f MW at peak; by category (MW): %s", named_out, len(named),
             named["peak_mw_ht_2030_31"].sum(),
             named.groupby("category")["peak_mw_ht_2030_31"].sum().round(1).to_dict())
    lines = extract_ed1_lines(fes24_workbook)
    write_csv(lines, lines_out)
    log.info("%s: %d lines", lines_out, len(lines))
    if etys_peak_out:
        peak = extract_etys_peak_by_gsp(etys_breakdown)
        write_csv(peak, etys_peak_out)
        log.info("%s: %d rows, %.1f MW (%.1f MW unlocated)", etys_peak_out, len(peak), peak["peak_mw"].sum(),
                 peak.loc[peak["latitude"].isna(), "peak_mw"].sum())
    return named, lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--etys-breakdown", default=ETYS_BREAKDOWN)
    p.add_argument("--gb-network", default=GB_NETWORK)
    p.add_argument("--node-choices", default=NODE_CHOICES)
    p.add_argument("--fes24-workbook", default=FES24_WORKBOOK)
    p.add_argument("--named-out", default=NAMED_OUT)
    p.add_argument("--lines-out", default=LINES_OUT)
    p.add_argument("--etys-peak-out", default=ETYS_PEAK_OUT)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(a.etys_breakdown, a.gb_network, a.node_choices, a.fes24_workbook, a.named_out, a.lines_out,
        a.etys_peak_out)
    return 0


if "snakemake" in globals():
    _root = str(Path.cwd())
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from scripts.utilities.logging_config import setup_logging

    _log = setup_logging(snakemake.log[0] if snakemake.log else "extract_cp30_demand_inputs")
    run(snakemake.input.etys_breakdown, snakemake.input.gb_network, snakemake.input.node_choices,
        snakemake.input.fes24_workbook, snakemake.output.named_loads, snakemake.output.national_lines,
        snakemake.output.etys_peak, log=_log)
elif __name__ == "__main__":
    sys.exit(main())
