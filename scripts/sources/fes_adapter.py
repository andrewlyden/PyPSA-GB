"""FES editions (2020-2025) to the scenario-source contract.

What each edition gives, and from where:

* **gsp_blocks** (every edition): the building blocks, GSP names cleaned by the edition's alias rows
  and mapped to GSP and node ids (``scripts/FES/FES_data.py``), one row per block, GSP and year as
  published (the Dem_BB energy blocks are fiscal years). Read from ``FES_{edition}_data.csv`` when given,
  else built from the raw cache.
* **gsp_info** (2021-2025): the committed table ``data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv``.
* **capacity, energy, storage** (editions with an ES1 line map in
  ``data/scenario_sources/es1_line_maps.csv``: 2024 and 2025): ES1 classified by
  ``fes2025_targets.classify_es1`` with the edition's map, summed by ``supply_tables``. Every year with
  ES1 values.
* **demand**: FES 2025 from ED1 (``fes2025_targets.ed1_demand``: fiscal published, calendar = 0.75 FY(Y)
  + 0.25 FY(Y-1)); FES 2024 from its ES1 demand lines (calendar; its ED1 has date headers this reader
  does not take). Other editions: none.
* **regional_capacity**: not from FES yet (an ES5-equivalent from the building blocks is stage 7).

The 2020-2023 workbooks lay ES1 out by Scenario with other categories; without a line map they give
no ES1 tables rather than a guessed mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from scripts.sources import registry
from scripts.sources.contract import TABLE_COLUMNS, years_with_values

SOURCE = "FES"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKBOOKS = {
    2020: PROJECT_ROOT / "data" / "FES" / "Data-workbook2020_v1_4.xlsx",
    2021: PROJECT_ROOT / "data" / "FES" / "FES 2021 Data Workbook V08.xlsx",
    2022: PROJECT_ROOT / "data" / "FES" / "Data-workbook2022_V006 (1).xlsx",
    2023: PROJECT_ROOT / "data" / "FES" / "FES 2023 Data Workbook V003.xlsx",
    2024: PROJECT_ROOT / "data" / "FES" / "Future Energy Scenarios 2024 Data Workbook_V006_0.xlsx",
    2025: PROJECT_ROOT / "data" / "FES" / "Future Energy Scenarios 2025 Data Workbook V005_0.xlsx",
}
#: Where each edition's demand comes from ("ED1", "ES1"); an edition not listed publishes none here.
DEMAND_SOURCE = {2024: "ES1", 2025: "ED1"}
#: ES1 demand lines (Type) of the editions whose demand comes from ES1, as demand.csv items.
ES1_DEMAND_TYPES = {
    2024: {"Demand Total (TWh) - Excluding Electrolysis": "demand_excluding_electrolysis",
           "Demand Total (TWh) - Excluding Electrolysis, Post-DSR": "demand_excluding_electrolysis_post_dsr",
           "Electrolysis Demand (TWh)": "electrolysis"},
}
#: The ED1 totals written as demand.csv items (``fes2025_targets.ed1_demand`` ``totals``).
ED1_TOTALS = ("system_excluding_electrolysis", "consumer_excluding_electrolysis", "losses", "electrolysis",
              "heat_pumps_residential", "evs")


def _frame(name: str, records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=list(TABLE_COLUMNS[name]))


def _resolve_names(edition: int, names, where: str) -> dict[str, registry.Pathway]:
    """Registry pathway of every pathway string in a table; an unknown string fails."""
    out = {}
    for name in sorted({str(n) for n in names if pd.notna(n) and str(n).strip()}):
        try:
            out[name] = registry.pathway_by_name(SOURCE, edition, name)
        except registry.UnknownPathway as exc:
            raise registry.UnknownPathway(f"{where}: {exc}") from None
    return out


# ───────────────────────────────────── ES1 ──────────────────────────────────

def es1_tables(edition: int, workbook: str | Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Capacity, energy and storage records of every pathway and ES1 year, with the unmapped lines."""
    from scripts.FES.fes2025_targets import classify_es1, edition_line_maps, read_es1, supply_tables

    generation_map, storage_map = edition_line_maps(edition)
    es1 = read_es1(workbook)
    year_columns = [c for c in es1.columns if str(c).isdigit()]
    pathways = _resolve_names(edition, es1["Pathway"].map(lambda v: str(v).strip() if pd.notna(v) else v),
                              f"FES {edition} ES1")
    records: dict[str, list[dict[str, Any]]] = {"capacity": [], "energy": [], "storage": []}
    unmapped, excluded = [], []
    for printed, pathway in pathways.items():
        key = {"pathway_code": pathway.code, "pathway": pathway.name}
        rows = es1[es1["Pathway"].map(lambda v: str(v).strip()).eq(printed)]
        for year in years_with_values(rows, year_columns):
            classified = classify_es1(es1, printed, year, generation_map=generation_map, storage_map=storage_map)
            if classified.empty:
                continue
            tables = supply_tables(classified)
            gen, ic, storage = tables["generation"], tables["ic"], tables["storage"]
            for carrier, row in gen.iterrows():
                if row["capacity_mw"] != 0.0:
                    records["capacity"].append({**key, "year": year, "carrier": carrier,
                                                "capacity_mw": float(row["capacity_mw"]), "lines": row["es1_lines"]})
                if row["generation_gwh"] != 0.0 or row["curtailment_gwh"] != 0.0:
                    records["energy"].append({**key, "year": year, "carrier": carrier,
                                              "generation_gwh": float(row["generation_gwh"]),
                                              "curtailment_gwh": float(row["curtailment_gwh"]),
                                              "gross_gwh": float(row["gross_gwh"]), "import_gwh": np.nan,
                                              "export_gwh": np.nan, "lines": row["es1_lines"]})
            if "capacity_mw" in ic:
                records["capacity"].append({**key, "year": year, "carrier": "interconnector",
                                            "capacity_mw": float(ic["capacity_mw"]), "lines": "Interconnectors"})
            if "import_gwh" in ic or "export_gwh" in ic:
                records["energy"].append({**key, "year": year, "carrier": "interconnector", "generation_gwh": np.nan,
                                          "curtailment_gwh": np.nan, "gross_gwh": np.nan,
                                          "import_gwh": float(ic.get("import_gwh", 0.0)),
                                          "export_gwh": float(ic.get("export_gwh", 0.0)), "lines": "Interconnectors"})
            for carrier, row in storage.iterrows():
                records["storage"].append({**key, "year": year, "carrier": carrier, "power_mw": float(row["capacity_mw"]),
                                           "energy_gwh": float(row["energy_gwh"]),
                                           "generation_gwh": float(row["generation_gwh"]),
                                           "consumption_gwh": float(row["consumption_gwh"]), "lines": "Storage"})
            for role, sink in (("unmapped", unmapped), ("excluded", excluded)):
                for _, line in classified[classified["role"].eq(role)].iterrows():
                    sink.append({"pathway_code": pathway.code, "year": year, "excel_row": int(line["excel_row"]),
                                 "connection": line["connection"], "variable": line["variable"],
                                 "category": line["category"], "type": line["type"], "subtype": line["subtype"],
                                 "value": float(line["value"])})
    return records, {"unmapped_es1_lines": unmapped, "excluded_es1_lines": excluded}


# ──────────────────────────────────── demand ────────────────────────────────

def ed1_demand_records(edition: int, workbook: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """ED1 demand items (calendar value, fiscal kept) of every pathway and year ED1 allows."""
    from scripts.FES.fes2025_targets import ed1_demand, read_ed1

    ed1 = read_ed1(workbook)
    year_columns = [c for c in ed1.columns if str(c).isdigit()]
    pathways = _resolve_names(edition, ed1["Pathway"].map(lambda v: str(v).strip() if pd.notna(v) else v),
                              f"FES {edition} ED1")
    records, skipped = [], []
    for printed, pathway in pathways.items():
        key = {"pathway_code": pathway.code, "pathway": pathway.name}
        rows = ed1[ed1["Pathway"].map(lambda v: str(v).strip()).eq(printed)]
        years = years_with_values(rows, year_columns)
        for year in years:
            if year - 1 not in years:
                continue   # the calendar value needs the previous fiscal year
            try:
                demand = ed1_demand(ed1, printed, year)
            except ValueError as exc:
                skipped.append(f"{pathway.code} {year}: {exc}")
                continue
            for item in ED1_TOTALS:
                total = demand["totals"][item]
                records.append({**key, "year": year, "item": item, "unit": "GWh", "value": total["calendar_gwh"],
                                "year_basis": "calendar", "fiscal_value": total["fy_gwh"],
                                "fiscal_previous_value": np.nan, "line": "ED1 (derived)"})
            for item, entry in demand["annual"].items():
                records.append({**key, "year": year, "item": item, "unit": "GWh", "value": entry["calendar_gwh"],
                                "year_basis": "calendar", "fiscal_value": entry["fy_gwh"],
                                "fiscal_previous_value": entry["fy_previous_gwh"],
                                "line": f"ED1 row {entry['excel_row']}: {entry['label']}"})
            for label, entry in demand["peak_gw"].items():
                records.append({**key, "year": year, "item": f"peak: {label}", "unit": "GW", "value": entry["gw"],
                                "year_basis": "as_published", "fiscal_value": np.nan, "fiscal_previous_value": np.nan,
                                "line": f"ED1 row {entry['excel_row']}"})
    return records, skipped


def es1_demand_records(edition: int, workbook: str | Path) -> list[dict[str, Any]]:
    """ES1 demand lines (calendar) of every pathway and year."""
    from scripts.FES.fes2025_targets import _norm, read_es1, split_variable, to_gwh

    types = ES1_DEMAND_TYPES[edition]
    es1 = read_es1(workbook)
    year_columns = [c for c in es1.columns if str(c).isdigit()]
    pathways = _resolve_names(edition, es1["Pathway"].map(lambda v: str(v).strip() if pd.notna(v) else v),
                              f"FES {edition} ES1")
    lines = es1[es1["Type"].map(_norm).isin(types)]
    records = []
    for _, row in lines.iterrows():
        pathway = pathways[_norm(row["Pathway"])]
        _, unit = split_variable(row["Variable"])
        for year in year_columns:
            value = pd.to_numeric(row[year], errors="coerce")
            if pd.isna(value):
                continue
            records.append({"pathway_code": pathway.code, "pathway": pathway.name, "year": int(year),
                            "item": types[_norm(row["Type"])], "unit": "GWh", "value": to_gwh(value, unit),
                            "year_basis": "calendar", "fiscal_value": np.nan, "fiscal_previous_value": np.nan,
                            "line": f"ES1 row {int(row['excel_row'])}: {_norm(row['Type'])}"})
    return records


# ───────────────────────────────── building blocks ──────────────────────────

def normalised_building_blocks(edition: int, fes_data: Optional[str | Path] = None) -> pd.DataFrame:
    """FES_{edition}_data.csv as written by ``FES_data.py`` (read, or built from the raw cache)."""
    if fes_data is not None:
        return pd.read_csv(fes_data, low_memory=False)
    from scripts.FES import FES_data
    merged = FES_data.raw_FES_data(edition)
    gsp_path = FES_data.GSP_INFO_DIR / f"fes{edition}_regional_breakdown_gsp_info.csv"
    if gsp_path.is_file():
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"FES_{edition}_data.csv"
            FES_data.add_node_id_to_gsp(edition, merged, outpath=out)
            return pd.read_csv(out, low_memory=False)
    # No GSP info for the edition (FES 2020): names cleaned, no ids.
    blocks = registry.apply_gsp_aliases(merged, edition, column="GSP").reset_index()
    blocks["GSP ID"], blocks["Node ID"] = np.nan, np.nan
    return blocks


def _first_column(frame: pd.DataFrame, *candidates: str) -> Optional[str]:
    for name in candidates:
        for column in frame.columns:
            if str(column).replace("ï»¿", "").replace("﻿", "").strip() == name:
                return column
    return None


def gsp_block_records(edition: int, blocks: pd.DataFrame) -> pd.DataFrame:
    """The building blocks long: one row per block, GSP and year column."""
    pathway_col = _first_column(blocks, "FES Pathway", "FES Scenario")
    if pathway_col is None:
        raise ValueError(f"FES {edition} building blocks have no pathway column: {list(blocks.columns)[:8]}")
    pathways = _resolve_names(edition, blocks[pathway_col], f"FES {edition} building blocks")
    codes = blocks[pathway_col].map(lambda v: pathways[str(v)].code if pd.notna(v) and str(v).strip() else np.nan)
    year_columns = [c for c in blocks.columns if str(c).isdigit()]
    detail_col = _first_column(blocks, "Technology Detail")
    frame = pd.DataFrame({
        "pathway_code": codes, "block_id": blocks["Building Block ID Number"],
        "unit": blocks[_first_column(blocks, "Unit")], "gsp": blocks["GSP"], "gsp_id": blocks["GSP ID"],
        "node_id": blocks["Node ID"], "dno_area": blocks[_first_column(blocks, "DNO License Area")],
        "technology": blocks[_first_column(blocks, "Technology")],
        "technology_detail": blocks[detail_col] if detail_col else np.nan,
    })
    long = pd.concat([frame.assign(year=int(y), value=pd.to_numeric(blocks[y], errors="coerce")) for y in year_columns],
                     ignore_index=True)
    return long[long["value"].notna() & long["pathway_code"].notna()]


def gsp_info_table(edition: int) -> Optional[pd.DataFrame]:
    """The edition's committed GSP info in contract columns, or None when there is none."""
    from scripts.FES.gsp_metadata import metadata_path_for_year
    path = metadata_path_for_year(edition, PROJECT_ROOT / "data" / "network" / "ETYS")
    if not path.is_file():
        return None
    info = pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame({"gsp_id": info["GSP ID"], "gsp_group": info["GSP Group"], "minor_flop": info["Minor FLOP"],
                         "name": info["Name"], "latitude": info["Latitude"], "longitude": info["Longitude"],
                         "comments": info["Comments"]})


# ───────────────────────────────────── build ────────────────────────────────

def build_fes_tables(edition: int, workbook: Optional[str | Path] = None, fes_data: Optional[str | Path] = None,
                     include_gsp_blocks: bool = True) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """The contract tables of one FES edition.

    Args:
        edition: FES edition (2020-2025).
        workbook: The edition's Data Workbook; default :data:`WORKBOOKS`.
        fes_data: ``FES_{edition}_data.csv``; default built from the raw cache.
        include_gsp_blocks: Write gsp_blocks (the largest table).

    Returns:
        (tables, meta) for :func:`scripts.sources.contract.write_source`.
    """
    edition = int(edition)
    registry.pathways(SOURCE, edition)   # an unknown edition fails here
    if workbook is None:
        if edition not in WORKBOOKS:
            raise KeyError(f"No FES {edition} Data Workbook registered; known {sorted(WORKBOOKS)}")
        workbook = WORKBOOKS[edition]
    tables: dict[str, pd.DataFrame] = {}
    meta: dict[str, Any] = {"inputs": {"workbook": str(workbook), "fes_data": str(fes_data) if fes_data else None},
                            "rules": {"units": "capacity MW, energy GWh (ES1 TWh x 1,000); exports positive",
                                      "gsp_blocks": "as published; Dem_BB energy blocks are fiscal years"}}
    try:
        registry.es1_line_map(SOURCE, edition, "supply")
        has_line_map = True
    except registry.UnknownEdition:
        has_line_map = False
    if has_line_map:
        records, notes = es1_tables(edition, workbook)
        for name, rows in records.items():
            tables[name] = _frame(name, rows)
        meta.update(notes)
    else:
        meta["es1"] = f"no ES1 line map for FES {edition}: capacity, energy and storage not published"

    demand_source = DEMAND_SOURCE.get(edition)
    if demand_source == "ED1":
        records, skipped = ed1_demand_records(edition, workbook)
        tables["demand"] = _frame("demand", records)
        meta["demand_skipped"] = skipped
        meta["rules"]["demand"] = "ED1 fiscal; calendar Y = 0.75 FY(Y) + 0.25 FY(Y-1)"
    elif demand_source == "ES1":
        tables["demand"] = _frame("demand", es1_demand_records(edition, workbook))
        meta["rules"]["demand"] = "ES1 demand lines, calendar"
    else:
        meta["rules"]["demand"] = f"no demand reader for FES {edition}"

    if include_gsp_blocks:
        tables["gsp_blocks"] = gsp_block_records(edition, normalised_building_blocks(edition, fes_data))
    info = gsp_info_table(edition)
    if info is not None:
        tables["gsp_info"] = info
    return tables, meta
