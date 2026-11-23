"""Clean Power 2030 (NESO, Dec 2024, workbook V003) to the scenario-source contract.

Reads the CSVs the ``CP30_data`` rule extracts from the workbook (``resources/CP30/CP30_ES1.csv``,
``CP30_ES5_regional.csv``, ``CP30_ED1.csv``) through the existing readers, so the numbers are the
ones the CP30 chain uses:

* capacity: ``generation.load_cp30_generation_capacities`` (every ES1 year with capacity: 2023-2030
  and 2035), interconnectors ``storage_interconnectors.load_cp30_interconnector_capacity_mw``;
* energy, ES1 demand and electrolysis: ``energy_validation.cp30_energy_targets`` (2023 and 2030 only:
  ES1 publishes no other year's energy);
* storage: ``storage_interconnectors.load_cp30_storage_targets`` (power and energy) and the ES1
  storage generation of ``cp30_energy_targets``;
* regional capacity: ES5 (``spatial_allocation.read_es5_targets``, ``storage_regions.es5_storage_targets``),
  published for 2030 only;
* demand: the two ES1 lines above, calendar years, and every ED1 row as published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from scripts.sources import registry
from scripts.sources.contract import TABLE_COLUMNS, years_with_values

SOURCE = "CP30"
EDITION = 2024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = {
    "es1": PROJECT_ROOT / "resources" / "CP30" / "CP30_ES1.csv",
    "es5": PROJECT_ROOT / "resources" / "CP30" / "CP30_ES5_regional.csv",
    "ed1": PROJECT_ROOT / "resources" / "CP30" / "CP30_ED1.csv",
}
#: ES5 has no year columns: the workbook's regional breakdown is of 2030.
ES5_YEAR = 2030
#: ES1 demand lines, as items of demand.csv.
ES1_DEMAND_ITEMS = {"demand_excluding_electrolysis": "Demand Total (GWh) - Excluding Electrolysis",
                    "electrolysis": "Electrolysis Demand (GWh)"}


def _frame(name: str, records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=list(TABLE_COLUMNS[name]))


def build_cp30_tables(es1_path: str | Path = DEFAULT_INPUTS["es1"], es5_path: str | Path = DEFAULT_INPUTS["es5"],
                      ed1_path: Optional[str | Path] = DEFAULT_INPUTS["ed1"]) -> tuple[dict[str, pd.DataFrame],
                                                                                         dict[str, Any]]:
    """The CP30 contract tables for every pathway and published year.

    Args:
        es1_path: CP30 ES1 CSV.
        es5_path: CP30 ES5 regional CSV.
        ed1_path: CP30 ED1 CSV (None: no ED1 rows in demand.csv).

    Returns:
        (tables, meta): contract tables by name, and provenance for ``availability.json``.

    Raises:
        UnknownPathway: ES1 names a pathway the registry does not know.
    """
    from scripts.CP30.energy_validation import DEMAND_VARIABLE, cp30_energy_targets
    from scripts.CP30.generation import load_cp30_generation_capacities
    from scripts.CP30.spatial_allocation import read_es5_targets
    from scripts.CP30.storage_interconnectors import (load_cp30_interconnector_capacity_mw,
                                                      load_cp30_storage_targets)
    from scripts.CP30.storage_regions import es5_storage_targets

    if ES1_DEMAND_ITEMS["demand_excluding_electrolysis"] != DEMAND_VARIABLE:
        raise ValueError(f"CP30 ES1 demand line is {DEMAND_VARIABLE!r} in energy_validation.py")
    es1 = pd.read_csv(es1_path)
    year_columns = [c for c in es1.columns if str(c).isdigit()]
    for name in es1["Pathway"].dropna().unique():
        registry.pathway_by_name(SOURCE, EDITION, name)   # an unknown pathway fails here
    es5 = pd.read_csv(es5_path)
    for name in es5["Scenario"].dropna().unique():
        registry.pathway_by_name(SOURCE, EDITION, name)
    ed1 = pd.read_csv(ed1_path) if ed1_path is not None else None

    capacity, energy, storage, demand, regional = [], [], [], [], []
    unmapped: list[dict[str, Any]] = []
    for pathway in registry.pathways(SOURCE, EDITION):
        rows = es1[es1["Pathway"].eq(pathway.name)]
        if rows.empty:
            raise ValueError(f"CP30 ES1 has no rows for registered pathway {pathway.name!r}")
        key = {"pathway_code": pathway.code, "pathway": pathway.name}
        capacity_years = years_with_values(rows[rows["Variable"].eq("Capacity (MW)")], year_columns)
        energy_years = years_with_values(rows[rows["Variable"].eq("Generation (GWh)")], year_columns)

        storage_by_year: dict[tuple[int, str], dict[str, Any]] = {}
        for year in capacity_years:
            gen = load_cp30_generation_capacities(es1_path, pathway.name, year)
            for _, row in gen.iterrows():
                capacity.append({**key, "year": year, "carrier": row["carrier"],
                                 "capacity_mw": float(row["capacity_mw"]), "lines": row["cp30_subtype"]})
            capacity.append({**key, "year": year, "carrier": "interconnector",
                             "capacity_mw": load_cp30_interconnector_capacity_mw(es1_path, pathway.name, year),
                             "lines": "Interconnectors"})
            for _, row in load_cp30_storage_targets(es1_path, pathway.name, year).iterrows():
                ratio = row.get("cp30_implied_round_trip", np.nan)
                storage_by_year[(year, row["carrier"])] = {
                    **key, "year": year, "carrier": row["carrier"], "power_mw": float(row["power_mw"]),
                    "energy_gwh": float(row["energy_gwh"]), "generation_gwh": np.nan,
                    "consumption_gwh": np.nan, "lines": row["cp30_subtype"], "_ratio": ratio}

        for year in energy_years:
            targets = cp30_energy_targets(es1_path, pathway.name, year)
            for carrier, row in targets["generation"].iterrows():
                energy.append({**key, "year": year, "carrier": carrier,
                               "generation_gwh": float(row["generation_gwh"]),
                               "curtailment_gwh": float(row["curtailment_gwh"]), "gross_gwh": float(row["gross_gwh"]),
                               "import_gwh": np.nan, "export_gwh": np.nan, "lines": row["es1_lines"]})
            energy.append({**key, "year": year, "carrier": "interconnector", "generation_gwh": np.nan,
                           "curtailment_gwh": np.nan, "gross_gwh": np.nan, "import_gwh": targets["import_gwh"],
                           "export_gwh": targets["export_gwh"], "lines": "Interconnectors"})
            for carrier, gwh in targets["storage_gwh"].items():
                entry = storage_by_year.setdefault((year, carrier), {
                    **key, "year": year, "carrier": carrier, "power_mw": np.nan, "energy_gwh": np.nan,
                    "generation_gwh": np.nan, "consumption_gwh": np.nan, "lines": "", "_ratio": np.nan})
                entry["generation_gwh"] = float(gwh)
                ratio = entry["_ratio"]
                entry["consumption_gwh"] = float(gwh) / ratio if ratio and np.isfinite(ratio) else np.nan
            for item, line in ES1_DEMAND_ITEMS.items():
                value = targets["demand_gwh"] if item == "demand_excluding_electrolysis" else targets["electrolysis_gwh"]
                demand.append({**key, "year": year, "item": item, "unit": "GWh", "value": float(value),
                               "year_basis": "calendar", "fiscal_value": np.nan, "fiscal_previous_value": np.nan,
                               "line": f"ES1 {line}"})
            for _, row in targets["unmapped"].iterrows():
                unmapped.append({"pathway_code": pathway.code, "year": year,
                                 **{c: str(row.get(c, "")) for c in ("Connection", "Variable", "Category", "Type",
                                                                     "SubType")}})
        storage.extend({k: v for k, v in entry.items() if k != "_ratio"} for entry in storage_by_year.values())

        if ed1 is not None:
            ed1_rows = ed1[ed1["Pathway"].eq(pathway.name)]
            for year in [c for c in ed1_rows.columns if str(c).isdigit()]:
                for _, row in ed1_rows.iterrows():
                    value = pd.to_numeric(row[year], errors="coerce")
                    if pd.isna(value):
                        continue
                    demand.append({**key, "year": int(year), "item": str(row["Data item"]).strip(),
                                   "unit": str(row["Unit"]).strip(), "value": float(value),
                                   "year_basis": "as_published", "fiscal_value": np.nan,
                                   "fiscal_previous_value": np.nan,
                                   "line": f"ED1 {str(row['Annual/ Peak/ Flex']).strip()}"})

        if pathway.name in set(es5["Scenario"]):
            for _, row in read_es5_targets(es5, pathway.name, ES5_YEAR).iterrows():
                regional.append({**key, "year": ES5_YEAR, "scheme": "ES5", "region": row["cp30_region"],
                                 "carrier": row["carrier"], "capacity_mw": float(row["capacity_mw"])})
            for _, row in es5_storage_targets(es5, pathway.name).iterrows():
                regional.append({**key, "year": ES5_YEAR, "scheme": "ES5", "region": row["cp30_region"],
                                 "carrier": row["carrier"], "capacity_mw": float(row["capacity_mw"])})

    tables = {"capacity": _frame("capacity", capacity), "energy": _frame("energy", energy),
              "storage": _frame("storage", storage), "demand": _frame("demand", demand),
              "regional_capacity": _frame("regional_capacity", regional)}
    meta = {
        "inputs": {"es1": str(es1_path), "es5": str(es5_path), "ed1": str(ed1_path) if ed1_path else None},
        "rules": {"units": "capacity MW, energy GWh (ES1 curtailment TWh x 1,000)",
                  "demand": "ES1 lines calendar; ED1 rows as published",
                  "regional_capacity": f"ES5 is the {ES5_YEAR} breakdown; no other year is published"},
        "unmapped_es1_lines": unmapped,
    }
    return tables, meta
