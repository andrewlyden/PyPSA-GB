"""FES 2025 Data Workbook energy targets for a pathway and year (task Y1).

The 2030 run validates its energy against CP30 ES1 (``scripts/CP30/energy_validation.py``).
This module gives the 2050 run the same thing against the FES 2025 Data Workbook
(``data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx``), and also returns
the demand components and hydrogen quantities the long-range demand build uses.

Layout differences from CP30 ES1 (each handled here, not by the caller):

* **Units.** FES 2025 ES1 energy is in TWh ("Generation (TWh)"), CP30's was GWh. The unit
  is parsed from each Variable label, so a label in another unit converts correctly and an
  unknown unit fails.
* **No demand row in ES1.** Demand comes from ED1, which is on fiscal years (ED1 column
  "Peak/ Annual/ Minimum" = "Annual [Fiscal]"). See :func:`fiscal_to_calendar`.
* **Categories.** ES1 groups supply as Base Load / Dispatchable / Renewable, and
  names technologies differently ("Nuclear - Small", "Hydrogen Peaking", "Gas CCGT CHP",
  "CCS Gas" as its own Type). :data:`FES2025_GENERATION_MAP` maps every line.
* **Off-grid supply.** The Non-Networked offshore wind line (0.44 GW, 1.68 TWh in HT
  2050) feeds non-networked electrolysis and is declared excluded, not dropped silently.

Every non-zero ES1 line is classified. A supply or storage line with no mapping, or a line
of a variable/category this module does not recognise, is returned in ``unmapped`` and makes
``strict=True`` fail.

Year bases:

* ES1 generation and capacity: calendar year (ES3 rows 5 to 8; capacity at 1 December).
* ED1, the Dem_BB building blocks and WS1 hydrogen energy: fiscal year, column "Y" =
  April Y to March Y+1 (the FES "2018 = 2018/19" convention stated on the FES 2024 CP1
  sheet). Evidence that WS1 shares the ED1 basis: its networked-electrolysis electricity
  feedstock (WS1 row 593) equals ED1 row 74 exactly in every year.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from scripts.sources import registry as _registry


FES2025_WORKBOOK = "data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx"
SOURCE_ID = "FES2025"
SOURCE_LABEL = "FES 2025"

# Pathway names as ES1/ED1 print them, keyed by the slug used in file names
# (data/scenario_sources/pathways.csv, FES 2025 rows).
FES2025_PATHWAYS = _registry.pathway_names("FES", 2025)

# Fiscal-to-calendar rule (declared): calendar Y = 9/12 FY(Y) + 3/12 FY(Y-1), where FY(Y) is
# April Y to March Y+1. A month-count weighting, not an energy weighting: January to March
# carries more than a quarter of annual demand, but between adjacent years of a smooth
# pathway the difference is second order (HT 2050: 704.10 TWh by this rule against 705.22 FY).
FISCAL_WEIGHT_CURRENT = 0.75
FISCAL_WEIGHT_PREVIOUS = 0.25

_UNIT_TO_GWH = {"TWh": 1e3, "GWh": 1.0, "MWh": 1e-3}
_UNIT_TO_MW = {"GW": 1e3, "MW": 1.0}


# ───────────────────────────────── mapping ──────────────────────────────────

@dataclass(frozen=True)
class FESSupplyMapping:
    """One FES 2025 ES1 supply line (Category, Type, SubType) and its model carrier."""

    category: str
    type: str
    subtype: str
    carrier: str
    note: str = ""


# ES1 supply and storage lines of FES 2025 (data/scenario_sources/es1_line_maps.csv). Carrier names
# follow CP30_GENERATION_MAP (scripts/CP30/generation.py), so the energy validation treats both sources
# alike: unabated gas is "gas" (CCGT + OCGT + gas engines), gas CHP is "CHP", every hydrogen power line is
# "Hydrogen". Storage lines are keyed on (Type, SubType) and carry the model's storage carriers
# (scripts/storage/add_storage.py; scripts/utilities/carrier_definitions.py).
FES2025_GENERATION_MAP: tuple[FESSupplyMapping, ...] = tuple(
    FESSupplyMapping(line.category, line.type, line.subtype, line.carrier, line.note)
    for line in _registry.es1_line_map("FES", 2025, "supply"))

FES2025_STORAGE_MAP: dict[tuple[str, str], str] = _registry.es1_carrier_lookup("FES", 2025, "storage")

# Connections whose supply is declared outside the grid model.
FES2025_EXCLUDED_CONNECTIONS = {
    "Non-Networked": "off-grid supply feeding non-networked electrolysis (ES1 row 246/595, WS1 row 598); "
                     "not put on the grid",
}

# ES1 variables per role.
_SUPPLY_VARIABLES = {"capacity": "Capacity", "generation": "Generation", "curtailment": "Curtailment"}
_STORAGE_VARIABLES = {"Capacity": "capacity_mw", "Generation": "generation_gwh",
                      "Consumption": "consumption_gwh", "Storage Capacity": "energy_gwh"}
_INTERCONNECTOR_VARIABLES = {"Capacity": "capacity_mw", "Import": "import_gwh",
                             "Export": "export_gwh", "Netflows": "netflow_gwh"}
# Non-supply lines recognised and reported, never mapped to a carrier.
_INFORMATION_LINES = {
    ("Curtailment including DACCS", "Renewable"): "gross curtailment less DACCS demand (ES1 row 606)",
    ("CO2 Intensity of Generation", "Emissions"): "emissions intensity",
    ("CO2 Intensity of Generation excluding BECCS", "Emissions"): "emissions intensity",
    ("CO2 Emissions", "Emissions"): "emissions",
    ("CO2 Emissions without BECCS", "Emissions"): "emissions",
}


def _norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def split_variable(variable: str) -> tuple[str, str]:
    """Split an ES1 Variable label into (name, unit): 'Generation (TWh)' -> ('Generation', 'TWh')."""
    match = re.match(r"^\s*(.*?)\s*\(([^()]*)\)\s*$", str(variable))
    if not match:
        return str(variable).strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def to_gwh(value: float, unit: str) -> float:
    """Convert an energy value in ``unit`` (TWh, GWh or MWh) to GWh; unknown units fail."""
    if unit not in _UNIT_TO_GWH:
        raise ValueError(f"Unknown energy unit {unit!r}; expected one of {sorted(_UNIT_TO_GWH)}")
    return float(value) * _UNIT_TO_GWH[unit]


def to_mw(value: float, unit: str) -> float:
    """Convert a power value in ``unit`` (GW or MW) to MW; unknown units fail."""
    if unit not in _UNIT_TO_MW:
        raise ValueError(f"Unknown power unit {unit!r}; expected one of {sorted(_UNIT_TO_MW)}")
    return float(value) * _UNIT_TO_MW[unit]


def _generation_lookup() -> dict[tuple[str, str, str], FESSupplyMapping]:
    return {(m.category, m.type, m.subtype): m for m in FES2025_GENERATION_MAP}


def mapping_for_fes_row(row: pd.Series,
                        lookup: Optional[dict[tuple[str, str, str], FESSupplyMapping]] = None) -> Optional[FESSupplyMapping]:
    """Supply mapping for an ES1 row, or None (``lookup``: default the FES 2025 map)."""
    lookup = _generation_lookup() if lookup is None else lookup
    return lookup.get((_norm(row["Category"]), _norm(row["Type"]), _norm(row["SubType"])))


def edition_line_maps(edition: int) -> tuple[tuple[FESSupplyMapping, ...], dict[tuple[str, str], str]]:
    """The ES1 supply and storage maps of a FES edition (data/scenario_sources/es1_line_maps.csv).

    Raises:
        KeyError: The edition has no ES1 line map.
    """
    supply = tuple(FESSupplyMapping(line.category, line.type, line.subtype, line.carrier, line.note)
                   for line in _registry.es1_line_map("FES", edition, "supply"))
    return supply, _registry.es1_carrier_lookup("FES", edition, "storage")


# ─────────────────────────────── workbook I/O ───────────────────────────────

def _is_year(value: Any) -> bool:
    try:
        year = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return False
    return 1990 <= year <= 2100 and float(str(value).strip()) == year


def read_sheet_table(workbook: str | Path, sheet: str, required: Iterable[str],
                     filter_column: Optional[str] = None, filter_values: Optional[Iterable[str]] = None,
                     max_header_row: int = 60) -> pd.DataFrame:
    """Read a FES data table: find the header row holding ``required`` labels, keep rows below it.

    Year columns are renamed to their year as a string ("2050"); every row keeps its Excel row
    number as ``excel_row``. ``filter_column``/``filter_values`` keep only matching rows while
    streaming (the building-block sheet has 48,000 rows).

    Args:
        workbook: Path to the FES Data Workbook.
        sheet: Sheet name, e.g. "ES1".
        required: Header labels that identify the header row.
        filter_column: Optional header label to filter on while reading.
        filter_values: Values of ``filter_column`` to keep.
        max_header_row: Rows searched for the header.

    Returns:
        The table, one row per data row, with ``excel_row`` and string year columns.
    """
    import openpyxl

    required = [str(r) for r in required]
    keep = {str(v) for v in filter_values} if filter_values is not None else None
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        if sheet not in wb.sheetnames:
            raise ValueError(f"Sheet {sheet!r} not in {workbook}")
        header: Optional[list[str]] = None
        columns: list[tuple[int, str]] = []
        filter_idx: Optional[int] = None
        records: list[dict[str, Any]] = []
        for excel_row, values in enumerate(wb[sheet].iter_rows(values_only=True), start=1):
            if header is None:
                labels = [_norm(v) for v in values]
                if all(r in labels for r in required):
                    header = labels
                    for idx, label in enumerate(labels):
                        if not label:
                            continue
                        name = str(int(float(label))) if _is_year(label) else label
                        columns.append((idx, name))
                    if filter_column is not None:
                        filter_idx = labels.index(filter_column)
                elif excel_row >= max_header_row:
                    break
                continue
            if filter_idx is not None and keep is not None:
                if filter_idx >= len(values) or _norm(values[filter_idx]) not in keep:
                    continue
            record = {name: (values[idx] if idx < len(values) else None) for idx, name in columns}
            if all(v is None for v in record.values()):
                continue
            record["excel_row"] = excel_row
            records.append(record)
    finally:
        wb.close()
    if header is None:
        raise ValueError(f"No header row with {required} in {workbook} sheet {sheet!r}")
    return pd.DataFrame.from_records(records)


def _year_value(rows: pd.DataFrame, year: int) -> pd.Series:
    col = str(year)
    if col not in rows.columns:
        raise ValueError(f"Year {year} not in table columns")
    return pd.to_numeric(rows[col], errors="coerce")


def fiscal_to_calendar(values_by_fiscal_year: dict[int, float], year: int) -> float:
    """Calendar-year value from fiscal-year values by the declared month-count rule.

    ``calendar(Y) = 0.75 FY(Y) + 0.25 FY(Y-1)``, with FY(Y) = April Y to March Y+1.

    Args:
        values_by_fiscal_year: Fiscal-year label -> value; must hold ``year`` and ``year - 1``.
        year: Calendar year.

    Returns:
        The calendar-year value.
    """
    if year not in values_by_fiscal_year or (year - 1) not in values_by_fiscal_year:
        raise ValueError(f"Fiscal to calendar {year} needs fiscal years {year - 1} and {year}")
    return (FISCAL_WEIGHT_CURRENT * float(values_by_fiscal_year[year])
            + FISCAL_WEIGHT_PREVIOUS * float(values_by_fiscal_year[year - 1]))


# ───────────────────────────────────── ES1 ──────────────────────────────────

def read_es1(workbook: str | Path = FES2025_WORKBOOK) -> pd.DataFrame:
    """ES1 (electricity supply) as a table with ``excel_row``; header on row 9."""
    return read_sheet_table(workbook, "ES1", ("Connection", "Pathway", "Variable", "Category", "Type", "SubType"))


def classify_es1(es1: pd.DataFrame, pathway: str, year: int,
                 generation_map: Optional[Iterable[FESSupplyMapping]] = None,
                 storage_map: Optional[dict[tuple[str, str], str]] = None) -> pd.DataFrame:
    """Classify every non-zero ES1 line of a pathway and year.

    Args:
        es1: ES1 table from :func:`read_es1`.
        pathway: FES pathway, e.g. "Holistic Transition".
        year: Year column.
        generation_map: Supply lines to carriers; default :data:`FES2025_GENERATION_MAP`.
        storage_map: (Type, SubType) of storage lines to carriers; default :data:`FES2025_STORAGE_MAP`.

    Returns:
        One row per non-zero line with ``role`` (supply, excluded, storage, interconnector,
        daccs, dsr, information, unmapped), ``carrier``, ``quantity`` (capacity_mw,
        generation_gwh, ...), ``value``, ``unit`` and ``value_std`` (MW or GWh).
    """
    lookup = (_generation_lookup() if generation_map is None
              else {(m.category, m.type, m.subtype): m for m in generation_map})
    storage_map = FES2025_STORAGE_MAP if storage_map is None else storage_map
    rows = es1[es1["Pathway"].map(_norm).eq(pathway)].copy()
    if rows.empty:
        raise ValueError(f"No ES1 rows for pathway {pathway!r}; available: {sorted(es1['Pathway'].dropna().unique())}")
    rows["value"] = _year_value(rows, year)
    rows = rows[rows["value"].notna() & rows["value"].ne(0.0)]
    records = []
    for _, row in rows.iterrows():
        name, unit = split_variable(row["Variable"])
        category, type_, subtype, connection = (_norm(row["Category"]), _norm(row["Type"]),
                                                _norm(row["SubType"]), _norm(row["Connection"]))
        record = {"excel_row": int(row["excel_row"]), "connection": connection, "variable": _norm(row["Variable"]),
                  "category": category, "type": type_, "subtype": subtype, "value": float(row["value"]),
                  "unit": unit, "role": "unmapped", "carrier": "", "quantity": "", "value_std": np.nan, "note": ""}
        is_power = unit in _UNIT_TO_MW
        if category not in {"Storage", "Interconnectors", "Demand", "Emissions"} and name in {"Capacity", "Generation", "Curtailment"}:
            mapping = mapping_for_fes_row(row, lookup)
            quantity = {"Capacity": "capacity_mw", "Generation": "generation_gwh", "Curtailment": "curtailment_gwh"}[name]
            if connection in FES2025_EXCLUDED_CONNECTIONS:
                record.update(role="excluded", carrier=mapping.carrier if mapping else "", quantity=quantity,
                              note=FES2025_EXCLUDED_CONNECTIONS[connection])
            elif mapping is not None:
                record.update(role="supply", carrier=mapping.carrier, quantity=quantity, note=mapping.note)
        elif category == "Storage" and name in _STORAGE_VARIABLES:
            carrier = storage_map.get((type_, subtype))
            if carrier is not None:
                record.update(role="storage", carrier=carrier, quantity=_STORAGE_VARIABLES[name])
        elif category == "Interconnectors" and name in _INTERCONNECTOR_VARIABLES:
            record.update(role="interconnector", carrier="interconnector", quantity=_INTERCONNECTOR_VARIABLES[name])
        elif category == "Demand" and type_ == "DACCS" and name == "Demand":
            record.update(role="daccs", carrier="DACCS", quantity="demand_gwh",
                          note="DACCS electrical demand; not in ED1 (decision 25)")
        elif category == "Demand" and type_ == "DSR" and name == "Demand Reduction":
            record.update(role="dsr", carrier="DSR", quantity="demand_reduction_gwh")
        elif (name, category) in _INFORMATION_LINES:
            record.update(role="information", quantity=name, note=_INFORMATION_LINES[(name, category)])
        if record["role"] in {"supply", "excluded", "storage", "interconnector", "daccs", "dsr"}:
            value = record["value"]
            if record["quantity"] == "export_gwh":
                value = abs(value)   # FES 2025 prints exports as negative flows
            record["value_std"] = to_mw(value, unit) if is_power else to_gwh(value, unit)
        records.append(record)
    return pd.DataFrame.from_records(records)


def unmapped_fes_lines(classified: pd.DataFrame) -> pd.DataFrame:
    """Non-zero ES1 lines with no mapping, in the column layout the energy validation reports."""
    lines = classified[classified["role"].eq("unmapped")]
    return pd.DataFrame({"Connection": lines["connection"], "Variable": lines["variable"],
                         "Category": lines["category"], "Type": lines["type"], "SubType": lines["subtype"],
                         "value": lines["value"], "excel_row": lines["excel_row"]}).reset_index(drop=True)


# ───────────────────────────────────── ED1 ──────────────────────────────────

# ED1 "Data item" labels (FES 2025, stripped), keyed by the name returned.
ED1_ANNUAL_ITEMS = {
    "system_total": "GBFES System Demand: Total",
    "residential": "GBFES Customer Demand: Residential",
    "residential_appliances": "Residential Appliance Demand",
    "residential_lighting": "Residential Light Demand",
    "residential_storage_heat": "Residential Storage Heat",
    "residential_resistive_heat": "Residential Resistive Heat Demand",
    "residential_smart_meter_reduction": "Residential Smart Metering - Demand Reduction",
    "residential_heat_pumps": "Residential Heat Pumps",
    "residential_evs": "Residential Electric Vehicles Demand (EVs)",
    "residential_air_conditioning": "Residential Air Conditioning Demand",
    "residential_district_heat": "Residential District Heat Demand",
    "residential_microgeneration": "Residential Micro-generation",
    "industrial_including_electrolysis": "GBFES Customer Demand: Total Industrial (Electrolysis & District Heat included)",
    "networked_electrolysis": "GBFES Customer Demand: Networked Electrolysis",
    "industrial_excluding_electrolysis": "GBFES Customer Demand: Total Industrial (Excluding Electrolysis)",
    "commercial": "GBFES Customer Demand: Underlying Commercial including unmetered demand",
    "commercial_microgeneration": "Commercial Micro-generation",
    "commercial_at_meter": "GBFES Customer Demand: Commercial electricity demand at the meter",
    "commercial_at_meter_excluding_evs": "Commercial electricity demand at the meter with EVs deducted",
    "commercial_evs": "Commercial Electric Vehicles Demand (EVs)",
    "transmission_losses": "GBFES System Losses: Transmission",
    "distribution_losses": "GBFES System Losses: Distribution",
    "direct_transmission_customers": "GBFES Customer Demand: Direct Transmission Customers",
    "rail_traction": "I&C Rail Traction Demand",
    "data_centres": "Data Centre Demand",
}


#: ED1 closure tolerance (consumer + losses + electrolysis against the system total).
ED1_CLOSURE_TOLERANCE_GWH = 1.0
ED1_CLOSURE_TOLERANCE_SHARE = 1e-5


def read_ed1(workbook: str | Path = FES2025_WORKBOOK) -> pd.DataFrame:
    """ED1 (electricity demand summary) with ``excel_row``; header on row 5."""
    return read_sheet_table(workbook, "ED1", ("Data item", "Unit", "Pathway", "Peak/ Annual/ Minimum"))


def _ed1_series(ed1: pd.DataFrame, label: str, years: Iterable[int]) -> tuple[dict[int, float], int, str]:
    rows = ed1[ed1["Data item"].map(_norm).eq(label)]
    if len(rows) != 1:
        raise ValueError(f"ED1 item {label!r}: expected one row for the pathway, found {len(rows)}")
    row = rows.iloc[0]
    values = {y: float(pd.to_numeric(row[str(y)], errors="coerce")) for y in years if str(y) in rows.columns}
    if any(np.isnan(v) for v in values.values()):
        raise ValueError(f"ED1 item {label!r} has no value in {sorted(values)}")
    return values, int(row["excel_row"]), _norm(row["Unit"])


def ed1_demand(ed1: pd.DataFrame, pathway: str, year: int) -> dict[str, Any]:
    """ED1 annual demand items and peak rows for a pathway, fiscal and calendar.

    Args:
        ed1: ED1 table from :func:`read_ed1`.
        pathway: FES pathway.
        year: Calendar year wanted; ED1 columns ``year`` and ``year - 1`` are read.

    Returns:
        ``annual`` (item -> {fy_gwh, fy_previous_gwh, calendar_gwh, excel_row, label}),
        derived ``totals`` (system and consumer bases, each excluding electrolysis),
        and ``peak_gw`` (every ED1 "Peak" row of the pathway, as published).
    """
    rows = ed1[ed1["Pathway"].map(_norm).eq(pathway)]
    if rows.empty:
        raise ValueError(f"No ED1 rows for pathway {pathway!r}")
    annual_rows = rows[rows["Peak/ Annual/ Minimum"].map(_norm).str.lower().eq("annual [fiscal]")]
    annual: dict[str, dict[str, Any]] = {}
    for key, label in ED1_ANNUAL_ITEMS.items():
        values, excel_row, unit = _ed1_series(annual_rows, label, (year - 1, year))
        scale = to_gwh(1.0, unit)
        annual[key] = {"label": label, "excel_row": excel_row, "unit": unit,
                       "fy_gwh": values[year] * scale, "fy_previous_gwh": values[year - 1] * scale,
                       "calendar_gwh": fiscal_to_calendar({y: v * scale for y, v in values.items()}, year)}

    def _combine(keys_plus: Iterable[str], keys_minus: Iterable[str] = ()) -> dict[str, float]:
        out = {}
        for basis in ("fy_gwh", "calendar_gwh"):
            out[basis] = (sum(annual[k][basis] for k in keys_plus) - sum(annual[k][basis] for k in keys_minus))
        return out

    consumer = ("residential", "industrial_excluding_electrolysis", "commercial", "direct_transmission_customers")
    totals = {
        # ED1 row 9 less row 74: includes transmission and distribution losses.
        "system_excluding_electrolysis": _combine(["system_total"], ["networked_electrolysis"]),
        # scripts/demand/load.py get_ed1_consumer_demand_gwh: excludes losses, what the FES load builder scales to.
        "consumer_excluding_electrolysis": _combine(consumer),
        "losses": _combine(["transmission_losses", "distribution_losses"]),
        "electrolysis": _combine(["networked_electrolysis"]),
        "heat_pumps_residential": _combine(["residential_heat_pumps"]),
        "evs": _combine(["residential_evs", "commercial_evs"]),
    }
    # The system total must close: consumer + losses + electrolysis (ED1 rows 9 = 14+69+84+109+114+119),
    # to the workbook's rounding: up to 1.9 GWh in 19 pathway-years of FES 2025 (26 Oct 2026), so
    # 1 GWh or 10 ppm of the total, whichever is larger.
    closure = (totals["consumer_excluding_electrolysis"]["fy_gwh"] + totals["losses"]["fy_gwh"]
               + totals["electrolysis"]["fy_gwh"] - annual["system_total"]["fy_gwh"])
    tolerance = max(ED1_CLOSURE_TOLERANCE_GWH, ED1_CLOSURE_TOLERANCE_SHARE * abs(annual["system_total"]["fy_gwh"]))
    if abs(closure) > tolerance:
        raise ValueError(f"ED1 {pathway} {year} does not close: consumer + losses + electrolysis - total = {closure:.1f} GWh")

    peak_rows = rows[rows["Peak/ Annual/ Minimum"].map(_norm).str.lower().eq("peak")]
    peak = {}
    for _, row in peak_rows.iterrows():
        value = float(pd.to_numeric(row[str(year)], errors="coerce"))
        peak[_norm(row["Data item"])] = {"gw": to_mw(value, _norm(row["Unit"])) / 1e3,
                                         "excel_row": int(row["excel_row"])}
    return {"annual": annual, "totals": totals, "peak_gw": peak, "year_basis": "ED1 Annual [Fiscal]"}


# ────────────────────────────── building blocks ─────────────────────────────

DEMAND_BUILDING_BLOCKS = {
    "Dem_BB003": "total annual consumption (GSP shares only: 56.9 TWh short of ED1 in HT 2050)",
    "Dem_BB004": "residential heat pumps",
    "Dem_BB005": "I&C heat pumps",
    "Dem_BB006": "EVs: cars, vans, motorbikes",
    "Dem_BB007": "EVs: HGVs, buses",
    "Dem_BB008": "baseline demand",
    "Dem_BB008a": "baseline domestic",
    "Dem_BB008b": "baseline non-domestic",
    "Dem_BB009": "hydrogen electrolysis capacity (MW)",
}


def read_building_blocks(workbook: str | Path, pathway: str,
                         block_ids: Iterable[str] = tuple(DEMAND_BUILDING_BLOCKS)) -> pd.DataFrame:
    """Demand building-block rows (sheet BB1) for one pathway; about 80 s to stream."""
    table = read_sheet_table(workbook, "BB1", ("FES Pathway", "Building Block ID Number", "Unit"),
                             filter_column="Building Block ID Number", filter_values=block_ids)
    return table[table["FES Pathway"].map(_norm).eq(pathway)]


def building_block_demand(bb: pd.DataFrame, year: int) -> dict[str, dict[str, Any]]:
    """National sums of the demand building blocks, fiscal and calendar (energy blocks only).

    Args:
        bb: Rows from :func:`read_building_blocks`.
        year: Calendar year.

    Returns:
        Block id -> {description, unit, fy, fy_previous, calendar (energy only), rows}.
    """
    out = {}
    for block, description in DEMAND_BUILDING_BLOCKS.items():
        rows = bb[bb["Building Block ID Number"].map(_norm).eq(block)]
        if rows.empty:
            raise ValueError(f"Building block {block} has no rows")
        units = set(rows["Unit"].map(_norm))
        if len(units) != 1:
            raise ValueError(f"Building block {block} mixes units {units}")
        unit = units.pop()
        fy = float(_year_value(rows, year).fillna(0.0).sum())
        entry = {"description": description, "unit": unit, "rows": int(len(rows)), "fy": fy}
        if unit in _UNIT_TO_GWH:
            fy_prev = float(_year_value(rows, year - 1).fillna(0.0).sum())
            entry.update(fy_gwh=to_gwh(fy, unit), fy_previous_gwh=to_gwh(fy_prev, unit),
                         calendar_gwh=to_gwh(fiscal_to_calendar({year: fy, year - 1: fy_prev}, year), unit))
        elif unit in _UNIT_TO_MW:
            entry.update(mw=to_mw(fy, unit))
        out[block] = entry
    return out


# ───────────────────────────── hydrogen (WS1, FLX1) ─────────────────────────

WS1_HYDROGEN_SUPPLY = ("Methane Reformation with CCUS", "Networked Electrolysis", "Non-networked Electrolysis",
                       "Biomass gasification", "Nuclear Electrolysis", "Imports", "Exports")
WS1_HYDROGEN_DEMAND = ("Road Transport", "Residential", "Commercial", "Industrial", "Power Generation", "Rail",
                       "Shipping", "Blending", "Aviation", "Direct Air Carbon Capture and Storage",
                       "Transmission Losses")


def read_ws1(workbook: str | Path = FES2025_WORKBOOK) -> pd.DataFrame:
    """WS1 (whole system and gas supply) with ``excel_row``; header on row 8."""
    return read_sheet_table(workbook, "WS1", ("Fuel Type", "Pathway", "Units", "Category", "Sector"))


def read_flx1(workbook: str | Path = FES2025_WORKBOOK) -> pd.DataFrame:
    """FLX1 (flexibility) with ``excel_row``; header on row 7."""
    return read_sheet_table(workbook, "FLX1", ("Flexibility type", "Data item", "Unit", "Pathway", "Detail"))


def hydrogen_quantities(ws1: pd.DataFrame, flx1: pd.DataFrame, classified_es1: pd.DataFrame,
                        pathway: str, year: int) -> dict[str, Any]:
    """HT hydrogen balance inputs: supply, capacity, feedstock, demand, storage, turbines.

    Energies are TWh of hydrogen (HHV as FES publishes) unless named ``_electric``.
    ``annual_twh`` is as published (fiscal basis, see the module docstring);
    ``annual_twh_calendar`` applies :func:`fiscal_to_calendar`. Capacities and storage are
    as published.

    Args:
        ws1: WS1 table.
        flx1: FLX1 table.
        classified_es1: Output of :func:`classify_es1` for the same pathway and year.
        pathway: FES pathway.
        year: Year.

    Returns:
        Hydrogen quantities with WS1/FLX1/ES1 row provenance.
    """
    h2 = ws1[ws1["Fuel Type"].map(_norm).eq("Hydrogen") & ws1["Pathway"].map(_norm).eq(pathway)]
    if h2.empty:
        raise ValueError(f"No WS1 hydrogen rows for {pathway!r}")
    provenance: dict[str, str] = {}

    def _pick(category: str, sector: Optional[str], unit: str) -> dict[str, float]:
        rows = h2[h2["Category"].map(_norm).eq(category) & h2["Units"].map(_norm).eq(unit)]
        if sector is not None:
            rows = rows[rows["Sector"].map(_norm).str.lower().eq(sector.lower())]
        if len(rows) != 1:
            raise ValueError(f"WS1 hydrogen {category}/{sector} ({unit}): expected one row, found {len(rows)}")
        row = rows.iloc[0]
        provenance[f"{category}/{sector}"] = f"WS1 row {int(row['excel_row'])}"
        return {y: float(pd.to_numeric(row[str(y)], errors="coerce") or 0.0) for y in (year - 1, year)}

    supply = {s: _pick("Supply", s, "TWh") for s in WS1_HYDROGEN_SUPPLY}
    capacity = {s: _pick("Capacity", s, "GW")[year] for s in WS1_HYDROGEN_SUPPLY}
    demand = {s: _pick("Demand", s, "TWh") for s in WS1_HYDROGEN_DEMAND}
    feedstock = {"Methane Reformation with CCUS": _pick("Feedstock", "Methane Reformation with CCUS", "TWh"),
                 "Networked Electrolysis": _pick("Feedstock", "Networked Electrolysis", "TWh"),
                 "Non-networked Electrolysis": _pick("Feedstock", "Non-networked Electrolysis", "TWh"),
                 "Biomass Gasification": _pick("Feedstock", "Biomass Gasification", "TWh")}
    storage = _pick("Storage", None, "TWh")[year]

    flx = flx1[flx1["Pathway"].map(_norm).eq(pathway)
               & flx1["Detail"].map(_norm).eq("Grid-connected electrolysis capacity (installed electric capacity)")]
    if len(flx) != 1:
        raise ValueError(f"FLX1 grid-connected electrolysis capacity: expected one row, found {len(flx)}")
    electrolysis_gw_electric = float(pd.to_numeric(flx.iloc[0][str(year)], errors="coerce"))
    provenance["electrolysis_capacity_gw_electric"] = f"FLX1 row {int(flx.iloc[0]['excel_row'])}"

    turbines = classified_es1[classified_es1["role"].eq("supply") & classified_es1["carrier"].eq("Hydrogen")]
    turbine_mw = float(turbines.loc[turbines["quantity"].eq("capacity_mw"), "value_std"].sum())
    turbine_gwh = float(turbines.loc[turbines["quantity"].eq("generation_gwh"), "value_std"].sum())

    def _year(d: dict[str, dict[int, float]]) -> dict[str, float]:
        return {k: v[year] for k, v in d.items()}

    def _calendar(d: dict[str, dict[int, float]]) -> dict[str, float]:
        return {k: fiscal_to_calendar(v, year) for k, v in d.items()}

    demand_y = _year(demand)
    non_power = sum(v for k, v in demand_y.items() if k != "Power Generation")
    electrolysis_in = feedstock["Networked Electrolysis"][year]
    electrolysis_out = supply["Networked Electrolysis"][year]
    efficiency = electrolysis_out / electrolysis_in if electrolysis_in else np.nan
    total_supply = sum(_year(supply).values())
    total_demand = sum(demand_y.values())
    return {
        "units": "TWh H2 (as FES publishes) unless named otherwise",
        "annual_twh": {"supply": _year(supply), "demand": demand_y, "feedstock": _year(feedstock)},
        "annual_twh_calendar": {"supply": _calendar(supply), "demand": _calendar(demand),
                                "feedstock": _calendar(feedstock)},
        "capacity_gw_h2_output": capacity,
        "electrolysis_capacity_gw_electric": electrolysis_gw_electric,
        "electrolysis_efficiency": efficiency,
        "gas_reforming_ccs": {"capacity_gw_h2": capacity["Methane Reformation with CCUS"],
                              "output_twh": supply["Methane Reformation with CCUS"][year],
                              "gas_feedstock_twh": feedstock["Methane Reformation with CCUS"][year]},
        "non_power_demand_twh": non_power,
        "power_generation_demand_twh": demand_y["Power Generation"],
        "total_supply_twh": total_supply,
        "total_demand_twh": total_demand,
        "storage_twh": storage,
        "h2_turbines": {"capacity_mw": turbine_mw, "generation_gwh": turbine_gwh,
                        "efficiency_electric_per_h2": (turbine_gwh / 1e3 / demand_y["Power Generation"]
                                                       if demand_y["Power Generation"] else np.nan),
                        "source": "ES1 Hydrogen, Hydrogen Peaking and Hydrogen CHP lines"},
        "feasibility": {
            "electrolysis_max_twh_h2": electrolysis_gw_electric * 8.76 * efficiency,
            "electrolysis_plus_reforming_max_twh_h2": (electrolysis_gw_electric * 8.76 * efficiency
                                                       + capacity["Methane Reformation with CCUS"] * 8.76),
            "note": "annual capacity x 8,760 h (x efficiency for electrolysis) against total demand; "
                    "electrolysis alone cannot meet it, reforming with CCS is needed (plan Y6)",
        },
        "provenance": provenance,
    }


# ──────────────────────────────── the targets ───────────────────────────────

def _by_carrier(classified: pd.DataFrame, quantity: str, role: str = "supply") -> pd.Series:
    lines = classified[classified["role"].eq(role) & classified["quantity"].eq(quantity)]
    return lines.groupby("carrier")["value_std"].sum()


def supply_tables(classified: pd.DataFrame) -> dict[str, Any]:
    """Supply, storage and interconnector totals of one pathway and year from classified ES1 lines.

    Args:
        classified: Output of :func:`classify_es1`.

    Returns:
        ``generation`` (by carrier: generation_gwh, es1_lines, curtailment_gwh, gross_gwh,
        capacity_mw), ``capacity_mw`` (Series), ``storage`` (by carrier: capacity_mw,
        generation_gwh, consumption_gwh, energy_gwh), ``ic`` (Series by quantity), ``excluded``
        (the excluded lines) and ``daccs_gwh``.
    """
    supply = classified[classified["role"].eq("supply")]
    capacity = _by_carrier(classified, "capacity_mw")
    generation = _by_carrier(classified, "generation_gwh")
    curtailment = _by_carrier(classified, "curtailment_gwh")
    carriers = sorted(set(capacity.index) | set(generation.index) | set(curtailment.index))
    lines = supply.assign(line=supply["type"] + " / " + supply["subtype"]).groupby("carrier")["line"].agg(
        lambda v: "; ".join(sorted(set(v))))
    gen_table = pd.DataFrame(index=pd.Index(carriers, name="carrier"))
    gen_table["generation_gwh"] = generation.reindex(carriers).fillna(0.0).astype(float)
    gen_table["es1_lines"] = lines.reindex(carriers).fillna("")
    gen_table["curtailment_gwh"] = curtailment.reindex(carriers).fillna(0.0).astype(float)
    gen_table["gross_gwh"] = gen_table["generation_gwh"] + gen_table["curtailment_gwh"]
    gen_table["capacity_mw"] = capacity.reindex(carriers).fillna(0.0).astype(float)

    storage_lines = classified[classified["role"].eq("storage")]
    storage = storage_lines.pivot_table(index="carrier", columns="quantity", values="value_std", aggfunc="sum")
    storage = storage.reindex(columns=list(_STORAGE_VARIABLES.values())).fillna(0.0)
    ic = classified[classified["role"].eq("interconnector")].groupby("quantity")["value_std"].sum()
    return {"generation": gen_table, "capacity_mw": capacity, "storage": storage, "ic": ic,
            "excluded": classified[classified["role"].eq("excluded")],
            "daccs_gwh": float(classified.loc[classified["role"].eq("daccs"), "value_std"].sum())}


def fes2025_energy_targets(workbook: str | Path, pathway: str, modelled_year: int,
                           year_basis: str = "calendar", demand_basis: str = "consumer",
                           include_building_blocks: bool = True, strict: bool = False) -> dict[str, Any]:
    """FES 2025 energy targets for a pathway and year, in the CP30 target schema.

    The keys ``generation``, ``storage_gwh``, ``demand_gwh``, ``electrolysis_gwh``,
    ``import_gwh``, ``export_gwh`` and ``unmapped`` match ``cp30_energy_targets()`` so
    ``build_energy_validation()`` takes either. The rest are FES extras.

    Args:
        workbook: FES 2025 Data Workbook.
        pathway: Pathway name, e.g. "Holistic Transition".
        modelled_year: Calendar year.
        year_basis: "calendar" (default; fiscal ED1 lines converted) or "fiscal" (ED1 as published).
        demand_basis: Which ED1 total ``demand_gwh`` is. "consumer" (default): ED1 residential +
            industrial excluding electrolysis + commercial + direct customers, excluding losses;
            what ``scripts/demand/load.py`` scales network loads to. "system": ED1 row 9 less
            electrolysis, including losses.
        include_building_blocks: Also sum the Dem_BB blocks (sheet BB1; about 80 s).
        strict: Raise if any non-zero ES1 line is unmapped.

    Returns:
        The targets dict.
    """
    if year_basis not in {"calendar", "fiscal"}:
        raise ValueError("year_basis must be 'calendar' or 'fiscal'")
    if demand_basis not in {"consumer", "system"}:
        raise ValueError("demand_basis must be 'consumer' or 'system'")
    workbook = Path(workbook)
    year = int(modelled_year)
    es1 = read_es1(workbook)
    classified = classify_es1(es1, pathway, year)
    unmapped = unmapped_fes_lines(classified)
    if strict and not unmapped.empty:
        raise ValueError("FES 2025 ES1 lines with no model mapping (add them to FES2025_GENERATION_MAP or "
                         f"FES2025_STORAGE_MAP rather than dropping them):\n{unmapped.to_string(index=False)}")

    tables = supply_tables(classified)
    gen_table, capacity, storage, ic = tables["generation"], tables["capacity_mw"], tables["storage"], tables["ic"]
    excluded, daccs_gwh = tables["excluded"], tables["daccs_gwh"]

    demand = ed1_demand(read_ed1(workbook), pathway, year)
    basis_key = "calendar_gwh" if year_basis == "calendar" else "fy_gwh"
    total_key = "consumer_excluding_electrolysis" if demand_basis == "consumer" else "system_excluding_electrolysis"
    demand_gwh = demand["totals"][total_key][basis_key]
    electrolysis_gwh = demand["totals"]["electrolysis"][basis_key]

    components: dict[str, Any] = {
        "district_heat_residential_gwh": demand["annual"]["residential_district_heat"][basis_key],
        "data_centres_gwh": demand["annual"]["data_centres"][basis_key],
        "daccs_gwh": daccs_gwh,
        "note": "ED1 data centres and rail are supplementary (level 5) lines inside commercial; never add them again",
    }
    if include_building_blocks:
        bb = building_block_demand(read_building_blocks(workbook, pathway), year)
        components["building_blocks"] = bb
        hp = bb["Dem_BB004"][basis_key] + bb["Dem_BB005"][basis_key]
        ev = bb["Dem_BB006"][basis_key] + bb["Dem_BB007"][basis_key]
        components.update(heat_pumps_gwh=hp, evs_gwh=ev,
                          residual_gwh=demand["totals"]["consumer_excluding_electrolysis"][basis_key] - hp - ev,
                          residual_note="ED1 consumer total less Dem_BB004-007 (plan Y6 base)")

    hydrogen = hydrogen_quantities(read_ws1(workbook), read_flx1(workbook), classified, pathway, year)

    return {
        "source": SOURCE_ID,
        "source_label": SOURCE_LABEL,
        "workbook": str(workbook),
        "pathway": pathway,
        "modelled_year": year,
        "year_basis": year_basis,
        "demand_basis": demand_basis,
        # CP30 target schema
        "generation": gen_table,
        "storage_gwh": storage["generation_gwh"].to_dict() if "generation_gwh" in storage else {},
        "demand_gwh": demand_gwh,
        "demand_line": f"ED1 {total_key.replace('_', ' ')} ({year_basis})",
        "electrolysis_gwh": electrolysis_gwh,
        "import_gwh": float(ic.get("import_gwh", 0.0)),
        "export_gwh": float(ic.get("export_gwh", 0.0)),
        "unmapped": unmapped,
        # FES extras
        "capacity_mw": capacity,
        "storage": storage,
        "interconnectors": {k: float(v) for k, v in ic.items()},
        "daccs_gwh": daccs_gwh,
        "excluded": excluded.reset_index(drop=True),
        "es1_lines": classified,
        "demand": demand,
        "demand_components": components,
        "hydrogen": hydrogen,
        "rules": {
            "fiscal_to_calendar": f"calendar Y = {FISCAL_WEIGHT_CURRENT} FY(Y) + {FISCAL_WEIGHT_PREVIOUS} FY(Y-1); "
                                  "FY(Y) = April Y to March Y+1",
            "es1_year_basis": "calendar (ES3 rows 5-8); capacities at 1 December",
            "units": "capacity MW, energy GWh (ES1 TWh x 1,000); exports positive",
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, pd.Series):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def targets_to_json(targets: dict[str, Any]) -> dict[str, Any]:
    """The targets as a JSON-serialisable dict (the ES1 line table is written separately)."""
    return _jsonable({k: v for k, v in targets.items() if k != "es1_lines"})


def targets_summary_table(targets: dict[str, Any]) -> pd.DataFrame:
    """One tidy table of the headline targets: section, item, value, unit, source."""
    rows: list[dict[str, Any]] = []
    gen = targets["generation"]
    for carrier, row in gen.iterrows():
        for col, unit in (("capacity_mw", "MW"), ("generation_gwh", "GWh"), ("curtailment_gwh", "GWh"),
                          ("gross_gwh", "GWh")):
            rows.append({"section": "supply", "item": carrier, "quantity": col, "value": row[col], "unit": unit,
                         "source": f"ES1: {row['es1_lines']}"})
    for carrier, row in targets["storage"].iterrows():
        for col, value in row.items():
            rows.append({"section": "storage", "item": carrier, "quantity": col, "value": value,
                         "unit": "MW" if col.endswith("_mw") else "GWh", "source": "ES1 Storage"})
    for key, value in targets["interconnectors"].items():
        rows.append({"section": "interconnectors", "item": "GB", "quantity": key, "value": value,
                     "unit": "MW" if key.endswith("_mw") else "GWh", "source": "ES1 Interconnectors"})
    for key, totals in targets["demand"]["totals"].items():
        for basis, value in totals.items():
            rows.append({"section": "demand", "item": key, "quantity": basis, "value": value, "unit": "GWh",
                         "source": "ED1 (derived)"})
    for key, item in targets["demand"]["annual"].items():
        for basis in ("fy_gwh", "calendar_gwh"):
            rows.append({"section": "demand_ed1", "item": key, "quantity": basis, "value": item[basis],
                         "unit": "GWh", "source": f"ED1 row {item['excel_row']}"})
    for label, item in targets["demand"]["peak_gw"].items():
        rows.append({"section": "peak", "item": label, "quantity": "peak_gw", "value": item["gw"], "unit": "GW",
                     "source": f"ED1 row {item['excel_row']}"})
    comp = targets["demand_components"]
    for key in ("heat_pumps_gwh", "evs_gwh", "residual_gwh", "district_heat_residential_gwh", "data_centres_gwh",
                "daccs_gwh"):
        if key in comp:
            rows.append({"section": "demand_components", "item": key, "quantity": targets["year_basis"],
                         "value": comp[key], "unit": "GWh", "source": "ED1 / Dem_BB / ES1"})
    for block, entry in comp.get("building_blocks", {}).items():
        for key in ("fy_gwh", "calendar_gwh", "mw"):
            if key in entry:
                rows.append({"section": "building_blocks", "item": block, "quantity": key, "value": entry[key],
                             "unit": "MW" if key == "mw" else "GWh", "source": f"BB1 {entry['description']}"})
    h2 = targets["hydrogen"]
    for group in ("supply", "demand", "feedstock"):
        for key, value in h2["annual_twh"][group].items():
            rows.append({"section": f"hydrogen_{group}", "item": key, "quantity": "published", "value": value,
                         "unit": "TWh", "source": h2["provenance"].get(f"{group.capitalize()}/{key}", "WS1")})
    for key in ("non_power_demand_twh", "power_generation_demand_twh", "total_supply_twh", "total_demand_twh",
                "storage_twh", "electrolysis_capacity_gw_electric", "electrolysis_efficiency"):
        rows.append({"section": "hydrogen", "item": key, "quantity": "published", "value": h2[key],
                     "unit": "GW" if key.endswith("_gw_electric") else ("ratio" if key.endswith("efficiency") else "TWh"),
                     "source": "WS1 / FLX1"})
    for key, value in h2["gas_reforming_ccs"].items():
        rows.append({"section": "hydrogen", "item": f"gas_reforming_ccs_{key}", "quantity": "published",
                     "value": value, "unit": "GW" if "capacity" in key else "TWh", "source": "WS1"})
    for key in ("capacity_mw", "generation_gwh"):
        rows.append({"section": "hydrogen", "item": f"h2_turbines_{key}", "quantity": "ES1", "value":
                     h2["h2_turbines"][key], "unit": "MW" if key == "capacity_mw" else "GWh", "source": "ES1"})
    out = pd.DataFrame(rows)
    out.insert(0, "modelled_year", targets["modelled_year"])
    out.insert(0, "pathway", targets["pathway"])
    return out
