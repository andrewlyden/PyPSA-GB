"""The scenario-source data contract: one directory of long tables per source edition.

Every adapter (``cp30_adapter.py``, ``fes_adapter.py``) writes ``resources/sources/{source}_{edition}/``
with the tables below, keyed by ``pathway_code`` (``data/scenario_sources/pathways.csv``) and ``year``,
and ``availability.json``, which lists the (pathway, year) pairs each table holds. A consumer asks
:func:`require` for a pair and gets an error when the source does not publish it, instead of another
year's rows (the CP30 ES5 table, published for 2030 only, was stamped on any year until 26 Oct 2026).

Units: capacity MW, energy GWh. ``demand.csv`` keeps each line's unit, with the calendar value in
``value`` and, where the source publishes fiscal years (FES ED1), the fiscal values beside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_DIR = PROJECT_ROOT / "resources" / "sources"

#: Columns of each contract table, in order. Adapters may not add or drop any.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "capacity": ("pathway_code", "pathway", "year", "carrier", "capacity_mw", "lines"),
    "energy": ("pathway_code", "pathway", "year", "carrier", "generation_gwh", "curtailment_gwh", "gross_gwh",
               "import_gwh", "export_gwh", "lines"),
    "storage": ("pathway_code", "pathway", "year", "carrier", "power_mw", "energy_gwh", "generation_gwh",
                "consumption_gwh", "lines"),
    "demand": ("pathway_code", "pathway", "year", "item", "unit", "value", "year_basis", "fiscal_value",
               "fiscal_previous_value", "line"),
    "regional_capacity": ("pathway_code", "pathway", "year", "scheme", "region", "carrier", "capacity_mw"),
    "gsp_blocks": ("pathway_code", "year", "block_id", "unit", "gsp", "gsp_id", "node_id", "dno_area",
                   "technology", "technology_detail", "value"),
    "gsp_info": ("gsp_id", "gsp_group", "minor_flop", "name", "latitude", "longitude", "comments"),
}
#: Tables keyed by pathway and year (gsp_info is per edition).
KEYED_TABLES = ("capacity", "energy", "storage", "demand", "regional_capacity", "gsp_blocks")
#: Sort keys of each table, so a rerun writes identical bytes.
SORT_KEYS: dict[str, tuple[str, ...]] = {
    "capacity": ("pathway_code", "year", "carrier"),
    "energy": ("pathway_code", "year", "carrier"),
    "storage": ("pathway_code", "year", "carrier"),
    "demand": ("pathway_code", "year", "item"),
    "regional_capacity": ("pathway_code", "year", "scheme", "carrier", "region"),
    "gsp_blocks": ("pathway_code", "block_id", "gsp", "year"),
    "gsp_info": ("gsp_id",),
}


class NotPublished(LookupError):
    """The source edition does not publish this table for this pathway and year."""


def source_dir(source: str, edition: int, root: str | Path = DEFAULT_SOURCES_DIR) -> Path:
    """``{root}/{source}_{edition}``, e.g. resources/sources/fes_2025."""
    return Path(root) / f"{str(source).lower()}_{int(edition)}"


def empty_table(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(TABLE_COLUMNS[name]))


def conform(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """A table in contract column order and sort order; a missing or extra column fails."""
    columns = list(TABLE_COLUMNS[name])
    missing = [c for c in columns if c not in frame.columns]
    extra = [c for c in frame.columns if c not in columns]
    if missing or extra:
        raise ValueError(f"Contract table {name}: missing {missing}, unexpected {extra}")
    out = frame[columns]
    if len(out):
        out = out.sort_values(list(SORT_KEYS[name]), kind="mergesort").reset_index(drop=True)
    return out


def availability_of(tables: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, list[int]]]:
    """{table: {pathway_code: [years]}} of the keyed tables that have rows."""
    out: dict[str, dict[str, list[int]]] = {}
    for name in KEYED_TABLES:
        frame = tables.get(name)
        if frame is None or frame.empty:
            continue
        pairs = frame[["pathway_code", "year"]].drop_duplicates()
        out[name] = {str(code): sorted(int(y) for y in group["year"])
                     for code, group in pairs.groupby("pathway_code", sort=True)}
    return out


def demand_item_availability(demand: Optional[pd.DataFrame]) -> dict[str, dict[str, list[int]]]:
    """{item: {pathway_code: [years]}} of demand.csv: a year can carry some demand lines and not others."""
    if demand is None or demand.empty:
        return {}
    pairs = demand[["item", "pathway_code", "year"]].drop_duplicates()
    return {str(item): {str(code): sorted(int(y) for y in g["year"]) for code, g in by_item.groupby("pathway_code")}
            for item, by_item in pairs.groupby("item", sort=True)}


def write_source(source: str, edition: int, tables: Mapping[str, pd.DataFrame], meta: Mapping[str, Any],
                 root: str | Path = DEFAULT_SOURCES_DIR) -> Path:
    """Write the contract tables present in ``tables`` and ``availability.json``.

    Args:
        source: "FES" or "CP30".
        edition: Edition.
        tables: Contract tables by name (absent or empty: not published).
        meta: Extra provenance for ``availability.json`` (inputs, rules, notes).
        root: Sources root.

    Returns:
        The source directory.
    """
    out_dir = source_dir(source, edition, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in TABLE_COLUMNS:
        path = out_dir / f"{name}.csv"
        frame = tables.get(name)
        if frame is None or frame.empty:
            if path.exists():
                path.unlink()
            continue
        conform(name, frame).to_csv(path, index=False, lineterminator="\n")
        written[name] = len(frame)
    availability = {
        "source": str(source).upper(),
        "edition": int(edition),
        "tables": availability_of(tables),
        "demand_items": demand_item_availability(tables.get("demand")),
        "gsp_info": bool(tables.get("gsp_info") is not None and not tables["gsp_info"].empty),
        "rows": written,
        **dict(meta),
    }
    text = json.dumps(availability, indent=1, sort_keys=False)
    # Each year list on one line.
    text = re.sub(r"\[\n\s+([0-9,\s]+?)\n\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    (out_dir / "availability.json").write_text(text + "\n", encoding="utf-8")
    return out_dir


def load_availability(source: str, edition: int, root: str | Path = DEFAULT_SOURCES_DIR) -> dict[str, Any]:
    path = source_dir(source, edition, root) / "availability.json"
    if not path.is_file():
        raise FileNotFoundError(f"No {source} {edition} source tables at {path.parent}; run the adapter "
                                f"(python -m scripts.sources.build --source {source} --edition {edition})")
    return json.loads(path.read_text(encoding="utf-8"))


def require(source: str, edition: int, table: str, pathway_code: str, year: int, item: Optional[str] = None,
            root: str | Path = DEFAULT_SOURCES_DIR) -> None:
    """Fail unless ``table`` (and, for demand, ``item``) holds ``pathway_code`` and ``year``.

    Raises:
        NotPublished: With the years the source does publish for that table and pathway.
    """
    availability = load_availability(source, edition, root)
    if item is not None:
        available, what = availability.get("demand_items", {}).get(item, {}), f"demand item {item!r}"
    else:
        available, what = availability["tables"].get(table, {}), table
    years = available.get(str(pathway_code), [])
    if int(year) not in years:
        raise NotPublished(f"{source} {edition} does not publish {what} for {pathway_code} {year}; "
                           f"published years: {years or 'none'}")


def read_table(source: str, edition: int, table: str, pathway_code: Optional[str] = None,
               year: Optional[int] = None, root: str | Path = DEFAULT_SOURCES_DIR) -> pd.DataFrame:
    """One contract table, optionally one pathway and year (checked with :func:`require`)."""
    if pathway_code is not None and year is not None:
        require(source, edition, table, pathway_code, year, root=root)
    path = source_dir(source, edition, root) / f"{table}.csv"
    if not path.is_file():
        raise NotPublished(f"{source} {edition} publishes no {table} table ({path} missing)")
    frame = pd.read_csv(path)
    if pathway_code is not None:
        frame = frame[frame["pathway_code"].eq(pathway_code)]
    if year is not None:
        frame = frame[frame["year"].eq(int(year))]
    return frame.reset_index(drop=True)


def years_with_values(frame: pd.DataFrame, year_columns: Iterable[str]) -> list[int]:
    """Year columns of ``frame`` with at least one numeric value."""
    years = []
    for column in year_columns:
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            years.append(int(column))
    return years
