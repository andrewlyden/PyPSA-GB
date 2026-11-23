"""
Extract Clean Power 2030 workbook tables into normalized CSV resources.

Stage 1 intentionally only materializes CP30 data and validation summaries.
Scenario routing and network integration are handled in later stages.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.utilities.logging_config import setup_logging
except Exception:  # pragma: no cover - fallback for very small test environments
    setup_logging = None


DEFAULT_WORKBOOK = PROJECT_ROOT / "data" / "CP30" / "Clean Power 2030 Data Workbook_V003.xlsx"
DEFAULT_OUTPUTS = {
    "es1": PROJECT_ROOT / "resources" / "CP30" / "CP30_ES1.csv",
    "es5": PROJECT_ROOT / "resources" / "CP30" / "CP30_ES5_regional.csv",
    "ed1": PROJECT_ROOT / "resources" / "CP30" / "CP30_ED1.csv",
    "reference": PROJECT_ROOT / "resources" / "CP30" / "CP30_reference_N1_N2_NA1.csv",
    "validation": PROJECT_ROOT / "resources" / "CP30" / "CP30_validation_report.csv",
    "interconnector_prices": PROJECT_ROOT / "resources" / "CP30" / "CP30_interconnector_prices.csv",
}

EXPECTED_PATHWAYS = {"Further Flex and Renewables", "New Dispatch"}
EXPECTED_SHEETS = {"ED1", "ES1", "ES5", "N1", "N2", "NA1", "CP.08", "CP.12"}
REFERENCE_SHEETS = {
    "N1": 4,
    "N2": 5,
    "NA1": 2,
    "CP.08": 9,
}


def get_logger(log_path: str | None = None) -> logging.Logger:
    if setup_logging is not None:
        return setup_logging(log_path or "CP30_data")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("CP30_data")


def normalize_column_name(column: object) -> str:
    """Normalize Excel year and text columns to stable CSV labels."""
    if isinstance(column, (int, float)) and not pd.isna(column):
        if 1900 <= int(column) <= 2100 and float(column).is_integer():
            return str(int(column))

    label = str(column).strip()
    year_match = re.fullmatch(r"(20\d{2})(?:\.0)?", label)
    if year_match:
        return year_match.group(1)
    label = re.sub(r"\s+", " ", label)
    return label


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names and empty columns removed."""
    result = df.copy()
    result.columns = [normalize_column_name(col) for col in result.columns]
    unnamed_cols = [col for col in result.columns if col.lower().startswith("unnamed:")]
    empty_unnamed = [col for col in unnamed_cols if result[col].isna().all()]
    if empty_unnamed:
        result = result.drop(columns=empty_unnamed)
    return result


def clean_table(df: pd.DataFrame, source_sheet: str) -> pd.DataFrame:
    """Apply common CP30 table cleanup."""
    result = normalize_columns(df)
    result = result.dropna(how="all").copy()
    result.insert(0, "source_sheet", source_sheet)
    return result


def read_cp30_table(workbook: Path | str, sheet_name: str, header: int) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet_name, header=header, dtype=object)
    return clean_table(df, sheet_name)


def read_cp30_core_tables(workbook: Path | str) -> dict[str, pd.DataFrame]:
    """Read the core CP30 data tables used by Stage 1."""
    return {
        "ed1": read_cp30_table(workbook, "ED1", 7),
        "es1": read_cp30_table(workbook, "ES1", 9),
        "es5": read_cp30_table(workbook, "ES5", 5),
    }


def read_cp30_interconnector_prices(workbook: Path | str) -> pd.DataFrame:
    """Extract CP.29's pathway-level 2030 import and export prices.

    CP.29 publishes GB-wide volume-weighted trade prices, rather than a price
    for every connected foreign bidding zone. They are therefore an
    authoritative CP30 boundary-price assumption, not country-specific market
    forecasts. The interconnector builder applies the import and export values
    to each cable while retaining country-specific ENTSO-E hourly shapes.
    """
    raw = pd.read_excel(workbook, sheet_name="CP.29", header=None, dtype=object)
    pathways = raw.iloc[6]
    import_row = raw[raw.iloc[:, 9].astype(str).str.strip().eq("Average import price")]
    export_row = raw[raw.iloc[:, 9].astype(str).str.strip().eq("Average export price")]
    if import_row.empty or export_row.empty:
        raise ValueError("CP.29 does not contain Average import/export price rows")

    rows: list[dict[str, object]] = []
    for column, pathway in pathways.items():
        if pd.isna(pathway) or not str(pathway).strip():
            continue
        import_price = pd.to_numeric(import_row.iloc[0, column], errors="coerce")
        export_price = pd.to_numeric(export_row.iloc[0, column], errors="coerce")
        if pd.notna(import_price) and pd.notna(export_price):
            rows.append({
                "source_sheet": "CP.29",
                "year": 2030,
                "pathway": str(pathway).strip(),
                "import_price_gbp_per_mwh": float(import_price),
                "export_price_gbp_per_mwh": float(export_price),
                "price_scope": "GB-wide volume-weighted interconnector trade price",
            })
    if not rows:
        raise ValueError("CP.29 contains no usable pathway price pairs")
    return pd.DataFrame(rows)

def read_reference_tables(workbook: Path | str) -> pd.DataFrame:
    """Read reference/validation sheets into one wide CSV with source metadata."""
    tables = []
    for sheet_name, header in REFERENCE_SHEETS.items():
        table = read_cp30_table(workbook, sheet_name, header)
        tables.append(table)

    cp12_raw = pd.read_excel(workbook, sheet_name="CP.12", header=None, dtype=object)
    cp12_raw = cp12_raw.dropna(how="all").copy()
    cp12_raw.columns = [f"column_{idx + 1:03d}" for idx in range(len(cp12_raw.columns))]
    cp12_raw.insert(0, "source_sheet", "CP.12")
    cp12_raw.insert(1, "source_row", range(1, len(cp12_raw) + 1))
    tables.append(cp12_raw)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


def year_columns(df: pd.DataFrame) -> list[str]:
    return sorted(col for col in df.columns if re.fullmatch(r"20\d{2}", str(col)))


def unique_values(df: pd.DataFrame, column: str) -> set[str]:
    if column not in df.columns:
        return set()
    return {str(value).strip() for value in df[column].dropna().unique() if str(value).strip()}


def add_check(rows: list[dict[str, object]], check: str, status: str, details: str = "") -> None:
    rows.append({"kind": "check", "name": check, "status": status, "value": "", "details": details})


def add_metric(rows: list[dict[str, object]], metric: str, value: object, details: str = "") -> None:
    rows.append({"kind": "metric", "name": metric, "status": "INFO", "value": value, "details": details})


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> tuple[bool, list[str]]:
    missing = [col for col in columns if col not in df.columns]
    return not missing, missing


def build_validation_report(
    workbook: Path | str,
    tables: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact machine-readable CP30 coverage report."""
    workbook = Path(workbook)
    rows: list[dict[str, object]] = []

    add_check(rows, "workbook_exists", "PASS" if workbook.exists() else "FAIL", str(workbook))

    try:
        excel = pd.ExcelFile(workbook)
        sheet_names = set(excel.sheet_names)
        missing_sheets = sorted(EXPECTED_SHEETS - sheet_names)
        add_check(
            rows,
            "expected_sheets_present",
            "PASS" if not missing_sheets else "FAIL",
            ", ".join(missing_sheets),
        )
    except Exception as exc:
        add_check(rows, "expected_sheets_present", "FAIL", str(exc))

    es1 = tables["es1"]
    es5 = tables["es5"]
    ed1 = tables["ed1"]

    for name, table in tables.items():
        add_metric(rows, f"{name}_rows", len(table))
        add_metric(rows, f"{name}_year_columns", ",".join(year_columns(table)))

    for name, table in {"ES1": es1, "ES5": es5, "ED1": ed1}.items():
        pathway_col = "Pathway" if "Pathway" in table.columns else "Scenario"
        pathways = unique_values(table, pathway_col)
        missing_pathways = sorted(EXPECTED_PATHWAYS - pathways)
        add_check(
            rows,
            f"{name}_pathways",
            "PASS" if not missing_pathways else "FAIL",
            f"found={sorted(pathways)} missing={missing_pathways}",
        )

    required_es1 = ["Connection", "Pathway", "Variable", "Category", "Type", "SubType", "2030"]
    ok, missing = require_columns(es1, required_es1)
    add_check(rows, "ES1_required_columns", "PASS" if ok else "FAIL", ", ".join(missing))

    required_es5 = [
        "Scenario",
        "Technology",
        "Variable",
        "East England",
        "Midlands",
        "North England",
        "North Scotland",
        "South East",
        "South Scotland",
        "South West",
        "Wales",
    ]
    ok, missing = require_columns(es5, required_es5)
    add_check(rows, "ES5_required_columns", "PASS" if ok else "FAIL", ", ".join(missing))

    required_ed1 = ["Data item", "Unit", "Pathway", "Fuel", "Annual/ Peak/ Flex", "2030"]
    ok, missing = require_columns(ed1, required_ed1)
    add_check(rows, "ED1_required_columns", "PASS" if ok else "FAIL", ", ".join(missing))

    if "Variable" in es1.columns:
        es1_variables = unique_values(es1, "Variable")
        for variable in [
            "Capacity (MW)",
            "Demand Total (GWh) - Excluding Electrolysis",
            "Storage Capacity (GWh)",
            "CO2 Emissions (Mt)",
        ]:
            add_check(rows, f"ES1_has_{variable}", "PASS" if variable in es1_variables else "FAIL")

    if {"Pathway", "Variable", "2030"}.issubset(es1.columns):
        capacity = es1[es1["Variable"] == "Capacity (MW)"].copy()
        capacity["2030_numeric"] = pd.to_numeric(capacity["2030"], errors="coerce")
        for pathway, group in capacity.groupby("Pathway", dropna=True):
            add_metric(rows, f"ES1_capacity_mw_total_{pathway}", round(float(group["2030_numeric"].sum()), 6))

    if {"Scenario", "Technology", "Variable"}.issubset(es5.columns):
        technologies = unique_values(es5, "Technology")
        expected_tech = {"Solar", "Offshore Wind", "Onshore Wind", "Batteries", "Storage"}
        missing_tech = sorted(expected_tech - technologies)
        add_check(rows, "ES5_expected_technologies", "PASS" if not missing_tech else "FAIL", ", ".join(missing_tech))

    add_metric(rows, "reference_rows", len(reference))
    add_metric(rows, "reference_sheets", ",".join(sorted(unique_values(reference, "source_sheet"))))

    return pd.DataFrame(rows, columns=["kind", "name", "status", "value", "details"])


def write_outputs(
    workbook: Path | str,
    outputs: dict[str, Path | str],
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract CP30 data and write all Stage 1 CSV outputs."""
    workbook = Path(workbook)
    output_paths = {key: Path(value) for key, value in outputs.items()}

    if logger:
        logger.info(f"Reading CP30 workbook: {workbook}")

    tables = read_cp30_core_tables(workbook)
    reference = read_reference_tables(workbook)
    interconnector_prices = read_cp30_interconnector_prices(workbook)
    validation = build_validation_report(workbook, tables, reference)

    dataframes = {
        "es1": tables["es1"],
        "es5": tables["es5"],
        "ed1": tables["ed1"],
        "reference": reference,
        "validation": validation,
        "interconnector_prices": interconnector_prices,
    }

    for key, df in dataframes.items():
        path = output_paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        if logger:
            logger.info(f"Wrote {key}: {path} ({len(df)} rows, {len(df.columns)} columns)")

    return dataframes


def outputs_from_snakemake(snakemake_obj) -> dict[str, Path]:
    return {
        "es1": Path(snakemake_obj.output.es1),
        "es5": Path(snakemake_obj.output.es5),
        "ed1": Path(snakemake_obj.output.ed1),
        "reference": Path(snakemake_obj.output.reference),
        "validation": Path(snakemake_obj.output.validation),
        "interconnector_prices": Path(snakemake_obj.output.interconnector_prices),
    }


def main() -> None:
    snk = globals().get("snakemake")
    if snk is not None:
        log_path = snk.log[0] if getattr(snk, "log", None) else None
        logger = get_logger(log_path)
        workbook = Path(snk.input.workbook)
        outputs = outputs_from_snakemake(snk)
    else:
        logger = get_logger("CP30_data")
        workbook = DEFAULT_WORKBOOK
        outputs = DEFAULT_OUTPUTS

    write_outputs(workbook, outputs, logger)


if __name__ == "__main__":
    main()

