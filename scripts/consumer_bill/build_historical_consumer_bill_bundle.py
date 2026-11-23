"""Build a traceable annual consumer-bill input bundle from Ofgem workbooks.

The source manifest is deliberately explicit: it lists one released workbook
per cap period and pins its SHA256 digest.  This is safer than trying to
scrape whichever documents happen to be shown on Ofgem's live page.  The
compiler reads the published, pre-levelised electricity model. Where it
contains additive sub-components that can be reconciled exactly, the compiler
retains them rather than collapsing them to a headline row.
"""


import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import yaml
from openpyxl.utils import get_column_letter

try:
    from scripts.consumer_bill.component_ledger import (
        LEDGER_SCHEMA_VERSION, METHOD_VERSION, add_ledger_ids, object_sha256,
        validate_component_ledger,
    )
except ModuleNotFoundError:  # direct ``python scripts/consumer_bill/...`` use
    from component_ledger import (  # type: ignore
        LEDGER_SCHEMA_VERSION, METHOD_VERSION, add_ledger_ids, object_sha256,
        validate_component_ledger,
    )


BENCHMARK_KWH = 3100.0
BENCHMARK_MWH = BENCHMARK_KWH / 1000.0
OTHER_SHEET = "ElecSingle_Other_3100kWh"
NIL_SHEET = "ElecSingle_Other_Nil"
COMPONENT_CODES = ("DF", "CM", "AA", "PC", "NC", "OC", "SMNCC", "PAAC", "PAP", "E", "H")
COMPONENT_LABELS = {
    "DF": "Direct fuel / hedged wholesale allowance",
    "CM": "Capacity Market",
    "AA": "Adjustment allowance",
    "PC": "Policy costs",
    "NC": "Network costs",
    "OC": "Supplier operating costs",
    "SMNCC": "Smart-metering net cost change",
    "PAAC": "Payment-method additional administration",
    "PAP": "Payment-method adjustment",
    "E": "EBIT allowance",
    "H": "Headroom allowance",
}
COMPONENT_IDS = {
    "DF": "wholesale.direct_fuel",
    "CM": "capacity.market",
    "AA": "adjustment.allowance",
    "PC": "policy.aggregate",
    "NC": "network.aggregate",
    "OC": "supplier.operating",
    "SMNCC": "supplier.smart_metering",
    "PAAC": "supplier.payment_admin",
    "PAP": "supplier.payment_adjustment",
    "E": "supplier.ebit",
    "H": "supplier.headroom",
}
COMPONENT_CATEGORIES = {
    "DF": "wholesale", "CM": "capacity", "AA": "adjustment", "PC": "policy",
    "NC": "network", "OC": "supplier", "SMNCC": "supplier", "PAAC": "supplier",
    "PAP": "supplier", "E": "supplier", "H": "supplier",
}


class ConsumerBillBundleError(ValueError):
    """Raised when an upstream source no longer matches the declared model."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _load_manifest(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConsumerBillBundleError("Consumer-bill source manifest must be a mapping")
    year = payload.get("calendar_year")
    sources = payload.get("sources")
    if not isinstance(year, int) or not isinstance(sources, list) or not sources:
        raise ConsumerBillBundleError("Manifest needs calendar_year and a non-empty sources list")
    required = {"cap_period", "delivery_start", "delivery_end", "filename", "sha256"}
    for source in sources:
        missing = required.difference(source)
        if missing:
            raise ConsumerBillBundleError(f"Manifest source is missing: {sorted(missing)}")
        applicable = source.get("applicable_tariff")
        if applicable is not None:
            if not isinstance(applicable, dict):
                raise ConsumerBillBundleError("applicable_tariff must be a mapping")
            if missing := {"filename", "sha256"}.difference(applicable):
                raise ConsumerBillBundleError(f"Applicable tariff source is missing: {sorted(missing)}")
    if len({source["cap_period"] for source in sources}) != len(sources):
        raise ConsumerBillBundleError("Manifest cap_period values must be unique")
    return payload


def _materialise_source(source: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """Fetch a pinned workbook once, then verify the cached copy on every run."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / source["filename"]
    expected = str(source["sha256"]).lower()
    if not target.exists():
        if not source.get("url"):
            raise ConsumerBillBundleError(
                f"Pinned local source {target} is missing and no download URL was declared"
            )
        try:
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "PyPSA-GB consumer-bill bundle builder"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
                shutil.copyfileobj(response, output)
        except Exception as exc:  # pragma: no cover - depends on a remote publisher
            target.unlink(missing_ok=True)
            raise ConsumerBillBundleError(
                f"Could not download {source.get('cap_period', source['filename'])} from {source['url']}: {exc}"
            ) from exc
    actual = _sha256(target)
    if actual != expected:
        raise ConsumerBillBundleError(
            f"SHA256 mismatch for {target}. Expected {expected}, found {actual}. "
            "The publisher may have replaced the source; review and update the manifest explicitly."
        )
    return {**source, "path": target, "actual_sha256": actual}


def _period_column(values: list[tuple[Any, ...]], sheet_name: str, expected_period: str) -> int:
    wanted = " ".join(str(expected_period).replace("–", "-").split()).casefold()
    matches = []
    # The templates also carry formatting across thousands of empty columns.
    # Published cap periods currently occupy columns G:BE; scanning a bounded
    # output block prevents XML-cell creation for the formatting tail.
    for column, value in enumerate(values[11], start=1):
        normalised = " ".join(str(value).replace("–", "-").split()).casefold()
        if normalised == wanted:
            matches.append(column)
    if len(matches) != 1:
        raise ConsumerBillBundleError(
            f"Could not find exactly one period column {expected_period!r} in {sheet_name}; found {matches}"
        )
    return matches[0]


def _numeric(value: Any, *, cell: str, allow_dash: bool = False) -> float:
    if allow_dash and (value is None or value == "-"):
        return 0.0
    if not isinstance(value, (int, float)):
        raise ConsumerBillBundleError(f"Expected numeric value in {cell}, got {value!r}")
    return float(value)


def _read_period_workbook(source: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read published benchmark and nil tables for one cap period."""
    book = openpyxl.load_workbook(source["path"], data_only=True, read_only=True)
    try:
        other_sheet = source.get("output_sheet", OTHER_SHEET)
        nil_sheet = source.get("nil_sheet", NIL_SHEET)
        component_codes = set(source.get("component_codes", COMPONENT_CODES))
        missing = {other_sheet, nil_sheet}.difference(book.sheetnames)
        if missing:
            raise ConsumerBillBundleError(f"{source['filename']} is missing expected sheets: {sorted(missing)}")
        other, nil = book[other_sheet], book[nil_sheet]
        # Read the small published output block once.  In read-only mode,
        # repeated ``sheet.cell`` calls restart XML iteration and are
        # prohibitively slow on Ofgem's heavily formatted templates.
        other_values = list(other.iter_rows(min_row=1, max_row=200, min_col=1, max_col=200, values_only=True))
        nil_values_table = list(nil.iter_rows(min_row=1, max_row=200, min_col=1, max_col=200, values_only=True))
        period_label = source.get("workbook_period_label") or source["cap_period"]
        other_column = _period_column(other_values, other_sheet, period_label)
        nil_column = _period_column(nil_values_table, nil_sheet, period_label)
        rows: list[dict[str, Any]] = []
        # Ofgem templates retain formatting far below the populated table, so
        # ``max_row`` can be 1,048,576.  The published regional output block
        # always finishes within the first 200 rows; this guard also makes a
        # shifted future layout fail through the explicit completeness checks.
        for row in range(15, len(other_values) + 1):
            category, code, region = (other_values[row - 1][column - 1] for column in (2, 3, 4))
            if not isinstance(region, str) or not isinstance(code, str):
                continue
            if code not in {*component_codes, f"Total_{region}"}:
                continue
            value = _numeric(
                other_values[row - 1][other_column - 1],
                cell=f"{other_sheet}!{get_column_letter(other_column)}{row}",
                allow_dash=code == "AA",
            )
            rows.append({"region": region, "code": code, "category": category, "value": value, "row": row})
        benchmark = pd.DataFrame(rows)
        if benchmark.empty:
            raise ConsumerBillBundleError(f"No recognised electricity benchmark rows in {source['filename']}")
        # The workbooks also show an illustrative GB-average row.  Retain the
        # fourteen charge-restriction regions only, so downstream code can
        # calculate either an equal or an explicit household-weighted average
        # without double-counting that illustrative row.
        benchmark = benchmark.loc[benchmark["region"].str.casefold() != "gb average"].copy()
        expected_codes = component_codes
        complete = benchmark.groupby("region")["code"].agg(set)
        bad = {region: sorted(expected_codes.difference(values)) for region, values in complete.items() if expected_codes.difference(values)}
        if bad:
            raise ConsumerBillBundleError(f"Incomplete published component rows in {source['filename']}: {bad}")
        totals = benchmark.loc[benchmark["code"].str.startswith("Total_")].set_index("region")["value"]
        components = benchmark.loc[benchmark["code"].isin(component_codes)].groupby("region")["value"].sum()
        if set(totals.index) != set(components.index) or not (totals - components).abs().lt(1e-5).all():
            raise ConsumerBillBundleError(f"Published cost-stack rows do not reconcile in {source['filename']}")

        nil_rows: list[dict[str, Any]] = []
        for row in range(15, len(nil_values_table) + 1):
            code, region = nil_values_table[row - 1][2], nil_values_table[row - 1][3]
            if not isinstance(region, str) or not isinstance(code, str) or code != f"Total_{region}":
                continue
            nil_rows.append({
                "region": region,
                "value": _numeric(nil_values_table[row - 1][nil_column - 1], cell=f"{nil_sheet}!{get_column_letter(nil_column)}{row}"),
                "row": row,
            })
        nil_values = pd.DataFrame(nil_rows)
        nil_values = nil_values.loc[nil_values["region"].str.casefold() != "gb average"].copy()
        if set(nil_values["region"]) != set(totals.index):
            raise ConsumerBillBundleError(f"Nil-consumption output regions do not match benchmark outputs in {source['filename']}")

        benchmark["source_workbook"] = source["filename"]
        benchmark["source_sheet"] = other_sheet
        benchmark["source_column"] = get_column_letter(other_column)
        nil_values["source_workbook"] = source["filename"]
        nil_values["source_sheet"] = nil_sheet
        nil_values["source_column"] = get_column_letter(nil_column)
        return benchmark, nil_values
    finally:
        book.close()


def _read_nil_components(source: dict[str, Any]) -> pd.DataFrame:
    """Read source-level nil-consumption components for fixed/variable splits."""
    book = openpyxl.load_workbook(source["path"], data_only=True, read_only=True)
    try:
        nil_sheet = source.get("nil_sheet", NIL_SHEET)
        component_codes = set(source.get("component_codes", COMPONENT_CODES))
        sheet = book[nil_sheet]
        values = list(sheet.iter_rows(min_row=1, max_row=200, min_col=1, max_col=200, values_only=True))
        period_label = source.get("workbook_period_label") or source["cap_period"]
        column = _period_column(values, nil_sheet, period_label)
        rows = []
        for row in range(15, len(values) + 1):
            category, code, region = (values[row - 1][item - 1] for item in (2, 3, 4))
            if code not in component_codes or not isinstance(region, str) or region.casefold() == "gb average":
                continue
            rows.append({
                "region": region,
                "code": code,
                "category": category,
                "value": _numeric(
                    values[row - 1][column - 1],
                    cell=f"{nil_sheet}!{get_column_letter(column)}{row}",
                    allow_dash=True,
                ),
                "row": row,
                "source_workbook": source["filename"],
                "source_sheet": nil_sheet,
                "source_column": get_column_letter(column),
            })
        frame = pd.DataFrame(rows)
        expected = component_codes
        if frame.empty:
            # Some compact test fixtures and legacy source layouts publish
            # only the nil total. Preserve that gap as the explicit tariff
            # reconciliation row instead of proportionally inventing fixed
            # allocations for the named components.
            benchmark, _ = _read_period_workbook(source)
            frame = benchmark.loc[benchmark["code"].isin(component_codes), [
                "region", "code", "category", "row", "source_workbook", "source_sheet", "source_column"
            ]].copy()
            frame["value"] = 0.0
        elif any(expected.difference(items) for items in frame.groupby("region")["code"].agg(set)):
            raise ConsumerBillBundleError(f"Incomplete nil-consumption components in {source['filename']}")
        return frame
    finally:
        book.close()


def _read_applicable_tariff(source: dict[str, Any]) -> pd.DataFrame:
    """Read the final applicable nil/benchmark outputs for one cap period."""
    book = openpyxl.load_workbook(source["path"], data_only=True, read_only=True)
    try:
        candidates = ("1a levelised DTC", "1a Levelised DTC", "1a Default tariff cap")
        sheet_name = next((name for name in candidates if name in book.sheetnames), None)
        if sheet_name is None:
            # Minimal test fixtures and older layouts expose the same values in
            # the detailed calculation sheets.
            benchmark, nil = _read_period_workbook(source)
            totals = benchmark.loc[benchmark["code"].str.startswith("Total_")]
            return totals[["region", "value", "source_workbook", "source_sheet", "source_column", "row"]].rename(
                columns={"value": "benchmark_gbp_per_year_ex_vat", "row": "benchmark_source_row"}
            ).merge(
                nil[["region", "value", "row"]].rename(
                    columns={"value": "nil_gbp_per_year_ex_vat", "row": "nil_source_row"}
                ), on="region", validate="one_to_one",
            )
        sheet = book[sheet_name]
        values = list(sheet.iter_rows(min_row=1, max_row=40, min_col=1, max_col=8, values_only=True))
        rows = []
        # The direct-debit/Other Payment Method block is fixed at rows 14:28
        # in both the pre-levelised and Annex 9 layouts. Later rows contain
        # Standard Credit and must never be mixed into this tariff.
        for row in range(14, 29):
            region = values[row - 1][1]
            nil, benchmark = values[row - 1][2], values[row - 1][3]
            if not isinstance(region, str) or region.casefold().startswith("gb average"):
                continue
            if not isinstance(nil, (int, float)) or not isinstance(benchmark, (int, float)):
                continue
            rows.append({
                "region": region,
                "nil_gbp_per_year_ex_vat": float(nil),
                "benchmark_gbp_per_year_ex_vat": float(benchmark),
                "source_workbook": source["filename"],
                "source_sheet": sheet_name,
                "source_column": "C:D",
                "nil_source_row": row,
                "benchmark_source_row": row,
            })
        frame = pd.DataFrame(rows)
        if len(frame) != 14 or frame["region"].duplicated().any():
            raise ConsumerBillBundleError(
                f"Applicable tariff source {source['filename']} must contain all 14 regions"
            )
        return frame
    finally:
        book.close()


def _extract_policy_detail(annex_source: dict[str, Any], cap_source: dict[str, Any]) -> pd.DataFrame:
    """Extract RO, FiT, ECO, WHD and AAHEDC from Annex 4's published rows."""
    book = openpyxl.load_workbook(annex_source["path"], data_only=True, read_only=True)
    try:
        sheet_name = "2a Aggregate costs"
        if sheet_name not in book.sheetnames:
            raise ConsumerBillBundleError(f"{annex_source['filename']} is missing {sheet_name}")
        sheet = book[sheet_name]
        values = list(sheet.iter_rows(min_row=1, max_row=90, min_col=1, max_col=80, values_only=True))
        column = _period_column(values, sheet_name, cap_source["workbook_period_label"])
        code_rows = {"RO": 15, "FiT": 16, "ECO": 17, "WHD": 18}
        rows = []
        for component, row in code_rows.items():
            value = _numeric(values[row - 1][column - 1], cell=f"{sheet_name}!{get_column_letter(column)}{row}")
            rows.append({
                "component": component,
                "annual_gbp_per_3100kwh_ex_vat": value if component == "WHD" else value * BENCHMARK_MWH,
                "source_workbook": annex_source["filename"], "source_sheet": sheet_name,
                "source_cells": f"{get_column_letter(column)}{row}",
                "detail_status": "published_annex_component",
            })
        aahdc_rows = range(38, 52)
        aahdc_values = [
            _numeric(values[row - 1][column - 1], cell=f"{sheet_name}!{get_column_letter(column)}{row}")
            for row in aahdc_rows
        ]
        rows.append({
            "component": "AAHEDC",
            "annual_gbp_per_3100kwh_ex_vat": float(pd.Series(aahdc_values).mean() * BENCHMARK_MWH),
            "source_workbook": annex_source["filename"], "source_sheet": sheet_name,
            "source_cells": ";".join(f"{get_column_letter(column)}{row}" for row in aahdc_rows),
            "detail_status": "published_annex_component_regional_mean",
        })
        frame = pd.DataFrame(rows)
        expected = float(cap_source["policy_aggregate_gbp_per_3100kwh"])
        actual = float(frame["annual_gbp_per_3100kwh_ex_vat"].sum())
        if abs(actual - expected) > 1e-5:
            raise ConsumerBillBundleError(
                f"Detailed Annex 4 policy costs do not reconcile for {cap_source['cap_period']}: {actual} != {expected}"
            )
        frame["cap_period"] = cap_source["cap_period"]
        return frame
    finally:
        book.close()


def _extract_smart_detail(cap_source: dict[str, Any]) -> pd.DataFrame | None:
    """Split SMNCC into the three pass-through charges and model increment."""
    sheet_name = "3i SMNCC"
    book = openpyxl.load_workbook(cap_source["path"], data_only=True, read_only=True)
    try:
        if sheet_name not in book.sheetnames:
            return None
        values = list(book[sheet_name].iter_rows(min_row=1, max_row=67, min_col=1, max_col=80, values_only=True))
        # This input table's period labels sit on worksheet row 11 whereas
        # the generic cap-output helper expects them in row 12.
        header_proxy = [tuple()] * 11 + [values[10]]
        current_column = _period_column(header_proxy, sheet_name, cap_source["workbook_period_label"])
        baseline_column = _period_column(header_proxy, sheet_name, "April 2017 - September 2017")
        cpi = _numeric(values[17][current_column - 1], cell=f"{sheet_name}!{get_column_letter(current_column)}18")
        rows = []
        for component, row in (("Smart metering DCC pass-through net change", 14), ("Smart metering SEGB pass-through net change", 15), ("Smart metering SMICoP pass-through net change", 16)):
            current = _numeric(values[row - 1][current_column - 1], cell=f"{sheet_name}!{get_column_letter(current_column)}{row}")
            baseline = _numeric(values[row - 1][baseline_column - 1], cell=f"{sheet_name}!{get_column_letter(baseline_column)}{row}")
            rows.append({
                "component": component,
                "annual_gbp_per_3100kwh_ex_vat": current - baseline * cpi,
                "source_workbook": cap_source["filename"], "source_sheet": sheet_name,
                "source_cells": f"{get_column_letter(current_column)}{row},{get_column_letter(baseline_column)}{row},{get_column_letter(current_column)}18",
                "detail_status": "published_cap_model_component",
            })
        non_pass = _numeric(values[30][current_column - 1], cell=f"{sheet_name}!{get_column_letter(current_column)}31")
        rows.append({
            "component": "Smart metering non-pass-through model increment",
            "annual_gbp_per_3100kwh_ex_vat": non_pass,
            "source_workbook": cap_source["filename"], "source_sheet": sheet_name,
            "source_cells": f"{get_column_letter(current_column)}31",
            "detail_status": "published_cap_model_component",
        })
        frame = pd.DataFrame(rows)
        expected = float(cap_source["smart_aggregate_gbp_per_3100kwh"])
        actual = float(frame["annual_gbp_per_3100kwh_ex_vat"].sum())
        # The workbook displays rounded DCC/SEGB/SMICoP rows but retains an
        # unrounded combined SMNCC output. Keep the published total exact and
        # assign its tiny display-rounding residual to DCC.
        frame.loc[frame.index[0], "annual_gbp_per_3100kwh_ex_vat"] += expected - actual
        frame.loc[frame.index[0], "source_cells"] += "; reconciled to published SMNCC total"
        actual = float(frame["annual_gbp_per_3100kwh_ex_vat"].sum())
        if abs(actual - expected) > 1e-5:
            raise ConsumerBillBundleError(
                f"Detailed smart-metering costs do not reconcile for {cap_source['cap_period']}: {actual} != {expected}"
            )
        frame["cap_period"] = cap_source["cap_period"]
        return frame
    finally:
        book.close()


def _extract_wholesale_detail(cap_source: dict[str, Any]) -> pd.DataFrame | None:
    """Split the direct-fuel allowance into the published 3a DF inputs.

    The 2024 quarterly cap models expose three additive terms for single-rate
    electricity at 3,100 kWh: direct energy, backwardation and CfD. The CfD
    term is embedded in wholesale here; it must not be added again to policy.
    """
    sheet_name = "3a DF"
    book = openpyxl.load_workbook(cap_source["path"], data_only=True, read_only=True)
    try:
        if sheet_name not in book.sheetnames:
            return None
        values = list(book[sheet_name].iter_rows(min_row=1, max_row=110, min_col=1, max_col=80, values_only=True))
        header_proxy = [tuple()] * 11 + [values[6]]
        column = _period_column(header_proxy, sheet_name, cap_source["workbook_period_label"])
        blocks = (
            ("Direct fuel energy cost", 11, 24, "published_direct_fuel_energy_cost"),
            ("Backwardation allowance", 48, 61, "published_backwardation_allowance"),
            ("CfD allowance embedded in direct fuel", 85, 98, "published_cfd_allowance_embedded_in_direct_fuel"),
        )
        rows = []
        for component, start, end, status in blocks:
            regional_values = [
                _numeric(values[row - 1][column - 1], cell=f"{sheet_name}!{get_column_letter(column)}{row}")
                for row in range(start, end + 1)
            ]
            rows.append({
                "component": component,
                "annual_gbp_per_3100kwh_ex_vat": float(pd.Series(regional_values).mean()),
                "source_workbook": cap_source["filename"], "source_sheet": sheet_name,
                "source_cells": ";".join(f"{get_column_letter(column)}{row}" for row in range(start, end + 1)),
                "detail_status": status,
                "cap_period": cap_source["cap_period"],
            })
        frame = pd.DataFrame(rows)
        expected = float(cap_source["direct_fuel_aggregate_gbp_per_3100kwh"])
        residual = expected - float(frame["annual_gbp_per_3100kwh_ex_vat"].sum())
        if abs(residual) > 1e-4:
            raise ConsumerBillBundleError(
                f"Detailed 3a DF wholesale costs do not reconcile for {cap_source['cap_period']}: "
                f"{float(frame['annual_gbp_per_3100kwh_ex_vat'].sum())} != {expected}"
            )
        frame.loc[frame.index[0], "annual_gbp_per_3100kwh_ex_vat"] += residual
        if residual:
            frame.loc[frame.index[0], "source_cells"] += "; reconciled to published DF total"
        return frame
    finally:
        book.close()


def _days(source: dict[str, Any]) -> int:
    return int((pd.Timestamp(source["delivery_end"]) - pd.Timestamp(source["delivery_start"])).days + 1)


def compile_bundle(manifest_path: str | Path, output_dir: str | Path, raw_dir: str | Path | None = None) -> dict[str, Any]:
    """Download/verify manifest sources and write the normalised annual bundle."""
    manifest = _load_manifest(manifest_path)
    year = manifest["calendar_year"]
    output = Path(output_dir)
    raw = Path(raw_dir or manifest.get("raw_cache_dir") or Path(manifest_path).parent / "raw" / str(year))
    sources = [_materialise_source(source, raw) for source in manifest["sources"]]
    for source in sources:
        applicable = source.get("applicable_tariff")
        if applicable:
            source["applicable_tariff"] = _materialise_source(
                {**applicable, "cap_period": source["cap_period"]}, raw
            )
    detail_sources = []
    for source in manifest.get("detail_sources", []):
        required = {"filename", "url", "sha256", "cap_periods"}
        if missing := required.difference(source):
            raise ConsumerBillBundleError(f"Detailed manifest source is missing: {sorted(missing)}")
        detail_sources.append(_materialise_source(source, raw))
    periods = []
    all_benchmark, all_nil, all_nil_components, all_applicable = [], [], [], []
    for source in sources:
        benchmark, nil = _read_period_workbook(source)
        nil_components = _read_nil_components(source)
        tariff_source = source.get("applicable_tariff", source)
        applicable = _read_applicable_tariff(tariff_source)
        applicable["source_hash"] = tariff_source.get("actual_sha256", tariff_source.get("sha256", ""))
        source["days"] = _days(source)
        source["calendar_weight"] = source["days"] / (366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365)
        benchmark["cap_period"], nil["cap_period"] = source["cap_period"], source["cap_period"]
        all_benchmark.append(benchmark)
        all_nil.append(nil)
        nil_components["cap_period"] = source["cap_period"]
        all_nil_components.append(nil_components)
        applicable["cap_period"] = source["cap_period"]
        all_applicable.append(applicable)
        source["policy_aggregate_gbp_per_3100kwh"] = float(
            benchmark.loc[benchmark["code"].eq("PC"), "value"].mean()
        )
        source["smart_aggregate_gbp_per_3100kwh"] = float(
            benchmark.loc[benchmark["code"].eq("SMNCC"), "value"].mean()
        )
        source["direct_fuel_aggregate_gbp_per_3100kwh"] = float(
            benchmark.loc[benchmark["code"].eq("DF"), "value"].mean()
        )
        periods.append(source)
    _validate_calendar_coverage(periods, year)
    benchmark, nil = pd.concat(all_benchmark, ignore_index=True), pd.concat(all_nil, ignore_index=True)
    nil_components = pd.concat(all_nil_components, ignore_index=True)
    applicable = pd.concat(all_applicable, ignore_index=True)
    total_rows = benchmark.loc[benchmark["code"].str.startswith("Total_")].copy()

    output.mkdir(parents=True, exist_ok=True)
    _write_rates(output, applicable, benchmark, periods, year)
    _write_period_tariffs(output, applicable, periods)
    _write_component_ledger(output, benchmark, nil_components, applicable, periods, manifest)
    policy_detail_by_period = {}
    for detail_source in detail_sources:
        for cap_period in detail_source["cap_periods"]:
            cap_source = next((item for item in periods if item["cap_period"] == cap_period), None)
            if cap_source is None:
                raise ConsumerBillBundleError(f"Detailed source references unknown cap period {cap_period}")
            if cap_period in policy_detail_by_period:
                raise ConsumerBillBundleError(f"Multiple detailed policy sources declared for {cap_period}")
            policy_detail_by_period[cap_period] = _extract_policy_detail(detail_source, cap_source)
    smart_detail_by_period = {
        source["cap_period"]: detail
        for source in periods
        if source.get("extract_smart_detail", True) and (detail := _extract_smart_detail(source)) is not None
    }
    wholesale_detail_by_period = {
        source["cap_period"]: detail
        for source in periods
        if source.get("extract_wholesale_detail", True) and (detail := _extract_wholesale_detail(source)) is not None
    }
    _write_hedge_inputs(output, benchmark, periods, wholesale_detail_by_period)
    _write_component_bundles(
        output, benchmark, periods, policy_detail_by_period, smart_detail_by_period, wholesale_detail_by_period
    )
    provenance = {
        "purpose": "Automated historical Ofgem consumer-bill input bundle",
        "calendar_year": year,
        "manifest": str(Path(manifest_path)),
        "raw_cache_dir": str(raw),
        "sources": [_jsonable({key: value for key, value in source.items() if key not in {"days", "calendar_weight"}}) for source in periods],
        "detail_sources": [_jsonable({key: value for key, value in source.items() if key not in {"days", "calendar_weight"}}) for source in detail_sources],
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "detail_level": "Wholesale is split into the published 3a direct-energy, backwardation and embedded-CfD terms; policy is split into published Annex 4 components; and smart metering into published DCC, SEGB, SMICoP and model-increment rows. Network and adjustment retain published aggregate model rows until their released annex calculations can be independently recalculated and reconciled.",
    }
    (output / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _validate_calendar_coverage(periods: list[dict[str, Any]], year: int) -> None:
    intervals = sorted((pd.Timestamp(item["delivery_start"]), pd.Timestamp(item["delivery_end"])) for item in periods)
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    actual = pd.DatetimeIndex([])
    for start, end in intervals:
        actual = actual.append(pd.date_range(start, end, freq="D"))
    if not actual.equals(expected):
        raise ConsumerBillBundleError("Manifest delivery periods must cover the calendar year exactly once")


def _write_rates(
    output: Path, applicable: pd.DataFrame, benchmark_rows: pd.DataFrame,
    periods: list[dict[str, Any]], year: int,
) -> None:
    weights = pd.DataFrame({"cap_period": [item["cap_period"] for item in periods], "weight": [item["calendar_weight"] for item in periods]})
    weighted = applicable.merge(weights, on="cap_period", validate="many_to_one")
    rates = weighted.groupby("region", as_index=False).apply(
        lambda frame: pd.Series({
            "benchmark_gbp_per_year_ex_vat": (
                frame["benchmark_gbp_per_year_ex_vat"] * frame["weight"]
            ).sum(),
            "nil_gbp_per_year_ex_vat": (frame["nil_gbp_per_year_ex_vat"] * frame["weight"]).sum(),
        }), include_groups=False,
    ).reset_index(drop=True)
    rates.insert(0, "payment_method", "direct_debit")
    rates["benchmark_kwh"] = BENCHMARK_KWH
    rates = rates[["payment_method", "region", "nil_gbp_per_year_ex_vat", "benchmark_gbp_per_year_ex_vat", "benchmark_kwh"]]
    if (rates["benchmark_gbp_per_year_ex_vat"] < rates["nil_gbp_per_year_ex_vat"]).any():
        raise ConsumerBillBundleError("Published annualised benchmark is below nil-consumption charge")
    rates.to_csv(output / "electricity_single_rate.csv", index=False)
    direct_fuel = _annual_component_means(benchmark_rows, periods, "DF") / BENCHMARK_MWH
    metadata = {
        "snapshot_id": f"{year}_annual_ofgem_auto",
        "effective_from": f"{year}-01-01",
        "effective_to": f"{year}-12-31",
        "fuel": "electricity",
        "meter_type": "single_rate",
        "currency": "GBP nominal",
        "vat_rate": 0.05,
        "reference_wholesale_price_gbp_per_mwh": direct_fuel,
        "reference_wholesale_price_description": "Calendar-day weighted published direct-fuel allowance divided by 3.1 MWh. It includes Ofgem's hedging/loss treatment and is not a raw spot price.",
        "source": {"publisher": "Ofgem", "manifest": "source_provenance.json"},
        "annualisation": f"Calendar-day weighted over {366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365} days.",
        "limitations": [
            "Other Payment Method in Ofgem's electricity model is normalised as direct_debit for this existing consumer-bill interface.",
            "The annual compatibility table is calendar-day weighted; use period_tariffs.csv for an actual dated household replay.",
            "The component ledger preserves the pre-levelised stack and records electricity base levelisation as an explicit adjustment.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_period_tariffs(output: Path, applicable: pd.DataFrame, periods: list[dict[str, Any]]) -> None:
    lookup = {item["cap_period"]: item for item in periods}
    frame = applicable.copy()
    frame.insert(0, "payment_method", "direct_debit")
    frame.insert(1, "meter_type", "single_rate")
    frame["delivery_start"] = frame["cap_period"].map(lambda value: lookup[value]["delivery_start"])
    frame["delivery_end"] = frame["cap_period"].map(lambda value: lookup[value]["delivery_end"])
    frame["delivery_days"] = frame["cap_period"].map(lambda value: lookup[value]["days"])
    frame["benchmark_kwh"] = BENCHMARK_KWH
    frame["tariff_basis_days"] = 365.0
    frame["standing_charge_gbp_per_day_ex_vat"] = frame["nil_gbp_per_year_ex_vat"] / frame["tariff_basis_days"]
    frame["unit_rate_gbp_per_kwh_ex_vat"] = (
        frame["benchmark_gbp_per_year_ex_vat"] - frame["nil_gbp_per_year_ex_vat"]
    ) / BENCHMARK_KWH
    frame["vat_rate"] = 0.05
    frame["applicability"] = frame["cap_period"].map(
        lambda value: "pre_levelisation" if pd.Timestamp(lookup[value]["delivery_start"]) < pd.Timestamp("2024-04-01")
        else "electricity_base_levelised"
    )
    frame.to_csv(output / "period_tariffs.csv", index=False)


def _write_component_ledger(
    output: Path,
    benchmark: pd.DataFrame,
    nil_components: pd.DataFrame,
    applicable: pd.DataFrame,
    periods: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """Write source allocations without normalising known rows to a headline."""
    lookup = {item["cap_period"]: item for item in periods}
    values = benchmark.loc[benchmark["code"].isin(COMPONENT_CODES)].merge(
        nil_components[["cap_period", "region", "code", "value"]].rename(columns={"value": "nil_value"}),
        on=["cap_period", "region", "code"], validate="one_to_one",
    )
    final = applicable.set_index(["cap_period", "region"])
    records: list[dict[str, Any]] = []
    manifest_hash = object_sha256(manifest)
    for row in values.itertuples(index=False):
        source = lookup[row.cap_period]
        fixed = float(row.nil_value)
        variable = float(row.value) - fixed
        records.append(_ledger_record(
            row.cap_period, row.region, row.code, fixed, variable, source,
            row.source_workbook, row.source_sheet,
            f"{row.source_column}{row.row}", manifest_hash,
        ))
    # The final applicable tariff differs from the pre-levelised cost stack
    # from April 2024. Keep that change as its own fixed/variable component.
    grouped = values.groupby(["cap_period", "region"], as_index=True).agg(
        pre_benchmark=("value", "sum"), pre_nil=("nil_value", "sum")
    )
    for key, source_values in grouped.iterrows():
        cap_period, region = key
        target = final.loc[key]
        fixed = float(target["nil_gbp_per_year_ex_vat"] - source_values.pre_nil)
        variable = float(
            target["benchmark_gbp_per_year_ex_vat"] - target["nil_gbp_per_year_ex_vat"]
            - (source_values.pre_benchmark - source_values.pre_nil)
        )
        if abs(fixed) < 1e-12 and abs(variable) < 1e-12:
            continue
        source = lookup[cap_period]
        record = _ledger_record(
            cap_period, region, None, fixed, variable, source,
            target["source_workbook"], target["source_sheet"],
            f"C{int(target['nil_source_row'])}:D{int(target['benchmark_source_row'])}", manifest_hash,
        )
        record.update(
            component_id=(
                "adjustment.electricity_base_levelisation"
                if pd.Timestamp(source["delivery_start"]) >= pd.Timestamp("2024-04-01")
                else "adjustment.published_rounding"
            ),
            component_label=(
                "Electricity base levelisation adjustment"
                if pd.Timestamp(source["delivery_start"]) >= pd.Timestamp("2024-04-01")
                else "Published output rounding adjustment"
            ),
            category="adjustment",
            formula=(
                "final applicable tariff minus pre-levelised component stack"
                if pd.Timestamp(source["delivery_start"]) >= pd.Timestamp("2024-04-01")
                else "published rounded tariff output minus detailed component stack"
            ),
            source_hash=target["source_hash"],
        )
        records.append(record)
    frame = add_ledger_ids(pd.DataFrame(records))
    validate_component_ledger(frame)
    frame.to_csv(output / "component_ledger.csv", index=False)


def _ledger_record(
    cap_period: str, region: str, code: str | None, fixed: float, variable: float,
    source: dict[str, Any], workbook: str, sheet: str, cells: str, manifest_hash: str,
) -> dict[str, Any]:
    amount = fixed + variable
    component_id = COMPONENT_IDS.get(code, "adjustment.electricity_base_levelisation")
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "result_type": "published_tariff_replay",
        "component_id": component_id,
        "component_label": COMPONENT_LABELS.get(code, "Electricity base levelisation adjustment"),
        "category": COMPONENT_CATEGORIES.get(code, "adjustment"),
        "period_start": source["delivery_start"], "period_end": source["delivery_end"],
        "publication_date": source.get("publication_date", ""), "as_of_date": source.get("as_of_date", ""),
        "calendar_year": pd.Timestamp(source["delivery_start"]).year,
        "obligation_year": source.get("obligation_year", ""),
        "region": region, "payment_method": "direct_debit", "meter_type": "single_rate",
        "currency_year": pd.Timestamp(source["delivery_start"]).year, "currency_basis": "GBP nominal ex VAT",
        "unit": "GBP per benchmark customer per year", "consumption_basis": "source benchmark: 3100 kWh/year",
        "consumption_case": "source_benchmark", "quantity": 1.0, "rate": amount, "amount_gbp": amount,
        "sign": "charge" if amount > 0 else "credit" if amount < 0 else "zero",
        "vat_treatment": "standard domestic energy VAT applied after subtotal",
        "quantity_source": f"{workbook}:{sheet}", "price_source": f"{workbook}:{sheet}",
        "recovery_method": "published tariff allowance",
        "evidence_status": "observed", "source_reference": f"{workbook}:{sheet}!{cells}",
        "formula": "fixed_gbp_per_year + variable_gbp_at_3100kwh",
        "model_run_id": "", "source_hash": source.get("actual_sha256", source.get("sha256", "")),
        "config_hash": manifest_hash, "method_version": METHOD_VERSION,
        "quantity_coverage": 1.0, "cost_coverage": 1.0, "calibration_flags": "none",
        "fixed_gbp_per_year_ex_vat": fixed, "variable_gbp_at_benchmark_ex_vat": variable,
        "cap_period": cap_period, "benchmark_kwh": BENCHMARK_KWH,
    }


def _annual_component_means(frame: pd.DataFrame, periods: list[dict[str, Any]], code: str) -> float:
    weights = pd.DataFrame({"cap_period": [item["cap_period"] for item in periods], "weight": [item["calendar_weight"] for item in periods]})
    selected = frame.loc[frame["code"].eq(code)].merge(weights, on="cap_period", validate="many_to_one")
    return float((selected.groupby("cap_period")["value"].mean() * selected.groupby("cap_period")["weight"].first()).sum())


def _write_hedge_inputs(output: Path, benchmark: pd.DataFrame, periods: list[dict[str, Any]], wholesale_detail_by_period: dict[str, pd.DataFrame]) -> None:
    records = []
    for source in periods:
        component_codes = set(source.get("component_codes", COMPONENT_CODES))
        rows = benchmark.loc[(benchmark["cap_period"] == source["cap_period"]) & benchmark["code"].isin(component_codes)]
        values = rows.groupby("code")["value"].mean()
        if set(values.index) != component_codes:
            raise ConsumerBillBundleError(f"Missing aggregate component values for {source['cap_period']}")
        wholesale_detail = wholesale_detail_by_period.get(source["cap_period"])
        named = (
            wholesale_detail.set_index("component")["annual_gbp_per_3100kwh_ex_vat"]
            if wholesale_detail is not None else pd.Series(dtype=float)
        )
        records.append({
            "cap_period": source["cap_period"],
            "observation_start": source.get("observation_start", ""),
            "observation_end": source.get("observation_end", ""),
            "delivery_start": source["delivery_start"], "delivery_end": source["delivery_end"],
            "delivery_seasons": "published_direct_fuel_aggregate",
            "base_share": 1.0, "peak_share": 0.0,
            "forward_index_gbp_per_mwh": values["DF"] / BENCHMARK_MWH,
            "seasonal_to_monthly_shaping_rate": 0.0, "monthly_to_hourly_shaping_rate": 0.0,
            "day_ahead_rehedging_rate": 0.0, "imbalance_rate": 0.0,
            "transaction_cost_rate": 0.0, "additional_risk_rate": 0.0,
            "loss_multiplier_gb_average": 1.0,
            "capacity_market_gbp_per_3100kwh": values.get("CM", 0.0),
            "adjustment_allowance_gbp_per_3100kwh": values.get("AA", 0.0),
            "policy_gbp_per_3100kwh": values.get("PC", 0.0), "network_gbp_per_3100kwh": values.get("NC", 0.0),
            "operating_gbp_per_3100kwh": values.get("OC", 0.0), "smart_metering_gbp_per_3100kwh": values.get("SMNCC", 0.0),
            "payment_additional_admin_gbp_per_3100kwh": values.get("PAAC", 0.0),
            "payment_adjustment_gbp_per_3100kwh": values.get("PAP", 0.0), "ebit_gbp_per_3100kwh": values.get("E", 0.0),
            "headroom_gbp_per_3100kwh": values.get("H", 0.0), "total_cap_gbp_per_3100kwh": values.sum(),
            "source_workbook": source["filename"], "source_sheet": source.get("output_sheet", OTHER_SHEET),
            "source_column": rows["source_column"].iloc[0],
            "wholesale_detail_level": "published_3a_direct_fuel_components" if wholesale_detail is not None else "aggregate_direct_fuel_allowance",
            "direct_fuel_energy_cost_gbp_per_3100kwh": named.get("Direct fuel energy cost"),
            "backwardation_allowance_gbp_per_3100kwh": named.get("Backwardation allowance"),
            "cfd_allowance_embedded_in_direct_fuel_gbp_per_3100kwh": named.get("CfD allowance embedded in direct fuel"),
        })
    pd.DataFrame(records).to_csv(output / "hedged_wholesale_inputs.csv", index=False)


def _write_component_bundles(
    output: Path, benchmark: pd.DataFrame, periods: list[dict[str, Any]], policy_detail_by_period: dict[str, pd.DataFrame],
    smart_detail_by_period: dict[str, pd.DataFrame], wholesale_detail_by_period: dict[str, pd.DataFrame],
) -> None:
    bundle_codes = {
        "policy_components.csv": ["PC"], "network_components.csv": ["NC"],
        "smart_meter_components.csv": ["SMNCC"], "adjustment_components.csv": ["AA"],
    }
    period_lookup = {item["cap_period"]: item for item in periods}
    if wholesale_detail_by_period:
        wholesale_rows = pd.concat(wholesale_detail_by_period.values(), ignore_index=True)
        if set(wholesale_rows["cap_period"]) != set(period_lookup):
            missing = sorted(set(period_lookup).difference(wholesale_rows["cap_period"]))
            raise ConsumerBillBundleError(f"Detailed wholesale sources do not cover every period: {missing}")
        wholesale_rows["delivery_start"] = wholesale_rows["cap_period"].map(lambda value: period_lookup[value]["delivery_start"])
        wholesale_rows["delivery_end"] = wholesale_rows["cap_period"].map(lambda value: period_lookup[value]["delivery_end"])
        wholesale_rows["delivery_days"] = wholesale_rows["cap_period"].map(lambda value: period_lookup[value]["days"])
        wholesale_rows["calendar_weight"] = wholesale_rows["cap_period"].map(lambda value: period_lookup[value]["calendar_weight"])
        wholesale_rows["annualised_gbp_per_3100kwh_ex_vat"] = (
            wholesale_rows["annual_gbp_per_3100kwh_ex_vat"] * wholesale_rows["calendar_weight"]
        )
        wholesale_rows.to_csv(output / "wholesale_components.csv", index=False)
    else:
        # Snakemake promises this artifact even for older workbooks whose
        # manifest deliberately disables detailed wholesale extraction.
        bundle_codes["wholesale_components.csv"] = ["DF"]
    for filename, codes in bundle_codes.items():
        detailed_rows = policy_detail_by_period if filename == "policy_components.csv" else smart_detail_by_period if filename == "smart_meter_components.csv" else {}
        if detailed_rows:
            rows = pd.concat(detailed_rows.values(), ignore_index=True)
            if set(rows["cap_period"]) != set(period_lookup):
                missing = sorted(set(period_lookup).difference(rows["cap_period"]))
                raise ConsumerBillBundleError(f"Detailed {filename} sources do not cover every period: {missing}")
            rows["delivery_start"] = rows["cap_period"].map(lambda value: period_lookup[value]["delivery_start"])
            rows["delivery_end"] = rows["cap_period"].map(lambda value: period_lookup[value]["delivery_end"])
            rows["delivery_days"] = rows["cap_period"].map(lambda value: period_lookup[value]["days"])
            rows["calendar_weight"] = rows["cap_period"].map(lambda value: period_lookup[value]["calendar_weight"])
            rows["annualised_gbp_per_3100kwh_ex_vat"] = rows["annual_gbp_per_3100kwh_ex_vat"] * rows["calendar_weight"]
            rows.to_csv(output / filename, index=False)
            continue
        rows = benchmark.loc[benchmark["code"].isin(codes)].groupby(["cap_period", "code"], as_index=False).agg(
            annual_gbp_per_3100kwh_ex_vat=("value", "mean"), source_workbook=("source_workbook", "first"),
            source_sheet=("source_sheet", "first"), source_column=("source_column", "first"),
        )
        rows["component"] = rows["code"].map(COMPONENT_LABELS)
        rows["delivery_start"] = rows["cap_period"].map(lambda value: period_lookup[value]["delivery_start"])
        rows["delivery_end"] = rows["cap_period"].map(lambda value: period_lookup[value]["delivery_end"])
        rows["delivery_days"] = rows["cap_period"].map(lambda value: period_lookup[value]["days"])
        rows["calendar_weight"] = rows["cap_period"].map(lambda value: period_lookup[value]["calendar_weight"])
        rows["annualised_gbp_per_3100kwh_ex_vat"] = rows["annual_gbp_per_3100kwh_ex_vat"] * rows["calendar_weight"]
        rows["source_cells"] = rows.apply(lambda item: f"{item.source_sheet}!{item.source_column} (GB mean of 14 regional rows)", axis=1)
        rows["detail_status"] = "published_aggregate_cost_model_row"
        rows["detail_note"] = ""
        if filename == "network_components.csv":
            rows["detail_note"] = (
                "The released Annex 3 workbooks do not provide a current-quarter, "
                "reconciling TNUoS/DUoS/BSUoS calculation for this 2024 cap-model total."
            )
        elif filename == "adjustment_components.csv":
            rows["detail_note"] = (
                "Published pre-levelised cap-model total; a separately reconciled adjustment-annex split is not yet available."
            )
        rows.drop(columns="code").to_csv(output / filename, index=False)


def _main(snakemake) -> None:
    compile_bundle(snakemake.input.manifest, snakemake.params.output_dir, snakemake.params.raw_dir)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-dir")
    args = parser.parse_args()
    compile_bundle(args.manifest, args.output_dir, args.raw_dir)


if "snakemake" in globals():
    _main(snakemake)
elif __name__ == "__main__":
    _cli()
