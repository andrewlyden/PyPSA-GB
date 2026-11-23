"""Extract the published 2020 Ofgem policy-cost components from Annex 4.

The three supplied workbooks are the primary-source calculations behind the
2020 price-cap policy allowance.  In particular, CfD is a distinct Annex 4
Policy Cost Allowance component in 2020.  This script creates a small,
machine-readable annualised extract while retaining workbook, sheet and cell
provenance for every value.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd


BENCHMARK_MWH = 3.1
SHEET = "2a Aggregate costs"

SOURCE_PERIODS = (
    {
        "cap_period": "2020_Jan_Mar",
        "delivery_start": "2020-01-01",
        "delivery_end": "2020-03-31",
        "workbook": "2019_10_to_2020_03_annex_4_v1.4.xlsx",
        "column": "S",
    },
    {
        "cap_period": "2020_Apr_Sep",
        "delivery_start": "2020-04-01",
        "delivery_end": "2020-09-30",
        "workbook": "2020_04_to_2020_09_annex_4_v1.5.xlsx",
        "column": "T",
    },
    {
        "cap_period": "2020_Oct_Dec",
        "delivery_start": "2020-10-01",
        "delivery_end": "2020-12-31",
        "workbook": "2020_10_to_2021_03_annex_4_v1.7.xlsx",
        "column": "U",
    },
)

# Row 14--19 are supplier scheme rates.  CfD and AAHEDC are transmission/GSP
# rates, so Ofgem applies region-specific losses in rows 74--87 and 37--50.
COMPONENTS = (
    ("Renewables Obligation", "RO", "supplied_rate", 14),
    ("Contracts for Difference", "CfD", "regional_average_rate", range(74, 88)),
    ("Feed-in Tariff", "FiT", "supplied_rate", 16),
    ("Energy Company Obligation", "ECO", "supplied_rate", 17),
    ("Warm Home Discount", "WHD", "customer_allowance", 18),
    ("Assistance for Areas with High Electricity Distribution Costs", "AAHEDC", "regional_average_rate", range(37, 51)),
)


class OfgemPolicyExtractionError(ValueError):
    """Raised when primary-source policy values cannot be reconciled."""


def _numeric(value: Any, cell: str) -> float:
    if not isinstance(value, (int, float)):
        raise OfgemPolicyExtractionError(f"Expected a numeric value in {SHEET}!{cell}, got {value!r}")
    return float(value)


def _period_component_rows(source_dir: str | Path) -> pd.DataFrame:
    """Read Ofgem's single-rate electricity policy values for each 2020 period."""
    source_dir = Path(source_dir)
    rows: list[dict[str, Any]] = []
    for period in SOURCE_PERIODS:
        workbook_path = source_dir / period["workbook"]
        if not workbook_path.exists():
            raise OfgemPolicyExtractionError(f"Missing Ofgem Annex 4 workbook: {workbook_path}")
        workbook = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
        try:
            worksheet = workbook[SHEET]
            for label, component, method, source_rows in COMPONENTS:
                source_rows = list(source_rows) if isinstance(source_rows, range) else [source_rows]
                values = [_numeric(worksheet[f"{period['column']}{row}"].value, f"{period['column']}{row}") for row in source_rows]
                rate = sum(values) / len(values)
                amount = rate if method == "customer_allowance" else rate * BENCHMARK_MWH
                source_cells = ",".join(f"{period['column']}{row}" for row in source_rows)
                rows.append({
                    **period,
                    "component": component,
                    "component_label": label,
                    "calculation_basis": method,
                    "published_rate_gbp": rate,
                    "published_rate_unit": "GBP per benchmark customer" if method == "customer_allowance" else "GBP/MWh supplied",
                    "annual_gbp_per_3100kwh_ex_vat": amount,
                    "source_sheet": SHEET,
                    "source_cells": source_cells,
                })
        finally:
            workbook.close()
    result = pd.DataFrame(rows)
    starts = pd.to_datetime(result["delivery_start"])
    ends = pd.to_datetime(result["delivery_end"])
    result["delivery_days"] = (ends - starts).dt.days + 1
    result["calendar_weight_2020"] = result["delivery_days"] / 366
    result["annualised_gbp_per_3100kwh_ex_vat"] = (
        result["annual_gbp_per_3100kwh_ex_vat"] * result["calendar_weight_2020"]
    )
    return result


def extract_2020_policy_components(source_dir: str | Path) -> pd.DataFrame:
    """Return period and annualised 2020 Annex 4 policy components."""
    return _period_component_rows(source_dir)


def validate_against_cost_stack(components: pd.DataFrame, hedge_inputs_path: str | Path) -> None:
    """Ensure Annex 4 scheme rows sum to each existing published policy total."""
    expected = pd.read_csv(hedge_inputs_path).set_index("cap_period")["policy_gbp_per_3100kwh"]
    actual = components.groupby("cap_period")["annual_gbp_per_3100kwh_ex_vat"].sum()
    missing = expected.index.difference(actual.index)
    if len(missing):
        raise OfgemPolicyExtractionError(f"Missing extracted policy periods: {list(missing)}")
    differences = (actual.reindex(expected.index) - expected).abs()
    if float(differences.max()) > 1e-5:
        raise OfgemPolicyExtractionError(
            "Annex 4 components do not reconcile to the published policy allowance: "
            f"{differences.to_dict()}"
        )


def build_summary(components: pd.DataFrame) -> dict[str, Any]:
    annual = components.groupby(["component", "component_label"], as_index=False)[
        "annualised_gbp_per_3100kwh_ex_vat"
    ].sum()
    return {
        "scope": "GB-average, single-rate electricity, 3,100 kWh benchmark, ex VAT",
        "annualisation": "Calendar-day weighted over the 366 days of 2020.",
        "source": {
            "publisher": "Ofgem",
            "workbook_series": "Default tariff cap: Annex 4 Policy Cost Allowance methodology",
            "workbook_sheet": SHEET,
            "cap_levels_page": "https://www.ofgem.gov.uk/energy-regulation/domestic-and-non-domestic/energy-pricing-rules/energy-price-cap/energy-price-cap-default-tariff-levels",
        },
        "annualised_components_gbp_per_3100kwh_ex_vat": {
            row.component: float(row.annualised_gbp_per_3100kwh_ex_vat) for row in annual.itertuples(index=False)
        },
        "annualised_policy_total_gbp_per_3100kwh_ex_vat": float(annual["annualised_gbp_per_3100kwh_ex_vat"].sum()),
    }


def write_extract(
    source_dir: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    hedge_inputs_path: str | Path | None = None,
) -> None:
    """Extract, reconcile and write a provenance-preserving policy dataset."""
    components = extract_2020_policy_components(source_dir)
    if hedge_inputs_path:
        validate_against_cost_stack(components, hedge_inputs_path)
    for output in (output_csv, output_summary):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_csv, index=False)
    with Path(output_summary).open("w", encoding="utf-8") as stream:
        json.dump(build_summary(components), stream, indent=2)


def _main(snakemake) -> None:
    write_extract(
        snakemake.params.source_dir,
        snakemake.output.policy_components_csv,
        snakemake.output.summary_json,
        snakemake.input.hedge_inputs_csv,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--hedge-inputs")
    args = parser.parse_args()
    write_extract(args.source_dir, args.output_csv, args.output_summary, args.hedge_inputs)


if "snakemake" in globals():
    _main(snakemake)
elif __name__ == "__main__":
    _cli()
