"""Extract the 2020 Annex 8 adjustment allowance and its non-additive lineage."""

import argparse
import json
from pathlib import Path

import openpyxl
import pandas as pd


def extract_adjustment_component(workbook_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the annualised final cost and the Q1 calculation lineage.

    Annex 8 has one additive consumer-bill amount: the adjustment allowance.
    Rows that construct the Q1 rate are deliberately emitted as a separate
    lineage table: adding them to the final allowance would double count it.
    """
    workbook = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    try:
        allowance = workbook["1a Adjustment Allowance"]
        rows = []
        for row in range(55, 69):  # Other Payment Method, electricity TDCV regions
            region, amount = allowance[f"F{row}"].value, allowance[f"W{row}"].value
            if not isinstance(region, str) or not isinstance(amount, (int, float)):
                raise ValueError(f"Expected Annex 8 OPM adjustment output in row {row}")
            rows.append((region, float(amount), f"W{row}"))
        regional = pd.DataFrame(rows, columns=["region", "gbp_per_3100kwh_ex_vat", "source_cells"])
        gb_average = float(regional["gbp_per_3100kwh_ex_vat"].mean())
        # This source is a six-month cap value.  It applies only from October
        # through December in the calendar-year 2020 analysis.
        annualised = gb_average * 92 / 366
        components = pd.DataFrame([{
            "cap_period": "2020_Oct_Dec",
            "delivery_start": "2020-10-01", "delivery_end": "2020-12-31",
            "component": "Q1 adjustment allowance",
            "annual_gbp_per_3100kwh_ex_vat": gb_average,
            "delivery_days": 92, "calendar_weight_2020": 92 / 366,
            "annualised_gbp_per_3100kwh_ex_vat": annualised,
            "source_workbook": Path(workbook_path).name,
            "source_sheet": "1a Adjustment Allowance",
            "source_cells": ";".join(regional["source_cells"]),
        }])

        q1 = workbook["2a Q1 Adjustment Component"]
        lineage_rows = []
        for row, label in (
            (13, "Volume-adjusted charge"),
            (17, "Q1 charge after shaping, forecast, imbalance, transaction and basis allowances"),
            (31, "Q1 charge after regional electricity-loss uplift"),
            (51, "Illustrative regional final customer allowance"),
        ):
            value = q1[f"H{row}"].value
            if isinstance(label, str) and isinstance(value, (int, float)):
                lineage_rows.append({
                    "lineage_step": label,
                    "example_value": float(value),
                    "source_sheet": "2a Q1 Adjustment Component",
                    "source_cells": f"B{row},H{row}",
                    "additive_bill_component": False,
                })
        lineage = pd.DataFrame(lineage_rows)
    finally:
        workbook.close()
    return components, lineage


def write_extract(workbook_path, output_csv, output_lineage_csv, output_summary, hedge_inputs_path=None) -> None:
    components, lineage = extract_adjustment_component(workbook_path)
    if hedge_inputs_path:
        expected = pd.read_csv(hedge_inputs_path).set_index("cap_period").loc[
            "2020_Oct_Dec", "adjustment_allowance_gbp_per_3100kwh"
        ]
        actual = float(components["annual_gbp_per_3100kwh_ex_vat"].iloc[0])
        if abs(actual - float(expected)) > 1e-6:
            raise ValueError(f"Annex 8 result does not reconcile: {actual} != {expected}")
    for path in (output_csv, output_lineage_csv, output_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_csv, index=False)
    lineage.to_csv(output_lineage_csv, index=False)
    Path(output_summary).write_text(json.dumps({
        "scope": "GB-average, Other Payment Method, single-rate electricity, 3,100 kWh, ex VAT",
        "annualised_component_gbp_per_3100kwh_ex_vat": float(components["annualised_gbp_per_3100kwh_ex_vat"].sum()),
        "lineage_is_additive": False,
        "note": "The Q1 steps explain the final allowance and are not additional bill costs.",
    }, indent=2), encoding="utf-8")


def _main(snakemake):
    write_extract(snakemake.input.annex_8, snakemake.output.components_csv, snakemake.output.lineage_csv, snakemake.output.summary_json, snakemake.input.hedge_inputs_csv)


if "snakemake" in globals():
    _main(snakemake)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annex-8", required=True); parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-lineage-csv", required=True); parser.add_argument("--output-summary", required=True)
    parser.add_argument("--hedge-inputs")
    args = parser.parse_args()
    write_extract(args.annex_8, args.output_csv, args.output_lineage_csv, args.output_summary, args.hedge_inputs)
