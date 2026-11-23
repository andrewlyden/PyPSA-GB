"""Extract the published 2020 smart-metering net-cost components.

The cap model combines DCC, SEGB and SMICoP into one pass-through net change.
Its underlying rows let us retain those three additive components separately.
"""

import argparse
import json
from pathlib import Path

import openpyxl
import pandas as pd


SOURCES = (
    ("2020_Jan_Mar", "2020-01-01", "2020-03-31", "2019_10_to_2020_03_cap_model_v1.3.xlsx", "3h SMNCC", "M", "F"),
    ("2020_Apr_Sep", "2020-04-01", "2020-09-30", "2020_04_to_2020_09_cap_model_v1.4.xlsx", "3h SMNCC", "N", "F"),
    ("2020_Oct_Dec", "2020-10-01", "2020-12-31", "2020_10_to_2021_03_cap_model_v1.7.xlsx", "3i SMNCC", "P", "G"),
)


def extract_2020_smart_meter_components(source_dir: str | Path) -> pd.DataFrame:
    rows = []
    for cap_period, start, end, workbook_name, sheet_name, column, baseline_column in SOURCES:
        workbook = openpyxl.load_workbook(Path(source_dir) / workbook_name, data_only=True, read_only=True)
        try:
            sheet = workbook[sheet_name]
            cpi_multiplier = sheet[f"{column}17"].value
            if not isinstance(cpi_multiplier, (int, float)):
                raise ValueError(f"Expected CPIH multiplier in {sheet_name}!{column}17")
            values = []
            for component, row in (
                ("Smart metering DCC pass-through net change", 13),
                ("Smart metering SEGB pass-through net change", 14),
                ("Smart metering SMICoP pass-through net change", 15),
            ):
                current = sheet[f"{column}{row}"].value
                # Column F is April--September 2017, the operating-cost
                # baseline specified by Annex 5.  Column G is only the next
                # historical example period.
                baseline = sheet[f"{baseline_column}{row}"].value
                if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
                    raise ValueError(f"Expected pass-through values in {sheet_name}!{column}{row}/{baseline_column}{row}")
                values.append((
                    component,
                    float(current) - float(baseline) * float(cpi_multiplier),
                    f"{column}{row},{baseline_column}{row},{column}17",
                ))
            combined = sheet[f"{column}18"].value
            if not isinstance(combined, (int, float)):
                raise ValueError(f"Expected a numeric combined pass-through value in {sheet_name}!{column}18")
            # The downloadable model displays rounded scheme charges whereas
            # row 18 retains a calculation from unrounded values.  Allocate
            # that tiny rounding residual to DCC and retain row 18 as the
            # reconciled published total.
            residual = float(combined) - sum(value for _, value, _ in values)
            name, amount, cells = values[0]
            values[0] = (name, amount + residual, f"{cells}; reconciled to {column}18")
            nonpass = sheet[f"{column}29"].value
            if not isinstance(nonpass, (int, float)):
                raise ValueError(f"Expected a numeric SMNCC value in {sheet_name}!{column}29")
            values.append(("Smart metering non-pass-through model increment", float(nonpass), f"{column}29"))
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            total = 0.0
            for component, value, source_cells in values:
                total += float(value)
                rows.append({
                    "cap_period": cap_period, "delivery_start": start, "delivery_end": end,
                    "component": component, "annual_gbp_per_3100kwh_ex_vat": float(value),
                    "delivery_days": days, "calendar_weight_2020": days / 366,
                    "annualised_gbp_per_3100kwh_ex_vat": float(value) * days / 366,
                    "source_workbook": workbook_name, "source_sheet": sheet_name, "source_cells": source_cells,
                })
            if total == 0:
                raise ValueError(f"SMNCC components are zero for {cap_period}")
        finally:
            workbook.close()
    return pd.DataFrame(rows)


def write_extract(source_dir, output_csv, output_summary, hedge_inputs_path=None) -> None:
    frame = extract_2020_smart_meter_components(source_dir)
    if hedge_inputs_path:
        expected = pd.read_csv(hedge_inputs_path).set_index("cap_period")["smart_metering_gbp_per_3100kwh"]
        actual = frame.groupby("cap_period")["annual_gbp_per_3100kwh_ex_vat"].sum()
        if float((actual.reindex(expected.index) - expected).abs().max()) > 1e-6:
            raise ValueError("SMNCC extract does not reconcile to the cap model")
    for output in (output_csv, output_summary): Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    annual = frame.groupby("component")["annualised_gbp_per_3100kwh_ex_vat"].sum()
    with Path(output_summary).open("w", encoding="utf-8") as stream:
        json.dump({"scope": "GB-average, Other Payment Method, 3,100 kWh, ex VAT", "annualised_components_gbp_per_3100kwh_ex_vat": annual.to_dict(), "annualised_total_gbp_per_3100kwh_ex_vat": float(annual.sum())}, stream, indent=2)


def _main(snakemake):
    write_extract(snakemake.params.source_dir, snakemake.output.components_csv, snakemake.output.summary_json, snakemake.input.hedge_inputs_csv)


if "snakemake" in globals(): _main(snakemake)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True); parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True); parser.add_argument("--hedge-inputs")
    args = parser.parse_args(); write_extract(args.source_dir, args.output_csv, args.output_summary, args.hedge_inputs)
