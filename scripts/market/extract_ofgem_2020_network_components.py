"""Extract detailed 2020 Ofgem Annex 3 electricity-network allowances."""

import argparse
import json
import re
from pathlib import Path

import openpyxl
import pandas as pd


SOURCES = (
    ("2020_Jan_Mar", "2020-01-01", "2020-03-31", "2019_10_to_2020_03_annex_3_network_v1.4.xlsx", "S"),
    ("2020_Apr_Sep", "2020-04-01", "2020-09-30", "2020_04_to_2020_09_annex_3_network_v1.5.xlsx", "T"),
    ("2020_Oct_Dec", "2020-10-01", "2020-12-31", "2020_10_to_2020_03_annex_3_network_v1.6.xlsx", "U"),
)
OUTPUT_SHEET = "1a Network Cost Allowance-Elec"
OUTPUT_ROWS = range(28, 42)
REFERENCE = re.compile(r"'(?P<sheet>2[abc] (?:TNUoS|DUoS|BSUoS))'!(?P<cell>[A-Z]+\d+)")


class OfgemNetworkExtractionError(ValueError):
    pass


def _value(book, sheet: str, cell: str) -> float:
    value = book[sheet][cell].value
    if not isinstance(value, (int, float)):
        raise OfgemNetworkExtractionError(f"Expected a numeric value at {sheet}!{cell}, got {value!r}")
    return float(value)


def extract_2020_network_components(source_dir: str | Path) -> pd.DataFrame:
    """Return annualised, GB-average TNUoS/DUoS/BSUoS components.

    The output formula in Annex 3 defines the exact component cells for each
    region.  We follow those references, rather than duplicating Ofgem's
    charging formula, and split the referenced DUoS value into fixed and unit
    charges using the corresponding nil-consumption row.
    """
    rows = []
    for cap_period, start, end, workbook_name, column in SOURCES:
        path = Path(source_dir) / workbook_name
        formulas = openpyxl.load_workbook(path, data_only=False, read_only=True)
        values = openpyxl.load_workbook(path, data_only=True, read_only=True)
        try:
            period = {"TNUoS": [], "DUoS fixed charges": [], "DUoS unit charges": [], "BSUoS": [], "total": []}
            sources = {key: [] for key in period}
            for row in OUTPUT_ROWS:
                output_cell = f"{column}{row}"
                formula = formulas[OUTPUT_SHEET][output_cell].value
                if not isinstance(formula, str):
                    raise OfgemNetworkExtractionError(f"Missing output formula in {OUTPUT_SHEET}!{output_cell}")
                references = {(match.group("sheet"), match.group("cell")) for match in REFERENCE.finditer(formula)}
                by_sheet = {sheet: cell for sheet, cell in references}
                try:
                    tnuos_cell = by_sheet["2a TNUoS"]
                    duos_cell = by_sheet["2b DUoS"]
                    bsuos_cell = by_sheet["2c BSUoS"]
                except KeyError as exc:
                    raise OfgemNetworkExtractionError(f"Could not parse Annex 3 formula {formula!r}") from exc
                duos_total = _value(values, "2b DUoS", duos_cell)
                duos_fixed_cell = f"{re.match(r'[A-Z]+', duos_cell).group()}11"
                duos_fixed = _value(values, "2b DUoS", duos_fixed_cell)
                terms = {
                    "TNUoS": _value(values, "2a TNUoS", tnuos_cell),
                    "DUoS fixed charges": duos_fixed,
                    "DUoS unit charges": duos_total - duos_fixed,
                    "BSUoS": _value(values, "2c BSUoS", bsuos_cell),
                }
                total = _value(values, OUTPUT_SHEET, output_cell)
                if abs(sum(terms.values()) - total) > 1e-6:
                    raise OfgemNetworkExtractionError(f"Network terms do not reconcile at {output_cell}")
                for name, amount in terms.items():
                    period[name].append(amount)
                period["total"].append(total)
                sources["TNUoS"].append(f"2a TNUoS!{tnuos_cell}")
                sources["DUoS fixed charges"].append(f"2b DUoS!{duos_fixed_cell}")
                sources["DUoS unit charges"].append(f"2b DUoS!{duos_cell}")
                sources["BSUoS"].append(f"2c BSUoS!{bsuos_cell}")

            days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            for component in ("TNUoS", "DUoS fixed charges", "DUoS unit charges", "BSUoS"):
                amount = sum(period[component]) / len(period[component])
                rows.append({
                    "cap_period": cap_period, "delivery_start": start, "delivery_end": end,
                    "component": component, "annual_gbp_per_3100kwh_ex_vat": amount,
                    "delivery_days": days, "calendar_weight_2020": days / 366,
                    "annualised_gbp_per_3100kwh_ex_vat": amount * days / 366,
                    "source_workbook": workbook_name, "source_cells": ";".join(sources[component]),
                    "source_method": "Mean of Ofgem's 14 regional single-rate benchmark outputs",
                })
            if abs(sum(sum(period[item]) for item in ("TNUoS", "DUoS fixed charges", "DUoS unit charges", "BSUoS")) / 14 - sum(period["total"]) / 14) > 1e-6:
                raise OfgemNetworkExtractionError(f"Regional network reconciliation failed for {cap_period}")
        finally:
            formulas.close(); values.close()
    return pd.DataFrame(rows)


def validate_against_cost_stack(components: pd.DataFrame, hedge_inputs_path: str | Path) -> None:
    expected = pd.read_csv(hedge_inputs_path).set_index("cap_period")["network_gbp_per_3100kwh"]
    actual = components.groupby("cap_period")["annual_gbp_per_3100kwh_ex_vat"].sum()
    difference = (actual.reindex(expected.index) - expected).abs()
    if difference.isna().any() or float(difference.max()) > 1e-5:
        raise OfgemNetworkExtractionError(f"Network extract does not reconcile: {difference.to_dict()}")


def write_extract(source_dir, output_csv, output_summary, hedge_inputs_path=None) -> None:
    components = extract_2020_network_components(source_dir)
    if hedge_inputs_path:
        validate_against_cost_stack(components, hedge_inputs_path)
    for output in (output_csv, output_summary): Path(output).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_csv, index=False)
    annual = components.groupby("component")["annualised_gbp_per_3100kwh_ex_vat"].sum()
    with Path(output_summary).open("w", encoding="utf-8") as stream:
        json.dump({"scope": "GB-average, single-rate electricity, 3,100 kWh, ex VAT", "source": "Ofgem Annex 3 electricity network cost allowance", "annualised_components_gbp_per_3100kwh_ex_vat": annual.to_dict(), "annualised_total_gbp_per_3100kwh_ex_vat": float(annual.sum())}, stream, indent=2)


def _main(snakemake):
    write_extract(snakemake.params.source_dir, snakemake.output.components_csv, snakemake.output.summary_json, snakemake.input.hedge_inputs_csv)


if "snakemake" in globals(): _main(snakemake)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True); parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True); parser.add_argument("--hedge-inputs")
    args = parser.parse_args(); write_extract(args.source_dir, args.output_csv, args.output_summary, args.hedge_inputs)
