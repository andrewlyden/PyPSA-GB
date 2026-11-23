"""Replay Ofgem's 2020 regional network allowances against domestic geography.

This is a regional domestic-bill calculation, not a reconstruction of the
complex DNO charging methodologies.  It uses the 14 single-rate 3,100 kWh
regional outputs in Ofgem's Annex 3, and weights them by official domestic
meters/consumption mapped to PES areas.  This is materially more specific than
the previous equal-weight GB average while retaining an auditable source.
"""


import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market.extract_ofgem_2020_network_components import (
    OUTPUT_ROWS,
    OUTPUT_SHEET,
    REFERENCE,
    SOURCES,
    _value,
)
from scripts.consumer_bill.build_domestic_pes_weights import PES_TO_OFGEM_REGION


COMPONENTS = ("TNUoS", "DUoS fixed charges", "DUoS unit charges", "BSUoS")


def extract_regional_period_components(source_dir: str | Path) -> pd.DataFrame:
    """Extract Ofgem's formula-linked Annex 3 values for every PES region."""
    import openpyxl

    rows: list[dict] = []
    for cap_period, start, end, workbook_name, column in SOURCES:
        path = Path(source_dir) / workbook_name
        formulas = openpyxl.load_workbook(path, data_only=False, read_only=True)
        values = openpyxl.load_workbook(path, data_only=True, read_only=True)
        try:
            for pes_id, row in zip(PES_TO_OFGEM_REGION, OUTPUT_ROWS, strict=True):
                formula = formulas[OUTPUT_SHEET][f"{column}{row}"].value
                if not isinstance(formula, str):
                    raise ValueError(f"Missing regional output formula at {OUTPUT_SHEET}!{column}{row}")
                extracted_region = values[OUTPUT_SHEET][f"E{row}"].value
                expected_region = PES_TO_OFGEM_REGION[pes_id]
                if extracted_region != expected_region:
                    raise ValueError(f"Unexpected Ofgem region {extracted_region!r}; expected {expected_region!r}")
                references = {(m.group("sheet"), m.group("cell")) for m in REFERENCE.finditer(formula)}
                by_sheet = {sheet: cell for sheet, cell in references}
                try:
                    tnuos_cell = by_sheet["2a TNUoS"]
                    duos_cell = by_sheet["2b DUoS"]
                    bsuos_cell = by_sheet["2c BSUoS"]
                except KeyError as exc:
                    raise ValueError(f"Could not parse network terms in {formula!r}") from exc
                duos_total = _value(values, "2b DUoS", duos_cell)
                fixed_cell = f"{re.match(r'[A-Z]+', duos_cell).group()}11"
                terms = {
                    "TNUoS": (_value(values, "2a TNUoS", tnuos_cell), f"2a TNUoS!{tnuos_cell}"),
                    "DUoS fixed charges": (_value(values, "2b DUoS", fixed_cell), f"2b DUoS!{fixed_cell}"),
                    "DUoS unit charges": (duos_total - _value(values, "2b DUoS", fixed_cell), f"2b DUoS!{duos_cell}"),
                    "BSUoS": (_value(values, "2c BSUoS", bsuos_cell), f"2c BSUoS!{bsuos_cell}"),
                }
                days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
                for component, (amount, cell) in terms.items():
                    rows.append({
                        "cap_period": cap_period, "delivery_start": start, "delivery_end": end,
                        "delivery_days": days, "calendar_weight_2020": days / 366,
                        "pes_id": pes_id, "ofgem_region": expected_region, "component": component,
                        "gbp_per_3100kwh_ex_vat": amount, "source_workbook": workbook_name,
                        "source_cell": cell,
                    })
        finally:
            formulas.close()
            values.close()
    return pd.DataFrame(rows)


def annualise_regional_network_components(period: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calendar-annualise regional cap values and apply domestic PES weights."""
    expected_period_columns = {
        "pes_id", "ofgem_region", "component", "gbp_per_3100kwh_ex_vat", "calendar_weight_2020"
    }
    expected_weight_columns = {"pes_id", "domestic_consumption_weight", "domestic_meter_weight"}
    if missing := expected_period_columns.difference(period.columns):
        raise ValueError(f"Regional Annex 3 values are missing: {sorted(missing)}")
    if missing := expected_weight_columns.difference(weights.columns):
        raise ValueError(f"Domestic PES weights are missing: {sorted(missing)}")
    expected_pes = set(PES_TO_OFGEM_REGION)
    available_pes = set(pd.to_numeric(weights["pes_id"], errors="raise").astype(int))
    if available_pes != expected_pes:
        raise ValueError(f"Domestic weights must cover exactly the 14 PES areas; got {sorted(available_pes)}")

    annual = period.copy()
    # Source lineage is present in the real Annex 3 extract.  Supplying blank
    # values here also makes the aggregation helper useful with compact inputs.
    for source_column in ("source_workbook", "source_cell"):
        if source_column not in annual:
            annual[source_column] = ""
    annual["annualised_gbp_per_3100kwh_ex_vat"] = (
        pd.to_numeric(annual["gbp_per_3100kwh_ex_vat"], errors="raise")
        * pd.to_numeric(annual["calendar_weight_2020"], errors="raise")
    )
    annual = annual.groupby(["pes_id", "ofgem_region", "component"], as_index=False).agg(
        annual_gbp_per_3100kwh_ex_vat=("annualised_gbp_per_3100kwh_ex_vat", "sum"),
        source_workbooks=("source_workbook", lambda values: ";".join(sorted(set(values)))),
        source_cells=("source_cell", lambda values: ";".join(values)),
    )
    annual = annual.merge(
        weights[["pes_id", "domestic_consumption_weight", "domestic_meter_weight"]],
        on="pes_id", how="left", validate="many_to_one",
    )
    annual["consumption_weighted_gbp_per_3100kwh_ex_vat"] = (
        annual["annual_gbp_per_3100kwh_ex_vat"] * annual["domestic_consumption_weight"]
    )
    annual["meter_weighted_gbp_per_3100kwh_ex_vat"] = (
        annual["annual_gbp_per_3100kwh_ex_vat"] * annual["domestic_meter_weight"]
    )
    summary = {
        "purpose": "2020 regional domestic network-cost tariff replay",
        "scope": "Ofgem default-tariff-cap Annex 3, single-rate meter, 3,100 kWh, ex VAT",
        "regional_tariff_source": "Ofgem Annex 3 formula-linked regional TNUoS, DUoS and BSUoS allowances",
        "domestic_weighting": "DESNZ domestic consumption and domestic-meter weights mapped to PES/DNO areas",
        "meter_weighted_gbp_per_3100kwh_ex_vat": annual.groupby("component")["meter_weighted_gbp_per_3100kwh_ex_vat"].sum().to_dict(),
        "consumption_weighted_gbp_per_3100kwh_ex_vat": annual.groupby("component")["consumption_weighted_gbp_per_3100kwh_ex_vat"].sum().to_dict(),
        "limitation": "This replays Ofgem's regional representative-household allowance. It is not a direct DNO tariff-class, time-band or supplier-loss-factor calculation.",
    }
    return annual.sort_values(["component", "pes_id"]).reset_index(drop=True), summary


def write_replay(source_dir, weights_path, output_csv, output_summary) -> None:
    annual, summary = annualise_regional_network_components(
        extract_regional_period_components(source_dir), pd.read_csv(weights_path)
    )
    for output in (output_csv, output_summary):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_replay(
        snakemake.params.source_dir, snakemake.input.domestic_weights,
        snakemake.output.replay_csv, snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--domestic-weights", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_replay(args.source_dir, args.domestic_weights, args.output_csv, args.output_summary)
