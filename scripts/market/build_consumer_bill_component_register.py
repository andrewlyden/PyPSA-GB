"""Build the fully itemised Ofgem benchmark component register.

Stage 2 (``consumer_bill_breakdown.py``) groups this register for a readable
consumer view.  The register deliberately retains every available underlying
Ofgem component and never sums it into a presentation category.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _annual_components(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "source_workbook" not in frame:
        frame["source_workbook"] = frame.get("workbook", "Primary-source workbook")
    if "source_cells" not in frame:
        frame["source_cells"] = "See primary-source extract"
    if "detail_status" not in frame:
        frame["detail_status"] = "primary_source_component"
    return frame.groupby("component", as_index=False).agg(
        annual_gbp_per_3100kwh_ex_vat=("annualised_gbp_per_3100kwh_ex_vat", "sum"),
        source_workbook=("source_workbook", lambda x: ";".join(sorted(set(x.astype(str))))),
        source_cells=("source_cells", lambda x: ";".join(sorted(set(x.astype(str))))),
        detail_status=("detail_status", lambda x: ";".join(sorted(set(x.astype(str))))),
    )


def build_component_register(cost_stack: pd.DataFrame, policy: pd.DataFrame, network: pd.DataFrame, smart: pd.DataFrame, adjustment: pd.DataFrame | None = None, source_label: str = "Annual price-cap component bundle") -> pd.DataFrame:
    """Replace published aggregates by their reconciled primary-source rows."""
    raw = cost_stack.loc[~cost_stack["stack_group"].eq("total")].copy()
    value = "annual_gbp_per_3100kwh_ex_vat"
    raw[value] = pd.to_numeric(raw[value], errors="raise")
    replacements = {
        "policy": _annual_components(policy),
        "network": _annual_components(network),
        "smart": _annual_components(smart),
    }
    parent_amounts = {
        "policy": float(raw.loc[raw["stack_group"].eq("policy"), value].sum()),
        "network": float(raw.loc[raw["stack_group"].eq("network"), value].sum()),
        "smart": float(raw.loc[raw["component"].eq("Smart-metering net cost change"), value].sum()),
    }
    if adjustment is not None:
        replacements["adjustment"] = _annual_components(adjustment)
        parent_amounts["adjustment"] = float(raw.loc[raw["component"].eq("Adjustment allowance"), value].sum())
    for key, frame in replacements.items():
        if abs(float(frame[value].sum()) - parent_amounts[key]) > 1e-5:
            raise ValueError(f"Detailed {key} rows do not reconcile to the published cost-stack parent")

    kept = raw.loc[
        ~raw["stack_group"].isin(["policy", "network"])
        & ~raw["component"].eq("Smart-metering net cost change")
    ].copy()
    if adjustment is not None:
        kept = kept.loc[~kept["component"].eq("Adjustment allowance")].copy()
    kept["component_id"] = kept["component"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    kept["published_parent_component"] = kept["component"]
    kept["source_workbook"] = source_label
    kept["source_cells"] = "See retail cost-stack source"
    kept["detail_status"] = "published_component"

    detail_rows = []
    for key, stack_group, parent, prefix in (
        ("policy", "policy", "Policy costs", "Ofgem Annex 4 — "),
        ("network", "network", "Network costs", "Ofgem Annex 3 — "),
        ("smart", "supplier_operating", "Smart-metering net cost change", "Ofgem Annex 5 — "),
    ):
        for item in replacements[key].itertuples(index=False):
            detail_rows.append({
                "stack_group": stack_group,
                "component": prefix + item.component,
                value: item.annual_gbp_per_3100kwh_ex_vat,
                "is_separately_published": True,
                "component_id": f"{key}_{str(item.component).lower().replace(' ', '_').replace('—', '')}",
                "published_parent_component": parent,
                "source_workbook": item.source_workbook,
                "source_cells": item.source_cells,
                "detail_status": item.detail_status,
            })
    if adjustment is not None:
        for item in replacements["adjustment"].itertuples(index=False):
            detail_rows.append({
                "stack_group": "system_balancing", "component": "Ofgem Annex 8 - " + item.component,
                value: item.annual_gbp_per_3100kwh_ex_vat, "is_separately_published": True,
                "component_id": f"adjustment_{str(item.component).lower().replace(' ', '_')}",
                "published_parent_component": "Adjustment allowance", "source_workbook": item.source_workbook,
                "source_cells": item.source_cells, "detail_status": item.detail_status,
            })
    register = pd.concat([kept, pd.DataFrame(detail_rows)], ignore_index=True, sort=False)
    if abs(float(register[value].sum()) - float(raw[value].sum())) > 1e-5:
        raise ValueError("Detailed register does not reconcile to the published retail cost stack")
    return register.sort_values(["stack_group", "component"]).reset_index(drop=True)


def write_register(cost_stack_path, policy_path, network_path, smart_path, adjustment_path, output_csv, output_summary, metadata_path=None) -> None:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8")) if metadata_path else {}
    source_label = metadata.get("snapshot_id", "Annual price-cap component bundle")
    register = build_component_register(
        pd.read_csv(cost_stack_path), pd.read_csv(policy_path), pd.read_csv(network_path),
        pd.read_csv(smart_path), pd.read_csv(adjustment_path), source_label=source_label,
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    register.to_csv(output_csv, index=False)
    summary = {
        "stage": "ofgem_benchmark_component_register",
        "scope": "Annual GB-average, Other Payment Method, single-rate electricity, 3,100 kWh benchmark, ex VAT",
        "snapshot_id": metadata.get("snapshot_id"),
        "component_count": int(len(register)),
        "total_gbp_ex_vat": float(register.loc[~register["stack_group"].eq("vat"), "annual_gbp_per_3100kwh_ex_vat"].sum()),
        "vat_gbp": float(register.loc[register["stack_group"].eq("vat"), "annual_gbp_per_3100kwh_ex_vat"].sum()),
        "total_gbp_inc_vat": float(register["annual_gbp_per_3100kwh_ex_vat"].sum()),
        "grouping": "No presentation groups are calculated in this artifact; stage 2 performs grouping.",
    }
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _main(snakemake):
    write_register(snakemake.input.retail_stack_csv, snakemake.input.policy_components_csv, snakemake.input.network_components_csv, snakemake.input.smart_components_csv, snakemake.input.adjustment_components_csv, snakemake.output.register_csv, snakemake.output.summary_json, snakemake.input.snapshot_metadata_json)


if "snakemake" in globals(): _main(snakemake)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("cost-stack", "policy-components", "network-components", "smart-components", "adjustment-components", "output-csv", "output-summary"): parser.add_argument(f"--{argument}", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args(); write_register(args.cost_stack, args.policy_components, args.network_components, args.smart_components, args.adjustment_components, args.output_csv, args.output_summary, args.metadata)
