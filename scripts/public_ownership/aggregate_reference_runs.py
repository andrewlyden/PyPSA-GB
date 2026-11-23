"""Safely aggregate annual public-ownership reference-run bundles.

This is deliberately a post-solve reader: it neither opens a PyPSA network nor
modifies dispatch.  A bundle is either one full-year market summary or twelve
unique monthly summaries.  It is never an annualisation of a short sample.
"""

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_ACCOUNTING_FIELDS = (
    "reported_period_hours", "demand_MWh", "modelled_redispatch_cost_GBP",
    "ordinary_redispatch_cost_excluding_feasibility_slack_GBP",
    "feasibility_slack_energy_MWh", "feasibility_slack_cost_GBP", "gross_redispatch_MWh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary_row(path: Path, expected_scenario: str | None) -> tuple[dict, dict]:
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    scenario = summary.get("scenario")
    if expected_scenario is not None and scenario != expected_scenario:
        raise ValueError(f"{path} identifies {scenario!r}, expected {expected_scenario!r}")
    accounting = summary.get("accounting") or {}
    missing = [key for key in REQUIRED_ACCOUNTING_FIELDS if key not in accounting]
    if missing:
        raise ValueError(f"{path} lacks accounting fields: {missing}")
    prices = summary.get("prices") or {}
    row = {"scenario": scenario, "source_file": str(path)}
    row.update({key: float(accounting[key]) for key in REQUIRED_ACCOUNTING_FIELDS})
    row["mean_wholesale_price_GBP_per_MWh"] = prices.get("mean_wholesale_price")
    row["mean_nodal_dual_GBP_per_MWh"] = prices.get("mean_balancing_nodal_dual")
    return row, summary


def aggregate_market_summaries(
    summary_paths: Iterable[str | Path], *, expected_scenarios: Iterable[str] | None = None,
    run_group: str = "reference_run",
) -> tuple[pd.DataFrame, dict, dict]:
    """Return monthly rows, an annual summary and a source manifest."""
    paths = [Path(path) for path in summary_paths]
    expected = list(expected_scenarios) if expected_scenarios is not None else [None] * len(paths)
    if not paths or len(paths) != len(expected):
        raise ValueError("summary paths and expected scenarios must be non-empty and aligned")
    if len(paths) not in (1, 12):
        raise ValueError("A reference bundle must contain one full-year or twelve monthly summaries")
    if len(set(paths)) != len(paths):
        raise ValueError("A reference bundle contains duplicate summary paths")
    if len(paths) == 12 and len(set(expected)) != 12:
        raise ValueError("A monthly reference bundle requires twelve unique scenario IDs")

    rows, source_summaries = [], []
    for path, scenario in zip(paths, expected):
        row, summary = _summary_row(path, scenario)
        rows.append(row)
        source_summaries.append(summary)
    monthly = pd.DataFrame(rows)
    hours = float(monthly["reported_period_hours"].sum())
    if hours not in (8760.0, 8784.0):
        raise ValueError(f"Reference bundle is not a complete calendar year: {hours:g} hours")
    demand = float(monthly["demand_MWh"].sum())
    if demand <= 0:
        raise ValueError("Reference bundle has non-positive demand")

    def weighted(column: str) -> float | None:
        values = pd.to_numeric(monthly[column], errors="coerce")
        valid = values.notna()
        if not valid.any():
            return None
        return float((values[valid] * monthly.loc[valid, "demand_MWh"]).sum() /
                     monthly.loc[valid, "demand_MWh"].sum())

    summed_fields = REQUIRED_ACCOUNTING_FIELDS[2:]
    annual = {
        "run_group": run_group,
        "aggregation": "complete_period_sum; price metrics are demand-weighted",
        "annualisation_allowed": True,
        "reported_period_hours": hours,
        "demand_MWh": demand,
        **{field: float(monthly[field].sum()) for field in summed_fields},
        "mean_wholesale_price_GBP_per_MWh_demand_weighted": weighted("mean_wholesale_price_GBP_per_MWh"),
        "mean_nodal_dual_GBP_per_MWh_demand_weighted": weighted("mean_nodal_dual_GBP_per_MWh"),
    }
    annual["modelled_redispatch_cost_GBP_per_MWh_demand"] = (
        annual["modelled_redispatch_cost_GBP"] / demand
    )
    manifest = {
        "run_group": run_group,
        "source_count": len(paths),
        "scenarios": monthly["scenario"].tolist(),
        "sources": [
            {"scenario": row["scenario"], "path": row["source_file"], "sha256": _sha256(path)}
            for row, path in zip(rows, paths)
        ],
    }
    return monthly, annual, manifest


def write_annual_market_bundle(
    summary_paths: Iterable[str | Path], output_monthly_csv: str | Path,
    output_annual_json: str | Path, output_manifest_json: str | Path, *,
    expected_scenarios: Iterable[str] | None = None, run_group: str = "reference_run",
) -> dict:
    """Write the three portable annual-market bundle outputs."""
    monthly, annual, manifest = aggregate_market_summaries(
        summary_paths, expected_scenarios=expected_scenarios, run_group=run_group
    )
    for destination in (output_monthly_csv, output_annual_json, output_manifest_json):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_monthly_csv, index=False)
    Path(output_annual_json).write_text(json.dumps(annual, indent=2), encoding="utf-8")
    Path(output_manifest_json).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return annual


if __name__ == "__main__":
    write_annual_market_bundle(
        list(snakemake.input.summary_jsons), snakemake.output.monthly_csv,
        snakemake.output.annual_json, snakemake.output.manifest_json,
        expected_scenarios=list(snakemake.params.scenarios),
        run_group=str(snakemake.params.run_group),
    )
