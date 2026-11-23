"""Aggregate complete monthly CP30/NESO comparator market summaries.

This is intentionally an addition, not an extrapolation: every configured
month must be present and the headline £/MWh is recomputed from summed cost and
summed demand.  It must not be used for partial-year or short-run scenarios.
"""


# No ``from __future__ import annotations``: Snakemake prepends a preamble to
# script-directive files, which puts a future import past the start of the
# compiled module and raises SyntaxError. Python 3.12 evaluates the PEP 604
# annotations below natively, so the import is not needed.
import json
from pathlib import Path

import pandas as pd


REQUIRED_ACCOUNTING_FIELDS = (
    "reported_period_hours",
    "demand_MWh",
    "accepted_offer_cost_GBP",
    "accepted_bid_cost_GBP",
    "modelled_redispatch_cost_GBP",
    "ordinary_redispatch_cost_excluding_feasibility_slack_GBP",
    "feasibility_slack_energy_MWh",
    "feasibility_slack_cost_GBP",
    "gross_redispatch_MWh",
)


def _optional_sum(monthly: pd.DataFrame, column: str):
    """Sum a diagnostic that older market summaries may not carry.

    A missing diagnostic returns ``None`` rather than a partial total, so a
    consumer of the aggregate cannot mistake eleven months for twelve.
    """
    values = pd.to_numeric(monthly[column], errors="coerce")
    if values.isna().any():
        return None
    return float(values.sum())


def _optional_max(monthly: pd.DataFrame, column: str):
    values = pd.to_numeric(monthly[column], errors="coerce")
    if values.isna().all():
        return None
    return int(values.max())


def _read_summary(path: str | Path, scenario: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("scenario") != scenario:
        raise ValueError(
            f"Monthly comparator input {path} identifies scenario "
            f"{summary.get('scenario')!r}, expected {scenario!r}"
        )
    accounting = summary.get("accounting") or {}
    missing = [field for field in REQUIRED_ACCOUNTING_FIELDS if field not in accounting]
    if missing:
        raise ValueError(f"Monthly comparator input {path} lacks accounting fields: {missing}")
    if accounting.get("is_short_run", True) or accounting.get("annualisation_allowed", False) is False:
        # A complete monthly calculation is short of an annual calculation but
        # valid as an input. The aggregate itself is explicitly not annualised.
        pass
    return summary


def aggregate_monthly_comparator_summaries(
    summary_paths: list[str | Path], scenarios: list[str], output_csv: str | Path,
    output_json: str | Path, scenario_family: str | None = None,
) -> dict:
    """Write monthly metrics and a summed, demand-weighted annual summary."""
    if len(summary_paths) != 12 or len(scenarios) != 12:
        raise ValueError("The NESO comparator aggregate requires exactly twelve monthly summaries")
    if len(set(scenarios)) != 12:
        raise ValueError("The NESO comparator aggregate received duplicate scenario IDs")

    rows = []
    for path, scenario in zip(summary_paths, scenarios):
        summary = _read_summary(path, scenario)
        accounting = summary["accounting"]
        prices = summary.get("prices") or {}
        congestion = summary.get("congestion") or {}
        interconnectors = summary.get("interconnectors") or {}
        revenue = summary.get("revenue") or {}
        rows.append({
            "scenario": scenario,
            "reported_period_hours": float(accounting["reported_period_hours"]),
            "demand_MWh": float(accounting["demand_MWh"]),
            "accepted_offer_cost_GBP": float(accounting["accepted_offer_cost_GBP"]),
            "accepted_bid_cost_GBP": float(accounting["accepted_bid_cost_GBP"]),
            "modelled_redispatch_cost_GBP": float(accounting["modelled_redispatch_cost_GBP"]),
            "ordinary_redispatch_cost_GBP": float(
                accounting["ordinary_redispatch_cost_excluding_feasibility_slack_GBP"]
            ),
            "feasibility_slack_energy_MWh": float(accounting["feasibility_slack_energy_MWh"]),
            "feasibility_slack_cost_GBP": float(accounting["feasibility_slack_cost_GBP"]),
            "gross_redispatch_MWh": float(accounting["gross_redispatch_MWh"]),
            "mean_wholesale_price_GBP_per_MWh": prices.get("mean_wholesale_price"),
            # Hourly sum(price*demand)/sum(demand) for the month, when the
            # monthly summary carries it; lets the annual figure be exact.
            "demand_weighted_wholesale_price_GBP_per_MWh": prices.get(
                "demand_weighted_wholesale_price"
            ),
            "mean_nodal_dual_GBP_per_MWh": prices.get("mean_balancing_nodal_dual"),
            # Bid/ask valuation of the balancing-stage change in cable schedules;
            # in the balancing objective but not in accepted offers/bids.
            "interconnector_retrading_cost_GBP": accounting.get(
                "interconnector_retrading_cost_GBP"
            ),
            # Reported apart from the NESO-comparable cost (16 Oct 2026).
            "non_participant_turn_down_cost_at_penalty_GBP": accounting.get(
                "non_participant_turn_down_cost_at_penalty_GBP"
            ),
            "churn_sink_held_MWh": accounting.get("churn_sink_held_MWh"),
            "congested_lines": congestion.get("congested_lines"),
            "congested_transformers": congestion.get("congested_transformers"),
            # Diagnostics the consumer bill reports but never charges: cap-and-floor
            # congestion rent belongs to TNUoS, and slack energy is adequacy context.
            "interconnector_congestion_rent_GBP": interconnectors.get("estimated_congestion_rent_GBP"),
            "cfd_archived_cost_GBP": revenue.get("cfd_total_cost"),
            "cfd_archived_dispatch_MWh": revenue.get("cfd_total_dispatch_MWh"),
            # Internal HVDC tie-break (16 Oct 2026): objective only, outside every cost above.
            "internal_dc_tie_break_cost_GBP": (summary.get("balancing") or {}).get(
                "internal_dc_tie_break_cost_GBP"),
            "stage1_internal_dc_tie_break_cost_GBP": (summary.get("wholesale") or {}).get(
                "internal_dc_tie_break_cost_GBP"),
        })

    monthly = pd.DataFrame(rows)
    total_hours = float(monthly["reported_period_hours"].sum())
    if total_hours not in (8760.0, 8784.0):
        raise ValueError(
            "Monthly comparator periods do not cover one complete calendar year: "
            f"{total_hours:g} hours (expected 8760 or 8784)"
        )
    demand = float(monthly["demand_MWh"].sum())
    if demand <= 0.0:
        raise ValueError("Monthly comparator aggregate has non-positive demand")

    def weighted(column: str):
        values = pd.to_numeric(monthly[column], errors="coerce")
        valid = values.notna()
        if not valid.any():
            return None
        return float((values[valid] * monthly.loc[valid, "demand_MWh"]).sum() /
                     monthly.loc[valid, "demand_MWh"].sum())

    # Exact annual demand-weighted price when every month carries the hourly
    # demand-weighted value (sum over months of dw_price_m * demand_m equals
    # the hourly sum(price*demand)); otherwise fall back to demand-weighting the
    # monthly simple means, which is only an approximation.
    hourly_dw = pd.to_numeric(
        monthly["demand_weighted_wholesale_price_GBP_per_MWh"], errors="coerce"
    )
    if hourly_dw.notna().all():
        mean_wholesale_dw = float((hourly_dw * monthly["demand_MWh"]).sum() / demand)
        mean_wholesale_dw_basis = "hourly_demand_weighted"
    else:
        mean_wholesale_dw = weighted("mean_wholesale_price_GBP_per_MWh")
        mean_wholesale_dw_basis = "demand_weighted_mean_of_monthly_simple_means"

    aggregate = {
        "scenario_family": scenario_family or "CP30 monthly comparator",
        "aggregation": "sum_of_12_complete_months; cost and volume rates use total demand",
        "annualisation_allowed": True,
        "reported_period_hours": total_hours,
        "demand_MWh": demand,
        "accepted_offer_cost_GBP": float(monthly["accepted_offer_cost_GBP"].sum()),
        "accepted_bid_cost_GBP": float(monthly["accepted_bid_cost_GBP"].sum()),
        "modelled_redispatch_cost_GBP": float(monthly["modelled_redispatch_cost_GBP"].sum()),
        "ordinary_redispatch_cost_GBP": float(monthly["ordinary_redispatch_cost_GBP"].sum()),
        "feasibility_slack_energy_MWh": float(monthly["feasibility_slack_energy_MWh"].sum()),
        "feasibility_slack_cost_GBP": float(monthly["feasibility_slack_cost_GBP"].sum()),
        "gross_redispatch_MWh": float(monthly["gross_redispatch_MWh"].sum()),
        "mean_wholesale_price_GBP_per_MWh_demand_weighted": mean_wholesale_dw,
        "mean_wholesale_price_GBP_per_MWh_demand_weighted_basis": mean_wholesale_dw_basis,
        "mean_nodal_dual_GBP_per_MWh_demand_weighted": weighted(
            "mean_nodal_dual_GBP_per_MWh"
        ),
        "interconnector_congestion_rent_GBP": _optional_sum(
            monthly, "interconnector_congestion_rent_GBP"
        ),
        "interconnector_retrading_cost_GBP": _optional_sum(
            monthly, "interconnector_retrading_cost_GBP"
        ),
        "non_participant_turn_down_cost_at_penalty_GBP": _optional_sum(
            monthly, "non_participant_turn_down_cost_at_penalty_GBP"
        ),
        "churn_sink_held_MWh": _optional_sum(monthly, "churn_sink_held_MWh"),
        "cfd_archived_cost_GBP": _optional_sum(monthly, "cfd_archived_cost_GBP"),
        "cfd_archived_dispatch_MWh": _optional_sum(monthly, "cfd_archived_dispatch_MWh"),
        "internal_dc_tie_break_cost_GBP": _optional_sum(monthly, "internal_dc_tie_break_cost_GBP"),
        "stage1_internal_dc_tie_break_cost_GBP": _optional_sum(monthly, "stage1_internal_dc_tie_break_cost_GBP"),
        "max_congested_lines": _optional_max(monthly, "congested_lines"),
        "max_congested_transformers": _optional_max(monthly, "congested_transformers"),
    }
    aggregate["modelled_redispatch_cost_GBP_per_MWh_demand"] = (
        aggregate["modelled_redispatch_cost_GBP"] / demand
    )
    aggregate["ordinary_redispatch_cost_GBP_per_MWh_demand"] = (
        aggregate["ordinary_redispatch_cost_GBP"] / demand
    )
    aggregate["gross_redispatch_pct_demand"] = 100.0 * aggregate["gross_redispatch_MWh"] / demand
    retrading = aggregate["interconnector_retrading_cost_GBP"]
    aggregate["modelled_redispatch_cost_including_retrading_GBP"] = (
        aggregate["modelled_redispatch_cost_GBP"] + retrading
        if retrading is not None
        else None
    )
    # NESO-comparable (16 Oct 2026): ordinary redispatch (no VoLL slack, no
    # non-participant penalty turn-down) plus interconnector re-trading, the basis of
    # NESO's GBP 3.57 bn thermal constraint cost. Summed from the same monthly fields,
    # so it exists for older summaries too; None when re-trading is missing.
    aggregate["neso_comparable_redispatch_cost_GBP"] = (
        aggregate["ordinary_redispatch_cost_GBP"] + retrading
        if retrading is not None
        else None
    )
    aggregate["notes"] = [
        "interconnector_retrading_cost_GBP is the bid/ask valuation of the change "
        "in interconnector schedules between the wholesale and balancing stages "
        "(balancing schedule valued at foreign ask on imports and "
        "efficiency-adjusted foreign bid on exports, minus the same valuation of "
        "the day-ahead schedule). It is part of the balancing LP objective but is "
        "not included in accepted offer/bid costs; "
        "modelled_redispatch_cost_including_retrading_GBP adds it to "
        "modelled_redispatch_cost_GBP. Both are None when any monthly summary "
        "lacks the diagnostic. neso_comparable_redispatch_cost_GBP is "
        "ordinary_redispatch_cost_GBP + interconnector_retrading_cost_GBP: NESO's "
        "thermal constraint cost basis, without feasibility slack "
        "(feasibility_slack_cost_GBP) or non-participant penalty turn-down "
        "(non_participant_turn_down_cost_at_penalty_GBP), which are reported apart.",
        "mean_wholesale_price_GBP_per_MWh_demand_weighted_basis records whether "
        "the annual demand-weighted price is exact (hourly_demand_weighted) or "
        "approximated from monthly simple means.",
    ]

    output_csv = Path(output_csv)
    output_json = Path(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps({"aggregate": aggregate, "monthly_scenarios": scenarios}, indent=2),
        encoding="utf-8",
    )
    return aggregate


if __name__ == "__main__":
    aggregate_monthly_comparator_summaries(
        list(snakemake.input.summary_jsons),
        list(snakemake.params.scenarios),
        snakemake.output.monthly_csv,
        snakemake.output.annual_json,
        getattr(snakemake.params, "scenario_family", None),
    )
