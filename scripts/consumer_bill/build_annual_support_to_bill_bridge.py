"""Connect annual CfD/ROC reconciliation evidence to the consumer-bill stack.

This is deliberately a *recovery bridge*, rather than a bill calculator.  A
modelled scheme cost becomes a household allowance only when its supplier-sales
denominator and domestic allocation are supplied.  Keeping that gate here
prevents a physical generation result from silently becoming a consumer levy.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


BENCHMARK_MWH = 3.1


class SupportToBillBridgeError(ValueError):
    """Raised when annual support evidence cannot safely be bridged."""


def _number(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    return None if value is None else float(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _published_allowance(policy: pd.DataFrame, component: str) -> float:
    required = {"component", "annualised_gbp_per_3100kwh_ex_vat"}
    if missing := required.difference(policy.columns):
        raise SupportToBillBridgeError(f"Policy component input is missing: {sorted(missing)}")
    values = pd.to_numeric(
        policy.loc[policy["component"].eq(component), "annualised_gbp_per_3100kwh_ex_vat"],
        errors="raise",
    )
    if values.empty:
        raise SupportToBillBridgeError(f"No annualised Ofgem policy allowance for {component}")
    return float(values.sum())


def _household_candidate(cost_gbp: float, assumptions: dict[str, Any]) -> float | None:
    sales_mwh = _number(assumptions, "billable_supplier_sales_mwh")
    share = _number(assumptions, "domestic_recovery_share")
    share = 1.0 if share is None else share
    if sales_mwh is None or sales_mwh <= 0 or not 0 <= share <= 1:
        return None
    return cost_gbp / sales_mwh * share * BENCHMARK_MWH


def _cfd_levy_recovery(levy: pd.DataFrame, year: int) -> dict[str, float]:
    required = {"month", "reconciled_eligible_demand_mwh", "reconciled_levy_cost_gbp"}
    if missing := required.difference(levy.columns):
        raise SupportToBillBridgeError(f"CfD levy recovery input is missing: {sorted(missing)}")
    frame = levy.copy()
    frame["month"] = pd.PeriodIndex(frame["month"], freq="M")
    frame = frame.loc[frame["month"].dt.year.eq(year)]
    expected = pd.period_range(f"{year}-01", f"{year}-12", freq="M")
    if set(frame["month"]) != set(expected) or frame["month"].duplicated().any():
        raise SupportToBillBridgeError("CfD levy recovery must contain one final value for every calendar month")
    demand = float(pd.to_numeric(frame["reconciled_eligible_demand_mwh"], errors="raise").sum())
    cost = float(pd.to_numeric(frame["reconciled_levy_cost_gbp"], errors="raise").sum())
    if demand <= 0:
        raise SupportToBillBridgeError("CfD levy recovery must have positive eligible demand")
    if not pd.notna(cost):
        raise SupportToBillBridgeError("CfD levy recovery must have a finite signed collected cost")
    return {"eligible_demand_mwh": demand, "recovered_cost_gbp": cost, "rate_gbp_per_mwh": cost / demand}


def _roc_scheme_year_values(
    monthly_roc: list[dict[str, Any]],
    monthly_starts: list[str],
    parameters: dict[str, Any],
    year: int,
) -> list[dict[str, Any]]:
    if len(monthly_roc) != len(monthly_starts) or len(monthly_roc) != 12:
        raise SupportToBillBridgeError("ROC scheme-year bridge requires twelve dated monthly reconciliations")
    periods = parameters.get("periods", [])
    if not isinstance(periods, list) or not periods:
        raise SupportToBillBridgeError("ROC scheme-value parameter file has no periods")
    values = []
    for period in periods:
        start = pd.Timestamp(period["calendar_start"])
        end = pd.Timestamp(period["calendar_end"])
        if start.year != year or end.year != year:
            continue
        values.append({
            **period,
            "start": start,
            "end": end,
            "reconciled_roc_certificates": 0.0,
            "strict_direct_roc_certificates": 0.0,
        })
    if len(values) != 2:
        raise SupportToBillBridgeError("ROC scheme-value parameters must cover both calendar-year bridge periods")
    for summary, start_text in zip(monthly_roc, monthly_starts):
        month_start = pd.Timestamp(start_text)
        matched = [value for value in values if value["start"] <= month_start <= value["end"]]
        if len(matched) != 1:
            raise SupportToBillBridgeError(f"No unique ROC scheme-year value for {month_start:%Y-%m}")
        matched[0]["reconciled_roc_certificates"] += float(summary["reconciled_roc_certificates"])
        matched[0]["strict_direct_roc_certificates"] += float(summary["strict_direct_roc_certificates"])
    for value in values:
        value["final_notional_roc_value_gbp_per_roc"] = float(value["final_notional_roc_value_gbp_per_roc"])
        value["reconciled_scheme_value_gbp"] = (
            value["reconciled_roc_certificates"] * value["final_notional_roc_value_gbp_per_roc"]
        )
    return values


def _validate_inputs(
    cfd: dict[str, Any],
    cfd_mapping: dict[str, Any],
    roc: dict[str, Any],
    roc_ledger: dict[str, Any],
    year: int,
    expected_months: int,
) -> None:
    if int(cfd.get("calendar_year", -1)) != year:
        raise SupportToBillBridgeError("CfD reconciliation calendar year does not match bridge year")
    if int(cfd_mapping.get("calendar_year", -1)) != year:
        raise SupportToBillBridgeError("CfD contract mapping calendar year does not match bridge year")
    if not cfd_mapping.get("contract_register_complete") or not cfd_mapping.get("model_mapping_complete"):
        raise SupportToBillBridgeError("CfD mapping is incomplete; it cannot enter a consumer-bill bridge")
    if int(cfd.get("solved_networks_aggregated", 0)) != expected_months:
        raise SupportToBillBridgeError("CfD reconciliation does not contain the expected number of monthly networks")
    if int(roc.get("solved_networks_aggregated", 0)) != expected_months:
        raise SupportToBillBridgeError("ROC reconciliation does not contain the expected number of monthly networks")
    if abs(float(roc_ledger.get("total_station_representation_coverage", 0.0)) - 1.0) > 1e-9:
        raise SupportToBillBridgeError("Official ROC ledger is not completely represented")
    cfd_comparator = float(cfd.get("actual_payment_gbp_mapped_contracts", 0.0))
    if not pd.notna(cfd_comparator):
        raise SupportToBillBridgeError("CfD reconciliation has no finite official payment comparator")
    if float(roc.get("ofgem_rocs_issued", 0.0)) <= 0:
        raise SupportToBillBridgeError("ROC reconciliation has no positive official certificate comparator")


def build_bridge(
    policy: pd.DataFrame,
    cfd: dict[str, Any],
    cfd_mapping: dict[str, Any],
    roc: dict[str, Any],
    roc_ledger: dict[str, Any],
    cfd_levy: pd.DataFrame,
    monthly_roc: list[dict[str, Any]],
    monthly_starts: list[str],
    roc_scheme_parameters: dict[str, Any],
    assumptions: dict[str, Any],
    year: int,
    expected_months: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Produce an auditable, non-substituting CfD/RO recovery bridge."""
    _validate_inputs(cfd, cfd_mapping, roc, roc_ledger, year, expected_months)

    cfd_cost = float(cfd["modelled_payment_independent_formula_gbp"])
    cfd_actual = float(cfd["actual_payment_gbp_mapped_contracts"])
    cfd_recovery = _cfd_levy_recovery(cfd_levy, year)
    roc_values = _roc_scheme_year_values(monthly_roc, monthly_starts, roc_scheme_parameters, year)
    roc_cost = sum(value["reconciled_scheme_value_gbp"] for value in roc_values)
    roc_reconciled = sum(value["reconciled_roc_certificates"] for value in roc_values)
    roc_direct = sum(value["strict_direct_roc_certificates"] for value in roc_values)
    rows = []
    for name, component, scheme_cost, actual_cost, direct_coverage, reconciled_coverage, actual_allowance, section, basis, caveat, candidate, status in [
        (
            "CfD",
            "Contracts for Difference",
            cfd_cost,
            cfd_recovery["recovered_cost_gbp"],
            cfd_cost / cfd_actual if cfd_actual else None,
            cfd_cost / cfd_recovery["recovered_cost_gbp"] if cfd_recovery["recovered_cost_gbp"] else None,
            cfd_recovery["rate_gbp_per_mwh"] * BENCHMARK_MWH,
            assumptions.get("cfd", {}),
            "Final physical dispatch × published daily indexed strike-minus-reference-price terms",
            "This uses LCCC's final reconciled levy demand for a same-year backtest. Future cases still need projected eligible supplier demand and the complete levy/reconciliation treatment.",
            cfd_cost / cfd_recovery["eligible_demand_mwh"] * BENCHMARK_MWH,
            "candidate_ready_for_same_year_backtest",
        ),
        (
            "RO",
            "Renewables Obligation",
            roc_cost,
            None,
            roc_direct / float(roc["ofgem_rocs_issued"]),
            roc_reconciled / float(roc["ofgem_rocs_issued"]),
            None,
            assumptions.get("ro", {}),
            "Monthly reconciled ROC certificates × final notional ROC value for the relevant RO obligation year",
            "This remains a certificate-value proxy. A household allowance requires supplier obligation volumes, buy-out/recycling cashflows and the supplied-energy denominator.",
            None,
            "modelled_certificates_pending_obligation_recovery",
        ),
    ]:
        if not isinstance(section, dict):
            raise SupportToBillBridgeError(f"{name} recovery assumptions must be an object")
        supplied_cost = _number(section, "modelled_recovery_cost_gbp")
        recovery_cost = scheme_cost if supplied_cost is None else supplied_cost
        if candidate is None:
            candidate = _household_candidate(recovery_cost, section)
            status = "candidate_ready_for_backtest" if candidate is not None else status
        rows.append({
            "scheme": name,
            "ofgem_policy_component": component,
            "published_allowance_gbp_per_3100kwh_ex_vat": _published_allowance(policy, name),
            "modelled_scheme_cost_gbp": scheme_cost,
            "official_comparator_cost_gbp": actual_cost,
            "official_recovery_allowance_gbp_per_3100kwh_ex_vat": actual_allowance,
            "direct_model_coverage_of_comparator": direct_coverage,
            "reconciled_coverage_of_comparator": reconciled_coverage,
            "recovery_cost_gbp": recovery_cost,
            "candidate_allowance_gbp_per_3100kwh_ex_vat": candidate,
            "pass_through_allowed": False,
            "recovery_status": status,
            "model_basis": basis,
            "remaining_gate": caveat,
        })

    table = pd.DataFrame(rows)
    # RO is recovered from supplier obligation rather than paid by a levy.  The
    # Ofgem price-cap RO rate already contains the applicable supplied-energy
    # denominator and scheme-year obligation.  Scale that published rate by
    # the explicit calendar-year ROC representation ratios, rather than
    # treating ROC generator income as a household charge directly.
    roc_mask = table["scheme"].eq("RO")
    roc_published = float(table.loc[roc_mask, "published_allowance_gbp_per_3100kwh_ex_vat"].iloc[0])
    roc_direct_allowance = roc_published * float(
        table.loc[roc_mask, "direct_model_coverage_of_comparator"].iloc[0]
    )
    roc_reconciled_allowance = roc_published * float(
        table.loc[roc_mask, "reconciled_coverage_of_comparator"].iloc[0]
    )
    table["direct_candidate_allowance_gbp_per_3100kwh_ex_vat"] = pd.NA
    table.loc[roc_mask, "direct_candidate_allowance_gbp_per_3100kwh_ex_vat"] = roc_direct_allowance
    table.loc[roc_mask, "candidate_allowance_gbp_per_3100kwh_ex_vat"] = roc_reconciled_allowance
    table.loc[roc_mask, "recovery_status"] = "candidate_ready_for_same_year_backtest_with_representation_calibration"
    table.loc[roc_mask, "remaining_gate"] = (
        "Same-year candidate uses Ofgem's supplied-energy obligation rate scaled by the explicit ROC "
        "representation bridge. Future cases must project only validated station cohorts and separately "
        "model supplier obligations, recycling and supplied energy."
    )
    table["same_year_backtest_allowed"] = table["recovery_status"].str.startswith(
        "candidate_ready_for_same_year_backtest"
    )
    summary = {
        "calendar_year": year,
        "purpose": "Annual CfD/ROC support-to-bill bridge. Its candidates may enter an explicitly labelled same-year backtest, but not a general consumer-bill pass-through.",
        "bridge_status": "candidates_ready_for_same_year_backtest" if table["same_year_backtest_allowed"].all() else "recovery_inputs_required",
        "pass_through_allowed": False,
        "input_validation": {
            "expected_monthly_networks": expected_months,
            "cfd_contract_register_complete": bool(cfd_mapping["contract_register_complete"]),
            "cfd_model_mapping_complete": bool(cfd_mapping["model_mapping_complete"]),
            "roc_official_ledger_coverage": float(roc_ledger["total_station_representation_coverage"]),
        },
        "modelled_support_coverage": {
            "cfd_independent_payment_coverage": float(table.loc[table.scheme.eq("CfD"), "direct_model_coverage_of_comparator"].iloc[0]),
            "roc_direct_certificate_coverage": float(table.loc[table.scheme.eq("RO"), "direct_model_coverage_of_comparator"].iloc[0]),
            "roc_reconciled_certificate_coverage": float(table.loc[table.scheme.eq("RO"), "reconciled_coverage_of_comparator"].iloc[0]),
        },
        "same_year_support_candidates_gbp_per_3100kwh_ex_vat": {
            row.scheme: float(row.candidate_allowance_gbp_per_3100kwh_ex_vat)
            for row in table.itertuples(index=False)
        },
        "ro_direct_physical_candidate_gbp_per_3100kwh_ex_vat": roc_direct_allowance,
        "cfd_levy_recovery": cfd_recovery,
        "cfd_actual_recovery_allowance_gbp_per_3100kwh_ex_vat": cfd_recovery["rate_gbp_per_mwh"] * BENCHMARK_MWH,
        "roc_scheme_year_values": [
            {key: value for key, value in period.items() if key not in {"start", "end"}}
            for period in roc_values
        ],
        "rule": "Candidates may replace the matching allowance only in an explicitly labelled same-year support backtest. A general consumer-bill pass-through still requires complete supplier-recovery inputs and out-of-sample validation.",
    }
    return table, summary


def _render_html(table: pd.DataFrame, summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row.scheme)}</td>"
        f"<td>£{row.published_allowance_gbp_per_3100kwh_ex_vat:,.2f}</td>"
        f"<td>£{row.modelled_scheme_cost_gbp:,.0f}</td>"
        f"<td>{row.direct_model_coverage_of_comparator:.1%}</td>"
        f"<td>{row.reconciled_coverage_of_comparator:.1%}</td>"
        f"<td>{'—' if pd.isna(row.candidate_allowance_gbp_per_3100kwh_ex_vat) else f'£{row.candidate_allowance_gbp_per_3100kwh_ex_vat:,.2f}'}</td>"
        f"<td>{html.escape(row.recovery_status)}</td></tr>"
        for row in table.itertuples(index=False)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>2020 support-to-bill bridge</title>
<style>body{{font-family:Arial;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dde3ea;padding:9px;text-align:left}}th{{background:#f3f6f9}}.gate{{background:#fff4e5;border-left:4px solid #c77d17;padding:12px}}</style></head><body>
<h1>2020 annual support-to-bill bridge</h1><p class="gate">{html.escape(summary['rule'])}</p>
<p>Status: <strong>{html.escape(summary['bridge_status'])}</strong>.</p>
<table><thead><tr><th>Scheme</th><th>Published allowance</th><th>Modelled scheme cost</th><th>Direct model coverage</th><th>Reconciled coverage</th><th>Household candidate</th><th>Recovery status</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def write_bridge(
    policy_path, cfd_path, cfd_mapping_path, roc_path, roc_ledger_path, cfd_levy_path,
    monthly_roc_paths, monthly_starts, roc_scheme_parameters_path, assumptions_path,
    output_csv, output_summary, output_html, year, expected_months,
) -> None:
    table, summary = build_bridge(
        pd.read_csv(policy_path), _load_json(cfd_path), _load_json(cfd_mapping_path),
        _load_json(roc_path), _load_json(roc_ledger_path), pd.read_csv(cfd_levy_path),
        [_load_json(path) for path in monthly_roc_paths], list(monthly_starts),
        _load_json(roc_scheme_parameters_path), _load_json(assumptions_path),
        int(year), int(expected_months),
    )
    for path in (output_csv, output_summary, output_html):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(output_html).write_text(_render_html(table, summary), encoding="utf-8")


if "snakemake" in globals():
    write_bridge(
        snakemake.input.policy_components_csv, snakemake.input.cfd_summary_json,
        snakemake.input.cfd_mapping_summary_json, snakemake.input.roc_summary_json,
        snakemake.input.roc_ledger_summary_json, snakemake.input.cfd_levy_csv,
        snakemake.input.monthly_roc_summaries, snakemake.params.monthly_starts,
        snakemake.input.roc_scheme_parameters_json, snakemake.input.assumptions_json,
        snakemake.output.bridge_csv, snakemake.output.summary_json, snakemake.output.report_html,
        snakemake.params.year, snakemake.params.expected_months,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-components", required=True); parser.add_argument("--cfd-summary", required=True)
    parser.add_argument("--cfd-mapping-summary", required=True); parser.add_argument("--roc-summary", required=True)
    parser.add_argument("--roc-ledger-summary", required=True); parser.add_argument("--cfd-levy", required=True)
    parser.add_argument("--monthly-roc-summaries", nargs="+", required=True); parser.add_argument("--monthly-starts", nargs="+", required=True)
    parser.add_argument("--roc-scheme-parameters", required=True); parser.add_argument("--assumptions", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True); parser.add_argument("--output-html", required=True)
    parser.add_argument("--year", type=int, required=True); parser.add_argument("--expected-months", type=int, default=12)
    args = parser.parse_args()
    write_bridge(args.policy_components, args.cfd_summary, args.cfd_mapping_summary, args.roc_summary, args.roc_ledger_summary, args.cfd_levy, args.monthly_roc_summaries, args.monthly_starts, args.roc_scheme_parameters, args.assumptions, args.output_csv, args.output_summary, args.output_html, args.year, args.expected_months)
