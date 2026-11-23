"""Audit which consumer-bill components are evidenced for a PyPSA scenario.

This is deliberately a *gate*, not a bill calculator.  It makes the boundary
between published Ofgem allowances, model outputs and unevidenced pass-through
assumptions machine-readable before any new component is added to a household
bill sensitivity.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


class ConsumerBillMethodologyAuditError(ValueError):
    """Raised when an audit cannot establish its required source scopes."""


STACK_MAPPING = {
    "hedged_wholesale": (
        "Wholesale index",
        "Wholesale and additional wholesale allowances",
        "published_annual_component",
        "held_fixed",
        "Reproduced from observed Ofgem forward procurement for the selected year; validate over multiple historical years before future pass-through.",
    ),
    "hedge_uplifts": (
        "Wholesale shaping, transaction and risk allowances",
        "Wholesale and additional wholesale allowances",
        "published_annual_component",
        "held_fixed",
        "Reproduced from observed Ofgem forward procurement for the selected year; validate over multiple historical years before future pass-through.",
    ),
    "losses": (
        "Electricity losses",
        "Wholesale and additional wholesale allowances",
        "published_annual_component",
        "held_fixed",
        "Do not change this from nodal redispatch alone; it needs supplier-loss methodology.",
    ),
    "system_balancing": (
        "Imbalance, Capacity Market and adjustment allowances",
        "Historic selected-year source mapping to be verified",
        "published_annual_component",
        "held_fixed",
        "Confirm each historic cap-period treatment before mapping it to current Ofgem categories.",
    ),
    "policy": (
        "Policy costs",
        "Policy allowance",
        "published_annual_component",
        "held_fixed",
        "This aggregate is not a CfD value. Reconcile it to the historic Annex 4 source before disaggregation.",
    ),
    "network": (
        "Network costs",
        "Network allowance",
        "published_annual_component",
        "held_fixed",
        "Keep TNUoS, DUoS and BSUoS treatment separate; modelled redispatch is not a replacement for this allowance.",
    ),
    "supplier_operating": (
        "Supplier operating and smart-metering costs",
        "Operating, debt and industry costs",
        "published_annual_component",
        "held_fixed",
        "Requires a separate supplier-cost assumption before any counterfactual change.",
    ),
    "payment": (
        "Payment-method costs",
        "Payment-method and levelisation allowances",
        "published_annual_component",
        "held_fixed",
        "Keep payment method and region explicit in any consumer result.",
    ),
    "bad_debt": (
        "Bad debt",
        "Operating, debt and industry costs",
        "not_separately_published_in_bundle",
        "held_fixed",
        "No separate selected-year published line is available in the current stack.",
    ),
    "margin": (
        "EBIT allowance",
        "EBIT allowance",
        "published_annual_component",
        "held_fixed",
        "Retain as a separate residual allowance.",
    ),
    "headroom": (
        "Headroom allowance",
        "Headroom allowance",
        "published_annual_component",
        "held_fixed",
        "Retain as a separate residual allowance.",
    ),
    "vat": (
        "VAT",
        "VAT",
        "published_annual_component",
        "held_fixed",
        "Apply only after the ex-VAT stack has been reconciled.",
    ),
}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ConsumerBillMethodologyAuditError(f"{label} is missing columns: {sorted(missing)}")


def _modelled_redispatch(market_summary: dict[str, Any] | None) -> float | None:
    if not market_summary:
        return None
    value = market_summary.get("redispatch", {}).get("total_net_cost")
    return float(value) if value is not None else None


def _modelled_cfd(market_summary: dict[str, Any] | None) -> float | None:
    if not market_summary:
        return None
    value = market_summary.get("revenue", {}).get("cfd_total_cost")
    return float(value) if value is not None else None


def _published_cfd(policy_components: pd.DataFrame | None) -> float | None:
    if policy_components is None:
        return None
    _require_columns(
        policy_components,
        {"component", "annualised_gbp_per_3100kwh_ex_vat"},
        "Ofgem policy-component extract",
    )
    rows = policy_components.loc[policy_components["component"].eq("CfD")].copy()
    if rows.empty:
        return None
    return float(pd.to_numeric(rows["annualised_gbp_per_3100kwh_ex_vat"], errors="raise").sum())


def build_methodology_audit(
    cost_stack: pd.DataFrame,
    market_summary: dict[str, Any] | None = None,
    policy_components: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map published components and model results to verified bill treatment."""
    _require_columns(
        cost_stack,
        {"stack_group", "component", "annual_gbp_per_3100kwh_ex_vat"},
        "Retail cost stack",
    )
    stack = cost_stack.loc[~cost_stack["stack_group"].eq("total")].copy()
    stack["annual_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(
        stack["annual_gbp_per_3100kwh_ex_vat"], errors="raise"
    )
    unknown = sorted(set(stack["stack_group"]).difference(STACK_MAPPING))
    if unknown:
        raise ConsumerBillMethodologyAuditError(f"Unmapped cost-stack groups: {unknown}")

    rows: list[dict[str, Any]] = []
    for _, item in stack.iterrows():
        label, ofgem_parent, evidence_status, treatment, next_step = STACK_MAPPING[item["stack_group"]]
        rows.append({
            "audit_area": label,
            "component": item["component"],
            "ofgem_parent": ofgem_parent,
            "published_amount": float(item["annual_gbp_per_3100kwh_ex_vat"]),
            "published_amount_unit": "GBP per 3,100 kWh benchmark customer, ex VAT",
            "modelled_amount": None,
            "modelled_amount_unit": None,
            "evidence_status": evidence_status,
            "consumer_bill_treatment": treatment,
            "next_validation_step": next_step,
        })

    redispatch = _modelled_redispatch(market_summary)
    cfd = _modelled_cfd(market_summary)
    published_cfd = _published_cfd(policy_components)
    cfd_parent = "Policy Cost Allowance (Ofgem Annex 4)"
    embedded = stack.loc[stack["component"].str.contains("CfD", case=False, na=False)]
    if not embedded.empty:
        if published_cfd is not None and abs(published_cfd) > 1e-8:
            raise ConsumerBillMethodologyAuditError("CfD appears in both direct fuel and policy; reconcile before reporting")
        published_cfd = float(embedded["annual_gbp_per_3100kwh_ex_vat"].sum())
        cfd_parent = "CfD embedded in direct fuel (wholesale allowance)"
    rows.extend([
        {
            "audit_area": "CfD Supplier Obligation Levy",
            "component": "CfD generator payments and paybacks",
            "ofgem_parent": cfd_parent,
            "published_amount": published_cfd,
            "published_amount_unit": "GBP per 3,100 kWh benchmark customer, ex VAT" if published_cfd is not None else None,
            "modelled_amount": cfd,
            "modelled_amount_unit": "GBP system total" if cfd is not None else None,
            "evidence_status": (
                "model_output_requires_levy_reconciliation" if cfd is not None
                else "published_annual_annex_4_allowance" if published_cfd is not None
                else "missing_historic_cfd_levy_and_contract_coverage"
            ),
            "consumer_bill_treatment": "pass_through_disabled",
            "next_validation_step": "Use the published Annex 4 allowance as the baseline; reconcile generator-level payments and LCCC Supplier Obligation Levy outturn before converting a modelled CfD result to a household change.",
        },
        {
            "audit_area": "Balancing / redispatch recovery",
            "component": "PyPSA two-stage redispatch net cost",
            "ofgem_parent": "Network allowance (BSUoS route requires historic verification)",
            "published_amount": None,
            "published_amount_unit": None,
            "modelled_amount": redispatch,
            "modelled_amount_unit": "GBP system total" if redispatch is not None else None,
            "evidence_status": "model_output_requires_bsuos_reconciliation" if redispatch is not None else "missing_market_summary",
            "consumer_bill_treatment": "pass_through_disabled",
            "next_validation_step": "Reconcile PyPSA redispatch with historic NESO balancing/BSUoS costs and the applicable supplier-recovery route; do not replace TNUoS or DUoS.",
        },
    ])
    frame = pd.DataFrame(rows)
    summary = {
        "audit_result": "not_ready_for_component_pass_through",
        "published_stack_scope": "GB-average, Other Payment Method, single-rate electricity, 3,100 kWh benchmark",
        "modelled_component_pass_through_enabled": [],
        "evidence_gaps": [
            "The published selected-year CfD allowance is a baseline, not historic LCCC Supplier Obligation Levy outturn or generator contract coverage.",
            "PyPSA redispatch cost has no validated mapping to historic BSUoS or a supplier household-recovery factor.",
            "The selected-year forward-procurement reconstruction uses observed Ofgem inputs; a future PyPSA pass-through remains disabled pending multi-year validation.",
        ],
        "modelled_results_seen": {
            "published_cfd_allowance_gbp_per_3100kwh_ex_vat": published_cfd,
            "cfd_total_cost_gbp": cfd,
            "redispatch_total_net_cost_gbp": redispatch,
        },
    }
    return frame, summary


def render_html(audit: pd.DataFrame, summary: dict[str, Any]) -> str:
    """Render a small static audit report with visible implementation gates."""
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.audit_area))}</td>"
        f"<td>{html.escape(str(row.component))}</td>"
        f"<td>{html.escape(str(row.ofgem_parent))}</td>"
        f"<td>{html.escape(str(row.evidence_status))}</td>"
        f"<td>{html.escape(str(row.consumer_bill_treatment))}</td>"
        f"<td>{html.escape(str(row.next_validation_step))}</td>"
        "</tr>"
        for row in audit.itertuples(index=False)
    )
    gaps = "".join(f"<li>{html.escape(item)}</li>" for item in summary["evidence_gaps"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Consumer-bill methodology audit</title><style>
body {{ font-family: Arial, sans-serif; max-width: 1240px; margin: 32px auto; color: #172033; padding: 0 20px; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }} th,td {{ border-bottom: 1px solid #dce2ea; padding: 9px; text-align: left; vertical-align: top; }} th {{ background: #f3f6f9; }}
.gate {{ background: #fff4e5; border-left: 4px solid #c77d17; padding: 14px 16px; }} code {{ background: #eef2f7; padding: 2px 4px; }}
</style></head><body><h1>Consumer-bill methodology audit</h1>
<p class="gate"><strong>Gate result:</strong> <code>{html.escape(summary['audit_result'])}</code>. No new modelled component is passed through to a household bill until its evidence gap is closed.</p>
<h2>Evidence gaps</h2><ul>{gaps}</ul><h2>Component register</h2>
<table><thead><tr><th>Area</th><th>Component</th><th>Ofgem parent</th><th>Evidence</th><th>Bill treatment</th><th>Required validation</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def write_methodology_audit(
    cost_stack_path: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    output_html: str | Path,
    market_summary_path: str | Path | None = None,
    policy_components_path: str | Path | None = None,
) -> None:
    """Build and write audit artifacts from published and model outputs."""
    market_summary = None
    if market_summary_path:
        path = Path(market_summary_path)
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                market_summary = json.load(stream)
    policy_components = pd.read_csv(policy_components_path) if policy_components_path else None
    audit, summary = build_methodology_audit(pd.read_csv(cost_stack_path), market_summary, policy_components)
    for output in (output_csv, output_summary, output_html):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_csv, index=False)
    with open(output_summary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    Path(output_html).write_text(render_html(audit, summary), encoding="utf-8")


def _main(snakemake) -> None:
    summary_path = str(snakemake.input.market_summary_json) if snakemake.input.market_summary_json else None
    write_methodology_audit(
        snakemake.input.retail_stack_csv,
        snakemake.output.audit_csv,
        snakemake.output.summary_json,
        snakemake.output.report_html,
        summary_path,
        snakemake.input.policy_components_csv,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-stack", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--market-summary")
    parser.add_argument("--policy-components")
    args = parser.parse_args()
    write_methodology_audit(
        args.cost_stack, args.output_csv, args.output_summary, args.output_html,
        args.market_summary, args.policy_components,
    )


if "snakemake" in globals():
    _main(snakemake)
elif __name__ == "__main__":
    _cli()
