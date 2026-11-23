"""Create a readable, auditable consumer-electricity-bill breakdown.

The published Ofgem cost stack and the PyPSA wholesale-price sensitivity have
different household scopes in the 2020 data.  This report intentionally keeps
them separate: it never scales or adds the 3,100 kWh Other Payment Method
stack to the 2,500 kWh Direct Debit headline bill.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


class ConsumerBillBreakdownError(ValueError):
    """Raised when consumer-bill inputs cannot form an auditable statement."""


GROUPS = (
    (
        "Wholesale & hedging",
        ("hedged_wholesale", "hedge_uplifts"),
        "Published forward procurement and hedge-shaping allowances.",
    ),
    (
        "System operation, balancing & losses",
        ("system_balancing", "losses"),
        "Published balancing, Capacity Market, adjustment and losses allowances.",
    ),
    (
        "Policy costs",
        ("policy",),
        "Published policy allowance, decomposed into Ofgem Annex 4 scheme calculations where available.",
    ),
    (
        "Network costs",
        ("network",),
        "Published transmission and distribution network allowance.",
    ),
    (
        "Supplier, payment, margin & headroom",
        ("supplier_operating", "payment", "bad_debt", "margin", "headroom"),
        "Published supplier operating, payment-method, margin and headroom allowances.",
    ),
    ("VAT", ("vat",), "VAT shown separately at the published rate."),
)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ConsumerBillBreakdownError(f"{label} is missing columns: {sorted(missing)}")


def _single_bill(bills: pd.DataFrame) -> pd.Series:
    required = {
        "annual_consumption_kwh",
        "payment_method",
        "region",
        "vat_rate",
        "standing_charge_gbp_ex_vat",
        "base_energy_charge_gbp_ex_vat",
        "wholesale_price_delta_gbp_ex_vat",
        "consumer_bill_gbp_ex_vat",
        "consumer_bill_gbp_inc_vat",
    }
    _require_columns(bills, required, "Consumer-bill output")
    if len(bills) != 1:
        raise ConsumerBillBreakdownError(
            "Consumer-bill breakdown currently requires one household selection; "
            "configure one consumption/payment-method/region combination."
        )
    return bills.iloc[0]


def _policy_component_rows(policy_components: pd.DataFrame, expected_total: float) -> pd.DataFrame:
    """Return annualised Annex 4 policy schemes after reconciliation."""
    _require_columns(
        policy_components,
        {"component", "component_label", "annualised_gbp_per_3100kwh_ex_vat"},
        "Ofgem policy-component extract",
    )
    rows = policy_components.copy()
    rows["annualised_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(
        rows["annualised_gbp_per_3100kwh_ex_vat"], errors="raise"
    )
    annual = rows.groupby(["component", "component_label"], as_index=False)[
        "annualised_gbp_per_3100kwh_ex_vat"
    ].sum()
    actual_total = float(annual["annualised_gbp_per_3100kwh_ex_vat"].sum())
    if abs(actual_total - expected_total) > 1e-5:
        raise ConsumerBillBreakdownError(
            "Ofgem Annex 4 policy components do not reconcile to the published policy allowance: "
            f"{actual_total} != {expected_total}"
        )
    return annual


def _consumer_bill_breakdown(bill: pd.Series, components: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Group an already allocated consumer-bill register without changing its scope."""
    required = {"stack_group", "component", "annual_gbp_ex_vat", "allocation_method", "is_modelled"}
    _require_columns(components, required, "Consumer-bill component register")
    rows = components.copy()
    rows["annual_gbp_ex_vat"] = pd.to_numeric(rows["annual_gbp_ex_vat"], errors="raise")
    if not np.isfinite(rows["annual_gbp_ex_vat"]).all():
        raise ConsumerBillBreakdownError("Bill register contains missing or non-finite amounts")
    rows["presentation_stack_group"] = rows["stack_group"]
    # A presentation regrouping only: preserve the source category and never
    # append a second CfD charge when it is embedded in direct fuel.
    rows.loc[rows["component"].str.contains("CfD", case=False, na=False), "presentation_stack_group"] = "policy"
    rows.loc[rows["component"].str.contains("BSUoS", case=False, na=False), "presentation_stack_group"] = "system_balancing"
    expected_groups = {group for _, source_groups, _ in GROUPS for group in source_groups}
    unknown = sorted(set(rows["stack_group"]).difference(expected_groups))
    if unknown:
        raise ConsumerBillBreakdownError(f"Consumer-bill register has ungrouped rows: {unknown}")
    total = float(rows["annual_gbp_ex_vat"].sum())
    expected = float(bill["consumer_bill_gbp_inc_vat"])
    if not np.isfinite(expected) or total <= 0:
        raise ConsumerBillBreakdownError("Selected bill total must be finite and positive")
    if abs(total - expected) > 1e-6:
        raise ConsumerBillBreakdownError(f"Consumer-bill register does not reconcile: {total} != {expected}")
    vat = float(rows.loc[rows["stack_group"].eq("vat"), "annual_gbp_ex_vat"].sum())
    groups = []
    output_rows = []
    for order, (name, source_groups, note) in enumerate(GROUPS, start=1):
        group_rows = rows.loc[rows["presentation_stack_group"].isin(source_groups)].copy()
        amount = float(group_rows["annual_gbp_ex_vat"].sum())
        groups.append({"order": order, "group": name, "annual_gbp_per_3100kwh": amount, "share_of_total_inc_vat": amount / total, "note": note})
        for _, component in group_rows.iterrows():
            output_rows.append({
                "group_order": order, "group": name, "source_stack_group": component["stack_group"],
                "component": component["component"], "annual_gbp_per_3100kwh": component["annual_gbp_ex_vat"],
                "share_of_total_inc_vat": component["annual_gbp_ex_vat"] / total,
                "group_note": note, "source_scope": "Selected consumer-bill tariff allocation",
                "allocation_method": component["allocation_method"], "is_modelled": bool(component["is_modelled"]),
                "annual_gbp_at_selected_consumption": component["annual_gbp_ex_vat"],
                "annual_consumption_kwh": float(bill["annual_consumption_kwh"]),
                **{key: component.get(key, "") for key in ("component_id", "source_workbook", "source_cells", "detail_status", "charge_basis")},
            })
    cfd = rows.loc[rows["component"].astype(str).str.contains("CfD", case=False, na=False), "annual_gbp_ex_vat"]
    wholesale_delta = float(bill["wholesale_price_delta_gbp_ex_vat"])
    baseline = float(bill["consumer_bill_gbp_ex_vat"]) - wholesale_delta
    summary = {
        "bill_basis": str(bill.get("bill_basis", "tariff_baseline_with_optional_wholesale_sensitivity")),
        "input_mode": str(bill.get("input_mode", "unspecified")),
        "modelled_year": int(bill["modelled_year"]) if "modelled_year" in bill else None,
        "currency": str(bill.get("currency", "GBP; see source bundle for price basis")),
        "headline_bill": {
            "scope": {"annual_consumption_kwh": float(bill["annual_consumption_kwh"]), "payment_method": str(bill["payment_method"]), "region": str(bill["region"])},
            "published_tariff_baseline_gbp_inc_vat": baseline * (1 + float(bill["vat_rate"])),
            "modelled_wholesale_price_effect_gbp_inc_vat": wholesale_delta * (1 + float(bill["vat_rate"])),
            "pypsa_informed_bill_gbp_inc_vat": expected,
            "standing_charge_gbp_ex_vat": float(bill["standing_charge_gbp_ex_vat"]),
            "published_variable_tariff_gbp_ex_vat": float(bill["base_energy_charge_gbp_ex_vat"]),
            "note": "Every component is allocated to the selected tariff. Modelled recovery candidates are separate audit outputs; only an explicitly labelled wholesale-price effect changes this headline bill.",
        },
        "published_cost_stack": {"scope": "Selected region, payment method and consumption; detailed components aligned to the selected tariff benchmark", "subtotal_gbp_ex_vat": total - vat, "vat_gbp": vat, "total_gbp_inc_vat": total, "groups": groups},
        "cfd_treatment": {
            "published_bill": f"The consumer-bill allocation contains a source-bundle CfD allowance of £{float(cfd.sum()):.2f}, ex VAT." if not cfd.empty else "No CfD component was found.",
            "modelled_bill": "CfD is shown once under policy costs, retaining its original source category in the CSV. Any modelled CfD recovery candidate remains separate from this headline bill.",
        },
        "limitations": [
            "The selected tariff total and standing charge reconcile exactly to Ofgem's published tariff snapshot.",
            "Ofgem does not publish a standing/unit split for mixed supplier, payment, margin and headroom components; those rows are transparently allocated in proportion to their published component values.",
            "CfD remains a published price-cap allowance, not an LCCC outturn or PyPSA levy payment.",
        ],
    }
    return pd.DataFrame(output_rows).sort_values(["group_order", "component"]).reset_index(drop=True), summary


def build_breakdown(
    bills: pd.DataFrame,
    cost_stack: pd.DataFrame,
    policy_components: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return grouped published components and a scoped headline-bill summary."""
    bill = _single_bill(bills)
    _require_columns(
        cost_stack,
        {"stack_group", "component"},
        "Retail cost stack",
    )

    if "annual_gbp_ex_vat" in cost_stack.columns:
        return _consumer_bill_breakdown(bill, cost_stack)
    _require_columns(cost_stack, {"annual_gbp_per_3100kwh_ex_vat"}, "Retail cost stack")

    stack = cost_stack.copy()
    stack["annual_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(
        stack["annual_gbp_per_3100kwh_ex_vat"], errors="raise"
    )
    components = stack.loc[~stack["stack_group"].eq("total")].copy()
    expected_groups = {group for _, source_groups, _ in GROUPS for group in source_groups}
    unknown = sorted(set(components["stack_group"]).difference(expected_groups))
    if unknown:
        raise ConsumerBillBreakdownError(f"Retail cost stack has ungrouped rows: {unknown}")

    vat_rows = components.loc[components["stack_group"].eq("vat")]
    if len(vat_rows) != 1:
        raise ConsumerBillBreakdownError("Retail cost stack must contain exactly one VAT row")
    subtotal = float(components.loc[~components["stack_group"].eq("vat"), "annual_gbp_per_3100kwh_ex_vat"].sum())
    vat = float(vat_rows["annual_gbp_per_3100kwh_ex_vat"].iloc[0])
    total_inc_vat = subtotal + vat
    if total_inc_vat <= 0:
        raise ConsumerBillBreakdownError("Published retail cost stack total must be positive")

    policy_annual = None
    if policy_components is not None:
        published_policy = float(
            components.loc[components["stack_group"].eq("policy"), "annual_gbp_per_3100kwh_ex_vat"].sum()
        )
        policy_annual = _policy_component_rows(policy_components, published_policy)

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for order, (name, source_groups, note) in enumerate(GROUPS, start=1):
        grouped = components.loc[components["stack_group"].isin(source_groups)]
        amount = float(grouped["annual_gbp_per_3100kwh_ex_vat"].sum())
        groups.append({
            "order": order,
            "group": name,
            "annual_gbp_per_3100kwh": amount,
            "share_of_total_inc_vat": amount / total_inc_vat,
            "note": note,
        })
        if name == "Policy costs" and policy_annual is not None:
            for _, component in policy_annual.iterrows():
                rows.append({
                    "group_order": order,
                    "group": name,
                    "source_stack_group": "policy",
                    "component": f"Ofgem Annex 4 — {component['component_label']}",
                    "annual_gbp_per_3100kwh": float(component["annualised_gbp_per_3100kwh_ex_vat"]),
                    "share_of_total_inc_vat": float(component["annualised_gbp_per_3100kwh_ex_vat"]) / total_inc_vat,
                    "group_note": note,
                    "source_scope": "Published 2020 Ofgem Annex 4, GB-average, single-rate electricity, 3,100 kWh benchmark",
                })
            continue
        for _, component in grouped.iterrows():
            rows.append({
                "group_order": order,
                "group": name,
                "source_stack_group": component["stack_group"],
                "component": component["component"],
                "annual_gbp_per_3100kwh": float(component["annual_gbp_per_3100kwh_ex_vat"]),
                "share_of_total_inc_vat": float(component["annual_gbp_per_3100kwh_ex_vat"]) / total_inc_vat,
                "group_note": note,
                "source_scope": "Published 2020 Ofgem GB-average, Other Payment Method, 3,100 kWh benchmark",
            })

    wholesale_delta_ex_vat = float(bill["wholesale_price_delta_gbp_ex_vat"])
    vat_rate = float(bill["vat_rate"])
    baseline_ex_vat = float(bill["consumer_bill_gbp_ex_vat"]) - wholesale_delta_ex_vat
    baseline_inc_vat = baseline_ex_vat * (1 + vat_rate)
    summary: dict[str, Any] = {
        "headline_bill": {
            "scope": {
                "annual_consumption_kwh": float(bill["annual_consumption_kwh"]),
                "payment_method": str(bill["payment_method"]),
                "region": str(bill["region"]),
            },
            "published_tariff_baseline_gbp_inc_vat": baseline_inc_vat,
            "modelled_wholesale_price_effect_gbp_inc_vat": wholesale_delta_ex_vat * (1 + vat_rate),
            "pypsa_informed_bill_gbp_inc_vat": float(bill["consumer_bill_gbp_inc_vat"]),
            "standing_charge_gbp_ex_vat": float(bill["standing_charge_gbp_ex_vat"]),
            "published_variable_tariff_gbp_ex_vat": float(bill["base_energy_charge_gbp_ex_vat"]),
            "note": "Only the wholesale-price effect is modelled; all other published tariff allowances are held fixed.",
        },
        "published_cost_stack": {
            "scope": "GB-average, Other Payment Method, single-rate electricity, 3,100 kWh benchmark",
            "subtotal_gbp_ex_vat": subtotal,
            "vat_gbp": vat,
            "total_gbp_inc_vat": total_inc_vat,
            "groups": groups,
        },
        "cfd_treatment": {
            "published_bill": (
                f"Ofgem's 2020 Annex 4 calculation identifies a calendar-year CfD allowance of "
                f"£{float(policy_annual.loc[policy_annual['component'].eq('CfD'), 'annualised_gbp_per_3100kwh_ex_vat'].iloc[0]):.2f} "
                "per 3,100 kWh customer, ex VAT. This is a published cap allowance, not an LCCC outturn."
                if policy_annual is not None and policy_annual["component"].eq("CfD").any()
                else "The published policy allowance has not yet been decomposed into Annex 4 schemes."
            ),
            "modelled_bill": "The completed 2020 run changes the wholesale-price component only; it has no modelled CfD levy payment.",
        },
        "limitations": [
            "The headline tariff sensitivity and the detailed published stack use different household/payment scopes and must not be combined.",
            "The published CfD allowance is a price-cap baseline; it is not a modelled CfD levy, payment or payback.",
            "This is an Ofgem-structured wholesale sensitivity, not a statutory price-cap recalculation or an observed household bill.",
        ],
    }
    return pd.DataFrame(rows).sort_values(["group_order", "component"]).reset_index(drop=True), summary


def _money(value: float) -> str:
    return f"£{value:,.2f}"


def render_html(summary: dict[str, Any], components: pd.DataFrame) -> str:
    """Render a dependency-free static HTML statement suitable for sharing."""
    headline = summary["headline_bill"]
    stack = summary["published_cost_stack"]
    cards = [
        ("Tariff baseline", headline["published_tariff_baseline_gbp_inc_vat"]),
        ("Modelled wholesale effect", headline["modelled_wholesale_price_effect_gbp_inc_vat"]),
        ("PyPSA-informed bill", headline["pypsa_informed_bill_gbp_inc_vat"]),
    ]
    card_html = "".join(
        f'<section class="card"><span>{html.escape(label)}</span><strong>{_money(amount)}</strong></section>'
        for label, amount in cards
    )
    group_html = "".join(
        "<tr>"
        f"<th>{html.escape(group['group'])}</th>"
        f"<td>{_money(group['annual_gbp_per_3100kwh'])}</td>"
        f"<td>{group['share_of_total_inc_vat']:.1%}</td>"
        f"<td>{html.escape(group['note'])}</td>"
        "</tr>"
        for group in stack["groups"]
    )
    component_html = "".join(
        "<tr>"
        f"<td>{html.escape(row.group)}</td>"
        f"<td>{html.escape(row.component)}</td>"
        f"<td>{_money(row.annual_gbp_per_3100kwh)}</td>"
        "</tr>"
        for row in components.itertuples(index=False)
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in summary["limitations"])
    scope = headline["scope"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Consumer electricity bill breakdown</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1060px; margin: 32px auto; color: #172033; line-height: 1.45; padding: 0 20px; }}
h1 {{ margin-bottom: 4px; }} .muted {{ color: #596579; }} .cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 24px 0; }}
.card {{ border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; flex: 1; min-width: 200px; background: #f8fafc; }}
.card span {{ display: block; color: #596579; font-size: 0.9rem; }} .card strong {{ display: block; font-size: 1.75rem; margin-top: 6px; }}
.card:last-child {{ background: #e7f5ee; border-color: #9bd2b3; }} table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ padding: 10px; border-bottom: 1px solid #dde3ea; text-align: left; vertical-align: top; }} th {{ background: #f3f6f9; }} td:nth-child(2) {{ white-space: nowrap; }}
.callout {{ border-left: 4px solid #7560b5; padding: 12px 16px; background: #f7f4ff; }} details {{ margin-top: 20px; }}
</style></head><body>
<h1>Consumer electricity bill breakdown</h1>
<p>{html.escape(str(summary.get("bill_basis", "Historical benchmark")))} | {html.escape(str(summary.get("currency", "GBP")))}</p>
<p class="muted">Headline: {scope['annual_consumption_kwh']:,.0f} kWh/year, {html.escape(scope['payment_method'])}, {html.escape(scope['region'])}. Components: {html.escape(stack['scope'])}.</p>
<div class="cards">{card_html}</div>
<p class="callout"><strong>Model scope:</strong> {html.escape(headline['note'])}</p>
<h2>Electricity-bill components</h2>
<p>Annual bill components: <strong>{_money(stack['total_gbp_inc_vat'])} including VAT</strong>. Group shares use that total.</p>
<table><thead><tr><th>Group</th><th>Annual cost</th><th>Share</th><th>What it covers</th></tr></thead><tbody>{group_html}</tbody></table>
<p class="callout"><strong>CfD treatment:</strong> {html.escape(summary['cfd_treatment']['published_bill'])} {html.escape(summary['cfd_treatment']['modelled_bill'])}</p>
<details><summary>Show the complete component audit trail</summary><table><thead><tr><th>Group</th><th>Component</th><th>Annual cost</th></tr></thead><tbody>{component_html}</tbody></table></details>
<h2>Important scope notes</h2><ul>{limitations}</ul>
</body></html>"""


def write_breakdown(
    bills_path: str | Path,
    stack_path: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    output_html: str | Path,
    policy_components_path: str | Path | None = None,
) -> None:
    """Build and write all consumer-bill breakdown artifacts."""
    policy_components = pd.read_csv(policy_components_path) if policy_components_path else None
    components, summary = build_breakdown(pd.read_csv(bills_path), pd.read_csv(stack_path), policy_components)
    for output in (output_csv, output_summary, output_html):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_csv, index=False)
    with open(output_summary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    Path(output_html).write_text(render_html(summary, components), encoding="utf-8")


def _main(snakemake) -> None:
    write_breakdown(
        snakemake.input.bills_csv,
        snakemake.input.retail_stack_csv,
        snakemake.output.breakdown_csv,
        snakemake.output.summary_json,
        snakemake.output.report_html,
        snakemake.input.policy_components_csv,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bills", required=True, help="Consumer-bill CSV")
    parser.add_argument("--cost-stack", required=True, help="Retail cost-stack CSV")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--policy-components")
    args = parser.parse_args()
    write_breakdown(
        args.bills, args.cost_stack, args.output_csv, args.output_summary, args.output_html,
        args.policy_components,
    )


if "snakemake" in globals():
    _main(snakemake)
elif __name__ == "__main__":
    _cli()
