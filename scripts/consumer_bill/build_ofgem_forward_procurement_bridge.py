"""Reconstruct the 2020 Ofgem electricity forward-procurement allowance.

This is the annual historical implementation of the consumer-bill wholesale method:
observed forward-contract inputs and Ofgem's named conversion, transaction,
risk and loss allowances are annualised over their delivery periods.  PyPSA
prices are retained only as an operational diagnostic.  They are not used to
turn a spot price into a supplier hedge price.
"""


import argparse
import html
import json
from pathlib import Path

import pandas as pd


COMPONENT_MAP = {
    "seasonal_base_forward_index": "Published seasonal base forward allowance",
    "seasonal_peak_forward_index": "Published seasonal peak forward allowance",
    "seasonal_to_monthly_shaping": "Seasonal-to-monthly shaping",
    "monthly_to_hourly_shaping": "Monthly peak/base to hourly shaping",
    "day_ahead_rehedging": "Day-ahead re-hedging / forecast error",
    "transaction_cost": "Transaction-cost uplift",
    "additional_risk_and_basis": "Additional risk / basis allowance",
    "regional_losses_gb_average": "GB-average regional electricity losses",
}


def build_ofgem_forward_procurement_bridge(
    register: pd.DataFrame,
    procurement_periods: pd.DataFrame,
    wholesale_impact: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Annualise observed Ofgem forward procurement and reconcile it to the stack."""
    required_periods = {
        "cap_period", "delivery_start", "delivery_end", "component",
        "gbp_per_benchmark_customer_ex_vat",
    }
    if missing := required_periods.difference(procurement_periods.columns):
        raise ValueError(f"Forward-procurement periods are missing: {sorted(missing)}")
    if missing := {"component", "annual_gbp_per_3100kwh_ex_vat"}.difference(register.columns):
        raise ValueError(f"Component register is missing: {sorted(missing)}")
    required_wholesale = {
        "cap_period", "modelled_demand_mwh",
        "modelled_demand_weighted_clearing_price_gbp_per_mwh",
        "published_forward_index_gbp_per_mwh",
    }
    if missing := required_wholesale.difference(wholesale_impact.columns):
        raise ValueError(f"Wholesale diagnostic is missing: {sorted(missing)}")

    periods = procurement_periods.loc[
        procurement_periods["component"].isin(COMPONENT_MAP)
    ].copy()
    periods["delivery_start"] = pd.to_datetime(periods["delivery_start"], errors="raise")
    periods["delivery_end"] = pd.to_datetime(periods["delivery_end"], errors="raise")
    periods["gbp_per_benchmark_customer_ex_vat"] = pd.to_numeric(
        periods["gbp_per_benchmark_customer_ex_vat"], errors="raise"
    )
    periods["delivery_days"] = (periods["delivery_end"] - periods["delivery_start"]).dt.days + 1
    if (periods["delivery_days"] <= 0).any():
        raise ValueError("Forward-procurement delivery periods must be positive")
    expected_periods = periods["cap_period"].drop_duplicates()
    if periods.groupby("component")["cap_period"].nunique().ne(len(expected_periods)).any():
        raise ValueError("Every procurement component must cover every delivery period")

    period_days = periods[["cap_period", "delivery_days"]].drop_duplicates()
    period_days["annual_delivery_share"] = period_days["delivery_days"] / period_days["delivery_days"].sum()
    candidate = periods.merge(period_days[["cap_period", "annual_delivery_share"]], on="cap_period", validate="many_to_one")
    candidate["annualised_gbp_per_3100kwh_ex_vat"] = (
        candidate["gbp_per_benchmark_customer_ex_vat"] * candidate["annual_delivery_share"]
    )
    candidate = candidate.groupby("component", as_index=False)[
        "annualised_gbp_per_3100kwh_ex_vat"
    ].sum()
    candidate["bill_component"] = candidate["component"].map(COMPONENT_MAP)

    published = register[["component", "annual_gbp_per_3100kwh_ex_vat"]].copy()
    published["annual_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(
        published["annual_gbp_per_3100kwh_ex_vat"], errors="raise"
    )
    output = candidate.merge(
        published, left_on="bill_component", right_on="component", how="left", validate="one_to_one",
        suffixes=("_source", "_published"),
    ).drop(columns="component_published")
    if output["annual_gbp_per_3100kwh_ex_vat"].isna().any():
        missing = output.loc[output["annual_gbp_per_3100kwh_ex_vat"].isna(), "bill_component"].tolist()
        raise ValueError(f"Component register is missing forward-procurement rows: {missing}")
    output = output.rename(columns={
        "component_source": "source_component",
        "annual_gbp_per_3100kwh_ex_vat": "published_allowance_gbp_per_3100kwh_ex_vat",
        "annualised_gbp_per_3100kwh_ex_vat": "candidate_allowance_gbp_per_3100kwh_ex_vat",
    })
    output["difference_gbp_per_3100kwh_ex_vat"] = (
        output["candidate_allowance_gbp_per_3100kwh_ex_vat"]
        - output["published_allowance_gbp_per_3100kwh_ex_vat"]
    )
    if output["difference_gbp_per_3100kwh_ex_vat"].abs().max() > 1e-6:
        raise ValueError("Ofgem forward-procurement reconstruction does not reconcile to the component register")
    output["same_year_backtest_allowed"] = True
    output["recovery_status"] = "reproduced_from_observed_ofgem_forward_inputs"
    output["model_basis"] = (
        "Observed Ofgem forward-contract index, published base/peak demand allocation, "
        "and named shaping, transaction, risk and loss allowances annualised by delivery days"
    )

    diagnostic = wholesale_impact.copy()
    diagnostic["modelled_demand_mwh"] = pd.to_numeric(diagnostic["modelled_demand_mwh"], errors="raise")
    diagnostic["modelled_demand_weighted_clearing_price_gbp_per_mwh"] = pd.to_numeric(
        diagnostic["modelled_demand_weighted_clearing_price_gbp_per_mwh"], errors="raise"
    )
    diagnostic["published_forward_index_gbp_per_mwh"] = pd.to_numeric(
        diagnostic["published_forward_index_gbp_per_mwh"], errors="raise"
    )
    if diagnostic["modelled_demand_mwh"].sum() <= 0:
        raise ValueError("Wholesale diagnostic demand must be positive")
    weights = diagnostic["modelled_demand_mwh"] / diagnostic["modelled_demand_mwh"].sum()
    modelled_price = float((diagnostic["modelled_demand_weighted_clearing_price_gbp_per_mwh"] * weights).sum())
    observed_forward = float((diagnostic["published_forward_index_gbp_per_mwh"] * weights).sum())
    summary = {
        "purpose": "Ofgem-style annual forward-procurement reconstruction for the consumer bill",
        "method": "Observed forward inputs and Ofgem named allowances, annualised by delivery days",
        "candidate_component_total_gbp_per_3100kwh_ex_vat": float(output["candidate_allowance_gbp_per_3100kwh_ex_vat"].sum()),
        "published_component_total_gbp_per_3100kwh_ex_vat": float(output["published_allowance_gbp_per_3100kwh_ex_vat"].sum()),
        "pypsa_diagnostic": {
            "demand_weighted_clearing_price_gbp_per_mwh": modelled_price,
            "observed_forward_index_gbp_per_mwh": observed_forward,
            "difference_gbp_per_mwh": modelled_price - observed_forward,
            "treatment": "diagnostic only; not passed through directly to the household bill",
        },
        "future_treatment": (
            "Disabled until an out-of-sample multi-year validation supplies date-stamped base/peak "
            "forward curves and a validated PyPSA counterfactual pass-through rule"
        ),
    }
    return output.sort_values("bill_component").reset_index(drop=True), summary


def _render_html(rows: pd.DataFrame, summary: dict) -> str:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.bill_component))}</td>"
        f"<td>£{row.published_allowance_gbp_per_3100kwh_ex_vat:,.2f}</td>"
        f"<td>£{row.candidate_allowance_gbp_per_3100kwh_ex_vat:,.2f}</td>"
        f"<td>£{row.difference_gbp_per_3100kwh_ex_vat:,.4f}</td></tr>"
        for row in rows.itertuples(index=False)
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Annual forward procurement</title>
<style>body{{font-family:Arial;max-width:1000px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dde3ea;padding:9px;text-align:left}}th{{background:#f3f6f9}}.callout{{background:#edf6ff;border-left:4px solid #3676b5;padding:12px}}</style></head><body>
<h1>Annual forward-procurement reconstruction</h1><p class=\"callout\">Observed forward costs are the bill input. PyPSA price is retained only as a diagnostic.</p>
<p>PyPSA delivery price: <strong>£{summary['pypsa_diagnostic']['demand_weighted_clearing_price_gbp_per_mwh']:,.2f}/MWh</strong>; observed forward index: <strong>£{summary['pypsa_diagnostic']['observed_forward_index_gbp_per_mwh']:,.2f}/MWh</strong>.</p>
<table><thead><tr><th>Component</th><th>Published</th><th>Reconstructed</th><th>Difference</th></tr></thead><tbody>{body}</tbody></table></body></html>"""


def write_bridge(register_path, periods_path, wholesale_path, output_csv, output_summary, output_html):
    rows, summary = build_ofgem_forward_procurement_bridge(
        pd.read_csv(register_path), pd.read_csv(periods_path), pd.read_csv(wholesale_path)
    )
    for path in (output_csv, output_summary, output_html):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(output_html).write_text(_render_html(rows, summary), encoding="utf-8")


if "snakemake" in globals():
    write_bridge(
        snakemake.input.register_csv, snakemake.input.procurement_periods_csv,
        snakemake.input.wholesale_impact_csv, snakemake.output.bridge_csv,
        snakemake.output.summary_json, snakemake.output.report_html,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True)
    parser.add_argument("--procurement-periods", required=True)
    parser.add_argument("--wholesale-impact", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()
    write_bridge(args.register, args.procurement_periods, args.wholesale_impact, args.output_csv, args.output_summary, args.output_html)
