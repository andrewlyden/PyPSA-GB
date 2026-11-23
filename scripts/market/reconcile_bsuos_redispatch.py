"""Compare the published BSUoS bill allowance with modelled redispatch activity.

This is a reconciliation diagnostic. It deliberately does not turn PyPSA BM
cost into a consumer levy: gross redispatch volume is a calibrated activity
index, while BSUoS recovery and balancing-service outturn need separate
settlement evidence.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def reconcile(network_components: pd.DataFrame, wholesale_impact: pd.DataFrame, market_summary: dict) -> tuple[pd.DataFrame, dict]:
    bsuos = network_components.loc[
        network_components["component"].astype(str).str.contains("BSUoS", case=False, na=False),
        "annualised_gbp_per_3100kwh_ex_vat",
    ]
    demand_mwh = float(pd.to_numeric(wholesale_impact["modelled_demand_mwh"], errors="raise").sum())
    redispatch = market_summary["redispatch"]
    net_cost = float(redispatch["total_net_cost"])
    increase_mwh = float(redispatch.get("total_increase_MWh", 0.0))
    decrease_mwh = float(redispatch.get("total_decrease_MWh", 0.0))
    gross_mwh = increase_mwh + decrease_mwh
    if bsuos.empty:
        raise ValueError("Expected annualised Ofgem BSUoS component rows")
    published_per_customer = float(pd.to_numeric(bsuos, errors="raise").sum())
    rows = pd.DataFrame([
        {"measure": "Published Ofgem BSUoS allowance", "value": published_per_customer, "unit": "GBP per 3,100 kWh customer, ex VAT", "treatment": "Included in bill baseline"},
        {"measure": "Published BSUoS implied rate", "value": published_per_customer / 3.1, "unit": "GBP per MWh customer benchmark", "treatment": "Reference only; customer energy differs from settlement volume"},
        {"measure": "Modelled redispatch net cost", "value": net_cost, "unit": "GBP in model year", "treatment": "Model diagnostic, not bill pass-through"},
        {"measure": "Modelled redispatch net cost rate", "value": net_cost / demand_mwh, "unit": "GBP per MWh model demand", "treatment": "Model diagnostic, not bill pass-through"},
        {"measure": "Modelled gross redispatch volume", "value": gross_mwh, "unit": "MWh", "treatment": "Calibrated balancing-activity index, not settlement expenditure"},
        {"measure": "Modelled gross redispatch intensity", "value": gross_mwh / demand_mwh, "unit": "MWh per MWh model demand", "treatment": "Central BSUoS/imbalance index after historical calibration"},
    ])
    summary = {
        "published_bsuos_gbp_per_3100kwh_ex_vat": published_per_customer,
        "modelled_redispatch_net_cost_gbp": net_cost,
        "modelled_demand_mwh": demand_mwh,
        "modelled_redispatch_gbp_per_mwh": net_cost / demand_mwh,
        "modelled_gross_redispatch_mwh": gross_mwh,
        "modelled_gross_redispatch_mwh_per_mwh_demand": gross_mwh / demand_mwh,
        "consumer_bill_pass_through": False,
        "next_evidence_required": [
            "NESO/Elexon BSUoS settlement outturn for the same period and volume basis",
            "A reconciliation of redispatch, balancing-energy and ancillary-service cost categories to that outturn",
            "The applicable supplier recovery and customer tariff allocation basis",
        ],
    }
    return rows, summary


def write_reconciliation(network_path, wholesale_path, market_summary_path, output_csv, output_summary):
    summary_in = json.loads(Path(market_summary_path).read_text(encoding="utf-8"))
    rows, summary = reconcile(pd.read_csv(network_path), pd.read_csv(wholesale_path), summary_in)
    for path in (output_csv, output_summary): Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _main(snakemake):
    write_reconciliation(snakemake.input.network_components_csv, snakemake.input.wholesale_impact_csv, snakemake.input.market_summary_json, snakemake.output.reconciliation_csv, snakemake.output.summary_json)


if "snakemake" in globals(): _main(snakemake)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-components", required=True); parser.add_argument("--wholesale-impact", required=True); parser.add_argument("--market-summary", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(); write_reconciliation(args.network_components, args.wholesale_impact, args.market_summary, args.output_csv, args.output_summary)
