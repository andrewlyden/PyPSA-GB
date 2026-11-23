"""Classify a detailed bill register for transparent future-scenario forecasting."""

import argparse
import json
from pathlib import Path

import pandas as pd


def _classification(component: str, stack_group: str) -> dict[str, str]:
    """Return the future-cost pathway; none of these labels change the amount."""
    text = component.lower()
    if "cfd" in text:
        return {"forecast_driver": "operational_model", "forecast_method": "modelled_recovery_bridge", "recovery_denominator": "supplier_sales_mwh", "required_input": "strike/reference prices, LCCC levy and reconciliation"}
    if "renewables obligation" in text or "ro " in text or text.endswith(" ro"):
        return {"forecast_driver": "operational_model", "forecast_method": "modelled_recovery_bridge", "recovery_denominator": "supplier_sales_mwh", "required_input": "eligible generation, certificate price and supplier recovery"}
    if "feed-in tariff" in text or text.endswith(" fit"):
        return {"forecast_driver": "operational_model", "forecast_method": "fit_levelisation_bridge", "recovery_denominator": "supplier_sales_mwh", "required_input": "FiT-labelled generation, tariff/levelisation outturn and supplier recovery"}
    if "bsuos" in text or "imbalance" in text:
        return {"forecast_driver": "operational_model", "forecast_method": "modelled_recovery_bridge", "recovery_denominator": "supplier_sales_mwh", "required_input": "balancing settlement, ancillary services and supplier recovery"}
    if "capacity market" in text:
        return {"forecast_driver": "system_transition", "forecast_method": "regulatory_recovery_bridge", "recovery_denominator": "supplier_sales_mwh", "required_input": "capacity auction/recovery forecast"}
    if "tnuos" in text or "duos" in text or stack_group == "network":
        return {"forecast_driver": "system_transition", "forecast_method": "regulatory_recovery_bridge", "recovery_denominator": "charging_basis", "required_input": "network investment, allowed revenue and charging allocation"}
    if "loss" in text:
        return {"forecast_driver": "operational_model", "forecast_method": "loss_factor_bridge", "recovery_denominator": "supplier_sales_mwh", "required_input": "forecast loss factors and billable sales"}
    if stack_group in {"hedged_wholesale", "hedge_uplifts"}:
        return {"forecast_driver": "operational_model", "forecast_method": "forward_procurement_bridge", "recovery_denominator": "supplier_sales_mwh", "required_input": "forward-price/hedging method and billable sales"}
    if stack_group == "policy":
        return {"forecast_driver": "policy", "forecast_method": "policy_assumption", "recovery_denominator": "defined_by_policy", "required_input": "published policy budget, eligibility and recovery rule"}
    if "smart" in text:
        return {"forecast_driver": "system_transition", "forecast_method": "industry_charge_assumption", "recovery_denominator": "meter_or_customer", "required_input": "DCC/SEGB/SMICoP and smart-meter rollout forecast"}
    if stack_group in {"supplier_operating", "payment", "bad_debt", "margin", "headroom"}:
        return {"forecast_driver": "retail", "forecast_method": "retail_assumption", "recovery_denominator": "customer_or_sales_mwh", "required_input": "supplier operating, debt, payment and margin assumptions"}
    if stack_group == "vat":
        return {"forecast_driver": "tax", "forecast_method": "fixed_rule", "recovery_denominator": "taxable_bill", "required_input": "VAT policy"}
    return {"forecast_driver": "regulatory", "forecast_method": "explicit_assumption", "recovery_denominator": "defined_by_methodology", "required_input": "future Ofgem/NESO methodology"}


def build_registry(register: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"component", "stack_group", "annual_gbp_per_3100kwh_ex_vat"}
    missing = required.difference(register.columns)
    if missing:
        raise ValueError(f"Component register is missing: {sorted(missing)}")
    result = register.copy()
    result = result.loc[~result["stack_group"].eq("vat")].copy()
    result = result.join(pd.DataFrame([_classification(str(r.component), str(r.stack_group)) for r in result.itertuples()], index=result.index))
    result["forecast_status"] = "input_required"
    summary = {
        "purpose": "Future consumer-bill forecast classification; it does not itself forecast costs.",
        "component_count": int(len(result)),
        "by_driver": result.groupby("forecast_driver")["component"].count().to_dict(),
        "recovery_demand_rule": "Use PyPSA system demand only for operational drivers. Convert it to billable supplier sales and apply the component's recovery denominator before producing a domestic bill.",
        "double_counting_rule": "A model output may enter either the wholesale price pathway or a separately evidenced levy/recovery bridge; never both without an explicit accounting reconciliation.",
    }
    return result, summary


def write_registry(input_csv, output_csv, output_summary):
    registry, summary = build_registry(pd.read_csv(input_csv))
    for output in (output_csv, output_summary): Path(output).parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_registry(snakemake.input.register_csv, snakemake.output.registry_csv, snakemake.output.summary_json)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True); parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(); write_registry(args.input, args.output_csv, args.output_summary)
