"""Project the published detailed register onto one selected consumer bill.

This is an allocation layer, not a new Ofgem cost calculation.  It uses the
published nil-consumption charge to preserve the selected tariff's standing
charge and the benchmark charge to preserve its unit-rate total exactly.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


VALUE = "annual_gbp_per_3100kwh_ex_vat"
FIXED_COMPONENTS = {
    "Ofgem Annex 3 â€” DUoS fixed charge",
    "Ofgem Annex 4 â€” Warm Home Discount (WHD)",
    "Payment-method additional administration",
}
MIXED_GROUPS = {"supplier_operating", "payment", "margin", "headroom"}


def _single_bill(bills: pd.DataFrame) -> pd.Series:
    if len(bills) != 1:
        raise ValueError("Consumer-bill component allocation requires one selected tariff row")
    required = {
        "annual_consumption_kwh", "benchmark_kwh", "nil_gbp_per_year_ex_vat",
        "benchmark_gbp_per_year_ex_vat", "consumer_bill_gbp_ex_vat",
        "consumer_bill_gbp_inc_vat", "wholesale_price_delta_gbp_ex_vat", "vat_rate",
        "payment_method", "region",
    }
    missing = required.difference(bills.columns)
    if missing:
        raise ValueError(f"Consumer bill is missing: {sorted(missing)}")
    return bills.iloc[0]


def build_consumer_bill_register(register: pd.DataFrame, bills: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bill = _single_bill(bills)
    required = {"stack_group", "component", VALUE}
    missing = required.difference(register.columns)
    if missing:
        raise ValueError(f"Detailed component register is missing: {sorted(missing)}")
    source = register.loc[~register["stack_group"].eq("vat")].copy()
    source[VALUE] = pd.to_numeric(source[VALUE], errors="raise")
    source_total = float(source[VALUE].sum())
    benchmark = float(bill["benchmark_gbp_per_year_ex_vat"])
    nil = float(bill["nil_gbp_per_year_ex_vat"])
    amounts = source[VALUE]
    consumption = float(bill["annual_consumption_kwh"])
    benchmark_kwh = float(bill["benchmark_kwh"])
    if (not np.isfinite(amounts).all() or not np.isfinite([benchmark, nil, consumption, benchmark_kwh]).all()
            or source_total <= 0 or nil < 0 or benchmark < nil or consumption < 0 or benchmark_kwh <= 0):
        raise ValueError("Invalid source register or selected tariff values")
    # The primary-source stack is a GB-average benchmark.  Normalising it to
    # the selected tariff benchmark prevents a small scope/rounding mismatch
    # from leaking into the selected consumer-bill total.
    source["scope_alignment_factor"] = benchmark / source_total
    source["aligned_benchmark_gbp_ex_vat"] = source[VALUE] * source["scope_alignment_factor"]
    source["charge_basis"] = "variable"
    source.loc[source["component"].isin(FIXED_COMPONENTS), "charge_basis"] = "fixed"
    # Match the scheme, independently of legacy dash encodings and singular/plural labels.
    fixed = source["component"].str.contains(
        r"DUoS fixed charges?$|Warm Home Discount|\bWHD$", case=False, regex=True, na=False
    )
    source.loc[fixed, "charge_basis"] = "fixed"
    source.loc[source["stack_group"].isin(MIXED_GROUPS), "charge_basis"] = "mixed_allocated"
    aggregate_network = source["stack_group"].eq("network") & ~source["component"].str.contains(
        "TNUoS|DUoS|BSUoS", case=False, regex=True, na=False
    )
    source.loc[aggregate_network, "charge_basis"] = "mixed_allocated"

    fixed_known = float(source.loc[source["charge_basis"].eq("fixed"), "aligned_benchmark_gbp_ex_vat"].sum())
    variable_known = float(source.loc[source["charge_basis"].eq("variable"), "aligned_benchmark_gbp_ex_vat"].sum())
    mixed = source["charge_basis"].eq("mixed_allocated")
    mixed_total = float(source.loc[mixed, "aligned_benchmark_gbp_ex_vat"].sum())
    residual_fixed = nil - fixed_known
    residual_variable = benchmark - nil - variable_known
    if mixed_total <= 0 or residual_fixed < -1e-8 or residual_variable < -1e-8:
        raise ValueError("Tariff standing/unit totals cannot be allocated to this component register")
    source["standing_charge_component_gbp_ex_vat"] = 0.0
    source["unit_charge_at_benchmark_gbp_ex_vat"] = source["aligned_benchmark_gbp_ex_vat"]
    source.loc[source["charge_basis"].eq("fixed"), "standing_charge_component_gbp_ex_vat"] = source.loc[source["charge_basis"].eq("fixed"), "aligned_benchmark_gbp_ex_vat"]
    source.loc[source["charge_basis"].eq("fixed"), "unit_charge_at_benchmark_gbp_ex_vat"] = 0.0
    shares = source.loc[mixed, "aligned_benchmark_gbp_ex_vat"] / mixed_total
    source.loc[mixed, "standing_charge_component_gbp_ex_vat"] = shares * residual_fixed
    source.loc[mixed, "unit_charge_at_benchmark_gbp_ex_vat"] = shares * residual_variable
    source["allocation_method"] = source["charge_basis"].map({
        "fixed": "Published fixed-charge component",
        "variable": "Published volume-related component",
        "mixed_allocated": "Allocated between standing and unit rates in proportion to the published component, to reproduce the selected tariff",
    })

    consumption_ratio = float(bill["annual_consumption_kwh"]) / float(bill["benchmark_kwh"])
    source["annual_gbp_ex_vat"] = source["standing_charge_component_gbp_ex_vat"] + source["unit_charge_at_benchmark_gbp_ex_vat"] * consumption_ratio
    source["is_modelled"] = False
    delta = float(bill["wholesale_price_delta_gbp_ex_vat"])
    if delta:
        source = pd.concat([source, pd.DataFrame([{
            "stack_group": "hedged_wholesale", "component": "Modelled wholesale-price effect",
            "annual_gbp_ex_vat": delta, "standing_charge_component_gbp_ex_vat": 0.0,
            "unit_charge_at_benchmark_gbp_ex_vat": delta / consumption_ratio,
            "charge_basis": "variable", "allocation_method": "PyPSA counterfactual wholesale sensitivity",
            "is_modelled": True, "detail_status": "modelled_sensitivity",
        }])], ignore_index=True, sort=False)
    subtotal = float(source["annual_gbp_ex_vat"].sum())
    expected = float(bill["consumer_bill_gbp_ex_vat"])
    if abs(subtotal - expected) > 1e-6:
        raise ValueError(f"Consumer-bill components do not reconcile: {subtotal} != {expected}")
    vat = float(bill["consumer_bill_gbp_inc_vat"]) - subtotal
    source = pd.concat([source, pd.DataFrame([{
        "stack_group": "vat", "component": f"VAT at {float(bill['vat_rate']) * 100:g}%",
        "annual_gbp_ex_vat": vat, "standing_charge_component_gbp_ex_vat": 0.0,
        "unit_charge_at_benchmark_gbp_ex_vat": 0.0, "charge_basis": "tax",
        "allocation_method": "Applied to the selected consumer bill, ex VAT", "is_modelled": False,
        "detail_status": "published_tax",
    }])], ignore_index=True, sort=False)
    for field in ("region", "payment_method", "annual_consumption_kwh", "benchmark_kwh"):
        source[field] = bill[field]
    for field in ("modelled_year", "input_mode", "bill_basis", "currency", "cap_snapshot"):
        if field in bill:
            source[field] = bill[field]
    summary = {
        "scope": {"region": str(bill["region"]), "payment_method": str(bill["payment_method"]), "annual_consumption_kwh": float(bill["annual_consumption_kwh"])},
        "published_tariff_baseline_ex_vat": benchmark if float(bill["annual_consumption_kwh"]) == float(bill["benchmark_kwh"]) else nil + (benchmark - nil) * consumption_ratio,
        "standing_charge_ex_vat": nil,
        "unit_charge_ex_vat_at_selected_consumption": (benchmark - nil) * consumption_ratio,
        "scope_alignment_factor": float(benchmark / source_total),
        "consumer_bill_ex_vat": expected,
        "consumer_bill_inc_vat": float(bill["consumer_bill_gbp_inc_vat"]),
        "note": "Mixed supplier/payment/margin and undisaggregated network components use estimated standing/unit splits, constrained to the selected tariff totals.",
    }
    return source.sort_values(["stack_group", "component"]).reset_index(drop=True), summary


# Backward-compatible internal alias; workflow outputs and user-facing wording
# use ``consumer_bill_component_register``.
build_household_register = build_consumer_bill_register


def write_register(register_path, bills_path, output_csv, output_summary) -> None:
    register, summary = build_consumer_bill_register(pd.read_csv(register_path), pd.read_csv(bills_path))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    register.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _main(snakemake):
    write_register(snakemake.input.component_register_csv, snakemake.input.bills_csv, snakemake.output.register_csv, snakemake.output.summary_json)


if "snakemake" in globals():
    _main(snakemake)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-register", required=True); parser.add_argument("--bills", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(); write_register(args.component_register, args.bills, args.output_csv, args.output_summary)
