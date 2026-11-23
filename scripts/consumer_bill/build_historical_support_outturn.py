"""Create a year-agnostic CfD/RO evidence report for a historic bill run.

The report deliberately separates published bill allowances, scheme outturn and
PyPSA support proxies.  It never converts a proxy to a customer charge: that
requires the separate contract/station reconciliation and recovery bridge.
"""


import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from config.config_loader import load_defaults


def _period_bounds(year: int, start: str | None, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = pd.Timestamp(start or f"{year}-01-01")
    last = pd.Timestamp(end or f"{year}-12-31")
    if len(str(end or "").strip()) <= 10:
        last += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return first, last


def lccc_outturn(actuals_path: str | Path, year: int, start: str | None = None,
                 end: str | None = None) -> dict:
    """Aggregate the locally cached LCCC daily contract outturn without loading all years."""
    first, last = _period_bounds(year, start, end)
    frames = []
    usecols = [
        "Settlement_Date", "CFD_Generation_MWh", "CFD_Payments_GBP",
        "Strike_Price_GBP_Per_MWh", "Market_Reference_Price_GBP_Per_MWh",
        "Weighted_IMRP_GBP_Per_MWh", "Reference_Type",
    ]
    for chunk in pd.read_csv(actuals_path, usecols=usecols, chunksize=200_000):
        dates = pd.to_datetime(chunk["Settlement_Date"], errors="coerce")
        frames.append(chunk.loc[dates.between(first, last)].assign(_date=dates.loc[dates.between(first, last)]))
    actual = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)
    if actual.empty:
        return {"available": False, "reason": f"No LCCC records for {first.date()} to {last.date()}"}
    numeric = [
        "CFD_Generation_MWh", "CFD_Payments_GBP", "Strike_Price_GBP_Per_MWh",
        "Market_Reference_Price_GBP_Per_MWh", "Weighted_IMRP_GBP_Per_MWh",
    ]
    actual[numeric] = actual[numeric].apply(pd.to_numeric, errors="coerce")
    reference = actual["Market_Reference_Price_GBP_Per_MWh"].copy()
    imrp = actual["Reference_Type"].eq("IMRP") & actual["Weighted_IMRP_GBP_Per_MWh"].notna()
    reference.loc[imrp] = actual.loc[imrp, "Weighted_IMRP_GBP_Per_MWh"]
    formula = (actual["Strike_Price_GBP_Per_MWh"] - reference) * actual["CFD_Generation_MWh"]
    generation = float(actual["CFD_Generation_MWh"].sum())
    payment = float(actual["CFD_Payments_GBP"].sum())
    formula_total = float(formula.sum())
    return {
        "available": True,
        "records": int(len(actual)),
        "generation_mwh": generation,
        "payment_gbp": payment,
        "payment_gbp_per_mwh": payment / generation if generation else None,
        "published_terms_formula_gbp": formula_total,
        "published_terms_formula_difference_gbp": formula_total - payment,
        "first_settlement_date": str(actual["_date"].min().date()),
        "last_settlement_date": str(actual["_date"].max().date()),
    }


def _allowance(frame: pd.DataFrame, matcher) -> float | None:
    if frame.empty or "component" not in frame.columns:
        return None
    column = "annualised_gbp_per_3100kwh_ex_vat"
    if column not in frame.columns:
        column = "annual_gbp_per_3100kwh_ex_vat"
    matched = frame.loc[frame["component"].astype(str).map(matcher), column]
    return float(pd.to_numeric(matched, errors="coerce").sum()) if not matched.empty else None


def _roc_reference(path: str | Path | None) -> float | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    for column in ("ofgem_roc_certificates", "rocs_issued"):
        if column in frame.columns:
            return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())
    raise ValueError(f"ROC reference {path} needs 'ofgem_roc_certificates' or 'rocs_issued'")


def build_audit(market_summary: dict, bill_register: pd.DataFrame,
                lccc_actuals_path: str | Path, year: int, roc_buyout_price: float,
                period_start: str | None = None, period_end: str | None = None,
                roc_reference_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Build a transparent support-evidence table for one historic model period."""
    cfd = lccc_outturn(lccc_actuals_path, year, period_start, period_end)
    revenue = market_summary.get("revenue", {})
    cfd_model_cost = revenue.get("cfd_total_cost")
    cfd_model_dispatch = revenue.get("cfd_total_dispatch_MWh")
    roc_model_cost = revenue.get("roc_total_income")
    roc_model_dispatch = revenue.get("roc_total_dispatch_MWh")
    roc_model_certificates = (
        float(roc_model_cost) / roc_buyout_price
        if roc_model_cost is not None and roc_buyout_price > 0 else None
    )
    cfd_bill = _allowance(bill_register, lambda text: "cfd" in text.casefold())
    roc_bill = _allowance(bill_register, lambda text: text.strip().casefold() in {"ro", "renewables obligation"})
    roc_official = _roc_reference(roc_reference_path)

    rows = [
        {"scheme": "CfD", "measure": "published_bill_allowance", "value": cfd_bill,
         "unit": "GBP per 3,100 kWh ex VAT", "treatment": "published historical allowance", "status": "not_modelled"},
        {"scheme": "CfD", "measure": "lccc_actual_generation", "value": cfd.get("generation_mwh"),
         "unit": "MWh", "treatment": "official scheme outturn comparator", "status": "available" if cfd["available"] else "source_unavailable"},
        {"scheme": "CfD", "measure": "lccc_actual_payment", "value": cfd.get("payment_gbp"),
         "unit": "GBP", "treatment": "official scheme outturn comparator", "status": "available" if cfd["available"] else "source_unavailable"},
        {"scheme": "CfD", "measure": "pypsa_support_proxy_cost", "value": cfd_model_cost,
         "unit": "GBP", "treatment": "generic support proxy; not a levy", "status": "requires_contract_reconciliation"},
        {"scheme": "CfD", "measure": "pypsa_support_proxy_dispatch", "value": cfd_model_dispatch,
         "unit": "MWh", "treatment": "generators labelled CfD", "status": "requires_contract_reconciliation"},
        {"scheme": "RO", "measure": "published_bill_allowance", "value": roc_bill,
         "unit": "GBP per 3,100 kWh ex VAT", "treatment": "published historical allowance", "status": "not_modelled"},
        {"scheme": "RO", "measure": "pypsa_support_proxy_income", "value": roc_model_cost,
         "unit": "GBP", "treatment": "ROC-labelled dispatch x configured buy-out price; not supplier recovery", "status": "requires_station_reconciliation"},
        {"scheme": "RO", "measure": "pypsa_support_proxy_certificates", "value": roc_model_certificates,
         "unit": "ROC", "treatment": "proxy implied by configured buy-out price", "status": "requires_station_reconciliation"},
        {"scheme": "RO", "measure": "pypsa_support_proxy_dispatch", "value": roc_model_dispatch,
         "unit": "MWh", "treatment": "generators labelled ROC", "status": "requires_station_reconciliation"},
        {"scheme": "RO", "measure": "ofgem_issued_certificates", "value": roc_official,
         "unit": "ROC", "treatment": "official historic issue comparator", "status": "available" if roc_official is not None else "historic_station_snapshot_required"},
    ]
    table = pd.DataFrame(rows)
    cfd_ratio = (
        float(cfd_model_cost) / cfd["payment_gbp"]
        if cfd.get("available") and cfd.get("payment_gbp") and cfd_model_cost is not None else None
    )
    roc_ratio = roc_model_certificates / roc_official if roc_model_certificates is not None and roc_official else None
    summary = {
        "calendar_year": int(year),
        "period_start": str(_period_bounds(year, period_start, period_end)[0]),
        "period_end": str(_period_bounds(year, period_start, period_end)[1]),
        "purpose": "Historic CfD/RO evidence screen before detailed contract/station reconciliation and consumer recovery.",
        "cfd": {
            "lccc_outturn": cfd,
            "pypsa_proxy_cost_gbp": cfd_model_cost,
            "pypsa_proxy_dispatch_mwh": cfd_model_dispatch,
            "proxy_cost_to_lccc_outturn_ratio": cfd_ratio,
            "next_gate": "Map contracts and apply daily indexed strike/reference terms to final PyPSA dispatch.",
        },
        "roc": {
            "configured_buyout_price_gbp_per_roc": roc_buyout_price,
            "pypsa_proxy_income_gbp": roc_model_cost,
            "pypsa_proxy_certificates": roc_model_certificates,
            "official_issued_certificates": roc_official,
            "proxy_certificate_coverage": roc_ratio,
            "next_gate": "Use a fixed historic Ofgem station/certificate snapshot, then reconcile station labels and supplier-obligation recovery.",
        },
        "future_use_rule": "No row in this report is a consumer-bill pass-through. Use it only to decide whether detailed support reconciliation is ready.",
    }
    return table, summary


def write_audit(**kwargs) -> None:
    build_kwargs = {
        key: value for key, value in kwargs.items()
        if key not in {"output_csv", "output_summary"}
    }
    table, summary = build_audit(**build_kwargs)
    for path in (kwargs["output_csv"], kwargs["output_summary"]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(kwargs["output_csv"], index=False)
    Path(kwargs["output_summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    # snakemake.input.defaults (every defaults file) is a change trigger only; the defaults are read here.
    defaults = load_defaults()
    prices = defaults["support"]["dispatch_costs"]["roc_buyout_prices"]
    value = prices.get(int(snakemake.params.year), prices.get(str(snakemake.params.year), 0.0))
    reference = getattr(snakemake.input, "roc_reference_csv", None)
    write_audit(
        market_summary=json.loads(Path(snakemake.input.market_summary_json).read_text(encoding="utf-8")),
        bill_register=pd.read_csv(snakemake.input.bill_register_csv),
        lccc_actuals_path=snakemake.input.lccc_actuals_csv,
        year=int(snakemake.params.year), roc_buyout_price=float(value),
        period_start=snakemake.params.period_start, period_end=snakemake.params.period_end,
        roc_reference_path=str(reference) if reference else None,
        output_csv=snakemake.output.audit_csv, output_summary=snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-summary", required=True); parser.add_argument("--bill-register", required=True)
    parser.add_argument("--lccc-actuals", required=True)
    parser.add_argument("--year", required=True, type=int); parser.add_argument("--roc-buyout-price", required=True, type=float)
    parser.add_argument("--period-start"); parser.add_argument("--period-end"); parser.add_argument("--roc-reference")
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_audit(
        market_summary=json.loads(Path(args.market_summary).read_text(encoding="utf-8")),
        bill_register=pd.read_csv(args.bill_register),
        lccc_actuals_path=args.lccc_actuals, year=args.year, roc_buyout_price=args.roc_buyout_price,
        period_start=args.period_start, period_end=args.period_end, roc_reference_path=args.roc_reference,
        output_csv=args.output_csv, output_summary=args.output_summary,
    )
