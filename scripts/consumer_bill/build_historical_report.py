"""Build the validation-first historical consumer-bill reporting directory.

This entry point consumes a pinned source bundle directly.  Model evidence is
optional and, when absent, the reconstruction explicitly retains the published
component rather than pretending that a model result exists.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.consumer_bill.compare_historical_components import compare_components
    from scripts.consumer_bill.component_ledger import (
        METHOD_VERSION, add_ledger_ids, file_sha256, validate_component_ledger,
    )
except ModuleNotFoundError:  # direct ``python scripts/consumer_bill/...`` use
    from compare_historical_components import compare_components
    from component_ledger import (  # type: ignore
        METHOD_VERSION, add_ledger_ids, file_sha256, validate_component_ledger,
    )


class HistoricalReportError(ValueError):
    """Raised when a validated historical report cannot be produced."""


def _validate_tariffs(tariffs: pd.DataFrame, *, expected_region_count: int = 14) -> pd.DataFrame:
    required = {
        "cap_period", "delivery_start", "delivery_end", "delivery_days", "payment_method",
        "meter_type", "region", "nil_gbp_per_year_ex_vat",
        "benchmark_gbp_per_year_ex_vat", "benchmark_kwh",
        "standing_charge_gbp_per_day_ex_vat", "unit_rate_gbp_per_kwh_ex_vat", "vat_rate",
    }
    if missing := required.difference(tariffs.columns):
        raise HistoricalReportError(f"Period tariffs are missing: {sorted(missing)}")
    result = tariffs.copy()
    result["delivery_start"] = pd.to_datetime(result["delivery_start"], errors="raise")
    result["delivery_end"] = pd.to_datetime(result["delivery_end"], errors="raise")
    calculated_days = (result["delivery_end"] - result["delivery_start"]).dt.days + 1
    if not calculated_days.eq(pd.to_numeric(result["delivery_days"], errors="raise")).all():
        raise HistoricalReportError("Tariff delivery_days must equal the inclusive delivery dates")
    keys = ["cap_period", "region", "payment_method", "meter_type"]
    if result.duplicated(keys).any():
        raise HistoricalReportError("Period tariff identities must be unique")
    regions = result.groupby("cap_period")["region"].agg(set)
    if len({frozenset(value) for value in regions}) != 1:
        raise HistoricalReportError("Every tariff period must contain the same regions")
    if len(next(iter(regions))) != expected_region_count:
        raise HistoricalReportError(
            f"Annual historical reports require all {expected_region_count} charge-restriction regions"
        )
    expected = result["nil_gbp_per_year_ex_vat"] + result["unit_rate_gbp_per_kwh_ex_vat"] * result["benchmark_kwh"]
    if not np.allclose(expected, result["benchmark_gbp_per_year_ex_vat"], atol=0.005, rtol=0):
        raise HistoricalReportError("Period tariff rates do not reproduce the source benchmark to one penny")
    intervals = result[["cap_period", "delivery_start", "delivery_end"]].drop_duplicates().sort_values("delivery_start")
    cursor = intervals.iloc[0]["delivery_start"]
    for row in intervals.itertuples(index=False):
        if row.delivery_start != cursor:
            raise HistoricalReportError(f"Tariff periods have a gap or overlap at {cursor.date()}")
        cursor = row.delivery_end + pd.Timedelta(days=1)
    year = int(intervals.iloc[0].delivery_start.year)
    if (
        intervals.iloc[0].delivery_start != pd.Timestamp(year=year, month=1, day=1)
        or intervals.iloc[-1].delivery_end != pd.Timestamp(year=year, month=12, day=31)
        or not intervals["delivery_start"].dt.year.eq(year).all()
        or not intervals["delivery_end"].dt.year.eq(year).all()
    ):
        raise HistoricalReportError("Annual historical reports must cover exactly one full calendar year")
    return result


def _period_consumption(
    tariffs: pd.DataFrame, annual_consumption_kwh: float, profile: pd.DataFrame | None,
) -> tuple[pd.Series, str]:
    periods = tariffs[["cap_period", "delivery_start", "delivery_end", "delivery_days"]].drop_duplicates()
    if profile is None:
        total_days = float(periods["delivery_days"].sum())
        return periods.set_index("cap_period")["delivery_days"] * annual_consumption_kwh / total_days, "calendar-day allocation proxy"
    if not {"timestamp", "consumption_kwh"}.issubset(profile.columns):
        raise HistoricalReportError("Consumption profile needs timestamp and consumption_kwh columns")
    timestamps = pd.to_datetime(profile["timestamp"], errors="raise", utc=True)
    amounts = pd.to_numeric(profile["consumption_kwh"], errors="raise")
    if timestamps.duplicated().any() or amounts.isna().any() or (amounts < 0).any():
        raise HistoricalReportError("Consumption profile must have unique timestamps and non-negative finite consumption")
    if not np.isclose(float(amounts.sum()), annual_consumption_kwh, atol=1e-6, rtol=0):
        raise HistoricalReportError("Consumption profile does not sum to annual_consumption_kwh")
    local_dates = timestamps.dt.tz_convert("Europe/London").dt.tz_localize(None)
    allocated = {}
    for row in periods.itertuples(index=False):
        mask = local_dates.between(row.delivery_start, row.delivery_end + pd.Timedelta(days=1), inclusive="left")
        allocated[row.cap_period] = float(amounts.loc[mask].sum())
    if not np.isclose(sum(allocated.values()), annual_consumption_kwh, atol=1e-6, rtol=0):
        raise HistoricalReportError("Consumption timestamps do not fall wholly inside the tariff year")
    return pd.Series(allocated), "supplied UTC profile allocated to Europe/London tariff days"


def build_report_tables(
    tariffs: pd.DataFrame,
    source_ledger: pd.DataFrame,
    *,
    annual_consumption_kwh: float = 3100.0,
    consumption_profile: pd.DataFrame | None = None,
    model_signals: pd.DataFrame | None = None,
    cfd_reconciliation: dict[str, Any] | None = None,
    cfd_mapping: dict[str, Any] | None = None,
    procurement_bridge: pd.DataFrame | None = None,
    comparison_register: pd.DataFrame | None = None,
    recovery_candidates: pd.DataFrame | None = None,
    network_bridge: pd.DataFrame | None = None,
    comparison_assumptions: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    tariffs = _validate_tariffs(tariffs)
    start, end = tariffs["delivery_start"].min(), tariffs["delivery_end"].max()
    source_ledger = validate_component_ledger(source_ledger, expected_start=str(start), expected_end=str(end))
    if annual_consumption_kwh < 0 or not np.isfinite(annual_consumption_kwh):
        raise HistoricalReportError("annual_consumption_kwh must be finite and non-negative")
    comparison_use, _ = _period_consumption(tariffs, float(annual_consumption_kwh), None)
    household_use, household_profile_basis = _period_consumption(
        tariffs, float(annual_consumption_kwh), consumption_profile
    )
    use_by_result = {
        "published_tariff_replay": (comparison_use, "calendar-day allocation proxy"),
        "historical_model_reconstruction": (comparison_use, "calendar-day allocation proxy"),
        "household_payment_illustration": (household_use, household_profile_basis),
    }

    period_rows = []
    component_rows = []
    residual_rows = []
    for tariff in tariffs.itertuples(index=False):
        common = {
            "cap_period": tariff.cap_period, "period_start": tariff.delivery_start.date().isoformat(),
            "period_end": tariff.delivery_end.date().isoformat(), "region": tariff.region,
            "payment_method": tariff.payment_method, "meter_type": tariff.meter_type,
        }
        bill_values = {}
        for result_type, (period_use, profile_basis) in use_by_result.items():
            use = float(period_use.loc[tariff.cap_period])
            standing = float(tariff.standing_charge_gbp_per_day_ex_vat * tariff.delivery_days)
            energy = float(tariff.unit_rate_gbp_per_kwh_ex_vat * use)
            subtotal = standing + energy
            vat = subtotal * float(tariff.vat_rate)
            bill_values[result_type] = (use, standing, energy, subtotal, vat, profile_basis)
            period_rows.append({
                **common, "result_type": result_type, "consumption_kwh": use,
                "consumption_profile_basis": profile_basis,
                "standing_charge_gbp_ex_vat": standing, "unit_charge_gbp_ex_vat": energy,
                "bill_gbp_ex_vat": subtotal, "vat_gbp": vat, "bill_gbp_inc_vat": subtotal + vat,
                "modelled_component_count": 0,
            })
        selected = source_ledger.loc[
            source_ledger["cap_period"].eq(tariff.cap_period)
            & source_ledger["region"].eq(tariff.region)
        ].copy()
        if selected.empty:
            raise HistoricalReportError(f"No component ledger rows for {tariff.cap_period}/{tariff.region}")
        component_subtotals = {}
        for result_type, (period_use, profile_basis) in use_by_result.items():
            use, _, _, _, vat, _ = bill_values[result_type]
            result_components = selected.copy()
            result_components["result_type"] = result_type
            result_components["consumption_case"] = f"selected_{annual_consumption_kwh:g}kwh"
            result_components["quantity"] = 1.0
            result_components["rate"] = (
                result_components["fixed_gbp_per_year_ex_vat"] * float(tariff.delivery_days) / 365.0
                + result_components["variable_gbp_at_benchmark_ex_vat"] * use / result_components["benchmark_kwh"]
            )
            result_components["amount_gbp"] = result_components["rate"]
            result_components["sign"] = np.select(
                [result_components["amount_gbp"].gt(0), result_components["amount_gbp"].lt(0)],
                ["charge", "credit"], default="zero",
            )
            result_components["consumption_basis"] = profile_basis
            component_subtotals[result_type] = float(result_components["amount_gbp"].sum())
            result_components = add_ledger_ids(result_components.drop(columns="ledger_id"))
            vat_row = result_components.iloc[[0]].copy()
            vat_row["component_id"] = "tax.vat"
            vat_row["component_label"] = f"VAT at {float(tariff.vat_rate) * 100:g}%"
            vat_row["category"] = "tax"
            vat_row["unit"] = "GBP per household for delivery period"
            vat_row["quantity"] = 1.0
            vat_row["rate"] = vat
            vat_row["amount_gbp"] = vat
            vat_row["sign"] = "charge" if vat > 0 else "credit" if vat < 0 else "zero"
            vat_row["vat_treatment"] = "tax charge"
            vat_row["quantity_source"] = "reconstructed ex-VAT bill subtotal"
            vat_row["price_source"] = "period tariff VAT rate"
            vat_row["recovery_method"] = "VAT applied once after component substitutions"
            vat_row["source_reference"] = f"period_tariffs.csv:{tariff.cap_period}"
            vat_row["formula"] = "bill_gbp_ex_vat * vat_rate"
            vat_row["fixed_gbp_per_year_ex_vat"] = 0.0
            vat_row["variable_gbp_at_benchmark_ex_vat"] = 0.0
            vat_row = add_ledger_ids(vat_row.drop(columns="ledger_id"))
            component_rows.extend([result_components, vat_row])
        _, _, _, published_subtotal, _, _ = bill_values["published_tariff_replay"]
        residual = published_subtotal - component_subtotals["published_tariff_replay"]
        residual_rows.append({
            **common, "residual_id": f"tariff_component_reconciliation.{tariff.cap_period}.{tariff.region}",
            "expected_gbp_ex_vat": published_subtotal,
            "accounted_gbp_ex_vat": component_subtotals["published_tariff_replay"],
            "residual_gbp_ex_vat": residual,
            "reason": "rounding only" if abs(residual) <= 0.01 else "unresolved source allocation",
            "status": "pass" if abs(residual) <= 0.01 else "fail",
        })
    components = pd.concat(component_rows, ignore_index=True)
    components = validate_component_ledger(components, expected_start=str(start), expected_end=str(end))
    period_bills = pd.DataFrame(period_rows)
    regional = period_bills.groupby(
        ["result_type", "region", "payment_method", "meter_type", "consumption_profile_basis"], as_index=False
    ).agg(
        annual_consumption_kwh=("consumption_kwh", "sum"),
        standing_charge_gbp_ex_vat=("standing_charge_gbp_ex_vat", "sum"),
        unit_charge_gbp_ex_vat=("unit_charge_gbp_ex_vat", "sum"),
        bill_gbp_ex_vat=("bill_gbp_ex_vat", "sum"), vat_gbp=("vat_gbp", "sum"),
        bill_gbp_inc_vat=("bill_gbp_inc_vat", "sum"),
        modelled_component_count=("modelled_component_count", "sum"),
    )
    residuals = pd.DataFrame(residual_rows)
    supplied_evidence = any(
        value is not None for value in (model_signals, cfd_reconciliation, cfd_mapping, procurement_bridge)
    )
    checks = {
        "validated": bool(residuals["status"].eq("pass").all()),
        "ledger_schema_version": source_ledger["schema_version"].iloc[0],
        "method_version": METHOD_VERSION,
        "period_count": int(tariffs["cap_period"].nunique()),
        "region_count": int(tariffs["region"].nunique()),
        "tariff_row_count": int(len(tariffs)),
        "component_row_count": int(len(components)),
        "maximum_absolute_reconciliation_residual_gbp": float(residuals["residual_gbp_ex_vat"].abs().max()),
        "model_substitutions": 0,
        "model_validation_status": (
            "evidence supplied; published period/regional accounting and annual Ofgem/modelled sensitivity are reported separately"
            if supplied_evidence else "not supplied; reconstruction retains published components"
        ),
        "consumption_profile_basis": household_profile_basis,
    }
    if not checks["validated"]:
        raise HistoricalReportError("Component accounting does not reconcile to the tariff within one penny")
    if model_signals is None:
        model_validation = pd.DataFrame(columns=["metric", "model_value", "observed_value", "unit", "status"])
    else:
        required = {"metric", "value", "unit"}
        if missing := required.difference(model_signals.columns):
            raise HistoricalReportError(f"Model signals are missing: {sorted(missing)}")
        model_validation = model_signals[["metric", "value", "unit"]].copy()
        model_validation = model_validation.rename(columns={"value": "model_value"})
        model_validation["observed_value"] = pd.NA
        model_validation["status"] = "reported model evidence; no bill substitution"
        model_validation = model_validation[["metric", "model_value", "observed_value", "unit", "status"]]
    support_rows = []
    if cfd_reconciliation is not None:
        actual = float(cfd_reconciliation.get("actual_payment_gbp_mapped_contracts", 0.0))
        modelled = float(cfd_reconciliation.get("modelled_payment_independent_formula_gbp", 0.0))
        support_rows.append({
            "scheme": "CfD", "cohort": "mapped LCCC contracts", "observed_gbp": actual,
            "modelled_gbp": modelled, "coverage": modelled / actual if actual else pd.NA,
            "status": "complete mapped cohort" if cfd_mapping and cfd_mapping.get("model_mapping_complete") else "incomplete mapping; diagnostic only",
        })
    support_reconciliation = pd.DataFrame(
        support_rows, columns=["scheme", "cohort", "observed_gbp", "modelled_gbp", "coverage", "status"]
    )
    recovery_rows = []
    if procurement_bridge is not None:
        required = {
            "bill_component", "candidate_allowance_gbp_per_3100kwh_ex_vat",
            "published_allowance_gbp_per_3100kwh_ex_vat", "recovery_status", "model_basis",
        }
        if missing := required.difference(procurement_bridge.columns):
            raise HistoricalReportError(f"Procurement bridge is missing: {sorted(missing)}")
        for row in procurement_bridge.itertuples(index=False):
            recovery_rows.append({
                "component_id": "wholesale.direct_fuel", "system_cost_gbp": pd.NA,
                "recovery_quantity": pd.NA,
                "household_gbp": row.candidate_allowance_gbp_per_3100kwh_ex_vat,
                "method": row.model_basis,
                "status": f"{row.recovery_status}; not substituted into period/regional ledger",
            })
    recovery_bridge = pd.DataFrame(
        recovery_rows,
        columns=["component_id", "system_cost_gbp", "recovery_quantity", "household_gbp", "method", "status"],
    )
    comparison, comparison_totals, comparison_checks = compare_components(
        components, regional, register=comparison_register, signals=model_signals,
        procurement=procurement_bridge, recovery=recovery_candidates,
        network_bridge=network_bridge, assumptions=comparison_assumptions,
        annual_consumption_kwh=annual_consumption_kwh,
    )
    checks["ofgem_model_comparison"] = comparison_checks
    return {
        "ofgem_model_comparison": comparison,
        "ofgem_model_totals": comparison_totals,
        "bill_components": components,
        "period_bills": period_bills,
        "regional_bills": regional,
        "model_validation": model_validation,
        "support_reconciliation": support_reconciliation,
        "recovery_bridge": recovery_bridge,
        "reconciliation_residuals": residuals,
    }, checks


def _report_html(tables: dict[str, pd.DataFrame], checks: dict[str, Any], year: int) -> str:
    regional = tables["regional_bills"].copy()
    summary = regional[["result_type", "region", "annual_consumption_kwh", "bill_gbp_inc_vat"]]
    sections = []
    for name, frame in tables.items():
        sections.append(
            f"<details{' open' if name in {'regional_bills', 'reconciliation_residuals'} else ''}>"
            f"<summary>{html.escape(name)} ({len(frame)} rows)</summary>{frame.to_html(index=False, border=0)}</details>"
        )
    comparison = tables["ofgem_model_comparison"]
    overview = comparison[["component", "ofgem_gbp_ex_vat", "model_gbp_ex_vat",
                           "calibrated_candidate_gbp_ex_vat", "ofgem_modelled_gbp_ex_vat",
                           "difference_gbp_ex_vat", "hybrid_treatment"]].rename(columns={
        "component": "Component", "ofgem_gbp_ex_vat": "Ofgem (GBP)",
        "model_gbp_ex_vat": "Model candidate (GBP)",
        "calibrated_candidate_gbp_ex_vat": "Calibrated / observed allocation (GBP)",
        "ofgem_modelled_gbp_ex_vat": "Ofgem/modelled (GBP)",
        "difference_gbp_ex_vat": "Difference (GBP)", "hybrid_treatment": "Treatment",
    })
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>{year} historical consumer bill</title>
<style>body{{font:14px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;margin:1rem 0}}th,td{{padding:.35rem .55rem;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}summary{{font-size:1.1rem;font-weight:600;margin-top:1rem;cursor:pointer}}.pass{{color:#176b34}}</style></head>
<body><h1>{year} historical electricity consumer bill</h1><p class=\"pass\">Published tariff accounting validated: {checks['validated']}</p>
<h2>Ofgem versus Ofgem/modelled</h2>
<p>Annual electricity bill at {float(regional.annual_consumption_kwh.iloc[0]):g} kWh, equal mean of 14 regions, nominal pounds in the reporting year. Component amounts exclude VAT; totals recalculate VAT after substitutions.</p>
{tables['ofgem_model_totals'].to_html(index=False, border=0, float_format=lambda x: f'{x:,.2f}')}
<p>{html.escape(checks['ofgem_model_comparison']['interpretation'])}</p>
<p>Model candidates: {checks['ofgem_model_comparison']['model_candidate_components']}; calibrated/observed allocations: {checks['ofgem_model_comparison']['calibrated_components']}; impacted components awaiting an estimate: {checks['ofgem_model_comparison']['impacted_components_without_candidate']}. A complete future bill still requires external regulatory and retail assumptions.</p>
{overview.to_html(index=False, border=0, na_rep='Not available', float_format=lambda x: f'{x:,.2f}')}
<p>Download <a href="ofgem_model_comparison.csv">component comparison and methods</a> or <a href="ofgem_model_totals.csv">bill totals</a>. Expand the comparison below for model signals, denominators and future-validation gaps.</p>
<h2>Published period and regional tariff accounting</h2>
<p>Published tariff replay, an explicit reconstruction retaining unsupported external components, and a dated household-payment illustration share one component ledger. {html.escape(checks['model_validation_status'])}.</p>
{summary.to_html(index=False, border=0)}{''.join(sections)}</body></html>"""


def write_report(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    annual_consumption_kwh: float = 3100.0,
    consumption_profile_path: str | Path | None = None,
    model_signals_path: str | Path | None = None,
    cfd_reconciliation_path: str | Path | None = None,
    cfd_mapping_path: str | Path | None = None,
    procurement_bridge_path: str | Path | None = None,
    comparison_register_path: str | Path | None = None,
    recovery_candidates_path: str | Path | None = None,
    network_bridge_path: str | Path | None = None,
    comparison_assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle, output = Path(bundle_dir), Path(output_dir)
    tariffs_path, ledger_path = bundle / "period_tariffs.csv", bundle / "component_ledger.csv"
    if not tariffs_path.exists() or not ledger_path.exists():
        raise HistoricalReportError("Bundle must contain period_tariffs.csv and component_ledger.csv")
    profile = pd.read_csv(consumption_profile_path) if consumption_profile_path else None
    evidence_paths = {
        "model_signals": model_signals_path,
        "cfd_reconciliation": cfd_reconciliation_path,
        "cfd_mapping": cfd_mapping_path,
        "procurement_bridge": procurement_bridge_path,
        "comparison_register": comparison_register_path,
        "recovery_candidates": recovery_candidates_path,
        "network_bridge": network_bridge_path,
    }
    tables, checks = build_report_tables(
        pd.read_csv(tariffs_path), pd.read_csv(ledger_path),
        annual_consumption_kwh=annual_consumption_kwh, consumption_profile=profile,
        model_signals=pd.read_csv(model_signals_path) if model_signals_path else None,
        cfd_reconciliation=json.loads(Path(cfd_reconciliation_path).read_text(encoding="utf-8")) if cfd_reconciliation_path else None,
        cfd_mapping=json.loads(Path(cfd_mapping_path).read_text(encoding="utf-8")) if cfd_mapping_path else None,
        procurement_bridge=pd.read_csv(procurement_bridge_path) if procurement_bridge_path else None,
        comparison_register=pd.read_csv(comparison_register_path) if comparison_register_path else None,
        recovery_candidates=pd.read_csv(recovery_candidates_path) if recovery_candidates_path else None,
        network_bridge=pd.read_csv(network_bridge_path) if network_bridge_path else None,
        comparison_assumptions=comparison_assumptions,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    year = int(pd.to_datetime(tables["period_bills"]["period_start"]).dt.year.iloc[0])
    provenance = {
        "schema_version": checks["ledger_schema_version"], "method_version": METHOD_VERSION,
        "calendar_year": year, "bundle_dir": str(bundle),
        "inputs": {
            "period_tariffs.csv": file_sha256(tariffs_path),
            "component_ledger.csv": file_sha256(ledger_path),
            **({str(consumption_profile_path): file_sha256(consumption_profile_path)} if consumption_profile_path else {}),
            **({str(path): file_sha256(path) for path in evidence_paths.values() if path}),
        },
        "model_run_id": None,
        "model_evidence": checks["model_validation_status"],
        "comparison_assumptions": comparison_assumptions or {},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (output / "report.html").write_text(_report_html(tables, checks, year), encoding="utf-8")
    return checks


def _main(snakemake) -> None:
    profile = list(snakemake.input.get("consumption_profile", []))
    def optional_input(name):
        values = list(snakemake.input.get(name, []))
        return values[0] if values else None
    write_report(
        snakemake.params.bundle_dir, snakemake.params.output_dir,
        annual_consumption_kwh=float(snakemake.params.annual_consumption_kwh),
        consumption_profile_path=profile[0] if profile else None,
        model_signals_path=optional_input("model_signals"),
        cfd_reconciliation_path=optional_input("cfd_reconciliation"),
        cfd_mapping_path=optional_input("cfd_mapping"),
        procurement_bridge_path=optional_input("procurement_bridge"),
        comparison_register_path=optional_input("comparison_register"),
        recovery_candidates_path=optional_input("recovery_candidates"),
        network_bridge_path=optional_input("comparison_network_bridge"),
        comparison_assumptions=snakemake.params.get("comparison_assumptions", {}),
    )


if "snakemake" in globals():
    _main(snakemake)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annual-consumption-kwh", type=float, default=3100.0)
    parser.add_argument("--consumption-profile")
    parser.add_argument("--model-signals")
    parser.add_argument("--cfd-reconciliation")
    parser.add_argument("--cfd-mapping")
    parser.add_argument("--procurement-bridge")
    parser.add_argument("--comparison-register")
    parser.add_argument("--recovery-candidates")
    parser.add_argument("--network-bridge")
    parser.add_argument("--support-recovery-method", choices=["modelled_system_demand_proxy", "eligible_supplier_sales"])
    parser.add_argument("--eligible-supplier-sales-mwh", type=float)
    args = parser.parse_args()
    write_report(
        args.bundle, args.output_dir,
        annual_consumption_kwh=args.annual_consumption_kwh,
        consumption_profile_path=args.consumption_profile,
        model_signals_path=args.model_signals,
        cfd_reconciliation_path=args.cfd_reconciliation,
        cfd_mapping_path=args.cfd_mapping,
        procurement_bridge_path=args.procurement_bridge,
        comparison_register_path=args.comparison_register,
        recovery_candidates_path=args.recovery_candidates,
        network_bridge_path=args.network_bridge,
        comparison_assumptions={"support_recovery_method": args.support_recovery_method,
                                "eligible_supplier_sales_mwh": args.eligible_supplier_sales_mwh},
    )
