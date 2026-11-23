"""Reconcile saved Ofgem component evidence and audit procurement transferability.

Run from the repository root with python -m scripts.consumer_bill.build_historical_evidence_panel.
Uses saved files only; never triggers optimisation or changes forecast assumptions.
"""

import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLES = {2020: "2020_historical_annual", 2024: "2024_annual"}
DETAILS = {"policy": "policy.aggregate", "smart_meter": "supplier.smart_metering",
           "wholesale": "wholesale.direct_fuel", "network": "network.aggregate",
           "adjustment": "adjustment.allowance"}
SPOT_FILES = {
    2020: "Validation_2020_pypsa_spot_procurement_periods.csv",
    2024: "Historical_2024_ETYS_14zone_market_pypsa_spot_procurement_periods.csv",
}


def validate_periods(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Require exactly one contiguous calendar year, without duplicate periods."""
    result = frame.copy().sort_values("delivery_start")
    start = pd.to_datetime(result.delivery_start)
    end = pd.to_datetime(result.delivery_end)
    if result.cap_period.duplicated().any() or (end < start).any():
        raise ValueError("Duplicate or invalid delivery periods")
    if start.iloc[0] != pd.Timestamp(year, 1, 1) or end.iloc[-1] != pd.Timestamp(year, 12, 31):
        raise ValueError("Incomplete calendar year")
    if len(start) > 1 and not np.array_equal(start.iloc[1:].to_numpy(), (end.iloc[:-1] + pd.Timedelta(days=1)).to_numpy()):
        raise ValueError("Overlapping or missing delivery days")
    result["calendar_weight"] = (end - start).dt.days.add(1) / (366 if calendar.isleap(year) else 365)
    return result


def build_panel(bundle: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return additive parent/detail panel and independent reconciliation checks."""
    ledger = pd.read_csv(bundle / "component_ledger.csv")
    keys = ["cap_period", "component_id"]
    if ledger.duplicated(keys + ["region"]).any():
        raise ValueError("Duplicate regional component")
    if not ledger.groupby(keys).region.nunique().eq(14).all():
        raise ValueError("Every component requires all 14 regions")
    amounts = ledger[["amount_gbp", "fixed_gbp_per_year_ex_vat", "variable_gbp_at_benchmark_ex_vat"]]
    if not np.isfinite(amounts.to_numpy()).all():
        raise ValueError("Non-finite ledger amount")
    if not np.allclose(amounts.amount_gbp, amounts.fixed_gbp_per_year_ex_vat + amounts.variable_gbp_at_benchmark_ex_vat, atol=1e-6, rtol=0):
        raise ValueError("Fixed and variable amounts do not reconcile")
    periods = ledger[["cap_period", "period_start", "period_end"]].drop_duplicates().rename(columns={"period_start": "delivery_start", "period_end": "delivery_end"})
    periods = validate_periods(periods, year).set_index("cap_period")
    parents = ledger.groupby(keys).agg(
        amount_gbp=("amount_gbp", "mean"),
        fixed_gbp=("fixed_gbp_per_year_ex_vat", "mean"),
        variable_gbp=("variable_gbp_at_benchmark_ex_vat", "mean"),
        label=("component_label", "first"),
        source_reference=("source_reference", lambda s: ";".join(sorted(set(s)))),
        source_hash=("source_hash", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()
    parents["row_role"] = "additive_parent"
    parents["parent_id"] = ""
    parents["source_file"] = str(bundle / "component_ledger.csv")
    rows = [parents]
    checks = []
    # Reconcile annualised component rates to tariff benchmark rates, not a
    # 366-day actual bill (standing-charge tariff denominators can be 365).
    tariffs = pd.read_csv(bundle / "period_tariffs.csv")
    for period, selected in ledger.groupby("cap_period"):
        expected = tariffs.loc[tariffs.cap_period.eq(period)].set_index("region").benchmark_gbp_per_year_ex_vat
        actual = selected.groupby("region").amount_gbp.sum().reindex(expected.index)
        if len(expected) != 14 or actual.isna().any():
            raise ValueError("Incomplete tariff region coverage")
        for region in expected.index:
            residual = float(actual[region] - expected[region])
            checks.append(dict(year=year, cap_period=period, parent_id="tariff_total", region=region,
                               residual_gbp=residual, status="pass" if abs(residual) < 1e-6 else "unresolved"))
    for name, parent in DETAILS.items():
        path = bundle / f"{name}_components.csv"
        detail = pd.read_csv(path)
        if detail.duplicated(["cap_period", "component"]).any():
            raise ValueError("Duplicate detail component")
        if set(detail.cap_period) - set(periods.index):
            raise ValueError("Unexpected detail period")
        for period in set(periods.index) - set(detail.cap_period):
            expected = parents.loc[parents.cap_period.eq(period) & parents.component_id.eq(parent), "amount_gbp"]
            checks.append(dict(year=year, cap_period=period, parent_id=parent, region="equal_mean_14",
                               residual_gbp=None, status="zero_parent_no_detail" if len(expected) == 1 and expected.iloc[0] == 0 else "missing_detail"))
        detail["amount_gbp"] = pd.to_numeric(detail.annual_gbp_per_3100kwh_ex_vat, errors="raise")
        if not np.isfinite(detail.amount_gbp).all():
            raise ValueError("Non-finite detail")
        for period, selected in detail.groupby("cap_period"):
            expected = parents.loc[parents.cap_period.eq(period) & parents.component_id.eq(parent), "amount_gbp"]
            if len(expected) != 1:
                raise ValueError("Missing parent")
            residual = float(selected.amount_gbp.sum() - expected.iloc[0])
            checks.append(dict(year=year, cap_period=period, parent_id=parent, region="equal_mean_14",
                               residual_gbp=residual, status="pass" if abs(residual) < 1e-5 else "unresolved"))
        detail["component_id"] = parent + ".detail." + detail.component
        detail["label"] = detail.component
        detail["parent_id"] = parent
        detail["row_role"] = "non_additive_detail"
        detail["source_file"] = str(path)
        detail["source_reference"] = detail.source_workbook + ":" + detail.source_cells.fillna("")
        rows.append(detail)
    panel = pd.concat(rows, ignore_index=True)
    panel["year"] = year
    panel["currency_basis"] = f"nominal GBP {year}, ex VAT, 3100 kWh, equal mean of 14 regions"
    panel["calendar_weight"] = panel.cap_period.map(periods.calendar_weight)
    panel["annual_contribution_gbp"] = panel.amount_gbp * panel.calendar_weight
    return panel, pd.DataFrame(checks)


def procurement_diagnostics(frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Audit target compatibility; evaluate a chronological within-year diagnostic.

    The diagnostic fixes the coefficient on earlier periods only. It is not
    a forward-curve forecast: both training and test inputs are delivery prices.
    """
    outputs, definitions = [], {}
    for year, frame in sorted(frames.items()):
        frame = validate_periods(frame, year)
        required = ["pypsa_spot_price_gbp_per_mwh", "published_energy_allowance_gbp_per_3100kwh_ex_vat"]
        if not np.isfinite(frame[required].to_numpy()).all():
            raise ValueError("Non-finite procurement data")
        definitions[year] = sorted(set(frame.source_energy_definition))
        frame["raw_prediction_gbp"] = frame.pypsa_spot_price_gbp_per_mwh * 3.1
        frame["raw_error_gbp"] = frame.raw_prediction_gbp - frame[required[1]]
        frame["year"] = year
        frame["diagnostic_prediction_gbp"] = np.nan
        frame["diagnostic_error_gbp"] = np.nan
        frame["diagnostic_multiplier"] = np.nan
        frame["split"] = "not_evaluated"
        if len(frame) >= 4 and len(definitions[year]) == 1:
            cut = len(frame) // 2
            train = frame.iloc[:cut]
            x, y, w = train.raw_prediction_gbp, train[required[1]], train.calendar_weight
            if float((w * x * x).sum()) <= 0:
                raise ValueError("No training price variation/energy")
            multiplier = float((w * x * y).sum() / (w * x * x).sum())
            frame["diagnostic_multiplier"] = multiplier
            frame["split"] = ["train"] * cut + ["test"] * (len(frame) - cut)
            mask = frame.split.eq("test")
            frame.loc[mask, "diagnostic_prediction_gbp"] = frame.loc[mask, "raw_prediction_gbp"] * multiplier
            frame.loc[mask, "diagnostic_error_gbp"] = frame.loc[mask, "diagnostic_prediction_gbp"] - frame.loc[mask, required[1]]
        outputs.append(frame)
    result = pd.concat(outputs, ignore_index=True)
    metrics = []
    for year, frame in result.groupby("year"):
        metrics.append(dict(year=int(year), scope="full_year_raw", bias_gbp=float((frame.raw_error_gbp * frame.calendar_weight).sum()),
                            mae_gbp=float((frame.raw_error_gbp.abs() * frame.calendar_weight).sum())))
        test = frame.loc[frame.split.eq("test")]
        if len(test):
            w = test.calendar_weight / test.calendar_weight.sum()
            metrics.append(dict(year=int(year), scope="held_out_delivery_periods", multiplier=float(test.diagnostic_multiplier.iloc[0]),
                                raw_mae_gbp=float((test.raw_error_gbp.abs() * w).sum()),
                                fitted_mae_gbp=float((test.diagnostic_error_gbp.abs() * w).sum())))
    compatible = len({tuple(v) for v in definitions.values()}) == 1
    return result, dict(target_definitions=definitions, cross_year_target_compatible=compatible,
                        cross_year_fit_status="not_performed_requires_forward_observation_inputs" if compatible else "blocked_incompatible_target_definitions",
                        forecast_validated=False, active_assumptions_changed=False, metrics=metrics,
                        limitations=["2020 base/peak blend and 2024 direct-energy term have different cost boundaries.",
                                     "Delivery-price diagnostics do not establish a forward procurement forecast.",
                                     "No cross-year nominal-currency intercept is fitted; no 2025-pound GBP52 add-on is applied to nominal historical prices.",
                                     "Saved model diagnostics are not independently rerun or certified as out-of-sample physical models."])


def main() -> None:
    """Write reproducible evidence artifacts and source fingerprints."""
    output = Path("resources/analysis/consumer_bill_evidence")
    output.mkdir(parents=True, exist_ok=True)
    panels, checks, inputs = [], [], []
    for year, name in BUNDLES.items():
        bundle = Path("data/market/ofgem_price_cap") / name
        panel, check = build_panel(bundle, year)
        panels.append(panel); checks.append(check)
        inputs.extend([bundle / "component_ledger.csv", bundle / "period_tariffs.csv"])
        inputs.extend(bundle / f"{name}_components.csv" for name in DETAILS)
        inputs.append(bundle / "source_provenance.json")
    panel = pd.concat(panels, ignore_index=True)
    check = pd.concat(checks, ignore_index=True)
    annual = panel.groupby(["year", "component_id", "label", "row_role"], as_index=False).annual_contribution_gbp.sum()
    frames = {}
    for year, name in SPOT_FILES.items():
        path = Path("resources/analysis") / name
        inputs.append(path)
        frames[year] = pd.read_csv(path)
    diagnostic, summary = procurement_diagnostics(frames)
    summary["source_reconciliation"] = check.status.value_counts().to_dict()
    summary["inputs_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    panel.to_csv(output / "historical_component_panel.csv", index=False)
    annual.to_csv(output / "historical_component_annual.csv", index=False)
    check.to_csv(output / "reconciliation_checks.csv", index=False)
    diagnostic.to_csv(output / "procurement_validation_periods.csv", index=False)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report = "# Historical evidence and procurement validation\n\n"
    report += "Source replay and forecasting are assessed separately. Amounts are nominal GBP in each historical year, excluding VAT, per 3,100 kWh benchmark.\n\n"
    report += f"Reconciliation checks: {summary['source_reconciliation']}. Parent rows alone are additive; detail rows must not be added again. Annual contributions are day-weighted allowance rates, not a dated household payment calculation.\n\n"
    report += f"Cross-year fit: **{summary['cross_year_fit_status']}**. Forecast validated: **false**.\n\n"
    report += "| Year | Scope | Bias | Raw MAE | Diagnostic MAE |\n| --- | --- | --- | --- | --- |\n"
    for m in summary["metrics"]:
        report += f"| {m['year']} | {m['scope']} | {m.get('bias_gbp', '')} | {m.get('mae_gbp', m.get('raw_mae_gbp', ''))} | {m.get('fitted_mae_gbp', '')} |\n"
    report += "\n" + "\n".join("- " + item for item in summary["limitations"])
    report += "\n\nNext requirement: align the cost boundaries using source procurement products, shaping/loss and CfD separation, then obtain observation-date forward inputs before a cross-year forecasting fit. The current future procurement allowance remains unvalidated.\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    main()
