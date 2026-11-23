"""Validate the portable annual input bundle used by a consumer-bill case.

The bill workflow deliberately consumes normalised annual inputs rather than
embedding a particular year's Ofgem workbook layout in every downstream rule.
Historical bundles contain observed allowances/outturn-derived inputs; future
bundles contain explicitly documented projections using the same schema.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.market.ofgem_hedging import load_hedge_periods
from scripts.market.ofgem_price_cap import load_snapshot
from scripts.consumer_bill.bill_scope import bill_scope
from scripts.consumer_bill.component_ledger import validate_component_ledger


COMPONENT_COLUMNS = {"component", "annualised_gbp_per_3100kwh_ex_vat"}


def _consumer_bill_config(scenario_config: dict) -> dict:
    return scenario_config.get("consumer_bill", {}).get("price_cap", {})


def _validate_components(path: str | Path, label: str, year: int) -> int:
    frame = pd.read_csv(path)
    missing = COMPONENT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if frame.empty or frame["component"].isna().any():
        raise ValueError(f"{label} must contain at least one named component")
    values = pd.to_numeric(frame["annualised_gbp_per_3100kwh_ex_vat"], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains missing or non-finite amounts")
    for column in ("delivery_start", "delivery_end"):
        if column in frame:
            dates = pd.to_datetime(frame[column], errors="raise")
            if dates.isna().any() or not dates.dt.year.eq(year).all():
                raise ValueError(f"{label} {column} must belong to modelled year {year}")
    return int(len(frame))


def validate_bundle(
    bundle_dir: str | Path,
    scenario_config: dict,
    hedge_inputs_path: str | Path,
    policy_path: str | Path,
    network_path: str | Path,
    smart_path: str | Path,
    adjustment_path: str | Path,
    source_provenance_path: str | Path | None = None,
) -> dict:
    """Validate standard inputs and return a provenance record."""
    root = Path(bundle_dir)
    rates, metadata = load_snapshot(root)
    periods = load_hedge_periods(hedge_inputs_path)
    if periods.empty:
        raise ValueError("Hedged-wholesale input must contain at least one delivery period")

    bill = _consumer_bill_config(scenario_config)
    scope = bill_scope(scenario_config, metadata, bill.get("mode", "official"))
    modelled_year = int(scenario_config.get("modelled_year"))
    if (pd.Timestamp(metadata["effective_from"]) != pd.Timestamp(modelled_year, 1, 1)
            or pd.Timestamp(metadata["effective_to"]) != pd.Timestamp(modelled_year, 12, 31)):
        raise ValueError("Annual bill tariff metadata must cover the full modelled calendar year")
    input_mode = bill.get(
        "input_mode", "historical_backtest" if modelled_year <= 2024 else "future_projection"
    )
    if input_mode not in {"historical_backtest", "future_projection"}:
        raise ValueError("consumer_bill.price_cap.input_mode must be historical_backtest or future_projection")

    delivery_start = pd.to_datetime(periods["delivery_start"], errors="raise")
    delivery_end = pd.to_datetime(periods["delivery_end"], errors="raise")
    if (delivery_end < delivery_start).any():
        raise ValueError("Hedged-wholesale delivery periods must have non-negative duration")
    ordered = pd.DataFrame({"start": delivery_start, "end": delivery_end}).sort_values("start")
    expected = pd.Timestamp(modelled_year, 1, 1)
    for row in ordered.itertuples(index=False):
        if row.start != expected:
            raise ValueError(f"Delivery periods have a gap, overlap or wrong year at {expected.date()}")
        expected = row.end + pd.Timedelta(days=1)
    if expected != pd.Timestamp(modelled_year + 1, 1, 1):
        raise ValueError("Delivery periods must cover the full modelled calendar year")
    component_rows = {
        "policy": _validate_components(policy_path, "Policy component bundle", modelled_year),
        "network": _validate_components(network_path, "Network component bundle", modelled_year),
        "smart_metering": _validate_components(smart_path, "Smart-metering component bundle", modelled_year),
        "adjustment": _validate_components(adjustment_path, "Adjustment component bundle", modelled_year),
    }
    period_tariffs_path = root / "period_tariffs.csv"
    ledger_path = root / "component_ledger.csv"
    strict_period_accounting = period_tariffs_path.exists() and ledger_path.exists()
    period_tariff_rows = 0
    ledger_rows = 0
    if strict_period_accounting:
        period_tariffs = pd.read_csv(period_tariffs_path)
        required_period = {
            "cap_period", "delivery_start", "delivery_end", "region", "payment_method",
            "nil_gbp_per_year_ex_vat", "benchmark_gbp_per_year_ex_vat",
            "standing_charge_gbp_per_day_ex_vat", "unit_rate_gbp_per_kwh_ex_vat",
        }
        if missing := required_period.difference(period_tariffs.columns):
            raise ValueError(f"Period tariff table is missing columns: {sorted(missing)}")
        if period_tariffs.duplicated(["cap_period", "region", "payment_method"]).any():
            raise ValueError("Period tariff identities must be unique")
        ledger = validate_component_ledger(
            pd.read_csv(ledger_path),
            expected_start=f"{modelled_year}-01-01", expected_end=f"{modelled_year}-12-31",
        )
        period_tariff_rows, ledger_rows = int(len(period_tariffs)), int(len(ledger))
    result = {
        "purpose": "Validated annual consumer-bill input bundle",
        **scope,
        "scenario": scenario_config.get("scenario_id"),
        "modelled_year": modelled_year,
        "input_mode": input_mode,
        "bundle_dir": str(root),
        "snapshot_id": metadata.get("snapshot_id"),
        "effective_from": metadata.get("effective_from"),
        "effective_to": metadata.get("effective_to"),
        "vat_rate": metadata.get("vat_rate"),
        "rate_rows": int(len(rates)),
        "procurement_period_count": int(len(periods)),
        "procurement_delivery_start": str(delivery_start.min().date()),
        "procurement_delivery_end": str(delivery_end.max().date()),
        "component_rows": component_rows,
        "strict_period_accounting": strict_period_accounting,
        "period_tariff_rows": period_tariff_rows,
        "component_ledger_rows": ledger_rows,
        "future_projection_rule": (
            "Projected values are inputs to a scenario, not observed outturns. "
            "Every changed component must retain its source and recovery assumption."
            if input_mode == "future_projection" else None
        ),
    }
    if source_provenance_path:
        provenance = json.loads(Path(source_provenance_path).read_text(encoding="utf-8"))
        result["source_provenance"] = {
            "purpose": provenance.get("purpose"),
            "calendar_year": provenance.get("calendar_year"),
            "source_count": (
                len(provenance.get("sources", []))
                + len(provenance.get("detail_sources", []))
                + sum(bool(source.get("applicable_tariff")) for source in provenance.get("sources", []))
            ),
        }
    return result


def _main(snakemake) -> None:
    result = validate_bundle(
        snakemake.params.bundle_dir,
        snakemake.params.scenario_config,
        snakemake.input.hedge_inputs_csv,
        snakemake.input.policy_components_csv,
        snakemake.input.network_components_csv,
        snakemake.input.smart_components_csv,
        snakemake.input.adjustment_components_csv,
        list(snakemake.input.source_provenance_json)[0] if snakemake.input.source_provenance_json else None,
    )
    output = Path(snakemake.output.validation_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if "snakemake" in globals():
    _main(snakemake)
