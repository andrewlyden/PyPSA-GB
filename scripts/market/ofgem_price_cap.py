"""Ofgem default-tariff-cap electricity bill utilities.

The module intentionally separates an exact reproduction of a published
levelised cap snapshot from a PyPSA wholesale-price counterfactual.  The latter
is an Ofgem-structured bill sensitivity, not a statutory cap calculation:
PyPSA produces spot prices whereas Ofgem's wholesale allowance is based on a
hedged forward procurement methodology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_RATE_COLUMNS = {
    "payment_method",
    "region",
    "nil_gbp_per_year_ex_vat",
    "benchmark_gbp_per_year_ex_vat",
    "benchmark_kwh",
}
VALID_MODES = {"official", "counterfactual"}


class OfgemPriceCapError(ValueError):
    """Raised when a consumer-bill request cannot be calculated safely."""


def load_snapshot(snapshot_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load and validate a normalised single-rate electricity cap snapshot."""
    root = Path(snapshot_dir)
    rates_path = root / "electricity_single_rate.csv"
    metadata_path = root / "metadata.json"
    if not rates_path.exists() or not metadata_path.exists():
        raise OfgemPriceCapError(
            f"Ofgem snapshot '{root}' must contain electricity_single_rate.csv and metadata.json"
        )

    rates = pd.read_csv(rates_path)
    missing = REQUIRED_RATE_COLUMNS.difference(rates.columns)
    if missing:
        raise OfgemPriceCapError(f"Snapshot rate table is missing columns: {sorted(missing)}")
    if rates.empty or rates.duplicated(["payment_method", "region"]).any():
        raise OfgemPriceCapError("Snapshot rate table must have one row per payment method and region")

    for column in REQUIRED_RATE_COLUMNS.difference({"payment_method", "region"}):
        rates[column] = pd.to_numeric(rates[column], errors="raise")
    if (rates["benchmark_kwh"] <= 0).any():
        raise OfgemPriceCapError("benchmark_kwh must be positive")
    if (rates["benchmark_gbp_per_year_ex_vat"] < rates["nil_gbp_per_year_ex_vat"]).any():
        raise OfgemPriceCapError("Benchmark charges must not be lower than nil-consumption charges")

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata.get("vat_rate"), (int, float)):
        raise OfgemPriceCapError("Snapshot metadata must provide numeric vat_rate")
    return rates, metadata


def _normalise_regions(
    rates: pd.DataFrame, regions: Iterable[str], region_weights: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Filter regions and derive an explicit GB average on request."""
    requested = list(regions)
    if not requested:
        requested = ["GB_average"]

    selected = []
    for region in requested:
        if str(region).lower().replace(" ", "_") in {"gb_average", "gbavg"}:
            fields = ["nil_gbp_per_year_ex_vat", "benchmark_gbp_per_year_ex_vat", "benchmark_kwh"]
            if region_weights is None:
                average = rates.groupby("payment_method", as_index=False)[fields].mean()
                average["region"] = "GB average (equal regional weight)"
            else:
                required = {"ofgem_region", "domestic_meter_weight"}
                if missing := required.difference(region_weights.columns):
                    raise OfgemPriceCapError(f"Regional weights are missing columns: {sorted(missing)}")
                weights = region_weights[["ofgem_region", "domestic_meter_weight"]].copy()
                weights["domestic_meter_weight"] = pd.to_numeric(
                    weights["domestic_meter_weight"], errors="raise"
                )
                if weights["ofgem_region"].duplicated().any() or not np.isclose(weights["domestic_meter_weight"].sum(), 1.0):
                    raise OfgemPriceCapError("Regional meter weights must be unique and sum to one")
                missing_regions = set(rates["region"]).difference(weights["ofgem_region"])
                if missing_regions:
                    raise OfgemPriceCapError(f"Regional weights do not cover: {sorted(missing_regions)}")
                weighted = rates.merge(weights, left_on="region", right_on="ofgem_region", how="left", validate="many_to_one")
                records = []
                for payment_method, group in weighted.groupby("payment_method"):
                    records.append({
                        "payment_method": payment_method,
                        "region": "GB average (domestic-meter weighted)",
                        **{field: float((group[field] * group["domestic_meter_weight"]).sum()) for field in fields},
                    })
                average = pd.DataFrame(records)
            selected.append(average[rates.columns])
        else:
            match = rates.loc[rates["region"] == region]
            if match.empty:
                raise OfgemPriceCapError(f"Unknown charge-restriction region: {region!r}")
            selected.append(match)
    return pd.concat(selected, ignore_index=True)


def _normalise_payment_methods(rates: pd.DataFrame, payment_methods: Iterable[str]) -> pd.DataFrame:
    requested = list(payment_methods)
    if not requested:
        requested = ["direct_debit"]
    unknown = sorted(set(requested).difference(rates["payment_method"].unique()))
    if unknown:
        raise OfgemPriceCapError(f"Unknown payment method(s): {unknown}")
    return rates.loc[rates["payment_method"].isin(requested)].copy()


def demand_weighted_price(
    prices_gbp_per_mwh: pd.Series,
    demand_mw: pd.Series,
    snapshot_hours: pd.Series | None = None,
) -> float:
    """Return the demand- and duration-weighted hourly wholesale price."""
    prices = pd.to_numeric(prices_gbp_per_mwh, errors="coerce")
    demand = pd.to_numeric(demand_mw, errors="coerce").reindex(prices.index)
    if snapshot_hours is None:
        hours = pd.Series(1.0, index=prices.index)
    else:
        hours = pd.to_numeric(snapshot_hours, errors="coerce").reindex(prices.index)
    valid = np.isfinite(prices) & np.isfinite(demand) & np.isfinite(hours) & (hours > 0) & (demand >= 0)
    if not valid.all() or prices.index.has_duplicates:
        raise OfgemPriceCapError("Demand-weighted prices require complete, finite, unique snapshots with valid demand and hours")
    denominator = (demand.loc[valid] * hours.loc[valid]).sum()
    if denominator <= 0:
        raise OfgemPriceCapError("Cannot calculate a demand-weighted price without positive demand")
    return float((prices.loc[valid] * demand.loc[valid] * hours.loc[valid]).sum() / denominator)


def require_full_calendar_year(index: pd.DatetimeIndex, snapshot_hours: pd.Series) -> None:
    """Reject partial-period counterfactuals unless a caller opts out explicitly."""
    if len(index) == 0:
        raise OfgemPriceCapError("Counterfactual bill requires non-empty wholesale price data")
    first, last = index.min(), index.max()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise OfgemPriceCapError("Full-year snapshots must be unique and chronological")
    if first.year != last.year or first.month != 1 or first.day != 1 or first.hour != 0:
        raise OfgemPriceCapError("Counterfactual bill requires a complete calendar year starting 1 January")
    expected_hours = 8784.0 if pd.Timestamp(first.year, 12, 31).is_leap_year else 8760.0
    actual_hours = float(snapshot_hours.reindex(index).sum())
    weights = snapshot_hours.reindex(index)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise OfgemPriceCapError("Full-year snapshot weights must be finite and positive")
    if not np.isclose(actual_hours, expected_hours, rtol=0, atol=0.01):
        raise OfgemPriceCapError(
            f"Counterfactual bill requires {expected_hours:.0f} weighted hours; found {actual_hours:.2f}"
        )
    ends = index[1:].append(pd.DatetimeIndex([pd.Timestamp(first.year + 1, 1, 1, tz=index.tz)]))
    intervals = (ends - index).total_seconds() / 3600
    if not np.allclose(weights.to_numpy(), intervals, rtol=0, atol=1e-6):
        raise OfgemPriceCapError("Snapshot weights must cover consecutive calendar intervals without gaps or overlaps")


def calculate_tariffs(
    rates: pd.DataFrame,
    *,
    regions: Iterable[str],
    payment_methods: Iterable[str],
    mode: str = "official",
    vat_rate: float | None = None,
    modelled_wholesale_price_gbp_per_mwh: float | None = None,
    reference_wholesale_price_gbp_per_mwh: float | None = None,
    region_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive standing charges and unit rates from nil and benchmark cap charges."""
    if mode not in VALID_MODES:
        raise OfgemPriceCapError(f"mode must be one of {sorted(VALID_MODES)}")
    if vat_rate is None:
        vat_rate = 0.05
    if vat_rate < 0:
        raise OfgemPriceCapError("vat_rate must be non-negative")

    selected = _normalise_payment_methods(_normalise_regions(rates, regions, region_weights), payment_methods)
    selected = selected.copy()
    selected["wholesale_price_delta_gbp_per_mwh"] = 0.0
    if mode == "counterfactual":
        if modelled_wholesale_price_gbp_per_mwh is None or reference_wholesale_price_gbp_per_mwh is None:
            raise OfgemPriceCapError("Counterfactual mode requires modelled and reference wholesale prices")
        selected["wholesale_price_delta_gbp_per_mwh"] = (
            float(modelled_wholesale_price_gbp_per_mwh) - float(reference_wholesale_price_gbp_per_mwh)
        )

    selected["mode"] = mode
    selected["vat_rate"] = float(vat_rate)
    selected["standing_charge_gbp_per_day_ex_vat"] = selected["nil_gbp_per_year_ex_vat"] / 365.0
    selected["unit_rate_gbp_per_kwh_ex_vat"] = (
        selected["benchmark_gbp_per_year_ex_vat"] - selected["nil_gbp_per_year_ex_vat"]
    ) / selected["benchmark_kwh"]
    selected["counterfactual_unit_delta_gbp_per_kwh_ex_vat"] = (
        selected["wholesale_price_delta_gbp_per_mwh"] / 1000.0
    )
    selected["unit_rate_gbp_per_kwh_ex_vat"] += selected["counterfactual_unit_delta_gbp_per_kwh_ex_vat"]
    selected["standing_charge_p_per_day_inc_vat"] = (
        selected["standing_charge_gbp_per_day_ex_vat"] * (1 + vat_rate) * 100
    )
    selected["unit_rate_p_per_kwh_inc_vat"] = (
        selected["unit_rate_gbp_per_kwh_ex_vat"] * (1 + vat_rate) * 100
    )
    return selected


def calculate_bills(tariffs: pd.DataFrame, annual_consumption_kwh: Iterable[float]) -> pd.DataFrame:
    """Apply a tariff table to one or more annual electricity-consumption values."""
    consumptions = [float(value) for value in annual_consumption_kwh]
    if not consumptions or any(value < 0 for value in consumptions):
        raise OfgemPriceCapError("annual_consumption_kwh must contain non-negative values")
    rows = []
    for consumption in consumptions:
        bill = tariffs.copy()
        bill["annual_consumption_kwh"] = consumption
        bill["standing_charge_gbp_ex_vat"] = bill["nil_gbp_per_year_ex_vat"]
        bill["base_energy_charge_gbp_ex_vat"] = (
            (bill["benchmark_gbp_per_year_ex_vat"] - bill["nil_gbp_per_year_ex_vat"])
            * consumption / bill["benchmark_kwh"]
        )
        bill["wholesale_price_delta_gbp_ex_vat"] = (
            bill["wholesale_price_delta_gbp_per_mwh"] * consumption / 1000.0
        )
        bill["consumer_bill_gbp_ex_vat"] = (
            bill["standing_charge_gbp_ex_vat"]
            + bill["base_energy_charge_gbp_ex_vat"]
            + bill["wholesale_price_delta_gbp_ex_vat"]
        )
        bill["vat_gbp"] = bill["consumer_bill_gbp_ex_vat"] * bill["vat_rate"]
        bill["consumer_bill_gbp_inc_vat"] = bill["consumer_bill_gbp_ex_vat"] + bill["vat_gbp"]
        rows.append(bill)
    return pd.concat(rows, ignore_index=True)
