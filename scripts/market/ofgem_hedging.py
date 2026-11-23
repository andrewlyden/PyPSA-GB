"""Published Ofgem wholesale-allowance building blocks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "cap_period", "observation_start", "observation_end", "delivery_start",
    "delivery_end", "base_share", "peak_share", "forward_index_gbp_per_mwh",
    "seasonal_to_monthly_shaping_rate", "monthly_to_hourly_shaping_rate",
    "day_ahead_rehedging_rate", "imbalance_rate", "transaction_cost_rate",
    "additional_risk_rate", "loss_multiplier_gb_average",
}


class OfgemHedgingError(ValueError):
    """Raised when a normalised Ofgem wholesale input is incomplete."""


def load_hedge_periods(path: str | Path) -> pd.DataFrame:
    """Load and validate the normalised period-level allowance inputs."""
    periods = pd.read_csv(
        path,
        parse_dates=["observation_start", "observation_end", "delivery_start", "delivery_end"],
    )
    missing = REQUIRED_COLUMNS.difference(periods.columns)
    if missing:
        raise OfgemHedgingError(f"Hedge input is missing columns: {sorted(missing)}")
    if periods.empty or periods["cap_period"].duplicated().any():
        raise OfgemHedgingError("Hedge input must contain one row for each cap period")
    numeric = REQUIRED_COLUMNS.difference(
        {"cap_period", "observation_start", "observation_end", "delivery_start", "delivery_end"}
    )
    periods.loc[:, list(numeric)] = periods.loc[:, list(numeric)].apply(pd.to_numeric, errors="raise")
    if not ((periods["base_share"] + periods["peak_share"]).round(12) == 1.0).all():
        raise OfgemHedgingError("Base and peak shares must sum to one")
    if (periods["loss_multiplier_gb_average"] < 1.0).any():
        raise OfgemHedgingError("Loss multipliers must be at least one")
    return periods.sort_values("delivery_start").reset_index(drop=True)


def direct_fuel_components(
    periods: pd.DataFrame, benchmark_consumption_mwh: float = 3.1
) -> pd.DataFrame:
    """Return the published forward-index, uplift, and loss bridge per period."""
    if benchmark_consumption_mwh <= 0:
        raise OfgemHedgingError("benchmark_consumption_mwh must be positive")
    detailed_columns = (
        ("direct_fuel_energy_cost", "direct_fuel_energy_cost_gbp_per_3100kwh", "published 3a DF direct energy cost"),
        ("backwardation_allowance", "backwardation_allowance_gbp_per_3100kwh", "published 3a DF backwardation allowance"),
        ("cfd_allowance_embedded_in_direct_fuel", "cfd_allowance_embedded_in_direct_fuel_gbp_per_3100kwh", "published 3a DF CfD allowance embedded within wholesale"),
    )
    rates = [
        ("seasonal_to_monthly_shaping", "seasonal_to_monthly_shaping_rate"),
        ("monthly_to_hourly_shaping", "monthly_to_hourly_shaping_rate"),
        ("day_ahead_rehedging", "day_ahead_rehedging_rate"),
        ("imbalance", "imbalance_rate"),
        ("transaction_cost", "transaction_cost_rate"),
        ("additional_risk_and_basis", "additional_risk_rate"),
    ]
    rows = []
    for _, period in periods.iterrows():
        if str(period.get("wholesale_detail_level", "")).casefold() == "published_3a_direct_fuel_components":
            detailed = []
            for component, column, method in detailed_columns:
                value = float(period[column])
                detailed.append(value)
                rows.append({
                    "cap_period": period["cap_period"], "component": component,
                    "gbp_per_benchmark_customer_ex_vat": value,
                    "gbp_per_mwh": value / benchmark_consumption_mwh,
                    "method": method,
                })
            total = sum(detailed)
            expected = float(period["forward_index_gbp_per_mwh"]) * benchmark_consumption_mwh
            if abs(total - expected) > 1e-5:
                raise OfgemHedgingError(
                    f"Published 3a wholesale components do not reconcile for {period['cap_period']}: "
                    f"{total} != {expected}"
                )
            rows.append({
                "cap_period": period["cap_period"], "component": "direct_fuel_allowance",
                "gbp_per_benchmark_customer_ex_vat": total, "gbp_per_mwh": total / benchmark_consumption_mwh,
                "method": "sum of published 3a DF direct-energy, backwardation and embedded-CfD terms",
            })
            continue
        index = float(period["forward_index_gbp_per_mwh"])
        raw = index * benchmark_consumption_mwh
        uplift_total = 0.0
        for component, column in rates:
            value = raw * float(period[column])
            uplift_total += value
            rows.append({
                "cap_period": period["cap_period"], "component": component,
                "gbp_per_benchmark_customer_ex_vat": value,
                "gbp_per_mwh": value / benchmark_consumption_mwh,
                "method": "published Ofgem percentage uplift applied to forward index",
            })
        post_uplift = raw + uplift_total
        supplied = post_uplift * float(period["loss_multiplier_gb_average"])
        # Annex 2 builds its forward index from separate seasonal base and
        # peak products.  The normalised 2020 source preserves their Ofgem
        # demand weights, so retain two additive rows rather than hiding the
        # blend behind one ``base_peak`` amount.
        base = raw * float(period["base_share"])
        peak = raw * float(period["peak_share"])
        rows.extend([
            {"cap_period": period["cap_period"], "component": "seasonal_base_forward_index", "gbp_per_benchmark_customer_ex_vat": base, "gbp_per_mwh": base / benchmark_consumption_mwh, "method": "published Annex 2 base-product demand-weight allocation of the six-month forward index"},
            {"cap_period": period["cap_period"], "component": "seasonal_peak_forward_index", "gbp_per_benchmark_customer_ex_vat": peak, "gbp_per_mwh": peak / benchmark_consumption_mwh, "method": "published Annex 2 peak-product demand-weight allocation of the six-month forward index"},
            {"cap_period": period["cap_period"], "component": "regional_losses_gb_average", "gbp_per_benchmark_customer_ex_vat": supplied - post_uplift, "gbp_per_mwh": (supplied - post_uplift) / benchmark_consumption_mwh, "method": "published GB-average regional loss multiplier"},
            {"cap_period": period["cap_period"], "component": "direct_fuel_allowance", "gbp_per_benchmark_customer_ex_vat": supplied, "gbp_per_mwh": supplied / benchmark_consumption_mwh, "method": "forward index plus published hedging uplifts and losses"},
        ])
    return pd.DataFrame(rows)

