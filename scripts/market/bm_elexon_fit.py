"""
Per-Carrier ELEXON Price Fit for Fallback Pricing

When a generator has no direct ELEXON BMU match, the default fallback
assigns it the carrier median — a single number, regardless of the
generator's marginal cost. This module fits a per-carrier linear
relationship between ELEXON-matched generator marginal costs and their
observed ELEXON offer/bid prices:

    offer_price[g] = α · marginal_cost[g] + β
    bid_price[g]   = γ · marginal_cost[g] + δ

so that fallback generators inherit the intra-carrier MC dispersion
(efficient vs inefficient CCGTs, legacy vs recent wind, etc.) rather
than being collapsed onto a single carrier median.

A fit is rejected (falls back to the upstream fallback) when:
  - The carrier has fewer than ``min_matched_gens`` ELEXON matches
  - The fit R² is below ``min_r2``
  - The predicted price exceeds ``max_price`` in magnitude (strategic
    outlier — keep the behaviour of the carrier_average sanity cap)

Optionally also builds a per-carrier normalised hourly shape from the
matched ELEXON data, so unmatched generators get time-varying prices
with the same peak/offpeak pattern as their carrier.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CarrierFit:
    """Regression coefficients and diagnostics for one carrier."""
    carrier: str
    alpha: float          # offer slope
    beta: float           # offer intercept
    gamma: float          # bid slope
    delta: float          # bid intercept
    r2_offer: float
    r2_bid: float
    n_matched: int
    mean_mc: float
    mean_offer: float
    mean_bid: float


def _safe_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(y_hat)
    if mask.sum() < 2:
        return float("nan")
    y = y[mask]
    y_hat = y_hat[mask]
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        # Constant target: R² is ill-defined. Treat as perfect when residual
        # is effectively zero (intercept fit), else zero.
        return 1.0 if ss_res <= 1e-9 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Linear least-squares fit. Returns (slope, intercept, R²)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    x_m = x[mask]
    y_m = y[mask]

    if np.allclose(x_m.std(), 0.0):
        # Zero-variance MC (e.g. wind, where MC≈0 for everyone) → intercept-only
        slope = 0.0
        intercept = float(y_m.mean())
    else:
        slope, intercept = np.polyfit(x_m, y_m, 1)
        slope = float(slope)
        intercept = float(intercept)

    y_hat = slope * x_m + intercept
    return slope, intercept, _safe_r2(y_m, y_hat)


def fit_carrier_prices(
    generators: pd.DataFrame,
    elexon_offers: pd.DataFrame,
    elexon_bids: pd.DataFrame,
    mapped_gen_names: set,
    min_matched_gens: int,
    logger: logging.Logger,
) -> dict[str, CarrierFit]:
    """Fit ``offer = α·MC + β`` and ``bid = γ·MC + δ`` per carrier.

    Parameters
    ----------
    generators
        PyPSA ``network.generators`` DataFrame (needs ``carrier`` and
        ``marginal_cost`` columns).
    elexon_offers, elexon_bids
        Hourly ELEXON price DataFrames with generator names as columns
        (already renamed from BMU IDs). Bids are in ESO-cost convention
        (positive = cost to ESO to decrease) — the caller is expected to
        have negated raw ELEXON bids before calling.
    mapped_gen_names
        Set of generator names that have at least one ELEXON-matched BMU.
    min_matched_gens
        A carrier with fewer ELEXON-matched generators is skipped (no fit
        returned; caller should fall back to the prior behaviour).

    Returns
    -------
    dict
        ``{carrier_name: CarrierFit}`` with one entry per carrier that
        yielded a valid fit. Carriers without enough data are absent.
    """
    fits: dict[str, CarrierFit] = {}

    for carrier in generators["carrier"].unique():
        car_gens = generators[generators["carrier"] == carrier]
        matched = car_gens.index[car_gens.index.isin(mapped_gen_names)]

        matched_offer = [g for g in matched if g in elexon_offers.columns]
        matched_bid = [g for g in matched if g in elexon_bids.columns]
        if len(matched_offer) < min_matched_gens and len(matched_bid) < min_matched_gens:
            continue

        # Per-generator ELEXON mean (matches what market_utils does for the
        # static fallback) paired with its marginal cost.
        mcs_offer = car_gens.loc[matched_offer, "marginal_cost"].to_numpy(dtype=float)
        mcs_bid = car_gens.loc[matched_bid, "marginal_cost"].to_numpy(dtype=float)
        offer_vals = elexon_offers[matched_offer].mean(axis=0).to_numpy(dtype=float)
        bid_vals = elexon_bids[matched_bid].mean(axis=0).to_numpy(dtype=float)

        alpha, beta, r2_offer = _fit_linear(mcs_offer, offer_vals)
        gamma, delta, r2_bid = _fit_linear(mcs_bid, bid_vals)

        mean_mc = float(
            np.nanmean(np.concatenate([mcs_offer, mcs_bid]))
            if (len(mcs_offer) + len(mcs_bid)) > 0 else np.nan
        )
        mean_offer = float(np.nanmean(offer_vals)) if len(offer_vals) else np.nan
        mean_bid = float(np.nanmean(bid_vals)) if len(bid_vals) else np.nan

        fits[carrier] = CarrierFit(
            carrier=carrier,
            alpha=alpha, beta=beta,
            gamma=gamma, delta=delta,
            r2_offer=r2_offer, r2_bid=r2_bid,
            n_matched=int(len(matched)),
            mean_mc=mean_mc,
            mean_offer=mean_offer,
            mean_bid=mean_bid,
        )

        logger.info(
            f"  fit carrier={carrier:30s} n={len(matched):3d}  "
            f"offer={alpha:+.3f}·MC+{beta:+.1f}  R²={r2_offer:.2f}   "
            f"bid={gamma:+.3f}·MC+{delta:+.1f}  R²={r2_bid:.2f}"
        )

    return fits


def apply_fitted_fallback(
    generators: pd.DataFrame,
    unmatched_index: pd.Index,
    fits: dict[str, CarrierFit],
    min_r2: float,
    max_price: float,
    logger: logging.Logger,
) -> tuple[pd.Series, pd.Series, pd.Index]:
    """Apply carrier fits to unmatched generators.

    Returns
    -------
    offer : pd.Series
        Fitted offer price per generator, indexed by the *applied* subset of
        ``unmatched_index`` (carriers without a valid fit are absent — the
        caller should fall back to its prior behaviour for those).
    bid : pd.Series
        Fitted bid price, same index.
    applied : pd.Index
        The subset of ``unmatched_index`` for which fitted values were
        returned.
    """
    offers: dict[str, float] = {}
    bids: dict[str, float] = {}
    used_carriers: dict[str, int] = {}
    rejected_carriers: dict[str, str] = {}

    for carrier, fit in fits.items():
        if not (np.isfinite(fit.r2_offer) and fit.r2_offer >= min_r2):
            rejected_carriers[carrier] = f"offer R²={fit.r2_offer:.2f} < {min_r2}"
            continue
        if not (np.isfinite(fit.r2_bid) and fit.r2_bid >= min_r2):
            rejected_carriers[carrier] = f"bid R²={fit.r2_bid:.2f} < {min_r2}"
            continue

    for idx in unmatched_index:
        carrier = generators.loc[idx, "carrier"]
        if carrier not in fits or carrier in rejected_carriers:
            continue
        fit = fits[carrier]
        mc = float(generators.loc[idx, "marginal_cost"])
        if not np.isfinite(mc):
            continue

        offer_pred = fit.alpha * mc + fit.beta
        bid_pred = fit.gamma * mc + fit.delta

        if abs(offer_pred) > max_price or abs(bid_pred) > max_price:
            # Strategic outlier — same behaviour as the £500 carrier_average cap
            continue

        offers[idx] = offer_pred
        bids[idx] = bid_pred
        used_carriers[carrier] = used_carriers.get(carrier, 0) + 1

    if used_carriers:
        logger.info(
            f"Fitted fallback applied: {sum(used_carriers.values())} generators "
            f"across {len(used_carriers)} carriers"
        )
        for car in sorted(used_carriers):
            fit = fits[car]
            logger.info(
                f"  {car}: {used_carriers[car]} gens, "
                f"offer={fit.alpha:+.3f}·MC+{fit.beta:+.1f} (R²={fit.r2_offer:.2f}), "
                f"bid={fit.gamma:+.3f}·MC+{fit.delta:+.1f} (R²={fit.r2_bid:.2f})"
            )
    if rejected_carriers:
        for car, reason in sorted(rejected_carriers.items()):
            logger.info(f"  Fit rejected for {car}: {reason}")

    offer_series = pd.Series(offers, dtype=float)
    bid_series = pd.Series(bids, dtype=float)
    applied = offer_series.index.intersection(bid_series.index)
    return offer_series.loc[applied], bid_series.loc[applied], applied


def build_carrier_shape(
    generators: pd.DataFrame,
    elexon_prices: pd.DataFrame,
    mapped_gen_names: set,
    min_matched_gens: int,
) -> pd.DataFrame:
    """Build a carrier-mean-normalised hourly shape per carrier.

    For each carrier with at least ``min_matched_gens`` ELEXON-matched
    generators, compute the carrier's mean hourly price (across matched
    gens) and divide by the overall time mean. The result is a (snapshots
    × carrier) DataFrame where every column has mean 1.0 and carries the
    hourly pattern of its carrier.

    Multiply an unmatched generator's static fitted price by its carrier
    column to get a per-snapshot price series.
    """
    shapes = {}
    for carrier in generators["carrier"].unique():
        car_gens = generators[generators["carrier"] == carrier]
        matched = [
            g for g in car_gens.index
            if g in mapped_gen_names and g in elexon_prices.columns
        ]
        if len(matched) < min_matched_gens:
            continue
        carrier_mean_t = elexon_prices[matched].mean(axis=1)
        overall_mean = carrier_mean_t.mean()
        if not np.isfinite(overall_mean) or abs(overall_mean) < 1e-6:
            continue
        shape = carrier_mean_t / overall_mean
        # Protect against divide-by-zero in downstream multiplication by
        # clipping extreme hourly multipliers; ELEXON prices can spike.
        shape = shape.clip(lower=-10.0, upper=10.0).fillna(1.0)
        shapes[carrier] = shape

    if not shapes:
        return pd.DataFrame()
    return pd.DataFrame(shapes)
