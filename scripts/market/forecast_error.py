"""Day-ahead forecast error for the scheduling stage.

Why this exists
---------------
Both market stages solve on the outturn, so Stage 1 schedules with perfect
foresight. That is a fair description of self-dispatch, where intra-day trading
lets positions track the improving forecast until gate closure, but it flatters
a central dispatch, whose day-ahead schedule is locked on a forecast and whose
operator must then pay to correct the error in real time. FTI's REMA scheduling
and dispatch assessment for NESO (February 2026) identifies that correction as
the main consumer cost of central dispatch, so a central dispatch modelled
without it is an upper bound and nothing more.

What it does
------------
When ``market.wholesale.forecast_error.enabled`` is true, Stage 1 sees a
*forecast*: a national AR(1) error per technology, added to every unit's
capacity factor, calibrated so the national error has the configured RMSE as a
fraction of installed capacity. Demand gets a small multiplicative error on the
same basis. Stage 2 reloads the network and so always sees the outturn; its
anchored redispatch against the Stage 1 schedule is then exactly the cost of
resolving the forecast error.

The error is generated for the whole calendar year from the seed and sliced to
the solve period, so a week gate and the month that contains it see identical
forecasts, and every month of a year is drawn from one continuous series.

Calibration
-----------
The NESO day-ahead wind forecast has a capacity-weighted mean absolute error of
about 4.2% (Significance, 2024), which is roughly 5.3% RMSE for near-Gaussian
errors. The national error is shaped by each unit's availability,
``s = 4*CF*(1-CF)``, so it vanishes where there is no resource (night-time solar,
a becalmed farm) and at full output, which keeps the forecast unbiased; the
configured number is the standard deviation before shaping and the realised
national RMSE, lower by the RMS of the shape, is logged so it can be checked
against the target. Applying one national series to every unit is perfect
spatial correlation, so the national RMSE is right but the locational error is
understated; that is the conservative direction for a study of constraints.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_SEED = 2030
DEFAULT_AUTOCORRELATION_HOURS = 6.0
# Standard deviation of the national error *before* availability shaping
# (s = 4*CF*(1-CF)). The realised national RMSE as a fraction of capacity is
# lower by the RMS of the shape over the period, roughly 0.8 for wind and 0.55
# for solar (zero at night), so these land near 5.5% and 5% and are logged.
DEFAULT_TECHNOLOGIES: dict[str, float] = {
    "wind_offshore": 0.07,
    "wind_onshore": 0.07,
    "solar_pv": 0.09,
}
DEFAULT_DEMAND_RMSE = 0.015

_LOG = logging.getLogger(__name__)


def year_index(year: int) -> pd.DatetimeIndex:
    """Hourly index for one calendar year, the domain every error series lives on."""
    return pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")


def _seed_for(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def ar1_series(
    index: pd.DatetimeIndex, sigma: float, tau_hours: float, rng: np.random.Generator
) -> pd.Series:
    """Stationary AR(1) noise with standard deviation ``sigma`` and e-folding time ``tau_hours``."""
    n = len(index)
    if n == 0 or sigma <= 0:
        return pd.Series(0.0, index=index)
    phi = math.exp(-1.0 / tau_hours) if tau_hours > 0 else 0.0
    innov_sd = sigma * math.sqrt(max(0.0, 1.0 - phi * phi))
    eps = rng.normal(0.0, innov_sd, size=n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return pd.Series(x, index=index)


def national_error_series(
    year: int, sigma: float, tau_hours: float, seed: int, key: str
) -> pd.Series:
    """The whole-year error series for one technology, reproducible from ``(seed, key)``."""
    rng = np.random.default_rng(_seed_for(seed, key))
    return ar1_series(year_index(year), sigma, tau_hours, rng)


def _gb_load_columns(network) -> list[str]:
    loads = network.loads
    cols = [c for c in network.loads_t.p_set.columns if c in loads.index]
    if "country" in network.buses.columns:
        gb = set(network.buses.index[network.buses["country"].eq("GB")])
        cols = [c for c in cols if loads.at[c, "bus"] in gb]
    return cols


def apply_day_ahead_forecast_error(
    network,
    cfg: Mapping[str, Any] | None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Replace outturn capacity factors and demand with a day-ahead forecast, in place.

    Returns a summary of the realised national error per technology (fraction of
    installed capacity and MW), empty when disabled. Only the scheduling stage
    should call this; the balancing stage must keep the outturn.
    """
    log = logger or _LOG
    cfg = dict(cfg or {})
    if not cfg.get("enabled", False):
        return {}
    if len(network.snapshots) == 0:
        return {}

    seed = int(cfg.get("seed", DEFAULT_SEED))
    tau = float(cfg.get("autocorrelation_hours", DEFAULT_AUTOCORRELATION_HOURS))
    technologies = dict(cfg.get("technologies", DEFAULT_TECHNOLOGIES))
    demand_rmse = float(cfg.get("demand_rmse", DEFAULT_DEMAND_RMSE))
    snapshots = pd.DatetimeIndex(network.snapshots)
    year = int(snapshots[0].year)

    log.info("=" * 80)
    log.info("DAY-AHEAD FORECAST ERROR: STAGE 1 SCHEDULES ON A FORECAST, NOT THE OUTTURN")
    log.info("=" * 80)
    log.info(
        f"  seed {seed}, autocorrelation {tau:g} h, year {year}, "
        f"{len(snapshots)} snapshots"
    )

    summary: dict[str, Any] = {"seed": seed, "technologies": {}, "demand": None}
    pmax = network.generators_t.p_max_pu
    gens = network.generators

    for tech, sigma in technologies.items():
        sigma = float(sigma)
        cols = [g for g in gens.index[gens["carrier"].eq(tech)] if g in pmax.columns]
        if not cols or sigma <= 0:
            log.info(f"  {tech:<14} no time-varying units or zero error; skipped")
            continue
        err = national_error_series(year, sigma, tau, seed, tech).reindex(snapshots).fillna(0.0)
        before = pmax[cols].copy()
        # Shape the national error by each unit's availability: s = 4*CF*(1-CF)
        # is 1 at half output and 0 at none or full. A forecast cannot see
        # resource that is not there, so the error must vanish at CF = 0; an
        # unshaped additive error clipped at zero could only push night-time
        # solar upwards and forecast 4 TWh a week of sun after dark. The shape
        # also keeps clipping from binding, so the forecast is unbiased.
        shape = 4.0 * before * (1.0 - before)
        after = (before + shape.mul(err, axis=0)).clip(lower=0.0, upper=1.0)
        pmax.loc[:, cols] = after.values
        p_nom = gens.loc[cols, "p_nom"]
        cap = float(p_nom.sum())
        delta_mw = (after - before).mul(p_nom, axis=1).sum(axis=1)
        rmse_mw = float(np.sqrt((delta_mw**2).mean()))
        mae_mw = float(delta_mw.abs().mean())
        summary["technologies"][tech] = {
            "units": len(cols),
            "capacity_mw": cap,
            "target_rmse_of_capacity": sigma,
            "realised_rmse_of_capacity": rmse_mw / cap if cap else float("nan"),
            "realised_mae_of_capacity": mae_mw / cap if cap else float("nan"),
            "forecast_minus_outturn_gwh": float(delta_mw.sum() / 1e3),
        }
        log.info(
            f"  {tech:<14} {len(cols):5d} units, {cap:9.0f} MW: target RMSE "
            f"{sigma:.3f}, realised {rmse_mw / cap if cap else float('nan'):.3f} of "
            f"capacity ({rmse_mw:,.0f} MW); forecast minus outturn "
            f"{delta_mw.sum() / 1e3:+,.1f} GWh over the period"
        )

    if demand_rmse > 0:
        cols = _gb_load_columns(network)
        if cols:
            err = national_error_series(year, demand_rmse, tau, seed, "demand").reindex(
                snapshots
            ).fillna(0.0)
            pset = network.loads_t.p_set
            before = pset[cols].sum(axis=1)
            pset.loc[:, cols] = pset[cols].mul(1.0 + err, axis=0).clip(lower=0.0).values
            after = pset[cols].sum(axis=1)
            delta = after - before
            rel = float(np.sqrt(((delta / before.replace(0, np.nan)) ** 2).mean()))
            summary["demand"] = {
                "loads": len(cols),
                "target_rmse_fraction": demand_rmse,
                "realised_rmse_fraction": rel,
                "forecast_minus_outturn_gwh": float(delta.sum() / 1e3),
            }
            log.info(
                f"  demand         {len(cols):5d} loads: target RMSE {demand_rmse:.3f}, "
                f"realised {rel:.3f}; forecast minus outturn {delta.sum() / 1e3:+,.1f} GWh"
            )

    return summary
