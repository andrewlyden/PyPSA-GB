"""
Diagnostic: compare ERA5 wind profiles against ESPENI observed generation.

⚠ IMPORTANT — this is a DIAGNOSTIC / VALIDATION tool, NOT the source of
correction factors applied to the model.

The model uses fixed performance factors configured in defaults.yaml under
``renewable_performance_factors.factors``.  Those factors represent the
uncurtailed available generation from real UK wind fleets (ERA5 bias +
idealised turbine curve correction), and are derived from literature rather
than ESPENI regression — see defaults.yaml for the derivation.

Why ESPENI is a WRONG calibration target for p_max_pu
------------------------------------------------------
``p_max_pu`` in PyPSA represents the *uncurtailed availability* — what a
generator can produce given the weather, before any network or economic
curtailment.  ESPENI records *metered dispatched* output, which is lower due to:

  - Network curtailment (B6 constraint curtails 5-15% of Scottish wind
    in high-wind periods; ESPENI captures the dispatched result)
  - Availability and maintenance (~3-5%)
  - Electrical losses (~1-2%)

In a copperplate wholesale model the optimiser dispatches all available wind.
Calibrating p_max_pu to ESPENI would bake in real-world curtailment as if it
were a lower wind resource, leaving the BM stage with too little wind to
curtail — the opposite of the intended two-stage architecture.

What this script does
---------------------
For each year Y and carrier it computes a *raw comparison ratio*:

  espeni_ratio = mean(ESPENI_wind_total) / mean(scaled_ERA5_total)

This ratio (printed to stdout and written to CSV) is useful for:
  - Quantifying the magnitude of ERA5 overestimate + curtailment combined
  - Sanity-checking profile quality by year
  - Informing (but not directly setting) the defaults.yaml performance factors

The reported ratio for 2020 is ~0.79, which is the sum of a genuine ERA5
overestimate (~20-25%) and real-world curtailment (~4-8%).  The defaults.yaml
factors use slightly higher values to avoid over-correcting for curtailment.

Usage
-----
  PYTHONPATH=. python scripts/generators/calibrate_wind_profiles.py \\
      --year 2020 \\
      --profiles-dir resources/renewable/profiles \\
      --generators-file resources/generators/SomeScenario_generators_full.csv \\
      --espeni-file data/demand/espeni.csv \\
      --output resources/renewable/wind_calibration_diagnostic.csv

  # Multiple years:
  PYTHONPATH=. python scripts/generators/calibrate_wind_profiles.py \\
      --year 2019 2020 2021 ...
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("calibrate_wind_profiles")

# ESPENI column names
_TIME_COL = "ELEC_elex_startTime[utc](datetime)"
_TOTAL_COL = "ELEC_POWER_TOTAL_WIND[MW](float32)"

CALIBRATION_CARRIERS = ["wind_onshore", "wind_offshore"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_espeni(espeni_file: str, year: int) -> pd.Series:
    """Return hourly ESPENI wind_total MW for *year* (UTC, tz-naive)."""
    esp = pd.read_csv(
        espeni_file,
        usecols=[_TIME_COL, _TOTAL_COL],
        dtype={_TOTAL_COL: float},
    )
    esp["time"] = pd.to_datetime(esp[_TIME_COL], utc=True).dt.tz_convert(None)
    esp = esp.set_index("time")[_TOTAL_COL].rename("wind_total").sort_index()
    esp = esp[esp.index.year == year]
    if esp.empty:
        raise ValueError(f"No ESPENI data for year {year}")
    esp = esp.resample("h").mean()  # half-hourly → hourly
    logger.info(
        f"ESPENI {year}: {len(esp)} hourly rows, "
        f"mean = {esp.mean():.1f} MW, annual sum = {esp.sum()/1000:.1f} TWh"
    )
    return esp


def _load_profile(profiles_dir: str, carrier: str, year: int) -> pd.DataFrame | None:
    """Return the raw ERA5 profile DataFrame (sites × time, MW) or None."""
    path = Path(profiles_dir) / f"{carrier}_{year}.csv"
    if not path.exists():
        logger.warning(f"Profile file not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.debug(f"Loaded {carrier} {year}: {df.shape[1]} sites, {len(df)} timesteps")
    return df


def _repd_capacity(generators_file: str, carrier: str) -> float | None:
    """Return total p_nom (MW) for *carrier* from the generators CSV."""
    p = Path(generators_file)
    if not p.exists():
        return None
    gen = pd.read_csv(p)
    mask = gen["carrier"] == carrier
    if not mask.any():
        return None
    return float(gen.loc[mask, "p_nom"].sum())


# ──────────────────────────────────────────────────────────────────────────────
# Core calibration
# ──────────────────────────────────────────────────────────────────────────────

def compute_factors(
    year: int,
    profiles_dir: str,
    generators_file: str | None,
    espeni_file: str,
    carriers: list[str] = None,
) -> pd.DataFrame:
    """
    Compute calibration factors for all *carriers* in *year*.

    Returns a DataFrame with columns: year, carrier, factor,
    espeni_mean_mw, model_mean_mw.
    """
    if carriers is None:
        carriers = CALIBRATION_CARRIERS

    espeni = _load_espeni(espeni_file, year)

    # Build scaled model total = Σ_carriers era5_total × (repd_cap / era5_cap)
    carrier_series = {}
    for carrier in carriers:
        profiles = _load_profile(profiles_dir, carrier, year)
        if profiles is None:
            continue

        era5_total = profiles.sum(axis=1)
        era5_cap = float(profiles.max().sum())

        repd_cap = None
        if generators_file:
            repd_cap = _repd_capacity(generators_file, carrier)

        if repd_cap is not None and era5_cap > 0:
            scale = repd_cap / era5_cap
            scaled = era5_total * scale
            logger.info(
                f"  {carrier}: ERA5 cap={era5_cap:.0f} MW, REPD cap={repd_cap:.0f} MW, "
                f"scale={scale:.3f}"
            )
        else:
            scaled = era5_total
            logger.warning(
                f"  {carrier}: no REPD capacity data, using raw ERA5 site total"
            )

        # Align profiles to hourly (ERA5 is already hourly, but guard anyway)
        if not isinstance(scaled.index, pd.DatetimeIndex):
            scaled.index = pd.to_datetime(scaled.index)
        scaled = scaled.resample("h").mean()
        carrier_series[carrier] = scaled

    if not carrier_series:
        logger.error(f"No profile data loaded for year {year}")
        return pd.DataFrame()

    # Combined model total aligned to ESPENI index
    combined_model = sum(
        s.reindex(espeni.index, method="nearest", tolerance=pd.Timedelta("30min"))
        for s in carrier_series.values()
    )
    combined_model_mean = float(combined_model.mean())
    espeni_mean = float(espeni.mean())

    if combined_model_mean <= 0:
        logger.error(f"Zero combined model output for {year}")
        return pd.DataFrame()

    combined_factor = espeni_mean / combined_model_mean
    logger.info(
        f"Year {year}: ESPENI mean={espeni_mean:.1f} MW, "
        f"model mean={combined_model_mean:.1f} MW, "
        f"combined factor={combined_factor:.4f}"
    )

    rows = []
    for carrier, series in carrier_series.items():
        model_mean = float(
            series.reindex(
                espeni.index, method="nearest", tolerance=pd.Timedelta("30min")
            ).mean()
        )
        rows.append(
            {
                "year": year,
                "carrier": carrier,
                "factor": round(combined_factor, 4),
                "espeni_mean_mw": round(espeni_mean, 1),
                "model_mean_mw": round(model_mean, 1),
            }
        )
        logger.info(
            f"  {carrier}: model={model_mean:.1f} MW, "
            f"factor={combined_factor:.4f} (combined ESPENI)"
        )

    return pd.DataFrame(rows)


def write_calibration(new_df: pd.DataFrame, output_file: str) -> None:
    """Merge new rows into the existing calibration CSV (upsert on year+carrier)."""
    out = Path(output_file)
    if out.exists():
        existing = pd.read_csv(out)
        # Drop any existing rows for the years we're updating
        existing = existing[~existing["year"].isin(new_df["year"].unique())]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["year", "carrier"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    logger.info(f"Written {len(new_df)} row(s) → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--year", type=int, nargs="+", required=True,
        help="Year(s) to calibrate (e.g. --year 2019 2020 2021)."
    )
    p.add_argument(
        "--profiles-dir", default="resources/renewable/profiles",
        help="Directory containing wind_{carrier}_{year}.csv files."
    )
    p.add_argument(
        "--generators-file", default=None,
        help=(
            "Path to *_generators_full.csv for the scenario/year "
            "(used to get REPD p_nom per carrier). May be omitted but "
            "factors will be less accurate without it."
        ),
    )
    p.add_argument(
        "--espeni-file", default="data/demand/espeni.csv",
        help="Path to the ESPENI CSV."
    )
    p.add_argument(
        "--output", default="resources/renewable/wind_calibration.csv",
        help="Output calibration CSV (upserted, not overwritten)."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    all_rows = []
    for year in args.year:
        logger.info(f"{'='*60}")
        logger.info(f"Calibrating year {year}")
        df = compute_factors(
            year=year,
            profiles_dir=args.profiles_dir,
            generators_file=args.generators_file,
            espeni_file=args.espeni_file,
        )
        if not df.empty:
            all_rows.append(df)

    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
        write_calibration(result, args.output)
    else:
        logger.error("No calibration factors computed — check inputs")


if __name__ == "__main__":
    main()
