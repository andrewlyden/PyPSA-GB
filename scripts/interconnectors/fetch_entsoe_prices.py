#!/usr/bin/env python3
"""
Fetch ENTSO-E Day-Ahead Electricity Prices
============================================

Downloads hourly day-ahead auction prices from the ENTSO-E Transparency
Platform for countries connected to GB via interconnectors.

Outputs a CSV with datetime index and one column per country (£/MWh).

The ENTSO-E API token is loaded from (in priority order):
  1. .env file in the project root  (ENTSOE_API_TOKEN=…)
  2. Environment variable ENTSOE_API_TOKEN

Usage:
  Standalone:
      python scripts/interconnectors/fetch_entsoe_prices.py --year 2021

  Via Snakemake:
      Called by the fetch_entsoe_day_ahead_prices rule in rules/interconnectors.smk

Author: PyPSA-GB Team
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Add project root for sibling imports
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.utilities.logging_config import setup_logging
except ImportError:
    def setup_logging(name: str) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(name)

# ---------------------------------------------------------------------------
# ENTSO-E bidding-zone mapping
# ---------------------------------------------------------------------------
# Maps PyPSA-GB interconnector country names → ENTSO-E bidding zones.
# Norway uses NO_2 (southern bidding zone for North Sea Link).
# Ireland uses the SEM (Single Electricity Market) zone.
COUNTRY_ZONE_MAP: dict[str, str] = {
    "France": "FR",
    "Belgium": "BE",
    "Netherlands": "NL",
    "Norway": "NO_2",
    "Denmark": "DK_1",
    "Ireland": "IE_SEM",
}


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------
def _load_api_token(logger: logging.Logger) -> str | None:
    """Return the ENTSO-E API token or *None* if unavailable.

    Priority:
      1.  .env file in the project root
      2.  ENTSOE_API_TOKEN environment variable
    """
    # Try .env file first (lightweight, no third-party dependency required)
    env_path = project_root / ".env"
    if env_path.is_file():
        with open(env_path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key == "ENTSOE_API_TOKEN" and value and value != "your-token-here":
                    logger.info("Loaded ENTSO-E API token from .env file")
                    return value

    # Fall back to environment variable
    token = os.environ.get("ENTSOE_API_TOKEN")
    if token:
        logger.info("Loaded ENTSO-E API token from environment variable")
        return token

    return None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_day_ahead_prices(
    year: int,
    output_path: str | Path,
    eur_to_gbp: float = 0.86,
    price_floor: float = 0.0,
    price_cap: float = 500.0,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fetch hourly day-ahead prices for all connected countries for *year*.

    Parameters
    ----------
    year : int
        Calendar year to fetch (e.g. 2021).
    output_path : str | Path
        Where to write the CSV result.
    eur_to_gbp : float
        EUR → GBP conversion rate.
    price_floor / price_cap : float
        Clip prices to [floor, cap] £/MWh after conversion.
    logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame
        Hourly prices in £/MWh (datetime index, one column per country).
        Saved to *output_path* as CSV.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    token = _load_api_token(logger)
    if token is None:
        raise RuntimeError(
            "ENTSO-E API token not found.  "
            "Either create a .env file with ENTSOE_API_TOKEN=<token> or "
            "set the ENTSOE_API_TOKEN environment variable.  "
            "Register free at https://transparency.entsoe.eu/"
        )

    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=token)
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

    all_prices: dict[str, pd.Series] = {}

    for country, zone in COUNTRY_ZONE_MAP.items():
        logger.info(f"Fetching {country} ({zone}) day-ahead prices for {year} …")
        try:
            prices = client.query_day_ahead_prices(zone, start=start, end=end)
            # entsoe-py returns a Series with a DatetimeIndex (UTC)
            prices = prices.resample("1h").mean()  # ensure hourly
            prices = prices * eur_to_gbp  # EUR → GBP
            prices = prices.clip(lower=price_floor, upper=price_cap)
            all_prices[country] = prices
            logger.info(
                f"  {country}: {len(prices)} hours, "
                f"mean £{prices.mean():.2f}/MWh, "
                f"range £{prices.min():.2f}–£{prices.max():.2f}/MWh"
            )
        except Exception as exc:
            logger.warning(f"  {country}: FAILED — {exc}")
            # We'll fill this country with NaN; handled later

    if not all_prices:
        raise RuntimeError("No prices fetched for any country")

    df = pd.DataFrame(all_prices)
    df.index.name = "datetime"

    # ---------- Gap handling ----------
    # Interpolate short gaps (≤6 h), then forward-fill, then back-fill
    df = df.interpolate(method="linear", limit=6, limit_direction="both")
    df = df.ffill(limit=24).bfill(limit=24)

    # Any columns still entirely NaN → drop (country was unavailable)
    dropped = [c for c in df.columns if df[c].isna().all()]
    if dropped:
        logger.warning(f"Dropping countries with no data: {dropped}")
        df = df.drop(columns=dropped)

    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        logger.warning(
            f"{remaining_nans} NaN values remain after gap-filling — "
            "filling with per-country mean"
        )
        df = df.fillna(df.mean())

    # ---------- Save ----------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info(f"Saved {len(df)} hourly prices × {len(df.columns)} countries → {output_path}")

    return df


# ---------------------------------------------------------------------------
# Snakemake entry point
# ---------------------------------------------------------------------------
if "snakemake" in dir():
    _logger = setup_logging("fetch_entsoe_prices")
    _year = int(snakemake.params.year)
    _output = snakemake.output[0]
    _pricing_cfg = snakemake.params.get("pricing_config", {})

    fetch_day_ahead_prices(
        year=_year,
        output_path=_output,
        eur_to_gbp=_pricing_cfg.get("eur_to_gbp", 0.86),
        price_floor=_pricing_cfg.get("price_floor", 0.0),
        price_cap=_pricing_cfg.get("price_cap", 500.0),
        logger=_logger,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch ENTSO-E day-ahead prices for interconnector countries"
    )
    parser.add_argument("--year", type=int, required=True, help="Calendar year")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: resources/interconnectors/entsoe_day_ahead_prices_<year>.csv)",
    )
    parser.add_argument(
        "--eur-to-gbp", type=float, default=0.86, help="EUR→GBP rate (default 0.86)"
    )
    args = parser.parse_args()

    logger = setup_logging("fetch_entsoe_prices")
    out = args.output or f"resources/interconnectors/entsoe_day_ahead_prices_{args.year}.csv"
    fetch_day_ahead_prices(
        year=args.year,
        output_path=out,
        eur_to_gbp=args.eur_to_gbp,
        logger=logger,
    )


if __name__ == "__main__":
    main()
