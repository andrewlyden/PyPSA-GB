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
# Germany uses DE_LU (Germany-Luxembourg): NeuConnect lands at Fedderwarden,
# Lower Saxony, and was priced on the Dutch zone until 26 Sep 2026.
COUNTRY_ZONE_MAP: dict[str, str] = {
    "France": "FR",
    "Belgium": "BE",
    "Netherlands": "NL",
    "Germany": "DE_LU",
    "Norway": "NO_2",
    "Denmark": "DK_1",
    "Ireland": "IE_SEM",
}

# The current ENTSO-E API begins in 2015. For an earlier weather/market year,
# use a checked export from ENTSO-E's legacy archive rather than borrowing a
# different year's profile. See data/interconnectors/entsoe_legacy/README.md.
LEGACY_INPUT_DIR = project_root / "data" / "interconnectors" / "entsoe_legacy"
LEGACY_REQUIRED_COUNTRIES = tuple(COUNTRY_ZONE_MAP)

# Day-ahead prices from 2015 shipped with the model (user, 18 Nov 2026), so a run needs no ENTSO-E API token: the
# API's hourly EUR/MWh, before conversion, capping and gap filling (data/interconnectors/entsoe_day_ahead/README.md).
# A shipped year is always read from here; the API is used only for a year not shipped.
ARCHIVE_DIR = project_root / "data" / "interconnectors" / "entsoe_day_ahead"


def archive_path(year: int) -> Path:
    return ARCHIVE_DIR / f"day_ahead_prices_{year}_eur_per_mwh.csv"


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


def load_legacy_day_ahead_prices(
    year: int,
    input_path: str | Path | None,
    eur_to_gbp: float,
    price_floor: float,
    price_cap: float,
) -> pd.DataFrame:
    """Load a canonical ENTSO-E pre-2015 day-ahead-price export.

    The legacy Transparency archive is a manual download, rather than an API
    endpoint. Its data are normalised once into the documented canonical CSV;
    keeping it under ``data/`` makes the comparator reproducible and prevents a
    later run from replacing its 2013 profile with another year.
    """
    source = Path(input_path) if input_path else (
        LEGACY_INPUT_DIR / f"day_ahead_prices_{year}_eur_per_mwh.csv"
    )
    if not source.is_file():
        raise RuntimeError(
            f"ENTSO-E's API does not provide {year} day-ahead prices. "
            f"Add the verified legacy export at {source} (see "
            "data/interconnectors/entsoe_legacy/README.md); no substitute "
            "year will be used."
        )

    raw = pd.read_csv(source)
    if "datetime" not in raw.columns:
        raise ValueError(f"{source} must contain a 'datetime' column")
    missing_countries = [
        country for country in LEGACY_REQUIRED_COUNTRIES if country not in raw.columns
    ]
    if missing_countries:
        raise ValueError(
            f"{source} is missing required country columns: {missing_countries}"
        )

    timestamps = pd.to_datetime(raw.pop("datetime"), errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(f"{source} contains invalid datetime values")
    frame = raw.loc[:, LEGACY_REQUIRED_COUNTRIES].apply(pd.to_numeric, errors="coerce")
    frame.index = timestamps
    frame.index.name = "datetime"
    frame = frame.sort_index()

    expected = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC"
    )
    if not frame.index.equals(expected):
        raise ValueError(
            f"{source} must contain exactly the {len(expected)} UTC hourly timestamps "
            f"from {expected[0]} through {expected[-1]}."
        )
    if frame.isna().any().any():
        missing = int(frame.isna().sum().sum())
        raise ValueError(f"{source} contains {missing} missing price values")

    # Canonical legacy exports are in EUR/MWh; retain the model's established
    # conversion and clipping convention for its generated resource.
    return (frame * eur_to_gbp).clip(lower=price_floor, upper=price_cap)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_day_ahead_prices(
    year: int,
    output_path: str | Path,
    eur_to_gbp: float = 0.86,
    price_floor: float = -500.0,
    price_cap: float = 500.0,
    legacy_input_path: str | Path | None = None,
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

    if year < 2015:
        df = load_legacy_day_ahead_prices(
            year=year,
            input_path=legacy_input_path,
            eur_to_gbp=eur_to_gbp,
            price_floor=price_floor,
            price_cap=price_cap,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info(
            "Saved verified ENTSO-E legacy prices: %s hourly values x %s countries -> %s",
            len(df), len(df.columns), output_path,
        )
        return df

    if archive_path(year).is_file():
        logger.info(f"Reading the shipped ENTSO-E prices for {year}: {archive_path(year)}")
        raw = read_archived_eur_prices(year)
    else:
        raw = fetch_raw_eur_prices(year, logger)
    df = model_prices_from_eur(raw, eur_to_gbp, price_floor, price_cap, logger)

    # ---------- Save ----------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info(f"Saved {len(df)} hourly prices × {len(df.columns)} countries → {output_path}")

    return df


def fetch_raw_eur_prices(year: int, logger: logging.Logger) -> pd.DataFrame:
    """Hourly day-ahead prices in EUR/MWh from the ENTSO-E API, one column per country; gaps left as NaN."""
    token = _load_api_token(logger)
    if token is None:
        raise RuntimeError(
            f"No shipped ENTSO-E prices for {year} ({archive_path(year)}) and no ENTSO-E API token.  "
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
            all_prices[country] = prices.resample("1h").mean()  # ensure hourly
            logger.info(f"  {country}: {len(all_prices[country])} hours, mean EUR {prices.mean():.2f}/MWh")
        except Exception as exc:
            logger.warning(f"  {country}: FAILED — {exc}")
            # We'll fill this country with NaN; handled later

    if not all_prices:
        raise RuntimeError("No prices fetched for any country")
    df = pd.DataFrame(all_prices)
    df.index.name = "datetime"
    return df


def read_archived_eur_prices(year: int) -> pd.DataFrame:
    """The shipped hourly EUR/MWh prices of ``year`` (``archive_path``), as written by ``--archive``."""
    df = pd.read_csv(archive_path(year), index_col="datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def archive_year(year: int, logger: logging.Logger) -> Path:
    """Fetch ``year`` from the API in EUR/MWh and write the shipped file (``archive_path``)."""
    raw = fetch_raw_eur_prices(year, logger)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(archive_path(year), float_format="%.4f")
    logger.info(f"Archived {len(raw)} hours × {len(raw.columns)} countries → {archive_path(year)}")
    return archive_path(year)


def model_prices_from_eur(
    raw: pd.DataFrame,
    eur_to_gbp: float,
    price_floor: float,
    price_cap: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """EUR/MWh to the model's GBP/MWh: convert, cap, then fill gaps (the order the API fetch has always used)."""
    df = (raw * eur_to_gbp).clip(lower=price_floor, upper=price_cap)
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
        price_floor=_pricing_cfg.get("price_floor", -500.0),
        price_cap=_pricing_cfg.get("price_cap", 500.0),
        legacy_input_path=_pricing_cfg.get("legacy_input_path"),
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
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Fetch the year from the API in EUR/MWh and write the shipped file in data/interconnectors/entsoe_day_ahead/",
    )
    parser.add_argument(
        "--legacy-input",
        type=str,
        default=None,
        help="Canonical ENTSO-E legacy CSV for pre-2015 years",
    )
    args = parser.parse_args()

    logger = setup_logging("fetch_entsoe_prices")
    if args.archive:
        archive_year(args.year, logger)
        return
    out = args.output or f"resources/interconnectors/entsoe_day_ahead_prices_{args.year}.csv"
    fetch_day_ahead_prices(
        year=args.year,
        output_path=out,
        eur_to_gbp=args.eur_to_gbp,
        legacy_input_path=args.legacy_input,
        logger=logger,
    )


if __name__ == "__main__" and "snakemake" not in dir():
    main()
