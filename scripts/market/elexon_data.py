"""
Fetch ELEXON BMRS bid/offer data for historical market scenarios.

Downloads settlement-period-level bid and offer data from the ELEXON Insights
API (v1), aggregates to hourly resolution, and writes CSV files for use by
the balancing mechanism solve step.

Based on the approach in GBPower (build_base.py + _elexon_helpers.py).

Usage:
    Called by Snakemake rule `retrieve_elexon_market_data` in rules/market.smk.
    Can also be run standalone for testing.

Outputs:
    resources/market/{scenario}/elexon/elexon_offers.csv
    resources/market/{scenario}/elexon/elexon_bids.csv

Each CSV has columns: settlement_period (datetime index), then one column per
BMU ID with the volume-weighted average price (£/MWh) for that period.
"""

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("elexon_data")

# ELEXON Insights API base URL
ELEXON_API_BASE = "https://data.elexon.co.uk/bmrs/api/v1"

# Rate limiting: max requests per second
ELEXON_RATE_LIMIT = 2.0  # conservative


def _check_requests_available():
    if requests is None:
        raise ImportError(
            "The 'requests' package is required for ELEXON data fetching. "
            "Install with: pip install requests"
        )


def fetch_bod_data(
    date: str,
    settlement_period: Optional[int] = None,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch Bid-Offer Data (BOD) from ELEXON API for a given date.

    Parameters
    ----------
    date : str
        Settlement date in YYYY-MM-DD format.
    settlement_period : int, optional
        If given, fetch only this settlement period (1-50). Otherwise fetch all.
    logger : Logger

    Returns
    -------
    DataFrame with columns: bmu_id, settlement_period, offer_price, bid_price,
        offer_volume, bid_volume, pair_id, level_from, level_to
    """
    _check_requests_available()

    url = f"{ELEXON_API_BASE}/datasets/BOD/stream"
    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    params = {
        "from": day_start.strftime("%Y-%m-%dT00:00Z"),
        "to": day_end.strftime("%Y-%m-%dT00:00Z"),
        "format": "json",
    }

    resp = requests.get(url, params=params, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        all_records = data.get("data", [])
    else:
        all_records = data

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    if settlement_period is not None and "settlementPeriod" in df.columns:
        df = df[df["settlementPeriod"] == settlement_period].copy()

    # Standardise column names
    col_map = {
        "bmUnit": "bmu_id",
        "settlementPeriod": "settlement_period",
        "offerPrice": "offer_price",
        "bidPrice": "bid_price",
        "offer": "offer_price",
        "bid": "bid_price",
        "offerLevel": "offer_volume",
        "bidLevel": "bid_volume",
        "pairId": "pair_id",
        "levelFrom": "level_from",
        "levelTo": "level_to",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def aggregate_bod_to_hourly(
    bod_df: pd.DataFrame,
    date: str,
    logger: logging.Logger = logger,
) -> tuple:
    """
    Aggregate BOD data to hourly offer/bid price per BMU.

    For each BMU and settlement period, takes the volume-weighted average of
    offer prices (positive pair IDs) and bid prices (negative pair IDs).
    Then converts from half-hourly settlement periods to hourly timestamps.

    Parameters
    ----------
    bod_df : DataFrame
        Raw BOD data from fetch_bod_data().
    date : str
        Settlement date (YYYY-MM-DD) for timestamp calculation.

    Returns
    -------
    (offers_df, bids_df) : tuple of DataFrames
        Each indexed by datetime (hourly), columns = BMU IDs, values = £/MWh.
    """
    if bod_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "offer_volume" not in bod_df.columns and "level_to" in bod_df.columns:
        bod_df["offer_volume"] = pd.to_numeric(
            bod_df["level_to"], errors="coerce"
        ).abs()
    if "bid_volume" not in bod_df.columns and "level_from" in bod_df.columns:
        bod_df["bid_volume"] = pd.to_numeric(
            bod_df["level_from"], errors="coerce"
        ).abs()

    offers = bod_df[bod_df["offer_price"].notna()].copy()
    bids = bod_df[bod_df["bid_price"].notna()].copy()

    def _weighted_avg(group, price_col, vol_col):
        if vol_col not in group.columns:
            return group[price_col].mean()
        vols = pd.to_numeric(group[vol_col], errors="coerce").abs()
        total = vols.sum()
        if vols.isna().all() or total == 0:
            return group[price_col].mean()
        return (group[price_col] * vols).sum() / total

    # Volume-weighted average offer price per (bmu, sp)
    offer_prices = (
        offers.groupby(["bmu_id", "settlement_period"])
        .apply(lambda g: _weighted_avg(g, "offer_price", "offer_volume"), include_groups=False)
        .unstack(level="bmu_id")
    )

    bid_prices = (
        bids.groupby(["bmu_id", "settlement_period"])
        .apply(lambda g: _weighted_avg(g, "bid_price", "bid_volume"), include_groups=False)
        .unstack(level="bmu_id")
    )

    # Convert settlement period index → hourly datetime
    base = pd.Timestamp(date)
    for df in [offer_prices, bid_prices]:
        if df.empty:
            continue
        # SP 1 = 00:00-00:30, SP 2 = 00:30-01:00, etc.
        # Average pairs of SPs into hourly values
        sp_to_hour = {sp: base + timedelta(minutes=30 * (sp - 1)) for sp in df.index}
        df.index = df.index.map(sp_to_hour)
        df.index.name = "datetime"

    # Resample to hourly (mean of two half-hours)
    if not offer_prices.empty:
        offer_prices = offer_prices.resample("h").mean()
    if not bid_prices.empty:
        bid_prices = bid_prices.resample("h").mean()

    return offer_prices, bid_prices


def retrieve_elexon_market_data(
    start_date: str,
    end_date: str,
    output_dir: str,
    logger: logging.Logger = logger,
) -> dict:
    """
    Fetch ELEXON bid/offer data for a date range and save to CSV.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    output_dir : str
        Directory to write output CSVs.
    logger : Logger

    Returns
    -------
    dict with keys 'offers_file', 'bids_file' pointing to output paths.
    """
    _check_requests_available()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq="D")

    logger.info(f"Fetching ELEXON BOD data: {start_date} to {end_date} ({len(dates)} days)")

    all_offers = []
    all_bids = []

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(f"  Day {i + 1}/{len(dates)}: {date_str}")

        try:
            bod = fetch_bod_data(date_str, logger=logger)
            if bod.empty:
                logger.warning(f"  No BOD data for {date_str}")
                continue

            offers, bids = aggregate_bod_to_hourly(bod, date_str, logger=logger)
            if not offers.empty:
                all_offers.append(offers)
            if not bids.empty:
                all_bids.append(bids)

        except Exception as e:
            logger.error(f"  Failed to fetch {date_str}: {e}")
            continue

        time.sleep(1.0 / ELEXON_RATE_LIMIT)

    # Combine all days
    offers_path = out_path / "elexon_offers.csv"
    bids_path = out_path / "elexon_bids.csv"

    if all_offers:
        combined_offers = pd.concat(all_offers, axis=0).sort_index()
        combined_offers.to_csv(offers_path)
        logger.info(f"Saved offers: {combined_offers.shape} to {offers_path}")
    else:
        pd.DataFrame().to_csv(offers_path)
        logger.warning("No offer data collected")

    if all_bids:
        combined_bids = pd.concat(all_bids, axis=0).sort_index()
        combined_bids.to_csv(bids_path)
        logger.info(f"Saved bids: {combined_bids.shape} to {bids_path}")
    else:
        pd.DataFrame().to_csv(bids_path)
        logger.warning("No bid data collected")

    return {"offers_file": str(offers_path), "bids_file": str(bids_path)}


# ══════════════════════════════════════════════════════════════════════════════
# ACTUAL GENERATION (B1610) + SYSTEM PRICE DATA
# ══════════════════════════════════════════════════════════════════════════════
# Used by per-generator marginal cost calibration.
#
# B1610: "Actual Generation Output Per Generation Unit" — actual MW output
#        per BMU per settlement period. Requires both settlementDate and
#        settlementPeriod parameters.
#
# System prices: /balancing/settlement/system-prices/{date} — returns all
#        48 settlement periods at once (SBP, SSP).

def fetch_generation_per_unit(
    date: str,
    settlement_period: int,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch actual generation output per BMU for one settlement period (B1610).

    Parameters
    ----------
    date : str
        Settlement date in YYYY-MM-DD format.
    settlement_period : int
        Settlement period (1-48).
    logger : Logger

    Returns
    -------
    DataFrame with columns: bmu_id, quantity (MW), settlement_period
    """
    _check_requests_available()

    url = f"{ELEXON_API_BASE}/datasets/B1610"
    params = {
        "settlementDate": date,
        "settlementPeriod": settlement_period,
        "format": "json",
    }

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Standardise and filter to transmission BMUs
    if "bmUnit" in df.columns:
        df = df.rename(columns={"bmUnit": "bmu_id", "quantity": "generation_mw"})
        df = df[df["bmu_id"].str.startswith("T_", na=False)].copy()
        df["settlement_period"] = settlement_period

    return df[["bmu_id", "generation_mw", "settlement_period"]] if not df.empty else pd.DataFrame()


def fetch_system_prices(
    date: str,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch system buy/sell prices from ELEXON for a given date.

    Returns all 48 settlement periods at once.

    Parameters
    ----------
    date : str
        Settlement date in YYYY-MM-DD format.

    Returns
    -------
    DataFrame with columns: settlement_period, system_buy_price, system_sell_price
    """
    _check_requests_available()

    url = f"{ELEXON_API_BASE}/balancing/settlement/system-prices/{date}"
    params = {"format": "json"}

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    if not data:
        logger.warning(f"No system prices for {date}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    result = pd.DataFrame({
        "settlement_period": df["settlementPeriod"].astype(int),
        "system_buy_price": df["systemBuyPrice"].astype(float),
    })
    logger.debug(f"System prices for {date}: {len(result)} periods, "
                 f"mean £{result['system_buy_price'].mean():.1f}/MWh")
    return result


def retrieve_pn_and_prices(
    start_date: str,
    end_date: str,
    output_dir: str,
    logger: logging.Logger = logger,
) -> dict:
    """
    Fetch actual generation (B1610) and system prices for a date range.

    Queries B1610 per settlement period (48 calls/day) and system prices
    per day (1 call/day). Builds a pivoted matrix of BMU generation and
    a system price time series.

    Outputs:
        {output_dir}/pn_data.csv       - BMU generation (datetime × bmu_id, MW)
        {output_dir}/system_prices.csv  - System price per datetime (£/MWh)

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    output_dir : str
        Directory to write output CSVs.

    Returns
    -------
    dict with keys 'pn_file', 'prices_file' pointing to output paths.
    """
    _check_requests_available()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    total_queries = n_days * 49  # 48 B1610 + 1 system-prices per day

    logger.info(f"Fetching B1610 generation + system prices: "
                f"{start_date} to {end_date} ({n_days} days, ~{total_queries} API calls)")

    all_gen_rows = []  # List of (datetime, bmu_id, mw) tuples
    all_prices = []    # List of (datetime, price) tuples

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        base = pd.Timestamp(date_str)
        logger.info(f"  Day {i + 1}/{n_days}: {date_str}")

        # 1. Fetch system prices for the whole day (1 API call)
        try:
            prices_df = fetch_system_prices(date_str, logger=logger)
            if not prices_df.empty:
                for _, row in prices_df.iterrows():
                    sp = int(row["settlement_period"])
                    dt = base + timedelta(minutes=30 * (sp - 1))
                    all_prices.append({"datetime": dt, "system_buy_price": row["system_buy_price"]})
        except Exception as e:
            logger.error(f"  Failed to fetch prices for {date_str}: {e}")

        time.sleep(1.0 / ELEXON_RATE_LIMIT)

        # 2. Fetch B1610 for each settlement period (48 API calls)
        for sp in range(1, 49):
            try:
                gen_df = fetch_generation_per_unit(date_str, sp, logger=logger)
                if not gen_df.empty:
                    dt = base + timedelta(minutes=30 * (sp - 1))
                    for _, row in gen_df.iterrows():
                        all_gen_rows.append({
                            "datetime": dt,
                            "bmu_id": row["bmu_id"],
                            "generation_mw": row["generation_mw"],
                        })
            except Exception as e:
                logger.warning(f"  Failed B1610 SP {sp}: {e}")

            time.sleep(1.0 / ELEXON_RATE_LIMIT)

        logger.info(f"    Fetched {len([r for r in all_gen_rows if r['datetime'] >= base])} "
                     f"generation records for {date_str}")

    # Build and save generation matrix (datetime × bmu_id)
    pn_path = out_path / "pn_data.csv"
    if all_gen_rows:
        gen_long = pd.DataFrame(all_gen_rows)
        gen_pivot = gen_long.pivot_table(
            index="datetime", columns="bmu_id", values="generation_mw",
            aggfunc="max"  # Take max if multiple entries per (dt, bmu)
        ).fillna(0.0)
        gen_pivot.to_csv(pn_path)
        logger.info(f"Saved generation data: {gen_pivot.shape[0]} periods × "
                     f"{gen_pivot.shape[1]} BMUs → {pn_path}")
    else:
        pd.DataFrame().to_csv(pn_path)
        logger.warning("No generation data collected")

    # Build and save price series
    prices_path = out_path / "system_prices.csv"
    if all_prices:
        prices_series = pd.DataFrame(all_prices).set_index("datetime")["system_buy_price"]
        prices_series.to_csv(prices_path)
        logger.info(f"Saved system prices: {len(prices_series)} periods → {prices_path}")
    else:
        pd.Series(dtype=float, name="system_buy_price").to_csv(prices_path)
        logger.warning("No price data collected")

    return {"pn_file": str(pn_path), "prices_file": str(prices_path)}


# ══════════════════════════════════════════════════════════════════════════════
# WHOLESALE MARKET CALIBRATION DATA
# ══════════════════════════════════════════════════════════════════════════════
# MID (Market Index Data) — half-hourly wholesale market index price from
# N2EX/EPEX. Available in bulk via the /datasets/MID/stream endpoint.
#
# For per-generator MC calibration we need:
#   1. Year-long MID prices (fast bulk download via stream)
#   2. B1610 generation per BMU (sampled at key SPs to keep API calls low)
# ══════════════════════════════════════════════════════════════════════════════


def fetch_mid_prices_bulk(
    start_date: str,
    end_date: str,
    data_provider: str = "APXMIDP",
    logger: logging.Logger = logger,
) -> pd.Series:
    """
    Fetch Market Index Data (MID) wholesale prices in bulk via stream endpoint.

    The MID stream endpoint supports large date ranges and returns data very
    quickly (~1 second per 3-month chunk). Uses quarterly chunks to stay
    within API limits.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    data_provider : str
        MID data provider to filter on. Default 'APXMIDP' (N2EX market index).
    logger : Logger

    Returns
    -------
    Series indexed by datetime with wholesale price (£/MWh).
    """
    _check_requests_available()

    url = f"{ELEXON_API_BASE}/datasets/MID/stream"
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    # Fetch in quarterly chunks
    all_records = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(months=3), end)
        params = {
            "from": chunk_start.strftime("%Y-%m-%dT00:00Z"),
            "to": chunk_end.strftime("%Y-%m-%dT00:00Z"),
            "format": "json",
        }

        logger.info(f"  Fetching MID {chunk_start.date()} to {chunk_end.date()}...")
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            all_records.extend(data)
        elif isinstance(data, dict):
            all_records.extend(data.get("data", []))

        chunk_start = chunk_end
        time.sleep(0.5)

    if not all_records:
        logger.warning("No MID data returned")
        return pd.Series(dtype=float, name="mid_price")

    df = pd.DataFrame(all_records)

    # Filter to requested data provider
    if data_provider and "dataProvider" in df.columns:
        df = df[df["dataProvider"] == data_provider].copy()

    # Build datetime from settlementDate + settlementPeriod
    df["datetime"] = pd.to_datetime(df["settlementDate"]) + \
        pd.to_timedelta((df["settlementPeriod"].astype(int) - 1) * 30, unit="min")
    df = df.set_index("datetime").sort_index()

    prices = df["price"].astype(float)
    prices.name = "mid_price"

    # Deduplicate timestamps that can arise from BST→GMT clock-change days
    # (ELEXON emits 50 settlement periods on a 25-hour day; periods 49/50 map
    # to the same wall-clock hours as SP1/SP2 of the following settlement day)
    if prices.index.duplicated().any():
        n_dups = prices.index.duplicated().sum()
        logger.debug(f"Deduplicating {n_dups} duplicate MID timestamps (clock change)")
        prices = prices.groupby(level=0).mean()

    logger.info(f"Fetched MID prices: {len(prices)} periods, "
                f"£{prices.min():.1f} to £{prices.max():.1f}/MWh, "
                f"mean £{prices.mean():.1f}/MWh")
    return prices


def fetch_b1610_bulk(
    start_date: str,
    end_date: str,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch B1610 actual generation per BMU in bulk via stream endpoint.

    Uses /datasets/B1610/stream for fast bulk download in monthly chunks,
    replacing the per-SP fetch approach (~730 individual API calls → ~12 bulk
    calls for a full year). Returns the same pivoted format as the sampled
    approach but with full settlement period coverage.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    logger : Logger

    Returns
    -------
    DataFrame with datetime index, BMU ID columns, MW values.
        Empty DataFrame if the stream endpoint is unavailable.
    """
    _check_requests_available()

    url = f"{ELEXON_API_BASE}/datasets/B1610/stream"
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    all_records = []
    chunk_start = start
    chunk_num = 0
    fetch_start = time.time()

    while chunk_start < end:
        # Monthly chunks to stay within API response size limits
        chunk_end = min(chunk_start + pd.DateOffset(months=1), end)
        chunk_num += 1
        params = {
            "from": chunk_start.strftime("%Y-%m-%dT00:00Z"),
            "to": chunk_end.strftime("%Y-%m-%dT00:00Z"),
            "format": "json",
        }

        elapsed = time.time() - fetch_start
        logger.info(f"  Fetching B1610 chunk {chunk_num}: "
                     f"{chunk_start.date()} to {chunk_end.date()} "
                     f"({elapsed:.0f}s elapsed)")

        try:
            resp = requests.get(url, params=params, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                all_records.extend(data)
            elif isinstance(data, dict):
                all_records.extend(data.get("data", []))
        except Exception as e:
            logger.warning(f"  B1610 stream chunk failed: {e}")
            # If stream endpoint is not available, return empty to signal fallback
            if chunk_num == 1:
                logger.warning("  B1610 stream endpoint unavailable — will fall back to per-SP fetch")
                return pd.DataFrame()

        chunk_start = chunk_end
        time.sleep(0.5)

    if not all_records:
        logger.warning("No B1610 data returned from stream")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Standardise column names
    col_map = {
        "bmUnit": "bmu_id",
        "settlementDate": "settlement_date",
        "settlementPeriod": "settlement_period",
        "quantity": "generation_mw",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Filter to transmission BMUs only
    if "bmu_id" in df.columns:
        df = df[df["bmu_id"].str.startswith("T_", na=False)].copy()

    if df.empty:
        logger.warning("No transmission BMUs in B1610 stream data")
        return pd.DataFrame()

    # Build datetime index
    df["datetime"] = pd.to_datetime(df["settlement_date"]) + \
        pd.to_timedelta((df["settlement_period"].astype(int) - 1) * 30, unit="min")

    # Pivot to matrix form (datetime × bmu_id)
    gen_pivot = df.pivot_table(
        index="datetime", columns="bmu_id", values="generation_mw",
        aggfunc="max"
    ).fillna(0.0).sort_index()

    elapsed = time.time() - fetch_start
    logger.info(f"  B1610 stream complete: {gen_pivot.shape[0]} periods × "
                f"{gen_pivot.shape[1]} BMUs in {elapsed:.0f}s")

    return gen_pivot


def retrieve_wholesale_calibration_data(
    start_date: str,
    end_date: str,
    output_dir: str,
    sample_sps: list = None,
    use_stream: bool = True,
    logger: logging.Logger = logger,
) -> dict:
    """
    Fetch year-long wholesale market data for per-generator MC calibration.

    Downloads:
      1. MID wholesale prices via bulk stream (fast, full resolution)
      2. B1610 actual generation per BMU — prefers the /stream bulk endpoint
         (~12 monthly API calls). Falls back to per-SP sampling if stream
         is unavailable (730 API calls for 2 SPs/day over a year).

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    output_dir : str
        Directory to write output CSVs.
    sample_sps : list, optional
        Settlement periods to sample B1610 at (fallback only).
        Default [20, 38] (10am, 7pm).
    use_stream : bool
        Try B1610 stream endpoint first (default True). Set False to force
        per-SP fetch.
    logger : Logger

    Returns
    -------
    dict with keys 'pn_file', 'prices_file' pointing to output CSVs.
    """
    _check_requests_available()

    if sample_sps is None:
        sample_sps = [20, 38]  # ~10:00 morning peak, ~19:00 evening peak

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    logger.info("=" * 70)
    logger.info("WHOLESALE MARKET CALIBRATION DATA FETCH")
    logger.info("=" * 70)
    logger.info(f"Period: {start_date} to {end_date} ({n_days} days)")

    # ── 1. Fetch MID wholesale prices (bulk, fast) ───────────────────
    logger.info("\n── Stage 1: Market Index Prices (MID) ──")
    mid_prices = fetch_mid_prices_bulk(start_date, end_date, logger=logger)

    prices_path = out_path / "mid_prices.csv"
    mid_prices.to_csv(prices_path)
    logger.info(f"Saved MID prices: {len(mid_prices)} periods → {prices_path}")

    # ── 2. Fetch B1610 generation data ───────────────────────────────
    pn_path = out_path / "pn_data.csv"
    gen_pivot = pd.DataFrame()

    # Try bulk stream first (fast: ~12 API calls vs ~730)
    if use_stream:
        logger.info("\n── Stage 2: B1610 Generation (bulk stream) ──")
        gen_pivot = fetch_b1610_bulk(start_date, end_date, logger=logger)

    # Fall back to per-SP sampling if stream returned empty
    if gen_pivot.empty:
        n_queries = n_days * len(sample_sps)
        logger.info(f"\n── Stage 2: B1610 Generation (per-SP fallback, "
                     f"{n_queries} API calls, ~{n_queries / ELEXON_RATE_LIMIT / 60:.0f} min) ──")
        all_gen_rows = []
        failed = 0
        fetch_start = time.time()

        for i, date in enumerate(dates):
            date_str = date.strftime("%Y-%m-%d")
            base = pd.Timestamp(date_str)

            if (i + 1) % 30 == 0 or i == 0:
                elapsed = time.time() - fetch_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (n_days - i - 1) / rate if rate > 0 else 0
                logger.info(f"  Day {i + 1}/{n_days}: {date_str} "
                            f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                            f"{len(all_gen_rows)} records)")

            for sp in sample_sps:
                try:
                    gen_df = fetch_generation_per_unit(date_str, sp, logger=logger)
                    if not gen_df.empty:
                        dt = base + timedelta(minutes=30 * (sp - 1))
                        for _, row in gen_df.iterrows():
                            all_gen_rows.append({
                                "datetime": dt,
                                "bmu_id": row["bmu_id"],
                                "generation_mw": row["generation_mw"],
                            })
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        logger.warning(f"  B1610 {date_str} SP{sp} failed: {e}")
                    elif failed == 6:
                        logger.warning("  (suppressing further B1610 warnings)")

                time.sleep(1.0 / ELEXON_RATE_LIMIT)

        if failed:
            logger.info(f"  B1610 fetch complete: {failed} failures out of {n_queries}")

        if all_gen_rows:
            gen_long = pd.DataFrame(all_gen_rows)
            gen_pivot = gen_long.pivot_table(
                index="datetime", columns="bmu_id", values="generation_mw",
                aggfunc="max"
            ).fillna(0.0)

    # Save generation matrix
    if not gen_pivot.empty:
        gen_pivot.to_csv(pn_path)
        logger.info(f"Saved generation data: {gen_pivot.shape[0]} periods × "
                     f"{gen_pivot.shape[1]} BMUs → {pn_path}")
    else:
        pd.DataFrame().to_csv(pn_path)
        logger.warning("No generation data collected")

    logger.info("=" * 70)
    return {"pn_file": str(pn_path), "prices_file": str(prices_path)}


# ══════════════════════════════════════════════════════════════════════════════
# BM VALIDATION DATA — BOALF & SYSTEM PRICES
# ══════════════════════════════════════════════════════════════════════════════
# Three datasets for validating the balancing mechanism solve:
#   1. BOALF  — actual BM acceptances (increase/decrease volumes by BMU)
#   2. System prices (SBP/SSP) — half-hourly imbalance cash-out prices
#   3. Full B1610 — actual generation at all 48 SPs (not sampled)
# ══════════════════════════════════════════════════════════════════════════════


def fetch_boalf_bulk(
    start_date: str,
    end_date: str,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch Bid-Offer Acceptance Level Flagged (BOALF) data via stream endpoint.

    BOALF records every BM acceptance issued by NESO — the actual increase and
    decrease volumes instructed in the balancing mechanism. This is the primary
    dataset for validating BM redispatch volumes.

    Uses /datasets/BOALF/stream for efficient bulk download in weekly chunks.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    logger : Logger

    Returns
    -------
    DataFrame with columns: datetime, bmu_id, acceptance_number,
        acceptance_level (MW), bid_offer_flag, so_flag, stor_flag, rr_flag,
        level_from, level_to
        Empty DataFrame if endpoint is unavailable.
    """
    _check_requests_available()

    url = f"{ELEXON_API_BASE}/datasets/BOALF/stream"
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    all_records = []
    chunk_start = start
    chunk_num = 0
    fetch_start = time.time()

    while chunk_start < end:
        # Weekly chunks to stay within API response limits
        chunk_end = min(chunk_start + pd.Timedelta(days=7), end)
        chunk_num += 1
        params = {
            "from": chunk_start.strftime("%Y-%m-%dT00:00Z"),
            "to": chunk_end.strftime("%Y-%m-%dT00:00Z"),
            "format": "json",
        }

        elapsed = time.time() - fetch_start
        logger.info(f"  Fetching BOALF chunk {chunk_num}: "
                     f"{chunk_start.date()} to {chunk_end.date()} "
                     f"({elapsed:.0f}s elapsed)")

        try:
            resp = requests.get(url, params=params, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                all_records.extend(data)
            elif isinstance(data, dict):
                all_records.extend(data.get("data", []))
        except Exception as e:
            logger.warning(f"  BOALF stream chunk failed: {e}")
            if chunk_num == 1:
                logger.warning("  BOALF stream endpoint unavailable")
                return pd.DataFrame()

        chunk_start = chunk_end
        time.sleep(0.5)

    if not all_records:
        logger.warning("No BOALF data returned from stream")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Standardise column names
    col_map = {
        "bmUnit": "bmu_id",
        "settlementDate": "settlement_date",
        "settlementPeriod": "settlement_period",
        "acceptanceNumber": "acceptance_number",
        "acceptanceLevel": "acceptance_level",
        "bidOfferAcceptanceNumber": "bo_acceptance_number",
        "soFlag": "so_flag",
        "storFlag": "stor_flag",
        "storProviderFlag": "stor_provider_flag",
        "rrFlag": "rr_flag",
        "levelFrom": "level_from",
        "levelTo": "level_to",
        "deemedBidOfferFlag": "bid_offer_flag",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "stor_provider_flag" in df.columns and "stor_flag" not in df.columns:
        df["stor_flag"] = df["stor_provider_flag"]

    for flag_col in ["so_flag", "stor_flag", "rr_flag"]:
        if flag_col not in df.columns:
            df[flag_col] = False

    # Build datetime index
    if "settlement_date" in df.columns and "settlement_period" in df.columns:
        df["datetime"] = pd.to_datetime(df["settlement_date"]) + \
            pd.to_timedelta((df["settlement_period"].astype(int) - 1) * 30, unit="min")

    # Filter to transmission BMUs
    if "bmu_id" in df.columns:
        df = df[df["bmu_id"].str.startswith("T_", na=False)].copy()

    elapsed = time.time() - fetch_start
    logger.info(f"  BOALF stream complete: {len(df)} records in {elapsed:.0f}s")

    return df


def fetch_disbsad_range(
    start_date: str,
    end_date: str,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch Disaggregated Balancing Services Adjustment Data for a date range.

    DISBSAD represents non-BM balancing-service adjustments received from NESO,
    including cost and volume per settlement period. This is useful as a sidecar
    diagnostic to quantify balancing actions the energy-only BM model does not
    explicitly represent.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    logger : Logger

    Returns
    -------
    DataFrame with columns: datetime, settlement_date, settlement_period,
        cost, volume, so_flag, stor_flag, service.
        Empty DataFrame if endpoint is unavailable.
    """
    _check_requests_available()

    url = f"{ELEXON_API_BASE}/datasets/DISBSAD/stream"
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    all_records = []
    chunk_start = start
    chunk_num = 0
    fetch_start = time.time()

    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=7), end)
        chunk_num += 1
        params = {
            "from": chunk_start.strftime("%Y-%m-%dT00:00Z"),
            "to": chunk_end.strftime("%Y-%m-%dT00:00Z"),
            "format": "json",
        }

        elapsed = time.time() - fetch_start
        logger.info(
            f"  Fetching DISBSAD chunk {chunk_num}: "
            f"{chunk_start.date()} to {chunk_end.date()} ({elapsed:.0f}s elapsed)"
        )

        try:
            resp = requests.get(url, params=params, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                all_records.extend(data)
            elif isinstance(data, dict):
                all_records.extend(data.get("data", []))
        except Exception as e:
            logger.warning(f"  DISBSAD stream chunk failed: {e}")
            if chunk_num == 1:
                logger.warning("  DISBSAD stream endpoint unavailable")
                return pd.DataFrame()

        chunk_start = chunk_end
        time.sleep(0.5)

    if not all_records:
        logger.warning("No DISBSAD data returned from stream")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    col_map = {
        "settlementDate": "settlement_date",
        "settlementPeriod": "settlement_period",
        "soFlag": "so_flag",
        "storFlag": "stor_flag",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "settlement_date" in df.columns and "settlement_period" in df.columns:
        dt = pd.to_datetime(df["settlement_date"], errors="coerce")
        sp_offset = pd.to_timedelta(
            (pd.to_numeric(df["settlement_period"], errors="coerce").fillna(1).astype(int) - 1) * 30,
            unit="min",
        )
        df["datetime"] = dt + sp_offset

    for numeric_col in ["cost", "volume"]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce").fillna(0.0)

    for flag_col in ["so_flag", "stor_flag"]:
        if flag_col not in df.columns:
            df[flag_col] = False

    if "service" not in df.columns:
        df["service"] = pd.NA

    elapsed = time.time() - fetch_start
    logger.info(f"  DISBSAD stream complete: {len(df)} records in {elapsed:.0f}s")
    return df


def fetch_system_prices_range(
    start_date: str,
    end_date: str,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch half-hourly system buy/sell prices for a date range.

    Downloads SBP (System Buy Price) and SSP (System Sell Price) for every
    settlement period. These are the imbalance cash-out prices that reflect
    BM actions, useful for comparing against model nodal prices.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    logger : Logger

    Returns
    -------
    DataFrame indexed by datetime with columns: system_buy_price,
        system_sell_price (both £/MWh). Resampled to hourly.
    """
    _check_requests_available()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq="D")

    logger.info(f"Fetching system prices: {start_date} to {end_date} ({len(dates)} days)")

    all_rows = []
    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        base = pd.Timestamp(date_str)

        try:
            url = f"{ELEXON_API_BASE}/balancing/settlement/system-prices/{date_str}"
            resp = requests.get(url, params={"format": "json"}, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            for rec in data:
                sp = int(rec.get("settlementPeriod", 0))
                dt = base + timedelta(minutes=30 * (sp - 1))
                all_rows.append({
                    "datetime": dt,
                    "system_buy_price": float(rec.get("systemBuyPrice", 0)),
                    "system_sell_price": float(rec.get("systemSellPrice", 0)),
                })
        except Exception as e:
            logger.warning(f"  System prices failed for {date_str}: {e}")

        time.sleep(1.0 / ELEXON_RATE_LIMIT)

    if not all_rows:
        logger.warning("No system price data collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).set_index("datetime").sort_index()

    # Resample to hourly (mean of two half-hours)
    df_hourly = df.resample("h").mean()

    logger.info(f"System prices: {len(df_hourly)} hourly periods, "
                f"SBP mean £{df_hourly['system_buy_price'].mean():.1f}/MWh")
    return df_hourly


def retrieve_bm_validation_data(
    start_date: str,
    end_date: str,
    output_dir: str,
    logger: logging.Logger = logger,
) -> dict:
    """
    Fetch all ELEXON data needed for BM validation for a date range.

        Downloads four datasets:
      1. BOALF (Bid-Offer Acceptance Levels) — actual BM acceptances
      2. System prices (SBP/SSP) — half-hourly imbalance prices
      3. Full B1610 generation — actual output per BMU (all settlement periods)
            4. DISBSAD — non-BM balancing-service adjustment volumes and costs

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    output_dir : str
        Directory to write output CSVs.
    logger : Logger

    Returns
    -------
    dict with keys 'boalf_file', 'system_prices_file', 'b1610_file',
        'disbsad_file'
        pointing to output paths.
    """
    _check_requests_available()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BM VALIDATION DATA FETCH")
    logger.info("=" * 70)
    logger.info(f"Period: {start_date} to {end_date}")

    # ── 1. Fetch BOALF (BM acceptances) ──────────────────────────────
    logger.info("\n── Stage 1: BOALF (BM Acceptances) ──")
    boalf = fetch_boalf_bulk(start_date, end_date, logger=logger)
    boalf_path = out_path / "boalf_data.csv"
    if not boalf.empty:
        boalf.to_csv(boalf_path, index=False)
        logger.info(f"Saved BOALF: {len(boalf)} records → {boalf_path}")
    else:
        if boalf_path.exists() and boalf_path.stat().st_size >= 100:
            logger.warning(
                "No BOALF data collected; preserving existing cached file "
                f"{boalf_path}"
            )
        else:
            logger.warning("No BOALF data collected")

    # ── 2. Fetch system prices ───────────────────────────────────────
    logger.info("\n── Stage 2: System Prices (SBP/SSP) ──")
    sys_prices = fetch_system_prices_range(start_date, end_date, logger=logger)
    sys_prices_path = out_path / "system_prices.csv"
    if not sys_prices.empty:
        sys_prices.to_csv(sys_prices_path)
        logger.info(f"Saved system prices: {len(sys_prices)} periods → {sys_prices_path}")
    else:
        pd.DataFrame().to_csv(sys_prices_path)
        logger.warning("No system price data collected")

    # ── 3. Fetch full B1610 generation ───────────────────────────────
    logger.info("\n── Stage 3: B1610 Actual Generation (full resolution) ──")
    b1610 = fetch_b1610_bulk(start_date, end_date, logger=logger)
    b1610_path = out_path / "b1610_actual.csv"
    if not b1610.empty:
        b1610.to_csv(b1610_path)
        logger.info(f"Saved B1610: {b1610.shape} → {b1610_path}")
    else:
        if b1610_path.exists() and b1610_path.stat().st_size >= 100:
            logger.warning(
                "No B1610 data collected; preserving existing cached file "
                f"{b1610_path}"
            )
        else:
            logger.warning("No B1610 data collected")

    # ── 4. Fetch DISBSAD ───────────────────────────────────────────────
    logger.info("\n── Stage 4: DISBSAD (non-BM balancing-service adjustments) ──")
    disbsad = fetch_disbsad_range(start_date, end_date, logger=logger)
    disbsad_path = out_path / "disbsad_data.csv"
    if not disbsad.empty:
        disbsad.to_csv(disbsad_path, index=False)
        logger.info(f"Saved DISBSAD: {len(disbsad)} records → {disbsad_path}")
    else:
        pd.DataFrame().to_csv(disbsad_path, index=False)
        logger.warning("No DISBSAD data collected")

    logger.info("=" * 70)
    logger.info("BM VALIDATION DATA FETCH COMPLETE")
    logger.info("=" * 70)

    return {
        "boalf_file": str(boalf_path),
        "system_prices_file": str(sys_prices_path),
        "b1610_file": str(b1610_path),
        "disbsad_file": str(disbsad_path),
    }


def main():
    """Snakemake entry point."""
    global logger

    log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else "elexon_data"
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("RETRIEVING ELEXON MARKET DATA")
    logger.info("=" * 80)

    scenario_config = snakemake.params.scenario_config
    solve_period = scenario_config.get("solve_period", {})

    start_date = solve_period.get("start", "2023-01-01 00:00")[:10]
    end_date = solve_period.get("end", "2023-01-07 23:00")[:10]

    output_dir = str(Path(snakemake.output.offers_file).parent)

    results = retrieve_elexon_market_data(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("ELEXON DATA RETRIEVAL COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
