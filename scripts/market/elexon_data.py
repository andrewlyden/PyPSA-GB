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

    url = f"{ELEXON_API_BASE}/balancing/settlement/bid-offer"
    params = {"settlementDate": date, "format": "json"}
    if settlement_period is not None:
        params["settlementPeriod"] = settlement_period

    all_records = []
    params["offset"] = 0
    params["limit"] = 1000

    while True:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data", [])
        if not records:
            break
        all_records.extend(records)
        if len(records) < params["limit"]:
            break
        params["offset"] += params["limit"]
        time.sleep(1.0 / ELEXON_RATE_LIMIT)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Standardise column names
    col_map = {
        "bmUnit": "bmu_id",
        "settlementPeriod": "settlement_period",
        "offerPrice": "offer_price",
        "bidPrice": "bid_price",
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

    # Separate offers (positive pair_id → higher volume = offer) and bids
    offers = bod_df[bod_df["offer_price"].notna()].copy()
    bids = bod_df[bod_df["bid_price"].notna()].copy()

    def _weighted_avg(group, price_col, vol_col):
        vols = group[vol_col].abs()
        total = vols.sum()
        if total == 0:
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

    logger.info(f"Fetched MID prices: {len(prices)} periods, "
                f"£{prices.min():.1f} to £{prices.max():.1f}/MWh, "
                f"mean £{prices.mean():.1f}/MWh")
    return prices


def retrieve_wholesale_calibration_data(
    start_date: str,
    end_date: str,
    output_dir: str,
    sample_sps: list = None,
    logger: logging.Logger = logger,
) -> dict:
    """
    Fetch year-long wholesale market data for per-generator MC calibration.

    Downloads:
      1. MID wholesale prices via bulk stream (fast, full resolution)
      2. B1610 actual generation per BMU at sampled settlement periods
         (2 SPs per day by default — morning peak SP20 + evening peak SP38)

    This gives ~730 observations per generator over a year, enough to see
    which generators cycle on/off at different price levels.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    output_dir : str
        Directory to write output CSVs.
    sample_sps : list, optional
        Settlement periods to sample B1610 at. Default [20, 38] (10am, 7pm).
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
    n_queries = n_days * len(sample_sps)

    logger.info("=" * 70)
    logger.info("WHOLESALE MARKET CALIBRATION DATA FETCH")
    logger.info("=" * 70)
    logger.info(f"Period: {start_date} to {end_date} ({n_days} days)")
    logger.info(f"Sample SPs: {sample_sps} ({len(sample_sps)} per day)")
    logger.info(f"B1610 API calls: {n_queries} (≈{n_queries / ELEXON_RATE_LIMIT / 60:.0f} min)")

    # ── 1. Fetch MID wholesale prices (bulk, fast) ───────────────────
    logger.info("\n── Stage 1: Market Index Prices (MID) ──")
    mid_prices = fetch_mid_prices_bulk(start_date, end_date, logger=logger)

    prices_path = out_path / "mid_prices.csv"
    mid_prices.to_csv(prices_path)
    logger.info(f"Saved MID prices: {len(mid_prices)} periods → {prices_path}")

    # ── 2. Fetch B1610 at sampled SPs ────────────────────────────────
    logger.info(f"\n── Stage 2: B1610 Generation ({n_queries} API calls) ──")
    all_gen_rows = []
    failed = 0

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        base = pd.Timestamp(date_str)

        if (i + 1) % 30 == 0 or i == 0:
            logger.info(f"  Day {i + 1}/{n_days}: {date_str} "
                        f"({len(all_gen_rows)} records so far)")

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

    # Build generation matrix
    pn_path = out_path / "pn_data.csv"
    if all_gen_rows:
        gen_long = pd.DataFrame(all_gen_rows)
        gen_pivot = gen_long.pivot_table(
            index="datetime", columns="bmu_id", values="generation_mw",
            aggfunc="max"
        ).fillna(0.0)
        gen_pivot.to_csv(pn_path)
        logger.info(f"Saved generation data: {gen_pivot.shape[0]} periods × "
                     f"{gen_pivot.shape[1]} BMUs → {pn_path}")
    else:
        pd.DataFrame().to_csv(pn_path)
        logger.warning("No generation data collected")

    logger.info("=" * 70)
    return {"pn_file": str(pn_path), "prices_file": str(prices_path)}


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
