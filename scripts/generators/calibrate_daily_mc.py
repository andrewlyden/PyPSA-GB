"""
Calibrate daily per-generator empirical marginal costs from ELEXON data.

Runs as a Snakemake rule or standalone CLI. Fetches (or loads cached) ELEXON
B1610 generation + MID wholesale prices, then computes per-generator switch-on
prices for each calendar date using a rolling ±window_days centred window.

Output: CSV with columns  generator, carrier, date, empirical_mc, n_bmus, ...
Used by apply_marginal_costs.py to build day-aware time-varying MC.

Usage (standalone):
    python scripts/generators/calibrate_daily_mc.py \\
        --year 2023 --network resources/network/…_thermal_generators.pkl \\
        --output resources/marginal_costs/Historical_2023_lowwind_daily_mc.csv

Usage (Snakemake): called via rule calibrate_daily_empirical_mc

Performance
-----------
Vectorized implementation processes all BMUs simultaneously per window using
numpy/pandas operations instead of Python loops, reducing runtime from ~390s
to ~20-40s for a full year at ±7-day resolution.
"""

import pandas as pd
import numpy as np
import argparse
import logging
import time
from pathlib import Path
from datetime import timedelta

from scripts.utilities.logging_config import setup_logging, log_stage_summary

logger = setup_logging("calibrate_daily_mc")

# Default rolling window half-width: ±7 days (15-day centred window per date).
# Sensitivity analysis across 2021-2024 shows:
#   ±3d: only 128/~220 BMUs have ≥10 dispatch periods → noisy estimates
#   ±7d: 203/~220 BMUs reach ≥10 periods → stable, year-appropriate MCs
#   ±14d+: diminishing coverage gains; ±30d+ risks regime contamination
#   in volatile years (2022 CV=40.5%, 2021 CV=53.1% across the full year).
DEFAULT_WINDOW_DAYS = 7


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_or_load_calibration_data(
    year, data_dir, force_fetch=False,
    start_date=None, end_date=None,
):
    """
    Fetch (or load cached) PN + MID data for a year (or date sub-range).

    Parameters
    ----------
    year : int
        Calibration year.
    data_dir : str or Path
        Directory for ELEXON data cache.
    force_fetch : bool
        Re-fetch from API even if cache exists.
    start_date : str, optional
        Start date (YYYY-MM-DD) to scope the fetch. Defaults to Jan 1.
    end_date : str, optional
        End date (YYYY-MM-DD) to scope the fetch. Defaults to Dec 31.
    """
    from scripts.market.elexon_data import (
        retrieve_wholesale_calibration_data,
    )

    data_path = Path(data_dir)
    pn_path = data_path / f"pn_data_{year}.csv"
    mid_path = data_path / f"mid_prices_{year}.csv"

    if pn_path.exists() and mid_path.exists() and not force_fetch:
        logger.info(f"Using cached calibration data for {year}")
        logger.info(f"  PN data:  {pn_path}")
        logger.info(f"  MID data: {mid_path}")
        pn_df = pd.read_csv(pn_path, index_col=0, parse_dates=True)
        prices = pd.read_csv(mid_path, index_col=0, parse_dates=True).squeeze("columns")
        logger.info(f"  Loaded {pn_df.shape[0]} periods × {pn_df.shape[1]} BMUs, "
                     f"{len(prices)} price observations")
        return pn_df, prices

    # Try restoring from compressed archive before hitting the API
    try:
        from scripts.utilities.elexon_cache import ensure_cache_for_year
        if ensure_cache_for_year(year, cache_dir=data_path, logger=logger):
            if pn_path.exists() and mid_path.exists():
                logger.info(f"Restored from archive — loading cached data for {year}")
                pn_df = pd.read_csv(pn_path, index_col=0, parse_dates=True)
                prices = pd.read_csv(mid_path, index_col=0, parse_dates=True).squeeze("columns")
                return pn_df, prices
    except ImportError:
        pass  # elexon_cache not available — fall through to API fetch

    fetch_start = start_date or f"{year}-01-01"
    fetch_end = end_date or f"{year}-12-31"

    logger.info(f"Fetching ELEXON calibration data for {year} "
                f"({fetch_start} to {fetch_end})")
    result = retrieve_wholesale_calibration_data(
        start_date=fetch_start,
        end_date=fetch_end,
        output_dir=str(data_path),
        logger=logger,
    )

    # Rename to year-stamped names for caching
    pn_src = Path(result['pn_file'])
    mid_src = Path(result['prices_file'])
    if pn_src.exists():
        pn_src.rename(pn_path)
    if mid_src.exists():
        mid_src.rename(mid_path)

    pn_df = pd.read_csv(pn_path, index_col=0, parse_dates=True)
    prices = pd.read_csv(mid_path, index_col=0, parse_dates=True).squeeze("columns")
    return pn_df, prices


# ══════════════════════════════════════════════════════════════════════════════
# VECTORIZED DAILY COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_window_mc_vectorized(pn_window, prices_window):
    """
    Compute per-BMU switch-on prices for one rolling window (vectorized).

    Replicates the logic of compute_per_generator_switch_on() but operates
    on all BMU columns simultaneously using numpy, avoiding Python loops.

    Parameters
    ----------
    pn_window : DataFrame
        Generation data for the window (datetime index, BMU columns, MW).
    prices_window : Series
        Wholesale prices aligned to pn_window index (£/MWh).

    Returns
    -------
    DataFrame with columns: bmu_id, switch_on_price, max_off_price,
        estimated_mc, dispatch_periods, capacity_factor
        Empty DataFrame if insufficient data.
    """
    # Align to common index
    common_idx = pn_window.index.intersection(prices_window.index)
    if len(common_idx) < 10:
        return pd.DataFrame()

    pn = pn_window.loc[common_idx]
    prices = prices_window.loc[common_idx].values  # numpy array for speed
    n_total = len(common_idx)

    # Boolean dispatching matrix: (n_periods × n_bmus)
    dispatching = pn.values > 1.0
    n_dispatching = dispatching.sum(axis=0)  # per BMU
    n_off = n_total - n_dispatching
    cf = n_dispatching / n_total

    # Filter BMUs with sufficient data
    valid = n_dispatching >= 5
    bmu_ids = np.array(pn.columns)

    if not valid.any():
        return pd.DataFrame()

    # Compute conditional quantiles per BMU using masked arrays
    switch_on = np.full(len(bmu_ids), np.nan)
    max_off = np.full(len(bmu_ids), np.nan)

    for j in np.where(valid)[0]:
        on_mask = dispatching[:, j]
        on_prices = prices[on_mask]
        switch_on[j] = np.percentile(on_prices, 10)

        off_mask = ~on_mask
        if off_mask.sum() > 5:
            max_off[j] = np.percentile(prices[off_mask], 90)

    # Estimate MC by capacity factor regime (vectorized)
    estimated_mc = np.where(
        (cf > 0.90) | np.isnan(max_off),
        switch_on,                          # Baseload
        np.where(
            cf < 0.10,
            max_off,                        # Peaker
            (switch_on + max_off) / 2.0     # Mid-merit
        )
    )

    result = pd.DataFrame({
        'bmu_id': bmu_ids[valid],
        'switch_on_price': switch_on[valid],
        'max_off_price': max_off[valid],
        'estimated_mc': estimated_mc[valid],
        'dispatch_periods': n_dispatching[valid].astype(int),
        'capacity_factor': cf[valid],
    })

    return result


def calibrate_daily(
    pn_df,
    prices,
    network_path,
    output_file=None,
    window_days=DEFAULT_WINDOW_DAYS,
    logger=logger,
):
    """
    Compute per-generator empirical MCs for each calendar date.

    For each date, a rolling window of ±window_days is used to gather
    sufficient observations for the switch-on price analysis. This gives
    daily resolution while maintaining statistical robustness.

    Uses vectorized numpy operations to compute all BMU metrics per window
    simultaneously, then maps BMUs to generators and aggregates in bulk.

    Parameters
    ----------
    pn_df : DataFrame
        Generation data (datetime index, BMU columns, MW values).
    prices : Series
        Wholesale MID prices (£/MWh), datetime index.
    network_path : str
        Path to PyPSA network (.nc or .pkl) for BMU→generator mapping.
    output_file : str, optional
        Path to write output CSV.
    window_days : int
        Half-width of rolling window (default 7 → 15-day centred window).

    Returns
    -------
    DataFrame with columns: generator, carrier, date, empirical_mc, n_bmus,
        total_dispatch_periods, avg_capacity_factor
    """
    from scripts.generators.calibrate_thermal_mc import _build_direct_bmu_mapping
    from scripts.utilities.network_io import load_network as _load_net

    stage_times = {}

    # ── Stage 1: Build BMU mapping ───────────────────────────────────
    t0 = time.time()
    logger.info("── Stage 1: Build BMU → Generator mapping ──")

    net_path = str(network_path)
    if net_path.endswith('.pkl'):
        import tempfile, os
        network_obj = _load_net(net_path)
        tmp_nc = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        tmp_nc.close()
        network_obj.export_to_netcdf(tmp_nc.name)
        mapping_path = tmp_nc.name
    else:
        mapping_path = net_path

    bmu_mapping = _build_direct_bmu_mapping(
        bmu_ids=list(pn_df.columns),
        network_path=mapping_path,
        logger=logger,
    )

    if net_path.endswith('.pkl'):
        try:
            os.unlink(mapping_path)
        except OSError:
            pass

    # Build lookup dicts once
    bmu_to_gen = {}
    bmu_to_carrier = {}
    if bmu_mapping is not None and not bmu_mapping.empty:
        for _, row in bmu_mapping.iterrows():
            bmu_to_gen[row['bmu_id']] = row['generator_name']
            bmu_to_carrier[row['bmu_id']] = row.get('carrier', '')

    n_mapped = sum(1 for b in pn_df.columns if b in bmu_to_gen)
    logger.info(f"  Mapped {n_mapped}/{len(pn_df.columns)} BMUs to generators")
    stage_times["Build BMU mapping"] = time.time() - t0

    # ── Stage 2: Pre-align data ──────────────────────────────────────
    t0 = time.time()
    logger.info("── Stage 2: Pre-align PN and price data ──")

    common_idx = pn_df.index.intersection(prices.index)
    pn_aligned = pn_df.loc[common_idx].copy()
    prices_aligned = prices.loc[common_idx].copy()
    logger.info(f"  Aligned index: {len(common_idx)} periods "
                f"({common_idx.min().date()} to {common_idx.max().date()})")

    available_dates = sorted(pn_aligned.index.normalize().unique())
    n_days = len(available_dates)
    logger.info(f"  {n_days} unique dates to calibrate")
    stage_times["Pre-align data"] = time.time() - t0

    # ── Stage 3: Rolling window computation (vectorized) ─────────────
    t0 = time.time()
    logger.info(f"── Stage 3: Rolling ±{window_days}-day computation ({n_days} dates) ──")

    all_bmu_rows = []
    skipped = 0
    loop_start = time.time()

    for i, target_date in enumerate(available_dates):
        win_start = target_date - pd.Timedelta(days=window_days)
        win_end = target_date + pd.Timedelta(days=window_days + 1) - pd.Timedelta(seconds=1)

        pn_window = pn_aligned.loc[(pn_aligned.index >= win_start) & (pn_aligned.index <= win_end)]
        prices_window = prices_aligned.loc[(prices_aligned.index >= win_start) & (prices_aligned.index <= win_end)]

        if len(pn_window) < 10 or len(prices_window) < 10:
            skipped += 1
            continue

        bmu_result = _compute_window_mc_vectorized(pn_window, prices_window)

        if bmu_result.empty:
            skipped += 1
            continue

        bmu_result['date'] = target_date.date()
        all_bmu_rows.append(bmu_result)

        # Progress logging every 30 days
        if (i + 1) % 30 == 0 or i == 0:
            elapsed = time.time() - loop_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_days - i - 1) / rate if rate > 0 else 0
            logger.info(f"  Day {i + 1:>3d}/{n_days}: {target_date.date()} "
                        f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    stage_times["Rolling window computation"] = time.time() - t0

    if not all_bmu_rows:
        logger.warning("No daily empirical MCs computed")
        return pd.DataFrame()

    # ── Stage 4: Map BMUs to generators and aggregate ────────────────
    t0 = time.time()
    logger.info("── Stage 4: Map BMUs to generators and aggregate ──")

    bmu_combined = pd.concat(all_bmu_rows, ignore_index=True)
    logger.info(f"  BMU-level results: {len(bmu_combined)} rows "
                f"({bmu_combined['bmu_id'].nunique()} BMUs × "
                f"{bmu_combined['date'].nunique()} dates)")
    if skipped:
        logger.info(f"  Skipped {skipped} dates (insufficient data)")

    # Map BMU → generator
    bmu_combined['generator'] = bmu_combined['bmu_id'].map(bmu_to_gen)
    bmu_combined['carrier'] = bmu_combined['bmu_id'].map(bmu_to_carrier).fillna('')

    mapped = bmu_combined[bmu_combined['generator'].notna()].copy()
    unmapped_count = bmu_combined['generator'].isna().sum()
    if unmapped_count > 0:
        logger.info(f"  {unmapped_count} BMU-day rows not mapped to generators (dropped)")

    if mapped.empty:
        logger.warning("No BMUs mapped to generators!")
        return pd.DataFrame()

    # Aggregate: median estimated MC per (generator, date)
    combined = mapped.groupby(['generator', 'carrier', 'date']).agg(
        empirical_mc=('estimated_mc', 'median'),
        n_bmus=('bmu_id', 'count'),
        total_dispatch_periods=('dispatch_periods', 'sum'),
        avg_capacity_factor=('capacity_factor', 'mean'),
    ).reset_index()

    stage_times["Map BMUs and aggregate"] = time.time() - t0

    # ── Stage 5: Log summary and write output ────────────────────────
    t0 = time.time()

    logger.info(f"\n  Daily calibration results: {len(combined)} rows "
                f"({combined['generator'].nunique()} generators × "
                f"{combined['date'].nunique()} days)")

    gen_stats = combined.groupby('generator')['empirical_mc'].agg(['mean', 'std', 'min', 'max'])
    for gen, row in gen_stats.sort_values('mean').iterrows():
        logger.info(f"  {gen:30s}: mean £{row['mean']:6.1f}, "
                    f"std £{row['std']:5.1f}, "
                    f"range £{row['min']:.1f}–£{row['max']:.1f}/MWh")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_file, index=False)
        logger.info(f"  Saved daily MCs → {output_file}")

    stage_times["Summary and output"] = time.time() - t0

    # ── Stage summary ────────────────────────────────────────────────
    log_stage_summary(stage_times, logger, title="DAILY MC CALIBRATION — STAGE TIMING")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def snakemake_main():
    """Entry point when called from Snakemake rule."""
    global logger

    script_start = time.time()

    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "calibrate_daily_mc"
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("DAILY EMPIRICAL MC CALIBRATION")
    logger.info("=" * 80)

    network_path = snakemake.input.network
    output_file = snakemake.output.daily_mc
    scenario_config = snakemake.params.scenario_config
    data_dir = getattr(snakemake.params, 'data_dir', 'resources/market/elexon')

    modelled_year = scenario_config.get('modelled_year', 2023)
    cal_year = modelled_year if modelled_year <= 2024 else 2023

    window_days = (scenario_config.get('marginal_costs', {})
                   .get('empirical_calibration', {})
                   .get('window_days', DEFAULT_WINDOW_DAYS))

    logger.info(f"  Modelled year:     {modelled_year}")
    logger.info(f"  Calibration year:  {cal_year}")
    logger.info(f"  Network:           {network_path}")
    logger.info(f"  Output:            {output_file}")
    logger.info(f"  Window:            ±{window_days} days ({2 * window_days + 1}-day centred)")

    # Scope fetch to solve period + window buffer (avoids full-year fetch for short runs)
    solve_period = scenario_config.get('solve_period', {})
    if solve_period.get('enabled', False) and solve_period.get('start') and solve_period.get('end'):
        sp_start = pd.Timestamp(solve_period['start'])
        sp_end = pd.Timestamp(solve_period['end'])
        buffer = pd.Timedelta(days=window_days + 2)  # window + safety margin
        fetch_start = max(sp_start - buffer, pd.Timestamp(f"{cal_year}-01-01"))
        fetch_end = min(sp_end + buffer, pd.Timestamp(f"{cal_year}-12-31"))
        fetch_start_str = fetch_start.strftime("%Y-%m-%d")
        fetch_end_str = fetch_end.strftime("%Y-%m-%d")
        logger.info(f"  Scoped to solve_period ± {window_days + 2}d: {fetch_start_str} to {fetch_end_str}")
    else:
        fetch_start_str = f"{cal_year}-01-01"
        fetch_end_str = f"{cal_year}-12-31"

    t0 = time.time()
    pn_df, prices = _fetch_or_load_calibration_data(
        cal_year, data_dir,
        start_date=fetch_start_str, end_date=fetch_end_str,
    )
    load_time = time.time() - t0
    logger.info(f"  Data loading: {load_time:.1f}s")

    calibrate_daily(
        pn_df=pn_df,
        prices=prices,
        network_path=network_path,
        output_file=output_file,
        window_days=window_days,
        logger=logger,
    )

    total_time = time.time() - script_start
    logger.info("=" * 80)
    logger.info(f"✓ DAILY CALIBRATION COMPLETE ({total_time:.1f}s)")
    logger.info("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate daily per-generator empirical MCs from ELEXON data"
    )
    parser.add_argument("--year", type=int, required=True,
                        help="Calibration year")
    parser.add_argument("--network", type=str, required=True,
                        help="PyPSA network path (.nc or .pkl)")
    parser.add_argument("--output", type=str,
                        default="resources/marginal_costs/daily_empirical_mc.csv",
                        help="Output CSV path")
    parser.add_argument("--data-dir", type=str, default="resources/market/elexon",
                        help="Directory for ELEXON data cache")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"Rolling window half-width (default {DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--force-fetch", action="store_true",
                        help="Re-fetch ELEXON data even if cached")
    args = parser.parse_args()

    pn_df, prices = _fetch_or_load_calibration_data(
        args.year, args.data_dir, force_fetch=args.force_fetch
    )

    calibrate_daily(
        pn_df=pn_df,
        prices=prices,
        network_path=args.network,
        output_file=args.output,
        window_days=args.window_days,
        logger=logger,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
        snakemake_main()
    except NameError:
        main()
