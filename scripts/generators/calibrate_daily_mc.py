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
"""

import pandas as pd
import numpy as np
import argparse
import logging
import time
from pathlib import Path
from datetime import timedelta

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("calibrate_daily_mc")

# Default rolling window half-width: ±7 days (15-day centred window per date).
# Sensitivity analysis across 2021-2024 shows:
#   ±3d: only 128/~220 BMUs have ≥10 dispatch periods → noisy estimates
#   ±7d: 203/~220 BMUs reach ≥10 periods → stable, year-appropriate MCs
#   ±14d+: diminishing coverage gains; ±30d+ risks regime contamination
#   in volatile years (2022 CV=40.5%, 2021 CV=53.1% across the full year).
DEFAULT_WINDOW_DAYS = 7


def _fetch_or_load_calibration_data(year, data_dir, force_fetch=False):
    """Fetch (or load cached) PN + MID data for a whole year."""
    from scripts.market.elexon_data import (
        retrieve_wholesale_calibration_data,
    )

    data_path = Path(data_dir)
    pn_path = data_path / f"pn_data_{year}.csv"
    mid_path = data_path / f"mid_prices_{year}.csv"

    if pn_path.exists() and mid_path.exists() and not force_fetch:
        logger.info(f"Using cached calibration data for {year}")
        pn_df = pd.read_csv(pn_path, index_col=0, parse_dates=True)
        prices = pd.read_csv(mid_path, index_col=0, parse_dates=True).squeeze("columns")
        return pn_df, prices

    logger.info(f"Fetching ELEXON calibration data for {year} (B1610 + MID)")
    result = retrieve_wholesale_calibration_data(
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
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
        Half-width of rolling window (default 3 → 7-day centred window).

    Returns
    -------
    DataFrame with columns: generator, carrier, date, empirical_mc, n_bmus,
        total_dispatch_periods, avg_capacity_factor
    """
    from scripts.generators.calibrate_thermal_mc import (
        compute_per_generator_switch_on,
        _build_direct_bmu_mapping,
    )
    from scripts.utilities.network_io import load_network as _load_net

    # _build_direct_bmu_mapping uses pypsa.Network() which can't open .pkl.
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

    # Build BMU mapping once (shared across all days)
    bmu_mapping = _build_direct_bmu_mapping(
        bmu_ids=list(pn_df.columns),
        network_path=mapping_path,
        logger=logger,
    )

    # Cleanup temp file
    if net_path.endswith('.pkl'):
        try:
            os.unlink(mapping_path)
        except OSError:
            pass

    all_days = []
    available_dates = sorted(pn_df.index.normalize().unique())
    logger.info(f"Calibrating {len(available_dates)} dates with ±{window_days}-day window")

    for target_date in available_dates:
        # Rolling window: target_date ± window_days
        win_start = target_date - pd.Timedelta(days=window_days)
        win_end = target_date + pd.Timedelta(days=window_days + 1) - pd.Timedelta(seconds=1)

        pn_window = pn_df.loc[(pn_df.index >= win_start) & (pn_df.index <= win_end)]
        prices_window = prices.loc[(prices.index >= win_start) & (prices.index <= win_end)]

        if len(pn_window) < 10 or len(prices_window) < 10:
            continue

        result = compute_per_generator_switch_on(
            pn_window, prices_window, bmu_mapping, logger=logging.getLogger('null')
        )

        if result.empty:
            continue

        result['date'] = target_date.date()
        all_days.append(result)

    if not all_days:
        logger.warning("No daily empirical MCs computed")
        return pd.DataFrame()

    combined = pd.concat(all_days, ignore_index=True)
    logger.info(f"\nDaily calibration complete: {len(combined)} rows "
                f"({combined['generator'].nunique()} generators × "
                f"{combined['date'].nunique()} days)")

    # Log summary statistics per generator
    gen_stats = combined.groupby('generator')['empirical_mc'].agg(['mean', 'std', 'min', 'max'])
    for gen, row in gen_stats.iterrows():
        logger.info(f"  {gen}: mean £{row['mean']:.1f}, "
                    f"std £{row['std']:.1f}, "
                    f"range £{row['min']:.1f}–£{row['max']:.1f}/MWh")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_file, index=False)
        logger.info(f"Saved daily MCs → {output_file}")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def snakemake_main():
    """Entry point when called from Snakemake rule."""
    global logger

    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "calibrate_daily_mc"
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("DAILY EMPIRICAL MC CALIBRATION (SNAKEMAKE)")
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

    logger.info(f"Modelled year: {modelled_year}, calibration year: {cal_year}")
    logger.info(f"Network: {network_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Window: ±{window_days} days")

    pn_df, prices = _fetch_or_load_calibration_data(cal_year, data_dir)

    calibrate_daily(
        pn_df=pn_df,
        prices=prices,
        network_path=network_path,
        output_file=output_file,
        window_days=window_days,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("DAILY CALIBRATION COMPLETE")
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
