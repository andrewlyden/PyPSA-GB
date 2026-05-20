"""
Calibrate monthly per-generator empirical marginal costs from ELEXON data.

Runs as a Snakemake rule or standalone CLI. Fetches (or loads cached) ELEXON
B1610 generation + MID wholesale prices, then computes per-generator switch-on
prices for each calendar month the scenario's solve period overlaps.

Output: CSV with columns  generator, carrier, month, empirical_mc, n_bmus, ...
Used by apply_marginal_costs.py to build month-aware time-varying MC.

Usage (standalone):
    python scripts/generators/calibrate_monthly_mc.py \\
        --year 2023 --network resources/network/…_thermal_generators.pkl \\
        --output resources/marginal_costs/Historical_2023_lowwind_monthly_mc.csv

Usage (Snakemake): called via rule calibrate_monthly_empirical_mc
"""

import pandas as pd
import numpy as np
import argparse
import logging
import time
from pathlib import Path
from datetime import timedelta

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("calibrate_monthly_mc")


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


def calibrate_monthly(
    pn_df,
    prices,
    network_path,
    output_file=None,
    logger=logger,
):
    """
    Compute per-generator empirical MCs for each calendar month.

    Slices the PN + price data by month, then calls the existing
    per-generator switch-on logic on each slice. Returns a long-form
    DataFrame with a `month` column (1-12).

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

    Returns
    -------
    DataFrame with columns: generator, carrier, month, empirical_mc, n_bmus,
        total_dispatch_periods, avg_capacity_factor
    """
    from scripts.generators.calibrate_thermal_mc import (
        compute_per_generator_switch_on,
        _build_direct_bmu_mapping,
    )
    from scripts.utilities.network_io import load_network as _load_net

    # _build_direct_bmu_mapping uses pypsa.Network() which can't open .pkl.
    # If the path is .pkl, save a temporary .nc for the mapping builder.
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

    # Build BMU mapping once (shared across all months)
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

    all_months = []
    available_months = sorted(pn_df.index.month.unique())

    for month in available_months:
        mask = pn_df.index.month == month
        pn_month = pn_df.loc[mask]
        prices_month = prices.loc[prices.index.month == month]

        if len(pn_month) < 5 or len(prices_month) < 5:
            logger.info(f"  Month {month:02d}: insufficient data "
                        f"(PN={len(pn_month)}, prices={len(prices_month)}) — skipping")
            continue

        result = compute_per_generator_switch_on(
            pn_month, prices_month, bmu_mapping, logger
        )

        if result.empty:
            logger.info(f"  Month {month:02d}: no generators matched")
            continue

        result['month'] = month
        all_months.append(result)
        logger.info(f"  Month {month:02d}: {len(result)} generators, "
                    f"mean MC £{result['empirical_mc'].mean():.1f}/MWh")

    if not all_months:
        logger.warning("No monthly empirical MCs computed")
        return pd.DataFrame()

    combined = pd.concat(all_months, ignore_index=True)
    logger.info(f"\nMonthly calibration complete: {len(combined)} rows "
                f"({combined['generator'].nunique()} generators × "
                f"{combined['month'].nunique()} months)")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_file, index=False)
        logger.info(f"Saved monthly MCs → {output_file}")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def snakemake_main():
    """Entry point when called from Snakemake rule."""
    global logger

    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "calibrate_monthly_mc"
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("MONTHLY EMPIRICAL MC CALIBRATION (SNAKEMAKE)")
    logger.info("=" * 80)

    network_path = snakemake.input.network
    output_file = snakemake.output.monthly_mc
    scenario_config = snakemake.params.scenario_config
    data_dir = getattr(snakemake.params, 'data_dir', 'resources/market/elexon')

    modelled_year = scenario_config.get('modelled_year', 2023)
    # For historical scenarios, calibration year = modelled year
    # For future scenarios, use the most recent historical year available
    cal_year = modelled_year if modelled_year <= 2024 else 2023

    logger.info(f"Modelled year: {modelled_year}, calibration year: {cal_year}")
    logger.info(f"Network: {network_path}")
    logger.info(f"Output: {output_file}")

    # Fetch or load cached ELEXON data
    pn_df, prices = _fetch_or_load_calibration_data(cal_year, data_dir)

    # Run monthly calibration
    calibrate_monthly(
        pn_df=pn_df,
        prices=prices,
        network_path=network_path,
        output_file=output_file,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("✓ MONTHLY CALIBRATION COMPLETE")
    logger.info("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate monthly per-generator empirical MCs from ELEXON data"
    )
    parser.add_argument("--year", type=int, required=True,
                        help="Calibration year (data will be fetched if needed)")
    parser.add_argument("--network", type=str, required=True,
                        help="PyPSA network path (.nc or .pkl)")
    parser.add_argument("--output", type=str,
                        default="resources/marginal_costs/monthly_empirical_mc.csv",
                        help="Output CSV path")
    parser.add_argument("--data-dir", type=str, default="resources/market/elexon",
                        help="Directory for ELEXON data cache")
    parser.add_argument("--force-fetch", action="store_true",
                        help="Re-fetch ELEXON data even if cached")
    args = parser.parse_args()

    pn_df, prices = _fetch_or_load_calibration_data(
        args.year, args.data_dir, force_fetch=args.force_fetch
    )

    calibrate_monthly(
        pn_df=pn_df,
        prices=prices,
        network_path=args.network,
        output_file=args.output,
        logger=logger,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
        snakemake_main()
    except NameError:
        main()
