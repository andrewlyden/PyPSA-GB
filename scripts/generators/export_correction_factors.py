"""
Export carrier correction factors from historical ELEXON-calibrated runs.

Reads existing daily empirical MC files from historical scenarios and computes
per-carrier correction factors (median_empirical / formula_mc). These can then
be applied to future scenarios to calibrate FES-based formula MCs using real
market data.

Usage (standalone):
    python scripts/generators/export_correction_factors.py \
        --input-dir resources/marginal_costs \
        --output resources/marginal_costs/historical_carrier_correction_factors.csv

The output CSV has columns:
    source_year, carrier, correction_factor, adder_gbp_per_mwh, empirical_mc,
    formula_mc, n_generators, source_file

A "median" row across all source years is appended for each carrier
(nuclear excluded, see FIX-6).

Fixes (November 2026), each marked "FIX-n" in the code
------------------------------------------------------------------
FIX-1         The source year was the first 4 digits of the file *name*:
              ``test_zonal_market_0001_...`` became year 1, names without
              digits were skipped, and same-year files overwrote each other.
              The year now comes from the file's own ``date`` column. Files
              spanning several years are skipped. When several files give the
              same year, the one with the most dates is kept and the others are
              logged. The glob is ``*_daily_empirical_mc*.csv`` so the
              ``*_corrected.csv`` files are found.
FIX-2         Fuel prices, carbon prices, emission factors and efficiencies
              were copies of ``apply_marginal_costs.py`` (only 2021-24; Biomass
              EF 120 vs 0). They are now imported from it, so the factor is
              computed with the same numbers it is applied with, and 2010-2024
              (including 2020) are available.
FIX-3         A carrier needs at least ``min_generators`` generators (default 3)
              in a year; below that the row is skipped and logged. Coal and
              OCGT factors rested on 1 generator.
FIX-4         One multiplier on fuel + carbon fits poorly across years. An
              additive column ``adder_gbp_per_mwh = empirical - formula`` is
              written next to ``correction_factor`` so the two can be compared.
              ``apply_marginal_costs.py`` still reads ``correction_factor`` only.
FIX-5         A carrier with formula MC 0 (no efficiency in the table) was
              written with factor 1.0. It is now skipped and logged.
FIX-6         Default output moved from ``data/market/`` (raw inputs, immutable)
              to ``resources/marginal_costs/``; ``config/defaults.yaml`` still
              points at the old ``data/market`` file.
              ``print`` replaced by logging. Nuclear is left out of the
              ``median`` rows because ``apply_marginal_costs`` never applies it.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.generators import apply_marginal_costs as amc  # noqa: E402
from scripts.utilities.logging_config import setup_logging  # noqa: E402

logger = setup_logging("export_correction_factors")

DEFAULT_OUTPUT = Path("resources/marginal_costs/historical_carrier_correction_factors.csv")
DEFAULT_MIN_GENERATORS = 3

# FIX-2: the same tables apply_marginal_costs uses (aliases kept for callers)
HISTORICAL_FUEL_PRICES = amc.HISTORICAL_FUEL_PRICES
HISTORICAL_CARBON_PRICES = amc.HISTORICAL_CARBON_PRICES


def find_daily_mc_files(input_dir: Path, logger: logging.Logger = logger) -> dict:
    """
    Find daily empirical MC CSVs and take the source year from their dates.

    FIX-1: the year is read from the ``date`` column, not the file name.

    Parameters
    ----------
    input_dir : Path
        Directory to search for ``*_daily_empirical_mc*.csv``.
    logger : Logger

    Returns
    -------
    dict
        {year: filepath}. One file per year: the one with the most dates.
    """
    best = {}  # year -> (n_dates, path)
    for f in sorted(Path(input_dir).glob("*_daily_empirical_mc*.csv")):
        try:
            dates = pd.to_datetime(pd.read_csv(f, usecols=["date"])["date"], errors="coerce").dropna()
        except (ValueError, KeyError):
            logger.info(f"  {f.name}: no 'date' column - skipped")
            continue
        years = dates.dt.year.unique()
        if len(years) != 1:
            logger.info(f"  {f.name}: spans years {sorted(years)} - skipped")
            continue
        year, n = int(years[0]), int(dates.nunique())
        if year in best:
            keep, drop = ((n, f), best[year]) if n > best[year][0] else (best[year], (n, f))
            logger.info(f"  {year}: kept {keep[1].name} ({keep[0]} dates), "
                        f"not {drop[1].name} ({drop[0]} dates)")
            best[year] = keep
        else:
            best[year] = (n, f)
    return {y: p for y, (_, p) in sorted(best.items())}


def compute_correction_factors(daily_mc_path: Path) -> pd.DataFrame:
    """
    Per-carrier median empirical MC from a single daily MC file.

    Parameters
    ----------
    daily_mc_path : Path
        CSV with columns generator, carrier, empirical_mc.

    Returns
    -------
    DataFrame
        Columns carrier, empirical_mc, n_generators (empty if columns missing).
    """
    df = pd.read_csv(daily_mc_path)
    if not {'generator', 'empirical_mc', 'carrier'}.issubset(df.columns):
        return pd.DataFrame()

    # Median empirical MC per carrier across all generators and days
    return (
        df.groupby('carrier')
        .agg(empirical_mc=('empirical_mc', 'median'),
             n_generators=('generator', 'nunique'))
        .reset_index()
    )


def _formula_mc(carrier: str, fuel_prices: dict, carbon_price: float) -> float:
    """
    Formula MC for a carrier with the workflow's own efficiency table.

    FIX-2: uses ``apply_marginal_costs.FORMULA_CARRIER_EFF``,
    ``CARBON_EMISSION_FACTORS`` and ``calculate_marginal_cost``.

    Parameters
    ----------
    carrier : str
    fuel_prices : dict
        Expanded fuel prices from ``apply_marginal_costs.load_fuel_prices``.
    carbon_price : float
        £/t CO2.

    Returns
    -------
    float
        £/MWh_e, or 0.0 when the carrier has no efficiency in the table.
    """
    eff = amc.FORMULA_CARRIER_EFF.get(carrier)
    if eff is None:
        return 0.0
    return amc.calculate_marginal_cost(carrier, fuel_prices.get(carrier, 0.0), carbon_price, eff)[2]


def export_correction_factors(input_dir: Path, output_path: Path,
                              verbose: bool = False,
                              min_generators: int = DEFAULT_MIN_GENERATORS,
                              logger: logging.Logger = logger) -> pd.DataFrame:
    """
    Find daily MC files, compute per-carrier factors and adders, write the CSV.

    Parameters
    ----------
    input_dir : Path
        Directory with ``*_daily_empirical_mc*.csv`` files.
    output_path : Path
        CSV to write.
    verbose : bool
        Log each row at INFO.
    min_generators : int
        Minimum generators per carrier-year (FIX-3).
    logger : Logger

    Returns
    -------
    DataFrame
        The rows written (empty if none).
    """
    mc_files = find_daily_mc_files(input_dir, logger=logger)
    if not mc_files:
        logger.warning(f"No daily empirical MC files found in {input_dir}")
        return pd.DataFrame()
    names = {y: p.name for y, p in mc_files.items()}
    logger.info(f"Daily MC files by year: {names}")

    all_rows = []
    for year, filepath in sorted(mc_files.items()):
        if year not in HISTORICAL_FUEL_PRICES:
            logger.info(f"  Skipping year {year}: no historical fuel prices available")
            continue
        cfg = {'modelled_year': year, 'marginal_costs': {}}
        fuel_prices = amc.load_fuel_prices(cfg)
        carbon_price = amc.get_carbon_price(cfg)

        carrier_stats = compute_correction_factors(filepath)
        for _, row in carrier_stats.iterrows():
            carrier, emp_mc, n_gen = row['carrier'], row['empirical_mc'], int(row['n_generators'])
            formula = _formula_mc(carrier, fuel_prices, carbon_price)
            if formula <= 0:  # FIX-5
                logger.info(f"  {year} {carrier}: no formula MC (no efficiency) - skipped")
                continue
            if n_gen < min_generators:  # FIX-3
                logger.info(f"  {year} {carrier}: {n_gen} generator(s) < {min_generators} - skipped")
                continue
            all_rows.append({
                'source_year': year,
                'carrier': carrier,
                'correction_factor': round(emp_mc / formula, 4),
                'adder_gbp_per_mwh': round(emp_mc - formula, 2),  # FIX-4
                'empirical_mc': round(emp_mc, 2),
                'formula_mc': round(formula, 2),
                'n_generators': n_gen,
                'source_file': filepath.name,
            })
            if verbose:
                r = all_rows[-1]
                logger.info(f"  {year} {carrier}: factor={r['correction_factor']:.4f}, "
                            f"adder={r['adder_gbp_per_mwh']:+.1f} "
                            f"(empirical={emp_mc:.1f} / formula={formula:.1f}, n={n_gen})")

    if not all_rows:
        logger.warning("No correction factors computed")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    # Median row per carrier across source years (FIX-6: nuclear excluded)
    median_rows = []
    for carrier in result['carrier'].unique():
        if str(carrier).lower() == 'nuclear':
            continue
        cdf = result[result['carrier'] == carrier]
        median_rows.append({
            'source_year': 'median',
            'carrier': carrier,
            'correction_factor': round(cdf['correction_factor'].median(), 4),
            'adder_gbp_per_mwh': round(cdf['adder_gbp_per_mwh'].median(), 2),
            'empirical_mc': round(cdf['empirical_mc'].median(), 2),
            'formula_mc': round(cdf['formula_mc'].median(), 2),
            'n_generators': int(cdf['n_generators'].median()),
            'source_file': ';'.join(sorted(cdf['source_file'].unique())),
        })
    result = pd.concat([result, pd.DataFrame(median_rows)], ignore_index=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(result)} rows to {output_path}")
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export carrier correction factors from historical ELEXON calibration"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("resources/marginal_costs"),
                        help="Directory containing *_daily_empirical_mc*.csv files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--min-generators", type=int, default=DEFAULT_MIN_GENERATORS,
                        help="Minimum generators per carrier-year")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    export_correction_factors(args.input_dir, args.output, verbose=args.verbose,
                              min_generators=args.min_generators)


if __name__ == "__main__":
    main()
