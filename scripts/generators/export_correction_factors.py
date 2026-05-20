"""
Export carrier correction factors from historical ELEXON-calibrated runs.

Reads existing daily empirical MC files from historical scenarios and computes
per-carrier correction factors (median_empirical / formula_mc). These can then
be applied to future scenarios to calibrate FES-based formula MCs using real
market data.

Usage (standalone):
    python scripts/generators/export_correction_factors.py \
        --input-dir resources/marginal_costs \
        --output data/market/historical_carrier_correction_factors.csv

The output CSV has columns:
    source_year, carrier, correction_factor, empirical_mc, formula_mc, n_generators

A "median" row across all source years is appended for each carrier.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_daily_mc_files(input_dir: Path) -> dict:
    """Find daily empirical MC CSVs and extract source year from filename.

    Expected pattern: *_daily_empirical_mc.csv with a 4-digit year in the name.
    Returns {year: filepath}.
    """
    files = {}
    for f in sorted(input_dir.glob("*_daily_empirical_mc.csv")):
        match = re.search(r'(\d{4})', f.stem)
        if match:
            year = int(match.group(1))
            files[year] = f
    return files


def compute_correction_factors(daily_mc_path: Path) -> pd.DataFrame:
    """Compute per-carrier correction factors from a single daily MC file.

    Returns DataFrame with columns: carrier, correction_factor, empirical_mc,
    formula_mc, n_generators.
    """
    df = pd.read_csv(daily_mc_path)
    if 'generator' not in df.columns or 'empirical_mc' not in df.columns:
        return pd.DataFrame()
    if 'carrier' not in df.columns:
        return pd.DataFrame()

    # Compute median empirical MC per carrier across all generators and days
    carrier_stats = (
        df.groupby('carrier')
        .agg(
            empirical_mc=('empirical_mc', 'median'),
            n_generators=('generator', 'nunique'),
        )
        .reset_index()
    )
    return carrier_stats


def _formula_mc(carrier: str, fuel_prices: dict, carbon_price: float) -> float:
    """Calculate formula MC for a carrier using standard efficiency assumptions."""
    CARRIER_EFF = {
        'CCGT': 0.49, 'OCGT': 0.35, 'Coal': 0.36, 'coal': 0.36,
        'Oil': 0.30, 'oil': 0.30, 'Biomass': 0.35, 'nuclear': 0.33,
    }
    EMISSION_FACTORS = {
        'CCGT': 488, 'OCGT': 488, 'Coal': 846, 'coal': 846,
        'Oil': 533, 'oil': 533, 'Biomass': 120, 'nuclear': 12,
    }

    eff = CARRIER_EFF.get(carrier)
    if eff is None:
        return 0.0
    fp = fuel_prices.get(carrier, 0)
    ef = EMISSION_FACTORS.get(carrier, 0)
    return (fp / eff) + (ef * carbon_price / 1000 / eff)


# Historical fuel and carbon prices (matching apply_marginal_costs.py)
HISTORICAL_FUEL_PRICES = {
    2021: {'CCGT': 45.0, 'OCGT': 45.0, 'Coal': 15.0, 'coal': 15.0, 'Oil': 50.0, 'oil': 50.0, 'Biomass': 35.0, 'nuclear': 8.0},
    2022: {'CCGT': 80.0, 'OCGT': 80.0, 'Coal': 25.0, 'coal': 25.0, 'Oil': 70.0, 'oil': 70.0, 'Biomass': 40.0, 'nuclear': 8.0},
    2023: {'CCGT': 40.0, 'OCGT': 40.0, 'Coal': 15.0, 'coal': 15.0, 'Oil': 55.0, 'oil': 55.0, 'Biomass': 40.0, 'nuclear': 8.0},
    2024: {'CCGT': 35.0, 'OCGT': 35.0, 'Coal': 12.0, 'coal': 12.0, 'Oil': 50.0, 'oil': 50.0, 'Biomass': 40.0, 'nuclear': 8.0},
}
HISTORICAL_CARBON_PRICES = {
    2021: 63.0, 2022: 88.0, 2023: 68.0, 2024: 85.0,
}


def export_correction_factors(input_dir: Path, output_path: Path,
                              verbose: bool = False) -> pd.DataFrame:
    """Main function: find files, compute factors, write output."""
    mc_files = find_daily_mc_files(input_dir)
    if not mc_files:
        print(f"No daily empirical MC files found in {input_dir}")
        return pd.DataFrame()

    if verbose:
        print(f"Found {len(mc_files)} daily MC files: {sorted(mc_files.keys())}")

    all_rows = []
    for year, filepath in sorted(mc_files.items()):
        if year not in HISTORICAL_FUEL_PRICES:
            if verbose:
                print(f"  Skipping year {year}: no historical fuel prices available")
            continue

        fuel_prices = HISTORICAL_FUEL_PRICES[year]
        carbon_price = HISTORICAL_CARBON_PRICES.get(year, 85.0)

        carrier_stats = compute_correction_factors(filepath)
        if carrier_stats.empty:
            continue

        for _, row in carrier_stats.iterrows():
            carrier = row['carrier']
            emp_mc = row['empirical_mc']
            formula = _formula_mc(carrier, fuel_prices, carbon_price)
            if formula > 0:
                factor = emp_mc / formula
            else:
                factor = 1.0

            all_rows.append({
                'source_year': year,
                'carrier': carrier,
                'correction_factor': round(factor, 4),
                'empirical_mc': round(emp_mc, 2),
                'formula_mc': round(formula, 2),
                'n_generators': int(row['n_generators']),
            })
            if verbose:
                print(f"  {year} {carrier}: factor={factor:.4f} "
                      f"(empirical={emp_mc:.1f} / formula={formula:.1f})")

    if not all_rows:
        print("No correction factors computed")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    # Add median row per carrier across all source years
    median_rows = []
    for carrier in result['carrier'].unique():
        cdf = result[result['carrier'] == carrier]
        median_rows.append({
            'source_year': 'median',
            'carrier': carrier,
            'correction_factor': round(cdf['correction_factor'].median(), 4),
            'empirical_mc': round(cdf['empirical_mc'].median(), 2),
            'formula_mc': round(cdf['formula_mc'].median(), 2),
            'n_generators': int(cdf['n_generators'].median()),
        })

    result = pd.concat([result, pd.DataFrame(median_rows)], ignore_index=True)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")
    if verbose:
        print("\nMedian correction factors:")
        for _, row in result[result['source_year'] == 'median'].iterrows():
            print(f"  {row['carrier']}: {row['correction_factor']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Export carrier correction factors from historical ELEXON calibration"
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("resources/marginal_costs"),
        help="Directory containing *_daily_empirical_mc.csv files"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/market/historical_carrier_correction_factors.csv"),
        help="Output CSV path"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    export_correction_factors(args.input_dir, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
