"""
Calibrate thermal marginal costs using ELEXON day-ahead price data.

One-time prerun script (NOT part of the per-scenario Snakemake DAG).
Compares formula-based marginal costs against empirical switch-on prices
derived from ELEXON Physical Notification (PN) dispatch patterns.

Methodology (based on GBPower build_thermal_generator_prices.py):
    For each carrier group, identifies the price at which generators switch
    from zero to positive output (PN > 0). The median of these switch-on
    prices across generators and days gives the empirical marginal cost.

    correction_factor = empirical_mc / formula_mc

Output:
    data/market/carrier_correction_factors.csv
    Columns: carrier, formula_mc, empirical_mc, correction_factor

Usage:
    python scripts/generators/calibrate_thermal_mc.py \\
        --year 2023 \\
        --output data/market/carrier_correction_factors.csv

    Or within Python:
        from scripts.generators.calibrate_thermal_mc import calibrate_correction_factors
        factors = calibrate_correction_factors(year=2023)
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("calibrate_thermal_mc")

# Default formula MCs by carrier (£/MWh) for 2023 baseline
# These approximate the formula: (fuel_price / efficiency) + (emission_factor × carbon_price / 1000 / efficiency)
DEFAULT_FORMULA_MC = {
    'CCGT': 65.0,
    'OCGT': 130.0,
    'Coal': 95.0,
    'Oil': 150.0,
    'Biomass': 25.0,
    'nuclear': 10.0,
}


def compute_switch_on_prices(
    pn_df: pd.DataFrame,
    dah_prices: pd.Series,
    carrier_map: dict,
    logger: logging.Logger = logger,
) -> dict:
    """
    Compute empirical switch-on marginal cost per carrier from PN + day-ahead data.

    For each generator, finds the minimum day-ahead price at which it dispatches
    (PN > 0). The median across all generators in a carrier group gives the
    carrier's empirical marginal cost.

    Parameters
    ----------
    pn_df : DataFrame
        Physical Notification data. Index = datetime, columns = generator/BMU IDs,
        values = MW output.
    dah_prices : Series
        Day-ahead prices (£/MWh) indexed by datetime.
    carrier_map : dict
        {generator_id: carrier} mapping.
    logger : Logger

    Returns
    -------
    dict: {carrier: empirical_mc (£/MWh)}
    """
    switch_on_prices = {}

    for gen_id in pn_df.columns:
        carrier = carrier_map.get(gen_id, None)
        if carrier is None:
            continue

        pn = pn_df[gen_id]
        # Align with day-ahead prices
        common_idx = pn.index.intersection(dah_prices.index)
        if len(common_idx) < 48:
            continue

        pn_aligned = pn.loc[common_idx]
        prices_aligned = dah_prices.loc[common_idx]

        # Find timesteps where generator is dispatching (PN > 1 MW threshold)
        dispatching = pn_aligned > 1.0
        if dispatching.sum() < 10:
            continue

        # Switch-on price = min price at which generator dispatches
        dispatch_prices = prices_aligned[dispatching]
        switch_on = dispatch_prices.quantile(0.1)  # 10th percentile ≈ switch-on

        if carrier not in switch_on_prices:
            switch_on_prices[carrier] = []
        switch_on_prices[carrier].append(switch_on)

    # Compute median per carrier
    empirical = {}
    for carrier, prices in switch_on_prices.items():
        empirical[carrier] = float(np.median(prices))
        logger.info(f"  {carrier}: median switch-on £{empirical[carrier]:.1f}/MWh "
                     f"({len(prices)} generators)")

    return empirical


def calibrate_correction_factors(
    year: int = 2023,
    pn_file: str = None,
    dah_file: str = None,
    bmu_mapping_file: str = None,
    formula_mc: dict = None,
    output_file: str = None,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Compute carrier correction factors from empirical vs formula MC.

    Parameters
    ----------
    year : int
        Calibration year.
    pn_file : str, optional
        Path to PN data CSV (datetime index, BMU columns). If None, attempts
        to load from resources/market/.
    dah_file : str, optional
        Path to day-ahead price CSV. If None, attempts to load from resources/.
    bmu_mapping_file : str, optional
        Path to BMU→carrier mapping CSV.
    formula_mc : dict, optional
        Formula MC by carrier. Defaults to DEFAULT_FORMULA_MC.
    output_file : str, optional
        Path to write output CSV.
    logger : Logger

    Returns
    -------
    DataFrame with columns: carrier, formula_mc, empirical_mc, correction_factor
    """
    if formula_mc is None:
        formula_mc = DEFAULT_FORMULA_MC.copy()

    logger.info(f"Calibrating correction factors for year {year}")

    # Load PN data
    if pn_file and Path(pn_file).exists():
        pn_df = pd.read_csv(pn_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded PN data: {pn_df.shape}")
    else:
        logger.warning("No PN data file provided. Using placeholder correction factors.")
        # Return default factors (1.0 = no correction)
        rows = [{'carrier': c, 'formula_mc': mc, 'empirical_mc': mc,
                 'correction_factor': 1.0} for c, mc in formula_mc.items()]
        df = pd.DataFrame(rows)
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved placeholder factors to {output_file}")
        return df

    # Load day-ahead prices
    if dah_file and Path(dah_file).exists():
        dah_prices = pd.read_csv(dah_file, index_col=0, parse_dates=True).squeeze()
    else:
        logger.error("No day-ahead price data — cannot calibrate")
        return pd.DataFrame()

    # Load BMU→carrier mapping
    carrier_map = {}
    if bmu_mapping_file and Path(bmu_mapping_file).exists():
        bmu_df = pd.read_csv(bmu_mapping_file)
        if 'bmu_id' in bmu_df.columns and 'carrier' in bmu_df.columns:
            carrier_map = dict(zip(bmu_df['bmu_id'], bmu_df['carrier']))
    if not carrier_map:
        logger.warning("No BMU→carrier mapping — using column names as carrier guess")

    # Compute empirical switch-on prices
    empirical = compute_switch_on_prices(pn_df, dah_prices, carrier_map, logger)

    # Build correction factor table
    rows = []
    for carrier, fmc in formula_mc.items():
        emp = empirical.get(carrier, fmc)
        if fmc > 0:
            factor = emp / fmc
        else:
            factor = 1.0
        rows.append({
            'carrier': carrier,
            'formula_mc': fmc,
            'empirical_mc': emp,
            'correction_factor': round(factor, 4),
        })
        logger.info(f"  {carrier}: formula £{fmc:.1f}, empirical £{emp:.1f}, "
                     f"factor {factor:.3f}")

    df = pd.DataFrame(rows)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved correction factors to {output_file}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PER-GENERATOR MARGINAL COST CALIBRATION (WHOLESALE MARKET)
# ══════════════════════════════════════════════════════════════════════════════


def compute_per_generator_switch_on(
    pn_df: pd.DataFrame,
    prices: pd.Series,
    bmu_mapping_df: pd.DataFrame = None,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Compute per-generator empirical marginal cost from wholesale market data.

    Uses the "revealed preference" approach: over a year of data, each
    generator is observed running at some prices and not at others. Its
    marginal cost is estimated as the wholesale price threshold at which
    it switches from off to on.

    Three metrics are computed per BMU and combined:
      - switch_on_price: 10th percentile of wholesale prices when dispatching
      - max_off_price: 90th percentile of wholesale prices when NOT dispatching
      - estimated_mc: midpoint of switch_on_price and max_off_price

    For baseload generators (CF > 0.90), the switch-on price alone is used
    since they are rarely off. For peakers (CF < 0.10) the max_off_price is
    used since they are rarely on.

    BMUs are mapped to PyPSA generator names. Multiple BMUs for the same
    generator are aggregated (median).

    Parameters
    ----------
    pn_df : DataFrame
        Generation data. Index = datetime, columns = BMU IDs, values = MW.
    prices : Series
        Wholesale market prices (£/MWh) indexed by datetime.
    bmu_mapping_df : DataFrame, optional
        BMU mapping with columns: bmu_id, generator_name, carrier.
    logger : Logger

    Returns
    -------
    DataFrame with columns: generator, carrier, empirical_mc, n_bmus,
        total_dispatch_periods, avg_capacity_factor
    """
    # Build BMU → generator/carrier lookup
    bmu_to_gen = {}
    bmu_to_carrier = {}
    if bmu_mapping_df is not None and not bmu_mapping_df.empty:
        for _, row in bmu_mapping_df.iterrows():
            bmu_to_gen[row['bmu_id']] = row['generator_name']
            bmu_to_carrier[row['bmu_id']] = row.get('carrier', '')

    results = []

    for bmu_id in pn_df.columns:
        pn = pn_df[bmu_id]

        # Align with prices
        common_idx = pn.index.intersection(prices.index)
        if len(common_idx) < 10:  # Need at least 10 observations
            continue

        pn_aligned = pn.loc[common_idx]
        prices_aligned = prices.loc[common_idx]

        # Find dispatching periods (generation > 1 MW)
        dispatching = pn_aligned > 1.0
        n_dispatching = int(dispatching.sum())
        n_total = len(common_idx)
        n_off = n_total - n_dispatching
        cf = n_dispatching / n_total

        if n_dispatching < 5:
            continue  # Skip BMUs that barely generate

        # Compute price metrics
        on_prices = prices_aligned[dispatching]
        switch_on = float(on_prices.quantile(0.10))

        if n_off > 5:
            off_prices = prices_aligned[~dispatching]
            max_off = float(off_prices.quantile(0.90))
        else:
            max_off = np.nan

        # Estimate MC by capacity factor regime
        if cf > 0.90 or np.isnan(max_off):
            # Baseload: almost always on — use low-end of dispatch prices
            estimated_mc = switch_on
        elif cf < 0.10:
            # Peaker: almost always off — use high-end of off prices
            estimated_mc = max_off
        else:
            # Mid-merit: average ON floor and OFF ceiling
            estimated_mc = (switch_on + max_off) / 2.0

        # Map to generator name
        generator_name = bmu_to_gen.get(bmu_id, None)
        carrier = bmu_to_carrier.get(bmu_id, '')

        results.append({
            'bmu_id': bmu_id,
            'generator': generator_name,
            'carrier': carrier,
            'switch_on_price': switch_on,
            'max_off_price': max_off,
            'estimated_mc': estimated_mc,
            'dispatch_periods': n_dispatching,
            'capacity_factor': cf,
        })

    df = pd.DataFrame(results)

    if df.empty:
        logger.warning("No per-generator switch-on prices computed")
        return df

    logger.info(f"Computed switch-on prices for {len(df)} BMUs")

    # Separate mapped vs unmapped
    mapped = df[df['generator'].notna()].copy()
    unmapped = df[df['generator'].isna()].copy()

    if not unmapped.empty:
        logger.info(f"  {len(unmapped)} BMUs not mapped to generators:")
        for _, row in unmapped.head(10).iterrows():
            logger.info(f"    {row['bmu_id']}: £{row['estimated_mc']:.1f}/MWh (CF={row['capacity_factor']:.2f})")

    if mapped.empty:
        logger.warning("No BMUs mapped to generators!")
        return pd.DataFrame()

    # Aggregate: median estimated MC per generator
    agg = mapped.groupby(['generator', 'carrier']).agg(
        empirical_mc=('estimated_mc', 'median'),
        n_bmus=('bmu_id', 'count'),
        total_dispatch_periods=('dispatch_periods', 'sum'),
        avg_capacity_factor=('capacity_factor', 'mean'),
    ).reset_index()

    logger.info(f"Per-generator MCs for {len(agg)} generators:")
    for _, row in agg.sort_values('empirical_mc').iterrows():
        logger.info(f"  {row['generator']:30s} ({row['carrier']:10s}): "
                     f"£{row['empirical_mc']:7.1f}/MWh  "
                     f"({row['n_bmus']} BMUs, CF={row['avg_capacity_factor']:.2f})")

    return agg


def calibrate_per_generator(
    pn_file: str = None,
    prices_file: str = None,
    bmu_mapping_file: str = None,
    network_path: str = None,
    output_file: str = "data/market/generator_marginal_costs.csv",
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Compute per-generator empirical MCs from wholesale market data.

    Uses Market Index Data (MID) wholesale prices and B1610 actual generation
    to determine each generator's marginal cost via revealed preference.

    Supports two price data formats:
      - MID prices (mid_prices.csv): preferred, wholesale market index
      - System prices (system_prices.csv): fallback, BM system buy price

    Parameters
    ----------
    pn_file : str
        Path to generation data CSV (datetime index, BMU columns, MW values).
    prices_file : str
        Path to price CSV (datetime index, price column — MID or system).
    bmu_mapping_file : str, optional
        Path to BMU mapping CSV (bmu_id, generator_name, carrier).
    network_path : str, optional
        Path to PyPSA network .nc file (used to build BMU mapping directly
        from B1610 BMU IDs + station prefix matching).
    output_file : str
        Path to write per-generator MC CSV.
    logger : Logger

    Returns
    -------
    DataFrame with per-generator empirical MCs.
    """
    logger.info("=" * 80)
    logger.info("PER-GENERATOR MARGINAL COST CALIBRATION (WHOLESALE MARKET)")
    logger.info("=" * 80)

    # Load PN data
    if pn_file and Path(pn_file).exists():
        pn_df = pd.read_csv(pn_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded PN data: {pn_df.shape[0]} periods × {pn_df.shape[1]} BMUs")
    else:
        logger.error(f"PN data file not found: {pn_file}")
        return pd.DataFrame()

    # Load system prices
    if prices_file and Path(prices_file).exists():
        prices = pd.read_csv(prices_file, index_col=0, parse_dates=True).squeeze("columns")
        logger.info(f"Loaded system prices: {len(prices)} periods, "
                     f"mean £{prices.mean():.1f}/MWh")
    else:
        logger.error(f"System prices file not found: {prices_file}")
        return pd.DataFrame()

    # Build BMU → generator mapping
    bmu_mapping = None
    if bmu_mapping_file and Path(bmu_mapping_file).exists():
        bmu_mapping = pd.read_csv(bmu_mapping_file)
        logger.info(f"Loaded BMU mapping: {len(bmu_mapping)} entries")
    elif network_path and Path(network_path).exists():
        # Build mapping directly from B1610 BMU IDs + network generators
        bmu_mapping = _build_direct_bmu_mapping(
            bmu_ids=list(pn_df.columns),
            network_path=network_path,
            logger=logger,
        )
    else:
        logger.warning("No BMU mapping available — results will be BMU-level only")

    # Compute per-generator switch-on prices
    result = compute_per_generator_switch_on(pn_df, prices, bmu_mapping, logger)

    if not result.empty and output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        logger.info(f"Saved per-generator MCs: {len(result)} generators → {output_file}")

    return result


def _build_direct_bmu_mapping(
    bmu_ids: list,
    network_path: str,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Build BMU→generator mapping directly from B1610 BMU IDs and network generators.

    Uses the STATION_TO_BMU_PREFIX dictionary to match BMU prefixes to station
    names, then fuzzy-matches station names to actual PyPSA generator names.
    Only matches to thermal/storage generators (not renewables).

    Parameters
    ----------
    bmu_ids : list
        List of BMU IDs from B1610 data (e.g., ['T_PEMB-11', 'T_KEAD-1', ...])
    network_path : str
        Path to PyPSA network .nc file.
    logger : Logger

    Returns
    -------
    DataFrame with columns: bmu_id, generator_name, carrier
    """
    import pypsa
    from scripts.generators.build_bmu_mapping import BMU_PREFIX_TO_STATION

    network = pypsa.Network(network_path)

    # Station name → generator name aliases for cases where fuzzy matching fails
    # (e.g. 'EP SHB LTD' is abbreviated 'South Humber Bank')
    STATION_ALIASES = {
        'south humber': 'EP SHB LTD',
        'south humber bank': 'EP SHB LTD',
    }

    # Build generator lookup from network — only thermal/storage carriers
    thermal_carriers = {'CCGT', 'OCGT', 'nuclear', 'Coal', 'coal', 'Oil', 'oil',
                        'Biomass', 'biomass', 'CHP'}
    gen_lookup = {}  # lowercase station → (generator_name, carrier, p_nom)
    for gen_name in network.generators.index:
        carrier = network.generators.loc[gen_name, 'carrier']
        if carrier not in thermal_carriers:
            continue
        p_nom = network.generators.loc[gen_name, 'p_nom']
        entry = (gen_name, carrier, p_nom)
        # Build multiple lookup keys: full name, lowered, without agg suffix
        base_name = gen_name.split('__agg')[0].lower().strip()
        # Keep the larger generator when keys collide (avoids small backup
        # units shadowing the main plant, e.g. Grain OCGT vs Grain CHP)
        if base_name not in gen_lookup or p_nom > gen_lookup[base_name][2]:
            gen_lookup[base_name] = entry
        if gen_name.lower() not in gen_lookup or p_nom > gen_lookup[gen_name.lower()][2]:
            gen_lookup[gen_name.lower()] = entry

    # Also add storage units (pumped hydro)
    for su_name in network.storage_units.index:
        base_name = su_name.split('__agg')[0].lower().strip()
        carrier = network.storage_units.loc[su_name, 'carrier'] if 'carrier' in network.storage_units.columns else ''
        p_nom = network.storage_units.loc[su_name, 'p_nom'] if 'p_nom' in network.storage_units.columns else 0
        gen_lookup[base_name] = (su_name, carrier, p_nom)

    results = []
    matched = 0
    unmatched_prefixes = set()

    for bmu_id in bmu_ids:
        if not isinstance(bmu_id, str) or not bmu_id.startswith('T_'):
            continue

        # Extract core prefix: strip T_ then take first 4 chars
        core = bmu_id[2:]  # Remove T_
        station_name = None

        for prefix_len in [4, 5, 3]:
            candidate = core[:prefix_len].upper()
            if candidate in BMU_PREFIX_TO_STATION:
                station_name = BMU_PREFIX_TO_STATION[candidate]
                break

        if station_name is None:
            unmatched_prefixes.add(core[:4])
            continue

        # Try to find matching generator
        generator_name = None
        carrier = ''

        # 0. Check station name aliases first
        if station_name in STATION_ALIASES:
            alias = STATION_ALIASES[station_name].lower()
            if alias in gen_lookup:
                generator_name, carrier = gen_lookup[alias][0], gen_lookup[alias][1]

        # 1. Direct match on station name
        if generator_name is None and station_name in gen_lookup:
            generator_name, carrier = gen_lookup[station_name][0], gen_lookup[station_name][1]

        # 2. Partial match: prefer the largest generator (by p_nom) among
        #    all matches to avoid small backup units shadowing the main plant
        if generator_name is None:
            best_match = None
            best_pnom = -1
            for key, (gname, gcarrier, gpnom) in gen_lookup.items():
                if station_name.replace(' ', '') in key.replace(' ', ''):
                    if gpnom > best_pnom:
                        best_match = (gname, gcarrier)
                        best_pnom = gpnom
            if best_match:
                generator_name, carrier = best_match

        if generator_name:
            results.append({
                'bmu_id': bmu_id,
                'generator_name': generator_name,
                'carrier': carrier,
            })
            matched += 1

    df = pd.DataFrame(results)
    n_gens = df['generator_name'].nunique() if not df.empty else 0
    logger.info(f"Direct BMU mapping: {matched} BMU IDs → {n_gens} generators "
                f"(from {len(bmu_ids)} B1610 BMU IDs)")
    if unmatched_prefixes:
        logger.debug(f"  Unmatched prefixes: {sorted(unmatched_prefixes)[:20]}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Calibrate thermal MC correction factors")
    parser.add_argument("--year", type=int, default=2023, help="Calibration year")
    parser.add_argument("--pn-file", type=str, default=None, help="PN data CSV")
    parser.add_argument("--dah-file", type=str, default=None, help="DAH price CSV")
    parser.add_argument("--prices-file", type=str, default=None, help="System prices CSV")
    parser.add_argument("--bmu-mapping", type=str, default=None, help="BMU→carrier CSV")
    parser.add_argument("--network", type=str, default=None,
                        help="PyPSA network .nc file (for per-generator BMU mapping)")
    parser.add_argument(
        "--output",
        type=str,
        default="data/market/carrier_correction_factors.csv",
        help="Output CSV path",
    )
    parser.add_argument("--per-generator", action="store_true",
                        help="Compute per-generator MCs instead of carrier-level factors")

    # Per-generator ELEXON data fetching
    parser.add_argument("--fetch-data", action="store_true",
                        help="Fetch PN + system price data from ELEXON before calibrating")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for ELEXON data fetch (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for ELEXON data fetch (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default="resources/market/elexon",
                        help="Directory for fetched ELEXON data")

    args = parser.parse_args()

    # Optionally fetch ELEXON data first
    if args.fetch_data:
        if not args.start_date or not args.end_date:
            logger.error("--start-date and --end-date required with --fetch-data")
            return
        from scripts.market.elexon_data import retrieve_wholesale_calibration_data
        logger.info(f"Fetching wholesale calibration data: {args.start_date} to {args.end_date}")
        fetch_result = retrieve_wholesale_calibration_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.data_dir,
            logger=logger,
        )
        args.pn_file = fetch_result['pn_file']
        args.prices_file = fetch_result['prices_file']

    if args.per_generator:
        # Per-generator calibration mode
        pn_file = args.pn_file
        prices_file = args.prices_file or args.dah_file
        output = args.output
        if output == "data/market/carrier_correction_factors.csv":
            output = "data/market/generator_marginal_costs.csv"

        calibrate_per_generator(
            pn_file=pn_file,
            prices_file=prices_file,
            bmu_mapping_file=args.bmu_mapping,
            network_path=args.network,
            output_file=output,
            logger=logger,
        )
    else:
        # Carrier-level calibration (original mode)
        calibrate_correction_factors(
            year=args.year,
            pn_file=args.pn_file,
            dah_file=args.dah_file,
            bmu_mapping_file=args.bmu_mapping,
            output_file=args.output,
            logger=logger,
        )


if __name__ == "__main__":
    main()
