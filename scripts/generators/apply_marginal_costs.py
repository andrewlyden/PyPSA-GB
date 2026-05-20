"""
Apply marginal costs to generators in PyPSA network.

Simplified two-tier approach:
  1. Empirical MCs (from ELEXON calibration) when available
  2. Formula/subsidy fallback otherwise

Flat if/elif logic — no layers, no static prerun files.
"""

import pypsa
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network, save_network

logger = setup_logging("apply_marginal_costs")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PROTECTED_CARRIERS = {
    'load_shedding', 'load shedding', 'voll', 'VOLL',
    'demand response', 'demand_response',
}

THERMAL_CARRIERS = {
    'CCGT', 'OCGT', 'Coal', 'coal', 'Oil', 'oil',
    'Biomass', 'biomass', 'Bioenergy', 'CHP', 'micro_CHP',
    'gas_engine', 'waste', 'Waste', 'waste_to_energy',
    'landfill_gas', 'sewage_gas', 'biogas', 'advanced_biofuel',
    'Conventional Steam', 'Conventional steam',
    'Biomass (dedicated)', 'Biomass (co-firing)',
}

RENEWABLE_CARRIERS = {
    'wind_onshore', 'wind_offshore', 'solar_pv',
    'large_hydro', 'small_hydro', 'tidal_stream',
    'shoreline_wave', 'tidal_lagoon', 'marine',
}

# Biofuel carriers eligible for ROC subsidy deduction from their thermal MC.
# Dedicated biomass receives 1.5 ROC/MWh under Ofgem banding; biogas/landfill/
# sewage gas and advanced biofuels may also hold ROC accreditation.
# Waste-to-energy is excluded (gate-fee economics differ from ROC subsidy).
BIOFUEL_CARRIERS = {
    'biomass', 'Biomass', 'Biomass (dedicated)', 'Biomass (co-firing)',
    'Bioenergy', 'advanced_biofuel', 'biogas',
    'landfill_gas', 'sewage_gas',
}



# Carbon emission factors (kg CO2/MWh_thermal)
CARBON_EMISSION_FACTORS = {
    'coal': 846, 'Coal': 846, 'Conventional Steam': 846, 'Conventional steam': 846,
    'gas': 488, 'Gas': 488, 'CCGT': 488, 'OCGT': 488, 'CHP': 488, 'micro_CHP': 488,
    'gas_engine': 600,
    'oil': 533, 'Oil': 533,
    'biomass': 0, 'Biomass': 0, 'Biomass (dedicated)': 0, 'Biomass (co-firing)': 0,
    'Bioenergy': 0, 'advanced_biofuel': 0,
    'waste': 180, 'Waste': 180, 'waste_to_energy': 180,
    'landfill_gas': 0, 'sewage_gas': 0, 'biogas': 0,
    'nuclear': 12, 'Nuclear': 12, 'PWR': 12, 'AGR': 12,
    'hydro': 24, 'Hydro': 24, 'Large Hydro': 24, 'large_hydro': 24,
    'Small Hydro': 24, 'small_hydro': 24, 'Hydro / pumped storage': 24,
    'Pumped Storage': 0, 'Battery': 0, 'battery': 0,
    'wind': 11, 'Wind': 11, 'wind_onshore': 11, 'wind_offshore': 11,
    'Wind (Onshore)': 11, 'Wind (Offshore)': 11,
    'solar': 48, 'Solar': 48, 'solar_pv': 48,
    'geothermal': 38, 'tidal_stream': 15, 'shoreline_wave': 15,
    'H2': 0, 'fuel_cell': 0,
}


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL FUEL AND CARBON PRICES (2010-2024)
# ══════════════════════════════════════════════════════════════════════════════

HISTORICAL_FUEL_PRICES = {
    2010: {'gas': 20.0, 'coal': 10.0, 'oil': 45.0, 'biomass': 35.0},
    2011: {'gas': 22.0, 'coal': 12.0, 'oil': 50.0, 'biomass': 35.0},
    2012: {'gas': 25.0, 'coal': 12.0, 'oil': 55.0, 'biomass': 35.0},
    2013: {'gas': 27.0, 'coal': 10.0, 'oil': 55.0, 'biomass': 35.0},
    2014: {'gas': 22.0, 'coal': 8.0, 'oil': 50.0, 'biomass': 35.0},
    2015: {'gas': 17.0, 'coal': 7.0, 'oil': 35.0, 'biomass': 35.0},
    2016: {'gas': 15.0, 'coal': 8.0, 'oil': 30.0, 'biomass': 35.0},
    2017: {'gas': 17.0, 'coal': 10.0, 'oil': 35.0, 'biomass': 35.0},
    2018: {'gas': 25.0, 'coal': 12.0, 'oil': 45.0, 'biomass': 35.0},
    2019: {'gas': 20.0, 'coal': 8.0, 'oil': 40.0, 'biomass': 35.0},
    2020: {'gas': 12.0, 'coal': 6.0, 'oil': 25.0, 'biomass': 35.0},
    2021: {'gas': 45.0, 'coal': 15.0, 'oil': 50.0, 'biomass': 35.0},
    2022: {'gas': 80.0, 'coal': 25.0, 'oil': 70.0, 'biomass': 40.0},
    2023: {'gas': 40.0, 'coal': 15.0, 'oil': 55.0, 'biomass': 40.0},
    2024: {'gas': 35.0, 'coal': 12.0, 'oil': 50.0, 'biomass': 40.0},
}

HISTORICAL_CARBON_PRICES = {
    2010: 12.0, 2011: 10.0, 2012: 7.0, 2013: 21.0, 2014: 21.0,
    2015: 25.0, 2016: 23.0, 2017: 23.0, 2018: 33.0, 2019: 43.0,
    2020: 43.0, 2021: 63.0, 2022: 88.0, 2023: 68.0, 2024: 85.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# PRICE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def get_historical_fuel_prices(year: int) -> dict:
    """Get historical fuel prices, clamped to available range."""
    if year in HISTORICAL_FUEL_PRICES:
        return HISTORICAL_FUEL_PRICES[year].copy()
    elif year < min(HISTORICAL_FUEL_PRICES):
        return HISTORICAL_FUEL_PRICES[min(HISTORICAL_FUEL_PRICES)].copy()
    else:
        return HISTORICAL_FUEL_PRICES[max(HISTORICAL_FUEL_PRICES)].copy()


def get_historical_carbon_price(year: int) -> float:
    """Get historical carbon price, clamped to available range."""
    if year in HISTORICAL_CARBON_PRICES:
        return HISTORICAL_CARBON_PRICES[year]
    elif year < min(HISTORICAL_CARBON_PRICES):
        return HISTORICAL_CARBON_PRICES[min(HISTORICAL_CARBON_PRICES)]
    else:
        return HISTORICAL_CARBON_PRICES[max(HISTORICAL_CARBON_PRICES)]


def _expand_fuel_prices(base_prices: dict) -> dict:
    """Expand base fuel prices {gas, coal, oil, biomass} to all carrier variants."""
    gas = base_prices.get('gas', 35.0)
    coal = base_prices.get('coal', 30.0)
    oil = base_prices.get('oil', 75.0)
    bio = base_prices.get('biomass', 45.0)
    return {
        'gas': gas, 'Gas': gas, 'CCGT': gas, 'OCGT': gas,
        'CHP': gas, 'micro_CHP': gas, 'gas_engine': gas,
        'coal': coal, 'Coal': coal,
        'Conventional Steam': coal, 'Conventional steam': coal,
        'oil': oil, 'Oil': oil,
        'biomass': bio, 'Biomass': bio,
        'Biomass (dedicated)': bio, 'Biomass (co-firing)': bio,
        'Bioenergy': bio, 'biogas': bio * 0.9,
        'landfill_gas': bio * 0.6, 'sewage_gas': bio * 0.7,
        'advanced_biofuel': bio * 1.1,
        'waste': bio * 0.5, 'Waste': bio * 0.5, 'waste_to_energy': bio * 0.5,
        'nuclear': 8.0, 'Nuclear': 8.0, 'PWR': 8.0, 'AGR': 8.0,
        'hydro': 5.0, 'Hydro': 5.0, 'Large Hydro': 5.0, 'large_hydro': 5.0,
        'Small Hydro': 5.0, 'small_hydro': 5.0, 'Hydro / pumped storage': 5.0,
        'Pumped Storage': 0.1, 'Battery': 0.1, 'battery': 0.1,
        'wind': 0.5, 'Wind': 0.5, 'wind_onshore': 0.5, 'wind_offshore': 0.5,
        'Wind (Onshore)': 0.5, 'Wind (Offshore)': 0.5,
        'solar': 0.5, 'Solar': 0.5, 'solar_pv': 0.5,
        'geothermal': 3.0, 'tidal_stream': 1.0, 'shoreline_wave': 1.0,
        'H2': 50.0, 'fuel_cell': 50.0,
    }


def load_fuel_prices(scenario_config: dict) -> dict:
    """Load fuel prices: historical table > scenario override > FES > defaults."""
    mc_config = scenario_config.get('marginal_costs', {})
    modelled_year = scenario_config.get('modelled_year')

    # Historical scenarios: always use lookup table
    if modelled_year and modelled_year <= 2024:
        prices = get_historical_fuel_prices(modelled_year)
        logger.info(f"Using HISTORICAL fuel prices for {modelled_year}: "
                    f"gas={prices['gas']}, coal={prices['coal']}")
        return _expand_fuel_prices(prices)

    # Scenario explicit override
    if 'fuel_prices' in mc_config:
        logger.info("Using scenario-specific fuel prices")
        return _expand_fuel_prices(mc_config['fuel_prices'])

    # FES dynamic prices for future scenarios
    if modelled_year and modelled_year > 2024 and mc_config.get('use_fes_prices', True):
        fes_year = scenario_config.get('FES_year')
        fuel_file = scenario_config.get('_fuel_price_file')
        if fes_year:
            fes_prices = _load_fes_fuel_prices(fes_year, modelled_year, fuel_file)
            if fes_prices:
                logger.info(f"Using FES fuel prices for {modelled_year} (FES {fes_year})")
                return _expand_fuel_prices(fes_prices)

    # Config defaults
    default_fuels = mc_config.get('fuel_prices', {'gas': 35, 'coal': 30, 'oil': 75, 'biomass': 45})
    logger.info("Using configured DEFAULT fuel prices")
    return _expand_fuel_prices(default_fuels)


def get_carbon_price(scenario_config: dict) -> float:
    """Get carbon price: historical table > scenario override > FES > defaults."""
    mc_config = scenario_config.get('marginal_costs', {})
    modelled_year = scenario_config.get('modelled_year')

    if modelled_year and modelled_year <= 2024:
        price = get_historical_carbon_price(modelled_year)
        logger.info(f"Using HISTORICAL carbon price for {modelled_year}: {price:.2f}/tCO2")
        return price

    if 'carbon_price' in mc_config:
        return mc_config['carbon_price']

    if modelled_year and modelled_year > 2024 and mc_config.get('use_fes_prices', True):
        fes_year = scenario_config.get('FES_year')
        carbon_file = scenario_config.get('_carbon_price_file')
        if fes_year:
            fes_carbon = _load_fes_carbon_price(fes_year, modelled_year, carbon_file)
            if fes_carbon is not None:
                return fes_carbon

    return mc_config.get('carbon_price', 85.0)


def _load_fes_fuel_prices(fes_year, modelled_year, fuel_price_file=None):
    """Load FES fuel prices CSV, interpolating for modelled_year."""
    if fuel_price_file is None:
        fuel_price_file = f"resources/marginal_costs/fuel_prices_{fes_year}.csv"
    path = Path(fuel_price_file)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not {'year', 'fuel', 'price_gbp_per_mwh_thermal'}.issubset(df.columns):
            return None
        prices = {}
        for fuel in df['fuel'].unique():
            fdf = df[df['fuel'] == fuel].sort_values('year')
            exact = fdf[fdf['year'] == modelled_year]
            if len(exact) > 0:
                prices[fuel] = float(exact.iloc[0]['price_gbp_per_mwh_thermal'])
            else:
                prices[fuel] = float(np.interp(modelled_year, fdf['year'], fdf['price_gbp_per_mwh_thermal']))
        logger.info(f"Loaded FES fuel prices for {modelled_year} from FES {fes_year}")
        return prices
    except Exception as e:
        logger.warning(f"Failed to load FES fuel prices: {e}")
        return None


def _load_fes_carbon_price(fes_year, modelled_year, carbon_price_file=None):
    """Load FES carbon price CSV, interpolating for modelled_year."""
    if carbon_price_file is None:
        carbon_price_file = f"resources/marginal_costs/carbon_prices_{fes_year}.csv"
    path = Path(carbon_price_file)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not {'year', 'carbon_price_gbp_per_tco2'}.issubset(df.columns):
            return None
        df = df.sort_values('year')
        exact = df[df['year'] == modelled_year]
        if len(exact) > 0:
            return float(exact.iloc[0]['carbon_price_gbp_per_tco2'])
        return float(np.interp(modelled_year, df['year'], df['carbon_price_gbp_per_tco2']))
    except Exception as e:
        logger.warning(f"Failed to load FES carbon price: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FORMULA MC CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_marginal_cost(carrier: str, fuel_price: float,
                            carbon_price: float, efficiency: float = 1.0) -> tuple:
    """Calculate (fuel_cost_elec, carbon_cost_elec, total_mc) for a generator."""
    if efficiency <= 0 or efficiency > 1.0:
        efficiency = 1.0
    fuel_cost = fuel_price / efficiency
    emission_factor = CARBON_EMISSION_FACTORS.get(carrier, 0)
    carbon_cost = (emission_factor * carbon_price) / 1000 / efficiency
    return fuel_cost, carbon_cost, fuel_cost + carbon_cost


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_thermal_empirical_mc(filepath):
    """Load daily thermal empirical MCs. Returns DataFrame or None."""
    if not filepath or not Path(filepath).exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if 'generator' not in df.columns or 'empirical_mc' not in df.columns:
            return None
        logger.info(f"Loaded thermal empirical MCs: {df['generator'].nunique()} generators, "
                    f"{df['date'].nunique() if 'date' in df.columns else '?'} days")
        return df
    except Exception as e:
        logger.warning(f"Failed to load thermal empirical MCs: {e}")
        return None


def _load_renewable_empirical_mc(filepath):
    """Load renewable empirical MCs. Returns {generator: mc} dict."""
    if not filepath or not Path(filepath).exists():
        return {}
    try:
        df = pd.read_csv(filepath)
        if 'generator' not in df.columns or 'empirical_mc' not in df.columns:
            return {}
        mcs = dict(zip(df['generator'], df['empirical_mc']))
        logger.info(f"Loaded renewable empirical MCs: {len(mcs)} generators")
        return mcs
    except Exception as e:
        logger.warning(f"Failed to load renewable empirical MCs: {e}")
        return {}


def _get_roc_buyout_price(buyout_prices: dict, modelled_year: int) -> float:
    """Get ROC buyout price for a given year, interpolating if needed."""
    if not buyout_prices:
        return 0.0
    prices = {int(k): v for k, v in buyout_prices.items()}
    if modelled_year in prices:
        return prices[modelled_year]
    years = sorted(prices.keys())
    if modelled_year <= years[0]:
        return prices[years[0]]
    if modelled_year >= years[-1]:
        return prices[years[-1]]
    for i in range(len(years) - 1):
        if years[i] <= modelled_year <= years[i + 1]:
            frac = (modelled_year - years[i]) / (years[i + 1] - years[i])
            return prices[years[i]] + frac * (prices[years[i + 1]] - prices[years[i]])
    return 0.0


def _build_renewable_mc_lookup(renewable_mcs, network):
    """Match renewable empirical MC keys to network generator names.

    Strategy: direct name match, then substring match (both directions).
    Only matches generators with a renewable carrier.
    """
    if not renewable_mcs:
        return {}

    lookup = {}
    sorted_stations = sorted(renewable_mcs.keys(), key=len, reverse=True)

    for gen_name in network.generators.index:
        carrier = network.generators.at[gen_name, 'carrier']
        if carrier not in RENEWABLE_CARRIERS:
            continue

        # Direct match
        if gen_name in renewable_mcs:
            lookup[gen_name] = renewable_mcs[gen_name]
            continue

        # Substring match (both directions)
        base = gen_name.lower().split('__agg')[0].strip()
        base_ns = base.replace(' ', '').replace('_', '')
        for st in sorted_stations:
            st_ns = st.lower().replace(' ', '').replace('_', '')
            if st_ns in base_ns or base_ns in st_ns:
                lookup[gen_name] = renewable_mcs[st]
                break

    logger.info(f"Renewable MC lookup: {len(lookup)} matched "
                f"(from {len(renewable_mcs)} calibrated stations)")
    return lookup


def _compute_carrier_factors(thermal_daily_df, fuel_prices, carbon_price,
                             scenario_config=None):
    """Derive carrier correction factors from empirical data or external file.

    When carrier_correction_factors.enabled=True and a file is configured,
    loads pre-computed factors (e.g. from historical ELEXON calibration).
    Otherwise, computes inline from thermal daily empirical data (historical behaviour).

    Returns {carrier: factor} where factor = median(empirical) / formula.
    """
    mc_config = (scenario_config or {}).get('marginal_costs', {})
    ccf_config = mc_config.get('carrier_correction_factors', {})

    # ── External file mode (for future scenarios) ─────────────────────────────
    if ccf_config.get('enabled', False) and ccf_config.get('file'):
        return _load_carrier_factors_from_file(ccf_config, scenario_config)

    # ── Inline mode (for historical scenarios with empirical data) ────────────
    if thermal_daily_df is None or thermal_daily_df.empty:
        return {}

    CARRIER_EFF = {
        'CCGT': 0.49, 'OCGT': 0.35, 'Coal': 0.36, 'coal': 0.36,
        'Oil': 0.30, 'oil': 0.30, 'Biomass': 0.35, 'nuclear': 0.33,
    }

    carrier_medians = thermal_daily_df.groupby('carrier')['empirical_mc'].median()
    factors = {}
    for carrier, emp_mc in carrier_medians.items():
        eff = CARRIER_EFF.get(carrier)
        if eff is None:
            continue
        fp = fuel_prices.get(carrier, 0)
        _, _, formula_mc = calculate_marginal_cost(carrier, fp, carbon_price, eff)
        if formula_mc > 0:
            factors[carrier] = emp_mc / formula_mc
            logger.info(f"  Carrier factor {carrier}: {factors[carrier]:.3f} "
                        f"(empirical {emp_mc:.1f} / formula {formula_mc:.1f})")
    return factors


def _load_carrier_factors_from_file(ccf_config, scenario_config):
    """Load carrier correction factors from external CSV.

    Selection logic:
      - source_year="closest": pick year nearest to modelled_year
      - source_year="median": use aggregated median rows
      - source_year=<int>: use that specific year
    """
    # Prefer Snakemake-resolved path, fall back to config value
    filepath = Path(
        scenario_config.get('_correction_factors_file') or ccf_config['file']
    )
    if not filepath.exists():
        logger.warning(f"Carrier correction factors file not found: {filepath}")
        return {}

    fallback = ccf_config.get('fallback', 1.0)
    source_year = ccf_config.get('source_year', 'closest')
    modelled_year = (scenario_config or {}).get('modelled_year', 2035)

    try:
        df = pd.read_csv(filepath)
        if not {'source_year', 'carrier', 'correction_factor'}.issubset(df.columns):
            logger.warning(f"Carrier correction factors CSV missing required columns")
            return {}

        # Select rows based on source_year config
        if source_year == 'median':
            selected = df[df['source_year'] == 'median']
        elif source_year == 'closest':
            numeric_years = df[df['source_year'] != 'median']['source_year'].astype(int)
            if len(numeric_years) == 0:
                selected = df[df['source_year'] == 'median']
            else:
                closest_year = int(numeric_years.iloc[
                    (numeric_years - modelled_year).abs().argmin()
                ])
                selected = df[df['source_year'].astype(str) == str(closest_year)]
                logger.info(f"  Using correction factors from closest year: {closest_year}")
        else:
            # Specific year
            selected = df[df['source_year'].astype(str) == str(source_year)]
            if selected.empty:
                logger.warning(f"No correction factors for source_year={source_year}, "
                               f"falling back to median")
                selected = df[df['source_year'] == 'median']

        if selected.empty:
            logger.warning("No matching correction factors found")
            return {}

        factors = {}
        for _, row in selected.iterrows():
            carrier = row['carrier']
            factor = float(row['correction_factor'])
            # Skip nuclear — nuclear_override takes precedence
            if carrier in ('nuclear', 'Nuclear'):
                logger.info(f"  Skipping nuclear correction factor ({factor:.3f}) — "
                            f"nuclear_override takes precedence")
                continue
            factors[carrier] = factor
            logger.info(f"  Loaded carrier factor {carrier}: {factor:.3f} "
                        f"(from file, source_year={row['source_year']})")

        return factors

    except Exception as e:
        logger.warning(f"Failed to load carrier correction factors: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: APPLY MARGINAL COSTS
# ══════════════════════════════════════════════════════════════════════════════

def apply_marginal_costs_to_network(network, fuel_prices, carbon_price,
                                    scenario_config=None):
    """Apply marginal costs to all generators using flat two-tier logic.

    Priority for each generator:
      1. Protected carriers (load_shedding) -> skip
      2. Nuclear -> config.nuclear_mc
      3. Thermal + in thermal_empirical_mc -> daily empirical value
      4. Thermal + not in empirical -> formula x carrier_factor
      5. Renewable + in renewable_empirical_mc -> empirical value
      6. Renewable + subsidy fallback -> CfD=0, ROC=negative, merchant=0
      7. Everything else -> formula MC
    """
    logger.info("Applying marginal costs to generators")
    logger.info(f"Total generators: {len(network.generators)}")

    mc_config = (scenario_config or {}).get('marginal_costs', {})
    modelled_year = (scenario_config or {}).get('modelled_year', 2024)
    sub_config = mc_config.get('subsidies', {})

    # Nuclear MC
    nuclear_mc = mc_config.get('nuclear_mc', 10.0)

    # Load thermal empirical MCs
    thermal_mc_file = (scenario_config or {}).get('_thermal_mc_file')
    thermal_daily_df = _load_thermal_empirical_mc(thermal_mc_file)

    # Load renewable empirical MCs
    renewable_mc_file = (scenario_config or {}).get('_renewable_mc_file')
    renewable_mcs_raw = _load_renewable_empirical_mc(renewable_mc_file)
    renewable_mc_lookup = _build_renewable_mc_lookup(renewable_mcs_raw, network)

    # Compute carrier correction factors inline from thermal daily data
    carrier_factors = _compute_carrier_factors(thermal_daily_df, fuel_prices, carbon_price,
                                               scenario_config=scenario_config)

    # Build thermal empirical lookup: {generator: median_mc}
    thermal_mc_lookup = {}
    if thermal_daily_df is not None and not thermal_daily_df.empty:
        thermal_mc_lookup = (
            thermal_daily_df.groupby('generator')['empirical_mc']
            .median().to_dict()
        )
        logger.info(f"Thermal empirical MC lookup: {len(thermal_mc_lookup)} generators")

    # Subsidy config
    subsidy_enabled = sub_config.get('enabled', False)
    cfd_dispatch_cost = sub_config.get('cfd_dispatch_cost', 0.0)
    roc_buyout = 0.0
    if subsidy_enabled:
        buyout_prices = sub_config.get('roc_buyout_prices', {})
        roc_buyout = _get_roc_buyout_price(buyout_prices, modelled_year)
        logger.info(f"Subsidies enabled: CfD={cfd_dispatch_cost}, ROC buyout={roc_buyout:.2f}")

    # Biofuel subsidy config — deducts ROC income from thermal MC for biomass/biogas
    # carriers. Separate from the renewable subsidy toggle because biomass is thermal.
    biofuel_cfg = mc_config.get('biofuel_subsidy', {})
    biofuel_subsidy_enabled = biofuel_cfg.get('enabled', False)
    biofuel_roc_buyout = 0.0
    biofuel_default_bandings = {}
    biofuel_mc_floor = biofuel_cfg.get('mc_floor', 5.0)  # Prevent negative MC from ROC income
    if biofuel_subsidy_enabled:
        # Use explicit buyout price if given, else fall back to the renewable ROC table
        if 'roc_buyout_price' in biofuel_cfg:
            biofuel_roc_buyout = float(biofuel_cfg['roc_buyout_price'])
        else:
            buyout_prices = sub_config.get('roc_buyout_prices', {})
            biofuel_roc_buyout = _get_roc_buyout_price(buyout_prices, modelled_year)
        biofuel_default_bandings = biofuel_cfg.get('default_roc_bandings', {})
        logger.info(f"Biofuel subsidy enabled: ROC buyout={biofuel_roc_buyout:.2f}, "
                    f"bandings={biofuel_default_bandings}")

    # Normalize unclassified carriers
    if 'carrier' in network.generators.columns:
        mask = network.generators['carrier'] == 'unclassified'
        if mask.any():
            logger.warning(f"Normalizing {mask.sum()} 'unclassified' generators to 'OCGT'")
            network.generators.loc[mask, 'carrier'] = 'OCGT'

    # ── Main loop ────────────────────────────────────────────────────────────
    mc_data = []

    for gen_name, gen in network.generators.iterrows():
        carrier = gen.get('carrier', 'unknown')
        efficiency = gen.get('efficiency', 1.0)
        p_nom = gen.get('p_nom', 0)
        existing_mc = gen.get('marginal_cost', 0.0)

        # 1. Protected — never touch
        if carrier in PROTECTED_CARRIERS:
            mc_data.append({'generator': gen_name, 'carrier': carrier,
                            'p_nom_MW': p_nom, 'marginal_cost': existing_mc,
                            'source': 'protected'})
            continue

        # 2. Nuclear — fixed override
        if carrier in ('nuclear', 'Nuclear', 'PWR', 'AGR'):
            mc = nuclear_mc
            source = 'nuclear_override'

        # 3. Thermal with empirical MC
        elif gen_name in thermal_mc_lookup:
            mc = thermal_mc_lookup[gen_name]
            source = 'thermal_empirical'

        # 4. Thermal without empirical — formula x carrier factor
        elif carrier in THERMAL_CARRIERS:
            fp = fuel_prices.get(carrier, 0)
            _, _, formula_mc = calculate_marginal_cost(carrier, fp, carbon_price, efficiency)
            factor = carrier_factors.get(carrier, 1.0)
            mc = formula_mc * factor
            source = f'formula*{factor:.3f}' if factor != 1.0 else 'formula'

            # 4b. Biofuel ROC subsidy deduction — biomass/biogas plants receive
            # ROC income per MWh that reduces their effective wholesale MC.
            if biofuel_subsidy_enabled and carrier in BIOFUEL_CARRIERS:
                ro_banding = gen.get('ro_banding', None)
                if pd.isna(ro_banding) or ro_banding is None or ro_banding == 0:
                    ro_banding = biofuel_default_bandings.get(carrier, 0.0)
                if ro_banding and biofuel_roc_buyout > 0:
                    subsidy_income = float(ro_banding) * biofuel_roc_buyout
                    mc = mc - subsidy_income
                    # Floor: prevent ROC income making biomass cost-free/negative to the system
                    # Negative MC causes unconditional 100% dispatch which is unrealistic
                    mc = max(mc, biofuel_mc_floor)
                    source = f'{source}_ROC({ro_banding:.1f}x{biofuel_roc_buyout:.0f})'

        # 5. Renewable — subsidy type takes priority, then empirical fallback
        #    CfD generators always get cfd_dispatch_cost (typically £0) because
        #    their ELEXON BM bid prices reflect CfD opportunity cost, not SRMC.
        #    ROC generators always get negative MC from banding × buyout.
        #    Empirical MC only applies to merchant/unclassified renewables.
        elif carrier in RENEWABLE_CARRIERS:
            support_type = gen.get('support_type', None)
            if subsidy_enabled and support_type == 'CfD':
                mc = cfd_dispatch_cost
                source = 'subsidy_CfD'
            elif subsidy_enabled and support_type == 'ROC':
                ro_banding = gen.get('ro_banding', 0.0)
                if pd.notna(ro_banding) and ro_banding > 0:
                    mc = -(float(ro_banding) * roc_buyout)
                    source = 'subsidy_ROC'
                else:
                    mc = 0.0
                    source = 'renewable_default'
            elif gen_name in renewable_mc_lookup:
                mc = renewable_mc_lookup[gen_name]
                source = 'renewable_empirical'
            else:
                mc = 0.0
                source = 'renewable_default'

        # 7. Everything else — formula
        else:
            fp = fuel_prices.get(carrier, 0)
            _, _, mc = calculate_marginal_cost(carrier, fp, carbon_price, efficiency)
            source = 'formula'

        network.generators.loc[gen_name, 'marginal_cost'] = mc
        mc_data.append({'generator': gen_name, 'carrier': carrier,
                        'p_nom_MW': p_nom, 'marginal_cost': mc,
                        'source': source})

    mc_df = pd.DataFrame(mc_data)

    # ── Logging ──────────────────────────────────────────────────────────────
    active = mc_df[mc_df['source'] != 'protected']
    protected = mc_df[mc_df['source'] == 'protected']
    if len(protected):
        logger.info(f"Protected {len(protected)} generators (load_shedding etc.)")

    logger.info("\nMarginal cost summary:")
    logger.info("=" * 80)
    if len(active) > 0:
        summary = active.groupby('carrier').agg(
            count=('generator', 'count'),
            capacity=('p_nom_MW', 'sum'),
            mean_mc=('marginal_cost', 'mean'),
            min_mc=('marginal_cost', 'min'),
            max_mc=('marginal_cost', 'max'),
        ).round(2)
        for carrier in summary.index:
            r = summary.loc[carrier]
            logger.info(f"  {carrier:20s}: {int(r['count']):4d} units, {r['capacity']:8.1f} MW, "
                        f"MC: {r['mean_mc']:6.2f} ({r['min_mc']:.2f}-{r['max_mc']:.2f})")
    logger.info("=" * 80)

    # Source breakdown
    source_counts = active.groupby('source').size()
    logger.info("MC source breakdown:")
    for src, count in source_counts.items():
        logger.info(f"  {src}: {count} generators")

    return mc_df


# ══════════════════════════════════════════════════════════════════════════════
# TIME-VARYING MC (daily empirical)
# ══════════════════════════════════════════════════════════════════════════════

def apply_time_varying_mc(network, scenario_config, daily_mc_file=None):
    """Convert static MCs to hourly time series using daily empirical MCs."""
    mc_cfg = scenario_config.get('marginal_costs', {})
    emp_cfg = mc_cfg.get('empirical_calibration', {})

    daily_enabled = emp_cfg.get('enabled', False) and daily_mc_file

    if not daily_enabled:
        logger.info("Time-varying MC: disabled")
        return

    logger.info("=" * 60)
    logger.info("APPLYING TIME-VARYING MARGINAL COSTS")
    logger.info("=" * 60)

    snapshots = network.snapshots
    gen_names = network.generators.index
    static_mc = network.generators['marginal_cost']

    # Start with static MC broadcast
    mc_t = pd.DataFrame(
        np.tile(static_mc.values, (len(snapshots), 1)),
        index=snapshots, columns=gen_names, dtype=float,
    )

    # ── Daily empirical MC override ──────────────────────────────────────────
    # Nuclear carriers must be excluded: their ELEXON bids reflect strategic
    # BM behaviour (high bids to avoid turn-down), not short-run marginal cost.
    # The nuclear_override (Layer 2) must take precedence.
    nuclear_carriers = {'nuclear', 'Nuclear', 'PWR', 'AGR'}
    gen_carriers = network.generators['carrier']
    skip_nuclear = set(gen_carriers[gen_carriers.isin(nuclear_carriers)].index)

    try:
        daily_df = pd.read_csv(daily_mc_file)
        if 'date' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
            daily_lookup = daily_df.set_index(['generator', 'date'])['empirical_mc']
            snap_dates = pd.Series(snapshots).dt.date.values

            overridden = 0
            skipped_nuclear = 0
            for gen in gen_names:
                if gen in skip_nuclear:
                    skipped_nuclear += 1
                    continue
                for d in sorted(set(snap_dates)):
                    if (gen, d) in daily_lookup.index:
                        emp_mc = daily_lookup.loc[(gen, d)]
                        if pd.notna(emp_mc) and emp_mc > 0:
                            mask = snap_dates == d
                            mc_t.loc[mask, gen] = emp_mc
                            overridden += 1

            logger.info(f"  Daily MC: overrode {overridden} (generator, date) cells")
            if skipped_nuclear:
                logger.info(f"  Preserved nuclear_override for {skipped_nuclear} nuclear generators")
    except Exception as e:
        logger.warning(f"  Failed to load daily MCs: {e}")

    # ── Carrier MC adjustments ───────────────────────────────────────────────
    # Additive corrections per carrier applied AFTER empirical calibration.
    # Primary use: widen coal-gas gap that empirical switch-on methodology
    # compresses (both fuels turn on at similar wholesale prices, but coal has
    # higher CO₂ costs that the switch-on method does not capture).
    adjustments = emp_cfg.get('carrier_mc_adjustments', {})
    if adjustments:
        gen_carriers = network.generators['carrier']
        adj_count = 0
        for carrier, adj_value in adjustments.items():
            adj_value = float(adj_value)
            carrier_mask = gen_carriers == carrier
            if not carrier_mask.any():
                continue
            carrier_gens = gen_carriers[carrier_mask].index
            mc_t[carrier_gens] = mc_t[carrier_gens] + adj_value
            adj_count += len(carrier_gens)
            logger.info(
                f"  MC adjustment: {carrier} += {adj_value:+.1f} £/MWh "
                f"({len(carrier_gens)} generators)"
            )
        if adj_count:
            logger.info(f"  Applied MC adjustments to {adj_count} generators")

    network.generators_t['marginal_cost'] = mc_t
    logger.info(f"  Result: {mc_t.shape[0]} snapshots x {mc_t.shape[1]} generators")
    logger.info(f"  MC range: {mc_t.min().min():.2f}-{mc_t.max().max():.2f}/MWh")


# ══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Snakemake entry point."""
    global logger

    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "apply_marginal_costs"
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("APPLYING MARGINAL COSTS TO NETWORK")
    logger.info("=" * 80)

    try:
        network = load_network(snakemake.input.network, custom_logger=logger)
        scenario_config = snakemake.params.scenario_config

        # Pass file paths through scenario_config for the apply function
        if hasattr(snakemake.params, 'fuel_price_file') and snakemake.params.fuel_price_file:
            scenario_config['_fuel_price_file'] = snakemake.params.fuel_price_file
        if hasattr(snakemake.params, 'carbon_price_file') and snakemake.params.carbon_price_file:
            scenario_config['_carbon_price_file'] = snakemake.params.carbon_price_file
        if hasattr(snakemake.params, 'thermal_mc_file') and snakemake.params.thermal_mc_file:
            scenario_config['_thermal_mc_file'] = snakemake.params.thermal_mc_file
        if hasattr(snakemake.params, 'renewable_mc_file') and snakemake.params.renewable_mc_file:
            scenario_config['_renewable_mc_file'] = snakemake.params.renewable_mc_file
        if hasattr(snakemake.params, 'correction_factors_file') and snakemake.params.correction_factors_file:
            scenario_config['_correction_factors_file'] = snakemake.params.correction_factors_file

        fuel_prices = load_fuel_prices(scenario_config)
        carbon_price = get_carbon_price(scenario_config)

        mc_df = apply_marginal_costs_to_network(
            network, fuel_prices, carbon_price, scenario_config
        )

        # Time-varying MC (daily empirical base)
        thermal_mc_file = getattr(snakemake.params, 'thermal_mc_file', None)
        apply_time_varying_mc(network, scenario_config, daily_mc_file=thermal_mc_file)

        # Save outputs
        mc_df.to_csv(snakemake.output.marginal_costs_csv, index=False)
        save_network(network, snakemake.output.network, custom_logger=logger)

        logger.info("=" * 80)
        logger.info("MARGINAL COSTS APPLIED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to apply marginal costs: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
