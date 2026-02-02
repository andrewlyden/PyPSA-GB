"""
Apply marginal costs to generators in PyPSA network.

This script computes time-varying marginal costs for thermal generators based on:
- Fuel prices (gas, coal, oil)
- Carbon prices (UK ETS + carbon support price)
- Generator efficiency
- Emission factors

The marginal costs are critical for optimization - without them, thermal
generators have zero cost, leading to unbounded optimization problems.
"""

import pypsa
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.utilities.logging_config import setup_logging

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

logger = setup_logging("apply_marginal_costs")

# Carriers that should NEVER have their marginal cost modified
# These are either already set correctly or should remain as-is
PROTECTED_CARRIERS = {
    'load_shedding',  # CRITICAL: Must keep VOLL cost, never overwrite
    'load shedding',
    'voll',
    'VOLL'
}

# Carbon emission factors (kg CO2/MWh_thermal)
# Source: UK Parliament POST Note 383 - Carbon footprint of electricity generation
CARBON_EMISSION_FACTORS = {
    'coal': 846,       # kg CO2/MWh
    'Coal': 846,
    'gas': 488,        # kg CO2/MWh (average CCGT/OCGT)
    'Gas': 488,
    'CCGT': 488,       # Combined Cycle Gas Turbine
    'OCGT': 488,       # Open Cycle Gas Turbine
    'oil': 533,        # kg CO2/MWh
    'Oil': 533,
    'biomass': 120,    # kg CO2/MWh (lifecycle - treated as low carbon)
    'Biomass': 120,
    'Biomass (dedicated)': 120,
    'Biomass (co-firing)': 120,
    'Bioenergy': 120,  # Alternative name for biomass
    'waste': 180,      # kg CO2/MWh (energy from waste)
    'Waste': 180,
    'waste_to_energy': 180,
    'landfill_gas': 240,  # Slightly higher than general waste
    'sewage_gas': 240,
    'biogas': 240,
    'advanced_biofuel': 120,
    'nuclear': 12,     # kg CO2/MWh (lifecycle only)
    'Nuclear': 12,
    'PWR': 12,         # Pressurized Water Reactor
    'AGR': 12,         # Advanced Gas-cooled Reactor
    'hydro': 24,       # kg CO2/MWh (lifecycle only)
    'Hydro': 24,
    'Large Hydro': 24,
    'large_hydro': 24,
    'Small Hydro': 24,
    'small_hydro': 24,
    'Hydro / pumped storage': 24,
    'Pumped Storage': 0,  # No emissions from pumping
    'wind': 11,        # kg CO2/MWh (lifecycle only)
    'Wind': 11,
    'Wind (Onshore)': 11,
    'Wind (Offshore)': 11,
    'wind_onshore': 11,
    'wind_offshore': 11,
    'solar': 48,       # kg CO2/MWh (lifecycle only)
    'Solar': 48,
    'solar_pv': 48,
    'Battery': 0,      # No direct emissions
    'battery': 0,
    'geothermal': 38,  # Lifecycle emissions
    'tidal_stream': 15,
    'shoreline_wave': 15,
    'Conventional Steam': 846,  # Assume coal-based
    'Conventional steam': 846,
    # Future/emerging technologies
    'H2': 0,               # Green hydrogen (negligible lifecycle emissions)
    'micro_CHP': 488,      # Natural gas basis (same as CCGT)
    'gas_engine': 600,     # Reciprocating engine (higher emissions than turbine)
    'fuel_cell': 0,        # Assume green hydrogen input
    'CHP': 488             # Combined Heat & Power (natural gas basis)
}

# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL FUEL AND CARBON PRICES (2010-2024)
# ══════════════════════════════════════════════════════════════════════════════
# Sources:
# - Gas: NBP Day-Ahead prices (BEIS/DESNZ Energy Price Statistics)
# - Coal: CIF ARA coal prices converted to £/MWh thermal
# - Carbon: EU ETS + UK Carbon Price Floor (from April 2013)
# ══════════════════════════════════════════════════════════════════════════════

HISTORICAL_FUEL_PRICES = {
    # Year: {gas, coal, oil, biomass} in £/MWh thermal
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
    2020: {'gas': 12.0, 'coal': 6.0, 'oil': 25.0, 'biomass': 35.0},  # COVID crash
    2021: {'gas': 45.0, 'coal': 15.0, 'oil': 50.0, 'biomass': 35.0},  # Energy crisis begins
    2022: {'gas': 80.0, 'coal': 25.0, 'oil': 70.0, 'biomass': 40.0},  # Peak energy crisis
    2023: {'gas': 40.0, 'coal': 15.0, 'oil': 55.0, 'biomass': 40.0},  # Prices moderating
    2024: {'gas': 35.0, 'coal': 12.0, 'oil': 50.0, 'biomass': 40.0},  # Current levels
}

HISTORICAL_CARBON_PRICES = {
    # Year: £/tonne CO2 (EU ETS + UK Carbon Price Floor from 2013)
    # Pre-2013: EU ETS only
    # 2013+: EU ETS + UK CPF (Carbon Price Floor)
    2010: 12.0,   # EU ETS only (~€14)
    2011: 10.0,   # EU ETS declining
    2012: 7.0,    # EU ETS crashed (~€7)
    2013: 21.0,   # EU ETS ~£5 + UK CPF £16
    2014: 21.0,   # EU ETS ~£5 + UK CPF £16
    2015: 25.0,   # EU ETS ~£7 + UK CPF £18
    2016: 23.0,   # EU ETS ~£5 + UK CPF £18
    2017: 23.0,   # EU ETS ~£5 + UK CPF £18
    2018: 33.0,   # EU ETS recovering ~£15 + UK CPF £18
    2019: 43.0,   # EU ETS ~£25 + UK CPF £18 (coal-to-gas switching)
    2020: 43.0,   # EU ETS ~£25 + UK CPF £18
    2021: 63.0,   # UK ETS launched at ~£45 + UK CPF £18
    2022: 88.0,   # UK ETS ~£70 + UK CPF £18
    2023: 68.0,   # UK ETS ~£50 + UK CPF £18
    2024: 85.0,   # UK ETS ~£65-70 + UK CPF ~£18
}


def get_historical_fuel_prices(year: int) -> dict:
    """
    Get historical fuel prices for a given year.
    
    Parameters
    ----------
    year : int
        The modelled year
        
    Returns
    -------
    dict
        Fuel prices in £/MWh thermal
    """
    if year in HISTORICAL_FUEL_PRICES:
        return HISTORICAL_FUEL_PRICES[year].copy()
    elif year < min(HISTORICAL_FUEL_PRICES.keys()):
        # Use earliest available
        return HISTORICAL_FUEL_PRICES[min(HISTORICAL_FUEL_PRICES.keys())].copy()
    else:
        # Use latest available
        return HISTORICAL_FUEL_PRICES[max(HISTORICAL_FUEL_PRICES.keys())].copy()


def get_historical_carbon_price(year: int) -> float:
    """
    Get historical carbon price for a given year.
    
    Parameters
    ----------
    year : int
        The modelled year
        
    Returns
    -------
    float
        Carbon price in £/tonne CO2
    """
    if year in HISTORICAL_CARBON_PRICES:
        return HISTORICAL_CARBON_PRICES[year]
    elif year < min(HISTORICAL_CARBON_PRICES.keys()):
        return HISTORICAL_CARBON_PRICES[min(HISTORICAL_CARBON_PRICES.keys())]
    else:
        return HISTORICAL_CARBON_PRICES[max(HISTORICAL_CARBON_PRICES.keys())]


def load_fuel_prices(scenario_config: dict) -> dict:
    """
    Load fuel prices with automatic source selection.

    Priority order:
    1. Scenario-specific override in scenarios.yaml (marginal_costs.fuel_prices)
    2. Historical lookup table (for modelled_year ≤ 2024)
    3. FES dynamic prices (if FES_year specified, use_fes_prices=true, and CSV exists)
    4. Configuration defaults from defaults.yaml (marginal_costs.fuel_prices)
    5. Fallback hardcoded values (for backward compatibility)

    Parameters
    ----------
    scenario_config : dict
        Scenario configuration dictionary (includes defaults merged in)

    Returns
    -------
    dict
        Fuel prices in £/MWh_thermal (before efficiency adjustment)
    """
    # Load marginal cost configuration
    mc_config = load_marginal_cost_config(scenario_config)

    # 1. Check if scenario explicitly provides fuel prices (highest priority override)
    if 'marginal_costs' in scenario_config and 'fuel_prices' in scenario_config['marginal_costs']:
        explicit_prices = scenario_config['marginal_costs']['fuel_prices']
        logger.info(f"Using scenario-specific fuel prices (explicit override):")
        for fuel, price in explicit_prices.items():
            logger.info(f"  {fuel}: £{price:.2f}/MWh thermal")
        return _expand_fuel_prices(explicit_prices)

    # 2. Check if this is a historical scenario
    modelled_year = scenario_config.get('modelled_year')

    if modelled_year and modelled_year <= 2024:
        # Historical scenario - use historical lookup
        historical_prices = get_historical_fuel_prices(modelled_year)
        logger.info(f"Using HISTORICAL fuel prices for {modelled_year}:")
        for fuel, price in historical_prices.items():
            logger.info(f"  {fuel}: £{price:.2f}/MWh thermal")
        return _expand_fuel_prices(historical_prices)

    # 3. Try FES dynamic prices for future scenarios
    if modelled_year and modelled_year > 2024 and mc_config['use_fes_prices']:
        fes_year = scenario_config.get('FES_year')
        fuel_price_file = scenario_config.get('_fuel_price_file')  # Passed from Snakemake params

        if fes_year:
            fes_prices = load_fes_fuel_prices(fes_year, modelled_year, fuel_price_file)
            if fes_prices:
                logger.info(f"Using FES DYNAMIC fuel prices for {modelled_year} (FES {fes_year})")
                return _expand_fuel_prices(fes_prices)
            else:
                logger.info(f"FES fuel prices not available, falling back to configured defaults")

    # 4. Use configured defaults from defaults.yaml
    logger.info(f"Using CONFIGURED DEFAULT fuel prices (future scenario):")
    for fuel, price in mc_config['fuel_prices'].items():
        logger.info(f"  {fuel}: £{price:.2f}/MWh thermal")
    return _expand_fuel_prices(mc_config['fuel_prices'])


def _expand_fuel_prices(base_prices: dict) -> dict:
    """
    Expand base fuel prices to include all carrier name variants.
    
    Parameters
    ----------
    base_prices : dict
        Base prices with keys: gas, coal, oil, biomass
        
    Returns
    -------
    dict
        Expanded prices with all carrier name variants
    """
    gas_price = base_prices.get('gas', 35.0)
    coal_price = base_prices.get('coal', 30.0)
    oil_price = base_prices.get('oil', 75.0)
    biomass_price = base_prices.get('biomass', 45.0)
    
    return {
        # Gas variants
        'gas': gas_price,
        'Gas': gas_price,
        'CCGT': gas_price,
        'OCGT': gas_price,
        
        # Coal variants
        'coal': coal_price,
        'Coal': coal_price,
        'Conventional Steam': coal_price,
        'Conventional steam': coal_price,
        
        # Oil variants
        'oil': oil_price,
        'Oil': oil_price,
        
        # Biomass/waste fuels (use biomass as base, adjust others)
        'biomass': biomass_price,
        'Biomass': biomass_price,
        'Biomass (dedicated)': biomass_price,
        'Biomass (co-firing)': biomass_price,
        'Bioenergy': biomass_price,
        'biogas': biomass_price * 0.9,      # Slightly cheaper
        'landfill_gas': biomass_price * 0.6,  # Waste gas is cheaper
        'sewage_gas': biomass_price * 0.7,
        'advanced_biofuel': biomass_price * 1.1,
        'waste_to_energy': biomass_price * 0.5,  # Low cost (paid to take waste)
        
        # Nuclear (very low marginal cost)
        'nuclear': 8.0,
        'Nuclear': 8.0,
        'PWR': 8.0,
        'AGR': 8.0,
        
        # Hydro (very low marginal cost)
        'hydro': 5.0,
        'Hydro': 5.0,
        'Large Hydro': 5.0,
        'large_hydro': 5.0,
        'Small Hydro': 5.0,
        'small_hydro': 5.0,
        'Hydro / pumped storage': 5.0,
        
        # Storage (minimal wear cost)
        'Pumped Storage': 0.1,
        'Battery': 0.1,
        'battery': 0.1,
        
        # Renewables (zero fuel cost, tiny O&M)
        'wind': 0.5,
        'Wind': 0.5,
        'Wind (Onshore)': 0.5,
        'Wind (Offshore)': 0.5,
        'wind_onshore': 0.5,
        'wind_offshore': 0.5,
        'solar': 0.5,
        'Solar': 0.5,
        'solar_pv': 0.5,
        
        # Marine renewables
        'geothermal': 3.0,
        'tidal_stream': 1.0,
        'shoreline_wave': 1.0,

        # Future/emerging technologies
        'H2': 50.0,                  # Hydrogen production cost
        'micro_CHP': gas_price,      # Same as CCGT/OCGT (natural gas)
        'gas_engine': gas_price,     # Natural gas fuel
        'fuel_cell': 50.0,           # High hydrogen cost
        'CHP': gas_price,            # Combined Heat & Power (was defaulting to 0, should be gas)
        'waste': biomass_price * 0.5 # Waste-to-energy (low/negative gate fee)
    }


def load_marginal_cost_config(scenario_config: dict) -> dict:
    """
    Load marginal cost configuration from scenario config (merged with defaults).

    Parameters
    ----------
    scenario_config : dict
        Scenario configuration dictionary (includes defaults.yaml merged in)

    Returns
    -------
    dict
        {'carbon_price': float, 'fuel_prices': dict, 'use_fes_prices': bool}
    """
    # Get marginal_costs block from scenario config (merged with defaults)
    mc_config = scenario_config.get('marginal_costs', {})

    # Fallback defaults (for backward compatibility if not in config)
    default_carbon = 85.0
    default_fuels = {
        'gas': 35.0, 'coal': 30.0, 'oil': 75.0, 'biomass': 45.0,
        'nuclear': 8.0, 'hydro': 5.0, 'wind': 0.5, 'solar': 0.5,
        'battery': 0.1
    }

    # Get carbon price (scenario override → config default → hardcoded)
    carbon_price = mc_config.get('carbon_price', default_carbon)

    # Get fuel prices (merge scenario overrides with defaults)
    config_fuels = mc_config.get('fuel_prices', {})
    fuel_prices = {**default_fuels, **config_fuels}

    # Get FES price usage flag
    use_fes_prices = mc_config.get('use_fes_prices', True)

    logger.debug("Marginal cost configuration:")
    logger.debug(f"  Carbon price: £{carbon_price:.2f}/tonne CO2")
    logger.debug(f"  Use FES prices: {use_fes_prices}")
    if config_fuels:
        logger.debug(f"  Fuel price overrides: {list(config_fuels.keys())}")

    return {
        'carbon_price': carbon_price,
        'fuel_prices': fuel_prices,
        'use_fes_prices': use_fes_prices
    }


def load_fes_fuel_prices(fes_year: int, modelled_year: int, fuel_price_file: str = None) -> dict:
    """
    Load dynamic fuel prices from FES extraction CSV files.

    Parameters
    ----------
    fes_year : int
        FES publication year (e.g., 2024)
    modelled_year : int
        Target year for prices (e.g., 2035)
    fuel_price_file : str, optional
        Path to FES fuel prices CSV. If None, constructs from fes_year.

    Returns
    -------
    dict
        Fuel prices by carrier {fuel: price_gbp_per_mwh_thermal}
    """
    from pathlib import Path

    if fuel_price_file is None:
        fuel_price_file = f"resources/marginal_costs/fuel_prices_{fes_year}.csv"

    fuel_price_path = Path(fuel_price_file)

    if not fuel_price_path.exists():
        logger.warning(f"FES fuel price file not found: {fuel_price_file}")
        return None

    try:
        # Read FES price CSV
        df = pd.read_csv(fuel_price_path)

        # Expected columns: [year, fuel, price_gbp_per_mwh_thermal]
        if 'year' not in df.columns or 'fuel' not in df.columns or 'price_gbp_per_mwh_thermal' not in df.columns:
            logger.warning(f"FES fuel price CSV has unexpected format: {df.columns.tolist()}")
            return None

        # Filter/interpolate for modelled_year
        fuel_prices = {}
        for fuel in df['fuel'].unique():
            fuel_df = df[df['fuel'] == fuel].sort_values('year')

            if len(fuel_df) == 0:
                continue

            # Exact match
            exact = fuel_df[fuel_df['year'] == modelled_year]
            if len(exact) > 0:
                fuel_prices[fuel] = float(exact.iloc[0]['price_gbp_per_mwh_thermal'])
                continue

            # Interpolate
            years = fuel_df['year'].values
            prices = fuel_df['price_gbp_per_mwh_thermal'].values

            if modelled_year < years.min():
                fuel_prices[fuel] = float(prices[0])
            elif modelled_year > years.max():
                fuel_prices[fuel] = float(prices[-1])
            else:
                fuel_prices[fuel] = float(np.interp(modelled_year, years, prices))

        logger.info(f"Loaded FES fuel prices for {modelled_year} from FES {fes_year}:")
        for fuel, price in fuel_prices.items():
            logger.info(f"  {fuel}: £{price:.2f}/MWh thermal")

        return fuel_prices

    except Exception as e:
        logger.warning(f"Failed to load FES fuel prices from {fuel_price_file}: {e}")
        return None


def load_fes_carbon_price(fes_year: int, modelled_year: int, carbon_price_file: str = None) -> float:
    """
    Load dynamic carbon price from FES extraction CSV.

    Parameters
    ----------
    fes_year : int
        FES publication year (e.g., 2024)
    modelled_year : int
        Target year for price (e.g., 2035)
    carbon_price_file : str, optional
        Path to FES carbon prices CSV. If None, constructs from fes_year.

    Returns
    -------
    float or None
        Carbon price in £/tonne CO2, or None if not available
    """
    from pathlib import Path

    if carbon_price_file is None:
        carbon_price_file = f"resources/marginal_costs/carbon_prices_{fes_year}.csv"

    carbon_price_path = Path(carbon_price_file)

    if not carbon_price_path.exists():
        logger.warning(f"FES carbon price file not found: {carbon_price_file}")
        return None

    try:
        # Read FES price CSV
        df = pd.read_csv(carbon_price_path)

        # Expected columns: [year, carbon_price_gbp_per_tco2]
        if 'year' not in df.columns or 'carbon_price_gbp_per_tco2' not in df.columns:
            logger.warning(f"FES carbon price CSV has unexpected format: {df.columns.tolist()}")
            return None

        df = df.sort_values('year')
        years = df['year'].values
        prices = df['carbon_price_gbp_per_tco2'].values

        # Exact match
        exact = df[df['year'] == modelled_year]
        if len(exact) > 0:
            carbon_price = float(exact.iloc[0]['carbon_price_gbp_per_tco2'])
            logger.info(f"Loaded FES carbon price for {modelled_year} from FES {fes_year}: £{carbon_price:.2f}/tonne CO2")
            return carbon_price

        # Interpolate
        if modelled_year < years.min():
            carbon_price = float(prices[0])
        elif modelled_year > years.max():
            carbon_price = float(prices[-1])
        else:
            carbon_price = float(np.interp(modelled_year, years, prices))

        logger.info(f"Interpolated FES carbon price for {modelled_year} from FES {fes_year}: £{carbon_price:.2f}/tonne CO2")
        return carbon_price

    except Exception as e:
        logger.warning(f"Failed to load FES carbon price from {carbon_price_file}: {e}")
        return None


def get_carbon_price(scenario_config: dict) -> float:
    """
    Get carbon price with automatic source selection.

    Priority order:
    1. Scenario-specific override in scenarios.yaml (marginal_costs.carbon_price)
    2. Historical lookup table (for modelled_year ≤ 2024)
    3. FES dynamic price (if FES_year specified, use_fes_prices=true, and CSV exists)
    4. Configuration default from defaults.yaml (marginal_costs.carbon_price)
    5. Fallback hardcoded value (for backward compatibility)

    Parameters
    ----------
    scenario_config : dict
        Scenario configuration dictionary (includes defaults merged in)

    Returns
    -------
    float
        Carbon price in £/tonne CO2
    """
    # Load marginal cost configuration
    mc_config = load_marginal_cost_config(scenario_config)

    # 1. Check if scenario explicitly provides carbon price (highest priority override)
    if 'marginal_costs' in scenario_config and 'carbon_price' in scenario_config['marginal_costs']:
        carbon_price = scenario_config['marginal_costs']['carbon_price']
        logger.info(f"Using scenario carbon price (explicit override): £{carbon_price:.2f}/tonne CO2")
        return carbon_price

    # 2. Check if this is a historical scenario
    modelled_year = scenario_config.get('modelled_year')

    if modelled_year and modelled_year <= 2024:
        # Historical scenario - use historical lookup
        historical_carbon = get_historical_carbon_price(modelled_year)
        logger.info(f"Using HISTORICAL carbon price for {modelled_year}: £{historical_carbon:.2f}/tonne CO2")
        return historical_carbon

    # 3. Try FES dynamic carbon price for future scenarios
    if modelled_year and modelled_year > 2024 and mc_config['use_fes_prices']:
        fes_year = scenario_config.get('FES_year')
        carbon_price_file = scenario_config.get('_carbon_price_file')  # Passed from Snakemake params

        if fes_year:
            fes_carbon = load_fes_carbon_price(fes_year, modelled_year, carbon_price_file)
            if fes_carbon is not None:
                logger.info(f"Using FES DYNAMIC carbon price for {modelled_year} (FES {fes_year})")
                return fes_carbon
            else:
                logger.info(f"FES carbon price not available, falling back to configured default")

    # 4. Use configured default from defaults.yaml
    logger.info(f"Using CONFIGURED DEFAULT carbon price (future scenario): £{mc_config['carbon_price']:.2f}/tonne CO2")
    return mc_config['carbon_price']


def calculate_marginal_cost(carrier: str, 
                           fuel_price: float,
                           carbon_price: float, 
                           efficiency: float = 1.0) -> tuple:
    """
    Calculate marginal cost for a generator.
    
    Parameters
    ----------
    carrier : str
        Generator carrier/fuel type
    fuel_price : float
        Fuel price in £/MWh_thermal
    carbon_price : float
        Carbon price in £/tonne CO2
    efficiency : float
        Generator efficiency (fraction, e.g., 0.5 for 50%)
        
    Returns
    -------
    tuple
        (fuel_cost_per_MWh_elec, carbon_cost_per_MWh_elec, total_marginal_cost)
    """
    # Ensure efficiency is valid
    if efficiency <= 0 or efficiency > 1.0:
        logger.warning(f"Invalid efficiency {efficiency} for {carrier}, using 1.0")
        efficiency = 1.0
    
    # Fuel cost per MWh electric = fuel_price_thermal / efficiency
    fuel_cost_elec = fuel_price / efficiency
    
    # Carbon cost per MWh electric
    emission_factor = CARBON_EMISSION_FACTORS.get(carrier, 0)
    # Carbon cost = (emission_factor kg/MWh) * (carbon_price £/tonne) / 1000
    carbon_cost_thermal = (emission_factor * carbon_price) / 1000  # £/MWh thermal
    carbon_cost_elec = carbon_cost_thermal / efficiency  # £/MWh electric
    
    # Total marginal cost
    total_marginal_cost = fuel_cost_elec + carbon_cost_elec
    
    return fuel_cost_elec, carbon_cost_elec, total_marginal_cost

def apply_marginal_costs_to_network(network: pypsa.Network,
                                    fuel_prices: dict,
                                    carbon_price: float) -> pd.DataFrame:
    """
    Apply marginal costs to all generators in the network.
    
    CRITICAL: Load shedding generators are NEVER modified - they must retain
    their VOLL (Value of Lost Load) cost to prevent unbounded optimization.
    
    Parameters
    ----------
    network : pypsa.Network
        Network with generators
    fuel_prices : dict
        Fuel prices by carrier (£/MWh thermal)
    carbon_price : float
        Carbon price (£/tonne CO2)
        
    Returns
    -------
    pd.DataFrame
        Breakdown of marginal costs by generator
    """
    logger.info("Computing marginal costs for generators")
    logger.info(f"Total generators: {len(network.generators)}")

    # Normalize carriers before processing
    if 'carrier' in network.generators.columns:
        unclassified_mask = network.generators['carrier'] == 'unclassified'
        if unclassified_mask.any():
            logger.warning(f"Normalizing {unclassified_mask.sum()} 'unclassified' generators to carrier 'OCGT'")
            network.generators.loc[unclassified_mask, 'carrier'] = 'OCGT'
    
    marginal_cost_data = []
    protected_count = 0
    unmapped_carriers = set()
    
    for gen_name, gen in network.generators.iterrows():
        carrier = gen.get('carrier', 'unknown')
        efficiency = gen.get('efficiency', 1.0)
        p_nom = gen.get('p_nom', 0)
        existing_mc = gen.get('marginal_cost', 0.0)
        
        # CRITICAL: Never modify protected carriers (especially load_shedding)
        if carrier in PROTECTED_CARRIERS:
            protected_count += 1
            marginal_cost_data.append({
                'generator': gen_name,
                'carrier': carrier,
                'p_nom_MW': p_nom,
                'efficiency': efficiency,
                'fuel_price_thermal': 0.0,
                'fuel_cost_electric': 0.0,
                'carbon_cost': 0.0,
                'marginal_cost_total': existing_mc,  # Keep existing VOLL
                'protected': True
            })
            continue
        
        # Get fuel price for this carrier
        fuel_price = fuel_prices.get(carrier, None)
        
        # Warn if carrier not mapped (but use 0.0 as fallback)
        if fuel_price is None:
            if carrier not in unmapped_carriers:
                unmapped_carriers.add(carrier)
                logger.warning(f"⚠️  Carrier '{carrier}' not in fuel_prices map - using £0/MWh")
            fuel_price = 0.0
        
        # Calculate marginal cost components
        fuel_cost, carbon_cost, total_mc = calculate_marginal_cost(
            carrier, fuel_price, carbon_price, efficiency
        )
        
        # Apply to network
        network.generators.loc[gen_name, 'marginal_cost'] = total_mc
        
        # Store breakdown
        marginal_cost_data.append({
            'generator': gen_name,
            'carrier': carrier,
            'p_nom_MW': p_nom,
            'efficiency': efficiency,
            'fuel_price_thermal': fuel_price,
            'fuel_cost_electric': fuel_cost,
            'carbon_cost': carbon_cost,
            'marginal_cost_total': total_mc,
            'protected': False
        })
    
    # Create summary DataFrame
    mc_df = pd.DataFrame(marginal_cost_data)
    
    # Report on protected generators
    if protected_count > 0:
        logger.info(f"\n✓ Protected {protected_count} generators from cost modification")
        protected_df = mc_df[mc_df['protected'] == True]
        for carrier in protected_df['carrier'].unique():
            carrier_data = protected_df[protected_df['carrier'] == carrier]
            count = len(carrier_data)
            voll = carrier_data['marginal_cost_total'].iloc[0]
            logger.info(f"  {carrier}: {count} units @ £{voll:,.0f}/MWh VOLL")
    
    # Warn about unmapped carriers
    if unmapped_carriers:
        logger.warning(f"\n⚠️  {len(unmapped_carriers)} carriers not mapped to fuel prices:")
        for carrier in sorted(unmapped_carriers):
            count = len(mc_df[mc_df['carrier'] == carrier])
            capacity = mc_df[mc_df['carrier'] == carrier]['p_nom_MW'].sum()
            logger.warning(f"  {carrier}: {count} units, {capacity:,.1f} MW → £0/MWh")
    
    # Log summary by carrier
    logger.info("\nMarginal cost summary by carrier:")
    logger.info("=" * 80)
    
    summary = mc_df[mc_df['protected'] == False].groupby('carrier').agg({
        'generator': 'count',
        'p_nom_MW': 'sum',
        'marginal_cost_total': ['mean', 'min', 'max']
    }).round(2)
    
    for carrier in summary.index:
        count = int(summary.loc[carrier, ('generator', 'count')])
        capacity = summary.loc[carrier, ('p_nom_MW', 'sum')]
        mean_mc = summary.loc[carrier, ('marginal_cost_total', 'mean')]
        min_mc = summary.loc[carrier, ('marginal_cost_total', 'min')]
        max_mc = summary.loc[carrier, ('marginal_cost_total', 'max')]
        
        logger.info(f"{carrier:20s}: {count:4d} units, {capacity:8.1f} MW, "
                   f"MC: £{mean_mc:6.2f}/MWh (£{min_mc:.2f}-£{max_mc:.2f})")
    
    logger.info("=" * 80)
    
    # Check for zero-cost generators (should only be renewables + storage)
    zero_cost = mc_df[(mc_df['marginal_cost_total'] == 0) & (mc_df['protected'] == False)]
    if len(zero_cost) > 0:
        logger.info(f"\nGenerators with ZERO marginal cost: {len(zero_cost)}")
        zero_carriers = zero_cost.groupby('carrier').size()
        for carrier, count in zero_carriers.items():
            logger.info(f"  {carrier}: {count} units")
    
    # Check for high-cost generators (thermal should be £50-150/MWh)
    thermal_carriers = ['CCGT', 'OCGT', 'Coal', 'Oil', 'gas', 'coal', 'oil', 
                       'Conventional Steam', 'Conventional steam']
    thermal_gens = mc_df[mc_df['carrier'].isin(thermal_carriers)]
    
    if len(thermal_gens) > 0:
        logger.info(f"\nThermal generator marginal costs:")
        for carrier in thermal_gens['carrier'].unique():
            carrier_gens = thermal_gens[thermal_gens['carrier'] == carrier]
            mean_mc = carrier_gens['marginal_cost_total'].mean()
            logger.info(f"  {carrier}: £{mean_mc:.2f}/MWh average")
    else:
        logger.warning("⚠️  No thermal generators found!")
    
    return mc_df

def main():
    """Main execution function."""
    global logger
    
    # Reinitialize logger with Snakemake log path
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "apply_marginal_costs"
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("APPLYING MARGINAL COSTS TO NETWORK")
    logger.info("=" * 80)
    
    try:
        # Load network
        input_path = snakemake.input.network
        logger.info(f"Loading network from: {input_path}")
        network = load_network(input_path, custom_logger=logger)
        
        logger.info(f"Network: {network.name}")
        logger.info(f"Buses: {len(network.buses)}")
        logger.info(f"Generators: {len(network.generators)}")
        
        # Get parameters
        scenario_config = snakemake.params.scenario_config

        # Add file paths to scenario_config if provided (for FES integration)
        if hasattr(snakemake.params, 'fuel_price_file') and snakemake.params.fuel_price_file:
            scenario_config['_fuel_price_file'] = snakemake.params.fuel_price_file
        if hasattr(snakemake.params, 'carbon_price_file') and snakemake.params.carbon_price_file:
            scenario_config['_carbon_price_file'] = snakemake.params.carbon_price_file

        # Load fuel prices and carbon price
        fuel_prices = load_fuel_prices(scenario_config)
        carbon_price = get_carbon_price(scenario_config)
        
        # Apply marginal costs
        marginal_cost_df = apply_marginal_costs_to_network(
            network, fuel_prices, carbon_price
        )
        
        # Save marginal cost breakdown
        output_csv = snakemake.output.marginal_costs_csv
        marginal_cost_df.to_csv(output_csv, index=False)
        logger.info(f"\nSaved marginal cost breakdown to: {output_csv}")
        
        # Save updated network
        output_network = snakemake.output.network
        save_network(network, output_network, custom_logger=logger)
        logger.info(f"Saved network with marginal costs to: {output_network}")
        
        logger.info("=" * 80)
        logger.info("✓ MARGINAL COSTS APPLIED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to apply marginal costs: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

