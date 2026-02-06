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
    'VOLL',
    'demand response',  # DSR marginal cost set by event_flex.py (incentive payment)
    'demand_response',
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
    'CHP': 488,             # Gas-fired CHP
    'gas_engine': 488,      # Gas engine
    'marine': 15,           # Tidal/wave lifecycle
    'H2': 0,               # No direct emissions (hydrogen combustion = water)
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
    Load fuel prices - automatically uses historical values for historical scenarios.
    
    Logic:
    1. If scenario explicitly provides fuel_prices, use those
    2. If modelled_year <= 2024 (historical), use historical lookup table
    3. Otherwise, use default future prices
    
    Parameters
    ----------
    scenario_config : dict
        Scenario configuration dictionary
        
    Returns
    -------
    dict
        Fuel prices in £/MWh_thermal (before efficiency adjustment)
    """
    # Check if scenario explicitly provides fuel prices (override)
    fuel_prices = scenario_config.get('fuel_prices', {})
    
    if fuel_prices:
        logger.info(f"Using scenario-specific fuel prices (explicit override): {fuel_prices}")
        return _expand_fuel_prices(fuel_prices)
    
    # Check if this is a historical scenario
    modelled_year = scenario_config.get('modelled_year')
    
    if modelled_year and modelled_year <= 2024:
        # Historical scenario - use historical lookup
        historical_prices = get_historical_fuel_prices(modelled_year)
        logger.info(f"Using HISTORICAL fuel prices for {modelled_year}:")
        for fuel, price in historical_prices.items():
            logger.info(f"  {fuel}: £{price:.2f}/MWh thermal")
        return _expand_fuel_prices(historical_prices)
    
    # Future scenario or no year specified - use defaults
    logger.info("Using DEFAULT fuel prices (future scenario):")
    default_prices = {
        'gas': 35.0,
        'coal': 30.0,
        'oil': 75.0,
        'biomass': 45.0
    }
    for fuel, price in default_prices.items():
        logger.info(f"  {fuel}: £{price:.2f}/MWh thermal")
    return _expand_fuel_prices(default_prices)


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
        'marine': 1.0,
        
        # CHP / gas engines (gas-fuelled, similar to CCGT/OCGT)
        'CHP': gas_price,
        'gas_engine': gas_price,
        
        # Hydrogen (future fuel, assumed marginal cost similar to gas)
        'H2': gas_price * 1.5,  # Hydrogen currently ~50% more expensive than gas
    }


def get_carbon_price(scenario_config: dict) -> float:
    """
    Get carbon price - automatically uses historical values for historical scenarios.
    
    Logic:
    1. If scenario explicitly provides carbon_price, use that
    2. If modelled_year <= 2024 (historical), use historical lookup table
    3. Otherwise, use default future price
    
    Parameters
    ----------
    scenario_config : dict
        Scenario configuration dictionary
        
    Returns
    -------
    float
        Carbon price in £/tonne CO2
    """
    # Check if scenario explicitly provides carbon price (override)
    carbon_price = scenario_config.get('carbon_price')
    
    if carbon_price is not None:
        logger.info(f"Using scenario carbon price (explicit override): £{carbon_price:.2f}/tonne CO2")
        return carbon_price
    
    # Check if this is a historical scenario
    modelled_year = scenario_config.get('modelled_year')
    
    if modelled_year and modelled_year <= 2024:
        # Historical scenario - use historical lookup
        historical_carbon = get_historical_carbon_price(modelled_year)
        logger.info(f"Using HISTORICAL carbon price for {modelled_year}: £{historical_carbon:.2f}/tonne CO2")
        return historical_carbon
    
    # Future scenario - use default
    default_carbon_price = 85.0  # £/tonne CO2
    logger.info(f"Using DEFAULT carbon price (future scenario): £{default_carbon_price:.2f}/tonne CO2")
    return default_carbon_price


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

