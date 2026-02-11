"""
Marginal Cost Calculations for PyPSA-GB - STANDALONE UTILITY

⚠️  NOT USED IN MAIN WORKFLOW ⚠️

This module provides standalone marginal cost calculation utilities.
The main workflow uses apply_marginal_costs.py instead.

WORKFLOW FILE (used in Snakemake pipeline):
    scripts/generators/apply_marginal_costs.py

THIS FILE (standalone utility):
    scripts/generators/marginal_costs.py

USE CASES FOR THIS FILE:
  - Standalone analysis of marginal costs
  - Debugging and testing calculations
  - External scripts requiring marginal cost functions

For normal workflow usage, configure marginal costs in:
  - config/defaults.yaml (project-wide defaults)
  - config/scenarios.yaml (scenario-specific overrides)

CLI USAGE:
    python scripts/generators/marginal_costs.py --network <path> --output <dir>
    python scripts/generators/marginal_costs.py --network resources/network/HT35_base.nc --scenario HT35
"""

import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
from scripts.utilities.logging_config import setup_logging

logger = setup_logging("marginal_costs")

# Carbon emission factors (kg CO2/MWh)
# Source: UK Parliament POST Note 383 - Carbon footprint of electricity generation
CARBON_EMISSION_FACTORS = {
    'coal': 846,  # kg CO2/MWh
    'gas': 488,   # kg CO2/MWh (average CCGT/OCGT)
    'oil': 533,   # kg CO2/MWh
    'biomass': 120,  # kg CO2/MWh (lifecycle emissions)
    'nuclear': 12,   # kg CO2/MWh (lifecycle emissions)
    'hydro': 24,     # kg CO2/MWh (lifecycle emissions)
    'wind': 11,      # kg CO2/MWh (lifecycle emissions)
    'solar': 48,     # kg CO2/MWh (lifecycle emissions)
    'battery': 0     # No direct emissions during operation
}

def load_fuel_prices(fuel_price_file: str = None) -> pd.DataFrame:
    """
    Load fuel price data.
    
    Parameters
    ----------
    fuel_price_file : str, optional
        Path to fuel price data file. If None, uses default values.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with fuel prices in £/MWh
    """
    if fuel_price_file and Path(fuel_price_file).exists():
        logger.info(f"Loading fuel prices from {fuel_price_file}")
        # Try to load existing fuel price data
        try:
            fuel_prices = pd.read_csv(fuel_price_file, index_col=0, parse_dates=True)
            return fuel_prices
        except Exception as e:
            logger.warning(f"Failed to load fuel prices from {fuel_price_file}: {e}")
    
    logger.info("Using default fuel prices")
    # Use default values based on recent UK averages (2023-2024)
    # Sources: BEIS, trading data, market reports
    default_prices = {
        'gas': 80.0,    # £/MWh (including efficiency adjustment)
        'coal': 95.0,   # £/MWh (including efficiency adjustment)  
        'oil': 150.0,   # £/MWh (including efficiency adjustment)
        'biomass': 65.0, # £/MWh
        'nuclear': 15.0, # £/MWh (marginal operating cost)
        'hydro': 5.0,    # £/MWh (very low marginal cost)
        'wind': 0.0,     # £/MWh (zero marginal cost)
        'solar': 0.0,    # £/MWh (zero marginal cost)
        'battery': 0.1   # £/MWh (minimal wear cost)
    }
    
    return default_prices

def load_carbon_prices(carbon_price_file: str = None, carbon_price: float = None) -> float:
    """
    Load carbon price data.
    
    Parameters
    ----------
    carbon_price_file : str, optional
        Path to carbon price time series
    carbon_price : float, optional
        Fixed carbon price in £/tonne CO2
        
    Returns
    -------
    float
        Carbon price in £/tonne CO2
    """
    if carbon_price is not None:
        logger.info(f"Using provided carbon price: £{carbon_price}/tonne CO2")
        return carbon_price
    
    if carbon_price_file and Path(carbon_price_file).exists():
        logger.info(f"Loading carbon prices from {carbon_price_file}")
        try:
            carbon_data = pd.read_csv(carbon_price_file, index_col=0, parse_dates=True)
            # Take mean if time series, or single value
            price = carbon_data.mean().iloc[0] if hasattr(carbon_data.mean(), 'iloc') else carbon_data.mean()
            logger.info(f"Loaded carbon price: £{price:.2f}/tonne CO2")
            return price
        except Exception as e:
            logger.warning(f"Failed to load carbon prices from {carbon_price_file}: {e}")
    
    # Default UK carbon price (ETS + carbon support price, ~2024 levels)
    default_carbon_price = 85.0  # £/tonne CO2
    logger.info(f"Using default carbon price: £{default_carbon_price}/tonne CO2")
    return default_carbon_price

def calculate_carbon_cost(carrier: str, carbon_price: float) -> float:
    """
    Calculate carbon cost component for a fuel type.
    
    Parameters
    ----------
    carrier : str
        Fuel/carrier type
    carbon_price : float
        Carbon price in £/tonne CO2
        
    Returns
    -------
    float
        Carbon cost in £/MWh
    """
    emission_factor = CARBON_EMISSION_FACTORS.get(carrier, 0)
    carbon_cost = (emission_factor * carbon_price) / 1000  # Convert kg to tonnes
    return carbon_cost

def compute_marginal_costs(network: pypsa.Network, 
                          fuel_prices: dict = None,
                          carbon_price: float = None,
                          include_carbon: bool = True) -> pd.DataFrame:
    """
    Compute marginal costs for all generators in the network.
    
    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with generators
    fuel_prices : dict, optional
        Dictionary of fuel prices by carrier (£/MWh)
    carbon_price : float, optional
        Carbon price in £/tonne CO2
    include_carbon : bool
        Whether to include carbon costs
        
    Returns
    -------
    pd.DataFrame
        DataFrame with marginal costs per generator
    """
    logger.info("Computing marginal costs for generators")
    
    if fuel_prices is None:
        fuel_prices = load_fuel_prices()
    
    if carbon_price is None:
        carbon_price = load_carbon_prices()
    
    # Get generators
    if network.generators.empty:
        logger.warning("No generators found in network")
        return pd.DataFrame()
    
    generators = network.generators.copy()
    logger.info(f"Computing marginal costs for {len(generators)} generators")
    
    marginal_costs = []
    
    for gen_name, gen in generators.iterrows():
        try:
            carrier = gen.get('carrier', 'unknown')
            efficiency = gen.get('efficiency', 1.0)
            
            # Get base fuel cost
            base_fuel_cost = fuel_prices.get(carrier, 0.0)
            
            # Adjust for efficiency if not already included in fuel price
            if efficiency > 0 and efficiency <= 1.0:
                fuel_cost_adjusted = base_fuel_cost / efficiency
            else:
                fuel_cost_adjusted = base_fuel_cost
                if efficiency <= 0 or efficiency > 1.0:
                    logger.warning(f"Generator {gen_name}: Invalid efficiency {efficiency}, using unadjusted fuel cost")
            
            # Calculate carbon cost
            carbon_cost = 0.0
            if include_carbon:
                carbon_cost = calculate_carbon_cost(carrier, carbon_price)
                # Adjust carbon cost for efficiency
                if efficiency > 0 and efficiency <= 1.0:
                    carbon_cost = carbon_cost / efficiency
            
            # Total marginal cost
            total_marginal_cost = fuel_cost_adjusted + carbon_cost
            
            # Use existing marginal_cost from generator if available and reasonable
            existing_cost = gen.get('marginal_cost', np.nan)
            if pd.notna(existing_cost) and existing_cost >= 0:
                # Use existing if it's reasonable, otherwise use calculated
                final_cost = existing_cost
                logger.debug(f"{gen_name}: Using existing marginal cost £{final_cost:.2f}/MWh")
            else:
                final_cost = total_marginal_cost
                logger.debug(f"{gen_name}: Calculated marginal cost £{final_cost:.2f}/MWh (fuel: £{fuel_cost_adjusted:.2f}, carbon: £{carbon_cost:.2f})")
            
            marginal_costs.append({
                'generator': gen_name,
                'carrier': carrier,
                'efficiency': efficiency,
                'fuel_cost': fuel_cost_adjusted,
                'carbon_cost': carbon_cost,
                'marginal_cost': final_cost,
                'p_nom': gen.get('p_nom', 0)
            })
            
        except Exception as e:
            logger.error(f"Failed to compute marginal cost for generator {gen_name}: {e}")
            # Use default or existing cost
            existing_cost = gen.get('marginal_cost', 100.0)  # Default fallback
            marginal_costs.append({
                'generator': gen_name,
                'carrier': gen.get('carrier', 'unknown'),
                'efficiency': gen.get('efficiency', 1.0),
                'fuel_cost': np.nan,
                'carbon_cost': np.nan,
                'marginal_cost': existing_cost,
                'p_nom': gen.get('p_nom', 0)
            })
    
    # Convert to DataFrame
    mc_df = pd.DataFrame(marginal_costs)
    mc_df.set_index('generator', inplace=True)
    
    # Log summary
    logger.info(f"Marginal cost summary:")
    logger.info(f"  Total generators: {len(mc_df)}")
    logger.info(f"  By carrier: {mc_df['carrier'].value_counts().to_dict()}")
    logger.info(f"  Cost range: £{mc_df['marginal_cost'].min():.2f} - £{mc_df['marginal_cost'].max():.2f}/MWh")
    logger.info(f"  Mean cost: £{mc_df['marginal_cost'].mean():.2f}/MWh")
    
    return mc_df

def update_network_marginal_costs(network: pypsa.Network, marginal_costs: pd.DataFrame):
    """
    Update the network with computed marginal costs.
    
    Parameters
    ----------
    network : pypsa.Network
        Network to update
    marginal_costs : pd.DataFrame
        Marginal costs by generator
    """
    logger.info("Updating network with marginal costs")
    
    # Update generator marginal costs
    for gen_name in marginal_costs.index:
        if gen_name in network.generators.index:
            network.generators.loc[gen_name, 'marginal_cost'] = marginal_costs.loc[gen_name, 'marginal_cost']
    
    updated_count = len(marginal_costs.index.intersection(network.generators.index))
    logger.info(f"Updated marginal costs for {updated_count} generators")

def save_marginal_costs(marginal_costs: pd.DataFrame, output_path: str, scenario: str = "default"):
    """
    Save marginal costs to CSV file.
    
    Parameters
    ----------
    marginal_costs : pd.DataFrame
        Marginal costs data
    output_path : str
        Output directory path
    scenario : str
        Scenario name for filename
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"marginal_costs_{scenario}.csv"
    filepath = output_dir / filename
    
    # Save with timestamp
    marginal_costs_with_meta = marginal_costs.copy()
    marginal_costs_with_meta.attrs = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'scenario': scenario
    }
    
    marginal_costs.to_csv(filepath)
    logger.info(f"Saved marginal costs to {filepath}")

def load_scenario_parameters(scenario_file: str = "config/scenarios_master.yaml", scenario: str = None):
    """Load scenario-specific parameters for marginal cost calculation."""
    try:
        import yaml
        with open(scenario_file, 'r') as f:
            scenarios = yaml.safe_load(f)
        
        if scenario and scenario in scenarios:
            scenario_params = scenarios[scenario]
            return {
                'carbon_price': scenario_params.get('carbon_price'),
                'fuel_prices': scenario_params.get('fuel_prices', {})
            }
    except Exception as e:
        logger.warning(f"Could not load scenario parameters: {e}")
    
    return {'carbon_price': None, 'fuel_prices': {}}

def main():
    """Main execution function for standalone use."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute marginal costs for PyPSA-GB network')
    parser.add_argument('--network', required=True, help='Path to PyPSA network file')
    parser.add_argument('--output', default='resources', help='Output directory')
    parser.add_argument('--scenario', default='default', help='Scenario name')
    parser.add_argument('--carbon-price', type=float, help='Carbon price in £/tonne CO2')
    parser.add_argument('--fuel-prices', help='Path to fuel prices CSV')
    parser.add_argument('--update-network', action='store_true', help='Update network file with marginal costs')
    
    args = parser.parse_args()
    
    try:
        # Load network
        logger.info(f"Loading network from {args.network}")
        network = pypsa.Network(args.network)
        
        # Load scenario parameters
        scenario_params = load_scenario_parameters(scenario=args.scenario)
        
        # Override with command line arguments
        carbon_price = args.carbon_price or scenario_params.get('carbon_price')
        fuel_prices = load_fuel_prices(args.fuel_prices)
        fuel_prices.update(scenario_params.get('fuel_prices', {}))
        
        # Compute marginal costs
        marginal_costs = compute_marginal_costs(
            network, 
            fuel_prices=fuel_prices,
            carbon_price=carbon_price
        )
        
        # Save results
        save_marginal_costs(marginal_costs, args.output, args.scenario)
        
        # Update network if requested
        if args.update_network:
            update_network_marginal_costs(network, marginal_costs)
            network.export_to_netcdf(args.network)
            logger.info(f"Updated network saved to {args.network}")
        
        logger.info("Marginal cost calculation completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to compute marginal costs: {e}")
        raise

if __name__ == "__main__":
    main()

