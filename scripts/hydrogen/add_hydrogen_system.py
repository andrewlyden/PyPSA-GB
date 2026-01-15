#!/usr/bin/env python3
"""
Add hydrogen system components to PyPSA network.

This script implements a simplified hydrogen sector coupling:
1. Adds a single GB-wide hydrogen bus
2. Adds electrolysis (Link: electricity → H2 bus)
3. Adds hydrogen storage (Store at H2 bus)
4. Converts H2 generators to Links (H2 bus → electricity)

This is a simplified "copper-plate" hydrogen network - all H2 assets share
a single storage pool. Future work will model regional hydrogen networks.

Key Parameters (based on literature):
- Electrolysis efficiency: 70% (LHV basis, including auxiliary loads)
- H2 turbine efficiency: 50% (combined cycle H2-ready turbines)
- Storage: Sized for 24h of generation capacity

Author: PyPSA-GB Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import logging
import warnings
from typing import Optional

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Suppress PyPSA warnings
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Import logging configuration
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("add_hydrogen_system")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


# =============================================================================
# HYDROGEN SYSTEM PARAMETERS
# =============================================================================

# Electrolysis parameters
ELECTROLYSIS_EFFICIENCY = 0.70  # 70% electrical → H2 (LHV basis)
ELECTROLYSIS_MARGINAL_COST = 0.0  # £/MWh_el (operating cost, excl. electricity)

# H2 power generation parameters
H2_TURBINE_EFFICIENCY = 0.50  # 50% H2 → electricity (combined cycle)
H2_TURBINE_MARGINAL_COST = 5.0  # £/MWh_el (maintenance, water, etc.)

# Hydrogen storage parameters
H2_STORAGE_HOURS = 168  # 1 week of H2 generation capacity (for seasonal flexibility)
H2_STORAGE_EFFICIENCY = 0.98  # 98% round-trip (compression/decompression losses)
H2_STORAGE_STANDING_LOSS = 0.0001  # 0.01% per hour (minimal for cavern storage)


def load_fes_electrolysis_capacity(
    fes_file: str,
    year: int,
    fes_scenario: str,
    logger: logging.Logger
) -> float:
    """
    Load electrolysis capacity from FES data.
    
    Args:
        fes_file: Path to FES data CSV
        year: Modelled year
        fes_scenario: FES pathway name (e.g., "Holistic Transition")
        logger: Logger instance
        
    Returns:
        Total electrolysis capacity in MW
    """
    logger.info(f"Loading FES electrolysis data for {fes_scenario} {year}")
    
    fes = pd.read_csv(fes_file)
    year_col = str(year)
    
    if year_col not in fes.columns:
        logger.warning(f"Year {year} not in FES data, using nearest available")
        year_cols = [c for c in fes.columns if c.isdigit()]
        year_col = min(year_cols, key=lambda x: abs(int(x) - year))
        logger.info(f"Using year column: {year_col}")
    
    # Filter for electrolysis in the specified pathway
    electrolysis = fes[
        (fes['Technology Detail'] == 'Hydrogen electrolysis') &
        (fes['FES Pathway'] == fes_scenario)
    ]
    
    if len(electrolysis) == 0:
        logger.warning(f"No electrolysis data found for {fes_scenario}")
        return 0.0
    
    total_capacity = electrolysis[year_col].sum()
    logger.info(f"FES electrolysis capacity for {fes_scenario} {year}: {total_capacity:,.1f} MW")
    
    return total_capacity


def add_hydrogen_system(
    network: pypsa.Network,
    electrolysis_capacity_mw: float,
    h2_generation_capacity_mw: float,
    logger: logging.Logger
) -> pypsa.Network:
    """
    Add hydrogen system components to the network.
    
    Creates a simplified "copper-plate" hydrogen network:
    - Single GB-wide H2 bus
    - Electrolysis links from electricity buses to H2 bus
    - H2 storage at the H2 bus
    - H2 power generation links from H2 bus to electricity buses
    
    Args:
        network: PyPSA network
        electrolysis_capacity_mw: Total electrolysis capacity (MW)
        h2_generation_capacity_mw: Total H2 generation capacity (MW)
        logger: Logger instance
        
    Returns:
        Modified network with hydrogen system
    """
    logger.info("=" * 60)
    logger.info("ADDING HYDROGEN SYSTEM")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Add hydrogen carrier
    # -------------------------------------------------------------------------
    if 'H2_gas' not in network.carriers.index:
        network.add("Carrier", "H2_gas", 
                    color="#FF69B4",  # Hot pink for hydrogen gas
                    nice_name="Hydrogen Gas",
                    co2_emissions=0.0)
        logger.info("Added H2_gas carrier")
    
    if 'electrolysis' not in network.carriers.index:
        network.add("Carrier", "electrolysis",
                    color="#8A2BE2",  # Blue violet
                    nice_name="Electrolysis",
                    co2_emissions=0.0)
        logger.info("Added electrolysis carrier")
    
    if 'H2_turbine' not in network.carriers.index:
        network.add("Carrier", "H2_turbine",
                    color="#FF1493",  # Deep pink
                    nice_name="H2 Power Generation",
                    co2_emissions=0.0)
        logger.info("Added H2_turbine carrier")
    
    # -------------------------------------------------------------------------
    # Step 2: Find existing H2 generators and their buses
    # -------------------------------------------------------------------------
    h2_generators = network.generators[network.generators.carrier == 'H2'].copy()
    
    if len(h2_generators) == 0:
        logger.warning("No H2 generators found in network - skipping hydrogen system")
        return network
    
    logger.info(f"Found {len(h2_generators)} H2 generators with {h2_generators.p_nom.sum():,.1f} MW total capacity")
    
    # Get unique electricity buses where H2 generators are located
    h2_buses = h2_generators['bus'].unique()
    logger.info(f"H2 generators located at {len(h2_buses)} unique electricity buses")
    
    # -------------------------------------------------------------------------
    # Step 3: Create single GB-wide hydrogen bus
    # -------------------------------------------------------------------------
    h2_bus_name = "GB_H2"
    
    # Use centroid of H2 generator locations for visualization
    h2_gen_coords = network.buses.loc[h2_buses, ['x', 'y']].mean()
    
    network.add("Bus", h2_bus_name,
                carrier="H2_gas",
                x=h2_gen_coords['x'],
                y=h2_gen_coords['y'],
                v_nom=1.0)  # Notional voltage for hydrogen
    
    logger.info(f"Added hydrogen bus: {h2_bus_name}")
    
    # -------------------------------------------------------------------------
    # Step 4: Add electrolysis links (electricity → H2)
    # -------------------------------------------------------------------------
    # Distribute electrolysis capacity proportionally to H2 generator capacity at each bus
    h2_gen_by_bus = h2_generators.groupby('bus')['p_nom'].sum()
    total_h2_gen = h2_gen_by_bus.sum()
    
    electrolysis_added = 0
    for elec_bus, h2_gen_capacity in h2_gen_by_bus.items():
        # Allocate electrolysis proportionally to H2 generation capacity
        share = h2_gen_capacity / total_h2_gen
        elec_capacity = electrolysis_capacity_mw * share
        
        if elec_capacity < 0.1:  # Skip tiny allocations
            continue
        
        link_name = f"electrolysis_{elec_bus}"
        
        network.add("Link", link_name,
                    bus0=elec_bus,          # Electricity input
                    bus1=h2_bus_name,       # Hydrogen output
                    carrier="electrolysis",
                    efficiency=ELECTROLYSIS_EFFICIENCY,
                    p_nom=elec_capacity,
                    p_nom_extendable=False,
                    marginal_cost=ELECTROLYSIS_MARGINAL_COST,
                    capital_cost=0.0)
        
        electrolysis_added += elec_capacity
    
    logger.info(f"Added {len(network.links[network.links.carrier == 'electrolysis'])} electrolysis links")
    logger.info(f"Total electrolysis capacity: {electrolysis_added:,.1f} MW")
    
    # -------------------------------------------------------------------------
    # Step 5: Add hydrogen storage
    # -------------------------------------------------------------------------
    # Size storage for H2_STORAGE_HOURS of full H2 generation
    h2_storage_energy = h2_generation_capacity_mw * H2_STORAGE_HOURS  # MWh (H2 LHV)
    
    network.add("Store", "GB_H2_storage",
                bus=h2_bus_name,
                carrier="H2_gas",
                e_nom=h2_storage_energy,
                e_nom_extendable=False,
                e_cyclic=True,  # SOC at end = SOC at start
                standing_loss=H2_STORAGE_STANDING_LOSS,
                marginal_cost=0.0,
                capital_cost=0.0)
    
    logger.info(f"Added H2 storage: {h2_storage_energy:,.0f} MWh ({H2_STORAGE_HOURS}h of generation)")
    
    # -------------------------------------------------------------------------
    # Step 6: Convert H2 generators to Links (H2 → electricity)
    # -------------------------------------------------------------------------
    # Remove existing H2 generators and replace with links
    h2_gen_data = h2_generators.copy()
    
    # Remove the generators
    network.remove("Generator", h2_generators.index.tolist())
    logger.info(f"Removed {len(h2_generators)} H2 generators (converting to links)")
    
    # Add as links (H2 → electricity)
    for idx, gen in h2_gen_data.iterrows():
        link_name = f"H2_turbine_{idx}"
        
        network.add("Link", link_name,
                    bus0=h2_bus_name,      # Hydrogen input
                    bus1=gen['bus'],       # Electricity output
                    carrier="H2_turbine",
                    efficiency=H2_TURBINE_EFFICIENCY,
                    p_nom=gen['p_nom'],    # Electrical output capacity
                    p_nom_extendable=False,
                    marginal_cost=H2_TURBINE_MARGINAL_COST,
                    capital_cost=0.0)
    
    h2_turbine_count = len(network.links[network.links.carrier == 'H2_turbine'])
    h2_turbine_capacity = network.links[network.links.carrier == 'H2_turbine']['p_nom'].sum()
    logger.info(f"Added {h2_turbine_count} H2 turbine links with {h2_turbine_capacity:,.1f} MW capacity")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("HYDROGEN SYSTEM SUMMARY")
    logger.info("-" * 60)
    logger.info(f"  Electrolysis capacity:  {electrolysis_added:,.1f} MW (η={ELECTROLYSIS_EFFICIENCY*100:.0f}%)")
    logger.info(f"  H2 storage capacity:    {h2_storage_energy:,.0f} MWh ({H2_STORAGE_HOURS}h)")
    logger.info(f"  H2 turbine capacity:    {h2_turbine_capacity:,.1f} MW (η={H2_TURBINE_EFFICIENCY*100:.0f}%)")
    logger.info(f"  Round-trip efficiency:  {ELECTROLYSIS_EFFICIENCY * H2_TURBINE_EFFICIENCY * 100:.1f}%")
    logger.info("-" * 60)
    
    return network


def main():
    """Main entry point when run as script."""
    
    # Check if running under Snakemake
    if 'snakemake' in globals():
        # Get inputs/outputs from Snakemake
        network_file = snakemake.input.network
        fes_file = snakemake.input.fes_data
        output_file = snakemake.output.network
        
        # Get parameters
        scenario = snakemake.params.scenario
        modelled_year = snakemake.params.modelled_year
        fes_scenario = snakemake.params.fes_scenario
        is_historical = snakemake.params.is_historical
        
        logger.info(f"Adding hydrogen system for scenario: {scenario}")
        logger.info(f"Modelled year: {modelled_year}, FES scenario: {fes_scenario}")
        
    else:
        # Standalone testing
        import argparse
        parser = argparse.ArgumentParser(description="Add hydrogen system to PyPSA network")
        parser.add_argument("--network", required=True, help="Input network file")
        parser.add_argument("--fes-data", required=True, help="FES data CSV")
        parser.add_argument("--output", required=True, help="Output network file")
        parser.add_argument("--year", type=int, default=2035, help="Modelled year")
        parser.add_argument("--fes-scenario", default="Holistic Transition", help="FES scenario")
        args = parser.parse_args()
        
        network_file = args.network
        fes_file = args.fes_data
        output_file = args.output
        modelled_year = args.year
        fes_scenario = args.fes_scenario
        is_historical = modelled_year <= 2024
    
    # Skip for historical scenarios (no hydrogen system in historical data)
    if is_historical:
        logger.info("Historical scenario - copying network without hydrogen system")
        import shutil
        shutil.copy(network_file, output_file)
        return
    
    # Load network
    logger.info(f"Loading network from: {network_file}")
    network = load_network(network_file)
    
    # Get H2 generation capacity from network (already added by thermal generator integration)
    h2_gen_capacity = network.generators[network.generators.carrier == 'H2']['p_nom'].sum()
    
    if h2_gen_capacity == 0:
        logger.warning("No H2 generators in network - skipping hydrogen system")
        save_network(network, output_file)
        return
    
    # Load electrolysis capacity from FES
    electrolysis_capacity = load_fes_electrolysis_capacity(
        fes_file, modelled_year, fes_scenario, logger
    )
    
    if electrolysis_capacity == 0:
        logger.warning("No electrolysis capacity in FES - using 2x H2 generation as default")
        electrolysis_capacity = h2_gen_capacity * 2  # Reasonable default
    
    # Add hydrogen system
    network = add_hydrogen_system(
        network=network,
        electrolysis_capacity_mw=electrolysis_capacity,
        h2_generation_capacity_mw=h2_gen_capacity,
        logger=logger
    )
    
    # Save network
    logger.info(f"Saving network to: {output_file}")
    save_network(network, output_file)
    logger.info("Hydrogen system integration complete")


if __name__ == "__main__":
    main()

