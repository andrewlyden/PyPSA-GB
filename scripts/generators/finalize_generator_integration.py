"""
Finalize Generator Integration - Stage 3 of Generator Integration Pipeline

This script finalizes the generator integration process by:
1. Adding load shedding generators (backup power at VoLL)
2. Exporting comprehensive generator data to CSV files
3. Creating summary statistics by carrier type
4. Generating technology capacity summaries
5. Creating integration reports
6. Generating HTML visualization maps

Author: PyPSA-GB Development Team
Date: 2025-10-28
"""

import sys
import time
import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pypsa
import folium
from folium.plugins import MarkerCluster

# Add project root to path for imports BEFORE other imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Suppress PyPSA warnings about unoptimized networks
warnings.filterwarnings('ignore', message='.*has not been optimized yet.*')

from scripts.utilities.logging_config import setup_logging

# Get logger for this script
logger = setup_logging("finalize_generator_integration")


def add_load_shedding_generators(n, voll=6000.0, capacity_margin=1.2):
    """
    Add load shedding generators to every bus as backup power source.
    
    These generators act as a last resort to meet demand when the system
    cannot otherwise balance. They have a very high marginal cost (VoLL)
    to ensure they are only used when absolutely necessary.
    
    **CRITICAL FIX**: Capacity is now sized based on ACTUAL peak load at each bus,
    not hardcoded to 10 GW. This prevents massive over-provisioning that was causing
    load shedding to be used as cheap generation instead of last resort.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to add load shedding generators to
    voll : float
        Value of Lost Load in £/MWh (default 6000.0)
    capacity_margin : float
        Safety margin above peak load (default 1.2 = 20% margin)
    
    Returns
    -------
    int
        Number of load shedding generators added
    """
    logger.info("=" * 80)
    logger.info("Adding load shedding generators (backup power) - SIZED BY PEAK LOAD")
    logger.info("=" * 80)
    
    # Ensure load_shedding carrier exists
    if 'load_shedding' not in n.carriers.index:
        n.add("Carrier",
              "load_shedding",
              nice_name="Load Shedding",
              color="#FF0000",
              co2_emissions=0)
        logger.info("Added 'load_shedding' carrier to network")
    
    # Calculate peak demand at each bus from load time series
    logger.info("Calculating peak demand at each bus from time series data...")
    bus_peak_demand = {}
    
    for bus_id in n.buses.index:
        # Get all loads at this bus
        bus_loads = n.loads[n.loads.bus == bus_id]
        
        if len(bus_loads) > 0:
            # Sum demand time series for all loads at this bus
            bus_demand_ts = n.loads_t.p_set[bus_loads.index].sum(axis=1)
            bus_peak_demand[bus_id] = bus_demand_ts.max()
        else:
            bus_peak_demand[bus_id] = 0.0
    
    # Get statistics
    total_peak = sum(bus_peak_demand.values())
    buses_with_load = sum(1 for v in bus_peak_demand.values() if v > 0.1)
    
    logger.info(f"Peak demand analysis:")
    logger.info(f"  Total system peak: {total_peak:,.0f} MW")
    logger.info(f"  Buses with load: {buses_with_load} / {len(n.buses)}")
    logger.info(f"VoLL (marginal cost): £{voll:,.0f}/MWh")
    logger.info(f"Capacity margin: {capacity_margin:.1f}× peak load")
    
    # Prepare generator data
    load_shedding_gens = []
    total_capacity = 0.0
    sized_by_load = 0
    sized_by_default = 0
    
    for bus in n.buses.index:
        gen_name = f"load_shedding_{bus}"
        peak_load = bus_peak_demand.get(bus, 0.0)
        
        # Size capacity based on actual peak load at this bus
        if peak_load > 0.1:  # Has meaningful load
            p_nom = peak_load * capacity_margin
            sized_by_load += 1
        else:  # No load or negligible load
            p_nom = 10.0  # Minimal capacity (10 MW, not 10 GW!)
            sized_by_default += 1
        
        total_capacity += p_nom
        
        load_shedding_gens.append({
            'name': gen_name,
            'bus': bus,
            'carrier': 'load_shedding',
            'p_nom': p_nom,
            'marginal_cost': voll,  # High cost ensures it's only used as last resort
            'committable': False,
            'p_nom_extendable': False
        })
    
    # Add generators
    for gen in load_shedding_gens:
        n.add("Generator",
              gen['name'],
              bus=gen['bus'],
              carrier=gen['carrier'],
              p_nom=gen['p_nom'],
              marginal_cost=gen['marginal_cost'],
              committable=gen['committable'],
              p_nom_extendable=gen['p_nom_extendable'])
    
    logger.info(f"[OK] Added {len(load_shedding_gens)} load shedding generators")
    logger.info(f"  Sized by load: {sized_by_load} buses")
    logger.info(f"  Sized by default (10 MW): {sized_by_default} buses")
    logger.info(f"  Total backup capacity: {total_capacity:,.0f} MW")
    if total_peak > 0:
        logger.info(f"  Ratio (capacity / peak demand): {total_capacity / total_peak:.2f}x")
    else:
        logger.info(f"  Ratio (capacity / peak demand): N/A (no load time series data)")
    logger.info(f"  [OK] Properly sized to prevent over-dispatch")
    
    return len(load_shedding_gens)


def export_generators_csv(n, output_file):
    """
    Export full generator data to CSV.
    
    Parameters
    ----------
    n : pypsa.Network
        Network containing generators
    output_file : str or Path
        Path to save CSV file
    """
    logger.info("Exporting full generator data to CSV")
    
    # Get generator data
    gens = n.generators.copy()
    
    # Add some useful calculated columns
    if 'carrier' in gens.columns:
        # Count by carrier
        carrier_counts = gens.groupby('carrier').size()
        logger.info(f"Generators by carrier:")
        for carrier, count in carrier_counts.items():
            logger.info(f"  {carrier}: {count} units")
    
    # Save to CSV
    gens.to_csv(output_file)
    logger.info(f"[OK] Saved {len(gens)} generators to {output_file}")


def create_summary_by_carrier(n, output_file):
    """
    Create summary of generators grouped by carrier type.
    
    Parameters
    ----------
    n : pypsa.Network
        Network containing generators
    output_file : str or Path
        Path to save summary CSV
    """
    logger.info("Creating generator summary by carrier")
    
    gens = n.generators.copy()
    
    # Group by carrier
    summary = gens.groupby('carrier').agg({
        'p_nom': ['count', 'sum', 'mean', 'min', 'max'],
        'marginal_cost': ['mean', 'min', 'max']
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Rename for clarity
    summary.columns = [
        'carrier', 'count', 'total_capacity_mw', 'avg_capacity_mw', 
        'min_capacity_mw', 'max_capacity_mw',
        'avg_marginal_cost', 'min_marginal_cost', 'max_marginal_cost'
    ]
    
    # Save
    summary.to_csv(output_file, index=False)
    logger.info(f"[OK] Saved carrier summary to {output_file}")
    
    # Log summary
    logger.info("\nGenerator Summary by Carrier:")
    logger.info(f"{'Carrier':<30} {'Count':>8} {'Total MW':>12} {'Avg MW':>10}")
    logger.info("-" * 80)
    for _, row in summary.iterrows():
        logger.info(f"{row['carrier']:<30} {int(row['count']):>8} {row['total_capacity_mw']:>12,.1f} {row['avg_capacity_mw']:>10,.1f}")


def create_technology_capacity_summary(n, output_file):
    """
    Create detailed technology capacity summary.
    
    Parameters
    ----------
    n : pypsa.Network
        Network containing generators
    output_file : str or Path
        Path to save summary CSV
    """
    logger.info("Creating technology capacity summary")
    
    gens = n.generators.copy()
    
    # Summary by carrier with more detail
    tech_summary = []
    for carrier in gens['carrier'].unique():
        carrier_gens = gens[gens['carrier'] == carrier]
        
        tech_summary.append({
            'technology': carrier,
            'count': len(carrier_gens),
            'total_capacity_mw': carrier_gens['p_nom'].sum(),
            'avg_capacity_mw': carrier_gens['p_nom'].mean(),
            'median_capacity_mw': carrier_gens['p_nom'].median(),
            'min_capacity_mw': carrier_gens['p_nom'].min(),
            'max_capacity_mw': carrier_gens['p_nom'].max(),
            'std_capacity_mw': carrier_gens['p_nom'].std()
        })
    
    tech_df = pd.DataFrame(tech_summary)
    tech_df = tech_df.sort_values('total_capacity_mw', ascending=False)
    tech_df.to_csv(output_file, index=False)
    
    logger.info(f"✓ Saved technology summary to {output_file}")


def create_integration_report(n, output_file):
    """
    Create text integration report.
    
    Parameters
    ----------
    n : pypsa.Network
        Network containing generators
    output_file : str or Path
        Path to save report
    """
    logger.info("Creating integration report")
    
    gens = n.generators.copy()
    
    report = []
    report.append("=" * 80)
    report.append("GENERATOR INTEGRATION REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Network: {n.name}")
    report.append(f"Total Generators: {len(gens):,}")
    report.append(f"Total Capacity: {gens['p_nom'].sum():,.1f} MW")
    report.append("")
    report.append("Network Statistics:")
    report.append(f"  Buses: {len(n.buses):,}")
    report.append(f"  Lines: {len(n.lines):,}")
    report.append(f"  Transformers: {len(n.transformers):,}")
    report.append(f"  Links: {len(n.links):,}")
    report.append(f"  Loads: {len(n.loads):,}")
    report.append(f"  Storage Units: {len(n.storage_units):,}")
    report.append("")
    report.append("Generator Breakdown by Carrier:")
    report.append("-" * 80)
    
    carrier_summary = gens.groupby('carrier').agg({
        'p_nom': ['count', 'sum']
    })
    carrier_summary.columns = ['count', 'capacity_mw']
    carrier_summary = carrier_summary.sort_values('capacity_mw', ascending=False)
    
    for carrier, row in carrier_summary.iterrows():
        report.append(f"  {carrier:<40} {int(row['count']):>6} units, {row['capacity_mw']:>12,.1f} MW")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✓ Saved integration report to {output_file}")
    
    # Also log to console
    for line in report:
        logger.info(line)


def create_generators_map(n, output_file):
    """
    Create HTML map visualization of generators.
    
    Parameters
    ----------
    n : pypsa.Network
        Network containing generators
    output_file : str or Path
        Path to save HTML map
    """
    logger.info("Creating generators map visualization")
    
    try:
        # Get generators with coordinates
        gens = n.generators.copy()
        buses = n.buses.copy()
        
        # Merge to get coordinates
        gens_with_coords = gens.merge(
            buses[['x', 'y']], 
            left_on='bus', 
            right_index=True,
            how='left'
        )
        
        # Filter to those with coordinates
        gens_with_coords = gens_with_coords.dropna(subset=['x', 'y'])
        
        if len(gens_with_coords) == 0:
            logger.warning("No generators with coordinates found for map")
            # Create empty file
            with open(output_file, 'w') as f:
                f.write("<html><body><h1>No generators with coordinates</h1></body></html>")
            return
        
        # Create base map centered on UK
        center_lat = gens_with_coords['y'].mean()
        center_lon = gens_with_coords['x'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Color mapping for carriers
        carrier_colors = {
            'wind_onshore': 'green',
            'wind_offshore': 'darkgreen',
            'solar_pv': 'orange',
            'nuclear': 'red',
            'ccgt': 'blue',
            'ocgt': 'lightblue',
            'coal': 'black',
            'biomass': 'brown',
            'load_shedding': 'red'
        }
        
        # Add markers with clustering
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, gen in gens_with_coords.iterrows():
            color = carrier_colors.get(gen.get('carrier', 'unknown'), 'gray')
            
            popup_text = f"""
            <b>Generator:</b> {idx}<br>
            <b>Carrier:</b> {gen.get('carrier', 'N/A')}<br>
            <b>Capacity:</b> {gen.get('p_nom', 0):.1f} MW<br>
            <b>Bus:</b> {gen.get('bus', 'N/A')}
            """
            
            folium.CircleMarker(
                location=[gen['y'], gen['x']],
                radius=min(gen.get('p_nom', 0) / 100, 10),  # Scale radius by capacity
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillOpacity=0.6
            ).add_to(marker_cluster)
        
        # Save map
        m.save(str(output_file))
        logger.info(f"✓ Saved generators map to {output_file}")
        logger.info(f"  Mapped {len(gens_with_coords)} generators")
        
    except Exception as e:
        logger.error(f"Error creating generators map: {e}")
        # Create simple HTML file with error message
        with open(output_file, 'w') as f:
            f.write(f"<html><body><h1>Error creating map</h1><p>{str(e)}</p></body></html>")


def main():
    """Main execution function for Snakemake."""
    global logger
    
    start_time = time.time()
    
    # Reinitialize logger with Snakemake log path
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "finalize_generator_integration"
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("FINALIZE GENERATOR INTEGRATION - STAGE 3")
    logger.info("=" * 80)
    
    try:
        # Get input/output from Snakemake
        input_network = snakemake.input.network
        output_network = snakemake.output.network
        csv_generators_full = snakemake.output.csv_generators_full
        csv_generators_summary = snakemake.output.csv_generators_summary
        csv_technology_summary = snakemake.output.csv_technology_summary
        csv_integration_report = snakemake.output.csv_integration_report
        
        # Get VoLL parameter
        voll = snakemake.params.voll
        
        logger.info(f"Input network: {input_network}")
        logger.info(f"Output network: {output_network}")
        logger.info(f"VoLL: £{voll:,.0f}/MWh")
        
        # Load network
        logger.info("-" * 80)
        logger.info("PART 1: LOADING NETWORK")
        logger.info("-" * 80)
        
        logger.info(f"Loading network from {input_network}")
        n = load_network(input_network, custom_logger=logger)
        
        logger.info(f"Network: {n.name}")
        logger.info(f"  - Buses: {len(n.buses)}")
        logger.info(f"  - Generators (before load shedding): {len(n.generators)}")
        logger.info(f"  - Storage Units: {len(n.storage_units)}")
        logger.info(f"  - Loads: {len(n.loads)}")
        logger.info(f"  - Snapshots: {len(n.snapshots)}")
        
        # Add load shedding generators
        logger.info("-" * 80)
        logger.info("PART 2: ADDING LOAD SHEDDING GENERATORS")
        logger.info("-" * 80)
        
        n_load_shedding = add_load_shedding_generators(n, voll=voll)
        
        logger.info(f"Generators after load shedding: {len(n.generators)}")
        
        # Export generators
        logger.info("-" * 80)
        logger.info("PART 3: EXPORTING GENERATOR DATA")
        logger.info("-" * 80)
        
        export_generators_csv(n, csv_generators_full)
        create_summary_by_carrier(n, csv_generators_summary)
        create_technology_capacity_summary(n, csv_technology_summary)
        
        # Create reports
        logger.info("-" * 80)
        logger.info("PART 4: CREATING REPORTS AND VISUALIZATIONS")
        logger.info("-" * 80)
        
        create_integration_report(n, csv_integration_report)
        # Map visualization removed - use plot_thermal_generators_pypsa rule instead
        
        # Save final network
        logger.info("-" * 80)
        logger.info("PART 5: SAVING FINAL NETWORK")
        logger.info("-" * 80)
        
        logger.info(f"Saving finalized network to {output_network}")
        save_network(n, output_network, custom_logger=logger)
        logger.info("✓ Network saved successfully")
        
        # Final summary
        duration = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info("FINALIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total generators: {len(n.generators):,}")
        logger.info(f"  - Load shedding: {n_load_shedding:,}")
        logger.info(f"  - Productive: {len(n.generators) - n_load_shedding:,}")
        logger.info(f"Total capacity: {n.generators['p_nom'].sum():,.1f} MW")
        logger.info("")
        logger.info("Output files created:")
        logger.info(f"  1. Network: {output_network}")
        logger.info(f"  2. Full CSV: {csv_generators_full}")
        logger.info(f"  3. Carrier summary: {csv_generators_summary}")
        logger.info(f"  4. Technology summary: {csv_technology_summary}")
        logger.info(f"  5. Integration report: {csv_integration_report}")
        # logger.info(f"  6. Map visualization: {map_file}")  # Disabled - use plot_thermal_generators_pypsa instead
        logger.info("")
        logger.info("✓ GENERATOR INTEGRATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR IN FINALIZE GENERATOR INTEGRATION")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    if "snakemake" in dir():
        main()
    else:
        print("This script is designed to be run by Snakemake.")
        print("Please use the Snakemake workflow.")
        sys.exit(1)

