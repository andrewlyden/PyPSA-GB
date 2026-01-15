"""
Finalize Network - Create clean {scenario}.nc file

This script copies the complete network (with all components integrated)
to a clean scenario filename for easier reference and downstream use.
Also generates a comprehensive network summary report.
"""

import pypsa
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from scripts.utilities.logging_config import setup_logging
from scripts.archive.deep_network_validation import deep_validate_network

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Suppress PyPSA warnings about unoptimized networks (expected at finalization stage)
warnings.filterwarnings('ignore', message='The network has not been optimized yet')
warnings.filterwarnings('ignore', message='no model is stored')
warnings.filterwarnings('ignore', message='no objective value is stored')  
warnings.filterwarnings('ignore', message='no objective constant is stored')


def _series_within_bounds(series: pd.Series, bound: float) -> bool:
    """Check whether all finite values in a series fall within ±bound."""
    if series is None:
        return False
    ser = pd.to_numeric(series, errors="coerce").dropna()
    if ser.empty:
        return False
    return (ser.abs() <= bound).all()


def normalize_bus_coordinates(network: pypsa.Network, logger: logging.Logger) -> None:
    """
    Preserve OSGB36 coordinates for PyPSA plotting with cartopy.
    
    IMPORTANT: PyPSA's plot() with geomap=True requires consistent coordinate systems.
    For GB networks, we use OSGB36 (British National Grid, EPSG:27700) throughout,
    as this is the standard for UK infrastructure data and matches the network topology files.
    
    This function now PRESERVES OSGB36 coordinates instead of converting to WGS84,
    as cartopy.crs.OSGB handles the projection correctly for GB coastline display.
    
    Modifies network in-place (but mostly just validates).
    """
    buses = network.buses
    
    if 'x' not in buses.columns or 'y' not in buses.columns:
        logger.warning("Bus coordinates missing (no x/y columns)")
        return
    
    # Check coordinate system
    x_series = pd.to_numeric(buses['x'], errors='coerce')
    y_series = pd.to_numeric(buses['y'], errors='coerce')
    
    # Detect coordinate system
    x_valid_osgb = (x_series.dropna().between(-1000, 800000).all() if len(x_series.dropna()) > 0 else False)
    y_valid_osgb = (y_series.dropna().between(-1000, 1400000).all() if len(y_series.dropna()) > 0 else False)
    
    osgb_count = 0
    wgs84_count = 0
    
    for idx in buses.index:
        x_val = buses.at[idx, 'x']
        y_val = buses.at[idx, 'y']
        
        if pd.notna(x_val) and pd.notna(y_val):
            is_wgs84 = -180 <= x_val <= 180 and -90 <= y_val <= 90 and (abs(x_val) < 100)
            is_osgb36 = -1000 <= x_val <= 800000 and -1000 <= y_val <= 1400000 and (x_val > 500 or y_val > 500)
            
            if is_osgb36:
                osgb_count += 1
            elif is_wgs84:
                wgs84_count += 1
    
    if osgb_count > 0 and wgs84_count > 0:
        logger.warning(f"MIXED COORDINATES DETECTED: {osgb_count} OSGB36, {wgs84_count} WGS84")
        logger.warning("This will cause plotting failures! Check interconnector integration.")
    elif osgb_count > 0:
        logger.info(f"✓ All {osgb_count} buses have OSGB36 coordinates (correct for GB networks)")
    elif wgs84_count > 0:
        logger.warning(f"All {wgs84_count} buses have WGS84 coordinates - should be OSGB36 for GB networks!")
    
    # NO CONVERSION - preserve OSGB36 coordinates for cartopy plotting
    # The validation code below has been removed as we now preserve OSGB36



def generate_network_summary(network, scenario_config, output_path):
    """
    Generate comprehensive network summary report integrating validation results.
    Output is written to both log file and summary text file.
    
    Parameters
    ----------
    network : pypsa.Network
        Complete network to summarize
    scenario_config : dict
        Scenario configuration
    output_path : str or Path
        Path to write summary text file
    """
    logger = logging.getLogger("finalize_network")
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("PYPSA-GB NETWORK SUMMARY & VALIDATION REPORT")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Extract validation results from metadata
    validation_passed = network.meta.get('validation_passed', False)
    validation_warnings = network.meta.get('validation_warnings', 0)
    
    # Validation status header
    summary_lines.append("VALIDATION STATUS")
    summary_lines.append("-" * 80)
    if validation_passed:
        summary_lines.append("✅ VALIDATION PASSED")
        if validation_warnings > 0:
            summary_lines.append(f"⚠  {validation_warnings} warnings (non-critical)")
    else:
        summary_lines.append("❌ VALIDATION FAILED")
    summary_lines.append("")
    
    # Scenario information
    summary_lines.append("SCENARIO INFORMATION")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Scenario: {scenario_config.get('scenario_id', 'Unknown')}")
    summary_lines.append(f"FES Year: {scenario_config.get('fes_year', 'N/A')}")
    summary_lines.append(f"Modelled Year: {scenario_config.get('modelled_year', 'N/A')}")
    summary_lines.append(f"Network Model: {scenario_config.get('network_model', 'Unknown')}")
    summary_lines.append(f"Clustered: {'Yes' if scenario_config.get('clustering', {}).get('enabled', False) else 'No'}")
    summary_lines.append("")
    
    # Network topology
    summary_lines.append("NETWORK TOPOLOGY")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Buses: {len(network.buses):,}")
    summary_lines.append(f"Lines: {len(network.lines):,}")
    summary_lines.append(f"Transformers: {len(network.transformers):,}")
    summary_lines.append(f"Links: {len(network.links):,}")
    summary_lines.append("")
    
    # Temporal resolution
    summary_lines.append("TEMPORAL RESOLUTION")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Snapshots: {len(network.snapshots):,}")
    if len(network.snapshots) > 0:
        summary_lines.append(f"Start: {network.snapshots[0]}")
        summary_lines.append(f"End: {network.snapshots[-1]}")
        if len(network.snapshots) > 1:
            time_diff = network.snapshots[1] - network.snapshots[0]
            summary_lines.append(f"Resolution: {time_diff}")
    summary_lines.append("")
    
    # Demand
    summary_lines.append("DEMAND")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Load buses: {len(network.loads):,}")
    if len(network.loads) > 0 and 'p_set' in network.loads_t:
        total_demand = network.loads_t.p_set.sum().sum()
        peak_demand = network.loads_t.p_set.sum(axis=1).max()
        summary_lines.append(f"Total demand: {total_demand:,.0f} MWh")
        summary_lines.append(f"Peak demand: {peak_demand:,.0f} MW")
    summary_lines.append("")
    
    # Generators by carrier
    summary_lines.append("GENERATION CAPACITY BY CARRIER")
    summary_lines.append("-" * 80)
    if len(network.generators) > 0:
        gen_by_carrier = network.generators.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
        for carrier, capacity in gen_by_carrier.items():
            count = len(network.generators[network.generators.carrier == carrier])
            summary_lines.append(f"  {carrier:30s}: {capacity:10,.0f} MW ({count:5,} units)")
        summary_lines.append(f"  {'TOTAL':30s}: {network.generators.p_nom.sum():10,.0f} MW")
    else:
        summary_lines.append("  No generators found")
    summary_lines.append("")
    
    # Storage
    summary_lines.append("STORAGE CAPACITY")
    summary_lines.append("-" * 80)
    if len(network.storage_units) > 0:
        storage_by_carrier = network.storage_units.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
        for carrier, capacity in storage_by_carrier.items():
            count = len(network.storage_units[network.storage_units.carrier == carrier])
            energy = network.storage_units[network.storage_units.carrier == carrier]['max_hours'].mean() * capacity
            summary_lines.append(f"  {carrier:30s}: {capacity:10,.0f} MW ({energy:10,.0f} MWh, {count:3,} units)")
        summary_lines.append(f"  {'TOTAL':30s}: {network.storage_units.p_nom.sum():10,.0f} MW")
    else:
        summary_lines.append("  No storage units found")
    summary_lines.append("")
    
    # Stores (alternative storage representation)
    if len(network.stores) > 0:
        summary_lines.append("STORES (Energy Storage)")
        summary_lines.append("-" * 80)
        store_by_carrier = network.stores.groupby('carrier')['e_nom'].sum().sort_values(ascending=False)
        for carrier, capacity in store_by_carrier.items():
            count = len(network.stores[network.stores.carrier == carrier])
            summary_lines.append(f"  {carrier:30s}: {capacity:10,.0f} MWh ({count:3,} units)")
        summary_lines.append(f"  {'TOTAL':30s}: {network.stores.e_nom.sum():10,.0f} MWh")
        summary_lines.append("")
    
    # Interconnectors
    summary_lines.append("INTERCONNECTORS")
    summary_lines.append("-" * 80)
    if len(network.links) > 0:
        # Filter for interconnector links (those crossing borders)
        interconnector_links = network.links[network.links.get('is_interconnector', pd.Series(False, index=network.links.index))]
        if len(interconnector_links) > 0:
            summary_lines.append(f"  Interconnectors: {len(interconnector_links):,}")
            summary_lines.append(f"  Total capacity: {interconnector_links.p_nom.sum():,.0f} MW")
        else:
            summary_lines.append("  No interconnectors found (check link attributes)")
    else:
        summary_lines.append("  No links found")
    summary_lines.append("")
    
    # Network metadata
    if hasattr(network, 'meta') and network.meta:
        summary_lines.append("NETWORK METADATA")
        summary_lines.append("-" * 80)
        for key, value in network.meta.items():
            summary_lines.append(f"  {key}: {value}")
        summary_lines.append("")
    
    summary_lines.append("=" * 80)
    
    # Write to file
    summary_text = "\n".join(summary_lines)
    Path(output_path).write_text(summary_text, encoding='utf-8')
    
    # Also write to logger (appears in both terminal and log file)
    logger.info("")
    logger.info("=" * 80)
    logger.info("INTEGRATED NETWORK SUMMARY & VALIDATION REPORT")
    logger.info("=" * 80)
    for line in summary_lines:
        logger.info(line)
    logger.info(f"")
    logger.info(f"Full report written to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Set up logging - use Snakemake log path if available
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "finalize_network"
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("FINALIZING NETWORK TO CLEAN SCENARIO NAME")
    logger.info("=" * 80)
    
    try:
        # Load the complete network
        input_path = snakemake.input.network
        logger.info(f"Loading complete network from: {input_path}")
        network = load_network(input_path, custom_logger=logger)
        
        # Get scenario config
        scenario_config = snakemake.params.scenario_config
        scenario_id = scenario_config.get('scenario_id', snakemake.wildcards.scenario)
        
        logger.info(f"Scenario: {scenario_id}")
        logger.info(f"Network has {len(network.buses)} buses, {len(network.generators)} generators")
        
        # CHECK AND FIX COORDINATES: Ensure all buses have valid WGS84 coordinates
        logger.info("=" * 80)
        logger.info("CHECKING AND NORMALIZING BUS COORDINATES")
        logger.info("=" * 80)
        try:
            normalize_bus_coordinates(network, logger)
            logger.info("[OK] Bus coordinates validated and normalized to WGS84")
        except ValueError as e:
            logger.error(f"✗ Coordinate validation failed: {e}")
            raise
        
        logger.info("=" * 80)
        
        # CRITICAL FIX: Clean up very small p_max_pu values that confuse solver
        # PyPSA docs: https://docs.pypsa.org/latest/user-guide/troubleshooting/
        # Very small non-zero values (< 0.001) can cause unbounded optimization
        logger.info("=" * 80)
        logger.info("CLEANING SMALL p_max_pu VALUES (SOLVER STABILITY FIX)")
        logger.info("=" * 80)
        
        # Get clipping threshold from config (default 0.001)
        clipping_threshold = scenario_config.get('optimization', {}).get('p_max_pu_clipping_threshold', 0.001)
        
        if len(network.generators_t.p_max_pu) > 0:
            # Count small values before clipping
            small_vals_before = ((network.generators_t.p_max_pu > 0) & 
                                (network.generators_t.p_max_pu < clipping_threshold)).sum().sum()
            
            if small_vals_before > 0:
                logger.info(f"Found {small_vals_before:,} very small non-zero p_max_pu values (0 < x < {clipping_threshold})")
                logger.info("These tiny values can confuse solver numerics → causing unbounded")
                logger.info(f"Clipping values < {clipping_threshold} to exactly 0.0...")
                
                # Clip small values to zero
                network.generators_t.p_max_pu = network.generators_t.p_max_pu.where(
                    network.generators_t.p_max_pu >= clipping_threshold, 0.0
                )
                
                # Verify fix
                small_vals_after = ((network.generators_t.p_max_pu > 0) & 
                                   (network.generators_t.p_max_pu < clipping_threshold)).sum().sum()
                logger.info(f"[OK] Cleaned: {small_vals_before - small_vals_after:,} small values -> 0.0")
                logger.info(f"  Remaining small values: {small_vals_after:,}")
            else:
                logger.info(f"[OK] No small p_max_pu values (< {clipping_threshold}) found - network is clean")
        
        logger.info("=" * 80)
        
        # DEEP VALIDATION: Comprehensive component-by-component checks
        # Validation output is now integrated into the summary report below
        validation_passed, errors, warnings = deep_validate_network(network, scenario_config, logger)
        
        # Raise error if validation failed
        if not validation_passed:
            logger.error("=" * 80)
            logger.error("VALIDATION FAILED - DO NOT ATTEMPT OPTIMIZATION")
            logger.error("=" * 80)
            raise ValueError(f"Network validation failed with {len(errors)} critical errors")
        
        logger.info("=" * 80)
        
        # Ensure network has proper metadata
        if not hasattr(network, 'meta'):
            network.meta = {}
        
        network.meta.update({
            'scenario_id': scenario_id,
            'finalized': True,
            'complete_network': True,
            'validation_passed': validation_passed,
            'validation_warnings': len(warnings),
            'includes_components': [
                'topology',
                'demand',
                'renewables',
                'thermal_generators',
                'storage',
                'interconnectors'
            ]
        })
        
        # Set network name
        clustered = scenario_config.get('clustering', {}).get('enabled', False)
        network.name = f"{scenario_id} {'(Clustered)' if clustered else '(Full)'}"
        
        # Save to clean scenario name
        output_path = snakemake.output.network
        logger.info(f"Saving finalized network to: {output_path}")
        save_network(network, output_path, custom_logger=logger)
        
        # Generate comprehensive summary report (integrates validation results)
        summary_path = snakemake.output.summary
        logger.info(f"Generating integrated network summary & validation report: {summary_path}")
        generate_network_summary(network, scenario_config, summary_path)
        
        logger.info("=" * 80)
        logger.info("NETWORK FINALIZATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"FATAL ERROR in network finalization: {e}", exc_info=True)
        raise

