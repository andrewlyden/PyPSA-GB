"""
Inspect Network Before Solving - Diagnostic Script

This script loads a network and performs comprehensive checks to identify
issues that could cause infeasibility or unbounded solutions.
"""

import pypsa
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

def setup_logging():
    """Setup logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def inspect_network(network_path, solve_period=None):
    """
    Perform comprehensive network inspection.
    
    Parameters
    ----------
    network_path : str
        Path to network file
    solve_period : dict, optional
        Solve period configuration with 'start' and 'end' dates
    """
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("NETWORK INSPECTION - PRE-SOLVE DIAGNOSTICS")
    logger.info("="*80)
    logger.info(f"Network file: {network_path}")
    
    # Load network
    logger.info("\n1. LOADING NETWORK")
    logger.info("-"*80)
    network = pypsa.Network(network_path)
    logger.info(f"Network name: {network.name}")
    logger.info(f"Buses: {len(network.buses)}")
    logger.info(f"Generators: {len(network.generators)}")
    logger.info(f"Loads: {len(network.loads)}")
    logger.info(f"Storage units: {len(network.storage_units)}")
    logger.info(f"Links: {len(network.links)}")
    logger.info(f"Lines: {len(network.lines)}")
    logger.info(f"Snapshots: {len(network.snapshots)}")
    
    # Apply solve period if specified
    if solve_period:
        logger.info("\n2. APPLYING SOLVE PERIOD")
        logger.info("-"*80)
        solve_start = pd.Timestamp(solve_period['start'])
        solve_end = pd.Timestamp(solve_period['end'])
        logger.info(f"Period: {solve_start} to {solve_end}")
        
        mask = (network.snapshots >= solve_start) & (network.snapshots <= solve_end)
        selected_snapshots = network.snapshots[mask]
        network.set_snapshots(selected_snapshots)
        logger.info(f"Snapshots after filtering: {len(network.snapshots)}")
    
    # Check demand vs generation capacity
    logger.info("\n3. DEMAND VS GENERATION CAPACITY CHECK")
    logger.info("-"*80)
    
    # Total generation capacity by carrier
    gen_capacity = network.generators.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
    total_gen_capacity = gen_capacity.sum()
    
    logger.info(f"Total generation capacity: {total_gen_capacity:,.1f} MW")
    logger.info("\nTop 10 carriers by capacity:")
    for carrier, capacity in gen_capacity.head(10).items():
        logger.info(f"  {carrier:30s}: {capacity:10,.1f} MW")
    
    # Check for load shedding generators
    load_shedding_gen = network.generators[network.generators['carrier'] == 'load_shedding']
    if len(load_shedding_gen) > 0:
        ls_capacity = load_shedding_gen['p_nom'].sum()
        logger.info(f"\n⚠️  Load shedding capacity: {ls_capacity:,.1f} MW ({len(load_shedding_gen)} units)")
        logger.info(f"    Marginal cost: £{load_shedding_gen['marginal_cost'].iloc[0]:,.0f}/MWh")
        
        # Productive capacity (excluding load shedding)
        productive_capacity = total_gen_capacity - ls_capacity
        logger.info(f"\nProductive generation capacity: {productive_capacity:,.1f} MW")
    else:
        productive_capacity = total_gen_capacity
        logger.warning("\n⚠️  NO LOAD SHEDDING GENERATORS FOUND!")
        logger.warning("    Network may be infeasible if demand exceeds generation!")
    
    # Demand analysis
    if len(network.loads_t.p_set) > 0:
        # Get demand for the current snapshots
        demand_timeseries = network.loads_t.p_set
        total_demand_per_snapshot = demand_timeseries.sum(axis=1)
        
        peak_demand = total_demand_per_snapshot.max()
        avg_demand = total_demand_per_snapshot.mean()
        min_demand = total_demand_per_snapshot.min()
        
        logger.info(f"\nDemand analysis:")
        logger.info(f"  Peak demand: {peak_demand:,.1f} MW")
        logger.info(f"  Average demand: {avg_demand:,.1f} MW")
        logger.info(f"  Minimum demand: {min_demand:,.1f} MW")
        
        # Check adequacy
        margin = productive_capacity - peak_demand
        margin_pct = (margin / peak_demand) * 100
        
        logger.info(f"\nCapacity margin:")
        logger.info(f"  Absolute: {margin:,.1f} MW")
        logger.info(f"  Percentage: {margin_pct:.1f}%")
        
        if margin < 0:
            logger.error(f"\n❌ CRITICAL: INSUFFICIENT GENERATION CAPACITY!")
            logger.error(f"    Peak demand ({peak_demand:,.1f} MW) exceeds productive capacity ({productive_capacity:,.1f} MW)")
            logger.error(f"    Shortfall: {-margin:,.1f} MW")
            if len(load_shedding_gen) == 0:
                logger.error(f"    This will cause INFEASIBILITY (no load shedding available)")
        elif margin_pct < 10:
            logger.warning(f"\n⚠️  WARNING: Low capacity margin ({margin_pct:.1f}%)")
            logger.warning(f"    Network may be tight on capacity")
        else:
            logger.info(f"\n[OK] Adequate capacity margin")
    else:
        logger.warning("\n⚠️  No demand timeseries found in network.loads_t.p_set")
    
    # Check storage
    logger.info("\n4. STORAGE ANALYSIS")
    logger.info("-"*80)
    if len(network.storage_units) > 0:
        storage_power = network.storage_units['p_nom'].sum()
        storage_energy = network.storage_units['max_hours'].fillna(0) * network.storage_units['p_nom']
        total_storage_energy = storage_energy.sum()
        
        logger.info(f"Storage units: {len(network.storage_units)}")
        logger.info(f"Total storage power: {storage_power:,.1f} MW")
        logger.info(f"Total storage energy: {total_storage_energy:,.1f} MWh")
        
        # Storage by carrier
        storage_by_carrier = network.storage_units.groupby('carrier').agg({
            'p_nom': 'sum',
            'max_hours': lambda x: (x.fillna(0) * network.storage_units.loc[x.index, 'p_nom']).sum()
        })
        logger.info("\nStorage by carrier:")
        for carrier in storage_by_carrier.index:
            power = storage_by_carrier.loc[carrier, 'p_nom']
            logger.info(f"  {carrier:30s}: {power:10,.1f} MW")
    else:
        logger.info("No storage units in network")
    
    # Check interconnectors
    logger.info("\n5. INTERCONNECTOR ANALYSIS")
    logger.info("-"*80)
    if len(network.links) > 0:
        logger.info(f"Links/Interconnectors: {len(network.links)}")
        
        # Check for fixed flows
        if 'p_set' in network.links.columns:
            fixed_links = network.links[network.links['p_set'].notna()]
            if len(fixed_links) > 0:
                logger.info(f"\n⚠️  {len(fixed_links)} links have FIXED p_set values:")
                for idx, link in fixed_links.iterrows():
                    logger.info(f"    {idx}: p_set={link['p_set']:.1f} MW")
        
        # Check timeseries flows
        if len(network.links_t.p_set) > 0:
            logger.info(f"\n⚠️  Links have FIXED TIMESERIES flows (p_set)")
            logger.info(f"    This may cause infeasibility if flows conflict with constraints!")
            
            # Analyze fixed flows
            fixed_flow_stats = network.links_t.p_set.describe()
            logger.info("\nFixed flow statistics (MW):")
            logger.info(fixed_flow_stats)
            
            # Check if any fixed flows are very large
            max_fixed_flow = network.links_t.p_set.abs().max().max()
            if max_fixed_flow > 10000:
                logger.warning(f"\n⚠️  WARNING: Very large fixed interconnector flows detected!")
                logger.warning(f"    Max absolute flow: {max_fixed_flow:,.1f} MW")
        
        # Link capacity
        logger.info("\nLink capacities:")
        for idx, link in network.links.iterrows():
            p_nom = link.get('p_nom', 'unlimited')
            logger.info(f"  {idx:40s}: {str(p_nom):>15s} MW")
    else:
        logger.info("No links/interconnectors in network")
    
    # Check for data quality issues
    logger.info("\n6. DATA QUALITY CHECKS")
    logger.info("-"*80)
    
    issues = []
    
    # Check for NaN values in critical columns
    if network.generators['p_nom'].isna().any():
        nan_count = network.generators['p_nom'].isna().sum()
        issues.append(f"❌ {nan_count} generators have NaN p_nom")
    
    if network.generators['marginal_cost'].isna().any():
        nan_count = network.generators['marginal_cost'].isna().sum()
        issues.append(f"⚠️  {nan_count} generators have NaN marginal_cost (will default to 0)")
    
    # Check for negative capacities
    if (network.generators['p_nom'] < 0).any():
        neg_count = (network.generators['p_nom'] < 0).sum()
        issues.append(f"❌ {neg_count} generators have negative p_nom")
    
    # Check for missing bus connections
    missing_buses = set(network.generators['bus']) - set(network.buses.index)
    if missing_buses:
        issues.append(f"❌ {len(missing_buses)} generators connected to non-existent buses")
    
    # Check for isolated buses
    connected_buses = set()
    connected_buses.update(network.generators['bus'])
    connected_buses.update(network.loads['bus'])
    if len(network.lines) > 0:
        connected_buses.update(network.lines['bus0'])
        connected_buses.update(network.lines['bus1'])
    
    isolated_buses = set(network.buses.index) - connected_buses
    if isolated_buses:
        issues.append(f"⚠️  {len(isolated_buses)} isolated buses (no generators, loads, or lines)")
    
    if issues:
        logger.warning("\nData quality issues found:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("[OK] No critical data quality issues detected")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("INSPECTION SUMMARY")
    logger.info("="*80)
    
    potential_issues = []
    
    if margin < 0:
        potential_issues.append("CRITICAL: Insufficient generation capacity (infeasibility likely)")
    
    if len(load_shedding_gen) == 0:
        potential_issues.append("No load shedding generators (network may be infeasible)")
    
    if len(network.links_t.p_set) > 0:
        potential_issues.append("Fixed interconnector flows may cause conflicts")
    
    if issues:
        potential_issues.extend([f"Data quality: {issue}" for issue in issues])
    
    if potential_issues:
        logger.warning("\nPOTENTIAL ISSUES THAT MAY CAUSE INFEASIBILITY:")
        for i, issue in enumerate(potential_issues, 1):
            logger.warning(f"  {i}. {issue}")
    else:
        logger.info("\n[OK] No obvious issues detected - network appears solvable")
    
    logger.info("\n" + "="*80)
    
    return network

if __name__ == "__main__":
    # Run inspection on the Historical_2020_clustered network
    network_path = "resources/network/Historical_2020_clustered.nc"
    
    # Match the solve period from scenario config
    solve_period = {
        'start': '2020-12-14',
        'end': '2020-12-20'
    }
    
    network = inspect_network(network_path, solve_period)

