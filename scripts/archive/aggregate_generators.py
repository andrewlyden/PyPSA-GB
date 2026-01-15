"""
Aggregate Generators by Carrier and Bus
========================================

Reduces optimization problem size by aggregating generators of the same carrier
type at each bus while preserving time-varying availability (p_max_pu) diversity.

This is critical for numerical stability and solver performance:
- Reduces number of decision variables (e.g., 4934 → ~500 generators)
- Maintains renewable generation diversity through weighted p_max_pu profiles
- Preserves total capacity, marginal costs, and technical constraints
- Improves Gurobi matrix conditioning

Author: AI Assistant
Date: 2025-10-30
"""

import sys
import logging
from pathlib import Path
import pypsa
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def aggregate_generators_by_carrier_and_bus(
    network: pypsa.Network,
    preserve_diversity_carriers: List[str] = None,
    aggregate_load_shedding: bool = False
) -> Tuple[pypsa.Network, Dict]:
    """
    Aggregate generators of same carrier type at each bus.
    
    This function combines multiple generators of the same carrier type at each
    bus into a single representative generator, dramatically reducing the size
    of the optimization problem while preserving key technical characteristics.
    
    Time-varying availability (p_max_pu) profiles are aggregated using
    capacity-weighted averaging, preserving renewable generation diversity.
    
    Parameters
    ----------
    network : pypsa.Network
        Network with generators to aggregate
    preserve_diversity_carriers : list of str, optional
        Carriers that should use capacity-weighted p_max_pu averaging
        Default: ['wind_onshore', 'wind_offshore', 'solar_pv', 'Solar', 
                  'Wind (Onshore)', 'Wind (Offshore)']
    aggregate_load_shedding : bool, default False
        Whether to aggregate load_shedding generators (usually keep separate)
        
    Returns
    -------
    network : pypsa.Network
        Network with aggregated generators
    aggregation_report : dict
        Statistics about aggregation (generators before/after, by carrier, etc.)
        
    Examples
    --------
    >>> n_aggregated, report = aggregate_generators_by_carrier_and_bus(network)
    >>> print(f"Reduced from {report['total_before']} to {report['total_after']} generators")
    
    Notes
    -----
    Aggregation preserves:
    - Total installed capacity (p_nom) per carrier per bus
    - Marginal costs (capacity-weighted average)
    - Technical constraints (committable, ramp limits, etc.)
    - Time-varying availability profiles (capacity-weighted)
    
    Does not preserve:
    - Individual generator names (creates new aggregated names)
    - Generator-specific attributes beyond standard PyPSA fields
    """
    logger.info("=" * 100)
    logger.info("AGGREGATING GENERATORS BY CARRIER AND BUS")
    logger.info("=" * 100)
    
    # Default carriers with time-varying profiles to preserve
    if preserve_diversity_carriers is None:
        preserve_diversity_carriers = [
            'wind_onshore', 'wind_offshore', 'solar_pv', 
            'Solar', 'Wind (Onshore)', 'Wind (Offshore)',
            'shoreline_wave', 'tidal_stream'
        ]
    
    logger.info(f"Initial generators: {len(network.generators)}")
    logger.info(f"Preserving p_max_pu diversity for: {preserve_diversity_carriers}")
    logger.info(f"Aggregate load shedding: {aggregate_load_shedding}")
    
    # Store original generator data for reporting
    original_count = len(network.generators)
    original_by_carrier = network.generators.groupby('carrier').size().to_dict()
    
    # Group generators by (bus, carrier)
    grouped = network.generators.groupby(['bus', 'carrier'])
    
    logger.info(f"\nFound {len(grouped)} unique (bus, carrier) combinations")
    
    # Create new aggregated generators
    new_generators = []
    new_p_max_pu = {}
    generators_to_remove = []
    aggregation_map = {}  # Maps old generator names to new aggregated names
    
    for (bus, carrier), group in grouped:
        # Skip if only one generator (no aggregation needed)
        if len(group) == 1:
            continue
            
        # Skip load_shedding unless explicitly requested
        if 'load' in carrier.lower() and 'shed' in carrier.lower():
            if not aggregate_load_shedding:
                continue
        
        # Generate aggregated generator name
        agg_name = f"agg_{carrier}_{bus}"
        
        # Aggregate capacity (sum)
        total_p_nom = group['p_nom'].sum()
        
        # Aggregate marginal cost (capacity-weighted average)
        if total_p_nom > 0:
            avg_marginal_cost = (group['p_nom'] * group['marginal_cost']).sum() / total_p_nom
        else:
            avg_marginal_cost = group['marginal_cost'].mean()
        
        # Technical parameters (take from first generator, or aggregate)
        # Most of these should be identical for same carrier type
        first_gen = group.iloc[0]
        
        # Committable: if ANY are committable, aggregate is committable
        committable = group['committable'].any() if 'committable' in group.columns else False
        
        # Extendable: if ANY are extendable, aggregate is extendable
        p_nom_extendable = group['p_nom_extendable'].any() if 'p_nom_extendable' in group.columns else False
        
        # Capital cost (capacity-weighted if extendable)
        if p_nom_extendable and total_p_nom > 0:
            capital_cost = (group['p_nom'] * group.get('capital_cost', 0)).sum() / total_p_nom
        else:
            capital_cost = first_gen.get('capital_cost', 0)
        
        # Efficiency (capacity-weighted average)
        if 'efficiency' in group.columns and total_p_nom > 0:
            efficiency = (group['p_nom'] * group['efficiency']).sum() / total_p_nom
        else:
            efficiency = first_gen.get('efficiency', 1.0)
        
        # Create aggregated generator dictionary
        agg_gen = {
            'name': agg_name,
            'bus': bus,
            'carrier': carrier,
            'p_nom': total_p_nom,
            'marginal_cost': avg_marginal_cost,
            'committable': committable,
            'p_nom_extendable': p_nom_extendable,
            'capital_cost': capital_cost,
            'efficiency': efficiency,
        }
        
        # Add optional fields if they exist
        for field in ['p_nom_min', 'p_nom_max', 'p_min_pu', 'ramp_limit_up', 
                      'ramp_limit_down', 'min_up_time', 'min_down_time',
                      'up_time_before', 'down_time_before', 'start_up_cost',
                      'shut_down_cost']:
            if field in group.columns:
                if field in ['p_nom_min', 'p_nom_max']:
                    # Sum capacity limits
                    agg_gen[field] = group[field].sum()
                else:
                    # Take from first (should be same for carrier type)
                    agg_gen[field] = first_gen.get(field, 0)
        
        new_generators.append(agg_gen)
        
        # Handle p_max_pu aggregation (capacity-weighted for renewables)
        if carrier in preserve_diversity_carriers:
            # Get p_max_pu for all generators in group
            if len(network.generators_t.p_max_pu.columns) > 0:
                # Get columns that exist in this group
                group_gens = [g for g in group.index if g in network.generators_t.p_max_pu.columns]
                
                if group_gens:
                    # Capacity-weighted average of p_max_pu profiles
                    capacities = group.loc[group_gens, 'p_nom'].values
                    profiles = network.generators_t.p_max_pu[group_gens].values
                    
                    # Weighted average: sum(capacity * profile) / sum(capacity)
                    if capacities.sum() > 0:
                        weighted_profile = (profiles * capacities).sum(axis=1) / capacities.sum()
                        new_p_max_pu[agg_name] = weighted_profile
                    else:
                        # Fallback to simple average
                        new_p_max_pu[agg_name] = profiles.mean(axis=1)
        else:
            # For non-renewable carriers, use max p_max_pu (conservative)
            if len(network.generators_t.p_max_pu.columns) > 0:
                group_gens = [g for g in group.index if g in network.generators_t.p_max_pu.columns]
                if group_gens:
                    new_p_max_pu[agg_name] = network.generators_t.p_max_pu[group_gens].max(axis=1)
        
        # Track mapping for reporting
        for gen_name in group.index:
            aggregation_map[gen_name] = agg_name
            generators_to_remove.append(gen_name)
    
    logger.info(f"\n✓ Created {len(new_generators)} aggregated generators")
    logger.info(f"  Removing {len(generators_to_remove)} original generators")
    
    # Remove old generators
    network.mremove("Generator", generators_to_remove)
    
    # Add new aggregated generators
    for gen in new_generators:
        gen_name = gen.pop('name')
        network.add("Generator", gen_name, **gen)
    
    # Update p_max_pu time series
    if new_p_max_pu:
        # Convert to DataFrame
        new_p_max_pu_df = pd.DataFrame(new_p_max_pu, index=network.snapshots)
        
        # Combine with existing p_max_pu (for non-aggregated generators)
        remaining_gens = [g for g in network.generators.index if g not in new_p_max_pu_df.columns]
        if remaining_gens:
            existing_p_max_pu = network.generators_t.p_max_pu[remaining_gens]
            network.generators_t.p_max_pu = pd.concat([existing_p_max_pu, new_p_max_pu_df], axis=1)
        else:
            network.generators_t.p_max_pu = new_p_max_pu_df
    
    # Generate aggregation report
    final_count = len(network.generators)
    final_by_carrier = network.generators.groupby('carrier').size().to_dict()
    
    aggregation_report = {
        'total_before': original_count,
        'total_after': final_count,
        'reduction': original_count - final_count,
        'reduction_pct': (original_count - final_count) / original_count * 100,
        'aggregated_generators': len(new_generators),
        'by_carrier_before': original_by_carrier,
        'by_carrier_after': final_by_carrier,
        'aggregation_map': aggregation_map
    }
    
    logger.info("\n" + "=" * 100)
    logger.info("AGGREGATION SUMMARY")
    logger.info("=" * 100)
    logger.info(f"Original generators: {original_count}")
    logger.info(f"Aggregated generators: {final_count}")
    logger.info(f"Reduction: {aggregation_report['reduction']} ({aggregation_report['reduction_pct']:.1f}%)")
    
    logger.info("\nBy carrier:")
    all_carriers = set(list(original_by_carrier.keys()) + list(final_by_carrier.keys()))
    for carrier in sorted(all_carriers):
        before = original_by_carrier.get(carrier, 0)
        after = final_by_carrier.get(carrier, 0)
        if before != after:
            logger.info(f"  {carrier}: {before} → {after} ({before - after} reduction)")
    
    logger.info("\n" + "=" * 100)
    
    return network, aggregation_report


def main():
    """
    Main function for Snakemake integration.
    """
    # Setup logging
    log_file = snakemake.log[0] if hasattr(snakemake, 'log') else None
    logger = setup_logging('aggregate_generators', log_file)
    
    try:
        # Get inputs/outputs
        input_network = snakemake.input.network
        output_network = snakemake.output.network
        output_report = snakemake.output.report
        
        # Get parameters
        params = snakemake.params
        preserve_diversity = params.get('preserve_diversity_carriers', None)
        aggregate_load_shedding = params.get('aggregate_load_shedding', False)
        
        logger.info(f"Input network: {input_network}")
        logger.info(f"Output network: {output_network}")
        
        # Load network
        logger.info("Loading network...")
        network = pypsa.Network(input_network)
        logger.info(f"✓ Loaded network with {len(network.generators)} generators")
        
        # Aggregate generators
        network, report = aggregate_generators_by_carrier_and_bus(
            network,
            preserve_diversity_carriers=preserve_diversity,
            aggregate_load_shedding=aggregate_load_shedding
        )
        
        # Save aggregated network
        logger.info(f"\nSaving aggregated network to: {output_network}")
        network.export_to_netcdf(output_network)
        logger.info("✓ Network saved")
        
        # Save aggregation report
        logger.info(f"Saving aggregation report to: {output_report}")
        report_df = pd.DataFrame([{
            'metric': 'total_before',
            'value': report['total_before']
        }, {
            'metric': 'total_after',
            'value': report['total_after']
        }, {
            'metric': 'reduction',
            'value': report['reduction']
        }, {
            'metric': 'reduction_pct',
            'value': report['reduction_pct']
        }])
        report_df.to_csv(output_report, index=False)
        logger.info("✓ Report saved")
        
        logger.info("\n" + "=" * 100)
        logger.info("✓ GENERATOR AGGREGATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"FATAL ERROR in generator aggregation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    if 'snakemake' in globals():
        main()
    else:
        print("This script is designed to be run via Snakemake")
        print("For standalone testing, load network and call aggregate_generators_by_carrier_and_bus()")

