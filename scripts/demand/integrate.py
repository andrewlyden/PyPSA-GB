"""
Integrate Disaggregated Demand Components

This script integrates all disaggregated demand components (heat pumps, EVs, etc.)
back into the PyPSA network. It ensures energy conservation and creates separate
Load components for each demand type.
"""

import pandas as pd
import pypsa
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging

# ──────────────────────────────────────────────────────────────────────────────
# Main Processing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger = setup_logging(
        log_path=snakemake.log[0],
        log_level="INFO"
    )
    
    try:
        logger.info("=" * 80)
        logger.info("INTEGRATING DISAGGREGATED DEMAND COMPONENTS")
        logger.info("=" * 80)
        
        # ──── Load Base Network ────
        logger.info("Loading base demand network...")
        network = pypsa.Network(snakemake.input.base_demand)
        base_profile = pd.read_csv(snakemake.input.base_profile, index_col=0, parse_dates=True)
        
        logger.info(f"Base network: {len(network.buses)} buses, {len(network.loads)} loads")
        logger.info(f"Base profile shape: {base_profile.shape}")
        
        # Get total base demand
        total_base_demand_gwh = base_profile.sum().sum()
        logger.info(f"Total base demand: {total_base_demand_gwh:.1f} GWh/year")
        
        # ──── Load Component Data ────
        component_names = snakemake.params.component_names
        logger.info(f"Loading {len(component_names)} components: {component_names}")
        
        component_data = {}
        total_component_demand = 0.0
        
        for idx, component_name in enumerate(component_names):
            logger.info(f"Loading component {idx + 1}/{len(component_names)}: {component_name}")
            
            # Load profile and allocation
            profile_path = snakemake.input.component_profiles[idx]
            allocation_path = snakemake.input.component_allocations[idx]
            
            profile = pd.read_csv(profile_path, index_col=0, parse_dates=True)
            allocation = pd.read_csv(allocation_path)
            
            component_total = profile.sum().sum()
            total_component_demand += component_total
            
            component_data[component_name] = {
                'profile': profile,
                'allocation': allocation,
                'total_gwh': component_total
            }
            
            logger.info(f"  {component_name}: {component_total:.1f} GWh/year")
        
        logger.info(f"Total component demand: {total_component_demand:.1f} GWh/year")
        logger.info(f"Component fraction: {total_component_demand / total_base_demand_gwh:.1%} of base")
        
        # ──── Adjust Base Demand ────
        logger.info("Adjusting base demand to remove component totals...")
        
        # Reduce base demand by component totals (to avoid double-counting)
        adjustment_factor = (total_base_demand_gwh - total_component_demand) / total_base_demand_gwh
        
        logger.info(f"Base demand adjustment factor: {adjustment_factor:.4f}")
        
        if adjustment_factor < 0:
            logger.error(f"Component demand ({total_component_demand:.1f} GWh) exceeds base demand!")
            raise ValueError("Component demand exceeds total demand - check fractions in config")
        
        if adjustment_factor < 0.5:
            logger.warning(f"Components represent {(1 - adjustment_factor):.1%} of demand - this seems high!")
        
        # Adjust all existing loads in the network
        if len(network.loads_t.p_set) > 0:
            network.loads_t.p_set = network.loads_t.p_set * adjustment_factor
            logger.info("Adjusted time-varying loads")
        
        if 'p_set' in network.loads.columns:
            network.loads.p_set = network.loads.p_set * adjustment_factor
            logger.info("Adjusted static loads")
        
        adjusted_demand = total_base_demand_gwh * adjustment_factor
        logger.info(f"Adjusted base demand: {adjusted_demand:.1f} GWh/year")
        
        # ──── Add Component Loads to Network ────
        logger.info("Adding component loads to network...")
        
        for component_name, data in component_data.items():
            logger.info(f"Adding {component_name} loads...")
            
            profile = data['profile']
            allocation = data['allocation']
            
            # Get bus column (handle different column names)
            bus_col = 'bus' if 'bus' in allocation.columns else allocation.columns[0]
            demand_col = [c for c in allocation.columns if c != bus_col][0]
            
            buses_added = 0
            for _, row in allocation.iterrows():
                bus_id = str(row[bus_col])
                annual_demand_gwh = row[demand_col]
                
                # Skip if bus not in network
                if bus_id not in network.buses.index:
                    logger.debug(f"  Skipping {bus_id} - not in network")
                    continue
                
                # Skip if demand is negligible
                if annual_demand_gwh < 0.001:  # Less than 1 MWh
                    continue
                
                # Create load name
                load_name = f"{component_name}_{bus_id}"
                
                try:
                    # Add load component
                    network.add(
                        "Load",
                        load_name,
                        bus=bus_id,
                        carrier=component_name
                    )
                    buses_added += 1
                    
                except Exception as e:
                    logger.warning(f"  Could not add load {load_name}: {e}")
                    continue
            
            logger.info(f"  Added {buses_added} {component_name} loads to network")
            
            # Add component timeseries
            # Scale profile to match spatial allocation
            bus_fractions = allocation.set_index(bus_col)[demand_col] / data['total_gwh']
            
            for bus_id, fraction in bus_fractions.items():
                load_name = f"{component_name}_{bus_id}"
                
                if load_name in network.loads.index:
                    # Create timeseries for this bus
                    bus_timeseries = profile.iloc[:, 0] * fraction
                    network.loads_t.p_set[load_name] = bus_timeseries.values
            
            logger.info(f"  Added timeseries for {component_name}")
        
        # ──── Validation ────
        logger.info("Validating final network...")
        
        # Calculate total demand in final network
        if len(network.loads_t.p_set) > 0:
            final_total_demand = network.loads_t.p_set.sum(axis=1).sum()
        else:
            final_total_demand = network.loads.p_set.sum()
        
        logger.info(f"Final total demand: {final_total_demand:.1f} GWh/year")
        logger.info(f"Original base demand: {total_base_demand_gwh:.1f} GWh/year")
        logger.info(f"Difference: {abs(final_total_demand - total_base_demand_gwh):.1f} GWh")
        
        tolerance = 0.1  # 100 MWh tolerance
        if abs(final_total_demand - total_base_demand_gwh) > tolerance:
            logger.warning(
                f"Energy balance check FAILED! "
                f"Difference: {abs(final_total_demand - total_base_demand_gwh):.1f} GWh > {tolerance} GWh"
            )
        else:
            logger.info("Energy balance check: PASSED ✓")
        
        # ──── Create Summary ────
        logger.info("Creating component summary...")
        
        summary_data = []
        for component_name, data in component_data.items():
            summary_data.append({
                'component': component_name,
                'total_gwh': data['total_gwh'],
                'fraction_of_base': data['total_gwh'] / total_base_demand_gwh,
                'num_buses': len(data['allocation']),
                'num_loads': len([l for l in network.loads.index if l.startswith(component_name)])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(snakemake.output.component_summary, index=False)
        logger.info(f"Saved component summary to {snakemake.output.component_summary}")
        
        # ──── Save Final Network ────
        logger.info("Saving final network...")
        network.export_to_netcdf(snakemake.output.final_network)
        logger.info(f"Saved final network to {snakemake.output.final_network}")
        
        # ──── Summary ────
        logger.info("=" * 80)
        logger.info("INTEGRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Components integrated: {len(component_names)}")
        for component_name, data in component_data.items():
            fraction = data['total_gwh'] / total_base_demand_gwh
            logger.info(f"  {component_name}: {data['total_gwh']:.1f} GWh ({fraction:.1%})")
        logger.info(f"Adjusted base demand: {adjusted_demand:.1f} GWh ({adjustment_factor:.1%})")
        logger.info(f"Final total demand: {final_total_demand:.1f} GWh")
        logger.info(f"Total loads in network: {len(network.loads)}")
        logger.info("=" * 80)
        logger.info("INTEGRATION COMPLETED SUCCESSFULLY ✓")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in integration: {e}", exc_info=True)
        raise

