"""
Heat Pump Demand Disaggregation

This script disaggregates heat pump electricity demand from the total demand.
It loads heat pump profiles, scales them to match the configured fraction of
total demand, and allocates them spatially across the network.
"""

import pandas as pd
import pypsa
import logging
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging

# ──────────────────────────────────────────────────────────────────────────────
# Spatial Allocation Functions
# ──────────────────────────────────────────────────────────────────────────────

def allocate_proportional(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
    """
    Allocate heat pump demand proportionally to existing base demand.
    
    This assumes heat pump adoption correlates with overall electricity consumption.
    """
    logger.info("Allocating proportionally to base demand")
    
    # Get base demand per bus
    if len(base_network.loads_t.p_set) > 0:
        bus_demand = base_network.loads_t.p_set.sum(axis=0)  # Sum over time
    else:
        logger.warning("No time-varying loads found, using static p_set")
        bus_demand = base_network.loads.p_set
    
    # Calculate fractions
    total_base = bus_demand.sum()
    if total_base == 0:
        logger.error("Total base demand is zero - cannot allocate proportionally")
        raise ValueError("Zero base demand")
    
    bus_fractions = bus_demand / total_base
    hp_allocation = bus_fractions * total_gwh
    
    logger.info(f"Allocated {hp_allocation.sum():.1f} GWh across {len(hp_allocation)} buses")
    logger.info(f"Min: {hp_allocation.min():.3f} GWh, Max: {hp_allocation.max():.3f} GWh")
    
    return hp_allocation


def allocate_uniform(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
    """
    Allocate heat pump demand uniformly across all buses.
    
    This assumes uniform heat pump adoption regardless of location.
    """
    logger.info("Allocating uniformly across all buses")
    
    n_buses = len(base_network.buses)
    hp_per_bus = total_gwh / n_buses
    hp_allocation = pd.Series(hp_per_bus, index=base_network.buses.index)
    
    logger.info(f"Allocated {hp_allocation.sum():.1f} GWh uniformly ({hp_per_bus:.3f} GWh per bus)")
    
    return hp_allocation


def allocate_urban_weighted(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
    """
    Allocate heat pump demand weighted towards urban areas (high demand buses).
    
    Uses a power function to weight allocation towards higher-demand areas.
    """
    logger.info("Allocating with urban weighting")
    
    # Get base demand per bus
    if len(base_network.loads_t.p_set) > 0:
        bus_demand = base_network.loads_t.p_set.sum(axis=0)
    else:
        bus_demand = base_network.loads.p_set
    
    # Apply power weighting (exponent > 1 favors high-demand areas)
    urban_weight = 1.5
    weighted_demand = bus_demand ** urban_weight
    
    total_weighted = weighted_demand.sum()
    if total_weighted == 0:
        logger.warning("Weighted demand is zero, falling back to proportional")
        return allocate_proportional(total_gwh, base_network, logger)
    
    bus_fractions = weighted_demand / total_weighted
    hp_allocation = bus_fractions * total_gwh
    
    logger.info(f"Allocated {hp_allocation.sum():.1f} GWh with urban weighting (exponent={urban_weight})")
    
    return hp_allocation


# Allocation method registry
ALLOCATION_METHODS = {
    'proportional': allocate_proportional,
    'uniform': allocate_uniform,
    'urban_weighted': allocate_urban_weighted,
}

# ──────────────────────────────────────────────────────────────────────────────
# Main Processing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(
        log_path=snakemake.log[0],
        log_level="INFO"
    )
    
    try:
        logger.info("=" * 80)
        logger.info("HEAT PUMP DEMAND DISAGGREGATION")
        logger.info("=" * 80)
        
        # ──── Load Inputs ────
        logger.info("Loading inputs...")
        base_network = pypsa.Network(snakemake.input.base_demand)
        base_profile = pd.read_csv(snakemake.input.base_profile, index_col=0, parse_dates=True)
        
        logger.info(f"Base network: {len(base_network.buses)} buses, {len(base_network.loads)} loads")
        logger.info(f"Base profile shape: {base_profile.shape}")
        
        # ──── Get Configuration ────
        config = snakemake.params.component_config
        fraction = config.get("fraction_of_total", 0.15)
        allocation_method = config.get("allocation_method", "proportional")
        source_file = config.get("source_file", "")
        
        logger.info(f"Configuration:")
        logger.info(f"  Fraction of total demand: {fraction:.1%}")
        logger.info(f"  Allocation method: {allocation_method}")
        logger.info(f"  Source file: {source_file}")
        
        # ──── Calculate Total HP Demand ────
        logger.info("Calculating heat pump demand...")
        
        # Get total base demand from profile
        total_base_demand_gwh = base_profile.sum().sum()
        total_hp_demand_gwh = total_base_demand_gwh * fraction
        
        logger.info(f"Total base demand: {total_base_demand_gwh:.1f} GWh/year")
        logger.info(f"Target HP demand: {total_hp_demand_gwh:.1f} GWh/year ({fraction:.1%})")
        
        # ──── Load or Generate HP Profile ────
        logger.info("Processing heat pump profile...")
        
        hp_source_path = Path(source_file)
        if hp_source_path.exists():
            # Load actual HP profile data
            logger.info(f"Loading HP profile from {hp_source_path}")
            hp_raw = pd.read_csv(hp_source_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded HP profile shape: {hp_raw.shape}")
            
            # Normalize and scale to target total
            hp_normalized = hp_raw / hp_raw.sum().sum()
            hp_profile = hp_normalized * total_hp_demand_gwh
            
        else:
            # Generate synthetic HP profile (for testing/skeleton mode)
            logger.warning(f"HP data file not found: {hp_source_path}")
            logger.info("Generating synthetic HP profile based on temperature patterns")
            
            # Create synthetic profile with higher demand in winter
            # This is a placeholder - real implementation would use actual HP data
            n_timesteps = len(base_profile)
            
            # Simple seasonal pattern (higher in winter)
            timesteps = np.arange(n_timesteps)
            seasonal_pattern = 1.0 + 0.5 * np.cos(2 * np.pi * timesteps / (365 * 48))  # Half-hourly
            daily_pattern = 1.0 + 0.3 * np.cos(2 * np.pi * (timesteps % 48) / 48)  # Daily cycle
            
            synthetic_profile = seasonal_pattern * daily_pattern
            synthetic_profile = synthetic_profile / synthetic_profile.sum() * total_hp_demand_gwh
            
            hp_profile = pd.DataFrame(
                synthetic_profile,
                index=base_profile.index,
                columns=['heat_pump_demand_gwh']
            )
            
            logger.info(f"Generated synthetic HP profile: {hp_profile.shape}")
        
        logger.info(f"HP profile total: {hp_profile.sum().sum():.1f} GWh (check: {total_hp_demand_gwh:.1f} GWh)")
        
        # ──── Spatial Allocation ────
        logger.info(f"Allocating HP demand across buses using '{allocation_method}' method...")
        
        if allocation_method not in ALLOCATION_METHODS:
            logger.warning(f"Unknown allocation method '{allocation_method}', using 'proportional'")
            allocation_method = 'proportional'
        
        allocator = ALLOCATION_METHODS[allocation_method]
        hp_allocation = allocator(total_hp_demand_gwh, base_network, logger)
        
        # ──── Validation ────
        logger.info("Validating outputs...")
        
        profile_total = hp_profile.sum().sum()
        allocation_total = hp_allocation.sum()
        
        tolerance = 0.01  # 0.01 GWh = 10 MWh
        if abs(profile_total - allocation_total) > tolerance:
            logger.warning(
                f"Energy balance mismatch! "
                f"Profile: {profile_total:.3f} GWh, "
                f"Allocation: {allocation_total:.3f} GWh, "
                f"Difference: {abs(profile_total - allocation_total):.3f} GWh"
            )
        else:
            logger.info(f"Energy balance check: PASSED ✓ (difference < {tolerance} GWh)")
        
        # Check that allocation matches network topology
        network_buses = set(base_network.buses.index)
        allocation_buses = set(hp_allocation.index)
        
        if network_buses != allocation_buses:
            missing = network_buses - allocation_buses
            extra = allocation_buses - network_buses
            if missing:
                logger.warning(f"{len(missing)} buses in network but not in allocation")
            if extra:
                logger.warning(f"{len(extra)} buses in allocation but not in network")
        else:
            logger.info("Bus topology check: PASSED ✓")
        
        # ──── Save Outputs ────
        logger.info("Saving outputs...")
        
        # Save profile (timeseries)
        hp_profile.to_csv(snakemake.output.profile)
        logger.info(f"Saved HP profile to {snakemake.output.profile}")
        
        # Save allocation (spatial distribution)
        hp_allocation_df = pd.DataFrame({
            'bus': hp_allocation.index,
            'heat_pump_demand_gwh': hp_allocation.values
        })
        hp_allocation_df.to_csv(snakemake.output.allocation, index=False)
        logger.info(f"Saved HP allocation to {snakemake.output.allocation}")
        
        # ──── Summary ────
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total HP demand: {total_hp_demand_gwh:.1f} GWh/year")
        logger.info(f"Fraction of base: {fraction:.1%}")
        logger.info(f"Number of buses: {len(hp_allocation)}")
        logger.info(f"Allocation method: {allocation_method}")
        logger.info(f"Profile timesteps: {len(hp_profile)}")
        logger.info(f"Min bus demand: {hp_allocation.min():.3f} GWh")
        logger.info(f"Max bus demand: {hp_allocation.max():.3f} GWh")
        logger.info(f"Mean bus demand: {hp_allocation.mean():.3f} GWh")
        logger.info("=" * 80)
        logger.info("HEAT PUMP DISAGGREGATION COMPLETED SUCCESSFULLY ✓")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in heat pump disaggregation: {e}", exc_info=True)
        raise

