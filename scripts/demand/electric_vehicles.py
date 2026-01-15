"""
Electric Vehicle Charging Demand Disaggregation

This script disaggregates EV charging electricity demand from the total demand.
It loads EV charging profiles, scales them to match the configured fraction of
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
    """Allocate EV demand proportionally to existing base demand."""
    logger.info("Allocating proportionally to base demand")
    
    if len(base_network.loads_t.p_set) > 0:
        bus_demand = base_network.loads_t.p_set.sum(axis=0)
    else:
        bus_demand = base_network.loads.p_set
    
    total_base = bus_demand.sum()
    if total_base == 0:
        raise ValueError("Zero base demand")
    
    bus_fractions = bus_demand / total_base
    ev_allocation = bus_fractions * total_gwh
    
    logger.info(f"Allocated {ev_allocation.sum():.1f} GWh across {len(ev_allocation)} buses")
    return ev_allocation


def allocate_uniform(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
    """Allocate EV demand uniformly across all buses."""
    logger.info("Allocating uniformly across all buses")
    
    n_buses = len(base_network.buses)
    ev_per_bus = total_gwh / n_buses
    ev_allocation = pd.Series(ev_per_bus, index=base_network.buses.index)
    
    logger.info(f"Allocated {ev_allocation.sum():.1f} GWh uniformly")
    return ev_allocation


def allocate_urban_weighted(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
    """Allocate EV demand weighted towards urban areas (high demand buses)."""
    logger.info("Allocating with urban weighting")
    
    if len(base_network.loads_t.p_set) > 0:
        bus_demand = base_network.loads_t.p_set.sum(axis=0)
    else:
        bus_demand = base_network.loads.p_set
    
    # EVs concentrated in urban areas - use higher exponent
    urban_weight = 2.0
    weighted_demand = bus_demand ** urban_weight
    
    total_weighted = weighted_demand.sum()
    if total_weighted == 0:
        logger.warning("Weighted demand is zero, falling back to proportional")
        return allocate_proportional(total_gwh, base_network, logger)
    
    bus_fractions = weighted_demand / total_weighted
    ev_allocation = bus_fractions * total_gwh
    
    logger.info(f"Allocated {ev_allocation.sum():.1f} GWh with urban weighting")
    return ev_allocation


ALLOCATION_METHODS = {
    'proportional': allocate_proportional,
    'uniform': allocate_uniform,
    'urban_weighted': allocate_urban_weighted,
}

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
        logger.info("ELECTRIC VEHICLE DEMAND DISAGGREGATION")
        logger.info("=" * 80)
        
        # ──── Load Inputs ────
        logger.info("Loading inputs...")
        base_network = pypsa.Network(snakemake.input.base_demand)
        base_profile = pd.read_csv(snakemake.input.base_profile, index_col=0, parse_dates=True)
        
        logger.info(f"Base network: {len(base_network.buses)} buses, {len(base_network.loads)} loads")
        logger.info(f"Base profile shape: {base_profile.shape}")
        
        # ──── Get Configuration ────
        config = snakemake.params.component_config
        fraction = config.get("fraction_of_total", 0.08)
        allocation_method = config.get("allocation_method", "urban_weighted")
        source_file = config.get("source_file", "")
        
        logger.info(f"Configuration:")
        logger.info(f"  Fraction of total demand: {fraction:.1%}")
        logger.info(f"  Allocation method: {allocation_method}")
        logger.info(f"  Source file: {source_file}")
        
        # ──── Calculate Total EV Demand ────
        logger.info("Calculating EV charging demand...")
        
        total_base_demand_gwh = base_profile.sum().sum()
        total_ev_demand_gwh = total_base_demand_gwh * fraction
        
        logger.info(f"Total base demand: {total_base_demand_gwh:.1f} GWh/year")
        logger.info(f"Target EV demand: {total_ev_demand_gwh:.1f} GWh/year ({fraction:.1%})")
        
        # ──── Load or Generate EV Profile ────
        logger.info("Processing EV charging profile...")
        
        ev_source_path = Path(source_file)
        if ev_source_path.exists():
            logger.info(f"Loading EV profile from {ev_source_path}")
            ev_raw = pd.read_csv(ev_source_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded EV profile shape: {ev_raw.shape}")
            
            ev_normalized = ev_raw / ev_raw.sum().sum()
            ev_profile = ev_normalized * total_ev_demand_gwh
            
        else:
            logger.warning(f"EV data file not found: {ev_source_path}")
            logger.info("Generating synthetic EV profile based on typical charging patterns")
            
            # Generate synthetic EV profile with peak evening charging
            n_timesteps = len(base_profile)
            timesteps = np.arange(n_timesteps)
            
            # Daily charging pattern (peaks in evening ~18:00-22:00)
            hour_of_day = (timesteps % 48) / 2  # Convert to hour (half-hourly data)
            evening_peak = np.exp(-((hour_of_day - 20) ** 2) / (2 * 2 ** 2))  # Peak at 20:00
            
            # Weekly pattern (higher on weekdays)
            day_of_week = (timesteps // 48) % 7
            weekday_factor = np.where(day_of_week < 5, 1.2, 0.8)  # Higher Mon-Fri
            
            synthetic_profile = evening_peak * weekday_factor
            synthetic_profile = synthetic_profile / synthetic_profile.sum() * total_ev_demand_gwh
            
            ev_profile = pd.DataFrame(
                synthetic_profile,
                index=base_profile.index,
                columns=['ev_charging_demand_gwh']
            )
            
            logger.info(f"Generated synthetic EV profile: {ev_profile.shape}")
        
        logger.info(f"EV profile total: {ev_profile.sum().sum():.1f} GWh")
        
        # ──── Spatial Allocation ────
        logger.info(f"Allocating EV demand using '{allocation_method}' method...")
        
        if allocation_method not in ALLOCATION_METHODS:
            logger.warning(f"Unknown allocation method '{allocation_method}', using 'urban_weighted'")
            allocation_method = 'urban_weighted'
        
        allocator = ALLOCATION_METHODS[allocation_method]
        ev_allocation = allocator(total_ev_demand_gwh, base_network, logger)
        
        # ──── Validation ────
        logger.info("Validating outputs...")
        
        profile_total = ev_profile.sum().sum()
        allocation_total = ev_allocation.sum()
        
        tolerance = 0.01
        if abs(profile_total - allocation_total) > tolerance:
            logger.warning(
                f"Energy mismatch! Profile: {profile_total:.3f} GWh, "
                f"Allocation: {allocation_total:.3f} GWh"
            )
        else:
            logger.info("Energy balance check: PASSED ✓")
        
        # ──── Save Outputs ────
        logger.info("Saving outputs...")
        
        ev_profile.to_csv(snakemake.output.profile)
        logger.info(f"Saved EV profile to {snakemake.output.profile}")
        
        ev_allocation_df = pd.DataFrame({
            'bus': ev_allocation.index,
            'ev_charging_demand_gwh': ev_allocation.values
        })
        ev_allocation_df.to_csv(snakemake.output.allocation, index=False)
        logger.info(f"Saved EV allocation to {snakemake.output.allocation}")
        
        # ──── Summary ────
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total EV demand: {total_ev_demand_gwh:.1f} GWh/year")
        logger.info(f"Fraction of base: {fraction:.1%}")
        logger.info(f"Number of buses: {len(ev_allocation)}")
        logger.info(f"Allocation method: {allocation_method}")
        logger.info(f"Min bus demand: {ev_allocation.min():.3f} GWh")
        logger.info(f"Max bus demand: {ev_allocation.max():.3f} GWh")
        logger.info("=" * 80)
        logger.info("EV DISAGGREGATION COMPLETED SUCCESSFULLY ✓")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in EV disaggregation: {e}", exc_info=True)
        raise

