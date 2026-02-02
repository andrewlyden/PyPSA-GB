"""
Event-Based Demand Flexibility (Saving Sessions)

This script implements event-based demand response where consumers receive
advance notice to reduce demand during specific time windows.

Based on National Grid ESO's "Demand Flexibility Service" and Octopus Energy's
"Saving Sessions" programs.

Event Types:
- Regular: ~2 events per week throughout year
- Winter: ~5 events per week during winter months (Oct-Mar)

PyPSA Representation:
- Generator with negative p (demand reduction capability)
- p_max_pu timeseries limits when events can occur
- Marginal cost represents incentive payment to consumers
"""

import logging
import numpy as np
import pandas as pd
import pypsa
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Default event parameters
DEFAULT_EVENT_WINDOW = ['17:00', '19:00']  # Peak demand hours
DEFAULT_PARTICIPATION_RATE = 0.33  # 33% of eligible consumers participate
DEFAULT_MAX_REDUCTION = 0.10  # 10% demand reduction per event
DEFAULT_WINTER_MONTHS = [10, 11, 12, 1, 2, 3]  # October to March

# GB population and household data for scaling
GB_POPULATION = 67_330_000
GB_HOUSEHOLD_OCCUPANCY = 2.36
GB_HOUSEHOLDS = GB_POPULATION / GB_HOUSEHOLD_OCCUPANCY

# Typical household demand reduction during events (kW per household)
HOUSEHOLD_TURNDOWN_KW = 0.3  # Based on Saving Sessions data


# ──────────────────────────────────────────────────────────────────────────────
# Event Schedule Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_event_schedule(snapshots: pd.DatetimeIndex,
                            mode: str,
                            events_per_week_regular: int = 2,
                            events_per_week_winter: int = 5,
                            winter_months: List[int] = None,
                            event_window_start: int = 17,
                            event_window_end: int = 19,
                            logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Generate event availability schedule.

    Events are scheduled during peak hours (typically 17:00-19:00) on weekdays.
    The number of events per week varies by mode and season.

    Args:
        snapshots: DatetimeIndex for the model period
        mode: Event mode - 'regular', 'winter', or 'both'
        events_per_week_regular: Events per week in regular mode
        events_per_week_winter: Events per week in winter mode
        winter_months: List of winter month numbers (1-12)
        event_window_start: Hour when event window opens
        event_window_end: Hour when event window closes
        logger: Logger instance

    Returns:
        Series with 1.0 during possible event times, 0.0 otherwise
    """
    if winter_months is None:
        winter_months = DEFAULT_WINTER_MONTHS

    if logger:
        logger.info(f"Generating event schedule, mode: {mode}")
        logger.info(f"Event window: {event_window_start:02d}:00 - {event_window_end:02d}:00")

    # Initialize schedule (0 = no event possible)
    schedule = pd.Series(0.0, index=snapshots)

    for idx in snapshots:
        hour = idx.hour
        month = idx.month
        day_of_week = idx.dayofweek  # 0=Monday, 6=Sunday
        week_of_year = idx.isocalendar()[1]

        # Only allow events during the event window
        if not (event_window_start <= hour < event_window_end):
            continue

        # Only allow events on weekdays
        if day_of_week >= 5:  # Saturday or Sunday
            continue

        # Determine if this is a winter month
        is_winter = month in winter_months

        # Determine events per week based on mode and season
        if mode == 'regular':
            events_this_week = events_per_week_regular
        elif mode == 'winter':
            events_this_week = events_per_week_winter if is_winter else 0
        elif mode == 'both':
            if is_winter:
                events_this_week = events_per_week_winter
            else:
                events_this_week = events_per_week_regular
        else:
            events_this_week = events_per_week_regular

        if events_this_week == 0:
            continue

        # Simple scheduling: spread events across weekdays
        # Use hash of week number to vary which days have events
        np.random.seed(week_of_year * 100 + month)
        event_days = np.random.choice(5, size=min(events_this_week, 5), replace=False)

        if day_of_week in event_days:
            schedule[idx] = 1.0

    if logger:
        total_event_hours = schedule.sum()
        logger.info(f"Total event hours scheduled: {total_event_hours:.0f}")

    return schedule


# ──────────────────────────────────────────────────────────────────────────────
# Demand Reduction Calculation
# ──────────────────────────────────────────────────────────────────────────────

def calculate_demand_reduction_capacity(base_demand_mw: pd.DataFrame,
                                        participation_rate: float,
                                        max_reduction_fraction: float,
                                        logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Calculate maximum demand reduction capacity per bus.

    The reduction capacity is a fraction of base demand that can be curtailed
    during events, limited by participation rate.

    Args:
        base_demand_mw: Base electricity demand (MW) by bus
        participation_rate: Fraction of consumers participating
        max_reduction_fraction: Maximum fraction of demand that can be reduced
        logger: Logger instance

    Returns:
        Maximum demand reduction capacity (MW) by bus
    """
    if logger:
        logger.info(f"Calculating demand reduction capacity...")
        logger.info(f"Participation rate: {participation_rate*100:.0f}%")
        logger.info(f"Max reduction: {max_reduction_fraction*100:.0f}%")

    # Maximum reduction = base demand * participation * max reduction
    reduction_capacity = base_demand_mw * participation_rate * max_reduction_fraction

    if logger:
        total_capacity = reduction_capacity.max().sum()
        logger.info(f"Total peak reduction capacity: {total_capacity:.1f} MW")

    return reduction_capacity


# ──────────────────────────────────────────────────────────────────────────────
# PyPSA Component Creation
# ──────────────────────────────────────────────────────────────────────────────

def add_event_flexibility(n: pypsa.Network,
                          base_demand_mw: pd.DataFrame,
                          config: Dict[str, Any],
                          logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Add event-based demand flexibility to network.

    Creates Generator components that represent demand reduction capability.
    The generators have:
    - p_nom: Maximum reduction capacity
    - p_max_pu: Time-varying availability (only during events)
    - marginal_cost: Incentive payment (high cost = last resort)

    Args:
        n: PyPSA network
        base_demand_mw: Base demand by bus (for scaling)
        config: Event response configuration
        logger: Logger instance

    Returns:
        Network with event flexibility components
    """
    if logger:
        logger.info("=" * 80)
        logger.info("ADDING EVENT-BASED DEMAND FLEXIBILITY")
        logger.info("=" * 80)

    # Get configuration
    mode = config.get('mode', 'regular')
    event_window = config.get('event_window', DEFAULT_EVENT_WINDOW)
    participation_rate = config.get('participation_rate', DEFAULT_PARTICIPATION_RATE)
    max_reduction = config.get('max_reduction_fraction', DEFAULT_MAX_REDUCTION)
    winter_months = config.get('winter_months', DEFAULT_WINTER_MONTHS)

    window_start = int(event_window[0].split(':')[0])
    window_end = int(event_window[1].split(':')[0])

    if logger:
        logger.info(f"Mode: {mode}")
        logger.info(f"Event window: {window_start:02d}:00 - {window_end:02d}:00")
        logger.info(f"Participation rate: {participation_rate*100:.0f}%")
        logger.info(f"Max reduction: {max_reduction*100:.0f}%")

    # Generate event schedule
    event_schedule = generate_event_schedule(
        snapshots=n.snapshots,
        mode=mode,
        winter_months=winter_months,
        event_window_start=window_start,
        event_window_end=window_end,
        logger=logger
    )

    # Calculate reduction capacity
    reduction_capacity = calculate_demand_reduction_capacity(
        base_demand_mw=base_demand_mw,
        participation_rate=participation_rate,
        max_reduction_fraction=max_reduction,
        logger=logger
    )

    # Add demand reduction generators for each bus
    buses = [bus for bus in base_demand_mw.columns if bus in n.buses.index]

    # Add 'demand response' carrier if not defined
    if 'demand response' not in n.carriers.index:
        n.add("Carrier", "demand response", co2_emissions=0.0, nice_name="Demand Response")

    for bus in buses:
        gen_name = f"{bus} demand response"

        # Maximum reduction capacity at this bus
        p_nom = reduction_capacity[bus].max()

        if p_nom < 0.001:  # Skip buses with negligible capacity
            continue

        # Add generator representing demand reduction
        # Using Generator with carrier="demand response"
        n.add("Generator",
              gen_name,
              bus=bus,
              carrier="demand response",
              p_nom=p_nom,
              p_nom_extendable=False,
              p_max_pu=event_schedule,  # Only available during events
              p_min_pu=0.0,
              marginal_cost=500.0)  # High cost = only use when needed

    if logger:
        n_gens = len([g for g in n.generators.index if 'demand response' in g])
        logger.info(f"Added {n_gens} demand response generators")
        logger.info("=" * 80)
        logger.info("EVENT FLEXIBILITY ADDED SUCCESSFULLY")
        logger.info("=" * 80)

    return n


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Functions for Analysis
# ──────────────────────────────────────────────────────────────────────────────

def estimate_event_savings(event_schedule: pd.Series,
                           reduction_capacity_mw: float,
                           duration_hours: int = 2,
                           logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    Estimate potential energy savings from demand response events.

    Args:
        event_schedule: Event availability schedule (0/1)
        reduction_capacity_mw: Maximum demand reduction (MW)
        duration_hours: Duration of each event (hours)
        logger: Logger instance

    Returns:
        Dictionary with savings estimates
    """
    # Count event hours
    total_event_hours = event_schedule.sum()

    # Estimate actual reduction (assume ~50% of capacity is utilized on average)
    utilization_factor = 0.5
    energy_saved_mwh = total_event_hours * reduction_capacity_mw * utilization_factor

    results = {
        'total_event_hours': total_event_hours,
        'reduction_capacity_mw': reduction_capacity_mw,
        'estimated_energy_saved_mwh': energy_saved_mwh,
        'estimated_energy_saved_gwh': energy_saved_mwh / 1000,
    }

    if logger:
        logger.info(f"Event savings estimate:")
        logger.info(f"  Total event hours: {total_event_hours:.0f}")
        logger.info(f"  Reduction capacity: {reduction_capacity_mw:.1f} MW")
        logger.info(f"  Estimated energy saved: {energy_saved_mwh:.1f} MWh")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Snakemake Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(
        log_path=snakemake.log[0],
        log_level="INFO"
    )

    try:
        logger.info("=" * 80)
        logger.info("EVENT FLEXIBILITY PROCESSING")
        logger.info("=" * 80)

        # Load network
        n = pypsa.Network(snakemake.input.network)
        logger.info(f"Loaded network with {len(n.buses)} buses")

        # Get base demand from loads
        if len(n.loads_t.p_set) > 0:
            base_demand_mw = n.loads_t.p_set.copy()
        else:
            # Create from static loads
            base_demand_mw = pd.DataFrame(
                {load: [n.loads.loc[load, 'p_set']] * len(n.snapshots)
                 for load in n.loads.index},
                index=n.snapshots
            )

        # Get configuration
        config = snakemake.params.get('event_config', {})

        # Add event flexibility
        n = add_event_flexibility(n, base_demand_mw, config, logger)

        # Save network
        n.export_to_netcdf(snakemake.output.network)
        logger.info(f"Saved network to {snakemake.output.network}")

        logger.info("=" * 80)
        logger.info("EVENT FLEXIBILITY COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in event flexibility: {e}", exc_info=True)
        raise
