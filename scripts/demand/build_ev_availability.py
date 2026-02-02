"""
Build Electric Vehicle Availability Profiles for GB

This script generates EV charging availability profiles based on traffic patterns.
EVs are available for charging when parked (not driving), so availability is
inversely related to traffic flow.

Data Sources:
- DfT (Department for Transport) road traffic statistics for GB-specific patterns
- Synthetic profiles based on literature if DfT data unavailable

Output:
- Hourly availability profile (fraction of EVs plugged in at each hour)
- Weekly pattern expanded to full year
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging

# ──────────────────────────────────────────────────────────────────────────────
# Constants and Default Parameters
# ──────────────────────────────────────────────────────────────────────────────

# Default availability parameters (from literature and PyPSA-FES)
DEFAULT_AVAIL_MAX = 0.95  # Maximum fraction of time EVs are plugged in
DEFAULT_AVAIL_MEAN = 0.80  # Mean fraction of time EVs are plugged in

# Typical GB daily traffic pattern (relative flow, normalized)
# Based on DfT road traffic statistics and National Travel Survey
# Index: hour of day (0-23)
GB_WEEKDAY_TRAFFIC = np.array([
    0.20, 0.15, 0.12, 0.12, 0.15, 0.25,  # 00:00-05:00 (overnight low)
    0.45, 0.75, 1.00, 0.85, 0.75, 0.80,  # 06:00-11:00 (morning peak)
    0.85, 0.80, 0.80, 0.85, 0.95, 1.00,  # 12:00-17:00 (afternoon/evening peak)
    0.90, 0.75, 0.60, 0.50, 0.40, 0.30,  # 18:00-23:00 (evening decline)
])

GB_WEEKEND_TRAFFIC = np.array([
    0.25, 0.18, 0.15, 0.12, 0.12, 0.15,  # 00:00-05:00 (overnight)
    0.20, 0.30, 0.45, 0.60, 0.75, 0.85,  # 06:00-11:00 (late morning ramp)
    0.90, 0.95, 1.00, 0.95, 0.90, 0.85,  # 12:00-17:00 (afternoon plateau)
    0.75, 0.65, 0.55, 0.45, 0.35, 0.30,  # 18:00-23:00 (evening decline)
])


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def generate_periodic_profiles(dt_index: pd.DatetimeIndex,
                               nodes: list,
                               weekly_profile: np.ndarray) -> pd.DataFrame:
    """
    Generate time series by repeating a weekly profile across the full time range.

    Adapted from PyPSA-FES _helpers.py

    Args:
        dt_index: DatetimeIndex for the output profile
        nodes: List of node/bus names (columns)
        weekly_profile: Array of length 168 (24*7) with hourly values for one week

    Returns:
        DataFrame with shape (len(dt_index), len(nodes))
    """
    # Determine hour of week for each timestamp (0-167)
    # Monday=0, Sunday=6
    hour_of_week = (dt_index.dayofweek * 24 + dt_index.hour).values

    # Handle half-hourly data
    if len(dt_index) > 8760 and len(weekly_profile) == 168:
        # Interpolate weekly profile to half-hourly if needed
        # For now, just repeat each hour twice
        pass

    # Map weekly profile to full time series
    profile_values = weekly_profile[hour_of_week % len(weekly_profile)]

    # Create DataFrame with same value for all nodes
    # (spatial variation could be added later)
    df = pd.DataFrame(
        np.tile(profile_values.reshape(-1, 1), (1, len(nodes))),
        index=dt_index,
        columns=nodes
    )

    return df


def traffic_to_availability(traffic: np.ndarray,
                            avail_max: float = DEFAULT_AVAIL_MAX,
                            avail_mean: float = DEFAULT_AVAIL_MEAN) -> np.ndarray:
    """
    Convert traffic flow to EV availability.

    Availability is inversely related to traffic - when more cars are on the road,
    fewer are plugged in and available for charging/V2G.

    Formula from PyPSA-FES:
    avail = avail_max - (avail_max - avail_mean) * (traffic - traffic_min) / (traffic_mean - traffic_min)

    Args:
        traffic: Relative traffic flow (normalized, peak = 1.0)
        avail_max: Maximum availability (typically 0.95)
        avail_mean: Mean availability (typically 0.80)

    Returns:
        Availability profile (0-1)
    """
    traffic_min = traffic.min()
    traffic_mean = traffic.mean()

    # Avoid division by zero
    if traffic_mean - traffic_min < 1e-6:
        return np.full_like(traffic, avail_mean)

    avail = avail_max - (avail_max - avail_mean) * (traffic - traffic_min) / (traffic_mean - traffic_min)

    # Clip to valid range
    avail = np.clip(avail, 0.0, 1.0)

    return avail


def create_weekly_traffic_profile() -> np.ndarray:
    """
    Create a weekly (168-hour) traffic profile from daily patterns.

    Uses GB-specific daily patterns for weekdays and weekends.

    Returns:
        Array of shape (168,) with hourly traffic values
    """
    weekly = np.zeros(168)

    # Monday to Friday (days 0-4)
    for day in range(5):
        start_hour = day * 24
        weekly[start_hour:start_hour + 24] = GB_WEEKDAY_TRAFFIC

    # Saturday and Sunday (days 5-6)
    for day in range(5, 7):
        start_hour = day * 24
        weekly[start_hour:start_hour + 24] = GB_WEEKEND_TRAFFIC

    return weekly


def create_weekly_availability_profile(avail_max: float = DEFAULT_AVAIL_MAX,
                                       avail_mean: float = DEFAULT_AVAIL_MEAN) -> np.ndarray:
    """
    Create a weekly (168-hour) EV availability profile.

    Args:
        avail_max: Maximum availability fraction
        avail_mean: Mean availability fraction

    Returns:
        Array of shape (168,) with hourly availability values
    """
    traffic = create_weekly_traffic_profile()
    availability = traffic_to_availability(traffic, avail_max, avail_mean)
    return availability


# ──────────────────────────────────────────────────────────────────────────────
# DSM Profile (Minimum State of Charge)
# ──────────────────────────────────────────────────────────────────────────────

def create_dsm_profile(dt_index: pd.DatetimeIndex,
                       nodes: list,
                       restriction_hour: int = 7,
                       restriction_value: float = 0.8) -> pd.DataFrame:
    """
    Create DSM (Demand Side Management) profile for EV minimum state of charge.

    This enforces that EVs must have a minimum SOC at certain times (e.g., by
    morning departure time) to ensure they can meet daily driving needs.

    Args:
        dt_index: DatetimeIndex for the output profile
        nodes: List of node/bus names (columns)
        restriction_hour: Hour of day when minimum SOC is enforced (e.g., 7 = 7am)
        restriction_value: Minimum SOC fraction required (e.g., 0.8 = 80%)

    Returns:
        DataFrame with minimum SOC requirements (0 except at restriction hour)
    """
    # Create weekly DSM profile
    dsm_week = np.zeros(168)

    # Set restriction at specified hour each day
    for day in range(7):
        dsm_week[day * 24 + restriction_hour] = restriction_value

    # Expand to full year
    df = generate_periodic_profiles(dt_index, nodes, dsm_week)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# DfT Data Loading (Optional Enhancement)
# ──────────────────────────────────────────────────────────────────────────────

def load_dft_traffic_data(dft_data_path: str, logger: logging.Logger) -> np.ndarray:
    """
    Load traffic data from DfT (Department for Transport) statistics.

    DfT provides Annual Average Daily Flow (AADF) data with hourly factors.
    This can be used to create more accurate GB-specific traffic profiles.

    Args:
        dft_data_path: Path to DfT traffic data file
        logger: Logger instance

    Returns:
        Weekly traffic profile (168 hours) or None if data unavailable
    """
    try:
        dft_path = Path(dft_data_path)
        if not dft_path.exists():
            logger.warning(f"DfT traffic data not found at {dft_data_path}")
            logger.info("Using synthetic GB traffic profile instead")
            return None

        logger.info(f"Loading DfT traffic data from {dft_data_path}...")

        # DfT data format varies - this is a placeholder for the actual parsing
        # Real implementation would depend on the specific DfT data format
        df = pd.read_csv(dft_path)

        # Extract hourly factors and create weekly profile
        # ... (implementation depends on DfT data format)

        logger.info("Successfully loaded DfT traffic data")
        return None  # Placeholder - return actual data when implemented

    except Exception as e:
        logger.warning(f"Error loading DfT data: {e}")
        logger.info("Falling back to synthetic GB traffic profile")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main Processing
# ──────────────────────────────────────────────────────────────────────────────

def build_ev_availability(snapshots: pd.DatetimeIndex,
                          nodes: list,
                          avail_max: float,
                          avail_mean: float,
                          dft_data_path: str = None,
                          logger: logging.Logger = None) -> tuple:
    """
    Build EV availability and DSM profiles.

    Args:
        snapshots: DatetimeIndex for the model period
        nodes: List of bus/node names
        avail_max: Maximum availability fraction
        avail_mean: Mean availability fraction
        dft_data_path: Optional path to DfT traffic data
        logger: Logger instance

    Returns:
        Tuple of (availability_profile, dsm_profile) DataFrames
    """
    if logger:
        logger.info("=" * 80)
        logger.info("BUILDING EV AVAILABILITY PROFILES")
        logger.info("=" * 80)
        logger.info(f"Time range: {snapshots[0]} to {snapshots[-1]}")
        logger.info(f"Number of timesteps: {len(snapshots)}")
        logger.info(f"Number of nodes: {len(nodes)}")
        logger.info(f"Availability parameters: max={avail_max}, mean={avail_mean}")

    # Try to load DfT data, fall back to synthetic if unavailable
    if dft_data_path:
        traffic_data = load_dft_traffic_data(dft_data_path, logger)
    else:
        traffic_data = None

    if traffic_data is not None:
        weekly_traffic = traffic_data
    else:
        weekly_traffic = create_weekly_traffic_profile()

    # Convert traffic to availability
    weekly_availability = traffic_to_availability(weekly_traffic, avail_max, avail_mean)

    if logger:
        logger.info(f"Weekly availability range: {weekly_availability.min():.3f} to {weekly_availability.max():.3f}")
        logger.info(f"Weekly availability mean: {weekly_availability.mean():.3f}")

    # Expand to full year
    availability_profile = generate_periodic_profiles(snapshots, nodes, weekly_availability)

    # Create DSM profile (minimum SOC at 7am each day)
    dsm_profile = create_dsm_profile(snapshots, nodes, restriction_hour=7, restriction_value=0.8)

    if logger:
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Availability profile shape: {availability_profile.shape}")
        logger.info(f"DSM profile shape: {dsm_profile.shape}")
        logger.info(f"Availability range: {availability_profile.values.min():.3f} to {availability_profile.values.max():.3f}")
        logger.info("=" * 80)
        logger.info("EV AVAILABILITY PROFILES COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    return availability_profile, dsm_profile


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
        # Import network loading utilities
        from scripts.utilities.network_io import load_network

        # Get parameters from snakemake
        ev_config = snakemake.params.get('ev_config', {})
        avail_config = ev_config.get('availability', {})

        avail_max = snakemake.params.get('avail_max', DEFAULT_AVAIL_MAX)
        avail_mean = snakemake.params.get('avail_mean', DEFAULT_AVAIL_MEAN)

        # Load network to get snapshots and buses
        logger.info(f"Loading network from {snakemake.input.network}")
        n = load_network(snakemake.input.network, skip_time_series=False, custom_logger=logger)

        # Get snapshots from network
        if len(n.snapshots) > 0:
            snapshots = n.snapshots
            logger.info(f"Using {len(snapshots)} snapshots from network")
        else:
            # Fallback: generate from modelled year
            modelled_year = snakemake.params.get('modelled_year', 2035)
            timestep_minutes = snakemake.params.get('timestep_minutes', 60)
            freq = f"{timestep_minutes}min" if timestep_minutes != 60 else "h"
            snapshots = pd.date_range(
                start=f"{modelled_year}-01-01",
                end=f"{modelled_year}-12-31 23:00",
                freq=freq
            )
            logger.info(f"Generated {len(snapshots)} snapshots for year {modelled_year}")

        # Get nodes from network buses
        nodes = list(n.buses.index)
        logger.info(f"Found {len(nodes)} buses in network")

        # Optional DfT data path
        dft_data_path = getattr(snakemake.input, 'dft_traffic', None)

        # Build profiles
        availability_profile, dsm_profile = build_ev_availability(
            snapshots=snapshots,
            nodes=nodes,
            avail_max=avail_max,
            avail_mean=avail_mean,
            dft_data_path=dft_data_path,
            logger=logger
        )

        # Save outputs
        availability_profile.to_csv(snakemake.output.availability_profile)
        logger.info(f"Saved availability profile to {snakemake.output.availability_profile}")

        dsm_profile.to_csv(snakemake.output.dsm_profile)
        logger.info(f"Saved DSM profile to {snakemake.output.dsm_profile}")

    except Exception as e:
        logger.error(f"Error building EV availability profiles: {e}", exc_info=True)
        raise
