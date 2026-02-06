"""
Build Heat Demand Profiles and Coefficient of Performance (COP) using Atlite

This script generates weather-dependent heat demand profiles and heat pump COPs
using Atlite cutouts with ERA5 temperature data.

Outputs:
- Heat demand profiles (degree-day based)
- Air Source Heat Pump (ASHP) COPs
- Ground Source Heat Pump (GSHP) COPs

The profiles are spatially resolved to network buses for use in demand-side
flexibility modeling.
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
import atlite
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Base temperature for heating degree days (°C)
HEATING_BASE_TEMP = 15.5

# Ground temperature parameters (for GSHP)
GROUND_TEMP_MEAN = 10.0  # Mean annual ground temperature (°C)
GROUND_TEMP_AMPLITUDE = 3.0  # Seasonal amplitude (°C)
GROUND_TEMP_PHASE_SHIFT = 30  # Days from Jan 1 to minimum (roughly early Feb)

# COP calculation parameters
# ASHP COP formula: COP = a - b * dT + c * dT^2 (based on EN 14825)
# Where dT = T_sink - T_source (typically ~35°C sink for underfloor heating)
ASHP_COP_A = 6.81
ASHP_COP_B = 0.121
ASHP_COP_C = 0.000630
ASHP_SINK_TEMP = 35.0  # Flow temperature for underfloor heating

# GSHP typically has higher COP due to more stable source temperature
GSHP_COP_A = 8.77
GSHP_COP_B = 0.150
GSHP_COP_C = 0.000734
GSHP_SINK_TEMP = 35.0

# Minimum and maximum COP bounds
COP_MIN = 1.5
COP_MAX = 6.0


# ──────────────────────────────────────────────────────────────────────────────
# Heat Demand Profile Generation
# ──────────────────────────────────────────────────────────────────────────────

def calculate_heating_degree_hours(temperature: xr.DataArray,
                                   base_temp: float = HEATING_BASE_TEMP) -> xr.DataArray:
    """
    Calculate heating degree hours from temperature data.

    Heating degree hours = max(0, base_temp - actual_temp)
    Higher values indicate more heating required.

    Args:
        temperature: xarray DataArray with temperature in Kelvin or Celsius
        base_temp: Base temperature for heating (°C)

    Returns:
        xarray DataArray with heating degree hours
    """
    # Convert Kelvin to Celsius if necessary
    if temperature.mean() > 200:  # Likely Kelvin
        temp_celsius = temperature - 273.15
    else:
        temp_celsius = temperature

    # Calculate heating degree hours (only positive values)
    hdh = xr.where(temp_celsius < base_temp, base_temp - temp_celsius, 0)

    return hdh


def generate_heat_demand_profile(cutout: atlite.Cutout,
                                 logger: logging.Logger) -> xr.DataArray:
    """
    Generate normalized heat demand profile from cutout temperature data.

    Uses heating degree hours method - heat demand is proportional to the
    difference between outdoor temperature and base temperature when below base.

    Args:
        cutout: Atlite cutout with temperature data
        logger: Logger instance

    Returns:
        Normalized heat demand profile (sums to 1.0 over time for each location)
    """
    logger.info("Extracting temperature data from cutout...")

    # Get temperature data from cutout
    # Atlite stores temperature as 2m air temperature
    temp = cutout.data['temperature']

    logger.info(f"Temperature data shape: {temp.shape}")
    logger.info(f"Temperature range: {float(temp.min()):.1f} to {float(temp.max()):.1f}")

    # Calculate heating degree hours
    logger.info(f"Calculating heating degree hours (base temp: {HEATING_BASE_TEMP}°C)...")
    hdh = calculate_heating_degree_hours(temp, HEATING_BASE_TEMP)

    # Normalize to get demand profile (fraction of annual demand per timestep)
    # Sum over time dimension for each spatial point
    hdh_total = hdh.sum(dim='time')

    # Avoid division by zero for locations with no heating demand
    hdh_total = xr.where(hdh_total > 0, hdh_total, 1.0)

    # Normalized profile
    heat_demand_profile = hdh / hdh_total

    logger.info(f"Heat demand profile generated, shape: {heat_demand_profile.shape}")

    return heat_demand_profile


# ──────────────────────────────────────────────────────────────────────────────
# COP Calculations
# ──────────────────────────────────────────────────────────────────────────────

def calculate_ashp_cop(temperature: xr.DataArray,
                       sink_temp: float = ASHP_SINK_TEMP,
                       logger: logging.Logger = None) -> xr.DataArray:
    """
    Calculate Air Source Heat Pump COP based on outdoor temperature.

    Uses a quadratic formula based on EN 14825 standard:
    COP = a - b * dT + c * dT^2
    where dT = sink_temp - source_temp (outdoor air)

    Args:
        temperature: Outdoor air temperature (K or °C)
        sink_temp: Heat distribution temperature (°C), typically 35-55°C
        logger: Logger instance

    Returns:
        COP values as xarray DataArray
    """
    if logger:
        logger.info("Calculating ASHP COPs...")

    # Convert Kelvin to Celsius if necessary
    if temperature.mean() > 200:
        temp_celsius = temperature - 273.15
    else:
        temp_celsius = temperature

    # Temperature difference (lift)
    dT = sink_temp - temp_celsius

    # COP calculation using quadratic formula
    cop = ASHP_COP_A - ASHP_COP_B * dT + ASHP_COP_C * dT**2

    # Apply bounds
    cop = xr.where(cop < COP_MIN, COP_MIN, cop)
    cop = xr.where(cop > COP_MAX, COP_MAX, cop)

    if logger:
        logger.info(f"ASHP COP range: {float(cop.min()):.2f} to {float(cop.max()):.2f}")
        logger.info(f"ASHP COP mean: {float(cop.mean()):.2f}")

    return cop


def calculate_ground_temperature(time_index: pd.DatetimeIndex,
                                 mean_temp: float = GROUND_TEMP_MEAN,
                                 amplitude: float = GROUND_TEMP_AMPLITUDE,
                                 phase_shift: int = GROUND_TEMP_PHASE_SHIFT) -> pd.Series:
    """
    Calculate ground temperature using a sinusoidal annual cycle.

    Ground temperature at typical borehole depth (50-100m) varies sinusoidally
    with a lag of about 1-2 months from air temperature.

    Args:
        time_index: Pandas DatetimeIndex
        mean_temp: Mean annual ground temperature (°C)
        amplitude: Seasonal temperature amplitude (°C)
        phase_shift: Days from Jan 1 to temperature minimum

    Returns:
        Ground temperature series (°C)
    """
    # Day of year
    day_of_year = time_index.dayofyear

    # Sinusoidal variation (minimum in early Feb, maximum in early Aug)
    ground_temp = mean_temp - amplitude * np.cos(
        2 * np.pi * (day_of_year - phase_shift) / 365
    )

    return pd.Series(ground_temp, index=time_index)


def calculate_gshp_cop(time_index: pd.DatetimeIndex,
                       sink_temp: float = GSHP_SINK_TEMP,
                       spatial_shape: tuple = None,
                       logger: logging.Logger = None) -> xr.DataArray:
    """
    Calculate Ground Source Heat Pump COP based on ground temperature.

    GSHP has more stable COP due to relatively constant ground temperature.
    Uses same quadratic formula but with ground temperature as source.

    Args:
        time_index: Time index for the profile
        sink_temp: Heat distribution temperature (°C)
        spatial_shape: Shape of spatial dimensions (y, x) for broadcasting
        logger: Logger instance

    Returns:
        COP values as xarray DataArray
    """
    if logger:
        logger.info("Calculating GSHP COPs...")

    # Calculate ground temperature
    ground_temp = calculate_ground_temperature(time_index)

    # Temperature difference (lift)
    dT = sink_temp - ground_temp

    # COP calculation
    cop_values = GSHP_COP_A - GSHP_COP_B * dT + GSHP_COP_C * dT**2

    # Apply bounds
    cop_values = np.clip(cop_values, COP_MIN, COP_MAX)

    if logger:
        logger.info(f"GSHP COP range: {cop_values.min():.2f} to {cop_values.max():.2f}")
        logger.info(f"GSHP COP mean: {cop_values.mean():.2f}")

    # Create xarray DataArray
    # GSHP COP is spatially uniform (same ground temp assumption)
    # but we create it with the same shape as ASHP for consistency
    if spatial_shape:
        # Broadcast to spatial dimensions
        cop_array = np.broadcast_to(
            cop_values.values[:, np.newaxis, np.newaxis],
            (len(time_index), *spatial_shape)
        )
    else:
        cop_array = cop_values.values

    return cop_array


# ──────────────────────────────────────────────────────────────────────────────
# Bus Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_to_buses(data: xr.DataArray,
                       bus_regions: xr.DataArray,
                       logger: logging.Logger) -> pd.DataFrame:
    """
    Aggregate gridded data to network bus regions.

    Args:
        data: Gridded data (time, y, x)
        bus_regions: Bus assignment for each grid cell
        logger: Logger instance

    Returns:
        DataFrame with columns for each bus, indexed by time
    """
    logger.info("Aggregating gridded data to bus regions...")

    # Get unique buses
    buses = np.unique(bus_regions.values)
    buses = buses[~np.isnan(buses)]  # Remove NaN

    # Aggregate by bus region (weighted average)
    result = {}
    for bus in buses:
        mask = bus_regions == bus
        bus_data = data.where(mask).mean(dim=['x', 'y'])
        result[str(int(bus))] = bus_data.values

    df = pd.DataFrame(result, index=data.time.values)

    logger.info(f"Aggregated to {len(df.columns)} buses")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main Processing
# ──────────────────────────────────────────────────────────────────────────────

def process_heat_profiles(cutout_path: str,
                          output_heat_demand: str,
                          output_cop_ashp: str,
                          output_cop_gshp: str,
                          logger: logging.Logger):
    """
    Main processing function to generate heat profiles and COPs.

    Args:
        cutout_path: Path to Atlite cutout file
        output_heat_demand: Output path for heat demand profile
        output_cop_ashp: Output path for ASHP COP profile
        output_cop_gshp: Output path for GSHP COP profile
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("BUILDING HEAT PROFILES AND COPs")
    logger.info("=" * 80)

    # Load cutout
    logger.info(f"Loading cutout from {cutout_path}...")
    cutout = atlite.Cutout(path=cutout_path)
    logger.info(f"Cutout time range: {cutout.data.time.values[0]} to {cutout.data.time.values[-1]}")
    logger.info(f"Cutout spatial extent: {cutout.extent}")

    # Generate heat demand profile
    heat_demand = generate_heat_demand_profile(cutout, logger)

    # Calculate ASHP COPs
    temperature = cutout.data['temperature']
    cop_ashp = calculate_ashp_cop(temperature, logger=logger)

    # Calculate GSHP COPs
    time_index = pd.DatetimeIndex(cutout.data.time.values)
    spatial_shape = (len(cutout.data.y), len(cutout.data.x))
    cop_gshp_array = calculate_gshp_cop(time_index, spatial_shape=spatial_shape, logger=logger)

    # Create GSHP xarray with same coordinates as ASHP
    cop_gshp = xr.DataArray(
        cop_gshp_array,
        dims=['time', 'y', 'x'],
        coords={
            'time': cutout.data.time,
            'y': cutout.data.y,
            'x': cutout.data.x
        }
    )

    # Save outputs as NetCDF
    # IMPORTANT: Compute lazy Dask arrays before saving to avoid slow writes
    logger.info("Computing lazy arrays before saving...")
    
    # Compute all arrays to load into memory (avoids slow on-the-fly computation during write)
    if hasattr(heat_demand, 'compute'):
        heat_demand = heat_demand.compute()
        logger.info("Heat demand computed")
    if hasattr(cop_ashp, 'compute'):
        cop_ashp = cop_ashp.compute()
        logger.info("ASHP COP computed")
    
    logger.info("Saving outputs...")

    # Heat demand
    heat_demand_ds = xr.Dataset({'heat_demand': heat_demand})
    heat_demand_ds.attrs['description'] = 'Normalized heat demand profile based on heating degree hours'
    heat_demand_ds.attrs['base_temperature'] = HEATING_BASE_TEMP
    heat_demand_ds.to_netcdf(output_heat_demand, engine='netcdf4')
    logger.info(f"Saved heat demand profile to {output_heat_demand}")

    # ASHP COP
    cop_ashp_ds = xr.Dataset({'cop': cop_ashp})
    cop_ashp_ds.attrs['description'] = 'Air Source Heat Pump Coefficient of Performance'
    cop_ashp_ds.attrs['sink_temperature'] = ASHP_SINK_TEMP
    cop_ashp_ds.attrs['formula'] = f'COP = {ASHP_COP_A} - {ASHP_COP_B}*dT + {ASHP_COP_C}*dT^2'
    cop_ashp_ds.to_netcdf(output_cop_ashp, engine='netcdf4')
    logger.info(f"Saved ASHP COP to {output_cop_ashp}")

    # GSHP COP
    cop_gshp_ds = xr.Dataset({'cop': cop_gshp})
    cop_gshp_ds.attrs['description'] = 'Ground Source Heat Pump Coefficient of Performance'
    cop_gshp_ds.attrs['sink_temperature'] = GSHP_SINK_TEMP
    cop_gshp_ds.attrs['ground_temp_mean'] = GROUND_TEMP_MEAN
    cop_gshp_ds.attrs['ground_temp_amplitude'] = GROUND_TEMP_AMPLITUDE
    cop_gshp_ds.to_netcdf(output_cop_gshp, engine='netcdf4')
    logger.info(f"Saved GSHP COP to {output_cop_gshp}")

    # Summary statistics
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Time steps: {len(heat_demand.time)}")
    logger.info(f"Spatial points: {len(heat_demand.y)} x {len(heat_demand.x)}")
    logger.info(f"Heat demand peak fraction: {float(heat_demand.max()):.6f}")
    logger.info(f"ASHP COP: {float(cop_ashp.min()):.2f} - {float(cop_ashp.max()):.2f} (mean: {float(cop_ashp.mean()):.2f})")
    logger.info(f"GSHP COP: {float(cop_gshp.min()):.2f} - {float(cop_gshp.max()):.2f} (mean: {float(cop_gshp.mean()):.2f})")
    logger.info("=" * 80)
    logger.info("HEAT PROFILES GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


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
        process_heat_profiles(
            cutout_path=snakemake.input.cutout,
            output_heat_demand=snakemake.output.heat_demand,
            output_cop_ashp=snakemake.output.cop_ashp,
            output_cop_gshp=snakemake.output.cop_gshp,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Error in heat profile generation: {e}", exc_info=True)
        raise
