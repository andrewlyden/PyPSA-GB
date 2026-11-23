#!/usr/bin/env python3
"""
Generate Interconnector Availability Profiles
==============================================

This script generates availability profiles for interconnectors. Currently
creates constant availability (p_max_pu=1.0) but provides framework for
more sophisticated availability modeling including maintenance schedules,
outages, and dynamic availability factors.

Key features:
- Constant availability profile generation
- Configurable reference year and time resolution
- Extensible framework for complex availability modeling
- Integration with demand year configuration
- CSV output format compatible with PyPSA

Author: PyPSA-GB Team
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import logging
import time

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.utilities.logging_config import setup_logging, log_execution_summary
except ImportError:
    import logging
    def setup_logging(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

# Check if running in Snakemake context
if 'snakemake' in globals():
    SNAKEMAKE_MODE = True
    input_clean = snakemake.input[0]
    output_availability = snakemake.output[0]
    # Try to get demand_year from params, default to 2020 if not provided
    demand_year = getattr(snakemake.params, 'demand_year', 2020) if hasattr(snakemake, 'params') else 2020
    availability_config = (
        getattr(snakemake.params, 'availability_config', {})
        if hasattr(snakemake, 'params')
        else {}
    )
    # CP30 cable composition (15 Oct 2026): cables it adds get profiles too.
    input_composition = snakemake.input.get('cp30_composition', None)
else:
    SNAKEMAKE_MODE = False
    availability_config = {}
    input_composition = None

def create_time_index(year: int, freq: str = 'h') -> pd.DatetimeIndex:
    """
    Create a time index for the given year at the specified frequency.
    
    Args:
        year: Reference year for time series
        freq: Frequency string (h for hourly, 30min for 30-minute)
        
    Returns:
        DatetimeIndex covering the full year
    """
    # Normalize deprecated 'H' to 'h' for pandas compatibility
    if freq == 'H':
        freq = 'h'
    
    start_date = f"{year}-01-01"
    end_date = f"{year+1}-01-01"
    
    time_index = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive='left')
    return time_index

def generate_constant_availability(interconnector_names: List[str], 
                                   time_index: pd.DatetimeIndex,
                                   availability_factor: float = 1.0) -> pd.DataFrame:
    """
    Generate constant availability profiles for interconnectors.
    
    Args:
        interconnector_names: List of interconnector names
        time_index: Time index for the profiles
        availability_factor: Constant availability factor (0-1)
        
    Returns:
        DataFrame with time, name, and p_max_pu columns
    """
    logger = logging.getLogger(__name__)
    
    # Handle empty list case
    if not interconnector_names:
        return pd.DataFrame(columns=['time', 'name', 'p_max_pu'])
    
    profiles = []
    
    for name in interconnector_names:
        # Create constant profile for this interconnector
        profile_data = {
            'time': time_index,
            'name': name,
            'p_max_pu': availability_factor
        }
        
        interconnector_profile = pd.DataFrame(profile_data)
        profiles.append(interconnector_profile)
    
    # Combine all profiles
    combined_profiles = pd.concat(profiles, ignore_index=True)
    
    logger.info(f"Generated constant availability profiles for {len(interconnector_names)} interconnectors")
    logger.info(f"Time series length: {len(time_index)} time steps")
    logger.info(f"Availability factor: {availability_factor}")
    
    return combined_profiles

def generate_realistic_availability(interconnector_names: List[str],
                                    time_index: pd.DatetimeIndex,
                                    base_availability: float = 0.95,
                                    maintenance_probability: float = 0.02) -> pd.DataFrame:
    """
    Generate more realistic availability profiles with maintenance periods.
    
    Args:
        interconnector_names: List of interconnector names
        time_index: Time index for the profiles
        base_availability: Base availability when operational
        maintenance_probability: Probability of maintenance in any given hour
        
    Returns:
        DataFrame with time, name, and p_max_pu columns
    """
    logger = logging.getLogger(__name__)
    
    profiles = []
    np.random.seed(42)  # For reproducible results
    
    for name in interconnector_names:
        # Generate random maintenance periods
        maintenance_mask = np.random.random(len(time_index)) < maintenance_probability
        
        # Create availability profile
        availability = np.full(len(time_index), base_availability)
        availability[maintenance_mask] = 0.0  # Zero availability during maintenance
        
        # Smooth maintenance transitions (gradual ramp down/up)
        for i in range(1, len(availability)):
            if availability[i-1] > 0 and availability[i] == 0:
                # Start of maintenance - ramp down
                if i > 0:
                    availability[i-1] = base_availability * 0.5
            elif availability[i-1] == 0 and availability[i] > 0:
                # End of maintenance - ramp up
                availability[i] = base_availability * 0.5
        
        profile_data = {
            'time': time_index,
            'name': name,
            'p_max_pu': availability
        }
        
        interconnector_profile = pd.DataFrame(profile_data)
        profiles.append(interconnector_profile)
    
    combined_profiles = pd.concat(profiles, ignore_index=True)
    
    logger.info(f"Generated realistic availability profiles for {len(interconnector_names)} interconnectors")
    logger.info(f"Base availability: {base_availability}")
    logger.info(f"Maintenance probability: {maintenance_probability}")
    
    return combined_profiles

def append_composition_cables(interconnector_names: List[str], composition_file, logger=None) -> List[str]:
    '''Cables a CP30 composition adds, appended after the register cables (15 Oct 2026).

    Outage draws are seeded by a cable's position in the list, so appending keeps
    every existing cable's profile unchanged; the added cable gets the next seed.
    '''
    from scripts.CP30.storage_interconnectors import load_cp30_interconnector_composition
    if isinstance(composition_file, (list, tuple)):
        composition_file = composition_file[0] if composition_file else None
    if not composition_file:
        return list(interconnector_names)
    composition = load_cp30_interconnector_composition(composition_file)
    added = [n for n in composition.loc[composition['action'].eq('add'), 'name'] if n not in interconnector_names]
    if added and logger is not None:
        logger.info(f"CP30 composition cables added to the availability profiles: {added}")
    return list(interconnector_names) + added


def generate_outage_availability(
    interconnector_names,
    time_index,
    planned_outages_per_year=1,
    planned_outage_duration_hours=168,
    planned_outage_months=(4, 5, 6, 7, 8, 9),
    forced_outage_rate=0.03,
    forced_outage_mean_duration_hours=24,
    seed=42,
):
    '''Create reproducible contiguous planned and forced cable outages.'''
    profiles = []
    n_hours = len(time_index)
    permitted = np.flatnonzero(time_index.month.isin(planned_outage_months))
    for cable_number, name in enumerate(interconnector_names):
        rng = np.random.default_rng(int(seed) + cable_number)
        availability = np.ones(n_hours, dtype=float)

        duration = max(1, int(planned_outage_duration_hours))
        for _ in range(max(0, int(planned_outages_per_year))):
            if len(permitted) == 0:
                break
            start = int(rng.choice(permitted))
            availability[start:min(start + duration, n_hours)] = 0.0

        mean_duration = max(float(forced_outage_mean_duration_hours), 1.0)
        expected_events = n_hours * max(float(forced_outage_rate), 0.0) / mean_duration
        for _ in range(int(rng.poisson(expected_events))):
            start = int(rng.integers(0, n_hours))
            event_duration = max(1, int(rng.exponential(mean_duration)))
            availability[start:min(start + event_duration, n_hours)] = 0.0

        profiles.append(pd.DataFrame({
            'time': time_index,
            'name': name,
            'p_max_pu': availability,
        }))
    return (
        pd.concat(profiles, ignore_index=True)
        if profiles
        else pd.DataFrame(columns=['time', 'name', 'p_max_pu'])
    )


def apply_seasonal_variations(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply seasonal variations to availability profiles.
    
    Args:
        profiles_df: Base availability profiles
        
    Returns:
        Profiles with seasonal variations applied
    """
    logger = logging.getLogger(__name__)
    
    # Create a copy to avoid modifying the original
    profiles_df = profiles_df.copy()
    
    # Extract month from time index
    profiles_df['month'] = pd.to_datetime(profiles_df['time']).dt.month
    
    # Define seasonal factors (winter months have slightly lower availability)
    seasonal_factors = {
        1: 0.95,   # January
        2: 0.95,   # February
        3: 0.98,   # March
        4: 1.00,   # April
        5: 1.00,   # May
        6: 1.00,   # June
        7: 1.00,   # July
        8: 1.00,   # August
        9: 1.00,   # September
        10: 0.98,  # October
        11: 0.95,  # November
        12: 0.95   # December
    }
    
    # Apply seasonal factors
    for month, factor in seasonal_factors.items():
        month_mask = profiles_df['month'] == month
        profiles_df.loc[month_mask, 'p_max_pu'] *= factor
    
    # Remove temporary month column
    profiles_df = profiles_df.drop('month', axis=1)
    
    logger.info("Applied seasonal availability variations")
    return profiles_df

def main():
    """Main execution function."""
    logger = setup_logging("generate_interconnector_availability")
    start_time = time.time()
    
    try:
        logger.info("Starting interconnector availability profile generation...")
        logger.info(f"Running in {'Snakemake' if SNAKEMAKE_MODE else 'standalone'} mode")
        
        if SNAKEMAKE_MODE:
            clean_file = input_clean
            output_file = output_availability
            reference_year = demand_year
        else:
            clean_file = "resources/interconnectors/interconnectors_clean.csv"
            output_file = "resources/interconnectors/interconnector_availability.csv"
            reference_year = 2020
        
        logger.info(f"Clean data file: {clean_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Reference year: {reference_year}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load clean interconnector data
        if not Path(clean_file).exists():
            logger.warning(f"Clean data file not found: {clean_file}")
            # Create empty availability file
            empty_df = pd.DataFrame(columns=['time', 'name', 'p_max_pu'])
            empty_df.to_csv(output_file, index=False)
            logger.info("Created empty availability file")
            return
        
        interconnectors_df = pd.read_csv(clean_file)
        logger.info(f"Loaded {len(interconnectors_df)} interconnector records")
        
        if len(interconnectors_df) == 0:
            logger.warning("No interconnectors found - creating empty availability file")
            empty_df = pd.DataFrame(columns=['time', 'name', 'p_max_pu'])
            empty_df.to_csv(output_file, index=False)
            return
        
        # Extract interconnector names
        interconnector_names = interconnectors_df['name'].unique().tolist()
        if input_composition:
            interconnector_names = append_composition_cables(interconnector_names, input_composition, logger)
        logger.info(f"Creating availability profiles for {len(interconnector_names)} interconnectors")
        
        # Create time index for the reference year
        time_index = create_time_index(reference_year)
        logger.info(f"Created time index: {len(time_index)} hourly time steps for {reference_year}")
        
        mode = str(availability_config.get('mode', 'outage')).lower()
        if mode == 'constant':
            availability_profiles = generate_constant_availability(
                interconnector_names,
                time_index,
                availability_factor=float(
                    availability_config.get('availability_factor', 1.0)
                ),
            )
        elif mode == 'outage':
            availability_profiles = generate_outage_availability(
                interconnector_names,
                time_index,
                planned_outages_per_year=availability_config.get(
                    'planned_outages_per_year', 1
                ),
                planned_outage_duration_hours=availability_config.get(
                    'planned_outage_duration_hours', 168
                ),
                planned_outage_months=tuple(
                    availability_config.get(
                        'planned_outage_months', [4, 5, 6, 7, 8, 9]
                    )
                ),
                forced_outage_rate=availability_config.get(
                    'forced_outage_rate', 0.03
                ),
                forced_outage_mean_duration_hours=availability_config.get(
                    'forced_outage_mean_duration_hours', 24
                ),
                seed=availability_config.get('seed', 42),
            )
        else:
            raise ValueError(f'Unsupported interconnector availability mode: {mode}')
        
        # Optional: Apply seasonal variations (uncomment to enable)
        # availability_profiles = apply_seasonal_variations(availability_profiles)
        
        # Ensure p_max_pu is within valid range
        availability_profiles['p_max_pu'] = availability_profiles['p_max_pu'].clip(0.0, 1.0)
        
        # Save availability profiles
        availability_profiles.to_csv(output_file, index=False)
        logger.info(f"Saved availability profiles to: {output_file}")
        
        # Calculate statistics
        interconnectors = availability_profiles['name'].nunique()
        time_periods = len(time_index)
        avg_availability = availability_profiles['p_max_pu'].mean()
        
        # Log execution summary
        log_execution_summary(
            logger,
            "generate_interconnector_availability",
            start_time,
            inputs={'clean_interconnectors': clean_file},
            outputs={'availability_profiles': output_file},
            context={
                'interconnectors': interconnectors,
                'time_periods': time_periods,
                'reference_year': reference_year,
                'avg_availability': f"{avg_availability:.3f}",
                'total_records': len(availability_profiles)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in availability profile generation: {e}")
        if SNAKEMAKE_MODE:
            raise
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()

