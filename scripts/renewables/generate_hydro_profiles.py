"""
Generate hydro renewable profiles with appropriate operational characteristics.

Creates profiles for:
- Large Hydro: Dispatchable operation (constant availability but flexible dispatch)
- Small Hydro: Run-of-river operation (seasonal river flow patterns)

Large hydro facilities typically have reservoirs allowing flexible dispatch,
while small hydro follows natural river flow patterns.
"""

import logging
import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Centralized logging
try:
    from logging_config import setup_logging, log_execution_summary
    logger = setup_logging("generate_hydro_profiles")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_execution_summary
        logger = setup_logging("generate_hydro_profiles")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("generate_hydro_profiles")

warnings.simplefilter('ignore')
logging.captureWarnings(False)


class HydroProfileGenerator:
    def __init__(self):
        # UK hydro characteristics
        self.large_hydro_config = {
            'base_availability': 0.95,  # High availability for dispatchable operation
            'maintenance_probability': 0.05,  # 5% chance of maintenance outage per month
            'seasonal_variation': 0.1,  # Small seasonal variation in availability
        }
        
        self.small_hydro_config = {
            'base_capacity_factor': 0.45,  # Typical run-of-river CF
            'seasonal_amplitude': 0.3,  # Strong seasonal variation
            'winter_peak': True,  # UK rivers peak in winter
            'drought_probability': 0.02,  # 2% chance of low flow periods
            'flood_probability': 0.01,  # 1% chance of high flow periods
        }

    def load_site_data(self, path):
        """Load hydro site data from CSV file."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Missing site file: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(p)
            logger.info(f"Loaded {len(df)} hydro sites from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return pd.DataFrame()

    def create_time_index(self, year, freq='h'):
        """Create hourly time index for the specified year."""
        start = f"{year}-01-01"
        end = f"{year+1}-01-01"
        return pd.date_range(start=start, end=end, freq=freq, inclusive='left')

    def generate_large_hydro_profile(self, time_index, site_name, capacity_mw):
        """
        Generate large hydro profile for dispatchable operation.
        
        Large hydro facilities with reservoirs can dispatch flexibly but have
        availability constraints due to maintenance and water levels.
        
        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Time index for the profile
        site_name : str
            Name of the hydro facility
        capacity_mw : float
            Installed capacity in MW
            
        Returns
        -------
        pd.Series
            Hourly maximum available capacity (p_max_pu * capacity)
        """
        n_hours = len(time_index)
        
        # Base availability (high for large hydro with reservoirs)
        base_availability = self.large_hydro_config['base_availability']
        
        # Seasonal variation in water levels
        day_of_year = time_index.dayofyear
        seasonal_factor = 1 + self.large_hydro_config['seasonal_variation'] * np.sin(
            2 * np.pi * (day_of_year - 90) / 365.25  # Peak in winter/spring
        )
        
        # Planned maintenance windows (typically in summer low-demand periods)
        maintenance_prob = self.large_hydro_config['maintenance_probability']
        maintenance_mask = np.random.random(n_hours) < (maintenance_prob / (30 * 24))  # Monthly prob to hourly
        
        # Apply summer bias for maintenance (months 6-8)
        summer_mask = (time_index.month >= 6) & (time_index.month <= 8)
        maintenance_mask = maintenance_mask & summer_mask
        
        # Calculate availability
        availability = base_availability * seasonal_factor
        availability = np.where(maintenance_mask, 0.0, availability)  # Zero availability during maintenance
        
        # Ensure availability doesn't exceed 100%
        availability = np.clip(availability, 0, 1)
        
        # Convert to power availability (MW)
        max_power = availability * capacity_mw
        
        logger.info(f"Large hydro {site_name}: {capacity_mw}MW, avg availability {availability.mean():.2f}")
        
        return pd.Series(max_power, index=time_index)

    def generate_small_hydro_profile(self, time_index, site_name, capacity_mw, site_lat=56.0):
        """
        Generate small hydro profile for run-of-river operation.
        
        Small hydro follows natural river flow patterns with seasonal variation
        and weather-dependent flow changes.
        
        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Time index for the profile
        site_name : str
            Name of the hydro facility
        capacity_mw : float
            Installed capacity in MW
        site_lat : float
            Site latitude for regional flow patterns
            
        Returns
        -------
        pd.Series
            Hourly power generation (capacity_factor * capacity)
        """
        n_hours = len(time_index)
        day_of_year = time_index.dayofyear
        
        # Base capacity factor for run-of-river
        base_cf = self.small_hydro_config['base_capacity_factor']
        
        # Seasonal river flow pattern (UK winter peak)
        seasonal_amplitude = self.small_hydro_config['seasonal_amplitude']
        if self.small_hydro_config['winter_peak']:
            # Peak in winter months (Dec-Feb), low in summer (Jul-Sep)
            seasonal_cf = base_cf + seasonal_amplitude * np.cos(
                2 * np.pi * (day_of_year - 365/4) / 365.25  # Phase shift for winter peak
            )
        else:
            seasonal_cf = base_cf * np.ones(n_hours)
        
        # Add stochastic flow variation (weather-driven)
        # Use autocorrelated random walk to simulate persistent wet/dry periods
        flow_variation = np.random.normal(0, 0.1, n_hours)
        alpha = 0.98  # High autocorrelation for persistent weather patterns
        
        correlated_flow = np.zeros(n_hours)
        correlated_flow[0] = flow_variation[0]
        for i in range(1, n_hours):
            correlated_flow[i] = alpha * correlated_flow[i-1] + (1-alpha) * flow_variation[i]
        
        # Apply flow variation to seasonal pattern
        capacity_factor = seasonal_cf * (1 + 0.3 * correlated_flow)  # 30% flow variability
        
        # Add extreme events
        # Drought periods (low generation)
        drought_prob = self.small_hydro_config['drought_probability']
        drought_mask = np.random.random(n_hours) < (drought_prob / 24)  # Daily prob to hourly
        capacity_factor = np.where(drought_mask, capacity_factor * 0.2, capacity_factor)  # Reduce to 20% during drought
        
        # Flood periods (high generation, but may require shutdown for safety)
        flood_prob = self.small_hydro_config['flood_probability']
        flood_mask = np.random.random(n_hours) < (flood_prob / 24)  # Daily prob to hourly
        # During floods, either high generation or forced shutdown
        if np.sum(flood_mask) > 0:  # Only apply if there are flood events
            flood_indices = np.where(flood_mask)[0]
            flood_generation = np.where(np.random.random(len(flood_indices)) > 0.3, 1.0, 0.0)
            for i, idx in enumerate(flood_indices):
                capacity_factor[idx] = flood_generation[i]
        
        # Apply realistic constraints
        capacity_factor = np.clip(capacity_factor, 0, 1)
        
        # Convert to power generation (MW)
        power_generation = capacity_factor * capacity_mw
        
        logger.info(f"Small hydro {site_name}: {capacity_mw}MW, avg CF {capacity_factor.mean():.2f}")
        
        return pd.Series(power_generation, index=time_index)

    def generate_hydro_profiles(self, sites_df, hydro_type, time_index):
        """
        Generate hydro profiles for all sites of a given type.
        
        Parameters
        ----------
        sites_df : pd.DataFrame
            Site data with columns: site_name, capacity_mw, lat, lon
        hydro_type : str
            Hydro type ('large_hydro' or 'small_hydro')
        time_index : pd.DatetimeIndex
            Time index for profiles
            
        Returns
        -------
        pd.DataFrame
            DataFrame with site profiles
        """
        if sites_df.empty:
            logger.warning(f"No sites found for {hydro_type}")
            return pd.DataFrame(index=time_index)
        
        logger.info(f"Generating {hydro_type} profiles for {len(sites_df)} sites")
        
        profiles = {}
        total_capacity = 0
        
        for idx, row in sites_df.iterrows():
            site_name = row.get('site_name', f"site_{idx}")
            capacity = float(row.get('capacity_mw', 0.0) or 0.0)
            lat = float(row.get('lat', 56.0))  # Default to Scottish Highlands
            
            if capacity <= 0:
                logger.warning(f"Site {site_name} has zero capacity, skipping")
                continue
            
            total_capacity += capacity
            
            # Generate type-specific profile
            if hydro_type == 'large_hydro':
                profile = self.generate_large_hydro_profile(time_index, site_name, capacity)
            elif hydro_type == 'small_hydro':
                profile = self.generate_small_hydro_profile(time_index, site_name, capacity, lat)
            else:
                logger.error(f"Unknown hydro type: {hydro_type}")
                continue
            
            profiles[site_name] = profile
        
        # Create DataFrame with all site profiles
        profiles_df = pd.DataFrame(profiles, index=time_index)
        
        logger.info(f"Generated {hydro_type} profiles: {len(profiles_df.columns)} sites, "
                   f"{total_capacity:.1f} MW total capacity, {len(profiles_df)} hours")
        
        return profiles_df

    def save_profiles(self, profiles_df, output_file):
        """Save profiles to CSV file."""
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            profiles_df.to_csv(output_file)
            logger.info(f"Saved hydro profiles to {output_file}")


def main():
    """Main function called by Snakemake."""
    logger.info("Starting hydro renewable profile generation")
    start_time = time.time()
    
    # Track processing metrics
    large_hydro_sites_count = 0
    small_hydro_sites_count = 0
    total_capacity_mw = 0
    profiles_created = 0
    
    try:
        generator = HydroProfileGenerator()
        years = snakemake.params.renewables_year
        
        for i, year in enumerate(years):
            logger.info(f"Processing year {year}")
            
            # Create time index for the year
            time_index = generator.create_time_index(year)
            
            # Large hydro (dispatchable)
            if hasattr(snakemake.output, 'large_hydro_profiles'):
                large_hydro_sites = generator.load_site_data(snakemake.input.large_hydro)
                large_hydro_profiles = generator.generate_hydro_profiles(
                    large_hydro_sites, 'large_hydro', time_index
                )
                generator.save_profiles(large_hydro_profiles, snakemake.output.large_hydro_profiles[i])
                large_hydro_sites_count += len(large_hydro_sites)
                total_capacity_mw += large_hydro_sites['capacity_mw'].sum()
                profiles_created += 1
            
            # Small hydro (run-of-river)
            if hasattr(snakemake.output, 'small_hydro_profiles'):
                small_hydro_sites = generator.load_site_data(snakemake.input.small_hydro)
                small_hydro_profiles = generator.generate_hydro_profiles(
                    small_hydro_sites, 'small_hydro', time_index
                )
                generator.save_profiles(small_hydro_profiles, snakemake.output.small_hydro_profiles[i])
                small_hydro_sites_count += len(small_hydro_sites)
                total_capacity_mw += small_hydro_sites['capacity_mw'].sum()
                profiles_created += 1
        
        logger.info("Hydro renewable profile generation completed successfully")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'years_processed': len(years),
            'large_hydro_sites': large_hydro_sites_count,
            'small_hydro_sites': small_hydro_sites_count,
            'total_capacity_mw': round(total_capacity_mw, 2),
            'profiles_created': profiles_created
        }
        log_execution_summary(logger, "generate_hydro_profiles", execution_time, summary_stats)
        
    except Exception as e:
        logger.error(f"Error in hydro renewable profile generation: {e}")
        raise


if __name__ == '__main__':
    main()

