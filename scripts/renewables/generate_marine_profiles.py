"""
Generate synthetic timeseries for marine renewable technologies.

Creates predictable cyclic timeseries for:
- Tidal stream: Based on tidal cycle harmonics
- Shoreline wave: Based on wave height patterns and seasonal variation  
- Tidal lagoon: Based on tidal range and generation cycles

These technologies have predictable patterns based on oceanic cycles rather than weather.
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
    logger = setup_logging("generate_marine_profiles")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_execution_summary
        logger = setup_logging("generate_marine_profiles")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("generate_marine_profiles")

warnings.simplefilter('ignore')
logging.captureWarnings(False)


class MarineRenewableGenerator:
    def __init__(self):
        # Tidal harmonic constituents (simplified model)
        self.tidal_periods = {
            'M2': 12.42,  # Principal lunar semi-diurnal tide (hours)
            'S2': 12.0,   # Principal solar semi-diurnal tide (hours) 
            'N2': 12.66,  # Lunar elliptic semi-diurnal tide (hours)
            'K1': 23.93,  # Lunar diurnal tide (hours)
            'O1': 25.82,  # Lunar diurnal tide (hours)
        }
        
        # Wave seasonal patterns (simplified)
        self.wave_seasonal_pattern = {
            'winter_amplitude': 0.7,  # Higher waves in winter
            'summer_amplitude': 0.3,  # Lower waves in summer
            'mean_capacity_factor': 0.25,  # Base wave capacity factor
        }

    def load_site_data(self, path):
        """Load marine site data from CSV file."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Missing site file: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(p)
            logger.info(f"Loaded {len(df)} marine sites from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return pd.DataFrame()

    def create_time_index(self, year, freq='h'):
        """Create hourly time index for the specified year."""
        start = f"{year}-01-01"
        end = f"{year+1}-01-01"
        return pd.date_range(start=start, end=end, freq=freq, inclusive='left')

    def generate_tidal_stream_profile(self, time_index, site_lat=58.5, site_lon=-3.0):
        """
        Generate tidal stream capacity factor profile using harmonic constituents.
        
        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Time index for the profile
        site_lat : float
            Site latitude for location-specific tidal characteristics
        site_lon : float  
            Site longitude for location-specific tidal characteristics
            
        Returns
        -------
        pd.Series
            Hourly capacity factor profile (0-1)
        """
        # Convert time to hours from start
        hours = (time_index - time_index[0]).total_seconds() / 3600
        
        # Generate tidal harmonic components
        tidal_height = np.zeros(len(hours))
        
        # Apply harmonic constituents with site-specific amplitudes
        # UK waters have strong M2 and S2 components
        amplitudes = {
            'M2': 0.6,  # Dominant semi-diurnal 
            'S2': 0.3,  # Solar semi-diurnal
            'N2': 0.15, # Lunar elliptic
            'K1': 0.1,  # Diurnal
            'O1': 0.08, # Diurnal
        }
        
        # Adjust amplitudes based on latitude (stronger tides in northern UK)
        lat_factor = max(0.5, min(1.5, (site_lat - 50) / 10 + 0.8))
        
        for constituent, period in self.tidal_periods.items():
            amplitude = amplitudes.get(constituent, 0.1) * lat_factor
            phase = np.random.uniform(0, 2*np.pi)  # Random phase for site
            tidal_height += amplitude * np.sin(2 * np.pi * hours / period + phase)
        
        # Convert tidal height to velocity (approximate relationship)
        # Tidal stream velocity ∝ d(height)/dt
        tidal_velocity = np.gradient(tidal_height)
        
        # Convert velocity to power (P ∝ v³)
        power_raw = np.abs(tidal_velocity) ** 3
        
        # Normalize to 0-1 capacity factor with realistic maximum
        max_power = np.percentile(power_raw, 95)  # Use 95th percentile as max
        capacity_factor = np.clip(power_raw / max_power, 0, 1)
        
        # Apply cut-in and cut-out thresholds
        capacity_factor[capacity_factor < 0.05] = 0  # Cut-in threshold
        
        return pd.Series(capacity_factor, index=time_index)

    def generate_wave_profile(self, time_index, site_lat=50.5, site_lon=-4.0):
        """
        Generate wave power capacity factor profile using seasonal and stochastic patterns.
        
        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Time index for the profile
        site_lat : float
            Site latitude (affects seasonal patterns)
        site_lon : float
            Site longitude (affects exposure)
            
        Returns
        -------
        pd.Series
            Hourly capacity factor profile (0-1)
        """
        # Extract day of year for seasonal pattern
        day_of_year = time_index.dayofyear
        
        # Seasonal wave height variation (winter storms, summer calm)
        seasonal_amplitude = (
            self.wave_seasonal_pattern['winter_amplitude'] * 
            (0.5 - 0.5 * np.cos(2 * np.pi * day_of_year / 365.25))
        ) + self.wave_seasonal_pattern['summer_amplitude']
        
        # Add stochastic weather-driven variation
        # Use correlated random walk to simulate storm patterns
        n_hours = len(time_index)
        random_component = np.random.normal(0, 0.1, n_hours)
        
        # Apply autocorrelation to simulate persistent weather patterns
        alpha = 0.95  # Autocorrelation parameter
        wave_variation = np.zeros(n_hours)
        wave_variation[0] = random_component[0]
        
        for i in range(1, n_hours):
            wave_variation[i] = alpha * wave_variation[i-1] + (1-alpha) * random_component[i]
        
        # Combine seasonal and stochastic components
        base_cf = self.wave_seasonal_pattern['mean_capacity_factor']
        capacity_factor = base_cf + seasonal_amplitude * (0.5 + 0.5 * wave_variation)
        
        # Apply realistic constraints
        capacity_factor = np.clip(capacity_factor, 0, 1)
        
        # Add occasional storm events (high capacity factors)
        storm_probability = 0.02  # 2% chance per hour
        storm_mask = np.random.random(n_hours) < storm_probability
        capacity_factor = np.where(storm_mask, np.minimum(capacity_factor * 2.5, 1.0), capacity_factor)
        
        return pd.Series(capacity_factor, index=time_index)

    def generate_tidal_lagoon_profile(self, time_index, site_lat=51.5, site_lon=-4.0):
        """
        Generate tidal lagoon capacity factor profile based on tidal range and generation cycles.
        
        Tidal lagoons generate power during both flood and ebb tides when 
        sufficient head difference exists across the barrage.
        
        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Time index for the profile
        site_lat : float
            Site latitude for tidal range characteristics
        site_lon : float
            Site longitude
            
        Returns
        -------
        pd.Series
            Hourly capacity factor profile (0-1)
        """
        # Generate base tidal height using simplified harmonics
        hours = (time_index - time_index[0]).total_seconds() / 3600
        
        # Tidal lagoons work best with large tidal ranges (e.g., Severn Estuary)
        # Main semi-diurnal components
        M2_amplitude = 2.0  # meters
        S2_amplitude = 0.8  # meters
        
        tidal_height = (
            M2_amplitude * np.sin(2 * np.pi * hours / self.tidal_periods['M2']) +
            S2_amplitude * np.sin(2 * np.pi * hours / self.tidal_periods['S2'])
        )
        
        # Lagoon operates when head difference is sufficient
        # Simplistic model: generate when tidal flow is significant
        tidal_velocity = np.abs(np.gradient(tidal_height))
        
        # Convert to power generation (operates on both flood and ebb)
        # Power available when velocity > threshold
        velocity_threshold = 0.1
        
        # Generate power proportional to velocity squared
        capacity_factor = np.zeros(len(tidal_velocity))
        active_mask = tidal_velocity > velocity_threshold
        
        capacity_factor[active_mask] = np.minimum(
            (tidal_velocity[active_mask] / np.max(tidal_velocity)) ** 2, 1.0
        )
        
        # Apply operational constraints (not always generating at peak flow)
        # Lagoons have operational windows based on water levels
        capacity_factor *= 0.7  # Operational efficiency
        
        return pd.Series(capacity_factor, index=time_index)

    def generate_site_profiles(self, sites_df, technology, time_index):
        """
        Generate capacity factor profiles for all sites of a given technology.
        
        Parameters
        ----------
        sites_df : pd.DataFrame
            Site data with columns: site_name, capacity_mw, lat, lon
        technology : str
            Technology type ('tidal_stream', 'shoreline_wave', 'tidal_lagoon')
        time_index : pd.DatetimeIndex
            Time index for profiles
            
        Returns
        -------
        pd.DataFrame
            DataFrame with site profiles (capacity_factor * capacity_mw)
        """
        if sites_df.empty:
            logger.warning(f"No sites found for {technology}")
            return pd.DataFrame(index=time_index)
        
        logger.info(f"Generating {technology} profiles for {len(sites_df)} sites")
        
        profiles = {}
        
        for idx, row in sites_df.iterrows():
            site_name = row.get('site_name', f"site_{idx}")
            capacity = float(row.get('capacity_mw', 0.0) or 0.0)
            lat = float(row.get('lat', 55.0))  # Default to UK center
            lon = float(row.get('lon', -3.0))
            
            if capacity <= 0:
                logger.warning(f"Site {site_name} has zero capacity, skipping")
                continue
            
            # Generate technology-specific profile
            if technology == 'tidal_stream':
                cf_profile = self.generate_tidal_stream_profile(time_index, lat, lon)
            elif technology == 'shoreline_wave':
                cf_profile = self.generate_wave_profile(time_index, lat, lon)
            elif technology == 'tidal_lagoon':
                cf_profile = self.generate_tidal_lagoon_profile(time_index, lat, lon)
            else:
                logger.error(f"Unknown technology: {technology}")
                continue
            
            # Convert capacity factor to power output (MW)
            power_profile = cf_profile * capacity
            profiles[site_name] = power_profile
        
        # Create DataFrame with all site profiles
        profiles_df = pd.DataFrame(profiles, index=time_index)
        logger.info(f"Generated {technology} profiles: {len(profiles_df.columns)} sites, {len(profiles_df)} hours")
        
        return profiles_df

    def save_profiles(self, profiles_df, output_file):
        """Save profiles to CSV file."""
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            profiles_df.to_csv(output_file)
            logger.info(f"Saved marine profiles to {output_file}")


def main():
    """Main function called by Snakemake."""
    logger.info("Starting marine renewable profile generation")
    start_time = time.time()
    
    # Track processing metrics
    technologies_processed = []
    total_sites = 0
    profiles_created = 0
    
    try:
        generator = MarineRenewableGenerator()
        years = snakemake.params.renewables_year
        
        for i, year in enumerate(years):
            logger.info(f"Processing year {year}")
            
            # Create time index for the year
            time_index = generator.create_time_index(year)
            
            # Tidal stream
            if hasattr(snakemake.output, 'tidal_stream_profiles'):
                tidal_sites = generator.load_site_data(snakemake.input.tidal_stream)
                tidal_profiles = generator.generate_site_profiles(
                    tidal_sites, 'tidal_stream', time_index
                )
                generator.save_profiles(tidal_profiles, snakemake.output.tidal_stream_profiles[i])
                total_sites += len(tidal_sites)
                profiles_created += 1
                if 'tidal_stream' not in technologies_processed:
                    technologies_processed.append('tidal_stream')
            
            # Shoreline wave
            if hasattr(snakemake.output, 'shoreline_wave_profiles'):
                wave_sites = generator.load_site_data(snakemake.input.shoreline_wave)
                wave_profiles = generator.generate_site_profiles(
                    wave_sites, 'shoreline_wave', time_index
                )
                generator.save_profiles(wave_profiles, snakemake.output.shoreline_wave_profiles[i])
                total_sites += len(wave_sites)
                profiles_created += 1
                if 'shoreline_wave' not in technologies_processed:
                    technologies_processed.append('shoreline_wave')
            
            # Tidal lagoon
            if hasattr(snakemake.output, 'tidal_lagoon_profiles'):
                lagoon_sites = generator.load_site_data(snakemake.input.tidal_lagoon)
                lagoon_profiles = generator.generate_site_profiles(
                    lagoon_sites, 'tidal_lagoon', time_index
                )
                generator.save_profiles(lagoon_profiles, snakemake.output.tidal_lagoon_profiles[i])
                total_sites += len(lagoon_sites)
                profiles_created += 1
                if 'tidal_lagoon' not in technologies_processed:
                    technologies_processed.append('tidal_lagoon')
        
        logger.info("Marine renewable profile generation completed successfully")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'years_processed': len(years),
            'technologies_processed': len(technologies_processed),
            'technology_list': technologies_processed,
            'total_sites_processed': total_sites,
            'profiles_created': profiles_created
        }
        log_execution_summary(logger, "generate_marine_profiles", execution_time, summary_stats)
        
    except Exception as e:
        logger.error(f"Error in marine renewable profile generation: {e}")
        raise


if __name__ == '__main__':
    main()

