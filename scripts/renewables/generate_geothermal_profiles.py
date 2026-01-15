"""
Generate geothermal renewable profiles with dispatchable baseload characteristics.

Creates profiles for:
- Geothermal: Dispatchable baseload operation (high availability but flexible dispatch)

Geothermal facilities can operate as baseload power plants with very high availability,
but can also be dispatched flexibly based on system needs.
"""

import logging
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Centralized logging
try:
    from logging_config import setup_logging
    logger = setup_logging("generate_geothermal_profiles")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging
        logger = setup_logging("generate_geothermal_profiles")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("generate_geothermal_profiles")

warnings.simplefilter('ignore')
logging.captureWarnings(False)


class GeothermalProfileGenerator:
    def __init__(self):
        # UK geothermal characteristics
        self.geothermal_config = {
            'base_availability': 0.90,  # Very high availability for baseload operation
            'maintenance_probability': 0.08,  # 8% chance of maintenance outage per year
            'seasonal_variation': 0.02,  # Very small seasonal variation
            'dispatch_flexibility': 0.95,  # Can operate from 95% down to minimum levels
            'min_output_fraction': 0.3,  # Minimum stable output (30% of capacity)
        }
    
    def create_time_index(self, year):
        """Create hourly time index for the given year."""
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31 23:00:00"
        return pd.date_range(start=start_date, end=end_date, freq='H')
    
    def generate_geothermal_profile(self, year, sites_df):
        """
        Generate geothermal power output profiles.
        
        Geothermal plants operate as dispatchable baseload with:
        - Very high availability (90%+)
        - Minimal seasonal variation 
        - Flexible dispatch capability
        - Occasional maintenance outages
        
        Parameters
        ----------
        year : int
            Year for which to generate profiles
        sites_df : pd.DataFrame
            Geothermal site data with columns: site_name, capacity_mw, lat, lon
            
        Returns
        -------
        pd.DataFrame
            Profiles with datetime index and site columns containing power output (MW)
        """
        logger.info(f"Generating geothermal profiles for {year}")
        
        if sites_df.empty:
            logger.warning("No geothermal sites found")
            time_index = self.create_time_index(year)
            return pd.DataFrame(index=time_index)
        
        # Create time index
        time_index = self.create_time_index(year)
        n_hours = len(time_index)
        
        # Initialize profiles dictionary
        profiles = {}
        
        config = self.geothermal_config
        
        for idx, site in sites_df.iterrows():
            site_name = site.get('site_name', f"geothermal_site_{idx}")
            capacity_mw = site.get('capacity_mw', 10.0)
            
            logger.debug(f"Generating profile for {site_name} ({capacity_mw} MW)")
            
            # Base availability pattern (very stable)
            base_cf = np.full(n_hours, config['base_availability'])
            
            # Add very small seasonal variation (geothermal is very stable)
            # Slightly lower in summer due to cooling system efficiency
            seasonal_factor = 1 + config['seasonal_variation'] * np.cos(
                2 * np.pi * np.arange(n_hours) / (24 * 365.25) - np.pi
            )
            base_cf *= seasonal_factor
            
            # Add random maintenance periods
            # Geothermal plants typically have planned maintenance once per year
            n_maintenance_hours = int(n_hours * config['maintenance_probability'])
            
            if n_maintenance_hours > 0:
                # Schedule maintenance in late spring (April-May) when demand is lower
                maintenance_start_range = range(int(24 * 90), int(24 * 150))  # April-May
                maintenance_start = np.random.choice(maintenance_start_range)
                maintenance_duration = min(n_maintenance_hours, 24 * 14)  # Max 2 weeks
                
                maintenance_end = min(maintenance_start + maintenance_duration, n_hours)
                base_cf[maintenance_start:maintenance_end] *= 0.1  # 10% output during maintenance
            
            # Add small random variations for dispatch flexibility
            # Geothermal can be dispatched down when not needed
            random_dispatch = np.random.uniform(
                config['min_output_fraction'], 
                config['dispatch_flexibility'], 
                n_hours
            )
            
            # Apply dispatch patterns during low demand periods (overnight)
            hour_of_day = time_index.hour
            low_demand_mask = (hour_of_day >= 2) & (hour_of_day <= 5)
            dispatch_factor = np.where(
                low_demand_mask,
                random_dispatch * 0.8,  # Lower dispatch during low demand
                random_dispatch
            )
            
            # Combine all factors
            profile_cf = base_cf * dispatch_factor
            
            # Ensure capacity factors are within valid range [0, 1]
            profile_cf = np.clip(profile_cf, 0.0, 1.0)
            
            # Convert capacity factor to power output (MW)
            # This is what generators.smk expects - power in MW
            profile_power_mw = profile_cf * capacity_mw
            
            # Convert to pandas Series
            profiles[site_name] = pd.Series(profile_power_mw, index=time_index)
        
        # Create DataFrame with all site profiles
        profiles_df = pd.DataFrame(profiles)
        
        # Log statistics
        if not profiles_df.empty and not sites_df.empty:
            mean_power = profiles_df.mean().mean()
            total_capacity = sites_df['capacity_mw'].sum()
            mean_cf = (profiles_df.sum(axis=1).mean() / total_capacity) if total_capacity > 0 else 0
            
            logger.info(f"Generated geothermal profiles for {len(profiles_df.columns)} sites")
            logger.info(f"Total capacity: {total_capacity:.1f} MW")
            logger.info(f"Average capacity factor: {mean_cf:.3f}")
            logger.info(f"Average power output: {mean_power:.1f} MW")
            logger.info(f"Profile shape: {profiles_df.shape}")
        else:
            logger.info(f"Generated empty geothermal profiles")
        
        return profiles_df
    
    def save_profiles(self, profiles_df, output_file):
        """Save profiles to CSV file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving geothermal profiles to {output_file}")
        profiles_df.to_csv(output_file)
        
        # Log summary statistics
        if not profiles_df.empty:
            logger.info(f"Profile statistics:")
            logger.info(f"  Mean power: {profiles_df.mean().mean():.1f} MW")
            logger.info(f"  Min power: {profiles_df.min().min():.1f} MW")
            logger.info(f"  Max power: {profiles_df.max().max():.1f} MW")
            logger.info(f"  Std power: {profiles_df.std().mean():.1f} MW")


def main():
    """Main execution function for Snakemake."""
    logger.info("Starting geothermal profile generation")
    
    try:
        # Access snakemake variables
        snk = globals().get('snakemake')
        if not snk:
            raise RuntimeError("This script must be run through Snakemake")
        
        # Get parameters
        years = snk.params.renewables_year
        
        # Load geothermal site data
        geothermal_sites_file = snk.input.geothermal
        if Path(geothermal_sites_file).exists():
            geothermal_sites = pd.read_csv(geothermal_sites_file)
            logger.info(f"Loaded {len(geothermal_sites)} geothermal sites")
        else:
            logger.warning(f"Geothermal sites file not found: {geothermal_sites_file}")
            geothermal_sites = pd.DataFrame()
        
        # Initialize generator
        generator = GeothermalProfileGenerator()
        
        # Generate profiles for each year
        for i, year in enumerate(years):
            logger.info(f"Processing year {year}")
            
            # Generate geothermal profiles
            geothermal_profiles = generator.generate_geothermal_profile(year, geothermal_sites)
            
            # Save profiles
            generator.save_profiles(geothermal_profiles, snk.output.geothermal_profiles[i])
        
        logger.info("Geothermal profile generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in geothermal profile generation: {e}")
        raise


if __name__ == "__main__":
    main()

