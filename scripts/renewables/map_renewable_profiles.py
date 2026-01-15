"""
Map renewable site data to capacity-factor profiles using optimized atlite methods.

Generates weather-dependent time series profiles for:
- Wind onshore
- Wind offshore (including pipeline projects)
- Solar PV

This script focuses on weather-variable renewables that require atlite-based timeseries.
Other renewable technologies are categorized as:
- Geothermal: Dispatchable baseload (constant ~90% capacity factor)
- Large Hydro: Storage-like operation (no generation profile needed)
- Small Hydro: Future timeseries implementation planned
- Marine renewables (tidal/wave): Synthetic cyclic timeseries (separate implementation)
"""

import logging
import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import atlite
import xarray as xr

# Centralized logging
try:
    from logging_config import setup_logging, log_execution_summary
    logger = setup_logging("map_renewable_profiles")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_execution_summary
        logger = setup_logging("map_renewable_profiles")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("map_renewable_profiles")

warnings.simplefilter('ignore')
logging.captureWarnings(False)


class RenewableProfileGenerator:
    def __init__(self):
        self.offshore_turbine_types = {
            3: 'Vestas_V112_3MW_offshore',
            5: 'NREL_ReferenceTurbine_5MW_offshore',
            7: 'Vestas_V164_7MW_offshore',
        }
        self.onshore_turbine_types = {
            0.66: 'Vestas_V47_660kW',
            2.3: 'Siemens_SWT_2300kW',
            3.0: 'Vestas_V112_3MW',
        }
        self.solar_panel_config = {'panel': 'CSi', 'orientation': 'latitude_optimal'}

    def load_cutout(self, path):
        logger.info(f"Loading cutout: {path}")
        return atlite.Cutout(path=path)

    def load_site_data(self, path):
        p = Path(path)
        if not p.exists():
            logger.warning(f"Missing site file: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(p)
            logger.info(f"Loaded {len(df)} sites from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return pd.DataFrame()

    def prepare_wind_sites(self, df):
        if df.empty:
            return pd.DataFrame()
        sdf = df.copy()
        for c in ['lat', 'lon', 'capacity_mw']:
            if c in sdf.columns:
                sdf[c] = pd.to_numeric(sdf[c], errors='coerce')
        sdf = sdf.dropna(subset=['lat', 'lon'])
        return sdf

    def prepare_solar_sites(self, df):
        return self.prepare_wind_sites(df)

    def _year_from_cutout(self, cutout):
        try:
            return pd.to_datetime(cutout.data.time[0].values).year
        except Exception:
            try:
                return pd.to_datetime(cutout.coords['time'].values[0]).year
            except Exception:
                return None

    def _get_capacity_factor_grid(self, cutout, technology):
        """Get capacity factor time series grid for a technology."""
        if technology in ('onshore', 'offshore'):
            turbine = 'Vestas_V112_3MW' if technology == 'onshore' else 'NREL_ReferenceTurbine_5MW_offshore'
            cf_grid = cutout.wind(turbine=turbine, capacity_factor_timeseries=True)
        elif technology == 'solar':
            cf_grid = cutout.pv(
                panel=self.solar_panel_config['panel'], 
                orientation=self.solar_panel_config['orientation'], 
                capacity_factor_timeseries=True
            )
        else:
            raise ValueError(f"Unknown technology: {technology}")
        
        return cf_grid

    def _compute_individual_site_profiles(self, cutout, sites_df, technology):
        """Compute individual site profiles using capacity factor time series grid.
        
        Returns power output in MW (capacity_factor * installed_capacity_mw).
        """
        # Get capacity factor time series grid for this technology
        cf_grid = self._get_capacity_factor_grid(cutout, technology)
        
        # Map each site to the nearest grid cell and get its time series
        time_index = pd.to_datetime(cf_grid.time.values)
        profiles = {}
        
        for idx, row in sites_df.iterrows():
            site_name = row.get('site_name', f"site_{idx}")
            capacity = float(row.get('capacity_mw', 0.0) or 0.0)
            
            if capacity <= 0:
                continue
            
            # Find nearest grid cell using atlite's grid selection
            site_cf = cf_grid.sel(x=row['lon'], y=row['lat'], method='nearest')
            
            # Get capacity factor time series
            site_cf_values = site_cf.values
            
            # Ensure capacity factors are in valid range [0, 1]
            site_cf_values = np.clip(site_cf_values, 0.0, 1.0)
            
            # Convert capacity factor to power output (MW)
            # This is what generators.smk expects - power in MW, not p_max_pu
            site_power_mw = site_cf_values * capacity
            
            profiles[site_name] = pd.Series(site_power_mw, index=time_index)
        
        # Create DataFrame with all site profiles
        profiles_df = pd.DataFrame(profiles)
        
        # Log statistics (convert back to CF for logging)
        mean_power = profiles_df.mean().mean()
        total_capacity = sites_df['capacity_mw'].sum()
        mean_cf = mean_power / (total_capacity / len(profiles_df.columns)) if len(profiles_df.columns) > 0 else 0
        
        logger.info(f"{technology} profiles: shape={profiles_df.shape}, mean_CF={mean_cf:.3f}, mean_power={mean_power:.1f}MW")
        
        return profiles_df

    def generate_wind_profiles(self, cutout, sites_df, technology, output_file):
        """Generate individual wind profiles for each site."""
        sdf = self.prepare_wind_sites(sites_df)
        logger.info(f"Generating {technology} wind for {len(sdf)} sites")
        
        if sdf.empty:
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(columns=['time']).to_csv(output_file, index=False)
            return pd.DataFrame()
        
        # Compute individual site profiles
        profiles_df = self._compute_individual_site_profiles(cutout, sdf, technology)
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            profiles_df.to_csv(output_file)
        
        return profiles_df

    def generate_solar_profiles(self, cutout, sites_df, output_file):
        """Generate individual solar profiles for each site."""
        sdf = self.prepare_solar_sites(sites_df)
        logger.info(f"Generating solar PV for {len(sdf)} sites")
        
        if sdf.empty:
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(columns=['time']).to_csv(output_file, index=False)
            return pd.DataFrame()
        
        # Compute individual site profiles
        profiles_df = self._compute_individual_site_profiles(cutout, sdf, 'solar')
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            profiles_df.to_csv(output_file)
        
        return profiles_df

    def combine_offshore_sites(self, repd_sites, pipeline_sites):
        """
        Combine REPD offshore wind sites with pipeline offshore wind sites.
        
        Parameters
        ----------
        repd_sites : pd.DataFrame
            REPD offshore wind sites
        pipeline_sites : pd.DataFrame
            Pipeline offshore wind sites
            
        Returns
        -------
        pd.DataFrame
            Combined offshore sites dataframe
        """
        logger.info("Combining REPD offshore sites with pipeline sites")
        
        # Handle empty dataframes
        if repd_sites.empty and pipeline_sites.empty:
            logger.warning("Both REPD and pipeline offshore datasets are empty")
            return pd.DataFrame()
        elif repd_sites.empty:
            logger.info("REPD offshore sites empty, using only pipeline sites")
            return pipeline_sites.copy()
        elif pipeline_sites.empty:
            logger.info("Pipeline sites empty, using only REPD offshore sites")
            return repd_sites.copy()
        
        # Ensure consistent column names
        repd_standardized = self._standardize_site_columns(repd_sites.copy(), 'REPD')
        pipeline_standardized = self._standardize_site_columns(pipeline_sites.copy(), 'Pipeline')
        
        # Combine the datasets
        combined = pd.concat([repd_standardized, pipeline_standardized], ignore_index=True)
        
        logger.info(f"Combined offshore sites: {len(repd_sites)} REPD + {len(pipeline_sites)} pipeline = {len(combined)} total")
        
        return combined
    
    def _standardize_site_columns(self, df, source_type):
        """
        Standardize column names for offshore site data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Site dataframe to standardize
        source_type : str
            Source type ('REPD' or 'Pipeline')
            
        Returns
        -------
        pd.DataFrame
            Standardized dataframe
        """
        # Standard column mapping
        column_map = {}
        
        # Handle different naming conventions
        if 'Site Name' in df.columns:
            column_map['Site Name'] = 'site_name'
        elif 'site_name' not in df.columns and 'name' in df.columns:
            column_map['name'] = 'site_name'
        
        if 'Installed Capacity (MWelec)' in df.columns:
            column_map['Installed Capacity (MWelec)'] = 'capacity_mw'
        elif 'capacity_mw' not in df.columns and 'max capacity (GW)' in df.columns:
            # Convert GW to MW
            df['capacity_mw'] = pd.to_numeric(df['max capacity (GW)'], errors='coerce') * 1000.0
        
        # Apply column renaming
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['site_name', 'capacity_mw', 'lat', 'lon']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column '{col}' in {source_type} data")
                if col == 'capacity_mw':
                    df[col] = 10.0  # Default capacity
                elif col == 'site_name':
                    df[col] = f"{source_type}_site_" + df.index.astype(str)
                else:
                    df[col] = 0.0
        
        # Add source identifier to site names to avoid duplicates
        if 'site_name' in df.columns:
            df['site_name'] = df['site_name'].astype(str) + f" ({source_type})"
        
        # Filter to required columns only
        df = df[required_cols].copy()
        
        # Clean data
        df = df.dropna(subset=['lat', 'lon'])
        df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce').fillna(10.0)
        
        return df


def main():
    logger.info("Starting renewable profile mapping for weather-variable renewables")
    start_time = time.time()
    
    # Track processing metrics
    technologies_processed = []
    total_sites = 0
    
    try:
        gen = RenewableProfileGenerator()
        years = snakemake.params.renewables_year
        
        # Build a mapping from year to cutout file and output files
        # This ensures correct matching regardless of expand() ordering
        cutout_files = list(snakemake.input.cutouts)
        onshore_outputs = list(snakemake.output.wind_onshore_profiles)
        offshore_outputs = list(snakemake.output.wind_offshore_profiles)
        solar_outputs = list(snakemake.output.solar_pv_profiles)
        
        # Create year-to-file mappings by parsing years from filenames
        def extract_year_from_path(path):
            """Extract year from filename like 'cutout_2015.nc' or 'wind_onshore_2015.csv'"""
            import re
            match = re.search(r'(\d{4})', Path(path).stem)
            return int(match.group(1)) if match else None
        
        cutout_by_year = {extract_year_from_path(f): f for f in cutout_files}
        onshore_output_by_year = {extract_year_from_path(f): f for f in onshore_outputs}
        offshore_output_by_year = {extract_year_from_path(f): f for f in offshore_outputs}
        solar_output_by_year = {extract_year_from_path(f): f for f in solar_outputs}
        
        logger.info(f"Processing {len(years)} years: {years}")
        
        for year in years:
            year_int = int(year)
            
            if year_int not in cutout_by_year:
                logger.error(f"No cutout found for year {year_int}")
                continue
            
            cutout_file = cutout_by_year[year_int]
            logger.info(f"Processing year {year_int} with cutout {cutout_file}")
            
            cutout = gen.load_cutout(cutout_file)
            
            # Verify cutout year matches expected year
            cutout_year = gen._year_from_cutout(cutout)
            if cutout_year and cutout_year != year_int:
                logger.warning(f"Cutout year {cutout_year} differs from expected {year_int}")
            
            # Wind onshore
            onshore_sites = gen.load_site_data(snakemake.input.wind_onshore)
            onshore_output = onshore_output_by_year.get(year_int)
            if onshore_output:
                gen.generate_wind_profiles(cutout, onshore_sites, 'onshore', onshore_output)
                total_sites += len(onshore_sites)
                if 'wind_onshore' not in technologies_processed:
                    technologies_processed.append('wind_onshore')
            
            # Wind offshore - combine REPD sites with pipeline sites
            offshore_sites = gen.load_site_data(snakemake.input.wind_offshore)
            pipeline_sites = gen.load_site_data("resources/renewable/offshore_pipeline_processed.csv")
            
            # Combine offshore datasets
            combined_offshore = gen.combine_offshore_sites(offshore_sites, pipeline_sites)
            offshore_output = offshore_output_by_year.get(year_int)
            if offshore_output:
                gen.generate_wind_profiles(cutout, combined_offshore, 'offshore', offshore_output)
                total_sites += len(combined_offshore)
                if 'wind_offshore' not in technologies_processed:
                    technologies_processed.append('wind_offshore')
            
            # Solar PV
            solar_sites = gen.load_site_data(snakemake.input.solar_pv)
            solar_output = solar_output_by_year.get(year_int)
            if solar_output:
                gen.generate_solar_profiles(cutout, solar_sites, solar_output)
                total_sites += len(solar_sites)
                if 'solar_pv' not in technologies_processed:
                    technologies_processed.append('solar_pv')
            
        logger.info("Weather-variable renewable profile mapping completed")
        logger.info("Note: Other renewable technologies (geothermal, hydro, marine) are categorized as:")
        logger.info("- Geothermal: Dispatchable baseload (no weather timeseries needed)")
        logger.info("- Large Hydro: Storage-like operation (handled separately)")
        logger.info("- Small Hydro: Future timeseries implementation planned")
        logger.info("- Marine renewables: Synthetic cyclic timeseries (separate implementation)")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'cutout_years': len(years),
            'technologies_processed': len(technologies_processed),
            'technology_list': technologies_processed,
            'total_sites_processed': total_sites,
            'output_files_created': len(years) * 3  # 3 technologies per year
        }
        log_execution_summary(logger, "map_renewable_profiles", execution_time, summary_stats)
        
    except Exception as e:
        logger.error(f"Error in renewable profile mapping: {e}")
        raise


if __name__ == '__main__':
    main()

