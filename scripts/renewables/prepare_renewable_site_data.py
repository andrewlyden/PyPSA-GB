"""
Prepare renewable site data for atlite profile generation.

This script processes renewable energy planning data (REPD) and offshore pipeline
data to create technology-specific site datasets for use with atlite.

Key functions:
- Process REPD data by technology and operational year
- Handle offshore wind pipeline projects  
- Convert coordinate systems (OSGB36 to WGS84)
- Create properly formatted CSV files for atlite input

Author: PyPSA-GB Team
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from pyproj import Transformer
import time

# Set up logging - compatible with both standalone and Snakemake execution
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("prepare_renewable_site_data")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    def log_execution_summary(logger, script_name, execution_time, summary_stats=None):
        """Fallback log_execution_summary when logging_config is not available."""
        logger.info(f"{script_name} completed in {execution_time:.2f}s")
        if summary_stats:
            logger.info(f"Summary: {summary_stats}")


class RenewableSiteDataProcessor:
    """Process renewable site data for atlite profile generation."""
    
    def __init__(self):
        """Initialize the processor with coordinate transformer."""
        # Set up coordinate transformation (OSGB36 -> WGS84)
        self.transformer = Transformer.from_crs('epsg:27700', 'epsg:4326', always_xy=True)
        
        # Technology mapping - used only as an allowlist reference historically.
        # We now process all unique Technology Type values present in REPD, but keep
        # this for backwards compatibility and variant names.
        self.technology_map = {
            'Solar Photovoltaics': 'Solar_Photovoltaics',
            'Wind Onshore': 'Wind_Onshore', 
            'Wind Offshore': 'Wind_Offshore',
            'Hot Dry Rocks (HDR)': 'Geothermal',
            'Geothermal': 'Geothermal',
            'Geothermal ': 'Geothermal',  # Handle trailing space
            'Hydro': 'Hydro',
            'Small Hydro': 'Hydro',
            'Large Hydro': 'Hydro',
            'Hydro (Small)': 'Hydro',
            'Hydro (Large)': 'Hydro',
            'Tidal Stream': 'Tidal_Stream',
            'Wave': 'Wave',
            'Shoreline Wave': 'Wave',
            'Tidal Lagoon': 'Tidal_Lagoon'
        }
    
    def convert_coordinates(self, x, y):
        """Convert OSGB36 coordinates to WGS84 lat/lon."""
        try:
            lon, lat = self.transformer.transform(float(x), float(y))
            return lat, lon
        except (ValueError, TypeError):
            logger.warning(f"Could not convert coordinates: x={x}, y={y}")
            return np.nan, np.nan
    
    def load_repd_data(self, repd_file):
        """
        Load and preprocess REPD data.
        
        Parameters
        ----------
        repd_file : str
            Path to REPD CSV file
            
        Returns
        -------
        pd.DataFrame
            Cleaned REPD dataframe
        """
        logger.info(f"Loading REPD data from {repd_file}")
        
        # Define required columns
        required_columns = [
            'Site Name', 'Technology Type', 'Installed Capacity (MWelec)',
            'CHP Enabled', 'Country', 'Turbine Capacity', 
            'No. of Turbines', 'Height of Turbines (m)',
            'Mounting Type for Solar', 'Development Status',
            'X-coordinate', 'Y-coordinate', 'Operational'
        ]
        
        try:
            # Load data
            df = pd.read_csv(
                repd_file, 
                encoding='unicode_escape', 
                usecols=required_columns,
                lineterminator='\n'
            )
            
            logger.info(f"Loaded {len(df)} records from REPD")
            
            # Filter data
            df = self._filter_repd_data(df)
            
            # Convert coordinates
            df = self._convert_repd_coordinates(df)
            
            logger.info(f"Processed REPD data: {len(df)} records remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error loading REPD data: {e}")
            raise
    
    def _filter_repd_data(self, df):
        """Filter REPD data to operational sites in GB with required technologies."""
        # Remove Northern Ireland
        df = df[df['Country'] != 'Northern Ireland'].copy()
        
        # Only operational sites
        df = df[df['Development Status'] == 'Operational'].copy()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=[
            'Installed Capacity (MWelec)', 
            'Technology Type', 
            'Site Name',
            'X-coordinate',
            'Y-coordinate'
        ])
        
        # Historically we filtered to a fixed allowlist; now we keep all
        # Technology Type values present to support broader coverage (e.g. storage,
        # conversion technologies) and generate generic site files for each type.
        # If a future need arises to exclude some, do it later in processing.
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _convert_repd_coordinates(self, df):
        """Convert REPD coordinates from OSGB36 to WGS84."""
        logger.info("Converting coordinates from OSGB36 to WGS84")
        
        # Apply coordinate conversion
        coords = df.apply(
            lambda row: self.convert_coordinates(row['X-coordinate'], row['Y-coordinate']), 
            axis=1, 
            result_type='expand'
        )
        
        df = df.copy()
        df[['lat', 'lon']] = coords
        
        # Remove rows with invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=['lat', 'lon'])
        final_count = len(df)
        
        if initial_count > final_count:
            logger.warning(f"Removed {initial_count - final_count} records with invalid coordinates")
        
        return df
    
    def filter_repd_by_year(self, df, year):
        """
        Filter REPD data to sites operational by specified year.
        
        Parameters
        ----------
        df : pd.DataFrame
            REPD dataframe
        year : int
            Cutoff year
            
        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        logger.info(f"Filtering REPD data for year {year}")
        
        # Convert operational dates to datetime
        operational_dates = pd.to_datetime(
            df['Operational'], 
            format='%d/%m/%Y', 
            errors='coerce'
        )
        
        # Filter to sites operational by end of specified year
        cutoff_date = pd.Timestamp(f'{year}-12-31')
        mask = operational_dates <= cutoff_date
        
        filtered_df = df[mask].copy()
        logger.info(f"Filtered to {len(filtered_df)} sites operational by {year}")
        
        return filtered_df
    
    def process_technology_data(self, df, technology, output_file):
        """
        Process data for a specific technology and save to file.
        
        Parameters
        ----------
        df : pd.DataFrame
            REPD dataframe
        technology : str
            Technology type (e.g., 'Geothermal', 'Wind Offshore')
        output_file : str
            Output CSV file path
        """
        logger.info(f"Processing {technology} data")
        
        # Map technology name to REPD technology types
        if technology == 'Geothermal':
            repd_tech_names = ['Hot Dry Rocks (HDR)', 'Geothermal', 'Geothermal ']
        elif technology == 'Small Hydro':
            # Accept both explicit labels and legacy 'Hydro'
            repd_tech_names = ['Small Hydro', 'Hydro', 'Hydro (Small)']
        elif technology == 'Large Hydro':
            repd_tech_names = ['Large Hydro', 'Hydro', 'Hydro (Large)']
        elif technology == 'Tidal Stream':
            repd_tech_names = ['Tidal Stream']
        elif technology in ['Wave', 'Shoreline Wave']:
            repd_tech_names = ['Shoreline Wave', 'Wave']
        elif technology == 'Tidal Lagoon':
            repd_tech_names = ['Tidal Lagoon']
        else:
            # For main technologies (wind, solar), use exact match
            repd_tech_names = [technology]
        
        # Filter to specific technology types
        tech_df = df[df['Technology Type'].isin(repd_tech_names)].copy()
        
        # For hydro, separate by capacity size
        if technology in ['Small Hydro', 'Large Hydro']:
            if len(tech_df) > 0:
                # Convert capacity to numeric, handling missing values
                tech_df['capacity_numeric'] = pd.to_numeric(tech_df['Installed Capacity (MWelec)'], errors='coerce')
                
                if technology == 'Small Hydro':
                    # Small hydro: <= 10 MW
                    tech_df = tech_df[tech_df['capacity_numeric'] <= 10].copy()
                else:
                    # Large hydro: > 10 MW
                    tech_df = tech_df[tech_df['capacity_numeric'] > 10].copy()
                
                # Drop the temporary column
                tech_df = tech_df.drop(columns=['capacity_numeric'])

        if len(tech_df) == 0:
            logger.warning(f"No {technology} sites found")
            # Create an empty CSV with expected headers to avoid read_csv errors downstream
            empty_cols = ['site_name', 'capacity_mw', 'lat', 'lon']
            pd.DataFrame(columns=empty_cols).to_csv(output_file, index=False)
            return

        # Rename columns to match downstream expectations
        rename_map = {
            'Site Name': 'site_name',
            'Installed Capacity (MWelec)': 'capacity_mw'
        }
        tech_df = tech_df.rename(columns={k: v for k, v in rename_map.items() if k in tech_df.columns})

        # Clean up columns for atlite
        # IMPORTANT: Keep 'Operational' column for year-based filtering in integrate_renewable_generators
        columns_to_drop = [
            'Technology Type', 'CHP Enabled', 'Development Status',
            'Mounting Type for Solar', 'Height of Turbines (m)',
            'Country'
        ]

        # Only drop columns that exist
        existing_columns_to_drop = [col for col in columns_to_drop if col in tech_df.columns]
        tech_df = tech_df.drop(columns=existing_columns_to_drop)
        
        # Rename and standardize the Operational column for downstream use
        if 'Operational' in tech_df.columns:
            tech_df = tech_df.rename(columns={'Operational': 'operational_date'})

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        tech_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(tech_df)} {technology} sites to {output_file}")
    
    def load_offshore_pipeline_data(self, pipeline_file):
        """
        Load offshore wind pipeline data.
        
        Parameters
        ----------
        pipeline_file : str
            Path to offshore pipeline CSV file
            
        Returns
        -------
        pd.DataFrame
            Processed pipeline dataframe
        """
        logger.info(f"Loading offshore pipeline data from {pipeline_file}")
        
        try:
            # Load pipeline data (do not force an index; schemas vary)
            df = pd.read_csv(pipeline_file, encoding='unicode_escape')
            
            # Normalize known SMP schema: name, area (km2), max capacity (GW), lon, lat
            if {'name','max capacity (GW)','lon','lat'}.issubset(df.columns):
                # Do NOT drop other columns (e.g., date fields); just add/normalize
                df = df.rename(columns={'name':'site_name'})
                # Convert GW -> MW for consistency with site files
                df['capacity_mw'] = pd.to_numeric(df['max capacity (GW)'], errors='coerce') * 1000.0
                # Ensure numeric coords
                df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
                df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            else:
                # Legacy REPD-style pipeline schema: drop irrelevant columns if present
                columns_to_drop = [
                    'Record Last Updated (dd/mm/yyyy)', 'Operator (or Applicant)',
                    'Under Construction', 'Technology Type',
                    'Planning Permission Expired', 'Operational',
                    'Heat Network Ref', 'Planning Authority',
                    'Planning Application Submitted', 'Region',
                    'Country', 'County', 'Development Status',
                    'Development Status (short)'
                ]
                existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
                if existing_columns_to_drop:
                    df = df.drop(columns=existing_columns_to_drop)
            
            # Remove columns that are all NaN
            df = df.dropna(axis='columns', how='all')
            
            logger.info(f"Loaded {len(df)} pipeline projects")
            return df
            
        except Exception as e:
            logger.error(f"Error loading pipeline data: {e}")
            raise
    
    def filter_pipeline_by_year(self, df, year):
        """
        Filter pipeline data to projects expected operational by specified year.

        Tries to locate an "expected operational" column by several common names.
        If none found, returns the dataframe unchanged (e.g., SMP dataset without dates).

        Parameters
        ----------
        df : pd.DataFrame
            Pipeline dataframe
        year : int
            Cutoff year

        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        logger.info(f"Filtering pipeline data for year {year}")

        # For years beyond 2030, include all pipeline projects
        filter_year = min(year, 2030)

        # Try to find a date column
        candidates = [
            'Expected Operational',
            'Expected Operational Date',
            'Expected Operation',
            'Operational Date',
            'Expected Operational Year',
            'Planned Operational',
        ]
        cols_norm = {str(c).strip().lower(): c for c in df.columns}
        date_col = None
        for cand in candidates:
            key = cand.strip().lower()
            if key in cols_norm:
                date_col = cols_norm[key]
                break

        if not date_col:
            logger.info("No expected operational date column found; skipping year filter for pipeline")
            return df

        # Convert operational dates
        operational_dates = pd.to_datetime(df[date_col], errors='coerce')

        # Filter to projects operational by end of specified year
        cutoff_date = pd.Timestamp(f'{filter_year}-12-31')
        mask = operational_dates <= cutoff_date

        filtered_df = df[mask].copy()
        logger.info(f"Filtered to {len(filtered_df)} pipeline projects for {year} using column '{date_col}'")

        return filtered_df
    
    def process_pipeline_coordinates(self, df):
        """Convert pipeline coordinates from OSGB36 to WGS84."""
        logger.info("Converting pipeline coordinates")
        
        if df.empty:
            logger.warning("Empty pipeline dataframe, skipping coordinate conversion")
            # Return empty df with expected columns
            df['lat'] = pd.Series(dtype=float)
            df['lon'] = pd.Series(dtype=float)
            return df
        
        if 'X-coordinate' not in df.columns or 'Y-coordinate' not in df.columns:
            logger.warning("Coordinate columns not found in pipeline data")
            return df
        
        # Convert coordinates
        coords = df.apply(
            lambda row: self.convert_coordinates(row['X-coordinate'], row['Y-coordinate']),
            axis=1,
            result_type='expand'
        )
        
        df[['lat', 'lon']] = coords
        
        # Drop original coordinate columns
        df = df.drop(columns=['X-coordinate', 'Y-coordinate'])
        
        # Remove rows with invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=['lat', 'lon'])
        final_count = len(df)
        
        if initial_count > final_count:
            logger.warning(f"Removed {initial_count - final_count} pipeline projects with invalid coordinates")
        
        return df
    
    def save_pipeline_data(self, df, output_file):
        """Save pipeline data to CSV file (no index)."""
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} pipeline projects to {output_file}")

    # ---------------------- Generic per-type processing ----------------------
    def _normalize_label(self, label: str) -> str:
        """Normalize Technology Type label to a filesystem-friendly snake_case name."""
        import re
        s = (label or '').strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
        return s

    def write_generic_site_file(self, df, output_file):
        """Write a generic site CSV with standard columns for downstream use."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure expected columns
        out_df = df.copy()
        rename_map = {
            'Site Name': 'site_name',
            'Installed Capacity (MWelec)': 'capacity_mw'
        }
        out_df = out_df.rename(columns={k: v for k, v in rename_map.items() if k in out_df.columns})
        # Drop obvious non-required columns if present
        drop_cols = [
            'Technology Type', 'CHP Enabled', 'Development Status', 'Operational',
            'Mounting Type for Solar', 'Height of Turbines (m)', 'Country'
        ]
        existing_drop = [c for c in drop_cols if c in out_df.columns]
        out_df = out_df.drop(columns=existing_drop)
        # Keep only rows with valid coordinates
        if {'lat', 'lon'}.issubset(out_df.columns):
            out_df = out_df.dropna(subset=['lat', 'lon'])
        out_df.to_csv(output_file, index=False)
        logger.info(f"Wrote generic site file: {output_file} ({len(out_df)} rows)")

    def process_all_repd_technologies(self, df, output_dir):
        """Process and export site CSVs for every unique Technology Type found in REPD."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tech_labels = sorted([t for t in df['Technology Type'].dropna().unique()])
        logger.info(f"Processing all REPD technology types: {tech_labels}")
        for label in tech_labels:
            normalized = self._normalize_label(label)
            out_file = output_dir / f"{normalized}_sites.csv"
            # Use the existing processor for consistency and special-cases
            try:
                self.process_technology_data(df, label, str(out_file))
            except Exception as e:
                logger.warning(f"Fell back to generic writer for '{label}' due to: {e}")
                # Fallback: filter directly by label and write generic
                fallback_df = df[df['Technology Type'] == label].copy()
                self.write_generic_site_file(fallback_df, str(out_file))


def main():
    """Main processing function called by Snakemake."""
    logger.info("Starting renewable site data processing")
    
def main():
    """Main function to process renewable site data."""
    start_time = time.time()
    logger.info("Starting renewable site data preparation...")
    
    try:
        # Initialize processor
        processor = RenewableSiteDataProcessor()
        
        # Check if running under Snakemake or standalone
        try:
            # Snakemake mode - try to access snakemake object
            renewables_years = snakemake.params.renewables_years
            modelled_years = snakemake.params.modelled_years
            repd_file = snakemake.input.repd
            pipeline_file = snakemake.input.offshore_pipeline
            wind_offshore_file = snakemake.output.wind_offshore_repd
            wind_onshore_file = snakemake.output.wind_onshore_repd
            solar_pv_file = snakemake.output.solar_pv_repd
            small_hydro_file = snakemake.output.small_hydro
            large_hydro_file = snakemake.output.large_hydro
            tidal_stream_file = snakemake.output.tidal_stream
            shoreline_wave_file = snakemake.output.shoreline_wave
            tidal_lagoon_file = snakemake.output.tidal_lagoon
            pipeline_output_file = snakemake.output.offshore_pipeline_processed
            logger.info("Running in Snakemake mode")
        except NameError:
            # Standalone mode - use default parameters and paths
            base_path = Path(__file__).parent.parent
            renewables_years = [2020, 2025, 2030]
            modelled_years = [2020, 2025, 2030]
            repd_file = base_path / "data" / "renewables" / "repd-q2-july-2024.csv"
            pipeline_file = base_path / "data" / "renewables" / "offshore_wind_pipeline.csv"
            output_dir = base_path / "resources" / "renewable"
            output_dir.mkdir(parents=True, exist_ok=True)
            wind_offshore_file = output_dir / "wind_offshore_repd.csv"
            wind_onshore_file = output_dir / "wind_onshore_repd.csv"
            solar_pv_file = output_dir / "solar_pv_repd.csv"
            small_hydro_file = output_dir / "small_hydro.csv"
            large_hydro_file = output_dir / "large_hydro.csv"
            tidal_stream_file = output_dir / "tidal_stream.csv"
            shoreline_wave_file = output_dir / "shoreline_wave.csv"
            tidal_lagoon_file = output_dir / "tidal_lagoon.csv"
            pipeline_output_file = output_dir / "offshore_pipeline_processed.csv"
            logger.info("Running in standalone mode")
        
        logger.info(f"Configured renewables years: {renewables_years}")
        logger.info(f"Configured modelled years: {modelled_years}")

        # Load and preprocess REPD data (operational sites only)
        repd_data = processor.load_repd_data(repd_file)

        # NOTE: We no longer filter by year here. Instead, we output the FULL dataset
        # with operational_date column preserved, and let integrate_renewable_generators.py
        # filter by the specific scenario's year. This ensures each historical year gets
        # the correct renewable capacity (e.g., 2010 gets only sites operational by 2010).
        logger.info("Using full REPD dataset with operational_date preserved for downstream filtering")
        filtered_repd = repd_data

        logger.info(f"Preparing technology site files from {len(filtered_repd)} operational sites")

        # Core technologies
        processor.process_technology_data(filtered_repd, 'Wind Offshore', wind_offshore_file)
        processor.process_technology_data(filtered_repd, 'Wind Onshore', wind_onshore_file)
        processor.process_technology_data(filtered_repd, 'Solar Photovoltaics', solar_pv_file)

        # Additional technologies (explicit labels in REPD; robust to variants)
        # Note: Geothermal now handled as dispatchable thermal in generators.smk
        processor.process_technology_data(filtered_repd, 'Small Hydro', small_hydro_file)
        processor.process_technology_data(filtered_repd, 'Large Hydro', large_hydro_file)
        processor.process_technology_data(filtered_repd, 'Tidal Stream', tidal_stream_file)
        processor.process_technology_data(filtered_repd, 'Shoreline Wave', shoreline_wave_file)
        processor.process_technology_data(filtered_repd, 'Tidal Lagoon', tidal_lagoon_file)

        # Note: Removed duplicate tech_types export to avoid saving sites in two places
        # All technology-specific site files are saved in the main renewable folder above

        # Offshore wind pipeline: write a single file filtered to the latest modelled year
        logger.info("Processing offshore wind pipeline data")
        pipeline_data = processor.load_offshore_pipeline_data(pipeline_file)
        if isinstance(modelled_years, (list, tuple, set)) and len(modelled_years) > 0:
            ref_year = max(modelled_years)
        else:
            # Fall back to a sensible default if not provided
            ref_year = pd.Timestamp.today().year
        logger.info(f"Filtering pipeline to projects operational by {ref_year}")
        year_pipeline = processor.filter_pipeline_by_year(pipeline_data, ref_year)
        year_pipeline = processor.process_pipeline_coordinates(year_pipeline)
        processor.save_pipeline_data(year_pipeline, pipeline_output_file)
        
        # Log execution summary
        execution_time = time.time() - start_time
        total_sites = len(filtered_repd) if 'filtered_repd' in locals() else 0
        pipeline_sites = len(year_pipeline) if 'year_pipeline' in locals() else 0
        
        summary_stats = {
            'repd_sites_processed': total_sites,
            'pipeline_sites_processed': pipeline_sites,
            'renewables_years': renewables_years,
            'modelled_years': modelled_years,
            'output_files_created': 10  # Number of technology output files
        }
        
        log_execution_summary(logger, "prepare_renewable_site_data", execution_time, summary_stats)
        logger.info("Renewable site data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in renewable site data processing: {e}")
        raise


if __name__ == '__main__':
    main()

