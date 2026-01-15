#!/usr/bin/env python3
"""
Extract storage assets from REPD data.

This script processes the REPD (Renewable Energy Planning Database) to extract
storage technologies and standardize them for PyPSA-GB integration.

Key functions:
- Filter REPD for storage technologies
- Handle coordinate conversion from OSGB36 to WGS84
- Standardize output schema
- Filter to Great Britain only

Storage technologies processed:
- Battery Energy Storage Systems
- Pumped Storage Hydroelectricity  
- Compressed Air Energy Storage (CAES)
- Liquid Air Energy Storage (LAES)
- Flywheels
- Other emerging storage technologies

Author: PyPSA-GB Team
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from pyproj import Transformer
import time

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set up logging - compatible with both standalone and Snakemake execution
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("storage_from_repd")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    def log_execution_summary(*args, **kwargs):
        pass  # No-op fallback

# Storage technology mapping
STORAGE_TECHNOLOGIES = {
    'Battery': ['Battery', 'Battery Energy Storage System', 'Battery Storage'],
    'Pumped Storage Hydroelectricity': ['Pumped Storage Hydroelectricity', 'Pumped Hydro', 'Pumped Storage'],
    'Compressed Air Energy Storage': ['Compressed Air Energy Storage', 'CAES'],
    'Liquid Air Energy Storage': ['Liquid Air Energy Storage', 'LAES', 'Cryogenic Energy Storage'],
    'Flywheel': ['Flywheel', 'Flywheels', 'Flywheel Energy Storage']
}

# Flatten technology mapping for easier lookup
STORAGE_TECH_LOOKUP = {}
for standard_name, variants in STORAGE_TECHNOLOGIES.items():
    for variant in variants:
        STORAGE_TECH_LOOKUP[variant.lower()] = standard_name

def is_storage_technology(technology_str):
    """
    Check if a technology string represents a storage technology.
    
    Args:
        technology_str: Technology string from REPD
        
    Returns:
        bool: True if storage technology, False otherwise
    """
    if pd.isna(technology_str):
        return False
    
    tech_lower = str(technology_str).lower()
    
    # Direct lookup
    if tech_lower in STORAGE_TECH_LOOKUP:
        return True
    
    # Fuzzy matching for common variants
    storage_keywords = ['battery', 'storage', 'pumped', 'compressed air', 'liquid air', 'flywheel']
    for keyword in storage_keywords:
        if keyword in tech_lower:
            return True
    
    return False

def standardize_technology_name(technology_str):
    """
    Standardize technology name to consistent format.
    
    Args:
        technology_str: Original technology string
        
    Returns:
        str: Standardized technology name
    """
    if pd.isna(technology_str):
        return 'Unknown'
    
    tech_lower = str(technology_str).lower()
    
    # Direct lookup
    if tech_lower in STORAGE_TECH_LOOKUP:
        return STORAGE_TECH_LOOKUP[tech_lower]
    
    # Fuzzy matching and classification
    if 'battery' in tech_lower:
        return 'Battery'
    elif 'pumped' in tech_lower:
        return 'Pumped Storage Hydroelectricity'
    elif 'compressed air' in tech_lower or 'caes' in tech_lower:
        return 'Compressed Air Energy Storage'
    elif 'liquid air' in tech_lower or 'laes' in tech_lower or 'cryogenic' in tech_lower:
        return 'Liquid Air Energy Storage'
    elif 'flywheel' in tech_lower:
        return 'Flywheel'
    else:
        logger.warning(f"Unknown storage technology: {technology_str}")
        return 'Other Storage'

def convert_coordinates(df):
    """
    Convert coordinates from OSGB36 to WGS84 where needed.
    
    Args:
        df: DataFrame with coordinate columns
        
    Returns:
        DataFrame: Updated with lat/lon coordinates
    """
    logger.info("Processing coordinate conversion")
    
    # Initialize coordinate columns
    df['lat'] = np.nan
    df['lon'] = np.nan
    
    # Check for existing lat/lon columns
    has_latlon = 'lat' in df.columns or 'lon' in df.columns or 'latitude' in df.columns or 'longitude' in df.columns
    has_xy = 'X-coordinate' in df.columns and 'Y-coordinate' in df.columns
    
    converted_count = 0
    
    if has_xy:
        # Set up coordinate transformer for OSGB36 to WGS84
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        
        # Identify rows with valid OSGB36 coordinates
        valid_xy = df['X-coordinate'].notna() & df['Y-coordinate'].notna()
        valid_xy_count = valid_xy.sum()
        
        if valid_xy_count > 0:
            logger.info(f"Converting {valid_xy_count} sites from OSGB36 to WGS84")
            
            # Convert coordinates
            x_coords = df.loc[valid_xy, 'X-coordinate'].values
            y_coords = df.loc[valid_xy, 'Y-coordinate'].values
            
            # Transform coordinates
            lons, lats = transformer.transform(x_coords, y_coords)
            
            df.loc[valid_xy, 'lon'] = lons
            df.loc[valid_xy, 'lat'] = lats
            converted_count = valid_xy_count
    
    # Use existing lat/lon if available and no conversion was done
    if has_latlon and converted_count == 0:
        for lat_col in ['latitude', 'Latitude', 'lat']:
            if lat_col in df.columns:
                df['lat'] = df['lat'].fillna(df[lat_col])
                break
        
        for lon_col in ['longitude', 'Longitude', 'lon']:
            if lon_col in df.columns:
                df['lon'] = df['lon'].fillna(df[lon_col])
                break
    
    # Validate coordinates are within GB bounds
    gb_bounds = {
        'lat_min': 49.5, 'lat_max': 61.0,  # Scotland to South England
        'lon_min': -8.5, 'lon_max': 2.0    # West Wales to East England
    }
    
    valid_coords = (
        (df['lat'] >= gb_bounds['lat_min']) & (df['lat'] <= gb_bounds['lat_max']) &
        (df['lon'] >= gb_bounds['lon_min']) & (df['lon'] <= gb_bounds['lon_max'])
    )
    
    invalid_count = (~valid_coords & df['lat'].notna()).sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} sites with coordinates outside GB bounds")
        df.loc[~valid_coords, ['lat', 'lon']] = np.nan
    
    valid_final = df['lat'].notna() & df['lon'].notna()
    logger.info(f"Final coordinate processing: {valid_final.sum()} sites with valid coordinates")
    
    return df

def extract_commissioning_year(df):
    """
    Extract commissioning year from available date columns.
    
    Args:
        df: DataFrame with potential date columns
        
    Returns:
        DataFrame: Updated with commissioning_year column
    """
    df['commissioning_year'] = np.nan
    
    # Common date column names in REPD
    date_columns = ['Record Last Updated', 'Operational Date', 'Planning Application Submitted', 
                   'Planning Permission Granted', 'Under Construction Date']
    
    for col in date_columns:
        if col in df.columns:
            try:
                # Convert to datetime and extract year
                dates = pd.to_datetime(df[col], errors='coerce')
                years = dates.dt.year
                
                # Fill missing years
                df['commissioning_year'] = df['commissioning_year'].fillna(years)
                logger.info(f"Extracted {years.notna().sum()} commissioning years from {col}")
                break
            except Exception as e:
                logger.warning(f"Failed to process date column {col}: {e}")
                continue
    
    return df

def filter_gb_only(df):
    """
    Filter data to Great Britain only (exclude Northern Ireland).
    
    Args:
        df: DataFrame with location data
        
    Returns:
        DataFrame: Filtered to GB only
    """
    initial_count = len(df)
    
    # Check for country/region columns
    region_columns = ['Country', 'Region', 'Nation']
    gb_regions = ['England', 'Scotland', 'Wales']
    
    for col in region_columns:
        if col in df.columns:
            gb_mask = df[col].isin(gb_regions)
            df = df[gb_mask]
            logger.info(f"Filtered by {col}: {len(df)}/{initial_count} sites in GB")
            break
    else:
        # If no explicit region column, use coordinate bounds
        if 'lat' in df.columns and 'lon' in df.columns:
            gb_bounds = {
                'lat_min': 49.5, 'lat_max': 61.0,
                'lon_min': -8.5, 'lon_max': 2.0
            }
            
            gb_mask = (
                (df['lat'] >= gb_bounds['lat_min']) & (df['lat'] <= gb_bounds['lat_max']) &
                (df['lon'] >= gb_bounds['lon_min']) & (df['lon'] <= gb_bounds['lon_max'])
            )
            
            df = df[gb_mask | df['lat'].isna()]  # Keep sites with missing coordinates
            logger.info(f"Filtered by coordinates: {len(df)}/{initial_count} sites in GB bounds")
    
    return df

def main():
    """Main function to extract storage assets from REPD."""
    start_time = time.time()
    logger.info("Starting storage extraction from REPD...")
    
    try:
        # Get input and output files
        try:
            # Snakemake mode
            input_file = snakemake.input.repd
            output_file = snakemake.output.storage_params
            include_pipeline = snakemake.params.include_pipeline
            logger.info("Running in Snakemake mode")
        except NameError:
            # Standalone mode
            base_path = Path(__file__).parent.parent.parent
            input_file = base_path / "data" / "renewables" / "repd-q2-jul-2025.csv"
            output_file = base_path / "resources" / "storage" / "storage_from_repd.csv"
            include_pipeline = False
            logger.info("Running in standalone mode")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load REPD data with encoding detection
        logger.info(f"Loading REPD data from: {input_file}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                logger.info(f"Successfully loaded REPD data using {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.debug(f"Failed to load with {encoding} encoding")
                continue
        
        if df is None:
            raise ValueError(f"Could not read REPD file with any of the tried encodings: {encodings}")
            
        logger.info(f"Loaded {len(df)} records from REPD")
        
        # Filter for storage technologies
        logger.info("Filtering for storage technologies...")
        storage_mask = df['Technology Type'].apply(is_storage_technology)
        storage_df = df[storage_mask].copy()
        logger.info(f"Found {len(storage_df)} storage technology records")
        
        if len(storage_df) == 0:
            logger.warning("No storage technologies found in REPD data")
            # Create empty output with correct schema
            empty_df = pd.DataFrame(columns=[
                'site_name', 'technology', 'capacity_mw', 'lat', 'lon', 
                'status', 'commissioning_year'
            ])
            empty_df.to_csv(output_file, index=False)
            return
        
        # Standardize technology names
        logger.info("Standardizing technology names...")
        storage_df['technology'] = storage_df['Technology Type'].apply(standardize_technology_name)
        
        # Extract basic information
        storage_df['site_name'] = storage_df.get('Site Name', 'Unknown')
        storage_df['capacity_mw'] = pd.to_numeric(storage_df.get('Installed Capacity (MWelec)', 0), errors='coerce')
        storage_df['status'] = storage_df.get('Development Status', 'Unknown')
        
        # Convert coordinates
        storage_df = convert_coordinates(storage_df)
        
        # Extract commissioning year
        storage_df = extract_commissioning_year(storage_df)
        
        # Filter to GB only
        storage_df = filter_gb_only(storage_df)
        
        # Filter by operational status
        if not include_pipeline:
            logger.info("Filtering to operational projects only...")
            operational_statuses = ['Operational', 'Built', 'Operating', 'In Operation']
            status_mask = storage_df['status'].isin(operational_statuses)
            storage_df = storage_df[status_mask]
            logger.info(f"After operational filter: {len(storage_df)} storage sites")
        
        # Apply technology-specific parameter defaults
        logger.info("Applying technology-specific parameter defaults...")
        
        # Technology defaults from rule parameters (simplified approach)
        tech_defaults = {
            'Battery': {
                'eta_charge': 0.95,
                'eta_discharge': 0.95,
                'duration_h': 2.0,
                'standing_loss': 0.001
            },
            'Pumped Storage Hydroelectricity': {
                'eta_charge': 0.87,
                'eta_discharge': 0.90,
                'duration_h': 8.0,
                'standing_loss': 0.0
            },
            'Compressed Air Energy Storage': {
                'eta_charge': 0.70,
                'eta_discharge': 0.70,
                'duration_h': 12.0,
                'standing_loss': 0.0001
            },
            'Liquid Air Energy Storage': {
                'eta_charge': 0.60,
                'eta_discharge': 0.60,
                'duration_h': 10.0,
                'standing_loss': 0.0004
            },
            'Flywheel': {
                'eta_charge': 0.95,
                'eta_discharge': 0.95,
                'duration_h': 0.25,  # 15 minutes typical
                'standing_loss': 0.01  # High self-discharge
            }
        }
        
        # Apply defaults based on technology type
        for tech, defaults in tech_defaults.items():
            mask = storage_df['technology'] == tech
            for param, value in defaults.items():
                storage_df.loc[mask, param] = value
        
        # Calculate energy capacity from power × duration
        storage_df['power_mw'] = storage_df['capacity_mw']
        storage_df['energy_mwh'] = storage_df['power_mw'] * storage_df['duration_h']
        
        # For technologies not in defaults, apply generic battery defaults
        mask_no_params = storage_df['eta_charge'].isna()
        if mask_no_params.any():
            logger.warning(f"Applying generic battery defaults to {mask_no_params.sum()} sites with unmapped technology")
            storage_df.loc[mask_no_params, 'eta_charge'] = 0.90
            storage_df.loc[mask_no_params, 'eta_discharge'] = 0.90
            storage_df.loc[mask_no_params, 'duration_h'] = 2.0
            storage_df.loc[mask_no_params, 'standing_loss'] = 0.001
            storage_df.loc[mask_no_params, 'energy_mwh'] = storage_df.loc[mask_no_params, 'power_mw'] * 2.0
        
        # Select and order output columns
        output_columns = [
            'site_name', 'technology', 'power_mw', 'energy_mwh', 'duration_h',
            'eta_charge', 'eta_discharge', 'standing_loss', 'lat', 'lon', 
            'status', 'commissioning_year'
        ]
        
        # Ensure all columns exist
        for col in output_columns:
            if col not in storage_df.columns:
                storage_df[col] = np.nan
        
        output_df = storage_df[output_columns].copy()
        
        # Clean up data
        output_df['power_mw'] = output_df['power_mw'].fillna(0)
        output_df['energy_mwh'] = output_df['energy_mwh'].fillna(0)
        output_df['duration_h'] = output_df['duration_h'].fillna(2.0)
        output_df['eta_charge'] = output_df['eta_charge'].fillna(0.90)
        output_df['eta_discharge'] = output_df['eta_discharge'].fillna(0.90)
        output_df['standing_loss'] = output_df['standing_loss'].fillna(0.001)
        output_df['site_name'] = output_df['site_name'].fillna('Unknown Site')
        output_df['technology'] = output_df['technology'].fillna('Battery')
        output_df['status'] = output_df['status'].fillna('Unknown')
        
        # Save results
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(output_df)} storage sites to: {output_file}")
        
        # Generate summary statistics
        tech_summary = output_df.groupby('technology').agg({
            'power_mw': ['count', 'sum'],
            'energy_mwh': 'sum',
            'lat': lambda x: x.notna().sum()  # Count with coordinates
        }).round(1)
        
        logger.info("Storage technology summary:")
        for tech in tech_summary.index:
            count = tech_summary.loc[tech, ('power_mw', 'count')]
            power = tech_summary.loc[tech, ('power_mw', 'sum')]
            energy = tech_summary.loc[tech, ('energy_mwh', 'sum')]
            with_coords = tech_summary.loc[tech, ('lat', '<lambda>')]
            logger.info(f"  {tech}: {count} sites, {power} MW, {energy} MWh, {with_coords} with coordinates")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'total_repd_records': len(df),
            'storage_records_found': len(storage_df),
            'final_storage_sites': len(output_df),
            'total_power_mw': output_df['power_mw'].sum(),
            'total_energy_mwh': output_df['energy_mwh'].sum(),
            'sites_with_coordinates': output_df['lat'].notna().sum(),
            'output_file': str(output_file)
        }
        
        log_execution_summary(logger, "storage_from_repd", execution_time, summary_stats)
        logger.info("Storage extraction from REPD completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in storage extraction: {e}")
        raise

if __name__ == "__main__":
    main()

