#!/usr/bin/env python3
"""
Load and process external storage data.

This script scans the data/storage/ directory for additional storage datasets
and normalizes them to a consistent schema for integration with REPD data.

Key functions:
- Scan data/storage/ directory for CSV and Excel files
- Normalize different data schemas to standard format
- Handle multiple file sources and merge intelligently
- Validate and clean data

Supported file formats:
- CSV files (.csv)
- Excel files (.xlsx, .xls)

Expected schema (flexible):
- site_name, technology, capacity_mw, energy_mwh, lat, lon, source

Author: PyPSA-GB Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import List, Optional

# Set up logging - compatible with both standalone and Snakemake execution
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("storage_external")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Column mapping for different data sources
COLUMN_MAPPINGS = {
    'site_name': ['site_name', 'project_name', 'name', 'facility_name', 'plant_name', 'station_name'],
    'technology': ['technology', 'tech', 'type', 'storage_type', 'technology_type'],
    'capacity_mw': ['capacity_mw', 'capacity', 'power_mw', 'rated_power', 'max_power', 'capacity(mw)'],
    'energy_mwh': ['energy_mwh', 'energy', 'storage_capacity', 'energy_capacity', 'capacity_mwh', 'energy(mwh)'],
    'lat': ['lat', 'latitude', 'y', 'northing'],
    'lon': ['lon', 'longitude', 'x', 'easting'],
    'source': ['source', 'data_source', 'provider', 'dataset']
}

# Technology standardization mapping
TECH_STANDARDIZATION = {
    'battery': 'Battery',
    'batteries': 'Battery',
    'battery storage': 'Battery',
    'battery energy storage': 'Battery',
    'lithium': 'Battery',
    'lithium-ion': 'Battery',
    'li-ion': 'Battery',
    'pumped storage': 'Pumped Storage Hydroelectricity',
    'pumped hydro': 'Pumped Storage Hydroelectricity',
    'pumped storage hydro': 'Pumped Storage Hydroelectricity',
    'caes': 'Compressed Air Energy Storage',
    'compressed air': 'Compressed Air Energy Storage',
    'laes': 'Liquid Air Energy Storage',
    'liquid air': 'Liquid Air Energy Storage',
    'cryogenic': 'Liquid Air Energy Storage',
    'flywheel': 'Flywheel',
    'flywheels': 'Flywheel'
}

def find_storage_files(storage_dir: Path) -> List[Path]:
    """
    Find all storage data files in the directory.
    
    Args:
        storage_dir: Path to storage data directory
        
    Returns:
        List of file paths
    """
    if not storage_dir.exists():
        logger.warning(f"Storage directory does not exist: {storage_dir}")
        return []
    
    file_patterns = ['*.csv', '*.xlsx', '*.xls']
    files = []
    
    for pattern in file_patterns:
        files.extend(storage_dir.glob(pattern))
    
    logger.info(f"Found {len(files)} potential storage data files")
    for f in files:
        logger.info(f"  - {f.name}")
    
    return files

def normalize_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Normalize column names to standard schema.
    
    Args:
        df: Input DataFrame
        filename: Source filename for logging
        
    Returns:
        DataFrame with normalized columns
    """
    logger.info(f"Normalizing columns for {filename}")
    
    # Create mapping for this file
    column_map = {}
    df_columns_lower = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    
    for standard_col, variants in COLUMN_MAPPINGS.items():
        found = False
        for variant in variants:
            variant_lower = variant.lower().replace(' ', '_').replace('-', '_')
            if variant_lower in df_columns_lower:
                original_col = df.columns[df_columns_lower.index(variant_lower)]
                column_map[original_col] = standard_col
                found = True
                break
        
        if not found and standard_col not in ['source']:  # source is optional
            logger.warning(f"Could not find column for {standard_col} in {filename}")
    
    # Apply mapping
    df_normalized = df.rename(columns=column_map)
    
    # Add missing columns with default values
    required_columns = ['site_name', 'technology', 'capacity_mw', 'energy_mwh', 'lat', 'lon', 'source']
    for col in required_columns:
        if col not in df_normalized.columns:
            if col == 'source':
                df_normalized[col] = filename
            else:
                df_normalized[col] = np.nan
    
    # Ensure source is set
    if df_normalized['source'].isna().all():
        df_normalized['source'] = filename
    
    logger.info(f"Normalized {filename}: {len(df_normalized)} rows, columns mapped: {list(column_map.keys())}")
    
    return df_normalized[required_columns]

def standardize_technology(tech_str: str) -> str:
    """
    Standardize technology names.
    
    Args:
        tech_str: Original technology string
        
    Returns:
        Standardized technology name
    """
    if pd.isna(tech_str):
        return 'Unknown'
    
    tech_lower = str(tech_str).lower().strip()
    
    # Direct lookup
    if tech_lower in TECH_STANDARDIZATION:
        return TECH_STANDARDIZATION[tech_lower]
    
    # Fuzzy matching
    for key, standard in TECH_STANDARDIZATION.items():
        if key in tech_lower:
            return standard
    
    # If no match found, return original with proper case
    return str(tech_str).title()

def load_file(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Load a single storage data file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        DataFrame or None if loading failed
    """
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            # Try to read Excel file, handling multiple sheets
            # Use context manager to ensure file is properly closed
            with pd.ExcelFile(file_path) as excel_file:
                if len(excel_file.sheet_names) == 1:
                    df = pd.read_excel(file_path)
                else:
                    # Multiple sheets - try to find the main data sheet
                    main_sheets = [s for s in excel_file.sheet_names 
                                  if any(keyword in s.lower() for keyword in ['data', 'storage', 'main', 'summary'])]
                    
                    if main_sheets:
                        df = pd.read_excel(file_path, sheet_name=main_sheets[0])
                        logger.info(f"Using sheet '{main_sheets[0]}' from {file_path.name}")
                    else:
                        # Use first sheet as default
                        df = pd.read_excel(file_path, sheet_name=0)
                        logger.info(f"Using first sheet from {file_path.name}")
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return None
        
        logger.info(f"Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the storage data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    logger.info(f"Cleaning and validating {initial_count} records...")
    
    # Clean capacity values
    df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce')
    df['energy_mwh'] = pd.to_numeric(df['energy_mwh'], errors='coerce')
    
    # Remove records with invalid or zero capacity
    valid_capacity = (df['capacity_mw'] > 0) | df['capacity_mw'].isna()
    df = df[valid_capacity].copy()  # Add .copy() to avoid SettingWithCopyWarning
    
    # Clean coordinate values
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    # Validate coordinates are within reasonable bounds for GB
    gb_bounds = {
        'lat_min': 49.0, 'lat_max': 61.5,
        'lon_min': -9.0, 'lon_max': 2.5
    }
    
    coord_mask = (
        ((df['lat'] >= gb_bounds['lat_min']) & (df['lat'] <= gb_bounds['lat_max']) &
         (df['lon'] >= gb_bounds['lon_min']) & (df['lon'] <= gb_bounds['lon_max'])) |
        df['lat'].isna()  # Keep records without coordinates
    )
    
    invalid_coords = len(df) - coord_mask.sum()
    if invalid_coords > 0:
        logger.warning(f"Removing {invalid_coords} records with invalid coordinates")
        df = df[coord_mask].copy()  # Add .copy() to avoid SettingWithCopyWarning
    
    # Standardize technology names
    df['technology'] = df['technology'].apply(standardize_technology)
    
    # Clean site names
    df['site_name'] = df['site_name'].fillna('Unknown Site')
    df['site_name'] = df['site_name'].astype(str).str.strip()
    
    final_count = len(df)
    logger.info(f"Validation complete: {final_count}/{initial_count} records retained")
    
    return df

def merge_files(dataframes: List[pd.DataFrame], filenames: List[str]) -> pd.DataFrame:
    """
    Merge data from multiple files with intelligent deduplication.
    
    Args:
        dataframes: List of DataFrames to merge
        filenames: List of source filenames
        
    Returns:
        Merged DataFrame
    """
    if not dataframes:
        logger.warning("No dataframes to merge")
        return pd.DataFrame(columns=['site_name', 'technology', 'capacity_mw', 'energy_mwh', 'lat', 'lon', 'source'])
    
    logger.info(f"Merging {len(dataframes)} datasets...")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} total records")
    
    # Deduplication strategy: group by site_name and technology
    # Keep record with highest capacity_mw, or most complete data if tied
    
    def select_best_record(group):
        """Select the best record from a group of duplicates."""
        if len(group) == 1:
            return group.iloc[0]
        
        # Prioritize records with energy data
        has_energy = group['energy_mwh'].notna()
        if has_energy.any():
            group = group[has_energy]
        
        # Prioritize records with coordinates
        has_coords = group['lat'].notna() & group['lon'].notna()
        if has_coords.any():
            group = group[has_coords]
        
        # Select record with highest capacity
        best_idx = group['capacity_mw'].idxmax()
        return group.loc[best_idx]
    
    # Group by site name and technology for deduplication
    grouped = combined_df.groupby(['site_name', 'technology'], dropna=False)
    deduplicated_records = []
    
    for _, group in grouped:
        best_record = select_best_record(group)
        deduplicated_records.append(best_record)
    
    result_df = pd.DataFrame(deduplicated_records).reset_index(drop=True)
    
    duplicates_removed = len(combined_df) - len(result_df)
    logger.info(f"Deduplication complete: {duplicates_removed} duplicates removed, {len(result_df)} unique sites")
    
    return result_df

def main():
    """Main function to load and process external storage data."""
    start_time = time.time()
    logger.info("Starting external storage data processing...")
    
    try:
        # Get input and output paths
        try:
            # Snakemake mode: prefer params, then inputs, then fallback to repo/data/storage
            storage_dir_candidate = None
            try:
                storage_dir_candidate = getattr(snakemake.params, 'storage_dir', None)
            except Exception:
                storage_dir_candidate = None

            if not storage_dir_candidate:
                # Try input (older rules may pass as input)
                try:
                    storage_dir_candidate = getattr(snakemake.input, 'storage_dir', None)
                except Exception:
                    storage_dir_candidate = None

            if storage_dir_candidate:
                storage_dir = Path(storage_dir_candidate)
            else:
                # Fallback to repository default
                base_path = Path(__file__).parent.parent.parent
                storage_dir = base_path / "data" / "storage"

            # Determine output path (named output preferred)
            try:
                output_file = Path(getattr(snakemake.output, 'storage_external', None) or snakemake.output[0])
            except Exception:
                base_path = Path(__file__).parent.parent.parent
                output_file = base_path / "resources" / "storage" / "storage_external.csv"

            logger.info("Running in Snakemake mode")
        except NameError:
            # Standalone mode
            base_path = Path(__file__).parent.parent.parent
            storage_dir = base_path / "data" / "storage"
            output_file = base_path / "resources" / "storage" / "storage_external.csv"
            logger.info("Running in standalone mode")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Find storage files
        storage_files = find_storage_files(storage_dir)
        
        if not storage_files:
            logger.warning("No storage data files found - creating empty output file")
            empty_df = pd.DataFrame(columns=[
                'site_name', 'technology', 'capacity_mw', 'energy_mwh', 'lat', 'lon', 'source'
            ])
            empty_df.to_csv(output_file, index=False)
            return
        
        # Load and process each file
        dataframes = []
        filenames = []
        
        for file_path in storage_files:
            df = load_file(file_path)
            if df is not None:
                # Normalize columns
                df_normalized = normalize_columns(df, file_path.name)
                
                # Only include if we have some meaningful data
                if len(df_normalized) > 0 and not df_normalized['site_name'].isna().all():
                    dataframes.append(df_normalized)
                    filenames.append(file_path.name)
                else:
                    logger.warning(f"Skipping {file_path.name} - no meaningful data found")
        
        if not dataframes:
            logger.warning("No valid storage data found - creating empty output file")
            empty_df = pd.DataFrame(columns=[
                'site_name', 'technology', 'capacity_mw', 'energy_mwh', 'lat', 'lon', 'source'
            ])
            empty_df.to_csv(output_file, index=False)
            return
        
        # Merge all datasets
        merged_df = merge_files(dataframes, filenames)
        
        # Clean and validate
        final_df = clean_and_validate(merged_df)
        
        # Save results
        final_df.to_csv(output_file, index=False)
        # Also save a simple external_sources.json summarizing processed files
        try:
            import json
            sources_meta = {
                'files_found': [str(p.name) for p in storage_files],
                'files_processed': filenames,
                'generated_at': time.ctime()
            }
            json_out = Path(output_file).parent / 'external_sources.json'
            with open(json_out, 'w', encoding='utf-8') as jf:
                json.dump(sources_meta, jf, indent=2)
            logger.info(f"Saved external sources metadata to: {json_out}")
        except Exception as e:
            logger.warning(f"Could not write external_sources.json: {e}")
        logger.info(f"Saved {len(final_df)} external storage sites to: {output_file}")
        
        # Generate summary statistics
        if len(final_df) > 0:
            tech_summary = final_df.groupby('technology').agg({
                'capacity_mw': ['count', 'sum'],
                'energy_mwh': lambda x: x.notna().sum(),
                'lat': lambda x: x.notna().sum()
            }).round(1)
            
            logger.info("External storage technology summary:")
            for tech in tech_summary.index:
                count = tech_summary.loc[tech, ('capacity_mw', 'count')]
                capacity = tech_summary.loc[tech, ('capacity_mw', 'sum')]
                with_energy = tech_summary.loc[tech, ('energy_mwh', '<lambda>')]
                with_coords = tech_summary.loc[tech, ('lat', '<lambda>')]
                logger.info(f"  {tech}: {count} sites, {capacity} MW, {with_energy} with energy data, {with_coords} with coordinates")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'files_found': len(storage_files),
            'files_processed': len(dataframes),
            'total_sites': len(final_df),
            'total_capacity_mw': final_df['capacity_mw'].sum() if len(final_df) > 0 else 0,
            'sites_with_coordinates': final_df['lat'].notna().sum() if len(final_df) > 0 else 0,
            'sites_with_energy_data': final_df['energy_mwh'].notna().sum() if len(final_df) > 0 else 0,
            'output_file': str(output_file)
        }
        
        log_execution_summary(logger, "storage_external", execution_time, summary_stats)
        logger.info("External storage data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in external storage processing: {e}")
        raise

if __name__ == "__main__":
    main()

