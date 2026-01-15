#!/usr/bin/env python3
"""
Clean Interconnector Data
=========================

This script cleans and standardizes interconnector data, applying overrides
and validating data quality. It ensures consistent naming, validates numeric
fields, and handles deduplication.

Key features:
- Data normalization and cleaning
- Override application with precedence handling
- Validation of capacity, losses, and other fields
- Deduplication by key identifiers
- Comprehensive data quality reporting

Author: PyPSA-GB Team
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import re
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
    input_raw = snakemake.input[0]
    overrides_file = snakemake.params.overrides_file
    output_clean = snakemake.output[0]
else:
    SNAKEMAKE_MODE = False

def slugify_name(name: str) -> str:
    """
    Convert a name to a consistent slug format for matching.
    
    Args:
        name: Original name string
        
    Returns:
        Slugified name for consistent matching
    """
    if pd.isna(name):
        return ""
    
    # Convert to lowercase and replace non-alphanumeric with underscores
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', str(name).lower().strip())
    # Remove leading/trailing underscores
    slug = slug.strip('_')
    return slug

def validate_interconnector_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate interconnector data and remove invalid records.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validated DataFrame with invalid records removed
    """
    logger = logging.getLogger(__name__)
    initial_count = len(df)
    
    # Track validation failures
    validation_issues = []
    
    # Check for required fields
    required_fields = ['name', 'capacity_mw']
    for field in required_fields:
        if field not in df.columns:
            logger.error(f"Required field '{field}' missing from data")
            return pd.DataFrame()
    
    # Validate capacity
    invalid_capacity = (df['capacity_mw'].isna()) | (df['capacity_mw'] <= 0)
    if invalid_capacity.any():
        count = invalid_capacity.sum()
        validation_issues.append(f"Invalid capacity: {count} records")
        df = df[~invalid_capacity]
    
    # Validate losses percentage (if present)
    if 'losses_percent' in df.columns:
        invalid_losses = (df['losses_percent'] < 0) | (df['losses_percent'] > 50)
        if invalid_losses.any():
            count = invalid_losses.sum()
            validation_issues.append(f"Invalid losses percentage: {count} records")
            # Set to default rather than remove
            df.loc[invalid_losses, 'losses_percent'] = 2.5
    
    # Validate commissioning year (if present)
    if 'commissioning_year' in df.columns:
        current_year = pd.Timestamp.now().year
        invalid_year = (df['commissioning_year'] < 1950) | (df['commissioning_year'] > current_year + 20)
        if invalid_year.any():
            count = invalid_year.sum()
            validation_issues.append(f"Invalid commissioning year: {count} records")
            # Set to NaN rather than remove
            df.loc[invalid_year, 'commissioning_year'] = np.nan
    
    # Check for empty names
    empty_names = df['name'].isna() | (df['name'].str.strip() == '')
    if empty_names.any():
        count = empty_names.sum()
        validation_issues.append(f"Empty names: {count} records")
        df = df[~empty_names]
    
    final_count = len(df)
    removed_count = initial_count - final_count
    
    if validation_issues:
        logger.warning(f"Validation issues found: {'; '.join(validation_issues)}")
    
    logger.info(f"Validation complete: {removed_count} invalid records removed, {final_count} records retained")
    return df

def apply_overrides(df: pd.DataFrame, overrides_file: str) -> pd.DataFrame:
    """
    Apply override data to the base interconnector dataset.
    
    Args:
        df: Base interconnector DataFrame
        overrides_file: Path to overrides CSV file
        
    Returns:
        DataFrame with overrides applied
    """
    logger = logging.getLogger(__name__)
    
    if not Path(overrides_file).exists():
        logger.info("No overrides file found - skipping override application")
        return df
    
    try:
        overrides_df = pd.read_csv(overrides_file)
        logger.info(f"Loaded {len(overrides_df)} override records")
        
        if len(overrides_df) == 0:
            return df
        
        # Create slugified names for matching
        df['name_slug'] = df['name'].apply(slugify_name)
        overrides_df['name_slug'] = overrides_df['name'].apply(slugify_name)
        
        # Track override applications
        overrides_applied = 0
        new_records_added = 0
        
        for _, override_row in overrides_df.iterrows():
            override_slug = override_row['name_slug']
            
            # Find matching records in base data
            matches = df[df['name_slug'] == override_slug]
            
            if len(matches) > 0:
                # Apply overrides to existing records
                for col in override_row.index:
                    if col in df.columns and col != 'name_slug' and pd.notna(override_row[col]):
                        df.loc[matches.index, col] = override_row[col]
                        overrides_applied += 1
                        logger.debug(f"Applied override: {override_slug}.{col} = {override_row[col]}")
            else:
                # Add new record if not found in base data
                new_row = override_row.drop('name_slug').to_dict()
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                new_records_added += 1
                logger.debug(f"Added new record from overrides: {override_slug}")
        
        # Remove temporary slug column
        df = df.drop('name_slug', axis=1)
        if 'name_slug' in overrides_df.columns:
            overrides_df = overrides_df.drop('name_slug', axis=1)
        
        logger.info(f"Overrides applied: {overrides_applied} field updates, {new_records_added} new records")
        
    except Exception as e:
        logger.warning(f"Error applying overrides: {e}")
        logger.warning("Continuing without overrides")
    
    return df

def deduplicate_interconnectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate interconnector records based on key identifiers.
    
    Args:
        df: DataFrame to deduplicate
        
    Returns:
        Deduplicated DataFrame
    """
    logger = logging.getLogger(__name__)
    initial_count = len(df)
    
    # Create deduplication key
    dedup_columns = ['name', 'landing_point_gb', 'counterparty_country']
    available_dedup_cols = [col for col in dedup_columns if col in df.columns]
    
    if not available_dedup_cols:
        logger.warning("No suitable columns for deduplication found")
        return df
    
    # Create slugified deduplication key
    df['dedup_key'] = ''
    for col in available_dedup_cols:
        df['dedup_key'] += df[col].fillna('').astype(str).apply(slugify_name) + '_'
    
    # Remove trailing underscore
    df['dedup_key'] = df['dedup_key'].str.rstrip('_')
    
    # Find duplicates
    duplicates = df.duplicated(subset=['dedup_key'], keep='first')
    duplicate_count = duplicates.sum()
    
    if duplicate_count > 0:
        logger.info(f"Found {duplicate_count} duplicate records")
        # Keep first occurrence of each duplicate
        df = df[~duplicates]
    
    # Remove temporary deduplication key
    df = df.drop('dedup_key', axis=1)
    
    final_count = len(df)
    logger.info(f"Deduplication complete: {initial_count - final_count} duplicates removed")
    
    return df

def clean_and_normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize interconnector data.
    
    Args:
        df: Raw interconnector DataFrame
        
    Returns:
        Cleaned and normalized DataFrame
    """
    logger = logging.getLogger(__name__)
    
    # Trim whitespace from string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', np.nan)
    
    # Ensure numeric columns are properly typed
    numeric_columns = {
        'capacity_mw': 'float64',
        'losses_percent': 'float64', 
        'commissioning_year': 'Int64'  # Nullable integer
    }
    
    for col, dtype in numeric_columns.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
    
    # Ensure boolean columns
    if 'dc' in df.columns:
        df['dc'] = df['dc'].fillna(True).astype(bool)
    
    # Standardize status values
    if 'status' in df.columns:
        status_mapping = {
            'operational': 'Operational',
            'operation': 'Operational', 
            'commissioned': 'Operational',
            'under construction': 'Under Construction',
            'construction': 'Under Construction',
            'planning': 'Planning',
            'planned': 'Planning',
            'proposed': 'Planning'
        }
        
        # Standardize status values (handle NaN values)
        df['status'] = df['status'].fillna('Unknown')  # Fill NaN first
        df['status'] = df['status'].astype(str).str.lower().map(status_mapping).fillna(df['status'])
    
    logger.info("Data cleaning and normalization completed")
    return df

def main():
    """Main processing function."""
    logger = setup_logging("clean_interconnectors")
    start_time = time.time()
    
    try:
        logger.info("Starting interconnector data cleaning...")
        logger.info(f"Running in {'Snakemake' if SNAKEMAKE_MODE else 'standalone'} mode")
        
        if SNAKEMAKE_MODE:
            raw_file = input_raw
            overrides_file_path = overrides_file
            output_file = output_clean
        else:
            raw_file = "resources/interconnectors/interconnectors_raw.csv"
            overrides_file_path = "data/interconnectors/overrides.csv"
            output_file = "resources/interconnectors/interconnectors_clean.csv"
        
        logger.info(f"Input file: {raw_file}")
        logger.info(f"Overrides file: {overrides_file_path}")
        logger.info(f"Output file: {output_file}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load raw interconnector data
        if not Path(raw_file).exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")
        
        df = pd.read_csv(raw_file)
        logger.info(f"Loaded {len(df)} raw interconnector records")
        
        # Clean and normalize the data
        df = clean_and_normalize_data(df)
        
        # Apply overrides if available
        df = apply_overrides(df, overrides_file_path)
        
        # Deduplicate records
        df = deduplicate_interconnectors(df)
        
        # Validate the cleaned data
        df = validate_interconnector_data(df)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} cleaned interconnector records to: {output_file}")
        
        # Log summary statistics
        rows_before = len(pd.read_csv(raw_file)) if Path(raw_file).exists() else 0
        rows_after = len(df)
        
        if 'capacity_mw' in df.columns:
            total_capacity = df['capacity_mw'].sum()
            logger.info(f"Total interconnector capacity: {total_capacity:.1f} MW")
        
        if 'counterparty_country' in df.columns:
            countries = df['counterparty_country'].value_counts()
            logger.info(f"Interconnectors by country: {dict(countries)}")
        
        if 'status' in df.columns:
            statuses = df['status'].value_counts()
            logger.info(f"Interconnectors by status: {dict(statuses)}")
        
        # Execution summary
        log_execution_summary(
            logger,
            "Clean Interconnector Data",
            start_time,
            inputs={'raw_data': raw_file, 'overrides': overrides_file_path},
            outputs={'clean_data': output_file},
            context={
                'rows_before': rows_before,
                'rows_after': rows_after,
                'duplicates_removed': rows_before - rows_after,
                'total_capacity_mw': float(df['capacity_mw'].sum()) if 'capacity_mw' in df.columns else 0
            }
        )
        logger.info("Interconnector data cleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in interconnector cleaning: {e}")
        if SNAKEMAKE_MODE:
            raise
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()

