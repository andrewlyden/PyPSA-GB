#!/usr/bin/env python3
"""
Combine DUKES and NESO Interconnector Data
==========================================

This script combines interconnector data from both DUKES 5.13 and NESO
Interconnector Register to create a comprehensive dataset. It handles
deduplication, data reconciliation, and creates a unified view.

Key features:
- Intelligent duplicate detection and merging
- Data source prioritization (NESO for operational status, DUKES for historical)
- Capacity reconciliation between sources
- Comprehensive data validation
- Missing data imputation where possible

Author: PyPSA-GB Team
"""

import sys
import pandas as pd
import logging
from pathlib import Path
import time
from typing import Dict, Tuple

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
    dukes_file = snakemake.input[0]  # First input (DUKES)
    neso_file = snakemake.input[1]  # Second input (NESO)
    output_file = snakemake.output[0]  # Output file
else:
    SNAKEMAKE_MODE = False

def normalize_name(name: str) -> str:
    """Normalize interconnector names for comparison."""
    if pd.isna(name):
        return ""
    
    name_clean = str(name).lower().strip()
    
    # Common normalizations (check longer patterns first)
    # Maps various name variants to a canonical form
    normalizations = {
        # IFA variants
        'ifa2 interconnector': 'ifa2',
        'ifa interconnector': 'ifa',
        'ifa2': 'ifa2',
        # NEMO variants (DUKES uses NEMO, NESO uses Nemo Link)
        'nemo link': 'nemo',
        'nemo': 'nemo',
        # NSL variants (DUKES uses NSL, NESO uses NS Link/North Sea Link)
        'ns link': 'nsl',
        'north sea link': 'nsl',
        'nsl': 'nsl',
        # BritNed
        'britned': 'britned',
        # East West
        'east west interconnector': 'east-west',
        # Viking Link
        'viking link denmark interconnector': 'viking',
        'viking link': 'viking',
        'viking': 'viking',
        # Moyle / Auchencrosh
        'moyle': 'moyle',
        'auchencrosh (interconnector cct)': 'moyle',
        'auchencrosh': 'moyle',
        # Others
        'greenlink': 'greenlink',
        'eleclink': 'eleclink'
    }
    
    for original, normalized in normalizations.items():
        if original in name_clean:
            return normalized
    
    return name_clean

def find_matching_interconnectors(dukes_df: pd.DataFrame, neso_df: pd.DataFrame) -> pd.DataFrame:
    """Find matching interconnectors between DUKES and NESO datasets."""
    logger = logging.getLogger(__name__)
    
    # Normalize names for matching
    dukes_df['name_normalized'] = dukes_df['name'].apply(normalize_name)
    neso_df['name_normalized'] = neso_df['name'].apply(normalize_name)
    
    # Find exact matches
    matches = []
    
    for _, dukes_row in dukes_df.iterrows():
        dukes_name = dukes_row['name_normalized']
        
        # Look for exact name match
        neso_matches = neso_df[neso_df['name_normalized'] == dukes_name]
        
        if len(neso_matches) == 1:
            matches.append({
                'dukes_idx': dukes_row.name,
                'neso_idx': neso_matches.index[0],
                'dukes_name': dukes_row['name'],
                'neso_name': neso_matches.iloc[0]['name'],
                'match_type': 'exact_name'
            })
        elif len(neso_matches) > 1:
            logger.warning(f"Multiple NESO matches for DUKES '{dukes_row['name']}': {list(neso_matches['name'])}")
            # Take the first match
            matches.append({
                'dukes_idx': dukes_row.name,
                'neso_idx': neso_matches.index[0],
                'dukes_name': dukes_row['name'],
                'neso_name': neso_matches.iloc[0]['name'],
                'match_type': 'exact_name_multiple'
            })
    
    matches_df = pd.DataFrame(matches)
    logger.info(f"Found {len(matches_df)} matches between DUKES and NESO")
    
    return matches_df

def merge_interconnector_data(dukes_row: pd.Series, neso_row: pd.Series) -> pd.Series:
    """Merge data from DUKES and NESO for a single interconnector."""
    
    # Start with NESO data as base (more detailed)
    merged = neso_row.copy()
    
    # Prefer DUKES name if it's a recognized interconnector name (simpler/cleaner)
    # DUKES uses standard names like "Moyle", "IFA" while NESO uses technical names
    # like "Auchencrosh (interconnector CCT)"
    dukes_name = str(dukes_row.get('name', ''))
    neso_name = str(neso_row.get('name', ''))
    
    # Prefer DUKES name if NESO name contains technical suffixes
    technical_suffixes = ['(interconnector cct)', '(converter station)', 'interconnector']
    neso_name_lower = neso_name.lower()
    has_technical_suffix = any(suffix in neso_name_lower for suffix in technical_suffixes)
    
    if has_technical_suffix and len(dukes_name) > 0 and len(dukes_name) < 30:
        merged['name'] = dukes_name
    elif len(dukes_name) > len(neso_name):
        merged['name'] = dukes_name
    
    # Capacity reconciliation - prefer DUKES historical data if available
    if pd.notna(dukes_row.get('capacity_mw')):
        dukes_capacity = float(dukes_row['capacity_mw'])
        neso_capacity = float(neso_row.get('capacity_mw', 0))
        
        # Use DUKES capacity if significantly different (might be more accurate historical)
        if abs(dukes_capacity - neso_capacity) > 50:  # More than 50MW difference
            merged['capacity_mw'] = dukes_capacity
            merged['capacity_source'] = 'DUKES_reconciled'
        else:
            merged['capacity_source'] = 'NESO'
    
    # Use DUKES commissioning year if available
    if pd.notna(dukes_row.get('commissioning_year')):
        merged['commissioning_year'] = dukes_row['commissioning_year']
    
    # Use DUKES counterparty if more specific
    if pd.notna(dukes_row.get('counterparty_country')):
        dukes_country = str(dukes_row['counterparty_country'])
        if dukes_country != 'Unknown' and dukes_country.lower() != 'unknown':
            merged['counterparty_country'] = dukes_country
    
    # Mark as combined source
    merged['source'] = 'DUKES+NESO_combined'
    merged['data_sources'] = f"DUKES:{dukes_row.get('source', 'Unknown')};NESO:{neso_row.get('source', 'Unknown')}"
    
    return merged

def combine_datasets(dukes_df: pd.DataFrame, neso_df: pd.DataFrame) -> pd.DataFrame:
    """Combine DUKES and NESO datasets with intelligent merging."""
    logger = logging.getLogger(__name__)
    
    # Find matches between datasets
    matches_df = find_matching_interconnectors(dukes_df, neso_df)
    
    combined_records = []
    used_dukes_indices = set()
    used_neso_indices = set()
    
    # Process matched interconnectors
    for _, match in matches_df.iterrows():
        dukes_idx = match['dukes_idx']
        neso_idx = match['neso_idx']
        
        dukes_row = dukes_df.loc[dukes_idx]
        neso_row = neso_df.loc[neso_idx]
        
        merged_row = merge_interconnector_data(dukes_row, neso_row)
        combined_records.append(merged_row)
        
        used_dukes_indices.add(dukes_idx)
        used_neso_indices.add(neso_idx)
        
        logger.debug(f"Merged: {match['dukes_name']} + {match['neso_name']}")
    
    # Add unmatched DUKES records
    unmatched_dukes = dukes_df[~dukes_df.index.isin(used_dukes_indices)]
    for _, row in unmatched_dukes.iterrows():
        row_copy = row.copy()
        row_copy['source'] = 'DUKES_only'
        row_copy['data_sources'] = f"DUKES:{row.get('source', 'Unknown')}"
        combined_records.append(row_copy)
        logger.debug(f"Added DUKES-only: {row['name']}")
    
    # Add unmatched NESO records
    unmatched_neso = neso_df[~neso_df.index.isin(used_neso_indices)]
    for _, row in unmatched_neso.iterrows():
        row_copy = row.copy()
        row_copy['source'] = 'NESO_only'
        row_copy['data_sources'] = f"NESO:{row.get('source', 'Unknown')}"
        combined_records.append(row_copy)
        logger.debug(f"Added NESO-only: {row['name']}")
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(combined_records)
    
    # Clean up temporary columns
    if 'name_normalized' in combined_df.columns:
        combined_df = combined_df.drop('name_normalized', axis=1)
    
    logger.info(f"Combined dataset summary:")
    logger.info(f"  - Total interconnectors: {len(combined_df)}")
    logger.info(f"  - Matched records: {len(matches_df)}")
    logger.info(f"  - DUKES-only records: {len(unmatched_dukes)}")
    logger.info(f"  - NESO-only records: {len(unmatched_neso)}")
    logger.info(f"  - Total capacity: {combined_df['capacity_mw'].sum():.0f} MW")
    
    return combined_df

def validate_combined_data(df: pd.DataFrame) -> bool:
    """Validate the combined interconnector dataset."""
    logger = logging.getLogger(__name__)
    
    # Check for duplicates
    duplicate_names = df[df['name'].duplicated(keep=False)]
    if len(duplicate_names) > 0:
        logger.warning(f"Found {len(duplicate_names)} potential duplicate names:")
        for name in duplicate_names['name'].unique():
            logger.warning(f"  - {name}")
    
    # Capacity validation
    invalid_capacity = df['capacity_mw'].isna() | (df['capacity_mw'] <= 0)
    if invalid_capacity.any():
        logger.warning(f"Found {invalid_capacity.sum()} records with invalid capacity")
    
    # Log final statistics
    logger.info(f"Final dataset statistics:")
    logger.info(f"  - Total interconnectors: {len(df)}")
    logger.info(f"  - Total capacity: {df['capacity_mw'].sum():.0f} MW")
    logger.info(f"  - Data sources: {df['source'].value_counts().to_dict()}")
    if 'counterparty_country' in df.columns:
        logger.info(f"  - Countries: {df['counterparty_country'].value_counts().to_dict()}")
    
    return True

def main():
    """Main execution function."""
    logger = setup_logging("combine_interconnectors")
    start_time = time.time()
    
    try:
        logger.info("Starting interconnector data combination...")
        
        if SNAKEMAKE_MODE:
            logger.info("Running in Snakemake mode")
            dukes_file_path = dukes_file
            neso_file_path = neso_file
            output_file_path = output_file
        else:
            # Default paths for standalone execution
            dukes_file_path = "resources/interconnectors/interconnectors_clean.csv"
            neso_file_path = "resources/interconnectors/neso_standardized.csv"
            output_file_path = "resources/interconnectors/interconnectors_combined.csv"
        
        logger.info(f"DUKES file: {dukes_file_path}")
        logger.info(f"NESO file: {neso_file_path}")
        logger.info(f"Output file: {output_file_path}")
        
        # Load datasets
        logger.info("Loading DUKES data...")
        dukes_df = pd.read_csv(dukes_file_path)
        logger.info(f"Loaded {len(dukes_df)} DUKES records")
        
        logger.info("Loading NESO data...")
        neso_df = pd.read_csv(neso_file_path)
        logger.info(f"Loaded {len(neso_df)} NESO records")
        
        # Combine datasets
        combined_df = combine_datasets(dukes_df, neso_df)
        
        # Validate combined data
        if not validate_combined_data(combined_df):
            raise ValueError("Combined data validation failed")
        
        # Create output directory if needed
        output_dir = Path(output_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined data
        combined_df.to_csv(output_file_path, index=False)
        logger.info(f"Saved {len(combined_df)} combined records to: {output_file_path}")
        
        # Calculate statistics
        dukes_entries = len(dukes_df)
        neso_entries = len(neso_df)
        combined_entries = len(combined_df)
        total_capacity = combined_df['capacity_mw'].sum()
        
        # Log execution summary
        log_execution_summary(
            logger,
            "combine_interconnectors",
            start_time,
            inputs={'dukes_data': dukes_file_path, 'neso_data': neso_file_path},
            outputs={'combined_data': output_file_path},
            context={
                'dukes_entries': dukes_entries,
                'neso_entries': neso_entries,
                'combined_entries': combined_entries,
                'deduplication_removed': (dukes_entries + neso_entries - combined_entries),
                'total_capacity_mw': total_capacity
            }
        )
        
    except Exception as e:
        logger.error(f"Error in interconnector combination: {e}")
        raise

if __name__ == "__main__":
    main()

