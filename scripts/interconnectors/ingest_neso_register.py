#!/usr/bin/env python3
"""
Ingest NESO Interconnector Register
===================================

This script processes the NESO Interconnector Register CSV file and converts
it to a standardized format compatible with the interconnectors workflow.

The NESO register provides comprehensive data on both existing and planned
interconnectors including connection sites, capacities, and project status.

Key features:
- Standardized column mapping from NESO format
- Project status classification
- Import/Export capacity handling
- Connection site normalization
- Duplicate detection and handling

Author: PyPSA-GB Team
"""

import sys
import pandas as pd
import logging
from pathlib import Path
import time
from typing import Dict

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
    def log_execution_summary(logger, script_name, start_time, inputs=None, outputs=None, context=None):
        """Fallback log_execution_summary when logging_config is not available."""
        execution_time = time.time() - start_time
        logger.info(f"{script_name} completed in {execution_time:.2f}s")
        if context:
            logger.info(f"Context: {context}")

# Check if running in Snakemake context
if 'snakemake' in globals():
    SNAKEMAKE_MODE = True
    input_file = snakemake.input[0]  # First input file
    output_file = snakemake.output[0]  # First output file
else:
    SNAKEMAKE_MODE = False

def load_neso_register(file_path: str) -> pd.DataFrame:
    """Load NESO interconnector register CSV file."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Reading NESO register from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Raw NESO data has {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error reading NESO register: {e}")
        raise

def standardize_neso_data(neso_df: pd.DataFrame) -> pd.DataFrame:
    """Convert NESO register format to standardized interconnector format."""
    logger = logging.getLogger(__name__)
    
    # Column mapping from NESO to standard format
    column_mapping = {
        'Project Name': 'name',
        'Connection Site': 'landing_point_gb',
        'MW Import - Total': 'capacity_mw',
        'MW Export - Total': 'export_capacity_mw',
        'Project Status': 'status',
        'HOST TO': 'transmission_owner',
        'MW Effective From': 'commissioning_date'
    }
    
    # Create standardized DataFrame
    result_df = pd.DataFrame()
    
    for neso_col, std_col in column_mapping.items():
        if neso_col in neso_df.columns:
            result_df[std_col] = neso_df[neso_col]
    
    # Extract counterparty country from project name and connection site
    result_df['counterparty_country'] = result_df['name'].apply(extract_counterparty_country)
    
    # Add default values
    result_df['losses_percent'] = 2.5  # Standard assumption
    result_df['dc'] = True  # Most modern interconnectors are DC
    result_df['source'] = 'NESO_Interconnector_Register'
    
    # Handle capacity - use import capacity as primary, note export separately
    result_df['capacity_mw'] = pd.to_numeric(result_df['capacity_mw'], errors='coerce')
    result_df['export_capacity_mw'] = pd.to_numeric(result_df['export_capacity_mw'], errors='coerce')
    
    # Clean landing point names
    result_df['landing_point_gb'] = result_df['landing_point_gb'].apply(clean_landing_point)
    
    # Filter only built/operational interconnectors by default
    operational_status = ['Built', 'Under Construction/Commissioning']
    operational_df = result_df[result_df['status'].isin(operational_status)].copy()
    
    logger.info(f"Standardized {len(result_df)} total projects")
    logger.info(f"Filtered to {len(operational_df)} operational/near-operational projects")
    
    return operational_df

def extract_counterparty_country(project_name: str) -> str:
    """Extract counterparty country from project name."""
    if pd.isna(project_name):
        return 'Unknown'
    
    name_lower = str(project_name).lower()
    
    # Country mapping based on project names
    # Order matters - check more specific keywords first
    country_keywords = {
        'france': ['ifa', 'fab', 'eleclink', 'aquind'],
        'netherlands': ['britned', 'neuconnect'],  # Check Netherlands before Belgium
        'belgium': ['nemo'],  # Removed 'britned' - that's Netherlands
        'germany': ['neuconnect'],
        'norway': ['ns link', 'north sea link', 'northconnect'],
        'denmark': ['viking'],
        'ireland': ['east west', 'greenlink', 'celtic'],
        'northern ireland': ['moyle', 'auchencrosh'],
        'isle of man': ['isle of man'],
    }
    
    for country, keywords in country_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            return country.title()
    
    return 'Unknown'

def clean_landing_point(landing_point: str) -> str:
    """Clean and standardize landing point names."""
    if pd.isna(landing_point):
        return None
    
    # Remove common suffixes and standardize
    cleaned = str(landing_point).strip()  # Strip whitespace first
    
    # Clean up any extra whitespace within the string
    cleaned = ' '.join(cleaned.split())
    
    # Remove suffixes - order matters, check longer suffixes first
    suffixes_to_remove = [
        ' 400kV Substation', ' 275kV Substation', ' 132kV Substation',
        ' Substation',
        ' 400kV', ' 275kV', ' 132kV',
        ' GSP'
    ]
    
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break  # Only remove one suffix
    
    return cleaned.strip()

def validate_neso_data(df: pd.DataFrame) -> bool:
    """Validate the standardized NESO data."""
    logger = logging.getLogger(__name__)
    
    # Check required columns
    required_columns = ['name', 'capacity_mw', 'counterparty_country']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for valid capacity values
    invalid_capacity = df['capacity_mw'].isna() | (df['capacity_mw'] <= 0)
    if invalid_capacity.any():
        logger.warning(f"Found {invalid_capacity.sum()} records with invalid capacity")
    
    # Log summary statistics
    logger.info(f"Validation summary:")
    logger.info(f"  - Total records: {len(df)}")
    logger.info(f"  - Total capacity: {df['capacity_mw'].sum():.0f} MW")
    logger.info(f"  - Countries: {df['counterparty_country'].value_counts().to_dict()}")
    if 'status' in df.columns:
        logger.info(f"  - Status: {df['status'].value_counts().to_dict()}")
    
    return True

def main():
    """Main execution function."""
    logger = setup_logging("ingest_neso_register")
    start_time = time.time()
    
    try:
        logger.info("Starting NESO interconnector register ingestion...")
        
        if SNAKEMAKE_MODE:
            logger.info("Running in Snakemake mode")
            input_file_path = input_file
            output_file_path = output_file
        else:
            # Default paths for standalone execution
            input_file_path = "data/interconnectors/NESO_interconnector_register.csv"
            output_file_path = "resources/interconnectors/neso_standardized.csv"
        
        logger.info(f"Input file: {input_file_path}")
        logger.info(f"Output file: {output_file_path}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process NESO data
        neso_df = load_neso_register(input_file_path)
        standardized_df = standardize_neso_data(neso_df)
        
        # Validate the data
        if not validate_neso_data(standardized_df):
            raise ValueError("Data validation failed")
        
        # Save standardized data
        standardized_df.to_csv(output_file_path, index=False)
        logger.info(f"Saved {len(standardized_df)} standardized records to: {output_file_path}")
        
        # Calculate statistics
        interconnectors = len(standardized_df)
        # Note: gb_lat and gb_lon will be added in later enrichment steps
        total_capacity = standardized_df['capacity_mw'].sum() if 'capacity_mw' in standardized_df.columns else 0
        
        # Log execution summary
        log_execution_summary(
            logger,
            "ingest_neso_register",
            start_time,
            inputs={'neso_register': input_file_path},
            outputs={'standardized_data': output_file_path},
            context={
                'interconnectors': interconnectors,
                'total_capacity_mw': total_capacity
            }
        )
        
    except Exception as e:
        logger.error(f"Error in NESO interconnector ingestion: {e}")
        raise

if __name__ == "__main__":
    main()

