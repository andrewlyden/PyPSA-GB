#!/usr/bin/env python3
"""
Ingest DUKES Table 5.13 Interconnector Data
===========================================

This script robustly parses DUKES table 5.13 to extract interconnector data.
It handles dynamic Excel sheet layouts and column names that may change
between DUKES editions.

Key features:
- Automatic sheet detection using fuzzy matching
- Dynamic column mapping with fallback alternatives
- Robust data type inference and cleaning
- Comprehensive logging and error handling
- Standardized output schema

Author: PyPSA-GB Team
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.utilities.logging_config import setup_logging, log_execution_summary
    import logging
except ImportError:
    # Fallback logging setup if import fails
    import logging
    def setup_logging(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)
    def log_execution_summary(*args, **kwargs):
        pass  # No-op fallback

# Check if running in Snakemake context
if 'snakemake' in globals():
    SNAKEMAKE_MODE = True
    input_file = snakemake.input[0]  # First input file
    output_file = snakemake.output[0]  # First output file
else:
    SNAKEMAKE_MODE = False

def detect_sheet_with_interconnectors(excel_file_path: str) -> str:
    """
    Detect the sheet containing interconnector data using fuzzy matching.
    
    Args:
        excel_file_path: Path to the Excel file
        
    Returns:
        str: Name of the sheet containing interconnector data
        
    Raises:
        ValueError: If no suitable sheet is found
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Read sheet names
        excel_file = pd.ExcelFile(excel_file_path)
        sheet_names = excel_file.sheet_names
        logger.info(f"Found {len(sheet_names)} sheets in Excel file")
        
        # Keywords to look for in sheet names (prioritize specific variants)
        keywords = ['5.13a', '5_13a', '5.13', '5_13', 'interconnector', 'interconnection', 'hvdc', 'link']
        
        # Score each sheet name
        best_sheet = None
        best_score = 0
        
        for sheet_name in sheet_names:
            score = 0
            sheet_lower = sheet_name.lower()
            
            # Direct keyword matches
            for keyword in keywords:
                if keyword in sheet_lower:
                    score += 10
            
            # Table number patterns (prioritize 'A' variants)
            if re.search(r'5\.13a|5_13a', sheet_name, re.IGNORECASE):
                score += 30  # Higher score for 'A' variants
            elif re.search(r'5\.13|5_13', sheet_name):
                score += 20
                
            # General interconnector references
            if any(word in sheet_lower for word in ['connect', 'cross', 'border', 'cable']):
                score += 5
                
            logger.debug(f"Sheet '{sheet_name}' scored {score}")
            
            if score > best_score:
                best_score = score
                best_sheet = sheet_name
        
        if best_sheet is None or best_score < 5:
            # Fallback: try first sheet or look for sheets with numeric data
            logger.warning("No clear interconnector sheet found, using fallback detection")
            for sheet_name in sheet_names:
                try:
                    df_test = pd.read_excel(excel_file_path, sheet_name=sheet_name, nrows=10)
                    if len(df_test.columns) > 3 and any('capacity' in str(col).lower() for col in df_test.columns):
                        best_sheet = sheet_name
                        break
                except:
                    continue
        
        if best_sheet is None:
            raise ValueError(f"Could not detect interconnector sheet in {excel_file_path}")
            
        logger.info(f"Selected sheet: '{best_sheet}' (score: {best_score})")
        return best_sheet
        
    except Exception as e:
        logger.error(f"Error detecting sheet: {e}")
        raise

def map_columns_dynamically(df: pd.DataFrame) -> Dict[str, str]:
    """
    Dynamically map DataFrame columns to standardized schema.
    
    Args:
        df: DataFrame with original column names
        
    Returns:
        Dict mapping original column names to standard names
    """
    logger = logging.getLogger(__name__)
    
    # Get the actual column names
    actual_columns = [str(col) for col in df.columns]
    logger.info(f"DataFrame columns: {actual_columns}")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info("Sample data:")
    logger.info(f"{df.head()}")
    
    # Standard schema mapping for DUKES 5.13A format
    schema_mapping = {
        'name': ['interconnector name', 'name', 'interconnector', 'connection', 'link'],
        'counterparty_country': ['connecting country', 'country', 'counterparty', 'partner'],
        'capacity_mw': ['capacity (mw)', 'capacity', 'mw', 'power', 'rating'],
        'commissioning_year': ['year commissioned', 'commission', 'year', 'operational'],
        'notes': ['notes', 'note', 'comments', 'remarks']
    }
    
    # Find best matches for each schema field
    column_mapping = {}
    available_columns = [str(col).lower() for col in df.columns]
    
    for standard_name, keywords in schema_mapping.items():
        best_match = None
        best_score = 0
        
        for original_col in df.columns:
            original_lower = str(original_col).lower().strip()
            score = 0
            
            # Check for exact matches first
            if original_lower in [kw.lower() for kw in keywords]:
                score = 100
            else:
                # Partial matches
                for keyword in keywords:
                    if keyword.lower() in original_lower:
                        score += 50
                    elif any(word in original_lower for word in keyword.lower().split()):
                        score += 25
            
            # Bonus for exact matches
            if original_lower in [kw.lower() for kw in keywords]:
                score += 50
                
            if score > best_score:
                best_score = score
                best_match = original_col
        
        if best_match and best_score > 25:
            column_mapping[best_match] = standard_name
            logger.info(f"Mapped '{best_match}' -> '{standard_name}' (score: {best_score})")
    
    logger.info(f"Successfully mapped {len(column_mapping)} columns")
    return column_mapping

def extract_capacity_values(df: pd.DataFrame) -> pd.Series:
    """
    Extract capacity values, handling various formats and units.
    
    Args:
        df: DataFrame with capacity columns
        
    Returns:
        Series with standardized capacity values in MW
    """
    logger = logging.getLogger(__name__)
    
    # Look for capacity columns
    capacity_cols = [col for col in df.columns if 'capacity' in str(col).lower() or 'mw' in str(col).lower()]
    
    if not capacity_cols:
        logger.warning("No capacity columns found")
        return pd.Series(np.nan, index=df.index)
    
    # Use the first capacity column found
    capacity_col = capacity_cols[0]
    logger.info(f"Using capacity column: '{capacity_col}'")
    
    capacity_series = df[capacity_col].copy()
    
    # Clean and convert to numeric
    if capacity_series.dtype == 'object':
        # Remove non-numeric characters except decimal points
        capacity_series = capacity_series.astype(str).str.replace(r'[^\d\.]', '', regex=True)
        capacity_series = pd.to_numeric(capacity_series, errors='coerce')
    
    # Handle potential unit conversions (if values are very small, might be in GW)
    if capacity_series.notna().any():
        median_value = capacity_series.median()
        if median_value < 10:  # Likely in GW, convert to MW
            capacity_series = capacity_series * 1000
            logger.info("Converted capacity from GW to MW")
    
    logger.info(f"Extracted capacity for {capacity_series.notna().sum()} interconnectors")
    return capacity_series

def clean_and_standardize_data(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Clean and standardize the interconnector data.
    
    Args:
        df: Raw DataFrame
        column_mapping: Mapping from original to standard column names
        
    Returns:
        Cleaned and standardized DataFrame
    """
    logger = logging.getLogger(__name__)
    
    # Create output DataFrame with standard schema
    output_columns = [
        'name', 'landing_point_gb', 'counterparty_country', 'counterparty_landing_point',
        'capacity_mw', 'dc', 'losses_percent', 'commissioning_year', 'status', 'source'
    ]
    
    result_df = pd.DataFrame(columns=output_columns)
    
    # Map existing columns
    for original_col, standard_name in column_mapping.items():
        if standard_name in output_columns:
            result_df[standard_name] = df[original_col].copy()
    
    # Handle capacity specially
    result_df['capacity_mw'] = extract_capacity_values(df)
    
    # Set defaults for missing columns
    if 'dc' not in result_df.columns or result_df['dc'].isna().all():
        result_df['dc'] = True  # DUKES interconnectors are typically DC
    
    if 'losses_percent' not in result_df.columns or result_df['losses_percent'].isna().all():
        result_df['losses_percent'] = 2.5  # Default 2.5% losses
    
    # Add source information
    result_df['source'] = 'DUKES_5.13_2025.xlsx'
    
    # Clean text fields
    text_columns = ['name', 'landing_point_gb', 'counterparty_country', 'counterparty_landing_point', 'status']
    for col in text_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].astype(str).str.strip()
            result_df[col] = result_df[col].replace('nan', np.nan).infer_objects(copy=False)
    
    # Clean numeric fields
    if 'commissioning_year' in result_df.columns:
        result_df['commissioning_year'] = pd.to_numeric(result_df['commissioning_year'], errors='coerce')
    
    # Remove completely empty rows
    result_df = result_df.dropna(how='all')
    
    # Remove rows without names or capacity
    valid_rows = result_df['name'].notna() & result_df['capacity_mw'].notna()
    result_df = result_df[valid_rows]
    
    logger.info(f"Cleaned data: {len(result_df)} valid interconnector records")
    return result_df


def find_data_table(df):
    """
    Find the actual data table in the DUKES 5.13A worksheet.
    
    Args:
        df: DataFrame with raw worksheet data (no headers)
        
    Returns:
        tuple: (header_row_idx, data_start_idx) or (None, None) if not found
    """
    logger = logging.getLogger(__name__)
    logger.info("Searching for data table headers in DUKES worksheet...")
    
    # Look for the specific header pattern from DUKES 5.13A
    for idx, row in df.iterrows():
        # Convert row to string and check for key column headers
        row_values = [str(cell) for cell in row if pd.notna(cell)]
        row_str = ' '.join(row_values).lower()
        
        # Look for the specific pattern from the real DUKES file
        if ('interconnector name' in row_str and 
            'connecting country' in row_str and 
            'capacity' in row_str):
            logger.info(f"Found header row at index {idx}")
            logger.info(f"Header values: {row_values}")
            return idx, idx + 1
            
        # Also check for individual column patterns in case of variations
        row_cells = [str(cell).lower() for cell in row if pd.notna(cell)]
        if (any('interconnector' in cell for cell in row_cells) and
            any('capacity' in cell for cell in row_cells) and
            any('country' in cell for cell in row_cells)):
            logger.info(f"Found header row at index {idx} (pattern match)")
            logger.info(f"Header values: {row_values}")
            return idx, idx + 1
    
    logger.warning("Could not find header row with expected patterns")
    return None, None


def main():
    """Main processing function."""
    logger = setup_logging("ingest_interconnectors_dukes")
    start_time = time.time()
    
    try:
        logger.info("Starting DUKES interconnector data ingestion...")
        logger.info(f"Running in {'Snakemake' if SNAKEMAKE_MODE else 'standalone'} mode")
        
        if SNAKEMAKE_MODE:
            input_file_path = input_file
            output_file_path = output_file
        else:
            # Standalone mode - use default paths
            input_file_path = "data/interconnectors/DUKES_5.13_2025.xlsx"
            output_file_path = "resources/interconnectors/interconnectors_raw.csv"
        
        logger.info(f"Input file: {input_file_path}")
        logger.info(f"Output file: {output_file_path}")
        
        # Ensure output directory exists
        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if input file exists and determine format
        if not Path(input_file_path).exists():
            # Try CSV alternative if Excel file not found
            csv_alternative = input_file_path.replace('.xlsx', '.csv')
            if Path(csv_alternative).exists():
                logger.info(f"Excel file not found, using CSV alternative: {csv_alternative}")
                input_file_path = csv_alternative
            else:
                raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
        # Read the data based on file type
        file_extension = Path(input_file_path).suffix.lower()
        
        if file_extension == '.csv':
            logger.info("Reading CSV file directly")
            df = pd.read_csv(input_file_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from CSV")
        else:
            # Detect and read the appropriate sheet for Excel files
            sheet_name = detect_sheet_with_interconnectors(input_file_path)
            logger.info(f"Reading Excel sheet: '{sheet_name}'")
            
            # Read the entire sheet first to find the data table
            df_raw = pd.read_excel(input_file_path, sheet_name=sheet_name, header=None)
            logger.info(f"Raw sheet has {len(df_raw)} rows and {len(df_raw.columns)} columns")
            
            # Find the actual data table within the sheet
            header_row, data_start_row = find_data_table(df_raw)
            
            if header_row is not None:
                logger.info(f"Found data table starting at row {data_start_row} with headers at row {header_row}")
                # Re-read with proper header - don't use skiprows when using header parameter
                df = pd.read_excel(input_file_path, sheet_name=sheet_name, header=header_row)
                # Remove any completely empty rows
                df = df.dropna(how='all')
                logger.info(f"Extracted {len(df)} data rows after cleaning")
            else:
                logger.warning("Could not find data table, using entire sheet")
                df = pd.read_excel(input_file_path, sheet_name=sheet_name)
                
            logger.info(f"Final DataFrame: {len(df)} rows and {len(df.columns)} columns")
        
        # Map columns dynamically
        column_mapping = map_columns_dynamically(df)
        
        if not column_mapping:
            logger.warning("No column mappings found - using fallback approach")
            # Create a basic mapping using the first few columns
            df_clean = df.copy()
            df_clean.columns = [f"col_{i}" for i in range(len(df.columns))]
        else:
            # Clean and standardize the data
            df_clean = clean_and_standardize_data(df, column_mapping)
        
        # Save the raw interconnector data
        df_clean.to_csv(output_file_path, index=False)
        logger.info(f"Saved {len(df_clean)} interconnector records to: {output_file_path}")
        
        # Calculate statistics
        interconnectors_found = len(df_clean)
        total_capacity = df_clean['capacity_mw'].sum() if 'capacity_mw' in df_clean.columns else 0
        
        # Log execution summary
        log_execution_summary(
            logger,
            "ingest_dukes",
            start_time,
            inputs={'dukes_file': input_file_path},
            outputs={'raw_interconnectors': output_file_path},
            context={
                'interconnectors_found': interconnectors_found,
                'total_capacity_mw': total_capacity,
                'file_format': Path(input_file_path).suffix
            }
        )
        
    except Exception as e:
        logger.error(f"Error in DUKES ingestion: {e}")
        if SNAKEMAKE_MODE:
            raise
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()

