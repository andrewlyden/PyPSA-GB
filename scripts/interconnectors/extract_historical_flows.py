#!/usr/bin/env python3
"""
Extract Historical Interconnector Flows from ESPENI Data
========================================================

This script extracts actual historical interconnector flow time series
from the ESPENI database for use in historical year scenarios.

ESPENI contains half-hourly interconnector flow data from 2008-2021:
- INTELEC: ElecLink (France)
- INTEW: East-West (Ireland)
- INTFR: IFA (France)
- INTGRNL: Greenlink (Ireland)
- INTIFA2: IFA2 (France)
- INTIRL: Moyle (Northern Ireland)
- INTNED: BritNed (Netherlands)
- INTNEM: Nemo (Belgium)
- INTNSL: North Sea Link (Norway)
- INTVKL: Viking Link (Denmark)

For historical years (≤2024), these flows represent actual imports/exports
and should be used as fixed time series rather than optimized.

Author: PyPSA-GB Team
Date: October 2025
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.utilities.logging_config import setup_logging, log_execution_summary
except ImportError:
    import logging
    def setup_logging(name: str) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(name)

# Check if running in Snakemake context
if 'snakemake' in globals():
    SNAKEMAKE_MODE = True
    input_espeni = snakemake.input.espeni
    output_flows = snakemake.output.flows
    output_metadata = snakemake.output.metadata
    target_year = snakemake.params.year
else:
    SNAKEMAKE_MODE = False

# Mapping from ESPENI column names to interconnector names
# Note: ESPENI columns include [MW](float32) suffix
ESPENI_INTERCONNECTOR_MAPPING = {
    'ELEC_POWER_ELEX_INTNED[MW](float32)': 'BritNed',          # Netherlands
    'ELEC_POWER_ELEX_INTEW[MW](float32)': 'EastWest',          # Ireland (East-West)
    'ELEC_POWER_ELEX_INTIRL[MW](float32)': 'Moyle',            # Northern Ireland
    'ELEC_POWER_ELEX_INTNEM[MW](float32)': 'Nemo',             # Belgium
    'ELEC_POWER_ELEX_INTFR[MW](float32)': 'IFA',               # France (IFA)
    'ELEC_POWER_ELEX_INTIFA2[MW](float32)': 'IFA2',            # France (IFA2)
    'ELEC_POWER_ELEX_INTELEC[MW](float32)': 'ElecLink',        # France (ElecLink)
    'ELEC_POWER_ELEX_INTGRNL[MW](float32)': 'Greenlink',       # Ireland (Greenlink)
    'ELEC_POWER_ELEX_INTNSL[MW](float32)': 'NorthSeaLink',     # Norway
    'ELEC_POWER_ELEX_INTVKL[MW](float32)': 'VikingLink',       # Denmark
}

# Mapping to country codes
INTERCONNECTOR_COUNTRIES = {
    'BritNed': 'Netherlands',
    'EastWest': 'Ireland',
    'Moyle': 'Northern Ireland',
    'Nemo': 'Belgium',
    'IFA': 'France',
    'IFA2': 'France',
    'ElecLink': 'France',
    'Greenlink': 'Ireland',
    'NorthSeaLink': 'Norway',
    'VikingLink': 'Denmark',
}

# Historical commissioning dates (for filtering)
COMMISSIONING_YEARS = {
    'BritNed': 2011,
    'EastWest': 2001,
    'Moyle': 2002,
    'Nemo': 2019,
    'IFA': 1986,
    'IFA2': 2021,
    'ElecLink': 2022,  # After ESPENI data range
    'Greenlink': 2024,  # After ESPENI data range
    'NorthSeaLink': 2021,
    'VikingLink': 2023,  # After ESPENI data range
}


def load_espeni_data(espeni_file: str) -> pd.DataFrame:
    """
    Load ESPENI data with interconnector flows.
    
    Args:
        espeni_file: Path to espeni.csv
        
    Returns:
        DataFrame with datetime index and interconnector flow columns
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading ESPENI data from: {espeni_file}")
    
    if not Path(espeni_file).exists():
        raise FileNotFoundError(f"ESPENI data file not found: {espeni_file}")
    
    # Read CSV
    df = pd.read_csv(espeni_file)
    logger.info(f"Loaded {len(df)} rows from ESPENI")
    
    # Parse datetime column
    time_col = 'ELEC_elex_startTime[utc](datetime)'
    if time_col not in df.columns:
        # Try alternative column names
        time_cols = [col for col in df.columns if 'startTime' in col and 'utc' in col]
        if not time_cols:
            raise ValueError(f"Could not find datetime column in ESPENI data")
        time_col = time_cols[0]
        logger.info(f"Using datetime column: {time_col}")
    
    df['datetime'] = pd.to_datetime(df[time_col])
    df = df.set_index('datetime')
    
    logger.info(f"ESPENI data range: {df.index.min()} to {df.index.max()}")
    
    return df


def extract_interconnector_flows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Extract interconnector flows for a specific year.
    
    Args:
        df: Full ESPENI dataframe
        year: Target year to extract
        
    Returns:
        DataFrame with renamed interconnector columns and filtered to target year
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting interconnector flows for year {year}")
    
    # Filter to target year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31 23:59:59"
    
    df_year = df.loc[start_date:end_date].copy()
    
    if len(df_year) == 0:
        raise ValueError(
            f"No ESPENI data available for year {year}. "
            f"Available range: {df.index.min().year} to {df.index.max().year}"
        )
    
    logger.info(f"Filtered to {len(df_year)} records for {year}")
    logger.info(f"Date range: {df_year.index.min()} to {df_year.index.max()}")
    
    # Extract and rename interconnector columns
    ic_flows = pd.DataFrame(index=df_year.index)
    
    available_ics = []
    missing_ics = []
    
    for espeni_col, ic_name in ESPENI_INTERCONNECTOR_MAPPING.items():
        if espeni_col in df_year.columns:
            # Extract flow data (MW)
            flows = df_year[espeni_col].copy()
            
            # Check commissioning year - zero out flows before commissioning
            commission_year = COMMISSIONING_YEARS.get(ic_name)
            if commission_year and year < commission_year:
                logger.info(
                    f"  {ic_name}: Not yet commissioned in {year} "
                    f"(commissioned {commission_year}) - setting to zero"
                )
                flows = pd.Series(0.0, index=df_year.index)
            
            # Handle missing/NaN values
            nan_count = flows.isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"  {ic_name}: {nan_count} NaN values found, filling with 0"
                )
                flows = flows.fillna(0.0)
            
            # Add to output dataframe
            ic_flows[ic_name] = flows
            available_ics.append(ic_name)
            
            # Log basic statistics
            flow_mean = flows.mean()
            flow_std = flows.std()
            flow_min = flows.min()
            flow_max = flows.max()
            
            logger.info(
                f"  {ic_name}: "
                f"mean={flow_mean:.1f} MW, "
                f"std={flow_std:.1f} MW, "
                f"range=[{flow_min:.1f}, {flow_max:.1f}] MW"
            )
        else:
            missing_ics.append(ic_name)
            logger.warning(f"  {ic_name}: Column {espeni_col} not found in ESPENI data")
    
    logger.info(f"Extracted {len(available_ics)} interconnector time series")
    if missing_ics:
        logger.warning(f"Missing interconnectors: {', '.join(missing_ics)}")
    
    return ic_flows


def create_metadata(ic_flows: pd.DataFrame, year: int) -> dict:
    """
    Create metadata about the extracted flows.
    
    Args:
        ic_flows: DataFrame of interconnector flows
        year: Target year
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        'year': year,
        'source': 'ESPENI',
        'start_date': str(ic_flows.index.min()),
        'end_date': str(ic_flows.index.max()),
        'n_timesteps': len(ic_flows),
        'frequency': '30min',  # ESPENI is half-hourly
        'interconnectors': {},
    }
    
    for ic_name in ic_flows.columns:
        flows = ic_flows[ic_name]
        
        metadata['interconnectors'][ic_name] = {
            'country': INTERCONNECTOR_COUNTRIES.get(ic_name, 'Unknown'),
            'commissioning_year': COMMISSIONING_YEARS.get(ic_name),
            'mean_flow_mw': float(flows.mean()),
            'std_flow_mw': float(flows.std()),
            'min_flow_mw': float(flows.min()),
            'max_flow_mw': float(flows.max()),
            'net_import_mwh': float(flows.sum() * 0.5),  # Half-hourly to MWh
            'n_export_periods': int((flows < 0).sum()),
            'n_import_periods': int((flows > 0).sum()),
            'n_zero_periods': int((flows == 0).sum()),
            'availability': float((flows != 0).sum() / len(flows)),
        }
    
    # Calculate total net imports
    total_net_import = sum(
        meta['net_import_mwh'] 
        for meta in metadata['interconnectors'].values()
    )
    metadata['total_net_import_mwh'] = total_net_import
    metadata['total_net_import_twh'] = total_net_import / 1e6
    
    return metadata


def validate_flows(ic_flows: pd.DataFrame) -> bool:
    """
    Validate extracted interconnector flows.
    
    Args:
        ic_flows: DataFrame of interconnector flows
        
    Returns:
        True if validation passes
    """
    logger = logging.getLogger(__name__)
    
    issues = []
    
    # Check for missing data
    total_nans = ic_flows.isna().sum().sum()
    if total_nans > 0:
        issues.append(f"Found {total_nans} NaN values")
    
    # Check for suspiciously constant values
    for ic_name in ic_flows.columns:
        flows = ic_flows[ic_name]
        
        # Check if all zeros (might be pre-commissioning)
        if (flows == 0).all():
            logger.warning(f"{ic_name}: All zeros - check commissioning date")
        
        # Check for constant non-zero values (suspicious)
        elif flows.nunique() == 1:
            issues.append(f"{ic_name}: Constant value {flows.iloc[0]} - suspicious")
        
        # Check for unrealistic magnitudes
        if flows.abs().max() > 10000:
            issues.append(
                f"{ic_name}: Unrealistic max flow {flows.abs().max():.1f} MW"
            )
    
    # Check temporal coverage
    expected_periods = 365 * 48  # Half-hourly for full year
    if len(ic_flows) < expected_periods * 0.95:
        issues.append(
            f"Incomplete year: {len(ic_flows)} periods vs "
            f"expected ~{expected_periods}"
        )
    
    if issues:
        logger.warning("Flow validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("Flow validation passed")
    return True


def main():
    """Main execution function."""
    logger = setup_logging("extract_historical_interconnector_flows")
    start_time = time.time()
    
    try:
        logger.info("=== Extracting Historical Interconnector Flows ===")
        
        if SNAKEMAKE_MODE:
            espeni_file = input_espeni
            output_file = output_flows
            metadata_file = output_metadata
            year = int(target_year)
        else:
            # Standalone mode defaults
            espeni_file = "data/demand/espeni.csv"
            year = 2020
            output_file = f"resources/interconnectors/historical_flows_{year}.csv"
            metadata_file = f"resources/interconnectors/historical_flows_{year}_metadata.json"
        
        logger.info(f"Configuration:")
        logger.info(f"  ESPENI file: {espeni_file}")
        logger.info(f"  Target year: {year}")
        logger.info(f"  Output flows: {output_file}")
        logger.info(f"  Output metadata: {metadata_file}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load ESPENI data
        espeni_df = load_espeni_data(espeni_file)
        
        # Extract interconnector flows for target year
        ic_flows = extract_interconnector_flows(espeni_df, year)
        
        # Validate flows
        validate_flows(ic_flows)
        
        # Create metadata
        metadata = create_metadata(ic_flows, year)
        
        # Save outputs
        ic_flows.to_csv(output_file)
        logger.info(f"Saved interconnector flows: {output_file}")
        logger.info(f"  Shape: {ic_flows.shape}")
        logger.info(f"  Columns: {', '.join(ic_flows.columns)}")
        
        # Save metadata as JSON
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_file}")
        
        # Calculate statistics
        interconnectors = len(ic_flows.columns)
        time_periods = len(ic_flows)
        total_flow_twh = metadata['total_net_import_twh']
        
        # Log execution summary
        log_execution_summary(
            logger,
            "extract_historical_interconnector_flows",
            start_time,
            inputs={'espeni_data': espeni_file},
            outputs={'flows': output_file, 'metadata': metadata_file},
            context={
                'year': year,
                'interconnectors': interconnectors,
                'time_periods': time_periods,
                'total_flow_twh': total_flow_twh
            }
        )
        
    except Exception as e:
        logger.error(f"Error extracting historical interconnector flows: {e}")
        if SNAKEMAKE_MODE:
            raise
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()

