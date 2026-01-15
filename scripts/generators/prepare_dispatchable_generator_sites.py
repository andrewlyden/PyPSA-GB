"""
Prepare Dispatchable Generator Site Data for PyPSA-GB

This script combines TEC register and REPD data to create comprehensive site CSV files
for all dispatchable generators in Great Britain. It categorizes generators based on
their operational characteristics and fuel types.

Dispatchable Generator Categories:
=================================

1. Thermal Generators (TEC Register):
   - CCGT (Combined Cycle Gas Turbine)
   - OCGT (Open Cycle Gas Turbine) 
   - Nuclear
   - Coal
   - Gas Reciprocating
   - CHP (Combined Heat and Power)
   - Oil & AGT

2. Storage Systems:
   - Battery (REPD + TEC)
   - Pumped Storage Hydroelectricity (REPD + TEC)
   - Energy Storage System (TEC - generic)
   - Hydrogen (REPD)

3. Biomass/Waste (REPD):
   - Biomass (dedicated)
   - EfW Incineration
   - Anaerobic Digestion
   - Landfill Gas
   - Sewage Sludge Digestion
   - Advanced Conversion Technologies

4. Large Hydro (REPD):
   - Large reservoir-based hydro plants (>20MW typically)

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

# Configure logging
logger = None
try:
    from logging_config import setup_logging, log_dataframe_info
    logger = setup_logging("prepare_dispatchable_generator_sites")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_dataframe_info
        logger = setup_logging("prepare_dispatchable_generator_sites")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("prepare_dispatchable_generator_sites")
        def log_dataframe_info(df, logger, name: str):
            logger.info(f"{name} shape: {df.shape}")

# Define dispatchable technology mappings
THERMAL_GENERATORS = {
    'CCGT (Combined Cycle Gas Turbine)': 'ccgt',
    'OCGT (Open Cycle Gas Turbine)': 'ocgt',
    'Nuclear': 'nuclear',
    'Coal': 'coal',
    'Gas Reciprocating': 'gas_reciprocating',
    'CHP (Combined Heat and Power)': 'chp',
    'Oil & AGT (Advanced Gas Turbine)': 'oil_gas_turbine',
    'Thermal': 'thermal_other',
    'Waste': 'waste_thermal'
}

STORAGE_SYSTEMS = {
    'Energy Storage System': 'battery_generic',
    'Battery': 'battery',
    'Pumped Storage Hydroelectricity': 'pumped_hydro',
    'Pump Storage': 'pumped_hydro',
    'Hydrogen': 'hydrogen_storage'
}

BIOMASS_WASTE = {
    'Biomass (dedicated)': 'biomass',
    'EfW Incineration': 'waste_to_energy',
    'Anaerobic Digestion': 'biogas',
    'Landfill Gas': 'landfill_gas',
    'Sewage Sludge Digestion': 'sewage_gas',
    'Advanced Conversion Technologies': 'advanced_biofuel'
}

GEOTHERMAL = {
    'Geothermal': 'geothermal',
    'Hot Dry Rocks (HDR)': 'geothermal'
}

LARGE_HYDRO = {
    'Large Hydro': 'large_hydro'
}

# Special handling for mixed plant types in TEC
MIXED_PLANT_PATTERNS = {
    'CCGT.*Energy Storage': 'ccgt_battery',
    'CCGT.*OCGT': 'ccgt_ocgt',
    'Energy Storage.*Gas': 'battery_gas',
    'Energy Storage.*Pump Storage': 'battery_pumped',
    'Demand.*Energy Storage': 'demand_response_storage'
}

def normalize_technology_name(tech_name: str) -> str:
    """Normalize technology name for consistent mapping."""
    if pd.isna(tech_name):
        return "unknown"
    return str(tech_name).strip()

def categorize_tec_technology(plant_type: str) -> Tuple[str, str]:
    """
    Categorize TEC plant type into technology category and specific type.
    
    Args:
        plant_type: Plant Type from TEC register
        
    Returns:
        Tuple of (category, technology)
    """
    if pd.isna(plant_type):
        return "unknown", "unknown"
    
    plant_type = normalize_technology_name(plant_type)
    
    # Check for mixed plant types first
    for pattern, tech in MIXED_PLANT_PATTERNS.items():
        if re.search(pattern, plant_type, re.IGNORECASE):
            return "hybrid", tech
    
    # Check exact matches for thermal generators
    if plant_type in THERMAL_GENERATORS:
        return "thermal", THERMAL_GENERATORS[plant_type]
    
    # Check exact matches for storage systems
    if plant_type in STORAGE_SYSTEMS:
        return "storage", STORAGE_SYSTEMS[plant_type]
    
    # Check for partial matches (e.g., plant types with additional descriptors)
    for thermal_type, tech in THERMAL_GENERATORS.items():
        if thermal_type.lower() in plant_type.lower():
            return "thermal", tech
    
    for storage_type, tech in STORAGE_SYSTEMS.items():
        if storage_type.lower() in plant_type.lower():
            return "storage", tech
    
    # Handle special cases
    if "interconnector" in plant_type.lower():
        return "interconnector", "interconnector"
    
    if "demand" in plant_type.lower() and "storage" not in plant_type.lower():
        return "demand_response", "demand_response"
    
    if "reactive" in plant_type.lower():
        return "reactive_power", "reactive_compensation"
    
    return "other", "unclassified"

def categorize_repd_technology(tech_type: str) -> Tuple[str, str]:
    """
    Categorize REPD technology type into category and specific type.
    
    Args:
        tech_type: Technology Type from REPD
        
    Returns:
        Tuple of (category, technology)
    """
    if pd.isna(tech_type):
        return "unknown", "unknown"
    
    tech_type = normalize_technology_name(tech_type)
    
    # Check exact matches
    if tech_type in STORAGE_SYSTEMS:
        return "storage", STORAGE_SYSTEMS[tech_type]
    
    if tech_type in BIOMASS_WASTE:
        return "biomass_waste", BIOMASS_WASTE[tech_type]
    
    if tech_type in GEOTHERMAL:
        return "geothermal", GEOTHERMAL[tech_type]
    
    if tech_type in LARGE_HYDRO:
        return "hydro", LARGE_HYDRO[tech_type]
    
    return "other", "unclassified"

def load_tec_dispatchable(tec_processed_file: str) -> pd.DataFrame:
    """Load processed TEC dispatchable generators with enhanced location mapping."""
    logger.info(f"Loading processed TEC dispatchable generators from {tec_processed_file}")
    
    if not Path(tec_processed_file).exists():
        raise FileNotFoundError(f"Processed TEC file not found: {tec_processed_file}")
    
    # Load the already processed TEC data
    tec_df = pd.read_csv(tec_processed_file)
    
    # The processed file already contains:
    # - Built projects only
    # - Technology categorization
    # - Standardized column names
    # - Location mapping
    # - Dispatchable categories only
    
    logger.info(f"Loaded {len(tec_df)} processed dispatchable generators from TEC")
    log_dataframe_info(tec_df, logger, "TEC processed dispatchable")
    
    return tec_df

def load_repd_dispatchable(repd_file: str) -> pd.DataFrame:
    """Load dispatchable generators from REPD data."""
    logger.info(f"Loading REPD dispatchable generators from {repd_file}")
    
    if not Path(repd_file).exists():
        raise FileNotFoundError(f"REPD file not found: {repd_file}")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin1', 'cp1252']:
        try:
            repd_df = pd.read_csv(repd_file, encoding=encoding)
            logger.info(f"Successfully loaded REPD with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not decode REPD file with any standard encoding")
    
    # Filter for dispatchable technologies
    dispatchable_techs = list(STORAGE_SYSTEMS.keys()) + list(BIOMASS_WASTE.keys()) + list(GEOTHERMAL.keys()) + list(LARGE_HYDRO.keys())
    repd_dispatchable = repd_df[repd_df['Technology Type'].isin(dispatchable_techs)].copy()
    
    # Remove Northern Ireland (PyPSA-GB models Great Britain only, not the full UK)
    logger.info(f"Before Northern Ireland filter: {len(repd_dispatchable)} sites")
    ni_mask = repd_dispatchable['Country'] == 'Northern Ireland'
    ni_count = ni_mask.sum()
    # Convert capacity to numeric for sum calculation
    ni_capacity = pd.to_numeric(repd_dispatchable[ni_mask]['Installed Capacity (MWelec)'], errors='coerce').sum()
    repd_dispatchable = repd_dispatchable[~ni_mask].copy()
    if ni_count > 0:
        logger.info(f"Filtered out {ni_count} Northern Ireland dispatchable generators ({ni_capacity:.1f} MW)")
        logger.info(f"  (PyPSA-GB models Great Britain only, not the full UK)")
    logger.info(f"After Northern Ireland filter: {len(repd_dispatchable)} sites")
    
    # Filter for operational sites only
    logger.info(f"Before operational filter: {len(repd_dispatchable)} sites")
    repd_dispatchable = repd_dispatchable[repd_dispatchable['Development Status'] == 'Operational'].copy()
    logger.info(f"After operational filter: {len(repd_dispatchable)} operational sites")
    
    # Add technology categorization
    repd_dispatchable[['category', 'technology']] = repd_dispatchable['Technology Type'].apply(
        lambda x: pd.Series(categorize_repd_technology(x))
    )
    
    # Standardize column names
    repd_dispatchable = repd_dispatchable.rename(columns={
        'Site Name': 'site_name',
        'Installed Capacity (MWelec)': 'capacity_mw',
        'Technology Type': 'plant_type',
        'Development Status': 'status',
        'Operator (or Applicant)': 'operator',
        'X-coordinate': 'x_coord',
        'Y-coordinate': 'y_coord'
    })
    
    # Add source identifier
    repd_dispatchable['data_source'] = 'REPD'
    
    logger.info(f"Loaded {len(repd_dispatchable)} dispatchable generators from REPD")
    log_dataframe_info(repd_dispatchable, logger, "REPD dispatchable")
    
    return repd_dispatchable

def create_site_csv_by_technology(combined_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Create separate CSV files for each dispatchable technology.
    
    Args:
        combined_df: Combined TEC + REPD data
        output_dir: Output directory for CSV files
        
    Returns:
        Dictionary mapping technology to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Group by technology
    for technology in combined_df['technology'].unique():
        if technology in ['unknown', 'unclassified']:
            continue
            
        tech_data = combined_df[combined_df['technology'] == technology].copy()
        
        if len(tech_data) == 0:
            continue
        
        # Standardize columns for PyPSA compatibility
        standardized_cols = ['site_name', 'capacity_mw', 'technology', 'category', 'status', 'data_source']
        
        # Add coordinates if available
        if 'x_coord' in tech_data.columns and 'y_coord' in tech_data.columns:
            standardized_cols.extend(['x_coord', 'y_coord'])
        
        # Add connection site if available
        if 'connection_site' in tech_data.columns:
            standardized_cols.append('connection_site')
        
        # Add operator if available  
        if 'operator' in tech_data.columns:
            standardized_cols.append('operator')
        
        # Select available columns
        available_cols = [col for col in standardized_cols if col in tech_data.columns]
        tech_csv = tech_data[available_cols].copy()
        
        # Sort by capacity (largest first)
        tech_csv = tech_csv.sort_values('capacity_mw', ascending=False)
        
        # Create filename
        filename = f"{technology}_sites.csv"
        filepath = output_dir / filename
        
        # Save to CSV
        tech_csv.to_csv(filepath, index=False)
        output_files[technology] = str(filepath)
        
        logger.info(f"Created {filename}: {len(tech_csv)} sites, {tech_csv['capacity_mw'].sum():.1f} MW total")
    
    return output_files

def create_summary_report(combined_df: pd.DataFrame, output_files: Dict[str, str], 
                         output_path: str) -> None:
    """Create summary report of dispatchable generator sites."""
    
    report_lines = [
        "PyPSA-GB Dispatchable Generator Sites Summary",
        "=" * 50,
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Technology Summary:",
        "-" * 20
    ]
    
    # Summary by technology
    tech_summary = combined_df.groupby('technology').agg({
        'capacity_mw': ['count', 'sum', 'mean'],
        'data_source': lambda x: ', '.join(x.unique())
    }).round(1)
    
    tech_summary.columns = ['sites', 'total_mw', 'avg_mw', 'sources']
    tech_summary = tech_summary.sort_values('total_mw', ascending=False)
    
    for tech, row in tech_summary.iterrows():
        if tech not in ['unknown', 'unclassified']:
            report_lines.append(
                f"{tech:<25} {row['sites']:>3} sites, {row['total_mw']:>7.1f} MW total, "
                f"{row['avg_mw']:>5.1f} MW avg ({row['sources']})"
            )
    
    report_lines.extend([
        "",
        "Category Summary:",
        "-" * 20
    ])
    
    # Summary by category
    cat_summary = combined_df.groupby('category').agg({
        'capacity_mw': ['count', 'sum'],
        'technology': lambda x: len(x.unique())
    }).round(1)
    
    cat_summary.columns = ['sites', 'total_mw', 'technologies']
    cat_summary = cat_summary.sort_values('total_mw', ascending=False)
    
    for category, row in cat_summary.iterrows():
        if category not in ['unknown', 'unclassified', 'other']:
            report_lines.append(
                f"{category:<20} {row['sites']:>3} sites, {row['total_mw']:>7.1f} MW total, "
                f"{int(row['technologies'])} technologies"
            )
    
    report_lines.extend([
        "",
        "Data Source Summary:",
        "-" * 20
    ])
    
    source_summary = combined_df.groupby('data_source').agg({
        'capacity_mw': ['count', 'sum']
    }).round(1)
    
    source_summary.columns = ['sites', 'total_mw']
    
    for source, row in source_summary.iterrows():
        report_lines.append(f"{source:<10} {row['sites']:>3} sites, {row['total_mw']:>7.1f} MW total")
    
    total_sites = len(combined_df)
    total_capacity = combined_df['capacity_mw'].sum()
    
    report_lines.extend([
        "",
        "Overall Summary:",
        "-" * 20,
        f"Total Sites: {total_sites}",
        f"Total Capacity: {total_capacity:.1f} MW",
        f"Average Capacity: {total_capacity/total_sites:.1f} MW",
        "",
        "Output Files:",
        "-" * 20
    ])
    
    for tech, filepath in output_files.items():
        report_lines.append(f"{tech}: {filepath}")
    
    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Created summary report: {output_path}")

def main(tec_processed_file: str = "resources/generators/tec_processed_complete.csv",
         repd_file: str = "data/renewables/repd-q2-jul-2025.csv",
         output_dir: str = "resources/generators",
         summary_report: str = "resources/generators/dispatchable_sites_summary.txt"):
    """
    Main function to prepare dispatchable generator site data.
    
    Args:
        tec_processed_file: Path to processed TEC register with enhanced location mapping
        repd_file: Path to REPD data
        output_dir: Output directory for site CSV files
        summary_report: Path for summary report
    """
    logger.info("Starting dispatchable generator site preparation")
    
    # Load data
    tec_data = load_tec_dispatchable(tec_processed_file)
    repd_data = load_repd_dispatchable(repd_file)
    
    # Combine datasets
    logger.info("Combining TEC and REPD dispatchable data")
    
    # Ensure common columns exist
    common_cols = ['site_name', 'capacity_mw', 'technology', 'category', 'status', 'data_source']
    
    # Add missing columns with NaN
    for col in common_cols:
        if col not in tec_data.columns:
            tec_data[col] = np.nan
        if col not in repd_data.columns:
            repd_data[col] = np.nan
    
    # Combine dataframes
    combined_df = pd.concat([tec_data, repd_data], ignore_index=True, sort=False)
    
    # Clean capacity data
    combined_df['capacity_mw'] = pd.to_numeric(combined_df['capacity_mw'], errors='coerce')
    combined_df = combined_df.dropna(subset=['capacity_mw'])
    combined_df = combined_df[combined_df['capacity_mw'] > 0]
    
    logger.info(f"Combined dataset: {len(combined_df)} total dispatchable sites")
    log_dataframe_info(combined_df, logger, "Combined dispatchable data")
    
    # Create technology-specific CSV files
    output_files = create_site_csv_by_technology(combined_df, output_dir)
    
    # Create summary report
    create_summary_report(combined_df, output_files, summary_report)
    
    logger.info("Dispatchable generator site preparation completed successfully")
    
    return {
        'total_sites': len(combined_df),
        'total_capacity_mw': combined_df['capacity_mw'].sum(),
        'technologies': len(output_files),
        'output_files': output_files
    }

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    if 'snakemake' in globals():
        # Snakemake execution
        tec_file = snakemake.input.tec_processed
        repd_file = snakemake.input.repd_file
        output_dir = "resources/generators/sites"  # Fixed output directory
        summary_report = snakemake.output.summary_report
    else:
        # Command line execution
        tec_file = sys.argv[1] if len(sys.argv) > 1 else "resources/generators/tec_processed_complete.csv"
        repd_file = sys.argv[2] if len(sys.argv) > 2 else "data/renewables/repd-q2-jul-2025.csv"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "resources/generators"
        summary_report = sys.argv[4] if len(sys.argv) > 4 else "resources/generators/dispatchable_sites_summary.txt"
    
    stats = main(tec_file, repd_file, output_dir, summary_report)
    
    logger.info("Dispatchable Generator Site Preparation Summary:")
    logger.info("Total sites: %d", stats['total_sites'])
    logger.info("Total capacity: %.1f MW", stats['total_capacity_mw'])
    logger.info("Technologies: %d", stats['technologies'])
    logger.info("Output files created: %d", len(stats['output_files']))

