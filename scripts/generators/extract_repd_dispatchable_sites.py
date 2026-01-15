"""
Extract Dispatchable Thermal Sites from REPD for PyPSA-GB

This script extracts dispatchable renewable thermal generator sites directly from REPD
(Renewable Energy Planning Database) without requiring TEC data.

These sites represent dispatchable renewable thermal generation that can be controlled
(unlike variable renewables like wind/solar), including:

1. Biomass/Waste (dispatchable renewable thermal):
   - Biomass (dedicated biomass plants)
   - Waste to Energy (EfW Incineration)
   - Biogas (Anaerobic Digestion)
   - Landfill Gas
   - Sewage Gas (Sewage Sludge Digestion)
   - Advanced Biofuels (Advanced Conversion Technologies)

2. Geothermal:
   - Baseload renewable thermal (constant output)

3. Large Hydro:
   - Large reservoir-based hydro (dispatchable, >20MW typically)

Note: Conventional thermal (CCGT, OCGT, Nuclear, Coal) now comes from DUKES (historical)
or FES (future) data sources, not TEC. Storage comes from REPD via storage.smk rules.

Author: PyPSA-GB Development Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logger = None
try:
    from logging_config import setup_logging, log_dataframe_info
    logger = setup_logging("extract_repd_dispatchable_sites")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_dataframe_info
        logger = setup_logging("extract_repd_dispatchable_sites")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("extract_repd_dispatchable_sites")
        def log_dataframe_info(df, logger, name: str):
            logger.info(f"{name} shape: {df.shape}")


# REPD technology type to output category mapping
REPD_TECHNOLOGY_MAPPING = {
    # Biomass
    'Dedicated Biomass': 'biomass',
    'Biomass (co-firing)': 'biomass',
    
    # Waste to Energy
    'EfW Incineration': 'waste_to_energy',
    'ACT (not CHP)': 'waste_to_energy',  # Advanced Conversion Technology
    
    # Biogas
    'Anaerobic Digestion': 'biogas',
    
    # Landfill Gas
    'Landfill Gas': 'landfill_gas',
    
    # Sewage Gas
    'Sewage Sludge Digestion': 'sewage_gas',
    
    # Advanced Biofuels
    'Advanced Conversion Technologies': 'advanced_biofuel',
    
    # Geothermal
    'Geothermal': 'geothermal',
    
    # Large Hydro
    'Large Hydro': 'large_hydro',
}


def convert_osgb36_to_wgs84(x_coords: pd.Series, y_coords: pd.Series) -> tuple:
    """
    Convert OSGB36 (British National Grid) coordinates to WGS84 (lat/lon).
    
    Args:
        x_coords: Easting values (X)
        y_coords: Northing values (Y)
    
    Returns:
        Tuple of (longitude, latitude) as pandas Series
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        
        # Filter valid coordinates
        valid_mask = x_coords.notna() & y_coords.notna()
        
        lon = pd.Series(index=x_coords.index, dtype=float)
        lat = pd.Series(index=y_coords.index, dtype=float)
        
        if valid_mask.any():
            lon_vals, lat_vals = transformer.transform(
                x_coords[valid_mask].values,
                y_coords[valid_mask].values
            )
            lon.loc[valid_mask] = lon_vals
            lat.loc[valid_mask] = lat_vals
        
        return lon, lat
    except ImportError:
        logger.warning("pyproj not available - coordinates will not be converted")
        return pd.Series(dtype=float), pd.Series(dtype=float)


def extract_dispatchable_sites_from_repd(repd_path: str) -> dict:
    """
    Extract dispatchable thermal generator sites from REPD.
    
    Args:
        repd_path: Path to REPD CSV file
    
    Returns:
        Dictionary mapping category names to DataFrames
    """
    logger.info(f"Loading REPD from {repd_path}")
    repd_df = pd.read_csv(repd_path, encoding='latin-1', low_memory=False)
    logger.info(f"Loaded {len(repd_df)} REPD records")
    
    # Filter for operational projects only
    operational_statuses = [
        'Operational',
        'Under Construction',  # Include these as they'll be operational soon
    ]
    
    # Check what status column exists
    status_col = None
    for col in ['Development Status', 'Development Status (short)', 'Status']:
        if col in repd_df.columns:
            status_col = col
            break
    
    if status_col:
        operational_mask = repd_df[status_col].isin(operational_statuses)
        repd_df = repd_df[operational_mask].copy()
        logger.info(f"Filtered to {len(repd_df)} operational/under construction projects")
    
    # Get technology column
    tech_col = None
    for col in ['Technology Type', 'Technology', 'Type']:
        if col in repd_df.columns:
            tech_col = col
            break
    
    if tech_col is None:
        logger.error(f"No technology column found. Columns: {repd_df.columns.tolist()[:20]}")
        raise KeyError("Technology column not found in REPD")
    
    # Get capacity column
    capacity_col = None
    for col in ['Installed Capacity (MWelec)', 'Installed Capacity (MW)', 'Capacity (MW)']:
        if col in repd_df.columns:
            capacity_col = col
            break
    
    if capacity_col is None:
        # Try to find any capacity column
        cap_cols = [c for c in repd_df.columns if 'capacity' in c.lower()]
        if cap_cols:
            capacity_col = cap_cols[0]
    
    # Get coordinate columns
    x_col = y_col = None
    for col in repd_df.columns:
        if 'x-coordinate' in col.lower() or col == 'X':
            x_col = col
        if 'y-coordinate' in col.lower() or col == 'Y':
            y_col = col
    
    # Get name column
    name_col = None
    for col in ['Site Name', 'Project Name', 'Name']:
        if col in repd_df.columns:
            name_col = col
            break
    
    # Initialize output dictionary
    site_categories = {
        'biomass': [],
        'waste_to_energy': [],
        'biogas': [],
        'landfill_gas': [],
        'sewage_gas': [],
        'advanced_biofuel': [],
        'geothermal': [],
        'large_hydro': [],
    }
    
    # Process each technology type
    for tech, category in REPD_TECHNOLOGY_MAPPING.items():
        tech_mask = repd_df[tech_col].str.contains(tech, case=False, na=False)
        tech_sites = repd_df[tech_mask].copy()
        
        if len(tech_sites) == 0:
            continue
        
        logger.info(f"  {tech}: {len(tech_sites)} sites -> {category}")
        
        # Create standardized site records
        for _, row in tech_sites.iterrows():
            site = {
                'site_name': row.get(name_col, f"Site_{_}") if name_col else f"Site_{_}",
                'capacity_mw': pd.to_numeric(row.get(capacity_col, 0), errors='coerce') or 0,
                'technology': tech,
                'category': category,
                'data_source': 'REPD',
            }
            
            # Add coordinates
            if x_col and y_col:
                site['x_coord'] = pd.to_numeric(row.get(x_col), errors='coerce')
                site['y_coord'] = pd.to_numeric(row.get(y_col), errors='coerce')
            
            # Add other useful fields
            if 'Operator (or Conditions)' in repd_df.columns:
                site['operator'] = row.get('Operator (or Conditions)', '')
            if 'Planning Authority' in repd_df.columns:
                site['planning_authority'] = row.get('Planning Authority', '')
            if 'Region' in repd_df.columns:
                site['region'] = row.get('Region', '')
            
            site_categories[category].append(site)
    
    # Convert lists to DataFrames and add lat/lon coordinates
    result = {}
    for category, sites in site_categories.items():
        if sites:
            df = pd.DataFrame(sites)
            
            # Convert coordinates
            if 'x_coord' in df.columns and 'y_coord' in df.columns:
                df['lon'], df['lat'] = convert_osgb36_to_wgs84(df['x_coord'], df['y_coord'])
            
            # Filter to GB only (exclude Northern Ireland)
            if 'lat' in df.columns and 'lon' in df.columns:
                ni_mask = (
                    (df['lat'] > 54.0) & 
                    (df['lat'] < 55.5) & 
                    (df['lon'] < -5.5) & 
                    (df['lon'] > -8.0)
                )
                ni_count = ni_mask.sum()
                if ni_count > 0:
                    logger.info(f"  Filtered {ni_count} Northern Ireland sites from {category}")
                df = df[~ni_mask].copy()
            
            result[category] = df
            logger.info(f"  Final {category}: {len(df)} sites, {df['capacity_mw'].sum():.1f} MW")
        else:
            result[category] = pd.DataFrame()
    
    return result


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("EXTRACTING DISPATCHABLE THERMAL SITES FROM REPD")
    logger.info("=" * 80)
    
    try:
        # Access snakemake variables
        snk = globals().get('snakemake')
        if snk:
            repd_path = snk.input.repd
            output_paths = {
                'biomass': snk.output.biomass_sites,
                'waste_to_energy': snk.output.waste_to_energy_sites,
                'biogas': snk.output.biogas_sites,
                'landfill_gas': snk.output.landfill_gas_sites,
                'sewage_gas': snk.output.sewage_gas_sites,
                'advanced_biofuel': snk.output.advanced_biofuel_sites,
                'geothermal': snk.output.geothermal_sites,
                'large_hydro': snk.output.large_hydro_sites,
            }
            summary_path = snk.output.summary_report
        else:
            # Standalone mode
            repd_path = "data/renewables/repd-q2-jul-2025.csv"
            output_paths = {
                'biomass': "resources/generators/sites/biomass_sites.csv",
                'waste_to_energy': "resources/generators/sites/waste_to_energy_sites.csv",
                'biogas': "resources/generators/sites/biogas_sites.csv",
                'landfill_gas': "resources/generators/sites/landfill_gas_sites.csv",
                'sewage_gas': "resources/generators/sites/sewage_gas_sites.csv",
                'advanced_biofuel': "resources/generators/sites/advanced_biofuel_sites.csv",
                'geothermal': "resources/generators/sites/geothermal_sites.csv",
                'large_hydro': "resources/generators/sites/large_hydro_sites.csv",
            }
            summary_path = "resources/generators/dispatchable_sites_summary.txt"
        
        # Extract sites from REPD
        site_categories = extract_dispatchable_sites_from_repd(repd_path)
        
        # Write output files
        total_sites = 0
        total_capacity = 0
        summary_lines = ["REPD Dispatchable Thermal Sites Summary", "=" * 50, ""]
        
        for category, output_path in output_paths.items():
            df = site_categories.get(category, pd.DataFrame())
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if len(df) > 0:
                df.to_csv(output_path, index=False)
                sites = len(df)
                capacity = df['capacity_mw'].sum()
                total_sites += sites
                total_capacity += capacity
                summary_lines.append(f"{category}: {sites} sites, {capacity:.1f} MW")
                logger.info(f"Wrote {sites} {category} sites to {output_path}")
            else:
                # Write empty CSV with headers
                pd.DataFrame(columns=['site_name', 'capacity_mw', 'technology', 'category', 
                                     'data_source', 'x_coord', 'y_coord', 'lat', 'lon']).to_csv(output_path, index=False)
                summary_lines.append(f"{category}: 0 sites")
                logger.info(f"Wrote empty file for {category} (no sites found)")
        
        # Write summary
        summary_lines.extend([
            "",
            "=" * 50,
            f"TOTAL: {total_sites} sites, {total_capacity:.1f} MW",
            "",
            "Data source: REPD (Renewable Energy Planning Database)",
            "Note: Conventional thermal (CCGT, OCGT, Nuclear, Coal) comes from DUKES/FES",
        ])
        
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info("=" * 80)
        logger.info("REPD DISPATCHABLE SITE EXTRACTION COMPLETED")
        logger.info(f"Total: {total_sites} sites, {total_capacity:.1f} MW")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to extract REPD dispatchable sites: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

