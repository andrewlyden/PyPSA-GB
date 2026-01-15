#!/usr/bin/env python3
"""
Enhanced interconnector location enrichment using NESO data and external sources.

This script enriches interconnector data with precise GB-side and international
landing point coordinates, substation details, and country context using:
1. NESO Interconnector Register for detailed GB connection sites
2. Transmission substation coordinate databases
3. International counterparty location mapping
4. OSGB coordinate conversion where needed

Author: AI Assistant
Date: January 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re
import time
from typing import Dict, Tuple, Optional

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from scripts.utilities.logging_config import setup_logging, log_execution_summary
except ImportError:
    pass

# Known substation coordinates (GB National Grid main sites)
# These are approximate coordinates for major transmission substations
SUBSTATION_COORDINATES = {
    # England - 400kV Primary Sites
    'Sellindge 400kV Substation': (51.1026, 1.0634),  # Kent, used by IFA & ElecLink
    'Grain 400kV Substation': (51.4454, 0.7089),      # Kent, used by BritNed
    'Grain West 400kV Substation': (51.4454, 0.7089), # Kent, used by NeuConnect
    'Richborough 400kV Substation': (51.2964, 1.3238), # Kent, used by Nemo Link
    'Chilling 400kV Substation': (50.8542, -1.2742),  # Hampshire, used by IFA2
    'Deeside 400kV Substation': (53.2258, -3.0204),   # Wales/England border, East West
    'Bicker Fen 400kV Substation': (52.9167, -0.0833), # Lincolnshire, Viking Link
    'Blyth GSP': (55.1269, -1.5085),                   # Northumberland, NS Link
    'Pembroke 400kV Substation': (51.6749, -4.9149),  # Wales, Greenlink
    'Exeter 400kV Substation': (50.7236, -3.5339),    # Devon, FAB Link
    'Canterbury North 400kV Substation': (51.2802, 1.0789), # Kent
    'Friston 400kV Substation': (52.2127, 1.4053),    # Suffolk, LionLink
    'Wallend 400kV Substation': (55.0083, -1.5333),   # Tyne and Wear, Nautilus
    'Kingsnorth 400kV Substation': (51.3833, 0.5833), # Kent
    'Kemsley 400kV Substation': (51.3404, 0.7584),    # Kent, Cronos
    'Mablethorpe 400kV Substation': (53.3417, 0.2667), # Lincolnshire, SENECA
    'East Anglia Connection Node 400kV Substation': (52.5, 1.5),  # Norfolk/Suffolk, Tarchon
    'Creyke Beck 400kV Substation': (53.7833, -0.4167), # Yorkshire, The Superconnection
    'East Kent Connection Node B 400kV Substation': (51.25, 1.1), # Kent, Project Cobalt
    'Wearside Connection Node B 400kV Substation': (54.85, -1.4), # Sunderland, Teeside MPI
    'Lovedean 400kV Substation': (50.9647, -1.0886),  # Hampshire, Aquind
    'Alverdiscott 400kV Substation': (50.9833, -4.2), # Devon, Celtic MPI
    'Birkhill Wood 400kV Substation': (55.4, -4.0),   # Scotland, Continental Link
    'Connah\'s Quay 400kV Substation': (53.2167, -3.05), # Wales, Western MPI
    'Bodelwyddan 400kV Substation': (53.2833, -3.5),  # Wales, MARES
    
    # Scotland - 400kV/275kV Sites  
    'Auchencrosh 275kV': (55.2167, -5.1833),          # Scotland, Moyle
    'Hunterston East 400kV': (55.7167, -4.9),         # Scotland, LIRIC
    'Peterhead 400kV Substation': (57.5, -1.7833),    # Scotland, NorthConnect
    'Fiddes 400kV Substation': (57.0, -2.0),          # Scotland, GB-EU MPI
    
    # Lower voltage connections
    'Bispham 132kV Substation': (53.8167, -3.0333),   # Blackpool, Isle of Man
    'Pembroke GSP': (51.6749, -4.9149),               # Wales, Low Carbon Link
    
    # Alternative names/spellings
    'Sellindge': (51.1026, 1.0634),
    'Grain': (51.4454, 0.7089),
    'Richborough': (51.2964, 1.3238),
    'Chilling': (50.8542, -1.2742),
    'Deeside': (53.2258, -3.0204),
    'Bicker Fen': (52.9167, -0.0833),
    'Pembroke': (51.6749, -4.9149),
    'Auchencrosh': (55.2167, -5.1833),
    'Bispham': (53.8167, -3.0333),
}

# International counterparty locations (major interconnector landing points)
INTERNATIONAL_LOCATIONS = {
    'France': {
        'Calais': (50.9513, 1.8587),           # IFA, ElecLink terminals
        'Les Mandarins': (50.9513, 1.8587),    # IFA specific terminal
        'Sangatte': (50.9365, 1.7853),         # ElecLink specific terminal
        'Caen': (49.1829, -0.3707),            # IFA2 terminal
        'Normandy': (49.1829, -0.3707),        # IFA2 region
    },
    'Netherlands': {
        'Maasvlakte': (51.9889, 4.0581),       # BritNed terminal
        'Rotterdam': (51.9244, 4.4777),        # General area
        'Eemshaven': (53.4506, 6.8389),        # NeuConnect terminal
    },
    'Belgium': {
        'Zeebrugge': (51.2993, 3.2026),        # Nemo Link terminal
        'Bruges': (51.2093, 3.2247),           # General area
    },
    'Denmark': {
        'Revsing': (55.4833, 9.1167),          # Viking Link terminal
        'Jutland': (55.4833, 9.1167),          # General region
    },
    'Norway': {
        'Kvilldal': (59.3500, 6.0333),         # NS Link terminal
        'Stavanger': (58.9700, 5.7331),        # General area
    },
    'Republic of Ireland': {
        'Woodland': (52.1667, -6.3667),        # East West terminal
        'Great Island': (52.2167, -6.95),      # Greenlink terminal
        'Wexford': (52.3369, -6.4633),         # General area
        'Rush North Beach': (53.5181, -6.0925), # East West Interconnector - Dublin
        'Rush': (53.5181, -6.0925),            # Alternative name
    },
    'Ireland': {
        # Default Ireland locations (same as Republic of Ireland for backward compatibility)
        # Order matters - first entry is the default
        'Great Island': (52.2297, -6.9603),    # Greenlink terminal - Wexford (DEFAULT)
        'Rush North Beach': (53.5181, -6.0925), # East West Interconnector - Dublin  
        'Rush': (53.5181, -6.0925),            # East West alternative name
        'Woodland': (52.1667, -6.3667),        # East West old/alternative terminal
        'Wexford': (52.3369, -6.4633),         # Greenlink general area
        'Dublin': (53.3498, -6.2603),          # East West general area
    },
    'Northern Ireland': {
        'Ballycronan More': (54.7333, -5.9333), # Moyle terminal
        'Belfast': (54.5973, -5.9301),         # General area
    },
    'Isle Of Man': {
        'Douglas': (54.1500, -4.4833),         # Terminal area
    },
}

# Regional mappings for coordinate validation
# Note: England-Scotland border varies by longitude:
#   - East coast (Berwick): ~55.8°N  
#   - West coast (Solway): ~55.0°N
# Blyth (55.13°N, -1.51°W) is in Northumberland, England
# Edinburgh (55.95°N, -3.19°W) is clearly in Scotland
# For simplicity, use generous England range and check Scotland first for northern areas
GB_REGIONS = {
    'England': {'lat_range': (49.8, 56.0), 'lon_range': (-6.5, 2.0)},   # Include all of Northumberland
    'Scotland': {'lat_range': (55.5, 60.9), 'lon_range': (-8.0, 0.0)},  # Clear Scottish territory
    'Wales': {'lat_range': (51.3, 53.5), 'lon_range': (-5.5, -2.6)},
    'Northern Ireland': {'lat_range': (54.0, 55.5), 'lon_range': (-8.2, -5.4)},
}

def get_substation_coordinates(connection_site: str) -> Optional[Tuple[float, float]]:
    """
    Get coordinates for a GB substation connection site.
    
    Args:
        connection_site: Name of the connection site/substation
        
    Returns:
        Tuple of (latitude, longitude) if found, None otherwise
    """
    if pd.isna(connection_site) or not connection_site:
        return None
        
    # Direct lookup
    if connection_site in SUBSTATION_COORDINATES:
        return SUBSTATION_COORDINATES[connection_site]
    
    # Try fuzzy matching for variations
    connection_clean = re.sub(r'\s+(400kV|275kV|132kV|GSP|Substation)\s*', ' ', connection_site, flags=re.IGNORECASE).strip()
    
    for substation, coords in SUBSTATION_COORDINATES.items():
        substation_clean = re.sub(r'\s+(400kV|275kV|132kV|GSP|Substation)\s*', ' ', substation, flags=re.IGNORECASE).strip()
        if connection_clean.lower() in substation_clean.lower() or substation_clean.lower() in connection_clean.lower():
            return coords
    
    logger.warning(f"No coordinates found for connection site: {connection_site}")
    return None

def get_international_coordinates(country: str, specific_location: str = None, interconnector_name: str = None) -> Optional[Tuple[float, float]]:
    """
    Get coordinates for international counterparty locations.
    
    Args:
        country: Country name
        specific_location: Specific location if known
        interconnector_name: Name of interconnector (for name-based matching)
        
    Returns:
        Tuple of (latitude, longitude) if found, None otherwise
    """
    if pd.isna(country) or not country:
        return None
        
    # Clean country name
    country_clean = country.strip()
    
    if country_clean not in INTERNATIONAL_LOCATIONS:
        logger.warning(f"No international location data for country: {country_clean}")
        return None
    
    locations = INTERNATIONAL_LOCATIONS[country_clean]
    
    # Try matching by interconnector name first (for Ireland specifically)
    if interconnector_name and pd.notna(interconnector_name) and country_clean == 'Ireland':
        ic_name_lower = interconnector_name.lower()
        # EastWest -> Rush/Dublin
        if 'east' in ic_name_lower and 'west' in ic_name_lower:
            return locations.get('Rush North Beach', locations.get('Rush', None))
        # Greenlink -> Great Island/Wexford  
        elif 'greenlink' in ic_name_lower or 'green' in ic_name_lower:
            return locations.get('Great Island', locations.get('Wexford', None))
    
    # If specific location provided, try to match
    if specific_location and pd.notna(specific_location):
        for loc_name, coords in locations.items():
            if specific_location.lower() in loc_name.lower() or loc_name.lower() in specific_location.lower():
                return coords
    
    # Return first (primary) location for country
    return list(locations.values())[0]

def determine_gb_region(lat: float, lon: float) -> str:
    """
    Determine GB region based on coordinates.
    Uses a priority-based approach considering both lat and lon.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Region name or 'Unknown'
    """
    # Northern Ireland check (westernmost, distinct region)
    if 54.0 <= lat <= 55.5 and -8.2 <= lon <= -5.4:
        return 'Northern Ireland'
    
    # Wales check (western Britain, south of Scotland)
    if 51.3 <= lat <= 53.5 and -5.5 <= lon <= -2.6:
        return 'Wales'
    
    # Scotland check - more nuanced due to east-west variation
    # West coast Scotland (including Auchencrosh): lon < -3.5, lat > 55.0
    # East coast Scotland (including Edinburgh): lon > -3.5, lat > 55.5  
    if lon < -3.5 and lat >= 55.0 and lat <= 60.9:
        return 'Scotland'
    if lon >= -3.5 and lat >= 55.5 and lat <= 60.9:
        return 'Scotland'
    
    # England (everything else in GB)
    if 49.8 <= lat <= 56.0 and -6.5 <= lon <= 2.0:
        return 'England'
    
    return 'Unknown'

def validate_coordinates(lat: float, lon: float, expected_region: str = None) -> bool:
    """
    Validate that coordinates are reasonable for GB/Europe.
    
    Args:
        lat: Latitude
        lon: Longitude
        expected_region: Expected GB region if applicable
        
    Returns:
        True if coordinates seem valid
    """
    # Basic bounds check for UK/Europe region
    if not (40.0 <= lat <= 70.0 and -15.0 <= lon <= 15.0):
        return False
    
    # If expected region specified, determine actual region and compare
    if expected_region and expected_region in GB_REGIONS:
        actual_region = determine_gb_region(lat, lon)
        
        # Check if actual region matches expected (with some tolerance for border areas)
        if actual_region == expected_region:
            return True
        elif actual_region == 'Unknown':
            # Unknown region - check against bounds with tolerance
            bounds = GB_REGIONS[expected_region]
            lat_min, lat_max = bounds['lat_range']
            lon_min, lon_max = bounds['lon_range']
            
            # Allow some tolerance for border areas
            if (lat_min - 0.5 <= lat <= lat_max + 0.5 and lon_min - 1.0 <= lon <= lon_max + 1.0):
                return True
            else:
                logger.warning(f"Coordinates ({lat}, {lon}) outside expected region {expected_region}")
                return False
        else:
            # Different region detected
            logger.warning(f"Coordinates ({lat}, {lon}) detected as {actual_region}, expected {expected_region}")
            return False
    
    return True

def enrich_interconnector_locations(input_file: str, neso_register_file: str, output_file: str):
    """
    Enrich interconnector dataset with detailed location information.
    
    Args:
        input_file: Path to input interconnectors CSV
        neso_register_file: Path to NESO interconnector register CSV  
        output_file: Path to output enriched CSV
    """
    logger.info("Starting interconnector location enrichment")
    
    # Load datasets
    logger.info(f"Loading interconnectors data from {input_file}")
    interconnectors = pd.read_csv(input_file)
    
    logger.info(f"Loading NESO register from {neso_register_file}")
    neso_register = pd.read_csv(neso_register_file)
    
    # Initialize new columns for enriched data
    new_columns = [
        'gb_latitude', 'gb_longitude', 'gb_region', 'gb_connection_site_full',
        'international_latitude', 'international_longitude', 'international_location',
        'location_source', 'coordinate_quality', 'location_notes'
    ]
    
    for col in new_columns:
        if col not in interconnectors.columns:
            interconnectors[col] = np.nan
    
    # Create NESO lookup by name (handle variations)
    neso_lookup = {}
    for _, row in neso_register.iterrows():
        project_name = row['Project Name']
        connection_site = row['Connection Site']
        
        # Store multiple name variations
        name_variations = [
            project_name,
            project_name.replace(' Interconnector', '').replace(' interconnector', ''),
            project_name.replace('Interconnector ', '').replace('interconnector ', ''),
        ]
        
        for name_var in name_variations:
            neso_lookup[name_var.strip()] = {
                'connection_site': connection_site,
                'project_status': row['Project Status'],
                'host_to': row['HOST TO'],
                'mw_import': row['MW Import - Total'],
                'mw_export': row['MW Export - Total']
            }
    
    # Process each interconnector
    enriched_count = 0
    
    for idx, row in interconnectors.iterrows():
        interconnector_name = row['name']
        landing_point = row.get('landing_point_gb', '')
        country = row.get('counterparty_country', '')
        
        logger.debug(f"Processing: {interconnector_name}")
        
        # Try to get enhanced data from NESO register
        neso_data = None
        for name_variation in [interconnector_name, 
                              interconnector_name.replace(' Interconnector', ''),
                              interconnector_name.replace('Interconnector ', ''),
                              landing_point]:
            if name_variation in neso_lookup:
                neso_data = neso_lookup[name_variation]
                break
        
        # Determine GB connection site
        gb_connection_site = None
        if neso_data and neso_data['connection_site']:
            gb_connection_site = neso_data['connection_site']
            interconnectors.at[idx, 'gb_connection_site_full'] = gb_connection_site
        elif landing_point:
            gb_connection_site = landing_point
            interconnectors.at[idx, 'gb_connection_site_full'] = landing_point
        
        # Get GB coordinates
        gb_coords = None
        location_source = []
        coordinate_quality = 'unknown'
        location_notes = []
        
        if gb_connection_site:
            gb_coords = get_substation_coordinates(gb_connection_site)
            if gb_coords:
                interconnectors.at[idx, 'gb_latitude'] = gb_coords[0]
                interconnectors.at[idx, 'gb_longitude'] = gb_coords[1]
                interconnectors.at[idx, 'gb_region'] = determine_gb_region(gb_coords[0], gb_coords[1])
                location_source.append('substation_database')
                coordinate_quality = 'high'
                
                # Validate coordinates
                if not validate_coordinates(gb_coords[0], gb_coords[1]):
                    location_notes.append('coordinate_validation_warning')
                    coordinate_quality = 'medium'
            else:
                location_notes.append('gb_coordinates_not_found')
                coordinate_quality = 'low'
        
        # Get international coordinates
        if country:
            # Pass interconnector name to help match specific locations
            ic_name = row.get('name', '')
            international_coords = get_international_coordinates(country, interconnector_name=ic_name)
            if international_coords:
                interconnectors.at[idx, 'international_latitude'] = international_coords[0]
                interconnectors.at[idx, 'international_longitude'] = international_coords[1]
                # Store the matched location name
                matched_location = None
                if country in INTERNATIONAL_LOCATIONS:
                    for loc_name, coords in INTERNATIONAL_LOCATIONS[country].items():
                        if coords == international_coords:
                            matched_location = loc_name
                            break
                interconnectors.at[idx, 'international_location'] = matched_location or list(INTERNATIONAL_LOCATIONS.get(country, {}).keys())[0] if country in INTERNATIONAL_LOCATIONS else country
                location_source.append('international_database')
                
                if coordinate_quality == 'unknown':
                    coordinate_quality = 'medium'
            else:
                location_notes.append('international_coordinates_not_found')
        
        # Store metadata
        interconnectors.at[idx, 'location_source'] = ';'.join(location_source) if location_source else 'none'
        interconnectors.at[idx, 'coordinate_quality'] = coordinate_quality
        interconnectors.at[idx, 'location_notes'] = ';'.join(location_notes) if location_notes else ''
        
        if gb_coords or international_coords:
            enriched_count += 1
    
    # Add summary statistics
    total_interconnectors = len(interconnectors)
    gb_coords_count = interconnectors['gb_latitude'].notna().sum()
    international_coords_count = interconnectors['international_latitude'].notna().sum()
    high_quality_count = (interconnectors['coordinate_quality'] == 'high').sum()
    
    logger.info(f"Location enrichment completed:")
    logger.info(f"  Total interconnectors: {total_interconnectors}")
    logger.info(f"  With GB coordinates: {gb_coords_count} ({gb_coords_count/total_interconnectors*100:.1f}%)")
    logger.info(f"  With international coordinates: {international_coords_count} ({international_coords_count/total_interconnectors*100:.1f}%)")
    logger.info(f"  High quality coordinates: {high_quality_count} ({high_quality_count/total_interconnectors*100:.1f}%)")
    
    # Save enriched dataset
    logger.info(f"Saving enriched data to {output_file}")
    interconnectors.to_csv(output_file, index=False)
    
    # Create summary report
    summary_file = output_file.replace('.csv', '_location_summary.csv')
    summary_data = []
    
    for _, row in interconnectors.iterrows():
        summary_data.append({
            'name': row['name'],
            'gb_connection_site': row.get('gb_connection_site_full', ''),
            'gb_coordinates': f"({row.get('gb_latitude', '')}, {row.get('gb_longitude', '')})" if pd.notna(row.get('gb_latitude')) else '',
            'gb_region': row.get('gb_region', ''),
            'international_location': row.get('international_location', ''),
            'international_coordinates': f"({row.get('international_latitude', '')}, {row.get('international_longitude', '')})" if pd.notna(row.get('international_latitude')) else '',
            'coordinate_quality': row.get('coordinate_quality', ''),
            'location_source': row.get('location_source', ''),
            'notes': row.get('location_notes', '')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Location summary saved to {summary_file}")

def main():
    """Main execution function for Snakemake workflow."""
    logger = setup_logging("enrich_interconnector_locations") if 'setup_logging' in globals() else logging.getLogger(__name__)
    start_time = time.time()
    
    # Get paths from Snakemake
    input_file = str(snakemake.input[0])
    neso_register_file = str(snakemake.input[1])
    output_file = str(snakemake.output[0])
    
    # Run enrichment
    enrich_interconnector_locations(input_file, neso_register_file, output_file)
    
    # Read enriched data for statistics
    enriched_df = pd.read_csv(output_file)
    total_interconnectors = len(enriched_df)
    with_coordinates = enriched_df[['gb_latitude', 'gb_longitude']].notna().all(axis=1).sum()
    missing_coordinates = total_interconnectors - with_coordinates
    
    # Log execution summary
    if 'log_execution_summary' in globals():
        log_execution_summary(
            logger,
            "enrich_interconnector_locations",
            start_time,
            inputs={'combined_data': input_file, 'neso_register': neso_register_file},
            outputs={'enriched_data': output_file},
            context={
                'total_interconnectors': total_interconnectors,
                'with_coordinates': with_coordinates,
                'missing_coordinates': missing_coordinates,
                'coordinate_coverage': f"{(with_coordinates/total_interconnectors*100):.1f}%" if total_interconnectors > 0 else "0%"
            }
        )
    
    logger.info("Interconnector location enrichment completed successfully")

if __name__ == "__main__":
    if 'snakemake' in globals():
        main()
    else:
        # For testing/standalone execution
        base_path = Path(__file__).parent.parent.parent
        input_file = base_path / "resources" / "interconnectors" / "interconnectors_combined.csv"
        neso_register_file = base_path / "data" / "interconnectors" / "neso_interconnector_register.csv"
        output_file = base_path / "resources" / "interconnectors" / "interconnectors_location_enriched.csv"
        
        enrich_interconnector_locations(str(input_file), str(neso_register_file), str(output_file))

