"""
Power Stations Location Integration for PyPSA-GB

This script integrates the legacy power_stations_locations.csv database
to find coordinates for remaining unmapped generators after DUKES integration.

This serves as a final fallback location source using historical power station data.

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fuzzywuzzy import fuzz, process
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = None
try:
    from scripts.utilities.logging_config import setup_logging
    logger = setup_logging("power_stations_integration")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging
        logger = setup_logging("power_stations_integration")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("power_stations_integration")

def load_power_stations_database(power_stations_file: str = "data/generators/power_stations_locations.csv") -> pd.DataFrame:
    """
    Load and process the power stations location database.
    
    Args:
        power_stations_file: Path to the power stations CSV file
        
    Returns:
        Processed power stations DataFrame with coordinates
    """
    logger.info("Loading power stations location database")
    
    try:
        # Load power stations database with proper encoding
        power_stations_df = pd.read_csv(power_stations_file, encoding='latin-1')
        
        logger.info(f"Loaded {len(power_stations_df)} records from power stations database")
        
        # Clean up column names (remove any BOM or weird characters)
        power_stations_df.columns = power_stations_df.columns.str.strip()
        
        # Check for geolocation column
        geoloc_columns = [col for col in power_stations_df.columns if 'geolocation' in col.lower()]
        
        if not geoloc_columns:
            raise ValueError("No geolocation column found in power stations database")
        
        geoloc_col = geoloc_columns[0]
        logger.info(f"Using geolocation column: {geoloc_col}")
        
        # Filter to records with coordinates
        with_coords = power_stations_df[
            (power_stations_df['Station Name'].notna()) & 
            (power_stations_df[geoloc_col].notna())
        ].copy()
        
        logger.info(f"Found {len(with_coords)} records with station names and coordinates")
        
        # Parse coordinates from geolocation column
        coord_results = []
        for _, row in with_coords.iterrows():
            parsed = parse_geolocation(row[geoloc_col])
            coord_results.append(parsed)
        
        # Add parsed coordinates as new columns
        with_coords['parsed_coords'] = coord_results
        
        # Filter to successfully parsed coordinates
        valid_coords = with_coords[pd.Series(coord_results).notna()].copy()
        
        logger.info(f"Successfully parsed {len(valid_coords)} coordinate pairs")
        
        # Extract lat/lon from parsed coordinates
        if len(valid_coords) > 0:
            lat_lon_list = [coord for coord in coord_results if coord is not None]
            valid_coords = valid_coords.iloc[:len(lat_lon_list)].copy()
            valid_coords[['latitude', 'longitude']] = pd.DataFrame(lat_lon_list, index=valid_coords.index)
        
        # Clean station names for matching
        valid_coords['station_name_clean'] = valid_coords['Station Name'].apply(normalize_station_name_for_matching)
        
        # Add source information
        valid_coords['source'] = 'power_stations_db'
        
        return valid_coords
        
    except Exception as e:
        logger.error(f"Failed to load power stations database: {e}")
        raise

def parse_geolocation(geoloc_str: str) -> Optional[Tuple[float, float]]:
    """Parse latitude,longitude from various geolocation string formats."""
    if pd.isna(geoloc_str):
        return None
    
    try:
        # Remove any weird characters and normalize
        clean_str = str(geoloc_str).strip().replace('Â', '').replace('�', '').replace('"', '')
        
        # Remove quotes and extra whitespace
        clean_str = clean_str.strip('"\'').strip()
        
        # Split by comma and extract numbers
        if ',' in clean_str:
            parts = clean_str.split(',')
            if len(parts) >= 2:
                lat_str = parts[0].strip()
                lon_str = parts[1].strip()
                
                # Extract numbers from each part
                lat_match = re.search(r'-?\d+\.?\d*', lat_str)
                lon_match = re.search(r'-?\d+\.?\d*', lon_str)
                
                if lat_match and lon_match:
                    lat = float(lat_match.group())
                    lon = float(lon_match.group())
                    
                    # Basic validity check (UK coordinates)
                    if 49.0 <= lat <= 61.0 and -8.0 <= lon <= 2.0:
                        return (lat, lon)
        
        return None
        
    except Exception as e:
        return None

def normalize_station_name_for_matching(name: str) -> str:
    """Normalize station names for matching against generator database."""
    if pd.isna(name):
        return ""
    
    # Convert to lowercase
    normalized = str(name).lower().strip()
    
    # Remove common suffixes
    remove_terms = [
        'power station', 'power plant', 'generating station', 'generation station',
        'powerstation', 'powerplant', 'station', 'plant', 'works', 'site', 'facility',
        'ccgt', 'ocgt', 'chp', 'gt', '*'
    ]
    
    for term in remove_terms:
        normalized = normalized.replace(term, '').strip()
    
    # Remove location descriptors
    normalized = re.sub(r'\b(north|south|east|west|central|new|old|phase|unit)\b', '', normalized)
    
    # Remove numbers and special characters
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\d+', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def convert_lat_lon_to_british_national_grid(lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert WGS84 lat/lon coordinates to British National Grid (EPSG:27700).
    
    This is a simplified conversion. For production use, consider using pyproj
    for more accurate coordinate transformation.
    """
    try:
        import pyproj
        
        # Define coordinate systems
        wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 lat/lon
        bng = pyproj.CRS('EPSG:27700')   # British National Grid
        
        # Create transformer
        transformer = pyproj.Transformer.from_crs(wgs84, bng, always_xy=True)
        
        # Transform coordinates (note: pyproj expects lon, lat order)
        x, y = transformer.transform(lon, lat)
        
        return (x, y)
        
    except ImportError:
        logger.warning("pyproj not available, using approximate conversion")
        # Approximate conversion for UK (very rough estimate)
        # This is not accurate and should be replaced with proper transformation
        x = (lon + 2.0) * 100000  # Very rough approximation
        y = (lat - 49.0) * 100000  # Very rough approximation
        return (x, y)

def match_power_stations_to_unmapped_generators(power_stations_df: pd.DataFrame, 
                                              unmapped_df: pd.DataFrame,
                                              min_score: int = 80) -> pd.DataFrame:
    """
    Match power stations to unmapped generators using fuzzy string matching.
    
    Args:
        power_stations_df: Power stations database with coordinates
        unmapped_df: Unmapped generators needing coordinates
        min_score: Minimum fuzzy matching score (0-100)
        
    Returns:
        DataFrame with successful matches
    """
    logger.info("Matching power stations to unmapped generators")
    
    # Create lookup dictionary for power stations data
    stations_lookup = {}
    for _, row in power_stations_df.iterrows():
        clean_name = row['station_name_clean']
        if clean_name:
            stations_lookup[clean_name] = {
                'original_name': row['Station Name'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'fuel': row.get('Fuel', ''),
                'capacity_mw': row.get('Installed Capacity (MW)', 0),
                'company': row.get('Company Name', ''),
                'type': row.get('Type', '')
            }
    
    logger.info(f"Created lookup for {len(stations_lookup)} power stations")
    
    # Match against unmapped generators
    matches = []
    station_names = list(stations_lookup.keys())
    
    for idx, generator in unmapped_df.iterrows():
        site_name = generator.get('site_name', '')
        normalized_site = normalize_station_name_for_matching(site_name)
        
        if not normalized_site:
            continue
        
        # Direct match first
        if normalized_site in stations_lookup:
            match_data = stations_lookup[normalized_site]
            lat, lon = match_data['latitude'], match_data['longitude']
            
            # Convert to British National Grid
            x_coord, y_coord = convert_lat_lon_to_british_national_grid(lat, lon)
            
            matches.append({
                'generator_index': idx,
                'generator_name': site_name,
                'station_name': match_data['original_name'],
                'latitude': lat,
                'longitude': lon,
                'x_coordinate': x_coord,
                'y_coordinate': y_coord,
                'match_type': 'direct',
                'match_score': 100,
                'station_fuel': match_data['fuel'],
                'station_capacity': match_data['capacity_mw'],
                'station_company': match_data['company'],
                'station_type': match_data['type']
            })
            
            logger.info(f"Direct match: '{site_name}' -> '{match_data['original_name']}'")
            continue
        
        # Fuzzy matching
        best_match = process.extractOne(normalized_site, station_names, scorer=fuzz.token_sort_ratio)
        
        if best_match and best_match[1] >= min_score:
            match_name = best_match[0]
            match_score = best_match[1]
            match_data = stations_lookup[match_name]
            lat, lon = match_data['latitude'], match_data['longitude']
            
            # Convert to British National Grid
            x_coord, y_coord = convert_lat_lon_to_british_national_grid(lat, lon)
            
            matches.append({
                'generator_index': idx,
                'generator_name': site_name,
                'station_name': match_data['original_name'],
                'latitude': lat,
                'longitude': lon,
                'x_coordinate': x_coord,
                'y_coordinate': y_coord,
                'match_type': 'fuzzy',
                'match_score': match_score,
                'station_fuel': match_data['fuel'],
                'station_capacity': match_data['capacity_mw'],
                'station_company': match_data['company'],
                'station_type': match_data['type']
            })
            
            logger.info(f"Fuzzy match ({match_score}): '{site_name}' -> '{match_data['original_name']}'")
    
    matches_df = pd.DataFrame(matches)
    logger.info(f"Found {len(matches_df)} power station coordinate matches")
    
    return matches_df

def apply_power_station_coordinates(generators_file: str,
                                  station_matches: pd.DataFrame,
                                  output_file: str) -> pd.DataFrame:
    """
    Apply power station coordinates to the generator database.
    
    Args:
        generators_file: Path to generator database
        station_matches: Matched power station coordinates
        output_file: Path for updated generator database
        
    Returns:
        Updated generator DataFrame
    """
    logger.info("Applying power station coordinates to generator database")
    
    # Load generator database
    generators_df = pd.read_csv(generators_file)
    
    # Track original location count
    original_located = generators_df[generators_df[['x_coord', 'y_coord']].notna().all(axis=1)]
    logger.info(f"Original locations: {len(original_located)}/{len(generators_df)}")
    
    # Apply power station coordinates
    updated_generators = generators_df.copy()
    
    for _, match in station_matches.iterrows():
        gen_idx = match['generator_index']
        x_coord = match['x_coordinate']
        y_coord = match['y_coordinate']
        
        # Update coordinates (converted to British National Grid)
        updated_generators.loc[gen_idx, 'x_coord'] = x_coord
        updated_generators.loc[gen_idx, 'y_coord'] = y_coord
        updated_generators.loc[gen_idx, 'location_source'] = 'power_stations_db'
    
    # Save updated database
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    updated_generators.to_csv(output_file, index=False)
    
    # Calculate improvement
    updated_located = updated_generators[updated_generators[['x_coord', 'y_coord']].notna().all(axis=1)]
    improvement = len(updated_located) - len(original_located)
    new_success_rate = len(updated_located) / len(updated_generators) * 100
    
    logger.info(f"Power station coordinate integration results:")
    logger.info(f"  Original locations: {len(original_located)}/{len(generators_df)}")
    logger.info(f"  New locations added: {improvement}")
    logger.info(f"  Updated locations: {len(updated_located)}/{len(updated_generators)}")
    logger.info(f"  New success rate: {new_success_rate:.1f}%")
    
    return updated_generators

def main(generators_file: str = "resources/generators/dispatchable_generators_with_dukes_locations.csv",
         power_stations_file: str = "data/generators/power_stations_locations.csv",
         output_file: str = "resources/generators/dispatchable_generators_with_power_stations_locations.csv",
         matches_file: str = "resources/generators/power_station_location_matches.csv"):
    """
    Main function to integrate power station location data with generator database.
    """
    logger.info("Starting power stations location integration")
    
    # Load power stations database
    power_stations_df = load_power_stations_database(power_stations_file)
    
    # Load generator database and identify unmapped generators
    generators_df = pd.read_csv(generators_file)
    unmapped_mask = generators_df[['x_coord', 'y_coord']].isna().any(axis=1)
    unmapped_df = generators_df[unmapped_mask].copy()
    
    logger.info(f"Found {len(unmapped_df)} unmapped generators to match against power stations")
    
    # Match power stations to unmapped generators
    station_matches = match_power_stations_to_unmapped_generators(power_stations_df, unmapped_df)
    
    # Save matches for review
    if len(station_matches) > 0:
        Path(matches_file).parent.mkdir(parents=True, exist_ok=True)
        station_matches.to_csv(matches_file, index=False)
        logger.info(f"Saved {len(station_matches)} power station matches to {matches_file}")
        
        # Apply coordinates to generator database
        updated_generators = apply_power_station_coordinates(
            generators_file, station_matches, output_file
        )
        
        logger.info("Power station location integration completed successfully")
        
        return {
            'total_power_stations': len(power_stations_df),
            'station_matches_found': len(station_matches),
            'generators_updated': len(station_matches),
            'output_file': output_file,
            'matches_file': matches_file
        }
    else:
        logger.warning("No power station matches found")
        return {
            'total_power_stations': len(power_stations_df),
            'station_matches_found': 0,
            'generators_updated': 0
        }

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    try:
        # Snakemake execution
        generators_file = snakemake.input.generators_with_dukes_locations
        power_stations_file = snakemake.input.power_stations_database
        output_file = snakemake.output.generators_with_power_stations_locations
        matches_file = snakemake.output.power_station_matches
    except NameError:
        # Command line execution
        generators_file = sys.argv[1] if len(sys.argv) > 1 else "resources/generators/dispatchable_generators_with_dukes_locations.csv"
        power_stations_file = sys.argv[2] if len(sys.argv) > 2 else "data/generators/power_stations_locations.csv"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "resources/generators/dispatchable_generators_with_power_stations_locations.csv"
        matches_file = sys.argv[4] if len(sys.argv) > 4 else "resources/generators/power_station_location_matches.csv"
    
    stats = main(generators_file, power_stations_file, output_file, matches_file)
    
    logger.info("Power Station Integration Summary:")
    logger.info("Power stations loaded: %d", stats['total_power_stations'])
    logger.info("Generator matches found: %d", stats['station_matches_found'])
    logger.info("Generators updated: %d", stats['generators_updated'])

