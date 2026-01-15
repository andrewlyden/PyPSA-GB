"""
DUKES 5.11 Power Station Location Integration for PyPSA-GB

This script integrates the DUKES (Digest of UK Energy Statistics) 5.11 dataset
to enhance location mapping for dispatchable generators. DUKES provides comprehensive
coverage of UK power stations with high-quality coordinate data.

The script matches DUKES data against unmapped generators using intelligent name
matching and applies the coordinate data to improve location coverage.

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
    from logging_config import setup_logging
    logger = setup_logging("dukes_location_integration")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging
        logger = setup_logging("dukes_location_integration")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("dukes_location_integration")

def load_dukes_dataset(dukes_file: str = "data/generators/DUKES_5.11_2025.xlsx") -> pd.DataFrame:
    """
    Load and process the DUKES 5.11 power station dataset.
    
    Args:
        dukes_file: Path to the DUKES Excel file
        
    Returns:
        Processed DUKES DataFrame with coordinates
    """
    logger.info("Loading DUKES 5.11 power station dataset")
    
    try:
        # Load with correct header row (row 5)
        dukes_df = pd.read_excel(dukes_file, sheet_name='5.11 Full list', header=5)
        
        # Remove empty rows
        dukes_df = dukes_df.dropna(how='all')
        
        logger.info(f"Loaded {len(dukes_df)} records from DUKES dataset")
        
        # Check for required columns
        required_cols = ['Site Name', 'X-Coordinate', 'Y-Coordinate']
        missing_cols = [col for col in required_cols if col not in dukes_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter to records with coordinates
        with_coords = dukes_df[
            (dukes_df['X-Coordinate'].notna()) & 
            (dukes_df['Y-Coordinate'].notna())
        ].copy()
        
        logger.info(f"Found {len(with_coords)} records with coordinates ({len(with_coords)/len(dukes_df)*100:.1f}%)")
        
        # Clean site names for matching
        with_coords['site_name_clean'] = with_coords['Site Name'].apply(normalize_site_name_for_matching)
        
        # Add source information
        with_coords['source'] = 'dukes_5.11'
        with_coords['scraped_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        return with_coords
        
    except Exception as e:
        logger.error(f"Failed to load DUKES dataset: {e}")
        raise

def normalize_site_name_for_matching(name: str) -> str:
    """Normalize site names for matching against generator database."""
    if pd.isna(name):
        return ""
    
    # Convert to lowercase
    normalized = str(name).lower().strip()
    
    # Remove common suffixes
    remove_terms = [
        'power station', 'power plant', 'generating station', 'generation station',
        'powerstation', 'powerplant', 'station', 'plant', 'works', 'site', 'facility'
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

def match_dukes_to_unmapped_generators(dukes_df: pd.DataFrame, 
                                     unmapped_df: pd.DataFrame,
                                     min_score: int = 80) -> pd.DataFrame:
    """
    Match DUKES power stations to unmapped generators using fuzzy string matching.
    
    Args:
        dukes_df: DUKES dataset with coordinates
        unmapped_df: Unmapped generators needing coordinates
        min_score: Minimum fuzzy matching score (0-100)
        
    Returns:
        DataFrame with successful matches
    """
    logger.info("Matching DUKES data to unmapped generators")
    
    # Create lookup dictionary for DUKES data
    dukes_lookup = {}
    for _, row in dukes_df.iterrows():
        clean_name = row['site_name_clean']
        if clean_name:
            dukes_lookup[clean_name] = {
                'original_name': row['Site Name'],
                'x_coordinate': row['X-Coordinate'],
                'y_coordinate': row['Y-Coordinate'],
                'technology': row.get('Technology', ''),
                'capacity_mw': row.get('InstalledCapacity (MW)', 0),
                'company': row.get('Company Name [note 30]', ''),
                'country': row.get('Country', ''),
                'region': row.get('Region', '')
            }
    
    logger.info(f"Created lookup for {len(dukes_lookup)} DUKES stations")
    
    # Match against unmapped generators
    matches = []
    dukes_names = list(dukes_lookup.keys())
    
    for idx, generator in unmapped_df.iterrows():
        site_name = generator.get('site_name', '')
        normalized_site = normalize_site_name_for_matching(site_name)
        
        if not normalized_site:
            continue
        
        # Direct match first
        if normalized_site in dukes_lookup:
            match_data = dukes_lookup[normalized_site]
            matches.append({
                'generator_index': idx,
                'generator_name': site_name,
                'dukes_name': match_data['original_name'],
                'x_coordinate': match_data['x_coordinate'],
                'y_coordinate': match_data['y_coordinate'],
                'match_type': 'direct',
                'match_score': 100,
                'dukes_technology': match_data['technology'],
                'dukes_capacity': match_data['capacity_mw'],
                'dukes_company': match_data['company'],
                'dukes_country': match_data['country'],
                'dukes_region': match_data['region']
            })
            
            logger.info(f"Direct match: '{site_name}' -> '{match_data['original_name']}'")
            continue
        
        # Fuzzy matching
        best_match = process.extractOne(normalized_site, dukes_names, scorer=fuzz.token_sort_ratio)
        
        if best_match and best_match[1] >= min_score:
            match_name = best_match[0]
            match_score = best_match[1]
            match_data = dukes_lookup[match_name]
            
            matches.append({
                'generator_index': idx,
                'generator_name': site_name,
                'dukes_name': match_data['original_name'],
                'x_coordinate': match_data['x_coordinate'],
                'y_coordinate': match_data['y_coordinate'],
                'match_type': 'fuzzy',
                'match_score': match_score,
                'dukes_technology': match_data['technology'],
                'dukes_capacity': match_data['capacity_mw'],
                'dukes_company': match_data['company'],
                'dukes_country': match_data['country'],
                'dukes_region': match_data['region']
            })
            
            logger.info(f"Fuzzy match ({match_score}): '{site_name}' -> '{match_data['original_name']}'")
    
    matches_df = pd.DataFrame(matches)
    logger.info(f"Found {len(matches_df)} DUKES coordinate matches")
    
    return matches_df

def apply_dukes_coordinates_to_generators(generators_file: str,
                                        dukes_matches: pd.DataFrame,
                                        output_file: str) -> pd.DataFrame:
    """
    Apply DUKES coordinates to the generator database.
    
    Args:
        generators_file: Path to generator database
        dukes_matches: Matched DUKES coordinates
        output_file: Path for updated generator database
        
    Returns:
        Updated generator DataFrame
    """
    logger.info("Applying DUKES coordinates to generator database")
    
    # Load generator database
    generators_df = pd.read_csv(generators_file)
    
    # Track original location count
    original_located = generators_df[generators_df[['x_coord', 'y_coord']].notna().all(axis=1)]
    logger.info(f"Original locations: {len(original_located)}/{len(generators_df)}")
    
    # Apply DUKES coordinates
    updated_generators = generators_df.copy()
    
    for _, match in dukes_matches.iterrows():
        gen_idx = match['generator_index']
        x_coord = match['x_coordinate']
        y_coord = match['y_coordinate']
        
        # Update coordinates (DUKES uses British National Grid)
        updated_generators.loc[gen_idx, 'x_coord'] = x_coord
        updated_generators.loc[gen_idx, 'y_coord'] = y_coord
        updated_generators.loc[gen_idx, 'location_source'] = 'dukes_5.11'
    
    # Save updated database
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    updated_generators.to_csv(output_file, index=False)
    
    # Calculate improvement
    updated_located = updated_generators[updated_generators[['x_coord', 'y_coord']].notna().all(axis=1)]
    improvement = len(updated_located) - len(original_located)
    new_success_rate = len(updated_located) / len(updated_generators) * 100
    
    logger.info(f"DUKES coordinate integration results:")
    logger.info(f"  Original locations: {len(original_located)}/{len(generators_df)}")
    logger.info(f"  New locations added: {improvement}")
    logger.info(f"  Updated locations: {len(updated_located)}/{len(updated_generators)}")
    logger.info(f"  New success rate: {new_success_rate:.1f}%")
    
    return updated_generators

def create_dukes_coordinate_database(dukes_df: pd.DataFrame,
                                   output_file: str = "data/generators/dukes_power_station_coordinates.csv") -> None:
    """Create a persistent database of DUKES power station coordinates."""
    logger.info("Creating DUKES coordinate database")
    
    # Select relevant columns for coordinate database
    coord_data = dukes_df[[
        'Site Name', 'X-Coordinate', 'Y-Coordinate', 'Technology', 
        'InstalledCapacity (MW)', 'Company Name [note 30]', 'Country', 
        'Region', 'source', 'scraped_date'
    ]].copy()
    
    # Rename columns for consistency
    coord_data = coord_data.rename(columns={
        'Site Name': 'station_name',
        'X-Coordinate': 'x_coordinate',
        'Y-Coordinate': 'y_coordinate',
        'Technology': 'technology',
        'InstalledCapacity (MW)': 'capacity_mw',
        'Company Name [note 30]': 'company_name',
        'Country': 'country',
        'Region': 'region'
    })
    
    # Save to CSV
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    coord_data.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(coord_data)} DUKES power station coordinates to {output_file}")

def main(generators_file: str = "resources/generators/dispatchable_generators_with_wikipedia_locations.csv",
         dukes_file: str = "data/generators/DUKES_5.11_2025.xlsx",
         output_file: str = "resources/generators/dispatchable_generators_with_dukes_locations.csv",
         dukes_coords_file: str = "data/generators/dukes_power_station_coordinates.csv",
         matches_file: str = "resources/generators/dukes_location_matches.csv"):
    """
    Main function to integrate DUKES location data with generator database.
    """
    logger.info("Starting DUKES location integration")
    
    # Load DUKES dataset
    dukes_df = load_dukes_dataset(dukes_file)
    
    # Create persistent DUKES coordinate database
    create_dukes_coordinate_database(dukes_df, dukes_coords_file)
    
    # Load generator database and identify unmapped generators
    generators_df = pd.read_csv(generators_file)
    unmapped_mask = generators_df[['x_coord', 'y_coord']].isna().any(axis=1)
    unmapped_df = generators_df[unmapped_mask].copy()
    
    logger.info(f"Found {len(unmapped_df)} unmapped generators to match against DUKES")
    
    # Match DUKES data to unmapped generators
    dukes_matches = match_dukes_to_unmapped_generators(dukes_df, unmapped_df)
    
    # Save matches for review
    if len(dukes_matches) > 0:
        Path(matches_file).parent.mkdir(parents=True, exist_ok=True)
        dukes_matches.to_csv(matches_file, index=False)
        logger.info(f"Saved {len(dukes_matches)} DUKES matches to {matches_file}")
        
        # Apply coordinates to generator database
        updated_generators = apply_dukes_coordinates_to_generators(
            generators_file, dukes_matches, output_file
        )
        
        logger.info("DUKES location integration completed successfully")
        
        return {
            'total_dukes_stations': len(dukes_df),
            'dukes_matches_found': len(dukes_matches),
            'generators_updated': len(dukes_matches),
            'output_file': output_file,
            'dukes_database': dukes_coords_file,
            'matches_file': matches_file
        }
    else:
        logger.warning("No DUKES matches found")
        # Still create the coordinate database for future use
        return {
            'total_dukes_stations': len(dukes_df),
            'dukes_matches_found': 0,
            'generators_updated': 0,
            'dukes_database': dukes_coords_file
        }

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    try:
        # Snakemake execution
        generators_file = snakemake.input.generators_with_wikipedia_locations
        dukes_file = snakemake.input.dukes_dataset
        output_file = snakemake.output.generators_with_dukes_locations
        dukes_coords_file = snakemake.output.dukes_coordinate_database
        matches_file = snakemake.output.dukes_matches
    except NameError:
        # Command line execution
        generators_file = sys.argv[1] if len(sys.argv) > 1 else "resources/generators/dispatchable_generators_with_wikipedia_locations.csv"
        dukes_file = sys.argv[2] if len(sys.argv) > 2 else "data/generators/DUKES_5.11_2025.xlsx"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "resources/generators/dispatchable_generators_with_dukes_locations.csv"
        dukes_coords_file = sys.argv[4] if len(sys.argv) > 4 else "data/generators/dukes_power_station_coordinates.csv"
        matches_file = sys.argv[5] if len(sys.argv) > 5 else "resources/generators/dukes_location_matches.csv"
    
    stats = main(generators_file, dukes_file, output_file, dukes_coords_file, matches_file)
    
    logger.info("DUKES Integration Summary:")
    logger.info("DUKES stations loaded: %d", stats['total_dukes_stations'])
    logger.info("Generator matches found: %d", stats['dukes_matches_found'])
    logger.info("Generators updated: %d", stats['generators_updated'])

