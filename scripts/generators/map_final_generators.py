#!/usr/bin/env python3
"""
Final generator mapping strategies to complete the database.

This script implements multiple strategies to map the remaining unmapped generators:
1. Connection site mapping using ETYS network data
2. Fuzzy matching with wider parameters 
3. Manual coordinate lookup for major power stations
4. Regional/area-based approximation for the remainder
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fuzzywuzzy import fuzz
import re
import time

# Set up logging - compatible with both standalone and Snakemake execution
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("map_final_generators")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

def load_etys_network_data():
    """Load ETYS network data for substation coordinates."""
    try:
        # Try to load the ETYS network data
        etys_file = Path("data/network/ETYS_2023_LOPF_Original.xlsx")
        if etys_file.exists():
            buses_df = pd.read_excel(etys_file, sheet_name='Buses')
            logger.info(f"Loaded {len(buses_df)} ETYS bus locations")
            return buses_df
        else:
            logger.warning("ETYS network file not found")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading ETYS data: {e}")
        return pd.DataFrame()

def load_renewable_database():
    """Load renewable sites database for additional location matching."""
    try:
        # Load REPD renewable sites database
        repd_file = Path("resources/renewable/renewable_sites_mapped.csv")
        if repd_file.exists():
            repd_df = pd.read_csv(repd_file)
            logger.info(f"Loaded {len(repd_df)} renewable sites for location matching")
            return repd_df
        else:
            logger.warning("REPD renewable sites file not found")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading renewable sites data: {e}")
        return pd.DataFrame()

def create_substation_mapping():
    """Create a mapping of substation names to coordinates."""
    substation_coords = {
        # Major power station substations - manual lookup
        "South Humber Bank 400kV Substation": (520000, 420000),  # Approximate Humber area
        "West Burton 400kV Substation": (480000, 386000),  # Near existing West Burton
        "Humber Refinery 400kV Substation": (520000, 414000),  # Near Immingham
        "Marchwood 400kV Substation": (440000, 111000),  # Near Southampton
        "Coryton South 400kV Substation": (568000, 184000),  # Essex, Thames Estuary
        "Spalding North 400kV Substation": (525000, 321000),  # Spalding area
        "Ratcliffe-On-Soar 400kV Substation": (450000, 329000),  # Near Nottingham
        "Blackhillock 275kV Substation": (343000, 850000),  # Scottish Highlands
        "Ferrybridge B 132kV Substation": (448000, 423000),  # West Yorkshire
        "Indian Queens 400kV Substation": (195000, 65000),  # Cornwall
        "Grangemouth Ineos GSP": (295000, 682000),  # Central Scotland
        "Capenhurst 275kV Substation": (345000, 375000),  # Cheshire
        "Bustleholm GSP": (403000, 295000),  # West Midlands
        "Monk Fryston 275kV Substation": (448000, 428000),  # West Yorkshire
        "Ocker Hill 275kV Substation": (396000, 299000),  # West Midlands
        "Rainhill GSP": (345000, 389000),  # Merseyside
        "Richborough 400kV Substation": (633000, 162000),  # Kent
        "Glenrothes GSP": (322000, 701000),  # Fife, Scotland
        "Washway Farm GSP": (386000, 393000),  # Greater Manchester
        "Creyke Beck GSP": (504000, 435000),  # East Yorkshire
        "Hatfield 33kV Substation": (465000, 205000),  # Hertfordshire
        "Roosecote 132kV Substation": (322000, 468000),  # Cumbria
        "Bustleholm 132kV Substation": (403000, 295000),  # West Midlands
        "Swansea North 132kV Substation": (265000, 195000),  # South Wales
        "Hunningley 132kV Substation": (435000, 411000),  # South Yorkshire
        "Abernethy GSP": (316000, 716000),  # Perth & Kinross
        "Beech Street/City Road Substation": (533000, 182000),  # London
        "Feckenham GSP": (400000, 261000),  # Worcestershire
        "Landulph GSP": (240000, 63000),  # Cornwall/Devon border
        "Bispham Primary 33kV Substation": (331000, 438000),  # Blackpool
        "Kearsley 132kV substation": (375000, 406000),  # Greater Manchester
        "Swansea North 400kV Substation": (265000, 195000),  # South Wales
        "Upperboat 132kV Substation": (308000, 184000),  # South Wales
        "Ratcliffe GSP": (452000, 329000),  # Nottinghamshire
        "Wrecclesham 33kV": (485000, 143000),  # Surrey
        "Meadowhead GSP": (295000, 682000),  # Central Scotland
        "Hawkers Hill": (385000, 124000),  # Hampshire
        "Carrington GSP (SP Manweb)": (375000, 390000),  # Greater Manchester
        "Lister Drive 132kV Substation": (338000, 392000),  # Liverpool
        "Cellarhead 132kV Substation": (395000, 345000),  # Staffordshire
        "Wymondley GSP": (520000, 233000),  # Hertfordshire
        "Willington 132kV Substation": (430000, 325000),  # Derbyshire
        "Northfleet East 132kV": (562000, 174000),  # Kent
        "Uskmouth 132kV Substation": (335000, 185000),  # South Wales
        "Bridgwater 400/132kV Substation": (330000, 135000),  # Somerset
    }
    
    return substation_coords

def apply_connection_site_mapping(unmapped_df):
    """Map generators based on their connection sites."""
    substation_coords = create_substation_mapping()
    mapped_count = 0
    
    for idx, row in unmapped_df.iterrows():
        connection_site = row.get('connection_site', '')
        if connection_site and connection_site in substation_coords:
            x_coord, y_coord = substation_coords[connection_site]
            unmapped_df.at[idx, 'x_coord'] = x_coord
            unmapped_df.at[idx, 'y_coord'] = y_coord
            unmapped_df.at[idx, 'location_source'] = 'connection_site_mapping'
            mapped_count += 1
            logger.info(f"Mapped '{row['site_name']}' via connection site: {connection_site}")
    
    logger.info(f"Connection site mapping completed: {mapped_count} generators mapped")
    return unmapped_df

def apply_fuzzy_matching_expanded(unmapped_df, repd_df):
    """Apply expanded fuzzy matching with lower thresholds for remaining generators."""
    mapped_count = 0
    
    for idx, row in unmapped_df.iterrows():
        if not pd.isna(row['x_coord']):  # Skip already mapped
            continue
            
        site_name = row['site_name']
        best_match = None
        best_score = 0
        
        # Extract key words from site name
        site_words = re.findall(r'\b[A-Za-z]{3,}\b', site_name.lower())
        
        for _, repd_row in repd_df.iterrows():
            repd_name = str(repd_row.get('site_name', ''))
            
            # Try different matching strategies
            scores = [
                fuzz.partial_ratio(site_name.lower(), repd_name.lower()),
                fuzz.token_sort_ratio(site_name.lower(), repd_name.lower()),
            ]
            
            # Check for significant word overlap
            repd_words = re.findall(r'\b[A-Za-z]{3,}\b', repd_name.lower())
            common_words = set(site_words) & set(repd_words)
            if len(common_words) > 0:
                word_score = len(common_words) / max(len(site_words), len(repd_words)) * 100
                scores.append(word_score)
            
            max_score = max(scores)
            if max_score > best_score and max_score >= 70:  # Lower threshold
                best_score = max_score
                best_match = repd_row
        
        if best_match is not None:
            unmapped_df.at[idx, 'x_coord'] = best_match.get('X-coordinate')
            unmapped_df.at[idx, 'y_coord'] = best_match.get('Y-coordinate')
            unmapped_df.at[idx, 'location_source'] = f'fuzzy_expanded_{best_score:.0f}'
            mapped_count += 1
            logger.info(f"Fuzzy mapped '{site_name}' to '{best_match.get('site_name')}' (score: {best_score:.0f})")
    
    logger.info(f"Expanded fuzzy matching completed: {mapped_count} generators mapped")
    return unmapped_df

def apply_regional_approximation(unmapped_df):
    """Apply regional approximation for remaining unmapped generators."""
    # Regional center coordinates for approximation
    regional_centers = {
        'london': (530000, 180000),
        'yorkshire': (445000, 420000),
        'scotland': (320000, 700000),
        'wales': (300000, 200000),
        'southwest': (250000, 100000),
        'northwest': (350000, 400000),
        'midlands': (400000, 300000),
        'northeast': (420000, 560000),
        'southeast': (550000, 150000),
    }
    
    mapped_count = 0
    
    for idx, row in unmapped_df.iterrows():
        if not pd.isna(row['x_coord']):  # Skip already mapped
            continue
        
        site_name = row['site_name'].lower()
        connection_site = str(row.get('connection_site', '')).lower()
        
        # Try to identify region from name or connection site
        region = None
        combined_text = f"{site_name} {connection_site}"
        
        if any(word in combined_text for word in ['london', 'city road']):
            region = 'london'
        elif any(word in combined_text for word in ['yorkshire', 'ferrybridge', 'monk fryston']):
            region = 'yorkshire'
        elif any(word in combined_text for word in ['scotland', 'grangemouth', 'blackhillock', 'abernethy', 'glenrothes']):
            region = 'scotland'
        elif any(word in combined_text for word in ['wales', 'swansea', 'upperboat', 'uskmouth']):
            region = 'wales'
        elif any(word in combined_text for word in ['cornwall', 'plymouth', 'indian queens']):
            region = 'southwest'
        elif any(word in combined_text for word in ['manchester', 'liverpool', 'carrington', 'rainhill', 'lister']):
            region = 'northwest'
        elif any(word in combined_text for word in ['midlands', 'bustleholm', 'ocker hill']):
            region = 'midlands'
        elif any(word in combined_text for word in ['kent', 'richborough', 'northfleet']):
            region = 'southeast'
        
        if region and region in regional_centers:
            x_coord, y_coord = regional_centers[region]
            # Add some random offset to avoid exact overlap
            x_offset = np.random.randint(-5000, 5000)
            y_offset = np.random.randint(-5000, 5000)
            
            unmapped_df.at[idx, 'x_coord'] = x_coord + x_offset
            unmapped_df.at[idx, 'y_coord'] = y_coord + y_offset
            unmapped_df.at[idx, 'location_source'] = f'regional_approx_{region}'
            mapped_count += 1
            logger.info(f"Regional approximation for '{row['site_name']}' in {region}")
    
    logger.info(f"Regional approximation completed: {mapped_count} generators mapped")
    return unmapped_df

def main():
    """Main function to apply all mapping strategies."""
    logger.info("Starting final generator mapping strategies")
    
    # Load data
    generators_df = pd.read_csv('resources/generators/dispatchable_generators_final.csv')
    repd_df = pd.read_csv('data/renewables/repd-q2-july-2024.csv')
    
    # Get unmapped generators
    unmapped_df = generators_df[generators_df['x_coord'].isna()].copy()
    logger.info(f"Starting with {len(unmapped_df)} unmapped generators ({unmapped_df['capacity_mw'].sum():.0f} MW)")
    
    # Strategy 1: Connection site mapping
    unmapped_df = apply_connection_site_mapping(unmapped_df)
    
def main():
    """Execute final generator mapping strategies."""
    start_time = time.time()
    logger.info("Starting final generator mapping process...")
    
    # Load generators needing mapping
    generators_df = pd.read_csv('resources/generators/dispatchable_generators_mapped.csv')
    unmapped_df = generators_df[generators_df['x_coord'].isna()].copy()
    
    logger.info(f"Found {len(unmapped_df)} unmapped generators with {unmapped_df['capacity_mw'].sum():.0f} MW capacity")
    
    # Load ETYS network and REPD data
    etys_df = load_etys_network_data()
    repd_df = load_renewable_database()
    
    # Strategy 1: Connection site mapping using ETYS
    if etys_df is not None:
        unmapped_df = apply_connection_site_mapping(unmapped_df, etys_df)
    
    # Strategy 2: Expanded fuzzy matching
    unmapped_df = apply_fuzzy_matching_expanded(unmapped_df, repd_df)
    
    # Strategy 3: Regional approximation for the rest
    unmapped_df = apply_regional_approximation(unmapped_df)
    
    # Update the main dataframe
    for idx, row in unmapped_df.iterrows():
        if not pd.isna(row['x_coord']):
            generators_df.at[idx, 'x_coord'] = row['x_coord'] 
            generators_df.at[idx, 'y_coord'] = row['y_coord']
            generators_df.at[idx, 'location_source'] = row['location_source']
    
    # Calculate final statistics
    mapped_after = generators_df.dropna(subset=['x_coord', 'y_coord'])
    success_rate = len(mapped_after) / len(generators_df) * 100
    
    logger.info(f"""
Final mapping results:
  Total generators: {len(generators_df)}
  Successfully mapped: {len(mapped_after)} ({success_rate:.1f}%)
  Total mapped capacity: {mapped_after['capacity_mw'].sum():.0f} MW
  Still unmapped: {len(generators_df) - len(mapped_after)} generators
""")
    
    # Save updated database
    output_file = 'resources/generators/dispatchable_generators_complete.csv'
    generators_df.to_csv(output_file, index=False)
    logger.info(f"Saved complete generator database to {output_file}")
    
    # Log execution summary
    execution_time = time.time() - start_time
    summary_stats = {
        'total_generators': len(generators_df),
        'mapped_generators': len(mapped_after),
        'success_rate_percent': success_rate,
        'total_capacity_mw': generators_df['capacity_mw'].sum(),
        'mapped_capacity_mw': mapped_after['capacity_mw'].sum(),
        'output_file': output_file
    }
    
    log_execution_summary(logger, "map_final_generators", execution_time, summary_stats)
    
    return generators_df

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

