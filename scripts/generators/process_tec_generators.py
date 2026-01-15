"""
Standalone TEC Generator Processing for PyPSA-GB

This script processes TEC register generators with enhanced location mapping
and technology categorization, creating a comprehensive processed TEC dataframe
for use throughout the PyPSA-GB workflow.

Key Features:
- Technology categorization (thermal, storage, hybrid)
- Enhanced location mapping using multiple data sources
- Comprehensive location source tracking
- Standardized output format for workflow integration

This script extracts the process_tec_generators functionality from
map_dispatchable_generator_locations.py as a standalone processing step.

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
import warnings
import unicodedata
from pathlib import Path
import re
import pypsa
from typing import Dict, List, Tuple, Optional
from difflib import get_close_matches
from rapidfuzz import process, fuzz
import time

# Suppress warnings
warnings.filterwarnings("ignore", message=".*import_components_from_dataframe.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("pypsa.network.io").setLevel(logging.ERROR)

# Configure logging
logger = None
try:
    from logging_config import setup_logging, log_dataframe_info
    logger = setup_logging("process_tec_generators")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_dataframe_info
        logger = setup_logging("process_tec_generators")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("process_tec_generators")
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

LARGE_HYDRO = {
    'Large Hydro': 'large_hydro'
}

MIXED_PLANT_PATTERNS = {
    'CCGT.*Energy Storage': 'ccgt_battery',
    'CCGT.*OCGT': 'ccgt_ocgt',
    'Energy Storage.*Gas': 'battery_gas',
    'Energy Storage.*Pump Storage': 'battery_pumped',
    'Demand.*Energy Storage': 'demand_response_storage'
}

def normalize_text(text: str) -> str:
    """Normalize text for robust matching."""
    if pd.isna(text):
        return ""
    
    # Convert to string and normalize unicode
    text = str(text).replace("\xa0", " ")  # Remove non-breaking spaces
    text = unicodedata.normalize("NFKC", text)
    
    # Convert to lowercase and remove punctuation
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def robust_csv_reader(file_path: str, encoding_list: List[str] = None, **kwargs) -> pd.DataFrame:
    """Read CSV with robust encoding detection and text normalization."""
    for encoding in ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            logger.info(f"Successfully loaded {file_path} with {encoding} encoding")
            
            # Normalize all text columns but preserve column names
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna("").astype(str).map(lambda s: s.replace("\xa0", " "))
                df[col] = df[col].map(lambda s: unicodedata.normalize("NFKC", s).strip())
            
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise UnicodeDecodeError(f"Could not decode {file_path} with any standard encoding")

def categorize_tec_technology(plant_type: str) -> Tuple[str, str]:
    """Categorize TEC plant type into technology category and specific type."""
    if pd.isna(plant_type):
        return "unknown", "unknown"
    
    plant_type = normalize_text(plant_type)
    
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
    
    # Check for partial matches
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

def load_etys_network(network_file: str) -> Optional[pypsa.Network]:
    """Load PyPSA ETYS network and extract bus coordinates. Returns None if file doesn't exist."""
    if not network_file or not Path(network_file).exists():
        logger.warning(f"Network file not available: {network_file}. Will use fallback data sources only.")
        return None
    
    logger.info(f"Loading ETYS network from {network_file}")
    
    try:
        network = pypsa.Network(network_file)
        logger.info(f"Loaded network with {len(network.buses)} buses")
        return network
    except Exception as e:
        logger.warning(f"Failed to load network: {e}. Will use fallback data sources only.")
        return None

def normalize_connection_site_name(site_name: str) -> str:
    """Normalize connection site names for matching."""
    if pd.isna(site_name):
        return ""
    
    # Start with basic text normalization
    normalized = normalize_text(site_name)
    
    # Remove common suffixes/prefixes specific to connection sites
    remove_terms = ['substation', 'gsp', 'connection node', 'node', 'switchgear']
    for term in remove_terms:
        normalized = normalized.replace(term, '').strip()
    
    # Remove voltage indicators
    normalized = re.sub(r'\d+kv', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\d+/\d+kv', '', normalized, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def create_connection_site_mapping(network: Optional[pypsa.Network]) -> Dict[str, Tuple[float, float]]:
    """Create mapping from TEC connection sites to network bus coordinates.
    
    Args:
        network: Optional PyPSA network. If None, returns empty mapping.
    
    Returns:
        Dictionary mapping normalized bus names to (x, y) coordinates.
    """
    if network is None:
        logger.info("No network provided - skipping bus coordinate mapping")
        return {}
    
    logger.info("Creating connection site to bus coordinate mapping")
    
    # Get all bus names and coordinates
    bus_mapping = {}
    
    for bus_id, bus_data in network.buses.iterrows():
        if pd.notna(bus_data['x']) and pd.notna(bus_data['y']):
            # Normalize bus name for matching
            normalized_bus = normalize_connection_site_name(bus_id)
            if normalized_bus:
                bus_mapping[normalized_bus] = (float(bus_data['x']), float(bus_data['y']))
    
    logger.info(f"Created mapping for {len(bus_mapping)} network buses")
    return bus_mapping

def load_dukes_coordinates(dukes_file: str) -> Dict[str, Tuple[float, float]]:
    """Load DUKES power station coordinates."""
    logger.info(f"Loading DUKES coordinate data from {dukes_file}")
    
    if not Path(dukes_file).exists():
        logger.warning(f"DUKES file not found: {dukes_file}")
        return {}
    
    try:
        dukes_df = robust_csv_reader(dukes_file)
        
        dukes_mapping = {}
        for _, row in dukes_df.iterrows():
            station_name = normalize_text(row.get('station_name', ''))
            x_coord = row.get('x_coordinate', np.nan)
            y_coord = row.get('y_coordinate', np.nan)
            
            if station_name and pd.notna(x_coord) and pd.notna(y_coord):
                dukes_mapping[station_name] = (float(x_coord), float(y_coord))
        
        logger.info(f"Loaded {len(dukes_mapping)} DUKES power station coordinates")
        return dukes_mapping
        
    except Exception as e:
        logger.error(f"Failed to load DUKES data: {e}")
        return {}

def load_power_stations_fallback(power_stations_file: str) -> Dict[str, Tuple[float, float]]:
    """Load power stations location data as fallback."""
    logger.info(f"Loading power stations fallback data from {power_stations_file}")
    
    if not Path(power_stations_file).exists():
        logger.warning(f"Power stations file not found: {power_stations_file}")
        return {}
    
    try:
        stations_df = robust_csv_reader(power_stations_file)
        
        # Clean up the geolocation column and extract coordinates
        fallback_mapping = {}
        
        for _, row in stations_df.iterrows():
            station_name = row.get('Station Name', '')
            geolocation = row.get('Geolocation', '')
            
            if station_name and geolocation:
                # Parse coordinates from geolocation string
                coord_match = re.search(r'([-\d.]+),\s*[^-\d]*?([-\d.]+)', str(geolocation))
                if coord_match:
                    try:
                        lat = float(coord_match.group(1))
                        lon = float(coord_match.group(2))
                        # Convert to same coordinate system as REPD (assuming BNG)
                        normalized_name = normalize_text(station_name)
                        if normalized_name:
                            fallback_mapping[normalized_name] = (lon, lat)  # Note: BNG format
                    except ValueError:
                        continue
        
        logger.info(f"Loaded {len(fallback_mapping)} power station locations")
        return fallback_mapping
        
    except Exception as e:
        logger.error(f"Failed to load power stations data: {e}")
        return {}

def enhanced_location_matching(site_name: str, connection_site: str, 
                              bus_mapping: Dict[str, Tuple[float, float]], 
                              fallback_mapping: Dict[str, Tuple[float, float]],
                              dukes_mapping: Dict[str, Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], str]:
    """Enhanced location matching with multiple fallback strategies."""
    
    normalized_site = normalize_text(site_name) if site_name else ""
    normalized_connection = normalize_connection_site_name(connection_site) if connection_site else ""
    
    # Strategy 1: Exact normalized connection site match
    if normalized_connection and normalized_connection in bus_mapping:
        return bus_mapping[normalized_connection], 'network_bus_exact'
    
    # Strategy 2: Exact normalized site name match in DUKES
    if normalized_site and normalized_site in dukes_mapping:
        return dukes_mapping[normalized_site], 'dukes_exact'
    
    # Strategy 3: Exact normalized site name match in fallback
    if normalized_site and normalized_site in fallback_mapping:
        return fallback_mapping[normalized_site], 'power_stations_exact'
    
    # Strategy 4: Fuzzy connection site matching (rapidfuzz)
    if normalized_connection:
        try:
            matches = process.extract(normalized_connection, list(bus_mapping.keys()), 
                                    scorer=fuzz.token_sort_ratio, limit=1)
            if matches and matches[0][1] >= 85:  # High threshold
                best_match = matches[0][0]
                return bus_mapping[best_match], 'network_bus_fuzzy'
        except Exception:
            pass
    
    # Strategy 5: Fuzzy site name matching in DUKES
    if normalized_site:
        try:
            matches = process.extract(normalized_site, list(dukes_mapping.keys()), 
                                    scorer=fuzz.token_sort_ratio, limit=1)
            if matches and matches[0][1] >= 85:
                best_match = matches[0][0]
                return dukes_mapping[best_match], 'dukes_fuzzy'
        except Exception:
            pass
    
    # Strategy 6: Single word matching
    if normalized_site:
        site_words = set(normalized_site.split())
        if site_words:
            # Check DUKES first (most reliable coordinates)
            for dukes_name, coords in dukes_mapping.items():
                dukes_words = set(dukes_name.split())
                if site_words.intersection(dukes_words):
                    return coords, 'dukes_word_match'
            
            # Check bus mapping
            for bus_name, coords in bus_mapping.items():
                bus_words = set(bus_name.split())
                if site_words.intersection(bus_words):
                    return coords, 'network_bus_word_match'
    
    return None, 'not_found'

def process_tec_generators(tec_file: str, network: Optional[pypsa.Network],
                          fallback_mapping: Dict[str, Tuple[float, float]],
                          dukes_mapping: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Process TEC register generators with enhanced location mapping.
    
    Args:
        network: Optional PyPSA network for bus coordinates. If None, uses fallback data only.
    """
    logger.info(f"Processing TEC generators from {tec_file}")
    
    tec_df = robust_csv_reader(tec_file)
    
    # Filter for built projects only (use actual column names)
    logger.info(f"Before built filter: {len(tec_df)} sites")
    tec_df = tec_df[tec_df['Project Status'] == 'Built'].copy()
    logger.info(f"After built filter: {len(tec_df)} built sites")
    
    # Add technology categorization
    tec_df[['category', 'technology']] = tec_df['Plant Type'].apply(
        lambda x: pd.Series(categorize_tec_technology(x))
    )
    
    # Filter for dispatchable categories
    dispatchable_categories = ['thermal', 'storage', 'hybrid']
    tec_dispatchable = tec_df[tec_df['category'].isin(dispatchable_categories)].copy()
    
    # Create connection site mapping (returns empty dict if network is None)
    bus_mapping = create_connection_site_mapping(network)
    
    # Enhanced location mapping
    coordinates = []
    location_sources = []
    
    for _, row in tec_dispatchable.iterrows():
        connection_site = row.get('Connection Site', '')
        site_name = row.get('Project Name', '')
        
        # Use enhanced matching
        coords, source = enhanced_location_matching(
            site_name, connection_site, bus_mapping, fallback_mapping, dukes_mapping
        )
        
        if coords:
            coordinates.append(coords)
            location_sources.append(source)
        else:
            coordinates.append((np.nan, np.nan))
            location_sources.append('not_found')
    
    # Add coordinate columns
    coord_array = np.array(coordinates)
    tec_dispatchable['x_coord'] = coord_array[:, 0]
    tec_dispatchable['y_coord'] = coord_array[:, 1]
    tec_dispatchable['location_source'] = location_sources
    
    # Standardize column names
    tec_dispatchable = tec_dispatchable.rename(columns={
        'Project Name': 'site_name',
        'MW Connected': 'capacity_mw',
        'Plant Type': 'plant_type',
        'Connection Site': 'connection_site',
        'Project Status': 'status',
        'Customer Name': 'operator'
    })
    
    # Add source identifier
    tec_dispatchable['data_source'] = 'TEC'
    
    logger.info(f"Processed {len(tec_dispatchable)} TEC dispatchable generators")
    
    # Log location mapping success
    found_locations = tec_dispatchable['location_source'] != 'not_found'
    success_rate = found_locations.mean() * 100
    logger.info(f"Location mapping success: {found_locations.sum()}/{len(tec_dispatchable)} ({success_rate:.1f}%)")
    
    return tec_dispatchable

def create_location_sources_report(tec_df: pd.DataFrame, output_file: str) -> None:
    """Create detailed report on TEC location mapping sources."""
    
    report_lines = [
        "PyPSA-GB TEC Generator Location Mapping Report",
        "=" * 50,
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Location Source Summary:",
        "-" * 25
    ]
    
    # Location source breakdown
    location_summary = tec_df.groupby('location_source').agg({
        'capacity_mw': ['count', 'sum']
    }).round(1)
    
    location_summary.columns = ['sites', 'total_mw']
    location_summary = location_summary.sort_values('total_mw', ascending=False)
    
    for source, row in location_summary.iterrows():
        report_lines.append(f"{source:<25} {row['sites']:>3} sites, {row['total_mw']:>7.1f} MW")
    
    # Overall location success
    has_coords = tec_df[['x_coord', 'y_coord']].notna().all(axis=1)
    total_sites = len(tec_df)
    located_sites = has_coords.sum()
    location_rate = (located_sites / total_sites * 100) if total_sites > 0 else 0
    
    report_lines.extend([
        "",
        "Overall Location Mapping:",
        "-" * 25,
        f"Total TEC sites: {total_sites}",
        f"Sites with coordinates: {located_sites}",
        f"Location success rate: {location_rate:.1f}%",
        "",
        "Technology Breakdown (with locations):",
        "-" * 35
    ])
    
    # Technology breakdown for located sites
    located_df = tec_df[has_coords]
    tech_summary = located_df.groupby('technology').agg({
        'capacity_mw': ['count', 'sum']
    }).round(1)
    
    tech_summary.columns = ['sites', 'total_mw']
    tech_summary = tech_summary.sort_values('total_mw', ascending=False)
    
    for tech, row in tech_summary.iterrows():
        if tech not in ['unknown', 'unclassified']:
            report_lines.append(f"{tech:<25} {row['sites']:>3} sites, {row['total_mw']:>7.1f} MW")
    
    # Write report
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Created TEC location sources report: {output_file}")

def main(tec_conventional_file: str,
         network_file: Optional[str],
         power_stations_file: str,
         dukes_file: str,
         processed_tec_output: str,
         location_report_output: str):
    """
    Process TEC generators with enhanced location mapping and technology categorization.
    
    This function creates the authoritative processed TEC register for use throughout
    the PyPSA-GB workflow, with comprehensive location mapping and standardized formatting.
    
    Args:
        network_file: Optional network file path. If None or doesn't exist, uses fallback data only.
    """
    logger.info("Starting TEC generator processing")
    
    # Load network for bus coordinate mapping (optional)
    network = load_etys_network(network_file) if network_file else None
    
    # Load fallback data sources
    fallback_mapping = load_power_stations_fallback(power_stations_file)
    dukes_mapping = load_dukes_coordinates(dukes_file)
    
    # Process TEC generators with enhanced location mapping
    tec_processed = process_tec_generators(tec_conventional_file, network, fallback_mapping, dukes_mapping)
    
    # Calculate final success rate
    has_coords = tec_processed[['x_coord', 'y_coord']].notna().all(axis=1)
    total_sites = len(tec_processed)
    located_sites = has_coords.sum()
    success_rate = (located_sites / total_sites * 100) if total_sites > 0 else 0
    
    # Log results
    if success_rate < 100.0:
        missing_sites = tec_processed[~has_coords]
        logger.warning(f"Still missing {len(missing_sites)} TEC site locations:")
        for _, site in missing_sites.iterrows():
            logger.warning(f"  - {site.get('site_name', 'Unknown')} ({site.get('capacity_mw', 0):.1f} MW)")
    
    logger.info(f"TEC processing success rate: {success_rate:.1f}%")
    
    # Save processed TEC data
    Path(processed_tec_output).parent.mkdir(parents=True, exist_ok=True)
    tec_processed.to_csv(processed_tec_output, index=False)
    logger.info(f"Saved processed TEC data to {processed_tec_output}")
    
    # Create location sources report
    create_location_sources_report(tec_processed, location_report_output)
    
    logger.info("TEC generator processing completed successfully")
    
    return {
        'total_sites': total_sites,
        'total_capacity_mw': tec_processed['capacity_mw'].sum(),
        'located_sites': located_sites,
        'success_rate': success_rate,
        'output_file': processed_tec_output
    }

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    if 'snakemake' in globals():
        # Snakemake execution
        tec_file = snakemake.input.tec_conventional
        network_file = snakemake.input.get('network_file', None)
        power_stations_file = snakemake.input.power_stations_file
        dukes_file = snakemake.input.dukes_file
        processed_output = snakemake.output.processed_tec
        location_report = snakemake.output.location_sources_report
    else:
        # Command line execution
        tec_file = sys.argv[1] if len(sys.argv) > 1 else "resources/generators/tec_conventional_only.csv"
        network_file = sys.argv[2] if len(sys.argv) > 2 else None
        power_stations_file = sys.argv[3] if len(sys.argv) > 3 else "data/generators/power_stations_locations.csv"
        dukes_file = sys.argv[4] if len(sys.argv) > 4 else "data/generators/dukes_power_station_coordinates.csv"
        processed_output = sys.argv[5] if len(sys.argv) > 5 else "resources/generators/tec_processed_complete.csv"
        location_report = sys.argv[6] if len(sys.argv) > 6 else "resources/generators/tec_location_mapping_sources.txt"
    
    stats = main(tec_file, network_file, power_stations_file, dukes_file, processed_output, location_report)
    
    logger.info("TEC Generator Processing Summary:")
    logger.info("Total sites: %d", stats['total_sites'])
    logger.info("Total capacity: %.1f MW", stats['total_capacity_mw'])
    logger.info("Sites with locations: %d", stats['located_sites'])
    logger.info("Location success rate: %.1f%%", stats['success_rate'])

