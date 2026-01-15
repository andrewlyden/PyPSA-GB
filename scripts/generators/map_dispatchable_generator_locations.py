"""
Dispatchable Generator Location Mapping for PyPSA-GB

This script creates a comprehensive mapping of dispatchable generator locations using:
1. REPD coordinates (direct X-coordinate, Y-coordinate)
2. TEC connection sites mapped to ETYS network nodes  
3. Power stations locations CSV as fallback
4. Geographic coordinate transformations and validation

The script outputs a single comprehensive CSV with all dispatchable generators
and their verified geographic locations, eliminating duplicate storage.

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
import warnings
import unicodedata
from pathlib import Path
import json
import re
import pypsa
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from difflib import get_close_matches
import geopandas as gpd
from shapely.geometry import Point
from rapidfuzz import process, fuzz
from sklearn.neighbors import NearestNeighbors
import time

# Suppress warnings
warnings.filterwarnings("ignore", message=".*import_components_from_dataframe.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("pypsa.network.io").setLevel(logging.ERROR)

# Configure logging
logger = None
try:
    from logging_config import setup_logging, log_dataframe_info
    logger = setup_logging("map_dispatchable_generator_locations")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_dataframe_info
        logger = setup_logging("map_dispatchable_generator_locations")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("map_dispatchable_generator_locations")
        def log_dataframe_info(df, logger, name: str):
            logger.info(f"{name} shape: {df.shape}")

# Define dispatchable technology mappings (from previous script)
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

class AutomatedGeocoder:
    """Automated geocoding service for unmapped generator locations."""
    
    def __init__(self, user_agent="pypsa-gb-geocoder", timeout=10, delay=1.0):
        """
        Initialize geocoder with rate limiting.
        
        Parameters
        ----------
        user_agent : str
            User agent string for Nominatim API
        timeout : int
            Request timeout in seconds
        delay : float
            Delay between requests in seconds (rate limiting)
        """
        self.geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        self.delay = delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def geocode_site(self, site_name: str, country="United Kingdom") -> Optional[Tuple[float, float]]:
        """
        Geocode a site name to coordinates.
        
        Parameters
        ----------
        site_name : str
            Name of the site to geocode
        country : str
            Country to constrain search
            
        Returns
        -------
        Optional[Tuple[float, float]]
            Latitude, longitude coordinates or None if failed
        """
        if not site_name or pd.isna(site_name):
            return None
            
        # Rate limit requests
        self._rate_limit()
        
        # Clean site name for geocoding
        clean_name = self._clean_site_name(site_name)
        
        # Try different search strategies
        search_queries = [
            f"{clean_name}, {country}",
            f"{clean_name} power station, {country}",
            f"{clean_name} power plant, {country}",
            f"{clean_name}, UK",
            clean_name
        ]
        
        for query in search_queries:
            try:
                logger.debug(f"Geocoding query: {query}")
                location = self.geolocator.geocode(query, exactly_one=True)
                
                if location:
                    lat, lon = location.latitude, location.longitude
                    
                    # Validate coordinates are in reasonable UK range
                    if self._validate_uk_coordinates(lat, lon):
                        logger.debug(f"Found coordinates for {site_name}: {lat:.4f}, {lon:.4f}")
                        return lat, lon
                    else:
                        logger.debug(f"Coordinates outside UK range for {site_name}: {lat:.4f}, {lon:.4f}")
                        
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Geocoding error for {site_name}: {e}")
                continue
                
        logger.debug(f"Failed to geocode: {site_name}")
        return None
    
    def _clean_site_name(self, site_name: str) -> str:
        """Clean site name for better geocoding results."""
        if not site_name:
            return ""
            
        # Remove common power station suffixes/prefixes that might confuse geocoding
        clean_name = site_name.strip()
        
        # Remove capacity information
        clean_name = re.sub(r'\s*\(\s*\d+\.?\d*\s*MW\s*\)', '', clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'\s*\d+\.?\d*\s*MW.*$', '', clean_name, flags=re.IGNORECASE)
        
        # Remove common suffixes
        suffixes_to_remove = [
            'Power Station', 'Power Plant', 'BESS', 'Battery', 'Storage',
            'Substation', 'Connection', 'Tertiary', 'Secondary', 'Primary',
            'CCGT', 'OCGT', 'CHP', 'Peaking Plant'
        ]
        
        for suffix in suffixes_to_remove:
            pattern = rf'\s*{re.escape(suffix)}\s*$'
            clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        clean_name = ' '.join(clean_name.split())
        
        return clean_name
    
    def _validate_uk_coordinates(self, lat: float, lon: float) -> bool:
        """Validate coordinates are within reasonable UK bounds."""
        # UK approximate bounds (expanded slightly for safety)
        uk_bounds = {
            'lat_min': 49.5,   # Scilly Isles
            'lat_max': 61.0,   # Shetland Islands
            'lon_min': -8.5,   # Northern Ireland
            'lon_max': 2.0     # East Anglia
        }
        
        return (uk_bounds['lat_min'] <= lat <= uk_bounds['lat_max'] and 
                uk_bounds['lon_min'] <= lon <= uk_bounds['lon_max'])
    
    def batch_geocode(self, unmapped_sites: List[str], progress_callback=None) -> Dict[str, Tuple[float, float]]:
        """
        Geocode multiple sites with progress tracking.
        
        Parameters
        ----------
        unmapped_sites : List[str]
            List of site names to geocode
        progress_callback : callable, optional
            Function to call with progress updates
            
        Returns
        -------
        Dict[str, Tuple[float, float]]
            Mapping of site names to coordinates
        """
        results = {}
        total_sites = len(unmapped_sites)
        
        logger.info(f"Starting batch geocoding of {total_sites} sites")
        
        for i, site_name in enumerate(unmapped_sites, 1):
            if progress_callback:
                progress_callback(i, total_sites, site_name)
            
            coordinates = self.geocode_site(site_name)
            if coordinates:
                results[site_name] = coordinates
                logger.info(f"Geocoded {site_name}: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
            else:
                logger.warning(f"Failed to geocode: {site_name}")
        
        success_rate = len(results) / total_sites * 100 if total_sites > 0 else 0
        logger.info(f"Geocoding complete. Success rate: {success_rate:.1f}% ({len(results)}/{total_sites})")
        
        return results


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

def load_dukes_coordinates(dukes_file: str = "data/generators/dukes_power_station_coordinates.csv") -> Dict[str, Tuple[float, float]]:
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

def normalize_technology_name(tech_name: str) -> str:
    """Normalize technology name for consistent mapping."""
    if pd.isna(tech_name):
        return "unknown"
    return normalize_text(tech_name)
    """Normalize technology name for consistent mapping."""
    if pd.isna(tech_name):
        return "unknown"
    return str(tech_name).strip()

def categorize_tec_technology(plant_type: str) -> Tuple[str, str]:
    """Categorize TEC plant type into technology category and specific type."""
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

def categorize_repd_technology(tech_type: str) -> Tuple[str, str]:
    """Categorize REPD technology type into category and specific type."""
    if pd.isna(tech_type):
        return "unknown", "unknown"
    
    tech_type = normalize_technology_name(tech_type)
    
    # Check exact matches
    if tech_type in STORAGE_SYSTEMS:
        return "storage", STORAGE_SYSTEMS[tech_type]
    
    if tech_type in BIOMASS_WASTE:
        return "biomass_waste", BIOMASS_WASTE[tech_type]
    
    if tech_type in LARGE_HYDRO:
        return "hydro", LARGE_HYDRO[tech_type]
    
    return "other", "unclassified"

def load_etys_network(network_file: str) -> pypsa.Network:
    """Load PyPSA ETYS network and extract bus coordinates."""
    logger.info(f"Loading ETYS network from {network_file}")
    
    try:
        network = pypsa.Network(network_file)
        logger.info(f"Loaded network with {len(network.buses)} buses")
        return network
    except Exception as e:
        logger.error(f"Failed to load network: {e}")
        raise

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

def create_connection_site_mapping(network: pypsa.Network) -> Dict[str, Tuple[float, float]]:
    """Create mapping from TEC connection sites to network bus coordinates."""
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

def enhanced_location_matching(site_name: str, connection_site: str, 
                              bus_mapping: Dict[str, Tuple[float, float]], 
                              fallback_mapping: Dict[str, Tuple[float, float]],
                              dukes_mapping: Dict[str, Tuple[float, float]],
                              network_buses_gdf: gpd.GeoDataFrame = None,
                              geocoder: AutomatedGeocoder = None) -> Tuple[Optional[Tuple[float, float]], str]:
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
    
    # Strategy 7: Spatial fallback using nearest bus (if coordinates available)
    if network_buses_gdf is not None and not network_buses_gdf.empty:
        # This would require having some approximate coordinates to start with
        # For now, skip this strategy as it needs more setup
        pass
    
    # Strategy 8: Automated geocoding (if enabled)
    if geocoder is not None and site_name:
        coordinates = geocoder.geocode_site(site_name)
        if coordinates:
            return coordinates, 'geocoded'
    
    return None, 'not_found'

def match_connection_site_to_coordinates(connection_site: str, 
                                       bus_mapping: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Match a TEC connection site to network bus coordinates (legacy function)."""
    if pd.isna(connection_site):
        return None
    
    normalized_site = normalize_connection_site_name(connection_site)
    if not normalized_site:
        return None
    
    # Direct match
    if normalized_site in bus_mapping:
        return bus_mapping[normalized_site]
    
    # Fuzzy matching
    possible_matches = get_close_matches(normalized_site, bus_mapping.keys(), n=1, cutoff=0.6)
    if possible_matches:
        best_match = possible_matches[0]
        logger.debug(f"Fuzzy matched '{connection_site}' to '{best_match}'")
        return bus_mapping[best_match]
    
    # Partial word matching
    site_words = set(normalized_site.split())
    for bus_name, coords in bus_mapping.items():
        bus_words = set(bus_name.split())
        if site_words.intersection(bus_words):
            logger.debug(f"Partial word matched '{connection_site}' to '{bus_name}'")
            return coords
    
    return None

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

def process_tec_generators(tec_file: str, network: pypsa.Network,
                          fallback_mapping: Dict[str, Tuple[float, float]],
                          dukes_mapping: Dict[str, Tuple[float, float]],
                          geocoder: AutomatedGeocoder = None) -> pd.DataFrame:
    """Process TEC register generators with enhanced location mapping."""
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
    
    # Create connection site mapping
    bus_mapping = create_connection_site_mapping(network)
    
    # Enhanced location mapping
    coordinates = []
    location_sources = []
    
    for _, row in tec_dispatchable.iterrows():
        connection_site = row.get('Connection Site', '')
        site_name = row.get('Project Name', '')
        
        # Use enhanced matching
        coords, source = enhanced_location_matching(
            site_name, connection_site, bus_mapping, fallback_mapping, dukes_mapping, geocoder=geocoder
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

def process_repd_generators(repd_file: str) -> pd.DataFrame:
    """Process REPD generators (already have coordinates)."""
    logger.info(f"Processing REPD generators from {repd_file}")
    
    repd_df = robust_csv_reader(repd_file)
    
    # Filter for dispatchable technologies
    dispatchable_techs = list(STORAGE_SYSTEMS.keys()) + list(BIOMASS_WASTE.keys()) + list(LARGE_HYDRO.keys())
    repd_dispatchable = repd_df[repd_df['Technology Type'].isin(dispatchable_techs)].copy()
    
    # Filter for operational sites only
    logger.info(f"Before operational filter: {len(repd_dispatchable)} sites")
    repd_dispatchable = repd_dispatchable[repd_dispatchable['Development Status'] == 'Operational'].copy()
    logger.info(f"After operational filter: {len(repd_dispatchable)} operational sites")
    
    # Add technology categorization
    repd_dispatchable[['category', 'technology']] = repd_dispatchable['Technology Type'].apply(
        lambda x: pd.Series(categorize_repd_technology(x))
    )
    
    # Standardize column names and add location source
    repd_dispatchable = repd_dispatchable.rename(columns={
        'Site Name': 'site_name',
        'Installed Capacity (MWelec)': 'capacity_mw',
        'Technology Type': 'plant_type',
        'Development Status': 'status',
        'Operator (or Applicant)': 'operator',
        'X-coordinate': 'x_coord',
        'Y-coordinate': 'y_coord'
    })
    
    # Add source identifiers
    repd_dispatchable['data_source'] = 'REPD'
    repd_dispatchable['location_source'] = 'repd_coordinates'
    repd_dispatchable['connection_site'] = np.nan  # REPD doesn't have connection sites
    
    logger.info(f"Processed {len(repd_dispatchable)} REPD dispatchable generators")
    
    return repd_dispatchable

def create_comprehensive_generator_database(tec_data: pd.DataFrame, repd_data: pd.DataFrame,
                                          output_file: str) -> pd.DataFrame:
    """Combine TEC and REPD data into comprehensive generator database."""
    logger.info("Creating comprehensive dispatchable generator database")
    
    # Ensure common columns exist
    common_cols = ['site_name', 'capacity_mw', 'technology', 'category', 'status', 'data_source', 
                   'x_coord', 'y_coord', 'location_source']
    
    # Add missing columns with NaN
    for col in common_cols:
        if col not in tec_data.columns:
            tec_data[col] = np.nan
        if col not in repd_data.columns:
            repd_data[col] = np.nan
    
    # Optional columns that may exist
    optional_cols = ['connection_site', 'operator']
    for col in optional_cols:
        if col not in tec_data.columns:
            tec_data[col] = np.nan
        if col not in repd_data.columns:
            repd_data[col] = np.nan
    
    # Combine datasets
    all_cols = common_cols + optional_cols
    combined_df = pd.concat([
        tec_data[all_cols],
        repd_data[all_cols]
    ], ignore_index=True, sort=False)
    
    # Clean capacity data
    combined_df['capacity_mw'] = pd.to_numeric(combined_df['capacity_mw'], errors='coerce')
    combined_df = combined_df.dropna(subset=['capacity_mw'])
    combined_df = combined_df[combined_df['capacity_mw'] > 0]
    
    # Sort by capacity (largest first)
    combined_df = combined_df.sort_values('capacity_mw', ascending=False)
    
    logger.info(f"Combined dataset: {len(combined_df)} total dispatchable sites")
    log_dataframe_info(combined_df, logger, "Combined dispatchable data")
    
    # Save to CSV
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved comprehensive generator database to {output_file}")
    
    return combined_df

def create_location_mapping_report(combined_df: pd.DataFrame, output_file: str) -> None:
    """Create detailed report on location mapping success."""
    
    report_lines = [
        "PyPSA-GB Dispatchable Generator Location Mapping Report",
        "=" * 60,
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Location Source Summary:",
        "-" * 25
    ]
    
    # Location source breakdown
    location_summary = combined_df.groupby(['data_source', 'location_source']).agg({
        'capacity_mw': ['count', 'sum']
    }).round(1)
    
    location_summary.columns = ['sites', 'total_mw']
    
    for (source, location), row in location_summary.iterrows():
        report_lines.append(f"{source} - {location:<20} {row['sites']:>3} sites, {row['total_mw']:>7.1f} MW")
    
    # Overall location success
    has_coords = combined_df[['x_coord', 'y_coord']].notna().all(axis=1)
    total_sites = len(combined_df)
    located_sites = has_coords.sum()
    location_rate = (located_sites / total_sites * 100) if total_sites > 0 else 0
    
    report_lines.extend([
        "",
        "Overall Location Mapping:",
        "-" * 25,
        f"Total sites: {total_sites}",
        f"Sites with coordinates: {located_sites}",
        f"Location success rate: {location_rate:.1f}%",
        "",
        "Technology Breakdown (with locations):",
        "-" * 35
    ])
    
    # Technology breakdown for located sites
    located_df = combined_df[has_coords]
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
    
    logger.info(f"Created location mapping report: {output_file}")

def main(tec_conventional_file: str = "resources/generators/tec_conventional_only.csv",
         repd_file: str = "data/renewables/repd-q2-jul-2025.csv",
         network_file: str = "resources/network/ETYS_base.nc",
         power_stations_file: str = "data/generators/power_stations_locations.csv",
         dukes_file: str = "data/generators/dukes_power_station_coordinates.csv",
         output_file: str = "resources/generators/dispatchable_generators_with_locations.csv",
         location_report: str = "resources/generators/location_mapping_report.txt",
         use_geocoding: bool = False,
         geocoding_batch_size: int = 10):
    """
    Main function to create comprehensive dispatchable generator database with locations.
    
    Parameters
    ----------
    use_geocoding : bool, default False
        Whether to use automated geocoding for unmapped sites
    geocoding_batch_size : int, default 10
        Maximum number of sites to geocode in one run (rate limiting)
    """
    logger.info("Starting dispatchable generator location mapping")
    
    # Initialize geocoder if requested
    geocoder = None
    if use_geocoding:
        try:
            geocoder = AutomatedGeocoder(user_agent="pypsa-gb-location-mapper", delay=1.5)
            logger.info("Geocoding service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize geocoder: {e}. Continuing without geocoding.")
            use_geocoding = False
    
    # Load network for bus coordinate mapping
    network = load_etys_network(network_file)
    
    # Load fallback data sources
    fallback_mapping = load_power_stations_fallback(power_stations_file)
    dukes_mapping = load_dukes_coordinates(dukes_file)
    
    # Process TEC generators with enhanced location mapping
    tec_data = process_tec_generators(tec_conventional_file, network, fallback_mapping, dukes_mapping, geocoder)
    
    # Process REPD generators (already have coordinates)
    repd_data = process_repd_generators(repd_file)
    
    # Create comprehensive database
    combined_df = create_comprehensive_generator_database(tec_data, repd_data, output_file)
    
    # Calculate final success rate
    has_coords = combined_df[['x_coord', 'y_coord']].notna().all(axis=1)
    total_sites = len(combined_df)
    located_sites = has_coords.sum()
    success_rate = (located_sites / total_sites * 100) if total_sites > 0 else 0
    
    # Assert 100% success or log remaining issues for manual review
    if success_rate < 100.0:
        missing_sites = combined_df[~has_coords]
        logger.warning(f"Still missing {len(missing_sites)} site locations:")
        for _, site in missing_sites.iterrows():
            logger.warning(f"  - {site.get('site_name', 'Unknown')} ({site.get('capacity_mw', 0):.1f} MW)")
        logger.info(f"Location success rate: {success_rate:.1f}%")
    else:
        logger.info("Location success rate: 100.0%")
    
    # Create location mapping report
    create_location_mapping_report(combined_df, location_report)
    
    logger.info("Dispatchable generator location mapping completed successfully")
    
    return {
        'total_sites': total_sites,
        'total_capacity_mw': combined_df['capacity_mw'].sum(),
        'located_sites': located_sites,
        'success_rate': success_rate,
        'output_file': output_file
    }

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    if 'snakemake' in globals():
        # Snakemake execution
        tec_file = snakemake.input.tec_conventional
        repd_file = snakemake.input.repd_file
        network_file = snakemake.input.network_file
        power_stations_file = snakemake.input.power_stations_file
        dukes_file = "data/generators/dukes_power_station_coordinates.csv"  # Static path
        output_file = snakemake.output.generators_with_locations
        location_report = snakemake.output.location_report
    else:
        # Command line execution
        tec_file = sys.argv[1] if len(sys.argv) > 1 else "resources/generators/tec_conventional_only.csv"
        repd_file = sys.argv[2] if len(sys.argv) > 2 else "data/renewables/repd-q2-jul-2025.csv"
        network_file = sys.argv[3] if len(sys.argv) > 3 else "resources/network/ETYS_base.nc"
        power_stations_file = sys.argv[4] if len(sys.argv) > 4 else "data/generators/power_stations_locations.csv"
        dukes_file = sys.argv[5] if len(sys.argv) > 5 else "data/generators/dukes_power_station_coordinates.csv"
        output_file = sys.argv[6] if len(sys.argv) > 6 else "resources/generators/dispatchable_generators_with_locations.csv"
        location_report = sys.argv[7] if len(sys.argv) > 7 else "resources/generators/location_mapping_report.txt"
    
    stats = main(tec_file, repd_file, network_file, power_stations_file, dukes_file, output_file, location_report)
    
    logger.info("Dispatchable Generator Location Mapping Summary:")
    logger.info("Total sites: %d", stats['total_sites'])
    logger.info("Total capacity: %.1f MW", stats['total_capacity_mw'])
    logger.info("Sites with locations: %d", stats['located_sites'])
    logger.info("Location success rate: %.1f%%", stats['success_rate'])

