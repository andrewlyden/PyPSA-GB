"""
Extract historical generator capacity data from DUKES 5.11

This script reads the DUKES (Digest of UK Energy Statistics) Excel file
and extracts year-specific generator capacity data, converting it to a
standardized CSV format compatible with FES data structure.

DUKES provides authoritative UK government data on energy generation capacity
from 2010-2024, with year-specific worksheets containing power station details.

Output format matches FES structure to enable seamless hybrid data integration:
  DUKES (thermal) + REPD (renewables) + FES (fallback for gaps)

Snakemake interface:
  - snakemake.input.dukes_file: Path to DUKES_5.11.xls
  - snakemake.output.dukes_generators: Path for output CSV
  - snakemake.params.dukes_year: Year to extract (2010-2024)
  - snakemake.log[0]: Log file path
  
DUKES Excel Format Notes:
  - 2010-2015: Header row 3, multi-line headers (4 rows), data starts row 7
  - 2016-2018: Header row 3, single-line headers, data starts row 4
  - 2019-2020: Header row 5, single-line headers, data starts row 6
  - 2021: Header row 4, single-line headers, data starts row 5
  - 2022-2024: Header row 5, single-line headers, data starts row 6
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np
from pyproj import Transformer
import requests
import time
import warnings

# Suppress pandas FutureWarnings about downcasting (comes from PyPSA io.py)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting.*')

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.postcode_geocoder import PostcodeGeocoder

# Setup logging with Snakemake log file path
logger = setup_logging(
    log_path=snakemake.log[0],
    log_level="INFO"
)


def get_dukes_sheet_params(year: int) -> dict:
    """
    Get the correct Excel parsing parameters for a specific DUKES year.
    
    Different years have different header structures in the DUKES Excel file.
    This function returns the correct skiprows and header handling for each year.
    
    Args:
        year: Year to extract (2010-2024)
        
    Returns:
        dict with 'skiprows' and 'has_multirow_header' keys
    """
    if 2010 <= year <= 2015:
        # Multi-row headers: Company Name/Station Name/Fuel in row 3
        # Capacity/(MW) in rows 4-5, data starts row 7
        return {'skiprows': 7, 'has_multirow_header': True, 'header_row': 3}
    elif 2016 <= year <= 2018:
        # Single-row header in row 3, data starts row 4
        return {'skiprows': 3, 'has_multirow_header': False, 'header_row': 3}
    elif year == 2021:
        # Anomaly year - header in row 4, data starts row 5
        return {'skiprows': 4, 'has_multirow_header': False, 'header_row': 4}
    else:
        # 2019-2020, 2022-2024: Header in row 5, data starts row 6
        return {'skiprows': 5, 'has_multirow_header': False, 'header_row': 5}

def read_dukes_worksheet(dukes_file: Path, year: int) -> pd.DataFrame:
    """
    Read DUKES Excel worksheet for a specific year using hybrid approach.
    
    Args:
        dukes_file: Path to DUKES_5.11_2025.xlsx
        year: Year to extract (2010-2024 for historical scenarios)
        
    Returns:
        DataFrame with raw DUKES data for the year, with coordinates filled from 5.11 Full list
        
    Raises:
        ValueError: If year worksheet doesn't exist
        
    Strategy:
        1. Load historical year sheet (e.g., "DUKES 2020") - contains ALL generators including retired
        2. Load "5.11 Full list" sheet - contains coordinates for current generators
        3. Match generators by name and fill in coordinates where possible
        4. Remaining generators without coordinates will be geocoded later based on location text
        
    This ensures we capture retired generators (missing from 5.11 Full list) while getting
    coordinates for generators that still exist.
    """
    logger.info(f"Reading DUKES data for year {year}")
    logger.info(f"Source file: {dukes_file}")
    
    # Get year-specific parsing parameters
    params = get_dukes_sheet_params(year)
    skiprows = params['skiprows']
    has_multirow = params['has_multirow_header']
    
    logger.info(f"Using skiprows={skiprows}, has_multirow_header={has_multirow}")
    
    # Load historical year sheet (has all generators including retired)
    historical_sheet = f"DUKES {year}"
    
    try:
        if has_multirow:
            # For 2010-2015: Multi-row headers need special handling
            # Read without skiprows first to get header structure
            df_raw = pd.read_excel(
                dukes_file,
                sheet_name=historical_sheet,
                engine='openpyxl',
                header=None
            )
            
            # Build column names from rows 3-6 (0-indexed: rows 3,4,5,6)
            header_parts = []
            for col_idx in range(df_raw.shape[1]):
                parts = []
                for row_idx in range(3, 7):  # Rows 3-6 contain header parts
                    if row_idx < len(df_raw):
                        val = df_raw.iloc[row_idx, col_idx]
                        if pd.notna(val) and str(val).strip():
                            parts.append(str(val).strip())
                # Combine parts into column name
                col_name = ' '.join(parts) if parts else f'Unnamed: {col_idx}'
                header_parts.append(col_name)
            
            # Extract data rows (starting from row 7)
            df_historical = df_raw.iloc[7:].copy()
            df_historical.columns = header_parts
            df_historical = df_historical.reset_index(drop=True)
            
            # Clean up column names to match expected format
            col_rename = {}
            for col in df_historical.columns:
                col_lower = col.lower()
                if 'company' in col_lower:
                    col_rename[col] = 'Company Name'
                elif 'station' in col_lower and 'name' in col_lower:
                    col_rename[col] = 'Station Name'
                elif 'fuel' in col_lower:
                    col_rename[col] = 'Fuel'
                elif 'capacity' in col_lower or '(mw)' in col_lower:
                    col_rename[col] = 'Installed Capacity\n(MW)'
                elif 'commission' in col_lower or 'year' in col_lower and 'generation' in col_lower:
                    col_rename[col] = 'Year of commission or year generation began'
                elif 'location' in col_lower or 'scotland' in col_lower or 'region' in col_lower:
                    col_rename[col] = 'Location\nScotland, Wales, Northern Ireland or English region'
            
            if col_rename:
                df_historical = df_historical.rename(columns=col_rename)
                logger.info(f"Renamed {len(col_rename)} columns to standard format")
        else:
            # Standard single-row header parsing
            df_historical = pd.read_excel(
                dukes_file,
                sheet_name=historical_sheet,
                engine='openpyxl',
                skiprows=skiprows
            )
        
        logger.info(f"Loaded historical sheet '{historical_sheet}' with {len(df_historical)} rows")
        logger.info(f"Columns: {list(df_historical.columns)}")
        
    except Exception as e:
        logger.error(f"Failed to read worksheet '{historical_sheet}': {e}")
        logger.error(f"Available sheets: Try opening {dukes_file} to verify worksheet names")
        raise ValueError(f"Could not read DUKES data for year {year}") from e
    
    # Load 5.11 Full list (has coordinates for current generators)
    try:
        df_full = pd.read_excel(
            dukes_file,
            sheet_name="5.11 Full list",
            engine='openpyxl',
            skiprows=5
        )
        logger.info(f"Loaded '5.11 Full list' with {len(df_full)} rows for coordinate matching")
        
        # Create coordinate lookup by station name with flexible matching
        coord_lookup = {}
        for idx, row in df_full.iterrows():
            site_name = str(row.get('Site Name', '')).strip()
            if not site_name or site_name == 'nan':
                continue
            
            # Create multiple keys for better matching
            site_name_lower = site_name.lower()
            site_name_clean = site_name_lower.replace(' ', '').replace('-', '').replace('_', '')
            
            x_coord = row.get('X-Coordinate')
            y_coord = row.get('Y-Coordinate')
            postcode = row.get('Postcode')
            
            if pd.notna(x_coord) or pd.notna(postcode):
                coord_data = {
                    'X-Coordinate': x_coord if pd.notna(x_coord) else None,
                    'Y-Coordinate': y_coord if pd.notna(y_coord) else None,
                    'Postcode': postcode if pd.notna(postcode) else None,
                    'original_name': site_name
                }
                # Store under multiple keys for flexible matching
                coord_lookup[site_name_lower] = coord_data
                coord_lookup[site_name_clean] = coord_data

        logger.info(f"Built coordinate lookup with {len(coord_lookup)} entries")

        # Add coordinate columns to historical data by matching station names
        df_historical['X-Coordinate'] = None
        df_historical['Y-Coordinate'] = None
        df_historical['Postcode'] = None

        matched_count = 0
        postcode_count = 0
        station_col = 'Station Name'

        if station_col in df_historical.columns:
            for idx, row in df_historical.iterrows():
                station = str(row[station_col]).strip()
                if not station or station == 'nan':
                    continue
                
                # Try multiple matching strategies
                station_lower = station.lower()
                station_clean = station_lower.replace(' ', '').replace('-', '').replace('_', '')
                
                match_found = None
                # Try exact match first (case-insensitive)
                if station_lower in coord_lookup:
                    match_found = coord_lookup[station_lower]
                # Try cleaned match (no spaces/hyphens)
                elif station_clean in coord_lookup:
                    match_found = coord_lookup[station_clean]
                # Try partial matches for common patterns
                else:
                    for key, data in coord_lookup.items():
                        # Match if one name contains the other (handles "Station A" vs "Station A Power Station")
                        if station_clean in key or key in station_clean:
                            if len(station_clean) > 5 and len(key) > 5:  # Avoid short spurious matches
                                match_found = data
                                break
                
                if match_found:
                    if match_found['X-Coordinate'] is not None:
                        df_historical.at[idx, 'X-Coordinate'] = match_found['X-Coordinate']
                        df_historical.at[idx, 'Y-Coordinate'] = match_found['Y-Coordinate']
                        matched_count += 1
                    if match_found['Postcode'] is not None:
                        df_historical.at[idx, 'Postcode'] = match_found['Postcode']
                        postcode_count += 1

        logger.info(f"Matched coordinates for {matched_count}/{len(df_historical)} generators from 5.11 Full list")
        logger.info(f"Matched postcodes for {postcode_count}/{len(df_historical)} generators from 5.11 Full list")
        logger.info(f"Generators without coordinates (will geocode): {len(df_historical) - matched_count}")

    except Exception as e:
        logger.warning(f"Could not load 5.11 Full list for coordinate matching: {e}")
        logger.warning("Proceeding with historical data only - coordinates will be geocoded")
        df_historical['X-Coordinate'] = None
        df_historical['Y-Coordinate'] = None
        df_historical['Postcode'] = None

    return df_historical
def geocode_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode generators without coordinates using region/location text.
    
    Provides approximate BNG coordinates for UK regions and countries.
    These are regional centroids - sufficient for bus mapping at transmission level.
    
    Args:
        df: DataFrame with location column and x_coord/y_coord columns (standardized names)
        
    Returns:
        DataFrame with estimated coordinates filled in for generators missing coordinates
    """
    # Regional centroids in British National Grid (EPSG:27700)
    # Source: Approximate geographic centers of UK regions
    REGION_COORDS = {
        # England regions
        'south west': (300000, 150000),      # Somerset/Devon area
        'south east': (480000, 160000),      # Surrey/Hampshire area  
        'london': (530000, 180000),          # Greater London
        'east': (590000, 250000),            # East Anglia
        'east midlands': (460000, 340000),   # Nottinghamshire area
        'west midlands': (390000, 280000),   # Birmingham area
        'yorkshire': (450000, 430000),       # Yorkshire area
        'yorkshire and humber': (450000, 430000),  # Alias
        'yorkshire and the humber': (450000, 430000),  # Alias
        'north west': (370000, 400000),      # Lancashire/Manchester area
        'north east': (420000, 550000),      # Tyne and Wear area
        
        # Countries
        'scotland': (280000, 680000),        # Central Scotland
        'wales': (290000, 280000),           # Mid Wales
        'northern ireland': (310000, 380000),  # Central NI (using Irish Grid approx)
        
        # Special cases
        'england': (420000, 280000),         # Central England
    }
    
    location_col = None
    for col in df.columns:
        if 'location' in col.lower() or 'region' in col.lower():
            location_col = col
            break
    
    if location_col is None:
        logger.warning("No location column found for geocoding")
        return df
    
    geocoded_count = 0
    
    for idx, row in df.iterrows():
        # Skip if already has coordinates
        if pd.notna(row.get('x_coord')) and pd.notna(row.get('y_coord')):
            continue
        
        location = str(row.get(location_col, '')).strip().lower()
        
        # Try to match region
        for region_name, (easting, northing) in REGION_COORDS.items():
            if region_name in location:
                df.at[idx, 'x_coord'] = easting
                df.at[idx, 'y_coord'] = northing
                geocoded_count += 1
                break
    
    if geocoded_count > 0:
        logger.info(f"Geocoded {geocoded_count} generators using regional centroids")

    return df


def geocode_thermal_generators_from_postcodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode thermal generators using postcode data (more reliable than DUKES X/Y coordinates).
    
    This function ONLY geocodes thermal generators (non-renewables) to avoid overwriting
    accurate REPD coordinates for renewable sites. Thermal generators often have incorrect
    coordinates in DUKES but have accurate postcodes.
    
    Args:
        df: DataFrame with 'postcode' column
        
    Returns:
        DataFrame with x_coord/y_coord updated from postcode geocoding for thermal generators
    """
    if 'postcode' not in df.columns:
        logger.warning("No 'postcode' column found - skipping postcode geocoding")
        return df
    
    # Identify thermal generators (everything except wind/solar/hydro/marine)
    renewable_fuels = ['Wind', 'Solar', 'Hydro', 'Tidal', 'Wave']
    is_thermal = True
    
    if 'fuel_type' in df.columns:
        # More specific filtering - only geocode non-renewable generators
        is_thermal = ~df['fuel_type'].str.contains('|'.join(renewable_fuels), case=False, na=False)
        thermal_count = is_thermal.sum()
        logger.info(f"Identified {thermal_count} thermal generators for postcode geocoding")
    else:
        logger.info("No fuel_type column - will geocode all generators with postcodes")
        is_thermal = pd.Series(True, index=df.index)
    
    # Filter to thermal generators with postcodes but missing/suspect coordinates
    needs_geocoding = (
        is_thermal & 
        df['postcode'].notna() &
        (df['x_coord'].isna() | df['y_coord'].isna() | True)  # Always use postcode for thermal
    )
    
    if not needs_geocoding.any():
        logger.info("No thermal generators need postcode geocoding")
        return df
    
    logger.info(f"Geocoding {needs_geocoding.sum()} thermal generators from postcodes...")
    logger.info("  (Using cached postcodes where available)")
    
    # Initialize postcode geocoder with cache
    geocoder = PostcodeGeocoder(cache_file="data/generators/postcode_cache.csv")
    
    # Geocode postcodes for thermal generators
    geocoded_count = 0
    for idx in df[needs_geocoding].index:
        postcode = df.loc[idx, 'postcode']
        result = geocoder.geocode(postcode)
        
        if result:
            # Update coordinates from postcode (OSGB36)
            df.loc[idx, 'x_coord'] = result['easting']
            df.loc[idx, 'y_coord'] = result['northing']
            geocoded_count += 1
        else:
            # If postcode geocoding fails, keep existing coordinates or mark as missing
            if pd.isna(df.loc[idx, 'x_coord']):
                logger.warning(f"  Could not geocode postcode '{postcode}' for {df.loc[idx, 'station_name']}")

    # Save updated cache
    geocoder._save_cache()
    
    logger.info(f"✓ Successfully geocoded {geocoded_count}/{needs_geocoding.sum()} thermal generators from postcodes")

    return df


def geocode_from_nominatim(df: pd.DataFrame, cache_file: str = "data/generators/nominatim_cache.csv") -> pd.DataFrame:
    """
    Geocode generators using OpenStreetMap Nominatim API as a fallback.
    
    This uses the free Nominatim API to find coordinates for power stations
    that don't have postcodes or DUKES coordinates. Rate limited to 1 req/sec.
    Results are cached to avoid repeated API calls.
    
    Args:
        df: DataFrame with 'station_name' column
        cache_file: Path to cache file for storing Nominatim results
        
    Returns:
        DataFrame with x_coord/y_coord/lat/lon updated from Nominatim geocoding
    """
    # Load cache if it exists
    cache_path = Path(cache_file)
    failure_cache_file = cache_file.replace('.csv', '_failures.txt')
    
    if cache_path.exists():
        cache_df = pd.read_csv(cache_file)
        logger.info(f"Loaded Nominatim cache with {len(cache_df)} entries")
        # Create lookup dictionary: station_name -> coordinates
        cache_lookup = {}
        for _, row in cache_df.iterrows():
            cache_lookup[row['station_name']] = {
                'lat': row['latitude'],
                'lon': row['longitude'],
                'easting': row['easting'],
                'northing': row['northing']
            }
    else:
        cache_df = pd.DataFrame(columns=['station_name', 'latitude', 'longitude', 'easting', 'northing'])
        cache_lookup = {}
        logger.info("No Nominatim cache found - will create new cache")
    
    # Load failure cache (stations that couldn't be geocoded)
    failed_stations = set()
    if Path(failure_cache_file).exists():
        with open(failure_cache_file, 'r') as f:
            failed_stations = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(failed_stations)} previously failed stations from cache")
    
    # Initialize lat/lon columns if they don't exist
    if 'lat' not in df.columns:
        df['lat'] = np.nan
    if 'lon' not in df.columns:
        df['lon'] = np.nan
    
    # Identify thermal generators (everything except wind/solar/hydro/marine)
    # This prevents overwriting accurate REPD coordinates for renewables
    renewable_fuels = ['Wind', 'Solar', 'Hydro', 'Tidal', 'Wave']
    is_thermal = True
    
    if 'fuel_type' in df.columns:
        is_thermal = ~df['fuel_type'].str.contains('|'.join(renewable_fuels), case=False, na=False)
        thermal_count = is_thermal.sum()
        logger.info(f"Identified {thermal_count} thermal generators for Nominatim geocoding")
    else:
        logger.info("No fuel_type column - will geocode all generators")
        is_thermal = pd.Series(True, index=df.index)
    
    # Find thermal generators still missing coordinates
    needs_geocoding = (
        is_thermal &
        (df['x_coord'].isna() | 
         df['y_coord'].isna() |
         df['lat'].isna() |
         df['lon'].isna())
    )
    
    if not needs_geocoding.any():
        logger.info("No thermal generators need Nominatim geocoding")
        return df
    
    logger.info(f"Geocoding {needs_geocoding.sum()} thermal generators using OpenStreetMap Nominatim API...")
    logger.info("  (Using cached results where available)")
    
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'PyPSA-GB/1.0 (academic research; andrew.lyden@ed.ac.uk)'
    }
    
    # Set up WGS84 -> BNG transformer for converting Nominatim results
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    
    geocoded_count = 0
    new_cache_entries = []
    new_failures = []
    
    for idx in df[needs_geocoding].index:
        station_name = df.loc[idx, 'station_name']
        
        # Check cache first
        if station_name in cache_lookup:
            cached = cache_lookup[station_name]
            df.loc[idx, 'lat'] = cached['lat']
            df.loc[idx, 'lon'] = cached['lon']
            df.loc[idx, 'x_coord'] = cached['easting']
            df.loc[idx, 'y_coord'] = cached['northing']
            geocoded_count += 1
            logger.info(f"  ✓ {station_name}: {cached['lat']:.6f}, {cached['lon']:.6f} (cached)")
            continue
        
        # Check failure cache - skip if previously failed
        if station_name in failed_stations:
            logger.debug(f"  ⊘ {station_name}: Skipping (previously failed)")
            continue
        
        # Try different query variations
        queries = [
            f"{station_name} Power Station, United Kingdom",
            f"{station_name} Nuclear Power Station, United Kingdom",
            f"{station_name} Power Plant, United Kingdom",
        ]
        
        found = False
        for query in queries:
            params = {
                'q': query,
                'format': 'json',
                'limit': 1
            }
            
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    results = response.json()
                    if results:
                        result = results[0]
                        lat = float(result['lat'])
                        lon = float(result['lon'])
                        
                        # Convert WGS84 to BNG
                        easting, northing = transformer.transform(lon, lat)
                        
                        # Update DataFrame
                        df.loc[idx, 'lat'] = lat
                        df.loc[idx, 'lon'] = lon
                        df.loc[idx, 'x_coord'] = easting
                        df.loc[idx, 'y_coord'] = northing
                        
                        # Add to cache
                        new_cache_entries.append({
                            'station_name': station_name,
                            'latitude': lat,
                            'longitude': lon,
                            'easting': easting,
                            'northing': northing
                        })
                        
                        geocoded_count += 1
                        logger.info(f"  ✓ {station_name}: {lat:.6f}, {lon:.6f} (new)")
                        found = True
                        break
                        
            except Exception as e:
                logger.warning(f"  Error geocoding {station_name}: {e}")
                continue
            
            time.sleep(1.1)  # Rate limit: 1 request per second
        
        if not found:
            logger.warning(f"  ✗ Could not geocode: {station_name}")
            new_failures.append(station_name)
        
        # Small delay between stations even if found
        time.sleep(1.1)
    
    # Save updated cache
    if new_cache_entries:
        new_entries_df = pd.DataFrame(new_cache_entries)
        updated_cache = pd.concat([cache_df, new_entries_df], ignore_index=True)
        
        # Ensure parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        updated_cache.to_csv(cache_file, index=False)
        logger.info(f"✓ Saved {len(new_cache_entries)} new entries to Nominatim cache: {cache_file}")
    
    # Save failure cache
    if new_failures:
        all_failures = failed_stations.union(set(new_failures))
        with open(failure_cache_file, 'w') as f:
            for station in sorted(all_failures):
                f.write(f"{station}\n")
        logger.info(f"✓ Saved {len(new_failures)} new failures to cache (total: {len(all_failures)})")
    
    logger.info(f"✓ Successfully geocoded {geocoded_count}/{needs_geocoding.sum()} thermal generators using Nominatim")
    
    return df


def geocode_from_location_name(df: pd.DataFrame, cache_file: str = "data/generators/nominatim_cache.csv") -> pd.DataFrame:
    """
    Final fallback geocoding using location field with partial name matching.
    
    For generators still missing coordinates after DUKES X/Y and Nominatim,
    this extracts location names from the 'location' field and tries geocoding
    with partial matches (e.g., "Scotland" → "Scotland, UK").
    
    Args:
        df: DataFrame with 'location' and 'station_name' columns
        cache_file: Path to Nominatim cache file (shared with geocode_from_nominatim)
        
    Returns:
        DataFrame with x_coord/y_coord/lat/lon updated from location-based geocoding
    """
    # Load cache to avoid duplicate queries
    cache_path = Path(cache_file)
    failure_cache_file = cache_file.replace('.csv', '_failures.txt')
    
    if cache_path.exists():
        cache_df = pd.read_csv(cache_file)
        cache_lookup = {}
        for _, row in cache_df.iterrows():
            cache_lookup[row['station_name']] = {
                'lat': row['latitude'],
                'lon': row['longitude'],
                'easting': row['easting'],
                'northing': row['northing']
            }
    else:
        cache_df = pd.DataFrame(columns=['station_name', 'latitude', 'longitude', 'easting', 'northing'])
        cache_lookup = {}
    
    # Load failure cache
    failed_stations = set()
    if Path(failure_cache_file).exists():
        with open(failure_cache_file, 'r') as f:
            failed_stations = set(line.strip() for line in f if line.strip())
    
    # Initialize columns if needed
    if 'lat' not in df.columns:
        df['lat'] = np.nan
    if 'lon' not in df.columns:
        df['lon'] = np.nan
    
    # Identify thermal generators still missing coordinates
    renewable_fuels = ['Wind', 'Solar', 'Hydro', 'Tidal', 'Wave']
    is_thermal = True
    
    if 'fuel_type' in df.columns:
        is_thermal = ~df['fuel_type'].str.contains('|'.join(renewable_fuels), case=False, na=False)
    else:
        is_thermal = pd.Series(True, index=df.index)
    
    needs_geocoding = (
        is_thermal &
        (df['x_coord'].isna() | df['y_coord'].isna() | df['lat'].isna() | df['lon'].isna())
    )
    
    if not needs_geocoding.any():
        logger.info("No thermal generators need location-based geocoding")
        return df
    
    logger.info(f"Final fallback: Geocoding {needs_geocoding.sum()} thermal generators using location field...")
    logger.info("  (Partial name matching for ambiguous locations)")
    
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'PyPSA-GB/1.0 (academic research; andrew.lyden@ed.ac.uk)'
    }
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    
    geocoded_count = 0
    new_cache_entries = []
    new_failures = []
    
    for idx in df[needs_geocoding].index:
        station_name = df.loc[idx, 'station_name']
        location = str(df.loc[idx, 'location']) if 'location' in df.columns else ''
        
        # Check cache first
        if station_name in cache_lookup:
            continue  # Already cached from earlier
        
        # Check failure cache
        if station_name in failed_stations:
            logger.debug(f"  ⊘ {station_name}: Skipping (previously failed)")
            continue
        
        # Try partial name matching with location context
        # Extract key location terms
        queries = []
        
        # Strategy 1: Use station name with UK
        queries.append(f"{station_name}, United Kingdom")
        
        # Strategy 2: If location is available, combine with station name
        if location and location != 'nan':
            # Clean location name (remove extra spaces, newlines)
            location_clean = ' '.join(location.split())
            queries.append(f"{station_name}, {location_clean}, United Kingdom")
            
            # Strategy 3: Try just the location (e.g., "West Midlands, UK")
            if len(location_clean) > 3:  # Avoid very short location names
                queries.append(f"{location_clean}, United Kingdom")
        
        found = False
        for query in queries:
            params = {
                'q': query,
                'format': 'json',
                'limit': 1
            }
            
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    results = response.json()
                    if results:
                        result = results[0]
                        lat = float(result['lat'])
                        lon = float(result['lon'])
                        
                        # Sanity check: ensure it's in UK (roughly 49-61°N, -8-2°E)
                        if 49 <= lat <= 61 and -8 <= lon <= 2:
                            # Convert WGS84 to BNG
                            easting, northing = transformer.transform(lon, lat)
                            
                            # Update DataFrame
                            df.loc[idx, 'lat'] = lat
                            df.loc[idx, 'lon'] = lon
                            df.loc[idx, 'x_coord'] = easting
                            df.loc[idx, 'y_coord'] = northing
                            
                            # Add to cache
                            new_cache_entries.append({
                                'station_name': station_name,
                                'latitude': lat,
                                'longitude': lon,
                                'easting': easting,
                                'northing': northing
                            })
                            
                            geocoded_count += 1
                            logger.info(f"  ✓ {station_name} → {location_clean if location != 'nan' else 'UK'}: {lat:.6f}, {lon:.6f} (location-based)")
                            found = True
                            break
                        else:
                            logger.warning(f"  ✗ {station_name}: Result outside UK bounds ({lat:.2f}, {lon:.2f})")
                        
            except Exception as e:
                logger.warning(f"  Error geocoding {station_name}: {e}")
                continue
            
            time.sleep(1.1)  # Rate limit
        
        if not found:
            logger.warning(f"  ✗ Could not geocode: {station_name} (location: {location})")
            new_failures.append(station_name)
        
        time.sleep(1.1)
    
    # Save updated cache
    if new_cache_entries:
        new_entries_df = pd.DataFrame(new_cache_entries)
        updated_cache = pd.concat([cache_df, new_entries_df], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        updated_cache.to_csv(cache_file, index=False)
        logger.info(f"✓ Saved {len(new_cache_entries)} new entries to Nominatim cache")
    
    # Save failure cache
    if new_failures:
        all_failures = failed_stations.union(set(new_failures))
        with open(failure_cache_file, 'w') as f:
            for station in sorted(all_failures):
                f.write(f"{station}\n")
        logger.info(f"✓ Saved {len(new_failures)} new failures to cache (total: {len(all_failures)})")
    
    logger.info(f"✓ Successfully geocoded {geocoded_count}/{needs_geocoding.sum()} thermal generators using location fallback")
    
    return df


def apply_dukes_coordinates_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply DUKES X-Coordinate/Y-Coordinate as final fallback for missing coordinates.
    
    DUKES X/Y coordinates are often inaccurate (e.g., Hunterston B was 30km off),
    so this is only used when all other geocoding methods have failed.
    
    The original DUKES coordinates were preserved in 'dukes_x_original' and 'dukes_y_original'
    before geocoding overwrote them.
    
    Args:
        df: DataFrame with dukes_x_original/dukes_y_original columns (preserved DUKES data)
        
    Returns:
        DataFrame with x_coord/y_coord filled from DUKES data where still missing
    """
    # Check if we have preserved DUKES coordinates
    if 'dukes_x_original' not in df.columns or 'dukes_y_original' not in df.columns:
        logger.info("No preserved DUKES coordinates found - skipping DUKES fallback")
        return df
    
    # Find generators still missing coordinates after all geocoding
    needs_coords = df['x_coord'].isna() | df['y_coord'].isna()
    
    if not needs_coords.any():
        logger.info("All generators have coordinates - no DUKES fallback needed")
        return df
    
    # Apply DUKES coordinates where missing
    filled_count = 0
    for idx in df[needs_coords].index:
        dukes_x = df.loc[idx, 'dukes_x_original']
        dukes_y = df.loc[idx, 'dukes_y_original']
        
        # Only fill if both DUKES coordinates are present and current coordinates are missing
        if pd.notna(dukes_x) and pd.notna(dukes_y):
            if pd.isna(df.loc[idx, 'x_coord']):
                df.loc[idx, 'x_coord'] = dukes_x
            if pd.isna(df.loc[idx, 'y_coord']):
                df.loc[idx, 'y_coord'] = dukes_y
            filled_count += 1
            
            # Log which generators got DUKES fallback coordinates (for debugging)
            station_name = df.loc[idx, 'station_name']
            logger.debug(f"  Applied DUKES fallback for: {station_name}")
    
    if filled_count > 0:
        logger.info(f"✓ Applied DUKES X/Y coordinates as fallback for {filled_count}/{needs_coords.sum()} generators")
        logger.warning(f"  ⚠ Note: DUKES coordinates may be inaccurate (e.g., Hunterston B was 30km off)")
        logger.info(f"  Remaining generators without coordinates: {needs_coords.sum() - filled_count}")
    else:
        logger.info(f"No valid DUKES X/Y coordinates available for {needs_coords.sum()} generators missing coordinates")
    
    # Clean up temporary columns
    df = df.drop(columns=['dukes_x_original', 'dukes_y_original'], errors='ignore')
    
    return df


def convert_bng_to_wgs84(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert British National Grid (BNG) coordinates to WGS84 lat/lon.
    
    Args:
        df: DataFrame with 'x_coord' (Easting) and 'y_coord' (Northing) columns
        
    Returns:
        DataFrame with 'lat' and 'lon' columns added
    """
    if 'x_coord' not in df.columns or 'y_coord' not in df.columns:
        logger.warning("No x_coord/y_coord columns found for BNG conversion")
        return df
    
    # Find rows with valid BNG coordinates
    valid_bng = df['x_coord'].notna() & df['y_coord'].notna()
    
    if not valid_bng.any():
        logger.warning("No valid BNG coordinates found")
        df['lat'] = None
        df['lon'] = None
        return df
    
    logger.info(f"Converting {valid_bng.sum()} BNG coordinates to WGS84 (lat/lon)")
    
    # Set up coordinate transformation: EPSG:27700 (BNG) -> EPSG:4326 (WGS84)
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    
    # Initialize lat/lon columns
    df['lat'] = np.nan
    df['lon'] = np.nan
    
    # Convert coordinates
    eastings = df.loc[valid_bng, 'x_coord'].astype(float)
    northings = df.loc[valid_bng, 'y_coord'].astype(float)
    
    lons, lats = transformer.transform(eastings, northings)
    
    df.loc[valid_bng, 'lon'] = lons
    df.loc[valid_bng, 'lat'] = lats
    
    logger.info(f"✓ BNG conversion complete: {valid_bng.sum()} coordinates converted")
    logger.info(f"  Sample: Easting {eastings.iloc[0]:.0f}, Northing {northings.iloc[0]:.0f} → {lats[0]:.5f}°N, {lons[0]:.5f}°W")
    
    return df


def _refine_gas_fuel_type(df: pd.DataFrame, year: int) -> None:
    """
    Refine fuel_type for gas generators to distinguish CCGT from OCGT.
    
    DUKES changed format over the years:
      - 2010-2018: 'Fuel' column already has granular values like 'CCGT', 'OCGT', 'Gas'
      - 2019-2021: 'Fuel' = 'Natural Gas/gas', 'Technology' = 'CCGT' or 'OCGT'
      - 2022-2024: 'Primary Fuel' = 'Natural Gas', 'Technology' = 'Fossil Fuel',
                   'Type' (-> generator_type) = 'CCGT', 'Single cycle', etc.
    
    This function checks whether the fuel_type column has lost the CCGT/OCGT
    distinction (i.e. all gas plants labelled as 'Natural Gas') and uses
    the technology or generator_type columns to restore it.
    
    Modifies df in-place.
    
    Args:
        df: Standardized DUKES DataFrame with fuel_type, technology, and
            optionally generator_type columns
        year: DUKES data year
    """
    if 'fuel_type' not in df.columns:
        return
    
    fuel_lower = df['fuel_type'].str.lower().str.strip()
    is_generic_gas = fuel_lower.isin(['natural gas', 'natural gas/gas', 'gas', 'sour gas'])
    
    if not is_generic_gas.any():
        # fuel_type already has specific values (e.g. 2010-2018 format) - no refinement needed
        logger.info(f"  DUKES {year}: fuel_type already distinguishes gas types - no refinement needed")
        return
    
    gas_count = is_generic_gas.sum()
    refined_count = 0
    
    # Strategy 1: Use generator_type column (2022+ has 'Type' -> 'generator_type')
    # This contains 'CCGT', 'Single cycle', 'Conventional steam', etc.
    if 'generator_type' in df.columns and df.loc[is_generic_gas, 'generator_type'].notna().any():
        gen_type_lower = df['generator_type'].str.lower().str.strip()
        
        # Map generator_type values to refined fuel_type
        ocgt_mask = is_generic_gas & gen_type_lower.isin(['single cycle', 'ocgt', 'open cycle gas turbine'])
        ccgt_mask = is_generic_gas & gen_type_lower.isin(['ccgt', 'combined cycle gas turbine'])
        
        if ocgt_mask.any():
            df.loc[ocgt_mask, 'fuel_type'] = 'OCGT'
            refined_count += ocgt_mask.sum()
            logger.info(f"  DUKES {year}: Refined {ocgt_mask.sum()} 'Natural Gas' generators to 'OCGT' "
                       f"(from generator_type='Single cycle')")
        if ccgt_mask.any():
            df.loc[ccgt_mask, 'fuel_type'] = 'CCGT'
            refined_count += ccgt_mask.sum()
            logger.info(f"  DUKES {year}: Refined {ccgt_mask.sum()} 'Natural Gas' generators to 'CCGT' "
                       f"(from generator_type='CCGT')")
    
    # Strategy 2: Use technology column (2019-2021 has 'Technology' -> 'technology')
    # This contains 'CCGT', 'OCGT', etc. directly
    # Re-check is_generic_gas in case strategy 1 already refined some
    fuel_lower = df['fuel_type'].str.lower().str.strip()
    is_still_generic = fuel_lower.isin(['natural gas', 'natural gas/gas', 'gas', 'sour gas'])
    
    if 'technology' in df.columns and is_still_generic.any() and df.loc[is_still_generic, 'technology'].notna().any():
        tech_lower = df['technology'].str.lower().str.strip()
        
        ocgt_mask = is_still_generic & tech_lower.isin(['ocgt', 'open cycle gas turbine', 'single cycle'])
        ccgt_mask = is_still_generic & tech_lower.isin(['ccgt', 'combined cycle gas turbine'])
        
        if ocgt_mask.any():
            df.loc[ocgt_mask, 'fuel_type'] = 'OCGT'
            refined_count += ocgt_mask.sum()
            logger.info(f"  DUKES {year}: Refined {ocgt_mask.sum()} 'Natural Gas' generators to 'OCGT' "
                       f"(from technology column)")
        if ccgt_mask.any():
            df.loc[ccgt_mask, 'fuel_type'] = 'CCGT'
            refined_count += ccgt_mask.sum()
            logger.info(f"  DUKES {year}: Refined {ccgt_mask.sum()} 'Natural Gas' generators to 'CCGT' "
                       f"(from technology column)")
    
    # Log summary
    remaining_generic = df['fuel_type'].str.lower().str.strip().isin(['natural gas', 'natural gas/gas', 'gas', 'sour gas']).sum()
    if remaining_generic > 0:
        logger.warning(f"  DUKES {year}: {remaining_generic} gas generators still have generic fuel_type "
                      f"(no type info available) - will default to CCGT in carrier mapping")
    if refined_count > 0:
        logger.info(f"  DUKES {year}: Refined {refined_count}/{gas_count} gas generators (CCGT/OCGT distinction)")


def standardize_dukes_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Convert raw DUKES data to standardized format matching FES structure.
    
    Args:
        df: Raw DUKES DataFrame
        year: Year being processed
        
    Returns:
        Standardized DataFrame ready for generator integration
        
    Expected DUKES columns (may vary by year):
        - Station Name / Power Station
        - Fuel Type / Primary Fuel
        - Capacity (MW) / Installed Capacity
        - Technology / Generator Type
        - Location / Region
        - X / Longitude (if available)
        - Y / Latitude (if available)
    """
    logger.info(f"Standardizing DUKES {year} data")
    logger.info(f"Raw columns: {list(df.columns)}")
    
    # Create standardized output dataframe
    standardized = pd.DataFrame()
    
    # Map DUKES columns to standard format
    # Note: Column names may vary across years - add flexible mapping
    column_mappings = {
        # Station identification
        'station_name': ['Site Name', 'Station Name', 'Power Station', 'Name'],
        'fuel_type': ['Primary Fuel', 'Fuel Type', 'Fuel'],
        'capacity_mw': ['InstalledCapacity (MW)', 'Installed Capacity\n(MW)', 'Capacity (MW)', 'Installed Capacity (MW)', 'Capacity MW', 'Capacity'],
        'technology': ['Technology', 'Generator Type'],
        'generator_type': ['Type'],  # 2022+ has separate 'Type' column (CCGT, Single cycle, etc.)
        'location': ['Region', 'Location\nScotland, Wales, Northern Ireland or English region', 'Location', 'Area', 'Country'],
        'postcode': ['Postcode', 'Post Code', 'Postal Code'],  # Added for postcode geocoding
        'x_coord': ['X-Coordinate', 'X', 'Longitude', 'Long', 'Easting'],
        'y_coord': ['Y-Coordinate', 'Y', 'Latitude', 'Lat', 'Northing'],
        'status': ['Year Commissioned', 'Status', 'Operational Status', 'Commissioned', 'Year of commission or year generation began'],
        'owner': ['Company Name [note 30]', 'Company Name', 'Owner', 'Operator', 'Company']
    }
    
    def find_column(possible_names):
        """Find first matching column name from list of possibilities"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    # Map each standard field
    for std_name, possible_cols in column_mappings.items():
        matched_col = find_column(possible_cols)
        if matched_col:
            standardized[std_name] = df[matched_col]
            logger.info(f"  Mapped '{matched_col}' → '{std_name}'")
        else:
            logger.warning(f"  Could not find column for '{std_name}' (tried: {possible_cols})")
            standardized[std_name] = None
    
    # === REFINE fuel_type FOR GAS GENERATORS (CCGT vs OCGT) ===
    # DUKES changed format over the years:
    #   2010-2018: 'Fuel' column is granular (CCGT, OCGT, Gas, etc.) - already correct
    #   2019-2021: 'Fuel' = 'Natural Gas/gas', 'Technology' = 'CCGT' or 'OCGT'
    #   2022-2024: 'Primary Fuel' = 'Natural Gas', 'Type' = 'CCGT' or 'Single cycle'
    # We need to use the technology/generator_type columns to distinguish CCGT from OCGT
    # when fuel_type is just 'Natural Gas' or 'Natural gas'
    _refine_gas_fuel_type(standardized, year)

    # Add metadata columns
    standardized['data_source'] = 'DUKES'
    standardized['data_year'] = year
    standardized['data_file'] = 'DUKES_5.11_2025.xlsx'
    
    # Filter for operational generators
    # Note: 'status' column contains commission year, not operational status text
    # All generators in DUKES are operational (retired ones are not included)
    # So we skip status filtering - just check for commission year if available
    if 'status' in standardized.columns and standardized['status'].notna().any():
        # If status is year, keep generators commissioned before or in target year
        try:
            standardized['commission_year'] = pd.to_numeric(standardized['status'], errors='coerce')
            commissioned_mask = (standardized['commission_year'] <= year) | standardized['commission_year'].isna()
            before_count = len(standardized)
            standardized = standardized[commissioned_mask]
            logger.info(f"  Filtered to commissioned generators: {len(standardized)}/{before_count}")
            standardized = standardized.drop(columns=['commission_year'])
        except Exception as e:
            logger.warning(f"  Could not filter by commission year: {e}")
    
    # Clean and validate data
    # Remove rows with missing critical fields
    standardized = standardized.dropna(subset=['station_name', 'capacity_mw'])
    
    # Convert capacity to numeric
    standardized['capacity_mw'] = pd.to_numeric(standardized['capacity_mw'], errors='coerce')
    
    # Remove zero/negative capacities
    standardized = standardized[standardized['capacity_mw'] > 0]
    
    # Filter out aggregated sites (sites with no specific location)
    # These are grouped entries like "Sites with capacity < 20 MW" that can't be geocoded
    before_filter = len(standardized)
    aggregated_mask = standardized['station_name'].str.contains(
        'Sites with capacity|sites with capacity|Aggregated|aggregated|Total|total',
        case=False,
        na=False
    )
    if aggregated_mask.any():
        aggregated_capacity = standardized[aggregated_mask]['capacity_mw'].sum()
        standardized = standardized[~aggregated_mask]
        logger.info(f"  Filtered out {aggregated_mask.sum()} aggregated sites ({aggregated_capacity:.1f} MW)")
        logger.info(f"  (Aggregated sites have no specific location and cannot be geocoded)")
    
    logger.info(f"Final standardized data: {len(standardized)} generators")
    logger.info(f"  Total capacity: {standardized['capacity_mw'].sum():.1f} MW")
    
    # === PRESERVE ORIGINAL DUKES COORDINATES FOR FALLBACK ===
    # Save the original DUKES X/Y coordinates before geocoding overwrites them
    # These are often inaccurate but better than nothing as final fallback
    standardized['dukes_x_original'] = standardized['x_coord'].copy()
    standardized['dukes_y_original'] = standardized['y_coord'].copy()
    dukes_coords_count = (standardized['dukes_x_original'].notna() & standardized['dukes_y_original'].notna()).sum()
    if dukes_coords_count > 0:
        logger.info(f"Preserved {dukes_coords_count} original DUKES X/Y coordinates for fallback")
    
    # === GEOCODING CASCADE (in order of reliability) ===
    # 1. Regional centroids (low precision, better than nothing)
    standardized = geocode_by_region(standardized)

    # 2. Postcode geocoding (high precision for thermal generators with postcodes)
    # This ONLY affects thermal generators - renewable coordinates from REPD are preserved
    standardized = geocode_thermal_generators_from_postcodes(standardized)

    # 3. Nominatim station name geocoding (good precision, free API)
    # Fallback for generators without postcodes
    standardized = geocode_from_nominatim(standardized)
    
    # 4. Nominatim location-based geocoding (partial matching, lower precision)
    # For generators where station name didn't work
    standardized = geocode_from_location_name(standardized)
    
    # 5. FINAL FALLBACK: Use DUKES X-Coordinate/Y-Coordinate if still missing
    # These are often inaccurate but better than no coordinates at all
    standardized = apply_dukes_coordinates_fallback(standardized)

    # Convert ALL BNG coordinates to WGS84 lat/lon
    standardized = convert_bng_to_wgs84(standardized)    # Log fuel type breakdown
    if 'fuel_type' in standardized.columns:
        fuel_breakdown = standardized.groupby('fuel_type')['capacity_mw'].agg(['count', 'sum'])
        logger.info(f"\nCapacity by fuel type:")
        for fuel, row in fuel_breakdown.iterrows():
            logger.info(f"  {fuel}: {row['count']} generators, {row['sum']:.1f} MW")
    
    return standardized


def validate_dukes_output(df: pd.DataFrame, year: int):
    """
    Validate DUKES output data for completeness and quality.
    
    Args:
        df: Standardized DUKES DataFrame
        year: Year being processed
        
    Raises:
        ValueError: If critical validation checks fail
    """
    logger.info(f"Validating DUKES {year} output")
    
    errors = []
    warnings = []
    
    # Check: Must have at least some generators
    if len(df) == 0:
        errors.append(f"No generators extracted for {year}")
    
    # Check: Total capacity should be reasonable (UK ~60-80 GW thermal)
    total_capacity_gw = df['capacity_mw'].sum() / 1000
    if total_capacity_gw < 20:
        warnings.append(f"Low total capacity: {total_capacity_gw:.1f} GW (expected ~50-70 GW)")
    elif total_capacity_gw > 100:
        warnings.append(f"High total capacity: {total_capacity_gw:.1f} GW (expected ~50-70 GW)")
    
    # Check: Should have multiple fuel types
    if 'fuel_type' in df.columns:
        n_fuel_types = df['fuel_type'].nunique()
        if n_fuel_types < 3:
            warnings.append(f"Only {n_fuel_types} fuel types found (expected at least 5)")
    
    # Check: Coordinates coverage
    coords_coverage = df[['x_coord', 'y_coord']].notna().all(axis=1).sum()
    coords_pct = 100 * coords_coverage / len(df) if len(df) > 0 else 0
    logger.info(f"  Coordinate coverage: {coords_coverage}/{len(df)} ({coords_pct:.1f}%)")
    if coords_pct < 50:
        warnings.append(f"Low coordinate coverage: {coords_pct:.1f}% (may need location matching)")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"  ⚠️  {warning}")
    
    # Raise errors if critical issues found
    if errors:
        for error in errors:
            logger.error(f"  ❌ {error}")
        raise ValueError(f"DUKES {year} validation failed: {'; '.join(errors)}")
    
    logger.info(f"✅ Validation passed: {len(df)} generators, {total_capacity_gw:.1f} GW")


def main():
    """Main DUKES data extraction workflow"""
    logger.info("="*80)
    logger.info("DUKES Historical Generator Data Extraction")
    logger.info("="*80)
    
    # Get Snakemake parameters
    dukes_file = Path(snakemake.input.dukes_file)
    output_file = Path(snakemake.output.dukes_generators)
    year = snakemake.params.dukes_year
    
    logger.info(f"Configuration:")
    logger.info(f"  DUKES file: {dukes_file}")
    logger.info(f"  Target year: {year}")
    logger.info(f"  Output: {output_file}")
    
    # Validate year range (DUKES 5.11 2025 edition contains 2010-2024)
    if not (2010 <= year <= 2024):
        raise ValueError(f"Year {year} outside DUKES 5.11 coverage (2010-2024)")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Read DUKES worksheet
        df_raw = read_dukes_worksheet(dukes_file, year)
        
        # Step 2: Standardize to FES-compatible format
        df_standardized = standardize_dukes_data(df_raw, year)
        
        # Step 3: Validate output
        validate_dukes_output(df_standardized, year)
        
        # Step 4: Save to CSV
        df_standardized.to_csv(output_file, index=False)
        logger.info(f"\n✅ DUKES {year} data extracted successfully")
        logger.info(f"   Output: {output_file}")
        logger.info(f"   Generators: {len(df_standardized)}")
        logger.info(f"   Total capacity: {df_standardized['capacity_mw'].sum():.1f} MW")
        
    except Exception as e:
        logger.error(f"\n❌ DUKES extraction failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

