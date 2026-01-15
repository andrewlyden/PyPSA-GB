import pandas as pd
import numpy as np
import pypsa
import logging
import time
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Suppress PyPSA warnings about unoptimized networks (expected during network building)
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Add logging import
from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info, log_execution_summary

# Constants
VOLTAGE_LEVELS = {
    '1': 132, '2': 275, '3': 33, '4': 400, 
    '5': 11, '6': 66, '7': 20.5
}

# Coordinate conversion constants for GB (approximate)
LAT_DEGREES_PER_KM = 1 / 111.0
LON_DEGREES_PER_KM_BASE = 1 / 111.0

# Default electrical parameters to avoid zero values
DEFAULT_R = 0.0001
DEFAULT_X = 0.0001  
DEFAULT_B = 0.0001

# Land boundary checking constants
GSP_REGIONS_FILE = "data/network/GSP/GSP_regions_4326_20250109.geojson"
LAND_BUFFER_KM = 0.5  # Buffer distance from coastline


def load_land_boundaries(logger: Optional[logging.Logger] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Load Great Britain land boundaries from GSP regions file.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        GeoDataFrame with GB land boundaries or None if loading fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        logger.info(f"Loading GB land boundaries from {GSP_REGIONS_FILE}")
        gdf = gpd.read_file(GSP_REGIONS_FILE)

        # Combine all GSP regions into a single land boundary using union_all()
        land_boundary = gdf.geometry.union_all()
        land_gdf = gpd.GeoDataFrame([1], geometry=[land_boundary], crs=gdf.crs)

        logger.info(f"Loaded land boundaries: {len(gdf)} GSP regions combined")
        logger.info(f"Land boundary CRS: {land_gdf.crs}")
        logger.info(f"Land boundary bounds: {land_gdf.total_bounds}")

        return land_gdf
        
    except Exception as e:
        logger.error(f"Failed to load land boundaries: {e}")
        logger.warning("Proceeding without land boundary checking")
        return None


def check_point_on_land(lat: float, lon: float, land_boundary: gpd.GeoDataFrame, 
                       logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if a coordinate point is on land (within GB boundaries).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees  
        land_boundary: GeoDataFrame with land boundaries
        logger: Optional logger instance
        
    Returns:
        True if point is on land, False if at sea
    """
    if land_boundary is None:
        return True  # Assume on land if no boundary data
        
    try:
        point = Point(lon, lat)
        return land_boundary.contains(point).any()
    except Exception as e:
        if logger:
            logger.debug(f"Error checking point ({lat}, {lon}) on land: {e}")
        return True  # Assume on land if check fails


def move_point_to_land(lat: float, lon: float, land_boundary: gpd.GeoDataFrame,
                      logger: Optional[logging.Logger] = None) -> Tuple[float, float]:
    """
    Move a point from sea to the nearest land location.
    
    Args:
        lat: Original latitude in degrees
        lon: Original longitude in degrees
        land_boundary: GeoDataFrame with land boundaries
        logger: Optional logger instance
        
    Returns:
        Tuple of (new_lat, new_lon) on land
    """
    # Handle NaN coordinates - return as-is (will be handled elsewhere)
    if pd.isna(lat) or pd.isna(lon):
        if logger:
            logger.warning(f"Cannot move point with NaN coordinates: ({lat}, {lon})")
        return lat, lon
        
    if land_boundary is None:
        return lat, lon  # Return original if no boundary data
        
    try:
        sea_point = Point(lon, lat)
        land_geom = land_boundary.geometry.iloc[0]
        
        # Find nearest point on land boundary
        nearest_geom, nearest_point = nearest_points(sea_point, land_geom)
        
        # Add small buffer to ensure point is clearly on land
        buffer_deg = LAND_BUFFER_KM / 111.0  # Rough conversion km to degrees
        land_centroid = land_geom.centroid
        
        # Move point slightly towards land centroid
        direction_x = land_centroid.x - nearest_point.x
        direction_y = land_centroid.y - nearest_point.y
        length = np.sqrt(direction_x**2 + direction_y**2)
        
        if length > 0:
            direction_x /= length
            direction_y /= length
            
            final_lon = nearest_point.x + direction_x * buffer_deg
            final_lat = nearest_point.y + direction_y * buffer_deg
        else:
            final_lon = nearest_point.x
            final_lat = nearest_point.y
            
        if logger:
            distance_km = sea_point.distance(nearest_point) * 111.0  # Rough conversion
            logger.debug(f"Moved point from sea ({lat:.4f}, {lon:.4f}) to land ({final_lat:.4f}, {final_lon:.4f}), distance: {distance_km:.2f} km")
            
        return final_lat, final_lon
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to move point ({lat}, {lon}) to land: {e}")
        return lat, lon  # Return original if correction fails


def ensure_buses_on_land(network: pypsa.Network, land_boundary: Optional[gpd.GeoDataFrame],
                        logger: Optional[logging.Logger] = None) -> int:
    """
    Ensure all buses in the network are located on land.
    
    Args:
        network: PyPSA network with bus coordinates
        land_boundary: GeoDataFrame with land boundaries
        logger: Optional logger instance
        
    Returns:
        Number of buses moved from sea to land
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if land_boundary is None:
        logger.warning("No land boundary data available - skipping land validation")
        return 0
        
    logger.info("Checking bus locations against land boundaries")
    buses_moved = 0
    
    for bus_id, bus_data in network.buses.iterrows():
        lat, lon = bus_data.lat, bus_data.lon
        
        if not check_point_on_land(lat, lon, land_boundary, logger):
            # Bus is at sea - move to land
            new_lat, new_lon = move_point_to_land(lat, lon, land_boundary, logger)
            network.buses.loc[bus_id, 'lat'] = new_lat
            network.buses.loc[bus_id, 'lon'] = new_lon
            buses_moved += 1
            
            logger.debug(f"Moved bus {bus_id} from sea ({lat:.4f}, {lon:.4f}) to land ({new_lat:.4f}, {new_lon:.4f})")
    
    if buses_moved > 0:
        logger.info(f"Moved {buses_moved} buses from sea to land")
    else:
        logger.info("All buses were already on land")
        
    return buses_moved

def sort_raw_ETYS_data(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Parse and process raw ETYS data from Excel sheets.
    
    Combines line, transformer, and interconnector data from multiple sheets,
    standardizes column names, and adds length information for coordinate estimation.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        DataFrame with processed network component data including length information
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Reading ETYS raw data from Excel sheets")
    # Read all sheets at once
    xls = pd.ExcelFile(snakemake.input[0])
    logger.info(f"Loaded Excel file with {len(xls.sheet_names)} sheets")
    
    sheets = {
        'B-2-1a': xls.parse('B-2-1a', skiprows=1),
        'B-2-1b': xls.parse('B-2-1b', skiprows=1),
        'B-2-1c': xls.parse('B-2-1c', skiprows=1),
        'B-2-1d': xls.parse('B-2-1d', skiprows=1),
        'B-3-1a': xls.parse('B-3-1a', skiprows=1),
        'B-3-1b': xls.parse('B-3-1b', skiprows=1),
        'B-3-1c': xls.parse('B-3-1c', skiprows=1),
        'B-3-1d': xls.parse('B-3-1d', skiprows=1),
        'B-5-1': xls.parse('B-5-1', skiprows=1)
    }

    logger.info("Processing line data sheets (B-2-1a to B-2-1d)")
    dfa, dfb, dfc, dfd = sheets['B-2-1a'], sheets['B-2-1b'], sheets['B-2-1c'], sheets['B-2-1d']
    for df in [dfa, dfb, dfc, dfd]:
        df.loc[:, 'component'] = 'line'
        df.loc[:, 'carrier'] = 'AC'
    dfd.rename(columns={'R (% on 100MVA)': 'R (% on 100 MVA)', 'X (% on 100MVA)': 'X (% on 100 MVA)', 'B (% on 100MVA)': 'B (% on 100 MVA)', 'Rating (MVA)': 'Winter Rating (MVA)'}, inplace=True)

    logger.info("Processing transformer data sheets (B-3-1a to B-3-1d)")
    dfe, dff, dfg, dfh = sheets['B-3-1a'], sheets['B-3-1b'], sheets['B-3-1c'], sheets['B-3-1d']
    dfe.rename(columns={'Rating (MVA)': 'Winter Rating (MVA)'}, inplace=True)
    for df in [dff, dfg, dfh]:
        df.rename(columns={'R (% on 100MVA)': 'R (% on 100 MVA)', 'X (% on 100MVA)': 'X (% on 100 MVA)', 'B (% on 100MVA)': 'B (% on 100 MVA)', 'Node1': 'Node 1', 'Node2': 'Node 2', 'Rating (MVA)': 'Winter Rating (MVA)'}, inplace=True)
    for df in [dfe, dff, dfg, dfh]:
        df.loc[:, 'component'] = 'transformer'
        df.loc[:, 'carrier'] = 'AC'

    logger.info("Processing interconnector data (B-5-1)")
    dfi = sheets['B-5-1']
    dfi = dfi.loc[dfi['Existing'] == 'Yes'].copy()
    logger.info(f"Found {len(dfi)} existing interconnectors")
    dfi.loc[:, 'component'] = 'link'
    dfi.loc[:, 'carrier'] = 'DC'

    logger.info("Loading additional wind farm and BMU edges from GB_network.xlsx")
    dfj = pd.read_excel(snakemake.input[1], sheet_name='Extra_WF_edges')
    dfj.loc[:, 'component'] = 'line'
    dfj.loc[:, 'carrier'] = 'AC'
    dfj.loc[:, 'Winter Rating (MVA)'] = 9999
    logger.info(f"Loaded {len(dfj)} extra wind farm edges")

    dfk = pd.read_excel(snakemake.input[1], sheet_name='Extra_BMUs_edges')
    dfk.loc[:, 'component'] = 'line'
    dfk.loc[:, 'carrier'] = 'AC'
    dfk.loc[:, 'Winter Rating (MVA)'] = 9999
    logger.info(f"Loaded {len(dfk)} extra BMU edges")

    logger.info("Concatenating all network components")
    df = pd.concat([dfa, dfb, dfc, dfd, dfe, dff, dfg, dfh, dfi, dfj, dfk], ignore_index=True)
    df.rename(columns={'Node 1': 'bus0', 'Node 2': 'bus1'}, inplace=True)
    df.index.name = 'name'
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={'R (% on 100 MVA)': 'r', 'X (% on 100 MVA)': 'x', 'B (% on 100 MVA)': 'b', 'Winter Rating (MVA)': 's_nom'}, inplace=True)
    
    # Calculate total length for distance-based coordinate estimation
    if 'OHL Length (km)' in df.columns and 'Cable Length (km)' in df.columns:
        df['length_km'] = df['OHL Length (km)'].fillna(0) + df['Cable Length (km)'].fillna(0)
    elif 'Length(km)' in df.columns:  # For interconnectors
        df['length_km'] = df['Length(km)'].fillna(0)
    else:
        df['length_km'] = 0
    
    df = df[['component', 'carrier', 'bus0', 'bus1', 'r', 'x', 'b', 's_nom', 'length_km']]

    logger.info("Processing electrical parameters")
    # Explicitly cast columns to float64
    df['r'] = df['r'].astype('float64')
    df['x'] = df['x'].astype('float64')
    df['b'] = df['b'].astype('float64')

    # CRITICAL: Convert from "% on 100 MVA" to per-unit (p.u.)
    # ETYS data is in percentage format (e.g., 8.0467% = 0.080467 p.u.)
    # PyPSA expects per-unit values, so divide by 100
    logger.info("Converting electrical parameters from % on 100 MVA base to per-unit")
    df['r'] = df['r'] / 100.0
    df['x'] = df['x'] / 100.0
    df['b'] = df['b'] / 100.0

    df['r'] = df['r'].replace(0, DEFAULT_R).fillna(DEFAULT_R)
    df['x'] = df['x'].replace(0, DEFAULT_X).fillna(DEFAULT_X)
    df['b'] = df['b'].replace(0, DEFAULT_B).fillna(DEFAULT_B)
    
    logger.info(f"Processed {len(df)} network components")
    log_dataframe_info(df, logger, "Network components summary")
    
    return df

def buses_from_line_data(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Extract unique buses from network line data and add voltage level and carrier information.
    
    Args:
        df: DataFrame containing network line data
        logger: Optional logger instance
        
    Returns:
        DataFrame with bus information including voltage levels and carriers
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Extracting buses from line data")
    # buses from line data
    df_buses = pd.concat([df['bus0'], df['bus1']]).unique()
    df_buses = pd.DataFrame(df_buses, columns=['name'])
    df_buses.index = df_buses['name']
    df_buses.index.name = 'name'
    logger.info(f"Found {len(df_buses)} unique buses")

    logger.info("Adding voltage level data to buses")
    # add voltage data using constants
    # extract first character of string which is a number from df_buses name column
    map = df_buses['name'].str.extract(r'(\d+)', expand=False)
    # remove all apart from the first character of the string
    map = map.str[0]
    # use this map to add voltage data to df_buses
    df_buses['v_nom'] = map.map(VOLTAGE_LEVELS)

    logger.info("Processing carrier information for buses")
    # create a new dataframe from df with the bus0 and bus1 columns and carrier column
    df_carrier = df[['bus0', 'bus1', 'carrier']]
    df_carrier2 = df_carrier.copy()
    # make bus0 index, while dropping column
    df_carrier = df_carrier.set_index('bus0', drop=True)
    # drop bus 1 column
    df_carrier = df_carrier.drop(columns=['bus1'])

    df_carrier2 = df_carrier2.set_index('bus1', drop=True)
    # drop bus 0 column
    df_carrier2 = df_carrier2.drop(columns=['bus0'])
    # set both index names to name
    df_carrier.index.name = 'name'
    df_carrier2.index.name = 'name'
    # concat dfs
    df_carrier = pd.concat([df_carrier, df_carrier2])
    # drop duplicates
    df_carrier = df_carrier[~df_carrier.index.duplicated(keep='first')]
    # add carrier column to df_buses
    df_buses['carrier'] = df_buses.index.map(df_carrier['carrier'])
    # set carrier of buses to AC
    df_buses['carrier'] = 'AC'

    logger.info(f"Completed bus processing with voltage levels and carriers")
    log_dataframe_info(df_buses, logger, "Buses summary")
    
    return df_buses

def GSP_locations_from_FES_data(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load Grid Supply Point (GSP) location data from Future Energy Scenarios (FES) data.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        DataFrame with GSP location data (coordinates)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Loading GSP location data from FES data")
    # FES data with GSP locations (input[2] is Regional breakdown of FES data)
    df2 = pd.read_excel(snakemake.input[2], sheet_name='GSP info', skiprows=4, index_col=1)
    # df2.index = df2.index.str[:4]
    df2 = df2[~df2.index.duplicated(keep='first')]
    logger.info(f"Loaded {len(df2)} GSP locations")
    df2.rename(columns={'Latitude':'lat', 'Longitude': 'lon'}, inplace=True)
    # df2.drop(['ROTI'], inplace=True)
    df2.drop(columns=['Name'], inplace=True)
    df2.index.name = 'name'
    df2['name'] = df2.index

    log_dataframe_info(df2, logger, "GSP locations summary")
    return df2

def add_GSP_location_data(df_buses: pd.DataFrame, df2: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Add GSP coordinate data to buses by matching the first 4 characters of bus names.
    
    Args:
        df_buses: DataFrame with bus data
        df2: DataFrame with GSP location data
        logger: Optional logger instance
        
    Returns:
        DataFrame with buses that have coordinate data where available
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Matching buses to GSP locations")
    # Step 1: Define a function that takes an index and returns the corresponding lon and lat values from `df2` where the first 4 characters of the index match.
    def get_coords(index):
        matching_df2 = df2[df2.index.str[:4] == index[:4]]
        if not matching_df2.empty:
            return matching_df2.iloc[0]['lon'], matching_df2.iloc[0]['lat']
        else:
            return None, None

    # Step 2: Apply this function to the index of `df_buses` to create the new columns.
    logger.info("Applying GSP coordinates to buses")
    df_buses['lon'], df_buses['lat'] = zip(*df_buses.index.map(get_coords))
    
    # Single, clean type conversion
    df_buses['lon'] = pd.to_numeric(df_buses['lon'], errors='coerce')
    df_buses['lat'] = pd.to_numeric(df_buses['lat'], errors='coerce')
    
    # remove the name column
    df_buses = df_buses.drop(columns=['name'])

    # Count matched vs unmatched buses
    matched_count = (~df_buses['lon'].isna()).sum()
    total_count = len(df_buses)
    logger.info(f"Successfully matched {matched_count}/{total_count} buses to GSP locations")
    
    return df_buses

def guess_GSP_location_of_remaining_buses(df: pd.DataFrame, df_buses: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Improved coordinate guessing using distance-weighted estimation and graph connectivity.
    Uses line lengths to better estimate bus positions and ensures all buses are on land.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Load land boundaries for validation
    land_boundary = load_land_boundaries(logger)

    logger.info("Creating temporary network for location guessing")
    network = pypsa.Network()
    network.set_snapshots(range(1))
    
    network.add("Bus", df_buses.index, **df_buses)
    
    # Add lines with length information (drop metadata columns first to avoid PyPSA 1.0.2 warnings)
    df_with_length = df.copy()
    df_with_length = df_with_length.drop(columns=[col for col in ['component', 'lon', 'lat', 'carrier'] if col in df_with_length.columns])
    network.add("Line", df_with_length.index, **df_with_length)
    network.add("Carrier", ["AC", "DC"])

    network.consistency_check()
    # if lon and lat are 0, set to NaN
    network.buses.loc[network.buses['lon'] == 0, 'lon'] = np.nan
    network.buses.loc[network.buses['lat'] == 0, 'lat'] = np.nan

    logger.info("Starting improved iterative coordinate guessing process")
    
    # Constants for coordinate estimation
    MAX_ITERATIONS = 50
    CONVERGENCE_THRESHOLD = 1e-6  # meters
    
    prev_count = len(network.buses) + 1
    iteration = 0

    while iteration < MAX_ITERATIONS:
        # Get buses with missing coordinates
        missing_coords = network.buses['lon'].isna() | network.buses['lat'].isna()
        curr_count = missing_coords.sum()

        # Check convergence
        if curr_count == 0 or prev_count == curr_count:
            break

        prev_count = curr_count
        iteration += 1
        
        logger.info(f"Iteration {iteration}: {curr_count} buses still need coordinates")
        
        # Strategy 1: Use distance-weighted positioning for buses with one known neighbor
        improved_count = 0
        
        for bus in network.buses[missing_coords].index:
            # Find all connected lines and their lengths
            connected_lines = network.lines[
                (network.lines['bus0'] == bus) | (network.lines['bus1'] == bus)
            ].copy()
            
            if connected_lines.empty:
                continue
                
            # Get connected buses with their distances
            connected_info = []
            for _, line in connected_lines.iterrows():
                other_bus = line['bus1'] if line['bus0'] == bus else line['bus0']
                other_coords = network.buses.loc[other_bus, ['lon', 'lat']]
                
                # Only use buses with known coordinates
                if not (pd.isna(other_coords['lon']) or pd.isna(other_coords['lat'])):
                    length_km = line.get('length_km', 0)
                    connected_info.append({
                        'bus': other_bus,
                        'lon': other_coords['lon'],
                        'lat': other_coords['lat'], 
                        'length_km': max(length_km, 0.1)  # Minimum 100m to avoid division by zero
                    })
            
            if not connected_info:
                continue
                
            # Strategy 1a: If only one connected bus with known coordinates
            if len(connected_info) == 1:
                # Use simple distance estimation with land validation
                ref = connected_info[0]
                
                # Try multiple directions to find one that lands on land
                best_coords = None
                best_distance_to_land = float('inf')
                
                for attempt in range(8):  # Try 8 different directions
                    angle = attempt * np.pi / 4  # 45-degree increments
                    
                    # Convert km to approximate lat/lon degrees (rough approximation for GB)
                    lat_deg_per_km = 1 / 111.0  # Approximately 111 km per degree latitude
                    lon_deg_per_km = 1 / (111.0 * np.cos(np.radians(ref['lat'])))  # Adjust for latitude
                    
                    dx = ref['length_km'] * np.cos(angle) * lon_deg_per_km
                    dy = ref['length_km'] * np.sin(angle) * lat_deg_per_km
                    
                    candidate_lon = ref['lon'] + dx
                    candidate_lat = ref['lat'] + dy
                    
                    # Check if this position is on land
                    if land_boundary is not None:
                        if check_point_on_land(candidate_lat, candidate_lon, land_boundary, logger):
                            # Found a position on land
                            best_coords = (candidate_lat, candidate_lon)
                            break
                        else:
                            # Calculate distance to land for this candidate
                            try:
                                point = Point(candidate_lon, candidate_lat)
                                land_geom = land_boundary.geometry.iloc[0]
                                distance_to_land = point.distance(land_geom)
                                if distance_to_land < best_distance_to_land:
                                    best_distance_to_land = distance_to_land
                                    best_coords = (candidate_lat, candidate_lon)
                            except:
                                continue
                
                # Use the best coordinates found (prefer on-land, otherwise closest to land)
                if best_coords:
                    candidate_lat, candidate_lon = best_coords
                    
                    # If still at sea, move to land
                    if land_boundary is not None and not check_point_on_land(candidate_lat, candidate_lon, land_boundary, logger):
                        candidate_lat, candidate_lon = move_point_to_land(candidate_lat, candidate_lon, land_boundary, logger)
                    
                    network.buses.loc[bus, 'lon'] = candidate_lon
                    network.buses.loc[bus, 'lat'] = candidate_lat
                    improved_count += 1
                
            # Strategy 1b: Multiple connected buses - use distance-weighted centroid
            elif len(connected_info) >= 2:
                # Calculate distance-weighted position
                weights = []
                lon_coords = []
                lat_coords = []
                
                for info in connected_info:
                    # Weight inversely proportional to distance (closer buses have more influence)
                    weight = 1.0 / info['length_km']
                    weights.append(weight)
                    lon_coords.append(info['lon'])
                    lat_coords.append(info['lat'])
                
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights
                
                estimated_lon = np.average(lon_coords, weights=weights)
                estimated_lat = np.average(lat_coords, weights=weights)
                
                # Validate position is on land
                if land_boundary is not None and not check_point_on_land(estimated_lat, estimated_lon, land_boundary, logger):
                    estimated_lat, estimated_lon = move_point_to_land(estimated_lat, estimated_lon, land_boundary, logger)
                
                network.buses.loc[bus, 'lon'] = estimated_lon
                network.buses.loc[bus, 'lat'] = estimated_lat
                improved_count += 1
        
        logger.debug(f"Improved coordinates for {improved_count} buses in iteration {iteration}")
        
        # Strategy 2: Simple averaging for remaining buses (fallback)
        remaining_missing = network.buses['lon'].isna() | network.buses['lat'].isna()
        for bus in network.buses[remaining_missing].index:
            connected_lines = network.lines[
                (network.lines['bus0'] == bus) | (network.lines['bus1'] == bus)
            ]
            
            if connected_lines.empty:
                continue
                
            # Get connected buses
            connected_buses_list = []
            for _, line in connected_lines.iterrows():
                other_bus = line['bus1'] if line['bus0'] == bus else line['bus0']
                connected_buses_list.append(other_bus)
            
            connected_buses = network.buses.loc[connected_buses_list]
            connected_buses = connected_buses[connected_buses.index != bus]
            
            # Only use buses with known coordinates
            known_connected = connected_buses.dropna(subset=['lon', 'lat'])
            if not known_connected.empty:
                # Simple average of connected bus coordinates
                avg_coords = known_connected[['lon', 'lat']].mean()
                estimated_lon, estimated_lat = avg_coords['lon'], avg_coords['lat']
                
                # Validate position is on land
                if land_boundary is not None and not check_point_on_land(estimated_lat, estimated_lon, land_boundary, logger):
                    estimated_lat, estimated_lon = move_point_to_land(estimated_lat, estimated_lon, land_boundary, logger)
                
                network.buses.loc[bus, ['lon', 'lat']] = [estimated_lon, estimated_lat]

    # Final land boundary check for all buses
    if land_boundary is not None:
        logger.info("Performing final land boundary validation for all buses")
        buses_moved = ensure_buses_on_land(network, land_boundary, logger)
        if buses_moved > 0:
            logger.info(f"Final land validation moved {buses_moved} additional buses to land")

    # Log final results
    final_missing = (network.buses['lon'].isna() | network.buses['lat'].isna()).sum()
    logger.info(f"Coordinate guessing complete after {iteration} iterations. {final_missing} buses still without coordinates")
    
    if final_missing > 0:
        missing_buses = network.buses[network.buses['lon'].isna() | network.buses['lat'].isna()]
        logger.warning(f"Buses without coordinates: {missing_buses.index.tolist()}")
    
    # Remove buses with NaN coordinates
    df_buses_with_locs = network.buses.dropna(subset=['lon', 'lat'])
    logger.info(f"Final bus count after removing NaN coordinates: {len(df_buses_with_locs)}")
    
    return df_buses_with_locs

def create_network(df: pd.DataFrame, df_buses_with_locs: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Create the final PyPSA network with all components and export to NetCDF.
    
    Args:
        df: DataFrame with all network components (lines, transformers, links)
        df_buses_with_locs: DataFrame with buses that have coordinates
        logger: Optional logger instance
        
    Returns:
        Complete PyPSA Network object
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Creating final PyPSA network")
    # new network with updated locations of buses
    network2 = pypsa.Network()
    network2.set_snapshots(range(1))

    logger.info(f"Adding {len(df_buses_with_locs)} buses to network")
    network2.add("Bus", df_buses_with_locs.index, **df_buses_with_locs)
    
    # CRITICAL: Convert WGS84 (lon/lat) to OSGB36 (x/y) for PyPSA plotting
    # PyPSA expects x/y in projected coordinates (meters), not lat/lon (degrees)
    # See: https://docs.pypsa.org/latest/user-guide/plotting/static-map/
    if 'lon' in network2.buses.columns and 'lat' in network2.buses.columns:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)  # WGS84 -> OSGB36
        network2.buses['x'], network2.buses['y'] = transformer.transform(
            network2.buses['lon'].values,
            network2.buses['lat'].values
        )
        logger.info(f"Converted coordinates: lon/lat (WGS84) -> x/y (OSGB36)")
        logger.info(f"  x range: [{network2.buses['x'].min():.0f}, {network2.buses['x'].max():.0f}] meters")
        logger.info(f"  y range: [{network2.buses['y'].min():.0f}, {network2.buses['y'].max():.0f}] meters")

    # filter df with component of line
    df_line = df[df['component'] == 'line']
    # remove lines if bus0 or bus1 is not in buses
    df_line = df_line[df_line['bus0'].isin(df_buses_with_locs.index) & df_line['bus1'].isin(df_buses_with_locs.index)]
    # Drop metadata columns (geolocation only) - keep electrical attributes 'r', 'x', 'b'
    df_line = df_line.drop(columns=[col for col in ['component', 'lon', 'lat', 'carrier'] if col in df_line.columns])
    logger.info(f"Adding {len(df_line)} lines to network")
    network2.add("Line", df_line.index, **df_line)
    # set DC to AC
    network2.lines['carrier'] = 'AC'

    # filter df with component of transformer
    df_transformer = df[df['component'] == 'transformer']
    # remove transformers if bus0 or bus1 is not in buses
    df_transformer = df_transformer[df_transformer['bus0'].isin(df_buses_with_locs.index) & df_transformer['bus1'].isin(df_buses_with_locs.index)]
    # Drop metadata columns (geolocation only) - keep electrical attributes 'r', 'x', 'b'
    df_transformer = df_transformer.drop(columns=[col for col in ['component', 'lon', 'lat', 'carrier'] if col in df_transformer.columns])
    logger.info(f"Adding {len(df_transformer)} transformers to network")
    network2.add("Transformer", df_transformer.index, **df_transformer)

    # filter df with component of link using loc
    df_link = df.loc[df['component'] == 'link']
    # rename s_nom to p_nom
    df_link = df_link.rename(columns={'s_nom': 'p_nom'})
    # Drop metadata columns that aren't PyPSA Link attributes (avoid PyPSA 1.0.2 warnings)
    df_link = df_link.drop(columns=[col for col in ['component', 'x', 'y', 'b', 'r'] if col in df_link.columns])
    
    # CRITICAL: Make internal HVDC links bidirectional (p_min_pu=-1)
    # Without this, links can only transfer power FROM bus0 TO bus1, which causes
    # infeasibility when power needs to flow the other direction
    df_link['p_min_pu'] = -1.0  # Bidirectional
    df_link['p_max_pu'] = 1.0
    
    logger.info(f"Adding {len(df_link)} links to network (bidirectional HVDC)")
    network2.add("Link", df_link.index, **df_link)

    # set links to AC using loc
    network2.links.loc[:, 'carrier'] = 'AC'

    # network2.import_components_from_dataframe(df_load, "Load")

    logger.info("Setting network metadata and performing final setup")
    network2.buses['country'] = 'GB'
    #manually modify lon and lat of interconnector bus location with index value HUCS4-
    network2.buses.loc['HUCS4-', 'lon'] = -4.897914663907308
    network2.buses.loc['HUCS4-', 'lat'] = 55.7173022715747
    network2.add("Carrier", ["AC", "DC"])
    # name network ETYS base
    network2.name = 'ETYS base'    
    network2.consistency_check()
    
    # ──────────────────────────────────────────────────────────────────────────
    # APPLY ETYS NETWORK UPGRADES (if enabled)
    # ──────────────────────────────────────────────────────────────────────────
    # Check if upgrades are enabled via snakemake params
    etys_upgrades_enabled = getattr(snakemake.params, 'etys_upgrades_enabled', False)
    
    if etys_upgrades_enabled:
        from ETYS_upgrades import apply_etys_network_upgrades
        
        # Get upgrade year (use modelled_year if upgrade_year not specified)
        modelled_year = getattr(snakemake.params, 'modelled_year', 2020)
        etys_upgrade_year = getattr(snakemake.params, 'etys_upgrade_year', None)
        upgrade_year = etys_upgrade_year if etys_upgrade_year else modelled_year
        
        # Path to ETYS upgrade data (same file as base network - input[0])
        etys_upgrade_file = str(snakemake.input[0])  # ETYS Appendix B 2023.xlsx
        
        logger.info("="*70)
        logger.info(f"APPLYING ETYS NETWORK UPGRADES through year {upgrade_year}")
        logger.info("="*70)
        
        network2 = apply_etys_network_upgrades(
            network2,
            modelled_year=upgrade_year,
            etys_file=etys_upgrade_file,
            logger=logger
        )
        
        # Ensure all buses (including newly added ones) have country='GB'
        network2.buses['country'] = network2.buses['country'].fillna('GB')
        network2.buses['country'] = network2.buses['country'].replace('', 'GB')
        
        # Re-run consistency check after upgrades
        network2.consistency_check()
        network2.name = f'ETYS base + upgrades ({upgrade_year})'
        
        logger.info(f"Network upgrades applied successfully (through {upgrade_year})")
    else:
        logger.info("ETYS network upgrades: DISABLED (set etys_upgrades.enabled: true to enable)")
    
    logger.info(f"Exporting network to {snakemake.output[0]}")
    
    # Suppress PyPSA warnings about unoptimized network during export
    pypsa_logger = logging.getLogger('pypsa.networks')
    original_level = pypsa_logger.level
    pypsa_logger.setLevel(logging.ERROR)
    try:
        network2.export_to_netcdf(snakemake.output[0])
    finally:
        pypsa_logger.setLevel(original_level)
    
    # Set version metadata before export to prevent compatibility warnings
    network2.meta = {"pypsa_version": pypsa.__version__}
    
    log_network_info(network2, logger)
    logger.info("Network creation completed successfully")
    
    return network2


if __name__ == "__main__":
    # Initialize timing
    start_time = time.time()
    
    # Set up logging using centralized system, writing to Snakemake log if available
    log_path = None
    if 'snakemake' in globals() and hasattr(snakemake, 'log') and snakemake.log:
        log_path = snakemake.log[0]
    logger = setup_logging(log_path or "ETYS_network")
    
    logger.info("="*50)
    logger.info("STARTING ETYS NETWORK CREATION")
    
    try:
        logger.info("Step 1: Processing raw ETYS data")
        df = sort_raw_ETYS_data(logger)
        
        logger.info("Step 2: Extracting buses from line data")
        df_buses = buses_from_line_data(df, logger)
        
        logger.info("Step 3: Loading GSP location data")
        df2 = GSP_locations_from_FES_data(logger)
        
        logger.info("Step 4: Adding GSP location data to buses")
        df_buses = add_GSP_location_data(df_buses, df2, logger)
        
        logger.info("Step 5: Guessing remaining bus locations")
        df_buses = guess_GSP_location_of_remaining_buses(df, df_buses, logger)
        
        logger.info("Step 6: Creating final network")
        network = create_network(df, df_buses, logger)
        
        logger.info("ETYS NETWORK CREATION COMPLETED SUCCESSFULLY")
        
        # Log execution summary
        inputs = [snakemake.input[i] for i in range(len(snakemake.input))] if 'snakemake' in globals() else []
        outputs = [snakemake.output[0]] if 'snakemake' in globals() else []
        log_execution_summary(logger, "ETYS Network Creation", start_time, inputs=inputs, outputs=outputs)
        
    except Exception as e:
        logger.exception(f"FATAL ERROR in ETYS network creation: {e}")
        raise
    
