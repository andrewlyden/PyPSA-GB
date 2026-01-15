import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import re
from pathlib import Path
from typing import Dict, Callable, Optional
from collections.abc import Sequence
import json
import geopandas as gpd
from shapely.geometry import Point

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Import shared spatial utilities for consistent bus mapping
from scripts.utilities.spatial_utils import map_sites_to_buses

# Add logging import
from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info, log_execution_summary
import time

# Configuration constants
BUILDING_BLOCK_ID = "Dem_BB003"
# Defaults (overridden in main based on scenario timestep_minutes)
FREQ_HALF_HOUR = '0.5h'
FREQ_HOUR = 'h'
TIMESTEP_HOURS = 0.5  # Duration of each timestep in hours
TIMESTEP_MINUTES = 30  # Used for weighting/logging; updated in main


def set_time_resolution(timestep_minutes: int, logger: Optional[logging.Logger] = None):
    """
    Update global timestep settings (frequency string and hours per step)
    so downstream profile generation and snapshot weighting stay in sync.
    """
    global FREQ_HALF_HOUR, FREQ_HOUR, TIMESTEP_HOURS, TIMESTEP_MINUTES
    TIMESTEP_MINUTES = timestep_minutes
    TIMESTEP_HOURS = timestep_minutes / 60.0
    # Use a single freq string for all date_range calls (even if name still says half-hour)
    freq_str = f"{timestep_minutes}min" if timestep_minutes % 60 != 0 else f"{int(timestep_minutes/60)}h"
    FREQ_HALF_HOUR = freq_str
    FREQ_HOUR = freq_str
    if logger:
        logger.info(f"Time resolution set to {timestep_minutes} minutes (freq='{freq_str}', weighting={TIMESTEP_HOURS} h)")

SCENARIO_COLUMNS = [
    "FES Pathway",
    "FES Scenario",
    "ï»¿FES Scenario",
    "\ufeffFES Scenario",
]


def get_param_value(param):
    """Return the first element of a Snakemake parameter if it is a sequence."""

    if isinstance(param, Sequence) and not isinstance(param, (str, bytes)):
        return param[0] if param else None
    return param


def get_single_input_path(file_arg):
    """Extract a single filesystem path from a Snakemake input argument."""

    if isinstance(file_arg, (str, Path)):
        return str(file_arg)
    if isinstance(file_arg, Sequence) and not isinstance(file_arg, (str, bytes)):
        if not file_arg:
            raise ValueError("Expected at least one input file, received an empty sequence")
        return str(file_arg[0])
    return str(file_arg)


def normalize_coordinates(df: pd.DataFrame, lon_col: str, lat_col: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Normalize coordinates to lat/lon (WGS84) format.
    Handles both projected coordinates (e.g., British National Grid EPSG:27700) and lat/lon.
    
    Args:
        df: DataFrame with coordinate columns
        lon_col: Name of the longitude/x column
        lat_col: Name of the latitude/y column
        logger: Optional logger for diagnostics
        
    Returns:
        DataFrame with normalized lat/lon coordinates
    """
    df = df.copy()
    
    try:
        # Try to import geopandas for coordinate transformation
        import geopandas as gpd
        from shapely.geometry import Point
        
        lon_vals = df[lon_col].values
        lat_vals = df[lat_col].values
        
        # Detect if coordinates are projected (BNG typical range: 0-700000, 0-1200000)
        if lon_vals.max() > 100 or lat_vals.max() > 100:
            if logger:
                logger.debug(f"Detected projected coordinates (BNG/EPSG:27700). Converting to WGS84...")
            
            # Create GeoDataFrame in projected CRS
            geometry = [Point(xy) for xy in zip(lon_vals, lat_vals)]
            gdf = gpd.GeoDataFrame(df[[col for col in df.columns if col not in [lon_col, lat_col]]], 
                                  geometry=geometry, crs="EPSG:27700")
            
            # Transform to WGS84 (lat/lon)
            gdf = gdf.to_crs("EPSG:4326")
            
            # Extract normalized coordinates
            df[lon_col] = gdf.geometry.x.values
            df[lat_col] = gdf.geometry.y.values
            
            if logger:
                logger.debug(f"Conversion complete: X range {df[lon_col].min():.2f} to {df[lon_col].max():.2f}, "
                           f"Y range {df[lat_col].min():.2f} to {df[lat_col].max():.2f}")
        else:
            if logger:
                logger.debug(f"Coordinates already in lat/lon format (WGS84)")
                
    except ImportError:
        if logger:
            logger.warning("geopandas not available; assuming coordinates are already in lat/lon format")
    except Exception as e:
        if logger:
            logger.warning(f"Error during coordinate normalization: {e}; assuming lat/lon format")
    
    return df


def resolve_gsp_metadata_path(fes_year: int, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """Return the path to the FES GSP metadata file, falling back where necessary."""

    candidates = [Path(f"data/network/ETYS/fes{fes_year}_regional_breakdown_gsp_info.csv")]
    if fes_year != 2024:
        candidates.append(Path("data/network/ETYS/fes2024_regional_breakdown_gsp_info.csv"))

    for idx, candidate in enumerate(candidates):
        if candidate.exists():
            if logger:
                if idx == 0:
                    logger.debug("Using GSP metadata file %s for FES year %s", candidate, fes_year)
                else:
                    logger.warning(
                        "FES year %s metadata missing; falling back to %s",
                        fes_year,
                        candidate,
                    )
            return candidate

    if logger:
        logger.error(
            "No GSP metadata file found for FES year %s (looked for %s)",
            fes_year,
            ", ".join(str(c) for c in candidates),
        )
    return None


def detect_scenario_column(df: pd.DataFrame) -> Optional[str]:
    """Identify the scenario column in an FES dataframe."""

    for col in SCENARIO_COLUMNS:
        if col in df.columns:
            return col

    # Fallback: look for columns containing "scenario" irrespective of spacing/encoding
    for col in df.columns:
        normalised = col.lower().strip().replace("_", " ")
        if "scenario" in normalised or "pathway" in normalised:
            return col
    return None


def extract_year_from_filename(path: Path) -> Optional[int]:
    """Extract a four-digit year from a standard FES filename."""

    match = re.search(r"FES_(\d{4})_data", path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def load_zone_demand_weights(network_model: str, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    Load demand weights for Zonal or Reduced networks from their definition files.
    
    For Zonal networks: zone_definitions.csv contains GSP-to-zone mappings with population weights.
    For Reduced networks: reduced_bus_definitions.csv contains direct demand weights per bus.
    
    Args:
        network_model: Network model type ("Zonal", "Reduced", etc.)
        logger: Optional logger for diagnostics
        
    Returns:
        Dictionary mapping bus IDs to demand weights (normalized to sum to 1.0)
        Returns empty dict if file not found or not applicable.
    """
    # Select the appropriate definitions file based on network model
    if network_model == "Zonal":
        zone_def_file = Path("data/network/zonal/zone_definitions.csv")
    elif network_model == "Reduced":
        zone_def_file = Path("data/network/reduced_network/reduced_bus_definitions.csv")
    else:
        if logger:
            logger.debug(f"Demand weights only applicable for Zonal/Reduced networks, not {network_model}")
        return {}
    
    if not zone_def_file.exists():
        if logger:
            logger.warning(f"Definitions file not found: {zone_def_file}")
        return {}
    
    try:
        zone_df = pd.read_csv(zone_def_file)
        if logger:
            logger.info(f"Loaded definitions: {len(zone_df)} entries from {zone_def_file.name}")
        
        # Handle different file formats
        if network_model == "Reduced":
            # Reduced network: direct bus_name -> demand_weight mapping
            required_cols = ['bus_name', 'demand_weight']
            for col in required_cols:
                if col not in zone_df.columns:
                    if logger:
                        logger.warning(f"Reduced definitions missing column: {col}")
                    return {}
            
            # Create direct mapping from bus_name to demand_weight
            zone_weights = zone_df.set_index('bus_name')['demand_weight']
        else:
            # Zonal network: aggregate by target_bus using population_weight
            required_cols = ['target_bus', 'population_weight']
            for col in required_cols:
                if col not in zone_df.columns:
                    if logger:
                        logger.warning(f"Zone definitions missing column: {col}")
                    return {}
            
            # Aggregate weights by zone (target_bus)
            zone_weights = zone_df.groupby('target_bus')['population_weight'].sum()
        
        # Normalize to sum to 1.0
        total_weight = zone_weights.sum()
        if total_weight <= 0:
            if logger:
                logger.warning("Zone weights sum to zero or negative")
            return {}
        
        normalized_weights = (zone_weights / total_weight).to_dict()
        
        if logger:
            logger.info(f"Zone demand weights (normalized):")
            for zone, weight in sorted(normalized_weights.items()):
                logger.info(f"  {zone}: {weight:.4f} ({weight*100:.2f}%)")
        
        return normalized_weights
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading zone demand weights: {e}")
        return {}

def create_reduced_network_mapping(gsp_names, fes_year: int, logger):
    """Create spatial mapping for the Reduced network using GSP coordinates."""

    try:
        gsp_file = resolve_gsp_metadata_path(fes_year, logger)
        if gsp_file is None:
            return None

        gsp_df = pd.read_csv(gsp_file)
        gsp_df = gsp_df.drop_duplicates(subset="Name", keep="first")
        gsp_df = gsp_df[gsp_df["Name"].notna()]
        if gsp_names:
            gsp_df = gsp_df[gsp_df["Name"].isin(gsp_names)]

        # Try multiple possible paths for reduced network buses
        possible_paths = [
            Path("data/network/reduced_network/buses.csv"),
            Path("data/network/BusesBasedGBsystem/buses.csv"),
        ]
        
        buses_file = None
        for path in possible_paths:
            if path.exists():
                buses_file = path
                logger.info(f"Found reduced network buses at: {path}")
                break
        
        if buses_file is None:
            logger.error(f"Reduced network buses file not found. Tried: {possible_paths}")
            return None
            
        buses_df = pd.read_csv(buses_file)
        
        # Normalize coordinates in both datasets to lat/lon (WGS84)
        gsp_df = normalize_coordinates(gsp_df, 'Longitude', 'Latitude', logger)
        buses_df = normalize_coordinates(buses_df, 'x', 'y', logger)

        # Create a temporary PyPSA network with buses for using shared mapping function
        temp_network = pypsa.Network()
        temp_network.import_components_from_dataframe(buses_df.set_index('name'), 'Bus')
        
        # Prepare GSP data for mapping
        gsp_for_mapping = gsp_df[['Name', 'Longitude', 'Latitude']].copy()
        gsp_for_mapping.columns = ['site_name', 'lon', 'lat']
        gsp_for_mapping = gsp_for_mapping.dropna(subset=['lat', 'lon'])
        
        # Use shared mapping function for consistent coordinate handling
        logger.info(f"Mapping {len(gsp_for_mapping)} GSPs to reduced network buses using shared spatial utilities")
        mapped_gsps_df = map_sites_to_buses(
            network=temp_network,
            sites_df=gsp_for_mapping,
            method='nearest',
            lat_col='lat',
            lon_col='lon',
            max_distance_km=500.0  # Higher limit for reduced networks (fewer buses)
        )
        
        # Convert to mapping list format
        mapping_list = []
        all_demand_gsps = set(gsp_names)
        mapped_gsps = set()
        
        for _, row in mapped_gsps_df.iterrows():
            if pd.notna(row.get('bus')):
                gsp_name = row['site_name']
                target_bus = row['bus']
                mapping_list.append(
                    {
                        "gsp_id": gsp_name,
                        "target_bus": target_bus,
                        "region": f"Reduced_{target_bus}",
                        "population_weight": 1.0,
                    }
                )
                mapped_gsps.add(gsp_name)
                logger.debug(
                    "Mapped GSP %s to reduced bus %s (distance %.3f km)",
                    gsp_name,
                    target_bus,
                    row.get('distance_km', 0.0),
                )

        missing_gsps = all_demand_gsps - mapped_gsps
        if missing_gsps:
            logger.warning(
                "%d GSPs missing coordinates for reduced mapping; applying fallbacks", len(missing_gsps)
            )
            fallback_assignments = {
                "Clydes Mill": "Neilston",
                "Connahs Quay": "Deeside",
                "Direct(SHETL)": "Beauly",
                "Direct(SPTL)": "Neilston",
                "Lodge Road (St Johns Wood)": "London",
                "St Johns Wood": "London",
            }

            for gsp_name in missing_gsps:
                fallback_bus = fallback_assignments.get(gsp_name, "London")
                mapping_list.append(
                    {
                        "gsp_id": gsp_name,
                        "target_bus": fallback_bus,
                        "region": f"Reduced_{fallback_bus}",
                        "population_weight": 1.0,
                    }
                )
                logger.debug("Fallback mapping for %s -> %s", gsp_name, fallback_bus)

        if not mapping_list:
            logger.error("Failed to build reduced network mapping; no assignments created")
            return None

        mapping_df = pd.DataFrame(mapping_list).set_index("gsp_id")
        logger.info("Created Reduced network spatial mapping with %d GSP assignments", len(mapping_df))
        return mapping_df

    except Exception as exc:
        logger.error("Error creating Reduced network mapping: %s", exc)
        return None

def create_spatial_zone_mapping(logger: Optional[logging.Logger] = None):
    """
    Create zone mapping for Zonal networks.
    
    This function supports both historical and future scenarios by:
    1. First trying to load pre-defined mappings from zone_definitions.csv (fast, reliable)
    2. Falling back to spatial assignment from GeoJSON if needed
    
    For historical scenarios, zone_definitions.csv is the primary source since
    FES data files may not be available.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # PRIORITY 1: Try zone_definitions.csv first (works for all scenarios including historical)
        zone_def_file = Path("data/network/zonal/zone_definitions.csv")
        if zone_def_file.exists():
            logger.info(f"Loading zone mapping from {zone_def_file}")
            zone_df = pd.read_csv(zone_def_file)
            
            # Check required columns
            required_cols = ['gsp_id', 'target_bus', 'region', 'population_weight']
            missing_cols = [c for c in required_cols if c not in zone_df.columns]
            
            if not missing_cols:
                # Rename gsp_id column if needed for consistent indexing
                mapping_df = zone_df.set_index('gsp_id')
                logger.info(f"Loaded {len(mapping_df)} zone mappings from zone_definitions.csv")
                return mapping_df
            else:
                logger.warning(f"zone_definitions.csv missing columns: {missing_cols}, falling back to spatial mapping")
        
        # PRIORITY 2: Fall back to spatial assignment from GeoJSON
        logger.info("Falling back to spatial zone mapping from GeoJSON")
        
        # Load zone boundaries from GeoJSON (correct path)
        zones_file = Path("data/network/zonal/zones.geojson")
        if not zones_file.exists():
            raise FileNotFoundError(f"Zones GeoJSON file not found: {zones_file}")
        
        zones_gdf = gpd.read_file(zones_file)
        
        # Load GSP coordinates
        gsp_file = Path("data/network/ETYS/fes2024_regional_breakdown_gsp_info.csv")
        if not gsp_file.exists():
            raise FileNotFoundError(f"GSP info file not found: {gsp_file}")
            
        gsp_df = pd.read_csv(gsp_file)
        
        # Remove duplicate GSP entries to prevent double-counting  
        gsp_df = gsp_df.drop_duplicates(subset='Name', keep='first')
        
        # Create spatial mapping using nearest neighbor approach for robustness
        zone_mapping = []
        
        # Try to load FES data for validation (optional - only used for future scenarios)
        all_demand_gsps = set()
        fes_data_path = Path("resources/FES/FES_2024_data.csv")
        if fes_data_path.exists():
            try:
                fes_data = pd.read_csv(fes_data_path)
                demand_blocks = fes_data[fes_data['Building Block ID Number'] == 'Dem_BB003']
                scenario_data = demand_blocks[demand_blocks['FES Pathway'] == 'Holistic Transition']
                all_demand_gsps = set(scenario_data['GSP'].unique())
                logger.info(f"Loaded {len(all_demand_gsps)} GSPs from FES data for validation")
            except Exception as e:
                logger.debug(f"Could not load FES data for validation: {e}")
        else:
            logger.debug("FES data not available (historical scenario?), skipping GSP validation")
        
        # Track which GSPs we've successfully mapped
        mapped_gsps = set()
        
        # Get zone centroids for distance calculations
        # Convert to British National Grid (EPSG:27700) for accurate centroid calculation
        zones_projected = zones_gdf.to_crs('EPSG:27700')
        zones_centroids = zones_projected.geometry.centroid
        # Convert centroids back to WGS84 for distance calculations with GSP coordinates
        zones_gdf['centroid'] = zones_centroids.to_crs('EPSG:4326')
        
        for idx, gsp in gsp_df.iterrows():
            if pd.notna(gsp['Latitude']) and pd.notna(gsp['Longitude']):
                point = Point(gsp['Longitude'], gsp['Latitude'])
                
                # First try exact containment
                assigned = False
                for zone_idx, zone in zones_gdf.iterrows():
                    if zone.geometry.contains(point):
                        zone_name = zone['Name_1']
                        
                        # Override for known problematic assignments due to overlapping boundaries
                        if gsp['Name'] in ['Ardmore', 'Dunvegan'] and zone_name == 'Z1_4':
                            # These are Outer Hebrides GSPs that should be in Z1_2, not Z1_4
                            zone_name = 'Z1_2'
                            logger.info(f"Corrected {gsp['Name']} from Z1_4 to Z1_2 (Outer Hebrides override)")
                        
                        zone_mapping.append({
                            'gsp_id': gsp['Name'],
                            'target_bus': zone_name,
                            'region': f'Zone_{zone_name}',
                            'population_weight': 1.0
                        })
                        assigned = True
                        mapped_gsps.add(gsp['Name'])
                        break
                
                # If not contained in any zone, assign to nearest zone
                if not assigned:
                    min_distance = float('inf')
                    nearest_zone = None
                    
                    for zone_idx, zone in zones_gdf.iterrows():
                        # Calculate distance to zone centroid
                        distance = point.distance(zone.centroid)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_zone = zone['Name_1']
                    
                    if nearest_zone:
                        zone_mapping.append({
                            'gsp_id': gsp['Name'],
                            'target_bus': nearest_zone,
                            'region': f'Zone_{nearest_zone}',
                            'population_weight': 1.0
                        })
                        mapped_gsps.add(gsp['Name'])
                        logger.info(f"Assigned {gsp['Name']} to nearest zone {nearest_zone} (distance: {min_distance:.3f})")
        
        # Handle GSPs that don't have coordinate data by assigning them geographically
        # Only do this if we have FES GSP list to compare against
        if all_demand_gsps:
            missing_gsps = all_demand_gsps - mapped_gsps
            if missing_gsps:
                logger.warning(f"{len(missing_gsps)} GSPs missing coordinates, assigning based on geography")
                
                # Geographic-based fallback assignments to appropriate zones
                fallback_assignments = {
                    'Clydes Mill': 'Z2',  # Central Scotland
                    'Connahs Quay': 'Z12',  # North Wales
                    'Direct(SHETL)': 'Z1_1',  # Northern Scotland transmission
                    'Direct(SPTL)': 'Z2',  # Central Scotland transmission
                    'Lodge Road (St Johns Wood)': 'Z15',  # Greater London area
                    'St Johns Wood': 'Z15'  # Greater London area
                }
                
                for gsp_name in missing_gsps:
                    fallback_zone = fallback_assignments.get(gsp_name, 'Z13')  # Default to Z13 (England) if unknown
                    zone_mapping.append({
                        'gsp_id': gsp_name,
                        'target_bus': fallback_zone,
                        'region': f'Zone_{fallback_zone}',
                        'population_weight': 1.0
                    })
                    logger.info(f"  - {gsp_name} -> {fallback_zone} (geographic fallback)")
        
        # Convert to DataFrame and return
        if zone_mapping:
            mapping_df = pd.DataFrame(zone_mapping)
            return mapping_df.set_index('gsp_id')
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error creating spatial zone mapping: {e}")
        return None

def get_network_model():
    """Get the network model from snakemake params."""
    if hasattr(snakemake.params, 'network_model'):
        network_model = snakemake.params.network_model
        return network_model[0] if isinstance(network_model, list) else network_model
    else:
        # Default fallback
        return "ETYS"

def load_spatial_mapping_data(network_model: str, gsp_names, fes_year: int, logger):
    """Load spatial mapping data appropriate for the network model."""
    
    if network_model == "ETYS":
        # Use existing ETYS-specific mapping
        logger.info("Loading ETYS spatial mapping")
        return None  # Current logic works as-is
    
    elif network_model == "Reduced":
        # Load reduced network bus-to-region mapping
        logger.info("Loading Reduced network spatial mapping")
        mapping = create_reduced_network_mapping(gsp_names, fes_year, logger)
        if mapping is not None:
            logger.info(f"Loaded {len(mapping)} bus-to-region mappings")
            return mapping
        else:
            logger.warning("Failed to load Reduced network mapping")
            return None
    
    elif network_model == "Zonal":
        # Load zonal network zone definitions using spatial assignment
        logger.info("Loading Zonal network spatial mapping")
        mapping = create_spatial_zone_mapping(logger)
        if mapping is not None:
            logger.info(f"Loaded spatial zone mapping with {len(mapping)} assignments")
            return mapping
        else:
            logger.warning("Failed to create spatial zone mapping")
            return None
    
    else:
        logger.warning(f"Unknown network model: {network_model}")
        return None


def get_transformer_capacity_into_buses(network: pypsa.Network, logger) -> dict:
    """
    Calculate total transformer capacity feeding into each bus from higher voltage levels.
    
    Args:
        network: PyPSA network object
        logger: Logger instance
        
    Returns:
        Dictionary mapping bus_id -> total transformer capacity (MVA)
    """
    tf_cap_into_bus = {}
    
    for idx, t in network.transformers.iterrows():
        bus0, bus1 = t.bus0, t.bus1
        v0 = network.buses.loc[bus0, 'v_nom'] if bus0 in network.buses.index else 0
        v1 = network.buses.loc[bus1, 'v_nom'] if bus1 in network.buses.index else 0
        
        # Step-down transformer: power flows from higher to lower voltage
        if v0 > v1:
            tf_cap_into_bus[bus1] = tf_cap_into_bus.get(bus1, 0) + t.s_nom
        elif v1 > v0:
            tf_cap_into_bus[bus0] = tf_cap_into_bus.get(bus0, 0) + t.s_nom
    
    logger.info(f"Calculated transformer capacity for {len(tf_cap_into_bus)} buses")
    return tf_cap_into_bus


def get_line_capacity_into_buses(network: pypsa.Network, logger) -> dict:
    """
    Calculate total line capacity connecting to each bus at the same voltage level.
    
    This is important for 132kV buses where line capacity may be more constraining
    than transformer capacity (e.g., USKM11 has 575 MVA transformer capacity but
    only 90 MW of 132kV line capacity).
    
    Args:
        network: PyPSA network object
        logger: Logger instance
        
    Returns:
        Dictionary mapping bus_id -> total line capacity (MW)
    """
    line_cap_into_bus = {}
    
    for idx, line in network.lines.iterrows():
        bus0, bus1 = line.bus0, line.bus1
        
        # Add capacity to both ends of the line
        if bus0 in network.buses.index:
            line_cap_into_bus[bus0] = line_cap_into_bus.get(bus0, 0) + line.s_nom
        if bus1 in network.buses.index:
            line_cap_into_bus[bus1] = line_cap_into_bus.get(bus1, 0) + line.s_nom
    
    logger.info(f"Calculated line capacity for {len(line_cap_into_bus)} buses")
    return line_cap_into_bus


def get_poorly_connected_buses(network: pypsa.Network, logger, max_voltage_kv: float = 66.0) -> dict:
    """
    Identify poorly-connected buses and find their well-connected parent bus.
    
    A poorly-connected bus is one that:
    - Has no line connections (only transformer connections)
    - Is at a voltage level <= max_voltage_kv (e.g., 33kV or 66kV)
    - Is at a lower voltage than the transformer-connected bus
    
    These buses should not have demand allocated directly as they can only be
    served through their transformer connection, creating artificial bottlenecks.
    
    Note: 132kV buses are NOT considered poorly-connected even if they have no
    lines, as they are valid GSP connection points in the ETYS network.
    
    Args:
        network: PyPSA network object
        logger: Logger instance
        max_voltage_kv: Maximum voltage to consider for redirection (default 66kV)
        
    Returns:
        Dictionary mapping poorly_connected_bus -> well_connected_parent_bus
    """
    # Count lines per bus
    line_count = {}
    for _, line in network.lines.iterrows():
        line_count[line.bus0] = line_count.get(line.bus0, 0) + 1
        line_count[line.bus1] = line_count.get(line.bus1, 0) + 1
    
    # Find buses with no lines but with transformer connections
    poorly_connected = {}
    
    for bus in network.buses.index:
        n_lines = line_count.get(bus, 0)
        bus_voltage = network.buses.loc[bus, 'v_nom']
        
        # Only consider low voltage buses (<=66kV) as candidates for redirection
        # 132kV and above are valid GSP connection points
        if n_lines == 0 and bus_voltage <= max_voltage_kv:
            # Bus has no lines and is low voltage - check for transformer connection
            
            # Find transformers connected to this bus
            connected_xfmrs = network.transformers[
                (network.transformers.bus0 == bus) | (network.transformers.bus1 == bus)
            ]
            
            if len(connected_xfmrs) > 0:
                # Find the highest-voltage connected bus via transformer
                best_parent = None
                best_voltage = bus_voltage
                
                for _, xfmr in connected_xfmrs.iterrows():
                    other_bus = xfmr.bus0 if xfmr.bus1 == bus else xfmr.bus1
                    if other_bus in network.buses.index:
                        other_voltage = network.buses.loc[other_bus, 'v_nom']
                        if other_voltage > best_voltage:
                            # Check if parent has lines OR is a valid GSP voltage (>=132kV)
                            if line_count.get(other_bus, 0) > 0 or other_voltage >= 132:
                                best_parent = other_bus
                                best_voltage = other_voltage
                
                if best_parent:
                    poorly_connected[bus] = best_parent
                    logger.debug(f"Bus {bus} ({bus_voltage}kV) is poorly connected - redirecting to {best_parent} ({best_voltage}kV)")
    
    if poorly_connected:
        logger.info(f"Identified {len(poorly_connected)} poorly-connected buses (no lines, voltage <={max_voltage_kv}kV)")
        for bus, parent in poorly_connected.items():
            logger.debug(f"  {bus} ({network.buses.loc[bus, 'v_nom']}kV) -> {parent} ({network.buses.loc[parent, 'v_nom']}kV)")
    
    return poorly_connected


def redirect_demand_from_poorly_connected_buses(
    node_demand: dict,
    node_timeseries: dict,
    network: pypsa.Network,
    logger
) -> tuple:
    """
    Redirect demand from poorly-connected buses to their well-connected parent buses.
    
    This prevents demand from being placed on buses that have no line connections
    and are only reachable via transformers, which can cause artificial bottlenecks.
    
    Args:
        node_demand: Dictionary of node_id -> annual demand (GWh)
        node_timeseries: Dictionary of node_id -> demand timeseries
        network: PyPSA network object
        logger: Logger instance
        
    Returns:
        Tuple of (updated_node_demand, updated_node_timeseries)
    """
    poorly_connected = get_poorly_connected_buses(network, logger)
    
    if not poorly_connected:
        return node_demand, node_timeseries
    
    # Redirect demand
    redirected_count = 0
    redirected_demand_gwh = 0.0
    
    updated_demand = dict(node_demand)
    updated_timeseries = dict(node_timeseries) if node_timeseries else {}
    
    for bus, parent in poorly_connected.items():
        if bus in updated_demand:
            demand = updated_demand.pop(bus)
            redirected_demand_gwh += demand
            redirected_count += 1
            
            if parent not in updated_demand:
                updated_demand[parent] = 0.0
            updated_demand[parent] += demand
            
            # Handle timeseries
            if bus in updated_timeseries:
                ts = updated_timeseries.pop(bus)
                if parent not in updated_timeseries:
                    updated_timeseries[parent] = ts.copy()
                else:
                    updated_timeseries[parent] = updated_timeseries[parent] + ts
            
            logger.info(f"Redirected {demand:.2f} GWh from {bus} to {parent}")
    
    if redirected_count > 0:
        logger.info(f"Redirected {redirected_demand_gwh:.1f} GWh from {redirected_count} poorly-connected buses")
    
    return updated_demand, updated_timeseries


def get_isolated_subnetwork_buses(network: pypsa.Network, logger) -> dict:
    """
    Identify buses in isolated sub-networks and map them to nearest main network bus.
    
    An isolated sub-network is a connected component that is not the main (largest)
    network. Demand should not be allocated to buses in isolated sub-networks
    because they cannot receive power from the main grid.
    
    This function computes network connectivity using networkx, which works
    on networks both before and after optimization.
    
    Args:
        network: PyPSA network object
        logger: Logger instance
        
    Returns:
        Dictionary mapping isolated_bus -> nearest_main_network_bus
    """
    from pyproj import Transformer
    import numpy as np
    import networkx as nx
    
    # Build connectivity graph from lines and transformers
    G = nx.Graph()
    
    # Add all buses as nodes
    for bus in network.buses.index:
        G.add_node(bus)
    
    # Add edges from lines
    for _, line in network.lines.iterrows():
        G.add_edge(line.bus0, line.bus1)
    
    # Add edges from transformers
    for _, trafo in network.transformers.iterrows():
        G.add_edge(trafo.bus0, trafo.bus1)
    
    # Add edges from links (HVDC, etc.)
    for _, link in network.links.iterrows():
        G.add_edge(link.bus0, link.bus1)
    
    # Find connected components
    components = list(nx.connected_components(G))
    
    if len(components) <= 1:
        logger.debug("Network is fully connected - no isolated sub-networks")
        return {}
    
    # Sort by size to find main component
    components = sorted(components, key=len, reverse=True)
    main_component = components[0]
    isolated_components = components[1:]
    
    logger.info(f"Found {len(components)} connected components")
    logger.info(f"  Main component: {len(main_component)} buses")
    for i, comp in enumerate(isolated_components):
        logger.info(f"  Isolated component {i+1}: {len(comp)} buses - {list(comp)[:5]}{'...' if len(comp) > 5 else ''}")
    
    # Get all isolated buses
    isolated_buses = set()
    for comp in isolated_components:
        isolated_buses.update(comp)
    
    if not isolated_buses:
        return {}
    
    # Get main network buses for nearest-neighbor matching
    main_buses = network.buses.loc[list(main_component)]
    
    # Detect coordinate system
    x_range = main_buses['x'].max() - main_buses['x'].min()
    is_osgb36 = x_range > 1000  # OSGB36 uses meters (range ~600km)
    
    if is_osgb36:
        logger.debug("Detected OSGB36 coordinates")
        main_coords = main_buses[['x', 'y']].values
        transformer = None
    else:
        # Convert WGS84 to OSGB36 for distance calculation
        logger.debug("Detected WGS84 coordinates - converting to OSGB36")
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        main_coords = np.array([
            transformer.transform(row['x'], row['y']) 
            for _, row in main_buses.iterrows()
        ])
    
    main_bus_names = main_buses.index.tolist()
    
    # Map each isolated bus to nearest main network bus
    isolated_mapping = {}
    
    for bus in isolated_buses:
        if bus not in network.buses.index:
            continue
            
        bus_x = network.buses.loc[bus, 'x']
        bus_y = network.buses.loc[bus, 'y']
        
        if is_osgb36:
            site_x, site_y = bus_x, bus_y
        else:
            site_x, site_y = transformer.transform(bus_x, bus_y)
        
        # Find nearest main network bus
        distances = np.sqrt((main_coords[:, 0] - site_x)**2 + (main_coords[:, 1] - site_y)**2)
        nearest_idx = np.argmin(distances)
        nearest_bus = main_bus_names[nearest_idx]
        
        isolated_mapping[bus] = nearest_bus
    
    logger.info(f"Mapped {len(isolated_mapping)} isolated buses to main network")
    
    return isolated_mapping


def redirect_demand_from_isolated_subnetworks(
    node_demand: dict,
    node_timeseries: dict,
    network: pypsa.Network,
    logger
) -> tuple:
    """
    Redirect demand from buses in isolated sub-networks to the main network.
    
    Isolated sub-networks cannot receive power from the main grid, so any
    demand placed on them will result in load shedding. This function
    redirects that demand to the nearest bus in the main network.
    
    Args:
        node_demand: Dictionary of node_id -> annual demand (GWh)
        node_timeseries: Dictionary of node_id -> demand timeseries
        network: PyPSA network object
        logger: Logger instance
        
    Returns:
        Tuple of (updated_node_demand, updated_node_timeseries)
    """
    isolated_mapping = get_isolated_subnetwork_buses(network, logger)
    
    if not isolated_mapping:
        return node_demand, node_timeseries
    
    # Redirect demand
    redirected_count = 0
    redirected_demand_gwh = 0.0
    
    updated_demand = dict(node_demand)
    updated_timeseries = dict(node_timeseries) if node_timeseries else {}
    
    for bus, target_bus in isolated_mapping.items():
        if bus in updated_demand:
            demand = updated_demand.pop(bus)
            redirected_demand_gwh += demand
            redirected_count += 1
            
            if target_bus not in updated_demand:
                updated_demand[target_bus] = 0.0
            updated_demand[target_bus] += demand
            
            # Handle timeseries
            if bus in updated_timeseries:
                ts = updated_timeseries.pop(bus)
                if target_bus not in updated_timeseries:
                    updated_timeseries[target_bus] = ts.copy()
                else:
                    updated_timeseries[target_bus] = updated_timeseries[target_bus] + ts
            
            logger.debug(f"Redirected {demand:.2f} GWh from isolated bus {bus} to {target_bus}")
    
    if redirected_count > 0:
        logger.info(f"Redirected {redirected_demand_gwh:.1f} GWh from {redirected_count} buses in isolated sub-networks to main network")
    
    return updated_demand, updated_timeseries


def redistribute_demand_by_transformer_capacity(
    node_weights: dict,
    network: pypsa.Network,
    peak_demand_mw: float,
    logger,
    safety_margin: float = 0.95
) -> dict:
    """
    Redistribute demand weights to respect transformer and line capacity constraints.
    
    Uses an iterative approach to:
    1. Identify buses where demand exceeds capacity
    2. Cap those buses at their capacity
    3. Redistribute excess to buses with headroom
    4. Repeat until all constraints are satisfied
    
    Args:
        node_weights: Dictionary of node_id -> demand weight (sum to 1.0)
        network: PyPSA network object  
        peak_demand_mw: Estimated peak total demand in MW
        logger: Logger instance
        safety_margin: Fraction of capacity to use (default 0.95 = 95%)
        
    Returns:
        Adjusted node_weights dictionary respecting capacity constraints
    """
    logger.info("Redistributing demand to respect transformer and line capacity constraints...")
    
    # Get transformer capacity into each bus
    tf_cap_into_bus = get_transformer_capacity_into_buses(network, logger)
    
    # Get line capacity into each bus (important for 132kV buses)
    line_cap_into_bus = get_line_capacity_into_buses(network, logger)
    
    # Pre-calculate effective capacity for each 132kV bus
    # Effective capacity = min(transformer_cap, line_cap) for buses with transformers
    effective_cap_bus = {}
    for node_id in node_weights.keys():
        if node_id not in network.buses.index:
            continue
        voltage = network.buses.loc[node_id, 'v_nom']
        tf_cap = tf_cap_into_bus.get(node_id, 0)
        line_cap = line_cap_into_bus.get(node_id, float('inf'))
        
        if voltage <= 132 and tf_cap > 0:
            effective_cap = min(tf_cap, line_cap) if line_cap > 0 else tf_cap
            effective_cap_bus[node_id] = effective_cap * safety_margin
    
    # Iterative redistribution - keep going until no more bottlenecks
    max_iterations = 10
    total_redistributed = 0
    adjusted_weights = node_weights.copy()
    
    for iteration in range(max_iterations):
        # Calculate expected demand at each node
        node_expected_demand = {node: weight * peak_demand_mw for node, weight in adjusted_weights.items()}
        
        # Find bottlenecked buses
        bottleneck_buses = {}
        excess_demand = 0
        
        for node_id, max_cap in effective_cap_bus.items():
            expected_demand = node_expected_demand.get(node_id, 0)
            if expected_demand > max_cap:
                excess = expected_demand - max_cap
                bottleneck_buses[node_id] = {
                    'expected_demand': expected_demand,
                    'max_capacity': max_cap,
                    'excess': excess
                }
                excess_demand += excess
        
        if not bottleneck_buses:
            if iteration == 0:
                logger.info("No capacity bottlenecks detected - no redistribution needed")
            else:
                logger.info(f"All constraints satisfied after {iteration} iteration(s)")
            break
        
        if iteration == 0:
            logger.info(f"Found {len(bottleneck_buses)} bottlenecked buses with {excess_demand:.0f} MW excess demand")
            # Log top bottlenecks
            sorted_bottlenecks = sorted(bottleneck_buses.items(), key=lambda x: x[1]['excess'], reverse=True)
            for bus_id, info in sorted_bottlenecks[:5]:
                tf_cap = tf_cap_into_bus.get(bus_id, 0)
                line_cap = line_cap_into_bus.get(bus_id, float('inf'))
                line_cap_str = f", line: {line_cap:.0f} MW" if line_cap < float('inf') else ""
                logger.info(f"  {bus_id}: {info['expected_demand']:.0f} MW demand vs tf: {tf_cap:.0f} MVA{line_cap_str}, max: {info['max_capacity']:.0f}, excess: {info['excess']:.0f} MW")
        
        total_redistributed += excess_demand
        
        # Cap bottlenecked buses at their max capacity
        for bus_id, info in bottleneck_buses.items():
            adjusted_weights[bus_id] = info['max_capacity'] / peak_demand_mw
        
        # Find buses that can receive redistributed demand
        recipient_buses = {}
        for node_id, weight in adjusted_weights.items():
            if node_id in bottleneck_buses:
                continue
            if node_id not in network.buses.index:
                continue
            
            voltage = network.buses.loc[node_id, 'v_nom']
            expected_demand = node_expected_demand.get(node_id, 0)
            
            # Transmission buses (275kV+) can take unlimited additional load
            if voltage >= 275:
                recipient_buses[node_id] = {
                    'weight': weight,
                    'available': float('inf')
                }
            # Distribution buses with headroom
            elif node_id in effective_cap_bus:
                available = effective_cap_bus[node_id] - expected_demand
                if available > 10:  # Only consider if >10 MW headroom
                    recipient_buses[node_id] = {
                        'weight': weight,
                        'available': available
                    }
        
        if not recipient_buses:
            logger.warning(f"No suitable recipient buses found in iteration {iteration+1}!")
            break
        
        # Redistribute excess proportionally to recipient buses based on weight
        total_recipient_weight = sum(info['weight'] for info in recipient_buses.values())
        if total_recipient_weight > 0:
            excess_weight = excess_demand / peak_demand_mw
            for node_id, info in recipient_buses.items():
                share = info['weight'] / total_recipient_weight
                additional_weight = excess_weight * share
                adjusted_weights[node_id] = adjusted_weights.get(node_id, 0) + additional_weight
        
        # Note: We don't normalize here to preserve the capped values
        # The total weight will still sum to 1.0 because we only moved weight around
    
    else:
        logger.warning(f"Reached max iterations ({max_iterations}) - some bottlenecks may remain")
    
    # Final verification
    final_violations = []
    for node_id, max_cap in effective_cap_bus.items():
        final_demand = adjusted_weights.get(node_id, 0) * peak_demand_mw
        if final_demand > max_cap * 1.01:  # 1% tolerance
            final_violations.append((node_id, final_demand, max_cap))
    
    if final_violations:
        logger.warning(f"{len(final_violations)} buses still exceed capacity:")
        for node_id, demand, cap in sorted(final_violations, key=lambda x: x[1]-x[2], reverse=True)[:5]:
            logger.warning(f"  {node_id}: {demand:.0f} MW vs {cap:.0f} MW capacity")
    else:
        logger.info(f"Successfully redistributed {total_redistributed:.0f} MW total")
    
    return adjusted_weights


def distribute_demand_to_etys_nodes(
    fes_demand: pd.DataFrame,
    fes_timeseries: pd.DataFrame,
    network: pypsa.Network,
    logger
):
    """
    Distribute historical GB demand across ETYS network nodes using GSP-based weights.
    
    For historical scenarios, we have total GB demand from ESPENI but need to distribute
    it spatially across the ~380 GSP nodes that connect to ETYS buses.
    
    Uses:
    1. FES Appendix E regional breakdown for per-GSP demand weights
    2. Dem_per_node from GB_network.xlsx for GSP → ETYS node mapping
    
    The weights are derived from FES data which provides demand by GSP, then mapped
    to ETYS network buses using the Dem_per_node mapping.
    """
    logger.info("Distributing historical demand to ETYS nodes")
    
    # Get available ETYS buses
    available_buses = set(network.buses.index)
    logger.info(f"ETYS network has {len(available_buses)} buses")
    
    try:
        # Step 1: Load FES Appendix E regional breakdown for GSP demand weights
        fes_appendix_e_path = Path("data/network/ETYS/Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb")
        
        if fes_appendix_e_path.exists():
            logger.info("Loading FES Appendix E regional breakdown for GSP weights...")
            
            # Read Active demand sheet (has demand by GSP and scenario)
            active_demand = pd.read_excel(
                fes_appendix_e_path, 
                sheet_name='Active', 
                engine='pyxlsb', 
                skiprows=3,
                header=None,
                names=['scenario', 'GSP', 'DemandPk', 'DemandAM', 'DemandPM', 'type', 'year', 'col7', 'col8', 'col9']
            )
            
            # Filter to valid scenarios and clean year
            active_demand = active_demand[active_demand['scenario'].isin(['FS', 'ST', 'CT', 'LW'])]
            active_demand['year'] = active_demand['year'].astype(int) + 2000
            
            # Use year 2024 as baseline (closest to historical period in FES23)
            # Take one scenario (LW = Leading the Way as a reasonable baseline)
            baseline_demand = active_demand[(active_demand['year'] == 2024) & (active_demand['scenario'] == 'LW')]
            
            # Sum demand by GSP (sum across all demand types: Commercial, Residential, etc.)
            gsp_demand = baseline_demand.groupby('GSP').agg({
                'DemandPk': 'sum',
                'DemandAM': 'sum', 
                'DemandPM': 'sum'
            }).reset_index()
            
            # Use average of AM and PM as representative demand weight
            gsp_demand['weight'] = (gsp_demand['DemandAM'] + gsp_demand['DemandPM']) / 2
            total_weight = gsp_demand['weight'].sum()
            gsp_demand['weight_norm'] = gsp_demand['weight'] / total_weight
            
            logger.info(f"Loaded demand weights for {len(gsp_demand)} GSPs from FES Appendix E")
            logger.info(f"Total demand (DemandPk): {gsp_demand['DemandPk'].sum():.0f} MW")
        else:
            logger.warning("FES Appendix E file not found - using equal weights per GSP Group")
            gsp_demand = None
        
        # Step 2: Load GSP -> Node mapping from GB_network.xlsx
        node_mapping_path = Path("data/network/ETYS/GB_network.xlsx")
        dem_per_node = pd.read_excel(node_mapping_path, sheet_name="Dem_per_node")
        logger.info(f"Loaded GSP->Node mapping: {len(dem_per_node)} entries")
        
        # Step 3: Create node-level weights
        # Each GSP maps to one or more ETYS nodes with a percentage
        node_weights = {}
        
        if gsp_demand is not None:
            # Use FES demand weights for GSPs
            gsp_weight_map = dict(zip(gsp_demand['GSP'], gsp_demand['weight_norm']))
            
            for _, row in dem_per_node.iterrows():
                gsp_id = row['GSP Id']
                node_id = row['Node Id']
                within_group_pct = row['Dem as % of demand within the GSP Group ID per each node']
                
                # Get GSP weight (0 if not found)
                gsp_weight = gsp_weight_map.get(gsp_id, 0)
                
                # Node weight = GSP weight * within-group percentage
                node_weight = gsp_weight * within_group_pct
                
                if node_id in node_weights:
                    node_weights[node_id] += node_weight
                else:
                    node_weights[node_id] = node_weight
                    
            logger.info(f"Created weights for {len(node_weights)} unique nodes using FES GSP weights")
        else:
            # Fallback: equal weight per GSP Group, then distribute within group
            n_groups = dem_per_node['GSP Group ID'].nunique()
            group_weight = 1.0 / n_groups
            
            for _, row in dem_per_node.iterrows():
                node_id = row['Node Id']
                within_group_pct = row['Dem as % of demand within the GSP Group ID per each node']
                overall_weight = group_weight * within_group_pct
                
                if node_id in node_weights:
                    node_weights[node_id] += overall_weight
                else:
                    node_weights[node_id] = overall_weight
                    
            logger.info(f"Created weights for {len(node_weights)} unique nodes using equal GSP Group weights")
        
        # Step 4: Match node IDs to ETYS buses
        matched_nodes = set(node_weights.keys()) & available_buses
        logger.info(f"Direct match: {len(matched_nodes)} nodes to ETYS buses")
        
        # Normalize weights to sum to 1.0
        total_weight = sum(node_weights.values())
        if total_weight > 0:
            node_weights = {k: v/total_weight for k, v in node_weights.items()}
        
        # Get total demand from timeseries
        # Detect time resolution
        if len(fes_timeseries) >= 2:
            time_step_hours = (fes_timeseries.index[1] - fes_timeseries.index[0]).total_seconds() / 3600.0
        else:
            time_step_hours = 1.0
        
        total_demand_gwh = (fes_timeseries.iloc[:, 0].sum() * time_step_hours) / 1000.0
        logger.info(f"Total GB demand to distribute: {total_demand_gwh:.1f} GWh")
        
        # Step 4b: Redirect demand from buses in isolated sub-networks to main network
        # This prevents load shedding on buses that are disconnected from the main grid
        temp_demand = {node: weight * total_demand_gwh for node, weight in node_weights.items()}
        temp_timeseries = {node: fes_timeseries.iloc[:, 0] * weight for node, weight in node_weights.items()}
        
        temp_demand, temp_timeseries = redirect_demand_from_isolated_subnetworks(
            node_demand=temp_demand,
            node_timeseries=temp_timeseries,
            network=network,
            logger=logger
        )
        
        # Convert back to weights
        new_total = sum(temp_demand.values())
        if new_total > 0:
            node_weights = {node: demand / new_total for node, demand in temp_demand.items()}
        
        # Step 5: Redirect demand from poorly-connected buses (DISABLED for now)
        # This feature was causing numerical issues - needs more investigation
        # temp_demand = {node: weight * total_demand_gwh for node, weight in node_weights.items()}
        # temp_timeseries = {node: fes_timeseries.iloc[:, 0] * weight for node, weight in node_weights.items()}
        #
        # temp_demand, temp_timeseries = redirect_demand_from_poorly_connected_buses(
        #     node_demand=temp_demand,
        #     node_timeseries=temp_timeseries,
        #     network=network,
        #     logger=logger
        # )
        #
        # # Convert back to weights
        # new_total = sum(temp_demand.values())
        # if new_total > 0:
        #     node_weights = {node: demand / new_total for node, demand in temp_demand.items()}
        
        # Step 6: Redistribute demand to respect transformer capacity constraints
        # Estimate peak demand from timeseries (MW)
        peak_demand_mw = fes_timeseries.iloc[:, 0].max()
        logger.info(f"Estimated peak demand: {peak_demand_mw:.0f} MW")
        
        # Redistribute demand away from bottlenecked buses
        # Use 90% safety margin to account for AC power flow limitations
        # (transformers can't always deliver full rated capacity due to network physics)
        node_weights = redistribute_demand_by_transformer_capacity(
            node_weights=node_weights,
            network=network,
            peak_demand_mw=peak_demand_mw,
            logger=logger,
            safety_margin=0.90  # Use 90% of transformer capacity for extra headroom
        )
        
        # Create demand DataFrame with node-level demand
        demand_list = []
        for node_id, weight in node_weights.items():
            demand_list.append({'node': node_id, 'demand': total_demand_gwh * weight})
        
        demand_data = pd.DataFrame(demand_list).set_index('node')
        
        # Create timeseries with node columns
        timeseries_dict = {}
        for node_id, weight in node_weights.items():
            timeseries_dict[node_id] = fes_timeseries.iloc[:, 0] * weight
        
        timeseries_data = pd.DataFrame(timeseries_dict, index=fes_timeseries.index)
        
        logger.info(f"Created demand for {len(demand_data)} nodes")
        logger.info(f"Timeseries shape: {timeseries_data.shape}")
        logger.info(f"Total distributed demand: {demand_data['demand'].sum():.1f} GWh")
        
        return demand_data, timeseries_data
        
    except Exception as e:
        logger.error(f"Error distributing demand to ETYS nodes: {e}")
        logger.warning("Falling back to equal distribution across all buses")
        
        # Fallback: distribute equally across all buses
        n_buses = len(available_buses)
        equal_weight = 1.0 / n_buses if n_buses > 0 else 0
        
        if len(fes_timeseries) >= 2:
            time_step_hours = (fes_timeseries.index[1] - fes_timeseries.index[0]).total_seconds() / 3600.0
        else:
            time_step_hours = 1.0
        
        total_demand_gwh = (fes_timeseries.iloc[:, 0].sum() * time_step_hours) / 1000.0
        
        demand_list = [{'node': bus, 'demand': total_demand_gwh * equal_weight} for bus in available_buses]
        demand_data = pd.DataFrame(demand_list).set_index('node')
        
        timeseries_dict = {bus: fes_timeseries.iloc[:, 0] * equal_weight for bus in available_buses}
        timeseries_data = pd.DataFrame(timeseries_dict, index=fes_timeseries.index)
        
        logger.info(f"Fallback: distributed {total_demand_gwh:.1f} GWh across {n_buses} buses equally")
        
        return demand_data, timeseries_data


def distribute_future_fes_demand_to_etys_nodes(
    fes_demand: pd.DataFrame,
    fes_timeseries: pd.DataFrame,
    network: pypsa.Network,
    logger
):
    """
    Distribute FES future demand projections across ETYS network nodes.
    
    FES API data is indexed by GSP names (e.g., "Abham", "Norwich") but ETYS buses
    use GSP codes (e.g., "ABHA1"). This function:
    1. Maps GSP names to GSP codes using FES 2023 metadata file
    2. Maps GSP codes to ETYS Node IDs using Dem_per_node from GB_network.xlsx
    3. Distributes FES demand to network buses based on this mapping
    
    Args:
        fes_demand: Demand data indexed by GSP names (from FES API)
        fes_timeseries: Time-varying demand profiles by GSP names
        network: PyPSA network with ETYS topology
        logger: Logger instance
    
    Returns:
        Tuple of (demand_data, timeseries_data) indexed by ETYS Node IDs
    """
    logger.info("Distributing FES future demand to ETYS nodes")
    
    available_buses = set(network.buses.index)
    logger.info(f"ETYS network has {len(available_buses)} buses")
    
    # Step 1: Load GSP Name -> GSP Code mapping from FES 2023 metadata
    # (FES 2023 has proper GSP codes; FES 2024 metadata only has numeric IDs)
    gsp_metadata_path = Path("data/network/ETYS/fes2023_regional_breakdown_gsp_info.csv")
    if not gsp_metadata_path.exists():
        logger.warning(f"GSP metadata file not found at {gsp_metadata_path}")
        # Try other years
        for year in [2022, 2021]:
            alt_path = Path(f"data/network/ETYS/fes{year}_regional_breakdown_gsp_info.csv")
            if alt_path.exists():
                gsp_metadata_path = alt_path
                break
    
    if gsp_metadata_path.exists():
        gsp_metadata = pd.read_csv(gsp_metadata_path)
        logger.info(f"Loaded GSP metadata from {gsp_metadata_path}: {len(gsp_metadata)} entries")
        
        # Create Name -> GSP ID (code) mapping
        # The 'Name' column has human-readable names, 'GSP ID' has codes like "ABHA1"
        name_to_code = {}
        for _, row in gsp_metadata.iterrows():
            gsp_name = str(row.get('Name', '')).strip()
            gsp_code = str(row.get('GSP ID', '')).strip()
            if gsp_name and gsp_code:
                # Normalize the name (lowercase, strip whitespace)
                name_to_code[gsp_name.lower()] = gsp_code
        logger.info(f"Created {len(name_to_code)} GSP name -> code mappings")
    else:
        logger.error("No GSP metadata file found - cannot map GSP names to codes")
        name_to_code = {}
    
    # Step 2: Load GSP Code -> Node ID mapping from Dem_per_node
    node_mapping_path = Path("data/network/ETYS/GB_network.xlsx")
    dem_per_node = pd.read_excel(node_mapping_path, sheet_name="Dem_per_node")
    logger.info(f"Loaded Dem_per_node mapping: {len(dem_per_node)} entries")
    
    # Create GSP Code -> [(Node ID, weight)] mapping
    gsp_code_to_nodes = {}
    for _, row in dem_per_node.iterrows():
        gsp_code = str(row.get('GSP Id', '')).strip()
        node_id = str(row.get('Node Id', '')).strip()
        weight_pct = row.get('Dem as % of demand within the GSP Group ID per each node', 0.0)
        
        if gsp_code and node_id and weight_pct > 0:
            if gsp_code not in gsp_code_to_nodes:
                gsp_code_to_nodes[gsp_code] = []
            gsp_code_to_nodes[gsp_code].append((node_id, weight_pct))
    
    logger.info(f"Created mappings for {len(gsp_code_to_nodes)} GSP codes to ETYS nodes")
    
    # Step 3: Map FES demand to ETYS nodes
    node_demand = {}
    node_timeseries_data = {}
    matched_gsps = 0
    unmatched_gsps = []
    
    for gsp_name in fes_demand.index:
        gsp_name_lower = str(gsp_name).lower().strip()
        annual_demand_gwh = fes_demand.loc[gsp_name].iloc[0]
        
        # Skip zero or negligible demand
        if annual_demand_gwh < 0.001:
            continue
        
        # Get GSP code from name
        gsp_code = name_to_code.get(gsp_name_lower)
        
        if gsp_code and gsp_code in gsp_code_to_nodes:
            # Map to ETYS nodes using Dem_per_node weights
            node_mappings = gsp_code_to_nodes[gsp_code]
            
            # Normalize weights for this GSP
            total_weight = sum(w for _, w in node_mappings)
            
            for node_id, weight_pct in node_mappings:
                if node_id in available_buses:
                    weight_norm = weight_pct / total_weight if total_weight > 0 else 1.0 / len(node_mappings)
                    
                    # Add to node demand
                    if node_id not in node_demand:
                        node_demand[node_id] = 0.0
                    node_demand[node_id] += annual_demand_gwh * weight_norm
                    
                    # Handle timeseries
                    if gsp_name in fes_timeseries.columns:
                        if node_id not in node_timeseries_data:
                            node_timeseries_data[node_id] = pd.Series(0.0, index=fes_timeseries.index)
                        node_timeseries_data[node_id] += fes_timeseries[gsp_name] * weight_norm
            
            matched_gsps += 1
        else:
            unmatched_gsps.append((gsp_name, annual_demand_gwh))
    
    logger.info(f"Mapped {matched_gsps} GSPs to {len(node_demand)} ETYS nodes")
    
    # Handle unmatched GSPs by distributing their demand proportionally
    if unmatched_gsps:
        total_unmatched_gwh = sum(d for _, d in unmatched_gsps)
        logger.warning(f"{len(unmatched_gsps)} GSPs ({total_unmatched_gwh:.1f} GWh) could not be mapped:")
        for gsp_name, demand in unmatched_gsps[:10]:
            logger.warning(f"  - {gsp_name}: {demand:.2f} GWh")
        if len(unmatched_gsps) > 10:
            logger.warning(f"  ... and {len(unmatched_gsps) - 10} more")
        
        # Distribute unmatched demand proportionally to existing nodes
        if node_demand:
            total_existing_demand = sum(node_demand.values())
            for node_id in node_demand:
                proportion = node_demand[node_id] / total_existing_demand
                node_demand[node_id] += total_unmatched_gwh * proportion
                
                # Also adjust timeseries
                if node_id in node_timeseries_data:
                    # Get total unmatched timeseries
                    for gsp_name, _ in unmatched_gsps:
                        if gsp_name in fes_timeseries.columns:
                            node_timeseries_data[node_id] += fes_timeseries[gsp_name] * proportion
            
            logger.info(f"Redistributed {total_unmatched_gwh:.1f} GWh from unmatched GSPs")
    
    # Step 3.5: Redirect demand from buses in isolated sub-networks to main network
    # This prevents load shedding on buses that are disconnected from the main grid
    node_demand, node_timeseries_data = redirect_demand_from_isolated_subnetworks(
        node_demand=node_demand,
        node_timeseries=node_timeseries_data,
        network=network,
        logger=logger
    )
    
    # Step 3.6: Redirect demand from poorly-connected buses (DISABLED for now)
    # This feature was causing numerical issues - needs more investigation
    # node_demand, node_timeseries_data = redirect_demand_from_poorly_connected_buses(
    #     node_demand=node_demand,
    #     node_timeseries=node_timeseries_data,
    #     network=network,
    #     logger=logger
    # )
    
    # Step 4: Redistribute demand to respect transformer capacity constraints
    # This prevents load shedding due to demand exceeding transformer capacity
    total_demand_gwh = sum(node_demand.values())
    
    # Estimate peak demand from timeseries (MW) - use max of sum across all nodes
    if node_timeseries_data:
        timeseries_temp = pd.DataFrame(node_timeseries_data)
        peak_demand_mw = timeseries_temp.sum(axis=1).max()
    else:
        # Fallback: estimate from total GWh assuming 8760h and load factor
        peak_demand_mw = total_demand_gwh * 1000 / 8760 / 0.6  # 60% load factor
    
    logger.info(f"Estimated peak demand: {peak_demand_mw:.0f} MW")
    
    # Convert node_demand to weights (sum to 1.0)
    node_weights = {node: demand / total_demand_gwh for node, demand in node_demand.items()}
    
    # Apply transformer capacity redistribution
    adjusted_weights = redistribute_demand_by_transformer_capacity(
        node_weights=node_weights,
        network=network,
        peak_demand_mw=peak_demand_mw,
        logger=logger,
        safety_margin=0.90  # Use 90% of transformer capacity for extra headroom
    )
    
    # Apply adjusted weights to demand and timeseries
    node_demand = {node: weight * total_demand_gwh for node, weight in adjusted_weights.items()}
    
    # Recreate timeseries with adjusted weights
    for node_id in node_timeseries_data:
        if node_id in adjusted_weights:
            old_weight = node_weights.get(node_id, 0)
            new_weight = adjusted_weights.get(node_id, 0)
            if old_weight > 0:
                # Scale timeseries by ratio of new/old weights
                scale_factor = new_weight / old_weight
                node_timeseries_data[node_id] = node_timeseries_data[node_id] * scale_factor
    
    # Create output DataFrames
    demand_list = []
    for node_id, demand_gwh in node_demand.items():
        demand_list.append({'node': node_id, 'demand': demand_gwh})
    
    demand_data = pd.DataFrame(demand_list).set_index('node')
    demand_data.columns = [fes_demand.columns[0]] if len(fes_demand.columns) > 0 else ['demand']
    
    timeseries_data = pd.DataFrame(node_timeseries_data, index=fes_timeseries.index)
    
    # Log summary
    total_demand = demand_data['demand'].sum() if 'demand' in demand_data.columns else demand_data.sum().iloc[0]
    logger.info(f"Created ETYS demand for {len(demand_data)} nodes")
    logger.info(f"Total annual demand: {total_demand:.1f} GWh")
    logger.info(f"Timeseries shape: {timeseries_data.shape}")
    
    return demand_data, timeseries_data


def aggregate_demand_to_network_topology(
    fes_demand: pd.DataFrame, 
    fes_timeseries: pd.DataFrame,
    network_model: str,
    network: pypsa.Network,
    spatial_mapping,
    logger,
    is_historical: bool = False
):
    """
    Aggregate FES demand data to match the network topology.
    
    Args:
        fes_demand: Demand data by node/GSP
        fes_timeseries: Time-varying demand profiles
        network_model: "ETYS", "Reduced", or "Zonal"
        network: PyPSA network object
        spatial_mapping: Optional mapping DataFrame for aggregation
        logger: Logger instance
        is_historical: If True, data comes from ESPENI/historical sources (already per-bus)
    """
    
    if network_model == "ETYS":
        if is_historical:
            # Historical ETYS: distribute GB demand across ETYS nodes
            logger.info("ETYS network - historical scenario, distributing demand to nodes")
            return distribute_demand_to_etys_nodes(
                fes_demand, fes_timeseries, network, logger
            )
        else:
            # Future ETYS: map GSP names to ETYS Node IDs
            logger.info("ETYS network - future scenario, mapping GSP names to node IDs")
            return distribute_future_fes_demand_to_etys_nodes(
                fes_demand, fes_timeseries, network, logger
            )
    
    elif network_model in ["Reduced", "Zonal"]:
        logger.info(f"Aggregating demand for {network_model} network")
        
        # Get target buses from the network
        target_buses = list(network.buses.index)
        logger.info(f"Target network has {len(target_buses)} buses")
        
        # For historical scenarios, the timeseries is already total GB demand
        # We need to distribute it to the network buses
        if is_historical:
            logger.info("Historical scenario detected - distributing GB demand to network buses")
            
            # The timeseries has a single column with total demand
            # We need to distribute it proportionally based on zone weights
            
            # Load zone-specific demand weights for Zonal networks
            zone_weights = load_zone_demand_weights(network_model, logger)
            
            # Filter to only GB buses (exclude external/interconnector buses)
            gb_target_buses = [b for b in target_buses if not b.startswith('HVDC_External')]
            n_buses = len(gb_target_buses)
            
            if zone_weights and network_model in ("Zonal", "Reduced"):
                logger.info(f"Using weighted demand distribution for {n_buses} GB buses ({network_model} network)")
                
                # Create demand factors from zone weights
                bus_demand_factors = {}
                matched_buses = 0
                unmatched_buses = []
                
                for bus in gb_target_buses:
                    if bus in zone_weights:
                        bus_demand_factors[bus] = zone_weights[bus]
                        matched_buses += 1
                    else:
                        unmatched_buses.append(bus)
                
                # Handle unmatched buses by distributing remaining weight equally
                if unmatched_buses:
                    matched_weight = sum(bus_demand_factors.values())
                    remaining_weight = 1.0 - matched_weight
                    fallback_weight = remaining_weight / len(unmatched_buses) if unmatched_buses else 0
                    
                    logger.warning(f"  {len(unmatched_buses)} buses not in definitions file, using fallback weight: {fallback_weight:.4f}")
                    for bus in unmatched_buses:
                        bus_demand_factors[bus] = fallback_weight
                        logger.debug(f"    Unmatched bus: {bus}")
                
                # Verify weights sum to 1.0
                total_weight = sum(bus_demand_factors.values())
                if abs(total_weight - 1.0) > 0.01:
                    logger.warning(f"Demand weights sum to {total_weight:.4f}, normalizing...")
                    bus_demand_factors = {k: v/total_weight for k, v in bus_demand_factors.items()}
                
                logger.info(f"  Matched {matched_buses}/{n_buses} buses to demand weights")
            else:
                # Fallback to equal distribution
                logger.info(f"Using equal demand distribution for {n_buses} buses (no zone weights available)")
                equal_weight = 1.0 / n_buses if n_buses > 0 else 0
                bus_demand_factors = {bus: equal_weight for bus in gb_target_buses}
            
            # Log demand distribution summary
            for bus, factor in sorted(bus_demand_factors.items()):
                logger.debug(f"  {bus}: {factor*100:.2f}%")
            
            # Detect time resolution from timeseries index
            if len(fes_timeseries) >= 2:
                time_step_hours = (fes_timeseries.index[1] - fes_timeseries.index[0]).total_seconds() / 3600.0
            else:
                time_step_hours = 1.0  # Default to hourly
            logger.info(f"Detected time resolution: {time_step_hours} hours per step")
            
            # Create aggregated demand data: one row per bus
            aggregated_demand_list = []
            total_gwh = 0
            for bus in target_buses:
                # Get weight (0 for external buses)
                weight = bus_demand_factors.get(bus, 0.0)
                # Total annual demand = sum of timeseries * timestep size (hours) / 1000 (MWh to GWh)
                annual_gwh = (fes_timeseries.iloc[:, 0].sum() * time_step_hours) / 1000.0 * weight
                aggregated_demand_list.append({'bus': bus, 'demand': annual_gwh})
                total_gwh += annual_gwh
            
            # Create DataFrame with demand indexed by bus
            aggregated_demand = pd.DataFrame(aggregated_demand_list).set_index('bus')
            aggregated_demand.columns = [fes_demand.columns[0]] if len(fes_demand.columns) > 0 else ["demand"]
            
            # Create timeseries with per-bus columns
            # Take the total GB demand and split by zone weights
            aggregated_timeseries_dict = {}
            for bus in target_buses:
                weight = bus_demand_factors.get(bus, 0.0)
                aggregated_timeseries_dict[bus] = fes_timeseries.iloc[:, 0] * weight
            
            aggregated_timeseries = pd.DataFrame(aggregated_timeseries_dict, index=fes_timeseries.index)
            
            logger.info(f"Created historical demand for {len(aggregated_demand)} buses")
            logger.info(f"Total annual demand: {total_gwh:.1f} GWh")
            logger.info(f"Timeseries shape: {aggregated_timeseries.shape}")
            return aggregated_demand, aggregated_timeseries
        
        if spatial_mapping is not None:
            # Use mapping file to aggregate
            aggregated_demand, aggregated_timeseries = aggregate_with_mapping(
                fes_demand, fes_timeseries, spatial_mapping, target_buses, logger
            )
        else:
            # Fallback: simple proportional aggregation
            logger.warning("No spatial mapping available - using proportional aggregation")
            aggregated_demand, aggregated_timeseries = aggregate_proportionally(
                fes_demand, fes_timeseries, target_buses, logger
            )
        
        return aggregated_demand, aggregated_timeseries
    
    else:
        raise ValueError(f"Unsupported network model: {network_model}")

def aggregate_with_mapping(
    fes_demand: pd.DataFrame,
    fes_timeseries: pd.DataFrame, 
    spatial_mapping: pd.DataFrame,
    target_buses: list,
    logger
):
    """Aggregate using spatial mapping data.
    
    This function handles the mismatch between:
    - spatial_mapping index: GSP IDs (e.g., 'BEAU_1', 'HARK_1') from zone_definitions.csv
    - fes_demand index: GSP names (e.g., 'Beauly', 'Harker') from FES API
    
    If no GSP ID matches are found, it falls back to weighted distribution using
    population_weight from the zone_definitions file.
    """
    
    logger.info("Performing spatial mapping-based aggregation")
    
    # Create aggregation groups
    aggregation_groups = {}
    for target_bus in target_buses:
        # Find which original nodes/GSPs map to this target bus
        if 'target_bus' in spatial_mapping.columns:
            source_nodes = spatial_mapping[spatial_mapping['target_bus'] == target_bus].index
        else:
            # Assume mapping has target bus as column name
            source_nodes = spatial_mapping[spatial_mapping.iloc[:, 0] == target_bus].index
        
        aggregation_groups[target_bus] = source_nodes
    
    # First pass: try direct matching of GSP IDs to FES demand index
    aggregated_demand_dict = {}
    aggregated_timeseries_dict = {}
    total_matched = 0
    total_sources = 0
    
    for target_bus, source_nodes in aggregation_groups.items():
        total_sources += len(source_nodes)
        # Find intersection of source nodes with available FES data
        available_sources = [node for node in source_nodes if node in fes_demand.index]
        total_matched += len(available_sources)
        
        if available_sources:
            # Sum demand from all source nodes
            total_demand = fes_demand.loc[available_sources].sum().iloc[0]
            aggregated_demand_dict[target_bus] = total_demand
            
            # Sum timeseries from all source nodes  
            available_ts_sources = [node for node in available_sources if node in fes_timeseries.columns]
            if available_ts_sources:
                total_timeseries = fes_timeseries[available_ts_sources].sum(axis=1)
                aggregated_timeseries_dict[target_bus] = total_timeseries
            else:
                # Create zero timeseries if no timeseries data available
                aggregated_timeseries_dict[target_bus] = pd.Series(0.0, index=fes_timeseries.index)
            
            logger.debug(f"Bus {target_bus}: aggregated {len(available_sources)} sources -> {total_demand:.1f} GWh")
        else:
            aggregated_demand_dict[target_bus] = 0.0
            aggregated_timeseries_dict[target_bus] = pd.Series(0.0, index=fes_timeseries.index)
    
    # Check if we have a significant matching problem (< 50% sources matched)
    match_ratio = total_matched / total_sources if total_sources > 0 else 0
    total_aggregated_demand = sum(aggregated_demand_dict.values())
    
    if match_ratio < 0.5 or total_aggregated_demand == 0:
        logger.warning(f"GSP ID matching failed: only {total_matched}/{total_sources} sources matched ({match_ratio:.1%})")
        logger.warning(f"Total aggregated demand: {total_aggregated_demand:.1f} GWh - falling back to weighted distribution")
        
        # The spatial_mapping index contains GSP IDs (like 'BEAU_1') but FES demand
        # uses GSP names (like 'Beauly'). Fall back to weighted distribution.
        
        # Calculate total FES demand
        total_fes_demand = fes_demand.sum().iloc[0]
        total_fes_timeseries = fes_timeseries.sum(axis=1)
        
        logger.info(f"Total FES demand: {total_fes_demand:.1f} GWh")
        
        # Use population_weight from spatial_mapping if available, otherwise equal weights
        if 'population_weight' in spatial_mapping.columns:
            logger.info("Using population_weight from zone definitions for demand distribution")
            
            # Aggregate weights by target_bus
            zone_weights = spatial_mapping.groupby('target_bus')['population_weight'].sum()
            total_weight = zone_weights.sum()
            
            if total_weight > 0:
                normalized_weights = zone_weights / total_weight
                
                # Distribute demand according to weights
                for target_bus in target_buses:
                    weight = normalized_weights.get(target_bus, 0.0)
                    aggregated_demand_dict[target_bus] = total_fes_demand * weight
                    aggregated_timeseries_dict[target_bus] = total_fes_timeseries * weight
                    logger.debug(f"Bus {target_bus}: weight={weight:.4f}, demand={total_fes_demand * weight:.1f} GWh")
            else:
                # Equal distribution fallback
                n_buses = len(target_buses)
                per_bus = total_fes_demand / n_buses if n_buses > 0 else 0
                for target_bus in target_buses:
                    aggregated_demand_dict[target_bus] = per_bus
                    aggregated_timeseries_dict[target_bus] = total_fes_timeseries / n_buses
        else:
            # No weights available - equal distribution
            logger.warning("No population_weight in mapping - using equal distribution")
            n_buses = len(target_buses)
            per_bus = total_fes_demand / n_buses if n_buses > 0 else 0
            for target_bus in target_buses:
                aggregated_demand_dict[target_bus] = per_bus
                aggregated_timeseries_dict[target_bus] = total_fes_timeseries / n_buses
        
        logger.info(f"Redistributed {total_fes_demand:.1f} GWh across {len(target_buses)} buses using weights")
    else:
        # Log any buses that didn't match
        for target_bus, source_nodes in aggregation_groups.items():
            available_sources = [node for node in source_nodes if node in fes_demand.index]
            if not available_sources:
                logger.warning(f"No source data found for target bus {target_bus}")
    
    # Convert to DataFrames
    aggregated_demand = pd.DataFrame({
        fes_demand.columns[0]: aggregated_demand_dict
    })
    aggregated_timeseries = pd.DataFrame(aggregated_timeseries_dict)
    
    logger.info(f"Aggregated to {len(aggregated_demand)} target buses")
    log_dataframe_info(aggregated_demand, logger, "Aggregated demand")
    
    return aggregated_demand, aggregated_timeseries

def aggregate_proportionally(
    fes_demand: pd.DataFrame,
    fes_timeseries: pd.DataFrame,
    target_buses: list, 
    logger
):
    """Fallback: distribute total demand proportionally across target buses."""
    
    logger.info("Performing proportional aggregation (fallback method)")
    
    # Calculate total demand and timeseries
    total_demand = fes_demand.sum().iloc[0]
    total_timeseries = fes_timeseries.sum(axis=1)
    
    # Distribute equally across target buses (could be enhanced with weighting)
    n_buses = len(target_buses)
    per_bus_demand = total_demand / n_buses
    per_bus_timeseries = total_timeseries / n_buses
    
    # Create aggregated data
    aggregated_demand = pd.DataFrame({
        fes_demand.columns[0]: {bus: per_bus_demand for bus in target_buses}
    })
    
    aggregated_timeseries = pd.DataFrame({
        bus: per_bus_timeseries for bus in target_buses
    })
    
    logger.info(f"Distributed {total_demand:.1f} GWh equally across {n_buses} buses ({per_bus_demand:.1f} GWh each)")
    
    return aggregated_demand, aggregated_timeseries

def get_fes_demand_data(fes_scenario: str, modelled_year: int, logger) -> pd.DataFrame:
    """
    Extract demand data for a specific FES scenario and year.
    """
    try:
        # Read the FES data file - use helper function to handle single path
        fes_data_path = get_single_input_path(snakemake.input.fes_data)
        FES_data = pd.read_csv(fes_data_path, low_memory=False)
        logger.info(f"Loaded FES data from: {fes_data_path}")
        logger.info(f"Original FES data shape: {FES_data.shape}")
        
        # Filter for demand building blocks
        demand_blocks = FES_data[FES_data['Building Block ID Number'] == BUILDING_BLOCK_ID]
        logger.info(f"Found {len(demand_blocks)} rows with building block {BUILDING_BLOCK_ID}")
        
        # Filter for the specific scenario (handle different column names across FES versions)
        scenario_column = None
        potential_scenario_columns = ['FES Pathway', 'FES Scenario', 'ï»¿FES Scenario', '\ufeffFES Scenario']
        for col in potential_scenario_columns:
            if col in demand_blocks.columns:
                scenario_column = col
                break
        
        if scenario_column is None:
            logger.error(f"Could not find scenario column. Available columns: {list(demand_blocks.columns)}")
            raise ValueError("Could not find FES Scenario/Pathway column in data")
        
        logger.info(f"Using scenario column: '{scenario_column}'")
        scenario_data = demand_blocks[demand_blocks[scenario_column] == fes_scenario]
        logger.info(f"Found {len(scenario_data)} rows matching scenario '{fes_scenario}'")
        
        if len(scenario_data) == 0:
            available_scenarios = demand_blocks[scenario_column].unique()
            raise ValueError(f"No data found for scenario '{fes_scenario}'. Available scenarios: {available_scenarios}")
        
        # Extract demand for the modelled year
        year_column = str(modelled_year)
        if year_column not in scenario_data.columns:
            available_years = [col for col in scenario_data.columns if col.isdigit()]
            raise ValueError(f"Year {modelled_year} not found in data. Available years: {available_years}")
        
        # Create demand DataFrame - use GSP column for all network models
        # GSP provides the granular spatial demand distribution needed for disaggregation
        # Node ID only contains 2 entries (Direct(SHETL) and Direct(SPTL)) which aren't useful
        network_model = get_network_model()
        index_column = 'GSP'  # Always use GSP for demand data extraction
        logger.info(f"Network model: {network_model}, using GSP column for demand extraction")
            
        # Handle duplicate entries by aggregating them
        demand_data = scenario_data.groupby(index_column)[year_column].sum().to_frame()
        demand_data = demand_data.dropna()  # Remove NaN values
        
        logger.info(f"Processed demand data shape: {demand_data.shape}")
        logger.info(f"Using index column: {index_column}")
        log_dataframe_info(demand_data, logger, "FES demand data")
        logger.info(f"Total annual demand: {demand_data.sum().iloc[0]:.1f} GWh")
        
        return demand_data
        
    except FileNotFoundError:
        logger.error(f"FES data file not found: {get_single_input_path(snakemake.input.fes_data)}")
        raise
    except Exception as e:
        logger.error(f"Error reading FES data: {e}")
        raise

def load_espeni_data():
    file = snakemake.input.espeni
    espeni_data = pd.read_csv(file, index_col=0)
    return espeni_data

def load_egy_data(modelled_year):
    file = snakemake.input.egy
    egy_df = pd.read_excel(file, sheet_name="egy_7649", index_col=0)
    # Select the column for modelled year (simplified - may need adjustment)
    return egy_df

def generate_load_timeseries(demand_data: pd.DataFrame, dataset_name: str, modelled_year: int, logger, demand_year: int = None, profile_year: int = None) -> pd.DataFrame:
    """
    Generate load timeseries data.
    
    Supported datasets:
      - "ESPENI": Historical half-hourly GB demand (recommended for realistic profiles)
      - "eload": eLOAD model hourly profiles from egy_7649_mmc1.xlsx (2010 or 2050)
      - "desstinee": DESSTINEE model hourly profiles from egy_7649_mmc1.xlsx (2010 or 2050)
    
    Args:
        demand_data: FES demand by GSP/node (GWh annual)
        dataset_name: "ESPENI", "eload", or "desstinee"
        modelled_year: Target year for output timestamps
        logger: Logger instance
        demand_year: For ESPENI only - which historical year to use for profile shape
        profile_year: For eload/desstinee - which profile to use (2010 or 2050, auto-selected if None)
    """
    logger.info(f"Generating load timeseries using dataset: {dataset_name}")
    
    # Convert dataset name to lowercase for comparison
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == "eload":
        return generate_eload_profiles_from_excel(demand_data, modelled_year, logger, profile_year)
    elif dataset_name_lower == "desstinee":
        return generate_desstinee_profiles_from_excel(demand_data, modelled_year, logger, profile_year)
    elif dataset_name_lower == "espeni":
        return generate_espeni_profiles(demand_data, modelled_year, logger, demand_year)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: ESPENI, eload, desstinee")


def _load_egy_profile(sheet_name: str, model_name: str, logger, profile_year: int = 2010) -> pd.Series:
    """
    Load hourly UK demand profile from the EGY Excel file.
    
    Args:
        sheet_name: "ELOAD" or "DESSTINEE"
        model_name: Human-readable name for logging
        logger: Logger instance
        profile_year: 2010 or 2050 (which profile to use, defaults to 2010)
        
    Returns:
        pd.Series with hourly demand in GW, indexed by datetime
    """
    try:
        egy_file = snakemake.input.egy
        
        # Handle None explicitly - default to 2010
        if profile_year is None:
            profile_year = 2010
            logger.info(f"profile_year is None, defaulting to 2010")
        
        logger.info(f"Loading {model_name} {profile_year} profile from: {egy_file}")
        
        # Read the sheet - data structure:
        # - Row 0-5: Headers and metadata
        # - Row 6: Country labels (UK, DE, etc.)  
        # - Row 7+: Timestamped data (datetime in col 0, values in UK columns)
        # - Column 1: UK 2010 baseline profile (GW)
        # - Column 5: UK 2050 profile (GW, includes EVs)
        
        df = pd.read_excel(egy_file, sheet_name=sheet_name, header=None)
        
        # Select appropriate column based on profile year
        if profile_year == 2010:
            data_col = 1
        elif profile_year == 2050:
            data_col = 5
        else:
            raise ValueError(f"Invalid profile_year: {profile_year}. Must be 2010 or 2050.")
        
        # Extract timestamp column (col 0) and UK data column, starting from row 7
        timestamps = pd.to_datetime(df.iloc[7:, 0])
        uk_demand_gw = pd.to_numeric(df.iloc[7:, data_col], errors='coerce')
        
        # Create series with datetime index
        profile = pd.Series(uk_demand_gw.values, index=timestamps, name='demand_gw')
        profile = profile.dropna()
        
        logger.info(f"Loaded {model_name} {profile_year} profile: {len(profile)} hourly timesteps")
        logger.info(f"  Date range: {profile.index.min()} to {profile.index.max()}")
        logger.info(f"  Demand range: {profile.min():.1f} to {profile.max():.1f} GW")
        logger.info(f"  Annual total: {profile.sum():.1f} GWh")
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to load {model_name} profile: {e}")
        raise


def _scale_hourly_profile_to_fes_demand(
    hourly_profile: pd.Series,
    demand_data: pd.DataFrame, 
    modelled_year: int,
    model_name: str,
    logger
) -> pd.DataFrame:
    """
    Scale an hourly profile to match FES demand projections.
    
    Args:
        hourly_profile: Hourly demand in GW (typically from 2010 base year)
        demand_data: FES demand by GSP/node (GWh annual)
        modelled_year: Target year for output timestamps
        model_name: Name for logging
        logger: Logger instance
        
    Returns:
        DataFrame with columns for each node, scaled to FES annual demand
    """
    logger.info(f"Scaling {model_name} profile to FES demand for year {modelled_year}")
    
    # Create time index for the modelled year (hourly - matching source data)
    start_date = f"{modelled_year}-01-01 00:00:00"
    end_date = f"{modelled_year}-12-31 23:00:00"
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')
    
    logger.info(f"Target time index: {len(time_index)} hourly timesteps")
    
    # Map source profile to target year by day-of-year and hour
    # (handles leap years by mapping Dec 31 if needed)
    source_doy = hourly_profile.index.dayofyear
    source_hour = hourly_profile.index.hour
    
    target_doy = time_index.dayofyear
    target_hour = time_index.hour
    
    profile_values = []
    for doy, hour in zip(target_doy, target_hour):
        # Find matching timestep in source
        matching_mask = (source_doy == doy) & (source_hour == hour)
        
        if matching_mask.any():
            profile_values.append(hourly_profile[matching_mask].iloc[0])
        else:
            # Leap year handling: use Dec 30 data for Dec 31 if missing
            fallback_mask = (source_doy == min(doy, 365)) & (source_hour == hour)
            if fallback_mask.any():
                profile_values.append(hourly_profile[fallback_mask].iloc[0])
            else:
                # Ultimate fallback: use mean
                profile_values.append(hourly_profile.mean())
    
    profile_mapped = pd.Series(profile_values, index=time_index)
    
    # Normalize profile so it sums to 1.0 when multiplied by timestep hours
    # This allows scaling by annual energy (GWh -> MW time series)
    timestep_hours = 1.0  # Hourly data
    profile_normalized = profile_mapped / (profile_mapped.sum() * timestep_hours)
    
    # Create timeseries for each demand node
    timeseries_data = {}
    for node_id in demand_data.index:
        annual_demand_gwh = demand_data.loc[node_id].iloc[0]
        # Convert to MWh and scale profile
        annual_demand_mwh = annual_demand_gwh * 1000.0
        # Result is in MW
        node_timeseries = profile_normalized * annual_demand_mwh
        timeseries_data[node_id] = node_timeseries
    
    p_set_data = pd.DataFrame(timeseries_data, index=time_index)
    
    # Compute totals for logging
    total_gwh = p_set_data.sum().sum() * timestep_hours / 1000.0
    total_twh = total_gwh / 1000.0
    
    logger.info(f"{model_name} profile scaled: {len(time_index)} timesteps")
    logger.info(f"  Shape: {p_set_data.shape}")
    logger.info(f"  Total annual energy: {total_twh:.3f} TWh")
    
    return p_set_data


def generate_eload_profiles_from_excel(demand_data: pd.DataFrame, modelled_year: int, logger, profile_year: int = None) -> pd.DataFrame:
    """
    Generate load profiles using eLOAD model data from egy_7649_mmc1.xlsx.
    
    eLOAD is a bottom-up electricity demand model that provides hourly
    demand profiles based on appliance-level modelling.
    
    Args:
        demand_data: FES demand by GSP/node (GWh annual)
        modelled_year: Target year for output timestamps
        logger: Logger instance
        profile_year: Which profile to use (2010 or 2050). If None, auto-select based on modelled_year.
    
    The selected profile is scaled to match FES annual demand projections.
    """
    # Auto-select profile year if not specified
    if profile_year is None:
        profile_year = 2050 if modelled_year >= 2040 else 2010
        logger.info(f"Auto-selected {profile_year} profile for modelled year {modelled_year}")
    
    logger.info("=" * 60)
    logger.info(f"Using eLOAD model {profile_year} profile for temporal distribution")
    logger.info(f"  Source: egy_7649_mmc1.xlsx, ELOAD sheet, UK {profile_year} profile")
    logger.info("=" * 60)
    
    # Load the eLOAD profile
    hourly_profile = _load_egy_profile("ELOAD", "eLOAD", logger, profile_year)
    
    # Scale to FES demand
    return _scale_hourly_profile_to_fes_demand(
        hourly_profile, demand_data, modelled_year, "eLOAD", logger
    )


def generate_desstinee_profiles_from_excel(demand_data: pd.DataFrame, modelled_year: int, logger, profile_year: int = None) -> pd.DataFrame:
    """
    Generate load profiles using DESSTINEE model data from egy_7649_mmc1.xlsx.
    
    DESSTINEE (Demand for Energy Services, Supply and Transmission in Europe)
    is an open-source energy demand model that generates synthetic hourly
    demand profiles based on weather, calendar, and socio-economic factors.
    
    Args:
        demand_data: FES demand by GSP/node (GWh annual)
        modelled_year: Target year for output timestamps
        logger: Logger instance
        profile_year: Which profile to use (2010 or 2050). If None, auto-select based on modelled_year.
    
    The selected profile is scaled to match FES annual demand projections.
    """
    # Auto-select profile year if not specified
    if profile_year is None:
        profile_year = 2050 if modelled_year >= 2040 else 2010
        logger.info(f"Auto-selected {profile_year} profile for modelled year {modelled_year}")
    
    logger.info("=" * 60)
    logger.info(f"Using DESSTINEE model {profile_year} profile for temporal distribution")
    logger.info(f"  Source: egy_7649_mmc1.xlsx, DESSTINEE sheet, UK {profile_year} profile")
    logger.info("=" * 60)
    
    # Load the DESSTINEE profile
    hourly_profile = _load_egy_profile("DESSTINEE", "DESSTINEE", logger, profile_year)
    
    # Scale to FES demand
    return _scale_hourly_profile_to_fes_demand(
        hourly_profile, demand_data, modelled_year, "DESSTINEE", logger
    )


def generate_espeni_profiles(demand_data: pd.DataFrame, modelled_year: int, logger, demand_year: int = None) -> pd.DataFrame:
    """Generate profiles using ESPENI historical data."""
    logger.info("Loading ESPENI historical demand data")
    
    # Load ESPENI data
    try:
        espeni_file = snakemake.input.espeni
        
        # Try to detect the ESPENI file format
        # Read a small sample to check column names
        sample_df = pd.read_csv(espeni_file, nrows=1)
        columns = list(sample_df.columns)
        
        # Determine the format based on available columns
        if 'ELEXM_utc' in columns and 'POWER_ESPENI_MW' in columns:
            # Old format
            espeni_data = pd.read_csv(espeni_file, parse_dates=['ELEXM_utc'], index_col='ELEXM_utc')
            demand_column = 'POWER_ESPENI_MW'
            logger.info(f"Loaded ESPENI data (old format) with shape: {espeni_data.shape}")
        elif 'ELEC_elex_startTime[utc](datetime)' in columns and 'ELEC_POWER_TOTAL_ESPENI[MW](float32)' in columns:
            # New format
            date_col = 'ELEC_elex_startTime[utc](datetime)'
            espeni_data = pd.read_csv(espeni_file, parse_dates=[date_col], index_col=date_col)
            demand_column = 'ELEC_POWER_TOTAL_ESPENI[MW](float32)'
            logger.info(f"Loaded ESPENI data (new format) with shape: {espeni_data.shape}")
        else:
            # Try to find similar column names
            date_candidates = [col for col in columns if 'time' in col.lower() and 'utc' in col.lower()]
            espeni_candidates = [col for col in columns if 'espeni' in col.lower() and 'mw' in col.lower()]
            
            if date_candidates and espeni_candidates:
                date_col = date_candidates[0]
                demand_column = espeni_candidates[0]
                espeni_data = pd.read_csv(espeni_file, parse_dates=[date_col], index_col=date_col)
                logger.info(f"Loaded ESPENI data (auto-detected format) with shape: {espeni_data.shape}")
                logger.info(f"Using date column: {date_col}")
                logger.info(f"Using demand column: {demand_column}")
            else:
                raise ValueError(f"Could not identify ESPENI date/demand columns. Available columns: {columns}")
        
    except Exception as e:
        logger.error(f"Failed to load ESPENI data: {e}")
        logger.warning("Falling back to eload profiles")
        return generate_eload_future_profiles(demand_data, modelled_year, logger)
    
    # Extract demand column
    if demand_column not in espeni_data.columns:
        logger.error(f"{demand_column} column not found in ESPENI data")
        logger.error(f"Available columns: {list(espeni_data.columns)[:10]}...")  # Show first 10 columns
        logger.warning("Falling back to eload profiles")
        return generate_eload_future_profiles(demand_data, modelled_year, logger)
    
    historical_demand = espeni_data[demand_column].dropna()
    logger.info(f"Historical ESPENI demand data: {len(historical_demand)} data points")
    
    # Create time index for the modelled year (half-hourly for full year)
    start_date = f"{modelled_year}-01-01 00:00:00"
    end_date = f"{modelled_year}-12-31 23:30:00"
    time_index = pd.date_range(start=start_date, end=end_date, freq=FREQ_HALF_HOUR)
    
    logger.info(f"Created time index: {len(time_index)} timesteps from {time_index[0]} to {time_index[-1]}")
    
    # Create normalized profile from historical data
    # Use configured demand year if provided, otherwise use the most recent complete year of data
    try:
        if demand_year is not None:
            # Use the specified demand year from configuration
            logger.info(f"Using configured demand year: {demand_year}")
            year_start = f"{demand_year}-01-01"
            year_end = f"{demand_year}-12-31 23:30:00"
            
            year_data = historical_demand.loc[year_start:year_end]
            logger.info(f"Using configured year {demand_year} with {len(year_data)} data points")
            
            if len(year_data) < 17000:  # Less than ~97% of year (allowing for some missing data)
                logger.warning(f"Insufficient data for configured year {demand_year}, creating composite profile")
                year_data = create_composite_espeni_profile(historical_demand, logger)
            else:
                # We have sufficient data for the configured year - use it directly
                logger.info(f"Using complete year {demand_year} data directly")
        else:
            # Fallback to most recent year behavior
            historical_years = historical_demand.index.year.unique()
            most_recent_year = max(historical_years)
            
            # Extract one full year of data
            year_start = f"{most_recent_year}-01-01"
            year_end = f"{most_recent_year}-12-31 23:30:00"
            
            year_data = historical_demand.loc[year_start:year_end]
            logger.info(f"Using historical year {most_recent_year} with {len(year_data)} data points")
            
            if len(year_data) < 17000:  # Less than ~97% of year (allowing for some missing data)
                logger.warning(f"Insufficient data for year {most_recent_year}, creating composite profile")
                year_data = create_composite_espeni_profile(historical_demand, logger)
            else:
                # We have sufficient data for a full year - use it directly
                logger.info(f"Using complete year {most_recent_year} data directly")
        
    except Exception as e:
        logger.warning(f"Could not extract full year from ESPENI data: {e}")
        year_data = create_composite_espeni_profile(historical_demand, logger)
    
    # Resample to match target year length if needed
    if len(year_data) != len(time_index):
        logger.info(f"Resampling ESPENI data from {len(year_data)} to {len(time_index)} data points")
        
        # Create a representative year profile by resampling/interpolating
        if len(year_data) > 0:
            # Create a proper time series with the original data
            year_data_series = pd.Series(year_data.values if hasattr(year_data, 'values') else year_data, 
                                       index=year_data.index if hasattr(year_data, 'index') else None)
            
            # If we have a proper time series, resample it to the target frequency
            if hasattr(year_data_series, 'index') and isinstance(year_data_series.index, pd.DatetimeIndex):
                # Resample to target year frequency and fill any gaps
                target_freq_series = year_data_series.resample(FREQ_HALF_HOUR).mean()
                target_freq_series = target_freq_series.interpolate(method='time')
                
                # Map to target year dates 
                target_year_index = time_index
                
                # Create mapping from day-of-year and time-of-day
                source_doy = target_freq_series.index.dayofyear
                source_hour = target_freq_series.index.hour  
                source_minute = target_freq_series.index.minute
                
                target_doy = target_year_index.dayofyear
                target_hour = target_year_index.hour
                target_minute = target_year_index.minute
                
                # Create a mapping based on day of year and time of day
                profile_values = []
                for i, (doy, hour, minute) in enumerate(zip(target_doy, target_hour, target_minute)):
                    # Find matching day and time in source data
                    matching_mask = (source_doy == doy) & (source_hour == hour) & (source_minute == minute)
                    
                    if matching_mask.any():
                        # Use exact match if available
                        profile_values.append(target_freq_series[matching_mask].iloc[0])
                    else:
                        # Find nearest time match on same day of year
                        day_mask = source_doy == doy
                        if day_mask.any():
                            day_data = target_freq_series[day_mask]
                            time_diff = abs((day_data.index.hour * 60 + day_data.index.minute) - (hour * 60 + minute))
                            nearest_idx = time_diff.argmin()
                            profile_values.append(day_data.iloc[nearest_idx])
                        else:
                            # Fallback to same time on nearest available day
                            time_mask = (source_hour == hour) & (source_minute == minute)
                            if time_mask.any():
                                profile_values.append(target_freq_series[time_mask].mean())
                            else:
                                profile_values.append(target_freq_series.mean())
                
                profile_normalized = pd.Series(profile_values, index=time_index)
                # Keep resampled MW series so historical path has correct length
                year_data = profile_normalized.copy()
                
            else:
                # Fallback to simple interpolation
                historical_profile = year_data / year_data.mean() if hasattr(year_data, 'mean') else year_data / np.mean(year_data)
                
                # Create new index spanning the year with appropriate frequency
                historical_index = pd.date_range(start=f"{modelled_year}-01-01 00:00:00",
                                               periods=len(historical_profile), 
                                               freq=f"{8760*60/len(historical_profile):.1f}min")
                
                historical_series = pd.Series(historical_profile.values if hasattr(historical_profile, 'values') else historical_profile, 
                                            index=historical_index)
                
                # Resample to target frequency
                resampled_profile = historical_series.resample(FREQ_HALF_HOUR).interpolate()
                
                # Trim or extend to exact length
                if len(resampled_profile) >= len(time_index):
                    profile_normalized = resampled_profile.iloc[:len(time_index)]
                else:
                    # Extend by repeating the pattern
                    profile_normalized = resampled_profile.reindex(time_index, method='ffill')
                # Keep resampled MW series so historical path has correct length
                year_data = profile_normalized.copy()
        else:
            logger.error("No valid ESPENI data found, falling back to eload profiles")
            return generate_eload_future_profiles(demand_data, modelled_year, logger)
    else:
        # Data length matches, just normalize
        profile_normalized = year_data / year_data.mean()
    
    # IMPORTANT: Save original year_data BEFORE any normalization for historical scenarios
    original_year_data = year_data.copy() if hasattr(year_data, 'copy') else year_data
    
    # Set the index to the target time index
    profile_normalized.index = time_index
    
    # Log some statistics about the profile to verify daily variation
    daily_profile_stats = profile_normalized.groupby(profile_normalized.index.hour).agg(['mean', 'std'])
    min_hour = daily_profile_stats.idxmin()['mean']
    max_hour = daily_profile_stats.idxmax()['mean'] 
    daily_variation = profile_normalized.max() / profile_normalized.min()
    
    logger.info(f"ESPENI profile daily variation: {daily_variation:.2f} (min hour: {min_hour}, max hour: {max_hour})")
    logger.info(f"Profile stats - Mean: {profile_normalized.mean():.3f}, Std: {profile_normalized.std():.3f}")
    
    # Check if this is a historical scenario (demand_data contains zeros)
    total_fes_demand = demand_data.sum().iloc[0] if len(demand_data.columns) > 0 else 0
    is_historical_scenario = total_fes_demand == 0.0
    
    if is_historical_scenario:
        # Historical scenario: Use actual ESPENI data directly without FES scaling
        logger.info("Historical scenario detected: Using actual ESPENI data without FES scaling")
        
        # For historical scenarios, original_year_data contains the actual MW values (before normalization)
        # We need to ensure it's aligned with the time_index
        if not isinstance(original_year_data, pd.Series):
            historical_demand_mw = pd.Series(original_year_data.values if hasattr(original_year_data, 'values') else original_year_data, index=time_index[:len(original_year_data)])
        else:
            # Ensure original_year_data has the right index
            historical_demand_mw = pd.Series(original_year_data.values, index=time_index[:len(original_year_data)])
        
        # If length doesn't match (shouldn't happen but defensive coding)
        if len(historical_demand_mw) < len(time_index):
            # Extend to full length by repeating pattern
            historical_demand_mw = historical_demand_mw.reindex(time_index, method='ffill')
        elif len(historical_demand_mw) > len(time_index):
            # Trim to exact length
            historical_demand_mw = historical_demand_mw.iloc[:len(time_index)]
        
        total_annual_mwh = historical_demand_mw.sum() * TIMESTEP_HOURS
        logger.info(f"Total historical demand: {total_annual_mwh / 1000:.1f} GWh")
        
        # Distribute demand equally across all buses
        # (spatial disaggregation will be handled by the network building code)
        n_buses = len(demand_data.index)
        per_bus_fraction = 1.0 / n_buses if n_buses > 0 else 1.0
        
        # Create DataFrame with actual MW values distributed across buses
        timeseries_data = {}
        for node_id in demand_data.index:
            # Each bus gets an equal fraction of the total demand
            node_timeseries = historical_demand_mw * per_bus_fraction
            timeseries_data[node_id] = node_timeseries
        
        p_set_data = pd.DataFrame(timeseries_data, index=time_index)
        
    else:
        # Future scenario: Scale normalized profile by FES demand projections
        logger.info("Future scenario detected: Scaling ESPENI profile to FES demand projections")
        
        # Ensure the profile sums correctly for energy scaling
        # Normalize so that sum(profile_normalized) * TIMESTEP_HOURS == 1
        profile_normalized = profile_normalized / (profile_normalized.sum() * TIMESTEP_HOURS)
        
        # Create timeseries for all demand nodes
        timeseries_data = {}
        for node_id in demand_data.index:
            annual_demand_gwh = demand_data.loc[node_id].iloc[0]
            # Convert annual demand to MWh
            annual_demand_mwh = annual_demand_gwh * 1000.0
            # Scale profile to match annual demand: resulting series is in MW
            node_timeseries = profile_normalized * annual_demand_mwh
            timeseries_data[node_id] = node_timeseries

        p_set_data = pd.DataFrame(timeseries_data, index=time_index)

    # Compute totals for logging
    total_gwh = p_set_data.sum().sum() * TIMESTEP_HOURS / 1000.0
    total_twh = total_gwh / 1000.0

    logger.info(f"ESPENI data: {len(time_index)} timesteps, {total_gwh:.1f} GWh total")
    logger.info(f"Generated load timeseries shape: {p_set_data.shape}")
    logger.info(f"Total annual energy in timeseries: {total_twh:.3f} TWh")

    return p_set_data

def create_composite_espeni_profile(historical_demand: pd.Series, logger) -> pd.Series:
    """Create a composite annual profile from available historical ESPENI data."""
    logger.info("Creating composite ESPENI profile from available historical data")
    
    try:
        # Ensure the index is properly parsed as datetime
        if not isinstance(historical_demand.index, pd.DatetimeIndex):
            historical_demand.index = pd.to_datetime(historical_demand.index)
        
        # Add day of year and hour columns
        df = pd.DataFrame({'demand': historical_demand})
        df['day_of_year'] = df.index.dayofyear
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        
        # Create half-hourly periods (0, 30 minutes)
        df['half_hour'] = df['hour'] * 2 + (df['minute'] >= 30).astype(int)
        
        # Calculate average demand for each day-of-year and half-hour combination
        composite_profile = df.groupby(['day_of_year', 'half_hour'])['demand'].mean()
        
        # Create full year index
        year_2020 = 2020  # Use 2020 as template (leap year for completeness)
        time_index = pd.date_range(start=f"{year_2020}-01-01 00:00:00", 
                                 end=f"{year_2020}-12-31 23:30:00", 
                                 freq=FREQ_HALF_HOUR)
        
        # Map composite profile to full year
        full_year_data = []
        for timestamp in time_index:
            day_of_year = timestamp.dayofyear
            half_hour = timestamp.hour * 2 + (timestamp.minute >= 30)
            
            if (day_of_year, half_hour) in composite_profile.index:
                full_year_data.append(composite_profile.loc[(day_of_year, half_hour)])
            else:
                # Fallback: use average of nearby periods
                nearby_values = []
                for offset in [-1, 0, 1]:
                    try:
                        nearby_values.append(composite_profile.loc[(day_of_year, half_hour + offset)])
                    except KeyError:
                        pass
                if nearby_values:
                    full_year_data.append(np.mean(nearby_values))
                else:
                    # Ultimate fallback: use overall mean
                    full_year_data.append(historical_demand.mean())
        
        result = pd.Series(full_year_data, index=time_index)
        logger.info(f"Created composite profile with {len(result)} data points")
        return result
        
    except Exception as e:
        logger.error(f"Failed to create composite ESPENI profile: {e}")
        # Return a simple seasonal profile as ultimate fallback
        logger.warning("Using simple seasonal fallback pattern")
        time_index = pd.date_range(start="2020-01-01 00:00:00", 
                                 end="2020-12-31 23:30:00", 
                                 freq=FREQ_HALF_HOUR)
        
        hour_of_year = np.arange(len(time_index)) * 0.5
        daily_cycle = 0.8 + 0.4 * np.sin(2 * np.pi * (hour_of_year % 24) / 24 - np.pi/2)
        seasonal_cycle = 1.0 + 0.3 * np.sin(2 * np.pi * (hour_of_year / (24*365)) - np.pi/2)
        profile_base = daily_cycle * seasonal_cycle
        
        mean_demand = historical_demand.mean() if not historical_demand.empty else 40000  # MW
        scaled_profile = profile_base * mean_demand
        
        return pd.Series(scaled_profile, index=time_index)


# ============================================================================
# DEMAND DISAGGREGATION SYSTEM (Skeleton Implementation)
# ============================================================================

def load_heat_pump_data(year: int, logger) -> pd.DataFrame:
    """Load heat pump demand data (skeleton)."""
    logger.info(f"Loading heat pump data for {year} (skeleton)")
    return pd.DataFrame()  # Return empty for now

def load_electric_vehicle_data(year: int, logger) -> pd.DataFrame:
    """Load electric vehicle demand data (skeleton)."""
    logger.info(f"Loading electric vehicle data for {year} (skeleton)")
    return pd.DataFrame()  # Return empty for now

def load_hydrogen_demand_data(year: int, logger) -> pd.DataFrame:
    """Load hydrogen demand data (skeleton)."""
    logger.info(f"Loading hydrogen demand data for {year} (skeleton)")
    return pd.DataFrame()  # Return empty for now

def load_storage_heating_data(year: int, logger) -> pd.DataFrame:
    """Load storage heating data (skeleton)."""
    logger.info(f"Loading storage heating data for {year} (skeleton)")
    return pd.DataFrame()  # Return empty for now

def allocate_by_population(component_data: pd.DataFrame, target_nodes: list, logger) -> pd.DataFrame:
    """Allocate component demand by population (skeleton)."""
    logger.info("Allocating by population (skeleton)")
    return pd.DataFrame()  # Return empty for now

def allocate_by_gdp(component_data: pd.DataFrame, target_nodes: list, logger) -> pd.DataFrame:
    """Allocate component demand by GDP (skeleton)."""
    logger.info("Allocating by GDP (skeleton)")
    return pd.DataFrame()  # Return empty for now

def allocate_by_existing_demand(component_data: pd.DataFrame, target_nodes: list, logger) -> pd.DataFrame:
    """Allocate component demand by existing demand (skeleton)."""
    logger.info("Allocating by existing demand (skeleton)")
    return pd.DataFrame()  # Return empty for now

def allocate_uniformly(component_data: pd.DataFrame, target_nodes: list, logger) -> pd.DataFrame:
    """Allocate component demand uniformly (skeleton)."""
    logger.info("Allocating uniformly (skeleton)")
    return pd.DataFrame()  # Return empty for now

# Component loaders registry
COMPONENT_LOADERS = {
    'heat_pumps': lambda year, logger: load_heat_pump_data(year, logger),
    'electric_vehicles': lambda year, logger: load_electric_vehicle_data(year, logger), 
    'hydrogen': lambda year, logger: load_hydrogen_demand_data(year, logger),
    'storage_heating': lambda year, logger: load_storage_heating_data(year, logger)
}

# Allocation methods registry  
ALLOCATION_METHODS = {
    'population': allocate_by_population,
    'gdp': allocate_by_gdp,
    'existing_demand': allocate_by_existing_demand,
    'uniform': allocate_uniformly
}

def disaggregate_demand(
    demand_data: pd.DataFrame,
    timeseries_data: pd.DataFrame, 
    disaggregation_config: dict,
    modelled_year: int,
    logger
) -> tuple:
    """
    Disaggregate total demand into components (skeleton implementation).
    
    Returns
    -------
    tuple
        (adjusted_demand, adjusted_timeseries, components)
    """
    if not disaggregation_config.get('enabled', False):
        logger.info("Demand disaggregation disabled or not configured")
        return demand_data, timeseries_data, {}
    
    logger.info("Demand disaggregation enabled - running skeleton implementation")
    components = {}
    
    # For now, just return original data unchanged
    # In a full implementation, this would:
    # 1. Load component-specific data
    # 2. Allocate components to network nodes
    # 3. Subtract component demand from total demand
    # 4. Return adjusted demand + component breakdowns
    
    logger.info("Skeleton disaggregation completed (no actual disaggregation performed)")
    return demand_data, timeseries_data, components

def add_loads_to_network_by_model(
    network: pypsa.Network,
    demand_data: pd.DataFrame,
    timeseries_data: pd.DataFrame, 
    network_model: str,
    logger
):
    """Add loads to network using model-specific logic."""
    
    available_buses = set(network.buses.index)
    demand_nodes = set(demand_data.index)
    
    logger.info(f"Network model: {network_model}")
    logger.info(f"Available buses: {len(available_buses)}")
    logger.info(f"Demand nodes: {len(demand_nodes)}")
    
    # Find matches between demand nodes and network buses
    matched_nodes = available_buses & demand_nodes
    missing_nodes = demand_nodes - available_buses
    
    logger.info(f"Direct matches: {len(matched_nodes)}")
    logger.info(f"Missing nodes: {len(missing_nodes)}")
    
    # Log details about missing nodes
    if missing_nodes:
        total_missing_demand = demand_data.loc[list(missing_nodes)].sum().iloc[0]
        logger.warning(f"Missing nodes represent {total_missing_demand:.1f} GWh of demand:")
        for node in list(missing_nodes)[:10]:  # Show first 10 missing nodes
            demand_gwh = demand_data.loc[node].iloc[0]
            logger.warning(f"  - {node}: {demand_gwh:.3f} GWh")
        if len(missing_nodes) > 10:
            logger.warning(f"  ... and {len(missing_nodes) - 10} more nodes")
    
    if network_model in ["Reduced", "Zonal"] and len(matched_nodes) == 0:
        logger.warning("No direct bus matches found - this is expected for aggregated networks")
        
        # For aggregated networks, all demand nodes should be network buses
        # since we aggregated specifically to match the network topology
        for bus_id in available_buses:
            if bus_id in demand_data.index:
                try:
                    network.add("Load", f"load_{bus_id}", bus=str(bus_id))
                    annual_energy_gwh = demand_data.loc[bus_id].iloc[0]
                    logger.debug(f"Added load for bus {bus_id}: {annual_energy_gwh:.3f} GWh")
                except Exception as e:
                    logger.warning(f"Could not add load for bus {bus_id}: {e}")
                    
    else:
        # ETYS or networks with direct matches - use existing logic
        matched_nodes_count = 0
        
        # Add loads for matched nodes
        for node_id in demand_data.index:
            if str(node_id) in available_buses:
                annual_energy_gwh = demand_data.loc[node_id].iloc[0]
                try:
                    network.add("Load", f"load_{node_id}", bus=str(node_id))
                    matched_nodes_count += 1
                    logger.debug(f"Added load for bus {node_id}: {annual_energy_gwh:.3f} GWh")
                except Exception as e:
                    logger.warning(f"Could not add load for bus {node_id}: {e}")
        
        # Handle missing nodes by redistributing their load
        if missing_nodes:
            total_missing_demand = demand_data.loc[list(missing_nodes)].sum().iloc[0]
            logger.info(f"Redistributing {total_missing_demand:.1f} GWh from {len(missing_nodes)} missing nodes")
            
            # Distribute missing load proportionally among existing loads
            if len(network.loads) > 0:
                existing_demand = demand_data.loc[demand_data.index.intersection(matched_nodes)]
                total_existing_demand = existing_demand.sum().iloc[0]
                
                if total_existing_demand > 0:
                    # Calculate redistribution factors
                    for load_name in network.loads.index:
                        node_id = load_name.replace('load_', '')
                        if node_id in existing_demand.index:
                            original_demand = existing_demand.loc[node_id].iloc[0]
                            proportion = original_demand / total_existing_demand
                            additional_demand = total_missing_demand * proportion
                            
                            logger.debug(f"Adding {additional_demand:.3f} GWh to load {load_name} "
                                       f"(original: {original_demand:.3f} GWh)")
                else:
                    logger.warning("Could not redistribute missing demand - no existing demand found")
    
    # Set timeseries data
    if len(network.loads) > 0:
        network.loads_t.p_set = network.loads_t.p_set.reindex(columns=network.loads.index, fill_value=0.0)
        
        # Calculate adjustment factors for redistributed load
        adjustment_factors = {}
        if missing_nodes:
            total_missing_demand = demand_data.loc[list(missing_nodes)].sum().iloc[0]
            existing_demand = demand_data.loc[demand_data.index.intersection(matched_nodes)]
            total_existing_demand = existing_demand.sum().iloc[0]
            
            if total_existing_demand > 0:
                for load_name in network.loads.index:
                    node_id = load_name.replace('load_', '')
                    if node_id in existing_demand.index:
                        original_demand = existing_demand.loc[node_id].iloc[0]
                        if original_demand > 0:  # Avoid division by zero
                            proportion = original_demand / total_existing_demand
                            additional_factor = (total_missing_demand * proportion) / original_demand
                            adjustment_factors[load_name] = 1.0 + additional_factor
                        else:
                            adjustment_factors[load_name] = 1.0
                    else:
                        adjustment_factors[load_name] = 1.0
            else:
                adjustment_factors = {load_name: 1.0 for load_name in network.loads.index}
        else:
            adjustment_factors = {load_name: 1.0 for load_name in network.loads.index}
        
        for load_name in network.loads.index:
            node_id = load_name.replace('load_', '')
            if node_id in timeseries_data.columns:
                base_timeseries = timeseries_data[node_id]
                adjusted_timeseries = base_timeseries * adjustment_factors.get(load_name, 1.0)
                network.loads_t.p_set[load_name] = adjusted_timeseries
            else:
                logger.warning(f"No timeseries data for load {load_name} - using zero load")
                # Create zero timeseries instead of leaving NaN
                zero_timeseries = pd.Series(0.0, index=timeseries_data.index)
                network.loads_t.p_set[load_name] = zero_timeseries
    
    # Log summary
    logger.info("Load assignment summary:")
    logger.info(f"  - Network model: {network_model}")
    logger.info(f"  - Loads added: {len(network.loads)}")
    logger.info(f"  - Total annual demand: {demand_data.sum().iloc[0]:.1f} GWh")

def add_component_loads_to_network(network: pypsa.Network, components: dict, logger):
    """Add disaggregated component loads to network."""
    
    for comp_name, comp_data in components.items():
        logger.info(f"Adding component loads for: {comp_name}")
        
        comp_demand = comp_data['demand']
        comp_timeseries = comp_data['timeseries']
        
        # Add loads with component-specific naming
        for node_id in comp_demand.index:
            if str(node_id) in network.buses.index:
                load_name = f"{node_id}_{comp_name}"
                try:
                    network.add("Load", load_name, bus=str(node_id))
                    logger.debug(f"Added component load: {load_name}")
                except Exception as e:
                    logger.warning(f"Could not add component load {load_name}: {e}")
        
        # Set component timeseries
        component_loads = [load for load in network.loads.index if comp_name in load]
        if component_loads:
            for load in component_loads:
                node_id = load.replace(f"_{comp_name}", "")
                if node_id in comp_timeseries.columns:
                    network.loads_t.p_set[load] = comp_timeseries[node_id]

def validate_network_loads(network: pypsa.Network, logger):
    """Validate that loads were added correctly."""
    
    n_loads = len(network.loads)
    if n_loads == 0:
        logger.error("No loads were added to network!")
        raise ValueError("Network has no loads")
    # total_load_mw is the sum of MW values across all timesteps and loads
    total_load_mw = network.loads_t.p_set.sum().sum()

    # Compute annual energy from MW timeseries: sum(MW) * hours_per_timestep = MWh
    total_annual_mwh = total_load_mw * TIMESTEP_HOURS
    total_annual_gwh = total_annual_mwh / 1000.0
    total_annual_twh = total_annual_gwh / 1000.0

    # Peak system load (instantaneous MW) for additional diagnostics
    try:
        system_instantaneous = network.loads_t.p_set.sum(axis=1)
        peak_mw = system_instantaneous.max()
    except Exception:
        peak_mw = None

    logger.info(f"Network validation: {n_loads} loads, sum(MW values)={total_load_mw:.3f} (used to compute energy)")
    logger.info(f"  - Peak instantaneous load: {peak_mw:.1f} MW" if peak_mw is not None else "  - Peak instantaneous load: N/A")
    logger.info(f"  - Annual energy (computed): {total_annual_mwh:.1f} MWh / {total_annual_gwh:.3f} GWh / {total_annual_twh:.6f} TWh")
    
    # Check for NaN values
    nan_count = network.loads_t.p_set.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in load timeseries")
    
    logger.info("Network load validation completed")

def add_FES_demand_data_to_network(demand_data, p_set_data, logger, is_historical: bool = False):
    """
    Add FES demand data to the PyPSA network with optional disaggregation.
    Now supports multiple network models (ETYS, Reduced, Zonal).
    """
    logger.info("Adding FES demand data to PyPSA network")
    
    # Get network model
    network_model = get_network_model()
    logger.info(f"Processing loads for network model: {network_model}")
    
    # Get disaggregation config from snakemake params
    disaggregation_configs = getattr(snakemake.params, 'disaggregation_configs', {})
    run_id = list(disaggregation_configs.keys())[0] if disaggregation_configs else None
    disaggregation_config = disaggregation_configs.get(run_id, {"enabled": False}) if run_id else {"enabled": False}
    
    logger.info(f"Disaggregation config: {disaggregation_config}")
    
    # Import network
    network = load_network(snakemake.input.base_network, custom_logger=logger)
    log_network_info(network, logger)
    
    # Get GSP names from demand_data index (for Reduced/Zonal networks)
    gsp_names = list(demand_data.index) if demand_data is not None else []
    
    # Get FES year from snakemake params
    fes_year = get_param_value(snakemake.params.fes_year) if hasattr(snakemake.params, 'fes_year') else 2024
    
    # Load spatial mapping data for network model
    spatial_mapping = load_spatial_mapping_data(network_model, gsp_names, fes_year, logger)
    
    # Aggregate demand to match network topology
    network_demand, network_timeseries = aggregate_demand_to_network_topology(
        demand_data, p_set_data, network_model, network, spatial_mapping, logger, is_historical=is_historical
    )
    
    # Set snapshots (use the timeseries index from aggregated timeseries)
    network.set_snapshots(network_timeseries.index)
    logger.info(f"Set network snapshots: {len(network_timeseries.index)} timesteps")

    # Apply consistent snapshot weighting (half-hourly -> 0.5 hours per step)
    try:
        if hasattr(network, "snapshot_weightings"):
            network.snapshot_weightings.loc[:, ["objective", "stores", "generators"]] = TIMESTEP_HOURS
            logger.info(f"Applied snapshot weighting of {TIMESTEP_HOURS} hours to objective/stores/generators")
        else:
            logger.warning("Network missing snapshot_weightings attribute; skipping weighting adjustment")
    except Exception as e:
        logger.warning(f"Failed to set snapshot_weightings: {e}")

    # Sanity check: ensure network.snapshots matches the timeseries index
    try:
        if not hasattr(network, 'snapshots') or len(network.snapshots) != len(network_timeseries.index):
            logger.warning("Network snapshots do not match timeseries index after set_snapshots()")
        else:
            logger.info(f"Network.snapshots confirmed: {len(network.snapshots)} entries")
    except Exception:
        logger.warning("Could not verify network.snapshots length")
    
    # Apply disaggregation (currently skeleton)
    adjusted_demand, adjusted_timeseries, components = disaggregate_demand(
        network_demand, network_timeseries, disaggregation_config, 
        get_param_value(snakemake.params.modelled_year), logger
    )
    
    # Add loads to network using network-specific logic
    add_loads_to_network_by_model(
        network, adjusted_demand, adjusted_timeseries, network_model, logger
    )
    
    # Handle component loads if any
    if components:
        add_component_loads_to_network(network, components, logger)
    
    # Final validation and export
    # Ensure network.loads_t.p_set uses the same snapshot index as the generated timeseries
    try:
        if hasattr(network, 'loads_t') and hasattr(network.loads_t, 'p_set'):
            # If the index does not match, reindex to the desired snapshots
            if not network.loads_t.p_set.index.equals(network_timeseries.index):
                logger.info("Reindexing network.loads_t.p_set to match generated timeseries snapshots")
                network.loads_t.p_set = network.loads_t.p_set.reindex(index=network_timeseries.index, fill_value=0.0)
            logger.info(f"Loads timeseries shape: {getattr(network.loads_t, 'p_set').shape}")
    except Exception as e:
        logger.warning(f"Could not verify/reindex network.loads_t.p_set: {e}")

    validate_network_loads(network, logger)
    save_network(network, snakemake.output.network_with_base_demand, custom_logger=logger)
    logger.info("Network export completed")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Set up logging using centralized system, writing to Snakemake log if available
    import sys
    from datetime import datetime
    start_time = time.time()
    
    log_path = None
    if 'snakemake' in globals() and hasattr(snakemake, 'log') and snakemake.log:
        log_path = snakemake.log[0]
    logger = setup_logging(log_path or "load", log_level="INFO")
    # Silence repetitive PyPSA network warnings about missing optimization/model info
    # These warnings are expected when working with networks that haven't been
    # optimized yet; lower the pypsa.networks logger level to ERROR to suppress them.
    import logging as _logging
    _pypsa_logger = _logging.getLogger("pypsa.networks")
    _pypsa_logger.setLevel(_logging.ERROR)
    # Helper function to extract parameter values (handles both single values and lists)
    def get_param_value(param):
        return param[0] if isinstance(param, list) else param
    
    # Determine if this is a historical scenario
    # Use Snakemake param if available, otherwise compute from modelled_year
    modelled_year = get_param_value(snakemake.params.modelled_year)
    if hasattr(snakemake.params, 'is_historical') and snakemake.params.is_historical is not None:
        is_historical = get_param_value(snakemake.params.is_historical)
    else:
        # Fallback: historical if modelled_year <= 2024
        is_historical = modelled_year <= 2024
    
    # Get FES params (may be None for historical scenarios)
    fes_year = get_param_value(snakemake.params.fes_year) if hasattr(snakemake.params, 'fes_year') else None
    fes_scenario = get_param_value(snakemake.params.fes_scenario) if hasattr(snakemake.params, 'fes_scenario') else None
    
    logger.info("="*50)
    logger.info("STARTING LOAD DATA PROCESSING")
    logger.info(f"FES Year: {fes_year if fes_year else 'N/A (historical)'}")
    logger.info(f"FES Scenario: {fes_scenario if fes_scenario else 'N/A (historical)'}")
    logger.info(f"Modelled Year: {modelled_year}")
    logger.info(f"Demand Year: {get_param_value(snakemake.params.demand_year) if hasattr(snakemake.params, 'demand_year') else 'Not specified'}")
    logger.info(f"Demand Timeseries: {get_param_value(snakemake.params.demand_timeseries)}")
    logger.info(f"Scenario Type: {'HISTORICAL' if is_historical else 'FUTURE'}")
    
    # Validate FES parameters for future scenarios
    if not is_historical:
        if not fes_year:
            raise ValueError(f"Future scenario (modelled_year={modelled_year}) requires FES_year parameter")
        if not fes_scenario:
            raise ValueError(f"Future scenario (modelled_year={modelled_year}) requires FES_scenario parameter")
        logger.info("  → FES parameters validated for future scenario")

    # Apply timestep from scenario (default 30 minutes)
    timestep_minutes_param = get_param_value(getattr(snakemake.params, 'timestep_minutes', 30))
    try:
        timestep_minutes_val = int(timestep_minutes_param) if timestep_minutes_param is not None else 30
    except Exception:
        timestep_minutes_val = 30
    set_time_resolution(timestep_minutes_val, logger)
    logger.info("="*50)
    
    try:
        # Step 1: Load demand data (source depends on scenario type)
        if is_historical:
            # Historical scenarios: Use actual historical demand from ESPENI/EGY
            # No FES demand data needed - we have actual historical consumption
            logger.info("Step 1: Historical scenario detected - using actual demand data")
            logger.info(f"Loading actual {modelled_year} demand from {get_param_value(snakemake.params.demand_timeseries)}")
            
            # For historical scenarios, create a simple placeholder
            # The network will be loaded later when needed, so we'll get actual bus indices then
            original_fes_demand = pd.DataFrame(
                {"demand": [0.0]},  # Placeholder - won't be used, just structure
                index=["placeholder"]
            )
            logger.info("Created placeholder demand structure")
            logger.info("  → Actual demand will be loaded from historical timeseries in Step 2")
        else:
            # Future scenarios: Use FES projected demand
            logger.info("Step 1: Future scenario detected - loading FES demand projections")
            original_fes_demand = get_fes_demand_data(
                get_param_value(snakemake.params.fes_scenario),
                modelled_year, 
                logger
            )
        
        # Step 2: Generate load timeseries
        logger.info("Step 2: Generating load timeseries")
        
        # Get demand_year from snakemake params (if available)
        demand_year = get_param_value(getattr(snakemake.params, 'demand_year', None)) if hasattr(snakemake.params, 'demand_year') else None
        if demand_year is not None:
            logger.info(f"Using configured demand year: {demand_year}")
        
        if is_historical:
            # Historical: Use actual historical timeseries (ESPENI/EGY for the modelled year)
            logger.info(f"Historical scenario: Loading actual {modelled_year} demand timeseries")
            demand_year_to_use = demand_year if demand_year is not None else modelled_year
            logger.info(f"  → Using demand data from year: {demand_year_to_use}")
        
        # Get profile_year parameter for eload/desstinee (optional)
        profile_year = get_param_value(getattr(snakemake.params, 'profile_year', None)) if hasattr(snakemake.params, 'profile_year') else None
        
        p_set_data = generate_load_timeseries(
            original_fes_demand,
            get_param_value(snakemake.params.demand_timeseries),
            modelled_year,
            logger,
            demand_year if demand_year is not None else (modelled_year if is_historical else None),
            profile_year
        )
        
        # Step 3: Write load profile CSV (if output is defined in rule)
        logger.info("Step 3: Writing load profile CSV output (if configured)")
        if hasattr(snakemake.output, 'base_demand_profile') and snakemake.output.base_demand_profile:
            try:
                p_set_data.to_csv(snakemake.output.base_demand_profile)
                logger.info(f"Load profile timeseries written to {snakemake.output.base_demand_profile}")
            except Exception as e:
                logger.error(f"Failed to write load profile CSV: {e}")
        else:
            logger.info("  → CSV output not configured, skipping (demand stored in network pickle)")

        # Step 4: Add to network (now network-aware)
        logger.info("Step 4: Adding demand to network")
        add_FES_demand_data_to_network(original_fes_demand, p_set_data, logger, is_historical=is_historical)

        logger.info("LOAD PROCESSING COMPLETED SUCCESSFULLY")
        log_execution_summary(logger, "load", start_time, context={
            "modelled_year": modelled_year,
            "scenario_type": "HISTORICAL" if is_historical else "FUTURE",
            "fes_scenario": get_param_value(snakemake.params.fes_scenario),
            "demand_timeseries": get_param_value(snakemake.params.demand_timeseries)
        })

    except Exception as e:
        logger.error(f"FATAL ERROR in load processing: {e}")
        raise
