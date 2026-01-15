#!/usr/bin/env python3
"""
Spatial Utilities for PyPSA-GB
===============================

Shared spatial mapping functions for mapping sites (generators, storage, loads)
to network buses using proper geographic distance calculations.

This module provides the definitive bus mapping implementation used across:
- Renewable generators
- Thermal generators
- Storage units
- Interconnectors
- Demand loads

Key Features:
- Automatic coordinate system detection (WGS84 vs OSGB36)
- Haversine distance for WGS84 (proper great-circle distance)
- Euclidean distance for OSGB36 projected coordinates
- Caching for performance optimization
- Fuzzy string matching for region-based mapping
- Comprehensive logging and validation

Author: PyPSA-GB Team
"""

import logging
import hashlib
import numpy as np
import pandas as pd
import pypsa
from typing import Optional, Dict, Tuple, Literal
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)


# =============================================================================
# COORDINATE SYSTEM UTILITIES
# =============================================================================
# CRITICAL: The ETYS network uses OSGB36 (British National Grid) coordinates
# in METERS, not WGS84 (lat/lon in degrees). All buses MUST use the same
# coordinate system to prevent spatial mapping failures.
# =============================================================================

def detect_coordinate_system(x_values: np.ndarray, y_values: np.ndarray) -> Literal['OSGB36', 'WGS84', 'MIXED', 'UNKNOWN']:
    """
    Detect coordinate system from x/y values.
    
    OSGB36 (British National Grid):
      - x (easting): ~0 to 700,000 meters
      - y (northing): ~0 to 1,300,000 meters
      
    WGS84 (lat/lon):
      - x (longitude): ~-10 to 2 degrees for UK
      - y (latitude): ~49 to 61 degrees for UK
    
    Returns:
        'OSGB36': British National Grid (meters)
        'WGS84': Latitude/longitude (degrees)
        'MIXED': Contains both coordinate systems (ERROR state)
        'UNKNOWN': Cannot determine
    """
    if len(x_values) == 0:
        return 'UNKNOWN'
    
    x_min, x_max = np.nanmin(x_values), np.nanmax(x_values)
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
    
    # OSGB36 detection: values in meters (large numbers)
    is_osgb36 = (x_max > 1000 and y_max > 1000)
    
    # WGS84 detection: values in degrees (small numbers, UK range)
    is_wgs84 = (abs(x_min) < 180 and abs(x_max) < 180 and 
                abs(y_min) < 90 and abs(y_max) < 90 and
                x_max < 100 and y_max < 100)
    
    if is_osgb36 and not is_wgs84:
        return 'OSGB36'
    elif is_wgs84 and not is_osgb36:
        return 'WGS84'
    elif is_osgb36 and is_wgs84:
        # Mixed - some values look like OSGB36, some like WGS84
        return 'MIXED'
    else:
        return 'UNKNOWN'


def wgs84_to_osgb36(lon: float, lat: float) -> Tuple[float, float]:
    """
    Convert WGS84 coordinates (lat/lon) to OSGB36 (British National Grid).
    
    Args:
        lon: Longitude in degrees (WGS84)
        lat: Latitude in degrees (WGS84)
        
    Returns:
        Tuple of (easting, northing) in meters (OSGB36)
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        easting, northing = transformer.transform(lon, lat)
        return float(easting), float(northing)
    except Exception as e:
        logger.warning(f"Coordinate conversion failed for ({lon}, {lat}): {e}")
        return np.nan, np.nan


def osgb36_to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    """
    Convert OSGB36 coordinates (British National Grid) to WGS84 (lat/lon).
    
    Args:
        easting: Easting in meters (OSGB36)
        northing: Northing in meters (OSGB36)
        
    Returns:
        Tuple of (longitude, latitude) in degrees (WGS84)
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        return float(lon), float(lat)
    except Exception as e:
        logger.warning(f"Coordinate conversion failed for ({easting}, {northing}): {e}")
        return np.nan, np.nan


def validate_network_coordinates(network: pypsa.Network, fix: bool = False) -> Dict:
    """
    Validate that all network buses use consistent coordinate system.
    
    The ETYS network uses OSGB36 (meters). This function detects any buses
    with WGS84 coordinates and optionally converts them.
    
    Args:
        network: PyPSA network to validate
        fix: If True, convert any WGS84 coordinates to OSGB36
        
    Returns:
        Dict with validation results:
            - coordinate_system: Detected system ('OSGB36', 'WGS84', 'MIXED')
            - osgb36_count: Number of buses with OSGB36 coordinates
            - wgs84_count: Number of buses with WGS84 coordinates
            - wgs84_buses: List of bus names with WGS84 coordinates
            - fixed: Number of buses fixed (if fix=True)
    """
    result = {
        'coordinate_system': 'UNKNOWN',
        'osgb36_count': 0,
        'wgs84_count': 0,
        'wgs84_buses': [],
        'fixed': 0
    }
    
    if network.buses.empty:
        return result
    
    buses = network.buses.copy()
    
    # Detect overall coordinate system
    result['coordinate_system'] = detect_coordinate_system(
        buses['x'].values, buses['y'].values
    )
    
    # Classify each bus
    for bus_name, bus in buses.iterrows():
        x, y = bus['x'], bus['y']
        
        # Skip NaN coordinates
        if pd.isna(x) or pd.isna(y):
            continue
        
        # WGS84: small values typical of lat/lon
        is_wgs84 = (abs(x) < 180 and abs(y) < 90 and abs(x) < 20 and abs(y) < 70)
        
        if is_wgs84:
            result['wgs84_count'] += 1
            result['wgs84_buses'].append(bus_name)
            
            if fix:
                # Convert to OSGB36
                new_x, new_y = wgs84_to_osgb36(x, y)
                if not (pd.isna(new_x) or pd.isna(new_y)):
                    network.buses.at[bus_name, 'x'] = new_x
                    network.buses.at[bus_name, 'y'] = new_y
                    result['fixed'] += 1
                    logger.debug(f"Converted bus {bus_name}: WGS84({x:.4f}, {y:.4f}) → OSGB36({new_x:.0f}, {new_y:.0f})")
        else:
            result['osgb36_count'] += 1
    
    return result


def ensure_osgb36_coordinates(network: pypsa.Network) -> int:
    """
    Ensure all network buses use OSGB36 coordinates.
    
    This is the recommended function to call after any operation that
    might add buses (interconnectors, demand, etc.) to ensure coordinate
    system consistency.
    
    Args:
        network: PyPSA network to normalize
        
    Returns:
        Number of buses that were converted from WGS84 to OSGB36
    """
    validation = validate_network_coordinates(network, fix=True)
    
    if validation['fixed'] > 0:
        logger.info(f"Coordinate normalization: Converted {validation['fixed']} buses from WGS84 to OSGB36")
        logger.info(f"  Buses converted: {validation['wgs84_buses'][:10]}{'...' if len(validation['wgs84_buses']) > 10 else ''}")
    
    return validation['fixed']


def get_bus_coordinates_for_external(lon: float, lat: float, network: pypsa.Network) -> Tuple[float, float]:
    """
    Get coordinates for an external bus that are consistent with the network's coordinate system.
    
    For ETYS networks (OSGB36), converts WGS84 lat/lon to OSGB36 easting/northing.
    For other networks (WGS84), returns lat/lon directly.
    
    Args:
        lon: Longitude in degrees (WGS84)
        lat: Latitude in degrees (WGS84)
        network: PyPSA network to check coordinate system
        
    Returns:
        Tuple of (x, y) coordinates in the network's coordinate system
    """
    # Detect network coordinate system from existing buses
    if network.buses.empty:
        # No buses - assume OSGB36 for ETYS networks
        return wgs84_to_osgb36(lon, lat)
    
    coord_system = detect_coordinate_system(
        network.buses['x'].values, 
        network.buses['y'].values
    )
    
    if coord_system == 'OSGB36':
        # Convert WGS84 to OSGB36
        return wgs84_to_osgb36(lon, lat)
    elif coord_system == 'WGS84':
        # Keep as WGS84 (x=lon, y=lat)
        return float(lon), float(lat)
    else:
        # Default to OSGB36 for safety
        logger.warning(f"Unknown coordinate system, defaulting to OSGB36 conversion")
        return wgs84_to_osgb36(lon, lat)

# Cache for bus spatial indexes (avoid rebuilding for same network)
_BUS_SPATIAL_INDEX_CACHE: Dict[str, Tuple] = {}


def fuzzy_match_string(query: str, candidates: list, threshold: float = 0.8) -> Tuple[Optional[str], float]:
    """
    Find best fuzzy match for a query string in a list of candidates.
    
    Args:
        query: String to match
        candidates: List of candidate strings
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Tuple of (best_match, score) or (None, 0.0) if no match above threshold
    """
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        score = fuzz.ratio(query.lower(), candidate.lower()) / 100.0
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, 0.0


def map_sites_to_buses(
    network: pypsa.Network,
    sites_df: pd.DataFrame,
    method: str = 'nearest',
    bus_key: str = 'name',
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    max_distance_km: float = 200.0
) -> pd.DataFrame:
    """
    Map sites to network buses using robust geographic distance calculation.
    
    This function automatically detects the coordinate system of both the network
    buses and the sites, then uses the appropriate distance metric:
    - WGS84 (degrees): Haversine distance (great-circle distance on Earth)
    - OSGB36 (meters): Euclidean distance in projected coordinates
    
    Args:
        network: PyPSA network with buses
        sites_df: DataFrame with sites to map (must have lat/lon columns)
        method: Mapping method ('nearest' or 'region')
        bus_key: Column name for bus identifier in network (default 'name')
        lat_col: Column name for latitude in sites_df
        lon_col: Column name for longitude in sites_df
        max_distance_km: Maximum distance to consider a valid match (km)
        
    Returns:
        DataFrame with added 'bus' and 'distance_km' columns
        
    Raises:
        ValueError: If required columns are missing or network has no buses
        
    Examples:
        >>> # Map renewable sites to buses
        >>> sites_with_buses = map_sites_to_buses(network, renewable_sites)
        
        >>> # Map storage sites with custom distance limit
        >>> storage_with_buses = map_sites_to_buses(
        ...     network, storage_sites, max_distance_km=50.0
        ... )
    """
    logger.info(f"Mapping {len(sites_df)} sites to buses using method '{method}'")
    
    # Validate inputs
    required_cols = [lat_col, lon_col]
    missing_cols = [col for col in required_cols if col not in sites_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in sites_df: {missing_cols}")
    
    if network.buses.empty:
        raise ValueError("Network has no buses")
    
    if 'x' not in network.buses.columns or 'y' not in network.buses.columns:
        raise ValueError("Network buses must have 'x' and 'y' coordinate columns")
    
    # Create working copy
    sites_df = sites_df.copy()
    
    # Early return for empty DataFrame
    if len(sites_df) == 0:
        logger.info("Empty sites DataFrame, returning unchanged")
        if 'bus' not in sites_df.columns:
            sites_df['bus'] = pd.Series(dtype=str)
        if 'distance_km' not in sites_df.columns:
            sites_df['distance_km'] = pd.Series(dtype=float)
        return sites_df
    
    # ══════════════════════════════════════════════════════════════════════════════
    # PRESERVE PRE-ASSIGNED BUSES
    # Some sites (e.g., FES Direct offshore wind) already have bus assignments from 
    # _distribute_direct_capacity_using_historical(). These should NOT be overwritten.
    # Only map sites where bus is currently None/NaN.
    # ══════════════════════════════════════════════════════════════════════════════
    
    # Check for existing bus assignments
    has_existing_bus = 'bus' in sites_df.columns
    if has_existing_bus:
        # Preserve existing bus assignments - convert to nullable string type
        existing_buses = sites_df['bus'].copy()
        # Count pre-assigned buses
        pre_assigned_mask = existing_buses.notna() & (existing_buses != '') & (existing_buses != 'None')
        n_pre_assigned = pre_assigned_mask.sum()
        if n_pre_assigned > 0:
            logger.info(f"Preserving {n_pre_assigned} pre-assigned bus mappings")
            # Validate pre-assigned buses exist in network
            valid_buses = set(network.buses.index)
            pre_assigned_buses = existing_buses[pre_assigned_mask].unique()
            invalid_buses = [b for b in pre_assigned_buses if b not in valid_buses]
            if invalid_buses:
                logger.warning(f"Found {len(invalid_buses)} pre-assigned buses not in network: {invalid_buses[:5]}...")
    else:
        existing_buses = pd.Series([None] * len(sites_df), index=sites_df.index)
        pre_assigned_mask = pd.Series([False] * len(sites_df), index=sites_df.index)
        n_pre_assigned = 0
    
    # Initialize columns (don't overwrite existing valid bus assignments)
    if 'distance_km' not in sites_df.columns:
        sites_df['distance_km'] = np.inf
    
    # Reset only unmapped sites (skip if empty DataFrame)
    if len(sites_df) > 0:
        needs_mapping_mask = ~pre_assigned_mask
        sites_df.loc[needs_mapping_mask, 'bus'] = None
        sites_df.loc[needs_mapping_mask, 'distance_km'] = np.inf
        
        # For pre-assigned sites, set distance to 0 (they're at the correct bus)
        sites_df.loc[pre_assigned_mask, 'distance_km'] = 0.0
    
    if method == 'nearest':
        # Detect coordinate system and use appropriate distance metric
        bus_coords_xy = network.buses[['x', 'y']].values
        site_coords_lonlat = sites_df[[lon_col, lat_col]].values
        
        # Check if coordinates are in WGS84 (degrees) or meters (OSGB36)
        bus_x_range = bus_coords_xy[:, 0].max() - bus_coords_xy[:, 0].min()
        site_x_range = site_coords_lonlat[:, 0].max() - site_coords_lonlat[:, 0].min()
        
        # WGS84: lon in [-180, 180], lat in [-90, 90]
        # OSGB36 meters: roughly [0, 700000] for UK
        is_wgs84_buses = bus_x_range < 180 and (bus_coords_xy[:, 0].max() <= 180)
        is_wgs84_sites = site_x_range < 180 and (site_coords_lonlat[:, 0].max() <= 180)
        
        logger.info(f"Bus coordinates: WGS84={is_wgs84_buses}, Site coordinates: WGS84={is_wgs84_sites}")
        
        if is_wgs84_buses and is_wgs84_sites:
            # Use haversine distance for WGS84 coordinates (proper geographic distance)
            logger.info("Using haversine distance for geographic coordinates")
            from sklearn.metrics.pairwise import haversine_distances
            
            # Convert to radians for haversine calculation
            bus_coords_radians = np.deg2rad(bus_coords_xy[:, [1, 0]])  # Convert (lon, lat) to (lat, lon)
            site_coords_radians = np.deg2rad(site_coords_lonlat[:, [1, 0]])  # Convert (lon, lat) to (lat, lon)
            
            # Calculate haversine distances (result is in radians)
            earth_radius_km = 6371.0
            distances_radians = haversine_distances(site_coords_radians, bus_coords_radians)
            distances_km = distances_radians * earth_radius_km
            
            # Find nearest bus for each site
            indices = np.argmin(distances_km, axis=1)
            nearest_distances_km = distances_km[np.arange(len(distances_km)), indices]
            
            # Ensure indices is 1D array for consistent indexing
            if indices.ndim > 1:
                indices = indices.flatten()
            
            valid_matches = nearest_distances_km <= max_distance_km
            within_distance = valid_matches.sum()
            outside_distance = (~valid_matches).sum()
            logger.info(f"{within_distance} sites within {max_distance_km}km, {outside_distance} sites outside")
            
            for i, (valid, bus_idx, dist_km) in enumerate(zip(valid_matches, indices, nearest_distances_km)):
                # Skip if this site already has a pre-assigned bus
                if pre_assigned_mask.iloc[i]:
                    continue
                if valid:
                    bus_name = network.buses.index[bus_idx]
                    sites_df.iloc[i, sites_df.columns.get_loc('bus')] = bus_name
                    sites_df.iloc[i, sites_df.columns.get_loc('distance_km')] = dist_km
        elif not is_wgs84_buses and is_wgs84_sites:
            # MIXED: Buses are in OSGB36 (meters), sites are in WGS84 (degrees)
            # Need to convert site coordinates from WGS84 to OSGB36 before distance calculation
            logger.info("Converting site coordinates from WGS84 to OSGB36 for distance calculation")
            try:
                from pyproj import Transformer
                # Transform sites from WGS84 (EPSG:4326) to OSGB36 (EPSG:27700)
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
                
                # Convert all site coordinates (handle NaNs)
                site_lons = sites_df[lon_col].values
                site_lats = sites_df[lat_col].values
                valid_coords = ~(np.isnan(site_lons) | np.isnan(site_lats))
                
                # Initialize output arrays
                all_indices = np.full(len(sites_df), -1, dtype=int)
                all_distances_km = np.full(len(sites_df), np.inf)
                
                if valid_coords.sum() > 0:
                    # Transform valid coordinates
                    site_x, site_y = transformer.transform(
                        site_lons[valid_coords], 
                        site_lats[valid_coords]
                    )
                    
                    # Create transformed coordinate array for valid sites only
                    site_coords_osgb36 = np.column_stack([site_x, site_y])
                    
                    # Build spatial index for buses
                    network_id = id(network.buses)
                    bus_hash = hashlib.md5(f"{len(network.buses)}_{network.buses.index[0] if len(network.buses) > 0 else ''}".encode()).hexdigest()[:8]
                    cache_key = f"{network_id}_{bus_hash}"
                    
                    if cache_key in _BUS_SPATIAL_INDEX_CACHE:
                        logger.debug("Using cached bus spatial index")
                        nbrs, _ = _BUS_SPATIAL_INDEX_CACHE[cache_key]
                    else:
                        logger.debug("Building bus spatial index")
                        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(bus_coords_xy)
                        _BUS_SPATIAL_INDEX_CACHE[cache_key] = (nbrs, bus_coords_xy)
                    
                    # Find nearest buses for valid coordinates
                    distances, indices_nn = nbrs.kneighbors(site_coords_osgb36)
                    distances_km_valid = distances.flatten() / 1000.0  # Convert meters to km
                    
                    # Store results for all sites (for fallback logic later)
                    valid_indices = np.where(valid_coords)[0]
                    all_indices[valid_indices] = indices_nn.flatten()
                    all_distances_km[valid_indices] = distances_km_valid
                    
                    # Map back to original indices (only valid coordinates within max_distance)
                    for j, (orig_idx, bus_idx, dist_km) in enumerate(zip(valid_indices, indices_nn.flatten(), distances_km_valid)):
                        # Skip if this site already has a pre-assigned bus
                        if pre_assigned_mask.iloc[orig_idx]:
                            continue
                        if dist_km <= max_distance_km:
                            bus_name = network.buses.index[bus_idx]
                            sites_df.iloc[orig_idx, sites_df.columns.get_loc('bus')] = bus_name
                            sites_df.iloc[orig_idx, sites_df.columns.get_loc('distance_km')] = dist_km
                    
                    logger.info(f"Transformed {valid_coords.sum()} sites from WGS84 to OSGB36 for bus matching")
                else:
                    logger.warning("No valid coordinates found in sites data")
                
                # Store for fallback logic
                indices = all_indices.reshape(-1, 1)
                distances_km = all_distances_km
                    
            except ImportError:
                logger.error("pyproj not available - cannot convert WGS84 to OSGB36")
                raise ValueError("pyproj is required for coordinate conversion between WGS84 and OSGB36")
        else:
            # Both are OSGB36, or sites are OSGB36 (use direct Euclidean distance)
            logger.info("Using Euclidean distance for meter-based coordinates (OSGB36)")
            
            # Filter out sites with NaN coordinates
            site_lons = site_coords_lonlat[:, 0]
            site_lats = site_coords_lonlat[:, 1]
            valid_coords = ~(np.isnan(site_lons) | np.isnan(site_lats))
            
            # Initialize output arrays
            all_indices = np.full(len(sites_df), -1, dtype=int)
            all_distances_km = np.full(len(sites_df), np.inf)
            
            if valid_coords.sum() > 0:
                # Get valid coordinates
                site_coords_valid = site_coords_lonlat[valid_coords]
                
                network_id = id(network.buses)
                bus_hash = hashlib.md5(f"{len(network.buses)}_{network.buses.index[0] if len(network.buses) > 0 else ''}".encode()).hexdigest()[:8]
                cache_key = f"{network_id}_{bus_hash}"
                
                if cache_key in _BUS_SPATIAL_INDEX_CACHE:
                    logger.debug("Using cached bus spatial index")
                    nbrs, bus_coords_array = _BUS_SPATIAL_INDEX_CACHE[cache_key]
                else:
                    logger.debug("Building bus spatial index")
                    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(bus_coords_xy)
                    _BUS_SPATIAL_INDEX_CACHE[cache_key] = (nbrs, bus_coords_xy)
                    logger.debug(f"Cached spatial index for {len(network.buses)} buses")
                
                # Only process valid coordinates
                distances_valid, indices_valid = nbrs.kneighbors(site_coords_valid)
                
                # Store results for valid sites
                all_indices[valid_coords] = indices_valid.flatten()
                all_distances_km[valid_coords] = distances_valid.flatten() / 1000.0
            
            # Now process all sites using the arrays
            valid_matches = all_distances_km <= max_distance_km
            
            for i, (valid, bus_idx, dist_km) in enumerate(zip(valid_matches, all_indices, all_distances_km)):
                # Skip if this site already has a pre-assigned bus
                if pre_assigned_mask.iloc[i]:
                    continue
                if valid and bus_idx >= 0:  # Check bus_idx is valid
                    bus_name = network.buses.index[bus_idx]
                    sites_df.iloc[i, sites_df.columns.get_loc('bus')] = bus_name
                    sites_df.iloc[i, sites_df.columns.get_loc('distance_km')] = dist_km
        
            # Store for fallback logic
            indices = all_indices.reshape(-1, 1)
            distances_km = all_distances_km
        
        matched_sites = sites_df['bus'].notna().sum()
        unmapped_sites = len(sites_df) - matched_sites
        newly_mapped = matched_sites - n_pre_assigned
        logger.info(f"Mapped {matched_sites}/{len(sites_df)} sites to buses ({n_pre_assigned} pre-assigned, {newly_mapped} newly mapped, max distance: {max_distance_km}km)")
        
        if unmapped_sites > 0:
            logger.info(f"Applying fallback mapping for {unmapped_sites} unmapped sites (no distance limit)")
            unmapped_mask = sites_df['bus'].isna()
            unmapped_indices = sites_df.index[unmapped_mask].tolist()
            
            for idx, i in enumerate(unmapped_indices[:5]):  # Log first 5
                idx_in_coords = sites_df.index.get_loc(i)
                
                if is_wgs84_buses and is_wgs84_sites:
                    # WGS84 branch: indices is 1D array
                    bus_idx = indices[idx_in_coords]
                    dist_km = nearest_distances_km[idx_in_coords]
                else:
                    # OSGB36/MIXED branch: indices is 2D array (n_samples, 1)
                    if idx_in_coords < len(indices):
                        bus_idx = indices[idx_in_coords, 0] if indices.ndim > 1 else indices[idx_in_coords]
                        dist_km = distances_km[idx_in_coords]  # distances_km is always 1D after flatten()
                
                bus_name = network.buses.index[bus_idx]
                logger.info(f"Fallback mapped site {i} (idx {idx_in_coords}) to bus {bus_name} (distance: {dist_km:.1f}km)")
            
            # Now map all unmapped sites
            for i in unmapped_indices:
                idx_in_coords = sites_df.index.get_loc(i)
                
                if is_wgs84_buses and is_wgs84_sites:
                    # WGS84 branch: indices is 1D array
                    bus_idx = indices[idx_in_coords]
                    dist_km = nearest_distances_km[idx_in_coords]
                else:
                    # OSGB36/MIXED branch: indices is 2D array (n_samples, 1)
                    if idx_in_coords < len(indices):
                        bus_idx = indices[idx_in_coords, 0] if indices.ndim > 1 else indices[idx_in_coords]
                        dist_km = distances_km[idx_in_coords] if distances_km.ndim == 1 else distances_km[idx_in_coords, 0]
            
            final_matched = sites_df['bus'].notna().sum()
            logger.info(f"Final mapping: {final_matched}/{len(sites_df)} sites mapped to buses")
    
    elif method == 'region':
        logger.info("Using regional/GSP mapping with fuzzy string matching")
        bus_names = network.buses.index.tolist()
        region_cols = [col for col in sites_df.columns if any(keyword in col.lower() for keyword in ['region', 'gsp', 'zone', 'area'])]
        
        if not region_cols:
            logger.warning("No region/GSP column found, falling back to nearest neighbor")
            return map_sites_to_buses(network, sites_df, method='nearest', bus_key=bus_key, lat_col=lat_col, lon_col=lon_col, max_distance_km=max_distance_km)
        
        region_col = region_cols[0]
        logger.info(f"Using region column: {region_col}")
        
        for idx, site in sites_df.iterrows():
            # Skip if this site already has a pre-assigned bus
            if pre_assigned_mask.loc[idx]:
                continue
            if pd.notna(site[region_col]):
                match, score = fuzzy_match_string(str(site[region_col]).lower(), [name.lower() for name in bus_names], threshold=0.8)
                if match and score > 0.8:
                    matched_bus = [name for name in bus_names if name.lower() == match][0]
                    sites_df.at[idx, 'bus'] = matched_bus
                    sites_df.at[idx, 'distance_km'] = 0.0
        
        matched_sites = sites_df['bus'].notna().sum()
        logger.info(f"Mapped {matched_sites}/{len(sites_df)} sites using regional matching")
    
    else:
        raise ValueError(f"Unknown mapping method: {method}")
    
    unmatched = sites_df['bus'].isna().sum()
    if unmatched > 0:
        logger.warning(f"{unmatched} sites could not be mapped to buses")
    
    return sites_df


def clear_spatial_index_cache():
    """Clear the cached bus spatial indexes (useful for testing or memory management)."""
    global _BUS_SPATIAL_INDEX_CACHE
    cache_size = len(_BUS_SPATIAL_INDEX_CACHE)
    _BUS_SPATIAL_INDEX_CACHE.clear()
    logger.info(f"Cleared {cache_size} cached bus spatial indexes")


def apply_etys_bmu_mapping(
    sites_df: pd.DataFrame, 
    network: pypsa.Network,
    gb_network_path: str = "data/network/ETYS/GB_network.xlsx"
) -> pd.DataFrame:
    """
    Apply official ETYS BMU-to-Node mapping to correct generator bus assignments.
    
    The ETYS network has large generators (Pembroke, West Burton, Torness, etc.) that
    should connect directly to 400kV buses, but spatial nearest-distance mapping often
    assigns them to nearby 132kV buses instead. This causes infeasibility because 
    132kV buses have limited transformer capacity to the 400kV grid.
    
    The GB_network.xlsx file contains the official 'Dir_con_BMUs_to_node' sheet which
    maps Balancing Mechanism Units (BMUs) to their correct ETYS network nodes.
    
    This function uses a two-stage approach:
    1. Official BMU mapping: Match generator names to BMU IDs and use official Node IDs
    2. Prefix heuristic fallback: For unmatched generators, try 400kV buses with same prefix
    
    Args:
        sites_df: DataFrame with generator sites (must have 'bus' column)
        network: PyPSA Network to validate bus existence
        gb_network_path: Path to GB_network.xlsx with BMU mappings
        
    Returns:
        DataFrame with corrected bus assignments
    """
    import os
    
    if 'bus' not in sites_df.columns:
        logger.warning("No 'bus' column in sites_df - cannot apply ETYS BMU mapping")
        return sites_df
    
    sites_df = sites_df.copy()
    corrections_made = 0
    corrections_log = []
    
    # Get bus voltage levels from network
    bus_v_nom = network.buses['v_nom'].to_dict()
    
    # ==========================================================================
    # STAGE 1: Load official BMU-to-Node mapping
    # ==========================================================================
    bmu_mapping = {}  # station_prefix -> official_node_id
    
    # Known power station name to BMU prefix mappings
    # These are hand-curated to match generator names to ETYS node prefixes
    STATION_TO_BMU_PREFIX = {
        'torness': 'TORN',
        'hunterston': 'HUNT',
        'hinkley': 'HINK',
        'heysham': 'HEYS',
        'hartlepool': 'HART',
        'sizewell': 'SIZE',
        'dungeness': 'DUNG',
        'pembroke': 'PEMB',
        'west burton': 'WBUR',
        'drax': 'DRAX',
        'cottam': 'COTT',
        'ratcliffe': 'RATC',
        'didcot': 'DIDC',
        'grain': 'GRAI',
        'seabank': 'SEAB',
        'sutton bridge': 'SUTB',
        'south humber': 'SHBA',
        'saltend': 'SALD',
        'peterhead': 'PEHE',
        'fiddler': 'FIDD',
        'baglan': 'BAGB',
        'staythorpe': 'STAY',
        'keadby': 'KEAD',
        'spalding': 'SPAE',
        'damhead': 'DAMH',
        'cockenzie': 'COCK',
        'longannet': 'LONG',
        'killingholme': 'KILL',
        'little barford': 'LITB',
        'connah': 'CONN',
        'carrington': 'CARR',
        'rocksavage': 'ROCK',
        'immingham': 'IMMM',
        'enfield': 'ENFI',
        'medway': 'MEDW',
        'shoreham': 'SHOR',
        'marchwood': 'MARC',
        'coryton': 'CORY',
        'teesside': 'TEES',
        'wilton': 'WILT',
        'hornsea': 'HORN',
        'beatrice': 'BEAT',
        'moray': 'MORA',
        'triton knoll': 'TRIK',
        'east anglia': 'EANG',
        'london array': 'LOAD',
        'gwynt y mor': 'GWYF',
        'walney': 'WALN',
        'rampion': 'RAMP',
        'race bank': 'RACB',
        'dudgeon': 'DUDG',
        'greater gabbard': 'GRGB',
        'thanet': 'THAN',
        'sheringham shoal': 'SHER',
        'westermost rough': 'WEST',
        'lincs': 'LINC',
        'humber gateway': 'HUMG',
        'robin rigg': 'ROBI',
        'ormonde': 'ORMO',
        'burbo bank': 'BURB',
        'barrow': 'BARR',
        'gunfleet': 'GUNF',
        'kentish flats': 'KENT',
        # Pumped storage stations
        'cruachan': 'CRUA',
        'foyers': 'FOYE',
        'dinorwig': 'DINO',
        'ffestiniog': 'FEST',
        # Interconnector landing points
        'auchencrosh': 'AUCH',
        'moyle': 'AUCH',  # Moyle lands at Auchencrosh
    }
    
    if os.path.exists(gb_network_path):
        try:
            bmu_df = pd.read_excel(gb_network_path, sheet_name='Dir_con_BMUs_to_node')
            logger.info(f"Loaded {len(bmu_df)} BMU mappings from {gb_network_path}")
            
            # Create mapping from prefix to preferred node (prefer 400kV)
            for _, row in bmu_df.iterrows():
                bmu_id = str(row.get('BM Unit Id', ''))
                node_id = str(row.get('Node Id', ''))
                
                if not node_id or node_id not in network.buses.index:
                    continue
                    
                # Extract prefix from node_id (first 4 chars)
                prefix = node_id[:4].upper()
                node_voltage = bus_v_nom.get(node_id, 0)
                
                # Prefer 400kV nodes
                if prefix not in bmu_mapping or node_voltage > bus_v_nom.get(bmu_mapping[prefix], 0):
                    bmu_mapping[prefix] = node_id
            
            logger.info(f"Created {len(bmu_mapping)} prefix-to-node mappings")
            
        except Exception as e:
            logger.warning(f"Could not load BMU mapping from {gb_network_path}: {e}")
    else:
        logger.warning(f"BMU mapping file not found: {gb_network_path}")
    
    # ==========================================================================
    # STAGE 2: Apply corrections to generators
    # ==========================================================================
    for idx, row in sites_df.iterrows():
        bus = row.get('bus')
        if pd.isna(bus) or bus not in bus_v_nom:
            continue
            
        v_nom = bus_v_nom[bus]
        capacity_mw = row.get('capacity_mw', row.get('p_nom', 0))
        
        # Get generator name for matching
        gen_name = str(row.get('station_name', row.get('name', row.get('Site Name', '')))).lower()
        
        # Only check generators > 50 MW (lowered threshold to catch more issues)
        if capacity_mw < 50:
            continue
        
        # Check transformer capacity from this bus (only real transformers, not links)
        xfmr_out = network.transformers[
            (network.transformers['bus0'] == bus) | 
            (network.transformers['bus1'] == bus)
        ]
        xfmr_capacity = xfmr_out['s_nom'].sum()
        
        # For capacity check, only use transformer capacity (lines at same voltage don't help for export)
        # Large generators at 33kV/132kV need transformer capacity to export to higher voltage
        effective_export_capacity = xfmr_capacity
        
        # Flag for voltage-based concern - large generators at low voltage are problematic
        needs_voltage_upgrade = False
        
        # Hard rule: Very large generators (>300MW) should ALWAYS be at 275kV+
        if capacity_mw >= 300 and v_nom < 275:
            needs_voltage_upgrade = True
        # Hard rule: Generators >200MW at 33kV need to move (no valid connection)
        elif capacity_mw >= 200 and v_nom <= 33:
            needs_voltage_upgrade = True
        # Soft rule: Generator exceeds 80% of export transformer capacity
        elif effective_export_capacity > 0 and capacity_mw / effective_export_capacity > 0.8:
            needs_voltage_upgrade = True
        # Soft rule: Generator >100MW with no transformer capacity at 33kV/132kV
        elif capacity_mw >= 100 and v_nom < 275 and effective_export_capacity == 0:
            needs_voltage_upgrade = True
        
        if not needs_voltage_upgrade:
            continue
        
        # Try to find correct bus using station name matching
        new_bus = None
        match_method = None
        
        # Method 1: Match generator name to known station prefixes
        for station_pattern, bmu_prefix in STATION_TO_BMU_PREFIX.items():
            if station_pattern in gen_name:
                if bmu_prefix in bmu_mapping:
                    candidate_bus = bmu_mapping[bmu_prefix]
                    if candidate_bus in network.buses.index and candidate_bus != bus:
                        # Accept station-matched buses that are 275kV or higher (not just 400kV)
                        candidate_v = bus_v_nom.get(candidate_bus, 0)
                        if candidate_v >= 275:
                            new_bus = candidate_bus
                            match_method = 'station_name'
                            break
                else:
                    # Fallback: look for any 275kV+ bus with this prefix (prefer 400kV)
                    potential_buses = [
                        b for b in network.buses.index 
                        if b.startswith(bmu_prefix) and bus_v_nom.get(b, 0) >= 275
                    ]
                    if potential_buses:
                        # Sort by voltage (prefer higher voltage)
                        potential_buses.sort(key=lambda b: -bus_v_nom.get(b, 0))
                        new_bus = potential_buses[0]
                        match_method = 'station_prefix'
                        break
        
        # Method 2: Try same prefix as current bus but at 400kV
        if new_bus is None and v_nom < 400:
            site_prefix = bus[:4]
            potential_400kv_buses = [
                b for b in network.buses.index 
                if b.startswith(site_prefix) and bus_v_nom.get(b, 0) == 400
            ]
            if potential_400kv_buses:
                new_bus = potential_400kv_buses[0]
                match_method = 'prefix_heuristic'
        
        # Method 3: Find nearest 400kV bus with sufficient line capacity
        # For generators that still have no suitable bus after prefix matching
        if new_bus is None and capacity_mw >= 100 and v_nom < 400:
            # Get coordinates of current bus
            try:
                bus_x = network.buses.loc[bus, 'x']
                bus_y = network.buses.loc[bus, 'y']
                
                if pd.notna(bus_x) and pd.notna(bus_y):
                    # Find all 400kV buses with sufficient line capacity
                    candidate_buses = []
                    for b in network.buses.index:
                        if bus_v_nom.get(b, 0) >= 400:
                            # Calculate line capacity at this 400kV bus
                            lines_at_b = network.lines[
                                (network.lines['bus0'] == b) | 
                                (network.lines['bus1'] == b)
                            ]
                            b_line_cap = lines_at_b['s_nom'].sum()
                            
                            # Must have at least 1.5x the generator capacity in line capacity
                            if b_line_cap >= capacity_mw * 1.5:
                                b_x = network.buses.loc[b, 'x']
                                b_y = network.buses.loc[b, 'y']
                                if pd.notna(b_x) and pd.notna(b_y):
                                    dist = ((bus_x - b_x)**2 + (bus_y - b_y)**2)**0.5
                                    candidate_buses.append((b, dist, b_line_cap))
                    
                    # Sort by distance and pick the nearest
                    if candidate_buses:
                        candidate_buses.sort(key=lambda x: x[1])
                        new_bus = candidate_buses[0][0]
                        match_method = f'nearest_400kV_{candidate_buses[0][1]/1000:.0f}km'
            except Exception as e:
                logger.debug(f"Could not find nearest 400kV bus for {gen_name}: {e}")
        
        # Apply correction if we found a better bus
        if new_bus and new_bus != bus:
            old_bus = bus
            sites_df.at[idx, 'bus'] = new_bus
            corrections_made += 1
            
            corrections_log.append({
                'generator': gen_name,
                'capacity_mw': capacity_mw,
                'old_bus': old_bus,
                'old_v_nom': v_nom,
                'old_xfmr_capacity': xfmr_capacity,
                'new_bus': new_bus,
                'new_v_nom': bus_v_nom.get(new_bus, 0),
                'match_method': match_method
            })
    
    if corrections_made > 0:
        logger.info(f"ETYS BMU Mapping: Corrected {corrections_made} generator bus assignments")
        logger.info("  Moved generators to correct connection points:")
        for c in corrections_log[:15]:  # Show first 15
            gen_name = str(c['generator'])[:30] if c['generator'] else 'Unknown'
            logger.info(f"    {gen_name:30s} ({c['capacity_mw']:6.0f} MW): {c['old_bus']} ({c['old_v_nom']:.0f}kV) → {c['new_bus']} ({c['new_v_nom']:.0f}kV) [{c['match_method']}]")
        if len(corrections_log) > 15:
            logger.info(f"    ... and {len(corrections_log) - 15} more")
        
        total_corrected_capacity = sum(c['capacity_mw'] for c in corrections_log)
        logger.info(f"  Total capacity corrected: {total_corrected_capacity:.0f} MW")
    else:
        logger.info("ETYS BMU Mapping: No corrections needed (all generators already at correct buses)")
    
    return sites_df

