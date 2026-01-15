#!/usr/bin/env python3
"""
Map Interconnectors to Network Buses
====================================

This script maps interconnector landing points to specific network buses
for different network models (ETYS, Reduced, Zonal). It handles the mapping
from geographic landing points to PyPSA network bus identifiers.

Key features:
- Network-specific bus mapping 
- Error handling for missing mappings
- Synthetic bus creation for external connections
- Comprehensive mapping coverage reporting
- Flexible mapping file format support

Author: PyPSA-GB Team
"""

import sys
import pandas as pd
import logging
from pathlib import Path
import re
import time
from typing import Dict, Optional

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared spatial utilities
from scripts.utilities.spatial_utils import map_sites_to_buses

try:
    from scripts.utilities.logging_config import setup_logging, log_execution_summary
except ImportError:
    import logging
    def setup_logging(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

# Check if running in Snakemake context
if 'snakemake' in globals():
    SNAKEMAKE_MODE = True
    input_clean = snakemake.input[0]
    network_file = snakemake.input[1]
    bus_mapping_file = snakemake.params.get('bus_mapping_file', None)
    output_mapped = snakemake.output[0]
    network_model = snakemake.wildcards.network_model
else:
    SNAKEMAKE_MODE = False

def load_bus_mapping(bus_mapping_file: str, target_network_model: str) -> pd.DataFrame:
    """
    Load bus mapping data for the specified network model.
    
    Args:
        bus_mapping_file: Path to bus mapping CSV file
        target_network_model: Network model to filter for (ETYS, Reduced, Zonal)
        
    Returns:
        DataFrame with bus mapping for the target network
    """
    logger = logging.getLogger(__name__)
    
    if not Path(bus_mapping_file).exists():
        logger.warning(f"Bus mapping file not found: {bus_mapping_file}")
        return pd.DataFrame()
    
    try:
        bus_mapping = pd.read_csv(bus_mapping_file)
        logger.info(f"Loaded bus mapping with {len(bus_mapping)} records")
        
        # Filter for target network model
        if 'network_model' in bus_mapping.columns:
            network_mapping = bus_mapping[bus_mapping['network_model'] == target_network_model]
            logger.info(f"Filtered to {len(network_mapping)} mappings for {target_network_model} network")
        else:
            logger.warning("No network_model column found - using all mappings")
            network_mapping = bus_mapping.copy()
        
        return network_mapping
        
    except Exception as e:
        logger.error(f"Error loading bus mapping: {e}")
        return pd.DataFrame()

def create_external_bus_name(country: str, landing_point: str = None) -> str:
    """
    Create a standardized external bus name for interconnector endpoints.
    
    Args:
        country: Counterparty country name
        landing_point: Optional specific landing point
        
    Returns:
        Standardized external bus name
    """
    # Clean country name
    country_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(country).strip().title())
    
    if landing_point and pd.notna(landing_point):
        landing_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(landing_point).strip().title())
        return f"HVDC_External_{country_clean}_{landing_clean}"
    else:
        return f"HVDC_External_{country_clean}"

def map_landing_points_to_buses(interconnectors_df: pd.DataFrame, 
                                bus_mapping_df: pd.DataFrame, 
                                network_model: str) -> pd.DataFrame:
    """
    Map interconnector landing points to network buses.
    
    Args:
        interconnectors_df: Clean interconnector data
        bus_mapping_df: Bus mapping data for the network model
        network_model: Target network model name
        
    Returns:
        DataFrame with from_bus and to_bus columns added
    """
    logger = logging.getLogger(__name__)
    
    # Create output DataFrame with exact required schema
    output_columns = [
        'name', 'from_bus', 'to_bus', 'capacity_mw', 'losses_percent', 
        'dc', 'commissioning_year', 'counterparty_country', 'source'
    ]
    
    result_df = pd.DataFrame(columns=output_columns)
    
    # Copy basic interconnector data
    for col in interconnectors_df.columns:
        if col in output_columns:
            result_df[col] = interconnectors_df[col].copy()
    
    # Initialize bus columns
    result_df['from_bus'] = None
    result_df['to_bus'] = None
    
    # Track mapping statistics
    mapped_count = 0
    unmapped_count = 0
    unmapped_landing_points = set()
    
    # Create mapping lookup
    if len(bus_mapping_df) > 0 and 'bus' in bus_mapping_df.columns:
        # Filter for the specific network model
        network_mapping = bus_mapping_df[bus_mapping_df['network_model'] == network_model].copy()
        
        # Create lookup dictionaries for both landing points and interconnector names
        landing_point_lookup = {}
        name_lookup = {}
        
        if 'landing_point' in network_mapping.columns:
            # Normalize landing point names for matching
            network_mapping['landing_point_normalized'] = network_mapping['landing_point'].str.lower().str.strip()
            landing_point_lookup = dict(zip(network_mapping['landing_point_normalized'], network_mapping['bus']))
        
        if 'interconnector_name' in network_mapping.columns:
            # Create interconnector name lookup (exclude empty names)
            name_subset = network_mapping[network_mapping['interconnector_name'].notna() & 
                                        (network_mapping['interconnector_name'] != '')]
            name_lookup = dict(zip(name_subset['interconnector_name'], name_subset['bus']))
        
        logger.info(f"Created bus lookup with {len(landing_point_lookup)} landing point mappings and {len(name_lookup)} name mappings")
    else:
        landing_point_lookup = {}
        name_lookup = {}
        logger.warning("No valid bus mapping data available")
    
    # Map each interconnector
    for idx, row in interconnectors_df.iterrows():
        name = row.get('name', f'interconnector_{idx}')
        landing_point_gb = row.get('landing_point_gb', '')
        counterparty_country = row.get('counterparty_country', 'Unknown')
        counterparty_landing = row.get('counterparty_landing_point', '')
        
        from_bus = None
        
        # First try mapping by interconnector name
        if name in name_lookup:
            from_bus = name_lookup[name]
            mapped_count += 1
            logger.debug(f"Mapped {name} by name -> {from_bus}")
        
        # If no name match, try landing point
        elif pd.notna(landing_point_gb) and landing_point_gb:
            landing_normalized = str(landing_point_gb).lower().strip()
            
            # Try to find matching bus
            from_bus = landing_point_lookup.get(landing_normalized)
            
            if from_bus:
                mapped_count += 1
                logger.debug(f"Mapped {name}: {landing_point_gb} -> {from_bus}")
            else:
                # Try fuzzy matching
                from_bus = find_fuzzy_bus_match(landing_normalized, landing_point_lookup)
                if from_bus:
                    mapped_count += 1
                    logger.debug(f"Fuzzy mapped {name}: {landing_point_gb} -> {from_bus}")
        
        if from_bus:
            result_df.loc[idx, 'from_bus'] = from_bus
        else:
            unmapped_count += 1
            unmapped_landing_points.add(f"{name} ({landing_point_gb if pd.notna(landing_point_gb) else 'No landing point'})")
            logger.warning(f"No bus mapping found for: {name}")
        
        # Create external bus name for counterparty
        external_bus = create_external_bus_name(counterparty_country, counterparty_landing)
        result_df.loc[idx, 'to_bus'] = external_bus
    
    # Report mapping coverage
    total_count = len(interconnectors_df)
    coverage_percent = (mapped_count / total_count * 100) if total_count > 0 else 0
    
    logger.info(f"Bus mapping coverage: {mapped_count}/{total_count} ({coverage_percent:.1f}%)")
    
    if unmapped_landing_points:
        logger.warning(f"Unmapped landing points ({len(unmapped_landing_points)}): {list(unmapped_landing_points)}")
        logger.warning("Consider adding these to the bus mapping file:")
        for landing_point in sorted(unmapped_landing_points):
            logger.warning(f"  {landing_point},{network_model},<bus_name>")
    
    return result_df

def find_fuzzy_bus_match(landing_point: str, bus_lookup: Dict[str, str]) -> Optional[str]:
    """
    Find fuzzy matches for landing points in bus lookup.
    
    Args:
        landing_point: Normalized landing point name
        bus_lookup: Dictionary of landing_point -> bus mappings
        
    Returns:
        Matched bus name or None
    """
    # Try partial matches
    for mapped_landing, bus in bus_lookup.items():
        # Check if landing point contains mapped name or vice versa
        if landing_point in mapped_landing or mapped_landing in landing_point:
            return bus
    
    # Try word-based matching
    landing_words = set(landing_point.split())
    for mapped_landing, bus in bus_lookup.items():
        mapped_words = set(mapped_landing.split())
        # If significant word overlap, consider it a match
        if len(landing_words & mapped_words) >= min(2, len(landing_words), len(mapped_words)):
            return bus
    
    return None

def validate_mapped_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the mapped interconnector data.
    
    Args:
        df: Mapped interconnector DataFrame
        
    Returns:
        Validated DataFrame
    """
    logger = logging.getLogger(__name__)
    initial_count = len(df)
    
    # Remove records without bus mappings
    valid_mapping = df['from_bus'].notna() & df['to_bus'].notna()
    invalid_count = (~valid_mapping).sum()
    
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} interconnectors without valid bus mappings")
        df = df[valid_mapping]
    
    # Ensure required numeric fields
    if 'capacity_mw' in df.columns:
        invalid_capacity = df['capacity_mw'].isna() | (df['capacity_mw'] <= 0)
        if invalid_capacity.any():
            capacity_invalid_count = invalid_capacity.sum()
            logger.warning(f"Removing {capacity_invalid_count} interconnectors with invalid capacity")
            df = df[~invalid_capacity]
    
    final_count = len(df)
    logger.info(f"Validation complete: {initial_count - final_count} records removed, {final_count} valid interconnectors")
    
    return df

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km).
    
    Args:
        lat1, lon1: Latitude and longitude of point 1
        lat2, lon2: Latitude and longitude of point 2
        
    Returns:
        Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return c * r

def map_by_coordinates(interconnectors_df: pd.DataFrame, bus_mapping_df: pd.DataFrame, 
                      max_distance_km: float = 50.0) -> pd.DataFrame:
    """
    Map interconnectors to buses using coordinate-based nearest neighbor matching.
    
    Args:
        interconnectors_df: DataFrame with interconnector data including gb_latitude, gb_longitude
        bus_mapping_df: DataFrame with bus data including latitude, longitude, bus_id
        max_distance_km: Maximum distance in km to consider a match valid (default 50km)
        
    Returns:
        DataFrame with added 'gb_bus' and 'match_distance_km' columns
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Coordinate-Based Bus Mapping ===")
    logger.info(f"Maximum matching distance: {max_distance_km} km")
    
    # Check that we have necessary coordinate columns
    required_ic_cols = ['gb_latitude', 'gb_longitude']
    required_bus_cols = ['latitude', 'longitude', 'bus_id']
    
    missing_ic = [col for col in required_ic_cols if col not in interconnectors_df.columns]
    missing_bus = [col for col in required_bus_cols if col not in bus_mapping_df.columns]
    
    if missing_ic:
        logger.warning(f"Missing interconnector columns: {missing_ic}")
        return interconnectors_df
    
    if missing_bus:
        logger.warning(f"Missing bus mapping columns: {missing_bus}")
        return interconnectors_df
    
    # Filter to valid coordinates
    valid_ics = interconnectors_df.dropna(subset=['gb_latitude', 'gb_longitude']).copy()
    valid_buses = bus_mapping_df.dropna(subset=['latitude', 'longitude']).copy()
    
    logger.info(f"Interconnectors with coordinates: {len(valid_ics)}/{len(interconnectors_df)}")
    logger.info(f"Buses with coordinates: {len(valid_buses)}/{len(bus_mapping_df)}")
    
    if len(valid_ics) == 0 or len(valid_buses) == 0:
        logger.warning("No valid coordinates available for matching")
        return interconnectors_df
    
    # Initialize result columns
    interconnectors_df['gb_bus'] = None
    interconnectors_df['match_distance_km'] = None
    interconnectors_df['match_method'] = None
    
    matched_count = 0
    
    # For each interconnector, find nearest bus
    for idx, ic_row in valid_ics.iterrows():
        ic_lat = ic_row['gb_latitude']
        ic_lon = ic_row['gb_longitude']
        ic_name = ic_row.get('name', f'IC_{idx}')
        
        # Calculate distance to all buses
        distances = []
        for _, bus_row in valid_buses.iterrows():
            dist = haversine_distance(ic_lat, ic_lon, bus_row['latitude'], bus_row['longitude'])
            distances.append({
                'bus_id': bus_row['bus_id'],
                'distance_km': dist
            })
        
        if not distances:
            continue
        
        # Find nearest bus
        distances_df = pd.DataFrame(distances)
        nearest = distances_df.loc[distances_df['distance_km'].idxmin()]
        
        if nearest['distance_km'] <= max_distance_km:
            interconnectors_df.at[idx, 'gb_bus'] = nearest['bus_id']
            interconnectors_df.at[idx, 'match_distance_km'] = nearest['distance_km']
            interconnectors_df.at[idx, 'match_method'] = 'coordinate'
            matched_count += 1
            logger.info(f"✓ Mapped {ic_name} to {nearest['bus_id']} ({nearest['distance_km']:.2f} km)")
        else:
            logger.warning(f"✗ {ic_name}: Nearest bus is {nearest['distance_km']:.2f} km away (exceeds {max_distance_km} km limit)")
    
    logger.info(f"=== Coordinate Matching Complete: {matched_count}/{len(valid_ics)} matched ===")
    
    return interconnectors_df


def apply_etys_interconnector_voltage_correction(
    interconnectors_df: pd.DataFrame,
    network: 'pypsa.Network',
    min_voltage: float = 275.0
) -> pd.DataFrame:
    """
    Correct interconnector bus assignments to ensure they connect to high-voltage buses.
    
    Interconnectors are major infrastructure (500-2000+ MW) that MUST connect to 
    275kV or 400kV buses. Coordinate-based mapping sometimes incorrectly assigns
    them to nearby 132kV buses (which have insufficient transformer capacity).
    
    Args:
        interconnectors_df: DataFrame with 'gb_bus' or 'from_bus' column
        network: PyPSA Network to validate bus existence and voltages
        min_voltage: Minimum acceptable voltage level (default 275kV)
        
    Returns:
        DataFrame with corrected bus assignments
    """
    logger = logging.getLogger(__name__)
    
    # Determine which column contains the GB bus
    bus_col = 'from_bus' if 'from_bus' in interconnectors_df.columns else 'gb_bus'
    if bus_col not in interconnectors_df.columns:
        logger.warning("No bus column found for voltage correction")
        return interconnectors_df
    
    interconnectors_df = interconnectors_df.copy()
    bus_v_nom = network.buses['v_nom'].to_dict()
    corrections = 0
    
    # Known interconnector landing point to correct bus prefix mappings
    INTERCONNECTOR_BUS_CORRECTIONS = {
        'auchencrosh': 'AUCH2-',  # 275kV bus
        'moyle': 'AUCH2-',         # Same location as Auchencrosh
    }
    
    for idx, row in interconnectors_df.iterrows():
        current_bus = row.get(bus_col)
        if pd.isna(current_bus) or current_bus not in network.buses.index:
            continue
        
        current_v = bus_v_nom.get(current_bus, 0)
        capacity = row.get('capacity_mw', 0)
        name = row.get('name', f'IC_{idx}').lower()
        landing = str(row.get('landing_point_gb', '')).lower()
        
        # Check if voltage is too low for the capacity
        if current_v >= min_voltage:
            continue  # Already at appropriate voltage
        
        # Try known corrections first
        new_bus = None
        for key, correct_bus in INTERCONNECTOR_BUS_CORRECTIONS.items():
            if key in name or key in landing:
                if correct_bus in network.buses.index:
                    new_bus = correct_bus
                    break
        
        # If no specific correction, find nearest 275kV+ bus
        if new_bus is None and capacity >= 200:
            try:
                bus_x = network.buses.loc[current_bus, 'x']
                bus_y = network.buses.loc[current_bus, 'y']
                
                if pd.notna(bus_x) and pd.notna(bus_y):
                    # Find nearest HV bus with sufficient capacity
                    candidates = []
                    for b in network.buses.index:
                        if bus_v_nom.get(b, 0) >= min_voltage:
                            lines = network.lines[(network.lines['bus0'] == b) | (network.lines['bus1'] == b)]
                            line_cap = lines['s_nom'].sum()
                            if line_cap >= capacity:
                                b_x = network.buses.loc[b, 'x']
                                b_y = network.buses.loc[b, 'y']
                                if pd.notna(b_x) and pd.notna(b_y):
                                    dist = ((bus_x - b_x)**2 + (bus_y - b_y)**2)**0.5
                                    candidates.append((b, dist, line_cap))
                    
                    if candidates:
                        candidates.sort(key=lambda x: x[1])
                        new_bus = candidates[0][0]
            except Exception as e:
                logger.debug(f"Could not find HV bus for {name}: {e}")
        
        if new_bus and new_bus != current_bus:
            new_v = bus_v_nom.get(new_bus, 0)
            logger.info(f"  Corrected {row.get('name', 'IC')}: {current_bus} ({current_v:.0f}kV) → {new_bus} ({new_v:.0f}kV)")
            interconnectors_df.at[idx, bus_col] = new_bus
            corrections += 1
    
    if corrections > 0:
        logger.info(f"ETYS Voltage Correction: Moved {corrections} interconnector(s) to high-voltage buses")
    
    return interconnectors_df


def main():
    """Main processing function."""
    logger = setup_logging("map_interconnectors_to_buses")
    start_time = time.time()
    
    try:
        logger.info("Starting interconnector bus mapping...")
        logger.info(f"Running in {'Snakemake' if SNAKEMAKE_MODE else 'standalone'} mode")
        
        if SNAKEMAKE_MODE:
            clean_file = input_clean
            # Use network file to extract buses instead of separate bus mapping file
            if bus_mapping_file and Path(bus_mapping_file).exists():
                bus_mapping_file_path = bus_mapping_file
            else:
                # Extract buses from network file
                import pypsa
                from pyproj import Transformer
                
                n = pypsa.Network(network_file)
                
                # Detect coordinate system (OSGB36 vs WGS84)
                # OSGB36 coordinates are in meters (large values), WGS84 in degrees (small values)
                sample_x = n.buses['x'].iloc[0] if 'x' in n.buses.columns else 0
                is_osgb36 = abs(sample_x) > 100  # OSGB36 coords are typically 100,000+
                
                if is_osgb36:
                    logger.info("Detected OSGB36 coordinates in network buses, converting to WGS84 for matching...")
                    # EPSG:27700 = British National Grid (OSGB36), EPSG:4326 = WGS84
                    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                else:
                    logger.info("Detected WGS84 coordinates in network buses")
                    transformer = None
                
                # Create bus mapping from network
                bus_mapping_data = []
                for bus_name in n.buses.index:
                    x = n.buses.loc[bus_name, 'x'] if 'x' in n.buses.columns else None
                    y = n.buses.loc[bus_name, 'y'] if 'y' in n.buses.columns else None
                    
                    if x is not None and y is not None and transformer:
                        # Convert OSGB36 (x,y) to WGS84 (lon,lat)
                        lon, lat = transformer.transform(x, y)
                    else:
                        # Already WGS84 or no coordinates
                        lon, lat = x, y
                    
                    bus_mapping_data.append({
                        'bus_id': bus_name,
                        'network_model': network_model,
                        'latitude': lat,
                        'longitude': lon,
                        'x_osgb36': x if is_osgb36 else None,
                        'y_osgb36': y if is_osgb36 else None
                    })
                bus_mapping_df = pd.DataFrame(bus_mapping_data)
                logger.info(f"Extracted {len(bus_mapping_df)} buses from network file: {network_file}")
                if is_osgb36:
                    logger.info(f"Converted {len(bus_mapping_df)} buses from OSGB36 to WGS84")
                bus_mapping_file_path = None  # Signal to use DataFrame directly
            output_file = output_mapped
            target_network_model = network_model
        else:
            clean_file = "resources/interconnectors/interconnectors_clean.csv"
            bus_mapping_file_path = "data/interconnectors/bus_mapping.csv"
            output_file = "resources/interconnectors/interconnectors_mapped_ETYS.csv"
            target_network_model = "ETYS"
        
        logger.info(f"Clean data file: {clean_file}")
        if bus_mapping_file_path:
            logger.info(f"Bus mapping file: {bus_mapping_file_path}")
        else:
            logger.info(f"Using buses extracted from network")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Target network model: {target_network_model}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load clean interconnector data
        if not Path(clean_file).exists():
            raise FileNotFoundError(f"Clean data file not found: {clean_file}")
        
        interconnectors_df = pd.read_csv(clean_file)
        logger.info(f"Loaded {len(interconnectors_df)} clean interconnector records")
        
        # Load bus mapping data
        if bus_mapping_file_path:
            bus_mapping_df = load_bus_mapping(bus_mapping_file_path, target_network_model)
        # else: Already created from network above
        
        # First try coordinate-based mapping if we have bus coordinates
        if 'latitude' in bus_mapping_df.columns and 'longitude' in bus_mapping_df.columns:
            logger.info("Attempting coordinate-based bus mapping...")
            coord_mapped_df = map_by_coordinates(interconnectors_df, bus_mapping_df, max_distance_km=50.0)
            
            # Count how many were mapped
            coord_mapped = coord_mapped_df['gb_bus'].notna().sum()
            logger.info(f"Coordinate mapping found {coord_mapped}/{len(interconnectors_df)} matches")
            
            # Convert coordinate mapping to standard format
            # Create a copy for the final output with proper columns
            mapped_df = coord_mapped_df.copy()
            
            # Initialize from_bus and to_bus columns FIRST (always needed, even if coord_mapped == 0)
            mapped_df['from_bus'] = None
            mapped_df['to_bus'] = None
            
            # For coordinate-matched ones, set from_bus to gb_bus
            if coord_mapped > 0:
                # For matched interconnectors, set from_bus to gb_bus
                matched_mask = mapped_df['gb_bus'].notna()
                mapped_df.loc[matched_mask, 'from_bus'] = mapped_df.loc[matched_mask, 'gb_bus']
                
                # Create external buses for to_bus
                for idx in mapped_df[matched_mask].index:
                    country = mapped_df.at[idx, 'counterparty_country'] if 'counterparty_country' in mapped_df.columns else 'External'
                    landing = mapped_df.at[idx, 'international_location'] if 'international_location' in mapped_df.columns else None
                    mapped_df.at[idx, 'to_bus'] = create_external_bus_name(country, landing)
            
            # For any that weren't matched by coordinates, try manual mapping
            unmapped_mask = mapped_df['from_bus'].isna()
            if unmapped_mask.any():
                logger.info(f"Attempting manual bus mapping for {unmapped_mask.sum()} remaining interconnectors...")
                # Apply the original mapping function only to unmapped ones
                temp_df = map_landing_points_to_buses(
                    interconnectors_df[unmapped_mask], 
                    bus_mapping_df, 
                    target_network_model
                )
                # Update the mapped_df with results - only update from_bus and to_bus
                if 'from_bus' in temp_df.columns:
                    mapped_df.loc[unmapped_mask, 'from_bus'] = temp_df['from_bus'].values
                if 'to_bus' in temp_df.columns:
                    mapped_df.loc[unmapped_mask, 'to_bus'] = temp_df['to_bus'].values
        else:
            logger.info("No coordinates available, using manual bus mapping only...")
            # Map landing points to buses (original method)
            mapped_df = map_landing_points_to_buses(interconnectors_df, bus_mapping_df, target_network_model)
        
        # Apply voltage-based corrections for ETYS network
        # This ensures large interconnectors connect to 275kV+ buses
        if target_network_model == 'ETYS':
            import pypsa
            n = pypsa.Network(network_file)
            mapped_df = apply_etys_interconnector_voltage_correction(mapped_df, n)
        
        # Validate mapped data
        mapped_df = validate_mapped_data(mapped_df)
        
        # Save mapped data
        mapped_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(mapped_df)} mapped interconnector records to: {output_file}")
        
        # Calculate statistics
        total_interconnectors = len(mapped_df)
        successfully_mapped = mapped_df['from_bus'].notna().sum()
        mapping_success_rate = (successfully_mapped / total_interconnectors * 100) if total_interconnectors > 0 else 0
        total_capacity = mapped_df['capacity_mw'].sum() if len(mapped_df) > 0 else 0
        
        # Log execution summary
        log_execution_summary(
            logger,
            "map_interconnectors_to_buses",
            start_time,
            inputs={'clean_interconnectors': clean_file, 'network': network_file},
            outputs={'mapped_interconnectors': output_file},
            context={
                'total_interconnectors': total_interconnectors,
                'successfully_mapped': successfully_mapped,
                'mapping_success_rate': f"{mapping_success_rate:.1f}%",
                'total_capacity_mw': total_capacity,
                'network_model': target_network_model
            }
        )
        
    except Exception as e:
        logger.error(f"Error in interconnector bus mapping: {e}")
        if SNAKEMAKE_MODE:
            raise
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()

