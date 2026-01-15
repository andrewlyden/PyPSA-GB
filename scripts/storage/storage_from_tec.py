#!/usr/bin/env python3
"""
Extract storage assets from TEC Register.

This script processes the TEC (Transmission Entry Capacity) Register to extract
storage technologies and prepare them for PyPSA-GB integration.

Key features:
- Extract storage from TEC Register (Energy Storage Systems, Pump Storage, etc.)
- Careful deduplication against REPD data
- Filter by operational status
- Standardize output schema matching REPD extraction

Storage technologies extracted:
- Energy Storage System (batteries, BESS)
- Pump Storage (pumped hydro)
- Hybrid systems (storage + generation)

Deduplication strategy:
- Match by site name similarity (fuzzy matching)
- Match by proximity (within 1km radius)
- Match by capacity similarity (within 10%)
- Prefer REPD data when duplicates found

Author: PyPSA-GB Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import difflib
from typing import Tuple, Set

# Set up logging - compatible with both standalone and Snakemake execution
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("storage_from_tec")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


# Storage technology mapping for TEC
STORAGE_PLANT_TYPES = {
    'Battery': [
        'Energy Storage System',
        'BESS',
        'Battery Energy Storage',
        'Battery Storage'
    ],
    'Pumped Storage Hydroelectricity': [
        'Pump Storage',
        'Pumped Storage',
        'Pumped Hydro'
    ]
}

# Flatten for easier lookup
STORAGE_KEYWORDS = []
for tech_list in STORAGE_PLANT_TYPES.values():
    STORAGE_KEYWORDS.extend([kw.lower() for kw in tech_list])


def is_storage_plant_type(plant_type_str: str) -> bool:
    """
    Check if a plant type represents storage.
    
    Args:
        plant_type_str: Plant Type string from TEC
        
    Returns:
        True if storage technology
    """
    if pd.isna(plant_type_str):
        return False
    
    plant_type_lower = str(plant_type_str).lower()
    
    # Check for storage keywords
    storage_indicators = ['storage', 'bess', 'battery', 'pump storage', 'pumped']
    
    for indicator in storage_indicators:
        if indicator in plant_type_lower:
            return True
    
    return False


def standardize_plant_type(plant_type_str: str) -> str:
    """
    Standardize TEC plant type to match REPD technology names.
    
    Args:
        plant_type_str: Original plant type from TEC
        
    Returns:
        Standardized technology name
    """
    if pd.isna(plant_type_str):
        return 'Unknown'
    
    plant_type_lower = str(plant_type_str).lower()
    
    # Check for pump storage first (more specific)
    if 'pump' in plant_type_lower:
        return 'Pumped Storage Hydroelectricity'
    
    # Then check for battery/energy storage
    if any(kw in plant_type_lower for kw in ['battery', 'bess', 'energy storage system']):
        return 'Battery'
    
    # Default
    logger.warning(f"Unknown storage plant type: {plant_type_str}")
    return 'Other Storage'


def map_connection_site_to_bus(connection_site: str, bus_coords: pd.DataFrame, 
                                min_similarity: float = 0.6) -> Tuple[float, float, str]:
    """
    Map a TEC connection site name to bus coordinates using fuzzy name matching.
    
    Args:
        connection_site: Connection site name from TEC (e.g., "DINORWIG 1 400 MAIN")
        bus_coords: DataFrame with bus names as index and x, y columns
        min_similarity: Minimum similarity score (0-1) for name matching
        
    Returns:
        Tuple of (longitude, latitude, matched_bus_name) or (nan, nan, '') if no match
    """
    if pd.isna(connection_site):
        return np.nan, np.nan, ''
    
    # Extract the first part of connection site (typically the substation name)
    # Examples: "DINORWIG 1 400 MAIN" -> "DINORWIG"
    #           "BLYTH 275 T" -> "BLYTH"
    site_parts = str(connection_site).split()
    if not site_parts:
        return np.nan, np.nan, ''
    
    # Try matching with progressively longer substrings
    for i in range(min(4, len(site_parts)), 0, -1):
        search_term = ' '.join(site_parts[:i]).upper()
        
        # Look for buses that start with this search term
        # Bus names are like "DINOR1", "BLYTH1", etc. (6 character codes)
        matches = []
        
        for bus_name in bus_coords.index:
            # Extract readable part of bus name (first 4-5 chars typically)
            bus_prefix = str(bus_name)[:5].upper()
            
            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, search_term[:5], bus_prefix).ratio()
            
            if similarity >= min_similarity:
                matches.append((bus_name, similarity))
        
        if matches:
            # Sort by similarity and take best match
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match = matches[0][0]
            lon = bus_coords.loc[best_match, 'x']
            lat = bus_coords.loc[best_match, 'y']
            
            return lon, lat, best_match
    
    return np.nan, np.nan, ''


def extract_connection_coordinates(tec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract coordinates from TEC connection site names by mapping to network buses.
    
    TEC doesn't provide lat/lon, but connection site names can be matched
    to network bus names to get approximate locations.
    
    Args:
        tec_df: TEC DataFrame with 'Connection Site' column
        
    Returns:
        DataFrame with coordinate columns added (lat, lon, coord_source, matched_bus)
    """
    logger.info("Processing TEC connection sites for coordinates...")
    
    # Initialize coordinate columns
    tec_df['lat'] = np.nan
    tec_df['lon'] = np.nan
    tec_df['coord_source'] = 'TEC_connection_site'
    tec_df['matched_bus'] = ''
    
    # Try to load network bus coordinates
    bus_coord_file = Path("resources/network/bus_coordinates.csv")
    
    if not bus_coord_file.exists():
        logger.warning(f"Bus coordinate file not found: {bus_coord_file}")
        logger.warning("Attempting to extract from existing network file...")
        
        # Try to extract from network file
        try:
            import pypsa
            network_file = Path("resources/network/HT35_clustered_gsp_with_storage.nc")
            if not network_file.exists():
                # Try base network
                network_file = Path("resources/network/HT35_clustered_gsp_base_demand_generators.nc")
            
            if network_file.exists():
                logger.info(f"Loading bus coordinates from network: {network_file}")
                n = pypsa.Network(str(network_file))
                bus_coords = n.buses[['x', 'y']].copy()
                
                # Save for future use
                bus_coords.to_csv(bus_coord_file)
                logger.info(f"Saved {len(bus_coords)} bus coordinates to {bus_coord_file}")
            else:
                logger.error("No network file found to extract bus coordinates")
                return tec_df
        except Exception as e:
            logger.error(f"Failed to extract bus coordinates from network: {e}")
            return tec_df
    else:
        logger.info(f"Loading bus coordinates from: {bus_coord_file}")
        bus_coords = pd.read_csv(bus_coord_file, index_col=0)
        logger.info(f"Loaded {len(bus_coords)} bus coordinates")
    
    # Map each connection site to coordinates
    if 'Connection Site' not in tec_df.columns:
        logger.warning("No 'Connection Site' column in TEC data")
        return tec_df
    
    mapped_count = 0
    for idx, row in tec_df.iterrows():
        connection_site = row['Connection Site']
        lon, lat, matched_bus = map_connection_site_to_bus(connection_site, bus_coords)
        
        if not pd.isna(lon):
            tec_df.loc[idx, 'lon'] = lon
            tec_df.loc[idx, 'lat'] = lat
            tec_df.loc[idx, 'matched_bus'] = matched_bus
            mapped_count += 1
    
    logger.info(f"Successfully mapped {mapped_count}/{len(tec_df)} TEC sites to bus coordinates")
    
    # Report unmapped sites
    unmapped = tec_df[tec_df['lat'].isna()]
    if len(unmapped) > 0:
        logger.warning(f"{len(unmapped)} TEC sites could not be mapped to bus coordinates")
        logger.debug("Unmapped connection sites:")
        for site in unmapped['Connection Site'].unique():
            if pd.notna(site):
                logger.debug(f"  - {site}")
    
    return tec_df


def normalize_site_name(name: str) -> str:
    """
    Normalize site name for comparison.
    
    Args:
        name: Original site name
        
    Returns:
        Normalized name
        
    Note:
        TODO: Improve normalization with:
        - Fuzzy string matching library (fuzzywuzzy, rapidfuzz)
        - UK-specific place name standardization
        - Handle common abbreviations consistently
    """
    if pd.isna(name):
        return ''
    
    # Convert to lowercase
    normalized = str(name).lower().strip()
    
    # Remove common suffixes and words
    remove_words = [
        'power station', 'power plant', 'energy storage', 'battery storage',
        'bess', 'facility', 'site', 'project', 'development',
        'phase 1', 'phase 2', 'phase i', 'phase ii',
        'ltd', 'limited', 'plc', 'inc', 'llc', '- ', ' -'
    ]
    
    for word in remove_words:
        normalized = normalized.replace(word, '')
    
    # Remove extra whitespace and punctuation
    normalized = ' '.join(normalized.split())
    normalized = normalized.replace(',', '').replace('.', '').replace('(', '').replace(')', '')
    
    return normalized.strip()


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Earth radius in km
    R = 6371.0
    
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def find_duplicates_with_repd(tec_df: pd.DataFrame, repd_df: pd.DataFrame,
                               name_similarity_threshold: float = 0.8,
                               distance_threshold_km: float = 1.0,
                               capacity_tolerance: float = 0.1) -> Tuple[Set[int], list]:
    """
    Identify TEC storage entries that are duplicates of REPD entries.
    
    Uses multiple criteria:
    1. Name similarity (fuzzy matching)
    2. Geographic proximity (if coordinates available)
    3. Capacity similarity
    
    Args:
        tec_df: TEC storage DataFrame
        repd_df: REPD storage DataFrame
        name_similarity_threshold: Minimum similarity ratio for name match (0-1)
        distance_threshold_km: Maximum distance for geographic match
        capacity_tolerance: Fractional tolerance for capacity match (0.1 = 10%)
        
    Returns:
        Tuple of (duplicate_indices, match_details)
        
    Note:
        TODO: Performance optimization for large datasets:
        - Use spatial indexing (R-tree) for coordinate matching
        - Vectorize similarity calculations where possible
        - Consider ML-based entity resolution for ambiguous cases
    """
    logger.info("Identifying duplicates between TEC and REPD storage...")
    logger.info(f"TEC entries: {len(tec_df)}, REPD entries: {len(repd_df)}")
    
    duplicates = set()
    match_details = []
    
    # Normalize names for comparison
    tec_df['_normalized_name'] = tec_df['site_name'].apply(normalize_site_name)
    repd_df['_normalized_name'] = repd_df['site_name'].apply(normalize_site_name)
    
    for tec_idx, tec_row in tec_df.iterrows():
        tec_name = tec_row['_normalized_name']
        tec_capacity = tec_row.get('capacity_mw', 0)
        tec_lat = tec_row.get('lat')
        tec_lon = tec_row.get('lon')
        
        for _, repd_row in repd_df.iterrows():
            repd_name = repd_row['_normalized_name']
            repd_capacity = repd_row.get('capacity_mw', 0)
            repd_lat = repd_row.get('lat')
            repd_lon = repd_row.get('lon')
            
            # Criteria 1: Name similarity
            if tec_name and repd_name:
                name_similarity = difflib.SequenceMatcher(None, tec_name, repd_name).ratio()
            else:
                name_similarity = 0.0
            
            # Criteria 2: Geographic proximity
            geographic_match = False
            distance_km = np.inf
            
            if (pd.notna(tec_lat) and pd.notna(tec_lon) and 
                pd.notna(repd_lat) and pd.notna(repd_lon)):
                distance_km = calculate_distance_km(tec_lat, tec_lon, repd_lat, repd_lon)
                geographic_match = distance_km < distance_threshold_km
            
            # Criteria 3: Capacity similarity
            capacity_match = False
            capacity_diff = 0.0
            
            if tec_capacity > 0 and repd_capacity > 0:
                capacity_diff = abs(tec_capacity - repd_capacity) / max(tec_capacity, repd_capacity)
                capacity_match = capacity_diff < capacity_tolerance
            
            # Decision logic: consider duplicate if ANY of these strong matches occur
            # Strategy: Conservative - prefer to mark as duplicate to avoid double-counting
            is_duplicate = False
            match_reason = []
            
            # Strong criterion 1: High name similarity alone (>=80%)
            if name_similarity >= name_similarity_threshold:
                is_duplicate = True
                match_reason.append(f"name_sim={name_similarity:.2f}")
            
            # Strong criterion 2: Geographic proximity + similar capacity
            # Both coordinates and capacity must match
            if geographic_match and capacity_match:
                is_duplicate = True
                match_reason.append(f"geo={distance_km:.2f}km+cap={capacity_diff:.1%}")
            
            # Strong criterion 3: Good name similarity + capacity match (even without coords)
            # Useful when one dataset has missing coordinates
            if name_similarity >= 0.7 and capacity_match:
                is_duplicate = True
                match_reason.append(f"name={name_similarity:.2f}+cap={capacity_diff:.1%}")
            
            if is_duplicate:
                duplicates.add(tec_idx)
                match_details.append({
                    'tec_idx': tec_idx,
                    'tec_name': tec_row['site_name'],
                    'repd_name': repd_row['site_name'],
                    'tec_capacity': tec_capacity,
                    'repd_capacity': repd_capacity,
                    'name_similarity': name_similarity,
                    'distance_km': distance_km if pd.notna(distance_km) and distance_km != np.inf else None,
                    'reason': '; '.join(match_reason)
                })
                break  # Found a match, no need to check other REPD entries
    
    # Log matches
    logger.info(f"Found {len(duplicates)} TEC entries that duplicate REPD storage")
    
    if match_details:
        logger.info("\nDuplicate matches (first 20):")
        for i, match in enumerate(match_details[:20], 1):
            logger.info(f"  {i}. TEC: '{match['tec_name']}' ({match['tec_capacity']} MW)")
            logger.info(f"     REPD: '{match['repd_name']}' ({match['repd_capacity']} MW)")
            logger.info(f"     Match: {match['reason']}")
    
    # Clean up temporary columns
    tec_df.drop('_normalized_name', axis=1, inplace=True, errors='ignore')
    repd_df.drop('_normalized_name', axis=1, inplace=True, errors='ignore')
    
    return duplicates, match_details


def main():
    """Main function to extract storage from TEC Register."""
    start_time = time.time()
    logger.info("Starting storage extraction from TEC Register...")
    
    try:
        # Get input and output files
        try:
            # Snakemake mode
            tec_file = snakemake.input.tec
            repd_storage_file = snakemake.input.repd_storage  # For deduplication
            output_file = snakemake.output.storage_tec
            include_pipeline = snakemake.params.get('include_pipeline', False)
            logger.info("Running in Snakemake mode")
        except NameError:
            # Standalone mode
            base_path = Path(__file__).parent.parent.parent
            tec_file = base_path / "data" / "generators" / "tec-register-02-september-2025.csv"
            repd_storage_file = base_path / "resources" / "storage" / "storage_from_repd.csv"
            output_file = base_path / "resources" / "storage" / "storage_from_tec.csv"
            include_pipeline = False
            logger.info("Running in standalone mode")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load TEC register
        logger.info(f"Loading TEC register from: {tec_file}")
        tec_df = pd.read_csv(tec_file)
        logger.info(f"Loaded {len(tec_df)} TEC records")
        
        # Filter for storage technologies
        logger.info("Filtering for storage technologies...")
        storage_mask = tec_df['Plant Type'].apply(is_storage_plant_type)
        storage_tec = tec_df[storage_mask].copy()
        logger.info(f"Found {len(storage_tec)} storage technology records in TEC")
        
        if len(storage_tec) == 0:
            logger.warning("No storage technologies found in TEC data")
            empty_df = pd.DataFrame(columns=[
                'site_name', 'technology', 'capacity_mw', 'lat', 'lon', 
                'status', 'commissioning_year', 'connection_site', 'source'
            ])
            empty_df.to_csv(output_file, index=False)
            return
        
        # Standardize technology names
        logger.info("Standardizing technology names...")
        storage_tec['technology'] = storage_tec['Plant Type'].apply(standardize_plant_type)
        
        # Extract basic information
        storage_tec['site_name'] = storage_tec['Project Name'].fillna('Unknown')
        storage_tec['capacity_mw'] = pd.to_numeric(storage_tec['MW Connected'], errors='coerce').fillna(0)
        storage_tec['status'] = storage_tec['Project Status'].fillna('Unknown')
        storage_tec['connection_site'] = storage_tec['Connection Site'].fillna('Unknown')
        storage_tec['source'] = 'TEC'
        
        # Extract commissioning year from MW Effective From date
        if 'MW Effective From' in storage_tec.columns:
            storage_tec['commissioning_year'] = pd.to_datetime(
                storage_tec['MW Effective From'], errors='coerce'
            ).dt.year
        else:
            storage_tec['commissioning_year'] = np.nan
        
        # Try to get coordinates (this is challenging for TEC data)
        storage_tec = extract_connection_coordinates(storage_tec)
        
        # Filter by operational status if not including pipeline
        if not include_pipeline:
            logger.info("Filtering to operational/built projects only...")
            operational_statuses = ['Built', 'Operational', 'Operating']
            status_mask = storage_tec['status'].isin(operational_statuses)
            storage_tec = storage_tec[status_mask]
            logger.info(f"After operational filter: {len(storage_tec)} storage sites")
        
        # Load REPD storage for deduplication
        logger.info(f"\nLoading REPD storage from: {repd_storage_file}")
        try:
            repd_storage = pd.read_csv(repd_storage_file)
            logger.info(f"Loaded {len(repd_storage)} REPD storage sites for deduplication")
            
            # Find duplicates
            duplicate_indices, match_details = find_duplicates_with_repd(storage_tec, repd_storage)
            
            # Remove duplicates
            if duplicate_indices:
                logger.info(f"Removing {len(duplicate_indices)} duplicate entries (already in REPD)")
                storage_tec = storage_tec[~storage_tec.index.isin(duplicate_indices)].copy()
                logger.info(f"After deduplication: {len(storage_tec)} unique TEC storage sites")
                # Save deduplication details to output dedup log if available
                try:
                    dedup_log_path = Path(getattr(snakemake.output, 'dedup_log', None) or snakemake.output[1])
                except Exception:
                    dedup_log_path = Path(output_file).parent / 'tec_repd_deduplication.csv'

                if match_details:
                    dedup_df = pd.DataFrame(match_details)
                    dedup_df.to_csv(dedup_log_path, index=False)
                    logger.info(f"Saved deduplication details to: {dedup_log_path}")
            else:
                logger.info("No duplicates found - all TEC storage is unique")
                # Ensure an empty dedup log exists for downstream rules
                try:
                    dedup_log_path = Path(getattr(snakemake.output, 'dedup_log', None) or snakemake.output[1])
                except Exception:
                    dedup_log_path = Path(output_file).parent / 'tec_repd_deduplication.csv'
                pd.DataFrame(columns=['tec_idx', 'tec_name', 'repd_name', 'tec_capacity', 'repd_capacity', 'name_similarity', 'distance_km', 'reason']).to_csv(dedup_log_path, index=False)
                logger.info(f"Saved empty deduplication log to: {dedup_log_path}")
                
        except FileNotFoundError:
            logger.warning(f"REPD storage file not found: {repd_storage_file}")
            logger.warning("Proceeding without deduplication")
        except Exception as e:
            logger.error(f"Error during deduplication: {e}")
            logger.warning("Proceeding without deduplication")
        
        # Select and order output columns
        output_columns = [
            'site_name', 'technology', 'capacity_mw', 'lat', 'lon', 
            'status', 'commissioning_year', 'connection_site', 'source'
        ]
        
        # Ensure all columns exist
        for col in output_columns:
            if col not in storage_tec.columns:
                storage_tec[col] = np.nan
        
        output_df = storage_tec[output_columns].copy()
        
        # Clean up data
        output_df['capacity_mw'] = output_df['capacity_mw'].fillna(0)
        output_df['site_name'] = output_df['site_name'].fillna('Unknown Site')
        output_df['technology'] = output_df['technology'].fillna('Unknown')
        output_df['status'] = output_df['status'].fillna('Unknown')
        
        # Save results
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(output_df)} TEC storage sites to: {output_file}")
        
        # Generate summary statistics
        if len(output_df) > 0:
            tech_summary = output_df.groupby('technology').agg({
                'capacity_mw': ['count', 'sum'],
                'lat': lambda x: x.notna().sum()
            }).round(1)
            
            logger.info("\nTEC storage technology summary:")
            for tech in tech_summary.index:
                count = int(tech_summary.loc[tech, ('capacity_mw', 'count')])
                capacity = tech_summary.loc[tech, ('capacity_mw', 'sum')]
                with_coords = int(tech_summary.loc[tech, ('lat', '<lambda>')])
                logger.info(f"  {tech}: {count} sites, {capacity:.1f} MW, {with_coords} with coordinates")
            
            # Status summary
            status_summary = output_df.groupby('status')['capacity_mw'].agg(['count', 'sum']).round(1)
            logger.info("\nTEC storage by status:")
            for status in status_summary.index:
                count = int(status_summary.loc[status, 'count'])
                capacity = status_summary.loc[status, 'sum']
                logger.info(f"  {status}: {count} sites, {capacity:.1f} MW")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'total_tec_records': len(tec_df),
            'storage_records_found': len(storage_tec) + len(duplicate_indices) if 'duplicate_indices' in locals() else len(storage_tec),
            'duplicates_removed': len(duplicate_indices) if 'duplicate_indices' in locals() else 0,
            'final_tec_storage_sites': len(output_df),
            'total_capacity_mw': output_df['capacity_mw'].sum() if len(output_df) > 0 else 0,
            'sites_with_coordinates': output_df['lat'].notna().sum() if len(output_df) > 0 else 0,
            'output_file': str(output_file)
        }
        
        log_execution_summary(logger, "storage_from_tec", execution_time, summary_stats)
        logger.info("Storage extraction from TEC completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in TEC storage extraction: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

