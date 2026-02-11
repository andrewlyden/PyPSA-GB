"""
Integrate thermal generators into PyPSA network with hybrid data source routing.

This script adds dispatchable thermal generators to the network using different
data sources based on scenario type (historical vs future):

HISTORICAL SCENARIOS (2010-2024):
  Data sources (priority order):
    1. DUKES 5.11 (primary) - Authoritative UK government thermal capacity data
    2. REPD (secondary) - Dispatchable renewable thermal (biomass, waste, biogas)
    NO FES FALLBACK - Using only authoritative historical data sources

FUTURE SCENARIOS (2025+):
  Data sources:
    1. FES (primary) - Comprehensive projections for all technologies

The hybrid approach ensures historical accuracy using only real data while
seamlessly transitioning to projections for future scenarios.

Technologies integrated:
- Conventional thermal: CCGT, OCGT, nuclear, coal (from DUKES/FES)
- Dispatchable renewable thermal: biomass, waste, biogas (from REPD)
- Geothermal: constant baseload renewable (from REPD)

Input:
  - Network with renewable generators already integrated
  - Historical: DUKES_{year}_generators.csv + REPD sites (NO FES)
  - Future: FES_{year}_data.csv
  - Generator characteristics (fuel data, efficiency, costs)

Output:
  - Network with thermal generators integrated
  - Summary CSV of thermal capacity by technology with data_source provenance

Author: PyPSA-GB Team
Date: 2025-10-10
"""

import pandas as pd
import pypsa
import os
from pathlib import Path
import time
import yaml

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info, log_execution_summary, log_stage_timing, log_stage_summary
from scripts.utilities.carrier_definitions import get_carrier_definitions, add_carriers_to_network
import difflib

# Import shared spatial utilities and renewable generator functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scripts.utilities.spatial_utils import map_sites_to_buses, apply_etys_bmu_mapping
from integrate_renewable_generators import load_generator_characteristics
import numpy as np

# Initialize logging
logger = setup_logging("integrate_thermal_generators")


def build_historical_bus_distribution(network: pypsa.Network, carrier: str) -> dict:
    """
    Build a bus distribution pattern based on existing generators of a carrier type.
    
    For carriers like CCGT, Nuclear, etc., this creates a probability distribution
    of capacity across buses based on historical/existing generator locations.
    
    Args:
        network: PyPSA Network with existing generators
        carrier: Carrier type to analyze (e.g., 'CCGT', 'nuclear')
        
    Returns:
        Dictionary mapping bus names to capacity weights (normalized to sum to 1)
    """
    # Find existing generators of this carrier type (case-insensitive match)
    carrier_lower = carrier.lower()
    existing_gens = network.generators[
        network.generators['carrier'].str.lower() == carrier_lower
    ]
    
    if len(existing_gens) == 0:
        # Try partial matching for similar carriers
        # e.g., 'gas' might match 'CCGT', 'OCGT', 'gas_engine'
        carrier_mappings = {
            'ccgt': ['ccgt', 'gas', 'natural_gas'],
            'ocgt': ['ocgt', 'gas_peaker', 'oil'],  # Oil peakers often co-located with gas peakers
            'nuclear': ['nuclear'],
            'coal': ['coal'],
            'solar': ['solar', 'solar_pv'],
            'wind': ['wind_onshore', 'wind_offshore', 'onshore_wind', 'offshore_wind'],
            'hydro': ['hydro', 'large_hydro', 'small_hydro'],
            'biomass': ['biomass', 'bioenergy'],
            'h2': ['h2', 'hydrogen'],
            'oil': ['oil', 'ocgt', 'gas_peaker'],  # Oil often at same sites as gas peakers
        }
        
        related_carriers = carrier_mappings.get(carrier_lower, [carrier_lower])
        existing_gens = network.generators[
            network.generators['carrier'].str.lower().isin(related_carriers)
        ]
    
    if len(existing_gens) == 0:
        logger.debug(f"No existing generators found for carrier '{carrier}'")
        return {}
    
    # Calculate capacity at each bus
    bus_capacity = existing_gens.groupby('bus')['p_nom'].sum()
    total_capacity = bus_capacity.sum()
    
    if total_capacity <= 0:
        return {}
    
    # Normalize to probability weights
    bus_weights = (bus_capacity / total_capacity).to_dict()
    
    logger.debug(f"Built distribution for '{carrier}': {len(bus_weights)} buses, "
                 f"total {total_capacity:.1f} MW")
    
    return bus_weights


def _extract_transmission_region(gsp_value: str) -> str:
    """
    Extract transmission region from GSP string like 'Direct(NGET)'.

    The three transmission network operators in GB are:
    - SHETL: Scottish Hydro Electric Transmission Limited (Northern Scotland, lat > 57.0)
    - SPTL: Scottish Power Transmission Limited (Southern Scotland, 55.5 < lat <= 57.0)
    - NGET: National Grid Electricity Transmission (England & Wales, lat <= 55.5)

    Args:
        gsp_value: GSP string, e.g., 'Direct(NGET)', 'Direct(SPTL)', 'Direct(SHETL)'

    Returns:
        Region code: 'SHETL', 'SPTL', or 'NGET'
    """
    gsp_str = str(gsp_value)
    if 'SHETL' in gsp_str:
        return 'SHETL'
    elif 'SPTL' in gsp_str:
        return 'SPTL'
    return 'NGET'  # Default for Direct(NGET) and unspecified


def _get_buses_in_region(network: pypsa.Network, region: str) -> list:
    """
    Get list of bus names within a transmission region based on latitude.

    Uses approximate latitude bands matching the transmission network boundaries:
    - SHETL: Northern Scotland (y > 57.0)
    - SPTL: Southern Scotland (55.5 < y <= 57.0)
    - NGET: England & Wales (y <= 55.5)

    Args:
        network: PyPSA Network with buses having 'y' coordinate (latitude)
        region: Region code ('SHETL', 'SPTL', or 'NGET')

    Returns:
        List of bus names in the region
    """
    buses = network.buses

    if region == 'SHETL':
        region_buses = buses[buses['y'] > 57.0]
    elif region == 'SPTL':
        region_buses = buses[(buses['y'] > 55.5) & (buses['y'] <= 57.0)]
    else:  # NGET
        region_buses = buses[buses['y'] <= 55.5]

    return region_buses.index.tolist()


def distribute_fes_generators_spatially(
    fes_data: pd.DataFrame,
    network: pypsa.Network,
    repd_sites: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Distribute FES "Direct" connected generators spatially based on transmission region.

    For FES generators connected to "Direct(NGET)", "Direct(SPTL)", or "Direct(SHETL)",
    this function:
    1. Extracts the transmission region from the GSP field
    2. Filters buses to only those within the geographic region:
       - SHETL: Northern Scotland (lat > 57.0)
       - SPTL: Southern Scotland (55.5 < lat <= 57.0)
       - NGET: England & Wales (lat <= 55.5)
    3. Looks up historical bus distribution for that carrier within the region
    4. Distributes the FES capacity proportionally across region-appropriate buses

    This ensures future scenarios maintain realistic spatial distribution of generation
    within the correct transmission network areas.

    Args:
        fes_data: DataFrame with FES generators (must have 'gsp', 'capacity_mw', 'fuel_type' columns)
        network: PyPSA Network with existing generators and buses
        repd_sites: Optional REPD sites DataFrame for additional spatial reference

    Returns:
        Updated DataFrame with generators distributed across buses within their regions
    """
    if 'gsp' not in fes_data.columns:
        logger.info("No GSP column in FES data - skipping spatial distribution")
        return fes_data
    
    # Identify "Direct" connected generators
    direct_patterns = ['Direct(NGET)', 'Direct(SPTL)', 'Direct', 'Transmission']
    direct_mask = fes_data['gsp'].apply(
        lambda x: any(p in str(x) for p in direct_patterns) if pd.notna(x) else False
    )
    
    direct_gens = fes_data[direct_mask].copy()
    non_direct_gens = fes_data[~direct_mask].copy()

    if len(direct_gens) == 0:
        logger.info("No 'Direct' connected FES generators to distribute")
        return fes_data

    # Extract transmission region from GSP (e.g., 'Direct(NGET)' -> 'NGET')
    direct_gens['region'] = direct_gens['gsp'].apply(_extract_transmission_region)

    logger.info(f"Distributing {len(direct_gens)} 'Direct' connected FES generators spatially")
    direct_capacity = direct_gens['capacity_mw'].sum()
    logger.info(f"  Total 'Direct' capacity: {direct_capacity:,.1f} MW")

    # Log breakdown by region
    for region in direct_gens['region'].unique():
        region_cap = direct_gens[direct_gens['region'] == region]['capacity_mw'].sum()
        logger.info(f"    {region}: {region_cap:,.1f} MW")

    # Group Direct generators by carrier/fuel_type AND region
    carrier_col = 'fuel_type' if 'fuel_type' in direct_gens.columns else 'carrier'

    distributed_rows = []

    for (carrier, region), carrier_gens in direct_gens.groupby([carrier_col, 'region']):
        carrier_capacity = carrier_gens['capacity_mw'].sum()
        logger.info(f"  Distributing {carrier} in {region}: {carrier_capacity:,.1f} MW")

        # Get the list of buses in this transmission region
        region_buses = set(_get_buses_in_region(network, region))
        if not region_buses:
            logger.warning(f"    No buses found in region {region}, using all buses")
            region_buses = set(network.buses.index)

        # Get historical bus distribution for this carrier
        bus_weights = build_historical_bus_distribution(network, carrier)

        # Filter to only buses in the region
        if bus_weights:
            bus_weights = {b: w for b, w in bus_weights.items() if b in region_buses}
            # Re-normalize weights after filtering
            if bus_weights:
                total_weight = sum(bus_weights.values())
                bus_weights = {b: w / total_weight for b, w in bus_weights.items()}
                logger.info(f"    Using historical pattern: {len(bus_weights)} buses in {region}")

        if not bus_weights:
            # No historical pattern in region - try to use REPD sites if available
            if repd_sites is not None and len(repd_sites) > 0:
                # Look for matching carrier in REPD
                repd_match = repd_sites[repd_sites['fuel_type'].str.lower() == carrier.lower()]
                if len(repd_match) > 0 and 'bus' in repd_match.columns:
                    # Filter REPD sites to region
                    repd_match = repd_match[repd_match['bus'].isin(region_buses)]
                    if len(repd_match) > 0:
                        bus_capacity = repd_match.groupby('bus')['capacity_mw'].sum()
                        total = bus_capacity.sum()
                        if total > 0:
                            bus_weights = (bus_capacity / total).to_dict()
                            logger.info(f"    Using REPD distribution: {len(bus_weights)} buses in {region}")

        if not bus_weights:
            # Still no pattern - use network bus degree in region as fallback
            logger.warning(f"    No historical pattern for {carrier} in {region} - using major buses in region")
            # Find major transmission buses (high connectivity) within the region
            bus_degree = {}
            for line in network.lines.index:
                b0, b1 = network.lines.loc[line, ['bus0', 'bus1']]
                if b0 in region_buses:
                    bus_degree[b0] = bus_degree.get(b0, 0) + 1
                if b1 in region_buses:
                    bus_degree[b1] = bus_degree.get(b1, 0) + 1

            if bus_degree:
                # Use top 20 most connected buses in the region
                sorted_buses = sorted(bus_degree.items(), key=lambda x: x[1], reverse=True)[:20]
                total_degree = sum(d for _, d in sorted_buses)
                bus_weights = {b: d / total_degree for b, d in sorted_buses}
            else:
                # Last resort: distribute evenly across all buses in region
                logger.warning(f"    No connected buses in {region}, using even distribution")
                bus_weights = {b: 1.0 / len(region_buses) for b in region_buses}

        # Distribute this carrier's capacity across buses
        buses_used = 0
        for bus, weight in bus_weights.items():
            if bus not in network.buses.index:
                continue

            allocated_capacity = carrier_capacity * weight
            if allocated_capacity < 0.1:  # Skip very small allocations
                continue

            # Create a new row for this bus allocation
            # Use the first generator in this carrier group as template
            template_row = carrier_gens.iloc[0].to_dict()
            template_row['bus'] = bus
            template_row['capacity_mw'] = allocated_capacity
            # Update station name to indicate distributed allocation
            template_row['station_name'] = f"FES_{carrier}_{region}_{bus}"

            distributed_rows.append(template_row)
            buses_used += 1

        logger.info(f"    Distributed to {buses_used} buses in {region}")
    
    # Combine distributed generators with non-direct generators
    if distributed_rows:
        distributed_df = pd.DataFrame(distributed_rows)
        result_df = pd.concat([non_direct_gens, distributed_df], ignore_index=True)
        logger.info(f"Spatial distribution complete: {len(fes_data)} → {len(result_df)} generator records")
        return result_df
    else:
        return fes_data


def apply_renewable_profiles_to_fes(
    network: pypsa.Network,
    fes_carriers: list,
    renewables_year: int,
    profiles_dir: str = "resources/renewable/profiles"
) -> pypsa.Network:
    """
    Apply weather-dependent capacity factor profiles to FES renewable generators.
    
    For FES generators with renewable carriers (Solar, Wind, Hydro), this function:
    1. Loads the appropriate renewable profile for the weather year
    2. Assigns profiles to FES generators based on their bus location
    3. Uses nearest-neighbor matching from REPD generators at the same/nearby buses
    
    Args:
        network: PyPSA Network with FES generators already added
        fes_carriers: List of FES carrier names that need profiles (e.g., ['Solar', 'Wind'])
        renewables_year: Weather year to use for profiles (e.g., 2020)
        profiles_dir: Directory containing renewable profile CSVs
        
    Returns:
        Network with p_max_pu profiles applied to FES renewable generators
    """
    profiles_path = Path(profiles_dir)
    
    # Map FES carriers to profile file carriers
    carrier_profile_mapping = {
        'Solar': 'solar_pv',
        'solar': 'solar_pv',
        'Wind': 'wind_onshore',  # Default to onshore for generic "Wind"
        'wind': 'wind_onshore',
        'Offshore-Wind': 'wind_offshore',
        'offwind': 'wind_offshore',
        'onwind': 'wind_onshore',
        'Hydro': 'large_hydro',
        'hydro': 'large_hydro',
        'Marine': 'tidal_stream',
        'marine': 'tidal_stream',
    }
    
    fes_gens_updated = 0
    
    for fes_carrier in fes_carriers:
        # Find FES generators with this carrier
        fes_gens = network.generators[network.generators['carrier'] == fes_carrier].copy()
        
        if len(fes_gens) == 0:
            continue
        
        # Skip if already have profiles
        existing_profiles = fes_gens.index.intersection(network.generators_t.p_max_pu.columns)
        fes_gens_needing_profiles = fes_gens.drop(existing_profiles)
        
        if len(fes_gens_needing_profiles) == 0:
            logger.info(f"FES {fes_carrier}: All {len(fes_gens)} generators already have profiles")
            continue
        
        logger.info(f"Applying profiles to {len(fes_gens_needing_profiles)} FES {fes_carrier} generators")
        
        # Get corresponding profile carrier
        profile_carrier = carrier_profile_mapping.get(fes_carrier, fes_carrier.lower())
        profile_file = profiles_path / f"{profile_carrier}_{renewables_year}.csv"
        
        if not profile_file.exists():
            # Try alternate years
            for alt_year in [2020, 2019, 2015]:
                alt_file = profiles_path / f"{profile_carrier}_{alt_year}.csv"
                if alt_file.exists():
                    profile_file = alt_file
                    logger.warning(f"Using {alt_year} profiles for {fes_carrier} (requested {renewables_year} not available)")
                    break
        
        if not profile_file.exists():
            logger.warning(f"No profile file found for {fes_carrier} - using default capacity factor")
            # Apply a reasonable default capacity factor
            default_cf = {
                'Solar': 0.11,  # UK average solar CF
                'solar': 0.11,
                'Wind': 0.28,
                'wind': 0.28,
                'Hydro': 0.35,
                'hydro': 0.35,
            }.get(fes_carrier, 0.3)
            
            for gen_name in fes_gens_needing_profiles.index:
                network.generators_t.p_max_pu[gen_name] = pd.Series(
                    default_cf, index=network.snapshots
                )
            fes_gens_updated += len(fes_gens_needing_profiles)
            continue
        
        # Load profiles
        try:
            profiles_df = pd.read_csv(profile_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded {profile_carrier} profiles: {profiles_df.shape[1]} sites")
        except Exception as e:
            logger.error(f"Failed to load {profile_file}: {e}")
            continue
        
        # Find REPD generators at same buses to get appropriate profiles
        repd_gens = network.generators[
            (network.generators['carrier'] == profile_carrier) &
            (network.generators.index.isin(profiles_df.columns))
        ]
        
        # Create bus-to-profile mapping from REPD generators
        bus_to_profile = {}
        for gen_name, gen in repd_gens.iterrows():
            bus = gen['bus']
            if bus not in bus_to_profile and gen_name in profiles_df.columns:
                bus_to_profile[bus] = gen_name
        
        # For buses without direct match, find nearest bus with a profile
        if len(bus_to_profile) > 0:
            available_buses = list(bus_to_profile.keys())
            
            for gen_name, gen in fes_gens_needing_profiles.iterrows():
                bus = gen['bus']
                
                if bus in bus_to_profile:
                    # Direct match - use profile from REPD generator at same bus
                    profile_source = bus_to_profile[bus]
                else:
                    # Find nearest bus with a profile
                    if bus in network.buses.index:
                        bus_x, bus_y = network.buses.loc[bus, ['x', 'y']]
                        
                        min_dist = float('inf')
                        nearest_profile = None
                        
                        for avail_bus in available_buses:
                            if avail_bus in network.buses.index:
                                ax, ay = network.buses.loc[avail_bus, ['x', 'y']]
                                dist = ((bus_x - ax)**2 + (bus_y - ay)**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_profile = bus_to_profile[avail_bus]
                        
                        profile_source = nearest_profile
                    else:
                        # Use first available profile as fallback
                        profile_source = list(bus_to_profile.values())[0]
                
                if profile_source and profile_source in profiles_df.columns:
                    # Get profile and align to network snapshots
                    profile = profiles_df[profile_source]
                    
                    # Align timestamps - handle year mismatch with leap year safety
                    if hasattr(network.snapshots[0], 'year'):
                        target_year = network.snapshots[0].year
                        
                        # Check if source has leap day but target doesn't
                        import calendar
                        source_is_leap = len(profile.index) > 0 and hasattr(profile.index[0], 'year') and \
                                        calendar.isleap(profile.index[0].year)
                        target_is_leap = calendar.isleap(target_year)
                        
                        if source_is_leap and not target_is_leap:
                            # Drop Feb 29 before year replacement
                            profile = profile[~((profile.index.month == 2) & (profile.index.day == 29))]
                        
                        # Now safe to replace year
                        profile.index = profile.index.map(
                            lambda x: x.replace(year=target_year) if hasattr(x, 'replace') else x
                        )
                    
                    # Reindex to network snapshots
                    profile_aligned = profile.reindex(network.snapshots, method='nearest')
                    profile_aligned = profile_aligned.fillna(profile.mean()).clip(0, 1)
                    
                    network.generators_t.p_max_pu[gen_name] = profile_aligned
                    fes_gens_updated += 1
        else:
            # No REPD reference profiles - use average from all available profiles
            logger.warning(f"No REPD generators with profiles for {profile_carrier} - using regional average")
            
            avg_profile = profiles_df.mean(axis=1)
            
            # Align to network snapshots with leap year safety
            if hasattr(network.snapshots[0], 'year'):
                target_year = network.snapshots[0].year
                
                # Check if source has leap day but target doesn't
                import calendar
                source_is_leap = len(avg_profile.index) > 0 and hasattr(avg_profile.index[0], 'year') and \
                                calendar.isleap(avg_profile.index[0].year)
                target_is_leap = calendar.isleap(target_year)
                
                if source_is_leap and not target_is_leap:
                    # Drop Feb 29 before year replacement
                    avg_profile = avg_profile[~((avg_profile.index.month == 2) & (avg_profile.index.day == 29))]
                
                avg_profile.index = avg_profile.index.map(
                    lambda x: x.replace(year=target_year) if hasattr(x, 'replace') else x
                )
            
            avg_aligned = avg_profile.reindex(network.snapshots, method='nearest')
            avg_aligned = avg_aligned.fillna(avg_profile.mean()).clip(0, 1)
            
            for gen_name in fes_gens_needing_profiles.index:
                network.generators_t.p_max_pu[gen_name] = avg_aligned
                fes_gens_updated += 1
    
    logger.info(f"Applied renewable profiles to {fes_gens_updated} FES generators")
    return network


def load_gsp_to_bus_mapping(network: pypsa.Network) -> dict:
    """
    Load GSP to bus mapping from FES regional breakdown data.
    
    This function creates a mapping from FES GSP names to network buses by:
    1. Loading the GSP-to-node lookup file with coordinates
    2. Finding the nearest network bus for each GSP using coordinates
    3. Adding region names as secondary keys for fuzzy matching
    4. Handling special cases like "Direct(NGET)" and "Direct(SPTL)"
    
    Args:
        network: PyPSA Network object with buses defined
        
    Returns:
        Dictionary mapping GSP names to network bus names
    """
    gsp_lookup_paths = [
        Path("data/FES/FES2022/gsp_gnode_directconnect_region_lookup.csv"),
        Path("data/network/ETYS/fes2024_regional_breakdown_gsp_info.csv"),
    ]
    
    gsp_mapping = {}
    
    # Try to load GSP coordinates from lookup file
    gsp_coords = None
    region_coords = None  # Also extract region names with coords
    
    for lookup_path in gsp_lookup_paths:
        if lookup_path.exists():
            try:
                df = pd.read_csv(lookup_path)
                if 'gsp_name' in df.columns and 'gsp_lat' in df.columns and 'gsp_lon' in df.columns:
                    gsp_coords = df[['gsp_name', 'gsp_lat', 'gsp_lon']].copy()
                    gsp_coords.columns = ['gsp', 'lat', 'lon']
                    # Drop rows with missing coordinates
                    gsp_coords = gsp_coords.dropna(subset=['lat', 'lon'])
                    # Drop duplicate GSP names (keep first)
                    gsp_coords = gsp_coords.drop_duplicates(subset=['gsp'])
                    logger.info(f"Loaded GSP coordinates from {lookup_path}: {len(gsp_coords)} GSPs with valid coordinates")
                    
                    # Also extract region_name with coordinates for human-readable matching
                    if 'region_name' in df.columns:
                        region_coords = df[['region_name', 'gsp_lat', 'gsp_lon']].copy()
                        region_coords.columns = ['gsp', 'lat', 'lon']
                        region_coords = region_coords.dropna(subset=['lat', 'lon', 'gsp'])
                        region_coords = region_coords.drop_duplicates(subset=['gsp'])
                        logger.info(f"  Also extracted {len(region_coords)} region names for matching")
                    break
            except Exception as e:
                logger.warning(f"Could not load GSP lookup from {lookup_path}: {e}")
    
    if gsp_coords is not None and len(gsp_coords) > 0:
        # Import mapping function
        try:
            from spatial_utils import map_sites_to_buses
            
            # Map GSP locations to network buses
            gsp_mapped = map_sites_to_buses(
                network,
                gsp_coords,
                method='nearest',
                lat_col='lat',
                lon_col='lon',
                max_distance_km=100.0
            )
            
            # Build mapping dictionary using gsp_name
            for _, row in gsp_mapped.iterrows():
                if pd.notna(row.get('bus')):
                    gsp_mapping[row['gsp']] = row['bus']
            
            logger.info(f"Mapped {len(gsp_mapping)} GSPs to network buses")
            
            # Also map region names if available
            if region_coords is not None and len(region_coords) > 0:
                region_mapped = map_sites_to_buses(
                    network,
                    region_coords,
                    method='nearest',
                    lat_col='lat',
                    lon_col='lon',
                    max_distance_km=100.0
                )
                
                region_count = 0
                for _, row in region_mapped.iterrows():
                    if pd.notna(row.get('bus')) and row['gsp'] not in gsp_mapping:
                        gsp_mapping[row['gsp']] = row['bus']
                        region_count += 1
                
                logger.info(f"Added {region_count} region name mappings")
                
        except Exception as e:
            logger.warning(f"Could not map GSP coordinates to buses: {e}")
    
    # Add special handling for "Direct(NGET)" and "Direct(SPTL)" entries
    # These are large generators connected directly to the transmission grid
    # We distribute them across major 400kV buses
    
    # Find major 400kV buses in the network (typically used for large generators)
    major_buses = []
    for bus in network.buses.index:
        # ETYS bus names often have voltage suffix; 400kV buses are the main ones
        # Look for buses that are likely 400kV nodes (large substations)
        if any(x in bus.upper() for x in ['DRAK', 'DRAX', 'RATS', 'BEAU', 'DINO', 'HINK', 'SIZE', 'HUNT', 'SELL']):
            major_buses.append(bus)
    
    # If no major buses found, use any bus with high degree (many connections)
    if not major_buses and len(network.buses) > 0:
        # Calculate degree for each bus (number of connected lines)
        bus_degree = {}
        for line in network.lines.index:
            b0, b1 = network.lines.loc[line, ['bus0', 'bus1']]
            bus_degree[b0] = bus_degree.get(b0, 0) + 1
            bus_degree[b1] = bus_degree.get(b1, 0) + 1
        
        # Get top 10 most connected buses
        sorted_buses = sorted(bus_degree.items(), key=lambda x: x[1], reverse=True)
        major_buses = [b for b, _ in sorted_buses[:10]]
    
    # Assign special GSPs to first major bus (will be distributed later)
    if major_buses:
        default_bus = major_buses[0]
        gsp_mapping['Direct(NGET)'] = default_bus
        gsp_mapping['Direct(SPTL)'] = default_bus
        gsp_mapping['Transmission - National Grid'] = default_bus
        gsp_mapping['Transmission'] = default_bus
        logger.info(f"Special GSPs mapped to major bus: {default_bus}")
    
    return gsp_mapping


def apply_gsp_to_bus_mapping(thermal_data: pd.DataFrame, network: pypsa.Network) -> pd.DataFrame:
    """
    Apply GSP-to-bus mapping for FES generators without coordinate-based bus assignments.
    
    This function handles generators that:
    1. Have GSP names but no coordinates (FES aggregated data)
    2. Have GSP names that don't match network bus names
    3. Have special GSP names like "Direct(NGET)"
    
    Args:
        thermal_data: DataFrame with generators to map
        network: PyPSA Network object
        
    Returns:
        Updated DataFrame with bus assignments for previously unmapped generators
    """
    if 'gsp' not in thermal_data.columns:
        logger.info("No GSP column in thermal data - skipping GSP-to-bus mapping")
        return thermal_data
    
    # Find generators without bus assignment
    needs_mapping = thermal_data['bus'].isna() if 'bus' in thermal_data.columns else pd.Series([True] * len(thermal_data), index=thermal_data.index)
    
    if needs_mapping.sum() == 0:
        logger.info("All generators already have bus assignments")
        return thermal_data
    
    logger.info(f"Applying GSP-to-bus mapping for {needs_mapping.sum()} generators without bus assignments")
    
    # Load GSP-to-bus mapping
    gsp_mapping = load_gsp_to_bus_mapping(network)
    
    if not gsp_mapping:
        logger.warning("No GSP-to-bus mapping available")
        return thermal_data
    
    # Initialize bus column if it doesn't exist
    if 'bus' not in thermal_data.columns:
        thermal_data['bus'] = pd.NA
    
    # Apply mapping
    mapped_count = 0
    unmapped_gsps = set()
    
    for idx in thermal_data[needs_mapping].index:
        gsp = thermal_data.loc[idx, 'gsp']
        if pd.isna(gsp):
            continue
        
        gsp_str = str(gsp).strip()
        
        # Try exact match first
        if gsp_str in gsp_mapping:
            thermal_data.loc[idx, 'bus'] = gsp_mapping[gsp_str]
            mapped_count += 1
        else:
            # Try fuzzy matching
            matches = difflib.get_close_matches(gsp_str, gsp_mapping.keys(), n=1, cutoff=0.7)
            if matches:
                thermal_data.loc[idx, 'bus'] = gsp_mapping[matches[0]]
                mapped_count += 1
                logger.debug(f"Fuzzy matched GSP '{gsp_str}' to '{matches[0]}' -> bus '{gsp_mapping[matches[0]]}'")
            else:
                unmapped_gsps.add(gsp_str)
    
    logger.info(f"GSP-to-bus mapping: {mapped_count} generators mapped")
    
    if unmapped_gsps:
        logger.warning(f"GSP-to-bus mapping: {len(unmapped_gsps)} unique GSPs could not be mapped:")
        for gsp in sorted(unmapped_gsps)[:10]:  # Show first 10
            logger.warning(f"  - {gsp}")
        if len(unmapped_gsps) > 10:
            logger.warning(f"  ... and {len(unmapped_gsps) - 10} more")
    
    return thermal_data


def get_solve_mode() -> str:
    """
    Load the solve_mode from config.yaml.
    
    Returns:
        "LP" or "MILP" - controls whether ramp limits are applied to thermal generators.
        Defaults to "LP" (no ramp limits) if not specified.
    """
    config_path = Path("config/config.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            optimization = config.get('optimization', {})
            solve_mode = optimization.get('solve_mode', 'LP')
            return solve_mode.upper()
        except Exception as e:
            logger.warning(f"Failed to read solve_mode from config: {e}, defaulting to LP")
            return 'LP'
    return 'LP'


def load_dukes_generators(dukes_path: str) -> pd.DataFrame:
    """
    Load DUKES generator data (historical fossil/nuclear thermal capacity).
    
    IMPORTANT: Filters out ALL renewables (variable + thermal) that are already
    included from REPD. DUKES only provides fossil/nuclear thermal here:
    - Natural Gas (CCGT, OCGT)
    - Coal
    - Nuclear
    - Oil/Diesel
    
    REPD is the authoritative source for all renewable generation to avoid overlap.
    
    Args:
        dukes_path: Path to DUKES_{year}_generators.csv
        
    Returns:
        DataFrame with DUKES fossil/nuclear thermal generators only
    """
    if not os.path.exists(dukes_path):
        logger.warning(f"DUKES file not found: {dukes_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(dukes_path)
    logger.info(f"Loaded {len(df)} generators from DUKES")
    logger.info(f"  Total capacity: {df['capacity_mw'].sum():.1f} MW")
    
    # CRITICAL: Filter out ALL renewables already in network from REPD
    # REPD is the authoritative source for all renewable generation (variable + thermal)
    # DUKES only provides fossil/nuclear thermal to avoid double-counting
    # 
    # Use lowercase patterns for case-insensitive matching (DUKES has inconsistent casing)
    excluded_fuel_type_patterns = [
        # Variable renewables (already from REPD with time series)
        'wind (onshore)', 'wind (offshore)', 'wind',
        'solar',
        'hydro',
        'hydro / pumped storage',
        'pumped storage',  # Also exclude pumped storage - it's handled as storage, not generators
        
        # Renewable thermal (already from REPD as dispatchable thermal)
        # These would be double-counted if we include them from DUKES
        'biomass',  # Match all biomass variants
        'waste',    # Match all waste variants (municipal, anaerobic, etc.)
    ]
    
    before_filter = len(df)
    # Case-insensitive matching using str.lower()
    df_fuel_lower = df['fuel_type'].str.lower()
    is_excluded = df_fuel_lower.apply(
        lambda x: any(pattern in x for pattern in excluded_fuel_type_patterns)
    )
    excluded_capacity = df[is_excluded]['capacity_mw'].sum()
    excluded_types = df[is_excluded]['fuel_type'].unique().tolist()
    df = df[~is_excluded]
    
    if before_filter > len(df):
        logger.info(f"  Filtered out {before_filter - len(df)} renewable generators ({excluded_capacity:.1f} MW)")
        logger.info(f"  Excluded fuel types: {excluded_types}")
        logger.info(f"  (All renewables come from REPD to avoid double-counting)")
        logger.info(f"  Remaining fossil/nuclear thermal: {len(df)}, {df['capacity_mw'].sum():.1f} MW")
    
    # Standardize column names to match expected format
    column_mapping = {}
    if 'station_name' not in df.columns and 'Station Name' in df.columns:
        column_mapping['Station Name'] = 'station_name'
    # fuel_type is already the column name in DUKES CSV, keep as-is for now
    # It will be mapped to 'fuel' later in the pipeline
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info(f"Applied DUKES column mapping: {column_mapping}")
    
    # Ensure data_source column exists
    if 'data_source' not in df.columns:
        df['data_source'] = 'DUKES'
    
    return df


def map_fes_technology_to_carrier(fes_tech: str) -> str:
    """
    Map FES technology names to PyPSA carrier names.
    
    Args:
        fes_tech: FES technology string
        
    Returns:
        PyPSA carrier name
    """
    # Technology mapping dictionary
    tech_map = {
        # Thermal generators
        'CCGTs (non CHP)': 'CCGT',
        'OCGTs (non CHP)': 'OCGT',
        'Nuclear': 'nuclear',
        'Coal': 'coal',
        'Hydrogen fuelled generation': 'H2',
        
        # CHP
        'Non-renewable CHP': 'CHP',
        'Micro CHP': 'micro_CHP',
        
        # Renewables (thermal)
        'Biomass & Energy Crops (including CHP)': 'biomass',
        'Waste Incineration (including CHP)': 'waste',
        'Renewable Engines (Landfill Gas, Sewage Gas, Biogas)': 'biogas',
        
        # Non-renewable engines
        'Non-renewable Engines (Diesel) (non CHP)': 'oil',
        'Non-renewable Engines (Gas) (non CHP)': 'gas_engine',
        
        # Other
        'Fuel Cells': 'fuel_cell',
        'Hydro': 'hydro',
        'Geothermal': 'geothermal',
        
        # Renewables (non-thermal - for completeness)
        'Solar Generation': 'solar',
        'Wind': 'onwind',
        'Offshore-Wind (off-Grid)': 'offwind',
        'Marine': 'marine',
        
        # Interconnector (handled separately)
        'Interconnector': 'interconnector',
    }
    
    return tech_map.get(fes_tech, 'other')


def load_fes_generators(fes_path: str, modelled_year: int, fes_scenario: str = None) -> pd.DataFrame:
    """
    Load FES generator data for specific year and scenario.
    
    FES data has pivot format with years as columns (2023-2050).
    This function extracts capacity for the specific modelled year.
    
    Args:
        fes_path: Path to FES_{year}_data.csv
        modelled_year: Year to extract capacity for (e.g., 2035)
        fes_scenario: FES pathway (e.g., 'Holistic Transition'). If None, uses first available.
        
    Returns:
        DataFrame with columns: technology, capacity_mw, fuel_type, gsp, data_source
    """
    if not os.path.exists(fes_path):
        logger.warning(f"FES file not found: {fes_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading FES generators for year {modelled_year}")
    logger.info(f"  FES file: {fes_path}")
    
    # Read FES data
    df = pd.read_csv(fes_path)
    
    # Filter to generation building blocks (have Technology field)
    gen_blocks = df[df['Technology'].notna()].copy()
    logger.info(f"  Found {len(gen_blocks)} generation records in FES")
    
    # Filter to actual generation technologies (exclude demand-side, storage, etc.)
    generation_technologies = [
        # Thermal generators
        'CCGTs (non CHP)', 'OCGTs (non CHP)', 'Nuclear', 'Coal',
        'Hydrogen fuelled generation',
        # CHP
        'Non-renewable CHP', 'Micro CHP',
        # Renewable thermal (dispatchable)
        'Biomass & Energy Crops (including CHP)',
        'Waste Incineration (including CHP)',
        'Renewable Engines (Landfill Gas, Sewage Gas, Biogas)',
        # Non-renewable engines
        'Non-renewable Engines (Diesel) (non CHP)',
        'Non-renewable Engines (Gas) (non CHP)',
        # Other dispatchable
        'Fuel Cells', 'Geothermal',
        # NOTE: Variable renewables (Solar, Wind, Offshore-Wind, Marine, Hydro) 
        # are now handled by Stage 1 (integrate_renewable_generators.py)
        # Do NOT include them here to avoid double-counting
    ]
    
    # Explicitly exclude variable renewables (handled in Stage 1)
    variable_renewables = [
        'Solar Generation', 'Wind', 'Offshore-Wind (off-Grid)', 'Marine', 'Hydro'
    ]
    
    # Filter to thermal/dispatchable technologies only (exclude variable renewables)
    gen_blocks = gen_blocks[gen_blocks['Technology'].isin(generation_technologies)].copy()
    gen_blocks = gen_blocks[~gen_blocks['Technology'].isin(variable_renewables)].copy()
    logger.info(f"  Filtered to {len(gen_blocks)} thermal/dispatchable generation records")
    logger.info(f"  (Variable renewables handled in Stage 1)")
    
    # Check available pathways
    available_pathways = gen_blocks['FES Pathway'].unique()
    logger.info(f"  Available FES pathways: {list(available_pathways)}")
    
    # Select FES scenario
    if fes_scenario is None:
        # Use first available pathway (usually 'Counterfactual' or similar)
        fes_scenario = available_pathways[0]
        logger.info(f"  No scenario specified, using: {fes_scenario}")
    else:
        if fes_scenario not in available_pathways:
            logger.warning(f"  Scenario '{fes_scenario}' not found. Available: {list(available_pathways)}")
            logger.warning(f"  Using first available: {available_pathways[0]}")
            fes_scenario = available_pathways[0]
        else:
            logger.info(f"  Using FES scenario: {fes_scenario}")
    
    # Filter to specific scenario
    gen_blocks = gen_blocks[gen_blocks['FES Pathway'] == fes_scenario].copy()
    logger.info(f"  Filtered to {len(gen_blocks)} records for scenario '{fes_scenario}'")
    
    # Check if year column exists
    year_col = str(modelled_year)
    if year_col not in df.columns:
        logger.error(f"Year {modelled_year} not in FES data. Available years: {[c for c in df.columns if c.isdigit()]}")
        return pd.DataFrame()
    
    # Extract generators with non-zero capacity for modelled year
    generators = []
    tech_summary = {}  # Track by technology
    
    for _, row in gen_blocks.iterrows():
        capacity = row[year_col]
        
        # Skip if capacity is missing or zero
        if pd.isna(capacity) or capacity <= 0:
            continue
        
        fes_tech = row['Technology']
        carrier = map_fes_technology_to_carrier(fes_tech)
        
        gen = {
            'technology': fes_tech,
            'capacity_mw': float(capacity),
            'fuel_type': carrier,
            'gsp': row.get('GSP', None),
            'data_source': 'FES'
        }
        generators.append(gen)
        
        # Track technology summary
        if fes_tech not in tech_summary:
            tech_summary[fes_tech] = {'count': 0, 'capacity': 0.0}
        tech_summary[fes_tech]['count'] += 1
        tech_summary[fes_tech]['capacity'] += float(capacity)
    
    # Create DataFrame
    fes_df = pd.DataFrame(generators)
    
    # Log results
    if len(fes_df) > 0:
        total_capacity = fes_df['capacity_mw'].sum()
        logger.info(f"Loaded {len(fes_df)} FES generators for {modelled_year}")
        logger.info(f"  Total capacity: {total_capacity:,.1f} MW")
        logger.info(f"  Technology breakdown (top 10):")
        
        # Sort by capacity and show top 10
        tech_sorted = sorted(tech_summary.items(), key=lambda x: x[1]['capacity'], reverse=True)
        for tech, stats in tech_sorted[:10]:
            logger.info(f"    {tech}: {stats['count']} units, {stats['capacity']:,.1f} MW")
    else:
        logger.warning(f"No FES generators found for year {modelled_year}")
    
    return fes_df


def load_repd_thermal_sites(site_files: dict) -> pd.DataFrame:
    """
    Load dispatchable renewable thermal sites from REPD.
    
    Args:
        site_files: Dictionary mapping fuel_type -> filepath
        
    Returns:
        Combined DataFrame of all REPD thermal sites
    """
    repd_sites = []
    
    for fuel_type, filepath in site_files.items():
        if not os.path.exists(filepath):
            logger.warning(f"REPD {fuel_type} file not found: {filepath}")
            continue
        
        df = pd.read_csv(filepath)
        if len(df) > 0:
            # Standardize columns
            if 'x_coord' in df.columns and 'y_coord' in df.columns:
                df['lon'] = pd.to_numeric(df['x_coord'], errors='coerce')
                df['lat'] = pd.to_numeric(df['y_coord'], errors='coerce')
            elif 'X-coordinate' in df.columns and 'Y-coordinate' in df.columns:
                df['lon'] = pd.to_numeric(df['X-coordinate'], errors='coerce')
                df['lat'] = pd.to_numeric(df['Y-coordinate'], errors='coerce')
            
            # Ensure capacity column
            if 'capacity_mw' not in df.columns:
                if 'Installed Capacity (MWelec)' in df.columns:
                    df['capacity_mw'] = pd.to_numeric(df['Installed Capacity (MWelec)'], errors='coerce')
            
            # Filter out sites with zero/null capacity (data quality issue)
            initial_count = len(df)
            df = df[df['capacity_mw'].notna() & (df['capacity_mw'] > 0)]
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                logger.info(f"  {fuel_type}: Filtered out {filtered_count} sites with zero/null capacity")
            
            df['fuel_type'] = fuel_type
            df['data_source'] = 'REPD'
            repd_sites.append(df)
            logger.info(f"  {fuel_type}: {len(df)} sites, {df['capacity_mw'].sum():.1f} MW")
    
    if repd_sites:
        combined = pd.concat(repd_sites, ignore_index=True)
        logger.info(f"Total REPD thermal sites: {len(combined)}, {combined['capacity_mw'].sum():.1f} MW")
        return combined
    else:
        logger.warning("No REPD thermal sites loaded")
        return pd.DataFrame()


def scale_repd_bioenergy_to_fes(repd_df: pd.DataFrame,
                                 modelled_year: int,
                                 fes_pathway: str = None,
                                 fes_data_path: str = None) -> pd.DataFrame:
    """
    Scale REPD bioenergy generators to match FES capacity targets for future scenarios.
    
    FES provides capacity targets for bioenergy technologies (waste, biogas, etc.)
    This function uniformly scales REPD sites within each technology category to
    match the FES total while preserving the spatial distribution of sites.
    
    FES Building Blocks mapped to REPD fuel types:
    - Gen_BB010 (Biomass) → biomass, advanced_biofuel
    - Gen_BB011 (Waste) → waste_to_energy  
    - Gen_BB004 (Biogas/Landfill) → biogas, landfill_gas, sewage_gas
    
    Args:
        repd_df: DataFrame with REPD bioenergy sites (fuel_type, capacity_mw)
        modelled_year: Target year (e.g., 2035, 2050)
        fes_pathway: FES pathway name (defaults to 'Holistic Transition')
        fes_data_path: Path to FES data CSV
        
    Returns:
        DataFrame with scaled capacity_mw column
    """
    if repd_df is None or len(repd_df) == 0:
        return repd_df
    
    if modelled_year is None or modelled_year <= 2024:
        logger.info("Historical scenario - not scaling REPD bioenergy to FES")
        return repd_df
    
    if fes_pathway is None:
        fes_pathway = 'Holistic Transition'  # Default pathway
    
    # Default FES data path
    if fes_data_path is None:
        fes_data_path = Path(__file__).parent.parent / "resources" / "FES" / "FES_2024_data.csv"
    
    if not Path(fes_data_path).exists():
        logger.warning(f"FES data file not found: {fes_data_path} - cannot scale REPD bioenergy")
        return repd_df
    
    try:
        # Load FES data
        fes_data = pd.read_csv(fes_data_path)
        
        # Filter to target pathway
        fes_data = fes_data[fes_data['FES Pathway'] == fes_pathway]
        
        if len(fes_data) == 0:
            logger.warning(f"No FES data for pathway '{fes_pathway}'")
            return repd_df
        
        # Get target year column
        year_col = str(modelled_year)
        if year_col not in fes_data.columns:
            logger.warning(f"Year {modelled_year} not in FES data columns")
            return repd_df
        
        # Define FES building block to REPD fuel type mapping
        fes_to_repd_mapping = {
            'Gen_BB010': ['biomass', 'advanced_biofuel'],           # Biomass
            'Gen_BB011': ['waste_to_energy'],                        # Waste  
            'Gen_BB004': ['biogas', 'landfill_gas', 'sewage_gas'],   # Biogas/Landfill/Sewage
        }
        
        logger.info(f"=== Scaling REPD Bioenergy to FES {fes_pathway} {modelled_year} ===")
        
        scaled_df = repd_df.copy()
        scaled_df['capacity_mw_original'] = scaled_df['capacity_mw']
        
        total_original = scaled_df['capacity_mw'].sum()
        
        for bb_id, repd_fuel_types in fes_to_repd_mapping.items():
            # Get FES target for this building block
            bb_fes = fes_data[fes_data['Building Block ID Number'] == bb_id]
            if len(bb_fes) == 0:
                logger.debug(f"  No FES data for {bb_id}")
                continue
            
            fes_target_mw = bb_fes[year_col].sum()
            
            # Find matching REPD sites
            mask = scaled_df['fuel_type'].str.lower().isin([ft.lower() for ft in repd_fuel_types])
            repd_current_mw = scaled_df.loc[mask, 'capacity_mw'].sum()
            
            if repd_current_mw <= 0:
                logger.debug(f"  {bb_id}: No REPD sites for {repd_fuel_types}")
                continue
            
            # Calculate and apply scale factor
            scale_factor = fes_target_mw / repd_current_mw
            
            scaled_df.loc[mask, 'capacity_mw'] = scaled_df.loc[mask, 'capacity_mw'] * scale_factor
            
            logger.info(f"  {bb_id} ({', '.join(repd_fuel_types)}):")
            logger.info(f"    REPD current: {repd_current_mw:.0f} MW → FES target: {fes_target_mw:.0f} MW (scale: {scale_factor:.2f}x)")
        
        total_scaled = scaled_df['capacity_mw'].sum()
        logger.info(f"  Total REPD bioenergy: {total_original:.0f} MW → {total_scaled:.0f} MW")
        
        return scaled_df
        
    except Exception as e:
        logger.error(f"Error scaling REPD bioenergy to FES: {e}")
        return repd_df


def merge_generator_data_sources(dukes_df: pd.DataFrame, 
                                   repd_df: pd.DataFrame, 
                                   fes_df: pd.DataFrame,
                                   scenario_year: int,
                                   fes_data_path: str = None,
                                   fes_pathway: str = None) -> pd.DataFrame:
    """
    Merge generator data from multiple sources with priority handling.
    
    Priority order:
      - Historical scenarios: DUKES (primary) > REPD (secondary), NO FES
      - Future scenarios: FES primary, REPD bioenergy SCALED to FES targets
    
    Args:
        dukes_df: DUKES thermal generators (historical only)
        repd_df: REPD dispatchable renewable thermal
        fes_df: FES generators (future scenarios only)
        scenario_year: Year being modeled
        fes_data_path: Path to FES data CSV for scaling REPD to FES targets
        fes_pathway: FES pathway name (e.g., 'Holistic Transition')
        
    Returns:
        Merged DataFrame with all generators and data_source tracking
    """
    logger.info("Merging generator data sources")
    logger.info(f"  DUKES: {len(dukes_df)} generators")
    logger.info(f"  REPD: {len(repd_df)} generators")
    logger.info(f"  FES: {len(fes_df)} generators")
    
    # Combine all sources
    all_sources = []
    
    is_future_scenario = len(fes_df) > 0 and len(dukes_df) == 0
    
    if len(dukes_df) > 0:
        all_sources.append(dukes_df)
        logger.info("  Using DUKES as primary thermal source (historical scenario)")
    
    if len(repd_df) > 0:
        if is_future_scenario:
            # For future scenarios, scale REPD bioenergy to FES targets
            repd_df = scale_repd_bioenergy_to_fes(
                repd_df, 
                scenario_year,
                fes_pathway=fes_pathway,
                fes_data_path=fes_data_path
            )
        all_sources.append(repd_df)
        logger.info("  Using REPD for dispatchable renewable thermal")
    
    if len(fes_df) > 0:
        all_sources.append(fes_df)
        logger.info("  Using FES (future scenario)")
    
    if not all_sources:
        logger.error("No generator data sources available!")
        return pd.DataFrame()
    
    merged = pd.concat(all_sources, ignore_index=True)
    
    # Log data source breakdown
    source_summary = merged.groupby('data_source').agg({
        'capacity_mw': ['count', 'sum']
    }).round(1)
    
    logger.info("\nData source summary:")
    for source in source_summary.index:
        count = source_summary.loc[source, ('capacity_mw', 'count')]
        capacity = source_summary.loc[source, ('capacity_mw', 'sum')]
        logger.info(f"  {source}: {int(count)} generators, {capacity:.1f} MW")
    
    return merged


def standardize_repd_for_merge(repd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize REPD thermal data to match DUKES/FES format.
    
    Args:
        repd_df: Raw REPD thermal sites
        
    Returns:
        Standardized DataFrame
    """
    if len(repd_df) == 0:
        return repd_df
    
    # Map REPD columns to standard format
    standard_df = repd_df.copy()
    
    # Ensure standard column names exist
    if 'station_name' not in standard_df.columns and 'Site Name' in standard_df.columns:
        standard_df['station_name'] = standard_df['Site Name']
    
    if 'technology' not in standard_df.columns:
        standard_df['technology'] = standard_df['fuel_type']
    
    return standard_df


def add_thermal_generators(network: pypsa.Network, thermal_data: pd.DataFrame, fuel_data_path: str = None, solve_mode: str = None) -> pypsa.Network:
    """
    Add thermal generators from a structured DataFrame to the network.
    
    Expects thermal_data to have columns:
    - fuel_type: carrier name (e.g., 'CCGT', 'Coal', 'Nuclear', 'biomass', etc.)
    - capacity_mw: installed capacity in MW
    - station_name: generator name (optional, will be auto-generated if missing)
    - gsp or bus: bus/location mapping (optional)
    - efficiency: generator efficiency (optional, defaults to 0.5)
    - marginal_cost: operating cost (optional, defaults to 0)
    
    Ramp limits are controlled by solve_mode:
    - LP: No ramp limits (ramp_limit_up/down = None)
    - MILP: Apply ramp limits from fuel characteristics
    
    Args:
        network: PyPSA Network object
        thermal_data: DataFrame with thermal generator data
        fuel_data_path: Optional path to fuel characteristics CSV
        solve_mode: "LP" (no ramp limits) or "MILP" (with ramp limits). If None, reads from config.
        
    Returns:
        Updated network with thermal generators added
    """
    # Determine solve mode
    if solve_mode is None:
        solve_mode = get_solve_mode()
    solve_mode = solve_mode.upper()
    
    apply_ramp_limits = (solve_mode == 'MILP')
    logger.info(f"Solve mode: {solve_mode} - Ramp limits: {'ENABLED' if apply_ramp_limits else 'DISABLED'}")
    
    if len(thermal_data) == 0:
        logger.warning("No thermal generators to add")
        return network
    
    # Load fuel characteristics if provided
    fuel_chars = {}
    if fuel_data_path and os.path.exists(fuel_data_path):
        try:
            fuel_df = pd.read_csv(fuel_data_path, index_col='fuel')
            fuel_chars = fuel_df.to_dict('index')
            logger.info(f"Loaded fuel characteristics for {len(fuel_chars)} fuel types")
        except Exception as e:
            logger.warning(f"Could not load fuel characteristics: {e}")
    
    # Ensure required columns exist
    required_cols = ['fuel_type', 'capacity_mw']
    for col in required_cols:
        if col not in thermal_data.columns:
            raise ValueError(f"thermal_data missing required column: {col}")
    
    generators_added = 0
    
    # Build carrier normalization helper using known carrier definitions
    carriers_df = get_carrier_definitions()
    carrier_names = list(carriers_df.index)

    def _normalize_fuel_to_carrier(fuel: str) -> str:
        if pd.isna(fuel):
            return 'unclassified'
        f = str(fuel).strip()
        if not f:
            return 'unclassified'
        # Common manual mappings covering all DUKES fuel type variants (2010-2024)
        manual_map = {
            # Gas-fired generation
            'natural gas': 'CCGT',
            'gas': 'CCGT',
            'ccgt': 'CCGT',
            'combined cycle gas turbine': 'CCGT',
            'sour gas': 'CCGT',              # Connahs Quay - sour gas CCGT plant
            'ocgt': 'OCGT',
            'open cycle gas turbine': 'OCGT',
            'single cycle': 'OCGT',          # DUKES 2022+ uses 'Single cycle' for OCGT
            # Coal
            'coal': 'coal',
            'coal (steam)': 'conventional_steam',
            'coal / oil': 'coal',            # Coal stations with oil backup/start-up
            # Nuclear
            'nuclear': 'nuclear',
            # Bioenergy / waste
            'biomass': 'biomass',
            'waste': 'waste_to_energy',
            'msw': 'waste_to_energy',        # Municipal Solid Waste (DUKES 2022+)
            'municipal solid waste': 'waste_to_energy',
            'meat & bone meal': 'biomass',   # Animal waste bioenergy
            'straw': 'biomass',              # Agricultural bioenergy
            'landfill gas': 'landfill_gas',
            'sewage gas': 'sewage_gas',
            'biogas': 'biogas',
            # Oil / diesel
            'oil': 'oil',
            'diesel': 'oil',
            'diesel/gas oil': 'oil',         # Peaking/backup diesel generators (DUKES 2022+)
            'gas oil': 'oil',                # Peaking/backup diesel generators (DUKES 2010-2018)
            'gas oil / kerosene': 'oil',     # Backup oil generators
            'gas / oil': 'oil',              # Mixed fuel, primarily oil
            'light oil': 'oil',              # Oil-fired generators
            # Hydro
            'hydro': 'Hydro',
        }
        key = f.lower()
        if key in manual_map:
            return manual_map[key]
        # Exact (case-insensitive) match to known carriers
        for cname in carrier_names:
            if cname.lower() == key:
                return cname
        # Fuzzy match as a last resort
        match = difflib.get_close_matches(f, carrier_names, n=1, cutoff=0.8)
        if match:
            return match[0]
        # Fallback to 'unclassified' carrier
        return 'unclassified'

    idx_count = 0
    for idx, row in thermal_data.iterrows():
        raw_fuel = row['fuel_type']
        fuel_type = _normalize_fuel_to_carrier(raw_fuel)
        capacity_mw = float(row['capacity_mw'])
        
        # Skip zero/negative capacity
        if capacity_mw <= 0:
            continue
        
        # Generate generator name if not provided
        gen_name = row.get('station_name', f"gen_{fuel_type}_{idx}")
        gen_name = str(gen_name).strip()
        if not gen_name or gen_name == 'nan':
            gen_name = f"gen_{fuel_type}_{idx}"
        
        # Determine bus location
        if 'bus' in row and pd.notna(row['bus']):
            bus = str(row['bus']).strip()
        elif 'gsp' in row and pd.notna(row['gsp']):
            bus = str(row['gsp']).strip()
        else:
            # Default to first bus if available
            if len(network.buses) == 0:
                logger.warning(f"Network has no buses, skipping generator {gen_name}")
                continue
            bus = network.buses.index[0]
        
        # Verify bus exists in network - use fallback if not found
        if bus not in network.buses.index:
            # Try to find a similar bus name (fuzzy match)
            bus_lower = bus.lower().replace(' ', '').replace('_', '').replace('-', '')
            matched_bus = None
            for network_bus in network.buses.index:
                network_bus_lower = network_bus.lower().replace(' ', '').replace('_', '').replace('-', '')
                if bus_lower in network_bus_lower or network_bus_lower in bus_lower:
                    matched_bus = network_bus
                    break
            
            if matched_bus:
                logger.debug(f"Bus '{bus}' matched to '{matched_bus}' for generator {gen_name}")
                bus = matched_bus
            else:
                # Fallback: use coordinates if available, otherwise use a major bus
                if 'lat' in row and 'lon' in row and pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                    # Find nearest bus by coordinates
                    lat, lon = float(row['lat']), float(row['lon'])
                    if 'x' in network.buses.columns and 'y' in network.buses.columns:
                        distances = ((network.buses['x'] - lon)**2 + (network.buses['y'] - lat)**2)**0.5
                        bus = distances.idxmin()
                        logger.debug(f"Bus '{row.get('bus', 'unknown')}' not found, mapped {gen_name} to nearest bus {bus}")
                else:
                    # Use first bus as last resort
                    bus = network.buses.index[0]
                    logger.debug(f"Bus '{row.get('bus', 'unknown')}' not found, using default bus {bus} for {gen_name}")
        
        # Get generator attributes from fuel characteristics or defaults
        # Ramp limits depend on solve_mode: LP = no ramps, MILP = with ramps
        gen_attrs = {
            'bus': bus,
            'carrier': fuel_type,
            'p_nom': capacity_mw,
            'efficiency': 0.5,
            'marginal_cost': 0.0,
            'committable': False,
            'min_up_time': 0,
            'min_down_time': 0,
        }
        
        # Only add ramp limits in MILP mode
        if apply_ramp_limits:
            gen_attrs['ramp_limit_up'] = 1.0  # Default: can ramp 100% of capacity per hour
            gen_attrs['ramp_limit_down'] = 1.0
        # In LP mode, leave ramp limits as None (no constraint)
        
        # Override with fuel characteristics if available
        if fuel_type in fuel_chars:
            fuel_data = fuel_chars[fuel_type]
            if 'efficiency' in fuel_data:
                try:
                    gen_attrs['efficiency'] = float(fuel_data['efficiency'])
                except (ValueError, TypeError):
                    pass
            if 'marginal_cost' in fuel_data:
                try:
                    gen_attrs['marginal_cost'] = float(fuel_data['marginal_cost'])
                except (ValueError, TypeError):
                    pass
            if 'committable' in fuel_data:
                # Only enable committable in MILP mode
                gen_attrs['committable'] = bool(fuel_data['committable']) if apply_ramp_limits else False
            if apply_ramp_limits:
                # Only set unit commitment parameters in MILP mode
                if 'min_up_time' in fuel_data:
                    try:
                        gen_attrs['min_up_time'] = int(fuel_data['min_up_time'])
                    except (ValueError, TypeError):
                        pass
                if 'min_down_time' in fuel_data:
                    try:
                        gen_attrs['min_down_time'] = int(fuel_data['min_down_time'])
                    except (ValueError, TypeError):
                        pass
                # Apply ramp limits from fuel data if available
                if 'ramp_limit_up' in fuel_data:
                    try:
                        ramp_up = float(fuel_data['ramp_limit_up'])
                        # Convert from %/hr to fraction of p_nom
                        gen_attrs['ramp_limit_up'] = ramp_up / 100.0 if ramp_up <= 100 else 1.0
                    except (ValueError, TypeError):
                        pass
                if 'ramp_limit_down' in fuel_data:
                    try:
                        ramp_down = float(fuel_data['ramp_limit_down'])
                        gen_attrs['ramp_limit_down'] = ramp_down / 100.0 if ramp_down <= 100 else 1.0
                    except (ValueError, TypeError):
                        pass
        
        # Add data source if available
        if 'data_source' in row:
            gen_attrs['data_source'] = str(row['data_source'])
        
        # Add coordinates if available (needed for plotting and validation)
        if 'lat' in row and pd.notna(row['lat']):
            gen_attrs['lat'] = float(row['lat'])
        if 'lon' in row and pd.notna(row['lon']):
            gen_attrs['lon'] = float(row['lon'])
        
        try:
            network.add("Generator", gen_name, **gen_attrs)
            generators_added += 1
            idx_count += 1
        except Exception as e:
            logger.warning(f"Failed to add generator {gen_name}: {e}")
            continue
    
    logger.info(f"Added {generators_added} thermal generators to network")
    return network


def main():
    """Main execution function for thermal generator integration."""
    global logger
    start_time = time.time()
    stage_times = {}  # Track timing for each stage
    
    # Reinitialize logger with Snakemake log path if available
    snk = globals().get('snakemake')
    if snk and hasattr(snk, 'log') and snk.log:
        logger = setup_logging(snk.log[0])
    
    logger.info("=" * 80)
    logger.info("THERMAL GENERATOR INTEGRATION (HYBRID DATA SOURCES)")
    logger.info("=" * 80)
    logger.info("Adding dispatchable thermal generators to network")
    
    try:
        # Access snakemake variables
        if not snk:
            raise RuntimeError("This script must be run via Snakemake")
        
        # =====================================================================
        # STAGE 1: LOAD NETWORK
        # =====================================================================
        stage_start = time.time()
        network_path = snk.input.network
        logger.info(f"Loading network from {network_path}")
        network = load_network(network_path, custom_logger=logger)
        logger.info("Input network (with renewables)")
        log_network_info(network, logger)
        
        initial_gen_count = len(network.generators)
        initial_capacity = network.generators['p_nom'].sum() if len(network.generators) > 0 else 0
        stage_times['1. Load network'] = time.time() - stage_start
        
        # Determine scenario type and data sources
        scenario_name = snk.wildcards.scenario
        scenario_config = snk.params.scenario_config
        modelled_year = scenario_config.get('modelled_year', 2020)
        fes_scenario = scenario_config.get('FES_scenario', None)  # FES pathway from config
        
        logger.info(f"Scenario: {scenario_name}, Modelled Year: {modelled_year}")
        if fes_scenario:
            logger.info(f"FES Scenario/Pathway: {fes_scenario}")
        
        # =====================================================================
        # STAGE 2: LOAD THERMAL DATA FROM MULTIPLE SOURCES
        # =====================================================================
        stage_start = time.time()
        # =====================================================================
        logger.info("-" * 80)
        logger.info("PART 1: LOADING THERMAL GENERATOR DATA")
        logger.info("-" * 80)
        
        # Load DUKES data (if historical scenario)
        dukes_df = pd.DataFrame()
        if hasattr(snk.input, 'dukes_data'):
            logger.info("Historical scenario detected - loading DUKES data")
            dukes_df = load_dukes_generators(snk.input.dukes_data)
        else:
            logger.info("Future scenario - DUKES data not applicable")
        
        # Load FES data (future scenarios only)
        fes_df = pd.DataFrame()
        if hasattr(snk.input, 'fes_data'):
            if len(dukes_df) == 0:
                # Future scenario - load FES as primary source
                logger.info("Future scenario - loading FES data as primary source")
                fes_df = load_fes_generators(snk.input.fes_data, modelled_year, fes_scenario)
            else:
                # Historical scenario - NO FES fallback
                logger.info("Historical scenario - NOT loading FES (using DUKES + REPD only)")
        else:
            if len(dukes_df) == 0:
                logger.warning("No thermal generator data available (no DUKES or FES)")
        
        # Load REPD dispatchable renewable thermal
        repd_df = pd.DataFrame()
        if hasattr(snk.input, 'biomass_sites'):
            logger.info("Loading REPD dispatchable renewable thermal sites")
            repd_site_files = {
                'biomass': snk.input.biomass_sites,
                'waste_to_energy': snk.input.waste_to_energy_sites,
                'biogas': snk.input.biogas_sites,
                'landfill_gas': snk.input.landfill_gas_sites,
                'sewage_gas': snk.input.sewage_gas_sites,
                'advanced_biofuel': snk.input.advanced_biofuel_sites,
                'geothermal': snk.input.geothermal_sites
            }
            repd_df = load_repd_thermal_sites(repd_site_files)
            repd_df = standardize_repd_for_merge(repd_df)
        else:
            logger.info("REPD thermal sites not available for this scenario")
        
        # =====================================================================
        # PART 2: MERGE DATA SOURCES WITH PRIORITY HANDLING
        # =====================================================================
        logger.info("-" * 80)
        logger.info("PART 2: MERGING DATA SOURCES")
        if len(dukes_df) > 0:
            logger.info("Historical scenario: DUKES > REPD (NO FES)")
        else:
            logger.info("Future scenario: FES + REPD (scaled to FES targets)")
        logger.info("-" * 80)
        
        # Get FES data path for REPD scaling (if available)
        fes_data_path = getattr(snk.input, 'fes_data', None)
        
        thermal_data = merge_generator_data_sources(
            dukes_df, repd_df, fes_df, modelled_year,
            fes_data_path=fes_data_path,
            fes_pathway=fes_scenario
        )
        
        if len(thermal_data) == 0:
            logger.error("No thermal generator data available after merging!")
            raise ValueError("Cannot proceed without thermal generator data")
        
        log_dataframe_info(thermal_data, logger, "Merged thermal data")
        stage_times['2. Load thermal data'] = time.time() - stage_start
        
        # =====================================================================
        # STAGE 2.5: DISTRIBUTE FES "DIRECT" GENERATORS SPATIALLY
        # =====================================================================
        # For FES generators connected "Direct" to transmission, distribute them
        # across buses based on historical spatial patterns of each carrier type
        stage_start_spatial = time.time()
        
        is_future_scenario = len(fes_df) > 0 and len(dukes_df) == 0
        if is_future_scenario:
            logger.info("-" * 80)
            logger.info("PART 2.5: DISTRIBUTING FES 'DIRECT' GENERATORS SPATIALLY")
            logger.info("-" * 80)
            logger.info("Using historical spatial patterns to distribute Direct-connected capacity")
            
            thermal_data = distribute_fes_generators_spatially(
                thermal_data, 
                network,
                repd_sites=repd_df if len(repd_df) > 0 else None
            )
            stage_times['2.5. Spatial distribution'] = time.time() - stage_start_spatial
        
        # =====================================================================
        # STAGE 3: MAP TO NETWORK BUSES
        # =====================================================================
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("PART 3: MAPPING THERMAL GENERATORS TO NETWORK BUSES")
        logger.info("-" * 80)
        
        # Detect network coordinate system (OSGB36 vs WGS84)
        # OSGB36: x values are in meters (typically 100000-700000 for UK)
        # WGS84: x values are longitude in degrees (typically -10 to 2 for UK)
        bus_x_max = network.buses.x.max()
        network_is_osgb36 = bus_x_max > 180  # WGS84 longitude is always <= 180
        
        if network_is_osgb36:
            logger.info(f"Network uses OSGB36 coordinates (x_max={bus_x_max:.1f})")
        else:
            logger.info(f"Network uses WGS84 coordinates (x_max={bus_x_max:.2f})")
        
        # Convert generator coordinates to match network coordinate system
        if network_is_osgb36:
            # Network is in OSGB36 - need to use or convert to OSGB36 coordinates
            if 'x_coord' in thermal_data.columns and 'y_coord' in thermal_data.columns:
                # Already have OSGB36 coordinates
                logger.info("Using existing OSGB36 coordinates (x_coord, y_coord)")
            elif 'lat' in thermal_data.columns and 'lon' in thermal_data.columns:
                # Convert WGS84 (lat/lon) to OSGB36 (x_coord/y_coord)
                logger.info("Converting generator lat/lon (WGS84) to x_coord/y_coord (OSGB36)")
                try:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
                    
                    lons = pd.to_numeric(thermal_data['lon'], errors='coerce')
                    lats = pd.to_numeric(thermal_data['lat'], errors='coerce')
                    valid_coords = lons.notna() & lats.notna()
                    
                    thermal_data['x_coord'] = pd.NA
                    thermal_data['y_coord'] = pd.NA
                    
                    if valid_coords.sum() > 0:
                        x, y = transformer.transform(lons[valid_coords].values, lats[valid_coords].values)
                        thermal_data.loc[valid_coords, 'x_coord'] = x
                        thermal_data.loc[valid_coords, 'y_coord'] = y
                        logger.info(f"Converted {valid_coords.sum()} generators from WGS84 to OSGB36")
                except ImportError:
                    logger.error("pyproj not available - cannot convert coordinates")
                    raise
            
            # Now map using OSGB36 coordinates
            if 'x_coord' in thermal_data.columns and 'y_coord' in thermal_data.columns:
                valid_coords = thermal_data['x_coord'].notna() & thermal_data['y_coord'].notna()
                thermal_with_coords = thermal_data[valid_coords].copy()
                thermal_without_coords = thermal_data[~valid_coords].copy()
                
                logger.info(f"Thermal generators: {len(thermal_with_coords)} with coordinates, {len(thermal_without_coords)} without")
                
                if len(thermal_with_coords) > 0:
                    thermal_with_coords = map_sites_to_buses(
                        network, 
                        thermal_with_coords,
                        method='nearest',
                        lat_col='y_coord',  # y_coord = northing
                        lon_col='x_coord',  # x_coord = easting
                        max_distance_km=150.0
                    )
                    
                    mapped_count = thermal_with_coords['bus'].notna().sum()
                    logger.info(f"Mapped {mapped_count}/{len(thermal_with_coords)} thermal generators to buses")
                
                # Recombine - preserve original indices to maintain bus mappings
                thermal_data = pd.concat([thermal_with_coords, thermal_without_coords], ignore_index=False).sort_index()
            else:
                logger.warning("No valid OSGB36 coordinates available for bus mapping")
        else:
            # Network is in WGS84 - use lat/lon coordinates directly
            if 'lat' in thermal_data.columns and 'lon' in thermal_data.columns:
                valid_coords = thermal_data['lat'].notna() & thermal_data['lon'].notna()
                thermal_with_coords = thermal_data[valid_coords].copy()
                thermal_without_coords = thermal_data[~valid_coords].copy()
                
                logger.info(f"Thermal generators: {len(thermal_with_coords)} with coordinates, {len(thermal_without_coords)} without")
                
                if len(thermal_with_coords) > 0:
                    thermal_with_coords = map_sites_to_buses(
                        network, 
                        thermal_with_coords,
                        method='nearest',
                        lat_col='lat',
                        lon_col='lon',
                        max_distance_km=150.0
                    )
                    
                    mapped_count = thermal_with_coords['bus'].notna().sum()
                    logger.info(f"Mapped {mapped_count}/{len(thermal_with_coords)} thermal generators to buses")
                
                # Recombine - preserve original indices to maintain bus mappings
                thermal_data = pd.concat([thermal_with_coords, thermal_without_coords], ignore_index=False).sort_index()
            elif 'x_coord' in thermal_data.columns and 'y_coord' in thermal_data.columns:
                # Convert OSGB36 (x_coord, y_coord) to WGS84 (lon, lat) for bus mapping
                logger.info("Converting OSGB36 coordinates to WGS84 for bus mapping")
                try:
                    from pyproj import Transformer
                    # EPSG:27700 = OSGB36 (British National Grid), EPSG:4326 = WGS84
                    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                    
                    # Convert coordinates
                    x_coords = pd.to_numeric(thermal_data['x_coord'], errors='coerce')
                    y_coords = pd.to_numeric(thermal_data['y_coord'], errors='coerce')
                    
                    valid_coords = x_coords.notna() & y_coords.notna()
                    
                    # Initialize lon/lat columns
                    thermal_data['lon'] = pd.NA
                    thermal_data['lat'] = pd.NA
                    
                    # Convert valid coordinates
                    if valid_coords.sum() > 0:
                        lon_wgs84, lat_wgs84 = transformer.transform(
                            x_coords[valid_coords].values,
                            y_coords[valid_coords].values
                        )
                        thermal_data.loc[valid_coords, 'lon'] = lon_wgs84
                        thermal_data.loc[valid_coords, 'lat'] = lat_wgs84
                        logger.info(f"Converted {valid_coords.sum()} OSGB36 coordinates to WGS84")
                    
                except ImportError:
                    logger.warning("pyproj not available - using raw x/y coordinates as lon/lat (may cause incorrect bus mapping)")
                    thermal_data['lon'] = pd.to_numeric(thermal_data['x_coord'], errors='coerce')
                    thermal_data['lat'] = pd.to_numeric(thermal_data['y_coord'], errors='coerce')
                
                valid_coords = thermal_data['lat'].notna() & thermal_data['lon'].notna()
                thermal_with_coords = thermal_data[valid_coords].copy()
                thermal_without_coords = thermal_data[~valid_coords].copy()
                
                logger.info(f"Thermal generators: {len(thermal_with_coords)} with coordinates, {len(thermal_without_coords)} without")
                
                if len(thermal_with_coords) > 0:
                    thermal_with_coords = map_sites_to_buses(
                        network, 
                        thermal_with_coords,
                        method='nearest',
                        lat_col='lat',
                        lon_col='lon',
                        max_distance_km=150.0
                    )
                    
                    mapped_count = thermal_with_coords['bus'].notna().sum()
                    logger.info(f"Mapped {mapped_count}/{len(thermal_with_coords)} thermal generators to buses")
                
                # Recombine - preserve original indices to maintain bus mappings
                thermal_data = pd.concat([thermal_with_coords, thermal_without_coords], ignore_index=False).sort_index()
            else:
                logger.warning("No coordinate columns found - thermal generators will need location matching")
        stage_times['3. Map to buses'] = time.time() - stage_start
        
        # =====================================================================
        # STAGE 3.25: APPLY GSP-TO-BUS MAPPING FOR FES GENERATORS
        # =====================================================================
        # For FES generators without coordinates (aggregated thermal capacity),
        # map GSP names to network buses using the GSP lookup file
        stage_start_gsp = time.time()
        logger.info("-" * 80)
        logger.info("PART 3.25: APPLYING GSP-TO-BUS MAPPING FOR FES GENERATORS")
        logger.info("-" * 80)
        
        thermal_data = apply_gsp_to_bus_mapping(thermal_data, network)
        stage_times['3.25. GSP-to-bus mapping'] = time.time() - stage_start_gsp
        
        # =====================================================================
        # STAGE 3.5: APPLY ETYS BMU-TO-NODE MAPPING (ETYS NETWORKS ONLY)
        # =====================================================================
        # For ETYS networks, correct generator bus assignments using official data
        # This moves large generators from 132kV to their correct 400kV connection points
        network_model = scenario_config.get('network_model', 'reduced')
        
        if network_model.upper() == 'ETYS':
            stage_start_bmu = time.time()
            logger.info("-" * 80)
            logger.info("PART 3.5: APPLYING ETYS BMU-TO-NODE BUS CORRECTIONS")
            logger.info("-" * 80)
            logger.info("Large generators should connect to 400kV buses (per GB_network.xlsx Dir_con_BMUs_to_node)")
            logger.info("Spatial nearest-distance mapping may have assigned them to nearby 132kV buses")
            
            thermal_data = apply_etys_bmu_mapping(thermal_data, network)
            
            stage_times['3.5. ETYS BMU mapping'] = time.time() - stage_start_bmu
        
        # =====================================================================
        # STAGE 4: ADD THERMAL GENERATORS TO NETWORK
        # =====================================================================
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("PART 4: ADDING THERMAL GENERATORS TO NETWORK")
        logger.info("-" * 80)
        
        # Filter out pumped storage (should be added as storage units, not generators)
        pumped_storage_carriers = ['Pumped Storage', 'Hydro / pumped storage', 'pumped_hydro']
        before_filter = len(thermal_data)
        thermal_data = thermal_data[~thermal_data['fuel_type'].isin(pumped_storage_carriers)]
        if 'carrier' in thermal_data.columns:
            thermal_data = thermal_data[~thermal_data['carrier'].isin(pumped_storage_carriers)]
        pumped_filtered = before_filter - len(thermal_data)
        if pumped_filtered > 0:
            logger.info(f"Filtered out {pumped_filtered} pumped storage units (should be added as storage, not generators)")
        
        # Filter out Northern Ireland generators (PyPSA-GB models GB only, not UK)
        # NI is approximately: lat 54.0-55.5, lon -8.0 to -5.5
        before_ni_filter = len(thermal_data)
        ni_filtered_count = 0
        ni_capacity = 0.0
        
        # Method 1: Filter by location column (DUKES data)
        if 'location' in thermal_data.columns:
            ni_mask = thermal_data['location'].str.contains('Northern Ireland', case=False, na=False)
            ni_capacity += thermal_data[ni_mask]['capacity_mw'].sum()
            thermal_data = thermal_data[~ni_mask]
            ni_filtered_count += ni_mask.sum()
        
        # Method 2: Filter by coordinates (REPD data and any others with lat/lon)
        if 'lat' in thermal_data.columns and 'lon' in thermal_data.columns:
            ni_coords_mask = (
                (thermal_data['lat'] > 54.0) & 
                (thermal_data['lat'] < 55.5) & 
                (thermal_data['lon'] < -5.5) & 
                (thermal_data['lon'] > -8.0)
            )
            ni_capacity += thermal_data[ni_coords_mask]['capacity_mw'].sum()
            thermal_data = thermal_data[~ni_coords_mask]
            ni_filtered_count += ni_coords_mask.sum()
        
        if ni_filtered_count > 0:
            logger.info(f"Filtered out {ni_filtered_count} Northern Ireland generators ({ni_capacity:.1f} MW)")
            logger.info(f"  (PyPSA-GB models Great Britain only, not the full UK)")
        
        thermal_before = len(network.generators)
        network = add_thermal_generators(
            network,
            thermal_data,
            fuel_data_path=str(snk.input.fuel_data)
        )
        thermal_added = len(network.generators) - thermal_before
        logger.info(f"Added {thermal_added} thermal generators to network")
        stage_times['4. Add generators'] = time.time() - stage_start
        
        # Ensure carrier definitions include any new carriers used by thermal generators
        logger.info("Adding/updating carrier definitions for thermal generators")
        network = add_carriers_to_network(network, logger)
        stage_times['4b. Add carriers for thermal'] = 0.1
        
        # =====================================================================
        # STAGE 4b.5: NOTE ON RENEWABLE PROFILES (NOW HANDLED IN STAGE 1)
        # =====================================================================
        # Variable renewables (Solar, Wind, Hydro, Marine) are now processed in
        # Stage 1 (integrate_renewable_generators.py) with their atlite profiles.
        # This stage only handles thermal/dispatchable generators.
        # 
        # The apply_renewable_profiles_to_fes function is kept for backwards
        # compatibility but should not be needed for new workflows.
        stage_start_profiles = time.time()
        
        is_future_scenario = len(fes_df) > 0 and len(dukes_df) == 0
        if is_future_scenario:
            logger.info("-" * 80)
            logger.info("PART 4b.5: CHECKING FOR FES RENEWABLES (NOW IN STAGE 1)")
            logger.info("-" * 80)
            
            # FES renewable carriers (should now be handled in Stage 1)
            fes_renewable_carriers = ['Solar', 'solar', 'Wind', 'wind', 'onwind', 
                                       'offwind', 'Hydro', 'hydro', 'Marine', 'marine',
                                       'solar_pv', 'wind_onshore', 'wind_offshore',
                                       'large_hydro', 'small_hydro']
            
            # Check if any renewables were added here (shouldn't happen in new workflow)
            fes_renewables_in_network = []
            for carrier in fes_renewable_carriers:
                carrier_gens = network.generators[network.generators['carrier'] == carrier]
                if len(carrier_gens) > 0:
                    fes_renewables_in_network.append(carrier)
            
            if fes_renewables_in_network:
                logger.info(f"  Found {len(fes_renewables_in_network)} renewable carrier types in network")
                logger.info(f"  These should have been processed in Stage 1 with profiles applied")
                # Get renewables_year for fallback profile application
                renewables_year = scenario_config.get('renewables_year', 2020)
                
                # Apply profiles as fallback (in case Stage 1 missed any)
                network = apply_renewable_profiles_to_fes(
                    network,
                    fes_carriers=fes_renewables_in_network,
                    renewables_year=renewables_year
                )
                stage_times['4b.5. FES renewable profiles (fallback)'] = time.time() - stage_start_profiles
            else:
                logger.info("  No FES renewable generators in thermal stage - correct (handled in Stage 1)")
        
        # =====================================================================
        # STAGE 4c: APPLY NUCLEAR AVAILABILITY FROM ESPENI (HISTORICAL ONLY)
        # =====================================================================
        # For historical scenarios, constrain nuclear output to match actual
        # historical performance using ESPENI data
        stage_start = time.time()
        
        # Check if this is a historical scenario (DUKES data was loaded)
        is_historical = len(dukes_df) > 0
        
        if is_historical:
            logger.info("-" * 80)
            logger.info("PART 4c: APPLYING NUCLEAR AVAILABILITY FROM ESPENI (HISTORICAL)")
            logger.info("-" * 80)
            
            # Load ESPENI data for nuclear availability
            espeni_path = Path("data/demand/espeni.csv")
            if espeni_path.exists():
                try:
                    # Extract year from scenario
                    espeni_year = modelled_year
                    
                    # Load ESPENI data (just the nuclear column and datetime)
                    logger.info(f"Loading ESPENI data for nuclear availability (year {espeni_year})")
                    espeni_df = pd.read_csv(espeni_path, low_memory=False)
                    
                    # ESPENI has datetime in 'ELEC_elex_startTime[utc](datetime)' column (UTC)
                    # or 'ELEC_startTime[localtime](datetime)' for local time
                    datetime_col = None
                    for col in ['ELEC_elex_startTime[utc](datetime)', 
                                'ELEC_startTime[localtime](datetime)',
                                'local_time']:
                        if col in espeni_df.columns:
                            datetime_col = col
                            break
                    
                    if datetime_col is None:
                        raise KeyError(f"No datetime column found in ESPENI. Columns: {espeni_df.columns.tolist()[:10]}")
                    
                    logger.info(f"Using datetime column: {datetime_col}")
                    espeni_df['datetime'] = pd.to_datetime(espeni_df[datetime_col], errors='coerce')
                    espeni_df = espeni_df.set_index('datetime')
                    
                    # Remove timezone if present (for consistency with network snapshots)
                    if espeni_df.index.tz is not None:
                        espeni_df.index = espeni_df.index.tz_localize(None)
                    
                    # Filter to modelled year
                    espeni_df = espeni_df[espeni_df.index.year == espeni_year]
                    
                    # Get nuclear output column
                    nuclear_col = None
                    for col in espeni_df.columns:
                        if 'nuclear' in col.lower():
                            nuclear_col = col
                            break
                    
                    if nuclear_col:
                        # Get installed nuclear capacity from network
                        nuclear_gens = network.generators[
                            network.generators['carrier'].str.lower() == 'nuclear'
                        ]
                        
                        if len(nuclear_gens) > 0:
                            installed_capacity = nuclear_gens['p_nom'].sum()
                            logger.info(f"Found {len(nuclear_gens)} nuclear generators with {installed_capacity:.1f} MW installed capacity")
                            
                            # Get nuclear output time series
                            nuclear_output = pd.to_numeric(espeni_df[nuclear_col], errors='coerce')
                            
                            # Calculate availability (p_max_pu) as output / capacity
                            # Clip to 0-1 range to handle any data issues
                            availability = (nuclear_output / installed_capacity).clip(0, 1)
                            
                            # Resample to match network snapshots
                            # Network snapshots may be half-hourly or hourly
                            availability.index = pd.to_datetime(availability.index)
                            
                            # Align to network snapshots
                            network_snaps = pd.DatetimeIndex(network.snapshots)
                            
                            # Reindex availability to match network snapshots
                            # Use forward fill for any missing values
                            availability_aligned = availability.reindex(network_snaps, method='ffill')
                            availability_aligned = availability_aligned.fillna(availability.mean())
                            
                            # Apply p_max_pu to each nuclear generator
                            logger.info(f"Applying nuclear availability profile to {len(nuclear_gens)} generators")
                            logger.info(f"  Mean availability: {availability_aligned.mean():.2%}")
                            logger.info(f"  Min availability: {availability_aligned.min():.2%}")
                            logger.info(f"  Max availability: {availability_aligned.max():.2%}")
                            
                            for gen_name in nuclear_gens.index:
                                network.generators_t.p_max_pu[gen_name] = availability_aligned.values
                            
                            logger.info(f"Nuclear availability constraints applied from ESPENI")
                            
                            # Calculate expected nuclear output
                            expected_twh = (availability_aligned * installed_capacity).sum() * 0.5 / 1e6  # 0.5 for half-hourly
                            logger.info(f"  Expected nuclear output with constraints: {expected_twh:.2f} TWh")
                        else:
                            logger.warning("No nuclear generators found in network - skipping availability constraints")
                    else:
                        logger.warning("Nuclear column not found in ESPENI data - skipping availability constraints")
                        
                except Exception as e:
                    logger.warning(f"Failed to apply nuclear availability from ESPENI: {e}")
                    logger.warning("Nuclear generators will run unconstrained (may overestimate output)")
            else:
                logger.warning(f"ESPENI file not found at {espeni_path} - nuclear availability not constrained")
                
            stage_times['4c. Nuclear availability'] = time.time() - stage_start
        else:
            logger.info("Future scenario - using FES nuclear availability (no ESPENI constraints)")
        
        # =====================================================================
        # STAGE 5: CREATE SUMMARY
        # =====================================================================
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("PART 5: CREATING SUMMARY WITH DATA SOURCE TRACKING")
        logger.info("-" * 80)
        
        final_gen_count = len(network.generators)
        total_added = final_gen_count - initial_gen_count
        
        logger.info(f"Thermal integration complete:")
        logger.info(f"  Generators before: {initial_gen_count}")
        logger.info(f"  Generators after: {final_gen_count}")
        logger.info(f"  Thermal generators added: {total_added}")
        
        # Create detailed summary with data source tracking
        thermal_gens = network.generators.iloc[initial_gen_count:]  # Only new generators
        
        if len(thermal_gens) > 0:
            # Try to preserve data_source from thermal_data if available
            summary_data = []
            
            for carrier in thermal_gens['carrier'].unique():
                carrier_gens = thermal_gens[thermal_gens['carrier'] == carrier]
                
                # Try to determine data source
                # This requires matching generators back to source data
                # For now, use a simple heuristic based on scenario type
                if len(dukes_df) > 0:
                    data_source = 'DUKES+REPD'  # Historical scenario (no FES)
                elif len(fes_df) > 0:
                    data_source = 'FES'  # Future scenario
                else:
                    data_source = 'Unknown'
                
                summary_data.append({
                    'technology': carrier,
                    'capacity_mw': carrier_gens['p_nom'].sum(),
                    'count': len(carrier_gens),
                    'data_source': data_source
                })
            
            summary_df = pd.DataFrame(summary_data)
            logger.info("\nThermal capacity summary:")
            for _, row in summary_df.iterrows():
                logger.info(f"  {row['technology']}: {row['capacity_mw']:.2f} MW ({row['count']} units) [{row['data_source']}]")
        else:
            summary_df = pd.DataFrame(columns=['technology', 'capacity_mw', 'count', 'data_source'])
            logger.warning("No thermal generators were added")
        stage_times['5. Create summary'] = time.time() - stage_start
        
        # =====================================================================
        # STAGE 6: SAVE OUTPUTS
        # =====================================================================
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("PART 6: SAVING OUTPUTS")
        logger.info("-" * 80)
        
        output_path = snk.output.network
        logger.info(f"Saving network to {output_path}")
        save_network(network, output_path, custom_logger=logger)
        
        summary_path = snk.output.summary
        logger.info(f"Saving thermal summary to {summary_path}")
        summary_df.to_csv(summary_path, index=False)
        stage_times['6. Save outputs'] = time.time() - stage_start
        
        # COORDINATE VALIDATION: Ensure all buses use consistent OSGB36 coordinates
        try:
            from spatial_utils import validate_network_coordinates, ensure_osgb36_coordinates
            validation = validate_network_coordinates(network, fix=False)
            if validation['wgs84_count'] > 0:
                logger.warning(f"COORDINATE CHECK: Found {validation['wgs84_count']} buses with WGS84 coordinates!")
                fixed = ensure_osgb36_coordinates(network)
                if fixed > 0:
                    logger.info(f"COORDINATE FIX: Converted {fixed} buses from WGS84 to OSGB36")
            else:
                logger.info(f"COORDINATE CHECK: All buses use OSGB36 coordinates ✓")
        except ImportError:
            pass
        
        # Final summary
        logger.info("Final network (with renewables + thermal)")
        log_network_info(network, logger)
        
        # Log stage timing summary
        logger.info("")
        log_stage_summary(stage_times, logger, "THERMAL INTEGRATION - STAGE TIMING")
        
        logger.info("=" * 80)
        logger.info("THERMAL GENERATOR INTEGRATION COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Thermal generator integration failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

