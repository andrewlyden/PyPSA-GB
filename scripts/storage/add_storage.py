#!/usr/bin/env python3
"""
Add storage units to PyPSA network.

This script integrates storage assets (batteries, pumped hydro, LAES, CAES, etc.)
into the PyPSA network as StorageUnit components.

Key functions:
- Load processed storage parameters
- Map storage sites to nearest network buses
- Add StorageUnit components to PyPSA network
- Validate storage integration

Author: PyPSA-GB Team
"""

import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import logging
import time
import warnings
from typing import Dict, Tuple, Optional

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Suppress PyPSA warnings about unoptimized networks (expected during network building)
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Import shared spatial utilities
from scripts.utilities.spatial_utils import map_sites_to_buses, apply_etys_bmu_mapping

# Import logging configuration
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    from scripts.utilities.carrier_definitions import add_carriers_to_network
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("add_storage")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def load_storage_parameters(storage_file: str) -> pd.DataFrame:
    """
    Load processed storage parameters.
    
    Args:
        storage_file: Path to storage_parameters.csv
        
    Returns:
        DataFrame with storage parameters
    """
    logger.info(f"Loading storage parameters from: {storage_file}")
    
    storage_df = pd.read_csv(storage_file)
    logger.info(f"Loaded {len(storage_df)} storage sites")
    
    # Validate required columns
    required_cols = ['site_name', 'technology', 'power_mw', 'energy_mwh', 'duration_h',
                     'eta_charge', 'eta_discharge', 'lat', 'lon']
    
    missing_cols = [col for col in required_cols if col not in storage_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out sites without coordinates
    valid_coords = storage_df['lat'].notna() & storage_df['lon'].notna()
    if not valid_coords.all():
        invalid_count = (~valid_coords).sum()
        logger.warning(f"Removing {invalid_count} sites without valid coordinates")
        storage_df = storage_df[valid_coords].copy()
    
    # Filter out sites without capacity
    valid_capacity = storage_df['power_mw'] > 0
    if not valid_capacity.all():
        invalid_count = (~valid_capacity).sum()
        logger.warning(f"Removing {invalid_count} sites without valid capacity")
        storage_df = storage_df[valid_capacity].copy()
    
    logger.info(f"Valid storage sites for integration: {len(storage_df)}")
    
    return storage_df


def filter_storage_by_commissioning_year(
    storage_df: pd.DataFrame, 
    scenario_year: int,
    is_historical: bool = True
) -> pd.DataFrame:
    """
    Filter storage sites to only include those operational by the scenario year.
    
    For historical scenarios, this ensures only storage sites that were commissioned
    by the modelled year are included (e.g., 2020 scenario should have ~1.0-1.5 GW storage,
    not the 2.88 GW that exists in 2024).
    
    Args:
        storage_df: DataFrame with storage data including 'commissioning_year'
        scenario_year: The year being modelled (e.g., 2020)
        is_historical: Whether this is a historical scenario (filtering only applies for historical)
        
    Returns:
        Filtered DataFrame containing only storage sites operational by scenario_year
    """
    if not is_historical:
        logger.info("Future scenario - no commissioning year filtering applied for storage")
        return storage_df
    
    if scenario_year is None:
        logger.warning("No scenario year provided - cannot filter storage by commissioning year")
        return storage_df
    
    initial_count = len(storage_df)
    initial_capacity = storage_df['power_mw'].sum()
    
    # Check if commissioning_year column exists
    if 'commissioning_year' not in storage_df.columns:
        logger.warning("No 'commissioning_year' column found - cannot filter storage by commissioning year")
        return storage_df
    
    # Filter: keep if commissioning_year <= scenario_year OR commissioning_year is NaN
    # For historical scenarios, NaN commissioning means we should be conservative and exclude it
    # unless it's a known legacy asset (pumped hydro typically predates REPD)
    mask_valid_year = storage_df['commissioning_year'] <= scenario_year
    mask_legacy_hydro = (
        (storage_df['technology'] == 'Pumped Storage Hydroelectricity') & 
        storage_df['commissioning_year'].isna()
    )
    mask = mask_valid_year | mask_legacy_hydro
    
    filtered_df = storage_df[mask].copy()
    excluded_df = storage_df[~mask]
    
    if len(excluded_df) > 0:
        excluded_capacity = excluded_df['power_mw'].sum()
        logger.info(f"=== STORAGE COMMISSIONING YEAR FILTER (scenario year: {scenario_year}) ===")
        logger.info(f"  Excluded {len(excluded_df)} storage sites ({excluded_capacity:.1f} MW)")
        
        # Log details by technology
        for tech in excluded_df['technology'].unique():
            tech_excluded = excluded_df[excluded_df['technology'] == tech]
            tech_capacity = tech_excluded['power_mw'].sum()
            logger.info(f"    {tech}: {len(tech_excluded)} sites, {tech_capacity:.1f} MW excluded")
    
    final_capacity = filtered_df['power_mw'].sum()
    logger.info(f"Storage commissioning year filter: {initial_count} → {len(filtered_df)} sites "
                f"({initial_capacity:.1f} → {final_capacity:.1f} MW)")
    
    return filtered_df


def scale_storage_for_future_scenario(
    storage_df: pd.DataFrame,
    fes_file: str,
    modelled_year: int,
    fes_scenario: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    DEPRECATED: This function is replaced by load_fes_storage_data().
    
    The old approach of uniformly scaling REPD storage was incorrect.
    FES provides GSP-level storage data which should be used directly.
    Only "Direct" connected capacity should use REPD spatial distribution.
    
    This function is kept for backwards compatibility but logs a warning.
    """
    logger.warning("scale_storage_for_future_scenario is deprecated - use load_fes_storage_data instead")
    return storage_df


# NESO API endpoints for GSP info by FES year
NESO_GSP_INFO_URLS = {
    2024: 'https://api.neso.energy/dataset/963525d6-5d83-4448-a99c-663f1c76330a/resource/21c2b09c-24ff-4837-a3b1-b6aea88f8124/download/fes2024_regional_breakdown_gsp_info.csv',
    2023: 'https://api.neso.energy/dataset/963525d6-5d83-4448-a99c-663f1c76330a/resource/805e20e4-6a8b-4113-9d2d-09efa0a4bbb9/download/fes2023_regional_breakdown_gsp_info.csv',
}


def _download_gsp_info(fes_year: int, logger: logging.Logger) -> pd.DataFrame:
    """Download GSP info from NESO API for coordinate mapping."""
    import urllib3
    import io
    
    url = NESO_GSP_INFO_URLS.get(fes_year, NESO_GSP_INFO_URLS[2024])
    
    try:
        http = urllib3.PoolManager()
        response = http.request('GET', url, timeout=30.0)
        if response.status != 200:
            logger.error(f"Failed to download GSP info: HTTP {response.status}")
            return pd.DataFrame()
        
        gsp_df = pd.read_csv(io.BytesIO(response.data), encoding='utf-8-sig')
        logger.debug(f"Downloaded GSP info: {len(gsp_df)} entries")
        return gsp_df
    except Exception as e:
        logger.error(f"Error downloading GSP info: {e}")
        return pd.DataFrame()


def load_fes_storage_data(
    fes_file: str,
    modelled_year: int,
    fes_scenario: str,
    repd_storage_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load FES storage data for future scenarios.
    
    FES provides multiple storage building blocks:
    - Srg_BB001: Grid-scale Batteries (GSP-level and Direct)
    - Srg_BB002: Domestic Batteries (G98) - behind-the-meter
    - Srg_BB003: Pumped Hydro (new capacity projections)
    - Srg_BB004: Other storage (CAES, LAES, etc.)
    
    Each can be at GSP-level or Direct-connected (transmission-connected).
    
    Args:
        fes_file: Path to FES data file
        modelled_year: Target year (e.g., 2035, 2050)
        fes_scenario: FES pathway name (e.g., 'Holistic Transition')
        repd_storage_df: REPD storage data for Direct capacity distribution
        logger: Logger instance
        
    Returns:
        DataFrame with storage sites for the scenario
    """
    import os
    
    if not os.path.exists(fes_file):
        logger.warning(f"FES file not found: {fes_file} - using REPD storage only")
        return repd_storage_df
    
    try:
        # Load FES data
        fes_df = pd.read_csv(fes_file, encoding='utf-8-sig')
        
        # Filter to scenario
        if fes_scenario:
            fes_df = fes_df[fes_df['FES Pathway'] == fes_scenario]
        
        # Get target year column
        year_col = str(modelled_year)
        if year_col not in fes_df.columns:
            logger.warning(f"Year {modelled_year} not in FES data - using REPD storage only")
            return repd_storage_df
        
        # Define storage building blocks and their PyPSA carrier mapping
        storage_bb_config = {
            'Srg_BB001': {
                'carrier': 'Battery',
                'nice_name': 'Grid-scale Batteries',
                'default_duration_h': 2.0,
                'eta_charge': 0.92,
                'eta_dispatch': 0.92,
            },
            'Srg_BB002': {
                'carrier': 'Domestic Battery',
                'nice_name': 'Domestic Batteries (G98)',
                'default_duration_h': 2.0,
                'eta_charge': 0.92,
                'eta_dispatch': 0.92,
            },
            'Srg_BB003': {
                'carrier': 'Pumped Storage Hydroelectricity',
                'nice_name': 'Pumped Hydro',
                'default_duration_h': 8.0,
                'eta_charge': 0.87,
                'eta_dispatch': 0.87,
            },
            'Srg_BB004': {
                'carrier': 'LAES',  # Default to LAES; could be CAES
                'nice_name': 'Other Storage (CAES/LAES)',
                'default_duration_h': 6.0,
                'eta_charge': 0.60,
                'eta_dispatch': 0.60,
            },
        }
        
        # Download GSP info for coordinates
        fes_year = 2024  # Default to 2024 GSP info
        gsp_info = _download_gsp_info(fes_year, logger)
        
        # Create storage sites list
        storage_sites = []
        
        logger.info(f"=== FES STORAGE DATA for {modelled_year} ({fes_scenario}) ===")
        
        # Process each storage building block
        for bb_id, bb_config in storage_bb_config.items():
            bb_data = fes_df[fes_df['Building Block ID Number'] == bb_id].copy()
            
            if len(bb_data) == 0:
                logger.debug(f"No data for {bb_id} ({bb_config['nice_name']})")
                continue
            
            # Filter for positive capacity
            bb_data = bb_data[bb_data[year_col] > 0]
            if len(bb_data) == 0:
                continue
            
            total_capacity = bb_data[year_col].sum()
            logger.info(f"  {bb_id} ({bb_config['nice_name']}): {total_capacity:,.0f} MW")
            
            # Split into GSP vs Direct
            gsp_rows = bb_data[~bb_data['GSP'].str.contains('Direct', case=False, na=False)]
            direct_rows = bb_data[bb_data['GSP'].str.contains('Direct', case=False, na=False)]
            
            carrier = bb_config['carrier']
            duration = bb_config['default_duration_h']
            eta_charge = bb_config['eta_charge']
            eta_dispatch = bb_config['eta_dispatch']
            
            # Process GSP-level storage
            for _, row in gsp_rows.iterrows():
                capacity = row[year_col]
                gsp_name = row['GSP']
                
                # Try to find GSP coordinates
                lat, lon = None, None
                if len(gsp_info) > 0:
                    # Try exact name match
                    match = gsp_info[gsp_info['Name'].str.lower() == gsp_name.lower()]
                    if len(match) == 0:
                        # Try partial match on first word
                        first_word = gsp_name.lower().split()[0] if gsp_name else ''
                        if first_word:
                            match = gsp_info[gsp_info['Name'].str.lower().str.contains(first_word, na=False)]
                    
                    if len(match) > 0:
                        lat = match.iloc[0]['Latitude']
                        lon = match.iloc[0]['Longitude']
                
                if lat is None or lon is None:
                    logger.debug(f"No coordinates found for GSP: {gsp_name} ({bb_id})")
                    continue
                
                # Clean GSP name for site name
                gsp_clean = gsp_name.replace(' ', '_').replace('(', '').replace(')', '')
                
                storage_sites.append({
                    'site_name': f"FES_{carrier.replace(' ', '_')}_{gsp_clean}_{modelled_year}",
                    'technology': carrier,
                    'power_mw': capacity,
                    'energy_mwh': capacity * duration,
                    'duration_h': duration,
                    'eta_charge': eta_charge,
                    'eta_discharge': eta_dispatch,
                    'lat': lat,
                    'lon': lon,
                    'source': f'FES_GSP_{bb_id}'
                })
            
            # Process Direct-connected storage using REPD distribution
            for _, direct_row in direct_rows.iterrows():
                direct_cap = direct_row[year_col]
                region = direct_row['GSP']  # e.g., "Direct(NGET)"
                
                # Determine which REPD sites to use for distribution
                # For batteries, use battery sites; for others, use all storage sites
                if 'Battery' in carrier:
                    repd_sites = repd_storage_df[repd_storage_df['technology'] == 'Battery'].copy()
                else:
                    # For LAES/CAES/Pumped Hydro, distribute proportionally across all storage
                    repd_sites = repd_storage_df.copy()
                
                if len(repd_sites) == 0:
                    # Fallback: create synthetic sites in the region
                    logger.warning(f"No REPD sites for distributing {region} {carrier}")
                    # Use region centroids as fallback
                    region_centroids = {
                        'SHETL': (57.5, -4.0),  # Northern Scotland
                        'SPTL': (56.0, -4.0),   # Southern Scotland
                        'NGET': (52.5, -1.5),   # England & Wales centroid
                    }
                    for reg_name, (lat, lon) in region_centroids.items():
                        if reg_name in region:
                            storage_sites.append({
                                'site_name': f"FES_{carrier.replace(' ', '_')}_{reg_name}_{modelled_year}",
                                'technology': carrier,
                                'power_mw': direct_cap,
                                'energy_mwh': direct_cap * duration,
                                'duration_h': duration,
                                'eta_charge': eta_charge,
                                'eta_discharge': eta_dispatch,
                                'lat': lat,
                                'lon': lon,
                                'source': f'FES_Direct_{bb_id}'
                            })
                            break
                    continue
                
                # Filter REPD sites by region (rough geographic filter)
                if 'SHETL' in region:
                    region_sites = repd_sites[repd_sites['lat'] > 57.0]
                elif 'SPTL' in region:
                    region_sites = repd_sites[(repd_sites['lat'] > 55.5) & (repd_sites['lat'] <= 57.0)]
                else:  # NGET
                    region_sites = repd_sites[repd_sites['lat'] <= 55.5]
                
                if len(region_sites) == 0:
                    region_sites = repd_sites  # Fallback to all sites
                
                region_total = region_sites['power_mw'].sum()
                scale_factor = direct_cap / region_total if region_total > 0 else 0
                
                for _, site in region_sites.iterrows():
                    scaled_capacity = site['power_mw'] * scale_factor
                    if scaled_capacity <= 0:
                        continue
                    
                    storage_sites.append({
                        'site_name': f"FES_Direct_{carrier.replace(' ', '_')}_{site['site_name']}_{modelled_year}",
                        'technology': carrier,
                        'power_mw': scaled_capacity,
                        'energy_mwh': scaled_capacity * duration,
                        'duration_h': duration,
                        'eta_charge': eta_charge,
                        'eta_discharge': eta_dispatch,
                        'lat': site['lat'],
                        'lon': site['lon'],
                        'source': f'FES_Direct_{bb_id}'
                    })
                
                logger.info(f"    Distributed {region}: {direct_cap:,.0f} MW across {len(region_sites)} sites")
        
        # Note: We no longer add REPD pumped hydro separately since FES Srg_BB003 
        # now provides pumped hydro projections which we've already processed above.
        # The FES data includes existing capacity in its baseline.
        
        if len(storage_sites) == 0:
            logger.warning("No FES storage sites created - using REPD storage only")
            return repd_storage_df
        
        result_df = pd.DataFrame(storage_sites)
        
        total_capacity = result_df['power_mw'].sum()
        logger.info(f"=== FES STORAGE SUMMARY ===")
        logger.info(f"  Total sites: {len(result_df)}")
        logger.info(f"  Total capacity: {total_capacity:,.0f} MW")
        for tech in result_df['technology'].unique():
            tech_cap = result_df[result_df['technology'] == tech]['power_mw'].sum()
            logger.info(f"    {tech}: {tech_cap:,.0f} MW")
        
        return result_df
        
    except Exception as e:
        logger.warning(f"Error loading FES storage data: {e} - using REPD storage only")
        import traceback
        logger.debug(traceback.format_exc())
        return repd_storage_df


# Note: Bus mapping now handled by shared map_sites_to_buses function
# from spatial_utils.py. This provides consistent coordinate handling
# across all PyPSA-GB components (renewables, thermal, storage, interconnectors).
# The shared function automatically detects coordinate systems (WGS84 vs OSGB36)
# and uses appropriate distance metrics (haversine vs Euclidean).


def add_storage_to_network(network: pypsa.Network, storage_df: pd.DataFrame,
                           standing_loss_default: float = 0.001) -> pypsa.Network:
    """
    Add storage units to the PyPSA network.
    
    Args:
        network: PyPSA network object
        storage_df: DataFrame with storage sites and assigned buses
        standing_loss_default: Default standing loss per hour (self-discharge)
        
    Returns:
        Updated network with storage units
    """
    logger.info(f"Adding {len(storage_df)} storage units to network...")
    
    # Filter out storage units with invalid bus assignments
    valid_bus_mask = storage_df['bus'].notna() & (storage_df['bus'] != '') & storage_df['bus'].isin(network.buses.index)
    invalid_count = (~valid_bus_mask).sum()
    
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} storage units with invalid/missing bus assignments")
        logger.info(f"  Invalid buses: {storage_df[~valid_bus_mask]['bus'].unique().tolist()}")
        storage_df = storage_df[valid_bus_mask].copy()
    
    if len(storage_df) == 0:
        logger.warning("No storage units to add after filtering")
        return network
    
    # Track what we're adding
    tech_counts = {}
    total_power = 0
    total_energy = 0
    
    for idx, row in storage_df.iterrows():
        # Use site_name if available, otherwise create unique name
        if 'site_name' in row and pd.notna(row['site_name']) and row['site_name']:
            # Clean site name for use as index (remove special chars, spaces -> underscore)
            site_name_clean = str(row['site_name']).replace(' ', '_').replace('/', '_').replace('-', '_')
            site_name_clean = ''.join(c for c in site_name_clean if c.isalnum() or c == '_')
            storage_name = f"{site_name_clean}_{idx}"
        else:
            storage_name = f"storage_{row['technology'].replace(' ', '_')}_{idx}"
        
        # Get coordinates from assigned bus (most reliable source)
        bus_name = row['bus']
        if bus_name in network.buses.index:
            bus_lon = network.buses.loc[bus_name, 'x'] if 'x' in network.buses.columns else row.get('lon', None)
            bus_lat = network.buses.loc[bus_name, 'y'] if 'y' in network.buses.columns else row.get('lat', None)
        else:
            bus_lon = row.get('lon', None)
            bus_lat = row.get('lat', None)
        
        # Prepare storage parameters
        storage_params = {
            'bus': row['bus'],
            'carrier': row['technology'],
            'p_nom': row['power_mw'],  # Nominal power capacity
            'max_hours': row['duration_h'],  # Energy/Power ratio
            'efficiency_store': row['eta_charge'],  # Charge efficiency
            'efficiency_dispatch': row['eta_discharge'],  # Discharge efficiency
            'standing_loss': standing_loss_default,  # Self-discharge per hour
            'capital_cost': row.get('capital_cost', 0),
            'marginal_cost': row.get('marginal_cost', 0),
        }
        
        # Add coordinates if available
        # Note: Don't use 'x'/'y' for StorageUnit as PyPSA warns these are standard attributes
        # for other components. Coordinates are inherited from the bus anyway.
        # If needed for visualization, use custom attributes like 'longitude'/'latitude'
        if bus_lon is not None and bus_lat is not None:
            storage_params['longitude'] = bus_lon
            storage_params['latitude'] = bus_lat
        
        # Add cyclic state of charge if available
        if 'cyclic_state_of_charge' not in storage_params:
            storage_params['cyclic_state_of_charge'] = False  # Don't force SOC to match at start/end
        
        # Add StorageUnit to network
        try:
            network.add("StorageUnit", storage_name, **storage_params)
            
            # Track statistics
            tech = row['technology']
            if tech not in tech_counts:
                tech_counts[tech] = {'count': 0, 'power': 0, 'energy': 0}
            tech_counts[tech]['count'] += 1
            tech_counts[tech]['power'] += row['power_mw']
            tech_counts[tech]['energy'] += row['energy_mwh']
            
            total_power += row['power_mw']
            total_energy += row['energy_mwh']
            
        except Exception as e:
            logger.error(f"Failed to add storage unit {storage_name}: {e}")
            continue
    
    # Log summary
    logger.info(f"Successfully added {len(network.storage_units)} storage units to network")
    logger.info(f"Total storage capacity: {total_power:.1f} MW, {total_energy:.1f} MWh")
    
    logger.info("\nStorage by technology:")
    for tech, stats in sorted(tech_counts.items()):
        logger.info(f"  {tech}:")
        logger.info(f"    Count: {stats['count']} units")
        logger.info(f"    Power: {stats['power']:.1f} MW")
        logger.info(f"    Energy: {stats['energy']:.1f} MWh")
        logger.info(f"    Avg duration: {stats['energy']/stats['power']:.1f} hours")
    
    return network


def validate_storage_integration(network: pypsa.Network, 
                                 storage_df: pd.DataFrame) -> Dict:
    """
    Validate that storage was integrated correctly.
    
    Args:
        network: PyPSA network with storage
        storage_df: Original storage DataFrame
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating storage integration...")
    
    validation = {
        'success': True,
        'issues': [],
        'stats': {}
    }
    
    # Check storage units exist
    if not hasattr(network, 'storage_units') or len(network.storage_units) == 0:
        validation['success'] = False
        validation['issues'].append("No storage units found in network")
        logger.error("VALIDATION FAILED: No storage units in network")
        return validation
    
    # Check count matches
    expected_count = len(storage_df)
    actual_count = len(network.storage_units)
    
    if actual_count != expected_count:
        validation['issues'].append(
            f"Storage count mismatch: expected {expected_count}, got {actual_count}"
        )
        logger.warning(f"Expected {expected_count} storage units, found {actual_count}")
    
    # Validate capacities
    storage_units = network.storage_units
    
    # Check for invalid capacities
    invalid_power = (storage_units['p_nom'] <= 0).sum()
    if invalid_power > 0:
        validation['issues'].append(f"{invalid_power} storage units have invalid power capacity")
        logger.warning(f"Found {invalid_power} storage units with p_nom <= 0")
    
    invalid_duration = (storage_units['max_hours'] <= 0).sum()
    if invalid_duration > 0:
        validation['issues'].append(f"{invalid_duration} storage units have invalid duration")
        logger.warning(f"Found {invalid_duration} storage units with max_hours <= 0")
    
    # Check efficiencies are in valid range
    invalid_eff_store = ((storage_units['efficiency_store'] < 0.1) | 
                         (storage_units['efficiency_store'] > 1.0)).sum()
    invalid_eff_dispatch = ((storage_units['efficiency_dispatch'] < 0.1) | 
                            (storage_units['efficiency_dispatch'] > 1.0)).sum()
    
    if invalid_eff_store > 0 or invalid_eff_dispatch > 0:
        validation['issues'].append(
            f"Invalid efficiencies: {invalid_eff_store} store, {invalid_eff_dispatch} dispatch"
        )
    
    # Check bus assignments
    invalid_buses = ~storage_units['bus'].isin(network.buses.index)
    if invalid_buses.any():
        validation['issues'].append(f"{invalid_buses.sum()} storage units assigned to non-existent buses")
        logger.error(f"Found {invalid_buses.sum()} storage units with invalid bus assignments")
    
    # Collect statistics
    validation['stats'] = {
        'total_units': actual_count,
        'total_power_mw': storage_units['p_nom'].sum(),
        'total_energy_mwh': (storage_units['p_nom'] * storage_units['max_hours']).sum(),
        'avg_duration_h': (storage_units['p_nom'] * storage_units['max_hours']).sum() / storage_units['p_nom'].sum(),
        'avg_efficiency_store': storage_units['efficiency_store'].mean(),
        'avg_efficiency_dispatch': storage_units['efficiency_dispatch'].mean(),
        'unique_buses': storage_units['bus'].nunique(),
        'carriers': storage_units['carrier'].unique().tolist() if 'carrier' in storage_units.columns else []
    }
    
    # Log results
    if validation['issues']:
        logger.warning(f"Validation found {len(validation['issues'])} issues:")
        for issue in validation['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✓ Storage integration validation passed!")
    
    logger.info("\nStorage integration statistics:")
    for key, value in validation['stats'].items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return validation


def main():
    """Main function to add storage to PyPSA network."""
    global logger
    start_time = time.time()
    
    # Reinitialize logger with Snakemake log path if available
    if 'snakemake' in globals() and hasattr(snakemake, 'log') and snakemake.log:
        logger = setup_logging(snakemake.log[0])
    
    logger.info("="*80)
    logger.info("Starting storage integration into PyPSA network...")
    logger.info("="*80)
    
    try:
        # Get input and output files
        try:
            # Snakemake mode
            network_input = snakemake.input.network
            storage_file = snakemake.input.storage_data
            network_output = snakemake.output.network
            max_distance_km = snakemake.params.get('max_distance_km', 50.0)
            standing_loss = snakemake.params.get('standing_loss', 0.001)
            modelled_year = snakemake.params.get('modelled_year', None)
            is_historical = snakemake.params.get('is_historical', False)
            logger.info("Running in Snakemake mode")
        except NameError:
            # Standalone mode
            import sys
            if len(sys.argv) < 4:
                logger.error("Usage: python add_storage.py <input_network> <storage_params> <output_network>")
                sys.exit(1)
            
            network_input = sys.argv[1]
            storage_file = sys.argv[2]
            network_output = sys.argv[3]
            max_distance_km = 50.0
            standing_loss = 0.001
            modelled_year = None
            is_historical = False
            logger.info("Running in standalone mode")
        
        logger.info(f"Input network: {network_input}")
        logger.info(f"Storage parameters: {storage_file}")
        logger.info(f"Output network: {network_output}")
        if modelled_year:
            logger.info(f"Modelled year: {modelled_year} ({'historical' if is_historical else 'future'} scenario)")
        
        # Load storage parameters
        storage_df = load_storage_parameters(storage_file)
        
        if len(storage_df) == 0:
            logger.warning("No storage sites to integrate - copying input network to output")
            import shutil
            shutil.copy(network_input, network_output)
            return
        
        # Filter storage by commissioning year for historical scenarios
        # This ensures only storage sites that existed in the modelled year are included
        storage_df = filter_storage_by_commissioning_year(
            storage_df, 
            scenario_year=modelled_year,
            is_historical=is_historical
        )
        
        if len(storage_df) == 0:
            logger.warning("No storage sites remaining after commissioning year filter - copying input network to output")
            import shutil
            shutil.copy(network_input, network_output)
            return
        
        # For future scenarios, load FES storage data directly
        # FES provides GSP-level storage - only "Direct" uses REPD distribution
        if not is_historical and modelled_year is not None:
            try:
                fes_file = None
                fes_scenario = None
                
                if 'snakemake' in globals():
                    # Try to get FES file from snakemake input
                    fes_file = getattr(snakemake.input, 'fes_data', None)
                    fes_scenario = getattr(snakemake.params, 'fes_scenario', None)
                    logger.info(f"FES scenario from snakemake params: '{fes_scenario}'")
                
                # Try to find FES file if not provided
                if fes_file is None:
                    import os
                    fes_paths = [
                        "resources/FES/FES_2024_data.csv",
                        "data/FES/FES_2024_data.csv",
                    ]
                    for path in fes_paths:
                        if os.path.exists(path):
                            fes_file = path
                            break
                
                if fes_file:
                    if fes_scenario is None:
                        logger.warning("FES scenario not specified - will load ALL FES pathways (likely incorrect!)")
                    # Use new FES-based storage loading (GSP-level data + REPD for Direct)
                    storage_df = load_fes_storage_data(
                        fes_file=fes_file,
                        modelled_year=modelled_year,
                        fes_scenario=fes_scenario,
                        repd_storage_df=storage_df,
                        logger=logger
                    )
            except Exception as e:
                logger.warning(f"Could not load FES storage data: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Load network
        logger.info(f"\nLoading PyPSA network from: {network_input}")
        network = load_network(network_input, custom_logger=logger)
        logger.info(f"Network loaded: {len(network.buses)} buses, {len(network.generators)} generators")
        
        # Add carrier definitions to ensure storage carriers are defined
        try:
            network = add_carriers_to_network(network, logger)
        except Exception as e:
            logger.warning(f"Could not add carrier definitions: {e}")
        
        # Map storage to buses using shared spatial utilities
        storage_df = map_sites_to_buses(
            network=network,
            sites_df=storage_df,
            method='nearest',
            lat_col='lat',
            lon_col='lon',
            max_distance_km=max_distance_km
        )
        
        # Rename distance_km column to match expected name
        if 'distance_km' in storage_df.columns:
            storage_df['distance_to_bus_km'] = storage_df['distance_km']
        
        # Apply ETYS BMU mapping for large storage units (pumped hydro, large batteries)
        # This ensures they connect to appropriate high-voltage buses
        if 'ETYS' in network_input or 'ETYS' in network_output:
            logger.info("Applying ETYS BMU mapping corrections for storage...")
            # Use station_name or site_name for matching
            if 'station_name' not in storage_df.columns and 'site_name' in storage_df.columns:
                storage_df['station_name'] = storage_df['site_name']
            # Use power_mw as capacity_mw for the function
            if 'capacity_mw' not in storage_df.columns and 'power_mw' in storage_df.columns:
                storage_df['capacity_mw'] = storage_df['power_mw']
            storage_df = apply_etys_bmu_mapping(storage_df, network)
        
        # Add storage to network
        network = add_storage_to_network(network, storage_df, standing_loss)
        
        # Validate integration
        validation = validate_storage_integration(network, storage_df)
        
        # Save network
        logger.info(f"\nSaving network with storage to: {network_output}")
        save_network(network, network_output, custom_logger=logger)
        logger.info("Network saved successfully")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'input_storage_sites': len(storage_df),
            'storage_units_added': len(network.storage_units) if hasattr(network, 'storage_units') else 0,
            'total_power_mw': validation['stats'].get('total_power_mw', 0),
            'total_energy_mwh': validation['stats'].get('total_energy_mwh', 0),
            'avg_duration_h': validation['stats'].get('avg_duration_h', 0),
            'unique_buses_used': validation['stats'].get('unique_buses', 0),
            'validation_issues': len(validation['issues']),
            'validation_success': validation['success']
        }
        
        log_execution_summary(logger, "add_storage", execution_time, summary_stats)
        
        # Save integration summary and bus mapping (optional outputs)
        try:
            # Get output file paths from snakemake params if available
            integration_summary_file = None
            bus_mapping_file = None
            
            if 'snakemake' in globals():
                integration_summary_file = getattr(snakemake.output, 'integration_summary', None)
                bus_mapping_file = getattr(snakemake.output, 'bus_mapping', None)
            
            if integration_summary_file:
                # Create integration summary
                summary_df = pd.DataFrame([validation['stats']])
                summary_df.to_csv(integration_summary_file, index=False)
                logger.info(f"Saved integration summary to: {integration_summary_file}")
            
            if bus_mapping_file and 'bus' in storage_df.columns:
                # Create bus mapping report
                bus_mapping_df = storage_df[['site_name', 'technology', 'power_mw', 'energy_mwh', 'lat', 'lon', 'bus']].copy()
                bus_mapping_df.to_csv(bus_mapping_file, index=False)
                logger.info(f"Saved bus mapping to: {bus_mapping_file}")
        except Exception as e:
            logger.debug(f"Optional summary/mapping files not configured: {e}")
        
        # COORDINATE VALIDATION: Ensure all buses use consistent OSGB36 coordinates
        try:
            from spatial_utils import validate_network_coordinates, ensure_osgb36_coordinates
            validation_coords = validate_network_coordinates(network, fix=False)
            if validation_coords['wgs84_count'] > 0:
                logger.warning(f"COORDINATE CHECK: Found {validation_coords['wgs84_count']} buses with WGS84 coordinates!")
                fixed = ensure_osgb36_coordinates(network)
                if fixed > 0:
                    logger.info(f"COORDINATE FIX: Converted {fixed} buses from WGS84 to OSGB36")
            else:
                logger.info(f"COORDINATE CHECK: All buses use OSGB36 coordinates ✓")
        except ImportError:
            pass
        
        if not validation['success']:
            logger.error("Storage integration completed with validation errors!")
            # Don't fail - network was saved, but log the issues
        else:
            logger.info("="*80)
            logger.info("Storage integration completed successfully!")
            logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in storage integration: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

