"""
Integrate renewable generators into PyPSA network.

This script adds weather-variable renewable generators to the network:
- Wind onshore/offshore (with weather-based profiles)
- Solar PV (with weather-based profiles)
- Marine renewables (tidal stream, wave, lagoon - with cyclic profiles)
- Hydro (small run-of-river and large reservoir-based)

Input:
  - Network with base demand loads
  - Renewable site data (individual generators with capacity and location)
  - Renewable time series profiles (weather-based or synthetic)

Output:
  - Network with renewable generators integrated
  - Summary CSV of renewable capacity by technology

Author: PyPSA-GB Team
Date: 2025-10-10
"""

import pandas as pd
import pypsa
import os
from pathlib import Path
import time

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

from scripts.utilities.logging_config import (
    setup_logging, 
    log_dataframe_info, 
    log_network_info, 
    log_execution_summary,
    log_stage_timing,
    log_stage_summary
)
from scripts.utilities.carrier_definitions import add_carriers_to_network

import numpy as np
from typing import Dict, List, Optional, Tuple
import difflib
import hashlib
import logging

# Import shared spatial utilities
from scripts.utilities.spatial_utils import map_sites_to_buses, apply_etys_bmu_mapping

# Inlined helpers and functions from scripts/add_generators.py to make this script
# self-contained and avoid cross-module imports that cause multiple loggers.

# Caches to avoid reloading profiles
_PROFILE_CACHE = {}


def filter_sites_by_year(sites_df: pd.DataFrame, modelled_year: int, 
                         logger: logging.Logger = None) -> pd.DataFrame:
    """
    Filter renewable sites by operational date for historical year modeling.
    
    For historical scenarios, only sites that were operational by the modelled year
    should be included. This ensures 2010 only has sites operational by 2010, etc.
    
    Parameters
    ----------
    sites_df : pd.DataFrame
        Renewable sites dataframe with 'operational_date' column (DD/MM/YYYY format)
    modelled_year : int
        The year being modelled (sites must be operational by end of this year)
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only sites operational by modelled_year
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    original_count = len(sites_df)
    
    # Check if operational_date column exists
    if 'operational_date' not in sites_df.columns:
        logger.warning("No 'operational_date' column found - cannot filter by year")
        return sites_df
    
    # Parse operational dates (DD/MM/YYYY format from REPD)
    operational_dates = pd.to_datetime(
        sites_df['operational_date'], 
        format='%d/%m/%Y', 
        errors='coerce'
    )
    
    # Filter to sites operational by end of modelled year
    cutoff_date = pd.Timestamp(f'{modelled_year}-12-31')
    mask = operational_dates <= cutoff_date
    
    # Handle NaT (unparseable dates) - include them for robustness
    mask = mask | operational_dates.isna()
    
    filtered_df = sites_df[mask].copy()
    filtered_count = len(filtered_df)
    
    if original_count > 0:
        pct_kept = (filtered_count / original_count) * 100
        logger.info(f"Year-based filtering: {filtered_count}/{original_count} sites operational by {modelled_year} ({pct_kept:.1f}%)")
    
    return filtered_df


def _hash_snapshots(snapshots) -> str:
    if len(snapshots) == 0:
        return "empty"
    key_str = f"{len(snapshots)}_{snapshots[0]}_{snapshots[-1]}"
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def _get_profile_cache_key(profile_path: Path, network_snapshots_hash: str) -> str:
    mtime = profile_path.stat().st_mtime if profile_path.exists() else 0
    return f"{profile_path}_{mtime}_{network_snapshots_hash}"


def fuzzy_match_string(query: str, choices: List[str], threshold: float = 0.8) -> Tuple[Optional[str], float]:
    if not choices:
        return None, 0.0
    best_match = None
    best_score = 0.0
    for choice in choices:
        similarity = difflib.SequenceMatcher(None, query.lower(), choice.lower()).ratio()
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = choice
    return best_match, best_score


# Mapping from network carrier names (lowercase_underscore) to CSV carrier names (Title Case)
# This ensures generator characteristics lookup works correctly
CARRIER_NAME_MAPPING = {
    # Variable renewables
    'wind_offshore': 'Wind Offshore',
    'wind_onshore': 'Wind Onshore',
    'solar_pv': 'Solar Photovoltaics',
    'solar': 'Solar Photovoltaics',
    # Hydro
    'large_hydro': 'Large Hydro',
    'small_hydro': 'Small Hydro',
    # Marine
    'shoreline_wave': 'Shoreline Wave',
    'tidal_stream': 'Tidal Barrage and Tidal Stream',
    'tidal_lagoon': 'Tidal Barrage and Tidal Stream',
    # Waste / biogas
    'anaerobic_digestion': 'Anaerobic Digestion',
    'landfill_gas': 'Landfill Gas',
    'sewage_sludge_digestion': 'Sewage Sludge Digestion',
    'efw_incineration': 'EfW Incineration',
    # Thermal
    'ccgt': 'CCGT',
    'ocgt': 'OCGT',
    'coal': 'Coal',
    'oil': 'Oil',
    'nuclear': 'Nuclear',
    'biomass': 'Biomass (dedicated)',
    'ccs_gas': 'CCS Gas',
    'ccs_biomass': 'CCS Biomass',
    'hydrogen': 'Hydrogen',
}

# Reverse mapping (CSV name -> network name) for convenience
CSV_TO_CARRIER_MAPPING = {v: k for k, v in CARRIER_NAME_MAPPING.items()}


def get_csv_carrier_name(network_carrier: str) -> str:
    """
    Convert network carrier name to CSV carrier name for characteristics lookup.
    
    Args:
        network_carrier: Carrier name as used in PyPSA network (e.g., 'wind_offshore')
        
    Returns:
        CSV carrier name (e.g., 'Wind Offshore') or original if no mapping exists
    """
    return CARRIER_NAME_MAPPING.get(network_carrier, network_carrier)


def load_generator_characteristics() -> Dict:
    try:
        char_path = Path("data/generators/generator_data_by_fuel.csv")
        if char_path.exists():
            df = pd.read_csv(char_path, index_col='fuel')
            if 'committable' in df.columns:
                df['committable'] = df['committable'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
            return df.to_dict('index')
        else:
            logger.warning(f"Generator characteristics file not found: {char_path}")
            return {}
    except Exception as e:
        logger.warning(f"Failed to load generator characteristics: {e}")
        return {}


def validate_renewable_data_quality(
    sites_df: pd.DataFrame,
    profiles: Dict[str, pd.DataFrame],
    logger,
    p_nom_col: str = 'capacity_mw'
) -> Dict[str, any]:
    metrics = {'total_sites': len(sites_df), 'valid_sites': 0, 'warnings': [], 'errors': []}
    logger.info("=" * 80)
    logger.info("DATA QUALITY VALIDATION - RENEWABLE GENERATORS")
    logger.info("=" * 80)
    if p_nom_col in sites_df.columns:
        capacities = sites_df[p_nom_col].dropna()
        if len(capacities) > 0:
            non_positive = (capacities <= 0).sum()
            if non_positive > 0:
                msg = f"Found {non_positive} sites with non-positive capacity"
                metrics['warnings'].append(msg)
                logger.warning(f"  ⚠️  {msg}")
            very_large = (capacities > 2000).sum()
            if very_large > 0:
                max_capacity = capacities.max()
                msg = f"Found {very_large} sites with capacity >2000 MW (max: {max_capacity:.1f} MW)"
                metrics['warnings'].append(msg)
                logger.warning(f"  ⚠️  {msg}")
            logger.info(f"[OK] Capacity statistics:")
            logger.info(f"    Total sites: {len(capacities)}")
            logger.info(f"    Total capacity: {capacities.sum():.1f} MW")
            logger.info(f"    Average: {capacities.mean():.1f} MW")
            logger.info(f"    Median: {capacities.median():.1f} MW")
            logger.info(f"    Range: {capacities.min():.1f} - {capacities.max():.1f} MW")
        else:
            msg = "No valid capacity data found"
            metrics['errors'].append(msg)
            logger.error(f"  ❌ {msg}")
    if profiles:
        logger.info(f"[OK] Validating {len(profiles)} technology profiles")
        profiles_exceeding_1 = []
        for tech, profile_df in profiles.items():
            nan_count = profile_df.isna().sum().sum()
            if nan_count > 0:
                nan_pct = (nan_count / profile_df.size) * 100
                msg = f"Profile '{tech}' has {nan_count} NaN values ({nan_pct:.2f}%)"
                metrics['warnings'].append(msg)
                logger.warning(f"  ⚠️  {msg}")
            if len(profile_df) > 0:
                min_cf = profile_df.min().min()
                max_cf = profile_df.max().max()
                if min_cf < 0:
                    msg = f"Profile '{tech}' has negative capacity factors (min: {min_cf:.3f})"
                    metrics['warnings'].append(msg)
                    logger.warning(f"  ⚠️  {msg}")
                if max_cf > 1.0:
                    msg = f"Profile '{tech}' exceeds 1.0 capacity factor (max: {max_cf:.3f}) - may be MW not p.u."
                    metrics['warnings'].append(msg)
                    profiles_exceeding_1.append(tech)
                logger.info(f"  '{tech}': {profile_df.shape[0]} timesteps, {profile_df.shape[1]} sites")
                logger.info(f"    CF range: {min_cf:.3f} - {max_cf:.3f}, mean: {profile_df.mean().mean():.3f}")
        if profiles_exceeding_1:
            logger.info(
                f"Profiles appear to be in MW for technologies: {', '.join(profiles_exceeding_1)}; these will be converted to capacity factors per-generator."
            )
    if 'lat' in sites_df.columns and 'lon' in sites_df.columns and 'carrier' in sites_df.columns:
        sites_copy = sites_df.copy()
        sites_copy['lat_round'] = sites_copy['lat'].round(4)
        sites_copy['lon_round'] = sites_copy['lon'].round(4)
        duplicates = sites_copy.groupby(['lat_round', 'lon_round', 'carrier']).size()
        duplicates = duplicates[duplicates > 1]
        if len(duplicates) > 0:
            total_dup = duplicates.sum() - len(duplicates)
            msg = f"Found {len(duplicates)} duplicate locations with {total_dup} total duplicate sites"
            metrics['warnings'].append(msg)
            logger.warning(f"  ⚠️  {msg}")
            top_dups = duplicates.nlargest(5)
            for (lat, lon, carrier), count in top_dups.items():
                logger.warning(f"    {carrier} at ({lat:.4f}, {lon:.4f}): {count} sites")
    if 'carrier' in sites_df.columns:
        tech_counts = sites_df['carrier'].value_counts()
        logger.info(f"[OK] Technology distribution ({len(tech_counts)} types):")
        for tech, count in tech_counts.items():
            capacity = sites_df[sites_df['carrier'] == tech][p_nom_col].sum() if p_nom_col in sites_df.columns else 0
            logger.info(f"    {tech}: {count} sites, {capacity:.1f} MW")
    metrics['valid_sites'] = len(sites_df)
    logger.info("=" * 80)
    logger.info(f"VALIDATION SUMMARY:")
    logger.info(f"  Total sites: {metrics['total_sites']}")
    logger.info(f"  Warnings: {len(metrics['warnings'])}")
    logger.info(f"  Errors: {len(metrics['errors'])}")
    return metrics


# Note: map_sites_to_buses function now imported from spatial_utils.py
# (See import statement at top of file)


def add_renewable_generators(
    network: pypsa.Network,
    sites_df: pd.DataFrame,
    profiles_dir: Optional[str] = None,
    p_nom_col: str = 'capacity_mw',
    carrier_col: str = 'technology',
    overwrite: bool = False,
    snapshot_weighting: Optional[pd.Series] = None
) -> pypsa.Network:
    logger.info(f"Adding {len(sites_df)} renewable generators to network")
    required_cols = ['bus', p_nom_col, carrier_col]
    missing_cols = [col for col in required_cols if col not in sites_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in sites_df: {missing_cols}")
    valid_sites = sites_df.dropna(subset=['bus']).copy()
    if len(valid_sites) < len(sites_df):
        logger.warning(f"Dropping {len(sites_df) - len(valid_sites)} sites without bus assignments")
    profiles = {}
    profile_technologies = []
    snapshots_hash = _hash_snapshots(network.snapshots)
    if profiles_dir and os.path.exists(profiles_dir):
        logger.info(f"Loading renewable profiles from {profiles_dir}")
        tech_to_profile = {
            'wind_onshore': 'wind_onshore',
            'wind_offshore': 'wind_offshore',
            'solar_pv': 'solar_pv',
            'tidal_stream': 'tidal_stream',
            'shoreline_wave': 'shoreline_wave',
            'tidal_lagoon': 'tidal_lagoon',
            'large_hydro': 'large_hydro',
            'small_hydro': 'small_hydro'
        }
        for tech in valid_sites[carrier_col].unique():
            profile_name = tech_to_profile.get(tech, tech.lower().replace(' ', '_'))
            profile_files = list(Path(profiles_dir).glob(f"*{profile_name}*.csv"))
            if profile_files:
                profile_path = profile_files[0]
                cache_key = _get_profile_cache_key(profile_path, snapshots_hash)
                if cache_key in _PROFILE_CACHE:
                    profiles[tech] = _PROFILE_CACHE[cache_key]
                    profile_technologies.append(tech)
                    logger.info(f"Using cached profile for {tech}: {profiles[tech].shape}")
                    continue
                try:
                    profile_data = pd.read_csv(profile_path, index_col=0, parse_dates=True)
                    _PROFILE_CACHE[cache_key] = profile_data
                    profiles[tech] = profile_data
                    profile_technologies.append(tech)
                    logger.info(f"Loaded and cached profile for {tech}: {profile_data.shape}")
                except Exception as e:
                    logger.warning(f"Could not load profile for {tech}: {e}")
            else:
                logger.info(f"No profile file found for {tech}, will use default availability")
    generator_characteristics = load_generator_characteristics()
    logger.info(f"Loaded characteristics for {len(generator_characteristics)} generator types")
    logger.info(f"Loading profiles for {len(profiles)} technologies: {profile_technologies}")
    quality_metrics = validate_renewable_data_quality(sites_df=valid_sites, profiles=profiles, logger=logger, p_nom_col=p_nom_col)
    generators_to_add = []
    profiles_to_add = {}
    # Initialize with existing network generators to avoid duplicates
    gen_names_used = set(network.generators.index)  # Fast O(1) lookup for duplicate names
    total_sites = len(valid_sites)
    logger.info(f"Processing {total_sites} sites to create generators...")
    
    for i, (idx, site) in enumerate(valid_sites.iterrows()):
        # Progress logging every 500 sites
        if i > 0 and i % 500 == 0:
            logger.info(f"  Processing site {i}/{total_sites} ({100*i/total_sites:.1f}%)")
        
        if 'site_name' in site and pd.notna(site['site_name']):
            base_name = str(site['site_name']).replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            gen_name = base_name
            counter = 1
            # Use set for O(1) lookup instead of O(n) list search
            while gen_name in gen_names_used:
                gen_name = f"{base_name}_{counter}"
                counter += 1
        else:
            gen_name = f"{site[carrier_col]}_{idx}"
        bus = site['bus']
        p_nom = site[p_nom_col]
        carrier = site[carrier_col]
        if gen_name in network.generators.index and not overwrite:
            logger.warning(f"Generator {gen_name} already exists, skipping")
            continue
        if bus not in network.buses.index:
            logger.warning(f"Bus {bus} not found in network, skipping generator {gen_name}")
            continue
        gen_attrs = {'name': gen_name, 'bus': bus, 'p_nom': float(p_nom), 'carrier': carrier}
        if 'x_coord' in site and pd.notna(site['x_coord']):
            gen_attrs['lon'] = float(site['x_coord'])
        if 'y_coord' in site and pd.notna(site['y_coord']):
            gen_attrs['lat'] = float(site['y_coord'])
        elif 'lat' in site and 'lon' in site and pd.notna(site['lat']) and pd.notna(site['lon']):
            gen_attrs['lon'] = float(site['lon'])
            gen_attrs['lat'] = float(site['lat'])
        
        # Look up generator characteristics using carrier name mapping
        # Network uses lowercase_underscore names (e.g., 'wind_offshore')
        # CSV uses Title Case names (e.g., 'Wind Offshore')
        csv_carrier = get_csv_carrier_name(carrier)
        if csv_carrier in generator_characteristics:
            chars = generator_characteristics[csv_carrier]
            gen_attrs.update({
                'marginal_cost': chars.get('marginal_costs', 0.0),
                'capital_cost': 0.0,
                'committable': chars.get('committable', False),
                # IMPORTANT: For renewables, ramp_limit should be NaN (no constraint)
                # PyPSA interprets NaN as "no ramp constraint" - but we must avoid
                # getting a 0 value which would prevent any output change
                # Setting to None here means PyPSA won't create ramp constraints
                'ramp_limit_up': None,  # Renewables: no ramp constraint
                'ramp_limit_down': None,  # Renewables: no ramp constraint
                'p_min_pu': chars.get('p_min_pu', 0.0) / 100.0 if chars.get('p_min_pu') else 0.0,
            })
            for attr in ['min_up_time', 'min_down_time', 'start_up_cost']:
                if attr in chars and pd.notna(chars[attr]) and chars[attr] != '':
                    gen_attrs[attr] = chars[attr]
            logger.debug(f"  Found characteristics for {carrier} (CSV: {csv_carrier})")
        elif carrier in generator_characteristics:
            # Fallback: try direct carrier name match
            chars = generator_characteristics[carrier]
            gen_attrs.update({
                'marginal_cost': chars.get('marginal_costs', 0.0),
                'capital_cost': 0.0,
                'committable': chars.get('committable', False),
                'ramp_limit_up': None,  # Renewables: no ramp constraint
                'ramp_limit_down': None,  # Renewables: no ramp constraint
                'p_min_pu': chars.get('p_min_pu', 0.0) / 100.0 if chars.get('p_min_pu') else 0.0,
            })
            for attr in ['min_up_time', 'min_down_time', 'start_up_cost']:
                if attr in chars and pd.notna(chars[attr]) and chars[attr] != '':
                    gen_attrs[attr] = chars[attr]
            logger.debug(f"  Found characteristics for {carrier} (direct match)")
        else:
            # No characteristics found - use safe defaults for renewables
            gen_attrs.update({
                'marginal_cost': 0.0,
                'capital_cost': 0.0,
                'committable': False,
                'ramp_limit_up': None,  # No ramp constraint
                'ramp_limit_down': None,  # No ramp constraint
            })
            logger.debug(f"  No characteristics found for {carrier}, using defaults")
        generators_to_add.append(gen_attrs)
        if carrier in profiles:
            profile_df = profiles[carrier]
            profile_column = None
            if 'site_name' in site and pd.notna(site['site_name']):
                site_name_clean = str(site['site_name']).strip()
                if site_name_clean in profile_df.columns:
                    profile_column = site_name_clean
                else:
                    close_matches = [col for col in profile_df.columns if site_name_clean.lower() in col.lower()]
                    if close_matches:
                        profile_column = close_matches[0]
            if profile_column is None:
                index_str = str(idx)
                if index_str in profile_df.columns:
                    profile_column = index_str
            if profile_column is None and len(profile_df.columns) == 1:
                profile_column = profile_df.columns[0]
            if profile_column is None:
                default_capacity_factors = {'geothermal': 0.9, 'large_hydro': 0.4, 'small_hydro': 0.3}
                if carrier in default_capacity_factors:
                    cf_value = default_capacity_factors[carrier]
                    if len(network.snapshots) > 0:
                        profile_series = pd.Series(cf_value, index=network.snapshots)
                    else:
                        dummy_index = pd.date_range('2020-01-01', periods=8760, freq='H')
                        profile_series = pd.Series(cf_value, index=dummy_index)
                    logger.debug(f"Created default profile for {gen_name} ({carrier}): CF={cf_value}")
                else:
                    if len(profile_df.columns) > 0:
                        # For unmatched generators aggregating FES capacity, we need fleet-average CF
                        # Sum all site outputs (MW) and divide by sum of max capacities (MW)
                        # This gives the fleet-average capacity factor that reflects actual wind conditions
                        total_fleet_output = profile_df.sum(axis=1)  # Total MW output at each timestep
                        site_capacities = profile_df.max()  # Max output per site = proxy for capacity
                        total_site_capacity = site_capacities.sum()  # Total capacity of all sites
                        if total_site_capacity > 0:
                            # Fleet-average capacity factor
                            profile_series = total_fleet_output / total_site_capacity
                            logger.debug(f"Using fleet-average CF profile for {gen_name} ({carrier}): "
                                        f"mean CF={profile_series.mean():.3f} from {len(profile_df.columns)} sites")
                        else:
                            profile_series = profile_df.mean(axis=1)
                            logger.warning(f"Zero total capacity for {gen_name}, using mean profile")
                    else:
                        continue
            else:
                profile_series = profile_df[profile_column]
                logger.debug(f"Using specific profile column '{profile_column}' for {gen_name}")
            if profile_series.max() > 1.0 and p_nom > 0:
                # Individual site profile in MW - convert to CF by dividing by generator capacity
                profile_series = profile_series / p_nom
                profile_series = profile_series.clip(0, 1)
                logger.debug(f"Converted MW profile to capacity factor for {gen_name}")
            elif profile_series.max() <= 1.0:
                profile_series = profile_series.clip(0, 1)
                logger.debug(f"Using capacity factor profile for {gen_name}")
            else:
                logger.warning(f"Profile values > 1.0 for {gen_name} with zero capacity, using zero profile")
                profile_series = profile_series * 0.0
            if len(network.snapshots) > 0:
                if len(profile_series) == 0:
                    logger.warning(f"Empty profile for {gen_name}, creating zero profile")
                    profile_series = pd.Series(0.0, index=network.snapshots)
                else:
                    try:
                        if not isinstance(profile_series.index, pd.DatetimeIndex):
                            profile_series.index = pd.to_datetime(profile_series.index)
                        if not isinstance(network.snapshots, pd.DatetimeIndex):
                            network.snapshots = pd.to_datetime(network.snapshots)
                        if len(profile_series.index) > 0 and len(network.snapshots) > 0:
                            profile_freq = pd.infer_freq(profile_series.index)
                            network_freq = pd.infer_freq(network.snapshots)
                            logger.debug(f"Profile frequency: {profile_freq}, Network frequency: {network_freq}")
                            if profile_series.index.year[0] != network.snapshots.year[0]:
                                year_offset = network.snapshots.year[0] - profile_series.index.year[0]
                                profile_series.index = profile_series.index + pd.DateOffset(years=year_offset)
                                logger.debug(f"Adjusted profile year by {year_offset} years for {gen_name}")
                            if len(profile_series) != len(network.snapshots):
                                logger.debug(f"Time resolution mismatch for {gen_name}: profile={len(profile_series)}, network={len(network.snapshots)}")
                                if profile_freq != network_freq:
                                    if profile_freq is not None and network_freq is not None:
                                        logger.debug(f"Resampling {gen_name} profile from {profile_freq} to {network_freq}")
                                        profile_series = profile_series.resample(network_freq).mean()
                    except (AttributeError, TypeError, IndexError) as e:
                        logger.warning(f"Could not adjust time index for {gen_name}, proceeding with reindex: {e}")
                    # Remove duplicate timestamps BEFORE reindexing to avoid ValueError
                    if profile_series.index.duplicated().any():
                        logger.debug(f"Removing {profile_series.index.duplicated().sum()} duplicate timestamps from {gen_name} profile")
                        profile_series = profile_series[~profile_series.index.duplicated(keep='first')]
                    try:
                        if len(network.snapshots) > len(profile_series):
                            logger.debug(f"Upsampling {gen_name} from {len(profile_series)} to {len(network.snapshots)} timesteps")
                            profile_series = profile_series.reindex(profile_series.index.union(network.snapshots)).sort_index()
                            profile_series = profile_series.interpolate(method='linear', limit_direction='both')
                            profile_series = profile_series.reindex(network.snapshots, fill_value=0.0)
                        else:
                            profile_series = profile_series.reindex(network.snapshots, method='nearest', fill_value=0.0)
                    except ValueError as e:
                        logger.warning(f"Failed to reindex profile for {gen_name}: {e}")
                        if carrier in ['geothermal']:
                            default_cf = 0.9
                        elif carrier in ['large_hydro', 'small_hydro']:
                            default_cf = 0.4
                        else:
                            default_cf = 0.3
                        profile_series = pd.Series(default_cf, index=network.snapshots)
            if len(profile_series) != len(network.snapshots):
                logger.error(f"Profile length mismatch for {gen_name}: {len(profile_series)} vs {len(network.snapshots)}")
                profile_series = pd.Series(0.3, index=network.snapshots)
            profile_series = profile_series.fillna(0.0).clip(0.0, 1.0)
            profiles_to_add[gen_name] = profile_series
    generators_added = 0
    for gen_attrs in generators_to_add:
        try:
            gen_name = gen_attrs.pop('name')
            network.add("Generator", gen_name, **gen_attrs)
            generators_added += 1
        except Exception as e:
            logger.error(f"Failed to add generator {gen_name}: {e}")
    if profiles_to_add:
        try:
            profiles_df = pd.DataFrame(profiles_to_add)
            if len(profiles_df) > 0:
                existing_profiles = network.generators_t.p_max_pu.copy()
                network.generators_t.p_max_pu = pd.concat([existing_profiles, profiles_df], axis=1)
                max_val = network.generators_t.p_max_pu.max().max()
                if max_val > 1.0:
                    network.generators_t.p_max_pu = network.generators_t.p_max_pu.clip(0.0, 1.0)
                    logger.info(f"Added {len(profiles_to_add)} generator profiles (some values >1.0 were clipped to [0,1])")
                else:
                    logger.info(f"Added {len(profiles_to_add)} generator profiles (values have been converted to capacity factors)")
        except Exception as e:
            logger.error(f"Failed to add time series profiles: {e}")
    logger.info(f"Added {generators_added} renewable generators to network")
    return network

# Initialize logging
logger = setup_logging("integrate_renewable_generators")

# Suppress specific benign warnings emitted by pypsa.networks about missing
# optimization/model/objective when we are only manipulating networks (not
# running an optimization). This avoids cluttering logs with repeated messages.
class _PyPSANetworkWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Only target the pypsa.networks logger
        if not record.name.startswith("pypsa.networks"):
            return True
        msg = record.getMessage()
        suppress_phrases = [
            "The network has not been optimized yet and no model is stored.",
            "The network has not been optimized yet and no objective value is stored.",
            "The network has not been optimized yet and no objective constant is stored.",
        ]
        for p in suppress_phrases:
            if p in msg:
                return False
        return True

# Attach the filter to the pypsa.networks logger
try:
    logging.getLogger("pypsa.networks").addFilter(_PyPSANetworkWarningFilter())
except Exception:
    # Don't fail if logger doesn't exist yet for some reason
    pass


# ══════════════════════════════════════════════════════════════════════════════
# FES RENEWABLE GENERATOR LOADING (FUTURE SCENARIOS)
# ══════════════════════════════════════════════════════════════════════════════

# FES data structure for renewables (2024 format):
# - Offshore Wind: Technology='Wind' + Technology Detail='Offshore Wind'
# - Onshore Wind: Technology=NaN + Technology Detail='Onshore Wind >=1MW' or '<1MW'
# - Solar: Technology='Solar Generation' + Technology Detail='Large (G99)' or 'Small (G98/G83)'
# - Marine: Technology='Marine' + Technology Detail='Tidal Stream, Wave Power, Tidal Lagoon'
# - Hydro: Technology='Hydro' + Technology Detail='Not pumped hydro'
# - Off-grid offshore: Technology='Offshore-Wind (off-Grid)' (small capacity)

# Primary mapping: Technology column -> carrier (for most renewables)
# NOTE: 'Wind' is intentionally NOT mapped here because it contains BOTH onshore and offshore wind.
# The carrier is determined by the Technology Detail column (handled in FES_TECHNOLOGY_DETAIL_MAP).
# 'Offshore-Wind (off-Grid)' is EXCLUDED - it's not grid-connected generation (Unit='Number', not MW).
FES_RENEWABLE_TECHNOLOGY_MAP = {
    'Solar Generation': 'solar_pv',
    # 'Wind' is handled via Technology Detail column to distinguish onshore vs offshore
    'Marine': 'marine',
    'Hydro': 'large_hydro',  # FES Hydro is large-scale (not run-of-river)
}

# Secondary mapping: Technology Detail column -> carrier (for records where Technology is NaN)
# This captures onshore wind which has Technology=NaN in FES data
FES_TECHNOLOGY_DETAIL_MAP = {
    'Onshore Wind >=1MW': 'wind_onshore',
    'Onshore Wind <1MW': 'wind_onshore',
    'Offshore Wind': 'wind_offshore',  # Backup if matching by Technology Detail
    'Large (G99)': 'solar_pv',  # Large-scale solar (covered by Technology mapping)
    'Small (G98/G83)': 'solar_pv',  # Small-scale solar (may have Technology=NaN)
    'Domestic (G98/G83)': 'solar_pv',  # Domestic solar
    'Tidal Stream, Wave Power, Tidal Lagoon': 'marine',
    'Not pumped hydro': 'large_hydro',
}

# NESO API endpoints for GSP info by FES year
NESO_GSP_INFO_URLS = {
    2024: 'https://api.neso.energy/dataset/963525d6-5d83-4448-a99c-663f1c76330a/resource/21c2b09c-24ff-4837-a3b1-b6aea88f8124/download/fes2024_regional_breakdown_gsp_info.csv',
    2023: 'https://api.neso.energy/dataset/963525d6-5d83-4448-a99c-663f1c76330a/resource/805e20e4-6a8b-4113-9d2d-09efa0a4bbb9/download/fes2023_regional_breakdown_gsp_info.csv',
    2022: 'https://api.neso.energy/dataset/963525d6-5d83-4448-a99c-663f1c76330a/resource/000d08b9-12d9-4396-95f8-6b3677664836/download/fes2022_regional_breakdown_gsp_info.csv',
    2021: 'https://api.neso.energy/dataset/963525d6-5d83-4448-a99c-663f1c76330a/resource/41fb4ca1-7b59-4fce-b480-b46682f346c9/download/fes2021_regional_breakdown_gsp_info.csv',
}


def _download_gsp_info_from_neso(fes_year: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Download GSP info from NESO API for the specified FES year.
    
    This provides the mapping from GSP names to coordinates.
    
    Args:
        fes_year: FES publication year (2021, 2022, 2023, 2024)
        logger: Logger instance
        
    Returns:
        DataFrame with columns: GSP ID, GSP Group, Minor FLOP, Name, Latitude, Longitude
    """
    import urllib3
    import io
    
    url = NESO_GSP_INFO_URLS.get(fes_year)
    if not url:
        # Default to latest available
        url = NESO_GSP_INFO_URLS[2024]
        logger.warning(f"No GSP info URL for FES {fes_year}, using FES 2024")
    
    try:
        http = urllib3.PoolManager()
        response = http.request('GET', url, timeout=30)
        
        if response.status != 200:
            logger.error(f"Failed to download GSP info: HTTP {response.status}")
            return pd.DataFrame()
        
        # Use utf-8-sig to handle BOM
        gsp_df = pd.read_csv(io.BytesIO(response.data), encoding='utf-8-sig')
        logger.info(f"Downloaded GSP info from NESO API: {len(gsp_df)} entries")
        return gsp_df
        
    except Exception as e:
        logger.error(f"Error downloading GSP info from NESO: {e}")
        return pd.DataFrame()


def _load_historical_renewable_sites(technology: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load historical renewable site data to use for spatial distribution.
    
    Uses the most recent data from REPD-based site files to determine
    where renewable capacity should be located.
    
    Args:
        technology: One of 'wind_offshore', 'wind_onshore', 'solar_pv', 'large_hydro', 'marine'
        logger: Logger instance
        
    Returns:
        DataFrame with columns: site_name, capacity_mw, lat, lon
    """
    site_files = {
        'wind_offshore': 'resources/renewable/wind_offshore_sites.csv',
        'wind_onshore': 'resources/renewable/wind_onshore_sites.csv',
        'solar_pv': 'resources/renewable/solar_pv_sites.csv',
        'large_hydro': 'resources/renewable/large_hydro_sites.csv',
        'marine': 'resources/renewable/tidal_stream_sites.csv',  # Use tidal as proxy
    }
    
    file_path = site_files.get(technology)
    if not file_path or not Path(file_path).exists():
        logger.warning(f"No historical site data found for {technology}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'capacity_mw' not in df.columns and 'Installed Capacity (MWelec)' in df.columns:
            df['capacity_mw'] = df['Installed Capacity (MWelec)']
        
        # Keep only required columns
        required = ['site_name', 'capacity_mw', 'lat', 'lon']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns {missing} in {file_path}")
            return pd.DataFrame()
        
        df = df[required].dropna()
        logger.debug(f"Loaded {len(df)} historical {technology} sites")
        return df
        
    except Exception as e:
        logger.warning(f"Error loading historical sites for {technology}: {e}")
        return pd.DataFrame()


def _distribute_direct_capacity_using_historical(
    capacity_mw: float,
    technology: str,
    region: str,
    network: pypsa.Network,
    logger: logging.Logger
) -> list:
    """
    Distribute 'Direct' connected capacity using historical site locations.
    
    For offshore wind, uses existing offshore wind farm locations.
    For other technologies, uses appropriate historical data.
    
    Args:
        capacity_mw: Total capacity to distribute (MW)
        technology: Technology type (wind_onshore, solar_pv, etc.)
        region: NGET, SHETL, or SPTL
        network: PyPSA network for bus mapping
        logger: Logger instance
        
    Returns:
        List of dicts with keys: bus, lat, lon, capacity_mw
    """
    # Load historical site data
    historical_sites = _load_historical_renewable_sites(technology, logger)
    
    if len(historical_sites) == 0:
        logger.warning(f"No historical sites for {technology}, using geographic distribution")
        return _distribute_by_region_geography(capacity_mw, region, network, logger)
    
    # Filter by region (approximate latitude bands)
    if region == 'SHETL':
        sites = historical_sites[historical_sites['lat'] > 57.0]
    elif region == 'SPTL':
        sites = historical_sites[(historical_sites['lat'] > 55.5) & (historical_sites['lat'] <= 57.0)]
    else:  # NGET
        sites = historical_sites[historical_sites['lat'] <= 55.5]
    
    if len(sites) == 0:
        # Fall back to all sites in the technology
        sites = historical_sites
        logger.debug(f"No {technology} sites in {region}, using all sites")
    
    if len(sites) == 0:
        return _distribute_by_region_geography(capacity_mw, region, network, logger)
    
    # Distribute capacity proportionally to existing site capacities
    total_historical_capacity = sites['capacity_mw'].sum()
    
    # First, collect all capacity going to each bus
    bus_capacity = {}  # {bus: {'capacity_mw': total, 'lat': avg_lat, 'lon': avg_lon}}
    
    for _, site in sites.iterrows():
        # Scale capacity proportionally
        site_capacity = (site['capacity_mw'] / total_historical_capacity) * capacity_mw
        
        # Find nearest bus
        lat, lon = site['lat'], site['lon']
        bus = _find_nearest_bus(network, lat, lon)
        
        if bus:
            if bus not in bus_capacity:
                bus_capacity[bus] = {'capacity_mw': 0.0, 'lat': lat, 'lon': lon, 'site_count': 0}
            bus_capacity[bus]['capacity_mw'] += site_capacity
            bus_capacity[bus]['site_count'] += 1
            # Average coordinates for display purposes
            n = bus_capacity[bus]['site_count']
            bus_capacity[bus]['lat'] = (bus_capacity[bus]['lat'] * (n-1) + lat) / n
            bus_capacity[bus]['lon'] = (bus_capacity[bus]['lon'] * (n-1) + lon) / n
    
    # Convert to list format
    generators = [
        {'bus': bus, 'lat': info['lat'], 'lon': info['lon'], 'capacity_mw': info['capacity_mw']}
        for bus, info in bus_capacity.items()
    ]
    
    return generators


def _distribute_by_region_geography(
    capacity_mw: float,
    region: str,
    network: pypsa.Network,
    logger: logging.Logger
) -> list:
    """
    Fallback: distribute capacity evenly across buses in a geographic region.
    """
    buses = network.buses.copy()
    
    # Filter by region
    if region == 'SHETL':
        region_buses = buses[buses['y'] > 57.0]
    elif region == 'SPTL':
        region_buses = buses[(buses['y'] > 55.5) & (buses['y'] <= 57.0)]
    else:  # NGET
        region_buses = buses[buses['y'] <= 55.5]
    
    if len(region_buses) == 0:
        region_buses = buses
    
    # Sample up to 20 buses for distribution
    if len(region_buses) > 20:
        region_buses = region_buses.sample(20)
    
    capacity_per_bus = capacity_mw / len(region_buses)
    
    generators = []
    for bus_name, bus in region_buses.iterrows():
        generators.append({
            'bus': bus_name,
            'lat': bus['y'],
            'lon': bus['x'],
            'capacity_mw': capacity_per_bus
        })
    
    return generators


def _find_nearest_bus(network: pypsa.Network, lat: float, lon: float) -> Optional[str]:
    """
    Find the nearest network bus to a given WGS84 coordinate.
    
    Handles coordinate system conversion between WGS84 (lat/lon) and 
    OSGB36 (British National Grid in meters).
    """
    if len(network.buses) == 0:
        return None
    
    buses = network.buses[['x', 'y']].copy()
    
    # Detect if network is in OSGB36 (meters) or WGS84 (degrees)
    x_range = buses['x'].max() - buses['x'].min()
    is_osgb36 = x_range > 1000  # OSGB36 has range of ~600,000 meters, WGS84 ~10 degrees
    
    if is_osgb36:
        # Convert WGS84 site coordinates to OSGB36 before distance calculation
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            site_x, site_y = transformer.transform(lon, lat)
            
            # Calculate Euclidean distance in meters
            buses['dist'] = np.sqrt((buses['x'] - site_x)**2 + (buses['y'] - site_y)**2)
        except ImportError:
            # Fallback: use haversine with converted bus coordinates
            # This is less accurate but works without pyproj
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
            bus_lons, bus_lats = transformer.transform(buses['x'].values, buses['y'].values)
            
            # Haversine distance
            R = 6371.0  # Earth radius in km
            lat1, lat2 = np.radians(lat), np.radians(bus_lats)
            lon1, lon2 = np.radians(lon), np.radians(bus_lons)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            buses['dist'] = R * c
    else:
        # WGS84 network - use haversine distance
        R = 6371.0  # Earth radius in km
        lat1, lat2 = np.radians(lat), np.radians(buses['y'].values)
        lon1, lon2 = np.radians(lon), np.radians(buses['x'].values)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        buses['dist'] = R * c
    
    return buses['dist'].idxmin()


def _normalize_gsp_name(name: str) -> str:
    """
    Normalize GSP name for fuzzy matching.
    
    Handles common variations:
    - Apostrophes: "Connah's Quay" vs "Connahs Quay"
    - Parentheticals: "Lodge Road (St Johns Wood)" vs "Lodge Road"
    - Case variations
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Remove apostrophes
    normalized = normalized.replace("'", "")
    
    # Remove content in parentheses for matching
    import re
    base_name = re.sub(r'\s*\([^)]*\)\s*', '', normalized).strip()
    
    return base_name


def _fuzzy_gsp_lookup(gsp_name: str, gsp_to_bus: dict) -> Optional[dict]:
    """
    Try to find a GSP in the mapping using fuzzy matching.
    
    Args:
        gsp_name: GSP name from FES data
        gsp_to_bus: Dictionary of GSP mappings
        
    Returns:
        Bus info dict if found, None otherwise
    """
    if not gsp_name or not gsp_to_bus:
        return None
    
    # First try exact match
    if gsp_name in gsp_to_bus:
        return gsp_to_bus[gsp_name]
    
    # Normalize the input name
    normalized_input = _normalize_gsp_name(gsp_name)
    
    # Build normalized lookup if not already cached
    normalized_lookup = {}
    for key, value in gsp_to_bus.items():
        normalized_key = _normalize_gsp_name(key)
        if normalized_key not in normalized_lookup:
            normalized_lookup[normalized_key] = value
    
    # Try normalized match
    if normalized_input in normalized_lookup:
        return normalized_lookup[normalized_input]
    
    # Try partial matching for names with parentheses
    # e.g., "Lodge Road (St Johns Wood)" should match "Lodge Road"
    for key, value in gsp_to_bus.items():
        normalized_key = _normalize_gsp_name(key)
        # Check if input contains the key (partial match)
        if normalized_key and normalized_key in normalized_input:
            return value
        # Check if key contains the input
        if normalized_input and normalized_input in normalized_key:
            return value
    
    return None


def load_fes_renewable_generators(
    fes_path: str,
    modelled_year: int,
    fes_scenario: str,
    network: pypsa.Network,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load renewable generator capacity from FES data for future scenarios.
    
    FES provides aggregate capacity projections by GSP and technology.
    This function:
    1. Loads FES data for the specified year and scenario
    2. Filters to renewable technologies using BOTH Technology and Technology Detail columns
    3. Maps GSP locations to network buses
    4. Returns a DataFrame in the same format as REPD site data
    
    FES 2024 Data Structure:
    - Offshore Wind: Technology='Wind' + Technology Detail='Offshore Wind'
    - Onshore Wind: Technology=NaN + Technology Detail='Onshore Wind >=1MW' or '<1MW'
    - Solar: Technology='Solar Generation' + Technology Detail='Large (G99)' or 'Small (G98/G83)'
    - Marine: Technology='Marine'
    - Hydro: Technology='Hydro' + Technology Detail='Not pumped hydro'
    
    Args:
        fes_path: Path to FES_{year}_data.csv
        modelled_year: Target year for capacity extraction (e.g., 2035)
        fes_scenario: FES pathway name (e.g., 'Holistic Transition')
        network: PyPSA Network for bus mapping
        logger: Logger instance
        
    Returns:
        DataFrame with columns: technology, capacity_mw, bus, lat, lon, site_name
    """
    if not os.path.exists(fes_path):
        logger.error(f"FES file not found: {fes_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading FES renewable capacity for year {modelled_year}")
    logger.info(f"  FES file: {fes_path}")
    
    # Read FES data with proper encoding for BOM
    df = pd.read_csv(fes_path, encoding='utf-8-sig')
    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Filter to renewable technologies using BOTH Technology and Technology Detail
    # ══════════════════════════════════════════════════════════════════════════
    
    # Method 1: Match by Technology column (Solar, Marine, Hydro, Wind/Offshore-Wind)
    renewable_techs = list(FES_RENEWABLE_TECHNOLOGY_MAP.keys())
    gen_blocks_by_tech = df[df['Technology'].isin(renewable_techs)].copy()
    logger.info(f"  Found {len(gen_blocks_by_tech)} records matching Technology column")
    
    # Method 2: Match by Technology Detail column (critical for Onshore Wind which has Technology=NaN)
    renewable_details = list(FES_TECHNOLOGY_DETAIL_MAP.keys())
    gen_blocks_by_detail = df[df['Technology Detail'].isin(renewable_details)].copy()
    # Exclude records already captured by Technology matching to avoid duplicates
    gen_blocks_by_detail = gen_blocks_by_detail[~gen_blocks_by_detail.index.isin(gen_blocks_by_tech.index)]
    logger.info(f"  Found {len(gen_blocks_by_detail)} additional records matching Technology Detail column")
    
    # Combine both sets
    gen_blocks = pd.concat([gen_blocks_by_tech, gen_blocks_by_detail], ignore_index=True)
    logger.info(f"  Total renewable records: {len(gen_blocks)}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Filter to selected FES scenario
    # ══════════════════════════════════════════════════════════════════════════
    
    # Check available pathways
    available_pathways = gen_blocks['FES Pathway'].unique()
    logger.info(f"  Available FES pathways: {list(available_pathways)}")
    
    # Select FES scenario
    if fes_scenario is None or fes_scenario not in available_pathways:
        if fes_scenario is not None:
            logger.warning(f"  Scenario '{fes_scenario}' not found. Using first available.")
        fes_scenario = available_pathways[0] if len(available_pathways) > 0 else None
        if fes_scenario is None:
            logger.error("No FES pathways found in data")
            return pd.DataFrame()
    
    logger.info(f"  Using FES scenario: {fes_scenario}")
    gen_blocks = gen_blocks[gen_blocks['FES Pathway'] == fes_scenario].copy()
    
    # Check if year column exists
    year_col = str(modelled_year)
    available_years = [c for c in df.columns if c.isdigit()]
    if year_col not in df.columns:
        logger.error(f"Year {modelled_year} not in FES data. Available: {available_years}")
        return pd.DataFrame()
    
    # Load GSP-to-bus mapping from NESO API (or fallback to local files)
    gsp_to_bus = _load_gsp_bus_mapping(network, logger)
    
    # Extract generators with non-zero capacity
    generators = []
    tech_summary = {}
    unmatched_summary = {}  # Track unmatched capacity by GSP
    
    for _, row in gen_blocks.iterrows():
        capacity = row[year_col]
        
        if pd.isna(capacity) or capacity <= 0:
            continue
        
        # ══════════════════════════════════════════════════════════════════════
        # Determine carrier using BOTH Technology and Technology Detail columns
        # Priority: Technology Detail (more specific) > Technology (general)
        # ══════════════════════════════════════════════════════════════════════
        fes_tech = row.get('Technology', None)
        fes_tech_detail = row.get('Technology Detail', None)
        
        # First, try Technology Detail (more specific, handles onshore wind)
        carrier = None
        if pd.notna(fes_tech_detail):
            carrier = FES_TECHNOLOGY_DETAIL_MAP.get(str(fes_tech_detail).strip(), None)
        
        # If no match from detail, try Technology column
        if carrier is None and pd.notna(fes_tech):
            carrier = FES_RENEWABLE_TECHNOLOGY_MAP.get(str(fes_tech).strip(), None)
        
        # Skip if still no carrier match (shouldn't happen with proper filtering)
        if carrier is None:
            logger.debug(f"Skipping unmatched record: Tech='{fes_tech}', Detail='{fes_tech_detail}'")
            continue
        
        gsp = row.get('GSP', None)
        
        # Map GSP to bus using fuzzy matching
        bus = None
        lat, lon = None, None
        
        if pd.notna(gsp):
            # Use fuzzy matching to handle name variations
            bus_info = _fuzzy_gsp_lookup(gsp, gsp_to_bus)
            if bus_info:
                bus = bus_info.get('bus')
                lat = bus_info.get('lat')
                lon = bus_info.get('lon')
        
        # Handle "Direct" transmission-connected generators and "Not Connected"
        # Distribute using historical site locations for spatial accuracy
        if bus is None and pd.notna(gsp):
            # Handle Direct connections and "Not Connected" category
            if 'Direct' in str(gsp) or str(gsp).lower() == 'not connected':
                # Extract region from GSP name (NGET, SHETL, SPTL)
                if 'SHETL' in str(gsp):
                    region = 'SHETL'
                elif 'SPTL' in str(gsp):
                    region = 'SPTL'
                else:
                    region = 'NGET'  # Default for "Not Connected" and NGET Direct
                
                # Use historical site data to distribute capacity spatially
                distributed_gens = _distribute_direct_capacity_using_historical(
                    capacity_mw=float(capacity),
                    technology=carrier,
                    region=region,
                    network=network,
                    logger=logger
                )
                
                for idx, gen_info in enumerate(distributed_gens):
                    gen = {
                        'technology': carrier,
                        'capacity_mw': gen_info['capacity_mw'],
                        'bus': gen_info['bus'],
                        'lat': gen_info['lat'],
                        'lon': gen_info['lon'],
                        'site_name': f"FES_{carrier}_{gsp}_{gen_info['bus']}_{modelled_year}",
                        'data_source': 'FES_Direct',
                        'gsp': gsp
                    }
                    generators.append(gen)
                    if carrier not in tech_summary:
                        tech_summary[carrier] = {'count': 0, 'capacity': 0.0}
                    tech_summary[carrier]['count'] += 1
                    tech_summary[carrier]['capacity'] += gen_info['capacity_mw']
                
                continue  # Already added, skip normal processing
            
            # Track unmatched GSPs for logging
            if gsp not in unmatched_summary:
                unmatched_summary[gsp] = 0.0
            unmatched_summary[gsp] += float(capacity)
            continue  # Skip unmapped GSPs
        
        if bus is None:
            continue  # Skip if still no bus mapping
        
        gen = {
            'technology': carrier,
            'capacity_mw': float(capacity),
            'bus': bus,
            'lat': lat,
            'lon': lon,
            'site_name': f"FES_{carrier}_{gsp}_{modelled_year}",
            'data_source': 'FES',
            'gsp': gsp
        }
        generators.append(gen)
        
        # Track summary
        if carrier not in tech_summary:
            tech_summary[carrier] = {'count': 0, 'capacity': 0.0}
        tech_summary[carrier]['count'] += 1
        tech_summary[carrier]['capacity'] += float(capacity)
    
    fes_df = pd.DataFrame(generators)
    
    # ══════════════════════════════════════════════════════════════════════════
    # AGGREGATE DUPLICATES: Same GSP can have multiple Technology Details (e.g., Large/Small/Domestic solar)
    # that all map to the same carrier. Aggregate by (technology, bus, site_name) to avoid duplicate generators
    # while preserving spatial distribution for Direct generators (which have unique site_names per bus).
    # ══════════════════════════════════════════════════════════════════════════
    if len(fes_df) > 0:
        before_count = len(fes_df)
        # Aggregate capacity for duplicate (technology, bus, site_name) combinations
        # Using site_name instead of gsp to preserve Direct generator distribution
        agg_cols = ['technology', 'bus', 'site_name']
        if all(col in fes_df.columns for col in agg_cols):
            fes_df = fes_df.groupby(agg_cols, as_index=False).agg({
                'capacity_mw': 'sum',
                'lat': 'first',
                'lon': 'first',
                'gsp': 'first',
                'data_source': 'first'
            })
            after_count = len(fes_df)
            if before_count > after_count:
                logger.info(f"  Aggregated {before_count} entries to {after_count} unique (technology, bus, site_name) combinations")
    
    if len(fes_df) > 0:
        total_capacity = fes_df['capacity_mw'].sum()
        logger.info(f"Loaded {len(fes_df)} FES renewable entries for {modelled_year}")
        logger.info(f"  Total capacity: {total_capacity:,.1f} MW")
        logger.info(f"  Technology breakdown:")
        for tech, stats in sorted(tech_summary.items(), key=lambda x: x[1]['capacity'], reverse=True):
            logger.info(f"    {tech}: {stats['count']} entries, {stats['capacity']:,.1f} MW")
        
        # Log unmatched GSPs if any significant capacity is missing
        if unmatched_summary:
            total_unmatched = sum(unmatched_summary.values())
            logger.warning(f"  Unmatched GSPs: {len(unmatched_summary)} GSPs, {total_unmatched:,.1f} MW total")
            for gsp, cap in sorted(unmatched_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.warning(f"    {gsp}: {cap:,.1f} MW")
    else:
        logger.warning(f"No FES renewable generators found for year {modelled_year}")
    
    return fes_df


def _load_gsp_bus_mapping(network: pypsa.Network, logger: logging.Logger, fes_year: int = 2024) -> dict:
    """
    Load mapping from FES GSP names to network buses.
    
    Downloads GSP info from NESO API, which contains human-readable GSP names
    that match the FES data format (e.g., 'Bredbury', 'Harker', 'Heysham (NGET)').
    
    Falls back to local files if API download fails.
    
    Args:
        network: PyPSA network for bus coordinate matching
        logger: Logger instance
        fes_year: FES publication year (2021-2024) for API endpoint
        
    Returns:
        Dictionary: {gsp_name: {'bus': bus_name, 'lat': lat, 'lon': lon}}
    """
    gsp_mapping = {}
    
    # First, try to download from NESO API
    gsp_df = _download_gsp_info_from_neso(fes_year, logger)
    
    if len(gsp_df) > 0 and 'Name' in gsp_df.columns:
        # Use NESO API data
        name_col, lat_col, lon_col = 'Name', 'Latitude', 'Longitude'
        
        for _, row in gsp_df.iterrows():
            gsp_name = row[name_col]
            lat = row[lat_col]
            lon = row[lon_col]
            
            if pd.isna(gsp_name) or pd.isna(lat) or pd.isna(lon):
                continue
            
            # Find nearest bus with proper coordinate system handling
            # Uses _find_nearest_bus which handles WGS84 to OSGB36 conversion
            if len(network.buses) > 0:
                nearest_bus = _find_nearest_bus(network, lat, lon)
                
                if nearest_bus is not None:
                    gsp_mapping[gsp_name] = {
                        'bus': nearest_bus,
                        'lat': lat,
                        'lon': lon
                    }
        
        logger.info(f"Loaded GSP-to-bus mapping: {len(gsp_mapping)} GSPs from NESO API")
        return gsp_mapping
    
    # Fallback: Use local static files
    logger.warning("NESO API download failed, falling back to local GSP files")
    
    gsp_lookup_paths = [
        # Primary: GSP_data.csv has 'Name', 'Latitude', 'Longitude' columns
        (Path("data/FES/FES2022/GSP_data.csv"), 'Name', 'Latitude', 'Longitude', 'latin-1'),
        # Fallback: gsp_gnode lookup has technical IDs
        (Path("data/FES/FES2022/gsp_gnode_directconnect_region_lookup.csv"), 'gsp_name', 'gsp_lat', 'gsp_lon', 'utf-8'),
    ]
    
    for lookup_path, name_col, lat_col, lon_col, encoding in gsp_lookup_paths:
        if lookup_path.exists():
            try:
                df = pd.read_csv(lookup_path, encoding=encoding)
                
                if name_col in df.columns and lat_col in df.columns and lon_col in df.columns:
                    mapped_count = 0
                    for _, row in df.iterrows():
                        gsp_name = row[name_col]
                        lat = row[lat_col]
                        lon = row[lon_col]
                        
                        if pd.isna(gsp_name) or pd.isna(lat) or pd.isna(lon):
                            continue
                        
                        # Find nearest bus with proper coordinate system handling
                        if len(network.buses) > 0:
                            nearest_bus = _find_nearest_bus(network, lat, lon)
                            
                            if nearest_bus is not None:
                                gsp_mapping[gsp_name] = {
                                    'bus': nearest_bus,
                                    'lat': lat,
                                    'lon': lon
                                }
                                mapped_count += 1
                    
                    logger.info(f"Loaded GSP-to-bus mapping: {mapped_count} GSPs from {lookup_path}")
                    if mapped_count > 300:
                        break
                    
            except Exception as e:
                logger.warning(f"Could not load GSP lookup from {lookup_path}: {e}")
    
    if len(gsp_mapping) == 0:
        logger.warning("No GSP-to-bus mapping loaded! FES renewable integration will fail.")
    
    return gsp_mapping


def main():
    """Main execution function for renewable generator integration."""
    global logger
    start_time = time.time()
    stage_times = {}  # Track timing for each major stage
    
    # Reinitialize logger with Snakemake log path if available
    snk = globals().get('snakemake')
    if snk and hasattr(snk, 'log') and snk.log:
        logger = setup_logging(snk.log[0])
    
    logger.info("=" * 80)
    logger.info("RENEWABLE GENERATOR INTEGRATION")
    logger.info("=" * 80)
    logger.info("Adding weather-variable renewable generators to network")
    
    try:
        # Access snakemake variables
        if not snk:
            raise RuntimeError("This script must be run via Snakemake")
        
        # Get scenario configuration for year-based filtering
        scenario_name = snk.wildcards.scenario
        scenario_config = snk.params.scenario_config
        modelled_year = scenario_config.get('modelled_year', None)
        is_historical = snk.params.get('is_historical', True)  # Default to historical for safety
        
        logger.info(f"Scenario: {scenario_name}")
        if modelled_year:
            logger.info(f"Modelled Year: {modelled_year}")
        logger.info(f"Scenario Type: {'HISTORICAL (REPD sites)' if is_historical else 'FUTURE (FES projections)'}")
        
        # STAGE 1: Load network
        stage_start = time.time()
        network_path = snk.input.network
        logger.info(f"Loading network from {network_path}")
        network = load_network(network_path, custom_logger=logger)
        logger.info("Input network (with base demand)")
        log_network_info(network, logger)
        stage_times['1. Load network'] = time.time() - stage_start
        
        # STAGE 2: Add carrier definitions
        stage_start = time.time()
        logger.info("Adding carrier definitions to network")
        network = add_carriers_to_network(network, logger)
        stage_times['2. Add carrier definitions'] = time.time() - stage_start
        
        # STAGE 3: Load renewable data (different paths for historical vs future)
        stage_start = time.time()
        logger.info("-" * 80)
        if is_historical:
            logger.info("LOADING RENEWABLE SITE DATA (HISTORICAL - REPD)")
        else:
            logger.info("LOADING RENEWABLE CAPACITY DATA (FUTURE - FES)")
        logger.info("-" * 80)
        
        if is_historical:
            # ================================================================
            # HISTORICAL PATH: Load individual sites from REPD
            # ================================================================
            renewable_site_files = {
                'wind_onshore': snk.input.wind_onshore_sites,
                'wind_offshore': snk.input.wind_offshore_sites,
                'solar_pv': snk.input.solar_pv_sites,
                'small_hydro': snk.input.small_hydro_sites,
                'large_hydro': snk.input.large_hydro_sites,
                'tidal_stream': snk.input.tidal_stream_sites,
                'shoreline_wave': snk.input.shoreline_wave_sites,
                'tidal_lagoon': snk.input.tidal_lagoon_sites
            }
            
            # Determine if year-based filtering is needed (historical scenarios only)
            current_year = pd.Timestamp.today().year
            apply_year_filter = modelled_year is not None and modelled_year <= current_year
            if apply_year_filter:
                logger.info(f"Historical scenario - filtering renewable sites to operational by {modelled_year}")
            else:
                logger.info("Future scenario or no modelled_year - using all available sites")
            
            renewable_sites_list = []
            for technology, site_file in renewable_site_files.items():
                if os.path.exists(site_file):
                    logger.info(f"Loading {technology} sites from {site_file}")
                    sites_df = pd.read_csv(site_file)
                    
                    if len(sites_df) > 0:
                        # Add technology column
                        sites_df['technology'] = technology
                        
                        # Apply year-based filtering for historical scenarios
                        if apply_year_filter:
                            sites_df = filter_sites_by_year(sites_df, modelled_year, logger)
                        
                        if len(sites_df) > 0:
                            renewable_sites_list.append(sites_df)
                            log_dataframe_info(sites_df, logger, f"{technology} sites (after filtering)")
                        else:
                            logger.info(f"  No {technology} sites operational by {modelled_year}")
                    else:
                        logger.info(f"  No sites found in {technology} file")
                else:
                    logger.warning(f"Site file not found: {site_file}")
            
            stage_times['3. Load renewable site data (REPD)'] = time.time() - stage_start
            
            if not renewable_sites_list:
                logger.warning("No renewable site files found or all files are empty")
                # Save network anyway (no changes) and create empty summary
                save_network(network, snk.output.network, custom_logger=logger)
                pd.DataFrame(columns=['technology', 'capacity_mw', 'count']).to_csv(snk.output.summary, index=False)
                log_stage_summary(stage_times, logger, "RENEWABLE INTEGRATION - STAGE TIMING")
                return
            
            # Combine all renewable sites
            renewable_sites = pd.concat(renewable_sites_list, ignore_index=True)
            
        else:
            # ================================================================
            # FUTURE PATH: Load aggregate capacity from FES
            # ================================================================
            renewable_sites = load_fes_renewable_generators(
                snk.input.fes_data,
                modelled_year,
                scenario_config.get('FES_scenario', None),
                network,
                logger
            )
            stage_times['3. Load renewable capacity data (FES)'] = time.time() - stage_start
            
            if len(renewable_sites) == 0:
                logger.warning("No FES renewable capacity found for this scenario")
                save_network(network, snk.output.network, custom_logger=logger)
                pd.DataFrame(columns=['technology', 'capacity_mw', 'count']).to_csv(snk.output.summary, index=False)
                log_stage_summary(stage_times, logger, "RENEWABLE INTEGRATION - STAGE TIMING")
                return
        
        # STAGE 4: Combine/summarize sites
        stage_start = time.time()
        total_sites = len(renewable_sites)
        total_capacity = renewable_sites['capacity_mw'].sum() if 'capacity_mw' in renewable_sites.columns else 0
        logger.info("-" * 80)
        logger.info(f"COMBINED RENEWABLE DATA: {total_sites} entries, {total_capacity:.2f} MW total")
        logger.info("-" * 80)
        log_dataframe_info(renewable_sites, logger, "Combined renewable data")
        stage_times['4. Combine data'] = time.time() - stage_start
        
        # STAGE 5: Map sites to network buses
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("MAPPING SITES TO NETWORK BUSES")
        logger.info("-" * 80)
        
        # Adjust max_distance based on network sparsity
        # Reduced networks (29 buses) need larger distances than ETYS (400 buses)
        n_buses = len(network.buses)
        if n_buses < 50:  # Sparse network (e.g., Reduced, Zonal)
            max_distance_km = 1000.0  # Allow large distances for sparse networks (use fallback mapping)
            logger.info(f"Using large distance threshold ({max_distance_km}km) for sparse network ({n_buses} buses)")
        else:  # Dense network (e.g., ETYS)
            max_distance_km = 200.0  # Normal threshold for dense networks
            logger.info(f"Using normal distance threshold ({max_distance_km}km) for dense network ({n_buses} buses)")
        
        renewable_sites = map_sites_to_buses(
            network, 
            renewable_sites,
            method='nearest',
            lon_col='lon',  # Use WGS84 longitude (consistent with network.buses['x'])
            lat_col='lat',  # Use WGS84 latitude (consistent with network.buses['y'])
            max_distance_km=max_distance_km
        )
        
        mapped_sites = renewable_sites['bus'].notna().sum()
        logger.info(f"Successfully mapped {mapped_sites}/{total_sites} sites to buses")
        stage_times['5. Map sites to buses'] = time.time() - stage_start
        
        # STAGE 5.5: Apply ETYS BMU-to-Node mapping for ETYS networks
        # This moves large renewable generators (e.g., offshore wind) from 132kV to 400kV buses
        network_model = scenario_config.get('network_model', 'reduced')
        
        if network_model.upper() == 'ETYS':
            stage_start_bmu = time.time()
            logger.info("-" * 80)
            logger.info("APPLYING ETYS BMU-TO-NODE BUS CORRECTIONS FOR RENEWABLES")
            logger.info("-" * 80)
            logger.info("Large renewable generators (e.g., offshore wind) should connect to 400kV buses")
            
            renewable_sites = apply_etys_bmu_mapping(renewable_sites, network)
            
            stage_times['5.5. ETYS BMU mapping'] = time.time() - stage_start_bmu
        
        # STAGE 6: Add renewable generators to network
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("INTEGRATING RENEWABLE GENERATORS INTO NETWORK")
        logger.info("-" * 80)
        
        profiles_dir = "resources/renewable/profiles"
        logger.info(f"Using profiles directory: {profiles_dir}")
        
        initial_gen_count = len(network.generators)
        
        network = add_renewable_generators(
            network, 
            renewable_sites,
            profiles_dir=profiles_dir,
            p_nom_col='capacity_mw'
        )
        
        final_gen_count = len(network.generators)
        added_generators = final_gen_count - initial_gen_count
        
        logger.info("-" * 80)
        logger.info(f"RENEWABLE INTEGRATION COMPLETE")
        logger.info(f"  Generators before: {initial_gen_count}")
        logger.info(f"  Generators after: {final_gen_count}")
        logger.info(f"  Renewable generators added: {added_generators}")
        logger.info("-" * 80)
        stage_times['6. Add generators to network'] = time.time() - stage_start
        
        # STAGE 7: Create summary by technology
        stage_start = time.time()
        summary_data = []
        for tech in renewable_sites['technology'].unique():
            tech_gens = network.generators[network.generators['carrier'] == tech]
            if len(tech_gens) > 0:
                summary_data.append({
                    'technology': tech,
                    'capacity_mw': tech_gens['p_nom'].sum(),
                    'count': len(tech_gens)
                })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info("Renewable capacity summary:")
        for _, row in summary_df.iterrows():
            logger.info(f"  {row['technology']}: {row['capacity_mw']:.2f} MW ({row['count']} units)")
        stage_times['7. Generate summary statistics'] = time.time() - stage_start
        
        # STAGE 8: Save outputs
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("SAVING OUTPUTS")
        logger.info("-" * 80)
        
        output_path = snk.output.network
        logger.info(f"Saving network to {output_path}")
        save_network(network, output_path, custom_logger=logger)
        
        summary_path = snk.output.summary
        logger.info(f"Saving renewable summary to {summary_path}")
        summary_df.to_csv(summary_path, index=False)
        stage_times['8. Save outputs'] = time.time() - stage_start
        
        # COORDINATE VALIDATION: Ensure all buses use consistent OSGB36 coordinates
        # This prevents spatial mapping failures caused by mixed coordinate systems
        try:
            from spatial_utils import validate_network_coordinates, ensure_osgb36_coordinates
            validation = validate_network_coordinates(network, fix=False)
            if validation['wgs84_count'] > 0:
                logger.warning(f"COORDINATE CHECK: Found {validation['wgs84_count']} buses with WGS84 coordinates!")
                fixed = ensure_osgb36_coordinates(network)
                if fixed > 0:
                    logger.info(f"COORDINATE FIX: Converted {fixed} buses from WGS84 to OSGB36")
            else:
                logger.info(f"COORDINATE CHECK: All {validation['osgb36_count']} buses use OSGB36 ✓")
        except ImportError:
            logger.debug("Could not validate coordinates (spatial_utils not available)")
        
        # Log stage timing summary
        log_stage_summary(stage_times, logger, "RENEWABLE INTEGRATION - STAGE TIMING")
        
        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'renewable_sites_processed': total_sites,
            'renewable_generators_added': added_generators,
            'total_renewable_capacity_mw': summary_df['capacity_mw'].sum() if len(summary_df) > 0 else 0,
            'technologies_integrated': len(summary_df),
            'buses_with_renewables': renewable_sites['bus'].nunique()
        }
        
        log_execution_summary(logger, "integrate_renewable_generators", execution_time, summary_stats)
        logger.info("Final network (with renewables)")
        log_network_info(network, logger)
        
        logger.info("=" * 80)
        logger.info("RENEWABLE GENERATOR INTEGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Renewable generator integration failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

