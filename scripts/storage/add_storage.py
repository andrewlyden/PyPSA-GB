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
import yaml
from typing import Dict, Tuple, Optional

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Suppress PyPSA warnings about unoptimized networks (expected during network building)
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Import shared spatial utilities
from scripts.utilities.spatial_utils import map_sites_to_buses, apply_etys_bmu_mapping
from scripts.CP30.storage_interconnectors import (
    COMMITTED_STORAGE_SITES_FILE,
    fixed_site_mask,
    load_committed_storage_sites,
    load_cp30_storage_targets,
    scale_storage_proxy_to_cp30_targets,
)

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


#: NESO's FES regional breakdown "GSP info" tables, committed (the 2024 file is byte-for-byte the API
#: download, checked 16 Oct 2026). The storage build reads these, not the API: on 16 Oct 2026 a DNS
#: failure at build time silently dropped every FES GSP-level storage row (13 GW), and CP30's
#: national scaling hid it by inflating the Direct rows (ELIB1A 174 -> 321 MW, CLFO1- 0 -> 253 MW).
#: FES 2025 has no NESO table; its file is derived from the GSP region polygons (scripts/FES/gsp_metadata.py).
#: An edition without a file fails: until 26 Oct 2026 every edition not listed silently took FES 2024's.
FES_GSP_INFO_FILES = {year: f"data/network/ETYS/fes{year}_regional_breakdown_gsp_info.csv"
                      for year in (2021, 2022, 2023, 2024, 2025)}


def fes_edition_of_file(fes_file) -> int:
    """FES edition of a processed building-block file, from its name ``FES_{edition}_data.csv``."""
    import re
    match = re.search(r"FES_(\d{4})_data\.csv$", str(fes_file).replace("\\", "/"))
    if not match:
        raise ValueError(f"Cannot tell the FES edition of {fes_file}; expected FES_<edition>_data.csv "
                         "or pass fes_year")
    return int(match.group(1))


def _declared_gsp_info_file(fes_file, logger: logging.Logger):
    """The rule's declared GSP info input (``fes_gsp_info``), or None outside Snakemake.

    The rule names the scenario's own edition's table (since 27 Oct 2026; before, FES 2024's for every
    edition). Where the building blocks are another edition the mismatch is logged, not hidden.
    """
    snk = globals().get('snakemake')
    declared = getattr(getattr(snk, 'input', None), 'fes_gsp_info', None) if snk else None
    if isinstance(declared, (list, tuple)):
        declared = declared[0] if declared else None
    if not declared:
        return None
    try:
        data_edition = fes_edition_of_file(fes_file)
    except ValueError:
        return str(declared)
    if f"fes{data_edition}_" not in Path(str(declared)).name:
        logger.warning(f"FES {data_edition} building blocks located with the declared GSP info table {declared} "
                       f"(not FES {data_edition}'s own, {FES_GSP_INFO_FILES.get(data_edition)})")
    return str(declared)


def load_gsp_info(fes_year: int, logger: logging.Logger, path=None) -> pd.DataFrame:
    """GSP coordinates for the FES GSP-level storage rows: the committed file, else the NESO API.

    Raises when neither gives a table: without it every GSP-level row would be dropped. An edition
    with neither a committed file nor a NESO URL raises; no other edition is used.
    """
    if path is None and int(fes_year) not in FES_GSP_INFO_FILES:
        raise RuntimeError(f"No FES {fes_year} GSP info: no committed file (FES_GSP_INFO_FILES covers "
                           f"{sorted(FES_GSP_INFO_FILES)}); the FES GSP-level storage rows cannot be located")
    path = Path(path or FES_GSP_INFO_FILES[int(fes_year)])
    if path.is_file():
        gsp_df = pd.read_csv(path, encoding='utf-8-sig')
        logger.info(f"GSP info for FES storage rows: {len(gsp_df)} GSPs from {path}")
    else:
        logger.warning(f"GSP info file {path} not found; downloading from the NESO API")
        gsp_df = _download_gsp_info(fes_year, logger)
    if gsp_df.empty or not {'Name', 'Latitude', 'Longitude'} <= set(gsp_df.columns):
        raise RuntimeError(f"No FES {fes_year} GSP info (file {path} or NESO API): the FES GSP-level storage "
                           "rows cannot be located")
    return gsp_df


class GspLocationError(RuntimeError):
    """A FES GSP-level storage row matches no GSP of the GSP info table."""


class GspLocator:
    """GSP coordinates by GSP ID, and by exact canonical name for rows without an id.

    Until 27 Oct 2026 rows were matched by lower-case name, else by the first word of the name, so
    'New Cross 66kV 66kV' (a doubled FES 2025 alias) took the first GSP containing 'new': Newton Stewart,
    483 km away. FES_{edition}_data.csv carries the GSP ID its own name mapping found; that id is the key.
    """

    def __init__(self, gsp_info: pd.DataFrame):
        from scripts.FES.gsp_metadata import canonical_gsp_name
        self._canonical = canonical_gsp_name
        located = gsp_info.dropna(subset=['Latitude', 'Longitude'])
        self.by_id = {str(i): (float(lat), float(lon))
                      for i, lat, lon in zip(located['GSP ID'], located['Latitude'], located['Longitude'])}
        names: Dict[str, set] = {}
        for name, gsp_id in zip(located['Name'], located['GSP ID']):
            names.setdefault(canonical_gsp_name(name), set()).add(str(gsp_id))
        # A canonical name shared by two GSPs is ambiguous and matches neither.
        self.by_name = {key: next(iter(ids)) for key, ids in names.items() if key and len(ids) == 1}

    def locate(self, gsp_id, gsp_name):
        """(lat, lon), or (None, None) when the row matches no GSP."""
        if gsp_id is not None and not pd.isna(gsp_id) and str(gsp_id).strip():
            return self.by_id.get(str(gsp_id).strip(), (None, None))
        key = self._canonical(gsp_name)
        if key in self.by_name:
            return self.by_id[self.by_name[key]]
        return None, None


def locate_fes_gsp(row: pd.Series, locator: GspLocator):
    """Coordinates of one FES GSP-level row (its 'GSP ID', else its 'GSP' name)."""
    return locator.locate(row.get('GSP ID'), row.get('GSP'))


def _download_gsp_info(fes_year: int, logger: logging.Logger) -> pd.DataFrame:
    """Download GSP info from NESO API for coordinate mapping."""
    import urllib3
    import io
    
    url = NESO_GSP_INFO_URLS.get(int(fes_year))
    if url is None:
        logger.error(f"No NESO GSP info URL for FES {fes_year}")
        return pd.DataFrame()

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


# Round-trip efficiency of each FES storage building block. These are round
# trips, not per-leg efficiencies: PyPSA takes efficiency_store and
# efficiency_dispatch separately, and the per-leg value is sqrt(round trip),
# the same symmetric split scripts/storage/storage_parameters.py makes.
#
# The distinction is not cosmetic. The block was previously written as per-leg
# values of 0.92 / 0.87 / 0.60, which is right for batteries and pumped hydro
# but gave "other storage" a 36% round trip against the 55% this project's own
# technology table declares for LAES. On the CP30 2030 fleet that one number
# turned 5.27 TWh of charging into 1.93 TWh of discharge instead of 2.90 TWh,
# and it is what made the fleet round trip 66% rather than the 85% a
# battery-only reading expects. See scripts/market/diagnose_storage_degeneracy.py.
#
# Defaults here must match config/defaults/base/storage.yaml storage.round_trip_efficiencies.
DEFAULT_STORAGE_ROUND_TRIP_EFFICIENCIES = {
    'Battery': 0.8464,                          # 0.92 per leg, grid-scale lithium
    'Domestic Battery': 0.8464,                 # same cells, behind the meter
    'Pumped Storage Hydroelectricity': 0.7569,  # 0.87 per leg
    'LAES': 0.55,                               # storage_parameters.py LAES RTE
    # CP30's LDES subtype, not a carrier (its units are carrier LAES). CP30 ES1
    # FFR 2030: 10,441.5 GWh generated from 14,848.7 GWh consumed = 0.703; the
    # fleet is mostly existing pumped hydro, whose own round trip is 0.7569.
    'LDES': 0.70,
}


# Standing loss (fraction of the stored energy lost per hour) by carrier.
# Defaults here must match config/defaults/base/storage.yaml storage.standing_losses.
# Pumped hydro and LAES/LDES lose nothing material at hourly scale; lithium
# batteries self-discharge ~0.005 %/h (user decision 27 Sep 2026). Before this
# every unit took a flat 0.001/h (0.1 %/h, ~2.4 %/day) and the pumped-hydro
# exemption missed carrier LAES, which is where CP30 puts its pumped hydro.
DEFAULT_STORAGE_STANDING_LOSSES = {
    'default': 0.0,
    'Battery': 0.00005,
    'Domestic Battery': 0.00005,
    'battery': 0.00005,
    'LAES': 0.0,
    'Liquid Air Energy Storage': 0.0,
    'Pumped Storage Hydroelectricity': 0.0,
    'pumped_hydro': 0.0,
}


# Capacity-aware storage placement. Must match config/defaults/base/storage.yaml storage.placement.
DEFAULT_STORAGE_PLACEMENT = {
    'enabled': True,
    'max_loading': 1.0,            # storage MW per MVA of branches touching the bus
    'search_radius_km': 40.0,      # fallback search when the substation has no headroom
    'min_target_voltage_kv': 132.0,
}


def _storage_standing_loss(carrier: str, standing_losses: Optional[Dict]) -> float:
    """Hourly standing loss for a carrier from the configured table (then 'default')."""
    configured = dict(DEFAULT_STORAGE_STANDING_LOSSES)
    configured.update(standing_losses or {})
    value = float(configured.get(carrier, configured.get('default', 0.0)))
    if not 0.0 <= value < 1.0:
        raise ValueError(f"Standing loss for {carrier!r} must be in [0, 1), got {value}")
    return value


def _leg_efficiency(carrier: str, round_trip_efficiencies: Optional[Dict],
                    logger: logging.Logger) -> float:
    """Per-leg efficiency for a carrier: the symmetric split of its round trip."""
    configured = dict(DEFAULT_STORAGE_ROUND_TRIP_EFFICIENCIES)
    configured.update(round_trip_efficiencies or {})
    if carrier not in configured:
        raise ValueError(
            f"No round-trip efficiency for storage carrier {carrier!r}; add it to "
            "storage.round_trip_efficiencies in config/defaults/base/storage.yaml"
        )
    round_trip = float(configured[carrier])
    if not 0.0 < round_trip <= 1.0:
        raise ValueError(
            f"Round-trip efficiency for {carrier!r} must be in (0, 1], got {round_trip}"
        )
    return float(np.sqrt(round_trip))


def load_fes_storage_data(
    fes_file: str,
    modelled_year: int,
    fes_scenario: str,
    repd_storage_df: pd.DataFrame,
    logger: logging.Logger,
    round_trip_efficiencies: Optional[Dict] = None,
    gsp_info_file=None,
    fes_year: Optional[int] = None,
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
        gsp_info_file: GSP info table; default the committed table of ``fes_year``
        fes_year: FES edition of ``fes_file``; default read from its name (FES_<edition>_data.csv)

    Returns:
        DataFrame with storage sites for the scenario
    """
    import os
    
    if not os.path.exists(fes_file):
        logger.warning(f"FES file not found: {fes_file} - using REPD storage only")
        return repd_storage_df

    # GSP coordinates, outside the try below: a missing table must fail the build, not fall back.
    # The table is the building blocks' own edition (until 26 Oct 2026 always FES 2024's).
    if gsp_info_file is None and fes_year is None:
        fes_year = fes_edition_of_file(fes_file)
    gsp_info = load_gsp_info(fes_year, logger, gsp_info_file)

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
            },
            'Srg_BB002': {
                'carrier': 'Domestic Battery',
                'nice_name': 'Domestic Batteries (G98)',
                'default_duration_h': 2.0,
            },
            'Srg_BB003': {
                'carrier': 'Pumped Storage Hydroelectricity',
                'nice_name': 'Pumped Hydro',
                'default_duration_h': 8.0,
            },
            'Srg_BB004': {
                'carrier': 'LAES',  # Default to LAES; could be CAES
                'nice_name': 'Other Storage (CAES/LAES)',
                'default_duration_h': 6.0,
            },
        }
        for bb_id, bb_config in storage_bb_config.items():
            leg = _leg_efficiency(bb_config['carrier'], round_trip_efficiencies, logger)
            bb_config['eta_charge'] = leg
            bb_config['eta_dispatch'] = leg
            logger.info(
                f"  {bb_id} ({bb_config['nice_name']}): round trip "
                f"{leg * leg:.3f}, {leg:.4f} per leg"
            )
        
        # Create storage sites list
        storage_sites = []
        unlocated: list = []
        gsp_locator = GspLocator(gsp_info)
        
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
                lat, lon = locate_fes_gsp(row, gsp_locator)
                if lat is None:
                    unlocated.append(f"{bb_id} {gsp_name!r} (GSP ID {row.get('GSP ID')}) {float(capacity):,.1f} MW")
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

        if unlocated:
            raise GspLocationError(f"{len(unlocated)} FES GSP-level storage rows match no GSP in the GSP info "
                                   "table (by GSP ID, or by exact canonical name where the row has no id): "
                                   + "; ".join(unlocated))
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

    except GspLocationError:
        raise
    except Exception as e:
        logger.warning(f"Error loading FES storage data: {e} - using REPD storage only")
        import traceback
        logger.debug(traceback.format_exc())
        return repd_storage_df


def load_cp30_storage_data(
    cp30_es1_file: str,
    fes_file: str,
    modelled_year: int,
    cp30_pathway: str,
    spatial_fes_scenario: str,
    repd_storage_df: pd.DataFrame,
    logger: logging.Logger,
    round_trip_efficiencies: Optional[Dict] = None,
    existing_sites=None,
    committed_sites=None,
) -> pd.DataFrame:
    """Load CP30 storage targets and distribute them using explicit FES spatial proxy rows.

    ``existing_sites`` (config ``hydro.pumped_hydro.existing_sites``) are the
    existing pumped hydro inside CP30's LDES: each is placed at its own site
    with its real power and energy, and only the remainder is new build.
    ``committed_sites`` (``load_committed_storage_sites`` of
    ``data/storage/cp30_committed_ldes_sites.csv``, 15 Oct 2026) are the named
    new-build projects the pathway counts, placed the same way.
    """
    logger.info(f"Loading CP30 storage targets for {cp30_pathway} {modelled_year}")
    configured_round_trips = dict(DEFAULT_STORAGE_ROUND_TRIP_EFFICIENCIES)
    configured_round_trips.update(round_trip_efficiencies or {})
    targets = load_cp30_storage_targets(
        cp30_es1_file, cp30_pathway, modelled_year, round_trip_efficiencies=configured_round_trips
    )
    if targets.empty:
        logger.warning("No CP30 storage targets found")
        return pd.DataFrame()
    for _, target in targets.iterrows():
        logger.info(
            f"  CP30 {target['cp30_subtype']} ({target['carrier']}): round trip "
            f"{target['round_trip_efficiency']:.3f} configured, {target['eta_charge']:.4f} per leg; "
            f"CP30 ES1 implies {target['cp30_implied_round_trip']:.3f}"
        )

    proxy = load_fes_storage_data(
        fes_file=fes_file,
        modelled_year=modelled_year,
        fes_scenario=spatial_fes_scenario,
        repd_storage_df=repd_storage_df,
        logger=logger,
        round_trip_efficiencies=round_trip_efficiencies,
    )
    if proxy.empty:
        logger.warning("No storage spatial proxy rows available for CP30")
        return pd.DataFrame()

    scaled = scale_storage_proxy_to_cp30_targets(proxy, targets, existing_sites=existing_sites,
                                                 committed_sites=committed_sites)
    if scaled.empty:
        logger.warning("CP30 storage targets could not be matched to storage spatial proxy technologies")
        return pd.DataFrame()

    target_power = targets.groupby("carrier")["power_mw"].sum()
    target_energy = targets.groupby("carrier")["energy_mwh"].sum()
    scaled_power = scaled.groupby("technology")["power_mw"].sum()
    scaled_energy = scaled.groupby("technology")["energy_mwh"].sum()
    logger.info("CP30 storage targets applied:")
    for carrier, target_mw in target_power.sort_values(ascending=False).items():
        logger.info(
            f"  {carrier}: power target={target_mw:,.1f} MW, scaled={scaled_power.get(carrier, 0.0):,.1f} MW; "
            f"energy target={target_energy.get(carrier, 0.0):,.1f} MWh, scaled={scaled_energy.get(carrier, 0.0):,.1f} MWh"
        )
    if 'existing_site' in scaled.columns:
        existing = scaled[scaled['existing_site'].notna()]
        for _, row in existing.iterrows():
            logger.info(
                f"  existing {row['technology']} site {row['existing_site']}: {row['power_mw']:,.0f} MW, "
                f"{row['energy_mwh']:,.0f} MWh ({row['duration_h']:.2f} h) at site code {row['site_code']}"
            )
        if 'committed_site' in scaled.columns:
            for _, row in scaled[scaled['committed_site'].notna()].iterrows():
                logger.info(
                    f"  committed {row['technology']} project {row['committed_site']}: {row['power_mw']:,.0f} MW, "
                    f"{row['energy_mwh']:,.0f} MWh ({row['duration_h']:.2f} h) at site code {row['site_code']}"
                )
        fixed = fixed_site_mask(scaled)
        new_build = scaled[~fixed & scaled['technology'].isin(scaled.loc[fixed, 'technology'])]
        if len(new_build):
            logger.info(
                f"  new-build remainder: {new_build['power_mw'].sum():,.0f} MW, "
                f"{new_build['energy_mwh'].sum():,.0f} MWh ({new_build['duration_h'].iloc[0]:.2f} h) "
                f"over {len(new_build)} proxy rows"
            )

    return scaled


# =============================================================================
# CAPACITY-AWARE PLACEMENT
# =============================================================================

#: Carriers moved by :func:`place_batteries_at_gsp_nodes` (CP30 batteries carry ``Battery``).
GSP_NODE_BATTERY_CARRIERS = ('Battery', 'Domestic Battery')


def place_batteries_at_gsp_nodes(storage_df: pd.DataFrame, network: pypsa.Network,
                                 log: Optional[logging.Logger] = None, locator=None,
                                 polygon_file=None, dem_per_node_file=None,
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Put distribution-size batteries on the demand node of the GSP polygon they sit in (15 Oct 2026).

    Nearest-bus matching put 21% of CP30's batteries (5.8 GW) on a substation other than their
    own GSP's, 3.4 GW on substations with no demand. A battery below its licence area's
    transmission threshold (SHET 10 MW, SPT 30 MW, England and Wales 100 MW) connects inside its
    GSP, so it goes to the bus that carries that GSP's demand; so does an FES GSP-level row
    (``proxy_source`` ``FES_GSP_*``, the GSP's distribution-connected batteries summed), whatever
    its total. Uses the shared locator, ``scripts.utilities.gsp_nodes`` (GSP polygon ->
    ``GB_network.xlsx`` ``Dem_per_node`` -> the demand build's node rule); ``polygon_file`` and
    ``dem_per_node_file`` default to its files. Rows at a named site (``site_code``: existing
    pumped hydro, committed projects) and transmission-size rows keep their bus; capacity is
    unchanged and only ``bus`` moves.
    """
    log = log or logger
    from scripts.utilities.gsp_nodes import (DEFAULT_DEM_PER_NODE_FILE, DEFAULT_POLYGON_FILE, GspNodeLocator,
                                             place_distributed_at_gsp_nodes)
    if locator is None:
        locator = GspNodeLocator(network, polygon_file or DEFAULT_POLYGON_FILE,
                                 dem_per_node_file=dem_per_node_file or DEFAULT_DEM_PER_NODE_FILE, log=log)
    result = storage_df.copy()
    named = (result['site_code'].fillna('').astype(str).str.strip().ne('')
             if 'site_code' in result.columns else pd.Series(False, index=result.index))
    named |= fixed_site_mask(result)
    carriers = GSP_NODE_BATTERY_CARRIERS
    # FES GSP-level rows (proxy_source FES_GSP_*) are a GSP's distribution-connected batteries
    # aggregated at that GSP, so they go to its node whatever their summed size; site-based rows
    # (FES Direct spread over REPD sites, REPD) follow the size threshold.
    origin = (result['proxy_source'] if 'proxy_source' in result.columns
              else result.get('source', pd.Series('', index=result.index)))
    gsp_aggregate = origin.fillna('').astype(str).str.startswith('FES_GSP')
    groups = (
        (result[~named & gsp_aggregate], {'_P': np.inf, '_N': np.inf, 'default': np.inf}),
        (result[~named & ~gsp_aggregate], None),
    )
    all_moves = []
    for candidates, thresholds in groups:
        if candidates.empty:
            continue
        placed, moves = place_distributed_at_gsp_nodes(
            candidates, locator, carriers=carriers, thresholds=thresholds,
            capacity_col='power_mw', carrier_col='technology', log=log, data_source='battery_gsp_node',
        )
        for column in placed.columns.difference(result.columns):
            result[column] = np.nan
            result[column] = result[column].astype(object)
        result.loc[placed.index, placed.columns] = placed
        all_moves.append(moves)
    moves = pd.concat(all_moves) if all_moves else pd.DataFrame()
    moved_mw = float(moves['power_mw'].sum()) if len(moves) else 0.0
    log.info(f"Batteries at their GSP node: {len(moves)} rows ({moved_mw:,.0f} MW) moved off "
             f"their nearest bus")
    return result, moves


def reconcile_placed_storage_regions(storage_df: pd.DataFrame, network: pypsa.Network, es5_reconcile,
                                     log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """CP30 ES5 storage split on the region of each row's placed bus (16 Oct 2026).

    ``es5_reconcile`` is (ES5 targets, GSP polygon file, GSP group region file). See
    ``scripts.CP30.storage_regions.reconcile_storage_regions_to_buses``; ``capacity_mw`` follows
    ``power_mw``.
    """
    from scripts.CP30.storage_regions import reconcile_storage_regions_to_buses
    targets, polygon_file, group_region_file = es5_reconcile
    result, _ = reconcile_storage_regions_to_buses(storage_df, network, targets, polygon_file, group_region_file,
                                                   log or logger)
    if 'capacity_mw' in result.columns:
        result['capacity_mw'] = result['power_mw']
    return result


#: data_source given to rows weak-bus relief must count but never move (a value
#: ``spatial_utils._PROTECTED_PLACEMENTS`` already holds), restored afterwards.
_RELIEF_PROTECTED_SOURCE = 'CP30_committed_overlay'


def storage_relief_capacity_mw(storage_df: pd.DataFrame) -> pd.Series:
    """A storage row's size for weak-bus relief: the larger of its charge and discharge power.

    A StorageUnit charges and discharges at ``p_nom`` (``power_mw``); separate
    ``charge_power_mw`` / ``discharge_power_mw`` columns, where a source gives them, win.
    """
    parts = [pd.to_numeric(storage_df[c], errors='coerce') for c in
             ('power_mw', 'charge_power_mw', 'discharge_power_mw') if c in storage_df.columns]
    if not parts:
        return pd.Series(0.0, index=storage_df.index)
    return pd.concat(parts, axis=1).max(axis=1).fillna(0.0)


def relieve_storage_weak_buses(storage_df: pd.DataFrame, network: pypsa.Network, relief_cfg: Optional[Dict] = None,
                               exclude_buses=None, log: Optional[logging.Logger] = None
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Weak-bus relief (``spatial_utils.relieve_weak_bus_placements``) for storage proxy rows (15 Oct 2026).

    Renewables were relieved; storage proxies were not, so a 33/132 kV bus could hold several
    hundred MW of proxy LDES and batteries behind a small transformer. Each row is sized at the
    larger of its charge and discharge power. Every row counts towards its bus's loading, but only
    proxy rows move: rows at a named site (``site_code``: existing pumped hydro, committed
    projects such as Coire Glas, Red John, Cruachan II) and batteries placed on their GSP node
    (``data_source`` ``battery_gsp_node``) keep their bus. Whole rows move; capacity,
    duration and labels are unchanged. Returns (storage_df, moves).
    """
    from scripts.utilities.spatial_utils import relieve_weak_bus_placements
    log = log or logger
    cfg = {'max_loading': 1.5, 'search_radius_km': 40.0, 'min_target_voltage_kv': 132.0}
    cfg.update({k: v for k, v in (relief_cfg or {}).items() if k in cfg})
    if storage_df.empty or 'bus' not in storage_df.columns:
        return storage_df, pd.DataFrame()
    rows = storage_df.copy()
    named = (rows['site_code'].fillna('').astype(str).str.strip().ne('') if 'site_code' in rows.columns
             else pd.Series(False, index=rows.index)) | fixed_site_mask(rows)
    source = rows['data_source'] if 'data_source' in rows.columns else pd.Series(np.nan, index=rows.index)
    keep = named | source.astype(str).eq('battery_gsp_node')
    original_source = source.copy()
    rows['data_source'] = source.where(~keep, _RELIEF_PROTECTED_SOURCE)
    rows['_relief_mw'] = storage_relief_capacity_mw(rows)
    relieved, moves = relieve_weak_bus_placements(
        rows, network, max_loading=float(cfg['max_loading']), search_radius_km=float(cfg['search_radius_km']),
        min_target_voltage_kv=float(cfg['min_target_voltage_kv']), capacity_col='_relief_mw', log=log,
        exclude_buses=exclude_buses,
    )
    relieved['data_source'] = original_source.reindex(relieved.index)
    if 'data_source' not in storage_df.columns:
        relieved = relieved.drop(columns='data_source')
    relieved = relieved.drop(columns='_relief_mw')
    if len(moves):
        log.info(f"Storage weak-bus relief: {len(moves)} proxy rows ({moves['capacity_mw'].sum():,.0f} MW) moved")
    return relieved, moves


def pin_storage_to_site_codes(storage_df: pd.DataFrame, network: pypsa.Network,
                              log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Put rows that name a ``site_code`` on that substation's best-connected node.

    An existing plant connects at a known substation; nearest-bus matching
    picks whichever of its busbars is closest (Cruachan went CRUA2R -> CRUA2Q
    between two builds). The row goes to the bus of that four-character code
    whose node has the most headroom: the branch capacity leaving the node
    (:func:`~scripts.utilities.spatial_utils.bus_branch_capacity`, bus sections
    and couplers one node) less the storage already on any of the node's
    busbars, most branch capacity among equals. With no other storage at the
    site that is the best-connected bus, as before; with some, two busbars of
    one coupled node no longer each look free (F25.2, 27 Oct 2026).
    :func:`spread_storage_over_branch_capacity` then shares a row with other
    nodes of the site if one cannot carry it.
    """
    log = log or logger
    if 'site_code' not in storage_df.columns:
        return storage_df
    from scripts.utilities.spatial_utils import bus_branch_capacity, bus_section_groups
    result = storage_df.copy()
    capacity = bus_branch_capacity(network)
    group = bus_section_groups(network)
    power = (pd.to_numeric(result['power_mw'], errors='coerce').fillna(0.0) if 'power_mw' in result.columns
             else pd.Series(0.0, index=result.index))
    on_network = result['bus'].isin(network.buses.index)
    node_load = power[on_network].groupby(result.loc[on_network, 'bus'].map(group)).sum()
    codes = result['site_code'].fillna('').astype(str).str.upper()
    for idx in result.index[codes.ne('')]:
        code = codes.at[idx]
        at_site = pd.Series(capacity.index.astype(str).str.upper().str[:4] == code, index=capacity.index)
        candidates = capacity[at_site & capacity.gt(0)]
        if candidates.empty:
            raise ValueError(
                f"Storage site {result.at[idx, 'site_name']!r}: no connected bus with site code {code!r} "
                "in the network"
            )
        current = result.at[idx, 'bus']
        if current in group.index:
            own_node = group.at[current]
            node_load.loc[own_node] = node_load.get(own_node, 0.0) - power.at[idx]
        room = candidates - group.reindex(candidates.index).map(node_load).fillna(0.0)
        order = pd.DataFrame({'room': room.round(6), 'capacity': candidates}).sort_values(
            ['room', 'capacity'], ascending=False, kind='mergesort')
        bus = order.index[0]
        if current != bus:
            log.info(f"  site-code pin: {result.at[idx, 'site_name']} {current} -> {bus} "
                     f"({candidates.at[bus]:.0f} MVA, {room.at[bus]:.0f} MW free on its node)")
        result.at[idx, 'bus'] = bus
        node_load.loc[group.at[bus]] = node_load.get(group.at[bus], 0.0) + power.at[idx]
    return result


def stranded_storage_buses(network: pypsa.Network) -> pd.Index:
    """Buses where storage is not on a demand-bearing AC system.

    A bus with no line or transformer (the Caithness-Moray-Shetland DC hub
    "Central DC Substation", which nearest-bus matching gave Orkney's
    battery), or one in an AC island other than the main one that carries no
    load (the Sofia converter pair SOFWD/SOFW61, reached only over the HVDC
    export link), is an export or converter node, not somewhere storage
    connects. Islands with demand (Shetland) are real systems and are kept.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    buses = network.buses.index
    position = pd.Series(np.arange(len(buses)), index=buses)
    pairs = [getattr(network, c)[['bus0', 'bus1']] for c in ('lines', 'transformers') if len(getattr(network, c))]
    if not pairs:
        return pd.Index([])
    pairs = pd.concat(pairs, ignore_index=True)
    pairs = pairs[pairs['bus0'].isin(buses) & pairs['bus1'].isin(buses)]
    i, j = position[pairs['bus0']].to_numpy(), position[pairs['bus1']].to_numpy()
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(buses), len(buses)))
    _, labels = connected_components(graph, directed=False)
    labels = pd.Series(labels, index=buses)
    has_branch = pd.Series(False, index=buses)
    has_branch[pd.unique(pd.concat([pairs['bus0'], pairs['bus1']]))] = True
    main = labels[has_branch].value_counts().idxmax()
    loaded = set(labels.reindex(network.loads.bus.unique()).dropna().astype(int))
    stranded = ~has_branch | ((labels != main) & ~labels.isin(loaded))
    return buses[stranded.to_numpy()]


def spread_storage_over_branch_capacity(
    storage_df: pd.DataFrame,
    network: pypsa.Network,
    max_loading: float = 1.0,
    search_radius_km: float = 40.0,
    min_target_voltage_kv: float = 132.0,
    excluded_buses=(),
    stranded_buses=None,
    log: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Share storage over sibling busbars so no node holds more than its branches carry.

    Storage on a stranded bus (:func:`stranded_storage_buses`, computed when
    ``stranded_buses`` is None) is moved in full, and to the nearest target at
    any distance: such a bus is where nearest-bus matching lands a site that
    has no network of its own nearby (Orkney on the DC hub).

    Buses joined by bus sections or couplers are one node
    (:func:`~scripts.utilities.spatial_utils.bus_section_groups`), as in
    :func:`~scripts.utilities.spatial_utils.bus_branch_capacity`: a node is
    overloaded when the storage power on all its busbars exceeds
    ``max_loading`` times the rating of the lines, transformers and links
    leaving it (MVA/MW; unchanged by any impedance correction), and storage is
    never moved within its own node. Judged busbar by busbar, two busbars of
    one node each saw the node's whole headroom (F25.2, 27 Oct 2026). CP30
    proxy placement put 1,052 MW on MITY12 (541 MVA), 391 MW on LYND1Q
    (186 MVA), 200 MW on TYNE51 (88 MVA) and a pumped-hydro site on one of
    Cruachan's two 283 MVA busbars. For each overloaded node the excess is
    moved, largest rows first, to

    1. buses of another node of the same substation (four-character site code)
       with headroom, most branch capacity first, at or above the overloaded
       node's voltage or ``min_target_voltage_kv``; then
    2. the nearest bus of another node at or above ``min_target_voltage_kv``
       within ``search_radius_km`` with headroom,

    splitting a row when only part of it fits. Split rows keep the site's
    duration, efficiencies and labels; power and energy are divided pro rata,
    so totals by carrier are unchanged. Returns (storage_df, log of moves).
    """
    log = log or logger
    from scripts.utilities.spatial_utils import bus_branch_capacity, bus_section_groups
    result = storage_df.copy()
    if result.empty or 'bus' not in result.columns:
        return result, pd.DataFrame()
    if not pd.api.types.is_integer_dtype(result.index):
        result = result.reset_index(drop=True)
    buses = network.buses
    capacity = bus_branch_capacity(network)
    group = bus_section_groups(network)
    node_of = group.to_numpy()
    names = buses.index.astype(str)
    stranded = pd.Index(stranded_storage_buses(network) if stranded_buses is None else stranded_buses)
    is_stranded = pd.Series(names.isin(stranded), index=buses.index)
    node_stranded = is_stranded.groupby(group).all()
    node_capacity = capacity.groupby(group).first()
    limit = (max_loading * node_capacity).where(~node_stranded.reindex(node_capacity.index), 0.0)
    power = pd.to_numeric(result['power_mw'], errors='coerce').fillna(0.0)
    result['power_mw'] = power
    on_network = result['bus'].isin(buses.index)
    load = power[on_network].groupby(result.loc[on_network, 'bus'].map(group)).sum()
    load = load.reindex(node_capacity.index, fill_value=0.0)
    excluded = (names.str.contains('External', case=False) | names.isin(list(excluded_buses))
                | capacity.le(0).values | is_stranded.values)
    v_nom = pd.to_numeric(buses['v_nom'], errors='coerce').fillna(0.0)
    xy = buses[['x', 'y']].astype(float)
    next_index = int(result.index.max()) + 1 if len(result) else 0

    moves = []
    overloaded = load[load > limit + 1e-6].sort_values(ascending=False)
    for node in overloaded.index:
        code = str(node)[:4]
        other_node = node_of != node
        # A lower-voltage sibling below min_target_voltage_kv (LYND3- under
        # LYND1Q) exports only through the overloaded node, so it is no relief.
        voltage_ok = ((v_nom >= min(float(v_nom.at[node]), min_target_voltage_kv))).values
        same_site = (names.str[:4] == code) & other_node & ~excluded & voltage_ok
        radius = np.inf if bool(node_stranded.at[node]) else search_radius_km
        node_buses = set(buses.index[node_of == node])
        rows = sorted(result.index[result['bus'].isin(node_buses)], key=lambda i: -result.at[i, 'power_mw'])
        for idx in rows:
            excess = load.at[node] - limit.at[node]
            if excess <= 1e-6:
                break
            bus = result.at[idx, 'bus']
            distance_km = np.hypot(xy['x'] - xy.at[bus, 'x'], xy['y'] - xy.at[bus, 'y']) / 1000.0
            near = (~excluded & other_node & (v_nom >= min_target_voltage_kv).values
                    & (distance_km <= radius).values)
            while excess > 1e-6 and result.at[idx, 'power_mw'] > 1e-6:
                headroom = pd.Series((limit - load).clip(lower=0.0).reindex(node_of).to_numpy(),
                                     index=buses.index)
                site_room = headroom[same_site & (headroom > 1e-3).values]
                if len(site_room):
                    target = capacity[site_room.index].idxmax()
                    method = 'same_substation'
                else:
                    near_room = headroom[near & (headroom > 1e-3).values]
                    if near_room.empty:
                        break
                    target = distance_km[near_room.index].idxmin()
                    method = f'nearest_{distance_km[target]:.0f}km'
                row_power = result.at[idx, 'power_mw']
                move = min(row_power, excess, headroom.at[target])
                if move >= row_power - 1e-9:
                    result.at[idx, 'bus'] = target
                else:
                    share = move / row_power
                    piece = result.loc[idx].copy()
                    piece['bus'] = target
                    piece['power_mw'] = move
                    result.at[idx, 'power_mw'] = row_power - move
                    if 'energy_mwh' in result.columns:
                        energy = float(result.at[idx, 'energy_mwh'])
                        piece['energy_mwh'] = energy * share
                        result.at[idx, 'energy_mwh'] = energy * (1.0 - share)
                    if 'site_name' in result.columns:
                        piece['site_name'] = f"{result.at[idx, 'site_name']}_{target}"
                    result.loc[next_index] = piece
                    next_index += 1
                load.at[node] -= move
                load.at[group.at[target]] += move
                excess -= move
                moves.append({
                    'site_name': result.at[idx, 'site_name'] if 'site_name' in result.columns else idx,
                    'technology': result.at[idx, 'technology'] if 'technology' in result.columns else '',
                    'capacity_mw': move, 'from_bus': bus, 'from_branch_capacity': capacity.at[bus],
                    'to_bus': target, 'to_branch_capacity': capacity.at[target], 'method': method,
                    'split': move < row_power - 1e-9,
                })
                if move >= row_power - 1e-9:
                    break
        if load.at[node] > limit.at[node] + 1e-6:
            log.warning(
                f"Storage placement: node {node} still carries {load.at[node]:.0f} MW of storage against "
                f"{node_capacity.at[node]:.0f} MVA of branches (no sibling or nearby bus with headroom)"
            )
    moves = pd.DataFrame(moves)
    if len(moves):
        log.info(
            f"Storage placement: moved {moves['capacity_mw'].sum():,.0f} MW off {moves['from_bus'].nunique()} buses "
            f"whose node's storage exceeded {max_loading:.2f}x its branch capacity "
            f"({int(moves['split'].sum())} rows split)"
        )
        summary = moves.groupby(['from_bus', 'to_bus', 'method']).agg(
            mw=('capacity_mw', 'sum'), rows=('capacity_mw', 'size'),
            from_cap=('from_branch_capacity', 'first'), to_cap=('to_branch_capacity', 'first'))
        for (from_bus, to_bus, method), row in summary.sort_values('mw', ascending=False).iterrows():
            log.info(f"  {from_bus} ({row.from_cap:.0f} MVA) -> {to_bus} ({row.to_cap:.0f} MVA) "
                     f"[{method}]: {row.mw:.1f} MW in {int(row.rows)} move(s)")
    return result, moves


# Note: Bus mapping now handled by shared map_sites_to_buses function
# from spatial_utils.py. This provides consistent coordinate handling
# across all PyPSA-GB components (renewables, thermal, storage, interconnectors).
# The shared function automatically detects coordinate systems (WGS84 vs OSGB36)
# and uses appropriate distance metrics (haversine vs Euclidean).


def _storage_marginal_cost(row, carrier: str, dispatch_costs: Optional[Dict]) -> float:
    """Return the unit's dispatch cost: the data value if given, else the carrier default.

    A strictly positive default matters for the market LP: with a zero cost
    every storage unit is indifferent between charging and discharging in
    zero-price hours, and a barrier solve without crossover can then charge
    and discharge simultaneously, dissipating energy through round-trip
    losses. Defaults live in config/defaults/base/storage.yaml ``storage.dispatch_costs``.
    """
    value = row.get('marginal_cost', None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float('nan')
    if np.isfinite(value) and value > 0.0:
        return value
    costs = dispatch_costs or {}
    return float(costs.get(carrier, costs.get('default', 0.0)))


def add_storage_to_network(network: pypsa.Network, storage_df: pd.DataFrame,
                           standing_losses: Optional[Dict] = None,
                           hydro_config: Optional[Dict] = None,
                           dispatch_costs: Optional[Dict] = None) -> pypsa.Network:
    """
    Add storage units to the PyPSA network.

    Args:
        network: PyPSA network object
        storage_df: DataFrame with storage sites and assigned buses
        standing_losses: Hourly standing loss by carrier (config
            ``storage.standing_losses``, with a ``default`` entry); missing
            keys fall back to DEFAULT_STORAGE_STANDING_LOSSES.

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
    hydro_config = hydro_config or {}
    pumped_hydro_config = hydro_config.get('pumped_hydro', {})
    
    for idx, row in storage_df.iterrows():
        # Use site_name if available, otherwise create unique name
        if 'site_name' in row and pd.notna(row['site_name']) and row['site_name']:
            # Clean site name for use as index (remove special chars, spaces -> underscore)
            site_name_clean = str(row['site_name']).replace(' ', '_').replace('/', '_').replace('-', '_')
            site_name_clean = ''.join(c for c in site_name_clean if c.isalnum() or c == '_')
            storage_name = f"{site_name_clean}_{idx}"
        else:
            storage_name = f"storage_{row['technology'].replace(' ', '_')}_{idx}"
        
        # Get WGS84 coordinates from assigned bus (most reliable source)
        # Prefer bus lon/lat (WGS84) over x/y (OSGB36) for consistency
        bus_name = row['bus']
        if bus_name in network.buses.index:
            if 'lon' in network.buses.columns and 'lat' in network.buses.columns:
                bus_lon = network.buses.loc[bus_name, 'lon']
                bus_lat = network.buses.loc[bus_name, 'lat']
                # Validate they look like WGS84
                if pd.notna(bus_lon) and pd.notna(bus_lat) and (abs(bus_lon) > 20 or abs(bus_lat) > 90):
                    # Mislabeled OSGB36 — convert
                    from scripts.utilities.spatial_utils import osgb36_to_wgs84
                    bus_lon, bus_lat = osgb36_to_wgs84(float(bus_lon), float(bus_lat))
            elif 'x' in network.buses.columns and 'y' in network.buses.columns:
                # Fall back to x/y and convert OSGB36 → WGS84
                from scripts.utilities.spatial_utils import osgb36_to_wgs84, detect_coordinate_system
                bus_crs = detect_coordinate_system(
                    network.buses['x'].values, network.buses['y'].values
                )
                if bus_crs == 'OSGB36':
                    bus_lon, bus_lat = osgb36_to_wgs84(
                        float(network.buses.loc[bus_name, 'x']),
                        float(network.buses.loc[bus_name, 'y'])
                    )
                else:
                    bus_lon = network.buses.loc[bus_name, 'x']
                    bus_lat = network.buses.loc[bus_name, 'y']
            else:
                bus_lon = row.get('lon', None)
                bus_lat = row.get('lat', None)
        else:
            bus_lon = row.get('lon', None)
            bus_lat = row.get('lat', None)
        
        # Prepare storage parameters
        carrier = str(row['technology'])
        standing_loss = _storage_standing_loss(carrier, standing_losses)
        initial_soc_fraction = 0.5
        cyclic_state_of_charge = False

        if carrier in {'Pumped Storage Hydroelectricity', 'pumped_hydro'}:
            initial_soc_fraction = float(pumped_hydro_config.get('initial_soc_fraction', 0.65))
            cyclic_state_of_charge = bool(pumped_hydro_config.get('cyclic_state_of_charge', False))

        storage_params = {
            'bus': row['bus'],
            'carrier': carrier,
            'p_nom': row['power_mw'],  # Nominal power capacity
            'max_hours': row['duration_h'],  # Energy/Power ratio
            'efficiency_store': row['eta_charge'],  # Charge efficiency
            'efficiency_dispatch': row['eta_discharge'],  # Discharge efficiency
            'standing_loss': standing_loss,  # Self-discharge per hour
            'capital_cost': row.get('capital_cost', 0),
            'marginal_cost': _storage_marginal_cost(row, carrier, dispatch_costs),
        }
        
        # Add WGS84 coordinates for visualization
        # Use 'lon'/'lat' for consistency with generators (WGS84 degrees)
        if bus_lon is not None and bus_lat is not None:
            storage_params['lon'] = bus_lon
            storage_params['lat'] = bus_lat
        
        # Add cyclic state of charge if available
        if 'cyclic_state_of_charge' not in storage_params:
            storage_params['cyclic_state_of_charge'] = cyclic_state_of_charge

        # Initialise state of charge at 50% of max energy capacity.
        # Without this, storage starts empty and cannot generate in the
        # first solve window, producing unrealistic pumped-storage profiles.
        max_energy = row['power_mw'] * row['duration_h']
        storage_params['state_of_charge_initial'] = max_energy * initial_soc_fraction
        
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
    
    # Standardize coordinates: ensure all storage units have WGS84 lon/lat
    from scripts.utilities.spatial_utils import standardize_component_coordinates
    coord_result = standardize_component_coordinates(network, components=['StorageUnit'])
    if coord_result.get('StorageUnit', 0) > 0:
        logger.info(f"Standardized WGS84 coordinates for {coord_result['StorageUnit']} storage units")
    
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

    def _merge_dicts(base: Dict, override: Dict) -> Dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _load_hydro_config(scenario_config: Optional[Dict] = None) -> Dict:
        from config.config_loader import load_defaults

        hydro_config = (load_defaults().get("renewables") or {}).get("hydro", {}) or {}
        scenario_hydro = ((scenario_config or {}).get("renewables") or {}).get("hydro")
        if scenario_hydro:
            hydro_config = _merge_dicts(hydro_config, scenario_hydro)
        return hydro_config
    
    try:
        # Get input and output files
        try:
            # Snakemake mode
            network_input = snakemake.input.network
            storage_file = snakemake.input.storage_data
            network_output = snakemake.output.network
            max_distance_km = snakemake.params.get('max_distance_km', 200.0)
            modelled_year = snakemake.params.get('modelled_year', None)
            is_historical = snakemake.params.get('is_historical', False)
            scenario_config = snakemake.params.get('scenario_config', {})
            # Where the targets come from: every CP30 gate below reads CP30 numbers (ES1, the named LDES).
            from scripts.utilities.scenario_detection import target_source
            scenario_source = target_source(scenario_config)
            cp30_pathway = scenario_config.get('CP30_pathway')
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
            max_distance_km = 200.0
            modelled_year = None
            is_historical = False
            scenario_config = {}
            scenario_source = None
            cp30_pathway = None
            logger.info("Running in standalone mode")
        
        logger.info(f"Input network: {network_input}")
        logger.info(f"Storage parameters: {storage_file}")
        logger.info(f"Output network: {network_output}")
        if modelled_year:
            logger.info(f"Modelled year: {modelled_year} ({'historical' if is_historical else 'future'} scenario)")

        hydro_config = _load_hydro_config(scenario_config)
        
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
        
        # (ES5 targets, polygon file, group map) when the CP30 ES5 storage split runs: the split is
        # reconciled again on the region of each row's placed bus once placement is done.
        es5_reconcile = None
        # For future scenarios, load scenario storage data directly.
        # CP30 uses CP30 ES1 power/energy targets; FES is only an explicit spatial proxy.
        if not is_historical and modelled_year is not None:
            try:
                fes_file = None
                fes_scenario = None
                cp30_es1_file = None
                
                if 'snakemake' in globals():
                    fes_file = getattr(snakemake.input, 'fes_data', None)
                    fes_scenario = getattr(snakemake.params, 'fes_scenario', None)
                    cp30_es1_file = getattr(snakemake.input, 'cp30_es1', None)
                    logger.info(f"Storage spatial FES scenario from snakemake params: '{fes_scenario}'")
                
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
                
                # Task Y6b (FES 2050 plan): FES long-range scenarios build storage by type at
                # FES ES1 power and energy. The block is absent in every other scenario.
                from scripts.generators.fes_supply import fes_supply_part
                fes_storage = fes_supply_part(scenario_config, 'storage')
                if fes_file and fes_storage:
                    from scripts.generators.fes_supply import build_fes_storage, load_targets
                    targets_file = getattr(snakemake.input, 'fes_targets', None)
                    if not targets_file:
                        raise ValueError("fleet.fes_supply.storage needs the fes_targets input")
                    network_for_nodes = load_network(network_input, custom_logger=logger)
                    from scripts.CP30.spatial_allocation import LicenceAreas
                    storage_df = build_fes_storage(
                        fes_file, load_targets(targets_file), fes_storage, network_for_nodes.buses,
                        storage_df, pathway=fes_scenario, year=int(modelled_year),
                        round_trip_efficiencies=((scenario_config or {}).get('storage') or {}).get('round_trip_efficiencies', {}),
                        log=logger,
                        # Direct storage by licence area, on its own licence's buses (27 Oct 2026).
                        licence_areas=LicenceAreas(network_for_nodes.buses),
                    )
                    del network_for_nodes
                elif fes_file:
                    if fes_scenario is None:
                        logger.warning("FES scenario not specified - will load ALL FES pathways (likely incorrect!)")
                    if scenario_source == 'CP30':
                        if not cp30_es1_file:
                            logger.warning("CP30 storage requested but cp30_es1 input is missing")
                        else:
                            # Named new-build LDES the pathway counts, at their TEC substations
                            # (15 Oct 2026; rule input storage_committed_sites).
                            committed_file = getattr(snakemake.input, 'storage_committed_sites', None)
                            if isinstance(committed_file, (list, tuple)):
                                committed_file = committed_file[0] if committed_file else None
                            committed_file = str(committed_file or COMMITTED_STORAGE_SITES_FILE)
                            committed_sites = load_committed_storage_sites(committed_file, cp30_pathway)
                            logger.info(f"CP30 committed storage projects for {cp30_pathway}: "
                                        f"{len(committed_sites)} from {committed_file}")
                            storage_df = load_cp30_storage_data(
                                cp30_es1_file=cp30_es1_file,
                                fes_file=fes_file,
                                modelled_year=modelled_year,
                                cp30_pathway=cp30_pathway,
                                spatial_fes_scenario=fes_scenario,
                                repd_storage_df=storage_df,
                                logger=logger,
                                round_trip_efficiencies=((scenario_config or {}).get('storage') or {}).get(
                                    'round_trip_efficiencies', {}),
                                existing_sites=(hydro_config.get('pumped_hydro') or {}).get('existing_sites'),
                                committed_sites=committed_sites,
                            )
                            # CP30 ES5 regional split for storage (9 Oct 2026; scripts/CP30/storage_regions.py).
                            regional = (((scenario_config or {}).get('fleet') or {}).get('spatial_proxy') or {}).get(
                                'storage_regional_allocation') or {}
                            if regional.get('enabled', False) and len(storage_df):
                                from scripts.CP30.storage_regions import (
                                    apply_es5_storage_regions, apply_site_code_regions, es5_storage_targets,
                                    point_regions)
                                es5_file = list(getattr(snakemake.input, 'cp30_es5', None) or [])
                                es5_file = str(es5_file[0]) if es5_file else None
                                if not es5_file:
                                    raise ValueError("fleet.spatial_proxy.storage_regional_allocation needs the cp30_es5 input")
                                regions = point_regions(storage_df['lat'], storage_df['lon'],
                                                        regional['gsp_polygon_file'], regional['gsp_group_region_file'])
                                regions = apply_site_code_regions(regions, storage_df, logger=logger)
                                es5_targets = es5_storage_targets(es5_file, cp30_pathway)
                                storage_df = apply_es5_storage_regions(storage_df, es5_targets, regions, logger)
                                # Reconciled again on the placed bus's region after placement (below).
                                es5_reconcile = (es5_targets, regional['gsp_polygon_file'],
                                                 regional['gsp_group_region_file'])
                    else:
                        storage_df = load_fes_storage_data(
                            fes_file=fes_file,
                            modelled_year=modelled_year,
                            fes_scenario=fes_scenario,
                            repd_storage_df=storage_df,
                            logger=logger,
                            round_trip_efficiencies=((scenario_config or {}).get('storage') or {}).get(
                                'round_trip_efficiencies', {}),
                            gsp_info_file=_declared_gsp_info_file(fes_file, logger),
                        )
            except Exception as e:
                from scripts.generators.fes_supply import fes_supply_config
                if scenario_source == 'CP30' or fes_supply_config(scenario_config):
                    # A CP30 run silently falling back to the REPD fleet is wrong, not degraded.
                    raise
                logger.warning(f"Could not load future storage data: {e}")
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

        # On ETYS networks (whose nodes the GSP demand table names): distribution-size
        # batteries on their own GSP's demand node (15 Oct 2026; rule inputs
        # battery_gsp_node_inputs, scripts/utilities/gsp_nodes.py).
        etys_build = str((scenario_config or {}).get('network_model', '')).upper() == 'ETYS'
        if etys_build and len(storage_df):
            gsp_inputs = list(getattr(getattr(globals().get('snakemake'), 'input', None),
                                      'battery_gsp_node_inputs', None) or [])
            storage_df, _ = place_batteries_at_gsp_nodes(
                storage_df, network,
                polygon_file=gsp_inputs[0] if len(gsp_inputs) > 0 else None,
                dem_per_node_file=gsp_inputs[1] if len(gsp_inputs) > 1 else None,
            )

        # Existing plant to its own substation, then no bus holding more
        # storage than its branches can carry (config storage.placement).
        storage_df = pin_storage_to_site_codes(storage_df, network)
        # CP30 ES5 split on the region of the placed bus (the GSP-node move can cross a region
        # edge; 16 Oct 2026), before relief and spreading size rows against their buses.
        if es5_reconcile is not None:
            storage_df = reconcile_placed_storage_regions(storage_df, network, es5_reconcile)
        # Weak-bus relief for the proxy rows, with the renewables' switch and parameters
        # (network.spatial_mapping.weak_bus_relief; ETYS builds, as for renewables; 15 Oct 2026).
        relief_params = ((((scenario_config or {}).get('network') or {}).get('spatial_mapping') or {})
                         .get('weak_bus_relief') or {})
        if etys_build and relief_params.get('enabled', True) and len(storage_df):
            storage_df, _ = relieve_storage_weak_buses(
                storage_df, network, relief_params, exclude_buses=stranded_storage_buses(network))
        placement = dict(DEFAULT_STORAGE_PLACEMENT)
        placement.update(((scenario_config or {}).get('storage') or {}).get('placement') or {})
        if placement.get('enabled', True):
            storage_df, _ = spread_storage_over_branch_capacity(
                storage_df, network,
                max_loading=float(placement['max_loading']),
                search_radius_km=float(placement['search_radius_km']),
                min_target_voltage_kv=float(placement['min_target_voltage_kv']),
            )
            if 'capacity_mw' in storage_df.columns:
                storage_df['capacity_mw'] = storage_df['power_mw']
        # Relief and spreading may also cross a region edge: reconcile once more (normally a no-op).
        if es5_reconcile is not None:
            storage_df = reconcile_placed_storage_regions(storage_df, network, es5_reconcile)

        # Add storage to network
        network = add_storage_to_network(
            network,
            storage_df,
            standing_losses=((scenario_config or {}).get('storage') or {}).get('standing_losses', {}),
            hydro_config=hydro_config,
            dispatch_costs=((scenario_config or {}).get('storage') or {}).get('dispatch_costs', {}),
        )
        
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

