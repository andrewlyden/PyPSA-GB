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
import re
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
import json
import logging

# Import shared spatial utilities
from scripts.utilities.spatial_utils import (
    map_sites_to_buses, apply_etys_bmu_mapping, relieve_weak_bus_placements,
    generation_placement_exclusions, reassign_excluded_placements, apply_island_networks,
    relieve_export_constrained_proxies,
)

# Import renewable aggregation (capacity-weighted merge per bus+carrier)
from scripts.generators.aggregate_renewable_generators import aggregate_renewables_by_bus
from scripts.generators.fes_supply import fes_supply_part
from scripts.utilities.calendar_year import fill_leap_day
from scripts.storage.add_storage import GspLocationError, fes_edition_of_file
from scripts.utilities.scenario_detection import target_source, uses_brownfield
from scripts.CP30.generation import load_cp30_generation_capacities, scale_proxy_to_cp30_targets
from scripts.CP30.connection_allocation import apply_connection_allocation, landing_buses_for_regions
from scripts.CP30.spatial_allocation import (
    LicenceAreas,
    apply_cp30_es5_regional_allocation,
    assign_cp30_regions,
    read_es5_targets,
)
from scripts.CP30.brownfield_fleet import (
    BROWNFIELD_CARRIERS,
    DEFAULT_NAMED_EXCLUSIONS_FILE,
    apply_named_fleet_exclusions,
    SMALL_SCALE_SOLAR_METHOD,
    apply_offshore_landing_crosswalk,
    build_fleet_audit,
    build_landing_refill,
    build_small_scale_solar_reservation,
    load_named_fleet,
    reconcile_with_es5,
)
from scripts.CP30.hnd_hybrid import apply_hnd_project_allocation
from scripts.utilities.gsp_nodes import (
    DEFAULT_POLYGON_FILE as GSP_POLYGON_FILE,
    GSP_NODE_SOURCES,
    GspDemandNodes,
    GspNodeLocator,
    place_distributed_at_gsp_nodes,
)
from scripts.utilities.spatial_utils import _PROTECTED_PLACEMENTS, documented_connection_mask
from scripts.generators.station_unit_splits import apply_historical_station_unit_splits
from scripts.generators.proxy_profiles import (
    area_keys,
    build_regional_proxy_profiles,
    proxy_profile_settings,
    regions_of_points,
    site_members,
)
from scripts.generators.renewable_performance_factors import (
    load_performance_factor_registry,
    resource_region_for_site,
    select_performance_factor,
)
from scripts.renewables.large_hydro_inflow import (
    cp30_hydro_capacity_factor,
    inflow_capacity_factor,
    large_hydro_proxy_p_max_pu,
    load_monthly_shape,
)

# Inlined helpers and functions from scripts/add_generators.py to make this script
# self-contained and avoid cross-module imports that cause multiple loggers.

# Caches to avoid reloading profiles
_PROFILE_CACHE = {}


def _parse_operational_dates(values: pd.Series) -> pd.Series:
    """Parse REPD day-first dates and evidence-override ISO dates explicitly."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    # Evidence-backed commissioning overrides use ISO dates. Slice to the date
    # component so an already-parsed Timestamp is handled consistently too.
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", na=False)
    parsed.loc[iso_mask] = pd.to_datetime(
        text.loc[iso_mask].str[:10],
        format="%Y-%m-%d",
        errors="coerce",
    )
    parsed.loc[~iso_mask] = pd.to_datetime(
        text.loc[~iso_mask],
        format="%d/%m/%Y",
        errors="coerce",
    )
    return parsed


def _parse_operational_date(value):
    """Scalar companion to :func:`_parse_operational_dates`."""
    return _parse_operational_dates(pd.Series([value], dtype="object")).iloc[0]


def filter_sites_by_year(sites_df: pd.DataFrame, modelled_year: int,
                         logger: logging.Logger = None,
                         start_date: str = None,
                         end_date: str = None) -> pd.DataFrame:
    """
    Filter renewable sites by operational date for historical year modeling.

    For historical scenarios, only sites that were operational by the simulation
    start date should be included. This ensures that sites commissioned mid-year
    are not included in scenarios representing earlier periods.

    Parameters
    ----------
    sites_df : pd.DataFrame
        Renewable sites dataframe with 'operational_date' column (DD/MM/YYYY format)
    modelled_year : int
        The year being modelled; used as Dec-31 fallback if start_date is None
    logger : logging.Logger, optional
        Logger instance
    start_date : str, optional
        ISO-format start date of the simulation window (e.g. "2020-01-07 00:00").
        When provided, used as the cutoff instead of Dec 31 of modelled_year.
    end_date : str, optional
        ISO-format end date of the simulation window. When supplied, it takes
        precedence over ``start_date`` for profile-gated commissioning runs.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only sites operational by the cutoff date
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    original_count = len(sites_df)

    # Check if operational_date column exists
    if 'operational_date' not in sites_df.columns:
        logger.warning("No 'operational_date' column found - cannot filter by year")
        return sites_df

    # Commissioning-profile runs include projects that start within the window;
    # their availability is set to zero before the recorded start date below.
    if end_date is not None:
        cutoff_date = pd.Timestamp(end_date)
        logger.info(f"REPD capacity cutoff: simulation end_date = {cutoff_date.date()}")
    elif start_date is not None:
        cutoff_date = pd.Timestamp(start_date)
        logger.info(f"REPD capacity cutoff: simulation start_date = {cutoff_date.date()}")
    else:
        cutoff_date = pd.Timestamp(f'{modelled_year}-12-31')
        logger.info(f"REPD capacity cutoff: year-end fallback = {cutoff_date.date()}")

    operational_dates = _parse_operational_dates(sites_df['operational_date'])

    mask = operational_dates <= cutoff_date

    # Handle NaT (unparseable dates) - include them for robustness
    mask = mask | operational_dates.isna()

    filtered_df = sites_df[mask].copy()
    filtered_count = len(filtered_df)

    if original_count > 0:
        pct_kept = (filtered_count / original_count) * 100
        logger.info(f"Year-based filtering: {filtered_count}/{original_count} sites operational by {cutoff_date.date()} ({pct_kept:.1f}%)")

    return filtered_df


def _apply_commissioning_date_overrides(
    sites_df: pd.DataFrame,
    overrides: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """Apply evidence-backed activation dates before solve-window filtering."""
    if not overrides or 'operational_date' not in sites_df.columns:
        return sites_df
    name_col = next(
        (column for column in ('site_name', 'source_site_name') if column in sites_df.columns),
        None,
    )
    if name_col is None:
        return sites_df
    activation = sites_df[name_col].map(overrides)
    changed = activation.notna()
    if not changed.any():
        return sites_df
    result = sites_df.copy()
    result.loc[changed, 'source_operational_date'] = result.loc[changed, 'operational_date']
    result.loc[changed, 'operational_date'] = activation[changed]
    return result


def _assign_subsidy_attributes(
    sites_df: pd.DataFrame,
    modelled_year: int,
    roc_eligibility_years: int,
    fes_cfd_fractions: dict,
    logger: logging.Logger,
    roc_closure_date: str = "2017-04-01",
    default_roc_bandings: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Assign support_type and region attributes to renewable sites.

    For REPD sites (historical): uses cfd_round and ro_banding columns, with a
    date-based fallback for the ~90% of sites missing ro_banding data.  Non-CfD
    renewables operational before the ROC closure date (April 2017) are inferred
    as ROC-accredited using technology-specific default banding values — following
    the same assumption used by GBPower (all non-CfD wind/solar are ROC).

    For FES sites (future): splits generators into CfD/merchant sub-units
    based on fes_cfd_fractions config (ROC scheme closed 2017, no new ROC).

    Also assigns region ('north' or 'south') based on latitude threshold 55.5°.

    Parameters
    ----------
    sites_df : DataFrame
        Renewable sites with at least 'technology' and 'capacity_mw' columns.
    modelled_year : int
        The year being modelled (for ROC expiry check).
    roc_eligibility_years : int
        Number of years ROC accreditation lasts (typically 20).
    fes_cfd_fractions : dict
        Carrier → fraction of new capacity assumed to be CfD (e.g. wind_offshore: 1.0).
    logger : Logger
    roc_closure_date : str
        Date after which new ROC accreditation was closed (default "2017-04-01").
    default_roc_bandings : dict, optional
        Carrier → default ROC banding (ROC/MWh) for sites missing REPD ro_banding data.

    Returns
    -------
    DataFrame with added columns: support_type, region.
    FES rows may be duplicated where CfD fraction is between 0 and 1.
    """
    df = sites_df.copy()

    has_cfd = 'cfd_round' in df.columns
    has_roc = 'ro_banding' in df.columns
    has_opdate = 'operational_date' in df.columns

    # --- Assign support_type ---
    if has_cfd or has_roc:
        # REPD path: use actual REPD subsidy columns
        support = pd.Series('merchant', index=df.index)

        if has_cfd:
            is_cfd = df['cfd_round'].fillna('').astype(str).str.strip().ne('')
            # REPD also records projects that entered an allocation round but
            # were not awarded any CfD capacity. Where the capacity field is
            # available, require a positive award rather than treating the
            # allocation-round label alone as proof of support.
            if 'cfd_capacity_mw' in df.columns:
                cfd_capacity = pd.to_numeric(df['cfd_capacity_mw'], errors='coerce')
                is_cfd &= cfd_capacity.gt(0)
            support[is_cfd] = 'CfD'

        if has_roc:
            roc_banding = pd.to_numeric(df['ro_banding'], errors='coerce')
            is_roc = roc_banding.gt(0)
            # Check ROC expiry: operational_date + roc_eligibility_years > modelled_year
            if has_opdate:
                op_dates = _parse_operational_dates(df['operational_date'])
                roc_expiry = op_dates + pd.DateOffset(years=roc_eligibility_years)
                roc_expired = roc_expiry <= pd.Timestamp(f'{modelled_year}-01-01')
                # Only assign ROC if not expired AND not already CfD
                is_roc = is_roc & ~roc_expired & (support != 'CfD')
            else:
                # No operational_date — assume ROC still valid if banding present
                is_roc = is_roc & (support != 'CfD')
            support[is_roc] = 'ROC'

        # --- Date-based ROC inference fallback ---
        # REPD ro_banding is blank for ~90% of sites. For non-CfD renewables
        # operational before the ROC closure date, infer ROC accreditation with
        # default banding by technology. This follows GBPower's approach where
        # all non-CfD wind/solar are assumed ROC.
        if default_roc_bandings and has_opdate:
            roc_cutoff = pd.Timestamp(roc_closure_date)
            op_dates = _parse_operational_dates(df['operational_date'])

            # Eligible: still 'merchant', operational before ROC closure, not expired
            is_merchant = support == 'merchant'
            is_pre_closure = op_dates.notna() & (op_dates < roc_cutoff)
            roc_expiry_inferred = op_dates + pd.DateOffset(years=roc_eligibility_years)
            not_expired = roc_expiry_inferred > pd.Timestamp(f'{modelled_year}-01-01')

            # Carrier must have a default banding defined
            # Dispatchable REPD extraction retains the source technology label
            # (for example ``Biomass (dedicated)``) alongside its normalized
            # PyPSA ``fuel_type``. Prefer the normalized value for banding
            # lookup; renewable-site inputs use ``technology`` directly.
            carrier_col = next(
                column for column in ('fuel_type', 'technology', 'carrier')
                if column in df.columns
            )
            has_default_banding = df[carrier_col].map(
                lambda c: c in default_roc_bandings
            )

            infer_mask = is_merchant & is_pre_closure & not_expired & has_default_banding
            n_inferred = infer_mask.sum()

            if n_inferred > 0:
                support[infer_mask] = 'ROC'
                # Fill in ro_banding with the default value for each carrier
                if 'ro_banding' not in df.columns:
                    df['ro_banding'] = np.nan
                for carrier, banding in default_roc_bandings.items():
                    carrier_mask = infer_mask & (df[carrier_col] == carrier)
                    df.loc[carrier_mask, 'ro_banding'] = float(banding)

                inferred_mw = df.loc[infer_mask, 'capacity_mw'].sum()
                logger.info(f"  ROC inference: {n_inferred} sites ({inferred_mw:.0f} MW) "
                            f"inferred as ROC (operational before {roc_closure_date}, "
                            f"default banding applied)")
                # Log per-carrier breakdown
                for carrier in df.loc[infer_mask, carrier_col].unique():
                    c_mask = infer_mask & (df[carrier_col] == carrier)
                    c_mw = df.loc[c_mask, 'capacity_mw'].sum()
                    c_banding = default_roc_bandings.get(carrier, '?')
                    logger.info(f"    {carrier}: {c_mask.sum()} sites, "
                                f"{c_mw:.0f} MW, banding={c_banding} ROC/MWh")

        df['support_type'] = support
        n_cfd = (support == 'CfD').sum()
        n_roc = (support == 'ROC').sum()
        n_merchant = (support == 'merchant').sum()
        logger.info(f"  REPD subsidy assignment: {n_cfd} CfD, "
                     f"{n_roc} ROC, {n_merchant} merchant")
    else:
        # FES path: no REPD columns — split by fes_cfd_fractions
        rows_cfd = []
        rows_merchant = []
        rows_unchanged = []

        for idx, row in df.iterrows():
            carrier = row.get('technology', '')
            fraction = fes_cfd_fractions.get(carrier, 0.0)
            cap = row.get('capacity_mw', 0.0)

            if fraction >= 1.0:
                row_copy = row.copy()
                row_copy['support_type'] = 'CfD'
                rows_cfd.append(row_copy)
            elif fraction <= 0.0:
                row_copy = row.copy()
                row_copy['support_type'] = 'merchant'
                rows_merchant.append(row_copy)
            else:
                # Split into CfD and merchant sub-units
                cfd_row = row.copy()
                cfd_row['capacity_mw'] = cap * fraction
                cfd_row['support_type'] = 'CfD'
                for name_column in ('site_name', 'station_name'):
                    if name_column in cfd_row.index and pd.notna(cfd_row.get(name_column)):
                        cfd_row[name_column] = str(cfd_row[name_column]) + '_CfD'
                rows_cfd.append(cfd_row)

                merch_row = row.copy()
                merch_row['capacity_mw'] = cap * (1.0 - fraction)
                merch_row['support_type'] = 'merchant'
                for name_column in ('site_name', 'station_name'):
                    if name_column in merch_row.index and pd.notna(merch_row.get(name_column)):
                        merch_row[name_column] = str(merch_row[name_column]) + '_merchant'
                rows_merchant.append(merch_row)

        all_rows = rows_cfd + rows_merchant
        # Include any rows that weren't processed (shouldn't happen, but safe)
        if all_rows:
            df = pd.DataFrame(all_rows).reset_index(drop=True)
        else:
            df['support_type'] = 'merchant'

        n_cfd = (df['support_type'] == 'CfD').sum()
        n_merch = (df['support_type'] == 'merchant').sum()
        logger.info(f"  FES subsidy assignment: {n_cfd} CfD, {n_merch} merchant "
                     f"({len(df)} total rows, was {len(sites_df)})")

    # --- Assign region based on latitude ---
    if 'lat' in df.columns:
        df['region'] = np.where(df['lat'] > 55.5, 'north', 'south')
        n_north = (df['region'] == 'north').sum()
        n_south = (df['region'] == 'south').sum()
        logger.info(f"  Region assignment: {n_north} north, {n_south} south")
    else:
        df['region'] = 'south'  # default fallback
        logger.warning("  No lat column found — defaulting all to 'south'")

    return df


def _assign_subsidy_attributes_by_origin(sites_df, modelled_year, roc_years, fes_fractions, logger,
                                         roc_closure_date, default_roc_bandings):
    """Route each row to the subsidy branch its evidence supports.

    A mixed brownfield table holds REPD columns, so the plain function would
    send every row down the REPD branch and label the proxy increment
    'merchant' for lack of a CfD round. Instead:

    * named operational projects, and named pipeline projects REPD records a CfD
      round for, use the REPD branch (rounds, ROC banding, 20-year expiry,
      date-based ROC inference);
    * the proxy increment, and named pipeline projects with no recorded support,
      use the FES branch (``fes_cfd_fractions`` split), because a consented
      project without a round today is exactly what the future auction
      fractions describe.
    """
    if 'fleet_origin' not in sites_df.columns or not sites_df['fleet_origin'].eq('named').any():
        return _assign_subsidy_attributes(
            sites_df, modelled_year, roc_years, fes_fractions, logger,
            roc_closure_date=roc_closure_date, default_roc_bandings=default_roc_bandings,
        )
    named = sites_df['fleet_origin'].eq('named')
    small_scale = sites_df.get('cp30_spatial_method', pd.Series('', index=sites_df.index)).eq(SMALL_SCALE_SOLAR_METHOD)
    has_round = sites_df.get('cfd_round', pd.Series(index=sites_df.index)).fillna('').astype(str).str.strip().ne('')
    status = sites_df.get('development_status', pd.Series(index=sites_df.index)).fillna('')
    repd_branch = named & (status.eq('Operational') | has_round)
    repd_rows = sites_df.loc[repd_branch]
    fes_rows = sites_df.loc[~repd_branch & ~small_scale].drop(columns=['cfd_round', 'ro_banding'], errors='ignore')
    small_rows = sites_df.loc[~repd_branch & small_scale].drop(columns=['cfd_round', 'ro_banding'], errors='ignore')
    parts = []
    if len(small_rows):
        # FES small-scale (rooftop) solar: export tariffs, never a CfD round.
        logger.info("  Subsidy assignment, small-scale solar: %d rows (%.0f MW) merchant",
                    len(small_rows), small_rows['capacity_mw'].sum())
        parts.append(_assign_subsidy_attributes(
            small_rows, modelled_year, roc_years, {**(fes_fractions or {}), 'solar_pv': 0.0}, logger,
            roc_closure_date=roc_closure_date, default_roc_bandings=default_roc_bandings,
        ))
    if len(repd_rows):
        logger.info("  Subsidy assignment, REPD branch: %d named rows (%.0f MW)",
                    len(repd_rows), repd_rows['capacity_mw'].sum())
        parts.append(_assign_subsidy_attributes(
            repd_rows, modelled_year, roc_years, fes_fractions, logger,
            roc_closure_date=roc_closure_date, default_roc_bandings=default_roc_bandings,
        ))
    if len(fes_rows):
        logger.info("  Subsidy assignment, FES branch: %d increment/unlabelled rows (%.0f MW)",
                    len(fes_rows), fes_rows['capacity_mw'].sum())
        parts.append(_assign_subsidy_attributes(
            fes_rows, modelled_year, roc_years, fes_fractions, logger,
            roc_closure_date=roc_closure_date, default_roc_bandings=default_roc_bandings,
        ))
    return pd.concat(parts, ignore_index=True, sort=False)


def _apply_renewable_support_overrides(
    sites_df: pd.DataFrame, path: str | Path | None, logger: logging.Logger
) -> pd.DataFrame:
    """Apply exact, source-backed support labels before aggregation."""
    if not path:
        return sites_df
    overrides = pd.read_csv(path)
    required = {"source_site_name", "support_type", "ro_banding"}
    if missing := required.difference(overrides.columns):
        raise ValueError(f"Renewable support override file is missing: {sorted(missing)}")
    name_col = next(
        (column for column in ("source_site_name", "site_name") if column in sites_df.columns),
        None,
    )
    if name_col is None:
        return sites_df
    result = sites_df.copy()
    lookup = overrides.drop_duplicates("source_site_name").set_index("source_site_name")
    applied = 0
    for idx, name in result[name_col].items():
        if name not in lookup.index:
            continue
        row = lookup.loc[name]
        result.at[idx, "support_type"] = row["support_type"]
        result.at[idx, "ro_banding"] = float(row["ro_banding"])
        if "ro_accreditation_reference" in row.index:
            result.at[idx, "ro_accreditation_reference"] = row["ro_accreditation_reference"]
        result.at[idx, "support_override_source"] = str(path)
        applied += 1
    logger.info("Applied %d exact renewable support overrides from %s", applied, path)
    return result


def _explicit_support_mask(sites_df: pd.DataFrame) -> pd.Series:
    """Return source-recorded CfD/ROC sites that must remain auditable."""
    mask = pd.Series(False, index=sites_df.index)
    if "cfd_round" in sites_df.columns:
        cfd_round = sites_df["cfd_round"].fillna("").astype(str).str.strip()
        cfd_mask = cfd_round.ne("")
        if "cfd_capacity_mw" in sites_df.columns:
            cfd_capacity = pd.to_numeric(
                sites_df["cfd_capacity_mw"], errors="coerce"
            )
            cfd_mask &= cfd_capacity.gt(0)
        mask |= cfd_mask
    if "ro_banding" in sites_df.columns:
        mask |= pd.to_numeric(
            sites_df["ro_banding"], errors="coerce"
        ).fillna(0.0).gt(0.0)
    return mask


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


def _normalise_profile_name(value: str) -> str:
    """Fold case, non-breaking spaces and repeated whitespace for name matching."""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip().lower()


def select_profile_column(columns, site_name: str) -> Tuple[Optional[str], str]:
    """Pick a site's profile column without letting a longer name win on order.

    Profile columns carry a source suffix (``Beatrice (REPD)``) and many sites
    share a prefix (``Beatrice`` vs ``Beatrice Demonstrator``; ``Moray East``
    matches four columns). Taking the first substring hit silently gave the
    588 MW Beatrice farm the 10 MW demonstrator's profile, so the canonical
    ``<site> (SOURCE)`` form is resolved first and any genuinely ambiguous
    match is reported rather than settled by column order.

    Returns the chosen column (or None) and how it was matched.
    """
    target = _normalise_profile_name(site_name)
    if not target:
        return None, "no_site_name"
    normalised = {str(column): _normalise_profile_name(column) for column in columns}
    exact = [column for column, name in normalised.items() if name == target]
    if exact:
        return exact[0], "exact"
    # ``<site>`` optionally followed by a single parenthesised source tag.
    pattern = re.compile(rf"^{re.escape(target)}\s*\([^()]*\)$")
    canonical = [column for column, name in normalised.items() if pattern.match(name)]
    if len(canonical) == 1:
        return canonical[0], "canonical"
    if canonical:
        return sorted(canonical, key=lambda c: (len(normalised[c]), c))[0], "canonical_ambiguous"
    contains = [column for column, name in normalised.items() if target in name]
    if not contains:
        return None, "no_match"
    if len(contains) == 1:
        return contains[0], "substring"
    return sorted(contains, key=lambda c: (len(normalised[c]), c))[0], "substring_ambiguous"


def _standardise_profile_sites(frame: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """Mirror ``RenewableProfileGenerator._standardize_site_columns`` (profile rule).

    Kept in step with scripts/renewables/map_renewable_profiles.py; a unit test
    checks the two produce the same site table.
    """
    df = frame.copy()
    column_map = {}
    if 'Site Name' in df.columns:
        column_map['Site Name'] = 'site_name'
    elif 'site_name' not in df.columns and 'name' in df.columns:
        column_map['name'] = 'site_name'
    if 'Installed Capacity (MWelec)' in df.columns:
        column_map['Installed Capacity (MWelec)'] = 'capacity_mw'
    elif 'capacity_mw' not in df.columns and 'max capacity (GW)' in df.columns:
        df['capacity_mw'] = pd.to_numeric(df['max capacity (GW)'], errors='coerce') * 1000.0
    df = df.rename(columns=column_map)
    for column in ('site_name', 'capacity_mw', 'lat', 'lon'):
        if column not in df.columns:
            if column == 'capacity_mw':
                df[column] = 10.0
            elif column == 'site_name':
                df[column] = f"{source_type}_site_" + df.index.astype(str)
            else:
                df[column] = 0.0
    df['site_name'] = df['site_name'].astype(str) + f" ({source_type})"
    df = df[['site_name', 'capacity_mw', 'lat', 'lon']].copy()
    df = df.dropna(subset=['lat', 'lon'])
    df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce').fillna(10.0)
    return df


def profile_column_capacities(base_sites: pd.DataFrame, *pipeline_frames: pd.DataFrame) -> Dict[str, float]:
    """Capacity (MW) behind each column of a wind/solar site-profile file.

    The profile rule writes each site's output as capacity factor x its
    ``capacity_mw`` under the column ``site_name`` (pipeline sites tagged
    `` (Pipeline)``, operational names unchanged, a later duplicate name
    overwriting an earlier one). This rebuilds the same table from the same
    site files, so dividing a column by this capacity recovers the site's
    capacity factor whatever p_nom the unit ends up with.
    """
    combined = profile_column_sites(base_sites, *pipeline_frames)
    capacities: Dict[str, float] = {}
    for name, capacity in zip(combined['site_name'], combined['capacity_mw']):
        capacity = float(capacity) if pd.notna(capacity) else np.nan
        if not capacity > 0:
            continue
        capacities[str(name)] = capacity
    return capacities


def profile_column_locations(base_sites: pd.DataFrame, *pipeline_frames: pd.DataFrame) -> pd.DataFrame:
    """Site behind each column of a wind/solar site-profile file: ``lat``, ``lon`` (WGS84), ``capacity_mw``.

    The same site table as ``profile_column_capacities`` (a later duplicate name wins), indexed by column.
    """
    combined = profile_column_sites(base_sites, *pipeline_frames)
    if combined.empty:
        return pd.DataFrame(columns=['lat', 'lon', 'capacity_mw'])
    combined = combined.assign(site_name=combined['site_name'].astype(str))
    combined = combined[pd.to_numeric(combined['capacity_mw'], errors='coerce').gt(0)]
    return combined.drop_duplicates(subset=['site_name'], keep='last').set_index('site_name')[
        ['lat', 'lon', 'capacity_mw']]


def profile_column_sites(base_sites: pd.DataFrame, *pipeline_frames: pd.DataFrame) -> pd.DataFrame:
    """The site rows behind a wind/solar site-profile file, as the profile rule combines them."""
    frames = [f for f in pipeline_frames if f is not None and not f.empty]
    base = base_sites.copy() if base_sites is not None else pd.DataFrame()
    if 'Site Name' in base.columns and 'site_name' not in base.columns:
        base = base.rename(columns={'Site Name': 'site_name'})
    if frames:
        pipeline = pd.concat([_standardise_profile_sites(f, 'Pipeline') for f in frames],
                             ignore_index=True).drop_duplicates(subset=['site_name'], keep='last')
        if not base.empty:
            pipeline = pipeline[~pipeline['site_name'].str.replace(r' \(Pipeline\)$', '', regex=True)
                                .isin(base['site_name'].astype(str))]
            combined = pd.concat([base, pipeline], ignore_index=True, sort=False)
        else:
            combined = pipeline
    else:
        combined = base
    if combined.empty or 'site_name' not in combined.columns:
        return pd.DataFrame(columns=['site_name', 'capacity_mw', 'lat', 'lon'])
    combined = combined.copy()
    for column in ('lat', 'lon', 'capacity_mw'):
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors='coerce')
    return combined.dropna(subset=['lat', 'lon'])


def build_profile_capacity_map(inputs, logger: logging.Logger) -> Dict[str, Dict[str, float]]:
    """``{carrier: {profile column: capacity MW}}`` from the rule's site files."""
    def _read(path):
        if not path or not Path(str(path)).exists():
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    result: Dict[str, Dict[str, float]] = {}
    for carrier in ('wind_onshore', 'wind_offshore', 'solar_pv'):
        base = _read(getattr(inputs, f'{carrier}_sites', None))
        pipelines = []
        if carrier == 'wind_offshore':
            pipelines.append(_read(getattr(inputs, 'offshore_pipeline_processed', None)))
        pipelines.append(_read(getattr(inputs, f'{carrier}_pipeline_sites', None)))
        capacities = profile_column_capacities(base, *pipelines)
        if capacities:
            result[carrier] = capacities
            logger.info("Profile column capacities for %s: %d columns (%.0f MW)",
                        carrier, len(capacities), sum(capacities.values()))
    # Marine and hydro profiles are written one column per site_name of their
    # site file (capacity factor x capacity_mw), later rows of a name winning.
    for carrier in ('tidal_stream', 'shoreline_wave', 'tidal_lagoon', 'large_hydro', 'small_hydro'):
        sites = _read(getattr(inputs, f'{carrier}_sites', None))
        if sites.empty or not {'site_name', 'capacity_mw'} <= set(sites.columns):
            continue
        capacity = pd.to_numeric(sites['capacity_mw'], errors='coerce')
        capacities = {
            str(name): float(value)
            for name, value in zip(sites['site_name'], capacity) if pd.notna(value) and value > 0
        }
        if capacities:
            result[carrier] = capacities
    # CP30 "marine" units read the tidal-stream profile file.
    if 'tidal_stream' in result:
        result.setdefault('marine', result['tidal_stream'])
    return result


def build_profile_location_map(inputs) -> Dict[str, pd.DataFrame]:
    """``{carrier: site table indexed by profile column}`` for wind and solar, from the rule's site files."""
    def _read(path):
        if not path or not Path(str(path)).exists():
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    result: Dict[str, pd.DataFrame] = {}
    for carrier in ('wind_onshore', 'wind_offshore', 'solar_pv'):
        base = _read(getattr(inputs, f'{carrier}_sites', None))
        pipelines = []
        if carrier == 'wind_offshore':
            pipelines.append(_read(getattr(inputs, 'offshore_pipeline_processed', None)))
        pipelines.append(_read(getattr(inputs, f'{carrier}_pipeline_sites', None)))
        locations = profile_column_locations(base, *pipelines)
        if len(locations):
            result[carrier] = locations
    return result


def add_renewable_generators(
    network: pypsa.Network,
    sites_df: pd.DataFrame,
    profiles_dir: Optional[str] = None,
    p_nom_col: str = 'capacity_mw',
    carrier_col: str = 'technology',
    overwrite: bool = False,
    snapshot_weighting: Optional[pd.Series] = None,
    renewables_year: Optional[int] = None,
    wind_calib_factors: Optional[Dict[str, float]] = None,  # kept for back-compat
    performance_factors: Optional[Dict[str, float]] = None,
    performance_factor_registry: Optional[pd.DataFrame] = None,
    performance_factor_context: Optional[Dict[str, int]] = None,
    commissioning_profile_gating: bool = False,
    commissioning_date_overrides: Optional[Dict[str, str]] = None,
    default_profiles: Optional[Dict[str, pd.Series]] = None,
    profile_capacities: Optional[Dict[str, Dict[str, float]]] = None,
    proxy_profiles: Optional[dict] = None,
    profile_audit: Optional[dict] = None,
    profile_locations: Optional[Dict[str, pd.DataFrame]] = None,
) -> pypsa.Network:
    """Add renewable generators with capacity-factor profiles.

    ``proxy_profiles`` is the scenario's ``renewables`` block. With
    ``proxy_profile_basis: regional`` a unit with no site profile column takes
    the capacity-weighted profile of the named sites of its carrier in its own
    area (``scripts/generators/proxy_profiles.py``) instead of the GB fleet
    average; ``profile_audit`` (a dict the caller supplies) receives the groups
    and the per-unit assignment.

    ``default_profiles`` ({carrier: p_max_pu series}) replaces the flat
    placeholder capacity factor for units of that carrier with no matching site
    profile column (large hydro: a CP30-calibrated, monthly-shaped inflow
    profile from ``scripts/renewables/large_hydro_inflow.py``).

    ``profile_locations`` ({carrier: site table by profile column}, from
    :func:`build_profile_location_map`) locates every site profile column; with
    ``proxy_profile_members: all_sites`` an area's group is every located site
    in it, and a unit without a region takes the region of its coordinates.

    ``profile_capacities`` ({carrier: {column: MW}}, from
    :func:`build_profile_capacity_map`) is the capacity each MW site-profile
    column was built on. A column is divided by it, not by the unit's p_nom:
    a unit trimmed by the ES5 reconciliation or split into ROC/merchant parts
    has a p_nom below its site's, and dividing by that p_nom over-scaled its
    capacity factor (Mona 883 of 1,500 MW at CF 0.65; 112 units, 4.45 GW;
    audit 27 Sep 2026). Columns absent from the map keep the p_nom divisor.
    """
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
            'small_hydro': 'small_hydro',
            # CP30/FES "Marine" carries no wave split and the fleet is tidal
            # stream (MeyGen and the AR4-AR6 tidal CfDs). With no mapping here
            # marine got no profile in this stage and fell to the thermal-stage
            # fallback, which averaged the site profiles in MW and read them as
            # a capacity factor: CF 0.467 and output pinned at 1.0 in 16% of
            # hours against a capacity-normalised fleet CF of 0.218 (audit 26
            # Sep 2026). Here it takes the capacity-normalised fleet average.
            'marine': 'tidal_stream',
        }
        for tech in valid_sites[carrier_col].unique():
            profile_name = tech_to_profile.get(tech, tech.lower().replace(' ', '_'))
            # Use year-specific glob to avoid loading profiles from wrong weather year
            if renewables_year is not None:
                profile_files = list(Path(profiles_dir).glob(f"{profile_name}_{renewables_year}.csv"))
                if not profile_files:
                    # Fallback: try broader glob but prefer correct year
                    all_files = sorted(Path(profiles_dir).glob(f"*{profile_name}*.csv"))
                    year_str = str(renewables_year)
                    profile_files = [f for f in all_files if year_str in f.name]
                    if not profile_files:
                        profile_files = all_files
                        if profile_files:
                            logger.warning(f"No {renewables_year} profile for {tech}, using {profile_files[0].name}")
            else:
                profile_files = sorted(Path(profiles_dir).glob(f"*{profile_name}*.csv"))
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
    for carrier, default_profile in (default_profiles or {}).items():
        if carrier not in profiles and carrier in set(valid_sites[carrier_col]):
            # No profile file for this carrier: an empty frame routes its units
            # to the default profile rather than an implicit static p_max_pu 1.0.
            profiles[carrier] = pd.DataFrame(index=default_profile.index)
    generator_characteristics = load_generator_characteristics()
    logger.info(f"Loaded characteristics for {len(generator_characteristics)} generator types")
    logger.info(f"Loading profiles for {len(profiles)} technologies: {profile_technologies}")
    quality_metrics = validate_renewable_data_quality(sites_df=valid_sites, profiles=profiles, logger=logger, p_nom_col=p_nom_col)
    generators_to_add = []
    profiles_to_add = {}
    # Future fleet entries commonly do not have a one-to-one historical profile
    # column. Cache the carrier fleet-average capacity factor so it is computed
    # once, not once for every unmatched FES/CP30 entry.
    fleet_average_profiles: Dict[str, pd.Series] = {}    # Initialize with existing network generators to avoid duplicates
    profile_normalisation: Dict[str, str] = {}
    fleet_average_basis: Dict[str, str] = {}
    profile_column_users: Dict[Tuple[str, str], List[str]] = {}
    gen_names_used = set(network.generators.index)  # Fast O(1) lookup for duplicate names
    total_sites = len(valid_sites)

    def _unit_performance_factor(site, carrier):
        """(factor, source, resource region) for one unit, from the registry or carrier fallback."""
        resource_region = site.get('resource_region')
        if pd.isna(resource_region) or not str(resource_region).strip():
            resource_region = resource_region_for_site(carrier, site)
        _perf = performance_factors or wind_calib_factors or {}
        fallback_factor = float(_perf.get(carrier, 1.0))
        factor, factor_source = select_performance_factor(
            performance_factor_registry,
            carrier,
            str(resource_region),
            (performance_factor_context or {}).get('calendar_year'),
            (performance_factor_context or {}).get('calendar_month'),
            fallback_factor,
            weather_year=(performance_factor_context or {}).get('weather_year'),
        )
        return factor, factor_source, resource_region

    def _national_fleet_average(carrier, profile_df):
        """The carrier's GB capacity-weighted fleet-average profile, cached."""
        if carrier not in fleet_average_profiles:
            total_fleet_output = profile_df.sum(axis=1)
            # Normalise by the capacity each column was built
            # on: a column's peak is below its capacity
            # wherever the site never reaches rated output
            # (solar peaks at ~0.8), and the peak basis put the
            # solar increments at CF 0.145 against 0.118 for
            # named solar (audit 27 Sep 2026). Peak stays the
            # fallback for columns with no known capacity.
            site_capacities = profile_df.max()
            known = (profile_capacities or {}).get(carrier, {})
            if known:
                mapped = pd.Series(
                    [known.get(str(c), np.nan) for c in profile_df.columns],
                    index=profile_df.columns, dtype=float,
                )
                site_capacities = mapped.where(mapped.gt(0), site_capacities)
                fleet_average_basis[carrier] = (
                    f"capacity ({int(mapped.gt(0).sum())}/{len(mapped)} columns)"
                )
            else:
                fleet_average_basis[carrier] = "peak"
            total_site_capacity = site_capacities.sum()
            if total_site_capacity > 0:
                fleet_average_profiles[carrier] = (
                    total_fleet_output / total_site_capacity
                )
                logger.info(
                    "Cached fleet-average CF for %s from %d profile sites "
                    "(mean CF=%.3f, normalised by %s)",
                    carrier,
                    len(profile_df.columns),
                    fleet_average_profiles[carrier].mean(),
                    fleet_average_basis[carrier],
                )
            else:
                fleet_average_profiles[carrier] = profile_df.mean(axis=1)
                logger.warning(
                    "Zero total profile capacity for %s; cached mean profile",
                    carrier,
                )
        return fleet_average_profiles[carrier]

    def _matched_profile_column(site, idx, profile_df):
        """The unit's own site profile column, as the loop below selects it (None if unmatched)."""
        column = None
        if 'site_name' in site and pd.notna(site['site_name']):
            column, _ = select_profile_column(profile_df.columns, site['site_name'])
        if (column is None and 'source_site_name' in site and pd.notna(site['source_site_name'])
                and str(site['source_site_name']).strip() != str(site.get('site_name', '')).strip()):
            column, _ = select_profile_column(profile_df.columns, site['source_site_name'])
        if column is None and str(idx) in profile_df.columns:
            column = str(idx)
        return column

    # Regional proxy profiles (15 Oct 2026): each unmatched unit of a listed
    # carrier takes the profile of the named sites in its own area. Keyed by
    # the unit's position in valid_sites.
    proxy_settings = proxy_profile_settings(proxy_profiles)
    regional_proxy_profiles: Dict[int, pd.Series] = {}
    proxy_group_label: Dict[int, str] = {}
    proxy_gen_names: Dict[int, str] = {}
    proxy_audit_parts = {"assignments": [], "groups": []}
    if proxy_settings["basis"] == "regional":
        members_by_carrier: Dict[str, list] = {}
        proxies_by_carrier: Dict[str, list] = {}
        all_sites = proxy_settings["members"] == "all_sites"
        unit_regions = pd.Series(dtype=object)
        if all_sites:
            # A unit without a region of its own (an FES capacity block) takes the region of its coordinates.
            unit_regions = regions_of_points(
                valid_sites.get("lat", pd.Series(np.nan, index=valid_sites.index)).to_numpy(),
                valid_sites.get("lon", pd.Series(np.nan, index=valid_sites.index)).to_numpy(),
                proxy_settings["region_polygon_file"], proxy_settings["region_group_file"],
            )
        for pos, (idx, site) in enumerate(valid_sites.iterrows()):
            carrier = site[carrier_col]
            if carrier not in proxy_settings["carriers"] or carrier not in profiles:
                continue
            profile_df = profiles[carrier]
            if len(profile_df.columns) <= 1 or carrier in (default_profiles or {}):
                continue
            if all_sites and pd.isna(site.get("cp30_region", np.nan)) and pd.notna(unit_regions.iloc[pos]):
                site = site.copy()
                site["cp30_region"] = unit_regions.iloc[pos]
            keys = area_keys(site, carrier)
            column = _matched_profile_column(site, idx, profile_df)
            if all_sites and column is not None:
                continue  # its own profile; the area groups come from the site table below
            if column is not None:
                capacity = float((profile_capacities or {}).get(carrier, {}).get(str(column), np.nan))
                if not capacity > 0:
                    capacity = float(profile_df[column].max())
                members_by_carrier.setdefault(carrier, []).append(
                    {"column": column, "capacity": capacity, "keys": keys})
            else:
                factor, _, _ = _unit_performance_factor(site, carrier)
                proxies_by_carrier.setdefault(carrier, []).append(
                    {"pos": pos, "keys": keys, "weight": float(site[p_nom_col]) * float(factor)})
        for carrier, rows in proxies_by_carrier.items():
            profile_df = profiles[carrier]
            if all_sites:
                locations = (profile_locations or {}).get(carrier, pd.DataFrame(columns=["lat", "lon", "capacity_mw"]))
                located = locations.loc[locations.index.isin(profile_df.columns)]
                members = site_members(
                    carrier, located, profile_df.columns,
                    regions_of_points(located["lat"].to_numpy(), located["lon"].to_numpy(),
                                      proxy_settings["region_polygon_file"],
                                      proxy_settings["region_group_file"]).set_axis(located.index),
                )
                logger.info("Proxy %s profile groups from all site profiles: %d of %d columns located",
                            carrier, len(members), profile_df.shape[1])
            else:
                members = pd.DataFrame(members_by_carrier.get(carrier, []), columns=["column", "capacity", "keys"])
            built, assignments, groups, _scale = build_regional_proxy_profiles(
                carrier, profile_df, _national_fleet_average(carrier, profile_df), members,
                pd.DataFrame(rows).set_index("pos"), (profile_capacities or {}).get(carrier), proxy_settings,
                logger,
            )
            regional_proxy_profiles.update(built)
            for row in assignments.itertuples(index=False):
                proxy_group_label[int(row.row)] = f"{row.level}:{row.group}"
            proxy_audit_parts["assignments"].append(assignments)
            proxy_audit_parts["groups"].append(groups)
            for g in groups.itertuples(index=False):
                logger.info(
                    "Proxy %s profile group %s:%s - %d named sites, %d proxy units, CF %.3f (scaled %.3f) "
                    "vs GB %.3f, correlation with GB %.3f", carrier, g.level, g.group, g.named_sites,
                    g.proxy_units, g.cf_raw, g.cf_scaled, g.cf_national, g.corr_with_national,
                )

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
        # Reserve the name immediately.  Several future-capacity allocations
        # can legitimately split one FES profile/site across different
        # electrical connection points while retaining the same site_name for
        # profile lookup.  Without this reservation every split row selects the
        # same name and is later skipped during the bulk PyPSA add operation.
        gen_names_used.add(gen_name)
        bus = site['bus']
        p_nom = site[p_nom_col]
        carrier = site[carrier_col]
        factor, factor_source, resource_region = _unit_performance_factor(site, carrier)
        if gen_name in network.generators.index and not overwrite:
            logger.warning(f"Generator {gen_name} already exists, skipping")
            continue
        if bus not in network.buses.index:
            logger.warning(f"Bus {bus} not found in network, skipping generator {gen_name}")
            continue
        gen_attrs = {
            'name': gen_name, 'bus': bus, 'p_nom': float(p_nom), 'carrier': carrier,
            # Physical profile metadata.  This deliberately contains no
            # support_type: CfD/ROC status is used for accounting, never for a
            # weather/performance correction.
            'resource_region': str(resource_region),
            'renewable_performance_factor': factor,
            'renewable_performance_factor_source': factor_source,
        }
        if i in proxy_group_label:
            gen_attrs['proxy_profile_group'] = proxy_group_label[i]
            proxy_gen_names[i] = gen_name
        if 'x_coord' in site and pd.notna(site['x_coord']):
            gen_attrs['lon'] = float(site['x_coord'])
        if 'y_coord' in site and pd.notna(site['y_coord']):
            gen_attrs['lat'] = float(site['y_coord'])
        elif 'lat' in site and 'lon' in site and pd.notna(site['lat']) and pd.notna(site['lon']):
            gen_attrs['lon'] = float(site['lon'])
            gen_attrs['lat'] = float(site['lat'])
        
        # Pass through subsidy tracking and region attributes (if assigned by Stage 5.75)
        for attr in [
            'support_type', 'ro_banding', 'cfd_round', 'region',
            'source_site_name', 'repd_ref_id', 'repd_old_ref_id',
            'operational_date', 'source_operational_date',
            'ro_accreditation_reference', 'support_override_source',
            'ro_support_source', 'cfd_id', 'cfd_support_source', 'repd_cfd_round_unconfirmed',
            'fleet_origin', 'development_status', 'cfd_capacity_mw',
            'cp30_region', 'landing_evidence', 'connection_allocation_source',
            'cfd_contract_part', 'cfd_technology',
        ]:
            if attr in site and pd.notna(site[attr]):
                gen_attrs[attr] = site[attr]

        # Retain the provenance of an explicit CP30 connection allocation on
        # the PyPSA component. The post-aggregation ETYS correction uses
        # network.generators, rather than the source table, and must therefore
        # retain this marker to avoid overriding the declared connection point.
        for attr in ['data_source', 'spatial_proxy_source']:
            if attr in site and pd.notna(site[attr]):
                gen_attrs[attr] = site[attr]
        
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
                profile_column, match_kind = select_profile_column(
                    profile_df.columns, site['site_name']
                )
                if match_kind.endswith("_ambiguous"):
                    logger.warning(
                        "Ambiguous profile column for %s (%s): site '%s' matched several "
                        "columns; using '%s'. Disambiguate the site or profile names.",
                        gen_name, carrier, str(site['site_name']).strip(), profile_column,
                    )
            if (profile_column is None and 'source_site_name' in site
                    and pd.notna(site['source_site_name'])
                    and str(site['source_site_name']).strip() != str(site.get('site_name', '')).strip()):
                # A named project split into support tranches carries a
                # suffixed site_name ("<site>_CfD"); its source name still
                # identifies its own profile column (G2, audit 27 Sep 2026).
                profile_column, match_kind = select_profile_column(
                    profile_df.columns, site['source_site_name']
                )
            if profile_column is None:
                index_str = str(idx)
                if index_str in profile_df.columns:
                    profile_column = index_str
            use_default_profile = profile_column is None and carrier in (default_profiles or {})
            single_column_fallback = False
            if profile_column is None and not use_default_profile and len(profile_df.columns) == 1:
                profile_column = profile_df.columns[0]
                single_column_fallback = True
            if use_default_profile:
                # Calibrated default (large hydro: CP30 energy, weather-year
                # monthly inflow shape) instead of the flat placeholder below
                # or some other station's single profile column.
                profile_series = default_profiles[carrier].copy()
            elif profile_column is None:
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
                        # For unmatched generators aggregating FES/CP30 capacity,
                        # use the same capacity-weighted fleet profile for every
                        # entry of this carrier. Computing it inside the site loop
                        # used to repeat a full 8,784 x N-column reduction for each
                        # FES entry, making month-long scenarios needlessly slow.
                        if i in regional_proxy_profiles:
                            # The named sites of this carrier in the unit's own area.
                            profile_series = regional_proxy_profiles[i].copy()
                        else:
                            profile_series = _national_fleet_average(carrier, profile_df).copy()
                    else:
                        continue
            else:
                profile_series = profile_df[profile_column]
                logger.debug(f"Using specific profile column '{profile_column}' for {gen_name}")
                ref = site.get('repd_ref_id') if 'repd_ref_id' in site else None
                profile_column_users.setdefault((carrier, str(profile_column)), []).append(
                    (gen_name, str(ref) if pd.notna(ref) else gen_name)
                )
                if single_column_fallback and profile_series.max() > 1.0:
                    # The only column is some other site's MW series; dividing
                    # it by this unit's p_nom (below) would scale it by the
                    # ratio of two unrelated capacities. Normalise by that
                    # site's own capacity (its peak when unknown), the same
                    # basis the fleet average uses.
                    own = float((profile_capacities or {}).get(carrier, {}).get(str(profile_column), np.nan))
                    profile_series = profile_series / (own if own > 0 else profile_series.max())
            column_capacity = np.nan
            if profile_column is not None and not use_default_profile and not single_column_fallback:
                column_capacity = float(
                    (profile_capacities or {}).get(carrier, {}).get(str(profile_column), np.nan)
                )
            if i in regional_proxy_profiles:
                # Already a capacity factor (capacity-normalised, energy-scaled):
                # its peak can exceed 1.0 before the performance factor, so it
                # must not be read as MW. The clip below applies after the factor.
                logger.debug(f"Using regional proxy capacity factor profile for {gen_name}")
            elif profile_series.max() > 1.0 and column_capacity > 0:
                # Site profile in MW: divide by the capacity it was built on, so
                # a trimmed or split unit keeps its site's capacity factor.
                profile_series = profile_series / column_capacity
                profile_normalisation[gen_name] = 'column_capacity'
            elif profile_series.max() > 1.0 and p_nom > 0:
                # Individual site profile in MW - convert to CF by dividing by generator capacity
                profile_series = profile_series / p_nom
                profile_normalisation[gen_name] = 'p_nom'
                logger.debug(f"Converted MW profile to capacity factor for {gen_name}")
            elif profile_series.max() <= 1.0:
                logger.debug(f"Using capacity factor profile for {gen_name}")
            else:
                logger.warning(f"Profile values > 1.0 for {gen_name} with zero capacity, using zero profile")
                profile_series = profile_series * 0.0
            # Apply the selected physical factor before clipping.  The
            # registry can refine a carrier factor by broad resource region;
            # support labels are intentionally absent from this selection.
            if factor != 1.0:
                profile_series = profile_series * factor
                logger.debug(
                    "Applied performance factor %.4f for %s (%s, %s; %s)",
                    factor, gen_name, carrier, resource_region, factor_source,
                )
            # Clip to valid CF range [0, 1]
            profile_series = profile_series.clip(0, 1)
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
                                # A leap modelled year (2040) from a weather year without 29 February
                                # takes 28 February's hours for it, not a day interpolated across it.
                                profile_series = fill_leap_day(profile_series)
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
            if commissioning_profile_gating and "operational_date" in site:
                site_name = str(site.get("site_name", ""))
                activation_value = (commissioning_date_overrides or {}).get(site_name, site["operational_date"])
                operational_date = _parse_operational_date(activation_value)
                if pd.notna(operational_date):
                    operational_date = pd.Timestamp(operational_date).tz_localize(None)
                    profile_series.loc[profile_series.index < operational_date] = 0.0
            profiles_to_add[gen_name] = profile_series
    if profile_audit is not None:
        profile_audit["basis"] = proxy_settings["basis"]
        profile_audit["settings"] = {k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in proxy_settings.items()}
        if proxy_audit_parts["groups"]:
            assignments = pd.concat(proxy_audit_parts["assignments"], ignore_index=True)
            assignments["generator"] = assignments["row"].map(proxy_gen_names)
            profile_audit["groups"] = pd.concat(proxy_audit_parts["groups"], ignore_index=True)
            profile_audit["assignments"] = assignments.drop(columns=["row"])
    if profile_normalisation:
        counts = pd.Series(profile_normalisation).value_counts().to_dict()
        logger.info("MW site profiles normalised by: %s", counts)
    shared = {
        key: users for key, users in profile_column_users.items()
        if len({ref for _, ref in users}) > 1
    }
    if shared:
        # Distinct projects sharing one name resolve to one profile column (the
        # profile rule keeps only the last site of a name). With column
        # capacities their scale is right; the location is the namesake's.
        logger.warning(
            "%d profile columns serve more than one distinct project (%d units), e.g. %s",
            len(shared), sum(len(u) for u in shared.values()),
            "; ".join(f"{c}:'{col}'" for (c, col) in list(shared)[:5]),
        )
    generators_added = 0
    skipped_existing = 0
    for gen_attrs in generators_to_add:
        try:
            gen_name = gen_attrs.pop('name')
            # Skip if generator already exists (avoid PyPSA print warning)
            if gen_name in network.generators.index:
                skipped_existing += 1
                continue
            network.add("Generator", gen_name, **gen_attrs)
            generators_added += 1
        except Exception as e:
            logger.error(f"Failed to add generator {gen_name}: {e}")
    if skipped_existing > 0:
        logger.debug(f"Skipped {skipped_existing} generators (already exist in network)")
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

    # Standardize coordinates: ensure all generators have WGS84 lon/lat
    from scripts.utilities.spatial_utils import standardize_component_coordinates
    coord_result = standardize_component_coordinates(network, components=['Generator'])
    if coord_result.get('Generator', 0) > 0:
        logger.info(f"Standardized WGS84 coordinates for {coord_result['Generator']} generators")

    return network


def build_large_hydro_default_profile(
    network: pypsa.Network,
    scenario_config: dict,
    renewables_year: Optional[int],
    shape_path: Optional[str],
    cp30_es1_path: Optional[str],
    log: logging.Logger,
) -> Optional[pd.Series]:
    """Default ``p_max_pu`` for large-hydro units with no site profile column.

    The annual inflow CF is CP30's own (ES1 hydro generation / capacity x
    hours, 0.210 for FFR 2030) for a CP30 scenario, else
    ``hydro.large_hydro.proxy_inflow_capacity_factor`` when configured; with
    neither the legacy flat placeholder is kept. The monthly shape is ESPENI
    metered hydro output in the renewables weather year. See
    ``scripts/renewables/large_hydro_inflow.py`` for why the profile is written
    in the reservoir model's ``inflow / inflow_profile_factor`` convention.
    """
    if not shape_path or len(network.snapshots) == 0:
        return None
    hydro_cfg = (scenario_config.get('renewables', {}) or {}).get('hydro', {}) or {}
    large_cfg = hydro_cfg.get('large_hydro', {}) or {}
    modelled_year = int(scenario_config.get('modelled_year', network.snapshots[0].year))
    annual_cf, cf_source = None, None
    if target_source(scenario_config) == 'CP30' and cp30_es1_path:
        hours = 8784.0 if pd.Timestamp(year=modelled_year, month=12, day=31).dayofyear == 366 else 8760.0
        annual_cf = cp30_hydro_capacity_factor(
            cp30_es1_path, scenario_config.get('CP30_pathway'), modelled_year, hours=hours)
        cf_source = f"CP30 ES1 hydro generation / capacity ({scenario_config.get('CP30_pathway')} {modelled_year})"
    elif large_cfg.get('proxy_inflow_capacity_factor') is not None:
        annual_cf = float(large_cfg['proxy_inflow_capacity_factor'])
        cf_source = "hydro.large_hydro.proxy_inflow_capacity_factor"
    if annual_cf is None:
        return None
    reservoir = bool(hydro_cfg.get('enabled', False) and large_cfg.get('reservoir_model', False))
    inflow_factor = float(large_cfg.get('inflow_profile_factor', 0.55)) if reservoir else None
    shape, shape_label = load_monthly_shape(shape_path, renewables_year)
    profile = large_hydro_proxy_p_max_pu(network.snapshots, annual_cf, shape, inflow_factor)
    implied_cf = inflow_capacity_factor(profile, inflow_factor)
    log.info(
        "Large-hydro default profile for units without a site profile: inflow CF %.3f (%s), "
        "monthly shape %s (min %.2f, max %.2f x mean); %s",
        annual_cf, cf_source, shape_label, shape.min(), shape.max(),
        (f"reservoir model on, p_max_pu = inflow / {inflow_factor:.2f} (mean {profile.mean():.3f}, "
         f"max {profile.max():.3f})") if reservoir else "reservoir model off, p_max_pu = inflow",
    )
    if implied_cf < annual_cf * 0.999 and len(network.snapshots) >= 8760:
        log.warning("Large-hydro default profile clipped at 1.0: inflow CF %.3f of %.3f target",
                    implied_cf, annual_cf)
    return profile


def log_applied_performance_factors(
    network: pypsa.Network,
    carrier_factors: Optional[Dict[str, float]],
    log: logging.Logger,
) -> pd.DataFrame:
    """Log the performance factors the generators actually carry.

    Until 26 Sep 2026 the log printed only the stacked carrier config (onshore
    0.805, solar 0.894) while the CP30 network carried 0.850 and 0.861 from
    stale ``physical_default`` rows of the registry, which take precedence.
    This prints what was applied, capacity-weighted by carrier and source, and
    notes wherever it differs from the carrier config.
    """
    gens = network.generators
    if 'renewable_performance_factor' not in gens.columns:
        return pd.DataFrame()
    applied = gens.dropna(subset=['renewable_performance_factor'])
    if applied.empty:
        return pd.DataFrame()
    applied = applied.assign(
        weighted=applied['renewable_performance_factor'].astype(float) * applied['p_nom'].astype(float),
        source=applied.get('renewable_performance_factor_source', pd.Series('unknown', index=applied.index)).fillna('unknown'),
    )
    table = applied.groupby(['carrier', 'source']).agg(
        p_nom=('p_nom', 'sum'), weighted=('weighted', 'sum'),
        factor_min=('renewable_performance_factor', 'min'),
        factor_max=('renewable_performance_factor', 'max'),
    )
    table['factor'] = table['weighted'] / table['p_nom'].where(table['p_nom'] > 0)
    log.info("Renewable performance factors applied (capacity-weighted):")
    for (carrier, source), row in table.iterrows():
        configured = (carrier_factors or {}).get(carrier)
        note = ""
        if configured is not None and abs(row['factor'] - configured) > 1e-4:
            note = f" (carrier config {configured:.4f}; the {source} value takes precedence)"
        log.info("  %s [%s]: %.4f (range %.4f-%.4f) on %.0f MW%s", carrier, source, row['factor'],
                 row['factor_min'], row['factor_max'], row['p_nom'], note)
    return table.drop(columns='weighted')

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
    logger: logging.Logger,
    licence_areas: Optional[LicenceAreas] = None,
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
        licence_areas: The FES long-range chain's licence areas by GSP polygon
            (``spatial_allocation.LicenceAreas``); None keeps the latitude bands (the CP30 proxy rows,
            reconciled to ES5 regions afterwards).
        
    Returns:
        List of dicts with keys: bus, lat, lon, capacity_mw
    """
    # Load historical site data
    historical_sites = _load_historical_renewable_sites(technology, logger)
    
    if len(historical_sites) == 0:
        logger.warning(f"No historical sites for {technology}, using geographic distribution")
        return _distribute_by_region_geography(capacity_mw, region, network, logger)
    
    # Filter by region: the GSP polygon's licence area, else approximate latitude bands
    if licence_areas is not None:
        sites = historical_sites[licence_areas.site_licence(historical_sites['lat'], historical_sites['lon'])
                                 .eq(region).to_numpy()]
    elif region == 'SHETL':
        sites = historical_sites[historical_sites['lat'] > 57.0]
    elif region == 'SPTL':
        sites = historical_sites[(historical_sites['lat'] > 55.5) & (historical_sites['lat'] <= 57.0)]
    else:  # NGET
        sites = historical_sites[historical_sites['lat'] <= 55.5]
    
    fallback = len(sites) == 0
    if fallback:
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
        
        # Find nearest bus (of the same licence area, on the FES long-range chain)
        lat, lon = site['lat'], site['lon']
        bus = (licence_areas.nearest_bus(lat, lon, region) if licence_areas is not None
               else _find_nearest_bus(network, lat, lon))
        if fallback and licence_areas is not None and bus:
            # 8 Nov 2026: a site outside the licence lends its weight only. The row takes its licence bus's
            # coordinates, or later coordinate rules undo the licence (FES 2050 SPT marine, weighted on
            # EMEC's Orkney sites, was put on CUPA1Q and then moved to the Orkney hub at FINS1-).
            lat, lon = licence_areas.bus_latlon(bus)
        
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
    logger: logging.Logger,
    gsp_placement: str = "nearest_bus",
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
    
    # GSP-to-bus mapping from this FES edition's committed GSP info table
    gsp_to_bus = _load_gsp_bus_mapping(network, logger, fes_edition_of_file(fes_path))
    # gsp_placement "demand_node" (the FES long-range chain, 27 Oct 2026): a GSP row's plant goes on
    # that GSP's demand node, as the 2030 build places distributed plant (gsp_nodes.GspDemandNodes);
    # "nearest_bus" (the CP30 proxy rows, which are rescaled and reconciled to ES5 afterwards): the bus
    # nearest the GSP's point.
    if gsp_placement not in ("nearest_bus", "demand_node"):
        raise ValueError(f"gsp_placement must be 'nearest_bus' or 'demand_node', not {gsp_placement!r}")
    demand_nodes = GspDemandNodes(network, log=logger) if gsp_placement == "demand_node" else None
    # The same chain splits Direct capacity by licence area (GSP polygons), not latitude bands.
    licence_areas = LicenceAreas(network.buses) if gsp_placement == "demand_node" else None
    placement_methods: Dict[str, float] = {}
    
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
            # By GSP id where the row has one, else by name (fuzzy, for name variations)
            bus_info = _locate_gsp(gsp, row.get('GSP ID', None), gsp_to_bus)
            if bus_info:
                bus = bus_info.get('bus')
                lat = bus_info.get('lat')
                lon = bus_info.get('lon')
                method = "nearest_bus"
                if demand_nodes is not None:
                    gsp_id = row.get('GSP ID', None)
                    if gsp_id is None or pd.isna(gsp_id) or not str(gsp_id).strip():
                        gsp_id = bus_info.get('gsp_id')
                    node, how = demand_nodes.node(gsp_id, lat, lon)
                    if node is not None:
                        bus, method = node, f"demand_node:{how}"
                    else:
                        method = f"nearest_bus:{how}"
                placement_methods[method] = placement_methods.get(method, 0.0) + float(capacity)
        
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
                    logger=logger,
                    licence_areas=licence_areas,
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
        
        logger.info("  GSP placement (%s), MW by method: %s", gsp_placement,
                    {k: round(v, 1) for k, v in sorted(placement_methods.items())})
    else:
        logger.warning(f"No FES renewable generators found for year {modelled_year}")

    # A GSP row neither the edition's GSP info nor the declared points can place is refused, as
    # the storage rows are (user decision, 27 Oct 2026): before, its capacity was dropped with a log
    # line (FES 2025: Bridgwater, Clydes Mill, Eday and Harris, 1.29 GW in 2050).
    if unmatched_summary:
        listed = ", ".join(f"{g} ({mw:,.1f} MW)" for g, mw in sorted(unmatched_summary.items(),
                                                                    key=lambda x: x[1], reverse=True))
        raise GspLocationError(
            f"FES renewable GSP rows not located: {listed}. Add the GSP to its edition's GSP info table "
            f"(data/network/ETYS/fes<edition>_regional_breakdown_gsp_info.csv), a declared position "
            f"({DECLARED_GSP_POINTS_FILE}) or an alias row (data/FES/gsp_aliases.csv)")

    return fes_df


def _load_fes_small_scale_solar(fes_path: str, modelled_year: int, fes_scenario: str,
                                network: pypsa.Network, logger: logging.Logger) -> pd.DataFrame:
    """FES small-scale solar (Gen_BB013, G98/G83) by GSP for the modelled year.

    One row per FES GSP with ``gsp``, ``capacity_mw`` and the GSP's ``bus``,
    ``lat`` and ``lon`` from the same GSP mapping the FES proxy rows use. GSPs
    the mapping cannot find are logged and left out (their MW stays with the
    regional proxy residual).
    """
    frame = pd.read_csv(fes_path, encoding='utf-8-sig', low_memory=False)
    if 'Building Block ID Number' not in frame.columns:
        raise ValueError(f"{fes_path} has no 'Building Block ID Number' column for Gen_BB013")
    rows = frame[frame['Building Block ID Number'].astype(str).str.strip().eq('Gen_BB013')]
    if fes_scenario is not None:
        rows = rows[rows['FES Pathway'].eq(fes_scenario)]
    year = str(int(modelled_year))
    if year not in rows.columns:
        raise ValueError(f"{fes_path} has no {year} column")
    rows = rows.assign(capacity_mw=pd.to_numeric(rows[year], errors='coerce').fillna(0.0))
    if 'GSP ID' not in rows.columns:
        rows = rows.assign(**{'GSP ID': None})
    rows = (rows[rows['capacity_mw'] > 0].groupby(['GSP', 'GSP ID'], as_index=False, dropna=False)['capacity_mw']
            .sum())
    gsp_to_bus = _load_gsp_bus_mapping(network, logger, fes_edition_of_file(fes_path))
    out, unmatched = [], []
    for row in rows.itertuples(index=False):
        info = _locate_gsp(row.GSP, row[1], gsp_to_bus)
        if not info or info.get('bus') is None:
            unmatched.append((row.GSP, row.capacity_mw))
            continue
        out.append({'gsp': row.GSP, 'capacity_mw': float(row.capacity_mw), 'bus': info.get('bus'),
                    'lat': info.get('lat'), 'lon': info.get('lon')})
    if unmatched:
        logger.warning("FES small-scale solar: %d GSPs (%.0f MW) not located; left to the regional proxy: %s",
                       len(unmatched), sum(mw for _, mw in unmatched), [g for g, _ in unmatched][:10])
    logger.info("FES small-scale solar (Gen_BB013, %s %s): %.0f MW at %d GSPs", fes_scenario, year,
                sum(r['capacity_mw'] for r in out), len(out))
    return pd.DataFrame(out)


#: Declared proxy relocations (bus, target_bus, carriers, reason, source): radial
#: collectors a named project already fills (rule add_renewables_to_network input).
DEFAULT_PROXY_RELOCATIONS_FILE = "data/network/ETYS/cp30_proxy_relocations.csv"


def relocate_proxy_rows(sites: pd.DataFrame, path, network: pypsa.Network,
                        logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move proxy rows off declared buses to a declared target (15 Oct 2026).

    ``path`` is a CSV of ``bus``, ``target_bus``, ``carriers`` (``;``-separated)
    and ``reason``: buses where proxy capacity of those carriers must not sit
    (a radial collector a named project already fills) and where it goes
    instead. Only rows passed in move (the caller passes the proxy increment);
    a row whose bus or target is absent from the network is left alone and
    logged. Returns (sites, moves).
    """
    table = pd.read_csv(path, dtype=str).fillna("")
    if missing := {"bus", "target_bus", "carriers", "reason"} - set(table.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = sites.copy()
    moves = []
    for row in table.itertuples(index=False):
        if row.bus not in network.buses.index or row.target_bus not in network.buses.index:
            logger.warning("Proxy relocation %s -> %s skipped: bus absent from this network", row.bus, row.target_bus)
            continue
        carriers = [c.strip() for c in str(row.carriers).split(";") if c.strip()]
        mask = result["bus"].astype(str).eq(row.bus) & result["technology"].isin(carriers)
        if not mask.any():
            continue
        mw = float(pd.to_numeric(result.loc[mask, "capacity_mw"], errors="coerce").sum())
        result.loc[mask, "bus"] = row.target_bus
        result.loc[mask, "connection_allocation_source"] = f"Proxy relocated from {row.bus}: {row.reason}"
        moves.append({"bus": row.bus, "target_bus": row.target_bus, "rows": int(mask.sum()), "capacity_mw": mw})
        logger.info("Proxy relocation: %.1f MW (%d rows) %s -> %s", mw, int(mask.sum()), row.bus, row.target_bus)
    return result, pd.DataFrame(moves)


def _build_brownfield_named_fleet(
    brownfield: dict, site_files: dict, cp30_es5_path: str, cp30_pathway: str,
    modelled_year: int, network: pypsa.Network, regional_allocation: dict, logger: logging.Logger,
    fes_path: Optional[str] = None, spatial_fes_scenario: Optional[str] = None,
):
    """Assemble, place, regionalise and reconcile the named REPD fleet.

    Returns (named fleet after reconciliation, residual ES5 targets for the
    proxy increment, brownfield bookkeeping dict for the audit).
    """
    operational_files = {c: site_files["operational"][c] for c in BROWNFIELD_CARRIERS
                         if c in site_files.get("operational", {})}
    pipeline_files = {c: site_files["pipeline"][c] for c in BROWNFIELD_CARRIERS
                      if c in site_files.get("pipeline", {})}
    named, retired = load_named_fleet(operational_files, pipeline_files, modelled_year, brownfield, logger)
    # Declared projects that cannot deliver this year (their capacity passes to
    # the proxy residual through the ES5 reconciliation below). The checked CfD
    # table adds the sites whose contracts are terminated or which hold none
    # (not built by 2030, user decision 14 Oct 2026): refilled the same way.
    checked_exclusions = None
    if brownfield.get("_checked_cfd_contracts"):
        from scripts.renewables.support_register_labels import (
            checked_cfd_exclusions,
            load_checked_cfd_contracts,
        )
        checked_exclusions = checked_cfd_exclusions(
            load_checked_cfd_contracts(brownfield["_checked_cfd_contracts"]))
    named, excluded = apply_named_fleet_exclusions(
        named, brownfield.get("named_exclusions_file", DEFAULT_NAMED_EXCLUSIONS_FILE),
        modelled_year, logger, extra=checked_exclusions,
    )
    if named.empty:
        logger.warning("Brownfield fleet enabled but no named projects were loaded")
        return named, None, {"retired": retired, "excluded": excluded,
                             "trimmed": pd.DataFrame(), "regional": pd.DataFrame()}

    # Refill at the excluded project's own landing (15 Oct 2026): the checked
    # CfD table's not-built sites are placed exactly as the named project would
    # have been (crosswalk/TEC landing for offshore, nearest bus to the REPD
    # coordinates otherwise) and kept as anonymous proxy rows there, instead of
    # their MW being spread pro rata over the region's FES proxy rows. Only the
    # checked-table exclusions: the declared exclusions file (the Lewis farms)
    # lists projects whose connection does not exist in the modelled year.
    refill_candidates = pd.DataFrame()
    if len(excluded) and checked_exclusions is not None and len(checked_exclusions):
        refill_ids = set(checked_exclusions["repd_ref_id"].astype(str).str.strip()
                         .str.replace(r"\.0$", "", regex=True))
        excluded_ids = excluded["repd_ref_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        pick = excluded_ids.isin(refill_ids) & excluded["technology"].isin(BROWNFIELD_CARRIERS)
        refill_candidates = excluded[pick].drop(
            columns=["exclusion_reason", "exclusion_evidence", "first_modelled_year"], errors="ignore"
        ).copy()
        refill_candidates["_refill_candidate"] = True
        named = pd.concat([named, refill_candidates], ignore_index=True, sort=False)
    if "_refill_candidate" not in named.columns:
        named["_refill_candidate"] = False
    named["_refill_candidate"] = named["_refill_candidate"].fillna(False).astype(bool)

    # Place named projects, then give each the same CP30 region the proxy rows
    # get, so the two are reconciled on one regional definition.
    n_buses = len(network.buses)
    named = map_sites_to_buses(
        network, named, method='nearest', lon_col='lon', lat_col='lat',
        max_distance_km=1000.0 if n_buses < 50 else 200.0,
    )
    unmapped = named["bus"].isna()
    refill_unmapped = named[unmapped & named["_refill_candidate"]]
    if len(refill_unmapped):
        # No bus to refill at: the ES5 residual keeps this MW for the regional proxy.
        logger.info("%d excluded not-built sites (%.0f MW) map to no bus; their MW stays with the "
                    "regional proxy", len(refill_unmapped), refill_unmapped["capacity_mw"].sum())
    if (unmapped & ~named["_refill_candidate"]).any():
        dropped = named[unmapped & ~named["_refill_candidate"]]
        logger.warning("Dropping %d named projects (%.0f MW) that map to no bus",
                       len(dropped), dropped["capacity_mw"].sum())
    named = named[~unmapped].copy()

    # Distribution-size onshore and solar at their own GSP's demand node, not the
    # nearest bus at any voltage (15 Oct 2026; scripts/utilities/gsp_nodes.py).
    # Licence-area thresholds: SHET 10 MW, SPT 30 MW, England and Wales 100 MW.
    # A documented connection point (spatial_utils.ONSHORE_RENEWABLE_DIRECT_BUS,
    # TEC register) wins: those farms keep their bus for the BMU step.
    regional = regional_allocation or {}
    locator = GspNodeLocator(network, regional.get("gsp_polygon_file") or GSP_POLYGON_FILE, log=logger)
    named, distributed_moves = place_distributed_at_gsp_nodes(
        named, locator, log=logger, keep=documented_connection_mask(named, network.buses.index),
    )
    named = apply_offshore_landing_crosswalk(
        named, brownfield.get("offshore_landing_crosswalk"), network,
        brownfield.get("offshore_landing_policy", "nearest_bus_fallback"), logger,
        fallback_min_voltage_kv=brownfield.get("offshore_fallback_min_voltage_kv", 275.0),
        fallback_min_voltage_capacity_mw=brownfield.get("offshore_fallback_min_voltage_capacity_mw", 300.0),
    )
    if not regional.get("enabled", False):
        raise ValueError("The brownfield fleet needs fleet.spatial_proxy.regional_allocation enabled: "
                         "reconciliation is defined per CP30 ES5 region")
    named = assign_cp30_regions(named, network, regional.get("region_anchor_file", ""), regional)
    # Split the placed refill candidates back out and make them anonymous
    # proxy rows at the bus and region they were just given.
    placed_refill = named[named["_refill_candidate"]]
    named = named[~named["_refill_candidate"]].drop(columns=["_refill_candidate"])
    refill, refill_detail = build_landing_refill(placed_refill, modelled_year, logger)
    if len(refill_candidates):
        # Candidates that mapped to no bus stay with the regional proxy; record them too.
        unplaced = refill_candidates[~refill_candidates["repd_ref_id"].astype(str).isin(
            placed_refill["repd_ref_id"].astype(str))]
        if len(unplaced):
            refill_detail = pd.concat([refill_detail, unplaced.reindex(
                columns=["site_name", "repd_ref_id", "technology", "capacity_mw"]).assign(
                refill="regional_proxy")], ignore_index=True)
    # FES small-scale (rooftop) solar held at its GSP node before the named
    # pipeline fills the ES5 envelope (15 Oct 2026). The named pipeline is
    # trimmed to make room, least mature first and uncontracted before any
    # project holding a CfD round; operational plant is never trimmed.
    reserved = pd.DataFrame()
    if brownfield.get("small_scale_solar_reservation", False):
        if not fes_path:
            raise ValueError("small_scale_solar_reservation needs the FES data file")
        small = _load_fes_small_scale_solar(fes_path, modelled_year, spatial_fes_scenario, network, logger)
        if len(small):
            small = locator.assign(small)
            small["bus"] = small["gsp_node"].where(small["gsp_node"].notna(), small["bus"])
            small = assign_cp30_regions(small.drop(columns=["gsp_node", "gsp_group"]), network,
                                        regional.get("region_anchor_file", ""), regional)
            reserved = build_small_scale_solar_reservation(small, modelled_year, logger)
    es5 = read_es5_targets(cp30_es5_path, cp30_pathway, modelled_year)
    named, residual_targets, trimmed, regional_audit = reconcile_with_es5(
        named, es5, brownfield.get("overshoot_policy", "trim_least_mature"), logger,
        refill=refill if len(refill) else None,
        reserved=reserved if len(reserved) else None,
        contracted_last=bool(len(reserved)),
    )
    # Regions with no ES5 row (hydro, marine) keep the national proxy scaling;
    # carry ES5 rows for other carriers unchanged so the increment step sees a
    # complete target table.
    others = es5[~es5["carrier"].isin(BROWNFIELD_CARRIERS)][["carrier", "cp30_region", "capacity_mw"]]
    residual_targets = pd.concat([residual_targets, others], ignore_index=True)
    is_named = named.get("fleet_origin", pd.Series("named", index=named.index)).fillna("named").eq("named")
    named["data_source"] = named.get("data_source", pd.Series(index=named.index)).fillna("REPD_named_fleet")
    named.loc[is_named, "spatial_proxy_source"] = "REPD project coordinates; CP30 ES5 regional envelope"
    return named, residual_targets, {"retired": retired, "excluded": excluded,
                                     "trimmed": trimmed, "regional": regional_audit,
                                     "refill_detail": refill_detail,
                                     "distributed_moves": distributed_moves}


def load_cp30_renewable_generators(
    cp30_es1_path: str,
    cp30_es5_path: str,
    fes_path: str,
    modelled_year: int,
    cp30_pathway: str,
    spatial_fes_scenario: str,
    network: pypsa.Network,
    logger: logging.Logger,
    regional_allocation: Optional[dict] = None,
    connection_allocation: Optional[dict] = None,
    hnd_project_allocation: Optional[dict] = None,
    brownfield: Optional[dict] = None,
    site_files: Optional[dict] = None,
    brownfield_state: Optional[dict] = None,
) -> pd.DataFrame:
    """Load CP30 renewable capacity and distribute it using explicit FES spatial proxy rows.

    With ``brownfield.enabled`` the named REPD fleet is fixed first and the FES
    proxy fills only the residual per CP30 region; ``brownfield_state`` (a dict
    the caller supplies) receives the retired/trimmed/regional audit tables.
    """
    logger.info(f"Loading CP30 renewable capacity for {cp30_pathway} {modelled_year}")
    targets = load_cp30_generation_capacities(
        cp30_es1_path,
        cp30_pathway,
        modelled_year,
        groups=["renewable"],
    )

    if targets.empty:
        logger.warning("No CP30 renewable capacity targets found")
        return pd.DataFrame()

    proxy = load_fes_renewable_generators(
        fes_path,
        modelled_year,
        spatial_fes_scenario,
        network,
        logger,
    )
    if proxy.empty:
        logger.warning("No FES renewable spatial proxy rows found for CP30")
        return pd.DataFrame()

    scaled = scale_proxy_to_cp30_targets(
        proxy,
        targets,
        proxy_carrier_col="technology",
        output_carrier_col="technology",
        capacity_col="capacity_mw",
        name_prefix="CP30",
    )

    if scaled.empty:
        logger.warning("CP30 renewable targets could not be matched to FES spatial proxy carriers")
        return pd.DataFrame()

    named = None
    residual_targets = None
    brownfield = brownfield or {}
    if brownfield.get("enabled", False):
        logger.info("Brownfield fleet: fixing named REPD projects before the proxy increment")
        named, residual_targets, state = _build_brownfield_named_fleet(
            brownfield, site_files or {}, cp30_es5_path, cp30_pathway, modelled_year,
            network, regional_allocation, logger,
            fes_path=fes_path, spatial_fes_scenario=spatial_fes_scenario,
        )
        if brownfield_state is not None:
            brownfield_state.update(state)

    # Offshore proxies take the CP30 region of the bus they land at, not of
    # their FES source bus: an offshore platform's nearest GSP polygon can be
    # another region (Hornsea West HOWW2A, 112 km offshore, is nearest the UKPN
    # Eastern polygon but lands at Killingholme, North England; 15 Oct 2026).
    region_buses = landing_buses_for_regions(scaled, connection_allocation)
    scaled = apply_cp30_es5_regional_allocation(
        scaled,
        cp30_es5_path,
        cp30_pathway,
        modelled_year,
        network,
        regional_allocation,
        logger,
        targets_override=residual_targets,
        region_buses=region_buses,
    )
    scaled = apply_hnd_project_allocation(
        scaled, network, hnd_project_allocation, logger
    )
    scaled = apply_connection_allocation(
        scaled,
        targets,
        network,
        connection_allocation,
        modelled_year,
        logger,
    )
    scaled["fleet_origin"] = "increment"
    if brownfield.get("enabled", False):
        # Declared proxy relocations: radial collectors a named project fills.
        scaled, moves = relocate_proxy_rows(
            scaled, brownfield.get("_proxy_relocations") or DEFAULT_PROXY_RELOCATIONS_FILE, network, logger)
        if brownfield_state is not None:
            brownfield_state["proxy_relocations"] = moves
    if named is not None and not named.empty:
        # The proxy rows are now the increment; rename so their origin is legible.
        increment = scaled["technology"].isin(BROWNFIELD_CARRIERS) & scaled["site_name"].astype(str).str.startswith("CP30_")
        scaled.loc[increment, "site_name"] = scaled.loc[increment, "site_name"].str.replace(
            "^CP30_", "CP30_increment_", regex=True
        )
        scaled = pd.concat([named, scaled], ignore_index=True, sort=False)
    target_summary = targets.groupby("carrier")["capacity_mw"].sum()
    scaled_summary = scaled.groupby("technology")["capacity_mw"].sum()
    logger.info("CP30 renewable capacity targets applied:")
    for carrier, target_mw in target_summary.sort_values(ascending=False).items():
        actual_mw = float(scaled_summary.get(carrier, 0.0))
        origin = ""
        if named is not None and not named.empty and carrier in BROWNFIELD_CARRIERS:
            named_mw = float(scaled.loc[scaled["technology"].eq(carrier) & scaled["fleet_origin"].eq("named"), "capacity_mw"].sum())
            origin = f", named={named_mw:,.1f} MW ({named_mw / actual_mw:.0%})" if actual_mw else ""
        logger.info(f"  {carrier}: target={target_mw:,.1f} MW, built={actual_mw:,.1f} MW{origin}")

    return scaled


#: FES GSPs that no GSP info table places (no GSP id, no published point), at declared positions
#: (user decision, 27 Oct 2026: the island GSPs Eday (Orkney) and Harris are placed, not dropped).
DECLARED_GSP_POINTS_FILE = "data/network/GSP/gsp_declared_points.csv"


def _load_gsp_bus_mapping(network: pypsa.Network, logger: logging.Logger, fes_year: int) -> dict:
    """Map an FES edition's GSPs to network buses, from that edition's committed GSP info table.

    The table is ``data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv``
    (``FES_data.load_gsp_info``): no NESO API call at build time and no other edition's table
    (user decision, 27 Oct 2026; storage has worked this way since 16 Oct). A missing table raises.

    Args:
        network: PyPSA network for bus coordinate matching
        logger: Logger instance
        fes_year: The FES edition of the building blocks being placed

    Returns:
        ``{gsp_name: {'bus', 'lat', 'lon', 'gsp_id'}}``; :func:`_locate_gsp` looks rows up by id first.
    """
    from scripts.FES.FES_data import load_gsp_info

    gsp_df = load_gsp_info(int(fes_year))
    gsp_mapping = {}
    columns = gsp_df[['GSP ID', 'Name', 'Latitude', 'Longitude']]
    for gsp_id, gsp_name, lat, lon in columns.itertuples(index=False, name=None):
        if pd.isna(gsp_name) or pd.isna(lat) or pd.isna(lon) or len(network.buses) == 0:
            continue
        nearest_bus = _find_nearest_bus(network, lat, lon)
        if nearest_bus is not None:
            gsp_mapping[gsp_name] = {'bus': nearest_bus, 'lat': lat, 'lon': lon,
                                     'gsp_id': '' if pd.isna(gsp_id) else str(gsp_id).strip()}
    declared = pd.read_csv(DECLARED_GSP_POINTS_FILE, encoding='utf-8-sig')
    for gsp_name, lat, lon in declared[['Name', 'Latitude', 'Longitude']].itertuples(index=False, name=None):
        if gsp_name in gsp_mapping or len(network.buses) == 0:
            continue
        nearest_bus = _find_nearest_bus(network, lat, lon)
        if nearest_bus is not None:
            gsp_mapping[gsp_name] = {'bus': nearest_bus, 'lat': lat, 'lon': lon, 'gsp_id': ''}
    logger.info(f"Loaded GSP-to-bus mapping: {len(gsp_mapping)} GSPs from FES {fes_year}'s GSP info "
                f"and {len(declared)} declared positions")
    return gsp_mapping


def _locate_gsp(gsp_name, gsp_id, gsp_to_bus: dict) -> Optional[dict]:
    """A building-block row's GSP location: by its GSP id where the table has it, else by name.

    The id is exact where names drift between editions and tables ("Bridgwater" and "Bridgewater",
    "Clydes Mill" and "Clyde's Mill", "Tynemouth 132kV32kV"); storage rows match the same way.
    """
    if gsp_id is not None and not (isinstance(gsp_id, float) and pd.isna(gsp_id)) and str(gsp_id).strip():
        by_id = {v.get('gsp_id'): v for v in gsp_to_bus.values() if v.get('gsp_id')}
        if str(gsp_id).strip() in by_id:
            return by_id[str(gsp_id).strip()]
    return _fuzzy_gsp_lookup(gsp_name, gsp_to_bus)


def _embedded_site_bus_share(network: pypsa.Network, sites: pd.DataFrame, load_buses) -> Optional[pd.Series]:
    """Capacity share by load bus of the small sites ESPENI's embedded wind stands for (nearest load bus).

    Returns None when no site has usable coordinates and capacity, so the caller keeps its fallback.
    """
    lat = pd.to_numeric(sites.get('lat'), errors='coerce')
    lon = pd.to_numeric(sites.get('lon'), errors='coerce')
    cap = pd.to_numeric(sites.get('capacity_mw'), errors='coerce')
    ok = lat.notna() & lon.notna() & cap.gt(0)
    if not ok.any():
        return None
    buses = network.buses.loc[list(load_buses), ['x', 'y']].astype(float)
    if (buses['x'].max() - buses['x'].min()) > 1000:   # OSGB36 metres: project the sites
        from pyproj import Transformer
        site_x, site_y = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(
            lon[ok].to_numpy(), lat[ok].to_numpy())
    else:
        site_x, site_y = lon[ok].to_numpy(), lat[ok].to_numpy()
    d2 = (site_x[:, None] - buses['x'].to_numpy()[None, :]) ** 2 + (site_y[:, None] - buses['y'].to_numpy()[None, :]) ** 2
    nearest = buses.index[np.argmin(d2, axis=1)]
    share = pd.Series(cap[ok].to_numpy(), index=nearest).groupby(level=0).sum()
    return share / share.sum()


def main():
    """Main execution function for renewable generator integration."""
    global logger
    start_time = time.time()
    stage_times = {}  # Track timing for each major stage
    # The small onshore wind sites left out of wind_onshore for ESPENI's embedded wind to represent: where that
    # embedded wind is placed (stage 6.75).
    embedded_wind_sites = None

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
        # Standard historical runs use a start-of-window capacity snapshot. A
        # full-year validation can include commissioning within the window and
        # gate availability to each REPD operational date instead.
        solve_period = scenario_config.get('solve_period', {})
        sim_start_date = solve_period.get('start', None)
        sim_end_date = solve_period.get('end', None)
        commissioning_config = scenario_config.get('fleet', {}).get('commissioning', {})
        commissioning_profile_gating = bool(
            commissioning_config.get('profile_gate_within_solve_period', False)
            and sim_start_date and sim_end_date
        )
        commissioning_date_overrides = {}
        overrides_file = commissioning_config.get('date_overrides_file')
        if commissioning_profile_gating and overrides_file:
            overrides_df = pd.read_csv(overrides_file)
            required = {'source_site_name', 'model_activation_date'}
            missing = required - set(overrides_df.columns)
            if missing:
                raise ValueError(f"Commissioning date override file is missing columns: {sorted(missing)}")
            commissioning_date_overrides = (
                overrides_df.dropna(subset=['source_site_name', 'model_activation_date'])
                .drop_duplicates('source_site_name')
                .set_index('source_site_name')['model_activation_date']
                .to_dict()
            )
            logger.info("Loaded %d LCCC/REPD commissioning date overrides", len(commissioning_date_overrides))
        
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
        
        # Bookkeeping the brownfield CP30 fleet fills in (retired, trimmed,
        # regional reconciliation) for the audit written at the end.
        brownfield_state = {}
        # Register-based support labelling audit (stage 5.75), added to the
        # brownfield summary.
        support_register_state = {}

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
                cutoff_desc = (
                    sim_end_date if commissioning_profile_gating else
                    (sim_start_date if sim_start_date else f"{modelled_year}-12-31")
                )
                logger.info(f"Historical scenario - filtering renewable sites to operational by {cutoff_desc}")
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
                            sites_df = _apply_commissioning_date_overrides(
                                sites_df, commissioning_date_overrides
                            )
                            sites_df = filter_sites_by_year(
                                sites_df,
                                modelled_year,
                                logger,
                                start_date=sim_start_date,
                                end_date=sim_end_date if commissioning_profile_gating else None,
                            )

                        # ESPENI separates transmission-metered wind (ELEX_WIND)
                        # from embedded wind (NGEM_EMBEDDED_WIND). REPD includes
                        # small distribution-connected onshore sites, so validation
                        # scenarios can exclude that subset from wind_onshore and
                        # let the ESPENI-derived embedded_wind generator represent it.
                        emb_cfg = scenario_config.get('renewables', {}).get('embedded', {})
                        overlap_cfg = emb_cfg.get('wind_overlap_exclusion', {})
                        if (
                            is_historical
                            and technology == 'wind_onshore'
                            and emb_cfg.get('enabled', False)
                            and emb_cfg.get('wind', False)
                            and overlap_cfg.get('enabled', False)
                        ):
                            max_capacity_mw = float(overlap_cfg.get('max_capacity_mw', 0.0) or 0.0)
                            if max_capacity_mw > 0 and 'capacity_mw' in sites_df.columns:
                                capacity = pd.to_numeric(sites_df['capacity_mw'], errors='coerce')
                                overlap_mask = capacity <= max_capacity_mw
                                preserve_support = overlap_cfg.get('preserve_explicit_support', True)
                                if preserve_support:
                                    explicit_support = _explicit_support_mask(sites_df)
                                    retained = overlap_mask & explicit_support
                                    if retained.any():
                                        logger.info(
                                            "Retaining %d explicitly supported small onshore wind sites "
                                            "(%.1f MW) as individual generators for CfD/ROC accounting",
                                            int(retained.sum()),
                                            float(capacity[retained].sum()),
                                        )
                                    overlap_mask &= ~explicit_support
                                excluded_count = int(overlap_mask.sum())
                                excluded_capacity = float(capacity[overlap_mask].sum())
                                if excluded_count:
                                    logger.info(
                                        "Excluding %d small onshore wind sites (%.1f MW <= %.1f MW) "
                                        "from wind_onshore; ESPENI embedded_wind will represent this output",
                                        excluded_count,
                                        excluded_capacity,
                                        max_capacity_mw,
                                    )
                                    embedded_wind_sites = sites_df.loc[overlap_mask].copy()
                                    sites_df = sites_df.loc[~overlap_mask].copy()

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
            # FUTURE PATH: brownfield (named fleet in the CP30 regional envelope)
            # or FES GSP building blocks
            # ================================================================
            if uses_brownfield(scenario_config):
                fleet_config = scenario_config.get('fleet', {}) or {}
                spatial_proxy = fleet_config.get('spatial_proxy', {})
                # Scenario keys override the documented defaults block.
                brownfield_config = {
                    **(fleet_config.get('brownfield') or {}),
                    **(spatial_proxy.get('brownfield') or {}),
                }
                brownfield_site_files = {
                    'operational': {
                        'wind_onshore': snk.input.wind_onshore_sites,
                        'wind_offshore': snk.input.wind_offshore_sites,
                        'solar_pv': snk.input.solar_pv_sites,
                    },
                    'pipeline': {
                        'wind_onshore': snk.input.wind_onshore_pipeline_sites,
                        'wind_offshore': snk.input.wind_offshore_pipeline_sites,
                        'solar_pv': snk.input.solar_pv_pipeline_sites,
                    },
                }
                # The checked CfD table also excludes sites not built by this year.
                checked_cfd_input = getattr(snk.input, 'checked_cfd_contracts', None)
                if checked_cfd_input:
                    brownfield_config['_checked_cfd_contracts'] = str(checked_cfd_input)
                relocations_input = getattr(snk.input, 'cp30_proxy_relocations', None)
                if relocations_input:
                    brownfield_config['_proxy_relocations'] = str(relocations_input)
                renewable_sites = load_cp30_renewable_generators(
                    snk.input.cp30_es1,
                    snk.input.cp30_es5,
                    snk.input.fes_data,
                    modelled_year,
                    scenario_config.get('CP30_pathway'),
                    spatial_proxy.get('FES_scenario'),
                    network,
                    logger,
                    spatial_proxy.get('regional_allocation'),
                    spatial_proxy.get('offshore_connection_allocation'),
                    spatial_proxy.get('hnd_project_allocation'),
                    brownfield=brownfield_config,
                    site_files=brownfield_site_files,
                    brownfield_state=brownfield_state,
                )
                stage_times['3. Load renewable capacity data (CP30)'] = time.time() - stage_start
                empty_message = "No CP30 renewable capacity found for this scenario"
            else:
                renewable_sites = load_fes_renewable_generators(
                    snk.input.fes_data,
                    modelled_year,
                    scenario_config.get('FES_scenario', None),
                    network,
                    logger,
                    gsp_placement="demand_node",
                )
                # Task Y6b (FES 2050 plan): the FES Direct offshore rows are replaced by
                # the Y5 landing table (bus, MW). Absent block = today's behaviour.
                landing_cfg = fes_supply_part(scenario_config, 'offshore_landing')
                if landing_cfg and len(renewable_sites):
                    from scripts.generators.fes_supply import apply_fes_offshore_landing
                    offshore_profile = next((p for p in snk.input.profiles if 'wind_offshore_' in str(p)), None)
                    profile_columns = (pd.read_csv(offshore_profile, nrows=0, index_col=0).columns
                                       if offshore_profile else [])
                    profile_sites = pd.concat(
                        [pd.read_csv(path, usecols=lambda c: c in ('site_name', 'lat', 'lon', 'capacity_mw'))
                         for path in (snk.input.wind_offshore_sites, snk.input.wind_offshore_pipeline_sites,
                                      snk.input.offshore_pipeline_processed) if Path(str(path)).exists()],
                        ignore_index=True)
                    renewable_sites = apply_fes_offshore_landing(
                        renewable_sites,
                        getattr(snk.input, 'fes_offshore_landing', None) or landing_cfg['file'],
                        network.buses,
                        projects_path=(getattr(snk.input, 'fes_offshore_landing_projects', None)
                                       or landing_cfg.get('projects_file')),
                        profile_columns=profile_columns, profile_sites=profile_sites,
                        modelled_year=int(modelled_year), log=logger,
                    )
                stage_times['3. Load renewable capacity data (FES)'] = time.time() - stage_start
                empty_message = "No FES renewable capacity found for this scenario"
            
            if len(renewable_sites) == 0:
                logger.warning(empty_message)
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
        
        # Named brownfield rows were placed (and offshore ones landed by
        # evidence) before reconciliation; only rows without a bus are mapped.
        if 'bus' in renewable_sites.columns and renewable_sites['bus'].notna().any():
            placed = renewable_sites['bus'].notna()
            to_map = renewable_sites.loc[~placed]
            if len(to_map):
                to_map = map_sites_to_buses(
                    network, to_map, method='nearest', lon_col='lon', lat_col='lat',
                    max_distance_km=max_distance_km,
                )
            renewable_sites = pd.concat([renewable_sites.loc[placed], to_map], ignore_index=False, sort=False)
        else:
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

            # Buses no onshore site may sit on: DC-only hubs, offshore
            # platforms, OFTO export substations and buses with misplaced
            # coordinates (G5, audit 27 Sep 2026). Onshore wind, solar and hydro
            # avoid all four; marine (MeyGen is genuinely offshore) avoids only
            # the DC-only hubs and misplaced buses.
            placement_exclusions = pd.Series(dtype=object)
            excl_cfg = ((scenario_config.get('network', {}) or {}).get('spatial_mapping', {}) or {}).get('placement_exclusions', {}) or {}
            if excl_cfg.get('enabled', True):
                placement_exclusions = generation_placement_exclusions(
                    network,
                    land_boundary_file=(getattr(snk.input, 'placement_land_boundary', None)
                                        or excl_cfg.get('land_boundary_file')),
                    land_buffer_m=float(excl_cfg.get('land_buffer_m', 2000.0)),
                    coordinate_outlier_km=float(excl_cfg.get('coordinate_outlier_km', 30.0)),
                    cluster_km=float(excl_cfg.get('cluster_km', 5.0)),
                )
                logger.info("Placement exclusions: %s", placement_exclusions.value_counts().to_dict())
                # Rows on their own GSP's demand node stay (16 Oct 2026): the exclusions describe
                # where nearest-bus placement by coordinates goes wrong, and the node carries the
                # GSP's demand whatever its coordinates say (LINM3-, Linnmill 33 kV, is a
                # coordinate outlier; its distributed farms were scattered over ten buses).
                renewable_sites, _ = reassign_excluded_placements(
                    renewable_sites, network, placement_exclusions,
                    carriers=['wind_onshore', 'solar_pv', 'small_hydro', 'large_hydro'], log=logger,
                    protected_sources=tuple(_PROTECTED_PLACEMENTS) + tuple(GSP_NODE_SOURCES),
                )
                renewable_sites, _ = reassign_excluded_placements(
                    renewable_sites, network,
                    placement_exclusions[placement_exclusions.isin(['dc_only', 'coordinate_outlier'])],
                    carriers=['marine', 'tidal_stream', 'shoreline_wave', 'tidal_lagoon'], log=logger,
                )
                # Orkney sites onto the Orkney AC network (its substations have
                # no coordinates of their own), mainland sites off it.
                renewable_sites, _ = apply_island_networks(
                    renewable_sites, network,
                    carriers=['wind_onshore', 'solar_pv', 'small_hydro', 'large_hydro', 'marine',
                              'tidal_stream', 'shoreline_wave', 'tidal_lagoon'],
                    exclusions=placement_exclusions, log=logger,
                    land_boundary_file=(getattr(snk.input, 'placement_land_boundary', None)
                                        or excl_cfg.get('land_boundary_file')),
                )

            # Nearest-bus placement can leave hundreds of MW on a 33 kV terminal
            # busbar behind one small transformer; move sites (proxy tranches
            # first) to an adequate bus of the same substation or nearby.
            relief_cfg = ((scenario_config.get('network', {}) or {}).get('spatial_mapping', {}) or {}).get('weak_bus_relief', {}) or {}
            if relief_cfg.get('enabled', True):
                renewable_sites, _ = relieve_weak_bus_placements(
                    renewable_sites, network,
                    max_loading=float(relief_cfg.get('max_loading', 1.5)),
                    search_radius_km=float(relief_cfg.get('search_radius_km', 40.0)),
                    min_target_voltage_kv=float(relief_cfg.get('min_target_voltage_kv', 132.0)),
                    log=logger,
                    exclude_buses=placement_exclusions.index,
                    # Plant on its GSP's node moves only to a busbar of the same substation.
                    site_bound_sources=GSP_NODE_SOURCES,
                )

            # Scottish 132 kV is constrained in the BM (SHE/SPT transmission),
            # so a proxy behind a weak 132 kV circuit creates a constraint no
            # real project has. Move proxy increments out of every pocket whose
            # generation (less minimum load) cannot reach the >= 275 kV grid
            # over BM-rated branches; named farms stay (27 Sep 2026).
            export_cfg = ((scenario_config.get('network', {}) or {}).get('spatial_mapping', {}) or {}).get('export_relief', {}) or {}
            if export_cfg.get('enabled', True):
                renewable_sites, export_moves, export_pockets = relieve_export_constrained_proxies(
                    renewable_sites, network, (scenario_config.get('network', {}) or {}).get('transmission', {}) or {},
                    carriers=tuple(export_cfg.get('carriers', ['wind_onshore', 'solar_pv'])),
                    exclude_buses=placement_exclusions.index,
                    search_radius_km=float(export_cfg.get('search_radius_km', 60.0)),
                    min_target_voltage_kv=float(export_cfg.get('min_target_voltage_kv', 132.0)),
                    sink_kv=float(export_cfg.get('sink_kv', 275.0)),
                    max_local_loading=float(relief_cfg.get('max_loading', 1.5)),
                    log=logger,
                )

            if fes_supply_part(scenario_config, 'offshore_landing'):
                # Task Y6b: the Y5 landing buses are evidence; put them back after the
                # generic BMU, weak-bus and export-relief heuristics.
                from scripts.generators.fes_supply import restore_declared_buses
                renewable_sites = restore_declared_buses(renewable_sites, logger)

            stage_times['5.5. ETYS BMU mapping'] = time.time() - stage_start_bmu

        # STAGE 5.75: Assign subsidy tracking attributes (support_type + region)
        subsidy_config = scenario_config.get('support', {}).get('labels', {})
        if subsidy_config.get('enabled', False):
            stage_start_sub = time.time()
            logger.info("-" * 80)
            logger.info("ASSIGNING SUBSIDY TRACKING ATTRIBUTES")
            logger.info("-" * 80)
            
            roc_years = subsidy_config.get('roc_eligibility_years', 20)
            fes_fractions = subsidy_config.get('fes_cfd_fractions', {})
            roc_closure = subsidy_config.get('roc_closure_date', '2017-04-01')
            default_bandings = subsidy_config.get('default_roc_bandings', {})

            # Scheme registers as the authority for the named fleet (future
            # builds): LCCC contract evidence before the assignment, the Ofgem
            # RO register after it. See scripts/renewables/support_register_labels.py.
            registers_config = subsidy_config.get('support_registers', {}) or {}
            use_registers = bool(registers_config.get('enabled', False)) and not is_historical
            if use_registers:
                from scripts.renewables.support_register_labels import (
                    apply_lccc_cfd_evidence,
                    apply_ro_register_labels,
                    load_crosswalk_overrides,
                    load_lccc_portfolio,
                    load_repd_location,
                    load_ro_register,
                    support_register_summary,
                )
                repd_location = load_repd_location(snk.input.support_repd)
                register_overrides = load_crosswalk_overrides(
                    getattr(snk.input, 'support_crosswalk_overrides', None) or None)
                renewable_sites, _, cfd_register_audit = apply_lccc_cfd_evidence(
                    renewable_sites, load_lccc_portfolio(snk.input.lccc_cfd_portfolio),
                    repd_location, register_overrides, logger,
                )

            renewable_sites = _assign_subsidy_attributes_by_origin(
                renewable_sites,
                modelled_year or 2024,
                roc_years,
                fes_fractions,
                logger,
                roc_closure_date=roc_closure,
                default_roc_bandings=default_bandings,
            )
            checked_cfd_audit = pd.DataFrame()
            terminated_cfd_capacity = None
            checked_cfd_file = getattr(snk.input, 'checked_cfd_contracts', None) if use_registers else None
            if checked_cfd_file:
                # F19 and its follow-up (rebuild batch, 13 Oct 2026): named
                # sites take the CfD contracts checked against LCCC. A part-
                # contracted site (Seagreen 454 of 1,075 MW) splits into its
                # rounds and a merchant remainder, a multi-round site (Moray
                # West AR4 + AR6) into one part per round. A site whose
                # contracts are terminated (Hornsea 4) or that holds none
                # (Dudgeon Extension) has left the 2030 named fleet already (its
                # MW refilled by the proxy); still present in a later year, it
                # has no CfD. Then the RO register labels what is not CfD.
                from scripts.renewables.support_register_labels import (
                    apply_checked_cfd_contracts,
                    load_checked_cfd_contracts,
                    terminated_round_capacity,
                )
                checked_cfd_table = load_checked_cfd_contracts(checked_cfd_file)
                renewable_sites, checked_cfd_audit = apply_checked_cfd_contracts(
                    renewable_sites, checked_cfd_table, logger,
                )
                terminated_cfd_capacity = terminated_round_capacity(checked_cfd_table)
            if use_registers:
                # 1 April of the modelled year: the obligation year the bill's
                # RO line counts, so labels and bill hold the same stations.
                ro_reference = registers_config.get('ro_reference_date') or f"{modelled_year}-04-01"
                ro_register = load_ro_register(
                    snk.input.ro_register, ro_reference,
                    certificates_path=snk.input.ro_certificates,
                    default_bandings=default_bandings,
                )
                renewable_sites, ro_mapping, ro_register_audit = apply_ro_register_labels(
                    renewable_sites, ro_register, repd_location, register_overrides, logger,
                )
                # Carriers CP30 holds only as anonymous proxies (marine) take
                # the register's accredited share of their capacity.
                proxy_share = registers_config.get('renewable_proxy_share', {}) or {}
                proxy_share_audit = pd.DataFrame()
                if proxy_share.get('enabled', False):
                    from scripts.renewables.support_register_labels import (
                        apply_ro_register_share_to_proxies,
                    )
                    origin = renewable_sites.get(
                        'fleet_origin', pd.Series('increment', index=renewable_sites.index))
                    renewable_sites, proxy_share_audit = apply_ro_register_share_to_proxies(
                        renewable_sites, ro_register,
                        carriers=tuple(proxy_share.get('carriers', ['marine'])),
                        min_certificate_year=proxy_share.get('min_certificate_year'),
                        carrier_col='technology',
                        proxy_mask=origin.ne('named'),
                        register_carrier='marine',
                        classify=lambda name: 'marine',
                        # Item C4: the rest is tidal stream on AR4-AR7a CfDs
                        # (rounds from the CfD vintage schedule below).
                        remainder_support=proxy_share.get('remainder_support', 'CfD'),
                        logger=logger,
                    )
                support_register_state = support_register_summary(
                    ro_register, ro_mapping, ro_register_audit, cfd_register_audit,
                    renewable_sites, ro_reference,
                )
                support_register_state['proxy_share'] = proxy_share_audit.to_dict('records')
                support_register_state['checked_cfd_contracts'] = checked_cfd_audit.to_dict('records')
            renewable_sites = _apply_renewable_support_overrides(
                renewable_sites,
                subsidy_config.get('renewable_support_overrides_file'),
                logger,
            )

            # Split CfD capacity into allocation-round tranches. Without this
            # every contract falls back to one carrier-wide ``future_*`` strike,
            # so a fleet spanning AR2 at 74.75 to AR7 at 65.45 in 2012 prices
            # settles at a single price.
            # AR7 held only offshore pots; the onshore/solar/tidal contracts of
            # that auction year are AR7a, which REPD and the LCCC portfolio
            # both label "AR7" (2,157 MW solar and 271 MW onshore were on a
            # round key the strike table does not have; audit 27 Sep 2026).
            from scripts.generators.cfd_vintage_labels import normalise_cfd_rounds
            renewable_sites = normalise_cfd_rounds(renewable_sites, carrier_column='technology')

            vintage_schedule = subsidy_config.get('cfd_vintage_schedule_file')
            if vintage_schedule:
                from scripts.generators.cfd_vintage_labels import (
                    apply_cfd_vintage_labels,
                    named_round_capacity,
                    summarise_vintage_labels,
                )
                # The schedule describes the proxy increment, whose tranches
                # stand for auctioned capacity in aggregate. A named pipeline
                # project REPD records no round for is one real project, not a
                # slice of every past auction: it is placed on the modelled AR8
                # round rather than being split into AR2..AR7 tranches.
                if 'fleet_origin' in renewable_sites.columns:
                    is_named = renewable_sites['fleet_origin'].eq('named')
                    named_rows = renewable_sites.loc[is_named].copy()
                    unlabelled_cfd = (
                        named_rows['support_type'].eq('CfD')
                        & named_rows.get('cfd_round', pd.Series(index=named_rows.index))
                        .fillna('').astype(str).str.strip().eq('')
                    )
                    if unlabelled_cfd.any():
                        from scripts.market.cfd_strike_table import MODELLED_ROUND
                        named_rows.loc[unlabelled_cfd, 'cfd_round'] = MODELLED_ROUND
                        logger.info(
                            "  %d named CfD projects (%.0f MW) have no REPD round and settle at the "
                            "modelled AR8 strike",
                            int(unlabelled_cfd.sum()),
                            named_rows.loc[unlabelled_cfd, 'capacity_mw'].sum(),
                        )
                    # Named projects already hold their rounds' capacity: net
                    # it off the schedule so the increment fills only what is
                    # left of each round (AR2/AR3/AR6 offshore were counted
                    # twice before, AR4/AR7 short; audit 27 Sep 2026).
                    # Awarded MW of a named site's terminated contract stays netted
                    # off its round: the increment must not refill a terminated
                    # award at that round's strike (checked CfD contracts, 13 Oct 2026).
                    named_capacity = named_round_capacity(named_rows, carrier_column='technology')
                    if terminated_cfd_capacity is not None and len(terminated_cfd_capacity):
                        named_capacity = pd.concat([named_capacity, terminated_cfd_capacity], ignore_index=True)
                    increment_rows = apply_cfd_vintage_labels(
                        renewable_sites.loc[~is_named], vintage_schedule, logger,
                        named_capacity=named_capacity,
                    )
                    renewable_sites = pd.concat([named_rows, increment_rows], ignore_index=True, sort=False)
                else:
                    renewable_sites = apply_cfd_vintage_labels(
                        renewable_sites, vintage_schedule, logger
                    )
                for row in summarise_vintage_labels(renewable_sites).itertuples(index=False):
                    logger.info(
                        "    %s %s: %.0f MW across %d rows",
                        row.carrier, row.cfd_round, row.capacity_mw, row.generators,
                    )

            stage_times['5.75. Subsidy attributes'] = time.time() - stage_start_sub
        
        # STAGE 6: Add renewable generators to network
        stage_start = time.time()
        logger.info("-" * 80)
        logger.info("INTEGRATING RENEWABLE GENERATORS INTO NETWORK")
        logger.info("-" * 80)
        
        profiles_dir = "resources/renewable/profiles"
        logger.info(f"Using profiles directory: {profiles_dir}")
        
        initial_gen_count = len(network.generators)
        
        renewables_year = scenario_config.get('renewables_year', modelled_year)

        # Load ERA5 performance factors (bias correction) directly from config.
        # Factors represent uncurtailed available generation — do NOT use ESPENI
        # metered output as target because ESPENI includes real-world curtailment
        # which the copperplate wholesale stage intentionally ignores.
        #
        # The correct calibration target is:
        #   model wholesale wind ≈ ESPENI metered + estimated real curtailment
        #
        # Two config formats are supported:
        #   Option A (stacked): era5_bias, turbine_curve, wake_losses, availability
        #                       dicts per carrier — effective factor = product
        #   Option B (flat):    factors dict per carrier — used directly
        # If 'factors' is provided and non-empty, Option B wins (backward compat).
        performance_factors = None
        perf_cfg = scenario_config.get('renewables', {}).get('performance_factors', {})
        if perf_cfg.get('enabled', False):
            cfg_factors = perf_cfg.get('factors', {})
            if cfg_factors:
                # Option B: flat override factors — use directly
                performance_factors = {c: float(v) for c, v in cfg_factors.items()}
                logger.info(f"Renewable performance factors (flat override): {performance_factors}")
            else:
                # Option A: stacked component factors — compute product per carrier
                component_names = ['era5_bias', 'turbine_curve', 'wake_losses', 'availability']
                components = {}
                for name in component_names:
                    d = perf_cfg.get(name, {})
                    if d:
                        components[name] = {c: float(v) for c, v in d.items()}

                if components:
                    # Collect all carriers mentioned across any component
                    all_carriers = set()
                    for d in components.values():
                        all_carriers.update(d.keys())

                    performance_factors = {}
                    for carrier in sorted(all_carriers):
                        product = 1.0
                        parts = []
                        for name in component_names:
                            val = components.get(name, {}).get(carrier, 1.0)
                            product *= val
                            if val != 1.0:
                                parts.append(f"{name}={val:.3f}")
                        performance_factors[carrier] = product
                        logger.info(
                            f"  {carrier}: effective={product:.4f} "
                            f"({' × '.join(parts) if parts else 'all 1.0'})"
                        )
                    logger.info(
                        f"Renewable performance factors (stacked carrier config): {performance_factors}. "
                        "These are the fallback: a matching registry row takes precedence, and the "
                        "factors actually applied are logged after the generators are added."
                    )
                else:
                    logger.warning("renewables.performance_factors.enabled=true but no factors or components defined")

        # Optional evidence registry for technology-and-resource-region
        # factors. It takes precedence over the carrier-wide value while the
        # latter remains an explicit fallback for backwards compatibility.
        factor_registry = None
        factor_context = {
            "calendar_year": int(modelled_year),
            "calendar_month": None,
            # Weather-pinned registry rows (e.g. the CP30 onshore calibration
            # against weather year 2013) match only their own renewables_year.
            "weather_year": int(renewables_year) if renewables_year is not None else None,
        }
        registry_cfg = scenario_config.get("renewables", {}).get("performance_factor_registry", {})
        if registry_cfg.get("enabled", False):
            registry_path = registry_cfg.get("file")
            if not registry_path:
                raise ValueError("renewables.performance_factor_registry.enabled needs a file")
            factor_registry = load_performance_factor_registry(registry_path)
            period_start = scenario_config.get("solve_period", {}).get("start")
            if period_start:
                factor_context["calendar_month"] = int(pd.Timestamp(period_start).month)
            else:
                # A network built without a solve period is shared by several
                # solve-time scenarios (scenarios.yaml ``network_from``), so a
                # month-specific factor for this modelled year would silently
                # be skipped for every month. Refuse rather than build a
                # network whose p_max_pu depends on which month reads it.
                year_rows = factor_registry[
                    factor_registry["calendar_year"].isna()
                    | factor_registry["calendar_year"].eq(int(modelled_year))
                ]
                monthly_rows = year_rows[year_rows["calendar_month"].notna()]
                if not monthly_rows.empty:
                    raise ValueError(
                        "renewables.performance_factor_registry has "
                        f"{len(monthly_rows)} month-specific rows reachable for "
                        f"modelled year {modelled_year} but the scenario has no "
                        "solve_period, so the factor cannot be resolved once for "
                        "a shared network. Either move the monthly factor to "
                        "solve time or build per month (drop network_from)."
                    )
            logger.info(
                "Loaded renewable performance-factor registry: %s (%d rows; year=%s month=%s)",
                registry_path, len(factor_registry), factor_context["calendar_year"],
                factor_context["calendar_month"],
            )

        # Brownfield audit: written for every scenario so the DAG is stable; a
        # scenario without the brownfield fleet records that explicitly.
        audit_csv = getattr(snk.output, 'brownfield_audit', None)
        audit_json = getattr(snk.output, 'brownfield_summary', None)
        if audit_csv and audit_json:
            regional_audit = brownfield_state.get('regional', pd.DataFrame())
            if brownfield_state:
                audit = build_fleet_audit(
                    renewable_sites, regional_audit,
                    brownfield_state.get('retired', pd.DataFrame()),
                    brownfield_state.get('trimmed', pd.DataFrame()),
                    excluded=brownfield_state.get('excluded'),
                )
                audit['trimmed_projects_detail'] = (
                    brownfield_state.get('trimmed', pd.DataFrame())
                    .reindex(columns=['site_name', 'technology', 'cp30_region', 'development_status',
                                      'capacity_mw', 'trimmed_mw', 'trim_action'])
                    .to_dict('records')
                    if len(brownfield_state.get('trimmed', pd.DataFrame())) else []
                )
                refill_detail = brownfield_state.get('refill_detail', pd.DataFrame())
                audit['refill_at_excluded_landing_detail'] = (
                    refill_detail.astype(str).to_dict('records') if len(refill_detail) else []
                )
                audit['retired_projects_detail'] = (
                    brownfield_state.get('retired', pd.DataFrame())
                    .reindex(columns=['site_name', 'technology', 'capacity_mw', 'operational_date',
                                      'retirement_expiry'])
                    .astype(str).to_dict('records')
                    if len(brownfield_state.get('retired', pd.DataFrame())) else []
                )
            else:
                audit = {"method": "proxy_only", "note": "fleet.spatial_proxy.brownfield not enabled for this scenario"}
            if support_register_state:
                audit['support_registers'] = support_register_state
            regional_audit.to_csv(audit_csv, index=False)
            with open(audit_json, 'w', encoding='utf-8') as handle:
                json.dump(audit, handle, indent=2, default=str)
            if brownfield_state:
                for carrier, entry in audit['carriers'].items():
                    logger.info(
                        "Brownfield %s: %.0f MW named of %.0f MW (%s), %d projects",
                        carrier, entry['named_mw'], entry['total_mw'],
                        f"{entry['named_share']:.0%}" if entry['named_share'] is not None else 'n/a',
                        entry['named_projects'],
                    )

        default_profiles = {}
        hydro_profile = build_large_hydro_default_profile(
            network, scenario_config, renewables_year,
            getattr(snk.input, 'large_hydro_monthly_shape', None) or None,
            getattr(snk.input, 'cp30_es1', None) or None,
            logger,
        )
        if hydro_profile is not None:
            default_profiles['large_hydro'] = hydro_profile

        # Capacity behind each site profile column (the MW the profile rule
        # multiplied the capacity factor by), so a unit whose p_nom was trimmed
        # or split keeps its site's capacity factor (G2, audit 27 Sep 2026).
        profile_capacities = build_profile_capacity_map(snk.input, logger)
        profile_locations = build_profile_location_map(snk.input)
        proxy_profile_audit = {}

        network = add_renewable_generators(
            network,
            renewable_sites,
            profiles_dir=profiles_dir,
            p_nom_col='capacity_mw',
            renewables_year=renewables_year,
            performance_factors=performance_factors,
            performance_factor_registry=factor_registry,
            performance_factor_context=factor_context,
            commissioning_profile_gating=commissioning_profile_gating,
            commissioning_date_overrides=commissioning_date_overrides,
            default_profiles=default_profiles,
            profile_capacities=profile_capacities,
            proxy_profiles=scenario_config.get('renewables'),
            profile_audit=proxy_profile_audit,
            profile_locations=profile_locations,
        )
        log_applied_performance_factors(network, performance_factors, logger)
        # The brownfield audit was written before the units existed: add the
        # proxy profile basis and each proxy unit's group now.
        if (proxy_profile_audit.get('basis') == 'regional' and audit_json
                and Path(str(audit_json)).is_file()):
            with open(audit_json, encoding='utf-8') as handle:
                audit = json.load(handle)
            for entry in (audit.get('carriers') or {}).values():
                if isinstance(entry, dict) and 'profile_basis' in entry:
                    entry['profile_basis']['increment'] = 'regional_fleet_average'
            groups = proxy_profile_audit.get('groups', pd.DataFrame())
            assignments = proxy_profile_audit.get('assignments', pd.DataFrame())
            audit['proxy_profiles'] = {
                'basis': 'regional_fleet_average',
                'settings': proxy_profile_audit.get('settings'),
                'groups': groups.to_dict('records') if len(groups) else [],
                'units': assignments.to_dict('records') if len(assignments) else [],
            }
            with open(audit_json, 'w', encoding='utf-8') as handle:
                json.dump(audit, handle, indent=2, default=str)

        # Replace reviewed planning-record aggregates with their separately
        # accredited physical units before optional renewable aggregation.
        network = apply_historical_station_unit_splits(
            network,
            subsidy_config.get("historical_station_unit_splits_file"),
            logger,
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
        
        # STAGE 6.5: Optionally aggregate merchant renewables by (bus, carrier)
        # This reduces thousands of individual REPD site generators into one
        # representative generator per (bus, carrier) group, using capacity-weighted
        # average p_max_pu profiles. Dramatically reduces problem size without
        # changing optimization results.
        agg_config = scenario_config.get('renewables', {}).get('aggregation', {})
        if agg_config.get('enabled', False):
            stage_start = time.time()
            logger.info("-" * 80)
            logger.info("AGGREGATING RENEWABLE GENERATORS BY (BUS, CARRIER)")
            logger.info("-" * 80)
            
            from scripts.generators.aggregate_renewable_generators import DEFAULT_RENEWABLE_CARRIERS
            agg_carriers = agg_config.get('carriers', DEFAULT_RENEWABLE_CARRIERS)
            preserve_support_types = agg_config.get('preserve_support_types', [])
            
            pre_agg_count = len(network.generators)
            # Log per-carrier capacity before aggregation for verification
            pre_agg_capacity = network.generators.groupby('carrier')['p_nom'].sum()
            
            network, removed_count = aggregate_renewables_by_bus(
                network,
                carriers=agg_carriers,
                preserve_support_types=preserve_support_types,
                logger=logger,
            )
            
            post_agg_count = len(network.generators)
            post_agg_capacity = network.generators.groupby('carrier')['p_nom'].sum()
            
            # Verify capacity conservation
            for carrier in agg_carriers:
                before = pre_agg_capacity.get(carrier, 0.0)
                after = post_agg_capacity.get(carrier, 0.0)
                if abs(before - after) > 0.01:
                    logger.error(
                        f"CAPACITY MISMATCH for {carrier}: "
                        f"{before:.2f} MW before vs {after:.2f} MW after aggregation!"
                    )
                else:
                    logger.debug(f"  {carrier}: capacity conserved at {after:.2f} MW")
            
            logger.info(f"Aggregation reduced generators from {pre_agg_count} to {post_agg_count} "
                       f"(removed {removed_count})")

            # STAGE 6.6: Re-run bus corrections on aggregated generators
            # After aggregation, individual 50 MW farms that were below the check
            # threshold may have been combined into 500+ MW aggregate generators
            # that now exceed the export capacity of their 33/132kV bus.
            # Re-run apply_etys_bmu_mapping with a lower threshold to catch these.
            if network_model.upper() == 'ETYS':
                stage_start_post = time.time()
                logger.info("-" * 80)
                logger.info("POST-AGGREGATION BUS CORRECTIONS")
                logger.info("-" * 80)

                # Build a DataFrame from network.generators for the correction function
                agg_gen_df = network.generators[
                    network.generators.carrier.isin(agg_carriers)
                ].copy()

                if not agg_gen_df.empty:
                    # Use 'name' index as site_name for matching
                    agg_gen_df['site_name'] = agg_gen_df.index

                    pre_buses = agg_gen_df['bus'].copy()
                    agg_gen_df = apply_etys_bmu_mapping(
                        agg_gen_df, network, min_check_mw=10.0
                    )

                    # Apply any corrections back to network.generators
                    changed = (agg_gen_df['bus'] != pre_buses).sum()
                    if changed > 0:
                        for gen_idx in agg_gen_df.index:
                            if agg_gen_df.at[gen_idx, 'bus'] != pre_buses[gen_idx]:
                                network.generators.at[gen_idx, 'bus'] = agg_gen_df.at[gen_idx, 'bus']
                        logger.info(f"Post-aggregation: corrected {changed} aggregate generator bus assignments")
                    else:
                        logger.info("Post-aggregation: no bus corrections needed")

                stage_times['6.6. Post-agg bus corrections'] = time.time() - stage_start_post

            stage_times['6.5. Aggregate renewables'] = time.time() - stage_start
        else:
            logger.info("Renewable aggregation disabled (set renewables.aggregation.enabled: true to enable)")
        
        # STAGE 6.75: Add embedded generation from ESPENI (historical only)
        # TOTAL_ESPENI already includes embedded solar/wind output, so demand
        # is unchanged.  The demand step saved the national ESPENI embedded MW
        # timeseries to CSV.  Here we subtract the output already represented
        # by REPD generators (to avoid double-counting) and add the remaining
        # gap as explicit embedded_solar / embedded_wind generators.
        emb_cfg = scenario_config.get('renewables', {}).get('embedded', {})
        if emb_cfg.get('enabled', False) and is_historical:
            stage_start = time.time()
            logger.info("-" * 80)
            logger.info("ADDING EMBEDDED GENERATION FROM ESPENI")
            logger.info("-" * 80)
            
            _EMBEDDED_CARRIERS = {
                'solar': 'embedded_solar',
                'wind':  'embedded_wind',
            }
            
            for source_key, carrier_name in _EMBEDDED_CARRIERS.items():
                if not emb_cfg.get(source_key, False):
                    continue
                
                # Load the profile saved by the demand step
                emb_path = Path(f"resources/demand/{scenario_name}_embedded_{source_key}.csv")
                if not emb_path.exists():
                    logger.warning(f"Embedded {source_key} profile not found at {emb_path} – skipping")
                    continue
                
                emb_df = pd.read_csv(emb_path, index_col=0, parse_dates=True)
                emb_national = emb_df['p_mw']
                raw_peak_mw = emb_national.max()
                
                if raw_peak_mw <= 0:
                    logger.warning(f"Embedded {source_key} has zero peak – skipping")
                    continue
                
                logger.info(f"Embedded {source_key} (raw ESPENI): peak {raw_peak_mw:.0f} MW, "
                           f"annual {emb_national.sum() * (scenario_config.get('timestep_minutes', 60) / 60) / 1e6:.2f} TWh")
                
                # ── Subtract REPD generator output to avoid double-counting ──
                # ESPENI NGEM_EMBEDDED_SOLAR includes output from ALL solar farms
                # (there is no ELEX solar column), so REPD solar_pv generators
                # already account for most of this.  The gap is rooftop / small-
                # scale solar that REPD does not capture.
                # For wind, NGEM_EMBEDDED_WIND covers distribution-connected wind
                # only; transmission-connected REPD wind is in ELEX_WIND instead.
                # These are separate ESPENI columns with no overlap — do NOT subtract.
                _OVERLAP_CARRIERS = {
                    'solar': ['solar_pv'],
                    'wind':  [],   # NGEM_EMBEDDED_WIND != ELEX_WIND; no overlap with REPD
                }
                overlap_carriers = _OVERLAP_CARRIERS.get(source_key, [])
                if source_key == 'wind':
                    # Explicitly supported small wind projects stay visible for
                    # project-level accounting, but their output is already in
                    # ESPENI embedded wind. Subtract only those retained sites.
                    overlap_cfg = emb_cfg.get('wind_overlap_exclusion', {})
                    max_capacity_mw = float(overlap_cfg.get('max_capacity_mw', 0.0) or 0.0)
                    support = network.generators.get(
                        'support_type', pd.Series('', index=network.generators.index)
                    ).fillna('').astype(str)
                    overlap_gens = network.generators[
                        network.generators.carrier.eq('wind_onshore')
                        & network.generators.p_nom.le(max_capacity_mw)
                        & support.isin(['CfD', 'ROC'])
                    ]
                    overlap_carriers = ['explicitly supported small wind_onshore']
                else:
                    overlap_gens = network.generators[
                        network.generators.carrier.isin(overlap_carriers)
                    ]
                if len(overlap_gens) > 0:
                    # Compute aggregate modelled output from overlapping generators
                    overlap_output = pd.Series(0.0, index=network.snapshots)
                    for gen_name in overlap_gens.index:
                        p_nom = overlap_gens.at[gen_name, 'p_nom']
                        if gen_name in network.generators_t.p_max_pu.columns:
                            ppu = network.generators_t.p_max_pu[gen_name]
                        else:
                            ppu = overlap_gens.at[gen_name, 'p_max_pu']
                        overlap_output += p_nom * ppu
                    
                    # Align ESPENI profile to network snapshots before subtraction
                    emb_national.index = pd.DatetimeIndex(emb_national.index)
                    emb_aligned = emb_national.reindex(network.snapshots, method='nearest').fillna(0.0)
                    
                    gap = (emb_aligned - overlap_output).clip(lower=0.0)
                    
                    logger.info(f"  Overlap with {overlap_carriers}: peak modelled output "
                               f"{overlap_output.max():.0f} MW")
                    logger.info(f"  Gap after subtraction: peak {gap.max():.0f} MW "
                               f"(raw was {raw_peak_mw:.0f} MW)")
                    
                    emb_national = gap
                else:
                    # No overlapping generators — use full ESPENI profile
                    emb_national.index = pd.DatetimeIndex(emb_national.index)
                    emb_national = emb_national.reindex(network.snapshots, method='nearest').fillna(0.0)
                
                peak_mw = emb_national.max()
                if peak_mw <= 0:
                    logger.info(f"  Embedded {source_key} gap is zero after subtracting REPD — skipping")
                    continue
                
                # Ensure carrier exists in network
                if carrier_name not in network.carriers.index:
                    network.add("Carrier", carrier_name, color="#FFA500" if 'solar' in carrier_name else "#87CEEB",
                                co2_emissions=0.0, nice_name=f"Embedded {source_key.title()}")
                
                # Distribute across load buses proportionally to their demand share
                load_buses = network.loads['bus'].unique()
                if len(load_buses) == 0:
                    logger.warning("No load buses found – cannot distribute embedded generation")
                    continue
                
                # Calculate demand share per bus from the load time series
                if not network.loads_t.p_set.empty:
                    bus_demand = network.loads_t.p_set.sum()  # Total MWh per load
                    # Map from load names to buses
                    load_to_bus = network.loads['bus']
                    bus_total = bus_demand.groupby(load_to_bus).sum()
                    total_demand = bus_total.sum()
                    if total_demand > 0:
                        bus_share = bus_total / total_demand
                    else:
                        bus_share = pd.Series(1.0 / len(load_buses), index=load_buses)
                else:
                    # Uniform distribution as fallback
                    bus_share = pd.Series(1.0 / len(load_buses), index=load_buses)

                # Embedded wind goes where the small wind farms it stands for are, not where the demand is
                # (18 Nov 2026: by demand it put about 90% in England, where 42% of the excluded farms are in
                # Scotland, and the 2020 validation's Scottish boundaries carried too little). Each excluded
                # site's capacity is placed on its nearest load bus; demand share stays the fallback.
                if source_key == 'wind' and embedded_wind_sites is not None and len(embedded_wind_sites):
                    site_share = _embedded_site_bus_share(network, embedded_wind_sites, load_buses)
                    if site_share is not None:
                        bus_share = site_share
                        logger.info(f"  Embedded wind placed by the {len(embedded_wind_sites)} excluded small "
                                    f"onshore sites' capacity on {len(site_share)} load buses")

                # Normalize the national profile to [0, 1]
                # (emb_national is already aligned to network.snapshots above)
                p_max_pu = emb_national / peak_mw
                p_max_pu.index = network.snapshots
                
                # Add one generator per load bus
                gen_count_before = len(network.generators)
                for bus in bus_share.index:
                    share = bus_share[bus]
                    if share <= 0:
                        continue
                    
                    gen_name = f"embedded_{source_key}_{bus}"
                    bus_p_nom = peak_mw * share
                    
                    network.add("Generator", gen_name,
                                bus=bus,
                                carrier=carrier_name,
                                p_nom=bus_p_nom,
                                p_max_pu=p_max_pu,
                                marginal_cost=0.0,
                                capital_cost=0.0,
                                p_nom_extendable=False)
                
                gen_count_after = len(network.generators)
                logger.info(f"  Added {gen_count_after - gen_count_before} embedded {source_key} generators "
                           f"({peak_mw:.0f} MW total across {len(bus_share[bus_share > 0])} buses)")
            
            stage_times['6.75. Embedded generation'] = time.time() - stage_start
        
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
        
        # Include embedded generation carriers in summary
        for emb_carrier in ['embedded_solar', 'embedded_wind']:
            emb_gens = network.generators[network.generators['carrier'] == emb_carrier]
            if len(emb_gens) > 0:
                summary_data.append({
                    'technology': emb_carrier,
                    'capacity_mw': emb_gens['p_nom'].sum(),
                    'count': len(emb_gens)
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

