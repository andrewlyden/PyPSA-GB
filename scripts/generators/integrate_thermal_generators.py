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
import logging
import os
from pathlib import Path
import time
import yaml

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info, log_execution_summary, log_stage_timing, log_stage_summary
from scripts.utilities.carrier_definitions import get_carrier_definitions, add_carriers_to_network
import difflib
import re

# Import shared spatial utilities and renewable generator functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scripts.utilities.spatial_utils import (
    map_sites_to_buses, apply_etys_bmu_mapping,
    detect_coordinate_system, fuzzy_match_string,
    configure_spatial_mapping,
    _DIST_GSP_TO_BUS_KM, _DIST_THERMAL_KM, _SHETL_LAT_MIN, _SPTL_LAT_MIN,
    _FALLBACK_MAJOR_BUS_COUNT,
)
from integrate_renewable_generators import (
    _assign_subsidy_attributes,
    load_generator_characteristics,
)
from scripts.CP30.generation import load_cp30_generation_capacities, scale_proxy_to_cp30_targets, target_member_carriers
from scripts.CP30.nuclear_schedule import (
    CCS_BIOMASS_DATA_SOURCE as CCS_BIOMASS_SCHEDULE_SOURCE, DATA_SOURCE as NUCLEAR_SCHEDULE_SOURCE,
    apply_nuclear_station_schedule,
)
from scripts.CP30.committed_overlay import (
    DATA_SOURCE as OVERLAY_SOURCE, apply_committed_overlay, apply_overlay_carve_out, resolve_overlay_carve_out,
)
from scripts.CP30.gas_schedule import (
    BIOMASS_CARRIERS as SCHEDULE_BIOMASS_CARRIERS, BIOMASS_DATA_SOURCE as BIOMASS_SCHEDULE_SOURCE,
    DATA_SOURCE as GAS_SCHEDULE_SOURCE, GAS_CARRIERS as SCHEDULE_GAS_CARRIERS, apply_gas_station_schedule,
)
from scripts.CP30.cfd_station_schedule import (
    CFD_STATION_SOURCE as CFD_STATION_SCHEDULE_SOURCE, apply_cfd_station_schedule,
)
from scripts.utilities.scenario_detection import target_source, uses_brownfield
from scripts.generators.fes_supply import (
    apply_fes_thermal, fes_supply_part, restore_declared_buses, schedule_data_source,
)
from scripts.generators.thermal_availability import apply_thermal_outage_availability
from scripts.generators.thermal_ramping import apply_thermal_ramp_limits
from scripts.generators.support_scheme_tranches import (
    apply_generator_monthly_availability,
    apply_physical_commissioning_gates,
    split_historical_cfd_tranches,
)
from scripts.generators.historical_dispatch_anchors import apply_historical_dispatch_anchors
from scripts.generators.historical_roc_physical_overlay import (
    apply_historical_roc_physical_overlay,
)
from scripts.generators.ofgem_roc_capacity_overlay import apply_ofgem_roc_capacity_overlay
from scripts.consumer_bill.map_pypsa_generators_to_ofgem_roc import (
    apply_historical_dispatch_labels,
)
import numpy as np

# Initialize logging
logger = setup_logging("integrate_thermal_generators")


def add_missing_dukes_hydro(
    thermal_data: pd.DataFrame,
    dukes_data: pd.DataFrame,
    repd_hydro_path: str | Path,
    logger,
) -> pd.DataFrame:
    """Supplement REPD with named legacy DUKES hydro stations.

    REPD is a planning register and omits much of the pre-1990 conventional
    hydro fleet.  DUKES 5.11 contains those major power producer stations.  We
    union the two station registers by normalised name and add only DUKES
    stations absent from the unaggregated REPD source list.
    """
    if dukes_data.empty or not Path(repd_hydro_path).exists():
        return thermal_data

    def normalise(value):
        text = str(value or "").lower()
        text = re.sub(r"\b(hydro scheme|hydro|power station)\b", " ", text)
        return re.sub(r"[^a-z0-9]+", "", text)

    repd = pd.read_csv(repd_hydro_path)
    repd_names = {
        normalise(value) for value in repd.get("site_name", pd.Series(dtype=object))
    }
    hydro = dukes_data.loc[
        dukes_data["fuel_type"].astype(str).str.fullmatch("Hydro", case=False, na=False)
    ].copy()
    hydro = hydro.loc[~hydro["station_name"].map(normalise).isin(repd_names)].copy()
    if hydro.empty:
        return thermal_data
    hydro["fuel_type"] = "large_hydro"
    hydro["source_site_name"] = hydro["station_name"]
    hydro["data_source"] = "DUKES_5.11_legacy_hydro_supplement"
    hydro["support_type"] = "merchant"
    logger.info(
        "Adding %d legacy DUKES hydro stations omitted from REPD (%.1f MW)",
        len(hydro), hydro["capacity_mw"].sum(),
    )
    return pd.concat([thermal_data, hydro], ignore_index=True, sort=False)


def add_roc_evidenced_dukes_hydro(
    thermal_data: pd.DataFrame,
    network: pypsa.Network,
    supplement_path: str | Path | None,
    logger,
) -> pd.DataFrame:
    """Add audited legacy hydro stations omitted by the historical DUKES extract.

    The historical DUKES table used by the standard build aggregates or omits
    some sub-20 MW hydro stations. This narrow supplement contains only stations
    independently evidenced by the Ofgem calendar-year ROC outturn and a named
    DUKES 5.11 station record, so it cannot introduce unreviewed capacity.
    """
    if not supplement_path:
        return thermal_data
    path = Path(str(supplement_path))
    if not path.exists():
        raise FileNotFoundError(f"ROC-evidenced DUKES hydro supplement not found: {path}")
    supplement = pd.read_csv(path)
    required = {
        "station_name", "fuel_type", "capacity_mw", "source_site_name",
        "support_type", "ro_banding", "ro_accreditation_reference",
    }
    if missing := required.difference(supplement.columns):
        raise ValueError(f"ROC-evidenced DUKES hydro supplement is missing: {sorted(missing)}")

    def normalise(value):
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    existing_names = {normalise(name) for name in network.generators.index}
    for column in ("station_name", "source_site_name"):
        if column in thermal_data.columns:
            existing_names.update(thermal_data[column].dropna().map(normalise))
    supplement = supplement.loc[
        ~supplement["station_name"].map(normalise).isin(existing_names)
    ].copy()
    if supplement.empty:
        logger.info("All ROC-evidenced DUKES hydro stations are already represented")
        return thermal_data
    logger.info(
        "Adding %d ROC-evidenced legacy DUKES hydro stations (%.1f MW) from %s",
        len(supplement), supplement["capacity_mw"].sum(), path,
    )
    return pd.concat([thermal_data, supplement], ignore_index=True, sort=False)


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
    - SHETL: Northern Scotland (lat > 57.0)
    - SPTL: Southern Scotland (55.5 < lat <= 57.0)
    - NGET: England & Wales (lat <= 55.5)

    Bus ``y`` is latitude on a WGS84 network but northing in metres on an
    OSGB36 one (the ETYS build); northings are converted to latitude first.
    Until 23 Sep 2026 they were compared with the degree thresholds directly,
    which put every ETYS bus in SHETL and left NGET and SPTL empty.

    Args:
        network: PyPSA Network with buses having 'x'/'y' coordinates
        region: Region code ('SHETL', 'SPTL', or 'NGET')

    Returns:
        List of bus names in the region
    """
    buses = network.buses
    lat = buses['y'].astype(float)
    if buses['x'].abs().max() > 180:
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        _, lat_values = to_wgs84.transform(buses['x'].astype(float).values, buses['y'].astype(float).values)
        lat = pd.Series(lat_values, index=buses.index)

    if region == 'SHETL':
        region_buses = buses[lat > _SHETL_LAT_MIN]
    elif region == 'SPTL':
        region_buses = buses[(lat > _SPTL_LAT_MIN) & (lat <= _SHETL_LAT_MIN)]
    else:  # NGET
        region_buses = buses[lat <= _SPTL_LAT_MIN]

    return region_buses.index.tolist()


def distribute_fes_generators_spatially(
    fes_data: pd.DataFrame,
    network: pypsa.Network,
    repd_sites: pd.DataFrame = None,
    reference_sites: dict | None = None,
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
        reference_sites: Optional ``{carrier: DataFrame}`` of placed sites (with
            ``bus`` and ``capacity_mw``) whose capacity-weighted buses are the
            pattern for that carrier, ahead of the historical and REPD patterns

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

        bus_weights = {}
        reference = (reference_sites or {}).get(carrier)
        if reference is not None and len(reference) > 0:
            in_region = reference[reference['bus'].isin(region_buses)]
            if len(in_region) > 0:
                bus_capacity = in_region.groupby('bus')['capacity_mw'].sum()
                bus_weights = (bus_capacity / bus_capacity.sum()).to_dict()
                logger.info(f"    Using reference-site pattern: {len(bus_weights)} buses in {region}")

        # Get historical bus distribution for this carrier
        if not bus_weights:
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


def check_min_output_within_availability(avail_config: dict) -> None:
    """Refuse a must-run floor above the carrier's availability cap.

    ``min_output`` above ``factors`` is p_min_pu > p_max_pu in every hour, an
    infeasible LP. The two halves are usually set in different places (the cap
    inherited from config/defaults/, the floor in the scenario): the CP30 waste
    floor 0.55 sat under the inherited 0.80 cap, and the CP30-implied cap of
    about 0.62 (ES1 12.8 TWh on 2,481 MW is CF 0.59; audit 26 Sep 2026) must
    stay above it.
    """
    factors = avail_config.get('factors', {}) or {}
    floors = avail_config.get('min_output', {}) or {}
    floor_over_cap = {
        carrier: (float(floors[carrier]), float(factors[carrier]))
        for carrier in set(factors) & set(floors)
        if float(floors[carrier]) > float(factors[carrier])
    }
    if floor_over_cap:
        raise ValueError(
            "thermal.availability_factors: min_output above the availability factor "
            f"(floor, cap) for {floor_over_cap}"
        )


GROUP_OVERRIDE_KEYS = {"carrier", "data_source", "factor", "min_output"}


def apply_thermal_group_overrides(network, overrides, logger: logging.Logger) -> pd.DataFrame:
    """``thermal.availability_factors.group_overrides``: cap and floor per (carrier, data_source).

    The carrier values (``factors``, ``min_output``) treat a carrier as one fleet. A
    CP30 carrier can hold groups that the pathway runs differently: ES1 runs
    transmission biomass (Drax, Lynemouth, MGT Teesside, the station schedule) at
    CF 0.40 and distributed biomass (the CP30 proxy) at 0.37, while the two are one
    ``biomass`` carrier here (gas_cal, 14 Oct 2026). Each entry names ``carrier`` and
    ``data_source`` and sets ``factor`` (static ``p_max_pu``) and/or ``min_output``
    (static ``p_min_pu``) on those generators, after the carrier values. An entry
    that matches no generator, or leaves a unit's floor above its cap, is refused.
    Call before outage availability, which multiplies the static cap.

    Returns one row per entry: carrier, data_source, generators, p_nom_mw, factor, min_output.
    """
    records = []
    if not overrides:
        return pd.DataFrame(columns=["carrier", "data_source", "generators", "p_nom_mw", "factor", "min_output"])
    if "data_source" not in network.generators.columns:
        raise ValueError("thermal.availability_factors.group_overrides needs generators.data_source")
    gens = network.generators
    for entry in overrides:
        entry = dict(entry or {})
        unknown = set(entry) - GROUP_OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"group_overrides entry has unknown keys {sorted(unknown)}: {entry}")
        if not entry.get("carrier") or not entry.get("data_source"):
            raise ValueError(f"group_overrides entry needs carrier and data_source: {entry}")
        if "factor" not in entry and "min_output" not in entry:
            raise ValueError(f"group_overrides entry sets neither factor nor min_output: {entry}")
        mask = gens["carrier"].eq(entry["carrier"]) & gens["data_source"].astype(str).eq(str(entry["data_source"]))
        names = gens.index[mask]
        if not len(names):
            raise ValueError(f"group_overrides entry matches no generator: {entry}")
        if "factor" in entry:
            factor = float(entry["factor"])
            if not 0.0 < factor <= 1.0:
                raise ValueError(f"group_overrides factor must be in (0, 1]: {entry}")
            if any(n in network.generators_t.p_max_pu.columns for n in names):
                raise ValueError(f"group_overrides factor on generators with a p_max_pu profile: {entry}")
            network.generators.loc[names, "p_max_pu"] = factor
        if "min_output" in entry:
            floor = float(entry["min_output"])
            if not 0.0 <= floor <= 1.0:
                raise ValueError(f"group_overrides min_output must be in [0, 1]: {entry}")
            network.generators.loc[names, "p_min_pu"] = floor
        caps = pd.to_numeric(network.generators.loc[names, "p_max_pu"], errors="coerce").fillna(1.0)
        floors = pd.to_numeric(network.generators.loc[names, "p_min_pu"], errors="coerce").fillna(0.0)
        if (floors > caps + 1e-12).any():
            raise ValueError(f"group_overrides leaves min_output above the availability factor: {entry}")
        p_nom = float(network.generators.loc[names, "p_nom"].sum())
        records.append({"carrier": entry["carrier"], "data_source": entry["data_source"],
                        "generators": len(names), "p_nom_mw": p_nom,
                        "factor": float(caps.mean()), "min_output": float(floors.mean())})
        logger.info(
            f"  group override {entry['carrier']} / {entry['data_source']}: {len(names)} generators, "
            f"{p_nom:,.0f} MW, p_max_pu {caps.mean():.3f}, p_min_pu {floors.mean():.3f}"
        )
    return pd.DataFrame(records)


def capacity_weighted_fleet_profile(profiles_df: pd.DataFrame) -> pd.Series:
    """Fleet capacity factor: total MW over total site capacity (column peaks).

    ``profiles_df.mean(axis=1)`` of MW columns is an average site OUTPUT in MW,
    not a CF; clipped to [0, 1] it pinned the CP30 marine fleet at full output
    in 16% of hours (CF 0.467 against 0.218; audit 26 Sep 2026).
    """
    if profiles_df.empty:
        return pd.Series(dtype=float)
    if float(profiles_df.max().max()) <= 1.0:
        return profiles_df.mean(axis=1)
    capacity = float(profiles_df.max().sum())
    return profiles_df.sum(axis=1) / capacity if capacity > 0 else profiles_df.sum(axis=1) * 0.0


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
                    # Get profile and align to network snapshots. The column is
                    # the REPD site's MW series; divide by that generator's
                    # capacity before it is clipped as a capacity factor.
                    profile = profiles_df[profile_source]
                    source_p_nom = float(network.generators.at[profile_source, 'p_nom'])                         if profile_source in network.generators.index else 0.0
                    if profile.max() > 1.0:
                        profile = profile / (source_p_nom if source_p_nom > 0 else profile.max())
                    
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
            
            avg_profile = capacity_weighted_fleet_profile(profiles_df)
            
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
                # The ETYS/FES 2024 GSP file uses Name/Latitude/Longitude; it was
                # never read before 23 Sep 2026, so no GSP mapped at all.
                if {'Name', 'Latitude', 'Longitude'}.issubset(df.columns) and 'gsp_name' not in df.columns:
                    df = df.rename(columns={'Name': 'gsp_name', 'Latitude': 'gsp_lat', 'Longitude': 'gsp_lon'})
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
        try:
            # Map GSP locations to network buses
            gsp_mapped = map_sites_to_buses(
                network,
                gsp_coords,
                method='nearest',
                lat_col='lat',
                lon_col='lon',
                max_distance_km=_DIST_GSP_TO_BUS_KM
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
                    max_distance_km=_DIST_GSP_TO_BUS_KM
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
        
        # Get top-N most connected buses (N from network.spatial_mapping config)
        sorted_buses = sorted(bus_degree.items(), key=lambda x: x[1], reverse=True)
        major_buses = [b for b, _ in sorted_buses[:_FALLBACK_MAJOR_BUS_COUNT]]
    
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

    # Initialize bus column if it doesn't exist
    if 'bus' not in thermal_data.columns:
        thermal_data['bus'] = pd.NA

    # FES 2025 names the ETYS node behind each GSP row; use it when it is a bus
    # of this network, before any name or coordinate matching.
    if 'node_id' in thermal_data.columns:
        node = thermal_data['node_id'].astype('string').str.strip()
        by_node = needs_mapping & node.isin(network.buses.index)
        thermal_data.loc[by_node, 'bus'] = node[by_node]
        logger.info(
            f"GSP-to-bus mapping: {int(by_node.sum())} generators "
            f"({thermal_data.loc[by_node, 'capacity_mw'].sum():,.1f} MW) placed at their FES node"
        )
        needs_mapping = needs_mapping & ~by_node
        if needs_mapping.sum() == 0:
            return thermal_data

    # Load GSP-to-bus mapping
    gsp_mapping = load_gsp_to_bus_mapping(network)
    
    if not gsp_mapping:
        logger.warning("No GSP-to-bus mapping available")
        return thermal_data

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
            match, score = fuzzy_match_string(gsp_str, list(gsp_mapping.keys()), threshold=0.7)
            if match:
                thermal_data.loc[idx, 'bus'] = gsp_mapping[match]
                mapped_count += 1
                logger.debug(f"Fuzzy matched GSP '{gsp_str}' to '{match}' (score={score:.2f}) -> bus '{gsp_mapping[match]}'")
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


def apply_plant_phase_out_filtering(df: pd.DataFrame, modelled_year: int) -> pd.DataFrame:
    """
    Remove generators that have been decommissioned before the modelled year.

    Uses coal_phase_out_dates.csv and nuclear_phase_out_dates.csv from data/generators/
    to filter out plants that closed before the modelled year. Also applies known
    capacity reductions for plants operating at reduced capacity in their final years.

    A plant is only removed if its closure date falls strictly before 1 January of the
    modelled year, so a plant closing partway through the modelled year (e.g. Sep 2024)
    is retained (at reduced capacity if applicable) for scenarios set in that year.

    Parameters
    ----------
    df : pd.DataFrame
        Generator DataFrame with 'station_name' and 'capacity_mw' columns
    modelled_year : int
        The scenario's modelled year

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with closed plants removed and capacities adjusted
    """
    if len(df) == 0 or modelled_year is None:
        return df

    initial_count = len(df)
    initial_capacity = df['capacity_mw'].sum()
    removed_plants = []

    # Load phase-out date files
    phase_out_files = [
        Path("data/generators/coal_phase_out_dates.csv"),
        Path("data/generators/nuclear_phase_out_dates.csv"),
    ]

    for phase_out_file in phase_out_files:
        if not phase_out_file.exists():
            logger.debug(f"Phase-out file not found: {phase_out_file}")
            continue

        phase_out_df = pd.read_csv(phase_out_file)
        phase_out_df['date'] = pd.to_datetime(phase_out_df['date'], dayfirst=True)
        phase_out_df['closure_year'] = phase_out_df['date'].dt.year

        for _, row in phase_out_df.iterrows():
            plant_name = row['name']
            closure_year = row['closure_year']

            if closure_year >= modelled_year:
                continue  # Plant still operational during modelled year

            # Match by partial name (e.g. "Ratcliffe" matches "Ratcliffe (Steam)")
            mask = df['station_name'].str.contains(plant_name, case=False, na=False)
            if mask.any():
                matched = df[mask]
                for _, gen in matched.iterrows():
                    removed_plants.append(f"{gen['station_name']} ({gen['capacity_mw']:.0f} MW, closed {closure_year})")
                df = df[~mask]

    if removed_plants:
        logger.info(f"Removed {len(removed_plants)} generators closed before {modelled_year}:")
        for plant in removed_plants:
            logger.info(f"  - {plant}")
        logger.info(f"  Capacity: {initial_capacity:.0f} MW -> {df['capacity_mw'].sum():.0f} MW")

    # Apply known capacity reductions for plants running at reduced capacity
    # Ratcliffe-on-Soar: 4 x ~500 MW units. By 2022-2024, only 1-2 units operational.
    # Source: Ofgem/National Grid ESO data on available coal capacity
    coal_capacity_reductions = {
        # plant_name_pattern: {year: available_mw}
        'Ratcliffe': {
            2022: 1000,  # 2 units operational
            2023: 500,   # 1 unit operational most of the time
            2024: 500,   # Final year before closure (Sept 2024)
        },
    }

    for plant_pattern, year_capacities in coal_capacity_reductions.items():
        if modelled_year not in year_capacities:
            continue

        mask = df['station_name'].str.contains(plant_pattern, case=False, na=False)
        if mask.any():
            reduced_cap = year_capacities[modelled_year]
            for idx in df.index[mask]:
                old_cap = df.loc[idx, 'capacity_mw']
                if old_cap > reduced_cap:
                    df.loc[idx, 'capacity_mw'] = reduced_cap
                    logger.info(f"Reduced {df.loc[idx, 'station_name']} capacity: "
                               f"{old_cap:.0f} MW -> {reduced_cap:.0f} MW "
                               f"(reduced operations in {modelled_year})")

    final_count = len(df)
    if final_count < initial_count:
        logger.info(f"Phase-out filtering: {initial_count} -> {final_count} generators "
                    f"({initial_capacity:.0f} -> {df['capacity_mw'].sum():.0f} MW)")

    return df


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
            # FES 2025 names the ETYS node behind each GSP row; the GSP mapping
            # uses it before falling back to GSP coordinates.
            'node_id': row.get('Node ID', None),
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


def load_cp30_generators(
    cp30_es1_path: str,
    fes_path: str,
    modelled_year: int,
    cp30_pathway: str,
    spatial_fes_scenario: str = None,
    nuclear_station_schedule: str | None = None,
    ccs_biomass_station_schedule: str | None = None,
    biomass_station_schedule: str | None = None,
    gas_station_schedule: str | None = None,
    gas_class_efficiency: dict | None = None,
    committed_overlay: str | None = None,
    ccs_gas_support_type: str | None = None,
    committed_overlay_carve_out: dict | None = None,
) -> pd.DataFrame:
    """Load CP30 dispatchable generation and distribute using FES spatial proxy rows.

    ``nuclear_station_schedule`` replaces the FES nuclear proxy with declared
    stations at their real connection buses, conserving the CP30 nuclear total.
    ``gas_station_schedule`` does the same for the CP30 gas line after it is
    split into CCGT, OCGT and gas engines (items MR1 and MR2), with per-unit
    CCGT efficiencies around ``gas_class_efficiency``. ``committed_overlay``
    appends plant CP30 omits but that is contracted and being built (Net Zero
    Teesside Power), refusing any carrier CP30 already gives capacity.
    ``committed_overlay_carve_out`` (``fleet.spatial_proxy.committed_overlay_carve_out``)
    takes an overlay carrier's MW out of a member carrier of a CP30 line before the
    station schedules run, so NZT sits inside CP30's gas total, not on top of it.
    ``ccs_gas_support_type`` (New Dispatch, 8 Oct 2026) labels CP30's own CCS gas
    proxies with that support (``DPA``): CP30's New Dispatch pathway carries 1,410 MW
    of gas CCS in 2030, which would otherwise be merchant; the label is kept through
    the projected CfD/merchant split (``support_label_declared``).
    """
    logger.info(f"Loading CP30 dispatchable generation for {cp30_pathway} {modelled_year}")
    targets = load_cp30_generation_capacities(
        cp30_es1_path,
        cp30_pathway,
        modelled_year,
        groups=["thermal"],
    )
    if targets.empty:
        logger.warning("No CP30 dispatchable generation targets found")
        return pd.DataFrame()

    proxy = load_fes_generators(fes_path, modelled_year, spatial_fes_scenario)
    if proxy.empty:
        logger.warning("No FES dispatchable spatial proxy rows found for CP30")
        return pd.DataFrame()

    scaled = scale_proxy_to_cp30_targets(
        proxy,
        targets,
        proxy_carrier_col="fuel_type",
        output_carrier_col="fuel_type",
        capacity_col="capacity_mw",
        name_prefix="CP30",
    )
    if scaled.empty:
        logger.warning("CP30 dispatchable targets could not be matched to FES spatial proxy carriers")
        return pd.DataFrame()

    # Keep the CP30 carrier as the technology label too; the original CP30 labels
    # remain available in cp30_source_type/cp30_source_subtype.
    scaled["technology"] = scaled["fuel_type"]

    if nuclear_station_schedule:
        nuclear_target = float(targets.loc[targets["carrier"].eq("nuclear"), "capacity_mw"].sum())
        if nuclear_target > 0:
            scaled = apply_nuclear_station_schedule(scaled, nuclear_station_schedule, nuclear_target, logger)

    # CP30's CCS biomass is a converted Drax unit (its transmission biomass line
    # falls by about one unit the year CCS biomass appears): one named,
    # transmission-connected station on a Power BECCS CfD (item MR7).
    if ccs_biomass_station_schedule:
        beccs_target = float(targets.loc[targets["carrier"].eq("CCS Biomass"), "capacity_mw"].sum())
        if beccs_target > 0:
            scaled = apply_nuclear_station_schedule(
                scaled, ccs_biomass_station_schedule, beccs_target, logger,
                carrier="CCS Biomass", data_source=CCS_BIOMASS_SCHEDULE_SOURCE)

    # Declared carve-out (user decision, 14 Oct 2026): the overlay's MW come out of
    # a member of a CP30 line (NZT out of CCGT) before the schedules fill it.
    carve_out = resolve_overlay_carve_out(committed_overlay_carve_out)
    if carve_out:
        if not committed_overlay:
            raise ValueError("committed_overlay_carve_out is set but there is no committed_overlay")
        scaled = apply_overlay_carve_out(scaled, committed_overlay, carve_out, targets, logger)

    # The CP30 gas line has already been split across CCGT, OCGT and gas
    # engines on the proxy's shares; each carrier's scaled total is its target.
    gas_targets = scaled[scaled["fuel_type"].isin(SCHEDULE_GAS_CARRIERS)].groupby("fuel_type")["capacity_mw"].sum()
    if len(gas_targets):
        logger.info(
            "CP30 gas line split on FES proxy shares: "
            + ", ".join(f"{carrier} {mw:,.1f} MW" for carrier, mw in gas_targets.items())
        )
    if gas_station_schedule and len(gas_targets):
        class_efficiency = {carrier: float((gas_class_efficiency or {}).get(carrier, np.nan))
                            for carrier in SCHEDULE_GAS_CARRIERS}
        missing = [carrier for carrier, value in class_efficiency.items()
                   if carrier in gas_targets.index and not np.isfinite(value)]
        if missing:
            raise ValueError(f"No class efficiency configured for gas carriers {missing} "
                             "(marginal_costs.future_costs.efficiency_defaults)")
        scaled = apply_gas_station_schedule(scaled, gas_station_schedule, gas_targets.to_dict(),
                                            class_efficiency, logger)

    # Transmission biomass at named stations; the distribution residual stays on
    # FES GSP rows (item MR6).
    if biomass_station_schedule:
        biomass_target = float(targets.loc[targets["carrier"].eq("biomass"), "capacity_mw"].sum())
        if biomass_target > 0:
            scaled = apply_gas_station_schedule(
                scaled, biomass_station_schedule, {"biomass": biomass_target}, {}, logger,
                carriers=SCHEDULE_BIOMASS_CARRIERS, data_source=BIOMASS_SCHEDULE_SOURCE,
                residual_rows="non_direct")

    if committed_overlay:
        scaled = apply_committed_overlay(scaled, committed_overlay, targets, logger, carve_out=carve_out)

    if ccs_gas_support_type:
        ccs = scaled["fuel_type"].eq("CCS Gas")
        if not ccs.any():
            raise ValueError(f"ccs_gas_support_type is {ccs_gas_support_type!r} but the CP30 pathway has no CCS gas")
        scaled.loc[ccs, "support_type"] = str(ccs_gas_support_type)
        scaled.loc[ccs, "support_label_declared"] = True
        logger.info(f"CP30 CCS gas: {int(ccs.sum())} proxy rows ({float(scaled.loc[ccs, 'capacity_mw'].sum()):,.0f} MW) "
                    f"labelled {ccs_gas_support_type}")

    scaled_summary = scaled.groupby("fuel_type")["capacity_mw"].sum()
    logger.info("CP30 dispatchable generation capacity targets applied:")
    for _, target in targets.sort_values("capacity_mw", ascending=False).iterrows():
        members = target_member_carriers(target)
        members = members + tuple(c for c, e in carve_out.items() if e["cp30_line"] == target["carrier"])
        actual_mw = float(sum(scaled_summary.get(member, 0.0) for member in members))
        label = target["carrier"] if len(members) == 1 else f"{target['carrier']} ({' + '.join(members)})"
        logger.info(f"  {label}: target={target['capacity_mw']:,.1f} MW, scaled={actual_mw:,.1f} MW")

    return scaled


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
            # Standardize columns — convert OSGB36 Eastings/Northings to WGS84 lon/lat
            for x_col, y_col in [('x_coord', 'y_coord'), ('X-coordinate', 'Y-coordinate')]:
                if x_col in df.columns and y_col in df.columns:
                    x_vals = pd.to_numeric(df[x_col], errors='coerce')
                    y_vals = pd.to_numeric(df[y_col], errors='coerce')
                    valid = x_vals.notna() & y_vals.notna()
                    if valid.any():
                        coord_sys = detect_coordinate_system(x_vals[valid].values, y_vals[valid].values)
                        logger.debug(f"REPD {fuel_type}: detected {coord_sys} coordinates in {x_col}/{y_col}")
                    else:
                        coord_sys = 'UNKNOWN'
                    if valid.any() and coord_sys == 'OSGB36':
                        from pyproj import Transformer
                        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                        lon_vals, lat_vals = transformer.transform(
                            x_vals[valid].values, y_vals[valid].values
                        )
                        df.loc[valid, 'lon'] = lon_vals
                        df.loc[valid, 'lat'] = lat_vals
                    else:
                        df['lon'] = x_vals
                        df['lat'] = y_vals
                    break
            
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


def load_historical_additional_thermal_sites(
    path: str | Path | None, modelled_year: int
) -> pd.DataFrame:
    """Return reviewed historical stations that are absent from DUKES/REPD."""
    if not path:
        return pd.DataFrame()
    sites = pd.read_csv(path)
    required = {
        'modelled_year', 'station_name', 'source_site_name', 'capacity_mw',
        'fuel_type', 'x_coord', 'y_coord', 'operational_date', 'data_source',
    }
    missing = required.difference(sites.columns)
    if missing:
        raise ValueError(f"Historical thermal site file is missing columns: {sorted(missing)}")
    years = pd.to_numeric(sites['modelled_year'], errors='raise')
    sites = sites.loc[years.eq(int(modelled_year))].copy()
    if sites.empty:
        return sites
    if sites['station_name'].duplicated().any():
        raise ValueError("Historical thermal site names must be unique within a model year")
    sites['capacity_mw'] = pd.to_numeric(sites['capacity_mw'], errors='raise')
    if (sites['capacity_mw'] <= 0).any():
        raise ValueError("Historical thermal sites require positive capacity")
    logger.info(
        "Loaded %d reviewed historical thermal site overlay(s) for %s",
        len(sites), modelled_year,
    )
    return sites


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
    # REPD thermal-site extraction already standardises ``Site Name`` to
    # ``site_name``.  Preserve it as the PyPSA component name; otherwise the
    # integration falls back to anonymous names such as ``gen_biogas_245`` and
    # an accredited-station / ROC mapping becomes impossible after the build.
    if 'station_name' not in standard_df.columns:
        if 'source_site_name' in standard_df.columns:
            standard_df['station_name'] = standard_df['source_site_name']
        elif 'site_name' in standard_df.columns:
            standard_df['station_name'] = standard_df['site_name']
        elif 'Site Name' in standard_df.columns:
            standard_df['station_name'] = standard_df['Site Name']
    
    if 'technology' not in standard_df.columns:
        standard_df['technology'] = standard_df['fuel_type']
    
    return standard_df


def apply_generator_support_overrides(
    network: pypsa.Network,
    overrides_path: str | Path | None,
    modelled_year: int,
) -> pypsa.Network:
    """Apply audited unit-level support labels after generator integration."""
    if not overrides_path:
        return network
    path = Path(overrides_path)
    if not path.exists():
        raise FileNotFoundError(f"Generator support override file not found: {path}")

    overrides = pd.read_csv(path)
    required = {"generator_name", "support_type"}
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(f"Support override file is missing columns: {sorted(missing)}")
    if "modelled_year" in overrides.columns:
        years = pd.to_numeric(overrides["modelled_year"], errors="coerce")
        overrides = overrides.loc[years.eq(int(modelled_year))].copy()
    if overrides.empty:
        logger.info("No generator support overrides apply in %s", modelled_year)
        return network
    if overrides["generator_name"].duplicated().any():
        duplicate_names = overrides.loc[
            overrides["generator_name"].duplicated(keep=False), "generator_name"
        ].tolist()
        raise ValueError(f"Duplicate generator support overrides: {duplicate_names}")

    absent = sorted(set(overrides["generator_name"]) - set(network.generators.index))
    if absent:
        raise ValueError(
            "Support overrides name generators absent from the integrated network: "
            + ", ".join(absent)
        )

    for row in overrides.to_dict("records"):
        name = row["generator_name"]
        support_type = str(row["support_type"]).strip()
        if support_type not in {"CfD", "ROC", "merchant"}:
            raise ValueError(f"Invalid support_type {support_type!r} for {name}")
        network.generators.loc[name, "support_type"] = support_type
        for attribute in ("ro_banding", "cfd_id", "support_evidence", "support_notes"):
            value = row.get(attribute)
            if pd.notna(value) and str(value).strip():
                network.generators.loc[name, attribute] = value
        # The original audited dispatchable-support file predates the separate
        # ROC-reference field.  Its ``cfd_id`` column therefore carries either
        # an LCCC contract id (for CfDs) or an Ofgem accreditation reference
        # (for ROCs).  Preserve the legacy column, but expose ROC evidence in
        # the unambiguous field consumed by the pre-solve station mapper.
        if support_type == "ROC":
            reference = row.get("ro_accreditation_reference", row.get("cfd_id"))
            if pd.notna(reference) and str(reference).strip():
                network.generators.loc[name, "ro_accreditation_reference"] = str(reference).strip()

    counts = overrides["support_type"].value_counts().to_dict()
    logger.info("Applied %d audited generator support overrides: %s", len(overrides), counts)
    return network


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
            'ccs gas': 'CCS Gas',
            'ccs biomass': 'CCS Biomass',
            'hydrogen': 'Hydrogen',
            'ocgt': 'OCGT',
            'open cycle gas turbine': 'OCGT',
            'single cycle': 'OCGT',          # DUKES 2022+ uses 'Single cycle' for OCGT
            # Coal
            'coal': 'coal',
            'coal (steam)': 'conventional_steam',
            'coal / oil': 'coal',            # Coal stations with oil backup/start-up
            # Nuclear - covers all DUKES format variants
            'nuclear': 'nuclear',
            'nuclear agr': 'nuclear',        # DUKES 2021 format: Advanced Gas-cooled Reactor
            'nuclear pwr': 'nuclear',        # DUKES 2021 format: Pressurised Water Reactor
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
        
        # Prefer the source station name.  REPD dispatchable-site extraction
        # supplies this for biomass, biogas, landfill and sewage-gas sites;
        # previously these rows became anonymous ``gen_<carrier>_<index>``
        # components, which prevented a traceable RO-station reconciliation.
        name_candidates = (
            row.get('station_name'),
            row.get('source_site_name'),
            row.get('site_name'),
        )
        gen_name = next(
            (str(value).strip() for value in name_candidates
             if pd.notna(value) and str(value).strip()),
            f"gen_{fuel_type}_{idx}",
        )

        # Source station names are not globally unique.  Preserve every REPD
        # site by making a collision unambiguous with its durable REPD ID,
        # rather than silently skipping capacity as the old integration did.
        data_source = str(row.get('data_source', '')).strip()
        if data_source.upper() == 'REPD' and gen_name in network.generators.index:
            repd_ref_id = row.get('repd_ref_id')
            suffix = str(repd_ref_id).strip() if pd.notna(repd_ref_id) else str(idx)
            candidate_name = f"{gen_name}__REPD_{suffix}"
            collision_number = 2
            while candidate_name in network.generators.index:
                candidate_name = f"{gen_name}__REPD_{suffix}_{collision_number}"
                collision_number += 1
            gen_name = candidate_name
        
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
            logger.warning(f"No bus, GSP or node for {gen_name} ({capacity_mw:,.1f} MW); placed at first bus {bus}")
        
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

        # Persist source identity and support-relevant REPD fields on the
        # component.  This gives downstream consumer-bill analysis a direct,
        # auditable link to the source site instead of recovering it from a
        # generated PyPSA name.
        for attribute in (
            'source_site_name', 'repd_ref_id', 'operational_date',
            'ro_banding', 'cfd_round', 'support_type',
            'cfd_id', 'cfd_technology', 'bmu_id', 'ro_accreditation_reference',
            'support_evidence', 'support_notes', 'ro_support_source',
            # Per-unit heat rate and its evidence (CP30 gas schedule, MR1);
            # apply_marginal_costs prefers unit_efficiency to the class value.
            'unit_efficiency', 'efficiency_source', 'gas_capacity_basis',
        ):
            value = row.get(attribute)
            if pd.notna(value) and str(value).strip():
                gen_attrs[attribute] = value
        # A dated station keeps its commissioning year, which the finance
        # register uses to vintage it.
        build_year = pd.to_numeric(row.get('build_year'), errors='coerce')
        if pd.notna(build_year) and build_year > 0:
            gen_attrs['build_year'] = int(build_year)
        
        # Add coordinates if available (needed for plotting and validation)
        if 'lat' in row and pd.notna(row['lat']):
            gen_attrs['lat'] = float(row['lat'])
            # Assign north/south region based on latitude threshold
            gen_attrs['region'] = 'north' if float(row['lat']) > 55.5 else 'south'
        if 'lon' in row and pd.notna(row['lon']):
            gen_attrs['lon'] = float(row['lon'])

        # Non-REPD duplicate names retain the existing conservative behaviour.
        # REPD collisions were disambiguated above with their reference ID.
        if gen_name in network.generators.index:
            logger.debug(f"Skipping duplicate generator: {gen_name}")
            continue

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

    # Configure spatial mapping constants from Snakemake config
    if snk and hasattr(snk, 'config'):
        configure_spatial_mapping(snk.config)

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
        # Where the numbers come from (CP30 targets: FES is only a spatial proxy) and how the
        # fleet is built (brownfield: CP30 totals placed through the named-station schedules).
        cp30_targets = target_source(scenario_config) == 'CP30'
        brownfield = uses_brownfield(scenario_config)
        spatial_proxy = scenario_config.get('fleet', {}).get('spatial_proxy', {}) if brownfield else {}
        fes_scenario = spatial_proxy.get('FES_scenario') if cp30_targets else scenario_config.get('FES_scenario', None)
        # Task Y6b: {} unless a FES long-range scenario sets fleet.fes_supply (today's behaviour otherwise).
        fes_thermal = fes_supply_part(scenario_config, 'thermal') if not brownfield else {}
        
        logger.info(f"Scenario: {scenario_name}, Modelled Year: {modelled_year}")
        if cp30_targets:
            logger.info(f"CP30 Pathway: {scenario_config.get('CP30_pathway')}")
            logger.info(f"Spatial FES proxy pathway: {fes_scenario}")
        elif fes_scenario:
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
        dukes_raw_df = pd.DataFrame()
        if hasattr(snk.input, 'dukes_data'):
            logger.info("Historical scenario detected - loading DUKES data")
            # Retain an unfiltered copy for the explicitly configured legacy-
            # hydro supplement. ``load_dukes_generators`` correctly removes
            # renewable fuels from the thermal stack, so passing its output to
            # that supplement would otherwise leave it no hydro stations.
            dukes_raw_df = pd.read_csv(snk.input.dukes_data)
            dukes_raw_df = apply_plant_phase_out_filtering(dukes_raw_df, modelled_year)
            dukes_df = load_dukes_generators(snk.input.dukes_data)
            # Apply phase-out filtering to remove closed/decommissioned plants
            # and adjust capacity for plants running at reduced output
            if len(dukes_df) > 0:
                dukes_df = apply_plant_phase_out_filtering(dukes_df, modelled_year)
        else:
            logger.info("Future scenario - DUKES data not applicable")
        
        # Load future projection data (FES or CP30 with explicit FES spatial proxy)
        fes_df = pd.DataFrame()
        if hasattr(snk.input, 'fes_data'):
            if len(dukes_df) == 0:
                if brownfield:
                    logger.info("Future CP30 scenario - loading CP30 capacities with FES spatial proxy")
                    fes_df = load_cp30_generators(
                        snk.input.cp30_es1,
                        snk.input.fes_data,
                        modelled_year,
                        scenario_config.get('CP30_pathway'),
                        fes_scenario,
                        nuclear_station_schedule=spatial_proxy.get('nuclear_station_schedule'),
                        ccs_biomass_station_schedule=spatial_proxy.get('ccs_biomass_station_schedule'),
                        biomass_station_schedule=spatial_proxy.get('biomass_station_schedule'),
                        gas_station_schedule=spatial_proxy.get('gas_station_schedule'),
                        gas_class_efficiency=scenario_config.get('marginal_costs', {})
                            .get('future_costs', {}).get('efficiency_defaults', {}),
                        committed_overlay=spatial_proxy.get('committed_overlay'),
                        ccs_gas_support_type=spatial_proxy.get('ccs_gas_support_type'),
                        committed_overlay_carve_out=spatial_proxy.get('committed_overlay_carve_out'),
                    )
                    # Named CfD units inside CP30's waste and Other Renewables
                    # proxies (K3, Energy Works Hull, AR5 geothermal), carved out
                    # of the proxy at the nearest bus to the site; totals unchanged.
                    cfd_station_schedule = spatial_proxy.get('cfd_station_schedule')
                    if cfd_station_schedule and len(fes_df):
                        fes_df = apply_cfd_station_schedule(
                            fes_df, cfd_station_schedule, network, logger,
                            protected_sources=(NUCLEAR_SCHEDULE_SOURCE, CCS_BIOMASS_SCHEDULE_SOURCE,
                                               BIOMASS_SCHEDULE_SOURCE, GAS_SCHEDULE_SOURCE, OVERLAY_SOURCE),
                            land_boundary_file=(getattr(snk.input, 'cp30_placement_land_boundary', None)
                                                or (((scenario_config.get('network', {}) or {}).get('spatial_mapping', {}) or {})
                                                    .get('placement_exclusions', {}) or {}).get('land_boundary_file')
                                                or "default"),
                        )
                    sources = fes_df.get('data_source', pd.Series(index=fes_df.index))
                    for schedule_source, label in ((NUCLEAR_SCHEDULE_SOURCE, 'Nuclear'),
                                                   (CCS_BIOMASS_SCHEDULE_SOURCE, 'CCS biomass'),
                                                   (BIOMASS_SCHEDULE_SOURCE, 'Biomass'),
                                                   (GAS_SCHEDULE_SOURCE, 'Gas'),
                                                   (OVERLAY_SOURCE, 'Committed overlay'),
                                                   (CFD_STATION_SCHEDULE_SOURCE, 'CfD station')):
                        scheduled = sources.eq(schedule_source)
                        if scheduled.any():
                            absent = sorted(set(fes_df.loc[scheduled, 'bus']) - set(network.buses.index))
                            if absent:
                                raise ValueError(
                                    f"{label} station schedule names buses absent from this network: {absent}"
                                )
                else:
                    logger.info("Future scenario - loading FES data as primary source")
                    fes_df = load_fes_generators(snk.input.fes_data, modelled_year, fes_scenario)
                    # Task Y6b (FES 2050 plan, year-generic in stage 5): the Gen_BB009 gas
                    # CCS split (or carrier remap) and the declared station schedule for Direct rows.
                    if fes_thermal:
                        es1_lines_path = (getattr(snk.input, 'fes_es1_lines', None)
                                          or fes_thermal.get('es1_lines_file'))
                        fes_df = apply_fes_thermal(
                            fes_df,
                            {**fes_thermal,
                             'station_schedule': getattr(snk.input, 'fes_station_schedule', None)
                             or fes_thermal.get('station_schedule')},
                            logger,
                            year=int(modelled_year),
                            es1_lines=pd.read_csv(es1_lines_path) if es1_lines_path else None,
                        )
                        scheduled = fes_df['data_source'].eq(schedule_data_source(modelled_year))
                        absent = sorted(set(fes_df.loc[scheduled, 'bus']) - set(network.buses.index))
                        if absent:
                            raise ValueError(f"FES {modelled_year} station schedule names buses absent from this network: {absent}")
            else:
                # Historical scenario - NO FES fallback
                logger.info("Historical scenario - NOT loading FES (using DUKES + REPD only)")
        else:
            if len(dukes_df) == 0:
                logger.warning("No thermal generator data available (no DUKES or FES/CP30)")
        
        # Load REPD dispatchable renewable thermal
        repd_df = pd.DataFrame()
        if brownfield:
            logger.info("CP30 scenario - skipping REPD thermal sites to avoid double-counting CP30 ES1 biomass/waste totals")
        elif fes_thermal:
            # The generic future path adds REPD bioenergy scaled to the FES totals ON TOP
            # of the FES bioenergy rows (merge_generator_data_sources), counting it twice.
            logger.info("FES long-range scenario - skipping REPD thermal sites: FES GSP rows carry bioenergy and waste")
        elif hasattr(snk.input, 'biomass_sites'):
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

        additional_thermal = load_historical_additional_thermal_sites(
            getattr(snk.input, 'historical_additional_thermal_sites', None),
            modelled_year,
        )
        if not additional_thermal.empty:
            repd_df = pd.concat([repd_df, additional_thermal], ignore_index=True, sort=False)

        subsidy_config = scenario_config.get('support', {}).get('labels', {})
        if subsidy_config.get('enabled', False):
            subsidy_kwargs = {
                'modelled_year': modelled_year,
                'roc_eligibility_years': subsidy_config.get('roc_eligibility_years', 20),
                'fes_cfd_fractions': subsidy_config.get('fes_cfd_fractions', {}),
                'logger': logger,
                'roc_closure_date': subsidy_config.get('roc_closure_date', '2017-04-01'),
                'default_roc_bandings': subsidy_config.get('default_roc_bandings', {}),
            }
            if len(repd_df) > 0:
                logger.info("Assigning CfD/ROC support to REPD dispatchable renewables")
                repd_df = _assign_subsidy_attributes(repd_df, **subsidy_kwargs)
            if len(fes_df) > 0:
                logger.info("Assigning projected CfD/merchant support to FES dispatchable generation")
                # Stations placed by the nuclear schedule already carry an
                # evidenced support label and contract round; the projected
                # split must not overwrite them.
                scheduled = fes_df.get('data_source', pd.Series(index=fes_df.index)).isin(
                    [NUCLEAR_SCHEDULE_SOURCE, CCS_BIOMASS_SCHEDULE_SOURCE, BIOMASS_SCHEDULE_SOURCE, OVERLAY_SOURCE,
                     CFD_STATION_SCHEDULE_SOURCE])
                if 'support_label_declared' in fes_df.columns:
                    scheduled |= fes_df['support_label_declared'].eq(True)
                projected = _assign_subsidy_attributes(fes_df.loc[~scheduled], **subsidy_kwargs)
                # CP30 distribution biomass and energy from waste are anonymous
                # proxies: give them the RO share the Ofgem register implies
                # (scripts/renewables/support_register_labels.py).
                registers_config = subsidy_config.get('support_registers', {}) or {}
                if brownfield and registers_config.get('enabled', False):
                    from scripts.renewables.support_register_labels import (
                        apply_ro_register_share_to_proxies,
                        load_ro_register,
                    )
                    proxy_share = registers_config.get('dispatchable_proxy_share', {}) or {}
                    if proxy_share.get('enabled', True):
                        ro_reference = registers_config.get('ro_reference_date') or f"{modelled_year}-04-01"
                        ro_register = load_ro_register(
                            snk.input.ro_register, ro_reference,
                            certificates_path=snk.input.ro_certificates,
                            default_bandings=subsidy_config.get('default_roc_bandings', {}),
                        )
                        schedule_names = fes_df.loc[scheduled].get(
                            'station_name', pd.Series(dtype=object)).dropna().astype(str).tolist()
                        projected, _ = apply_ro_register_share_to_proxies(
                            projected, ro_register,
                            carriers=tuple(proxy_share.get('carriers', ['biomass', 'waste_to_energy'])),
                            min_certificate_year=proxy_share.get('min_certificate_year', 2023),
                            carrier_col='fuel_type',
                            exclude_names=schedule_names,
                            logger=logger,
                        )
                fes_df = pd.concat([projected, fes_df.loc[scheduled]], ignore_index=True, sort=False)
        
        # =====================================================================
        # PART 2: MERGE DATA SOURCES WITH PRIORITY HANDLING
        # =====================================================================
        logger.info("-" * 80)
        logger.info("PART 2: MERGING DATA SOURCES")
        if len(dukes_df) > 0:
            logger.info("Historical scenario: DUKES > REPD (NO FES)")
        elif brownfield:
            logger.info("Future CP30 scenario: CP30 ES1 capacities + explicit FES spatial proxy")
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
        hydro_config = scenario_config.get('renewables', {}).get('hydro', {}).get('large_hydro', {})
        if len(dukes_raw_df) > 0 and hydro_config.get('dukes_legacy_supplement', True):
            thermal_data = add_missing_dukes_hydro(
                thermal_data,
                dukes_raw_df,
                snk.input.large_hydro_source_sites,
                logger,
            )
        thermal_data = add_roc_evidenced_dukes_hydro(
            thermal_data,
            network,
            getattr(snk.input, "dukes_roc_hydro_supplement", None),
            logger,
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
            
            # Residual Direct OCGT and gas-engine capacity follows the named gas
            # stations (co-location with existing gas connections) rather than
            # the best-connected buses.
            sources = thermal_data.get('data_source', pd.Series(index=thermal_data.index))
            gas_sites = thermal_data[sources.eq(GAS_SCHEDULE_SOURCE)]
            thermal_data = distribute_fes_generators_spatially(
                thermal_data, 
                network,
                repd_sites=repd_df if len(repd_df) > 0 else None,
                reference_sites={carrier: gas_sites for carrier in SCHEDULE_GAS_CARRIERS} if len(gas_sites) else None,
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
                    
                    # Float columns: an object-typed column breaks np.isnan in map_sites_to_buses.
                    thermal_data['x_coord'] = np.nan
                    thermal_data['y_coord'] = np.nan
                    
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
                        max_distance_km=_DIST_THERMAL_KM
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
                        max_distance_km=_DIST_THERMAL_KM
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
                        max_distance_km=_DIST_THERMAL_KM
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
            if fes_thermal:
                # The declared stations keep their evidenced buses (task Y6b).
                thermal_data = restore_declared_buses(thermal_data, logger)

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
        network = apply_generator_support_overrides(
            network,
            subsidy_config.get('dispatchable_support_overrides_file'),
            modelled_year,
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
                            
                            # Optional smoothing controlled by config
                            # "hourly" = raw signal, "daily" = daily average, "monthly" = legacy
                            smoothing = scenario_config.get('thermal', {}).get('nuclear_smoothing', 'hourly')
                            if smoothing == 'monthly':
                                monthly_avg = availability.resample('MS').mean()
                                availability = monthly_avg.reindex(availability.index, method='ffill')
                                logger.info(f"  Smoothed to monthly averages (legacy mode)")
                            elif smoothing == 'daily':
                                daily_avg = availability.resample('D').mean()
                                availability = daily_avg.reindex(availability.index, method='ffill')
                                logger.info(f"  Smoothed to daily averages")
                            else:
                                logger.info(f"  Using hourly availability (no smoothing)")
                            
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
        # STAGE 4d: THERMAL AVAILABILITY FACTORS
        # =====================================================================
        # Constrains biomass/biofuel generators to reflect real-world availability.
        # Biomass (Drax, Lynemouth) does not run at 100% CF due to pellet supply
        # logistics and planned maintenance. Small biofuel (biogas, LFG, sewage)
        # are CHP plants that follow heat demand, not wholesale price signals.
        avail_config = scenario_config.get('thermal', {}).get('availability_factors', {})
        if avail_config.get('enabled', False):
            stage_start = time.time()
            logger.info("-" * 80)
            logger.info("APPLYING THERMAL AVAILABILITY FACTORS")
            logger.info("-" * 80)

            avail_factors = avail_config.get('factors', {})
            check_min_output_within_availability(avail_config)
            corrections = 0
            for carrier, factor in avail_factors.items():
                factor = float(factor)
                if factor <= 0.0 or factor >= 1.0:
                    logger.warning(f"  Skipping {carrier}: factor {factor} out of range (0, 1)")
                    continue
                gens = network.generators[network.generators['carrier'] == carrier]
                if gens.empty:
                    continue
                for gen_name in gens.index:
                    # If already has a time-varying p_max_pu, scale it; else set static
                    if gen_name in network.generators_t.p_max_pu.columns:
                        network.generators_t.p_max_pu[gen_name] = (
                            network.generators_t.p_max_pu[gen_name] * factor
                        ).clip(upper=1.0)
                    else:
                        network.generators.at[gen_name, 'p_max_pu'] = factor
                    corrections += 1
                total_mw = gens['p_nom'].sum()
                logger.info(
                    f"  {carrier}: {len(gens)} generators, {total_mw:.0f} MW, "
                    f"p_max_pu capped at {factor:.2f} ({total_mw * factor:.0f} MW effective)"
                )
            logger.info(f"Applied availability factors to {corrections} generators")

            # ── Must-run minimum output (p_min_pu) ──────────────────────────
            # For carriers with must-run obligations (biomass CHP, ROC revenue
            # protection), set p_min_pu so they can't be turned below a floor.
            min_output = avail_config.get('min_output', {})
            min_corrections = 0
            for carrier, min_pu in min_output.items():
                min_pu = float(min_pu)
                if min_pu <= 0.0 or min_pu > 1.0:
                    logger.warning(f"  Skipping min_output {carrier}: {min_pu} out of range (0, 1]")
                    continue
                gens = network.generators[network.generators['carrier'] == carrier]
                if gens.empty:
                    continue
                for gen_name in gens.index:
                    network.generators.at[gen_name, 'p_min_pu'] = min_pu
                    min_corrections += 1
                total_mw = gens['p_nom'].sum()
                logger.info(
                    f"  {carrier}: {len(gens)} generators, {total_mw:.0f} MW, "
                    f"p_min_pu set to {min_pu:.2f} ({total_mw * min_pu:.0f} MW must-run)"
                )
            if min_corrections:
                logger.info(f"Applied must-run min_output to {min_corrections} generators")

            # Per-group values (ES1 calibration by connection), after the carrier ones.
            apply_thermal_group_overrides(network, avail_config.get('group_overrides') or [], logger)

            stage_times['4d. Thermal availability'] = time.time() - stage_start

        # Historical years: the 2030 placement audit's declared connection buses (stations and
        # offshore landings), after the name-prefix heuristic (scripts/generators/declared_buses.py).
        if target_source(scenario_config) == 'historical':
            from scripts.generators.declared_buses import apply_declared_buses
            apply_declared_buses(network, getattr(snk.input, 'historical_station_buses', None),
                                 getattr(snk.input, 'offshore_landing_crosswalk', None), logger)

        # Historical years: the whole large-hydro fleet (site units and the DUKES supplements added
        # above) takes ESPENI's metered hydro output as its daily inflow.
        from scripts.renewables.large_hydro_inflow import apply_historical_metered_hydro
        apply_historical_metered_hydro(
            network, scenario_config, scenario_config.get('renewables_year', modelled_year),
            getattr(snk.input, 'espeni_metered_hydro', None) or None, logger,
        )

        # Named station/month observations replace the broad carrier factor.
        # This is intentionally an explicit historical-validation input, not a
        # future-scenario assumption.
        monthly_availability = avail_config.get('generator_monthly_file')
        if monthly_availability and is_historical:
            apply_generator_monthly_availability(
                network, monthly_availability, modelled_year, logger
            )

        # One physical project may contain multiple CfD contracts with
        # different support start dates.  Split those contract shares before
        # marginal costs and dispatch are calculated, preserving any merchant
        # residual and the source p_max_pu profile.
        tranche_config = subsidy_config.get('historical_cfd_tranches', {})
        if subsidy_config.get('enabled', False) and tranche_config.get('enabled', False):
            split_historical_cfd_tranches(
                network,
                tranche_config['mapping_file'],
                tranche_config['actuals_file'],
                logger,
            )

        # Evidence-backed ROC labels must be available before marginal costs
        # are calculated, rather than being applied only to a post-solve audit
        # copy of the network.  The full calendar-year outturn supplies actual
        # bands, while the strict post-solve reconciliation remains the guard
        # against claiming an inferred label as a certificate observation.
        roc_label_config = subsidy_config.get('historical_roc_station_labels', {})
        roc_label_output = Path(str(snk.output.pre_solve_roc_mapping))
        roc_overlay_output = Path(str(snk.output.pre_solve_roc_overlay_actions))
        roc_capacity_output = Path(str(snk.output.pre_solve_roc_capacity_ledger))
        roc_physical_overlay_config = subsidy_config.get(
            'historical_roc_physical_overlay', {}
        )
        if is_historical and roc_physical_overlay_config.get('enabled', False):
            network, roc_overlay_report = apply_historical_roc_physical_overlay(
                network,
                snk.input.historical_roc_registry,
                snk.input.historical_roc_physical_overlay_actions,
                roc_physical_overlay_config,
                logger,
            )
            roc_overlay_output.parent.mkdir(parents=True, exist_ok=True)
            roc_overlay_report.to_csv(roc_overlay_output, index=False)
        else:
            roc_overlay_output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=[
                'Accreditation reference', 'capacity_action', 'execution_status',
                'source_generators', 'created_generators', 'ofgem_dnc_mw',
                'model_capacity_before_mw', 'model_capacity_after_mw',
                'capacity_added_mw', 'method',
            ]).to_csv(roc_overlay_output, index=False)
        if is_historical and roc_label_config.get('enabled', False):
            required_inputs = {
                'historical_roc_outturn': snk.input.historical_roc_outturn,
                'historical_roc_registry': snk.input.historical_roc_registry,
                'historical_roc_crosswalk': snk.input.historical_roc_crosswalk,
                'historical_roc_overrides': snk.input.historical_roc_overrides,
                'historical_roc_source_evidence': snk.input.historical_roc_source_evidence,
            }
            missing_inputs = [name for name, path in required_inputs.items() if not path]
            if missing_inputs:
                raise ValueError(
                    'historical_roc_station_labels is enabled but inputs are missing: '
                    f'{missing_inputs}'
                )
            outturn = pd.read_csv(str(snk.input.historical_roc_outturn))
            registry = pd.read_csv(str(snk.input.historical_roc_registry))
            crosswalk = pd.read_csv(str(snk.input.historical_roc_crosswalk))
            overrides = pd.read_csv(str(snk.input.historical_roc_overrides))
            source_evidence = pd.read_csv(str(snk.input.historical_roc_source_evidence))
            network, roc_mapping, roc_summary = apply_historical_dispatch_labels(
                network,
                outturn,
                registry,
                crosswalk,
                overrides,
                float(roc_label_config.get('buyout_price', 50.05)),
                source_evidence,
            )
            roc_label_output.parent.mkdir(parents=True, exist_ok=True)
            roc_mapping.to_csv(roc_label_output, index=False)
            network, roc_capacity_ledger = apply_ofgem_roc_capacity_overlay(
                network, outturn, registry, roc_mapping,
                snk.input.historical_roc_representation_assumptions, logger,
                residual_output_calibration_path=(
                    snk.input.historical_roc_residual_output_calibration
                    if snk.input.historical_roc_residual_output_calibration
                    else None
                ),
                modelled_year=modelled_year,
            )
            roc_capacity_output.parent.mkdir(parents=True, exist_ok=True)
            roc_capacity_ledger.to_csv(roc_capacity_output, index=False)
            logger.info(
                'Applied %d Ofgem-station-evidenced ROC labels before solve; '
                '%d inferred labels remain distinct.',
                roc_summary['labels_applied'],
                roc_summary['inherited_roc_labels_preserved'],
            )
        else:
            roc_label_output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=[
                'generator', 'mapping_status', 'mapping_method',
                'ro_accreditation_reference',
            ]).to_csv(roc_label_output, index=False)
            pd.DataFrame().to_csv(roc_capacity_output, index=False)

        anchor_config = scenario_config.get('thermal', {}).get('dispatch_anchors', {})
        if is_historical and anchor_config.get('enabled', False):
            apply_historical_dispatch_anchors(
                network, anchor_config, snk.input.espeni_dispatch, logger
            )

        # Planned maintenance and forced outage profiles are independent of unit
        # commitment: they only reduce available capacity at each snapshot.
        outage_config = scenario_config.get('thermal', {}).get('outages', {})
        if outage_config.get('enabled', False):
            stage_start = time.time()
            logger.info("Applying thermal planned/forced outage availability")
            apply_thermal_outage_availability(network, outage_config, logger)
            stage_times['4e. Thermal outage availability'] = time.time() - stage_start

        ramping_config = scenario_config.get('thermal', {}).get('ramping', {})
        if ramping_config.get('enabled', False):
            stage_start = time.time()
            logger.info("Applying LP-compatible thermal ramp limits")
            apply_thermal_ramp_limits(network, ramping_config, logger)
            stage_times['4f. Thermal ramp limits'] = time.time() - stage_start

        # Apply commissioning last, after carrier/monthly/outage availability,
        # must-run floors and CfD tranche splitting. This guarantees every
        # physical unit and contract tranche has p_min_pu <= p_max_pu.
        commissioning_config = scenario_config.get('fleet', {}).get('commissioning', {})
        if int(modelled_year) <= 2024 and commissioning_config.get(
            'profile_gate_within_solve_period', False
        ):
            apply_physical_commissioning_gates(network, logger)

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

