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
# SPATIAL MAPPING DEFAULTS
# Override at runtime by calling configure_spatial_mapping(snakemake.config).
# Values here match config/defaults/base/network.yaml → network.spatial_mapping section.
# =============================================================================
_OSGB36_MIN_VALUE: float = 1000.0      # x or y > this → OSGB36 (metres)
_WGS84_MAX_ABSOLUTE: float = 100.0     # x and y < this → WGS84 (degrees)
_SHETL_LAT_MIN: float = 57.0           # Northern Scotland boundary (°N)
_SPTL_LAT_MIN: float = 55.5            # Southern Scotland boundary (°N)
_DIST_SITE_TO_BUS_KM: float = 200.0    # General site → nearest bus
_DIST_GSP_TO_BUS_KM: float = 100.0     # GSP coordinate → nearest bus
_DIST_THERMAL_KM: float = 150.0        # Thermal gen → nearest bus
_FUZZY_THRESHOLD: float = 0.8          # Fuzzy string match score (0–1)
_FALLBACK_MAJOR_BUS_COUNT: int = 10    # Top-N buses used as fallback
_GEN_MIN_CHECK_MW: float = 50.0        # Skip bus check below this MW
_GEN_VERY_LARGE_MW: float = 300.0      # Must connect at ≥ _GEN_MIN_PREFERRED_KV
_GEN_LARGE_LOW_V_MW: float = 200.0     # Must NOT connect at ≤ _GEN_LOW_KV
_GEN_MEDIUM_MW: float = 100.0          # Needs transformer capacity check
_GEN_MAX_XFMR_RATIO: float = 0.8       # Max generator / transformer s_nom ratio
_GEN_MIN_PREFERRED_KV: float = 275.0   # Preferred minimum voltage (kV)
_GEN_HIGH_KV: float = 400.0            # 400 kV-only search threshold (kV)
_GEN_LOW_KV: float = 33.0              # Low-voltage threshold (kV)
_EARTH_RADIUS_KM: float = 6371.0       # Earth radius for haversine distance
_GEN_FALLBACK_HV_RADIUS_KM: float = 80.0  # Max km for post-agg nearest HV bus search


# =============================================================================
# EXPLICIT ONSHORE RENEWABLE CONNECTIONS
# Named onshore projects whose connection point is documented but which
# nearest-bus placement puts on a different, far weaker substation. Applied by
# apply_etys_bmu_mapping() to every matching row regardless of size or voltage
# heuristics; skipped when the bus is absent from the network (other network
# models, or years before the asset exists), so placement then falls back to
# the usual nearest-bus logic. Patterns match the lower-cased site name on
# whole words, so 'clyde wind farm' does not match 'Inverclyde Wind Farm' and
# 'whitelee' does not match 'Whitelees Hydrogen'. Order matters: the first
# matching pattern wins, so extensions precede their parent project.
# Evidence: GB_network.xlsx Dir_con_BMUs_to_node (official BMU -> node map)
# and the TEC register (data/generators/tec-register-*.csv).
# =============================================================================
ONSHORE_RENEWABLE_DIRECT_BUS: Dict[str, str] = {
    # T_CLDCW-1 (Clyde Central, the extension) -> CLYN3A-D; TEC "Clyde North
    # 275/33kV" 374.5 MW. The 275 kV collector CLYN2Q exports over 476 MVA.
    'clyde wind farm extension': 'CLYN2Q',
    # The original 350 MW farm feeds both collectors (T_CLDNW-1 -> CLYN,
    # T_CLDSW-1 -> CLYS; TEC Clyde South 128.8 MW). One REPD row cannot be
    # split, so it sits at Elvanfoot 275 kV where the CLYS cable lands; putting
    # all 350 MW behind either collector cable would invent a constraint.
    # (Was NLEI3A, a 2029 upgrade bus 5 km away behind one 90 MVA transformer.)
    'clyde wind farm': 'ELVA2R',
    # T_WHILW-2 -> WLEX3A-C (was MEAD1Q, 100 MVA, shared with Whitelee).
    'whitelee windfarm extension': 'WLEX2-',
    # T_WHILW-1 -> WLEE3A-C.
    'whitelee': 'WLEE2-',
    # T_KLGLW-1 -> KILG3A-C (was GLLU1Q, 109 MVA).
    'kilgallioch': 'KILG2-',
    # T_FALGW-1 -> FALL4- (was POGB3-, 57 MVA).
    'fallago rig': 'FALL4-',
    # T_EWHLW-1 -> EWEH3- (was ECCF1J, 18 MVA line to Chapelcross).
    'ewe hill': 'EWEH3-',
    # T_CRDEW-1/2 -> CRDY31 (Crossdykes 33 kV, tied to EWEH3-; was ECCF1J).
    'crossdykes': 'CRDY31',
    # T_MKHLW-1 -> MAHI3- (was TRLO3-, 25 MVA line).
    'mark hill': 'MAHI3-',
    # T_KPMRW-1 -> KYPE3- (120 MVA); the extension stays on the other
    # Kype Muir 33 kV section, KYPE3A (90 MVA).
    'kype muir wind farm extension': 'KYPE3A',
    'kype muir': 'KYPE3-',
    # T_CRYRW-2/-3 (and T_CRYRD-2) -> CRYR1-, the Crystal Rig 132 kV collector
    # behind the 240 MVA CRYR4- transformer (its 33 kV sections CRYR31/32 hang
    # off it). Nearest-bus put the phases on CRYR1A and Extension II on the
    # Neart na Gaoithe export substation NNGO21 (audit 27 Sep 2026).
    'crystal rig': 'CRYR1-',
    # T_ARCHW-1 -> AREC3A 50% / AREC3B 50% (2 x 90 MVA to AREC1-). One REPD
    # row cannot be split without doubling its register matches, so it sits on
    # AREC1-, the 132 kV bus both sections feed: the same export path for a
    # 120 MW farm (was AREC3-, 60 MVA transformer + 40 MVA line).
    'arecleoch': 'AREC1-',
    # Scottish 132 kV connection points from the TEC register (02 Sep 2025)
    # and GB_network.xlsx Dir_con_BMUs_to_node, for named farms nearest-bus
    # put behind the wrong 132 kV circuit (audit 27 Sep 2026, once SHE/SPT
    # 132 kV became constrained in the BM):
    # TEC "Heathland 132/33kV Substation" 80 MW -> HLND1- (was BLKL1-, which
    # then carried 394 MW on the 161 MVA Black Law-Wishaw circuit).
    'heathland wind farm': 'HLND1-',
    # TEC "Blacklaw Extension 132/33kV" 60 MW -> BLLX1- (was BLKL1-/BLKL3B).
    'black law extension': 'BLLX1-',
    # BMUs T_FARR-1 -> FAAR3J, T_FARR-2 -> FAAR3K: Farr splits across both
    # Farr 132 kV sections; one row sits on FAAR1R, leaving Moy and Glen
    # Kyllachy on FAAR1Q (all three were on FAAR1Q: 206 MW on 132 MVA).
    'farr wind farm': 'FAAR1R',
    # TEC "Maybole 132/33kV" / "Maybole GSP" (Assel Valley 25 MW, Kirk Hill
    # 20 MW); were on Hadyard Hill's own 140 MVA export circuit.
    'assel valley': 'MAYB1-',
    'kirk hill wind farm': 'MAYB1-',
    # TEC "Linnmill 132/33kV" 36 MW (was KYPE3A, the Kype Muir extension).
    'auchrobert': 'LINM1R',
    # BMUs T_MILWW-1 / T_MILWWD-1 -> MILW3-; TEC "Millennium GSP" (were on
    # CEAN1Q, Ceannacroc hydro).
    'millennium extension': 'MILW3-',
    'millennium windfarm': 'MILW3-',
    # TEC "Shin GSP" (Strath Oykel 74 MW, Meall Buidhe 40 MW); were on
    # CASS1Q behind the 111 MVA Cassley-Dalchork circuit.
    'strath oykel': 'SHIN1-',
    'meall buidhe': 'SHIN1-',
    # TEC "Achany Extension Wind Farm 132/33kV Substation": Achany (ACHE1-,
    # 323 MVA). Nearest-bus put it on CASS1Q; weak-bus relief used to move it
    # back only while Strath Oykel/Meall Buidhe overloaded Cassley.
    'achany extension': 'ACHE1-',
    # Strathy: TEC "Strathy North/South/Wood 132/33kV Substation"; ETYS STRW
    # (Strathy Wind; BMU T_STRNW-1 -> STRW31, the 33 kV side), STSO (Strathy
    # South) and STWO (Strathy Wood). All three were on ARMA1- (Armadale).
    'strathy north': 'STRW3-',
    'strathy south': 'STSO1B',
    'strathy wood': 'STWO3-',
    # Lewis projects: TEC "Stornoway 132/33kV" 180 MW and "Muaitheabhal
    # 132/33kV" 189 MW, both on the Lewis 132 kV network behind the Western
    # Isles HVDC (LEWI4R--FNLN4R), which ETYS builds from 2032. Inert while the
    # link is absent; the 2030 CP30 fleet excludes both projects outright
    # (data/renewables/cp30_named_fleet_exclusions.csv).
    'stornoway wind farm': 'LEWI1J',
    'muaitheabhal': 'LEWI1J',
    # Radial 132 kV pockets the network audit of 15 Oct 2026 traced to placement,
    # not ratings (TEC register 02 Sep 2025; GB_network.xlsx Dir_con_BMUs_to_node):
    # TEC "Glenglass 132kV Substation" 250 MW; BMUs T_SANQW-1/T_SANQD-1 -> GLGL3-.
    # Was SAKN1- (Sandy Knowe), 390 MW behind its 166 MVA circuit; nearest-bus matching still picks it,
    # so the TEC substation is pinned here. GLGL1Q is the main Glenglass 132 kV busbar. Since 27 Oct 2026
    # (F25.12) the Appendix B section GLGL1- (2025, the Sanquhar cable) is coupled to it as one node
    # (ETYS_upgrades: glenglass_2025_terminal_reconciliation; the GLGL1B change rows apply), so GLGL1Q
    # exports over SANQ1-, SAKN1-, BLAH1- and GHPS1-.
    'sanquhar 2': 'GLGL1Q',
    'sanquhar ii': 'GLGL1Q',
    # TEC "New Cumnock 275/132kV Substation" 212 MW. Was DESA1Q (Dersalloch's
    # 167 MVA radial), where only Dersalloch belongs.
    'north kyle': 'NECU1-',
    # TEC "Benbrack Wind Farm 132/33kV Substation": its own substation BENB1-
    # in the New Cumnock pocket. Was DESA1Q.
    'benbrack': 'BENB1-',
    # TEC "Cloich 132kV Substation" 200 MW -> CLOI1A. Was BHLA11 (Bhlaraidh).
    # Anchored: Coire Na Cloiche (Alness GSP) is a different farm.
    '^cloiche wind farm': 'CLOI1A',
    # Five farms (342 MW) were on OHLL1- behind its 180 MVA line to NECU1B.
    # TEC: Enoch Hill at its own "Enoch Hill 132/33kV" (ENHI1-, on NECU1C);
    # Pencloe ("Pencloe 132/33kV"), Windy Standard III, Greenburn ("Greenburn
    # Mixed Technology 132kV") and Over Hill have no own bus in the network, so
    # they are spread over the New Cumnock 132 kV sections: NECU1C (2 x 360 MVA
    # SGTs) and NECU1B (3 x 240 MVA SGTs).
    'enoch hill': 'ENHI1-',
    'pencloe': 'NECU1C',
    'windy standard iii': 'NECU1C',
    'greenburn wind park': 'NECU1B',
    'over hill wind farm': 'NECU1B',
}


def _explicit_onshore_connection(site_name: str, bus_index) -> Optional[Tuple[str, str]]:
    """Return (pattern, bus) of the first explicit connection matching site_name.

    Only buses present in ``bus_index`` are returned, so an explicit entry is
    inert on a network that does not contain its substation. A pattern starting
    with ``^`` must match at the start of the name ('^cloiche wind farm' is not
    'Coire Na Cloiche Wind Farm').
    """
    import re
    # Generator names in the network use underscores for spaces
    # ("Clyde_Wind_Farm"); the post-aggregation pass matches on those.
    name = str(site_name).lower().replace('_', ' ')
    for pattern, bus in ONSHORE_RENEWABLE_DIRECT_BUS.items():
        if pattern.startswith('^'):
            regex = rf"^{re.escape(pattern[1:])}(?![a-z])"
        else:
            regex = rf"(?<![a-z]){re.escape(pattern)}(?![a-z])"
        if re.search(regex, name):
            if bus in bus_index:
                return pattern, bus
            return None
    return None


def documented_connection_mask(sites_df: pd.DataFrame, bus_index, name_col: str = 'site_name') -> pd.Series:
    """True for rows whose name has a documented connection point on this network.

    The rows :data:`ONSHORE_RENEWABLE_DIRECT_BUS` places (TEC register): a
    placement rule that moves sites by geography (the GSP-node placement)
    leaves them where the register says they connect.
    """
    names = sites_df.get(name_col, pd.Series('', index=sites_df.index)).fillna('').astype(str)
    return names.map(lambda n: _explicit_onshore_connection(n, bus_index) is not None).astype(bool)


def configure_spatial_mapping(cfg: dict) -> None:
    """
    Update spatial mapping constants from a Snakemake config dictionary.

    Call once at script startup:  ``configure_spatial_mapping(snakemake.config)``

    Reads the ``network.spatial_mapping`` key and updates all module-level
    constants.  Unknown keys are silently ignored; missing keys keep defaults.
    """
    global \
        _OSGB36_MIN_VALUE, _WGS84_MAX_ABSOLUTE, \
        _SHETL_LAT_MIN, _SPTL_LAT_MIN, \
        _DIST_SITE_TO_BUS_KM, _DIST_GSP_TO_BUS_KM, _DIST_THERMAL_KM, \
        _FUZZY_THRESHOLD, _FALLBACK_MAJOR_BUS_COUNT, \
        _GEN_MIN_CHECK_MW, _GEN_VERY_LARGE_MW, _GEN_LARGE_LOW_V_MW, \
        _GEN_MEDIUM_MW, _GEN_MAX_XFMR_RATIO, \
        _GEN_MIN_PREFERRED_KV, _GEN_HIGH_KV, _GEN_LOW_KV, \
        _EARTH_RADIUS_KM, _GEN_FALLBACK_HV_RADIUS_KM
    sm = (cfg.get('network') or {}).get('spatial_mapping', {})
    if not sm:
        return

    coord = sm.get('coordinate_detection', {})
    _OSGB36_MIN_VALUE = float(coord.get('osgb36_min_value', _OSGB36_MIN_VALUE))
    _WGS84_MAX_ABSOLUTE = float(coord.get('wgs84_max_absolute', _WGS84_MAX_ABSOLUTE))

    tr = sm.get('transmission_regions', {})
    _SHETL_LAT_MIN = float(tr.get('shetl_lat_min', _SHETL_LAT_MIN))
    _SPTL_LAT_MIN = float(tr.get('sptl_lat_min', _SPTL_LAT_MIN))

    dist = sm.get('distance_km', {})
    _DIST_SITE_TO_BUS_KM = float(dist.get('site_to_bus', _DIST_SITE_TO_BUS_KM))
    _DIST_GSP_TO_BUS_KM = float(dist.get('gsp_to_bus', _DIST_GSP_TO_BUS_KM))
    _DIST_THERMAL_KM = float(dist.get('thermal_generator', _DIST_THERMAL_KM))

    _FUZZY_THRESHOLD = float(sm.get('fuzzy_threshold', _FUZZY_THRESHOLD))
    _FALLBACK_MAJOR_BUS_COUNT = int(sm.get('fallback_major_bus_count', _FALLBACK_MAJOR_BUS_COUNT))

    gv = sm.get('generator_voltage', {})
    _GEN_MIN_CHECK_MW = float(gv.get('min_check_mw', _GEN_MIN_CHECK_MW))
    _GEN_VERY_LARGE_MW = float(gv.get('very_large_mw', _GEN_VERY_LARGE_MW))
    _GEN_LARGE_LOW_V_MW = float(gv.get('large_low_voltage_mw', _GEN_LARGE_LOW_V_MW))
    _GEN_MEDIUM_MW = float(gv.get('medium_mw', _GEN_MEDIUM_MW))
    _GEN_MAX_XFMR_RATIO = float(gv.get('max_transformer_ratio', _GEN_MAX_XFMR_RATIO))
    _GEN_MIN_PREFERRED_KV = float(gv.get('min_preferred_kv', _GEN_MIN_PREFERRED_KV))
    _GEN_HIGH_KV = float(gv.get('high_voltage_kv', _GEN_HIGH_KV))
    _GEN_LOW_KV = float(gv.get('low_voltage_kv', _GEN_LOW_KV))

    _EARTH_RADIUS_KM = float(sm.get('earth_radius_km', _EARTH_RADIUS_KM))
    _GEN_FALLBACK_HV_RADIUS_KM = float(gv.get('fallback_hv_radius_km', _GEN_FALLBACK_HV_RADIUS_KM))

    logger.debug(
        f"Spatial mapping configured: dist_site={_DIST_SITE_TO_BUS_KM}km, "
        f"dist_gsp={_DIST_GSP_TO_BUS_KM}km, dist_thermal={_DIST_THERMAL_KM}km, "
        f"fuzzy={_FUZZY_THRESHOLD}, gen_check≥{_GEN_MIN_CHECK_MW}MW"
    )


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
    is_osgb36 = (x_max > _OSGB36_MIN_VALUE and y_max > _OSGB36_MIN_VALUE)

    # WGS84 detection: values in degrees (small numbers, UK range)
    is_wgs84 = (abs(x_min) < 180 and abs(x_max) < 180 and
                abs(y_min) < 90 and abs(y_max) < 90 and
                x_max < _WGS84_MAX_ABSOLUTE and y_max < _WGS84_MAX_ABSOLUTE)
    
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


def standardize_component_coordinates(network: pypsa.Network, components: list = None) -> Dict:
    """
    Ensure all network components have consistent WGS84 lon/lat coordinates.
    
    This is the **core coordinate standardization function** that should be called
    after generators, storage, or other components are added to the network.
    
    It ensures every component has ``lon`` and ``lat`` columns in WGS84 degrees
    so that spatial plots work uniformly. The logic for each component:
    
    1. If the component already has ``lon``/``lat`` and the values look like WGS84 → keep.
    2. If it has ``lon``/``lat`` but values look like OSGB36 meters → convert to WGS84.
    3. If it has ``longitude``/``latitude`` → normalize to ``lon``/``lat`` (converting
       from OSGB36 if needed).
    4. If it has no coordinates → derive from its assigned bus (prefer bus ``lon``/``lat``,
       fall back to converting bus ``x``/``y`` from OSGB36).
    
    Args:
        network: PyPSA network whose components should be standardized.
        components: List of component names to process.  Defaults to
                    ``['Generator', 'StorageUnit']``.
    
    Returns:
        Dict mapping component name → number of rows that were updated.
    """
    if components is None:
        components = ['Generator', 'StorageUnit']
    
    result = {}
    
    # Pre-compute bus lon/lat lookup (prefer existing WGS84 lon/lat on buses,
    # otherwise convert bus x/y from OSGB36)
    bus_lonlat = _get_bus_lonlat_lookup(network)
    
    for comp_name in components:
        df = network.df(comp_name)
        if df.empty:
            result[comp_name] = 0
            continue
        
        updated = 0
        
        # --- Detect existing coordinate columns ---------------------------------
        has_lon_lat = 'lon' in df.columns and 'lat' in df.columns
        has_longitude_latitude = 'longitude' in df.columns and 'latitude' in df.columns
        
        # Ensure lon/lat columns exist
        if not has_lon_lat:
            df['lon'] = np.nan
            df['lat'] = np.nan
        
        for idx in df.index:
            lon_val = df.at[idx, 'lon'] if has_lon_lat else np.nan
            lat_val = df.at[idx, 'lat'] if has_lon_lat else np.nan
            
            # Check if lon/lat are already valid WGS84
            if _is_valid_wgs84(lon_val, lat_val):
                continue  # already good
            
            # Try longitude/latitude columns (may be mislabeled OSGB36)
            if has_longitude_latitude:
                lng_val = df.at[idx, 'longitude']
                lt_val = df.at[idx, 'latitude']
                if pd.notna(lng_val) and pd.notna(lt_val):
                    if _is_valid_wgs84(lng_val, lt_val):
                        df.at[idx, 'lon'] = float(lng_val)
                        df.at[idx, 'lat'] = float(lt_val)
                        updated += 1
                        continue
                    elif _looks_like_osgb36(lng_val, lt_val):
                        conv_lon, conv_lat = osgb36_to_wgs84(float(lng_val), float(lt_val))
                        if _is_valid_wgs84(conv_lon, conv_lat):
                            df.at[idx, 'lon'] = conv_lon
                            df.at[idx, 'lat'] = conv_lat
                            updated += 1
                            continue
            
            # Check if lon/lat look like OSGB36 (mislabeled meters)
            if pd.notna(lon_val) and pd.notna(lat_val) and _looks_like_osgb36(lon_val, lat_val):
                conv_lon, conv_lat = osgb36_to_wgs84(float(lon_val), float(lat_val))
                if _is_valid_wgs84(conv_lon, conv_lat):
                    df.at[idx, 'lon'] = conv_lon
                    df.at[idx, 'lat'] = conv_lat
                    updated += 1
                    continue
            
            # Fall back to bus coordinates
            bus_name = df.at[idx, 'bus'] if 'bus' in df.columns else None
            if bus_name and bus_name in bus_lonlat:
                b_lon, b_lat = bus_lonlat[bus_name]
                if _is_valid_wgs84(b_lon, b_lat):
                    df.at[idx, 'lon'] = b_lon
                    df.at[idx, 'lat'] = b_lat
                    updated += 1
        
        result[comp_name] = updated
        if updated > 0:
            logger.info(f"Standardized coordinates for {updated}/{len(df)} {comp_name} entries")
    
    return result


# ---------------------------------------------------------------------------
# Internal helpers for coordinate classification
# ---------------------------------------------------------------------------

def _is_valid_wgs84(lon, lat) -> bool:
    """Return True if lon/lat look like valid WGS84 for GB."""
    if pd.isna(lon) or pd.isna(lat):
        return False
    return -12 < float(lon) < 5 and 49 < float(lat) < 62


def _looks_like_osgb36(x, y) -> bool:
    """Return True if x/y values are plausible OSGB36 meters."""
    if pd.isna(x) or pd.isna(y):
        return False
    return float(x) > 1000 and float(y) > 1000


def _get_bus_lonlat_lookup(network: pypsa.Network) -> Dict[str, Tuple[float, float]]:
    """
    Build a {bus_name: (lon_wgs84, lat_wgs84)} lookup from the network buses.
    
    Prefers bus ``lon``/``lat`` columns (WGS84).  Falls back to converting
    bus ``x``/``y`` from OSGB36 when ``lon``/``lat`` are missing.
    """
    lookup: Dict[str, Tuple[float, float]] = {}
    buses = network.buses
    
    has_lonlat = 'lon' in buses.columns and 'lat' in buses.columns
    has_xy = 'x' in buses.columns and 'y' in buses.columns
    
    for bus_name in buses.index:
        # Try lon/lat first
        if has_lonlat:
            lon_val = buses.at[bus_name, 'lon']
            lat_val = buses.at[bus_name, 'lat']
            if _is_valid_wgs84(lon_val, lat_val):
                lookup[bus_name] = (float(lon_val), float(lat_val))
                continue
        
        # Fall back to x/y → OSGB36 conversion
        if has_xy:
            x_val = buses.at[bus_name, 'x']
            y_val = buses.at[bus_name, 'y']
            if _looks_like_osgb36(x_val, y_val):
                conv_lon, conv_lat = osgb36_to_wgs84(float(x_val), float(y_val))
                if _is_valid_wgs84(conv_lon, conv_lat):
                    lookup[bus_name] = (conv_lon, conv_lat)
    
    return lookup


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


def fuzzy_match_string(query: str, candidates: list, threshold: float = _FUZZY_THRESHOLD) -> Tuple[Optional[str], float]:
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
    max_distance_km: float = _DIST_SITE_TO_BUS_KM
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
            earth_radius_km = _EARTH_RADIUS_KM
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
                    
                    # Build spatial index for buses (content-based cache key
                    # to avoid stale data from reused memory addresses)
                    bus_hash = hashlib.md5(bus_coords_xy.tobytes()).hexdigest()[:16]
                    cache_key = f"bus_spatial_{bus_hash}"
                    
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
                    else:
                        continue
                
                # Actually assign the bus and distance to the DataFrame
                bus_name = network.buses.index[bus_idx]
                sites_df.iloc[sites_df.index.get_loc(i), sites_df.columns.get_loc('bus')] = bus_name
                sites_df.iloc[sites_df.index.get_loc(i), sites_df.columns.get_loc('distance_km')] = dist_km
            
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
                match, score = fuzzy_match_string(str(site[region_col]).lower(), [name.lower() for name in bus_names], threshold=_FUZZY_THRESHOLD)
                if match and score > _FUZZY_THRESHOLD:
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
    gb_network_path: str = "data/network/ETYS/GB_network.xlsx",
    min_check_mw: float = None,
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
        min_check_mw: Override _GEN_MIN_CHECK_MW threshold. Use a lower value
            (e.g. 10) for post-aggregation checks where individual small farms
            have been summed into large aggregate generators.

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
    
    # Direct bus mappings for large stations with radial topology
    # These take precedence over prefix matching - maps station name to exact bus ID
    # Use this for stations where the terminal busbar (e.g. HUNN4A) has insufficient
    # export capacity compared to the main switching station (e.g. HUNE4-)
    LARGE_STATION_DIRECT_BUS = {
        # Thermal stations with radial topology issues
        'hunterston': 'HUNE4-',  # HUNN4A only has 847 MVA export, HUNE4- is main switching station
        # Offshore wind farms → onshore 400kV grid connection points
        # These override nearest-neighbour which places farms at sea coordinates
        # Traced from ETYS OFTO cable topology (Appendix B sheets B-2-1d, B-3-1d)
        'triton knoll': 'BICF41',       # Bicker Fen 400kV (via TKNO41/42)
        'beatrice': 'BLHI4-',           # Blackhillock 400kV (via BEAT4A/4B)
        'dudgeon': 'NECT41',            # Necton 400kV (via DUDO11/12 transformers)
        'race bank': 'WALP41',          # Walpole 400kV (via RACO41/42)
        'sheringham shoal': 'NECT41',   # Necton 400kV (via SSAW/SSBW)
        'humber gateway': 'HEDO21',     # Hedon 275kV (via HUMO11/12)
        'westermost rough': 'HEDO21',   # Hedon 275kV (via WERO21/22)
        'hornsea': 'KILL41',            # Killingholme 400kV (via HOWA41/HOWB41)
        'london array': 'CLEH41',       # Cleve Hill 400kV (via LONO4A/4B)
        'thanet': 'CLEH41',             # Cleve Hill 400kV (via THOW11/12)
        'east anglia': 'BRFO41',        # Bramford 400kV (via BRST41/42)
        'greater gabbard': 'LEIS4A',    # Leiston/Sizewell 400kV (via GGON)
        'galloper': 'LEIS4A',           # Leiston/Sizewell 400kV (via GALO)
        'rampion': 'BOLN41',            # Bolney 400kV (via TWNE41/42)
        'gwynt y mor': 'BODE41',        # Bodelwyddan 400kV (transformer)
        'walney extension': 'MIDL41',   # Middleton 400kV (via WOWE41/42)
        'west of duddon': 'HEYS41',     # Heysham 400kV (via WDSO41/42)
        'moray east': 'BLHI4-',         # Blackhillock 400kV (via MOWO41/42)
        'seagreen': 'TEAL2K',           # Tealing 275kV (via SGRX21/22/23)
        'neart na gaoithe': 'CRYR4-',   # Crystal Rig 400kV (via NNGO41/42)
        'dogger bank': 'CREB41',        # Creyke Beck 400kV
        'inner dowsing': 'WALP41',      # Walpole 400kV (via LINO)
        'lynn': 'WALP41',              # Walpole 400kV (via LINO)
        'lincs': 'WALP41',             # Walpole 400kV (via LINO41)
    }

    # Known power station name to BMU prefix mappings
    # These are hand-curated to match generator names to ETYS node prefixes
    STATION_TO_BMU_PREFIX = {
        'torness': 'TORN',
        'hunterston': 'HUNE',  # Changed: was 'HUNN' (terminal busbar) → 'HUNE' (main switching station)
        'hinkley': 'HINP',    # Fixed: was 'HINK' but network uses HINP* (Hinkley Point)
        'heysham': 'HEYS',
        'hartlepool': 'HATL', # Fixed: was 'HART' but network uses HATL* (Hartlepool)
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
        'little barford': 'WALP',  # Fixed: was 'LITB' (132kV only) - connects via Walpole 400kV
        # 'connah': 'CONN' REMOVED — CONN is Conon Bridge (Scotland), not Connah's Quay (Wales)
        # Connah's Quay is near BODE (Bodelwyddan) — nearest-neighbor handles this correctly
        'carrington': 'CARR',
        'rocksavage': 'ROCK',
        'immingham': 'IMMM',
        'enfield': 'ENFI',
        'medway': 'MEDW',
        'shoreham': 'BOLN',  # Fixed: was 'SHOR' (doesn't exist) - connects via Bolney 400kV
        'marchwood': 'MARC',
        'coryton': 'CORY',
        'teesside': 'TEES',
        'wilton': 'WILT',
        'great yarmouth': 'NORM',  # Added: connects at Norwich Main 400kV
        # Offshore wind — ETYS node prefixes (not Elexon BMU prefixes)
        # These are backup for Method 1; Method 0 (LARGE_STATION_DIRECT_BUS) handles most
        'hornsea': 'HOWA',       # Was HORN (BMU); ETYS onshore relay is HOWA/HOWB
        'beatrice': 'BEAT',      # BEAT4A/4B exist in ETYS
        'moray': 'MOWO',        # Was MORA (BMU); ETYS onshore is MOWO41/42
        'triton knoll': 'TKNO', # Was TRIK (BMU); ETYS onshore is TKNO41/42
        'east anglia': 'BRST',  # Was EANG (BMU); ETYS onshore is BRST21/22→BRFO41
        'london array': 'LONO', # Was LOAD (BMU); ETYS onshore is LONO4A/4B
        'gwynt y mor': 'BODE',  # Was GWYF (BMU); connects at Bodelwyddan 400kV
        'walney': 'WALN',       # WALN matches some ETYS buses
        'rampion': 'TWNE',      # Was RAMP (BMU); ETYS onshore is TWNE41/42
        'race bank': 'RACO',    # Was RACB (BMU); ETYS onshore is RACO41/42
        'dudgeon': 'DUDO',      # Was DUDG (BMU); ETYS onshore is DUDO11/12
        'greater gabbard': 'GGON', # Was GRGB (BMU); ETYS onshore is GGON11/12/13
        'thanet': 'THOW',       # Was THAN (BMU); ETYS onshore is THOW11/12
        'sheringham shoal': 'SSAW', # Was SHER (BMU); ETYS onshore is SSAW/SSBW
        'westermost rough': 'WERO', # Was WEST (BMU); ETYS onshore is WERO11
        'lincs': 'LINO',        # Was LINC (BMU); ETYS onshore is LINO13/14
        'humber gateway': 'HUMO',   # Was HUMG (BMU); ETYS onshore is HUMO11/12
        'robin rigg': 'RORE',   # Was ROBI (BMU); ETYS onshore is RORE11
        'ormonde': 'ORMO',
        'burbo bank': 'BIRK',   # Was BURB (BMU); connects at Birkenhead 275kV
        'barrow': 'BOSO',       # Was BARR (BMU); ETYS onshore is BOSO11
        'gunfleet': 'CLTO',     # Was GUNF (BMU); ETYS onshore is CLTO11
        'kentish flats': 'KENT',
        'seagreen': 'SGRO',     # Was SGRW (offshore); ETYS onshore is SGRO21/22
        'neart na gaoithe': 'NNGO',  # Was COCK; ETYS onshore is NNGO41/42
        'dogger bank': 'CREB',       # Connects at Creyke Beck (correct)
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
        # An explicit CP30 connection allocation records the intended physical
        # connection point.  Do not subsequently move it using generic BMU/
        # voltage heuristics merely because a sensitivity network contains a
        # nearby higher-voltage bus; this preserves paired topology tests.
        # The CP30 gas station schedule declares each station's bus with its
        # evidence (items MR1/MR2); it is not re-guessed here either.
        # Distribution-size plant placed at its own GSP's demand node, and FES
        # small-scale solar held there, connect inside the GSP (15 Oct 2026):
        # moving them to a same-site 400 kV busbar would bypass the SGTs.
        if str(row.get('data_source', '')).strip() in (
                'CP30_connection_allocation', 'CP30_gas_station_schedule',
                'CP30_biomass_station_schedule', 'CP30_ccs_biomass_station_schedule',
                'CP30_committed_overlay', 'CP30_distributed_gsp_node', 'CP30_small_scale_solar_gsp'):
            continue

        bus = row.get('bus')
        if pd.isna(bus) or bus not in bus_v_nom:
            continue
            
        v_nom = bus_v_nom[bus]
        capacity_mw = row.get('capacity_mw', row.get('p_nom', 0))
        
        # Get generator name for matching
        gen_name = str(row.get('station_name', row.get('site_name', row.get('name', row.get('Site Name', ''))))).lower()

        # Documented onshore connection points win over every heuristic below.
        explicit = _explicit_onshore_connection(gen_name, network.buses.index)
        if explicit is not None:
            pattern, new_bus = explicit
            if new_bus != bus:
                sites_df.at[idx, 'bus'] = new_bus
                corrections_made += 1
                corrections_log.append({
                    'generator': gen_name,
                    'capacity_mw': capacity_mw,
                    'old_bus': bus,
                    'old_v_nom': v_nom,
                    'old_xfmr_capacity': np.nan,
                    'new_bus': new_bus,
                    'new_v_nom': bus_v_nom.get(new_bus, 0),
                    'match_method': f"explicit_connection '{pattern}'",
                })
            continue

        # Only check generators above minimum threshold
        check_threshold = min_check_mw if min_check_mw is not None else _GEN_MIN_CHECK_MW
        if capacity_mw < check_threshold:
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

        # Hard rule: Very large generators (nuclear, large CCGT) should be at 400kV
        # They're too big for 275kV buses with limited line/transformer capacity
        if capacity_mw >= _GEN_VERY_LARGE_MW and v_nom < _GEN_HIGH_KV:
            needs_voltage_upgrade = True
        # Hard rule: Large generators at low voltage need to move
        elif capacity_mw >= _GEN_LARGE_LOW_V_MW and v_nom <= _GEN_LOW_KV:
            needs_voltage_upgrade = True
        # Soft rule: Generator exceeds max_transformer_ratio of export transformer capacity
        elif effective_export_capacity > 0 and capacity_mw / effective_export_capacity > _GEN_MAX_XFMR_RATIO:
            needs_voltage_upgrade = True
        # Soft rule: Medium generator with no transformer capacity at low voltage
        elif capacity_mw >= _GEN_MEDIUM_MW and v_nom < _GEN_MIN_PREFERRED_KV and effective_export_capacity == 0:
            needs_voltage_upgrade = True
        
        if not needs_voltage_upgrade:
            continue

        # Try to find correct bus using station name matching
        new_bus = None
        match_method = None

        # Method 0: Direct bus mapping for large stations with radial topology
        # These have been manually identified as needing the main switching station
        # rather than a terminal busbar with insufficient export capacity
        for station_pattern, direct_bus in LARGE_STATION_DIRECT_BUS.items():
            if station_pattern in gen_name:
                if direct_bus in network.buses.index and direct_bus != bus:
                    candidate_v = bus_v_nom.get(direct_bus, 0)
                    if candidate_v >= _GEN_MIN_PREFERRED_KV:
                        new_bus = direct_bus
                        match_method = 'direct_bus'
                        break

        # Method 1: Match generator name to known station prefixes
        if new_bus is None:
            for station_pattern, bmu_prefix in STATION_TO_BMU_PREFIX.items():
                if station_pattern in gen_name:
                    if bmu_prefix in bmu_mapping:
                        candidate_bus = bmu_mapping[bmu_prefix]
                        if candidate_bus in network.buses.index and candidate_bus != bus:
                            # Accept station-matched buses at or above preferred voltage
                            candidate_v = bus_v_nom.get(candidate_bus, 0)
                            if candidate_v >= _GEN_MIN_PREFERRED_KV:
                                new_bus = candidate_bus
                                match_method = 'station_name'
                                break
                    else:
                        # Fallback: look for any 275kV+ bus with this prefix (prefer 400kV)
                        potential_buses = [
                            b for b in network.buses.index
                            if b.startswith(bmu_prefix) and bus_v_nom.get(b, 0) >= _GEN_MIN_PREFERRED_KV
                        ]
                        if potential_buses:
                            # Sort by voltage (prefer higher voltage)
                            potential_buses.sort(key=lambda b: -bus_v_nom.get(b, 0))
                            new_bus = potential_buses[0]
                            match_method = 'station_prefix'
                            break
        
        # Method 2: Try same prefix as current bus but at 400kV
        if new_bus is None and v_nom < _GEN_HIGH_KV:
            site_prefix = bus[:4]
            potential_400kv_buses = [
                b for b in network.buses.index
                if b.startswith(site_prefix) and bus_v_nom.get(b, 0) == _GEN_HIGH_KV
            ]
            if potential_400kv_buses:
                new_bus = potential_400kv_buses[0]
                match_method = 'prefix_heuristic'

        # Method 3: Nearest high-voltage bus within radius (post-aggregation only)
        # This was previously removed for pre-aggregation because it moved generators
        # too far (e.g., Seagreen 75km to KINT4J). For post-aggregation, it's essential
        # because aggregated wind farms at 33kV/132kV radial buses have no prefix-matched
        # 400kV alternative. Guard rails: only fires when min_check_mw override is set
        # (post-aggregation), bus voltage is sub-preferred (<275kV), and Methods 0/1/2 all failed.
        is_post_aggregation = min_check_mw is not None
        if new_bus is None and is_post_aggregation and v_nom < _GEN_MIN_PREFERRED_KV:
            # Get coordinates of current bus (OSGB36 meters)
            if bus in network.buses.index:
                bus_x = network.buses.at[bus, 'x']
                bus_y = network.buses.at[bus, 'y']
                if pd.notna(bus_x) and pd.notna(bus_y):
                    # Find all buses with v_nom >= 275kV
                    hv_buses = network.buses[
                        network.buses['v_nom'] >= _GEN_MIN_PREFERRED_KV
                    ]
                    if not hv_buses.empty:
                        # Calculate Euclidean distance (OSGB36 is in meters)
                        hv_x = hv_buses['x'].values
                        hv_y = hv_buses['y'].values
                        distances_m = np.sqrt((hv_x - bus_x)**2 + (hv_y - bus_y)**2)
                        distances_km = distances_m / 1000.0

                        # Find nearest within radius
                        radius_km = _GEN_FALLBACK_HV_RADIUS_KM
                        within_radius = distances_km <= radius_km
                        if within_radius.any():
                            # Sort by (voltage descending, distance ascending)
                            candidates = pd.DataFrame({
                                'bus': hv_buses.index[within_radius],
                                'v_nom': hv_buses['v_nom'].values[within_radius],
                                'dist_km': distances_km[within_radius],
                            })
                            candidates = candidates.sort_values(
                                by=['v_nom', 'dist_km'],
                                ascending=[False, True]
                            )
                            new_bus = candidates.iloc[0]['bus']
                            match_method = f'nearest_hv_{candidates.iloc[0]["dist_km"]:.0f}km'
        
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
        elif needs_voltage_upgrade:
            logger.debug(
                f"  No 400kV bus found for {gen_name} ({capacity_mw:.0f} MW) at {bus} "
                f"({v_nom:.0f}kV) - left at current bus"
            )
    
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




# =============================================================================
# WEAK-BUS RELIEF (placement check on branch capacity)
# =============================================================================

# Data sources whose bus is an evidenced connection and is never moved here.
_PROTECTED_PLACEMENTS = (
    'CP30_connection_allocation', 'CP30_connection_allocation_direct',
    'CP30_ES5_regional_anchor_proxy', 'CP30_gas_station_schedule',
    'CP30_biomass_station_schedule', 'CP30_ccs_biomass_station_schedule',
    'CP30_committed_overlay',
)


#: ETYS rates bus sections and couplers at 9,999 MVA (the "Zero Length" rows
#: and the 0.01 % couplers); no real circuit is rated this high.
BUS_SECTION_MIN_RATING_MVA = 9998.0


def bus_section_groups(network: pypsa.Network,
                       min_rating_mva: float = BUS_SECTION_MIN_RATING_MVA) -> pd.Series:
    """Map each bus to its bus-section group (one label per electrical node).

    Buses of the same substation (four-character site code) and nominal
    voltage joined by a line rated at least ``min_rating_mva`` (ETYS bus
    sections and couplers, 9,999 MVA) form one node: ABNE1Q and ABNE1R
    (Abernethy 132 kV) are joined by such a coupler. Lines to other sites are
    never merged, even at 9,999 MVA (unrated offshore connections). The label
    is the group's first bus in sorted order; a bus with no coupler is its own
    group.
    """
    buses = network.buses
    parent = {bus: bus for bus in buses.index}

    def find(bus):
        while parent[bus] != bus:
            parent[bus] = parent[parent[bus]]
            bus = parent[bus]
        return bus

    lines = network.lines
    if len(lines):
        v_nom = buses['v_nom']
        for bus0, bus1, s_nom in zip(lines['bus0'], lines['bus1'], lines['s_nom']):
            if bus0 not in parent or bus1 not in parent or bus0 == bus1:
                continue
            if float(s_nom) < min_rating_mva or str(bus0)[:4] != str(bus1)[:4]:
                continue
            if not np.isclose(float(v_nom.at[bus0]), float(v_nom.at[bus1])):
                continue
            a, b = find(bus0), find(bus1)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return pd.Series({bus: find(bus) for bus in buses.index}, dtype=object).reindex(buses.index)


def bus_branch_capacity(network: pypsa.Network) -> pd.Series:
    """Total MVA/MW of lines, transformers and links leaving each bus's node.

    Buses joined by bus sections or couplers (:func:`bus_section_groups`) are
    one node: every bus of the group gets the rating of the branches leaving
    the group, and the coupler itself is not counted; a bus with no coupler is
    its own node, so this is the sum of its branches. Counting per busbar made
    the 9,999 MVA coupler ABNE1Q-ABNE1R look like a 10 GVA connection, so
    512 MW of storage behind 264 MVA was never relieved (network audit
    15 Oct 2026).
    """
    group = bus_section_groups(network)
    parts = []
    for component, rating in (('lines', 's_nom'), ('transformers', 's_nom'), ('links', 'p_nom')):
        frame = getattr(network, component)
        if not len(frame):
            continue
        g0 = frame['bus0'].map(group)
        g1 = frame['bus1'].map(group)
        leaving = g0.ne(g1)
        for g in (g0, g1):
            parts.append(pd.DataFrame({'group': g[leaving].values,
                                       'capacity': frame.loc[leaving, rating].values}))
    if not parts:
        return pd.Series(0.0, index=network.buses.index)
    by_group = pd.concat(parts).dropna(subset=['group']).groupby('group')['capacity'].sum()
    return group.map(by_group).fillna(0.0).astype(float)


#: x_pu given to an AC branch with no positive reactance (ETYS couplers are 0.01 % on 100 MVA, ~1e-6).
_COUPLER_X_PU = 1.0e-6


def branch_x_pu(buses: pd.DataFrame, lines: pd.DataFrame, transformers: pd.DataFrame) -> pd.Series:
    """Series reactance on a 1 MVA base, as PyPSA computes it: lines x / v_nom(bus0)^2, transformers x / s_nom.

    Indexed by branch name (lines then transformers). A branch with no positive x gets
    ``_COUPLER_X_PU`` (a bus coupler).
    """
    v_nom = pd.to_numeric(buses['v_nom'], errors='coerce')
    parts = []
    if len(lines):
        v = lines['bus0'].map(v_nom).astype(float)
        x = lines['x'] if 'x' in lines.columns else pd.Series(np.nan, index=lines.index)
        parts.append(pd.to_numeric(x, errors='coerce') / v.pow(2))
    if len(transformers):
        x = transformers['x'] if 'x' in transformers.columns else pd.Series(np.nan, index=transformers.index)
        parts.append(pd.to_numeric(x, errors='coerce') / pd.to_numeric(transformers['s_nom'], errors='coerce'))
    if not parts:
        return pd.Series(dtype=float)
    x = pd.concat(parts)
    return x.where(np.isfinite(x) & x.gt(0.0), _COUPLER_X_PU).astype(float)


def withdrawal_flow_shares(buses: pd.DataFrame, branches: pd.DataFrame, source_kv: float = 275.0) -> pd.DataFrame:
    """DC flow on each AC branch per MW withdrawn at each bus (branch x bus; + is bus0 -> bus1).

    The withdrawn MW is supplied by every bus at or above ``source_kv`` pro rata (a
    distributed source on the transmission grid), on the main AC island of
    ``branches`` (``bus0``, ``bus1``, ``x_pu``; lines and transformers). A relaxed
    rating does not change a branch's impedance, so the shares are the balancing
    solve's. Buses off the main island are absent.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import splu

    ac = branches[branches['bus0'].isin(buses.index) & branches['bus1'].isin(buses.index)
                  & branches['bus0'].ne(branches['bus1'])]
    if ac.empty:
        return pd.DataFrame()
    names = pd.Index(sorted(set(ac['bus0']) | set(ac['bus1'])))
    pos = pd.Series(np.arange(len(names)), index=names)
    i, j = pos[ac['bus0']].to_numpy(), pos[ac['bus1']].to_numpy()
    _, labels = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(names),) * 2), directed=False)
    main = np.bincount(labels).argmax()
    keep = labels[i] == main
    ac, i, j = ac[keep], i[keep], j[keep]
    island = names[labels == main]
    pos = pd.Series(np.arange(len(island)), index=island)
    i, j = pos[ac['bus0']].to_numpy(), pos[ac['bus1']].to_numpy()
    b = 1.0 / pd.to_numeric(ac['x_pu'], errors='coerce').fillna(_COUPLER_X_PU).clip(lower=_COUPLER_X_PU).to_numpy()
    n = len(island)
    lap = coo_matrix((np.concatenate([b, b, -b, -b]), (np.concatenate([i, j, i, j]), np.concatenate([i, j, j, i]))),
                     shape=(n, n)).tocsc()
    v = pd.to_numeric(buses['v_nom'], errors='coerce').reindex(island).fillna(0.0)
    sources = np.flatnonzero(v.to_numpy() >= float(source_kv))
    if not len(sources):
        sources = np.array([int(np.argmax(v.to_numpy()))])
    ref = int(sources[0])
    free = np.setdiff1d(np.arange(n), [ref])
    lu = splu(lap[free][:, free].tocsc())
    # One right-hand side per bus: the pro-rata source minus a unit withdrawal at the bus.
    rhs = np.zeros((n, n))
    rhs[sources, :] += 1.0 / len(sources)
    rhs[np.arange(n), np.arange(n)] -= 1.0
    theta = np.zeros((n, n))
    theta[free, :] = lu.solve(rhs[free, :])
    flows = b[:, None] * (theta[i, :] - theta[j, :])
    return pd.DataFrame(flows, index=ac.index, columns=island)


class NodeImportModel:
    """Demand-side headroom of each electrical node: its demand against what its circuits can bring in (16 Oct 2026).

    A node is a bus-section group (:func:`bus_section_groups`). Its pocket is the node
    and the lower-voltage busbars of its own substation (four-character site code) it
    reaches through the substation's own branches, which draw through it; its demand
    is the pocket's (supplied from the >= ``source_kv`` buses outside it), and its circuits are the
    branches with exactly one end in the pocket, which that demand must cross. Each
    circuit carries its DC share of the demand (:func:`withdrawal_flow_shares`), so
    the node's capacity is set by the first circuit to reach its rating: the demand
    at which ``max_i loading_i / rating_i`` is 1. GB-internal DC links
    (``kind == 'dc_link'``, controllable) add their rating on top. A 400 kV busbar fed
    only through its own site's transformers is judged on the site's circuits.

    Summing a node's ratings is optimistic twice over. Busbars without a coupler do not
    share: BAIN1Q takes 97% of its own withdrawals on its own 114 MVA circuit, and half
    of BAIN3-'s (the balancing stage held that circuit at its limit while BAIN1R's was
    half used). Parallel circuits to different substations share by impedance: a
    withdrawal at ABNE goes 68% via CHAR1- and 32% via AMUL1G, so 2 x 106 MVA delivers
    156 MW, not 212. And counting the 132/33 kV transformers into a node's own load
    busbar as capacity, as :func:`bus_branch_capacity` does for generation, passed
    390 MW at BUMU1Q behind 264 MVA.

    ``network`` needs ``buses`` (``v_nom``) and ``lines`` (bus sections);
    ``branches``: ``bus0``, ``bus1``, ``rating`` (MW the solve enforces, inf when
    relaxed), ``kind`` and, for lines and transformers, ``x_pu``. A node off the
    main AC island falls back to the sum of its circuits' ratings.
    """

    def __init__(self, network, branches: pd.DataFrame, source_kv: float = 275.0):
        self.source_kv = float(source_kv)
        self.group = bus_section_groups(network)
        v_nom = pd.to_numeric(network.buses['v_nom'], errors='coerce').fillna(0.0)
        self.node_v = v_nom.groupby(self.group).max()
        b = branches[branches['bus0'].isin(self.group.index) & branches['bus1'].isin(self.group.index)].copy()
        kind = b['kind'].astype(str) if 'kind' in b.columns else pd.Series('line', index=b.index)
        self.branches = b
        ac = b[kind.isin(['line', 'transformer'])]
        self.flows = withdrawal_flow_shares(network.buses, ac, source_kv)
        site = pd.Series(self.node_v.index.astype(str).str[:4], index=self.node_v.index)
        bus_site = pd.Series(self.group.index.astype(str).str[:4], index=self.group.index)
        bus_v = v_nom.reindex(self.group.index)
        self.circuits, self.behind, self.is_dc = {}, {}, kind.eq('dc_link')
        # A branch can cross a node's pocket only if it touches a bus of the node's substation.
        bus0, bus1 = b['bus0'].to_numpy(), b['bus1'].to_numpy()
        by_site = {}
        for k, (a0, a1) in enumerate(zip(b['bus0'].astype(str).str[:4], b['bus1'].astype(str).str[:4])):
            by_site.setdefault(a0, []).append(k)
            if a1 != a0:
                by_site.setdefault(a1, []).append(k)
        # Same-substation adjacency: a node's pocket is its buses and the lower-voltage busbars of its
        # substation it reaches through the substation's own branches (not a 132 kV group at the
        # same site that is fed from elsewhere).
        inside = {}
        for k, (a0, a1) in enumerate(zip(bus0, bus1)):
            if str(a0)[:4] == str(a1)[:4]:
                inside.setdefault(a0, []).append(a1)
                inside.setdefault(a1, []).append(a0)
        for node in self.node_v.index:
            own_buses = self.group.index[self.group.eq(node).to_numpy()]
            pocket, stack = set(own_buses), list(own_buses)
            while stack:
                for nb in inside.get(stack.pop(), ()):
                    if nb not in pocket and nb in bus_v.index and bus_v[nb] < self.node_v[node] - 1e-9:
                        pocket.add(nb)
                        stack.append(nb)
            self.behind[node] = own_buses.append(pd.Index(sorted(pocket - set(own_buses))))
            cut = [k for k in by_site.get(site[node], []) if (bus0[k] in pocket) != (bus1[k] in pocket)]
            self.circuits[node] = b.index[cut]
        self.members = pd.Series(self.group.index, index=self.group.values).groupby(level=0).agg(
            lambda s: '|'.join(sorted(map(str, s))))
        # Positions into the flow matrix, and which nodes each bus's demand counts towards.
        self._F = self.flows.to_numpy() if len(self.flows) else np.zeros((0, 0))
        row = pd.Series(np.arange(len(self.flows.index)), index=self.flows.index)
        col = pd.Series(np.arange(len(self.flows.columns)), index=self.flows.columns)
        src_v = v_nom.reindex(self.flows.columns).fillna(0.0).to_numpy()
        self._src = np.flatnonzero(src_v >= float(source_kv))
        if not len(self._src) and len(src_v):
            self._src = np.array([int(np.argmax(src_v))])
        self._src_rowsum = self._F[:, self._src].sum(axis=1) if len(self._src) else np.zeros(len(self._F))
        self._is_src = np.zeros(len(self.flows.columns), dtype=bool)
        self._is_src[self._src] = True
        self._pos = {}
        self.nodes_of = {}
        for node in self.node_v.index:
            circuits = [c for c in self.circuits[node] if c in row.index]
            behind = [bus for bus in self.behind[node] if bus in col.index]
            own = np.array([self.group.get(bus) == node for bus in behind], dtype=bool)
            self._pos[node] = (row[circuits].to_numpy(dtype=int), col[behind].to_numpy(dtype=int),
                               pd.Index(circuits), pd.Index(behind), own)
            for bus in self.behind[node]:
                self.nodes_of.setdefault(bus, []).append(node)

    def shares(self, node, demand: Optional[pd.Series] = None) -> pd.Series:
        """Each circuit's flow per MW of the node's demand (the demand-weighted withdrawal shares).

        With no demand behind the node, a withdrawal spread evenly over the node's own buses.
        """
        ci, bi, circuits, behind, own = self._pos[node]
        if not len(ci) or not len(bi):
            return pd.Series(dtype=float)
        if demand is None:
            w = own.astype(float) if own.any() else np.ones(len(bi))
        else:
            w = demand.reindex(behind).fillna(0.0).to_numpy(dtype=float)
            if w.sum() <= 1e-9:
                w = own.astype(float) if own.any() else np.ones(len(bi))
        w = w / w.sum()
        flow = self._F[np.ix_(ci, bi)] @ w + self._source_correction(node)
        return pd.Series(np.abs(flow), index=circuits)

    def _source_correction(self, node) -> np.ndarray:
        """Per-MW flow change on the node's circuits from supplying only from the sources outside its pocket.

        ``withdrawal_flow_shares`` supplies every MW from all >= source_kv buses pro rata. F[:, s] is the
        flow for (source - e_s), so moving the source by c (summing to zero) adds -F[:, S] @ c (exact, by
        linearity): the same for any withdrawal of one MW inside the pocket.
        """
        ci, bi, _, _, _ = self._pos[node]
        inner = bi[self._is_src[bi]]
        n_src, n_out = len(self._src), len(self._src) - len(inner)
        if not len(inner) or n_out <= 0:
            return np.zeros(len(ci))
        inner_sum = self._F[np.ix_(ci, inner)].sum(axis=1)
        outer_sum = self._src_rowsum[ci] - inner_sum
        return -(1.0 / n_out - 1.0 / n_src) * outer_sum + inner_sum / n_src

    def bus_flow(self, node, bus) -> pd.Series:
        """Flow on each of the node's circuits per MW withdrawn at ``bus`` (a bus of its pocket), absolute."""
        ci, bi, circuits, behind, _ = self._pos[node]
        if bus not in behind or not len(ci):
            return pd.Series(0.0, index=circuits)
        col = self.flows.columns.get_loc(bus)
        return pd.Series(np.abs(self._F[ci, col] + self._source_correction(node)), index=circuits)

    def evaluate(self, demand_mw: pd.Series, ratings: Optional[pd.Series] = None,
                 all_nodes: bool = False) -> pd.DataFrame:
        """One row per node with demand (every node with ``all_nodes``), indexed by node label.

        Columns ``buses``, ``v_nom``, ``own_mw`` (demand on the node's buses), ``demand_mw``
        (with its substation's lower-voltage busbars), ``import_mw`` (the sum of its
        circuits' finite ratings), ``capacity_mw`` (the demand its first-binding circuit allows,
        plus DC links), ``binding`` (that circuit), ``binding_share`` (its flow per MW of the
        node's demand) and ``ratio`` (demand / capacity).
        """
        rating = pd.to_numeric(self.branches['rating'] if ratings is None else ratings, errors='coerce') \
            .reindex(self.branches.index).fillna(0.0).astype(float)
        demand = pd.to_numeric(pd.Series(demand_mw, dtype=float), errors='coerce').fillna(0.0)
        demand = demand[demand.index.isin(self.group.index) & demand.gt(0.0)]
        own = demand.groupby(demand.index.map(self.group)).sum()
        nodes = self.node_v.index if all_nodes else pd.Index(sorted(
            {node for bus in demand.index for node in self.nodes_of.get(bus, [])}))
        rows = []
        for node in nodes:
            behind = self.behind[node]
            total = float(demand.reindex(behind).fillna(0.0).sum())
            if total <= 1e-9 and not all_nodes:
                continue
            circuits = self.circuits[node]
            dc = [c for c in circuits if self.is_dc.get(c, False)]
            ac = [c for c in circuits if c not in dc]
            dc_mw = float(rating[dc].sum()) if dc else 0.0
            share = self.shares(node, demand if total > 1e-9 else None)
            binding, bshare = None, np.nan
            if len(share):
                share = share[share > 1e-6]
                limit = rating[share.index] / share
                if len(limit) and np.isfinite(limit).any():
                    binding = limit.idxmin()
                    ac_mw, bshare = float(limit[binding]), float(share[binding])
                else:
                    ac_mw = np.inf
            else:
                ac_mw = float(rating[ac].sum()) if ac else 0.0
            capacity = ac_mw + dc_mw
            ratio = total / capacity if capacity > 0 else (np.inf if total > 1e-9 else 0.0)
            rows.append({'node': node, 'buses': self.members.get(node, node), 'v_nom': float(self.node_v[node]),
                         'own_mw': float(own.get(node, 0.0)), 'demand_mw': total,
                         'import_mw': float(rating[circuits][np.isfinite(rating[circuits])].sum()) if len(circuits) else 0.0,
                         'capacity_mw': capacity, 'binding': binding, 'binding_share': bshare, 'ratio': ratio})
        cols = ['buses', 'v_nom', 'own_mw', 'demand_mw', 'import_mw', 'capacity_mw', 'binding', 'binding_share',
                'ratio']
        return pd.DataFrame(rows).set_index('node')[cols] if rows else pd.DataFrame(columns=cols)


def node_import_headroom(network, branches: pd.DataFrame, demand_mw: pd.Series,
                         all_nodes: bool = False) -> pd.DataFrame:
    """:class:`NodeImportModel` evaluated once (build the model to evaluate several demands)."""
    return NodeImportModel(network, branches).evaluate(demand_mw, all_nodes=all_nodes)


def relieve_weak_bus_placements(
    sites_df: pd.DataFrame,
    network: pypsa.Network,
    max_loading: float = 1.5,
    search_radius_km: float = 40.0,
    min_target_voltage_kv: float = 132.0,
    capacity_col: str = 'capacity_mw',
    log: Optional[logging.Logger] = None,
    exclude_buses=None,
    site_bound_sources=(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Move sites off buses whose branches cannot carry what is placed on them.

    ``exclude_buses`` (e.g. :func:`generation_placement_exclusions`) are never
    chosen as a target. Rows whose ``data_source`` is in ``site_bound_sources``
    (plant placed on its own GSP's demand node, ``gsp_nodes.GSP_NODE_SOURCES``)
    move only to a bus of the same substation, never to a nearby one: they
    connect inside that GSP (16 Oct 2026).

    Nearest-bus placement puts a site (or a CP30 proxy tranche) on whichever bus
    is closest, often a 33 kV terminal busbar behind one small transformer:
    505 MW of CP30 proxy at POGB3- behind a 57 MVA line, 389 MW at RAVE3S behind
    32 MVA. Such a bus turns placement noise into curtailment and constraint
    cost. For every bus where the placed capacity exceeds
    ``max_loading`` times its total branch capacity, sites are moved, proxy
    tranches first and then named projects, largest first, until the bus is
    within the limit. Each goes to:

    1. a bus of the same substation (four-character site code) with headroom,
       highest branch capacity first; else
    2. the nearest bus at or above ``min_target_voltage_kv`` within
       ``search_radius_km`` with headroom,

    where headroom is ``max_loading * branch capacity - capacity already
    placed``. Offshore wind (landed by its own crosswalk), evidenced
    connections (``_PROTECTED_PLACEMENTS``) and explicit onshore connections
    are never moved but do count towards loading. Capacity, coordinates,
    profiles and CP30 regional labels are unchanged; only ``bus`` moves.

    Buses joined by bus sections or couplers are one node
    (:func:`bus_section_groups`): loading and limit are the node's, its
    capacity is the branches leaving it, and a site is never moved within its
    own node.

    Returns (sites_df with updated buses, log of moves).
    """
    log = log or logger
    result = sites_df.copy()
    if result.empty or 'bus' not in result.columns:
        return result, pd.DataFrame()
    capacity = bus_branch_capacity(network)
    group = bus_section_groups(network)
    buses = network.buses
    v_nom = buses['v_nom']
    xy = buses[['x', 'y']].astype(float)
    external = buses.index.astype(str).str.contains('External', case=False)
    if exclude_buses is not None and len(exclude_buses):
        external = external | buses.index.isin(list(exclude_buses))
    technology = result.get('technology', pd.Series('', index=result.index)).astype(str)
    placed = pd.to_numeric(result[capacity_col], errors='coerce').fillna(0.0)
    on_network = result['bus'].isin(buses.index)
    onshore = ~technology.eq('wind_offshore') & on_network
    # Every placed row loads its bus (an offshore farm landed on a target bus
    # uses its headroom); only onshore rows are candidates to move.
    row_group = result['bus'].map(group)
    node_capacity = capacity.groupby(group).first()
    load = placed[on_network].groupby(row_group[on_network]).sum()
    load = load.reindex(node_capacity.index, fill_value=0.0)
    limit = max_loading * node_capacity

    names = result.get('site_name', pd.Series('', index=result.index)).astype(str)
    source = result.get('data_source', pd.Series('', index=result.index)).fillna('').astype(str)
    origin = result.get('fleet_origin', pd.Series('', index=result.index)).fillna('').astype(str)
    movable = onshore & ~source.isin(_PROTECTED_PLACEMENTS) & placed.gt(0)
    movable &= ~names.map(lambda n: _explicit_onshore_connection(n, buses.index) is not None)
    site_bound = source.isin(list(site_bound_sources or ()))

    moves = []
    overloaded = load[load > limit + 1e-6].sort_values(ascending=False)
    for node in overloaded.index:
        rows = result.index[movable & row_group.eq(node)]
        order = sorted(rows, key=lambda i: (origin.at[i] == 'named', -placed.at[i]))
        for idx in order:
            if load.at[node] <= limit.at[node] + 1e-6:
                break
            bus = result.at[idx, 'bus']
            site = str(bus)[:4]
            need = placed.at[idx]
            headroom = (limit - load).reindex(group.to_numpy()).to_numpy()
            fits = (headroom >= need) & ~external & (group.to_numpy() != node)
            same_site = fits & (buses.index.str[:4] == site) & (capacity > capacity.at[bus])
            target, method = None, None
            if same_site.any():
                target = capacity[same_site].idxmax()
                method = 'same_substation'
            elif not site_bound.at[idx]:
                distance_km = np.hypot(xy['x'] - xy.at[bus, 'x'], xy['y'] - xy.at[bus, 'y']) / 1000.0
                near = fits & (v_nom >= min_target_voltage_kv) & (distance_km <= search_radius_km)
                if near.any():
                    target = distance_km[near].idxmin()
                    method = f'nearest_{distance_km[target]:.0f}km'
            if target is None:
                continue
            result.at[idx, 'bus'] = target
            load.at[node] -= need
            load.at[group.at[target]] += need
            moves.append({
                'site_name': names.at[idx], 'fleet_origin': origin.at[idx], 'capacity_mw': need,
                'from_bus': bus, 'from_branch_capacity': capacity.at[bus],
                'to_bus': target, 'to_branch_capacity': capacity.at[target], 'method': method,
            })
        if load.at[node] > limit.at[node] + 1e-6:
            log.warning(
                f"Weak-bus relief: {node} still carries {load.at[node]:.0f} MW against "
                f"{node_capacity.at[node]:.0f} MVA of branches (no movable site or no target with headroom)"
            )
    moves = pd.DataFrame(moves)
    if len(moves):
        log.info(
            f"Weak-bus relief: moved {len(moves)} sites ({moves['capacity_mw'].sum():,.0f} MW) off "
            f"{moves['from_bus'].nunique()} buses loaded above {max_loading:.1f}x their branch capacity"
        )
        summary = moves.groupby(['from_bus', 'to_bus', 'method']).agg(
            mw=('capacity_mw', 'sum'), sites=('site_name', 'count'),
            from_cap=('from_branch_capacity', 'first'), to_cap=('to_branch_capacity', 'first'))
        for (from_bus, to_bus, method), row in summary.sort_values('mw', ascending=False).iterrows():
            log.info(
                f"  {from_bus} ({row.from_cap:.0f} MVA) -> {to_bus} ({row.to_cap:.0f} MVA) "
                f"[{method}]: {row.mw:.1f} MW in {int(row.sites)} site(s)"
            )
    return result, moves


# =============================================================================
# EXPORT-CAPACITY RELIEF (proxy placement against the BM-constrained network)
# =============================================================================

BM_RELAXED_MVA = 9999.0
_INF_CAPACITY = 1.0e9


def bm_branch_capacities(network: pypsa.Network, transmission: Optional[dict] = None) -> pd.DataFrame:
    """Undirected branch capacities as the balancing mechanism sees them.

    Mirrors ``solve_network.apply_line_rating_overrides``: a line or
    transformer with an end below ``min_bm_constraint_voltage_kv`` is relaxed
    (infinite here), except Scottish (SHE/SPT) branches whose lower end is at
    or above ``bm_voltage_floor.scottish_min_constrained_kv`` when
    ``scottish_transmission_constrained``; ``line_rating_overrides`` win.
    Links contribute their p_nom. Returns columns bus0, bus1, capacity, kind.
    """
    from scripts.network_build.branch_attributes import is_scottish_branch
    transmission = transmission or {}
    min_v = float(transmission.get('min_bm_constraint_voltage_kv', 220) or 0)
    floor = transmission.get('bm_voltage_floor', {}) or {}
    scot_on = bool(floor.get('scottish_transmission_constrained', True))
    scot_min = float(floor.get('scottish_min_constrained_kv', 132.0))
    overrides = transmission.get('line_rating_overrides', {}) or {}
    v_nom = pd.to_numeric(network.buses['v_nom'], errors='coerce')
    parts = []
    for kind, table in (('line', network.lines), ('transformer', network.transformers)):
        if not len(table):
            continue
        lo = pd.concat([table['bus0'].map(v_nom), table['bus1'].map(v_nom)], axis=1).min(axis=1)
        cap = pd.to_numeric(table['s_nom'], errors='coerce').fillna(0.0).astype(float)
        if min_v > 0:
            relaxed = lo < min_v
            if scot_on:
                relaxed &= ~(is_scottish_branch(network, table) & (lo >= scot_min))
            cap = cap.where(~relaxed, _INF_CAPACITY)
        cap = cap.where(cap < BM_RELAXED_MVA - 1e-6, _INF_CAPACITY)
        for name, value in overrides.items():
            if name in cap.index:
                cap.at[name] = float(value)
        parts.append(pd.DataFrame({'bus0': table['bus0'], 'bus1': table['bus1'], 'capacity': cap, 'kind': kind}))
    if len(network.links):
        links = network.links
        parts.append(pd.DataFrame({'bus0': links['bus0'], 'bus1': links['bus1'],
                                   'capacity': pd.to_numeric(links['p_nom'], errors='coerce').fillna(0.0),
                                   'kind': 'link'}))
    return pd.concat(parts) if parts else pd.DataFrame(columns=['bus0', 'bus1', 'capacity', 'kind'])


def _minimum_bus_load(network: pypsa.Network) -> pd.Series:
    """Lowest hourly demand at each bus (MW, >= 0): the load an injection can always rely on."""
    loads = network.loads
    if not len(loads):
        return pd.Series(dtype=float)
    static = pd.to_numeric(loads.get('p_set', pd.Series(0.0, index=loads.index)), errors='coerce').fillna(0.0)
    low = static.copy()
    series = getattr(network.loads_t, 'p_set', pd.DataFrame())
    cols = [c for c in series.columns if c in loads.index]
    if cols:
        low.loc[cols] = series[cols].min()
    return low.clip(lower=0.0).groupby(loads['bus']).sum()


class _ExportModel:
    """Transport model: every bus injects its generation less its minimum load;
    power must reach the >= ``sink_kv`` grid over BM-constrained branches."""

    def __init__(self, network, capacities, sink_kv):
        import networkx as nx
        self.nx = nx
        graph = nx.DiGraph()
        for a, b, cap in zip(capacities['bus0'], capacities['bus1'], capacities['capacity']):
            if a == b or cap <= 0:
                continue
            for u, v in ((a, b), (b, a)):
                existing = graph.get_edge_data(u, v)
                graph.add_edge(u, v, capacity=(existing['capacity'] if existing else 0.0) + float(cap))
        v_nom = pd.to_numeric(network.buses['v_nom'], errors='coerce')
        for bus in network.buses.index[v_nom.ge(sink_kv).to_numpy()]:
            graph.add_edge(bus, '__T__')          # no capacity attribute: unbounded
        self.graph = graph
        self.min_load = _minimum_bus_load(network)

    def solve(self, injection: pd.Series):
        """(max flow, flow dict, per-bus unserved MW, source-side set)."""
        graph = self.graph
        graph.remove_node('__S__') if '__S__' in graph else None
        for bus, mw in injection.items():
            if mw > 1e-6 and bus in graph:
                graph.add_edge('__S__', bus, capacity=float(mw))
        if '__S__' not in graph:
            return 0.0, {}, pd.Series(dtype=float), set()
        value, flows = self.nx.maximum_flow(graph, '__S__', '__T__')
        unserved = {b: graph['__S__'][b]['capacity'] - flows['__S__'].get(b, 0.0) for b in graph['__S__']}
        # residual reachability from the source
        seen, stack = {'__S__'}, ['__S__']
        while stack:
            u = stack.pop()
            for v, data in graph[u].items():
                cap = data.get('capacity', np.inf)
                if v not in seen and flows[u].get(v, 0.0) < cap - 1e-6:
                    seen.add(v); stack.append(v)
            for v in graph.predecessors(u):
                if v not in seen and flows[v].get(u, 0.0) > 1e-6:
                    seen.add(v); stack.append(v)
        seen.discard('__S__')
        return value, flows, pd.Series(unserved), seen


def relieve_export_constrained_proxies(
    sites_df: pd.DataFrame,
    network: pypsa.Network,
    transmission: Optional[dict] = None,
    capacity_col: str = 'capacity_mw',
    carriers=('wind_onshore', 'solar_pv'),
    exclude_buses=None,
    search_radius_km: float = 60.0,
    min_target_voltage_kv: float = 132.0,
    sink_kv: float = 275.0,
    max_local_loading: float = 1.5,
    fixed_injection: Optional[pd.Series] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Move proxy increments out of pockets that cannot export their generation.

    ``relieve_weak_bus_placements`` compares a bus's generation with the sum
    of its own branches (1.5x). With Scottish 132 kV constrained in the BM that
    is too loose: CP30 proxies sat behind 112 MVA Brechin-Fiddes and the 240
    MVA Crystal Rig transformer and bound there (27 Sep 2026). This models
    the export path instead: every bus injects its placed generation less its
    minimum hourly load, power flows over the BM-rated branches
    (:func:`bm_branch_capacities`) to any bus at or above ``sink_kv``, and a
    maximum flow finds the pockets whose generation cannot get out. Proxy rows
    (``fleet_origin`` not ``named``, of ``carriers``, not an evidenced
    placement) in a pocket in deficit move - the smallest that covers the
    pocket's deficit, else the largest - to the nearest bus outside every
    deficit pocket that has export headroom for them (checked by re-solving),
    within ``search_radius_km``, at or above ``min_target_voltage_kv``, not in
    ``exclude_buses`` and within ``max_local_loading`` x its own branch sum.
    Named projects stay at their connection points and count towards the
    loading; a pocket with a deficit and no proxy is genuine and is reported.

    Returns (sites with updated buses, moves log, pocket table before/after).
    """
    log = log or logger
    result = sites_df.copy()
    if result.empty or 'bus' not in result.columns:
        return result, pd.DataFrame(), pd.DataFrame()
    buses = network.buses
    model = _ExportModel(network, bm_branch_capacities(network, transmission), sink_kv)
    xy = _bus_osgb_xy(buses)
    v_nom = pd.to_numeric(buses['v_nom'], errors='coerce')
    local_cap = bus_branch_capacity(network)
    if isinstance(exclude_buses, pd.Series):
        excluded = set(exclude_buses.index)
    else:
        excluded = set(exclude_buses) if exclude_buses is not None else set()
    excluded |= set(buses.index[buses.index.astype(str).str.contains('External', case=False)])
    placed = pd.to_numeric(result[capacity_col], errors='coerce').fillna(0.0)
    fixed = fixed_injection if fixed_injection is not None else pd.Series(dtype=float)
    technology = result.get('technology', pd.Series('', index=result.index)).astype(str)
    origin = result.get('fleet_origin', pd.Series('', index=result.index)).fillna('').astype(str)
    source = result.get('data_source', pd.Series('', index=result.index)).fillna('').astype(str)
    names = result.get('site_name', pd.Series('', index=result.index)).astype(str)
    movable = (~origin.eq('named') & technology.isin(list(carriers))
               & ~source.isin(_PROTECTED_PLACEMENTS) & placed.gt(0))

    def gen_by_bus():
        on = result['bus'].isin(buses.index)
        g = placed[on].groupby(result.loc[on, 'bus']).sum()
        return g.add(fixed, fill_value=0.0)

    def injection():
        net = gen_by_bus().sub(model.min_load, fill_value=0.0).clip(lower=0.0)
        return net[net > 1e-6]

    def pockets(unserved, reach):
        """Connected deficit pockets: {frozenset(buses): deficit MW}."""
        reach = [b for b in reach if b != '__T__']
        sub = model.graph.subgraph(reach).to_undirected()
        out = {}
        for comp in model.nx.connected_components(sub):
            deficit = float(sum(unserved.get(b, 0.0) for b in comp))
            if deficit > 1.0:
                out[frozenset(comp)] = deficit
        return out

    def pocket_table(found, label):
        rows = []
        gen = gen_by_bus()
        for comp, deficit in found.items():
            members = sorted(comp)
            g = float(gen.reindex(members).fillna(0.0).sum())
            rows.append({'state': label, 'buses': ';'.join(members[:12]) + (';...' if len(members) > 12 else ''),
                         'n_buses': len(members), 'generation_mw': g,
                         'min_load_mw': float(model.min_load.reindex(members).fillna(0.0).sum()),
                         'deficit_mw': deficit, 'export_mw': g - deficit,
                         'gen_to_export': g / max(g - deficit, 1e-6),
                         'proxy_mw': float(placed[movable & result['bus'].isin(members)].sum())})
        return rows

    _, _, unserved, reach = model.solve(injection())
    found = pockets(unserved, reach)
    report = pocket_table(found, 'before')
    moves, frozen = [], set()
    for _ in range(2000):
        live = {c: d for c, d in found.items() if not (c & frozen)}
        if not live:
            break
        comp, deficit = max(live.items(), key=lambda kv: kv[1])
        rows = result.index[movable & result['bus'].isin(comp)]
        if not len(rows):
            frozen |= comp
            continue
        sizes = placed[rows].sort_values()
        covering = sizes[sizes >= deficit]
        idx = covering.index[0] if len(covering) else sizes.index[-1]
        need = float(placed.at[idx])
        origin_bus = result.at[idx, 'bus']
        blocked = set().union(*found.keys()) | excluded
        dist = np.hypot(xy['x'] - xy.at[origin_bus, 'x'], xy['y'] - xy.at[origin_bus, 'y']) / 1000.0
        gen = gen_by_bus()
        load = gen.reindex(buses.index).fillna(0.0)
        ok = (dist <= search_radius_km) & (v_nom >= min_target_voltage_kv) \
            & ~buses.index.isin(list(blocked)) \
            & (load + need <= max_local_loading * local_cap + 1e-6)
        target = None
        for cand in dist[ok].sort_values().index:
            if v_nom.at[cand] >= sink_kv:
                target = cand
                break
            result.at[idx, 'bus'] = cand
            _, _, trial_unserved, trial_reach = model.solve(injection())
            result.at[idx, 'bus'] = origin_bus
            if not any(cand in c for c in pockets(trial_unserved, trial_reach)):
                target = cand
                break
        if target is None:
            log.warning("Export relief: no target within %.0f km for %s (%.0f MW) at %s",
                        search_radius_km, names.at[idx], need, origin_bus)
            movable.at[idx] = False
            _, _, unserved, reach = model.solve(injection())
            found = pockets(unserved, reach)
            continue
        result.at[idx, 'bus'] = target
        moves.append({'site_name': names.at[idx], 'carrier': technology.at[idx], 'capacity_mw': need,
                      'from_bus': origin_bus, 'to_bus': target, 'distance_km': float(dist[target]),
                      'target_kv': float(v_nom.at[target]), 'pocket_deficit_mw': deficit})
        _, _, unserved, reach = model.solve(injection())
        found = pockets(unserved, reach)
    report += pocket_table(found, 'after')
    moves = pd.DataFrame(moves)
    report = pd.DataFrame(report)
    if len(moves):
        log.info("Export relief: moved %d proxy rows (%.0f MW) out of %d export-constrained pockets",
                 len(moves), moves['capacity_mw'].sum(), moves['from_bus'].nunique())
        for row in moves.itertuples():
            log.info("  %s (%.0f MW): %s -> %s (%.0f kV, %.0f km)", row.site_name, row.capacity_mw,
                     row.from_bus, row.to_bus, row.target_kv, row.distance_km)
    remaining = report[report['state'].eq('after')] if len(report) else report
    for row in remaining.itertuples():
        log.info("  pocket still short (no proxy to move): %s gen %.0f MW, export %.0f MW",
                 row.buses, row.generation_mw, row.export_mw)
    return result, moves, report


# =============================================================================
# PLACEMENT EXCLUSIONS (buses onshore generation and storage must not use)
# =============================================================================

DEFAULT_LAND_BOUNDARY_FILE = "data/network/GSP/GSP_regions_27700_20250109.geojson"

# Island AC networks whose substations nearest-bus placement cannot find from
# coordinates. Orkney's 2029 network (ETYS 2024 Appendix B, B-1-1a: FINS
# "Orkney Finstown", ELIB "Ellibister switching station", plus the COST and
# HEST wind-farm substations off FINS1-) has no entry in
# substation_coordinates.csv, so its buses sit on the Caithness coast and
# Orkney's own generation went to the nearest coordinate: the HVDC hub
# "Central DC Substation" (audit 27 Sep 2026). Rows inside ``bbox`` (lon_min,
# lat_min, lon_max, lat_max; WGS84) that are on land (land boundary + 1 km)
# or at sea north of ``sea_lat_min`` go to ``hub``; the Pentland Firth south
# of it (MeyGen, landed in Caithness) is not Orkney. Rows elsewhere may not
# sit on the island's buses. Inert when the hub is absent (before 2029).
ISLAND_NETWORKS: Dict[str, dict] = {
    'Orkney': {
        'sites': ('FINS', 'ELIB', 'COST', 'HEST'),
        'hub': 'FINS1-',
        'bbox': (-3.50, 58.70, -2.30, 59.45),
        'sea_lat_min': 58.80,
    },
}
_LAND_GEOMETRY_CACHE: Dict[Tuple[str, float], object] = {}


def _land_geometry(path: str, buffer_m: float):
    """Union of the GSP-region polygons (EPSG:27700), buffered; cached per path."""
    key = (str(path), float(buffer_m))
    if key not in _LAND_GEOMETRY_CACHE:
        import geopandas as gpd
        from shapely.ops import unary_union
        frame = gpd.read_file(path)
        if frame.crs is not None and frame.crs.to_epsg() != 27700:
            frame = frame.to_crs(27700)
        _LAND_GEOMETRY_CACHE[key] = unary_union(frame.geometry.buffer(0)).buffer(float(buffer_m))
    return _LAND_GEOMETRY_CACHE[key]


def _bus_osgb_xy(buses: pd.DataFrame) -> pd.DataFrame:
    """Bus coordinates in OSGB36 metres, whichever system the network uses."""
    x = pd.to_numeric(buses['x'], errors='coerce')
    y = pd.to_numeric(buses['y'], errors='coerce')
    if (x.max() - x.min()) > _OSGB36_MIN_VALUE:
        return pd.DataFrame({'x': x, 'y': y}, index=buses.index)
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    ox, oy = transformer.transform(x.to_numpy(), y.to_numpy())
    return pd.DataFrame({'x': ox, 'y': oy}, index=buses.index)


def _ac_edges(network: pypsa.Network) -> pd.DataFrame:
    parts = [getattr(network, c)[['bus0', 'bus1']] for c in ('lines', 'transformers')
             if len(getattr(network, c))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['bus0', 'bus1'])


def dc_island_buses(network: pypsa.Network) -> pd.Index:
    """Buses of an AC island other than the main one that carries no load.

    Such an island reaches the grid only over a DC link: a converter or platform
    pair such as Sofia's SOFWD/SOFW61 (two placeholder transformers behind the
    Sofia HVDC poles), where nearest-bus matching had put 106 MW of onshore solar
    (item F3, 13 Oct 2026). Islands with demand (Shetland) are real systems and
    are kept; buses with no AC branch at all are ``dc_only``. The rule is
    ``add_storage.stranded_storage_buses``'s, for generation.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    buses = network.buses.index
    edges = _ac_edges(network)
    edges = edges[edges['bus0'].isin(buses) & edges['bus1'].isin(buses)]
    if edges.empty:
        return pd.Index([])
    position = pd.Series(np.arange(len(buses)), index=buses)
    i, j = position[edges['bus0']].to_numpy(), position[edges['bus1']].to_numpy()
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(buses), len(buses)))
    _, labels = connected_components(graph, directed=False)
    labels = pd.Series(labels, index=buses)
    has_branch = pd.Series(False, index=buses)
    has_branch[pd.unique(pd.concat([edges['bus0'], edges['bus1']]))] = True
    main = labels[has_branch].value_counts().idxmax()
    load_buses = network.loads['bus'].unique() if len(network.loads) else []
    loaded = set(labels.reindex(load_buses).dropna().astype(int))
    island = has_branch & labels.ne(main) & ~labels.isin(loaded)
    return buses[island.to_numpy()]


def coordinate_outlier_buses(network: pypsa.Network, outlier_km: float = 30.0,
                             cluster_km: float = 5.0) -> pd.Index:
    """Buses whose coordinates disagree with the rest of their substation.

    For each four-character site code with at least three buses, the largest
    group of buses within ``cluster_km`` of one another is taken as the site's
    location when it holds a strict majority; any bus of the site more than
    ``outlier_km`` from it is flagged (COST1- sat in the Borders while COST1A
    and COST3- are on Orkney). Sites with no majority (two pairs) are left
    alone because which pair is right cannot be told from the data.
    """
    buses = network.buses
    xy = _bus_osgb_xy(buses)
    site = pd.Series(buses.index.astype(str).str[:4], index=buses.index)
    flagged = []
    for _code, members in site.groupby(site).groups.items():
        members = list(members)
        if len(members) < 3:
            continue
        pts = xy.loc[members].to_numpy(dtype=float)
        if np.isnan(pts).any():
            continue
        dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1]) / 1000.0
        counts = (dist <= cluster_km).sum(axis=1)
        anchor = int(np.argmax(counts))
        if counts[anchor] * 2 <= len(members):
            continue
        far = dist[anchor] > outlier_km
        flagged.extend(np.asarray(members)[far].tolist())
    return pd.Index(flagged)


def generation_placement_exclusions(
    network: pypsa.Network,
    land_boundary_file: Optional[str] = DEFAULT_LAND_BOUNDARY_FILE,
    land_buffer_m: float = 2000.0,
    coordinate_outlier_km: float = 30.0,
    cluster_km: float = 5.0,
) -> pd.Series:
    """Buses nearest-bus placement must not give onshore generation or storage.

    Returns a Series indexed by bus with the reason, first reason winning:

    * ``dc_only``: no AC line or transformer (HVDC hubs such as "Central DC
      Substation", external interconnector buses);
    * ``dc_island``: an AC island other than the main one with no load, reached
      only over DC (:func:`dc_island_buses`; the Sofia converter pair, F3);
    * ``offshore``: outside the GSP land boundary plus ``land_buffer_m``
      (offshore wind platforms);
    * ``coordinate_outlier``: see :func:`coordinate_outlier_buses`;
    * ``offshore_export``: every onshore bus of a substation that is AC-connected
      to an offshore bus, carries no demand and reaches the rest of the onshore
      grid through at most one other substation (an OFTO onshore export
      substation such as NNGO, whose branches exist only to carry the wind
      farm to its grid connection).

    Offshore wind is landed by its own crosswalk and is not subject to this
    list; thermal plant and interconnectors are placed by declared schedules.
    Storage placement can reuse the same Series.
    """
    buses = network.buses
    reasons: Dict[str, str] = {}
    edges = _ac_edges(network)
    ac = set(edges['bus0']) | set(edges['bus1'])
    for bus in buses.index:
        if bus not in ac:
            reasons[bus] = 'dc_only'
    for bus in dc_island_buses(network):
        reasons.setdefault(bus, 'dc_island')

    offshore = set()
    if land_boundary_file:
        from pathlib import Path
        if Path(land_boundary_file).exists():
            from shapely import contains_xy
            land = _land_geometry(land_boundary_file, land_buffer_m)
            xy = _bus_osgb_xy(buses)
            valid = xy.notna().all(axis=1)
            inside = pd.Series(True, index=buses.index)
            inside[valid] = contains_xy(land, xy.loc[valid, 'x'].to_numpy(), xy.loc[valid, 'y'].to_numpy())
            offshore = set(buses.index[~inside])
            for bus in offshore:
                reasons.setdefault(bus, 'offshore')
        else:
            logger.warning("Placement exclusions: land boundary %s not found; offshore buses not detected",
                           land_boundary_file)

    outliers = set(coordinate_outlier_buses(network, coordinate_outlier_km, cluster_km))
    for bus in outliers:
        reasons.setdefault(bus, 'coordinate_outlier')

    if offshore:
        ignored = offshore | outliers
        site = pd.Series(buses.index.astype(str).str[:4], index=buses.index)
        loaded = set(network.loads['bus']) if len(network.loads) else set()
        adjacent: Dict[str, set] = {}
        for a, b in zip(edges['bus0'], edges['bus1']):
            adjacent.setdefault(a, set()).add(b)
            adjacent.setdefault(b, set()).add(a)
        landing_sites = {
            site[b] for b in buses.index
            if b not in ignored and any(n in offshore for n in adjacent.get(b, ()))
        }
        for code in sorted(landing_sites):
            members = [b for b in buses.index[site == code] if b not in ignored]
            if any(b in loaded for b in members):
                continue
            neighbour_sites = {
                site[n] for b in members for n in adjacent.get(b, ())
                if n not in ignored and site[n] != code
            }
            if len(neighbour_sites) <= 1:
                for b in members:
                    reasons.setdefault(b, 'offshore_export')
    return pd.Series(reasons, name='reason', dtype=object).sort_index()


def reassign_excluded_placements(
    sites_df: pd.DataFrame,
    network: pypsa.Network,
    exclusions,
    carrier_col: str = 'technology',
    carriers=None,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    tie_km: float = 0.5,
    protected_sources=_PROTECTED_PLACEMENTS,
    log: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Move rows off excluded buses to the nearest allowed bus.

    ``exclusions`` is a collection of bus names (or the Series from
    :func:`generation_placement_exclusions`). Rows of ``carriers`` (all rows
    when None) on an excluded bus move to the nearest bus not excluded, from
    the row's own coordinates (WGS84 ``lon``/``lat``) or, failing that, its
    current bus's; among buses within ``tie_km`` of the nearest (co-located
    busbars of one substation) the one with most branch capacity wins.
    Evidenced placements (``protected_sources`` in ``data_source``) and
    documented onshore connections are left alone. Only ``bus`` changes.

    Returns (sites with updated buses, log of moves).
    """
    log = log or logger
    result = sites_df.copy()
    if result.empty or 'bus' not in result.columns:
        return result, pd.DataFrame()
    excluded = set(exclusions.index if isinstance(exclusions, pd.Series) else exclusions)
    if not excluded:
        return result, pd.DataFrame()
    buses = network.buses
    xy = _bus_osgb_xy(buses)
    allowed = buses.index[~buses.index.isin(list(excluded)) & xy.notna().all(axis=1).to_numpy()]
    if len(allowed) == 0:
        return result, pd.DataFrame()
    capacity = bus_branch_capacity(network)
    carrier = result.get(carrier_col, pd.Series('', index=result.index)).astype(str)
    source = result.get('data_source', pd.Series('', index=result.index)).fillna('').astype(str)
    names = result.get('site_name', pd.Series('', index=result.index)).astype(str)
    mask = result['bus'].isin(list(excluded)) & ~source.isin(list(protected_sources))
    if carriers is not None:
        mask &= carrier.isin(list(carriers))
    mask &= ~names.map(lambda n: _explicit_onshore_connection(n, buses.index) is not None)
    if not mask.any():
        return result, pd.DataFrame()
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    ax = xy.loc[allowed, 'x'].to_numpy(dtype=float)
    ay = xy.loc[allowed, 'y'].to_numpy(dtype=float)
    lons = pd.to_numeric(result[lon_col], errors='coerce') if lon_col in result.columns else pd.Series(np.nan, index=result.index)
    lats = pd.to_numeric(result[lat_col], errors='coerce') if lat_col in result.columns else pd.Series(np.nan, index=result.index)
    moves = []
    for idx in result.index[mask]:
        old = result.at[idx, 'bus']
        lon, lat = lons.at[idx], lats.at[idx]
        if pd.notna(lon) and pd.notna(lat) and -12 < lon < 4 and 48 < lat < 62:
            px, py = transformer.transform(float(lon), float(lat))
        else:
            px, py = float(xy.at[old, 'x']), float(xy.at[old, 'y'])
        dist = np.hypot(ax - px, ay - py) / 1000.0
        tied = np.flatnonzero(dist <= dist.min() + tie_km)
        choice = max(tied, key=lambda i: (capacity.get(allowed[i], 0.0), -dist[i]))
        target = allowed[choice]
        result.at[idx, 'bus'] = target
        moves.append({'site_name': names.at[idx], 'carrier': carrier.at[idx],
                      'from_bus': old, 'to_bus': target, 'distance_km': float(dist[choice])})
    moves = pd.DataFrame(moves)
    if len(moves):
        log.info("Placement exclusions: moved %d rows off %d excluded buses",
                 len(moves), moves['from_bus'].nunique())
        for (frm, to), grp in moves.groupby(['from_bus', 'to_bus']):
            log.info("  %s -> %s: %d row(s)", frm, to, len(grp))
    return result, moves


def apply_island_networks(
    sites_df: pd.DataFrame,
    network: pypsa.Network,
    carrier_col: str = 'technology',
    carriers=None,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    exclusions=None,
    islands: Optional[Dict[str, dict]] = None,
    protected_sources=_PROTECTED_PLACEMENTS,
    log: Optional[logging.Logger] = None,
    land_boundary_file: Optional[str] = DEFAULT_LAND_BOUNDARY_FILE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Put island sites on their island's AC hub and keep mainland sites off it.

    See ``ISLAND_NETWORKS``. A row of ``carriers`` whose WGS84 coordinates lie
    inside an island's bounding box, on land or at sea north of the island's
    ``sea_lat_min``, and whose bus is not one of the island's substations moves
    to the island hub; any other row sitting on an island substation moves to
    the nearest other bus not in ``exclusions``. Evidenced placements are left
    alone. Without a land boundary every point in the box counts as the island.
    Returns (sites, log of moves).
    """
    log = log or logger
    result = sites_df.copy()
    if result.empty or 'bus' not in result.columns:
        return result, pd.DataFrame()
    islands = ISLAND_NETWORKS if islands is None else islands
    carrier = result.get(carrier_col, pd.Series('', index=result.index)).astype(str)
    source = result.get('data_source', pd.Series('', index=result.index)).fillna('').astype(str)
    eligible = ~source.isin(list(protected_sources))
    if carriers is not None:
        eligible &= carrier.isin(list(carriers))
    lons = pd.to_numeric(result[lon_col], errors='coerce') if lon_col in result.columns else pd.Series(np.nan, index=result.index)
    lats = pd.to_numeric(result[lat_col], errors='coerce') if lat_col in result.columns else pd.Series(np.nan, index=result.index)
    site_code = result['bus'].astype(str).str[:4]
    excluded = set(exclusions.index) if isinstance(exclusions, pd.Series) else set(exclusions if exclusions is not None else [])
    moves = []
    for island, spec in islands.items():
        hub = spec['hub']
        if hub not in network.buses.index:
            continue
        codes = set(spec['sites'])
        lon_min, lat_min, lon_max, lat_max = spec['bbox']
        inside = lons.between(lon_min, lon_max) & lats.between(lat_min, lat_max)
        from pathlib import Path
        if inside.any() and land_boundary_file and Path(land_boundary_file).exists():
            from pyproj import Transformer
            from shapely import contains_xy
            land = _land_geometry(land_boundary_file, 1000.0)
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            px, py = transformer.transform(lons[inside].to_numpy(), lats[inside].to_numpy())
            on_land = pd.Series(contains_xy(land, px, py), index=lons.index[inside])
            north = lats[inside].ge(float(spec.get('sea_lat_min', lat_min)))
            inside.loc[inside] = on_land | north
        on_island_bus = site_code.isin(codes)
        to_hub = eligible & inside & ~on_island_bus
        for idx in result.index[to_hub]:
            moves.append({'site_name': result.at[idx, 'site_name'] if 'site_name' in result.columns else idx,
                          'carrier': carrier.at[idx], 'from_bus': result.at[idx, 'bus'], 'to_bus': hub,
                          'rule': f'{island}_hub'})
            result.at[idx, 'bus'] = hub
        off_island = eligible & ~inside & on_island_bus & lons.notna()
        if off_island.any():
            island_buses = network.buses.index[network.buses.index.astype(str).str[:4].isin(codes)]
            moved, log_off = reassign_excluded_placements(
                result.loc[off_island], network, set(island_buses) | excluded,
                carrier_col=carrier_col, lon_col=lon_col, lat_col=lat_col,
                protected_sources=protected_sources, log=log,
            )
            result.loc[moved.index, 'bus'] = moved['bus']
            if len(log_off):
                moves.extend(log_off.assign(rule=f'off_{island}').to_dict('records'))
    moves = pd.DataFrame(moves)
    if len(moves):
        for (rule, frm, to), grp in moves.groupby(['rule', 'from_bus', 'to_bus']):
            log.info("Island networks [%s]: %s -> %s: %d row(s)", rule, frm, to, len(grp))
    return result, moves
