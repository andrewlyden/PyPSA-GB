"""
Electric Vehicle Charging Demand Disaggregation and Flexibility

This script disaggregates EV charging electricity demand from the total demand
and optionally adds smart charging flexibility (GO, INT, or V2G modes).

Flexibility Modes:
- GO: Octopus Go style - fixed 4-hour night charging window (00:00-04:00)
- INT: Intelligent tariff - fully optimizable smart charging
- V2G: Vehicle-to-Grid - bidirectional charging/discharging

Without flexibility, EVs are modeled as simple Load components with evening peak.
With flexibility enabled, Store and Link components are added for optimization.
"""

import pandas as pd
import pypsa
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network
from scripts.demand.load import load_spatial_mapping_data, aggregate_demand_to_network_topology

# ──────────────────────────────────────────────────────────────────────────────
# Spatial Allocation Functions
# ──────────────────────────────────────────────────────────────────────────────

def allocate_proportional(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
    """Allocate EV demand proportionally to existing base demand."""
    logger.info("Allocating proportionally to base demand")

    # Get base demand per load
    if len(base_network.loads_t.p_set) > 0:
        load_demand = base_network.loads_t.p_set.sum(axis=0)
    else:
        load_demand = base_network.loads.p_set

    # Map load demand to buses (aggregate by bus)
    load_to_bus = base_network.loads["bus"]
    valid_loads = [l for l in load_demand.index if l in load_to_bus.index]
    if not valid_loads:
        raise ValueError("No valid loads for allocation")

    load_demand = load_demand.loc[valid_loads]
    bus_mapping = load_to_bus.loc[valid_loads]
    bus_demand = load_demand.groupby(bus_mapping).sum()

    total_base = bus_demand.sum()
    if total_base == 0:
        raise ValueError("Zero base demand")

    bus_fractions = bus_demand / total_base
    ev_allocation = bus_fractions * total_gwh

    logger.info(f"Allocated {ev_allocation.sum():.1f} GWh across {len(ev_allocation)} buses")
    return ev_allocation


def allocate_uniform(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
    """Allocate EV demand uniformly across all buses."""
    logger.info("Allocating uniformly across all buses")
    
    n_buses = len(base_network.buses)
    ev_per_bus = total_gwh / n_buses
    ev_allocation = pd.Series(ev_per_bus, index=base_network.buses.index)
    
    logger.info(f"Allocated {ev_allocation.sum():.1f} GWh uniformly")
    return ev_allocation


def allocate_urban_weighted(total_gwh: float, base_network: pypsa.Network, logger,
                            urban_weight: float = 2.0) -> pd.Series:
    """Allocate EV demand weighted towards urban areas (high demand buses)."""
    logger.info("Allocating with urban weighting")

    # Get base demand per load
    if len(base_network.loads_t.p_set) > 0:
        load_demand = base_network.loads_t.p_set.sum(axis=0)
    else:
        load_demand = base_network.loads.p_set

    # Map load demand to buses (aggregate by bus)
    load_to_bus = base_network.loads["bus"]
    valid_loads = [l for l in load_demand.index if l in load_to_bus.index]
    if not valid_loads:
        logger.warning("No valid loads for urban weighting, falling back to proportional")
        return allocate_proportional(total_gwh, base_network, logger)

    load_demand = load_demand.loc[valid_loads]
    bus_mapping = load_to_bus.loc[valid_loads]
    bus_demand = load_demand.groupby(bus_mapping).sum()

    # EVs concentrated in urban areas - use higher exponent
    weighted_demand = bus_demand ** urban_weight

    total_weighted = weighted_demand.sum()
    if total_weighted == 0:
        logger.warning("Weighted demand is zero, falling back to proportional")
        return allocate_proportional(total_gwh, base_network, logger)

    bus_fractions = weighted_demand / total_weighted
    ev_allocation = bus_fractions * total_gwh

    logger.info(f"Allocated {ev_allocation.sum():.1f} GWh with urban weighting")
    return ev_allocation


ALLOCATION_METHODS = {
    'proportional': allocate_proportional,
    'uniform': allocate_uniform,
    'urban_weighted': allocate_urban_weighted,
}


def _bus_coords_kwargs(network: pypsa.Network, source_bus: str) -> Dict[str, float]:
    if source_bus in network.buses.index and {'x', 'y'}.issubset(network.buses.columns):
        return {
            'x': float(network.buses.at[source_bus, 'x']),
            'y': float(network.buses.at[source_bus, 'y']),
        }
    return {}


def _get_timestep_hours(index: pd.DatetimeIndex, default: float = 1.0) -> float:
    if index is None or len(index) < 2:
        return default
    delta_hours = (index[1] - index[0]).total_seconds() / 3600.0
    return delta_hours if delta_hours > 0 else default


def _get_fes_scenario_column(df: pd.DataFrame) -> Optional[str]:
    for col in ['FES Pathway', 'FES Scenario', '\ufeffFES Scenario', 'ï»¿FES Scenario']:
        if col in df.columns:
            return col
    return None


def _calculate_fes_fraction(fes_path: str,
                            fes_scenario: str,
                            modelled_year: int,
                            component_blocks: list,
                            logger: logging.Logger) -> Optional[float]:
    if not fes_path or not Path(fes_path).exists():
        return None

    try:
        fes = pd.read_csv(fes_path, low_memory=False)
    except Exception as exc:
        logger.warning(f"Failed to read FES data: {exc}")
        return None

    scenario_col = _get_fes_scenario_column(fes)
    if scenario_col is None or 'Building Block ID Number' not in fes.columns:
        return None

    year_col = str(modelled_year)
    if year_col not in fes.columns:
        return None

    total = fes[
        (fes['Building Block ID Number'] == 'Dem_BB003') &
        (fes[scenario_col] == fes_scenario)
    ]
    comp = fes[
        (fes['Building Block ID Number'].isin(component_blocks)) &
        (fes[scenario_col] == fes_scenario)
    ]

    total_val = pd.to_numeric(total[year_col], errors='coerce').sum()
    comp_val = pd.to_numeric(comp[year_col], errors='coerce').sum()

    if total_val <= 0:
        return None
    return comp_val / total_val


def _load_fes_component_gsp_demand(
    fes_path: str,
    fes_scenario: str,
    modelled_year: int,
    component_blocks: list,
    logger: logging.Logger
) -> Optional[pd.Series]:
    if isinstance(fes_path, list):
        fes_path = fes_path[0] if fes_path else None

    if not fes_path or not Path(fes_path).exists():
        return None

    try:
        fes = pd.read_csv(fes_path, low_memory=False)
    except Exception as exc:
        logger.warning(f"Failed to read FES data: {exc}")
        return None

    scenario_col = _get_fes_scenario_column(fes)
    if scenario_col is None or 'Building Block ID Number' not in fes.columns:
        return None

    year_col = str(modelled_year)
    if year_col not in fes.columns or 'GSP' not in fes.columns:
        return None

    filtered = fes[
        (fes['Building Block ID Number'].isin(component_blocks)) &
        (fes[scenario_col] == fes_scenario)
    ].copy()

    if filtered.empty:
        return None

    filtered = filtered[filtered['GSP'].notna()]
    filtered['GSP'] = filtered['GSP'].astype(str).str.strip()
    filtered['_value'] = pd.to_numeric(filtered[year_col], errors='coerce')

    gsp_demand = filtered.groupby('GSP')['_value'].sum().dropna()
    gsp_demand = gsp_demand[gsp_demand > 0]

    if gsp_demand.empty:
        return None

    logger.info(f"Loaded FES GSP demand for {len(gsp_demand)} GSPs (total {gsp_demand.sum():.1f} GWh)")
    return gsp_demand


def _allocate_using_fes_gsp(
    total_gwh: float,
    base_network: pypsa.Network,
    component_profile: pd.DataFrame,
    fes_path: str,
    fes_scenario: str,
    modelled_year: int,
    network_model: str,
    fes_year: Optional[int],
    component_blocks: list,
    logger: logging.Logger
) -> Optional[pd.Series]:
    if not fes_scenario:
        return None

    gsp_demand = _load_fes_component_gsp_demand(
        fes_path, fes_scenario, modelled_year, component_blocks, logger
    )
    if gsp_demand is None:
        return None

    total_fes = gsp_demand.sum()
    if total_fes <= 0:
        return None

    scaled_gsp = gsp_demand * (total_gwh / total_fes)
    weights = scaled_gsp / scaled_gsp.sum()

    profile_series = component_profile.iloc[:, 0]
    gsp_timeseries = pd.DataFrame(
        {gsp: profile_series * weight for gsp, weight in weights.items()},
        index=component_profile.index
    )

    fes_demand = scaled_gsp.to_frame(name='demand_gwh')
    if isinstance(network_model, list):
        network_model = network_model[0] if network_model else "ETYS"
    if not network_model:
        network_model = "ETYS"
    if isinstance(fes_year, list):
        fes_year = fes_year[0] if fes_year else None
    mapping_year = int(fes_year) if fes_year else 2024
    spatial_mapping = load_spatial_mapping_data(
        network_model, list(fes_demand.index), mapping_year, logger
    )

    aggregated_demand, _ = aggregate_demand_to_network_topology(
        fes_demand,
        gsp_timeseries,
        network_model,
        base_network,
        spatial_mapping,
        logger,
        is_historical=False
    )

    allocation = aggregated_demand.iloc[:, 0]
    allocation = allocation.reindex(base_network.buses.index, fill_value=0.0)
    logger.info(
        f"Allocated {allocation.sum():.1f} GWh across {len(allocation)} buses using FES GSP distribution"
    )
    return allocation

# ──────────────────────────────────────────────────────────────────────────────
# FES V2G Capacity Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_fes_v2g_capacity(
    fes_path: str,
    fes_scenario: str,
    modelled_year: int,
    network: pypsa.Network,
    logger: logging.Logger
) -> Optional[pd.Series]:
    """
    Load V2G capacity from FES Srg_BB005 data by GSP.
    
    FES Srg_BB005 provides V2G capacity (MW availability) at GSP level.
    Negative values indicate export capacity (V2G provides power to grid).
    
    Args:
        fes_path: Path to FES data CSV
        fes_scenario: FES scenario name (e.g., 'Holistic Transition')
        modelled_year: Target year (e.g., 2035)
        network: PyPSA network for bus mapping
        logger: Logger instance
        
    Returns:
        Series of V2G capacity (MW) indexed by network bus, or None if unavailable
    """
    if isinstance(fes_path, list):
        fes_path = fes_path[0] if fes_path else None
    
    if not fes_path or not Path(fes_path).exists():
        logger.info("FES data not available for V2G capacity")
        return None
    
    try:
        fes = pd.read_csv(fes_path, low_memory=False)
    except Exception as exc:
        logger.warning(f"Failed to read FES data: {exc}")
        return None
    
    scenario_col = _get_fes_scenario_column(fes)
    if scenario_col is None or 'Building Block ID Number' not in fes.columns:
        logger.warning("FES data missing required columns for V2G")
        return None
    
    year_col = str(modelled_year)
    if year_col not in fes.columns or 'GSP' not in fes.columns:
        logger.warning(f"FES data missing year {modelled_year} or GSP column")
        return None
    
    # Filter for V2G capacity (Srg_BB005)
    v2g_data = fes[
        (fes['Building Block ID Number'] == 'Srg_BB005') &
        (fes[scenario_col] == fes_scenario)
    ].copy()
    
    if v2g_data.empty:
        logger.info(f"No V2G data (Srg_BB005) found for {fes_scenario}")
        return None
    
    v2g_data['GSP'] = v2g_data['GSP'].astype(str).str.strip()
    v2g_data['v2g_mw'] = pd.to_numeric(v2g_data[year_col], errors='coerce')
    
    # V2G capacity is negative in FES (export to grid), take absolute value
    v2g_data['v2g_mw'] = v2g_data['v2g_mw'].abs()
    
    gsp_v2g = v2g_data.groupby('GSP')['v2g_mw'].sum().dropna()
    gsp_v2g = gsp_v2g[gsp_v2g > 0]
    
    if gsp_v2g.empty:
        logger.info("No non-zero V2G capacity in FES data")
        return None
    
    total_v2g = gsp_v2g.sum()
    logger.info(f"Loaded FES V2G capacity: {total_v2g:,.0f} MW across {len(gsp_v2g)} GSPs")
    
    # Map GSP to network buses (simplified - use existing spatial mapping logic)
    # For now, return GSP-level data; the caller can handle bus mapping
    return gsp_v2g


def load_fes_smart_charging_capacity(
    fes_path: str,
    fes_scenario: str,
    modelled_year: int,
    logger: logging.Logger
) -> Optional[float]:
    """
    Load smart charging capacity from FES Srg_BB007a data.
    
    FES Srg_BB007a provides smart charging availability (MW) at GB level.
    
    Args:
        fes_path: Path to FES data CSV
        fes_scenario: FES scenario name (e.g., 'Holistic Transition')
        modelled_year: Target year (e.g., 2035)
        logger: Logger instance
        
    Returns:
        Total smart charging capacity (MW), or None if unavailable
    """
    if isinstance(fes_path, list):
        fes_path = fes_path[0] if fes_path else None
    
    if not fes_path or not Path(fes_path).exists():
        logger.info("FES data not available for smart charging capacity")
        return None
    
    try:
        fes = pd.read_csv(fes_path, low_memory=False)
    except Exception as exc:
        logger.warning(f"Failed to read FES data: {exc}")
        return None
    
    scenario_col = _get_fes_scenario_column(fes)
    if scenario_col is None or 'Building Block ID Number' not in fes.columns:
        logger.warning("FES data missing required columns for smart charging")
        return None
    
    year_col = str(modelled_year)
    if year_col not in fes.columns:
        logger.warning(f"FES data missing year {modelled_year}")
        return None
    
    # Filter for smart charging capacity (Srg_BB007a)
    smart_data = fes[
        (fes['Building Block ID Number'] == 'Srg_BB007a') &
        (fes[scenario_col] == fes_scenario)
    ].copy()
    
    if smart_data.empty:
        logger.info(f"No smart charging data (Srg_BB007a) found for {fes_scenario}")
        return None
    
    smart_data['smart_mw'] = pd.to_numeric(smart_data[year_col], errors='coerce')
    
    # Smart charging capacity should be positive (MW available for DSR)
    total_smart = smart_data['smart_mw'].abs().sum()
    
    if total_smart <= 0:
        logger.info("No non-zero smart charging capacity in FES data")
        return None
    
    logger.info(f"Loaded FES smart charging capacity: {total_smart:,.0f} MW")
    return total_smart


def calculate_mixed_mode_shares(
    ev_config: Dict[str, Any],
    fes_v2g_capacity: Optional[pd.Series],
    fes_smart_capacity: Optional[float],
    total_ev_demand_mw: float,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Calculate the shares for MIXED mode (GO, INT, V2G).
    
    Uses FES building blocks to derive realistic shares:
    - V2G share: FES Srg_BB005 (V2G MW) / total EV capacity
    - INT share: FES Srg_BB007a (Smart charging MW) / total EV capacity - V2G share
    - GO share: Remainder (non-smart charging EVs)
    
    Args:
        ev_config: EV flexibility configuration
        fes_v2g_capacity: FES V2G capacity per GSP (Srg_BB005)
        fes_smart_capacity: FES smart charging capacity (Srg_BB007a)
        total_ev_demand_mw: Total EV demand for capacity reference
        logger: Logger instance
        
    Returns:
        Dictionary with go_share, int_share, v2g_share (sum to 1.0)
    """
    mixed_config = ev_config.get('mixed', {})
    mode = mixed_config.get('mode', 'fes')
    
    # Default manual shares
    go_share = mixed_config.get('go_share', 0.30)
    int_share = mixed_config.get('int_share', 0.50)
    v2g_share = mixed_config.get('v2g_share', 0.20)
    
    if mode == 'fes':
        # Get FES-derived shares
        fes_overrides = mixed_config.get('fes_share_overrides') or {}
        
        # Estimate total EV charging capacity (MW) from demand
        charging_hours_per_day = ev_config.get('charging_hours_per_day', 4.0)
        estimated_capacity_mw = total_ev_demand_mw * 24 / charging_hours_per_day
        
        if estimated_capacity_mw <= 0:
            logger.warning("Cannot calculate FES shares - zero EV capacity")
            logger.info("Using manual shares")
        else:
            # Calculate V2G share from FES Srg_BB005
            if fes_v2g_capacity is not None and len(fes_v2g_capacity) > 0:
                total_v2g_mw = fes_v2g_capacity.sum()
                fes_v2g_share = min(0.95, total_v2g_mw / estimated_capacity_mw)
                logger.info(f"FES V2G share: {fes_v2g_share:.1%} ({total_v2g_mw:,.0f} MW / {estimated_capacity_mw:,.0f} MW est. capacity)")
            else:
                fes_v2g_share = None
                logger.info("No FES V2G data - using manual v2g_share")
            
            # Calculate smart charging share from FES Srg_BB007a
            if fes_smart_capacity is not None and fes_smart_capacity > 0:
                fes_smart_share = min(0.95, fes_smart_capacity / estimated_capacity_mw)
                logger.info(f"FES smart charging share: {fes_smart_share:.1%} ({fes_smart_capacity:,.0f} MW / {estimated_capacity_mw:,.0f} MW est. capacity)")
            else:
                fes_smart_share = None
                logger.info("No FES smart charging data - using manual int_share")
            
            # Apply FES values with optional overrides
            if fes_v2g_share is not None:
                v2g_share = fes_overrides.get('v2g_share', fes_v2g_share)
                if v2g_share is None:
                    v2g_share = fes_v2g_share
            
            if fes_smart_share is not None:
                # INT share = total smart - V2G (since V2G is a subset of smart charging)
                fes_int_share = max(0, fes_smart_share - v2g_share)
                int_share = fes_overrides.get('int_share', fes_int_share)
                if int_share is None:
                    int_share = fes_int_share
            
            # GO share is the remainder
            go_share = fes_overrides.get('go_share')
            if go_share is None:
                go_share = max(0, 1.0 - int_share - v2g_share)
    
    # Normalize to ensure sum = 1.0
    total = go_share + int_share + v2g_share
    if total > 0 and abs(total - 1.0) > 0.001:
        logger.warning(f"Shares sum to {total:.3f}, normalizing to 1.0")
        go_share /= total
        int_share /= total
        v2g_share /= total
    
    shares = {
        'go_share': go_share,
        'int_share': int_share,
        'v2g_share': v2g_share
    }
    
    logger.info(f"MIXED mode shares: GO={go_share:.1%}, INT={int_share:.1%}, V2G={v2g_share:.1%}")
    return shares


def add_ev_mixed_mode(n: pypsa.Network,
                      buses: list,
                      ev_demand_mw: pd.DataFrame,
                      availability_profile: pd.DataFrame,
                      dsm_profile: pd.DataFrame,
                      config: Dict[str, Any],
                      shares: Dict[str, float],
                      logger: Optional[logging.Logger] = None,
                      fes_v2g_capacity: Optional[pd.Series] = None) -> pypsa.Network:
    """
    Add EV flexibility with MIXED mode - splitting fleet across GO, INT, and V2G.
    
    Creates separate components for each tariff type, scaled by their shares.
    
    Args:
        n: PyPSA network
        buses: List of bus names
        ev_demand_mw: EV daily driving demand
        availability_profile: When EVs are plugged in
        dsm_profile: Minimum SOC requirements
        config: Flexibility configuration
        shares: Dictionary with go_share, int_share, v2g_share
        logger: Logger instance
        fes_v2g_capacity: Optional FES V2G capacity per GSP
        
    Returns:
        Network with MIXED mode components
    """
    if logger:
        logger.info("Adding EV MIXED mode flexibility...")
        logger.info(f"  GO share: {shares['go_share']:.1%}")
        logger.info(f"  INT share: {shares['int_share']:.1%}")
        logger.info(f"  V2G share: {shares['v2g_share']:.1%}")
    
    # Get EV parameters
    ev_params = _get_ev_params(config)
    battery_capacity_kwh = ev_params['battery_capacity_kwh']
    charge_efficiency = ev_params['charge_efficiency']
    flex_participation = ev_params['flexibility_participation']
    flex_share = config.get('flex_share', 1.0)
    
    go_config = config.get('go', {})
    int_config = config.get('int', {})
    v2g_config = config.get('v2g', {})
    
    window = go_config.get('window', ['00:00', '04:00'])
    window_start = int(window[0].split(':')[0])
    window_end = int(window[1].split(':')[0])
    
    charger_power_kw = int_config.get('charger_power_kw', 7.0)
    min_soc = int_config.get('min_soc', 0.20)
    
    discharge_efficiency = v2g_config.get('discharge_efficiency', 0.90)
    max_discharge_soc = v2g_config.get('max_discharge_soc', 0.80)
    degradation_cost = v2g_config.get('degradation_cost_per_mwh', 50.0)
    
    # Create GO tariff cost profile (cheap during window, normal price outside)
    go_window_cost = go_config.get('window_cost', 0.0)  # £/MWh during cheap window
    go_offpeak_cost = go_config.get('offpeak_cost', 100.0)  # £/MWh outside window
    go_cost_profile = create_go_tariff_cost_profile(
        n.snapshots, window_start, window_end, go_window_cost, go_offpeak_cost
    )
    
    if logger:
        logger.info(f"GO charging window: {window_start}:00-{window_end}:00 at £{go_window_cost}/MWh")
        logger.info(f"GO off-window cost: £{go_offpeak_cost}/MWh (charging still allowed)")
    
    # Calculate per-bus V2G capacity from FES if available
    bus_v2g_capacity = None
    use_fes_capacity = v2g_config.get('use_fes_capacity', True)
    using_fes_v2g = (fes_v2g_capacity is not None and 
                     len(fes_v2g_capacity) > 0 and 
                     use_fes_capacity)
    
    if using_fes_v2g:
        total_fes_v2g = fes_v2g_capacity.sum()
        valid_buses = [b for b in buses if b in ev_demand_mw.columns]
        total_ev_demand = sum(ev_demand_mw[b].sum() for b in valid_buses)
        
        if total_ev_demand > 0:
            bus_v2g_capacity = {}
            for bus in valid_buses:
                bus_demand = ev_demand_mw[bus].sum()
                demand_share = bus_demand / total_ev_demand
                # V2G capacity is the V2G share of the FES total, scaled by flex_share
                bus_v2g_capacity[bus] = total_fes_v2g * demand_share * shares['v2g_share'] * flex_share
    
    for bus in buses:
        if bus not in ev_demand_mw.columns:
            continue
        
        # Estimate fleet size
        energy_per_vehicle = config.get('energy_per_vehicle_kwh_per_day', 10.0)
        daily_demand_mwh = ev_demand_mw[bus].sum() / 365
        n_vehicles = max(1, int(daily_demand_mwh * 1000 / energy_per_vehicle))
        n_flex_vehicles = max(1, int(n_vehicles * flex_participation))

        # Split demand across modes - ONLY flex_share fraction goes to EV battery buses
        # The remaining (1 - flex_share) is dumb load handled separately
        go_demand = ev_demand_mw[bus] * shares['go_share'] * flex_share
        int_demand = ev_demand_mw[bus] * shares['int_share'] * flex_share
        v2g_demand = ev_demand_mw[bus] * shares['v2g_share'] * flex_share
        
        # Get profiles
        if bus in availability_profile.columns:
            avail = availability_profile[bus]
        else:
            avail = availability_profile.mean(axis=1)
        
        if bus in dsm_profile.columns:
            dsm = dsm_profile[bus]
        else:
            dsm = dsm_profile.mean(axis=1) if len(dsm_profile.columns) > 0 else pd.Series(min_soc, index=n.snapshots)
        
        e_min_pu = pd.Series(min_soc, index=n.snapshots)
        e_min_pu = e_min_pu.combine(dsm, max)
        
        # ──── GO Mode Components ────
        min_share = config.get('min_share_threshold', 0.001)
        if shares['go_share'] > min_share:
            go_battery_bus = f"{bus} EV battery GO"
            go_store_name = f"{bus} EV fleet battery GO"
            go_charger_name = f"{bus} EV charger GO"
            go_demand_name = f"{bus} EV driving GO"
            
            n_go_vehicles = max(1, int(n_flex_vehicles * shares['go_share']))
            
            if go_battery_bus not in n.buses.index:
                bus_kwargs = {"carrier": "EV battery"}
                bus_kwargs.update(_bus_coords_kwargs(n, bus))
                n.add("Bus", go_battery_bus, **bus_kwargs)
            
            fleet_capacity_mwh = n_go_vehicles * battery_capacity_kwh / 1000 * flex_share
            n.add("Store",
                  go_store_name,
                  bus=go_battery_bus,
                  carrier="EV battery",
                  e_nom=fleet_capacity_mwh,
                  e_nom_extendable=False,
                  e_cyclic=True,
                  e_min_pu=min_soc)
            
            charger_power_mw = n_go_vehicles * charger_power_kw / 1000 * flex_share
            n.add("Link",
                  go_charger_name,
                  bus0=bus,
                  bus1=go_battery_bus,
                  carrier="EV charger",
                  efficiency=charge_efficiency,
                  p_nom=charger_power_mw,
                  p_nom_extendable=False,
                  p_max_pu=avail,  # Use availability profile (same as INT)
                  marginal_cost=go_cost_profile)  # Use cost to incentivize window charging
            
            n.add("Load",
                  go_demand_name,
                  bus=go_battery_bus,
                  carrier="EV driving",
                  p_set=go_demand)
        
        # ──── INT Mode Components ────
        if shares['int_share'] > min_share:
            int_battery_bus = f"{bus} EV battery INT"
            int_store_name = f"{bus} EV fleet battery INT"
            int_charger_name = f"{bus} EV charger INT"
            int_demand_name = f"{bus} EV driving INT"
            
            n_int_vehicles = max(1, int(n_flex_vehicles * shares['int_share']))
            
            if int_battery_bus not in n.buses.index:
                bus_kwargs = {"carrier": "EV battery"}
                bus_kwargs.update(_bus_coords_kwargs(n, bus))
                n.add("Bus", int_battery_bus, **bus_kwargs)
            
            fleet_capacity_mwh = n_int_vehicles * battery_capacity_kwh / 1000 * flex_share
            n.add("Store",
                  int_store_name,
                  bus=int_battery_bus,
                  carrier="EV battery",
                  e_nom=fleet_capacity_mwh,
                  e_nom_extendable=False,
                  e_cyclic=True,
                  e_min_pu=e_min_pu)
            
            charger_power_mw = n_int_vehicles * charger_power_kw / 1000 * flex_share
            n.add("Link",
                  int_charger_name,
                  bus0=bus,
                  bus1=int_battery_bus,
                  carrier="EV charger",
                  efficiency=charge_efficiency,
                  p_nom=charger_power_mw,
                  p_nom_extendable=False,
                  p_max_pu=avail)
            
            n.add("Load",
                  int_demand_name,
                  bus=int_battery_bus,
                  carrier="EV driving",
                  p_set=int_demand)
        
        # ──── V2G Mode Components ────
        if shares['v2g_share'] > min_share:
            v2g_battery_bus = f"{bus} EV battery V2G"
            v2g_store_name = f"{bus} EV fleet battery V2G"
            v2g_charger_name = f"{bus} EV charger V2G"
            v2g_discharge_name = f"{bus} V2G"
            v2g_demand_name = f"{bus} EV driving V2G"
            
            n_v2g_vehicles = max(1, int(n_flex_vehicles * shares['v2g_share']))
            
            if v2g_battery_bus not in n.buses.index:
                bus_kwargs = {"carrier": "EV battery"}
                bus_kwargs.update(_bus_coords_kwargs(n, bus))
                n.add("Bus", v2g_battery_bus, **bus_kwargs)
            
            fleet_capacity_mwh = n_v2g_vehicles * battery_capacity_kwh / 1000 * flex_share
            n.add("Store",
                  v2g_store_name,
                  bus=v2g_battery_bus,
                  carrier="EV battery",
                  e_nom=fleet_capacity_mwh,
                  e_nom_extendable=False,
                  e_cyclic=True,
                  e_min_pu=e_min_pu)
            
            charger_power_mw = n_v2g_vehicles * charger_power_kw / 1000 * flex_share
            n.add("Link",
                  v2g_charger_name,
                  bus0=bus,
                  bus1=v2g_battery_bus,
                  carrier="EV charger",
                  efficiency=charge_efficiency,
                  p_nom=charger_power_mw,
                  p_nom_extendable=False,
                  p_max_pu=avail)
            
            # V2G discharge link
            if using_fes_v2g and bus_v2g_capacity is not None and bus in bus_v2g_capacity:
                v2g_power_mw = bus_v2g_capacity[bus]
            else:
                v2g_power_mw = n_v2g_vehicles * charger_power_kw / 1000 * flex_share
            
            if v2g_power_mw > 0:
                n.add("Link",
                      v2g_discharge_name,
                      bus0=v2g_battery_bus,
                      bus1=bus,
                      carrier="V2G",
                      efficiency=discharge_efficiency,
                      p_nom=v2g_power_mw,
                      p_nom_extendable=False,
                      p_max_pu=avail * max_discharge_soc,
                      marginal_cost=degradation_cost)
            
            n.add("Load",
                  v2g_demand_name,
                  bus=v2g_battery_bus,
                  carrier="EV driving",
                  p_set=v2g_demand)
    
    if logger:
        logger.info(f"Added MIXED mode flexibility for {len(buses)} buses")
    
    return n


# ──────────────────────────────────────────────────────────────────────────────
# EV Flexibility Functions
# ──────────────────────────────────────────────────────────────────────────────

def _get_ev_params(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract EV parameters from config with sensible defaults.

    All defaults match config/defaults.yaml values.

    Args:
        config: EV flexibility configuration dictionary

    Returns:
        Dictionary with EV parameters
    """
    return {
        'battery_capacity_kwh': config.get('battery_capacity_kwh', 60.0),
        'charge_efficiency': config.get('charge_efficiency', 0.90),
        'flexibility_participation': config.get('flexibility_participation', 0.10),
    }


def add_ev_as_load(n: pypsa.Network,
                   buses: list,
                   ev_demand_mw: pd.DataFrame,
                   carrier: str = "EV charging",
                   logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Add EVs as simple Load components (no flexibility).

    Args:
        n: PyPSA network
        buses: List of bus names
        ev_demand_mw: EV charging demand time series (MW) with bus columns
        carrier: Carrier name for the loads
        logger: Logger instance

    Returns:
        Network with EV loads added
    """
    if logger:
        logger.info("Adding EVs as Load components (no flexibility)...")

    for bus in buses:
        if bus in ev_demand_mw.columns:
            load_name = f"{bus} EV charging"
            n.add("Load",
                  load_name,
                  bus=bus,
                  carrier=carrier,
                  p_set=ev_demand_mw[bus])

    if logger:
        logger.info(f"Added {len(buses)} EV charging loads")

    return n


def create_go_tariff_window(snapshots: pd.DatetimeIndex,
                            window_start: int = 0,
                            window_end: int = 4) -> pd.Series:
    """
    Create charging window profile for GO tariff (fixed night window).

    Args:
        snapshots: DatetimeIndex for the model period
        window_start: Hour when charging window opens (default 0 = midnight)
        window_end: Hour when charging window closes (default 4 = 4am)

    Returns:
        Series with 1.0 during window, 0.0 outside
    """
    hours = snapshots.hour
    in_window = (hours >= window_start) & (hours < window_end)
    return pd.Series(in_window.astype(float), index=snapshots)


def create_go_tariff_cost_profile(snapshots: pd.DatetimeIndex,
                                   window_start: int = 0,
                                   window_end: int = 4,
                                   window_cost: float = 0.0,
                                   offpeak_cost: float = 100.0) -> pd.Series:
    """
    Create marginal cost profile for GO tariff charging.
    
    GO tariff encourages charging during the cheap window by using price
    incentives rather than physical constraints. Charging can occur at
    any time, but is much cheaper during the window.
    
    Args:
        snapshots: DatetimeIndex for the model period
        window_start: Hour when cheap window opens (default 0 = midnight)
        window_end: Hour when cheap window closes (default 4 = 4am)
        window_cost: Marginal cost during cheap window (£/MWh, default 0)
        offpeak_cost: Marginal cost outside window (£/MWh, default 100)
    
    Returns:
        Series with marginal costs - low during window, high outside
    """
    hours = snapshots.hour
    in_window = (hours >= window_start) & (hours < window_end)
    costs = pd.Series(offpeak_cost, index=snapshots)
    costs[in_window] = window_cost
    return costs


def add_ev_go_tariff(n: pypsa.Network,
                     buses: list,
                     ev_demand_mw: pd.DataFrame,
                     config: Dict[str, Any],
                     logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Add EV smart charging with GO tariff (fixed night window).

    Octopus Go style tariff - charging only available during a fixed
    4-hour window overnight (typically 00:00-04:00).

    Creates PyPSA components:
    - Store: EV battery storage
    - Link: Charger with time-limited availability (p_max_pu)
    - Load: Daily driving demand (must be met)

    Args:
        n: PyPSA network
        buses: List of bus names
        ev_demand_mw: EV charging demand (daily energy requirements)
        config: Flexibility configuration
        logger: Logger instance

    Returns:
        Network with GO tariff components
    """
    if logger:
        logger.info("Adding EV GO tariff flexibility (fixed night window)...")

    # Get EV parameters from config
    ev_params = _get_ev_params(config)
    battery_capacity_kwh = ev_params['battery_capacity_kwh']
    charge_efficiency = ev_params['charge_efficiency']
    flex_participation = ev_params['flexibility_participation']

    go_config = config.get('go', {})
    window = go_config.get('window', ['00:00', '04:00'])
    window_start = int(window[0].split(':')[0])
    window_end = int(window[1].split(':')[0])
    
    # GO tariff cost parameters
    window_cost = go_config.get('window_cost', 0.0)
    offpeak_cost = go_config.get('offpeak_cost', 100.0)

    int_config = config.get('int', {})
    charger_power_kw = int_config.get('charger_power_kw', 7.0)
    min_soc = int_config.get('min_soc', 0.20)

    if logger:
        logger.info(f"GO window: {window_start:02d}:00 - {window_end:02d}:00 at £{window_cost}/MWh")
        logger.info(f"GO off-window cost: £{offpeak_cost}/MWh (charging still allowed)")
        logger.info(f"Charger power: {charger_power_kw} kW, Battery: {battery_capacity_kwh} kWh")
        logger.info(f"Flexibility participation: {flex_participation:.1%}")

    # Create cost profile for GO tariff (cheap during window, expensive outside)
    go_cost_profile = create_go_tariff_cost_profile(
        n.snapshots, window_start, window_end, window_cost, offpeak_cost
    )

    for bus in buses:
        if bus not in ev_demand_mw.columns:
            continue

        battery_bus = f"{bus} EV battery"
        store_name = f"{bus} EV fleet battery"
        charger_name = f"{bus} EV charger"
        demand_name = f"{bus} EV driving demand"

        # Estimate fleet size from demand
        energy_per_vehicle = config.get('energy_per_vehicle_kwh_per_day', 10.0)
        daily_demand_mwh = ev_demand_mw[bus].sum() / 365
        n_vehicles = max(1, int(daily_demand_mwh * 1000 / energy_per_vehicle))

        # Scale by participation rate - only a fraction of fleet participates in flexibility
        n_flex_vehicles = max(1, int(n_vehicles * flex_participation))

        # Add EV battery bus
        if battery_bus not in n.buses.index:
            bus_kwargs = {"carrier": "EV battery"}
            bus_kwargs.update(_bus_coords_kwargs(n, bus))
            n.add("Bus", battery_bus, **bus_kwargs)

        # Add EV fleet battery (Store) - scaled by participation
        fleet_capacity_mwh = n_flex_vehicles * battery_capacity_kwh / 1000
        n.add("Store",
              store_name,
              bus=battery_bus,
              carrier="EV battery",
              e_nom=fleet_capacity_mwh,
              e_nom_extendable=False,
              e_cyclic=True,
              e_min_pu=min_soc)  # Minimum SOC constraint

        # Add charger (Link with cost-based incentive for window charging)
        # GO tariff uses marginal cost to encourage window charging,
        # but charging CAN occur at any time if needed
        charger_power_mw = n_flex_vehicles * charger_power_kw / 1000
        n.add("Link",
              charger_name,
              bus0=bus,  # Grid
              bus1=battery_bus,  # Battery
              carrier="EV charger",
              efficiency=charge_efficiency,
              p_nom=charger_power_mw,
              p_nom_extendable=False,
              marginal_cost=go_cost_profile)  # Cost incentive for window charging

        # Add driving demand (Load from battery)
        n.add("Load",
              demand_name,
              bus=battery_bus,
              carrier="EV driving",
              p_set=ev_demand_mw[bus])

    if logger:
        logger.info(f"Added GO tariff flexibility for {len(buses)} buses")

    return n


def add_ev_smart_charging(n: pypsa.Network,
                          buses: list,
                          ev_demand_mw: pd.DataFrame,
                          availability_profile: pd.DataFrame,
                          dsm_profile: pd.DataFrame,
                          config: Dict[str, Any],
                          logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Add EV smart charging with INT tariff (fully optimizable).

    Intelligent tariff - charging can be optimized across all hours when
    vehicles are plugged in (availability profile). Must meet minimum SOC
    requirements (DSM profile).

    Creates PyPSA components:
    - Store: EV battery with DSM constraint (e_min_pu)
    - Link: Charger with availability constraint (p_max_pu)
    - Load: Driving demand

    Args:
        n: PyPSA network
        buses: List of bus names
        ev_demand_mw: EV daily driving demand
        availability_profile: When EVs are plugged in (0-1)
        dsm_profile: Minimum SOC requirements (0-1)
        config: Flexibility configuration
        logger: Logger instance

    Returns:
        Network with INT smart charging components
    """
    if logger:
        logger.info("Adding EV INT tariff flexibility (smart charging)...")

    # Get EV parameters from config
    ev_params = _get_ev_params(config)
    battery_capacity_kwh = ev_params['battery_capacity_kwh']
    charge_efficiency = ev_params['charge_efficiency']
    flex_participation = ev_params['flexibility_participation']
    
    # Get flex_share - fraction of total EV demand that participates in flexibility
    flex_share = config.get('flex_share', 1.0)

    int_config = config.get('int', {})
    charger_power_kw = int_config.get('charger_power_kw', 7.0)
    min_soc = int_config.get('min_soc', 0.20)
    target_departure_soc = int_config.get('target_departure_soc', 0.80)

    if logger:
        logger.info(f"Charger power: {charger_power_kw} kW, Battery: {battery_capacity_kwh} kWh")
        logger.info(f"Min SOC: {min_soc*100:.0f}%, Target departure: {target_departure_soc*100:.0f}%")
        logger.info(f"Flex share: {flex_share:.0%}, Flexibility participation: {flex_participation:.1%}")

    for bus in buses:
        if bus not in ev_demand_mw.columns:
            continue

        battery_bus = f"{bus} EV battery"
        store_name = f"{bus} EV fleet battery"
        charger_name = f"{bus} EV charger"
        demand_name = f"{bus} EV driving demand"

        # Estimate fleet size from demand
        energy_per_vehicle = config.get('energy_per_vehicle_kwh_per_day', 10.0)
        daily_demand_mwh = ev_demand_mw[bus].sum() / 365
        n_vehicles = max(1, int(daily_demand_mwh * 1000 / energy_per_vehicle))

        # Scale by participation rate - only a fraction of fleet participates in flexibility
        n_flex_vehicles = max(1, int(n_vehicles * flex_participation))

        # Add EV battery bus
        if battery_bus not in n.buses.index:
            bus_kwargs = {"carrier": "EV battery"}
            bus_kwargs.update(_bus_coords_kwargs(n, bus))
            n.add("Bus", battery_bus, **bus_kwargs)

        # Get availability profile for this bus
        if bus in availability_profile.columns:
            avail = availability_profile[bus]
        else:
            avail = availability_profile.mean(axis=1)

        # Get DSM profile (minimum SOC) for this bus
        if bus in dsm_profile.columns:
            dsm = dsm_profile[bus]
        else:
            dsm = dsm_profile.mean(axis=1) if len(dsm_profile.columns) > 0 else pd.Series(min_soc, index=n.snapshots)

        # Combine base min_soc with DSM constraints
        e_min_pu = pd.Series(min_soc, index=n.snapshots)
        e_min_pu = e_min_pu.combine(dsm, max)  # Take maximum of base and DSM

        # Add EV fleet battery (Store) - scaled by participation and flex_share
        fleet_capacity_mwh = n_flex_vehicles * battery_capacity_kwh / 1000 * flex_share
        n.add("Store",
              store_name,
              bus=battery_bus,
              carrier="EV battery",
              e_nom=fleet_capacity_mwh,
              e_nom_extendable=False,
              e_cyclic=True,
              e_min_pu=e_min_pu)  # Time-varying minimum SOC

        # Add charger (Link with availability constraint) - scaled by participation and flex_share
        charger_power_mw = n_flex_vehicles * charger_power_kw / 1000 * flex_share
        n.add("Link",
              charger_name,
              bus0=bus,  # Grid
              bus1=battery_bus,  # Battery
              carrier="EV charger",
              efficiency=charge_efficiency,
              p_nom=charger_power_mw,
              p_nom_extendable=False,
              p_max_pu=avail)  # Only charge when plugged in

        # Add driving demand (Load from battery)
        n.add("Load",
              demand_name,
              bus=battery_bus,
              carrier="EV driving",
              p_set=ev_demand_mw[bus])

    if logger:
        logger.info(f"Added INT smart charging for {len(buses)} buses")

    return n


def add_ev_v2g(n: pypsa.Network,
               buses: list,
               ev_demand_mw: pd.DataFrame,
               availability_profile: pd.DataFrame,
               dsm_profile: pd.DataFrame,
               config: Dict[str, Any],
               logger: Optional[logging.Logger] = None,
               fes_v2g_capacity: Optional[pd.Series] = None) -> pypsa.Network:
    """
    Add EV Vehicle-to-Grid (V2G) bidirectional flexibility.

    V2G allows EVs to discharge back to the grid, providing additional
    flexibility. Includes separate charge and discharge links with
    different efficiencies and constraints.
    
    V2G capacity can come from:
    1. FES Srg_BB005 data (if fes_v2g_capacity provided) - distributed proportionally by EV demand
    2. Calculated from vehicle count × charger power × participation rates

    Creates PyPSA components:
    - Store: EV battery
    - Link (charge): Grid -> Battery
    - Link (discharge): Battery -> Grid (V2G)
    - Load: Driving demand

    Args:
        n: PyPSA network
        buses: List of bus names
        ev_demand_mw: EV daily driving demand
        availability_profile: When EVs are plugged in
        dsm_profile: Minimum SOC requirements
        config: Flexibility configuration
        logger: Logger instance
        fes_v2g_capacity: Optional GSP-level V2G capacity from FES Srg_BB005 (MW)

    Returns:
        Network with V2G components
    """
    if logger:
        logger.info("Adding EV V2G flexibility (bidirectional)...")

    # Get EV parameters from config
    ev_params = _get_ev_params(config)
    battery_capacity_kwh = ev_params['battery_capacity_kwh']
    charge_efficiency = ev_params['charge_efficiency']
    flex_participation = ev_params['flexibility_participation']
    
    # Get flex_share - fraction of total EV demand that participates in flexibility
    flex_share = config.get('flex_share', 1.0)

    v2g_config = config.get('v2g', {})
    participation_rate = v2g_config.get('participation_rate', 0.30)
    discharge_efficiency = v2g_config.get('discharge_efficiency', 0.90)
    max_discharge_soc = v2g_config.get('max_discharge_soc', 0.80)
    degradation_cost = v2g_config.get('degradation_cost_per_mwh', 50.0)
    use_fes_capacity = v2g_config.get('use_fes_capacity', True)

    int_config = config.get('int', {})
    charger_power_kw = int_config.get('charger_power_kw', 7.0)
    min_soc = int_config.get('min_soc', 0.20)

    # Check if we should use FES V2G capacity
    using_fes = (fes_v2g_capacity is not None and 
                 len(fes_v2g_capacity) > 0 and 
                 use_fes_capacity)
    
    # Calculate per-bus V2G capacity distribution based on EV demand
    # This handles the case where FES GSP names don't match network bus names
    bus_v2g_capacity = None
    if using_fes:
        total_fes_v2g = fes_v2g_capacity.sum()
        # Calculate total EV demand across all buses
        valid_buses = [b for b in buses if b in ev_demand_mw.columns]
        total_ev_demand = sum(ev_demand_mw[b].sum() for b in valid_buses)
        
        if total_ev_demand > 0:
            # Distribute FES V2G capacity proportionally by EV demand at each bus
            bus_v2g_capacity = {}
            for bus in valid_buses:
                bus_demand = ev_demand_mw[bus].sum()
                demand_share = bus_demand / total_ev_demand
                bus_v2g_capacity[bus] = total_fes_v2g * demand_share * flex_share
            
            if logger:
                logger.info(f"Using FES Srg_BB005 V2G capacity: {total_fes_v2g:,.0f} MW total")
                logger.info(f"  Distributed proportionally to {len(bus_v2g_capacity)} buses by EV demand")
                logger.info(f"  After flex_share ({flex_share:.0%}): {total_fes_v2g * flex_share:,.0f} MW")
        else:
            using_fes = False
            if logger:
                logger.warning("No EV demand found, falling back to calculated V2G capacity")
    
    if not using_fes and logger:
        logger.info("Calculating V2G capacity from vehicle count × charger power")

    if logger:
        logger.info(f"Charger power: {charger_power_kw} kW, Battery: {battery_capacity_kwh} kWh")
        logger.info(f"Flex share: {flex_share:.0%}, Flexibility participation: {flex_participation:.1%}")
        if not using_fes:
            logger.info(f"V2G participation rate: {participation_rate*100:.0f}%")
        logger.info(f"Discharge efficiency: {discharge_efficiency*100:.0f}%")
        logger.info(f"Max discharge SOC: {max_discharge_soc*100:.0f}%")
        logger.info(f"Degradation cost: £{degradation_cost}/MWh")

    for bus in buses:
        if bus not in ev_demand_mw.columns:
            continue

        battery_bus = f"{bus} EV battery"
        store_name = f"{bus} EV fleet battery"
        charger_name = f"{bus} EV charger"
        v2g_name = f"{bus} V2G"
        demand_name = f"{bus} EV driving demand"

        # Estimate fleet size from demand
        energy_per_vehicle = config.get('energy_per_vehicle_kwh_per_day', 10.0)
        daily_demand_mwh = ev_demand_mw[bus].sum() / 365
        n_vehicles = max(1, int(daily_demand_mwh * 1000 / energy_per_vehicle))

        # Scale by participation rate - only a fraction of fleet participates in flexibility
        n_flex_vehicles = max(1, int(n_vehicles * flex_participation))
        n_v2g_vehicles = max(1, int(n_flex_vehicles * participation_rate))

        # Add EV battery bus
        if battery_bus not in n.buses.index:
            bus_kwargs = {"carrier": "EV battery"}
            bus_kwargs.update(_bus_coords_kwargs(n, bus))
            n.add("Bus", battery_bus, **bus_kwargs)

        # Get profiles
        if bus in availability_profile.columns:
            avail = availability_profile[bus]
        else:
            avail = availability_profile.mean(axis=1)

        if bus in dsm_profile.columns:
            dsm = dsm_profile[bus]
        else:
            dsm = dsm_profile.mean(axis=1) if len(dsm_profile.columns) > 0 else pd.Series(min_soc, index=n.snapshots)

        e_min_pu = pd.Series(min_soc, index=n.snapshots)
        e_min_pu = e_min_pu.combine(dsm, max)

        # Add EV fleet battery (Store) - scaled by participation and flex_share
        fleet_capacity_mwh = n_flex_vehicles * battery_capacity_kwh / 1000 * flex_share
        n.add("Store",
              store_name,
              bus=battery_bus,
              carrier="EV battery",
              e_nom=fleet_capacity_mwh,
              e_nom_extendable=False,
              e_cyclic=True,
              e_min_pu=e_min_pu)

        # Add charger (Grid -> Battery) - scaled by participation and flex_share and flex_share
        charger_power_mw = n_flex_vehicles * charger_power_kw / 1000 * flex_share
        n.add("Link",
              charger_name,
              bus0=bus,
              bus1=battery_bus,
              carrier="EV charger",
              efficiency=charge_efficiency,
              p_nom=charger_power_mw,
              p_nom_extendable=False,
              p_max_pu=avail)

        # Add V2G discharge link (Battery -> Grid)
        # Only V2G-capable vehicles can discharge
        if using_fes and bus_v2g_capacity is not None and bus in bus_v2g_capacity:
            # Use FES Srg_BB005 V2G capacity distributed by EV demand (already scaled by flex_share)
            v2g_power_mw = bus_v2g_capacity[bus]
        else:
            # Calculate from vehicle count × charger power, scaled by flex_share
            v2g_power_mw = n_v2g_vehicles * charger_power_kw / 1000 * flex_share
        if v2g_power_mw > 0:
            n.add("Link",
                  v2g_name,
                  bus0=battery_bus,
                  bus1=bus,
                  carrier="V2G",
                  efficiency=discharge_efficiency,
                  p_nom=v2g_power_mw,
                  p_nom_extendable=False,
                  p_max_pu=avail * max_discharge_soc,  # Limited by SOC
                  marginal_cost=degradation_cost)  # Battery degradation cost

        # Add driving demand
        n.add("Load",
              demand_name,
              bus=battery_bus,
              carrier="EV driving",
              p_set=ev_demand_mw[bus])

    if logger:
        logger.info(f"Added V2G flexibility for {len(buses)} buses")

    return n


def add_ev_flexibility(n: pypsa.Network,
                       buses: list,
                       ev_demand_mw: pd.DataFrame,
                       availability_profile: pd.DataFrame,
                       dsm_profile: pd.DataFrame,
                       flex_config: Dict[str, Any],
                       logger: Optional[logging.Logger] = None,
                       fes_v2g_capacity: Optional[pd.Series] = None) -> pypsa.Network:
    """
    Add EV flexibility to network based on configuration.

    This is the main entry point for EV flexibility modeling.

    Args:
        n: PyPSA network
        buses: List of bus names with EVs
        ev_demand_mw: EV charging demand time series
        availability_profile: When EVs are plugged in
        dsm_profile: Minimum SOC requirements
        flex_config: Flexibility configuration from defaults.yaml
        logger: Logger instance
        fes_v2g_capacity: Optional FES V2G capacity per bus (from Srg_BB005)

    Returns:
        Network with appropriate flexibility components
    """
    ev_config = flex_config.get('electric_vehicles', {})
    enabled = ev_config.get('enabled', False)
    tariff = ev_config.get('tariff', 'INT')

    if not enabled:
        if logger:
            logger.info("EV flexibility disabled, adding as simple loads")
        return add_ev_as_load(n, buses, ev_demand_mw, logger=logger)

    if logger:
        logger.info(f"EV flexibility enabled, tariff: {tariff}")

    if tariff.upper() == 'GO':
        return add_ev_go_tariff(n, buses, ev_demand_mw, ev_config, logger)
    elif tariff.upper() == 'INT':
        return add_ev_smart_charging(n, buses, ev_demand_mw, availability_profile,
                                     dsm_profile, ev_config, logger)
    elif tariff.upper() == 'V2G':
        return add_ev_v2g(n, buses, ev_demand_mw, availability_profile,
                          dsm_profile, ev_config, logger, fes_v2g_capacity)
    elif tariff.upper() == 'MIXED':
        # Calculate average demand for share calculation
        valid_buses = [b for b in buses if b in ev_demand_mw.columns]
        total_ev_demand_mw = sum(ev_demand_mw[b].mean() for b in valid_buses)
        
        # Calculate shares (uses FES data if available)
        shares = calculate_mixed_mode_shares(
            ev_config=ev_config,
            fes_v2g_capacity=fes_v2g_capacity,
            fes_smart_capacity=flex_config.get('_fes_smart_capacity'),  # Passed from caller
            total_ev_demand_mw=total_ev_demand_mw,
            logger=logger
        )
        
        return add_ev_mixed_mode(n, buses, ev_demand_mw, availability_profile,
                                 dsm_profile, ev_config, shares, logger, fes_v2g_capacity)
    else:
        if logger:
            logger.warning(f"Unknown tariff '{tariff}', defaulting to simple loads")
        return add_ev_as_load(n, buses, ev_demand_mw, logger=logger)


# ──────────────────────────────────────────────────────────────────────────────
# Main Processing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger = setup_logging(
        log_path=snakemake.log[0],
        log_level="INFO"
    )
    
    try:
        logger.info("=" * 80)
        logger.info("ELECTRIC VEHICLE DEMAND DISAGGREGATION")
        logger.info("=" * 80)
        
        # ──── Load Inputs ────
        logger.info("Loading inputs...")
        base_network = load_network(snakemake.input.base_demand, skip_time_series=False, custom_logger=logger)
        base_profile = pd.read_csv(snakemake.input.base_profile, index_col=0, parse_dates=True)
        
        logger.info(f"Base network: {len(base_network.buses)} buses, {len(base_network.loads)} loads")
        logger.info(f"Base profile shape: {base_profile.shape}")

        timestep_hours = _get_timestep_hours(base_profile.index)
        timesteps_per_day = int(round(24 / timestep_hours)) if timestep_hours > 0 else 24
        
        # ──── Get Configuration ────
        config = snakemake.params.component_config
        fraction = config.get("fraction_of_total", 0.08)
        allocation_method = config.get("allocation_method", "urban_weighted")
        source_file = config.get("source_file")
        use_fes_fraction = config.get("use_fes_fraction", False)
        min_gwh_threshold = float(config.get("min_gwh_threshold", 0.0))
        fes_fraction = None

        if not snakemake.params.is_historical:
            fes_fraction = _calculate_fes_fraction(
                snakemake.input.fes_data,
                snakemake.params.fes_scenario,
                snakemake.params.modelled_year,
                ['Dem_BB006', 'Dem_BB007'],
                logger
            )
            if fes_fraction is not None:
                logger.info(f"FES EV fraction: {fes_fraction:.2%}")
                if use_fes_fraction:
                    fraction = fes_fraction
        
        logger.info(f"Configuration:")
        logger.info(f"  Fraction of total demand: {fraction:.1%}")
        logger.info(f"  Allocation method: {allocation_method}")
        logger.info(f"  Source file: {source_file}")
        logger.info(f"  Min GWh threshold: {min_gwh_threshold}")
        
        # ──── Calculate Total EV Demand ────
        logger.info("Calculating EV charging demand...")
        
        total_base_demand_mwh = base_profile.sum().sum() * timestep_hours
        total_base_demand_gwh = total_base_demand_mwh / 1000.0
        total_ev_demand_gwh = total_base_demand_gwh * fraction
        total_ev_demand_mwh = total_ev_demand_gwh * 1000.0
        
        logger.info(f"Total base demand: {total_base_demand_gwh:.1f} GWh/year")
        logger.info(f"Target EV demand: {total_ev_demand_gwh:.1f} GWh/year ({fraction:.1%})")
        
        # ──── Load or Generate EV Profile ────
        logger.info("Processing EV charging profile...")
        
        ev_source_path = Path(source_file) if source_file else None
        profile_shape = None
        if ev_source_path and ev_source_path.is_file():
            logger.info(f"Loading EV profile from {ev_source_path}")
            ev_raw = pd.read_csv(ev_source_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded EV profile shape: {ev_raw.shape}")

            if ev_raw.empty:
                raise ValueError("EV profile source file is empty")

            profile_series = ev_raw.iloc[:, 0].copy()
            if isinstance(profile_series.index, pd.DatetimeIndex):
                profile_series = profile_series.reindex(base_profile.index)
                profile_series = profile_series.interpolate(method="time").ffill().bfill()
            elif len(profile_series) == len(base_profile.index):
                profile_series.index = base_profile.index
            else:
                raise ValueError("EV profile length does not match base profile")

            profile_shape = profile_series
        else:
            logger.warning(f"EV data file not found: {ev_source_path}")
            logger.info("Generating synthetic EV profile based on typical charging patterns")

            n_timesteps = len(base_profile)
            timesteps = np.arange(n_timesteps)
            steps_per_day = max(1, timesteps_per_day)

            # Daily EV driving pattern - realistic multi-modal distribution
            # Represents when EVs consume electricity (driving demand)
            hour_of_day = (timesteps % steps_per_day) * timestep_hours
            
            # Morning commute peak (08:00) - 30% of daily driving
            morning_peak = 0.3 * np.exp(-((hour_of_day - 8) ** 2) / (2 * 1.5 ** 2))
            
            # Evening commute peak (18:00) - 30% of daily driving  
            evening_peak = 0.3 * np.exp(-((hour_of_day - 18) ** 2) / (2 * 1.5 ** 2))
            
            # Midday usage (10:00-16:00) - 40% of daily driving, distributed
            midday_usage = np.where(
                (hour_of_day >= 10) & (hour_of_day <= 16),
                0.4 * (1 - np.abs(hour_of_day - 13) / 3) * 0.5,
                0
            )
            
            # Combined profile (peak/mean ratio ~2.5x, realistic)
            daily_profile = morning_peak + evening_peak + midday_usage

            # Weekly pattern (higher on weekdays)
            day_of_week = (timesteps // steps_per_day) % 7
            weekday_factor = np.where(day_of_week < 5, 1.2, 0.8)  # Higher Mon-Fri

            synthetic_profile = daily_profile * weekday_factor
            profile_shape = pd.Series(synthetic_profile, index=base_profile.index)

            logger.info(f"Generated synthetic EV profile: {profile_shape.shape}")

        shape_energy_mwh = profile_shape.sum() * timestep_hours
        if shape_energy_mwh <= 0:
            raise ValueError("EV profile has zero or negative energy")

        ev_profile_mw = profile_shape / shape_energy_mwh * total_ev_demand_mwh
        ev_profile = pd.DataFrame(
            ev_profile_mw,
            index=base_profile.index,
            columns=['ev_charging_demand_mw']
        )
        
        ev_profile_total_gwh = ev_profile.sum().sum() * timestep_hours / 1000.0
        logger.info(f"EV profile total: {ev_profile_total_gwh:.1f} GWh")
        if fes_fraction is not None and not use_fes_fraction:
            diff = abs(fes_fraction - fraction)
            logger.info(f"FES fraction vs config: {fes_fraction:.2%} vs {fraction:.2%} (diff {diff:.2%})")
        
        # ──── Spatial Allocation ────
        logger.info(f"Allocating EV demand using '{allocation_method}' method...")

        if allocation_method in {"fes", "fes_gsp"} and not snakemake.params.is_historical:
            ev_allocation = _allocate_using_fes_gsp(
                total_ev_demand_gwh,
                base_network,
                ev_profile,
                snakemake.input.fes_data,
                snakemake.params.fes_scenario,
                snakemake.params.modelled_year,
                snakemake.params.network_model,
                snakemake.params.fes_year,
                ['Dem_BB006', 'Dem_BB007'],
                logger
            )
            if ev_allocation is None:
                logger.warning("FES GSP allocation unavailable - falling back to urban_weighted allocation")
                ev_allocation = allocate_urban_weighted(total_ev_demand_gwh, base_network, logger)
                allocation_method = 'urban_weighted'
        else:
            if allocation_method not in ALLOCATION_METHODS:
                logger.warning(f"Unknown allocation method '{allocation_method}', using 'urban_weighted'")
                allocation_method = 'urban_weighted'

            allocator = ALLOCATION_METHODS[allocation_method]
            if allocation_method == 'urban_weighted':
                ev_urban_weight = config.get('urban_weight', 2.0)
                ev_allocation = allocator(total_ev_demand_gwh, base_network, logger, urban_weight=ev_urban_weight)
            else:
                ev_allocation = allocator(total_ev_demand_gwh, base_network, logger)

        if min_gwh_threshold > 0:
            below = ev_allocation < min_gwh_threshold
            dropped = int(below.sum())
            dropped_total = ev_allocation[below].sum()
            remaining_total = ev_allocation[~below].sum()
            if remaining_total <= 0:
                raise ValueError("Min GWh threshold removed all EV allocations")
            scale = total_ev_demand_gwh / remaining_total
            ev_allocation = ev_allocation.copy()
            ev_allocation[below] = 0.0
            ev_allocation[~below] = ev_allocation[~below] * scale
            logger.info(
                f"Dropped {dropped} buses below {min_gwh_threshold} GWh "
                f"(removed {dropped_total:.3f} GWh, rescaled by {scale:.4f})"
            )
        # ──── Validation ────
        logger.info("Validating outputs...")
        
        profile_total = ev_profile.sum().sum() * timestep_hours / 1000.0
        allocation_total = ev_allocation.sum()
        
        tolerance = 0.01
        if abs(profile_total - allocation_total) > tolerance:
            logger.warning(
                f"Energy mismatch! Profile: {profile_total:.3f} GWh, "
                f"Allocation: {allocation_total:.3f} GWh"
            )
        else:
            logger.info("Energy balance check: PASSED ✓")
        
        # ──── Save Outputs ────
        logger.info("Saving outputs...")
        
        ev_profile.to_csv(snakemake.output.profile)
        logger.info(f"Saved EV profile to {snakemake.output.profile}")
        
        ev_allocation_df = pd.DataFrame({
            'bus': ev_allocation.index,
            'ev_charging_demand_gwh': ev_allocation.values
        })
        ev_allocation_df.to_csv(snakemake.output.allocation, index=False)
        logger.info(f"Saved EV allocation to {snakemake.output.allocation}")
        
        # ──── Summary ────
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total EV demand: {total_ev_demand_gwh:.1f} GWh/year")
        logger.info(f"Fraction of base: {fraction:.1%}")
        logger.info(f"Number of buses: {len(ev_allocation)}")
        logger.info(f"Allocation method: {allocation_method}")
        logger.info(f"Min bus demand: {ev_allocation.min():.3f} GWh")
        logger.info(f"Max bus demand: {ev_allocation.max():.3f} GWh")
        logger.info("=" * 80)
        logger.info("EV DISAGGREGATION COMPLETED SUCCESSFULLY ✓")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in EV disaggregation: {e}", exc_info=True)
        raise

