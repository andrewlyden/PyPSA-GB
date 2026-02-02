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


def allocate_urban_weighted(total_gwh: float, base_network: pypsa.Network, logger) -> pd.Series:
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
    urban_weight = 2.0
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
# EV Flexibility Functions
# ──────────────────────────────────────────────────────────────────────────────

# Default EV parameters
DEFAULT_BATTERY_CAPACITY_KWH = 60.0  # Average EV battery size
DEFAULT_CHARGER_POWER_KW = 7.0       # Typical home charger
DEFAULT_CHARGE_EFFICIENCY = 0.90     # Charging efficiency
DEFAULT_DISCHARGE_EFFICIENCY = 0.90  # V2G discharge efficiency
# Flexibility participation - not all EVs participate in smart charging
# This scales down the modeled storage to represent available flexibility
DEFAULT_FLEXIBILITY_PARTICIPATION = 0.10  # 10% of fleet participates in flexibility


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

    go_config = config.get('go', {})
    window = go_config.get('window', ['00:00', '04:00'])
    window_start = int(window[0].split(':')[0])
    window_end = int(window[1].split(':')[0])

    int_config = config.get('int', {})
    charger_power_kw = int_config.get('charger_power_kw', DEFAULT_CHARGER_POWER_KW)
    min_soc = int_config.get('min_soc', 0.20)

    if logger:
        logger.info(f"GO window: {window_start:02d}:00 - {window_end:02d}:00")
        logger.info(f"Charger power: {charger_power_kw} kW")

    # Create charging window profile
    charging_window = create_go_tariff_window(n.snapshots, window_start, window_end)

    for bus in buses:
        if bus not in ev_demand_mw.columns:
            continue

        battery_bus = f"{bus} EV battery"
        store_name = f"{bus} EV fleet battery"
        charger_name = f"{bus} EV charger"
        demand_name = f"{bus} EV driving demand"

        # Estimate fleet size from demand
        daily_demand_mwh = ev_demand_mw[bus].sum() / 365
        n_vehicles = max(1, int(daily_demand_mwh * 1000 / 10))  # ~10 kWh/day per vehicle

        # Scale by participation rate - only a fraction of fleet participates in flexibility
        n_flex_vehicles = max(1, int(n_vehicles * DEFAULT_FLEXIBILITY_PARTICIPATION))

        # Add EV battery bus
        if battery_bus not in n.buses.index:
            bus_kwargs = {"carrier": "EV battery"}
            bus_kwargs.update(_bus_coords_kwargs(n, bus))
            n.add("Bus", battery_bus, **bus_kwargs)

        # Add EV fleet battery (Store) - scaled by participation
        fleet_capacity_mwh = n_flex_vehicles * DEFAULT_BATTERY_CAPACITY_KWH / 1000
        n.add("Store",
              store_name,
              bus=battery_bus,
              carrier="EV battery",
              e_nom=fleet_capacity_mwh,
              e_nom_extendable=False,
              e_cyclic=True,
              e_min_pu=min_soc)  # Minimum SOC constraint

        # Add charger (Link with time-limited availability) - scaled by participation
        charger_power_mw = n_flex_vehicles * charger_power_kw / 1000
        n.add("Link",
              charger_name,
              bus0=bus,  # Grid
              bus1=battery_bus,  # Battery
              carrier="EV charger",
              efficiency=DEFAULT_CHARGE_EFFICIENCY,
              p_nom=charger_power_mw,
              p_nom_extendable=False,
              p_max_pu=charging_window)  # Only charge during window

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

    int_config = config.get('int', {})
    charger_power_kw = int_config.get('charger_power_kw', DEFAULT_CHARGER_POWER_KW)
    min_soc = int_config.get('min_soc', 0.20)
    target_departure_soc = int_config.get('target_departure_soc', 0.80)

    if logger:
        logger.info(f"Charger power: {charger_power_kw} kW")
        logger.info(f"Min SOC: {min_soc*100:.0f}%, Target departure: {target_departure_soc*100:.0f}%")

    for bus in buses:
        if bus not in ev_demand_mw.columns:
            continue

        battery_bus = f"{bus} EV battery"
        store_name = f"{bus} EV fleet battery"
        charger_name = f"{bus} EV charger"
        demand_name = f"{bus} EV driving demand"

        # Estimate fleet size from demand
        daily_demand_mwh = ev_demand_mw[bus].sum() / 365
        n_vehicles = max(1, int(daily_demand_mwh * 1000 / 10))

        # Scale by participation rate - only a fraction of fleet participates in flexibility
        n_flex_vehicles = max(1, int(n_vehicles * DEFAULT_FLEXIBILITY_PARTICIPATION))

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

        # Add EV fleet battery (Store) - scaled by participation
        fleet_capacity_mwh = n_flex_vehicles * DEFAULT_BATTERY_CAPACITY_KWH / 1000
        n.add("Store",
              store_name,
              bus=battery_bus,
              carrier="EV battery",
              e_nom=fleet_capacity_mwh,
              e_nom_extendable=False,
              e_cyclic=True,
              e_min_pu=e_min_pu)  # Time-varying minimum SOC

        # Add charger (Link with availability constraint) - scaled by participation
        charger_power_mw = n_flex_vehicles * charger_power_kw / 1000
        n.add("Link",
              charger_name,
              bus0=bus,  # Grid
              bus1=battery_bus,  # Battery
              carrier="EV charger",
              efficiency=DEFAULT_CHARGE_EFFICIENCY,
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
               logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Add EV Vehicle-to-Grid (V2G) bidirectional flexibility.

    V2G allows EVs to discharge back to the grid, providing additional
    flexibility. Includes separate charge and discharge links with
    different efficiencies and constraints.

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

    Returns:
        Network with V2G components
    """
    if logger:
        logger.info("Adding EV V2G flexibility (bidirectional)...")

    v2g_config = config.get('v2g', {})
    participation_rate = v2g_config.get('participation_rate', 0.30)
    discharge_efficiency = v2g_config.get('discharge_efficiency', DEFAULT_DISCHARGE_EFFICIENCY)
    max_discharge_soc = v2g_config.get('max_discharge_soc', 0.80)
    degradation_cost = v2g_config.get('degradation_cost_per_mwh', 50.0)

    int_config = config.get('int', {})
    charger_power_kw = int_config.get('charger_power_kw', DEFAULT_CHARGER_POWER_KW)
    min_soc = int_config.get('min_soc', 0.20)

    if logger:
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
        daily_demand_mwh = ev_demand_mw[bus].sum() / 365
        n_vehicles = max(1, int(daily_demand_mwh * 1000 / 10))

        # Scale by participation rate - only a fraction of fleet participates in flexibility
        n_flex_vehicles = max(1, int(n_vehicles * DEFAULT_FLEXIBILITY_PARTICIPATION))
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

        # Add EV fleet battery (Store) - scaled by participation
        fleet_capacity_mwh = n_flex_vehicles * DEFAULT_BATTERY_CAPACITY_KWH / 1000
        n.add("Store",
              store_name,
              bus=battery_bus,
              carrier="EV battery",
              e_nom=fleet_capacity_mwh,
              e_nom_extendable=False,
              e_cyclic=True,
              e_min_pu=e_min_pu)

        # Add charger (Grid -> Battery) - scaled by participation
        charger_power_mw = n_flex_vehicles * charger_power_kw / 1000
        n.add("Link",
              charger_name,
              bus0=bus,
              bus1=battery_bus,
              carrier="EV charger",
              efficiency=DEFAULT_CHARGE_EFFICIENCY,
              p_nom=charger_power_mw,
              p_nom_extendable=False,
              p_max_pu=avail)

        # Add V2G discharge link (Battery -> Grid)
        # Only V2G-capable vehicles can discharge
        v2g_power_mw = n_v2g_vehicles * charger_power_kw / 1000
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
                       logger: Optional[logging.Logger] = None) -> pypsa.Network:
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
                          dsm_profile, ev_config, logger)
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

            # Daily charging pattern (peaks in evening ~18:00-22:00)
            hour_of_day = (timesteps % steps_per_day) * timestep_hours
            evening_peak = np.exp(-((hour_of_day - 20) ** 2) / (2 * 2 ** 2))  # Peak at 20:00

            # Weekly pattern (higher on weekdays)
            day_of_week = (timesteps // steps_per_day) % 7
            weekday_factor = np.where(day_of_week < 5, 1.2, 0.8)  # Higher Mon-Fri

            synthetic_profile = evening_peak * weekday_factor
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

