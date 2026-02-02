"""
Integrate Demand-Side Flexibility into PyPSA Network

This script is the main integration point for adding all demand-side flexibility
components to a PyPSA network. It coordinates:
- Heat pump flexibility (TANK/COSY modes)
- Electric vehicle flexibility (GO/INT/V2G tariffs)
- Event-based demand response (Saving Sessions)

The script reads the flexibility configuration and conditionally adds the
appropriate PyPSA components (Stores, Links, Generators) for each enabled
flexibility type.
"""

import logging
import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import sys
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging

# Import flexibility modules
from scripts.demand.heat_pumps import add_heat_pump_flexibility
from scripts.demand.electric_vehicles import add_ev_flexibility
from scripts.demand.event_flex import add_event_flexibility


def _shift_index_to_year(index: pd.DatetimeIndex, target_year: int) -> pd.DatetimeIndex:
    shifted = []
    for ts in index:
        try:
            shifted.append(ts.replace(year=target_year))
        except ValueError:
            # Handle Feb 29 for non-leap target years.
            shifted.append(ts.replace(year=target_year, day=28))
    return pd.DatetimeIndex(shifted)


def _aggregate_loads_to_buses(n: pypsa.Network,
                              logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Aggregate load-level timeseries to bus-level demand."""
    if n.loads.empty or n.loads_t.p_set.empty:
        return pd.DataFrame()

    load_to_bus = n.loads["bus"]
    load_cols = [c for c in n.loads_t.p_set.columns if c in load_to_bus.index]
    if not load_cols:
        if logger:
            logger.warning("No load columns align with loads index for aggregation")
        return pd.DataFrame()

    p_set = n.loads_t.p_set[load_cols]
    bus_map = load_to_bus.loc[load_cols]
    bus_demand = p_set.T.groupby(bus_map).sum().T
    return bus_demand


def _build_static_bus_demand(n: pypsa.Network) -> pd.DataFrame:
    """Fallback bus-level demand from static loads."""
    if n.loads.empty:
        return pd.DataFrame()
    by_bus = n.loads.groupby("bus")["p_set"].sum()
    demand = pd.DataFrame(index=n.snapshots)
    for bus, value in by_bus.items():
        demand[bus] = value
    return demand


def _scale_loads_by_carrier(n: pypsa.Network,
                            carrier: str,
                            scale: float,
                            logger: Optional[logging.Logger] = None) -> int:
    """Scale loads matching carrier by a factor; returns count scaled."""
    if n.loads.empty:
        return 0

    if scale < 0:
        raise ValueError(f"Load scale must be non-negative (got {scale})")

    load_mask = n.loads["carrier"] == carrier
    load_names = n.loads.index[load_mask]

    if load_names.empty:
        return 0

    if hasattr(n, "loads_t") and hasattr(n.loads_t, "p_set") and not n.loads_t.p_set.empty:
        cols = [c for c in load_names if c in n.loads_t.p_set.columns]
        if cols:
            n.loads_t.p_set[cols] = n.loads_t.p_set[cols] * scale

    if "p_set" in n.loads.columns:
        n.loads.loc[load_names, "p_set"] = n.loads.loc[load_names, "p_set"] * scale

    if logger:
        logger.info(f"Scaled {len(load_names)} '{carrier}' loads by {scale:.2f}")

    return len(load_names)


def _expand_profile_to_buses(profile_df: Optional[pd.DataFrame],
                             allocation_df: Optional[pd.DataFrame],
                             buses: pd.Index,
                             label: str,
                             logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Expand a total profile into bus-level profiles using allocation data."""
    if profile_df is None or profile_df.empty:
        return None

    bus_cols = [c for c in profile_df.columns if c in buses]
    if bus_cols:
        return profile_df[bus_cols]

    if allocation_df is None or allocation_df.empty:
        if logger:
            logger.warning(f"{label} profile has no bus columns and no allocation data")
        return None

    bus_col = "bus" if "bus" in allocation_df.columns else allocation_df.columns[0]
    demand_cols = [c for c in allocation_df.columns if c != bus_col]
    if not demand_cols:
        if logger:
            logger.warning(f"{label} allocation data missing demand column")
        return None

    demand_col = demand_cols[0]
    alloc = allocation_df[[bus_col, demand_col]].copy()
    alloc = alloc[alloc[bus_col].isin(buses)]
    alloc = alloc[alloc[demand_col] > 0]
    total = alloc[demand_col].sum()
    if total <= 0:
        if logger:
            logger.warning(f"{label} allocation total is zero")
        return None

    fractions = alloc.set_index(bus_col)[demand_col] / total
    if logger:
        logger.info(f"{label} allocation nonzero buses: {len(fractions)}")
    total_profile = profile_df.iloc[:, 0]
    expanded = {
        bus: total_profile * frac for bus, frac in fractions.items()
    }
    return pd.DataFrame(expanded, index=profile_df.index)


def _load_cop_profile(path: str,
                      snapshots: pd.DatetimeIndex = None,
                      logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Load COP profile from CSV or NetCDF; returns time-indexed DataFrame.

    If snapshots is provided, reindex the COP to match the network snapshots.
    """
    if not path:
        return None

    if not Path(path).exists():
        raise FileNotFoundError(f"COP profile not found: {path}")

    cop_df = None

    if path.endswith(".csv"):
        cop_df = pd.read_csv(path, index_col=0, parse_dates=True)

    elif path.endswith(".nc"):
        import xarray as xr

        ds = xr.open_dataset(path)
        if "cop" not in ds:
            raise ValueError("COP NetCDF missing 'cop' variable")
        cop = ds["cop"]
        # Average spatially if needed to get a single timeseries.
        for dim in [d for d in cop.dims if d != "time"]:
            cop = cop.mean(dim=dim)
        cop_df = cop.to_dataframe(name="cop").reset_index()
        cop_df = cop_df.set_index("time")[["cop"]]
    else:
        raise ValueError(f"Unsupported COP profile format: {path}")

    # Align with network snapshots if provided
    if cop_df is not None and snapshots is not None:
        if logger:
            logger.info(f"Aligning COP to {len(snapshots)} snapshots")

        if not isinstance(cop_df.index, pd.DatetimeIndex):
            raise ValueError("COP profile index must be datetime for alignment")

        target_year = int(snapshots[0].year)
        cop_df = cop_df.copy()
        cop_df.index = _shift_index_to_year(cop_df.index, target_year)
        cop_df = cop_df.sort_index()
        if cop_df.index.has_duplicates:
            if logger:
                logger.warning("COP profile has duplicate timestamps; averaging duplicates")
            cop_df = cop_df.groupby(cop_df.index).mean()

        # Resample to match snapshot frequency before reindexing.
        # This preserves temporal variation when source and target have different
        # lengths (e.g. leap year cutout → non-leap year snapshots).
        snap_freq = pd.infer_freq(snapshots)
        if snap_freq:
            cop_df = cop_df.resample(snap_freq).mean()

        cop_df = cop_df.reindex(snapshots)
        cop_df = cop_df.interpolate(method="time").ffill().bfill()

        if cop_df.isna().any().any():
            src_range = f"{cop_df.index.min()} to {cop_df.index.max()}"
            snap_range = f"{snapshots.min()} to {snapshots.max()}"
            raise ValueError(
                "COP profile does not cover all snapshots after alignment "
                f"(COP range: {src_range}, snapshots range: {snap_range})"
            )

        if logger:
            logger.info(f"  COP after alignment: {cop_df.iloc[:, 0].min():.2f} to {cop_df.iloc[:, 0].max():.2f}")

    return cop_df


# ──────────────────────────────────────────────────────────────────────────────
# Main Integration Function
# ──────────────────────────────────────────────────────────────────────────────

def integrate_demand_flexibility(n: pypsa.Network,
                                 flex_config: Dict[str, Any],
                                 hp_demand_mw: Optional[pd.DataFrame] = None,
                                 hp_cop_profile: Optional[pd.DataFrame] = None,
                                 hp_allocation: Optional[pd.DataFrame] = None,
                                 ev_demand_mw: Optional[pd.DataFrame] = None,
                                 ev_availability: Optional[pd.DataFrame] = None,
                                 ev_dsm: Optional[pd.DataFrame] = None,
                                 ev_allocation: Optional[pd.DataFrame] = None,
                                 base_demand_mw: Optional[pd.DataFrame] = None,
                                 add_load_shedding: bool = True,
                                 logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Integrate all enabled demand-side flexibility into the network.

    This function orchestrates the addition of flexibility components based
    on the configuration. Each flexibility type can be independently enabled.

    Args:
        n: PyPSA network
        flex_config: Flexibility configuration dictionary
        hp_demand_mw: Heat pump demand profiles (MW)
        hp_cop_profile: Heat pump COP profiles
        ev_demand_mw: EV charging demand (MW)
        ev_availability: EV plugged-in availability (0-1)
        ev_dsm: EV minimum SOC requirements (0-1)
        base_demand_mw: Base electricity demand for event flex scaling
        add_load_shedding: Add load shedding generators for flexibility buses
        logger: Logger instance

    Returns:
        Network with flexibility components added
    """
    if logger:
        logger.info("=" * 80)
        logger.info("DEMAND-SIDE FLEXIBILITY INTEGRATION")
        logger.info("=" * 80)

    # Check if flexibility is enabled at all
    if not flex_config.get('enabled', False):
        if logger:
            logger.info("Demand flexibility is DISABLED in configuration")
            logger.info("Returning network unchanged")
        return n

    if logger:
        logger.info("Demand flexibility is ENABLED")

    # Track what was added
    components_added = []

    # ──── Heat Pump Flexibility ────
    hp_config = flex_config.get('heat_pumps', {})
    if hp_config.get('enabled', False):
        if logger:
            logger.info("")
            logger.info("─" * 40)
            logger.info("HEAT PUMP FLEXIBILITY")
            logger.info("─" * 40)

        if hp_demand_mw is not None:
            hp_demand_mw = _expand_profile_to_buses(
                hp_demand_mw, hp_allocation, n.buses.index, "Heat pump", logger
            )
        if hp_demand_mw is not None:
            # Get buses with heat pump demand
            hp_buses = [col for col in hp_demand_mw.columns if col in n.buses.index]

            flex_share = float(hp_config.get('flex_share', 1.0))
            if flex_share < 0 or flex_share > 1:
                raise ValueError(f"Heat pump flex_share must be between 0 and 1 (got {flex_share})")

            if flex_share == 0:
                if logger:
                    logger.info("Heat pump flex_share is 0 - skipping flexibility")
                components_added.append("Heat pumps (0% flexible)")
            else:
                # Reduce existing heat pump loads to avoid double-counting
                scaled = _scale_loads_by_carrier(n, "heat_pumps", 1.0 - flex_share, logger)
                if scaled == 0 and logger:
                    logger.warning("No 'heat_pumps' loads found to scale; check disaggregation inputs")

                if hp_cop_profile is None or hp_cop_profile.empty:
                    raise ValueError("Heat pump flexibility enabled but COP profile is missing")

                n = add_heat_pump_flexibility(
                    n=n,
                    buses=hp_buses,
                    hp_demand_mw=hp_demand_mw * flex_share,
                    cop_profile=hp_cop_profile,
                    flex_config=flex_config,
                    logger=logger
                )
                components_added.append(f"Heat pumps ({hp_config.get('mode', 'MIXED')}, {flex_share:.0%} flexible)")
        else:
            if logger:
                logger.warning("Heat pump flexibility enabled but no demand data provided")
    else:
        if logger:
            logger.info("Heat pump flexibility: DISABLED")

    # ──── Electric Vehicle Flexibility ────
    ev_config = flex_config.get('electric_vehicles', {})
    if ev_config.get('enabled', False):
        if logger:
            logger.info("")
            logger.info("─" * 40)
            logger.info("ELECTRIC VEHICLE FLEXIBILITY")
            logger.info("─" * 40)

        if ev_demand_mw is not None:
            ev_demand_mw = _expand_profile_to_buses(
                ev_demand_mw, ev_allocation, n.buses.index, "EV", logger
            )
        if ev_demand_mw is not None:
            # Get buses with EV demand
            ev_buses = [col for col in ev_demand_mw.columns if col in n.buses.index]

            # Create default profiles if not provided
            if ev_availability is None:
                if logger:
                    logger.warning("No availability profile provided, using default 0.8")
                ev_availability = pd.DataFrame(
                    0.8,
                    index=n.snapshots,
                    columns=ev_buses
                )

            if ev_dsm is None:
                if logger:
                    logger.warning("No DSM profile provided, using default min_soc=0.2")
                ev_dsm = pd.DataFrame(
                    0.2,
                    index=n.snapshots,
                    columns=ev_buses
                )

            n = add_ev_flexibility(
                n=n,
                buses=ev_buses,
                ev_demand_mw=ev_demand_mw,
                availability_profile=ev_availability,
                dsm_profile=ev_dsm,
                flex_config=flex_config,
                logger=logger
            )
            components_added.append(f"EVs ({ev_config.get('tariff', 'INT')})")
        else:
            if logger:
                logger.warning("EV flexibility enabled but no demand data provided")
    else:
        if logger:
            logger.info("EV flexibility: DISABLED")

    # ──── Event Response Flexibility ────
    event_config = flex_config.get('event_response', {})
    if event_config.get('enabled', False):
        if logger:
            logger.info("")
            logger.info("─" * 40)
            logger.info("EVENT RESPONSE FLEXIBILITY")
            logger.info("─" * 40)

        # Use base demand for scaling event response
        if base_demand_mw is None:
            # Try to get bus-level demand from network loads
            if len(n.loads_t.p_set) > 0:
                base_demand_mw = _aggregate_loads_to_buses(n, logger)
            else:
                base_demand_mw = _build_static_bus_demand(n)

            if base_demand_mw.empty and logger:
                logger.warning("No base demand data for event response scaling")

        if not base_demand_mw.empty:
            n = add_event_flexibility(
                n=n,
                base_demand_mw=base_demand_mw,
                config=event_config,
                logger=logger
            )
            components_added.append(f"Event response ({event_config.get('mode', 'regular')})")
        else:
            if logger:
                logger.warning("Event response enabled but no base demand data")
    else:
        if logger:
            logger.info("Event response: DISABLED")

    # ──── Add missing carriers for flexibility components ────
    flex_carriers = [
        ('heat', 0.0, 'Heat'),
        ('hot water', 0.0, 'Hot Water'),
        ('hot water demand', 0.0, 'Hot Water Demand'),
        ('heat pump', 0.0, 'Heat Pump'),
        ('thermal inertia', 0.0, 'Thermal Inertia'),
        ('space heating', 0.0, 'Space Heating'),
        ('thermal demand', 0.0, 'Thermal Demand'),
        ('EV battery', 0.0, 'EV Battery'),
        ('EV charger', 0.0, 'EV Charger'),
        ('EV driving', 0.0, 'EV Driving'),
        ('EV driving demand', 0.0, 'EV Driving Demand'),
        ('V2G', 0.0, 'Vehicle-to-Grid'),
    ]
    for carrier, co2, nice_name in flex_carriers:
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier, co2_emissions=co2, nice_name=nice_name)

    # ──── Add load shedding for new flexibility buses ────
    # Find new buses that have loads but no load shedding
    if add_load_shedding:
        flex_buses_with_loads = set()
        for load_name in n.loads.index:
            bus = n.loads.loc[load_name, 'bus']
            # Check if this bus has a load shedding generator
            if bus not in flex_buses_with_loads:
                has_load_shed = any(
                    'load_shedding' in g or 'load shedding' in g.lower()
                    for g in n.generators[n.generators.bus == bus].index
                )
                if not has_load_shed:
                    flex_buses_with_loads.add(bus)

        if flex_buses_with_loads and logger:
            logger.info(f"Adding load shedding to {len(flex_buses_with_loads)} flexibility buses")

        for bus in flex_buses_with_loads:
            # Get the load on this bus to set appropriate shedding capacity
            bus_loads = n.loads[n.loads.bus == bus]
            if bus_loads.empty:
                continue

            # Get peak demand for this bus
            load_names = bus_loads.index.tolist()
            if load_names[0] in n.loads_t.p_set.columns:
                peak_demand = n.loads_t.p_set[load_names].sum(axis=1).max()
            else:
                peak_demand = bus_loads.p_set.sum()

            if peak_demand > 0:
                n.add("Generator",
                      f"{bus} load_shedding",
                      bus=bus,
                      carrier="load_shedding",
                      p_nom=peak_demand * 1.5,  # 150% of peak for margin
                      marginal_cost=6000.0)  # VOLL
    elif logger:
        logger.info("Skipping load shedding additions (handled later in workflow)")

    # ──── Summary ────
    if logger:
        logger.info("")
        logger.info("=" * 80)
        logger.info("INTEGRATION SUMMARY")
        logger.info("=" * 80)

        if components_added:
            logger.info(f"Components added: {len(components_added)}")
            for comp in components_added:
                logger.info(f"  - {comp}")
        else:
            logger.info("No flexibility components were added")

        # Network statistics
        logger.info("")
        logger.info("Network statistics after integration:")
        logger.info(f"  Buses: {len(n.buses)}")
        logger.info(f"  Loads: {len(n.loads)}")
        logger.info(f"  Generators: {len(n.generators)}")
        logger.info(f"  Links: {len(n.links)}")
        logger.info(f"  Stores: {len(n.stores)}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("DEMAND FLEXIBILITY INTEGRATION COMPLETED")
        logger.info("=" * 80)

    return n


def generate_integration_summary(n: pypsa.Network,
                                 logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Generate a summary of flexibility components in the network.

    Args:
        n: PyPSA network with flexibility components
        logger: Logger instance

    Returns:
        DataFrame summarizing flexibility components by type
    """
    summary_data = []

    # Heat pump components
    hp_stores = n.stores[n.stores.carrier.isin(['hot water', 'thermal inertia'])]
    hp_links = n.links[n.links.carrier == 'heat pump']
    if len(hp_stores) > 0:
        summary_data.append({
            'Component Type': 'Heat Pump Flexibility',
            'Count': len(hp_stores),
            'Total Capacity': f"{hp_stores.e_nom.sum():.2f} MWh (thermal)",
            'Carrier': hp_stores.carrier.unique().tolist()
        })

    # EV components
    ev_stores = n.stores[n.stores.carrier == 'EV battery']
    ev_chargers = n.links[n.links.carrier == 'EV charger']
    v2g_links = n.links[n.links.carrier == 'V2G']
    if len(ev_stores) > 0:
        summary_data.append({
            'Component Type': 'EV Battery Storage',
            'Count': len(ev_stores),
            'Total Capacity': f"{ev_stores.e_nom.sum():.2f} MWh",
            'Carrier': 'EV battery'
        })
    if len(ev_chargers) > 0:
        summary_data.append({
            'Component Type': 'EV Chargers',
            'Count': len(ev_chargers),
            'Total Capacity': f"{ev_chargers.p_nom.sum():.2f} MW",
            'Carrier': 'EV charger'
        })
    if len(v2g_links) > 0:
        summary_data.append({
            'Component Type': 'V2G Discharge',
            'Count': len(v2g_links),
            'Total Capacity': f"{v2g_links.p_nom.sum():.2f} MW",
            'Carrier': 'V2G'
        })

    # Event response
    dr_gens = n.generators[n.generators.carrier == 'demand response']
    if len(dr_gens) > 0:
        summary_data.append({
            'Component Type': 'Demand Response',
            'Count': len(dr_gens),
            'Total Capacity': f"{dr_gens.p_nom.sum():.2f} MW",
            'Carrier': 'demand response'
        })

    summary_df = pd.DataFrame(summary_data)

    if logger:
        logger.info("Flexibility Components Summary:")
        logger.info(summary_df.to_string())

    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# Snakemake Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(
        log_path=snakemake.log[0],
        log_level="INFO"
    )

    try:
        logger.info("=" * 80)
        logger.info("DEMAND FLEXIBILITY INTEGRATION")
        logger.info("=" * 80)

        # Load network
        n = pypsa.Network(snakemake.input.network)
        logger.info(f"Loaded network: {len(n.buses)} buses, {len(n.snapshots)} snapshots")

        # Get flexibility configuration
        flex_config = snakemake.params.get('flex_config', {})

        # Load optional input data
        hp_demand_mw = None
        hp_cop_profile = None
        hp_allocation = None
        ev_demand_mw = None
        ev_availability = None
        ev_dsm = None
        ev_allocation = None

        # Heat pump data
        if hasattr(snakemake.input, 'hp_demand') and snakemake.input.hp_demand:
            hp_demand_mw = pd.read_csv(snakemake.input.hp_demand, index_col=0, parse_dates=True)
            logger.info(f"Loaded HP demand: {hp_demand_mw.shape}")

        if hasattr(snakemake.input, 'hp_allocation') and snakemake.input.hp_allocation:
            hp_allocation = pd.read_csv(snakemake.input.hp_allocation)
            logger.info(f"Loaded HP allocation: {hp_allocation.shape}")

        cop_path = None
        if hasattr(snakemake.input, 'hp_cop') and snakemake.input.hp_cop:
            cop_path = snakemake.input.hp_cop
        elif hasattr(snakemake.input, 'cop_ashp') and snakemake.input.cop_ashp:
            cop_path = snakemake.input.cop_ashp

        if cop_path:
            hp_cop_profile = _load_cop_profile(cop_path, snapshots=n.snapshots, logger=logger)
            if hp_cop_profile is not None:
                logger.info(f"Loaded HP COP: {hp_cop_profile.shape}")

        # EV data
        if hasattr(snakemake.input, 'ev_demand') and snakemake.input.ev_demand:
            ev_demand_mw = pd.read_csv(snakemake.input.ev_demand, index_col=0, parse_dates=True)
            logger.info(f"Loaded EV demand: {ev_demand_mw.shape}")

        if hasattr(snakemake.input, 'ev_availability') and snakemake.input.ev_availability:
            ev_availability = pd.read_csv(snakemake.input.ev_availability, index_col=0, parse_dates=True)
            logger.info(f"Loaded EV availability: {ev_availability.shape}")

        if hasattr(snakemake.input, 'ev_dsm') and snakemake.input.ev_dsm:
            ev_dsm = pd.read_csv(snakemake.input.ev_dsm, index_col=0, parse_dates=True)
            logger.info(f"Loaded EV DSM: {ev_dsm.shape}")
        
        if hasattr(snakemake.input, 'ev_allocation') and snakemake.input.ev_allocation:
            ev_allocation = pd.read_csv(snakemake.input.ev_allocation)
            logger.info(f"Loaded EV allocation: {ev_allocation.shape}")

        # Integrate flexibility
        n = integrate_demand_flexibility(
            n=n,
            flex_config=flex_config,
            hp_demand_mw=hp_demand_mw,
            hp_cop_profile=hp_cop_profile,
            hp_allocation=hp_allocation,
            ev_demand_mw=ev_demand_mw,
            ev_availability=ev_availability,
            ev_dsm=ev_dsm,
            ev_allocation=ev_allocation,
            logger=logger
        )

        # Generate summary
        summary = generate_integration_summary(n, logger)
        summary.to_csv(snakemake.output.integration_summary, index=False)
        logger.info(f"Saved integration summary to {snakemake.output.integration_summary}")

        # Save network
        n.export_to_netcdf(snakemake.output.network)
        logger.info(f"Saved network to {snakemake.output.network}")

        logger.info("=" * 80)
        logger.info("INTEGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in demand flexibility integration: {e}", exc_info=True)
        raise
