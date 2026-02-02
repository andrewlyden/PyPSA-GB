"""
Integrate Disaggregated Demand Components

This script integrates all disaggregated demand components (heat pumps, EVs, etc.)
back into the PyPSA network. It ensures energy conservation and creates separate
Load components for each demand type.
"""

import pandas as pd
import pypsa
import logging
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _get_timestep_hours(index: pd.DatetimeIndex, default: float = 1.0) -> float:
    if index is None or len(index) < 2:
        return default
    delta_hours = (index[1] - index[0]).total_seconds() / 3600.0
    return delta_hours if delta_hours > 0 else default


def integrate_disaggregated_components(
    n: pypsa.Network,
    base_profile: pd.DataFrame,
    component_data: Dict[str, Dict[str, pd.DataFrame]],
    logger: logging.Logger,
    component_names: Optional[list] = None
) -> Tuple[pypsa.Network, pd.DataFrame, Dict[str, float]]:
    """
    Integrate disaggregated demand components into the network.

    Args:
        n: Base network with aggregate demand already attached
        base_profile: Base demand profile used to compute total demand
        component_data: Dict of component profiles and allocations
        logger: Logger instance
        component_names: Optional ordered list of component names

    Returns:
        (network, summary_df, stats)
    """
    if component_names is None:
        component_names = list(component_data.keys())

    if not component_names:
        logger.info("No disaggregated components provided - skipping integration")
        summary_df = pd.DataFrame(columns=[
            "component", "total_gwh", "fraction_of_base", "num_buses", "num_loads"
        ])
        stats = {
            "total_base_demand_gwh": 0.0,
            "total_component_demand_gwh": 0.0,
            "adjustment_factor": 1.0,
            "adjusted_base_demand_gwh": 0.0,
            "final_total_demand_gwh": 0.0
        }
        return n, summary_df, stats

    timestep_hours = _get_timestep_hours(base_profile.index)
    total_base_demand_gwh = base_profile.sum().sum() * timestep_hours / 1000.0
    logger.info(f"Total base demand: {total_base_demand_gwh:.1f} GWh/year")

    total_component_demand = 0.0
    for component_name in component_names:
        if component_name not in component_data:
            logger.warning(f"Component '{component_name}' enabled but no data provided")
            continue

        data = component_data[component_name]
        profile = data["profile"]
        component_timestep_hours = _get_timestep_hours(profile.index, default=timestep_hours)
        component_total = profile.sum().sum() * component_timestep_hours / 1000.0
        data["total_gwh"] = component_total
        total_component_demand += component_total
        logger.info(f"  {component_name}: {component_total:.1f} GWh/year")

    if total_base_demand_gwh <= 0:
        raise ValueError("Base demand is zero; cannot integrate components")

    adjustment_factor = (total_base_demand_gwh - total_component_demand) / total_base_demand_gwh
    logger.info(f"Base demand adjustment factor: {adjustment_factor:.4f}")

    if adjustment_factor < 0:
        raise ValueError("Component demand exceeds total demand - check disaggregation inputs")

    if adjustment_factor < 0.5:
        logger.warning(f"Components represent {(1 - adjustment_factor):.1%} of demand - this seems high")

    if len(n.loads_t.p_set) > 0:
        n.loads_t.p_set = n.loads_t.p_set * adjustment_factor
        logger.info("Adjusted time-varying loads")

    if "p_set" in n.loads.columns:
        n.loads.p_set = n.loads.p_set * adjustment_factor
        logger.info("Adjusted static loads")

    adjusted_demand = total_base_demand_gwh * adjustment_factor
    logger.info(f"Adjusted base demand: {adjusted_demand:.1f} GWh/year")

    for component_name in component_names:
        if component_name not in component_data:
            continue

        data = component_data[component_name]
        profile = data["profile"]
        allocation = data["allocation"].copy()

        bus_col = "bus" if "bus" in allocation.columns else allocation.columns[0]
        allocation[bus_col] = allocation[bus_col].astype(str)
        demand_col = [c for c in allocation.columns if c != bus_col][0]

        buses_added = 0
        for _, row in allocation.iterrows():
            bus_id = str(row[bus_col])
            annual_demand_gwh = row[demand_col]

            if bus_id not in n.buses.index:
                logger.debug(f"  Skipping {bus_id} - not in network")
                continue

            if annual_demand_gwh < 0.001:
                continue

            load_name = f"{component_name}_{bus_id}"
            try:
                n.add(
                    "Load",
                    load_name,
                    bus=bus_id,
                    carrier=component_name
                )
                buses_added += 1
            except Exception as exc:
                logger.warning(f"  Could not add load {load_name}: {exc}")

        logger.info(f"  Added {buses_added} {component_name} loads to network")

        total_gwh = data.get("total_gwh", None)
        if total_gwh is None or total_gwh <= 0:
            logger.warning(f"  No valid total demand for {component_name}; skipping timeseries")
            continue

        bus_ids = allocation[bus_col].astype(str).tolist()
        profile_bus_cols = [bus for bus in bus_ids if bus in profile.columns]

        if profile_bus_cols:
            for bus_id in profile_bus_cols:
                load_name = f"{component_name}_{bus_id}"
                if load_name in n.loads.index:
                    n.loads_t.p_set[load_name] = profile[bus_id].values

            missing_buses = [bus for bus in bus_ids if bus not in profile.columns]
            if missing_buses:
                bus_fractions = allocation.set_index(bus_col)[demand_col] / total_gwh
                for bus_id in missing_buses:
                    load_name = f"{component_name}_{bus_id}"
                    if load_name not in n.loads.index:
                        continue
                    bus_profile = profile.iloc[:, 0]
                    n.loads_t.p_set[load_name] = (bus_profile * bus_fractions[bus_id]).values
        else:
            bus_fractions = allocation.set_index(bus_col)[demand_col] / total_gwh
            for bus_id, fraction in bus_fractions.items():
                load_name = f"{component_name}_{bus_id}"
                if load_name not in n.loads.index:
                    continue
                bus_profile = profile.iloc[:, 0]
                n.loads_t.p_set[load_name] = (bus_profile * fraction).values

        logger.info(f"  Added timeseries for {component_name}")

    if len(n.loads_t.p_set) > 0:
        final_total_demand = n.loads_t.p_set.sum(axis=1).sum() * timestep_hours / 1000.0
    else:
        final_total_demand = n.loads.p_set.sum() * timestep_hours / 1000.0

    diff = abs(final_total_demand - total_base_demand_gwh)
    tolerance = 0.1
    if diff > tolerance:
        logger.warning(
            f"Energy balance check FAILED: {diff:.1f} GWh > {tolerance:.1f} GWh tolerance"
        )
    else:
        logger.info("Energy balance check: PASSED")

    summary_data = []
    for component_name in component_names:
        if component_name not in component_data:
            continue
        data = component_data[component_name]
        total_gwh = data.get("total_gwh", 0.0)
        summary_data.append({
            "component": component_name,
            "total_gwh": total_gwh,
            "fraction_of_base": total_gwh / total_base_demand_gwh if total_base_demand_gwh else 0.0,
            "num_buses": len(data["allocation"]),
            "num_loads": len([l for l in n.loads.index if l.startswith(component_name)])
        })

    summary_df = pd.DataFrame(summary_data)
    stats = {
        "total_base_demand_gwh": total_base_demand_gwh,
        "total_component_demand_gwh": total_component_demand,
        "adjustment_factor": adjustment_factor,
        "adjusted_base_demand_gwh": adjusted_demand,
        "final_total_demand_gwh": final_total_demand
    }

    return n, summary_df, stats

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
        logger.info("INTEGRATING DISAGGREGATED DEMAND COMPONENTS")
        logger.info("=" * 80)

        logger.info("Loading base demand network...")
        network = load_network(snakemake.input.base_demand, skip_time_series=False, custom_logger=logger)
        base_profile = pd.read_csv(snakemake.input.base_profile, index_col=0, parse_dates=True)
        logger.info(f"Base network: {len(network.buses)} buses, {len(network.loads)} loads")

        component_names = snakemake.params.component_names
        logger.info(f"Loading {len(component_names)} components: {component_names}")

        component_data = {}
        for idx, component_name in enumerate(component_names):
            profile_path = snakemake.input.component_profiles[idx]
            allocation_path = snakemake.input.component_allocations[idx]
            component_data[component_name] = {
                "profile": pd.read_csv(profile_path, index_col=0, parse_dates=True),
                "allocation": pd.read_csv(allocation_path)
            }

        network, summary_df, stats = integrate_disaggregated_components(
            n=network,
            base_profile=base_profile,
            component_data=component_data,
            logger=logger,
            component_names=component_names
        )

        summary_df.to_csv(snakemake.output.component_summary, index=False)
        logger.info(f"Saved component summary to {snakemake.output.component_summary}")

        logger.info("Saving final network...")
        network.export_to_netcdf(snakemake.output.final_network)
        logger.info(f"Saved final network to {snakemake.output.final_network}")

        logger.info("=" * 80)
        logger.info("INTEGRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Components integrated: {len(summary_df)}")
        if stats["total_base_demand_gwh"] > 0:
            logger.info(
                f"Adjusted base demand: {stats['adjusted_base_demand_gwh']:.1f} GWh "
                f"({stats['adjustment_factor']:.1%})"
            )
            logger.info(f"Final total demand: {stats['final_total_demand_gwh']:.1f} GWh")
        logger.info(f"Total loads in network: {len(network.loads)}")
        logger.info("=" * 80)
        logger.info("INTEGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in integration: {e}", exc_info=True)
        raise
