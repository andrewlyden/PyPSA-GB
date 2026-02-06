"""
Finalize Demand Integration

Orchestrates:
1. Load base demand network
2. Integrate disaggregated components (if enabled)
3. Add demand-side flexibility (if enabled)
4. Output complete demand-side network
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Optional

import pandas as pd
import pypsa

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network, save_network
from scripts.demand.integrate import integrate_disaggregated_components
from scripts.demand.add_demand_flexibility import (
    integrate_demand_flexibility,
    generate_integration_summary,
    _load_cop_profile
)
from scripts.demand.electric_vehicles import load_fes_v2g_capacity


def _extract_component_names(disaggregation_config: Dict) -> list:
    components = disaggregation_config.get("components", [])
    names = []
    for comp in components:
        if isinstance(comp, str):
            names.append(comp)
        elif isinstance(comp, dict):
            name = comp.get("name")
            if name:
                names.append(name)
    return names


def _load_optional_csv(path: Optional[str], **kwargs) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if isinstance(path, (list, tuple)):
        if not path:
            return None
        path = path[0]
    if not Path(path).exists():
        return None
    return pd.read_csv(path, **kwargs)


def _build_summary(
    n: pypsa.Network,
    disagg_summary: pd.DataFrame,
    flex_summary: pd.DataFrame,
    stats: Dict[str, float]
) -> pd.DataFrame:
    rows = [
        {"section": "network", "metric": "buses", "value": len(n.buses)},
        {"section": "network", "metric": "loads", "value": len(n.loads)},
        {"section": "network", "metric": "generators", "value": len(n.generators)},
        {"section": "network", "metric": "links", "value": len(n.links)},
        {"section": "network", "metric": "stores", "value": len(n.stores)},
        {"section": "network", "metric": "storage_units", "value": len(n.storage_units)},
    ]

    if stats:
        for key, value in stats.items():
            rows.append({"section": "demand", "metric": key, "value": value})

    if not disagg_summary.empty:
        for _, row in disagg_summary.iterrows():
            rows.append({
                "section": "disaggregation",
                "metric": row["component"],
                "value": row["total_gwh"]
            })

    if not flex_summary.empty:
        for _, row in flex_summary.iterrows():
            rows.append({
                "section": "flexibility",
                "metric": row.get("Component Type", "unknown"),
                "value": row.get("Count", 0)
            })

    load_carriers = n.loads["carrier"].value_counts() if not n.loads.empty else {}
    for carrier, count in load_carriers.items():
        rows.append({"section": "load_carriers", "metric": carrier, "value": int(count)})

    return pd.DataFrame(rows)


def finalize_demand_integration(
    n: pypsa.Network,
    base_profile: pd.DataFrame,
    disaggregation_config: Dict,
    flexibility_config: Dict,
    component_data: Dict[str, Dict[str, pd.DataFrame]],
    hp_cop_profile: Optional[pd.DataFrame],
    ev_availability: Optional[pd.DataFrame],
    ev_dsm: Optional[pd.DataFrame],
    logger: logging.Logger,
    fes_v2g_capacity: Optional[pd.Series] = None
) -> tuple[pypsa.Network, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    logger.info("=" * 80)
    logger.info("FINALIZING DEMAND INTEGRATION")
    logger.info("=" * 80)

    disagg_summary = pd.DataFrame()
    stats: Dict[str, float] = {}

    if disaggregation_config.get("enabled", False):
        logger.info("Integrating disaggregated demand components")
        component_names = _extract_component_names(disaggregation_config)
        n, disagg_summary, stats = integrate_disaggregated_components(
            n=n,
            base_profile=base_profile,
            component_data=component_data,
            logger=logger,
            component_names=component_names
        )
    else:
        logger.info("Disaggregation disabled - using baseline demand only")

    if flexibility_config.get("enabled", False):
        logger.info("Adding demand flexibility components")
        hp_data = component_data.get("heat_pumps", {})
        ev_data = component_data.get("electric_vehicles", {})

        # Get FES parameters for MIXED mode
        fes_path = snakemake.input.fes_data if hasattr(snakemake.input, 'fes_data') else None
        fes_scenario = snakemake.params.get('fes_scenario')
        modelled_year = snakemake.params.get('modelled_year')
        
        n = integrate_demand_flexibility(
            n=n,
            flex_config=flexibility_config,
            hp_demand_mw=hp_data.get("profile"),
            hp_cop_profile=hp_cop_profile,
            hp_allocation=hp_data.get("allocation"),
            ev_demand_mw=ev_data.get("profile"),
            ev_availability=ev_availability,
            ev_dsm=ev_dsm,
            ev_allocation=ev_data.get("allocation"),
            fes_v2g_capacity=fes_v2g_capacity,
            fes_path=fes_path,
            fes_scenario=fes_scenario,
            modelled_year=modelled_year,
            add_load_shedding=False,
            logger=logger
        )
        flex_summary = generate_integration_summary(n, logger)
    else:
        logger.info("Flexibility disabled - loads only")
        flex_summary = pd.DataFrame()

    summary = _build_summary(n, disagg_summary, flex_summary, stats)
    return n, disagg_summary, summary, stats


if __name__ == "__main__":
    logger = setup_logging(snakemake.log[0], "INFO")

    base_network = load_network(snakemake.input.base_network, skip_time_series=False, custom_logger=logger)
    base_profile = pd.read_csv(snakemake.input.base_profile, index_col=0, parse_dates=True)

    component_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    if hasattr(snakemake.input, "hp_profile") and snakemake.input.hp_profile:
        component_data["heat_pumps"] = {
            "profile": pd.read_csv(snakemake.input.hp_profile, index_col=0, parse_dates=True),
            "allocation": pd.read_csv(snakemake.input.hp_allocation),
            "cop": pd.read_csv(snakemake.input.hp_cop, index_col=0, parse_dates=True),
            "thermal": pd.read_csv(snakemake.input.hp_thermal, index_col=0, parse_dates=True)
        }

    if hasattr(snakemake.input, "ev_profile") and snakemake.input.ev_profile:
        component_data["electric_vehicles"] = {
            "profile": pd.read_csv(snakemake.input.ev_profile, index_col=0, parse_dates=True),
            "allocation": pd.read_csv(snakemake.input.ev_allocation)
        }

    hp_cop_profile = None
    if "heat_pumps" in component_data and "cop" in component_data["heat_pumps"]:
        hp_cop_profile = component_data["heat_pumps"]["cop"]
    elif hasattr(snakemake.input, "cop_ashp") and snakemake.input.cop_ashp:
        hp_cop_profile = _load_cop_profile(snakemake.input.cop_ashp, snapshots=base_network.snapshots, logger=logger)

    ev_availability = _load_optional_csv(
        getattr(snakemake.input, "ev_availability", None),
        index_col=0,
        parse_dates=True
    )
    ev_dsm = _load_optional_csv(
        getattr(snakemake.input, "ev_dsm", None),
        index_col=0,
        parse_dates=True
    )

    # Load FES V2G capacity if V2G tariff is enabled and use_fes_capacity is true
    fes_v2g_capacity = None
    flexibility_config = snakemake.params.flexibility_config
    ev_config = flexibility_config.get('electric_vehicles', {})
    tariff = ev_config.get('tariff', 'INT').upper()
    use_fes_v2g = ev_config.get('v2g', {}).get('use_fes_capacity', False)
    
    if tariff == 'V2G' and use_fes_v2g:
        if hasattr(snakemake.input, 'fes_data') and snakemake.input.fes_data:
            fes_scenario = snakemake.params.get('fes_scenario')
            modelled_year = snakemake.params.get('modelled_year')
            if fes_scenario and modelled_year:
                fes_v2g_capacity = load_fes_v2g_capacity(
                    fes_path=snakemake.input.fes_data,
                    fes_scenario=fes_scenario,
                    modelled_year=modelled_year,
                    network=base_network,
                    logger=logger
                )
                if fes_v2g_capacity is not None:
                    logger.info(f"Loaded FES V2G capacity: {fes_v2g_capacity.sum():,.0f} MW total")
            else:
                logger.warning("V2G FES capacity requested but fes_scenario or modelled_year not provided")
        else:
            logger.warning("V2G FES capacity requested but FES data file not available")

    n, disagg_summary, summary, _ = finalize_demand_integration(
        n=base_network,
        base_profile=base_profile,
        disaggregation_config=snakemake.params.disaggregation_config,
        flexibility_config=flexibility_config,
        component_data=component_data,
        hp_cop_profile=hp_cop_profile,
        ev_availability=ev_availability,
        ev_dsm=ev_dsm,
        logger=logger,
        fes_v2g_capacity=fes_v2g_capacity
    )

    save_network(n, snakemake.output.network, custom_logger=logger)
    summary.to_csv(snakemake.output.summary, index=False)
    logger.info(f"Saved demand network: {snakemake.output.network}")
    logger.info(f"Saved integration summary: {snakemake.output.summary}")
