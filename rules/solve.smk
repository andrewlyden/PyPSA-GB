"""
Network Finalization and Optimization Rules for PyPSA-GB

Pipeline Stages:
  1. finalize_network - Create clean {scenario}.nc from complete network
  2. solve_network - Run PyPSA optimization

Inputs:
  - Complete network with all components (.nc)

Outputs:
  - Finalized network (.nc)
  - Solved network with optimization results (.nc)
  - CSV exports (generation, storage, flows, costs, emissions)

See Also:
  - interconnectors.smk - Produces complete networks
  - analysis.smk - Post-solve analysis (spatial plots, dashboards, notebooks)
"""

import re

# Use scenarios and run_ids from main Snakefile (no config reloading)
_scenarios = scenarios
_run_ids = run_ids
_cfg = config

def _get_solver_config(scenario_id=None):
    """
    Get solver configuration from merged scenario config (includes defaults.yaml).
    
    Args:
        scenario_id: Optional scenario ID to get scenario-specific solver config.
                     If None, returns defaults from first scenario or config.yaml.
    
    Returns:
        Tuple of (solver_name, solver_options)
    """
    # Get solver config from scenario (which inherits from defaults.yaml)
    if scenario_id and scenario_id in _scenarios:
        solver_config = _scenarios[scenario_id].get("solver", {})
    elif _run_ids:
        # Fall back to first active scenario's config
        first_scenario = _run_ids[0]
        solver_config = _scenarios.get(first_scenario, {}).get("solver", {})
    else:
        # Last resort: main config
        solver_config = _cfg.get("solver", {})
    
    # Get solver name (default to gurobi per defaults.yaml)
    solver_name = solver_config.get("name", "gurobi")
    
    # Build solver options based on solver type
    if solver_name == "gurobi":
        solver_options = {
            "threads": solver_config.get("threads", 4),
            "method": solver_config.get("method", 2),
            "crossover": solver_config.get("crossover", 0),
            "BarHomogeneous": solver_config.get("BarHomogeneous", 1),  # Handle numerical issues
            "BarConvTol": solver_config.get("BarConvTol", 1.e-4),
            "FeasibilityTol": solver_config.get("FeasibilityTol", 1.e-4),
            "OptimalityTol": solver_config.get("OptimalityTol", 1.e-4),
            "NumericFocus": solver_config.get("NumericFocus", 3),  # Max numerical stability
            "ScaleFlag": solver_config.get("ScaleFlag", 2),  # Aggressive scaling
            "DualReductions": solver_config.get("DualReductions", 0),
            "BarIterLimit": solver_config.get("BarIterLimit", 200),  # More iterations
        }
    elif solver_name == "highs":
        solver_options = {
            "threads": solver_config.get("threads", 4),
            "log_to_console": False,
        }
    else:
        # Generic options for other solvers
        solver_options = {"threads": solver_config.get("threads", 4)}
    
    return solver_name, solver_options

def _regex_from_list(items):
    """Generate regex pattern that matches exactly the items in the list."""
    if not items:
        return r"(?!x)x"  # Never-matching pattern
    return r"(?:%s)" % "|".join(map(re.escape, items))

SCENARIO_REGEX = _regex_from_list(_run_ids)

def _is_clustering_enabled(scenario_id):
    """
    Determine if clustering is enabled for a scenario.
    
    Logic:
    - String preset name (e.g., "gsp_spatial") means enabled
    - Dict with 'method' key means enabled (unless explicitly disabled)
    - Dict with 'enabled: false' means disabled
    - None or False means disabled
    
    This logic is consistent with config_loader.py's resolve_clustering().
    """
    clustering_config = _scenarios.get(scenario_id, {}).get('clustering', None)
    if isinstance(clustering_config, str):
        return True  # String preset names are always enabled
    elif isinstance(clustering_config, dict):
        if 'enabled' in clustering_config:
            return clustering_config['enabled']
        else:
            return 'method' in clustering_config
    return False


def _is_component_aggregation_enabled(scenario_id):
    """Return True if post-clustering aggregation is enabled."""
    clustering_config = _scenarios.get(scenario_id, {}).get('clustering', None)
    if isinstance(clustering_config, dict):
        agg_cfg = clustering_config.get("aggregate_components", {})
        if isinstance(agg_cfg, bool):
            return agg_cfg
        if isinstance(agg_cfg, dict):
            return agg_cfg.get("enabled", False)
    return False


def _clustered_network_output(scenario_id):
    """Return path to clustered network (aggregated variant if configured)."""
    base = f"{resources_path}/network/{scenario_id}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    if _is_component_aggregation_enabled(scenario_id):
        return f"{resources_path}/network/{scenario_id}_network_clustered_aggregated_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    return base

# ══════════════════════════════════════════════════════════════════════════════
# RULES
# ══════════════════════════════════════════════════════════════════════════════

rule finalize_network:
    """
    Finalize complete network to clean scenario name.
    
    This rule creates a clean {scenario}.nc file from the complete network
    (either clustered or unclustered depending on scenario configuration).
    This provides a simple, clean interface for users and downstream analysis.
    
    The complete network contains:
    - Network topology (buses, lines, transformers)
    - Demand loads
    - Renewable generators (wind, solar, hydro, tidal, wave)
    - Thermal generators (CCGT, coal, nuclear, biomass, etc.)
    - Load shedding backup (VoLL)
    - Storage units (batteries, pumped hydro, etc.)
    - Interconnectors (cross-border links)
    
    Input:
      - Complete network with all components (clustered if enabled)
    
    Output:
      - Network file {scenario}.nc
      - Network summary: {scenario}_network_summary.txt
    
    Usage:
      snakemake resources/network/Historical_2020_clustered.nc --cores 1
    """
    input:
        network=lambda wildcards: (
            _clustered_network_output(wildcards.scenario)
            if _is_clustering_enabled(wildcards.scenario)
            else f"{resources_path}/network/{wildcards.scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
        )
    output:
        network=f"{resources_path}/network/{{scenario}}.nc",
        summary=f"{resources_path}/network/{{scenario}}_network_summary.txt"
    log:
        "logs/finalize_network_{scenario}.log"
    benchmark:
        "benchmarks/solve/finalize_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=SCENARIO_REGEX
    params:
        # Merge global config (config.yaml) with scenario-specific config
        scenario_config=lambda wildcards: {**_cfg, **_scenarios.get(wildcards.scenario, {})}
    script:
        "../scripts/solve/finalize_network.py"


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

rule solve_network:
    """
    Solve network using PyPSA optimization.
    
    This rule performs optimal power flow optimization to determine:
    - Generator dispatch schedules (which generators run when)
    - Storage charging/discharging patterns
    - Interconnector flows (imports/exports)
    - Line loading and potential congestion
    - System costs (generation, storage, load shedding)
    - Carbon emissions
    
    Optimization Formulation:
      - Objective: Minimize total system cost
      - Constraints:
        * Power balance at each bus
        * Generator capacity limits
        * Line thermal limits
        * Storage energy/power limits
        * Ramp rate constraints
        * Minimum up/down times (optional)
    
    Solve Period (Optional):
      - Networks are built with full year data for consistency
      - Can optimize subset of year via 'solve_period' in scenario config
      - Options:
        * Explicit dates: start/end (YYYY-MM-DD)
        * Auto-select: peak_demand_week, peak_wind_week, low_wind_week
      - Example: Solve one week in December for peak demand analysis
      - Reduces solve time while maintaining data consistency
    
    Solver Configuration:
      - Default: Gurobi (if available) or HiGHS (open source)
      - Configurable via scenario config or config.yaml
      - Solver options: threads, method, tolerances, time limits
    
    Input:
      - Finalized network: {scenario}.nc (full year data)
      - Solver configuration (from config.yaml)
    
    Output:
      - Solved network: {scenario}_solved.nc (includes optimization results)
      - Optimization summary: {scenario}_optimization_summary.txt
      - Results CSV exports: generation, storage, flows, costs
    
    Performance:
      - Runtime varies greatly with network size, solver, and period
      - Full year ETYS network: 30 min - 2 hours
      - Full year clustered: 5-30 minutes
      - One week clustered: 1-5 minutes (much faster!)
      - Zonal networks: 1-10 minutes
    
    Usage:
      snakemake resources/network/Historical_2020_clustered_solved.nc --cores 1
    """
    input:
        network=f"{resources_path}/network/{{scenario}}.nc"
    output:
        network=f"{resources_path}/network/{{scenario}}_solved.nc",
        summary=f"{resources_path}/network/{{scenario}}_optimization_summary.txt",
        generation_csv=f"{resources_path}/results/{{scenario}}_generation.csv",
        storage_csv=f"{resources_path}/results/{{scenario}}_storage.csv",
        flows_csv=f"{resources_path}/results/{{scenario}}_flows.csv",
        costs_csv=f"{resources_path}/results/{{scenario}}_costs.csv",
        emissions_csv=f"{resources_path}/results/{{scenario}}_emissions.csv"
    log:
        "logs/solve_network_{scenario}.log"
    benchmark:
        "benchmarks/solve/solve_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    threads: 4
    wildcard_constraints:
        scenario=SCENARIO_REGEX
    params:
        # Merge global config (config.yaml) with scenario-specific config
        # Scenario config takes precedence
        scenario_config=lambda wildcards: {**_cfg, **_scenarios.get(wildcards.scenario, {})},
        solver=lambda wildcards: _get_solver_config(wildcards.scenario)[0],
        solver_options=lambda wildcards: _get_solver_config(wildcards.scenario)[1]
    script:
        "../scripts/solve/solve_network.py"


# NOTE: Post-solve analysis rules (plotting, visualization, notebooks) are now
# in analysis.smk for better organization. See rules/analysis.smk.
