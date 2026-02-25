"""
Market Simulation Rules for PyPSA-GB — Two-Stage Dispatch

Pipeline Stages:
  1. solve_wholesale_market    - Stage 1: copperplate dispatch (no network constraints)
  2. solve_balancing_mechanism - Stage 2: anchored redispatch with full constraints
  3. analyze_market_results    - Post-processing dashboard and summary

Inputs:
  - Finalized network: {scenario}.nc (from main workflow)

Outputs:
  - Wholesale solved network + dispatch CSVs
  - Balancing solved network + redispatch/congestion CSVs
  - Market dashboard (HTML) + summary (JSON)

Configuration:
  Enable per-scenario via market.enabled: true in scenarios.yaml.
  See config/defaults.yaml 'market:' section for full configuration.

See Also:
  - solve.smk - Standard (single-stage) network optimisation
  - analysis.smk - Standard post-solve analysis
  - scripts/market/ - Python modules for market simulation
"""

import re

# Use scenarios and run_ids from main Snakefile (no config reloading)
_scenarios = scenarios
_run_ids = run_ids
_cfg = config


def _get_solver_config_market(scenario_id=None):
    """
    Get solver configuration for market rules.

    Mirrors _get_solver_config from solve.smk but is local to this module.
    """
    if scenario_id and scenario_id in _scenarios:
        solver_config = _scenarios[scenario_id].get("solver", {})
    elif _run_ids:
        solver_config = _scenarios.get(_run_ids[0], {}).get("solver", {})
    else:
        solver_config = _cfg.get("solver", {})

    solver_name = solver_config.get("name", "gurobi")

    if solver_name == "gurobi":
        solver_options = {
            "threads": solver_config.get("threads", 4),
            "method": solver_config.get("method", 2),
            "crossover": solver_config.get("crossover", 0),
            "BarHomogeneous": solver_config.get("BarHomogeneous", 1),
            "BarConvTol": solver_config.get("BarConvTol", 1.0e-4),
            "FeasibilityTol": solver_config.get("FeasibilityTol", 1.0e-4),
            "OptimalityTol": solver_config.get("OptimalityTol", 1.0e-4),
            "NumericFocus": solver_config.get("NumericFocus", 3),
            "ScaleFlag": solver_config.get("ScaleFlag", 2),
            "DualReductions": solver_config.get("DualReductions", 0),
            "BarIterLimit": solver_config.get("BarIterLimit", 200),
        }
    elif solver_name == "highs":
        solver_options = {
            "threads": solver_config.get("threads", 4),
            "log_to_console": False,
        }
    else:
        solver_options = {"threads": solver_config.get("threads", 4)}

    return solver_name, solver_options


def _is_market_enabled(scenario_id):
    """Check if market simulation is enabled for a scenario."""
    return _scenarios.get(scenario_id, {}).get("market", {}).get("enabled", False)


def _regex_from_list(items):
    """Generate regex pattern that matches exactly the items in the list."""
    if not items:
        return r"(?!x)x"  # Never-matching pattern
    return r"(?:%s)" % "|".join(map(re.escape, items))


# Build regex for scenarios with market enabled
_market_scenario_ids = [rid for rid in _run_ids if _is_market_enabled(rid)]
MARKET_SCENARIO_REGEX = _regex_from_list(_market_scenario_ids) if _market_scenario_ids else _regex_from_list(_run_ids)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: WHOLESALE MARKET (COPPERPLATE DISPATCH)
# ══════════════════════════════════════════════════════════════════════════════

rule solve_wholesale_market:
    """
    Stage 1: Solve wholesale market with copperplate (unconstrained) dispatch.

    Removes all network constraints by setting line/transformer ratings to
    a very large value, producing a single uniform clearing price. This
    represents the GB wholesale electricity market.

    Input:
      - Finalized network with all components: {scenario}.nc

    Output:
      - Wholesale solved network: {scenario}_wholesale.nc
      - Generator dispatch: {scenario}_wholesale_dispatch.csv
      - Storage dispatch: {scenario}_wholesale_storage.csv
      - Link dispatch: {scenario}_wholesale_links.csv
      - Uniform price: {scenario}_wholesale_price.csv
    """
    input:
        network=f"{resources_path}/network/{{scenario}}.nc"
    output:
        network=f"{resources_path}/market/{{scenario}}_wholesale.nc",
        wholesale_dispatch_csv=f"{resources_path}/market/{{scenario}}_wholesale_dispatch.csv",
        wholesale_storage_csv=f"{resources_path}/market/{{scenario}}_wholesale_storage.csv",
        wholesale_links_csv=f"{resources_path}/market/{{scenario}}_wholesale_links.csv",
        wholesale_price_csv=f"{resources_path}/market/{{scenario}}_wholesale_price.csv",
    log:
        "logs/market/solve_wholesale_{scenario}.log"
    benchmark:
        "benchmarks/market/solve_wholesale_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    threads: 4
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: {**_cfg, **_scenarios.get(wildcards.scenario, {})},
        solver=lambda wildcards: _get_solver_config_market(wildcards.scenario)[0],
        solver_options=lambda wildcards: _get_solver_config_market(wildcards.scenario)[1],
    script:
        "../scripts/market/solve_wholesale.py"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: BALANCING MECHANISM (ANCHORED REDISPATCH)
# ══════════════════════════════════════════════════════════════════════════════

rule solve_balancing_mechanism:
    """
    Stage 2: Solve balancing mechanism with full network constraints.

    Starting from wholesale dispatch positions, re-solves with actual
    line/transformer ratings. Generators and storage are anchored to their
    wholesale position, with increase/decrease variables penalised by
    bid/offer prices — mimicking the GB Balancing Mechanism.

    Input:
      - Finalized network (original constraints): {scenario}.nc
      - Wholesale dispatch CSVs from Stage 1

    Output:
      - Balancing solved network: {scenario}_balancing.nc
      - Physical dispatch: {scenario}_balancing_dispatch.csv
      - Redispatch summary: {scenario}_redispatch_summary.csv
      - Constraint costs: {scenario}_constraint_costs.csv
      - Congestion analysis: {scenario}_congestion.csv
      - Price comparison: {scenario}_price_comparison.csv
    """
    input:
        network=f"{resources_path}/network/{{scenario}}.nc",
        wholesale_dispatch_csv=f"{resources_path}/market/{{scenario}}_wholesale_dispatch.csv",
        wholesale_storage_csv=f"{resources_path}/market/{{scenario}}_wholesale_storage.csv",
        wholesale_links_csv=f"{resources_path}/market/{{scenario}}_wholesale_links.csv",
        wholesale_price_csv=f"{resources_path}/market/{{scenario}}_wholesale_price.csv",
    output:
        network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        balancing_dispatch_csv=f"{resources_path}/market/{{scenario}}_balancing_dispatch.csv",
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        congestion_csv=f"{resources_path}/market/{{scenario}}_congestion.csv",
        price_comparison_csv=f"{resources_path}/market/{{scenario}}_price_comparison.csv",
    log:
        "logs/market/solve_balancing_{scenario}.log"
    benchmark:
        "benchmarks/market/solve_balancing_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    threads: 4
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: {**_cfg, **_scenarios.get(wildcards.scenario, {})},
        solver=lambda wildcards: _get_solver_config_market(wildcards.scenario)[0],
        solver_options=lambda wildcards: _get_solver_config_market(wildcards.scenario)[1],
    script:
        "../scripts/market/solve_balancing.py"


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS: MARKET DASHBOARD AND SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

rule analyze_market_results:
    """
    Post-processing analysis of two-stage market simulation results.

    Compares wholesale and balancing dispatch to produce:
    - Interactive dashboard (HTML) with 6 panels
    - Machine-readable summary (JSON) with key metrics

    Input:
      - Both solved networks (.nc) and all result CSVs

    Output:
      - Market dashboard: {scenario}_market_dashboard.html
      - Market summary: {scenario}_market_summary.json
    """
    input:
        wholesale_network=f"{resources_path}/market/{{scenario}}_wholesale.nc",
        balancing_network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        wholesale_dispatch_csv=f"{resources_path}/market/{{scenario}}_wholesale_dispatch.csv",
        balancing_dispatch_csv=f"{resources_path}/market/{{scenario}}_balancing_dispatch.csv",
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        congestion_csv=f"{resources_path}/market/{{scenario}}_congestion.csv",
        price_comparison_csv=f"{resources_path}/market/{{scenario}}_price_comparison.csv",
    output:
        dashboard=f"{resources_path}/analysis/{{scenario}}_market_dashboard.html",
        summary_json=f"{resources_path}/analysis/{{scenario}}_market_summary.json",
    log:
        "logs/market/analyze_market_{scenario}.log"
    benchmark:
        "benchmarks/market/analyze_market_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: {**_cfg, **_scenarios.get(wildcards.scenario, {})},
    script:
        "../scripts/market/analyze_market.py"


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS NOTEBOOK: PER-SCENARIO MARKET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

rule generate_market_analysis_notebook:
    """
    Generate a per-scenario Jupyter notebook for two-stage market analysis.

    Produces a self-contained .ipynb with eight analysis sections:
      1. Wholesale dispatch overview (generation + storage by carrier)
      2. Wholesale price time series and copperplate diagnostic
      3. BM redispatch volume by carrier
      4. BM constraint costs by carrier (stacked bar + pie)
      5. Top BM assets by increase / decrease
      6. Network congestion (hours congested, mean loading fraction)
      7. Wholesale vs BM nodal prices comparison
      8. Summary statistics table

    Paths to all input CSVs are embedded in the notebook at generation time,
    so the notebook can be opened and re-run from any working directory.

    Input:
      - All market CSVs produced by Stage 1 + Stage 2 rules
      - Both solved network .nc files (for carrier lookup)

    Output:
      - {scenario}_market_notebook.ipynb  → resources/analysis/

    Usage:
      snakemake resources/analysis/HT35_market_market_notebook.ipynb --cores 1
    """
    input:
        wholesale_network=f"{resources_path}/market/{{scenario}}_wholesale.nc",
        balancing_network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        wholesale_dispatch_csv=f"{resources_path}/market/{{scenario}}_wholesale_dispatch.csv",
        wholesale_storage_csv=f"{resources_path}/market/{{scenario}}_wholesale_storage.csv",
        wholesale_links_csv=f"{resources_path}/market/{{scenario}}_wholesale_links.csv",
        wholesale_price_csv=f"{resources_path}/market/{{scenario}}_wholesale_price.csv",
        balancing_dispatch_csv=f"{resources_path}/market/{{scenario}}_balancing_dispatch.csv",
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        congestion_csv=f"{resources_path}/market/{{scenario}}_congestion.csv",
        price_comparison_csv=f"{resources_path}/market/{{scenario}}_price_comparison.csv",
    output:
        notebook=f"{resources_path}/analysis/{{scenario}}_market_notebook.ipynb",
    log:
        "logs/market/generate_market_notebook_{scenario}.log"
    benchmark:
        "benchmarks/market/generate_market_notebook_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    script:
        "../scripts/market/generate_market_analysis_notebook.py"
