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
MARKET_SCENARIO_REGEX = _regex_from_list(_market_scenario_ids)


def _is_historical_market_elexon(scenario_id):
    """Check if scenario uses ELEXON bid/offer source."""
    sc = _scenarios.get(scenario_id, {})
    if not sc.get("market", {}).get("enabled", False):
        return False
    source = sc.get("market", {}).get("balancing", {}).get("bid_offer_source", "derived")
    return source == "elexon"


_elexon_scenario_ids = [rid for rid in _run_ids if _is_historical_market_elexon(rid)]
ELEXON_SCENARIO_REGEX = _regex_from_list(_elexon_scenario_ids) if _elexon_scenario_ids else r"(?!x)x"


def _balancing_extra_inputs(wildcards):
    """Return additional input files for ELEXON scenarios (BMU mapping + data)."""
    sid = wildcards.scenario
    if _is_historical_market_elexon(sid):
        return {
            "elexon_offers": f"{resources_path}/market/{sid}/elexon/elexon_offers.csv",
            "elexon_bids": f"{resources_path}/market/{sid}/elexon/elexon_bids.csv",
            "bmu_mapping": f"{resources_path}/generators/{sid}_bmu_mapping.csv",
        }
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: BUILD BMU-TO-GENERATOR MAPPING (ELEXON SCENARIOS ONLY)
# ══════════════════════════════════════════════════════════════════════════════

rule build_bmu_mapping:
    """
    Build mapping from ELEXON BMU IDs to PyPSA generator names.

    Uses ETYS Dir_con_BMUs_to_node sheet and station-to-prefix lookups
    to map BMU IDs (e.g., T_PEMB-11) to generator names (e.g., Pembroke).

    Only runs for scenarios with bid_offer_source: "elexon".
    """
    input:
        network=f"{resources_path}/network/{{scenario}}.nc",
    output:
        bmu_mapping=f"{resources_path}/generators/{{scenario}}_bmu_mapping.csv",
    log:
        "logs/market/build_bmu_mapping_{scenario}.log"
    benchmark:
        "benchmarks/market/build_bmu_mapping_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=ELEXON_SCENARIO_REGEX
    params:
        etys_path="data/network/ETYS/GB_network.xlsx",
    script:
        "../scripts/generators/build_bmu_mapping.py"


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: RETRIEVE ELEXON MARKET DATA (HISTORICAL SCENARIOS ONLY)
# ══════════════════════════════════════════════════════════════════════════════

rule retrieve_elexon_market_data:
    """
    Fetch ELEXON BMRS bid/offer data for historical market scenarios.

    Only runs when market.balancing.bid_offer_source is "elexon".
    Downloads settlement-period-level data and aggregates to hourly.

    Output:
      - ELEXON offer prices: elexon/elexon_offers.csv
      - ELEXON bid prices: elexon/elexon_bids.csv
    """
    output:
        offers_file=f"{resources_path}/market/{{scenario}}/elexon/elexon_offers.csv",
        bids_file=f"{resources_path}/market/{{scenario}}/elexon/elexon_bids.csv",
    log:
        "logs/market/retrieve_elexon_{scenario}.log"
    benchmark:
        "benchmarks/market/retrieve_elexon_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=ELEXON_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: {**_cfg, **_scenarios.get(wildcards.scenario, {})},
    script:
        "../scripts/market/elexon_data.py"


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
        unpack(_balancing_extra_inputs),
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


# WHOLESALE-ONLY ANALYSIS NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════

rule generate_wholesale_notebook:
    """
    Generate a per-scenario Jupyter notebook for wholesale market analysis.

    Produces a self-contained .ipynb with four analysis sections:
      1. Setup — libraries, file paths, data loading
      2. Merit Order — supply curve (unit-level)
      3. Dispatch Stack — stacked area chart by carrier
      4. Wholesale Price vs Market — SMP overlaid with ELEXON MID

    Used for scenarios with market.wholesale_only: true (Stage 1 only, no BM).

    Input:
      - Wholesale CSVs (dispatch, storage, links, price) from solve_wholesale_market
      - Solved wholesale network .nc (for carrier/capacity lookup)
      - Marginal costs breakdown CSV (for merit order plot)
      - scenario_config param (for modelled_year + solve_period for MID fetch)

    Output:
      - {scenario}_wholesale_notebook.ipynb  → resources/analysis/

    Usage:
      snakemake resources/analysis/Historical_2024_lowwind_wholesale_notebook.ipynb --cores 1
    """
    input:
        wholesale_network=f"{resources_path}/market/{{scenario}}_wholesale.nc",
        wholesale_dispatch_csv=f"{resources_path}/market/{{scenario}}_wholesale_dispatch.csv",
        wholesale_storage_csv=f"{resources_path}/market/{{scenario}}_wholesale_storage.csv",
        wholesale_links_csv=f"{resources_path}/market/{{scenario}}_wholesale_links.csv",
        wholesale_price_csv=f"{resources_path}/market/{{scenario}}_wholesale_price.csv",
        marginal_costs_csv=f"{resources_path}/generators/{{scenario}}_marginal_costs_breakdown.csv",
    output:
        notebook=f"{resources_path}/analysis/{{scenario}}_wholesale_notebook.ipynb",
    params:
        scenario_config=lambda wildcards: {**_cfg, **_scenarios.get(wildcards.scenario, {})},
    log:
        "logs/market/generate_wholesale_notebook_{scenario}.log"
    benchmark:
        "benchmarks/market/generate_wholesale_notebook_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    script:
        "../scripts/market/generate_wholesale_notebook.py"
