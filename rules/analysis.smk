"""
Post-Solve Analysis Rules for PyPSA-GB

This module contains all post-processing analysis rules for solved networks.

Pipeline Stages:
  1. analyze_and_visualize_solved_network - Interactive spatial plot + dashboard + JSON summary
  2. generate_analysis_notebook - Jupyter notebook for detailed analysis
  3. solve_scenario - Aggregate rule for complete workflow

Inputs:
  - Solved network (.nc file)
  - Generator summary data (.csv)

Outputs:
  - Interactive spatial plot (HTML)
  - Results dashboard (HTML)
  - Analysis summary (JSON)
  - Jupyter notebook (.ipynb)

See Also:
  - solve.smk - Produces solved networks that these rules analyze
  - scripts/analyze_solved_network.py - Spatial plotting and dashboard generation
  - scripts/generate_analysis_notebook.py - Notebook generation
"""

import re

# Use scenarios and run_ids from main Snakefile (no config reloading)
_scenarios = scenarios
_run_ids = run_ids

def _regex_from_list(items):
    """Generate regex pattern that matches exactly the items in the list."""
    if not items:
        return r"(?!x)x"  # Never-matching pattern
    return r"(?:%s)" % "|".join(map(re.escape, items))

SCENARIO_REGEX = _regex_from_list(_run_ids)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS AND VISUALIZATION RULES
# ══════════════════════════════════════════════════════════════════════════════

rule analyze_and_visualize_solved_network:
    """
    Comprehensive post-processing of solved network: plotting, visualization, and analysis.
    
    This consolidated rule combines all post-solve analysis tasks into one output:
    
    1. Interactive Spatial Plot (HTML):
       - Network topology with buses (color-coded by voltage level)
       - Transmission lines (color-coded by loading)
       - Generator locations (size/color by capacity and technology)
       - Storage units
       - Fully interactive Plotly map with pan/zoom/hover
       
    2. Results Dashboard (HTML):
       - Hourly generation mix (stacked area)
       - Peak generation by carrier (bar chart)
       - Storage state of charge (line chart for top 5 units)
       - Load shedding events (area chart)
       - Transmission line loading distribution (histogram)
       - System cost breakdown (pie chart)
       - All synchronized with hover interaction
       
    3. Analysis Summary (JSON):
       - Network size metrics (buses, lines, generators, etc.)
       - Results: total cost, generation, demand, load shedding
       - Generation by carrier (MWh)
       - Peak and average demand
       - Energy balance validation
    
    This replaces separate plotting.smk rules and notebooks.smk generation,
    providing a single, efficient post-processing step.
    
    Input:
      - Solved network: {scenario}_solved.nc
    
    Output:
      - Spatial interactive plot: {scenario}_spatial.html
      - Results dashboard: {scenario}_dashboard.html
      - Summary metrics: {scenario}_summary.json
    
    Performance:
      - ~10-30 seconds for full analysis
      - No re-solve needed
      - Can be run independently on existing solved networks
    
    Usage:
      snakemake resources/analysis/HT35_spatial.html --cores 1
      snakemake resources/analysis/HT35_dashboard.html --cores 1
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_solved.nc"
    output:
        spatial_plot=f"{resources_path}/analysis/{{scenario}}_spatial.html",
        dashboard=f"{resources_path}/analysis/{{scenario}}_dashboard.html",
        summary=f"{resources_path}/analysis/{{scenario}}_summary.json"
    log:
        "logs/analysis/analyze_solved_network_{scenario}.log"
    benchmark:
        "benchmarks/analysis/analyze_solved_network_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=SCENARIO_REGEX
    script:
        "../scripts/analysis/analyze_solved_network.py"


rule generate_analysis_notebook:
    """
    Generate an interactive Jupyter notebook for detailed analysis of a solved network.
    
    Creates a notebook with:
    - Network summary and scenario parameters
    - Generation mix by technology (interactive plots)
    - Storage dispatch and state of charge
    - Transmission line loading analysis
    - Locational marginal prices (LMP)
    - Renewable curtailment analysis
    - Load shedding events
    
    This notebook provides deeper analysis capabilities than the HTML dashboard,
    allowing users to modify analysis and create custom visualizations.
    
    Input:
      - Solved network: {scenario}_solved.nc
      - Generator summary: {scenario}_generators_summary_by_carrier.csv
    
    Output:
      - Analysis notebook: {scenario}_notebook.ipynb
    
    Performance: ~5-10s
    
    Usage:
      snakemake resources/analysis/HT35_notebook.ipynb --cores 1
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_solved.nc",
        generators_summary=f"{resources_path}/generators/{{scenario}}_generators_summary_by_carrier.csv"
    output:
        notebook=f"{resources_path}/analysis/{{scenario}}_notebook.ipynb"
    params:
        scenario=lambda wc: wc.scenario
    wildcard_constraints:
        scenario=SCENARIO_REGEX
    log:
        "logs/analysis/generate_analysis_notebook_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/analysis/generate_analysis_notebook.py"


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE RULES
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: To run full analysis for a scenario, use:
#   snakemake resources/analysis/{scenario}_spatial.html --cores 4
#   (this will automatically trigger solve → analysis → notebook)
#
# Or request specific outputs:
#   snakemake resources/analysis/HT35_spatial.html resources/analysis/HT35_notebook.ipynb --cores 4
