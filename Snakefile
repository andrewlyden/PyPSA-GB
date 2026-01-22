# ==============================================================================
# PyPSA-GB: Energy System Model for Great Britain
# ==============================================================================
"""
MAIN WORKFLOW

This workflow assumes weather cutouts already exist and focuses on:
  * Renewable profile generation
  * Network building (ETYS/Reduced/Zonal models)
  * Network clustering (spatial/k-means)
  * Demand processing and disaggregation
  * Generator integration (DUKES + REPD + FES - historical vs future routing)
  * Storage integration (REPD)
  * Interconnector mapping
  * Validation and visualization

PREREQUISITES:
  If weather cutouts don't exist, run the cutout workflow FIRST:
      snakemake -s Snakefile_cutouts --cores 4

  Then run this main workflow:
      snakemake --cores 8

"""

# ------------------------------------------------------------------------------
# IMPORTS AND SETUP
# ------------------------------------------------------------------------------

import yaml
import sys
from pathlib import Path

# Add scripts directory to Python path for imports
sys.path.insert(0, str(Path.cwd() / "scripts"))
sys.path.insert(0, str(Path.cwd() / "config"))
from scripts.utilities.scenario_detection import (
    is_historical_scenario,
    auto_configure_scenario,
    summarize_scenario_configuration,
    validate_historical_scenario,
    validate_future_scenario,
    validate_scenario_complete
)
from config_loader import load_config

# ------------------------------------------------------------------------------
# CONFIGURATION LOADING
# ------------------------------------------------------------------------------

# Load main configuration file
configfile: "config/config.yaml"

# Load scenarios with proper inheritance from defaults.yaml
# This ensures all scenario configs are merged with defaults
_full_config = load_config()
scenarios = _full_config["scenarios"]  # These are already merged with defaults

# ------------------------------------------------------------------------------
# PATH DEFINITIONS (shared with all rule modules)
# ------------------------------------------------------------------------------

resources_path = "resources"
data_path = "data"

# ------------------------------------------------------------------------------
# SCENARIO SELECTION
# ------------------------------------------------------------------------------
# Determine which scenarios to run: command-line override or config file
# ------------------------------------------------------------------------------

if config.get("scenario"):
    # Single scenario specified via command line: snakemake --config scenario=HT35
    run_ids = [config["scenario"]]
    print(f"Using command-line scenario: {config['scenario']}")
else:
    # Use scenarios from config.yaml run_scenarios list
    run_ids = config["run_scenarios"] 
    print(f"Using config file scenarios: {run_ids}")

print(f"Active scenarios: {run_ids}")

# Validate scenario IDs exist in scenarios_master.yaml
unknown = set(run_ids) - set(scenarios)
if unknown:
    raise ValueError(f"Unknown scenario IDs: {unknown}. Check scenarios_master.yaml for valid IDs.")

# ------------------------------------------------------------------------------
# SCENARIO AUTO-CONFIGURATION AND VALIDATION
# ------------------------------------------------------------------------------
# Automatically detect historical vs future scenarios and configure data sources
# Validate configuration for both scenario types
# Check data freshness for historical scenarios
# ------------------------------------------------------------------------------

import logging
logging.basicConfig(level=logging.WARNING)  # Suppress INFO logging during initialization

enhanced_scenarios = {}
validation_errors = []
validation_warnings = []
validation_info = []

for rid in run_ids:
    # Auto-configure with metadata
    enhanced_scenarios[rid] = auto_configure_scenario(scenarios[rid])
    
    # Complete validation (includes data freshness checks)
    validation_result = validate_scenario_complete(scenarios[rid])
    
    # Collect errors (critical issues)
    if validation_result['errors']:
        validation_errors.extend([(rid, err) for err in validation_result['errors']])
    
    # Collect warnings (non-critical issues)
    if validation_result['warnings']:
        validation_warnings.extend([(rid, warn) for warn in validation_result['warnings']])
    
    # Collect info messages (data freshness, etc.)
    if validation_result.get('info'):
        validation_info.extend([(rid, info) for info in validation_result['info']])

# Print data freshness info
if validation_info:
    print("\n" + "="*80)
    print("INFO: DATA FRESHNESS INFORMATION")
    print("="*80)
    for rid, info in validation_info:
        print(f"  {info}")
    print("="*80 + "\n")

# Print validation warnings
if validation_warnings:
    print("\n" + "="*80)
    print("WARNING: CONFIGURATION WARNINGS")
    print("="*80)
    for rid, warning in validation_warnings:
        print(f"\nScenario '{rid}':")
        print(f"  {warning}")
    print("="*80 + "\n")

# Print validation errors and halt if any critical errors found
if validation_errors:
    print("\n" + "="*80)
    print("ERROR: CONFIGURATION ERRORS")
    print("="*80)
    for rid, error in validation_errors:
        print(f"\nScenario '{rid}':")
        print(f"  {error}")
    print("="*80 + "\n")
    raise ValueError(
        f"Found {len(validation_errors)} configuration error(s). "
        "Please fix the errors above and re-run the workflow."
    )

# Print summary of what data sources will be used
summary = summarize_scenario_configuration({rid: scenarios[rid] for rid in run_ids})
print(f"\nScenario Summary:")
if summary['historical']:
    print(f"  Historical scenarios ({len(summary['historical'])}): {', '.join(summary['historical'])}")
    print(f"      -> Using DUKES data for years: {summary['dukes_years_needed']}")
if summary['future']:
    print(f"  Future scenarios ({len(summary['future'])}): {', '.join(summary['future'])}")
    print(f"      -> Downloading FES data for years: {summary['fes_years_needed']}")
print(f"  Weather cutouts needed for years: {summary['cutout_years_needed']}")
print()

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def extract_from_scenarios(key):
    """
    Extract unique values for a given key from all active scenarios.
    Skips scenarios that don't have the specified key.
    
    Args:
        key (str): Configuration key to extract (e.g., 'demand_year', 'network_model')
    
    Returns:
        list: Sorted list of unique values
    """
    values = set()
    for rid in run_ids:
        if key in scenarios[rid]:
            values.add(scenarios[rid][key])
    return sorted(list(values))


def extract_fes_years():
    """
    Extract FES years only from future scenarios that need them.
    Historical scenarios are excluded automatically.
    
    Returns:
        list: Sorted list of FES years needed
    """
    fes_years = set()
    for rid in run_ids:
        if enhanced_scenarios[rid]['_needs_fes']:
            if 'FES_year' in scenarios[rid]:
                fes_years.add(scenarios[rid]['FES_year'])
    return sorted(list(fes_years))


def extract_disaggregation_configs():
    """
    Extract demand disaggregation configurations from scenarios.
    Returns dictionary with scenario IDs as keys.
    
    Returns:
        dict: {scenario_id: disaggregation_config} or {scenario_id: {'enabled': False}}
    """
    configs = {}
    for rid in run_ids:
        if "demand_disaggregation" in scenarios[rid]:
            configs[rid] = scenarios[rid]["demand_disaggregation"]
        else:
            configs[rid] = {"enabled": False}  # Default: disabled
    return configs


def get_base_network_file(network_model):
    """
    Get the path to the base network file for a given network model.
    
    Args:
        network_model (str): Network model name (ETYS, Reduced, Zonal)
    
    Returns:
        str: Path to base network NetCDF file
    """
    return f"{resources_path}/network/{network_model}_base.nc"


def get_base_demand_file(network_model):
    """
    Get the path to the base demand file for a given network model.
    
    Args:
        network_model (str): Network model name (ETYS, Reduced, Zonal)
    
    Returns:
        str: Path to base demand NetCDF file
    """
    return f"{resources_path}/network/{network_model}_base_demand.nc"


def get_network_outputs():
    """
    Generate list of network output files for all scenarios.
    Includes base networks and clustered networks if clustering is enabled.
    
    Returns:
        list: Paths to all network output files
    """
    outputs = []
    for rid in run_ids:
        network_model = scenarios[rid]["network_model"]
        outputs.extend([
            get_base_network_file(network_model),
            get_base_demand_file(network_model)
        ])
        
        # Add clustered network outputs if clustering is enabled
        clustering_config = scenarios[rid].get("clustering", None)
        clustering_enabled = False
        if isinstance(clustering_config, str):
            clustering_enabled = True
        elif isinstance(clustering_config, dict):
            if 'enabled' in clustering_config:
                clustering_enabled = clustering_config['enabled']
            else:
                clustering_enabled = 'method' in clustering_config
        if clustering_enabled:
            outputs.append(f"{resources_path}/network/{network_model}_clustered_{rid}.nc")
            
    return outputs

# ------------------------------------------------------------------------------
# EXTRACT SCENARIO PARAMETERS
# ------------------------------------------------------------------------------
# Build lists of unique values needed across all scenarios
# These drive Snakemake's input/output file expansion
# ------------------------------------------------------------------------------

demand_year               = extract_from_scenarios("demand_year")
modelled_year             = extract_from_scenarios("modelled_year")
fes_year                  = extract_fes_years()  # Only non-historical scenarios
fes_scenario              = extract_from_scenarios("FES_scenario")
demand_timeseries         = extract_from_scenarios("demand_timeseries")
network_models            = extract_from_scenarios("network_model")
renewables_year           = extract_from_scenarios("renewables_year")
demand_disaggregation_configs = extract_disaggregation_configs()

# ------------------------------------------------------------------------------
# ATLITE PATH DEFINITIONS
# ------------------------------------------------------------------------------
# Additional paths for weather data processing
# (resources_path and data_path defined above in CONFIGURATION LOADING section)
# ------------------------------------------------------------------------------

atlite_cutouts_path    = f"{resources_path}/atlite/cutouts"
atlite_inputs_path     = f"{resources_path}/atlite/inputs"
atlite_outputs_path    = f"{resources_path}/atlite/outputs"

# ------------------------------------------------------------------------------
# CONDA ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------------------------
# Default environment for all rules unless specifically overridden
# ------------------------------------------------------------------------------

default_conda_env = "envs/pypsa-gb.yaml"

# ------------------------------------------------------------------------------
# INCLUDE RULE MODULES
# ------------------------------------------------------------------------------
# Modular rule organization by workflow component
# ------------------------------------------------------------------------------

include: "rules/renewables.smk"           # Renewable generation profiles
include: "rules/network_build.smk"        # Network topology construction
include: "rules/network_clustering.smk"   # Network clustering algorithms
include: "rules/demand.smk"               # CONSOLIDATED: Base demand + disaggregation + flexibility
include: "rules/FES.smk"                  # Future Energy Scenarios data
include: "rules/generators.smk"           # Generator integration (DUKES/REPD/FES - historical vs future routing)
include: "rules/storage.smk"              # Energy storage integration (battery, pumped hydro, CAES, LAES)
include: "rules/hydrogen.smk"             # Hydrogen system (electrolysis, H2 storage, H2 turbines)
include: "rules/interconnectors.smk"      # Cross-border interconnections
include: "rules/solve.smk"                # Network finalization and optimization
include: "rules/analysis.smk"             # Post-solve analysis: spatial plots, dashboards, notebooks

# ------------------------------------------------------------------------------
# FINAL TARGET CONSTRUCTION
# ------------------------------------------------------------------------------
# Build explicit target lists for the 'all' rule
# Organized by workflow component for clarity and maintainability
# ------------------------------------------------------------------------------

# --- Network Targets ----------------------------------------------------------
# Fully assembled networks with all components, clustered variants, and validation
# ------------------------------------------------------------------------------

network_targets = []

# Final fully-assembled networks (with all components) for each scenario
network_targets += expand(
    f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc", 
    scenario=run_ids
)

# Add clustering and validation targets based on clustering configuration
for rid in run_ids:
    network_model = scenarios[rid]["network_model"]
    # Check if clustering is enabled - can be a string preset name or dict config
    clustering_config = scenarios[rid].get("clustering", None)
    # Determine if clustering is enabled:
    # - String preset name (e.g., "gsp_spatial") means enabled
    # - Dict with 'method' key means enabled (unless explicitly disabled)
    # - Dict with 'enabled: false' means disabled
    # - None or False means disabled
    if isinstance(clustering_config, str):
        clustering_enabled = True  # String preset names are always enabled
    elif isinstance(clustering_config, dict):
        # If 'enabled' is explicitly set, use that value
        # Otherwise, if 'method' is specified, assume enabled (consistent with config_loader.py)
        if 'enabled' in clustering_config:
            clustering_enabled = clustering_config['enabled']
        else:
            clustering_enabled = 'method' in clustering_config
    else:
        clustering_enabled = False
    if clustering_enabled:
        # Optional post-clustering aggregation of identical components
        agg_cfg = clustering_config.get("aggregate_components", {}) if isinstance(clustering_config, dict) else {}
        if isinstance(agg_cfg, bool):
            aggregation_enabled = agg_cfg
        elif isinstance(agg_cfg, dict):
            aggregation_enabled = agg_cfg.get("enabled", False)
        else:
            aggregation_enabled = False

        clustered_network_path = (
            f"{resources_path}/network/{rid}_network_clustered_aggregated_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
            if aggregation_enabled
            else f"{resources_path}/network/{rid}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
        )

        # Clustered network outputs
        network_targets.append(clustered_network_path)
        network_targets.append(f"{resources_path}/network/{rid}_clustering_busmap.csv")
        network_targets.append(f"{resources_path}/validation/{rid}_clustered_network_validation_report.html")
    # Network validation targets for all scenarios (optional - only for clustered networks)
    # network_targets.append(f"{resources_path}/validation/{rid}_network_validation_report.html")

# --- Generator Targets --------------------------------------------------------
# REPD matching, DUKES/FES integration, and reporting
# ------------------------------------------------------------------------------

generator_targets = []

# LEGACY: TEC data processing rules remain available but are not part of the main workflow
# The workflow now uses DUKES (historical) + REPD + FES (future) for generator data
# TEC rules are preserved in rules/generators.smk for reference/debugging if needed

# Per-scenario generator integration
generator_targets += expand(
    f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal.pkl", 
    scenario=run_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_generators_full.csv", 
    scenario=run_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_generators_summary_by_carrier.csv", 
    scenario=run_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_technology_capacity_summary.csv", 
    scenario=run_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_generator_integration_report.txt", 
    scenario=run_ids
)

# Network CSV export (for external analysis)
# generator_targets += expand(
#     f"{resources_path}/network/csv/{{scenario}}_base_demand_generators/export_complete.txt", 
#     scenario=run_ids
# )

# --- Renewable Targets --------------------------------------------------------
# Weather-dependent generation profiles (wind offshore/onshore, solar PV)
# ------------------------------------------------------------------------------

renewable_targets = []

# Renewable profiles organized by renewables_year (not scenario)
renewable_targets += expand(
    f"{resources_path}/renewable/profiles/wind_offshore_{{renewables_year}}.csv", 
    renewables_year=renewables_year
)
renewable_targets += expand(
    f"{resources_path}/renewable/profiles/wind_onshore_{{renewables_year}}.csv", 
    renewables_year=renewables_year
)
renewable_targets += expand(
    f"{resources_path}/renewable/profiles/solar_pv_{{renewables_year}}.csv", 
    renewables_year=renewables_year
)

# Optional: Renewable validation report (commented out for faster workflow)
# renewable_targets += [f"{resources_path}/renewable/validation/renewable_profiles_validation_report.html"]

# --- Flexibility Targets ------------------------------------------------------
# Storage (Battery/Pumped Hydro/LAES), EVs, thermal storage
# ------------------------------------------------------------------------------

flexibility_targets = []

# Storage data extraction and merging (scenario-independent)
flexibility_targets += [
    # Legacy targets - commenting out as they're not produced by current rules
    # f"{resources_path}/storage/storage_from_repd.csv",
    # f"{resources_path}/storage/storage_from_tec.csv",
    # f"{resources_path}/storage/storage_sites_merged.csv",
    f"{resources_path}/storage/storage_parameters.csv"
]

# Per-scenario storage integration into networks
flexibility_targets += expand(
    f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage.pkl", 
    scenario=run_ids
)

# Future flexibility options (placeholders for now) - COMMENTED OUT FOR NOW
# flexibility_targets += [
#     f"{resources_path}/flexibility/ev_placeholder.csv",
#     f"{resources_path}/flexibility/thermal_storage_placeholder.csv"
# ]

# --- Interconnector Targets ---------------------------------------------------
# Cross-border connections (Europe, Ireland, Norway)
# ------------------------------------------------------------------------------

interconnector_targets = []

# Interconnector data cleaning and availability (scenario-independent)
interconnector_targets += [
    f"{resources_path}/interconnectors/interconnectors_clean.csv",
    f"{resources_path}/interconnectors/interconnector_availability.csv",
    # f"{resources_path}/interconnectors/pipeline_placeholder.csv"  # COMMENTED OUT FOR NOW
]

# Per-scenario interconnector mapping and integration
for rid in run_ids:
    network_model = scenarios[rid]["network_model"]
    interconnector_targets += [
        f"{resources_path}/interconnectors/interconnectors_mapped_{network_model}.csv",
        f"{resources_path}/network/{rid}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    ]


# --- Finalization and Optimization Targets -----------------------------------
# Clean scenario.nc files and optimization results
# ------------------------------------------------------------------------------

finalize_targets = []

# Finalized networks with clean names (scenario.nc)
finalize_targets += expand(
    f"{resources_path}/network/{{scenario}}.nc", 
    scenario=run_ids
)

# Network summaries
finalize_targets += expand(
    f"{resources_path}/network/{{scenario}}_network_summary.txt", 
    scenario=run_ids
)

# Optimization results (solving enabled)
optimization_targets = []
optimization_targets += expand(
    f"{resources_path}/network/{{scenario}}_solved.nc", 
    scenario=run_ids
)
optimization_targets += expand(
    f"{resources_path}/network/{{scenario}}_optimization_summary.txt", 
    scenario=run_ids
)

# --- Analysis Outputs --------------------------------------------------------
# Consolidated analysis: spatial plots, dashboards, notebooks
# All outputs in resources/analysis/ folder
# ------------------------------------------------------------------------------

analysis_targets = []

# Spatial plots, dashboards, and notebooks for each solved scenario
analysis_targets += expand(
    f"{resources_path}/analysis/{{scenario}}_spatial.html", 
    scenario=run_ids
)
analysis_targets += expand(
    f"{resources_path}/analysis/{{scenario}}_dashboard.html", 
    scenario=run_ids
)
analysis_targets += expand(
    f"{resources_path}/analysis/{{scenario}}_notebook.ipynb", 
    scenario=run_ids
)

# ------------------------------------------------------------------------------
# MAIN RULE: 'all'
# ------------------------------------------------------------------------------
# Default target when running: snakemake --cores 8
# Uncomment/comment target groups to control what gets built
# ------------------------------------------------------------------------------

rule all:
    input:
        # Core workflow components (always enabled)
        network_targets,          # Networks, clustering, validation
        generator_targets,        # Generator integration and reporting
        renewable_targets,        # Renewable generation profiles
        flexibility_targets,      # Storage integration
        interconnector_targets,   # Cross-border connections
        finalize_targets,         # Clean {scenario}.nc files
        
        # Optimization and analysis (solving enabled)
        optimization_targets,     # Solved networks and results
        analysis_targets,         # Spatial plots, dashboards, notebooks
