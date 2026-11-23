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

# Set GDAL_DATA / PROJ_DATA / PATH when run without `conda activate` (Windows).
# Must happen before any job is spawned so subprocesses inherit the variables.
from scripts.utilities.conda_env import ensure_conda_env_vars
ensure_conda_env_vars()

from scripts.utilities.scenario_detection import (
    is_historical_scenario,
    target_source,
    auto_configure_scenario,
    summarize_scenario_configuration,
    validate_historical_scenario,
    validate_future_scenario,
    validate_scenario_complete
)
from config_loader import load_config, scenario_files, defaults_files, model_scenario_param, scenario_param

# ------------------------------------------------------------------------------
# CONFIGURATION LOADING
# ------------------------------------------------------------------------------

# Load main configuration file
configfile: "config/config.yaml"

# Load scenarios merged over the defaults (config/defaults/: the base, then the method's overlay)
_full_config = load_config()
scenarios = _full_config["scenarios"]  # These are already merged with defaults
# scenarios.yaml and the studies config.yaml lists (config/studies/*.yaml): the files that define scenarios.
SCENARIO_FILES = [p.relative_to(Path(workflow.basedir)).as_posix() for p in scenario_files(Path(workflow.basedir) / "config")]
# Every file of the defaults (config/defaults/): the inputs of a rule that reads them.
DEFAULTS_FILES = [p.relative_to(Path(workflow.basedir)).as_posix() for p in defaults_files(Path(workflow.basedir) / "config")]

# ------------------------------------------------------------------------------
# PATH DEFINITIONS (shared with all rule modules)
# ------------------------------------------------------------------------------

# F20 phase 0: a parity run in a worktree reads the solved outputs of another tree
# (--config resources_path=<that tree>/resources); unset, "resources" as always.
resources_path = str(config.get("resources_path") or "resources")
if resources_path != "resources":
    # Reading another tree's outputs is for a parity run only, which writes the ownership layer
    # into its own root and runs the ownership rules alone (--allowed-rules), so nothing is ever
    # written into the other tree (scripts/public_ownership/f20_parity/run_parity.py).
    from scripts.public_ownership.f20_parity.run_parity import check_roots as _f20_check_roots

    _f20_check_roots(resources_path, config.get("po_resources_path"), config.get("f20_parity"))
data_path = "data"

# ------------------------------------------------------------------------------
# SCENARIO SELECTION
# ------------------------------------------------------------------------------
# Determine which scenarios to run: command-line override or config file
# ------------------------------------------------------------------------------

if config.get("scenario"):
    # Scenarios given on the command line: snakemake --config scenario=HT35, or several at once (solved in
    # parallel within the --cores and --resources budget): snakemake --config scenario=Validation_Jan2020,Validation_Feb2020
    run_ids = [s.strip() for s in str(config["scenario"]).split(",") if s.strip()]
    unknown = [s for s in run_ids if s not in scenarios]
    if unknown:
        raise ValueError(f"--config scenario names unknown scenarios: {unknown}")
    print(f"Using command-line scenario(s): {run_ids}")
else:
    # Use scenarios from config.yaml run_scenarios list
    run_ids = _full_config["run_scenarios"]   # family names expanded to their months (config_loader)
    print(f"Using config file scenarios: {run_ids}")

print(f"Active scenarios: {run_ids}")

# Validate scenario IDs exist in config/scenarios.yaml or a loaded study
unknown = set(run_ids) - set(scenarios)
if unknown:
    raise ValueError(f"Unknown scenario IDs: {unknown}. Valid ids are those in config/scenarios.yaml and in the "
                     "study folders config/studies/<study>/; a study is loaded only when config/config.yaml "
                     "lists it under studies.")

# ------------------------------------------------------------------------------
# SHARED NETWORKS (network_from)
# ------------------------------------------------------------------------------
# A scenario may declare ``network_from: <parent>`` to solve on the parent's
# finalized network instead of building its own (the twelve CP30 2030 months
# are one full-year network solved twelve times: solve_period is applied at
# solve time, and the build reads it only through the renewable performance
# factor registry, which integrate_renewable_generators.py refuses to resolve
# per month for a shared network). The parent's network is built exactly once
# and the child builds nothing. Everything that is *solved* keys on run_ids;
# everything that is *built* keys on build_ids.
#
# The child must be the parent plus solve-time keys only; anything else the
# build could read would make the shared network wrong for that child.
# ------------------------------------------------------------------------------

# Keys (dotted paths) a child may differ on because nothing in the build chain reads them.
# ``market`` and ``solve.solver`` are consumed only by the solve and analysis rules,
# so a counterfactual that changes how a network is dispatched can share the
# network it is dispatched on. ``support.bm_bids``, ``support.revenue`` and
# ``consumer_bill.price_cap`` were inside ``market`` before the config restructure.
NETWORK_FROM_SOLVE_TIME_KEYS = {
    "network_from", "description", "solve_period", "_scenario_id", "market", "solve.solver",
    "support.bm_bids", "support.revenue", "consumer_bill.price_cap",
    # Where a scenario takes its Stage 1 result from is a solve-time choice; the
    # build chain never reads it (item BM1).
    "wholesale_from",
}


def _build_relevant(config_dict):
    # _without_dotted_paths is defined below, beside WHOLESALE_FROM_ALLOWED_DIFFERENCES.
    return _without_dotted_paths(config_dict, NETWORK_FROM_SOLVE_TIME_KEYS)


def _network_from_parents(ids):
    parents = []
    for rid in ids:
        parent = scenarios[rid].get("network_from")
        if not parent:
            continue
        if parent not in scenarios:
            raise ValueError(f"Scenario '{rid}' has network_from '{parent}', which is not a known scenario.")
        if parent == rid:
            raise ValueError(f"Scenario '{rid}' has network_from pointing at itself.")
        if scenarios[parent].get("network_from"):
            raise ValueError(
                f"Scenario '{rid}' has network_from '{parent}', which itself has network_from "
                f"'{scenarios[parent]['network_from']}'. Point every child at the built parent."
            )
        # A parent restricted to a period may still be shared, but only by a
        # child solving exactly that period: the build reads solve_period through
        # the renewable performance-factor registry, so two different periods
        # could need two different networks.
        parent_period = scenarios[parent].get("solve_period", {}) or {}
        child_period = scenarios[rid].get("solve_period", {}) or {}
        if parent_period.get("enabled", False) and parent_period != child_period:
            raise ValueError(
                f"network_from parent '{parent}' has an enabled solve_period that differs from "
                f"'{rid}'. Share a whole-year parent, or give the child the parent's period."
            )
        child_build, parent_build = _build_relevant(scenarios[rid]), _build_relevant(scenarios[parent])
        differing = sorted(
            k for k in set(child_build) | set(parent_build)
            if child_build.get(k) != parent_build.get(k)
        )
        if differing:
            raise ValueError(
                f"Scenario '{rid}' cannot share the network of '{parent}': these build-relevant "
                f"keys differ: {differing}. Only {sorted(NETWORK_FROM_SOLVE_TIME_KEYS)} may differ."
            )
        if parent not in parents:
            parents.append(parent)
    return parents


# ------------------------------------------------------------------------------
# SHARED WHOLESALE RESULTS (wholesale_from)
# ------------------------------------------------------------------------------
# A scenario may declare ``wholesale_from: <parent>`` to take the parent's Stage 1
# dispatch, price and storage CSVs instead of re-solving an identical wholesale
# market. Item BM1 is why: a public-ownership balancing counterfactual changes
# only ``market.balancing``, so its wholesale stage is identical by construction
# -- the seven-day probe confirmed 0.0 MW and 0.0 GBP/MWh difference -- and
# re-solving it twelve times costs about three CPU-hours to reproduce a file that
# already exists.
#
# The check below makes that equality structural rather than lucky: everything
# the wholesale stage reads must match, so only ``market.balancing`` (and the
# descriptive keys) may differ. Sharing across a different period, network or
# wholesale configuration is refused at parse time.
# validation.stage1 (19 Oct 2026): how the Stage 1 check judges its family (energy_basis), reporting only.
# support.bm_bids: how the BM bids renewables, inside market.balancing before the config restructure.
# Dotted paths.
WHOLESALE_FROM_ALLOWED_DIFFERENCES = {"wholesale_from", "description", "_scenario_id", "validation.stage1",
                                      "support.bm_bids"}


def _without_dotted_paths(config_dict, paths):
    """``config_dict`` without the dotted ``paths`` (copied only along the paths it changes)."""
    out = dict(config_dict)
    for path in paths:
        parts = path.split(".")
        cur = out
        for part in parts[:-1]:
            if not isinstance(cur.get(part), dict):
                cur = None
                break
            cur[part] = dict(cur[part])
            cur = cur[part]
        if cur is not None:
            cur.pop(parts[-1], None)
    return out


def _wholesale_from_parents(ids):
    for rid in ids:
        parent = scenarios[rid].get("wholesale_from")
        if not parent:
            continue
        if parent not in scenarios:
            raise ValueError(f"Scenario '{rid}' has wholesale_from '{parent}', which is not a known scenario.")
        if parent == rid:
            raise ValueError(f"Scenario '{rid}' has wholesale_from pointing at itself.")
        if scenarios[parent].get("wholesale_from"):
            raise ValueError(
                f"Scenario '{rid}' has wholesale_from '{parent}', which itself has wholesale_from "
                f"'{scenarios[parent]['wholesale_from']}'. Point every child at the solved parent."
            )
        if scenarios[rid].get("solve_period") != scenarios[parent].get("solve_period"):
            raise ValueError(
                f"Scenario '{rid}' cannot share the wholesale solve of '{parent}': they solve "
                "different periods."
            )
        if _build_source(rid) != _build_source(parent):
            raise ValueError(
                f"Scenario '{rid}' cannot share the wholesale solve of '{parent}': they solve on "
                "different networks."
            )
        child = _without_dotted_paths(scenarios[rid], WHOLESALE_FROM_ALLOWED_DIFFERENCES)
        parent_cfg = _without_dotted_paths(scenarios[parent], WHOLESALE_FROM_ALLOWED_DIFFERENCES)
        differing = sorted(
            k for k in set(child) | set(parent_cfg)
            if child.get(k) != parent_cfg.get(k)
        )
        # ``market`` is compared by its non-balancing content, because changing
        # how the BM is bid is exactly what this mechanism exists to allow.
        if differing == ["market"]:
            child_market = {k: v for k, v in (child.get("market") or {}).items() if k != "balancing"}
            parent_market = {k: v for k, v in (parent_cfg.get("market") or {}).items() if k != "balancing"}
            differing = [] if child_market == parent_market else ["market (outside balancing)"]
        if differing:
            raise ValueError(
                f"Scenario '{rid}' cannot share the wholesale solve of '{parent}': these keys differ: "
                f"{differing}. Only market.balancing may differ."
            )


def _build_source(scenario_id):
    return scenarios.get(scenario_id, {}).get("network_from") or scenario_id


_network_from_parents_list = _network_from_parents(run_ids)
_wholesale_from_parents(run_ids)
build_ids = [rid for rid in run_ids if not scenarios[rid].get("network_from")]
build_ids += [p for p in _network_from_parents_list if p not in build_ids]
if _network_from_parents_list:
    _shared = {p: [r for r in run_ids if scenarios[r].get("network_from") == p] for p in _network_from_parents_list}
    for _parent, _children in _shared.items():
        print(f"Shared network: '{_parent}' is built once for {len(_children)} scenario(s)")

# ------------------------------------------------------------------------------
# SCENARIO AUTO-CONFIGURATION AND VALIDATION
# ------------------------------------------------------------------------------
# Automatically detect historical vs future scenarios and configure data sources
# Validate configuration for both scenario types
# Check data freshness for historical scenarios
# ------------------------------------------------------------------------------

import logging
logging.basicConfig(level=logging.INFO)  # Enable INFO logging during initialization

enhanced_scenarios = {}
validation_errors = []
validation_warnings = []
validation_info = []

for rid in build_ids + [r for r in run_ids if r not in build_ids]:
    # Auto-configure with metadata (network_from parents are validated too:
    # their build runs even though they are never solved on their own)
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


def _demand_config_ids(key):
    """The scenarios whose demand configs the build reads: run_ids, plus each network_from
    parent that is not itself a run id (it is built, so finalize_demand reads its config).

    Such a parent enters only when its ``demand.<key>`` block enables something (the block itself or
    a sub-block such as ``demand.disaggregation.fes``). A parent with nothing enabled keeps
    the ``{'enabled': False}`` default it has always had, so its finalize_demand params do not
    change (the CP30 2030 parent). FES 2050 plan, audit list A item 19: before this, the
    FES25_HT_2050 parent built without its FES 2050 demand components when only its months ran.
    """
    def _enables(block):
        return isinstance(block, dict) and (bool(block.get("enabled")) or any(
            isinstance(v, dict) and v.get("enabled") for v in block.values()))

    extra = [p for p in build_ids if p not in run_ids and _enables((scenarios[p].get("demand") or {}).get(key))]
    return list(run_ids) + extra


def extract_disaggregation_configs():
    """
    Extract demand disaggregation configurations from scenarios.
    Returns dictionary with scenario IDs as keys.

    Returns:
        dict: {scenario_id: disaggregation_config} or {scenario_id: {'enabled': False}}
    """
    configs = {}
    for rid in _demand_config_ids("disaggregation"):
        if "disaggregation" in (scenarios[rid].get("demand") or {}):
            configs[rid] = scenarios[rid]["demand"]["disaggregation"]
        else:
            configs[rid] = {"enabled": False}  # Default: disabled
    return configs


def is_flexibility_enabled(scenario_id):
    """
    Check if demand flexibility is enabled for a scenario.

    Returns:
        bool: True if flexibility is enabled
    """
    if "flexibility" in (scenarios[scenario_id].get("demand") or {}):
        return scenarios[scenario_id]["demand"]["flexibility"].get("enabled", False)
    return False


def extract_flexibility_configs():
    """
    Extract demand flexibility configurations from scenarios.
    Returns dictionary with scenario IDs as keys.

    Returns:
        dict: {scenario_id: flexibility_config} or {scenario_id: {'enabled': False}}
    """
    configs = {}
    for rid in _demand_config_ids("flexibility"):
        if "flexibility" in (scenarios[rid].get("demand") or {}):
            configs[rid] = scenarios[rid]["demand"]["flexibility"]
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
        clustering_config = (scenarios[rid].get("network") or {}).get("clustering", None)
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
demand_timeseries         = sorted({(scenarios[rid].get("demand") or {})["timeseries"] for rid in run_ids
                                    if "timeseries" in (scenarios[rid].get("demand") or {})})
network_models            = extract_from_scenarios("network_model")
renewables_year           = extract_from_scenarios("renewables_year")
demand_disaggregation_configs = extract_disaggregation_configs()
demand_flexibility_configs = extract_flexibility_configs()

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
include: "rules/neso_documents.smk"       # NESO website documents fetched, not shipped
include: "rules/CP30.smk"                 # Clean Power 2030 data
include: "rules/fes_long_range.smk"        # FES long-range chain: targets, prices, station schedule, validation (Y1, stage 5)
include: "rules/generator_data_prep.smk"  # Legacy DUKES/TEC data preparation
include: "rules/generators.smk"           # Generator integration (DUKES/REPD/FES - historical vs future routing)
include: "rules/storage.smk"              # Energy storage integration (battery, pumped hydro, CAES, LAES)
include: "rules/hydrogen.smk"             # Hydrogen system (electrolysis, H2 storage, H2 turbines)
include: "rules/interconnectors.smk"      # Cross-border interconnections
include: "rules/solve.smk"                # Network finalization and optimization
include: "rules/analysis.smk"             # Post-solve analysis: spatial plots, dashboards, notebooks
include: "rules/market.smk"               # Market simulation: wholesale + balancing mechanism
include: "rules/turn_down_allocation.smk" # Pro rata zonal turn-down re-pricing (both bases; 20 Oct 2026)
include: "rules/consumer_bill.smk"        # Consumer bill: Ofgem accounting + forecast interfaces
include: "rules/public_ownership.smk"     # Opt-in fixed-dispatch finance/accounting layer

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
# whose network is built (build_ids: run_ids without network_from, plus the
# network_from parents; see SHARED NETWORKS above).
network_targets += expand(
    f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc",
    scenario=build_ids
)

# Add clustering and validation targets based on clustering configuration
for rid in build_ids:
    network_model = scenarios[rid]["network_model"]
    # Check if clustering is enabled - can be a string preset name or dict config
    clustering_config = (scenarios[rid].get("network") or {}).get("clustering", None)
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
        clustered_network_path = f"{resources_path}/network/{rid}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"

        # Clustered network outputs
        network_targets.append(clustered_network_path)
        network_targets.append(f"{resources_path}/network/{rid}_clustering_busmap.csv")
        network_targets.append(f"{resources_path}/validation/{rid}_clustered_network_validation_report.html")
    # Network validation targets for all scenarios (optional - only for clustered networks)
    # network_targets.append(f"{resources_path}/validation/{rid}_network_validation_report.html")

# --- CP30 Validation Targets --------------------------------------------------
# Exact target-vs-actual component audit for active CP30 scenarios.
# ------------------------------------------------------------------------------

cp30_validation_targets = []
_cp30_ids = [
    rid for rid in build_ids
    if target_source(scenarios.get(rid, {})) == "CP30"
]
for rid in _cp30_ids:
    cp30_validation_targets.extend([
        f"{resources_path}/CP30/validation/{rid}_component_totals.csv",
        f"{resources_path}/CP30/validation/{rid}_network_reference_summary.csv",
        f"{resources_path}/CP30/validation/{rid}_network_assumption.csv",
        f"{resources_path}/CP30/validation/{rid}_energy_totals.csv",
    ])
    # Where capacity sits (15 Oct 2026): rule validate_CP30_locational, rules/CP30.smk.
    if ((scenarios.get(rid, {}).get("validation") or {}).get("locational") or {}).get("enabled", True):
        cp30_validation_targets.append(f"{resources_path}/CP30/validation/{rid}_locational_checks.csv")
        # Fleet location and bus positions (19 Oct 2026): rule validate_CP30_fleet_location, rules/CP30.smk.
        if ((scenarios.get(rid, {}).get("validation") or {}).get("fleet_location") or {}).get("enabled", True):
            cp30_validation_targets.append(f"{resources_path}/CP30/validation/{rid}_fleet_location.csv")
# Stage 1 against CP30 (16 Oct 2026): rule validate_CP30_stage1, rules/CP30.smk, one per
# twelve-month Stage 1 family in run_scenarios (validation.stage1.enabled).
cp30_validation_targets.extend(CP30_STAGE1_VALIDATION_TARGETS)
cp30_validation_targets.extend(CP30_BALANCING_VALIDATION_TARGETS)  # NESO balancing check, rules/CP30.smk (studies/cp30_2030/study.yaml balancing_validation_families)
cp30_validation_targets.extend(SECURED_CAPABILITY_AUDIT_TARGETS)  # boundary capability by family, rules/CP30.smk (report only)
cp30_validation_targets.extend(GAS_SCHEDULE_CHECK_TARGETS)  # gas schedule CM / FES retention columns, rules/CP30.smk (report only)
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
    scenario=build_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_generators_full.csv",
    scenario=build_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_generators_summary_by_carrier.csv",
    scenario=build_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_technology_capacity_summary.csv",
    scenario=build_ids
)
generator_targets += expand(
    f"{resources_path}/generators/{{scenario}}_generator_integration_report.txt",
    scenario=build_ids
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
    scenario=build_ids
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

# Interconnector data cleaning (scenario-independent). Availability is
# per scenario ({scenario}_interconnector_availability.csv) and is pulled in
# as an input of add_interconnectors_to_network; no rule writes a global file.
interconnector_targets += [
    f"{resources_path}/interconnectors/interconnectors_clean.csv",
    # f"{resources_path}/interconnectors/pipeline_placeholder.csv"  # COMMENTED OUT FOR NOW
]

# Per-scenario interconnector mapping and integration
for rid in build_ids:
    interconnector_targets += [
        f"{resources_path}/interconnectors/{rid}_interconnectors_mapped.csv",
        f"{resources_path}/network/{rid}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    ]


# --- Finalization and Optimization Targets -----------------------------------
# Clean scenario.nc files and optimization results
# ------------------------------------------------------------------------------

finalize_targets = []

# Finalized networks with clean names (scenario.nc), one per built network
finalize_targets += expand(
    f"{resources_path}/network/{{scenario}}.nc",
    scenario=build_ids
)

# Network summaries
finalize_targets += expand(
    f"{resources_path}/network/{{scenario}}_network_summary.txt",
    scenario=build_ids
)

# Optimization results (solving enabled)
# Market-enabled scenarios use two-stage dispatch instead of standard solve
_market_ids = [rid for rid in run_ids if scenarios.get(rid, {}).get('market', {}).get('enabled', False)]
_solve_ids = [rid for rid in run_ids if rid not in _market_ids]

optimization_targets = []
optimization_targets += expand(
    f"{resources_path}/network/{{scenario}}_solved.nc", 
    scenario=_solve_ids
)
optimization_targets += expand(
    f"{resources_path}/network/{{scenario}}_optimization_summary.txt", 
    scenario=_solve_ids
)

# --- Market Simulation Targets ------------------------------------------------
# Two-stage dispatch: wholesale (copperplate) + balancing mechanism (redispatch)
# Only built for scenarios with market.enabled: true
# ------------------------------------------------------------------------------

market_targets = []
for rid in run_ids:
    if scenarios.get(rid, {}).get('market', {}).get('enabled', False):
        # Wholesale outputs (always included for market scenarios). A
        # wholesale_from child has no Stage 1 of its own: request its parent's.
        wholesale_rid = _wholesale_source(rid)
        market_targets.extend([
            f"{resources_path}/market/{wholesale_rid}_wholesale.nc",
            f"{resources_path}/market/{wholesale_rid}_wholesale_dispatch.csv",
            f"{resources_path}/market/{wholesale_rid}_wholesale_storage.csv",
            f"{resources_path}/market/{wholesale_rid}_wholesale_links.csv",
            f"{resources_path}/market/{wholesale_rid}_wholesale_price.csv",
        ])

        # This depends only on the wholesale stage, so it also works with
        # wholesale-only market scenarios.
        if _consumer_bill_config(rid).get('enabled', False):
            market_targets.extend([
                f"{resources_path}/analysis/{rid}_consumer_tariff.csv",
                f"{resources_path}/analysis/{rid}_consumer_bill.csv",
                f"{resources_path}/analysis/{rid}_consumer_bill_summary.json",
            ])

        # A monthly support-scheme validation is independent of the retail
        # cost-stack switch. Request its CfD and ROC reconciliations whenever
        # support.audit is enabled, so a plain ``snakemake`` run for the
        # configured scenario produces the intended audit outputs.
        if rid in _support_monthly_ids:
            # The CfD side needs the year's reviewed contract mapping, so it is
            # requested only where detailed CfD reconciliation is enabled (the
            # default is on for 2020 only); the ROC side runs for every month.
            if _consumer_detailed_cfd_enabled(rid):
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_lccc_cfd_model_dispatch_reconciliation.csv",
                    f"{resources_path}/analysis/{rid}_lccc_cfd_model_dispatch_reconciliation_summary.json",
                    f"{resources_path}/analysis/{rid}_renewable_performance_factor_contracts.csv",
                    f"{resources_path}/analysis/{rid}_renewable_performance_factor_assets.csv",
                    f"{resources_path}/analysis/{rid}_renewable_performance_factor_cohorts.csv",
                    f"{resources_path}/analysis/{rid}_renewable_performance_factor_summary.json",
                    f"{resources_path}/analysis/{rid}_historical_alignment_scorecard.csv",
                    f"{resources_path}/analysis/{rid}_historical_alignment_scorecard_summary.json",
                ])
            market_targets.extend([
                f"{resources_path}/analysis/{rid}_roc_model_dispatch_reconciliation.csv",
                f"{resources_path}/analysis/{rid}_roc_model_dispatch_reconciliation_summary.json",
                f"{resources_path}/analysis/{rid}_roc_station_representation_mapping.csv",
                f"{resources_path}/analysis/{rid}_roc_station_representation_mapping_summary.json",
                f"{resources_path}/analysis/{rid}_roc_official_station_ledger.csv",
                f"{resources_path}/analysis/{rid}_roc_official_residual_cohorts.csv",
                f"{resources_path}/analysis/{rid}_roc_official_residual_cohorts_summary.json",
                f"{resources_path}/analysis/{rid}_roc_mapping_priority_worklist.csv",
                f"{resources_path}/analysis/{rid}_roc_mapping_priority_candidates.csv",
                f"{resources_path}/analysis/{rid}_roc_mapping_priority_summary.json",
                f"{resources_path}/analysis/{rid}_roc_unresolved_source_audit.csv",
                f"{resources_path}/analysis/{rid}_roc_unresolved_source_audit_candidates.csv",
                f"{resources_path}/analysis/{rid}_roc_unresolved_source_audit_summary.json",
            ])

        if rid in _consumer_cost_stack_ids:
            market_targets.extend([
                f"{resources_path}/analysis/{rid}_consumer_bill_input_bundle.json",
                f"{resources_path}/analysis/{rid}_wholesale_energy_impact.csv",
                f"{resources_path}/analysis/{rid}_forward_procurement_periods.csv",
                f"{resources_path}/analysis/{rid}_retail_cost_stack.csv",
                f"{resources_path}/analysis/{rid}_retail_cost_stack_summary.json",
                f"{resources_path}/analysis/{rid}_ofgem_benchmark_component_register.csv",
                f"{resources_path}/analysis/{rid}_ofgem_benchmark_component_register_summary.json",
                f"{resources_path}/analysis/{rid}_consumer_bill_component_register.csv",
                f"{resources_path}/analysis/{rid}_consumer_bill_component_register_summary.json",
                f"{resources_path}/analysis/{rid}_consumer_bill_forecast_registry.csv",
                f"{resources_path}/analysis/{rid}_consumer_bill_forecast_registry_summary.json",
                f"{resources_path}/analysis/{rid}_consumer_bill_breakdown.csv",
                f"{resources_path}/analysis/{rid}_consumer_bill_breakdown_summary.json",
                f"{resources_path}/analysis/{rid}_consumer_bill_breakdown.html",
                f"{resources_path}/analysis/{rid}_consumer_bill_review.html",
            ])
            if rid in _consumer_historical_support_outturn_ids:
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_historical_support_outturn.csv",
                    f"{resources_path}/analysis/{rid}_historical_support_outturn_summary.json",
                ])
            if _consumer_bill_config(rid).get("forward_procurement", {}).get("method") == "pypsa_spot_to_procurement_raw_backtest":
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_pypsa_spot_procurement_bridge.csv",
                    f"{resources_path}/analysis/{rid}_pypsa_spot_procurement_periods.csv",
                    f"{resources_path}/analysis/{rid}_pypsa_spot_procurement_bridge_summary.json",
                    f"{resources_path}/analysis/{rid}_pypsa_spot_procurement_bridge.html",
                ])
            else:
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_ofgem_forward_procurement_bridge.csv",
                    f"{resources_path}/analysis/{rid}_ofgem_forward_procurement_bridge_summary.json",
                    f"{resources_path}/analysis/{rid}_ofgem_forward_procurement_bridge.html",
                ])
            # The methodology audit uses the two-stage market summary. It is
            # therefore unavailable to wholesale-only bill sensitivities.
            if not scenarios.get(rid, {}).get('market', {}).get('wholesale_only', False):
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_consumer_bill_methodology_audit.csv",
                    f"{resources_path}/analysis/{rid}_consumer_bill_methodology_audit_summary.json",
                    f"{resources_path}/analysis/{rid}_consumer_bill_methodology_audit.html",
                    f"{resources_path}/analysis/{rid}_bsuos_redispatch_reconciliation.csv",
                    f"{resources_path}/analysis/{rid}_bsuos_redispatch_reconciliation_summary.json",
                ])
                if rid in _consumer_network_layer_ids:
                    market_targets.extend([
                        f"{resources_path}/analysis/{rid}_network_cost_tariff_layers.csv",
                        f"{resources_path}/analysis/{rid}_network_cost_allowed_revenue_projection.csv",
                        f"{resources_path}/analysis/{rid}_network_cost_neso_tnuos_historic_comparator.csv",
                        f"{resources_path}/analysis/{rid}_network_cost_neso_tnuos_future_outlook.csv",
                        f"{resources_path}/analysis/{rid}_network_cost_layers_summary.json",
                    ])
            if rid in _consumer_backtest_ids:
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_consumer_bill_backtest.csv",
                    f"{resources_path}/analysis/{rid}_consumer_bill_backtest_summary.json",
                    f"{resources_path}/analysis/{rid}_consumer_bill_backtest.html",
                ])
                # Detailed reconciliation is explicitly enabled per historic
                # year. Its mapping is year-specific; automatic matching never
                # reuses a reviewed 2020 override in a later fleet.
                if rid in _consumer_lccc_audit_ids:
                    market_targets.extend([
                        f"{resources_path}/analysis/{rid}_lccc_cfd_contract_mapping.csv",
                        f"{resources_path}/analysis/{rid}_lccc_cfd_contract_mapping_summary.json",
                        f"{resources_path}/analysis/{rid}_lccc_cfd_model_dispatch_reconciliation.csv",
                        f"{resources_path}/analysis/{rid}_lccc_cfd_model_dispatch_reconciliation_summary.json",
                    ])
        # Wholesale-only notebook (generated when wholesale_only: true)
        if scenarios.get(rid, {}).get('market', {}).get('wholesale_only', False):
            market_targets.append(
                f"{resources_path}/analysis/{rid}_wholesale_notebook.ipynb"
            )
        # An explicit historical market-coupled case leaves flows optimizable and
        # compares them with ESPENI after the wholesale solve. Fixed-flow
        # historical scenarios intentionally do not request this diagnostic.
        if (
            scenarios.get(rid, {}).get('interconnectors', {}).get(
                'historical_flow_mode', 'fixed'
            ) == 'market_coupled'
        ):
            market_targets.extend([
                f"{resources_path}/analysis/{rid}_interconnector_backtest_summary.csv",
                f"{resources_path}/analysis/{rid}_interconnector_backtest_timeseries.csv",
            ])
        # Balancing + analysis outputs (skipped when wholesale_only: true)
        if not scenarios.get(rid, {}).get('market', {}).get('wholesale_only', False):
            market_targets.extend([
                f"{resources_path}/market/{rid}_balancing.nc",
                f"{resources_path}/market/{rid}_balancing_dispatch.csv",
                f"{resources_path}/market/{rid}_redispatch_summary.csv",
                f"{resources_path}/market/{rid}_constraint_costs.csv",
                f"{resources_path}/market/{rid}_congestion.csv",
                f"{resources_path}/market/{rid}_price_comparison.csv",
                f"{resources_path}/analysis/{rid}_market_dashboard.html",
                f"{resources_path}/analysis/{rid}_market_summary.json",
                f"{resources_path}/analysis/{rid}_market_notebook.ipynb",
            ])
            if (
                scenarios.get(rid, {}).get("fleet", {}).get("spatial_proxy", {}).get(
                    "offshore_connection_allocation", {}
                ).get("audit", False)
            ):
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_offshore_link_audit.csv",
                    f"{resources_path}/analysis/{rid}_offshore_link_audit_summary.json",
                ])
            if (
                scenarios.get(rid, {}).get("fleet", {}).get("spatial_proxy", {}).get(
                    "offshore_connection_allocation", {}
                ).get("audit_mapping", False)
            ):
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_offshore_landing_mapping_review.csv",
                    f"{resources_path}/analysis/{rid}_offshore_landing_mapping_review_summary.json",
                ])
            if (
                scenarios.get(rid, {}).get("fleet", {}).get("spatial_proxy", {}).get(
                    "offshore_connection_allocation", {}
                ).get("audit_project_crosswalk", False)
            ):
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_offshore_project_crosswalk_audit.csv",
                    f"{resources_path}/analysis/{rid}_offshore_project_crosswalk_summary.json",
                ])
            if scenarios.get(rid, {}).get("network", {}).get("topology_adequacy", {}).get("enabled", False):
                market_targets.extend([
                    f"{resources_path}/analysis/{rid}_topology_adequacy_summary.json",
                    f"{resources_path}/analysis/{rid}_adequacy_binding_components.csv",
                    f"{resources_path}/analysis/{rid}_adequacy_shedding_by_bus.csv",
                ])
            # Participation reporting is useful for future CP30 cases too, but
            # only scenarios that explicitly opt in request this extra output.
            if (
                scenarios.get(rid, {})
                .get('market', {})
                .get('balancing', {})
                .get('participation_audit', {})
                .get('enabled', False)
            ):
                market_targets.append(
                    f"{resources_path}/market/{rid}_bm_participation.csv"
                )

            # BM validation for historical market scenarios
            modelled_year = scenarios.get(rid, {}).get('modelled_year', 9999)
            if modelled_year <= 2024:
                market_targets.extend([
                    f"{resources_path}/market/{rid}_bm_validation.csv",
                    f"{resources_path}/market/{rid}_boav_by_carrier.csv",
                    f"{resources_path}/market/{rid}_redispatch_mapped_unmapped.csv",
                    f"{resources_path}/analysis/{rid}_bm_validation.html",
                    f"{resources_path}/market/{rid}_neso_validation.csv",
                    f"{resources_path}/analysis/{rid}_neso_validation.html",
                ])
                # Calibration scorecard: only meaningful when ELEXON BOD data
                # is in use (auto or explicit). Derived-pricing scenarios have
                # no ELEXON median to compare against.
                bid_offer_source = (
                    scenarios.get(rid, {})
                    .get('market', {})
                    .get('balancing', {})
                    .get('bid_offer_source', 'auto')
                )
                if bid_offer_source in ('auto', 'elexon'):
                    market_targets.extend([
                        f"{resources_path}/market/{rid}_bm_calibration.csv",
                        f"{resources_path}/analysis/{rid}_bm_calibration.html",
                    ])

# --- Public station-identity registry -----------------------------------------
# A registry is shared by any active scenarios that request the same explicit
# registry_id.  It is source data only: it does not alter dispatch or mappings.
station_identity_registry_targets = []
for rid in run_ids:
    registry_cfg = scenarios.get(rid, {}).get("fleet", {}).get("station_registry", {})
    if registry_cfg.get("enabled", False):
        registry_id = str(registry_cfg.get("registry_id", "gb_public_current"))
        registry_root = f"{resources_path}/market/station_identity/{registry_id}"
        station_identity_registry_targets.extend([
            f"{registry_root}/station_identity_registry.csv",
            f"{registry_root}/station_identity_registry_evidence.json",
            f"{registry_root}/bmu_station_support_crosswalk.csv",
            f"{registry_root}/bmu_station_support_review.csv",
            f"{registry_root}/bmu_station_support_summary.json",
        ])
station_identity_registry_targets = list(dict.fromkeys(station_identity_registry_targets))
# --- Analysis Outputs --------------------------------------------------------
# Consolidated analysis: spatial plots, dashboards, notebooks
# All outputs in resources/analysis/ folder
# ------------------------------------------------------------------------------

analysis_targets = []

# Spatial plots, dashboards, and notebooks for each solved scenario
# Market-enabled scenarios have their own analysis via market.smk
analysis_targets += expand(
    f"{resources_path}/analysis/{{scenario}}_spatial.html", 
    scenario=_solve_ids
)
analysis_targets += expand(
    f"{resources_path}/analysis/{{scenario}}_dashboard.html", 
    scenario=_solve_ids
)
analysis_targets += expand(
    f"{resources_path}/analysis/{{scenario}}_notebook.ipynb", 
    scenario=_solve_ids
)

# ------------------------------------------------------------------------------
# MAIN RULE: 'all'
# ------------------------------------------------------------------------------
# Default target when running: snakemake --cores 8
# Uncomment/comment target groups to control what gets built
# ------------------------------------------------------------------------------

rule all:
    input:
        # Tariff-only bill cases are independent of market optimisation.
        expand(f"{resources_path}/analysis/{{scenario}}_consumer_bill.csv", scenario=_consumer_bill_ids),
        expand(f"{resources_path}/analysis/{{scenario}}_consumer_bill_review.html", scenario=_consumer_cost_stack_ids),
        consumer_bill_historical_report_targets,  # Pinned historical replay reports
        # Core workflow components (always enabled)
        network_targets,          # Networks, clustering, validation
        cp30_validation_targets,  # CP30 target-vs-actual audit outputs
        generator_targets,        # Generator integration and reporting
        renewable_targets,        # Renewable generation profiles
        flexibility_targets,      # Storage integration
        interconnector_targets,   # Cross-border connections
        finalize_targets,         # Clean {scenario}.nc files
        
        # Optimization and analysis (solving enabled)
        optimization_targets,     # Solved networks and results
        analysis_targets,         # Spatial plots, dashboards, notebooks
        
        # Market simulation (only for market-enabled scenarios)
        market_targets,           # Wholesale + BM dispatch, dashboards
        consumer_support_audit_targets, # Annual ROC/CfD audit assembled from monthly runs
        station_identity_registry_targets,  # Opt-in public source registry
        public_ownership_targets,  # Opt-in ownership financing accounts


# ── Code dependencies of the bill layer (user, 15 Nov 2026) ─────────────────────────────────────────────────────────
# Every rule in the bill layer (consumer_bill.smk, public_ownership.smk) declares the project code it runs: its script
# and every project module that script imports, transitively (scripts/utilities/code_fingerprint.py). Snakemake then
# rebuilds an output whenever that code changes, so no run needs -R (forcing rules made Snakemake drop the P2-bid
# registers from its plan, 14-15 Nov). Rules whose outputs a solve reads (the share and commissioning-year tables) get
# their code as ancient(): a change never re-solves by itself; the results audit's code and content checks and the
# ownership-table checks stop the run instead, for the user to decide.
from scripts.utilities.code_fingerprint import closure as _cf_closure, entry_files as _cf_entries, ROOT as _CF_ROOT

_BILL_LAYER_SNAKEFILES = ("rules/consumer_bill.smk", "rules/public_ownership.smk")
_SOLVE_INPUT_RULES = {"build_public_ownership_deployment_share_tables", "build_public_ownership_state_shares",
                      "build_public_ownership_commissioning_years"}


def _rule_code(rule):
    """The project code a rule runs: scripts its command names, its script: directive, its declared .py inputs."""
    entries = list(_cf_entries(str(getattr(rule, "shellcmd", "") or "")))
    script = getattr(rule, "script", None)
    if script:
        path = (Path(str(rule.snakefile)).parent / str(script)).resolve()
        if path.is_file():
            entries.append(path)
    for item in rule.input:
        if isinstance(item, str) and item.startswith("scripts/") and item.endswith(".py") and (_CF_ROOT / item).is_file():
            entries.append(_CF_ROOT / item)
    return sorted(str(p.relative_to(_CF_ROOT)).replace("\\", "/") for p in _cf_closure(sorted(set(entries))))


for _rule in list(workflow.rules):
    if not str(getattr(_rule, "snakefile", "")).replace("\\", "/").endswith(_BILL_LAYER_SNAKEFILES):
        continue
    _code = _rule_code(_rule)
    if _code:
        _rule.set_input(auto_code=[ancient(f) for f in _code] if _rule.name in _SOLVE_INPUT_RULES else _code)
