# ══════════════════════════════════════════════════════════════════════════════
# Network Building Rules for PyPSA-GB
# ══════════════════════════════════════════════════════════════════════════════
"""
Network topology construction for PyPSA-GB energy system model.

This module handles building three network representation types:
  • ETYS: Full transmission network from NESO ETYS data (400+ buses)
  • Reduced: Simplified regional network (30-50 buses)
  • Zonal: Zone-based network with aggregated links (17 zones)

WORKFLOW OVERVIEW:
  1. Build base network topology (buses, lines, transformers)
  2. Validate network connectivity and basic properties

NETWORK MODEL SELECTION:
  - Specified per-scenario in scenarios_master.yaml via 'network_model' parameter
  - Different scenarios can use different network models simultaneously
  - Wildcards constrained to prevent invalid scenario/model combinations

DATA SOURCES:
  • ETYS: NESO ETYS Appendix B (network topology), GB_network.xlsx (coordinates)
  • Reduced: CSV files with simplified bus/line topology
  • Zonal: CSV files with zone definitions and inter-zone links

DOCUMENTATION:
  • docs/NETWORK_BUILD_RULE.md
  
For network clustering rules, see network_clustering.smk
"""

# ──────────────────────────────────────────────────────────────────────────────
# PATH DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

resources_path = "resources"
data_path = "data"

# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO FILTERING AND WILDCARD CONSTRAINTS
# ──────────────────────────────────────────────────────────────────────────────
# Pre-compute which scenarios use which network models to constrain wildcards
# This prevents Snakemake from attempting invalid scenario/model combinations
# Uses: scenarios, run_ids from main Snakefile
# ──────────────────────────────────────────────────────────────────────────────

import re

# Use scenarios and run_ids from main Snakefile (no config reloading)
_sc_defs = scenarios
_run_ids = run_ids

# Categorize scenarios by network model type
_etys_scenarios = sorted([
    sid for sid in _run_ids
    if _sc_defs.get(sid, {}).get("network_model") == "ETYS"
])
_reduced_scenarios = sorted([
    sid for sid in _run_ids
    if _sc_defs.get(sid, {}).get("network_model") == "Reduced"
])
_zonal_scenarios = sorted([
    sid for sid in _run_ids
    if _sc_defs.get(sid, {}).get("network_model") == "Zonal"
])
_other_scenarios = sorted(_reduced_scenarios + _zonal_scenarios)

def _regex_from_list(items):
    """
    Generate regex pattern that matches exactly the items in the list.
    Returns never-matching pattern if list is empty.
    
    Args:
        items (list): List of scenario IDs
    
    Returns:
        str: Regex pattern matching any item in list
    """
    if not items:
        return r"(?!x)x"  # Never-matching pattern
    return r"(?:%s)" % "|".join(map(re.escape, items))

# Pre-compiled regex patterns for wildcard constraints
ETYS_SCENARIO_REGEX = _regex_from_list(_etys_scenarios)
REDUCED_SCENARIO_REGEX = _regex_from_list(_reduced_scenarios)
ZONAL_SCENARIO_REGEX = _regex_from_list(_zonal_scenarios)
OTHER_SCENARIO_REGEX = _regex_from_list(_other_scenarios)
ALL_SCENARIO_REGEX = _regex_from_list(_run_ids)

# Ensure explicit network construction rules take precedence over the
# model->scenario mapping rule to avoid ambiguous rule selection when both
# could produce the same scenario-specific filename.
ruleorder: build_ETYS_base_network > build_reduced_base_network > build_zonal_base_network > model_to_scenario_network


# Map model-based base network names (e.g. ETYS_base.nc) to scenario-specific
# filenames (e.g. Historical_2020_clustered_base.nc) so downstream rules that
# expect per-scenario network files can find them without duplicating network
# construction logic. This is a simple copy rule and will noop if the target
# already exists.
rule model_to_scenario_network:
    input:
        model_network=lambda wildcards: f"{resources_path}/network/{scenarios[wildcards.scenario]['network_model']}_base.nc"
    output:
        scenario_network=f"{resources_path}/network/{{scenario}}_network.nc"
    wildcard_constraints:
        scenario=ALL_SCENARIO_REGEX
    params:
        scenario=lambda wildcards: wildcards.scenario
    log:
        "logs/network_build/map_model_to_scenario_{scenario}.log"
    run:
        import shutil
        from pathlib import Path
        src = Path(str(input.model_network))
        dst = Path(str(output.scenario_network))
        log.info(f"Mapping model network {src} → scenario network {dst}")
        if not src.exists():
            log.info(f"Source model network does not exist: {src} — leaving for upstream rules to build")
            # Snakemake will still consider this rule's input unresolved and will
            # schedule the required upstream rules to create it. We simply return
            # here to avoid raising errors when the source is not yet present.
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log.info(f"Copied {src} to {dst}")


# Create model-based ETYS_base.nc from the first ETYS scenario's network
# This provides backward compatibility for rules that hardcode the model
# filename. Only create if we have at least one ETYS scenario configured.
if _etys_scenarios:
    _first_etys = _etys_scenarios[0]

    rule scenario_to_ETYS_model:
        input:
            f"{resources_path}/network/{_first_etys}_network.nc"
        output:
            f"{resources_path}/network/ETYS_base.nc"
        log:
            f"logs/network_build/copy_{_first_etys}_to_ETYS_base.log"
        run:
                import shutil
                from pathlib import Path
                src = Path(str(input[0]))
                dst = Path(str(output[0]))
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not src.exists():
                    # Write a simple message to the Snakemake log file and exit gracefully
                    with open(str(log), 'a') as _lf:
                        _lf.write(f"Source scenario network not yet present: {src}\n")
                    return
                shutil.copy2(src, dst)
                with open(str(log), 'a') as _lf:
                    _lf.write(f"Copied {src} to {dst}\n")

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_network_model(scenario):
    """
    Get network model type for a given scenario.
    
    Args:
        scenario (str): Scenario ID (e.g., 'HT35', 'Historical_2020')
    
    Returns:
        str: Network model name ('ETYS', 'Reduced', or 'Zonal')
    """
    # Uses the pre-loaded `_sc_defs` from main Snakefile (via `scenarios`).
    # This avoids repeated file I/O while Snakemake builds the DAG.
    try:
        return _sc_defs[scenario]["network_model"]
    except KeyError:
        raise ValueError(f"Scenario '{scenario}' not found or missing 'network_model'. "
                        f"Available scenarios: {list(_sc_defs.keys())}")


def get_network_inputs(scenario):
    """
    Return appropriate input data files based on scenario's network model.
    
    Different network models require different source data:
    - ETYS: Excel files from NESO ETYS publication
    - Reduced: CSV files with simplified topology
    - Zonal: CSV files with zone definitions
    
    Args:
        scenario (str): Scenario ID
    
    Returns:
        list: Paths to required input files for network construction
    """
    network_model = get_network_model(scenario)
    base_inputs = []
    
    if network_model == "ETYS":
        base_inputs.extend([
            f"{data_path}/network/ETYS/ETYS Appendix B 2023.xlsx",
            f"{data_path}/network/ETYS/GB_network.xlsx",
            f"{data_path}/network/ETYS/Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb"
        ])
    elif network_model == "Reduced":
        base_inputs.extend([
            f"{data_path}/network/reduced_network/buses.csv",
            f"{data_path}/network/reduced_network/lines.csv"
        ])
    elif network_model == "Zonal":
        base_inputs.extend([
            f"{data_path}/network/zonal/buses.csv",
            f"{data_path}/network/zonal/links.csv"
        ])
    
    return base_inputs


# ──────────────────────────────────────────────────────────────────────────────
# NETWORK CONSTRUCTION RULES
# ──────────────────────────────────────────────────────────────────────────────

rule build_ETYS_base_network:
    """
    Build the ETYS (Electricity Ten Year Statement) transmission network.
    
    This rule constructs the detailed ETYS network representation with 2000+ buses
    representing the GB transmission system. The network includes:
      • Transmission buses at 400kV, 275kV, and 132kV voltage levels
      • Transmission lines with thermal capacity limits and reactances
      • Transformers connecting different voltage levels
      • Geographic coordinates for all buses (WGS84)
    
    DATA SOURCES:
      • ETYS Appendix B 2023: Network topology, line parameters, bus voltage levels
      • ETYS Appendix B 2023: Network upgrade data (2024-2031)
      • GB_network.xlsx: Geographic coordinates for all network buses
      • ETYS 2023 Appendix E: Regional FES data breakdown (optional)
    
    NETWORK UPGRADES:
      When etys_upgrades.enabled=true in scenario config, applies:
      • Circuit additions/removals/modifications from ETYS Appendix B 2023
      • Transformer additions/removals from ETYS Appendix B 2023
      • (Future) New HVDC interconnectors
      Upgrades are filtered to include only those ≤ modelled_year.
    
    OUTPUTS:
      • PyPSA Network object saved as NetCDF (.nc file)
      • Contains: buses, lines, transformers, links DataFrames
      • Ready for demand and generator integration
    
    PERFORMANCE:
      • Typical runtime: 10-30 seconds (15-45s with upgrades)
      • Memory: ~500MB for full ETYS network
    
    SEE ALSO:
      • scripts/ETYS_network.py - Detailed network construction logic
      • scripts/ETYS_upgrades.py - Network upgrade application logic
      • docs/NETWORK_MODELS.md - ETYS network documentation
    """
    input:
        lambda wildcards: get_network_model(wildcards.scenario) == "ETYS" and [
            f"{data_path}/network/ETYS/ETYS Appendix B 2023.xlsx",  # Base network topology + upgrades
            f"{data_path}/network/ETYS/GB_network.xlsx",
            f"{data_path}/network/ETYS/Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb"
        ] or []
    output:
        network=f"{resources_path}/network/{{scenario}}_network.nc"
    params:
        scenario=lambda wildcards: wildcards.scenario,
        modelled_year=lambda wildcards: scenarios[wildcards.scenario].get("modelled_year", 2020),
        etys_upgrades_enabled=lambda wildcards: scenarios[wildcards.scenario].get("etys_upgrades", {}).get("enabled", False),
        etys_upgrade_year=lambda wildcards: scenarios[wildcards.scenario].get("etys_upgrades", {}).get("upgrade_year", None)
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/network_build/build_{scenario}_base_network.log"
    wildcard_constraints:
        scenario=ETYS_SCENARIO_REGEX
    benchmark:
        "benchmarks/network_build/build_{scenario}_ETYS.txt"
    message:
        "Building ETYS network for scenario: {wildcards.scenario}"
    script:
        "../scripts/network_build/ETYS_network.py"


rule build_reduced_base_network:
    """
    Build the Reduced network (simplified regional representation).
    
    This rule constructs a simplified network with 30-50 buses representing
    major GB regions and key transmission corridors. The Reduced network:
      • Aggregates detailed ETYS topology into regional nodes
      • Preserves critical transmission bottlenecks
      • Reduces computational complexity for faster analysis
      • Maintains geographic fidelity for demand/generation allocation
    
    DATA SOURCES:
      • reduced_network/buses.csv: Regional bus definitions with coordinates
      • reduced_network/lines.csv: Aggregated transmission line parameters
    
    USE CASES:
      • Fast scenario screening and sensitivity analysis
      • Long-term planning with reduced detail requirements
      • Multi-year simulations where speed is critical
    
    PERFORMANCE:
      • Typical runtime: 5-10 seconds
      • Memory: ~100MB
      • ~10x faster simulation vs ETYS
    
    SEE ALSO:
      • scripts/build_network.py - Flexible network builder
      • docs/NETWORK_MODELS.md - Network model comparison
    """
    input:
        lambda wildcards: get_network_inputs(wildcards.scenario) if get_network_model(wildcards.scenario) == "Reduced" else []
    output:
        network=f"{resources_path}/network/{{scenario}}_network.nc"
    params:
        network_model=lambda wildcards: [get_network_model(wildcards.scenario)],
        scenario=lambda wildcards: wildcards.scenario
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/network_build/build_{scenario}_base_network.log"
    wildcard_constraints:
        scenario=REDUCED_SCENARIO_REGEX
    benchmark:
        "benchmarks/network_build/build_{scenario}_Reduced.txt"
    message:
        "Building Reduced network for scenario: {wildcards.scenario}"
    script:
        "../scripts/network_build/build_network.py"


rule build_zonal_base_network:
    """
    Build the Zonal network (17-zone aggregated representation).
    
    This rule constructs a highly aggregated zonal network representing GB as
    17 distinct zones. The Zonal network:
      • Groups regions into coherent geographic/market zones
      • Uses links (not lines) for inter-zonal transmission
      • Ideal for market modeling and high-level policy analysis
      • Fastest computational performance
    
    ZONE DEFINITIONS:
      • Typically based on DNO/GSP regions or market zones
      • Each zone has aggregated demand and generation
      • Inter-zone transmission capacity limits preserved
    
    DATA SOURCES:
      • zonal/buses.csv: Zone definitions with representative coordinates
      • zonal/links.csv: Inter-zone transmission capacity
    
    USE CASES:
      • Market clearing and price analysis
      • Multi-decade scenario analysis
      • High-level policy screening
      • Rapid prototyping
    
    PERFORMANCE:
      • Typical runtime: <5 seconds
      • Memory: <50MB
      • ~100x faster simulation vs ETYS
    
    SEE ALSO:
      • scripts/build_network.py - Handles both Reduced and Zonal
      • docs/NETWORK_MODELS.md - When to use each network type
    """
    input:
        lambda wildcards: get_network_inputs(wildcards.scenario) if get_network_model(wildcards.scenario) == "Zonal" else []
    output:
        network=f"{resources_path}/network/{{scenario}}_network.nc"
    params:
        network_model=lambda wildcards: [get_network_model(wildcards.scenario)],
        scenario=lambda wildcards: wildcards.scenario
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/network_build/build_{scenario}_base_network.log"
    wildcard_constraints:
        scenario=ZONAL_SCENARIO_REGEX
    benchmark:
        "benchmarks/network_build/build_{scenario}_Zonal.txt"
    message:
        "Building Zonal network for scenario: {wildcards.scenario}"
    script:
        "../scripts/network_build/build_network.py"

