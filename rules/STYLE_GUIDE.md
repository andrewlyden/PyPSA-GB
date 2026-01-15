# ══════════════════════════════════════════════════════════════════════════════
# PyPSA-GB Snakemake Rules Style Guide
# ══════════════════════════════════════════════════════════════════════════════
#
# This document defines the standard structure for all .smk rule files.
# Follow this template when creating new rules or refactoring existing ones.
#
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: MODULE DOCSTRING (Required)
# ──────────────────────────────────────────────────────────────────────────────
#
# Every .smk file must start with a docstring explaining:
#   - What this module does (1-2 sentences)
#   - Pipeline stages (numbered list)
#   - Key inputs/outputs (file types, not paths)
#
# Template:
"""
[Module Name] Rules for PyPSA-GB

[One sentence describing the purpose of this module.]

Pipeline Stages:
  1. [First stage name] - [brief description]
  2. [Second stage name] - [brief description]
  ...

Inputs:
  - [Input type 1]: [description]
  - [Input type 2]: [description]

Outputs:
  - [Output type 1]: [description]

See Also:
  - [related_module.smk] - [relationship]
"""

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: IMPORTS (if needed)
# ──────────────────────────────────────────────────────────────────────────────
#
# DO NOT import yaml or reload configs - use variables from main Snakefile:
#   - scenarios (dict)
#   - run_ids (list)
#   - config (dict)
#   - resources_path, data_path (strings)
#
# Only import if you need specific functionality not in Snakefile:
#
# Example (avoid unless necessary):
# import re  # Only if needed for this module

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: HELPER FUNCTIONS (if needed)
# ──────────────────────────────────────────────────────────────────────────────
#
# Naming convention: _function_name (underscore prefix for module-local)
# Keep functions short and focused
#
# Example:
# def _get_network_model(wildcards):
#     """Get network model for scenario."""
#     return scenarios[wildcards.scenario]["network_model"]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: RULES (Main content)
# ──────────────────────────────────────────────────────────────────────────────
#
# Rule Structure (in order):
#   1. rule name (snake_case, descriptive)
#   2. docstring (required - see template below)
#   3. input (required)
#   4. output (required)
#   5. params (if needed)
#   6. wildcard_constraints (if using wildcards)
#   7. log (required)
#   8. benchmark (optional but encouraged for slow rules)
#   9. conda (always: "../envs/pypsa-gb.yaml")
#   10. script OR shell OR run (required)
#
# Example Rule:

rule example_rule:
    """
    [One sentence describing what this rule does.]
    
    [Optional: 2-3 sentences with more detail about the process.]
    
    Transforms: input_file.ext → output_file.ext
    
    Inputs:
      - input_name: [description] (file type)
    
    Outputs:
      - output_name: [description] (file type)
    
    Performance: ~Xs for [network model]
    """
    input:
        data_file=f"{resources_path}/input/{{scenario}}_input.csv",
        network=f"{resources_path}/network/{{scenario}}_network.nc"
    output:
        result=f"{resources_path}/output/{{scenario}}_output.csv"
    params:
        scenario=lambda wc: wc.scenario,
        network_model=lambda wc: scenarios[wc.scenario]["network_model"]
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    log:
        "logs/module_name/example_rule_{scenario}.log"
    benchmark:
        "benchmarks/module_name/example_rule_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/example_script.py"


# ══════════════════════════════════════════════════════════════════════════════
# NAMING CONVENTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# Rule names:
#   - verb_noun format: build_network, add_storage, extract_data
#   - Avoid: process (too generic), handle, do
#   - Good: integrate_generators, map_to_buses, cluster_network
#
# File paths:
#   - Use f-strings with {resources_path}/{data_path}
#   - Wildcards: {{scenario}}, {{year}}, {{network_model}}
#   - Network naming: {scenario}_network_{component1}_{component2}.nc
#
# Log paths:
#   - logs/{module_name}/{rule_name}_{wildcard}.log
#   - Example: logs/storage/add_storage_{scenario}.log
#
# Benchmark paths:
#   - benchmarks/{module_name}/{rule_name}_{wildcard}.txt
#   - Example: benchmarks/generators/integrate_{scenario}.txt

# ══════════════════════════════════════════════════════════════════════════════
# DOCSTRING LENGTH GUIDELINES
# ══════════════════════════════════════════════════════════════════════════════
#
# Short (5-10 lines): Simple data extraction or format conversion
# Medium (15-30 lines): Standard processing with parameters
# Long (30-50 lines): Complex integration with multiple options
#
# AVOID: Docstrings over 50 lines - split into multiple rules instead
#
# What to include:
#   ✓ What the rule does (always)
#   ✓ Input/output descriptions (always)
#   ✓ Performance estimate (for rules >10s)
#   ✓ Network model differences (if behavior varies)
#
# What to exclude:
#   ✗ Implementation details (belongs in script)
#   ✗ Full parameter documentation (link to script instead)
#   ✗ Historical context (put in docs/)

# ══════════════════════════════════════════════════════════════════════════════
# ANTI-PATTERNS (Do NOT do these)
# ══════════════════════════════════════════════════════════════════════════════
#
# ❌ Reloading config files:
#     with open("config/config.yaml") as f:  # BAD - use 'config' from Snakefile
#         cfg = yaml.safe_load(f)
#
# ❌ Redefining paths:
#     resources_path = "resources"  # BAD - already defined in Snakefile
#
# ❌ Duplicating helper functions:
#     def extract_from_scenarios(key):  # BAD - already in Snakefile
#
# ❌ Inline Python in 'run:' blocks for complex logic:
#     run:
#         import pandas as pd  # BAD - use script instead
#         df = pd.read_csv(...)
#
# ❌ Hardcoded paths:
#     input: "resources/network/ETYS_base.nc"  # BAD - use f-string with variable

# ══════════════════════════════════════════════════════════════════════════════
# STANDARD INPUTS/OUTPUTS BY STAGE
# ══════════════════════════════════════════════════════════════════════════════
#
# Network file naming convention: Progressive naming with compound suffixes.
# Each rule adds a component name to the suffix. File format changes from .pkl to .nc
# at the final assembly stage.
#
# Pipeline progression:
#
#   {scenario}_network.nc                                        (base topology)
#     ↓ build_network / extract_demand
#   {scenario}_network_demand.pkl                                (add demand loads)
#     ↓ integrate_renewable_generators
#   {scenario}_network_demand_renewables.pkl                     (add renewable generators)
#     ↓ integrate_thermal_generators
#   {scenario}_network_demand_renewables_thermal_generators.pkl  (add thermal/dispatchable)
#     ↓ add_storage
#   {scenario}_network_demand_renewables_thermal_generators_storage.pkl  (add battery/hydro/LAES)
#     ↓ add_hydrogen_system
#   {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl  (add H2 electrolysis/turbines)
#     ↓ add_interconnectors
#   {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc  (add cross-border)
#     ↓ [optional: cluster_network, apply_etys_upgrades]
#   {scenario}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc  (e.g., _clustered)
#     ↓ solve_network
#   {scenario}_solved.nc                                         (optimization results + stats)
#
# Format notes:
#   - Use .pkl for intermediate stages (fast I/O during development)
#   - Switch to .nc at final stage (portable, archival format)
#   - Non-network data (profiles, CSVs) use their native format
#
# Naming rules:
#   - Component order (immutable): demand → renewables → thermal_generators → storage → hydrogen → interconnectors
#   - Use underscore to separate components: {base}_{component1}_{component2}...
#   - Optional transforms (e.g., _clustered, _with_upgrades) appear BEFORE final _solved suffix
#   - Rule names should use verb_noun format matching component names
#
# Common pitfalls:
#   ❌ Wrong order (e.g., storage_thermal_generators)
#   ❌ Missing intermediate stages (must be cumulative)
#   ❌ Inconsistent naming across rules (follow template exactly)
