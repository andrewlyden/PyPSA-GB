"""
Hydrogen System Integration Rules for PyPSA-GB

This module integrates a simplified hydrogen sector coupling:
- Electrolysis (power-to-hydrogen)
- Hydrogen storage (underground caverns / tanks)
- H2 power generation (H2-ready turbines)

Architecture:
=============
The current implementation uses a "copper-plate" hydrogen network:
- Single GB-wide hydrogen bus (no spatial resolution)
- All electrolysers feed into common H2 storage
- All H2 turbines draw from common H2 storage

This is a simplification - future work will model regional H2 networks.

Data Sources:
=============
- FES: Electrolysis capacity projections by GSP
- FES: Hydrogen generation capacity (already integrated as generators)

Technical Parameters:
====================
- Electrolysis efficiency: 70% (electrical → H2 LHV)
- H2 turbine efficiency: 50% (H2 → electrical)
- Round-trip efficiency: 35% (electricity → H2 → electricity)
- Storage: Sized for seasonal flexibility (168 hours of generation)

Usage:
======
# Run hydrogen integration for a scenario
snakemake resources/network/{scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl --cores 1

# Run full workflow including hydrogen
snakemake --cores 4
"""

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Every CP30 gate here is about where the numbers come from (the CP30 ES1/ED1 files, and which
# FES edition serves as the spatial proxy), so it reads target_source().
from scripts.utilities.scenario_detection import is_historical_scenario, target_source

resources_path = "resources"
data_path = "data"


def get_hydrogen_spatial_proxy_config(scenario_config):
    return scenario_config.get("fleet", {}).get("spatial_proxy", {}) if target_source(scenario_config) == "CP30" else {}


def get_hydrogen_fes_year(scenario_config):
    if target_source(scenario_config) == "CP30":
        return get_hydrogen_spatial_proxy_config(scenario_config).get("FES_year", 2025)
    return scenario_config.get("FES_year", 2024)


def get_hydrogen_fes_scenario(scenario_config):
    if target_source(scenario_config) == "CP30":
        return get_hydrogen_spatial_proxy_config(scenario_config).get("FES_scenario")
    return scenario_config.get("FES_scenario", None)


def get_hydrogen_cp30_es1(scenario_config):
    return scenario_config.get("source_data", {}).get("files", {}).get("es1", []) if target_source(scenario_config) == "CP30" else []


def get_fes_hydrogen_input(scenario_config, key):
    """A file of an enabled ``hydrogen.fes`` block (task Y6b), else [].

    ``key`` None returns the module path, so the FES long-range jobs track its code.
    """
    from scripts.hydrogen.fes_hydrogen import fes_hydrogen_config
    block = fes_hydrogen_config(scenario_config.get("hydrogen", {}))
    if not block.get("enabled", False):
        return []
    if key is None:
        return "scripts/hydrogen/fes_hydrogen.py"
    return block.get(key) or []


def get_electrolyser_siting_inputs(scenario_config):
    """FES file (and siting code) of hydrogen.electrolyser_siting method fes_gsp, else [] (15 Oct 2026)."""
    siting = (scenario_config.get("hydrogen", {}) or {}).get("electrolyser_siting", {}) or {}
    if is_historical_scenario(scenario_config) or siting.get("method", "h2_turbine_pro_rata") != "fes_gsp":
        return []
    return [f"{resources_path}/FES/FES_{int(siting.get('fes_year', 2024))}_data.csv",
            "scripts/hydrogen/fes_hydrogen.py"]


def electrolyser_relief_enabled(scenario_config):
    """hydrogen.electrolyser_siting fes_gsp with the import relief on (max_loading > 0; 16 Oct 2026)."""
    siting = (scenario_config.get("hydrogen", {}) or {}).get("electrolyser_siting", {}) or {}
    return (not is_historical_scenario(scenario_config) and siting.get("method", "h2_turbine_pro_rata") == "fes_gsp"
            and float(siting.get("max_loading", 1.0)) > 0)


def get_electrolyser_relief_input(wildcards, key):
    """The electrolyser import relief's geography: the scenario's GSP polygons and region map (as the
    locational checks read them, _cp30_locational_build_inputs) and the checks' boundary file; [] when off."""
    if not electrolyser_relief_enabled(scenarios[wildcards.scenario]):
        return []
    if key == "boundaries":
        return _cp30_locational_config(wildcards).get("boundary_file",
                                                      "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson")
    return _cp30_locational_build_inputs(wildcards)[key]


def get_hydrogen_cp30_ed1(scenario_config):
    return scenario_config.get("source_data", {}).get("files", {}).get("ed1", []) if target_source(scenario_config) == "CP30" else []


# =============================================================================
# HYDROGEN SYSTEM INTEGRATION
# =============================================================================

rule add_hydrogen_system:
    """
    Add hydrogen system components to PyPSA network.
    
    This rule transforms the network by:
    1. Creating a single GB-wide hydrogen bus
    2. Adding electrolysis Links (electricity → H2)
    3. Adding hydrogen storage Store
    4. Converting H2 generators to Links (H2 → electricity)
    
    The result is a proper power-to-gas-to-power (P2G2P) system where:
    - Electrolysis consumes electricity to produce hydrogen
    - Hydrogen is stored (weekly/seasonal buffer)
    - H2 turbines consume hydrogen to produce electricity
    
    This creates the correct energy balance constraint - H2 turbines can
    only generate if there's sufficient hydrogen from electrolysis.
    
    For historical scenarios (≤2024), this rule is a no-op as there was
    minimal hydrogen infrastructure.
    
    Parameters:
    -----------
    electrolysis_efficiency: 0.70 (70% electrical → H2)
    h2_turbine_efficiency: 0.50 (50% H2 → electrical)
    storage_hours: 168 (1 week of H2 generation capacity)
    
    Performance:
    ------------
    ~5-10 seconds (network modification, no heavy computation)
    
    Transforms:
    -----------
    {scenario}_network_demand_renewables_thermal_generators_storage.pkl
    → {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage.pkl",
        # Historical scenarios copy the network through without reading FES
        # (as in storage.smk), so they need no FES download.
        fes_data=lambda wc: [] if is_historical_scenario(scenarios[wc.scenario]) else f"{resources_path}/FES/FES_{get_hydrogen_fes_year(scenarios[wc.scenario])}_data.csv",
        cp30_es1=lambda wc: get_hydrogen_cp30_es1(scenarios[wc.scenario]),
        cp30_ed1=lambda wc: get_hydrogen_cp30_ed1(scenarios[wc.scenario]),
        # Electrolysers by FES GSP rows (hydrogen.electrolyser_siting.method fes_gsp; [] otherwise).
        electrolyser_fes_data=lambda wc: get_electrolyser_siting_inputs(scenarios[wc.scenario]),
        # The import relief (16 Oct 2026): regions and boundary sides as the locational checks read them.
        electrolyser_gsp_polygons=lambda wc: get_electrolyser_relief_input(wc, "gsp_polygons"),
        electrolyser_gsp_group_regions=lambda wc: get_electrolyser_relief_input(wc, "gsp_group_regions"),
        electrolyser_boundaries=lambda wc: get_electrolyser_relief_input(wc, "boundaries"),
        # Task Y6b: the FES long-range hydrogen system reads the Y1 targets and FES prices
        # ([] for every scenario without hydrogen.fes.enabled).
        fes_targets=lambda wc: get_fes_hydrogen_input(scenarios[wc.scenario], 'targets_file'),
        fes_future_costs=lambda wc: get_fes_hydrogen_input(scenarios[wc.scenario], 'future_costs_file'),
        fes_hydrogen_logic=lambda wc: get_fes_hydrogen_input(scenarios[wc.scenario], None)
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl"
    params:
        scenario=lambda wc: wc.scenario,
        modelled_year=lambda wc: scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year'),
        is_historical=lambda wc: (scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year', 9999)) <= 2024,
        fes_scenario=lambda wc: get_hydrogen_fes_scenario(scenarios[wc.scenario]),
        scenario_source=lambda wc: scenarios[wc.scenario].get('scenario_source', 'FES'),
        cp30_pathway=lambda wc: scenarios[wc.scenario].get('CP30_pathway', None),
        electrolysis_full_load_hours=lambda wc: scenarios[wc.scenario].get('hydrogen', {}).get('electrolysis_full_load_hours', 4000.0),
        hydrogen_config=lambda wc: scenarios[wc.scenario].get('hydrogen', {}),
        # The import relief's BM ratings (voltage floor, rating overrides, seasonal ratings) and the
        # locational checks' boundary remote points; {} when the relief is off.
        transmission_config=lambda wc: (scenarios[wc.scenario].get('network', {}).get('transmission', {}) or {})
            if electrolyser_relief_enabled(scenarios[wc.scenario]) else {},
        locational_config=lambda wc: (scenarios[wc.scenario].get('validation', {}).get('locational', {}) or {})
            if electrolyser_relief_enabled(scenarios[wc.scenario]) else {}
    message:
        "Adding hydrogen system to network for {wildcards.scenario}"
    log:
        "logs/storage/add_hydrogen_system_{scenario}.log"
    benchmark:
        "benchmarks/storage/add_hydrogen_system_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/hydrogen/add_hydrogen_system.py"
