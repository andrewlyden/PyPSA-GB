"""
Generator Integration Rules for PyPSA-GB

Pipeline Map (concise):
- Inputs: DUKES (historical thermal), REPD sites (renewables + dispatchable thermal), FES (future projections)
- Scenario Detection: Automatically routes to DUKES for historical years, FES for future years
- Integrate: Add generators to network (components + timeseries aligned to snapshots)
- Export: Write generator CSVs and full network CSV tables

Authoritative outputs:
- resources/network/{scenario}_network_demand_renewables_thermal_generators.pkl
- resources/generators/{scenario}_generators_full.csv (+ summaries)
- resources/network/csv/{scenario}_base_demand_generators/ (exported tables)

Notes:
- Time series can be any timestep; the integration scripts in scripts/generators/ align to network.snapshots.
- Rules tagged LEGACY are supported but not part of the canonical path.
- TEC processing rules are preserved as LEGACY for reference but not used in main workflow.

Data update points:
1) resources/renewable/*_sites.csv
2) resources/renewable/profiles/*.csv
3) data/generators/generator_data_by_fuel.csv
"""

# Import scenario detection utility. Gates on where the targets come from use target_source();
# gates on how the fleet is built (and the brownfield method's fleet.spatial_proxy blocks) use uses_brownfield().
from scripts.utilities.scenario_detection import is_historical_scenario, target_source, uses_brownfield

# === Extract lists of key inputs ===
def _extract_from_scenarios(key, default=None):
    """Extract values from active scenarios, skipping those without the key."""
    values = set()
    for rid in build_ids:
        if rid in scenarios and key in scenarios[rid]:
            values.add(scenarios[rid][key])
    return sorted(values) if values else ([default] if default is not None else [])

# Define variables for repeated paths (only extract keys that exist in all scenarios)
fes_years = _extract_from_scenarios("FES_year")  # Empty for historical-only runs
for _rid in build_ids:
    _scenario = scenarios.get(_rid, {})
    _proxy_year = _scenario.get("fleet", {}).get("spatial_proxy", {}).get("FES_year") if target_source(_scenario) == "CP30" else None
    if _proxy_year is not None:
        fes_years = sorted(set(fes_years + [_proxy_year]))
renewables_years = _extract_from_scenarios("renewables_year", 2020)
resources_path = "resources"
data_path = "data"


def get_spatial_proxy_config(scenario_config):
    return scenario_config.get("fleet", {}).get("spatial_proxy", {}) if target_source(scenario_config) == "CP30" else {}


def get_generation_fes_year(scenario_config):
    if target_source(scenario_config) == "CP30":
        return get_spatial_proxy_config(scenario_config).get("FES_year", 2025)
    return scenario_config.get("FES_year", 2025)


def get_generator_cp30_resource(scenario_config, resource_name):
    return scenario_config.get("source_data", {}).get("files", {}).get(resource_name, []) if target_source(scenario_config) == "CP30" else []

# ══════════════════════════════════════════════════════════════════════════════
# RULE ORDER: Prefer new REPD-only rule over legacy TEC-based rule
# ══════════════════════════════════════════════════════════════════════════════


def get_cost_fes_year(scenario_config):
    if scenario_config.get("modelled_year", 2020) <= 2024:
        return None
    if target_source(scenario_config) == "CP30":
        return get_spatial_proxy_config(scenario_config).get("FES_year")
    return scenario_config.get("FES_year")


def get_cost_price_file(scenario_id, price_type):
    fes_year = get_cost_fes_year(scenarios[scenario_id])
    if not fes_year:
        return []
    return f"{resources_path}/marginal_costs/{price_type}_{fes_year}.csv"


def get_cost_price_param(scenario_id, price_type):
    value = get_cost_price_file(scenario_id, price_type)
    return value or None


# ══════════════════════════════════════════════════════════════════════════════
# FES LONG-RANGE SUPPLY (any FES year)
# ══════════════════════════════════════════════════════════════════════════════
# Only the FES long-range scenarios (HT50_* today) set ``fleet.fes_supply.enabled``. For
# every other scenario these helpers return [] so their DAG and inputs are exactly as
# before. The storage, hydrogen and interconnector rules (included later) use them too.

def fes_supply_block(scenario_id, part=None):
    """The scenario's enabled ``fleet.fes_supply`` block (or one sub-block), else {}."""
    from scripts.generators.fes_supply import fes_supply_config, fes_supply_part
    scenario_config = scenarios.get(scenario_id, {})
    return fes_supply_part(scenario_config, part) if part else fes_supply_config(scenario_config)


def fes_supply_input(scenario_id, part, key):
    """A file named in an enabled ``fleet.fes_supply.<part>`` block, else []."""
    block = fes_supply_block(scenario_id, part)
    return (block.get(key) or []) if block else []


def fes_supply_targets_input(scenario_id, part):
    """The Y1 FES targets JSON when ``fleet.fes_supply.<part>`` is enabled, else []."""
    if not fes_supply_block(scenario_id, part):
        return []
    return fes_supply_block(scenario_id).get("targets_file") or []


def fes_gas_ccs_split_input(scenario_id):
    """The year's classified ES1 lines when ``fleet.fes_supply.thermal.gas_ccs_split`` is on, else []."""
    thermal = fes_supply_block(scenario_id, "thermal")
    split = (thermal or {}).get("gas_ccs_split") or {}
    if not split or not split.get("enabled", True):
        return []
    path = thermal.get("es1_lines_file")
    if not path:
        raise ValueError(f"{scenario_id}: fleet.fes_supply.thermal.gas_ccs_split needs fleet.fes_supply.thermal.es1_lines_file")
    return path


def fes_supply_logic_input(scenario_id, path="scripts/generators/fes_supply.py"):
    """Track a FES long-range module as an input of those scenarios' jobs only."""
    return path if fes_supply_block(scenario_id) else []
ruleorder: extract_repd_dispatchable_sites > prepare_dispatchable_generator_sites

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR HYBRID DATA ROUTING (DUKES/FES)
# ══════════════════════════════════════════════════════════════════════════════

def get_generator_data_sources(wildcards):
    """
    Determine generator data sources based on scenario type.
    
    Historical scenarios (modelled_year ≤ current_year - 1):
      - Primary: DUKES_{year}_generators.csv (thermal capacity data)
      - Secondary: REPD (renewable sites - handled separately)
      - NO FES DATA (using only authoritative historical sources)
    
    Future scenarios (modelled_year > current_year - 1):
      - Primary: FES_data (all projections)
      - REPD not used (FES provides renewable projections)
    
    Returns:
        Dictionary with conditional inputs for thermal integration
    """
    from scripts.utilities.scenario_detection import is_historical_scenario, target_source
    
    scenario_id = wildcards.scenario
    scenario_config = scenarios[scenario_id]
    modelled_year = scenario_config['modelled_year']
    
    inputs = {}
    
    if is_historical_scenario(scenario_config):
        # Historical: Use DUKES ONLY (no FES fallback)
        inputs['dukes_data'] = f"{resources_path}/generators/DUKES/DUKES_{modelled_year}_generators.csv"
    elif target_source(scenario_config) == "CP30":
        # CP30: CP30 ES1 capacities with explicit FES spatial proxy
        fes_year = get_generation_fes_year(scenario_config)
        inputs['fes_data'] = f"{resources_path}/FES/FES_{fes_year}_data.csv"
        inputs['cp30_es1'] = get_generator_cp30_resource(scenario_config, 'es1')
    else:
        # Future: Use FES as primary source (DUKES not applicable)
        fes_year = scenario_config['FES_year']  # Required for future scenarios
        inputs['fes_data'] = f"{resources_path}/FES/FES_{fes_year}_data.csv"
    
    return inputs


def get_renewable_data_sources(wildcards):
    """
    Determine renewable generator data sources based on scenario type.
    
    Historical scenarios (modelled_year ≤ 2024):
      - Use REPD site data (individual renewable generators with locations)
      - Atlite profiles applied per-site
    
    Future scenarios (modelled_year > 2024):
      - Use FES capacity projections (aggregate capacity per GSP/technology)
      - Atlite profiles applied per-bus using nearest available profiles
      - REPD sites NOT used (FES provides capacity projections)
    
    Returns:
        Dictionary with conditional inputs for renewable integration
    """
    from scripts.utilities.scenario_detection import is_historical_scenario, target_source
    
    scenario_id = wildcards.scenario
    scenario_config = scenarios[scenario_id]
    
    inputs = {}
    
    if is_historical_scenario(scenario_config):
        # Historical: Use REPD site data for individual renewable generators
        inputs['use_fes'] = False
        inputs['wind_onshore_sites'] = f"{resources_path}/renewable/wind_onshore_sites.csv"
        inputs['wind_offshore_sites'] = f"{resources_path}/renewable/wind_offshore_sites.csv"
        inputs['solar_pv_sites'] = f"{resources_path}/renewable/solar_pv_sites.csv"
        inputs['small_hydro_sites'] = f"{resources_path}/renewable/small_hydro_sites.csv"
        inputs['large_hydro_sites'] = f"{resources_path}/renewable/large_hydro_sites.csv"
        inputs['tidal_stream_sites'] = f"{resources_path}/renewable/tidal_stream_sites.csv"
        inputs['shoreline_wave_sites'] = f"{resources_path}/renewable/shoreline_wave_sites.csv"
        inputs['tidal_lagoon_sites'] = f"{resources_path}/renewable/tidal_lagoon_sites.csv"
    elif target_source(scenario_config) == "CP30":
        # CP30: Use CP30 capacity totals and FES only as explicit spatial proxy
        fes_year = get_generation_fes_year(scenario_config)
        inputs['use_fes'] = True
        inputs['fes_data'] = f"{resources_path}/FES/FES_{fes_year}_data.csv"
        inputs['cp30_es1'] = get_generator_cp30_resource(scenario_config, 'es1')
        # Still need profiles for capacity factors
    else:
        # Future: Use FES capacity projections
        fes_year = scenario_config['FES_year']
        inputs['use_fes'] = True
        inputs['fes_data'] = f"{resources_path}/FES/FES_{fes_year}_data.csv"
        # Still need profiles for capacity factors
    
    return inputs


def get_renewable_profiles(wildcards):
    """
    Get renewable profile files for the scenario's renewables_year.
    
    These atlite-generated profiles are used for both historical (per-site)
    and future (per-bus) scenarios.
    """
    scenario_config = scenarios.get(wildcards.scenario, {})
    renewables_year = scenario_config.get('renewables_year', 2020)
    
    profiles = [
        f"{resources_path}/renewable/profiles/wind_onshore_{renewables_year}.csv",
        f"{resources_path}/renewable/profiles/wind_offshore_{renewables_year}.csv",
        f"{resources_path}/renewable/profiles/solar_pv_{renewables_year}.csv",
        f"{resources_path}/renewable/profiles/tidal_stream_{renewables_year}.csv",
        f"{resources_path}/renewable/profiles/shoreline_wave_{renewables_year}.csv",
        f"{resources_path}/renewable/profiles/tidal_lagoon_{renewables_year}.csv",
        f"{resources_path}/renewable/profiles/large_hydro_{renewables_year}.csv",
        f"{resources_path}/renewable/profiles/small_hydro_{renewables_year}.csv",
    ]
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# MODULAR GENERATOR INTEGRATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
# The generator integration is split into 3 focused stages for clarity,
# testability, and easier debugging:
#
#   Stage 1: integrate_renewable_generators
#            → Add weather-variable renewables (wind, solar, marine, hydro)
#
#   Stage 2: integrate_thermal_generators
#            → Add dispatchable thermal (CCGT, nuclear, biomass, waste)
#
#   Stage 3: finalize_generator_integration
#            → Add load shedding backup + create comprehensive exports
#
# Benefits:
#   - Clear separation of concerns (renewable vs thermal vs backup)
#   - Independent testing: `snakemake {scenario}_base_demand_renewables.nc`
#   - Easier debugging (isolate issues to specific generator type)
#   - Better performance tracking (benchmark each stage)
# ══════════════════════════════════════════════════════════════════════════════


# Stage 1: Add weather-variable renewable generators
def _support_register_input(scenario_id, key, brownfield_only=False):
    """A scheme-register file read by the support labelling, or [] when unused.

    ``support.labels.support_registers`` (the brownfield defaults) makes the Ofgem RO
    register and the LCCC CfD portfolio the authority for named-fleet support
    labels in future builds (scripts/renewables/support_register_labels.py).
    ``brownfield_only``: only where the fleet is built brownfield (its anonymous proxies).
    """
    scenario = scenarios.get(scenario_id, {})
    registers = scenario.get("support", {}).get("labels", {}).get("support_registers", {}) or {}
    if not scenario.get("support", {}).get("labels", {}).get("enabled", False) or not registers.get("enabled", False):
        return []
    if is_historical_scenario(scenario) or (brownfield_only and not uses_brownfield(scenario)):
        return []
    if key == "logic":
        return "scripts/renewables/support_register_labels.py"
    return registers.get(key) or []


def _cp30_brownfield_enabled(scenario_id):
    """True for a brownfield scenario whose renewables are built from the named REPD fleet."""
    sc = scenarios.get(scenario_id, {})
    brownfield = {**(sc.get("fleet", {}).get("brownfield") or {}), **((sc.get("fleet", {}).get("spatial_proxy") or {}).get("brownfield") or {})}
    return uses_brownfield(sc) and bool(brownfield.get("enabled", False))


def _cp30_gsp_placement_inputs(scenario_id):
    """GSP polygons, GSP group -> CP30 region map and Dem_per_node behind CP30 regions and GSP nodes."""
    sc = scenarios.get(scenario_id, {})
    regional = (sc.get("fleet", {}).get("spatial_proxy") or {}).get("regional_allocation") or {}
    if not (uses_brownfield(sc) and regional.get("enabled", False)):
        return []
    files = [regional.get(key) for key in ("gsp_polygon_file", "gsp_group_region_file") if regional.get(key)]
    if _cp30_brownfield_enabled(scenario_id):
        files.append("data/network/ETYS/GB_network.xlsx")
    return files


def _proxy_profile_region_inputs(scenario_id):
    """GSP polygons and region map that locate every site profile under proxy_profile_members: all_sites."""
    renewables = scenarios.get(scenario_id, {}).get("renewables") or {}
    if renewables.get("proxy_profile_basis") != "regional" or renewables.get("proxy_profile_members") != "all_sites":
        return []
    return [
        renewables.get("proxy_profile_region_polygon_file") or "data/network/GSP/GSP_regions_27700_20250109.geojson",
        renewables.get("proxy_profile_region_group_file") or "data/network/ETYS/cp30_es5_region_by_gsp_group.csv",
    ]


rule add_renewables_to_network:
    """
    STAGE 1: Integrate weather-variable renewable generators into network.
    
    This rule adds renewable generators with weather-dependent time series profiles.
    
    HISTORICAL SCENARIOS (≤2024):
      - Uses REPD site data for individual renewable generators
      - Each site has capacity and location from REPD
      - Atlite profiles applied per-site
    
    FUTURE SCENARIOS (>2024):
      - Uses FES capacity projections (aggregate per GSP/technology)
      - Capacity distributed across buses based on GSP mapping
      - Atlite profiles applied per-bus using nearest available profiles
    
    Technologies:
      - Wind onshore/offshore
      - Solar PV
      - Marine renewables (tidal stream, wave, lagoon)
      - Small hydro (run-of-river, ≤10 MW)
      - Large hydro (reservoir-based, >10 MW)
    
    Input (conditional based on scenario type):
      - Network with base demand loads
      - Historical: REPD site CSVs (8 technology types)
      - Future: FES_{year}_data.csv
      - Renewable profiles (weather-based time series from atlite)
    
    Output:
      - Network with renewable generators: {scenario}_network_demand_renewables.pkl
      - Renewable summary CSV: {scenario}_renewables_summary.csv
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand.pkl",
        # Renewable profiles (always needed for capacity factors)
        profiles=get_renewable_profiles,
        # REPD site files (for historical scenarios - always include for DAG stability)
        wind_onshore_sites=f"{resources_path}/renewable/wind_onshore_sites.csv",
        wind_offshore_sites=f"{resources_path}/renewable/wind_offshore_sites.csv",
        solar_pv_sites=f"{resources_path}/renewable/solar_pv_sites.csv",
        small_hydro_sites=f"{resources_path}/renewable/small_hydro_sites.csv",
        large_hydro_sites=f"{resources_path}/renewable/large_hydro_sites.csv",
        tidal_stream_sites=f"{resources_path}/renewable/tidal_stream_sites.csv",
        shoreline_wave_sites=f"{resources_path}/renewable/shoreline_wave_sites.csv",
        tidal_lagoon_sites=f"{resources_path}/renewable/tidal_lagoon_sites.csv",
        commissioning_date_overrides=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("fleet", {}).get("commissioning", {}).get("date_overrides_file") or [],
        # FES data (future FES source, or explicit spatial proxy for CP30)
        fes_data=lambda wildcards: f"{resources_path}/FES/FES_{get_generation_fes_year(scenarios[wildcards.scenario])}_data.csv" if not is_historical_scenario(scenarios[wildcards.scenario]) else [],
        # The FES edition's own GSP info, the declared island positions, and Dem_per_node (GSP rows on
        # their demand node on the FES long-range chain), 27 Oct 2026.
        fes_gsp_info=lambda wildcards: f"data/network/ETYS/fes{get_generation_fes_year(scenarios[wildcards.scenario])}_regional_breakdown_gsp_info.csv" if not is_historical_scenario(scenarios[wildcards.scenario]) else [],
        gsp_declared_points=lambda wildcards: "data/network/GSP/gsp_declared_points.csv" if not is_historical_scenario(scenarios[wildcards.scenario]) else [],
        gsp_dem_per_node=lambda wildcards: "data/network/ETYS/GB_network.xlsx" if not is_historical_scenario(scenarios[wildcards.scenario]) else [],
        cp30_es1=lambda wildcards: get_generator_cp30_resource(scenarios[wildcards.scenario], 'es1'),
        cp30_es5=lambda wildcards: get_generator_cp30_resource(scenarios[wildcards.scenario], 'es5'),
        # Imported helpers can modify explicit CP30 connection points; make
        # them DAG inputs so their changes cannot leave stale renewable mapping.
        cp30_connection_allocation_logic="scripts/CP30/connection_allocation.py",
        cp30_spatial_allocation_logic="scripts/CP30/spatial_allocation.py",
        hnd_hybrid_logic="scripts/CP30/hnd_hybrid.py",
        spatial_mapping_logic="scripts/utilities/spatial_utils.py",
        # Declared even when disabled so a future source-backed allocation change
        # invalidates the CP30 renewable integration reproducibly.
        cp30_connection_allocation=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("offshore_connection_allocation", {})
            .get("file", [])
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        cp30_regional_anchors=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("regional_allocation", {})
            .get("region_anchor_file", [])
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        renewable_support_overrides=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("support", {}).get("labels", {}).get("renewable_support_overrides_file") or [],
        historical_station_unit_splits=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("support", {}).get("labels", {}).get("historical_station_unit_splits_file") or [],
        renewable_performance_factor_registry=lambda wildcards: (
            scenarios.get(wildcards.scenario, {})
            .get("renewables", {}).get("performance_factor_registry", {})
            .get("file", [])
            if scenarios.get(wildcards.scenario, {})
            .get("renewables", {}).get("performance_factor_registry", {})
            .get("enabled", False) else []
        ),
        hnd_project_portfolio=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("hnd_project_allocation", {}).get("portfolio_file", [])
            if scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("hnd_project_allocation", {}).get("enabled", False) else []
        ),
        hnd_project_crosswalk=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("hnd_project_allocation", {}).get("crosswalk_file", [])
            if scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("hnd_project_allocation", {}).get("enabled", False) else []
        ),
        # Brownfield CP30 fleet: named REPD pipeline sites and the offshore landing
        # crosswalk. Declared for every scenario so the DAG is stable; only a CP30
        # scenario with fleet.spatial_proxy.brownfield.enabled reads them.
        wind_onshore_pipeline_sites=f"{resources_path}/renewable/wind_onshore_pipeline_sites.csv",
        wind_offshore_pipeline_sites=f"{resources_path}/renewable/wind_offshore_pipeline_sites.csv",
        solar_pv_pipeline_sites=f"{resources_path}/renewable/solar_pv_pipeline_sites.csv",
        cp30_brownfield_logic="scripts/CP30/brownfield_fleet.py",
        # Regional proxy profiles and the GSP-node placement are imported by the
        # integration script; declared so a change to either re-runs it (15 Oct 2026).
        proxy_profiles_logic="scripts/generators/proxy_profiles.py",
        gsp_nodes_logic="scripts/utilities/gsp_nodes.py",
        # CP30 brownfield placement data (15 Oct 2026): the declared proxy relocations,
        # and the GSP polygons, group-to-region map and Dem_per_node table behind the
        # CP30 regions and the GSP-node placement ([] for every other scenario).
        cp30_proxy_relocations=lambda wildcards: (
            "data/network/ETYS/cp30_proxy_relocations.csv"
            if _cp30_brownfield_enabled(wildcards.scenario) else []
        ),
        cp30_gsp_placement=lambda wildcards: _cp30_gsp_placement_inputs(wildcards.scenario),
        proxy_profile_regions=lambda wildcards: _proxy_profile_region_inputs(wildcards.scenario),
        # Large-hydro units with no site profile (the CP30 proxies) get a
        # CP30-calibrated inflow CF with the weather year's monthly shape from
        # ESPENI metered hydro (audit 26 Sep 2026).
        large_hydro_monthly_shape=f"{data_path}/renewables/hydro/espeni_npshyd_monthly_shape.csv",
        large_hydro_inflow_logic="scripts/renewables/large_hydro_inflow.py",
        renewable_performance_factor_logic="scripts/generators/renewable_performance_factors.py",
        # Declared named projects not deliverable in early years (Lewis farms
        # behind the Western Isles link); read by brownfield_fleet.py.
        cp30_named_fleet_exclusions=lambda wildcards: (
            {**(scenarios[wildcards.scenario].get("fleet", {}).get("brownfield") or {}),
             **(scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("brownfield") or {})}
            .get("named_exclusions_file", "data/renewables/cp30_named_fleet_exclusions.csv")
            if scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("brownfield", {}).get("enabled", False) else []
        ),
        cp30_offshore_landing_crosswalk=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("brownfield", {}).get("offshore_landing_crosswalk", [])
            if scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {})
            .get("brownfield", {}).get("enabled", False) else []
        ),
        # Scheme registers: named sites are labelled ROC/CfD from the Ofgem RO
        # register and the LCCC portfolio (26 Sep 2026).
        ro_register=lambda wildcards: _support_register_input(wildcards.scenario, "ro_register"),
        ro_certificates=lambda wildcards: _support_register_input(wildcards.scenario, "ro_certificates"),
        lccc_cfd_portfolio=lambda wildcards: _support_register_input(wildcards.scenario, "lccc_portfolio"),
        support_repd=lambda wildcards: _support_register_input(wildcards.scenario, "repd"),
        support_crosswalk_overrides=lambda wildcards: _support_register_input(wildcards.scenario, "crosswalk_overrides"),
        # Named sites' CfD contracts checked against the LCCC portfolio: part-
        # contracted, multi-round and terminated (F19, rebuild batch of 13 Oct 2026).
        checked_cfd_contracts=lambda wildcards: _support_register_input(wildcards.scenario, "checked_cfd_contracts"),
        support_register_logic=lambda wildcards: _support_register_input(wildcards.scenario, "logic"),
        # CfD allocation-round schedule the increment is labelled from, net of
        # the named fleet's rounds (G1, 27 Sep 2026).
        cfd_vintage_schedule=lambda wildcards: scenarios.get(wildcards.scenario, {}).get("support", {}).get(
            "labels", {}).get("cfd_vintage_schedule_file") or [],
        cfd_vintage_logic="scripts/generators/cfd_vintage_labels.py",
        # The profile rule's third offshore site file: its capacities are the
        # divisors that turn MW profile columns into capacity factors (G2).
        offshore_pipeline_processed=f"{resources_path}/renewable/offshore_pipeline_processed.csv",
        # Task Y6b: the Y5 offshore landing table of the year replaces the FES Direct
        # offshore rows in FES long-range builds ([] for every other scenario).
        fes_offshore_landing=lambda wildcards: fes_supply_input(wildcards.scenario, "offshore_landing", "file"),
        fes_offshore_landing_projects=lambda wildcards: fes_supply_input(
            wildcards.scenario, "offshore_landing", "projects_file"),
        fes_supply_logic=lambda wildcards: fes_supply_logic_input(wildcards.scenario),
        # Land boundary for the offshore-bus placement exclusion (G5).
        placement_land_boundary=lambda wildcards: (
            (scenarios.get(wildcards.scenario, {}).get("network", {}).get("spatial_mapping", {}) or {})
            .get("placement_exclusions", {}).get("land_boundary_file") or []
            if (scenarios.get(wildcards.scenario, {}).get("network", {}).get("spatial_mapping", {}) or {})
            .get("placement_exclusions", {}).get("enabled", False) else []
        ),
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables.pkl",
        summary=f"{resources_path}/generators/{{scenario}}_renewables_summary.csv",
        brownfield_audit=f"{resources_path}/generators/{{scenario}}_cp30_brownfield_audit.csv",
        brownfield_summary=f"{resources_path}/generators/{{scenario}}_cp30_brownfield_audit.json"
    params:
        scenario_config=lambda wildcards: model_scenario_param(scenarios.get(wildcards.scenario, {})),
        is_historical=lambda wildcards: is_historical_scenario(scenarios.get(wildcards.scenario, {}))
    log:
        "logs/integrate_renewable_generators_{scenario}.log"
    benchmark:
        "benchmarks/generators/integrate_renewables_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/generators/integrate_renewable_generators.py"


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHABLE THERMAL SITE EXTRACTION (REPD-based, no TEC dependency)
# ══════════════════════════════════════════════════════════════════════════════

rule extract_repd_dispatchable_sites:
    """
    Extract dispatchable renewable thermal sites from REPD.
    
    This rule extracts dispatchable renewable thermal generator sites directly from
    REPD without requiring TEC data. These are renewable generators that can be
    dispatched (unlike variable renewables like wind/solar).
    
    Technologies extracted:
    - Biomass (dedicated biomass plants)
    - Waste to Energy (EfW Incineration)
    - Biogas (Anaerobic Digestion)
    - Landfill Gas
    - Sewage Gas (Sewage Sludge Digestion)
    - Advanced Biofuels (Advanced Conversion Technologies)
    - Geothermal (constant baseload renewable)
    - Large Hydro (dispatchable reservoir hydro)
    
    Note: Conventional thermal (CCGT, OCGT, Nuclear, Coal) comes from DUKES (historical)
    or FES (future) data sources. Storage comes from REPD via storage.smk rules.
    
    Performance: ~5-10 seconds
    """
    input:
        repd=f"{data_path}/renewables/repd-q2-jul-2025.csv"
    output:
        biomass_sites=f"{resources_path}/generators/sites/biomass_sites.csv",
        waste_to_energy_sites=f"{resources_path}/generators/sites/waste_to_energy_sites.csv",
        biogas_sites=f"{resources_path}/generators/sites/biogas_sites.csv",
        landfill_gas_sites=f"{resources_path}/generators/sites/landfill_gas_sites.csv",
        sewage_gas_sites=f"{resources_path}/generators/sites/sewage_gas_sites.csv",
        advanced_biofuel_sites=f"{resources_path}/generators/sites/advanced_biofuel_sites.csv",
        geothermal_sites=f"{resources_path}/generators/sites/geothermal_sites.csv",
        large_hydro_sites=f"{resources_path}/generators/sites/large_hydro_sites.csv",
        summary_report=f"{resources_path}/generators/dispatchable_sites_summary.txt"
    log:
        "logs/extract_repd_dispatchable_sites.log"
    benchmark:
        "benchmarks/generators/extract_repd_dispatchable_sites.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/generators/extract_repd_dispatchable_sites.py"


# Historical support labels are built before marginal costs, so evidence-backed
# ROC stations influence the dispatch rather than being added only to a
# post-solve accounting copy.
def _historical_roc_station_labels(scenario_id):
    return scenarios.get(scenario_id, {}).get("support", {}).get("labels", {}).get(
        "historical_roc_station_labels", {}
    )


def _historical_roc_outturn_input(scenario_id):
    labels = _historical_roc_station_labels(scenario_id)
    if not labels.get("enabled", False):
        return []
    year = int(scenarios[scenario_id].get("modelled_year"))
    return f"{resources_path}/analysis/ofgem_roc_{year}_calendar_issued_stations.csv"


def _historical_roc_crosswalk_input(scenario_id):
    labels = _historical_roc_station_labels(scenario_id)
    return (
        f"{resources_path}/analysis/{scenario_id}_repd_roc_station_mapping.csv"
        if labels.get("enabled", False) else []
    )


def _historical_roc_registry_input(scenario_id):
    labels = _historical_roc_station_labels(scenario_id)
    return labels.get("registry_file", []) if labels.get("enabled", False) else []


def _historical_roc_overrides_input(scenario_id):
    labels = _historical_roc_station_labels(scenario_id)
    return labels.get("station_overrides_file", []) if labels.get("enabled", False) else []


def _historical_roc_source_evidence_input(scenario_id):
    labels = _historical_roc_station_labels(scenario_id)
    return (
        f"{resources_path}/analysis/{scenario_id}_roc_station_source_evidence.csv"
        if labels.get("enabled", False) else []
    )


def _historical_roc_physical_overlay_actions_input(scenario_id):
    return scenarios.get(scenario_id, {}).get("support", {}).get("labels", {}).get(
        "historical_roc_physical_overlay", {}
    ).get("actions_file") or []


def _historical_roc_representation_assumptions_input(scenario_id):
    return _historical_roc_station_labels(scenario_id).get(
        "residual_representation_assumptions_file"
    ) or []


rule build_historical_roc_station_source_evidence:
    """Search all Ofgem issuing stations in REPD and DUKES before a historical solve."""
    input:
        outturn=lambda wc: _historical_roc_outturn_input(wc.scenario),
        registry=lambda wc: _historical_roc_registry_input(wc.scenario),
        renewable_overrides=lambda wc: scenarios[wc.scenario].get("support", {}).get(
            "labels", {}
        ).get("renewable_support_overrides_file") or [],
        generator_overrides=lambda wc: scenarios[wc.scenario].get("support", {}).get(
            "labels", {}
        ).get("dispatchable_support_overrides_file") or [],
        repd_2021="data/renewables/renewable-energy-planning-database-q1-march-2021.csv",
        repd_2024="data/renewables/repd-q2-july-2024.csv",
        repd_2025="data/renewables/repd-q2-jul-2025.csv",
        dukes="data/generators/DUKES_5.11_2025.xlsx",
    output:
        mapping=f"{resources_path}/analysis/{{scenario}}_roc_station_source_evidence.csv",
        summary=f"{resources_path}/analysis/{{scenario}}_roc_station_source_evidence_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    shell:
        "python scripts/consumer_bill/build_roc_station_source_evidence.py "
        "--outturn {input.outturn} --registry {input.registry} "
        "--renewable-overrides {input.renewable_overrides} "
        "--generator-overrides {input.generator_overrides} "
        "--repd {input.repd_2021} {input.repd_2024} {input.repd_2025} "
        "--dukes {input.dukes} --mapping-output {output.mapping} "
        "--summary-output {output.summary}"


rule extract_historical_roc_calendar_outturn:
    """Create the full-year ROC station register needed by a historic build."""
    input:
        certificates_csv="data/market/ofgem_renewables_obligation/certificates_ro_current.csv",
    output:
        stations_csv=f"{resources_path}/analysis/ofgem_roc_{{year}}_calendar_issued_stations.csv",
        technology_csv=f"{resources_path}/analysis/ofgem_roc_{{year}}_calendar_issued_by_technology.csv",
        summary_json=f"{resources_path}/analysis/ofgem_roc_{{year}}_calendar_issued_summary.json",
    params:
        year=lambda wc: int(wc.year),
        period_start=lambda wc: None,
        period_end=lambda wc: None,
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        year="[0-9]{4}"
    script:
        "../scripts/consumer_bill/extract_ofgem_roc_calendar_outturn.py"


# Stage 2: Add dispatchable thermal generators
rule add_thermal_generators_to_network:
    """
    STAGE 2: Integrate dispatchable thermal generators into network.
    
    This rule adds thermal generators that can be dispatched on demand,
    using different data sources based on scenario type:
    
    HISTORICAL SCENARIOS (2010-2024):
      Data Sources (priority order):
        1. DUKES 5.11 (primary) - Conventional thermal capacity (coal, gas, nuclear)
        2. REPD (secondary) - Dispatchable renewable thermal (biomass, waste, biogas)
        NO FES FALLBACK - Using only authoritative historical sources
      
      Ensures historical accuracy using authoritative UK government data ONLY
    
    FUTURE SCENARIOS (2025+):
      Data Sources:
        1. FES (primary) - All projected capacity (thermal + renewable + storage)
        2. FES provides comprehensive projections for all technologies
    
    Technologies Added (from DUKES):
      - CCGT (Combined Cycle Gas Turbines - flexible mid-merit)
      - OCGT (Open Cycle Gas Turbines - peaking plants)
      - Nuclear (baseload, constant output)
      - Coal (legacy baseload, being phased out)
    
    Technologies Added (from REPD - dispatchable renewables):
      - Biomass (renewable baseload/mid-merit)
      - Waste-to-Energy (dispatchable renewable thermal)
      - Biogas (dispatchable renewable gas)
      - Landfill gas, sewage gas, advanced biofuels
      - Geothermal (constant baseload renewable)
    
    Process:
      1. Determine data source (DUKES for historical, FES for future)
      2. Load thermal capacity data (DUKES or FES)
      3. Load REPD dispatchable renewable thermal sites (historical only)
      4. Merge data sources with priority: DUKES > REPD (no FES for historical)
      5. Map thermal plants to network buses
      6. Load fuel characteristics (efficiency, costs, emissions)
      7. Add thermal generators to network with data_source tracking
      8. Create thermal capacity summary
    
    Input (conditional based on scenario type):
      - Network with renewable generators
      - Historical: DUKES_{year}_generators.csv + REPD sites (NO FES)
      - Future: FES_{year}_data.csv
      - Dispatchable renewable thermal site CSVs (7 types, historical only)
      - Generator fuel characteristics database
    
    Output:
      - Network with renewables + thermal: {scenario}_base_demand_renewables_thermal.nc
      - Thermal summary CSV: {scenario}_thermal_summary.csv (includes data_source column)
    
    Performance: ~30-45 seconds (depends on number of thermal plants)
    
    Next Stage: finalize_generator_integration (adds backup + exports)
    """
    input:
        unpack(get_generator_data_sources),  # Conditional: DUKES for historical, FES for future
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables.pkl",
        # Dispatchable renewable thermal sites (from REPD - historical scenarios only)
        biomass_sites=f"{resources_path}/generators/sites/biomass_sites.csv",
        waste_to_energy_sites=f"{resources_path}/generators/sites/waste_to_energy_sites.csv",
        biogas_sites=f"{resources_path}/generators/sites/biogas_sites.csv",
        landfill_gas_sites=f"{resources_path}/generators/sites/landfill_gas_sites.csv",
        sewage_gas_sites=f"{resources_path}/generators/sites/sewage_gas_sites.csv",
        advanced_biofuel_sites=f"{resources_path}/generators/sites/advanced_biofuel_sites.csv",
        geothermal_sites=f"{resources_path}/generators/sites/geothermal_sites.csv",
        historical_additional_thermal_sites="data/generators/historical_additional_thermal_sites.csv",
        large_hydro_source_sites=f"{resources_path}/renewable/large_hydro_sites.csv",
        # Historical years: large-hydro inflow = ESPENI metered hydro, day by day (large_hydro_inflow.py).
        historical_station_buses=lambda wildcards: "data/generators/historical_station_buses.csv" if is_historical_scenario(scenarios[wildcards.scenario]) else [],
        offshore_landing_crosswalk=lambda wildcards: "data/network/ETYS/repd_offshore_landing_crosswalk.csv" if is_historical_scenario(scenarios[wildcards.scenario]) else [],
        espeni_metered_hydro=lambda wildcards: f"{data_path}/demand/espeni.csv" if is_historical_scenario(scenarios[wildcards.scenario]) else [],
        dukes_roc_hydro_supplement=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("renewables", {}).get("hydro", {}).get("large_hydro", {}).get(
            "roc_hydro_supplement_file"
        ) or [],
        support_overrides=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("support", {}).get("labels", {}).get("dispatchable_support_overrides_file") or [],
        cfd_tranche_mapping=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("support", {}).get("labels", {}).get("historical_cfd_tranches", {}).get("mapping_file") or [],
        cfd_tranche_actuals=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("support", {}).get("labels", {}).get("historical_cfd_tranches", {}).get("actuals_file") or [],
        historical_roc_outturn=lambda wildcards: _historical_roc_outturn_input(wildcards.scenario),
        historical_roc_registry=lambda wildcards: _historical_roc_registry_input(wildcards.scenario),
        historical_roc_crosswalk=lambda wildcards: _historical_roc_crosswalk_input(wildcards.scenario),
        historical_roc_overrides=lambda wildcards: _historical_roc_overrides_input(wildcards.scenario),
        historical_roc_source_evidence=lambda wildcards: _historical_roc_source_evidence_input(wildcards.scenario),
        historical_roc_physical_overlay_actions=lambda wildcards: _historical_roc_physical_overlay_actions_input(wildcards.scenario),
        historical_roc_representation_assumptions=lambda wildcards: _historical_roc_representation_assumptions_input(wildcards.scenario),
        historical_roc_residual_output_calibration=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("support", {}).get("labels", {}).get("historical_roc_station_labels", {}).get(
            "residual_output_calibration_file"
        ) or [],
        generator_monthly_availability=lambda wildcards: scenarios.get(
            wildcards.scenario, {}
        ).get("thermal", {}).get("availability_factors", {}).get("generator_monthly_file") or [],
        espeni_dispatch=f"{data_path}/demand/espeni.csv",
        # Declared nuclear stations for CP30 scenarios (replaces the FES nuclear
        # proxy; see scripts/CP30/nuclear_schedule.py).
        cp30_nuclear_station_schedule=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("nuclear_station_schedule") or []
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        cp30_nuclear_schedule_logic="scripts/CP30/nuclear_schedule.py",
        # Declared gas stations and the CP30 gas split (items MR1 and MR2; see
        # scripts/CP30/gas_schedule.py).
        cp30_ccs_biomass_station_schedule=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("ccs_biomass_station_schedule") or []
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        cp30_biomass_station_schedule=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("biomass_station_schedule") or []
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        cp30_gas_station_schedule=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("gas_station_schedule") or []
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        cp30_gas_schedule_logic="scripts/CP30/gas_schedule.py",
        # Plant CP30 omits but that is contracted for 2030 (Net Zero Teesside Power).
        cp30_committed_overlay=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("committed_overlay") or []
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        cp30_committed_overlay_logic="scripts/CP30/committed_overlay.py",
        # Named CfD units carved out of the CP30 waste and biogas proxies
        # (scripts/CP30/cfd_station_schedule.py).
        cp30_cfd_station_schedule=lambda wildcards: (
            scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("cfd_station_schedule") or []
            if uses_brownfield(scenarios[wildcards.scenario]) else []
        ),
        cp30_cfd_station_schedule_logic="scripts/CP30/cfd_station_schedule.py",
        # Land boundary for the offshore-bus exclusion when placing those units
        # (network.spatial_mapping.placement_exclusions.land_boundary_file, else the default).
        cp30_placement_land_boundary=lambda wildcards: (
            ((scenarios[wildcards.scenario].get("network", {}).get("spatial_mapping", {}) or {})
             .get("placement_exclusions", {}) or {}).get("land_boundary_file")
            or "data/network/GSP/GSP_regions_27700_20250109.geojson"
            if uses_brownfield(scenarios[wildcards.scenario])
            and scenarios[wildcards.scenario].get("fleet", {}).get("spatial_proxy", {}).get("cfd_station_schedule") else []
        ),
        cp30_generation_logic="scripts/CP30/generation.py",
        # Task Y6b: the year's declared station schedule for FES long-range builds ([] otherwise),
        # and the year's ES1 lines when the gas CCS split is on (stage 5).
        fes_station_schedule=lambda wildcards: fes_supply_input(wildcards.scenario, "thermal", "station_schedule"),
        fes_es1_lines=lambda wildcards: fes_gas_ccs_split_input(wildcards.scenario),
        fes_supply_logic=lambda wildcards: fes_supply_logic_input(wildcards.scenario),
        # RO share of the anonymous CP30 biomass and energy-from-waste proxies
        # from the Ofgem register (26 Sep 2026).
        ro_register=lambda wildcards: _support_register_input(wildcards.scenario, "ro_register", brownfield_only=True),
        ro_certificates=lambda wildcards: _support_register_input(wildcards.scenario, "ro_certificates", brownfield_only=True),
        support_register_logic=lambda wildcards: _support_register_input(wildcards.scenario, "logic", brownfield_only=True),
        # Generator characteristics database
        fuel_data=f"{data_path}/generators/generator_data_by_fuel.csv"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal.pkl",
        summary=f"{resources_path}/generators/{{scenario}}_thermal_summary.csv",
        pre_solve_roc_mapping=f"{resources_path}/generators/{{scenario}}_pre_solve_roc_mapping.csv",
        pre_solve_roc_overlay_actions=f"{resources_path}/generators/{{scenario}}_pre_solve_roc_overlay_actions.csv",
        pre_solve_roc_capacity_ledger=f"{resources_path}/generators/{{scenario}}_pre_solve_roc_capacity_ledger.csv"
    params:
        scenario_config=lambda wildcards: model_scenario_param(scenarios[wildcards.scenario])
    log:
        "logs/integrate_thermal_generators_{scenario}.log"
    benchmark:
        "benchmarks/generators/integrate_thermal_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/generators/integrate_thermal_generators.py"


# Stage 3: Finalize integration with backup generators and comprehensive exports
rule finalize_generator_integration:
    """
    STAGE 3: Finalize generator integration with backup and comprehensive exports.
    
    This rule completes the generator integration workflow:
    
    Processing Steps:
      1. Add load shedding generators (VoLL backup at every bus)
      2. Export full generators CSV (all generator data)
      3. Create summary by carrier type (wind, solar, CCGT, etc.)
      4. Create technology capacity summary (detailed breakdown)
      5. Generate integration report (text summary)
      6. Create HTML visualization (interactive summary page)
    
    Load Shedding Generators:
      - Added to every bus as backup power source
      - Marginal cost = Value of Lost Load (VoLL, default £6000/MWh)
      - Ensures optimization always has feasible solution
      - Represents involuntary demand reduction
    
    Input:
      - Network with renewable + thermal generators
      - VoLL parameter (from scenario configuration)
    
    Output:
      - Network: {scenario}_network_demand_renewables_thermal_generators.pkl (storage, hydrogen and
        interconnectors are added after it)
      - Full generators CSV: {scenario}_generators_full.csv
      - Summary by carrier: {scenario}_generators_summary_by_carrier.csv
      - Technology summary: {scenario}_technology_capacity_summary.csv
      - Integration report: {scenario}_generator_integration_report.txt
      - HTML visualization: {scenario}_generators_map.html
    
    Performance: ~15-30 seconds (depends on number of buses for load shedding)
    
    Final Output: Complete network ready for optimization analysis
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal.pkl"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators.pkl",
        csv_generators_full=f"{resources_path}/generators/{{scenario}}_generators_full.csv",
        csv_generators_summary=f"{resources_path}/generators/{{scenario}}_generators_summary_by_carrier.csv",
        csv_technology_summary=f"{resources_path}/generators/{{scenario}}_technology_capacity_summary.csv",
        csv_integration_report=f"{resources_path}/generators/{{scenario}}_generator_integration_report.txt"
    params:
        voll=lambda w: scenarios[w.scenario].get('solve', {}).get('voll', 6000.0)  # Value of Lost Load in £/MWh
    log:
        "logs/finalize_generator_integration_{scenario}.log"
    benchmark:
        "benchmarks/generators/finalize_integration_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/generators/finalize_generator_integration.py"


# ══════════════════════════════════════════════════════════════════════════════
# DAILY EMPIRICAL MC CALIBRATION (from ELEXON data)
# ══════════════════════════════════════════════════════════════════════════════

def _daily_mc_enabled(scenario_id):
    """Check if daily empirical MC calibration is enabled for a scenario."""
    sc = scenarios.get(scenario_id, {})
    return (sc.get('marginal_costs', {})
              .get('empirical_calibration', {})
              .get('enabled', False))


def _renewable_mc_enabled(scenario_id):
    """Check if renewable empirical MC calibration is enabled for a scenario."""
    sc = scenarios.get(scenario_id, {})
    return (sc.get('marginal_costs', {})
              .get('empirical_calibration', {})
              .get('enabled', False))


def _carrier_correction_factors_file(scenario_id):
    """Return the carrier correction factors CSV path if enabled, else empty list."""
    sc = scenarios.get(scenario_id, {})
    ccf = sc.get('marginal_costs', {}).get('carrier_correction_factors', {})
    if ccf.get('enabled', False) and ccf.get('file'):
        return ccf['file']
    return []


def _future_cost_source_file(scenario_id):
    """Return the generated commodity source required by a future-cost case."""
    future_costs = scenarios.get(scenario_id, {}).get('marginal_costs', {}).get(
        'future_costs', {}
    )
    if not future_costs.get('enabled', False):
        return []
    cases = future_costs.get('cases', {})
    case_name = str(future_costs.get('case', 'central')).lower()
    source_file = cases.get(case_name, {}).get('source_file')
    return source_file or []


rule calibrate_daily_empirical_mc:
    """
    Fetch ELEXON B1610 + MID data and calibrate per-generator empirical
    marginal costs at daily resolution (rolling +/-N-day window).

    Only runs when marginal_costs.empirical_calibration.enabled: true.
    Output is consumed by apply_marginal_costs_to_network to build day-aware
    time-varying MC (generators_t.marginal_cost).
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators.pkl",
    output:
        daily_mc=f"{resources_path}/marginal_costs/{{scenario}}_daily_empirical_mc.csv",
    params:
        scenario_config=lambda w: scenario_param(scenarios[w.scenario]),
        data_dir="resources/market/elexon",
    log:
        "logs/calibrate_daily_mc_{scenario}.log"
    benchmark:
        "benchmarks/generators/calibrate_daily_mc_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/generators/calibrate_daily_mc.py"


rule calibrate_renewable_empirical_mc:
    """
    Calibrate renewable generator MCs from ELEXON BOD bid prices.

    Only runs when marginal_costs.empirical_calibration.enabled: true.
    For future scenarios (>2024), produces an empty CSV.
    Output is consumed by apply_marginal_costs_to_network.
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators.pkl",
    output:
        renewable_mc=f"{resources_path}/marginal_costs/{{scenario}}_renewable_empirical_mc.csv",
    params:
        scenario_config=lambda w: scenario_param(scenarios[w.scenario]),
        data_dir="resources/market/elexon",
    log:
        "logs/calibrate_renewable_mc_{scenario}.log"
    benchmark:
        "benchmarks/generators/calibrate_renewable_mc_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/generators/calibrate_renewable_mc.py"


# ══════════════════════════════════════════════════════════════════════════════
# MARGINAL COST CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

rule fetch_neso_cp2030_annex4_costs:
    """Download and parse NESO Clean Power 2030 Annex 4 into an auditable JSON input."""
    output:
        pdf=f"{resources_path}/market/source/neso_clean_power_2030_annex4.pdf",
        costs=f"{resources_path}/market/neso_clean_power_2030_annex4_costs.json",
    log:
        "logs/market/fetch_neso_cp2030_annex4_costs.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/market/fetch_neso_cp2030_annex4_costs.py"

rule apply_marginal_costs_to_network:
    """
    Compute and apply marginal costs to thermal generators.

    CONFIGURATION:
    --------------
    Marginal costs are configured in YAML files (no code editing needed):

    In config/defaults/ (project-wide defaults, file base/marginal_costs.yaml):
        marginal_costs:
          carbon_price: 85.0              # £/tonne CO2
          fuel_prices:
            gas: 35.0                     # £/MWh thermal
            coal: 30.0
          use_fes_prices: true            # Use FES dynamic prices if available

    In config/scenarios.yaml (scenario-specific override):
        My_Scenario:
          modelled_year: 2035
          FES_year: 2024
          marginal_costs:
            carbon_price: 150.0           # Override carbon price
            fuel_prices:
              gas: 50.0                   # Override gas price only

    AUTOMATIC BEHAVIOR:
    -------------------
    Priority order for price selection:
      1. Scenario-specific override (scenarios.yaml)
      2. Historical lookup tables (for modelled_year ≤ 2024)
      3. FES dynamic prices (if FES_year specified and use_fes_prices=true)
      4. Configuration defaults (config/defaults/)
      5. Fallback hardcoded values

    Historical scenarios (≤2024): Uses built-in historical lookup tables
    Future scenarios (>2024): Uses FES projections → config defaults → fallback

    MARGINAL COST FORMULA:
    ----------------------
      MC = (Fuel_Price / Efficiency) + (Carbon_Price × Emission_Factor / Efficiency)

    Typical Values (2024):
      - CCGT: £50-80/MWh (gas £30-50 + carbon £20-30)
      - OCGT: £80-120/MWh (less efficient than CCGT)
      - Coal: £90-150/MWh (higher carbon emissions)
      - Nuclear: £15/MWh (low marginal operating cost)
      - Renewables: £0/MWh (zero fuel cost)
      - Storage: £0.05-0.20/MWh (wear and tear)

    INPUT/OUTPUT:
    -------------
    Input:
      - Network with generators (zero marginal costs from integration)
      - Optional: FES fuel/carbon price CSVs (auto-loaded for future scenarios)

    Output:
      - Updated network with marginal costs applied to generators
      - CSV with marginal cost breakdown (fuel + carbon components)

    Performance: ~5-10 seconds

    Critical for Optimization:
      Without marginal costs, thermal generators have zero cost, creating
      unbounded optimization (infinite free generation + storage arbitrage).
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators.pkl",
        # Optional: FES price inputs for future scenarios (>2024)
        fuel_prices=lambda w: get_cost_price_file(w.scenario, 'fuel_prices'),
        carbon_prices=lambda w: get_cost_price_file(w.scenario, 'carbon_prices'),
        # Optional: Daily thermal empirical MC calibration
        daily_mc=lambda w: f"{resources_path}/marginal_costs/{w.scenario}_daily_empirical_mc.csv" if _daily_mc_enabled(w.scenario) else [],
        # Optional: Renewable empirical MC calibration
        renewable_mc=lambda w: f"{resources_path}/marginal_costs/{w.scenario}_renewable_empirical_mc.csv" if _renewable_mc_enabled(w.scenario) else [],
        # Optional: Carrier correction factors (bridges historical calibration to future)
        correction_factors=lambda w: _carrier_correction_factors_file(w.scenario),
        # Optional: generated external commodity source (e.g. NESO Annex 4).
        future_cost_source=lambda w: _future_cost_source_file(w.scenario),
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_costs.pkl",
        marginal_costs_csv=f"{resources_path}/generators/{{scenario}}_marginal_costs_breakdown.csv"
    params:
        scenario_config=lambda w: model_scenario_param(scenarios[w.scenario]),
        fuel_price_file=lambda w: get_cost_price_param(w.scenario, 'fuel_prices'),
        carbon_price_file=lambda w: get_cost_price_param(w.scenario, 'carbon_prices'),
        thermal_mc_file=lambda w: f"{resources_path}/marginal_costs/{w.scenario}_daily_empirical_mc.csv" if _daily_mc_enabled(w.scenario) else None,
        renewable_mc_file=lambda w: f"{resources_path}/marginal_costs/{w.scenario}_renewable_empirical_mc.csv" if _renewable_mc_enabled(w.scenario) else None,
        correction_factors_file=lambda w: _carrier_correction_factors_file(w.scenario) or None,
    log:
        "logs/marginal_costs_{scenario}.log"
    benchmark:
        "benchmarks/generators/marginal_costs_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/generators/apply_marginal_costs.py"


# Export a PyPSA network with generators to CSV tables
rule export_network_to_csv:
    """
    Export the complete PyPSA network (after generator integration) to CSV files.

    Writes all component tables to a stable folder and a completion marker
    file used by Snakemake for dependency tracking.
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators.pkl"
    output:
        marker=f"{resources_path}/network/csv/{{scenario}}_base_demand_generators/export_complete.txt"
    params:
        export_dir=lambda wildcards: f"{resources_path}/network/csv/{wildcards.scenario}_base_demand_generators"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        f"logs/export_network_csv_{{scenario}}.log"
    script:
        "../scripts/generators/export_network_to_csv.py"


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE RULES FOR GENERATOR PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# Aggregate rule: Complete 3-stage generator integration pipeline
rule integrate_generators:
    """
    Complete 3-stage modular generator integration pipeline (RECOMMENDED).
    
    This rule orchestrates the full generator integration workflow:
      Stage 1: Renewable generators (wind, solar, marine, hydro)
      Stage 2: Thermal generators (CCGT, nuclear, biomass, waste)
      Stage 3: Finalization (load shedding + comprehensive exports)
    
    Target it through its marker file:
        snakemake resources/generators/{scenario}_integration_complete.txt --cores 4
    """
    input:
        # Final outputs from 3-stage pipeline
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators.pkl",
        csv_generators_full=f"{resources_path}/generators/{{scenario}}_generators_full.csv",
        csv_generators_summary=f"{resources_path}/generators/{{scenario}}_generators_summary_by_carrier.csv",
        csv_technology_summary=f"{resources_path}/generators/{{scenario}}_technology_capacity_summary.csv",
        csv_integration_report=f"{resources_path}/generators/{{scenario}}_generator_integration_report.txt",
        csv_export_marker=f"{resources_path}/network/csv/{{scenario}}_base_demand_generators/export_complete.txt"
    output:
        marker=f"{resources_path}/generators/{{scenario}}_integration_complete.txt"
    shell:
        """
        echo "3-stage modular generator integration pipeline complete!" > {output.marker}
        echo "  ✓ Stage 1: Renewable generators integrated" >> {output.marker}
        echo "  ✓ Stage 2: Thermal generators integrated" >> {output.marker}
        echo "  ✓ Stage 3: Integration finalized with exports" >> {output.marker}
        echo "" >> {output.marker}
        echo "Network ready for optimization: {input.network}" >> {output.marker}
        """


# =============================================================================
# LEGACY MARGINAL COST RULES - REMOVED
# =============================================================================
# The legacy compute_marginal_costs and compute_marginal_costs_with_update rules
# have been removed. These used scripts/generators/marginal_costs.py which is NOT
# part of the main workflow.
#
# For marginal cost calculation, the active rule is:
#   apply_marginal_costs_to_network (line 958)
#
# This rule uses scripts/generators/apply_marginal_costs.py and is integrated
# into the standard workflow.
#
# Configuration is now via config/defaults/ and config/scenarios.yaml.
# See documentation for details.
# =============================================================================

