# ══════════════════════════════════════════════════════════════════════════════
# DEMAND RULES - CONSOLIDATED
# ══════════════════════════════════════════════════════════════════════════════
#
# This unified module combines all demand-related rules:
#   1. Base demand integration (build_base_demand)
#   2. Demand disaggregation (heat pumps, EVs, etc.)
#   3. Demand-side flexibility (V1G/V2G, thermal storage, DR)
#
# Workflow Stages:
#   Stage 1: Build base (total) electricity demand from FES/historical data
#   Stage 2: (Optional) Disaggregate total demand into components
#   Stage 3: (Optional) Model demand-side flexibility resources
#
# Architecture:
#   - Modular: Each stage is independent
#   - Flexible: Stages can be enabled/disabled via config
#   - Extensible: Easy to add new components
#   - Debuggable: Separate logs for each rule
#
# Configuration:
#   - Base demand: scenarios_master.yaml (FES year, demand year, timeseries source)
#   - Disaggregation: scenarios_master.yaml (demand.disaggregation section)
#   - Flexibility: scenarios_master.yaml (flexibility section, if enabled)
#
# Dependencies:
#   - Base network topology (from network_build.smk)
#   - FES data (from FES.smk)
#   - Historical demand timeseries (data/demand/)
#
# ══════════════════════════════════════════════════════════════════════════════


     # CSV demand profile no longer produced; network NetCDF contains base demand.
     # The script `scripts/demand/load.py` writes the network to `network_with_base_demand`.
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def scenario_is_historical(scenario_id):
    """Check if a scenario is historical (modelled_year <= 2024)."""
    scenario = scenarios.get(scenario_id, {})
    modelled_year = scenario.get("modelled_year", 2035)
    return modelled_year <= 2024


def scenario_has_cp30_targets(scenario_id):
    """True when the scenario's targets come from CP30 (its FES data is then a spatial proxy)."""
    from scripts.utilities.scenario_detection import target_source
    return target_source(scenarios.get(scenario_id, {})) == "CP30"


def scenario_uses_brownfield(scenario_id):
    """True when the scenario is built brownfield (the method whose defaults carry demand.build)."""
    from scripts.utilities.scenario_detection import uses_brownfield
    return uses_brownfield(scenarios.get(scenario_id, {}))


def get_demand_fes_proxy(scenario_id):
    """Return FES proxy config used by demand for future scenarios."""
    scenario = scenarios.get(scenario_id, {})
    if scenario_has_cp30_targets(scenario_id):
        return scenario.get("fleet", {}).get("spatial_proxy", {})
    return scenario


def get_fes_capacity_redistribution(scenario_id):
    """demand.disaggregation.fes.capacity_redistribution of a gsp_block ETYS scenario, else {}.

    F25.8 (user, 27 Oct 2026): the FES long-range chain moves demand above a capped <=132 kV bus to the
    same site's supergrid bus, as the CP30 2030 demand build does; {} keeps the pro rata step (and the
    params) of every other scenario.
    """
    sc = scenarios.get(scenario_id, {})
    if scenario_uses_brownfield(scenario_id) or str(sc.get("network_model", "")).upper() != "ETYS":
        return {}
    return ((((sc.get("demand") or {}).get("disaggregation") or {}).get("fes") or {})
            .get("capacity_redistribution") or {})


def get_cp30_demand_config(scenario_id):
    """demand.build of a brownfield scenario when enabled (CP30 2030 demand build), else {}.

    {} keeps build_base_demand's behaviour (and its params) unchanged for every scenario
    that does not enable it.
    """
    if not scenario_uses_brownfield(scenario_id):
        return {}
    # Named loads sit on ETYS nodes; Reduced/Zonal CP30 builds keep the original method.
    if str(scenarios.get(scenario_id, {}).get("network_model", "")).upper() != "ETYS":
        return {}
    cfg = (scenarios.get(scenario_id, {}).get("demand") or {}).get("build") or {}
    return cfg if cfg.get("enabled", False) else {}


def _demand_regex_from_list(items):
    """A regex matching exactly the given ids (never-matching when the list is empty)."""
    import re
    if not items:
        return r"(?!x)x"
    return r"(?:%s)" % "|".join(map(re.escape, items))


def get_demand_fes_year(scenario_id):
    # The CP30 demand build takes its GSP shares from its own FES vintage (FES 2024, the
    # one CP30 was built on), not the generation spatial proxy (demand audit, fix 1).
    spatial = get_cp30_demand_config(scenario_id).get("spatial") or {}
    if spatial.get("FES_year"):
        return spatial["FES_year"]
    proxy = get_demand_fes_proxy(scenario_id)
    return proxy.get("FES_year")


def get_demand_fes_scenario(scenario_id):
    spatial = get_cp30_demand_config(scenario_id).get("spatial") or {}
    if spatial.get("FES_scenario"):
        return spatial["FES_scenario"]
    proxy = get_demand_fes_proxy(scenario_id)
    return proxy.get("FES_scenario")


# Outputs of rule extract_cp30_demand_inputs (below); cp30_demand.DEFAULT_CONFIG names the same files.
CP30_NAMED_LOADS_FILE = f"{resources_path}/demand/cp30_2030_named_large_loads.csv"
CP30_DEMAND_LINES_FILE = f"{resources_path}/demand/fes24_ht_2030_demand_lines.csv"
# FES24 ETYS HT 2030/31 winter peak by ETYS id, located: read by validate_CP30_locational only.
CP30_ETYS_PEAK_FILE = f"{resources_path}/demand/fes24_etys_ht_2030_peak_by_gsp.csv"


def cp30_demand_inputs(scenario_id):
    """Data files the CP30 demand build reads (empty unless demand.build.enabled)."""
    cfg = get_cp30_demand_config(scenario_id)
    if not cfg:
        return {}
    out = {
        "cp30_national_lines": cfg.get("national_lines_file", CP30_DEMAND_LINES_FILE),
        "cp30_logic": "scripts/demand/cp30_demand.py",
    }
    named = cfg.get("named_loads") or {}
    if named.get("enabled", True):
        out["cp30_named_loads"] = named.get("file", CP30_NAMED_LOADS_FILE)
    if (cfg.get("shape") or {}).get("enabled", True):
        year = scenarios[scenario_id].get("renewables_year", 2020)
        out["cutout"] = f"{resources_path}/atlite/cutouts/uk-{year}.nc"
    # F5 consumer flexibility (demand.build.flexibility, off by default): its logic re-runs the build.
    if (cfg.get("flexibility") or {}).get("enabled", False):
        out["cp30_flex_logic"] = "scripts/demand/cp30_flexibility.py"
    return out


def get_demand_cp30_resource(scenario_id, key):
    return scenarios.get(scenario_id, {}).get("source_data", {}).get("files", {}).get(key)


def get_fes_data_input(wildcards):
    """Return FES data path for future scenarios, or placeholder for historical."""
    scenario_id = wildcards.scenario
    if scenario_is_historical(scenario_id):
        # Historical scenarios don't need FES data
        # Return espeni file as placeholder (it's always required anyway)
        return "data/demand/espeni.csv"
    else:
        fes_year = get_demand_fes_year(scenario_id) or 2024
        return f"{resources_path}/FES/FES_{fes_year}_data.csv"


def get_disaggregation_config(scenario):
    """Get disaggregation config for a scenario, or return disabled default."""
    return demand_disaggregation_configs.get(scenario, {"enabled": False})


def is_disaggregation_enabled(scenario):
    """Check if disaggregation is enabled for this scenario."""
    config = get_disaggregation_config(scenario)
    return config.get("enabled", False)


def get_component_names(scenario):
    """Get list of component names configured for this scenario."""
    config = get_disaggregation_config(scenario)
    if not config.get("enabled", False):
        return []
    components = config.get("components", [])
    names = []
    for comp in components:
        if isinstance(comp, str):
            names.append(comp)
        elif isinstance(comp, dict):
            name = comp.get("name")
            if name:
                names.append(name)
    return names


def get_component_config(scenario, component_name):
    """Get configuration for a specific component."""
    config = get_disaggregation_config(scenario)
    components = config.get("components", [])
    for comp in components:
        if isinstance(comp, str) and comp == component_name:
            return {}
        if isinstance(comp, dict) and comp.get("name") == component_name:
            return comp
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Demand Flexibility Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_flexibility_config(scenario):
    """Get demand flexibility config for a scenario."""
    scenario_config = scenarios.get(scenario, {})
    return scenario_config.get("demand", {}).get("flexibility", config.get("demand", {}).get("flexibility", {}))


def is_flexibility_enabled(scenario):
    """Check if demand flexibility is enabled for this scenario."""
    flex_config = get_flexibility_config(scenario)
    return flex_config.get("enabled", False)


def is_hp_flexibility_enabled(scenario):
    """Check if heat pump flexibility is enabled for this scenario."""
    if not is_flexibility_enabled(scenario):
        return False
    flex_config = get_flexibility_config(scenario)
    hp_config = flex_config.get("heat_pumps", {})
    return hp_config.get("enabled", False)


def is_ev_flexibility_enabled(scenario):
    """Check if EV flexibility is enabled for this scenario."""
    if not is_flexibility_enabled(scenario):
        return False
    flex_config = get_flexibility_config(scenario)
    ev_config = flex_config.get("electric_vehicles", {})
    return ev_config.get("enabled", False)


def is_event_flexibility_enabled(scenario):
    """Check if event response flexibility is enabled for this scenario."""
    if not is_flexibility_enabled(scenario):
        return False
    flex_config = get_flexibility_config(scenario)
    event_config = flex_config.get("event_response", {})
    return event_config.get("enabled", False)


def is_v2g_with_fes_enabled(scenario):
    """Check if V2G tariff with FES capacity is enabled."""
    if not is_ev_flexibility_enabled(scenario):
        return False
    flex_config = get_flexibility_config(scenario)
    ev_config = flex_config.get("electric_vehicles", {})
    tariff = ev_config.get("tariff", "INT").upper()
    use_fes = ev_config.get("v2g", {}).get("use_fes_capacity", False)
    return tariff == "V2G" and use_fes


def is_event_response_with_fes_enabled(scenario):
    """Check if event response with FES capacity is enabled."""
    if not is_event_flexibility_enabled(scenario):
        return False
    flex_config = get_flexibility_config(scenario)
    event_config = flex_config.get("event_response", {})
    return event_config.get("use_fes_capacity", False)


def requires_fes_data_for_flexibility(scenario):
    """Check if any flexibility component requires FES data."""
    return is_v2g_with_fes_enabled(scenario) or is_event_response_with_fes_enabled(scenario)


# ──────────────────────────────────────────────────────────────────────────────
# FES long-range demand components (any FES year)
# ──────────────────────────────────────────────────────────────────────────────
# Settings live under demand.disaggregation.fes (with demand.disaggregation.enabled:
# false), so finalize_demand's params are unchanged for every scenario without them.

def get_fes_demand_config(scenario):
    """demand.disaggregation.fes of a scenario, or {} (disabled)."""
    from scripts.demand.fes_demand import fes_demand_config
    return fes_demand_config(get_disaggregation_config(scenario))


def is_fes_demand_enabled(scenario):
    return bool(get_fes_demand_config(scenario).get("enabled", False))


def fes_demand_path(scenario, kind):
    # Keyed by scenario id; the directory keeps its first name so existing outputs keep theirs.
    return f"{resources_path}/demand/fes2050/{scenario}_{kind}"


def fes_demand_targets_file(scenario):
    """The targets JSON the components are pinned to: the block's ``targets_file``, else the
    scenario's own FES edition, pathway and year (rule build_fes2025_energy_targets)."""
    block = get_fes_demand_config(scenario)
    if block.get("targets_file"):
        return block["targets_file"]
    return fes_targets_path(scenarios[scenario])


def fes_demand_workbook(scenario):
    """The FES Data Workbook of the block (``workbook``), else of the scenario's FES edition."""
    block = get_fes_demand_config(scenario)
    if block.get("workbook"):
        return block["workbook"]
    from scripts.sources.fes_adapter import WORKBOOKS
    edition = int(scenarios[scenario]["targets"]["edition"])
    if edition not in WORKBOOKS:
        raise ValueError(f"{scenario}: no FES {edition} Data Workbook registered (scripts/sources/fes_adapter.WORKBOOKS)")
    return str(WORKBOOKS[edition].relative_to(WORKBOOKS[edition].parents[2])).replace("\\", "/")


# The scenarios that build the components (exact ids, never a name pattern).
FES_DEMAND_SCENARIO_REGEX = _demand_regex_from_list(sorted(s for s in scenarios if is_fes_demand_enabled(s)))


def get_final_demand_network(wildcards):
    """Return the finalized demand-side network for downstream rules."""
    scenario = wildcards.scenario
    return f"{resources_path}/network/{scenario}_network_demand.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: BASE DEMAND INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

rule build_base_demand:
    """
    Build base (total) electricity demand profile and attach to network.
    
    This is the FIRST STAGE of demand modeling. It creates the total system
    demand WITHOUT any sector-specific disaggregation.
    
    Future Scenario Demand Components:
    ===================================
    1. ANNUAL DEMAND (from FES):
       - Total annual electricity consumption for the modelled_year
       - Loaded from FES CSV file, filtered by FES_scenario
       - Units: GWh per GSP (Grid Supply Point)
       
    2. SPATIAL DISTRIBUTION (from FES):
       - Demand is distributed across network buses using FES GSP-level data
       - GSP names are mapped to ETYS node IDs (for ETYS networks)
       - Uses Dem_per_node weights from GB_network.xlsx
       
    3. TEMPORAL PROFILE (configurable via demand.timeseries):
       - "ESPENI": Historical ESPENI half-hourly demand profile
         → Real GB demand data, scaled to FES annual total
         → Uses demand_year for profile shape selection (default: 2020)
       - "eload": eLOAD model hourly profile (from egy_7649_mmc1.xlsx)
         → Bottom-up appliance-level demand model
         → Uses profile_year (2010 or 2050, auto-selected if not specified)
         → Auto-selects: 2050 for modelled_year >= 2040, otherwise 2010
       - "desstinee": DESSTINEE model hourly profile (from egy_7649_mmc1.xlsx)
         → Synthetic demand model based on weather/calendar/socio-economic factors
         → Uses profile_year (2010 or 2050, auto-selected if not specified)
         → Auto-selects: 2050 for modelled_year >= 2040, otherwise 2010
    
    Process:
      1. Load FES demand forecasts for the target year (annual demand by GSP)
      2. Select temporal profile source (ESPENI, eload, or desstinee)
      3. Scale temporal profile to match FES total annual demand per GSP
      4. Map GSP-level demand to network buses
      5. Attach base demand to network buses
      6. Export network with demand attached
    
    Transforms: {scenario}_network.nc → {scenario}_network_demand_base.pkl
    
    Inputs:
      - fes_data: FES demand totals by GSP for future scenarios (CSV)
      - espeni: ESPENI historical demand timeseries (CSV, half-hourly)
      - egy: eload historical demand timeseries (Excel, hourly) [Not yet implemented]
      - base_network: Base network topology from network_build rule (NetCDF)
    
    Outputs:
      - network_with_base_demand: Network with total demand attached (Pickle)
    
    Parameters:
      - fes_year: Year of FES dataset (2021-2025) - REQUIRED for future scenarios
      - fes_scenario: FES scenario name (e.g., "Holistic Transition") - REQUIRED for future
      - modelled_year: Target year for demand forecast (e.g., 2035)
      - demand_year: Historical year for ESPENI profile shape (e.g., 2020)
      - demand_timeseries: Source for temporal profile ("ESPENI", "eload", or "desstinee")
      - profile_year: For eload/desstinee only - which profile (2010 or 2050, auto-selected if omitted)
      - network_model: Network topology type (ETYS/Reduced/Zonal)
      - scenario: Scenario identifier for logging
    
    Network Model Handling:
      - ETYS: Demand distributed to ~2000 buses using GSP mapping
      - Reduced: Demand aggregated to 29 regional buses  
      - Zonal: Demand aggregated to 20 zone buses
    
    Next Steps:
      - finalize_demand combines disaggregation + flexibility (or passes through)
    
    Performance:
      - ETYS: ~30-40s (GSP spatial mapping)
      - Reduced: ~5-8s (simple aggregation)
      - Zonal: ~3-5s (simple aggregation)
    
    See Also:
      - Stage 2: Demand disaggregation (heat pumps, EVs, etc.)
      - scripts/demand/load.py for implementation details
    """
    input:
        fes_data=get_fes_data_input,
        espeni="data/demand/espeni.csv",
        egy="data/demand/egy_7649_mmc1.xlsx",
        # Declared locations for FES GSPs without a Dem_per_node row (read by load.py)
        gsp_location_overrides="data/demand/gsp_location_overrides.csv",
        base_network=f"{resources_path}/network/{{scenario}}_network.nc",
        # CP30 demand build (demand.build.enabled): FES 2024 lines and named large loads (rule
        # extract_cp30_demand_inputs), weather-year cutout for the heat shapes. Empty for
        # every other scenario.
        cp30_national_lines=lambda w: cp30_demand_inputs(w.scenario).get("cp30_national_lines", []),
        cp30_named_loads=lambda w: cp30_demand_inputs(w.scenario).get("cp30_named_loads", []),
        cp30_logic=lambda w: cp30_demand_inputs(w.scenario).get("cp30_logic", []),
        cp30_flex_logic=lambda w: cp30_demand_inputs(w.scenario).get("cp30_flex_logic", []),
        cutout=lambda w: cp30_demand_inputs(w.scenario).get("cutout", [])
    output:
        network_with_base_demand=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl",
        base_demand_profile=f"{resources_path}/demand/{{scenario}}_base_profile.csv"
    params:
        fes_year=lambda wildcards: get_demand_fes_year(wildcards.scenario),
        fes_scenario=lambda wildcards: get_demand_fes_scenario(wildcards.scenario),
        modelled_year=lambda wildcards: scenarios[wildcards.scenario]["modelled_year"],
        demand_year=lambda wildcards: scenarios[wildcards.scenario]["demand_year"],
        demand_timeseries=lambda wildcards: scenarios[wildcards.scenario].get("demand", {}).get("timeseries", "ESPENI"),
        profile_year=lambda wildcards: scenarios[wildcards.scenario].get("profile_year", None),
        network_model=lambda wildcards: scenarios[wildcards.scenario]["network_model"],
        timestep_minutes=lambda wildcards: scenarios[wildcards.scenario].get("timestep_minutes", 60),
        scenario=lambda wildcards: wildcards.scenario,
        scenario_source=lambda wildcards: scenarios[wildcards.scenario].get("scenario_source"),
        cp30_pathway=lambda wildcards: scenarios[wildcards.scenario].get("CP30_pathway"),
        cp30_ed1=lambda wildcards: get_demand_cp30_resource(wildcards.scenario, "ed1"),
        cp30_es1=lambda wildcards: get_demand_cp30_resource(wildcards.scenario, "es1"),
        is_historical=lambda wildcards: scenario_is_historical(wildcards.scenario),
        embedded_generation=lambda wildcards: scenarios[wildcards.scenario].get("renewables", {}).get(
            "embedded", config.get("renewables", {}).get("embedded", {})),
        cp30_demand=lambda wildcards: get_cp30_demand_config(wildcards.scenario),
        fes_capacity_redistribution=lambda wildcards: get_fes_capacity_redistribution(wildcards.scenario),
    message:
        "Building base demand for {wildcards.scenario} (Year: {params.demand_year}, Source: {params.demand_timeseries})"
    benchmark:
        "benchmarks/demand/build_base_demand_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/demand/build_base_demand_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/demand/load.py"


rule extract_cp30_demand_inputs:
    """
    CP30 2030 demand inputs from the FES 2024 workbooks (demand audit fix 2, 15 Oct 2026).

    Writes the transmission-connected named large loads (FES24 ETYS Holistic Transition
    2030/31, demand type T: data centres, HS2 and Network Rail feeders, St Fergus INTOG,
    Port Talbot, the industrial direct connects) with their ETYS nodes, and the FES 2024
    ED1 Holistic Transition 2030 lines the CP30 demand build uses, and the FES24 ETYS HT
    2030/31 winter peak by ETYS id (all types, located). Nodes: GB_network.xlsx
    Dir_con_BMUs_to_node for BMUs; data/demand/cp30_named_load_nodes.csv (hand choices,
    each with its evidence) for the B_EXTRA rows. Read by build_base_demand
    (demand.build.enabled) and validate_CP30_locational (the demand check's references), so
    a change to a workbook or to the node choices re-runs both.
    scripts/demand/extract_cp30_demand_inputs.py (also a CLI).
    """
    input:
        etys_breakdown="data/network/ETYS/Regional breakdown of FES24 data.xlsx",
        gb_network="data/network/ETYS/GB_network.xlsx",
        node_choices="data/demand/cp30_named_load_nodes.csv",
        fes24_workbook="data/FES/Future Energy Scenarios 2024 Data Workbook_V006_0.xlsx",
        logic="scripts/demand/extract_cp30_demand_inputs.py"
    output:
        named_loads=CP30_NAMED_LOADS_FILE,
        national_lines=CP30_DEMAND_LINES_FILE,
        etys_peak=CP30_ETYS_PEAK_FILE
    log:
        "logs/demand/extract_cp30_demand_inputs.log"
    benchmark:
        "benchmarks/demand/extract_cp30_demand_inputs.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/extract_cp30_demand_inputs.py"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: DEMAND DISAGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Optional second stage: Disaggregate total demand into sector components.
#
# Architecture:
#   1. build_base_demand (above) → Creates total demand
#   2. disaggregate_* rules (below) → Process individual components in PARALLEL
#   3. finalize_demand -> Combine components + flexibility into final demand network
#
# To Add a New Component:
#   1. Create scripts/demand/your_component.py
#   2. Add rule disaggregate_your_component (copy template below)
#   3. Update integrate rule inputs
#   4. Configure in scenarios_master.yaml
#
# ──────────────────────────────────────────────────────────────────────────────

rule disaggregate_heat_pumps:
    """
    Disaggregate heat pump demand from total electricity demand.

    Process:
      1. Generate temperature-based thermal demand profile (heating degree hours)
      2. Calculate temperature-dependent COP profile
      3. Compute electrical demand = thermal demand / COP
      4. Scale to match configured fraction of total demand
      5. Allocate spatially across network buses using selected method

    Key Physics:
      - Thermal demand driven by outdoor temperature (heating degree hours)
      - COP inversely related to temperature lift (outdoor to indoor)
      - Electrical demand = Thermal demand / COP
      - Cold weather: high thermal demand + low COP = very high electrical demand

    Allocation Methods:
      - proportional: Distribute proportional to existing demand
      - uniform: Equal distribution across all buses
      - urban_weighted: Weighted towards high-demand (urban) areas
      - fes_gsp: Use FES building block GSP distribution (future scenarios)

    Outputs:
      - profile: Hourly/half-hourly HP electrical demand timeseries (MW)
      - allocation: Spatial distribution across buses (GWh per bus per year)
      - cop_profile: Temperature-dependent COP timeseries
      - thermal_profile: Thermal demand timeseries (MW thermal)

    See Also:
      - scripts/demand/heat_pumps.py for implementation
      - scripts/demand/build_heat_profiles.py for Atlite-based profiles
    """
    input:
        # Use the network pickle that contains base demand (fast I/O)
        base_demand=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl",
        base_profile=f"{resources_path}/demand/{{scenario}}_base_profile.csv",
        fes_data=get_fes_data_input,
        # Atlite-generated heat profiles (temperature-based)
        heat_demand_profile=f"{resources_path}/demand/heat_demand_{{scenario}}.nc",
        cop_ashp_profile=f"{resources_path}/demand/cop_ashp_{{scenario}}.nc",
    output:
        profile=f"{resources_path}/demand/components/{{scenario}}_heat_pumps_profile.csv",
        allocation=f"{resources_path}/demand/components/{{scenario}}_heat_pumps_allocation.csv",
        cop_profile=f"{resources_path}/demand/components/{{scenario}}_heat_pumps_cop.csv",
        thermal_profile=f"{resources_path}/demand/components/{{scenario}}_heat_pumps_thermal.csv"
    params:
        component_config=lambda wildcards: get_component_config(wildcards.scenario, "heat_pumps"),
        scenario=lambda wildcards: wildcards.scenario,
        modelled_year=lambda wildcards: scenarios[wildcards.scenario]["modelled_year"],
        fes_scenario=lambda wildcards: scenarios[wildcards.scenario].get("FES_scenario"),
        fes_year=lambda wildcards: scenarios[wildcards.scenario].get("FES_year"),
        network_model=lambda wildcards: scenarios[wildcards.scenario]["network_model"],
        is_historical=lambda wildcards: scenario_is_historical(wildcards.scenario)
    message:
        "Disaggregating heat pump demand for {wildcards.scenario}"
    benchmark:
        "benchmarks/demand/disaggregate_heat_pumps_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/demand/disaggregate_heat_pumps_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/demand/heat_pumps.py"


rule disaggregate_electric_vehicles:
    """
    Disaggregate electric vehicle charging demand from total electricity demand.
    
    Process:
      1. Load EV charging profile from source file (or generate synthetic)
      2. Scale to match configured fraction of total demand
      3. Allocate spatially (e.g., urban-weighted for higher city adoption)
      4. Create separate timeseries for EV charging load
    
    EV Charging Patterns:
      - Peak evening charging (18:00-22:00)
      - Higher weekday usage
      - Concentrated in urban areas

    Allocation Methods:
      - proportional: Distribute proportional to existing demand
      - uniform: Equal distribution across all buses
      - urban_weighted: Weighted towards high-demand (urban) areas
      - fes_gsp: Use FES building block GSP distribution (future scenarios)
    
    Outputs:
      - profile: Hourly/half-hourly EV charging demand timeseries (GWh)
      - allocation: Spatial distribution across buses (GWh per bus per year)
    
    See Also:
      - scripts/demand/electric_vehicles.py for implementation
    """
    input:
        # Use the network pickle that contains base demand (fast I/O)
        base_demand=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl",
        base_profile=f"{resources_path}/demand/{{scenario}}_base_profile.csv",
        fes_data=get_fes_data_input,
    output:
        profile=f"{resources_path}/demand/components/{{scenario}}_ev_profile.csv",
        allocation=f"{resources_path}/demand/components/{{scenario}}_ev_allocation.csv"
    params:
        component_config=lambda wildcards: get_component_config(wildcards.scenario, "electric_vehicles"),
        scenario=lambda wildcards: wildcards.scenario,
        modelled_year=lambda wildcards: scenarios[wildcards.scenario]["modelled_year"],
        fes_scenario=lambda wildcards: scenarios[wildcards.scenario].get("FES_scenario"),
        fes_year=lambda wildcards: scenarios[wildcards.scenario].get("FES_year"),
        network_model=lambda wildcards: scenarios[wildcards.scenario]["network_model"],
        is_historical=lambda wildcards: scenario_is_historical(wildcards.scenario)
    message:
        "Disaggregating EV charging demand for {wildcards.scenario}"
    benchmark:
        "benchmarks/demand/disaggregate_ev_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/demand/disaggregate_ev_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/demand/electric_vehicles.py"


rule finalize_demand:
    """
    Finalize complete demand-side network (disaggregation + flexibility).

    This rule orchestrates:
    1. Load base demand network
    2. Integrate disaggregated components (if enabled)
    3. Add demand flexibility components (if enabled)

    Output is complete demand-side network ready for generator integration.
    """
    input:
        base_network=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl",
        base_profile=f"{resources_path}/demand/{{scenario}}_base_profile.csv",
        # Disaggregation inputs (conditional)
        hp_profile=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_profile.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        hp_allocation=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_allocation.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        hp_cop=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_cop.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        hp_thermal=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_thermal.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        ev_profile=lambda w: f"{resources_path}/demand/components/{w.scenario}_ev_profile.csv" if is_disaggregation_enabled(w.scenario) and "electric_vehicles" in get_component_names(w.scenario) else [],
        ev_allocation=lambda w: f"{resources_path}/demand/components/{w.scenario}_ev_allocation.csv" if is_disaggregation_enabled(w.scenario) and "electric_vehicles" in get_component_names(w.scenario) else [],
        # Flexibility inputs (conditional)
        cop_ashp=lambda w: f"{resources_path}/demand/cop_ashp_{w.scenario}.nc" if is_hp_flexibility_enabled(w.scenario) else [],
        ev_availability=lambda w: f"{resources_path}/demand/ev_availability_{w.scenario}.csv" if is_ev_flexibility_enabled(w.scenario) else [],
        ev_dsm=lambda w: f"{resources_path}/demand/ev_dsm_{w.scenario}.csv" if is_ev_flexibility_enabled(w.scenario) else [],
        # FES data for V2G capacity and event response DSR capacity (conditional)
        fes_data=lambda w: get_fes_data_input(w) if requires_fes_data_for_flexibility(w.scenario) else [],
        # FES long-range components (plan Y6a; conditional, empty for every other scenario)
        fes_demand_profiles=lambda w: fes_demand_path(w.scenario, "profiles.csv") if is_fes_demand_enabled(w.scenario) else [],
        fes_demand_allocation=lambda w: fes_demand_path(w.scenario, "allocation.csv") if is_fes_demand_enabled(w.scenario) else [],
        fes_demand_spec=lambda w: fes_demand_path(w.scenario, "spec.json") if is_fes_demand_enabled(w.scenario) else []
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand.pkl",
        summary=f"{resources_path}/demand/{{scenario}}_demand_integration_summary.csv"
    params:
        disaggregation_config=lambda w: get_disaggregation_config(w.scenario),
        flexibility_config=lambda w: get_flexibility_config(w.scenario),
        fes_scenario=lambda w: scenarios.get(w.scenario, {}).get("FES_scenario", "Holistic Transition"),
        modelled_year=lambda w: scenarios.get(w.scenario, {}).get("modelled_year", 2035)
    log:
        "logs/demand/finalize_demand_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/finalize_demand_integration.py"


rule build_fes_demand_components:
    """
    FES long-range demand components and consumer flexibility (plan Y6a), any pathway and year.

    Only for scenarios with demand.disaggregation.fes.enabled: true (HT50_* today). Builds,
    from the base network (ESPENI base scaled to ED1), the weather-year cutout and FES 2025
    (the figures below are HT 2050's):
      - heat pumps (Dem_BB004+005) on 2013 temperatures with hot water, a daily heating
        shape and temperature COPs; district heat (ED1) on daily HDH; data centres flat;
      - EVs (Dem_BB006+007) on a plug-in charging profile, with the flexible share solved
        so EV load at the ED1 peak hours is ED1's (1.4 + 7.6 GW);
      - consumer flexibility at FES scale (FLX1): EV smart charging, V2G (81.2 GW max,
        40.6 GW at peak), heat-pump and district-heat pre-heating, I&C DSR; DACCS (ES1);
      - bus allocation shares from the FES building blocks through the base GSP mapping.
    finalize_demand then carves them out of the base loads (scripts/demand/fes_demand.py).
    """
    input:
        base_network=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl",
        fes_data=get_fes_data_input,
        cutout=lambda w: f"{resources_path}/atlite/cutouts/uk-{scenarios[w.scenario].get('renewables_year', 2020)}.nc",
        targets=lambda w: fes_demand_targets_file(w.scenario),
        workbook=lambda w: fes_demand_workbook(w.scenario),
        logic=["scripts/demand/fes_demand.py", "scripts/demand/demand_carriers.py"],
    output:
        profiles=f"{resources_path}/demand/fes2050/{{scenario}}_profiles.csv",
        allocation=f"{resources_path}/demand/fes2050/{{scenario}}_allocation.csv",
        spec=f"{resources_path}/demand/fes2050/{{scenario}}_spec.json",
    params:
        fes_demand=lambda w: get_fes_demand_config(w.scenario),
        fes_scenario=lambda w: get_demand_fes_scenario(w.scenario),
        fes_year=lambda w: get_demand_fes_year(w.scenario),
        modelled_year=lambda w: scenarios[w.scenario]["modelled_year"],
        network_model=lambda w: scenarios[w.scenario]["network_model"],
    wildcard_constraints:
        scenario=FES_DEMAND_SCENARIO_REGEX
    log:
        "logs/demand/build_fes_demand_components_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/fes_demand.py"


# STAGE 3: DEMAND-SIDE FLEXIBILITY
# ══════════════════════════════════════════════════════════════════════════════
#
# Optional third stage: Model demand-side flexibility resources.
#
# Flexibility Types:
#   - Heat Pump Flexibility: TANK (hot water storage) or COSY (thermal inertia)
#   - EV Flexibility: GO (night window), INT (smart charging), V2G (bidirectional)
#   - Event Response: Saving Sessions style demand reduction
#
# Configuration (in the scenario files or config/defaults/base/demand.yaml):
#   demand.flexibility:
#     enabled: true/false  # Master switch
#     heat_pumps:
#       enabled: true/false
#       mode: "TANK" / "COSY"
#     electric_vehicles:
#       enabled: true/false
#       tariff: "GO" / "INT" / "V2G"
#     event_response:
#       enabled: true/false
#       mode: "regular" / "winter" / "both"
#
# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 Pre-requisites: Heat Profiles and EV Availability
# ──────────────────────────────────────────────────────────────────────────────

rule build_heat_profiles_atlite:
    """
    Generate heat demand profiles and COPs using Atlite weather data.

    This rule uses Atlite cutouts (weather data) to generate:
    - Spatially-resolved heat demand profiles using degree-day method
    - Temperature-dependent ASHP COPs
    - Ground-temperature dependent GSHP COPs

    Heat Demand Method:
    - Uses heating degree hours (HDH) with reference temperature
    - HDH = max(0, T_ref - T_ambient) for each hour
    - Profiles normalized and scaled to annual heat demand

    COP Calculations:
    - ASHP: COP = 6.81 - 0.121*dT + 0.000630*dT² (Staffell et al.)
    - GSHP: Uses stable ground temperature (~10-12°C)

    Inputs:
      - cutout: Atlite cutout with temperature data
      - network: Network for bus locations

    Outputs:
      - heat_demand: Heat demand profiles by bus (NetCDF)
      - cop_ashp: ASHP COP profiles by bus (NetCDF)
      - cop_gshp: GSHP COP profiles by bus (NetCDF)

    Performance: ~30-60 seconds depending on cutout size
    """
    input:
        cutout=lambda wildcards: (
            f"{resources_path}/atlite/cutouts/uk-{scenarios[wildcards.scenario].get('renewables_year', 2020)}.nc"
        ),
        network=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl"
    output:
        heat_demand=f"{resources_path}/demand/heat_demand_{{scenario}}.nc",
        cop_ashp=f"{resources_path}/demand/cop_ashp_{{scenario}}.nc",
        cop_gshp=f"{resources_path}/demand/cop_gshp_{{scenario}}.nc"
    params:
        modelled_year=lambda wildcards: scenarios[wildcards.scenario]["modelled_year"],
        heat_config=lambda wildcards: get_flexibility_config(wildcards.scenario).get("heat_pumps", {}),
        reference_temp=15.5,
        indoor_temp=20.0,
        ground_temp=11.0
    message:
        "Building Atlite heat profiles for {wildcards.scenario}"
    benchmark:
        "benchmarks/demand/build_heat_profiles_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/demand/build_heat_profiles_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/demand/build_heat_profiles.py"


rule build_ev_availability:
    """
    Build EV availability and DSM profiles from GB traffic patterns.

    Creates time-series profiles for:
    - Charging availability (when vehicles are plugged in)
    - DSM minimum SOC requirements (departure readiness)

    Methodology:
    - EV availability is inversely related to traffic flow
    - High traffic = cars on road = low charging availability
    - Uses GB-specific weekday/weekend traffic patterns from DfT

    Formula (from PyPSA-FES):
      avail = avail_max - (avail_max - avail_mean) *
              (traffic - traffic_min) / (traffic_mean - traffic_min)

    DSM Profile:
    - Enforces minimum SOC at departure times (default 7am)
    - Ensures EVs meet daily driving needs

    Inputs:
      - network: Network for bus locations and timestamps
      - dft_traffic (optional): DfT traffic count data

    Outputs:
      - availability_profile: Hourly EV plugged-in availability (0-1)
      - dsm_profile: Minimum SOC requirements at departure times

    Performance: ~5-10 seconds
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl"
    output:
        availability_profile=f"{resources_path}/demand/ev_availability_{{scenario}}.csv",
        dsm_profile=f"{resources_path}/demand/ev_dsm_{{scenario}}.csv"
    params:
        modelled_year=lambda wildcards: scenarios[wildcards.scenario]["modelled_year"],
        timestep_minutes=lambda wildcards: scenarios[wildcards.scenario].get("timestep_minutes", 60),
        ev_config=lambda wildcards: get_flexibility_config(wildcards.scenario).get("electric_vehicles", {}),
        avail_max=lambda wildcards: get_flexibility_config(wildcards.scenario).get("electric_vehicles", {}).get("availability", {}).get("avail_max", 0.95),
        avail_mean=lambda wildcards: get_flexibility_config(wildcards.scenario).get("electric_vehicles", {}).get("availability", {}).get("avail_mean", 0.80)
    message:
        "Building EV availability profiles for {wildcards.scenario}"
    benchmark:
        "benchmarks/demand/build_ev_availability_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/demand/build_ev_availability_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/demand/build_ev_availability.py"


# ──────────────────────────────────────────────────────────────────────────────
# New Flexibility Workflow Helper Rules
# ──────────────────────────────────────────────────────────────────────────────

rule build_flexibility_prerequisites:
    """
    Build all prerequisite data for flexibility integration.

    This rule builds the heat profiles and EV availability data needed for
    the flexibility integration, regardless of whether flexibility is enabled.
    Useful for generating data for analysis without running the full integration.

    Usage:
        snakemake build_flexibility_prerequisites_{scenario} --cores 1
    """
    input:
        heat_demand=f"{resources_path}/demand/heat_demand_{{scenario}}.nc",
        cop_ashp=f"{resources_path}/demand/cop_ashp_{{scenario}}.nc",
        cop_gshp=f"{resources_path}/demand/cop_gshp_{{scenario}}.nc",
        ev_availability=f"{resources_path}/demand/ev_availability_{{scenario}}.csv",
        ev_dsm=f"{resources_path}/demand/ev_dsm_{{scenario}}.csv"
    output:
        marker=f"{resources_path}/demand/flexibility_prerequisites_{{scenario}}.txt"
    shell:
        "echo 'Flexibility prerequisites built for scenario: {wildcards.scenario}' > {output.marker}"
