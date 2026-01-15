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
#   - Disaggregation: scenarios_master.yaml (demand_disaggregation section)
#   - Flexibility: scenarios_master.yaml (flexibility section, if enabled)
#
# Dependencies:
#   - Base network topology (from network_build.smk)
#   - FES data (from FES.smk)
#   - Historical demand timeseries (data/demand/)
#
# ══════════════════════════════════════════════════════════════════════════════


     # CSV demand profile no longer produced; network NetCDF contains base demand.
     # The script `scripts/load.py` writes the network to `network_with_base_demand`.
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def scenario_is_historical(scenario_id):
    """Check if a scenario is historical (modelled_year <= 2024)."""
    scenario = scenarios.get(scenario_id, {})
    modelled_year = scenario.get("modelled_year", 2035)
    return modelled_year <= 2024


def get_fes_data_input(wildcards):
    """Return FES data path for future scenarios, or placeholder for historical."""
    scenario_id = wildcards.scenario
    if scenario_is_historical(scenario_id):
        # Historical scenarios don't need FES data
        # Return espeni file as placeholder (it's always required anyway)
        return "data/demand/espeni.csv"
    else:
        fes_year = scenarios[scenario_id].get("FES_year", 2024)
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
    return [c["name"] for c in components]


def get_component_config(scenario, component_name):
    """Get configuration for a specific component."""
    config = get_disaggregation_config(scenario)
    components = config.get("components", [])
    for comp in components:
        if comp["name"] == component_name:
            return comp
    return None


def get_final_demand_network(wildcards):
    """
    Return the appropriate final network file based on disaggregation config.
    
    - If disaggregation disabled → base_demand.nc (simple case)
    - If disaggregation enabled → final.nc (with all components integrated)
    
    This function is used by downstream rules (generators, storage, etc.) to
    automatically use the correct network file.
    """
    scenario = wildcards.scenario
    if is_disaggregation_enabled(scenario):
        return f"{resources_path}/network/{scenario}_final.nc"
    else:
        # Return the network file containing base demand (pickle for fast I/O).
        # CSV profiles are no longer saved; downstream rules should use the network pickle instead.
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
       
    3. TEMPORAL PROFILE (configurable via demand_timeseries):
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
    
    Transforms: {scenario}_network.nc → {scenario}_network_demand.pkl
    
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
      - If disaggregation DISABLED → This is the final demand
      - If disaggregation ENABLED → Components processed by Stage 2
    
    Performance:
      - ETYS: ~30-40s (GSP spatial mapping)
      - Reduced: ~5-8s (simple aggregation)
      - Zonal: ~3-5s (simple aggregation)
    
    See Also:
      - Stage 2: Demand disaggregation (heat pumps, EVs, etc.)
      - scripts/load.py for implementation details
    """
    input:
        fes_data=get_fes_data_input,
        espeni="data/demand/espeni.csv",
        egy="data/demand/egy_7649_mmc1.xlsx",
        base_network=f"{resources_path}/network/{{scenario}}_network.nc"
    output:
        network_with_base_demand=f"{resources_path}/network/{{scenario}}_network_demand.pkl"
    params:
        fes_year=lambda wildcards: scenarios[wildcards.scenario].get("FES_year"),
        fes_scenario=lambda wildcards: scenarios[wildcards.scenario].get("FES_scenario"),
        modelled_year=lambda wildcards: scenarios[wildcards.scenario]["modelled_year"],
        demand_year=lambda wildcards: scenarios[wildcards.scenario]["demand_year"],
        demand_timeseries=lambda wildcards: scenarios[wildcards.scenario].get("demand_timeseries", "ESPENI"),
        profile_year=lambda wildcards: scenarios[wildcards.scenario].get("profile_year", None),
        network_model=lambda wildcards: scenarios[wildcards.scenario]["network_model"],
        timestep_minutes=lambda wildcards: scenarios[wildcards.scenario].get("timestep_minutes", 60),
        scenario=lambda wildcards: wildcards.scenario,
        is_historical=lambda wildcards: scenario_is_historical(wildcards.scenario)
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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: DEMAND DISAGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Optional second stage: Disaggregate total demand into sector components.
#
# Architecture:
#   1. build_base_demand (above) → Creates total demand
#   2. disaggregate_* rules (below) → Process individual components in PARALLEL
#   3. integrate_disaggregated_loads → Combine all components into final network
#
# To Add a New Component:
#   1. Create scripts/demand_components/your_component.py
#   2. Add rule disaggregate_your_component (copy template below)
#   3. Update integrate rule inputs
#   4. Configure in scenarios_master.yaml
#
# ──────────────────────────────────────────────────────────────────────────────

rule disaggregate_heat_pumps:
    """
    Disaggregate heat pump demand from total electricity demand.
    
    Process:
      1. Load heat pump demand profile from source file (or generate synthetic)
      2. Scale to match configured fraction of total demand
      3. Allocate spatially across network buses using selected method
      4. Create separate timeseries for heat pump load
    
    Allocation Methods:
      - proportional: Distribute proportional to existing demand
      - uniform: Equal distribution across all buses
      - urban_weighted: Weighted towards high-demand (urban) areas
    
    Outputs:
      - profile: Hourly/half-hourly heat pump demand timeseries (GWh)
      - allocation: Spatial distribution across buses (GWh per bus per year)
    
    See Also:
      - scripts/demand_components/heat_pumps.py for implementation
    """
    input:
        # Use the network pickle that contains base demand (fast I/O)
        base_network_demand=f"{resources_path}/network/{{scenario}}_network_demand.pkl",
    output:
        profile=f"{resources_path}/demand/components/{{scenario}}_heat_pumps_profile.csv",
        allocation=f"{resources_path}/demand/components/{{scenario}}_heat_pumps_allocation.csv"
    params:
        component_config=lambda wildcards: get_component_config(wildcards.scenario, "heat_pumps"),
        scenario=lambda wildcards: wildcards.scenario
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
    
    Outputs:
      - profile: Hourly/half-hourly EV charging demand timeseries (GWh)
      - allocation: Spatial distribution across buses (GWh per bus per year)
    
    See Also:
      - scripts/demand_components/electric_vehicles.py for implementation
    """
    input:
        # Use the network pickle that contains base demand (fast I/O)
        base_network_demand=f"{resources_path}/network/{{scenario}}_network_demand.pkl",
    output:
        profile=f"{resources_path}/demand/components/{{scenario}}_ev_profile.csv",
        allocation=f"{resources_path}/demand/components/{{scenario}}_ev_allocation.csv"
    params:
        component_config=lambda wildcards: get_component_config(wildcards.scenario, "electric_vehicles"),
        scenario=lambda wildcards: wildcards.scenario
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


rule integrate_disaggregated_loads:
    """
    Integrate all disaggregated components back into the network.
    
    This is the FINAL STAGE of demand modeling for disaggregated scenarios.
    
    Process:
      1. Load base demand network
      2. Load all component profiles and allocations
      3. Adjust base demand to remove component totals (avoid double-counting)
      4. Add each component as separate Load in PyPSA network
      5. Attach component-specific timeseries
      6. Validate energy conservation (total = original base)
      7. Export final network with all load components
    
    Energy Balance:
      Adjusted Base = Original Base - Sum(Component Totals)
      Final Total = Adjusted Base + Sum(Components) = Original Base ✓
    
    Validation:
      - Energy conservation check (tolerance: 0.1 GWh)
      - Bus topology validation
      - Component fraction sanity checks
    
    Outputs:
      - final_network: Network with base + all disaggregated components (NetCDF)
      - component_summary: Summary table of all components (CSV)
    
    See Also:
      - scripts/demand_components/integrate.py for implementation
    """
    input:
        # Use the network pickle that contains base demand (fast I/O)
        base_network_demand=f"{resources_path}/network/{{scenario}}_network_demand.pkl",
        # Dynamic inputs based on configured components
        component_profiles=lambda wildcards: [
            f"{resources_path}/demand/components/{wildcards.scenario}_{comp}_profile.csv"
            for comp in get_component_names(wildcards.scenario)
        ],
        component_allocations=lambda wildcards: [
            f"{resources_path}/demand/components/{wildcards.scenario}_{comp}_allocation.csv"
            for comp in get_component_names(wildcards.scenario)
        ]
    output:
        final_network=f"{resources_path}/network/{{scenario}}_final.nc",
        component_summary=f"{resources_path}/demand/components/{{scenario}}_summary.csv"
    params:
        component_names=lambda wildcards: get_component_names(wildcards.scenario),
        scenario=lambda wildcards: wildcards.scenario
    message:
        "Integrating disaggregated components for {wildcards.scenario}"
    benchmark:
        "benchmarks/demand/integrate_components_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/demand/integrate_components_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/demand/integrate.py"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: DEMAND-SIDE FLEXIBILITY
# ══════════════════════════════════════════════════════════════════════════════
#
# Optional third stage: Model demand-side flexibility resources.
#
# Flexibility Types:
#   - EV Smart Charging (V1G): Load shifting within availability window
#   - Vehicle-to-Grid (V2G): Bidirectional battery storage
#   - Thermal Storage: Pre-heating/cooling during low-price periods
#   - Demand Response: Industrial/commercial load shedding/shifting
#
# Architecture:
#   Stage 3a: EV flexibility (V1G + V2G)
#   Stage 3b: Thermal storage
#   Stage 3c: Demand response
#   Stage 3d: Network integration
#
# ──────────────────────────────────────────────────────────────────────────────
# Stage 3a: Electric Vehicle Flexibility
# ──────────────────────────────────────────────────────────────────────────────

rule calculate_ev_fleet_projections:
    """
    Calculate EV fleet size and distribution projections.
    
    Uses FES data or external projections to estimate:
    - Total EV fleet size by year
    - Geographic distribution (by region or bus)
    - Vehicle types (BEV, PHEV, light/heavy duty)
    - Ownership patterns (private, fleet, commercial)
    
    Data Sources:
    - FES EV uptake scenarios
    - DfT vehicle statistics
    - SMMT registration data
    - Academic studies
    
    Outputs:
    - EV fleet projections by scenario and year
    - Regional distribution factors
    - Technology mix (BEV vs PHEV)
    
    Performance: ~3-5 seconds
    """
    input:
        fes_data=f"{resources_path}/FES/FES_{{fes_year}}_processed.csv"
    output:
        ev_projections=f"{resources_path}/flexibility/ev/fleet_projections_{{fes_year}}.csv",
        regional_distribution=f"{resources_path}/flexibility/ev/regional_distribution_{{fes_year}}.csv"
    params:
        base_year=2020,
        projection_years=[2025, 2030, 2035, 2040, 2045, 2050],
        vehicle_types=["BEV", "PHEV"]
    log:
        "logs/flexibility/ev_fleet_projections_{fes_year}.log"
    benchmark:
        "benchmarks/flexibility/ev_fleet_projections_{fes_year}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/flex/ev_fleet_projections.py"


rule model_ev_charging_patterns:
    """
    Model EV charging availability and demand patterns.
    
    Creates time-series profiles for:
    - Charging availability (when vehicles are plugged in)
    - Charging demand (uncontrolled baseline)
    - Smart charging potential (shiftable load)
    - V2G availability (when discharge is possible)
    
    Charging Locations:
    - Home (overnight, typically 18:00-08:00)
    - Work (daytime, typically 08:00-17:00)
    - Public/rapid (opportunistic)
    
    Charging Behavior Assumptions:
    - Home charging: 60-70% of total energy
    - Work charging: 20-30% of total energy
    - Public charging: 5-15% of total energy
    
    Smart Charging Constraints:
    - Must meet daily travel energy needs
    - Respect user preferences (minimum SOC requirements)
    - Consider grid constraints (local transformer capacity)
    
    Outputs:
    - Hourly charging availability profiles (0-1 for each location type)
    - Baseline demand profiles (uncontrolled charging)
    - Maximum shift potential (MW by time of day)
    
    Performance: ~10-15 seconds
    """
    input:
        ev_projections=f"{resources_path}/flexibility/ev/fleet_projections_{{fes_year}}.csv"
    output:
        charging_availability=f"{resources_path}/flexibility/ev/charging_availability_{{fes_year}}.csv",
        baseline_demand=f"{resources_path}/flexibility/ev/baseline_demand_{{fes_year}}.csv",
        smart_charging_potential=f"{resources_path}/flexibility/ev/smart_charging_potential_{{fes_year}}.csv"
    params:
        home_charging_share=0.65,
        work_charging_share=0.25,
        public_charging_share=0.10,
        home_plug_in_time="18:00",
        home_plug_out_time="08:00",
        work_plug_in_time="09:00",
        work_plug_out_time="17:00",
        home_charger_power=7.0,
        work_charger_power=11.0,
        rapid_charger_power=50.0,
        average_battery_capacity_kwh=60.0,
        average_daily_travel_kwh=10.0
    log:
        "logs/flexibility/ev_charging_patterns_{fes_year}.log"
    benchmark:
        "benchmarks/flexibility/ev_charging_patterns_{fes_year}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/flex/ev_charging_patterns.py"


rule configure_ev_flexibility_parameters:
    """
    Configure EV flexibility parameters for PyPSA integration.
    
    Determines:
    - V1G (smart charging) capacity and constraints
    - V2G (vehicle-to-grid) capacity and constraints
    - Battery degradation considerations
    - User participation rates
    
    V1G Capabilities:
    - Delayed/shifted charging (within availability window)
    - Reduced peak demand through smart scheduling
    - Grid service provision (frequency response)
    
    V2G Capabilities:
    - Bidirectional power flow
    - Peak shaving and arbitrage
    - Grid services (frequency, voltage support)
    - Backup power for critical loads
    
    Constraints:
    - Minimum SOC requirements (user comfort, range anxiety)
    - Maximum cycles per day (battery degradation)
    - Round-trip efficiency losses
    - V2G participation rates (likely lower than V1G)
    
    Economic Parameters:
    - V1G compensation (reduced tariff, time-of-use)
    - V2G compensation (£/kWh for discharge)
    - Battery degradation costs
    - Infrastructure costs (smart chargers, V2G-capable EVSEs)
    
    Outputs:
    - EV flexibility parameters by scenario
    - V1G and V2G capacity estimates
    - Cost-benefit analysis inputs
    
    Performance: ~5-8 seconds
    """
    input:
        fleet_projections=f"{resources_path}/flexibility/ev/fleet_projections_{{fes_year}}.csv",
        charging_availability=f"{resources_path}/flexibility/ev/charging_availability_{{fes_year}}.csv",
        smart_charging_potential=f"{resources_path}/flexibility/ev/smart_charging_potential_{{fes_year}}.csv"
    output:
        ev_flexibility_params=f"{resources_path}/flexibility/ev/flexibility_parameters_{{fes_year}}.csv",
        v1g_capacity=f"{resources_path}/flexibility/ev/v1g_capacity_{{fes_year}}.csv",
        v2g_capacity=f"{resources_path}/flexibility/ev/v2g_capacity_{{fes_year}}.csv"
    params:
        v1g_participation_rate=0.80,
        v2g_participation_rate=0.30,
        v2g_efficiency=0.90,
        max_cycles_per_day=1.5,
        min_soc=0.20,
        target_soc=0.80,
        cycle_degradation_cost_per_kwh=0.05,
        v1g_tariff_discount=0.10,
        v2g_discharge_payment=0.20
    log:
        "logs/flexibility/ev_flexibility_params_{fes_year}.log"
    benchmark:
        "benchmarks/flexibility/ev_flexibility_params_{fes_year}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/flex/ev_flexibility_params.py"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3b: Thermal Energy Storage Flexibility
# ──────────────────────────────────────────────────────────────────────────────

rule assess_thermal_storage_potential:
    """
    Assess thermal energy storage potential in buildings.
    
    Thermal storage types:
    1. Hot Water Tanks: Domestic hot water cylinders, commercial systems
    2. Building Thermal Mass: Space heating inertia, pre-heating
    3. Phase Change Materials: High energy density (future)
    
    Assessment Method:
    - Building stock characterization (residential, commercial, industrial)
    - Heating system prevalence (heat pumps, direct electric, gas boiler)
    - Storage capacity estimation (tank volume, thermal mass)
    - Operational constraints (comfort temperatures, legionella risk)
    
    Data Sources:
    - Energy Performance Certificate (EPC) database
    - Building Research Establishment models
    - Heating system surveys
    - Heat pump deployment projections (from FES)
    
    Outputs:
    - Thermal storage capacity by region (MWh_thermal)
    - Charging/discharging characteristics
    - Temperature constraints and comfort bounds
    
    Performance: ~10-15 seconds
    """
    input:
        fes_data=f"{resources_path}/FES/FES_{{fes_year}}_processed.csv",
        heat_pump_projections=f"{resources_path}/demand/components/heat_pump_projections_{{fes_year}}.csv"
    output:
        thermal_storage_potential=f"{resources_path}/flexibility/thermal/storage_potential_{{fes_year}}.csv",
        building_characteristics=f"{resources_path}/flexibility/thermal/building_characteristics_{{fes_year}}.csv"
    params:
        average_hw_tank_volume_liters=200,
        hw_tank_temp_range_celsius=[50, 65],
        hw_useful_energy_per_liter_kwh=0.058,
        average_building_thermal_mass_kwh_per_m2=0.5,
        temperature_relaxation_celsius=2.0,
        hp_cop_average=3.0,
        heat_pump_deployment_rate=0.05
    log:
        "logs/flexibility/thermal_storage_potential_{fes_year}.log"
    benchmark:
        "benchmarks/flexibility/thermal_storage_potential_{fes_year}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/flex/thermal_storage_potential.py"


rule configure_thermal_flexibility_parameters:
    """
    Configure thermal flexibility parameters for PyPSA integration.
    
    Thermal flexibility operates by:
    - Pre-heating during low-price periods (store energy as heat)
    - Reduced heating during high-price periods (release stored heat)
    - Maintaining comfort constraints throughout
    
    Key Parameters:
    - Storage capacity (MWh_thermal)
    - Charge/discharge rates (MW_thermal)
    - Conversion efficiency (electric → thermal)
    - Standing losses (heat dissipation)
    - Temperature constraints
    
    PyPSA Representation:
    - Link component: electric load → thermal storage
    - Store component: thermal energy reservoir
    - Load component: thermal demand (space + water heating)
    
    Economic Parameters:
    - No direct compensation (benefits from ToU tariffs)
    - Capital cost: smart controls, improved insulation
    - Value: reduced electricity costs, grid benefits
    
    Outputs:
    - Thermal flexibility parameters
    - Operational constraints
    - Integration specification for PyPSA
    
    Performance: ~5-8 seconds
    """
    input:
        storage_potential=f"{resources_path}/flexibility/thermal/storage_potential_{{fes_year}}.csv",
        building_characteristics=f"{resources_path}/flexibility/thermal/building_characteristics_{{fes_year}}.csv"
    output:
        thermal_flexibility_params=f"{resources_path}/flexibility/thermal/flexibility_parameters_{{fes_year}}.csv",
        pypsa_spec=f"{resources_path}/flexibility/thermal/pypsa_integration_spec_{{fes_year}}.json"
    params:
        electric_to_thermal_efficiency=0.95,
        standing_loss_per_hour=0.01,
        participation_rate=0.50,
        max_temperature_deviation_celsius=2.0,
        max_cycles_per_day=3
    log:
        "logs/flexibility/thermal_flexibility_params_{fes_year}.log"
    benchmark:
        "benchmarks/flexibility/thermal_flexibility_params_{fes_year}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/flex/thermal_flexibility_params.py"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3c: Demand Response Flexibility
# ──────────────────────────────────────────────────────────────────────────────

rule assess_demand_response_potential:
    """
    Assess demand response potential across sectors.
    
    Demand Response Types:
    1. Industrial Processes: Aluminum smelting, cement, steel, chemicals, data centers
    2. Commercial Buildings: HVAC, refrigeration, lighting, office equipment
    3. Residential: Wet appliances, pool pumps, smart home devices
    
    Assessment Method:
    - Sector-by-sector load analysis
    - Process flexibility characterization
    - Historical DR program performance
    - Economic viability assessment
    
    Data Sources:
    - BEIS industrial energy consumption
    - Commercial building surveys
    - Smart meter data (aggregated)
    - DR program reports (National Grid ESO)
    
    Outputs:
    - DR potential by sector and type (MW)
    - Response characteristics (speed, duration, frequency)
    - Participation likelihood and costs
    
    Performance: ~8-12 seconds
    """
    input:
        fes_data=f"{resources_path}/FES/FES_{{fes_year}}_processed.csv",
        industrial_load_profiles=f"{data_path}/demand/industrial_profiles.csv"
    output:
        dr_potential=f"{resources_path}/flexibility/dr/potential_{{fes_year}}.csv",
        sector_breakdown=f"{resources_path}/flexibility/dr/sector_breakdown_{{fes_year}}.csv"
    params:
        industrial_flexibility_share=0.15,
        industrial_response_time_minutes=30,
        commercial_flexibility_share=0.10,
        commercial_response_time_minutes=15,
        residential_flexibility_share=0.05,
        residential_response_time_minutes=60,
        participation_rates={
            "industrial": 0.40,
            "commercial": 0.25,
            "residential": 0.10
        }
    log:
        "logs/flexibility/dr_potential_{fes_year}.log"
    benchmark:
        "benchmarks/flexibility/dr_potential_{fes_year}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/flex/dr_potential.py"


rule configure_demand_response_parameters:
    """
    Configure demand response parameters for PyPSA integration.
    
    DR Representation in PyPSA:
    - Link component with time-varying efficiency
    - Load reduction = load shift to later time
    - Rebound effect (must pay back shifted load)
    
    Key Parameters:
    - Maximum shift capacity (MW)
    - Maximum shift duration (hours)
    - Rebound ratio (shifted MWh must be paid back)
    - Response time (minutes to activate)
    - Recovery time (time between activations)
    
    Economic Parameters:
    - Availability payment (£/MW/year for being enrolled)
    - Utilization payment (£/MWh when activated)
    - Opportunity cost (lost productivity, inconvenience)
    
    Constraints:
    - Maximum activations per day/week/month
    - Minimum notice period
    - Seasonal availability (e.g., heating season only)
    
    Outputs:
    - DR flexibility parameters
    - Activation cost curves
    - PyPSA integration specification
    
    Performance: ~5-7 seconds
    """
    input:
        dr_potential=f"{resources_path}/flexibility/dr/potential_{{fes_year}}.csv",
        sector_breakdown=f"{resources_path}/flexibility/dr/sector_breakdown_{{fes_year}}.csv"
    output:
        dr_flexibility_params=f"{resources_path}/flexibility/dr/flexibility_parameters_{{fes_year}}.csv",
        activation_cost_curve=f"{resources_path}/flexibility/dr/activation_costs_{{fes_year}}.csv",
        pypsa_spec=f"{resources_path}/flexibility/dr/pypsa_integration_spec_{{fes_year}}.json"
    params:
        max_shift_duration_hours=4.0,
        rebound_ratio=1.1,
        availability_payment_per_mw_year=10000,
        utilization_payment_per_mwh=50,
        max_activations_per_week=3,
        min_notice_hours=4
    log:
        "logs/flexibility/dr_flexibility_params_{fes_year}.log"
    benchmark:
        "benchmarks/flexibility/dr_flexibility_params_{fes_year}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/demand/flex/dr_flexibility_params.py"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3d: Flexibility Network Integration
# ──────────────────────────────────────────────────────────────────────────────

rule integrate_demand_flexibility_into_network:
    """
    Integrate all demand-side flexibility into PyPSA network.
    
    Integration Approach:
    1. Load EV, thermal, and DR flexibility parameters
    2. Map flexibility resources to network buses (by region)
    3. Add appropriate PyPSA components:
       - EV: Link (charge) + Store (battery) + Link (V2G discharge)
       - Thermal: Link (heat) + Store (thermal mass) + Load (demand)
       - DR: Link (shift) with time-coupled constraints
    4. Configure operational constraints
    5. Validate network consistency
    
    Component Mapping:
    - EV Smart Charging (V1G): Link + Store (electricity → battery storage)
    - EV Vehicle-to-Grid (V2G): Link (battery → electricity)
    - Thermal Storage: Link + Store (electricity → thermal store)
    - Demand Response: Link (load reduction/shift)
    
    Regional Aggregation:
    - Flexibility resources aggregated to bus level
    - Weighted by population, industrial activity, etc.
    
    Dependencies:
    - Network with generators and storage integrated
    - EV, thermal, and DR parameter databases
    
    Outputs:
    - Network with demand flexibility integrated
    - Flexibility integration summary
    - Component mapping report
    
    Performance: ~30-45 seconds
    
    Usage:
        snakemake resources/network/{scenario}_with_flexibility.nc --cores 1
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_with_storage.nc",
        ev_params=f"{resources_path}/flexibility/ev/flexibility_parameters_{{fes_year}}.csv",
        thermal_params=f"{resources_path}/flexibility/thermal/flexibility_parameters_{{fes_year}}.csv",
        dr_params=f"{resources_path}/flexibility/dr/flexibility_parameters_{{fes_year}}.csv"
    output:
        network=f"{resources_path}/network/{{scenario}}_with_flexibility.nc",
        integration_summary=f"{resources_path}/flexibility/{{scenario}}_integration_summary.csv",
        component_mapping=f"{resources_path}/flexibility/{{scenario}}_component_mapping.csv"
    params:
        distribution_method="population_weighted",
        enable_ev_v1g=True,
        enable_ev_v2g=True,
        enable_thermal_storage=True,
        enable_demand_response=True
    log:
        "logs/flexibility/integrate_demand_flexibility_{scenario}.log"
    benchmark:
        "benchmarks/flexibility/integrate_flexibility_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/add_demand_flexibility.py"


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE RULES
# ══════════════════════════════════════════════════════════════════════════════

rule build_ev_flexibility_database:
    """Complete EV flexibility data processing (Stage 3a)."""
    input:
        ev_params=f"{resources_path}/flexibility/ev/flexibility_parameters_{{fes_year}}.csv",
        v1g_capacity=f"{resources_path}/flexibility/ev/v1g_capacity_{{fes_year}}.csv",
        v2g_capacity=f"{resources_path}/flexibility/ev/v2g_capacity_{{fes_year}}.csv"
    output:
        marker=f"{resources_path}/flexibility/ev/database_complete_{{fes_year}}.txt"
    shell:
        "echo 'EV flexibility database complete!' > {output.marker}"


rule build_thermal_flexibility_database:
    """Complete thermal flexibility data processing (Stage 3b)."""
    input:
        thermal_params=f"{resources_path}/flexibility/thermal/flexibility_parameters_{{fes_year}}.csv",
        pypsa_spec=f"{resources_path}/flexibility/thermal/pypsa_integration_spec_{{fes_year}}.json"
    output:
        marker=f"{resources_path}/flexibility/thermal/database_complete_{{fes_year}}.txt"
    shell:
        "echo 'Thermal flexibility database complete!' > {output.marker}"


rule build_demand_response_database:
    """Complete demand response data processing (Stage 3c)."""
    input:
        dr_params=f"{resources_path}/flexibility/dr/flexibility_parameters_{{fes_year}}.csv",
        activation_costs=f"{resources_path}/flexibility/dr/activation_costs_{{fes_year}}.csv",
        pypsa_spec=f"{resources_path}/flexibility/dr/pypsa_integration_spec_{{fes_year}}.json"
    output:
        marker=f"{resources_path}/flexibility/dr/database_complete_{{fes_year}}.txt"
    shell:
        "echo 'Demand response database complete!' > {output.marker}"


rule build_demand_flexibility_database:
    """
    Complete demand-side flexibility data processing (all stages).
    
    Combines EV, thermal, and DR databases for comprehensive flexibility modeling.
    
    Usage:
        snakemake build_demand_flexibility_database_{fes_year} --cores 1
        
    Example:
        snakemake build_demand_flexibility_database_2024 --cores 1
    """
    input:
        ev_marker=f"{resources_path}/flexibility/ev/database_complete_{{fes_year}}.txt",
        thermal_marker=f"{resources_path}/flexibility/thermal/database_complete_{{fes_year}}.txt",
        dr_marker=f"{resources_path}/flexibility/dr/database_complete_{{fes_year}}.txt"
    output:
        marker=f"{resources_path}/flexibility/database_complete_{{fes_year}}.txt"
    shell:
        "echo 'Demand-side flexibility database complete!' > {output.marker}"


rule integrate_demand_flexibility:
    """
    Complete demand flexibility integration for a specific scenario.
    
    This is the recommended entry point for demand flexibility integration.
    
    Usage:
        snakemake integrate_demand_flexibility_{scenario} --cores 1
        
    Example:
        snakemake integrate_demand_flexibility_HT35_clustered_gsp --cores 1
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_with_flexibility.nc",
        integration_summary=f"{resources_path}/flexibility/{{scenario}}_integration_summary.csv"
    output:
        marker=f"{resources_path}/flexibility/{{scenario}}_integration_complete.txt"
    shell:
        "echo 'Demand flexibility integration complete for scenario: {wildcards.scenario}' > {output.marker}"
