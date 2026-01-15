"""
Energy Storage Integration Rules for PyPSA-GB

Simplified storage pipeline using REPD as the authoritative data source.

Pipeline Architecture:
======================
Stage 1: Extract storage from REPD (Renewable Energy Planning Database)
  - Battery Energy Storage Systems (BESS)
  - Pumped Hydro Storage
  - Other storage technologies (CAES, LAES, Flywheels)

Stage 2: Network Integration
  - Map storage sites to network buses
  - Add StorageUnit components to PyPSA network
  - Apply technology-specific parameters and constraints

Input Data:
===========
REPD Data (data/renewables/repd-q2-jul-2025.csv):
- Storage technologies with capacity (MW), coordinates, status
- Commissioning dates, ownership information

Usage Examples:
===============
# Full storage pipeline for a scenario
snakemake resources/network/{scenario}_network_demand_renewables_thermal_generators_storage.nc --cores 1

# Extract storage from REPD only
snakemake resources/storage/storage_from_repd.csv --cores 1
"""

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

resources_path = "resources"
data_path = "data"


# =============================================================================
# STAGE 1: EXTRACT STORAGE FROM REPD
# =============================================================================

rule extract_storage_from_repd:
    """
    Extract storage assets from REPD (Renewable Energy Planning Database).
    
    REPD is the authoritative source for GB storage assets, containing:
    - Battery Energy Storage Systems (BESS)
    - Pumped Hydro Storage
    - Compressed Air Energy Storage (CAES)
    - Liquid Air Energy Storage (LAES)
    - Flywheels and other mechanical storage
    
    Processing:
    - Filters for storage technology types
    - Converts OSGB36 coordinates to WGS84 lat/lon
    - Standardizes capacity units to MW
    - Extracts commissioning dates from status updates
    - Applies technology-specific parameter defaults
    
    Technology-Specific Defaults:
    - Battery: 90% round-trip efficiency, 1-4 hour duration
    - Pumped Hydro: 75-82% efficiency, 4-24 hour duration
    - CAES: 60-70% efficiency, 8+ hour duration
    - LAES: 50-70% efficiency, 8+ hour duration
    
    Parameters:
    - include_pipeline: Include non-operational projects (planning/construction)
    
    Outputs:
    - Storage parameters ready for PyPSA network integration
    
    Performance: ~5-10 seconds
    """
    input:
        repd=f"{data_path}/renewables/repd-q2-jul-2025.csv"
    output:
        storage_params=f"{resources_path}/storage/storage_parameters.csv"
    params:
        include_pipeline=False  # Set to True to include non-operational projects
    log:
        "logs/storage/extract_storage_from_repd.log"
    benchmark:
        "benchmarks/storage/extract_repd.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/storage/storage_from_repd.py"


# =============================================================================
# STAGE 2: NETWORK INTEGRATION
# =============================================================================

rule add_storage_to_network:
    """
    Add energy storage units to PyPSA network.
    
    Complete storage integration pipeline:
    1. Load storage parameters from REPD (authoritative source)
    2. Map each storage site to nearest network bus (geographic proximity)
    3. Add StorageUnit components to PyPSA network with:
       - Power capacity (p_nom in MW)
       - Energy capacity (max_hours × p_nom in MWh)
       - Technology-specific efficiency, standing loss, operational constraints
    4. Validate network consistency and capacity bounds
    5. Export integrated network to NetCDF
    
    Storage Technologies:
    - Battery Energy Storage Systems (BESS): 85-90% efficiency, 1-4 hour duration
    - Pumped Hydro Storage: 70-80% efficiency, 4-24 hour duration
    - Compressed Air Energy Storage (CAES): 60-70% efficiency, 8+ hour duration
    - Liquid Air Energy Storage (LAES): 50-65% efficiency, 8+ hour duration
    - Flywheels: 95%+ efficiency, short-duration (minutes)
    
    Network Model Handling:
    - ETYS: Individual storage sites mapped to nearest bus
    - Reduced: Storage aggregated by technology to regional buses
    - Zonal: Storage aggregated to zone buses with capacity preserved
    
    PyPSA StorageUnit Parameters:
    - p_nom (MW): Rated power capacity
    - max_hours (h): Energy/power ratio (determines e_nom = p_nom × max_hours)
    - efficiency_store: Charging efficiency (0-1)
    - efficiency_dispatch: Discharging efficiency (0-1)
    - standing_loss: Self-discharge per hour (0-1)
    - marginal_cost (£/MWh): Operating cost
    - cyclic_state_of_charge: True (start SOC = end SOC)
    
    Performance:
    - ETYS: ~40-60s (detailed network)
    - Reduced: ~8-12s (regional aggregation)
    - Zonal: ~3-6s (minimal sites)
    
    Transforms: 
    {scenario}_network_demand_renewables_thermal_generators_costs.nc 
    → {scenario}_network_demand_renewables_thermal_generators_storage.nc
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_costs.pkl",
        storage_data=f"{resources_path}/storage/storage_parameters.csv",
        fes_data=f"{resources_path}/FES/FES_2024_data.csv"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage.pkl"
    params:
        scenario=lambda wc: wc.scenario,
        network_model=lambda wc: scenarios[wc.scenario]["network_model"],
        modelled_year=lambda wc: scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year'),
        is_historical=lambda wc: (scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year', 9999)) <= 2024,
        fes_scenario=lambda wc: scenarios[wc.scenario].get('FES_scenario', None)
    message:
        "Adding storage units to network for {wildcards.scenario} (model: {params.network_model})"
    benchmark:
        "benchmarks/storage/add_storage_to_network_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/storage/add_storage_to_network_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/storage/add_storage.py"
