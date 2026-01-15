"""
Interconnectors workflow for PyPSA-GB with European generation mix integration.

This workflow processes interconnector data from DUKES and NESO sources,
enriches it with locations and capacities, and integrates it into the PyPSA network.
Additionally, downloads European electricity generation mix data for border pricing.
"""

resources_path = "resources"
data_path = "data"

# =============================================================================
# European Generation Mix Integration
# =============================================================================

rule download_european_generation_mix:
    """Download European electricity generation mix data from NESO FES API."""
    output:
        generation_mix=f"{resources_path}/interconnectors/european_generation_mix_{{fes_year}}.csv",
        metadata=f"{resources_path}/interconnectors/european_generation_mix_{{fes_year}}_metadata.json"
    params:
        api_endpoint="https://api.neso.energy/api/3/action/datapackage_show",
        dataset_id="fes-european-electricity-supply-data-table-es2"
    log:
        "logs/download_european_generation_mix_{fes_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/download_european_generation.py"

rule process_european_generation_mix:
    """Process European generation mix into marginal costs and price differentials."""
    input:
        generation_mix=f"{resources_path}/interconnectors/european_generation_mix_{{fes_year}}.csv",
        fuel_prices=f"{resources_path}/marginal_costs/fuel_prices_{{fes_year}}.csv",
        carbon_prices=f"{resources_path}/marginal_costs/carbon_prices_{{fes_year}}.csv"
    output:
        marginal_costs=f"{resources_path}/interconnectors/european_marginal_costs_{{fes_year}}.csv",
        price_differentials=f"{resources_path}/interconnectors/price_differentials_{{fes_year}}.csv"
    log:
        "logs/process_european_generation_mix_{fes_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/process_european_generation.py"

# =============================================================================
# Historical Interconnector Flows
# =============================================================================

rule extract_historical_interconnector_flows:
    """Extract actual historical interconnector flows from ESPENI data."""
    input:
        espeni=data_path + "/demand/espeni.csv"
    output:
        flows=resources_path + "/interconnectors/historical_flows_{year}.csv",
        metadata=resources_path + "/interconnectors/historical_flows_{year}_metadata.json"
    params:
        year=lambda wildcards: wildcards.year
    log:
        "logs/extract_historical_interconnector_flows_{year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/extract_historical_flows.py"

# =============================================================================
# Core Interconnector Data Processing
# =============================================================================

rule ingest_dukes:
    """Ingest DUKES interconnector data."""
    input:
        data_path + "/interconnectors/DUKES_5.13_2025.csv"
    output:
        resources_path + "/interconnectors/interconnectors_dukes.csv"
    log:
        "logs/ingest_dukes_interconnectors.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/ingest_dukes.py"

rule ingest_neso_register:
    """Ingest NESO Interconnector Register data."""
    input:
        data_path + "/interconnectors/NESO_interconnector_register.csv"
    output:
        resources_path + "/interconnectors/interconnectors_neso.csv"
    log:
        "logs/ingest_neso_register.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/ingest_neso_register.py"

rule combine_datasets:
    """Combine DUKES and NESO register datasets."""
    input:
        dukes=resources_path + "/interconnectors/interconnectors_dukes.csv",
        neso_register=resources_path + "/interconnectors/interconnectors_neso.csv"
    output:
        resources_path + "/interconnectors/interconnector_data_combined.csv"
    log:
        "logs/combine_interconnector_datasets.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/combine_datasets.py"

rule enrich_locations:
    """Enrich interconnector data with location information."""
    input:
        resources_path + "/interconnectors/interconnector_data_combined.csv",
        data_path + "/interconnectors/NESO_interconnector_register.csv"
    output:
        resources_path + "/interconnectors/interconnector_data_with_locations.csv"
    log:
        "logs/enrich_interconnector_locations.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/enrich_locations.py"

rule clean:
    """Clean and standardize interconnector data."""
    input:
        resources_path + "/interconnectors/interconnector_data_with_locations.csv"
    output:
        resources_path + "/interconnectors/interconnectors_clean.csv"
    params:
        overrides_file="data/interconnectors/overrides.csv"
    log:
        "logs/clean_interconnector_data.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/clean_interconnectors.py"

rule pipeline_placeholder:
    """Create placeholder for future pipeline data."""
    output:
        resources_path + "/interconnectors/pipeline_placeholder.csv"
    log:
        "logs/pipeline_placeholder.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/pipeline_placeholder.py"

def get_network_for_model(wildcards):
    """Get the base network file for a given network_model by finding a matching scenario."""
    network_model = wildcards.network_model
    
    # Get run_ids from config (same logic as main Snakefile)
    if config.get("scenario"):
        run_ids = [config["scenario"]]
    else:
        run_ids = config.get("run_scenarios", [])
    
    for rid in run_ids:
        if scenarios.get(rid, {}).get("network_model") == network_model:
            return f"{resources_path}/network/{rid}_network.nc"
    
    # If no matching scenario found, raise informative error
    raise ValueError(
        f"No scenario found with network_model='{network_model}'. "
        f"Available scenarios: {run_ids}, "
        f"network_models: {[scenarios.get(r, {}).get('network_model') for r in run_ids]}"
    )

rule map_to_buses:
    """Map interconnectors to network buses using any scenario with the target network_model."""
    input:
        interconnectors=resources_path + "/interconnectors/interconnectors_clean.csv",
        network=get_network_for_model
    output:
        resources_path + "/interconnectors/interconnectors_mapped_{network_model}.csv"
    wildcard_constraints:
        network_model="[^/]+"
    log:
        "logs/map_interconnectors_to_buses_{network_model}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/map_to_buses.py"

rule availability:
    """Generate interconnector availability time series."""
    input:
        resources_path + "/interconnectors/interconnectors_clean.csv"
    output:
        resources_path + "/interconnectors/interconnector_availability.csv"
    log:
        "logs/interconnector_availability.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/availability.py"

def get_interconnector_inputs(wildcards):
    """
    Get appropriate interconnector inputs based on scenario type.
    
    For historical scenarios: Use actual ESPENI flows
    For future scenarios: Use availability profiles + price differentials
    """
    import sys
    from pathlib import Path
    
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.utilities.scenario_detection import is_historical_scenario
    
    scenario_config = scenarios[wildcards.scenario]
    modelled_year = scenario_config.get('modelled_year') or scenario_config.get('year')
    
    # Base inputs always required
    inputs = {
        'interconnectors': f"{resources_path}/interconnectors/interconnectors_mapped_{scenario_config['network_model']}.csv",
    }
    
    if is_historical_scenario(modelled_year):
        # Historical: use actual flows from ESPENI
        inputs['historical_flows'] = f"{resources_path}/interconnectors/historical_flows_{modelled_year}.csv"
    else:
        # Future: use availability + price differentials
        inputs['availability'] = f"{resources_path}/interconnectors/interconnector_availability.csv"
        inputs['price_differentials'] = f"{resources_path}/interconnectors/price_differentials_{scenario_config.get('FES_year', 2024)}.csv"
    
    return inputs

rule add_interconnectors_to_network:
    """
    Add cross-border interconnectors to network with European supply modeling.
    
    Takes the network with base demand, renewables, thermal generators, and storage,
    then integrates international electricity interconnector links. Interconnectors
    enable electricity exchange between Great Britain and neighboring countries.
    
    Transforms: {scenario}_network_demand_renewables_thermal_storage.nc → {scenario}_network_demand_renewables_thermal_storage_interconnectors.nc
    
    Architecture:
      - External buses: Connection points to European countries (France, Belgium, etc.)
      - External generators: Large generators (100 GW) on external buses representing
        European electricity supply with marginal costs from European generation mix data
      - DC links: Connect GB buses to external buses with near-zero marginal cost
        (only transmission efficiency losses ~2-5%)
      - This ensures European imports are properly costed, not treated as infinite free power
    
    Interconnector Coverage:
      - France: Interconnecteur France-Angleterre (IFA, IFA2, ElecLink)
      - Netherlands: BritNed
      - Belgium: Nemolink
      - Norway: NSL (Norwegian connection)
      - Denmark: Viking Link
      - Ireland: EuroConnector, EastWest, Moyle
    
    Processing:
      1. Load network with demand, renewables, thermal generators, storage
      2. Load interconnector mapping data (locations, capacities)
      3. Create external buses at international connection points
      4. [Future scenarios] Add large generators on external buses with European marginal costs
      5. Add Link objects for bidirectional power flows with:
         - name, bus0 (GB side), bus1 (foreign side)
         - p_nom (transmission capacity)
         - p_min_pu=-1.0 (bidirectional)
         - efficiency (transmission losses ~95-98%)
         - marginal_cost=0.0 (economics handled by external generators)
      6. [Future] Apply availability profiles (time-varying capacity if needed)
      7. [Historical] Apply actual ESPENI flows as fixed constraints (p_set)
      8. Validate network (capacity, bidirectionality, external generators)
      9. Export to NetCDF
    
    Inputs:
      - network: Network with all domestic components (NetCDF)
      - interconnector_data: Mapped interconnector links with capacities (CSV)
      - [Historical] historical_flows: Actual interconnector flows from ESPENI (CSV)
      - [Future] availability: Interconnector availability profiles (CSV)
      - [Future] price_differentials: European marginal costs by country (CSV)
    
    Output:
      - network: Fully assembled network ready for clustering or solving
    
    Network Model Handling:
      - ETYS: Interconnectors mapped to transmission nodes (typically southern UK)
      - Reduced: Interconnectors at representative regional buses
      - Zonal: Interconnectors at zone buses closest to connection points
    
    Capacity Assumptions (GW):
      - France (IFA/IFA2/ElecLink): 2-5 GW total
      - Netherlands (BritNed): 1 GW
      - Belgium (Nemolink): 1 GW
      - Norway (NSL): 0.7-1.4 GW
      - Denmark (Viking Link): 1.4 GW
      - Note: Capacities increase over time with new interconnectors
    
    Transmission Efficiency:
      - HVAC submarine cables: 95-98%
      - HVDC submarine cables: 96-99%
      - Overhead lines: 99%+
      - Typical loss: 0.5-2% per interconnector
    
    European Supply Modeling:
      - External generators have 100 GW capacity (non-binding constraint)
      - Marginal costs from European generation mix analysis:
        * Calculated from European renewable/thermal generation shares
        * Based on fuel prices, carbon costs, and merit order dispatch
        * Typical range: £30-70/MWh depending on country and scenario
      - Allows optimizer to balance GB generation vs. imports economically
    
    Historical vs Future Data:
      - Historical (≤2024): ESPENI actual flows stored for post-solve validation/comparison
        * Flows are NOT constrained (p_set) - this causes infeasibility
        * Instead, stored in network.historical_interconnector_flows for validation
        * Used as initial guess (p0) to aid solver convergence
        * Post-solve analysis compares optimized vs actual flows
      - Future (>2024): Optimizable flows with European marginal costs + availability profiles
      - Both scenarios allow full optimization, historical validates against market data
    
    Performance:
      - ETYS: ~15-25s (many domestic nodes to connect to)
      - Reduced: ~5-10s (fewer regional nodes)
      - Zonal: ~2-5s (minimal zone nodes)
    
    See Also:
      - scripts/interconnectors/add_to_network.py - Integration implementation
      - scripts/interconnectors/process_european_generation.py - European cost calculation
      - Fully assembled network ready for: network_clustering.smk or solve.smk
      - docs/INTERCONNECTOR_MODELING.md - Interconnector details
    """
    input:
        unpack(get_interconnector_inputs),
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    params:
        scenario=lambda wc: wc.scenario,
        network_model=lambda wc: scenarios[wc.scenario]["network_model"],
        modelled_year=lambda wc: scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year'),
        is_historical=lambda wc: (scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year', 9999)) <= 2024,
        fes_pathway=lambda wc: scenarios[wc.scenario].get('FES_scenario', None)
    message:
        "Adding interconnectors to network for {wildcards.scenario} (model: {params.network_model})"
    benchmark:
        "benchmarks/interconnectors/add_interconnectors_to_network_{scenario}.txt"
    log:
        "logs/interconnectors/add_interconnectors_to_network_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/add_to_network.py"
