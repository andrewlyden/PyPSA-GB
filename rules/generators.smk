"""
Generator Integration Rules for PyPSA-GB

Pipeline Map (concise):
- Inputs: DUKES (historical thermal), REPD sites (renewables + dispatchable thermal), FES (future projections)
- Scenario Detection: Automatically routes to DUKES for historical years, FES for future years
- Integrate: Add generators to network (components + timeseries aligned to snapshots)
- Export: Write generator CSVs and full network CSV tables

Authoritative outputs:
- resources/network/{network_model}_base_demand_generators.nc
- resources/generators/{network_model}_generators_full.csv (+ summaries)
- resources/network/csv/{network_model}_base_demand_generators/ (exported tables)

Notes:
- Time series can be any timestep; scripts/add_generators.py aligns to network.snapshots.
- Rules tagged LEGACY are supported but not part of the canonical path.
- TEC processing rules are preserved as LEGACY for reference but not used in main workflow.

Data update points:
1) resources/renewable/*_sites.csv
2) resources/renewable/profiles/*.csv
3) data/generators/generator_data_by_fuel.csv
"""

# Import scenario detection utility
from scripts.utilities.scenario_detection import is_historical_scenario

# === Extract lists of key inputs ===
def _extract_from_scenarios(key, default=None):
    """Extract values from active scenarios, skipping those without the key."""
    values = set()
    for rid in run_ids:
        if rid in scenarios and key in scenarios[rid]:
            values.add(scenarios[rid][key])
    return sorted(values) if values else ([default] if default is not None else [])

# Define variables for repeated paths (only extract keys that exist in all scenarios)
fes_years = _extract_from_scenarios("FES_year")  # Empty for historical-only runs
renewables_years = _extract_from_scenarios("renewables_year", 2020)
resources_path = "resources"
data_path = "data"

# ══════════════════════════════════════════════════════════════════════════════
# RULE ORDER: Prefer new REPD-only rule over legacy TEC-based rule
# ══════════════════════════════════════════════════════════════════════════════
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
    from scripts.utilities.scenario_detection import is_historical_scenario
    
    scenario_id = wildcards.scenario
    scenario_config = scenarios[scenario_id]
    modelled_year = scenario_config['modelled_year']
    
    inputs = {}
    
    if is_historical_scenario(scenario_config):
        # Historical: Use DUKES ONLY (no FES fallback)
        inputs['dukes_data'] = f"{resources_path}/generators/DUKES/DUKES_{modelled_year}_generators.csv"
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
    from scripts.utilities.scenario_detection import is_historical_scenario
    
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
# DUKES HISTORICAL DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
# Extract historical generator capacity data from DUKES 5.11 (2004-2024)
# Used as primary data source for historical scenarios (thermal generators)
# ══════════════════════════════════════════════════════════════════════════════

# Rule to extract DUKES generator data for a specific year
rule extract_dukes_generator_data:
    """
    Extract historical generator capacity data from DUKES 5.11 for a specific year.
    
    DUKES (Digest of UK Energy Statistics) provides authoritative historical data
    on UK energy generation from 2004-2024. This rule extracts year-specific
    capacity data from the DUKES_5.11_2025.xlsx file and converts it to a standardized
    CSV format compatible with the FES data structure.
    
    Data Source Priority for Historical Scenarios:
      1. DUKES (this rule) - Conventional thermal generators (coal, gas, nuclear)
      2. REPD - Renewable generators (wind, solar, hydro, marine)
      3. FES (fallback) - Storage, small distributed generation, recent additions
    
    Input:
      - data/generators/DUKES_5.11_2025.xlsx (multi-worksheet Excel file)
      - Year-specific worksheet (e.g., "DUKES 2015")
    
    Output:
      - resources/generators/DUKES/DUKES_{year}_generators.csv
      - Standardized format: [Company Name, Station Name, Fuel, Technology, Capacity, ...]
    
    Parameters:
      - dukes_year: Year to extract (2004-2024, from wildcards)
    
    Process:
      1. Read DUKES Excel file worksheet for specified year (skip 5 header rows)
      2. Extract generator records (power stations)
      3. Standardize fuel type and technology codes
      4. Convert to consistent format matching FES structure
      5. Add data_source='DUKES' column for tracking
    
    Usage:
      snakemake resources/generators/DUKES/DUKES_2020_generators.csv
    
    Notes:
      - DUKES provides more accurate historical data than FES retrospective views
      - Each year has a separate worksheet in DUKES_5.11_2025.xlsx
      - Output format designed for seamless merging with REPD and FES data
      - Includes only operational generators for the specified year
    """
    input:
        dukes_file=f"{data_path}/generators/DUKES_5.11_2025.xlsx"
    output:
        dukes_generators=f"{resources_path}/generators/DUKES/DUKES_{{year}}_generators.csv"
    params:
        dukes_year=lambda wildcards: int(wildcards.year)
    wildcard_constraints:
        year="20(0[4-9]|1[0-9]|2[0-4])"  # Match 2004-2024
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/extract_dukes_generator_data_{year}.log"
    benchmark:
        "benchmarks/DUKES/extract_dukes_{year}.txt"
    script:
        "../scripts/generators/DUKES_generator_data.py"


# Aggregate rule to extract DUKES data for all historical years
rule extract_all_dukes_data:
    """
    Extract DUKES generator data for all historical years (2004-2024).
    
    This aggregate rule triggers extraction of DUKES data for all years
    in the historical range. Useful for pre-populating the DUKES data cache.
    
    Usage:
      snakemake extract_all_dukes_data --cores 4
    
    Output:
      21 DUKES CSV files (one per year from 2004-2024)
    """
    input:
        expand(f"{resources_path}/generators/DUKES/DUKES_{{year}}_generators.csv", 
               year=range(2004, 2025))
    output:
        summary=f"{resources_path}/generators/DUKES/extraction_complete.txt"
    shell:
        """
        echo "DUKES historical generator data extracted successfully!" > {output.summary}
        echo "Years covered: 2004-2024" >> {output.summary}
        echo "Data source: DUKES 5.11 (Digest of UK Energy Statistics)" >> {output.summary}
        """


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY: TEC AND REPD PROCESSING
# These rules are preserved for reference but not used in the main workflow.
# The main workflow now uses DUKES (historical) + REPD + FES (future) directly.
# ══════════════════════════════════════════════════════════════════════════════

# LEGACY: Deduplicate TEC and REPD to remove renewables from TEC register
rule deduplicate_tec_repd:
    """
    Remove renewable technologies from TEC register to avoid duplication with REPD.
    
    This rule filters out renewable plant types from the TEC register, keeping only
    conventional thermal plants that are not covered by REPD (which provides better
    renewable generation profiles and site data).
    """
    input:
        tec_file="data/generators/tec-register-02-september-2025.csv"
    output:
        matches="resources/generators/tec_repd_matches.csv",
        conventional_tec="resources/generators/tec_conventional_only.csv",
        citations="resources/generators/references_used_dedup.json"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/deduplicate_tec_repd.log"
    script:
        "../scripts/generators/deduplicate_tec_repd.py"


# Process TEC generators with enhanced location mapping and technology categorization
rule process_tec_generators:
    """
    Process TEC register generators with comprehensive enhancement:
    
    1. Load and filter TEC register for built dispatchable projects only
    2. Apply advanced technology categorization (thermal, storage, hybrid)
    3. Enhanced location mapping using multiple data sources:
       - Network bus coordinates (if available - connection site matching)
       - DUKES power station database (authoritative coordinates)
       - Power stations location fallback data
       - Optional automated geocoding for unmapped sites
    4. Generate comprehensive processed TEC dataframe for workflow use
    
    This rule creates the authoritative processed TEC register that's used
    throughout the PyPSA-GB workflow, eliminating the need for multiple
    TEC processing steps in downstream rules.
    
    Note: network_file input is optional - will use fallback data sources if not available.
    """
    input:
        tec_conventional="resources/generators/tec_conventional_only.csv",
        power_stations_file="data/generators/power_stations_locations.csv",
        dukes_file="data/generators/dukes_power_station_coordinates.csv"
    output:
        processed_tec="resources/generators/tec_processed_complete.csv",
        location_sources_report="resources/generators/tec_location_mapping_sources.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/process_tec_generators.log"
    script:
        "../scripts/generators/process_tec_generators.py"


rule enhance_tec_locations_with_geocoding:
    """
    Enhance TEC location mapping with geocoding backup for unmapped sites.
    
    This rule provides an additional layer of location mapping by:
    1. Loading processed TEC data with initial location mapping
    2. Identifying sites without coordinates (location_source = 'not_found')
    3. Using Nominatim geocoding API to find coordinates for unmapped sites
    4. Rate limiting to respect API usage guidelines
    5. Generating comprehensive geocoding reports
    
    The enhanced location mapping improves coordinate coverage for better
    generator placement accuracy in the PyPSA network.
    """
    input:
        tec_processed="resources/generators/tec_processed_complete.csv"
    output:
        tec_enhanced="resources/generators/tec_enhanced_with_geocoding.csv",
        geocoding_report="resources/generators/geocoding_backup_report.csv"
    params:
        max_geocoding_attempts=50  # Rate limiting - adjust as needed
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/enhance_tec_locations_with_geocoding.log"
    script:
        "../scripts/generators/enhance_tec_locations_with_geocoding.py"


# Map dispatchable generator locations using REPD coordinates, network buses, and fallback data
rule map_dispatchable_generator_locations:
    """
    Create comprehensive dispatchable generator database with geographic locations.
    
    This rule combines TEC register and REPD data, mapping generator locations using:
    1. REPD X/Y coordinates (direct from dataset)
    2. TEC connection sites mapped to ETYS network bus coordinates
    3. Power stations location database as fallback
    
    Outputs a single comprehensive CSV with all operational dispatchable generators
    and their verified geographic coordinates, eliminating duplicate site storage.
    """
    input:
        tec_conventional="data/generators/tec-register-02-september-2025.csv",
        repd_file="data/renewables/repd-q2-jul-2025.csv",
        network_file="resources/network/ETYS_base.nc",
        power_stations_file="data/generators/power_stations_locations.csv"
    output:
        generators_with_locations="resources/generators/dispatchable_generators_with_locations.csv",
        location_report="resources/generators/location_mapping_report.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/map_dispatchable_generator_locations.log"
    script:
        "../scripts/generators/map_dispatchable_generator_locations.py"


# Prepare dispatchable generator site data from TEC + REPD
rule prepare_dispatchable_generator_sites:
    """
    Create comprehensive site CSV files for all dispatchable generators in GB.
    
    This rule combines TEC register (conventional thermal plants) with REPD 
    (dispatchable renewables/storage) to create technology-specific site files
    similar to the renewable site structure.
    
    Dispatchable technologies include:
    - Thermal: CCGT, OCGT, Nuclear, Coal, CHP, Gas Reciprocating
    - Storage: Battery, Pumped Hydro, Hydrogen
    - Biomass/Waste: Biomass, EfW, Biogas, Landfill Gas
    - Large Hydro: Reservoir-based hydroelectric plants
    """
    input:
        tec_processed="resources/generators/tec_enhanced_with_geocoding.csv",
        repd_file="data/renewables/repd-q2-jul-2025.csv"
    output:
        # Core thermal generators (note: ccgt/ocgt created dynamically based on TEC data availability)
        nuclear_sites="resources/generators/sites/nuclear_sites.csv",
        # Storage systems
        battery_sites="resources/generators/sites/battery_sites.csv",
        pumped_hydro_sites="resources/generators/sites/pumped_hydro_sites.csv",
        hydrogen_storage_sites="resources/generators/sites/hydrogen_storage_sites.csv",
        # Biomass and waste (dispatchable renewable thermal from REPD)
        biomass_sites="resources/generators/sites/biomass_sites.csv",
        waste_to_energy_sites="resources/generators/sites/waste_to_energy_sites.csv",
        biogas_sites="resources/generators/sites/biogas_sites.csv",
        landfill_gas_sites="resources/generators/sites/landfill_gas_sites.csv",
        sewage_gas_sites="resources/generators/sites/sewage_gas_sites.csv",
        advanced_biofuel_sites="resources/generators/sites/advanced_biofuel_sites.csv",
        # Geothermal (baseload renewable thermal from REPD)
        geothermal_sites="resources/generators/sites/geothermal_sites.csv",
        # Large hydro
        large_hydro_sites="resources/generators/sites/large_hydro_sites.csv",
        # Summary report
        summary_report="resources/generators/dispatchable_sites_summary.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/prepare_dispatchable_generator_sites.log"
    script:
        "../scripts/generators/prepare_dispatchable_generator_sites.py"


# Analyze unmapped generators and create comprehensive statistics
rule analyze_unmapped_generators:
    """
    Analyze generators still missing location coordinates after all mapping attempts.
    
    This rule creates comprehensive statistics about unmapped generators including:
    - Technology breakdown and capacity analysis
    - Priority ranking for manual lookup
    - Recommendations for filling remaining gaps
    - Detailed CSV export for further analysis
    """
    input:
        generators_with_wikipedia_locations="resources/generators/dispatchable_generators_with_wikipedia_locations.csv"
    output:
        unmapped_generators_csv="resources/generators/unmapped_generators.csv",
        unmapped_analysis_report="resources/generators/unmapped_generators_analysis.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/analyze_unmapped_generators.log"
    script:
        "../scripts/generators/analyze_unmapped_generators.py"


# OPTIONAL ENRICHMENT: Enhance generator locations with Wikipedia coordinate data
rule enhance_locations_with_wikipedia:
    """
    Enhance generator location database with Wikipedia power station coordinates.
    
    This rule scrapes Wikipedia's comprehensive lists of UK power stations to extract
    geographic coordinates for generators missing location data. It improves the
    location mapping success rate by matching Wikipedia data to unlocated generators.
    """
    input:
        generators_with_locations="resources/generators/dispatchable_generators_with_locations.csv"
    output:
        generators_with_wikipedia_locations="resources/generators/dispatchable_generators_with_wikipedia_locations.csv",
        wikipedia_coordinate_database="data/generators/wikipedia_power_station_coordinates.csv"
    params:
        wikipedia_coords_file="data/generators/wikipedia_power_station_coordinates.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/enhance_locations_with_wikipedia.log"
    script:
        "../scripts/generators/enhance_locations_with_wikipedia.py"


# OPTIONAL ENRICHMENT: Integrate DUKES 5.11 power station location data
rule integrate_dukes_locations:
    """
    Integrate DUKES (Digest of UK Energy Statistics) 5.11 dataset for comprehensive
    power station location mapping.
    
    This rule uses the authoritative DUKES dataset to fill remaining location gaps
    in the generator database. DUKES provides official government data with high-quality
    X/Y coordinates for UK power stations, offering the most comprehensive coverage
    available for completing the location mapping process.
    
    The rule performs intelligent name matching between DUKES stations and unmapped
    generators, then applies the official coordinates to achieve near-complete
    location coverage for the generator database.
    """
    input:
        generators_with_wikipedia_locations="resources/generators/dispatchable_generators_with_wikipedia_locations.csv",
        dukes_dataset="data/generators/DUKES_5.11_2025.xlsx"
    output:
        generators_with_dukes_locations="resources/generators/dispatchable_generators_with_dukes_locations.csv",
        dukes_coordinate_database="data/generators/dukes_power_station_coordinates.csv",
        dukes_matches="resources/generators/dukes_location_matches.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/integrate_dukes_locations.log"
    script:
        "../scripts/generators/integrate_dukes_locations.py"


# OPTIONAL ENRICHMENT: Apply single word matching for final location mapping
rule apply_single_word_matching:
    """
    Apply single word matching to find additional TEC-REPD location matches.
    
    This rule uses sophisticated word-based matching to identify generators
    with similar location names between TEC and REPD datasets, significantly
    improving location mapping success rates.
    """
    input:
        generators_with_dukes_locations="resources/generators/dispatchable_generators_with_dukes_locations.csv",
        tec_file="data/generators/tec-register-02-september-2025.csv",
        repd_file="data/renewables/repd-q2-july-2024.csv"
    output:
        generators_final="resources/generators/dispatchable_generators_final.csv",
        single_word_matches="resources/generators/single_word_tec_repd_matches.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/apply_single_word_matching.log"
    script:
        "../scripts/generators/single_word_tec_repd_matching.py"


# Aggregate rule for comprehensive dispatchable generator database with DUKES integration
rule all_dispatchable_generators_with_locations:
    """
    Aggregate rule to create comprehensive dispatchable generator database.
    
    This rule serves as the main target for generating a complete database of
    operational dispatchable generators with verified geographic locations,
    enhanced with Wikipedia coordinate data and comprehensive DUKES 5.11
    integration for maximum location coverage, plus comprehensive analysis
    of any remaining unmapped generators.
    """
    input:
        "resources/generators/dispatchable_generators_final.csv",
        "resources/generators/location_mapping_report.txt",
        "data/generators/wikipedia_power_station_coordinates.csv",
        "data/generators/dukes_power_station_coordinates.csv",
        "resources/generators/dukes_location_matches.csv",
        "resources/generators/single_word_tec_repd_matches.csv"
    output:
        "resources/generators/dispatchable_generators_complete.txt"
    shell:
        """
        echo "Comprehensive dispatchable generator database with all location matching techniques created successfully!" > {output}
        """


# Build final dispatchable generators CSV used by mapping/plotting
rule build_dispatchable_generators_complete_csv:
    input:
        "resources/generators/dispatchable_generators_final.csv"
    output:
        "resources/generators/dispatchable_generators_complete.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/build_dispatchable_generators_complete.log"
    script:
        "../scripts/generators/map_final_generators.py"


# Build unified generators_full.csv for plotting and analysis
rule build_generators_full:
    input:
        dispatchable="resources/generators/dispatchable_generators_complete.csv",
        wind_onshore="resources/renewable/wind_onshore_sites.csv",
        wind_offshore="resources/renewable/wind_offshore_sites.csv",
        solar_pv="resources/renewable/solar_pv_sites.csv",
        small_hydro="resources/renewable/small_hydro_sites.csv",
        large_hydro="resources/renewable/large_hydro_sites.csv",
        tidal_stream="resources/renewable/tidal_stream_sites.csv",
        shoreline_wave="resources/renewable/shoreline_wave_sites.csv",
        tidal_lagoon="resources/renewable/tidal_lagoon_sites.csv"
    output:
        "resources/generators/generators_full.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    log:
        "logs/build_generators_full.log"
    script:
        "../scripts/generators/create_generators_full.py"


## LEGACY: aggregate rule for individual site files (kept for compatibility)
rule all_dispatchable_generator_sites:
    """
    Aggregate rule to ensure all dispatchable generator site files are created.
    
    This rule serves as a convenient target to generate all dispatchable generator
    site CSV files and summary reports in one command.
    """
    input:
        # Core thermal generators (dynamically created based on data)
        "resources/generators/sites/nuclear_sites.csv",
        # Storage systems
        "resources/generators/sites/battery_sites.csv",
        "resources/generators/sites/pumped_hydro_sites.csv",
        "resources/generators/sites/hydrogen_storage_sites.csv",
        # Biomass and waste (dispatchable renewable thermal)
        "resources/generators/sites/biomass_sites.csv",
        "resources/generators/sites/waste_to_energy_sites.csv",
        "resources/generators/sites/biogas_sites.csv",
        "resources/generators/sites/landfill_gas_sites.csv",
        "resources/generators/sites/sewage_gas_sites.csv",
        "resources/generators/sites/advanced_biofuel_sites.csv",
        # Geothermal (baseload renewable thermal)
        "resources/generators/sites/geothermal_sites.csv",
        # Large hydro
        "resources/generators/sites/large_hydro_sites.csv",
        # Summary report
        "resources/generators/dispatchable_sites_summary.txt"
    output:
        "resources/generators/all_dispatchable_sites_summary.txt"
    shell:
        """
        echo "All dispatchable generator sites generated successfully!" > {output}
        """


# Rule to add FES generator data to the ETYS network
rule add_FES_generator_data_to_network:
    """
    This rule adds FES generator data to the ETYS network.

    Input:
      - CSV file for each year in the FES_year list, saved in the resources directory.
      - TEC register CSV files.
      - ETYS Appendix B Excel file.
      - ETYS base NetCDF file.

    Output:
      - Processed TEC register CSV file.
      - Updated NetCDF file with FES generator data.

    Script:
      - The script "scripts/generators.py" is executed to add the FES generator data to the network.
    """
    input:
        expand(f"{resources_path}/FES/FES_{{year}}_data.csv", year=fes_years),
        f"{data_path}/generators/tec-register-11-10-2024.csv",
        f"{data_path}/generators/tec-register-11-10-2024-manually-added-site-names.csv",
        f"{data_path}/network/ETYS/ETYS Appendix B 2023.xlsx",
        f"{resources_path}/network/ETYS_base.nc"
    output:
        f"{resources_path}/generators/TEC_register_processed.csv",
        f"{resources_path}/network/ETYS_base_demand_generator.nc"
    script:
        "../scripts/generators.py"


# Rule to generate capacities for generators
## LEGACY: generator capacity derivation (kept for compatibility)
rule generators_capacities:
    """
    This rule processes the TEC register and ETYS base demand generator data to generate capacities for generators.

    Input:
      - TEC register processed CSV file
      - ETYS base demand generator NetCDF file

    Script:
      - The script "scripts/generators_capacities.py" is executed to generate the capacities.
    """
    input:
        f"{resources_path}/generators/TEC_register_processed.csv",
        f"{resources_path}/network/ETYS_base_demand_generator.nc",
    script:
        "../scripts/generators/generators_capacities.py"

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
        # FES data (for future scenarios - use function to get conditional path)
        fes_data=lambda wildcards: f"{resources_path}/FES/FES_{scenarios[wildcards.scenario].get('FES_year', 2025)}_data.csv" if not is_historical_scenario(scenarios[wildcards.scenario]) else []
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables.pkl",
        summary=f"{resources_path}/generators/{{scenario}}_renewables_summary.csv"
    params:
        scenario_config=lambda wildcards: scenarios.get(wildcards.scenario, {}),
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
        # Generator characteristics database
        fuel_data=f"{data_path}/generators/generator_data_by_fuel.csv"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal.pkl",
        summary=f"{resources_path}/generators/{{scenario}}_thermal_summary.csv"
    params:
        scenario_config=lambda wildcards: scenarios[wildcards.scenario]
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
      - Final network: {scenario}_base_demand_generators.nc (READY FOR OPTIMIZATION)
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
        voll=lambda w: scenarios[w.scenario].get('voll', 6000.0)  # Value of Lost Load in £/MWh
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
# MARGINAL COST CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

rule apply_marginal_costs_to_network:
    """
    Compute and apply marginal costs to thermal generators.

    CONFIGURATION:
    --------------
    Marginal costs are configured in YAML files (no code editing needed):

    In config/defaults.yaml (project-wide defaults):
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
      4. Configuration defaults (defaults.yaml)
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
        fuel_prices=lambda w: f"{resources_path}/marginal_costs/fuel_prices_{scenarios[w.scenario].get('FES_year', 2024)}.csv" if scenarios[w.scenario].get('modelled_year', 2020) > 2024 and scenarios[w.scenario].get('FES_year') else [],
        carbon_prices=lambda w: f"{resources_path}/marginal_costs/carbon_prices_{scenarios[w.scenario].get('FES_year', 2024)}.csv" if scenarios[w.scenario].get('modelled_year', 2020) > 2024 and scenarios[w.scenario].get('FES_year') else []
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_costs.pkl",
        marginal_costs_csv=f"{resources_path}/generators/{{scenario}}_marginal_costs_breakdown.csv"
    params:
        scenario_config=lambda w: scenarios[w.scenario],
        # Optional: Pass FES price file paths to script for dynamic price loading
        fuel_price_file=lambda w: f"{resources_path}/marginal_costs/fuel_prices_{scenarios[w.scenario].get('FES_year', 2024)}.csv" if scenarios[w.scenario].get('modelled_year', 2020) > 2024 and scenarios[w.scenario].get('FES_year') else None,
        carbon_price_file=lambda w: f"{resources_path}/marginal_costs/carbon_prices_{scenarios[w.scenario].get('FES_year', 2024)}.csv" if scenarios[w.scenario].get('modelled_year', 2020) > 2024 and scenarios[w.scenario].get('FES_year') else None
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
    
    Use this as the main target for generator integration:
        snakemake integrate_generators_{scenario} --cores 4
    
    Or target final output directly:
        snakemake resources/network/{scenario}_base_demand_generators.nc --cores 4
    """
    input:
        # Final outputs from 3-stage pipeline
        network=f"{resources_path}/network/{{scenario}}_base_demand_generators.nc",
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


# Legacy aggregate rule (backward compatibility)
rule add_generators:
    """
    LEGACY: Aggregate target for old single-stage integration workflow.
    
    ⚠️  This is kept for backward compatibility only.
    ⚠️  New workflows should use: integrate_generators (3-stage pipeline)
    
    This was the original single-rule approach (all-in-one integration).
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_base_demand_generators.nc",
        csv_full=f"{resources_path}/generators/{{scenario}}_generators_full.csv",
        csv_summary=f"{resources_path}/generators/{{scenario}}_generators_summary_by_carrier.csv",
        csv_tech=f"{resources_path}/generators/{{scenario}}_technology_capacity_summary.csv",
        csv_report=f"{resources_path}/generators/{{scenario}}_generator_integration_report.txt",
        csv_export_marker=f"{resources_path}/network/csv/{{scenario}}_base_demand_generators/export_complete.txt"
    output:
        ready=f"{resources_path}/generators/{{scenario}}_pipeline_complete.txt"
    shell:
        """
        echo "Generator integration pipeline complete" > {output.ready}
        echo "(Using 3-stage modular approach)" >> {output.ready}
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
# Configuration is now via config/defaults.yaml and config/scenarios.yaml.
# See documentation for details.
# =============================================================================

