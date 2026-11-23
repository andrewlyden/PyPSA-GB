"""
Legacy generator data-preparation rules.

These rules materialise and enrich DUKES/TEC/REPD-derived generator tables.
They are retained for reproducible data maintenance, but are separate from the
scenario-specific generator-integration pipeline in generators.smk.
"""

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


