#!/usr/bin/env python3

"""
Snakemake Rules: Renewable Energy Profile Generation

This module processes renewable energy site data and generates individual
time series profiles for each renewable generator.

OUTPUTS (per technology):
  1. Site Data Files: {technology}_sites.csv
     - Columns: site_name, capacity_mw, lat, lon, [technology-specific metadata]
     - One row per individual renewable generator site
     - Used by generators.smk for adding generators to network
  
  2. Profile Files: profiles/{technology}_{year}.csv
     - Columns: One column per site (named by site_name)
     - Index: Hourly timestamps
     - Values: Power output in MW (not capacity factors)
     - Later converted to p_max_pu by dividing by installed capacity
     - Resampled to match network snapshots during integration

TECHNOLOGIES COVERED:
  - Weather-variable (atlite-based):
    * Wind onshore (includes onshore wind farms from REPD)
    * Wind offshore (REPD sites + future pipeline projects)
    * Solar PV (ground-mounted and rooftop solar from REPD)
  
  - Marine renewables (synthetic cyclic profiles):
    * Tidal stream (predictable tidal cycles)
    * Shoreline wave (wave power potential)
    * Tidal lagoon (tidal barrage systems)
  
  - Hydro renewables (flow-based profiles):
    * Large hydro (>10 MW, reservoir-based, dispatchable)
    * Small hydro (≤10 MW, run-of-river, weather-dependent)

"""

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
# Uses: scenarios, run_ids from main Snakefile

# Paths (use from main Snakefile where possible)
atlite_cutouts_path = f"{resources_path}/atlite/cutouts"
renewable_data_path = f"{resources_path}/renewable"


def _get_required_profile_years():
    """Weather years whose profiles the built scenarios read."""
    return sorted({scenarios[sid].get("renewables_year", 2020) for sid in build_ids if sid in scenarios})

required_years = _get_required_profile_years()

# Missing cutouts are declared as rule inputs below, so Snakemake can report
# them for the requesting rule without filesystem probing during workflow parsing.

# The offshore pipeline is cut at the latest modelled year of ANY defined
# scenario, not of the selected ones: the processed file (and so every offshore
# profile) must not change with the run selection. Profile files are shared by
# every scenario with that weather year whatever its modelled year, so extra
# pipeline columns are simply not picked up by an earlier-year integration.
pipeline_reference_year = max(
    int(cfg["modelled_year"]) for cfg in scenarios.values() if cfg.get("modelled_year")
)

# Profiles are made one weather year per job, so adding a year to the run
# selection makes that year's profiles without regenerating the others.
wildcard_constraints:
    renewables_year=r"\d{4}"


# Prepare site data for renewables in REPD
rule prepare_renewable_site_data:
    """
    Extract individual renewable generator sites from REPD database.
    
    This rule processes the Renewable Energy Planning Database (REPD) to create
    technology-specific site files with complete metadata for each generator.
    
    Processing Steps:
      1. Load REPD data and offshore pipeline projects
      2. Filter to operational sites in GB (exclude Northern Ireland)
      3. Convert coordinates (OSGB36 → WGS84 for geographic mapping)
      4. Split by technology type (wind, solar, hydro, marine, etc.)
      5. Export individual site CSVs with standardized format
    
    Output Format (all *_sites.csv files):
      - site_name: Unique identifier for each generator
      - capacity_mw: Installed electrical capacity (MWelec)
      - lat: Latitude in WGS84 (for bus assignment and mapping)
      - lon: Longitude in WGS84 (for bus assignment and mapping)
      - [Additional technology-specific metadata if available]
    
    Technology Categorization:
      - Wind: Separated into onshore vs offshore
      - Solar: Ground-mounted and rooftop PV combined
      - Hydro: Split by capacity (small ≤10 MW, large >10 MW)
      - Marine: Tidal stream, wave, tidal lagoon
    
    Offshore Wind Special Handling:
      - Combines operational REPD sites with future pipeline projects
      - Pipeline projects filtered by expected operational year
      - Ensures comprehensive coverage of offshore wind capacity
    
    Usage Downstream:
      - generators.smk: Reads site files to add individual generators to network
      - Provides capacity and location for each renewable generator
      - Site names used to match with time series profiles
    
    Performance: ~5-10 seconds (depends on REPD file size)
    """
    input:
        repd="data/renewables/repd-q2-jul-2025.csv",
        offshore_pipeline="data/renewables/future_offshore_sites/offshore_pipeline.csv",
        historical_site_overrides="data/renewables/historical_renewable_site_overrides.csv",
        cfd_round_corrections="data/market/lccc/cfd_round_corrections.csv",
        support_labels="scripts/renewables/repd_support_labels.py"
    output:
        wind_onshore_repd=f"{renewable_data_path}/wind_onshore_sites.csv",
        wind_offshore_repd=f"{renewable_data_path}/wind_offshore_sites.csv",
        solar_pv_repd=f"{renewable_data_path}/solar_pv_sites.csv",
        small_hydro=f"{renewable_data_path}/small_hydro_sites.csv",
        large_hydro=f"{renewable_data_path}/large_hydro_sites.csv",
        tidal_stream=f"{renewable_data_path}/tidal_stream_sites.csv",
        shoreline_wave=f"{renewable_data_path}/shoreline_wave_sites.csv",
        tidal_lagoon=f"{renewable_data_path}/tidal_lagoon_sites.csv",
        offshore_pipeline_processed=f"{renewable_data_path}/offshore_pipeline_processed.csv"
    params:
        pipeline_reference_year=pipeline_reference_year
    log:
        "logs/prepare_renewable_site_data.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/renewables/prepare_renewable_site_data.py"


rule prepare_future_pipeline_sites:
    """
    Extract the named REPD pipeline (under construction, consented, in planning)
    for the three weather-variable carriers, one site file per technology.

    The historical site files above keep only operational projects, which a
    2030 scenario cannot be built from. These companion files carry status,
    planning milestones, coordinates and REPD's own CfD round and ROC banding so
    a brownfield CP30 fleet can place, date and label named projects and proxy
    only the residual. Which statuses a scenario includes is decided in its
    ``spatial_proxy.brownfield`` configuration, not here.
    """
    input:
        repd="data/renewables/repd-q2-jul-2025.csv",
        offshore_pipeline="data/renewables/future_offshore_sites/offshore_pipeline.csv",
        logic="scripts/renewables/prepare_future_pipeline_sites.py",
        cfd_round_corrections="data/market/lccc/cfd_round_corrections.csv",
        support_labels="scripts/renewables/repd_support_labels.py",
    output:
        wind_offshore=f"{renewable_data_path}/wind_offshore_pipeline_sites.csv",
        wind_onshore=f"{renewable_data_path}/wind_onshore_pipeline_sites.csv",
        solar_pv=f"{renewable_data_path}/solar_pv_pipeline_sites.csv",
        summary=f"{renewable_data_path}/pipeline_sites_summary.csv",
    log:
        "logs/prepare_future_pipeline_sites.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/renewables/prepare_future_pipeline_sites.py"


# Fetch a weather cutout when it is missing, so a fresh clone can run without
# Snakefile_cutouts: a copy in data/atlite/cutouts/ first, then the published
# Zenodo record (10.5281/zenodo.18325225, ~700 MB a year), and only then ERA5
# through atlite (hours per year, needs a CDS API key).
rule retrieve_cutout:
    output:
        f"{atlite_cutouts_path}/uk-{{cutout_year}}.nc"
    wildcard_constraints:
        cutout_year=r"\d{4}"
    params:
        years=lambda wc: [int(wc.cutout_year)],
        config={"zenodo": {"enabled": True, "verify_checksum": True}},
    log:
        "logs/retrieve_cutout_{cutout_year}.log"
    retries: 1
    script:
        "../scripts/utilities/prepare_cutouts.py"


# Generate renewable profiles using atlite methods
rule generate_renewable_profiles:
    """
    Generate individual time series profiles for weather-variable renewable generators.
    
    This rule uses atlite to compute weather-dependent power output for each
    renewable site based on historical weather data (cutouts).
    
    Process Flow:
      1. Load weather cutouts for specified years
      2. Load site data (individual generators with capacity and location)
      3. For each site:
         a. Find nearest weather grid cell
         b. Extract capacity factor time series
         c. Convert to power output: power_MW = capacity_factor × installed_capacity_mw
      4. Create one CSV per technology-year with columns = site names
    
    Output Format:
      - Index: Hourly timestamps (matching cutout resolution)
      - Columns: One column per site (column name = site_name)
      - Values: Power output in MW (NOT capacity factors)
      - Note: Later resampled to network snapshots and converted to p_max_pu
    
    Technology-Specific Atlite Methods:
      - Wind onshore: cutout.wind(turbine='Vestas_V112_3MW')
      - Wind offshore: cutout.wind(turbine='NREL_ReferenceTurbine_5MW_offshore')
      - Solar PV: cutout.pv(panel='CSi', orientation='latitude_optimal')
    
    Offshore Wind Special Handling:
      - Combines REPD operational sites with pipeline projects
      - Pipeline sites use same weather-based methodology
      - Ensures comprehensive future offshore capacity coverage
    
    Performance Optimization:
      - Uses capacity_factor_timeseries=True for efficient grid computation
      - Processes all sites in parallel using atlite grid methods
      - Typical runtime: 30-60 seconds per technology-year
    
    Downstream Usage:
      - generators.smk reads these profiles during generator integration
      - add_generators.py converts MW → p_max_pu and resamples to network timestep
      - Each site becomes an individual generator in the PyPSA network
    
    See Also:
      - scripts/renewables/map_renewable_profiles.py for implementation details
      - rules/generators.smk for how profiles are integrated into network
    """
    input:
        cutouts=lambda wc: [f"{atlite_cutouts_path}/uk-{wc.renewables_year}.nc"],
        offshore_pipeline_processed=rules.prepare_renewable_site_data.output.offshore_pipeline_processed,
        wind_onshore=rules.prepare_renewable_site_data.output.wind_onshore_repd,
        wind_offshore=rules.prepare_renewable_site_data.output.wind_offshore_repd,
        solar_pv=rules.prepare_renewable_site_data.output.solar_pv_repd,
        small_hydro=rules.prepare_renewable_site_data.output.small_hydro,  # For future hydro timeseries
        # Named pipeline projects receive coordinate-based profiles too, so a
        # brownfield future scenario can place them by site.
        wind_onshore_pipeline=rules.prepare_future_pipeline_sites.output.wind_onshore,
        wind_offshore_pipeline=rules.prepare_future_pipeline_sites.output.wind_offshore,
        solar_pv_pipeline=rules.prepare_future_pipeline_sites.output.solar_pv,
    output:
        # Weather-variable renewables (require atlite-based timeseries)
        wind_onshore_profiles=[f"{renewable_data_path}/profiles/wind_onshore_{{renewables_year}}.csv"],
        wind_offshore_profiles=[f"{renewable_data_path}/profiles/wind_offshore_{{renewables_year}}.csv"],
        solar_pv_profiles=[f"{renewable_data_path}/profiles/solar_pv_{{renewables_year}}.csv"]
        # Note: Removed other technologies as they don't require weather-based timeseries:
        # - Geothermal: dispatchable baseload (p_max_pu ~0.9)
        # - Large Hydro: storage-like operation (no generation profile needed)
        # - Tidal Stream/Wave/Lagoon: separate rule
        # - Storage technologies: modeled as storage units, not generators
    params:
        renewables_year=lambda wc: [int(wc.renewables_year)],
    log:
        "logs/generate_renewable_profiles_optimized_{renewables_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/renewables/map_renewable_profiles.py"

# Generate marine renewable profiles using synthetic cyclic timeseries
rule generate_marine_profiles:
    """
    Generate marine renewable power profiles.
    
    Outputs power generation in MW (not capacity factors) using synthetic
    timeseries based on tidal/wave cycles rather than weather data.
    """
    input:
        tidal_stream=rules.prepare_renewable_site_data.output.tidal_stream,
        shoreline_wave=rules.prepare_renewable_site_data.output.shoreline_wave,
        tidal_lagoon=rules.prepare_renewable_site_data.output.tidal_lagoon
    output:
        # Marine renewables (predictable cyclic timeseries)
        tidal_stream_profiles=[f"{renewable_data_path}/profiles/tidal_stream_{{renewables_year}}.csv"],
        shoreline_wave_profiles=[f"{renewable_data_path}/profiles/shoreline_wave_{{renewables_year}}.csv"],
        tidal_lagoon_profiles=[f"{renewable_data_path}/profiles/tidal_lagoon_{{renewables_year}}.csv"]
    params:
        renewables_year=lambda wc: [int(wc.renewables_year)]
    log:
        "logs/generate_marine_profiles_{renewables_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/renewables/generate_marine_profiles.py"

# Generate hydro renewable profiles with operational characteristics
rule generate_hydro_profiles:
    """
    Generate hydro power profiles.
    
    Outputs power generation in MW (not capacity factors).
    - Large hydro: Dispatchable with reservoir storage
    - Small hydro: Run-of-river with seasonal flow patterns
    """
    input:
        large_hydro=rules.prepare_renewable_site_data.output.large_hydro,
        small_hydro=rules.prepare_renewable_site_data.output.small_hydro
    output:
        # Hydro renewables (different operational models)
        large_hydro_profiles=[f"{renewable_data_path}/profiles/large_hydro_{{renewables_year}}.csv"],
        small_hydro_profiles=[f"{renewable_data_path}/profiles/small_hydro_{{renewables_year}}.csv"]
    params:
        renewables_year=lambda wc: [int(wc.renewables_year)]
    log:
        "logs/generate_hydro_profiles_{renewables_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/renewables/generate_hydro_profiles.py"

# Combined rule to generate all renewable profiles
rule all_renewable_profiles:
    input:
        # Weather-variable renewables
        wind_onshore_profiles=expand(f"{renewable_data_path}/profiles/wind_onshore_{{renewables_year}}.csv", renewables_year=required_years),
        wind_offshore_profiles=expand(f"{renewable_data_path}/profiles/wind_offshore_{{renewables_year}}.csv", renewables_year=required_years),
        solar_pv_profiles=expand(f"{renewable_data_path}/profiles/solar_pv_{{renewables_year}}.csv", renewables_year=required_years),
        # Marine renewables
        tidal_stream_profiles=expand(f"{renewable_data_path}/profiles/tidal_stream_{{renewables_year}}.csv", renewables_year=required_years),
        shoreline_wave_profiles=expand(f"{renewable_data_path}/profiles/shoreline_wave_{{renewables_year}}.csv", renewables_year=required_years),
        tidal_lagoon_profiles=expand(f"{renewable_data_path}/profiles/tidal_lagoon_{{renewables_year}}.csv", renewables_year=required_years),
        # Hydro renewables
        large_hydro_profiles=expand(f"{renewable_data_path}/profiles/large_hydro_{{renewables_year}}.csv", renewables_year=required_years),
        small_hydro_profiles=expand(f"{renewable_data_path}/profiles/small_hydro_{{renewables_year}}.csv", renewables_year=required_years)
    output:
        summary=f"{renewable_data_path}/all_renewable_profiles_summary.txt"
