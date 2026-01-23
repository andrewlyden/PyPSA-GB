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


def _extract_from_scenarios(key):
    """Extract values from active scenarios for the renewables module."""
    try:
        return sorted({scenarios[rid][key] for rid in run_ids if key in scenarios.get(rid, {})})
    except Exception as e:
        print(f"Warning: Could not extract {key} from scenarios: {e}")
        return [2020]


def _get_required_cutout_years():
    """Return set of cutout years required by active scenarios."""
    years = set()
    try:
        for sid in run_ids:
            if sid in scenarios:
                sc = scenarios[sid]
                years.add(sc.get("renewables_year", 2020))
                years.add(sc.get("demand_year", 2020))
    except Exception as e:
        print(f"Warning: Could not determine required years: {e}")
        years.add(2020)
    return years


def _get_available_cutouts():
    """List existing cutout files for required years, error if missing."""
    from pathlib import Path
    import sys
    
    # Add scripts to path for imports
    sys.path.insert(0, 'scripts')
    from scripts.utilities.scenario_detection import check_cutout_availability, is_historical_scenario

    cutout_dir = Path(atlite_cutouts_path)
    if not cutout_dir.exists():
        raise FileNotFoundError(
            "Cutout directory not found: "
            f"{cutout_dir}\n"
            "Generate cutouts first: snakemake -s Snakefile_cutouts --cores 4"
        )

    req_years = _get_required_cutout_years()
    have, missing = [], []
    missing_details = []
    
    for y in sorted(req_years):
        p = cutout_dir / f"uk-{y}.nc"
        if p.exists():
            have.append(str(p))
        else:
            missing.append(str(p))
            
            # Find which scenarios need this year
            scenarios_needing_year = [
                sid for sid in run_ids 
                if sid in scenarios and scenarios[sid].get('renewables_year') == y
            ]
            
            if scenarios_needing_year:
                scenario_type = "HISTORICAL" if y <= 2024 else "FUTURE"
                missing_details.append(
                    f"  • uk-{y}.nc (required by {', '.join(scenarios_needing_year)}) [{scenario_type}]"
                )

    if missing:
        error_msg = (
            "=" * 80 + "\n"
            "❌ MISSING WEATHER CUTOUT FILES\n"
            "=" * 80 + "\n"
            f"Required cutout files not found:\n" +
            "\n".join(missing_details) + "\n\n" +
            "🔧 SOLUTION:\n"
            "1. Update config/cutouts_config.yaml to include missing years:\n"
            f"   years_to_process: {sorted(list(req_years))}\n\n"
            "2. Run cutout generation workflow:\n"
            "   snakemake -s Snakefile_cutouts --cores 2\n\n"
            "This will download and process ERA5 weather data for the required years.\n"
            "Note: Cutout generation can take 30-60 minutes per year.\n"
            "=" * 80
        )
        raise FileNotFoundError(error_msg)

    return have


# Static lists used by rules
required_years = _get_required_cutout_years()
expected_cutouts = [f"{atlite_cutouts_path}/uk-{y}.nc" for y in sorted(required_years)]

# Early check (fail-fast) for missing cutouts
try:
    _avail = _get_available_cutouts()
    print(
        "[OK] Found "
        f"{len(_avail)} cutout files for required years {sorted(required_years)}"
    )
    for _c in _avail:
        print(f"   - {_c}")
except FileNotFoundError as e:
    print(f"\n{e}\n")
    import sys
    sys.exit(1)


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
        offshore_pipeline="data/renewables/future_offshore_sites/offshore_pipeline.csv"
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
        renewables_years=required_years,
        modelled_years=_extract_from_scenarios("modelled_year")
    log:
        "logs/prepare_renewable_site_data.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/renewables/prepare_renewable_site_data.py"


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
      - scripts/map_renewable_profiles.py for implementation details
      - rules/generators.smk for how profiles are integrated into network
    """
    input:
        cutouts=expected_cutouts,
        wind_onshore=rules.prepare_renewable_site_data.output.wind_onshore_repd,
        wind_offshore=rules.prepare_renewable_site_data.output.wind_offshore_repd,
        solar_pv=rules.prepare_renewable_site_data.output.solar_pv_repd,
        small_hydro=rules.prepare_renewable_site_data.output.small_hydro  # For future hydro timeseries
    output:
        # Weather-variable renewables (require atlite-based timeseries)
        wind_onshore_profiles=expand(
            f"{renewable_data_path}/profiles/wind_onshore_{{renewables_year}}.csv",
            renewables_year=required_years,
        ),
        wind_offshore_profiles=expand(
            f"{renewable_data_path}/profiles/wind_offshore_{{renewables_year}}.csv",
            renewables_year=required_years,
        ),
        solar_pv_profiles=expand(
            f"{renewable_data_path}/profiles/solar_pv_{{renewables_year}}.csv",
            renewables_year=required_years,
        )
        # Note: Removed other technologies as they don't require weather-based timeseries:
        # - Geothermal: dispatchable baseload (p_max_pu ~0.9)
        # - Large Hydro: storage-like operation (no generation profile needed)
        # - Tidal Stream/Wave/Lagoon: separate rule
        # - Storage technologies: modeled as storage units, not generators
    params:
        renewables_year=lambda wc: sorted(required_years),
        scenario_configs=lambda wc: {rid: {} for rid in ["HT35"]}
    log:
        expand(
            "logs/generate_renewable_profiles_optimized_{renewables_year}.log",
            renewables_year=required_years,
        )
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
        tidal_stream_profiles=expand(
            f"{renewable_data_path}/profiles/tidal_stream_{{renewables_year}}.csv",
            renewables_year=required_years,
        ),
        shoreline_wave_profiles=expand(
            f"{renewable_data_path}/profiles/shoreline_wave_{{renewables_year}}.csv",
            renewables_year=required_years,
        ),
        tidal_lagoon_profiles=expand(
            f"{renewable_data_path}/profiles/tidal_lagoon_{{renewables_year}}.csv",
            renewables_year=required_years,
        )
    params:
        renewables_year=lambda wc: sorted(required_years)
    log:
        expand(
            "logs/generate_marine_profiles_{renewables_year}.log",
            renewables_year=required_years,
        )
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
        large_hydro_profiles=expand(
            f"{renewable_data_path}/profiles/large_hydro_{{renewables_year}}.csv",
            renewables_year=required_years,
        ),
        small_hydro_profiles=expand(
            f"{renewable_data_path}/profiles/small_hydro_{{renewables_year}}.csv",
            renewables_year=required_years,
        )
    params:
        renewables_year=lambda wc: sorted(required_years)
    log:
        expand(
            "logs/generate_hydro_profiles_{renewables_year}.log",
            renewables_year=required_years,
        )
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/renewables/generate_hydro_profiles.py"

# Combined rule to generate all renewable profiles
rule all_renewable_profiles:
    input:
        # Weather-variable renewables
        wind_onshore_profiles=rules.generate_renewable_profiles.output.wind_onshore_profiles,
        wind_offshore_profiles=rules.generate_renewable_profiles.output.wind_offshore_profiles,
        solar_pv_profiles=rules.generate_renewable_profiles.output.solar_pv_profiles,
        # Marine renewables
        tidal_stream_profiles=rules.generate_marine_profiles.output.tidal_stream_profiles,
        shoreline_wave_profiles=rules.generate_marine_profiles.output.shoreline_wave_profiles,
        tidal_lagoon_profiles=rules.generate_marine_profiles.output.tidal_lagoon_profiles,
        # Hydro renewables
        large_hydro_profiles=rules.generate_hydro_profiles.output.large_hydro_profiles,
        small_hydro_profiles=rules.generate_hydro_profiles.output.small_hydro_profiles
    output:
        summary=f"{renewable_data_path}/all_renewable_profiles_summary.txt"
