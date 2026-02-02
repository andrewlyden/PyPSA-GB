"""
FES Data Rules for PyPSA-GB

Downloads and processes Future Energy Scenarios data from NESO API.

Pipeline Stages:
  1. FES_data - Download and process FES dataset for a specific year

Inputs:
  - FES API configuration (data/FES/FES_api_urls.yaml)

Outputs:
  - FES data CSV per year (resources/FES/FES_{year}_data.csv)

"""

# ══════════════════════════════════════════════════════════════════════════════
# RULES
# ══════════════════════════════════════════════════════════════════════════════

rule FES_data:
    """
    Download and preprocess Future Energy Scenarios dataset for a single year.
    
    Each invocation handles exactly one FES year, allowing Snakemake to
    re-run or cache individual years independently.
    
    Transforms: FES API → FES_{year}_data.csv
    
    Performance: ~30-60s (depends on API response time)
    """
    input:
        config_file=f"{data_path}/FES/FES_api_urls.yaml"
    output:
        fes_data=f"{resources_path}/FES/FES_{{year}}_data.csv"
    params:
        fes_year=lambda wc: int(wc.year)
    wildcard_constraints:
        year="[0-9]{4}"
    log:
        "logs/FES/FES_data_{year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/FES_data.py"

def get_fes_workbook_path(wildcards):
    """Get path to FES workbook, dynamically detecting available files."""
    import re
    from pathlib import Path
    
    year = int(wildcards.year)
    fes_dir = Path(data_path) / "FES"
    
    # Scan for all .xlsx files in FES directory
    workbooks = list(fes_dir.glob("*.xlsx"))
    
    # Extract year from filename using multiple patterns
    year_patterns = [
        r'(?:FES\s*)?(20\d{2})',  # FES 2021, FES2021, 2021
        r'Data-workbook(20\d{2})',  # Data-workbook2020
        r'Future Energy Scenarios (20\d{2})',  # Future Energy Scenarios 2024
    ]
    
    workbook_map = {}
    for wb in workbooks:
        wb_name = wb.name
        for pattern in year_patterns:
            match = re.search(pattern, wb_name, re.IGNORECASE)
            if match:
                wb_year = int(match.group(1))
                workbook_map[wb_year] = str(wb)
                break
    
    # Use exact year if available
    if year in workbook_map:
        return workbook_map[year]
    
    # Fall back to closest available year (prefer same or newer)
    available_years = sorted(workbook_map.keys())
    if not available_years:
        raise FileNotFoundError(f"No FES workbooks found in {fes_dir}")
    
    # Find closest year >= requested year, otherwise use most recent
    fallback_year = max(available_years)
    for avail_year in available_years:
        if avail_year >= year:
            fallback_year = avail_year
            break
    
    print(f"Warning: FES workbook for {year} not found, using {fallback_year}")
    return workbook_map[fallback_year]


rule extract_FES_building_block_definitions:
    """
    Download and save FES Building Block Definitions for a specific year.

    Building block definitions describe what each building block ID represents:
    - Technology type (e.g., Solar, Wind, EVs, Heat Pumps)
    - Units (MW, GWh, Number of)
    - Detailed descriptions

    This provides the complete reference for interpreting the building block data
    in FES_{year}_data.csv.

    Categories include:
    - Dem_BB*: Demand building blocks (customers, consumption)
    - Gen_BB*: Generation building blocks (renewables, thermal)
    - Lct_BB*: Low Carbon Technology building blocks (EVs, heat pumps)
    - Srg_BB*: Storage & Flexibility building blocks (batteries, V2G, DSR)

    Transforms: FES API → building_block_definitions_{year}.csv

    Performance: ~5-10s (API download)
    """
    input:
        config_file=f"{data_path}/FES/FES_api_urls.yaml"
    output:
        definitions=f"{resources_path}/FES/building_block_definitions_{{year}}.csv"
    params:
        fes_year=lambda wc: int(wc.year)
    wildcard_constraints:
        year="[0-9]{4}"
    log:
        "logs/FES/extract_building_block_definitions_{year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/extract_building_block_definitions.py"


rule extract_FES_prices:
    """
    Extract fuel and carbon prices from FES workbooks.

    Extracts price assumptions for fuels (gas, coal, oil) and carbon from
    the FES Data Workbook sheets CP1 (fuel prices) and CP2 (carbon prices).

    Falls back to most recent available workbook (FES2022) if requested year not available.

    Transforms: FES Workbook → fuel_prices_{year}.csv, carbon_prices_{year}.csv
    """
    input:
        workbook=get_fes_workbook_path
    output:
        fuel_prices=f"{resources_path}/marginal_costs/fuel_prices_{{year}}.csv",
        carbon_prices=f"{resources_path}/marginal_costs/carbon_prices_{{year}}.csv"
    params:
        fes_year=lambda wc: int(wc.year)
    wildcard_constraints:
        year="[0-9]{4}"
    log:
        "logs/FES/extract_FES_prices_{year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/extract_FES_prices.py"