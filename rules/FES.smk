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
        config_file=f"{data_path}/FES/FES_api_urls.yaml",
        # The normalisation's own inputs (stage 2): the GSP name clean-up per edition, the
        # edition's committed GSP info table (Node IDs; FES 2020 has none) and the registry.
        gsp_aliases=f"{data_path}/FES/gsp_aliases.csv",
        gsp_info=lambda wc: _fes_gsp_info_input(wc.year),
        logic=["scripts/sources/registry.py", "scripts/FES/FES_data.py"],
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


def _fes_gsp_info_input(year):
    """The edition's committed GSP info table, where the edition has one (scripts/FES/FES_data.load_gsp_info)."""
    path = Path(data_path) / "network" / "ETYS" / f"fes{int(year)}_regional_breakdown_gsp_info.csv"
    return str(path).replace("\\", "/") if path.exists() else []


def get_fes_workbook_path(wildcards):
    """The FES Data Workbook of edition ``wildcards.year`` (scripts/sources/fes_adapter.WORKBOOKS).

    An edition with no registered workbook fails: no other edition's prices stand in for it
    (stage 5; before, the nearest later workbook was used with a warning).
    """
    from scripts.sources.fes_adapter import WORKBOOKS
    year = int(wildcards.year)
    if year not in WORKBOOKS:
        raise FileNotFoundError(f"No FES {year} Data Workbook registered (scripts/sources/fes_adapter.WORKBOOKS has "
                                f"{sorted(WORKBOOKS)}); add the workbook to data/FES/ and register it")
    return WORKBOOKS[year].relative_to(WORKBOOKS[year].parents[2]).as_posix()


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

    The edition's own workbook only (get_fes_workbook_path); a missing one fails.

    Transforms: FES Workbook → fuel_prices_{year}.csv, carbon_prices_{year}.csv
    """
    input:
        workbook=get_fes_workbook_path
    output:
        fuel_prices=f"{resources_path}/marginal_costs/fuel_prices_{{year}}.csv",
        carbon_prices=f"{resources_path}/marginal_costs/carbon_prices_{{year}}.csv"
    params:
        fes_year=lambda wc: int(wc.year),
        price_case="Base Case",
        # FES 2024 publishes no Base Case gas price, only High and Low: take their mean (user, 18 Nov 2026).
        missing_case="midpoint"
    wildcard_constraints:
        year="[0-9]{4}"
    log:
        "logs/FES/extract_FES_prices_{year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/extract_FES_prices.py"
