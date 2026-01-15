"""
Process European electricity generation mix data for interconnector modeling.

This script converts European generation technology mix data into marginal cost
estimates and price differentials used for interconnector flow modeling.

Author: PyPSA-GB
License: MIT
"""

import sys
import logging
from pathlib import Path
import time

import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utilities.logging_config import setup_logging, log_execution_summary

# Setup logging
logger = setup_logging(snakemake.log[0] if "snakemake" in dir() else "process_european_generation.log")


def load_fuel_prices(fuel_prices_file: str) -> pd.DataFrame:
    """
    Load fuel price assumptions.
    
    Args:
        fuel_prices_file: Path to fuel prices CSV
        
    Returns:
        pd.DataFrame: Fuel prices by fuel type and year
    """
    logger.info(f"Loading fuel prices from {fuel_prices_file}")
    
    try:
        df = pd.read_csv(fuel_prices_file)
        logger.info(f"✓ Loaded {len(df)} fuel price records")
        return df
    except FileNotFoundError:
        logger.warning(f"Fuel prices file not found: {fuel_prices_file}")
        logger.warning("Using default fuel price assumptions")
        
        # Default assumptions (£/MWh thermal)
        default_prices = pd.DataFrame({
            "fuel": ["gas", "coal", "oil", "nuclear", "biomass"],
            "price_gbp_per_mwh_thermal": [20, 10, 40, 5, 15]
        })
        return default_prices


def load_carbon_prices(carbon_price_file: str) -> pd.DataFrame:
    """
    Load carbon price assumptions.
    
    Args:
        carbon_price_file: Path to carbon prices CSV
        
    Returns:
        pd.DataFrame: Carbon prices by year
    """
    logger.info(f"Loading carbon prices from {carbon_price_file}")
    
    try:
        df = pd.read_csv(carbon_price_file)
        logger.info(f"✓ Loaded {len(df)} carbon price records")
        return df
    except FileNotFoundError:
        logger.warning(f"Carbon prices file not found: {carbon_price_file}")
        logger.warning("Using default carbon price assumptions")
        
        # Default assumptions (£/tCO2)
        default_prices = pd.DataFrame({
            "year": [2025, 2030, 2040, 2050],
            "carbon_price_gbp_per_tco2": [40, 60, 80, 100]
        })
        return default_prices


def estimate_marginal_costs(
    generation_mix: pd.DataFrame,
    fuel_prices: pd.DataFrame,
    carbon_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Estimate marginal electricity costs from generation mix.
    
    Uses merit order approach:
    1. Calculate marginal cost for each technology
    2. Order technologies by cost (merit order)
    3. Marginal price = cost of last technology needed to meet demand
    
    Args:
        generation_mix: Generation by technology and country
        fuel_prices: Fuel price assumptions
        carbon_prices: Carbon price assumptions
        
    Returns:
        pd.DataFrame: Estimated marginal costs by country, scenario, year
    """
    logger.info("Estimating marginal costs from generation mix...")
    
    # Technology-specific parameters
    tech_params = {
        "wind_offshore": {"efficiency": 1.0, "carbon_intensity": 0.0, "fuel": None},
        "wind_onshore": {"efficiency": 1.0, "carbon_intensity": 0.0, "fuel": None},
        "solar": {"efficiency": 1.0, "carbon_intensity": 0.0, "fuel": None},
        "nuclear": {"efficiency": 0.33, "carbon_intensity": 0.0, "fuel": "nuclear"},
        "hydro": {"efficiency": 1.0, "carbon_intensity": 0.0, "fuel": None},
        "biomass": {"efficiency": 0.35, "carbon_intensity": 0.0, "fuel": "biomass"},  # Assumed carbon neutral
        "gas_ccgt": {"efficiency": 0.55, "carbon_intensity": 0.202, "fuel": "gas"},  # tCO2/MWh thermal
        "gas_ocgt": {"efficiency": 0.40, "carbon_intensity": 0.202, "fuel": "gas"},
        "coal": {"efficiency": 0.38, "carbon_intensity": 0.341, "fuel": "coal"},
        "oil": {"efficiency": 0.35, "carbon_intensity": 0.279, "fuel": "oil"}
    }
    
    # Calculate marginal costs for each technology
    marginal_costs = []
    
    for tech, params in tech_params.items():
        # Fuel cost component
        if params["fuel"]:
            fuel_row = fuel_prices[fuel_prices["fuel"] == params["fuel"]]
            if not fuel_row.empty:
                fuel_cost_thermal = fuel_row["price_gbp_per_mwh_thermal"].iloc[0]
                fuel_cost_electric = fuel_cost_thermal / params["efficiency"]
            else:
                fuel_cost_electric = 0
        else:
            fuel_cost_electric = 0  # Renewables, hydro have no fuel cost
        
        # Carbon cost component (assuming average carbon price)
        carbon_price = carbon_prices["carbon_price_gbp_per_tco2"].mean()
        carbon_cost = params["carbon_intensity"] * carbon_price / params["efficiency"]
        
        # Total marginal cost
        total_mc = fuel_cost_electric + carbon_cost
        
        marginal_costs.append({
            "technology": tech,
            "marginal_cost_gbp_per_mwh": total_mc,
            "fuel_component": fuel_cost_electric,
            "carbon_component": carbon_cost
        })
    
    mc_df = pd.DataFrame(marginal_costs)
    logger.info(f"✓ Calculated marginal costs for {len(mc_df)} technologies")
    
    # Log merit order
    logger.info("Merit order (cheapest to most expensive):")
    for _, row in mc_df.sort_values("marginal_cost_gbp_per_mwh").iterrows():
        logger.info(f"  {row['technology']:20s}: £{row['marginal_cost_gbp_per_mwh']:6.2f}/MWh")
    
    return mc_df


def calculate_country_prices(
    parsed_mix: pd.DataFrame,
    marginal_costs: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate country-level electricity prices from generation mix.
    
    Uses capacity-weighted average of marginal costs as a first approximation.
    A more sophisticated approach would use merit order dispatch with demand levels.
    
    Args:
        parsed_mix: Parsed generation mix (country, year, technology, capacity)
        marginal_costs: Marginal cost by technology
        
    Returns:
        pd.DataFrame: Estimated prices by country, scenario, year
    """
    logger.info("Calculating country-level electricity prices...")
    
    # Filter for countries connected to GB
    connected_countries = ["France", "Belgium", "Netherlands", "SEM", "Norway", "Denmark"]
    country_data = parsed_mix[parsed_mix["Country"].isin(connected_countries)].copy()
    
    logger.info(f"Processing {len(country_data)} records for {country_data['Country'].nunique()} connected countries")
    
    # Add technology parameters
    country_data["tech_params"] = country_data.apply(
        lambda row: map_technologies_to_fuel_type(row["Type"], row["SubType"]),
        axis=1
    )
    
    # Extract parameters into columns
    country_data["fuel"] = country_data["tech_params"].apply(lambda x: x["fuel"])
    country_data["efficiency"] = country_data["tech_params"].apply(lambda x: x["efficiency"])
    country_data["carbon_intensity"] = country_data["tech_params"].apply(lambda x: x["carbon_intensity"])
    country_data["is_renewable"] = country_data["tech_params"].apply(lambda x: x["is_renewable"])
    
    # Merge with marginal costs (simplified - using technology type)
    country_data["technology_key"] = country_data["Type"].str.lower().str.replace(" ", "_")
    marginal_costs["technology_key"] = marginal_costs["technology"].str.lower().str.replace(" ", "_")
    
    country_data = country_data.merge(
        marginal_costs[["technology_key", "marginal_cost_gbp_per_mwh"]],
        on="technology_key",
        how="left"
    )
    
    # For technologies not in marginal_costs, calculate on the fly
    # (This handles new technology types)
    mask = country_data["marginal_cost_gbp_per_mwh"].isna()
    if mask.sum() > 0:
        logger.warning(f"Calculating marginal costs for {mask.sum()} unmapped technologies")
        # Use default approach - will add proper calculation if needed
        country_data.loc[mask, "marginal_cost_gbp_per_mwh"] = 40.0  # Default placeholder
    
    # Calculate capacity-weighted average price by country, scenario, year
    country_prices = []
    
    grouped = country_data.groupby(["Country", "EU Scenario", "FES Pathway Alignment", "year"])
    
    for (country, scenario, pathway, year), group in grouped:
        total_capacity = group["capacity_mw"].sum()
        
        if total_capacity > 0:
            # Capacity-weighted average marginal cost
            weighted_price = (
                (group["capacity_mw"] * group["marginal_cost_gbp_per_mwh"]).sum() / total_capacity
            )
            
            # Calculate renewable share
            renewable_capacity = group[group["is_renewable"]]["capacity_mw"].sum()
            renewable_share = renewable_capacity / total_capacity
            
            country_prices.append({
                "country": country,
                "scenario": scenario,
                "pathway": pathway,
                "year": year,
                "estimated_price_gbp_per_mwh": weighted_price,
                "total_capacity_mw": total_capacity,
                "renewable_share": renewable_share,
                "method": "capacity_weighted_average"
            })
    
    df = pd.DataFrame(country_prices)
    logger.info(f"✓ Calculated prices for {len(df)} country-scenario-year combinations")
    
    # Log sample results
    logger.info("\nSample price estimates (2030):")
    sample_2030 = df[df["year"] == 2030].groupby("country")["estimated_price_gbp_per_mwh"].mean()
    for country, price in sample_2030.items():
        logger.info(f"  {country:15s}: £{price:.2f}/MWh")
    
    return df


def calculate_price_differentials(
    country_prices: pd.DataFrame,
    gb_price: float = 50.0  # Placeholder GB price - should come from GB generation mix
) -> pd.DataFrame:
    """
    Calculate price differentials between GB and connected countries.
    
    Positive differential = GB price higher = incentive to import
    Negative differential = GB price lower = incentive to export
    
    Args:
        country_prices: Estimated prices by country, scenario, year
        gb_price: GB electricity price (£/MWh) - currently placeholder
        
    Returns:
        pd.DataFrame: Price differentials and flow indicators by country, scenario, year
    """
    logger.info("Calculating price differentials...")
    
    differentials = country_prices.copy()
    differentials["gb_price_gbp_per_mwh"] = gb_price
    differentials["price_differential_gbp_per_mwh"] = (
        gb_price - differentials["estimated_price_gbp_per_mwh"]
    )
    
    # Positive differential = GB imports (GB price higher, electricity flows TO GB)
    # Negative differential = GB exports (GB price lower, electricity flows FROM GB)
    differentials["expected_flow_direction"] = differentials["price_differential_gbp_per_mwh"].apply(
        lambda x: "import_to_gb" if x > 0 else "export_from_gb"
    )
    
    differentials["flow_incentive_strength"] = differentials["price_differential_gbp_per_mwh"].abs()
    
    logger.info("✓ Price differentials calculated")
    
    # Log summary by country (averaging across scenarios and years)
    logger.info("\nExpected flow patterns (average across scenarios/years):")
    summary = differentials.groupby("country").agg({
        "price_differential_gbp_per_mwh": "mean",
        "flow_incentive_strength": "mean"
    }).round(2)
    
    for country, row in summary.iterrows():
        diff = row["price_differential_gbp_per_mwh"]
        direction = "import_to_gb" if diff > 0 else "export_from_gb"
        logger.info(
            f"  {country:15s}: {direction:15s} "
            f"(avg Δ = £{abs(diff):5.2f}/MWh, strength = £{row['flow_incentive_strength']:5.2f}/MWh)"
        )
    
    return differentials


def parse_generation_mix(generation_mix: pd.DataFrame) -> pd.DataFrame:
    """
    Parse European generation mix data into structured format.
    
    The data has years as columns (2023-2050). We need to:
    1. Melt year columns into rows
    2. Filter for capacity data only
    3. Map to standardized technology names
    
    Args:
        generation_mix: Raw generation mix from NESO API
        
    Returns:
        pd.DataFrame: Structured capacity data by country, year, technology
    """
    logger.info("Parsing generation mix data structure...")
    
    # Get year columns (they're numeric year values like 2023, 2024, etc.)
    year_cols = [col for col in generation_mix.columns if col.isdigit()]
    logger.info(f"Found {len(year_cols)} year columns: {year_cols[0]} to {year_cols[-1]}")
    
    # Filter for capacity data only
    capacity_data = generation_mix[generation_mix["Variable"] == "Capacity (MW)"].copy()
    logger.info(f"Filtered to {len(capacity_data)} capacity records")
    
    # Melt year columns into rows
    id_cols = ["Country", "EU Scenario", "FES Pathway Alignment", "Variable", "Category", "Type", "SubType"]
    melted = capacity_data.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name="capacity_mw"
    )
    
    # Convert year to int and capacity to float
    melted["year"] = melted["year"].astype(int)
    melted["capacity_mw"] = pd.to_numeric(melted["capacity_mw"], errors="coerce")
    
    # Drop rows with null capacity
    melted = melted.dropna(subset=["capacity_mw"])
    
    logger.info(f"✓ Parsed {len(melted)} capacity records across {melted['year'].nunique()} years")
    logger.info(f"  Countries: {melted['Country'].nunique()}")
    logger.info(f"  Technologies: {melted['Type'].nunique()}")
    
    return melted


def map_technologies_to_fuel_type(technology: str, subtype: str) -> dict:
    """
    Map NESO technology names to fuel types and parameters.
    
    Args:
        technology: Technology type from NESO data
        subtype: Technology subtype from NESO data
        
    Returns:
        dict: Technology parameters (fuel, efficiency, carbon_intensity)
    """
    # Technology mapping
    tech_map = {
        "Offshore Wind": {"fuel": None, "efficiency": 1.0, "carbon_intensity": 0.0, "is_renewable": True},
        "Onshore Wind": {"fuel": None, "efficiency": 1.0, "carbon_intensity": 0.0, "is_renewable": True},
        "Solar": {"fuel": None, "efficiency": 1.0, "carbon_intensity": 0.0, "is_renewable": True},
        "Hydro": {"fuel": None, "efficiency": 1.0, "carbon_intensity": 0.0, "is_renewable": True},
        "Marine": {"fuel": None, "efficiency": 1.0, "carbon_intensity": 0.0, "is_renewable": True},
        "Nuclear": {"fuel": "nuclear", "efficiency": 0.33, "carbon_intensity": 0.0, "is_renewable": False},
        "Biomass": {"fuel": "biomass", "efficiency": 0.35, "carbon_intensity": 0.0, "is_renewable": True},  # Carbon neutral
        "Waste": {"fuel": "biomass", "efficiency": 0.25, "carbon_intensity": 0.0, "is_renewable": True},
        "Gas": {"fuel": "gas", "efficiency": 0.50, "carbon_intensity": 0.202, "is_renewable": False},  # tCO2/MWh thermal
        "Coal": {"fuel": "coal", "efficiency": 0.38, "carbon_intensity": 0.341, "is_renewable": False},
        "Other Thermal": {"fuel": "oil", "efficiency": 0.35, "carbon_intensity": 0.279, "is_renewable": False},
        "Storage": {"fuel": None, "efficiency": 0.85, "carbon_intensity": 0.0, "is_renewable": False},  # Round-trip efficiency
        "CCS": {"fuel": "gas", "efficiency": 0.45, "carbon_intensity": 0.02, "is_renewable": False},  # 90% capture
        "Low Carbon": {"fuel": "gas", "efficiency": 0.50, "carbon_intensity": 0.05, "is_renewable": False},
    }
    
    # Get base params
    params = tech_map.get(technology, {"fuel": "gas", "efficiency": 0.40, "carbon_intensity": 0.202, "is_renewable": False})
    
    # Refine for CCGT vs OCGT
    if technology == "Gas":
        if "CCGT" in subtype or "Combined" in subtype:
            params["efficiency"] = 0.55  # CCGT more efficient
        else:
            params["efficiency"] = 0.40  # OCGT less efficient
    
    return params


def main():
    """Main execution function."""
    start_time = time.time()
    
    # Get parameters from Snakemake
    generation_mix_file = snakemake.input.generation_mix
    fuel_prices_file = snakemake.input.fuel_prices
    carbon_price_file = snakemake.input.carbon_prices
    
    output_costs = snakemake.output.marginal_costs
    output_differentials = snakemake.output.price_differentials
    
    logger.info("=" * 80)
    logger.info("European Generation Mix Processing")
    logger.info("=" * 80)
    
    # Load generation mix
    logger.info(f"Loading generation mix from {generation_mix_file}")
    generation_mix = pd.read_csv(generation_mix_file)
    logger.info(f"✓ Loaded {len(generation_mix)} generation records")
    logger.info(f"  Columns: {', '.join(generation_mix.columns[:10])}...")
    
    # Parse generation mix into structured format
    parsed_mix = parse_generation_mix(generation_mix)
    
    # Load fuel and carbon prices
    fuel_prices = load_fuel_prices(fuel_prices_file)
    carbon_prices = load_carbon_prices(carbon_price_file)
    
    # Estimate marginal costs
    marginal_costs = estimate_marginal_costs(generation_mix, fuel_prices, carbon_prices)
    
    # Calculate country prices (using parsed data)
    country_prices = calculate_country_prices(parsed_mix, marginal_costs)
    
    # Calculate price differentials
    price_differentials = calculate_price_differentials(country_prices)
    
    # Save outputs
    logger.info(f"Saving marginal costs to {output_costs}")
    marginal_costs.to_csv(output_costs, index=False)
    
    logger.info(f"Saving price differentials to {output_differentials}")
    price_differentials.to_csv(output_differentials, index=False)
    
    # Calculate statistics
    countries = parsed_mix['country'].nunique() if 'country' in parsed_mix.columns else 0
    years = parsed_mix['year'].nunique() if 'year' in parsed_mix.columns else 0
    scenarios = len(marginal_costs)
    
    # Log execution summary
    log_execution_summary(
        logger,
        "process_european_generation",
        start_time,
        inputs={'generation_mix': generation_mix_file, 'fuel_prices': fuel_prices_file, 'carbon_prices': carbon_price_file},
        outputs={'marginal_costs': output_costs, 'price_differentials': output_differentials},
        context={
            'countries': countries,
            'years': years,
            'scenarios': scenarios,
            'total_records': len(marginal_costs)
        }
    )


if __name__ == "__main__":
    if "snakemake" not in dir():
        logger.error("This script must be run via Snakemake")
        sys.exit(1)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Failed to process European generation mix: {e}")
        raise

