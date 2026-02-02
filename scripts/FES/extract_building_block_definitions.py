"""
Extract FES Building Block Definitions from NESO API.

This script downloads and saves the complete building block definitions file
from the FES API. The definitions describe what each building block ID represents,
including technology type, units, and detailed descriptions.

Building Block Categories:
- Dem_BB*: Demand (customers, consumption by sector)
- Gen_BB*: Generation (renewables, thermal, nuclear, etc.)
- Lct_BB*: Low Carbon Technologies (EVs, heat pumps, district heating)
- Srg_BB*: Storage & Flexibility (batteries, pumped hydro, V2G, DSR)

Usage:
    Called via Snakemake rule extract_FES_building_block_definitions
    or standalone: python extract_building_block_definitions.py --year 2024
"""

import requests
import pandas as pd
import io
import time
import logging
import yaml
import os
import sys
import argparse
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Configure logging
try:
    from scripts.utilities.logging_config import setup_logging
    logger = setup_logging("extract_building_block_definitions")
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("extract_building_block_definitions")

# Retry configuration
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)


def load_api_config(config_path: str = None) -> dict:
    """Load FES API configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join('data', 'FES', 'FES_api_urls.yaml')

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def fetch_building_block_definitions(fes_year: int, config: dict) -> pd.DataFrame:
    """
    Fetch building block definitions from NESO API.

    Args:
        fes_year: Year of FES data (2020-2025)
        config: API configuration dictionary

    Returns:
        DataFrame with building block definitions
    """
    url = config.get('building_block_definitions', {}).get(fes_year)

    if not url:
        logger.error(f"Building block definitions URL not found for year {fes_year}")
        raise ValueError(f"No API URL configured for FES {fes_year} building block definitions")

    logger.info(f"Fetching building block definitions for FES {fes_year}")
    logger.info(f"URL: {url}")

    # Polite delay before request
    time.sleep(1)

    try:
        response = http.get(url, timeout=30)
        response.raise_for_status()

        # Parse CSV content
        df = pd.read_csv(
            io.BytesIO(response.content),
            encoding='utf-8-sig',  # Handle BOM
            low_memory=False
        )

        logger.info(f"Downloaded {len(df)} building block definitions")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download building block definitions: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise


def clean_definitions(df: pd.DataFrame, fes_year: int) -> pd.DataFrame:
    """
    Clean and standardize building block definitions.

    Args:
        df: Raw definitions DataFrame
        fes_year: Year of FES data

    Returns:
        Cleaned DataFrame
    """
    # Handle year-specific header quirks
    if fes_year in [2021, 2022]:
        # Some years have the first row as headers
        if df.iloc[0].astype(str).str.contains('Building Block', case=False).any():
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # Remove any completely empty rows
    df = df.dropna(how='all')

    # Ensure Building Block ID Number column exists
    id_col = None
    for col in df.columns:
        if 'building block' in col.lower() and 'id' in col.lower():
            id_col = col
            break

    if id_col and id_col != 'Building Block ID Number':
        df = df.rename(columns={id_col: 'Building Block ID Number'})

    logger.info(f"Cleaned definitions: {len(df)} rows")

    return df


def add_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a category column based on building block ID prefix.

    Categories:
    - Demand: Dem_BB*
    - Generation: Gen_BB*
    - Low Carbon Technology: Lct_BB*
    - Storage & Flexibility: Srg_BB*
    """
    def get_category(bb_id):
        if pd.isna(bb_id):
            return 'Unknown'
        bb_id = str(bb_id)
        if bb_id.startswith('Dem_'):
            return 'Demand'
        elif bb_id.startswith('Gen_'):
            return 'Generation'
        elif bb_id.startswith('Lct_'):
            return 'Low Carbon Technology'
        elif bb_id.startswith('Srg_'):
            return 'Storage & Flexibility'
        else:
            return 'Other'

    if 'Building Block ID Number' in df.columns:
        df['Category'] = df['Building Block ID Number'].apply(get_category)
        # Move Category column to be second
        cols = df.columns.tolist()
        if 'Category' in cols:
            cols.remove('Category')
            cols.insert(1, 'Category')
            df = df[cols]

    return df


def save_definitions(df: pd.DataFrame, output_path: str) -> None:
    """Save definitions to CSV file."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved building block definitions to: {output_path}")

    # Print summary
    if 'Category' in df.columns:
        logger.info("Building blocks by category:")
        for cat, count in df['Category'].value_counts().items():
            logger.info(f"  {cat}: {count}")


def main(fes_year: int, config_path: str = None, output_path: str = None):
    """
    Main function to extract and save building block definitions.

    Args:
        fes_year: Year of FES data
        config_path: Path to API configuration YAML
        output_path: Path to save output CSV
    """
    logger.info("=" * 80)
    logger.info(f"EXTRACTING FES {fes_year} BUILDING BLOCK DEFINITIONS")
    logger.info("=" * 80)

    # Load configuration
    config = load_api_config(config_path)

    # Fetch definitions
    df = fetch_building_block_definitions(fes_year, config)

    # Clean data
    df = clean_definitions(df, fes_year)

    # Add category column
    df = add_category_column(df)

    # Save output
    if output_path is None:
        output_path = f"resources/FES/building_block_definitions_{fes_year}.csv"

    save_definitions(df, output_path)

    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)

    return df


if __name__ == "__main__":
    # Check if running from Snakemake
    snk = globals().get('snakemake')

    if snk:
        # Running from Snakemake
        if hasattr(snk, 'log') and snk.log:
            logger = setup_logging(snk.log[0])

        fes_year = snk.params.fes_year
        config_path = snk.input.config_file
        output_path = snk.output.definitions

        try:
            main(fes_year, config_path, output_path)
        except Exception as e:
            logger.exception(f"Failed to extract building block definitions: {e}")
            raise
    else:
        # Running standalone
        parser = argparse.ArgumentParser(
            description="Extract FES Building Block Definitions"
        )
        parser.add_argument(
            "--year", "-y",
            type=int,
            default=2024,
            help="FES year (default: 2024)"
        )
        parser.add_argument(
            "--config", "-c",
            type=str,
            default="data/FES/FES_api_urls.yaml",
            help="Path to API config YAML"
        )
        parser.add_argument(
            "--output", "-o",
            type=str,
            default=None,
            help="Output path for definitions CSV"
        )

        args = parser.parse_args()

        if args.output is None:
            args.output = f"resources/FES/building_block_definitions_{args.year}.csv"

        main(args.year, args.config, args.output)
