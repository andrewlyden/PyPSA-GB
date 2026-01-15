"""
Download European electricity generation mix data from NESO FES API.

This script downloads generation technology mix data for European countries
that GB has interconnectors with. The data is used to estimate cross-border
electricity prices and inform interconnector flow modeling.

Author: PyPSA-GB
License: MIT
"""

import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utilities.logging_config import setup_logging, log_execution_summary

# Setup logging
logger = setup_logging(snakemake.log[0] if "snakemake" in dir() else "download_european_generation.log")


def fetch_dataset_info(api_endpoint: str, dataset_id: str) -> dict:
    """
    Fetch dataset information from NESO API.
    
    Args:
        api_endpoint: API base URL
        dataset_id: Dataset identifier
        
    Returns:
        dict: Dataset metadata including resource URLs
    """
    logger.info(f"Fetching dataset info for: {dataset_id}")
    
    url = f"{api_endpoint}?id={dataset_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("success"):
            raise ValueError(f"API request failed: {data.get('error', 'Unknown error')}")
        
        logger.info("✓ Successfully fetched dataset metadata")
        return data["result"]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch dataset info: {e}")
        raise


def download_generation_data(resource_url: str, fes_year: str) -> pd.DataFrame:
    """
    Download generation mix CSV from NESO API.
    
    Args:
        resource_url: Direct URL to CSV resource
        fes_year: FES year to download
        
    Returns:
        pd.DataFrame: Generation mix data
    """
    logger.info(f"Downloading European generation data for {fes_year}")
    logger.info(f"URL: {resource_url[:100]}...")
    
    try:
        # Download CSV
        response = requests.get(resource_url, timeout=60)
        response.raise_for_status()
        
        # Read into DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        logger.info(f"✓ Downloaded {len(df)} rows")
        logger.info(f"  Columns: {', '.join(df.columns.tolist()[:10])}...")
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download generation data: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate downloaded data structure and contents.
    
    Args:
        df: Downloaded generation mix DataFrame
    """
    logger.info("Validating data structure...")
    
    # Check for expected columns (will vary by FES version)
    required_cols = ["Country", "Year"]  # Minimal requirements
    
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Expected column '{col}' not found in dataset")
    
    # Check for data completeness
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.info("Null value counts by column:")
        for col, count in null_counts[null_counts > 0].items():
            logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check year range
    if "Year" in df.columns:
        years = df["Year"].unique()
        logger.info(f"  Year range: {years.min()} - {years.max()}")
    
    # Check countries
    if "Country" in df.columns:
        countries = df["Country"].unique()
        logger.info(f"  Countries ({len(countries)}): {', '.join(sorted(countries)[:10])}...")
    
    logger.info("✓ Data validation complete")


def save_metadata(dataset_info: dict, resource_info: dict, output_path: Path) -> None:
    """
    Save metadata about the downloaded dataset.
    
    Args:
        dataset_info: Full dataset metadata
        resource_info: Specific resource metadata
        output_path: Path to save metadata JSON
    """
    metadata = {
        "download_timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_info.get("id"),
        "dataset_name": dataset_info.get("name"),
        "dataset_title": dataset_info.get("title"),
        "resource_name": resource_info.get("name"),
        "resource_format": resource_info.get("format"),
        "resource_url": resource_info.get("path"),
        "resource_created": resource_info.get("created"),
        "resource_last_modified": resource_info.get("last_modified"),
        "api_version": "3"
    }
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Metadata saved to {output_path}")


def main():
    """Main execution function."""
    start_time = time.time()
    
    # Get parameters from Snakemake
    api_endpoint = snakemake.params.api_endpoint
    dataset_id = snakemake.params.dataset_id
    fes_year = snakemake.wildcards.fes_year
    
    output_csv = snakemake.output.generation_mix
    output_metadata = snakemake.output.metadata
    
    logger.info("=" * 80)
    logger.info("NESO European Generation Mix Download")
    logger.info("=" * 80)
    logger.info(f"FES Year: {fes_year}")
    logger.info(f"API Endpoint: {api_endpoint}")
    logger.info(f"Dataset ID: {dataset_id}")
    
    # Fetch dataset info
    dataset_info = fetch_dataset_info(api_endpoint, dataset_id)
    
    # Find resource for requested year
    resources = dataset_info.get("resources", [])
    logger.info(f"Found {len(resources)} resources in dataset")
    
    # Match by year in resource name
    matching_resource = None
    for resource in resources:
        resource_name = resource.get("name", "")
        if fes_year in resource_name:
            matching_resource = resource
            logger.info(f"✓ Found matching resource: {resource_name}")
            break
    
    if not matching_resource:
        # Try finding by year in filename
        for resource in resources:
            resource_path = resource.get("path", "")
            if fes_year in resource_path:
                matching_resource = resource
                logger.info(f"✓ Found matching resource by path: {resource_path}")
                break
    
    if not matching_resource:
        logger.error(f"No resource found for FES year {fes_year}")
        logger.info("Available resources:")
        for resource in resources:
            logger.info(f"  - {resource.get('name')}")
        raise ValueError(f"No European generation data found for year {fes_year}")
    
    # Download data
    resource_url = matching_resource.get("path")
    df = download_generation_data(resource_url, fes_year)
    
    # Validate
    validate_data(df)
    
    # Save CSV
    logger.info(f"Saving generation mix to {output_csv}")
    df.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved {len(df)} rows to {output_csv}")
    
    # Save metadata
    save_metadata(dataset_info, matching_resource, output_metadata)
    
    # Calculate statistics
    records_downloaded = len(df)
    countries = df['country'].nunique() if 'country' in df.columns else 0
    years_covered = df['year'].nunique() if 'year' in df.columns else 1
    
    # Log execution summary
    log_execution_summary(
        logger,
        "download_european_generation",
        start_time,
        inputs={'api': api_endpoint, 'dataset': dataset_id},
        outputs={'generation_mix': output_csv, 'metadata': output_metadata},
        context={
            'records_downloaded': records_downloaded,
            'countries': countries,
            'years': years_covered,
            'fes_year': fes_year
        }
    )


if __name__ == "__main__":
    if "snakemake" not in dir():
        logger.error("This script must be run via Snakemake")
        sys.exit(1)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Failed to download European generation data: {e}")
        raise

