"""
TEC vs REPD Deduplication Script - Simplified Approach

This script removes renewable technologies from the TEC register to avoid duplication
with the REPD (Renewable Energy Planning Database) which provides better renewable
generation profiles and site data.

Strategy: Filter out renewable plant types from TEC register, keeping only conventional
thermal plants (gas, coal, nuclear, etc.) that are not covered by REPD.

Author: PyPSA-GB Development Team  
Date: September 2025
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import re

# Configure logging with centralized config and graceful fallbacks
logger = None
try:
    # Prefer local import when running from repo root
    from logging_config import setup_logging, log_dataframe_info
    logger = setup_logging("deduplicate_tec_repd")
except Exception:
    try:
        # Fallback for module-style path
        from scripts.utilities.logging_config import setup_logging, log_dataframe_info
        logger = setup_logging("deduplicate_tec_repd")
    except Exception:
        # Last resort basic config
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("deduplicate_tec_repd")
        def log_dataframe_info(df, logger, name: str):
            logger.info(f"{name} shape: {df.shape}")

# Define renewable technologies to remove from TEC register
RENEWABLE_PLANT_TYPES = {
    # Solar technologies
    'PV Array (Photo Voltaic/solar)',
    'Solar PV',
    'Solar',
    'Photovoltaic',
    'PV',
    
    # Wind technologies  
    'Wind Onshore',
    'Wind Offshore', 
    'Wind',
    'Wind Turbine',
    'Onshore Wind',
    'Offshore Wind',
    
    # Hydro (small scale renewables)
    'Hydro',
    'Run of River Hydro',
    'Small Hydro',
    'Micro Hydro',
    'Pumped Storage', # Can be debated - sometimes considered storage
    
    # Other renewables
    'Biomass',
    'Biogas',
    'Landfill Gas',
    'Energy from Waste',
    'Anaerobic Digestion',
    'Wave',
    'Tidal',
    'Geothermal'
}

# Plant types that contain renewables in combination (e.g., "Energy Storage System;PV Array")
RENEWABLE_KEYWORDS = {
    'pv', 'solar', 'wind', 'hydro', 'biomass', 'biogas', 'tidal', 'wave', 'geothermal'
}

def normalize_plant_type(plant_type: str) -> str:
    """Normalize plant type string for comparison."""
    if pd.isna(plant_type):
        return ""
    return str(plant_type).lower().strip()

def is_renewable_plant(plant_type: str) -> bool:
    """
    Determine if a plant type is renewable and should be removed from TEC register.
    
    Args:
        plant_type: Plant type string from TEC register
        
    Returns:
        True if plant should be removed (is renewable), False if should be kept
    """
    if pd.isna(plant_type):
        return False
        
    normalized = normalize_plant_type(plant_type)
    
    # Check exact matches first
    if plant_type in RENEWABLE_PLANT_TYPES:
        return True
        
    # Check for renewable keywords in the plant type
    for keyword in RENEWABLE_KEYWORDS:
        if keyword in normalized:
            return True
            
    return False

def load_tec_register(tec_file_path: str) -> pd.DataFrame:
    """Load TEC register CSV file."""
    logger.info(f"Loading TEC register from {tec_file_path}")
    
    if not Path(tec_file_path).exists():
        raise FileNotFoundError(f"TEC register file not found: {tec_file_path}")
        
    tec_df = pd.read_csv(tec_file_path)
    logger.info(f"Loaded {len(tec_df)} entries from TEC register")
    try:
        log_dataframe_info(tec_df, logger, "TEC register")
    except Exception:
        pass
    
    return tec_df

def filter_conventional_plants(tec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter TEC register to keep only conventional (non-renewable) plants.
    
    Args:
        tec_df: TEC register DataFrame
        
    Returns:
        Filtered DataFrame with only conventional plants
    """
    # Validate required column
    if 'Plant Type' not in tec_df.columns:
        logger.warning("'Plant Type' column not found in TEC register; skipping deduplication")
        return tec_df.copy()

    initial_count = len(tec_df)

    # Create mask for conventional (non-renewable) plants
    conventional_mask = ~tec_df['Plant Type'].apply(is_renewable_plant)
    
    # Filter the dataframe
    conventional_df = tec_df[conventional_mask].copy()
    
    removed_count = initial_count - len(conventional_df)
    removal_pct = (removed_count / initial_count * 100.0) if initial_count else 0.0
    
    logger.info(
        "TEC dedup: removed %d renewable plants (%.1f%%), kept %d conventional",
        removed_count, removal_pct, len(conventional_df)
    )
    
    # Log breakdown of removed technologies
    if removed_count > 0:
        removed_plants = tec_df[~conventional_mask]
        removed_breakdown = removed_plants['Plant Type'].fillna('Unknown').str.strip().value_counts()
        logger.debug("Top removed plant types: %s", dict(removed_breakdown.head(10)))
    
    return conventional_df

def save_deduplication_report(tec_df: pd.DataFrame, conventional_df: pd.DataFrame, 
                             output_path: str = "resources/tec_repd_matches.csv"):
    """
    Save a deduplication report showing what was removed.
    
    Args:
        tec_df: Original TEC register
        conventional_df: Filtered conventional plants
        output_path: Output file path for the report
    """
    # Create a simple report 
    removed_mask = ~tec_df.index.isin(conventional_df.index)
    removed_plants = tec_df[removed_mask].copy()
    
    if len(removed_plants) > 0:
        # Create report of removed renewable plants
        report_data = []
        
        for idx, plant in removed_plants.iterrows():
            report_data.append({
                'TEC_row_id': idx,
                'Project_Name': plant.get('Project Name', ''),
                'Plant_Type': plant.get('Plant Type', ''),
                'MW_Connected': plant.get('MW Connected', 0),
                'REPD_row_id': 'N/A - Renewable',
                'match_score': 1.0,
                'reason': 'renewable_plant_type',
                'action_taken': 'removed_from_tec'
            })
            
        report_df = pd.DataFrame(report_data)
    else:
        # Empty report if no renewables found
        report_df = pd.DataFrame(columns=[
            'TEC_row_id', 'Project_Name', 'Plant_Type', 'MW_Connected', 
            'REPD_row_id', 'match_score', 'reason', 'action_taken'
        ])
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save report
    report_df.to_csv(output_path, index=False)
    try:
        log_dataframe_info(report_df, logger, "Dedup report")
    except Exception:
        pass
    logger.info(f"Saved deduplication report to {output_path}")
    
    return report_df

def save_conventional_tec_register(conventional_df: pd.DataFrame, 
                                  output_path: str = "resources/tec_conventional_only.csv"):
    """Save the filtered conventional-only TEC register."""
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save filtered dataset
    conventional_df.to_csv(output_path, index=False)
    try:
        log_dataframe_info(conventional_df, logger, "Conventional TEC register")
    except Exception:
        pass
    logger.info(f"Saved conventional TEC register to {output_path}")

def save_references(output_path: str = "resources/generators/references_used_dedup.json"):
    """Save references for the deduplication approach."""
    references = {
        "tec_repd_deduplication": {
            "approach": "Remove renewable plant types from TEC register",
            "rationale": "REPD provides superior renewable generation profiles and site data",
            "renewable_definitions": "Based on UK renewable energy classifications",
            "date_applied": "2025-09-04",
            "script": "scripts/deduplicate_tec_repd.py"
        }
    }
    
    # Load existing references if they exist
    if Path(output_path).exists():
        try:
            with open(output_path, 'r') as f:
                existing_refs = json.load(f)
            existing_refs.update(references)
            references = existing_refs
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Could not load existing references from {output_path}, creating new file")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save references
    with open(output_path, 'w') as f:
        json.dump(references, f, indent=2)
    
    logger.info(f"Saved references to {output_path}")

def main(tec_file_path: str = "data/generators/tec-register-02-september-2025.csv",
         output_report: str = "resources/generators/tec_repd_matches.csv",
         output_conventional: str = "resources/generators/tec_conventional_only.csv",
         output_references: str = "resources/generators/references_used_dedup.json"):
    """
    Main deduplication function - removes renewables from TEC register.
    
    Args:
        tec_file_path: Path to TEC register CSV
        output_report: Path for deduplication report
        output_conventional: Path for filtered conventional TEC register
    """
    logger.info("Starting TEC/REPD deduplication (remove renewables approach)")
    
    # Load TEC register
    tec_df = load_tec_register(tec_file_path)
    
    # Filter to keep only conventional plants
    conventional_df = filter_conventional_plants(tec_df)
    
    # Save deduplication report
    save_deduplication_report(tec_df, conventional_df, output_report)
    
    # Save conventional-only TEC register
    save_conventional_tec_register(conventional_df, output_conventional)
    
    # Save references
    save_references(output_references)
    
    logger.info("TEC/REPD deduplication completed successfully")
    
    # Return summary stats
    return {
        'total_tec_entries': len(tec_df),
        'conventional_entries': len(conventional_df),
        'renewable_entries_removed': len(tec_df) - len(conventional_df),
        'removal_percentage': ((len(tec_df) - len(conventional_df)) / len(tec_df)) * 100
    }

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    if 'snakemake' in globals():
        # Snakemake execution - use input/output objects
        tec_file = snakemake.input.tec_file
        output_report = snakemake.output.matches
        output_conventional = snakemake.output.conventional_tec
        output_references = snakemake.output.citations
    else:
        # Command line execution - use sys.argv
        tec_file = sys.argv[1] if len(sys.argv) > 1 else "data/generators/tec-register-02-september-2025.csv"
        output_report = sys.argv[2] if len(sys.argv) > 2 else "resources/generators/tec_repd_matches.csv" 
        output_conventional = sys.argv[3] if len(sys.argv) > 3 else "resources/generators/tec_conventional_only.csv"
        output_references = sys.argv[4] if len(sys.argv) > 4 else "resources/generators/references_used_dedup.json"
    
    stats = main(tec_file, output_report, output_conventional, output_references)
    
    logger.info("Deduplication Summary:")
    logger.info("Total TEC entries: %d", stats['total_tec_entries'])
    logger.info("Conventional plants kept: %d", stats['conventional_entries'])
    logger.info("Renewable plants removed: %d", stats['renewable_entries_removed'])
    logger.info("Removal percentage: %.1f%%", stats['removal_percentage'])

