#!/usr/bin/env python3
"""
Storage Integration Validation Script

This script validates that storage is being correctly extracted and integrated
into the PyPSA-GB network. It checks:

1. Storage extraction from REPD
2. Storage extraction from TEC Register
3. External storage data processing
4. Storage parameter building
5. Storage integration into PyPSA network
6. Comparison with expected storage capacity

Usage:
    python scripts/validate_storage_integration.py

Author: PyPSA-GB Team
"""

import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_repd_storage():
    """Check storage extraction from REPD."""
    logger.info("="*80)
    logger.info("1. CHECKING REPD STORAGE EXTRACTION")
    logger.info("="*80)
    
    repd_file = Path("data/renewables/repd-q2-jul-2025.csv")
    if not repd_file.exists():
        logger.error(f"REPD file not found: {repd_file}")
        return None
    
    # Load full REPD (try different encodings)
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    repd_df = None
    for encoding in encodings:
        try:
            repd_df = pd.read_csv(repd_file, encoding=encoding)
            logger.info(f"Loaded REPD with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if repd_df is None:
        logger.error("Could not load REPD file with any encoding")
        return None, None
    
    logger.info(f"Total REPD records: {len(repd_df)}")
    
    # Check for storage technologies
    storage_keywords = ['battery', 'storage', 'pumped', 'laes', 'caes', 'flywheel']
    storage_mask = repd_df['Technology Type'].str.contains('|'.join(storage_keywords), case=False, na=False)
    
    storage_in_repd = repd_df[storage_mask]
    logger.info(f"Storage entries in REPD: {len(storage_in_repd)}")
    
    # Technology breakdown
    logger.info("\nStorage technologies in REPD:")
    tech_counts = storage_in_repd['Technology Type'].value_counts()
    for tech, count in tech_counts.items():
        tech_data = storage_in_repd[storage_in_repd['Technology Type'] == tech]
        capacity = pd.to_numeric(tech_data['Installed Capacity (MWelec)'], errors='coerce').sum()
        logger.info(f"  {tech}: {count} sites, {capacity:.1f} MW")
    
    # Check processed file
    processed_file = Path("resources/storage/storage_from_repd.csv")
    if processed_file.exists():
        processed_df = pd.read_csv(processed_file)
        logger.info(f"\nProcessed REPD storage file: {len(processed_df)} sites")
        logger.info(f"Total capacity: {processed_df['capacity_mw'].sum():.1f} MW")
    else:
        logger.warning(f"Processed REPD storage file not found: {processed_file}")
        processed_df = None
    
    return storage_in_repd, processed_df

def check_tec_storage():
    """Check storage in TEC Register."""
    logger.info("\n" + "="*80)
    logger.info("2. CHECKING TEC REGISTER STORAGE")
    logger.info("="*80)
    
    tec_file = Path("data/generators/tec-register-02-september-2025.csv")
    if not tec_file.exists():
        logger.error(f"TEC file not found: {tec_file}")
        return None
    
    # Load TEC register
    tec_df = pd.read_csv(tec_file)
    logger.info(f"Total TEC records: {len(tec_df)}")
    
    # Check for storage
    storage_keywords = ['storage', 'battery', 'energy storage system', 'bess']
    storage_mask = tec_df['Plant Type'].str.contains('|'.join(storage_keywords), case=False, na=False)
    
    storage_in_tec = tec_df[storage_mask]
    logger.info(f"Storage entries in TEC: {len(storage_in_tec)}")
    
    # Technology breakdown
    if len(storage_in_tec) > 0:
        logger.info("\nStorage types in TEC:")
        tech_counts = storage_in_tec['Plant Type'].value_counts()
        for tech, count in tech_counts.head(10).items():
            tech_data = storage_in_tec[storage_in_tec['Plant Type'] == tech]
            capacity = pd.to_numeric(tech_data['MW Connected'], errors='coerce').sum()
            logger.info(f"  {tech}: {count} sites, {capacity:.1f} MW")
        
        # Show status breakdown
        logger.info("\nTEC Storage by status:")
        status_counts = storage_in_tec['Project Status'].value_counts()
        for status, count in status_counts.items():
            status_data = storage_in_tec[storage_in_tec['Project Status'] == status]
            capacity = pd.to_numeric(status_data['MW Connected'], errors='coerce').sum()
            logger.info(f"  {status}: {count} sites, {capacity:.1f} MW")
    
    return storage_in_tec

def check_external_storage():
    """Check external storage data sources."""
    logger.info("\n" + "="*80)
    logger.info("3. CHECKING EXTERNAL STORAGE DATA")
    logger.info("="*80)
    
    storage_dir = Path("data/storage")
    if not storage_dir.exists():
        logger.warning(f"Storage directory not found: {storage_dir}")
        return None
    
    # List storage files
    storage_files = list(storage_dir.glob("*.csv")) + list(storage_dir.glob("*.xlsx"))
    logger.info(f"Found {len(storage_files)} external storage files:")
    for f in storage_files:
        logger.info(f"  - {f.name}")
    
    # Check processed external file
    external_file = Path("resources/storage/storage_external.csv")
    if external_file.exists():
        external_df = pd.read_csv(external_file)
        logger.info(f"\nProcessed external storage: {len(external_df)} sites")
        if len(external_df) > 0:
            logger.info(f"Total capacity: {external_df['capacity_mw'].sum():.1f} MW")
    else:
        logger.warning(f"Processed external storage file not found: {external_file}")
        external_df = None
    
    return external_df

def check_merged_storage():
    """Check merged storage data."""
    logger.info("\n" + "="*80)
    logger.info("4. CHECKING MERGED STORAGE DATA")
    logger.info("="*80)
    
    merged_file = Path("resources/storage/storage_sites_merged.csv")
    if not merged_file.exists():
        logger.error(f"Merged storage file not found: {merged_file}")
        return None
    
    merged_df = pd.read_csv(merged_file)
    logger.info(f"Total merged storage sites: {len(merged_df)}")
    logger.info(f"Total capacity: {merged_df['capacity_mw'].sum():.1f} MW")
    
    # Technology breakdown
    logger.info("\nMerged storage by technology:")
    tech_summary = merged_df.groupby('technology').agg({
        'capacity_mw': ['count', 'sum']
    })
    for tech in tech_summary.index:
        count = int(tech_summary.loc[tech, ('capacity_mw', 'count')])
        capacity = tech_summary.loc[tech, ('capacity_mw', 'sum')]
        logger.info(f"  {tech}: {count} sites, {capacity:.1f} MW")
    
    # Source breakdown
    logger.info("\nMerged storage by source:")
    source_counts = merged_df['source'].value_counts()
    for source, count in source_counts.items():
        capacity = merged_df[merged_df['source'] == source]['capacity_mw'].sum()
        logger.info(f"  {source}: {count} sites, {capacity:.1f} MW")
    
    return merged_df

def check_storage_parameters():
    """Check storage parameters file."""
    logger.info("\n" + "="*80)
    logger.info("5. CHECKING STORAGE PARAMETERS")
    logger.info("="*80)
    
    params_file = Path("resources/storage/storage_parameters.csv")
    if not params_file.exists():
        logger.error(f"Storage parameters file not found: {params_file}")
        return None
    
    params_df = pd.read_csv(params_file)
    logger.info(f"Total storage sites with parameters: {len(params_df)}")
    logger.info(f"Total power capacity: {params_df['power_mw'].sum():.1f} MW")
    logger.info(f"Total energy capacity: {params_df['energy_mwh'].sum():.1f} MWh")
    
    # Calculate system average duration
    total_power = params_df['power_mw'].sum()
    total_energy = params_df['energy_mwh'].sum()
    avg_duration = total_energy / total_power if total_power > 0 else 0
    logger.info(f"System average duration: {avg_duration:.1f} hours")
    
    # Technology summary
    logger.info("\nStorage parameters by technology:")
    tech_summary = params_df.groupby('technology').agg({
        'power_mw': ['count', 'sum'],
        'energy_mwh': 'sum',
        'duration_h': 'mean',
        'rte': 'mean'
    })
    
    for tech in tech_summary.index:
        count = int(tech_summary.loc[tech, ('power_mw', 'count')])
        power = tech_summary.loc[tech, ('power_mw', 'sum')]
        energy = tech_summary.loc[tech, ('energy_mwh', 'sum')]
        duration = tech_summary.loc[tech, ('duration_h', 'mean')]
        rte = tech_summary.loc[tech, ('rte', 'mean')]
        logger.info(f"  {tech}:")
        logger.info(f"    - {count} sites, {power:.1f} MW, {energy:.1f} MWh")
        logger.info(f"    - Avg duration: {duration:.1f}h, RTE: {rte:.1%}")
    
    # Check for missing coordinates
    missing_coords = params_df[params_df['lat'].isna() | params_df['lon'].isna()]
    if len(missing_coords) > 0:
        logger.warning(f"\nWARNING: {len(missing_coords)} sites missing coordinates!")
    else:
        logger.info(f"\nAll {len(params_df)} sites have coordinates [OK]")
    
    return params_df

def check_network_integration(scenario='HT35_clustered_gsp'):
    """Check if storage is integrated into PyPSA network."""
    logger.info("\n" + "="*80)
    logger.info("6. CHECKING PYPSA NETWORK INTEGRATION")
    logger.info("="*80)
    
    network_file = Path(f"resources/network/{scenario}_with_storage.nc")
    if not network_file.exists():
        logger.warning(f"Network file not found: {network_file}")
        logger.warning("Storage integration into PyPSA network has NOT been implemented yet!")
        return None
    
    # Load network
    logger.info(f"Loading network from: {network_file}")
    n = pypsa.Network(str(network_file))
    
    # Check for storage units
    if hasattr(n, 'storage_units') and len(n.storage_units) > 0:
        logger.info(f"✓ Storage units found in network: {len(n.storage_units)}")
        logger.info(f"Total storage capacity: {n.storage_units['p_nom'].sum():.1f} MW")
        
        # Technology breakdown
        if 'carrier' in n.storage_units.columns:
            logger.info("\nStorage by carrier:")
            carrier_summary = n.storage_units.groupby('carrier')['p_nom'].agg(['count', 'sum'])
            for carrier in carrier_summary.index:
                count = int(carrier_summary.loc[carrier, 'count'])
                capacity = carrier_summary.loc[carrier, 'sum']
                logger.info(f"  {carrier}: {count} units, {capacity:.1f} MW")
        
        return n.storage_units
    else:
        logger.warning("✗ NO STORAGE UNITS FOUND IN NETWORK!")
        logger.warning("Storage has been extracted but NOT integrated into PyPSA network")
        logger.info(f"\nNetwork contains:")
        logger.info(f"  - Buses: {len(n.buses)}")
        logger.info(f"  - Generators: {len(n.generators)}")
        logger.info(f"  - Loads: {len(n.loads)}")
        logger.info(f"  - Storage Units: {len(n.storage_units) if hasattr(n, 'storage_units') else 0}")
        
        return None

def generate_summary_report():
    """Generate comprehensive summary report."""
    logger.info("\n" + "="*80)
    logger.info("STORAGE INTEGRATION SUMMARY REPORT")
    logger.info("="*80)
    
    repd_storage, processed_repd = check_repd_storage()
    tec_storage = check_tec_storage()
    external_storage = check_external_storage()
    merged_storage = check_merged_storage()
    params_storage = check_storage_parameters()
    network_storage = check_network_integration()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL ASSESSMENT")
    logger.info("="*80)
    
    issues = []
    
    # Check data extraction
    if processed_repd is not None and len(processed_repd) > 0:
        logger.info("✓ REPD storage extraction: WORKING")
    else:
        logger.warning("✗ REPD storage extraction: FAILED")
        issues.append("REPD storage extraction not working")
    
    if tec_storage is not None and len(tec_storage) > 0:
        logger.info(f"✓ TEC storage identified: {len(tec_storage)} sites")
        logger.warning("⚠ TEC storage NOT integrated into processing pipeline")
        issues.append("TEC storage exists but not being used")
    else:
        logger.info("ℹ TEC storage: None found or file not available")
    
    if params_storage is not None and len(params_storage) > 0:
        logger.info("✓ Storage parameters: BUILT")
    else:
        logger.warning("✗ Storage parameters: FAILED")
        issues.append("Storage parameters not built")
    
    if network_storage is not None and len(network_storage) > 0:
        logger.info("✓ Network integration: COMPLETE")
    else:
        logger.warning("✗ Network integration: NOT IMPLEMENTED")
        issues.append("Storage not integrated into PyPSA network")
    
    # Print issues
    if issues:
        logger.info("\n" + "="*80)
        logger.info("ISSUES FOUND:")
        logger.info("="*80)
        for i, issue in enumerate(issues, 1):
            logger.warning(f"{i}. {issue}")
    else:
        logger.info("\n✓ All storage integration checks passed!")
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*80)
    
    if "TEC storage exists but not being used" in issues:
        logger.info("1. Add TEC storage extraction script (similar to storage_from_repd.py)")
        logger.info("   - Extract storage from TEC register")
        logger.info("   - Merge with REPD storage data")
    
    if "Storage not integrated into PyPSA network" in issues:
        logger.info("2. Implement storage integration in add_generators.py or create separate script:")
        logger.info("   - Load storage_parameters.csv")
        logger.info("   - Map storage to nearest buses")
        logger.info("   - Add as StorageUnit components to PyPSA network")
        logger.info("   - Set p_nom (power capacity), max_hours (duration), efficiency parameters")
    
    logger.info("\n" + "="*80)

if __name__ == "__main__":
    generate_summary_report()

