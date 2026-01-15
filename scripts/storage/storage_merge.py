#!/usr/bin/env python3
"""
Merge storage data from REPD and external sources.

This script combines storage data from REPD extraction and external datasets,
applying intelligent deduplication and data prioritization rules.

Key functions:
- Merge REPD and external storage datasets
- Intelligent deduplication by site name and technology
- Prioritize external data for capacity where both sources exist
- Maintain data lineage and source tracking

Deduplication rules:
1. Match by (site_name, technology, rounded coordinates)
2. Prefer external data for capacity_mw if available
3. Keep energy_mwh from any source
4. Maintain highest quality coordinate data

Author: PyPSA-GB Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# Set up logging - compatible with both standalone and Snakemake execution
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("storage_merge")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

def normalize_site_name(name_str):
    """
    Normalize site names for better matching.
    
    Args:
        name_str: Original site name
        
    Returns:
        Normalized site name
    """
    if pd.isna(name_str):
        return 'unknown'
    
    # Convert to lowercase and remove common words
    normalized = str(name_str).lower().strip()
    
    # Remove common suffixes/prefixes
    remove_words = [
        'power station', 'power plant', 'energy storage', 'battery storage',
        'facility', 'site', 'project', 'development', 'phase 1', 'phase 2',
        'ltd', 'limited', 'plc', 'inc', 'llc'
    ]
    
    for word in remove_words:
        normalized = normalized.replace(word, '').strip()
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized

def create_matching_key(row):
    """
    Create a key for matching duplicate sites.
    
    Args:
        row: DataFrame row
        
    Returns:
        Tuple for matching
    """
    site_norm = normalize_site_name(row['site_name'])
    tech_norm = str(row['technology']).lower().strip()
    
    # Round coordinates for fuzzy matching (to ~100m precision)
    lat_round = round(row['lat'], 3) if pd.notna(row['lat']) else None
    lon_round = round(row['lon'], 3) if pd.notna(row['lon']) else None
    
    return (site_norm, tech_norm, lat_round, lon_round)

def merge_duplicate_records(group):
    """
    Merge records that are identified as duplicates.
    
    Args:
        group: Group of duplicate records
        
    Returns:
        Single merged record
    """
    if len(group) == 1:
        return group.iloc[0]
    
    logger.debug(f"Merging {len(group)} duplicate records for {group.iloc[0]['site_name']}")
    
    # Initialize result with first record
    result = group.iloc[0].copy()
    
    # Data prioritization rules
    for _, row in group.iterrows():
        # Prefer external data for capacity
        if row['source'] != 'REPD' and pd.notna(row['capacity_mw']) and row['capacity_mw'] > 0:
            if pd.isna(result['capacity_mw']) or row['capacity_mw'] > result['capacity_mw']:
                result['capacity_mw'] = row['capacity_mw']
                result['source'] = f"{result['source']};{row['source']}"
        
        # Keep energy data if missing
        if pd.isna(result['energy_mwh']) and pd.notna(row['energy_mwh']):
            result['energy_mwh'] = row['energy_mwh']
        
        # Prefer coordinates with higher precision or from external sources
        if pd.isna(result['lat']) or (pd.notna(row['lat']) and row['source'] != 'REPD'):
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                result['lat'] = row['lat']
                result['lon'] = row['lon']
        
        # Update commissioning year if missing
        if pd.isna(result['commissioning_year']) and pd.notna(row['commissioning_year']):
            result['commissioning_year'] = row['commissioning_year']
        
        # Combine status information
        if pd.notna(row['status']) and row['status'] != result['status']:
            if pd.isna(result['status']):
                result['status'] = row['status']
            elif row['status'] not in str(result['status']):
                result['status'] = f"{result['status']};{row['status']}"
    
    return result

def validate_merged_data(df):
    """
    Validate the merged storage data.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        Validated DataFrame
    """
    logger.info(f"Validating {len(df)} merged storage records...")
    
    initial_count = len(df)
    
    # Remove records with zero or negative capacity
    valid_capacity = (df['capacity_mw'] > 0) | df['capacity_mw'].isna()
    df = df[valid_capacity]
    
    # Validate coordinates
    valid_coords = (
        ((df['lat'] >= 49.0) & (df['lat'] <= 61.5) & 
         (df['lon'] >= -9.0) & (df['lon'] <= 2.5)) |
        df['lat'].isna()
    )
    df = df[valid_coords]
    
    # Clean up source field (remove duplicates)
    def clean_source(source_str):
        if pd.isna(source_str):
            return 'Unknown'
        sources = str(source_str).split(';')
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
        return ';'.join(unique_sources)
    
    df['source'] = df['source'].apply(clean_source)
    
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} invalid records during validation")
    
    logger.info(f"Validation complete: {len(df)} valid storage sites")
    return df

def main():
    """Main function to merge storage data sources."""
    start_time = time.time()
    logger.info("Starting storage data merge...")
    
    try:
        # Get input and output files
        try:
            # Snakemake mode
            repd_file = snakemake.input.repd_storage
            tec_file = snakemake.input.get('tec_storage', None)  # Optional TEC storage
            external_file = snakemake.input.external_storage
            output_file = snakemake.output.merged_storage
            logger.info("Running in Snakemake mode")
        except NameError:
            # Standalone mode
            base_path = Path(__file__).parent.parent.parent
            repd_file = base_path / "resources" / "storage" / "storage_from_repd.csv"
            tec_file = base_path / "resources" / "storage" / "storage_from_tec.csv"
            external_file = base_path / "resources" / "storage" / "storage_external.csv"
            output_file = base_path / "resources" / "storage" / "storage_sites_merged.csv"
            logger.info("Running in standalone mode")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load REPD storage data
        logger.info(f"Loading REPD storage data from: {repd_file}")
        try:
            repd_df = pd.read_csv(repd_file)
            repd_df['source'] = 'REPD'
            logger.info(f"Loaded {len(repd_df)} REPD storage sites")
        except FileNotFoundError:
            logger.warning(f"REPD storage file not found: {repd_file}")
            repd_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading REPD storage data: {e}")
            repd_df = pd.DataFrame()
        
        # Load TEC storage data (optional)
        tec_df = pd.DataFrame()
        if tec_file and Path(tec_file).exists():
            logger.info(f"Loading TEC storage data from: {tec_file}")
            try:
                tec_df = pd.read_csv(tec_file)
                logger.info(f"Loaded {len(tec_df)} TEC storage sites")
            except Exception as e:
                logger.error(f"Error loading TEC storage data: {e}")
                tec_df = pd.DataFrame()
        else:
            logger.info("TEC storage file not provided or not found - skipping")
        
        # Load external storage data
        logger.info(f"Loading external storage data from: {external_file}")
        try:
            external_df = pd.read_csv(external_file)
            logger.info(f"Loaded {len(external_df)} external storage sites")
        except FileNotFoundError:
            logger.warning(f"External storage file not found: {external_file}")
            external_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading external storage data: {e}")
            external_df = pd.DataFrame()
        
        # Check if we have any data to merge
        if len(repd_df) == 0 and len(tec_df) == 0 and len(external_df) == 0:
            logger.warning("No storage data found - creating empty output file")
            empty_df = pd.DataFrame(columns=[
                'site_name', 'technology', 'capacity_mw', 'energy_mwh', 'lat', 'lon', 
                'status', 'commissioning_year', 'source'
            ])
            empty_df.to_csv(output_file, index=False)
            return
        
        # Ensure consistent columns across datasets
        required_columns = [
            'site_name', 'technology', 'capacity_mw', 'energy_mwh', 'lat', 'lon',
            'status', 'commissioning_year', 'source'
        ]
        
        for df_name, df in [('REPD', repd_df), ('TEC', tec_df), ('External', external_df)]:
            if len(df) > 0:
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                        logger.info(f"Added missing column '{col}' to {df_name} data")
        
        # Combine datasets
        dataframes_to_merge = []
        if len(repd_df) > 0:
            dataframes_to_merge.append(repd_df[required_columns])
        if len(tec_df) > 0:
            dataframes_to_merge.append(tec_df[required_columns])
        if len(external_df) > 0:
            dataframes_to_merge.append(external_df[required_columns])
        
        if len(dataframes_to_merge) > 1:
            combined_df = pd.concat(dataframes_to_merge, ignore_index=True)
            logger.info(f"Combined datasets: {len(repd_df)} REPD + {len(tec_df)} TEC + {len(external_df)} external = {len(combined_df)} total")
        elif len(dataframes_to_merge) == 1:
            combined_df = dataframes_to_merge[0].copy()
            source_name = "REPD" if len(repd_df) > 0 else ("TEC" if len(tec_df) > 0 else "external")
            logger.info(f"Using {source_name} data only: {len(combined_df)} sites")
        else:
            logger.warning("No data to merge")
            empty_df = pd.DataFrame(columns=required_columns)
            empty_df.to_csv(output_file, index=False)
            return
        
        # Create matching keys for deduplication
        logger.info("Creating matching keys for deduplication...")
        combined_df['_match_key'] = combined_df.apply(create_matching_key, axis=1)
        
        # Group by matching key and merge duplicates
        logger.info("Identifying and merging duplicate records...")
        grouped = combined_df.groupby('_match_key', dropna=False)
        
        merged_records = []
        duplicates_found = 0
        
        for match_key, group in grouped:
            if len(group) > 1:
                duplicates_found += len(group) - 1
                merged_record = merge_duplicate_records(group)
                merged_records.append(merged_record)
            else:
                merged_records.append(group.iloc[0])
        
        # Create final DataFrame
        merged_df = pd.DataFrame(merged_records)
        merged_df = merged_df.drop('_match_key', axis=1)
        
        logger.info(f"Deduplication complete: {duplicates_found} duplicates merged, {len(merged_df)} unique sites")
        
        # Validate merged data
        final_df = validate_merged_data(merged_df)
        
        # Ensure proper column order
        final_df = final_df[required_columns]
        
        # Save results
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(final_df)} merged storage sites to: {output_file}")
        
        # Generate summary statistics
        if len(final_df) > 0:
            # Technology summary
            tech_summary = final_df.groupby('technology').agg({
                'capacity_mw': ['count', 'sum'],
                'energy_mwh': lambda x: x.notna().sum(),
                'lat': lambda x: x.notna().sum()
            }).round(1)
            
            logger.info("Merged storage technology summary:")
            for tech in tech_summary.index:
                count = int(tech_summary.loc[tech, ('capacity_mw', 'count')])
                capacity = tech_summary.loc[tech, ('capacity_mw', 'sum')]
                with_energy = int(tech_summary.loc[tech, ('energy_mwh', '<lambda>')])
                with_coords = int(tech_summary.loc[tech, ('lat', '<lambda>')])
                logger.info(f"  {tech}: {count} sites, {capacity:.1f} MW, {with_energy} with energy, {with_coords} with coords")
            
            # Source summary
            source_summary = final_df['source'].value_counts()
            logger.info("Data source summary:")
            for source, count in source_summary.items():
                logger.info(f"  {source}: {count} sites")
        
    # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'repd_sites': len(repd_df) if len(repd_df) > 0 else 0,
            'tec_sites': len(tec_df) if len(tec_df) > 0 else 0,
            'external_sites': len(external_df) if len(external_df) > 0 else 0,
            'combined_sites': len(combined_df) if 'combined_df' in locals() else 0,
            'duplicates_merged': duplicates_found,
            'final_unique_sites': len(final_df),
            'total_capacity_mw': final_df['capacity_mw'].sum() if len(final_df) > 0 else 0,
            'sites_with_coordinates': final_df['lat'].notna().sum() if len(final_df) > 0 else 0,
            'output_file': str(output_file)
        }
        
        log_execution_summary(logger, "storage_merge", execution_time, summary_stats)
        logger.info("Storage data merge completed successfully!")
        # Write human-readable merge report and JSON quality summary if requested by Snakemake
        try:
            merge_report_path = snakemake.output.merge_report
            quality_summary_path = snakemake.output.quality_summary
        except Exception:
            merge_report_path = Path(output_file).parent / "merge_report.txt"
            quality_summary_path = Path(output_file).parent / "data_quality_summary.json"

        try:
            # Write a brief merge report
            with open(merge_report_path, 'w', encoding='utf-8') as f:
                f.write("Storage Merge Report\n")
                f.write("====================\n")
                f.write(f"Merged datasets: {len(combined_df)} entries before deduplication\n")
                f.write(f"Duplicates merged: {duplicates_found}\n")
                f.write(f"Final unique sites: {len(final_df)}\n")
                f.write(f"Total capacity (MW): {summary_stats['total_capacity_mw']}\n")

            # Write JSON quality summary
            import json
            # Convert any numpy/pandas scalars to native Python types for JSON
            def to_py(o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                return o

            serializable_stats = {k: to_py(v) for k, v in summary_stats.items()}
            with open(quality_summary_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2)

            logger.info(f"Wrote merge report to: {merge_report_path}")
            logger.info(f"Wrote data quality summary to: {quality_summary_path}")
        except Exception as e:
            logger.warning(f"Could not write merge/quality reports: {e}")
        
    except Exception as e:
        logger.error(f"Error in storage data merge: {e}")
        raise

if __name__ == "__main__":
    main()

