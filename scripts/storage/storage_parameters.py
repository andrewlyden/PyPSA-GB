#!/usr/bin/env python3
"""
Build storage parameters for PyPSA integration.

This script takes merged storage site data and adds technology-specific
parameters needed for PyPSA modeling, including duration, efficiency,
and operational characteristics.

Key functions:
- Add default duration by technology type
- Calculate energy capacity from power and duration
- Set round-trip and charge/discharge efficiencies
- Validate and clean parameter data
- Prepare final storage parameter database

Technology defaults:
- Battery: 2h duration, 90% RTE
- Pumped Hydro: 6h duration, 80% RTE  
- LAES: 4h duration, 55% RTE
- CAES: 8h duration, 65% RTE
- Flywheel: 0.25h duration, 85% RTE

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
        logger = setup_logging("storage_parameters")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Technology-specific parameters
STORAGE_PARAMETERS = {
    'Battery': {
        'duration_h': 2.0,
        'rte': 0.90,  # Round-trip efficiency
        'min_capacity_factor': 0.0,
        'max_capacity_factor': 1.0,
        'capital_cost': 400,  # £/kW (placeholder)
        'marginal_cost': 0.1   # £/MWh (placeholder)
    },
    'Pumped Storage Hydroelectricity': {
        'duration_h': 6.0,
        'rte': 0.80,
        'min_capacity_factor': 0.0,
        'max_capacity_factor': 1.0,
        'capital_cost': 2000,  # £/kW (placeholder)
        'marginal_cost': 0.1
    },
    'Liquid Air Energy Storage': {
        'duration_h': 4.0,
        'rte': 0.55,
        'min_capacity_factor': 0.0,
        'max_capacity_factor': 1.0,
        'capital_cost': 800,  # £/kW (placeholder)
        'marginal_cost': 0.2
    },
    'Compressed Air Energy Storage': {
        'duration_h': 8.0,
        'rte': 0.65,
        'min_capacity_factor': 0.0,
        'max_capacity_factor': 1.0,
        'capital_cost': 600,  # £/kW (placeholder)
        'marginal_cost': 0.2
    },
    'Flywheel': {
        'duration_h': 0.25,
        'rte': 0.85,
        'min_capacity_factor': 0.0,
        'max_capacity_factor': 1.0,
        'capital_cost': 2500,  # £/kW (placeholder)
        'marginal_cost': 0.05
    },
    'Other Storage': {
        'duration_h': 4.0,
        'rte': 0.70,
        'min_capacity_factor': 0.0,
        'max_capacity_factor': 1.0,
        'capital_cost': 1000,  # £/kW (placeholder)
        'marginal_cost': 0.3
    }
}

def add_technology_defaults(df):
    """
    Add technology-specific default parameters.
    
    Args:
        df: Input DataFrame with storage sites
        
    Returns:
        DataFrame with technology defaults added
    """
    logger.info("Adding technology-specific default parameters...")
    
    # Initialize parameter columns
    param_columns = ['duration_h', 'rte', 'eta_charge', 'eta_discharge', 
                    'min_capacity_factor', 'max_capacity_factor', 
                    'capital_cost', 'marginal_cost']
    
    for col in param_columns:
        df[col] = np.nan
    
    defaults_applied = {}
    
    # Apply defaults by technology
    for tech, params in STORAGE_PARAMETERS.items():
        tech_mask = df['technology'] == tech
        tech_count = tech_mask.sum()
        
        if tech_count > 0:
            logger.info(f"Applying defaults to {tech_count} {tech} sites")
            defaults_applied[tech] = tech_count
            
            for param, value in params.items():
                if param != 'rte':  # RTE handled separately
                    df.loc[tech_mask, param] = value
            
            # Calculate charge/discharge efficiencies from RTE
            rte = params['rte']
            eta_single = np.sqrt(rte)  # Assume symmetric charge/discharge
            df.loc[tech_mask, 'rte'] = rte
            df.loc[tech_mask, 'eta_charge'] = eta_single
            df.loc[tech_mask, 'eta_discharge'] = eta_single
    
    # Handle unknown technologies
    unknown_mask = ~df['technology'].isin(STORAGE_PARAMETERS.keys())
    unknown_count = unknown_mask.sum()
    
    if unknown_count > 0:
        logger.warning(f"Found {unknown_count} sites with unknown technologies - applying 'Other Storage' defaults")
        defaults_applied['Unknown Technologies'] = unknown_count
        
        other_params = STORAGE_PARAMETERS['Other Storage']
        for param, value in other_params.items():
            if param != 'rte':
                df.loc[unknown_mask, param] = value
        
        rte = other_params['rte']
        eta_single = np.sqrt(rte)
        df.loc[unknown_mask, 'rte'] = rte
        df.loc[unknown_mask, 'eta_charge'] = eta_single
        df.loc[unknown_mask, 'eta_discharge'] = eta_single
    
    logger.info(f"Technology defaults applied: {defaults_applied}")
    return df

def calculate_energy_capacity(df):
    """
    Calculate energy capacity from power and duration where missing.
    
    Args:
        df: DataFrame with power and duration data
        
    Returns:
        DataFrame with calculated energy capacities
    """
    logger.info("Calculating energy capacities...")
    
    # Use existing energy_mwh if available, otherwise calculate from power and duration
    missing_energy = df['energy_mwh'].isna()
    has_power_duration = df['capacity_mw'].notna() & df['duration_h'].notna()
    
    calculate_mask = missing_energy & has_power_duration
    calculate_count = calculate_mask.sum()
    
    if calculate_count > 0:
        logger.info(f"Calculating energy capacity for {calculate_count} sites using power × duration")
        df.loc[calculate_mask, 'energy_mwh'] = (
            df.loc[calculate_mask, 'capacity_mw'] * df.loc[calculate_mask, 'duration_h']
        )
    
    # Also calculate duration for sites where we have power and energy but no duration
    missing_duration = df['duration_h'].isna()
    has_power_energy = df['capacity_mw'].notna() & df['energy_mwh'].notna() & (df['capacity_mw'] > 0)
    
    duration_calc_mask = missing_duration & has_power_energy
    duration_calc_count = duration_calc_mask.sum()
    
    if duration_calc_count > 0:
        logger.info(f"Calculating duration for {duration_calc_count} sites using energy ÷ power")
        df.loc[duration_calc_mask, 'duration_h'] = (
            df.loc[duration_calc_mask, 'energy_mwh'] / df.loc[duration_calc_mask, 'capacity_mw']
        )
    
    return df

def validate_parameters(df):
    """
    Validate storage parameters and fix any issues.
    
    Args:
        df: DataFrame with storage parameters
        
    Returns:
        Validated DataFrame
    """
    logger.info(f"Validating parameters for {len(df)} storage sites...")
    
    initial_count = len(df)
    
    # Remove sites with invalid or missing capacity
    valid_capacity = (df['capacity_mw'] > 0) & df['capacity_mw'].notna()
    df = df[valid_capacity]
    
    # Remove sites without coordinates (required for network integration)
    valid_coords = df['lat'].notna() & df['lon'].notna()
    df = df[valid_coords]
    
    # Validate efficiency values
    efficiency_cols = ['eta_charge', 'eta_discharge', 'rte']
    for col in efficiency_cols:
        # Clip efficiencies to reasonable range (0.1 to 1.0)
        invalid_eff = (df[col] < 0.1) | (df[col] > 1.0)
        if invalid_eff.any():
            logger.warning(f"Found {invalid_eff.sum()} sites with invalid {col} values - clipping to [0.1, 1.0]")
            df[col] = df[col].clip(0.1, 1.0)
    
    # Validate duration (should be positive)
    invalid_duration = (df['duration_h'] <= 0) | df['duration_h'].isna()
    if invalid_duration.any():
        logger.warning(f"Found {invalid_duration.sum()} sites with invalid duration - applying technology defaults")
        # Re-apply duration defaults for invalid values
        for tech, params in STORAGE_PARAMETERS.items():
            tech_mask = (df['technology'] == tech) & invalid_duration
            if tech_mask.any():
                df.loc[tech_mask, 'duration_h'] = params['duration_h']
    
    # Validate capacity factors
    capacity_factor_cols = ['min_capacity_factor', 'max_capacity_factor']
    for col in capacity_factor_cols:
        df[col] = df[col].clip(0.0, 1.0)
    
    # Ensure min <= max capacity factor
    invalid_cf_range = df['min_capacity_factor'] > df['max_capacity_factor']
    if invalid_cf_range.any():
        logger.warning(f"Found {invalid_cf_range.sum()} sites with min > max capacity factor - swapping values")
        df.loc[invalid_cf_range, ['min_capacity_factor', 'max_capacity_factor']] = \
            df.loc[invalid_cf_range, ['max_capacity_factor', 'min_capacity_factor']].values
    
    # Recalculate energy capacity after validation
    df = calculate_energy_capacity(df)
    
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} sites with invalid parameters")
    
    logger.info(f"Parameter validation complete: {len(df)} valid storage sites")
    return df

def prepare_final_output(df):
    """
    Prepare final output with standardized column names and order.
    
    Args:
        df: Validated DataFrame
        
    Returns:
        Final output DataFrame
    """
    logger.info("Preparing final output format...")
    
    # Rename capacity column for consistency with PyPSA
    df = df.rename(columns={'capacity_mw': 'power_mw'})
    
    # Define output column order
    output_columns = [
        'site_name', 'technology', 'power_mw', 'energy_mwh', 'duration_h',
        'eta_charge', 'eta_discharge', 'rte', 'lat', 'lon', 'source',
        'status', 'commissioning_year', 'min_capacity_factor', 'max_capacity_factor',
        'capital_cost', 'marginal_cost'
    ]
    
    # Ensure all columns exist
    for col in output_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Round numerical values for cleaner output
    numerical_columns = ['power_mw', 'energy_mwh', 'duration_h', 'eta_charge', 
                        'eta_discharge', 'rte', 'lat', 'lon', 'commissioning_year',
                        'min_capacity_factor', 'max_capacity_factor', 'capital_cost', 'marginal_cost']
    
    for col in numerical_columns:
        if col in df.columns:
            if col in ['lat', 'lon']:
                df[col] = df[col].round(6)  # High precision for coordinates
            elif col in ['eta_charge', 'eta_discharge', 'rte', 'min_capacity_factor', 'max_capacity_factor']:
                df[col] = df[col].round(3)  # 3 decimal places for efficiencies
            elif col == 'commissioning_year':
                df[col] = df[col].round(0)  # Whole years
            else:
                df[col] = df[col].round(2)  # 2 decimal places for others
    
    return df[output_columns]

def main():
    """Main function to build storage parameters."""
    start_time = time.time()
    logger.info("Starting storage parameter building...")
    
    try:
        # Get input and output files
        try:
            # Snakemake mode
            input_file = snakemake.input.merged_storage
            output_file = snakemake.output.storage_params
            logger.info("Running in Snakemake mode")
        except NameError:
            # Standalone mode
            base_path = Path(__file__).parent.parent.parent
            input_file = base_path / "resources" / "storage" / "storage_sites_merged.csv"
            output_file = base_path / "resources" / "storage" / "storage_parameters.csv"
            logger.info("Running in standalone mode")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load merged storage data
        logger.info(f"Loading merged storage data from: {input_file}")
        try:
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} storage sites")
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading input file: {e}")
            raise
        
        if len(df) == 0:
            logger.warning("No storage sites found in input - creating empty output")
            empty_df = pd.DataFrame(columns=[
                'site_name', 'technology', 'power_mw', 'energy_mwh', 'duration_h',
                'eta_charge', 'eta_discharge', 'rte', 'lat', 'lon', 'source',
                'status', 'commissioning_year', 'min_capacity_factor', 'max_capacity_factor',
                'capital_cost', 'marginal_cost'
            ])
            empty_df.to_csv(output_file, index=False)
            return
        # Record initial input count for validation report
        initial_input_count = len(df)

        # Add technology-specific defaults
        df = add_technology_defaults(df)
        
        # Calculate energy capacities
        df = calculate_energy_capacity(df)
        
        # Validate parameters
        df = validate_parameters(df)
        
        # Prepare final output
        final_df = prepare_final_output(df)
        
        # Save results
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(final_df)} storage sites with parameters to: {output_file}")
        
        # Generate summary statistics
        if len(final_df) > 0:
            # Technology summary
            tech_summary = final_df.groupby('technology').agg({
                'power_mw': ['count', 'sum'],
                'energy_mwh': 'sum',
                'duration_h': 'mean',
                'rte': 'mean'
            }).round(2)
            
            logger.info("Final storage technology summary:")
            for tech in tech_summary.index:
                count = int(tech_summary.loc[tech, ('power_mw', 'count')])
                power = tech_summary.loc[tech, ('power_mw', 'sum')]
                energy = tech_summary.loc[tech, ('energy_mwh', 'sum')]
                duration = tech_summary.loc[tech, ('duration_h', 'mean')]
                rte = tech_summary.loc[tech, ('rte', 'mean')]
                logger.info(f"  {tech}: {count} sites, {power:.1f} MW, {energy:.1f} MWh, {duration:.1f}h avg, {rte:.0%} RTE")
            
            # Overall statistics
            total_power = final_df['power_mw'].sum()
            total_energy = final_df['energy_mwh'].sum()
            avg_duration = total_energy / total_power if total_power > 0 else 0
            
            logger.info(f"Total storage capacity: {total_power:.1f} MW, {total_energy:.1f} MWh")
            logger.info(f"System average duration: {avg_duration:.1f} hours")
        
        # Attempt to write technology summary and validation report
        try:
            tech_summary_path = snakemake.output.tech_summary
        except Exception:
            tech_summary_path = Path(output_file).parent / "technology_summary.csv"

        try:
            validation_report_path = snakemake.output.validation_report
        except Exception:
            validation_report_path = Path(output_file).parent / "validation_report.txt"

        try:
            if 'tech_summary' in locals():
                tech_summary.to_csv(tech_summary_path)
                logger.info(f"Wrote technology summary to: {tech_summary_path}")

            # Write validation report
            total_power = final_df['power_mw'].sum() if len(final_df) > 0 else 0
            total_energy = final_df['energy_mwh'].sum() if len(final_df) > 0 else 0
            avg_duration = total_energy / total_power if total_power > 0 else 0

            with open(validation_report_path, 'w', encoding='utf-8') as vf:
                vf.write('Storage Parameter Validation Report\n')
                vf.write('=================================\n')
                vf.write(f'Initial merged sites: {initial_input_count}\n')
                vf.write(f'Final validated sites: {len(final_df)}\n')
                vf.write(f'Total power (MW): {total_power:.2f}\n')
                vf.write(f'Total energy (MWh): {total_energy:.2f}\n')
                vf.write(f'Average system duration (h): {avg_duration:.2f}\n')
            logger.info(f"Wrote validation report to: {validation_report_path}")
        except Exception as e:
            logger.warning(f"Could not write technology summary or validation report: {e}")

        # Attempt to write technology summary and validation report
        try:
            tech_summary_path = snakemake.output.tech_summary
        except Exception:
            tech_summary_path = Path(output_file).parent / "technology_summary.csv"

        try:
            validation_report_path = snakemake.output.validation_report
        except Exception:
            validation_report_path = Path(output_file).parent / "validation_report.txt"

        try:
            if 'tech_summary' in locals():
                tech_summary.to_csv(tech_summary_path)
                logger.info(f"Wrote technology summary to: {tech_summary_path}")

            # Write validation report
            total_power = final_df['power_mw'].sum() if len(final_df) > 0 else 0
            total_energy = final_df['energy_mwh'].sum() if len(final_df) > 0 else 0
            avg_duration = total_energy / total_power if total_power > 0 else 0

            with open(validation_report_path, 'w', encoding='utf-8') as vf:
                vf.write('Storage Parameter Validation Report\n')
                vf.write('=================================\n')
                vf.write(f'Initial merged sites: {initial_input_count}\n')
                vf.write(f'Final validated sites: {len(final_df)}\n')
                vf.write(f'Total power (MW): {total_power:.2f}\n')
                vf.write(f'Total energy (MWh): {total_energy:.2f}\n')
                vf.write(f'Average system duration (h): {avg_duration:.2f}\n')
            logger.info(f"Wrote validation report to: {validation_report_path}")
        except Exception as e:
            logger.warning(f"Could not write technology summary or validation report: {e}")

        # Log execution summary
        execution_time = time.time() - start_time
        summary_stats = {
            'input_sites': len(df) if 'df' in locals() else 0,
            'final_sites': len(final_df),
            'total_power_mw': final_df['power_mw'].sum() if len(final_df) > 0 else 0,
            'total_energy_mwh': final_df['energy_mwh'].sum() if len(final_df) > 0 else 0,
            'technologies_processed': final_df['technology'].nunique() if len(final_df) > 0 else 0,
            'avg_round_trip_efficiency': final_df['rte'].mean() if len(final_df) > 0 else 0,
            'output_file': str(output_file)
        }
        
        log_execution_summary(logger, "storage_parameters", execution_time, summary_stats)
        logger.info("Storage parameter building completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in storage parameter building: {e}")
        raise

if __name__ == "__main__":
    main()

