#!/usr/bin/env python3
"""
Validate Historical Interconnector Flows
=========================================

Post-solve validation script to compare optimized interconnector flows
against actual historical ESPENI data.

This script:
1. Loads solved network with historical interconnector validation data
2. Extracts optimized interconnector flows from solution
3. Compares against stored historical flows
4. Generates comparison statistics and plots
5. Identifies periods of significant deviation

Usage:
    python scripts/interconnectors/validate_historical_flows.py \
        --network resources/network/Historical_2020_clustered_solved.nc \
        --output resources/validation/interconnector_flows_2020.csv

Author: PyPSA-GB Team
"""

import sys
import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utilities.logging_config import setup_logging

def calculate_validation_metrics(optimized: pd.Series, 
                                 historical: pd.Series) -> dict:
    """
    Calculate validation metrics between optimized and historical flows.
    
    Args:
        optimized: Optimized flow time series (MW)
        historical: Historical flow time series (MW)
        
    Returns:
        Dictionary of validation metrics
    """
    # Align time series
    aligned_opt = optimized.reindex(historical.index)
    
    # Calculate metrics
    mae = (aligned_opt - historical).abs().mean()
    rmse = np.sqrt(((aligned_opt - historical) ** 2).mean())
    mape = ((aligned_opt - historical).abs() / (historical.abs() + 1e-6)).mean() * 100
    
    # Correlation
    correlation = aligned_opt.corr(historical)
    
    # Energy metrics
    opt_energy_twh = aligned_opt.sum() * 0.5 / 1e6  # Half-hourly to TWh
    hist_energy_twh = historical.sum() * 0.5 / 1e6
    energy_error_twh = opt_energy_twh - hist_energy_twh
    energy_error_pct = (energy_error_twh / hist_energy_twh) * 100 if hist_energy_twh != 0 else 0
    
    return {
        'mae_mw': mae,
        'rmse_mw': rmse,
        'mape_percent': mape,
        'correlation': correlation,
        'optimized_energy_twh': opt_energy_twh,
        'historical_energy_twh': hist_energy_twh,
        'energy_error_twh': energy_error_twh,
        'energy_error_percent': energy_error_pct
    }

def validate_interconnector_flows(network: pypsa.Network, 
                                  logger: logging.Logger) -> pd.DataFrame:
    """
    Validate optimized interconnector flows against historical data.
    
    Args:
        network: Solved PyPSA network with historical_interconnector_flows attribute
        logger: Logger instance
        
    Returns:
        DataFrame with validation results for each interconnector
    """
    if not hasattr(network, 'historical_interconnector_flows'):
        logger.error("Network does not contain historical_interconnector_flows attribute")
        logger.error("This network may not have been built with historical validation data")
        return pd.DataFrame()
    
    logger.info(f"Validating {len(network.historical_interconnector_flows)} interconnectors...")
    
    validation_results = []
    
    for link_name, historical_flows in network.historical_interconnector_flows.items():
        # Get optimized flows from solution
        if link_name not in network.links_t.p1.columns:
            logger.warning(f"Link {link_name} not found in solution - skipping")
            continue
        
        optimized_flows = network.links_t.p1[link_name]
        
        # Calculate metrics
        metrics = calculate_validation_metrics(optimized_flows, historical_flows)
        
        # Store results
        result = {
            'interconnector': link_name.replace('IC_', ''),
            'link_name': link_name,
            **metrics
        }
        validation_results.append(result)
        
        # Log summary
        logger.info(f"{link_name}:")
        logger.info(f"  Energy: {metrics['optimized_energy_twh']:.2f} TWh (opt) vs "
                   f"{metrics['historical_energy_twh']:.2f} TWh (hist), "
                   f"error: {metrics['energy_error_percent']:.1f}%")
        logger.info(f"  MAE: {metrics['mae_mw']:.1f} MW, "
                   f"RMSE: {metrics['rmse_mw']:.1f} MW, "
                   f"Correlation: {metrics['correlation']:.3f}")
    
    return pd.DataFrame(validation_results)

def plot_flow_comparison(network: pypsa.Network,
                        link_name: str,
                        output_dir: Path,
                        logger: logging.Logger) -> None:
    """
    Plot comparison of optimized vs historical flows for one interconnector.
    
    Args:
        network: Solved PyPSA network
        link_name: Name of interconnector link
        output_dir: Directory to save plot
        logger: Logger instance
    """
    historical_flows = network.historical_interconnector_flows[link_name]
    optimized_flows = network.links_t.p1[link_name]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Time series comparison
    ax = axes[0]
    ax.plot(historical_flows.index, historical_flows, 
            label='Historical (ESPENI)', color='blue', alpha=0.7, linewidth=0.8)
    ax.plot(optimized_flows.index, optimized_flows, 
            label='Optimized', color='red', alpha=0.7, linewidth=0.8)
    ax.set_ylabel('Power Flow (MW)')
    ax.set_title(f'{link_name} - Optimized vs Historical Flows')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Difference plot
    ax = axes[1]
    difference = optimized_flows.reindex(historical_flows.index) - historical_flows
    ax.plot(difference.index, difference, color='purple', linewidth=0.8)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Difference (MW)')
    ax.set_title('Difference (Optimized - Historical)')
    ax.grid(True, alpha=0.3)
    
    # Scatter plot
    ax = axes[2]
    aligned_opt = optimized_flows.reindex(historical_flows.index)
    ax.scatter(historical_flows, aligned_opt, alpha=0.3, s=1)
    
    # Add perfect correlation line
    min_val = min(historical_flows.min(), aligned_opt.min())
    max_val = max(historical_flows.max(), aligned_opt.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', label='Perfect correlation', linewidth=1)
    
    ax.set_xlabel('Historical Flow (MW)')
    ax.set_ylabel('Optimized Flow (MW)')
    ax.set_title('Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{link_name}_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved plot: {output_file}")

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description='Validate historical interconnector flows'
    )
    parser.add_argument(
        '--network',
        required=True,
        help='Path to solved network file (.nc)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save validation results (.csv)'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate validation plots for each interconnector'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging("validate_historical_flows")
    
    try:
        logger.info("Starting historical interconnector flow validation...")
        logger.info(f"Network: {args.network}")
        logger.info(f"Output: {args.output}")
        
        # Load network
        if not Path(args.network).exists():
            raise FileNotFoundError(f"Network file not found: {args.network}")
        
        logger.info("Loading solved network...")
        network = pypsa.Network(args.network)
        
        # Validate flows
        validation_df = validate_interconnector_flows(network, logger)
        
        if len(validation_df) == 0:
            logger.error("No validation results generated")
            return 1
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validation_df.to_csv(output_path, index=False)
        logger.info(f"Saved validation results: {output_path}")
        
        # Generate summary statistics
        logger.info("\n=== VALIDATION SUMMARY ===")
        logger.info(f"Total interconnectors validated: {len(validation_df)}")
        logger.info(f"Mean MAE: {validation_df['mae_mw'].mean():.1f} MW")
        logger.info(f"Mean RMSE: {validation_df['rmse_mw'].mean():.1f} MW")
        logger.info(f"Mean correlation: {validation_df['correlation'].mean():.3f}")
        logger.info(f"Total energy error: {validation_df['energy_error_twh'].sum():.2f} TWh "
                   f"({validation_df['energy_error_percent'].mean():.1f}% avg)")
        
        # Generate plots if requested
        if args.plots:
            logger.info("\nGenerating validation plots...")
            plots_dir = output_path.parent / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            for link_name in network.historical_interconnector_flows.keys():
                plot_flow_comparison(network, link_name, plots_dir, logger)
            
            logger.info(f"Saved {len(network.historical_interconnector_flows)} plots to {plots_dir}")
        
        logger.info("\nValidation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

