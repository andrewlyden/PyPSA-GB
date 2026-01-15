"""
Visualize European price differentials and renewable integration.

This script creates plots showing:
1. Price differentials over time by country
2. Renewable share evolution
3. Expected flow patterns
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import time

# Setup logging
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utilities.logging_config import setup_logging, log_execution_summary

logger = setup_logging("visualize_european_prices")

def main():
    """Main execution function."""
    start_time = time.time()
    
    try:
        # Load data
        logger.info("Loading price differentials data...")
        df = pd.read_csv("resources/interconnectors/price_differentials_2024.csv")
        logger.info(f"✓ Loaded {len(df)} records")
        
        # Create output directory
        output_dir = Path("resources/plots/interconnectors")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # =============================================================================
        # Plot 1: Price Differentials Over Time
        # =============================================================================
        logger.info("Creating price differential plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("GB Price Differentials with Connected Countries (2024 FES Data)", 
                     fontsize=16, fontweight='bold')
        
        countries = df['country'].unique()
        for idx, country in enumerate(sorted(countries)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Filter data for this country
            country_df = df[df['country'] == country].copy()
            
            # Plot by scenario
            for scenario in country_df['scenario'].unique():
                scenario_data = country_df[country_df['scenario'] == scenario]
                scenario_data = scenario_data.sort_values('year')
                
                ax.plot(scenario_data['year'], 
                       scenario_data['price_differential_gbp_per_mwh'],
                       label=scenario, 
                       linewidth=2,
                       alpha=0.7)
            
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Price Differential (£/MWh)', fontsize=10)
            ax.set_title(f'{country}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add annotation for flow direction
            avg_diff = country_df['price_differential_gbp_per_mwh'].mean()
            direction = "Import to GB" if avg_diff > 0 else "Export from GB"
            ax.text(0.05, 0.95, f"Avg: {direction}", 
                    transform=ax.transAxes, 
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "price_differentials_by_country.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir / 'price_differentials_by_country.png'}")        # =============================================================================
        # Plot 2: Renewable Share Evolution
        # =============================================================================
        logger.info("Creating renewable share plots...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for country in sorted(countries):
            country_df = df[df['country'] == country].copy()
            
            # Average across scenarios for cleaner visualization
            avg_by_year = country_df.groupby('year')['renewable_share'].mean()
            
            ax.plot(avg_by_year.index, 
                   avg_by_year.values * 100,  # Convert to percentage
                   label=country,
                   linewidth=2.5,
                   marker='o',
                   markersize=4)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Renewable Share (%)', fontsize=12)
        ax.set_title('Renewable Energy Share by Country (Average Across Scenarios)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_dir / "renewable_share_evolution.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir / 'renewable_share_evolution.png'}")
        
        # =============================================================================
        # Plot 3: Price vs Renewable Share Scatter
        # =============================================================================
        logger.info("Creating price vs renewable share scatter plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot colored by country
        colors_plot = plt.cm.tab10(np.linspace(0, 1, len(countries)))
        
        for idx, country in enumerate(sorted(countries)):
            country_df = df[df['country'] == country].copy()
            
            ax.scatter(country_df['renewable_share'] * 100,
                      country_df['estimated_price_gbp_per_mwh'],
                      label=country,
                      alpha=0.5,
                      s=50,
                      color=colors_plot[idx])
        
        ax.set_xlabel('Renewable Share (%)', fontsize=12)
        ax.set_ylabel('Estimated Electricity Price (£/MWh)', fontsize=12)
        ax.set_title('Electricity Price vs Renewable Share by Country', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        # Add trendline
        x = df['renewable_share'].values * 100
        y = df['estimated_price_gbp_per_mwh'].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        ax.plot(x[mask], p(x[mask]), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / "price_vs_renewable_share.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir / 'price_vs_renewable_share.png'}")
        
        # =============================================================================
        # Summary Statistics
        # =============================================================================
        logger.info("=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)
        
        logger.info("Average Price Differentials (GB vs Connected Countries):")
        logger.info("(Positive = GB imports, Negative = GB exports)")
        logger.info("-" * 60)
        
        summary = df.groupby('country').agg({
            'price_differential_gbp_per_mwh': ['mean', 'std', 'min', 'max'],
            'estimated_price_gbp_per_mwh': 'mean',
            'renewable_share': 'mean'
        }).round(2)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.sort_values('price_differential_gbp_per_mwh_mean', ascending=False)
        
        for country, row in summary.iterrows():
            logger.info(f"{country}:")
            logger.info(f"  Price Differential: £{row['price_differential_gbp_per_mwh_mean']:.2f}/MWh "
                  f"(±{row['price_differential_gbp_per_mwh_std']:.2f})")
            logger.info(f"  Range: £{row['price_differential_gbp_per_mwh_min']:.2f} to £{row['price_differential_gbp_per_mwh_max']:.2f}")
            logger.info(f"  Avg Country Price: £{row['estimated_price_gbp_per_mwh_mean']:.2f}/MWh")
            logger.info(f"  Avg Renewable Share: {row['renewable_share_mean']*100:.1f}%")
        
        # Calculate final statistics
        total_countries = len(countries)
        total_records = len(df)
        plots_created = 3
        
        # Log execution summary
        log_execution_summary(
            logger,
            "visualize_european_prices",
            start_time,
            inputs={'price_differentials': "resources/interconnectors/price_differentials_2024.csv"},
            outputs={
                'plot1': str(output_dir / 'price_differentials_by_country.png'),
                'plot2': str(output_dir / 'renewable_share_evolution.png'),
                'plot3': str(output_dir / 'price_vs_renewable_share.png')
            },
            context={
                'countries': total_countries,
                'records': total_records,
                'plots_created': plots_created
            }
        )
        
    except Exception as e:
        logger.error(f"Error visualizing European prices: {e}")
        raise

if __name__ == "__main__":
    main()
