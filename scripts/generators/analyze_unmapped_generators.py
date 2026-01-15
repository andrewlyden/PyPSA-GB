"""
Unmapped Generator Analysis for PyPSA-GB

This script analyzes generators that are still missing location coordinates after
all location mapping attempts (REPD, TEC network bus mapping, Wikipedia).
It creates comprehensive statistics and exports unmapped generators to CSV.

The analysis helps identify the remaining gaps in location data and provides
insights for manual location lookup or alternative data sources.

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Configure logging
logger = None
try:
    from logging_config import setup_logging
    logger = setup_logging("analyze_unmapped_generators")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging
        logger = setup_logging("analyze_unmapped_generators")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("analyze_unmapped_generators")

def analyze_unmapped_generators(generators_file: str) -> Dict:
    """
    Analyze generators that are still missing location coordinates.
    
    Args:
        generators_file: Path to the enhanced generator database CSV
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    logger.info("Starting unmapped generator analysis")
    
    # Load enhanced generator database
    generators_df = pd.read_csv(generators_file)
    logger.info(f"Loaded {len(generators_df)} total generators")
    
    # Identify unmapped generators (missing x_coord or y_coord)
    unmapped_mask = generators_df[['x_coord', 'y_coord']].isna().any(axis=1)
    unmapped_df = generators_df[unmapped_mask].copy()
    mapped_df = generators_df[~unmapped_mask].copy()
    
    logger.info(f"Found {len(unmapped_df)} unmapped generators ({len(unmapped_df)/len(generators_df)*100:.1f}%)")
    
    # Calculate basic statistics
    total_capacity = generators_df['capacity_mw'].sum()
    unmapped_capacity = unmapped_df['capacity_mw'].sum()
    mapped_capacity = mapped_df['capacity_mw'].sum()
    
    # Technology breakdown for unmapped generators
    tech_breakdown = unmapped_df.groupby('technology').agg({
        'capacity_mw': ['count', 'sum', 'mean', 'max'],
        'site_name': 'count'
    }).round(2)
    
    tech_breakdown.columns = ['site_count', 'total_capacity_mw', 'avg_capacity_mw', 'max_capacity_mw', 'site_count_2']
    tech_breakdown = tech_breakdown.drop('site_count_2', axis=1)
    tech_breakdown = tech_breakdown.sort_values('total_capacity_mw', ascending=False)
    
    # Data source breakdown for unmapped generators
    source_breakdown = unmapped_df['data_source'].value_counts()
    
    # Capacity percentiles for unmapped generators
    capacity_percentiles = unmapped_df['capacity_mw'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    
    # Largest unmapped generators
    largest_unmapped = unmapped_df.nlargest(20, 'capacity_mw')[
        ['site_name', 'technology', 'capacity_mw', 'data_source', 'operator']
    ]
    
    # Technology comparison: mapped vs unmapped
    tech_comparison = generators_df.groupby('technology').agg({
        'capacity_mw': 'sum'
    })
    
    mapped_by_tech = mapped_df.groupby('technology')['capacity_mw'].sum()
    unmapped_by_tech = unmapped_df.groupby('technology')['capacity_mw'].sum()
    
    tech_comparison['mapped_capacity_mw'] = mapped_by_tech
    tech_comparison['unmapped_capacity_mw'] = unmapped_by_tech.fillna(0)
    tech_comparison['unmapped_percentage'] = (
        tech_comparison['unmapped_capacity_mw'] / tech_comparison['capacity_mw'] * 100
    ).round(1)
    tech_comparison = tech_comparison.sort_values('unmapped_capacity_mw', ascending=False)
    
    # Regional analysis (if region data available)
    regional_analysis = None
    if 'region' in unmapped_df.columns and unmapped_df['region'].notna().any():
        regional_analysis = unmapped_df.groupby('region').agg({
            'capacity_mw': ['count', 'sum'],
            'site_name': 'count'
        })
        regional_analysis.columns = ['site_count', 'total_capacity_mw', 'site_count_2']
        regional_analysis = regional_analysis.drop('site_count_2', axis=1)
        regional_analysis = regional_analysis.sort_values('total_capacity_mw', ascending=False)
    
    # Compile analysis results
    analysis_results = {
        'summary': {
            'total_generators': len(generators_df),
            'mapped_generators': len(mapped_df),
            'unmapped_generators': len(unmapped_df),
            'unmapped_percentage': len(unmapped_df) / len(generators_df) * 100,
            'total_capacity_mw': total_capacity,
            'mapped_capacity_mw': mapped_capacity,
            'unmapped_capacity_mw': unmapped_capacity,
            'unmapped_capacity_percentage': unmapped_capacity / total_capacity * 100
        },
        'technology_breakdown': tech_breakdown,
        'source_breakdown': source_breakdown,
        'capacity_statistics': capacity_percentiles,
        'largest_unmapped': largest_unmapped,
        'technology_comparison': tech_comparison,
        'regional_analysis': regional_analysis,
        'unmapped_data': unmapped_df
    }
    
    logger.info("Unmapped generator analysis completed")
    return analysis_results

def create_unmapped_generators_report(analysis_results: Dict, 
                                    output_csv: str = "resources/generators/unmapped_generators.csv",
                                    output_report: str = "resources/generators/unmapped_generators_analysis.txt") -> None:
    """
    Create comprehensive report of unmapped generators.
    
    Args:
        analysis_results: Results from analyze_unmapped_generators
        output_csv: Path for unmapped generators CSV export
        output_report: Path for analysis report
    """
    logger.info("Creating unmapped generators report")
    
    # Save unmapped generators to CSV
    unmapped_df = analysis_results['unmapped_data']
    
    # Add priority ranking based on capacity
    unmapped_df_export = unmapped_df.copy()
    unmapped_df_export['priority_rank'] = unmapped_df_export['capacity_mw'].rank(ascending=False, method='dense').astype(int)
    
    # Sort by priority (largest capacity first)
    unmapped_df_export = unmapped_df_export.sort_values('capacity_mw', ascending=False)
    
    # Save to CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    unmapped_df_export.to_csv(output_csv, index=False)
    
    logger.info(f"Saved {len(unmapped_df_export)} unmapped generators to {output_csv}")
    
    # Create detailed analysis report
    summary = analysis_results['summary']
    tech_breakdown = analysis_results['technology_breakdown']
    source_breakdown = analysis_results['source_breakdown']
    capacity_stats = analysis_results['capacity_statistics']
    largest_unmapped = analysis_results['largest_unmapped']
    tech_comparison = analysis_results['technology_comparison']
    
    report_lines = [
        "UNMAPPED GENERATORS ANALYSIS REPORT",
        "=" * 50,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 20,
        f"Total generators in database: {summary['total_generators']:,}",
        f"Successfully mapped: {summary['mapped_generators']:,} ({100-summary['unmapped_percentage']:.1f}%)",
        f"Still unmapped: {summary['unmapped_generators']:,} ({summary['unmapped_percentage']:.1f}%)",
        "",
        f"Total capacity in database: {summary['total_capacity_mw']:,.0f} MW",
        f"Mapped capacity: {summary['mapped_capacity_mw']:,.0f} MW ({100-summary['unmapped_capacity_percentage']:.1f}%)",
        f"Unmapped capacity: {summary['unmapped_capacity_mw']:,.0f} MW ({summary['unmapped_capacity_percentage']:.1f}%)",
        "",
        "TECHNOLOGY BREAKDOWN (Unmapped Only)",
        "-" * 40,
    ]
    
    for tech, row in tech_breakdown.iterrows():
        report_lines.append(
            f"{tech:20s}: {row['site_count']:3.0f} sites, {row['total_capacity_mw']:8,.0f} MW "
            f"(avg: {row['avg_capacity_mw']:6.1f} MW, max: {row['max_capacity_mw']:6.0f} MW)"
        )
    
    report_lines.extend([
        "",
        "DATA SOURCE BREAKDOWN (Unmapped Only)",
        "-" * 40,
    ])
    
    for source, count in source_breakdown.items():
        pct = count / summary['unmapped_generators'] * 100
        report_lines.append(f"{source:15s}: {count:3d} sites ({pct:5.1f}%)")
    
    report_lines.extend([
        "",
        "CAPACITY STATISTICS (Unmapped Only)",
        "-" * 40,
        f"Mean capacity: {capacity_stats['mean']:8.1f} MW",
        f"Median capacity: {capacity_stats['50%']:8.1f} MW",
        f"Standard deviation: {capacity_stats['std']:8.1f} MW",
        f"10th percentile: {capacity_stats['10%']:8.1f} MW",
        f"90th percentile: {capacity_stats['90%']:8.1f} MW",
        f"99th percentile: {capacity_stats['99%']:8.1f} MW",
        f"Maximum capacity: {capacity_stats['max']:8.1f} MW",
        "",
        "TOP 15 LARGEST UNMAPPED GENERATORS",
        "-" * 50,
        f"{'Rank':<4} {'Site Name':<30} {'Technology':<15} {'Capacity (MW)':<12} {'Source':<8}",
        "-" * 70,
    ])
    
    for i, (_, gen) in enumerate(largest_unmapped.head(15).iterrows(), 1):
        report_lines.append(
            f"{i:3d}. {gen['site_name'][:29]:<29} {gen['technology']:<15} "
            f"{gen['capacity_mw']:8.0f} MW    {gen['data_source']:<8}"
        )
    
    report_lines.extend([
        "",
        "TECHNOLOGY MAPPING SUCCESS RATES",
        "-" * 40,
        f"{'Technology':<20} {'Total (MW)':<12} {'Unmapped (MW)':<13} {'Unmapped %':<10}",
        "-" * 55,
    ])
    
    for tech, row in tech_comparison.head(10).iterrows():
        report_lines.append(
            f"{tech[:19]:<20} {row['capacity_mw']:8.0f} MW  "
            f"{row['unmapped_capacity_mw']:8.0f} MW     {row['unmapped_percentage']:6.1f}%"
        )
    
    # Add regional analysis if available
    if analysis_results['regional_analysis'] is not None:
        regional = analysis_results['regional_analysis']
        report_lines.extend([
            "",
            "REGIONAL BREAKDOWN (Unmapped Only)",
            "-" * 40,
            f"{'Region':<15} {'Sites':<6} {'Capacity (MW)':<12}",
            "-" * 33,
        ])
        
        for region, row in regional.head(10).iterrows():
            report_lines.append(
                f"{region[:14]:<15} {row['site_count']:4.0f}   {row['total_capacity_mw']:8.0f} MW"
            )
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 20,
        "1. Prioritize manual lookup for generators >500 MW (highest impact)",
        "2. Focus on CCGT/Nuclear technologies (largest unmapped capacity)",
        "3. Consider additional data sources (company websites, planning documents)",
        "4. Investigate geographic clustering for batch location lookup",
        "5. Manual verification for high-priority sites (top 20 by capacity)",
        "",
        f"CSV export with full details: {output_csv}",
        "Priority ranking included based on generator capacity."
    ])
    
    # Save report
    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved analysis report to {output_report}")

def main(generators_file: str = "resources/generators/dispatchable_generators_with_wikipedia_locations.csv",
         output_csv: str = "resources/generators/unmapped_generators.csv",
         output_report: str = "resources/generators/unmapped_generators_analysis.txt"):
    """
    Main function to analyze unmapped generators and create comprehensive reports.
    """
    logger.info("Starting comprehensive unmapped generator analysis")
    
    # Analyze unmapped generators
    analysis_results = analyze_unmapped_generators(generators_file)
    
    # Create reports
    create_unmapped_generators_report(analysis_results, output_csv, output_report)
    
    # Log summary
    summary = analysis_results['summary']
    logger.info("Analysis Summary:")
    logger.info(f"  Unmapped generators: {summary['unmapped_generators']}/{summary['total_generators']} ({summary['unmapped_percentage']:.1f}%)")
    logger.info(f"  Unmapped capacity: {summary['unmapped_capacity_mw']:,.0f}/{summary['total_capacity_mw']:,.0f} MW ({summary['unmapped_capacity_percentage']:.1f}%)")
    logger.info(f"  CSV export: {output_csv}")
    logger.info(f"  Analysis report: {output_report}")
    
    return {
        'unmapped_count': summary['unmapped_generators'],
        'unmapped_capacity': summary['unmapped_capacity_mw'],
        'unmapped_percentage': summary['unmapped_percentage'],
        'output_files': [output_csv, output_report]
    }

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    if 'snakemake' in globals():
        # Snakemake execution
        generators_file = snakemake.input.generators_with_wikipedia_locations
        output_csv = snakemake.output.unmapped_generators_csv
        output_report = snakemake.output.unmapped_analysis_report
    else:
        # Command line execution
        generators_file = sys.argv[1] if len(sys.argv) > 1 else "resources/generators/dispatchable_generators_with_wikipedia_locations.csv"
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "resources/generators/unmapped_generators.csv"
        output_report = sys.argv[3] if len(sys.argv) > 3 else "resources/generators/unmapped_generators_analysis.txt"
    
    stats = main(generators_file, output_csv, output_report)
    
    logger.info("Unmapped Generator Analysis Completed!")
    logger.info("Results: %d unmapped sites (%.1f%%) with %.0f MW capacity", 
               stats['unmapped_count'], stats['unmapped_percentage'], stats['unmapped_capacity'])

