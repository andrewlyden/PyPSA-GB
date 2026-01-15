#!/usr/bin/env python3
"""
Manual script to create clustered network plots.
This bypasses Snakemake wildcard issues and creates plots directly.
"""

import sys
from pathlib import Path
sys.path.append('scripts')

from plotting_clustered import create_clustered_pypsa_folium_map
from scripts.utilities.logging_config import setup_logging

def main():
    """Create clustered network visualization manually."""
    
    # Set up logging
    logger = setup_logging("plot_clustered_manual", log_level="INFO")
    
    logger.info("="*60)
    logger.info("MANUAL CLUSTERED NETWORK PLOTTING")
    logger.info("="*60)
    
    # Configuration
    scenario = "HT35_clustered_gsp"
    network_model = "ETYS"
    
    # Paths
    network_path = f"resources/{network_model}_clustered_{scenario}.nc"
    boundaries_path = "data/network/GSP_regions_27700_20250109.geojson"
    output_path = f"resources/{network_model}_clustered_{scenario}_preview_network_map.html"
    
    # Check if files exist
    if not Path(network_path).exists():
        logger.error(f"Clustered network not found: {network_path}")
        logger.info("Run: snakemake resources/ETYS_clustered_HT35_clustered_gsp.nc --cores 1")
        return
    
    if not Path(boundaries_path).exists():
        logger.error(f"GSP boundaries not found: {boundaries_path}")
        return
    
    logger.info(f"Network: {network_path}")
    logger.info(f"Boundaries: {boundaries_path}")
    logger.info(f"Output: {output_path}")
    
    # Scenario configuration
    scenario_config = {
        'clustering': {
            'method': 'spatial',
            'config': {
                'boundaries_path': boundaries_path,
                'cluster_column': 'GSPs'
            }
        }
    }
    
    try:
        # Create the clustered map
        logger.info("Creating clustered network visualization...")
        
        map_obj = create_clustered_pypsa_folium_map(
            network_path,
            boundaries_path=boundaries_path,
            scenario_config=scenario_config,
            output_html=output_path,
            logger=logger
        )
        
        logger.info("="*60)
        logger.info(f"✅ SUCCESS: Clustered map created at {output_path}")
        logger.info("Features included:")
        logger.info("  • GSP region polygons (349 regions)")
        logger.info("  • Clustered network buses (285 clusters)")
        logger.info("  • Network connections (lines, transformers, links)")
        logger.info("  • Interactive popups with cluster information")
        logger.info("  • Layer controls for toggling visibility")
        logger.info("  • Embedded demand plots (if available)")
        logger.info("  • Network metadata information box")
        logger.info("="*60)
        logger.info(f"Open the map in your browser: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create clustered map: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

