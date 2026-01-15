#!/usr/bin/env python3
"""
Comprehensive point-based Folium map for PyPSA-GB.

Creates a single HTML map with reliable layer toggles showing:
- Network buses (gray points)
- Generators (all technologies, color-coded points)
- Flexibility assets (storage + placeholder points)
- Interconnector landing points (GB + counterparty where available)

Features:
- Point markers only (no scaling, no lines)
- One FeatureGroup per category with MarkerCluster
- Reliable layer toggles
- Normalized coordinate inputs
- Robust error handling

Author: AI Assistant  
Date: January 2025
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
import pypsa
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import warnings
warnings.filterwarnings('ignore')

# Configure logging
import sys
sys.path.append('..')
try:
    from scripts.utilities.logging_config import setup_logging
    logger = setup_logging("plot_comprehensive_map")
except:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("plot_comprehensive_map")

# Detect Snakemake mode
SNAKEMAKE_MODE = 'snakemake' in globals()

if SNAKEMAKE_MODE:
    network_file = snakemake.input.network
    generators_file = snakemake.input.generators
    storage_file = snakemake.input.storage_merged
    interconnectors_file = snakemake.input.interconnectors
    ev_file = snakemake.input.get('ev_placeholder', None)
    thermal_file = snakemake.input.get('thermal_placeholder', None)
    output_file = snakemake.output.map_file
    network_model = snakemake.params.network_model
    scenario = snakemake.params.get('scenario', '')

# Technology color mapping (deterministic)
GENERATOR_COLORS = {
    'wind_onshore': '#2ca25f',
    'wind_offshore': '#006d2c', 
    'solar': '#fec44f',
    'solar_pv': '#fec44f',
    'ccgt': '#f16913',
    'ocgt': '#d94801',
    'nuclear': '#54278f',
    'biomass': '#8c510a',
    'biogas': '#8c510a',
    'landfill_gas': '#8c510a',
    'sewage_gas': '#8c510a',
    'waste_to_energy': '#8c510a',
    'advanced_biofuel': '#8c510a',
    'hydro': '#2b8cbe',
    'large_hydro': '#2b8cbe',
    'small_hydro': '#2b8cbe',
    'pumped_hydro': '#1f77b4',
    'geothermal': '#636363',
    'gas_reciprocating': '#f16913',
    'oil_gas_turbine': '#d94801',
    'thermal_other': '#7f8c8d',
    'tidal_stream': '#2b8cbe',
    'shoreline_wave': '#2b8cbe',
    'chp': '#f16913',
    'battery': '#17becf',
    'battery_gas': '#17becf',
    'battery_generic': '#17becf',
    'demand_response_storage': '#17becf',
    'hydrogen_storage': '#9467bd',
    'other': '#7f8c8d',
    'unknown': '#7f8c8d'
}

STORAGE_COLORS = {
    'battery': '#17becf',
    'pumped': '#1f77b4', 
    'caes': '#2ca02c',
    'laes': '#8c564b',
    'flywheel': '#9467bd',
    'other': '#7f8c8d'
}

INTERCONNECTOR_COLORS = {
    'gb': '#e41a1c',
    'counterparty': '#377eb8'
}

def convert_osgb_to_latlon(x: float, y: float) -> Tuple[float, float]:
    """Convert OSGB36 coordinates to WGS84 lat/lon (approximate)."""
    try:
        # Approximate OSGB to WGS84 conversion
        # Using simplified transformation (accurate enough for visualization)
        lat = 49.766 + (y - 100000) * 8.983e-6
        lon = -7.557 + (x - 400000) * 1.399e-5
        return lat, lon
    except:
        return None, None

def normalize_coordinates(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normalize coordinates from various formats to lat/lon.
    Priority: lon/lat > OSGB conversion > x_coord/y_coord > bus fallback
    """
    normalized = df.copy()
    
    # Initialize lat/lon columns if they don't exist
    if 'lat' not in normalized.columns:
        normalized['lat'] = np.nan
    if 'lon' not in normalized.columns:
        normalized['lon'] = np.nan
    
    missing_coords = 0
    
    for idx, row in normalized.iterrows():
        lat, lon = row.get('lat'), row.get('lon')
        
        # Check if we already have valid lat/lon
        if pd.notna(lat) and pd.notna(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            continue
            
        # Try OSGB coordinates (x_coord, y_coord)
        x_coord = row.get('x_coord') or row.get('x')
        y_coord = row.get('y_coord') or row.get('y')
        
        if pd.notna(x_coord) and pd.notna(y_coord):
            lat_conv, lon_conv = convert_osgb_to_latlon(x_coord, y_coord)
            if lat_conv and lon_conv:
                normalized.at[idx, 'lat'] = lat_conv
                normalized.at[idx, 'lon'] = lon_conv
                continue
        
        # Try bus coordinates if available
        bus_name = row.get('bus') or row.get('Bus')
        if pd.notna(bus_name):
            # This would require bus coordinate lookup - placeholder for now
            pass
            
        missing_coords += 1
    
    if missing_coords > 0:
        logger.warning(f"Could not normalize coordinates for {missing_coords} records")
    
    # Filter out rows without valid coordinates
    valid_coords = normalized.dropna(subset=['lat', 'lon'])
    logger.info(f"Normalized coordinates for {len(valid_coords)}/{len(df)} records")
    
    return valid_coords

def create_base_map() -> folium.Map:
    """Create base Folium map centered on UK."""
    # UK approximate center
    uk_center = [54.5, -3.0]
    
    m = folium.Map(
        location=uk_center,
        zoom_start=6,
        tiles='CartoDB positron',
        control_scale=True,
        prefer_canvas=True
    )
    
    return m

def add_network_buses(m: folium.Map, network: pypsa.Network, logger: logging.Logger) -> folium.FeatureGroup:
    """Add network buses as gray points."""
    logger.info("Adding network buses...")
    
    # Create feature group for buses
    bus_group = folium.FeatureGroup(name="Network Buses", show=True)
    bus_cluster = plugins.MarkerCluster(name="Bus Cluster")
    
    buses_df = network.buses.copy()
    buses_df = normalize_coordinates(buses_df, logger)
    
    for idx, bus in buses_df.iterrows():
        try:
            popup_text = f"""
            <b>Bus:</b> {bus.name}<br>
            <b>Voltage:</b> {bus.get('v_nom', 'N/A')} kV<br>
            <b>Country:</b> {bus.get('country', 'N/A')}
            """
            
            folium.CircleMarker(
                location=[bus['lat'], bus['lon']],
                radius=3,
                popup=folium.Popup(popup_text, max_width=300),
                color='#636363',
                fillColor='#636363',
                fillOpacity=0.7,
                weight=1
            ).add_to(bus_cluster)
            
        except Exception as e:
            logger.warning(f"Failed to add bus {bus.name}: {e}")
    
    bus_cluster.add_to(bus_group)
    bus_group.add_to(m)
    
    logger.info(f"Added {len(buses_df)} network buses")
    return bus_group

def add_generators(m: folium.Map, generators_file: str, logger: logging.Logger) -> folium.FeatureGroup:
    """Add generators as color-coded points."""
    logger.info("Adding generators...")
    
    # Create feature group for generators
    gen_group = folium.FeatureGroup(name="Generators", show=True)
    gen_cluster = plugins.MarkerCluster(name="Generator Cluster")
    
    try:
        generators_df = pd.read_csv(generators_file)
        generators_df = normalize_coordinates(generators_df, logger)
        
        for idx, gen in generators_df.iterrows():
            try:
                technology = gen.get('technology', gen.get('Technology', 'unknown')).lower()
                color = GENERATOR_COLORS.get(technology, GENERATOR_COLORS['unknown'])
                
                popup_text = f"""
                <b>Generator:</b> {gen.get('name', gen.get('Station Name', 'N/A'))}<br>
                <b>Technology:</b> {technology}<br>
                <b>Capacity:</b> {gen.get('p_nom', gen.get('Installed Capacity (MW)', 'N/A'))} MW<br>
                <b>Bus:</b> {gen.get('bus', gen.get('Bus', 'N/A'))}
                """
                
                folium.CircleMarker(
                    location=[gen['lat'], gen['lon']],
                    radius=4,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(gen_cluster)
                
            except Exception as e:
                logger.warning(f"Failed to add generator {gen.get('name', 'Unknown')}: {e}")
        
        gen_cluster.add_to(gen_group)
        gen_group.add_to(m)
        
        logger.info(f"Added {len(generators_df)} generators")
        
    except Exception as e:
        logger.error(f"Failed to load generators: {e}")
    
    return gen_group

def add_storage(m: folium.Map, storage_file: str, logger: logging.Logger) -> folium.FeatureGroup:
    """Add storage as blue points."""
    logger.info("Adding storage...")
    
    # Create feature group for storage
    storage_group = folium.FeatureGroup(name="Storage", show=True)
    storage_cluster = plugins.MarkerCluster(name="Storage Cluster")
    
    try:
        if os.path.exists(storage_file):
            storage_df = pd.read_csv(storage_file)
            storage_df = normalize_coordinates(storage_df, logger)
            
            for idx, storage in storage_df.iterrows():
                try:
                    storage_type = storage.get('technology', storage.get('type', 'battery')).lower()
                    color = STORAGE_COLORS.get(storage_type, STORAGE_COLORS['other'])
                    
                    popup_text = f"""
                    <b>Storage:</b> {storage.get('name', 'N/A')}<br>
                    <b>Type:</b> {storage_type}<br>
                    <b>Capacity:</b> {storage.get('p_nom', storage.get('capacity', 'N/A'))} MW<br>
                    <b>Energy:</b> {storage.get('e_nom', storage.get('energy', 'N/A'))} MWh
                    """
                    
                    folium.CircleMarker(
                        location=[storage['lat'], storage['lon']],
                        radius=4,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=color,
                        fillColor=color,
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(storage_cluster)
                    
                except Exception as e:
                    logger.warning(f"Failed to add storage {storage.get('name', 'Unknown')}: {e}")
            
            storage_cluster.add_to(storage_group)
            logger.info(f"Added {len(storage_df)} storage units")
        else:
            logger.warning(f"Storage file not found: {storage_file}")
            
    except Exception as e:
        logger.error(f"Failed to load storage: {e}")
    
    storage_group.add_to(m)
    return storage_group

def add_ev_placeholder(m: folium.Map, logger: logging.Logger) -> folium.FeatureGroup:
    """Add EV charging placeholder points."""
    logger.info("Adding EV placeholder...")
    
    # Create feature group for EV
    ev_group = folium.FeatureGroup(name="EV Charging", show=False)
    
    # Add a few placeholder points for demonstration
    ev_locations = [
        [51.5074, -0.1278, "London EV Hub"],
        [53.4808, -2.2426, "Manchester EV Hub"], 
        [55.9533, -3.1883, "Edinburgh EV Hub"],
        [52.4862, -1.8904, "Birmingham EV Hub"]
    ]
    
    for lat, lon, name in ev_locations:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            popup=f"<b>EV Charging:</b> {name}",
            color='#9370DB',
            fillColor='#9370DB',
            fillOpacity=0.7,
            weight=2
        ).add_to(ev_group)
    
    ev_group.add_to(m)
    logger.info(f"Added {len(ev_locations)} EV placeholder points")
    return ev_group

def add_thermal_placeholder(m: folium.Map, logger: logging.Logger) -> folium.FeatureGroup:
    """Add thermal storage placeholder points."""
    logger.info("Adding thermal storage placeholder...")
    
    # Create feature group for thermal storage
    thermal_group = folium.FeatureGroup(name="Thermal Storage", show=False)
    
    # Add a few placeholder points
    thermal_locations = [
        [51.5074, -0.1278, "London Thermal"],
        [53.4808, -2.2426, "Manchester Thermal"],
        [55.9533, -3.1883, "Edinburgh Thermal"]
    ]
    
    for lat, lon, name in thermal_locations:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            popup=f"<b>Thermal Storage:</b> {name}",
            color='#FF69B4',
            fillColor='#FF69B4', 
            fillOpacity=0.7,
            weight=2
        ).add_to(thermal_group)
    
    thermal_group.add_to(m)
    logger.info(f"Added {len(thermal_locations)} thermal storage placeholder points")
    return thermal_group

def add_interconnectors(m: folium.Map, interconnectors_file: str, logger: logging.Logger) -> Tuple[folium.FeatureGroup, folium.FeatureGroup]:
    """Add interconnector landing points."""
    logger.info("Adding interconnectors...")
    
    # Create feature groups
    gb_group = folium.FeatureGroup(name="Interconnectors (GB)", show=True)
    counterparty_group = folium.FeatureGroup(name="Interconnectors (Counterparty)", show=False)
    
    try:
        if os.path.exists(interconnectors_file):
            interconnectors_df = pd.read_csv(interconnectors_file)
            
            # Process GB side
            gb_df = interconnectors_df[['name', 'gb_lat', 'gb_lon', 'capacity_mw']].copy()
            gb_df.rename(columns={'gb_lat': 'lat', 'gb_lon': 'lon'}, inplace=True)
            gb_df = gb_df.dropna(subset=['lat', 'lon'])
            
            for idx, ic in gb_df.iterrows():
                try:
                    popup_text = f"""
                    <b>Interconnector:</b> {ic['name']}<br>
                    <b>Side:</b> GB<br>
                    <b>Capacity:</b> {ic.get('capacity_mw', 'N/A')} MW
                    """
                    
                    folium.CircleMarker(
                        location=[ic['lat'], ic['lon']],
                        radius=5,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=INTERCONNECTOR_COLORS['gb'],
                        fillColor=INTERCONNECTOR_COLORS['gb'],
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(gb_group)
                    
                except Exception as e:
                    logger.warning(f"Failed to add GB interconnector {ic.get('name', 'Unknown')}: {e}")
            
            # Process counterparty side
            cp_df = interconnectors_df[['name', 'counterparty_lat', 'counterparty_lon', 'capacity_mw']].copy()
            cp_df.rename(columns={'counterparty_lat': 'lat', 'counterparty_lon': 'lon'}, inplace=True)
            cp_df = cp_df.dropna(subset=['lat', 'lon'])
            
            for idx, ic in cp_df.iterrows():
                try:
                    popup_text = f"""
                    <b>Interconnector:</b> {ic['name']}<br>
                    <b>Side:</b> Counterparty<br>
                    <b>Capacity:</b> {ic.get('capacity_mw', 'N/A')} MW
                    """
                    
                    folium.CircleMarker(
                        location=[ic['lat'], ic['lon']],
                        radius=5,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=INTERCONNECTOR_COLORS['counterparty'],
                        fillColor=INTERCONNECTOR_COLORS['counterparty'],
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(counterparty_group)
                    
                except Exception as e:
                    logger.warning(f"Failed to add counterparty interconnector {ic.get('name', 'Unknown')}: {e}")
            
            logger.info(f"Added {len(gb_df)} GB and {len(cp_df)} counterparty interconnector points")
            
        else:
            logger.warning(f"Interconnectors file not found: {interconnectors_file}")
            
    except Exception as e:
        logger.error(f"Failed to load interconnectors: {e}")
    
    gb_group.add_to(m)
    counterparty_group.add_to(m)
    
    return gb_group, counterparty_group

def add_legend(m: folium.Map, logger: logging.Logger):
    """Add color legend to map."""
    logger.info("Adding legend...")
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 300px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>PyPSA-GB Comprehensive Map</b></p>
    <p><i class="fa fa-circle" style="color:#636363"></i> Network Buses</p>
    <p><b>Generators:</b></p>
    <p><i class="fa fa-circle" style="color:#2ca25f"></i> Wind Onshore</p>
    <p><i class="fa fa-circle" style="color:#006d2c"></i> Wind Offshore</p>
    <p><i class="fa fa-circle" style="color:#fec44f"></i> Solar PV</p>
    <p><i class="fa fa-circle" style="color:#f16913"></i> Gas (CCGT/OCGT)</p>
    <p><i class="fa fa-circle" style="color:#54278f"></i> Nuclear</p>
    <p><i class="fa fa-circle" style="color:#2b8cbe"></i> Hydro</p>
    <p><b>Storage & Flexibility:</b></p>
    <p><i class="fa fa-circle" style="color:#17becf"></i> Battery Storage</p>
    <p><i class="fa fa-circle" style="color:#9370DB"></i> EV Charging</p>
    <p><i class="fa fa-circle" style="color:#FF69B4"></i> Thermal Storage</p>
    <p><b>Interconnectors:</b></p>
    <p><i class="fa fa-circle" style="color:#e41a1c"></i> GB Side</p>
    <p><i class="fa fa-circle" style="color:#377eb8"></i> Counterparty Side</p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))

def main():
    """Main function to create comprehensive map."""
    logger.info("Starting comprehensive map generation...")
    
    try:
        # Load network
        logger.info(f"Loading network from: {network_file}")
        network = pypsa.Network(network_file)
        
        # Create base map
        m = create_base_map()
        
        # Add all components
        add_network_buses(m, network, logger)
        add_generators(m, generators_file, logger)
        add_storage(m, storage_file, logger)
        add_ev_placeholder(m, logger)
        add_thermal_placeholder(m, logger)
        add_interconnectors(m, interconnectors_file, logger)
        
        # Add legend
        add_legend(m, logger)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Save map
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        m.save(output_file)
        
        logger.info(f"Comprehensive map saved to: {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("COMPREHENSIVE MAP GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Map saved to: {output_file}")
        print(f"Network model: {network_model}")
        print(f"Scenario: {scenario}")
        print(f"Total buses: {len(network.buses)}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Failed to generate comprehensive map: {e}")
        raise

if __name__ == "__main__":
    main()

