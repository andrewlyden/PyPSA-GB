"""
ETYS Coordinate Analysis and Validation Script

This script creates detailed visualizations to showcase the improvements
in the ETYS network coordinate guessing algorithm, including:

- Distance-weighted coordinate estimation validation
- Line length accuracy analysis
- Coordinate confidence mapping
- Before/after algorithm comparison
"""

import pandas as pd
import numpy as np
import pypsa
import folium
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic

# Add logging import
from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info


def load_etys_raw_data(etys_file: str, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Load raw ETYS data to extract line lengths for validation."""
    logger.info(f"Loading raw ETYS data from {etys_file}")
    
    xls = pd.ExcelFile(etys_file)
    logger.info(f"Found {len(xls.sheet_names)} sheets in ETYS file")
    
    # Load line data sheets
    line_sheets = ['B-2-1a', 'B-2-1b', 'B-2-1c', 'B-2-1d']
    line_data = []
    
    for sheet in line_sheets:
        if sheet in xls.sheet_names:
            df = xls.parse(sheet, skiprows=1)
            logger.info(f"Loaded {len(df)} lines from sheet {sheet}")
            line_data.append(df)
    
    # Combine all line data
    if line_data:
        combined_lines = pd.concat(line_data, ignore_index=True)
        logger.info(f"Combined line data: {len(combined_lines)} total lines")
        
        # Calculate total line lengths
        combined_lines['total_length_km'] = (
            combined_lines.get('OHL Length (km)', 0).fillna(0) + 
            combined_lines.get('Cable Length (km)', 0).fillna(0)
        )
        
        return {
            'lines': combined_lines,
            'sheets_processed': line_sheets
        }
    else:
        logger.warning("No line data found in ETYS file")
        return {'lines': pd.DataFrame(), 'sheets_processed': []}


def calculate_coordinate_accuracy(network: pypsa.Network, raw_data: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Calculate accuracy metrics for coordinate estimation."""
    logger.info("Calculating coordinate estimation accuracy")
    
    if raw_data['lines'].empty:
        logger.warning("No raw line data available for validation")
        return pd.DataFrame()
    
    lines_df = raw_data['lines']
    validation_results = []
    
    # Get network buses and their coordinates
    buses = {bus_id: (row.y, row.x) for bus_id, row in network.buses.iterrows()}
    
    for _, line in lines_df.iterrows():
        bus0 = line.get('Node 1')
        bus1 = line.get('Node 2') 
        reported_length = line.get('total_length_km', 0)
        
        if pd.isna(bus0) or pd.isna(bus1) or reported_length <= 0:
            continue
            
        # Check if both buses exist in network
        if bus0 in buses and bus1 in buses:
            coord0 = buses[bus0]  # (lat, lon)
            coord1 = buses[bus1]
            
            # Calculate geodesic distance
            try:
                calculated_distance = geodesic(coord0, coord1).kilometers
                
                # Calculate accuracy metrics
                distance_error = abs(calculated_distance - reported_length)
                relative_error = (distance_error / reported_length) * 100 if reported_length > 0 else float('inf')
                
                validation_results.append({
                    'bus0': bus0,
                    'bus1': bus1,
                    'bus0_lat': coord0[0],
                    'bus0_lon': coord0[1], 
                    'bus1_lat': coord1[0],
                    'bus1_lon': coord1[1],
                    'reported_length_km': reported_length,
                    'calculated_length_km': calculated_distance,
                    'distance_error_km': distance_error,
                    'relative_error_pct': relative_error,
                    'accuracy_category': _categorize_accuracy(relative_error)
                })
                
            except Exception as e:
                logger.debug(f"Error calculating distance for {bus0}-{bus1}: {e}")
                continue
    
    if validation_results:
        validation_df = pd.DataFrame(validation_results)
        logger.info(f"Validated {len(validation_df)} line distances")
        
        # Log summary statistics
        logger.info(f"Mean distance error: {validation_df['distance_error_km'].mean():.3f} km")
        logger.info(f"Median relative error: {validation_df['relative_error_pct'].median():.1f}%")
        
        accuracy_counts = validation_df['accuracy_category'].value_counts()
        for category, count in accuracy_counts.items():
            pct = (count / len(validation_df)) * 100
            logger.info(f"{category}: {count} lines ({pct:.1f}%)")
        
        return validation_df
    else:
        logger.warning("No valid line distances could be calculated")
        return pd.DataFrame()


def _categorize_accuracy(relative_error: float) -> str:
    """Categorize coordinate accuracy based on relative error."""
    if relative_error <= 5:
        return "Excellent (<5%)"
    elif relative_error <= 15:
        return "Good (5-15%)"
    elif relative_error <= 30:
        return "Fair (15-30%)"
    else:
        return "Poor (>30%)"


def create_coordinate_analysis_map(
    network: pypsa.Network, 
    validation_df: pd.DataFrame, 
    output_file: str,
    logger: logging.Logger
) -> folium.Map:
    """Create detailed Folium map showing coordinate analysis."""
    logger.info("Creating coordinate analysis map")
    
    # Calculate map center
    buses_df = pd.DataFrame({
        'lat': network.buses.y,
        'lon': network.buses.x
    })
    center_lat = buses_df['lat'].median()
    center_lon = buses_df['lon'].median()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Define color scheme for accuracy categories
    accuracy_colors = {
        "Excellent (<5%)": "green",
        "Good (5-15%)": "orange", 
        "Fair (15-30%)": "red",
        "Poor (>30%)": "darkred"
    }
    
    # Add accuracy validation layers
    if not validation_df.empty:
        logger.info("Adding accuracy validation layers")
        
        for category, color in accuracy_colors.items():
            category_data = validation_df[validation_df['accuracy_category'] == category]
            
            if not category_data.empty:
                layer = folium.FeatureGroup(name=f"Distance Accuracy: {category} ({len(category_data)} lines)")
                
                for _, line in category_data.iterrows():
                    # Draw line between buses
                    coords = [
                        [line['bus0_lat'], line['bus0_lon']],
                        [line['bus1_lat'], line['bus1_lon']]
                    ]
                    
                    popup_text = f"""
                    <b>Line Distance Validation</b><br>
                    <b>Buses:</b> {line['bus0']} → {line['bus1']}<br>
                    <b>Reported Length:</b> {line['reported_length_km']:.2f} km<br>
                    <b>Calculated Length:</b> {line['calculated_length_km']:.2f} km<br>
                    <b>Error:</b> {line['distance_error_km']:.2f} km ({line['relative_error_pct']:.1f}%)<br>
                    <b>Accuracy:</b> {line['accuracy_category']}
                    """
                    
                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=2,
                        opacity=0.7,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{line['bus0']}-{line['bus1']}: {line['relative_error_pct']:.1f}% error"
                    ).add_to(layer)
                
                layer.add_to(m)
                logger.info(f"Added {len(category_data)} lines to {category} layer")
    
    # Add bus layers
    logger.info("Adding bus coordinate layers")
    
    # Categorize buses by coordinate source (this is a simplified assumption)
    # In practice, you'd track this during coordinate estimation
    buses_layer = folium.FeatureGroup(name=f"Network Buses ({len(network.buses)})")
    
    for bus_id, bus_data in network.buses.iterrows():
        lat, lon = bus_data.y, bus_data.x
        
        # Simple heuristic: assume buses with GSP matches have "known" coordinates
        # and others were estimated (this could be improved with actual tracking)
        if bus_id.endswith(('1Q', '1R', '1S', '1T')):  # Common GSP suffixes
            coord_type = "GSP-matched"
            color = "blue"
        else:
            coord_type = "Estimated"
            color = "orange"
        
        popup_text = f"""
        <b>Bus: {bus_id}</b><br>
        <b>Coordinate Type:</b> {coord_type}<br>
        <b>Voltage:</b> {bus_data.get('v_nom', 'N/A')} kV<br>
        <b>Location:</b> {lat:.4f}, {lon:.4f}
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{bus_id} ({coord_type})",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=1
        ).add_to(buses_layer)
    
    buses_layer.add_to(m)
    
    # Add network topology layer (simplified)
    if len(network.lines) > 0:
        topology_layer = folium.FeatureGroup(name=f"Network Topology ({len(network.lines)} lines)", show=False)
        
        for line_id, line_data in network.lines.iterrows():
            bus0_coord = (network.buses.loc[line_data.bus0, 'y'], network.buses.loc[line_data.bus0, 'x'])
            bus1_coord = (network.buses.loc[line_data.bus1, 'y'], network.buses.loc[line_data.bus1, 'x'])
            
            folium.PolyLine(
                [bus0_coord, bus1_coord],
                color="gray",
                weight=1,
                opacity=0.3,
                tooltip=f"Line {line_id}"
            ).add_to(topology_layer)
        
        topology_layer.add_to(m)
        logger.info(f"Added {len(network.lines)} network lines")
    
    # Add summary statistics box
    if not validation_df.empty:
        stats_html = _create_stats_overlay(validation_df)
        from branca.element import Element
        m.get_root().html.add_child(Element(stats_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"Coordinate analysis map saved to {output_path}")
    
    return m


def _create_stats_overlay(validation_df: pd.DataFrame) -> str:
    """Create HTML overlay with validation statistics."""
    stats = {
        'Total Lines Validated': len(validation_df),
        'Mean Distance Error': f"{validation_df['distance_error_km'].mean():.2f} km",
        'Median Relative Error': f"{validation_df['relative_error_pct'].median():.1f}%",
        'Lines with <15% Error': f"{len(validation_df[validation_df['relative_error_pct'] <= 15])}/{len(validation_df)} ({len(validation_df[validation_df['relative_error_pct'] <= 15])/len(validation_df)*100:.1f}%)"
    }
    
    stats_html = """
    <div style='position:absolute; top:10px; right:10px; width:300px; 
                background:white; padding:10px; border:1px solid #ccc; 
                border-radius:5px; box-shadow:0 2px 5px rgba(0,0,0,0.2); 
                z-index:9999; font-family:Arial, sans-serif; font-size:12px;'>
        <h4 style='margin:0 0 10px 0; color:#333;'>Coordinate Validation Summary</h4>
    """
    
    for key, value in stats.items():
        stats_html += f"<div><b>{key}:</b> {value}</div>"
    
    stats_html += "</div>"
    
    return stats_html


def main():
    """Main function for coordinate analysis."""
    logger = setup_logging("plot_etys_coordinate_analysis", log_level="INFO")
    
    logger.info("="*50)
    logger.info("STARTING ETYS COORDINATE ANALYSIS")
    
    try:
        # Load network
        network_path = snakemake.input.network
        logger.info(f"Loading network from {network_path}")
        network = pypsa.Network(network_path)
        log_network_info(network, logger)
        
        # Load raw ETYS data
        raw_data = load_etys_raw_data(snakemake.input.etys_raw, logger)
        
        # Calculate validation metrics
        validation_df = calculate_coordinate_accuracy(network, raw_data, logger)
        
        # Create analysis map
        create_coordinate_analysis_map(
            network, 
            validation_df, 
            snakemake.output.analysis_map,
            logger
        )
        
        # Save validation report
        if not validation_df.empty:
            validation_df.to_csv(snakemake.output.validation_report, index=False)
            logger.info(f"Validation report saved to {snakemake.output.validation_report}")
        else:
            # Create empty file
            Path(snakemake.output.validation_report).touch()
            logger.warning("No validation data available - created empty report file")
        
        logger.info("ETYS COORDINATE ANALYSIS COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.exception(f"FATAL ERROR in coordinate analysis: {e}")
        raise


if __name__ == "__main__":
    main()

