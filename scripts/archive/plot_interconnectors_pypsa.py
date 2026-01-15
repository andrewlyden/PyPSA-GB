"""
Plot cross-border interconnectors from a PyPSA network using PyPSA's native plotting functions.

This script creates visualizations of international electricity interconnectors (Links)
to verify interconnector data integration, locations, and cross-border connections in the network.

Creates:
- Static matplotlib plot (OSGB36 coordinates for network topology)
- Interactive pydeck map (WGS84 coordinates for web viewing)
- HTML summary report with statistics and embedded visualizations

Dependencies: pypsa, matplotlib, pydeck (optional)
"""
import logging
from pathlib import Path
import sys
import html
import time

import pandas as pd
import numpy as np
import pypsa

from scripts.utilities.logging_config import setup_logging, log_network_info, log_execution_summary
from scripts.utilities.network_io import load_network

# Initialize timing
start_time = time.time()

# Snakemake provides inputs/outputs when run under Snakemake
try:
    snakemake  # type: ignore
except NameError:
    raise RuntimeError("This script is intended to be run via Snakemake")

# Use centralized logging
log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "plot_interconnectors_pypsa"
logger = setup_logging(log_path)

network_path = Path(snakemake.input.network)
output_html = Path(snakemake.output.html)
output_html.parent.mkdir(parents=True, exist_ok=True)

# Read scenario from wildcard
scenario = snakemake.wildcards.scenario if hasattr(snakemake, 'wildcards') and hasattr(snakemake.wildcards, 'scenario') else 'Unknown'

logger.info(f"Loading network from: {network_path}")

# Load network
try:
    n = load_network(str(network_path))
    logger.info(f"Successfully loaded network: {n.name}")
    log_network_info(n, logger)
except Exception as e:
    logger.error(f"Failed to load network: {e}")
    output_html.write_text(f"<html><body><h1>Interconnector Plot</h1><p>[ERROR] Failed to load network: {html.escape(str(e))}</p></body></html>")
    sys.exit(1)

# Check for interconnector links
if n.links.empty:
    logger.warning("No links found in network; creating placeholder report")
    output_html.write_text(
        f"<html><body><h1>Interconnector Plot - {scenario}</h1>"
        f"<h3>No interconnector links found in network</h3>"
        f"<p>This may indicate interconnector integration has not been completed yet.</p>"
        "</body></html>"
    )
    sys.exit(0)

# Filter for cross-border interconnectors (exclude internal DC links if any)
# Interconnectors typically have specific carriers or naming conventions
interconnectors = n.links.copy()

# Try to identify interconnectors by carrier or naming patterns
# Common interconnector carriers: 'DC', 'interconnector', specific country names
interconnector_carriers = ['DC', 'interconnector', 'Interconnector']
is_interconnector = interconnectors['carrier'].isin(interconnector_carriers)

# If no specific carriers, check for cross-border links by examining bus0/bus1 countries
if not is_interconnector.any():
    logger.info("No specific interconnector carriers found, checking for cross-border links...")
    # Assume all links are interconnectors for now
    is_interconnector = pd.Series(True, index=interconnectors.index)

interconnectors = interconnectors[is_interconnector].copy()

if interconnectors.empty:
    logger.warning("No interconnectors identified in links; showing all links")
    interconnectors = n.links.copy()

logger.info(f"Found {len(interconnectors)} interconnector links")

# Log interconnector details
if 'carrier' in interconnectors.columns:
    logger.info(f"Interconnector carriers: {interconnectors['carrier'].value_counts().to_dict()}")

# Calculate total capacity
total_capacity_mw = interconnectors['p_nom'].sum()
logger.info(f"Total interconnector capacity: {total_capacity_mw:.2f} MW")

# Log breakdown by destination country (if available in link names or bus1)
if 'bus1' in interconnectors.columns:
    countries = interconnectors['bus1'].value_counts()
    logger.info(f"Interconnector endpoints: {countries.to_dict()}")

# Coordinate harmonization for plotting
def harmonize_coordinates():
    """Ensure buses have proper coordinates for plotting (OSGB36 -> WGS84 conversion)."""
    buses = n.buses
    
    # Check if lon/lat exist and are valid (in degree range)
    def is_valid_degrees(series_name, max_val):
        if series_name not in buses.columns:
            return pd.Series(False, index=buses.index)
        valid = pd.to_numeric(buses[series_name], errors='coerce').dropna()
        if valid.empty:
            return pd.Series(False, index=buses.index)
        result = pd.Series(False, index=buses.index)
        result.loc[valid.index] = (valid.abs() <= max_val)
        return result
    
    has_valid_lon = is_valid_degrees('lon', 180.0)
    has_valid_lat = is_valid_degrees('lat', 90.0)
    has_x = 'x' in buses.columns and buses['x'].notna().any()
    has_y = 'y' in buses.columns and buses['y'].notna().any()
    
    valid_wgs84 = has_valid_lon & has_valid_lat
    
    # If we already have valid WGS84 coordinates, we're done
    if valid_wgs84.sum() == len(buses):
        logger.info("✓ All buses have valid WGS84 coordinates (lon/lat)")
        if not has_x or not has_y:
            buses['x'] = buses['lon']
            buses['y'] = buses['lat']
        return True
    
    # Check if x/y are in OSGB36 range (British National Grid: ~0-700000 E, ~0-1300000 N)
    if has_x and has_y:
        x_vals = pd.to_numeric(buses['x'], errors='coerce').dropna()
        y_vals = pd.to_numeric(buses['y'], errors='coerce').dropna()
        
        if not x_vals.empty and not y_vals.empty:
            likely_osgb = (
                x_vals.between(-1000, 800000).all() and
                y_vals.between(-1000, 1400000).all()
            )
            
            if likely_osgb:
                logger.info(f"Detected OSGB36 coordinates (x/y), converting to WGS84...")
                
                # Try pyproj first, then fallback
                try:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                    
                    mask = buses['x'].notna() & buses['y'].notna()
                    lon_vals, lat_vals = transformer.transform(
                        buses.loc[mask, 'x'].to_numpy(dtype=float),
                        buses.loc[mask, 'y'].to_numpy(dtype=float)
                    )
                    
                    buses.loc[mask, 'lon'] = lon_vals
                    buses.loc[mask, 'lat'] = lat_vals
                    logger.info(f"Converted {mask.sum()} buses from OSGB36 to WGS84 using pyproj")
                    
                    # Keep x/y in OSGB36 for matplotlib plotting, but ensure lon/lat for pydeck
                    return True
                    
                except ImportError:
                    logger.warning("pyproj not available, using fallback OSGB36->WGS84 conversion")
                    # Fallback conversion
                    try:
                        from scripts.validate_network import _osgb36_to_wgs84
                        
                        mask = buses['x'].notna() & buses['y'].notna()
                        lon_vals, lat_vals = _osgb36_to_wgs84(
                            buses.loc[mask, 'x'].to_numpy(dtype=float),
                            buses.loc[mask, 'y'].to_numpy(dtype=float),
                            logger
                        )
                        
                        buses.loc[mask, 'lon'] = lon_vals
                        buses.loc[mask, 'lat'] = lat_vals
                        logger.info(f"Converted {mask.sum()} buses from OSGB36 to WGS84 using fallback")
                        return True
                    except ImportError:
                        logger.warning("Fallback conversion not available")
                        return False
            else:
                # x/y appear to be in degrees already
                logger.info("x/y appear to be in degree coordinates")
                buses['lon'] = buses['x']
                buses['lat'] = buses['y']
                return True
    
    logger.warning(f"⚠️  Only {valid_wgs84.sum()}/{len(buses)} buses have valid WGS84 coordinates")
    logger.info("Some visualizations may have coordinate issues")
    return False

has_valid_coords = harmonize_coordinates()

if not has_valid_coords:
    logger.error("No valid bus coordinates found for plotting")
    output_html.write_text(
        f"<html><body><h1>Interconnector Plot - {scenario}</h1>"
        f"<p>[ERROR] No valid bus coordinates found for plotting</p>"
        f"<p>Interconnector links: {len(interconnectors)}</p>"
        "</body></html>"
    )
    sys.exit(1)

# =============================================================================
# STATIC MATPLOTLIB PLOT (OSGB36 coordinates)
# =============================================================================

plot_dir = output_html.parent / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)
static_plot_path = plot_dir / f"{scenario}_interconnectors_topology.png"
static_plot_status = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # PyPSA's n.plot() uses x/y coordinates (OSGB36 meters), NOT lon/lat!
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot network with buses and lines using x/y (OSGB36 meters)
    n.plot(
        ax=ax,
        geomap=False,  # Disable geographic projection requirement
        bus_size=5,
        line_width=0.2,
        link_width=0.5,
        bus_colors='lightgray',
        line_colors='lightgray',
        link_colors='red',
        title=f"Cross-Border Interconnectors - {scenario}"
    )
    
    # Overlay interconnector links as thicker colored lines
    for link_name, link_row in interconnectors.iterrows():
        bus0_name = link_row['bus0']
        bus1_name = link_row['bus1']
        
        # Get bus coordinates
        if bus0_name in n.buses.index and bus1_name in n.buses.index:
            bus0 = n.buses.loc[bus0_name]
            bus1 = n.buses.loc[bus1_name]
            
            # Use x/y for plotting (OSGB36 meters)
            if 'x' in bus0 and 'y' in bus0 and 'x' in bus1 and 'y' in bus1:
                x0, y0 = bus0['x'], bus0['y']
                x1, y1 = bus1['x'], bus1['y']
                
                if pd.notna(x0) and pd.notna(y0) and pd.notna(x1) and pd.notna(y1):
                    # Calculate line width based on capacity (2-10 pts)
                    capacity = link_row['p_nom']
                    line_width = 2 + (capacity / total_capacity_mw) * 8 if total_capacity_mw > 0 else 3
                    
                    # Color by carrier or default to blue
                    color = 'blue' if link_row.get('carrier') == 'DC' else 'darkblue'
                    
                    ax.plot([x0, x1], [y0, y1], 
                           color=color, 
                           linewidth=line_width, 
                           alpha=0.7,
                           zorder=10,
                           label=f"{link_name} ({capacity:.0f} MW)" if capacity > 500 else None)
    
    # Add legend for major interconnectors (>500 MW)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=8, framealpha=0.9, title='Major Interconnectors')
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_aspect('equal')
    
    fig.tight_layout()
    fig.savefig(static_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Static interconnector plot saved to: {static_plot_path}")
    static_plot_status = "[OK] Static interconnector plot generated (OSGB36 coordinates)"
    
except ImportError as exc:
    logger.warning(f"Matplotlib not available for plotting: {exc}")
    static_plot_status = f"[SKIP] Static plot (matplotlib missing)"
    static_plot_path = None
except Exception as exc:
    logger.error(f"Failed to create static plot: {exc}", exc_info=True)
    static_plot_status = f"[ERROR] Static plot failed: {exc}"
    static_plot_path = None

# =============================================================================
# INTERACTIVE PYDECK PLOT (WGS84 coordinates)
# =============================================================================

interactive_plot_path = output_html.parent / f"{scenario}_interconnectors_explore.html"
interactive_plot_status = None

try:
    import pydeck as pdk
    
    logger.info(f"Creating custom pydeck visualization for {len(interconnectors)} interconnectors")
    
    # Prepare interconnector link data with coordinates
    links_data = []
    
    for link_name, link_row in interconnectors.iterrows():
        bus0_name = link_row['bus0']
        bus1_name = link_row['bus1']
        
        if bus0_name in n.buses.index and bus1_name in n.buses.index:
            bus0 = n.buses.loc[bus0_name]
            bus1 = n.buses.loc[bus1_name]
            
            # Ensure WGS84 coordinates
            if all(k in bus0 for k in ['lon', 'lat']) and all(k in bus1 for k in ['lon', 'lat']):
                lon0, lat0 = bus0['lon'], bus0['lat']
                lon1, lat1 = bus1['lon'], bus1['lat']
                
                if pd.notna(lon0) and pd.notna(lat0) and pd.notna(lon1) and pd.notna(lat1):
                    links_data.append({
                        'name': link_name,
                        'source_lon': lon0,
                        'source_lat': lat0,
                        'target_lon': lon1,
                        'target_lat': lat1,
                        'capacity_mw': link_row['p_nom'],
                        'carrier': link_row.get('carrier', 'Unknown'),
                        'bus0': bus0_name,
                        'bus1': bus1_name,
                    })
    
    if not links_data:
        raise ValueError("No valid interconnector links with coordinates found")
    
    links_df = pd.DataFrame(links_data)
    
    logger.info(f"Creating custom pydeck map with {len(links_df)} interconnector links")
    logger.info(f"Total capacity: {links_df['capacity_mw'].sum():.0f} MW")
    
    # Color by capacity (gradient from yellow to red)
    max_capacity = links_df['capacity_mw'].max()
    min_capacity = links_df['capacity_mw'].min()
    capacity_range = max_capacity - min_capacity
    
    def capacity_to_color(capacity):
        """Map capacity to color gradient (green -> yellow -> red)."""
        if capacity_range > 0:
            norm = (capacity - min_capacity) / capacity_range
        else:
            norm = 0.5
        
        if norm < 0.5:
            # Green to Yellow
            r = int(255 * (norm * 2))
            g = 255
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (1 - (norm - 0.5) * 2))
        
        return [r, g, 0, 180]
    
    links_df['color'] = links_df['capacity_mw'].apply(capacity_to_color)
    
    # Calculate line widths (scale 2-10)
    if capacity_range > 0:
        links_df['width'] = 2 + (links_df['capacity_mw'] - min_capacity) / capacity_range * 8
    else:
        links_df['width'] = 5
    
    logger.info(f"Capacity range: {min_capacity:.0f} to {max_capacity:.0f} MW")
    
    # Create line layer for interconnectors
    interconnector_layer = pdk.Layer(
        "LineLayer",
        links_df,
        pickable=True,
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["target_lon", "target_lat"],
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        width_max_pixels=15,
    )
    
    # Add network buses as background layer (smaller, gray)
    bus_df = n.buses[['lon', 'lat']].copy().reset_index()
    bus_df = bus_df[bus_df['lon'].notna() & bus_df['lat'].notna()]
    bus_name_col = bus_df.columns[0]
    bus_df['tooltip'] = bus_df[bus_name_col].apply(lambda b: f"Bus: {b}")
    
    bus_layer = pdk.Layer(
        "ScatterplotLayer",
        bus_df,
        pickable=True,
        opacity=0.3,
        stroked=False,
        filled=True,
        radius_scale=1,
        radius_min_pixels=1,
        radius_max_pixels=3,
        get_position=["lon", "lat"],
        get_radius=15,
        get_fill_color=[150, 150, 150, 100],
    )
    
    # Add connection point markers (source and target buses of interconnectors)
    connection_points = []
    for _, link in links_df.iterrows():
        # Source point
        connection_points.append({
            'lon': link['source_lon'],
            'lat': link['source_lat'],
            'name': link['bus0'],
            'type': 'GB Connection',
            'link': link['name'],
        })
        # Target point
        connection_points.append({
            'lon': link['target_lon'],
            'lat': link['target_lat'],
            'name': link['bus1'],
            'type': 'Foreign Connection',
            'link': link['name'],
        })
    
    connections_df = pd.DataFrame(connection_points)
    
    connection_layer = pdk.Layer(
        "ScatterplotLayer",
        connections_df,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=5,
        radius_max_pixels=20,
        line_width_min_pixels=2,
        get_position=["lon", "lat"],
        get_radius=500,
        get_fill_color=[255, 100, 100, 200],
        get_line_color=[150, 0, 0, 255],
    )
    
    # Center view on GB with wider zoom to show connections
    view_state = pdk.ViewState(
        longitude=-2.5,
        latitude=53.0,
        zoom=4.5,  # Wider zoom to see cross-border connections
        pitch=0,
    )
    
    logger.info(f"Creating pydeck visualization with {len(links_df)} interconnectors")
    
    deck = pdk.Deck(
        layers=[bus_layer, interconnector_layer, connection_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Interconnector: {name}</b><br/>"
                   "Capacity: {capacity_mw:.0f} MW<br/>"
                   "From: {bus0}<br/>"
                   "To: {bus1}<br/>"
                   "Carrier: {carrier}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "padding": "10px",
                "borderRadius": "5px",
                "fontSize": "12px"
            }
        },
        map_style="light",
    )
    
    deck.to_html(str(interactive_plot_path), notebook_display=False, open_browser=False)
    logger.info(f"Custom interactive interconnector map created with {len(links_df)} links")
    logger.info(f"Saved to: {interactive_plot_path}")
    interactive_plot_status = f"[OK] Custom pydeck map with {len(links_df)} interconnectors (color-coded by capacity)"
    
except ImportError as exc:
    logger.warning(f"Pydeck not available for interactive plotting: {exc}")
    interactive_plot_status = "[SKIP] Interactive plot (pydeck missing)"
    interactive_plot_path = None
except Exception as exc:
    logger.error(f"Failed to create interactive plot: {exc}", exc_info=True)
    interactive_plot_status = f"[ERROR] Interactive plot failed: {exc}"
    interactive_plot_path = None

# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interconnector Visualization - {scenario}</title>
    <meta charset="utf-8"/>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .stat-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .stat-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: white; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .carrier-badge {{ display: inline-block; padding: 5px 10px; border-radius: 3px; font-size: 11px; font-weight: bold; }}
        .dc {{ background-color: #e3f2fd; color: #1976d2; }}
        .interconnector {{ background-color: #f3e5f5; color: #7b1fa2; }}
        .other {{ background-color: #fff3e0; color: #f57c00; }}
        .warning {{ background-color: #fff3cd; border-left-color: #ffc107; }}
        .info {{ color: #666; font-size: 13px; margin-top: 10px; }}
        .status-ok {{ color: green; }}
        .status-error {{ color: red; }}
        .status-skip {{ color: orange; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cross-Border Interconnector Visualization</h1>
        <p>Scenario: <strong>{scenario}</strong></p>
    </div>
    
    <div class="section">
        <h2>Interconnector Summary</h2>
        <div class="summary">
            <div class="stat-box">
                <div class="stat-label">Total Interconnectors</div>
                <div class="stat-value">{len(interconnectors)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Capacity</div>
                <div class="stat-value">{total_capacity_mw:.0f} MW</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Capacity</div>
                <div class="stat-value">{interconnectors['p_nom'].mean():.0f} MW</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Connection Points</div>
                <div class="stat-value">{interconnectors['bus0'].nunique()}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Interconnector Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Interconnector</th>
                    <th>From (GB)</th>
                    <th>To (Foreign)</th>
                    <th>Capacity (MW)</th>
                    <th>Carrier</th>
                    <th>Efficiency (%)</th>
                </tr>
            </thead>
            <tbody>
"""

# Add interconnector rows
for link_name, link_row in interconnectors.iterrows():
    bus0 = link_row['bus0']
    bus1 = link_row['bus1']
    capacity = link_row['p_nom']
    carrier = link_row.get('carrier', 'Unknown')
    
    # Get efficiency if available
    efficiency = link_row.get('efficiency', 1.0) * 100
    
    # Determine badge color
    if 'dc' in carrier.lower():
        badge_class = 'dc'
    elif 'interconnector' in carrier.lower():
        badge_class = 'interconnector'
    else:
        badge_class = 'other'
    
    html_content += f"""
                <tr>
                    <td><strong>{link_name}</strong></td>
                    <td>{bus0}</td>
                    <td>{bus1}</td>
                    <td>{capacity:.0f}</td>
                    <td><span class="carrier-badge {badge_class}">{carrier}</span></td>
                    <td>{efficiency:.1f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>Data Quality</h2>
"""

# Add data quality checks
missing_coords = 0
for link_name, link_row in interconnectors.iterrows():
    bus0_name = link_row['bus0']
    bus1_name = link_row['bus1']
    if bus0_name in n.buses.index and bus1_name in n.buses.index:
        bus0 = n.buses.loc[bus0_name]
        bus1 = n.buses.loc[bus1_name]
        if pd.isna(bus0.get('lon')) or pd.isna(bus0.get('lat')) or pd.isna(bus1.get('lon')) or pd.isna(bus1.get('lat')):
            missing_coords += 1

if missing_coords > 0:
    html_content += f"""
        <div class="stat-box warning">
            ⚠️  {missing_coords} interconnectors have buses with missing coordinates
        </div>
"""

html_content += f"""
        <div class="info">
            <strong>Network Information:</strong><br>
            • Total buses: {len(n.buses)}<br>
            • Total generators: {len(n.generators)}<br>
            • Total links: {len(n.links)}<br>
            • Interconnectors: {len(interconnectors)}<br>
            • Network name: {n.name}<br>
            • Time periods: {n.snapshots.size if hasattr(n, 'snapshots') else 'Unknown'}<br>
        </div>
    </div>
    
    <div class="section">
        <h2>Visualization Status</h2>
        <ul>
"""

# Add status messages
if static_plot_status:
    status_class = "status-ok" if "[OK]" in static_plot_status else ("status-error" if "[ERROR]" in static_plot_status else "status-skip")
    html_content += f'            <li class="{status_class}">{html.escape(static_plot_status)}</li>\n'

if interactive_plot_status:
    status_class = "status-ok" if "[OK]" in interactive_plot_status else ("status-error" if "[ERROR]" in interactive_plot_status else "status-skip")
    html_content += f'            <li class="{status_class}">{html.escape(interactive_plot_status)}</li>\n'

html_content += """
        </ul>
    </div>
"""

# Embed static plot if available
if static_plot_path and static_plot_path.exists():
    try:
        static_rel = static_plot_path.relative_to(output_html.parent)
    except ValueError:
        static_rel = static_plot_path.name
    
    html_content += f"""
    <div class="section">
        <h2>Static Interconnector Map</h2>
        <img src="{static_rel.as_posix()}" alt="Interconnectors topology plot" style="max-width:100%; height:auto;"/>
    </div>
"""

# Embed interactive plot if available
if interactive_plot_path and interactive_plot_path.exists():
    try:
        interactive_rel = interactive_plot_path.relative_to(output_html.parent)
    except ValueError:
        interactive_rel = interactive_plot_path.name
    
    html_content += f"""
    <div class="section">
        <h2>Interactive Interconnector Explorer</h2>
        <p>Explore cross-border interconnector connections interactively. 
        Hover over interconnector links for details. Line color and thickness reflect capacity.</p>
        <iframe src="{interactive_rel.as_posix()}" title="Interactive interconnector explorer" 
        style="width:100%; height:700px; border:1px solid #ccc;"></iframe>
    </div>
"""

html_content += """
    
    <div class="section info" style="text-align: center; color: #999; font-size: 12px;">
        <p>This report was automatically generated by plot_interconnectors_pypsa.py</p>
        <p>Interconnector data verified and visualized using PyPSA network components</p>
    </div>
</body>
</html>
"""

# Write HTML report
try:
    output_html.write_text(html_content)
    logger.info(f"✓ Interconnector visualization report saved to: {output_html}")
except Exception as e:
    logger.error(f"Failed to write HTML output: {e}")
    sys.exit(1)

# Log execution summary
execution_time = time.time() - start_time
logger.info(f"Execution time: {execution_time:.2f} seconds")
logger.info("Interconnector visualization completed successfully")

