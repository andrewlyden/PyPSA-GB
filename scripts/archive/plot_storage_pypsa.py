"""
Plot energy storage units from a PyPSA network using PyPSA's native plotting functions.

This script creates visualizations of storage facilities (battery, pumped hydro, CAES, LAES, etc.)
to verify storage data integration and locations in the network.

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
log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "plot_storage_pypsa"
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
    output_html.write_text(f"<html><body><h1>Storage Unit Plot</h1><p>[ERROR] Failed to load network: {html.escape(str(e))}</p></body></html>")
    sys.exit(1)

# Check for storage units
if n.storage_units.empty:
    logger.warning("No storage units found in network; creating placeholder report")
    output_html.write_text(
        f"<html><body><h1>Storage Unit Plot - {scenario}</h1>"
        f"<h3>No storage units found in network</h3>"
        f"<p>This may indicate storage integration has not been completed yet.</p>"
        "</body></html>"
    )
    sys.exit(0)

storage_units = n.storage_units.copy()

logger.info(f"Found {len(storage_units)} storage units")
logger.info(f"Storage carriers: {storage_units['carrier'].value_counts().to_dict()}")
logger.info(f"Total power capacity: {storage_units['p_nom'].sum():.2f} MW")

# Calculate energy capacity from p_nom * max_hours
storage_units['e_nom'] = storage_units['p_nom'] * storage_units['max_hours']
logger.info(f"Total energy capacity: {storage_units['e_nom'].sum():.2f} MWh")

# Log breakdown by storage technology
storage_techs = storage_units['carrier'].unique()
for tech in sorted(storage_techs):
    tech_units = storage_units[storage_units['carrier'] == tech]
    logger.info(f"  {tech}: {len(tech_units)} units, {tech_units['p_nom'].sum():.1f} MW, {tech_units['e_nom'].sum():.1f} MWh")

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
        f"<html><body><h1>Storage Unit Plot - {scenario}</h1>"
        f"<p>[ERROR] No valid bus coordinates found for plotting</p>"
        f"<p>Storage units: {len(storage_units)}</p>"
        "</body></html>"
    )
    sys.exit(1)

# Storage units need coordinates from their buses
# Check if storage units already have x/y coordinates
if 'x' not in storage_units.columns or storage_units['x'].isna().all():
    logger.info("Storage units missing coordinates - extracting from buses")
    
    # Add x/y coordinates from buses to storage_units
    for idx in storage_units.index:
        bus_name = storage_units.loc[idx, 'bus']
        if bus_name in n.buses.index:
            if 'x' in n.buses.columns and n.buses.loc[bus_name, 'x'] is not None:
                storage_units.loc[idx, 'x'] = n.buses.loc[bus_name, 'x']
            if 'y' in n.buses.columns and n.buses.loc[bus_name, 'y'] is not None:
                storage_units.loc[idx, 'y'] = n.buses.loc[bus_name, 'y']

# Also check lon/lat for pydeck plotting
if 'lon' not in storage_units.columns or storage_units['lon'].isna().all():
    logger.info("Adding lon/lat to storage units from buses")
    
    for idx in storage_units.index:
        bus_name = storage_units.loc[idx, 'bus']
        if bus_name in n.buses.index:
            if 'lon' in n.buses.columns and n.buses.loc[bus_name, 'lon'] is not None:
                storage_units.loc[idx, 'lon'] = n.buses.loc[bus_name, 'lon']
            if 'lat' in n.buses.columns and n.buses.loc[bus_name, 'lat'] is not None:
                storage_units.loc[idx, 'lat'] = n.buses.loc[bus_name, 'lat']

# Preserve storage names from index (only if 'name' column doesn't exist)
if 'name' not in storage_units.columns:
    storage_units['name'] = storage_units.index

# Verify we have coordinates for plotting
storage_has_coords_xy = storage_units[['x', 'y']].notna().all(axis=1)
storage_has_coords_lonlat = storage_units[['lon', 'lat']].notna().all(axis=1)

logger.info(f"Storage units with x/y coordinates: {storage_has_coords_xy.sum()}/{len(storage_units)}")
logger.info(f"Storage units with lon/lat coordinates: {storage_has_coords_lonlat.sum()}/{len(storage_units)}")

# =============================================================================
# STATIC MATPLOTLIB PLOT (OSGB36 coordinates)
# =============================================================================

plot_dir = output_html.parent / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)
static_plot_path = plot_dir / f"{scenario}_storage_topology.png"
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
        bus_colors='lightgray',
        line_colors='lightgray',
        title=f"Energy Storage Units - {scenario}"
    )
    
    # Overlay storage unit locations as colored markers
    storage_buses = storage_units['bus'].values
    bus_coords = n.buses.loc[storage_buses]
    
    # Use x/y for plotting (OSGB36 meters)
    lon_col, lat_col = 'x', 'y'
    
    # Color code by carrier (storage technologies)
    carrier_colors = {
        'Battery': 'gold',
        'battery': 'gold',
        'Pumped Storage Hydroelectricity': 'blue',
        'Pumped Storage': 'blue',
        'Pumped Hydro': 'blue',
        'pumped_hydro': 'blue',
        'Flywheel': 'purple',
        'flywheel': 'purple',
        'CAES': 'orange',
        'Liquid Air Energy Storage': 'cyan',
        'LAES': 'cyan',
    }
    
    # Calculate marker sizes with linear scaling (2x bigger than before)
    min_marker_size = 40    # Minimum size in points^2
    max_marker_size = 1000  # Maximum size in points^2
    
    # Get capacity range across all storage units
    min_capacity = storage_units['p_nom'].min()
    max_capacity = storage_units['p_nom'].max()
    capacity_range = max_capacity - min_capacity
    
    # Plot each carrier type
    for carrier in storage_units['carrier'].unique():
        carrier_storage = storage_units[storage_units['carrier'] == carrier]
        carrier_buses = n.buses.loc[carrier_storage['bus'].values]
        
        # Skip if no valid coordinates
        if carrier_buses[lon_col].isna().all() or carrier_buses[lat_col].isna().all():
            logger.warning(f"Skipping {carrier} - no valid coordinates")
            continue
        
        color = carrier_colors.get(carrier, 'red')
        
        # Linear scaling between min and max marker sizes
        if capacity_range > 0:
            sizes = min_marker_size + (carrier_storage['p_nom'].values - min_capacity) / capacity_range * (max_marker_size - min_marker_size)
        else:
            sizes = np.full(len(carrier_storage), min_marker_size)
        
        ax.scatter(
            carrier_buses[lon_col].values,
            carrier_buses[lat_col].values,
            s=sizes,
            c=color,
            alpha=0.6,
            label=f"{carrier} ({len(carrier_storage)} units, {carrier_storage['p_nom'].sum():.0f} MW)",
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_aspect('equal')
    
    fig.tight_layout()
    fig.savefig(static_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Static storage plot saved to: {static_plot_path}")
    static_plot_status = "[OK] Static storage plot generated (OSGB36 coordinates)"
    
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

interactive_plot_path = output_html.parent / f"{scenario}_storage_explore.html"
interactive_plot_status = None

try:
    import pydeck as pdk
    
    logger.info(f"Creating custom pydeck visualization for {len(storage_units)} storage units")
    
    # Prepare storage data with coordinates
    storage_df = storage_units.copy()
    
    # Ensure we have a 'name' column (from index)
    if 'name' not in storage_df.columns:
        storage_df['name'] = storage_df.index
    
    # Reset index but don't create duplicate 'name' column
    if storage_df.index.name:
        storage_df = storage_df.reset_index(drop=True)
    else:
        # Index is already unnamed or we have 'name' column, just reset
        storage_df = storage_df.copy()  # Just use the dataframe as-is
    
    # Ensure WGS84 coordinates are valid
    storage_df = storage_df[storage_df['lon'].notna() & storage_df['lat'].notna()]
    
    # Check if coordinates are in WGS84 range (degrees)
    lon_vals = storage_df['lon']
    lat_vals = storage_df['lat']
    
    is_wgs84 = (lon_vals.between(-10, 5).all() and lat_vals.between(48, 62).all())
    
    if not is_wgs84:
        logger.warning(f"Storage lon/lat appear to be in OSGB36 (sample: E={lon_vals.iloc[0]:.1f}, N={lat_vals.iloc[0]:.1f})")
        logger.info("Attempting to convert OSGB36 coordinates to WGS84")
        
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
            
            lons_wgs84, lats_wgs84 = transformer.transform(storage_df['lon'].values, storage_df['lat'].values)
            storage_df['lon'] = lons_wgs84
            storage_df['lat'] = lats_wgs84
            logger.info(f"✓ Converted {len(storage_df)} storage coordinates from OSGB36 to WGS84")
            logger.info(f"New range: lon {storage_df['lon'].min():.2f} to {storage_df['lon'].max():.2f}, lat {storage_df['lat'].min():.2f} to {storage_df['lat'].max():.2f}")
        except ImportError:
            logger.warning("pyproj not available for coordinate conversion")
        except Exception as e:
            logger.warning(f"Coordinate conversion failed: {e}")
    
    logger.info(f"Creating custom pydeck map with {len(storage_df)} storage units at {storage_df['bus'].nunique()} unique locations")
    logger.info(f"Coordinate range: lon {storage_df['lon'].min():.2f} to {storage_df['lon'].max():.2f}, lat {storage_df['lat'].min():.2f} to {storage_df['lat'].max():.2f}")
    
    # Map carriers to colors (RGB tuples)
    carrier_color_rgb = {
        'Battery': [255, 193, 7],               # Gold
        'battery': [255, 193, 7],
        'Pumped Storage Hydroelectricity': [33, 150, 243],  # Blue
        'Pumped Storage': [33, 150, 243],
        'Pumped Hydro': [33, 150, 243],
        'pumped_hydro': [33, 150, 243],
        'Flywheel': [156, 39, 176],             # Purple
        'flywheel': [156, 39, 176],
        'CAES': [255, 152, 0],                  # Orange
        'Liquid Air Energy Storage': [0, 188, 212],  # Cyan
        'LAES': [0, 188, 212],
    }
    
    # Add color column with fallback for unmapped carriers
    storage_df['color'] = storage_df['carrier'].map(carrier_color_rgb).fillna(pd.Series([[100, 100, 100]] * len(storage_df)))
    # Add alpha channel (200) to each RGB color
    storage_df['color'] = storage_df['color'].apply(lambda c: c + [200] if isinstance(c, list) else [100, 100, 100, 200])
    
    # Ensure index column is renamed to 'name' for tooltips
    if 'name' not in storage_df.columns:
        # Get the first column (the index that was reset)
        name_col = storage_df.columns[0]
        storage_df = storage_df.rename(columns={name_col: 'name'})
    
    # Ensure all tooltip fields are present and formatted properly
    storage_df['capacity_str'] = storage_df['p_nom'].apply(lambda x: f"{x:.1f}")
    storage_df['energy_str'] = storage_df['e_nom'].apply(lambda x: f"{x:.1f}")
    storage_df['duration_str'] = storage_df['max_hours'].apply(lambda x: f"{x:.2f}")
    
    # Calculate marker sizes with linear scaling
    min_radius = 100      # Minimum radius in meters
    max_radius = 10000    # Maximum radius in meters
    
    min_capacity = storage_df['p_nom'].min()
    max_capacity = storage_df['p_nom'].max()
    capacity_range = max_capacity - min_capacity
    
    if capacity_range > 0:
        storage_df['radius'] = min_radius + (storage_df['p_nom'] - min_capacity) / capacity_range * (max_radius - min_radius)
    else:
        storage_df['radius'] = min_radius
    
    logger.info(f"Marker radius range: {storage_df['radius'].min():.1f} to {storage_df['radius'].max():.1f} meters")
    logger.info(f"Capacity range: {storage_df['p_nom'].min():.1f} to {storage_df['p_nom'].max():.1f} MW")
    
    storage_layer = pdk.Layer(
        "ScatterplotLayer",
        storage_df,
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=5,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        get_line_color=[0, 0, 0, 255],
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
    
    # Center view on GB
    view_state = pdk.ViewState(
        longitude=-2.5,
        latitude=55.0,
        zoom=5.5,
        pitch=0,
    )
    
    logger.info(f"Creating pydeck visualization with {len(storage_df)} storage units")
    
    deck = pdk.Deck(
        layers=[bus_layer, storage_layer],  # Buses behind, storage on top
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Storage: {name}</b><br/>"
                   "Technology: {carrier}<br/>"
                   "Power: {capacity_str} MW<br/>"
                   "Energy: {energy_str} MWh<br/>"
                   "Duration: {duration_str} h<br/>"
                   "Bus: {bus}",
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
    logger.info(f"Custom interactive storage map created with {len(storage_df)} units")
    logger.info(f"Saved to: {interactive_plot_path}")
    interactive_plot_status = f"[OK] Custom pydeck map with {len(storage_df)} units (color-coded by technology)"
    
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
    <title>Storage Unit Visualization - {scenario}</title>
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
        .battery {{ background-color: #e8f4f8; color: #0277bd; }}
        .pumped_hydro {{ background-color: #e8f5e9; color: #2e7d32; }}
        .caes {{ background-color: #fff3e0; color: #e65100; }}
        .laes {{ background-color: #f3e5f5; color: #6a1b9a; }}
        .other {{ background-color: #ede7f6; color: #3f51b5; }}
        .warning {{ background-color: #fff3cd; border-left-color: #ffc107; }}
        .info {{ color: #666; font-size: 13px; margin-top: 10px; }}
        .status-ok {{ color: green; }}
        .status-error {{ color: red; }}
        .status-skip {{ color: orange; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Energy Storage Units Visualization</h1>
        <p>Scenario: <strong>{scenario}</strong></p>
    </div>
    
    <div class="section">
        <h2>Storage Summary</h2>
        <div class="summary">
            <div class="stat-box">
                <div class="stat-label">Total Storage Units</div>
                <div class="stat-value">{len(storage_units)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Power Capacity</div>
                <div class="stat-value">{storage_units['p_nom'].sum():.0f} MW</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Energy Capacity</div>
                <div class="stat-value">{storage_units['e_nom'].sum():.0f} MWh</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Technologies</div>
                <div class="stat-value">{storage_units['carrier'].nunique()}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Storage by Technology</h2>
        <table>
            <thead>
                <tr>
                    <th>Technology</th>
                    <th>Units</th>
                    <th>Power (MW)</th>
                    <th>Energy (MWh)</th>
                    <th>Avg Duration (h)</th>
                    <th>Avg Efficiency (%)</th>
                </tr>
            </thead>
            <tbody>
"""

# Add technology breakdown rows
for tech in sorted(storage_units['carrier'].unique()):
    tech_units = storage_units[storage_units['carrier'] == tech]
    p_nom_total = tech_units['p_nom'].sum()
    e_nom_total = tech_units['e_nom'].sum()
    avg_duration = (e_nom_total / p_nom_total * 100) if p_nom_total > 0 else 0
    
    # Calculate average efficiency (round-trip)
    if 'efficiency_dispatch' in tech_units.columns and 'efficiency_store' in tech_units.columns:
        avg_efficiency = (tech_units['efficiency_dispatch'] * tech_units['efficiency_store']).mean() * 100
    else:
        avg_efficiency = 85.0  # Default assumption
    
    # Determine badge color
    if 'battery' in tech.lower():
        badge_class = 'battery'
    elif 'hydro' in tech.lower():
        badge_class = 'pumped_hydro'
    elif 'caes' in tech.lower():
        badge_class = 'caes'
    elif 'laes' in tech.lower():
        badge_class = 'laes'
    else:
        badge_class = 'other'
    
    html_content += f"""
                <tr>
                    <td><span class="carrier-badge {badge_class}">{tech}</span></td>
                    <td>{len(tech_units)}</td>
                    <td>{p_nom_total:.1f}</td>
                    <td>{e_nom_total:.1f}</td>
                    <td>{avg_duration:.2f}</td>
                    <td>{avg_efficiency:.1f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>Top 10 Largest Storage Facilities</h2>
        <table>
            <thead>
                <tr>
                    <th>Storage Unit</th>
                    <th>Technology</th>
                    <th>Bus</th>
                    <th>Power (MW)</th>
                    <th>Energy (MWh)</th>
                    <th>Duration (h)</th>
                </tr>
            </thead>
            <tbody>
"""

# Add top 10 storage units
top_storage = storage_units.nlargest(10, 'p_nom')
for idx, (su_name, su_row) in enumerate(top_storage.iterrows(), 1):
    duration = (su_row['e_nom'] / su_row['p_nom']) if su_row['p_nom'] > 0 else 0
    html_content += f"""
                <tr>
                    <td>{su_name}</td>
                    <td><span class="carrier-badge">{su_row['carrier']}</span></td>
                    <td>{su_row['bus']}</td>
                    <td>{su_row['p_nom']:.1f}</td>
                    <td>{su_row['e_nom']:.1f}</td>
                    <td>{duration:.2f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>Data Quality</h2>
"""

# Check coordinate completeness
storage_has_complete_coords = storage_has_coords_xy | storage_has_coords_lonlat
if storage_has_complete_coords.sum() < len(storage_units):
    html_content += f"""
        <div class="stat-box warning">
            ⚠️  {len(storage_units) - storage_has_complete_coords.sum()} storage units missing coordinate data
        </div>
"""

html_content += f"""
        <div class="info">
            <strong>Network Information:</strong><br>
            • Total buses: {len(n.buses)}<br>
            • Total generators: {len(n.generators)}<br>
            • Total storage units: {len(n.storage_units)}<br>
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
        <h2>Static Storage Unit Map</h2>
        <img src="{static_rel.as_posix()}" alt="Storage units topology plot" style="max-width:100%; height:auto;"/>
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
        <h2>Interactive Storage Unit Explorer</h2>
        <p>Explore storage locations and network buses interactively. 
        Hover over buses and storage units for details. Storage size reflects power capacity.</p>
        <iframe src="{interactive_rel.as_posix()}" title="Interactive storage explorer" 
        style="width:100%; height:700px; border:1px solid #ccc;"></iframe>
    </div>
"""

html_content += """
    
    <div class="section info" style="text-align: center; color: #999; font-size: 12px;">
        <p>This report was automatically generated by plot_storage_pypsa.py</p>
        <p>Storage data verified and visualized using PyPSA network components</p>
    </div>
</body>
</html>
"""

# Write HTML report
try:
    output_html.write_text(html_content)
    logger.info(f"✓ Storage visualization report saved to: {output_html}")
except Exception as e:
    logger.error(f"Failed to write HTML output: {e}")
    sys.exit(1)

# Log execution summary
execution_time = time.time() - start_time
logger.info(f"Execution time: {execution_time:.2f} seconds")
logger.info("Storage visualization completed successfully")

