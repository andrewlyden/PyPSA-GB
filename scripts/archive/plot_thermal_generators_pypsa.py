"""
Plot thermal/dispatchable generators from a PyPSA network using PyPSA's native plotting functions.

This script creates visualizations of thermal generators (gas, coal, nuclear, etc.)
to verify generator data integration and locations in the network.

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
log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "plot_thermal_generators_pypsa"
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
    output_html.write_text(f"<html><body><h1>Thermal Generator Plot</h1><p>[ERROR] Failed to load network: {html.escape(str(e))}</p></body></html>")
    sys.exit(1)

# Check for generators
if n.generators.empty:
    logger.warning("No generators found in network; creating placeholder report")
    output_html.write_text("<html><body><h1>Thermal Generator Plot</h1><h3>No generators found in network</h3></body></html>")
    sys.exit(0)

# Filter for thermal/dispatchable generators (non-renewable)
renewable_carriers = [
    'wind_onshore', 'wind_offshore', 'solar_pv', 'small_hydro', 'large_hydro',
    'tidal_stream', 'shoreline_wave', 'tidal_lagoon', 'geothermal',
    'Wind (Onshore)', 'Wind (Offshore)', 'Solar', 'Hydro',
    'wind', 'solar', 'onwind', 'offwind'
]

# Get non-renewable generators (excluding load shedding which is just backup)
thermal_gens = n.generators[
    ~n.generators['carrier'].isin(renewable_carriers) & 
    (n.generators['carrier'] != 'load_shedding')
].copy()

if thermal_gens.empty:
    logger.warning("No thermal generators found in network")
    output_html.write_text(
        f"<html><body><h1>Thermal Generator Plot - {scenario}</h1>"
        f"<p>Total generators in network: {len(n.generators)}</p>"
        f"<p>Thermal generators: 0</p>"
        f"<p>Available carriers: {', '.join(n.generators['carrier'].unique())}</p>"
        "</body></html>"
    )
    sys.exit(0)

logger.info(f"Found {len(thermal_gens)} thermal generators (excluding load shedding)")
logger.info(f"Thermal carriers: {thermal_gens['carrier'].value_counts().to_dict()}")
logger.info(f"Total thermal capacity: {thermal_gens['p_nom'].sum():.2f} MW")

# Log breakdown by major carrier types
major_carriers = ['CCGT', 'OCGT', 'AGR', 'PWR', 'Coal', 'biomass', 'waste_to_energy']
for carrier in major_carriers:
    carrier_gens = thermal_gens[thermal_gens['carrier'] == carrier]
    if not carrier_gens.empty:
        logger.info(f"  {carrier}: {len(carrier_gens)} generators, {carrier_gens['p_nom'].sum():.1f} MW total")

# Coordinate harmonization for plotting
def harmonize_coordinates():
    """Ensure buses have proper coordinates for plotting (OSGB36 -> WGS84 conversion)."""
    buses = n.buses
    
    # Check if lon/lat exist and are valid (in degree range)
    def is_valid_degrees(series, max_val):
        if series is None or series.name not in buses.columns:
            return False
        valid = pd.to_numeric(buses[series.name], errors='coerce').dropna()
        if valid.empty:
            return False
        return (valid.abs() <= max_val).all()
    
    has_valid_lon = is_valid_degrees(pd.Series(name='lon'), 180.0)
    has_valid_lat = is_valid_degrees(pd.Series(name='lat'), 90.0)
    has_x = 'x' in buses.columns and buses['x'].notna().any()
    has_y = 'y' in buses.columns and buses['y'].notna().any()
    
    # If we already have valid WGS84 coordinates, we're done
    if has_valid_lon and has_valid_lat:
        logger.info("Found valid WGS84 coordinates (lon/lat)")
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
                    # Fallback conversion (simplified, less accurate)
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
            else:
                # x/y appear to be in degrees already
                logger.info("x/y appear to be in degree coordinates")
                buses['lon'] = buses['x']
                buses['lat'] = buses['y']
                return True
    
    logger.warning("No valid coordinates found in buses")
    return False

has_coords = harmonize_coordinates()

if not has_coords:
    output_html.write_text(
        f"<html><body><h1>Thermal Generator Plot - {scenario}</h1>"
        f"<p>[ERROR] No valid bus coordinates found for plotting</p>"
        f"<p>Thermal generators: {len(thermal_gens)}</p>"
        "</body></html>"
    )
    sys.exit(1)

# Generate static matplotlib plot
plot_dir = output_html.parent / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)
static_plot_path = plot_dir / f"{scenario}_thermal_topology.png"
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
        title=f"Thermal Generators - {scenario}"
    )
    
    # Overlay thermal generator locations as colored markers
    gen_buses = thermal_gens['bus'].values
    bus_coords = n.buses.loc[gen_buses]
    
    # Use x/y for plotting (OSGB36 meters)
    lon_col, lat_col = 'x', 'y'
    
    # Color code by carrier (thermal technologies)
    carrier_colors = {
        'gas': 'gray',
        'ccgt': 'gray',
        'CCGT': 'gray',
        'ocgt': 'lightgray',
        'OCGT': 'lightgray',
        'coal': 'black',
        'Coal': 'black',
        'nuclear': 'purple',
        'Nuclear': 'purple',
        'oil': 'brown',
        'Oil': 'brown',
        'biomass': 'green',
        'Biomass': 'green',
        'battery': 'yellow',
        'Battery': 'yellow',
        'pumped_hydro': 'darkblue',
        'Pumped Hydro': 'darkblue',
    }
    
    # Calculate marker sizes with linear scaling (2x bigger than before)
    min_marker_size = 40    # Minimum size in points^2 (was 20)
    max_marker_size = 1000  # Maximum size in points^2 (was 500)
    
    # Get capacity range across all thermal generators
    min_capacity = thermal_gens['p_nom'].min()
    max_capacity = thermal_gens['p_nom'].max()
    capacity_range = max_capacity - min_capacity
    
    # Plot each carrier type
    for carrier in thermal_gens['carrier'].unique():
        carrier_gens = thermal_gens[thermal_gens['carrier'] == carrier]
        carrier_buses = n.buses.loc[carrier_gens['bus'].values]
        
        # Skip if no valid coordinates
        if carrier_buses[lon_col].isna().all() or carrier_buses[lat_col].isna().all():
            logger.warning(f"Skipping {carrier} - no valid coordinates")
            continue
        
        color = carrier_colors.get(carrier, 'red')
        
        # Linear scaling between min and max marker sizes
        if capacity_range > 0:
            sizes = min_marker_size + (carrier_gens['p_nom'].values - min_capacity) / capacity_range * (max_marker_size - min_marker_size)
        else:
            sizes = np.full(len(carrier_gens), min_marker_size)
        
        ax.scatter(
            carrier_buses[lon_col].values,
            carrier_buses[lat_col].values,
            s=sizes,
            c=color,
            alpha=0.6,
            label=f"{carrier} ({len(carrier_gens)} units, {carrier_gens['p_nom'].sum():.0f} MW)",
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
    
    logger.info(f"Static thermal plot saved to: {static_plot_path}")
    static_plot_status = "[OK] Static thermal plot generated (OSGB36 coordinates)"
    
except ImportError as exc:
    logger.warning(f"Matplotlib not available for plotting: {exc}")
    static_plot_status = f"[SKIP] Static plot (matplotlib missing)"
    static_plot_path = None
except Exception as exc:
    logger.error(f"Failed to create static plot: {exc}", exc_info=True)
    static_plot_status = f"[ERROR] Static plot failed: {exc}"
    static_plot_path = None

# Generate interactive pydeck plot
interactive_plot_path = output_html.parent / f"{scenario}_thermal_explore.html"
interactive_plot_status = None

try:
    import pydeck as pdk
    
    logger.info(f"Creating custom pydeck visualization for {len(thermal_gens)} thermal generators")
    
    # Prepare generator data with coordinates from their buses
    gen_df = thermal_gens.copy()
    
    # Check if generators have valid WGS84 coordinates (lat should be ~50-60, lon should be ~-8 to 2 for GB)
    has_valid_wgs84 = False
    if 'lon' in gen_df.columns and 'lat' in gen_df.columns:
        # Check if coordinates are in WGS84 range (degrees)
        lon_vals = gen_df['lon'].dropna()
        lat_vals = gen_df['lat'].dropna()
        if len(lon_vals) > 0 and len(lat_vals) > 0:
            # WGS84 coordinates for GB should be: lat ~49-61, lon ~-8 to 2
            is_wgs84 = (lon_vals.between(-10, 5).all() and lat_vals.between(48, 62).all())
            if is_wgs84:
                logger.info("Generators have valid WGS84 coordinates (lon/lat)")
                has_valid_wgs84 = True
            else:
                # Coordinates appear to be in OSGB36 (British National Grid) - convert them
                logger.warning(f"Generator lon/lat appear to be in OSGB36 (sample: E={lon_vals.iloc[0]:.1f}, N={lat_vals.iloc[0]:.1f})")
                logger.info("Attempting to convert OSGB36 coordinates to WGS84")
                
                try:
                    from pyproj import Transformer
                    # EPSG:27700 is OSGB36 (British National Grid)
                    # EPSG:4326 is WGS84 (lat/lon)
                    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                    
                    # Convert coordinates (OSGB36 uses easting/northing, not lon/lat)
                    lons_wgs84, lats_wgs84 = transformer.transform(gen_df['lon'].values, gen_df['lat'].values)
                    gen_df['lon'] = lons_wgs84
                    gen_df['lat'] = lats_wgs84
                    logger.info(f"[OK] Converted {len(gen_df)} generator coordinates from OSGB36 to WGS84")
                    logger.info(f"New range: lon {gen_df['lon'].min():.2f} to {gen_df['lon'].max():.2f}, lat {gen_df['lat'].min():.2f} to {gen_df['lat'].max():.2f}")
                    has_valid_wgs84 = True
                except ImportError:
                    logger.warning("pyproj not available for coordinate conversion, will fall back to bus coordinates")
                except Exception as e:
                    logger.warning(f"Coordinate conversion failed: {e}, will fall back to bus coordinates")
    
    if not has_valid_wgs84:
        # Need to get coordinates from buses
        logger.info("Getting WGS84 coordinates from network buses")
        # Make sure buses have WGS84 coordinates
        if 'lon' not in n.buses.columns or 'lat' not in n.buses.columns:
            logger.error("Network buses don't have lon/lat coordinates!")
            raise ValueError("Network buses missing WGS84 coordinates")
        
        # Drop existing lon/lat if present (they're in wrong coordinate system)
        if 'lon' in gen_df.columns:
            gen_df = gen_df.drop(columns=['lon', 'lat'])
        
        # Merge bus coordinates
        gen_df = gen_df.merge(n.buses[['lon', 'lat']], left_on='bus', right_index=True, how='left')
        logger.info(f"Merged bus coordinates for {len(gen_df)} generators")
    
    # Filter to valid coordinates and reset index to get generator names as column
    gen_df = gen_df[gen_df['lon'].notna() & gen_df['lat'].notna()].reset_index()
    
    logger.info(f"Creating custom pydeck map with {len(gen_df)} generators at {gen_df['bus'].nunique()} unique locations")
    logger.info(f"Coordinate range check: lon {gen_df['lon'].min():.2f} to {gen_df['lon'].max():.2f}, lat {gen_df['lat'].min():.2f} to {gen_df['lat'].max():.2f}")
    
    # Map carriers to colors (RGB tuples)
    carrier_color_rgb = {
        'gas': [128, 128, 128],          # Gray
        'ccgt': [128, 128, 128],         # Gray
        'CCGT': [128, 128, 128],         # Gray
        'ocgt': [192, 192, 192],         # Light gray
        'OCGT': [192, 192, 192],         # Light gray
        'coal': [0, 0, 0],               # Black
        'Coal': [0, 0, 0],               # Black
        'nuclear': [147, 112, 219],      # Purple
        'Nuclear': [147, 112, 219],      # Purple
        'oil': [139, 69, 19],            # Brown
        'Oil': [139, 69, 19],            # Brown
        'biomass': [34, 139, 34],        # Forest green
        'Biomass': [34, 139, 34],        # Forest green
        'battery': [255, 255, 0],        # Yellow
        'Battery': [255, 255, 0],        # Yellow
        'pumped_hydro': [0, 0, 139],     # Dark blue
        'Pumped Hydro': [0, 0, 139],     # Dark blue
    }
    
    # Add color column with fallback for unmapped carriers
    gen_df['color'] = gen_df['carrier'].map(carrier_color_rgb).fillna(pd.Series([[100, 100, 100]] * len(gen_df)))
    # Add alpha channel (200) to each RGB color
    gen_df['color'] = gen_df['color'].apply(lambda c: c + [200] if isinstance(c, list) else [100, 100, 100, 200])
    
    # Ensure all tooltip fields are present and formatted properly
    gen_df['capacity_str'] = gen_df['p_nom'].apply(lambda x: f"{x:.1f}")
    gen_df['marginal_cost_str'] = gen_df.get('marginal_cost', pd.Series(['N/A'] * len(gen_df))).apply(
        lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
    )
    gen_df['efficiency_str'] = gen_df.get('efficiency', pd.Series(['N/A'] * len(gen_df))).apply(
        lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
    )
    
    logger.info(f"Generator dataframe columns: {gen_df.columns.tolist()}")
    logger.info(f"Sample generator data:\n{gen_df[['name', 'carrier', 'p_nom', 'lon', 'lat', 'bus']].head()}")
    
    # Calculate marker sizes with linear scaling (2x bigger than before)
    min_radius = 100      # Minimum radius in meters (was 50)
    max_radius = 10000    # Maximum radius in meters (was 5000)
    
    min_capacity = gen_df['p_nom'].min()
    max_capacity = gen_df['p_nom'].max()
    capacity_range = max_capacity - min_capacity
    
    if capacity_range > 0:
        gen_df['radius'] = min_radius + (gen_df['p_nom'] - min_capacity) / capacity_range * (max_radius - min_radius)
    else:
        gen_df['radius'] = min_radius
    
    logger.info(f"Marker radius range: {gen_df['radius'].min():.1f} to {gen_df['radius'].max():.1f} meters")
    logger.info(f"Capacity range: {gen_df['p_nom'].min():.1f} to {gen_df['p_nom'].max():.1f} MW")
    
    gen_layer = pdk.Layer(
        "ScatterplotLayer",
        gen_df,
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=5,      # Increased from 3
        radius_max_pixels=100,    # Increased from 50 for better visibility
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
    
    logger.info(f"Creating pydeck visualization with {len(gen_df)} generators")
    
    deck = pdk.Deck(
        layers=[bus_layer, gen_layer],  # Buses behind, generators on top
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Generator: {name}</b><br/>"
                   "Carrier: {carrier}<br/>"
                   "Capacity: {capacity_str} MW<br/>"
                   "Marginal Cost: {marginal_cost_str}<br/>"
                   "Efficiency: {efficiency_str}<br/>"
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
    logger.info(f"Custom interactive thermal generator map created with {len(gen_df)} generators")
    logger.info(f"Saved to: {interactive_plot_path}")
    interactive_plot_status = f"[OK] Custom pydeck map with {len(gen_df)} generators (color-coded by carrier)"
    
except ImportError as exc:
    logger.warning(f"Pydeck not available for interactive plotting: {exc}")
    interactive_plot_status = "[SKIP] Interactive plot (pydeck missing)"
    interactive_plot_path = None
except Exception as exc:
    logger.error(f"Failed to create interactive plot: {exc}", exc_info=True)
    interactive_plot_status = f"[ERROR] Interactive plot failed: {exc}"
    interactive_plot_path = None

# Generate summary statistics
carrier_summary = thermal_gens.groupby('carrier').agg({
    'p_nom': ['count', 'sum', 'mean', 'min', 'max']
}).round(2)

# Add efficiency and marginal cost stats if available
if 'efficiency' in thermal_gens.columns:
    efficiency_summary = thermal_gens.groupby('carrier')['efficiency'].agg(['mean', 'min', 'max']).round(3)
else:
    efficiency_summary = None

if 'marginal_cost' in thermal_gens.columns:
    marginal_cost_summary = thermal_gens.groupby('carrier')['marginal_cost'].agg(['mean', 'min', 'max']).round(2)
else:
    marginal_cost_summary = None

# Create HTML report
html_report = [
    "<html>",
    "<head>",
    "<title>Thermal Generator Visualization Report</title>",
    "<style>",
    "body { font-family: Arial, sans-serif; margin: 20px; }",
    "table { border-collapse: collapse; margin: 20px 0; }",
    "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
    "th { background-color: #4CAF50; color: white; }",
    "tr:nth-child(even) { background-color: #f2f2f2; }",
    ".status-ok { color: green; }",
    ".status-error { color: red; }",
    ".status-skip { color: orange; }",
    "</style>",
    "</head>",
    "<body>",
    f"<h1>Thermal Generator Visualization - {scenario}</h1>",
    f"<p><strong>Network file:</strong> {network_path}</p>",
    f"<p><strong>Total generators:</strong> {len(n.generators)}</p>",
    f"<p><strong>Thermal generators:</strong> {len(thermal_gens)}</p>",
    f"<p><strong>Total thermal capacity:</strong> {thermal_gens['p_nom'].sum():.2f} MW</p>",
    "",
    "<h2>Thermal Generation by Carrier</h2>",
    "<table>",
    "<tr><th>Carrier</th><th>Count</th><th>Total (MW)</th><th>Mean (MW)</th><th>Min (MW)</th><th>Max (MW)</th></tr>",
]

# Add carrier summary to table
for carrier, row in carrier_summary.iterrows():
    html_report.append(
        f"<tr>"
        f"<td>{carrier}</td>"
        f"<td>{int(row['p_nom']['count'])}</td>"
        f"<td>{row['p_nom']['sum']:.2f}</td>"
        f"<td>{row['p_nom']['mean']:.2f}</td>"
        f"<td>{row['p_nom']['min']:.2f}</td>"
        f"<td>{row['p_nom']['max']:.2f}</td>"
        f"</tr>"
    )

html_report.extend([
    "</table>",
])

# Add efficiency table if available
if efficiency_summary is not None and not efficiency_summary.empty:
    html_report.extend([
        "",
        "<h2>Efficiency by Carrier</h2>",
        "<table>",
        "<tr><th>Carrier</th><th>Mean Efficiency</th><th>Min Efficiency</th><th>Max Efficiency</th></tr>",
    ])
    
    for carrier, row in efficiency_summary.iterrows():
        html_report.append(
            f"<tr>"
            f"<td>{carrier}</td>"
            f"<td>{row['mean']:.3f}</td>"
            f"<td>{row['min']:.3f}</td>"
            f"<td>{row['max']:.3f}</td>"
            f"</tr>"
        )
    
    html_report.extend([
        "</table>",
    ])

# Add marginal cost table if available
if marginal_cost_summary is not None and not marginal_cost_summary.empty:
    html_report.extend([
        "",
        "<h2>Marginal Cost by Carrier (£/MWh)</h2>",
        "<table>",
        "<tr><th>Carrier</th><th>Mean Cost</th><th>Min Cost</th><th>Max Cost</th></tr>",
    ])
    
    for carrier, row in marginal_cost_summary.iterrows():
        html_report.append(
            f"<tr>"
            f"<td>{carrier}</td>"
            f"<td>{row['mean']:.2f}</td>"
            f"<td>{row['min']:.2f}</td>"
            f"<td>{row['max']:.2f}</td>"
            f"</tr>"
        )
    
    html_report.extend([
        "</table>",
    ])

html_report.extend([
    "",
    "<h2>Visualization Status</h2>",
    "<ul>",
])

# Add status messages
if static_plot_status:
    status_class = "status-ok" if "[OK]" in static_plot_status else ("status-error" if "[ERROR]" in static_plot_status else "status-skip")
    html_report.append(f"<li class=\"{status_class}\">{html.escape(static_plot_status)}</li>")

if interactive_plot_status:
    status_class = "status-ok" if "[OK]" in interactive_plot_status else ("status-error" if "[ERROR]" in interactive_plot_status else "status-skip")
    html_report.append(f"<li class=\"{status_class}\">{html.escape(interactive_plot_status)}</li>")

html_report.extend([
    "</ul>",
])

# Embed static plot if available
if static_plot_path and static_plot_path.exists():
    try:
        static_rel = static_plot_path.relative_to(output_html.parent)
    except ValueError:
        static_rel = static_plot_path.name
    
    html_report.extend([
        "<h2>Static Thermal Generator Map</h2>",
        f"<img src=\"{static_rel.as_posix()}\" alt=\"Thermal generators topology plot\" style=\"max-width:100%; height:auto;\"/>",
    ])

# Embed interactive plot if available
if interactive_plot_path and interactive_plot_path.exists():
    try:
        interactive_rel = interactive_plot_path.relative_to(output_html.parent)
    except ValueError:
        interactive_rel = interactive_plot_path.name
    
    html_report.extend([
        "<h2>Interactive Thermal Generator Explorer</h2>",
        "<p>Explore thermal generator locations and network buses interactively. "
        "Hover over buses and generators for details. Generator size reflects capacity.</p>",
        f"<iframe src=\"{interactive_rel.as_posix()}\" title=\"Interactive thermal generator explorer\" "
        "style=\"width:100%; height:700px; border:1px solid #ccc;\"></iframe>",
    ])

html_report.extend([
    "</body>",
    "</html>"
])

# Write HTML report
output_html.write_text("\n".join(html_report), encoding="utf-8")
logger.info(f"Thermal generator visualization report written to: {output_html}")

# Log execution summary
log_execution_summary(
    logger, "Thermal Generator Visualization", start_time,
    inputs=[str(network_path)],
    outputs=[str(output_html)]
)

logger.info("Thermal generator visualization completed successfully")

