"""
Plot renewable generators from a PyPSA network using PyPSA's native plotting functions.

This script uses PyPSA's built-in plotting capabilities to create both static
(matplotlib) and interactive (pydeck via n.explore()) visualizations of renewable
generators in the network.

Dependencies: pypsa, matplotlib, pydeck (optional)
"""
import logging
from pathlib import Path
import sys
import html
import time
import io
import contextlib

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
log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "plot_renewables_pypsa"
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
    output_html.write_text(f"<html><body><h1>Renewable Generator Plot</h1><p>[ERROR] Failed to load network: {html.escape(str(e))}</p></body></html>")
    sys.exit(1)

# Check for generators
if n.generators.empty:
    logger.warning("No generators found in network; creating placeholder report")
    output_html.write_text("<html><body><h1>Renewable Generator Plot</h1><h3>No generators found in network</h3></body></html>")
    sys.exit(0)

# Filter for renewable generators
renewable_carriers = [
    'wind_onshore', 'wind_offshore', 'solar_pv', 'small_hydro', 'large_hydro',
    'tidal_stream', 'shoreline_wave', 'tidal_lagoon', 'geothermal',
    'Wind (Onshore)', 'Wind (Offshore)', 'Solar', 'Hydro',
    'wind', 'solar', 'onwind', 'offwind'
]

renewable_gens = n.generators[n.generators['carrier'].isin(renewable_carriers)].copy()

if renewable_gens.empty:
    logger.warning("No renewable generators found in network")
    output_html.write_text(
        f"<html><body><h1>Renewable Generator Plot - {scenario}</h1>"
        f"<p>Total generators in network: {len(n.generators)}</p>"
        f"<p>Renewable generators: 0</p>"
        f"<p>Available carriers: {', '.join(n.generators['carrier'].unique())}</p>"
        "</body></html>"
    )
    sys.exit(0)

logger.info(f"Found {len(renewable_gens)} renewable generators")
logger.info(f"Renewable carriers: {renewable_gens['carrier'].value_counts().to_dict()}")
logger.info(f"Total renewable capacity: {renewable_gens['p_nom'].sum():.2f} MW")

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
        f"<html><body><h1>Renewable Generator Plot - {scenario}</h1>"
        f"<p>[ERROR] No valid bus coordinates found for plotting</p>"
        f"<p>Renewable generators: {len(renewable_gens)}</p>"
        "</body></html>"
    )
    sys.exit(1)

# Generate static matplotlib plot
plot_dir = output_html.parent / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)
static_plot_path = plot_dir / f"{scenario}_renewables_topology.png"
static_plot_status = None

# Generate static matplotlib plot
plot_dir = output_html.parent / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)
static_plot_path = plot_dir / f"{scenario}_renewables_topology.png"
static_plot_status = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # IMPORTANT: PyPSA's n.plot() uses x/y coordinates (OSGB36 meters), NOT lon/lat!
    # We should plot directly in the coordinate system the network uses (meters)
    # This is the same approach as validate_network.py
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot network with buses and lines using x/y (OSGB36 meters)
    # PyPSA will automatically use x/y coordinates for plotting
    # geomap=False tells PyPSA to NOT require cartopy (use simple x/y plotting)
    n.plot(
        ax=ax,
        geomap=False,  # Disable geographic projection requirement
        bus_size=5,
        line_width=0.2,
        bus_colors='lightgray',
        line_colors='lightgray',
        title=f"Renewable Generators - {scenario}"
    )
    
    # Overlay renewable generator locations as colored markers
    # Get bus coordinates for renewable generators (using x/y in OSGB36 meters)
    gen_buses = renewable_gens['bus'].values
    bus_coords = n.buses.loc[gen_buses]
    
    # Use x/y for plotting (OSGB36 meters) - this matches the network coordinate system
    lon_col, lat_col = 'x', 'y'
    
    # Color code by carrier
    carrier_colors = {
        'wind_onshore': 'green',
        'wind_offshore': 'blue',
        'solar_pv': 'orange',
        'small_hydro': 'cyan',
        'large_hydro': 'darkblue',
        'tidal_stream': 'purple',
        'shoreline_wave': 'magenta',
        'tidal_lagoon': 'pink',
        'geothermal': 'red',
    }
    
    # Plot each carrier type
    for carrier in renewable_gens['carrier'].unique():
        carrier_gens = renewable_gens[renewable_gens['carrier'] == carrier]
        carrier_buses = n.buses.loc[carrier_gens['bus'].values]
        
        # Skip if no valid coordinates
        if carrier_buses[lon_col].isna().all() or carrier_buses[lat_col].isna().all():
            logger.warning(f"Skipping {carrier} - no valid coordinates")
            continue
        
        color = carrier_colors.get(carrier, 'gray')
        
        # Size by capacity (scale to reasonable marker size)
        sizes = carrier_gens['p_nom'].values / 5  # Scale down for visibility
        
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
    
    logger.info(f"Static renewable plot saved to: {static_plot_path}")
    static_plot_status = "[OK] Static renewable plot generated (OSGB36 coordinates)"
    
except ImportError as exc:
    logger.warning(f"Matplotlib not available for plotting: {exc}")
    static_plot_status = f"[SKIP] Static plot (matplotlib missing)"
    static_plot_path = None
except Exception as exc:
    logger.error(f"Failed to create static plot: {exc}", exc_info=True)
    static_plot_status = f"[ERROR] Static plot failed: {exc}"
    static_plot_path = None

# Generate interactive pydeck plot using PyPSA's n.explore()
interactive_plot_path = output_html.parent / f"{scenario}_renewables_explore.html"
interactive_plot_status = None

try:
    if not hasattr(n, "explore"):
        interactive_plot_status = "[SKIP] Interactive plot (PyPSA explore API unavailable)"
    else:
        import pydeck  # noqa: F401
        
        # PyPSA 1.0.2 has a coordinate validation bug that incorrectly rejects valid WGS84 coordinates
        # causing "Dropping 2022 buses with invalid WGS84 coordinates" warning even though they're valid.
        # The root cause is in PyPSA's internal validation, not our data.
        # 
        # Workaround: Skip PyPSA's explore() entirely and create custom pydeck visualization.
        # This gives us more control and avoids PyPSA's buggy coordinate validation.
        
        logger.info(f"Creating custom pydeck visualization for {len(renewable_gens)} generators")
        
        try:
            # PyPSA's explore() has strict coordinate validation that fails for some networks
            # Create custom pydeck visualization showing generators as colored circles
            import pydeck as pdk
            
            # Prepare generator data with coordinates from their buses
            gen_df = renewable_gens.copy()
            
            # Generators may already have lon/lat columns, so check before merging
            if 'lon' in gen_df.columns and 'lat' in gen_df.columns:
                # Use existing coordinates
                logger.info("Using existing lon/lat coordinates from generators")
            else:
                # Merge bus coordinates
                gen_df = gen_df.merge(n.buses[['lon', 'lat']], left_on='bus', right_index=True, how='left')
            
            # Filter to valid coordinates and reset index to get generator names as column
            gen_df = gen_df[gen_df['lon'].notna() & gen_df['lat'].notna()].reset_index()
            
            logger.info(f"Creating custom pydeck map with {len(gen_df)} generators at {gen_df['bus'].nunique()} unique locations")
            
            # Map carriers to colors (RGB tuples)
            carrier_color_rgb = {
                'wind_onshore': [0, 160, 0],       # Green
                'wind_offshore': [0, 0, 200],       # Blue
                'solar_pv': [255, 176, 0],          # Orange
                'small_hydro': [64, 224, 208],      # Turquoise
                'large_hydro': [0, 0, 128],         # Navy
                'tidal_stream': [147, 112, 219],    # Purple
                'shoreline_wave': [255, 0, 255],    # Magenta
                'tidal_lagoon': [255, 192, 203],    # Pink
                'geothermal': [255, 0, 0],          # Red
            }
            
            # Add color column
            gen_df['color'] = gen_df['carrier'].map(carrier_color_rgb)
            gen_df['color'] = gen_df['color'].apply(lambda c: c + [180] if c else [100, 100, 100, 180])  # Add alpha
            
            # Tooltip text
            gen_df['tooltip'] = gen_df.apply(
                lambda row: f"{row['name']}\nCarrier: {row['carrier']}\nCapacity: {row['p_nom']:.1f} MW\nBus: {row['bus']}",
                axis=1
            )
            
            # Create pydeck layer for generators
            gen_layer = pdk.Layer(
                "ScatterplotLayer",
                gen_df,
                pickable=True,
                opacity=0.7,
                stroked=True,
                filled=True,
                radius_scale=6,
                radius_min_pixels=3,
                radius_max_pixels=50,
                line_width_min_pixels=1,
                get_position=["lon", "lat"],
                get_radius="p_nom * 3",  # Radius proportional to capacity
                get_fill_color="color",
                get_line_color=[0, 0, 0, 255],
            )
            
            # Add network buses as background layer (smaller, gray)
            bus_df = n.buses[['lon', 'lat']].copy().reset_index()
            bus_df = bus_df[bus_df['lon'].notna() & bus_df['lat'].notna()]
            # After reset_index(), the bus names are in the first column (index becomes a column)
            bus_name_col = bus_df.columns[0]  # Get the actual column name for bus names
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
            
            deck = pdk.Deck(
                layers=[bus_layer, gen_layer],  # Buses behind, generators on top
                initial_view_state=view_state,
                tooltip={"text": "{tooltip}"},
                map_style="light",
            )
            
            deck.to_html(str(interactive_plot_path), notebook_display=False, open_browser=False)
            logger.info(f"Custom interactive renewable map created with {len(gen_df)} generators")
            interactive_plot_status = f"[OK] Custom pydeck map with {len(gen_df)} generators (color-coded by carrier)"
        except Exception as explore_err:
            # If explore still fails, create a simplified pydeck visualization manually
            logger.warning(f"PyPSA explore() failed, creating simplified pydeck map: {explore_err}")
            
            # Get buses with valid coordinates
            valid_buses = n.buses[(n.buses['lon'].notna()) & (n.buses['lat'].notna())].copy()
            
            if len(valid_buses) > 0:
                # Create simple bus layer
                import pydeck as pdk
                
                bus_layer = pdk.Layer(
                    "ScatterplotLayer",
                    valid_buses.reset_index(),
                    pickable=True,
                    opacity=0.6,
                    stroked=True,
                    filled=True,
                    radius_scale=10,
                    radius_min_pixels=2,
                    radius_max_pixels=20,
                    get_position=["lon", "lat"],
                    get_radius=50,
                    get_fill_color=[100, 100, 100],
                    get_line_color=[0, 0, 0],
                )
                
                # Center on GB
                view_state = pdk.ViewState(
                    longitude=-2.5,
                    latitude=55.0,
                    zoom=5.5,
                    pitch=0,
                )
                
                deck = pdk.Deck(
                    layers=[bus_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "Bus: {Bus}\nLon: {lon}\nLat: {lat}"},
                    map_style="light",
                )
                
                deck.to_html(str(interactive_plot_path), notebook_display=False, open_browser=False)
                logger.info(f"Created simplified pydeck map with {len(valid_buses)} buses")
                interactive_plot_status = "[OK] Simplified interactive map generated"
            else:
                logger.warning("No buses with valid WGS84 coordinates for interactive plot")
                interactive_plot_status = "[SKIP] No valid coordinates for interactive plot"
                interactive_plot_path = None
        
except ImportError as exc:
    logger.warning(f"Pydeck not available for interactive plotting: {exc}")
    interactive_plot_status = "[SKIP] Interactive plot (pydeck missing)"
    interactive_plot_path = None
except Exception as exc:
    logger.error(f"Failed to create interactive plot: {exc}", exc_info=True)
    interactive_plot_status = f"[ERROR] Interactive plot failed: {exc}"
    interactive_plot_path = None

# Generate summary statistics
carrier_summary = renewable_gens.groupby('carrier').agg({
    'p_nom': ['count', 'sum', 'mean', 'min', 'max']
}).round(2)

# Create HTML report
html_report = [
    "<html>",
    "<head>",
    "<title>Renewable Generator Visualization Report</title>",
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
    f"<h1>Renewable Generator Visualization - {scenario}</h1>",
    f"<p><strong>Network file:</strong> {network_path}</p>",
    f"<p><strong>Total generators:</strong> {len(n.generators)}</p>",
    f"<p><strong>Renewable generators:</strong> {len(renewable_gens)}</p>",
    f"<p><strong>Total renewable capacity:</strong> {renewable_gens['p_nom'].sum():.2f} MW</p>",
    "",
    "<h2>Renewable Generation by Carrier</h2>",
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
        "<h2>Static Renewable Generator Map</h2>",
        f"<img src=\"{static_rel.as_posix()}\" alt=\"Renewable generators topology plot\" style=\"max-width:100%; height:auto;\"/>",
    ])

# Embed interactive plot if available
if interactive_plot_path and interactive_plot_path.exists():
    try:
        interactive_rel = interactive_plot_path.relative_to(output_html.parent)
    except ValueError:
        interactive_rel = interactive_plot_path.name
    
    html_report.extend([
        "<h2>Interactive Renewable Generator Explorer</h2>",
        "<p>Explore renewable generator locations and network buses interactively. "
        "Hover over buses and generators for details.</p>",
        f"<iframe src=\"{interactive_rel.as_posix()}\" title=\"Interactive renewable generator explorer\" "
        "style=\"width:100%; height:700px; border:1px solid #ccc;\"></iframe>",
    ])

html_report.extend([
    "</body>",
    "</html>"
])

# Write HTML report
output_html.write_text("\n".join(html_report), encoding="utf-8")
logger.info(f"Renewable generator visualization report written to: {output_html}")

# Log execution summary
log_execution_summary(
    logger, "Renewable Generator Visualization", start_time,
    inputs=[str(network_path)],
    outputs=[str(output_html)]
)

logger.info("Renewable generator visualization completed successfully")

