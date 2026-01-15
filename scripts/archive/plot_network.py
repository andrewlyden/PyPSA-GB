"""
General PyPSA network plotting and validation report writer for PyPSA-GB.

- Loads any PyPSA network file (base, clustered, or other)
- Runs consistency checks (network.consistency_check)
- Summarizes key statistics (buses, lines, generators, etc.)
- Outputs an HTML report with visualizations
"""

from pathlib import Path

import sys
import html
import time
import math
import io
import contextlib
import logging

import numpy as np
import pandas as pd
import pypsa

from scripts.utilities.logging_config import setup_logging, log_network_info, log_execution_summary


# Initialize timing
start_time = time.time()

# Use centralized logging: log to Snakemake log file if available, else default
log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "plot_network"
logger = setup_logging(log_path)

net_path = Path(snakemake.input.network)
# Use the correct output path specified by Snakemake (now outputs to plots directory)
out_html = Path(snakemake.output.html)
out_html.parent.mkdir(parents=True, exist_ok=True)

# Load network
try:
    n = pypsa.Network(net_path)
    load_status = "[OK] Loaded successfully"
    logger.info(f"Loaded network from {net_path}")
    log_network_info(n, logger)
except Exception as e:
    logger.error(f"Failed to load network: {e}")
    load_status = f"[ERROR] Failed to load: {e}"
    # Write minimal error report and exit
    out_html.write_text(f"<html><body><h1>Network Plot Report</h1><p>[ERROR] Failed to load network: {html.escape(str(e))}</p></body></html>")
    logger.error(f"Wrote error report to {out_html}")
    sys.exit(1)

# Coordinate preparation helpers
def _series_within_bounds(series: pd.Series, bound: float) -> bool:
    """Check whether all finite values in a series fall within ±bound."""
    if series is None:
        return False
    ser = pd.to_numeric(series, errors="coerce").dropna()
    if ser.empty:
        return False
    return (ser.abs() <= bound).all()


def _osgb36_to_wgs84(easting: np.ndarray, northing: np.ndarray, logger: logging.Logger):
    """
    Convert British National Grid (EPSG:27700) eastings/northings to WGS84 lon/lat.

    Implementation derived from Ordnance Survey formulas (see OSGB36 to WGS84 conversion).
    """
    # Ensure numpy arrays
    easting = np.asarray(easting, dtype=float)
    northing = np.asarray(northing, dtype=float)
    lon = np.full_like(easting, np.nan, dtype=float)
    lat = np.full_like(northing, np.nan, dtype=float)

    mask = ~np.isnan(easting) & ~np.isnan(northing)
    if not mask.any():
        return lon, lat

    E = easting[mask]
    N = northing[mask]

    # Airy 1830 ellipsoid parameters for OSGB36
    a = 6377563.396
    b = 6356256.909
    F0 = 0.9996012717
    lat0 = math.radians(49)
    lon0 = math.radians(-2)
    N0 = -100000.0
    E0 = 400000.0
    e2 = 1 - (b * b) / (a * a)
    n = (a - b) / (a + b)

    # Initial estimates of latitude
    phi = (N - N0) / (a * F0) + lat0

    def meridional_arc(phi_val):
        return (
            b * F0
            * (
                (1 + n + (5 / 4) * n**2 + (5 / 4) * n**3) * (phi_val - lat0)
                - (3 * n + 3 * n**2 + (21 / 8) * n**3) * np.sin(phi_val - lat0) * np.cos(phi_val + lat0)
                + ((15 / 8) * n**2 + (15 / 8) * n**3) * np.sin(2 * (phi_val - lat0)) * np.cos(2 * (phi_val + lat0))
                - (35 / 24) * n**3 * np.sin(3 * (phi_val - lat0)) * np.cos(3 * (phi_val + lat0))
            )
        )

    # Iterate to refine latitude
    for _ in range(6):
        phi = (N - N0 - meridional_arc(phi)) / (a * F0) + phi

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)

    nu = a * F0 / np.sqrt(1 - e2 * sin_phi**2)
    rho = a * F0 * (1 - e2) / (1 - e2 * sin_phi**2) ** 1.5
    eta2 = nu / rho - 1
    sec_phi = 1 / cos_phi
    dE = E - E0

    VII = tan_phi / (2 * rho * nu)
    VIII = tan_phi / (24 * rho * nu**3) * (5 + 3 * tan_phi**2 + eta2 - 9 * tan_phi**2 * eta2)
    IX = tan_phi / (720 * rho * nu**5) * (61 + 90 * tan_phi**2 + 45 * tan_phi**4)
    X = sec_phi / nu
    XI = sec_phi / (6 * nu**3) * (nu / rho + 2 * tan_phi**2)
    XII = sec_phi / (120 * nu**5) * (5 + 28 * tan_phi**2 + 24 * tan_phi**4)
    XIIA = sec_phi / (5040 * nu**7) * (61 + 662 * tan_phi**2 + 1320 * tan_phi**4 + 720 * tan_phi**6)

    phi_prime = (
        phi
        - VII * dE**2
        + VIII * dE**4
        - IX * dE**6
    )
    lam_prime = (
        lon0
        + X * dE
        - XI * dE**3
        + XII * dE**5
        - XIIA * dE**7
    )

    # Convert to cartesian coordinates (OSGB36)
    sin_phi_prime = np.sin(phi_prime)
    cos_phi_prime = np.cos(phi_prime)
    nu_prime = a * F0 / np.sqrt(1 - e2 * sin_phi_prime**2)
    x1 = (nu_prime) * cos_phi_prime * np.cos(lam_prime)
    y1 = (nu_prime) * cos_phi_prime * np.sin(lam_prime)
    z1 = (nu_prime * (1 - e2)) * sin_phi_prime

    # Helmert transformation to WGS84
    tx, ty, tz = 446.448, -125.157, 542.060
    s = 20.4894 * 1e-6
    rx = math.radians(0.1502 / 3600)
    ry = math.radians(0.2470 / 3600)
    rz = math.radians(0.8421 / 3600)

    x2 = tx + (1 + s) * x1 + (-rz) * y1 + (ry) * z1
    y2 = ty + (rz) * x1 + (1 + s) * y1 + (-rx) * z1
    z2 = tz + (-ry) * x1 + (rx) * y1 + (1 + s) * z1

    # Convert cartesian to lat/lon on GRS80
    a_wgs84 = 6378137.0
    b_wgs84 = 6356752.3141
    e2_wgs84 = 1 - (b_wgs84 * b_wgs84) / (a_wgs84 * a_wgs84)

    p = np.sqrt(x2**2 + y2**2)
    phi_w = np.arctan2(z2, p * (1 - e2_wgs84))

    for _ in range(6):
        nu_w = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi_w) ** 2)
        phi_w = np.arctan2(z2 + e2_wgs84 * nu_w * np.sin(phi_w), p)

    lam_w = np.arctan2(y2, x2)

    lon[mask] = np.degrees(lam_w)
    lat[mask] = np.degrees(phi_w)

    logger.debug("Converted %d bus coordinates from OSGB36 to WGS84 without pyproj.", mask.sum())
    return lon, lat


def harmonize_geographic_coordinates(network: pypsa.Network, logger: logging.Logger) -> str:
    """
    Ensure bus coordinate columns are suitable for geographic plotting.

    Returns a short status message describing any action taken.
    """
    buses = network.buses
    status = "Bus coordinate system left unchanged."

    lon_in_deg = 'lon' in buses and _series_within_bounds(buses['lon'], 180.0)
    lat_in_deg = 'lat' in buses and _series_within_bounds(buses['lat'], 90.0)
    x_in_deg = 'x' in buses and _series_within_bounds(buses['x'], 180.0)
    y_in_deg = 'y' in buses and _series_within_bounds(buses['y'], 90.0)

    # If lon/lat exist and look valid, propagate to x/y for plotting.
    if lon_in_deg and lat_in_deg and (not x_in_deg or not y_in_deg):
        buses['lon'] = pd.to_numeric(buses['lon'], errors='coerce')
        buses['lat'] = pd.to_numeric(buses['lat'], errors='coerce')
        buses['x'] = buses['lon']
        buses['y'] = buses['lat']
        logger.info("Copied bus lon/lat values into x/y for plotting.")
        status = "Copied lon/lat to x/y for geographic plotting."

    # If only x/y look valid, populate lon/lat for interactive explore.
    elif x_in_deg and y_in_deg and (not lon_in_deg or not lat_in_deg):
        buses['x'] = pd.to_numeric(buses['x'], errors='coerce')
        buses['y'] = pd.to_numeric(buses['y'], errors='coerce')
        buses['lon'] = buses['x']
        buses['lat'] = buses['y']
        logger.info("Copied bus x/y values into lon/lat for interactive plotting.")
        status = "Copied x/y to lon/lat for interactive plotting."

    # Attempt conversion from British National Grid (EPSG:27700) if degrees not detected.
    elif 'x' in buses and 'y' in buses:
        x_series = pd.to_numeric(buses['x'], errors='coerce').dropna()
        y_series = pd.to_numeric(buses['y'], errors='coerce').dropna()
        if not x_series.empty and not y_series.empty:
            likely_bng = (
                x_series.between(-1000, 800000).all() and
                y_series.between(-1000, 1400000).all()
            )
            if likely_bng:
                mask = (~buses['x'].isna()) & (~buses['y'].isna())
                if mask.any():
                    try:
                        from pyproj import Transformer  # Lazy import to avoid hard dependency at import time

                        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                        lon_vals, lat_vals = transformer.transform(
                            buses.loc[mask, 'x'].to_numpy(dtype=float),
                            buses.loc[mask, 'y'].to_numpy(dtype=float)
                        )
                        logger.info("Transformed bus coordinates from EPSG:27700 to WGS84 using pyproj.")
                    except ImportError:
                        lon_vals, lat_vals = _osgb36_to_wgs84(
                            buses.loc[mask, 'x'].to_numpy(dtype=float),
                            buses.loc[mask, 'y'].to_numpy(dtype=float),
                            logger,
                        )
                        logger.info("Transformed bus coordinates from EPSG:27700 to WGS84 using fallback converter.")
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("Failed to transform bus coordinates from EPSG:27700: %s", exc)
                        return f"ERROR: Failed to transform projected bus coordinates ({exc})."

                    buses.loc[mask, 'lon'] = lon_vals
                    buses.loc[mask, 'lat'] = lat_vals
                    # IMPORTANT: Do NOT overwrite x/y - PyPSA plotting needs them in projected coordinates (OSGB36)
                    # Only lon/lat should be in WGS84 for geographic operations
                    status = "Transformed bus coordinates from EPSG:27700 to WGS84 (lon/lat only, x/y kept in OSGB36)."

    # Ensure geometry column is aligned with lon/lat for PyPSA plotting utilities.
    lon_valid = 'lon' in buses and _series_within_bounds(buses['lon'], 180.0)
    lat_valid = 'lat' in buses and _series_within_bounds(buses['lat'], 90.0)
    if lon_valid and lat_valid:
        buses['lon'] = pd.to_numeric(buses['lon'], errors='coerce')
        buses['lat'] = pd.to_numeric(buses['lat'], errors='coerce')
        try:
            import geopandas as gpd  # noqa: F401
            # geopandas.points_from_xy is lazy-imported when available
            buses['geometry'] = gpd.points_from_xy(buses['lon'], buses['lat'])
            try:
                buses.attrs["crs"] = "EPSG:4326"
            except Exception:
                pass
        except Exception:  # pragma: no cover - optional dependency
            logger.debug("Geopandas unavailable - skipping geometry column refresh.", exc_info=True)

    return status


# Consistency check
try:
   
    # Capture consistency check output
    f = io.StringIO()
    
    # Suppress PyPSA warnings about zero x values (these are fixed by scripts but logged during check)
    pypsa_logger = logging.getLogger('pypsa.consistency')
    original_level = pypsa_logger.level
    pypsa_logger.setLevel(logging.ERROR)
    
    try:
        with contextlib.redirect_stdout(f):
            consistent = n.consistency_check()
    finally:
        pypsa_logger.setLevel(original_level)
    
    consistency_output = f.getvalue()
    
    if consistency_output.strip() == "":
        consistency_status = "[OK] Network passes all consistency checks"
        logger.info(f"Consistency check: {consistency_status}")
    else:
        consistency_status = "[WARNING] Network has consistency issues"
        logger.warning(f"Consistency check: {consistency_status}\nDetails:\n{consistency_output}")

except Exception as e:
    logger.warning(f"Failed to run consistency check: {e}")
    consistency_status = f"[ERROR] Consistency check failed: {e}"
    consistency_output = ""

# Harmonize coordinates for plotting
coordinate_status = harmonize_geographic_coordinates(n, logger)

# Gather network statistics
stats = {}
try:
    stats['buses'] = len(n.buses)
    stats['lines'] = len(n.lines)
    stats['generators'] = len(n.generators)
    stats['loads'] = len(n.loads)
    stats['links'] = len(n.links) if hasattr(n, 'links') else 0
    stats['storage'] = len(n.storage_units) if hasattr(n, 'storage_units') else 0
    stats['snapshots'] = len(n.snapshots)
    
    logger.info(f"Network stats: {stats}")
    
except Exception as e:
    logger.error(f"Failed to gather network stats: {e}")
    stats = {'error': str(e)}

# Generate plots (static + interactive) if dependencies are available
plot_dir = out_html.parent / "figures"
static_plot_path = None
interactive_plot_path = None
plot_messages = [coordinate_status]

# Static Matplotlib plot (using PyPSA n.plot())
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    static_plot_path = plot_dir / f"{net_path.stem}_topology.png"
    
    # Verify we have valid coordinates before plotting
    has_valid_coords = (
        'x' in n.buses.columns and n.buses['x'].notna().any() and
        'y' in n.buses.columns and n.buses['y'].notna().any()
    )
    
    if not has_valid_coords:
        logger.warning("Cannot create static plot: no valid bus coordinates found.")
        plot_messages.append("[SKIP] Static topology plot (no valid coordinates).")
        static_plot_path = None
    else:
        try:
            # Try to use cartopy projection for better geographic visualization
            import cartopy.crs as ccrs
            fig, ax = plt.subplots(figsize=(16, 14), subplot_kw={"projection": ccrs.PlateCarree()})
            ax.coastlines(resolution='10m', linewidths=0.5)
            ax.gridlines(draw_labels=False, alpha=0.3)
            
            # Use smaller bus sizes for clarity
            n.plot(ax=ax, geomap=False, bus_size=5, line_width=0.5, title=f"Network Topology: {net_path.stem}")
            
            # Set extent to GB approximate bounds for zoomed view
            ax.set_extent([-8.5, 2.5, 49.5, 60.5], crs=ccrs.PlateCarree())
            logger.info("Static plot created with cartopy geographic projection.")
        except ImportError:
            # Fallback: simple matplotlib without geographic projection
            logger.debug("Cartopy not available, using simple matplotlib plot.")
            fig, ax = plt.subplots(figsize=(14, 12))
            # Smaller bus and line sizes for cleaner plot
            n.plot(ax=ax, bus_size=5, line_width=0.5, title=f"Network Topology: {net_path.stem}")
        
        fig.tight_layout()
        fig.savefig(static_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Static network topology plot written to %s", static_plot_path)
        plot_messages.append("[OK] Static topology plot generated.")

except ImportError as exc:
    logger.warning("Matplotlib not available for plotting: %s", exc)
    plot_messages.append("[SKIP] Static topology plot (matplotlib missing).")
    static_plot_path = None
except Exception as exc:
    logger.error("Failed to create static topology plot: %s", exc)
    plot_messages.append(f"[ERROR] Static topology plot failed: {exc}")
    static_plot_path = None

# Interactive Pydeck plot via PyPSA n.explore()
try:
    has_valid_coords = (
        'lon' in n.buses.columns and n.buses['lon'].notna().any() and
        'lat' in n.buses.columns and n.buses['lat'].notna().any()
    )
    
    if not has_valid_coords:
        logger.warning("Cannot create interactive plot: no valid lon/lat coordinates.")
        plot_messages.append("[SKIP] Interactive explore map (no valid coordinates).")
    elif hasattr(n, "explore"):
        try:
            import pydeck  # noqa: F401

            interactive_plot_path = out_html.parent / f"{net_path.stem}_explore.html"
            
            # Use modern n.explore() API with visible bus markers and proper basemap
            deck = n.explore(
                bus_size=200,               # Large enough to be visible at zoom 5.5
                line_width=2,
                link_width=2,
                transformer_width=1,
                bus_alpha=0.9,
                line_alpha=0.7,
                link_alpha=0.7,
                transformer_alpha=0.6,
                tooltip=True,
                geomap=True,                # Use proper basemap for geographic context
                map_style="light",
            )
            
            deck.to_html(str(interactive_plot_path), notebook_display=False, open_browser=False)
            logger.info("Interactive explore map written to %s", interactive_plot_path)
            plot_messages.append("[OK] Interactive explore map generated.")
        except Exception as explore_e:
            logger.error("Interactive plot generation failed: %s", explore_e)
            plot_messages.append(f"[ERROR] Interactive explore map failed: {explore_e}")
            interactive_plot_path = None
    else:
        plot_messages.append("[SKIP] Interactive explore map (PyPSA explore API unavailable).")
except ImportError as exc:
    logger.warning("pydeck not available for interactive plotting: %s", exc)
    plot_messages.append("[SKIP] Interactive explore map (pydeck missing).")
except Exception as exc:
    logger.error("Unexpected error in interactive plotting: %s", exc)
    plot_messages.append(f"[ERROR] Interactive explore map failed: {exc}")
    interactive_plot_path = None


# Create HTML report
html_report = [
    "<html>",
    "<head><title>PyPSA Network Plot Report</title></head>",
    "<body>",
    "<h1>PyPSA Network Plot Report</h1>",
    f"<p><strong>Network file:</strong> {net_path}</p>",
    f"<p><strong>Load status:</strong> {load_status}</p>",
    f"<p><strong>Consistency check:</strong> {consistency_status}</p>",
    "",
    "<h2>Network Statistics</h2>",
    "<table border='1'>",
    "<tr><th>Component</th><th>Count</th></tr>",
]

# Add stats to table
for component, count in stats.items():
    html_report.append(f"<tr><td>{component.title()}</td><td>{count}</td></tr>")

html_report.extend([
    "</table>",
    "",
    "<h2>Plotting Summary</h2>",
    "<ul>",
])

for message in plot_messages:
    html_report.append(f"<li>{html.escape(message)}</li>")

html_report.extend([
    "</ul>",
])

if static_plot_path and static_plot_path.exists():
    try:
        static_rel = static_plot_path.relative_to(out_html.parent)
    except ValueError:
        static_rel = static_plot_path.name
    html_report.extend([
        "<h2>Static Network Map</h2>",
        f"<img src=\"{static_rel.as_posix()}\" alt=\"Network topology plot\" style=\"max-width:100%; height:auto;\"/>",
    ])

if interactive_plot_path and interactive_plot_path.exists():
    try:
        interactive_rel = interactive_plot_path.relative_to(out_html.parent)
    except ValueError:
        interactive_rel = interactive_plot_path.name
    html_report.extend([
        "<h2>Interactive Network Explorer</h2>",
        "<p>Explore buses, lines, and components interactively using the map below.</p>",
        f"<iframe src=\"{interactive_rel.as_posix()}\" title=\"Interactive network explorer\" "
        "style=\"width:100%; height:600px; border:1px solid #ccc;\"></iframe>",
    ])

html_report.extend([
    "<h2>Consistency Check Details</h2>",
    f"<pre>{html.escape(consistency_output)}</pre>",
    "",
    "</body>",
    "</html>"
])

# Write report
out_html.write_text("\n".join(html_report), encoding="utf-8")
logger.info(f"Network plot report written to {out_html}")

# Log execution summary
log_execution_summary(
    logger, "Network Plotting", start_time,
    inputs=[str(net_path)],
    outputs=[str(out_html)]
)

