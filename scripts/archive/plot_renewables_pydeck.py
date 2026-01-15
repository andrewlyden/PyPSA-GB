"""
Plot renewable generators from a PyPSA network using pydeck.

This script is intended to be called from a Snakemake rule. It reads the
first input (network file), extracts renewable generators, determines
coordinates (generator-level geo coords or bus coords fallback), and writes
an interactive pydeck html map to the Snakemake output path.

Dependencies: pypsa, pandas, pydeck
"""
import logging
from pathlib import Path
import sys

# Snakemake provides inputs/outputs when run under Snakemake
try:
    snakemake  # type: ignore
except NameError:
    raise RuntimeError("This script is intended to be run via Snakemake")

logger = logging.getLogger("plot_renewables_pydeck")
logging.basicConfig(level=logging.INFO)

network_path = Path(snakemake.input.network)
output_html = Path(snakemake.output[0])
# Read scenario from wildcard so the rule follows the active scenario at runtime
scenario = snakemake.wildcards.scenario if hasattr(snakemake, 'wildcards') and hasattr(snakemake.wildcards, 'scenario') else None
if not scenario:
    # Fallback to params if present (backwards compatibility)
    scenario = snakemake.params.scenario if hasattr(snakemake.params, 'scenario') else 'Historical_2020_clustered'

logger.info(f"Loading network from: {network_path}")

import pypsa
import pandas as pd
import pydeck as pdk

# Load network
n = pypsa.Network(network_path)

# Generator dataframe
g = n.generators.copy()
if g.empty:
    logger.warning("No generators found in network; writing empty map placeholder")
    # create a minimal empty html
    output_html.parent.mkdir(parents=True, exist_ok=True)
    html = "<html><body><h3>No generators found in network</h3></body></html>"
    output_html.write_text(html)
    sys.exit(0)

# Determine coordinate columns on generators
possible_lon = [c for c in ['geo_lon', 'lon', 'longitude', 'x'] if c in g.columns]
possible_lat = [c for c in ['geo_lat', 'lat', 'latitude', 'y'] if c in g.columns]

# We'll build a table with columns: name, carrier, p_nom, lon, lat
rows = []
for idx, row in g.iterrows():
    carrier = row.get('carrier', '')
    # Simple filter for renewable carriers
    if isinstance(carrier, str) and carrier.lower() not in ['wind', 'solar', 'onwind', 'offwind', 'wind onshore', 'wind offshore', 'wind_onshore', 'wind_offshore', 'solar', 'hydro', 'tidal', 'wave', 'shoreline_wave', 'tidal_stream', 'tidal_lagoon']:
        # Many projects use 'wind'/'solar' etc. We do a permissive check instead of strict exact match
        # allow carriers that contain known renewable keywords
        renewable_keywords = ['wind', 'solar', 'hydro', 'tidal', 'wave', 'shoreline', 'river']
        if not any(k in str(carrier).lower() for k in renewable_keywords):
            continue

    # Try to get generator-level lon/lat
    lon = None
    lat = None
    for c in possible_lon:
        if pd.notnull(row.get(c)):
            lon = row.get(c)
            break
    for c in possible_lat:
        if pd.notnull(row.get(c)):
            lat = row.get(c)
            break

    # Fallback: get bus coordinates
    if (lon is None or lat is None) and 'bus' in row.index and pd.notnull(row.get('bus')):
        bus = row['bus']
        if bus in n.buses.index:
            bus_row = n.buses.loc[bus]
            # Try common bus coord columns
            for c in ['x', 'lon', 'longitude', 'geo_lon']:
                if c in bus_row.index and pd.notnull(bus_row.get(c)):
                    lon = lon or bus_row.get(c)
                    break
            for c in ['y', 'lat', 'latitude', 'geo_lat']:
                if c in bus_row.index and pd.notnull(bus_row.get(c)):
                    lat = lat or bus_row.get(c)
                    break

    # If still missing, skip
    if lon is None or lat is None:
        continue

    rows.append({
        'name': idx,
        'carrier': row.get('carrier', ''),
        'p_nom_mw': row.get('p_nom', 0.0),
        'lon': float(lon),
        'lat': float(lat)
    })

if not rows:
    logger.warning("No renewable generators with coordinates found; writing placeholder HTML")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("<html><body><h3>No renewable generators with coordinates found in network.</h3></body></html>")
    sys.exit(0)

df = pd.DataFrame(rows)

# Prepare pydeck layer
layer = pdk.Layer(
    "ScatterplotLayer",
    df,
    pickable=True,
    opacity=0.8,
    stroked=True,
    filled=True,
    radius_scale=10,
    radius_min_pixels=3,
    radius_max_pixels=50,
    get_position=["lon", "lat"],
    get_radius="p_nom_mw",
    get_fill_color="[math.floor(255*(1 - (p_nom_mw / (p_nom_mw.max()+1)))), 120, math.floor(255*(p_nom_mw / (p_nom_mw.max()+1)))]",
    get_line_color=[0, 0, 0],
)

# Create deck, center on data mean
center_lon = df['lon'].mean()
center_lat = df['lat'].mean()

view_state = pdk.ViewState(
    longitude=float(center_lon),
    latitude=float(center_lat),
    zoom=6,
    pitch=30,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Name: {name}\nCarrier: {carrier}\nCapacity (MW): {p_nom_mw}"},
)

# Ensure output directory exists
output_html.parent.mkdir(parents=True, exist_ok=True)

# Write html
logger.info(f"Writing pydeck html to: {output_html}")
# pydeck uses math in style expressions; ensure math available in template by importing
import math

deck.to_html(str(output_html), notebook_display=False)
logger.info("Done")

