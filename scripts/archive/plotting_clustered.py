# plotting_clustered.py
"""
Enhanced plotting script for clustered PyPSA networks.

This script creates interactive Folium maps that show:
- Clustered network buses with aggregated loads
- Network connections (lines, transformers, links)
- GSP region polygons (when spatial clustering is used)
- Embedded demand time series plots
- Cluster boundaries and labels
"""

import numpy as np
import pandas as pd
import xarray as xr
import logging
from pathlib import Path
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import io
import base64

# Add logging import
from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info

def _norm(v):
    """bytes → str, ints stay ints (for dict keys)."""
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, bytes):
        try:
            return v.decode()
        except Exception:
            return str(v)
    return str(v)


def _center(lon, lat):
    mask = ~((lon == 0) & (lat == 0))
    return [float(lat[mask].median()), float(lon[mask].median())]


def create_clustered_pypsa_folium_map(
    nc_path,
    boundaries_path=None,
    scenario_config=None,
    *,
    map_tiles="CartoDB positron",
    initial_zoom=6,
    output_html=None,
    renewable_sites=None,
    logger=None,
):
    """
    Create an interactive Folium map for clustered PyPSA networks.
    
    Args:
        nc_path: Path to clustered network NetCDF file
        boundaries_path: Path to GeoJSON file with cluster boundaries (optional)
        scenario_config: Scenario configuration dictionary
        map_tiles: Folium map tiles to use
        initial_zoom: Initial map zoom level
        output_html: Output HTML file path
        renewable_sites: Dictionary of renewable site file paths (optional)
        logger: Logger instance
        
    Returns:
        folium.Map: The created map
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    nc_path = Path(nc_path)
    logger.info(f"Loading clustered PyPSA network from: {nc_path}")
    
    # Use PyPSA to read the network directly
    import pypsa
    network = pypsa.Network(nc_path)
    
    logger.info(f"Clustered network loaded: {len(network.buses)} buses, {len(network.loads)} loads")
    logger.info(f"Network components: {[comp for comp in ['lines', 'transformers', 'links'] if len(getattr(network, comp)) > 0]}")

    # ---------------- Load GSP boundaries if available ------------------
    gsp_boundaries = None
    if boundaries_path and Path(boundaries_path).exists():
        try:
            logger.info(f"Loading GSP boundaries from: {boundaries_path}")
            gsp_boundaries = gpd.read_file(boundaries_path)
            logger.info(f"Loaded {len(gsp_boundaries)} GSP regions")
            
            # Reproject to WGS84 for Folium if needed
            if gsp_boundaries.crs and gsp_boundaries.crs.to_string() != 'EPSG:4326':
                gsp_boundaries = gsp_boundaries.to_crs('EPSG:4326')
                logger.info("Reprojected GSP boundaries to WGS84")
                
        except Exception as e:
            logger.warning(f"Could not load GSP boundaries: {e}")
            gsp_boundaries = None

    # ---------------- Buses --------------------------------------------------
    logger.info("Processing clustered buses data")
    buses = pd.DataFrame({
        "bus_id": network.buses.index.tolist(),
        "lon": network.buses.x.values.astype(float),
        "lat": network.buses.y.values.astype(float),
    })
    buses = buses[~((buses.lon == 0) & (buses.lat == 0))]
    logger.info(f"Found {len(buses)} clustered buses with valid coordinates")
    coord = {bid: (row.lat, row.lon) for bid, row in buses.set_index("bus_id").iterrows()}

    # ---------------- Loads --------------------------------------------------
    logger.info("Processing aggregated loads data")
    loads_timeseries = None
    try:
        if len(network.loads) == 0:
            logger.warning("No loads found in clustered network")
            load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
            load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])
        else:
            logger.info(f"Found {len(network.loads)} aggregated loads in clustered network")
            
            # Get load data
            load_buses = network.loads.bus
            logger.info(f"Loads assigned to {len(load_buses.unique())} unique clustered buses")
            
            if hasattr(network.loads_t, 'p_set') and not network.loads_t.p_set.empty:
                # Get timeseries data (MW)
                loads_timeseries = network.loads_t.p_set
                logger.info(f"Loads timeseries shape: {loads_timeseries.shape}")
                
                # Check if we have actual timeseries data or just zeros
                total_load_sum = loads_timeseries.sum().sum()
                logger.info(f"Total clustered load sum: {total_load_sum:.3f} MW")
                
                if total_load_sum > 0:
                    # Estimate timestep hours
                    num_snapshots = len(loads_timeseries.index)
                    if num_snapshots > 1:
                        timestep_hours = 0.5
                        logger.info(f"Using {timestep_hours} hour timesteps for {num_snapshots} snapshots")
                    else:
                        timestep_hours = 8760.0
                        logger.warning("Single snapshot - estimating annual energy")
                    
                    # Calculate per-load metrics
                    load_annual_mwh = loads_timeseries.sum(axis=0) * timestep_hours
                    load_peak_mw = loads_timeseries.max(axis=0)
                    
                    # Aggregate by bus (clustered buses may have multiple loads)
                    load_per_bus_mwh = pd.Series(index=buses["bus_id"], dtype=float)
                    load_peak_per_bus = pd.Series(index=buses["bus_id"], dtype=float)
                    
                    for bus_id in buses["bus_id"]:
                        loads_on_bus = load_buses[load_buses == bus_id].index
                        if len(loads_on_bus) > 0:
                            bus_annual_mwh = load_annual_mwh.reindex(loads_on_bus).fillna(0).sum()
                            bus_peak_mw = load_peak_mw.reindex(loads_on_bus).fillna(0).sum()
                            load_per_bus_mwh[bus_id] = bus_annual_mwh
                            load_peak_per_bus[bus_id] = bus_peak_mw
                        else:
                            load_per_bus_mwh[bus_id] = 0.0
                            load_peak_per_bus[bus_id] = 0.0
                    
                    load_per_bus_mwh = load_per_bus_mwh.fillna(0.0)
                    load_peak_per_bus = load_peak_per_bus.fillna(0.0)
                    
                    # Log totals
                    total_annual_twh = load_per_bus_mwh.sum() / 1e6
                    logger.info(f"Total clustered annual load: {total_annual_twh:.3f} TWh")
                else:
                    load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
                    load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])
            else:
                logger.warning("No load timeseries data found in clustered network")
                load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
                load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])
                
    except Exception as e:
        logger.error(f"Error processing clustered loads data: {e}")
        load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
        load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])

    # ---------------- Create Folium Map --------------------------------------------
    logger.info("Creating Folium map for clustered network")
    map_center = _center(buses.lon, buses.lat)
    logger.info(f"Map centered at: {map_center}")
    m = folium.Map(location=map_center, tiles=map_tiles, zoom_start=initial_zoom)

    # ---------------- Add GSP Region Polygons First (Background Layer) -----------
    if gsp_boundaries is not None:
        logger.info("Adding GSP region polygons to map")
        
        # Create a feature group for GSP regions
        gsp_group = folium.FeatureGroup(name="GSP Regions", show=True)
        
        # Add each GSP region as a polygon
        for idx, region in gsp_boundaries.iterrows():
            # Get region properties
            gsp_name = region.get('GSPs', f'Region_{idx}')
            gsp_group_name = region.get('GSPGroup', 'Unknown')
            
            # Create popup text
            popup_text = f"<b>{gsp_name}</b><br>Group: {gsp_group_name}"
            
            # Add polygon with styling
            folium.GeoJson(
                region.geometry.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': 'lightblue',
                    'color': 'blue',
                    'weight': 1,
                    'fillOpacity': 0.2,
                    'opacity': 0.7
                },
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"GSP: {gsp_name}"
            ).add_to(gsp_group)
        
        gsp_group.add_to(m)
        logger.info(f"Added {len(gsp_boundaries)} GSP region polygons")

    # ---------------- Add Clustered Buses (Overlay Layer) ------------------------
    logger.info("Adding clustered buses to map")
    
    # Create a feature group for buses
    bus_group = folium.FeatureGroup(name="Clustered Buses", show=True)
    
    # Group buses by rounded lat/lon (for nearby clusters)
    buses["lat_rounded"] = buses["lat"].round(4)
    buses["lon_rounded"] = buses["lon"].round(4)
    grouped = buses.groupby(["lat_rounded", "lon_rounded"])
    logger.info(f"Grouped {len(buses)} clustered buses into {len(grouped)} location groups")

    for (lat, lon), group in grouped:
        bus_ids = group["bus_id"].tolist()
        # Annual MWh for this cluster group
        cluster_mwh = load_per_bus_mwh.reindex(bus_ids).fillna(0).sum()
        cluster_gwh = cluster_mwh / 1e3
        cluster_peak_mw = load_peak_per_bus.reindex(bus_ids).fillna(0).max()
        cluster_peak_gw = float(cluster_peak_mw) / 1e3

        # Create label with cluster names
        if len(bus_ids) == 1:
            label = bus_ids[0]
        else:
            label = f"{len(bus_ids)} clusters: " + ", ".join(bus_ids[:3])
            if len(bus_ids) > 3:
                label += f", ... (+{len(bus_ids)-3} more)"
        
        popup = f"<b>{label}</b><br>Annual: {cluster_gwh:.1f} GWh<br>Peak: {cluster_peak_gw:.2f} GW"

        # Larger radius for clustered buses to show aggregation
        # Scale with load but make more prominent than base network
        radius = 8 + min(np.sqrt(float(cluster_mwh)) * 0.008, 120)
        
        # Use different color for clustered buses
        folium.CircleMarker(
            [lat, lon],
            radius=radius,
            color="red",
            fill=True,
            fillColor="orange",
            fill_opacity=0.7,
            popup=popup,
            tooltip=f"Cluster: {cluster_gwh:.1f} GWh"
        ).add_to(bus_group)

    bus_group.add_to(m)

    # ---------------- Add Network Connections ------------------------------------
    def _draw_connections(bus0, bus1, color, weight, label, group):
        """Draw network connections on the map."""
        count = 0
        for b0, b1 in zip(bus0, bus1):
            b0, b1 = _norm(b0), _norm(b1)
            if b0 in coord and b1 in coord:
                lat0, lon0 = coord[b0]
                lat1, lon1 = coord[b1]
                folium.PolyLine(
                    [[lat0, lon0], [lat1, lon1]],
                    color=color, 
                    weight=weight, 
                    opacity=0.6,
                    tooltip=label
                ).add_to(group)
                count += 1
        return count

    logger.info("Adding clustered network connections")
    
    # Create feature group for connections
    connections_group = folium.FeatureGroup(name="Network Connections", show=True)
    
    # Handle lines (if present)
    if len(network.lines) > 0:
        lines_count = _draw_connections(
            network.lines.bus0.values, 
            network.lines.bus1.values, 
            "gray", 2, "Clustered Line", connections_group
        )
        logger.info(f"Added {lines_count} clustered lines")

    # Handle transformers (if present) 
    if len(network.transformers) > 0:
        transformers_count = _draw_connections(
            network.transformers.bus0.values,
            network.transformers.bus1.values, 
            "orange", 3, "Clustered Transformer", connections_group
        )
        logger.info(f"Added {transformers_count} clustered transformers")

    # Handle links (if present)
    if len(network.links) > 0:
        links_count = _draw_connections(
            network.links.bus0.values,
            network.links.bus1.values, 
            "purple", 2, "Clustered Link", connections_group
        )
        logger.info(f"Added {links_count} clustered links")

    connections_group.add_to(m)

    # Add renewable sites as toggleable layers if provided
    if renewable_sites:
        logger.info("Adding renewable sites as toggleable layers")
        _add_renewable_sites_layers(m, renewable_sites, logger)

    # ---------------- Add Layer Control AFTER all layers are added ----------
    folium.LayerControl().add_to(m)

    # ---------------- Add Embedded Demand Plot ------------------------------
    # Move plot down to avoid overlap with layer control
    additional_top_px = 120
    top_base_px = 10 + int(additional_top_px)

    try:
        if loads_timeseries is not None and not loads_timeseries.empty:
            # Sum across loads to get total clustered system demand (MW)
            system_demand = loads_timeseries.sum(axis=1)

            if len(system_demand) <= 1:
                # Single snapshot
                snapshot_value = system_demand.iloc[0]
                snapshot_gw = float(snapshot_value) / 1e3
                demand_html = (
                    f"<div style='position:absolute; left:10px; top:{top_base_px}px; width:300px; "
                    f"background:white; padding:8px; border:1px solid #ccc; z-index:9999;'>"
                    f"<b>Clustered Network</b><br>Total demand: {snapshot_gw:.2f} GW</div>"
                )
            else:
                # Time series - try Plotly first, fallback to matplotlib
                try:
                    import plotly.graph_objects as go
                    import plotly.offline as pyo

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=system_demand.index.astype(str), 
                        y=system_demand.values / 1e3,
                        mode='lines', 
                        name='Clustered Demand', 
                        line=dict(color='orangered', width=2)
                    ))
                    fig.update_layout(
                        margin=dict(l=20, r=8, t=30, b=20), 
                        height=240,
                        xaxis_title='Time', 
                        yaxis_title='GW', 
                        title='Clustered Network Demand'
                    )

                    # Save demand plot
                    if output_html:
                        out_dir = Path(output_html).parent
                        demand_file = out_dir / (Path(output_html).stem + "_demand.html")
                        plot_div = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
                        html_page = (
                            "<!doctype html><html><head><meta charset='utf-8'>"
                            "<title>Clustered Network Demand</title></head><body>"
                            + plot_div + "</body></html>"
                        )
                        demand_file.write_text(html_page, encoding='utf-8')
                        demand_html = (
                            f"<div style='position:absolute; left:10px; top:{top_base_px}px; "
                            f"width:540px; z-index:9999;'>"
                            f"<iframe src='{demand_file.name}' style='border:1px solid #ccc; "
                            f"width:540px; height:280px;'></iframe></div>"
                        )
                    else:
                        plot_div = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
                        demand_html = (
                            f"<div style='position:absolute; left:10px; top:{top_base_px}px; "
                            f"width:540px; z-index:9999;'>{plot_div}</div>"
                        )
                        
                except Exception:
                    logger.warning("Plotly not available, using matplotlib for demand plot")
                    # Matplotlib fallback
                    fig, ax = plt.subplots(figsize=(6, 2.4))
                    ax.plot(system_demand.index, system_demand.values / 1e3, 
                           color='orangered', linewidth=2, label='Clustered')
                    ax.set_ylabel('GW')
                    ax.set_title('Clustered Network Demand')
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=90)
                    plt.close(fig)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('ascii')
                    demand_html = (
                        f"<div style='position:absolute; left:10px; top:{top_base_px}px; "
                        f"width:440px; background:white; padding:6px; border:1px solid #ccc; "
                        f"z-index:9999;'>"
                        f"<img src='data:image/png;base64,{img_b64}' style='width:100%'></div>"
                    )

            # Add demand plot to map
            from branca.element import Element
            m.get_root().html.add_child(Element(demand_html))

    except Exception as e:
        logger.warning(f"Could not create clustered demand plot: {e}")

    # ---------------- Add Network Metadata Box ------------------------------
    try:
        # Add clustering info box
        clustering_method = scenario_config.get('clustering', {}).get('method', 'Unknown') if scenario_config else 'Unknown'
        metadata_html = (
            f"<div style='position:absolute; right:10px; top:10px; width:280px; "
            f"background:rgba(255,255,255,0.9); padding:8px; border:1px solid #ccc; "
            f"border-radius:4px; z-index:9999; font-size:12px;'>"
            f"<b>Network Clustering Info</b><br>"
            f"Method: {clustering_method.title()}<br>"
            f"Clustered Buses: {len(network.buses)}<br>"
            f"Connections: {len(network.lines)} lines"
        )
        
        if len(network.transformers) > 0:
            metadata_html += f", {len(network.transformers)} transformers"
        if len(network.links) > 0:
            metadata_html += f", {len(network.links)} links"
            
        if gsp_boundaries is not None:
            metadata_html += f"<br>GSP Regions: {len(gsp_boundaries)}"
            
        metadata_html += "</div>"
        
        from branca.element import Element
        m.get_root().html.add_child(Element(metadata_html))
        
    except Exception as e:
        logger.warning(f"Could not add metadata box: {e}")

    # ---------------- Save Map -----------------------------------------------
    if output_html:
        logger.info(f"Saving clustered network map to: {output_html}")
        Path(output_html).write_text(m.get_root().render(), encoding="utf-8")
        logger.info(f"Clustered map saved successfully: {output_html}")
    else:
        logger.info("No output path specified - clustered map created in memory only")

    return m


def _add_renewable_sites_layers(map_obj, renewable_sites, logger):
    """Add renewable sites as toggleable layers to the Folium map."""
    
    # Define colors and icons for each renewable type
    renewable_config = {
        'wind_onshore': {
            'color': 'green',
            'icon': 'fa-wind',
            'layer_name': 'Wind Onshore Sites',
            'popup_prefix': 'Onshore Wind'
        },
        'wind_offshore': {
            'color': 'blue', 
            'icon': 'fa-water',
            'layer_name': 'Wind Offshore Sites',
            'popup_prefix': 'Offshore Wind'
        },
        'solar_pv': {
            'color': 'orange',
            'icon': 'fa-sun',
            'layer_name': 'Solar PV Sites',
            'popup_prefix': 'Solar PV'
        },
        'geothermal': {
            'color': 'red',
            'icon': 'fa-fire',
            'layer_name': 'Geothermal Sites',
            'popup_prefix': 'Geothermal'
        },
        'small_hydro': {
            'color': 'lightblue',
            'icon': 'fa-tint',
            'layer_name': 'Small Hydro Sites', 
            'popup_prefix': 'Small Hydro'
        },
        'large_hydro': {
            'color': 'darkblue',
            'icon': 'fa-tint',
            'layer_name': 'Large Hydro Sites',
            'popup_prefix': 'Large Hydro'
        },
        'tidal_stream': {
            'color': 'purple',
            'icon': 'fa-water',
            'layer_name': 'Tidal Stream Sites',
            'popup_prefix': 'Tidal Stream'
        },
        'shoreline_wave': {
            'color': 'cadetblue',
            'icon': 'fa-water',
            'layer_name': 'Shoreline Wave Sites',
            'popup_prefix': 'Shoreline Wave'
        },
        'tidal_lagoon': {
            'color': 'darkslateblue',
            'icon': 'fa-water',
            'layer_name': 'Tidal Lagoon Sites',
            'popup_prefix': 'Tidal Lagoon'
        }
    }
    
    for tech_type, site_file in renewable_sites.items():
        if not Path(site_file).exists():
            logger.warning(f"Renewable sites file not found: {site_file}")
            continue
            
        config = renewable_config.get(tech_type, {
            'color': 'gray', 'icon': 'fa-bolt', 
            'layer_name': f'{tech_type.title()} Sites',
            'popup_prefix': tech_type.replace('_', ' ').title()
        })
        
        try:
            # Load renewable sites data
            sites_df = pd.read_csv(site_file)
            logger.info(f"Loading {len(sites_df)} {tech_type} sites")
            
            # Create a feature group for this technology type
            feature_group = folium.FeatureGroup(name=config['layer_name'], show=True)
            
            # Add each site as a marker
            sites_added = 0
            for idx, site in sites_df.iterrows():
                try:
                    lat = site.get('lat')
                    lon = site.get('lon')
                    
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                        
                    # Convert to float to ensure proper handling
                    lat = float(lat)
                    lon = float(lon)
                    
                    # Skip invalid coordinates
                    if lat == 0 and lon == 0:
                        continue
                    if lat < 49 or lat > 61 or lon < -8 or lon > 2:  # UK bounds check
                        continue
                        
                    # Create popup content
                    popup_content = f"""
                    <b>{config['popup_prefix']} Site</b><br>
                    <b>Name:</b> {site.get('site_name', 'N/A')}<br>
                    <b>Capacity:</b> {site.get('capacity_mw', 'N/A')} MW<br>
                    <b>Technology:</b> {site.get('technology', tech_type)}<br>
                    <b>Coordinates:</b> {lat:.4f}, {lon:.4f}
                    """
                    
                    # Determine marker size based on capacity
                    capacity = site.get('capacity_mw', 1)
                    if pd.isna(capacity):
                        capacity = 1
                    else:
                        capacity = float(capacity)
                    marker_size = max(6, min(20, int(capacity / 5) + 6))
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=marker_size,
                        popup=folium.Popup(popup_content, max_width=300),
                        color=config['color'],
                        fill=True,
                        fillColor=config['color'],
                        fillOpacity=0.8,
                        weight=2,
                        tooltip=f"{config['popup_prefix']}: {site.get('site_name', 'Unknown')} ({capacity:.1f} MW)"
                    ).add_to(feature_group)
                    
                    sites_added += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing site {idx}: {e}")
                    continue
            
            # Add the feature group to the map
            feature_group.add_to(map_obj)
            logger.info(f"Added {sites_added} {tech_type} sites to map layer '{config['layer_name']}'")
            
        except Exception as e:
            logger.error(f"Error loading {tech_type} sites from {site_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Set up logging with Snakemake log path
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "plotting_clustered"
    logger = setup_logging(log_path)
    
    # Get Snakemake inputs and parameters
    network_path = snakemake.input.network
    boundaries_path = snakemake.input.boundaries if hasattr(snakemake.input, 'boundaries') else None
    scenario_config = snakemake.params.scenario_config if hasattr(snakemake.params, 'scenario_config') else {}
    out_html = snakemake.output[0]
    
    # Collect renewable sites if provided
    renewable_sites = {}
    if hasattr(snakemake.input, 'wind_onshore_sites'):
        renewable_sites['wind_onshore'] = snakemake.input.wind_onshore_sites
    if hasattr(snakemake.input, 'wind_offshore_sites'):
        renewable_sites['wind_offshore'] = snakemake.input.wind_offshore_sites
    if hasattr(snakemake.input, 'solar_pv_sites'):
        renewable_sites['solar_pv'] = snakemake.input.solar_pv_sites
    if hasattr(snakemake.input, 'geothermal_sites'):
        renewable_sites['geothermal'] = snakemake.input.geothermal_sites
    if hasattr(snakemake.input, 'small_hydro_sites'):
        renewable_sites['small_hydro'] = snakemake.input.small_hydro_sites
    if hasattr(snakemake.input, 'large_hydro_sites'):
        renewable_sites['large_hydro'] = snakemake.input.large_hydro_sites
    if hasattr(snakemake.input, 'tidal_stream_sites'):
        renewable_sites['tidal_stream'] = snakemake.input.tidal_stream_sites
    if hasattr(snakemake.input, 'shoreline_wave_sites'):
        renewable_sites['shoreline_wave'] = snakemake.input.shoreline_wave_sites
    if hasattr(snakemake.input, 'tidal_lagoon_sites'):
        renewable_sites['tidal_lagoon'] = snakemake.input.tidal_lagoon_sites
    
    logger.info("="*60)
    logger.info("STARTING CLUSTERED NETWORK MAP GENERATION")
    logger.info(f"Network: {network_path}")
    logger.info(f"Boundaries: {boundaries_path}")
    logger.info(f"Output: {out_html}")
    if renewable_sites:
        logger.info(f"Including renewable sites: {list(renewable_sites.keys())}")
    
    create_clustered_pypsa_folium_map(
        network_path, 
        boundaries_path=boundaries_path,
        scenario_config=scenario_config,
        output_html=out_html,
        renewable_sites=renewable_sites,
        logger=logger
    )
    
    logger.info(f"CLUSTERED MAP GENERATION COMPLETED - open {out_html} in your browser")
    logger.info("="*60)

