# folium_pypsa_map_full.py
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


def create_pypsa_folium_map(
    nc_path,
    *,
    map_tiles="CartoDB positron",
    initial_zoom=6,
    output_html=None,
    renewable_sites=None,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    nc_path = Path(nc_path)
    logger.info(f"Loading PyPSA network from: {nc_path}")
    
    # Use PyPSA to read the network directly
    import pypsa
    network = pypsa.Network(nc_path)
    
    logger.info(f"Network loaded: {len(network.buses)} buses, {len(network.loads)} loads")

    # ---------------- Buses --------------------------------------------------
    logger.info("Processing buses data")
    buses = pd.DataFrame({
        "bus_id": network.buses.index.tolist(),
        "lon": network.buses.x.values.astype(float),
        "lat": network.buses.y.values.astype(float),
    })
    buses = buses[~((buses.lon == 0) & (buses.lat == 0))]
    logger.info(f"Found {len(buses)} buses with valid coordinates")
    coord = {bid: (row.lat, row.lon) for bid, row in buses.set_index("bus_id").iterrows()}

    # ---------------- Loads --------------------------------------------------
    logger.info("Processing loads data")
    loads_timeseries = None
    try:
        if len(network.loads) == 0:
            logger.warning("No loads found in network")
            load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
            load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])
        else:
            logger.info(f"Found {len(network.loads)} loads in network")
            
            # Get load data - network.loads has the static data (bus assignment)
            # network.loads_t.p_set has the timeseries data
            load_buses = network.loads.bus
            logger.info(f"Loads assigned to {len(load_buses.unique())} unique buses")
            
            if hasattr(network.loads_t, 'p_set') and not network.loads_t.p_set.empty:
                # Get timeseries data (MW)
                loads_timeseries = network.loads_t.p_set
                logger.info(f"Loads timeseries shape: {loads_timeseries.shape}")
                logger.info(f"Timeseries index: {len(loads_timeseries.index)} snapshots")
                
                # Check if we have actual timeseries data or just zeros
                total_load_sum = loads_timeseries.sum().sum()
                logger.info(f"Total load sum across all snapshots: {total_load_sum:.3f} MW")
                
                if total_load_sum > 0:
                    # Estimate timestep hours (assume half-hourly if we have multiple snapshots)
                    num_snapshots = len(loads_timeseries.index)
                    if num_snapshots > 1:
                        # Assume half-hourly timesteps
                        timestep_hours = 0.5
                        logger.info(f"Using {timestep_hours} hour timesteps for {num_snapshots} snapshots")
                    else:
                        # Single snapshot - estimate as annual average
                        timestep_hours = 8760.0
                        logger.warning("Single snapshot found - estimating annual energy assuming this is average MW")
                    
                    # Calculate per-load metrics
                    load_annual_mwh = loads_timeseries.sum(axis=0) * timestep_hours
                    load_peak_mw = loads_timeseries.max(axis=0)
                    
                    # Aggregate by bus (sum all loads on each bus)
                    load_per_bus_mwh = pd.Series(index=buses["bus_id"], dtype=float)
                    load_peak_per_bus = pd.Series(index=buses["bus_id"], dtype=float)
                    
                    for bus_id in buses["bus_id"]:
                        # Find loads on this bus
                        loads_on_bus = load_buses[load_buses == bus_id].index
                        if len(loads_on_bus) > 0:
                            bus_annual_mwh = load_annual_mwh.reindex(loads_on_bus).fillna(0).sum()
                            bus_peak_mw = load_peak_mw.reindex(loads_on_bus).fillna(0).sum()
                            load_per_bus_mwh[bus_id] = bus_annual_mwh
                            load_peak_per_bus[bus_id] = bus_peak_mw
                        else:
                            load_per_bus_mwh[bus_id] = 0.0
                            load_peak_per_bus[bus_id] = 0.0
                    
                    # Fill NaNs with zeros
                    load_per_bus_mwh = load_per_bus_mwh.fillna(0.0)
                    load_peak_per_bus = load_peak_per_bus.fillna(0.0)
                    
                    # Log totals
                    total_annual_mwh = load_per_bus_mwh.sum()
                    total_annual_gwh = total_annual_mwh / 1e3
                    total_annual_twh = total_annual_gwh / 1e3
                    logger.info(
                        f"Processed loads for {len(load_per_bus_mwh)} buses; "
                        f"Total annual load = {total_annual_mwh:.0f} MWh = {total_annual_gwh:.3f} GWh = {total_annual_twh:.6f} TWh"
                    )
                    # keep full loads_timeseries for plotting whole-network demand
                    loads_timeseries = loads_timeseries
                else:
                    logger.warning("All loads are zero - checking common PyPSA timeseries attributes on network.loads_t")
                    try:
                        alt_attrs = ('p_set', 'p', 'p_unscaled', 'p_expected', 'p0')
                        found_attr = None
                        for attr in alt_attrs:
                            if hasattr(network.loads_t, attr):
                                try:
                                    cand = getattr(network.loads_t, attr)
                                    # convert xarray DataArray to pandas if possible
                                    if hasattr(cand, 'to_pandas'):
                                        df_cand = cand.to_pandas()
                                    else:
                                        df_cand = cand
                                    # normalize Series -> DataFrame
                                    if isinstance(df_cand, pd.Series):
                                        df_cand = df_cand.to_frame()
                                    if isinstance(df_cand, pd.DataFrame) and df_cand.size > 0:
                                        total = df_cand.sum().sum()
                                        logger.info(f"network.loads_t.{attr} total = {total:.6f}")
                                        if total > 0:
                                            loads_timeseries = df_cand
                                            found_attr = attr
                                            logger.info(f"Using network.loads_t.{attr} as loads_timeseries")
                                            break
                                except Exception:
                                    continue

                        if found_attr is None:
                            logger.warning("No non-zero attribute found on network.loads_t; using zero loads")
                            load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
                            load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])
                        else:
                            # compute annual/peak from discovered loads_timeseries
                            num_snapshots = len(loads_timeseries.index)
                            timestep_hours = 0.5 if num_snapshots > 1 else 8760.0
                            load_annual_mwh = loads_timeseries.sum(axis=0) * timestep_hours
                            load_peak_mw = loads_timeseries.max(axis=0)
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
                    except Exception as e:
                        logger.exception(f"Attribute discovery failed: {e}")
                        load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
                        load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])
            else:
                logger.warning("No load timeseries data found")
                load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
                load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])
                
    except Exception as e:
        logger.error(f"Error processing loads data: {e}")
        load_per_bus_mwh = pd.Series(0.0, index=buses["bus_id"])
        load_peak_per_bus = pd.Series(0.0, index=buses["bus_id"])

    # ---------------- Folium Map --------------------------------------------
    logger.info("Creating Folium map")
    map_center = _center(buses.lon, buses.lat)
    logger.info(f"Map centered at: {map_center}")
    m = folium.Map(location=map_center,
                   tiles=map_tiles, zoom_start=initial_zoom)

    # Move the embedded demand plot down by ~1 inch (CSS pixels).
    # 1 inch ≈ 96 CSS pixels on most displays. Adjust this value if needed.
    additional_top_px = 96
    top_base_px = 10 + int(additional_top_px)

    # Group buses by rounded lat/lon
    buses["lat_rounded"] = buses["lat"].round(5)
    buses["lon_rounded"] = buses["lon"].round(5)
    grouped = buses.groupby(["lat_rounded", "lon_rounded"])
    logger.info(f"Grouped buses into {len(grouped)} location clusters")

    for (lat, lon), group in grouped:
        bus_ids = group["bus_id"].tolist()
        # annual MWh for this cluster and peak MW (max across buses in cluster)
        cluster_mwh = load_per_bus_mwh.reindex(bus_ids).fillna(0).sum()
        cluster_gwh = cluster_mwh / 1e3
        cluster_peak_mw = load_peak_per_bus.reindex(bus_ids).fillna(0).max()
        # Convert peak from MW to GW for display
        cluster_peak_gw = float(cluster_peak_mw) / 1e3

        label = "<br>".join(bus_ids)
        popup = f"{label}<br>Annual: {cluster_gwh:.3f} GWh<br>Peak: {cluster_peak_gw:.3f} GW"

        # Radius proportional to cluster annual MWh (sqrt scaling, reduced multiplier for better proportionality)
        # Reduced further by factor 5 per request
        radius = 4 + min(np.sqrt(float(cluster_mwh)) * 0.0032, 80)

        folium.CircleMarker(
            [lat, lon],
            radius=radius,
            color="blue",
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(m)

    def _draw(bus0, bus1, col, w=1, lab=""):
        for b0, b1 in zip(bus0, bus1):
            b0, b1 = _norm(b0), _norm(b1)
            if b0 in coord and b1 in coord:
                lat0, lon0 = coord[b0]
                lat1, lon1 = coord[b1]
                folium.PolyLine([[lat0, lon0], [lat1, lon1]],
                                color=col, weight=w, opacity=0.5,
                                tooltip=lab).add_to(m)

    logger.info("Adding network components to map")
    
    # Handle lines (if present)
    if len(network.lines) > 0:
        lines_count = len(network.lines)
        _draw(network.lines.bus0.values, network.lines.bus1.values, "gray", 1, "Line")
        logger.info(f"Added {lines_count} lines")
    else:
        logger.info("No lines found in network")

    # Handle transformers (if present)
    if len(network.transformers) > 0:
        transformers_count = len(network.transformers)
        _draw(network.transformers.bus0.values,
              network.transformers.bus1.values, "orange", 2, "Transformer")
        logger.info(f"Added {transformers_count} transformers")
    else:
        logger.info("No transformers found in network")

    # Handle links (if present)
    if len(network.links) > 0:
        links_count = len(network.links)
        _draw(network.links.bus0.values,
              network.links.bus1.values, "purple", 1, "Link")
        logger.info(f"Added {links_count} links")
    else:
        logger.info("No links found in network")

    # Add renewable sites as toggleable layers if provided
    if renewable_sites:
        logger.info("Adding renewable sites as toggleable layers")
        _add_renewable_sites_layers(m, renewable_sites, logger)

    # Add generators from the network as toggleable layers
    logger.info("Adding generators from network as toggleable layers")
    _add_generators_layers(m, network, coord, logger)

    # Add layer control AFTER all layers have been added
    folium.LayerControl().add_to(m)

    # If we have full loads timeseries, create a small inline plot of total network demand
    try:
        if loads_timeseries is not None and not loads_timeseries.empty:
            # Sum across loads to get system demand (MW)
            system_demand = loads_timeseries.sum(axis=1)

            # If single snapshot, create a simple text box instead (show GW)
            if len(system_demand) <= 1:
                snapshot_value = system_demand.iloc[0]
                snapshot_gw = float(snapshot_value) / 1e3
                demand_html = (
                    "<div style='position:absolute; left:10px; top:" + str(top_base_px) + "px; width:300px; background:white; padding:8px; border:1px solid #ccc; z-index:9999;'>"
                    + "Total demand snapshot: " + str(round(snapshot_gw, 3)) + " GW</div>"
                )
            else:
                # Try to create an interactive Plotly chart inline. If Plotly is not
                # available, fall back to the PNG approach used previously.
                try:
                    import plotly.graph_objects as go
                    import plotly.offline as pyo

                    fig = go.Figure()
                    # Convert demand to GW for plotting
                    fig.add_trace(go.Scatter(x=system_demand.index.astype(str), y=system_demand.values / 1e3,
                                             mode='lines', name='Demand', line=dict(color='royalblue')))
                    fig.update_layout(margin=dict(l=20, r=8, t=24, b=20), height=220,
                                      xaxis_title='', yaxis_title='GW', title='Total network demand')

                    # Build a small standalone HTML file for the interactive plot
                    # and embed it via an iframe to avoid template parsing issues.
                    try:
                        # Save the demand HTML next to the output map so the iframe can load it by relative path
                        if output_html:
                            out_dir = Path(output_html).parent
                            demand_file = out_dir / (Path(output_html).stem + "_demand.html")
                            plot_div = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
                            html_page = "<!doctype html><html><head><meta charset='utf-8'><title>Network demand</title></head><body>"
                            html_page += plot_div
                            html_page += "</body></html>"
                            demand_file.write_text(html_page, encoding='utf-8')
                            demand_html = (
                                "<div style='position:absolute; left:10px; top:" + str(top_base_px) + "px; width:520px; z-index:9999;'>"
                                "<iframe src='" + demand_file.name + "' style='border:none; width:520px; height:260px;'></iframe>"
                                "</div>"
                            )
                        else:
                            # No output path available; embed the plot div directly (may include raw braces)
                            plot_div = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
                            demand_html = ("<div style='position:absolute; left:10px; top:" + str(top_base_px) + "px; width:520px; z-index:9999;'>" + plot_div + "</div>")
                    except Exception:
                        # If writing the file fails, fall back to inline div
                        plot_div = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
                        demand_html = ("<div style='position:absolute; left:10px; top:" + str(top_base_px) + "px; width:520px; z-index:9999;'>" + plot_div + "</div>")
                except Exception as exc:
                    import traceback
                    logger.exception("Plotly inline embed failed: %s", exc)
                    # Fallback to PNG if Plotly is not installed or embedding failed
                    fig, ax = plt.subplots(figsize=(6,2.2))
                    # Plot in GW for consistency
                    ax.plot(system_demand.index, system_demand.values / 1e3, color='tab:blue')
                    ax.set_ylabel('GW')
                    ax.set_title('Total network demand')
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=90)
                    plt.close(fig)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('ascii')
                    demand_html = (
                        "<div style='position:absolute; left:10px; top:" + str(top_base_px) + "px; width:420px; background:white; padding:6px; border:1px solid #ccc; z-index:9999;'>"
                        "<img src=\"data:image/png;base64," + img_b64 + "\" style=\"width:100%\">"
                        "</div>"
                    )

            # Add the HTML overlay to the map
            from branca.element import Element
            m.get_root().html.add_child(Element(demand_html))
    except Exception as e:
        logger.warning(f"Could not create embedded demand plot: {e}")

    if output_html:
        logger.info(f"Saving map to: {output_html}")
        Path(output_html).write_text(m.get_root().render(), encoding="utf‑8")
        logger.info(f"Map saved successfully: {output_html}")
    else:
        logger.info("No output path specified - map created in memory only")

    return m

def plot_assets_folium(network, out_html, use_real_location=True):
    """
    Create folium map showing generators and other assets at their real locations.
    
    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with generators
    out_html : str
        Output HTML file path
    use_real_location : bool
        Whether to use real generator locations (geo_lon/geo_lat) or bus locations
    """
    logger = setup_logging("plot_assets_folium")
    logger.info(f"Creating asset map with real locations: {use_real_location}")
    
    if network.generators.empty:
        logger.warning("No generators found in network")
        return
    
    generators = network.generators.copy()
    logger.info(f"Plotting {len(generators)} generators")
    
    # Get bus coordinates for fallback
    bus_coord = {bus_id: (row.y, row.x) for bus_id, row in network.buses.iterrows()}
    
    # Get generator locations
    valid_generators = []
    for gen_id, gen in generators.iterrows():
        lat, lon = None, None
        
        if use_real_location and 'geo_lat' in gen and 'geo_lon' in gen:
            # Use real generator location if available
            if pd.notna(gen['geo_lat']) and pd.notna(gen['geo_lon']):
                lat, lon = float(gen['geo_lat']), float(gen['geo_lon'])
                logger.debug(f"{gen_id}: Using real location ({lat:.4f}, {lon:.4f})")
        
        # Fallback to bus location
        if lat is None or lon is None:
            bus = gen.get('bus')
            if bus in bus_coord:
                lat, lon = bus_coord[bus]
                logger.debug(f"{gen_id}: Using bus location ({lat:.4f}, {lon:.4f})")
            else:
                logger.warning(f"{gen_id}: No location available, skipping")
                continue
        
        # Validate coordinates (UK bounds)
        if lat < 49 or lat > 61 or lon < -8 or lon > 2:
            logger.warning(f"{gen_id}: Coordinates ({lat:.4f}, {lon:.4f}) outside UK bounds")
            continue
        
        valid_generators.append({
            'id': gen_id,
            'lat': lat,
            'lon': lon,
            'carrier': gen.get('carrier', 'unknown'),
            'p_nom': gen.get('p_nom', 0),
            'marginal_cost': gen.get('marginal_cost', 'N/A'),
            'commissioning_date': gen.get('commissioning_date', 'N/A'),
            'bus': gen.get('bus', 'N/A'),
            'efficiency': gen.get('efficiency', 'N/A')
        })
    
    logger.info(f"Found {len(valid_generators)} generators with valid coordinates")
    
    if not valid_generators:
        logger.error("No generators with valid coordinates found")
        return
    
    # Create map centered on generators
    lats = [g['lat'] for g in valid_generators]
    lons = [g['lon'] for g in valid_generators]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')
    
    # Define colors for different carriers
    carrier_colors = {
        'gas': 'gray',
        'coal': 'black', 
        'nuclear': 'purple',
        'hydro': 'lightblue',
        'wind': 'green',
        'solar': 'orange',
        'biomass': 'brown',
        'battery': 'yellow',
        'oil': 'darkred'
    }
    
    # Group by carrier and create layers
    generators_by_carrier = {}
    for gen in valid_generators:
        carrier = gen['carrier']
        if carrier not in generators_by_carrier:
            generators_by_carrier[carrier] = []
        generators_by_carrier[carrier].append(gen)
    
    for carrier, gen_list in generators_by_carrier.items():
        color = carrier_colors.get(carrier, 'red')
        layer = folium.FeatureGroup(name=f"{carrier.title()} Generators ({len(gen_list)})")
        
        for gen in gen_list:
            # Create detailed popup
            popup_text = f"""
            <b>{carrier.title()} Generator</b><br>
            <b>ID:</b> {gen['id']}<br>
            <b>Capacity:</b> {gen['p_nom']:.1f} MW<br>
            <b>Marginal Cost:</b> {gen['marginal_cost']}<br>
            <b>Efficiency:</b> {gen['efficiency']}<br>
            <b>Bus:</b> {gen['bus']}<br>
            <b>Commissioning:</b> {gen['commissioning_date']}<br>
            <b>Location:</b> {gen['lat']:.4f}, {gen['lon']:.4f}
            """
            
            # Size marker based on capacity
            marker_size = max(5, min(25, gen['p_nom'] / 100 + 5))
            
            folium.CircleMarker(
                location=[gen['lat'], gen['lon']],
                radius=marker_size,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{gen['id']}: {gen['p_nom']:.1f} MW",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(layer)
        
        layer.add_to(m)
        logger.info(f"Added {len(gen_list)} {carrier} generators to map")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    output_path = Path(out_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"Asset map saved to {output_path}")

def _add_generators_layers(map_obj, network, coord, logger):
    """Add generators from the PyPSA network as toggleable layers to the Folium map."""
    
    if len(network.generators) == 0:
        logger.info("No generators found in network")
        return
    
    logger.info(f"Adding {len(network.generators)} generators to map")
    
    # Define colors and icons for different generator types
    generator_config = {
        'Wind Onshore': {'color': 'green', 'icon': 'fa-wind', 'layer_name': 'Wind Onshore Generators'},
        'Wind_Onshore': {'color': 'green', 'icon': 'fa-wind', 'layer_name': 'Wind Onshore Generators'},
        'Wind Offshore': {'color': 'darkblue', 'icon': 'fa-ship', 'layer_name': 'Wind Offshore Generators'},
        'Wind_Offshore': {'color': 'darkblue', 'icon': 'fa-ship', 'layer_name': 'Wind Offshore Generators'},
        'Solar PV': {'color': 'orange', 'icon': 'fa-sun-o', 'layer_name': 'Solar PV Generators'},
        'Solar_PV': {'color': 'orange', 'icon': 'fa-sun-o', 'layer_name': 'Solar PV Generators'},
        'CCGT': {'color': 'gray', 'icon': 'fa-fire', 'layer_name': 'CCGT Generators'},
        'Nuclear': {'color': 'purple', 'icon': 'fa-atom', 'layer_name': 'Nuclear Generators'},
        'Hydro': {'color': 'lightblue', 'icon': 'fa-tint', 'layer_name': 'Hydro Generators'},
        'Coal': {'color': 'black', 'icon': 'fa-cube', 'layer_name': 'Coal Generators'},
        'Oil': {'color': 'brown', 'icon': 'fa-oil-can', 'layer_name': 'Oil Generators'},
        'gas': {'color': 'gray', 'icon': 'fa-fire', 'layer_name': 'Gas Generators'},
        'battery': {'color': 'yellow', 'icon': 'fa-battery', 'layer_name': 'Battery Storage'},
        'biomass': {'color': 'brown', 'icon': 'fa-leaf', 'layer_name': 'Biomass Generators'},
    }
    
    # Group generators by carrier type
    generators_by_carrier = network.generators.groupby('carrier')
    
    # Create a layer for each generator type
    for carrier, gen_group in generators_by_carrier:
        config = generator_config.get(carrier, {
            'color': 'red', 
            'icon': 'fa-bolt', 
            'layer_name': f'{carrier} Generators'
        })
        
        # Create FeatureGroup for this generator type
        layer = folium.FeatureGroup(name=config['layer_name'])
        generators_added = 0
        
        for gen_id, gen_data in gen_group.iterrows():
            # Try to use real coordinates first, then fallback to bus
            lat, lon = None, None
            
            if 'geo_lat' in gen_data and 'geo_lon' in gen_data:
                if pd.notna(gen_data['geo_lat']) and pd.notna(gen_data['geo_lon']):
                    lat, lon = float(gen_data['geo_lat']), float(gen_data['geo_lon'])
            
            # Fallback to bus coordinates
            if lat is None or lon is None:
                bus = gen_data['bus']
                if bus in coord:
                    lat, lon = coord[bus]
                else:
                    continue
            
            capacity_mw = gen_data.get('p_nom', 0)
            
            # Create popup with generator information
            popup_text = f"""
            <b>{config['layer_name']}</b><br>
            Generator ID: {gen_id}<br>
            Bus: {gen_data['bus']}<br>
            Capacity: {capacity_mw:.1f} MW<br>
            Carrier: {carrier}<br>
            Marginal Cost: {gen_data.get('marginal_cost', 'N/A')}<br>
            Real Location: {lat:.4f}, {lon:.4f}<br>
            """
            
            # Add marker with size proportional to capacity (min 5, max 20 pixels)
            marker_size = max(5, min(20, capacity_mw / 50))
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_size,
                popup=popup_text,
                tooltip=f"{carrier}: {capacity_mw:.1f} MW",
                color=config['color'],
                fill=True,
                fillColor=config['color'],
                fillOpacity=0.6,
                weight=2
            ).add_to(layer)
            
            generators_added += 1
        
        if generators_added > 0:
            layer.add_to(map_obj)
            logger.info(f"Added {generators_added} {carrier} generators to map")
        else:
            logger.warning(f"No {carrier} generators could be mapped (no valid coordinates)")

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


def look_at_GSP_data(logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Loading GSP data from GeoJSON file")
    # Load the GeoJSON file
    gdf = gpd.read_file("data/network/GSP_regions_27700_20250109.geojson")

    # --- Explore the dataset ---
    # 1. Print the column names
    logger.info(f"Columns: {gdf.columns.tolist()}")

    # 2. Print unique geometry types
    logger.info(f"Geometry types: {gdf.geom_type.unique()}")

    # 3. Show the number of features
    logger.info(f"Number of regions: {len(gdf)}")

    # 4. Show the first few records
    logger.info("First few records:")
    log_dataframe_info(gdf.head(), logger)

    # 5. Plot the geometries with labels
    logger.info("Creating GSP regions plot")
    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.6)
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row.get("GSPs", ""), fontsize=6, ha='center')
    plt.title("GSP Regions (with GSP labels)")
    plt.axis("off")
    plt.close()  # Save instead of show for consistency
    logger.info("GSP regions plot created")



if __name__ == "__main__":
    # Set up logging
    logger = setup_logging("plotting", log_level="INFO")
    
    _PATH = snakemake.input.network if hasattr(snakemake.input, 'network') else snakemake.input[0]
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
    
    logger.info("="*50)
    logger.info("STARTING MAP GENERATION")
    logger.info(f"Generating Folium map from {_PATH}")
    if renewable_sites:
        logger.info(f"Including renewable sites: {list(renewable_sites.keys())}")
    
    create_pypsa_folium_map(_PATH, output_html=out_html, renewable_sites=renewable_sites, logger=logger)
    
    logger.info(f"MAP GENERATION COMPLETED - open {out_html} in your browser")

    # look_at_GSP_data(logger)

