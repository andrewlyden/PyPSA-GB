#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of solved PyPSA networks.

Generates:
  1. Interactive spatial HTML plot (Plotly with map background)
  2. Results analysis dashboard (Plotly subplots)
  3. Comprehensive JSON summary

This is the consolidated post-processing step for solved networks,
combining plotting, visualization, and analysis.
"""

import pypsa
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from pathlib import Path

# Import logging configuration
try:
    from logging_config import setup_logging
except ImportError:
    from scripts.utilities.logging_config import setup_logging

# Initialize logger at module level (will be reconfigured in main() if needed)
logger = logging.getLogger(__name__)


def load_network(network_path):
    """Load solved network"""
    logger.info(f"Loading network from {network_path}")
    n = pypsa.Network(network_path)
    logger.info(f"Network loaded: {len(n.buses)} buses, {len(n.generators)} generators, {len(n.loads)} loads")
    return n


def create_spatial_plot(n, output_path):
    """
    Create interactive spatial plot of network with Plotly.
    
    Shows:
    - Network topology with buses and lines
    - Generator locations and capacity (color/size coded by carrier and capacity)
    - Storage locations
    - Transmission flows and loading
    
    Handles both OSGB36 (meters) and WGS84 (degrees) coordinate systems,
    including networks with mixed coordinate systems.
    """
    logger.info("Creating interactive spatial plot...")
    
    # Prepare data
    buses = n.buses.copy()
    lines = n.lines.copy()
    generators = n.generators.copy()
    storage = n.storage_units.copy() if len(n.storage_units) > 0 else None
    
    from pyproj import Transformer
    
    # Detect coordinate system for each bus individually
    # OSGB36: x values are in meters (typically 100,000 to 700,000 for GB)
    # WGS84: x values are in degrees (-10 to 2 for GB)
    # Use a threshold of 100 - WGS84 values are always between -180 and 180
    buses['is_osgb36'] = (buses['x'].abs() > 100) | (buses['y'].abs() > 100)
    osgb_count = buses['is_osgb36'].sum()
    wgs_count = len(buses) - osgb_count
    
    logger.info(f"Coordinate system detection: {wgs_count} WGS84 buses, {osgb_count} OSGB36 buses")
    
    # Initialize WGS84 columns
    buses['lon_wgs84'] = buses['x']
    buses['lat_wgs84'] = buses['y']
    
    # Convert OSGB36 buses to WGS84 if any exist
    if osgb_count > 0:
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        osgb_mask = buses['is_osgb36']
        osgb_buses = buses[osgb_mask]
        
        if len(osgb_buses) > 0:
            lon_conv, lat_conv = transformer.transform(
                osgb_buses['x'].values, osgb_buses['y'].values
            )
            buses.loc[osgb_mask, 'lon_wgs84'] = lon_conv
            buses.loc[osgb_mask, 'lat_wgs84'] = lat_conv
            logger.info(f"Converted {len(osgb_buses)} OSGB36 buses to WGS84")
    
    # Create base figure
    fig = go.Figure()
    
    # Add transmission lines
    if len(lines) > 0:
        line_lons = []
        line_lats = []
        
        for idx, line in lines.iterrows():
            bus0 = buses.loc[line['bus0']]
            bus1 = buses.loc[line['bus1']]
            
            line_lons.extend([bus0['lon_wgs84'], bus1['lon_wgs84'], None])
            line_lats.extend([bus0['lat_wgs84'], bus1['lat_wgs84'], None])
        
        fig.add_trace(go.Scattergeo(
            lon=line_lons,
            lat=line_lats,
            mode='lines',
            line=dict(width=1, color='rgba(100, 100, 200, 0.5)'),
            name='Transmission Lines',
            showlegend=True
        ))
    
    # Add buses (colored by voltage level)
    if len(buses) > 0:
        bus_sizes = []
        bus_colors = []
        bus_hover = []
        
        for idx, bus in buses.iterrows():
            gen_cap = generators[generators['bus'] == idx]['p_nom'].sum() if len(generators) > 0 else 0
            bus_sizes.append(max(5, min(15, 5 + gen_cap / 1500)))
            
            if 'v_nom' in bus.index:
                if bus['v_nom'] >= 400:
                    bus_colors.append('darkred')
                elif bus['v_nom'] >= 275:
                    bus_colors.append('orange')
                elif bus['v_nom'] >= 132:
                    bus_colors.append('gold')
                else:
                    bus_colors.append('lightblue')
            else:
                bus_colors.append('blue')
            
            bus_hover.append(f"<b>{idx}</b><br>Generators: {gen_cap:.0f} MW")
        
        fig.add_trace(go.Scattergeo(
            lon=buses['lon_wgs84'],
            lat=buses['lat_wgs84'],
            mode='markers',
            marker=dict(
                size=bus_sizes,
                color=bus_colors,
                opacity=0.8,
                line=dict(width=1, color='darkgray')
            ),
            text=bus_hover,
            hovertemplate='%{text}<extra></extra>',
            name='Network Buses',
            showlegend=True
        ))
    
    # Add generators
    gen_data = generators.copy()
    if 'x' in gen_data.columns and 'y' in gen_data.columns:
        gen_data = gen_data.dropna(subset=['x', 'y'])
        
        if len(gen_data) > 0:
            # Convert coordinates - handle mixed coordinate systems
            gen_data['is_osgb36'] = (gen_data['x'].abs() > 100) | (gen_data['y'].abs() > 100)
            gen_data['lon_wgs84'] = gen_data['x']
            gen_data['lat_wgs84'] = gen_data['y']
            
            gen_osgb_mask = gen_data['is_osgb36']
            if gen_osgb_mask.any():
                transformer_gen = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                osgb_gens = gen_data[gen_osgb_mask]
                lon_conv, lat_conv = transformer_gen.transform(osgb_gens['x'].values, osgb_gens['y'].values)
                gen_data.loc[gen_osgb_mask, 'lon_wgs84'] = lon_conv
                gen_data.loc[gen_osgb_mask, 'lat_wgs84'] = lat_conv
            
            carrier_colors = {
                'wind_offshore': '#0073E6',
                'wind_onshore': '#66B2FF',
                'solar_pv': '#FFD700',
                'CCGT': '#FF4500',
                'nuclear': '#9400D3',
                'large_hydro': '#228B22',
                'battery': '#FFA500',
                'pumped_hydro': '#32CD32',
                'H2_turbine': '#00CED1',
                'load_shedding': '#000000'
            }
            
            gen_data['color'] = gen_data['carrier'].map(carrier_colors).fillna('#808080')
            
            fig.add_trace(go.Scattergeo(
                lon=gen_data['lon_wgs84'],
                lat=gen_data['lat_wgs84'],
                mode='markers',
                marker=dict(
                    size=np.sqrt(gen_data['p_nom'].clip(0, 5000)) / 4 + 2,
                    color=gen_data['color'],
                    opacity=0.7,
                    line=dict(width=0.5, color='darkgray')
                ),
                text=[f"<b>{row['carrier']}</b><br>{row['p_nom']:.0f} MW<br>Bus: {row['bus']}" 
                      for _, row in gen_data.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name='Generators',
                showlegend=True
            ))
    
    # Add storage
    if storage is not None and len(storage) > 0:
        storage_data = storage.copy()
        if 'x' in storage_data.columns and 'y' in storage_data.columns:
            storage_data = storage_data.dropna(subset=['x', 'y'])
            
            if len(storage_data) > 0:
                # Convert coordinates - handle mixed coordinate systems
                storage_data['is_osgb36'] = (storage_data['x'].abs() > 100) | (storage_data['y'].abs() > 100)
                storage_data['lon_wgs84'] = storage_data['x']
                storage_data['lat_wgs84'] = storage_data['y']
                
                stor_osgb_mask = storage_data['is_osgb36']
                if stor_osgb_mask.any():
                    transformer_stor = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                    osgb_stor = storage_data[stor_osgb_mask]
                    lon_conv, lat_conv = transformer_stor.transform(osgb_stor['x'].values, osgb_stor['y'].values)
                    storage_data.loc[stor_osgb_mask, 'lon_wgs84'] = lon_conv
                    storage_data.loc[stor_osgb_mask, 'lat_wgs84'] = lat_conv
                
                fig.add_trace(go.Scattergeo(
                    lon=storage_data['lon_wgs84'],
                    lat=storage_data['lat_wgs84'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='darkgreen',
                        symbol='diamond',
                        opacity=0.8,
                        line=dict(width=1, color='darkgray')
                    ),
                    text=[f"<b>{row['carrier']}</b><br>Power: {row['p_nom']:.0f} MW" 
                          for _, row in storage_data.iterrows()],
                    hovertemplate='%{text}<extra></extra>',
                    name='Storage',
                    showlegend=True
                ))
    
    # Update layout with GB-focused map
    fig.update_layout(
        title=dict(
            text=f"<b>Network Topology Map: {n.name}</b><br><sup>Interactive visualization</sup>",
            x=0.5,
            xanchor='center'
        ),
        geo=dict(
            scope='europe',
            projection_type='mercator',
            center=dict(lon=-2.5, lat=54.5),
            lonaxis_range=[-8, 2],
            lataxis_range=[50, 60],
            showland=True,
            landcolor='rgb(240, 240, 240)',
            showocean=True,
            oceancolor='rgb(220, 235, 255)',
            showcountries=True,
            countrycolor='rgb(200, 200, 200)',
            showcoastlines=True,
            coastlinecolor='rgb(100, 100, 100)',
        ),
        height=900,
        hovermode='closest',
        margin=dict(r=10, t=100, l=10, b=10),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    fig.write_html(output_path)
    logger.info(f"Spatial plot saved to {output_path}")
    return fig


def get_generation_including_links(n):
    """
    Get combined generation time series from generators AND power-producing links.
    
    This includes:
    - All generators (n.generators_t.p)
    - H2_turbine links (hydrogen power plants modeled as links)
    
    Returns:
        tuple: (gen_ts by carrier, gen_capacity by carrier)
    """
    # Standard generators
    gen_ts = n.generators_t.p.groupby(n.generators['carrier'], axis=1).sum()
    gen_capacity = n.generators.groupby('carrier')['p_nom'].sum()
    
    # Add H2_turbine links (power output is |p1| which is positive to bus1)
    h2_links = n.links[n.links['carrier'] == 'H2_turbine']
    if len(h2_links) > 0 and len(n.links_t.p1) > 0:
        # p1 is negative (power out of link), so take absolute value
        h2_output = n.links_t.p1[h2_links.index].abs()
        h2_ts = h2_output.sum(axis=1)
        h2_ts.name = 'H2_turbine'
        
        # Add to generation time series
        if 'H2_turbine' in gen_ts.columns:
            gen_ts['H2_turbine'] = gen_ts['H2_turbine'] + h2_ts
        else:
            gen_ts['H2_turbine'] = h2_ts
        
        # Add capacity
        h2_cap = h2_links['p_nom'].sum()
        if 'H2_turbine' in gen_capacity.index:
            gen_capacity['H2_turbine'] = gen_capacity['H2_turbine'] + h2_cap
        else:
            gen_capacity['H2_turbine'] = h2_cap
        
        logger.info(f"Added H2_turbine generation: {h2_cap:.0f} MW capacity, {h2_ts.sum():.0f} MWh total")
    
    return gen_ts, gen_capacity


def create_results_dashboard(n, output_path):
    """
    Create comprehensive results dashboard with Plotly subplots.
    
    Includes:
    - Generation mix time series (stacked area) - includes H2_turbine links
    - Peak generation by carrier (bar chart)
    - Storage state of charge (line chart)
    - Load shedding events (area chart)
    - Line loading distribution (histogram)
    - System cost breakdown (pie chart)
    """
    logger.info("Creating results dashboard...")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Hourly Generation Mix",
            "Peak Generation by Carrier",
            "Storage State of Charge",
            "Load Shedding Events",
            "Line Loading Distribution",
            "System Cost Breakdown"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # 1. Generation mix time series (including H2_turbine links)
    gen_ts, gen_capacity = get_generation_including_links(n)
    top_carriers = gen_ts.sum().nlargest(12).index
    gen_ts_top = gen_ts[top_carriers]
    
    for carrier in gen_ts_top.columns:
        fig.add_trace(
            go.Scatter(
                x=gen_ts_top.index,
                y=gen_ts_top[carrier],
                name=carrier,
                mode='lines',
                stackgroup='gen',
            ),
            row=1, col=1
        )
    
    # 2. Peak generation by carrier (including H2_turbine links)
    peak_gen = gen_ts.max()  # Use the combined gen_ts which includes H2_turbine
    top_carriers_peak = peak_gen.nlargest(10)
    
    fig.add_trace(
        go.Bar(
            x=top_carriers_peak.index,
            y=top_carriers_peak.values,
            name='Peak',
            marker_color='steelblue',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Storage state of charge
    if len(n.storage_units) > 0 and len(n.storage_units_t.state_of_charge) > 0:
        for i, idx in enumerate(n.storage_units.index[:5]):
            fig.add_trace(
                go.Scatter(
                    x=n.storage_units_t.state_of_charge.index,
                    y=n.storage_units_t.state_of_charge[idx],
                    name=f"Storage {i+1}",
                    mode='lines',
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # 4. Load shedding events
    load_shed_gens = n.generators[n.generators['carrier'] == 'load_shedding']
    if len(load_shed_gens) > 0:
        load_shed = n.generators_t.p[load_shed_gens.index].sum(axis=1)
        fig.add_trace(
            go.Scatter(
                x=load_shed.index,
                y=load_shed,
                name='Load Shedding',
                mode='lines',
                fill='tozeroy',
                line=dict(color='red'),
                showlegend=False
            ),
            row=2, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=n.snapshots,
                y=[0] * len(n.snapshots),
                name='No Load Shedding',
                mode='lines',
                line=dict(color='green'),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # 5. Line loading distribution
    if len(n.lines) > 0 and len(n.lines_t.p0) > 0:
        max_loading = (n.lines_t.p0.abs().max() / n.lines['s_nom'].values).replace([np.inf, -np.inf], np.nan)
        
        fig.add_trace(
            go.Histogram(
                x=max_loading.dropna(),
                name='Line Loading',
                nbinsx=40,
                marker_color='purple',
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 6. Cost breakdown
    total_cost = n.objective if hasattr(n, 'objective') and n.objective else 0
    gen_cost = (n.generators['marginal_cost'] * n.generators_t.p.sum()).sum()
    
    costs = {'Generation': gen_cost, 'Other': max(0, total_cost - gen_cost)}
    
    fig.add_trace(
        go.Pie(
            labels=list(costs.keys()),
            values=list(costs.values()),
            name='Cost',
            showlegend=False
        ),
        row=3, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Carrier", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_xaxes(title_text="Max Loading (p.u.)", row=3, col=1)
    
    fig.update_yaxes(title_text="Generation (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Peak MW", row=1, col=2)
    fig.update_yaxes(title_text="State (MWh)", row=2, col=1)
    fig.update_yaxes(title_text="Load Shed (MW)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text=f"<b>Results Dashboard: {n.name}</b>",
        hovermode='x unified',
        font=dict(size=10)
    )
    
    fig.write_html(output_path)
    logger.info(f"Dashboard saved to {output_path}")
    return fig


def create_analysis_summary(n, output_path):
    """Generate JSON summary of key results and metrics (including H2_turbine links)"""
    logger.info("Creating analysis summary...")
    
    load_shed_gens = n.generators[n.generators['carrier'] == 'load_shedding']
    load_shedding = n.generators_t.p[load_shed_gens.index].sum().sum() if len(load_shed_gens) > 0 else 0
    
    # Get combined generation including H2_turbine links
    gen_ts, gen_capacity = get_generation_including_links(n)
    gen_by_carrier = gen_ts.sum()
    total_demand = n.loads_t.p.sum().sum()
    
    summary = {
        'scenario': str(n.name),
        'timestamp': pd.Timestamp.now().isoformat(),
        'network_size': {
            'buses': int(len(n.buses)),
            'lines': int(len(n.lines)),
            'transformers': int(len(n.transformers)),
            'generators': int(len(n.generators)),
            'loads': int(len(n.loads)),
            'storage_units': int(len(n.storage_units)),
            'h2_turbine_links': int(len(n.links[n.links['carrier'] == 'H2_turbine']))
        },
        'results': {
            'total_cost_gbp': float(n.objective) if hasattr(n, 'objective') and n.objective else 0,
            'total_generation_mwh': float(gen_by_carrier.sum()),
            'total_demand_mwh': float(total_demand),
            'load_shedding_mwh': float(load_shedding),
            'load_shedding_pct': float(load_shedding / total_demand * 100 if total_demand > 0 else 0),
            'energy_balance_error_pct': float(abs(gen_by_carrier.sum() - total_demand) / total_demand * 100 if total_demand > 0 else 0),
        },
        'peak_demand_mw': float(n.loads_t.p.sum(axis=1).max()),
        'avg_demand_mw': float(n.loads_t.p.sum(axis=1).mean()),
        'generation_by_carrier': {k: float(v) for k, v in gen_by_carrier.items()},
        'installed_capacity_by_carrier': {k: float(v) for k, v in gen_capacity.items()},
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {output_path}")
    return summary


def main():
    """Main analysis pipeline - consolidates plotting, analysis, and notebook generation"""
    global logger
    
    # Reinitialize logger with Snakemake log path
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "analyze_solved_network"
    logger = setup_logging(log_path)
    
    # Read snakemake inputs/outputs/params
    network_path = snakemake.input.network
    spatial_plot_output = snakemake.output.spatial_plot
    dashboard_output = snakemake.output.dashboard
    summary_output = snakemake.output.summary
    
    # Load and analyze network
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE NETWORK ANALYSIS")
    logger.info("=" * 80)
    
    n = load_network(network_path)
    
    # Generate all outputs
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    create_spatial_plot(n, spatial_plot_output)
    create_results_dashboard(n, dashboard_output)
    create_analysis_summary(n, summary_output)
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✓ Spatial plot: {spatial_plot_output}")
    logger.info(f"✓ Results dashboard: {dashboard_output}")
    logger.info(f"✓ Analysis summary: {summary_output}")


if __name__ == "__main__":
    main()

