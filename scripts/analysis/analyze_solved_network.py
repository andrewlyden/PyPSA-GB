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


def _to_wgs84(df, x_col='x', y_col='y'):
    """Convert a DataFrame's coordinates to WGS84 lon/lat.

    Automatically detects OSGB36 (meters) vs WGS84 (degrees) per row.
    Returns (lon_series, lat_series) in WGS84.
    """
    from pyproj import Transformer

    lon = df[x_col].copy()
    lat = df[y_col].copy()
    is_osgb = (lon.abs() > 100) | (lat.abs() > 100)

    if is_osgb.any():
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        osgb = df[is_osgb]
        lon_conv, lat_conv = transformer.transform(
            osgb[x_col].values, osgb[y_col].values
        )
        lon.loc[is_osgb] = lon_conv
        lat.loc[is_osgb] = lat_conv

    return lon, lat


def create_spatial_plot(n, output_path):
    """
    Create interactive spatial plot of network with Plotly.

    Shows all network components:
    - Transmission lines, transformers, and links (interconnectors)
    - Buses colored by voltage level
    - Generators colored by carrier (per-carrier legend entries)
    - Storage units

    Handles both OSGB36 (meters) and WGS84 (degrees) coordinate systems.
    """
    logger.info("Creating interactive spatial plot...")

    buses = n.buses.copy()
    buses['lon_wgs84'], buses['lat_wgs84'] = _to_wgs84(buses)

    osgb_count = ((buses['x'].abs() > 100) | (buses['y'].abs() > 100)).sum()
    logger.info(f"Coordinate systems: {len(buses) - osgb_count} WGS84, {osgb_count} OSGB36")

    fig = go.Figure()

    # Helper: build line segments between bus pairs
    def _build_segments(comp_df):
        lons, lats = [], []
        for _, row in comp_df.iterrows():
            b0, b1 = row['bus0'], row['bus1']
            if b0 in buses.index and b1 in buses.index:
                lons.extend([buses.loc[b0, 'lon_wgs84'], buses.loc[b1, 'lon_wgs84'], None])
                lats.extend([buses.loc[b0, 'lat_wgs84'], buses.loc[b1, 'lat_wgs84'], None])
        return lons, lats

    # --- Transmission Lines ---
    if len(n.lines) > 0:
        lons, lats = _build_segments(n.lines)
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats, mode='lines',
            line=dict(width=1, color='rgba(100, 100, 200, 0.5)'),
            name=f'Lines ({len(n.lines)})', showlegend=True,
        ))

    # --- Transformers ---
    if len(n.transformers) > 0:
        lons, lats = _build_segments(n.transformers)
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats, mode='lines',
            line=dict(width=1, color='rgba(200, 140, 50, 0.4)'),
            name=f'Transformers ({len(n.transformers)})', showlegend=True,
        ))

    # --- Links (split by type for clarity) ---
    if len(n.links) > 0:
        # Categorise links into distinct groups
        ic_mask = n.links.index.str.startswith('IC_')
        ic_links = n.links[ic_mask]
        internal = n.links[~ic_mask]

        # Internal HVDC (carrier == 'DC', not interconnectors)
        hvdc_internal = internal[internal['carrier'] == 'DC']
        # Hydrogen electrolysis links
        h2_elec = internal[internal['carrier'] == 'electrolysis']
        # Hydrogen turbine links
        h2_turb = internal[internal['carrier'] == 'H2_turbine']
        # Anything else (future-proof)
        other_carriers = internal[
            ~internal['carrier'].isin(['DC', 'electrolysis', 'H2_turbine'])
        ]

        if len(hvdc_internal) > 0:
            lons, lats = _build_segments(hvdc_internal)
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode='lines',
                line=dict(width=3, color='rgba(180, 50, 180, 0.85)', dash='dash'),
                name=f'Internal HVDC ({len(hvdc_internal)})', showlegend=True,
            ))

        if len(ic_links) > 0:
            lons, lats = _build_segments(ic_links)
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode='lines',
                line=dict(width=2.5, color='rgba(220, 20, 60, 0.8)', dash='dot'),
                name=f'Interconnectors ({len(ic_links)})', showlegend=True,
            ))

        if len(h2_elec) > 0:
            lons, lats = _build_segments(h2_elec)
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode='lines',
                line=dict(width=1.5, color='rgba(0, 180, 120, 0.6)', dash='dashdot'),
                name=f'H\u2082 Electrolysis ({len(h2_elec)})', showlegend=True,
            ))

        if len(h2_turb) > 0:
            lons, lats = _build_segments(h2_turb)
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode='lines',
                line=dict(width=1.5, color='rgba(0, 120, 180, 0.5)', dash='dashdot'),
                name=f'H\u2082 Turbines ({len(h2_turb)})', showlegend=True,
            ))

        if len(other_carriers) > 0:
            lons, lats = _build_segments(other_carriers)
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode='lines',
                line=dict(width=1.5, color='rgba(150, 150, 150, 0.6)', dash='dot'),
                name=f'Other Links ({len(other_carriers)})', showlegend=True,
            ))

    # --- Buses ---
    if len(buses) > 0:
        generators = n.generators
        gen_cap_per_bus = generators.groupby('bus')['p_nom'].sum() if len(generators) > 0 else pd.Series(dtype=float)

        bus_sizes = []
        bus_colors = []
        bus_hover = []

        for idx, bus in buses.iterrows():
            cap = gen_cap_per_bus.get(idx, 0)
            bus_sizes.append(max(4, min(15, 4 + cap / 1500)))

            v = bus.get('v_nom', 0)
            if v >= 400:
                bus_colors.append('darkred')
            elif v >= 275:
                bus_colors.append('orange')
            elif v >= 132:
                bus_colors.append('gold')
            elif v > 0:
                bus_colors.append('lightblue')
            else:
                bus_colors.append('gray')

            bus_hover.append(f"<b>{idx}</b><br>v_nom: {v:.0f} kV<br>Gen capacity: {cap:.0f} MW")

        fig.add_trace(go.Scattergeo(
            lon=buses['lon_wgs84'], lat=buses['lat_wgs84'], mode='markers',
            marker=dict(size=bus_sizes, color=bus_colors, opacity=0.8,
                        line=dict(width=0.5, color='darkgray')),
            text=bus_hover, hovertemplate='%{text}<extra></extra>',
            name=f'Buses ({len(buses)})', showlegend=True,
        ))

    # --- Generators (per-carrier traces for legend toggling) ---
    carrier_colors = {
        'wind_offshore': '#0073E6',
        'wind_onshore': '#66B2FF',
        'solar_pv': '#FFD700',
        'CCGT': '#FF4500',
        'OCGT': '#FF6347',
        'nuclear': '#9400D3',
        'large_hydro': '#228B22',
        'small_hydro': '#3CB371',
        'battery': '#FFA500',
        'pumped_hydro': '#32CD32',
        'H2_turbine': '#00CED1',
        'coal': '#8B4513',
        'oil': '#A0522D',
        'biogas': '#6B8E23',
        'advanced_biofuel': '#556B2F',
        'waste_to_energy': '#708090',
        'landfill_gas': '#808000',
        'EU_import': '#DC143C',
    }

    if len(n.generators) > 0:
        gen_data = n.generators.copy()
        # Exclude load_shedding — it's at every bus and clutters the map
        gen_data = gen_data[gen_data['carrier'] != 'load_shedding']

        # Resolve generator coordinates: try lon/lat (auto-detect OSGB36), then x/y, then bus
        if 'lon' in gen_data.columns and 'lat' in gen_data.columns:
            has_own = gen_data['lon'].notna() & gen_data['lat'].notna()
            if has_own.any():
                lon_conv, lat_conv = _to_wgs84(gen_data[has_own], x_col='lon', y_col='lat')
                gen_data.loc[has_own, 'lon_wgs84'] = lon_conv
                gen_data.loc[has_own, 'lat_wgs84'] = lat_conv
        else:
            has_own = pd.Series(False, index=gen_data.index)

        if 'x' in gen_data.columns and 'y' in gen_data.columns:
            has_xy = (~has_own) & gen_data['x'].notna() & gen_data['y'].notna()
            if has_xy.any():
                lon_xy, lat_xy = _to_wgs84(gen_data[has_xy])
                gen_data.loc[has_xy, 'lon_wgs84'] = lon_xy
                gen_data.loc[has_xy, 'lat_wgs84'] = lat_xy
                has_own = has_own | has_xy

        # Fall back to bus location for remaining generators
        needs_bus = ~has_own | gen_data.get('lon_wgs84', pd.Series(dtype=float)).isna()
        if needs_bus.any():
            for idx in gen_data[needs_bus].index:
                bus_name = gen_data.loc[idx, 'bus']
                if bus_name in buses.index:
                    gen_data.loc[idx, 'lon_wgs84'] = buses.loc[bus_name, 'lon_wgs84']
                    gen_data.loc[idx, 'lat_wgs84'] = buses.loc[bus_name, 'lat_wgs84']

        gen_data = gen_data.dropna(subset=['lon_wgs84', 'lat_wgs84'])
        logger.info(f"Plotting {len(gen_data)} generators (excl. load_shedding)")

        # Plot one trace per carrier for legend toggling
        for carrier, group in gen_data.groupby('carrier'):
            color = carrier_colors.get(carrier, '#808080')
            fig.add_trace(go.Scattergeo(
                lon=group['lon_wgs84'], lat=group['lat_wgs84'], mode='markers',
                marker=dict(
                    size=np.sqrt(group['p_nom'].clip(1, 5000)) / 3 + 3,
                    color=color, opacity=0.75,
                    line=dict(width=0.5, color='darkgray'),
                ),
                text=[f"<b>{name}</b><br>{row['carrier']}<br>{row['p_nom']:.0f} MW<br>Bus: {row['bus']}"
                      for name, row in group.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name=f"{carrier} ({len(group)})",
                legendgroup='generators',
                showlegend=True,
            ))

    # --- Storage ---
    if len(n.storage_units) > 0:
        stor = n.storage_units.copy()
        # Resolve coordinates: x/y or fall back to bus
        if 'x' in stor.columns and 'y' in stor.columns:
            has_xy = stor['x'].notna() & stor['y'].notna()
            if has_xy.any():
                lon_s, lat_s = _to_wgs84(stor[has_xy])
                stor.loc[has_xy, 'lon_wgs84'] = lon_s
                stor.loc[has_xy, 'lat_wgs84'] = lat_s
        for idx in stor.index:
            if pd.isna(stor.loc[idx].get('lon_wgs84')):
                bus_name = stor.loc[idx, 'bus']
                if bus_name in buses.index:
                    stor.loc[idx, 'lon_wgs84'] = buses.loc[bus_name, 'lon_wgs84']
                    stor.loc[idx, 'lat_wgs84'] = buses.loc[bus_name, 'lat_wgs84']

        stor = stor.dropna(subset=['lon_wgs84', 'lat_wgs84'])
        if len(stor) > 0:
            fig.add_trace(go.Scattergeo(
                lon=stor['lon_wgs84'], lat=stor['lat_wgs84'], mode='markers',
                marker=dict(size=8, color='darkgreen', symbol='diamond',
                            opacity=0.8, line=dict(width=1, color='darkgray')),
                text=[f"<b>{row['carrier']}</b><br>Power: {row['p_nom']:.0f} MW<br>Bus: {row['bus']}"
                      for _, row in stor.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name=f'Storage ({len(stor)})', showlegend=True,
            ))

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text=f"<b>Network Topology Map: {n.name}</b><br>"
                 f"<sup>{len(n.buses)} buses, {len(n.lines)} lines, "
                 f"{len(n.transformers)} xfmrs, {len(n.links)} links, "
                 f"{len(n.generators)} generators</sup>",
            x=0.5, xanchor='center',
        ),
        geo=dict(
            scope='europe',
            projection_type='mercator',
            center=dict(lon=-2.5, lat=54.5),
            lonaxis_range=[-9, 4],
            lataxis_range=[49, 61],
            showland=True,
            landcolor='rgb(240, 240, 240)',
            showocean=True,
            oceancolor='rgb(220, 235, 255)',
            showcountries=True,
            countrycolor='rgb(200, 200, 200)',
            showcoastlines=True,
            coastlinecolor='rgb(100, 100, 100)',
        ),
        height=1000,
        hovermode='closest',
        margin=dict(r=10, t=100, l=10, b=10),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray', borderwidth=1,
        ),
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
    gen_ts = n.generators_t.p.T.groupby(n.generators['carrier']).sum().T
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

