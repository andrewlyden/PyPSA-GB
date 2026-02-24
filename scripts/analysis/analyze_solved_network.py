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


def _convert_to_wgs84(df, x_col='x', y_col='y'):
    """
    Convert a DataFrame's coordinate columns to WGS84, handling mixed coordinate systems.
    
    Adds 'lon_wgs84' and 'lat_wgs84' columns in-place.
    Returns the modified DataFrame (rows with valid coordinates only).
    """
    from pyproj import Transformer
    
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    if len(df) == 0:
        df['lon_wgs84'] = pd.Series(dtype=float)
        df['lat_wgs84'] = pd.Series(dtype=float)
        return df
    
    # Detect OSGB36 vs WGS84 per row
    df['_is_osgb'] = (df[x_col].abs() > 100) | (df[y_col].abs() > 100)
    df['lon_wgs84'] = df[x_col]
    df['lat_wgs84'] = df[y_col]
    
    osgb_mask = df['_is_osgb']
    if osgb_mask.any():
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lon_conv, lat_conv = transformer.transform(
            df.loc[osgb_mask, x_col].values,
            df.loc[osgb_mask, y_col].values
        )
        df.loc[osgb_mask, 'lon_wgs84'] = lon_conv
        df.loc[osgb_mask, 'lat_wgs84'] = lat_conv
    
    df.drop(columns=['_is_osgb'], inplace=True)
    return df


def _get_bus_wgs84(buses_wgs84, bus_name):
    """Get WGS84 lon/lat for a bus name. Returns (lon, lat) or (None, None)."""
    if bus_name in buses_wgs84.index:
        row = buses_wgs84.loc[bus_name]
        return row['lon_wgs84'], row['lat_wgs84']
    return None, None


def _add_line_traces(fig, component_df, buses_wgs84, name, color, width=1, dash=None):
    """Add a line-type network component (lines/transformers/links) as a trace."""
    if len(component_df) == 0:
        return
    
    lons, lats = [], []
    for _, row in component_df.iterrows():
        lon0, lat0 = _get_bus_wgs84(buses_wgs84, row['bus0'])
        lon1, lat1 = _get_bus_wgs84(buses_wgs84, row['bus1'])
        if lon0 is not None and lon1 is not None:
            lons.extend([lon0, lon1, None])
            lats.extend([lat0, lat1, None])
    
    if not lons:
        return
    
    count = len(component_df)
    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats,
        mode='lines',
        line=dict(width=width, color=color, dash=dash),
        name=f'{name} ({count})',
        showlegend=True,
        hoverinfo='skip',
    ))


def create_spatial_plot(n, output_path):
    """
    Create comprehensive interactive spatial plot of network with Plotly.
    
    Shows per the original detailed format:
    - Lines (AC transmission lines)
    - Transformers (separate trace, different style)
    - Internal HVDC links (dashed purple)
    - Interconnectors (dotted red)
    - H2 Electrolysis links (dash-dot gray)
    - H2 Turbine links (dash-dot blue)
    - Buses (small yellow dots)
    - Each generator carrier as its own legend entry with count
    - Storage units (green diamonds) with count
    
    Handles both OSGB36 (meters) and WGS84 (degrees) coordinate systems,
    including networks with mixed coordinate systems.
    """
    logger.info("Creating interactive spatial plot...")
    
    from pyproj import Transformer
    
    # --- Prepare bus coordinates (everything needs this) ---
    buses = n.buses.copy()
    buses = _convert_to_wgs84(buses)
    
    n_buses = len(n.buses)
    n_lines = len(n.lines)
    n_xfmrs = len(n.transformers)
    n_links = len(n.links)
    n_gens = len(n.generators)
    n_storage = len(n.storage_units)
    
    logger.info(f"Network components: {n_buses} buses, {n_lines} lines, "
                f"{n_xfmrs} xfmrs, {n_links} links, {n_gens} generators, {n_storage} storage")
    
    # Determine network name for title
    # Try to produce a clean display name like "HT35 (Full)" or "HT35 (Clustered)"
    net_name = n.name if n.name else Path(output_path).stem.replace('_spatial', '')
    
    # Create base figure
    fig = go.Figure()
    
    # =====================================================================
    # 1. NETWORK INFRASTRUCTURE (line-type traces)
    # =====================================================================
    
    # 1a. AC Transmission Lines
    _add_line_traces(fig, n.lines, buses,
                     name='Lines', color='rgba(120, 120, 120, 0.4)', width=1)
    
    # 1b. Transformers
    _add_line_traces(fig, n.transformers, buses,
                     name='Transformers', color='rgba(180, 180, 120, 0.4)', width=0.8)
    
    # 1c. Categorize links by carrier/type
    # Internal HVDC and cross-border interconnectors both use carrier='DC'.
    # Distinguish them by naming convention: interconnectors start with 'IC_'.
    # H2 system links have carrier='electrolysis', 'H2_turbine', or 'H2_gas'.
    links = n.links.copy()
    if len(links) > 0:
        # H2 system links (by carrier)
        h2e_mask = links['carrier'] == 'electrolysis'
        h2t_mask = links['carrier'] == 'H2_turbine'
        h2g_mask = links['carrier'] == 'H2_gas'
        
        # Interconnectors: name starts with 'IC_' or carrier is EU_import/EU_export
        ic_mask = (
            links.index.str.startswith('IC_') |
            links['carrier'].isin(['EU_import', 'EU_export'])
        )
        
        # Internal HVDC: everything else (carrier 'DC' or 'AC', not IC_ prefixed)
        hvdc_mask = ~(ic_mask | h2e_mask | h2t_mask | h2g_mask)
        
        internal_hvdc = links[hvdc_mask]
        interconnectors = links[ic_mask]
        h2_electrolysis = links[h2e_mask]
        h2_turbines = links[h2t_mask]
        
        logger.info(f"Link categorization: {len(internal_hvdc)} internal HVDC, "
                     f"{len(interconnectors)} interconnectors, "
                     f"{len(h2_electrolysis)} H2 electrolysis, "
                     f"{len(h2_turbines)} H2 turbines")
        
        _add_line_traces(fig, internal_hvdc, buses,
                         name='Internal HVDC', color='rgba(148, 0, 211, 0.7)',
                         width=2, dash='dash')
        
        _add_line_traces(fig, interconnectors, buses,
                         name='Interconnectors', color='rgba(220, 20, 60, 0.7)',
                         width=2, dash='dot')
        
        _add_line_traces(fig, h2_electrolysis, buses,
                         name='H₂ Electrolysis', color='rgba(100, 100, 100, 0.5)',
                         width=1, dash='dashdot')
        
        _add_line_traces(fig, h2_turbines, buses,
                         name='H₂ Turbines', color='rgba(70, 130, 180, 0.5)',
                         width=1, dash='dashdot')
    
    # =====================================================================
    # 2. BUSES
    # =====================================================================
    if len(buses) > 0:
        fig.add_trace(go.Scattergeo(
            lon=buses['lon_wgs84'],
            lat=buses['lat_wgs84'],
            mode='markers',
            marker=dict(size=2.5, color='#DAA520', opacity=0.6),
            text=[f"<b>{idx}</b>" for idx in buses.index],
            hovertemplate='%{text}<extra></extra>',
            name=f'Buses ({n_buses})',
            showlegend=True,
        ))
    
    # =====================================================================
    # 3. GENERATORS — one trace per carrier, with count and color
    # =====================================================================
    generators = n.generators.copy()
    # Exclude load_shedding from map (virtual generators at every bus)
    generators = generators[generators['carrier'] != 'load_shedding']
    
    # Resolve generator coordinates — prefer lon/lat, fallback to bus coordinates
    if 'lon' in generators.columns and 'lat' in generators.columns:
        gen_data = _convert_to_wgs84(generators, x_col='lon', y_col='lat')
    elif 'x' in generators.columns and 'y' in generators.columns:
        gen_data = _convert_to_wgs84(generators, x_col='x', y_col='y')
    else:
        # No direct coordinates — use bus coordinates
        gen_data = generators.copy()
        gen_data['lon_wgs84'] = gen_data['bus'].map(buses['lon_wgs84'])
        gen_data['lat_wgs84'] = gen_data['bus'].map(buses['lat_wgs84'])
        gen_data = gen_data.dropna(subset=['lon_wgs84', 'lat_wgs84'])
    
    # Load carrier color definitions
    try:
        from scripts.utilities.carrier_definitions import get_carrier_definitions
        carrier_defs = get_carrier_definitions()
        carrier_color_map = carrier_defs['color'].to_dict()
    except Exception:
        carrier_color_map = {}
    
    # Fallback colors for carriers not in definitions
    fallback_colors = {
        'wind_offshore': '#6BAED6', 'wind_onshore': '#3B6182',
        'solar_pv': '#FFBB00', 'CCGT': '#8B8B8B', 'OCGT': '#A9A9A9',
        'nuclear': '#CC4C02', 'large_hydro': '#0868AC', 'biomass': '#238B45',
        'biogas': '#74C476', 'landfill_gas': '#A1D99B', 'sewage_gas': '#C7E9C0',
        'waste_to_energy': '#66C2A4', 'advanced_biofuel': '#41AB5D',
        'CHP': '#B22222', 'gas_engine': '#708090', 'oil': '#4A4A4A',
        'marine': '#4EB3D3', 'geothermal': '#D95F0E',
        'EU_import': '#DC143C', 'H2_turbine': '#FF1493',
    }
    # Merge: carrier_defs takes precedence, then fallback
    for k, v in fallback_colors.items():
        carrier_color_map.setdefault(k, v)
    
    if len(gen_data) > 0 and 'lon_wgs84' in gen_data.columns:
        # Sort carriers: large marker types first (wind_offshore, nuclear), then alphabetical
        carrier_order = gen_data.groupby('carrier')['p_nom'].sum().sort_values(ascending=False).index
        
        for carrier in carrier_order:
            cgen = gen_data[gen_data['carrier'] == carrier]
            count = len(cgen)
            color = carrier_color_map.get(carrier, '#808080')
            
            # Scale marker size by capacity: bigger generators get bigger dots
            sizes = np.sqrt(cgen['p_nom'].clip(1, 5000)) / 3 + 3
            # Larger minimum for big generators (nuclear, offshore wind)
            max_cap = cgen['p_nom'].max()
            if max_cap > 500:
                sizes = sizes * 1.3
            
            fig.add_trace(go.Scattergeo(
                lon=cgen['lon_wgs84'],
                lat=cgen['lat_wgs84'],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.3, color='darkgray'),
                ),
                text=[f"<b>{carrier}</b><br>{row['p_nom']:.0f} MW<br>Bus: {row['bus']}"
                      for _, row in cgen.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name=f'{carrier} ({count})',
                showlegend=True,
            ))
    
    # =====================================================================
    # 4. STORAGE — green diamonds with count
    # =====================================================================
    storage = n.storage_units.copy() if n_storage > 0 else None
    if storage is not None and len(storage) > 0:
        # Resolve storage coordinates
        if 'lon' in storage.columns and 'lat' in storage.columns:
            storage_data = _convert_to_wgs84(storage, x_col='lon', y_col='lat')
        elif 'x' in storage.columns and 'y' in storage.columns:
            storage_data = _convert_to_wgs84(storage, x_col='x', y_col='y')
        elif 'longitude' in storage.columns and 'latitude' in storage.columns:
            storage_data = _convert_to_wgs84(storage, x_col='longitude', y_col='latitude')
        else:
            # Fall back to bus coordinates
            storage_data = storage.copy()
            storage_data['lon_wgs84'] = storage_data['bus'].map(buses['lon_wgs84'])
            storage_data['lat_wgs84'] = storage_data['bus'].map(buses['lat_wgs84'])
            storage_data = storage_data.dropna(subset=['lon_wgs84', 'lat_wgs84'])
        
        if len(storage_data) > 0 and 'lon_wgs84' in storage_data.columns:
            stor_count = len(storage_data)
            fig.add_trace(go.Scattergeo(
                lon=storage_data['lon_wgs84'],
                lat=storage_data['lat_wgs84'],
                mode='markers',
                marker=dict(
                    size=7,
                    color='#228B22',
                    symbol='diamond',
                    opacity=0.8,
                    line=dict(width=0.5, color='darkgray'),
                ),
                text=[f"<b>{row['carrier']}</b><br>Power: {row['p_nom']:.0f} MW<br>Bus: {row['bus']}"
                      for _, row in storage_data.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name=f'Storage ({stor_count})',
                showlegend=True,
            ))
    
    # =====================================================================
    # 5. LAYOUT — GB-focused map with detailed title
    # =====================================================================
    subtitle = f"{n_buses} buses, {n_lines} lines, {n_xfmrs} xfmrs, {n_links} links, {n_gens} generators"
    
    fig.update_layout(
        title=dict(
            text=f"<b>Network Topology Map: {net_name}</b><br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor='center',
        ),
        geo=dict(
            scope='europe',
            projection_type='mercator',
            center=dict(lon=-2.5, lat=54.5),
            lonaxis_range=[-8, 2],
            lataxis_range=[50, 60],
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(220, 235, 255)',
            showcountries=True,
            countrycolor='rgb(200, 200, 200)',
            showcoastlines=True,
            coastlinecolor='rgb(100, 100, 100)',
        ),
        height=950,
        hovermode='closest',
        margin=dict(r=10, t=100, l=10, b=10),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10),
            itemsizing='constant',
        ),
    )
    
    fig.write_html(output_path)
    logger.info(f"Spatial plot saved to {output_path} ({subtitle})")
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

