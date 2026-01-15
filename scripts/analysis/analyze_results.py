"""
Analyze Optimization Results - Post-processing and Visualization

This script analyzes solved network results and generates:
- Interactive HTML dashboard with Plotly charts
- Summary statistics (JSON)
- Time series visualizations
- Cost breakdowns
"""

import pypsa
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scripts.utilities.logging_config import setup_logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_results_dashboard(network, scenario_config, output_path, logger):
    """
    Create interactive HTML dashboard with optimization results.
    
    Parameters
    ----------
    network : pypsa.Network
        Solved network
    scenario_config : dict
        Scenario configuration
    output_path : str or Path
        Path to write HTML dashboard
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating comprehensive results dashboard...")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 1. GENERATION MIX - STACKED AREA CHART
    # ══════════════════════════════════════════════════════════════════════════
    
    logger.info("Creating generation mix plot...")
    
    # Get generation time series by carrier
    gen_ts = network.generators_t.p.copy()
    
    # Map generators to carriers
    gen_by_carrier = {}
    for gen in gen_ts.columns:
        if gen in network.generators.index:
            carrier = network.generators.loc[gen, 'carrier']
            if carrier not in gen_by_carrier:
                gen_by_carrier[carrier] = []
            gen_by_carrier[carrier].append(gen)
    
    # Aggregate by carrier
    carrier_ts = pd.DataFrame(index=gen_ts.index)
    for carrier, gens in gen_by_carrier.items():
        carrier_ts[carrier] = gen_ts[gens].sum(axis=1)
    
    # Sort carriers by total generation (descending)
    carrier_totals = carrier_ts.sum().sort_values(ascending=False)
    carrier_order = carrier_totals.index.tolist()
    
    # Color mapping for carriers
    carrier_colors = {
        'wind_onshore': '#74add1',
        'wind_offshore': '#4575b4',
        'solar_pv': '#fee090',
        'Solar': '#fee090',
        'nuclear': '#d73027',
        'Nuclear': '#d73027',
        'AGR': '#d73027',
        'PWR': '#fc8d59',
        'CCGT': '#fdae61',
        'OCGT': '#fee08b',
        'coal': '#636363',
        'Coal': '#636363',
        'Conventional steam': '#636363',
        'biomass': '#2ca25f',
        'Biomass': '#2ca25f',
        'waste_to_energy': '#43a2ca',
        'large_hydro': '#74c476',
        'small_hydro': '#a1d99b',
        'EU_import': '#9e9ac8',
        'load_shedding': '#e31a1c',
        'biogas': '#66c2a4',
        'landfill_gas': '#8da0cb',
        'sewage_gas': '#b3cde3',
        'advanced_biofuel': '#238b45',
        'geothermal': '#d94801',
        'tidal_stream': '#6baed6',
        'shoreline_wave': '#9ecae1'
    }
    
    # Create stacked area chart
    fig_gen = go.Figure()
    
    for carrier in reversed(carrier_order):  # Reverse to stack bottom-up
        if carrier_ts[carrier].sum() > 0:  # Only plot if has generation
            fig_gen.add_trace(go.Scatter(
                x=carrier_ts.index,
                y=carrier_ts[carrier],
                name=carrier,
                mode='lines',
                stackgroup='one',
                fillcolor=carrier_colors.get(carrier, None),
                line=dict(width=0.5, color=carrier_colors.get(carrier, None)),
                hovertemplate='%{y:.0f} MW<extra></extra>'
            ))
    
    fig_gen.update_layout(
        title='Generation Mix Over Time',
        xaxis_title='Time',
        yaxis_title='Generation (MW)',
        hovermode='x unified',
        height=500,
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 2. GENERATION PIE CHART - TOTAL ENERGY BY CARRIER
    # ══════════════════════════════════════════════════════════════════════════
    
    logger.info("Creating generation pie chart...")
    
    # Calculate total energy (MWh) by carrier
    gen_totals = carrier_ts.sum() * 0.5  # Convert to MWh (30-min resolution)
    gen_totals = gen_totals[gen_totals > 0].sort_values(ascending=False)
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=gen_totals.index,
        values=gen_totals.values,
        marker=dict(colors=[carrier_colors.get(c, None) for c in gen_totals.index]),
        hovertemplate='%{label}<br>%{value:.0f} MWh<br>%{percent}<extra></extra>'
    )])
    
    fig_pie.update_layout(
        title=f'Total Energy Generation: {gen_totals.sum():,.0f} MWh',
        height=500
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 3. STORAGE STATE OF CHARGE
    # ══════════════════════════════════════════════════════════════════════════
    
    logger.info("Creating storage state of charge plot...")
    
    fig_storage = go.Figure()
    
    if len(network.storage_units) > 0 and hasattr(network, 'storage_units_t'):
        # Get storage SOC
        if 'state_of_charge' in network.storage_units_t:
            soc = network.storage_units_t.state_of_charge.copy()
            
            # Group by carrier
            storage_by_carrier = {}
            for storage_unit in soc.columns:
                if storage_unit in network.storage_units.index:
                    carrier = network.storage_units.loc[storage_unit, 'carrier']
                    if carrier not in storage_by_carrier:
                        storage_by_carrier[carrier] = []
                    storage_by_carrier[carrier].append(storage_unit)
            
            # Plot SOC for each carrier
            for carrier, units in storage_by_carrier.items():
                total_soc = soc[units].sum(axis=1)
                fig_storage.add_trace(go.Scatter(
                    x=soc.index,
                    y=total_soc,
                    name=carrier,
                    mode='lines',
                    hovertemplate='%{y:.0f} MWh<extra></extra>'
                ))
    
    fig_storage.update_layout(
        title='Storage State of Charge',
        xaxis_title='Time',
        yaxis_title='State of Charge (MWh)',
        hovermode='x unified',
        height=400
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 4. INTERCONNECTOR FLOWS
    # ══════════════════════════════════════════════════════════════════════════
    
    logger.info("Creating interconnector flows plot...")
    
    fig_ic = go.Figure()
    
    if len(network.links) > 0 and 'p1' in network.links_t:
        ic_links = [l for l in network.links.index if l.startswith('IC_')]
        
        if len(ic_links) > 0:
            flows = network.links_t.p1[ic_links].copy()
            
            for link in flows.columns:
                # Get cleaner name
                link_name = link.replace('IC_', '').replace('_', ' ')
                
                fig_ic.add_trace(go.Scatter(
                    x=flows.index,
                    y=flows[link],
                    name=link_name,
                    mode='lines',
                    hovertemplate='%{y:.0f} MW<extra></extra>'
                ))
    
    fig_ic.update_layout(
        title='Interconnector Flows (Positive = Import)',
        xaxis_title='Time',
        yaxis_title='Power Flow (MW)',
        hovermode='x unified',
        height=400
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 5. SYSTEM DEMAND VS SUPPLY
    # ══════════════════════════════════════════════════════════════════════════
    
    logger.info("Creating demand vs supply plot...")
    
    # Total generation
    total_gen = carrier_ts.sum(axis=1)
    
    # Total demand
    if hasattr(network, 'loads_t') and len(network.loads_t.p) > 0:
        total_demand = network.loads_t.p.sum(axis=1)
    else:
        total_demand = pd.Series(0, index=total_gen.index)
    
    fig_balance = go.Figure()
    
    fig_balance.add_trace(go.Scatter(
        x=total_gen.index,
        y=total_demand,
        name='Demand',
        mode='lines',
        line=dict(color='red', width=2),
        hovertemplate='%{y:.0f} MW<extra></extra>'
    ))
    
    fig_balance.add_trace(go.Scatter(
        x=total_gen.index,
        y=total_gen,
        name='Generation',
        mode='lines',
        line=dict(color='blue', width=2),
        hovertemplate='%{y:.0f} MW<extra></extra>'
    ))
    
    fig_balance.update_layout(
        title='System Balance: Demand vs Generation',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        hovermode='x unified',
        height=400
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 6. COST BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    
    logger.info("Creating cost breakdown...")
    
    # Calculate costs by carrier
    costs_by_carrier = {}
    for gen in network.generators.index:
        carrier = network.generators.loc[gen, 'carrier']
        mc = network.generators.loc[gen, 'marginal_cost']
        
        if gen in gen_ts.columns:
            output = gen_ts[gen].sum() * 0.5  # MWh
            cost = output * mc
            
            if carrier not in costs_by_carrier:
                costs_by_carrier[carrier] = 0
            costs_by_carrier[carrier] += cost
    
    costs_df = pd.Series(costs_by_carrier).sort_values(ascending=False)
    costs_df = costs_df[costs_df > 0]
    
    fig_costs = go.Figure(data=[go.Bar(
        x=costs_df.index,
        y=costs_df.values,
        marker=dict(color=[carrier_colors.get(c, '#999') for c in costs_df.index]),
        hovertemplate='%{x}<br>£%{y:,.0f}<extra></extra>'
    )])
    
    fig_costs.update_layout(
        title=f'Generation Costs by Carrier (Total: £{costs_df.sum():,.0f})',
        xaxis_title='Carrier',
        yaxis_title='Cost (£)',
        height=400
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # ASSEMBLE HTML DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    
    logger.info("Assembling HTML dashboard...")
    
    scenario_id = scenario_config.get('scenario_id', 'Unknown')
    total_cost = float(getattr(network, 'objective', 0))
    
    # Convert Plotly figures to HTML divs
    gen_mix_html = fig_gen.to_html(full_html=False, include_plotlyjs=False)
    gen_pie_html = fig_pie.to_html(full_html=False, include_plotlyjs=False)
    balance_html = fig_balance.to_html(full_html=False, include_plotlyjs=False)
    storage_html = fig_storage.to_html(full_html=False, include_plotlyjs=False)
    ic_html = fig_ic.to_html(full_html=False, include_plotlyjs=False)
    costs_html = fig_costs.to_html(full_html=False, include_plotlyjs=False)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PyPSA-GB Results: {scenario_id}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        h2 {{ margin: 5px 0 0 0; font-weight: 300; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .plot-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 50px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PyPSA-GB Optimization Results</h1>
        <h2>Scenario: {scenario_id}</h2>
    </div>
    
    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total System Cost</h3>
            <div class="value">£{total_cost:,.0f}</div>
        </div>
        <div class="summary-card">
            <h3>Network Size</h3>
            <div class="value">{len(network.buses):,} buses</div>
        </div>
        <div class="summary-card">
            <h3>Generators</h3>
            <div class="value">{len(network.generators):,}</div>
        </div>
        <div class="summary-card">
            <h3>Time Period</h3>
            <div class="value">{len(network.snapshots)} hours</div>
        </div>
        <div class="summary-card">
            <h3>Total Generation</h3>
            <div class="value">{gen_totals.sum():,.0f} MWh</div>
        </div>
        <div class="summary-card">
            <h3>Storage Units</h3>
            <div class="value">{len(network.storage_units)}</div>
        </div>
    </div>
    
    <div class="plot-container">
        {gen_mix_html}
    </div>
    
    <div class="plot-container">
        {gen_pie_html}
    </div>
    
    <div class="plot-container">
        {balance_html}
    </div>
    
    <div class="plot-container">
        {storage_html}
    </div>
    
    <div class="plot-container">
        {ic_html}
    </div>
    
    <div class="plot-container">
        {costs_html}
    </div>
    
    <div class="footer">
        <p>Generated by PyPSA-GB Analysis Pipeline</p>
        <p>Scenario: {scenario_id} | Network Model: {scenario_config.get('network_model', 'Unknown')}</p>
    </div>
</body>
</html>
    """
    
    Path(output_path).write_text(html_content, encoding='utf-8')
    logger.info(f"✓ Interactive dashboard written to {output_path}")
    logger.info(f"  Generated 6 interactive plots")


def create_summary_json(network, scenario_config, output_path, logger):
    """
    Create JSON summary of optimization results.
    
    Parameters
    ----------
    network : pypsa.Network
        Solved network
    scenario_config : dict
        Scenario configuration
    output_path : str or Path
        Path to write JSON summary
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating results summary JSON...")
    
    summary = {
        'scenario_id': scenario_config.get('scenario_id', 'Unknown'),
        'network_model': scenario_config.get('network_model', 'Unknown'),
        'clustered': scenario_config.get('clustering', {}).get('enabled', False),
        'buses': len(network.buses),
        'generators': len(network.generators),
        'storage_units': len(network.storage_units),
        'snapshots': len(network.snapshots),
        'total_cost': float(getattr(network, 'objective', None) or 0),
    }
    
    # Add generation summary if available
    if len(network.generators_t.p) > 0:
        gen_by_carrier = {}
        for gen in network.generators.index:
            carrier = network.generators.loc[gen, 'carrier']
            if gen in network.generators_t.p.columns:
                output = float(network.generators_t.p[gen].sum())
                if carrier not in gen_by_carrier:
                    gen_by_carrier[carrier] = 0
                gen_by_carrier[carrier] += output
        summary['generation_by_carrier_MWh'] = gen_by_carrier
    
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info(f"Summary JSON written to {output_path}")


if __name__ == "__main__":
    # Set up logging - use Snakemake log path if available
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "analyze_results"
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("ANALYZING OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    
    try:
        # Load solved network
        input_path = snakemake.input.network
        logger.info(f"Loading solved network from: {input_path}")
        network = pypsa.Network(input_path)
        
        # Get scenario config
        scenario_config = snakemake.params.scenario_config
        scenario_id = scenario_config.get('scenario_id', snakemake.wildcards.scenario)
        
        logger.info(f"Scenario: {scenario_id}")
        
        # Create dashboard
        dashboard_path = snakemake.output.dashboard
        create_results_dashboard(network, scenario_config, dashboard_path, logger)
        
        # Create summary JSON
        summary_path = snakemake.output.summary_json
        create_summary_json(network, scenario_config, summary_path, logger)
        
        logger.info("=" * 80)
        logger.info("RESULTS ANALYSIS COMPLETED")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"FATAL ERROR in results analysis: {e}", exc_info=True)
        raise

