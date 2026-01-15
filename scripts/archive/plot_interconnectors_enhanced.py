"""
Enhanced Interconnector Visualization - Shows External Generator Architecture
=================================================================================

This script creates detailed visualizations showing the interconnector implementation
with external generators representing European electricity supply.

Visualizes:
- Interconnector Links (DC cables between GB and Europe)
- External Buses (connection points to European countries)
- External Generators (EU_import carriers with European marginal costs)
- Link marginal costs (should be zero/minimal)
- External generator capacities and costs

Color coding:
- GB buses: Gray
- External buses: Orange
- Interconnector links: Blue (thickness = capacity)
- External generators: Purple (size = marginal cost)

This demonstrates the November 2025 interconnector fix where European supply
is modeled on external generators rather than treating external buses as
infinite free power sources.

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

# Initialize timing
start_time = time.time()

# Snakemake provides inputs/outputs when run under Snakemake
try:
    snakemake  # type: ignore
except NameError:
    raise RuntimeError("This script is intended to be run via Snakemake")

# Use centralized logging
log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "plot_interconnectors_enhanced"
logger = setup_logging(log_path)

network_path = Path(snakemake.input.network)
output_html = Path(snakemake.output.html)
output_html.parent.mkdir(parents=True, exist_ok=True)

# Read scenario from wildcard
scenario = snakemake.wildcards.scenario if hasattr(snakemake, 'wildcards') and hasattr(snakemake.wildcards, 'scenario') else 'Unknown'

logger.info(f"Loading network from: {network_path}")
logger.info("=" * 80)
logger.info("ENHANCED INTERCONNECTOR VISUALIZATION")
logger.info("Showing new architecture with external generators")
logger.info("=" * 80)

# Load network
try:
    n = pypsa.Network(str(network_path))
    logger.info(f"Successfully loaded network: {n.name}")
    log_network_info(n, logger)
except Exception as e:
    logger.error(f"Failed to load network: {e}")
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(f"<html><body><h1>Enhanced Interconnector Plot</h1><p>[ERROR] Failed to load network: {html.escape(str(e))}</p></body></html>")
    sys.exit(1)

# =============================================================================
# ANALYZE INTERCONNECTOR ARCHITECTURE
# =============================================================================

logger.info("\n" + "="*80)
logger.info("ANALYZING INTERCONNECTOR ARCHITECTURE")
logger.info("="*80)

# 1. Find interconnector links
ic_links = n.links[n.links.index.str.startswith('IC_')]
logger.info(f"\n1. Interconnector Links: {len(ic_links)} found")

if len(ic_links) == 0:
    logger.warning("No interconnector links found (no links starting with 'IC_')")
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(
            f"<html><body><h1>Enhanced Interconnector Plot - {scenario}</h1>"
            f"<h3>No interconnector links found in network</h3>"
            f"<p>This network may not have interconnectors added yet.</p>"
            "</body></html>"
        )
    sys.exit(0)

# Log link details
total_capacity = ic_links['p_nom'].sum()
logger.info(f"  Total capacity: {total_capacity:.0f} MW")
logger.info(f"  Average capacity: {ic_links['p_nom'].mean():.0f} MW")

# Check link marginal costs (should be zero/minimal)
high_cost_links = ic_links[ic_links['marginal_cost'] > 5.0]
if len(high_cost_links) > 0:
    logger.warning(f"  ⚠️  {len(high_cost_links)} links have marginal costs >£5/MWh (expected: near-zero)")
    for link_name, link in high_cost_links.iterrows():
        logger.warning(f"    - {link_name}: £{link['marginal_cost']:.2f}/MWh")
else:
    logger.info(f"  ✓ All {len(ic_links)} links have appropriate marginal costs (≤£5/MWh)")
    mean_cost = ic_links['marginal_cost'].mean()
    logger.info(f"    Mean link cost: £{mean_cost:.4f}/MWh")

# 2. Find external buses
external_buses = set(ic_links['bus1'].unique())  # bus1 is the external side
logger.info(f"\n2. External Buses: {len(external_buses)} found")
for bus in sorted(external_buses):
    if bus in n.buses.index:
        country = n.buses.loc[bus].get('country', 'Unknown')
        logger.info(f"  - {bus}: {country}")

# 3. Find external generators
eu_generators = n.generators[n.generators['carrier'] == 'EU_import']
logger.info(f"\n3. External Generators (EU_import): {len(eu_generators)} found")

# Check if this is a historical scenario with fixed flows
has_fixed_flows = False
if hasattr(n, 'links_t') and hasattr(n.links_t, 'p_set'):
    if not n.links_t.p_set.empty:
        # Check if any IC links have p_set values
        ic_link_names = ic_links.index.tolist()
        ic_p_set_cols = [col for col in n.links_t.p_set.columns if col in ic_link_names]
        if ic_p_set_cols:
            has_fixed_flows = True
            logger.info("  ℹ️  Historical scenario detected: Links have FIXED flows (p_set)")
            logger.info("      External generators are NOT needed (flows are pre-determined)")

if len(eu_generators) == 0:
    if has_fixed_flows:
        logger.info("  ✓ No external generators needed for historical scenario")
        logger.info("    Interconnector flows are FIXED from historical data")
    else:
        logger.error("  ✗ CRITICAL: No European supply generators found!")
        logger.error("    External buses may act as unbounded power sources")
        logger.error("    This indicates the interconnector fix has NOT been applied")
else:
    logger.info(f"  ✓ Found {len(eu_generators)} European supply generators")
    
    # Log generator details
    for gen_name, gen in eu_generators.iterrows():
        bus = gen['bus']
        p_nom = gen['p_nom']
        marginal_cost = gen.get('marginal_cost', 0)
        logger.info(f"  - {gen_name}")
        logger.info(f"      Bus: {bus}, Capacity: {p_nom:.0f} MW, Cost: £{marginal_cost:.2f}/MWh")
    
    # Check if all external buses have generators
    buses_with_generators = set(eu_generators['bus'].unique())
    missing_generators = external_buses - buses_with_generators
    
    if missing_generators:
        logger.warning(f"  ⚠️  {len(missing_generators)} external buses WITHOUT generators:")
        for bus in sorted(missing_generators):
            logger.warning(f"    - {bus}")
    else:
        logger.info(f"  ✓ All {len(external_buses)} external buses have generators")

# 4. Summary statistics
logger.info(f"\n4. Architecture Summary:")
logger.info(f"  Interconnector links: {len(ic_links)}")
logger.info(f"  External buses: {len(external_buses)}")
logger.info(f"  External generators: {len(eu_generators)}")
logger.info(f"  Total IC capacity: {total_capacity:.0f} MW")

if len(eu_generators) > 0:
    total_eu_capacity = eu_generators['p_nom'].sum()
    mean_eu_cost = eu_generators['marginal_cost'].mean()
    logger.info(f"  Total EU generator capacity: {total_eu_capacity:.0f} MW")
    logger.info(f"  Mean EU marginal cost: £{mean_eu_cost:.2f}/MWh")
    
    # Check capacity ratio
    if total_eu_capacity > total_capacity * 10:
        logger.info(f"  ✓ EU capacity >> IC capacity (good - non-binding constraint)")
    else:
        logger.warning(f"  ⚠️  EU capacity may be binding constraint")

# =============================================================================
# COORDINATE HARMONIZATION
# =============================================================================

def harmonize_coordinates():
    """Ensure buses have proper coordinates for plotting."""
    buses = n.buses
    
    # Check for WGS84 coordinates
    has_lon = 'lon' in buses.columns and buses['lon'].notna().any()
    has_lat = 'lat' in buses.columns and buses['lat'].notna().any()
    
    if has_lon and has_lat:
        # Check if they're in degree range
        lon_in_range = buses['lon'].between(-180, 180).all()
        lat_in_range = buses['lat'].between(-90, 90).all()
        
        if lon_in_range and lat_in_range:
            logger.info("✓ All buses have valid WGS84 coordinates (lon/lat)")
            # Copy to x/y if missing
            if 'x' not in buses.columns or buses['x'].isna().all():
                buses['x'] = buses['lon']
            if 'y' not in buses.columns or buses['y'].isna().all():
                buses['y'] = buses['lat']
            return True
    
    # Try OSGB36 conversion
    has_x = 'x' in buses.columns and buses['x'].notna().any()
    has_y = 'y' in buses.columns and buses['y'].notna().any()
    
    if has_x and has_y:
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
            logger.info(f"Converted {mask.sum()} buses from OSGB36 to WGS84")
            return True
        except ImportError:
            logger.warning("pyproj not available for coordinate conversion")
            return False
    
    logger.error("No valid coordinates found for plotting")
    return False

has_valid_coords = harmonize_coordinates()

if not has_valid_coords:
    logger.error("Cannot create visualizations without valid coordinates")
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(
            f"<html><body><h1>Enhanced Interconnector Plot - {scenario}</h1>"
            f"<p>[ERROR] No valid bus coordinates found for plotting</p>"
            "</body></html>"
        )
    sys.exit(1)

# =============================================================================
# ENHANCED MATPLOTLIB PLOT
# =============================================================================

plot_dir = output_html.parent / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)
static_plot_path = plot_dir / f"{scenario}_interconnectors_enhanced.png"
static_plot_status = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.lines import Line2D
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Plot GB buses (gray, small)
    gb_buses = n.buses[~n.buses.index.isin(external_buses)]
    if len(gb_buses) > 0 and 'x' in gb_buses.columns and 'y' in gb_buses.columns:
        ax.scatter(gb_buses['x'], gb_buses['y'], 
                  s=5, c='lightgray', alpha=0.3, zorder=1, label='GB Buses')
    
    # Plot external buses (orange, larger)
    ext_bus_data = []
    for bus_name in external_buses:
        if bus_name in n.buses.index:
            bus = n.buses.loc[bus_name]
            if 'x' in bus and 'y' in bus and pd.notna(bus['x']) and pd.notna(bus['y']):
                ext_bus_data.append({
                    'name': bus_name,
                    'x': bus['x'],
                    'y': bus['y'],
                    'country': bus.get('country', 'Unknown')
                })
    
    if ext_bus_data:
        ext_df = pd.DataFrame(ext_bus_data)
        ax.scatter(ext_df['x'], ext_df['y'], 
                  s=200, c='orange', alpha=0.8, zorder=5,
                  edgecolors='darkorange', linewidths=2,
                  label='External Buses (Europe)')
        
        # Label external buses
        for _, row in ext_df.iterrows():
            ax.annotate(f"{row['country']}", 
                       (row['x'], row['y']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.7))
    
    # Plot interconnector links (blue lines, thickness = capacity)
    max_capacity = ic_links['p_nom'].max()
    for link_name, link in ic_links.iterrows():
        bus0_name = link['bus0']
        bus1_name = link['bus1']
        
        if bus0_name in n.buses.index and bus1_name in n.buses.index:
            bus0 = n.buses.loc[bus0_name]
            bus1 = n.buses.loc[bus1_name]
            
            if all(k in bus0 for k in ['x', 'y']) and all(k in bus1 for k in ['x', 'y']):
                x0, y0 = bus0['x'], bus0['y']
                x1, y1 = bus1['x'], bus1['y']
                
                if pd.notna(x0) and pd.notna(y0) and pd.notna(x1) and pd.notna(y1):
                    capacity = link['p_nom']
                    line_width = 1 + (capacity / max_capacity) * 10 if max_capacity > 0 else 3
                    
                    # Color by marginal cost
                    mc = link.get('marginal_cost', 0)
                    if mc <= 1.0:
                        color = 'blue'  # Good - near zero
                        alpha = 0.7
                    else:
                        color = 'red'  # Warning - high cost
                        alpha = 0.9
                    
                    ax.plot([x0, x1], [y0, y1],
                           color=color, linewidth=line_width, alpha=alpha, zorder=3)
                    
                    # Add arrow
                    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                               arrowprops=dict(arrowstyle='->', color=color, 
                                             lw=line_width/2, alpha=alpha),
                               zorder=4)
                    
                    # Label capacity
                    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                    ax.text(mid_x, mid_y, f"{capacity:.0f} MW",
                           fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    # Plot external generators (purple circles, size = marginal cost)
    if len(eu_generators) > 0:
        gen_data = []
        for gen_name, gen in eu_generators.iterrows():
            bus_name = gen['bus']
            if bus_name in n.buses.index:
                bus = n.buses.loc[bus_name]
                if 'x' in bus and 'y' in bus and pd.notna(bus['x']) and pd.notna(bus['y']):
                    gen_data.append({
                        'name': gen_name,
                        'x': bus['x'],
                        'y': bus['y'],
                        'p_nom': gen['p_nom'],
                        'marginal_cost': gen.get('marginal_cost', 0)
                    })
        
        if gen_data:
            gen_df = pd.DataFrame(gen_data)
            
            # Size by marginal cost (larger = more expensive)
            sizes = 100 + gen_df['marginal_cost'] * 10
            
            ax.scatter(gen_df['x'], gen_df['y'],
                      s=sizes, c='purple', alpha=0.6, zorder=6,
                      edgecolors='darkviolet', linewidths=2,
                      label=f'EU Generators ({len(gen_df)})')
            
            # Label generators with cost
            for _, row in gen_df.iterrows():
                ax.annotate(f"£{row['marginal_cost']:.0f}/MWh",
                           (row['x'], row['y']),
                           xytext=(0, -20), textcoords='offset points',
                           fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='purple', alpha=0.7, ec='white'),
                           color='white', fontweight='bold')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
               markersize=5, label='GB Buses', alpha=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=10, label='External Buses', markeredgecolor='darkorange', markeredgewidth=2),
        Line2D([0], [0], color='blue', linewidth=3, label='Interconnector Links (£0/MWh)'),
        Line2D([0], [0], color='red', linewidth=3, label='High Cost Links (>£1/MWh)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
               markersize=12, label='EU Generators (size=cost)', markeredgecolor='darkviolet', markeredgewidth=2),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    ax.set_title(f"Enhanced Interconnector Architecture - {scenario}\n"
                f"Showing External Generators (Nov 2025 Fix)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(static_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Enhanced static plot saved to: {static_plot_path}")
    static_plot_status = "[OK] Enhanced static plot with external generators"
    
except Exception as exc:
    logger.error(f"Failed to create enhanced static plot: {exc}", exc_info=True)
    static_plot_status = f"[ERROR] Static plot failed: {exc}"
    static_plot_path = None

# =============================================================================
# ENHANCED HTML REPORT
# =============================================================================

# Architecture validation status
if has_fixed_flows:
    # Historical scenario - fixed flows don't need external generators
    arch_status = "✅ PASSED"
    arch_color = "green"
    arch_msg = "Historical scenario with FIXED flows (external generators not required)"
elif len(eu_generators) == 0:
    arch_status = "❌ FAILED"
    arch_color = "red"
    arch_msg = "No external generators found - external buses are unbounded sources!"
elif len(missing_generators) > 0:
    arch_status = "⚠️  WARNING"
    arch_color = "orange"
    arch_msg = f"{len(missing_generators)} external buses missing generators"
elif len(high_cost_links) > 0:
    arch_status = "⚠️  WARNING"
    arch_color = "orange"
    arch_msg = f"{len(high_cost_links)} links have high marginal costs (should be ~£0/MWh)"
else:
    arch_status = "✅ PASSED"
    arch_color = "green"
    arch_msg = "Interconnector architecture correctly implemented!"

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Interconnector Visualization - {scenario}</title>
    <meta charset="utf-8"/>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .summary {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }}
        .stat-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .stat-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .architecture-status {{ background-color: {arch_color}; color: white; padding: 20px; border-radius: 5px; margin: 20px 0; text-align: center; }}
        .architecture-status h2 {{ margin: 0; font-size: 28px; }}
        .architecture-status p {{ margin: 10px 0 0 0; font-size: 16px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: white; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .check-pass {{ color: green; font-weight: bold; }}
        .check-warn {{ color: orange; font-weight: bold; }}
        .check-fail {{ color: red; font-weight: bold; }}
        .info-box {{ background-color: #e8f4fd; border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; }}
        .warning-box {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; }}
        .error-box {{ background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; }}
        code {{ background-color: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔌 Enhanced Interconnector Visualization</h1>
        <p>Scenario: <strong>{scenario}</strong></p>
        <p style="font-size: 14px; margin-top: 10px;">Showing new architecture with external generators (November 2025 fix)</p>
    </div>
    
    <div class="architecture-status">
        <h2>{arch_status}</h2>
        <p>{arch_msg}</p>
    </div>
    
    <div class="section">
        <h2>📊 Architecture Summary</h2>
        <div class="summary">
            <div class="stat-box">
                <div class="stat-label">Interconnector Links</div>
                <div class="stat-value">{len(ic_links)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">External Buses</div>
                <div class="stat-value">{len(external_buses)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">EU Generators</div>
                <div class="stat-value">{len(eu_generators)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total IC Capacity</div>
                <div class="stat-value">{total_capacity:.0f} MW</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total EU Capacity</div>
                <div class="stat-value">{eu_generators['p_nom'].sum():.0f} MW</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean EU Cost</div>
                <div class="stat-value">£{eu_generators['marginal_cost'].mean():.0f}/MWh</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>✅ Architecture Validation</h2>
        
        <h3>1. External Generators Check</h3>
"""

if has_fixed_flows:
    html_content += f"""
        <div class="info-box">
            <strong class="check-pass">✓ HISTORICAL SCENARIO:</strong> Links have FIXED flows from historical data<br>
            External generators are <strong>NOT required</strong> when flows are predetermined.<br>
            Flow optimization is disabled (p_set is used instead of p_nom_opt).
        </div>
"""
elif len(eu_generators) == 0:
    html_content += f"""
        <div class="error-box">
            <strong class="check-fail">✗ CRITICAL ERROR:</strong> No external generators found!<br>
            External buses are acting as <strong>unbounded power sources</strong>.<br>
            This will cause optimization issues (unbounded imports, unrealistic costs).
        </div>
"""
else:
    html_content += f"""
        <div class="info-box">
            <strong class="check-pass">✓ PASSED:</strong> Found {len(eu_generators)} European supply generators<br>
            External buses have proper electricity supply representation.
        </div>
"""
    
    if missing_generators:
        html_content += f"""
        <div class="warning-box">
            <strong class="check-warn">⚠ WARNING:</strong> {len(missing_generators)} external buses without generators:<br>
            {', '.join(sorted(missing_generators))}
        </div>
"""

html_content += """
        <h3>2. Link Marginal Cost Check</h3>
"""

if len(high_cost_links) > 0:
    html_content += f"""
        <div class="warning-box">
            <strong class="check-warn">⚠ WARNING:</strong> {len(high_cost_links)} links have marginal costs >£5/MWh<br>
            Expected: near-zero costs (economics should be on external generators)<br>
            <ul>
"""
    for link_name, link in high_cost_links.iterrows():
        html_content += f"                <li><code>{link_name}</code>: £{link['marginal_cost']:.2f}/MWh</li>\n"
    
    html_content += """
            </ul>
        </div>
"""
else:
    mean_cost = ic_links['marginal_cost'].mean()
    html_content += f"""
        <div class="info-box">
            <strong class="check-pass">✓ PASSED:</strong> All links have appropriate marginal costs<br>
            Mean link cost: £{mean_cost:.4f}/MWh (close to zero ✓)
        </div>
"""

html_content += f"""
        <h3>3. Capacity Check</h3>
"""

if len(eu_generators) > 0:
    eu_capacity = eu_generators['p_nom'].sum()
    if eu_capacity > total_capacity * 10:
        html_content += f"""
        <div class="info-box">
            <strong class="check-pass">✓ PASSED:</strong> EU generator capacity ({eu_capacity:.0f} MW) >> interconnector capacity ({total_capacity:.0f} MW)<br>
            EU generators are non-binding constraints (as intended).
        </div>
"""
    else:
        html_content += f"""
        <div class="warning-box">
            <strong class="check-warn">⚠ WARNING:</strong> EU generator capacity ({eu_capacity:.0f} MW) may be too small<br>
            Should be >> interconnector capacity ({total_capacity:.0f} MW) to avoid binding constraint.
        </div>
"""

html_content += """
    </div>
    
    <div class="section">
        <h2>🔗 Interconnector Links Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Link Name</th>
                    <th>From (GB)</th>
                    <th>To (External)</th>
                    <th>Capacity (MW)</th>
                    <th>Marginal Cost</th>
                    <th>Efficiency (%)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

for link_name, link in ic_links.iterrows():
    bus0 = link['bus0']
    bus1 = link['bus1']
    capacity = link['p_nom']
    mc = link.get('marginal_cost', 0)
    eff = link.get('efficiency', 1.0) * 100
    
    if mc <= 1.0:
        status = '<span class="check-pass">✓ Good</span>'
    else:
        status = '<span class="check-warn">⚠ High Cost</span>'
    
    html_content += f"""
                <tr>
                    <td><code>{link_name}</code></td>
                    <td>{bus0}</td>
                    <td>{bus1}</td>
                    <td>{capacity:.0f}</td>
                    <td>£{mc:.4f}/MWh</td>
                    <td>{eff:.1f}</td>
                    <td>{status}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>⚡ External Generators Details</h2>
"""

if len(eu_generators) == 0:
    html_content += """
        <div class="error-box">
            <strong>No external generators found!</strong><br>
            The interconnector fix has NOT been applied to this network.
        </div>
"""
else:
    html_content += """
        <table>
            <thead>
                <tr>
                    <th>Generator Name</th>
                    <th>Bus</th>
                    <th>Country</th>
                    <th>Capacity (MW)</th>
                    <th>Marginal Cost (£/MWh)</th>
                    <th>Carrier</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for gen_name, gen in eu_generators.iterrows():
        bus_name = gen['bus']
        country = n.buses.loc[bus_name].get('country', 'Unknown') if bus_name in n.buses.index else 'Unknown'
        p_nom = gen['p_nom']
        mc = gen.get('marginal_cost', 0)
        carrier = gen.get('carrier', 'Unknown')
        
        html_content += f"""
                <tr>
                    <td><code>{gen_name}</code></td>
                    <td>{bus_name}</td>
                    <td>{country}</td>
                    <td>{p_nom:.0f}</td>
                    <td>£{mc:.2f}</td>
                    <td>{carrier}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
"""

html_content += """
    </div>
    
    <div class="section">
        <h2>📖 Architecture Explanation</h2>
        <div class="info-box">
            <h3>New Architecture (November 2025 Fix):</h3>
            <p><strong>Before:</strong> External buses had no generators → infinite free power → unbounded optimization</p>
            <p><strong>After:</strong> Each external bus has a large generator (100 GW) with European marginal cost (£40-60/MWh)</p>
            
            <h4>Components:</h4>
            <ul>
                <li><strong>Interconnector Links (DC):</strong> Physical cables between GB and Europe
                    <ul>
                        <li>Marginal cost: £0/MWh (cost is on external generator)</li>
                        <li>Efficiency: ~97.5% (transmission losses)</li>
                        <li>Capacity: Real interconnector capacity limits</li>
                    </ul>
                </li>
                <li><strong>External Buses:</strong> Connection points to European countries
                    <ul>
                        <li>Located at international coordinates</li>
                        <li>bus1 side of interconnector links</li>
                    </ul>
                </li>
                <li><strong>External Generators (EU_import):</strong> Represent European electricity supply
                    <ul>
                        <li>Capacity: 100 GW (large, non-binding)</li>
                        <li>Marginal cost: £40-60/MWh (from European generation mix data)</li>
                        <li>Carrier: EU_import (purple in visualizations)</li>
                    </ul>
                </li>
            </ul>
            
            <h4>Why This Works:</h4>
            <p>The optimizer balances GB generation vs. European imports based on <strong>economic dispatch</strong>:</p>
            <ul>
                <li>If GB CCGT costs £60/MWh and EU costs £50/MWh → imports from EU</li>
                <li>If GB wind costs £0/MWh → uses GB wind instead of imports</li>
                <li>Link efficiency (97.5%) represents transmission losses</li>
                <li>Link capacity limits physical flow (e.g., 2 GW IFA)</li>
            </ul>
        </div>
    </div>
"""

# Embed static plot if available
if static_plot_path and static_plot_path.exists():
    try:
        static_rel = static_plot_path.relative_to(output_html.parent)
    except ValueError:
        static_rel = static_plot_path.name
    
    html_content += f"""
    <div class="section">
        <h2>🗺️ Enhanced Interconnector Map</h2>
        <p><strong>Color Legend:</strong></p>
        <ul>
            <li>Gray dots: GB network buses</li>
            <li>Orange circles: External buses (European connection points)</li>
            <li>Blue lines: Interconnector links (thickness = capacity, cost ≤£1/MWh)</li>
            <li>Red lines: High-cost links (cost >£1/MWh - warning!)</li>
            <li>Purple circles: External generators (size = marginal cost)</li>
        </ul>
        <img src="{static_rel.as_posix()}" alt="Enhanced interconnector map" style="max-width:100%; height:auto; border: 1px solid #ddd;"/>
    </div>
"""

html_content += f"""
    
    <div class="section info" style="text-align: center; color: #999; font-size: 12px;">
        <p>Enhanced Interconnector Visualization | November 2025 Interconnector Fix</p>
        <p>Generated by plot_interconnectors_enhanced.py</p>
        <p>Execution time: {time.time() - start_time:.2f} seconds</p>
    </div>
</body>
</html>
"""

# Write HTML report
try:
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"✓ Enhanced interconnector report saved to: {output_html}")
except Exception as e:
    logger.error(f"Failed to write HTML output: {e}")
    sys.exit(1)

# Log execution summary
log_execution_summary(
    logger,
    "plot_interconnectors_enhanced",
    start_time,
    inputs={'network': str(network_path)},
    outputs={'report': str(output_html)},
    context={
        'scenario': scenario,
        'interconnector_links': len(ic_links),
        'external_buses': len(external_buses),
        'eu_generators': len(eu_generators),
        'architecture_status': arch_status,
        'total_capacity_mw': float(total_capacity)
    }
)

logger.info("="*80)
logger.info(f"Architecture validation: {arch_status}")
logger.info("Enhanced interconnector visualization completed successfully!")
logger.info("="*80)

