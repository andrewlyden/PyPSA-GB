"""
Generate plots for 5-hour optimization results.

Visualizes:
- Generation dispatch by carrier
- Interconnector flows (optimized vs historical)
- Storage operation
- Electricity prices
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import pypsa

# Configuration
NETWORK_PATH = "resources/network/Historical_2020_clustered_solved.nc"
RESULTS_PATH = "resources/results"
OUTPUT_DIR = "resources/plots/5hour_results"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════

print("Loading network and results...")
network = pypsa.Network(NETWORK_PATH)

# Get generation data from network (contains actual optimized dispatch)
gen_df = network.generators_t.p

# Load CSV results
storage_df = pd.read_csv(f"{RESULTS_PATH}/Historical_2020_clustered_storage.csv")
flows_df = pd.read_csv(f"{RESULTS_PATH}/Historical_2020_clustered_flows.csv",
                       index_col=0, parse_dates=True)

# Get interconnector flows from network (not in CSV)
ic_flows = network.links_t.p0  # Optimized flows

print(f"Loaded data: {len(gen_df)} snapshots, {len(gen_df.columns)} generators")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1: GENERATION DISPATCH BY CARRIER
# ═════════════════════════════════════════════════════════════════════════════

print("\n1. Creating generation dispatch plot...")

# Aggregate by carrier - use a better approach
carrier_generation = {}

for gen_name in gen_df.columns:
    carrier = network.generators.loc[gen_name, 'carrier']
    if carrier not in carrier_generation:
        carrier_generation[carrier] = gen_df[gen_name].copy()
    else:
        carrier_generation[carrier] = carrier_generation[carrier] + gen_df[gen_name]

# Convert to DataFrame
gen_by_carrier = pd.DataFrame(carrier_generation)

# Sort by total generation
gen_totals = gen_by_carrier.sum().sort_values(ascending=False)
gen_by_carrier = gen_by_carrier[gen_totals.index]

# Create stacked area plot
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors for key carriers
colors = {
    'EU_import': '#FF6B6B',
    'AGR': '#A0522D',
    'CCGT': '#FFA500',
    'OCGT': '#FFD700',
    'biomass': '#228B22',
    'wind_offshore': '#4682B4',
    'wind_onshore': '#87CEEB',
    'solar_pv': '#FFD700',
    'large_hydro': '#1E90FF',
    'small_hydro': '#87CEFA',
    'load_shedding': '#FF0000',
    'PWR': '#9370DB',
    'waste_to_energy': '#8B4513'
}

# Get colors for carriers
carrier_colors = [colors.get(c, '#CCCCCC') for c in gen_by_carrier.columns]

ax.stackplot(gen_by_carrier.index, 
             *[gen_by_carrier[col] for col in gen_by_carrier.columns],
             labels=gen_by_carrier.columns,
             colors=carrier_colors,
             alpha=0.8)

ax.set_xlabel('Time (UTC)', fontsize=12)
ax.set_ylabel('Power (MW)', fontsize=12)
ax.set_title('Generation Dispatch by Carrier Type (2020-06-14)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_generation_dispatch.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/01_generation_dispatch.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2: TOP 10 CARRIERS (CLEANER VIEW)
# ═════════════════════════════════════════════════════════════════════════════

print("2. Creating top 10 carriers plot...")

# Get top 10 by total generation
top10 = gen_totals.head(10)
gen_top10 = gen_by_carrier[top10.index]

fig, ax = plt.subplots(figsize=(14, 8))

carrier_colors_top10 = [colors.get(c, '#CCCCCC') for c in gen_top10.columns]

ax.stackplot(gen_top10.index,
             *[gen_top10[col] for col in gen_top10.columns],
             labels=gen_top10.columns,
             colors=carrier_colors_top10,
             alpha=0.8)

ax.set_xlabel('Time (UTC)', fontsize=12)
ax.set_ylabel('Power (MW)', fontsize=12)
ax.set_title('Top 10 Generation Sources (2020-06-14)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_top10_carriers.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/02_top10_carriers.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3: INTERCONNECTOR FLOWS (OPTIMIZED VS HISTORICAL)
# ═════════════════════════════════════════════════════════════════════════════

print("3. Creating interconnector flows plot...")

# Get interconnector links
ic_links = [link for link in ic_flows.columns if link.startswith('IC_')]

if len(ic_links) > 0:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    # Check if historical flows are available
    has_historical = hasattr(network, 'historical_interconnector_flows')
    
    for idx, link_name in enumerate(ic_links[:9]):  # Max 9 plots
        ax = axes[idx]
        
        # Optimized flows
        optimized = ic_flows[link_name]
        ax.plot(optimized.index, optimized.values, 
                label='Optimized', linewidth=2, color='blue', marker='o')
        
        # Historical flows (if available)
        if has_historical and link_name in network.historical_interconnector_flows:
            historical = network.historical_interconnector_flows[link_name]
            if len(historical) == len(optimized):
                ax.plot(optimized.index, historical.values,
                        label='Historical', linewidth=2, color='red', 
                        linestyle='--', marker='x', alpha=0.7)
        
        ax.set_title(link_name.replace('IC_', ''), fontsize=11, fontweight='bold')
        ax.set_ylabel('Power (MW)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Rotate labels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    # Hide unused subplots
    for idx in range(len(ic_links), 9):
        axes[idx].axis('off')
    
    plt.suptitle('Interconnector Flows: Optimized vs Historical (2020-06-14)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_interconnector_flows.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/03_interconnector_flows.png")
    plt.close()
else:
    print("   No interconnector flow data found")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4: STORAGE OPERATION
# ═════════════════════════════════════════════════════════════════════════════

print("4. Creating storage operation plot...")

# Get storage power data
storage_p_data = network.storage_units_t.p
storage_state = network.storage_units_t.state_of_charge

if not storage_p_data.empty:
    # Aggregate storage by carrier
    storage_carriers = network.storage_units.carrier.unique()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Storage power (charge/discharge)
    for carrier in storage_carriers[:5]:  # Top 5 carriers
        carrier_units = network.storage_units[network.storage_units.carrier == carrier].index
        carrier_total = storage_p_data[carrier_units].sum(axis=1)
        ax1.plot(carrier_total.index, carrier_total.values, 
                label=carrier, linewidth=2, marker='o')
    
    ax1.set_ylabel('Power (MW)\n(+ve = discharge, -ve = charge)', fontsize=11)
    ax1.set_title('Storage Operation (Power)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Plot 2: State of charge
    for carrier in storage_carriers[:5]:
        carrier_units = network.storage_units[network.storage_units.carrier == carrier].index
        carrier_soc = storage_state[carrier_units].sum(axis=1)
        ax2.plot(carrier_soc.index, carrier_soc.values,
                label=carrier, linewidth=2, marker='o')
    
    ax2.set_xlabel('Time (UTC)', fontsize=11)
    ax2.set_ylabel('State of Charge (MWh)', fontsize=11)
    ax2.set_title('Storage State of Charge', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.suptitle('Storage System Operation (2020-06-14)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_storage_operation.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/04_storage_operation.png")
    plt.close()
else:
    print("   No storage operation data found")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 5: ELECTRICITY PRICES (MARGINAL PRICES AT BUSES)
# ═════════════════════════════════════════════════════════════════════════════

print("5. Creating electricity prices plot...")

# Get marginal prices at buses
bus_prices = network.buses_t.marginal_price

if not bus_prices.empty:
    # Calculate system-wide statistics
    price_mean = bus_prices.mean(axis=1)
    price_min = bus_prices.min(axis=1)
    price_max = bus_prices.max(axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot price range
    ax.fill_between(price_mean.index, price_min.values, price_max.values,
                    alpha=0.3, label='Price range (min-max)', color='lightblue')
    ax.plot(price_mean.index, price_mean.values, 
            label='Average price', linewidth=2.5, color='darkblue', marker='o')
    
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.set_ylabel('Electricity Price (£/MWh)', fontsize=12)
    ax.set_title('System Marginal Prices (2020-06-14)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Add statistics annotation
    avg_price = price_mean.mean()
    max_price = price_max.max()
    min_price = price_min.min()
    ax.text(0.02, 0.98, 
            f'Avg: £{avg_price:.2f}/MWh\nMin: £{min_price:.2f}/MWh\nMax: £{max_price:.2f}/MWh',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_electricity_prices.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/05_electricity_prices.png")
    plt.close()
else:
    print("   No price data found")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

print("\n6. Creating summary statistics...")

summary = []
summary.append("═" * 80)
summary.append("5-HOUR OPTIMIZATION RESULTS SUMMARY (2020-06-14 00:00-05:00)")
summary.append("═" * 80)
summary.append("")

# Generation by carrier
summary.append("GENERATION BY CARRIER:")
summary.append("-" * 80)
for carrier, total in gen_totals.head(15).items():
    pct = (total / gen_totals.sum()) * 100
    summary.append(f"  {carrier:30s} : {total:10,.0f} MWh ({pct:5.1f}%)")
summary.append(f"  {'TOTAL':30s} : {gen_totals.sum():10,.0f} MWh")
summary.append("")

# Interconnector summary
if len(ic_links) > 0:
    summary.append("INTERCONNECTOR FLOWS (OPTIMIZED):")
    summary.append("-" * 80)
    for link in ic_links:
        total_flow = ic_flows[link].sum()
        avg_flow = ic_flows[link].mean()
        summary.append(f"  {link:20s} : Total={total_flow:8,.0f} MW, Avg={avg_flow:8,.1f} MW")
    summary.append("")

# System costs
summary.append("SYSTEM COSTS:")
summary.append("-" * 80)
summary.append(f"  Total system cost: £{network.objective:,.2f}")
summary.append(f"  Average ££/hour: £{network.objective/11:,.2f}")
summary.append("")

summary.append("═" * 80)

summary_text = "\n".join(summary)
print(summary_text)

# Save to file
with open(f"{OUTPUT_DIR}/00_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"\n   Saved: {OUTPUT_DIR}/00_summary.txt")

print(f"\n✓ All plots saved to: {OUTPUT_DIR}/")
print("=" * 80)

