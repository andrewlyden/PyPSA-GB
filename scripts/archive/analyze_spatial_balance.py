#!/usr/bin/env python3
"""
Analyze spatial distribution of generation, demand, storage, and interconnectors
to understand why transmission overloads occur.
"""

import pypsa
import pandas as pd
import numpy as np
import networkx as nx

print("Loading network...")
n = pypsa.Network('resources/network/Historical_2020_ETYS.nc')

print(f"Network: {len(n.buses)} buses, {len(n.generators)} generators")
print()

# =============================================================================
# 1. BUILD NETWORK GRAPH TO UNDERSTAND TOPOLOGY
# =============================================================================
print("=" * 70)
print("1. NETWORK TOPOLOGY ANALYSIS")
print("=" * 70)

G = nx.Graph()
G.add_nodes_from(n.buses.index)
for _, line in n.lines.iterrows():
    G.add_edge(line.bus0, line.bus1, weight=1/line.s_nom, name=line.name, type='line', s_nom=line.s_nom)
for _, t in n.transformers.iterrows():
    G.add_edge(t.bus0, t.bus1, weight=1/t.s_nom, name=t.name, type='transformer', s_nom=t.s_nom)

# Identify bottleneck lines (previously found)
overloaded = [
    ('565', 'DUNB1R', 'INWI1R', 115),
    ('675', 'INWI1R', 'TORN1-', 115),
    ('1423', 'HUMO21', 'HEDO21', 160),
    ('265', 'INRU1R', 'PEHE1-', 126),
]

print("\nOverloaded corridors identified:")
for name, bus0, bus1, s_nom in overloaded:
    print(f"  Line {name}: {bus0} <-> {bus1} ({s_nom} MW)")

# =============================================================================
# 2. ANALYZE WHAT'S CONNECTED BEYOND EACH BOTTLENECK
# =============================================================================
print()
print("=" * 70)
print("2. GENERATION AND DEMAND BEYOND BOTTLENECKS")
print("=" * 70)

def get_buses_beyond_bottleneck(G, bottleneck_bus0, bottleneck_bus1):
    """Get all buses reachable from bus1 without going through bus0."""
    # Temporarily remove the edge
    G_temp = G.copy()
    if G_temp.has_edge(bottleneck_bus0, bottleneck_bus1):
        G_temp.remove_edge(bottleneck_bus0, bottleneck_bus1)
    
    # Find buses reachable from bus1
    try:
        reachable = nx.node_connected_component(G_temp, bottleneck_bus1)
        return reachable
    except:
        return set()

for name, bus0, bus1, s_nom in overloaded[:4]:
    print(f"\n--- Bottleneck: {bus0} <-> {bus1} ({s_nom} MW) ---")
    
    # Get buses on the "far" side (bus1 side, away from main grid)
    far_side_buses = get_buses_beyond_bottleneck(G, bus0, bus1)
    
    if len(far_side_buses) == 0:
        print("  Could not determine far side buses")
        continue
    
    print(f"  Buses beyond {bus1}: {len(far_side_buses)}")
    
    # Generators on far side
    far_gens = n.generators[n.generators.bus.isin(far_side_buses)]
    far_gen_capacity = far_gens.p_nom.sum()
    
    # Exclude load shedding
    far_gens_real = far_gens[far_gens.carrier != 'load_shedding']
    far_gen_real_capacity = far_gens_real.p_nom.sum()
    
    print(f"  Generation capacity (excl load shedding): {far_gen_real_capacity:.0f} MW")
    
    # Breakdown by carrier
    carrier_breakdown = far_gens_real.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
    for carrier, cap in carrier_breakdown.head(5).items():
        print(f"    {carrier}: {cap:.0f} MW")
    
    # Loads on far side (at snapshot 56 - the problematic hour)
    far_loads = n.loads[n.loads.bus.isin(far_side_buses)]
    far_load_power = 0
    snapshot = n.snapshots[56]
    for load in far_loads.index:
        if load in n.loads_t.p_set.columns:
            far_load_power += n.loads_t.p_set.loc[snapshot, load]
        else:
            far_load_power += n.loads.loc[load, 'p_set']
    
    print(f"  Demand at hour 57: {far_load_power:.0f} MW")
    
    # Net flow needed
    net_flow = far_load_power - far_gen_real_capacity
    print(f"  Net flow needed (demand - gen): {net_flow:.0f} MW")
    print(f"  Line capacity: {s_nom} MW")
    if abs(net_flow) > s_nom:
        print(f"  ⚠️  OVERLOAD: {abs(net_flow)/s_nom:.0f}x capacity needed!")

# =============================================================================
# 3. ANALYZE REGIONAL IMBALANCES
# =============================================================================
print()
print("=" * 70)
print("3. REGIONAL GENERATION vs DEMAND IMBALANCE")
print("=" * 70)

# Group buses by rough geographic regions (using lat/lon)
if 'lat' in n.buses.columns and 'lon' in n.buses.columns:
    n.buses['region'] = pd.cut(n.buses['lat'], bins=5, labels=['South', 'S-Central', 'Central', 'N-Central', 'North'])
    
    snapshot = n.snapshots[56]
    
    for region in ['South', 'S-Central', 'Central', 'N-Central', 'North']:
        region_buses = n.buses[n.buses['region'] == region].index
        
        # Generation (excluding load shedding)
        region_gens = n.generators[(n.generators.bus.isin(region_buses)) & 
                                   (n.generators.carrier != 'load_shedding')]
        region_gen_capacity = region_gens.p_nom.sum()
        
        # Demand
        region_loads = n.loads[n.loads.bus.isin(region_buses)]
        region_demand = 0
        for load in region_loads.index:
            if load in n.loads_t.p_set.columns:
                region_demand += n.loads_t.p_set.loc[snapshot, load]
        
        balance = region_gen_capacity - region_demand
        print(f"{region:10s}: Gen={region_gen_capacity:8.0f} MW, Demand={region_demand:8.0f} MW, Balance={balance:+8.0f} MW")

# =============================================================================
# 4. CHECK SCOTTISH GENERATION vs DEMAND (common bottleneck)
# =============================================================================
print()
print("=" * 70)
print("4. SCOTLAND vs ENGLAND ANALYSIS")
print("=" * 70)

# Scottish buses typically have lat > 55
scotland_buses = n.buses[n.buses['lat'] > 55.5].index if 'lat' in n.buses.columns else []
england_buses = n.buses[n.buses['lat'] <= 55.5].index if 'lat' in n.buses.columns else []

print(f"Scotland buses: {len(scotland_buses)}")
print(f"England/Wales buses: {len(england_buses)}")

# Scottish generation
scot_gens = n.generators[(n.generators.bus.isin(scotland_buses)) & 
                         (n.generators.carrier != 'load_shedding')]
scot_gen_capacity = scot_gens.p_nom.sum()

# Scottish demand at hour 57
snapshot = n.snapshots[56]
scot_loads = n.loads[n.loads.bus.isin(scotland_buses)]
scot_demand = 0
for load in scot_loads.index:
    if load in n.loads_t.p_set.columns:
        scot_demand += n.loads_t.p_set.loc[snapshot, load]

print(f"\nScotland at hour 57:")
print(f"  Generation capacity: {scot_gen_capacity:.0f} MW")
print(f"  Demand: {scot_demand:.0f} MW")
print(f"  Net export to England: {scot_gen_capacity - scot_demand:.0f} MW")

# Breakdown of Scottish generation
print(f"\nScottish generation by carrier:")
scot_carrier = scot_gens.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
for carrier, cap in scot_carrier.head(8).items():
    print(f"  {carrier}: {cap:.0f} MW")

# =============================================================================
# 5. CHECK THE BOUNDARY LINES BETWEEN SCOTLAND AND ENGLAND
# =============================================================================
print()
print("=" * 70)
print("5. SCOTLAND-ENGLAND BOUNDARY TRANSFER CAPACITY")
print("=" * 70)

# Find lines connecting Scottish to English buses
boundary_lines = n.lines[
    ((n.lines.bus0.isin(scotland_buses)) & (n.lines.bus1.isin(england_buses))) |
    ((n.lines.bus1.isin(scotland_buses)) & (n.lines.bus0.isin(england_buses)))
]

print(f"Number of boundary lines: {len(boundary_lines)}")
print(f"Total boundary transfer capacity: {boundary_lines.s_nom.sum():.0f} MW")

print("\nBoundary lines:")
for idx, line in boundary_lines.iterrows():
    print(f"  {idx}: {line.bus0} <-> {line.bus1}, s_nom={line.s_nom:.0f} MW")

# Also check transformers
boundary_trafos = n.transformers[
    ((n.transformers.bus0.isin(scotland_buses)) & (n.transformers.bus1.isin(england_buses))) |
    ((n.transformers.bus1.isin(scotland_buses)) & (n.transformers.bus0.isin(england_buses)))
]
print(f"\nBoundary transformers: {len(boundary_trafos)}")
if len(boundary_trafos) > 0:
    print(f"Total transformer capacity: {boundary_trafos.s_nom.sum():.0f} MW")

# =============================================================================
# 6. SUMMARY - ROOT CAUSE ANALYSIS
# =============================================================================
print()
print("=" * 70)
print("6. ROOT CAUSE SUMMARY")
print("=" * 70)

scot_export = scot_gen_capacity - scot_demand
boundary_capacity = boundary_lines.s_nom.sum() + (boundary_trafos.s_nom.sum() if len(boundary_trafos) > 0 else 0)

print(f"Scottish net export potential: {scot_export:.0f} MW")
print(f"Scotland-England boundary capacity: {boundary_capacity:.0f} MW")

if scot_export > boundary_capacity:
    print(f"\n⚠️  CRITICAL: Scottish export ({scot_export:.0f} MW) exceeds")
    print(f"   boundary transfer capacity ({boundary_capacity:.0f} MW)")
    print(f"   Overload factor: {scot_export/boundary_capacity:.1f}x")
    print()
    print("RECOMMENDATIONS:")
    print("1. Reduce Scottish generation capacity in the model")
    print("2. Increase Scottish demand allocation")
    print("3. Add more boundary transfer capacity")
    print("4. Re-map some Scottish generators to English buses")

print("\nDone.")

