#!/usr/bin/env python3
"""
Analyze Generator Mapping Across Different Network Models

This script compares how generators are mapped to buses across:
- ETYS (full 400+ bus transmission network)
- Reduced (medium complexity)  
- Zonal (17 zones)

Key checks:
1. Voltage level distribution of generators
2. Large generators at appropriate buses
3. Spatial distribution by carrier type
4. Comparison across network types
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def load_network(path):
    """Load network from pickle or netcdf."""
    if path.suffix == '.pkl':
        return pickle.load(open(path, 'rb'))
    else:
        import pypsa
        return pypsa.Network(str(path))


def analyze_network_generators(n, network_name):
    """Analyze generator mapping for a network."""
    print(f"\n{'='*60}")
    print(f"NETWORK: {network_name}")
    print(f"{'='*60}")
    print(f"Buses: {len(n.buses)}, Generators: {len(n.generators)}")
    
    gens = n.generators.copy()
    if gens.empty:
        print("No generators in network!")
        return None
    
    # Add bus info
    gens['bus_v_nom'] = gens['bus'].map(n.buses['v_nom'])
    gens['bus_x'] = gens['bus'].map(n.buses['x'])
    gens['bus_y'] = gens['bus'].map(n.buses['y'])
    
    # 1. Bus voltage distribution  
    print("\n--- Generator Capacity by Bus Voltage ---")
    v_summary = gens.groupby('bus_v_nom').agg({
        'p_nom': ['count', 'sum']
    }).round(0)
    v_summary.columns = ['count', 'capacity_MW']
    print(v_summary.sort_index(ascending=False).to_string())
    
    # 2. Carrier breakdown
    print("\n--- Top Carriers by Capacity ---")
    carrier_summary = gens.groupby('carrier').agg({
        'p_nom': ['count', 'sum']
    }).round(0)
    carrier_summary.columns = ['count', 'capacity_MW']
    print(carrier_summary.sort_values('capacity_MW', ascending=False).head(10).to_string())
    
    # 3. Large generators check
    print("\n--- Large Generators (>500 MW) ---")
    large = gens[gens['p_nom'] > 500].sort_values('p_nom', ascending=False)
    for idx, gen in large.head(15).iterrows():
        v = gen['bus_v_nom']
        status = 'OK' if v >= 275 else 'LOW V!'
        print(f"  {idx[:30]:30s} {gen['carrier']:12s} {gen['p_nom']:7.0f} MW at {gen['bus']:12s} ({v:.0f}kV) {status}")
    
    # 4. Check for potential issues
    print("\n--- Potential Issues ---")
    
    # Large gens at low voltage
    large_low_v = gens[(gens['p_nom'] > 100) & (gens['bus_v_nom'] < 275)]
    large_low_v = large_low_v[~large_low_v['carrier'].isin(['load_shedding', 'solar_pv', 'wind_onshore'])]
    if len(large_low_v) > 0:
        print(f"  ! {len(large_low_v)} generators >100MW at <275kV (excluding renewables/load_shed)")
        for idx, gen in large_low_v.head(5).iterrows():
            print(f"    {idx[:25]:25s} {gen['carrier']:12s} {gen['p_nom']:.0f} MW at {gen['bus']} ({gen['bus_v_nom']:.0f}kV)")
    else:
        print("  ✓ No large thermal generators at inappropriate voltage levels")
    
    # Missing bus mappings
    missing_bus = gens[gens['bus'].isna() | ~gens['bus'].isin(n.buses.index)]
    if len(missing_bus) > 0:
        print(f"  ! {len(missing_bus)} generators with invalid/missing bus")
        print(f"    Total capacity affected: {missing_bus['p_nom'].sum():.0f} MW")
    else:
        print("  ✓ All generators have valid bus assignments")
    
    # Generators per bus concentration
    gens_per_bus = gens.groupby('bus')['p_nom'].agg(['count', 'sum'])
    heavy_buses = gens_per_bus[gens_per_bus['sum'] > 5000]
    if len(heavy_buses) > 0:
        print(f"\n  High capacity buses (>5GW):")
        for bus, row in heavy_buses.sort_values('sum', ascending=False).head(5).iterrows():
            v = n.buses.loc[bus, 'v_nom'] if bus in n.buses.index else 0
            print(f"    {bus:15s}: {row['count']:.0f} generators, {row['sum']:.0f} MW ({v:.0f}kV)")
    
    return gens


def compare_networks():
    """Compare generator mapping across different network types."""
    base_path = Path("resources/network")
    
    networks = {
        'ETYS': 'Historical_2019_ETYS_network_demand_renewables_thermal_generators.pkl',
        'Reduced': 'Historical_2019_reduced_network_demand_renewables_thermal_generators.pkl',
        'Zonal': 'Historical_2019_zonal_network_demand_renewables_thermal_generators.pkl',
    }
    
    all_gens = {}
    
    for name, filename in networks.items():
        path = base_path / filename
        if path.exists():
            n = load_network(path)
            all_gens[name] = analyze_network_generators(n, name)
        else:
            print(f"\n⚠ Network not found: {filename}")
    
    # Comparison summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    for name, gens in all_gens.items():
        if gens is not None:
            total_cap = gens['p_nom'].sum() / 1000  # GW
            n_gens = len(gens[gens['carrier'] != 'load_shedding'])
            n_buses = gens['bus'].nunique()
            print(f"  {name:12s}: {n_gens:4d} generators, {n_buses:4d} buses used, {total_cap:.1f} GW total")


def check_specific_generators(network_path, gen_names):
    """Check mapping of specific generators across networks."""
    n = load_network(Path(network_path))
    
    print(f"\n--- Checking Specific Generators ---")
    for name in gen_names:
        matches = n.generators[n.generators.index.str.contains(name, case=False)]
        if len(matches) > 0:
            for idx, gen in matches.iterrows():
                bus = gen['bus']
                v = n.buses.loc[bus, 'v_nom'] if bus in n.buses.index else 0
                print(f"  {idx}: {gen['p_nom']:.0f} MW at {bus} ({v:.0f}kV)")
        else:
            print(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    compare_networks()
    
    # Check some specific important generators
    print("\n" + "="*60)
    print("SPECIFIC GENERATOR CHECKS (ETYS)")
    print("="*60)
    
    etys_path = "resources/network/Historical_2019_ETYS_network_demand_renewables_thermal_generators.pkl"
    if Path(etys_path).exists():
        check_specific_generators(
            etys_path,
            ['Hinkley', 'Sizewell', 'Drax', 'Hornsea', 'Dogger']
        )

