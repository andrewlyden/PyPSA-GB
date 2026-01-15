"""
Diagnose infeasibility in PyPSA network.

This script identifies:
1. Isolated buses with no supply capability
2. Buses with more demand than any generator can supply
3. Network connectivity issues
4. Line/transformer capacity too low for required flows
"""

import pypsa
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

def check_network_connectivity(n):
    """Check if network is fully connected."""
    print("\n" + "=" * 80)
    print("1. NETWORK CONNECTIVITY CHECK")
    print("=" * 80)
    
    # Build graph from lines, links, and transformers
    G = nx.Graph()
    G.add_nodes_from(n.buses.index)
    
    # Add edges from lines
    for idx, line in n.lines.iterrows():
        G.add_edge(line.bus0, line.bus1)
    
    # Add edges from links
    for idx, link in n.links.iterrows():
        G.add_edge(link.bus0, link.bus1)
    
    # Add edges from transformers
    for idx, trafo in n.transformers.iterrows():
        G.add_edge(trafo.bus0, trafo.bus1)
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Connected components: {len(components)}")
    
    if len(components) > 1:
        print("\n❌ PROBLEM: Network is NOT fully connected!")
        for i, comp in enumerate(sorted(components, key=len, reverse=True)):
            buses_in_comp = list(comp)
            load_in_comp = n.loads.loc[n.loads.bus.isin(buses_in_comp), 'p_set'].sum() if len(n.loads) > 0 else 0
            gen_in_comp = n.generators.loc[n.generators.bus.isin(buses_in_comp), 'p_nom'].sum() if len(n.generators) > 0 else 0
            print(f"  Component {i+1}: {len(buses_in_comp)} buses, {load_in_comp:.0f} MW load, {gen_in_comp:.0f} MW capacity")
            if i < 5:  # Show first few buses for smaller components
                if len(buses_in_comp) <= 10:
                    print(f"    Buses: {buses_in_comp}")
        return components
    else:
        print("✓ Network is fully connected")
        return None


def check_supply_demand_balance(n, snapshot=None):
    """Check if total supply can meet total demand at each bus."""
    print("\n" + "=" * 80)
    print("2. SUPPLY/DEMAND BALANCE CHECK")
    print("=" * 80)
    
    if snapshot is None:
        snapshot = n.snapshots[0]
    
    # Get max generation capacity per bus
    gen_capacity = n.generators.groupby('bus')['p_nom'].sum()
    
    # Get load per bus (using p_set or loads_t.p_set)
    if 'p_set' in n.loads.columns:
        # Static loads
        load_per_bus = n.loads.groupby('bus')['p_set'].sum()
    else:
        # Time-varying loads
        if snapshot in n.loads_t.p_set.index:
            load_per_bus = n.loads_t.p_set.loc[snapshot].groupby(n.loads.bus).sum()
        else:
            load_per_bus = pd.Series(dtype=float)
    
    # Find buses with no generation
    all_buses = set(n.buses.index)
    buses_with_gen = set(gen_capacity.index)
    buses_without_gen = all_buses - buses_with_gen
    
    print(f"\nBuses with generation: {len(buses_with_gen)}")
    print(f"Buses without generation: {len(buses_without_gen)}")
    
    # Check buses with load but insufficient generation
    deficit_buses = []
    for bus in load_per_bus.index:
        load = load_per_bus.get(bus, 0)
        capacity = gen_capacity.get(bus, 0)
        if load > 0 and capacity == 0:
            deficit_buses.append((bus, load, 0, load))
    
    if deficit_buses:
        print(f"\n⚠️  Buses with load but NO generation (may be connected to other buses):")
        deficit_buses.sort(key=lambda x: x[3], reverse=True)
        for bus, load, cap, deficit in deficit_buses[:20]:
            print(f"    {bus}: {load:.0f} MW demand, {cap:.0f} MW capacity")
    
    # Overall balance
    total_capacity = n.generators.p_nom.sum()
    total_load = load_per_bus.sum()
    print(f"\nOverall balance:")
    print(f"  Total generation capacity: {total_capacity:,.0f} MW")
    print(f"  Total load (snapshot): {total_load:,.0f} MW")
    print(f"  Margin: {total_capacity - total_load:,.0f} MW ({100*(total_capacity - total_load)/total_load:.1f}%)")
    
    return deficit_buses


def check_transmission_capacity(n, snapshot=None):
    """Check if transmission capacity is sufficient for required flows."""
    print("\n" + "=" * 80)
    print("3. TRANSMISSION CAPACITY CHECK")
    print("=" * 80)
    
    if snapshot is None:
        snapshot = n.snapshots[0]
    
    # Calculate net injection at each bus (generation - load)
    gen_capacity = n.generators.groupby('bus')['p_nom'].sum()
    
    if 'p_set' in n.loads.columns:
        load_per_bus = n.loads.groupby('bus')['p_set'].sum()
    elif snapshot in n.loads_t.p_set.index:
        load_per_bus = n.loads_t.p_set.loc[snapshot].groupby(n.loads.bus).sum()
    else:
        load_per_bus = pd.Series(dtype=float)
    
    # Net injection = generation capacity - load
    all_buses = n.buses.index
    net_injection = pd.Series(0.0, index=all_buses)
    for bus in gen_capacity.index:
        if bus in net_injection.index:
            net_injection[bus] += gen_capacity[bus]
    for bus in load_per_bus.index:
        if bus in net_injection.index:
            net_injection[bus] -= load_per_bus[bus]
    
    # Find heavily net-importing and net-exporting regions
    print(f"\nBuses with largest net export requirement (surplus capacity):")
    exporters = net_injection.nlargest(10)
    for bus, val in exporters.items():
        print(f"  {bus}: +{val:,.0f} MW")
    
    print(f"\nBuses with largest net import requirement (load exceeds local capacity):")
    importers = net_injection.nsmallest(10)
    for bus, val in importers.items():
        print(f"  {bus}: {val:,.0f} MW")
    
    # Check transmission capacity INTO heavily importing regions
    print("\n" + "-" * 40)
    print("Transmission capacity into importing buses:")
    print("-" * 40)
    
    bottlenecks = []
    for bus, deficit in importers.items():
        if deficit < -100:  # Only check significant importers
            # Sum capacity of lines/trafos connecting to this bus
            line_capacity = n.lines[(n.lines.bus0 == bus) | (n.lines.bus1 == bus)]['s_nom'].sum()
            trafo_capacity = n.transformers[(n.transformers.bus0 == bus) | (n.transformers.bus1 == bus)]['s_nom'].sum()
            link_capacity = n.links[(n.links.bus0 == bus) | (n.links.bus1 == bus)]['p_nom'].sum()
            
            total_capacity = line_capacity + trafo_capacity + link_capacity
            
            if total_capacity < abs(deficit):
                bottlenecks.append((bus, deficit, total_capacity))
                print(f"  ❌ {bus}: needs {abs(deficit):,.0f} MW import, only {total_capacity:,.0f} MW transmission")
    
    if not bottlenecks:
        print("  ✓ No obvious transmission bottlenecks detected")
    
    return bottlenecks


def check_load_shedding(n):
    """Check if load shedding generators exist and are configured correctly."""
    print("\n" + "=" * 80)
    print("4. LOAD SHEDDING CONFIGURATION")
    print("=" * 80)
    
    # Find load shedding generators
    ls_gens = n.generators[n.generators.carrier.str.contains('load.*shed', case=False, na=False)]
    
    if len(ls_gens) == 0:
        print("❌ NO LOAD SHEDDING GENERATORS FOUND!")
        print("   This means infeasibility occurs when demand > supply")
        return False
    
    print(f"✓ Load shedding generators: {len(ls_gens)}")
    print(f"  Total capacity: {ls_gens.p_nom.sum():,.0f} MW")
    print(f"  Marginal cost: £{ls_gens.marginal_cost.min():.0f} - £{ls_gens.marginal_cost.max():.0f}/MWh")
    
    # Check which buses have load shedding
    buses_with_ls = set(ls_gens.bus.unique())
    buses_with_load = set(n.loads.bus.unique())
    
    unprotected_buses = buses_with_load - buses_with_ls
    if unprotected_buses:
        print(f"\n⚠️  {len(unprotected_buses)} buses with load but no load shedding generator:")
        for bus in list(unprotected_buses)[:10]:
            load = n.loads[n.loads.bus == bus]['p_set'].sum() if 'p_set' in n.loads.columns else 0
            print(f"    {bus}: {load:.0f} MW load")
    else:
        print(f"  ✓ All {len(buses_with_load)} load buses have load shedding")
    
    return True


def check_line_transformer_limits(n):
    """Check for zero or very low transmission limits."""
    print("\n" + "=" * 80)
    print("5. LINE & TRANSFORMER LIMIT CHECK")
    print("=" * 80)
    
    # Lines
    zero_lines = n.lines[n.lines.s_nom == 0]
    if len(zero_lines) > 0:
        print(f"❌ {len(zero_lines)} lines with ZERO capacity!")
        for idx in zero_lines.index[:10]:
            line = zero_lines.loc[idx]
            print(f"    {idx}: {line.bus0} -> {line.bus1}")
    else:
        print(f"✓ All {len(n.lines)} lines have positive capacity")
    
    print(f"  Line capacity range: {n.lines.s_nom.min():.0f} - {n.lines.s_nom.max():.0f} MVA")
    
    # Transformers
    if len(n.transformers) > 0:
        zero_trafos = n.transformers[n.transformers.s_nom == 0]
        if len(zero_trafos) > 0:
            print(f"❌ {len(zero_trafos)} transformers with ZERO capacity!")
            for idx in zero_trafos.index[:10]:
                trafo = zero_trafos.loc[idx]
                print(f"    {idx}: {trafo.bus0} -> {trafo.bus1}")
        else:
            print(f"✓ All {len(n.transformers)} transformers have positive capacity")
        
        print(f"  Transformer capacity range: {n.transformers.s_nom.min():.0f} - {n.transformers.s_nom.max():.0f} MVA")
    
    # Links
    if len(n.links) > 0:
        zero_links = n.links[n.links.p_nom == 0]
        if len(zero_links) > 0:
            print(f"❌ {len(zero_links)} links with ZERO capacity!")
        else:
            print(f"✓ All {len(n.links)} links have positive capacity")
        print(f"  Link capacity range: {n.links.p_nom.min():.0f} - {n.links.p_nom.max():.0f} MW")


def check_generator_constraints(n):
    """Check for impossible generator constraints."""
    print("\n" + "=" * 80)
    print("6. GENERATOR CONSTRAINT CHECK")
    print("=" * 80)
    
    issues = []
    
    # Check p_min > p_max
    if 'p_min_pu' in n.generators.columns:
        bad_gens = n.generators[n.generators.p_min_pu > 1.0]
        if len(bad_gens) > 0:
            print(f"❌ {len(bad_gens)} generators with p_min_pu > 1.0!")
            issues.extend(bad_gens.index.tolist())
    
    # Check for negative p_nom
    if 'p_nom' in n.generators.columns:
        neg_pnom = n.generators[n.generators.p_nom < 0]
        if len(neg_pnom) > 0:
            print(f"❌ {len(neg_pnom)} generators with negative p_nom!")
            issues.extend(neg_pnom.index.tolist())
        else:
            print(f"✓ All generators have non-negative p_nom")
    
    # Check for p_max_pu > 1 (unusual)
    if 'p_max_pu' in n.generators.columns:
        high_pmax = n.generators[n.generators.p_max_pu > 1.0]
        if len(high_pmax) > 0:
            print(f"⚠️  {len(high_pmax)} generators with p_max_pu > 1.0")
    
    if not issues:
        print("✓ Generator constraints look valid")
    
    return issues


def diagnose_infeasibility(network_path):
    """Main entry point for infeasibility diagnosis."""
    print("=" * 80)
    print("INFEASIBILITY DIAGNOSIS")
    print("=" * 80)
    print(f"\nNetwork: {network_path}")
    
    n = pypsa.Network(network_path)
    
    print(f"\nNetwork summary:")
    print(f"  Buses: {len(n.buses)}")
    print(f"  Lines: {len(n.lines)}")
    print(f"  Transformers: {len(n.transformers)}")
    print(f"  Links: {len(n.links)}")
    print(f"  Generators: {len(n.generators)}")
    print(f"  Loads: {len(n.loads)}")
    print(f"  Storage units: {len(n.storage_units)}")
    print(f"  Snapshots: {len(n.snapshots)}")
    
    # Run all checks
    components = check_network_connectivity(n)
    deficit_buses = check_supply_demand_balance(n)
    bottlenecks = check_transmission_capacity(n)
    has_load_shedding = check_load_shedding(n)
    check_line_transformer_limits(n)
    gen_issues = check_generator_constraints(n)
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    issues = []
    if components and len(components) > 1:
        issues.append(f"Network has {len(components)} disconnected components")
    if bottlenecks:
        issues.append(f"{len(bottlenecks)} transmission bottlenecks found")
    if not has_load_shedding:
        issues.append("No load shedding generators")
    if gen_issues:
        issues.append(f"{len(gen_issues)} generators with invalid constraints")
    
    if issues:
        print("\n❌ LIKELY CAUSES OF INFEASIBILITY:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n⚠️  No obvious causes found in quick check.")
        print("   Consider:")
        print("   - Time-varying constraints (check multiple snapshots)")
        print("   - Storage state of charge constraints")
        print("   - Ramp rate constraints")
        print("   - Running with expanded transmission capacity to test")
    
    return n, issues


if __name__ == "__main__":
    if len(sys.argv) < 2:
        network_path = "resources/network/Historical_2020_ETYS.nc"
    else:
        network_path = sys.argv[1]
    
    diagnose_infeasibility(network_path)

