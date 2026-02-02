"""Analyze solved PyPSA network"""
import pypsa
import pandas as pd
import numpy as np

# Load network
print("Loading network...")
n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print("\n" + "="*80)
print("NETWORK OVERVIEW")
print("="*80)
print(f"Buses: {len(n.buses)}")
print(f"Generators: {len(n.generators)}")
print(f"Loads: {len(n.loads)}")
print(f"Lines: {len(n.lines)}")
print(f"Links: {len(n.links)}")
print(f"Storage units: {len(n.storage_units)}")
print(f"Stores: {len(n.stores)}")
print(f"Snapshots: {len(n.snapshots)}")
print(f"Time period: {n.snapshots[0]} to {n.snapshots[-1]}")

print("\n" + "="*80)
print("GENERATOR TYPES")
print("="*80)
gen_types = n.generators.groupby('carrier').size().sort_values(ascending=False)
for carrier, count in gen_types.items():
    capacity = n.generators[n.generators.carrier == carrier]['p_nom'].sum()
    print(f"  {carrier:30s}: {count:4d} units, {capacity:10.1f} MW capacity")

print("\n" + "="*80)
print("LOAD TYPES")
print("="*80)
if 'carrier' in n.loads.columns:
    load_types = n.loads.groupby('carrier').size().sort_values(ascending=False)
    for carrier, count in load_types.items():
        print(f"  {carrier:30s}: {count:4d} loads")
else:
    print("  No carrier information available")
    print(f"  Total loads: {len(n.loads)}")

print("\n" + "="*80)
print("LINK TYPES (Heat Pump Flexibility)")
print("="*80)
if len(n.links) > 0:
    if 'carrier' in n.links.columns:
        link_types = n.links.groupby('carrier').size().sort_values(ascending=False)
        for carrier, count in link_types.items():
            print(f"  {carrier:30s}: {count:4d} links")
    else:
        print(f"  Total links: {len(n.links)} (no carrier info)")
else:
    print("  No links found")

print("\n" + "="*80)
print("STORE TYPES (Thermal Storage)")
print("="*80)
if len(n.stores) > 0:
    if 'carrier' in n.stores.columns:
        store_types = n.stores.groupby('carrier').size().sort_values(ascending=False)
        for carrier, count in store_types.items():
            capacity = n.stores[n.stores.carrier == carrier]['e_nom'].sum()
            print(f"  {carrier:30s}: {count:4d} stores, {capacity:10.1f} MWh capacity")
    else:
        print(f"  Total stores: {len(n.stores)} (no carrier info)")
else:
    print("  No stores found")

print("\n" + "="*80)
print("GENERATION RESULTS")
print("="*80)
if len(n.generators_t.p) > 0:
    total_gen = n.generators_t.p.sum().sum()
    gen_by_carrier = n.generators_t.p.sum().groupby(n.generators.carrier).sum().sort_values(ascending=False)
    print(f"Total generation: {total_gen:,.0f} MWh\n")
    for carrier, energy in gen_by_carrier.items():
        pct = 100 * energy / total_gen
        print(f"  {carrier:30s}: {energy:12,.0f} MWh ({pct:5.1f}%)")
else:
    print("  No generation data found")

print("\n" + "="*80)
print("LOAD SHEDDING ANALYSIS")
print("="*80)
load_shedding_gens = n.generators[n.generators.carrier == 'load_shedding']
if len(load_shedding_gens) > 0:
    ls_generation = n.generators_t.p[load_shedding_gens.index].sum().sum()
    total_demand = n.loads_t.p_set.sum().sum()
    print(f"Load shedding generators: {len(load_shedding_gens)}")
    print(f"Total load shedding: {ls_generation:,.0f} MWh")
    print(f"Total demand: {total_demand:,.0f} MWh")
    print(f"Percentage shed: {100*ls_generation/total_demand:.2f}%")

    # When did load shedding occur?
    ls_by_snapshot = n.generators_t.p[load_shedding_gens.index].sum(axis=1)
    ls_snapshots = ls_by_snapshot[ls_by_snapshot > 0]
    print(f"\nLoad shedding occurred in {len(ls_snapshots)} of {len(n.snapshots)} snapshots")
    if len(ls_snapshots) > 0:
        print(f"Peak load shedding: {ls_by_snapshot.max():,.0f} MW at {ls_by_snapshot.idxmax()}")
else:
    print("  No load shedding generators found")

print("\n" + "="*80)
print("HEAT PUMP FLEXIBILITY ANALYSIS")
print("="*80)
# Check for heat pump related components
hp_loads = n.loads[n.loads.index.str.contains('heat_pump', case=False)] if len(n.loads) > 0 else pd.DataFrame()
hp_links = n.links[n.links.index.str.contains('heat_pump|hp_', case=False)] if len(n.links) > 0 else pd.DataFrame()
hp_stores = n.stores[n.stores.index.str.contains('heat_pump|hp_|thermal', case=False)] if len(n.stores) > 0 else pd.DataFrame()

print(f"Heat pump loads: {len(hp_loads)}")
print(f"Heat pump links: {len(hp_links)}")
print(f"Thermal stores: {len(hp_stores)}")

if len(hp_links) > 0:
    print("\nHeat pump link carriers:")
    print(hp_links.groupby('carrier').size() if 'carrier' in hp_links.columns else "  No carrier info")

if len(hp_stores) > 0:
    print("\nThermal store carriers:")
    print(hp_stores.groupby('carrier').size() if 'carrier' in hp_stores.columns else "  No carrier info")
    total_thermal_capacity = hp_stores['e_nom'].sum()
    print(f"Total thermal storage capacity: {total_thermal_capacity:,.0f} MWh")

print("\n" + "="*80)
print("NETWORK STATISTICS")
print("="*80)
print(f"Total line length: {n.lines['length'].sum():,.0f} km")
print(f"Average line capacity: {n.lines['s_nom'].mean():,.0f} MVA")
if len(n.lines_t.p0) > 0:
    line_utilization = (n.lines_t.p0.abs().max() / n.lines.s_nom).mean() * 100
    print(f"Average peak line utilization: {line_utilization:.1f}%")
    congested_lines = ((n.lines_t.p0.abs().max() / n.lines.s_nom) > 0.95).sum()
    print(f"Congested lines (>95% utilization): {congested_lines}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
