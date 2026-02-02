"""Diagnose why network only has load shedding generators"""
import pypsa
import pandas as pd

# Load network
print("Loading solved network...")
n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print("\n" + "="*80)
print("GENERATOR DIAGNOSIS")
print("="*80)

# Check all unique carriers
carriers = n.generators['carrier'].unique()
print(f"\nUnique generator carriers: {carriers}")

# Count by carrier
carrier_counts = n.generators.groupby('carrier').size()
print(f"\nGenerator counts by carrier:")
print(carrier_counts)

# Check if there are any non-load-shedding generators
non_ls = n.generators[n.generators.carrier != 'load_shedding']
print(f"\nNon-load-shedding generators: {len(non_ls)}")

if len(non_ls) > 0:
    print("\nNon-load-shedding generator carriers:")
    print(non_ls.groupby('carrier').size())

# Check the original (unsolved) network
print("\n" + "="*80)
print("CHECKING ORIGINAL NETWORK (before solving)")
print("="*80)

try:
    n_orig = pypsa.Network('resources/network/HT35_flex.nc')
    print(f"\nOriginal network generators: {len(n_orig.generators)}")
    orig_carriers = n_orig.generators.groupby('carrier').size().sort_values(ascending=False)
    print("\nOriginal generator carriers:")
    for carrier, count in orig_carriers.items():
        capacity = n_orig.generators[n_orig.generators.carrier == carrier]['p_nom'].sum()
        print(f"  {carrier:30s}: {count:4d} units, {capacity:10.1f} MW")
except Exception as e:
    print(f"Could not load original network: {e}")

# Check what happened during optimization
print("\n" + "="*80)
print("OPTIMIZATION CHECK")
print("="*80)

# Check if any generators were active
gen_output = n.generators_t.p.sum()
active_gens = gen_output[gen_output > 0]
print(f"\nActive generators (generated >0 MWh): {len(active_gens)}")

if len(active_gens) > 10:
    print("\nTop 10 active generators:")
    top_gens = gen_output.nlargest(10)
    for gen_name, energy in top_gens.items():
        carrier = n.generators.loc[gen_name, 'carrier']
        print(f"  {gen_name:40s}: {energy:10.0f} MWh ({carrier})")
else:
    print("\nAll active generators:")
    for gen_name, energy in active_gens.items():
        carrier = n.generators.loc[gen_name, 'carrier']
        print(f"  {gen_name:40s}: {energy:10.0f} MWh ({carrier})")

# Check demand vs capacity
print("\n" + "="*80)
print("CAPACITY VS DEMAND")
print("="*80)

total_demand = n.loads_t.p_set.sum().sum()
total_gen_capacity = n.generators['p_nom'].sum()
non_ls_capacity = n.generators[n.generators.carrier != 'load_shedding']['p_nom'].sum()

print(f"Total demand: {total_demand:,.0f} MWh")
print(f"Total generator capacity: {total_gen_capacity:,.0f} MW")
print(f"Non-load-shedding capacity: {non_ls_capacity:,.0f} MW")
print(f"Peak demand: {n.loads_t.p_set.sum(axis=1).max():,.0f} MW")

# Check if generators had availability
if len(n.generators_t.p_max_pu) > 0:
    print("\n" + "="*80)
    print("GENERATOR AVAILABILITY")
    print("="*80)

    # Sample a few non-load-shedding generators if they exist
    sample_gens = n.generators[n.generators.carrier != 'load_shedding'].head(5)
    if len(sample_gens) > 0:
        print("\nSample generator availability (p_max_pu):")
        for gen in sample_gens.index:
            if gen in n.generators_t.p_max_pu.columns:
                avg_avail = n.generators_t.p_max_pu[gen].mean()
                max_avail = n.generators_t.p_max_pu[gen].max()
                print(f"  {gen:40s}: avg={avg_avail:.3f}, max={max_avail:.3f}")
