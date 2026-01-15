"""
Deep investigation of unbounded model - check for profit cycles.
"""
import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network('resources/network/Historical_2020_clustered.nc')

print("=" * 80)
print("DEEP UNBOUNDEDNESS INVESTIGATION")
print("=" * 80)

# Filter to solve period
solve_start = pd.Timestamp('2020-12-14')
solve_end = pd.Timestamp('2020-12-20')
mask = (n.snapshots >= solve_start) & (n.snapshots <= solve_end)
solve_snapshots = n.snapshots[mask]
print(f"\nSolve period: {len(solve_snapshots)} snapshots ({solve_start.date()} to {solve_end.date()})")

# 1. Check if generators can have negative costs when accounting for p_max_pu
print("\n" + "=" * 80)
print("1. GENERATOR DISPATCH ECONOMICS")
print("=" * 80)

renewables_with_profile = []
for gen in n.generators.index:
    carrier = n.generators.loc[gen, 'carrier']
    mc = n.generators.loc[gen, 'marginal_cost']
    p_nom = n.generators.loc[gen, 'p_nom']
    
    # Check if has availability profile
    if gen in n.generators_t.p_max_pu.columns:
        profile = n.generators_t.p_max_pu[gen].loc[solve_snapshots]
        renewables_with_profile.append({
            'gen': gen,
            'carrier': carrier,
            'mc': mc,
            'p_nom': p_nom,
            'mean_avail': profile.mean(),
            'max_avail': profile.max(),
            'min_avail': profile.min()
        })

df_ren = pd.DataFrame(renewables_with_profile)
print(f"\nRenewables with profiles: {len(df_ren)}")
print(f"All have MC >= 0: {(df_ren['mc'] >= 0).all()}")
print(f"All have max_avail <= 1.0: {(df_ren['max_avail'] <= 1.0).all()}")

# 2. Check storage can make money
print("\n" + "=" * 80)
print("2. STORAGE PROFIT POTENTIAL")
print("=" * 80)

if len(n.storage_units) > 0:
    print(f"\nStorage units: {len(n.storage_units)}")
    
    for idx, storage in n.storage_units.iterrows():
        eff_store = storage.get('efficiency_store', 1.0)
        eff_dispatch = storage.get('efficiency_dispatch', 1.0)
        roundtrip_eff = eff_store * eff_dispatch
        mc = storage.get('marginal_cost', 0)
        standing_loss = storage.get('standing_loss', 0)
        
        print(f"\n  {idx}:")
        print(f"    Round-trip efficiency: {roundtrip_eff:.2%}")
        print(f"    Marginal cost: £{mc:.2f}/MWh")
        print(f"    Standing loss: {standing_loss:.4f} per hour ({standing_loss*100:.2f}%/hr)")
        
        # Calculate break-even price spread
        # To profit: (price_high - price_low) * efficiency > marginal_cost
        # Break-even: price_spread > mc / efficiency
        if roundtrip_eff > 0:
            min_spread = mc / roundtrip_eff
            print(f"    Min profitable spread: £{min_spread:.2f}/MWh")

# 3. Check if loads have any unusual properties
print("\n" + "=" * 80)
print("3. LOAD CONFIGURATION")
print("=" * 80)

print(f"\nLoads: {len(n.loads)}")
if len(n.loads) > 0 and len(n.loads_t.p_set) > 0:
    load_profile = n.loads_t.p_set.loc[solve_snapshots]
    total_demand = load_profile.sum(axis=1)
    print(f"  Total demand over solve period:")
    print(f"    Min: {total_demand.min():,.0f} MW")
    print(f"    Max: {total_demand.max():,.0f} MW")
    print(f"    Mean: {total_demand.mean():,.0f} MW")
    print(f"    Total energy: {total_demand.sum():,.0f} MWh")

# 4. Check for extendable capacity
print("\n" + "=" * 80)
print("4. EXTENDABLE CAPACITY CHECK")
print("=" * 80)

if 'p_nom_extendable' in n.generators.columns:
    extendable = n.generators[n.generators.p_nom_extendable == True]
    print(f"\nExtendable generators: {len(extendable)}")
    if len(extendable) > 0:
        print("  THIS COULD CAUSE UNBOUNDEDNESS if capital_cost = 0!")
        for idx, gen in extendable.head(10).iterrows():
            cap_cost = gen.get('capital_cost', 0)
            mc = gen.get('marginal_cost', 0)
            print(f"    {idx}: carrier={gen['carrier']}, capital_cost=£{cap_cost:.2f}, MC=£{mc:.2f}/MWh")
else:
    print("\n✓ No p_nom_extendable column (all capacities fixed)")

# 5. Check storage extendability
if len(n.storage_units) > 0 and 'p_nom_extendable' in n.storage_units.columns:
    ext_storage = n.storage_units[n.storage_units.p_nom_extendable == True]
    if len(ext_storage) > 0:
        print(f"\n⚠️  EXTENDABLE STORAGE: {len(ext_storage)} units")
        print("  THIS IS VERY LIKELY THE UNBOUNDEDNESS ISSUE!")
        for idx, storage in ext_storage.head(10).iterrows():
            cap_cost = storage.get('capital_cost', 0)
            print(f"    {idx}: capital_cost=£{cap_cost:.2f}")

# 6. Check line extendability
print("\n" + "=" * 80)
print("5. LINE/LINK EXTENDABILITY")
print("=" * 80)

if 's_nom_extendable' in n.lines.columns:
    ext_lines = n.lines[n.lines.s_nom_extendable == True]
    if len(ext_lines) > 0:
        print(f"\n⚠️  EXTENDABLE LINES: {len(ext_lines)}")
        print("  Could cause unboundedness if capital_cost = 0")

if len(n.links) > 0 and 'p_nom_extendable' in n.links.columns:
    ext_links = n.links[n.links.p_nom_extendable == True]
    if len(ext_links) > 0:
        print(f"\n⚠️  EXTENDABLE LINKS: {len(ext_links)}")

print("\n" + "=" * 80)

