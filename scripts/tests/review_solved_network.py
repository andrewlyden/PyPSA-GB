"""
Review HT35_flex solved network to validate DSR and flexibility components.
"""

import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('HT35_FLEX SOLVED NETWORK REVIEW')
print('=' * 80)

# ===== BASIC INFO =====
print('\n📋 NETWORK OVERVIEW')
print(f'Period: {n.snapshots[0]} to {n.snapshots[-1]}')
print(f'Duration: {len(n.snapshots)} hours ({len(n.snapshots)/24:.1f} days)')
print(f'Buses: {len(n.buses)}')
print(f'Generators: {len(n.generators)}')
print(f'Storage Units: {len(n.storage_units)}')
print(f'Stores: {len(n.stores)}')
print(f'Links: {len(n.links)}')
print(f'Loads: {len(n.loads)}')

# ===== OPTIMIZATION STATUS =====
print('\n🎯 OPTIMIZATION STATUS')
if hasattr(n, 'objective'):
    print(f'Objective: £{n.objective:,.0f}')
else:
    print('No objective value found')

# ===== DEMAND ANALYSIS =====
print('\n📊 DEMAND ANALYSIS')
total_demand = n.loads_t.p_set.sum().sum() / 1000  # GWh
print(f'Total demand: {total_demand:,.1f} GWh')
print(f'Peak demand: {n.loads_t.p_set.sum(axis=1).max():,.0f} MW')
print(f'Average demand: {n.loads_t.p_set.sum(axis=1).mean():,.0f} MW')

# ===== GENERATION MIX =====
print('\n⚡ GENERATION MIX (by carrier)')
gen_by_carrier = n.generators.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
print('Capacity (MW):')
for carrier, capacity in gen_by_carrier.head(10).items():
    print(f'  {carrier:20s}: {capacity:>8,.0f} MW')

# ===== DSR ANALYSIS =====
print('\n🏠 DEMAND SIDE RESPONSE (DSR)')
dsr_gens = n.generators[n.generators.carrier == 'demand response']
if len(dsr_gens) > 0:
    print(f'DSR generators: {len(dsr_gens)}')
    print(f'Total DSR capacity: {dsr_gens.p_nom.sum():,.0f} MW')
    print(f'DSR marginal cost: £{dsr_gens.marginal_cost.iloc[0]:.0f}/MWh')
    
    # Check dispatch
    dsr_cols = [c for c in n.generators_t.p.columns if c in dsr_gens.index]
    if dsr_cols:
        dsr_dispatch = n.generators_t.p[dsr_cols].sum(axis=1)
        dsr_energy = dsr_dispatch.sum() / 1000  # GWh
        dsr_hours = (dsr_dispatch > 0.001).sum()
        
        print(f'\nDSR Dispatch:')
        print(f'  Peak dispatch: {dsr_dispatch.max():,.0f} MW ({dsr_dispatch.max()/dsr_gens.p_nom.sum()*100:.1f}% utilization)')
        print(f'  Total energy: {dsr_energy:.2f} GWh')
        print(f'  Active hours: {dsr_hours} / {len(n.snapshots)} ({dsr_hours/len(n.snapshots)*100:.1f}%)')
        
        # Check availability schedule
        sample_dsr = dsr_gens.index[0]
        if sample_dsr in n.generators_t.p_max_pu.columns:
            avail_hours = (n.generators_t.p_max_pu[sample_dsr] > 0).sum()
            print(f'  Available hours (event windows): {avail_hours} ({avail_hours/len(n.snapshots)*100:.1f}%)')
            if avail_hours > 0:
                print(f'  Utilization during events: {dsr_hours/avail_hours*100:.1f}% of available hours')
else:
    print('❌ No DSR components found')

# ===== EV FLEXIBILITY =====
print('\n🚗 ELECTRIC VEHICLE FLEXIBILITY')
ev_stores = n.stores[n.stores.carrier == 'EV battery']
ev_chargers = n.links[n.links.carrier.str.contains('EV', na=False)]
ev_loads = n.loads[n.loads.bus.str.contains('EV battery', na=False)]

if len(ev_stores) > 0:
    print(f'EV battery stores: {len(ev_stores)}')
    print(f'Total EV storage: {ev_stores.e_nom.sum()/1000:,.1f} GWh')
    print(f'EV chargers: {len(ev_chargers)}')
    print(f'Total charger capacity: {ev_chargers.p_nom.sum():,.0f} MW')
    
    # EV demand
    ev_load_cols = [c for c in ev_loads.index if c in n.loads_t.p_set.columns]
    if ev_load_cols:
        ev_demand = n.loads_t.p_set[ev_load_cols].sum().sum() / 1000
        print(f'EV driving demand: {ev_demand:,.1f} GWh')
else:
    print('❌ No EV flexibility components found')

# ===== LOAD SHEDDING =====
print('\n⚠️  LOAD SHEDDING / UNMET DEMAND')
ls_gens = n.generators[n.generators.carrier == 'load_shedding']
if len(ls_gens) > 0:
    ls_cols = [c for c in n.generators_t.p.columns if c in ls_gens.index]
    if ls_cols:
        ls_dispatch = n.generators_t.p[ls_cols].sum(axis=1)
        ls_energy = ls_dispatch.sum() / 1000
        
        if ls_energy > 0.001:
            print(f'❌ LOAD SHEDDING OCCURRED!')
            print(f'  Total load shed: {ls_energy:.2f} GWh ({ls_energy/total_demand*100:.3f}% of demand)')
            print(f'  Peak load shedding: {ls_dispatch.max():,.0f} MW')
            print(f'  Hours with shedding: {(ls_dispatch > 0.001).sum()}')
        else:
            print('✅ No load shedding (all demand met)')
    else:
        print('✅ No load shedding components dispatched')
else:
    print('ℹ️  No load shedding generators in network')

# ===== CURTAILMENT =====
print('\n💨 RENEWABLE CURTAILMENT')
renewables = ['wind_onshore', 'wind_offshore', 'solar_pv']
for carrier in renewables:
    gens = n.generators[n.generators.carrier == carrier]
    if len(gens) > 0:
        gen_cols = [c for c in gens.index if c in n.generators_t.p.columns]
        if gen_cols:
            actual = n.generators_t.p[gen_cols].sum().sum()
            potential = (n.generators_t.p_max_pu[gen_cols] * gens.p_nom).sum().sum()
            curtailment = potential - actual
            curtail_pct = curtailment / potential * 100 if potential > 0 else 0
            
            print(f'{carrier:15s}: {curtail_pct:>5.1f}% curtailed ({curtailment/1000:>6,.1f} GWh lost)')

# ===== SYSTEM PRICES =====
print('\n💰 SYSTEM MARGINAL PRICES')
avg_price = n.buses_t.marginal_price.mean(axis=1)
print(f'Average price: £{avg_price.mean():.2f}/MWh')
print(f'Min price: £{avg_price.min():.2f}/MWh')
print(f'Max price: £{avg_price.max():.2f}/MWh')
print(f'Std dev: £{avg_price.std():.2f}/MWh')

# Price distribution
print('\nPrice distribution:')
print(f'  < £50/MWh: {(avg_price < 50).sum()} hours ({(avg_price < 50).sum()/len(n.snapshots)*100:.1f}%)')
print(f'  £50-100: {((avg_price >= 50) & (avg_price < 100)).sum()} hours ({((avg_price >= 50) & (avg_price < 100)).sum()/len(n.snapshots)*100:.1f}%)')
print(f'  £100-200: {((avg_price >= 100) & (avg_price < 200)).sum()} hours ({((avg_price >= 100) & (avg_price < 200)).sum()/len(n.snapshots)*100:.1f}%)')
print(f'  > £200/MWh: {(avg_price >= 200).sum()} hours ({(avg_price >= 200).sum()/len(n.snapshots)*100:.1f}%)')

# When DSR is active
if len(dsr_gens) > 0 and len(dsr_cols) > 0:
    dsr_active_mask = dsr_dispatch > 0.001
    if dsr_active_mask.sum() > 0:
        print(f'\n💡 Prices when DSR active:')
        print(f'  Average: £{avg_price[dsr_active_mask].mean():.2f}/MWh')
        print(f'  Min: £{avg_price[dsr_active_mask].min():.2f}/MWh')
        print(f'  Max: £{avg_price[dsr_active_mask].max():.2f}/MWh')
        print(f'  DSR threshold: £{dsr_gens.marginal_cost.iloc[0]:.0f}/MWh')

print('\n' + '=' * 80)
print('SUMMARY')
print('=' * 80)

# Validation checks
issues = []
if len(dsr_gens) == 0:
    issues.append("❌ DSR not configured")
if len(ev_stores) == 0:
    issues.append("❌ EV flexibility not configured")
if ls_energy > 0.001:
    issues.append(f"⚠️  Load shedding: {ls_energy:.2f} GWh")

if issues:
    print("\nIssues found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ All flexibility components configured and operational")
    print("✅ No load shedding")
    print("✅ Network solved successfully")

print('=' * 80)
