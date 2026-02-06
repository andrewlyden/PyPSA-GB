"""Quick analysis of HT35_flex solved network."""

import pypsa
import pandas as pd

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('HT35_FLEX SOLVED NETWORK ANALYSIS')
print('=' * 80)

# ============================================================================
# 1. BASIC STATS
# ============================================================================
print('\n1. NETWORK STATISTICS')
print('-' * 40)
print(f'Snapshots: {len(n.snapshots)} hours ({n.snapshots[0]} to {n.snapshots[-1]})')
print(f'Buses: {len(n.buses)}')
print(f'Generators: {len(n.generators)}')
print(f'Storage Units: {len(n.storage_units)}')
print(f'Stores: {len(n.stores)}')
print(f'Links: {len(n.links)}')
print(f'Loads: {len(n.loads)}')

# ============================================================================
# 2. OPTIMIZATION STATUS
# ============================================================================
print('\n2. OPTIMIZATION STATUS')
print('-' * 40)
if hasattr(n, 'optimization_status'):
    print(f'Status: {n.optimization_status}')
if hasattr(n, 'objective'):
    print(f'Objective: £{n.objective:,.0f}')
else:
    print('Objective: (not stored in network file)')

# ============================================================================
# 3. GENERATION MIX
# ============================================================================
print('\n3. GENERATION BY CARRIER')
print('-' * 40)

gen_by_carrier = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum().sum()
gen_by_carrier = gen_by_carrier.sort_values(ascending=False)
total_gen = gen_by_carrier.sum()

print(f'{"Carrier":<25} {"Energy (GWh)":>15} {"Share":>10}')
print('-' * 50)
for carrier, energy in gen_by_carrier.items():
    if energy > 0:
        print(f'{carrier:<25} {energy/1000:>15,.1f} {energy/total_gen:>10.1%}')
print('-' * 50)
print(f'{"TOTAL":<25} {total_gen/1000:>15,.1f}')

# ============================================================================
# 4. LOAD SHEDDING CHECK
# ============================================================================
print('\n4. LOAD SHEDDING CHECK')
print('-' * 40)

load_shedding = n.generators[n.generators.carrier == 'load_shedding']
if len(load_shedding) > 0:
    ls_cols = load_shedding.index
    ls_dispatch = n.generators_t.p[ls_cols].sum(axis=1)
    total_ls = ls_dispatch.sum()
    max_ls = ls_dispatch.max()
    hours_ls = (ls_dispatch > 0.1).sum()
    
    if total_ls > 1:
        print(f'⚠️  Load shedding occurred!')
        print(f'   Total energy: {total_ls:,.0f} MWh')
        print(f'   Peak: {max_ls:,.0f} MW')
        print(f'   Hours: {hours_ls}')
        
        # Top buses with load shedding
        ls_by_bus = n.generators_t.p[ls_cols].sum()
        top_ls = ls_by_bus.sort_values(ascending=False).head(5)
        print(f'\n   Top buses with load shedding:')
        for gen, mwh in top_ls.items():
            if mwh > 0:
                print(f'      {gen}: {mwh:,.0f} MWh')
    else:
        print('✓ No significant load shedding')
else:
    print('✓ No load shedding generators in network')

# ============================================================================
# 5. CURTAILMENT CHECK
# ============================================================================
print('\n5. RENEWABLE CURTAILMENT')
print('-' * 40)

re_carriers = ['wind_onshore', 'wind_offshore', 'solar_pv']
for carrier in re_carriers:
    gens = n.generators[n.generators.carrier == carrier]
    if len(gens) > 0:
        available = (gens.p_nom * n.generators_t.p_max_pu[gens.index]).sum().sum()
        dispatched = n.generators_t.p[gens.index].sum().sum()
        curtailed = available - dispatched
        if available > 0:
            curt_pct = curtailed / available * 100
            print(f'{carrier:<20}: {curtailed/1000:>8,.1f} GWh curtailed ({curt_pct:.1f}%)')

# ============================================================================
# 6. STORAGE USAGE
# ============================================================================
print('\n6. STORAGE USAGE')
print('-' * 40)

# Storage units (batteries, pumped hydro)
if len(n.storage_units) > 0:
    su_by_carrier = n.storage_units.groupby('carrier')['p_nom'].sum()
    print('Storage Units (p_nom):')
    for carrier, cap in su_by_carrier.items():
        print(f'  {carrier}: {cap:,.0f} MW')
    
    # Cycling
    su_dispatch = n.storage_units_t.p.sum()
    charge = su_dispatch[su_dispatch < 0].sum()
    discharge = su_dispatch[su_dispatch > 0].sum()
    print(f'\nTotal cycling: {abs(charge)/1000:,.1f} GWh charge, {discharge/1000:,.1f} GWh discharge')

# ============================================================================
# 7. DEMAND BALANCE
# ============================================================================
print('\n7. DEMAND BALANCE')
print('-' * 40)

# Regular loads
regular_loads = n.loads[~n.loads.carrier.isin(['EV driving', 'HP heat demand', 'heat pump heat demand'])]
ev_loads = n.loads[n.loads.carrier == 'EV driving']
hp_loads = n.loads[n.loads.carrier.isin(['HP heat demand', 'heat pump heat demand'])]

if len(regular_loads) > 0:
    reg_demand = n.loads_t.p_set[regular_loads.index].sum().sum()
    print(f'Regular demand: {reg_demand/1000:,.1f} GWh')

if len(ev_loads) > 0:
    ev_demand = n.loads_t.p_set[ev_loads.index].sum().sum()
    print(f'EV driving demand: {ev_demand/1000:,.1f} GWh')

if len(hp_loads) > 0:
    hp_demand = n.loads_t.p_set[hp_loads.index].sum().sum()
    print(f'Heat pump demand: {hp_demand/1000:,.1f} GWh')

# ============================================================================
# 8. EV FLEXIBILITY SUMMARY
# ============================================================================
print('\n8. EV FLEXIBILITY SUMMARY')
print('-' * 40)

ev_stores = n.stores[n.stores.carrier == 'EV battery']
if len(ev_stores) > 0:
    go_stores = ev_stores[ev_stores.index.str.contains('GO')]
    int_stores = ev_stores[ev_stores.index.str.contains('INT')]
    v2g_stores = ev_stores[ev_stores.index.str.contains('V2G')]
    
    print(f'GO fleet battery: {go_stores.e_nom.sum():,.0f} MWh')
    print(f'INT fleet battery: {int_stores.e_nom.sum():,.0f} MWh')
    print(f'V2G fleet battery: {v2g_stores.e_nom.sum():,.0f} MWh')

# V2G discharge
v2g_links = n.links[n.links.carrier == 'V2G']
if len(v2g_links) > 0:
    v2g_dispatch = n.links_t.p0[v2g_links.index].sum().sum()
    v2g_capacity = v2g_links.p_nom.sum()
    print(f'\nV2G capacity: {v2g_capacity:,.0f} MW')
    print(f'V2G discharge: {v2g_dispatch/1000:,.1f} GWh')

# ============================================================================
# 9. LINE LOADING
# ============================================================================
print('\n9. TRANSMISSION LINE LOADING')
print('-' * 40)

if len(n.lines) > 0 and len(n.lines_t.p0) > 0:
    line_loading = n.lines_t.p0.abs().max() / n.lines.s_nom
    congested = line_loading[line_loading > 0.95]
    print(f'Lines at >95% loading: {len(congested)} of {len(n.lines)}')
    if len(congested) > 0:
        print('Top congested lines:')
        for line in congested.sort_values(ascending=False).head(5).index:
            print(f'  {line}: {line_loading[line]:.1%}')

print('\n' + '=' * 80)
print('ANALYSIS COMPLETE')
print('=' * 80)
