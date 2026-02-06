"""Diagnose load shedding in HT35_flex solved network."""

import pypsa
import pandas as pd

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('LOAD SHEDDING DIAGNOSIS')
print('=' * 80)

# ============================================================================
# 1. Load shedding by bus type
# ============================================================================
print('\n1. LOAD SHEDDING BY BUS TYPE')
print('-' * 40)

load_shedding = n.generators[n.generators.carrier == 'load_shedding']
ls_dispatch = n.generators_t.p[load_shedding.index].sum()

# Categorize by bus type
ev_battery_ls = ls_dispatch[ls_dispatch.index.str.contains('EV battery')]
hp_ls = ls_dispatch[ls_dispatch.index.str.contains('HP|heat')]
regular_ls = ls_dispatch[~ls_dispatch.index.str.contains('EV battery|HP|heat')]

print(f'EV battery buses: {ev_battery_ls.sum():,.0f} MWh ({len(ev_battery_ls[ev_battery_ls > 0])} buses)')
print(f'Heat pump buses: {hp_ls.sum():,.0f} MWh ({len(hp_ls[hp_ls > 0])} buses)')
print(f'Regular buses: {regular_ls.sum():,.0f} MWh ({len(regular_ls[regular_ls > 0])} buses)')

# ============================================================================
# 2. Top EV battery load shedding
# ============================================================================
print('\n2. TOP EV BATTERY LOAD SHEDDING')
print('-' * 40)

ev_ls = ev_battery_ls.sort_values(ascending=False).head(10)
for gen, mwh in ev_ls.items():
    if mwh > 0:
        bus = n.generators.loc[gen, 'bus']
        # Find the connected charger
        charger_name = bus.replace('EV battery ', 'EV charger ')
        if charger_name in n.links.index:
            charger = n.links.loc[charger_name]
            charger_cap = charger.p_nom
            grid_bus = charger.bus0
            print(f'{gen}:')
            print(f'   Load shed: {mwh:,.0f} MWh')
            print(f'   Charger capacity: {charger_cap:.1f} MW')
            print(f'   Grid bus: {grid_bus}')

# ============================================================================
# 3. Check EV charger utilization
# ============================================================================
print('\n3. EV CHARGER UTILIZATION')
print('-' * 40)

# Check GO chargers
go_chargers = n.links[n.links.index.str.contains('charger GO')]
int_chargers = n.links[n.links.index.str.contains('charger INT')]
v2g_chargers = n.links[n.links.index.str.contains('charger V2G')]

if len(go_chargers) > 0:
    go_cols = [c for c in n.links_t.p0.columns if c in go_chargers.index]
    go_dispatch = n.links_t.p0[go_cols]
    go_capacity = go_chargers.loc[go_cols, 'p_nom']
    go_utilization = go_dispatch.max() / go_capacity
    print(f'GO chargers: mean max utilization = {go_utilization.mean():.1%}')

if len(int_chargers) > 0:
    int_cols = [c for c in n.links_t.p0.columns if c in int_chargers.index]
    int_dispatch = n.links_t.p0[int_cols]
    int_capacity = int_chargers.loc[int_cols, 'p_nom']
    int_utilization = int_dispatch.max() / int_capacity
    print(f'INT chargers: mean max utilization = {int_utilization.mean():.1%}')
    
    # Find chargers at full capacity
    maxed_out = int_utilization[int_utilization > 0.99]
    print(f'   Chargers at 100% capacity: {len(maxed_out)}')

if len(v2g_chargers) > 0:
    v2g_cols = [c for c in n.links_t.p0.columns if c in v2g_chargers.index]
    v2g_dispatch = n.links_t.p0[v2g_cols]
    v2g_capacity = v2g_chargers.loc[v2g_cols, 'p_nom']
    v2g_utilization = v2g_dispatch.max() / v2g_capacity
    print(f'V2G chargers: mean max utilization = {v2g_utilization.mean():.1%}')

# ============================================================================
# 4. Check when load shedding occurs
# ============================================================================
print('\n4. LOAD SHEDDING TIMING')
print('-' * 40)

ls_timeseries = n.generators_t.p[load_shedding.index].sum(axis=1)
ls_hours = ls_timeseries[ls_timeseries > 0]

if len(ls_hours) > 0:
    print('Hours with load shedding:')
    for ts, mw in ls_hours.sort_values(ascending=False).head(10).items():
        print(f'   {ts}: {mw:,.0f} MW')
    
    # Check generation availability at those times
    print('\n   System state at peak load shedding hour:')
    peak_hour = ls_timeseries.idxmax()
    print(f'   Time: {peak_hour}')
    
    # Total demand
    total_demand = n.loads_t.p_set.loc[peak_hour].sum()
    print(f'   Total demand: {total_demand:,.0f} MW')
    
    # Total generation
    total_gen = n.generators_t.p.loc[peak_hour].sum()
    print(f'   Total generation: {total_gen:,.0f} MW')
    
    # Wind generation
    wind_gens = n.generators[n.generators.carrier.isin(['wind_onshore', 'wind_offshore'])]
    wind_gen = n.generators_t.p.loc[peak_hour, wind_gens.index].sum()
    wind_avail = (wind_gens.p_nom * n.generators_t.p_max_pu.loc[peak_hour, wind_gens.index]).sum()
    print(f'   Wind generation: {wind_gen:,.0f} MW (available: {wind_avail:,.0f} MW)')
    print(f'   Wind curtailment: {wind_avail - wind_gen:,.0f} MW ({(wind_avail - wind_gen)/wind_avail*100:.1f}%)')

# ============================================================================
# 5. EV demand vs charger capacity
# ============================================================================
print('\n5. EV DEMAND VS CHARGER CAPACITY CHECK')
print('-' * 40)

# Get INT EV loads
int_loads = n.loads[n.loads.index.str.contains('EV driving INT')]
if len(int_loads) > 0:
    int_load_buses = int_loads.bus.unique()
    
    # For each INT battery bus, check if charger is adequate
    inadequate = []
    for load in int_loads.index[:10]:  # Check first 10
        battery_bus = n.loads.loc[load, 'bus']
        load_demand = n.loads_t.p_set[load].sum()  # Total energy needed
        
        # Find charger
        charger_name = battery_bus.replace('EV battery', 'EV charger')
        if charger_name in n.links.index:
            charger = n.links.loc[charger_name]
            # Total charging capacity over the week
            if charger_name in n.links_t.p_max_pu.columns:
                avail = n.links_t.p_max_pu[charger_name]
                max_charge = (charger.p_nom * avail).sum()
            else:
                max_charge = charger.p_nom * len(n.snapshots)
            
            if load_demand > max_charge * 0.9:
                inadequate.append({
                    'load': load,
                    'demand': load_demand,
                    'charger_capacity': max_charge,
                    'ratio': load_demand / max_charge
                })
    
    if inadequate:
        print('⚠️  Some INT chargers may have insufficient capacity:')
        for item in inadequate[:5]:
            print(f"   {item['load']}: demand={item['demand']:.0f} MWh, capacity={item['charger_capacity']:.0f} MWh")
    else:
        print('✓ Sampled INT chargers have adequate capacity')

# ============================================================================
# 6. Network congestion check
# ============================================================================
print('\n6. NETWORK CONGESTION ANALYSIS')
print('-' * 40)

# Find which regions have load shedding
ls_buses = n.generators.loc[ls_dispatch[ls_dispatch > 0].index, 'bus'].unique()
print(f'Load shedding at {len(ls_buses)} unique buses')

# Check if these are connected to congested lines
congested_lines = n.lines[n.lines_t.p0.abs().max() / n.lines.s_nom > 0.95]
print(f'Congested lines (>95% loading): {len(congested_lines)}')

if len(congested_lines) > 0:
    print('\nCongested line details:')
    for line in congested_lines.head(5).index:
        bus0, bus1 = n.lines.loc[line, ['bus0', 'bus1']]
        s_nom = n.lines.loc[line, 's_nom']
        flow = n.lines_t.p0[line].abs().max()
        print(f'   {line}: {bus0} <-> {bus1}, {s_nom:.0f} MW, flow={flow:.0f} MW')

print('\n' + '=' * 80)
