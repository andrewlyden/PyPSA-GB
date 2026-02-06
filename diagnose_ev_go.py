"""Investigate EV GO tariff load shedding issue."""

import pypsa
import pandas as pd

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('EV GO TARIFF INVESTIGATION')
print('=' * 80)

# Check one problematic bus
problem_bus = 'EV battery_WALP11 EV battery GO'

# Get components at this bus
bus_gens = n.generators[n.generators.bus == problem_bus]
bus_loads = n.loads[n.loads.bus == problem_bus]
bus_stores = n.stores[n.stores.bus == problem_bus]
bus_links_from = n.links[n.links.bus0 == problem_bus]
bus_links_to = n.links[n.links.bus1 == problem_bus]

print(f'\nExample problem bus: {problem_bus}')
print(f'\nGenerators: {len(bus_gens)}')
for gen in bus_gens.index:
    carrier = bus_gens.loc[gen, 'carrier']
    p_nom = bus_gens.loc[gen, 'p_nom']
    print(f'  {gen}: {carrier}, p_nom={p_nom:.2f} MW')

print(f'\nLoads: {len(bus_loads)}')
for load in bus_loads.index:
    demand = n.loads_t.p_set[load].sum()
    carrier = bus_loads.loc[load, 'carrier']
    print(f'  {load}: {carrier}, total={demand:,.0f} MWh')

print(f'\nStores: {len(bus_stores)}')
for store in bus_stores.index:
    e_nom = bus_stores.loc[store, 'e_nom']
    print(f'  {store}: e_nom={e_nom:.2f} MWh')

print(f'\nLinks TO this bus: {len(bus_links_to)}')
for link in bus_links_to.index:
    charge = n.links_t.p0[link].clip(lower=0).sum()
    p_nom = n.links.loc[link, 'p_nom']
    bus0 = n.links.loc[link, 'bus0']
    print(f'  {link}: from {bus0}, p_nom={p_nom:.2f} MW, charged={charge:,.0f} MWh')

print(f'\nLinks FROM this bus: {len(bus_links_from)}')
for link in bus_links_from.index:
    flow = n.links_t.p0[link].sum()
    bus1 = n.links.loc[link, 'bus1']
    print(f'  {link}: to {bus1}, flow={flow:,.0f} MWh')

# Check the charger availability profile
charger_link = 'WALP11 EV charger GO'
if charger_link in n.links_t.p_max_pu.columns:
    avail = n.links_t.p_max_pu[charger_link]
    print(f'\n\nCharger availability profile for {charger_link}:')
    print(f'  Mean: {avail.mean():.2f}')
    print(f'  Min: {avail.min():.2f}')
    print(f'  Max: {avail.max():.2f}')
    print(f'  Hours available (>0): {(avail > 0).sum()} of {len(avail)}')
    
    # Calculate energy available vs required
    p_nom = n.links.loc[charger_link, 'p_nom']
    max_energy = (avail * p_nom).sum() * 0.9  # 90% efficiency
    
    driving_load = 'WALP11 EV driving GO'
    demand_energy = n.loads_t.p_set[driving_load].sum()
    
    print(f'\n  Energy available (with efficiency): {max_energy:,.0f} MWh')
    print(f'  Driving demand: {demand_energy:,.0f} MWh')
    print(f'  Ratio (available/demand): {max_energy/demand_energy:.2f}x')
    
    if max_energy < demand_energy:
        print(f'\n  ⚠️  INSUFFICIENT CHARGING CAPACITY!')
        print(f'  Shortfall: {demand_energy - max_energy:,.0f} MWh')
    
else:
    print(f'\n\n⚠️  No p_max_pu profile found for {charger_link}')

# Check battery size vs daily cycle
battery_capacity = bus_stores.loc[bus_stores.index[0], 'e_nom']
daily_demand = n.loads_t.p_set[driving_load].sum() / 7  # 7 days
print(f'\n\nBattery sizing:')
print(f'  Battery capacity: {battery_capacity:.2f} MWh')
print(f'  Average daily demand: {daily_demand:.2f} MWh')
print(f'  Ratio: {battery_capacity/daily_demand:.2f}x')

# Check actual SOC profile
store_name = bus_stores.index[0]
if store_name in n.stores_t.e.columns:
    soc = n.stores_t.e[store_name]
    print(f'\n  SOC range: {soc.min():.2f} to {soc.max():.2f} MWh')
    print(f'  Utilization: {soc.max()/battery_capacity:.1%}')
    
    # Check if SOC hits zero (causing load shedding)
    zero_soc = (soc < 0.01).sum()
    if zero_soc > 0:
        print(f'  ⚠️  SOC near zero for {zero_soc} timesteps - causing load shedding!')
