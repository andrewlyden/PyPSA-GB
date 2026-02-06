"""Check actual load carriers in the network to understand the demand structure."""

import pypsa
import pandas as pd

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('LOAD CARRIER ANALYSIS')
print('=' * 80)

# Check all unique carriers
print('\n1. UNIQUE LOAD CARRIERS')
print('-' * 40)
carriers = n.loads.carrier.value_counts()
for carrier, count in carriers.items():
    print(f'{carrier:<30}: {count:>5} loads')

# Check if there are both electric_vehicles loads and EV driving loads
print('\n2. EV-RELATED LOADS')
print('-' * 40)

ev_vehicles = n.loads[n.loads.carrier == 'electric_vehicles']
ev_driving = n.loads[n.loads.carrier == 'EV driving']
electricity = n.loads[n.loads.carrier == 'electricity']

print(f'electric_vehicles carrier: {len(ev_vehicles)} loads')
print(f'EV driving carrier: {len(ev_driving)} loads')
print(f'electricity carrier: {len(electricity)} loads')

# Check demand for each
if len(ev_vehicles) > 0:
    ev_v_cols = [l for l in ev_vehicles.index if l in n.loads_t.p_set.columns]
    if ev_v_cols:
        ev_v_demand = n.loads_t.p_set[ev_v_cols].sum().sum()
        print(f'\nelectric_vehicles total demand: {ev_v_demand:,.0f} MWh')

if len(ev_driving) > 0:
    ev_d_cols = [l for l in ev_driving.index if l in n.loads_t.p_set.columns]
    if ev_d_cols:
        ev_d_demand = n.loads_t.p_set[ev_d_cols].sum().sum()
        print(f'EV driving total demand: {ev_d_demand:,.0f} MWh')

if len(electricity) > 0:
    e_cols = [l for l in electricity.index if l in n.loads_t.p_set.columns]
    if e_cols:
        e_demand = n.loads_t.p_set[e_cols].sum().sum()
        print(f'electricity total demand: {e_demand:,.0f} MWh')

# Check example loads
print('\n3. EXAMPLE EV DRIVING LOADS')
print('-' * 40)
if len(ev_driving) > 0:
    sample = ev_driving.head(5)
    for load in sample.index:
        bus = ev_driving.loc[load, 'bus']
        if load in n.loads_t.p_set.columns:
            demand = n.loads_t.p_set[load].sum()
            print(f'{load}: bus={bus}, demand={demand:,.0f} MWh')

# Key question: are the EV driving loads on the EV battery bus or grid bus?
print('\n4. EV LOAD BUS TYPE CHECK')
print('-' * 40)

if len(ev_driving) > 0:
    ev_driving_buses = ev_driving.bus.unique()
    battery_buses = [b for b in ev_driving_buses if 'EV battery' in b]
    grid_buses = [b for b in ev_driving_buses if 'EV battery' not in b]
    print(f'EV driving loads on EV battery buses: {len(battery_buses)}')
    print(f'EV driving loads on grid buses: {len(grid_buses)}')

# Check if electric_vehicles carrier loads exist (from disaggregation)
print('\n5. LOOKING FOR DOUBLE-COUNTING')
print('-' * 40)

if len(ev_vehicles) > 0:
    print('⚠️  electric_vehicles carrier loads EXIST in network')
    ev_v_buses = ev_vehicles.bus.unique()
    print(f'   These are on {len(ev_v_buses)} unique buses')
    # Check if any are time-varying
    ev_v_cols = [l for l in ev_vehicles.index if l in n.loads_t.p_set.columns]
    if ev_v_cols:
        print(f'   {len(ev_v_cols)} have time-varying demand')
        ev_v_total = n.loads_t.p_set[ev_v_cols].sum().sum()
        print(f'   Total demand: {ev_v_total:,.0f} MWh')
else:
    print('✓ No electric_vehicles carrier loads in network')
    print('  (This is correct if flexibility is handling all EV demand)')

print('\n' + '=' * 80)
