"""Investigate if EV disaggregation is creating unrealistic demand."""

import pypsa
import pandas as pd

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('DEMAND COMPOSITION ANALYSIS')
print('=' * 80)

# ============================================================================
# 1. Breakdown of all loads
# ============================================================================
print('\n1. LOAD BREAKDOWN BY CARRIER')
print('-' * 40)

loads_by_carrier = n.loads.groupby('carrier').size()
print(f'{"Carrier":<35} {"Count":>10}')
for carrier, count in loads_by_carrier.items():
    print(f'{carrier:<35} {count:>10}')

# ============================================================================
# 2. Energy demand by carrier
# ============================================================================
print('\n2. ENERGY DEMAND BY CARRIER (weekly)')
print('-' * 40)

energy_by_carrier = {}
for carrier in n.loads.carrier.unique():
    carrier_loads = n.loads[n.loads.carrier == carrier].index
    carrier_loads = [l for l in carrier_loads if l in n.loads_t.p_set.columns]
    if carrier_loads:
        energy = n.loads_t.p_set[carrier_loads].sum().sum()
        energy_by_carrier[carrier] = energy

total_energy = sum(energy_by_carrier.values())
print(f'{"Carrier":<35} {"Energy (GWh)":>15} {"Share":>10}')
print('-' * 60)
for carrier, energy in sorted(energy_by_carrier.items(), key=lambda x: -x[1]):
    print(f'{carrier:<35} {energy/1000:>15,.1f} {energy/total_energy:>10.1%}')
print('-' * 60)
print(f'{"TOTAL":<35} {total_energy/1000:>15,.1f}')

# ============================================================================
# 3. Check if EV driving is additional or subtracted
# ============================================================================
print('\n3. EV DEMAND ANALYSIS')
print('-' * 40)

# Get base electricity demand (the main load)
base_loads = n.loads[n.loads.carrier == 'electricity']
ev_loads = n.loads[n.loads.carrier == 'EV driving']
hp_loads = n.loads[n.loads.carrier.str.contains('heat', case=False)]

print(f'Base electricity loads: {len(base_loads)}')
print(f'EV driving loads: {len(ev_loads)}')
print(f'Heat pump loads: {len(hp_loads)}')

if len(base_loads) > 0:
    base_energy = n.loads_t.p_set[base_loads.index].sum().sum()
    print(f'\nBase electricity demand: {base_energy/1000:,.1f} GWh')

if len(ev_loads) > 0:
    ev_energy = n.loads_t.p_set[ev_loads.index].sum().sum()
    print(f'EV driving demand: {ev_energy/1000:,.1f} GWh')

if len(hp_loads) > 0:
    hp_cols = [l for l in hp_loads.index if l in n.loads_t.p_set.columns]
    hp_energy = n.loads_t.p_set[hp_cols].sum().sum()
    print(f'Heat pump demand: {hp_energy/1000:,.1f} GWh')

# ============================================================================
# 4. Peak demand composition
# ============================================================================
print('\n4. PEAK DEMAND COMPOSITION')
print('-' * 40)

total_demand = n.loads_t.p_set.sum(axis=1)
peak_hour = total_demand.idxmax()
peak_mw = total_demand.max()

print(f'Peak hour: {peak_hour}')
print(f'Peak total demand: {peak_mw:,.0f} MW')

# Break down by carrier at peak
print(f'\nDemand at peak hour by carrier:')
for carrier in n.loads.carrier.unique():
    carrier_loads = n.loads[n.loads.carrier == carrier].index
    carrier_loads = [l for l in carrier_loads if l in n.loads_t.p_set.columns]
    if carrier_loads:
        demand = n.loads_t.p_set.loc[peak_hour, carrier_loads].sum()
        if demand > 0:
            print(f'  {carrier:<30}: {demand:>10,.0f} MW ({demand/peak_mw*100:>5.1f}%)')

# ============================================================================
# 5. Compare with expected FES peak demand
# ============================================================================
print('\n5. FES PEAK DEMAND COMPARISON')
print('-' * 40)

# FES 2024 Holistic Transition peak demand projections
# From FES data - typical winter peak for 2035
fes_peak_2035_holistic = 60000  # ~60 GW expected peak for 2035

print(f'Expected FES peak demand (~2035): ~{fes_peak_2035_holistic:,} MW')
print(f'Model peak demand: {peak_mw:,.0f} MW')
print(f'Difference: {peak_mw - fes_peak_2035_holistic:+,.0f} MW ({(peak_mw/fes_peak_2035_holistic-1)*100:+.1f}%)')

# ============================================================================
# 6. Check for double counting
# ============================================================================
print('\n6. DOUBLE COUNTING CHECK')
print('-' * 40)

# The question is: was EV demand subtracted from base demand before being added back?
# If not, we're double counting

# Check the profile of EV driving vs base demand
ev_timeseries = n.loads_t.p_set[ev_loads.index].sum(axis=1) if len(ev_loads) > 0 else pd.Series(0, index=n.snapshots)
base_timeseries = n.loads_t.p_set[base_loads.index].sum(axis=1) if len(base_loads) > 0 else pd.Series(0, index=n.snapshots)

print(f'Base demand range: {base_timeseries.min():,.0f} - {base_timeseries.max():,.0f} MW')
print(f'EV driving range: {ev_timeseries.min():,.0f} - {ev_timeseries.max():,.0f} MW')

# Calculate what the system should look like
print(f'\nIf EV is additional (not subtracted from base):')
print(f'  Total = base + EV = {base_timeseries.max() + ev_timeseries.max():,.0f} MW peak')
print(f'\nIf EV was subtracted from base first:')
print(f'  Total should equal original FES demand')

# ============================================================================
# 7. Hourly demand pattern
# ============================================================================
print('\n7. DEMAND PATTERN BY HOUR')
print('-' * 40)

hourly_base = base_timeseries.groupby(base_timeseries.index.hour).mean()
hourly_ev = ev_timeseries.groupby(ev_timeseries.index.hour).mean()
hourly_total = total_demand.groupby(total_demand.index.hour).mean()

print(f'{"Hour":>4} {"Base MW":>12} {"EV MW":>12} {"Total MW":>12}')
print('-' * 45)
for hour in range(24):
    print(f'{hour:>4} {hourly_base.get(hour, 0):>12,.0f} {hourly_ev.get(hour, 0):>12,.0f} {hourly_total.get(hour, 0):>12,.0f}')

print('\n' + '=' * 80)
