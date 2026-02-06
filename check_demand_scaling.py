"""Check if base demand was properly scaled during integration."""

import pypsa
import pandas as pd

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('DEMAND SCALING CHECK')
print('=' * 80)

# Check what's in the electricity loads
electricity = n.loads[n.loads.carrier == 'electricity']
ev_driving = n.loads[n.loads.carrier == 'EV driving']

# Calculate totals
e_cols = [l for l in electricity.index if l in n.loads_t.p_set.columns]
ev_cols = [l for l in ev_driving.index if l in n.loads_t.p_set.columns]

e_demand = n.loads_t.p_set[e_cols].sum().sum()
ev_demand = n.loads_t.p_set[ev_cols].sum().sum()
total_demand = e_demand + ev_demand

print(f'\n1. WEEKLY DEMAND TOTALS')
print('-' * 40)
print(f'Electricity loads: {e_demand/1000:,.1f} GWh')
print(f'EV driving loads:  {ev_demand/1000:,.1f} GWh')
print(f'Total:             {total_demand/1000:,.1f} GWh')

# Annualize
annual_e = e_demand / 1000 * 52
annual_ev = ev_demand / 1000 * 52
annual_total = total_demand / 1000 * 52

print(f'\n2. ANNUALIZED DEMAND')
print('-' * 40)
print(f'Electricity: {annual_e:,.0f} GWh/year')
print(f'EV driving:  {annual_ev:,.0f} GWh/year')
print(f'Total:       {annual_total:,.0f} GWh/year')

# FES comparison
fes_base = 286830  # GWh - from FES Dem_BB003
fes_ev = 64346     # GWh - from FES Dem_BB006 + 007
fes_total = fes_base + fes_ev  # ~351 TWh

print(f'\n3. FES 2035 HOLISTIC TRANSITION PROJECTIONS')
print('-' * 40)
print(f'FES base demand (Dem_BB003): {fes_base:,} GWh/year')
print(f'FES EV demand (Dem_BB006+7): {fes_ev:,} GWh/year')
print(f'FES total:                   {fes_base + fes_ev:,} GWh/year')

print(f'\n4. COMPARISON')
print('-' * 40)
print(f'Model electricity vs FES base: {annual_e:,.0f} vs {fes_base:,} GWh ({(annual_e/fes_base-1)*100:+.1f}%)')
print(f'Model EV vs FES EV: {annual_ev:,.0f} vs {fes_ev:,} GWh ({(annual_ev/fes_ev-1)*100:+.1f}%)')
print(f'Model total vs FES total: {annual_total:,.0f} vs {fes_total:,} GWh ({(annual_total/fes_total-1)*100:+.1f}%)')

# Check the scaling issue
print(f'\n5. SCALING ANALYSIS')
print('-' * 40)
# If disaggregation properly subtracted EV from base:
# Base should be: FES_base = 287 TWh (not 326 TWh)
# EV should be: FES_EV = 64 TWh (matches!)
# 
# The issue: electricity = 326 TWh, but should be 287 TWh
# Difference: 326 - 287 = 39 TWh
# This is ~61% of EV demand (64 TWh)

expected_e = fes_base
actual_e = annual_e
excess = actual_e - expected_e

print(f'Expected electricity demand: {expected_e:,} GWh')
print(f'Actual electricity demand:   {actual_e:,.0f} GWh')
print(f'Excess:                      {excess:,.0f} GWh')

# What fraction of EV is this?
print(f'\nExcess as fraction of EV demand: {excess/annual_ev*100:.1f}%')

# The issue seems to be that the ESPENI base demand is NOT scaled to FES total
# Let me check what the original base demand would be

# Check hourly pattern at peak
peak_hour = n.loads_t.p_set[e_cols].sum(axis=1).idxmax()
print(f'\n6. PEAK HOUR ANALYSIS')
print('-' * 40)
print(f'Peak hour: {peak_hour}')

e_peak = n.loads_t.p_set.loc[peak_hour, e_cols].sum()
ev_peak = n.loads_t.p_set.loc[peak_hour, ev_cols].sum()

print(f'Electricity at peak: {e_peak:,.0f} MW')
print(f'EV driving at peak:  {ev_peak:,.0f} MW')
print(f'Total at peak:       {e_peak + ev_peak:,.0f} MW')

# FES suggests peak around 60 GW, not 114 GW
fes_peak_estimate = 60000  # MW
print(f'\nFES estimated peak: ~{fes_peak_estimate:,} MW')
print(f'Model peak:         {e_peak + ev_peak:,.0f} MW')
print(f'Ratio:              {(e_peak + ev_peak)/fes_peak_estimate:.2f}x expected')

print('\n' + '=' * 80)
print('DIAGNOSIS')
print('=' * 80)

print('''
The electricity (base) demand is ~326 TWh/year but FES projects ~287 TWh/year.
The difference of ~39 TWh is roughly 61% of the EV demand.

This suggests that either:
1. The ESPENI base profile was not scaled down to remove EV contribution, OR
2. The FES scaling is using a different base year/profile

The EV demand (64 TWh) matches FES almost exactly, which is good.

KEY FIX NEEDED:
The base electricity demand needs to be reduced by the EV fraction before
adding EV flexibility components. Currently the EV demand is being added
ON TOP of a base that already includes an EV component.
''')
