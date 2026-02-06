"""Analyze FES demand building blocks with correct definitions."""

import pandas as pd

fes = pd.read_csv('resources/FES/FES_2024_data.csv')

print('=' * 80)
print('FES 2024 DEMAND BUILDING BLOCKS - Holistic Transition 2035')
print('=' * 80)

# Filter for Holistic Transition
ht = fes[fes['FES Pathway'] == 'Holistic Transition']

# Get values for each building block
bbs = {
    'Dem_BB003': ('Total Annual Demand', 'GWh'),
    'Dem_BB004': ('Residential HP Demand', 'GWh'),
    'Dem_BB005': ('I&C HP Demand', 'GWh'),
    'Dem_BB006': ('EV Demand 1 (Transport)', 'GWh'),
    'Dem_BB007': ('EV Demand 2 (Transport)', 'GWh'),
    'Dem_BB008': ('Baseline Demand (Gross)', 'GWh'),
    'Dem_BB008a': ('Baseline Domestic True Demand', 'GWh'),
    'Dem_BB008b': ('Non-domestic Demand', 'GWh'),
}

print('\nBuilding Block Values:')
print('-' * 60)
totals = {}
for bb, (desc, unit) in bbs.items():
    data = ht[ht['Building Block ID Number'] == bb]
    if len(data) > 0 and '2035' in data.columns:
        val = data['2035'].sum()
        totals[bb] = val
        print(f'{bb:<12} {desc:<35} {val:>12,.0f} {unit}')
    else:
        totals[bb] = 0
        print(f'{bb:<12} {desc:<35} {"N/A":>12}')

# Calculate relationships
print('\n' + '=' * 80)
print('DEMAND COMPOSITION ANALYSIS')
print('=' * 80)

bb003 = totals.get('Dem_BB003', 0)  # Total demand
bb004 = totals.get('Dem_BB004', 0)  # Residential HP
bb005 = totals.get('Dem_BB005', 0)  # I&C HP
bb006 = totals.get('Dem_BB006', 0)  # EV 1
bb007 = totals.get('Dem_BB007', 0)  # EV 2
bb008 = totals.get('Dem_BB008', 0)  # Baseline
bb008a = totals.get('Dem_BB008a', 0)  # Baseline Domestic
bb008b = totals.get('Dem_BB008b', 0)  # Non-domestic

total_hp = bb004 + bb005
total_ev = bb006 + bb007

print(f'\n1. Component Totals:')
print(f'   Total Demand (BB003):     {bb003:>12,.0f} GWh')
print(f'   Baseline (BB008):         {bb008:>12,.0f} GWh')
print(f'   Heat Pump (BB004+005):    {total_hp:>12,.0f} GWh')
print(f'   EV (BB006+007):           {total_ev:>12,.0f} GWh')

print(f'\n2. Testing if BB003 = BB008 + HP + EV:')
calculated_total = bb008 + total_hp + total_ev
print(f'   BB008 + HP + EV = {bb008:,.0f} + {total_hp:,.0f} + {total_ev:,.0f}')
print(f'                   = {calculated_total:,.0f} GWh')
print(f'   BB003 (actual) = {bb003:,.0f} GWh')
print(f'   Difference:      {calculated_total - bb003:+,.0f} GWh')

print(f'\n3. Testing if BB003 = BB008 (Baseline already includes everything):')
print(f'   BB003 - BB008 = {bb003 - bb008:,.0f} GWh')
print(f'   HP + EV       = {total_hp + total_ev:,.0f} GWh')

print(f'\n4. Interpretation:')
if abs(calculated_total - bb003) < 1000:  # Within 1 TWh
    print('   ✓ BB003 = BB008 + HP + EV (Baseline is EXCLUDING electrification)')
    print('   → The model should use BB008 as base, then ADD HP and EV separately')
elif abs(bb003 - bb008) < 1000:
    print('   ✓ BB003 ≈ BB008 (Total already includes electrification)')
    print('   → The model should use BB003 as total, HP/EV are already included')
else:
    print('   ? Relationship unclear - manual review needed')

# Check our model's approach
print('\n' + '=' * 80)
print('MODEL APPROACH CHECK')
print('=' * 80)

model_base = 325931  # GWh/year (from earlier analysis - electricity carrier)
model_ev = 64151     # GWh/year (from earlier analysis - EV driving carrier)
model_total = model_base + model_ev

print(f'\nCurrent model:')
print(f'   Base electricity: {model_base:>12,.0f} GWh')
print(f'   EV driving:       {model_ev:>12,.0f} GWh')
print(f'   Total:            {model_total:>12,.0f} GWh')

print(f'\nFES targets:')
print(f'   BB003 (Total):    {bb003:>12,.0f} GWh')
print(f'   BB008 (Baseline): {bb008:>12,.0f} GWh')
print(f'   EV (BB006+007):   {total_ev:>12,.0f} GWh')

print(f'\nIf BB008 is true baseline (no electrification):')
target_base = bb008
target_ev = total_ev
target_total = bb003
print(f'   Expected base:    {target_base:>12,.0f} GWh')
print(f'   Expected EV:      {target_ev:>12,.0f} GWh')
print(f'   Expected total:   {target_total:>12,.0f} GWh')
print(f'\n   Model base excess:  {model_base - target_base:+,.0f} GWh')
print(f'   Model total excess: {model_total - target_total:+,.0f} GWh')

print('\n' + '=' * 80)
