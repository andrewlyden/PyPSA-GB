"""Deeper FES demand analysis to understand composition."""

import pandas as pd

fes = pd.read_csv('resources/FES/FES_2024_data.csv')

print('=' * 80)
print('FES 2024 DEMAND BREAKDOWN - Holistic Transition 2035')
print('=' * 80)

# Filter for Holistic Transition
ht = fes[fes['FES Pathway'] == 'Holistic Transition']

# Check Dem_BB003 - this is typically total underlying demand
dem003 = ht[ht['Building Block ID Number'] == 'Dem_BB003']['2035'].sum()
dem004 = ht[ht['Building Block ID Number'] == 'Dem_BB004']['2035'].sum()
dem005 = ht[ht['Building Block ID Number'] == 'Dem_BB005']['2035'].sum()
dem006 = ht[ht['Building Block ID Number'] == 'Dem_BB006']['2035'].sum()  # EV
dem007 = ht[ht['Building Block ID Number'] == 'Dem_BB007']['2035'].sum()  # EV
dem008 = ht[ht['Building Block ID Number'] == 'Dem_BB008']['2035'].sum()  # Heat pumps
dem008a = ht[ht['Building Block ID Number'] == 'Dem_BB008a']['2035'].sum()
dem008b = ht[ht['Building Block ID Number'] == 'Dem_BB008b']['2035'].sum()

print('\nDemand Building Blocks (GWh):')
print('-' * 50)
print(f'Dem_BB003 (Underlying demand?):  {dem003:>10,.0f} GWh')
print(f'Dem_BB004 (Industrial?):         {dem004:>10,.0f} GWh')
print(f'Dem_BB005 (Losses?):             {dem005:>10,.0f} GWh')
print(f'Dem_BB006 (EV domestic?):        {dem006:>10,.0f} GWh')
print(f'Dem_BB007 (EV non-domestic?):    {dem007:>10,.0f} GWh')
print(f'Dem_BB008 (Heat pumps?):         {dem008:>10,.0f} GWh')
print(f'Dem_BB008a:                      {dem008a:>10,.0f} GWh')
print(f'Dem_BB008b:                      {dem008b:>10,.0f} GWh')

# Total
total_ev = dem006 + dem007
total_hp = dem008 + dem008a + dem008b

print('\n' + '-' * 50)
print(f'Total EV demand:                 {total_ev:>10,.0f} GWh')
print(f'Total heat pump demand:          {total_hp:>10,.0f} GWh')
print(f'\nBase demand (BB003):             {dem003:>10,.0f} GWh')

# Check if BB003 INCLUDES or EXCLUDES EV/HP
print('\n' + '=' * 50)
print('KEY QUESTION: Does BB003 include EV and HP?')
print('=' * 50)

sum_all = dem003 + dem004 + dem005 + dem006 + dem007 + dem008 + dem008a + dem008b
sum_wo_ev_hp = dem003 + dem004 + dem005
sum_w_ev_hp = dem003 + dem004 + dem005 + total_ev + total_hp

print(f'\nIf BB003 EXCLUDES EV/HP:')
print(f'  Total = BB003 + EV + HP = {sum_wo_ev_hp + total_ev + total_hp:,.0f} GWh')
print(f'  Weekly demand (÷52) = {(sum_wo_ev_hp + total_ev + total_hp)/52:,.0f} GWh')

print(f'\nIf BB003 INCLUDES EV/HP:')
print(f'  Total = BB003 only = {dem003:,.0f} GWh')
print(f'  Weekly demand (÷52) = {dem003/52:,.0f} GWh')

# Compare with our model
print('\n' + '=' * 50)
print('COMPARISON WITH MODEL')
print('=' * 50)

model_weekly_base = 6267.9  # GWh (from analysis)
model_weekly_ev = 1233.7    # GWh
model_weekly_total = model_weekly_base + model_weekly_ev

annual_base = model_weekly_base * 52
annual_ev = model_weekly_ev * 52
annual_total = annual_base + annual_ev

print(f'\nModel demand (weekly -> annual):')
print(f'  Base electricity: {model_weekly_base:,.1f} GWh/week x 52 = {annual_base:,.0f} GWh/year')
print(f'  EV driving:       {model_weekly_ev:,.1f} GWh/week x 52 = {annual_ev:,.0f} GWh/year')
print(f'  Total:            {model_weekly_total:,.1f} GWh/week x 52 = {annual_total:,.0f} GWh/year')

print(f'\nFES demand projections:')
print(f'  BB003 (base?):   {dem003:,.0f} GWh/year')
print(f'  EV (BB006+007):  {total_ev:,.0f} GWh/year')
print(f'  HP (BB008):      {total_hp:,.0f} GWh/year')

# Check the ratio
ev_share = total_ev / dem003
model_ev_share = annual_ev / annual_base

print(f'\nEV as % of base:')
print(f'  FES: {ev_share:.1%}')
print(f'  Model: {model_ev_share:.1%}')

# The key insight
print('\n' + '=' * 80)
print('DIAGNOSIS')
print('=' * 80)

if annual_total > dem003 * 1.1:  # If model total is >10% more than FES total
    print('''
⚠️  MODEL IS DOUBLE-COUNTING EV DEMAND!

The model is:
1. Taking the base demand from ESPENI/FES (which likely INCLUDES EV demand)
2. Adding EV demand ON TOP of that

This results in counting EV demand twice.

FIX: Before adding flexible EV components, SUBTRACT the EV demand from the 
base electricity loads.
''')
else:
    print('✓ Demand appears consistent with FES projections')

print('=' * 80)
