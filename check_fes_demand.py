"""Check FES data for peak demand and EV projections."""

import pandas as pd

fes = pd.read_csv('resources/FES/FES_2024_data.csv')

print('=' * 80)
print('FES 2024 DEMAND DATA ANALYSIS')
print('=' * 80)

# Check Dem_BB009 which is in MW (likely peak demand)
print('\n1. PEAK DEMAND (Dem_BB009)')
print('-' * 40)
dem009 = fes[fes['Building Block ID Number'] == 'Dem_BB009']
print(f'Description: {dem009["Detail"].iloc[0] if "Detail" in dem009.columns and len(dem009) > 0 else "N/A"}')
print(f'Technology: {dem009["Technology"].iloc[0] if "Technology" in dem009.columns and len(dem009) > 0 else "N/A"}')

for scenario in ['Holistic Transition', 'Electric Engagement', 'System Transformation']:
    s = dem009[dem009['FES Pathway'] == scenario]
    if len(s) > 0 and '2035' in s.columns:
        val = s['2035'].sum()
        print(f'  {scenario} 2035: {val:,.0f} MW')

# Check EV demand building blocks
print('\n2. EV DEMAND PROJECTIONS')
print('-' * 40)

for bb in ['Dem_BB006', 'Dem_BB007']:
    dem = fes[fes['Building Block ID Number'] == bb]
    print(f'\n{bb}:')
    if len(dem) > 0:
        detail = dem['Detail'].iloc[0] if 'Detail' in dem.columns else 'N/A'
        tech = dem['Technology'].iloc[0] if 'Technology' in dem.columns else 'N/A'
        print(f'  Detail: {detail}')
        print(f'  Technology: {tech}')
        
        for scenario in ['Holistic Transition']:
            s = dem[dem['FES Pathway'] == scenario]
            if len(s) > 0 and '2035' in s.columns:
                val = s['2035'].sum()
                print(f'  {scenario} 2035: {val:,.0f} GWh')

# Check total demand building blocks
print('\n3. TOTAL DEMAND (Dem_BB003/BB004/BB005)')
print('-' * 40)

for bb in ['Dem_BB003', 'Dem_BB004', 'Dem_BB005']:
    dem = fes[fes['Building Block ID Number'] == bb]
    if len(dem) > 0:
        detail = dem['Detail'].iloc[0] if 'Detail' in dem.columns else 'N/A'
        print(f'\n{bb}: {detail}')
        
        for scenario in ['Holistic Transition']:
            s = dem[dem['FES Pathway'] == scenario]
            if len(s) > 0 and '2035' in s.columns:
                val = s['2035'].sum()
                print(f'  {scenario} 2035: {val:,.0f} GWh')

# Look at what's in each BB
print('\n4. ALL DEMAND BUILDING BLOCKS (Detail)')
print('-' * 40)
dem_bbs = fes[fes['Building Block ID Number'].str.startswith('Dem_')]
for bb in dem_bbs['Building Block ID Number'].unique():
    subset = dem_bbs[dem_bbs['Building Block ID Number'] == bb]
    detail = subset['Detail'].iloc[0] if 'Detail' in subset.columns and pd.notna(subset['Detail'].iloc[0]) else ''
    unit = subset['Unit'].iloc[0] if 'Unit' in subset.columns else ''
    print(f'{bb:<15}: {detail[:50]:<50} ({unit})')

print('\n' + '=' * 80)
