"""Identify true offshore platform buses from ETYS B-2-1d OFTO data."""
import pandas as pd
import numpy as np

xls = pd.ExcelFile('data/network/ETYS/ETYS Appendix B 2023.xlsx')
dfa = xls.parse('B-2-1a', skiprows=1)
dfb = xls.parse('B-2-1b', skiprows=1)
dfc = xls.parse('B-2-1c', skiprows=1)
dfd = xls.parse('B-2-1d', skiprows=1)
dfe = xls.parse('B-3-1a', skiprows=1)
dff = xls.parse('B-3-1b', skiprows=1)
dfg = xls.parse('B-3-1c', skiprows=1)
dfh = xls.parse('B-3-1d', skiprows=1)

# Standardize column names
for df in [dff, dfg, dfh]:
    df.rename(columns={'Node1': 'Node 1', 'Node2': 'Node 2'}, inplace=True)

# Collect ALL buses from main ETYS (a,b,c = lines; e,f,g = transformers)
main_buses = set()
for df in [dfa, dfb, dfc]:
    main_buses.update(df['Node 1'].dropna().tolist())
    main_buses.update(df['Node 2'].dropna().tolist())
for df in [dfe, dff, dfg]:
    main_buses.update(df['Node 1'].dropna().tolist())
    main_buses.update(df['Node 2'].dropna().tolist())

# Collect ALL buses from OFTO data (d = OFTO lines, h = OFTO transformers)
ofto_buses = set()
for df in [dfd, dfh]:
    ofto_buses.update(df['Node 1'].dropna().tolist())
    ofto_buses.update(df['Node 2'].dropna().tolist())

# True offshore platforms: buses ONLY in OFTO data, not in main ETYS
ofto_only = ofto_buses - main_buses
print('=== OFTO-only buses (not in main ETYS) ===')
for b in sorted(ofto_only):
    print(f'  {b}')
print(f'Total: {len(ofto_only)}')

# Check OFTO cable connections for OFTO-only buses  
print()
print('=== OFTO cable connections for OFTO-only buses ===')
for bus in sorted(ofto_only):
    matches = dfd[(dfd['Node 1'] == bus) | (dfd['Node 2'] == bus)]
    for _, row in matches.iterrows():
        cable_km = row.get('Cable Length (km)', 0)
        ohl_km = row.get('OHL Length (km)', 0)
        total = (cable_km if pd.notna(cable_km) else 0) + (ohl_km if pd.notna(ohl_km) else 0)
        other = row['Node 2'] if row['Node 1'] == bus else row['Node 1']
        rating = row.get('Rating (MVA)', 'N/A')
        print(f'  {bus} -- {other}: {total:.1f}km, {rating} MVA')

# Also check OFTO transformer connections
print()
print('=== OFTO transformer connections for OFTO-only buses ===')
for bus in sorted(ofto_only):
    matches = dfh[(dfh['Node 1'] == bus) | (dfh['Node 2'] == bus)]
    for _, row in matches.iterrows():
        other = row['Node 2'] if row['Node 1'] == bus else row['Node 1']
        rating = row.get('Rating (MVA)', 'N/A')
        print(f'  {bus} -- {other}: {rating} MVA (transformer)')

# Check Extra_WF_edges
print()
dfj = pd.read_excel('data/network/ETYS/GB_network.xlsx', sheet_name='Extra_WF_edges')
node1_col = 'Node 1' if 'Node 1' in dfj.columns else 'Node1'
node2_col = 'Node 2' if 'Node 2' in dfj.columns else 'Node2'
wf_node1 = set(dfj[node1_col].dropna().tolist())
print('=== Extra_WF_edges Node 1 buses: are they offshore? ===')
for b in sorted(wf_node1):
    in_ofto_only = 'TRUE OFFSHORE' if b in ofto_only else 'ONSHORE (also in main ETYS or not in OFTO)'
    print(f'  {b}: {in_ofto_only}')

# Show Extra_WF_edges Node 2 buses
wf_node2 = set(dfj[node2_col].dropna().tolist())
print()
print('=== Extra_WF_edges Node 2 buses: are they offshore? ===')
for b in sorted(wf_node2):
    in_ofto_only = 'TRUE OFFSHORE' if b in ofto_only else 'ONSHORE'
    print(f'  {b}: {in_ofto_only}')

# Summary: which Extra_WF_edges buses are OFTO-only vs not
print()
print('=== SUMMARY ===')
all_wf = wf_node1 | wf_node2
offshore_wf = all_wf & ofto_only
onshore_wf = all_wf - ofto_only
print(f'Extra_WF_edges total unique buses: {len(all_wf)}')
print(f'  Truly offshore (OFTO-only): {len(offshore_wf)} - {sorted(offshore_wf)}')
print(f'  Onshore (in main ETYS): {len(onshore_wf)} - {sorted(onshore_wf)}')
