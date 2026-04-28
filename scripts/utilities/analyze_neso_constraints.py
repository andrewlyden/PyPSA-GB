"""Deep analysis of NESO constraint data vs model congestion."""
import requests
import pandas as pd
import io
import numpy as np

def fetch_csv(url):
    resp = requests.get(url)
    return pd.read_csv(io.StringIO(resp.text))

def fetch_xlsx(url):
    resp = requests.get(url)
    return pd.read_excel(io.BytesIO(resp.content))

# ─── 1. THERMAL CONSTRAINT COSTS BY BOUNDARY (2019-20 + 2020-21) ───────────
print("=" * 90)
print("1. NESO THERMAL CONSTRAINT COSTS BY BOUNDARY — Calendar Year 2020")
print("=" * 90)

# 2019-20 file covers Apr 2019 - Mar 2020 (need Jan-Mar 2020)
tc_1920 = fetch_xlsx('https://api.neso.energy/dataset/f0055054-c55c-4068-a01c-61da4334e58f/resource/d195f1d8-7d9e-46f1-96a6-4251e75e9bd0/download/map-of-outturn-system-costs-19-20.xlsx')
print(f"\n19-20 file shape: {tc_1920.shape}, columns: {list(tc_1920.columns)}")
print(tc_1920.head())

# 2020-21 file covers Apr 2020 - Mar 2021 (need Apr-Dec 2020) 
tc_2021 = fetch_csv('https://api.neso.energy/dataset/f0055054-c55c-4068-a01c-61da4334e58f/resource/4357dd3b-5c7a-4caa-8d1a-8cf848521143/download/outturn-system-costs-2021-2022.csv')
print(f"\n21-22 file shape: {tc_2021.shape}, columns: {list(tc_2021.columns)}")
print(tc_2021.head())

# The 20-21 file is the main one for our validation year
tc_main = fetch_xlsx('https://api.neso.energy/dataset/f0055054-c55c-4068-a01c-61da4334e58f/resource/a4302916-85af-4a4d-8171-1bdf8f0697a7/download/map-of-outturn-system-costs-20-21.xlsx')
print(f"\n20-21 file shape: {tc_main.shape}, columns: {list(tc_main.columns)}")
print(tc_main.head(10))
print(f"\nConstraint Groups: {tc_main['Constraint Group'].unique()}")

# Filter to calendar year 2020 (Jan-Mar from 19-20, Apr-Dec from 20-21)
# 19-20 file: get Jan-Mar 2020  
tc_1920['Settlement Date'] = pd.to_datetime(tc_1920['Settlement Date'])
jan_mar_2020 = tc_1920[(tc_1920['Settlement Date'] >= '2020-01-01') & 
                        (tc_1920['Settlement Date'] < '2020-04-01')]

# 20-21 file: get Apr-Dec 2020
tc_main['Settlement Date'] = pd.to_datetime(tc_main['Settlement Date'])
apr_dec_2020 = tc_main[(tc_main['Settlement Date'] >= '2020-04-01') & 
                        (tc_main['Settlement Date'] < '2021-01-01')]

cal_2020 = pd.concat([jan_mar_2020, apr_dec_2020], ignore_index=True)
print(f"\nCalendar year 2020: {len(cal_2020)} rows, {cal_2020['Settlement Date'].min()} to {cal_2020['Settlement Date'].max()}")

# Annual costs by boundary
boundary_costs = cal_2020.groupby('Constraint Group')['Daily Cost (GBP)'].sum().sort_values(ascending=False)
total_thermal = boundary_costs.sum()
print(f"\n{'Constraint Boundary':<25} {'Annual Cost (GBP)':>20} {'% of Total':>12}")
print("-" * 60)
for boundary, cost in boundary_costs.items():
    if cost > 0:
        print(f"{boundary:<25} {cost:>20,.0f} {cost/total_thermal*100:>11.1f}%")
print(f"{'TOTAL':<25} {total_thermal:>20,.0f} {'100.0%':>12}")

# ─── 2. CONSTRAINT BREAKDOWN — Cost type split for 2020 ─────────────────────
print("\n" + "=" * 90)
print("2. NESO CONSTRAINT BREAKDOWN BY TYPE — Calendar Year 2020")
print("=" * 90)

cb_1920 = fetch_csv('https://api.neso.energy/dataset/fb56b46e-cef3-4eb8-9294-0ca19769b7eb/resource/c4d2be3a-4c05-4fac-be7e-56368ca46142/download/constraint-breakdown-2019-2020.csv')
cb_2021 = fetch_csv('https://api.neso.energy/dataset/fb56b46e-cef3-4eb8-9294-0ca19769b7eb/resource/87088ac4-72d5-48ff-9ee1-f2a99e18277a/download/constraint-breakdown-2020-2021.csv')

cb_1920['Date'] = pd.to_datetime(cb_1920['Date'])
cb_2021['Date'] = pd.to_datetime(cb_2021['Date'])

jan_mar = cb_1920[(cb_1920['Date'] >= '2020-01-01') & (cb_1920['Date'] < '2020-04-01')]
apr_dec = cb_2021[(cb_2021['Date'] >= '2020-04-01') & (cb_2021['Date'] < '2021-01-01')]

cb_2020 = pd.concat([jan_mar, apr_dec], ignore_index=True)

cost_cols = [c for c in cb_2020.columns if 'cost' in c.lower()]
vol_cols = [c for c in cb_2020.columns if 'volume' in c.lower()]

print(f"\nConstraint Costs (Calendar Year 2020, £M):")
for col in cost_cols:
    total_m = cb_2020[col].sum() / 1e6
    print(f"  {col}: £{total_m:.1f}M")

total_all_constraints = sum(cb_2020[col].sum() for col in cost_cols)
print(f"  TOTAL ALL CONSTRAINTS: £{total_all_constraints/1e6:.1f}M")

thermal_cost = cb_2020['Thermal constraints cost'].sum()
print(f"\n  Thermal constraints only: £{thermal_cost/1e6:.1f}M")
print(f"  (This is what our model tries to reproduce)")

print(f"\nConstraint Volumes (Calendar Year 2020, MWh):")
for col in vol_cols:
    total = cb_2020[col].sum()
    print(f"  {col}: {total:,.0f} MWh")

# ─── 3. DAY-AHEAD FLOWS FOR TOP BOUNDARIES FOR 2020 ─────────────────────────
print("\n" + "=" * 90)
print("3. DAY-AHEAD CONSTRAINT FLOWS — Year 2020 (Limit vs Flow)")
print("=" * 90)

# This is a huge file, but let's process it
da = fetch_csv('https://api.neso.energy/dataset/cf3cbc92-2d5d-4c2b-bd29-e11a21070b26/resource/38a18ec1-9e40-465d-93fb-301e80fd1352/download/day-ahead-constraints-limits-and-flow-output-v1.5.csv')
da['Date'] = pd.to_datetime(da['Date (GMT/BST)'])
da_2020 = da[(da['Date'] >= '2020-01-01') & (da['Date'] < '2021-01-01')]

print(f"\nDay-ahead data for 2020: {len(da_2020)} rows")
print(f"Boundaries in 2020: {da_2020['Constraint Group'].unique()}")

# For each boundary, compute: avg limit, avg flow, % utilization, hours at limit
for group in sorted(da_2020['Constraint Group'].unique()):
    grp = da_2020[da_2020['Constraint Group'] == group].copy()
    grp['Limit (MW)'] = pd.to_numeric(grp['Limit (MW)'], errors='coerce')
    grp['Flow (MW)'] = pd.to_numeric(grp['Flow (MW)'], errors='coerce')
    grp = grp.dropna(subset=['Limit (MW)', 'Flow (MW)'])
    if len(grp) == 0:
        continue
    avg_limit = grp['Limit (MW)'].mean()
    avg_flow = grp['Flow (MW)'].mean()
    max_flow = grp['Flow (MW)'].max()
    # Utilization = flow / limit (where both positive)
    mask = (grp['Limit (MW)'] > 0)
    if mask.sum() > 0:
        utilization = (grp.loc[mask, 'Flow (MW)'] / grp.loc[mask, 'Limit (MW)']).mean() * 100
        near_limit = ((grp.loc[mask, 'Flow (MW)'] / grp.loc[mask, 'Limit (MW)']) > 0.9).sum()
    else:
        utilization = 0
        near_limit = 0
    
    print(f"\n  {group}:")
    print(f"    Avg Limit: {avg_limit:,.0f} MW, Avg Flow: {avg_flow:,.0f} MW, Max Flow: {max_flow:,.0f} MW")
    print(f"    Avg Utilization: {utilization:.1f}%, Periods >90% loaded: {near_limit} ({near_limit/max(1,len(grp))*100:.1f}%)")

# ─── 4. KEY COMPARISON: Model boundaries vs NESO boundaries ─────────────────
print("\n" + "=" * 90)
print("4. MAPPING MODEL CONGESTED LINES → NESO CONSTRAINT BOUNDARIES")
print("=" * 90)

print("""
NESO Constraint Boundaries (from thermal costs data):
  SCOTEX  = Scotland Export (B6 boundary)
  SSE-SP  = SSE to SP Transfer
  SSHARN  = South Harnser (Norfolk area)
  SWALEX  = South Wales Export  
  ESTEX   = East to Southeast Export
  SEIMP   = Southeast Import
  
  (From day-ahead data, additional boundaries:)
  ERROEX   = Errochty Export (Highland hydro area)
  FLOWSTH  = Flow South  
  GALLEX   = Galloway Export
  SPANOREX = SPA/North Export
  SSEN-S   = SSE-N to South
  SHARN    = Harnser (Norfolk)
  
Model's Top Congested Lines (from Validation_2020):
  1. ERRO1T_KIIN1-_0 (132 MVA, 55.2% congested) → ERROEX boundary
  2. MAHI3-_TRLO3-_0 (40 MVA, 38.6% congested)  → 33kV sub-transmission (ARTIFACT?)
  3. BREC1R_DENS1Q_0 (112 MVA, 38.2% congested)  → Perthshire area
  4. GLGL3-_WHLL3-_0 (30 MVA, 36.3% congested)  → 33kV sub-transmission (ARTIFACT?)
  5. WADW21_WACW21_0 (220 MVA, 34.0% congested)  → Walney offshore wind
  6. NORT41_OSBA42_0 (2009 MVA, 29.0% congested) → Norton-Osbaldwick (close to FLOWSTH?)
  7. HARR3-_KYPE3A_0 (40 MVA, 25.5% congested)  → 33kV sub-transmission (ARTIFACT?)
  8. CONQ41_FLIB41_0 (1438 MVA, 20.9% congested) → Connahs Quay-Flintshire Bridge (North Wales)
""")

print("""
KEY OBSERVATIONS:
1. ERROEX is a REAL NESO constraint boundary — our #1 congested line ERRO1T_KIIN1- maps to it
2. Three of our top 8 congested lines are 33kV sub-transmission (MAHI, GLGL, HARR) — these
   are likely ARTIFACTS of including 33kV lines in a transmission model
3. WADW21_WACW21 (Walney wind, 659 MW behind 220 MVA line) is a REAL constraint — 
   large offshore wind farm connected via undersized export cable
4. NORT41_OSBA42 (2009 MVA, 29% congested) — Norton-Osbaldwick 400kV circuit
5. CONQ41_FLIB41 (Connahs Quay 1380 MW CCGT behind 1438 MVA line) — close to capacity
""")

if __name__ == '__main__':
    pass
