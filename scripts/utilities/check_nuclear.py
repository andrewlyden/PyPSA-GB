"""Investigate ESPENI nuclear profile variability."""
import pandas as pd
import numpy as np
import sys

esp = pd.read_csv('data/demand/espeni.csv')
ts_col = 'ELEC_elex_startTime[utc](datetime)'
esp[ts_col] = pd.to_datetime(esp[ts_col], utc=True)

# Nuclear column
nuc_col = [c for c in esp.columns if 'NUCLEAR' in c.upper()]
print('Nuclear columns:', nuc_col)
print()

# --- January 2020 ---
mask = (esp[ts_col] >= '2020-01-01') & (esp[ts_col] < '2020-02-01')
sub = esp[mask].copy().set_index(ts_col)
nuc = sub[nuc_col[0]]

print('=== ESPENI Nuclear Jan 2020 (half-hourly) ===')
print(f'Mean:  {nuc.mean():.0f} MW')
print(f'Min:   {nuc.min():.0f} MW')
print(f'Max:   {nuc.max():.0f} MW')
print(f'Std:   {nuc.std():.0f} MW')
print(f'CV:    {nuc.std()/nuc.mean():.3f}')
print()

# Hourly resample
nuc_h = nuc.resample('h').mean()

# Daily averages
nuc_d = nuc.resample('D').mean()
print('=== Daily Average Nuclear Jan 2020 (MW) ===')
for dt, val in nuc_d.items():
    print(f'  {dt.strftime("%Y-%m-%d")}: {val:7.0f} MW')
print()

# Step changes > 100 MW (hourly)
diff = nuc_h.diff().abs()
big_changes = diff[diff > 100]
print(f'=== Step changes > 100 MW (hourly, Jan 2020) ===')
print(f'Count: {len(big_changes)}')
for dt, val in big_changes.items():
    print(f'  {dt.strftime("%Y-%m-%d %H:%M")}: {val:.0f} MW change -> {nuc_h.loc[dt]:.0f} MW')
print()

# --- Full year 2020 ---
mask2 = (esp[ts_col] >= '2020-01-01') & (esp[ts_col] < '2021-01-01')
sub2 = esp[mask2].copy().set_index(ts_col)
nuc2 = sub2[nuc_col[0]]

nuc_w = nuc2.resample('W').mean()
print('=== Weekly Average Nuclear 2020 (MW) ===')
for dt, val in nuc_w.items():
    print(f'  {dt.strftime("%Y-%m-%d")}: {val:7.0f} MW')
print()
print(f'Annual mean: {nuc2.mean():.0f} MW')
print(f'Annual min:  {nuc2.min():.0f} MW')
print(f'Annual max:  {nuc2.max():.0f} MW')
print(f'Annual std:  {nuc2.std():.0f} MW')
print(f'Annual CV:   {nuc2.std()/nuc2.mean():.3f}')
print()

# --- Check what the model does ---
# Jan 7-8 test period
mask3 = (esp[ts_col] >= '2020-01-07') & (esp[ts_col] < '2020-01-09')
sub3 = esp[mask3].copy().set_index(ts_col)
nuc3 = sub3[nuc_col[0]].resample('h').mean()
print('=== ESPENI Nuclear Jan 7-8 2020 (hourly) ===')
for dt, val in nuc3.items():
    print(f'  {dt.strftime("%Y-%m-%d %H:%M")}: {val:7.0f} MW')
print()

# --- Compare with model ---
try:
    import pypsa
    n = pypsa.Network('resources/network/Test_Rolling_Market_solved.nc')
    nuc_gens = n.generators[n.generators.carrier == 'nuclear']
    print(f'=== Model Nuclear Generators ===')
    print(f'Count: {len(nuc_gens)}')
    print(f'Total p_nom: {nuc_gens.p_nom.sum():.0f} MW')
    print()
    
    # Check if p_max_pu varies
    nuc_in_pmax = [g for g in nuc_gens.index if g in n.generators_t.p_max_pu.columns]
    if nuc_in_pmax:
        pmax = n.generators_t.p_max_pu[nuc_in_pmax]
        print(f'Nuclear p_max_pu time series: {len(nuc_in_pmax)} generators have varying profiles')
        print(f'  Mean: {pmax.mean().mean():.3f}')
        print(f'  Min:  {pmax.min().min():.3f}')
        print(f'  Max:  {pmax.max().max():.3f}')
    else:
        print('Nuclear p_max_pu: STATIC (no time-varying profiles)')
        print(f'  p_max_pu values: {nuc_gens.p_max_pu.unique()}')
    print()
    
    # Model dispatch for Jan 7-8
    if hasattr(n, 'generators_t') and 'p' in dir(n.generators_t):
        nuc_dispatch = n.generators_t.p[nuc_gens.index].sum(axis=1)
        print('=== Model Nuclear Dispatch Jan 7-8 ===')
        for dt, val in nuc_dispatch.items():
            print(f'  {dt}: {val:7.0f} MW')
        print()
        print(f'Model nuclear mean: {nuc_dispatch.mean():.0f} MW')
        print(f'Model nuclear std:  {nuc_dispatch.std():.0f} MW')
        
        # Compare with ESPENI for same period
        model_total = nuc_dispatch.sum()
        espeni_total = nuc3.sum()
        print(f'Model total: {model_total:.0f} MWh')
        print(f'ESPENI total: {espeni_total:.0f} MWh')
        print(f'Ratio: {model_total/espeni_total:.3f}')
except Exception as e:
    print(f'Could not load model network: {e}')

# --- Identify GB nuclear stations operating in 2020 ---
print()
print('=== GB Nuclear Fleet in 2020 ===')
print('Station              | Type  | Capacity (MW) | Status in Jan 2020')
print('-' * 75)
stations = [
    ('Hinkley Point B', 'AGR', 965, 'Operating (reduced)'),
    ('Hunterston B', 'AGR', 965, 'Operating (reduced, cracks found 2018)'),
    ('Torness', 'AGR', 1185, 'Operating'),
    ('Heysham 1', 'AGR', 1150, 'Operating'),
    ('Heysham 2', 'AGR', 1220, 'Operating'),
    ('Hartlepool', 'AGR', 1185, 'Operating'),
    ('Dungeness B', 'AGR', 1040, 'Operating (reduced, extended outage history)'),
    ('Sizewell B', 'PWR', 1198, 'Operating'),
]
total_cap = 0
for name, typ, cap, status in stations:
    print(f'{name:<20} | {typ:<5} | {cap:>13} | {status}')
    total_cap += cap
print(f'{"TOTAL":<20} | {"":5} | {total_cap:>13} |')
print()
print('Note: Several AGR stations had reduced output due to graphite core')
print('cracking inspections (Hunterston B, Hinkley Point B, Dungeness B).')
print('This creates week-to-week variability that a flat p_max_pu cannot capture.')
