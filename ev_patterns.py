"""Compare EV flexibility behavior across the week."""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('EV CHARGING PATTERNS ANALYSIS')
print('=' * 80)

# Get charging for each mode
go_chargers = n.links[n.links.index.str.contains('charger GO')]
int_chargers = n.links[n.links.index.str.contains('charger INT')]
v2g_chargers = n.links[n.links.index.str.contains('charger V2G')]

go_cols = [c for c in n.links_t.p0.columns if c in go_chargers.index]
int_cols = [c for c in n.links_t.p0.columns if c in int_chargers.index]
v2g_cols = [c for c in n.links_t.p0.columns if c in v2g_chargers.index]

go_charge = n.links_t.p0[go_cols].sum(axis=1)
int_charge = n.links_t.p0[int_cols].sum(axis=1)
v2g_charge = n.links_t.p0[v2g_cols].sum(axis=1)

# V2G discharge
v2g_links = n.links[n.links.carrier == 'V2G']
v2g_dcols = [c for c in n.links_t.p0.columns if c in v2g_links.index]
v2g_discharge = n.links_t.p0[v2g_dcols].sum(axis=1)

# System price proxy - use marginal generator
wind_offshore = n.generators[n.generators.carrier == 'wind_offshore']
wind_avail = (wind_offshore.p_nom * n.generators_t.p_max_pu[wind_offshore.index]).sum(axis=1)
wind_gen = n.generators_t.p[wind_offshore.index].sum(axis=1)
wind_curtailed = wind_avail - wind_gen

# Load shedding
ls_gens = n.generators[n.generators.carrier == 'load_shedding']
ls_total = n.generators_t.p[ls_gens.index].sum(axis=1)

# ============================================================================
# Hourly summary
# ============================================================================
print('\n1. CHARGING BY HOUR OF DAY')
print('-' * 40)

hourly_go = go_charge.groupby(go_charge.index.hour).mean()
hourly_int = int_charge.groupby(int_charge.index.hour).mean()
hourly_v2g = v2g_charge.groupby(v2g_charge.index.hour).mean()
hourly_v2g_d = v2g_discharge.groupby(v2g_discharge.index.hour).mean()
hourly_curt = wind_curtailed.groupby(wind_curtailed.index.hour).mean()
hourly_ls = ls_total.groupby(ls_total.index.hour).mean()

print(f'{"Hour":>4} {"GO":>8} {"INT":>8} {"V2G chg":>8} {"V2G dis":>8} {"Curtail":>10} {"LoadShed":>10}')
print('-' * 60)
for hour in range(24):
    print(f'{hour:>4} {hourly_go.get(hour, 0):>8,.0f} {hourly_int.get(hour, 0):>8,.0f} '
          f'{hourly_v2g.get(hour, 0):>8,.0f} {hourly_v2g_d.get(hour, 0):>8,.0f} '
          f'{hourly_curt.get(hour, 0):>10,.0f} {hourly_ls.get(hour, 0):>10,.0f}')

# ============================================================================
# Peak vs off-peak
# ============================================================================
print('\n2. PEAK VS OFF-PEAK COMPARISON')
print('-' * 40)

peak_hours = [17, 18, 19, 20, 21]  # 5pm - 10pm
offpeak_hours = [0, 1, 2, 3, 4, 5]  # midnight - 6am

peak_mask = go_charge.index.hour.isin(peak_hours)
offpeak_mask = go_charge.index.hour.isin(offpeak_hours)

print('Average charging (MW):')
print(f'  GO peak: {go_charge[peak_mask].mean():,.0f}  | GO off-peak: {go_charge[offpeak_mask].mean():,.0f}')
print(f'  INT peak: {int_charge[peak_mask].mean():,.0f} | INT off-peak: {int_charge[offpeak_mask].mean():,.0f}')
print(f'  V2G peak: {v2g_charge[peak_mask].mean():,.0f} | V2G off-peak: {v2g_charge[offpeak_mask].mean():,.0f}')

print('\nAverage wind curtailment (MW):')
print(f'  Peak: {wind_curtailed[peak_mask].mean():,.0f} | Off-peak: {wind_curtailed[offpeak_mask].mean():,.0f}')

print('\nAverage load shedding (MW):')
print(f'  Peak: {ls_total[peak_mask].mean():,.0f} | Off-peak: {ls_total[offpeak_mask].mean():,.0f}')

# ============================================================================
# Check if GO tariff is working
# ============================================================================
print('\n3. GO TARIFF WINDOW BEHAVIOR')
print('-' * 40)

go_window = [0, 1, 2, 3]  # midnight - 4am
window_mask = go_charge.index.hour.isin(go_window)

go_in_window = go_charge[window_mask].sum()
go_out_window = go_charge[~window_mask].sum()
total_go = go_in_window + go_out_window

print(f'GO charging in window (00:00-04:00): {go_in_window:,.0f} MWh ({go_in_window/total_go*100:.1f}%)')
print(f'GO charging outside window: {go_out_window:,.0f} MWh ({go_out_window/total_go*100:.1f}%)')

# This is because outside the window it costs £100/MWh, so only happens if necessary

# ============================================================================
# Energy balance check
# ============================================================================
print('\n4. EV ENERGY BALANCE')
print('-' * 40)

# EV driving demand
ev_loads = n.loads[n.loads.carrier == 'EV driving']
ev_demand = n.loads_t.p_set[ev_loads.index].sum().sum()

# Total EV charging
total_charge = go_charge.sum() + int_charge.sum() + v2g_charge.sum()

# EV load shedding
ev_ls = n.generators_t.p[ls_gens[ls_gens.bus.str.contains('EV battery')].index].sum().sum()

print(f'EV driving demand: {ev_demand:,.0f} MWh')
print(f'EV charging supplied: {total_charge:,.0f} MWh')
print(f'EV load shedding: {ev_ls:,.0f} MWh')
print(f'Balance: demand - (charge - ls) = {ev_demand - total_charge + ev_ls:,.0f} MWh')

# ============================================================================
# What's causing the shortage?
# ============================================================================
print('\n5. SYSTEM SHORTAGE ANALYSIS')
print('-' * 40)

total_demand = n.loads_t.p_set.sum(axis=1)
total_gen_cap = n.generators.p_nom.sum()
available_gen = (n.generators.p_nom * n.generators_t.p_max_pu).sum(axis=1)

print(f'Total installed generation: {total_gen_cap:,.0f} MW')
print(f'Average available generation: {available_gen.mean():,.0f} MW')
print(f'Average demand: {total_demand.mean():,.0f} MW')
print(f'Peak demand: {total_demand.max():,.0f} MW')
print(f'Min available at peak demand: {available_gen.loc[total_demand.idxmax()]:,.0f} MW')

# Interconnectors
imports = n.links[n.links.carrier == 'DC']
if len(imports) > 0:
    import_flow = n.links_t.p0[imports.index].sum(axis=1)
    print(f'\nInterconnector imports:')
    print(f'  Peak: {import_flow.max():,.0f} MW')
    print(f'  Mean: {import_flow.mean():,.0f} MW')
    print(f'  Total capacity: {imports.p_nom.sum():,.0f} MW')

print('\n' + '=' * 80)
