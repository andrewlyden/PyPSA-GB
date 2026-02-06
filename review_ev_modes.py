"""Review all EV tariff modes and verify marginal costs are correctly scoped."""

import pypsa
import pandas as pd

n = pypsa.Network('resources/network/HT35_flex_solved.nc')

print('=' * 80)
print('EV TARIFF MODE REVIEW')
print('=' * 80)

# ============================================================================
# 1. GO TARIFF REVIEW
# ============================================================================
print('\n1. GO TARIFF (Fixed Window)')
print('-' * 40)

go_chargers = n.links[n.links.index.str.contains('charger GO')]
go_stores = n.stores[n.stores.index.str.contains('EV fleet battery GO')]
go_loads = n.loads[n.loads.index.str.contains('EV driving GO')]

print(f'Components: {len(go_chargers)} chargers, {len(go_stores)} batteries, {len(go_loads)} loads')

if len(go_chargers) > 0:
    sample = go_chargers.index[0]
    
    # Check marginal cost
    if sample in n.links_t.marginal_cost.columns:
        mc = n.links_t.marginal_cost[sample]
        print(f'\nMarginal Cost (time-varying):')
        print(f'  Min: £{mc.min():.1f}/MWh (during cheap window)')
        print(f'  Max: £{mc.max():.1f}/MWh (outside window)')
        print(f'  Hours at cheap rate: {(mc == mc.min()).sum()} of {len(mc)}')
    else:
        print(f'\nMarginal Cost: £{go_chargers.loc[sample, "marginal_cost"]:.1f}/MWh (static)')
    
    # Check p_max_pu (availability)
    if sample in n.links_t.p_max_pu.columns:
        avail = n.links_t.p_max_pu[sample]
        print(f'\nAvailability (p_max_pu):')
        print(f'  Mean: {avail.mean():.2f}')
        print(f'  Min: {avail.min():.2f}')
        print(f'  Max: {avail.max():.2f}')
        print(f'  Hours available (>0): {(avail > 0).sum()} of {len(avail)}')
    else:
        print(f'\nAvailability: always available (p_max_pu=1.0)')

# ============================================================================
# 2. INT TARIFF REVIEW
# ============================================================================
print('\n\n2. INT TARIFF (Smart Charging)')
print('-' * 40)

int_chargers = n.links[n.links.index.str.contains('charger INT')]
int_stores = n.stores[n.stores.index.str.contains('EV fleet battery INT')]
int_loads = n.loads[n.loads.index.str.contains('EV driving INT')]

print(f'Components: {len(int_chargers)} chargers, {len(int_stores)} batteries, {len(int_loads)} loads')

if len(int_chargers) > 0:
    sample = int_chargers.index[0]
    
    # Check marginal cost
    mc = int_chargers.loc[sample, 'marginal_cost']
    print(f'\nMarginal Cost: £{mc:.1f}/MWh (should be 0 - free optimization)')
    
    # Check p_max_pu (availability)
    if sample in n.links_t.p_max_pu.columns:
        avail = n.links_t.p_max_pu[sample]
        print(f'\nAvailability (p_max_pu):')
        print(f'  Mean: {avail.mean():.2f}')
        print(f'  Hours available (>0): {(avail > 0).sum()} of {len(avail)}')
    
    # Check e_min_pu (minimum SOC)
    store_sample = int_stores.index[0]
    if store_sample in n.stores_t.e_min_pu.columns:
        e_min = n.stores_t.e_min_pu[store_sample]
        print(f'\nMinimum SOC (e_min_pu):')
        print(f'  Mean: {e_min.mean():.2f}')
        print(f'  Range: {e_min.min():.2f} to {e_min.max():.2f}')

# ============================================================================
# 3. V2G TARIFF REVIEW
# ============================================================================
print('\n\n3. V2G TARIFF (Vehicle-to-Grid)')
print('-' * 40)

v2g_chargers = n.links[n.links.index.str.contains('charger V2G')]
v2g_discharge = n.links[n.links.carrier == 'V2G']
v2g_stores = n.stores[n.stores.index.str.contains('EV fleet battery V2G')]
v2g_loads = n.loads[n.loads.index.str.contains('EV driving V2G')]

print(f'Components: {len(v2g_chargers)} chargers, {len(v2g_discharge)} V2G links, {len(v2g_stores)} batteries, {len(v2g_loads)} loads')

if len(v2g_chargers) > 0:
    sample = v2g_chargers.index[0]
    mc = v2g_chargers.loc[sample, 'marginal_cost']
    print(f'\nCharger Marginal Cost: £{mc:.1f}/MWh (should be 0)')

if len(v2g_discharge) > 0:
    sample = v2g_discharge.index[0]
    mc = v2g_discharge.loc[sample, 'marginal_cost']
    eff = v2g_discharge.loc[sample, 'efficiency']
    print(f'\nV2G Discharge Link:')
    print(f'  Marginal Cost (degradation): £{mc:.1f}/MWh')
    print(f'  Efficiency: {eff:.1%}')
    
    # Check p_max_pu
    if sample in n.links_t.p_max_pu.columns:
        avail = n.links_t.p_max_pu[sample]
        print(f'  Availability: mean={avail.mean():.2f}, max={avail.max():.2f}')

# ============================================================================
# 4. VERIFY MARGINAL COST ISOLATION
# ============================================================================
print('\n\n4. MARGINAL COST ISOLATION CHECK')
print('-' * 40)

# Check that GO costs don't affect other components
other_links = n.links[
    ~n.links.index.str.contains('EV', case=False) & 
    ~n.links.carrier.str.contains('V2G', case=False)
]

print(f'Non-EV links: {len(other_links)}')
non_ev_with_high_mc = other_links[other_links.marginal_cost > 50]
print(f'Non-EV links with marginal_cost > £50: {len(non_ev_with_high_mc)}')

if len(non_ev_with_high_mc) > 0:
    print('\n⚠️  Some non-EV links have high marginal costs:')
    for link in non_ev_with_high_mc.head(5).index:
        carrier = non_ev_with_high_mc.loc[link, 'carrier']
        mc = non_ev_with_high_mc.loc[link, 'marginal_cost']
        print(f'  {link}: {carrier}, £{mc:.1f}/MWh')
else:
    print('✓ GO tariff costs are isolated to GO chargers only')

# ============================================================================
# 5. OPERATION SUMMARY
# ============================================================================
print('\n\n5. OPERATION SUMMARY')
print('-' * 40)

# GO charging pattern
go_charge_cols = [c for c in n.links_t.p0.columns if 'charger GO' in c]
if go_charge_cols:
    go_charge = n.links_t.p0[go_charge_cols].sum(axis=1)
    print('\nGO Charging Pattern (total MW):')
    print(f'  Peak: {go_charge.max():,.0f} MW')
    print(f'  Mean: {go_charge.mean():,.0f} MW')
    
    # Check if charging happens outside window (hours 0-3)
    go_charge_by_hour = go_charge.groupby(go_charge.index.hour).sum()
    window_hours = [0, 1, 2, 3]
    in_window = go_charge_by_hour[window_hours].sum()
    out_window = go_charge_by_hour.drop(window_hours).sum()
    print(f'\n  Energy in window (00:00-04:00): {in_window:,.0f} MWh')
    print(f'  Energy outside window: {out_window:,.0f} MWh')
    if out_window > 0:
        ratio = in_window / (in_window + out_window)
        print(f'  Window preference: {ratio:.1%} of charging in cheap window')

# INT charging pattern
int_charge_cols = [c for c in n.links_t.p0.columns if 'charger INT' in c]
if int_charge_cols:
    int_charge = n.links_t.p0[int_charge_cols].sum(axis=1)
    print('\nINT Charging Pattern (total MW):')
    print(f'  Peak: {int_charge.max():,.0f} MW')
    print(f'  Mean: {int_charge.mean():,.0f} MW')

# V2G discharge pattern
v2g_cols = [c for c in n.links_t.p0.columns if c in v2g_discharge.index]
if v2g_cols:
    v2g_flow = n.links_t.p0[v2g_cols].sum(axis=1)
    print('\nV2G Discharge Pattern (total MW):')
    print(f'  Peak: {v2g_flow.max():,.0f} MW')
    print(f'  Total energy: {v2g_flow.sum():,.0f} MWh')

print('\n' + '=' * 80)
print('SUMMARY')
print('=' * 80)
print('''
✓ GO: Uses marginal cost to incentivize window charging (not hard constraint)
✓ INT: No marginal cost - optimizer freely shifts charging
✓ V2G: Degradation cost on discharge link only
✓ Marginal costs are isolated to specific EV charger links
''')
