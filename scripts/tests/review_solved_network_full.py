"""
Comprehensive review of a solved PyPSA-GB network.
Checks demand, generation, storage, interconnectors, network loading, DSR, and EV.
"""
import pypsa
import pandas as pd
import numpy as np
import sys

pd.set_option('display.float_format', lambda x: f'{x:,.1f}')
pd.set_option('display.max_rows', 60)
pd.set_option('display.width', 120)

network_path = sys.argv[1] if len(sys.argv) > 1 else "resources/network/HT35_flex_solved.nc"
n = pypsa.Network(network_path)

snapshots = n.snapshots
n_hours = len(snapshots)
print("=" * 90)
print(f"  COMPREHENSIVE NETWORK REVIEW: {network_path}")
print(f"  Period: {snapshots[0]} → {snapshots[-1]}  ({n_hours} hours)")
print("=" * 90)

# ═══════════════════════════════════════════════════════════════════════════
# 1. OPTIMISATION STATUS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  1. OPTIMISATION STATUS")
print("─" * 90)
status = getattr(n, 'status', 'unknown')
termination = getattr(n, 'termination_condition', 'unknown')
objective = getattr(n, 'objective', None)
# Try alternative attribute names
if objective is None:
    objective = getattr(n, 'objective_constant', None)
print(f"  Status:                {status}")
print(f"  Termination:           {termination}")
if objective is not None:
    print(f"  Objective (total cost): £{objective:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. NETWORK TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  2. NETWORK TOPOLOGY")
print("─" * 90)
print(f"  Buses:           {len(n.buses):,}")
print(f"  Lines:           {len(n.lines):,}")
print(f"  Transformers:    {len(n.transformers):,}")
print(f"  Links:           {len(n.links):,}")
print(f"  Generators:      {len(n.generators):,}")
print(f"  Loads:           {len(n.loads):,}")
print(f"  Storage units:   {len(n.storage_units):,}")
print(f"  Stores:          {len(n.stores):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. DEMAND
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  3. DEMAND")
print("─" * 90)

if not n.loads_t.p_set.empty:
    total_demand = n.loads_t.p_set.sum(axis=1)
    total_demand_gwh = total_demand.sum() / 1000
    print(f"  Total demand:     {total_demand_gwh:,.1f} GWh over {n_hours} hours")
    print(f"  Annualised:       {total_demand_gwh * 8760 / n_hours:,.0f} GWh/year")
    print(f"  Peak demand:      {total_demand.max():,.0f} MW")
    print(f"  Min demand:       {total_demand.min():,.0f} MW")
    print(f"  Mean demand:      {total_demand.mean():,.0f} MW")

    # Demand by carrier
    if 'carrier' in n.loads.columns:
        print("\n  Demand by carrier:")
        for carrier in n.loads['carrier'].unique():
            carrier_loads = n.loads[n.loads['carrier'] == carrier].index
            cols = [c for c in carrier_loads if c in n.loads_t.p_set.columns]
            if cols:
                carrier_gwh = n.loads_t.p_set[cols].sum().sum() / 1000
                carrier_peak = n.loads_t.p_set[cols].sum(axis=1).max()
                print(f"    {carrier:25s}: {carrier_gwh:8,.1f} GWh, peak {carrier_peak:,.0f} MW")
else:
    print("  ⚠️  No time-varying demand found!")

# ═══════════════════════════════════════════════════════════════════════════
# 4. GENERATION CAPACITY & DISPATCH
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  4. GENERATION — INSTALLED CAPACITY")
print("─" * 90)

gen_cap = n.generators.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
total_cap = gen_cap.sum()
print(f"  Total installed:  {total_cap:,.0f} MW  ({total_cap/1000:,.1f} GW)")
print()
print(f"  {'Carrier':25s}  {'Units':>6s}  {'Capacity (MW)':>14s}  {'Share':>6s}")
print(f"  {'─'*25}  {'─'*6}  {'─'*14}  {'─'*6}")
for carrier in gen_cap.index:
    units = (n.generators['carrier'] == carrier).sum()
    cap = gen_cap[carrier]
    share = cap / total_cap * 100
    print(f"  {carrier:25s}  {units:6d}  {cap:14,.0f}  {share:5.1f}%")

print("\n" + "─" * 90)
print("  5. GENERATION — DISPATCH (ENERGY)")
print("─" * 90)

if not n.generators_t.p.empty:
    gen_dispatch = n.generators_t.p.groupby(n.generators['carrier'], axis=1).sum() if hasattr(n.generators_t.p, 'groupby') else None
    
    # Manual groupby for newer pandas
    if gen_dispatch is None or gen_dispatch.empty:
        dispatch_by_carrier = {}
        for carrier in n.generators['carrier'].unique():
            gens = n.generators[n.generators['carrier'] == carrier].index
            cols = [g for g in gens if g in n.generators_t.p.columns]
            if cols:
                dispatch_by_carrier[carrier] = n.generators_t.p[cols].sum(axis=1)
        gen_dispatch = pd.DataFrame(dispatch_by_carrier)
    
    gen_energy = gen_dispatch.sum() / 1000  # GWh
    total_gen_gwh = gen_energy.sum()
    gen_energy = gen_energy.sort_values(ascending=False)
    
    print(f"  Total generation: {total_gen_gwh:,.1f} GWh")
    print()
    print(f"  {'Carrier':25s}  {'Energy (GWh)':>12s}  {'Share':>6s}  {'Peak (MW)':>10s}  {'CF':>6s}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*6}  {'─'*10}  {'─'*6}")
    
    for carrier in gen_energy.index:
        energy = gen_energy[carrier]
        share = energy / total_gen_gwh * 100
        peak = gen_dispatch[carrier].max()
        cap = gen_cap.get(carrier, 0)
        cf = energy * 1000 / (cap * n_hours) * 100 if cap > 0 else 0
        print(f"  {carrier:25s}  {energy:12,.1f}  {share:5.1f}%  {peak:10,.0f}  {cf:5.1f}%")
    
    # Supply-demand balance
    print(f"\n  Supply-demand balance:")
    print(f"    Total generation:   {total_gen_gwh:10,.1f} GWh")
    if not n.loads_t.p_set.empty:
        print(f"    Total demand:       {total_demand_gwh:10,.1f} GWh")
        print(f"    Ratio:              {total_gen_gwh / total_demand_gwh:10.3f}")
else:
    print("  ⚠️  No dispatch results found!")

# ═══════════════════════════════════════════════════════════════════════════
# 6. CURTAILMENT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  6. CURTAILMENT")
print("─" * 90)

renewables = ['wind_onshore', 'wind_offshore', 'solar_pv', 'large_hydro', 'marine']
for carrier in renewables:
    gens = n.generators[n.generators['carrier'] == carrier]
    if gens.empty:
        continue
    cols = [g for g in gens.index if g in n.generators_t.p.columns]
    if not cols:
        continue
    
    dispatched = n.generators_t.p[cols].sum().sum() / 1000
    
    # Calculate available energy
    p_max_cols = [g for g in cols if g in n.generators_t.p_max_pu.columns]
    if p_max_cols:
        available = sum(
            (n.generators_t.p_max_pu[g] * n.generators.loc[g, 'p_nom']).sum()
            for g in p_max_cols
        ) / 1000
        curtailed = available - dispatched
        curt_pct = curtailed / available * 100 if available > 0 else 0
        print(f"  {carrier:25s}: {dispatched:8,.1f} GWh dispatched / {available:8,.1f} GWh available  → {curt_pct:5.1f}% curtailed")
    else:
        print(f"  {carrier:25s}: {dispatched:8,.1f} GWh dispatched (no p_max_pu data)")

# ═══════════════════════════════════════════════════════════════════════════
# 7. LOAD SHEDDING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  7. LOAD SHEDDING")
print("─" * 90)

ls_gens = n.generators[n.generators['carrier'].isin(['load_shedding', 'load shedding'])]
if not ls_gens.empty:
    ls_cols = [g for g in ls_gens.index if g in n.generators_t.p.columns]
    if ls_cols:
        ls_dispatch = n.generators_t.p[ls_cols]
        ls_total = ls_dispatch.sum().sum() / 1000
        ls_peak = ls_dispatch.sum(axis=1).max()
        ls_hours = (ls_dispatch.sum(axis=1) > 0.1).sum()
        print(f"  Total load shedding:  {ls_total:,.1f} GWh")
        print(f"  Peak load shedding:   {ls_peak:,.0f} MW")
        print(f"  Hours with shedding:  {ls_hours} / {n_hours}")
        if ls_total > 0.01:
            print(f"  ⚠️  LOAD SHEDDING DETECTED — system cannot meet all demand!")
        else:
            print(f"  ✓ No load shedding — all demand met")
    else:
        print(f"  ✓ No load shedding dispatch")
else:
    print(f"  ⚠️  No load shedding generators found")

# ═══════════════════════════════════════════════════════════════════════════
# 8. DEMAND SIDE RESPONSE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  8. DEMAND SIDE RESPONSE (DSR)")
print("─" * 90)

dsr_gens = n.generators[n.generators['carrier'].isin(['demand response', 'demand_response'])]
if not dsr_gens.empty:
    dsr_mc = dsr_gens['marginal_cost'].unique()
    dsr_cap = dsr_gens['p_nom'].sum()
    print(f"  Units:              {len(dsr_gens)}")
    print(f"  Total capacity:     {dsr_cap:,.0f} MW")
    print(f"  Marginal cost:      £{dsr_mc[0]:,.0f}/MWh" if len(dsr_mc) == 1 else f"  Marginal costs: {dsr_mc}")
    
    dsr_cols = [g for g in dsr_gens.index if g in n.generators_t.p.columns]
    if dsr_cols:
        dsr_dispatch = n.generators_t.p[dsr_cols]
        dsr_total = dsr_dispatch.sum().sum() / 1000
        dsr_peak = dsr_dispatch.sum(axis=1).max()
        dsr_hours_active = (dsr_dispatch.sum(axis=1) > 0.1).sum()
        
        # Check event schedule
        dsr_pmax_cols = [g for g in dsr_gens.index if g in n.generators_t.p_max_pu.columns]
        if dsr_pmax_cols:
            event_hours = (n.generators_t.p_max_pu[dsr_pmax_cols[0]] > 0).sum()
            print(f"  Event hours:        {event_hours} / {n_hours}")
        
        utilisation = dsr_total * 1000 / (dsr_cap * n_hours) * 100 if dsr_cap > 0 else 0
        print(f"  Total dispatch:     {dsr_total:,.1f} GWh")
        print(f"  Peak dispatch:      {dsr_peak:,.0f} MW")
        print(f"  Hours dispatched:   {dsr_hours_active} / {n_hours}")
        print(f"  Overall CF:         {utilisation:.1f}%")
        
        if dsr_mc[0] < 1:
            print(f"  ⚠️  MARGINAL COST = £0 — DSR dispatching as free energy!")
        elif dsr_total < 0.001:
            print(f"  ℹ️  DSR not dispatched — system prices never exceeded £{dsr_mc[0]:,.0f}/MWh")
        elif dsr_hours_active > 0 and dsr_pmax_cols:
            dispatch_during_events = (dsr_dispatch.sum(axis=1) > 0.1) & (n.generators_t.p_max_pu[dsr_pmax_cols[0]] > 0)
            dispatch_outside_events = (dsr_dispatch.sum(axis=1) > 0.1) & (n.generators_t.p_max_pu[dsr_pmax_cols[0]] == 0)
            print(f"  Hours dispatched during events:  {dispatch_during_events.sum()}")
            print(f"  Hours dispatched outside events: {dispatch_outside_events.sum()}")
            if dispatch_outside_events.sum() > 0:
                print(f"  ⚠️  DSR dispatched outside event windows!")
        
        # Selective dispatch analysis
        if dsr_hours_active > 0 and event_hours > 0:
            selectivity = dsr_hours_active / event_hours * 100
            print(f"  Selectivity:        {selectivity:.0f}% of event hours used")
            if selectivity > 95:
                print(f"  ⚠️  Dispatching in nearly ALL event hours — marginal cost may be too low")
            else:
                print(f"  ✓ Selective dispatch — optimizer choosing when DSR is economic")
    else:
        print(f"  ⚠️  No dispatch data for DSR generators")
else:
    print(f"  ℹ️  No DSR generators in network")

# ═══════════════════════════════════════════════════════════════════════════
# 9. STORAGE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  9. STORAGE")
print("─" * 90)

if not n.storage_units.empty:
    su_cap = n.storage_units.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)
    print(f"  Total storage units:  {len(n.storage_units)}")
    print(f"  Total power capacity: {su_cap.sum():,.0f} MW ({su_cap.sum()/1000:,.1f} GW)")
    print()
    print(f"  {'Carrier':25s}  {'Units':>6s}  {'Power (MW)':>12s}  {'Max Hours':>10s}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*12}  {'─'*10}")
    for carrier in su_cap.index:
        units = (n.storage_units['carrier'] == carrier).sum()
        cap = su_cap[carrier]
        # Check max_hours
        carrier_su = n.storage_units[n.storage_units['carrier'] == carrier]
        max_h = carrier_su['max_hours'].mean() if 'max_hours' in carrier_su.columns else 0
        print(f"  {carrier:25s}  {units:6d}  {cap:12,.0f}  {max_h:10.1f}")
    
    # Storage dispatch
    if not n.storage_units_t.p.empty:
        su_dispatch = {}
        for carrier in n.storage_units['carrier'].unique():
            sus = n.storage_units[n.storage_units['carrier'] == carrier].index
            cols = [s for s in sus if s in n.storage_units_t.p.columns]
            if cols:
                ts = n.storage_units_t.p[cols].sum(axis=1)
                charging = ts[ts < 0].sum() / 1000  # GWh (negative = charging)
                discharging = ts[ts > 0].sum() / 1000  # GWh (positive = discharging)
                su_dispatch[carrier] = {'charging': charging, 'discharging': discharging}
        
        if su_dispatch:
            print(f"\n  Storage dispatch:")
            print(f"  {'Carrier':25s}  {'Charging (GWh)':>15s}  {'Discharging (GWh)':>18s}  {'Efficiency':>10s}")
            print(f"  {'─'*25}  {'─'*15}  {'─'*18}  {'─'*10}")
            for carrier, data in su_dispatch.items():
                eff = abs(data['discharging'] / data['charging']) * 100 if data['charging'] != 0 else 0
                print(f"  {carrier:25s}  {data['charging']:15,.1f}  {data['discharging']:18,.1f}  {eff:9.0f}%")
else:
    print(f"  ℹ️  No storage units in network")

# Also check Stores (used by EV model)
if not n.stores.empty:
    print(f"\n  Stores (energy buffers): {len(n.stores)}")
    if 'carrier' in n.stores.columns:
        store_cap = n.stores.groupby('carrier')['e_nom'].sum()
        for carrier in store_cap.index:
            units = (n.stores['carrier'] == carrier).sum()
            print(f"    {carrier:25s}: {units:4d} units, {store_cap[carrier]:,.0f} MWh capacity")

# ═══════════════════════════════════════════════════════════════════════════
# 10. INTERCONNECTORS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  10. INTERCONNECTORS")
print("─" * 90)

# Interconnectors can be links or lines connecting to non-GB buses
# Usually modelled as links with specific carrier names
ic_carriers = ['AC', 'DC', 'interconnector']
ic_links = n.links[n.links['carrier'].isin(ic_carriers)] if 'carrier' in n.links.columns else pd.DataFrame()

# Also look for links with "IC" or interconnector-like names
if ic_links.empty:
    ic_pattern = n.links.index[n.links.index.str.contains('IC|IFA|BritNed|Moyle|EWIC|NSL|Viking|ElecLink|NeuConnect|Nemo', case=False, na=False)]
    if len(ic_pattern) > 0:
        ic_links = n.links.loc[ic_pattern]

if not ic_links.empty:
    print(f"  Interconnector links: {len(ic_links)}")
    total_ic_cap = ic_links['p_nom'].sum()
    print(f"  Total capacity:       {total_ic_cap:,.0f} MW ({total_ic_cap/1000:,.1f} GW)")
    
    print()
    print(f"  {'Name':30s}  {'Capacity (MW)':>14s}  {'From':>15s}  {'To':>15s}")
    print(f"  {'─'*30}  {'─'*14}  {'─'*15}  {'─'*15}")
    for name, link in ic_links.iterrows():
        short_name = str(name)[:30]
        cap = link.get('p_nom', 0)
        bus0 = str(link.get('bus0', ''))[:15]
        bus1 = str(link.get('bus1', ''))[:15]
        print(f"  {short_name:30s}  {cap:14,.0f}  {bus0:>15s}  {bus1:>15s}")
    
    # IC dispatch
    ic_cols = [c for c in ic_links.index if c in n.links_t.p0.columns]
    if ic_cols:
        ic_p0 = n.links_t.p0[ic_cols]
        imports = ic_p0[ic_p0 > 0].sum().sum() / 1000
        exports = ic_p0[ic_p0 < 0].sum().sum() / 1000
        print(f"\n  Imports (into GB):    {imports:,.1f} GWh")
        print(f"  Exports (from GB):    {abs(exports):,.1f} GWh")
        print(f"  Net import:           {imports + exports:,.1f} GWh")
else:
    print(f"  ℹ️  No interconnectors found in links")

# ═══════════════════════════════════════════════════════════════════════════
# 11. ELECTRIC VEHICLES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  11. ELECTRIC VEHICLES")
print("─" * 90)

ev_stores = n.stores[n.stores.index.str.contains('EV', case=False)] if not n.stores.empty else pd.DataFrame()
ev_links = n.links[n.links.index.str.contains('EV', case=False)] if not n.links.empty else pd.DataFrame()
ev_loads = n.loads[n.loads.index.str.contains('EV', case=False)] if not n.loads.empty else pd.DataFrame()
ev_buses = n.buses[n.buses.index.str.contains('EV', case=False)]

if not ev_buses.empty or not ev_stores.empty or not ev_links.empty:
    print(f"  EV buses:   {len(ev_buses)}")
    print(f"  EV stores:  {len(ev_stores)}")
    print(f"  EV links:   {len(ev_links)}")
    print(f"  EV loads:   {len(ev_loads)}")
    
    if not ev_stores.empty and 'e_nom' in ev_stores.columns:
        total_ev_energy = ev_stores['e_nom'].sum()
        print(f"  Total EV battery capacity: {total_ev_energy:,.0f} MWh ({total_ev_energy/1000:,.0f} GWh)")
    
    if not ev_links.empty and 'p_nom' in ev_links.columns:
        charger_links = ev_links[ev_links.index.str.contains('charger', case=False)]
        if not charger_links.empty:
            total_charger = charger_links['p_nom'].sum()
            print(f"  Total charger capacity:    {total_charger:,.0f} MW ({total_charger/1000:,.1f} GW)")
    
    if not ev_loads.empty:
        ev_load_cols = [c for c in ev_loads.index if c in n.loads_t.p_set.columns]
        if ev_load_cols:
            total_ev_demand = n.loads_t.p_set[ev_load_cols].sum().sum() / 1000
            peak_ev_demand = n.loads_t.p_set[ev_load_cols].sum(axis=1).max()
            print(f"  Total EV demand:           {total_ev_demand:,.1f} GWh (weekly)")
            print(f"  Peak EV demand:            {peak_ev_demand:,.0f} MW")
            print(f"  Annualised EV demand:      {total_ev_demand * 8760 / n_hours:,.0f} GWh/year")
else:
    print(f"  ℹ️  No EV components in network")

# ═══════════════════════════════════════════════════════════════════════════
# 12. NETWORK LOADING (Lines & Transformers)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  12. NETWORK LOADING")
print("─" * 90)

# Lines
if not n.lines.empty and not n.lines_t.p0.empty:
    line_loading = n.lines_t.p0.abs()
    s_nom = n.lines['s_nom']
    
    # Percentage loading
    max_loading_pct = pd.Series(dtype=float)
    for line in line_loading.columns:
        if line in s_nom.index and s_nom[line] > 0:
            max_loading_pct[line] = line_loading[line].max() / s_nom[line] * 100
    
    if not max_loading_pct.empty:
        print(f"  Lines:")
        print(f"    Total lines:        {len(n.lines)}")
        print(f"    Mean max loading:   {max_loading_pct.mean():.1f}%")
        print(f"    Median max loading: {max_loading_pct.median():.1f}%")
        
        congested = (max_loading_pct > 95).sum()
        heavily_loaded = ((max_loading_pct > 80) & (max_loading_pct <= 95)).sum()
        print(f"    Congested (>95%):   {congested} lines")
        print(f"    Heavy (80-95%):     {heavily_loaded} lines")
        print(f"    Light (<50%):       {(max_loading_pct < 50).sum()} lines")
        
        if congested > 0:
            print(f"\n    Top 10 most loaded lines:")
            top = max_loading_pct.nlargest(10)
            for line, loading in top.items():
                bus0 = n.lines.loc[line, 'bus0'][:20] if line in n.lines.index else '?'
                bus1 = n.lines.loc[line, 'bus1'][:20] if line in n.lines.index else '?'
                s = s_nom[line] if line in s_nom.index else 0
                print(f"      {str(line)[:30]:30s}  {loading:5.1f}%  ({s:,.0f} MW)  {bus0} → {bus1}")

# Transformers
if not n.transformers.empty and not n.transformers_t.p0.empty:
    trafo_loading = n.transformers_t.p0.abs()
    trafo_s_nom = n.transformers['s_nom']
    
    max_trafo_pct = pd.Series(dtype=float)
    for trafo in trafo_loading.columns:
        if trafo in trafo_s_nom.index and trafo_s_nom[trafo] > 0:
            max_trafo_pct[trafo] = trafo_loading[trafo].max() / trafo_s_nom[trafo] * 100
    
    if not max_trafo_pct.empty:
        print(f"\n  Transformers:")
        print(f"    Total transformers: {len(n.transformers)}")
        print(f"    Mean max loading:   {max_trafo_pct.mean():.1f}%")
        congested_t = (max_trafo_pct > 95).sum()
        heavy_t = ((max_trafo_pct > 80) & (max_trafo_pct <= 95)).sum()
        print(f"    Congested (>95%):   {congested_t} transformers")
        print(f"    Heavy (80-95%):     {heavy_t} transformers")
        
        if congested_t > 0:
            print(f"\n    Top 10 most loaded transformers:")
            top_t = max_trafo_pct.nlargest(10)
            for trafo, loading in top_t.items():
                s = trafo_s_nom[trafo] if trafo in trafo_s_nom.index else 0
                print(f"      {str(trafo)[:40]:40s}  {loading:5.1f}%  ({s:,.0f} MVA)")

# ═══════════════════════════════════════════════════════════════════════════
# 13. SYSTEM MARGINAL PRICES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 90)
print("  13. SYSTEM MARGINAL PRICES")
print("─" * 90)

if not n.buses_t.marginal_price.empty:
    # Use electricity buses only
    elec_buses = n.buses[~n.buses.index.str.contains('EV|heat|H2', case=False)].index
    price_cols = [b for b in elec_buses if b in n.buses_t.marginal_price.columns]
    
    if price_cols:
        prices = n.buses_t.marginal_price[price_cols]
        system_price = prices.mean(axis=1)  # Average across buses
        
        print(f"  Mean system price:    £{system_price.mean():,.1f}/MWh")
        print(f"  Median price:         £{system_price.median():,.1f}/MWh")
        print(f"  Min price:            £{system_price.min():,.1f}/MWh")
        print(f"  Max price:            £{system_price.max():,.1f}/MWh")
        print(f"  Std deviation:        £{system_price.std():,.1f}/MWh")
        
        # Price duration
        high_price_hours = (system_price > 100).sum()
        zero_price_hours = (system_price < 1).sum()
        negative_hours = (system_price < 0).sum()
        print(f"\n  Price duration:")
        print(f"    > £100/MWh:         {high_price_hours} hours ({high_price_hours/n_hours*100:.1f}%)")
        print(f"    < £1/MWh:           {zero_price_hours} hours ({zero_price_hours/n_hours*100:.1f}%)")
        print(f"    < £0/MWh:           {negative_hours} hours ({negative_hours/n_hours*100:.1f}%)")
        
        # Price spread (max - min across buses at each hour)
        price_spread = prices.max(axis=1) - prices.min(axis=1)
        print(f"\n  Locational price spread:")
        print(f"    Mean spread:        £{price_spread.mean():,.1f}/MWh")
        print(f"    Max spread:         £{price_spread.max():,.1f}/MWh")
        if price_spread.max() > 50:
            max_spread_hour = price_spread.idxmax()
            print(f"    Max spread at:      {max_spread_hour}")
else:
    print(f"  ⚠️  No marginal prices — network may not be solved")

# ═══════════════════════════════════════════════════════════════════════════
# 14. SUMMARY & FLAGS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 90)
print("  14. SUMMARY — KEY FLAGS")
print("═" * 90)

issues = []
info = []

# Check load shedding
if not ls_gens.empty and ls_cols:
    if ls_total > 0.01:
        issues.append(f"🔴 Load shedding: {ls_total:,.1f} GWh ({ls_hours} hours)")

# Check DSR marginal cost
if not dsr_gens.empty:
    if dsr_mc[0] < 1:
        issues.append(f"🔴 DSR marginal cost = £{dsr_mc[0]:,.0f}/MWh (should be >£0)")
    elif dsr_mc[0] > 0:
        info.append(f"✅ DSR marginal cost = £{dsr_mc[0]:,.0f}/MWh")

# Check curtailment
for carrier in renewables:
    gens = n.generators[n.generators['carrier'] == carrier]
    if gens.empty:
        continue
    cols = [g for g in gens.index if g in n.generators_t.p.columns]
    p_max_cols = [g for g in cols if g in n.generators_t.p_max_pu.columns]
    if cols and p_max_cols:
        dispatched = n.generators_t.p[cols].sum().sum()
        available = sum(
            (n.generators_t.p_max_pu[g] * n.generators.loc[g, 'p_nom']).sum()
            for g in p_max_cols
        )
        if available > 0:
            curt_pct = (1 - dispatched / available) * 100
            if curt_pct > 20:
                issues.append(f"🟡 {carrier} curtailment: {curt_pct:.0f}%")
            elif curt_pct > 5:
                info.append(f"ℹ️  {carrier} curtailment: {curt_pct:.1f}%")

# Check network congestion
if not max_loading_pct.empty:
    if congested > 20:
        issues.append(f"🟡 {congested} congested lines (>95%)")
    elif congested > 0:
        info.append(f"ℹ️  {congested} congested lines (>95%)")

# Check demand vs generation
if not n.loads_t.p_set.empty and total_gen_gwh > 0:
    ratio = total_gen_gwh / total_demand_gwh
    if ratio < 0.98:
        issues.append(f"🔴 Generation < demand (ratio: {ratio:.3f})")
    elif ratio > 1.05:
        info.append(f"ℹ️  Generation > demand by {(ratio-1)*100:.1f}% (losses + exports)")

# EV check
if ev_buses.empty:
    info.append(f"ℹ️  No EV components (EV may be disabled in config)")

if issues:
    print("\n  ISSUES:")
    for issue in issues:
        print(f"    {issue}")

if info:
    print("\n  INFO:")
    for i in info:
        print(f"    {i}")

if not issues:
    print("\n  ✅ No critical issues detected")

print("\n" + "=" * 90)
print("  REVIEW COMPLETE")
print("=" * 90)
