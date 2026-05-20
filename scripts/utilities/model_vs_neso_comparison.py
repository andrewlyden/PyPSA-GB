"""Model congestion analysis vs NESO constraint data."""
import pandas as pd
import numpy as np
import pypsa

n = pypsa.Network('resources/market/Validation_2020_balancing.nc')

# ─── MODEL BM COST BY CONGESTED LINE ───────────────────────────────────────
lines = n.lines.copy()
p0 = n.lines_t.p0

congestion_hours = {}
for line_name in lines.index:
    s_nom = lines.loc[line_name, 's_nom']
    if s_nom <= 0:
        continue
    if line_name not in p0.columns:
        continue
    flow = p0[line_name].abs()
    loading = flow / s_nom
    congested = (loading > 0.99).sum()
    if congested > 0:
        congestion_hours[line_name] = {
            's_nom': s_nom,
            'hours_congested': congested,
            'pct': congested / len(n.snapshots) * 100,
            'voltage': lines.loc[line_name, 'v_nom'] if 'v_nom' in lines.columns else 0,
            'bus0': lines.loc[line_name, 'bus0'],
            'bus1': lines.loc[line_name, 'bus1'],
        }

cong_df = pd.DataFrame(congestion_hours).T.sort_values('hours_congested', ascending=False)

print("MODEL CONGESTED LINES (>99% loaded for >100 hours):")
header = f"{'Line':<25} {'s_nom':>8} {'V_nom':>6} {'Hours':>6} {'Pct':>6}"
print(header)
print("-" * 55)
for idx, row in cong_df.head(20).iterrows():
    if row['hours_congested'] > 100:
        print(f"  {idx:<23} {row['s_nom']:>8.0f} {row['voltage']:>6.0f} {row['hours_congested']:>6.0f} {row['pct']:>5.1f}%")

# ─── IDENTIFY 33kV ARTIFACT LINES ─────────────────────────────────────────
print(f"\n{'=' * 80}")
print("33kV LINES CAUSING ARTIFICIAL CONGESTION:")
print(f"{'=' * 80}")
low_v_congested = cong_df[(cong_df['voltage'] <= 33) & (cong_df['hours_congested'] > 100)]
total_curtailed_33kv = 0
for idx, row in low_v_congested.iterrows():
    for bus in [row['bus0'], row['bus1']]:
        gens = n.generators[n.generators.bus == bus]
        real_gens = gens[gens.carrier != 'load_shedding']
        if len(real_gens) > 0:
            cap = real_gens.p_nom.sum()
            carriers = ', '.join(f"{c}: {v:.0f}MW" for c, v in real_gens.groupby('carrier')['p_nom'].sum().items())
            other_lines = n.lines[(n.lines.bus0 == bus) | (n.lines.bus1 == bus)]
            other_xfmr = n.transformers[(n.transformers.bus0 == bus) | (n.transformers.bus1 == bus)]
            print(f"\n  {idx} ({row['s_nom']:.0f} MVA, {row['voltage']:.0f}kV)")
            print(f"    Hours congested: {row['hours_congested']:.0f} ({row['pct']:.1f}%)")
            print(f"    Bus {bus}: {cap:.0f} MW gen ({carriers})")
            print(f"    Connections: {len(other_lines)} lines, {len(other_xfmr)} transformers")
            
            for gen_name in real_gens.index:
                if gen_name in n.generators_t.p.columns:
                    gen_dispatch = n.generators_t.p[gen_name]
                    if gen_name in n.generators_t.p_max_pu.columns:
                        available = real_gens.loc[gen_name, 'p_nom'] * n.generators_t.p_max_pu[gen_name]
                    else:
                        available = real_gens.loc[gen_name, 'p_nom']
                    curtailed = available - gen_dispatch
                    curtailed_mwh = curtailed[curtailed > 0.1].sum()
                    if curtailed_mwh > 100:
                        print(f"    -> {gen_name}: {curtailed_mwh:.0f} MWh curtailed")
                        total_curtailed_33kv += curtailed_mwh

print(f"\n  TOTAL curtailment behind 33kV lines: {total_curtailed_33kv:.0f} MWh")

# ─── NODAL PRICE SPREAD ───────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("NODAL PRICE SPREAD ACROSS TOP CONGESTED LINES:")
print(f"{'=' * 80}")

if hasattr(n, 'buses_t') and not n.buses_t.marginal_price.empty:
    mp = n.buses_t.marginal_price
    for idx, row in cong_df.head(15).iterrows():
        if row['hours_congested'] > 100:
            bus0, bus1 = row['bus0'], row['bus1']
            if bus0 in mp.columns and bus1 in mp.columns:
                spread = (mp[bus0] - mp[bus1]).abs()
                nonzero = spread[spread > 0.01]
                if len(nonzero) > 0:
                    implied_cost = nonzero.sum()  # rough proxy
                    print(f"  {idx}: avg={nonzero.mean():.1f} GBP/MWh, max={nonzero.max():.1f}, hours={len(nonzero)}, sum_spread={implied_cost:.0f}")
else:
    print("No marginal prices available")

# ─── GENERATION BEHIND KEY CONGESTED LINES ─────────────────────────────────
print(f"\n{'=' * 80}")
print("GENERATION TRAPPED BEHIND CONGESTED BOUNDARIES:")
print(f"{'=' * 80}")

# For each congested line, trace the "island" behind it
key_lines = {
    'ERRO1T_KIIN1-_0': 'ERROEX (Highland hydro)',
    'WADW21_WACW21_0': 'Walney offshore wind',
    'NORT41_OSBA42_0': 'Norton-Osbaldwick 400kV',
    'CONQ41_FLIB41_1': 'Connahs Quay CCGT',
    'BREC1R_DENS1Q_0': 'Perthshire 132kV',
}

for line_name, desc in key_lines.items():
    if line_name not in lines.index:
        continue
    line = lines.loc[line_name]
    print(f"\n  {line_name} - {desc} ({line['s_nom']:.0f} MVA)")
    for bus_label, bus in [('bus0', line['bus0']), ('bus1', line['bus1'])]:
        gens = n.generators[n.generators.bus == bus]
        real_gens = gens[gens.carrier != 'load_shedding']
        if len(real_gens) > 0:
            for _, g in real_gens.iterrows():
                gen_name = g.name
                if gen_name in n.generators_t.p.columns:
                    dispatch = n.generators_t.p[gen_name].sum()
                    pnom = g['p_nom']
                    cf = dispatch / (pnom * len(n.snapshots))
                    print(f"    {bus} -> {gen_name} ({g['carrier']}, {pnom:.0f}MW): {dispatch:.0f} MWh dispatched, CF={cf:.1%}")

# ─── COMPARISON SUMMARY ───────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("NESO vs MODEL CONSTRAINT COMPARISON SUMMARY")
print(f"{'=' * 80}")

print("""
NESO REAL CONSTRAINTS (Calendar Year 2020):
  Total all constraints:     GBP 1,022.9M
  Thermal constraints only:  GBP   596.1M  (58% of total)
  Non-thermal (voltage/inertia/loss): GBP 426.8M (42%)
  
  Top Thermal Boundaries:
    SSHARN (Norfolk/East):  GBP 238.5M  (53% of thermal boundary total)
    SCOTEX (Scotland B6):   GBP 129.6M  (29%)
    SSE-SP (SSE to SP):     GBP  72.0M  (16%)
    SEIMP (Southeast):      GBP   7.9M  (2%)
    ESTEX (East Export):    GBP   2.2M  (<1%)
    SWALEX (South Wales):   GBP   0.1M  (<1%)
    Boundary subtotal:      GBP 450.3M
    (Gap to GBP 596M = unlisted smaller constraints)

MODEL CONSTRAINTS (Validation_2020):
  Total BM cost:             GBP ~126M  (from previous analysis)
  
  Model only captures THERMAL constraint resolution.
  NESO thermal cost = GBP 596.1M
  Model/NESO thermal ratio = 126/596 = 21.1%

  But NESO thermal includes maintenance outages we don't model!
  In reality, NESO's GBP 596M thermal cost is 'intact network + outaged network'.
  Our model = 'intact network only' = subset of thermal.

KEY FINDINGS:
  1. SSHARN (GBP 238.5M, 53%) — Norfolk wind constraint.
     Model has NO direct equivalent. SSHARN is a 'soft' boundary 
     managed through BM, not a single transmission line.
     
  2. SCOTEX (GBP 129.6M, 29%) — Scotland export (B6 boundary).
     Model captures this via B4/B6 corridor congestion.
     
  3. SSE-SP (GBP 72.0M, 16%) — SSE to SPT transfer.
     Model partially captures this.

  4. Model's top constraint ERRO1T_KIIN1- maps to ERROEX —
     a real but minor constraint boundary (not in top-6 costs).
     
  5. Three 33kV lines (MAHI, GLGL, HARR) are SUB-TRANSMISSION
     ARTIFACTS — these shouldn't appear in a transmission model.
     Each traps 70-81 MW of wind behind a 30-40 MVA bottleneck.

WHAT THE MODEL MISSES:
  a) Outage-related constraints (planned maintenance reduces capacity)
  b) SSHARN-type 'embedded' constraints (multiple parallel circuits,
     managed operationally, not one bottleneck line)
  c) Voltage constraints (GBP 76.1M)
  d) Inertia constraints (GBP 26.0M)
  e) Largest loss management (GBP 324.7M)

  Categories c-e are fundamentally outside DC-OPF scope.
  Category a could be addressed with outage schedules.
  Category b is a modeling limitation of nodal vs zonal approach.
""")
