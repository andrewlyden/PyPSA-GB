"""Deep dive into specific congested lines."""
import pypsa
import pandas as pd

n = pypsa.Network('resources/market/Validation_2020_balancing.nc')

# ═══ 33kV congested lines ═══
print("=== 33kV CONGESTED LINE DETAILS ===")
for line_name in ['MAHI3-_TRLO3-_0', 'GLGL3-_WHLL3-_0', 'HARR3-_KYPE3A_0']:
    line = n.lines.loc[line_name]
    print(f"\n{line_name}:")
    for attr in ['bus0', 'bus1', 's_nom', 'v_nom', 'x', 'r', 'length', 'num_parallel', 'type', 's_nom_extendable']:
        if attr in line.index:
            print(f"  {attr}: {line[attr]}")

# ═══ Buses for 33kV lines ═══
print("\n=== BUSES FOR 33kV LINES ===")
for bus in ['MAHI3-', 'TRLO3-', 'GLGL3-', 'WHLL3-', 'HARR3-', 'KYPE3A']:
    if bus in n.buses.index:
        b = n.buses.loc[bus]
        v = b.get('v_nom', '?')
        print(f"  {bus}: v_nom={v} kV, x={b['x']:.0f}, y={b['y']:.0f}")

# ═══ Walney constraint ═══
print("\n=== WALNEY (WADW21) CONSTRAINT ===")
line = n.lines.loc['WADW21_WACW21_0']
for attr in ['bus0', 'bus1', 's_nom', 'v_nom', 'x', 'r', 'length', 'num_parallel', 'type']:
    if attr in line.index:
        print(f"  {attr}: {line[attr]}")

wadw_lines = n.lines[(n.lines.bus0 == 'WADW21') | (n.lines.bus1 == 'WADW21')]
wadw_xfmr = n.transformers[(n.transformers.bus0 == 'WADW21') | (n.transformers.bus1 == 'WADW21')]
print(f"\n  Lines at WADW21: {len(wadw_lines)}")
for idx, l in wadw_lines.iterrows():
    print(f"    {idx}: {l['bus0']} -> {l['bus1']}, s_nom={l['s_nom']:.0f}, v_nom={l['v_nom']:.0f}")
print(f"  Transformers at WADW21: {len(wadw_xfmr)}")
for idx, t in wadw_xfmr.iterrows():
    print(f"    {idx}: {t['bus0']} -> {t['bus1']}, s_nom={t['s_nom']:.0f}")

flow = n.lines_t.p0['WADW21_WACW21_0']
print(f"\n  Flow: mean={flow.mean():.0f} MW, max={flow.max():.0f}, min={flow.min():.0f}")
congested = (flow.abs() / 220 > 0.99).sum()
print(f"  Hours at limit: {congested} ({congested/8784*100:.1f}%)")

# ═══ Errochty network trace ═══
print("\n=== ERROCHTY (ERRO1T) - TRACE ISLAND ===")
line = n.lines.loc['ERRO1T_KIIN1-_0']
for attr in ['bus0', 'bus1', 's_nom', 'v_nom', 'x', 'r', 'length', 'num_parallel']:
    if attr in line.index:
        print(f"  {attr}: {line[attr]}")

visited = set()
queue = ['ERRO1T']
gen_behind = []
while queue:
    bus = queue.pop(0)
    if bus in visited:
        continue
    visited.add(bus)
    gens_here = n.generators[(n.generators.bus == bus) & (n.generators.carrier != 'load_shedding')]
    for _, g in gens_here.iterrows():
        gen_behind.append((g.name, g['carrier'], g['p_nom'], bus))
    adj_lines = n.lines[(n.lines.bus0 == bus) | (n.lines.bus1 == bus)]
    for idx, l in adj_lines.iterrows():
        if idx == 'ERRO1T_KIIN1-_0':
            continue
        next_bus = l['bus1'] if l['bus0'] == bus else l['bus0']
        if next_bus not in visited:
            queue.append(next_bus)
    adj_xfmr = n.transformers[(n.transformers.bus0 == bus) | (n.transformers.bus1 == bus)]
    for idx, t in adj_xfmr.iterrows():
        next_bus = t['bus1'] if t['bus0'] == bus else t['bus0']
        if next_bus not in visited:
            queue.append(next_bus)

print(f"\n  Buses behind ERRO1T: {len(visited)}")
print(f"  Generators: {len(gen_behind)}")
total_cap = sum(g[2] for g in gen_behind)
print(f"  Total capacity: {total_cap:.0f} MW")
for name, carrier, pnom, bus in gen_behind:
    print(f"    {name} ({carrier}): {pnom:.0f} MW at {bus}")

# ═══ Check NORT41-OSBA42 (high s_nom but still congested) ═══
print("\n=== NORTON-OSBALDWICK (NORT41_OSBA42_0) ===")
line = n.lines.loc['NORT41_OSBA42_0']
for attr in ['bus0', 'bus1', 's_nom', 'v_nom', 'x', 'r', 'length', 'num_parallel']:
    if attr in line.index:
        print(f"  {attr}: {line[attr]}")

flow = n.lines_t.p0['NORT41_OSBA42_0']
print(f"\n  Flow: mean={flow.mean():.0f} MW, max={flow.max():.0f}, min={flow.min():.0f}")

# What's on either side?
for bus in ['NORT41', 'OSBA42']:
    gens = n.generators[(n.generators.bus == bus) & (n.generators.carrier != 'load_shedding')]
    loads = n.loads[n.loads.bus == bus]
    if len(gens) > 0 or len(loads) > 0:
        gen_total = gens.p_nom.sum()
        load_total = 0
        for l_name in loads.index:
            if l_name in n.loads_t.p_set.columns:
                load_total = n.loads_t.p_set[l_name].mean()
        print(f"  {bus}: gen={gen_total:.0f} MW, load={load_total:.0f} MW avg")
    # How many connections?
    blines = n.lines[(n.lines.bus0 == bus) | (n.lines.bus1 == bus)]
    bxfmr = n.transformers[(n.transformers.bus0 == bus) | (n.transformers.bus1 == bus)]
    total_line_cap = blines['s_nom'].sum()
    total_xfmr_cap = bxfmr['s_nom'].sum()
    print(f"  {bus}: {len(blines)} lines ({total_line_cap:.0f} MVA), {len(bxfmr)} xfmrs ({total_xfmr_cap:.0f} MVA)")

# ═══ Summary: What % of BM cost comes from each boundary? ═══
print("\n=== MODEL BM COST ATTRIBUTION (rough estimate from price spreads) ===")
# Price spread * flow gives rough constraint cost per line
for line_name in ['ERRO1T_KIIN1-_0', 'MAHI3-_TRLO3-_0', 'GLGL3-_WHLL3-_0',
                  'BREC1R_DENS1Q_0', 'NORT41_OSBA42_0', 'HARR3-_KYPE3A_0',
                  'WADW21_WACW21_0', 'CONQ41_FLIB41_1', 'DRAX41_EGGB42_0',
                  'CRUA2Q_DALL2-_0']:
    if line_name not in n.lines.index:
        continue
    line = n.lines.loc[line_name]
    bus0, bus1 = line['bus0'], line['bus1']
    if bus0 in n.buses_t.marginal_price.columns and bus1 in n.buses_t.marginal_price.columns:
        spread = n.buses_t.marginal_price[bus0] - n.buses_t.marginal_price[bus1]
        flow_mw = n.lines_t.p0[line_name]
        # Constraint cost proxy = |spread| * |flow| / 1e6 (rough £M)
        line_cost = (spread.abs() * flow_mw.abs()).sum() / 1e6
        print(f"  {line_name}: ~GBP {line_cost:.1f}M (spread*flow proxy)")
