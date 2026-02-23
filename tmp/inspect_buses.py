"""Temporary script to inspect bus connections for EAAW, BEIW, MOWE."""
import pypsa
import pandas as pd

n = pypsa.Network('resources/network/Historical_2023_etys_network.nc')

print(f"Network: {len(n.buses)} buses, {len(n.lines)} lines, {len(n.transformers)} xfmrs, {len(n.links)} links")
print(f"Generators: {len(n.generators)}, Storage: {len(n.storage_units)}")

# Check all links (interconnectors)
print(f"\n=== ALL LINKS ({len(n.links)}) ===")
for idx, lk in n.links.iterrows():
    print(f"  {idx}: {lk.bus0} -> {lk.bus1} (p_nom={lk.p_nom:.0f}, carrier={lk.carrier})")

# Check buses with EAAW, BEIW, MOWE prefixes
for prefix in ['EAAW', 'BEIW', 'MOWE', 'BEAT', 'BLHI', 'BRST']:
    buses = [b for b in n.buses.index if b.startswith(prefix)]
    if not buses:
        continue
    print(f'\n=== {prefix} buses ===')
    for b in buses:
        bus = n.buses.loc[b]
        print(f'  {b}: v_nom={bus.v_nom}, x={bus.x:.0f}, y={bus.y:.0f}')
        # Lines
        for _, l in n.lines[n.lines['bus0'] == b].iterrows():
            print(f'    Line -> {l.bus1} (s_nom={l.s_nom:.0f})')
        for _, l in n.lines[n.lines['bus1'] == b].iterrows():
            print(f'    Line <- {l.bus0} (s_nom={l.s_nom:.0f})')
        # Transformers
        for _, t in n.transformers[n.transformers['bus0'] == b].iterrows():
            print(f'    Xfmr -> {t.bus1} (s_nom={t.s_nom:.0f})')
        for _, t in n.transformers[n.transformers['bus1'] == b].iterrows():
            print(f'    Xfmr <- {t.bus0} (s_nom={t.s_nom:.0f})')
        # Links
        for _, lk in n.links[n.links['bus0'] == b].iterrows():
            print(f'    Link -> {lk.bus1} (p_nom={lk.p_nom:.0f})')
        for _, lk in n.links[n.links['bus1'] == b].iterrows():
            print(f'    Link <- {lk.bus0} (p_nom={lk.p_nom:.0f})')

# Also check what generator carriers exist
print(f"\n=== Generator carriers ===")
print(n.generators.groupby('carrier')['p_nom'].agg(['count', 'sum']).to_string())

# Check generators that have x,y vs those that don't
has_xy = n.generators[['x','y']].notna().all(axis=1).sum() if 'x' in n.generators.columns and 'y' in n.generators.columns else 0
print(f"\nGenerators with x,y coordinates: {has_xy} / {len(n.generators)}")
