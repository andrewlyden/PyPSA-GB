"""One-off script to identify PyPSA lines crossing NESO constraint boundaries."""
import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network('resources/network/Test_Rolling_Market_network.nc')
print(f'Lines: {len(n.lines)}, Buses: {len(n.buses)}')

lines = n.lines.copy()
lines['from_y'] = n.buses.loc[lines['bus0'], 'y'].values
lines['to_y'] = n.buses.loc[lines['bus1'], 'y'].values
lines['from_x'] = n.buses.loc[lines['bus0'], 'x'].values
lines['to_x'] = n.buses.loc[lines['bus1'], 'x'].values

def find_crossing_lines(df, y_threshold, min_s_nom=100, label=""):
    """Find lines that cross a latitude threshold."""
    mask = (
        ((df['from_y'] > y_threshold) & (df['to_y'] < y_threshold)) |
        ((df['to_y'] > y_threshold) & (df['from_y'] < y_threshold))
    )
    result = df[mask & (df['s_nom'] >= min_s_nom)].sort_values('s_nom', ascending=False)
    print(f"\n=== {label} (cross y~{y_threshold}, s_nom>={min_s_nom}) ===")
    for idx, row in result.head(25).iterrows():
        print(f"  {idx}: s_nom={row['s_nom']:.0f} MVA, "
              f"{row['bus0']}(y={row['from_y']:.0f}) -> {row['bus1']}(y={row['to_y']:.0f})")
    print(f"  Total: {result['s_nom'].sum():.0f} MVA across {len(result)} lines")
    return result

# SCOTEX (B6): Scotland-England border
# Official B6 boundary crosses between Harker/Hutton/Stella West (England) 
# and substations north of there
scotex = find_crossing_lines(lines, 585000, 100, "SCOTEX (B6)")

# SSE-SP: SSE Transmission / SP Transmission boundary
# Roughly the Highland boundary line near Beauly/Denny
ssesp = find_crossing_lines(lines, 710000, 100, "SSE-SP (B1/B2)")

# Also check HVDC links crossing boundaries
print("\n=== HVDC Links ===")
for idx, row in n.links.iterrows():
    if row.get('bus0', '') in n.buses.index and row.get('bus1', '') in n.buses.index:
        y0 = n.buses.loc[row['bus0'], 'y']
        y1 = n.buses.loc[row['bus1'], 'y']
        if abs(y0 - y1) > 50000:
            print(f"  {idx}: p_nom={row['p_nom']:.0f} MW, "
                  f"{row['bus0']}(y={y0:.0f}) -> {row['bus1']}(y={y1:.0f})")

# SSHARN: Lines in Norfolk/East Anglia area
# SSHARN is roughly the boundary around the Norfolk area
# Key substations: Norwich Main (NORM), Sizewell (SIZB), Bramford (BRAF)
print("\n=== Norfolk/East Anglia substations ===")
ea_buses = n.buses[
    (n.buses['x'] > 580000) & (n.buses['y'] > 260000) & (n.buses['y'] < 360000)
]
for idx, row in ea_buses.iterrows():
    print(f"  {idx}: x={row['x']:.0f}, y={row['y']:.0f}")

# Lines connecting East Anglia to rest of network (potential SSHARN boundary)
ea_bus_set = set(ea_buses.index)
ssharn_boundary = []
for idx, row in lines.iterrows():
    b0_in = row['bus0'] in ea_bus_set
    b1_in = row['bus1'] in ea_bus_set
    if b0_in != b1_in and row['s_nom'] >= 100:
        ssharn_boundary.append(idx)

ssharn = lines.loc[ssharn_boundary].sort_values('s_nom', ascending=False)
print(f"\n=== SSHARN boundary (lines crossing East Anglia, s_nom>=100) ===")
for idx, row in ssharn.head(25).iterrows():
    print(f"  {idx}: s_nom={row['s_nom']:.0f} MVA, "
          f"{row['bus0']}(x={row['from_x']:.0f},y={row['from_y']:.0f}) -> "
          f"{row['bus1']}(x={row['to_x']:.0f},y={row['to_y']:.0f})")
print(f"  Total: {ssharn['s_nom'].sum():.0f} MVA across {len(ssharn)} lines")

# ESTEX: Broader east boundary
print("\n\n=== Summary for boundary mapping ===")
for name, df in [("SCOTEX", scotex), ("SSE-SP", ssesp), ("SSHARN", ssharn)]:
    print(f"\n{name}:")
    print(f"  Lines: {list(df.index)}")
    print(f"  Total s_nom: {df['s_nom'].sum():.0f} MVA")
