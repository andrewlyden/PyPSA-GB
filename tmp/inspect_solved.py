"""Inspect solved network for generators and links."""
import pypsa
import pandas as pd

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')

print(f"Network: {len(n.buses)} buses, {len(n.lines)} lines, {len(n.transformers)} xfmrs, {len(n.links)} links")
print(f"Generators: {len(n.generators)}, Storage: {len(n.storage_units)}")

# Generator carriers
if len(n.generators) > 0:
    print(f"\n=== Generator carriers ===")
    print(n.generators.groupby('carrier')['p_nom'].agg(['count', 'sum']).to_string())

    # Check coordinate columns
    cols = n.generators.columns.tolist()
    coord_cols = [c for c in cols if c in ['x', 'y', 'geo_lat', 'geo_lon', 'lat', 'lon']]
    print(f"\nGenerator coordinate columns: {coord_cols}")

    if 'x' in cols and 'y' in cols:
        has_xy = n.generators[['x','y']].notna().all(axis=1)
        print(f"Generators with x,y: {has_xy.sum()} / {len(n.generators)}")
        if not has_xy.all():
            missing = n.generators[~has_xy][['carrier', 'bus', 'p_nom']].head(10)
            print(f"Missing x,y (first 10):\n{missing}")
    else:
        print("No x,y columns in generators")

# Links
if len(n.links) > 0:
    print(f"\n=== Links ===")
    for idx, lk in n.links.iterrows():
        print(f"  {idx}: {lk.bus0} -> {lk.bus1} (p_nom={lk.p_nom:.0f}, carrier={lk.get('carrier', 'N/A')})")

# Check what BRST buses exist and their coordinates
print(f"\n=== BRST buses ===")
brst = n.buses[n.buses.index.str.startswith('BRST')]
for idx, bus in brst.iterrows():
    print(f"  {idx}: x={bus.x:.0f}, y={bus.y:.0f}, v_nom={bus.v_nom}")

# Check EAAW coordinates compared to BRST
print(f"\n=== EAAW vs BRST coordinates ===")
eaaw = n.buses[n.buses.index.str.startswith('EAAW')]
for idx, bus in eaaw.iterrows():
    print(f"  {idx}: x={bus.x:.0f}, y={bus.y:.0f}")
