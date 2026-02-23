"""Investigate generation-to-bus mapping and central DC substation issue."""
import pypsa
import pandas as pd
from pyproj import Transformer
import numpy as np

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# Convert all bus coords to WGS84
buses = n.buses.copy()
is_osgb = (buses['x'].abs() > 100) | (buses['y'].abs() > 100)
if is_osgb.any():
    lon_c, lat_c = t.transform(buses.loc[is_osgb, 'x'].values, buses.loc[is_osgb, 'y'].values)
    buses.loc[is_osgb, 'lon_w'] = lon_c
    buses.loc[is_osgb, 'lat_w'] = lat_c
buses.loc[~is_osgb, 'lon_w'] = buses.loc[~is_osgb, 'x']
buses.loc[~is_osgb, 'lat_w'] = buses.loc[~is_osgb, 'y']

# ========== 1. CENTRAL DC SUBSTATION ==========
print("=" * 80)
print("BUSES NEAR GB CENTROID (54.5, -2.0) — within 0.5 degrees")
print("=" * 80)
central = buses[
    (buses['lat_w'].between(54.0, 55.0)) &
    (buses['lon_w'].between(-2.5, -1.5))
]
for idx, bus in central.iterrows():
    # Check what connects to this bus
    lines_from = n.lines[n.lines['bus0'] == idx]
    lines_to = n.lines[n.lines['bus1'] == idx]
    xfmr_from = n.transformers[n.transformers['bus0'] == idx]
    xfmr_to = n.transformers[n.transformers['bus1'] == idx]
    links_from = n.links[n.links['bus0'] == idx]
    links_to = n.links[n.links['bus1'] == idx]
    n_conn = len(lines_from) + len(lines_to) + len(xfmr_from) + len(xfmr_to) + len(links_from) + len(links_to)
    print(f"  {idx}: WGS84({bus['lon_w']:.4f}, {bus['lat_w']:.4f}) OSGB({bus['x']:.0f}, {bus['y']:.0f}) "
          f"v_nom={bus['v_nom']:.0f} connections={n_conn}")
    if len(links_from) > 0:
        for _, lk in links_from.iterrows():
            print(f"    Link-> {lk.name}: to {lk['bus1']} (p_nom={lk['p_nom']:.0f})")
    if len(links_to) > 0:
        for _, lk in links_to.iterrows():
            print(f"    Link<- {lk.name}: from {lk['bus0']} (p_nom={lk['p_nom']:.0f})")

# ========== 2. ALL HVDC LINK BUSES ==========
print("\n" + "=" * 80)
print("ALL LINKS AND THEIR BUS COORDINATES")
print("=" * 80)
for idx, lk in n.links.iterrows():
    b0, b1 = lk['bus0'], lk['bus1']
    b0_lon = buses.loc[b0, 'lon_w'] if b0 in buses.index else float('nan')
    b0_lat = buses.loc[b0, 'lat_w'] if b0 in buses.index else float('nan')
    b1_lon = buses.loc[b1, 'lon_w'] if b1 in buses.index else float('nan')
    b1_lat = buses.loc[b1, 'lat_w'] if b1 in buses.index else float('nan')
    print(f"  {idx}: {b0}({b0_lon:.3f},{b0_lat:.3f}) -> {b1}({b1_lon:.3f},{b1_lat:.3f}) p_nom={lk['p_nom']:.0f}")

# ========== 3. SEAGREEN / KINT / SGRW INVESTIGATION ==========
print("\n" + "=" * 80)
print("SEAGREEN WIND FARM INVESTIGATION")
print("=" * 80)

# Find the 1075MW generator
big_gens = n.generators[n.generators['p_nom'] > 1000].copy()
print(f"\nGenerators > 1000 MW ({len(big_gens)}):")
for idx, gen in big_gens.iterrows():
    print(f"  {idx}: {gen['carrier']} {gen['p_nom']:.0f} MW bus={gen['bus']}")

# Check all generators at KINT buses
kint_gens = n.generators[n.generators['bus'].str.startswith('KINT')]
print(f"\nGenerators at KINT* buses ({len(kint_gens)}):")
for idx, gen in kint_gens.iterrows():
    print(f"  {idx}: {gen['carrier']} {gen['p_nom']:.0f} MW bus={gen['bus']}")

# Check KINT and SGRW buses
print("\nKINT buses:")
kint_buses = buses[buses.index.str.startswith('KINT')]
for idx, bus in kint_buses.iterrows():
    print(f"  {idx}: WGS84({bus['lon_w']:.4f}, {bus['lat_w']:.4f}) v_nom={bus['v_nom']:.0f}")

print("\nSGRW buses:")
sgrw_buses = buses[buses.index.str.startswith('SGRW')]
for idx, bus in sgrw_buses.iterrows():
    print(f"  {idx}: WGS84({bus['lon_w']:.4f}, {bus['lat_w']:.4f}) v_nom={bus['v_nom']:.0f}")

# ========== 4. TOP 30 LARGEST GENERATORS AND THEIR BUSES ==========
print("\n" + "=" * 80)
print("TOP 30 GENERATORS BY CAPACITY")
print("=" * 80)
top30 = n.generators.nlargest(30, 'p_nom')
for idx, gen in top30.iterrows():
    bus = gen['bus']
    if bus in buses.index:
        b_lon, b_lat = buses.loc[bus, 'lon_w'], buses.loc[bus, 'lat_w']
    else:
        b_lon, b_lat = float('nan'), float('nan')
    gen_lon = gen.get('lon', float('nan'))
    gen_lat = gen.get('lat', float('nan'))
    print(f"  {idx[:50]:50s} {gen['carrier']:20s} {gen['p_nom']:8.0f} MW  "
          f"bus={bus:10s} bus@({b_lon:.3f},{b_lat:.3f})  gen@({gen_lon:.3f},{gen_lat:.3f})")

# ========== 5. CHECK FOR COORDINATE SYSTEM MISMATCHES ==========
print("\n" + "=" * 80)
print("GENERATORS WITH LARGE DISTANCE FROM BUS (potential misassignment)")
print("=" * 80)
gens = n.generators.copy()
gens = gens[gens['carrier'] != 'load_shedding']
if 'lon' in gens.columns and 'lat' in gens.columns:
    has_coords = gens['lon'].notna() & gens['lat'].notna()
    gens_with = gens[has_coords].copy()

    # Calculate distance from generator to its bus
    distances = []
    for idx, gen in gens_with.iterrows():
        bus = gen['bus']
        if bus in buses.index:
            bus_lon, bus_lat = buses.loc[bus, 'lon_w'], buses.loc[bus, 'lat_w']
            gen_lon, gen_lat = gen['lon'], gen['lat']
            # Haversine approx
            dlat = np.radians(gen_lat - bus_lat)
            dlon = np.radians(gen_lon - bus_lon)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(bus_lat)) * np.cos(np.radians(gen_lat)) * np.sin(dlon/2)**2
            dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))
            distances.append((idx, gen['carrier'], gen['p_nom'], bus, dist_km))

    distances.sort(key=lambda x: -x[4])
    print(f"Top 20 generators furthest from their assigned bus:")
    for name, carrier, pnom, bus, dist in distances[:20]:
        print(f"  {name[:45]:45s} {carrier:20s} {pnom:8.0f} MW  bus={bus:10s}  dist={dist:.1f} km")
