"""Check the actual nearest-neighbor distance in OSGB36."""
import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')

# Seagreen OSGB36 coords from REPD
sg_x, sg_y = 404600, 749300

# All KINT and SGRW buses
print("OSGB36 distances from Seagreen (404600, 749300):")
print()
for prefix in ['KINT', 'SGRW']:
    for b in n.buses.index:
        if b.startswith(prefix):
            bx, by = n.buses.loc[b, 'x'], n.buses.loc[b, 'y']
            dist = np.sqrt((bx - sg_x)**2 + (by - sg_y)**2) / 1000
            print(f"  {b:10s} OSGB({bx:.0f}, {by:.0f}) dist={dist:.1f} km  v_nom={n.buses.loc[b, 'v_nom']:.0f}")

# Now find the ACTUAL nearest bus overall
all_dists = []
for b in n.buses.index:
    bx, by = n.buses.loc[b, 'x'], n.buses.loc[b, 'y']
    if np.isnan(bx) or np.isnan(by) or bx < 100:
        continue  # Skip non-OSGB buses
    dist = np.sqrt((bx - sg_x)**2 + (by - sg_y)**2) / 1000
    all_dists.append((b, dist, n.buses.loc[b, 'v_nom']))

all_dists.sort(key=lambda x: x[1])
print("\n10 nearest buses to Seagreen by OSGB36 Euclidean distance:")
for b, dist, v in all_dists[:10]:
    print(f"  {b:10s} dist={dist:.1f} km  v_nom={v:.0f}")

# Check the WGS84 mapping path — what happens when we convert Seagreen WGS84 to OSGB36?
from pyproj import Transformer
t_to_osgb = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
sg_lon, sg_lat = -1.926620733768669, 56.63551233113348
sg_x2, sg_y2 = t_to_osgb.transform(sg_lon, sg_lat)
print(f"\nSeagreen WGS84 ({sg_lon}, {sg_lat}) -> OSGB36 ({sg_x2:.0f}, {sg_y2:.0f})")
print(f"Original REPD OSGB36: ({sg_x}, {sg_y})")
print(f"Difference: ({sg_x2-sg_x:.0f}, {sg_y2-sg_y:.0f})")

# What does map_sites_to_buses receive? Check the lon/lat columns
print("\n\nChecking how the mapping function sees the coordinates...")
# The call is: map_sites_to_buses(network, sites, method='nearest', lon_col='lon', lat_col='lat')
# Sites have lon=-1.927, lat=56.636 (WGS84)
# Bus x,y are in OSGB36 (hundreds of thousands)
# So it should hit the MIXED branch (line 447)

# Let's simulate exactly what the function does
bus_xs = n.buses['x'].values
bus_ys = n.buses['y'].values
is_osgb_buses = np.any(np.abs(bus_xs) > 1000) or np.any(np.abs(bus_ys) > 1000)
is_wgs_sites = np.abs(sg_lon) < 180 and np.abs(sg_lat) < 180
print(f"Bus coord system detected as OSGB36: {is_osgb_buses}")
print(f"Site coord system detected as WGS84: {is_wgs_sites}")

# What the MIXED branch does: convert site to OSGB36, then Euclidean
dists_mixed = []
for b in n.buses.index:
    bx, by = n.buses.loc[b, 'x'], n.buses.loc[b, 'y']
    if np.isnan(bx) or np.isnan(by) or bx < 100:
        continue
    dist = np.sqrt((bx - sg_x2)**2 + (by - sg_y2)**2) / 1000
    dists_mixed.append((b, dist, n.buses.loc[b, 'v_nom']))

dists_mixed.sort(key=lambda x: x[1])
print("\n10 nearest buses to Seagreen (WGS84->OSGB36 converted) by Euclidean distance:")
for b, dist, v in dists_mixed[:10]:
    print(f"  {b:10s} dist={dist:.1f} km  v_nom={v:.0f}")

# ======= CONN buses investigation =======
print("\n" + "=" * 80)
print("CONN BUS COORDINATE INVESTIGATION")
print("=" * 80)
# CONN2J is at OSGB (294365, 958499) — that's y=958499 which is in northern Scotland!
# Connah's Quay is at approximately OSGB (295000, 371000)
# y=958499 is WAY too high — this is Caithness, not North Wales

# Check where CONN appears in substation_coordinates.csv
coords = pd.read_csv('data/network/ETYS/substation_coordinates.csv')
conn_match = coords[coords['site_code'].str.upper().str.startswith('CONN')]
print(f"CONN entries in substation_coordinates.csv: {len(conn_match)}")
for _, row in conn_match.iterrows():
    print(f"  {row['site_code']}: {row['site_name']} ({row['lat']}, {row['lon']})")

# So CONN buses must have been assigned coordinates by the guessing algorithm
# Check what buses CONN is connected to
print("\nCONN bus connectivity:")
for b in n.buses.index:
    if b.startswith('CONN'):
        bus = n.buses.loc[b]
        print(f"\n  {b}: OSGB({bus['x']:.0f}, {bus['y']:.0f}) v_nom={bus['v_nom']:.0f}")
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0'] == b else l['bus0']
            print(f"    Line to {other}: s_nom={l['s_nom']:.0f}")
        for _, tr in n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)].iterrows():
            other = tr['bus1'] if tr['bus0'] == b else tr['bus0']
            print(f"    Xfmr to {other}: s_nom={tr['s_nom']:.0f}")

# Find other buses starting with same 4 chars that CONN connects to
print("\nWhat sites do CONN buses connect to?")
conn_connected = set()
for b in n.buses.index:
    if b.startswith('CONN'):
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0'] == b else l['bus0']
            conn_connected.add(other[:4])
        for _, tr in n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)].iterrows():
            other = tr['bus1'] if tr['bus0'] == b else tr['bus0']
            conn_connected.add(other[:4])
print(f"Connected site prefixes: {sorted(conn_connected)}")
for prefix in sorted(conn_connected):
    if prefix in coords['site_code'].values:
        row = coords[coords['site_code'] == prefix].iloc[0]
        print(f"  {prefix}: {row['site_name']} ({row['lat']}, {row['lon']})")
    else:
        print(f"  {prefix}: NOT in substation_coordinates.csv")
