"""Investigate WHSO32 and buses with suspiciously long line connections."""
import pypsa
import pandas as pd
import numpy as np
from pyproj import Transformer

n = pypsa.Network('resources/network/Historical_2023_etys_network.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

def bus_wgs84(bus_name):
    if bus_name not in n.buses.index:
        return None, None
    b = n.buses.loc[bus_name]
    if b['x'] > 100:
        return t.transform(b['x'], b['y'])
    return b['x'], b['y']

def haversine(lon1, lat1, lon2, lat2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

# ========== 1. WHSO32 specific ==========
print("=" * 80)
print("WHSO32 INVESTIGATION")
print("=" * 80)
for b in n.buses.index:
    if b.startswith('WHSO'):
        bus = n.buses.loc[b]
        lon, lat = bus_wgs84(b)
        print(f"  {b}: OSGB({bus['x']:.0f}, {bus['y']:.0f}) WGS84({lon:.4f}, {lat:.4f}) v_nom={bus['v_nom']:.0f}")
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0'] == b else l['bus0']
            olon, olat = bus_wgs84(other)
            if olon and lon:
                dist = haversine(lon, lat, olon, olat)
                print(f"    Line to {other} @ ({olon:.4f},{olat:.4f}) dist={dist:.1f}km s_nom={l['s_nom']:.0f}")
            else:
                print(f"    Line to {other} (no coords)")

coords = pd.read_csv('data/network/ETYS/substation_coordinates.csv')
whso = coords[coords['site_code'].str.startswith('WHSO')]
print(f"\n  substation_coordinates.csv for WHSO*: {len(whso)}")
for _, row in whso.iterrows():
    print(f"    {row['site_code']}: {row['site_name']} ({row['lat']}, {row['lon']}) source={row.get('source','N/A')}")

# ========== 2. ALL LINE LENGTHS — find implausible connections ==========
print("\n" + "=" * 80)
print("LINES WITH IMPLAUSIBLE GEOGRAPHIC DISTANCE (>100km)")
print("=" * 80)

long_lines = []
for idx, line in n.lines.iterrows():
    b0, b1 = line['bus0'], line['bus1']
    lon0, lat0 = bus_wgs84(b0)
    lon1, lat1 = bus_wgs84(b1)
    if None in (lon0, lat0, lon1, lat1):
        continue
    dist = haversine(lon0, lat0, lon1, lat1)
    if dist > 100:
        long_lines.append((idx, b0, b1, dist, lon0, lat0, lon1, lat1, line['s_nom']))

long_lines.sort(key=lambda x: -x[3])
print(f"Found {len(long_lines)} lines >100km:")
for idx, b0, b1, dist, lon0, lat0, lon1, lat1, snom in long_lines[:40]:
    print(f"  {dist:6.0f}km  {b0}({lon0:.3f},{lat0:.3f}) <-> {b1}({lon1:.3f},{lat1:.3f})  s_nom={snom:.0f}")

# ========== 3. Identify which buses are causing most long-line issues ==========
print("\n" + "=" * 80)
print("BUSES INVOLVED IN MOST LONG LINES (likely wrong coordinates)")
print("=" * 80)
bus_longline_count = {}
for _, b0, b1, dist, *_ in long_lines:
    for b in [b0, b1]:
        bus_longline_count[b] = bus_longline_count.get(b, 0) + 1

sorted_buses = sorted(bus_longline_count.items(), key=lambda x: -x[1])
for bus, count in sorted_buses[:20]:
    lon, lat = bus_wgs84(bus)
    prefix = bus[:4]
    coord_row = coords[coords['site_code'] == prefix]
    src = coord_row.iloc[0].get('source', 'N/A') if len(coord_row) > 0 else 'NO_ENTRY'
    site = coord_row.iloc[0]['site_name'] if len(coord_row) > 0 else '???'
    print(f"  {bus:10s} {count:2d} long lines  WGS84({lon:.3f},{lat:.3f})  prefix_src={src}  site={site}")
