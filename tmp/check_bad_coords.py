"""Identify buses with wrong geocoded coordinates and propose corrections."""
import pypsa
import pandas as pd
import numpy as np
from pyproj import Transformer

n = pypsa.Network('resources/network/Historical_2023_etys_network.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
coords = pd.read_csv('data/network/ETYS/substation_coordinates.csv')

def bus_wgs84(b):
    bus = n.buses.loc[b]
    if bus['x'] > 100:
        return t.transform(bus['x'], bus['y'])
    return bus['x'], bus['y']

def haversine(lon1, lat1, lon2, lat2):
    dlat = np.radians(lat2-lat1); dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(a))

# Check the suspected wrong-coordinate buses in detail
suspects = {
    'WHSO': ('Whitson, Newport, South Wales', 51.564, -2.829),
    'COWT': ('Cowbridge Tee, Vale of Glamorgan', 51.473, -3.460),
    'SPIT': ('Spittal (Caithness HVDC terminal)', 58.430, -3.364),
    'ORMO': ('Ormonde Onshore, Flimby/Workington', 54.680, -3.508),
    'DAIN': ('Daines? - check connections', None, None),
    'THTO': ('Thornton? - check connections', None, None),
    'DUDW': ('Duddon? - check connections', None, None),
    'DUDO': ('Duddon? - check connections', None, None),
    'DALL': ('Dalrymple? - check connections', None, None),
    'LACK': ('check connections', None, None),
    'CONQ': ('check connections', None, None),
}

print("=" * 80)
print("SUSPECTED BAD COORDINATES — TOPOLOGY CHECK")
print("=" * 80)
for prefix, (desc, true_lat, true_lon) in suspects.items():
    buses = [b for b in n.buses.index if b.startswith(prefix)]
    if not buses:
        print(f"\n{prefix}: NOT IN NETWORK")
        continue
    b0 = buses[0]
    lon, lat = bus_wgs84(b0)
    coord_row = coords[coords['site_code'] == prefix]
    source = coord_row.iloc[0].get('source', 'N/A') if len(coord_row) > 0 else 'NO_ENTRY'
    site_name = coord_row.iloc[0]['site_name'] if len(coord_row) > 0 else '???'
    print(f"\n{prefix}: {site_name} ({source})")
    print(f"  Current coords: ({lat:.4f}N, {lon:.4f}E)")
    if true_lat:
        dist = haversine(lon, lat, true_lon, true_lat)
        print(f"  Expected ({desc}): ({true_lat:.3f}N, {true_lon:.3f}E) — {dist:.0f}km off")
    else:
        print(f"  {desc}")

    # Show connections
    connected_sites = set()
    all_conns = []
    for _, l in n.lines[(n.lines['bus0'].str.startswith(prefix))|(n.lines['bus1'].str.startswith(prefix))].iterrows():
        other = l['bus1'] if l['bus0'].startswith(prefix) else l['bus0']
        olon, olat = bus_wgs84(other)
        if olon:
            dist_c = haversine(lon, lat, olon, olat)
            connected_sites.add(other[:4])
            all_conns.append((other[:4], olat, olon, dist_c))
    seen = set()
    for site, olat, olon, dist_c in sorted(set(all_conns), key=lambda x: x[0]):
        if site not in seen:
            seen.add(site)
            cr = coords[coords['site_code'] == site]
            sn = cr.iloc[0]['site_name'] if len(cr) > 0 else '???'
            print(f"  -> {site} ({sn}): ({olat:.3f}N, {olon:.3f}E) dist={dist_c:.0f}km")

# Check SPIT connections specifically - is 'Spittal' Caithness correct?
print("\n" + "="*80)
print("SPIT CONNECTIONS (should all be far-north Scotland)")
print("="*80)
for b in n.buses.index:
    if b.startswith('SPIT'):
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0']==b else l['bus0']
            olon, olat = bus_wgs84(other)
            cr = coords[coords['site_code']==other[:4]]
            sn = cr.iloc[0]['site_name'] if len(cr) > 0 else '???'
            print(f"  {b} -> {other} ({sn}): ({olat:.3f}N, {olon:.3f}E)")
        for _, lk in n.links[(n.links['bus0']==b)|(n.links['bus1']==b)].iterrows():
            other = lk['bus1'] if lk['bus0']==b else lk['bus0']
            olon, olat = bus_wgs84(other)
            print(f"  {b} =HVDC=> {other}: ({olat:.3f}N, {olon:.3f}E)")
        break

# Check THUS connections
print("\nTHUS connections (Thurso, north Scotland):")
for b in n.buses.index:
    if b.startswith('THUS'):
        lon, lat = bus_wgs84(b)
        print(f"  {b}: ({lat:.4f}N, {lon:.4f}E)")
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0']==b else l['bus0']
            olon, olat = bus_wgs84(other)
            cr = coords[coords['site_code']==other[:4]]
            sn = cr.iloc[0]['site_name'] if len(cr) > 0 else '???'
            print(f"    -> {other} ({sn}): ({olat:.3f}N, {olon:.3f}E)")

# Summary of all buses with nominatim coords involved in >100km lines
print("\n" + "="*80)
print("ALL NOMINATIM-GEOCODED BUSES INVOLVED IN >100km LINES: TOPOLOGY VERDICT")
print("="*80)
# Recalculate all long lines
long_bus_prefixes = {}
for _, l in n.lines.iterrows():
    b0, b1 = l['bus0'], l['bus1']
    lon0, lat0 = bus_wgs84(b0)
    lon1, lat1 = bus_wgs84(b1)
    if None in (lon0,lat0,lon1,lat1): continue
    dist = haversine(lon0,lat0,lon1,lat1)
    if dist > 100:
        for b in [b0, b1]:
            p = b[:4]
            if p not in long_bus_prefixes:
                long_bus_prefixes[p] = {'count': 0, 'max_dist': 0, 'connections': set()}
            long_bus_prefixes[p]['count'] += 1
            long_bus_prefixes[p]['max_dist'] = max(long_bus_prefixes[p]['max_dist'], dist)
            other = b1 if b == b0 else b0
            long_bus_prefixes[p]['connections'].add(other[:4])

for prefix, info in sorted(long_bus_prefixes.items(), key=lambda x: -x[1]['max_dist']):
    cr = coords[coords['site_code']==prefix]
    src = cr.iloc[0].get('source','NO_ENTRY') if len(cr)>0 else 'NO_ENTRY'
    sn = cr.iloc[0]['site_name'] if len(cr)>0 else '???'
    buses = [b for b in n.buses.index if b.startswith(prefix)]
    if not buses: continue
    lon, lat = bus_wgs84(buses[0])
    print(f"  {prefix:8s} ({src:25s}) site={sn:30s} at({lat:.2f}N,{lon:.2f}E) max_dist={info['max_dist']:.0f}km connects_to={sorted(info['connections'])}")
