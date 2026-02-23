"""Check KINC2-, WIYH location validity, and Seagreen rebuild."""
import pypsa
import pandas as pd
import numpy as np
from pyproj import Transformer

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# KINC2- bus
print("KINC2- bus:")
for b in n.buses.index:
    if b.startswith('KINC'):
        bus = n.buses.loc[b]
        lon, lat = t.transform(bus['x'], bus['y']) if bus['x'] > 100 else (bus['x'], bus['y'])
        print(f"  {b}: WGS84({lon:.4f},{lat:.4f}) OSGB({bus['x']:.0f},{bus['y']:.0f}) v_nom={bus['v_nom']:.0f}")
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0'] == b else l['bus0']
            print(f"    Line to {other}")
        for _, tr in n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)].iterrows():
            other = tr['bus1'] if tr['bus0'] == b else tr['bus0']
            print(f"    Xfmr to {other}")

coords = pd.read_csv('data/network/ETYS/substation_coordinates.csv')
kinc_coords = coords[coords['site_code'].str.startswith('KINC')]
print(f"\n  substation_coordinates.csv: {len(kinc_coords)} entries")
for _, row in kinc_coords.iterrows():
    print(f"    {row['site_code']}: {row['site_name']} ({row['lat']}, {row['lon']}) source={row.get('source','N/A')}")

# Distance from Kincardine wind farm to KINC2-
kinc_wf_lon, kinc_wf_lat = -1.874, 56.997  # WGS84
t2 = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
kx, ky = t2.transform(kinc_wf_lon, kinc_wf_lat)
for b in n.buses.index:
    if b.startswith('KINC'):
        bus = n.buses.loc[b]
        d = np.sqrt((bus['x']-kx)**2 + (bus['y']-ky)**2)/1000
        print(f"\n  Kincardine WF to {b}: {d:.1f}km")

# Check the output file timestamps to understand rebuild
import os
files = [
    'resources/network/Historical_2023_etys_network_demand_renewables.pkl',
    'resources/network/Historical_2023_etys_network_demand_renewables_thermal_generators.pkl',
    'resources/network/Historical_2023_etys_solved.nc',
]
print("\n\nFile timestamps:")
for f in files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"  {f}: {dt}")
    else:
        print(f"  {f}: NOT FOUND")

# Also check spatial_utils.py timestamp
su = 'scripts/utilities/spatial_utils.py'
if os.path.exists(su):
    mtime = os.path.getmtime(su)
    dt = datetime.datetime.fromtimestamp(mtime)
    print(f"  {su}: {dt}")
