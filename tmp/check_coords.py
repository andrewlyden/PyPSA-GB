"""Check coordinate issues for offshore buses."""
import pypsa
from pyproj import Transformer

n = pypsa.Network('resources/network/Historical_2023_etys_network.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# Check key buses
prefixes = ['EAAW', 'BRST', 'BRFO', 'BEIW', 'BEAT', 'MOWE', 'MORO', 'BLHI']
for prefix in prefixes:
    buses = [b for b in n.buses.index if b.startswith(prefix)]
    if not buses:
        continue
    print(f"\n=== {prefix} ===")
    for b in buses[:6]:
        x, y = n.buses.loc[b, 'x'], n.buses.loc[b, 'y']
        lon, lat = t.transform(x, y)
        print(f"  {b}: OSGB({x:.0f}, {y:.0f}) -> WGS84({lon:.3f}, {lat:.3f})")

# Find all buses with x > 700000 (potentially far east)
print("\n=== Buses with x > 700000 ===")
far_east = n.buses[n.buses['x'] > 700000].copy()
far_east['lon_w'], far_east['lat_w'] = t.transform(far_east['x'].values, far_east['y'].values)
for idx, row in far_east.iterrows():
    print(f"  {idx}: OSGB({row['x']:.0f}, {row['y']:.0f}) -> WGS84({row['lon_w']:.3f}, {row['lat_w']:.3f})")
