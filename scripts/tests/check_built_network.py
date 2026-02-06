"""Quick check: verify Extra_WF_edge ratings and offshore buses in built network."""
import pypsa
import pandas as pd

n = pypsa.Network("resources/network/HT35_flex_network.nc")

# Check lines with s_nom=9999
inf_lines = n.lines[n.lines.s_nom == 9999]
print(f"Lines with s_nom=9999: {len(inf_lines)} of {len(n.lines)}")

# Check lines with s_nom > 3000 (likely infinite capacity stubs)
high_cap = n.lines[n.lines.s_nom > 3000]
print(f"Lines with s_nom > 3000 MVA: {len(high_cap)}")
if len(high_cap) > 0:
    for idx, row in high_cap.iterrows():
        print(f"  {row.bus0} -- {row.bus1}: {row.s_nom:.0f} MVA")

# Check for buses at sea (lon/lat indicating offshore)
# GB bounding box: lon roughly -8 to 2, lat roughly 49 to 61
# Offshore buses will be outside the land polygon
print()
print("=== Offshore buses (x,y coordinates) ===")
# Check if any buses are clearly at sea based on coordinates
# With OSGB36, x should be roughly 0-700000, y 0-1200000
# Buses clearly east or west of GB or far from coast
import geopandas as gpd
from shapely.geometry import Point

# Load land boundary
gdf = gpd.read_file("data/network/GSP/GSP_regions_4326_20250109.geojson")
land = gdf.geometry.union_all()

at_sea = []
for bus_id, bus in n.buses.iterrows():
    point = Point(bus.lon, bus.lat)
    if not land.contains(point):
        at_sea.append(bus_id)

print(f"Buses at sea: {len(at_sea)}")
if len(at_sea) > 0 and len(at_sea) <= 30:
    for b in at_sea:
        print(f"  {b}: ({n.buses.loc[b, 'lat']:.4f}, {n.buses.loc[b, 'lon']:.4f})")

# Check specific known WF buses
print()
print("=== Extra_WF_edges bus ratings in network ===")
wf_buses = ['BOSO11', 'THAW11', 'ORMO11', 'SALL11', 'GUNS11', 'LONO4A', 'BODE41', 'LINO41', 
            'RORE11', 'RORW11', 'CREB2A', 'NECT41']
for bus in wf_buses:
    lines = n.lines[(n.lines.bus0 == bus) | (n.lines.bus1 == bus)]
    if len(lines) > 0:
        for idx, line in lines.iterrows():
            other = line.bus1 if line.bus0 == bus else line.bus0
            print(f"  {bus} -- {other}: s_nom={line.s_nom:.0f} MVA")
    else:
        print(f"  {bus}: No lines found")
