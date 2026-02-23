"""Find all potential coordinate issues - onshore buses that ended up offshore."""
import pypsa
import pandas as pd
from pyproj import Transformer

n = pypsa.Network('resources/network/Historical_2023_etys_network.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# Convert all bus coords to WGS84
buses = n.buses.copy()
buses['lon_w'], buses['lat_w'] = t.transform(buses['x'].values, buses['y'].values)

# GB mainland approximate bounds in WGS84
# Buses with lon > 2.0 are likely in the North Sea (except legitimate offshore buses)
# Check which are connected to offshore buses
offshore_prefixes = set()
for b in buses.index:
    if buses.loc[b, 'lon_w'] > 2.0:
        # Check connections
        connected = set()
        for _, l in n.lines[n.lines['bus0'] == b].iterrows():
            connected.add(l['bus1'][:4])
        for _, l in n.lines[n.lines['bus1'] == b].iterrows():
            connected.add(l['bus0'][:4])
        for _, tr in n.transformers[n.transformers['bus0'] == b].iterrows():
            connected.add(tr['bus1'][:4])
        for _, tr in n.transformers[n.transformers['bus1'] == b].iterrows():
            connected.add(tr['bus0'][:4])
        print(f"  {b}: WGS84({buses.loc[b,'lon_w']:.3f}, {buses.loc[b,'lat_w']:.3f}) "
              f"v_nom={buses.loc[b,'v_nom']:.0f} connects to: {connected}")

# Also check substation_coordinates.csv coverage
coords_df = pd.read_csv('data/network/ETYS/substation_coordinates.csv')
known_sites = set(coords_df['site_code'].values)

# Get all unique 4-char site codes from the network
network_sites = set(b[:4] for b in n.buses.index if len(b) >= 4)
missing = network_sites - known_sites
print(f"\nTotal network sites: {len(network_sites)}")
print(f"Sites in substation_coordinates.csv: {len(known_sites)}")
print(f"Missing sites: {len(missing)}")
if missing:
    # Show missing sites that have buses with high lon
    for site in sorted(missing):
        site_buses = buses[buses.index.str.startswith(site)]
        max_lon = site_buses['lon_w'].max()
        if max_lon > 1.5:  # Potentially offshore-displaced
            print(f"  {site}: max_lon={max_lon:.3f} ({len(site_buses)} buses)")
