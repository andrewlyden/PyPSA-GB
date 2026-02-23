"""Deeper investigation: Central DC Substation, Seagreen, and thermal coord bugs."""
import pypsa
import pandas as pd
from pyproj import Transformer
import numpy as np

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# ========== 1. CENTRAL DC SUBSTATION ==========
print("=" * 80)
print("CENTRAL DC SUBSTATION — bus details")
print("=" * 80)
cdc = 'Central DC Substation'
if cdc in n.buses.index:
    bus = n.buses.loc[cdc]
    print(f"  Name: {cdc}")
    print(f"  x={bus['x']:.6f}, y={bus['y']:.6f}")
    print(f"  v_nom={bus['v_nom']}")
    # What's connected
    for _, lk in n.links[n.links['bus0'] == cdc].iterrows():
        print(f"  Link-> {lk.name}: to {lk['bus1']} p_nom={lk['p_nom']:.0f}")
    for _, lk in n.links[n.links['bus1'] == cdc].iterrows():
        print(f"  Link<- {lk.name}: from {lk['bus0']} p_nom={lk['p_nom']:.0f}")
    for _, l in n.lines[(n.lines['bus0'] == cdc) | (n.lines['bus1'] == cdc)].iterrows():
        print(f"  Line: {l['bus0']} <-> {l['bus1']} s_nom={l['s_nom']:.0f}")
    for _, tr in n.transformers[(n.transformers['bus0'] == cdc) | (n.transformers['bus1'] == cdc)].iterrows():
        print(f"  Xfmr: {tr['bus0']} <-> {tr['bus1']} s_nom={tr['s_nom']:.0f}")
else:
    print(f"  Bus '{cdc}' NOT FOUND")

# Also check SPIT and BLHI buses
for prefix in ['SPIT', 'BLHI']:
    print(f"\n{prefix} buses:")
    for b in n.buses.index:
        if b.startswith(prefix):
            bus = n.buses.loc[b]
            lon, lat = t.transform(bus['x'], bus['y']) if bus['x'] > 100 else (bus['x'], bus['y'])
            n_lines = len(n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)])
            n_xfmr = len(n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)])
            n_links = len(n.links[(n.links['bus0']==b)|(n.links['bus1']==b)])
            print(f"  {b}: WGS84({lon:.4f},{lat:.4f}) v_nom={bus['v_nom']:.0f} "
                  f"lines={n_lines} xfmrs={n_xfmr} links={n_links}")

# ========== 2. SEAGREEN: WHY KINT INSTEAD OF SGRW? ==========
print("\n" + "=" * 80)
print("SEAGREEN: DISTANCE ANALYSIS")
print("=" * 80)

# Seagreen gen coords (WGS84)
sg_lon, sg_lat = -1.927, 56.636

# KINT4J bus coords
kint_bus = n.buses.loc['KINT4J']
kint_lon, kint_lat = t.transform(kint_bus['x'], kint_bus['y']) if kint_bus['x'] > 100 else (kint_bus['x'], kint_bus['y'])

# SGRW buses
sgrw_buses = {}
for b in n.buses.index:
    if b.startswith('SGRW'):
        bus = n.buses.loc[b]
        lon, lat = t.transform(bus['x'], bus['y']) if bus['x'] > 100 else (bus['x'], bus['y'])
        sgrw_buses[b] = (lon, lat, bus['v_nom'])

# Haversine distance
def haversine(lon1, lat1, lon2, lat2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

dist_kint = haversine(sg_lon, sg_lat, kint_lon, kint_lat)
print(f"  Seagreen gen @ ({sg_lon}, {sg_lat})")
print(f"  KINT4J bus    @ ({kint_lon:.4f}, {kint_lat:.4f}) v_nom={kint_bus['v_nom']:.0f} dist={dist_kint:.1f} km")
for b, (lon, lat, v) in sorted(sgrw_buses.items()):
    dist = haversine(sg_lon, sg_lat, lon, lat)
    print(f"  {b:8s} bus   @ ({lon:.4f}, {lat:.4f}) v_nom={v:.0f} dist={dist:.1f} km")

# Now check what the site data says — was bus pre-assigned?
print("\n  Checking REPD wind offshore site data...")
try:
    woff = pd.read_csv('resources/renewable/wind_offshore_sites.csv')
    sg = woff[woff['site_name'].str.contains('Seagreen', case=False, na=False)]
    if len(sg) > 0:
        print(f"  Found {len(sg)} Seagreen entries in wind_offshore_sites.csv:")
        for _, row in sg.iterrows():
            bus_col = row.get('bus', 'N/A')
            gsp_col = row.get('gsp', 'N/A')
            conn = row.get('Connection Site', row.get('connection_site', 'N/A'))
            print(f"    site_name={row['site_name']}, capacity={row.get('capacity_mw', 'N/A')}, "
                  f"bus={bus_col}, gsp={gsp_col}, connection={conn}")
            print(f"    lat={row.get('lat','N/A')}, lon={row.get('lon','N/A')}")
            # Show all columns
            for col in row.index:
                if col not in ['site_name', 'lat', 'lon', 'bus', 'gsp'] and pd.notna(row[col]):
                    print(f"    {col}={row[col]}")
    else:
        print("  No Seagreen entries found")
except Exception as e:
    print(f"  Error reading wind offshore sites: {e}")

# ========== 3. CONNAHS QUAY: CONN2J IS WRONG ==========
print("\n" + "=" * 80)
print("CONNAHS QUAY BUS LOCATION")
print("=" * 80)
conn_bus = n.buses.loc['CONN2J']
conn_lon, conn_lat = t.transform(conn_bus['x'], conn_bus['y']) if conn_bus['x'] > 100 else (conn_bus['x'], conn_bus['y'])
print(f"  CONN2J: WGS84({conn_lon:.4f}, {conn_lat:.4f}) OSGB({conn_bus['x']:.0f}, {conn_bus['y']:.0f}) v_nom={conn_bus['v_nom']:.0f}")
print(f"  Expected: near Connah's Quay, North Wales (~53.23°N, -3.08°W)")
print(f"  Actual:   ({conn_lat:.2f}°N, {conn_lon:.2f}°W)")
# Check CONN buses
for b in n.buses.index:
    if b.startswith('CONN'):
        bus = n.buses.loc[b]
        lon, lat = t.transform(bus['x'], bus['y']) if bus['x'] > 100 else (bus['x'], bus['y'])
        print(f"  {b}: WGS84({lon:.4f},{lat:.4f}) OSGB({bus['x']:.0f},{bus['y']:.0f}) v_nom={bus['v_nom']:.0f}")

# Check substation_coordinates.csv for CONN
print("\n  Checking substation_coordinates.csv...")
coords = pd.read_csv('data/network/ETYS/substation_coordinates.csv')
conn_coords = coords[coords['site_code'].str.startswith('CONN')]
for _, row in conn_coords.iterrows():
    print(f"  {row['site_code']}: {row['site_name']} ({row['lat']:.4f}, {row['lon']:.4f}) source={row.get('source', 'N/A')}")

# ========== 4. THERMAL GENERATOR COORDINATE SYSTEM CHECK ==========
print("\n" + "=" * 80)
print("THERMAL GENERATORS: COORDINATE SYSTEM CHECK")
print("=" * 80)
gens = n.generators.copy()
if 'lon' in gens.columns and 'lat' in gens.columns:
    has_coords = gens['lon'].notna() & gens['lat'].notna()
    gens_with = gens[has_coords].copy()

    # Check if any generators have coords that look like OSGB36 (values > 100)
    osgb_gens = gens_with[(gens_with['lon'].abs() > 100) | (gens_with['lat'].abs() > 100)]
    print(f"  Total generators with coords: {len(gens_with)}")
    print(f"  Generators with OSGB36-like coords (|lon|>100 or |lat|>100): {len(osgb_gens)}")

    if len(osgb_gens) > 0:
        # Show by carrier
        for carrier, group in osgb_gens.groupby('carrier'):
            print(f"\n  {carrier}: {len(group)} generators with OSGB36 coords")
            for idx, gen in group.head(3).iterrows():
                print(f"    {idx[:50]:50s} lon={gen['lon']:.1f} lat={gen['lat']:.1f} bus={gen['bus']}")
