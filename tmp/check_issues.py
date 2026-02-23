"""Check WIYH2R, NNGW62, Seagreen bus, Kincardine after rerun."""
import pypsa
import pandas as pd
import numpy as np
from pyproj import Transformer

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

def bus_wgs84(bus_name):
    if bus_name not in n.buses.index:
        return None, None
    b = n.buses.loc[bus_name]
    if b['x'] > 100:
        return t.transform(b['x'], b['y'])
    return b['x'], b['y']

# ========== 1. WIYH2R ==========
print("=" * 80)
print("WIYH2R INVESTIGATION")
print("=" * 80)
for b in n.buses.index:
    if b.startswith('WIYH'):
        bus = n.buses.loc[b]
        lon, lat = bus_wgs84(b)
        n_lines = len(n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)])
        n_xfmr = len(n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)])
        print(f"  {b}: OSGB({bus['x']:.0f}, {bus['y']:.0f}) WGS84({lon:.4f}, {lat:.4f}) v_nom={bus['v_nom']:.0f} lines={n_lines} xfmrs={n_xfmr}")
        # Show connections
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0'] == b else l['bus0']
            print(f"    Line to {other}")
        for _, tr in n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)].iterrows():
            other = tr['bus1'] if tr['bus0'] == b else tr['bus0']
            print(f"    Xfmr to {other}")

# Check substation_coordinates.csv
coords = pd.read_csv('data/network/ETYS/substation_coordinates.csv')
wiyh = coords[coords['site_code'].str.startswith('WIYH')]
print(f"\n  substation_coordinates.csv entries for WIYH: {len(wiyh)}")
for _, row in wiyh.iterrows():
    print(f"    {row['site_code']}: {row['site_name']} ({row['lat']}, {row['lon']}) source={row.get('source','N/A')}")

# ========== 2. NNGW62 ==========
print("\n" + "=" * 80)
print("NNGW62 INVESTIGATION")
print("=" * 80)
for b in n.buses.index:
    if b.startswith('NNGW') or b.startswith('NNG'):
        bus = n.buses.loc[b]
        lon, lat = bus_wgs84(b)
        n_lines = len(n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)])
        n_xfmr = len(n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)])
        print(f"  {b}: OSGB({bus['x']:.0f}, {bus['y']:.0f}) WGS84({lon:.4f}, {lat:.4f}) v_nom={bus['v_nom']:.0f} lines={n_lines} xfmrs={n_xfmr}")
        for _, l in n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)].iterrows():
            other = l['bus1'] if l['bus0'] == b else l['bus0']
            print(f"    Line to {other}")
        for _, tr in n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)].iterrows():
            other = tr['bus1'] if tr['bus0'] == b else tr['bus0']
            print(f"    Xfmr to {other}")

nngw = coords[coords['site_code'].str.startswith('NNGW') | coords['site_code'].str.startswith('NNG')]
print(f"\n  substation_coordinates.csv entries for NNG*: {len(nngw)}")
for _, row in nngw.iterrows():
    print(f"    {row['site_code']}: {row['site_name']} ({row['lat']}, {row['lon']}) source={row.get('source','N/A')}")

# ========== 3. SEAGREEN BUS ==========
print("\n" + "=" * 80)
print("SEAGREEN BUS ASSIGNMENT (after rerun)")
print("=" * 80)
sg_gens = n.generators[n.generators.index.str.contains('Seagreen|seagreen', case=False)]
for idx, gen in sg_gens.iterrows():
    bus = gen['bus']
    lon, lat = bus_wgs84(bus) if bus in n.buses.index else (None, None)
    print(f"  {idx}: {gen['carrier']} {gen['p_nom']:.0f} MW bus={bus} bus_wgs84=({lon:.4f},{lat:.4f})")
    print(f"    gen_lon={gen.get('lon', 'N/A')}, gen_lat={gen.get('lat', 'N/A')}")

# Check SGRW buses
print("\n  SGRW buses:")
for b in n.buses.index:
    if b.startswith('SGRW'):
        lon, lat = bus_wgs84(b)
        print(f"    {b}: WGS84({lon:.4f}, {lat:.4f}) v_nom={n.buses.loc[b,'v_nom']:.0f}")

# ========== 4. KINCARDINE OFFSHORE ==========
print("\n" + "=" * 80)
print("KINCARDINE OFFSHORE WINDFARM")
print("=" * 80)
kinc_gens = n.generators[n.generators.index.str.contains('Kincardine|kincardine', case=False)]
for idx, gen in kinc_gens.iterrows():
    bus = gen['bus']
    lon, lat = bus_wgs84(bus) if bus in n.buses.index else (None, None)
    print(f"  {idx}: {gen['carrier']} {gen['p_nom']:.0f} MW bus={bus}")
    if lon: print(f"    bus_wgs84=({lon:.4f},{lat:.4f})")
    print(f"    gen_lon={gen.get('lon', 'N/A')}, gen_lat={gen.get('lat', 'N/A')}")

# Check if there's a KINC bus prefix
kinc_buses = [b for b in n.buses.index if b.startswith('KINC')]
print(f"\n  KINC* buses: {kinc_buses}")

# Check the wind_offshore_sites.csv for Kincardine
try:
    woff = pd.read_csv('resources/renewable/wind_offshore_sites.csv')
    kinc = woff[woff['site_name'].str.contains('Kincardine', case=False, na=False)]
    print(f"\n  wind_offshore_sites.csv entries:")
    for _, row in kinc.iterrows():
        print(f"    site_name={row['site_name']}, capacity={row.get('capacity_mw','N/A')}")
        print(f"    lat={row.get('lat','N/A')}, lon={row.get('lon','N/A')}")
        print(f"    X-coord={row.get('X-coordinate','N/A')}, Y-coord={row.get('Y-coordinate','N/A')}")
except Exception as e:
    print(f"  Error: {e}")

# Also check what bus Kincardine's location maps to
print("\n  Nearest buses to Kincardine (~57.0N, -2.0W):")
kinc_lon, kinc_lat = -2.0, 57.0
t2 = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
kx, ky = t2.transform(kinc_lon, kinc_lat)
dists = []
for b in n.buses.index:
    bus = n.buses.loc[b]
    if pd.notna(bus['x']) and bus['x'] > 100:
        d = np.sqrt((bus['x']-kx)**2 + (bus['y']-ky)**2)/1000
        dists.append((b, d, bus['v_nom']))
dists.sort(key=lambda x: x[1])
for b, d, v in dists[:8]:
    print(f"    {b}: dist={d:.1f}km v_nom={v:.0f}")

# ========== 5. Check BMU mapping log ==========
print("\n" + "=" * 80)
print("SEAGREEN: check apply_etys_bmu_mapping flow")
print("=" * 80)
# Check what the BMU mapping would do with Seagreen
# The issue: does Method 1 match 'seagreen'?
# gen_name for Seagreen = str(site['site_name']).lower() -> 'seagreen'
# STATION_TO_BMU_PREFIX has 'seagreen': 'SGRW'
# bmu_mapping['SGRW'] should be SGRW21 or SGRW22 (whichever is preferred)

# Simulate: what v_nom does the nearest-neighbor assign?
# Seagreen OSGB: 404600, 749300
# SGRW21 OSGB: 411086, 746583 -> dist 7.0km, v_nom=275
# So bus = SGRW21, v_nom = 275

# needs_voltage_upgrade check:
# capacity_mw = 1075
# v_nom = 275
# 1075 >= 300 and 275 < 275? -> False!
# So needs_voltage_upgrade = False, BMU mapping is NOT triggered at all.
# Seagreen stays at SGRW21.

# But user says it's STILL not at SGRW62. Let's check what bus it actually got.
print("  Current Seagreen bus:", sg_gens.iloc[0]['bus'] if len(sg_gens) > 0 else "NOT FOUND")
