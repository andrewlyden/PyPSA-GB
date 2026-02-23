"""Verify Seagreen BMU path and Connah's Quay CONN mismatch."""
import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')

# ========== 1. SGRW21 transformer capacity ==========
print("=" * 80)
print("SGRW21 TRANSFORMER CAPACITY (explains BMU trigger)")
print("=" * 80)
for b in ['SGRW21', 'SGRW22', 'SGRW61', 'SGRW62']:
    xfmrs = n.transformers[(n.transformers['bus0']==b)|(n.transformers['bus1']==b)]
    xfmr_cap = xfmrs['s_nom'].sum()
    print(f"  {b}: transformer capacity = {xfmr_cap:.0f} MVA ({len(xfmrs)} transformers)")
    for _, tr in xfmrs.iterrows():
        print(f"    {tr.name}: {tr['bus0']} <-> {tr['bus1']} s_nom={tr['s_nom']:.0f}")

# ========== 2. CONN is NOT Connah's Quay ==========
print("\n" + "=" * 80)
print("CONNAH'S QUAY: WHERE SHOULD IT CONNECT?")
print("=" * 80)
# Search for DEES (Deeside) buses — Connah's Quay connects to Deeside substation
for prefix in ['DEES', 'DSID', 'CQPS']:
    buses = [b for b in n.buses.index if b.startswith(prefix)]
    if buses:
        print(f"\n  {prefix} buses found: {buses}")
        for b in buses:
            bus = n.buses.loc[b]
            print(f"    {b}: OSGB({bus['x']:.0f}, {bus['y']:.0f}) v_nom={bus['v_nom']:.0f}")

# Search for any bus near Connah's Quay location: OSGB ~(295000, 370000)
print("\n  Buses near Connah's Quay (OSGB ~295000, 370000), within 30km:")
for b in n.buses.index:
    bus = n.buses.loc[b]
    if pd.notna(bus['x']) and bus['x'] > 100:
        dist = np.sqrt((bus['x'] - 295000)**2 + (bus['y'] - 370000)**2) / 1000
        if dist < 30:
            print(f"    {b}: OSGB({bus['x']:.0f}, {bus['y']:.0f}) v_nom={bus['v_nom']:.0f} dist={dist:.1f}km")

# ========== 3. Which generators are at wrong buses due to STATION_TO_BMU? ==========
print("\n" + "=" * 80)
print("GENERATORS WITH NAME 'CONNAH' OR SIMILAR")
print("=" * 80)
for idx, gen in n.generators.iterrows():
    if 'connah' in idx.lower() or 'connahs' in idx.lower():
        bus = gen['bus']
        bus_data = n.buses.loc[bus] if bus in n.buses.index else None
        if bus_data is not None:
            print(f"  {idx}: {gen['carrier']} {gen['p_nom']:.0f} MW bus={bus} "
                  f"OSGB({bus_data['x']:.0f}, {bus_data['y']:.0f}) v_nom={bus_data['v_nom']:.0f}")
        else:
            print(f"  {idx}: {gen['carrier']} {gen['p_nom']:.0f} MW bus={bus} (bus not found)")

# Check the actual Connah's Quay location
from pyproj import Transformer
t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
# Connah's Quay PS is at approximately OSGB (293000, 370500)
cq_lon, cq_lat = t.transform(293000, 370500)
print(f"\n  Connah's Quay actual location: WGS84({cq_lon:.4f}, {cq_lat:.4f})")
print(f"  CONN2J bus location: OSGB(294365, 958499) → WGS84({t.transform(294365, 958499)[0]:.4f}, {t.transform(294365, 958499)[1]:.4f})")
print(f"  Distance between them: {np.sqrt((294365-293000)**2 + (958499-370500)**2)/1000:.0f} km!")

# ========== 4. SGRW doesn't have 400kV — triggers Method 3 ==========
print("\n" + "=" * 80)
print("SGRW: NO 400kV BUSES → Method 3 finds nearest 400kV")
print("=" * 80)
# Check what 400kV buses are near SGRW
sgrw_x, sgrw_y = 411086, 746583
print(f"  SGRW21 OSGB: ({sgrw_x}, {sgrw_y})")
nearest_400kv = []
for b in n.buses.index:
    bus = n.buses.loc[b]
    if bus['v_nom'] == 400 and pd.notna(bus['x']) and bus['x'] > 100:
        dist = np.sqrt((bus['x'] - sgrw_x)**2 + (bus['y'] - sgrw_y)**2) / 1000
        # Check line capacity
        lines = n.lines[(n.lines['bus0']==b)|(n.lines['bus1']==b)]
        line_cap = lines['s_nom'].sum()
        nearest_400kv.append((b, dist, line_cap, bus['v_nom']))

nearest_400kv.sort(key=lambda x: x[1])
print(f"\n  5 nearest 400kV buses to SGRW:")
for b, dist, lcap, v in nearest_400kv[:5]:
    enough = "✓" if lcap >= 1075 * 1.5 else "✗"
    print(f"    {b:10s} dist={dist:.1f}km line_cap={lcap:.0f} MVA (need {1075*1.5:.0f}) {enough}")

# ========== 5. FULL BMU MAPPING ANALYSIS ==========
print("\n" + "=" * 80)
print("OTHER LARGE GENERATORS THAT MAY BE MIS-MAPPED")
print("=" * 80)
# Check all generators > 200 MW for distance from bus
for idx, gen in n.generators.iterrows():
    if gen['carrier'] == 'load_shedding' or gen['carrier'] == 'EU_import':
        continue
    if gen['p_nom'] < 200:
        continue
    bus = gen['bus']
    if bus not in n.buses.index:
        continue
    bus_data = n.buses.loc[bus]
    if pd.isna(bus_data['x']) or bus_data['x'] < 100:
        continue
    if 'lon' not in gen or pd.isna(gen.get('lon')) or 'lat' not in gen or pd.isna(gen.get('lat')):
        continue
    gen_lon, gen_lat = gen['lon'], gen['lat']
    bus_lon, bus_lat = t.transform(bus_data['x'], bus_data['y'])
    # Haversine
    dlat = np.radians(gen_lat - bus_lat)
    dlon = np.radians(gen_lon - bus_lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(bus_lat))*np.cos(np.radians(gen_lat))*np.sin(dlon/2)**2
    dist = 2*6371*np.arcsin(np.sqrt(a))
    if dist > 50:  # More than 50km off
        print(f"  {idx[:45]:45s} {gen['carrier']:20s} {gen['p_nom']:8.0f} MW  bus={bus:10s} dist={dist:.0f}km")
