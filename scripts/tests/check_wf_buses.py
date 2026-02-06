"""Check wind farm bus locations in the ETYS network."""
import pypsa
import pandas as pd
from pyproj import Transformer

import pickle
with open("resources/network/HT35_flex_network_demand.pkl", "rb") as f:
    base = pickle.load(f)
print(f"Base network: {len(base.buses)} buses, {len(base.lines)} lines")

wf_buses = [
    "CREB2A", "CREB2B", "CREB21", "CREB22",
    "BEAT4A", "BEAT4B", "BEAT41", "BEAT42",
    "LONO4A", "LONO4B", "LONO41", "LONO42",
    "BODE41", "BODE42",
    "BLHI41", "BLHI42", "BLHI4-",
    "LINO41", "LINO42",
    "THAW11", "THAW12",
    "ORMO11",
    "BOSO11",
    "GUNS11",
    "SALL11", "SALL12",
    "RORE11", "RORW11",
    "WAAO11",
    "WABO11",
    "NECT41", "NECT4A",
    "GANW14",
    "BRST42", "BRST22",
    "WALP41",
    "HEYS41", "HEYS11",
    "BRFO41",
    "NORM41",
    "STAH11",
    "RICH41",
    "HARK11", "HARK12",
    "SIZE41",
]

transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

print()
header = f"{'Bus':15s} {'X(OSGB)':>10s} {'Y(OSGB)':>10s} {'Lat':>8s} {'Lon':>8s}  Present?"
print(header)
print("-" * len(header))

for bus in sorted(set(wf_buses)):
    if bus in base.buses.index:
        x = base.buses.loc[bus, "x"]
        y = base.buses.loc[bus, "y"]
        lon, lat = transformer.transform(x, y)
        print(f"{bus:15s} {x:10.0f} {y:10.0f} {lat:8.3f} {lon:8.3f}  YES")
    else:
        print(f"{bus:15s} {'':>10s} {'':>10s} {'':>8s} {'':>8s}  NO")

# Also find ALL buses that are likely offshore wind platforms
# These often have names ending in 4A, 4B, or contain WF/OWF
print("\n\n=== BUSES LIKELY TO BE OFFSHORE (name patterns) ===")
for bus in sorted(base.buses.index):
    x = base.buses.loc[bus, "x"]
    y = base.buses.loc[bus, "y"]
    lon, lat = transformer.transform(x, y)
    # Check if this is a wind farm node (connected via Extra_WF_edges)
    # or if it has unusual name patterns
    if any(bus.startswith(p) for p in ["BEAT4", "CREB2", "LONO4", "BODE4", "BLHI4",
                                        "LINO4", "THAW1", "ORMO1", "BOSO1", "GUNS1",
                                        "SALL1", "RORE1", "RORW1", "WAAO1", "WABO1",
                                        "NECT4", "GANW1"]):
        print(f"  {bus:15s}  lat={lat:.3f}, lon={lon:.3f}  (x={x:.0f}, y={y:.0f})")

# Now check: what does the ACTUAL ETYS data say about these locations?
# Load ETYS appendix for node info
print("\n\n=== OFFSHORE WIND GENERATORS: CURRENT BUS vs ACTUAL LOCATION ===")
ow = base.generators[base.generators.carrier == "wind_offshore"]
print(f"Offshore wind generators: {len(ow)}")
print()
print(f"{'Generator':50s} {'p_nom':>8s} {'Bus':>15s} {'Bus_lat':>8s} {'Bus_lon':>8s}")
print("-" * 100)
for gen, row in ow.iterrows():
    bus = row["bus"]
    if bus in base.buses.index:
        x = base.buses.loc[bus, "x"]
        y = base.buses.loc[bus, "y"]
        lon, lat = transformer.transform(x, y)
        print(f"{str(gen)[:50]:50s} {row['p_nom']:8.0f} {bus:>15s} {lat:8.3f} {lon:8.3f}")
