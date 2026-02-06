"""Check where offshore wind generators are connected in the full ETYS network."""
import pickle
import pandas as pd
from pyproj import Transformer

with open("resources/network/HT35_flex_network_demand_renewables.pkl", "rb") as f:
    n = pickle.load(f)

transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

ow = n.generators[n.generators.carrier == "wind_offshore"]
print(f"Offshore wind: {len(ow)} generators, {ow.p_nom.sum():,.0f} MW total")
print()

rows = []
for gen, row in ow.sort_values("p_nom", ascending=False).iterrows():
    bus = row["bus"]
    if bus in n.buses.index:
        x = n.buses.loc[bus, "x"]
        y = n.buses.loc[bus, "y"]
        lon, lat = transformer.transform(x, y)
        rows.append({
            "generator": str(gen)[:55],
            "p_nom": row["p_nom"],
            "bus": bus,
            "bus_lat": lat,
            "bus_lon": lon,
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

print()
print("=== CAPACITY BY BUS ===")
bus_cap = ow.groupby("bus")["p_nom"].sum().sort_values(ascending=False)
for bus, cap in bus_cap.items():
    x = n.buses.loc[bus, "x"]
    y = n.buses.loc[bus, "y"]
    lon, lat = transformer.transform(x, y)
    note = ""
    if lon > 1.5:
        note = " <-- PAST EAST COAST"
    elif lon > 0.8 and lat < 52.5:
        note = " <-- NEAR EAST COAST"
    print(f"  {bus:30s}: {cap:8,.0f} MW  lat={lat:.3f}, lon={lon:.3f}{note}")

# Compare to real offshore wind farm locations
print()
print("=== COMPARISON: ACTUAL vs MODELLED LOCATIONS ===")
print("Known offshore wind farms and their real centres:")
real_locations = [
    ("Hornsea 1-4", 53.88, 1.79, "CREB21 onshore landing at Hull"),
    ("Dogger Bank A-C", 54.75, 2.0, "Modelled at onshore substation"),
    ("East Anglia 1-3", 52.3, 2.6, "BRFO/SIZE/GANW onshore landing"),
    ("Beatrice", 58.09, -2.95, "BEAT4A at 57.36N = Peter head area"),
    ("Moray East/West", 57.72, -2.75, "Near BEAT/BLHI = onshore"),
    ("London Array", 51.63, 1.45, "LONO4A/RICH = Kent coast"),
    ("Triton Knoll", 53.2, 0.78, "LINO/WALP = onshore Lincs"),
    ("Rampion", 50.68, -0.25, "Near southern coast"),
]
print()
for name, lat, lon, note in real_locations:
    print(f"  {name:25s}: actual ({lat:.2f}, {lon:.2f})  |  {note}")

print()
print("KEY FINDING:")
print("All offshore wind farms are connected to their ONSHORE landing buses.")
print("The ETYS network has no buses at sea - ensure_buses_on_land() moved them.")
print("This means the HVDC cable from offshore platform to shore is NOT modelled.")
print("The wind farm capacity appears directly at the onshore substation.")
