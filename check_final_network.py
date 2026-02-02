"""Check the HT35_flex demand network"""
from scripts.utilities.network_io import load_network

print("Loading HT35_flex_network_demand.pkl...")
n = load_network('resources/network/HT35_flex_network_demand.pkl')

print(f"\nGenerators: {len(n.generators)}")
carrier_counts = n.generators.groupby('carrier').size().sort_values(ascending=False)
print(f"Generator counts by carrier:")
for carrier, count in carrier_counts.head(15).items():
    capacity = n.generators[n.generators.carrier == carrier]['p_nom'].sum()
    print(f"  {carrier:30s}: {count:4d} units, {capacity:10.1f} MW")

total_non_ls = len(n.generators[n.generators.carrier != 'load_shedding'])
print(f"\nNon-load-shedding generators: {total_non_ls}")
print(f"Loads: {len(n.loads)}")
