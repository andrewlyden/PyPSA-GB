"""Check the flexibility network that's input to finalize_network"""
import pypsa

print("Loading flexibility network...")
n = pypsa.Network('resources/network/HT35_flex_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc')

print(f"\nGenerators: {len(n.generators)}")
print(f"Generators by carrier:")
carrier_counts = n.generators.groupby('carrier').size().sort_values(ascending=False)
for carrier, count in carrier_counts.items():
    capacity = n.generators[n.generators.carrier == carrier]['p_nom'].sum()
    print(f"  {carrier:30s}: {count:4d} units, {capacity:10.1f} MW")

print(f"\nLoads: {len(n.loads)}")
print(f"Links: {len(n.links)}")
print(f"Stores: {len(n.stores)}")
print(f"Storage units: {len(n.storage_units)}")
