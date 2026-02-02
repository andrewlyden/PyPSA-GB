"""Check the interconnectors network"""
import pypsa

print("Loading interconnectors network...")
n = pypsa.Network('resources/network/HT35_flex_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc')

print(f"\nGenerators: {len(n.generators)}")
carrier_counts = n.generators.groupby('carrier').size().sort_values(ascending=False)
print(f"Generator counts by carrier:")
for carrier, count in carrier_counts.head(10).items():
    print(f"  {carrier:30s}: {count:4d} units")

total_non_ls = len(n.generators[n.generators.carrier != 'load_shedding'])
print(f"\nNon-load-shedding generators: {total_non_ls}")
