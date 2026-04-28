"""Analyse unmatched generators to identify BMU mapping expansion opportunities."""
import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\network\Validation_2020.nc")
gens = n.generators
bmu = pd.read_csv(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\generators\Validation_2020_bmu_mapping.csv")
mapped = set(bmu["generator_name"].unique())
offers = pd.read_csv(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\market\Validation_2020\elexon\elexon_offers.csv", index_col=0, nrows=1)

# Which ELEXON BMU IDs are NOT yet in our mapping?
mapped_bmu_ids = set(bmu["bmu_id"])
all_elexon_bmus = set(offers.columns)
unmapped_elexon = sorted(all_elexon_bmus - mapped_bmu_ids)

# Filter to T_ (transmission) BMUs only
unmapped_t = [b for b in unmapped_elexon if b.startswith("T_")]
print(f"ELEXON BMUs not in mapping: {len(unmapped_elexon)} total, {len(unmapped_t)} transmission (T_)")
print()

# Show the large unmatched generators by carrier  
unmatched_gens = gens[~gens.index.isin(mapped)]
large_unmatched = unmatched_gens.nlargest(30, "p_nom")[["carrier", "p_nom", "bus"]]
print("Top 30 unmatched generators by capacity:")
for idx, row in large_unmatched.iterrows():
    print(f"  {row['p_nom']:8.1f} MW  {row['carrier']:20s}  {idx}")

# Show unmatched T_ BMUs grouped by prefix
print("\nUnmatched T_ BMU prefixes (potential new mappings):")
prefix_counts = {}
for bmu_id in unmapped_t:
    core = bmu_id[2:]  # Strip T_
    prefix = core[:4]
    if prefix not in prefix_counts:
        prefix_counts[prefix] = []
    prefix_counts[prefix].append(bmu_id)

# Sort by count (most BMUs first)  
for prefix, bmus in sorted(prefix_counts.items(), key=lambda x: -len(x[1])):
    if len(bmus) >= 2:
        print(f"  {prefix}: {len(bmus)} BMUs - {bmus[:5]}")

# Show CCGT unmatched specifically with their names
print("\nUnmatched CCGT generators (potential mapping targets):")
ccgt_unmatched = unmatched_gens[unmatched_gens.carrier == "CCGT"].sort_values("p_nom", ascending=False)
for idx, row in ccgt_unmatched.iterrows():
    print(f"  {row['p_nom']:8.1f} MW  {idx}")

# Show unmatched wind farms > 50MW
print("\nUnmatched wind generators > 50 MW:")
wind_unmatched = unmatched_gens[unmatched_gens.carrier.isin(["wind_onshore", "wind_offshore"]) & (unmatched_gens.p_nom > 50)].sort_values("p_nom", ascending=False)
for idx, row in wind_unmatched.head(30).iterrows():
    print(f"  {row['p_nom']:8.1f} MW  {row['carrier']:20s}  {idx}")
