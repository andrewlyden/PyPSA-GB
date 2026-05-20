"""Test expanded BMU mapping - compare old vs new coverage."""
import sys
sys.path.insert(0, ".")
from scripts.generators.build_bmu_mapping import build_bmu_mapping
import pandas as pd

network_path = r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\network\Validation_2020.nc"
elexon_path = r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\market\Validation_2020\elexon\elexon_offers.csv"

# Old mapping (existing file)
old_mapping = pd.read_csv(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\generators\Validation_2020_bmu_mapping.csv")
print(f"OLD mapping: {len(old_mapping)} BMU entries, {old_mapping['generator_name'].nunique()} unique generators")

# New mapping (with expanded prefixes + ELEXON supplementary pass)
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test_mapping")

new_mapping = build_bmu_mapping(
    network_path=network_path,
    elexon_offers_path=elexon_path,
    logger=logger,
)
print(f"\nNEW mapping: {len(new_mapping)} BMU entries, {new_mapping['generator_name'].nunique()} unique generators")

# Compare
old_gens = set(old_mapping["generator_name"].unique())
new_gens = set(new_mapping["generator_name"].unique())
added_gens = new_gens - old_gens
removed_gens = old_gens - new_gens

print(f"\nGenerators ADDED: {len(added_gens)}")
for g in sorted(added_gens):
    rows = new_mapping[new_mapping["generator_name"] == g]
    carrier = rows["carrier"].iloc[0] if "carrier" in rows.columns else "?"
    n_bmus = len(rows)
    print(f"  + {g} ({carrier}, {n_bmus} BMUs)")

if removed_gens:
    print(f"\nGenerators REMOVED: {len(removed_gens)}")
    for g in sorted(removed_gens):
        print(f"  - {g}")

print(f"\nMatch method breakdown:")
print(new_mapping["match_method"].value_counts().to_string())

# Coverage comparison
import pypsa
n = pypsa.Network(network_path)
gens = n.generators
for label, mapping in [("OLD", old_mapping), ("NEW", new_mapping)]:
    mapped_names = set(mapping["generator_name"].unique())
    matched = gens.index[gens.index.isin(mapped_names)]
    total_cap = gens["p_nom"].sum()
    matched_cap = gens.loc[matched, "p_nom"].sum()
    print(f"\n{label}: {len(matched)}/{len(gens)} generators ({matched_cap:.0f}/{total_cap:.0f} MW = {100*matched_cap/total_cap:.1f}%)")
    
    # By carrier
    for carrier in ["CCGT", "nuclear", "wind_offshore", "wind_onshore", "coal", "OCGT", "large_hydro"]:
        g = gens[gens.carrier == carrier]
        m = g.index[g.index.isin(mapped_names)]
        if len(g) > 0:
            print(f"  {carrier:20s}: {len(m)}/{len(g)} gens, {g.loc[m, 'p_nom'].sum():.0f}/{g['p_nom'].sum():.0f} MW ({100*g.loc[m, 'p_nom'].sum()/g['p_nom'].sum():.0f}%)")
