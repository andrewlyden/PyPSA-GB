"""Temporary script to analyze biogas/landfill/biofuel dispatch from wholesale results."""
import pandas as pd
import pypsa
import sys

# Load the wholesale network
network_path = r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\market\Test_Rolling_Market_wholesale.nc"
n = pypsa.Network(network_path)

# Define biogas/biofuel carriers
biofuel_carriers = ['biogas', 'landfill_gas', 'sewage_gas', 'advanced_biofuel', 'biomass']

# Get generators with these carriers
bio_gens = n.generators[n.generators.carrier.isin(biofuel_carriers)]
print(f"\n=== Biogas/Biofuel Generators in Network ===")
print(f"Total generators: {len(bio_gens)}")
print(f"Total installed capacity (p_nom): {bio_gens.p_nom.sum():.1f} MW")
print(f"\nBy carrier:")
for carrier in biofuel_carriers:
    gens = bio_gens[bio_gens.carrier == carrier]
    if len(gens) > 0:
        print(f"  {carrier}: {len(gens)} generators, {gens.p_nom.sum():.1f} MW capacity")

# Get dispatch from generators_t.p
print(f"\n=== Dispatch Analysis ===")
if hasattr(n, 'generators_t') and 'p' in n.generators_t:
    dispatch = n.generators_t.p
    bio_dispatch = dispatch[[g for g in bio_gens.index if g in dispatch.columns]]

    total_dispatch = bio_dispatch.sum(axis=1)
    print(f"Number of snapshots: {len(total_dispatch)}")
    print(f"Mean total dispatch: {total_dispatch.mean():.1f} MW")
    print(f"Max total dispatch: {total_dispatch.max():.1f} MW")
    print(f"Min total dispatch: {total_dispatch.min():.1f} MW")

    print(f"\nBy carrier mean dispatch:")
    for carrier in biofuel_carriers:
        carrier_gens = [g for g in bio_gens[bio_gens.carrier == carrier].index if g in dispatch.columns]
        if carrier_gens:
            carrier_dispatch = dispatch[carrier_gens].sum(axis=1)
            print(f"  {carrier}: {carrier_dispatch.mean():.1f} MW (capacity factor: {carrier_dispatch.mean() / bio_gens[bio_gens.carrier == carrier].p_nom.sum() * 100:.1f}%)")
else:
    print("No dispatch data (generators_t.p) found in network")

# Check marginal costs
print(f"\n=== Marginal Costs ===")
for carrier in biofuel_carriers:
    gens = bio_gens[bio_gens.carrier == carrier]
    if len(gens) > 0:
        mc = gens.marginal_cost
        print(f"  {carrier}: mean={mc.mean():.2f}, min={mc.min():.2f}, max={mc.max():.2f} £/MWh")

# Check p_min_pu and p_max_pu
print(f"\n=== Operating Constraints ===")
print(f"p_min_pu values: {bio_gens.p_min_pu.unique()}")
print(f"p_max_pu values: {bio_gens.p_max_pu.unique()}")
