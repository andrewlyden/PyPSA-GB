"""Diagnose ELEXON per-generator price distributions to understand carrier_average issues."""
import sys
sys.path.insert(0, ".")

import pypsa
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("diag")

# Load network and ELEXON data
n = pypsa.Network("resources/network/Validation_2020.nc")
bmu_map = pd.read_csv("resources/generators/Validation_2020_bmu_mapping.csv")
elexon_offers = pd.read_csv("resources/market/Validation_2020/elexon/elexon_offers.csv", index_col=0)
elexon_bids = pd.read_csv("resources/market/Validation_2020/elexon/elexon_bids.csv", index_col=0)

# Map BMUs to generators
bmu_to_gen = dict(zip(bmu_map["bmu_id"], bmu_map["generator_name"]))
offers = elexon_offers.rename(columns=bmu_to_gen)
bids = elexon_bids.rename(columns=bmu_to_gen)
if offers.columns.has_duplicates:
    offers = offers.T.groupby(level=0).mean().T
if bids.columns.has_duplicates:
    bids = bids.T.groupby(level=0).mean().T

# Per-generator stats
matched_gens = [g for g in n.generators.index if g in offers.columns]
print(f"{'Generator':40s} {'Carrier':20s} {'p_nom':>8s} | "
      f"{'Offer mean':>10s} {'Offer med':>10s} {'Offer p5':>10s} {'Offer p95':>10s} | "
      f"{'Bid mean':>10s} {'Bid med':>10s}")
print("-" * 170)

for carrier in ["CCGT", "OCGT", "nuclear", "coal", "wind_offshore", "wind_onshore", "large_hydro", "oil"]:
    carrier_gens = [g for g in matched_gens if n.generators.loc[g, "carrier"] == carrier]
    if not carrier_gens:
        continue
    print(f"\n  ── {carrier} ──")
    for g in sorted(carrier_gens, key=lambda x: -n.generators.loc[x, "p_nom"]):
        pnom = n.generators.loc[g, "p_nom"]
        o = offers[g]
        b = bids[g]
        print(f"  {g:38s} {carrier:20s} {pnom:8.0f} | "
              f"  £{o.mean():8.1f}  £{o.median():8.1f}  £{o.quantile(0.05):8.1f}  £{o.quantile(0.95):8.1f} | "
              f"  £{(-b).mean():8.1f}  £{(-b).median():8.1f}")

# Show carrier-level summary: mean vs median
print(f"\n{'='*80}")
print("CARRIER AGGREGATION: Mean vs Median")
print(f"{'='*80}")
print(f"{'Carrier':20s} | {'Cap-Wt Mean Offer':>18s} {'Median Offer':>15s} | "
      f"{'Cap-Wt Mean Bid':>18s} {'Median Bid':>15s}")

for carrier in ["CCGT", "OCGT", "nuclear", "coal", "wind_offshore", "wind_onshore", "large_hydro"]:
    carrier_gens = [g for g in matched_gens if n.generators.loc[g, "carrier"] == carrier]
    if not carrier_gens:
        continue
    p_noms = n.generators.loc[carrier_gens, "p_nom"]
    
    gen_offers = pd.Series({g: offers[g].mean() for g in carrier_gens})
    gen_bids = pd.Series({g: -bids[g].mean() for g in carrier_gens})
    
    # Cap-weighted mean
    wt_offer = (gen_offers * p_noms).sum() / p_noms.sum()
    wt_bid = (gen_bids * p_noms).sum() / p_noms.sum()
    
    # Simple median
    med_offer = gen_offers.median()
    med_bid = gen_bids.median()
    
    print(f"  {carrier:18s} | £{wt_offer:16.1f} £{med_offer:13.1f} | "
          f"£{wt_bid:16.1f} £{med_bid:13.1f}")
