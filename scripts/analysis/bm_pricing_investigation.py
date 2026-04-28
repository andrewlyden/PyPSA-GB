"""Investigate BM bid/offer pricing: ELEXON vs derived breakdown."""
import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\network\Validation_2020.nc")
gens = n.generators

bmu = pd.read_csv(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\generators\Validation_2020_bmu_mapping.csv")
mapped_gen_names = set(bmu["generator_name"].unique())

elexon_offers = pd.read_csv(
    r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\market\Validation_2020\elexon\elexon_offers.csv",
    index_col=0,
)
elexon_bids = pd.read_csv(
    r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\market\Validation_2020\elexon\elexon_bids.csv",
    index_col=0,
)

# Map BMU columns to generator names
bmu_to_gen = dict(zip(bmu["bmu_id"], bmu["generator_name"]))
offers_mapped = elexon_offers.rename(columns=bmu_to_gen)
bids_mapped = elexon_bids.rename(columns=bmu_to_gen)
# Average duplicates (multiple BMUs per generator)
if offers_mapped.columns.has_duplicates:
    offers_mapped = offers_mapped.T.groupby(level=0).mean().T
if bids_mapped.columns.has_duplicates:
    bids_mapped = bids_mapped.T.groupby(level=0).mean().T

print("=" * 80)
print("BM BID/OFFER PRICING INVESTIGATION")
print("=" * 80)

# Section 1: Match rate by carrier
print("\n--- 1. ELEXON MATCH RATE BY CARRIER ---")
summary = []
for carrier in gens.carrier.unique():
    g = gens[gens.carrier == carrier]
    m = g.index[g.index.isin(mapped_gen_names)]
    u = g.index[~g.index.isin(mapped_gen_names)]
    summary.append({
        "carrier": carrier,
        "total_gens": len(g),
        "matched_gens": len(m),
        "unmatched_gens": len(u),
        "matched_MW": g.loc[m, "p_nom"].sum(),
        "unmatched_MW": g.loc[u, "p_nom"].sum(),
        "pct_matched_by_capacity": 100 * g.loc[m, "p_nom"].sum() / g["p_nom"].sum() if g["p_nom"].sum() > 0 else 0,
    })

summary_df = pd.DataFrame(summary).sort_values("matched_MW", ascending=False)
summary_df = summary_df[summary_df.total_gens > 0]
print(summary_df.to_string(index=False))
print(f"\nOverall: {summary_df.matched_gens.sum()}/{summary_df.total_gens.sum()} generators matched")
print(f"Capacity: {summary_df.matched_MW.sum():.0f}/{(summary_df.matched_MW + summary_df.unmatched_MW).sum():.0f} MW ({100*summary_df.matched_MW.sum()/((summary_df.matched_MW + summary_df.unmatched_MW).sum()):.1f}%)")

# Section 2: What ELEXON prices look like for matched generators
print("\n--- 2. ELEXON MATCHED GENERATOR PRICES ---")
for carrier in ["CCGT", "nuclear", "wind_onshore", "wind_offshore", "coal", "OCGT", "large_hydro"]:
    g = gens[gens.carrier == carrier]
    m = g.index[g.index.isin(mapped_gen_names) & g.index.isin(offers_mapped.columns)]
    if len(m) == 0:
        continue
    offer_prices = offers_mapped[m].mean()
    bid_prices = -bids_mapped[m].mean()  # Negate for ESO cost convention
    print(f"\n  {carrier} ({len(m)} matched):")
    print(f"    ELEXON Offer £/MWh: mean={offer_prices.mean():.1f}, min={offer_prices.min():.1f}, max={offer_prices.max():.1f}")
    print(f"    ELEXON Bid £/MWh:   mean={bid_prices.mean():.1f}, min={bid_prices.min():.1f}, max={bid_prices.max():.1f}")

# Section 3: What derived prices look like for unmatched generators
print("\n--- 3. DERIVED (FORMULA) PRICES FOR UNMATCHED GENERATORS ---")
print("carrier_overrides from defaults.yaml:")
print("  CCGT:          markup mode, offer +50%, bid -10%")
print("  nuclear:       absolute, offer=£999, bid=£150")
print("  wind_onshore:  absolute, offer=£0, bid=£45")
print("  wind_offshore: absolute, offer=£0, bid=£90")
print("  large_hydro:   absolute, offer=£0, bid=£5")
print("  Pumped Storage: absolute, offer=£50, bid=£60")
print("  Battery:       absolute, offer=£120, bid=£50")

# Get time-varying MC for each carrier
mc_t = n.generators_t.get("marginal_cost", pd.DataFrame())
for carrier in ["CCGT", "nuclear", "wind_onshore", "wind_offshore", "coal", "OCGT"]:
    g = gens[gens.carrier == carrier]
    u = g.index[~g.index.isin(mapped_gen_names)]
    if len(u) == 0:
        print(f"\n  {carrier}: ALL matched (0 unmatched)")
        continue
    
    # Get MC for unmatched generators
    if not mc_t.empty:
        mc_cols = [c for c in u if c in mc_t.columns]
        if mc_cols:
            mc_vals = mc_t[mc_cols].mean().mean()
        else:
            mc_vals = g.loc[u, "marginal_cost"].mean()
    else:
        mc_vals = g.loc[u, "marginal_cost"].mean()
    
    print(f"\n  {carrier} ({len(u)} unmatched, {g.loc[u, 'p_nom'].sum():.0f} MW):")
    print(f"    Mean MC: £{mc_vals:.1f}/MWh")
    if carrier == "CCGT":
        print(f"    Derived offer: £{mc_vals * 1.50:.1f}/MWh  (mc × 1.50)")
        print(f"    Derived bid:   £{mc_vals * 0.90:.1f}/MWh  (mc × 0.90)")
    elif carrier in ["wind_onshore"]:
        print(f"    Derived offer: £0/MWh  (absolute override)")
        print(f"    Derived bid:   £45/MWh (absolute override)")

# Section 4: The blended effective price (capacity-weighted)
print("\n--- 4. BLENDED EFFECTIVE PRICES (ELEXON + DERIVED) ---")
for carrier in ["CCGT", "nuclear", "wind_onshore", "wind_offshore", "coal", "OCGT", "large_hydro"]:
    g = gens[gens.carrier == carrier]
    m = g.index[g.index.isin(mapped_gen_names) & g.index.isin(offers_mapped.columns)]
    u = g.index[~g.index.isin(mapped_gen_names)]
    
    if len(g) == 0:
        continue
    
    # ELEXON-matched prices
    elexon_offer = offers_mapped[m].mean().values if len(m) > 0 else np.array([])
    elexon_bid = (-bids_mapped[m].mean()).values if len(m) > 0 else np.array([])
    elexon_pnom = g.loc[m, "p_nom"].values if len(m) > 0 else np.array([])
    
    # Derived prices for unmatched
    if len(u) > 0:
        if not mc_t.empty:
            mc_cols = [c for c in u if c in mc_t.columns]
            if mc_cols:
                mc_vals = mc_t[mc_cols].mean()
            else:
                mc_vals = g.loc[u, "marginal_cost"]
        else:
            mc_vals = g.loc[u, "marginal_cost"]
        
        # Apply carrier overrides
        if carrier == "CCGT":
            derived_offer = mc_vals * 1.50
            derived_bid = mc_vals * 0.90
        elif carrier == "nuclear":
            derived_offer = pd.Series(999.0, index=u)
            derived_bid = pd.Series(150.0, index=u)
        elif carrier == "wind_onshore":
            derived_offer = pd.Series(0.0, index=u)
            derived_bid = pd.Series(45.0, index=u)
        elif carrier == "wind_offshore":
            derived_offer = pd.Series(0.0, index=u)
            derived_bid = pd.Series(90.0, index=u)
        elif carrier == "large_hydro":
            derived_offer = pd.Series(0.0, index=u)
            derived_bid = pd.Series(5.0, index=u)
        else:
            # Default markup
            derived_offer = mc_vals * 1.10
            derived_bid = mc_vals * 0.90
        
        derived_pnom = g.loc[u, "p_nom"].values
    else:
        derived_offer = np.array([])
        derived_bid = np.array([])
        derived_pnom = np.array([])
    
    # Capacity-weighted blend
    all_offers = np.concatenate([elexon_offer, derived_offer.values if hasattr(derived_offer, 'values') else derived_offer])
    all_bids = np.concatenate([elexon_bid, derived_bid.values if hasattr(derived_bid, 'values') else derived_bid])
    all_pnom = np.concatenate([elexon_pnom, derived_pnom])
    
    if all_pnom.sum() > 0:
        blend_offer = np.average(all_offers, weights=all_pnom)
        blend_bid = np.average(all_bids, weights=all_pnom)
    else:
        blend_offer = all_offers.mean() if len(all_offers) > 0 else 0
        blend_bid = all_bids.mean() if len(all_bids) > 0 else 0
    
    elex_part = elexon_pnom.sum() / all_pnom.sum() * 100 if all_pnom.sum() > 0 else 0
    print(f"\n  {carrier}:")
    print(f"    ELEXON coverage: {elex_part:.0f}% by capacity ({len(m)}/{len(g)} gens)")
    if len(m) > 0:
        print(f"    ELEXON offer: £{np.average(elexon_offer, weights=elexon_pnom):.1f}/MWh (cap-weighted)")
        print(f"    ELEXON bid:   £{np.average(elexon_bid, weights=elexon_pnom):.1f}/MWh (cap-weighted)")
    if len(u) > 0:
        print(f"    Derived offer: £{np.average(derived_offer.values if hasattr(derived_offer, 'values') else derived_offer, weights=derived_pnom):.1f}/MWh (cap-weighted)")
        print(f"    Derived bid:   £{np.average(derived_bid.values if hasattr(derived_bid, 'values') else derived_bid, weights=derived_pnom):.1f}/MWh (cap-weighted)")
    print(f"    BLENDED offer: £{blend_offer:.1f}/MWh")
    print(f"    BLENDED bid:   £{blend_bid:.1f}/MWh")

# Section 5: Why only 64 matched? Check what's in BMU mapping but NOT in network
print("\n--- 5. MAPPING QUALITY DIAGNOSTICS ---")
not_in_network = [g for g in mapped_gen_names if g not in gens.index]
print(f"BMU mapping generator names NOT in network: {len(not_in_network)}")
if not_in_network:
    for g in sorted(not_in_network)[:10]:
        print(f"  - {g}")

# Check BMU IDs in mapping that aren't in ELEXON data
bmu_ids_in_mapping = set(bmu["bmu_id"])
bmu_ids_in_elexon = set(elexon_offers.columns)
missing_from_elexon = bmu_ids_in_mapping - bmu_ids_in_elexon
print(f"\nBMU IDs in mapping but NOT in ELEXON data: {len(missing_from_elexon)}")
if missing_from_elexon:
    for b in sorted(missing_from_elexon)[:10]:
        gen = bmu[bmu["bmu_id"] == b]["generator_name"].values[0]
        print(f"  - {b} -> {gen}")

# Section 6: The bottom line - what's actually determining costs
print("\n--- 6. BOTTOM LINE: WHAT DRIVES BM COSTS ---")
print("""
The model has 4,164 generators. Of these:
  - 64 (1.5%) get ELEXON-sourced bid/offer prices (realistic)
  - 4,100 (98.5%) get derived/formula bid/offer prices

By CAPACITY the picture is better for conventional:
  - CCGT: 23,110 MW ELEXON (78%) vs 6,563 MW derived (22%)
  - But wind/solar/embedded: nearly 100% derived

The derived prices use carrier_overrides from defaults.yaml:
  - CCGT offer = mc × 1.50 ≈ £54/MWh (vs ELEXON ~£90/MWh)
  - CCGT bid = mc × 0.90 ≈ £32/MWh (vs ELEXON ~varies)
  - Wind bid = £45/MWh flat (absolute override)

KEY INSIGHT: The 64 matched generators DO use ELEXON prices.
The problem is the 4,100 unmatched generators using formula 
pricing that systematically underestimates BM costs.

Most BM action in reality is from ~100 large dispatchable units.
Our BMU mapping covers 69 of these. The 4,000+ small embedded 
generators (solar, wind, landfill, biogas) rarely participate 
in the real BM but DO participate in the model's BM because 
the copperplate→constrained redispatch affects them.
""")
