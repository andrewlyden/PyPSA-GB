"""Get ELEXON carrier stats for tuning carrier_overrides."""
import pypsa, pandas as pd, numpy as np

n = pypsa.Network(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\network\Validation_2020.nc")
gens = n.generators
bmu = pd.read_csv(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\generators\Validation_2020_bmu_mapping.csv")
mapped = set(bmu["generator_name"].unique())
offers = pd.read_csv(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\market\Validation_2020\elexon\elexon_offers.csv", index_col=0)
bids = pd.read_csv(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\resources\market\Validation_2020\elexon\elexon_bids.csv", index_col=0)
bmu_to_gen = dict(zip(bmu["bmu_id"], bmu["generator_name"]))
offers_m = offers.rename(columns=bmu_to_gen)
bids_m = bids.rename(columns=bmu_to_gen)
if offers_m.columns.has_duplicates:
    offers_m = offers_m.T.groupby(level=0).mean().T
if bids_m.columns.has_duplicates:
    bids_m = bids_m.T.groupby(level=0).mean().T

print("Per-carrier: median of generator means (robust to outliers)")
print(f"{'Carrier':20s} {'Offer_med':>10s} {'Offer_p25':>10s} {'Offer_p75':>10s} {'Bid_med':>10s} {'Bid_p25':>10s} {'Bid_p75':>10s} {'N':>4s}")
matched_mask = gens.index.isin(mapped) & gens.index.isin(offers_m.columns)
for carrier in sorted(gens.loc[matched_mask, "carrier"].unique()):
    cgens = gens.index[matched_mask & (gens["carrier"] == carrier)]
    gen_offers = pd.Series({g: offers_m[g].mean() for g in cgens})
    gen_bids = pd.Series({g: -bids_m[g].mean() for g in cgens})
    print(f"{carrier:20s} {gen_offers.median():10.1f} {gen_offers.quantile(0.25):10.1f} {gen_offers.quantile(0.75):10.1f} {gen_bids.median():10.1f} {gen_bids.quantile(0.25):10.1f} {gen_bids.quantile(0.75):10.1f} {len(cgens):4d}")

# CCGT MC for reference
mc_t = n.generators_t.get("marginal_cost", pd.DataFrame())
ccgt_gens = gens.index[(gens.carrier == "CCGT") & matched_mask]
if not mc_t.empty:
    mc_cols = [c for c in ccgt_gens if c in mc_t.columns]
    if mc_cols:
        mc_mean = mc_t[mc_cols].mean().mean()
        ccgt_offers = pd.Series({g: offers_m[g].mean() for g in ccgt_gens})
        ccgt_bids = pd.Series({g: -bids_m[g].mean() for g in ccgt_gens})
        print(f"\nCCGT MC reference: mean={mc_mean:.1f}")
        print(f"  Median offer/MC ratio: {ccgt_offers.median() / mc_mean:.2f}")
        print(f"  p25 offer/MC ratio: {ccgt_offers.quantile(0.25) / mc_mean:.2f}")
        print(f"  p75 offer/MC ratio: {ccgt_offers.quantile(0.75) / mc_mean:.2f}")

# OCGT MC for reference
ocgt_gens = gens.index[(gens.carrier == "OCGT") & matched_mask]
if not mc_t.empty and len(ocgt_gens) > 0:
    mc_cols = [c for c in ocgt_gens if c in mc_t.columns]
    if mc_cols:
        mc_mean = mc_t[mc_cols].mean().mean()
        ocgt_offers = pd.Series({g: offers_m[g].mean() for g in ocgt_gens})
        print(f"\nOCGT MC reference: mean={mc_mean:.1f}")
        print(f"  Median offer/MC ratio: {ocgt_offers.median() / mc_mean:.2f}")
