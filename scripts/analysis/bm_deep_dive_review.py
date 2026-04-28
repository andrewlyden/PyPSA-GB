"""
BM Deep Dive Review — Validation_2020
Compare model BM volumes, costs, and pricing against ELEXON benchmarks.
Identify systematic issues in redispatch behaviour.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1")
MARKET = ROOT / "resources" / "market"
ANALYSIS = ROOT / "resources" / "analysis"
SCENARIO = "Validation_2020"

print("=" * 90)
print("BM DEEP DIVE REVIEW — Validation_2020 (Full Year 2020, 8784 hours)")
print("=" * 90)

# ─── 1. LOAD FILES ─────────────────────────────────────────────────────────────
costs = pd.read_csv(MARKET / f"{SCENARIO}_constraint_costs.csv")
redispatch = pd.read_csv(MARKET / f"{SCENARIO}_redispatch_summary.csv")
validation = pd.read_csv(MARKET / f"{SCENARIO}_bm_validation.csv")
prices = pd.read_csv(MARKET / f"{SCENARIO}_price_comparison.csv", index_col=0, parse_dates=True)
congestion = pd.read_csv(MARKET / f"{SCENARIO}_congestion.csv")

# Load wholesale and balancing dispatch
wholesale_gen = pd.read_csv(MARKET / f"{SCENARIO}_wholesale_dispatch.csv", index_col=0, parse_dates=True)
balancing_gen = pd.read_csv(MARKET / f"{SCENARIO}_balancing_dispatch.csv", index_col=0, parse_dates=True)

# Try loading storage dispatch
try:
    wholesale_su = pd.read_csv(MARKET / f"{SCENARIO}_wholesale_storage.csv", index_col=0, parse_dates=True)
    balancing_su = pd.read_csv(MARKET / f"{SCENARIO}_balancing_storage.csv", index_col=0, parse_dates=True)
    has_storage = True
except:
    has_storage = False

# Load market summary
summary_path = ANALYSIS / f"{SCENARIO}_market_summary.json"
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)

# Load network for generator metadata
import pypsa
net_path = MARKET / f"{SCENARIO}_balancing.nc"
if net_path.exists():
    network = pypsa.Network(str(net_path))
else:
    network = None

print("\n✓ All files loaded successfully")
print(f"  Wholesale dispatch: {wholesale_gen.shape}")
print(f"  Balancing dispatch: {balancing_gen.shape}")
if has_storage:
    print(f"  Wholesale storage: {wholesale_su.shape}")
    print(f"  Balancing storage: {balancing_su.shape}")

# ─── 2. COST OVERVIEW ──────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 1: TOTAL BM COST COMPARISON")
print("=" * 90)

total_row = costs[costs.carrier == "TOTAL"].iloc[0]
model_cost = total_row["net_cost"]
elexon_cost = 1_400_000_000  # NESO published benchmark for 2020

print(f"\n  Model total BM cost:    £{model_cost:>14,.0f}")
print(f"  ELEXON 2020 benchmark:  £{elexon_cost:>14,.0f}")
print(f"  Ratio:                  {model_cost/elexon_cost:.3f}x ({model_cost/elexon_cost*100:.1f}%)")
print(f"\n  Model offer cost (ESO pays generators to turn UP):   £{total_row['offer_cost']:>14,.0f}")
print(f"  Model bid cost (generators pay ESO to turn DOWN):    £{total_row['bid_cost']:>14,.0f}")
print(f"  Net cost to ESO:                                     £{total_row['net_cost']:>14,.0f}")

print(f"\n  Total increase volume: {total_row['increase_MWh']:>12,.0f} MWh ({total_row['increase_MWh']/1e6:.2f} TWh)")
print(f"  Total decrease volume: {total_row['decrease_MWh']:>12,.0f} MWh ({total_row['decrease_MWh']/1e6:.2f} TWh)")

# ─── 3. CARRIER-LEVEL BREAKDOWN ────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 2: CARRIER-LEVEL BM COSTS & VOLUMES")
print("=" * 90)

carrier_data = costs[costs.carrier != "TOTAL"].copy()
carrier_data = carrier_data.sort_values("net_cost", ascending=False)

print(f"\n{'Carrier':<30} {'Net Cost (£M)':>14} {'Offer (£M)':>12} {'Bid (£M)':>12} {'Inc (GWh)':>10} {'Dec (GWh)':>10} {'Avg Offer':>10} {'Avg Bid':>10}")
print("-" * 130)
for _, row in carrier_data.iterrows():
    c = row['carrier']
    avg_offer = row['offer_cost'] / max(row['increase_MWh'], 0.001)
    avg_bid = row['bid_cost'] / max(row['decrease_MWh'], 0.001)
    print(f"{c:<30} {row['net_cost']/1e6:>14.2f} {row['offer_cost']/1e6:>12.2f} {row['bid_cost']/1e6:>12.2f} "
          f"{row['increase_MWh']/1e3:>10.1f} {row['decrease_MWh']/1e3:>10.1f} "
          f"£{avg_offer:>8.1f} £{avg_bid:>8.1f}")

# ─── 4. PER-CARRIER VOLUME COMPARISON WITH ELEXON ──────────────────────────────
print("\n" + "=" * 90)
print("SECTION 3: PER-CARRIER INCREASE VOLUMES — MODEL vs ELEXON BOALF")
print("=" * 90)

elexon_increases = {
    "CCGT": 7_012_012,
    "Pumped Storage Hydroelectricity": 1_959_773,
    "biomass": 209_420,
    "coal": 818_779,
    "large_hydro": 197_184,
    "nuclear": 108_068,
    "wind_offshore": 193_593,
    "wind_onshore": 885_506,
}

print(f"\n{'Carrier':<35} {'Model (GWh)':>12} {'ELEXON (GWh)':>13} {'Ratio':>8} {'Gap (GWh)':>10} {'Issue'}")
print("-" * 110)
for carrier, elexon_mwh in sorted(elexon_increases.items(), key=lambda x: -x[1]):
    model_row = carrier_data[carrier_data.carrier == carrier]
    model_mwh = model_row['increase_MWh'].values[0] if len(model_row) > 0 else 0
    ratio = model_mwh / max(elexon_mwh, 1)
    gap = (model_mwh - elexon_mwh) / 1e3
    if ratio < 0.1:
        issue = "⚠️  SEVERELY MISSING"
    elif ratio < 0.5:
        issue = "⚠️  TOO LOW"
    elif ratio > 2.0:
        issue = "⚠️  TOO HIGH"
    elif ratio > 1.3:
        issue = "⚡ High"
    else:
        issue = "✓ OK"
    print(f"{carrier:<35} {model_mwh/1e3:>12.1f} {elexon_mwh/1e3:>13.1f} {ratio:>8.2f} {gap:>+10.1f} {issue}")

# Total comparison
total_elexon_inc = 13_357_086
total_elexon_dec = 16_277_754
print(f"\n{'TOTAL INCREASES':<35} {total_row['increase_MWh']/1e3:>12.1f} {total_elexon_inc/1e3:>13.1f} "
      f"{total_row['increase_MWh']/total_elexon_inc:>8.2f}")
print(f"{'TOTAL DECREASES':<35} {total_row['decrease_MWh']/1e3:>12.1f} {total_elexon_dec/1e3:>13.1f} "
      f"{total_row['decrease_MWh']/total_elexon_dec:>8.2f}")

# ─── 5. PRICING ANALYSIS ───────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 4: BM PRICING ANALYSIS — ARE PRICES REALISTIC?")
print("=" * 90)

# Compare average effective prices per carrier
print(f"\n{'Carrier':<30} {'Model Avg Offer':>15} {'Model Avg Bid':>15} {'Elexon Avg*':>12} {'Issue'}")
print("-" * 90)

# ELEXON typical BM prices from BOALF (2020 approximate ranges)
elexon_typical = {
    "CCGT":      {"offer": (50, 80),  "bid": (-20, -5)},
    "OCGT":      {"offer": (80, 150), "bid": (-30, -10)},
    "coal":      {"offer": (40, 70),  "bid": (-20, -5)},
    "nuclear":   {"offer": (200, 999),"bid": (-150, -50)},
    "Pumped Storage Hydroelectricity": {"offer": (50, 100), "bid": (-40, -10)},
    "wind_onshore": {"offer": (0, 5), "bid": (-50, -25)},
    "wind_offshore": {"offer": (0, 5), "bid": (-80, -50)},
    "Battery":   {"offer": (50, 100), "bid": (-50, -20)},
    "large_hydro": {"offer": (40, 80), "bid": (-30, -10)},
}

for carrier in ["CCGT", "OCGT", "coal", "nuclear", "Pumped Storage Hydroelectricity",
                "wind_onshore", "wind_offshore", "Battery", "large_hydro"]:
    row = carrier_data[carrier_data.carrier == carrier]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    avg_offer = row['offer_cost'] / max(row['increase_MWh'], 0.001)
    avg_bid = row['bid_cost'] / max(row['decrease_MWh'], 0.001)
    
    typical = elexon_typical.get(carrier, {})
    typical_str = ""
    issue = ""
    if typical:
        o_lo, o_hi = typical["offer"]
        b_lo, b_hi = typical["bid"]
        typical_str = f"O:{o_lo}-{o_hi}"
        
        if avg_offer < o_lo * 0.5 and row['increase_MWh'] > 100:
            issue += "Offer TOO LOW | "
        if avg_offer > o_hi * 2 and row['increase_MWh'] > 100:
            issue += "Offer TOO HIGH | "
        if avg_bid > 0 and row['decrease_MWh'] > 100:
            issue += "⚠️ Bid POSITIVE (should be negative) | "
    
    print(f"{carrier:<30} £{avg_offer:>12.1f} £{avg_bid:>12.1f} {typical_str:>12} {issue}")

# ─── 6. NEGATIVE NET COST CARRIERS (ANOMALOUS) ─────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 5: NEGATIVE NET COST CARRIERS (ESO MAKES MONEY?)")
print("=" * 90)

negative = carrier_data[carrier_data.net_cost < -100_000].copy()
negative = negative.sort_values("net_cost")
print(f"\n{'Carrier':<30} {'Net Cost (£M)':>14} {'Bid Cost (£M)':>14} {'Dec (GWh)':>10} {'Avg Bid £/MWh':>14}")
print("-" * 90)
for _, row in negative.iterrows():
    avg_bid = row['bid_cost'] / max(row['decrease_MWh'], 0.001)
    print(f"{row['carrier']:<30} {row['net_cost']/1e6:>14.2f} {row['bid_cost']/1e6:>14.2f} "
          f"{row['decrease_MWh']/1e3:>10.1f} £{avg_bid:>12.1f}")

total_neg = negative['net_cost'].sum()
print(f"\nTotal negative net cost:     £{total_neg/1e6:.1f}M")
print(f"This REDUCES the headline BM cost. Without these, BM cost would be:")
print(f"  £{(model_cost - total_neg)/1e6:.1f}M (vs £{model_cost/1e6:.1f}M actual)")

# ─── 7. BID SIGN ANALYSIS ──────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 6: BID SIGN CHECK — POSITIVE BIDS (generator PAYS ESO to turn down)")
print("=" * 90)
print("In BM: bid = generator offers to reduce output. Bid cost to ESO should be NEGATIVE")
print("(ESO pays generator negative amount = generator pays ESO).")
print("POSITIVE bid_cost means ESO is paying generators to turn DOWN — unusual.\n")

positive_bid = carrier_data[carrier_data.bid_cost > 10_000].copy()
positive_bid = positive_bid.sort_values("bid_cost", ascending=False)
print(f"{'Carrier':<30} {'Bid Cost (£M)':>14} {'Dec (GWh)':>10} {'Avg Bid £/MWh':>14} {'Issue'}")
print("-" * 90)
for _, row in positive_bid.iterrows():
    avg_bid = row['bid_cost'] / max(row['decrease_MWh'], 0.001)
    if avg_bid > 0:
        issue = "✗ ESO paying to turn DOWN"
    else:
        issue = ""
    print(f"{row['carrier']:<30} {row['bid_cost']/1e6:>14.2f} {row['decrease_MWh']/1e3:>10.1f} £{avg_bid:>12.1f} {issue}")

# ─── 8. CONCENTRATION ANALYSIS ─────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 7: REDISPATCH CONCENTRATION — WHO DOES ALL THE WORK?")
print("=" * 90)

# Increase concentration
inc_total = total_row['increase_MWh']
print(f"\n  INCREASES (turning up):")
inc_sorted = carrier_data.sort_values('increase_MWh', ascending=False)
cumulative = 0
for _, row in inc_sorted.head(10).iterrows():
    pct = row['increase_MWh'] / inc_total * 100
    cumulative += pct
    if row['increase_MWh'] > 10:
        print(f"    {row['carrier']:<30} {row['increase_MWh']/1e3:>10.1f} GWh  ({pct:>5.1f}%)  cumulative: {cumulative:>5.1f}%")

# Decrease concentration
dec_total = total_row['decrease_MWh']
print(f"\n  DECREASES (turning down):")
dec_sorted = carrier_data.sort_values('decrease_MWh', ascending=False)
cumulative = 0
for _, row in dec_sorted.head(10).iterrows():
    pct = row['decrease_MWh'] / dec_total * 100
    cumulative += pct
    if row['decrease_MWh'] > 10:
        print(f"    {row['carrier']:<30} {row['decrease_MWh']/1e3:>10.1f} GWh  ({pct:>5.1f}%)  cumulative: {cumulative:>5.1f}%")

# ─── 9. STORAGE DEEP DIVE ──────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 8: PUMPED STORAGE & BATTERY DEEP DIVE")
print("=" * 90)

if has_storage:
    diff_su = balancing_su - wholesale_su
    print(f"\n  Storage units: {balancing_su.shape[1]}")
    
    # Per-storage type analysis
    if network is not None:
        su_info = network.storage_units[["carrier", "p_nom", "bus"]].copy()
        print(f"\n  Storage carriers: {su_info.carrier.value_counts().to_dict()}")
        
        for carrier_name in su_info.carrier.unique():
            units = su_info[su_info.carrier == carrier_name].index
            units_in_dispatch = [u for u in units if u in balancing_su.columns]
            if len(units_in_dispatch) == 0:
                continue
            
            ws_vals = wholesale_su[units_in_dispatch]
            bm_vals = balancing_su[units_in_dispatch]
            diff_vals = bm_vals - ws_vals
            
            inc = diff_vals.clip(lower=0).sum().sum()
            dec = (-diff_vals).clip(lower=0).sum().sum()
            
            ws_range = ws_vals.sum(axis=1)
            bm_range = bm_vals.sum(axis=1)
            
            print(f"\n  {carrier_name}:")
            print(f"    Units: {len(units_in_dispatch)}, Total p_nom: {su_info.loc[units_in_dispatch, 'p_nom'].sum():.0f} MW")
            print(f"    Wholesale dispatch range: [{ws_range.min():.0f}, {ws_range.max():.0f}] MW")
            print(f"    BM dispatch range:        [{bm_range.min():.0f}, {bm_range.max():.0f}] MW")
            print(f"    BM increase: {inc:,.0f} MWh, decrease: {dec:,.0f} MWh")
            print(f"    Net movement from wholesale: {(inc - dec):,.0f} MWh")
            
            # Check how much the wholesale locked them
            ws_abs_mean = ws_vals.abs().mean().mean()
            bm_abs_mean = bm_vals.abs().mean().mean()
            print(f"    Avg |dispatch| wholesale: {ws_abs_mean:.1f} MW, BM: {bm_abs_mean:.1f} MW")
            print(f"    Flexibility used: {(bm_abs_mean - ws_abs_mean) / max(ws_abs_mean, 0.1) * 100:+.1f}% change from wholesale")

# ─── 10. CONGESTION ANALYSIS ───────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 9: CONGESTION — TOP BOTTLENECKS")
print("=" * 90)

cong = congestion.sort_values("hours_congested", ascending=False)
print(f"\n  Total congested components: {len(cong)}")
print(f"\n{'Component':<40} {'Type':<12} {'s_nom (MVA)':>12} {'Hours':>8} {'% Year':>8} {'Max Load':>10}")
print("-" * 100)
for _, row in cong.head(15).iterrows():
    comp = row.get('component', row.get('Component', ''))
    comp_type = row.get('type', row.get('Type', ''))
    s_nom = row.get('s_nom', row.get('s_nom_MVA', 0))
    hours = row.get('hours_congested', row.get('Hours_Congested', 0))
    max_load = row.get('max_loading_fraction', row.get('max_loading', 0))
    pct_year = hours / 8784 * 100
    print(f"{str(comp)[:39]:<40} {str(comp_type):<12} {s_nom:>12.0f} {hours:>8.0f} {pct_year:>7.1f}% {max_load:>9.1%}")

# ─── 11. PRICE ANALYSIS ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 10: WHOLESALE & NODAL PRICE ANALYSIS")
print("=" * 90)

print(f"\n  Wholesale SMP stats:")
print(f"    Mean:   £{prices['wholesale_price'].mean():.2f}/MWh")
print(f"    Median: £{prices['wholesale_price'].median():.2f}/MWh")
print(f"    Std:    £{prices['wholesale_price'].std():.2f}/MWh")
print(f"    Min:    £{prices['wholesale_price'].min():.2f}/MWh")
print(f"    Max:    £{prices['wholesale_price'].max():.2f}/MWh")

print(f"\n  ELEXON MID reference: £33.54/MWh (N2EX)")
print(f"  Model SMP / ELEXON MID ratio: {prices['wholesale_price'].mean() / 33.54:.3f}")

print(f"\n  Nodal spread stats:")
print(f"    Mean spread:   £{prices['nodal_spread'].mean():.2f}/MWh")
print(f"    Median spread: £{prices['nodal_spread'].median():.2f}/MWh")
print(f"    Max spread:    £{prices['nodal_spread'].max():.2f}/MWh")

# Hours with extreme spreads
extreme = (prices['nodal_spread'] > 500).sum()
print(f"\n  Hours with nodal spread > £500/MWh: {extreme} ({extreme/8784*100:.1f}%)")
extreme2 = (prices['nodal_spread'] > 1000).sum()
print(f"  Hours with nodal spread > £1000/MWh: {extreme2} ({extreme2/8784*100:.1f}%)")

# ─── 12. MONTHLY BM COST PROFILE ───────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 11: MONTHLY BM COST PROFILE")
print("=" * 90)

# Compute monthly from redispatch hourly data
diff_gen = balancing_gen - wholesale_gen
inc_hourly = diff_gen.clip(lower=0).sum(axis=1)
dec_hourly = (-diff_gen).clip(lower=0).sum(axis=1)

monthly_inc = inc_hourly.resample("M").sum()
monthly_dec = dec_hourly.resample("M").sum()

print(f"\n{'Month':<12} {'Increase (GWh)':>15} {'Decrease (GWh)':>15} {'Total Redispatch':>17}")
print("-" * 65)
for i, (idx, val) in enumerate(monthly_inc.items()):
    dec_val = monthly_dec.iloc[i]
    print(f"{idx.strftime('%Y-%m'):<12} {val/1e3:>15.1f} {dec_val/1e3:>15.1f} {(val + dec_val)/1e3:>17.1f}")

# ─── 13. ASSET-LEVEL TOP REDISPATCHERS ─────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 12: TOP 20 ASSETS BY REDISPATCH COST")
print("=" * 90)

top_assets = redispatch.sort_values("net_cost", ascending=False).head(20)
print(f"\n{'Component':<45} {'Carrier':<25} {'Net Cost (£M)':>14} {'Inc (GWh)':>10} {'Dec (GWh)':>10}")
print("-" * 110)
for _, row in top_assets.iterrows():
    print(f"{str(row['component'])[:44]:<45} {str(row['carrier'])[:24]:<25} "
          f"{row['net_cost']/1e6:>14.2f} {row['increase_MWh']/1e3:>10.1f} {row['decrease_MWh']/1e3:>10.1f}")

# ─── 14. BOTTOM 20 (most negative = ESO revenue makers) ────────────────────────
print("\n" + "=" * 90)
print("SECTION 13: BOTTOM 20 ASSETS BY REDISPATCH COST (ESO earns money)")
print("=" * 90)

bottom_assets = redispatch.sort_values("net_cost", ascending=True).head(20)
print(f"\n{'Component':<45} {'Carrier':<25} {'Net Cost (£M)':>14} {'Inc (GWh)':>10} {'Dec (GWh)':>10}")
print("-" * 110)
for _, row in bottom_assets.iterrows():
    print(f"{str(row['component'])[:44]:<45} {str(row['carrier'])[:24]:<25} "
          f"{row['net_cost']/1e6:>14.2f} {row['increase_MWh']/1e3:>10.1f} {row['decrease_MWh']/1e3:>10.1f}")

# ─── 15. SUMMARY OF ISSUES ─────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 14: IDENTIFIED ISSUES SUMMARY")
print("=" * 90)

issues = []

# Issue 1: Total cost
ratio = model_cost / elexon_cost
if ratio < 0.5:
    issues.append(("CRITICAL", f"Total BM cost {ratio:.1%} of ELEXON (£{model_cost/1e6:.0f}M vs £{elexon_cost/1e6:.0f}M)"))

# Issue 2: CCGT dominance
ccgt_row = carrier_data[carrier_data.carrier == "CCGT"]
if len(ccgt_row) > 0:
    ccgt_inc_pct = ccgt_row.iloc[0]['increase_MWh'] / inc_total * 100
    if ccgt_inc_pct > 90:
        issues.append(("HIGH", f"CCGT provides {ccgt_inc_pct:.1f}% of all increases (ELEXON ~52%). Other carriers barely participate."))

# Issue 3: Missing carriers
for carrier, elexon_mwh in elexon_increases.items():
    model_row = carrier_data[carrier_data.carrier == carrier]
    model_mwh = model_row['increase_MWh'].values[0] if len(model_row) > 0 else 0
    if elexon_mwh > 100_000 and model_mwh / max(elexon_mwh, 1) < 0.05:
        issues.append(("HIGH", f"{carrier}: increase volume {model_mwh/1e3:.1f} GWh vs ELEXON {elexon_mwh/1e3:.0f} GWh ({model_mwh/max(elexon_mwh,1):.1%})"))

# Issue 4: Negative net costs
if total_neg < -10_000_000:
    issues.append(("MEDIUM", f"Carrier net costs total £{total_neg/1e6:.0f}M NEGATIVE - OCGT/wave/hydro turning down at profit to ESO"))

# Issue 5: Positive bid costs 
pos_bid_total = carrier_data[carrier_data.bid_cost > 0]['bid_cost'].sum()
if pos_bid_total > 10_000_000:
    issues.append(("MEDIUM", f"£{pos_bid_total/1e6:.0f}M in POSITIVE bid costs — ESO paying generators to turn DOWN (usually generators pay ESO)"))

# Issue 6: Price spread
if prices['nodal_spread'].max() > 5000:
    issues.append(("MEDIUM", f"Extreme nodal spread: £{prices['nodal_spread'].max():.0f}/MWh max — indicates severe transmission bottlenecks"))

# Issue 7: Pumped hydro
ps_row = carrier_data[carrier_data.carrier == "Pumped Storage Hydroelectricity"]
if len(ps_row) > 0:
    ps_inc = ps_row.iloc[0]['increase_MWh']
    if ps_inc / 1_959_773 < 0.1:
        issues.append(("HIGH", f"Pumped storage: {ps_inc/1e3:.0f} GWh increase vs ELEXON 1,960 GWh (3%) — locked by wholesale anchor?"))

# Issue 8: Wind bid cost
wind_on_row = carrier_data[carrier_data.carrier == "wind_onshore"]
if len(wind_on_row) > 0:
    wind_bid = wind_on_row.iloc[0]['bid_cost']
    wind_dec = wind_on_row.iloc[0]['decrease_MWh']
    avg_wind_bid = wind_bid / max(wind_dec, 0.001)
    if avg_wind_bid > 0:
        issues.append(("HIGH", f"Wind onshore avg bid: +£{avg_wind_bid:.1f}/MWh — ESO paying wind to curtail (should be negative or zero)"))

for severity, issue in issues:
    print(f"\n  [{severity}] {issue}")

print(f"\n\nTotal issues found: {len(issues)}")
print("=" * 90)
