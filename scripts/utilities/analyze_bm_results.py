"""
Analyze PyPSA-GB Balancing Mechanism (BM) results and compare to actual ELEXON data.
Scenario: Test_Rolling_Market (48-hour period: 2020-01-07 to 2020-01-08)
"""

import pandas as pd
import numpy as np
import json
import os

BASE = r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1"
MARKET = os.path.join(BASE, "resources", "market")
ELEXON_VAL = os.path.join(BASE, "resources", "market", "elexon", "validation", "2020")
SCENARIO = "Test_Rolling_Market"

print("=" * 80)
print("  PyPSA-GB BALANCING MECHANISM ANALYSIS")
print(f"  Scenario: {SCENARIO}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODEL BM RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  1. MODEL BM RESULTS (constraint_costs.csv)")
print("=" * 80)

costs = pd.read_csv(os.path.join(MARKET, f"{SCENARIO}_constraint_costs.csv"))
total_row = costs[costs["carrier"] == "TOTAL"].iloc[0]
carriers = costs[costs["carrier"] != "TOTAL"].copy()

total_offer = total_row["offer_cost"]
total_bid = total_row["bid_cost"]
total_net = total_row["net_cost"]
total_inc = total_row["increase_MWh"]
total_dec = total_row["decrease_MWh"]

print(f"\n  Solve period: 48 hours (2020-01-07 00:00 to 2020-01-08 23:00)")
print(f"\n  Total BM cost (net):     GBP {total_net:>14,.0f}")
print(f"    - Offer costs (turn-up):  GBP {total_offer:>14,.0f}")
print(f"    - Bid costs (turn-down):  GBP {total_bid:>14,.0f}")
print(f"\n  Total increase volume:   {total_inc:>12,.0f} MWh")
print(f"  Total decrease volume:   {total_dec:>12,.0f} MWh")

# Annualize (48h -> 8760h)
annualisation_factor = 8760 / 48
annualised_cost = total_net * annualisation_factor
print(f"\n  Annualised BM cost:      GBP {annualised_cost:>14,.0f}")
print(f"  (annualisation factor: {annualisation_factor:.1f}x)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CARRIER BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  2. BM COST BREAKDOWN BY CARRIER")
print("=" * 80)

carriers_sorted = carriers.sort_values("net_cost", ascending=False)
print(f"\n  {'Carrier':<35} {'Net Cost':>14} {'% Total':>8} {'Inc MWh':>12} {'Dec MWh':>12}")
print(f"  {'-'*35} {'-'*14} {'-'*8} {'-'*12} {'-'*12}")

for _, row in carriers_sorted.iterrows():
    pct = 100.0 * row["net_cost"] / total_net if total_net > 0 else 0
    if row["net_cost"] > 100:  # Only show significant carriers
        print(f"  {row['carrier']:<35} GBP {row['net_cost']:>10,.0f} {pct:>7.1f}% {row['increase_MWh']:>11,.0f} {row['decrease_MWh']:>11,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. KEY METRICS: CURTAILMENT AND TURN-UP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  3. KEY REDISPATCH METRICS")
print("=" * 80)

def get_carrier(name):
    row = carriers[carriers["carrier"] == name]
    if len(row) == 0:
        return {"increase_MWh": 0, "decrease_MWh": 0, "net_cost": 0}
    return row.iloc[0]

nuclear = get_carrier("nuclear")
wind_off = get_carrier("wind_offshore")
wind_on = get_carrier("wind_onshore")
ccgt = get_carrier("CCGT")
coal = get_carrier("coal")
biomass = get_carrier("biomass")

print(f"\n  CURTAILMENT (turn-down):")
print(f"    Nuclear:        {nuclear['decrease_MWh']:>10,.0f} MWh  (cost: GBP {nuclear['net_cost']:>12,.0f})")
print(f"    Wind offshore:  {wind_off['decrease_MWh']:>10,.0f} MWh  (cost: GBP {wind_off['net_cost']:>12,.0f})")
print(f"    Wind onshore:   {wind_on['decrease_MWh']:>10,.0f} MWh  (cost: GBP {wind_on['net_cost']:>12,.0f})")
print(f"    Total wind:     {wind_off['decrease_MWh'] + wind_on['decrease_MWh']:>10,.0f} MWh")

print(f"\n  TURN-UP (offers):")
print(f"    CCGT:           {ccgt['increase_MWh']:>10,.0f} MWh  (cost: GBP {ccgt['net_cost']:>12,.0f})")
print(f"    Coal:           {coal['increase_MWh']:>10,.0f} MWh  (cost: GBP {coal['net_cost']:>12,.0f})")
print(f"    Wind onshore:   {wind_on['increase_MWh']:>10,.0f} MWh")

# Cost shares
print(f"\n  COST SHARES:")
top4 = [("Nuclear", nuclear), ("CCGT", ccgt), ("Wind offshore", wind_off), ("Wind onshore", wind_on)]
for name, c in top4:
    pct = 100.0 * c["net_cost"] / total_net
    print(f"    {name:<20} {pct:>6.1f}%  (GBP {c['net_cost']:>12,.0f})")

top4_total = sum(c["net_cost"] for _, c in top4)
print(f"    {'Top 4 total':<20} {100.0*top4_total/total_net:>6.1f}%  (GBP {top4_total:>12,.0f})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD ACTUAL ELEXON DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  4. ACTUAL ELEXON BM DATA (BOALF + System Prices)")
print("=" * 80)

# System prices
sys_prices = pd.read_csv(os.path.join(ELEXON_VAL, "system_prices.csv"),
                          parse_dates=["datetime"], index_col="datetime")
# Filter to our 48-hour period
mask = (sys_prices.index >= "2020-01-07") & (sys_prices.index <= "2020-01-08 23:59:59")
sp = sys_prices.loc[mask]

print(f"\n  System prices (2020-01-07 to 2020-01-08):")
print(f"    Mean SBP (system buy price):  GBP {sp['system_buy_price'].mean():>8.2f}/MWh")
print(f"    Mean SSP (system sell price): GBP {sp['system_sell_price'].mean():>8.2f}/MWh")
print(f"    SBP range: GBP {sp['system_buy_price'].min():.2f} to GBP {sp['system_buy_price'].max():.2f}/MWh")
print(f"    Hours: {len(sp)}")

# BOALF data
boalf = pd.read_csv(os.path.join(ELEXON_VAL, "boalf_data.csv"))
# Filter to our period - BOALF settlement_date covers 2020-01-07 and 2020-01-08
boalf_period = boalf[boalf["settlement_date"].isin(["2020-01-07", "2020-01-08"])]
print(f"\n  BOALF acceptances in period: {len(boalf_period)} records")
print(f"  Unique BMUs: {boalf_period['bmu_id'].nunique() if len(boalf_period) > 0 else 0}")

# B1610 actual generation
b1610 = pd.read_csv(os.path.join(ELEXON_VAL, "b1610_actual.csv"),
                      parse_dates=["datetime"], index_col="datetime")
mask_b = (b1610.index >= "2020-01-07") & (b1610.index <= "2020-01-08 23:59:59")
b1610_period = b1610.loc[mask_b]

print(f"\n  B1610 actual generation data:")
print(f"    Rows: {len(b1610_period)}, Columns (BMUs): {len(b1610_period.columns)}")
if len(b1610_period) > 0:
    total_gen = b1610_period.sum(axis=1).mean()
    print(f"    Mean total generation: {total_gen:.0f} MW")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LOAD BM VALIDATION FILE (pre-computed comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  5. MODEL vs ACTUAL COMPARISON (bm_validation.csv)")
print("=" * 80)

validation = pd.read_csv(os.path.join(MARKET, f"{SCENARIO}_bm_validation.csv"))

print(f"\n  {'Metric':<50} {'Model':>18} {'Actual/Benchmark':>18} {'Ratio':>8}")
print(f"  {'-'*50} {'-'*18} {'-'*18} {'-'*8}")

for _, row in validation.iterrows():
    metric = row["metric"]
    model = str(row["model_value"])
    elexon = str(row["elexon_value"])
    ratio = str(row["ratio"])
    if "BM cost" in metric or "volume" in metric or "Annualised" in metric or "wholesale" in metric or "nodal" in metric:
        print(f"  {metric:<50} {model:>18} {elexon:>18} {ratio:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON TO PREVIOUS RUN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  6. COMPARISON TO PREVIOUS RUN (before demand allocation fix)")
print("=" * 80)

# Previous run values (from user's report)
prev_model_cost = 5_800_000  # ~5.8M (annualized? or 48h?)
prev_actual_cost = 1_060_000  # ~1.06M
prev_ratio = 5.47

# The validation file shows annualized comparison
# Current annualized: ~6.32B vs 1.4B benchmark = 4.51x
current_ratio = 4.51  # from validation file

# But let's also compute 48h actual cost from system prices
# NESO benchmark £1.4B/year -> £1.4B * 48/8760 = ~7.67M for 48h
neso_48h = 1_400_000_000 * 48 / 8760

print(f"\n  NESO 2020 benchmark (annualised):   GBP 1,400,000,000")
print(f"  NESO 2020 benchmark (48h equiv):    GBP {neso_48h:>12,.0f}")
print(f"\n  Current model (48h):                GBP {total_net:>12,.0f}")
print(f"  Current model (annualised):         GBP {annualised_cost:>12,.0f}")
print(f"\n  Current ratio (annualised):         {annualised_cost/1_400_000_000:.2f}x")
print(f"  Previous ratio (user-reported):     {prev_ratio:.2f}x")

if annualised_cost/1_400_000_000 < prev_ratio:
    improvement = (1 - (annualised_cost/1_400_000_000) / prev_ratio) * 100
    print(f"\n  Improvement:                        {improvement:.1f}% reduction in ratio")
else:
    worsening = ((annualised_cost/1_400_000_000) / prev_ratio - 1) * 100
    print(f"\n  WORSENING:                          {worsening:.1f}% increase in ratio")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONGESTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  7. NETWORK CONGESTION")
print("=" * 80)

congestion = pd.read_csv(os.path.join(MARKET, f"{SCENARIO}_congestion.csv"))
print(f"\n  Congested lines/transformers: {len(congestion)}")
print(f"\n  Most congested (hours_congested):")
cong_sorted = congestion.sort_values("hours_congested", ascending=False).head(10)
print(f"  {'Component':<30} {'Type':<8} {'s_nom MVA':>10} {'Hours':>6} {'Mean Loading':>13}")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*6} {'-'*13}")
for _, row in cong_sorted.iterrows():
    print(f"  {row['component']:<30} {row['type']:<8} {row['s_nom_MVA']:>10.0f} {row['hours_congested']:>6} {row['mean_loading_fraction']:>12.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. B1610 DISPATCH COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  8. B1610 DISPATCH COMPARISON (from validation file)")
print("=" * 80)

b1610_rows = validation[validation["metric"].str.startswith("B1610 generation")]
if len(b1610_rows) > 0:
    print(f"\n  {'Carrier':<40} {'Model':>12} {'Actual':>12} {'Ratio':>8}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8}")
    for _, row in b1610_rows.iterrows():
        carrier = row["metric"].replace("B1610 generation: ", "")
        print(f"  {carrier:<40} {str(row['model_value']):>12} {str(row['elexon_value']):>12} {str(row['ratio']):>8}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. DETAILED DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  9. DIAGNOSIS: WHY IS BM COST STILL HIGH?")
print("=" * 80)

print(f"""
  The model BM cost is GBP {total_net:,.0f} for 48 hours, which annualises to
  GBP {annualised_cost:,.0f} — about {annualised_cost/1_400_000_000:.1f}x the NESO benchmark.

  KEY OBSERVATIONS:

  a) Nuclear curtailment dominates bid costs:
     - {nuclear['decrease_MWh']:,.0f} MWh curtailed (cost: GBP {nuclear['net_cost']:,.0f})
     - This is {100*nuclear['net_cost']/total_net:.1f}% of total BM cost
     - Nuclear bid price is GBP {nuclear['bid_cost']/nuclear['decrease_MWh'] if nuclear['decrease_MWh'] > 0 else 0:.0f}/MWh

  b) CCGT turn-up is by far the largest offer action:
     - {ccgt['increase_MWh']:,.0f} MWh turned up (cost: GBP {ccgt['net_cost']:,.0f})
     - CCGT also turned DOWN {ccgt['decrease_MWh']:,.0f} MWh
     - Net CCGT cost is {100*ccgt['net_cost']/total_net:.1f}% of total BM cost

  c) Wind curtailment is massive:
     - Offshore: {wind_off['decrease_MWh']:,.0f} MWh curtailed
     - Onshore:  {wind_on['decrease_MWh']:,.0f} MWh curtailed
     - Combined wind cost: GBP {wind_off['net_cost']+wind_on['net_cost']:,.0f} ({100*(wind_off['net_cost']+wind_on['net_cost'])/total_net:.1f}%)

  d) The bid side (GBP {total_bid:,.0f}) is 2.4x the offer side (GBP {total_offer:,.0f})
     - This suggests excess generation in Scotland/north is being turned down
     - And insufficient generation in south is being turned up
     - Classic B6 boundary congestion pattern

  e) 26 congested lines, with {len(congestion[congestion['hours_congested']==48])} lines congested for ALL 48 hours
     - Persistent congestion drives high redispatch volumes

  f) Comparison to previous run:
     - Previous annualised ratio: ~5.47x benchmark
     - Current annualised ratio:  ~{annualised_cost/1_400_000_000:.2f}x benchmark
     - The demand allocation fix has {'IMPROVED' if annualised_cost/1_400_000_000 < prev_ratio else 'NOT IMPROVED'} the BM cost ratio
""")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PER-CARRIER INCREASE VOLUME COMPARISON (Model vs ELEXON BOALF)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  10. INCREASE VOLUME: MODEL vs ELEXON BOALF")
print("=" * 80)

inc_rows = validation[validation["metric"].str.startswith("Increase volume")]
if len(inc_rows) > 0:
    print(f"\n  {'Carrier':<40} {'Model MWh':>12} {'ELEXON MWh':>12} {'Ratio':>8}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8}")
    for _, row in inc_rows.iterrows():
        carrier = row["metric"].replace("Increase volume: ", "")
        print(f"  {carrier:<40} {str(row['model_value']):>12} {str(row['elexon_value']):>12} {str(row['ratio']):>8}")

print(f"\n  Model total increase:  {total_inc:>12,.0f} MWh")
print(f"  ELEXON total increase: ~83,564 MWh (from validation)")
print(f"  Ratio:                 {total_inc/83564:.2f}x")

print("\n" + "=" * 80)
print("  ANALYSIS COMPLETE")
print("=" * 80)
