"""BM deep-dive: volumes, costs, effective prices, spatial patterns."""
import pandas as pd
import numpy as np

rd = pd.read_csv("resources/market/Validation_2020_redispatch_summary.csv")
cst = pd.read_csv("resources/market/Validation_2020_constraint_costs.csv")
boalf = pd.read_csv("resources/market/Validation_2020_boalf_by_flag.csv")

# ── 1. Carrier summary with effective prices ────────────────────────────────
carrier = rd.groupby("carrier").agg({
    "increase_MWh": "sum",
    "decrease_MWh": "sum",
    "offer_cost": "sum",
    "bid_cost": "sum",
    "net_cost": "sum"
}).sort_values("net_cost", ascending=False)

carrier["inc_TWh"] = carrier["increase_MWh"] / 1e6
carrier["dec_TWh"] = carrier["decrease_MWh"] / 1e6
carrier["eff_offer"] = carrier["offer_cost"] / carrier["increase_MWh"].clip(lower=1)
carrier["eff_bid"] = carrier["bid_cost"] / carrier["decrease_MWh"].clip(lower=1)

print("=" * 120)
print("  1. CARRIER BM SUMMARY — effective bid/offer prices")
print("=" * 120)
hdr = f"{'Carrier':30s}  {'inc TWh':>8s}  {'dec TWh':>8s}  {'eff offer':>12s}  {'eff bid':>12s}  {'net cost':>12s}"
print(hdr)
print("-" * 120)
for idx, r in carrier.iterrows():
    print(f"{idx:30s}  {r['inc_TWh']:8.3f}  {r['dec_TWh']:8.3f}  "
          f"£{r['eff_offer']:8.1f}/MWh  £{r['eff_bid']:8.1f}/MWh  "
          f"£{r['net_cost']/1e6:9.1f}m")

# ── 2. Top 15 INCREASE generators ───────────────────────────────────────────
print("\n" + "=" * 120)
print("  2. TOP 15 INCREASE generators (turned UP in BM to resolve constraints)")
print("=" * 120)
top_inc = rd.nlargest(15, "increase_MWh")
print(f"{'Generator':50s} {'Carrier':20s}  {'inc GWh':>9s}  {'offer £/MWh':>12s}  {'offer £m':>10s}")
print("-" * 110)
for _, r in top_inc.iterrows():
    eff = r["offer_cost"] / max(r["increase_MWh"], 1)
    print(f"{r['component']:50s} {r['carrier']:20s}  {r['increase_MWh']/1e3:9.0f}  "
          f"£{eff:9.1f}/MWh  £{r['offer_cost']/1e6:8.1f}m")

# ── 3. Top 15 DECREASE generators ───────────────────────────────────────────
print("\n" + "=" * 120)
print("  3. TOP 15 DECREASE generators (turned DOWN in BM — curtailed/constrained off)")
print("=" * 120)
top_dec = rd.nlargest(15, "decrease_MWh")
print(f"{'Generator':50s} {'Carrier':20s}  {'dec GWh':>9s}  {'bid £/MWh':>12s}  {'bid £m':>10s}")
print("-" * 110)
for _, r in top_dec.iterrows():
    eff = r["bid_cost"] / max(r["decrease_MWh"], 1)
    print(f"{r['component']:50s} {r['carrier']:20s}  {r['decrease_MWh']/1e3:9.0f}  "
          f"£{eff:9.1f}/MWh  £{r['bid_cost']/1e6:8.1f}m")

# ── 4. Carrier-vs-BOALF comparison (unflagged only = constraint) ────────────
print("\n" + "=" * 120)
print("  4. MODEL vs ELEXON BOALF (unflagged = constraint management)")
print("     Note: unflagged BOALF is the best comparator — flagged includes reserve/response")
print("=" * 120)

boalf_all = boalf[(boalf.scope == "carrier") & (boalf.group == "all")]
boalf_uf = boalf[(boalf.scope == "carrier") & (boalf.group == "unflagged")]
boalf_flag = boalf[(boalf.scope == "carrier") & (boalf.group == "flagged_any")]

# BOALF carrier map to model carriers
boalf_map = {
    "CCGT": "CCGT",
    "Pumped Storage Hydroelectricity": "Pumped Storage Hydroelectricity",
    "biomass": "biomass",
    "coal": "coal",
    "large_hydro": "large_hydro",
    "nuclear": "nuclear",
    "wind_offshore": "wind_offshore",
    "wind_onshore": "wind_onshore",
}

print(f"\n{'Carrier':30s}  {'Model inc':>10s}  {'BOALF UF inc':>12s}  {'Ratio':>7s}  │  "
      f"{'Model dec':>10s}  {'BOALF UF dec':>12s}  {'Ratio':>7s}  │  "
      f"{'BOALF flag inc':>14s}  {'BOALF flag dec':>14s}")
print("-" * 160)

for b_carrier, m_carrier in boalf_map.items():
    m_row = cst[cst.carrier == m_carrier]
    b_uf_row = boalf_uf[boalf_uf.carrier == b_carrier]
    b_fl_row = boalf_flag[boalf_flag.carrier == b_carrier]
    
    m_inc = m_row["increase_MWh"].values[0] / 1e3 if len(m_row) else 0
    m_dec = m_row["decrease_MWh"].values[0] / 1e3 if len(m_row) else 0
    uf_inc = b_uf_row["increase_mwh"].values[0] / 1e3 if len(b_uf_row) else 0
    uf_dec = b_uf_row["decrease_mwh"].values[0] / 1e3 if len(b_uf_row) else 0
    fl_inc = b_fl_row["increase_mwh"].values[0] / 1e3 if len(b_fl_row) else 0
    fl_dec = b_fl_row["decrease_mwh"].values[0] / 1e3 if len(b_fl_row) else 0
    
    r_inc = f"{m_inc/uf_inc:.2f}x" if uf_inc > 0 else "—"
    r_dec = f"{m_dec/uf_dec:.2f}x" if uf_dec > 0 else "—"
    
    print(f"{b_carrier:30s}  {m_inc:8.0f} GWh  {uf_inc:10.0f} GWh  {r_inc:>7s}  │  "
          f"{m_dec:8.0f} GWh  {uf_dec:10.0f} GWh  {r_dec:>7s}  │  "
          f"{fl_inc:12.0f} GWh  {fl_dec:12.0f} GWh")

# ── 5. Generator-count analysis ─────────────────────────────────────────────
print("\n" + "=" * 120)
print("  5. HOW MANY GENERATORS PARTICIPATE IN BM?")
print("=" * 120)

n_inc = (rd["increase_MWh"] > 1).sum()
n_dec = (rd["decrease_MWh"] > 1).sum()
n_both = ((rd["increase_MWh"] > 1) & (rd["decrease_MWh"] > 1)).sum()
print(f"  Generators with increase > 1 MWh: {n_inc}")
print(f"  Generators with decrease > 1 MWh: {n_dec}")
print(f"  Generators doing both: {n_both}")
print(f"  Total generators in redispatch: {len(rd)}")

# Participation by carrier
print(f"\n  Carrier participation (generators with >1 MWh action):")
for c in rd["carrier"].unique():
    sub = rd[rd.carrier == c]
    ni = (sub["increase_MWh"] > 1).sum()
    nd = (sub["decrease_MWh"] > 1).sum()
    if ni + nd > 0:
        print(f"    {c:30s}  inc: {ni:5d}  dec: {nd:5d}")

# ── 6. Congestion lines ─────────────────────────────────────────────────────
print("\n" + "=" * 120)
print("  6. MOST CONGESTED LINES (>1000 hours at limit)")
print("=" * 120)

cong = pd.read_csv("resources/market/Validation_2020_congestion.csv")
cong_severe = cong[cong.hours_congested > 1000].sort_values("hours_congested", ascending=False)
print(f"{'Line':35s}  {'s_nom MVA':>10s}  {'Hours':>8s}  {'Mean load':>10s}")
print("-" * 70)
for _, r in cong_severe.iterrows():
    print(f"{r['component']:35s}  {r['s_nom_MVA']:10.0f}  {r['hours_congested']:8d}  "
          f"{r['mean_loading_fraction']:10.1%}")

# ── 7. Anomalous carriers ───────────────────────────────────────────────────
print("\n" + "=" * 120)
print("  7. ANOMALOUS CARRIERS — negative net BM costs")
print("=" * 120)

neg_carriers = cst[(cst.net_cost < 0) & (cst.carrier != "TOTAL")]
for _, r in neg_carriers.iterrows():
    c = r["carrier"]
    print(f"\n  {c}:")
    print(f"    Increase: {r['increase_MWh']:.0f} MWh, decrease: {r['decrease_MWh']:.0f} MWh")
    if r['decrease_MWh'] > 0:
        eff_bid = r['bid_cost'] / r['decrease_MWh']
        print(f"    Effective bid price: £{eff_bid:.1f}/MWh (this is what ESO pays per MWh to curtail)")
    print(f"    Net BM cost: £{r['net_cost']/1e6:.1f}m (negative = saves money vs wholesale)")
    
    # Check if carrier exists in reality in 2020
    if c in ("shoreline_wave", "tidal_stream"):
        print(f"    ⚠ This carrier had near-zero real capacity in 2020!")
        print(f"      → Model dispatches it wholesale (cheap MC), then BM curtails it")
        print(f"      → The BM pays the bid price to turn it down, but because the bid price")
        print(f"        is lower than what was saved by dispatching it wholesale, net BM cost is negative")

# ── 8. Balance check ────────────────────────────────────────────────────────
print("\n" + "=" * 120)
print("  8. ENERGY BALANCE CHECK")
print("=" * 120)

total_row = cst[cst.carrier == "TOTAL"].iloc[0]
print(f"  Total BM increase: {total_row['increase_MWh']/1e6:.3f} TWh")
print(f"  Total BM decrease: {total_row['decrease_MWh']/1e6:.3f} TWh")
print(f"  Difference: {(total_row['increase_MWh'] - total_row['decrease_MWh'])/1e3:.1f} GWh")
print(f"  → Should be ~0 (increase = decrease for energy balance)")
print(f"  Total BM GROSS cost (offer + |bid|): £{(total_row['offer_cost'] + abs(total_row['bid_cost']))/1e6:.0f}m")
print(f"  Total BM NET cost: £{total_row['net_cost']/1e6:.1f}m")

# ── 9. Cost decomposition ───────────────────────────────────────────────────
print("\n" + "=" * 120)
print("  9. BM COST WATERFALL (what makes up the £155m)")
print("=" * 120)

carriers_sorted = cst[cst.carrier != "TOTAL"].sort_values("net_cost", ascending=False)
cumulative = 0
print(f"{'Carrier':30s}  {'Net cost':>12s}  {'Cumulative':>12s}  {'% of total':>10s}  Direction")
print("-" * 100)
for _, r in carriers_sorted.iterrows():
    cumulative += r["net_cost"]
    pct = r["net_cost"] / total_row["net_cost"] * 100
    direction = ""
    if r["increase_MWh"] > 10 * r["decrease_MWh"]:
        direction = "← almost all INCREASE (turn-up)"
    elif r["decrease_MWh"] > 10 * r["increase_MWh"]:
        direction = "← almost all DECREASE (turn-down)"
    elif r["increase_MWh"] > 0.1 and r["decrease_MWh"] > 0.1:
        direction = "← both directions"
    print(f"{r['carrier']:30s}  £{r['net_cost']/1e6:9.1f}m  £{cumulative/1e6:9.1f}m  {pct:9.1f}%  {direction}")

print(f"\n  {'TOTAL':30s}  £{total_row['net_cost']/1e6:9.1f}m")
print(f"\n  Note: gross cost = £{(total_row['offer_cost'] + abs(total_row['bid_cost']))/1e6:.0f}m "
      f"→ large offsets from negative-cost carriers (shoreline_wave, large_hydro, tidal_stream, OCGT)")
print(f"  Without the four negative-cost carriers, net would be £{(total_row['net_cost'] + 27676096 + 3869641 + 2563840 + 2121356)/1e6:.0f}m")

print("\nDone.")
