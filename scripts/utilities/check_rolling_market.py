"""Quick analysis script: compare Test_Rolling_Market output against ESPENI."""
import sys
import pypsa
import pandas as pd
import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "dispatch"

# ── 1. Load network and wholesale/BM dispatch ──
n = pypsa.Network("resources/network/Test_Rolling_Market.nc")
ws = pd.read_csv(
    "resources/market/Test_Rolling_Market_wholesale_dispatch.csv",
    index_col=0, parse_dates=True,
)
bm = pd.read_csv(
    "resources/market/Test_Rolling_Market_balancing_dispatch.csv",
    index_col=0, parse_dates=True,
)

gen_carrier = n.generators["carrier"]

# ── 2. Wholesale dispatch by carrier ──
common = ws.columns.intersection(gen_carrier.index)
print(f"Matched generators: {len(common)} / {len(ws.columns)}")

cd = pd.DataFrame({"mwh": ws[common].sum(), "carrier": gen_carrier[common].values})
ws_gwh = cd.groupby("carrier")["mwh"].sum() / 1000

# ── 3. Physical (BM) dispatch by carrier ──
common_bm = bm.columns.intersection(gen_carrier.index)
bd = pd.DataFrame({"mwh": bm[common_bm].sum(), "carrier": gen_carrier[common_bm].values})
bm_gwh = bd.groupby("carrier")["mwh"].sum() / 1000

# ── 4. Combined table ──
result = pd.DataFrame({"wholesale_gwh": ws_gwh, "physical_gwh": bm_gwh}).fillna(0)
result["curtailed_gwh"] = result["wholesale_gwh"] - result["physical_gwh"]
result = result.sort_values("wholesale_gwh", ascending=False)

print("\n=== DISPATCH BY CARRIER (GWh, 48h: Jan 7-8 2020) ===")
header = "  {:30s} {:>10s} {:>10s} {:>10s}".format(
    "Carrier", "Wholesale", "Physical", "Curtailed"
)
print(header)
for c, row in result.iterrows():
    print("  {:30s} {:10.2f} {:10.2f} {:10.2f}".format(
        c, row.wholesale_gwh, row.physical_gwh, row.curtailed_gwh
    ))
totals = result.sum()
print("  {:30s} {:10.2f} {:10.2f} {:10.2f}".format(
    "TOTAL", totals.wholesale_gwh, totals.physical_gwh, totals.curtailed_gwh
))

# ── 5. Wind summary ──
wind_carriers = ["wind_offshore", "wind_onshore", "embedded_wind"]
for phase, df in [("wholesale", ws_gwh), ("physical", bm_gwh)]:
    total = sum(df.get(c, 0) for c in wind_carriers)
    print(f"\nWind total ({phase}): {total:.2f} GWh")
    for c in wind_carriers:
        print(f"  {c}: {df.get(c, 0):.2f} GWh")

ws_wind = sum(ws_gwh.get(c, 0) for c in wind_carriers)
bm_wind = sum(bm_gwh.get(c, 0) for c in wind_carriers)
curt = ws_wind - bm_wind
curt_pct = curt / ws_wind * 100 if ws_wind > 0 else 0
print(f"\nWind curtailment: {curt:.2f} GWh ({curt_pct:.1f}%)")

# ── 6. Compare against ESPENI ──
print("\n=== ESPENI COMPARISON (Jan 7-8, 2020) ===")
try:
    espeni = pd.read_csv(
        "data/demand/espeni-2020.csv", parse_dates=["SETTLEMENT_DATE"]
    )
    mask = espeni["SETTLEMENT_DATE"].dt.month == 1
    mask &= espeni["SETTLEMENT_DATE"].dt.day.isin([7, 8])
    esp = espeni[mask].copy()

    wind_cols = [c for c in esp.columns if "WIND" in c.upper()]
    solar_cols = [c for c in esp.columns if "SOLAR" in c.upper()]
    demand_cols = [c for c in esp.columns if "INDO" in c.upper() or "ND" == c.upper()]

    for col in wind_cols:
        val = esp[col].sum() / 2 / 1e6  # half-hourly MW -> GWh
        print(f"  {col}: {val:.2f} GWh")
    for col in solar_cols:
        val = esp[col].sum() / 2 / 1e6
        print(f"  {col}: {val:.2f} GWh")
    for col in demand_cols[:2]:
        val = esp[col].sum() / 2 / 1e6
        print(f"  {col}: {val:.2f} GWh")
except Exception as e:
    print(f"  Error: {e}")

# ── 7. Wholesale price ──
try:
    prices = pd.read_csv(
        "resources/market/Test_Rolling_Market_wholesale_price.csv",
        index_col=0, parse_dates=True,
    )
    if prices.shape[1] == 1:
        p = prices.iloc[:, 0]
    else:
        p = prices.mean(axis=1)
    print(f"\n=== WHOLESALE PRICE ===")
    print(f"  Mean: {p.mean():.2f} GBP/MWh")
    print(f"  Min:  {p.min():.2f} GBP/MWh")
    print(f"  Max:  {p.max():.2f} GBP/MWh")
except Exception as e:
    print(f"  Price error: {e}")

# ── 8. Capacity factors (with perf factors baked in) ──
print("\n=== CAPACITY FACTORS (p_max_pu, after perf factors) ===")
for carrier in ["wind_offshore", "wind_onshore", "solar_pv"]:
    gens = n.generators[n.generators.carrier == carrier].index
    if len(gens) == 0:
        continue
    ts_gens = [g for g in gens if g in n.generators_t.p_max_pu.columns]
    if ts_gens:
        snap_mask = (n.snapshots >= "2020-01-07") & (n.snapshots <= "2020-01-08 23:00")
        if snap_mask.any():
            cf = n.generators_t.p_max_pu.loc[snap_mask, ts_gens]
        else:
            cf = n.generators_t.p_max_pu[ts_gens]
        p_nom = n.generators.loc[ts_gens, "p_nom"]
        weighted_cf = (cf * p_nom).sum(axis=1) / p_nom.sum()
        print("  {}: mean CF = {:.3f} (min={:.3f}, max={:.3f})".format(
            carrier, weighted_cf.mean(), weighted_cf.min(), weighted_cf.max()
        ))
        print("    {} generators, {:.0f} MW total".format(len(ts_gens), p_nom.sum()))

# ── 9. Load shedding check ──
ls_gens = n.generators[n.generators.carrier == "load_shedding"].index
ls_ws = ws.reindex(columns=ls_gens.intersection(ws.columns)).sum().sum()
ls_bm = bm.reindex(columns=ls_gens.intersection(bm.columns)).sum().sum()
print(f"\n=== LOAD SHEDDING ===")
print(f"  Wholesale: {ls_ws:.1f} MWh")
print(f"  Physical:  {ls_bm:.1f} MWh")
