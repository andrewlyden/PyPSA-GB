"""Investigate solar mismatch: model total solar vs ESPENI embedded solar."""
import pypsa
import pandas as pd
import numpy as np

# ── 1. Load network and wholesale dispatch ──
n = pypsa.Network("resources/network/Test_Rolling_Market.nc")
ws = pd.read_csv(
    "resources/market/Test_Rolling_Market_wholesale_dispatch.csv",
    index_col=0, parse_dates=True,
)

gen_carrier = n.generators["carrier"]

# Get model solar dispatch per timestep
solar_pv_gens = gen_carrier[gen_carrier == "solar_pv"].index.intersection(ws.columns)
emb_solar_gens = gen_carrier[gen_carrier == "embedded_solar"].index.intersection(ws.columns)

model_solar_pv = ws[solar_pv_gens].sum(axis=1)
model_emb_solar = ws[emb_solar_gens].sum(axis=1)
model_total_solar = model_solar_pv + model_emb_solar

# ── 2. Load ESPENI for Jan 7-8 ──
esp = pd.read_csv("data/demand/espeni.csv")
ts_col = "ELEC_elex_startTime[utc](datetime)"
esp[ts_col] = pd.to_datetime(esp[ts_col], utc=True)
mask = (esp[ts_col] >= "2020-01-07") & (esp[ts_col] < "2020-01-09")
sub = esp[mask].copy().set_index(ts_col)

solar_col = "ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"
espeni_solar_hh = sub[solar_col]
espeni_solar_h = espeni_solar_hh.resample("h").mean()
espeni_solar_h.index = espeni_solar_h.index.tz_localize(None)

# Align
common_idx = model_solar_pv.index.intersection(espeni_solar_h.index)
print("Common timesteps: {}".format(len(common_idx)))
print()

# ── 3. Hourly comparison (daylight hours only) ──
print("Hour-by-hour comparison (MW) — daylight hours only:")
print("{:>16s}  {:>10s}  {:>10s}  {:>11s}  {:>8s}  {:>6s}".format(
    "Hour", "REPD_solar", "emb_solar", "model_total", "ESPENI", "ratio"
))
for t in common_idx:
    spv = model_solar_pv.loc[t]
    emb = model_emb_solar.loc[t]
    total = model_total_solar.loc[t]
    esp_val = espeni_solar_h.loc[t]
    if total < 1 and esp_val < 1:
        continue  # skip night hours
    ratio = total / esp_val if esp_val > 1 else float("nan")
    print("{:>16s}  {:10.1f}  {:10.1f}  {:11.1f}  {:8.1f}  {:6.2f}".format(
        str(t), spv, emb, total, esp_val, ratio
    ))

# ── 4. Totals ──
total_spv = model_solar_pv.loc[common_idx].sum() / 1000
total_emb = model_emb_solar.loc[common_idx].sum() / 1000
total_model = total_spv + total_emb
total_esp = espeni_solar_h.loc[common_idx].sum() / 1000
print()
print("=== TOTALS (GWh, 48h) ===")
print("  REPD solar_pv:    {:.3f} GWh".format(total_spv))
print("  Embedded solar:   {:.3f} GWh".format(total_emb))
print("  Model total:      {:.3f} GWh".format(total_model))
print("  ESPENI total:     {:.3f} GWh".format(total_esp))
print("  Model/ESPENI:     {:.2f}".format(total_model / total_esp if total_esp > 0 else float("nan")))

# ── 5. Check p_max_pu profiles for embedded solar ──
print()
print("=== EMBEDDED SOLAR p_max_pu PROFILE (Jan 7-8) ===")
snap_mask = (n.snapshots >= "2020-01-07") & (n.snapshots <= "2020-01-08 23:00")
emb_ts = [g for g in emb_solar_gens if g in n.generators_t.p_max_pu.columns]
if emb_ts:
    ppu = n.generators_t.p_max_pu.loc[snap_mask, emb_ts]
    p_nom_emb = n.generators.loc[emb_ts, "p_nom"]
    weighted_cf = (ppu * p_nom_emb).sum(axis=1) / p_nom_emb.sum()
    # Only show daylight hours
    for t, cf in weighted_cf.items():
        if cf > 0.001:
            mw = cf * p_nom_emb.sum()
            print("  {}: CF={:.4f}, MW={:.1f}".format(t, cf, mw))
    print("  Peak CF: {:.4f}  ({:.1f} MW)".format(
        weighted_cf.max(), weighted_cf.max() * p_nom_emb.sum()
    ))

# ── 6. Check REPD solar p_max_pu profiles ──
print()
print("=== REPD SOLAR p_max_pu PROFILE (Jan 7-8) ===")
spv_ts = [g for g in solar_pv_gens if g in n.generators_t.p_max_pu.columns]
if spv_ts:
    ppu = n.generators_t.p_max_pu.loc[snap_mask, spv_ts]
    p_nom_spv = n.generators.loc[spv_ts, "p_nom"]
    weighted_cf = (ppu * p_nom_spv).sum(axis=1) / p_nom_spv.sum()
    for t, cf in weighted_cf.items():
        if cf > 0.001:
            mw = cf * p_nom_spv.sum()
            print("  {}: CF={:.4f}, MW={:.1f}".format(t, cf, mw))
    print("  Peak CF: {:.4f}  ({:.1f} MW)".format(
        weighted_cf.max(), weighted_cf.max() * p_nom_spv.sum()
    ))

# ── 7. Check the overlap subtraction results ──
print()
print("=== OVERLAP DIAGNOSIS ===")
# Recompute what the integration script did:
# gap = ESPENI - model_solar_pv, clipped to >=0
# Re-derive using full-year profiles if possible
full_snap = (n.snapshots >= "2020-01-07") & (n.snapshots <= "2020-01-08 23:00")
if spv_ts:
    model_avail = (n.generators_t.p_max_pu.loc[full_snap, spv_ts] *
                   n.generators.loc[spv_ts, "p_nom"].values).sum(axis=1)
    espeni_aligned = espeni_solar_h.reindex(model_avail.index, method="nearest").fillna(0)
    gap = (espeni_aligned - model_avail).clip(lower=0)
    excess = (model_avail - espeni_aligned).clip(lower=0)

    for t in model_avail.index:
        g = gap.loc[t]
        e = excess.loc[t]
        m = model_avail.loc[t]
        es = espeni_aligned.loc[t]
        if m > 1 or es > 1:
            print("  {}: model_avail={:.0f} MW, ESPENI={:.0f} MW, gap={:.0f} MW, excess={:.0f} MW".format(
                t, m, es, g, e
            ))

    print()
    print("  Total model solar available: {:.2f} GWh".format(model_avail.sum() / 1000))
    print("  Total ESPENI solar: {:.2f} GWh".format(espeni_aligned.sum() / 1000))
    print("  Total gap (ESPENI > model): {:.2f} GWh".format(gap.sum() / 1000))
    print("  Total excess (model > ESPENI): {:.2f} GWh".format(excess.sum() / 1000))
    print("  Hours where model > ESPENI: {}".format((model_avail > espeni_aligned + 1).sum()))
    print("  Hours where ESPENI > model: {}".format((espeni_aligned > model_avail + 1).sum()))
