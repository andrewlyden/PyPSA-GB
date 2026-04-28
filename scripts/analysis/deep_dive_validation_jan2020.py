"""
Deep-dive comparison: Validation_Jan2020 model results vs ESPENI + ELEXON data.

Run from project root:
    python scripts/analysis/deep_dive_validation_jan2020.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pypsa

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
summary_dir = project_root / "resources" / "analysis"

SEP = "=" * 80
SEP2 = "-" * 80

# ── helpers ────────────────────────────────────────────────────────────────────

def agg_by_carrier(df: pd.DataFrame, carrier_map: dict) -> pd.DataFrame:
    result = {}
    for gen in df.columns:
        c = carrier_map.get(gen, "unknown")
        if c not in result:
            result[c] = pd.Series(0.0, index=df.index)
        result[c] = result[c] + df[gen].fillna(0)
    return pd.DataFrame(result)


def hourly_mwh(series_hh: pd.Series) -> float:
    """Half-hourly MW series → total MWh (sum * 0.5h)."""
    return float(series_hh.sum() * 0.5)


def corr(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 10:
        return float("nan")
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def show_row(label, model_val, ref_val, unit="MWh", extra=""):
    ratio = model_val / ref_val if ref_val and ref_val != 0 else float("nan")
    ratio_str = f"{ratio:6.3f}" if not np.isnan(ratio) else "  n/a"
    print(f"  {label:<40}  {model_val:>14,.0f}  {ref_val:>14,.0f}  {ratio_str}  {unit}  {extra}")


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading model networks...")
n_ws = pypsa.Network("resources/market/Validation_Jan2020_wholesale.nc")
n_bm = pypsa.Network("resources/market/Validation_Jan2020_balancing.nc")

gen_carrier  = n_ws.generators["carrier"]
su_carrier   = n_ws.storage_units["carrier"]

print("Loading ESPENI...")
TIME_COL = "ELEC_elex_startTime[utc](datetime)"
espeni_raw = pd.read_csv("data/demand/espeni.csv")
espeni_raw["time"] = pd.to_datetime(espeni_raw[TIME_COL], utc=True).dt.tz_convert(None)
espeni_raw = espeni_raw.set_index("time").drop(columns=[c for c in espeni_raw.columns if "datetime" in c.lower() or "time" in c.lower()], errors="ignore")
# Keep only numeric columns
espeni_raw = espeni_raw.select_dtypes(include="number")
espeni_raw = espeni_raw.sort_index()
esp = espeni_raw[(espeni_raw.index >= "2020-01-01") & (espeni_raw.index < "2020-02-01")]
esp_h = esp.resample("h").mean()   # hourly averages (values in MW)

print("Loading ELEXON data...")
mid = pd.read_csv("resources/market/elexon/mid_prices_2020.csv", index_col=0, parse_dates=True)
mid_h = mid.resample("h").mean()
mid_jan = mid_h[(mid_h.index >= "2020-01-01") & (mid_h.index < "2020-02-01")]

elexon_offers = pd.read_csv(
    "resources/market/Validation_Jan2020/elexon/elexon_offers.csv",
    index_col=0, parse_dates=True
)
elexon_bids = pd.read_csv(
    "resources/market/Validation_Jan2020/elexon/elexon_bids.csv",
    index_col=0, parse_dates=True
)

print("Loading model results CSVs...")
ws_price  = pd.read_csv("resources/market/Validation_Jan2020_wholesale_price.csv",
                        index_col=0, parse_dates=True)
bm_d      = pd.read_csv("resources/market/Validation_Jan2020_balancing_dispatch.csv",
                        index_col=0, parse_dates=True)
ws_d      = pd.read_csv("resources/market/Validation_Jan2020_wholesale_dispatch.csv",
                        index_col=0, parse_dates=True)
bm_s      = pd.read_csv("resources/market/Validation_Jan2020_balancing_storage.csv",
                        index_col=0, parse_dates=True)
ws_links  = pd.read_csv("resources/market/Validation_Jan2020_wholesale_links.csv",
                        index_col=0, parse_dates=True)
pc        = pd.read_csv("resources/market/Validation_Jan2020_price_comparison.csv",
                        index_col=0, parse_dates=True)
cong      = pd.read_csv("resources/market/Validation_Jan2020_congestion.csv", index_col=0)
rdsp      = pd.read_csv("resources/market/Validation_Jan2020_redispatch_summary.csv",
                        index_col=0)
cc        = pd.read_csv("resources/market/Validation_Jan2020_constraint_costs.csv",
                        index_col=0)
cc_no_total = cc[cc.index != "TOTAL"].copy()

print("Aggregating dispatch by carrier...")
bm_by_c = agg_by_carrier(bm_d, gen_carrier.to_dict())
ws_by_c = agg_by_carrier(ws_d, gen_carrier.to_dict())

# Storage
ps_cols = [c for c in bm_s.columns
           if any(x in c for x in ["Dinorwig", "Ffestiniog", "Cruachan", "Foyers"])]
batt_cols = [c for c in bm_s.columns if c not in ps_cols]

ps_gen  = bm_s[ps_cols].clip(lower=0).sum(axis=1)   # +ve = generating
ps_chg  = (-bm_s[ps_cols].clip(upper=0)).sum(axis=1) # -ve = charging

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: WHOLESALE PRICE vs MID
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 1: WHOLESALE PRICE vs ELEXON MID (N2EX)")
print(SEP)

sns = ws_price.index
mid_aligned = mid_jan["mid_price"].reindex(sns, method="nearest")
model_smp = ws_price["wholesale_price"]

print(f"\n  {'Metric':<40}  {'Model':>12}  {'MID/ELEXON':>12}  {'Ratio':>7}")
print(f"  {SEP2}")
show_row("Mean SMP/MID (£/MWh)",
         model_smp.mean(), mid_aligned.mean(), unit="£/MWh")
show_row("Median SMP/MID (£/MWh)",
         model_smp.median(), mid_aligned.median(), unit="£/MWh")
show_row("Std SMP/MID (£/MWh)",
         model_smp.std(), mid_aligned.std(), unit="£/MWh")
show_row("Min SMP/MID (£/MWh)",
         model_smp.min(), mid_aligned.min(), unit="£/MWh")
show_row("Max SMP/MID (£/MWh)",
         model_smp.max(), mid_aligned.max(), unit="£/MWh")

r_price = corr(model_smp, mid_aligned)
print(f"\n  Hourly correlation (r):  {r_price:.4f}")
print(f"  R² (price):              {r_price**2:.4f}")

# Hourly absolute error
both_price = pd.concat([model_smp, mid_aligned], axis=1).dropna()
both_price.columns = ["model", "mid"]
mae = float((both_price["model"] - both_price["mid"]).abs().mean())
rmse = float(((both_price["model"] - both_price["mid"])**2).mean()**0.5)
print(f"  MAE (£/MWh):             {mae:.2f}")
print(f"  RMSE (£/MWh):            {rmse:.2f}")

# Price by hour of day
print(f"\n  === Price profile by hour of day (model vs MID) ===")
print(f"  {'Hour':>5}  {'Model £/MWh':>12}  {'MID £/MWh':>12}  {'Delta':>8}")
for h in range(24):
    mask = model_smp.index.hour == h
    m_h = model_smp[mask].mean()
    e_h = mid_aligned[mask].mean()
    print(f"  {h:>5}  {m_h:>12.2f}  {e_h:>12.2f}  {m_h-e_h:>+8.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CARRIER DISPATCH vs ESPENI
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 2: CARRIER DISPATCH (BM final) vs ESPENI")
print(SEP)
print(f"\n  {'Carrier':<32}  {'Model MWh':>12}  {'ESPENI MWh':>12}  {'Ratio':>7}  {'r':>6}")
print(f"  {SEP2}")

# ESPENI column mapping (half-hourly MW → *0.5 for MWh)
ESPENI_MAP = {
    "CCGT":          "ELEC_POWER_ELEX_CCGT[MW](float32)",
    "nuclear":       "ELEC_POWER_ELEX_NUCLEAR[MW](float32)",
    "coal":          "ELEC_POWER_ELEX_COAL[MW](float32)",
    "biomass":       "ELEC_POWER_ELEX_BIOMASS_POSTCALC[MW](float32)",
    "OCGT":          "ELEC_POWER_ELEX_OCGT[MW](float32)",
    "large_hydro":   "ELEC_POWER_ELEX_NPSHYD[MW](float32)",
    "embedded_wind": "ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)",
    "embedded_solar":"ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)",
}

for carrier, ecol in ESPENI_MAP.items():
    if carrier not in bm_by_c.columns:
        continue
    model_t = float(bm_by_c[carrier].sum())
    esp_t = hourly_mwh(esp[ecol].dropna()) if ecol in esp.columns else float("nan")
    r_val = corr(bm_by_c[carrier], esp_h[ecol]) if ecol in esp_h.columns else float("nan")
    ratio = model_t / esp_t if esp_t and not np.isnan(esp_t) else float("nan")
    r_str = f"{r_val:6.3f}" if not np.isnan(r_val) else "  n/a"
    ratio_str = f"{ratio:7.3f}" if not np.isnan(ratio) else "    n/a"
    print(f"  {carrier:<32}  {model_t:>12,.0f}  {esp_t:>12,.0f}  {ratio_str}  {r_str}")

# Wind (ELEX only covers metered, not embedded)
w_onshore  = float(bm_by_c.get("wind_onshore", pd.Series([0])).sum())
w_offshore = float(bm_by_c.get("wind_offshore", pd.Series([0])).sum())
w_embedded = float(bm_by_c.get("embedded_wind", pd.Series([0])).sum())
w_model_excl_emb = w_onshore + w_offshore
w_model_total = w_model_excl_emb + w_embedded

elex_wind_mwh = hourly_mwh(esp["ELEC_POWER_ELEX_WIND[MW](float32)"])
ngem_wind_mwh = float(esp_h["ELEC_POWER_TOTAL_WIND[MW](float32)"].sum())

# Hourly correlation (wind_onshore+offshore vs ELEX_WIND)
wind_model_h = bm_by_c.get("wind_onshore", pd.Series(0, index=bm_by_c.index)) + \
               bm_by_c.get("wind_offshore", pd.Series(0, index=bm_by_c.index))
wind_esp_h  = esp_h["ELEC_POWER_ELEX_WIND[MW](float32)"]
r_wind = corr(wind_model_h, wind_esp_h)

r_str = f"{r_wind:6.3f}"
ratio_str = f"{w_model_excl_emb/elex_wind_mwh:7.3f}"
print(f"  {'wind_on+offshore (vs ELEX_WIND)':<32}  {w_model_excl_emb:>12,.0f}  {elex_wind_mwh:>12,.0f}  {ratio_str}  {r_str}")

ratio_tot = w_model_total / ngem_wind_mwh if ngem_wind_mwh else float("nan")
print(f"  {'wind_total incl embedded (NGEM)':<32}  {w_model_total:>12,.0f}  {ngem_wind_mwh:>12,.0f}  {ratio_tot:7.3f}")

# Solar
solar_pv  = float(bm_by_c.get("solar_pv", pd.Series([0])).sum())
emb_solar = float(bm_by_c.get("embedded_solar", pd.Series([0])).sum())
solar_total_model = solar_pv + emb_solar
solar_esp = float(esp_h["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"].sum())
print(f"  {'solar_pv+embedded_solar':<32}  {solar_total_model:>12,.0f}  {solar_esp:>12,.0f}  {solar_total_model/solar_esp:7.3f}")

# Pumped hydro: compare ELEX_PS_DISCHARGING vs model PS generation
ps_gen_mwh   = float(ps_gen.sum())
ps_elex_dis  = hourly_mwh(esp["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"])
ps_elex_chg  = abs(hourly_mwh(esp["ELEC_POWER_ELEX_PS_CHARGING[MW](float32)"]))
ps_chg_mwh   = float(ps_chg.sum())
r_ps = corr(ps_gen, esp_h["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"])
r_str = f"{r_ps:6.3f}" if not np.isnan(r_ps) else "  n/a"
ratio_str = f"{ps_gen_mwh/ps_elex_dis:7.3f}" if ps_elex_dis > 0 else "    n/a"
print(f"  {'pumped_hydro dispatch (gen side)':<32}  {ps_gen_mwh:>12,.0f}  {ps_elex_dis:>12,.0f}  {ratio_str}  {r_str}")
print(f"  {'pumped_hydro charging':<32}  {ps_chg_mwh:>12,.0f}  {ps_elex_chg:>12,.0f}  {ps_chg_mwh/ps_elex_chg:7.3f}")

# Total demand
demand_model = float(n_bm.loads_t.p_set.sum(axis=1).sum()) if len(n_bm.loads_t.p_set) > 0 else 0
esp_demand   = float(esp_h["ELEC_POWER_TOTAL_ESPENI[MW](float32)"].sum())
print()
print(f"  {'TOTAL DEMAND (ESPENI)':<32}  {demand_model:>12,.0f}  {esp_demand:>12,.0f}  {demand_model/esp_demand:7.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: HOURLY PROFILE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 3: HOURLY PROFILE CORRELATION SUMMARY")
print(SEP)
print(f"\n  {'Carrier':<32}  {'r_ws':>8}  {'r_bm':>8}  {'MAE_ws (MW)':>12}  {'MAE_bm (MW)':>12}")
print(f"  {SEP2}")

for carrier, ecol in ESPENI_MAP.items():
    if ecol not in esp_h.columns:
        continue
    e_series = esp_h[ecol]
    for label, df in [("ws", ws_by_c), ("bm", bm_by_c)]:
        if carrier in df.columns:
            pass
    ws_c = ws_by_c.get(carrier, pd.Series(0, index=ws_by_c.index)) if carrier in ws_by_c.columns else pd.Series(0, index=ws_by_c.index)
    bm_c = bm_by_c.get(carrier, pd.Series(0, index=bm_by_c.index)) if carrier in bm_by_c.columns else pd.Series(0, index=bm_by_c.index)
    r_ws = corr(ws_c, e_series)
    r_bm = corr(bm_c, e_series)
    mae_ws = float((ws_c - e_series.reindex(ws_c.index)).abs().mean())
    mae_bm = float((bm_c - e_series.reindex(bm_c.index)).abs().mean())
    r_ws_s = f"{r_ws:8.4f}" if not np.isnan(r_ws) else "     n/a"
    r_bm_s = f"{r_bm:8.4f}" if not np.isnan(r_bm) else "     n/a"
    print(f"  {carrier:<32}  {r_ws_s}  {r_bm_s}  {mae_ws:>12.1f}  {mae_bm:>12.1f}")

# Wind profile correlation
r_wind_ws = corr(
    ws_by_c.get("wind_onshore", pd.Series(0)) + ws_by_c.get("wind_offshore", pd.Series(0)),
    wind_esp_h
)
r_wind_bm = corr(wind_model_h, wind_esp_h)
print(f"  {'wind_on+offshore':<32}  {r_wind_ws:8.4f}  {r_wind_bm:8.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BM VOLUMES vs ELEXON BOD
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 4: BALANCING MECHANISM VOLUMES vs ELEXON BOD")
print(SEP)

# Aggregate elexon offers by summing positive acceptance volumes
# Elexon offers are recorded as prices; the volume columns are prices per BMU per hour
# For volume comparison we use the bm_validation csv which already has Elexon benchmarks
bm_val = pd.read_csv("resources/market/Validation_Jan2020_bm_validation.csv",
                     index_col=0, skipinitialspace=True)
bm_val.columns = bm_val.columns.str.strip()
bm_val.index = bm_val.index.str.strip()

print(f"\n  {'Metric':<45}  {'Model':>14}  {'Elexon':>14}  {'Ratio':>7}")
print(f"  {SEP2}")

def try_get(df, idx, col):
    try:
        return float(str(df.loc[idx, col]).strip().replace("£", "").replace(",", "").replace("—", "nan"))
    except:
        return float("nan")

rows_to_show = [
    ("Annualised BM cost",          "Annualised BM cost", "£/year"),
    ("Total increase volume",       "Total increase volume", "MWh"),
    ("Total decrease volume",       "Total decrease volume", "MWh"),
    ("Mean wholesale price (SMP)",  "Mean wholesale price (SMP)", "£/MWh"),
    ("Increase volume: CCGT",       "Increase volume: CCGT", "MWh"),
    ("Increase volume: Pumped Storage Hydroelectricity", "Increase volume: Pumped Storage Hydroelectricity", "MWh"),
    ("Increase volume: biomass",    "Increase volume: biomass", "MWh"),
    ("Increase volume: coal",       "Increase volume: coal", "MWh"),
    ("Increase volume: large_hydro","Increase volume: large_hydro", "MWh"),
    ("Increase volume: nuclear",    "Increase volume: nuclear", "MWh"),
    ("Increase volume: wind_onshore","Increase volume: wind_onshore", "MWh"),
    ("Increase volume: wind_offshore","Increase volume: wind_offshore", "MWh"),
]

for label, idx, unit in rows_to_show:
    m_val = try_get(bm_val, idx, "model_value")
    e_val = try_get(bm_val, idx, "elexon_value")
    r_val = try_get(bm_val, idx, "ratio")
    m_str = f"{m_val:>14,.0f}" if not np.isnan(m_val) else f"{'—':>14}"
    e_str = f"{e_val:>14,.0f}" if not np.isnan(e_val) else f"{'—':>14}"
    r_str = f"{r_val:>7.3f}" if not np.isnan(r_val) else f"{'—':>7}"
    print(f"  {label:<45}  {m_str}  {e_str}  {r_str}  {unit}")

# PS BM detail
print()
print(f"  === Pumped Hydro BM Detail ===")
ps_bm_inc = float(rdsp[rdsp["carrier"].str.contains("Pumped Storage", case=False, na=False)]["increase_MWh"].sum()) \
    if "carrier" in rdsp.columns else 0.0
ps_bm_dec = float(rdsp[rdsp["carrier"].str.contains("Pumped Storage", case=False, na=False)]["decrease_MWh"].sum()) \
    if "carrier" in rdsp.columns else 0.0
print(f"  Model PS BM increase (dispatch up):  {ps_bm_inc:>10,.0f} MWh")
print(f"  Model PS BM decrease (dispatch dn):  {ps_bm_dec:>10,.0f} MWh")
print(f"  Elexon PS BM increase:               {139436:>10,} MWh (reference)")
print(f"  PS BM offer cost:  {cc.loc['Pumped Storage Hydroelectricity','offer_cost'] if 'Pumped Storage Hydroelectricity' in cc.index else 0:.0f} £")
print(f"  PS wholesale gen:  {ps_gen_mwh:>10,.0f} MWh  (ELEX: {ps_elex_dis:,.0f} MWh)")
print(f"  PS wholesale chg:  {ps_chg_mwh:>10,.0f} MWh  (ELEX: {ps_elex_chg:,.0f} MWh)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: INTERCONNECTORS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 5: INTERCONNECTOR FLOWS vs ESPENI")
print(SEP)

ESPENI_IC = {
    "IFA": {
        "esp_col": "ELEC_POWER_ELEX_INTFR[MW](float32)",
        "links": ["IC_IFA"],
    },
    "Britned": {
        "esp_col": "ELEC_POWER_ELEX_INTNED[MW](float32)",
        "links": ["IC_Britned"],
    },
    "Nemo Link": {
        "esp_col": "ELEC_POWER_ELEX_INTNEM[MW](float32)",
        "links": ["IC_Nemo Link"],
    },
    "Moyle + Auchencrosh": {
        "esp_col": "ELEC_POWER_ELEX_INTIRL[MW](float32)",
        "links": ["IC_Moyle", "IC_Auchencrosh (interconnector CCT)"],
    },
    "East West": {
        "esp_col": "ELEC_POWER_ELEX_INTEW[MW](float32)",
        "links": ["IC_East West Interconnector"],
    },
}

ic_links = [l for l in n_ws.links.index if l.startswith("IC_")]
print(f"\n  {'IC':<35}  {'Model net import':>18}  {'ESPENI net import':>18}  {'Ratio':>7}  {'r':>6}")
print(f"  {SEP2}")

for short, cfg in ESPENI_IC.items():
    ecol = cfg["esp_col"]
    link_names = [lk for lk in cfg["links"] if lk in n_ws.links_t.p_set.columns]
    if link_names:
        model_h_ic = -n_ws.links_t.p_set[link_names].sum(axis=1)
        model_import = float(model_h_ic.sum())
    else:
        model_h_ic = None
        model_import = float("nan")

    if ecol in esp.columns:
        e_import = hourly_mwh(esp[ecol].dropna())
        r_ic = corr(model_h_ic, esp_h[ecol]) if model_h_ic is not None else float("nan")
    else:
        e_import = float("nan")
        r_ic = float("nan")

    ratio = model_import / e_import if e_import and not np.isnan(e_import) else float("nan")
    r_str = f"{r_ic:6.3f}" if not np.isnan(r_ic) else "  n/a"
    ratio_str = f"{ratio:7.3f}" if not np.isnan(ratio) else "    n/a"
    print(f"  {short:<35}  {model_import:>18,.0f}  {e_import:>18,.0f}  {ratio_str}  {r_str}")

# Totals
total_model_import = sum(
    float(-n_ws.links_t.p_set[lk].sum())
    if lk in n_ws.links_t.p_set.columns else 0
    for lk in ic_links
)
total_espeni_import = float(esp_h["ELEC_POWER_ELEX_NET_IMPORTS[MW](float32)"].sum())
print(f"  {'TOTAL NET IMPORT':<35}  {total_model_import:>18,.0f}  {total_espeni_import:>18,.0f}  "
      f"{total_model_import/total_espeni_import if total_espeni_import else 0:7.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CONSTRAINT COSTS BY CARRIER
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 6: CONSTRAINT COSTS BY CARRIER (BM final)")
print(SEP)
total_cost = float(cc.loc["TOTAL", "net_cost"]) if "TOTAL" in cc.index else float(cc_no_total["net_cost"].sum())
print(f"\n  {'Carrier':<30}  {'Offer cost £':>14}  {'Bid cost £':>14}  {'Net cost £':>14}  {'% total':>8}")
print(f"  {SEP2}")
for carrier, row in cc_no_total.sort_values("net_cost", ascending=False).iterrows():
    pct = row["net_cost"] / total_cost * 100 if total_cost else 0
    print(f"  {str(carrier):<30}  {row['offer_cost']:>14,.0f}  {row['bid_cost']:>14,.0f}  "
          f"{row['net_cost']:>14,.0f}  {pct:>7.1f}%")
print(f"  {'TOTAL':<30}  {'':>14}  {'':>14}  {total_cost:>14,.0f}  {'100.0%':>8}")
print(f"\n  Annualised: £{total_cost * 8760 / 744:,.0f}/year")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CONGESTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 7: NETWORK CONGESTION")
print(SEP)
print(f"\n  Total congested lines/transformers: {len(cong)}")
print(f"\n  {'Component':<30}  {'Type':>12}  {'s_nom MVA':>10}  {'Hours':>8}  {'Max load':>10}  {'Mean load':>10}")
print(f"  {SEP2}")
for comp, row in cong.sort_values("hours_congested", ascending=False).head(20).iterrows():
    print(f"  {comp:<30}  {row['type']:>12}  {row['s_nom_MVA']:>10.0f}  "
          f"{row['hours_congested']:>8.0f}  {row['max_loading_fraction']:>10.3f}  "
          f"{row['mean_loading_fraction']:>10.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: NODAL PRICE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 8: NODAL PRICE SPREAD (Network Constraint Signal)")
print(SEP)

# pc has hourly wholesale, mean/min/max nodal
pc_jan = pc[(pc.index >= "2020-01-01") & (pc.index < "2020-02-01")]
print(f"\n  Mean wholesale (SMP):       £{pc_jan['wholesale_price'].mean():>8.2f}/MWh")
print(f"  Mean nodal price (demand buses): £{pc_jan['mean_nodal_price'].mean():>8.2f}/MWh")
print(f"  Mean nodal spread (max-min):  £{pc_jan['nodal_spread'].mean():>8.2f}/MWh")
print(f"  Max nodal spread (any hour):  £{pc_jan['nodal_spread'].max():>8.2f}/MWh")
print(f"  % hours with spread > £50:    {(pc_jan['nodal_spread'] > 50).mean()*100:>8.1f}%")
print(f"  % hours with any neg price:   {(pc_jan['min_nodal_price'] < 0).mean()*100:>8.1f}%")
print(f"  % hours spread < £5 (copperplate-like): {(pc_jan['nodal_spread'] < 5).mean()*100:>8.1f}%")

# Spread distribution
bins = [0, 5, 20, 50, 100, 200, 500, 1e6]
labels = ["0-5", "5-20", "20-50", "50-100", "100-200", "200-500", ">500"]
spread = pc_jan["nodal_spread"]
print(f"\n  Nodal spread distribution (£/MWh):")
for i, (lo, hi) in enumerate(zip(bins, bins[1:])):
    n = ((spread >= lo) & (spread < hi)).sum()
    pct = n / len(spread) * 100
    print(f"    {labels[i]:>8}:  {n:>4} hours  ({pct:>5.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: WIND PERFORMANCE FACTOR DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 9: WIND PERFORMANCE DIAGNOSTIC")
print(SEP)

wind_model_ws_total = float(ws_by_c.get("wind_onshore", pd.Series([0])).sum() +
                            ws_by_c.get("wind_offshore", pd.Series([0])).sum())
wind_model_bm_total = float(bm_by_c.get("wind_onshore", pd.Series([0])).sum() +
                            bm_by_c.get("wind_offshore", pd.Series([0])).sum())
wind_espeni_elex = elex_wind_mwh

print(f"\n  Wind wholesale (model):      {wind_model_ws_total:>14,.0f} MWh")
print(f"  Wind BM final (model):       {wind_model_bm_total:>14,.0f} MWh")
print(f"  Wind dispatch (ELEX_WIND):   {wind_espeni_elex:>14,.0f} MWh")
print(f"  Wind curtailment (wholesale->BM): {wind_model_ws_total - wind_model_bm_total:>14,.0f} MWh")
print(f"  Wind curtailment %:          {(wind_model_ws_total - wind_model_bm_total)/wind_model_ws_total*100:>14.1f}%")
print(f"\n  Wholesale / ESPENI ratio:  {wind_model_ws_total/wind_espeni_elex:.3f}")
print(f"  BM final / ESPENI ratio:   {wind_model_bm_total/wind_espeni_elex:.3f}")
print(f"\n  Onshore only:")
ws_on = float(ws_by_c.get("wind_onshore", pd.Series([0])).sum())
bm_on = float(bm_by_c.get("wind_onshore", pd.Series([0])).sum())
ws_off = float(ws_by_c.get("wind_offshore", pd.Series([0])).sum())
bm_off = float(bm_by_c.get("wind_offshore", pd.Series([0])).sum())
print(f"    Onshore wholesale {ws_on:>14,.0f} MWh  →  BM {bm_on:>14,.0f} MWh  curtailment {ws_on-bm_on:,.0f}")
print(f"    Offshore wholesale {ws_off:>13,.0f} MWh  →  BM {bm_off:>14,.0f} MWh  curtailment {ws_off-bm_off:,.0f}")
print(f"\n  Performance factors currently applied (from defaults.yaml):")
print(f"    wind_onshore:  0.80")
print(f"    wind_offshore: 0.82")
print(f"  Required factor to reach ESPENI total (from 0-factor baseline ~5.5 TWh offshore, ~5.4 TWh onshore):")
print(f"    Combined factor needed: {wind_espeni_elex / (wind_model_ws_total / (0.80 if wind_model_ws_total > 0 else 1)):.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: KEY FINDINGS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 10: KEY FINDINGS SUMMARY vs VALIDATION TARGETS")
print(SEP)

targets = {
    "Wind total (BM/ESPENI)":        (wind_model_bm_total / wind_espeni_elex,          0.95, 1.05),
    "CCGT total (BM/ESPENI)":        (float(bm_by_c["CCGT"].sum()) / hourly_mwh(esp["ELEC_POWER_ELEX_CCGT[MW](float32)"]) if "CCGT" in bm_by_c.columns else 0,  0.95, 1.05),
    "Coal total (BM/ESPENI)":        (float(bm_by_c["coal"].sum()) / hourly_mwh(esp["ELEC_POWER_ELEX_COAL[MW](float32)"]) if "coal" in bm_by_c.columns else 0,    0.92, 1.08),
    "Hydro total (BM/ESPENI)":       (float(bm_by_c.get("large_hydro", pd.Series([0])).sum()) / hourly_mwh(esp["ELEC_POWER_ELEX_NPSHYD[MW](float32)"]),  0.90, 1.10),
    "IC total (model/ESPENI)":       (total_model_import / total_espeni_import if total_espeni_import else 0, 0.90, 1.10),
    "SMP (model/MID)":               (model_smp.mean() / mid_aligned.mean(), 0.95, 1.05),
    "CCGT hourly r":                 (corr(bm_by_c.get("CCGT", pd.Series(0)), esp_h["ELEC_POWER_ELEX_CCGT[MW](float32)"]), 0.99, 1.00),
    "Coal hourly r":                 (corr(bm_by_c.get("coal", pd.Series(0)), esp_h["ELEC_POWER_ELEX_COAL[MW](float32)"]), 0.85, 1.00),
    "Nuclear hourly r":              (corr(bm_by_c.get("nuclear", pd.Series(0)), esp_h["ELEC_POWER_ELEX_NUCLEAR[MW](float32)"]), 0.80, 1.00),
    "BM ann. cost (model/1.4B)":     (total_cost * 8760 / 744 / 1.4e9, 0.86, 1.07),
    "BM CCGT increase (model/ref)":  (float(cc.loc["CCGT", "increase_MWh"]) / 595556 if "CCGT" in cc.index else 0, 0.0, 1.30),
    "BM PS increase (model/139GWh)": (ps_bm_inc / 139436 if len(rdsp) > 0 else 0, 0.50, 2.00),
}

print(f"\n  {'Metric':<40}  {'Value':>8}  {'Target range':>18}  {'Status':>8}")
print(f"  {SEP2}")
summary_rows = []
for metric, (val, lo, hi) in targets.items():
    if np.isnan(val):
        status = "  ???"
        val_str = "  n/a"
    elif lo <= val <= hi:
        status = "  PASS"
    elif abs(val - np.mean([lo, hi])) < 0.15 * np.mean([lo, hi]):
        status = "  NEAR"
    else:
        status = "  FAIL"
    val_str = f"{val:8.3f}" if not np.isnan(val) else "     n/a"
    print(f"  {metric:<40}  {val_str}  {lo:.2f} – {hi:.2f}       {status}")
    summary_rows.append({
        "metric": metric,
        "value": None if np.isnan(val) else float(val),
        "target_low": lo,
        "target_high": hi,
        "status": status.strip(),
    })

summary_dir.mkdir(parents=True, exist_ok=True)
summary_csv = summary_dir / "Validation_Jan2020_deep_dive_summary.csv"
summary_md = summary_dir / "Validation_Jan2020_deep_dive_summary.md"

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_csv, index=False)

headline_rows = [
    ("Mean SMP / MID", model_smp.mean() / mid_aligned.mean(), "0.95-1.05"),
    ("Wind BM / ELEX_WIND", wind_model_bm_total / wind_espeni_elex, "0.95-1.05"),
    ("IC net import / ESPENI", total_model_import / total_espeni_import, "0.90-1.10"),
    ("BM annualised cost / 1.4bn", total_cost * 8760 / 744 / 1.4e9, "0.86-1.07"),
    ("PS BM increase / Elexon", ps_bm_inc / 139436 if 139436 else float("nan"), ">0.50"),
]

md_lines = [
    "# Validation_Jan2020 Deep-Dive Summary",
    "",
    "Generated by scripts/analysis/deep_dive_validation_jan2020.py",
    "",
    "## Headline Metrics",
    "",
    "| Metric | Value | Target |",
    "|--------|------:|--------|",
]
for metric, value, target in headline_rows:
    value_str = "n/a" if np.isnan(value) else f"{value:.3f}"
    md_lines.append(f"| {metric} | {value_str} | {target} |")

md_lines.extend([
    "",
    "## Key Dispatch Comparisons",
    "",
    "| Metric | Model | Reference | Ratio |",
    "|--------|------:|----------:|------:|",
    f"| CCGT BM final (MWh) | {float(bm_by_c['CCGT'].sum()):,.0f} | {hourly_mwh(esp['ELEC_POWER_ELEX_CCGT[MW](float32)']):,.0f} | {float(bm_by_c['CCGT'].sum()) / hourly_mwh(esp['ELEC_POWER_ELEX_CCGT[MW](float32)']):.3f} |",
    f"| Wind BM final excl embedded (MWh) | {wind_model_bm_total:,.0f} | {wind_espeni_elex:,.0f} | {wind_model_bm_total / wind_espeni_elex:.3f} |",
    f"| Pumped hydro generation (MWh) | {ps_gen_mwh:,.0f} | {ps_elex_dis:,.0f} | {ps_gen_mwh / ps_elex_dis:.3f} |",
    f"| Total net imports (MWh) | {total_model_import:,.0f} | {total_espeni_import:,.0f} | {total_model_import / total_espeni_import:.3f} |",
    "",
    "## Target Check",
    "",
    "| Metric | Value | Target Low | Target High | Status |",
    "|--------|------:|-----------:|------------:|--------|",
])
for row in summary_rows:
    value_str = "n/a" if row["value"] is None else f"{row['value']:.3f}"
    md_lines.append(
        f"| {row['metric']} | {value_str} | {row['target_low']:.2f} | {row['target_high']:.2f} | {row['status']} |"
    )

summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print()
print(f"Summary CSV written to: {summary_csv}")
print(f"Summary Markdown written to: {summary_md}")

print()
print(SEP)
print("END OF DEEP-DIVE REPORT")
print(SEP)
