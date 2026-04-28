"""
Deep-dive comparison: Validation_2020 (full year) model results vs ESPENI + ELEXON data.

Run from project root:
    python scripts/analysis/deep_dive_validation_2020.py
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
SCENARIO = "Validation_2020"
YEAR = 2020
HOURS_IN_YEAR = 8784  # 2020 is a leap year

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


def try_get_numeric(df, idx, col):
    try:
        return float(str(df.loc[idx, col]).strip().replace("£", "").replace(",", "").replace("—", "nan").replace("%", ""))
    except Exception:
        return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading model networks...")
n_ws = pypsa.Network(f"resources/market/{SCENARIO}_wholesale.nc")
n_bm = pypsa.Network(f"resources/market/{SCENARIO}_balancing.nc")

gen_carrier = n_ws.generators["carrier"]
su_carrier = n_ws.storage_units["carrier"]

print(f"  Snapshots: {len(n_ws.snapshots)}")
print(f"  Generators: {len(n_ws.generators)}")
print(f"  Storage units: {len(n_ws.storage_units)}")
print(f"  Lines: {len(n_ws.lines)}")
print(f"  Links (interconnectors): {len(n_ws.links)}")

print("\nLoading ESPENI (full year 2020)...")
TIME_COL = "ELEC_elex_startTime[utc](datetime)"
espeni_raw = pd.read_csv("data/demand/espeni.csv")
espeni_raw["time"] = pd.to_datetime(espeni_raw[TIME_COL], utc=True).dt.tz_convert(None)
espeni_raw = espeni_raw.set_index("time").drop(
    columns=[c for c in espeni_raw.columns if "datetime" in c.lower() or "time" in c.lower()],
    errors="ignore"
)
espeni_raw = espeni_raw.select_dtypes(include="number").sort_index()
esp = espeni_raw[(espeni_raw.index >= "2020-01-01") & (espeni_raw.index < "2021-01-01")]
esp_h = esp.resample("h").mean()

print("Loading ELEXON price data...")
mid = pd.read_csv("resources/market/elexon/mid_prices_2020.csv", index_col=0, parse_dates=True)
mid_h = mid.resample("h").mean()
mid_2020 = mid_h[(mid_h.index >= "2020-01-01") & (mid_h.index < "2021-01-01")]

print("Loading model results CSVs...")
ws_price = pd.read_csv(f"resources/market/{SCENARIO}_wholesale_price.csv",
                       index_col=0, parse_dates=True)
bm_d = pd.read_csv(f"resources/market/{SCENARIO}_balancing_dispatch.csv",
                    index_col=0, parse_dates=True)
bm_s = pd.read_csv(f"resources/market/{SCENARIO}_balancing_storage.csv",
                    index_col=0, parse_dates=True)
ws_links = pd.read_csv(f"resources/market/{SCENARIO}_wholesale_links.csv",
                       index_col=0, parse_dates=True)
pc = pd.read_csv(f"resources/market/{SCENARIO}_price_comparison.csv",
                 index_col=0, parse_dates=True)
cong = pd.read_csv(f"resources/market/{SCENARIO}_congestion.csv", index_col=0)
rdsp = pd.read_csv(f"resources/market/{SCENARIO}_redispatch_summary.csv", index_col=0)
cc = pd.read_csv(f"resources/market/{SCENARIO}_constraint_costs.csv", index_col=0)
cc_no_total = cc[cc.index != "TOTAL"].copy()
bm_val = pd.read_csv(f"resources/market/{SCENARIO}_bm_validation.csv",
                     index_col=0, skipinitialspace=True)
bm_val.columns = bm_val.columns.str.strip()
bm_val.index = bm_val.index.str.strip()

print("Aggregating dispatch by carrier...")
bm_by_c = agg_by_carrier(bm_d, gen_carrier.to_dict())

# Also load wholesale dispatch (may be >50 MB, try from network if CSV fails)
try:
    # Large CSV — load directly from network .nc
    raise NotImplementedError("Use network data")
except Exception:
    ws_gen_p = n_ws.generators_t.p
    ws_by_c = agg_by_carrier(ws_gen_p, gen_carrier.to_dict())

# Storage: pumped hydro vs batteries
ps_cols = [c for c in bm_s.columns
           if any(x in c for x in ["Dinorwig", "Ffestiniog", "Cruachan", "Foyers"])]
batt_cols = [c for c in bm_s.columns if c not in ps_cols]

ps_gen = bm_s[ps_cols].clip(lower=0).sum(axis=1)
ps_chg = (-bm_s[ps_cols].clip(upper=0)).sum(axis=1)
batt_gen = bm_s[batt_cols].clip(lower=0).sum(axis=1)
batt_chg = (-bm_s[batt_cols].clip(upper=0)).sum(axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: WHOLESALE PRICE vs ELEXON MID (N2EX)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 1: WHOLESALE PRICE vs ELEXON MID (N2EX) — Full Year 2020")
print(SEP)

sns = ws_price.index
mid_aligned = mid_2020["mid_price"].reindex(sns, method="nearest")
model_smp = ws_price["wholesale_price"]

print(f"\n  {'Metric':<40}  {'Model':>12}  {'MID/ELEXON':>12}  {'Ratio':>7}")
print(f"  {SEP2}")
show_row("Mean SMP/MID (£/MWh)", model_smp.mean(), mid_aligned.mean(), unit="£/MWh")
show_row("Median SMP/MID (£/MWh)", model_smp.median(), mid_aligned.median(), unit="£/MWh")
show_row("Std SMP/MID (£/MWh)", model_smp.std(), mid_aligned.std(), unit="£/MWh")
show_row("Min SMP/MID (£/MWh)", model_smp.min(), mid_aligned.min(), unit="£/MWh")
show_row("Max SMP/MID (£/MWh)", model_smp.max(), mid_aligned.max(), unit="£/MWh")

r_price = corr(model_smp, mid_aligned)
both_price = pd.concat([model_smp, mid_aligned], axis=1).dropna()
both_price.columns = ["model", "mid"]
mae = float((both_price["model"] - both_price["mid"]).abs().mean())
rmse = float(((both_price["model"] - both_price["mid"])**2).mean()**0.5)

print(f"\n  Hourly correlation (r):  {r_price:.4f}")
print(f"  R² (price):              {r_price**2:.4f}")
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

# Price by month
print(f"\n  === Monthly average price (model vs MID) ===")
print(f"  {'Month':>7}  {'Model £/MWh':>12}  {'MID £/MWh':>12}  {'Delta':>8}  {'r':>6}")
for m in range(1, 13):
    mask_m = model_smp.index.month == m
    mask_e = mid_aligned.index.month == m
    m_m = model_smp[mask_m].mean()
    e_m = mid_aligned[mask_e].mean()
    r_m = corr(model_smp[mask_m], mid_aligned[mask_e])
    r_str = f"{r_m:6.3f}" if not np.isnan(r_m) else "  n/a"
    print(f"  {m:>7}  {m_m:>12.2f}  {e_m:>12.2f}  {m_m-e_m:>+8.2f}  {r_str}")

# Negative price hours
neg_model = (model_smp < 0).sum()
neg_mid = (mid_aligned < 0).sum()
print(f"\n  Negative price hours:  Model {neg_model},  MID {neg_mid}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CARRIER DISPATCH (BM final) vs ESPENI — ANNUAL TOTALS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 2: CARRIER DISPATCH (BM final) vs ESPENI — Full Year 2020")
print(SEP)

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

print(f"\n  {'Carrier':<32}  {'Model TWh':>10}  {'ESPENI TWh':>10}  {'Ratio':>7}  {'r':>6}")
print(f"  {SEP2}")

for carrier, ecol in ESPENI_MAP.items():
    if carrier not in bm_by_c.columns:
        continue
    model_t = float(bm_by_c[carrier].sum())
    esp_t = hourly_mwh(esp[ecol].dropna()) if ecol in esp.columns else float("nan")
    r_val = corr(bm_by_c[carrier], esp_h[ecol]) if ecol in esp_h.columns else float("nan")
    ratio = model_t / esp_t if esp_t and not np.isnan(esp_t) and esp_t != 0 else float("nan")
    r_str = f"{r_val:6.3f}" if not np.isnan(r_val) else "  n/a"
    ratio_str = f"{ratio:7.3f}" if not np.isnan(ratio) else "    n/a"
    print(f"  {carrier:<32}  {model_t/1e6:>10.2f}  {esp_t/1e6:>10.2f}  {ratio_str}  {r_str}")

# Wind
w_onshore = float(bm_by_c.get("wind_onshore", pd.Series([0])).sum())
w_offshore = float(bm_by_c.get("wind_offshore", pd.Series([0])).sum())
w_embedded = float(bm_by_c.get("embedded_wind", pd.Series([0])).sum())
w_model_excl_emb = w_onshore + w_offshore
w_model_total = w_model_excl_emb + w_embedded

elex_wind_mwh = hourly_mwh(esp["ELEC_POWER_ELEX_WIND[MW](float32)"])
try:
    ngem_wind_mwh = float(esp_h["ELEC_POWER_TOTAL_WIND[MW](float32)"].sum())
except KeyError:
    ngem_wind_mwh = elex_wind_mwh + hourly_mwh(esp.get("ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)", pd.Series([0])))

wind_model_h = bm_by_c.get("wind_onshore", pd.Series(0, index=bm_by_c.index)) + \
               bm_by_c.get("wind_offshore", pd.Series(0, index=bm_by_c.index))
wind_esp_h = esp_h["ELEC_POWER_ELEX_WIND[MW](float32)"]
r_wind = corr(wind_model_h, wind_esp_h)

print(f"  {'wind_on+offshore (vs ELEX_WIND)':<32}  {w_model_excl_emb/1e6:>10.2f}  {elex_wind_mwh/1e6:>10.2f}  "
      f"{w_model_excl_emb/elex_wind_mwh:7.3f}  {r_wind:6.3f}")
print(f"  {'wind_total incl embedded (NGEM)':<32}  {w_model_total/1e6:>10.2f}  {ngem_wind_mwh/1e6:>10.2f}  "
      f"{w_model_total/ngem_wind_mwh:7.3f}")

# Solar
solar_pv = float(bm_by_c.get("solar_pv", pd.Series([0])).sum())
emb_solar = float(bm_by_c.get("embedded_solar", pd.Series([0])).sum())
solar_total_model = solar_pv + emb_solar
solar_esp = float(esp_h["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"].sum())
print(f"  {'solar_pv+embedded_solar':<32}  {solar_total_model/1e6:>10.2f}  {solar_esp/1e6:>10.2f}  "
      f"{solar_total_model/solar_esp:7.3f}")

# Pumped hydro
ps_gen_mwh = float(ps_gen.sum())
ps_chg_mwh = float(ps_chg.sum())
ps_elex_dis = hourly_mwh(esp["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"])
ps_elex_chg = abs(hourly_mwh(esp["ELEC_POWER_ELEX_PS_CHARGING[MW](float32)"]))
r_ps = corr(ps_gen, esp_h["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"])
print(f"  {'pumped_hydro dispatch (gen)':<32}  {ps_gen_mwh/1e6:>10.2f}  {ps_elex_dis/1e6:>10.2f}  "
      f"{ps_gen_mwh/ps_elex_dis:7.3f}  {r_ps:6.3f}")
print(f"  {'pumped_hydro charging':<32}  {ps_chg_mwh/1e6:>10.2f}  {ps_elex_chg/1e6:>10.2f}  "
      f"{ps_chg_mwh/ps_elex_chg:7.3f}")

# Battery
batt_gen_mwh = float(batt_gen.sum())
batt_chg_mwh = float(batt_chg.sum())
print(f"  {'battery dispatch (gen)':<32}  {batt_gen_mwh/1e6:>10.2f}")
print(f"  {'battery charging':<32}  {batt_chg_mwh/1e6:>10.2f}")

# Total demand
demand_model = float(n_bm.loads_t.p_set.sum(axis=1).sum()) if len(n_bm.loads_t.p_set) > 0 else 0
esp_demand = float(esp_h["ELEC_POWER_TOTAL_ESPENI[MW](float32)"].sum())
print()
print(f"  {'TOTAL DEMAND (ESPENI)':<32}  {demand_model/1e6:>10.2f}  {esp_demand/1e6:>10.2f}  "
      f"{demand_model/esp_demand:7.3f}")

# Generation mix shares
print(f"\n  === Generation Mix Shares (BM final) ===")
all_gen_mwh = float(bm_by_c.sum().sum()) + ps_gen_mwh + batt_gen_mwh
print(f"  {'Carrier':<25}  {'Model %':>8}  {'ESPENI %':>8}")
print(f"  {SEP2}")
# Get ESPENI total generation: sum of fuel types
total_espeni_gen = sum(hourly_mwh(esp[c].dropna()) for c in ESPENI_MAP.values() if c in esp.columns) + \
                   elex_wind_mwh + ps_elex_dis
for carrier, ecol in ESPENI_MAP.items():
    if carrier not in bm_by_c.columns:
        continue
    model_share = float(bm_by_c[carrier].sum()) / all_gen_mwh * 100
    esp_t = hourly_mwh(esp[ecol].dropna()) if ecol in esp.columns else 0
    esp_share = esp_t / total_espeni_gen * 100
    print(f"  {carrier:<25}  {model_share:>7.1f}%  {esp_share:>7.1f}%")
print(f"  {'wind_on+offshore':<25}  {w_model_excl_emb / all_gen_mwh * 100:>7.1f}%  "
      f"{elex_wind_mwh / total_espeni_gen * 100:>7.1f}%")
print(f"  {'pumped_hydro':<25}  {ps_gen_mwh / all_gen_mwh * 100:>7.1f}%  "
      f"{ps_elex_dis / total_espeni_gen * 100:>7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MONTHLY GENERATION BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 3: MONTHLY GENERATION COMPARISON (Model vs ESPENI, TWh)")
print(SEP)

key_carriers = ["CCGT", "nuclear", "coal", "wind_onshore", "wind_offshore"]
key_espeni = {
    "CCGT": "ELEC_POWER_ELEX_CCGT[MW](float32)",
    "nuclear": "ELEC_POWER_ELEX_NUCLEAR[MW](float32)",
    "coal": "ELEC_POWER_ELEX_COAL[MW](float32)",
    "wind (ELEX)": "ELEC_POWER_ELEX_WIND[MW](float32)",
}

print(f"\n  {'Carrier':<16}  {'Month':>5}  {'Model TWh':>10}  {'ESPENI TWh':>10}  {'Ratio':>7}")
print(f"  {SEP2}")
for carrier_name, ecol in key_espeni.items():
    if carrier_name == "wind (ELEX)":
        # Use on+offshore combined
        model_series = wind_model_h
    else:
        model_series = bm_by_c.get(carrier_name, pd.Series(0, index=bm_by_c.index))
    esp_series = esp_h.get(ecol, pd.Series(0, index=esp_h.index))
    for m in range(1, 13):
        mask_m = model_series.index.month == m
        mask_e = esp_series.index.month == m
        model_twh = float(model_series[mask_m].sum()) / 1e6
        esp_twh = float(esp_series[mask_e].sum()) / 2 / 1e6  # half-hourly→hourly avg already done
        # esp_h is already hourly mean, so sum gives MWh
        esp_twh = float(esp_series[mask_e].sum()) / 1e6
        ratio = model_twh / esp_twh if esp_twh != 0 else float("nan")
        ratio_str = f"{ratio:7.2f}" if not np.isnan(ratio) else "    n/a"
        print(f"  {carrier_name:<16}  {m:>5}  {model_twh:>10.3f}  {esp_twh:>10.3f}  {ratio_str}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BM VOLUMES vs ELEXON BOALF
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 4: BALANCING MECHANISM VOLUMES vs ELEXON BOALF — Full Year 2020")
print(SEP)

print(f"\n  {'Metric':<55}  {'Model':>14}  {'Elexon':>14}  {'Ratio':>7}")
print(f"  {SEP2}")

rows_to_show = [
    "Annualised BM cost",
    "Total increase volume",
    "Total decrease volume",
    "Total increase volume vs BOALF (unflagged)",
    "Total decrease volume vs BOALF (unflagged)",
    "Mean wholesale price (SMP)",
    "Mean wholesale price (SMP) vs SBP",
    "BM constraint cost per hour",
    "Increase volume: CCGT",
    "Increase volume: Pumped Storage Hydroelectricity",
    "Increase volume: biomass",
    "Increase volume: coal",
    "Increase volume: large_hydro",
    "Increase volume: nuclear",
    "Increase volume: wind_onshore",
    "Increase volume: wind_offshore",
]

for idx in rows_to_show:
    m_val = try_get_numeric(bm_val, idx, "model_value")
    e_val = try_get_numeric(bm_val, idx, "elexon_value")
    r_val = try_get_numeric(bm_val, idx, "ratio")
    m_str = f"{m_val:>14,.0f}" if not np.isnan(m_val) else f"{'—':>14}"
    e_str = f"{e_val:>14,.0f}" if not np.isnan(e_val) else f"{'—':>14}"
    r_str = f"{r_val:>7.2f}" if not np.isnan(r_val) else f"{'—':>7}"
    unit = try_get_numeric(bm_val, idx, "unit") if False else ""
    print(f"  {idx:<55}  {m_str}  {e_str}  {r_str}")

# Curtailment summary
print(f"\n  === Wind Curtailment (BM) ===")
wind_bm_dec_on = float(cc.loc["wind_onshore", "decrease_MWh"]) if "wind_onshore" in cc.index else 0
wind_bm_dec_off = float(cc.loc["wind_offshore", "decrease_MWh"]) if "wind_offshore" in cc.index else 0
emb_wind_dec = float(cc.loc["embedded_wind", "decrease_MWh"]) if "embedded_wind" in cc.index else 0
print(f"  Wind onshore BM decrease:   {wind_bm_dec_on:>14,.0f} MWh ({wind_bm_dec_on/1e6:.2f} TWh)")
print(f"  Wind offshore BM decrease:  {wind_bm_dec_off:>14,.0f} MWh ({wind_bm_dec_off/1e6:.2f} TWh)")
print(f"  Embedded wind BM decrease:  {emb_wind_dec:>14,.0f} MWh ({emb_wind_dec/1e6:.2f} TWh)")
total_wind_curtail = wind_bm_dec_on + wind_bm_dec_off + emb_wind_dec
ws_wind_total = float(ws_by_c.get("wind_onshore", pd.Series([0])).sum() +
                      ws_by_c.get("wind_offshore", pd.Series([0])).sum() +
                      ws_by_c.get("embedded_wind", pd.Series([0])).sum())
print(f"  Total wind curtailment:     {total_wind_curtail:>14,.0f} MWh ({total_wind_curtail/1e6:.2f} TWh)")
if ws_wind_total > 0:
    print(f"  Wind curtailment rate:      {total_wind_curtail/ws_wind_total*100:>14.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: INTERCONNECTORS vs ESPENI
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 5: INTERCONNECTOR FLOWS vs ESPENI — Full Year 2020")
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
print(f"\n  {'IC':<35}  {'Model TWh':>10}  {'ESPENI TWh':>10}  {'Ratio':>7}  {'r':>6}")
print(f"  {SEP2}")

total_model_import = 0
for short, cfg in ESPENI_IC.items():
    ecol = cfg["esp_col"]
    link_names = [lk for lk in cfg["links"] if lk in n_ws.links_t.p_set.columns]
    if link_names:
        model_h_ic = -n_ws.links_t.p_set[link_names].sum(axis=1)
        model_import = float(model_h_ic.sum())
        total_model_import += model_import
    else:
        model_h_ic = None
        model_import = float("nan")

    if ecol in esp.columns:
        e_import = hourly_mwh(esp[ecol].dropna())
        r_ic = corr(model_h_ic, esp_h[ecol]) if model_h_ic is not None else float("nan")
    else:
        e_import = float("nan")
        r_ic = float("nan")

    ratio = model_import / e_import if e_import and not np.isnan(e_import) and e_import != 0 else float("nan")
    r_str = f"{r_ic:6.3f}" if not np.isnan(r_ic) else "  n/a"
    ratio_str = f"{ratio:7.3f}" if not np.isnan(ratio) else "    n/a"
    print(f"  {short:<35}  {model_import/1e6:>10.2f}  {e_import/1e6:>10.2f}  {ratio_str}  {r_str}")

total_espeni_import = float(esp_h["ELEC_POWER_ELEX_NET_IMPORTS[MW](float32)"].sum())
print(f"  {'TOTAL NET IMPORT':<35}  {total_model_import/1e6:>10.2f}  {total_espeni_import/1e6:>10.2f}  "
      f"{total_model_import/total_espeni_import if total_espeni_import else 0:7.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CONSTRAINT COSTS BY CARRIER
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 6: CONSTRAINT COSTS BY CARRIER (BM final)")
print(SEP)
total_cost = float(cc.loc["TOTAL", "net_cost"]) if "TOTAL" in cc.index else float(cc_no_total["net_cost"].sum())
print(f"\n  {'Carrier':<30}  {'Offer £m':>10}  {'Bid £m':>10}  {'Net £m':>10}  {'% total':>8}")
print(f"  {SEP2}")
for carrier, row in cc_no_total.sort_values("net_cost", ascending=False).iterrows():
    pct = row["net_cost"] / total_cost * 100 if total_cost else 0
    print(f"  {str(carrier):<30}  {row['offer_cost']/1e6:>10.1f}  {row['bid_cost']/1e6:>10.1f}  "
          f"{row['net_cost']/1e6:>10.1f}  {pct:>7.1f}%")
print(f"  {'TOTAL':<30}  {'':>10}  {'':>10}  {total_cost/1e6:>10.1f}  {'100.0%':>8}")

# ELEXON benchmark: ~£1.4bn/year BM cost
bm_benchmark = 1.4e9
print(f"\n  Model BM cost:             £{total_cost/1e6:,.1f}m")
print(f"  NESO published benchmark:  £{bm_benchmark/1e6:,.0f}m/year")
print(f"  Model / benchmark ratio:   {total_cost/bm_benchmark:.3f}")
print(f"  BM cost per hour:          £{total_cost/HOURS_IN_YEAR:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CONGESTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 7: NETWORK CONGESTION — Full Year 2020")
print(SEP)
print(f"\n  Total congested lines/transformers: {len(cong)}")
print(f"\n  {'Component':<30}  {'Type':>12}  {'s_nom MVA':>10}  {'Hours':>8}  {'Max load':>10}  {'Mean load':>10}")
print(f"  {SEP2}")
for comp, row in cong.sort_values("hours_congested", ascending=False).head(25).iterrows():
    print(f"  {comp:<30}  {row['type']:>12}  {row['s_nom_MVA']:>10.0f}  "
          f"{row['hours_congested']:>8.0f}  {row['max_loading_fraction']:>10.3f}  "
          f"{row['mean_loading_fraction']:>10.3f}")

# Congestion distribution
total_hours = HOURS_IN_YEAR
print(f"\n  Congestion severity distribution:")
bins_c = [0, 500, 1000, 2000, 4000, 8784]
labels_c = ["0-500h", "500-1000h", "1000-2000h", "2000-4000h", ">4000h"]
for i, (lo, hi) in enumerate(zip(bins_c, bins_c[1:])):
    n = ((cong["hours_congested"] >= lo) & (cong["hours_congested"] < hi)).sum()
    print(f"    {labels_c[i]:>12}: {n:>3} lines/transformers")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: NODAL PRICE SPREAD
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 8: NODAL PRICE SPREAD (Network Constraint Signal)")
print(SEP)

print(f"\n  Mean wholesale (SMP):           £{pc['wholesale_price'].mean():>8.2f}/MWh")
print(f"  Mean nodal price (demand):      £{pc['mean_nodal_price'].mean():>8.2f}/MWh")
print(f"  Mean nodal spread (max-min):    £{pc['nodal_spread'].mean():>8.2f}/MWh")
print(f"  Max nodal spread (any hour):    £{pc['nodal_spread'].max():>8.2f}/MWh")
print(f"  % hours with spread > £50:      {(pc['nodal_spread'] > 50).mean()*100:>8.1f}%")
print(f"  % hours with any neg price:     {(pc['min_nodal_price'] < 0).mean()*100:>8.1f}%")
print(f"  % hours spread < £5 (copperplate): {(pc['nodal_spread'] < 5).mean()*100:>8.1f}%")

# Monthly spread
print(f"\n  Monthly nodal spread (£/MWh):")
print(f"  {'Month':>7}  {'Mean':>8}  {'Max':>8}  {'%>£50':>8}")
for m in range(1, 13):
    mask = pc.index.month == m
    sp = pc.loc[mask, "nodal_spread"]
    print(f"  {m:>7}  {sp.mean():>8.1f}  {sp.max():>8.0f}  {(sp>50).mean()*100:>7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: HOURLY PROFILE CORRELATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 9: HOURLY PROFILE CORRELATION (BM vs ESPENI)")
print(SEP)

print(f"\n  {'Carrier':<32}  {'r':>8}  {'MAE (MW)':>12}")
print(f"  {SEP2}")
for carrier, ecol in ESPENI_MAP.items():
    if ecol not in esp_h.columns or carrier not in bm_by_c.columns:
        continue
    e_series = esp_h[ecol]
    bm_c = bm_by_c[carrier]
    r_bm = corr(bm_c, e_series)
    mae_bm = float((bm_c - e_series.reindex(bm_c.index)).abs().mean())
    r_bm_s = f"{r_bm:8.4f}" if not np.isnan(r_bm) else "     n/a"
    print(f"  {carrier:<32}  {r_bm_s}  {mae_bm:>12.1f}")

# Wind + price
r_wind_bm = corr(wind_model_h, wind_esp_h)
mae_wind = float((wind_model_h - wind_esp_h.reindex(wind_model_h.index)).abs().mean())
print(f"  {'wind_on+offshore':<32}  {r_wind_bm:8.4f}  {mae_wind:>12.1f}")
print(f"  {'wholesale price vs MID':<32}  {r_price:8.4f}  {mae:>12.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: WIND PERFORMANCE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 10: WIND PERFORMANCE DIAGNOSTIC")
print(SEP)

ws_wind_on = float(ws_by_c.get("wind_onshore", pd.Series([0])).sum())
ws_wind_off = float(ws_by_c.get("wind_offshore", pd.Series([0])).sum())
ws_wind_emb = float(ws_by_c.get("embedded_wind", pd.Series([0])).sum())
ws_wind_all = ws_wind_on + ws_wind_off

bm_wind_on = float(bm_by_c.get("wind_onshore", pd.Series([0])).sum())
bm_wind_off = float(bm_by_c.get("wind_offshore", pd.Series([0])).sum())
bm_wind_emb = float(bm_by_c.get("embedded_wind", pd.Series([0])).sum())
bm_wind_all = bm_wind_on + bm_wind_off

print(f"\n  {'Component':<30}  {'Wholesale TWh':>14}  {'BM Final TWh':>14}  {'Curtailed TWh':>14}  {'Curt %':>8}")
print(f"  {SEP2}")
for name, ws_val, bm_val_w in [("wind_onshore", ws_wind_on, bm_wind_on),
                                ("wind_offshore", ws_wind_off, bm_wind_off),
                                ("embedded_wind", ws_wind_emb, bm_wind_emb)]:
    curt = ws_val - bm_val_w
    curt_pct = curt / ws_val * 100 if ws_val > 0 else 0
    print(f"  {name:<30}  {ws_val/1e6:>14.2f}  {bm_val_w/1e6:>14.2f}  {curt/1e6:>14.2f}  {curt_pct:>7.1f}%")

print(f"  {'TOTAL':<30}  {(ws_wind_all+ws_wind_emb)/1e6:>14.2f}  {(bm_wind_all+bm_wind_emb)/1e6:>14.2f}  "
      f"{((ws_wind_all+ws_wind_emb)-(bm_wind_all+bm_wind_emb))/1e6:>14.2f}  "
      f"{((ws_wind_all+ws_wind_emb)-(bm_wind_all+bm_wind_emb))/(ws_wind_all+ws_wind_emb)*100:>7.1f}%")

print(f"\n  ESPENI metered wind dispatch (ELEX_WIND):  {elex_wind_mwh/1e6:.2f} TWh")
print(f"  Model BM wind (excl embedded):             {bm_wind_all/1e6:.2f} TWh")
print(f"  Model BM / ESPENI ratio:                   {bm_wind_all/elex_wind_mwh:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: KEY FINDINGS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SECTION 11: KEY FINDINGS SUMMARY vs VALIDATION TARGETS")
print(SEP)

ccgt_model = float(bm_by_c["CCGT"].sum()) if "CCGT" in bm_by_c.columns else 0
ccgt_espeni = hourly_mwh(esp["ELEC_POWER_ELEX_CCGT[MW](float32)"]) if "ELEC_POWER_ELEX_CCGT[MW](float32)" in esp.columns else 1
nuclear_model = float(bm_by_c["nuclear"].sum()) if "nuclear" in bm_by_c.columns else 0
nuclear_espeni = hourly_mwh(esp["ELEC_POWER_ELEX_NUCLEAR[MW](float32)"]) if "ELEC_POWER_ELEX_NUCLEAR[MW](float32)" in esp.columns else 1
coal_model = float(bm_by_c["coal"].sum()) if "coal" in bm_by_c.columns else 0
coal_espeni = hourly_mwh(esp["ELEC_POWER_ELEX_COAL[MW](float32)"]) if "ELEC_POWER_ELEX_COAL[MW](float32)" in esp.columns else 1
hydro_model = float(bm_by_c.get("large_hydro", pd.Series([0])).sum())
hydro_espeni = hourly_mwh(esp["ELEC_POWER_ELEX_NPSHYD[MW](float32)"]) if "ELEC_POWER_ELEX_NPSHYD[MW](float32)" in esp.columns else 1

targets = {
    "Mean SMP / MID":                (model_smp.mean() / mid_aligned.mean(), 0.90, 1.10),
    "SMP hourly r":                  (r_price, 0.70, 1.00),
    "CCGT total (BM/ESPENI)":        (ccgt_model / ccgt_espeni, 0.85, 1.15),
    "CCGT hourly r":                 (corr(bm_by_c.get("CCGT", pd.Series(0, index=bm_by_c.index)), esp_h.get("ELEC_POWER_ELEX_CCGT[MW](float32)", pd.Series(0))), 0.85, 1.00),
    "Nuclear total (BM/ESPENI)":     (nuclear_model / nuclear_espeni, 0.85, 1.15),
    "Coal total (BM/ESPENI)":        (coal_model / coal_espeni, 0.80, 1.20),
    "Wind total (BM/ESPENI)":        (bm_wind_all / elex_wind_mwh, 0.85, 1.15),
    "Wind hourly r":                 (r_wind, 0.90, 1.00),
    "Hydro total (BM/ESPENI)":       (hydro_model / hydro_espeni, 0.70, 1.30),
    "IC net import (model/ESPENI)":  (total_model_import / total_espeni_import if total_espeni_import else 0, 0.80, 1.20),
    "Demand (model/ESPENI)":         (demand_model / esp_demand, 0.95, 1.05),
    "BM cost / £1.4bn benchmark":    (total_cost / bm_benchmark, 0.05, 0.25),
    "PS BM dispatch / ESPENI":       (ps_gen_mwh / ps_elex_dis if ps_elex_dis else 0, 0.50, 2.00),
}

print(f"\n  {'Metric':<40}  {'Value':>8}  {'Target range':>18}  {'Status':>8}")
print(f"  {SEP2}")
summary_rows = []
for metric, (val, lo, hi) in targets.items():
    if np.isnan(val):
        status = "  ???"
    elif lo <= val <= hi:
        status = "  PASS"
    elif abs(val - (lo + hi) / 2) < 0.15 * (lo + hi) / 2:
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

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
summary_dir.mkdir(parents=True, exist_ok=True)
summary_csv = summary_dir / f"{SCENARIO}_deep_dive_summary.csv"
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_csv, index=False)
print(f"\n  Summary saved to: {summary_csv}")

print()
print(SEP)
print("DONE — Full year 2020 deep-dive validation complete.")
print(SEP)
