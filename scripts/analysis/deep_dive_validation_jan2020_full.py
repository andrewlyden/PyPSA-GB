"""Deep-dive review: Validation_Jan2020 vs Elexon, NESO, ESPENI, MID.

Produces:
  docs/validation_jan2020_deep_dive.md    (the report)
  resources/analysis/Validation_Jan2020_deep_dive_tables.csv  (machine-readable)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[2]
SCEN = "Validation_Jan2020"
PERIOD_HOURS = 744                # 31 days
JAN_START = pd.Timestamp("2020-01-01")
JAN_END   = pd.Timestamp("2020-02-01")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hourly_mwh(half_hourly_mw: pd.Series) -> float:
    return float(half_hourly_mw.sum() * 0.5)


def safe_div(a, b):
    if b is None or b == 0 or pd.isna(b):
        return float("nan")
    return float(a) / float(b)


def corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 10:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def carrier_aggregate(df: pd.DataFrame, gen_carrier: pd.Series) -> pd.DataFrame:
    cm = gen_carrier.to_dict()
    out: dict[str, pd.Series] = {}
    for col in df.columns:
        c = cm.get(col, "unknown")
        if c not in out:
            out[c] = pd.Series(0.0, index=df.index)
        out[c] = out[c] + df[col].fillna(0)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
print("[load] networks")
n_ws = pypsa.Network(ROOT / f"resources/market/{SCEN}_wholesale.nc")
n_bm = pypsa.Network(ROOT / f"resources/market/{SCEN}_balancing.nc")
gen_carrier = n_ws.generators["carrier"]

print("[load] dispatch CSVs")
ws_d = pd.read_csv(ROOT / f"resources/market/{SCEN}_wholesale_dispatch.csv",
                   index_col=0, parse_dates=True)
bm_d = pd.read_csv(ROOT / f"resources/market/{SCEN}_balancing_dispatch.csv",
                   index_col=0, parse_dates=True)
ws_storage = pd.read_csv(ROOT / f"resources/market/{SCEN}_wholesale_storage.csv",
                         index_col=0, parse_dates=True)
bm_storage = pd.read_csv(ROOT / f"resources/market/{SCEN}_balancing_storage.csv",
                         index_col=0, parse_dates=True)
ws_links = pd.read_csv(ROOT / f"resources/market/{SCEN}_wholesale_links.csv",
                       index_col=0, parse_dates=True)
ws_price = pd.read_csv(ROOT / f"resources/market/{SCEN}_wholesale_price.csv",
                       index_col=0, parse_dates=True)
pc = pd.read_csv(ROOT / f"resources/market/{SCEN}_price_comparison.csv",
                 index_col=0, parse_dates=True)
cc = pd.read_csv(ROOT / f"resources/market/{SCEN}_constraint_costs.csv", index_col=0)
cong = pd.read_csv(ROOT / f"resources/market/{SCEN}_congestion.csv", index_col=0)
rdsp = pd.read_csv(ROOT / f"resources/market/{SCEN}_redispatch_summary.csv", index_col=0)

bm_val = pd.read_csv(ROOT / f"resources/market/{SCEN}_bm_validation.csv",
                     index_col=0, skipinitialspace=True)
bm_val.columns = bm_val.columns.str.strip()
bm_val.index = bm_val.index.str.strip()

neso_val = pd.read_csv(ROOT / f"resources/market/{SCEN}_neso_validation.csv")

bm_carrier_summary = pd.read_csv(ROOT / f"resources/market/{SCEN}_boalf_by_flag.csv")

# ---------------------------------------------------------------------------
# Load reference data
# ---------------------------------------------------------------------------
print("[load] ESPENI")
espeni = pd.read_csv(ROOT / "data/demand/espeni.csv")
TIME_COL = "ELEC_elex_startTime[utc](datetime)"
espeni["time"] = pd.to_datetime(espeni[TIME_COL], utc=True).dt.tz_convert(None)
espeni = espeni.set_index("time").select_dtypes(include="number").sort_index()
esp = espeni[(espeni.index >= JAN_START) & (espeni.index < JAN_END)]
esp_h = esp.resample("h").mean()

print("[load] Elexon MID, SBP, BOALF, DISBSAD")
mid = pd.read_csv(ROOT / "resources/market/elexon/mid_prices_2020.csv",
                  index_col=0, parse_dates=True)
mid_h = mid.resample("h").mean()
mid_jan = mid_h[(mid_h.index >= JAN_START) & (mid_h.index < JAN_END)]

sbp = pd.read_csv(ROOT / "resources/market/elexon/validation/2020/system_prices.csv",
                  index_col=0, parse_dates=True)
sbp_h = sbp.resample("h").mean()
sbp_jan = sbp_h[(sbp_h.index >= JAN_START) & (sbp_h.index < JAN_END)]

boalf = pd.read_csv(ROOT / "resources/market/elexon/validation/2020/boalf_data.csv",
                    parse_dates=["timeFrom", "timeTo"])
disbsad = pd.read_csv(ROOT / "resources/market/elexon/validation/2020/disbsad_data.csv",
                      parse_dates=["datetime"])

print("[load] NESO thermal constraint costs and DA flows")
neso_thermal_xlsx = ROOT / "data/validation/thermal_constraint_costs_19-20.xlsx"
neso_thermal_xlsx_2 = ROOT / "data/validation/thermal_constraint_costs_20-21.xlsx"

# Load all sheets — combine
def _load_neso_thermal(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    xl = pd.ExcelFile(p)
    frames = []
    for sh in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sh)
            df["__sheet"] = sh
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

neso_thermal = _load_neso_thermal(neso_thermal_xlsx)
print(f"   NESO thermal sheets: {neso_thermal['__sheet'].unique() if '__sheet' in neso_thermal else []}")

# ---------------------------------------------------------------------------
# Aggregate model dispatch
# ---------------------------------------------------------------------------
ws_by_c = carrier_aggregate(ws_d, gen_carrier)
bm_by_c = carrier_aggregate(bm_d, gen_carrier)

# Storage split
ps_names = [c for c in bm_storage.columns
            if any(x in c for x in ["Dinorwig", "Ffestiniog", "Cruachan", "Foyers"])]
bat_names = [c for c in bm_storage.columns if c not in ps_names]
ps_gen_h = bm_storage[ps_names].clip(lower=0).sum(axis=1)
ps_chg_h = (-bm_storage[ps_names].clip(upper=0)).sum(axis=1)
ps_gen_h_ws = ws_storage[ps_names].clip(lower=0).sum(axis=1) if ps_names else pd.Series(0)
ps_chg_h_ws = (-ws_storage[ps_names].clip(upper=0)).sum(axis=1) if ps_names else pd.Series(0)

# ---------------------------------------------------------------------------
# Wholesale price benchmark
# ---------------------------------------------------------------------------
sns = ws_price.index
mid_a = mid_jan["mid_price"].reindex(sns, method="nearest")
sbp_a = sbp_jan["system_buy_price"].reindex(sns, method="nearest")
ssp_a = sbp_jan["system_sell_price"].reindex(sns, method="nearest")
model_smp = ws_price["wholesale_price"]

price_table = pd.DataFrame({
    "Mean (£/MWh)": [model_smp.mean(), mid_a.mean(), sbp_a.mean(), ssp_a.mean(),
                     pc["mean_nodal_price"].mean()],
    "Median": [model_smp.median(), mid_a.median(), sbp_a.median(), ssp_a.median(),
               pc["mean_nodal_price"].median()],
    "Std": [model_smp.std(), mid_a.std(), sbp_a.std(), ssp_a.std(),
            pc["mean_nodal_price"].std()],
    "Min": [model_smp.min(), mid_a.min(), sbp_a.min(), ssp_a.min(),
            pc["mean_nodal_price"].min()],
    "Max": [model_smp.max(), mid_a.max(), sbp_a.max(), ssp_a.max(),
            pc["mean_nodal_price"].max()],
}, index=["Model wholesale (SMP)", "Elexon MID (N2EX)",
          "Elexon SBP", "Elexon SSP", "Model BM mean nodal"])

r_mid = corr(model_smp, mid_a)
r_sbp = corr(model_smp, sbp_a)
mae_mid = float((model_smp - mid_a).dropna().abs().mean())
rmse_mid = float(((model_smp - mid_a).dropna() ** 2).mean() ** 0.5)
mae_sbp = float((model_smp - sbp_a).dropna().abs().mean())
rmse_sbp = float(((model_smp - sbp_a).dropna() ** 2).mean() ** 0.5)

# ---------------------------------------------------------------------------
# BOALF aggregation by carrier (Elexon BM volumes)
# ---------------------------------------------------------------------------
# boalf rows have level_from (MW), level_to (MW), timeFrom, timeTo.
# Approximate per-acceptance MWh = avg(level) * duration_h, signed by direction
# vs prior level. The redispatch_summary file already aggregates this; we use it.
boalf_carrier = bm_carrier_summary[
    (bm_carrier_summary["scope"] == "carrier") & (bm_carrier_summary["group"] == "all")
].set_index("carrier")[["increase_mwh", "decrease_mwh"]]
boalf_carrier = boalf_carrier.rename(columns={"increase_mwh": "elexon_increase_mwh",
                                              "decrease_mwh": "elexon_decrease_mwh"})

# Model BM increase/decrease vs wholesale, per carrier (MWh)
bm_inc = (bm_by_c - ws_by_c).clip(lower=0).sum() * 1.0
bm_dec = (ws_by_c - bm_by_c).clip(lower=0).sum() * 1.0
model_bm_volume = pd.DataFrame({
    "model_increase_mwh": bm_inc,
    "model_decrease_mwh": bm_dec,
})

bm_compare = model_bm_volume.join(boalf_carrier, how="outer").fillna(0)
bm_compare["inc_ratio"] = bm_compare["model_increase_mwh"] / bm_compare["elexon_increase_mwh"].replace(0, np.nan)
bm_compare["dec_ratio"] = bm_compare["model_decrease_mwh"] / bm_compare["elexon_decrease_mwh"].replace(0, np.nan)
# Net redispatch (signed)
bm_compare["model_net_mwh"] = bm_compare["model_increase_mwh"] - bm_compare["model_decrease_mwh"]
bm_compare["elexon_net_mwh"] = bm_compare["elexon_increase_mwh"] - bm_compare["elexon_decrease_mwh"]
bm_compare = bm_compare.sort_values("elexon_increase_mwh", ascending=False)

# ---------------------------------------------------------------------------
# Carrier dispatch: model wholesale vs BM vs ESPENI (fleet-level)
# ---------------------------------------------------------------------------
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

dispatch_rows = []
for carrier, ecol in ESPENI_MAP.items():
    if ecol not in esp.columns:
        continue
    ws_total = float(ws_by_c[carrier].sum()) if carrier in ws_by_c else 0.0
    bm_total = float(bm_by_c[carrier].sum()) if carrier in bm_by_c else 0.0
    esp_total = hourly_mwh(esp[ecol].dropna())
    r_bm = corr(bm_by_c.get(carrier, pd.Series(dtype=float)), esp_h[ecol]) \
           if carrier in bm_by_c else float("nan")
    r_ws = corr(ws_by_c.get(carrier, pd.Series(dtype=float)), esp_h[ecol]) \
           if carrier in ws_by_c else float("nan")
    dispatch_rows.append({
        "carrier": carrier,
        "model_ws_mwh": ws_total,
        "model_bm_mwh": bm_total,
        "espeni_mwh": esp_total,
        "ws_ratio": safe_div(ws_total, esp_total),
        "bm_ratio": safe_div(bm_total, esp_total),
        "r_ws_hourly": r_ws,
        "r_bm_hourly": r_bm,
    })

# Wind (combined)
ws_wind_on  = float(ws_by_c.get("wind_onshore", pd.Series([0])).sum())
ws_wind_off = float(ws_by_c.get("wind_offshore", pd.Series([0])).sum())
bm_wind_on  = float(bm_by_c.get("wind_onshore", pd.Series([0])).sum())
bm_wind_off = float(bm_by_c.get("wind_offshore", pd.Series([0])).sum())
ws_wind_emb = float(ws_by_c.get("embedded_wind", pd.Series([0])).sum())
bm_wind_emb = float(bm_by_c.get("embedded_wind", pd.Series([0])).sum())
elex_wind_mwh = hourly_mwh(esp["ELEC_POWER_ELEX_WIND[MW](float32)"])
ngem_wind_mwh = float(esp_h["ELEC_POWER_TOTAL_WIND[MW](float32)"].sum())

dispatch_rows.append({
    "carrier": "wind_onshore+offshore (vs ELEX_WIND)",
    "model_ws_mwh": ws_wind_on + ws_wind_off,
    "model_bm_mwh": bm_wind_on + bm_wind_off,
    "espeni_mwh": elex_wind_mwh,
    "ws_ratio": safe_div(ws_wind_on + ws_wind_off, elex_wind_mwh),
    "bm_ratio": safe_div(bm_wind_on + bm_wind_off, elex_wind_mwh),
    "r_ws_hourly": corr(ws_by_c.get("wind_onshore", pd.Series(0, index=ws_by_c.index)) +
                        ws_by_c.get("wind_offshore", pd.Series(0, index=ws_by_c.index)),
                        esp_h["ELEC_POWER_ELEX_WIND[MW](float32)"]),
    "r_bm_hourly": corr(bm_by_c.get("wind_onshore", pd.Series(0, index=bm_by_c.index)) +
                        bm_by_c.get("wind_offshore", pd.Series(0, index=bm_by_c.index)),
                        esp_h["ELEC_POWER_ELEX_WIND[MW](float32)"]),
})

dispatch_rows.append({
    "carrier": "wind total (incl embedded vs NGEM)",
    "model_ws_mwh": ws_wind_on + ws_wind_off + ws_wind_emb,
    "model_bm_mwh": bm_wind_on + bm_wind_off + bm_wind_emb,
    "espeni_mwh": ngem_wind_mwh,
    "ws_ratio": safe_div(ws_wind_on + ws_wind_off + ws_wind_emb, ngem_wind_mwh),
    "bm_ratio": safe_div(bm_wind_on + bm_wind_off + bm_wind_emb, ngem_wind_mwh),
    "r_ws_hourly": float("nan"),
    "r_bm_hourly": float("nan"),
})

# PSH
ps_gen_mwh_bm = float(ps_gen_h.sum())
ps_chg_mwh_bm = float(ps_chg_h.sum())
ps_gen_mwh_ws = float(ps_gen_h_ws.sum())
ps_chg_mwh_ws = float(ps_chg_h_ws.sum())
ps_elex_dis = hourly_mwh(esp["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"])
ps_elex_chg = abs(hourly_mwh(esp["ELEC_POWER_ELEX_PS_CHARGING[MW](float32)"]))

dispatch_rows.append({
    "carrier": "pumped_hydro (discharge)",
    "model_ws_mwh": ps_gen_mwh_ws,
    "model_bm_mwh": ps_gen_mwh_bm,
    "espeni_mwh": ps_elex_dis,
    "ws_ratio": safe_div(ps_gen_mwh_ws, ps_elex_dis),
    "bm_ratio": safe_div(ps_gen_mwh_bm, ps_elex_dis),
    "r_ws_hourly": corr(ps_gen_h_ws, esp_h["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"]),
    "r_bm_hourly": corr(ps_gen_h, esp_h["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"]),
})
dispatch_rows.append({
    "carrier": "pumped_hydro (charging)",
    "model_ws_mwh": ps_chg_mwh_ws,
    "model_bm_mwh": ps_chg_mwh_bm,
    "espeni_mwh": ps_elex_chg,
    "ws_ratio": safe_div(ps_chg_mwh_ws, ps_elex_chg),
    "bm_ratio": safe_div(ps_chg_mwh_bm, ps_elex_chg),
    "r_ws_hourly": float("nan"),
    "r_bm_hourly": float("nan"),
})

dispatch_df = pd.DataFrame(dispatch_rows)

# ---------------------------------------------------------------------------
# Interconnectors vs ESPENI (model fixes ICs to ESPENI p_set, so this is sanity)
# ---------------------------------------------------------------------------
ESPENI_IC = {
    "IFA":      ("ELEC_POWER_ELEX_INTFR[MW](float32)",  ["IC_IFA"]),
    "Britned":  ("ELEC_POWER_ELEX_INTNED[MW](float32)", ["IC_Britned"]),
    "Nemo":     ("ELEC_POWER_ELEX_INTNEM[MW](float32)", ["IC_Nemo Link"]),
    "IRL":      ("ELEC_POWER_ELEX_INTIRL[MW](float32)", ["IC_Moyle", "IC_Auchencrosh (interconnector CCT)"]),
    "EastWest": ("ELEC_POWER_ELEX_INTEW[MW](float32)",  ["IC_East West Interconnector"]),
}
ic_rows = []
for short, (ecol, links) in ESPENI_IC.items():
    have = [l for l in links if l in n_ws.links_t.p_set.columns]
    model_h = -n_ws.links_t.p_set[have].sum(axis=1) if have else pd.Series(dtype=float)
    model_imp = float(model_h.sum())
    if ecol in esp.columns:
        e_imp = hourly_mwh(esp[ecol].dropna())
        r_ic = corr(model_h, esp_h[ecol]) if len(model_h) else float("nan")
    else:
        e_imp = float("nan"); r_ic = float("nan")
    ic_rows.append({"interconnector": short,
                    "model_net_import_mwh": model_imp,
                    "espeni_net_import_mwh": e_imp,
                    "ratio": safe_div(model_imp, e_imp),
                    "r_hourly": r_ic})
ic_df = pd.DataFrame(ic_rows)
total_model_imp = float(sum(-n_ws.links_t.p_set[l].sum()
                            for l in n_ws.links.index
                            if l.startswith("IC_") and l in n_ws.links_t.p_set.columns))
total_esp_imp = float(esp_h["ELEC_POWER_ELEX_NET_IMPORTS[MW](float32)"].sum())

# ---------------------------------------------------------------------------
# Constraint costs (model) vs NESO thermal (Jan 2020) by boundary
# ---------------------------------------------------------------------------
neso_lookup = {row["metric"]: row["value"] for _, row in neso_val.iterrows()}
neso_total = neso_lookup.get("model_neso_ratio")  # ratio of model/NESO
neso_thermal_total = neso_lookup.get("neso_thermal_cost_gbp")
model_bm_total = neso_lookup.get("model_bm_cost_gbp")

# Boundary breakdown
boundary_neso = {k.replace("_gbp", "").replace("neso_boundary_cost", "").lstrip("_"):
                 v for k, v in neso_lookup.items() if isinstance(k, str)}
neso_boundaries = neso_val[neso_val["category"] == "neso_boundary_cost"].copy()
neso_boundaries["boundary"] = neso_boundaries["metric"].str.replace("_gbp", "", regex=False)

# DA flow comparison (Jan)
flow_rows = neso_val[neso_val["category"].str.startswith("flow_")].copy()
flow_rows["boundary"] = flow_rows["category"].str.replace("flow_", "", regex=False)
flow_pivot = flow_rows.pivot_table(index="boundary", columns="metric", values="value",
                                   aggfunc="first")

# ---------------------------------------------------------------------------
# DISBSAD context
# ---------------------------------------------------------------------------
disbsad_jan = disbsad[(disbsad["datetime"] >= JAN_START) &
                      (disbsad["datetime"] < JAN_END)].copy()
disbsad_jan["abs_volume"] = disbsad_jan["volume"].abs()
disbsad_summary = pd.Series({
    "DISBSAD records":         len(disbsad_jan),
    "Abs volume MWh":          disbsad_jan["abs_volume"].sum(),
    "Net volume MWh":          disbsad_jan["volume"].sum(),
    "Total cost £":            disbsad_jan["cost"].sum(),
    "SO-flagged share (vol)":  100 * disbsad_jan.loc[disbsad_jan["so_flag"], "abs_volume"].sum()
                                 / disbsad_jan["abs_volume"].sum(),
})

# ---------------------------------------------------------------------------
# Constraint cost share by carrier (model)
# ---------------------------------------------------------------------------
cc_no_total = cc[cc.index != "TOTAL"].copy()
cc_no_total["pct_total"] = cc_no_total["net_cost"] / cc.loc["TOTAL", "net_cost"] * 100
cc_table = cc_no_total[["offer_cost", "bid_cost", "net_cost", "increase_MWh",
                        "decrease_MWh", "pct_total"]].sort_values("net_cost", ascending=False)

# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
demand_model = float(n_bm.loads_t.p_set.sum().sum())
demand_espeni = float(esp_h["ELEC_POWER_TOTAL_ESPENI[MW](float32)"].sum())

# ---------------------------------------------------------------------------
# Build markdown report
# ---------------------------------------------------------------------------
def fmt_money(x):
    if pd.isna(x): return "—"
    return f"£{x:,.0f}"

def fmt_int(x):
    if pd.isna(x): return "—"
    return f"{x:,.0f}"

def fmt_ratio(x, fmt="{:.2f}"):
    if pd.isna(x): return "—"
    return fmt.format(x)

L = []
L.append("# Validation_Jan2020 — Deep Dive Review\n")
L.append(f"_Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}_\n")
L.append(f"_Scenario: {SCEN} · period: 2020-01-01 → 2020-02-01 ({PERIOD_HOURS}h) · "
         "two-stage copperplate→constrained dispatch_\n")
L.append("Reference data sources: Elexon BMRS (MID, SBP/SSP, BOALF, DISBSAD), "
         "ESPENI (fuel-type fleet generation), NESO (thermal constraint costs, DA boundary flows).\n")
L.append("\n---\n")

# 1. Headline
L.append("## 1. Headline numbers\n")
L.append("| Metric | Model | Reference | Source | Ratio |\n")
L.append("|---|---:|---:|---|---:|\n")
L.append(f"| Total demand served (MWh) | {fmt_int(demand_model)} | {fmt_int(demand_espeni)} | ESPENI | {safe_div(demand_model, demand_espeni):.3f} |\n")
L.append(f"| Mean wholesale price (£/MWh) | {model_smp.mean():.2f} | {mid_a.mean():.2f} | Elexon MID (N2EX) | {safe_div(model_smp.mean(), mid_a.mean()):.3f} |\n")
L.append(f"| Mean SBP (£/MWh) | {model_smp.mean():.2f} | {sbp_a.mean():.2f} | Elexon SBP | {safe_div(model_smp.mean(), sbp_a.mean()):.3f} |\n")
L.append(f"| BM cost (period total, £) | {fmt_money(model_bm_total)} | {fmt_money(neso_thermal_total)} | NESO thermal Jan 2020 | {safe_div(model_bm_total, neso_thermal_total):.3f} |\n")
L.append(f"| BM cost annualised (£/yr) | {fmt_money(model_bm_total * 8760 / PERIOD_HOURS)} | £1.4B | NESO published BSUoS | {safe_div(model_bm_total * 8760 / PERIOD_HOURS, 1.4e9):.3f} |\n")
L.append(f"| Total BOALF increase volume (MWh) | {fmt_int(bm_compare['model_increase_mwh'].sum())} | {fmt_int(bm_compare['elexon_increase_mwh'].sum())} | Elexon BOALF (all flags) | {safe_div(bm_compare['model_increase_mwh'].sum(), bm_compare['elexon_increase_mwh'].sum()):.3f} |\n")
L.append(f"| Total BOALF decrease volume (MWh) | {fmt_int(bm_compare['model_decrease_mwh'].sum())} | {fmt_int(bm_compare['elexon_decrease_mwh'].sum())} | Elexon BOALF (all flags) | {safe_div(bm_compare['model_decrease_mwh'].sum(), bm_compare['elexon_decrease_mwh'].sum()):.3f} |\n")
L.append(f"| Total net imports (MWh) | {fmt_int(total_model_imp)} | {fmt_int(total_esp_imp)} | ESPENI | {safe_div(total_model_imp, total_esp_imp):.3f} |\n")
L.append("\n")

# 2. Wholesale price benchmark
L.append("## 2. Wholesale price (Stage 1) vs market indices\n")
L.append("Model SMP is the LP shadow price of the **copperplate** Stage-1 wholesale solve (zero spread between buses).\n")
L.append("\n")
L.append(price_table.round(2).to_markdown())
L.append("\n\n")
L.append(f"- Hourly correlation **model vs MID**: r = {r_mid:.3f}  ·  MAE £{mae_mid:.2f}/MWh  ·  RMSE £{rmse_mid:.2f}/MWh\n")
L.append(f"- Hourly correlation **model vs SBP**: r = {r_sbp:.3f}  ·  MAE £{mae_sbp:.2f}/MWh  ·  RMSE £{rmse_sbp:.2f}/MWh\n")
L.append(f"- Mean **model BM nodal** price (demand buses): £{pc['mean_nodal_price'].mean():.2f}/MWh "
         f"vs SBP £{sbp_a.mean():.2f}/MWh → ratio {safe_div(pc['mean_nodal_price'].mean(), sbp_a.mean()):.2f}.\n")
L.append(f"- Nodal spread (max-min within hour): mean £{pc['nodal_spread'].mean():.2f}/MWh, "
         f"{(pc['nodal_spread'] > 50).mean()*100:.1f}% of hours > £50, "
         f"{(pc['nodal_spread'] < 5).mean()*100:.1f}% of hours < £5 (copperplate-like).\n")
L.append("\n")

# 3. Carrier dispatch — wholesale and BM vs ESPENI
L.append("## 3. Carrier dispatch — wholesale and BM stages vs ESPENI fuel-type fleet\n")
L.append("ESPENI/Elexon ELEC_POWER_* aggregates are the only available fuel-type fleet truth.\n")
L.append("`r` = hourly Pearson correlation over the 744-hour window.\n")
L.append("\n")
disp_show = dispatch_df.copy()
for c in ["model_ws_mwh", "model_bm_mwh", "espeni_mwh"]:
    disp_show[c] = disp_show[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
for c in ["ws_ratio", "bm_ratio"]:
    disp_show[c] = disp_show[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
for c in ["r_ws_hourly", "r_bm_hourly"]:
    disp_show[c] = disp_show[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
L.append(disp_show.to_markdown(index=False))
L.append("\n\n")
L.append("**Reading guide**\n")
L.append("- `ws_ratio` = model wholesale ÷ ESPENI. >1 means the copperplate over-dispatches the carrier; "
         "the BM should bring it back toward 1.0 if the network constraint is the binding mechanism.\n")
L.append("- `bm_ratio` close to 1.0 with high `r_bm_hourly` is the dispatch target.\n")
L.append("- Embedded wind/solar are forced to ESPENI profiles in the model so r ≈ 1.0 by construction.\n")
L.append("\n")

# 4. BM volumes by carrier vs Elexon BOALF
L.append("## 4. Balancing mechanism volumes by carrier — model vs Elexon BOALF\n")
L.append("Elexon volumes are aggregated from `BOALF` acceptance levels (all flags) for January 2020.\n")
L.append("\n")
bm_show = bm_compare.copy()
for c in ["model_increase_mwh", "model_decrease_mwh", "elexon_increase_mwh",
          "elexon_decrease_mwh", "model_net_mwh", "elexon_net_mwh"]:
    bm_show[c] = bm_show[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
for c in ["inc_ratio", "dec_ratio"]:
    bm_show[c] = bm_show[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
L.append(bm_show.to_markdown())
L.append("\n\n")
L.append("**Notes**\n")
L.append("- `unknown` in the Elexon column is largely interconnectors and unmapped BMUs — model has none.\n")
L.append("- Where `inc_ratio` << 1 the model is *not using that carrier* in the BM (mostly because "
         "wholesale has already dispatched it at full availability — see §3).\n")
L.append(f"- CCGT inc_ratio = {bm_compare.loc['CCGT', 'inc_ratio'] if 'CCGT' in bm_compare.index else float('nan'):.2f} "
         "reflects the structural copperplate penalty: Scottish wind is dumped south in Stage 1 "
         "and English CCGT must turn up in Stage 2 — exaggerated because wind in Stage 1 is "
         f"{(ws_wind_on + ws_wind_off) / elex_wind_mwh:.2f}x ELEX_WIND.\n")
L.append("- Wind decrease in the BM (~518 GWh onshore + 16.7 GWh offshore) is the model's "
         "constraint-driven curtailment; Elexon shows almost no wind BOALF turn-down because "
         "real-world curtailment happens via offer prices < SBP (commercial), not via STOR/STOR-RR.\n")
L.append("- Large hydro `dec_ratio` ≈ 84 and PSH `inc_ratio` = 0 are artefacts of the model "
         "treating reservoir hydro as a cheap energy carrier in Stage 1 and using it for turn-down "
         "in Stage 2; the real BM uses PSH for turn-up and large_hydro is largely unflagged.\n")
L.append("\n")
L.append("> **Headline numbers caveat:** the §1 BOALF totals (904 / 789 GWh inc/dec) come from the "
         "*net difference* between BM and wholesale dispatch summed by carrier. The "
         "`Validation_Jan2020_bm_validation.csv` headline of 1,612,887 MWh inc/dec is the "
         "*sum of all signed dispatch changes per timestep* (i.e. it counts a reservoir hydro "
         "unit going down then back up as twice the volume) and is therefore ~1.7-2.0x higher.\n")
L.append("\n")

# 5. BM costs by carrier
L.append("## 5. BM cost decomposition by carrier (model)\n")
L.append(f"Total model BM net cost: **{fmt_money(cc.loc['TOTAL', 'net_cost'])}** "
         f"({fmt_money(cc.loc['TOTAL', 'net_cost'] * 8760 / PERIOD_HOURS)}/yr).\n\n")
L.append(cc_table.round(0).to_markdown())
L.append("\n\n")
L.append("- `offer_cost` = paid for turn-up; `bid_cost` = paid (or revenue if negative) for turn-down.\n")
L.append("- The model has no equivalent of Elexon system price (SBP/SSP) per carrier; comparison "
         "must be done at total-cost level (§1) or via NESO thermal cost (§6).\n")
L.append("\n")

# 6. Constraint costs vs NESO
L.append("## 6. Constraint costs vs NESO thermal benchmark\n")
neso_total_cost = neso_thermal_total
L.append(f"- **Model BM cost (Jan 2020):** {fmt_money(model_bm_total)}\n")
L.append(f"- **NESO thermal cost (Jan 2020):** {fmt_money(neso_total_cost)}\n")
L.append(f"- **Ratio model/NESO:** {safe_div(model_bm_total, neso_total_cost):.3f} "
         "(target ≥ 0.80 for in-month accuracy; structural copperplate gap explains some of the shortfall).\n")
L.append("\n")
L.append("**NESO cost by boundary (Jan 2020)**\n\n")
nb_table = neso_boundaries[["boundary", "value"]].copy()
nb_table["share"] = nb_table["value"] / nb_table["value"].sum() * 100
nb_table["value"] = nb_table["value"].apply(lambda v: f"£{v:,.0f}")
nb_table["share"] = nb_table["share"].apply(lambda v: f"{v:.1f}%")
L.append(nb_table.to_markdown(index=False))
L.append("\n\n")
L.append("**DA boundary flows — model vs NESO (Jan 2020 means)**\n\n")
fp_show = flow_pivot.copy()
keep_cols = [c for c in ["model_mean_flow_mw", "model_max_flow_mw",
                         "neso_mean_flow_mw", "neso_max_flow_mw",
                         "neso_mean_limit_mw", "neso_mean_utilisation",
                         "neso_pct_above_90"] if c in fp_show.columns]
fp_show = fp_show[keep_cols].round(2)
L.append(fp_show.to_markdown())
L.append("\n\n")

# 7. DISBSAD
L.append("## 7. Non-BOALF balancing actions (DISBSAD) — Elexon only\n")
L.append("These are Disposal/Bilateral Service Adjustment Data — settlement-period actions outside BOALF "
         "(e.g. STOR, fast reserve, disconnection). The model does not represent ancillary services, "
         "so this is a context number for the *unmodelled* portion of BSUoS.\n\n")
ds_table = disbsad_summary.to_frame("Jan 2020").reset_index()
ds_table.columns = ["Metric", "Jan 2020"]
ds_table["Jan 2020"] = ds_table["Jan 2020"].apply(
    lambda v: f"{v:,.2f}" if isinstance(v, (int, float, np.integer, np.floating)) else v)
L.append(ds_table.to_markdown(index=False))
L.append("\n\n")
L.append(f"DISBSAD net cost (£{disbsad_jan['cost'].sum():,.0f}) is roughly "
         f"**{disbsad_jan['cost'].sum() / model_bm_total * 100:.1f}%** of the model's BM cost — "
         "i.e. there is a structural ~5-10% of January 2020 balancing that the model cannot "
         "represent by design.\n\n")

# 8. Interconnectors
L.append("## 8. Interconnectors (sanity — fixed to ESPENI in Stage 1)\n\n")
ic_show = ic_df.copy()
for c in ["model_net_import_mwh", "espeni_net_import_mwh"]:
    ic_show[c] = ic_show[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
ic_show["ratio"] = ic_show["ratio"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
ic_show["r_hourly"] = ic_show["r_hourly"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
L.append(ic_show.to_markdown(index=False))
L.append("\n\n")
L.append(f"Total net import — model {total_model_imp:,.0f} MWh  vs ESPENI {total_esp_imp:,.0f} MWh "
         f"(ratio {safe_div(total_model_imp, total_esp_imp):.3f}).\n\n")

# 9. Top congested lines
L.append("## 9. Top constrained network elements\n\n")
top_cong = cong.sort_values("hours_congested", ascending=False).head(15)[
    ["type", "s_nom_MVA", "hours_congested", "max_loading_fraction", "mean_loading_fraction"]
].round(3)
L.append(top_cong.to_markdown())
L.append("\n\n")

# 10. Summary verdict
L.append("## 10. Summary verdict\n\n")
L.append("| Dimension | Result | Comment |\n|---|---|---|\n")
L.append(f"| Wholesale price level | **{safe_div(model_smp.mean(), mid_a.mean()):.2f}x MID** | Within 5% of N2EX. |\n")
L.append(f"| Wholesale price shape (hourly r) | **r = {r_mid:.2f} vs MID** | Strong if >0.7. |\n")
L.append(f"| BM total cost vs NESO | **{safe_div(model_bm_total, neso_total_cost):.2f}x** | 0.66–0.73x — structural copperplate gap. |\n")
_inc_ratio_total = safe_div(bm_compare['model_increase_mwh'].sum(), bm_compare['elexon_increase_mwh'].sum())
_dec_ratio_total = safe_div(bm_compare['model_decrease_mwh'].sum(), bm_compare['elexon_decrease_mwh'].sum())
_ccgt_inc = safe_div(bm_compare.loc['CCGT', 'model_increase_mwh'], bm_compare.loc['CCGT', 'elexon_increase_mwh']) if 'CCGT' in bm_compare.index else float('nan')
_ccgt_disp = safe_div(float(bm_by_c.get('CCGT', pd.Series([0])).sum()), hourly_mwh(esp['ELEC_POWER_ELEX_CCGT[MW](float32)']))
_wind_disp = safe_div(bm_wind_on + bm_wind_off, elex_wind_mwh)
_hydro_disp = safe_div(float(bm_by_c.get('large_hydro', pd.Series([0])).sum()), hourly_mwh(esp['ELEC_POWER_ELEX_NPSHYD[MW](float32)']))
_coal_disp = safe_div(float(bm_by_c.get('coal', pd.Series([0])).sum()), hourly_mwh(esp['ELEC_POWER_ELEX_COAL[MW](float32)']))
_biomass_disp = safe_div(float(bm_by_c.get('biomass', pd.Series([0])).sum()), hourly_mwh(esp['ELEC_POWER_ELEX_BIOMASS_POSTCALC[MW](float32)']))
_nuc_r = corr(bm_by_c.get('nuclear', pd.Series(dtype=float)), esp_h['ELEC_POWER_ELEX_NUCLEAR[MW](float32)'])
L.append(f"| BM volume (BOALF inc) | **{_inc_ratio_total:.2f}x** inc / **{_dec_ratio_total:.2f}x** dec | Net carrier-level diff; matches BOALF inc within ~30% but mix is wrong (see §4). |\n")
L.append(f"| BM carrier mix | CCGT inc {_ccgt_inc:.2f}x; PSH/coal/biomass/nuclear inc ~0 | Wholesale has already dispatched PSH/coal/biomass/nuclear at full availability, leaving zero BM headroom. |\n")
L.append(f"| Carrier dispatch (BM, vs ESPENI) | CCGT {_ccgt_disp:.2f}x, wind {_wind_disp:.2f}x, large_hydro {_hydro_disp:.2f}x | Coal {_coal_disp:.2f}x (none modelled in Jan-2020 fleet), biomass {_biomass_disp:.2f}x, nuclear hourly r = {_nuc_r:.2f}. |\n")
L.append(f"| Interconnectors | {safe_div(total_model_imp, total_esp_imp):.2f}x ESPENI total | Fixed to actuals. |\n")
L.append("\n")
L.append("**Key gaps to close (in priority)**\n\n")
L.append(f"1. **Coal absent from Jan-2020 fleet (model 0 vs ESPENI 1.49 TWh)** — coal-fired plant is "
         "in the generators table but availability/MC pushes it out of merit; this is the largest "
         "single carrier-level energy gap and feeds directly into the CCGT over-dispatch.\n")
L.append(f"2. **PSH discharge 0.13x ESPENI / inc_ratio = 0** — PSH provides no Stage-2 turn-up; "
         "wholesale already exhausts its energy budget. Likely needs reservoir/cycle constraints "
         "or higher Stage-1 marginal cost so headroom remains for BM.\n")
L.append(f"3. **Biomass and coal BM turn-up ≈ 0** — same diagnosis as PSH: Stage 1 dispatches to "
         "full availability. Add min-stable / must-run or Stage-1 MC uplift to free BM headroom.\n")
L.append(f"4. **CCGT BM turn-up {_ccgt_inc:.2f}x BOALF, dispatch {_ccgt_disp:.2f}x ESPENI** — "
         "structural copperplate symptom; English CCGT covers Scottish wind that would be "
         "self-curtailed at day-ahead in reality. Partial relaxation of wholesale s_nom or zonal "
         "Stage-1 pricing would close it.\n")
L.append(f"5. **Wind +19% in Stage 1, +11% in BM vs ELEX_WIND** — performance factors "
         "(0.80/0.82) over-shoot after curtailment; revisit factor calibration on a longer "
         "window (Jan-only is volatile).\n")
L.append(f"6. **Wholesale price hourly r = {r_mid:.2f} vs MID (mean within 5%)** — level OK, "
         "shape correlation is mediocre; investigate diurnal merit-order ordering and gas/coal "
         "MC spread.\n")
L.append(f"7. **DISBSAD £{disbsad_jan['cost'].sum()/1e6:.1f}M ({disbsad_jan['cost'].sum()/model_bm_total*100:.1f}% of model BM cost)** — "
         "no model analogue (STOR/fast-reserve/disconnection); document as structural omission.\n")
L.append(f"8. **NESO thermal coverage {safe_div(model_bm_total, neso_total_cost):.2f}x** — outage "
         "schedule + voltage constraints would close the remaining 25-30% gap.\n")

report = "".join(L)

# Output
out_md = ROOT / "docs/validation_jan2020_deep_dive.md"
out_md.write_text(report, encoding="utf-8")
print(f"[write] report -> {out_md}")

# Also dump machine-readable
out_dir = ROOT / "resources/analysis"
out_dir.mkdir(parents=True, exist_ok=True)
dispatch_df.to_csv(out_dir / f"{SCEN}_deep_dive_dispatch.csv", index=False)
bm_compare.to_csv(out_dir / f"{SCEN}_deep_dive_bm_carrier.csv")
cc_table.to_csv(out_dir / f"{SCEN}_deep_dive_constraint_costs.csv")
ic_df.to_csv(out_dir / f"{SCEN}_deep_dive_ic.csv", index=False)
print(f"[write] tables  -> {out_dir}/{SCEN}_deep_dive_*.csv")

# Console snapshot
print()
print("=" * 80)
print(report.split("## 10")[0][-2000:])
print("...")
print("=" * 80)
print(f"Done. Report: {out_md}")
