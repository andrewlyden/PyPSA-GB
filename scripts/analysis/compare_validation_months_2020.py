"""
Compare Validation_Jan2020 and Validation_Jun2020 against NESO, ELEXON, and ESPENI.

Outputs:
  - resources/analysis/validation_2020_month_comparison_summary.csv
  - resources/analysis/validation_2020_month_comparison_espeni_dispatch.csv
  - resources/analysis/validation_2020_month_comparison_neso_boundaries.csv
  - resources/analysis/validation_2020_month_comparison.md

Run from project root:
    conda run -n pypsa-gb python scripts/analysis/compare_validation_months_2020.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pypsa


ROOT = Path(__file__).resolve().parents[2]
MARKET_DIR = ROOT / "resources" / "market"
ANALYSIS_DIR = ROOT / "resources" / "analysis"
ESPENI_PATH = ROOT / "data" / "demand" / "espeni.csv"
OUT_PREFIX = ANALYSIS_DIR / "validation_2020_month_comparison"

TIME_COL = "ELEC_elex_startTime[utc](datetime)"
ESPENI_MAP = {
    "CCGT": "ELEC_POWER_ELEX_CCGT[MW](float32)",
    "nuclear": "ELEC_POWER_ELEX_NUCLEAR[MW](float32)",
    "coal": "ELEC_POWER_ELEX_COAL[MW](float32)",
    "biomass": "ELEC_POWER_ELEX_BIOMASS_POSTCALC[MW](float32)",
    "OCGT": "ELEC_POWER_ELEX_OCGT[MW](float32)",
    "large_hydro": "ELEC_POWER_ELEX_NPSHYD[MW](float32)",
    "embedded_wind": "ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)",
    "embedded_solar": "ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)",
}
BOUNDARIES = ["SSHARN", "SSE-SP", "SCOTEX", "ESTEX", "SEIMP", "SWALEX"]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario: str
    label: str
    start: str
    end: str


SCENARIOS = [
    ScenarioSpec("Validation_Jan2020", "Jan 2020", "2020-01-01", "2020-02-01"),
    ScenarioSpec("Validation_Jun2020", "Jun 2020", "2020-06-01", "2020-07-01"),
]


def hourly_mwh(series_hh: pd.Series) -> float:
    return float(series_hh.sum() * 0.5)


def corr(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 10:
        return float("nan")
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def parse_metric_value(value) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if text in {"-", "--", "---", "N/A", "n/a", "nan", "NaN", "None", "none", "0"}:
        return 0.0 if text == "0" else float("nan")
    if text in {"-", "—"}:
        return float("nan")
    text = (
        text.replace("£", "")
        .replace("Ł", "")
        .replace(",", "")
        .replace("%", "")
        .replace("x", "")
        .strip()
    )
    if text in {"", "-", "—"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def agg_by_carrier(dispatch: pd.DataFrame, carrier_map: dict[str, str]) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    for generator in dispatch.columns:
        carrier = carrier_map.get(generator, "unknown")
        if carrier not in result:
            result[carrier] = pd.Series(0.0, index=dispatch.index)
        result[carrier] = result[carrier] + dispatch[generator].fillna(0.0)
    return pd.DataFrame(result)


def load_espeni() -> pd.DataFrame:
    df = pd.read_csv(ESPENI_PATH)
    df["time"] = pd.to_datetime(df[TIME_COL], utc=True).dt.tz_convert(None)
    df = df.set_index("time")
    df = df.drop(
        columns=[c for c in df.columns if "datetime" in c.lower() or "time" in c.lower()],
        errors="ignore",
    )
    return df.select_dtypes(include="number").sort_index()


def metric_lookup(df: pd.DataFrame, metric: str, value_col: str) -> float:
    rows = df.loc[df["metric"] == metric, value_col]
    if rows.empty:
        return float("nan")
    return parse_metric_value(rows.iloc[0])


def neso_lookup(df: pd.DataFrame, category: str, metric: str) -> float:
    rows = df.loc[(df["category"] == category) & (df["metric"] == metric), "value"]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0])


def fmt_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def fmt_currency(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"GBP {value / 1e6:,.{decimals}f}m"


def fmt_ratio(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}x"


def pick_top_rows(df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(by, ascending=False).head(n)


def build_dispatch_rows(
    spec: ScenarioSpec,
    espeni: pd.DataFrame,
) -> list[dict]:
    network = pypsa.Network(MARKET_DIR / f"{spec.scenario}_balancing.nc")
    dispatch = pd.read_csv(MARKET_DIR / f"{spec.scenario}_balancing_dispatch.csv", index_col=0, parse_dates=True)
    storage = pd.read_csv(MARKET_DIR / f"{spec.scenario}_balancing_storage.csv", index_col=0, parse_dates=True)

    carrier_dispatch = agg_by_carrier(dispatch, network.generators["carrier"].to_dict())
    esp = espeni[(espeni.index >= spec.start) & (espeni.index < spec.end)]
    esp_h = esp.resample("h").mean()

    rows: list[dict] = []
    for carrier, espeni_col in ESPENI_MAP.items():
        if carrier not in carrier_dispatch.columns or espeni_col not in esp.columns:
            continue
        model_mwh = float(carrier_dispatch[carrier].sum())
        ref_mwh = hourly_mwh(esp[espeni_col].dropna())
        rows.append(
            {
                "scenario": spec.scenario,
                "period": spec.label,
                "comparison": carrier,
                "model_mwh": model_mwh,
                "reference_mwh": ref_mwh,
                "ratio": model_mwh / ref_mwh if ref_mwh else float("nan"),
                "hourly_correlation": corr(carrier_dispatch[carrier], esp_h[espeni_col]),
                "reference_source": "ESPENI",
            }
        )

    wind_model = float(
        carrier_dispatch.get("wind_onshore", pd.Series(0.0, index=carrier_dispatch.index)).sum()
        + carrier_dispatch.get("wind_offshore", pd.Series(0.0, index=carrier_dispatch.index)).sum()
    )
    wind_ref = hourly_mwh(esp["ELEC_POWER_ELEX_WIND[MW](float32)"])
    wind_corr = corr(
        carrier_dispatch.get("wind_onshore", pd.Series(0.0, index=carrier_dispatch.index))
        + carrier_dispatch.get("wind_offshore", pd.Series(0.0, index=carrier_dispatch.index)),
        esp_h["ELEC_POWER_ELEX_WIND[MW](float32)"],
    )
    rows.append(
        {
            "scenario": spec.scenario,
            "period": spec.label,
            "comparison": "wind_onshore+wind_offshore",
            "model_mwh": wind_model,
            "reference_mwh": wind_ref,
            "ratio": wind_model / wind_ref if wind_ref else float("nan"),
            "hourly_correlation": wind_corr,
            "reference_source": "ESPENI_ELEX_WIND",
        }
    )

    solar_model = float(
        carrier_dispatch.get("solar_pv", pd.Series(0.0, index=carrier_dispatch.index)).sum()
        + carrier_dispatch.get("embedded_solar", pd.Series(0.0, index=carrier_dispatch.index)).sum()
    )
    solar_ref = float(esp_h["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"].sum())
    rows.append(
        {
            "scenario": spec.scenario,
            "period": spec.label,
            "comparison": "solar_pv+embedded_solar",
            "model_mwh": solar_model,
            "reference_mwh": solar_ref,
            "ratio": solar_model / solar_ref if solar_ref else float("nan"),
            "hourly_correlation": float("nan"),
            "reference_source": "ESPENI_embedded_solar_only",
        }
    )

    ps_cols = [c for c in storage.columns if any(name in c for name in ["Dinorwig", "Ffestiniog", "Cruachan", "Foyers"])]
    ps_gen = storage[ps_cols].clip(lower=0).sum(axis=1)
    ps_charge = (-storage[ps_cols].clip(upper=0)).sum(axis=1)
    ps_dis_ref = hourly_mwh(esp["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"])
    ps_chg_ref = abs(hourly_mwh(esp["ELEC_POWER_ELEX_PS_CHARGING[MW](float32)"]))

    rows.append(
        {
            "scenario": spec.scenario,
            "period": spec.label,
            "comparison": "pumped_hydro_dispatch",
            "model_mwh": float(ps_gen.sum()),
            "reference_mwh": ps_dis_ref,
            "ratio": float(ps_gen.sum()) / ps_dis_ref if ps_dis_ref else float("nan"),
            "hourly_correlation": corr(ps_gen, esp_h["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"]),
            "reference_source": "ESPENI_PS_DISCHARGING",
        }
    )
    rows.append(
        {
            "scenario": spec.scenario,
            "period": spec.label,
            "comparison": "pumped_hydro_charging",
            "model_mwh": float(ps_charge.sum()),
            "reference_mwh": ps_chg_ref,
            "ratio": float(ps_charge.sum()) / ps_chg_ref if ps_chg_ref else float("nan"),
            "hourly_correlation": float("nan"),
            "reference_source": "ESPENI_PS_CHARGING",
        }
    )

    demand_model = float(network.loads_t.p_set.sum(axis=1).sum())
    demand_ref = float(esp_h["ELEC_POWER_TOTAL_ESPENI[MW](float32)"].sum())
    rows.append(
        {
            "scenario": spec.scenario,
            "period": spec.label,
            "comparison": "total_demand",
            "model_mwh": demand_model,
            "reference_mwh": demand_ref,
            "ratio": demand_model / demand_ref if demand_ref else float("nan"),
            "hourly_correlation": corr(network.loads_t.p_set.sum(axis=1), esp_h["ELEC_POWER_TOTAL_ESPENI[MW](float32)"]),
            "reference_source": "ESPENI_total_demand",
        }
    )
    return rows


def build_neso_boundary_rows(spec: ScenarioSpec) -> list[dict]:
    df = pd.read_csv(MARKET_DIR / f"{spec.scenario}_neso_validation.csv")
    rows: list[dict] = []
    for boundary in BOUNDARIES:
        flow_cat = f"flow_{boundary}"
        cong_cat = f"congestion_{boundary}"
        neso_cost = neso_lookup(df, "neso_boundary_cost", f"{boundary}_gbp")
        model_mean_flow = neso_lookup(df, flow_cat, "model_mean_flow_mw")
        neso_mean_flow = neso_lookup(df, flow_cat, "neso_mean_flow_mw")
        model_hours_above_90 = neso_lookup(df, cong_cat, "model_boundary_hours_above_90")
        neso_hours_above_90 = neso_lookup(df, cong_cat, "neso_hours_above_90")
        rows.append(
            {
                "scenario": spec.scenario,
                "period": spec.label,
                "boundary": boundary,
                "neso_thermal_cost_gbp": neso_cost,
                "model_mean_flow_mw": model_mean_flow,
                "neso_mean_flow_mw": neso_mean_flow,
                "flow_ratio": model_mean_flow / neso_mean_flow if neso_mean_flow else float("nan"),
                "model_hours_above_90": model_hours_above_90,
                "neso_hours_above_90": neso_hours_above_90,
                "hours_above_90_ratio": (
                    model_hours_above_90 / neso_hours_above_90 if neso_hours_above_90 else float("nan")
                ),
            }
        )
    return rows


def build_summary_rows(spec: ScenarioSpec) -> list[dict]:
    bm = pd.read_csv(MARKET_DIR / f"{spec.scenario}_bm_validation.csv")
    neso = pd.read_csv(MARKET_DIR / f"{spec.scenario}_neso_validation.csv")
    disbsad = pd.read_csv(MARKET_DIR / f"{spec.scenario}_disbsad_summary.csv")

    disbsad_total = disbsad.loc[(disbsad["scope"] == "flag_group") & (disbsad["group"] == "all"), "cost_gbp"]
    disbsad_abs = disbsad.loc[(disbsad["scope"] == "flag_group") & (disbsad["group"] == "all"), "abs_volume_mwh"]

    summary = {
        "scenario": spec.scenario,
        "period": spec.label,
        "hours": metric_lookup(bm, "Total BM cost (solve period)", "note"),
        "model_bm_cost_gbp": neso_lookup(neso, "total_cost", "model_bm_cost_gbp"),
        "neso_thermal_cost_gbp": neso_lookup(neso, "total_cost", "neso_thermal_cost_gbp"),
        "model_vs_neso_cost_ratio": neso_lookup(neso, "total_cost", "model_neso_ratio"),
        "model_bm_annualised_gbp": metric_lookup(bm, "Annualised BM cost", "model_value"),
        "elexon_annual_benchmark_gbp": metric_lookup(bm, "Annualised BM cost", "elexon_value"),
        "model_vs_elexon_annual_cost_ratio": metric_lookup(bm, "Annualised BM cost", "ratio"),
        "model_increase_mwh": metric_lookup(bm, "Total increase volume", "model_value"),
        "elexon_increase_mwh": metric_lookup(bm, "Total increase volume", "elexon_value"),
        "model_vs_elexon_increase_ratio": metric_lookup(bm, "Total increase volume", "ratio"),
        "model_decrease_mwh": metric_lookup(bm, "Total decrease volume", "model_value"),
        "elexon_decrease_mwh": metric_lookup(bm, "Total decrease volume", "elexon_value"),
        "model_vs_elexon_decrease_ratio": metric_lookup(bm, "Total decrease volume", "ratio"),
        "elexon_unflagged_increase_mwh": metric_lookup(bm, "Total increase volume vs BOALF (unflagged)", "elexon_value"),
        "elexon_unflagged_decrease_mwh": metric_lookup(bm, "Total decrease volume vs BOALF (unflagged)", "elexon_value"),
        "elexon_flagged_increase_share_pct": metric_lookup(bm, "BOALF flagged share of increase volume", "elexon_value"),
        "elexon_flagged_decrease_share_pct": metric_lookup(bm, "BOALF flagged share of decrease volume", "elexon_value"),
        "elexon_disbsad_abs_mwh": float(disbsad_abs.iloc[0]) if not disbsad_abs.empty else float("nan"),
        "elexon_disbsad_cost_gbp": float(disbsad_total.iloc[0]) if not disbsad_total.empty else float("nan"),
        "bm_constraint_cost_per_hour_gbp": metric_lookup(bm, "BM constraint cost per hour", "model_value"),
        "mean_smp_to_mid_ratio": metric_lookup(bm, "Mean wholesale price (SMP)", "ratio"),
        "mean_smp_to_sbp_ratio": metric_lookup(bm, "Mean wholesale price (SMP) vs SBP", "ratio"),
    }
    return [summary]


def build_markdown(
    summary_df: pd.DataFrame,
    dispatch_df: pd.DataFrame,
    neso_df: pd.DataFrame,
) -> str:
    jan = summary_df.loc[summary_df["scenario"] == "Validation_Jan2020"].iloc[0]
    jun = summary_df.loc[summary_df["scenario"] == "Validation_Jun2020"].iloc[0]

    dispatch_focus = dispatch_df[dispatch_df["comparison"].isin(
        ["CCGT", "nuclear", "coal", "biomass", "large_hydro", "wind_onshore+wind_offshore", "pumped_hydro_dispatch", "total_demand"]
    )].copy()
    dispatch_focus["abs_log_error"] = np.abs(np.log(dispatch_focus["ratio"].replace(0, np.nan)))
    dispatch_focus = dispatch_focus.replace([np.inf, -np.inf], np.nan)

    lines: list[str] = []
    lines.append("# Validation 2020 Month Comparison")
    lines.append("")
    lines.append("Fresh comparison built from the latest `resources/market` validation CSVs for `Validation_Jan2020` and `Validation_Jun2020`.")
    lines.append("")
    lines.append("## 1. Headline comparison")
    lines.append("")
    lines.append("| Metric | Jan 2020 | Jun 2020 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Model BM cost | {fmt_currency(jan['model_bm_cost_gbp'])} | {fmt_currency(jun['model_bm_cost_gbp'])} |")
    lines.append(f"| NESO thermal cost | {fmt_currency(jan['neso_thermal_cost_gbp'])} | {fmt_currency(jun['neso_thermal_cost_gbp'])} |")
    lines.append(f"| Model / NESO thermal | {fmt_ratio(jan['model_vs_neso_cost_ratio'])} | {fmt_ratio(jun['model_vs_neso_cost_ratio'])} |")
    lines.append(f"| Model increase volume | {fmt_number(jan['model_increase_mwh'])} MWh | {fmt_number(jun['model_increase_mwh'])} MWh |")
    lines.append(f"| ELEXON increase volume | {fmt_number(jan['elexon_increase_mwh'])} MWh | {fmt_number(jun['elexon_increase_mwh'])} MWh |")
    lines.append(f"| Model / ELEXON increase | {fmt_ratio(jan['model_vs_elexon_increase_ratio'])} | {fmt_ratio(jun['model_vs_elexon_increase_ratio'])} |")
    lines.append(f"| Model decrease volume | {fmt_number(jan['model_decrease_mwh'])} MWh | {fmt_number(jun['model_decrease_mwh'])} MWh |")
    lines.append(f"| ELEXON decrease volume | {fmt_number(jan['elexon_decrease_mwh'])} MWh | {fmt_number(jun['elexon_decrease_mwh'])} MWh |")
    lines.append(f"| Model / ELEXON decrease | {fmt_ratio(jan['model_vs_elexon_decrease_ratio'])} | {fmt_ratio(jun['model_vs_elexon_decrease_ratio'])} |")
    lines.append(f"| ELEXON DISBSAD cost | {fmt_currency(jan['elexon_disbsad_cost_gbp'])} | {fmt_currency(jun['elexon_disbsad_cost_gbp'])} |")
    lines.append(f"| ELEXON DISBSAD abs volume | {fmt_number(jan['elexon_disbsad_abs_mwh'])} MWh | {fmt_number(jun['elexon_disbsad_abs_mwh'])} MWh |")
    lines.append("")
    lines.append("## 2. ESPENI final dispatch")
    lines.append("")
    lines.append("| Period | Comparison | Model MWh | ESPENI MWh | Ratio | Hourly r |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in dispatch_focus.sort_values(["period", "comparison"]).iterrows():
        lines.append(
            f"| {row['period']} | {row['comparison']} | {fmt_number(row['model_mwh'])} | "
            f"{fmt_number(row['reference_mwh'])} | {row['ratio']:.2f} | "
            f"{row['hourly_correlation']:.2f} |"
        )
    lines.append("")
    lines.append("## 3. NESO boundary comparison")
    lines.append("")
    lines.append("| Period | Boundary | NESO cost | Model mean flow MW | NESO mean flow MW | Flow ratio | Model h >90% | NESO h >90% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for period in [spec.label for spec in SCENARIOS]:
        period_rows = pick_top_rows(
            neso_df.loc[neso_df["period"] == period].copy(),
            "neso_thermal_cost_gbp",
            3,
        )
        for _, row in period_rows.iterrows():
            lines.append(
                f"| {row['period']} | {row['boundary']} | {fmt_currency(row['neso_thermal_cost_gbp'])} | "
                f"{fmt_number(row['model_mean_flow_mw'])} | {fmt_number(row['neso_mean_flow_mw'])} | "
                f"{row['flow_ratio']:.2f} | {fmt_number(row['model_hours_above_90'])} | "
                f"{fmt_number(row['neso_hours_above_90'])} |"
            )
    lines.append("")
    lines.append("## 4. Notes")
    lines.append("")
    lines.append("- NESO comparison uses thermal constraint cost plus day-ahead boundary flow and utilisation metrics from `*_neso_validation.csv`.")
    lines.append("- ELEXON comparison uses BOALF acceptance volumes plus DISBSAD cost and volume from `*_bm_validation.csv` and `*_disbsad_summary.csv`.")
    lines.append("- The current repo outputs do not include a like-for-like period-total ELEXON BOALF cost benchmark, so ELEXON cost is represented here by the annual published BM benchmark and solve-period DISBSAD cost.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    espeni = load_espeni()

    summary_rows: list[dict] = []
    dispatch_rows: list[dict] = []
    neso_rows: list[dict] = []

    for spec in SCENARIOS:
        summary_rows.extend(build_summary_rows(spec))
        dispatch_rows.extend(build_dispatch_rows(spec, espeni))
        neso_rows.extend(build_neso_boundary_rows(spec))

    summary_df = pd.DataFrame(summary_rows)
    dispatch_df = pd.DataFrame(dispatch_rows)
    neso_df = pd.DataFrame(neso_rows)

    summary_df.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.csv"), index=False)
    dispatch_df.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_espeni_dispatch.csv"), index=False)
    neso_df.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_neso_boundaries.csv"), index=False)
    OUT_PREFIX.with_suffix(".md").write_text(build_markdown(summary_df, dispatch_df, neso_df), encoding="utf-8")

    print(f"Wrote {OUT_PREFIX.with_name(OUT_PREFIX.name + '_summary.csv')}")
    print(f"Wrote {OUT_PREFIX.with_name(OUT_PREFIX.name + '_espeni_dispatch.csv')}")
    print(f"Wrote {OUT_PREFIX.with_name(OUT_PREFIX.name + '_neso_boundaries.csv')}")
    print(f"Wrote {OUT_PREFIX.with_suffix('.md')}")


if __name__ == "__main__":
    main()
