"""Review completed 2020 validation months against NESO, ELEXON, and ESPENI.

Writes:
  resources/analysis/validation_2020_completed_months_scorecard.csv
  resources/analysis/validation_2020_completed_months_dispatch_vs_espeni.csv
  resources/analysis/validation_2020_completed_months_neso_boundaries.csv
  resources/analysis/validation_2020_completed_months_review.md
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pypsa


ROOT = Path(__file__).resolve().parents[2]
MARKET_DIR = ROOT / "resources" / "market"
ANALYSIS_DIR = ROOT / "resources" / "analysis"
ESPENI_PATH = ROOT / "data" / "demand" / "espeni.csv"
TIME_COL = "ELEC_elex_startTime[utc](datetime)"

MONTHS = [
    ("Jan", "2020-01-01", "2020-02-01"),
    ("Feb", "2020-02-01", "2020-03-01"),
    ("Mar", "2020-03-01", "2020-04-01"),
    ("Apr", "2020-04-01", "2020-05-01"),
    ("May", "2020-05-01", "2020-06-01"),
    ("Jun", "2020-06-01", "2020-07-01"),
    ("Jul", "2020-07-01", "2020-08-01"),
    ("Aug", "2020-08-01", "2020-09-01"),
    ("Sep", "2020-09-01", "2020-10-01"),
    ("Oct", "2020-10-01", "2020-11-01"),
    ("Nov", "2020-11-01", "2020-12-01"),
    ("Dec", "2020-12-01", "2021-01-01"),
]

ESPENI_MAP = {
    "CCGT": "ELEC_POWER_ELEX_CCGT[MW](float32)",
    "coal": "ELEC_POWER_ELEX_COAL[MW](float32)",
    "nuclear": "ELEC_POWER_ELEX_NUCLEAR[MW](float32)",
    "biomass": "ELEC_POWER_ELEX_BIOMASS_POSTCALC[MW](float32)",
    "OCGT": "ELEC_POWER_ELEX_OCGT[MW](float32)",
    "hydro_non_ps": "ELEC_POWER_ELEX_NPSHYD[MW](float32)",
    "embedded_wind": "ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)",
    "embedded_solar": "ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)",
}


def clean_num(value) -> float:
    if pd.isna(value):
        return math.nan
    text = (
        str(value)
        .replace("GBP", "")
        .replace("£", "")
        .replace("Â£", "")
        .replace(",", "")
        .replace("%", "")
        .replace("x", "")
        .replace("â€”", "")
        .replace("—", "")
        .strip()
    )
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def metric(df: pd.DataFrame, name: str, col: str = "model_value") -> float:
    rows = df.loc[df["metric"] == name, col]
    return clean_num(rows.iloc[0]) if not rows.empty else math.nan


def neso_metric(df: pd.DataFrame, category: str, name: str) -> float:
    rows = df.loc[(df["category"] == category) & (df["metric"] == name), "value"]
    return float(rows.iloc[0]) if not rows.empty else math.nan


def neso_boundary_metric(df: pd.DataFrame, prefix: str, boundary: str, name: str) -> float:
    category = f"{prefix}_{boundary}"
    rows = df.loc[(df["category"] == category) & (df["metric"] == name), "value"]
    if rows.empty:
        rows = df.loc[
            df["category"].str.startswith(f"{category}::", na=False)
            & (df["metric"] == name),
            "value",
        ]
    return float(rows.iloc[0]) if not rows.empty else math.nan


def corr(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) <= 10:
        return math.nan
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def timestep_hours(index: pd.DatetimeIndex) -> float:
    """Infer the fixed snapshot duration in hours for energy totals."""
    if len(index) < 2:
        return 1.0
    delta = index.to_series().diff().dropna().median()
    if pd.isna(delta):
        return 1.0
    return float(delta / pd.Timedelta(hours=1))


def agg_by_carrier(dispatch: pd.DataFrame, carrier_map: dict[str, str]) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    for generator in dispatch.columns:
        carrier = carrier_map.get(generator, "unknown")
        if carrier not in result:
            result[carrier] = pd.Series(0.0, index=dispatch.index)
        result[carrier] = result[carrier].add(dispatch[generator].fillna(0.0), fill_value=0.0)
    return pd.DataFrame(result)


def load_espeni() -> pd.DataFrame:
    df = pd.read_csv(ESPENI_PATH)
    df["time"] = pd.to_datetime(df[TIME_COL], utc=True).dt.tz_convert(None)
    return df.set_index("time").select_dtypes(include="number").sort_index()


def add_dispatch_rows(
    scenario: str,
    period: str,
    start: str,
    end: str,
    espeni: pd.DataFrame,
    rows: list[dict],
) -> None:
    network = pypsa.Network(MARKET_DIR / f"{scenario}_balancing.nc")
    dispatch = pd.read_csv(
        MARKET_DIR / f"{scenario}_balancing_dispatch.csv",
        index_col=0,
        parse_dates=True,
    )
    storage = pd.read_csv(
        MARKET_DIR / f"{scenario}_balancing_storage.csv",
        index_col=0,
        parse_dates=True,
    )
    by_carrier = agg_by_carrier(dispatch, network.generators["carrier"].to_dict())
    dt_hours = timestep_hours(dispatch.index)
    esp = espeni[(espeni.index >= start) & (espeni.index < end)]
    esp_h = esp.resample("h").mean()

    for comparison, col in ESPENI_MAP.items():
        if comparison not in by_carrier.columns or col not in esp.columns:
            continue
        model = float(by_carrier[comparison].sum() * dt_hours)
        reference = float(esp[col].sum() * 0.5)
        rows.append(
            {
                "scenario": scenario,
                "period": period,
                "comparison": comparison,
                "model_mwh": model,
                "espeni_mwh": reference,
                "ratio": model / reference if reference else math.nan,
                "hourly_correlation": corr(by_carrier[comparison], esp_h[col]),
            }
        )

    wind = (
        by_carrier.get("wind_onshore", pd.Series(0.0, index=by_carrier.index))
        + by_carrier.get("wind_offshore", pd.Series(0.0, index=by_carrier.index))
        + by_carrier.get("embedded_wind", pd.Series(0.0, index=by_carrier.index))
    )
    wind_ref_h = (
        esp_h["ELEC_POWER_ELEX_WIND[MW](float32)"]
        + esp_h["ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)"]
    )
    wind_ref = float(
        (
            esp["ELEC_POWER_ELEX_WIND[MW](float32)"].sum()
            + esp["ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)"].sum()
        )
        * 0.5
    )
    rows.append(
        {
            "scenario": scenario,
            "period": period,
            "comparison": "wind_total",
            "model_mwh": float(wind.sum() * dt_hours),
            "espeni_mwh": wind_ref,
            "ratio": float(wind.sum() * dt_hours) / wind_ref if wind_ref else math.nan,
            "hourly_correlation": corr(wind, wind_ref_h),
        }
    )

    hydro = (
        by_carrier.get("large_hydro", pd.Series(0.0, index=by_carrier.index))
        + by_carrier.get("small_hydro", pd.Series(0.0, index=by_carrier.index))
    )
    hydro_ref = float(esp["ELEC_POWER_ELEX_NPSHYD[MW](float32)"].sum() * 0.5)
    rows.append(
        {
            "scenario": scenario,
            "period": period,
            "comparison": "hydro_non_ps",
            "model_mwh": float(hydro.sum() * dt_hours),
            "espeni_mwh": hydro_ref,
            "ratio": float(hydro.sum() * dt_hours) / hydro_ref if hydro_ref else math.nan,
            "hourly_correlation": corr(hydro, esp_h["ELEC_POWER_ELEX_NPSHYD[MW](float32)"]),
        }
    )

    solar = (
        by_carrier.get("solar_pv", pd.Series(0.0, index=by_carrier.index))
        + by_carrier.get("embedded_solar", pd.Series(0.0, index=by_carrier.index))
    )
    solar_ref = float(esp_h["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"].sum())
    rows.append(
        {
            "scenario": scenario,
            "period": period,
            "comparison": "solar_total",
            "model_mwh": float(solar.sum() * dt_hours),
            "espeni_mwh": solar_ref,
            "ratio": float(solar.sum() * dt_hours) / solar_ref if solar_ref else math.nan,
            "hourly_correlation": corr(
                solar,
                esp_h["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"],
            ),
        }
    )

    ps_cols = [
        col
        for col in storage.columns
        if any(name in col for name in ["Dinorwig", "Ffestiniog", "Cruachan", "Foyers"])
    ]
    if ps_cols:
        ps = storage[ps_cols]
        discharge = ps.clip(lower=0).sum(axis=1)
        charge = (-ps.clip(upper=0)).sum(axis=1)
        discharge_ref = float(esp["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"].sum() * 0.5)
        charge_ref = abs(float(esp["ELEC_POWER_ELEX_PS_CHARGING[MW](float32)"].sum() * 0.5))
        rows.append(
            {
                "scenario": scenario,
                "period": period,
                "comparison": "pumped_hydro_discharge",
                "model_mwh": float(discharge.sum() * dt_hours),
                "espeni_mwh": discharge_ref,
                "ratio": float(discharge.sum() * dt_hours) / discharge_ref if discharge_ref else math.nan,
                "hourly_correlation": corr(
                    discharge,
                    esp_h["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"],
                ),
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "period": period,
                "comparison": "pumped_hydro_charge",
                "model_mwh": float(charge.sum() * dt_hours),
                "espeni_mwh": charge_ref,
                "ratio": float(charge.sum() * dt_hours) / charge_ref if charge_ref else math.nan,
                "hourly_correlation": math.nan,
            }
        )

    demand = network.loads_t.p_set.sum(axis=1)
    demand_ref = esp_h["ELEC_POWER_TOTAL_ESPENI[MW](float32)"]
    rows.append(
        {
            "scenario": scenario,
            "period": period,
            "comparison": "model_demand_profile",
            "model_mwh": float(demand.sum() * dt_hours),
            "espeni_mwh": float(demand_ref.sum()),
            "ratio": float(demand.sum() * dt_hours) / float(demand_ref.sum()) if float(demand_ref.sum()) else math.nan,
            "hourly_correlation": corr(demand, demand_ref),
        }
    )


def build_markdown(score: pd.DataFrame, dispatch: pd.DataFrame, boundaries: pd.DataFrame) -> str:
    lines = [
        "# Completed 2020 Validation Review",
        "",
        "Completed months included: Jan-Dec 2020.",
        "",
        "## Headline Scorecard",
        score[
            [
                "period",
                "bm_cost_gbp",
                "neso_thermal_cost_gbp",
                "bm_vs_neso_thermal_ratio",
                "bm_vs_neso_breakdown_thermal_ratio",
                "model_gross_thermal_volume_ratio",
                "model_one_sided_thermal_volume_ratio",
                "mean_smp_gbp_mwh",
                "mean_mid_gbp_mwh",
                "smp_mid_ratio",
                "increase_ratio",
                "decrease_ratio",
                "annualised_vs_elexon",
            ]
        ]
        .round(3)
        .to_markdown(index=False),
        "",
        "## Dispatch Ratios vs ESPENI",
    ]
    key = [
        "CCGT",
        "coal",
        "nuclear",
        "biomass",
        "hydro_non_ps",
        "wind_total",
        "solar_total",
        "pumped_hydro_discharge",
        "model_demand_profile",
    ]
    lines.append(
        dispatch[dispatch["comparison"].isin(key)]
        .pivot(index="period", columns="comparison", values="ratio")
        .round(3)
        .to_markdown()
    )
    lines.extend(["", "## Top NESO Boundary Costs"])
    lines.append(
        boundaries.sort_values("neso_cost_gbp", ascending=False)
        .head(20)
        .round(3)
        .to_markdown(index=False)
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    espeni = load_espeni()
    score_rows: list[dict] = []
    dispatch_rows: list[dict] = []
    boundary_rows: list[dict] = []

    for month, start, end in MONTHS:
        scenario = f"Validation_{month}2020"
        bm_path = MARKET_DIR / f"{scenario}_bm_validation.csv"
        neso_path = MARKET_DIR / f"{scenario}_neso_validation.csv"
        if not bm_path.exists() or not neso_path.exists():
            continue

        bm = pd.read_csv(bm_path)
        neso = pd.read_csv(neso_path)
        score_rows.append(
            {
                "scenario": scenario,
                "period": f"{month} 2020",
                "bm_cost_gbp": metric(bm, "Total BM cost (solve period)"),
                "neso_thermal_cost_gbp": neso_metric(neso, "total_cost", "neso_thermal_cost_gbp"),
                "neso_breakdown_thermal_cost_gbp": neso_metric(
                    neso,
                    "total_cost",
                    "neso_breakdown_thermal_cost_gbp",
                ),
                "bm_vs_neso_thermal_ratio": neso_metric(neso, "total_cost", "model_neso_ratio"),
                "bm_vs_neso_breakdown_thermal_ratio": neso_metric(
                    neso,
                    "total_cost",
                    "model_neso_breakdown_cost_ratio",
                ),
                "neso_thermal_volume_abs_daily_mwh": neso_metric(
                    neso,
                    "thermal_volume",
                    "neso_thermal_volume_abs_daily_mwh",
                ),
                "model_gross_thermal_volume_ratio": neso_metric(
                    neso,
                    "thermal_volume",
                    "model_gross_vs_neso_abs_thermal_volume_ratio",
                ),
                "model_one_sided_thermal_volume_ratio": neso_metric(
                    neso,
                    "thermal_volume",
                    "model_one_sided_vs_neso_abs_thermal_volume_ratio",
                ),
                "annualised_vs_elexon": metric(bm, "Annualised BM cost", "ratio"),
                "mean_smp_gbp_mwh": metric(bm, "Mean wholesale price (SMP)"),
                "mean_mid_gbp_mwh": metric(bm, "Mean wholesale price (SMP)", "elexon_value"),
                "smp_mid_ratio": metric(bm, "Mean wholesale price (SMP)", "ratio"),
                "model_increase_mwh": metric(bm, "Total increase volume"),
                "elexon_increase_mwh": metric(bm, "Total increase volume", "elexon_value"),
                "increase_ratio": metric(bm, "Total increase volume", "ratio"),
                "model_decrease_mwh": metric(bm, "Total decrease volume"),
                "elexon_decrease_mwh": metric(bm, "Total decrease volume", "elexon_value"),
                "decrease_ratio": metric(bm, "Total decrease volume", "ratio"),
                "disbsad_cost_gbp": metric(bm, "DISBSAD total cost", "elexon_value"),
                "disbsad_abs_mwh": metric(bm, "DISBSAD absolute volume", "elexon_value"),
            }
        )

        for _, cost_row in neso[neso["category"] == "neso_boundary_cost"].iterrows():
            boundary = cost_row["metric"].replace("_gbp", "")
            neso_cost = float(cost_row["value"])
            if neso_cost <= 0:
                continue
            model_flow = neso_boundary_metric(neso, "flow", boundary, "model_mean_flow_mw")
            neso_flow = neso_boundary_metric(neso, "flow", boundary, "neso_mean_flow_mw")
            boundary_rows.append(
                {
                    "scenario": scenario,
                    "period": f"{month} 2020",
                    "boundary": boundary,
                    "neso_cost_gbp": neso_cost,
                    "model_mean_flow_mw": model_flow,
                    "neso_mean_flow_mw": neso_flow,
                    "flow_ratio": model_flow / neso_flow if neso_flow else math.nan,
                    "model_hours_above_90": neso_boundary_metric(
                        neso,
                        "congestion",
                        boundary,
                        "model_boundary_hours_above_90",
                    ),
                    "neso_hours_above_90": neso_boundary_metric(
                        neso,
                        "congestion",
                        boundary,
                        "neso_hours_above_90",
                    ),
                }
            )

        add_dispatch_rows(scenario, f"{month} 2020", start, end, espeni, dispatch_rows)

    score = pd.DataFrame(score_rows)
    dispatch = pd.DataFrame(dispatch_rows)
    boundaries = pd.DataFrame(boundary_rows)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    score.to_csv(ANALYSIS_DIR / "validation_2020_completed_months_scorecard.csv", index=False)
    dispatch.to_csv(ANALYSIS_DIR / "validation_2020_completed_months_dispatch_vs_espeni.csv", index=False)
    boundaries.to_csv(ANALYSIS_DIR / "validation_2020_completed_months_neso_boundaries.csv", index=False)
    (ANALYSIS_DIR / "validation_2020_completed_months_review.md").write_text(
        build_markdown(score, dispatch, boundaries),
        encoding="utf-8",
    )

    print("Wrote completed-month review artefacts.")


if __name__ == "__main__":
    main()
