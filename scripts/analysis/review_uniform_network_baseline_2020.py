"""Review the 12-month 2020 uniform network baseline validation run.

Writes:
  resources/analysis/uniform_network_baseline_12month_completion_status.csv
  resources/analysis/uniform_network_baseline_12month_scorecard.csv
  resources/analysis/uniform_network_baseline_12month_neso_flows.csv
  resources/analysis/uniform_network_baseline_12month_dispatch_vs_espeni.csv
  resources/analysis/uniform_network_baseline_12month_boav_by_carrier.csv
  resources/analysis/uniform_network_baseline_12month_bm_calibration_by_carrier.csv
  resources/analysis/uniform_network_baseline_12month_review.md
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
}

EXPECTED_SUFFIXES = [
    "_wholesale.nc",
    "_balancing.nc",
    "_balancing_dispatch.csv",
    "_balancing_storage.csv",
    "_bm_validation.csv",
    "_boav_by_carrier.csv",
    "_neso_validation.csv",
    "_constraint_costs.csv",
]


def clean_num(value) -> float:
    if pd.isna(value):
        return math.nan
    text = (
        str(value)
        .replace("GBP", "")
        .replace("£", "")
        .replace("Â£", "")
        .replace("Ã‚Â£", "")
        .replace(",", "")
        .replace("%", "")
        .replace("x", "")
        .replace("—", "")
        .replace("â€”", "")
        .strip()
    )
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def bm_metric(df: pd.DataFrame, name: str, col: str = "model_value") -> float:
    rows = df.loc[df["metric"] == name, col]
    return clean_num(rows.iloc[0]) if not rows.empty else math.nan


def neso_metric(df: pd.DataFrame, category: str, name: str) -> float:
    rows = df.loc[(df["category"] == category) & (df["metric"] == name), "value"]
    return float(rows.iloc[0]) if not rows.empty else math.nan


def corr(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) <= 10:
        return math.nan
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    delta = index.to_series().diff().dropna().median()
    return 1.0 if pd.isna(delta) else float(delta / pd.Timedelta(hours=1))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator and not pd.isna(denominator) else math.nan


def completion_rows() -> list[dict]:
    rows = []
    for month, _, _ in MONTHS:
        scenario = f"Validation_{month}2020_UniformNetworkBaseline"
        missing = [
            f"{scenario}{suffix}"
            for suffix in EXPECTED_SUFFIXES
            if not (MARKET_DIR / f"{scenario}{suffix}").exists()
        ]
        latest = [
            (MARKET_DIR / f"{scenario}{suffix}").stat().st_mtime
            for suffix in EXPECTED_SUFFIXES
            if (MARKET_DIR / f"{scenario}{suffix}").exists()
        ]
        rows.append(
            {
                "scenario": scenario,
                "period": f"{month} 2020",
                "complete": not missing,
                "missing": ";".join(missing),
                "latest_expected_output_epoch": max(latest) if latest else math.nan,
            }
        )
    return rows


def load_espeni() -> pd.DataFrame:
    df = pd.read_csv(ESPENI_PATH)
    df["time"] = pd.to_datetime(df[TIME_COL], utc=True).dt.tz_convert(None)
    return df.set_index("time").select_dtypes(include="number").sort_index()


def aggregate_dispatch_by_carrier(dispatch: pd.DataFrame, carrier_map: dict[str, str]) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    for generator in dispatch.columns:
        carrier = carrier_map.get(generator, "unknown")
        if carrier not in result:
            result[carrier] = pd.Series(0.0, index=dispatch.index)
        result[carrier] = result[carrier].add(dispatch[generator].fillna(0.0), fill_value=0.0)
    return pd.DataFrame(result)


def append_dispatch_rows(
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
    by_carrier = aggregate_dispatch_by_carrier(dispatch, network.generators["carrier"].to_dict())
    dt_hours = timestep_hours(dispatch.index)
    esp = espeni[(espeni.index >= start) & (espeni.index < end)]
    esp_h = esp.resample("h").mean()

    def add_row(comparison: str, model: pd.Series, reference: pd.Series, ref_dt_hours: float) -> None:
        model_mwh = float(model.sum() * dt_hours)
        ref_mwh = float(reference.sum() * ref_dt_hours)
        rows.append(
            {
                "scenario": scenario,
                "period": period,
                "comparison": comparison,
                "model_mwh": model_mwh,
                "espeni_mwh": ref_mwh,
                "ratio": safe_ratio(model_mwh, ref_mwh),
                "hourly_correlation": corr(model, reference.resample("h").mean() if reference.index.freq is None else reference),
            }
        )

    for comparison, col in ESPENI_MAP.items():
        if col not in esp.columns:
            continue
        if comparison == "hydro_non_ps":
            model = (
                by_carrier.get("large_hydro", pd.Series(0.0, index=by_carrier.index))
                + by_carrier.get("small_hydro", pd.Series(0.0, index=by_carrier.index))
            )
        else:
            model = by_carrier.get(comparison, pd.Series(0.0, index=by_carrier.index))
        add_row(comparison, model, esp[col], 0.5)

    wind = (
        by_carrier.get("wind_onshore", pd.Series(0.0, index=by_carrier.index))
        + by_carrier.get("wind_offshore", pd.Series(0.0, index=by_carrier.index))
        + by_carrier.get("embedded_wind", pd.Series(0.0, index=by_carrier.index))
    )
    wind_ref_h = (
        esp_h["ELEC_POWER_ELEX_WIND[MW](float32)"]
        + esp_h["ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)"]
    )
    rows.append(
        {
            "scenario": scenario,
            "period": period,
            "comparison": "wind_total",
            "model_mwh": float(wind.sum() * dt_hours),
            "espeni_mwh": float(wind_ref_h.sum()),
            "ratio": safe_ratio(float(wind.sum() * dt_hours), float(wind_ref_h.sum())),
            "hourly_correlation": corr(wind, wind_ref_h),
        }
    )

    solar = (
        by_carrier.get("solar_pv", pd.Series(0.0, index=by_carrier.index))
        + by_carrier.get("embedded_solar", pd.Series(0.0, index=by_carrier.index))
    )
    solar_ref_h = esp_h["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"]
    rows.append(
        {
            "scenario": scenario,
            "period": period,
            "comparison": "solar_total",
            "model_mwh": float(solar.sum() * dt_hours),
            "espeni_mwh": float(solar_ref_h.sum()),
            "ratio": safe_ratio(float(solar.sum() * dt_hours), float(solar_ref_h.sum())),
            "hourly_correlation": corr(solar, solar_ref_h),
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
        rows.append(
            {
                "scenario": scenario,
                "period": period,
                "comparison": "pumped_hydro_discharge",
                "model_mwh": float(discharge.sum() * dt_hours),
                "espeni_mwh": float(esp["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"].sum() * 0.5),
                "ratio": safe_ratio(
                    float(discharge.sum() * dt_hours),
                    float(esp["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"].sum() * 0.5),
                ),
                "hourly_correlation": corr(
                    discharge,
                    esp_h["ELEC_POWER_ELEX_PS_DISCHARGING[MW](float32)"],
                ),
            }
        )
        charge_ref = abs(float(esp["ELEC_POWER_ELEX_PS_CHARGING[MW](float32)"].sum() * 0.5))
        rows.append(
            {
                "scenario": scenario,
                "period": period,
                "comparison": "pumped_hydro_charge",
                "model_mwh": float(charge.sum() * dt_hours),
                "espeni_mwh": charge_ref,
                "ratio": safe_ratio(float(charge.sum() * dt_hours), charge_ref),
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
            "ratio": safe_ratio(float(demand.sum() * dt_hours), float(demand_ref.sum())),
            "hourly_correlation": corr(demand, demand_ref),
        }
    )


def append_neso_flow_rows(scenario: str, period: str, neso: pd.DataFrame, rows: list[dict]) -> None:
    cost_by_boundary = {
        row["metric"].replace("_gbp", ""): float(row["value"])
        for _, row in neso[neso["category"] == "neso_boundary_cost"].iterrows()
    }
    for category in sorted(neso.loc[neso["category"].str.startswith("flow_", na=False), "category"].unique()):
        boundary = category.removeprefix("flow_")
        base_boundary = boundary.split("::", 1)[0]

        def value(metric: str) -> float:
            return neso_metric(neso, category, metric)

        rows.append(
            {
                "scenario": scenario,
                "period": period,
                "boundary": boundary,
                "neso_cost_gbp": cost_by_boundary.get(boundary, cost_by_boundary.get(base_boundary, 0.0)),
                "model_mean_flow_mw": value("model_mean_flow_mw"),
                "neso_mean_flow_mw": value("neso_mean_flow_mw"),
                "mean_flow_ratio": safe_ratio(value("model_mean_flow_mw"), value("neso_mean_flow_mw")),
                "model_max_flow_mw": value("model_max_flow_mw"),
                "neso_max_flow_mw": value("neso_max_flow_mw"),
                "max_flow_ratio": safe_ratio(value("model_max_flow_mw"), value("neso_max_flow_mw")),
                "neso_mean_limit_mw": value("neso_mean_limit_mw"),
                "neso_pct_above_90": value("neso_pct_above_90"),
                "model_hours_above_90": neso_metric(neso, f"congestion_{boundary}", "model_boundary_hours_above_90"),
                "neso_hours_above_90": neso_metric(neso, f"congestion_{boundary}", "neso_hours_above_90"),
                "model_boundary_mean_loading": neso_metric(neso, f"congestion_{boundary}", "model_boundary_mean_loading"),
            }
        )


def load_shedding_mwh(scenario: str) -> float:
    path = MARKET_DIR / f"{scenario}_constraint_costs.csv"
    if not path.exists():
        return math.nan
    df = pd.read_csv(path)
    row = df.loc[df["carrier"] == "load_shedding"]
    if row.empty:
        return 0.0
    return float(row["increase_MWh"].fillna(0.0).sum() + row["decrease_MWh"].fillna(0.0).sum())


def build_markdown(score: pd.DataFrame, dispatch: pd.DataFrame, flows: pd.DataFrame) -> str:
    totals = {
        "model_bm_cost_gbp": score["bm_cost_gbp"].sum(),
        "neso_breakdown_thermal_cost_gbp": score["neso_breakdown_thermal_cost_gbp"].sum(),
        "model_one_sided_redispatch_mwh": score["model_one_sided_redispatch_mwh"].sum(),
        "neso_thermal_volume_abs_daily_mwh": score["neso_thermal_volume_abs_daily_mwh"].sum(),
        "model_increase_mwh": score["model_increase_mwh"].sum(),
        "elexon_increase_mwh": score["elexon_increase_mwh"].sum(),
        "model_decrease_mwh": score["model_decrease_mwh"].sum(),
        "elexon_decrease_mwh": score["elexon_decrease_mwh"].sum(),
    }
    lines = [
        "# Uniform Network Baseline 2020 Review",
        "",
        "Completed scenarios: Jan-Dec 2020 UniformNetworkBaseline.",
        "",
        "## Annual Totals",
        pd.DataFrame(
            [
                {
                    "metric": "BM cost vs NESO thermal breakdown",
                    "model": totals["model_bm_cost_gbp"],
                    "reference": totals["neso_breakdown_thermal_cost_gbp"],
                    "ratio": safe_ratio(
                        totals["model_bm_cost_gbp"],
                        totals["neso_breakdown_thermal_cost_gbp"],
                    ),
                },
                {
                    "metric": "One-sided thermal volume vs NESO",
                    "model": totals["model_one_sided_redispatch_mwh"],
                    "reference": totals["neso_thermal_volume_abs_daily_mwh"],
                    "ratio": safe_ratio(
                        totals["model_one_sided_redispatch_mwh"],
                        totals["neso_thermal_volume_abs_daily_mwh"],
                    ),
                },
                {
                    "metric": "BOAV increase volume",
                    "model": totals["model_increase_mwh"],
                    "reference": totals["elexon_increase_mwh"],
                    "ratio": safe_ratio(totals["model_increase_mwh"], totals["elexon_increase_mwh"]),
                },
                {
                    "metric": "BOAV decrease volume",
                    "model": totals["model_decrease_mwh"],
                    "reference": totals["elexon_decrease_mwh"],
                    "ratio": safe_ratio(totals["model_decrease_mwh"], totals["elexon_decrease_mwh"]),
                },
            ]
        ).round(3).to_markdown(index=False),
        "",
        "## Monthly Scorecard",
        score[
            [
                "period",
                "bm_cost_gbp",
                "neso_breakdown_thermal_cost_gbp",
                "bm_vs_neso_breakdown_thermal_ratio",
                "model_one_sided_thermal_volume_ratio",
                "increase_ratio",
                "decrease_ratio",
                "smp_mid_ratio",
                "load_shedding_mwh",
            ]
        ].round(3).to_markdown(index=False),
        "",
        "## Final Dispatch Ratios vs ESPENI",
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
    lines.extend(["", "## Largest NESO Boundary-Cost Months"])
    lines.append(
        flows.sort_values("neso_cost_gbp", ascending=False)
        .head(24)
        .round(3)
        .to_markdown(index=False)
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    espeni = load_espeni()
    status = pd.DataFrame(completion_rows())
    score_rows: list[dict] = []
    dispatch_rows: list[dict] = []
    flow_rows: list[dict] = []
    carrier_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []

    for month, start, end in MONTHS:
        scenario = f"Validation_{month}2020_UniformNetworkBaseline"
        bm_path = MARKET_DIR / f"{scenario}_bm_validation.csv"
        neso_path = MARKET_DIR / f"{scenario}_neso_validation.csv"
        boav_path = MARKET_DIR / f"{scenario}_boav_by_carrier.csv"
        calibration_path = MARKET_DIR / f"{scenario}_bm_calibration.csv"
        if not bm_path.exists() or not neso_path.exists():
            continue

        bm = pd.read_csv(bm_path)
        neso = pd.read_csv(neso_path)
        score_rows.append(
            {
                "scenario": scenario,
                "period": f"{month} 2020",
                "bm_cost_gbp": bm_metric(bm, "Total BM cost (solve period)"),
                "neso_thermal_cost_gbp": neso_metric(neso, "total_cost", "neso_thermal_cost_gbp"),
                "neso_breakdown_thermal_cost_gbp": neso_metric(neso, "total_cost", "neso_breakdown_thermal_cost_gbp"),
                "bm_vs_neso_thermal_ratio": neso_metric(neso, "total_cost", "model_neso_ratio"),
                "bm_vs_neso_breakdown_thermal_ratio": neso_metric(neso, "total_cost", "model_neso_breakdown_cost_ratio"),
                "model_one_sided_redispatch_mwh": neso_metric(neso, "thermal_volume", "model_one_sided_redispatch_mwh"),
                "neso_thermal_volume_abs_daily_mwh": neso_metric(neso, "thermal_volume", "neso_thermal_volume_abs_daily_mwh"),
                "model_one_sided_thermal_volume_ratio": neso_metric(
                    neso,
                    "thermal_volume",
                    "model_one_sided_vs_neso_abs_thermal_volume_ratio",
                ),
                "model_gross_thermal_volume_ratio": neso_metric(
                    neso,
                    "thermal_volume",
                    "model_gross_vs_neso_abs_thermal_volume_ratio",
                ),
                "model_increase_mwh": bm_metric(bm, "Total increase volume"),
                "elexon_increase_mwh": bm_metric(bm, "Total increase volume", "elexon_value"),
                "increase_ratio": bm_metric(bm, "Total increase volume", "ratio"),
                "model_decrease_mwh": bm_metric(bm, "Total decrease volume"),
                "elexon_decrease_mwh": bm_metric(bm, "Total decrease volume", "elexon_value"),
                "decrease_ratio": bm_metric(bm, "Total decrease volume", "ratio"),
                "boalf_unflagged_increase_ratio": bm_metric(bm, "Total increase volume vs BOALF (unflagged)", "ratio"),
                "boalf_unflagged_decrease_ratio": bm_metric(bm, "Total decrease volume vs BOALF (unflagged)", "ratio"),
                "disbsad_abs_mwh": bm_metric(bm, "DISBSAD absolute volume", "elexon_value"),
                "disbsad_cost_gbp": bm_metric(bm, "DISBSAD total cost", "elexon_value"),
                "mean_smp_gbp_mwh": bm_metric(bm, "Mean wholesale price (SMP)"),
                "mean_mid_gbp_mwh": bm_metric(bm, "Mean wholesale price (SMP)", "elexon_value"),
                "smp_mid_ratio": bm_metric(bm, "Mean wholesale price (SMP)", "ratio"),
                "load_shedding_mwh": load_shedding_mwh(scenario),
            }
        )
        append_neso_flow_rows(scenario, f"{month} 2020", neso, flow_rows)
        append_dispatch_rows(scenario, f"{month} 2020", start, end, espeni, dispatch_rows)

        if boav_path.exists():
            boav = pd.read_csv(boav_path)
            boav.insert(0, "period", f"{month} 2020")
            boav.insert(0, "scenario", scenario)
            carrier_rows.append(boav)

        if calibration_path.exists():
            calibration = pd.read_csv(calibration_path)
            calibration.insert(0, "period", f"{month} 2020")
            calibration.insert(0, "scenario", scenario)
            calibration_rows.append(calibration)

    score = pd.DataFrame(score_rows)
    dispatch = pd.DataFrame(dispatch_rows)
    flows = pd.DataFrame(flow_rows)
    carriers = pd.concat(carrier_rows, ignore_index=True) if carrier_rows else pd.DataFrame()
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    status.to_csv(ANALYSIS_DIR / "uniform_network_baseline_12month_completion_status.csv", index=False)
    score.to_csv(ANALYSIS_DIR / "uniform_network_baseline_12month_scorecard.csv", index=False)
    flows.to_csv(ANALYSIS_DIR / "uniform_network_baseline_12month_neso_flows.csv", index=False)
    dispatch.to_csv(ANALYSIS_DIR / "uniform_network_baseline_12month_dispatch_vs_espeni.csv", index=False)
    carriers.to_csv(ANALYSIS_DIR / "uniform_network_baseline_12month_boav_by_carrier.csv", index=False)
    calibration.to_csv(ANALYSIS_DIR / "uniform_network_baseline_12month_bm_calibration_by_carrier.csv", index=False)
    (ANALYSIS_DIR / "uniform_network_baseline_12month_review.md").write_text(
        build_markdown(score, dispatch, flows),
        encoding="utf-8",
    )

    print("Wrote uniform-network-baseline 12-month review artefacts.")


if __name__ == "__main__":
    main()
