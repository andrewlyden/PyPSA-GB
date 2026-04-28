"""
Deep dive on Validation_Jun2020 to explain why NESO June constraints do not
translate into similar model BM costs.

Outputs:
  - resources/analysis/Validation_Jun2020_deep_dive.md
  - resources/analysis/Validation_Jun2020_deep_dive_boundary_summary.csv
  - resources/analysis/Validation_Jun2020_deep_dive_carrier_delta.csv
  - resources/analysis/Validation_Jun2020_deep_dive_asset_delta.csv

Run from project root:
    conda run -n pypsa-gb python scripts/analysis/deep_dive_validation_jun2020.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.market.validate_neso_constraints import _compute_model_boundary_flows

SCEN = "Validation_Jun2020"
MARKET = ROOT / "resources" / "market"
ANALYSIS = ROOT / "resources" / "analysis"
MAPPING = ROOT / "data" / "network" / "neso_boundary_mapping.yaml"
NESO_DA = ROOT / "resources" / "neso_cache" / "day_ahead_constraint_flows_limits.csv"
BM_LOG = ROOT / "logs" / "market" / f"solve_balancing_{SCEN}.log"


class _SilentLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def fmt_num(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def fmt_gbp(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"GBP {value / 1e6:,.2f}m"


def agg_by_carrier(dispatch: pd.DataFrame, carrier_map: dict[str, str]) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for col in dispatch.columns:
        carrier = carrier_map.get(col, "unknown")
        if carrier not in out:
            out[carrier] = pd.Series(0.0, index=dispatch.index)
        out[carrier] = out[carrier] + dispatch[col].fillna(0.0)
    return pd.DataFrame(out)


def load_neso_da() -> pd.DataFrame:
    df = pd.read_csv(NESO_DA, low_memory=False)
    date_col = [c for c in df.columns if "Date" in c][0]
    df = df.rename(
        columns={
            "Constraint Group": "constraint_group",
            date_col: "datetime",
            "Limit (MW)": "limit_mw",
            "Flow (MW)": "flow_mw",
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True).dt.tz_localize(None)
    return df.loc[
        (df["datetime"] >= "2020-06-01 00:00:00")
        & (df["datetime"] <= "2020-06-30 23:00:00")
    ].copy()


def parse_log_metrics() -> dict[str, float]:
    text = BM_LOG.read_text(encoding="utf-8")
    out: dict[str, float] = {}

    m = re.search(
        r"ELEXON bid/offer: (\d+)/(\d+) generators matched .*?, (\d+)/\d+ filled via .*?\(([\d.]+)%\)",
        text,
    )
    if m:
        out["matched_generators"] = float(m.group(1))
        out["total_generators"] = float(m.group(2))
        out["fallback_generators"] = float(m.group(3))
        out["fallback_pct"] = float(m.group(4))

    m = re.search(r"Rolling BM complete: 720 windows, total BM cost .([0-9,]+\.[0-9]+)", text)
    if m:
        out["rolling_bm_cost_gbp"] = float(m.group(1).replace(",", ""))

    prepared = re.findall(r"aggregate boundary constraint prepared with .*", text)
    out["prepared_boundary_constraints"] = float(len(prepared))
    return out


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    with open(MAPPING, "r", encoding="utf-8") as f:
        boundary_mapping = yaml.safe_load(f)["boundaries"]

    n_bm = pypsa.Network(MARKET / f"{SCEN}_balancing.nc")
    n_ws = pypsa.Network(MARKET / f"{SCEN}_wholesale.nc")
    bm_dispatch = pd.read_csv(MARKET / f"{SCEN}_balancing_dispatch.csv", index_col=0, parse_dates=True)
    ws_dispatch = pd.read_csv(MARKET / f"{SCEN}_wholesale_dispatch.csv", index_col=0, parse_dates=True)
    redispatch = pd.read_csv(MARKET / f"{SCEN}_redispatch_summary.csv")
    constraint_costs = pd.read_csv(MARKET / f"{SCEN}_constraint_costs.csv")
    neso_validation = pd.read_csv(MARKET / f"{SCEN}_neso_validation.csv")
    price_comparison = pd.read_csv(MARKET / f"{SCEN}_price_comparison.csv", index_col=0, parse_dates=True)
    neso_da = load_neso_da()
    log_metrics = parse_log_metrics()

    bm_flows = _compute_model_boundary_flows(n_bm, boundary_mapping, _SilentLogger())
    ws_flows = _compute_model_boundary_flows(n_ws, boundary_mapping, _SilentLogger())

    boundary_rows: list[dict] = []
    for boundary in sorted(boundary_mapping):
        neso_b = neso_da.loc[neso_da["constraint_group"] == boundary].copy()
        if neso_b.empty:
            continue
        neso_b = neso_b.set_index("datetime").sort_index()
        limit = neso_b["limit_mw"].reindex(bm_flows.index).interpolate().ffill().bfill()
        neso_flow = neso_b["flow_mw"].reindex(bm_flows.index).interpolate().ffill().bfill()
        ws = ws_flows[boundary]
        bm = bm_flows[boundary]
        flow_rows = neso_validation.loc[
            (neso_validation["category"] == "neso_boundary_cost")
            & (neso_validation["metric"] == f"{boundary}_gbp"),
            "value",
        ]
        neso_cost = float(flow_rows.iloc[0]) if not flow_rows.empty else float("nan")
        boundary_rows.append(
            {
                "boundary": boundary,
                "neso_thermal_cost_gbp": neso_cost,
                "ws_mean_flow_mw": float(ws.mean()),
                "ws_max_flow_mw": float(ws.max()),
                "bm_mean_flow_mw": float(bm.mean()),
                "bm_max_flow_mw": float(bm.max()),
                "neso_mean_flow_mw": float(neso_flow.mean()),
                "neso_max_flow_mw": float(neso_flow.max()),
                "limit_mean_mw": float(limit.mean()),
                "limit_min_mw": float(limit.min()),
                "limit_max_mw": float(limit.max()),
                "ws_hours_above_limit": int((ws > limit + 1e-6).sum()),
                "bm_hours_above_limit": int((bm > limit + 1e-6).sum()),
                "bm_hours_at_95pct_limit": int((bm >= 0.95 * limit).sum()),
                "bm_hours_at_99pct_limit": int((bm >= 0.99 * limit).sum()),
                "mean_ws_to_bm_reduction_mw": float((ws - bm).mean()),
                "max_ws_to_bm_reduction_mw": float((ws - bm).max()),
            }
        )

    boundary_df = pd.DataFrame(boundary_rows).sort_values("neso_thermal_cost_gbp", ascending=False)

    main_boundary = str(boundary_df.iloc[0]["boundary"])
    main_limit = (
        neso_da.loc[neso_da["constraint_group"] == main_boundary]
        .set_index("datetime")
        .sort_index()["limit_mw"]
        .reindex(bm_flows.index)
        .interpolate()
        .ffill()
        .bfill()
    )
    main_bind = bm_flows[main_boundary] >= 0.95 * main_limit

    delta_dispatch = bm_dispatch - ws_dispatch.reindex(index=bm_dispatch.index, columns=bm_dispatch.columns).fillna(0.0)
    carrier_map = n_bm.generators["carrier"].to_dict()
    ws_by_carrier = agg_by_carrier(ws_dispatch, carrier_map)
    bm_by_carrier = agg_by_carrier(bm_dispatch, carrier_map)

    carrier_rows: list[dict] = []
    all_carriers = sorted(set(ws_by_carrier.columns).union(bm_by_carrier.columns))
    for carrier in all_carriers:
        ws_series = ws_by_carrier.get(carrier, pd.Series(0.0, index=bm_dispatch.index))
        bm_series = bm_by_carrier.get(carrier, pd.Series(0.0, index=bm_dispatch.index))
        delta = bm_series - ws_series
        carrier_rows.append(
            {
                "carrier": carrier,
                "bm_minus_ws_mwh_full_month": float(delta.sum()),
                "bm_minus_ws_mwh_bind_hours": float(delta.loc[main_bind].sum()),
                "increase_mwh_bind_hours": float(delta.loc[main_bind].clip(lower=0.0).sum()),
                "decrease_mwh_bind_hours": float((-delta.loc[main_bind].clip(upper=0.0)).sum()),
                "bm_energy_mwh_full_month": float(bm_series.sum()),
                "ws_energy_mwh_full_month": float(ws_series.sum()),
                "p_nom_mw": float(n_bm.generators.loc[n_bm.generators["carrier"] == carrier, "p_nom"].sum()),
            }
        )
    carrier_df = pd.DataFrame(carrier_rows).sort_values("bm_minus_ws_mwh_bind_hours", ascending=False)

    bind_delta = delta_dispatch.loc[main_bind]
    inc = bind_delta.clip(lower=0.0).sum().sort_values(ascending=False).head(20)
    dec = (-bind_delta.clip(upper=0.0)).sum().sort_values(ascending=False).head(20)

    asset_rows: list[dict] = []
    for direction, series in [("increase", inc), ("decrease", dec)]:
        for name, mwh in series.items():
            asset_rows.append(
                {
                    "direction": direction,
                    "component": name,
                    "carrier": carrier_map.get(name, "unknown"),
                    "bind_hours_mwh": float(mwh),
                    "bus": n_bm.generators.loc[name, "bus"] if name in n_bm.generators.index else "",
                }
            )
    asset_df = pd.DataFrame(asset_rows)

    constraint_costs = constraint_costs.copy()
    positive_costs = constraint_costs.loc[constraint_costs["carrier"] != "TOTAL"].sort_values("net_cost", ascending=False)
    negative_costs = constraint_costs.loc[constraint_costs["carrier"] != "TOTAL"].sort_values("net_cost", ascending=True)

    total_row = constraint_costs.loc[constraint_costs["carrier"] == "TOTAL"].iloc[0]
    offer_cost = float(total_row["offer_cost"])
    bid_cost = float(total_row["bid_cost"])
    net_cost = float(total_row["net_cost"])

    md: list[str] = []
    md.append("# Validation_Jun2020 Deep Dive")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- The June 2020 BM is **not** missing NESO boundary enforcement. The balancing log prepared `{int(log_metrics.get('prepared_boundary_constraints', 0))}` aggregate NESO boundary constraints and the solved flows sit on those limits for long stretches.")
    md.append(f"- The dominant real June constraint is `{main_boundary}` at `{fmt_gbp(boundary_df.iloc[0]['neso_thermal_cost_gbp'])}`, about `{boundary_df.iloc[0]['neso_thermal_cost_gbp'] / boundary_df['neso_thermal_cost_gbp'].sum():.0%}` of total NESO thermal cost.")
    md.append(f"- The model BM cost stays tiny at `{fmt_gbp(net_cost)}` because `GBP {offer_cost/1e6:,.2f}m` of turn-up cost is almost entirely cancelled by `GBP {bid_cost/1e6:,.2f}m` of turn-down credits.")
    md.append("")
    md.append("## What The Boundaries Show")
    md.append("")
    md.append("| Boundary | NESO cost | WS > limit h | BM > limit h | BM >=95% h | WS mean MW | BM mean MW | Limit mean MW |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in boundary_df.iterrows():
        md.append(
            f"| {row['boundary']} | {fmt_gbp(row['neso_thermal_cost_gbp'])} | "
            f"{int(row['ws_hours_above_limit'])} | {int(row['bm_hours_above_limit'])} | "
            f"{int(row['bm_hours_at_95pct_limit'])} | {fmt_num(row['ws_mean_flow_mw'])} | "
            f"{fmt_num(row['bm_mean_flow_mw'])} | {fmt_num(row['limit_mean_mw'])} |"
        )
    md.append("")
    md.append(f"- `{main_boundary}` is the real June bottleneck. The wholesale solve breaches its published limit for `{int(boundary_df.iloc[0]['ws_hours_above_limit'])}` hours, and the BM pushes it back to the cap with `{int(boundary_df.iloc[0]['bm_hours_at_95pct_limit'])}` hours at or above 95% of the limit.")
    md.append("- `congestion.csv` showing almost no line congestion is not evidence that the boundary is inactive. The current formulation uses **aggregate boundary constraints**, so the cut can bind while no individual line reaches 95% thermal loading.")
    md.append("")
    md.append("## Why The Cost Collapses")
    md.append("")
    md.append(f"- The net BM cost is `{fmt_gbp(net_cost)}` from `offer_cost - bid_cost = {fmt_gbp(offer_cost)} + ({fmt_gbp(bid_cost)})`.")
    md.append(f"- Direct ELEXON pricing coverage is only `{fmt_num(log_metrics.get('matched_generators', np.nan))}` of `{fmt_num(log_metrics.get('total_generators', np.nan))}` generators, with `{log_metrics.get('fallback_pct', np.nan):.1f}%` fallback pricing. June BM economics are therefore dominated by fallback assumptions rather than observed BMU prices.")
    md.append("- Several turn-down carriers create large **negative** bid costs, so relieving a constraint often earns the model money instead of costing it.")
    md.append("")
    md.append("| Biggest positive net-cost carriers | Net cost | Increase MWh | Decrease MWh |")
    md.append("|---|---:|---:|---:|")
    for _, row in positive_costs.head(6).iterrows():
        md.append(
            f"| {row['carrier']} | {fmt_gbp(row['net_cost'])} | {fmt_num(row['increase_MWh'])} | {fmt_num(row['decrease_MWh'])} |"
        )
    md.append("")
    md.append("| Biggest negative net-cost carriers | Net cost | Increase MWh | Decrease MWh |")
    md.append("|---|---:|---:|---:|")
    for _, row in negative_costs.head(6).iterrows():
        md.append(
            f"| {row['carrier']} | {fmt_gbp(row['net_cost'])} | {fmt_num(row['increase_MWh'])} | {fmt_num(row['decrease_MWh'])} |"
        )
    md.append("")
    md.append("## What The BM Is Actually Doing On SSHARN Hours")
    md.append("")
    focus_carriers = carrier_df.loc[
        carrier_df["carrier"].isin(["CCGT", "wind_onshore", "wind_offshore", "large_hydro", "Battery", "Pumped Storage Hydroelectricity", "nuclear", "solar_pv", "embedded_wind"])
    ].copy()
    focus_carriers = focus_carriers.sort_values("bm_minus_ws_mwh_bind_hours", ascending=False)
    md.append("| Carrier | BM-WS on SSHARN bind hours (MWh) | Increase MWh | Decrease MWh | Fleet p_nom MW |")
    md.append("|---|---:|---:|---:|---:|")
    for _, row in focus_carriers.iterrows():
        md.append(
            f"| {row['carrier']} | {fmt_num(row['bm_minus_ws_mwh_bind_hours'])} | "
            f"{fmt_num(row['increase_mwh_bind_hours'])} | {fmt_num(row['decrease_mwh_bind_hours'])} | "
            f"{fmt_num(row['p_nom_mw'])} |"
        )
    md.append("")
    md.append("- The pattern is the expected one for an East Anglia export constraint: CCGT turns up while wind turns down. The problem is that the turn-down side is too cheap, so the redispatch volume does not translate into realistic cost.")
    md.append("")
    md.append("| Top asset increases during SSHARN bind hours | Carrier | MWh | Bus |")
    md.append("|---|---|---:|---|")
    for _, row in asset_df.loc[asset_df["direction"] == "increase"].head(10).iterrows():
        md.append(f"| {row['component']} | {row['carrier']} | {fmt_num(row['bind_hours_mwh'])} | {row['bus']} |")
    md.append("")
    md.append("| Top asset decreases during SSHARN bind hours | Carrier | MWh | Bus |")
    md.append("|---|---|---:|---|")
    for _, row in asset_df.loc[asset_df["direction"] == "decrease"].head(10).iterrows():
        md.append(f"| {row['component']} | {row['carrier']} | {fmt_num(row['bind_hours_mwh'])} | {row['bus']} |")
    md.append("")
    md.append("## Diagnosis")
    md.append("")
    md.append("1. June 2020 is primarily a `SSHARN` problem and the model does enforce that boundary.")
    md.append("2. The June under-costing is mainly an **economics problem**, not a missing-boundary problem.")
    md.append("3. The two biggest drivers are fallback pricing dominance and very large negative bid credits on turn-down carriers.")
    md.append("4. The aggregate-boundary formulation explains why the boundary can bind without showing line-level congestion in `congestion.csv`.")
    md.append("5. A secondary modelling issue is that some historically questionable carriers, especially `shoreline_wave` and `tidal_stream`, still participate in redispatch and contribute negative cost offsets.")
    md.append("")
    md.append("## Recommended Next Checks")
    md.append("")
    md.append("- Re-run June with bid prices clipped to non-negative ESO-cost convention and compare BM cost again.")
    md.append("- Exclude non-existent historical marine carriers from 2020 validation scenarios.")
    md.append("- Improve direct BMU mapping coverage in East Anglia and CCGT fleets before using ELEXON-driven BM cost comparisons.")
    md.append("- For debugging, add a boundary-shadow-price export so aggregate-boundary binding is visible directly instead of inferring it from flow/limit proximity.")

    (ANALYSIS / f"{SCEN}_deep_dive_boundary_summary.csv").write_text(boundary_df.to_csv(index=False), encoding="utf-8")
    (ANALYSIS / f"{SCEN}_deep_dive_carrier_delta.csv").write_text(carrier_df.to_csv(index=False), encoding="utf-8")
    (ANALYSIS / f"{SCEN}_deep_dive_asset_delta.csv").write_text(asset_df.to_csv(index=False), encoding="utf-8")
    (ANALYSIS / f"{SCEN}_deep_dive.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {ANALYSIS / f'{SCEN}_deep_dive.md'}")
    print(f"Wrote {ANALYSIS / f'{SCEN}_deep_dive_boundary_summary.csv'}")
    print(f"Wrote {ANALYSIS / f'{SCEN}_deep_dive_carrier_delta.csv'}")
    print(f"Wrote {ANALYSIS / f'{SCEN}_deep_dive_asset_delta.csv'}")


if __name__ == "__main__":
    main()
