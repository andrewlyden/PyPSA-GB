"""Build a historical wholesale backtest using the PyPSA spot price.

This is deliberately distinct from an Ofgem-source reconstruction.  It puts
the modelled, demand-weighted delivery price into the energy/procurement part
of the consumer-bill stack and leaves the observed historical non-energy
adders in place.  Thus a historic error measures the spot-to-procurement
bridge, rather than being hidden by the observed forward index.
"""


import argparse
import html
import json
from pathlib import Path

import pandas as pd


BENCHMARK_MWH = 3.1
BASE_COMPONENT = "Published seasonal base forward allowance"
PEAK_COMPONENT = "Published seasonal peak forward allowance"
DIRECT_ENERGY_COMPONENT = "Published direct fuel energy cost"


def _period_weights(periods: pd.DataFrame) -> pd.DataFrame:
    result = periods.copy()
    result["delivery_start"] = pd.to_datetime(result["delivery_start"], errors="raise")
    result["delivery_end"] = pd.to_datetime(result["delivery_end"], errors="raise")
    result["delivery_days"] = (result["delivery_end"] - result["delivery_start"]).dt.days + 1
    if (result["delivery_days"] <= 0).any():
        raise ValueError("Wholesale delivery periods must be positive")
    result["calendar_weight"] = result["delivery_days"] / result["delivery_days"].sum()
    return result


def build_pypsa_spot_procurement_bridge(
    register: pd.DataFrame,
    hedge_periods: pd.DataFrame,
    wholesale_impact: pd.DataFrame,
    spot_multiplier: float = 1.0,
    spot_intercept_gbp_per_mwh: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return bill-component candidates, period diagnostics and a summary.

    ``spot_multiplier`` and ``spot_intercept_gbp_per_mwh`` are deliberately
    explicit bridge parameters.  The historical default is raw PyPSA spot
    pass-through (1.0 and 0.0), not a same-year fitted reproduction.
    """
    required_register = {"component", "annual_gbp_per_3100kwh_ex_vat"}
    required_periods = {
        "cap_period", "delivery_start", "delivery_end", "base_share", "peak_share",
        "forward_index_gbp_per_mwh",
    }
    required_wholesale = {
        "cap_period", "modelled_demand_mwh", "modelled_demand_weighted_clearing_price_gbp_per_mwh",
    }
    if missing := required_register.difference(register.columns):
        raise ValueError(f"Component register is missing: {sorted(missing)}")
    if missing := required_periods.difference(hedge_periods.columns):
        raise ValueError(f"Hedged-wholesale inputs are missing: {sorted(missing)}")
    if missing := required_wholesale.difference(wholesale_impact.columns):
        raise ValueError(f"Wholesale impact is missing: {sorted(missing)}")
    if spot_multiplier < 0:
        raise ValueError("spot_multiplier must be non-negative")

    periods = _period_weights(hedge_periods)
    impact = wholesale_impact[list(required_wholesale)].copy()
    if impact["cap_period"].duplicated().any():
        raise ValueError("Wholesale impact must contain one row per cap period")
    periods = periods.merge(impact, on="cap_period", validate="one_to_one")
    periods["modelled_demand_weighted_clearing_price_gbp_per_mwh"] = pd.to_numeric(
        periods["modelled_demand_weighted_clearing_price_gbp_per_mwh"], errors="raise"
    )
    periods["modelled_procurement_price_gbp_per_mwh"] = (
        periods["modelled_demand_weighted_clearing_price_gbp_per_mwh"] * float(spot_multiplier)
        + float(spot_intercept_gbp_per_mwh)
    )
    if (periods["modelled_procurement_price_gbp_per_mwh"] < 0).any():
        raise ValueError("Spot bridge produces a negative procurement price")
    periods["modelled_energy_allowance_gbp_per_3100kwh_ex_vat"] = (
        periods["modelled_procurement_price_gbp_per_mwh"] * BENCHMARK_MWH
    )

    detailed = (
        periods.get("wholesale_detail_level", pd.Series(index=periods.index, dtype=str))
        .eq("published_3a_direct_fuel_components")
    )
    candidates: list[dict] = []
    period_rows: list[dict] = []
    for period, is_detailed in zip(periods.itertuples(index=False), detailed, strict=True):
        observed_spot = float(period.modelled_demand_weighted_clearing_price_gbp_per_mwh)
        modelled = float(period.modelled_energy_allowance_gbp_per_3100kwh_ex_vat)
        if is_detailed:
            source_energy = float(period.direct_fuel_energy_cost_gbp_per_3100kwh)
            components = [(DIRECT_ENERGY_COMPONENT, modelled)]
            source_definition = "Ofgem published direct-energy term in 3a DF"
        else:
            source_energy = float(period.forward_index_gbp_per_mwh) * BENCHMARK_MWH
            components = [
                (BASE_COMPONENT, modelled * float(period.base_share)),
                (PEAK_COMPONENT, modelled * float(period.peak_share)),
            ]
            source_definition = "Ofgem published base/peak forward-index blend"
        period_rows.append({
            "cap_period": period.cap_period,
            "delivery_start": period.delivery_start,
            "delivery_end": period.delivery_end,
            "delivery_days": int(period.delivery_days),
            "calendar_weight": float(period.calendar_weight),
            "source_energy_definition": source_definition,
            "pypsa_spot_price_gbp_per_mwh": observed_spot,
            "spot_multiplier": float(spot_multiplier),
            "spot_intercept_gbp_per_mwh": float(spot_intercept_gbp_per_mwh),
            "modelled_procurement_price_gbp_per_mwh": float(period.modelled_procurement_price_gbp_per_mwh),
            "published_energy_allowance_gbp_per_3100kwh_ex_vat": source_energy,
            "modelled_energy_allowance_gbp_per_3100kwh_ex_vat": modelled,
            "difference_gbp_per_3100kwh_ex_vat": modelled - source_energy,
        })
        for component, candidate in components:
            candidates.append({
                "cap_period": period.cap_period,
                "bill_component": component,
                "candidate_period_gbp_per_3100kwh_ex_vat": candidate,
                "calendar_weight": float(period.calendar_weight),
            })

    period_output = pd.DataFrame(period_rows)
    bridge = pd.DataFrame(candidates)
    bridge["candidate_allowance_gbp_per_3100kwh_ex_vat"] = (
        bridge["candidate_period_gbp_per_3100kwh_ex_vat"] * bridge["calendar_weight"]
    )
    bridge = bridge.groupby("bill_component", as_index=False)[
        "candidate_allowance_gbp_per_3100kwh_ex_vat"
    ].sum()
    published = register[["component", "annual_gbp_per_3100kwh_ex_vat"]].copy()
    published["annual_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(
        published["annual_gbp_per_3100kwh_ex_vat"], errors="raise"
    )
    bridge = bridge.merge(
        published, left_on="bill_component", right_on="component", how="left", validate="one_to_one"
    ).drop(columns="component")
    if bridge["annual_gbp_per_3100kwh_ex_vat"].isna().any():
        missing = bridge.loc[bridge["annual_gbp_per_3100kwh_ex_vat"].isna(), "bill_component"].tolist()
        raise ValueError(f"Component register is missing spot-bridge component(s): {missing}")
    bridge = bridge.rename(columns={"annual_gbp_per_3100kwh_ex_vat": "published_allowance_gbp_per_3100kwh_ex_vat"})
    bridge["difference_gbp_per_3100kwh_ex_vat"] = (
        bridge["candidate_allowance_gbp_per_3100kwh_ex_vat"] - bridge["published_allowance_gbp_per_3100kwh_ex_vat"]
    )
    bridge["same_year_backtest_allowed"] = True
    bridge["recovery_status"] = "pypsa_spot_historical_backtest_not_same_year_fitted"
    bridge["model_basis"] = (
        "PyPSA demand-weighted delivery spot price converted to the energy/procurement allowance "
        f"with fixed multiplier {spot_multiplier:g} and intercept {spot_intercept_gbp_per_mwh:g} GBP/MWh. "
        "Published historical hedge, loss, backwardation and recovery adders remain outside this bridge."
    )
    summary = {
        "purpose": "Historical PyPSA spot-price to procurement-energy backtest",
        "calibration": {
            "spot_multiplier": float(spot_multiplier),
            "spot_intercept_gbp_per_mwh": float(spot_intercept_gbp_per_mwh),
            "same_year_fitted": False,
        },
        "published_energy_allowance_gbp_per_3100kwh_ex_vat": float(
            (period_output["published_energy_allowance_gbp_per_3100kwh_ex_vat"] * period_output["calendar_weight"]).sum()
        ),
        "modelled_energy_allowance_gbp_per_3100kwh_ex_vat": float(
            (period_output["modelled_energy_allowance_gbp_per_3100kwh_ex_vat"] * period_output["calendar_weight"]).sum()
        ),
        "difference_gbp_per_3100kwh_ex_vat": float(
            (period_output["difference_gbp_per_3100kwh_ex_vat"] * period_output["calendar_weight"]).sum()
        ),
        "future_use": "Fit the fixed bridge only on other historical years, then apply it out-of-sample.",
    }
    return bridge.sort_values("bill_component").reset_index(drop=True), period_output, summary


def _render_html(bridge: pd.DataFrame, periods: pd.DataFrame, summary: dict) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.bill_component))}</td><td>£{row.published_allowance_gbp_per_3100kwh_ex_vat:,.2f}</td>"
        f"<td>£{row.candidate_allowance_gbp_per_3100kwh_ex_vat:,.2f}</td><td>£{row.difference_gbp_per_3100kwh_ex_vat:,.2f}</td></tr>"
        for row in bridge.itertuples(index=False)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>PyPSA spot procurement backtest</title>
<style>body{{font-family:Arial;max-width:1000px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dde3ea;padding:9px;text-align:left}}th{{background:#f3f6f9}}.callout{{background:#fff4e5;border-left:4px solid #c77d17;padding:12px}}</style></head><body>
<h1>PyPSA spot-price procurement backtest</h1><p class="callout">This uses the modelled price in the bill. It is intentionally not fitted to this historical year.</p>
<p>Published energy allowance: <strong>£{summary['published_energy_allowance_gbp_per_3100kwh_ex_vat']:,.2f}</strong>; modelled: <strong>£{summary['modelled_energy_allowance_gbp_per_3100kwh_ex_vat']:,.2f}</strong>.</p>
<table><thead><tr><th>Bill component</th><th>Published</th><th>Modelled</th><th>Difference</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def write_bridge(register_path, hedge_path, wholesale_path, output_csv, output_periods, output_summary, output_html, spot_multiplier=1.0, spot_intercept_gbp_per_mwh=0.0):
    bridge, periods, summary = build_pypsa_spot_procurement_bridge(
        pd.read_csv(register_path), pd.read_csv(hedge_path), pd.read_csv(wholesale_path),
        spot_multiplier, spot_intercept_gbp_per_mwh,
    )
    for path in (output_csv, output_periods, output_summary, output_html):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(output_csv, index=False)
    periods.to_csv(output_periods, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(output_html).write_text(_render_html(bridge, periods, summary), encoding="utf-8")


if "snakemake" in globals():
    cfg = snakemake.params.forward_procurement
    write_bridge(
        snakemake.input.register_csv, snakemake.input.hedge_inputs_csv, snakemake.input.wholesale_impact_csv,
        snakemake.output.bridge_csv, snakemake.output.periods_csv, snakemake.output.summary_json, snakemake.output.report_html,
        cfg.get("spot_multiplier", 1.0), cfg.get("spot_intercept_gbp_per_mwh", 0.0),
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True); parser.add_argument("--hedge-inputs", required=True); parser.add_argument("--wholesale-impact", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-periods", required=True); parser.add_argument("--output-summary", required=True); parser.add_argument("--output-html", required=True)
    parser.add_argument("--spot-multiplier", type=float, default=1.0); parser.add_argument("--spot-intercept-gbp-per-mwh", type=float, default=0.0)
    args = parser.parse_args()
    write_bridge(args.register, args.hedge_inputs, args.wholesale_impact, args.output_csv, args.output_periods, args.output_summary, args.output_html, args.spot_multiplier, args.spot_intercept_gbp_per_mwh)
