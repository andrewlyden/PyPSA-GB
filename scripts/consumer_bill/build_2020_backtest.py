"""Build an honest annual historical-vs-reconstructed consumer-bill backtest."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd


def _weighted_model_price(wholesale: pd.DataFrame) -> float:
    required = {"modelled_demand_mwh", "modelled_demand_weighted_clearing_price_gbp_per_mwh"}
    missing = required.difference(wholesale.columns)
    if missing:
        raise ValueError(f"Wholesale-impact input is missing: {sorted(missing)}")
    demand = pd.to_numeric(wholesale["modelled_demand_mwh"], errors="raise")
    price = pd.to_numeric(wholesale["modelled_demand_weighted_clearing_price_gbp_per_mwh"], errors="raise")
    if demand.sum() <= 0:
        raise ValueError("Modelled demand must be positive")
    return float((demand * price).sum() / demand.sum())


def build_backtest(
    register: pd.DataFrame,
    forecast_registry: pd.DataFrame,
    wholesale: pd.DataFrame,
    recovery_bridges: pd.DataFrame | None = None,
    support_bridge: pd.DataFrame | None = None,
    include_same_year_support_backtest: bool = False,
    forward_procurement_bridge: pd.DataFrame | None = None,
    network_bridge: pd.DataFrame | None = None,
    include_same_year_network_backtest: bool = False,
) -> tuple[pd.DataFrame, dict]:
    required = {"component", "stack_group", "annual_gbp_per_3100kwh_ex_vat"}
    missing = required.difference(register.columns)
    if missing:
        raise ValueError(f"Ofgem benchmark register is missing: {sorted(missing)}")
    forecast_fields = {"component", "forecast_driver", "forecast_method"}
    missing = forecast_fields.difference(forecast_registry.columns)
    if missing:
        raise ValueError(f"Forecast registry is missing: {sorted(missing)}")
    model_price = _weighted_model_price(wholesale)
    rows = register.merge(
        forecast_registry[list(forecast_fields)].drop_duplicates("component"),
        on="component", how="left", validate="one_to_one",
    ).copy()
    rows["historical_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(rows["annual_gbp_per_3100kwh_ex_vat"], errors="raise")
    rows["backtest_gbp_per_3100kwh_ex_vat"] = rows["historical_gbp_per_3100kwh_ex_vat"]
    rows["backtest_treatment"] = "historical_fixed_external_or_regulatory"
    rows["backtest_method"] = "Published annual allowance retained pending a component-specific recovery model"
    rows["is_directly_modelled"] = False
    rows["is_modelled_support_recovery"] = False
    rows["is_reconstructed_forward_procurement"] = False
    rows["is_modelled_network_recovery"] = False
    rows["recovery_bridge_status"] = "not_applicable_or_not_started"
    rows["recovery_bridge_candidate_gbp_per_3100kwh_ex_vat"] = pd.NA

    if recovery_bridges is not None:
        bridge_fields = {"component", "evidence_status", "candidate_gbp_per_3100kwh_ex_vat"}
        if missing := bridge_fields.difference(recovery_bridges.columns):
            raise ValueError(f"Recovery-bridge input is missing: {sorted(missing)}")
        bridges = recovery_bridges[list(bridge_fields)].drop_duplicates("component")
        rows = rows.merge(bridges, on="component", how="left", validate="one_to_one")
        rows["recovery_bridge_status"] = rows["evidence_status"].fillna(rows["recovery_bridge_status"])
        rows["recovery_bridge_candidate_gbp_per_3100kwh_ex_vat"] = rows["candidate_gbp_per_3100kwh_ex_vat"]
        rows = rows.drop(columns=["evidence_status", "candidate_gbp_per_3100kwh_ex_vat"])

    if support_bridge is not None and include_same_year_support_backtest:
        required_support = {
            "scheme", "candidate_allowance_gbp_per_3100kwh_ex_vat",
            "same_year_backtest_allowed", "recovery_status", "model_basis",
        }
        if missing := required_support.difference(support_bridge.columns):
            raise ValueError(f"Annual support bridge is missing: {sorted(missing)}")
        support = support_bridge.loc[support_bridge["same_year_backtest_allowed"].astype(bool)].copy()
        for scheme, matcher in {
            "CfD": rows["component"].astype(str).str.contains("CfD", regex=False, na=False),
            "RO": rows["component"].astype(str).str.strip().str.endswith("RO", na=False),
        }.items():
            source = support.loc[support["scheme"].eq(scheme)]
            if len(source) != 1:
                raise ValueError(f"Expected one eligible {scheme} row in the annual support bridge")
            if matcher.sum() != 1:
                raise ValueError(f"Expected one {scheme} component in the benchmark register")
            candidate = float(source["candidate_allowance_gbp_per_3100kwh_ex_vat"].iloc[0])
            if candidate < 0 and scheme != "CfD":
                raise ValueError(f"{scheme} support candidate cannot be negative")
            rows.loc[matcher, "backtest_gbp_per_3100kwh_ex_vat"] = candidate
            rows.loc[matcher, "backtest_treatment"] = "modelled_support_same_year_backtest"
            rows.loc[matcher, "backtest_method"] = str(source["model_basis"].iloc[0])
            rows.loc[matcher, "is_modelled_support_recovery"] = True
            rows.loc[matcher, "recovery_bridge_status"] = str(source["recovery_status"].iloc[0])
            rows.loc[matcher, "recovery_bridge_candidate_gbp_per_3100kwh_ex_vat"] = candidate

    if forward_procurement_bridge is None:
        raise ValueError("A named wholesale-procurement bridge is required")
    else:
        required_wholesale = {
            "bill_component", "candidate_allowance_gbp_per_3100kwh_ex_vat",
            "same_year_backtest_allowed", "recovery_status", "model_basis",
        }
        if missing := required_wholesale.difference(forward_procurement_bridge.columns):
            raise ValueError(f"Ofgem forward-procurement bridge is missing: {sorted(missing)}")
        bridge = forward_procurement_bridge.loc[forward_procurement_bridge["same_year_backtest_allowed"].astype(bool)].copy()
        if bridge["bill_component"].duplicated().any():
            raise ValueError("Ofgem forward-procurement bridge contains duplicate bill components")
        for source in bridge.itertuples(index=False):
            mask = rows["component"].eq(source.bill_component)
            if mask.sum() != 1:
                raise ValueError(f"Expected one {source.bill_component!r} component in the historical benchmark register")
            candidate = float(source.candidate_allowance_gbp_per_3100kwh_ex_vat)
            if candidate < 0:
                raise ValueError(f"Wholesale-procurement candidate cannot be negative: {source.bill_component}")
            rows.loc[mask, "backtest_gbp_per_3100kwh_ex_vat"] = candidate
            rows.loc[mask, "backtest_treatment"] = (
                "pypsa_spot_procurement_historical_backtest"
                if "pypsa_spot" in str(source.recovery_status) else "ofgem_forward_procurement_reconstruction"
            )
            rows.loc[mask, "backtest_method"] = str(source.model_basis)
            rows.loc[mask, "is_reconstructed_forward_procurement"] = True
            rows.loc[mask, "recovery_bridge_status"] = str(source.recovery_status)
            rows.loc[mask, "recovery_bridge_candidate_gbp_per_3100kwh_ex_vat"] = candidate

    if network_bridge is not None and include_same_year_network_backtest:
        required_network = {
            "component", "candidate_allowance_gbp_per_3100kwh_ex_vat",
            "same_year_backtest_allowed", "recovery_status", "model_basis",
        }
        if missing := required_network.difference(network_bridge.columns):
            raise ValueError(f"Network-cost bridge is missing: {sorted(missing)}")
        bridge = network_bridge.loc[network_bridge["same_year_backtest_allowed"].astype(bool)].copy()
        if bridge["component"].duplicated().any():
            raise ValueError("Network-cost bridge contains duplicate components")
        for source in bridge.itertuples(index=False):
            mask = rows["component"].eq(source.component)
            if mask.sum() != 1:
                raise ValueError(f"Expected one {source.component!r} component in the 2020 benchmark register")
            candidate = float(source.candidate_allowance_gbp_per_3100kwh_ex_vat)
            if candidate < 0:
                raise ValueError(f"Network-cost candidate cannot be negative: {source.component}")
            rows.loc[mask, "backtest_gbp_per_3100kwh_ex_vat"] = candidate
            rows.loc[mask, "backtest_treatment"] = "modelled_network_same_year_allocation"
            rows.loc[mask, "backtest_method"] = str(source.model_basis)
            rows.loc[mask, "is_modelled_network_recovery"] = True
            rows.loc[mask, "recovery_bridge_status"] = str(source.recovery_status)
            rows.loc[mask, "recovery_bridge_candidate_gbp_per_3100kwh_ex_vat"] = candidate
    vat_mask = rows["stack_group"].eq("vat")
    if vat_mask.sum() != 1:
        raise ValueError("Expected one VAT row")
    pre_vat = ~vat_mask
    rows.loc[vat_mask, "backtest_gbp_per_3100kwh_ex_vat"] = float(rows.loc[pre_vat, "backtest_gbp_per_3100kwh_ex_vat"].sum()) * 0.05
    rows.loc[vat_mask, "backtest_treatment"] = "derived_tax"
    rows.loc[vat_mask, "backtest_method"] = "5% VAT applied to reconstructed ex-VAT amount"
    rows["difference_gbp_per_3100kwh"] = rows["backtest_gbp_per_3100kwh_ex_vat"] - rows["historical_gbp_per_3100kwh_ex_vat"]

    historical_total = float(rows["historical_gbp_per_3100kwh_ex_vat"].sum())
    backtest_total = float(rows["backtest_gbp_per_3100kwh_ex_vat"].sum())
    direct_historical = float(rows.loc[rows["is_directly_modelled"], "historical_gbp_per_3100kwh_ex_vat"].sum())
    support_historical = float(rows.loc[rows["is_modelled_support_recovery"], "historical_gbp_per_3100kwh_ex_vat"].sum())
    forward_historical = float(rows.loc[rows["is_reconstructed_forward_procurement"], "historical_gbp_per_3100kwh_ex_vat"].sum())
    network_historical = float(rows.loc[rows["is_modelled_network_recovery"], "historical_gbp_per_3100kwh_ex_vat"].sum())
    summary = {
        "purpose": "Annual component-level backtest for validating future consumer-bill pathways",
        "benchmark": "GB-average Ofgem benchmark, 3,100 kWh/year, including VAT",
        "historical_bill_gbp": historical_total,
        "reconstructed_bill_gbp": backtest_total,
        "difference_gbp": backtest_total - historical_total,
        "difference_percent": (backtest_total - historical_total) / historical_total,
        "pypsa_demand_weighted_wholesale_price_gbp_per_mwh": model_price,
        "direct_modelled_historical_coverage_gbp": direct_historical,
        "direct_modelled_historical_coverage_share": direct_historical / historical_total,
        "same_year_modelled_support_historical_coverage_gbp": support_historical,
        "same_year_modelled_support_historical_coverage_share": support_historical / historical_total,
        "same_year_support_backtest_included": bool(include_same_year_support_backtest and support_bridge is not None),
        "wholesale_procurement_historical_coverage_gbp": forward_historical,
        "wholesale_procurement_historical_coverage_share": forward_historical / historical_total,
        "wholesale_procurement_bridge_included": True,
        "same_year_modelled_network_historical_coverage_gbp": network_historical,
        "same_year_modelled_network_historical_coverage_share": network_historical / historical_total,
        "same_year_network_backtest_included": bool(include_same_year_network_backtest and network_bridge is not None),
        "is_end_to_end_modelled_bill": False,
        "recovery_bridge_status_counts": rows["recovery_bridge_status"].value_counts().to_dict(),
        "interpretation": "This is a hybrid backtest. The named wholesale-procurement bridge determines whether observed forward inputs or a PyPSA delivery-price backtest enters the bill. The CfD/RO and network paths are clearly labelled same-year reconstructions; none is a future pass-through or out-of-sample validation result.",
        "next_recovery_bridges": ["CfD", "RO", "balancing/BSUoS", "Capacity Market", "network allowed revenue and charging"],
    }
    return rows.sort_values(["stack_group", "component"]).reset_index(drop=True), summary


def _render_html(rows: pd.DataFrame, summary: dict) -> str:
    table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.component))}</td><td>£{row.historical_gbp_per_3100kwh_ex_vat:,.2f}</td>"
        f"<td>£{row.backtest_gbp_per_3100kwh_ex_vat:,.2f}</td><td>£{row.difference_gbp_per_3100kwh:,.2f}</td>"
        f"<td>{html.escape(str(row.backtest_treatment))}</td><td>{html.escape(str(row.recovery_bridge_status))}</td></tr>"
        for row in rows.itertuples(index=False)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Consumer-bill backtest</title>
<style>body{{font-family:Arial;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dde3ea;padding:9px;text-align:left}}th{{background:#f3f6f9}}.callout{{background:#fff4e5;border-left:4px solid #c77d17;padding:12px}}</style></head><body>
<h1>Consumer-bill backtest</h1><p class="callout">{html.escape(summary['interpretation'])}</p>
<p>Historical bill: <strong>£{summary['historical_bill_gbp']:,.2f}</strong>; reconstructed bill: <strong>£{summary['reconstructed_bill_gbp']:,.2f}</strong>; difference: <strong>£{summary['difference_gbp']:,.2f}</strong>.</p>
<p>Wholesale-procurement bridge coverage: {summary['wholesale_procurement_historical_coverage_share']:.1%}. See the component treatments for whether it uses observed forwards or PyPSA spot.</p>
<table><thead><tr><th>Component</th><th>Historical</th><th>Backtest</th><th>Difference</th><th>Treatment</th><th>Recovery bridge</th></tr></thead><tbody>{table}</tbody></table></body></html>"""


def write_backtest(register_path, forecast_registry_path, wholesale_path, output_csv, output_summary, output_html,
                   recovery_bridges_path=None, support_bridge_path=None,
                   include_same_year_support_backtest=False, forward_procurement_bridge_path=None, network_bridge_path=None,
                   include_same_year_network_backtest=False):
    bridges = pd.read_csv(recovery_bridges_path) if recovery_bridges_path else None
    support = pd.read_csv(support_bridge_path) if support_bridge_path else None
    forward_procurement_bridge = pd.read_csv(forward_procurement_bridge_path) if forward_procurement_bridge_path else None
    network_bridge = pd.read_csv(network_bridge_path) if network_bridge_path else None
    rows, summary = build_backtest(
        pd.read_csv(register_path), pd.read_csv(forecast_registry_path), pd.read_csv(wholesale_path),
        bridges, support, include_same_year_support_backtest, forward_procurement_bridge, network_bridge,
        include_same_year_network_backtest,
    )
    for output in (output_csv, output_summary, output_html): Path(output).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(output_html).write_text(_render_html(rows, summary), encoding="utf-8")


if "snakemake" in globals():
    support_paths = list(snakemake.input.support_bridge_csv)
    network_paths = list(snakemake.input.network_bridge_csv)
    write_backtest(
        snakemake.input.register_csv, snakemake.input.forecast_registry_csv,
        snakemake.input.wholesale_impact_csv, snakemake.output.backtest_csv,
        snakemake.output.summary_json, snakemake.output.report_html,
        snakemake.input.recovery_bridges_csv,
        support_paths[0] if support_paths else None,
        snakemake.params.include_same_year_support_backtest,
        snakemake.input.forward_procurement_bridge_csv,
        network_paths[0] if network_paths else None,
        snakemake.params.include_same_year_network_backtest,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True); parser.add_argument("--forecast-registry", required=True); parser.add_argument("--wholesale-impact", required=True)
    parser.add_argument("--recovery-bridges")
    parser.add_argument("--support-bridge")
    parser.add_argument("--include-same-year-support-backtest", action="store_true")
    parser.add_argument("--forward-procurement-bridge", required=True)
    parser.add_argument("--network-bridge")
    parser.add_argument("--include-same-year-network-backtest", action="store_true")
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True); parser.add_argument("--output-html", required=True)
    args = parser.parse_args(); write_backtest(args.register, args.forecast_registry, args.wholesale_impact, args.output_csv, args.output_summary, args.output_html, args.recovery_bridges, args.support_bridge, args.include_same_year_support_backtest, args.forward_procurement_bridge, args.network_bridge, args.include_same_year_network_backtest)
