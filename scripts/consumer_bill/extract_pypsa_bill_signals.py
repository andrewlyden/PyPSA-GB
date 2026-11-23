"""Extract bill-relevant system signals from a solved PyPSA market network.

These are deliberately signals, not consumer charges.  The recovery-bridge
module decides whether a regulatory recovery denominator is available.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

# Allow direct execution as well as Snakemake's ``script:`` invocation.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market.revenue_tracking import compute_cfd_payments, compute_roc_income
from config.config_loader import load_defaults


def _weights(network) -> pd.Series:
    if "generators" in network.snapshot_weightings:
        return network.snapshot_weightings["generators"]
    return pd.Series(1.0, index=network.snapshots)


def _price_series(path: str, snapshots: pd.Index) -> pd.Series:
    price = pd.read_csv(path, parse_dates=["snapshot"]).set_index("snapshot")["wholesale_price"]
    price.index = pd.DatetimeIndex(price.index)
    if price.index.has_duplicates:
        raise ValueError("Wholesale reference price has duplicate snapshots")
    aligned = pd.to_numeric(price.reindex(pd.DatetimeIndex(snapshots)), errors="raise")
    if not np.isfinite(aligned).all():
        raise ValueError("Wholesale reference price does not cover every PyPSA snapshot")
    return aligned


def _roc_price(config: dict, year: int, required: bool) -> float:
    """Use the requested year's assumption; never silently reuse 2020."""
    subsidies = config.get("support", {}).get("dispatch_costs", {})
    prices = subsidies.get("roc_buyout_prices", {})
    value = prices.get(year, prices.get(str(year)))
    if value is None:
        if required:
            raise ValueError(f"Missing ROC buy-out price for {year}; configure support.dispatch_costs.roc_buyout_prices")
        return 0.0
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("ROC buy-out price must be finite and non-negative")
    return value


def _roc_price_series(config: dict, snapshots: pd.DatetimeIndex, required: bool) -> pd.Series:
    """Configuration keys identify the April start of each obligation year."""
    years = snapshots.year - (snapshots.month < 4).astype(int)
    values = {int(year): _roc_price(config, int(year), required) for year in set(years)}
    return pd.Series([values[int(year)] for year in years], index=snapshots, dtype=float)


def _metric(metric, value, unit, treatment, note):
    return {"metric": metric, "value": float(value), "unit": unit, "treatment": treatment, "note": note}


def _ac_network_losses_mwh(network, weights: pd.Series) -> float:
    """Return represented AC line and transformer losses, excluding links.

    Links represent many non-network conversions in PyPSA-GB (electrolysers,
    turbines and interconnectors), so their efficiency losses must not be
    interpreted as domestic supplier network losses.
    """
    total = pd.Series(0.0, index=network.snapshots)
    for component in ("lines", "transformers"):
        static = getattr(network, component)
        dynamic = getattr(network, f"{component}_t")
        if static.empty or not hasattr(dynamic, "p0") or not hasattr(dynamic, "p1"):
            continue
        total = total.add(dynamic.p0.add(dynamic.p1).sum(axis=1), fill_value=0.0)
    return max(float((total * weights).sum()), 0.0)


def extract_signals(network, reference_price: pd.Series, defaults: dict, market_summary: dict,
                    roc_reconciliation: dict | None = None,
                    cfd_reconciliation: dict | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    weights = _weights(network)
    snapshots = pd.DatetimeIndex(network.snapshots)
    year = int(defaults.get("modelled_year", snapshots[0].year))
    if set(snapshots.year) != {year}:
        raise ValueError("Bill signals must use snapshots from the modelled year")
    reference_price = pd.to_numeric(reference_price.reindex(snapshots), errors="raise")
    if not np.isfinite(reference_price).all():
        raise ValueError("CfD reference price must cover every model snapshot")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Bill signal snapshot weights must be finite and positive")
    try:
        load = network.get_switchable_as_dense("Load", "p_set", network.snapshots).sum(axis=1)
    except AttributeError:
        load = network.loads_t.p_set.sum(axis=1)
    demand_mwh = float((load * weights).sum())
    peak_mw = float(load.max())
    dispatch = network.generators_t.p
    carriers = network.generators.carrier.astype(str)
    variable = carriers.str.contains("wind|solar", case=False, regex=True)
    variable_dispatch = dispatch.loc[:, variable.index[variable]].sum(axis=1) if variable.any() else pd.Series(0.0, index=network.snapshots)
    residual_peak = float((load - variable_dispatch).max())
    shed = carriers.str.contains("load_shedding", case=False, regex=False)
    shedding_mwh = float((dispatch.loc[:, shed.index[shed]].sum(axis=1) * weights).sum()) if shed.any() else 0.0
    firm = ~carriers.str.contains("wind|solar|load_shedding", case=False, regex=True)
    firm_p_nom = float(pd.to_numeric(network.generators.loc[firm, "p_nom"], errors="coerce").fillna(0.0).sum())
    transmission_loss_mwh = _ac_network_losses_mwh(network, weights)
    transmission_loss_rate = transmission_loss_mwh / demand_mwh if demand_mwh else 0.0

    subsidy = defaults.get("support", {}).get("labels", {})
    cfd = compute_cfd_payments(network, subsidy.get("cfd_strike_prices", {}), reference_price=reference_price)
    support = network.generators.get("support_type", pd.Series("", index=network.generators.index))
    roc_prices = _roc_price_series(defaults, snapshots, bool(support.eq("ROC").any() or roc_reconciliation))
    roc_price = float((roc_prices * weights).sum() / weights.sum())
    roc = compute_roc_income(network, roc_prices)
    all_cfd = network.generators.index[network.generators.get("support_type", pd.Series("", index=network.generators.index)).eq("CfD")]
    support_types = network.generators.get("support_type", pd.Series("", index=network.generators.index)).astype(str)
    fit_generators = network.generators.index[support_types.str.casefold().eq("fit")]
    fit_dispatch = float((dispatch.loc[:, fit_generators].sum(axis=1) * weights).sum()) if len(fit_generators) else 0.0
    all_cfd_dispatch = float((dispatch.loc[:, all_cfd].sum(axis=1) * weights).sum()) if len(all_cfd) else 0.0
    priced_cfd_dispatch = float(cfd["total_dispatch_MWh"].sum()) if not cfd.empty else 0.0
    all_cfd_details = network.generators.reindex(columns=["carrier", "cfd_round", "p_nom"]).loc[all_cfd].copy()
    all_cfd_details["physical_dispatch_MWh"] = [(dispatch[name] * weights).sum() for name in all_cfd_details.index]
    priced_columns = ["generator", "carrier", "cfd_round", "strike_price", "avg_reference_price", "total_dispatch_MWh", "net_cfd_cost"]
    cfd_details = all_cfd_details.reset_index(names="generator").merge(
        cfd[priced_columns] if not cfd.empty else pd.DataFrame(columns=priced_columns),
        on=["generator", "carrier", "cfd_round"], how="left", validate="one_to_one",
    )
    cfd_details["payment_gbp_per_mwh"] = cfd_details["net_cfd_cost"] / cfd_details["total_dispatch_MWh"]
    cfd_details["has_strike_price"] = cfd_details["strike_price"].notna()

    length = pd.to_numeric(network.lines.get("length", pd.Series(0.0, index=network.lines.index)), errors="coerce").fillna(0.0)
    capacity = pd.to_numeric(network.lines.get("s_nom_original", network.lines.get("s_nom", pd.Series(0.0, index=network.lines.index))), errors="coerce").fillna(0.0)
    congestion = market_summary.get("congestion", {})
    redispatch = market_summary.get("redispatch", {})
    redispatch_increase_mwh = float(redispatch.get("total_increase_MWh", 0.0))
    redispatch_decrease_mwh = float(redispatch.get("total_decrease_MWh", 0.0))
    gross_redispatch_mwh = redispatch_increase_mwh + redispatch_decrease_mwh
    metrics = [
        _metric("model_demand_mwh", demand_mwh, "MWh", "recovery denominator context", "System demand; convert to supplier sales before bill allocation."),
        _metric("peak_demand_mw", peak_mw, "MW", "capacity adequacy context", "Peak model load, not a Capacity Market payment."),
        _metric("wind_solar_residual_peak_mw", residual_peak, "MW", "capacity adequacy context", "Peak load after modelled wind and solar dispatch."),
        _metric("firm_generator_capacity_mw", firm_p_nom, "MW", "capacity adequacy context", "Nameplate proxy only; it is not de-rated auction capacity."),
        _metric("load_shedding_mwh", shedding_mwh, "MWh", "adequacy diagnostic", "Reliability diagnostic, not a consumer bill item."),
        _metric("transmission_loss_mwh", transmission_loss_mwh, "MWh", "loss-factor recovery index", "Represented AC line and transformer losses only; excludes distribution losses and non-network link conversions."),
        _metric("transmission_loss_rate", transmission_loss_rate, "MWh/MWh", "loss-factor recovery index", "Use only as a calibrated index for the published supplier-loss allowance."),
        _metric("cfd_total_dispatch_mwh", all_cfd_dispatch, "MWh", "CfD scheme-cost input", "All generators labelled CfD in the solved PyPSA network."),
        _metric("cfd_priced_dispatch_mwh", priced_cfd_dispatch, "MWh", "CfD scheme-cost input", "Only CfD dispatch with an available strike-price mapping."),
        _metric("cfd_payment_proxy_gbp", float(cfd["net_cfd_cost"].sum()) if not cfd.empty else 0.0, "GBP", "CfD scheme-cost proxy", "Difference-payment proxy using model dispatch and wholesale reference price; reconcile to LCCC levy/outturn."),
        _metric("roc_dispatch_mwh", float(roc["total_dispatch_MWh"].sum()) if not roc.empty else 0.0, "MWh", "RO scheme-cost input", "ROC-labelled dispatch with a valid banding."),
        _metric("roc_obligation_proxy_gbp", float(roc["roc_income_total"].sum()) if not roc.empty else 0.0, "GBP", "RO scheme-cost proxy", "ROC income proxy using the configured buy-out price; reconcile to obligation recovery."),
        _metric("fit_dispatch_mwh", fit_dispatch, "MWh", "FiT levelisation context", "FiT-labelled model generation only. Zero means the current PyPSA support classification does not yet represent FiT."),
        _metric("transmission_capacity_mw", float(capacity.sum()), "MW", "network allowed-revenue context", "Modelled transmission thermal-capacity proxy, not an allowed-revenue calculation."),
        _metric("transmission_line_count", len(network.lines), "count", "network allowed-revenue context", "Modelled transmission circuits, excluding the distribution system."),
        _metric("congested_line_count", congestion.get("congested_lines", 0), "count", "network investment context", "Constrained transmission assets in the balancing result."),
        _metric("gross_redispatch_mwh", gross_redispatch_mwh, "MWh", "balancing recovery index", "Gross upward plus downward redispatch; use as a calibrated BSUoS/imbalance activity index, not as settlement expenditure."),
        _metric("gross_redispatch_mwh_per_mwh_demand", gross_redispatch_mwh / demand_mwh if demand_mwh else 0.0, "MWh/MWh", "balancing recovery index", "Gross redispatch divided by model demand; central operational index for the BSUoS and imbalance bridges."),
        _metric("redispatch_net_cost_gbp", redispatch.get("total_net_cost", 0.0), "GBP", "balancing recovery index", "Input to calibrated BSUoS and imbalance bridges."),
    ]
    if roc_reconciliation:
        reconciled_rocs = float(roc_reconciliation.get("reconciled_roc_certificates", 0.0))
        direct_model_rocs = float(roc_reconciliation.get("modelled_labelled_roc_certificates", 0.0))
        official_rocs = float(roc_reconciliation.get("ofgem_rocs_issued", 0.0))
        external_rocs = float(roc_reconciliation.get("external_representation_bridge_roc_certificates", 0.0))
        recovered_model_rocs = float(roc_reconciliation.get("recovered_from_unlabelled_model_roc_certificates", 0.0))
        metrics.extend([
            _metric("roc_direct_model_certificates", direct_model_rocs, "ROC", "direct physical-model validation", "Primary validation result before any recovery calibration."),
            _metric("roc_direct_model_coverage_of_ofgem", direct_model_rocs / official_rocs if official_rocs else 0.0, "ratio", "direct physical-model validation", "Direct station-labelled model certificates divided by Ofgem calendar-year issue volume."),
            _metric("roc_direct_model_obligation_proxy_gbp", direct_model_rocs * roc_price, "GBP", "direct physical-model validation", "Direct model certificates multiplied by the configured buy-out price, before recovery bridges."),
            _metric("roc_reconciled_certificates", reconciled_rocs, "ROC", "calendar-year RO recovery benchmark", "Station-labelled model ROCs plus separately reported model-candidate and external representation bridges."),
            _metric("roc_recovered_from_unlabelled_model_certificates", recovered_model_rocs, "ROC", "calibrated model recovery", "Certificates recovered from compatible PyPSA dispatch using calendar-year technology eligibility shares."),
            _metric("roc_external_representation_bridge_certificates", external_rocs, "ROC", "external representation bridge", "Certificates from embedded or otherwise unrepresented generation, including any negative over-model adjustment."),
            _metric("roc_reconciled_obligation_proxy_gbp", reconciled_rocs * roc_price, "GBP", "calendar-year RO scheme-cost benchmark", "Calendar-aligned certificate volume multiplied by the configured buy-out price; recycling value remains separate."),
        ])
    if cfd_reconciliation:
        independent_cfd = float(cfd_reconciliation.get("modelled_payment_independent_formula_gbp", 0.0))
        actual_cfd = float(cfd_reconciliation.get("actual_payment_gbp_mapped_contracts", 0.0))
        intensity_cfd = float(cfd_reconciliation.get("modelled_dispatch_at_lccc_intensity_gbp", 0.0))
        metrics.extend([
            _metric("cfd_independent_formula_payment_gbp", independent_cfd, "GBP", "direct physical-model validation", "Primary historical CfD result: final physical dispatch multiplied by published daily indexed strike-minus-settlement-reference prices."),
            _metric("cfd_actual_lccc_payment_gbp", actual_cfd, "GBP", "official historical comparator", "Published LCCC payment for the mapped calendar-year contract portfolio."),
            _metric("cfd_independent_formula_payment_coverage", independent_cfd / actual_cfd if actual_cfd else 0.0, "ratio", "direct physical-model validation", "Independent formula payment divided by the official LCCC payment."),
            _metric("cfd_realised_intensity_bridge_payment_gbp", intensity_cfd, "GBP", "diagnostic bridge", "Model physical dispatch multiplied by realised official GBP/MWh; retained as a secondary diagnostic, not the independent result."),
        ])
    summary = {
        "modelled_year": year,
        "represented_hours": float(weights.sum()),
        "snapshot_start": str(snapshots.min()),
        "snapshot_end": str(snapshots.max()),
        "roc_price_basis": "Dispatch uses April-to-March obligation-year buy-out prices. Annual reconciled certificate totals use a represented-hours-weighted price proxy; recycling and supplier recovery remain separate.",
        "roc_obligation_prices_gbp_per_roc": {str(y): _roc_price(defaults, int(y), False) for y in sorted(set(snapshots.year - (snapshots.month < 4).astype(int)))},
        "cfd_priced_dispatch_coverage": priced_cfd_dispatch / all_cfd_dispatch if all_cfd_dispatch else None,
        "roc_buyout_price_gbp_per_roc": roc_price,
        "purpose": "PyPSA bill-relevant signals; none are consumer charges without a recovery bridge.",
    }
    return pd.DataFrame(metrics), summary, cfd_details


def write_signals(network_path, wholesale_price_path, market_summary_path, output_csv, output_summary, output_cfd_details=None, roc_reconciliation_path=None, cfd_reconciliation_path=None, scenario_config=None):
    import pypsa
    network = pypsa.Network(network_path)
    if scenario_config is not None:
        # Snakemake supplies the resolved scenario including inherited defaults.
        defaults = scenario_config
    else:
        defaults = load_defaults()
    market_summary = json.loads(Path(market_summary_path).read_text(encoding="utf-8"))
    price = _price_series(wholesale_price_path, network.snapshots)
    roc_reconciliation = json.loads(Path(roc_reconciliation_path).read_text(encoding="utf-8")) if roc_reconciliation_path else None
    cfd_reconciliation = json.loads(Path(cfd_reconciliation_path).read_text(encoding="utf-8")) if cfd_reconciliation_path else None
    metrics, summary, cfd_details = extract_signals(
        network, price, defaults, market_summary, roc_reconciliation, cfd_reconciliation
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if output_cfd_details:
        Path(output_cfd_details).parent.mkdir(parents=True, exist_ok=True)
        cfd_details.to_csv(output_cfd_details, index=False)


if "snakemake" in globals():
    # snakemake.input.defaults (every defaults file) is a change trigger only: the resolved scenario is the config.
    write_signals(snakemake.input.network_nc, snakemake.input.wholesale_price_csv, snakemake.input.market_summary_json, snakemake.output.signals_csv, snakemake.output.summary_json, snakemake.output.cfd_details_csv, getattr(snakemake.input, "roc_reconciliation_json", None), getattr(snakemake.input, "cfd_reconciliation_json", None), snakemake.params.scenario_config)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True); parser.add_argument("--wholesale-price", required=True); parser.add_argument("--market-summary", required=True)
    parser.add_argument("--output-cfd-details")
    parser.add_argument("--roc-reconciliation")
    parser.add_argument("--cfd-reconciliation")
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_signals(args.network, args.wholesale_price, args.market_summary, args.output_csv, args.output_summary, args.output_cfd_details, args.roc_reconciliation, args.cfd_reconciliation)
