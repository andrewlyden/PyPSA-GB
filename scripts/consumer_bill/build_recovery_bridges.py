"""Build transparent consumer-bill recovery bridges from PyPSA and regulatory inputs.

The calculation is deliberately conservative: a bridge produces a candidate
allowance only when its recovery denominator and all required non-PyPSA inputs
are provided.  In particular, redispatch cost is *not* passed directly through
as BSUoS.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import math


BENCHMARK_MWH = 3.1


def _value(mapping: dict, name: str):
    value = mapping.get(name)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _per_customer(cost_gbp, sales_mwh, share=1.0) -> float | None:
    if cost_gbp is None or sales_mwh is None or sales_mwh <= 0:
        return None
    if not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("domestic_recovery_share must be between zero and one")
    return cost_gbp * share / sales_mwh * BENCHMARK_MWH


def _share(section):
    value = _value(section, "domestic_recovery_share")
    return 1.0 if value is None else value


def _historical(register: pd.DataFrame, component: str) -> float:
    def repair_mojibake(value: object) -> str:
        text = str(value)
        # Some legacy CSVs contain an em dash decoded through Windows-1252 one
        # or even two times. Repair it before component-key comparison.
        for _ in range(2):
            try:
                repaired = text.encode("cp1252").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
            if repaired == text:
                break
            text = repaired
        return " ".join(text.replace("—", "-").split()).strip()

    normalise = lambda values: values.map(repair_mojibake)
    matched = register.loc[normalise(register["component"]).eq(normalise(pd.Series([component])).iloc[0]), "annual_gbp_per_3100kwh_ex_vat"]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one benchmark component: {component}")
    return float(pd.to_numeric(matched.iloc[0], errors="raise"))


def _bridge_row(component, bridge, historical, candidate, status, signal=None, signal_unit=None, required="", note=""):
    return {
        "component": component,
        "bridge": bridge,
        "historical_gbp_per_3100kwh_ex_vat": historical,
        "candidate_gbp_per_3100kwh_ex_vat": candidate,
        "evidence_status": status,
        "backtest_substitution_allowed": False,
        "modelled_signal_value": signal,
        "modelled_signal_unit": signal_unit,
        "required_inputs": required,
        "method_note": note,
    }


def _has_component(register: pd.DataFrame, component: str) -> bool:
    """Check a benchmark component using the mojibake-safe lookup."""
    try:
        _historical(register, component)
    except ValueError:
        return False
    return True


def _indexed_allowance(historical: float, section: dict, signal: float | None,
                       calibration_key: str) -> tuple[float | None, str]:
    """Index a published allowance by a model signal, never by raw cost."""
    if signal is None or signal < 0:
        return None, "modelled_signal_pending"
    calibration = _value(section, calibration_key)
    same_year = bool(section.get("use_current_run_as_calibration_baseline", False))
    if same_year:
        calibration = signal
    if calibration is None or calibration <= 0:
        return None, "input_required"
    return historical * signal / calibration, (
        "assumption_calibrated_same_year_index" if same_year else "candidate_unvalidated"
    )


def _redispatch_metrics(market_summary: dict, demand_mwh: float) -> tuple[float, float | None, float | None]:
    """Return raw cost (diagnostic) and gross-volume balancing activity.

    PyPSA bid/offer costs are not total BSUoS expenditure. Gross upward plus
    downward redispatch is therefore the central operational index; the cost
    result remains available solely as a diagnostic/sensitivity input.
    """
    redispatch = market_summary.get("redispatch", {})
    net_cost = float(redispatch.get("total_net_cost", 0.0))
    increase = _value(redispatch, "total_increase_MWh")
    decrease = _value(redispatch, "total_decrease_MWh")
    if increase is None or decrease is None or increase < 0 or decrease < 0:
        return net_cost, None, None
    gross_mwh = increase + decrease
    return net_cost, gross_mwh, gross_mwh / demand_mwh


def build_bridges(register: pd.DataFrame, wholesale: pd.DataFrame, market_summary: dict, assumptions: dict, signals: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    required = {"component", "annual_gbp_per_3100kwh_ex_vat"}
    if missing := required.difference(register.columns):
        raise ValueError(f"Component register is missing: {sorted(missing)}")
    demand_mwh = float(pd.to_numeric(wholesale["modelled_demand_mwh"], errors="raise").sum())
    if demand_mwh <= 0:
        raise ValueError("Modelled demand must be positive")
    benchmark_mwh = float(assumptions.get("representative_customer_consumption_kwh", 3100)) / 1000
    if benchmark_mwh != BENCHMARK_MWH:
        raise ValueError("This consumer-bill register is fixed at a 3,100 kWh benchmark")
    signal_values: dict[str, float] = {}
    if signals is not None:
        if missing := {"metric", "value"}.difference(signals.columns):
            raise ValueError(f"PyPSA signals input is missing: {sorted(missing)}")
        signal_values = dict(zip(signals["metric"], pd.to_numeric(signals["value"], errors="raise")))

    def volumetric(section):
        cost = _value(section, "modelled_recovery_cost_gbp")
        if cost is None:
            cost = _value(section, "allowed_revenue_recovered_from_suppliers_gbp")
        return _per_customer(cost, _value(section, "billable_supplier_sales_mwh"), _share(section))

    missing_baselines = set()
    baseline_aliases = {}

    def _benchmark(component):
        # New cap layouts may embed CfD in direct fuel and publish only an
        # aggregate network allowance. Never invent a network split.
        alias = component
        if component.endswith(" CfD") and not _has_component(register, component):
            matches = register.loc[register["component"].str.contains("CfD", case=False, na=False), "component"]
            if len(matches) == 1:
                alias = str(matches.iloc[0])
                baseline_aliases[component] = alias
        if not _has_component(register, alias):
            missing_baselines.add(component)
            return float("nan")
        return _historical(register, alias)

    rows = []
    revenue = market_summary.get("revenue", {})
    for component, key, revenue_key, signal_key, label in [
        ("Ofgem Annex 4 — CfD", "cfd", "cfd_total_cost", "cfd_payment_proxy_gbp", "CfD levy recovery"),
        ("Ofgem Annex 4 — RO", "ro", "roc_total_income", "roc_obligation_proxy_gbp", "Renewables Obligation recovery"),
    ]:
        section = assumptions.get(key, {})
        modelled_cost = _value(section, "modelled_recovery_cost_gbp")
        if modelled_cost is None and key == "cfd" and "cfd_independent_formula_payment_gbp" in signal_values:
            modelled_cost = signal_values["cfd_independent_formula_payment_gbp"]
        if modelled_cost is None and key == "ro" and "roc_reconciled_obligation_proxy_gbp" in signal_values:
            modelled_cost = signal_values["roc_reconciled_obligation_proxy_gbp"]
        if modelled_cost is None:
            modelled_cost = signal_values.get(signal_key)
        if modelled_cost is None:
            modelled_cost = _value(revenue, revenue_key)
        candidate = _per_customer(modelled_cost, _value(section, "billable_supplier_sales_mwh"), _share(section))
        status = "candidate_unvalidated" if candidate is not None else ("modelled_cost_pending_recovery_allocation" if modelled_cost is not None else "input_required")
        rows.append(_bridge_row(component, label, _benchmark(component), candidate, status,
            modelled_cost, "GBP in model year" if modelled_cost is not None else None,
            required="modelled scheme cost, billable supplier sales and domestic allocation", 
            note=("Uses the calendar-aligned ROC reconciliation when available, then freshly extracted model signals, then archived market revenue. "
                  "Supplier-sales allocation and ROC recycling value remain external evidence." if key == "ro" else
                  "Uses independent daily LCCC contract terms when available, then freshly extracted model signals, then archived market revenue; supplier-sales allocation remains external evidence.")))

    bsuos = assumptions.get("bsuos", {})
    redispatch_cost, gross_redispatch_mwh, redispatch_volume_intensity = _redispatch_metrics(
        market_summary, demand_mwh
    )
    historical_bsuos = _benchmark("Ofgem Annex 3 — BSUoS")
    settlement_cost = _value(bsuos, "historical_settlement_cost_gbp")
    settlement_volume = _value(bsuos, "historical_settlement_volume_mwh")
    candidate = None
    calibration_rate = _value(bsuos, "calibration_gross_redispatch_mwh_per_mwh")
    calibration_allowance = _value(bsuos, "calibration_bsuos_gbp_per_3100kwh_ex_vat") or historical_bsuos
    same_year_calibration = bool(bsuos.get("use_current_run_as_calibration_baseline", False))
    if same_year_calibration and redispatch_volume_intensity is not None:
        calibration_rate = redispatch_volume_intensity
    if calibration_rate is not None and calibration_rate > 0 and redispatch_volume_intensity is not None:
        candidate = calibration_allowance * redispatch_volume_intensity / calibration_rate
    outturn_rate = settlement_cost / settlement_volume if settlement_cost and settlement_volume else None
    retail_recovery_factor = (
        (historical_bsuos / BENCHMARK_MWH) / outturn_rate
        if outturn_rate is not None and outturn_rate > 0 else None
    )
    status = "calibrated_same_year_volume_index" if same_year_calibration else ("candidate_unvalidated" if candidate is not None else "input_required")
    rows.append(_bridge_row("Ofgem Annex 3 — BSUoS", "Balancing / BSUoS recovery", historical_bsuos, candidate,
        status, redispatch_volume_intensity, "gross redispatch MWh per MWh model demand",
        "calibration gross redispatch intensity and BSUoS allowance; future cases also need settlement outturn, supplier sales and allocation",
        "Projected BSUoS allowance = calibration allowance x projected gross PyPSA redispatch volume per demand MWh / calibration intensity. Modelled bid/offer cost is retained only as a diagnostic because it excludes balancing services and settlement recovery."))

    if _has_component(register, "GB-average regional electricity losses"):
        losses = assumptions.get("losses", {})
        historical_losses = _benchmark("GB-average regional electricity losses")
        loss_rate = signal_values.get("transmission_i2r_loss_proxy_rate")
        loss_signal_unit = "resistance-weighted I²R proxy MWh per MWh model demand"
        loss_note = (
            "Resistance-weighted flow-squared transmission-loss proxy, calibrated to the published allowance. "
            "It excludes distribution and regional supplier loss factors, so it is used only as an index."
        )
        if loss_rate is None:
            loss_rate = signal_values.get("transmission_loss_rate")
            loss_signal_unit = "represented AC MWh losses per MWh model demand"
            loss_note = (
                "Represented AC transmission losses are an operational index only. The published allowance also reflects "
                "distribution and regional supplier-loss factors, so modelled physical losses are never passed through directly."
            )
        candidate, loss_status = _indexed_allowance(
            historical_losses, losses, loss_rate, "calibration_transmission_loss_rate"
        )
        if candidate is None and _value(losses, "assumed_multiplier") is not None:
            candidate = historical_losses * _value(losses, "assumed_multiplier")
            loss_status = "assumption_held_baseline"
        rows.append(_bridge_row(
            "GB-average regional electricity losses", "Loss-factor recovery",
            historical_losses, candidate, loss_status, loss_rate,
            loss_signal_unit,
            "calibration resistance-weighted I²R proxy rate; future supplier loss factors, billable supplier sales and allocation",
            loss_note,
        ))

    if register["component"].astype(str).eq("Imbalance uplift").any():
        imbalance = assumptions.get("imbalance", {})
        imbalance_historical = _benchmark("Imbalance uplift")
        imbalance_rate = _value(imbalance, "calibration_gross_redispatch_mwh_per_mwh")
        same_year_imbalance = bool(imbalance.get("use_current_run_as_calibration_baseline", False))
        if same_year_imbalance and redispatch_volume_intensity is not None:
            imbalance_rate = redispatch_volume_intensity
        imbalance_candidate = (
            imbalance_historical * redispatch_volume_intensity / imbalance_rate
            if imbalance_rate is not None and imbalance_rate > 0 and redispatch_volume_intensity is not None else None
        )
        rows.append(_bridge_row("Imbalance uplift", "Balancing-energy / imbalance recovery", imbalance_historical, imbalance_candidate,
            "calibrated_same_year_volume_index" if same_year_imbalance else ("candidate_unvalidated" if imbalance_candidate is not None else "modelled_signal_pending_settlement_reconciliation"),
            redispatch_volume_intensity, "gross redispatch MWh per MWh model demand",
            "calibration gross redispatch intensity and imbalance uplift; validate against cash-out and balancing settlement outturn",
            "Uses gross redispatch volume per demand MWh as an operational index for the published imbalance allowance; it is separate from the BSUoS index to avoid merging two Ofgem components."))

    capacity = assumptions.get("capacity_market", {})
    clearing = _value(capacity, "auction_clearing_price_gbp_per_kw_year")
    derated = _value(capacity, "contracted_derated_capacity_mw")
    capacity_cost = clearing * derated * 1000 if clearing is not None and derated is not None else None
    candidate = _per_customer(capacity_cost, _value(capacity, "billable_supplier_sales_mwh"), _share(capacity))
    capacity_signal = signal_values.get("wind_solar_residual_peak_mw", signal_values.get("peak_demand_mw"))
    capacity_status = "candidate_unvalidated" if candidate is not None else (
        "modelled_adequacy_pending_auction" if capacity_signal is not None else "input_required"
    )
    if candidate is None and (
        capacity.get("use_current_run_as_calibration_baseline", False)
        or _value(capacity, "calibration_residual_peak_mw") is not None
    ):
        candidate, capacity_status = _indexed_allowance(
            _benchmark("Capacity Market"), capacity, capacity_signal,
            "calibration_residual_peak_mw",
        )
    rows.append(_bridge_row("Capacity Market", "Capacity Market recovery", _benchmark("Capacity Market"), candidate,
        capacity_status, capacity_signal, "MW residual peak demand",
        "auction clearing price, contracted de-rated capacity, supplier sales and allocation",
        "PyPSA residual peak can index an explicitly assumed Capacity Market baseline, but does not calculate auction payments or supplier recovery."))

    for component, key, label in [
        ("Ofgem Annex 3 — TNUoS", "tnuos", "Transmission allowed-revenue recovery"),
        ("Ofgem Annex 3 — DUoS unit charges", "duos_unit", "Distribution volumetric allowed-revenue recovery"),
    ]:
        section = assumptions.get(key, {})
        candidate = volumetric(section)
        transmission_signal = signal_values.get("transmission_capacity_mw") if key == "tnuos" else None
        status = "candidate_unvalidated" if candidate is not None else (
            "modelled_network_pending_allowed_revenue" if transmission_signal is not None else "input_required"
        )
        if candidate is None and key == "tnuos":
            candidate, status = _indexed_allowance(
                _benchmark(component), section, transmission_signal,
                "calibration_transmission_capacity_mw",
            )
        if candidate is None and _value(section, "assumed_multiplier") is not None:
            candidate = _benchmark(component) * _value(section, "assumed_multiplier")
            status = "assumption_held_baseline"
        rows.append(_bridge_row(component, label, _benchmark(component), candidate,
            status, transmission_signal, "MW transmission capacity proxy" if transmission_signal is not None else None,
            "allowed revenue recovered from suppliers, supplier sales and domestic allocation",
            "Network expansion in PyPSA is an input to a regulatory revenue model; flows do not determine consumer network charges."))

    duos_fixed = assumptions.get("duos_fixed", {})
    fixed_revenue = _value(duos_fixed, "allowed_revenue_recovered_from_domestic_customers_gbp")
    accounts = _value(duos_fixed, "domestic_customer_accounts")
    candidate = fixed_revenue / accounts if fixed_revenue is not None and accounts is not None and accounts > 0 else None
    rows.append(_bridge_row("Ofgem Annex 3 — DUoS fixed charges", "Distribution fixed allowed-revenue recovery", _benchmark("Ofgem Annex 3 — DUoS fixed charges"), candidate,
        "candidate_unvalidated" if candidate is not None else "distribution_model_not_represented", None, None,
        "domestic fixed-recovery allowed revenue and domestic customer accounts",
        "This is account/meter recovery, so it must not be divided by total model demand."))

    if _has_component(register, "Ofgem Annex 4 â€” FiT"):
        fit = assumptions.get("fit", {})
        fit_cost = _value(fit, "modelled_recovery_cost_gbp")
        candidate = _per_customer(
            fit_cost, _value(fit, "billable_supplier_sales_mwh"),
            _share(fit),
        )
        fit_dispatch = signal_values.get("fit_dispatch_mwh")
        status = "candidate_unvalidated" if candidate is not None else (
            "modelled_generation_pending_levelisation" if fit_dispatch else "not_represented_in_pypsa"
        )
        rows.append(_bridge_row(
            "Ofgem Annex 4 â€” FiT", "FiT levelisation recovery",
            _benchmark("Ofgem Annex 4 â€” FiT"), candidate, status,
            fit_dispatch, "MWh FiT-labelled model dispatch" if fit_dispatch is not None else None,
            "FiT generation/tariff and levelisation cost, supplier sales and domestic allocation",
            "FiT is a supplier-levelised scheme. It can enter once FiT-labelled generation and published tariff/levelisation inputs are available; it is not inferred from merchant renewable dispatch.",
        ))

    result = pd.DataFrame(rows)
    if _value(duos_fixed, "assumed_multiplier") is not None:
        mask = result["bridge"].eq("Distribution fixed allowed-revenue recovery")
        result.loc[mask, "candidate_gbp_per_3100kwh_ex_vat"] = (
            result.loc[mask, "historical_gbp_per_3100kwh_ex_vat"]
            * _value(duos_fixed, "assumed_multiplier")
        )
        result.loc[mask, "evidence_status"] = "assumption_held_baseline"
    missing = result["component"].isin(missing_baselines)
    result.loc[missing, "candidate_gbp_per_3100kwh_ex_vat"] = None
    result.loc[missing, "evidence_status"] = "baseline_component_not_disaggregated"
    result.loc[missing, "required_inputs"] = "A separately reconciled baseline component for this year's cap layout"
    result["component"] = result["component"].replace(baseline_aliases)
    summary = {
        "undisaggregated_baseline_components": sorted(missing_baselines),
        "purpose": "Recovery-bridge evidence register for a historical backtest or future consumer-bill projection",
        "pypsa_modelled_demand_mwh": demand_mwh,
        "pypsa_redispatch_net_cost_gbp": redispatch_cost,
        "pypsa_redispatch_cost_gbp_per_mwh": redispatch_cost / demand_mwh,
        "pypsa_gross_redispatch_mwh": gross_redispatch_mwh,
        "pypsa_gross_redispatch_mwh_per_mwh_demand": redispatch_volume_intensity,
        "bsuos_settlement_outturn_gbp": settlement_cost,
        "bsuos_settlement_outturn_volume_mwh": settlement_volume,
        "bsuos_settlement_outturn_gbp_per_mwh": outturn_rate,
        "bsuos_benchmark_allowance_to_settlement_rate_factor": retail_recovery_factor,
        "candidate_count": int(result["candidate_gbp_per_3100kwh_ex_vat"].notna().sum()),
        "substitution_rule": "No recovery-bridge candidate is substituted into a bill automatically. It must first reconcile to the relevant historical settlement/allowed-revenue outturn, then be validated out of sample.",
        "not_a_bill_pass_through": ["PyPSA redispatch cost", "line flows", "peak demand"],
    }
    return result, summary


def write_bridges(register_path, wholesale_path, market_summary_path, assumptions_path, output_csv, output_summary,
                  signals_path=None, loss_proxy_summary_path=None):
    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    market_summary = json.loads(Path(market_summary_path).read_text(encoding="utf-8"))
    signals = pd.read_csv(signals_path) if signals_path else None
    if loss_proxy_summary_path:
        proxy_summary = json.loads(Path(loss_proxy_summary_path).read_text(encoding="utf-8"))
        proxy_rate = proxy_summary.get("i2r_loss_proxy_rate")
        if proxy_rate is None or float(proxy_rate) < 0:
            raise ValueError("Loss-proxy summary must provide a non-negative i2r_loss_proxy_rate")
        proxy_signal = pd.DataFrame([{
            "metric": "transmission_i2r_loss_proxy_rate",
            "value": float(proxy_rate),
            "unit": "proxy MWh/MWh",
            "treatment": "loss-factor recovery index",
            "note": "Resistance-weighted flow-squared proxy from completed physical balancing networks.",
        }])
        signals = proxy_signal if signals is None else pd.concat([signals, proxy_signal], ignore_index=True)
    rows, summary = build_bridges(pd.read_csv(register_path), pd.read_csv(wholesale_path), market_summary, assumptions, signals)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    loss_proxy = list(snakemake.input.loss_proxy_summary)
    write_bridges(snakemake.input.register_csv, snakemake.input.wholesale_impact_csv, snakemake.input.market_summary_json, snakemake.input.assumptions_json, snakemake.output.bridges_csv, snakemake.output.summary_json, snakemake.input.pypsa_signals_csv, loss_proxy[0] if loss_proxy else None)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True); parser.add_argument("--wholesale-impact", required=True); parser.add_argument("--market-summary", required=True); parser.add_argument("--assumptions", required=True)
    parser.add_argument("--pypsa-signals")
    parser.add_argument("--loss-proxy-summary")
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_bridges(args.register, args.wholesale_impact, args.market_summary, args.assumptions, args.output_csv, args.output_summary, args.pypsa_signals, args.loss_proxy_summary)
