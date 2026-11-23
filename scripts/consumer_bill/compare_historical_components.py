"""Annual Ofgem/model comparison with explicit recovery and validation boundaries.

The hybrid is an annual benchmark sensitivity. No national model estimate is
silently allocated to individual regions or tariff periods. Missing estimates
remain missing in the model column, with their Ofgem value retained in the
hybrid and identified as an outstanding forecasting dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _key(value):
    text = str(value)
    for _ in range(2):
        try:
            text = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    return " ".join(text.split())


def _finite(value):
    return value is not None and pd.notna(value) and np.isfinite(float(value))


def _family(component, group):
    name = component.lower()
    if "cfd" in name:
        return "cfd"
    if name.endswith(" ro"):
        return "ro"
    if name.endswith(" fit"):
        return "fit"
    if "bsuos" in name:
        return "bsuos"
    if "imbalance" in name:
        return "imbalance"
    if "loss" in name:
        return "losses"
    if "capacity market" in name:
        return "capacity"
    if group == "network":
        return "network"
    if group == "hedged_wholesale":
        return "wholesale"
    if group == "hedge_uplifts":
        return "procurement_adders"
    if group in {"margin", "headroom"}:
        return "retail_formula"
    return "external"


GAPS = {
    "cfd": "Eligible contract coverage, strike/reference prices, levy timing and eligible supplier sales.",
    "ro": "Eligible generation, certificate banding, obligation year, recycling and supplier sales.",
    "fit": "FiT cohort generation, tariffs, levelisation and supplier sales.",
    "bsuos": "Balancing services and settlement recovery beyond redispatch; independently validated activity-to-cost conversion.",
    "imbalance": "Forecast error, cash-out and settlement exposure beyond perfect-foresight redispatch.",
    "losses": "Transmission and distribution loss factors; zero losses in a lossless LP are not an estimate.",
    "capacity": "De-rated capacity procurement, auction prices and recovery; residual peak alone is not a payment.",
    "network": "Network investment, allowed revenue, supplier/account allocation and charging rules; topology/flows alone do not set tariffs.",
    "wholesale": "Spot-to-hedged-procurement method needs validation across historical years.",
    "procurement_adders": "Forward/spot basis, shaping, hedging and transaction-cost assumptions.",
    "retail_formula": "Recalculate applicable regulatory margin/headroom formula on changed cost bases for future years.",
    "external": "Requires a separately sourced future assumption; agreement here does not test PyPSA.",
}


def compare_components(components, regional_bills, *, register=None, signals=None,
                       procurement=None, recovery=None, network_bridge=None,
                       assumptions=None, annual_consumption_kwh=3100.0):
    assumptions = assumptions or {}
    published = regional_bills.loc[regional_bills.result_type.eq("published_tariff_replay")]
    if published[["payment_method", "meter_type"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Annual comparison requires one payment method and meter type")
    total = float(published.bill_gbp_inc_vat.mean())
    subtotal = float(published.bill_gbp_ex_vat.mean())
    vat_rate = (total / subtotal - 1) if subtotal else 0.0
    scale = float(annual_consumption_kwh) / 3100.0
    # Archived detailed registers contain benchmark amounts, without a fixed /
    # variable decomposition. Scaling their fixed charges would be misleading.
    if register is not None and annual_consumption_kwh != 3100:
        raise ValueError("Detailed comparison register requires its 3100 kWh benchmark")
    if register is None:
        source = components.loc[components.result_type.eq("published_tariff_replay")]
        register = source.groupby(["component_label", "category"], as_index=False).amount_gbp.sum()
        register = register.rename(columns={"component_label": "component", "category": "stack_group",
                                            "amount_gbp": "annual_gbp_per_3100kwh_ex_vat"})
        register.annual_gbp_per_3100kwh_ex_vat /= len(published)
    required = {"component", "stack_group", "annual_gbp_per_3100kwh_ex_vat"}
    if missing := required.difference(register.columns):
        raise ValueError(f"Comparison register missing {sorted(missing)}")
    frame = register.copy()
    frame["component"] = frame.component.map(_key)
    if frame.component.duplicated().any():
        raise ValueError("Comparison register contains duplicate components")
    frame = frame.loc[~frame.stack_group.isin(["vat", "tax"])].copy()
    frame["ofgem_gbp_ex_vat"] = pd.to_numeric(frame.annual_gbp_per_3100kwh_ex_vat, errors="raise")
    if not np.isfinite(frame.ofgem_gbp_ex_vat).all():
        raise ValueError("Ofgem comparison amounts must be finite")
    frame["family"] = [_family(c, g) for c, g in zip(frame.component, frame.stack_group)]
    # A coarser Ofgem release must not make impacted components disappear.
    # These diagnostic rows have no separate bill amount and are not summed.
    for family, label in {"bsuos": "BSUoS (not separately published in this register)",
                          "imbalance": "Imbalance (not separately published in this register)",
                          "losses": "Losses (not separately published in this register)"}.items():
        if family not in set(frame.family):
            frame = pd.concat([frame, pd.DataFrame([{
                "component": label, "stack_group": "diagnostic", "family": family,
                "ofgem_gbp_ex_vat": np.nan,
            }])], ignore_index=True)
    frame["pypsa_impacted"] = frame.family.ne("external")
    frame["model_gbp_ex_vat"] = np.nan
    frame["calibrated_candidate_gbp_ex_vat"] = np.nan
    frame["model_signal"] = ""
    frame["model_signal_value"] = np.nan
    frame["model_signal_unit"] = ""
    frame["recovery_denominator_mwh"] = np.nan
    frame["method"] = "Published external input retained"
    frame["evidence_status"] = np.where(frame.pypsa_impacted, "model_or_recovery_input_missing", "external_input")
    frame["future_validation"] = frame.family.map(GAPS)
    frame["candidate_source"] = ""

    values = {}
    if signals is not None:
        if signals.metric.duplicated().any():
            raise ValueError("Duplicate model signal metrics")
        values = dict(zip(signals.metric, pd.to_numeric(signals.value, errors="raise")))
    signal_map = {
        "cfd": "cfd_payment_proxy_gbp", "ro": "roc_obligation_proxy_gbp",
        "fit": "fit_dispatch_mwh", "bsuos": "gross_redispatch_mwh_per_mwh_demand",
        "imbalance": "gross_redispatch_mwh_per_mwh_demand", "losses": "transmission_loss_rate",
        "capacity": "wind_solar_residual_peak_mw", "network": "transmission_capacity_mw",
    }
    for family, metric in signal_map.items():
        mask = frame.family.eq(family)
        if family == "network":
            mask &= ~frame.component.str.contains("DUoS", case=False)
        frame.loc[mask, "model_signal"] = metric
        if metric in values:
            frame.loc[mask, "model_signal_value"] = values[metric]
            frame.loc[mask, "model_signal_unit"] = str(signals.loc[signals.metric.eq(metric), "unit"].iloc[0])

    def apply_bridge(bridge, candidate_col, status_col, method_col, source):
        if bridge is None:
            return
        field = "bill_component" if source == "procurement_bridge" else "component"
        needed = {field, candidate_col, status_col, method_col}
        if missing := needed.difference(bridge.columns):
            raise ValueError(f"{source} missing {sorted(missing)}")
        keys = bridge[field].map(_key)
        if keys.duplicated().any():
            raise ValueError(f"{source} has duplicate components")
        for (_, row), name in zip(bridge.iterrows(), keys):
            mask = frame.component.eq(name)
            if not mask.any():
                continue
            if (source == "network_bridge" and frame.loc[mask, "family"].eq("losses").any()
                    and frame.loc[mask, "calibrated_candidate_gbp_ex_vat"].notna().any()):
                continue  # retain the explicit physical-loss calibration
            status = str(row[status_col])
            frame.loc[mask, "evidence_status"] = status
            frame.loc[mask, "method"] = str(row[method_col])
            frame.loc[mask, "candidate_source"] = source
            if source == "recovery_bridge" and _finite(row.get("modelled_signal_value")):
                frame.loc[mask, "model_signal_value"] = float(row.modelled_signal_value)
                frame.loc[mask, "model_signal_unit"] = str(row.get("modelled_signal_unit", ""))
            candidate = row[candidate_col]
            if not _finite(candidate):
                continue
            is_calibrated = any(x in status for x in ("same_year", "calibrat", "held_baseline"))
            # A raw spot backtest explicitly states that it is NOT same-year fitted.
            if "not_same_year_fitted" in status:
                is_calibrated = False
            column = "calibrated_candidate_gbp_ex_vat" if is_calibrated else "model_gbp_ex_vat"
            frame.loc[mask, column] = float(candidate) * scale
            if is_calibrated:
                frame.loc[mask, "future_validation"] = "Same-year calibration / observed allocation / held input: not independent forecast validation. " + frame.loc[mask, "future_validation"]

    apply_bridge(recovery, "candidate_gbp_per_3100kwh_ex_vat", "evidence_status", "method_note", "recovery_bridge")
    apply_bridge(network_bridge, "candidate_allowance_gbp_per_3100kwh_ex_vat", "recovery_status", "model_basis", "network_bridge")
    apply_bridge(procurement, "candidate_allowance_gbp_per_3100kwh_ex_vat", "recovery_status", "model_basis", "procurement_bridge")

    # Use the same explicitly selectable recovery denominator as the future
    # bill method. A system-demand proxy is a sensitivity, not eligible sales.
    method = assumptions.get("support_recovery_method")
    if method not in {None, "modelled_system_demand_proxy", "eligible_supplier_sales"}:
        raise ValueError(f"Unknown support recovery method: {method}")
    if method:
        denominator = (values.get("model_demand_mwh") if method == "modelled_system_demand_proxy"
                       else assumptions.get("eligible_supplier_sales_mwh"))
        if not _finite(denominator) or denominator <= 0:
            raise ValueError("Selected support recovery method needs a positive denominator")
        for family in ("cfd", "ro"):
            mask = frame.family.eq(family)
            if mask.sum() != 1:
                continue  # no inferred split of a published aggregate
            cost = values.get(signal_map[family])
            if not _finite(cost):
                continue
            frame.loc[mask, "model_gbp_ex_vat"] = float(cost) / denominator * annual_consumption_kwh / 1000
            frame.loc[mask, "recovery_denominator_mwh"] = denominator
            frame.loc[mask, "evidence_status"] = "raw_model_support_recovery_sensitivity"
            frame.loc[mask, "candidate_source"] = "model_signals"
            frame.loc[mask, "model_signal"] = signal_map[family]
            frame.loc[mask, "model_signal_value"] = float(cost)
            frame.loc[mask, "model_signal_unit"] = "GBP"
            frame.loc[mask, "method"] = f"Raw model scheme payment / {method} x household MWh; no historical representation top-up"
            if family == "cfd":
                priced, physical = values.get("cfd_priced_dispatch_mwh"), values.get("cfd_total_dispatch_mwh")
                if _finite(priced) and _finite(physical) and physical > 0:
                    frame.loc[mask, "future_validation"] += f" Priced generation covers {priced / physical:.1%} of CfD-labelled dispatch."
            elif family == "ro":
                frame.loc[mask, "future_validation"] += " Raw ROC value excludes historical representation top-ups and the supplier-obligation/recycling bridge."

    # Preserve the difference between an archived annual benchmark and the
    # current period/regional tariff replay as a named scope reconciliation.
    adjustment = subtotal - float(frame.ofgem_gbp_ex_vat.sum())
    frame = pd.concat([frame, pd.DataFrame([{
        "component": "Benchmark-to-tariff scope reconciliation", "stack_group": "reconciliation",
        "ofgem_gbp_ex_vat": adjustment, "family": "external", "pypsa_impacted": False,
        "method": "Published period/regional ex-VAT bill minus detailed annual benchmark subtotal",
        "evidence_status": "external_scope_reconciliation",
        "future_validation": "Includes calendar, geography, rounding and levelisation differences; not a model correction.",
    }])], ignore_index=True)
    frame["ofgem_modelled_gbp_ex_vat"] = frame.model_gbp_ex_vat.combine_first(
        frame.calibrated_candidate_gbp_ex_vat).combine_first(frame.ofgem_gbp_ex_vat)
    frame["hybrid_treatment"] = np.select(
        [frame.model_gbp_ex_vat.notna(), frame.calibrated_candidate_gbp_ex_vat.notna()],
        ["model_candidate", "calibrated_or_observed_allocation"], default="Ofgem_retained")
    frame.loc[frame.ofgem_gbp_ex_vat.isna(), "hybrid_treatment"] = "aggregate_only_no_separate_estimate"
    frame["difference_gbp_ex_vat"] = frame.ofgem_modelled_gbp_ex_vat - frame.ofgem_gbp_ex_vat
    frame["model_difference_gbp_ex_vat"] = frame.model_gbp_ex_vat - frame.ofgem_gbp_ex_vat
    frame["model_difference_percent"] = frame.model_difference_gbp_ex_vat / frame.ofgem_gbp_ex_vat.replace(0, np.nan) * 100
    hybrid_subtotal = float(frame.ofgem_modelled_gbp_ex_vat.sum())
    totals = pd.DataFrame([
        {"basis": "Ofgem published tariff", "ex_vat_gbp": subtotal, "vat_gbp": subtotal * vat_rate, "inc_vat_gbp": total},
        {"basis": "Ofgem/modelled available-component sensitivity", "ex_vat_gbp": hybrid_subtotal,
         "vat_gbp": hybrid_subtotal * vat_rate, "inc_vat_gbp": hybrid_subtotal * (1 + vat_rate)},
    ])
    totals["difference_from_ofgem_gbp"] = totals.inc_vat_gbp - total
    totals["annual_consumption_kwh"] = annual_consumption_kwh
    totals["geography"] = "equal mean of 14 charge-restriction regions"
    counts = {
        "model_candidate_components": int(frame.model_gbp_ex_vat.notna().sum()),
        "calibrated_components": int(frame.hybrid_treatment.eq("calibrated_or_observed_allocation").sum()),
        "impacted_components_without_candidate": int((frame.pypsa_impacted & frame.model_gbp_ex_vat.isna() & frame.calibrated_candidate_gbp_ex_vat.isna()).sum()),
        "forecast_validation_established": False,
        "support_recovery_method": method,
        "interpretation": "Annual benchmark sensitivity. Model columns are blank where recovery inputs are missing. Calibrated agreement and retained Ofgem inputs do not validate forecasting. VAT is recomputed once; regional/period model allocations are not inferred.",
    }
    columns = ["component", "stack_group", "ofgem_gbp_ex_vat", "model_gbp_ex_vat",
               "calibrated_candidate_gbp_ex_vat", "ofgem_modelled_gbp_ex_vat", "difference_gbp_ex_vat",
               "model_difference_gbp_ex_vat", "model_difference_percent", "hybrid_treatment", "pypsa_impacted",
               "model_signal", "model_signal_value", "model_signal_unit", "recovery_denominator_mwh",
               "evidence_status", "method", "candidate_source", "future_validation"]
    return frame.reindex(columns=columns), totals, counts
