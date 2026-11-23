"""Reconcile mapped PyPSA CfD dispatch with LCCC contract-level outturn.

This is a historical-validation diagnostic. It reports both the legacy
realised-payment-intensity bridge and an independent arithmetic reconstruction
using published daily strike/reference prices and modelled physical dispatch.
It is not a future CfD levy forecast.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _weights(network) -> pd.Series:
    return network.snapshot_weightings.get("generators", pd.Series(1.0, index=network.snapshots))


def prepare_daily_contract_terms(actuals_path: str | Path, year: int) -> pd.DataFrame:
    """Return the published daily settlement terms used by each CfD contract."""
    actual = pd.read_csv(actuals_path)
    actual["settlement_date"] = pd.to_datetime(actual["Settlement_Date"], errors="raise").dt.normalize()
    actual = actual.loc[actual["settlement_date"].dt.year.eq(int(year))].copy()
    for column in (
        "CFD_Generation_MWh", "CFD_Payments_GBP", "Strike_Price_GBP_Per_MWh",
        "Market_Reference_Price_GBP_Per_MWh", "Weighted_IMRP_GBP_Per_MWh",
    ):
        actual[column] = pd.to_numeric(actual[column], errors="coerce")
    # Intermittent contracts settle against the weighted IMRP. Baseload
    # contracts use the market reference price. This reproduces the public LCCC
    # payment arithmetic without using the published payment as an input.
    actual["settlement_reference_price_gbp_per_mwh"] = actual[
        "Market_Reference_Price_GBP_Per_MWh"
    ]
    use_weighted_imrp = actual["Reference_Type"].eq("IMRP") & actual[
        "Weighted_IMRP_GBP_Per_MWh"
    ].notna()
    actual.loc[use_weighted_imrp, "settlement_reference_price_gbp_per_mwh"] = actual.loc[
        use_weighted_imrp, "Weighted_IMRP_GBP_Per_MWh"
    ]
    actual["formula_payment_gbp_per_mwh"] = (
        actual["Strike_Price_GBP_Per_MWh"]
        - actual["settlement_reference_price_gbp_per_mwh"]
    )
    actual["formula_payment_on_actual_generation_gbp"] = (
        actual["CFD_Generation_MWh"] * actual["formula_payment_gbp_per_mwh"]
    )
    return actual[[
        "CfD_ID", "settlement_date", "Reference_Type",
        "Strike_Price_GBP_Per_MWh", "settlement_reference_price_gbp_per_mwh",
        "formula_payment_gbp_per_mwh", "CFD_Generation_MWh",
        "CFD_Payments_GBP", "formula_payment_on_actual_generation_gbp",
    ]]


def _validate_distinct_snapshots(seen: pd.DatetimeIndex, snapshots, source: str) -> pd.DatetimeIndex:
    """Reject repeated solved timestamps before any CfD dispatch is summed."""
    current = pd.DatetimeIndex(snapshots)
    if current.empty:
        raise ValueError(f"Solved CfD network has no snapshots: {source}")
    if current.has_duplicates:
        raise ValueError(f"Solved CfD network has duplicate snapshots: {source}")
    overlap = seen.intersection(current)
    if not overlap.empty:
        raise ValueError(
            "Solved CfD networks have overlapping snapshots; model dispatch would be double counted "
            f"(first overlap: {overlap.min()})"
        )
    return seen.union(current)


def _physical_dispatch(network_paths, mapped_names: list[str]):
    """Return physical dispatch and weather/availability-limited output.

    Keeping available energy separate from dispatched energy is essential for
    the CfD backtest: the former diagnoses renewable profiles and performance
    factors, while their difference is genuine model curtailment.
    """
    import pypsa

    represented = set()
    dispatch_frames = []
    availability_frames = []
    hours_parts = []
    period_hours = 0.0
    seen_snapshots = pd.DatetimeIndex([])
    for path in network_paths:
        network = pypsa.Network(path)
        seen_snapshots = _validate_distinct_snapshots(seen_snapshots, network.snapshots, str(path))
        available = sorted(set(mapped_names) & set(network.generators_t.p.columns))
        represented.update(available)
        if available:
            weights = _weights(network)
            snapshot_hours = pd.Series(
                pd.to_numeric(weights, errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(network.snapshots).tz_localize(None),
            )
            hours_parts.append(snapshot_hours)
            physical_mwh = network.generators_t.p[available].multiply(weights, axis=0)
            physical_mwh.index = pd.DatetimeIndex(physical_mwh.index).tz_localize(None)
            dispatch_frames.append(physical_mwh)

            p_max_pu = network.generators_t.p_max_pu.reindex(
                index=network.snapshots, columns=available
            )
            static_p_max_pu = pd.to_numeric(
                network.generators.reindex(available)["p_max_pu"], errors="coerce"
            ).fillna(1.0)
            p_max_pu = p_max_pu.fillna(static_p_max_pu).fillna(1.0)
            nominal = pd.to_numeric(
                network.generators.reindex(available)["p_nom"], errors="coerce"
            ).fillna(0.0)
            available_mwh = p_max_pu.mul(nominal, axis=1).multiply(weights, axis=0)
            available_mwh.index = pd.DatetimeIndex(available_mwh.index).tz_localize(None)
            availability_frames.append(available_mwh)
            period_hours += float(weights.sum())
    if not dispatch_frames:
        empty = pd.DataFrame(columns=mapped_names)
        return empty, empty, pd.Series(dtype=float), represented, period_hours
    dispatch = pd.concat(dispatch_frames).groupby(level=0).sum().sort_index()
    availability = pd.concat(availability_frames).groupby(level=0).sum().sort_index()
    hours = pd.concat(hours_parts).groupby(level=0).sum().sort_index()
    return (
        dispatch.reindex(columns=mapped_names, fill_value=0.0),
        availability.reindex(columns=mapped_names, fill_value=0.0),
        hours,
        represented,
        period_hours,
    )


def calculate_contract_formula_payments(
    dispatch: pd.DataFrame,
    mapping: pd.DataFrame,
    daily_terms: pd.DataFrame,
) -> pd.DataFrame:
    """Apply daily published contract terms to allocated physical dispatch."""
    daily_parts = []
    for row in mapping.to_dict("records"):
        generator = row["model_generator"]
        if generator not in dispatch.columns:
            continue
        allocated = dispatch[generator] * float(row["allocation_share"])
        daily = allocated.groupby(allocated.index.normalize()).sum()
        daily_parts.append(pd.DataFrame({
            "CfD_ID": row["CfD_ID"],
            "settlement_date": daily.index,
            "modelled_contract_dispatch_mwh": daily.values,
        }))
    if not daily_parts:
        return pd.DataFrame(columns=["CfD_ID"])
    model_daily = pd.concat(daily_parts, ignore_index=True).groupby(
        ["CfD_ID", "settlement_date"], as_index=False
    )["modelled_contract_dispatch_mwh"].sum()
    joined = model_daily.merge(
        daily_terms, on=["CfD_ID", "settlement_date"], how="left", validate="many_to_one"
    )
    joined["modelled_formula_payment_gbp"] = (
        joined["modelled_contract_dispatch_mwh"]
        * joined["formula_payment_gbp_per_mwh"]
    )
    joined["settlement_eligible_dispatch_mwh"] = joined[
        "modelled_contract_dispatch_mwh"
    ].where(joined["formula_payment_gbp_per_mwh"].notna(), 0.0)
    joined["dispatch_without_price_terms_mwh"] = joined[
        "modelled_contract_dispatch_mwh"
    ].where(joined["formula_payment_gbp_per_mwh"].isna(), 0.0)
    return joined.groupby("CfD_ID", as_index=False).agg(
        modelled_payment_independent_formula_gbp=("modelled_formula_payment_gbp", "sum"),
        modelled_contract_settlement_eligible_dispatch_mwh=("settlement_eligible_dispatch_mwh", "sum"),
        modelled_dispatch_without_price_terms_mwh=("dispatch_without_price_terms_mwh", "sum"),
    )


def _settlement_window_mask(index: pd.DatetimeIndex, start, end) -> pd.Series:
    """Restrict model timestamps to a contract's published settlement window.

    LCCC reports settled generation only for the days a contract was live.
    Comparing that against a full modelled period understates every factor
    derived from it, so each contract is measured over its own window.
    """
    mask = pd.Series(True, index=index)
    start = pd.to_datetime(start, errors="coerce")
    end = pd.to_datetime(end, errors="coerce")
    if pd.notna(start):
        mask &= index >= start.normalize()
    if pd.notna(end):
        mask &= index < end.normalize() + pd.Timedelta(days=1)
    return mask


def reconcile(
    network_path: str | Path | list[str | Path],
    mapping_path: str | Path,
    actuals_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    network_paths = list(network_path) if isinstance(network_path, (list, tuple)) else [network_path]
    mapping = pd.read_csv(mapping_path)
    mapping = mapping.loc[mapping["mapping_status"].eq("mapped")].copy()
    if mapping.empty:
        raise ValueError("No mapped CfD contracts to reconcile")
    mapped_names = mapping["model_generator"].unique().tolist()
    dispatch_timeseries, availability_timeseries, hours_timeseries, represented, period_hours = _physical_dispatch(
        network_paths, mapped_names
    )
    missing = sorted(set(mapped_names) - represented)
    if missing:
        raise ValueError(f"Mapped generators are absent from all solved networks: {missing}")
    windowed = []
    for row in mapping.to_dict("records"):
        generator = row["model_generator"]
        mask = _settlement_window_mask(
            dispatch_timeseries.index,
            row.get("first_settlement_date"),
            row.get("last_settlement_date"),
        )
        windowed.append({
            "model_generator_dispatch_mwh": float(dispatch_timeseries.loc[mask, generator].sum()),
            "model_generator_available_mwh": float(availability_timeseries.loc[mask, generator].sum()),
            "settlement_window_hours": float(hours_timeseries.loc[mask].sum()),
        })
    mapping = pd.concat([mapping, pd.DataFrame(windowed, index=mapping.index)], axis=1)
    mapping["modelled_contract_dispatch_mwh"] = mapping["model_generator_dispatch_mwh"] * mapping["allocation_share"]
    mapping["modelled_contract_available_mwh"] = mapping["model_generator_available_mwh"] * mapping["allocation_share"]
    contract = mapping.groupby("CfD_ID", as_index=False).agg(
        cfd_unit=("Name_of_CfD_Unit", "first"),
        technology=("Technology", "first"),
        actual_generation_mwh=("actual_generation_mwh", "first"),
        actual_payment_gbp=("actual_payment_gbp", "first"),
        modelled_contract_physical_dispatch_mwh_full_period=("modelled_contract_dispatch_mwh", "sum"),
        modelled_contract_available_mwh_full_period=("modelled_contract_available_mwh", "sum"),
        mapped_capacity_mw=("mapped_capacity_mw", "sum"),
        contract_capacity_mw=("Maximum_Contract_Capacity_MW", "first"),
        settlement_window_hours=("settlement_window_hours", "first"),
    )
    contract["modelled_contract_curtailed_mwh_full_period"] = (
        contract["modelled_contract_available_mwh_full_period"]
        - contract["modelled_contract_physical_dispatch_mwh_full_period"]
    ).clip(lower=0.0)
    contract["modelled_contract_curtailment_share"] = (
        contract["modelled_contract_curtailed_mwh_full_period"]
        / contract["modelled_contract_available_mwh_full_period"].where(
            contract["modelled_contract_available_mwh_full_period"] > 0
        )
    )
    if period_hours > 0:
        # Capacity factors use each contract's own settlement window, so a
        # part-year contract is not diluted by days it never settled.
        window_hours = contract["settlement_window_hours"].where(
            contract["settlement_window_hours"] > 0, period_hours
        )
        contract["actual_capacity_factor"] = contract["actual_generation_mwh"] / (
            contract["contract_capacity_mw"] * window_hours
        )
        contract["modelled_available_capacity_factor"] = (
            contract["modelled_contract_available_mwh_full_period"]
            / (contract["mapped_capacity_mw"] * window_hours)
        )
        contract["modelled_dispatched_capacity_factor"] = (
            contract["modelled_contract_physical_dispatch_mwh_full_period"]
            / (contract["mapped_capacity_mw"] * window_hours)
        )
    else:
        contract["actual_capacity_factor"] = float("nan")
        contract["modelled_available_capacity_factor"] = float("nan")
        contract["modelled_dispatched_capacity_factor"] = float("nan")
    contract["available_energy_coverage_against_actual"] = (
        contract["modelled_contract_available_mwh_full_period"]
        / contract["actual_generation_mwh"].where(contract["actual_generation_mwh"] > 0)
    )
    contract["official_payment_gbp_per_mwh"] = contract["actual_payment_gbp"] / contract["actual_generation_mwh"]
    calendar_year = int(pd.to_datetime(mapping["first_settlement_date"]).dt.year.mode().iloc[0])
    daily_terms = prepare_daily_contract_terms(actuals_path, calendar_year)
    if not dispatch_timeseries.empty:
        dispatch_days = pd.DatetimeIndex(dispatch_timeseries.index).normalize()
        daily_terms = daily_terms.loc[
            daily_terms["settlement_date"].between(
                dispatch_days.min(), dispatch_days.max()
            )
        ].copy()
    formula = calculate_contract_formula_payments(dispatch_timeseries, mapping, daily_terms)
    actual_formula = daily_terms.groupby("CfD_ID", as_index=False).agg(
        lccc_payment_reproduced_from_terms_gbp=("formula_payment_on_actual_generation_gbp", "sum"),
    )
    contract = contract.merge(formula, on="CfD_ID", how="left").merge(
        actual_formula, on="CfD_ID", how="left"
    )
    contract["modelled_contract_dispatch_mwh"] = contract[
        "modelled_contract_settlement_eligible_dispatch_mwh"
    ]
    contract["modelled_payment_proxy_gbp"] = (
        contract["modelled_contract_dispatch_mwh"]
        * contract["official_payment_gbp_per_mwh"]
    )
    contract["dispatch_ratio_modelled_to_actual"] = (
        contract["modelled_contract_dispatch_mwh"] / contract["actual_generation_mwh"]
    )
    contract["lccc_formula_reproduction_difference_gbp"] = (
        contract["lccc_payment_reproduced_from_terms_gbp"] - contract["actual_payment_gbp"]
    )
    contract["independent_formula_payment_coverage"] = (
        contract["modelled_payment_independent_formula_gbp"] / contract["actual_payment_gbp"]
    )
    result = contract.sort_values("actual_payment_gbp", ascending=False)
    actual_generation = float(result["actual_generation_mwh"].sum())
    actual_payment = float(result["actual_payment_gbp"].sum())
    modelled_generation = float(result["modelled_contract_dispatch_mwh"].sum())
    full_period_physical = float(result["modelled_contract_physical_dispatch_mwh_full_period"].sum())
    full_period_available = float(result["modelled_contract_available_mwh_full_period"].sum())
    full_period_curtailed = float(result["modelled_contract_curtailed_mwh_full_period"].sum())
    modelled_proxy = float(result["modelled_payment_proxy_gbp"].sum())
    modelled_formula = float(result["modelled_payment_independent_formula_gbp"].sum())
    reproduced_actual_payment = float(result["lccc_payment_reproduced_from_terms_gbp"].sum())
    unpriced_dispatch = float(result["modelled_dispatch_without_price_terms_mwh"].sum())
    summary = {
        "treatment": "Historical diagnostic using final physical dispatch. Realised-intensity and independent strike-minus-reference-price results are reported separately.",
        "dispatch_basis": "final balancing/physical generator dispatch",
        "calendar_year": calendar_year,
        "solved_networks_aggregated": len(network_paths),
        "modelled_period_hours": period_hours,
        "mapped_contracts": int(len(result)),
        "actual_generation_mwh_mapped_contracts": actual_generation,
        "modelled_dispatch_mwh_mapped_contracts": modelled_generation,
        "modelled_physical_dispatch_mwh_before_contract_date_filter": full_period_physical,
        "modelled_available_energy_mwh_before_contract_date_filter": full_period_available,
        "modelled_curtailment_mwh_before_contract_date_filter": full_period_curtailed,
        "modelled_curtailment_share_of_available_energy": (
            full_period_curtailed / full_period_available if full_period_available else None
        ),
        "available_energy_coverage_against_actual_mapped_contracts": (
            full_period_available / actual_generation if actual_generation else None
        ),
        "dispatch_coverage_against_actual_mapped_contracts": modelled_generation / actual_generation if actual_generation else None,
        "actual_payment_gbp_mapped_contracts": actual_payment,
        "modelled_dispatch_at_lccc_intensity_gbp": modelled_proxy,
        "payment_coverage_against_actual_mapped_contracts": modelled_proxy / actual_payment if actual_payment else None,
        "lccc_payment_reproduced_from_published_terms_gbp": reproduced_actual_payment,
        "lccc_formula_reproduction_difference_gbp": reproduced_actual_payment - actual_payment,
        "lccc_formula_reproduction_relative_error": (
            (reproduced_actual_payment - actual_payment) / actual_payment
            if actual_payment else None
        ),
        "modelled_payment_independent_formula_gbp": modelled_formula,
        "independent_formula_payment_coverage_against_actual": modelled_formula / actual_payment if actual_payment else None,
        "modelled_dispatch_without_daily_price_terms_mwh": unpriced_dispatch,
        "independent_formula": "Daily allocated physical dispatch x (published indexed strike price - published settlement reference price); weighted IMRP is used for intermittent contracts and market reference price for baseload contracts.",
    }
    return result, summary


def write_reconciliation(network_path, mapping_path, actuals_path, output_csv, output_summary):
    result, summary = reconcile(network_path, mapping_path, actuals_path)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_reconciliation(
        list(snakemake.input.network_nc) if not isinstance(snakemake.input.network_nc, str) else snakemake.input.network_nc,
        snakemake.input.mapping_csv,
        snakemake.input.actuals_csv,
        snakemake.output.reconciliation_csv, snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", nargs="+", required=True); parser.add_argument("--mapping", required=True)
    parser.add_argument("--actuals", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_reconciliation(args.network, args.mapping, args.actuals, args.output_csv, args.output_summary)
