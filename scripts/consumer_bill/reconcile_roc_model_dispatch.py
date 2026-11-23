"""Reconcile a solved PyPSA ROC proxy against calendar-year Ofgem outturn.

Station-labelled certificates remain the model diagnostic.  Any generation
that is genuinely embedded, aggregated or not represented is then exposed as a
separate technology recovery bridge; it is never hidden inside station labels.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PYOFGEM_TECHNOLOGY = {
    "wind_offshore": "offshore_wind",
    "wind_onshore": "onshore_wind",
    "embedded_wind": "onshore_wind",
    "solar_pv": "solar_pv",
    "large_hydro": "hydro",
    "small_hydro": "hydro",
    "tidal_stream": "tidal",
    # Ofgem reports tidal separately. Wave must not be folded into it merely
    # because both technologies can receive high ROC bands.
    "shoreline_wave": "wave",
    "biomass": "fuelled",
    "waste_to_energy": "fuelled",
    "advanced_biofuel": "fuelled",
    "biogas": "fuelled",
    "landfill_gas": "landfill_gas",
    "sewage_gas": "sewage_gas",
}

OFGEM_TO_MODEL_TECHNOLOGY = {
    "Fuelled": "fuelled",
    "Fuelled with a DNC of 50kW or less": "fuelled",
    "Hydro with a DNC greater than 20 MW": "hydro",
    "Hydro with a DNC of 20MW or less": "hydro",
    "Micro hydro": "hydro",
    "Landfill gas": "landfill_gas",
    "Offshore wind": "offshore_wind",
    "Offshore wind - demonstration turbines": "offshore_wind",
    "Offshore wind - floating turbines": "offshore_wind",
    "Onshore wind": "onshore_wind",
    "Photovoltaic": "solar_pv",
    "Sewage gas": "sewage_gas",
    "Tidal stream": "tidal",
}


def normalise_reference(reference: pd.DataFrame) -> pd.DataFrame:
    """Accept the detailed Ofgem calendar output or the legacy audit schema."""
    if {"technology", "rocs_issued"}.issubset(reference.columns):
        result = reference.copy()
        result["ofgem_eligible_generation_mwh"] = pd.to_numeric(
            result.get("ofgem_eligible_generation_mwh"), errors="coerce"
        )
        return result
    required = {"Technology", "ofgem_roc_certificates", "ofgem_eligible_generation_mwh"}
    if missing := required.difference(reference.columns):
        raise ValueError(f"Reference is missing required columns: {sorted(missing)}")
    result = reference.copy()
    result["technology"] = result["Technology"].map(OFGEM_TO_MODEL_TECHNOLOGY)
    result = result.loc[result["technology"].notna()].copy()
    return result.groupby("technology", as_index=False).agg(
        rocs_issued=("ofgem_roc_certificates", "sum"),
        ofgem_eligible_generation_mwh=("ofgem_eligible_generation_mwh", "sum"),
    )


def _weights(network) -> pd.Series:
    if "generators" in network.snapshot_weightings:
        return network.snapshot_weightings["generators"]
    return pd.Series(1.0, index=network.snapshots)


def _validate_distinct_network_snapshots(networks) -> None:
    """Reject overlapping solved periods before annual certificate aggregation.

    A repeated monthly artifact can otherwise double count dispatch while still
    producing a plausible-looking technology total.  This guard intentionally
    operates on the actual solved snapshots as well as any external manifest.
    """
    seen = pd.DatetimeIndex([])
    for position, network in enumerate(networks, start=1):
        snapshots = pd.DatetimeIndex(network.snapshots)
        if snapshots.empty:
            raise ValueError(f"Solved ROC network {position} has no snapshots")
        if snapshots.has_duplicates:
            raise ValueError(f"Solved ROC network {position} has duplicate snapshots")
        overlap = seen.intersection(snapshots)
        if not overlap.empty:
            raise ValueError(
                "Solved ROC networks have overlapping snapshots; annual dispatch "
                f"would be double counted (first overlap: {overlap.min()})"
            )
        seen = seen.union(snapshots)


def modelled_roc_certificates(network) -> pd.DataFrame:
    """Return certificate proxy by Ofgem technology, retaining unlabeled candidates."""
    weights = _weights(network)
    rows = []
    for name, gen in network.generators.iterrows():
        carrier = str(gen.get("carrier", ""))
        technology = PYOFGEM_TECHNOLOGY.get(carrier)
        if technology is None or name not in network.generators_t.p.columns:
            continue
        dispatch_mwh = float((network.generators_t.p[name] * weights).sum())
        support = gen.get("support_type", "")
        banding = pd.to_numeric(pd.Series([gen.get("ro_banding")]), errors="coerce").iloc[0]
        labelled = support == "ROC" and pd.notna(banding) and banding > 0
        source = str(gen.get("roc_label_source", "") or "")
        direct_labelled = labelled and source == "ofgem_calendar_certificate_outturn"
        eligible_share = pd.to_numeric(
            pd.Series([gen.get("roc_eligible_share", 1.0)]), errors="coerce"
        ).fillna(1.0).iloc[0]
        eligible_share = float(min(1.0, max(0.0, eligible_share)))
        eligible_dispatch = dispatch_mwh * eligible_share if labelled else 0.0
        rows.append({
            "technology": technology,
            "carrier": carrier,
            "generator": name,
            "support_type": support if pd.notna(support) else "",
            "ro_banding": float(banding) if pd.notna(banding) else None,
            "roc_eligible_share": eligible_share,
            "dispatch_mwh": dispatch_mwh,
            "labelled_roc_dispatch_mwh": eligible_dispatch,
            "labelled_roc_certificates": eligible_dispatch * float(banding) if labelled else 0.0,
            "direct_roc_dispatch_mwh": dispatch_mwh * eligible_share if direct_labelled else 0.0,
            "direct_roc_certificates": dispatch_mwh * eligible_share * float(banding) if direct_labelled else 0.0,
            "unlabelled_roc_compatible_dispatch_mwh": dispatch_mwh - eligible_dispatch,
        })
    return pd.DataFrame(rows)


def reconcile(network, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    reference = normalise_reference(reference)
    model = modelled_roc_certificates(network)
    aggregate = model.groupby("technology", as_index=False).agg(
        modelled_labelled_roc_dispatch_mwh=("labelled_roc_dispatch_mwh", "sum"),
        modelled_labelled_roc_certificates=("labelled_roc_certificates", "sum"),
        unlabelled_roc_compatible_dispatch_mwh=("unlabelled_roc_compatible_dispatch_mwh", "sum"),
        strict_direct_roc_dispatch_mwh=("direct_roc_dispatch_mwh", "sum"),
        strict_direct_roc_certificates=("direct_roc_certificates", "sum"),
    ) if not model.empty else pd.DataFrame(columns=["technology", "modelled_labelled_roc_dispatch_mwh", "modelled_labelled_roc_certificates", "unlabelled_roc_compatible_dispatch_mwh"])
    result = reference.merge(aggregate, on="technology", how="outer")
    result["rocs_issued"] = pd.to_numeric(result["rocs_issued"], errors="coerce")
    result["reference_available"] = result["rocs_issued"].notna()
    for column in ["modelled_labelled_roc_dispatch_mwh", "modelled_labelled_roc_certificates", "unlabelled_roc_compatible_dispatch_mwh", "strict_direct_roc_dispatch_mwh", "strict_direct_roc_certificates"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result["certificate_difference"] = (result["strict_direct_roc_certificates"] - result["rocs_issued"]).where(result["reference_available"])
    result["labelled_coverage_of_ofgem_issue"] = result["strict_direct_roc_certificates"] / result["rocs_issued"].where(result["rocs_issued"] > 0)
    result["capacity_overlay_certificate_difference"] = (result["modelled_labelled_roc_certificates"] - result["rocs_issued"]).where(result["reference_available"])
    result["capacity_overlay_coverage_of_ofgem_issue"] = result["modelled_labelled_roc_certificates"] / result["rocs_issued"].where(result["rocs_issued"] > 0)
    result["ofgem_eligible_generation_mwh"] = pd.to_numeric(
        result.get("ofgem_eligible_generation_mwh"), errors="coerce"
    )
    result["ofgem_effective_roc_band"] = result["rocs_issued"] / result["ofgem_eligible_generation_mwh"].where(result["ofgem_eligible_generation_mwh"] > 0)
    result["aggregate_recovery_bridge_roc_certificates"] = (
        result["rocs_issued"] - result["strict_direct_roc_certificates"]
    ).clip(lower=0.0).where(result["reference_available"])
    result["modelled_overcoverage_roc_certificates"] = (
        result["strict_direct_roc_certificates"] - result["rocs_issued"]
    ).clip(lower=0.0).where(result["reference_available"])
    result["residual_rocs_per_unlabelled_model_mwh"] = (
        result["aggregate_recovery_bridge_roc_certificates"]
        / result["unlabelled_roc_compatible_dispatch_mwh"].where(
            result["unlabelled_roc_compatible_dispatch_mwh"] > 0
        )
    )
    result["unlabelled_model_roc_potential_at_ofgem_band"] = (
        result["unlabelled_roc_compatible_dispatch_mwh"] * result["ofgem_effective_roc_band"]
    )
    result["calibrated_eligible_share_of_unlabelled_dispatch"] = (
        result["aggregate_recovery_bridge_roc_certificates"]
        / result["unlabelled_model_roc_potential_at_ofgem_band"].where(
            result["unlabelled_model_roc_potential_at_ofgem_band"] > 0
        )
    ).clip(lower=0.0, upper=1.0)
    result["recovered_from_unlabelled_model_roc_certificates"] = (
        result["unlabelled_model_roc_potential_at_ofgem_band"]
        * result["calibrated_eligible_share_of_unlabelled_dispatch"]
    ).fillna(0.0)
    result["external_representation_bridge_roc_certificates"] = (
        result["aggregate_recovery_bridge_roc_certificates"]
        - result["recovered_from_unlabelled_model_roc_certificates"]
    )
    result["reconciled_roc_certificates"] = (
        result["strict_direct_roc_certificates"]
        + result["recovered_from_unlabelled_model_roc_certificates"]
        + result["external_representation_bridge_roc_certificates"]
    ).where(result["reference_available"])
    result["comparison_note"] = (
        "Strict-direct result includes only calendar-year, high-confidence station labels; capacity-overlay result includes the full Ofgem-DNC-constrained ROC layer. "
        "Unlabelled explicit and embedded generation remains visible in the recovery bridge; "
        "the bridge is a historical calibration, not an independent validation result."
    )
    result = result.sort_values("rocs_issued", ascending=False).reset_index(drop=True)
    summary = {
        "comparison_status": "calendar_aligned_station_labels_with_explicit_recovery_bridge",
        "dispatch_basis": "final balancing/physical generator dispatch",
        "reference_period": "Ofgem output months selected by the upstream outturn rule",
        "reference_scope": "Great Britain",
        "ofgem_rocs_issued": float(result.loc[result["reference_available"], "rocs_issued"].sum()),
        "modelled_labelled_roc_certificates": float(result.loc[result["reference_available"], "modelled_labelled_roc_certificates"].sum()),
        "strict_direct_roc_certificates": float(result.loc[result["reference_available"], "strict_direct_roc_certificates"].sum()),
        "unbenchmarked_modelled_roc_certificates": float(result.loc[~result["reference_available"], "modelled_labelled_roc_certificates"].sum()),
        "unlabelled_roc_compatible_dispatch_mwh": float(result["unlabelled_roc_compatible_dispatch_mwh"].sum()),
        "aggregate_recovery_bridge_roc_certificates": float(result["aggregate_recovery_bridge_roc_certificates"].sum()),
        "modelled_overcoverage_roc_certificates": float(result["modelled_overcoverage_roc_certificates"].sum()),
        "recovered_from_unlabelled_model_roc_certificates": float(result["recovered_from_unlabelled_model_roc_certificates"].sum()),
        "external_representation_bridge_roc_certificates": float(result["external_representation_bridge_roc_certificates"].sum()),
        "reconciled_roc_certificates": float(result["reconciled_roc_certificates"].sum()),
        "recovery_bridge_is_calibration": True,
        "reconciled_total_note": "Positive technology shortfalls are bridged; technology overcoverage is retained separately and is never cancelled by a negative bridge.",
        "primary_validation_metric": "direct_model_coverage_of_ofgem_issue",
        "future_use_rule": "Project station-labelled cohorts directly; use recovery factors only for the matching aggregate representation and expire historic ROC cohorts explicitly.",
    }
    direct = summary["strict_direct_roc_certificates"]
    official = summary["ofgem_rocs_issued"]
    summary["direct_model_difference_roc_certificates"] = direct - official
    summary["direct_model_coverage_of_ofgem_issue"] = direct / official if official else None
    overlay = summary["modelled_labelled_roc_certificates"]
    summary["capacity_overlay_difference_roc_certificates"] = overlay - official
    summary["capacity_overlay_coverage_of_ofgem_issue"] = overlay / official if official else None
    return result, summary


def reconcile_networks(networks, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reconcile a calendar year assembled from independently solved periods."""
    _validate_distinct_network_snapshots(networks)
    model_parts = [modelled_roc_certificates(network) for network in networks]
    model = pd.concat(model_parts, ignore_index=True) if model_parts else pd.DataFrame()

    # Reuse the single-network reconciliation without manufacturing a merged
    # PyPSA file: only generator metadata and weighted dispatch are required.
    reference = normalise_reference(reference)
    aggregate = model.groupby("technology", as_index=False).agg(
        modelled_labelled_roc_dispatch_mwh=("labelled_roc_dispatch_mwh", "sum"),
        modelled_labelled_roc_certificates=("labelled_roc_certificates", "sum"),
        unlabelled_roc_compatible_dispatch_mwh=("unlabelled_roc_compatible_dispatch_mwh", "sum"),
        strict_direct_roc_dispatch_mwh=("direct_roc_dispatch_mwh", "sum"),
        strict_direct_roc_certificates=("direct_roc_certificates", "sum"),
    ) if not model.empty else pd.DataFrame(columns=[
        "technology", "modelled_labelled_roc_dispatch_mwh",
        "modelled_labelled_roc_certificates", "unlabelled_roc_compatible_dispatch_mwh",
        "strict_direct_roc_dispatch_mwh", "strict_direct_roc_certificates",
    ])

    # Build a minimal synthetic object whose per-generator rows reproduce the
    # already aggregated technology totals, then use the established bridge.
    synthetic = SimpleNamespace()
    synthetic.snapshots = pd.Index([pd.Timestamp("2020-01-01")])
    synthetic.snapshot_weightings = pd.DataFrame(
        {"generators": [1.0]}, index=synthetic.snapshots
    )
    rows = []
    dispatch = {}
    for row in aggregate.to_dict("records"):
        carrier = next(
            (key for key, value in PYOFGEM_TECHNOLOGY.items() if value == row["technology"]),
            None,
        )
        if carrier is None:
            continue
        labelled_dispatch = row["modelled_labelled_roc_dispatch_mwh"]
        labelled_certificates = row["modelled_labelled_roc_certificates"]
        direct_dispatch = row["strict_direct_roc_dispatch_mwh"]
        direct_certificates = row["strict_direct_roc_certificates"]

        # Preserve the direct-evidence subset separately.  Previously the
        # synthetic annual network retained only the full capacity-overlay
        # total, so ``reconcile`` quite correctly reported no direct labels.
        # The two ROC records below are mutually exclusive and sum exactly to
        # the aggregated labelled annual dispatch and certificates.
        if direct_dispatch > 0:
            direct_name = f"annual_{row['technology']}_strict_direct_roc"
            rows.append({
                "name": direct_name, "carrier": carrier, "support_type": "ROC",
                "ro_banding": direct_certificates / direct_dispatch,
                "roc_label_source": "ofgem_calendar_certificate_outturn",
            })
            dispatch[direct_name] = [direct_dispatch]

        overlay_dispatch = labelled_dispatch - direct_dispatch
        overlay_certificates = labelled_certificates - direct_certificates
        if overlay_dispatch > 0:
            roc_name = f"annual_{row['technology']}_capacity_overlay_roc"
            rows.append({
                "name": roc_name, "carrier": carrier, "support_type": "ROC",
                "ro_banding": overlay_certificates / overlay_dispatch,
                "roc_label_source": "ofgem_capacity_overlay_aggregate",
            })
            dispatch[roc_name] = [overlay_dispatch]
        unlabelled_name = f"annual_{row['technology']}_unlabelled"
        rows.append({
            "name": unlabelled_name, "carrier": carrier, "support_type": "", "ro_banding": 0.0,
        })
        dispatch[unlabelled_name] = [row["unlabelled_roc_compatible_dispatch_mwh"]]
    synthetic.generators = pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame()
    synthetic.generators_t = type("GeneratorTimeSeries", (), {})()
    synthetic.generators_t.p = pd.DataFrame(dispatch, index=synthetic.snapshots)
    result, summary = reconcile(synthetic, reference)
    summary["solved_networks_aggregated"] = len(networks)
    return result, summary


def write_reconciliation(network_path, reference_path, output_csv, output_summary):
    import pypsa
    network_paths = list(network_path) if isinstance(network_path, (list, tuple)) else [network_path]
    networks = [pypsa.Network(path) for path in network_paths]
    reference = pd.read_csv(reference_path)
    result, summary = (
        reconcile_networks(networks, reference)
        if len(networks) > 1
        else reconcile(networks[0], reference)
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    network_input = snakemake.input.network_nc
    write_reconciliation(
        list(network_input) if not isinstance(network_input, str) else network_input,
        snakemake.input.reference_csv,
        snakemake.output.reconciliation_csv,
        snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", nargs="+", required=True); parser.add_argument("--reference", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_reconciliation(args.network, args.reference, args.output_csv, args.output_summary)
