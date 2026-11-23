"""Account for every PyPSA renewable-capacity block in the RO mapping audit.

This deliberately works from the solved PyPSA network outward.  It distinguishes
explicit REPD-derived stations from the model's embedded wind/solar aggregates,
which cannot be assigned to a single RO accreditation without double counting.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


REPD_TO_PYPSA = {
    "Wind Onshore": "wind_onshore",
    "Onshore Wind": "wind_onshore",  # legacy spelling
    "Wind Offshore": "wind_offshore",
    "Offshore Wind": "wind_offshore",  # legacy spelling
    "Solar Photovoltaics": "solar_pv",
    "Small Hydro": "small_hydro",
    "Large Hydro": "large_hydro",
    "Tidal Stream": "tidal_stream",
    "Shoreline Wave": "shoreline_wave",
    "Biomass (dedicated)": "biomass",
    "Biomass (co-firing)": "biomass",
    "Anaerobic Digestion": "biogas",
    "Advanced Conversion Technologies": "advanced_biofuel",
    "Landfill Gas": "landfill_gas",
    "Sewage Sludge Digestion": "sewage_gas",
}
ROC_CARRIERS = set(REPD_TO_PYPSA.values())
EMBEDDED_CARRIERS = {"embedded_wind", "embedded_solar"}


def _model_capacity(network) -> pd.DataFrame:
    generators = network.generators.copy()
    generators["generator"] = generators.index.astype(str)
    generators["carrier"] = generators["carrier"].astype(str)
    generators["p_nom_mw"] = pd.to_numeric(generators["p_nom"], errors="coerce").fillna(0.0)
    generators["support_type"] = generators.get("support_type", pd.Series("", index=generators.index)).fillna("")
    renewable = generators[generators["carrier"].isin(ROC_CARRIERS | EMBEDDED_CARRIERS)].copy()
    renewable["representation"] = "explicit_station_derived"
    renewable.loc[renewable["carrier"].isin(EMBEDDED_CARRIERS), "representation"] = "embedded_or_distributed_aggregate"
    renewable.loc[
        renewable["generator"].str.contains("__agg", regex=False)
        & renewable["representation"].eq("explicit_station_derived"),
        "representation",
    ] = "aggregated_explicit_stations"
    return renewable


def audit_capacity(network, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"repd_technology", "repd_capacity_mw", "mapping_status"}
    if missing := required.difference(mapping.columns):
        raise ValueError(f"Mapping is missing required columns: {sorted(missing)}")
    model = _model_capacity(network)
    model_summary = model.groupby(["carrier", "representation"], as_index=False).agg(
        pypsa_capacity_mw=("p_nom_mw", "sum"),
        generator_count=("generator", "size"),
        roc_labelled_capacity_mw=("p_nom_mw", lambda values: values[model.loc[values.index, "support_type"].eq("ROC")].sum()),
    )
    mapping = mapping.copy()
    mapping["carrier"] = mapping["repd_technology"].map(REPD_TO_PYPSA)
    mapping["repd_capacity_mw"] = pd.to_numeric(mapping["repd_capacity_mw"], errors="coerce").fillna(0.0)
    mapping["high_confidence"] = mapping["mapping_status"].astype(str).str.startswith("high_confidence")
    source_summary = mapping.groupby("carrier", as_index=False).agg(
        register_high_confidence_capacity_mw=("repd_capacity_mw", lambda values: values[mapping.loc[values.index, "high_confidence"]].sum()),
        register_review_capacity_mw=("repd_capacity_mw", lambda values: values[~mapping.loc[values.index, "high_confidence"]].sum()),
        source_site_count=("repd_capacity_mw", "size"),
    )
    total = model.groupby("carrier", as_index=False).agg(pypsa_total_capacity_mw=("p_nom_mw", "sum"))
    explicit = model.loc[model["representation"].ne("embedded_or_distributed_aggregate")].groupby("carrier", as_index=False).agg(
        pypsa_explicit_station_capacity_mw=("p_nom_mw", "sum")
    )
    embedded = model.loc[model["representation"].eq("embedded_or_distributed_aggregate")].groupby("carrier", as_index=False).agg(
        pypsa_embedded_capacity_mw=("p_nom_mw", "sum")
    )
    coverage = total.merge(explicit, on="carrier", how="left").merge(embedded, on="carrier", how="left").merge(source_summary, on="carrier", how="outer")
    for column in coverage.columns:
        if column != "carrier":
            coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0.0)
    coverage["high_confidence_capacity_coverage"] = coverage["register_high_confidence_capacity_mw"] / coverage["pypsa_explicit_station_capacity_mw"].where(coverage["pypsa_explicit_station_capacity_mw"] > 0)
    coverage["interpretation"] = "Directly comparable source capacities only; embedded/distributed aggregate is intentionally excluded from station-level RO matching."
    coverage = coverage.sort_values("pypsa_total_capacity_mw", ascending=False).reset_index(drop=True)
    summary = {
        "purpose": "Account for model renewable capacity before using RO accreditation matches.",
        "pypsa_renewable_capacity_mw": float(model["p_nom_mw"].sum()),
        "pypsa_explicit_station_capacity_mw": float(model.loc[model["representation"].ne("embedded_or_distributed_aggregate"), "p_nom_mw"].sum()),
        "pypsa_embedded_or_distributed_capacity_mw": float(model.loc[model["representation"].eq("embedded_or_distributed_aggregate"), "p_nom_mw"].sum()),
        "high_confidence_register_matched_capacity_mw": float(source_summary["register_high_confidence_capacity_mw"].sum()),
        "rule": "Embedded generators are a demand/production residual and must not be matched to individual RO stations unless their underlying station portfolio is supplied.",
    }
    return coverage, model_summary, summary


def write_audit(network_path, mapping_path, output_csv, output_generators_csv, output_summary):
    import pypsa
    coverage, generators, summary = audit_capacity(pypsa.Network(network_path), pd.read_csv(mapping_path))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_csv, index=False)
    generators.to_csv(output_generators_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_audit(snakemake.input.network_nc, snakemake.input.mapping_csv, snakemake.output.coverage_csv, snakemake.output.generators_csv, snakemake.output.summary_json)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True); parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-generators-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_audit(args.network, args.mapping, args.output_csv, args.output_generators_csv, args.output_summary)
