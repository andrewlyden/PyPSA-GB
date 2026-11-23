"""Create a transparent evidence/review register for offshore landing mappings."""

import json
from pathlib import Path

import pandas as pd

from scripts.utilities.network_io import load_network


def audit_offshore_landing_mapping(network_path, mapping_path, output_csv, output_json, scenario_id):
    """Verify landing buses and label the evidence status of every mapping row.

    The registry intentionally distinguishes an ETYS bus-location sensitivity
    from external project-level connection evidence. It must not manufacture a
    project claim where the input mapping supplies none.
    """
    network = load_network(network_path)
    mapping = pd.read_csv(mapping_path)
    required = {"carrier", "source_bus", "landing_bus", "source"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Offshore landing mapping lacks columns: {sorted(missing)}")
    if mapping.duplicated(["carrier", "source_bus"]).any():
        raise ValueError("Offshore landing mapping has duplicate carrier/source rows")
    unknown = sorted(set(mapping["landing_bus"]) - set(network.buses.index))
    if unknown:
        raise ValueError(f"Offshore landing mapping references unknown ETYS buses: {unknown}")

    result = mapping.copy()
    result["source_bus_exists_in_network"] = result["source_bus"].isin(network.buses.index)
    result["landing_bus_voltage_kV"] = result["landing_bus"].map(network.buses["v_nom"])
    result["zero_displacement_mapping"] = result["source_bus"].eq(result["landing_bus"])
    external_reference = result.get("evidence_reference", pd.Series("", index=result.index))
    has_external_reference = external_reference.fillna("").astype(str).str.strip().ne("")
    result["evidence_status"] = "sensitivity_only_requires_project_evidence"
    result.loc[has_external_reference, "evidence_status"] = "external_reference_declared_review_required"
    result["project_level_claim_allowed"] = False
    result["review_action"] = "obtain_project_or_OFTO_connection_evidence"
    result.loc[result["zero_displacement_mapping"], "review_action"] = (
        "confirm_direct_onshore_connection_or_remove_artificial_export_link"
    )

    output_csv = Path(output_csv)
    output_json = Path(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    summary = {
        "scenario": scenario_id,
        "mapping_rows": int(len(result)),
        "landing_buses": int(result["landing_bus"].nunique()),
        "zero_displacement_mappings": int(result["zero_displacement_mapping"].sum()),
        "rows_with_external_reference": int(has_external_reference.sum()),
        "rows_requiring_project_or_OFTO_evidence": int((~has_external_reference).sum()),
        "project_level_claim_allowed": False,
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    audit_offshore_landing_mapping(
        snakemake.input.network,
        snakemake.input.mapping_csv,
        snakemake.output.review_csv,
        snakemake.output.summary_json,
        snakemake.wildcards.scenario,
    )
