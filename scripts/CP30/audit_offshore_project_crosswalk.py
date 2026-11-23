"""Audit whether a CP30 landing map has a project and TEC evidence chain.

The current FES source buses are spatial proxies, not project identifiers.  This
script reports that limitation explicitly, while checking that every nominated
landing has a cited project and a matching record in a versioned TEC register.
"""

import json
from pathlib import Path

import pandas as pd


def _as_id_set(value):
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def _project_tec_snapshot(tec, project_ids, year):
    """Return one capacity/status record per project, effective by year-end."""
    tec = tec.copy()
    tec["MW Effective From"] = pd.to_datetime(
        tec["MW Effective From"], format="%Y/%m/%d", errors="coerce"
    )
    cutoff = pd.Timestamp(year=year, month=12, day=31)
    rows = tec.loc[
        tec["Project ID"].astype(str).isin(project_ids)
        & (tec["MW Effective From"].isna() | (tec["MW Effective From"] <= cutoff))
    ].copy()
    if rows.empty:
        return rows
    rows["_sort_date"] = rows["MW Effective From"].fillna(pd.Timestamp("1900-01-01"))
    rows = rows.sort_values("_sort_date").groupby("Project ID", as_index=False).tail(1)
    rows["effective_capacity_mw"] = rows["Cumulative Total Capacity (MW)"].fillna(
        rows["MW Connected"]
    )
    return rows


def audit_offshore_project_crosswalk(
    mapping_path, evidence_path, tec_path, output_csv, output_json, scenario_id, modelled_year
):
    mapping = pd.read_csv(mapping_path)
    evidence = pd.read_csv(evidence_path)
    tec = pd.read_csv(tec_path)
    required_mapping = {"carrier", "source_bus", "landing_bus", "source"}
    required_evidence = {"landing_bus", "confirmed_project", "tec_project_ids", "source_url"}
    required_tec = {
        "Project ID", "Project Name", "Connection Site", "MW Connected",
        "Cumulative Total Capacity (MW)", "MW Effective From", "Project Status",
    }
    for name, frame, required in (
        ("mapping", mapping, required_mapping),
        ("evidence register", evidence, required_evidence),
        ("TEC register", tec, required_tec),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} lacks required columns: {sorted(missing)}")
    if evidence["landing_bus"].duplicated().any():
        raise ValueError("Evidence register must contain exactly one record per landing bus")

    evidence = evidence.copy().set_index("landing_bus")
    evidence_by_landing = evidence.reindex(mapping["landing_bus"])
    result = mapping.copy()
    for column in evidence.columns:
        result[f"target_{column}"] = evidence_by_landing[column].to_numpy()
    result["target_connection_evidence_status"] = result["target_confirmed_project"].notna().map(
        {True: "project_and_target_declared", False: "missing_target_project_evidence"}
    )

    project_ids = set().union(*result["target_tec_project_ids"].map(_as_id_set))
    snapshot = _project_tec_snapshot(tec, project_ids, int(modelled_year))
    snapshot_by_id = snapshot.set_index("Project ID") if not snapshot.empty else pd.DataFrame()

    def tec_details(ids_value):
        ids = _as_id_set(ids_value)
        found = snapshot_by_id.reindex(sorted(ids)).dropna(how="all") if not snapshot.empty else pd.DataFrame()
        if found.empty:
            return pd.Series({
                "tec_project_records_found": 0, "tec_effective_capacity_mw": 0.0,
                "tec_project_statuses": "", "tec_connection_sites": "",
            })
        return pd.Series({
            "tec_project_records_found": int(len(found)),
            "tec_effective_capacity_mw": float(found["effective_capacity_mw"].sum()),
            "tec_project_statuses": "; ".join(sorted(found["Project Status"].dropna().astype(str).unique())),
            "tec_connection_sites": "; ".join(sorted(found["Connection Site"].dropna().astype(str).unique())),
        })

    result = pd.concat(
        [result, result["target_tec_project_ids"].apply(tec_details)], axis=1
    )
    declared_project_id = result.get("project_id", pd.Series("", index=result.index))
    declared_evidence = result.get("project_evidence_reference", pd.Series("", index=result.index))
    result["proxy_project_id_declared"] = declared_project_id.fillna("").astype(str).str.strip().ne("")
    result["proxy_project_evidence_declared"] = declared_evidence.fillna("").astype(str).str.strip().ne("")
    result["proxy_to_project_status"] = "unresolved_no_project_id"
    result.loc[result["proxy_project_id_declared"], "proxy_to_project_status"] = (
        "declared_project_id_requires_source_evidence_review"
    )
    result.loc[
        result["proxy_project_id_declared"] & result["proxy_project_evidence_declared"],
        "proxy_to_project_status",
    ] = "declared_requires_independent_review"
    # A declaration is not validation.  This remains false until reviewers add a
    # source-specific crosswalk and deliberately approve it outside this audit.
    result["project_level_claim_allowed"] = False
    result["review_action"] = "supply_FES_proxy_to_project_crosswalk_with_primary_reference"

    output_csv = Path(output_csv)
    output_json = Path(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    summary = {
        "scenario": scenario_id,
        "modelled_year": int(modelled_year),
        "mapping_rows": int(len(result)),
        "rows_with_target_connection_evidence": int(
            result["target_connection_evidence_status"].eq("project_and_target_declared").sum()
        ),
        "rows_with_matching_tec_project": int((result["tec_project_records_found"] > 0).sum()),
        "rows_with_proxy_to_project_evidence": int(
            result["proxy_project_evidence_declared"].sum()
        ),
        "rows_unresolved_at_proxy_to_project_stage": int(
            result["proxy_to_project_status"].eq("unresolved_no_project_id").sum()
        ),
        "project_level_claim_allowed": False,
        "tec_register": str(tec_path),
        "evidence_register": str(evidence_path),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    audit_offshore_project_crosswalk(
        snakemake.input.mapping_csv,
        snakemake.input.evidence_register,
        snakemake.input.tec_register,
        snakemake.output.audit_csv,
        snakemake.output.summary_json,
        snakemake.wildcards.scenario,
        snakemake.params.modelled_year,
    )
