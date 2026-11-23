"""Create a station-first historical ROC overlay without double-counting PyPSA.

DUKES/REPD remain the physical fleet.  This overlay starts from the complete
Ofgem certificate inventory and records, for every accredited station, whether
it is already a named model asset, needs a capacity split, is a missing but
source-evidenced asset, or still requires physical linkage.  It is deliberately
an accounting-and-build specification: rows that are not physically linked are
never added to a solved network by this script.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def clean_reference(value: object) -> str:
    return re.sub(r"\.0$", "", str(value).strip()) if pd.notna(value) else ""


def normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    text = text.replace("_", " ").replace("&", " and ")
    text = re.sub(r"\b(limited|ltd|llp|plc|power station|generating station|wind ?farm|solar ?(?:farm|park|pv)|facility|plant|extension|repowering)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def load_generators(path: Path) -> pd.DataFrame:
    generators = pd.read_csv(path, index_col=0, low_memory=False).copy()
    generators["generator"] = generators.index.astype(str)
    generators["p_nom_mw"] = pd.to_numeric(generators["p_nom"], errors="coerce")
    generators["repd_ref_id"] = generators.get("repd_ref_id", pd.Series("", index=generators.index)).map(clean_reference)
    generators["source_site_name"] = generators.get("source_site_name", pd.Series("", index=generators.index)).fillna("").astype(str)
    generators["_aliases"] = generators.apply(
        lambda row: {normalise(row["generator"]), normalise(row["source_site_name"]) } - {""}, axis=1
    )
    return generators


def build_overlay(outturn: pd.DataFrame, registry: pd.DataFrame, pre_solve_mapping: pd.DataFrame, source_evidence: pd.DataFrame, generators: pd.DataFrame, action_overrides: pd.DataFrame | None = None, physical_application: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    registry = registry.copy()
    registry["Accreditation reference"] = registry["Accreditation reference"].map(clean_reference)
    registry["ofgem_dnc_mw"] = pd.to_numeric(
        registry["Station DNC in kW"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ) / 1000.0
    registry = registry.drop_duplicates("Accreditation reference")
    outturn = outturn.copy()
    outturn["Accreditation reference"] = outturn["Accreditation reference"].map(clean_reference)
    stations = outturn.merge(
        registry[["Accreditation reference", "ofgem_dnc_mw", "Station address", "Commissioning date", "Effective date"]],
        on="Accreditation reference", how="left",
    ).drop_duplicates("Accreditation reference")

    mapping = pre_solve_mapping.copy()
    mapping["ro_accreditation_reference"] = mapping["ro_accreditation_reference"].map(clean_reference)
    mapping = mapping.loc[mapping["mapping_status"].astype(str).str.startswith("high_confidence")].copy()
    direct_by_reference = {
        reference: group["generator"].astype(str).tolist()
        for reference, group in mapping.groupby("ro_accreditation_reference")
        if reference
    }

    evidence = source_evidence.copy()
    evidence["Accreditation reference"] = evidence["Accreditation reference"].map(clean_reference)
    evidence = evidence.loc[evidence["mapping_status"].astype(str).str.startswith("high_confidence")].copy()
    evidence_by_reference = {
        reference: group.iloc[0].to_dict()
        for reference, group in evidence.groupby("Accreditation reference")
    }
    generator_by_name = generators.set_index("generator", drop=False)
    generator_by_repd = {
        ref: group.iloc[0].to_dict()
        for ref, group in generators.loc[generators["repd_ref_id"].ne("")].groupby("repd_ref_id")
    }
    generator_by_alias: dict[str, list[dict[str, object]]] = {}
    for row in generators.to_dict("records"):
        for alias in row["_aliases"]:
            generator_by_alias.setdefault(alias, []).append(row)
    overrides_by_reference = {}
    if action_overrides is not None and not action_overrides.empty:
        required = {"Accreditation reference", "capacity_action", "physical_linkage_status", "base_model_generators", "evidence"}
        if missing := required.difference(action_overrides.columns):
            raise ValueError(f"ROC station overlay actions are missing required columns: {sorted(missing)}")
        actions = action_overrides.copy()
        actions["Accreditation reference"] = actions["Accreditation reference"].map(clean_reference)
        if actions["Accreditation reference"].duplicated().any():
            raise ValueError("ROC station overlay actions must have one row per accreditation reference")
        overrides_by_reference = actions.set_index("Accreditation reference").to_dict("index")
    physical_application_by_reference = {}
    if physical_application is not None and not physical_application.empty:
        physical_application = physical_application.copy()
        physical_application["Accreditation reference"] = physical_application[
            "Accreditation reference"
        ].map(clean_reference)
        physical_application_by_reference = physical_application.set_index(
            "Accreditation reference"
        ).to_dict("index")

    rows: list[dict[str, object]] = []
    for station in stations.to_dict("records"):
        reference = station["Accreditation reference"]
        direct_generators = [name for name in direct_by_reference.get(reference, []) if name in generator_by_name.index]
        evidence_row = evidence_by_reference.get(reference)
        linked: list[dict[str, object]] = [generator_by_name.loc[name].to_dict() for name in direct_generators]
        link_method = "direct_ofgem_station_mapping" if linked else ""
        if not linked and evidence_row is not None:
            source = str(evidence_row.get("source", ""))
            source_id = clean_reference(evidence_row.get("source_id"))
            if source.startswith("REPD_") and source_id in generator_by_repd:
                linked = [generator_by_repd[source_id]]
                link_method = "ofgem_source_evidence_repd_id"
            else:
                aliases = {normalise(evidence_row.get("candidate_name")), normalise(evidence_row.get("source_id"))} - {""}
                linked_by_name = {
                    str(row["generator"]): row
                    for alias in aliases
                    for row in generator_by_alias.get(alias, [])
                }
                linked = list(linked_by_name.values())
                if linked:
                    link_method = "ofgem_source_evidence_name"

        model_capacity = sum(float(row.get("p_nom_mw") or 0.0) for row in linked)
        dnc = pd.to_numeric(pd.Series([station.get("ofgem_dnc_mw")]), errors="coerce").iloc[0]
        difference = abs(model_capacity - dnc) / max(model_capacity, dnc) if linked and pd.notna(dnc) and max(model_capacity, dnc) else None
        if linked and difference is not None and (difference <= 0.05 or abs(model_capacity - dnc) <= 1.0):
            action = "use_existing_model_asset"
            physical_status = "linked_station_capacity_consistent"
        elif linked and pd.notna(dnc) and dnc < model_capacity:
            action = "split_existing_asset_roc_and_non_roc"
            physical_status = "linked_requires_roc_eligible_capacity_split"
        elif linked:
            action = "reconcile_existing_asset_capacity"
            physical_status = "linked_requires_capacity_review"
        elif evidence_row is not None:
            action = "add_evidenced_missing_historical_asset"
            physical_status = "source_evidenced_missing_from_base_fleet"
        else:
            action = "resolve_physical_identity_before_network_change"
            physical_status = "ofgem_inventory_only_no_safe_base_link"
        override = overrides_by_reference.get(reference)
        override_evidence = ""
        if override is not None:
            base_names = override["base_model_generators"]
            base_names = "" if pd.isna(base_names) else str(base_names)
            names = [name.strip() for name in base_names.split("|") if name.strip()]
            override_linked = []
            missing = []
            for name in names:
                if name in generator_by_name.index:
                    override_linked.append(generator_by_name.loc[name].to_dict())
                    continue
                candidates = generator_by_alias.get(normalise(name), [])
                if len(candidates) == 1:
                    override_linked.append(candidates[0])
                else:
                    missing.append(name)
            if missing:
                # A capacity-conserving physical split removes its source
                # component.  Follow the replacement components recorded by
                # the pre-solve action report instead of treating that planned
                # removal as a broken station identity.
                application = physical_application_by_reference.get(reference, {})
                replacement_names = [
                    value.strip()
                    for value in str(application.get("created_generators", "")).split("|")
                    if value.strip()
                ]
                replacements = [
                    generator_by_name.loc[value].to_dict()
                    for value in replacement_names
                    if value in generator_by_name.index
                ]
                if str(application.get("execution_status", "")).startswith("applied_") and replacements:
                    override_linked = replacements
                    missing = []
                    link_method = "reviewed_overlay_action_replacement_components"
                else:
                    raise ValueError(f"ROC overlay action for {reference} names absent model generators: {missing}")
            if override_linked:
                linked = override_linked
                model_capacity = sum(float(row.get("p_nom_mw") or 0.0) for row in linked)
                difference = abs(model_capacity - dnc) / max(model_capacity, dnc) if pd.notna(dnc) and max(model_capacity, dnc) else None
                link_method = "reviewed_overlay_action"
            action = str(override["capacity_action"])
            physical_status = str(override["physical_linkage_status"])
            override_evidence = str(override["evidence"])
        physical_application_row = physical_application_by_reference.get(reference, {})
        rows.append({
            "Accreditation reference": reference,
            "Station name": station["Station name"],
            "Technology": station["Technology"],
            "Country": station["Country"],
            "Station address": station.get("Station address", ""),
            "Commissioning date": station.get("Commissioning date", ""),
            "Effective date": station.get("Effective date", ""),
            "ofgem_dnc_mw": dnc,
            "ofgem_roc_certificates": pd.to_numeric(pd.Series([station.get("ofgem_roc_certificates")]), errors="coerce").iloc[0] or 0.0,
            "ofgem_eligible_generation_mwh": pd.to_numeric(pd.Series([station.get("ofgem_eligible_generation_mwh")]), errors="coerce").iloc[0] or 0.0,
            "physical_linkage_status": physical_status,
            "capacity_action": action,
            "base_model_generators": " | ".join(str(row["generator"]) for row in linked),
            "base_model_capacity_mw": model_capacity if linked else None,
            "capacity_relative_difference": difference,
            "link_method": link_method,
            "source_evidence_status": evidence_row.get("mapping_status", "") if evidence_row else "",
            "source_evidence_source": evidence_row.get("source", "") if evidence_row else "",
            "source_evidence_candidate": evidence_row.get("candidate_name", "") if evidence_row else "",
            "reviewed_action_evidence": override_evidence,
            "pre_solve_physical_execution_status": physical_application_row.get("execution_status", ""),
            "pre_solve_physical_created_generators": physical_application_row.get("created_generators", ""),
            "pre_solve_physical_capacity_added_mw": physical_application_row.get("capacity_added_mw", ""),
            "station_inventory_status": "ofgem_authoritative_issuing_station",
        })
    overlay = pd.DataFrame(rows).sort_values(["capacity_action", "ofgem_roc_certificates"], ascending=[True, False]).reset_index(drop=True)
    total_rocs = float(overlay["ofgem_roc_certificates"].sum())
    linked = overlay["physical_linkage_status"].astype(str).str.startswith("linked_")
    source_evidenced_missing = overlay["physical_linkage_status"].eq("source_evidenced_missing_from_base_fleet")
    summary = {
        "method": "Ofgem-first ROC station inventory overlaid on the DUKES/REPD physical fleet",
        "official_issuing_station_count": int(len(overlay)),
        "official_roc_certificates": total_rocs,
        "station_inventory_coverage": 1.0,
        "linked_physical_station_count": int(linked.sum()),
        "linked_physical_roc_certificates": float(overlay.loc[linked, "ofgem_roc_certificates"].sum()),
        "linked_physical_certificate_coverage": float(overlay.loc[linked, "ofgem_roc_certificates"].sum() / total_rocs) if total_rocs else None,
        "source_evidenced_missing_asset_count": int(source_evidenced_missing.sum()),
        "source_evidenced_missing_asset_roc_certificates": float(overlay.loc[source_evidenced_missing, "ofgem_roc_certificates"].sum()),
        "capacity_action_counts": overlay["capacity_action"].value_counts().to_dict(),
        "capacity_action_roc_certificates": overlay.groupby("capacity_action")["ofgem_roc_certificates"].sum().to_dict(),
        "pre_solve_physical_execution_counts": overlay["pre_solve_physical_execution_status"].replace("", "not_reviewed_action").value_counts().to_dict(),
        "safety_rule": "Rows without a physical linkage are complete Ofgem accounting inventory, not PyPSA generators. They must be geolocated/profiled and capacity-reconciled before being added, preventing duplicate capacity and dispatch.",
    }
    return overlay, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outturn", type=Path, required=True); parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pre-solve-mapping", type=Path, required=True); parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--generators", type=Path, required=True); parser.add_argument("--action-overrides", type=Path)
    parser.add_argument("--physical-application", type=Path)
    parser.add_argument("--overlay-output", type=Path, required=True); parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    actions = pd.read_csv(args.action_overrides) if args.action_overrides else None
    physical_application = pd.read_csv(args.physical_application) if args.physical_application else None
    overlay, summary = build_overlay(pd.read_csv(args.outturn), pd.read_csv(args.registry), pd.read_csv(args.pre_solve_mapping), pd.read_csv(args.source_evidence), load_generators(args.generators), actions, physical_application)
    args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(args.overlay_output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
