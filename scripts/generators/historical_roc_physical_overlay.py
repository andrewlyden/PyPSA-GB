"""Apply reviewed, physical 2020 ROC station corrections before dispatch.

The Ofgem ROC register is authoritative for accreditation and DNC, but it is
not a network asset register. This module therefore executes only actions
which preserve an existing asset's physical profile or add a documented phase
at an already modelled station. Every other Ofgem station remains in the
station-first accounting overlay until it has a reviewed bus and availability
profile; this avoids inventing capacity or double-counting dispatch.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
from pandas.errors import PerformanceWarning


SAFE_ACTIONS = {
    "split_existing_asset_roc_and_non_roc",
    "add_missing_phase_and_link_existing_assets",
}


def _clean_reference(value: object) -> str:
    return re.sub(r"\.0$", "", str(value).strip()) if pd.notna(value) else ""


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _resolve_generators(network, names: list[str]) -> list[str]:
    """Resolve reviewed source names to one unambiguous PyPSA component each."""
    aliases: dict[str, list[str]] = {}
    for generator, row in network.generators.iterrows():
        for value in (generator, row.get("source_site_name", "")):
            alias = _normalise_name(value)
            if alias:
                aliases.setdefault(alias, []).append(str(generator))
    resolved = []
    for name in names:
        if name in network.generators.index:
            resolved.append(name)
            continue
        candidates = sorted(set(aliases.get(_normalise_name(name), [])))
        if len(candidates) != 1:
            raise ValueError(
                f"ROC physical overlay cannot uniquely resolve {name!r}; candidates: {candidates}"
            )
        resolved.append(candidates[0])
    return resolved


def _load_dnc_by_reference(registry_path: str | Path) -> dict[str, float]:
    registry = pd.read_csv(registry_path, low_memory=False)
    required = {"Accreditation reference", "Station DNC in kW"}
    if missing := required.difference(registry.columns):
        raise ValueError(f"Ofgem ROC registry is missing columns: {sorted(missing)}")
    registry["Accreditation reference"] = registry["Accreditation reference"].map(_clean_reference)
    registry["dnc_mw"] = pd.to_numeric(
        registry["Station DNC in kW"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ) / 1000.0
    return registry.dropna(subset=["dnc_mw"]).drop_duplicates(
        "Accreditation reference"
    ).set_index("Accreditation reference")["dnc_mw"].to_dict()


def _copy_time_series(network, source: str, target: str) -> None:
    """Copy every source generator time series, including p_max_pu."""
    for _, table in network.generators_t.items():
        if source in table.columns:
            # The network already contains many component columns. Appending a
            # handful of audited split/phase columns is intentional and does
            # not affect the numerical result, so avoid a noisy pandas warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PerformanceWarning)
                table[target] = table[source].copy()


def _add_clone(network, source: str, target: str, p_nom_mw: float, **overrides) -> None:
    """Clone static and dynamic fields, replacing only reviewed attributes."""
    if target in network.generators.index:
        raise ValueError(f"ROC physical overlay would duplicate generator {target!r}")
    attributes = network.generators.loc[source].dropna().to_dict()
    attributes.pop("p_nom_opt", None)  # solved-result field, not an input
    attributes["p_nom"] = float(p_nom_mw)
    attributes.update(overrides)
    network.add("Generator", target, **attributes)
    _copy_time_series(network, source, target)


def _set_roc_identity(network, generators: list[str], reference: str, evidence: str) -> None:
    for name in generators:
        network.generators.at[name, "ro_accreditation_reference"] = reference
        network.generators.at[name, "support_evidence"] = evidence


def _split_existing_asset(network, row: dict[str, object], dnc_mw: float) -> dict[str, object]:
    names = [name.strip() for name in str(row["base_model_generators"]).split("|") if name.strip()]
    names = _resolve_generators(network, names)
    if len(names) != 1:
        raise ValueError(f"ROC split {row['Accreditation reference']} needs exactly one existing generator; got {names}")
    source = names[0]
    original_capacity = float(network.generators.at[source, "p_nom"])
    if not 0.0 < dnc_mw < original_capacity:
        raise ValueError(
            f"ROC split {row['Accreditation reference']} has Ofgem DNC {dnc_mw:.3f} MW, "
            f"which must be between zero and the {original_capacity:.3f} MW source capacity."
        )
    reference = str(row["Accreditation reference"])
    roc_name = f"{source}__ROC_{reference}"
    non_roc_name = f"{source}__non_ROC_residual"
    evidence = f"reviewed_Ofgem_DNC_split:{reference}"
    _add_clone(
        network, source, roc_name, dnc_mw,
        support_type="ROC", ro_accreditation_reference=reference,
        roc_eligible_share=1.0, roc_label_source="reviewed_physical_split_pending_calendar_label",
        support_evidence=evidence,
    )
    _add_clone(
        network, source, non_roc_name, original_capacity - dnc_mw,
        support_type="merchant", ro_banding=0.0, ro_accreditation_reference="",
        roc_eligible_share=0.0, roc_label_source="reviewed_non_roc_residual",
        support_evidence=evidence,
    )
    network.remove("Generator", source)
    return {
        "Accreditation reference": reference, "capacity_action": row["capacity_action"],
        "execution_status": "applied_capacity_conserving_split", "source_generators": source,
        "created_generators": f"{roc_name} | {non_roc_name}", "ofgem_dnc_mw": dnc_mw,
        "model_capacity_before_mw": original_capacity, "model_capacity_after_mw": original_capacity,
        "capacity_added_mw": 0.0,
        "method": "One source profile is copied to ROC and non-ROC capacity shares; total MW and available output are conserved.",
    }


def _add_documented_phase(network, row: dict[str, object], dnc_mw: float) -> dict[str, object]:
    names = [name.strip() for name in str(row["base_model_generators"]).split("|") if name.strip()]
    if not names:
        raise ValueError(f"ROC phase action {row['Accreditation reference']} has no base generator")
    names = _resolve_generators(network, names)
    reference = str(row["Accreditation reference"])
    existing_capacity = float(pd.to_numeric(network.generators.loc[names, "p_nom"], errors="coerce").sum())
    missing_capacity = dnc_mw - existing_capacity
    if missing_capacity <= 0.0:
        raise ValueError(f"ROC phase action {reference} has no positive Ofgem-DNC capacity balance ({dnc_mw:.3f} - {existing_capacity:.3f} MW).")
    source = names[0]
    phase_name = f"{source}__ROC_missing_phase_{reference}"
    evidence = f"reviewed_Ofgem_DNC_phase_balance:{reference}"
    _set_roc_identity(network, names, reference, evidence)
    _add_clone(
        network, source, phase_name, missing_capacity,
        support_type="ROC", ro_accreditation_reference=reference,
        roc_eligible_share=1.0, roc_label_source="reviewed_missing_phase_pending_calendar_label",
        support_evidence=evidence,
        source_site_name=f"{row.get('Station name', source)} (documented missing phase)",
    )
    return {
        "Accreditation reference": reference, "capacity_action": row["capacity_action"],
        "execution_status": "applied_documented_missing_phase", "source_generators": " | ".join(names),
        "created_generators": phase_name, "ofgem_dnc_mw": dnc_mw,
        "model_capacity_before_mw": existing_capacity,
        "model_capacity_after_mw": existing_capacity + missing_capacity,
        "capacity_added_mw": missing_capacity,
        "method": "Adds only the Ofgem-DNC balance using a documented co-located phase and the existing project's weather profile.",
    }


def apply_historical_roc_physical_overlay(network, registry_path, actions_path, config: dict | None, logger):
    """Apply explicitly permitted reviewed actions and return an audit report."""
    config = config or {}
    report_columns = [
        "Accreditation reference", "capacity_action", "execution_status", "source_generators",
        "created_generators", "ofgem_dnc_mw", "model_capacity_before_mw",
        "model_capacity_after_mw", "capacity_added_mw", "method",
    ]
    if not config.get("enabled", False):
        return network, pd.DataFrame(columns=report_columns)
    if not actions_path:
        raise ValueError("historical_roc_physical_overlay is enabled but has no actions_file")
    actions = pd.read_csv(actions_path, low_memory=False)
    required = {"Accreditation reference", "capacity_action", "base_model_generators"}
    if missing := required.difference(actions.columns):
        raise ValueError(f"ROC physical overlay actions are missing columns: {sorted(missing)}")
    actions["Accreditation reference"] = actions["Accreditation reference"].map(_clean_reference)
    if actions["Accreditation reference"].duplicated().any():
        raise ValueError("ROC physical overlay actions must be unique by accreditation reference")
    permitted = set(config.get("permitted_actions", sorted(SAFE_ACTIONS)))
    if unsupported := permitted.difference(SAFE_ACTIONS):
        raise ValueError("ROC physical overlay permits unsafe actions: " + ", ".join(sorted(unsupported)))
    dnc_by_reference = _load_dnc_by_reference(registry_path)
    report_rows = []
    for row in actions.to_dict("records"):
        reference, action = row["Accreditation reference"], str(row["capacity_action"])
        if action not in permitted:
            report_rows.append({
                "Accreditation reference": reference, "capacity_action": action,
                "execution_status": "inventory_only_requires_station_placement_and_profile",
                "method": "Retained in the Ofgem-first accounting overlay; not added to PyPSA without a reviewed physical connection and availability profile.",
            })
            continue
        dnc_mw = dnc_by_reference.get(reference)
        if dnc_mw is None or dnc_mw <= 0:
            raise ValueError(f"ROC physical overlay cannot find a positive Ofgem DNC for {reference}")
        if action == "split_existing_asset_roc_and_non_roc":
            report_rows.append(_split_existing_asset(network, row, dnc_mw))
        elif action == "add_missing_phase_and_link_existing_assets":
            report_rows.append(_add_documented_phase(network, row, dnc_mw))
    report = pd.DataFrame(report_rows).reindex(columns=report_columns)
    applied = report["execution_status"].astype(str).str.startswith("applied_")
    logger.info(
        "Applied %d reviewed physical ROC-overlay actions (%.3f MW added); %d actions remain inventory-only.",
        int(applied.sum()), float(pd.to_numeric(report.loc[applied, "capacity_added_mw"], errors="coerce").fillna(0.0).sum()), int((~applied).sum()),
    )
    return network, report
