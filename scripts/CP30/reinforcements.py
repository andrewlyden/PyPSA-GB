"""Audit the dated CP30 N1 reinforcement register against a model run.

The CP30 workbook lists project codes and delivery dates, but it does not
provide the electrical branch parameters needed to create each project in a
PyPSA network.  ETYS Appendix B remains the electrical source.  This layer
therefore makes the coverage gap explicit: it records every CP30-required N1
project due by the model year and whether a reviewed ETYS representation has
been registered.  It prevents a scenario from being described as a full NESO
network replication when named projects still need branch-level mapping.
"""

from pathlib import Path

import pandas as pd


_REPRESENTED_STATUSES = {"represented_in_etys", "represented_in_cp30_layer"}


def _is_due(value, modelled_year):
    text = str(value).strip()
    if text.lower() == "built":
        return True
    try:
        return int(float(text)) <= int(modelled_year)
    except (TypeError, ValueError):
        return False


def _missing_mapping_metadata(description):
    """Classify an unreviewed CP30 project without inventing topology."""
    text = str(description).lower()
    if any(token in text for token in ("voltage", "reactive", "msc", "transformer")):
        return (
            "outside_thermal_network_scope",
            "cp30_register_only",
            "obtain_operability_or_voltage_representation",
            "not_quantified_by_thermal_model",
        )
    if "power control" in text:
        return (
            "unrepresented_power_flow_control_asset",
            "cp30_register_only",
            "obtain_control_setting_and_etys_component_mapping",
            "may_overstate_thermal_congestion_if_omitted",
        )
    return (
        "unreviewed_transfer_or_connection_asset",
        "cp30_register_only",
        "review_etys_appendix_b_or_project_electrical_data",
        "may_overstate_thermal_congestion_if_omitted_but_effect_is_not_monotonic",
    )


def _nonempty_or_default(value, default):
    """Return ``default`` for merge-created NaNs and blank manifest fields."""
    if pd.isna(value) or not str(value).strip():
        return default
    return value


def _expected_component_records(record):
    """Expand one manifest row into one or more required network assets.

    A CP30 project can comprise several ETYS branches.  Semicolon-separated
    fields retain the one-project-per-manifest-row structure while ensuring an
    audit does not call the project represented after finding only one branch.
    """
    fields = ("expected_component", "expected_bus0", "expected_bus1", "expected_min_nominal")
    values = []
    for field in fields:
        value = record.get(field)
        if pd.isna(value) or not str(value).strip():
            return []
        values.append([item.strip() for item in str(value).split(";")])
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(
            "Manifest expected-component fields must have matching semicolon-separated lengths"
        )
    return [dict(zip(fields, items)) for items in zip(*values)]


def _apply_traceability_metadata(audit):
    """Add explicit coverage and remediation fields to the project audit."""
    audit = audit.copy()
    for column in ("mapping_basis", "evidence_status", "directional_effect_if_missing"):
        if column not in audit.columns:
            audit[column] = ""
    audit["coverage_class"] = ""
    audit["mapping_action"] = ""
    for index, row in audit.iterrows():
        status = str(row["audit_status"])
        if status in _REPRESENTED_STATUSES:
            audit.at[index, "coverage_class"] = "represented_source_backed"
            audit.at[index, "evidence_status"] = _nonempty_or_default(
                row["evidence_status"], "reviewed_etys_or_cp30_component_mapping"
            )
            audit.at[index, "directional_effect_if_missing"] = "not_applicable"
            audit.at[index, "mapping_action"] = "none"
        elif status == "expected_etys_component_missing":
            audit.at[index, "coverage_class"] = "expected_representation_not_found"
            audit.at[index, "evidence_status"] = _nonempty_or_default(
                row["evidence_status"], "expected_etys_component_not_found"
            )
            audit.at[index, "directional_effect_if_missing"] = _nonempty_or_default(
                row["directional_effect_if_missing"],
                "may_overstate_thermal_congestion_if_omitted_but_effect_is_not_monotonic",
            )
            audit.at[index, "mapping_action"] = "review_etys_upgrade_application_and_terminals"
        else:
            coverage, evidence, action, effect = _missing_mapping_metadata(
                row["cp30_description"]
            )
            audit.at[index, "coverage_class"] = coverage
            audit.at[index, "evidence_status"] = _nonempty_or_default(
                row["evidence_status"], evidence
            )
            audit.at[index, "directional_effect_if_missing"] = _nonempty_or_default(
                row["directional_effect_if_missing"], effect
            )
            audit.at[index, "mapping_action"] = action
    return audit


def audit_cp30_reinforcements(
    reference_path, manifest_path, modelled_year, output_path, logger, strict=False,
    applied_projects=None, network=None,
):
    """Write the CP30 N1 dated-reinforcement coverage audit.

    ``strict=True`` is useful once every required project has an approved
    electrical mapping; until then it fails early rather than silently treating
    the date register as a completed topology build.
    """
    reference_path = Path(reference_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not reference_path.is_file():
        raise FileNotFoundError(f"CP30 N1 reference file not found: {reference_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CP30 reinforcement manifest not found: {manifest_path}")

    reference = pd.read_csv(reference_path)
    manifest = pd.read_csv(manifest_path)
    required_reference = {"source_sheet", "Code", "Description", "Latest Delivery Year (updated September 2024)"}
    required_manifest = {"project_code", "representation_status", "evidence"}
    missing_reference = required_reference - set(reference.columns)
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_reference:
        raise ValueError(f"CP30 reference file missing columns: {sorted(missing_reference)}")
    if missing_manifest:
        raise ValueError(f"CP30 reinforcement manifest missing columns: {sorted(missing_manifest)}")

    due = reference[reference["source_sheet"].eq("N1")].copy()
    due["project_code"] = due["Code"].astype(str).str.strip()
    due = due[due["project_code"].ne("")]
    delivery_col = "Latest Delivery Year (updated September 2024)"
    required_col = "Required for CP30"
    if required_col not in due.columns:
        raise ValueError(f"CP30 reference file missing column: {required_col}")
    required = due[required_col].notna()
    due_by_year = due[delivery_col].map(lambda value: _is_due(value, modelled_year))
    # Include every project CP30 says is required even if the September-2024
    # delivery register has slipped it beyond the target year (e.g. AENC/ATNC).
    due = due[required | due_by_year]
    manifest = manifest.copy()
    manifest["project_code"] = manifest["project_code"].astype(str).str.strip()
    if manifest.duplicated("project_code").any():
        raise ValueError("CP30 reinforcement manifest has duplicate project_code rows")

    audit = due[["project_code", "Description", delivery_col, required_col]].merge(
        manifest, on="project_code", how="left", validate="one_to_one"
    )
    audit = audit.rename(columns={
        "Description": "cp30_description",
        delivery_col: "cp30_latest_delivery_year",
    })
    audit["modelled_year"] = int(modelled_year)
    audit["cp30_required"] = audit[required_col].notna()
    audit["delivery_due_by_model_year"] = audit["cp30_latest_delivery_year"].map(
        lambda value: _is_due(value, modelled_year)
    )
    audit["audit_status"] = audit["representation_status"].fillna("unmapped")
    applied_projects = set(applied_projects or ())
    audit.loc[audit["project_code"].isin(applied_projects), "audit_status"] = "represented_in_cp30_layer"

    # EGL1/EGL2 are normal ETYS B-5-1 upgrades rather than accelerated overlay
    # rows.  Validate their realised terminals/capacity if the caller supplies
    # the built network; otherwise retain a deliberately unresolved status.
    expected_columns = {"expected_component", "expected_bus0", "expected_bus1", "expected_min_nominal"}
    if network is not None and expected_columns.issubset(manifest.columns):
        expected = manifest.set_index("project_code")
        for index, row in audit.iterrows():
            code = row["project_code"]
            if code not in expected.index:
                continue
            record = expected.loc[code]
            expected_records = _expected_component_records(record)
            if not expected_records:
                continue
            all_components_found = True
            for expected_record in expected_records:
                component = expected_record["expected_component"].lower()
                table_name = {
                    "link": "links", "line": "lines", "transformer": "transformers",
                }.get(component)
                if table_name is None:
                    raise ValueError(f"Unsupported manifest component type: {component}")
                table = getattr(network, table_name, pd.DataFrame())
                terminals = {
                    expected_record["expected_bus0"], expected_record["expected_bus1"],
                }
                matched = table[
                    table.apply(
                        lambda asset: {str(asset["bus0"]), str(asset["bus1"])} == terminals,
                        axis=1,
                    )
                ] if not table.empty else table
                nominal = "p_nom" if component == "link" else "s_nom"
                required_nominal = float(expected_record["expected_min_nominal"])
                if matched.empty or float(matched[nominal].max()) < required_nominal:
                    all_components_found = False
                    break
            if all_components_found:
                audit.at[index, "audit_status"] = "represented_in_etys"
            elif row["audit_status"] == "represented_in_etys":
                audit.at[index, "audit_status"] = "expected_etys_component_missing"
    audit = _apply_traceability_metadata(audit)
    audit.to_csv(output_path, index=False)

    unresolved = audit[~audit["audit_status"].isin(_REPRESENTED_STATUSES)]
    logger.info(
        "CP30 N1 reinforcement audit: %d projects due by %s; %d are not source-backed "
        "representations (%s).",
        len(audit), modelled_year, len(unresolved), output_path,
    )
    if strict and not unresolved.empty:
        codes = ", ".join(unresolved["project_code"].tolist())
        raise ValueError(
            "CP30 dated-reinforcement strict mode found unresolved projects: " + codes
        )
    return audit
