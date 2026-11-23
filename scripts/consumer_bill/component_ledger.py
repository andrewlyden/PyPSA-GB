"""Versioned, validation-first accounting ledger for consumer bills.

The ledger is intentionally a narrow interchange format.  Source extractors,
model reconciliations and recovery bridges may retain richer private tables,
but anything presented as a bill component must pass through this schema.
Amounts are signed (charges positive, credits negative); ``sign`` is retained
as a human-readable cross-check rather than being applied a second time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEDGER_SCHEMA_VERSION = "1.0"
METHOD_VERSION = "historical-consumer-bill-1.0"
IDENTITY_COLUMNS = (
    "result_type", "component_id", "period_start", "period_end", "region",
    "payment_method", "meter_type", "consumption_case", "model_run_id",
)
REQUIRED_COLUMNS = {
    "ledger_id", "schema_version", "result_type", "component_id", "component_label",
    "category", "period_start", "period_end", "publication_date", "as_of_date",
    "calendar_year", "obligation_year", "region", "payment_method", "meter_type",
    "currency_year", "currency_basis", "unit", "consumption_basis",
    "consumption_case", "quantity", "rate", "amount_gbp", "sign", "vat_treatment",
    "quantity_source", "price_source", "recovery_method", "evidence_status",
    "source_reference", "formula", "model_run_id", "source_hash", "config_hash",
    "method_version", "quantity_coverage", "cost_coverage", "calibration_flags",
}
VALID_RESULT_TYPES = {
    "published_tariff_replay", "historical_model_reconstruction",
    "household_payment_illustration", "system_outturn", "scheme_outturn",
    "reconciliation_residual",
}
VALID_EVIDENCE = {"observed", "modelled", "calibrated", "externally_completed"}


class LedgerValidationError(ValueError):
    """Raised when a bill ledger is ambiguous or internally inconsistent."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def ledger_id(row: pd.Series | dict[str, Any]) -> str:
    values = ["" if pd.isna(row.get(column)) else str(row.get(column)) for column in IDENTITY_COLUMNS]
    return hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:24]


def add_ledger_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["ledger_id"] = result.apply(ledger_id, axis=1)
    return result


def _validate_intervals(frame: pd.DataFrame, expected_start=None, expected_end=None) -> None:
    starts = pd.to_datetime(frame["period_start"], errors="raise")
    ends = pd.to_datetime(frame["period_end"], errors="raise")
    if (ends < starts).any():
        raise LedgerValidationError("Ledger rows contain a period ending before it starts")
    if expected_start is not None and starts.min() < pd.Timestamp(expected_start):
        raise LedgerValidationError("Ledger begins before the declared reporting period")
    if expected_end is not None and ends.max() > pd.Timestamp(expected_end):
        raise LedgerValidationError("Ledger ends after the declared reporting period")


def validate_component_ledger(
    frame: pd.DataFrame,
    *,
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> pd.DataFrame:
    """Validate and return a normalised ledger without changing its amounts."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise LedgerValidationError(f"Component ledger is missing columns: {sorted(missing)}")
    result = frame.copy()
    if result.empty:
        raise LedgerValidationError("Component ledger must contain at least one row")
    if result[list(REQUIRED_COLUMNS - {"obligation_year", "model_run_id", "publication_date", "as_of_date"})].isna().any().any():
        raise LedgerValidationError("Required ledger values cannot be missing")
    if not result["schema_version"].astype(str).eq(LEDGER_SCHEMA_VERSION).all():
        raise LedgerValidationError(f"Unsupported component-ledger schema; expected {LEDGER_SCHEMA_VERSION}")
    result["schema_version"] = LEDGER_SCHEMA_VERSION
    unknown_results = sorted(set(result["result_type"]) - VALID_RESULT_TYPES)
    if unknown_results:
        raise LedgerValidationError(f"Unknown result_type values: {unknown_results}")
    unknown_evidence = sorted(set(result["evidence_status"]) - VALID_EVIDENCE)
    if unknown_evidence:
        raise LedgerValidationError(f"Unknown evidence_status values: {unknown_evidence}")
    for column in ("quantity", "rate", "amount_gbp", "quantity_coverage", "cost_coverage"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all():
            raise LedgerValidationError(f"{column} must contain finite values")
    if not result["quantity_coverage"].between(0, 1).all() or not result["cost_coverage"].between(0, 1).all():
        raise LedgerValidationError("Coverage values must be between zero and one")
    expected_sign = np.select(
        [result["amount_gbp"].gt(0), result["amount_gbp"].lt(0)],
        ["charge", "credit"], default="zero",
    )
    if not result["sign"].astype(str).eq(expected_sign).all():
        raise LedgerValidationError("sign must agree with the signed amount_gbp")
    _validate_intervals(result, expected_start, expected_end)
    calculated_ids = result.apply(ledger_id, axis=1)
    if not result["ledger_id"].astype(str).eq(calculated_ids).all():
        raise LedgerValidationError("ledger_id does not match the stable identity fields")
    if result["ledger_id"].duplicated().any():
        duplicates = sorted(result.loc[result["ledger_id"].duplicated(False), "ledger_id"].unique())
        raise LedgerValidationError(f"Duplicate ledger identities: {duplicates}")
    return result


def validate_run_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a full-year or partitioned model-artifact manifest.

    Timestamps must be timezone-aware UTC.  Adjacent partitions use half-open
    intervals, so the next ``period_start`` must equal the previous
    ``period_end``. Missing files are rejected; missing dispatch is never
    interpreted as zero.
    """
    required = {"schema_version", "run_id", "period_start", "period_end", "artifacts"}
    if missing := required.difference(manifest):
        raise LedgerValidationError(f"Run manifest is missing: {sorted(missing)}")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise LedgerValidationError("Run manifest artifacts must be a non-empty list")
    overall_start = pd.Timestamp(manifest["period_start"])
    overall_end = pd.Timestamp(manifest["period_end"])
    if overall_start.tz is None or overall_end.tz is None:
        raise LedgerValidationError("Run manifest period boundaries must be timezone-aware UTC")
    if str(overall_start.tz) != "UTC" or str(overall_end.tz) != "UTC" or overall_end <= overall_start:
        raise LedgerValidationError("Run manifest requires an increasing UTC period")
    ordered = sorted(artifacts, key=lambda item: pd.Timestamp(item["period_start"]))
    cursor = overall_start
    common_units = manifest.get("units")
    common_config_hash = manifest.get("config_hash")
    for artifact in ordered:
        for key in ("path", "period_start", "period_end"):
            if key not in artifact:
                raise LedgerValidationError(f"Run artifact is missing {key}")
        start, end = pd.Timestamp(artifact["period_start"]), pd.Timestamp(artifact["period_end"])
        if start.tz is None or end.tz is None or str(start.tz) != "UTC" or str(end.tz) != "UTC":
            raise LedgerValidationError("Every model artifact period must use UTC")
        if start != cursor or end <= start:
            raise LedgerValidationError(f"Run artifacts have a gap, overlap or invalid period at {cursor}")
        if common_units is not None and artifact.get("units", common_units) != common_units:
            raise LedgerValidationError("Run artifacts use incompatible units")
        if common_config_hash is not None and artifact.get("config_hash", common_config_hash) != common_config_hash:
            raise LedgerValidationError("Run artifacts use incompatible configurations")
        path = Path(artifact["path"])
        if not path.exists():
            raise LedgerValidationError(f"Declared model artifact does not exist: {path}")
        if artifact.get("sha256") and file_sha256(path) != str(artifact["sha256"]).lower():
            raise LedgerValidationError(f"Model artifact hash does not match: {path}")
        cursor = end
    if cursor != overall_end:
        raise LedgerValidationError("Run artifacts do not cover the declared period")
    return manifest


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LedgerValidationError("Run manifest must be a JSON object")
    return validate_run_manifest(payload)
