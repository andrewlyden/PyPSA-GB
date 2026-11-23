"""Build a provenance-preserving public station-identity registry.

The registry deliberately records source facts; it does not infer a BMU-to-
support-scheme match.  That later matching step can therefore use several
independent identifiers (name, capacity, technology, party, GSP and dates)
and retain the underlying evidence for review.

The fetch command is used by the Snakemake source rule.  It downloads the
three machine-readable public datasets that have stable APIs.  Ofgem's RO
accredited-stations report is supplied as a locally saved official extract:
the Renewable Electricity Register's public-report delivery mechanism changes
occasionally, so silently scraping its user interface would not be robust.
"""

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests


LOGGER = logging.getLogger("station_identity_registry")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ELEXON_BMU_URL = "https://data.elexon.co.uk/bmrs/api/v1/reference/bmunits/all"
LCCC_PACKAGE_URL = (
    "https://dp.lowcarboncontracts.uk/api/3/action/package_show"
    "?id=cfd-contract-portfolio-status"
)
NESO_TEC_PACKAGE_URL = (
    "https://api.neso.energy/api/3/action/datapackage_show"
    "?id=transmission-entry-capacity-tec-register"
)

CANONICAL_COLUMNS = [
    "source", "source_record_id", "station_name", "station_name_normalized",
    "technology", "capacity_mw", "developer_or_lead_party", "bmu_id",
    "national_grid_bmu", "eic", "gsp_group", "cfd_id", "support_scheme",
    "support_status", "allocation_round", "repd_cfd_allocation_round", "ro_banding",
    "development_status", "valid_from", "valid_to", "latitude", "longitude",
    "source_url", "retrieved_at", "source_file",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalise_name(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).lower().replace("&", " and ")
    value = re.sub(r"\b(wind\s*farm|power\s*station|generating\s*station)\b", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    indexed = {re.sub(r"[^a-z0-9]", "", str(col).lower()): col for col in frame.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in indexed:
            return frame[indexed[key]]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _capacity_mw(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    # Ofgem report fields are normally explicitly in kW.  The caller chooses
    # the relevant conversion rather than guessing from a numeric magnitude.
    return numeric


def _read_csv(path: Path) -> pd.DataFrame:
    """Read official CSVs that are commonly published as UTF-8 or Windows-1252."""
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error  # pragma: no cover - defensive fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str) -> Any:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def _csv_resource(package_url: str, preferred_name: str) -> tuple[pd.DataFrame, str]:
    package = _request_json(package_url)
    result = package.get("result", package)
    resources = result.get("resources", [])
    resource = next(
        (item for item in resources if str(item.get("format", "")).lower() == "csv"),
        None,
    )
    if resource is None:
        raise ValueError(f"No CSV resource exposed by {preferred_name}")
    url = resource.get("url") or resource.get("path") or resource.get("download_url")
    if not url:
        raise ValueError(f"CSV resource for {preferred_name} does not expose a download URL")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content), encoding="utf-8-sig", low_memory=False), url


def fetch_sources(output_dir: Path, manifest_path: Path) -> None:
    """Fetch Elexon, LCCC and NESO source records into a reproducible cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"retrieved_at": _timestamp(), "sources": {}}

    elexon_payload = _request_json(ELEXON_BMU_URL)
    elexon_rows = elexon_payload.get("data", elexon_payload) if isinstance(elexon_payload, dict) else elexon_payload
    elexon_path = output_dir / "elexon_bmu_reference.csv"
    pd.DataFrame(elexon_rows).to_csv(elexon_path, index=False)
    manifest["sources"]["elexon_bmu_reference"] = {
        "url": ELEXON_BMU_URL, "file": elexon_path.name, "rows": len(elexon_rows),
        "sha256": _sha256(elexon_path),
    }

    for name, package_url, filename in [
        ("lccc_cfd_contract_portfolio", LCCC_PACKAGE_URL, "lccc_cfd_contract_portfolio.csv"),
        ("neso_tec_register", NESO_TEC_PACKAGE_URL, "neso_tec_register.csv"),
    ]:
        frame, resource_url = _csv_resource(package_url, name)
        path = output_dir / filename
        frame.to_csv(path, index=False)
        manifest["sources"][name] = {
            "url": resource_url, "file": path.name, "rows": len(frame),
            "sha256": _sha256(path),
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Fetched station-identity source cache to %s", output_dir)


def _records(
    frame: pd.DataFrame,
    source: str,
    source_url: str,
    source_file: str,
    *,
    capacity_scale: float = 1.0,
    support_scheme: str | None = None,
) -> pd.DataFrame:
    """Map a source table into the long, intentionally non-matched registry."""
    out = pd.DataFrame(index=frame.index)
    out["source"] = source
    out["source_record_id"] = _column(frame, "dictionary_id", "cfd_id", "project_id", "project_number", "accreditation_reference", "elexon_bm_unit", "bmu_id", "eic", "ref_id", "id")
    out["station_name"] = _column(frame, "name_of_cfd_unit", "station_name", "bm_unit_name", "project_name", "site_name", "customer_name", "name")
    out["station_name_normalized"] = out["station_name"].map(_normalise_name)
    out["technology"] = _column(frame, "technology_type", "technology", "plant_type", "fuel_type", "generation_type")
    out["capacity_mw"] = _capacity_mw(_column(
        frame, "maximum_contract_capacity_mw", "maximum_contract_capacity",
        "generation_capacity", "cumulative_total_capacity_mw", "tec_mw", "capacity_mw", "installed_capacity_mwelec", "station_tic_kw", "station_dnc_kw",
    )) * capacity_scale
    out["developer_or_lead_party"] = _column(frame, "lead_party_name", "company_name", "developer", "customer_name", "operator", "operator_or_applicant")
    out["bmu_id"] = _column(frame, "elexon_bm_unit", "elexon_bmu", "bm_unit_id", "sett_bmu_id")
    out["national_grid_bmu"] = _column(frame, "national_grid_bm_unit", "nationalgridbmunit", "ngc_bmu_name", "ngc_bmu_id")
    out["eic"] = _column(frame, "eic", "eic_id")
    out["gsp_group"] = _column(frame, "gsp_group_name", "gsp_group_id")
    out["cfd_id"] = _column(frame, "cfd_id", "cfdid")
    out["support_scheme"] = support_scheme if support_scheme else _column(frame, "scheme", "support_scheme")
    out["support_status"] = _column(frame, "status", "contract_status", "accreditation_status")
    out["allocation_round"] = _column(frame, "allocation_round", "allocationround")
    out["repd_cfd_allocation_round"] = _column(frame, "cfd_allocation_round")
    out["ro_banding"] = _column(frame, "ro_banding_roc_mwh", "ro_banding")
    out["development_status"] = _column(frame, "development_status", "development_status_short")
    out["valid_from"] = _column(frame, "start_date", "expected_start_date", "estimated_start_date", "mw_effective_from", "effective_from_date", "effective_date", "accreditation_date", "operational")
    out["valid_to"] = _column(frame, "effective_to_date", "end_date", "termination_date")
    out["latitude"] = _column(frame, "latitude", "lat")
    out["longitude"] = _column(frame, "longitude", "lon", "long")
    out["source_url"] = source_url
    out["retrieved_at"] = _timestamp()
    out["source_file"] = source_file
    return out.reindex(columns=CANONICAL_COLUMNS)


def build_registry(
    elexon_path: Path,
    lccc_path: Path,
    neso_path: Path,
    repd_path: Path,
    dictionary_path: Path,
    output_path: Path,
    evidence_path: Path,
    *,
    ofgem_path: Path | None = None,
) -> None:
    """Create source-preserving records, without performing probabilistic matching."""
    sources = [
        ("elexon_bmu_reference", elexon_path, ELEXON_BMU_URL, 1.0, None),
        ("lccc_cfd_contract_portfolio", lccc_path, LCCC_PACKAGE_URL, 1.0, "CfD"),
        ("neso_tec_register", neso_path, NESO_TEC_PACKAGE_URL, 1.0, None),
        ("desnz_repd", repd_path, "local REPD extract", 1.0, None),
        ("power_station_dictionary_ids", dictionary_path, "local curated Power Station Dictionary", 1.0, None),
    ]
    if ofgem_path:
        sources.append(("ofgem_ro_accredited_stations", ofgem_path, "local official Ofgem RER extract", 0.001, "ROC"))

    records = []
    evidence = []
    for name, path, url, scale, scheme in sources:
        if not path.exists():
            raise FileNotFoundError(f"Configured station-identity source does not exist: {path}")
        frame = _read_csv(path)
        records.append(_records(frame, name, url, str(path), capacity_scale=scale, support_scheme=scheme))
        evidence.append({"source": name, "path": str(path), "rows": len(frame), "sha256": _sha256(path)})

    registry = pd.concat(records, ignore_index=True)
    registry = registry.dropna(how="all", subset=["station_name", "bmu_id", "eic", "cfd_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_path, index=False)
    evidence_path.write_text(json.dumps({"built_at": _timestamp(), "sources": evidence}, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d source records to %s", len(registry), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--output-dir", required=True)
    fetch.add_argument("--manifest", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--elexon", required=True)
    build.add_argument("--lccc", required=True)
    build.add_argument("--neso", required=True)
    build.add_argument("--repd", required=True)
    build.add_argument("--dictionary-ids", required=True)
    build.add_argument("--ofgem-ro")
    build.add_argument("--output", required=True)
    build.add_argument("--evidence", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_sources(Path(args.output_dir), Path(args.manifest))
    else:
        build_registry(
            Path(args.elexon), Path(args.lccc), Path(args.neso), Path(args.repd),
            Path(args.dictionary_ids), Path(args.output), Path(args.evidence),
            ofgem_path=Path(args.ofgem_ro) if args.ofgem_ro else None,
        )


def _run_from_snakemake() -> None:
    """Allow the same module to be called by the workflow without a shell."""
    if snakemake.params.mode == "fetch":
        fetch_sources(
            Path(snakemake.params.source_dir),
            Path(snakemake.output.manifest),
        )
        return
    ofgem_path = None
    if "ofgem_ro" in snakemake.input.keys():
        ofgem_path = Path(str(snakemake.input.ofgem_ro))
    build_registry(
        Path(snakemake.input.elexon),
        Path(snakemake.input.lccc),
        Path(snakemake.input.neso),
        Path(snakemake.input.repd),
        Path(snakemake.input.dictionary_ids),
        Path(snakemake.output.registry),
        Path(snakemake.output.evidence),
        ofgem_path=ofgem_path,
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        _run_from_snakemake()
    else:
        main()
