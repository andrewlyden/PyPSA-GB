"""Build the committed NESO day-ahead boundary-limit cache (F25.12).

Historical solves with ``outage_schedule.source: neso`` read NESO's published
day-ahead constraint limits per boundary. They used to fetch them from
api.neso.energy at solve time and silently skip a boundary when the request
failed. The solve now reads this committed file instead and fails if a
boundary it needs is missing, so this script is the only place the data is
refreshed.

Input is NESO's full dataset export ("Day Ahead Constraint Flows and Limits",
https://www.neso.energy/data-portal/day-ahead-constraint-flows-and-limits,
resource ``38a18ec1-9e40-465d-93fb-301e80fd1352``), either a local CSV
(``--source``) or downloaded from the datastore dump (``--download``). The
output keeps only the boundaries named by the boundary-mapping files, in the
columns the solve reads (``boundary, date, limit_mw, flow_mw``), with a JSON
sidecar recording the source, its SHA-256, the row count and date range.

Usage::

    python scripts/solve/build_neso_boundary_limit_cache.py \
        --source resources/neso_cache/day_ahead_constraint_flows_limits.csv
    python scripts/solve/build_neso_boundary_limit_cache.py --download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT = "data/network/neso_day_ahead_constraint_limits.csv.gz"
DEFAULT_MAPPINGS = (
    "data/network/neso_boundary_mapping.yaml",
    "data/network/neso_boundary_mapping_ssesp_strict.yaml",
)
DEFAULT_RESOURCE_ID = "38a18ec1-9e40-465d-93fb-301e80fd1352"
DUMP_URL = "https://api.neso.energy/datastore/dump/{resource_id}"

SOURCE_COLUMNS = {
    "Constraint Group": "boundary",
    "Date (GMT/BST)": "date",
    "Limit (MW)": "limit_mw",
    "Flow (MW)": "flow_mw",
}
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def mapped_neso_boundaries(mapping_paths: Iterable[Path]) -> List[str]:
    """Return the NESO constraint groups named by the boundary-mapping files.

    Subconstraints read their parent's NESO series (``neso_boundary`` defaults
    to the parent name), so only top-level names and explicit
    ``neso_boundary`` entries are collected.
    """
    names = set()
    for path in mapping_paths:
        with open(path, "r", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle) or {}
        for name, info in (mapping.get("boundaries") or {}).items():
            info = info or {}
            names.add(str(info.get("neso_boundary", name)))
            for sub in (info.get("subconstraints") or {}).values():
                if isinstance(sub, dict) and sub.get("neso_boundary"):
                    names.add(str(sub["neso_boundary"]))
    return sorted(names)


def normalise_neso_export(raw: pd.DataFrame, boundaries: Iterable[str]) -> pd.DataFrame:
    """Keep the solve's columns and boundaries from a NESO export, sorted."""
    missing = [col for col in SOURCE_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"NESO export lacks columns {missing}")
    df = raw[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    df = df[df["boundary"].isin(set(boundaries))].copy()
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    df["limit_mw"] = pd.to_numeric(df["limit_mw"], errors="coerce")
    df["flow_mw"] = pd.to_numeric(df["flow_mw"], errors="coerce")
    absent = sorted(set(boundaries) - set(df["boundary"]))
    if absent:
        raise ValueError(f"NESO export has no rows for mapped boundaries {absent}")
    return df.sort_values(["boundary", "date"], kind="mergesort").reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache(source: Path, output: Path, mapping_paths: Iterable[Path],
                source_label: Optional[str] = None) -> dict:
    """Write the committed cache and its sidecar; return the sidecar dict."""
    mapping_paths = [Path(p) for p in mapping_paths]
    boundaries = mapped_neso_boundaries(mapping_paths)
    raw = pd.read_csv(source, low_memory=False)
    df = normalise_neso_export(raw, boundaries)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, date_format=DATE_FORMAT,
              compression={"method": "gzip", "mtime": 0})
    meta = {
        "description": "NESO day-ahead constraint limits per boundary, committed for "
                       "historical NESO outage-schedule solves (F25.12)",
        "source": source_label or str(source),
        "source_sha256": _sha256(source),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "boundaries": boundaries,
        "rows": int(len(df)),
        "date_min": df["date"].min().strftime(DATE_FORMAT),
        "date_max": df["date"].max().strftime(DATE_FORMAT),
        "mapping_files": [p.as_posix() for p in mapping_paths],
    }
    with open(sidecar_path(output), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return meta


def sidecar_path(output: Path) -> Path:
    name = output.name
    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            return output.with_name(name[: -len(suffix)] + ".json")
    return output.with_suffix(".json")


def _download(resource_id: str, target: Path) -> None:
    import requests  # only the refresh step touches the network

    url = DUMP_URL.format(resource_id=resource_id)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            for chunk in resp.iter_content(1 << 20):
                handle.write(chunk)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path, help="Local NESO export CSV")
    group.add_argument("--download", action="store_true",
                       help="Download the full export from the NESO datastore dump")
    parser.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    parser.add_argument("--download-to", type=Path,
                        default=Path("resources/neso_cache/day_ahead_constraint_flows_limits.csv"))
    parser.add_argument("--source-label", default=None,
                        help="Provenance label for the sidecar (default: the source path)")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--mapping", type=Path, action="append",
                        help="Boundary mapping YAML (repeatable; default: both committed mappings)")
    args = parser.parse_args(argv)

    source = args.source
    label = args.source_label
    if args.download:
        _download(args.resource_id, args.download_to)
        source = args.download_to
        label = DUMP_URL.format(resource_id=args.resource_id)
    mappings = args.mapping or [Path(p) for p in DEFAULT_MAPPINGS]
    meta = build_cache(source, args.output, mappings, source_label=label)
    print(json.dumps({k: meta[k] for k in ("rows", "date_min", "date_max", "boundaries")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
