"""Audit unresolved Ofgem ROC stations against historical source registers.

This is deliberately an *evidence* audit, not an automatic mapper.  A station
may occur in REPD or DUKES yet still need a split/aggregation change before it
can be represented by one PyPSA generator.  The audit keeps those two questions
separate and records the best candidates from each source for review.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


def _normalise(value: object) -> str:
    """Normalise legal/project naming noise while retaining geographic tokens."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(
        r"\b(limited|ltd|llp|plc|full|power station|generating station|"
        r"wind ?farm|solar ?(?:farm|park|pv)|facility|plant|extension|"
        r"resubmission|repowering)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _postcode(value: object) -> str:
    match = re.search(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", str(value).upper())
    return re.sub(r"\s", "", match.group(0)) if match else ""


def _load_repd(path: Path, source: str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="latin1")
    return pd.DataFrame(
        {
            "source": source,
            "source_id": data["Ref ID"].astype(str),
            "candidate_name": data["Site Name"].astype(str),
            "candidate_capacity_mw": pd.to_numeric(
                data["Installed Capacity (MWelec)"], errors="coerce"
            ),
            "candidate_address": data.get("Address", "").astype(str),
            "candidate_technology": data.get("Technology Type", "").astype(str),
        }
    )


def _load_dukes(path: Path, sheet: str, source: str, *, full: bool) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name=sheet, header=5)
    if full:
        name = data["Site Name"]
        capacity = data["InstalledCapacity (MW)"]
        address = data.get("Postcode", "")
    else:
        name = data["Station Name"]
        capacity = data["Installed Capacity\n(MW)"]
        address = ""
    return pd.DataFrame(
        {
            "source": source,
            "source_id": data.index.astype(str),
            "candidate_name": name.astype(str),
            "candidate_capacity_mw": pd.to_numeric(capacity, errors="coerce"),
            "candidate_address": pd.Series(address, index=data.index).astype(str),
            "candidate_technology": data["Technology"].astype(str),
        }
    )


def _candidate_score(station: pd.Series, candidate: pd.Series) -> dict[str, float | bool]:
    station_name = _normalise(station["Station name"])
    candidate_name = _normalise(candidate["candidate_name"])
    # token-set identifies aliases; token-sort prevents a generic partial name
    # being treated as an exact match (e.g. North Clyde Recycling vs Clyde).
    token_set = fuzz.token_set_ratio(station_name, candidate_name)
    token_sort = fuzz.token_sort_ratio(station_name, candidate_name)
    name_score = 0.4 * token_set + 0.6 * token_sort
    station_capacity = float(station["station_capacity_mw"])
    candidate_capacity = candidate["candidate_capacity_mw"]
    if pd.notna(candidate_capacity) and max(station_capacity, candidate_capacity) > 0:
        capacity_difference = abs(station_capacity - candidate_capacity) / max(
            station_capacity, candidate_capacity
        )
    else:
        capacity_difference = float("nan")
    station_postcode = _postcode(station.get("station_address", ""))
    candidate_postcode = _postcode(candidate.get("candidate_address", ""))
    postcode_agrees = bool(station_postcode and candidate_postcode and station_postcode == candidate_postcode)
    capacity_score = max(0.0, 100.0 * (1.0 - capacity_difference / 0.30)) if pd.notna(capacity_difference) else 0.0
    joint_score = 0.70 * name_score + 0.25 * capacity_score + 5.0 * postcode_agrees
    return {
        "name_score": round(name_score, 3),
        "token_set_score": round(token_set, 3),
        "token_sort_score": round(token_sort, 3),
        "capacity_relative_difference": round(capacity_difference, 6) if pd.notna(capacity_difference) else float("nan"),
        "postcode_agrees": postcode_agrees,
        "joint_score": round(joint_score, 3),
    }


def _status(candidates: pd.DataFrame) -> str:
    """Evidence status, deliberately stricter than a mapping decision."""
    if candidates.empty:
        return "no_candidate_in_searched_sources"
    best = candidates.iloc[0]
    if best["name_score"] >= 95 and best["capacity_relative_difference"] <= 0.05:
        return "confirmed_station_in_source"
    if (
        best["name_score"] >= 80
        and best["capacity_relative_difference"] <= 0.20
        and (best["postcode_agrees"] or best["source"].startswith("DUKES"))
    ):
        return "strong_alias_or_station_record"
    if best["name_score"] >= 80:
        return "name_found_but_requires_capacity_or_phase_review"
    return "no_reliable_single_station_match"


def audit(
    worklist_path: Path,
    renewable_overrides_path: Path,
    generator_overrides_path: Path,
    repd_2021_path: Path,
    repd_2024_path: Path,
    repd_2025_path: Path,
    dukes_path: Path,
    min_capacity_mw: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    worklist = pd.read_csv(worklist_path)
    renewables = pd.read_csv(renewable_overrides_path)
    generators = pd.read_csv(generator_overrides_path)
    mapped_references = set(renewables["ro_accreditation_reference"].dropna().astype(str))
    mapped_references.update(generators["cfd_id"].dropna().astype(str))
    unresolved = worklist.loc[
        (worklist["station_capacity_mw"] >= min_capacity_mw)
        & ~worklist["Accreditation reference"].astype(str).isin(mapped_references)
    ].copy()

    sources = pd.concat(
        [
            _load_repd(repd_2021_path, "REPD_2021"),
            _load_repd(repd_2024_path, "REPD_2024"),
            _load_repd(repd_2025_path, "REPD_2025"),
            _load_dukes(dukes_path, "DUKES 2020", "DUKES_2020", full=False),
            _load_dukes(dukes_path, "5.11 Full list", "DUKES_2025_FULL", full=True),
        ],
        ignore_index=True,
    ).dropna(subset=["candidate_name"])
    sources["normalised_name"] = sources["candidate_name"].map(_normalise)

    candidate_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for _, station in unresolved.iterrows():
        ranked: list[dict[str, object]] = []
        query = _normalise(station["Station name"])
        # Inspect the five best string candidates from every source.  This avoids
        # one large REPD extract crowding out DUKES' historically precise entry.
        for source, group in sources.groupby("source", sort=False):
            choices = group["normalised_name"].tolist()
            for _, _, index in process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=5):
                candidate = group.iloc[index]
                scored = _candidate_score(station, candidate)
                ranked.append(
                    {
                        "Accreditation reference": station["Accreditation reference"],
                        "Station name": station["Station name"],
                        "Technology": station["Technology"],
                        "station_capacity_mw": station["station_capacity_mw"],
                        "ofgem_roc_certificates": station["ofgem_roc_certificates"],
                        "station_address": station.get("station_address", ""),
                        **candidate.drop(labels=["normalised_name"]).to_dict(),
                        **scored,
                    }
                )
        ranked_frame = pd.DataFrame(ranked).sort_values(
            ["joint_score", "name_score", "capacity_relative_difference"],
            ascending=[False, False, True],
        )
        ranked_frame["candidate_rank"] = range(1, len(ranked_frame) + 1)
        candidate_rows.extend(ranked_frame.to_dict("records"))
        best = ranked_frame.iloc[0]
        summaries.append(
            {
                "Accreditation reference": station["Accreditation reference"],
                "Station name": station["Station name"],
                "Technology": station["Technology"],
                "station_capacity_mw": station["station_capacity_mw"],
                "ofgem_roc_certificates": station["ofgem_roc_certificates"],
                "source_audit_status": _status(ranked_frame),
                "best_source": best["source"],
                "best_source_id": best["source_id"],
                "best_candidate_name": best["candidate_name"],
                "best_candidate_capacity_mw": best["candidate_capacity_mw"],
                "best_name_score": best["name_score"],
                "best_capacity_relative_difference": best["capacity_relative_difference"],
                "best_postcode_agrees": best["postcode_agrees"],
                "best_joint_score": best["joint_score"],
            }
        )
    summary = pd.DataFrame(summaries).sort_values("ofgem_roc_certificates", ascending=False)
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["Accreditation reference", "candidate_rank"]
    )
    metadata = {
        "minimum_station_capacity_mw": min_capacity_mw,
        "unresolved_station_count": int(len(summary)),
        "unresolved_roc_certificates": float(summary["ofgem_roc_certificates"].sum()),
        "status_counts": summary.groupby("source_audit_status").size().to_dict(),
        "status_roc_certificates": summary.groupby("source_audit_status")["ofgem_roc_certificates"].sum().to_dict(),
        "sources_searched": sorted(sources["source"].unique().tolist()),
    }
    return summary, candidates, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--renewable-overrides", type=Path, required=True)
    parser.add_argument("--generator-overrides", type=Path, required=True)
    parser.add_argument("--repd-2021", type=Path, required=True)
    parser.add_argument("--repd-2024", type=Path, required=True)
    parser.add_argument("--repd-2025", type=Path, required=True)
    parser.add_argument("--dukes", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--minimum-capacity-mw", type=float, default=10.0)
    args = parser.parse_args()

    summary, candidates, metadata = audit(
        args.worklist,
        args.renewable_overrides,
        args.generator_overrides,
        args.repd_2021,
        args.repd_2024,
        args.repd_2025,
        args.dukes,
        args.minimum_capacity_mw,
    )
    for path in (args.summary_output, args.candidates_output, args.metadata_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    candidates.to_csv(args.candidates_output, index=False)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
