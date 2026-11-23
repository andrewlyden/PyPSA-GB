"""Build a reusable Ofgem ROC-station evidence crosswalk from source registers.

The crosswalk is intentionally made *before* the network solve.  It searches
every issuing station, not only the large residuals exposed by a previous
post-solve ledger.  Only explicit support overrides and unambiguous source
records are marked high-confidence; weaker candidates remain visible for
review and cannot silently change dispatch labels.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


def normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    text = re.sub(
        r"\b(limited|ltd|llp|plc|power station|generating station|wind ?farm|"
        r"solar ?(?:farm|park|pv)|facility|plant|extension|repowering)\b", " ", text,
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def clean_reference(value: object) -> str:
    return re.sub(r"\.0$", "", str(value).strip()) if pd.notna(value) else ""


def postcode(value: object) -> str:
    match = re.search(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", str(value or "").upper())
    return re.sub(r"\s", "", match.group(0)) if match else ""


def ofgem_category(value: object) -> str:
    text = str(value or "").lower()
    for term, category in (
        ("offshore wind", "wind"), ("onshore wind", "wind"),
        ("photovoltaic", "solar"), ("solar", "solar"),
        ("landfill", "landfill"), ("sewage", "sewage"),
        ("tidal", "tidal"), ("wave", "wave"), ("hydro", "hydro"),
        ("fuelled", "fuelled"),
    ):
        if term in text:
            return category
    return "other"


def source_category(value: object) -> str:
    text = str(value or "").lower()
    if "wind" in text:
        return "wind"
    if "solar" in text or "photovoltaic" in text:
        return "solar"
    if "landfill" in text:
        return "landfill"
    if "sewage" in text or "sludge" in text:
        return "sewage"
    if "tidal" in text:
        return "tidal"
    if "wave" in text:
        return "wave"
    if "hydro" in text:
        return "hydro"
    if any(term in text for term in ("biomass", "waste", "biogas", "anaerobic", "energy from", "fuel")):
        return "fuelled"
    return "other"


def load_repd(path: Path, source: str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="latin1", low_memory=False)
    return pd.DataFrame({
        "source": source,
        "source_id": data["Ref ID"].astype(str),
        "candidate_name": data["Site Name"].astype(str),
        "candidate_capacity_mw": pd.to_numeric(data["Installed Capacity (MWelec)"], errors="coerce"),
        "candidate_technology": data.get("Technology Type", "").astype(str),
        "candidate_address": data.get("Address", "").astype(str),
    })


def load_dukes(path: Path, sheet: str, source: str, full: bool) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name=sheet, header=5)
    names = data["Site Name"] if full else data["Station Name"]
    capacity = data["InstalledCapacity (MW)"] if full else data["Installed Capacity\n(MW)"]
    address = data.get("Postcode", "") if full else ""
    return pd.DataFrame({
        "source": source,
        "source_id": data.index.astype(str),
        "candidate_name": names.astype(str),
        "candidate_capacity_mw": pd.to_numeric(capacity, errors="coerce"),
        "candidate_technology": data.get("Technology", "").astype(str),
        "candidate_address": pd.Series(address, index=data.index).astype(str),
    })


def capacity_difference(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right) or max(float(left), float(right)) <= 0:
        return float("nan")
    return abs(float(left) - float(right)) / max(float(left), float(right))


def score(station: pd.Series, candidate: pd.Series) -> dict[str, float | bool]:
    name_score = fuzz.token_sort_ratio(normalise(station["Station name"]), normalise(candidate["candidate_name"]))
    cap_diff = capacity_difference(station["station_capacity_mw"], candidate["candidate_capacity_mw"])
    capacity_agrees = bool(pd.notna(cap_diff) and (cap_diff <= 0.20 or abs(float(station["station_capacity_mw"]) - float(candidate["candidate_capacity_mw"])) <= 1.0))
    station_postcode = postcode(station.get("Station address", ""))
    candidate_postcode = candidate.get("candidate_postcode", postcode(candidate.get("candidate_address", "")))
    postcode_agrees = bool(station_postcode and station_postcode == candidate_postcode)
    joint = 0.75 * name_score + 25.0 * max(0.0, 1.0 - (cap_diff / 0.30)) if pd.notna(cap_diff) else 0.75 * name_score
    joint += 10.0 if postcode_agrees else 0.0
    return {"name_score": round(float(name_score), 3), "capacity_relative_difference": round(float(cap_diff), 6) if pd.notna(cap_diff) else None, "capacity_agrees": capacity_agrees, "postcode_agrees": postcode_agrees, "joint_score": round(float(joint), 3)}


def automatic_status(row: dict[str, object]) -> str:
    if row["name_score"] >= 95 and row["capacity_agrees"]:
        return "high_confidence_confirmed_source_record"
    if row.get("postcode_agrees", False) and row["name_score"] >= 70 and row["capacity_agrees"]:
        return "high_confidence_postcode_and_capacity"
    # DUKES is a station register rather than a planning record.  A clearly
    # matching name and DNC is sufficient even when its retained total differs
    # slightly because Ofgem records a ROC-accredited unit/phase.
    if str(row["source"]).startswith("DUKES") and row["name_score"] >= 88 and row["capacity_agrees"]:
        return "high_confidence_dukes_alias"
    if row["name_score"] >= 80 and row["capacity_agrees"]:
        return "review_required_source_candidate"
    return "unresolved_source_candidate"


def explicit_rows(renewables: pd.DataFrame, generators: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in renewables.to_dict("records"):
        reference = clean_reference(row.get("ro_accreditation_reference"))
        if reference:
            rows.append({"Accreditation reference": reference, "source": "renewable_support_override", "source_id": row.get("source_site_name", ""), "candidate_name": row.get("source_site_name", ""), "candidate_capacity_mw": None, "candidate_technology": "", "candidate_address": "", "mapping_status": "high_confidence_explicit_support_override", "mapping_evidence": row.get("evidence", "Audited renewable support override")})
    for row in generators.loc[generators["support_type"].astype(str).eq("ROC")].to_dict("records"):
        # Legacy support file used cfd_id for both LCCC ids and ROC refs.
        reference = clean_reference(row.get("ro_accreditation_reference", row.get("cfd_id")))
        if reference:
            rows.append({"Accreditation reference": reference, "source": "generator_support_override", "source_id": row.get("generator_name", ""), "candidate_name": row.get("generator_name", ""), "candidate_capacity_mw": None, "candidate_technology": "", "candidate_address": "", "mapping_status": "high_confidence_explicit_support_override", "mapping_evidence": row.get("support_evidence", "Audited generator support override")})
    return pd.DataFrame(rows)


def build(outturn_path: Path, registry_path: Path, renewable_overrides_path: Path, generator_overrides_path: Path, repd_paths: list[Path], dukes_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    outturn = pd.read_csv(outturn_path)
    registry = pd.read_csv(registry_path)
    registry["Accreditation reference"] = registry["Accreditation reference"].map(clean_reference)
    registry["station_capacity_mw"] = pd.to_numeric(registry["Station DNC in kW"].astype(str).str.replace(",", "", regex=False), errors="coerce") / 1000.0
    outturn["Accreditation reference"] = outturn["Accreditation reference"].map(clean_reference)
    stations = outturn.merge(registry[["Accreditation reference", "station_capacity_mw", "Station address"]], on="Accreditation reference", how="left")
    stations["station_category"] = stations["Technology"].map(ofgem_category)

    renewable_overrides = pd.read_csv(renewable_overrides_path)
    generator_overrides = pd.read_csv(generator_overrides_path)
    explicit = explicit_rows(renewable_overrides, generator_overrides)
    sources = pd.concat(
        [*[load_repd(path, f"REPD_{path.stem}") for path in repd_paths], load_dukes(dukes_path, "DUKES 2020", "DUKES_2020", False), load_dukes(dukes_path, "5.11 Full list", "DUKES_2025_FULL", True)],
        ignore_index=True,
    ).dropna(subset=["candidate_name"])
    sources["source_category"] = sources["candidate_technology"].map(source_category)
    sources["normalised_name"] = sources["candidate_name"].map(normalise)
    sources["candidate_postcode"] = sources["candidate_address"].map(postcode)

    rows: list[dict[str, object]] = []
    explicit_by_ref = {reference: group.iloc[0].to_dict() for reference, group in explicit.groupby("Accreditation reference")} if not explicit.empty else {}
    for _, station in stations.iterrows():
        reference = station["Accreditation reference"]
        base = {"Accreditation reference": reference, "Station name": station["Station name"], "Technology": station["Technology"], "station_capacity_mw": station["station_capacity_mw"], "ofgem_roc_certificates": station.get("ofgem_roc_certificates", 0.0)}
        if reference in explicit_by_ref:
            row = {**base, **explicit_by_ref[reference], "name_score": 100.0, "capacity_relative_difference": None, "capacity_agrees": True, "postcode_agrees": False, "joint_score": 100.0}
            rows.append(row)
            continue
        pool = sources.loc[sources["source_category"].eq(station["station_category"])]
        if pool.empty:
            pool = sources
        choices = process.extract(normalise(station["Station name"]), pool["normalised_name"].tolist(), scorer=fuzz.token_sort_ratio, limit=5)
        # A planning application can use a development/company name while the
        # Ofgem accreditation uses a meter or unit name.  Exact postcode plus
        # compatible capacity is independent evidence, so include those rows
        # even if a string search would not rank them in its first five.
        station_postcode = postcode(station.get("Station address", ""))
        postcode_indices = []
        if station_postcode:
            postcode_indices = [
                pool.index.get_loc(label)
                for label in pool.index[pool["candidate_postcode"].eq(station_postcode)]
            ]
        selected_indices = list(dict.fromkeys([index for _, _, index in choices] + postcode_indices))
        candidates = []
        for index in selected_indices:
            candidate = pool.iloc[index]
            candidates.append({**base, **candidate.drop(labels=["normalised_name", "source_category"]).to_dict(), **score(station, candidate)})
        if not candidates:
            rows.append({**base, "source": "", "source_id": "", "candidate_name": "", "candidate_capacity_mw": None, "candidate_technology": "", "candidate_address": "", "mapping_status": "unresolved_no_source_candidate", "mapping_evidence": "No technology-compatible record found in REPD or DUKES.", "name_score": None, "capacity_relative_difference": None, "capacity_agrees": False, "postcode_agrees": False, "joint_score": None})
            continue
        best = max(candidates, key=lambda item: (item["joint_score"], item["name_score"]))
        best["mapping_status"] = automatic_status(best)
        best["mapping_evidence"] = "Automated REPD/DUKES station search; inspect non-high-confidence rows before adopting them."
        rows.append(best)
    mapping = pd.DataFrame(rows).sort_values(["mapping_status", "ofgem_roc_certificates"], ascending=[True, False]).reset_index(drop=True)
    high = mapping["mapping_status"].astype(str).str.startswith("high_confidence")
    threshold = pd.to_numeric(mapping["station_capacity_mw"], errors="coerce").gt(10.0)
    summary = {
        "scope": "All calendar-year issuing ROC stations searched against three REPD vintages and DUKES 2020/2025.",
        "station_count": int(len(mapping)),
        "high_confidence_station_count": int(high.sum()),
        "high_confidence_roc_certificates": float(pd.to_numeric(mapping.loc[high, "ofgem_roc_certificates"], errors="coerce").fillna(0).sum()),
        "above_10mw_station_count": int(threshold.sum()),
        "above_10mw_high_confidence_count": int((threshold & high).sum()),
        "above_10mw_remaining_review_count": int((threshold & ~high).sum()),
        "status_counts": mapping["mapping_status"].value_counts().to_dict(),
        "rule": "No small station is assumed absent. Every issuing station is searched; only high-confidence evidence is allowed to affect pre-solve ROC labels.",
    }
    return mapping, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outturn", type=Path, required=True); parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--renewable-overrides", type=Path, required=True); parser.add_argument("--generator-overrides", type=Path, required=True)
    parser.add_argument("--repd", type=Path, nargs="+", required=True); parser.add_argument("--dukes", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True); parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    mapping, summary = build(args.outturn, args.registry, args.renewable_overrides, args.generator_overrides, args.repd, args.dukes)
    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.mapping_output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
