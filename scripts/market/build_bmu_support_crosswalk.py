"""Build an evidence-scored BMU-to-CfD/ROC crosswalk from curated IDs.

This stage does not change model pricing.  It expands the curated Power Station
Dictionary station rows into BMU-level evidence and verifies them against the
cached Elexon, LCCC and REPD records.  Ambiguous support evidence is retained
but marked for review rather than silently assigned to a balancing bid stack.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


COLUMNS = [
    "bmu_id", "dictionary_id", "station_name", "support_type", "confidence",
    "score", "match_route", "elexon_current_match", "cfd_id", "lccc_status", "lccc_technology",
    "lccc_capacity_mw", "elexon_capacity_mw", "name_match_score", "capacity_ratio",
    "cfd_valid_from", "cfd_date_source", "historical_bid_calibration_eligible",
    "repd_ref_id", "repd_site_name", "repd_capacity_mw",
    "ro_banding", "ofgem_accreditation_ref", "ofgem_capacity_mw", "ofgem_technology",
    "ofgem_effective_date", "support_conflict", "review_reason",
]


def _split(frame: pd.DataFrame, column: str, output: str) -> pd.DataFrame:
    result = frame[["dictionary_id", "name", column]].dropna().copy()
    result[output] = result[column].astype(str).str.split(r"\s*,\s*")
    result = result.explode(output)
    if output != column:
        result = result.drop(columns=column)
    result[output] = result[output].astype(str).str.strip()
    return result[result[output].ne("")]


def _nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _dictionary_bmus(dictionary: pd.DataFrame) -> pd.DataFrame:
    return _split(dictionary[dictionary["sett_bmu_id"].notna()], "sett_bmu_id", "bmu_id").rename(
        columns={"name": "station_name"}
    )


def _elexon_bmus(registry: pd.DataFrame) -> set[str]:
    values = registry.loc[
        registry.source.eq("elexon_bmu_reference"), "bmu_id"
    ].dropna().astype(str)
    return set(values)


def _canonical_identifier(series: pd.Series) -> pd.Series:
    """Normalise spreadsheet IDs without turning missing values into matches."""
    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return values.mask(values.isin(["", "nan", "None", "<NA>"]))


def _cfd_repd_dates(dictionary: pd.DataFrame, repd: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic REPD commissioning dates for dictionary-linked CfD units."""
    repd_fields = repd.copy()
    repd_fields["repd_ref_id"] = _canonical_identifier(repd_fields["Ref ID"])
    repd_fields["repd_old_ref_id"] = _canonical_identifier(repd_fields["Old Ref ID"])
    repd_fields["cfd_valid_from"] = pd.to_datetime(
        repd_fields["Operational"], dayfirst=True, errors="coerce"
    )
    repd_fields["repd_has_cfd"] = _nonempty(repd_fields["CfD Allocation Round"])
    repd_fields = repd_fields.loc[
        repd_fields.repd_has_cfd & repd_fields.cfd_valid_from.notna(),
        ["repd_ref_id", "repd_old_ref_id", "cfd_valid_from"],
    ]

    links = []
    for dictionary_column in ("new_repd_id", "old_repd_id"):
        dictionary_links = dictionary[["dictionary_id", dictionary_column]].dropna().copy()
        dictionary_links["repd_lookup_id"] = _canonical_identifier(dictionary_links[dictionary_column])
        dictionary_links = dictionary_links.dropna(subset=["repd_lookup_id"])
        for repd_column in ("repd_ref_id", "repd_old_ref_id"):
            links.append(dictionary_links.merge(
                repd_fields[[repd_column, "cfd_valid_from"]],
                left_on="repd_lookup_id", right_on=repd_column, how="inner",
            )[["dictionary_id", "cfd_valid_from"]])
    if not links:
        return pd.DataFrame(columns=["dictionary_id", "repd_cfd_valid_from"])
    dates = pd.concat(links, ignore_index=True).dropna(subset=["cfd_valid_from"])
    dates = dates.sort_values("cfd_valid_from").drop_duplicates("dictionary_id")
    return dates.rename(columns={"cfd_valid_from": "repd_cfd_valid_from"})


def _capacity_ratio(left: object, right: object) -> float:
    """Return an order-independent capacity ratio, or NaN if either value is unusable."""
    left, right = pd.to_numeric(pd.Series([left, right]), errors="coerce")
    if pd.isna(left) or pd.isna(right) or left <= 0 or right <= 0:
        return float("nan")
    return float(max(left, right) / min(left, right))


def _elexon_lccc_repd_name_capacity_evidence(registry: pd.DataFrame, repd: pd.DataFrame) -> pd.DataFrame:
    """Add only independently corroborated Elexon--LCCC--REPD wind mappings.

    Token-set similarity must be 100, Elexon and LCCC capacities must agree
    within 15%, and a matching REPD CfD record must have a comparable capacity
    and an operational date.  This is a deliberately strict route suitable for
    dated historic calibration, not a general fuzzy matching rule.
    """
    elexon = registry.loc[registry.source.eq("elexon_bmu_reference")].copy()
    elexon = elexon.loc[
        elexon.bmu_id.astype(str).str.match(r"^[ET]_")
        & elexon.technology.fillna("").str.contains("wind", case=False)
        & pd.to_numeric(elexon.capacity_mw, errors="coerce").gt(1)
    ]
    lccc = registry.loc[registry.source.eq("lccc_cfd_contract_portfolio")].copy()
    lccc = lccc.loc[
        lccc.support_status.fillna("").str.contains("live", case=False)
        & lccc.technology.fillna("").str.contains("wind", case=False)
    ].dropna(subset=["cfd_id", "station_name_normalized", "capacity_mw"])
    if elexon.empty or lccc.empty:
        return pd.DataFrame(columns=COLUMNS)

    repd_rows = repd.copy()
    repd_rows["station_name_normalized"] = repd_rows["Site Name"].map(_normalise_name)
    repd_rows["capacity_mw"] = pd.to_numeric(repd_rows["Installed Capacity (MWelec)"], errors="coerce")
    repd_rows["valid_from"] = pd.to_datetime(repd_rows["Operational"], dayfirst=True, errors="coerce")
    repd_rows = repd_rows.loc[
        _nonempty(repd_rows["CfD Allocation Round"])
        & repd_rows.valid_from.notna()
        & repd_rows.capacity_mw.gt(0),
        ["station_name_normalized", "Site Name", "Ref ID", "capacity_mw", "valid_from"],
    ]
    lccc_choices = lccc.station_name_normalized.tolist()
    rows = []
    for item in elexon.itertuples(index=False):
        matches = process.extract(
            item.station_name_normalized, lccc_choices,
            scorer=fuzz.token_set_ratio, score_cutoff=100,
        )
        candidates = []
        for _, score, position in matches:
            contract = lccc.iloc[position]
            ratio = _capacity_ratio(item.capacity_mw, contract.capacity_mw)
            if ratio <= 1.15:
                candidates.append((contract, float(score), ratio))
        if len({str(contract.cfd_id) for contract, _, _ in candidates}) != 1:
            continue
        contract, name_score, ratio = candidates[0]
        repd_candidates = repd_rows.loc[
            repd_rows.station_name_normalized.eq(contract.station_name_normalized)
        ].copy()
        repd_candidates["capacity_ratio"] = repd_candidates.capacity_mw.map(
            lambda value: _capacity_ratio(contract.capacity_mw, value)
        )
        repd_candidates = repd_candidates.loc[repd_candidates.capacity_ratio.le(1.15)]
        if len(repd_candidates) != 1:
            continue
        repd_match = repd_candidates.iloc[0]
        rows.append({
            "bmu_id": item.bmu_id, "dictionary_id": pd.NA,
            "station_name": item.station_name, "support_type": "cfd",
            "confidence": "high", "score": 95,
            "match_route": "elexon_lccc_repd_name_capacity",
            "elexon_current_match": True, "cfd_id": contract.cfd_id,
            "lccc_status": contract.support_status,
            "lccc_technology": contract.technology,
            "lccc_capacity_mw": contract.capacity_mw,
            "elexon_capacity_mw": item.capacity_mw,
            "name_match_score": name_score, "capacity_ratio": ratio,
            "cfd_valid_from": pd.Timestamp(repd_match.valid_from).strftime("%Y-%m-%d"),
            "cfd_date_source": "repd_operational_date_name_capacity",
            "historical_bid_calibration_eligible": True,
            "repd_ref_id": repd_match["Ref ID"], "repd_site_name": repd_match["Site Name"],
            "repd_capacity_mw": repd_match.capacity_mw, "ro_banding": pd.NA,
            "ofgem_accreditation_ref": pd.NA, "ofgem_capacity_mw": pd.NA,
            "ofgem_technology": pd.NA, "ofgem_effective_date": pd.NA,
            "support_conflict": False, "review_reason": "",
        })
    return pd.DataFrame(rows, columns=COLUMNS)

def _cfd_evidence(
    dictionary: pd.DataFrame, lccc: pd.DataFrame, repd: pd.DataFrame,
    elexon_bmus: set[str],
) -> pd.DataFrame:
    """Create legal CfD evidence and attach only deterministic historical dates.

    LCCC's public portfolio identifies the contract but generally has no start
    date.  REPD's operational date is used only after an explicit Power Station
    Dictionary ID link and a REPD CfD allocation-round flag.  Undated legal
    matches remain in the crosswalk for review but are excluded from cached
    historical bid calibration.
    """
    bmus = _dictionary_bmus(dictionary)
    cfd = _split(dictionary[dictionary["cfd_id"].notna()], "cfd_id", "cfd_id").rename(columns={"name": "station_name"})
    candidates = bmus.merge(cfd, on=["dictionary_id", "station_name"], how="inner")
    lccc_fields = lccc[["cfd_id", "support_status", "capacity_mw", "technology", "valid_from"]].drop_duplicates("cfd_id").rename(
        columns={"valid_from": "lccc_valid_from"}
    )
    candidates = candidates.merge(lccc_fields, on="cfd_id", how="left")
    candidates = candidates.merge(_cfd_repd_dates(dictionary, repd), on="dictionary_id", how="left")
    lccc_date = pd.to_datetime(candidates["lccc_valid_from"], errors="coerce")
    repd_date = pd.to_datetime(candidates["repd_cfd_valid_from"], dayfirst=True, errors="coerce")
    candidates["cfd_valid_from"] = lccc_date.fillna(repd_date)
    candidates["cfd_date_source"] = pd.NA
    candidates.loc[lccc_date.notna(), "cfd_date_source"] = "lccc_expected_start_date"
    candidates.loc[lccc_date.isna() & repd_date.notna(), "cfd_date_source"] = "repd_operational_date"
    candidates["historical_bid_calibration_eligible"] = candidates.cfd_valid_from.notna()
    candidates["cfd_valid_from"] = candidates.cfd_valid_from.dt.strftime("%Y-%m-%d")
    candidates["elexon_current_match"] = candidates.bmu_id.isin(elexon_bmus)
    candidates["support_type"] = "cfd"
    candidates["lccc_status"] = candidates.support_status
    candidates["lccc_technology"] = candidates.technology
    candidates["lccc_capacity_mw"] = candidates.capacity_mw
    candidates["repd_ref_id"] = pd.NA
    candidates["repd_site_name"] = pd.NA
    candidates["repd_capacity_mw"] = pd.NA
    candidates["ro_banding"] = pd.NA
    candidates["match_route"] = "dictionary_bmu_cfd_id_lccc"
    candidates["score"] = 85 + candidates.cfd_id.isin(set(lccc.cfd_id.dropna().astype(str))).astype(int) * 10
    candidates.loc[candidates.elexon_current_match, "score"] += 5
    candidates["review_reason"] = ""
    no_lccc = candidates.lccc_status.isna()
    candidates.loc[no_lccc, "review_reason"] = "CfD ID not present in current LCCC portfolio"
    not_live = ~candidates.lccc_status.fillna("").str.contains("Live", case=False)
    candidates.loc[~no_lccc & not_live, "review_reason"] = "Current LCCC status is not Live; do not assume historical validity"
    undated = ~candidates.historical_bid_calibration_eligible
    candidates.loc[~no_lccc & ~not_live & undated, "review_reason"] = (
        "No deterministic LCCC/REPD commissioning date; exclude from historical bid calibration"
    )
    candidates["confidence"] = "high"
    candidates.loc[no_lccc | ~candidates.elexon_current_match, "confidence"] = "medium"
    candidates.loc[~no_lccc & not_live, "confidence"] = "review"
    return candidates
def _normalise_name(value: object) -> str:
    value = str(value).lower().replace("&", " and ")
    value = pd.Series([value]).str.replace(r"\b(wind\s*farm|power\s*station|generating\s*station)\b", "", regex=True).iloc[0]
    value = pd.Series([value]).str.replace(r"[^a-z0-9]+", " ", regex=True).iloc[0]
    return " ".join(str(value).split())


def _ofgem_roc_evidence(dictionary: pd.DataFrame, registry: pd.DataFrame, elexon_bmus: set[str]) -> pd.DataFrame:
    """Use the legal RO accreditation report when it has been supplied locally."""
    ofgem = registry.loc[registry.source.eq("ofgem_ro_accredited_stations")].copy()
    if ofgem.empty:
        return pd.DataFrame(columns=COLUMNS)
    bmus = _dictionary_bmus(dictionary)
    bmus["station_name_normalized"] = bmus.station_name.map(_normalise_name)
    fields = ["source_record_id", "station_name_normalized", "capacity_mw", "technology", "valid_from"]
    ofgem = ofgem[fields].dropna(subset=["station_name_normalized"]).drop_duplicates()
    candidates = bmus.merge(ofgem, on="station_name_normalized", how="inner")
    if candidates.empty:
        return pd.DataFrame(columns=COLUMNS)
    candidates["support_type"] = "roc"
    candidates["elexon_current_match"] = candidates.bmu_id.isin(elexon_bmus)
    candidates["cfd_id"] = pd.NA
    candidates["lccc_status"] = pd.NA
    candidates["lccc_technology"] = pd.NA
    candidates["lccc_capacity_mw"] = pd.NA
    candidates["repd_ref_id"] = pd.NA
    candidates["repd_site_name"] = pd.NA
    candidates["repd_capacity_mw"] = pd.NA
    candidates["ro_banding"] = pd.NA
    candidates["ofgem_accreditation_ref"] = candidates.source_record_id
    candidates["ofgem_capacity_mw"] = candidates.capacity_mw
    candidates["ofgem_technology"] = candidates.technology
    candidates["ofgem_effective_date"] = candidates.valid_from
    candidates["match_route"] = "dictionary_bmu_ofgem_ro_station"
    candidates["score"] = 70 + candidates.elexon_current_match.astype(int) * 5
    candidates["confidence"] = "medium"
    candidates["review_reason"] = "Legal RO accreditation; validate station phase/capacity before automatic use"
    return candidates

def _repd_evidence(
    dictionary: pd.DataFrame, repd: pd.DataFrame, elexon_bmus: set[str]
) -> pd.DataFrame:
    bmus = _dictionary_bmus(dictionary)
    repd = repd.copy()
    repd["new_repd_id"] = repd["Ref ID"].astype(str).str.strip()
    repd["old_repd_id"] = repd["Old Ref ID"].astype(str).str.strip()
    repd["capacity_mw"] = pd.to_numeric(repd["Installed Capacity (MWelec)"], errors="coerce")
    repd["ro_banding"] = pd.to_numeric(repd["RO Banding (ROC/MWh)"], errors="coerce")
    repd["has_cfd"] = _nonempty(repd["CfD Allocation Round"])
    repd["has_roc"] = repd.ro_banding.gt(0)

    links = []
    for dictionary_column, repd_column in (("new_repd_id", "new_repd_id"), ("old_repd_id", "old_repd_id")):
        dictionary_links = _split(dictionary[dictionary[dictionary_column].notna()], dictionary_column, "repd_lookup_id").rename(columns={"name": "station_name"})
        matched = dictionary_links.merge(
            repd,
            left_on="repd_lookup_id",
            right_on=repd_column,
            how="inner",
        )
        links.append(matched)
    linked = pd.concat(links, ignore_index=True) if links else pd.DataFrame()
    if linked.empty:
        return pd.DataFrame(columns=COLUMNS)

    candidates = bmus.merge(linked, on=["dictionary_id", "station_name"], how="inner")
    rows = []
    for support_type, selector in (("cfd", candidates.has_cfd), ("roc", candidates.has_roc)):
        part = candidates.loc[selector].copy()
        if part.empty:
            continue
        part["support_type"] = support_type
        part["elexon_current_match"] = part.bmu_id.isin(elexon_bmus)
        part["cfd_id"] = pd.NA
        part["lccc_status"] = pd.NA
        part["lccc_technology"] = pd.NA
        part["lccc_capacity_mw"] = pd.NA
        part["repd_ref_id"] = part["Ref ID"]
        part["repd_site_name"] = part["Site Name"]
        part["repd_capacity_mw"] = part["capacity_mw"]
        part["match_route"] = "dictionary_bmu_repd_id"
        part["score"] = 75 + part.elexon_current_match.astype(int) * 5
        part["confidence"] = "medium"
        part["review_reason"] = "REPD support evidence; validate against legal register before automatic use"
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)


def build_crosswalk(dictionary_path: Path, registry_path: Path, repd_path: Path, output: Path, review: Path, summary: Path) -> None:
    dictionary = pd.read_csv(dictionary_path, low_memory=False)
    registry = pd.read_csv(registry_path, low_memory=False)
    repd = pd.read_csv(repd_path, encoding="cp1252", low_memory=False)
    elexon_bmus = _elexon_bmus(registry)
    lccc = registry.loc[registry.source.eq("lccc_cfd_contract_portfolio")].copy()
    lccc["cfd_id"] = lccc.cfd_id.astype(str).str.strip()

    evidence_frames = [
        _cfd_evidence(dictionary, lccc, repd, elexon_bmus).reindex(columns=COLUMNS),
        _elexon_lccc_repd_name_capacity_evidence(registry, repd).reindex(columns=COLUMNS),
        _repd_evidence(dictionary, repd, elexon_bmus).reindex(columns=COLUMNS),
        _ofgem_roc_evidence(dictionary, registry, elexon_bmus).reindex(columns=COLUMNS),
    ]
    # Pandas warns when a source has a schema column that is wholly absent
    # (for example, REPD evidence has no LCCC status).  Keep the shared schema
    # while making that intentional absence explicit before concatenation.
    for frame in evidence_frames:
        for column in COLUMNS:
            if frame[column].isna().all():
                frame[column] = ""
    crosswalk = pd.concat([frame for frame in evidence_frames if not frame.empty], ignore_index=True)
    crosswalk["support_conflict"] = crosswalk.groupby("bmu_id")["support_type"].transform("nunique").gt(1)
    crosswalk.loc[crosswalk.support_conflict, "confidence"] = "review"
    crosswalk.loc[crosswalk.support_conflict, "review_reason"] = "Conflicting CfD and ROC evidence on linked station records"
    crosswalk = crosswalk.drop_duplicates().sort_values(["bmu_id", "score"], ascending=[True, False])
    undated_legal_cfd = (crosswalk.support_type.eq("cfd") & crosswalk.confidence.eq("high") & ~crosswalk.historical_bid_calibration_eligible.fillna(False).astype(bool))
    review_rows = crosswalk[(crosswalk.confidence.eq("review")) | (crosswalk.support_conflict) | undated_legal_cfd].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(output, index=False)
    review_rows.to_csv(review, index=False)
    payload = {
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "rows": len(crosswalk),
        "unique_bmus": int(crosswalk.bmu_id.nunique()),
        "by_support_confidence": {f"{support}:{confidence}": int(count) for (support, confidence), count in crosswalk.groupby(["support_type", "confidence"]).size().items()},
        "review_rows": len(review_rows),
        "dictionary_path": str(dictionary_path),
        "registry_path": str(registry_path),
        "repd_path": str(repd_path),
    }
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--repd", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    build_crosswalk(Path(args.dictionary), Path(args.registry), Path(args.repd), Path(args.output), Path(args.review), Path(args.summary))


def _run_from_snakemake() -> None:
    build_crosswalk(
        Path(snakemake.input.dictionary), Path(snakemake.input.registry), Path(snakemake.input.repd),
        Path(snakemake.output.crosswalk), Path(snakemake.output.review), Path(snakemake.output.summary),
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        _run_from_snakemake()
    else:
        main()