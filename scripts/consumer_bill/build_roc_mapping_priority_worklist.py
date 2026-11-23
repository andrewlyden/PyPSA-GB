"""Create a review worklist for high-volume, officially unmapped ROC stations.

This script does not change any support label.  It makes the residual ledger
actionable by comparing each large Ofgem station with both the raw REPD project
register and the generator identities supplied to the PyPSA build.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


REPD_TO_TECHNOLOGY = {
    "Wind Onshore": "onshore_wind", "Onshore Wind": "onshore_wind",
    "Wind Offshore": "offshore_wind", "Offshore Wind": "offshore_wind",
    "Solar Photovoltaics": "solar_pv", "Small Hydro": "hydro", "Large Hydro": "hydro",
    "Tidal Stream": "tidal", "Shoreline Wave": "wave",
    "Biomass (dedicated)": "fuelled", "Biomass (co-firing)": "fuelled",
    "EfW Incineration": "fuelled", "ACT (not CHP)": "fuelled",
    "Anaerobic Digestion": "fuelled", "Advanced Conversion Technologies": "fuelled",
    "Landfill Gas": "landfill_gas", "Sewage Sludge Digestion": "sewage_gas",
}

CARRIER_TO_TECHNOLOGY = {
    "wind_onshore": "onshore_wind", "wind_offshore": "offshore_wind",
    "solar_pv": "solar_pv", "small_hydro": "hydro", "large_hydro": "hydro",
    "tidal_stream": "tidal", "shoreline_wave": "wave", "biomass": "fuelled",
    "biogas": "fuelled", "advanced_biofuel": "fuelled", "waste_to_energy": "fuelled",
    "landfill_gas": "landfill_gas", "sewage_gas": "sewage_gas",
}


def normalise(value: object) -> str:
    text = str(value or "").lower().replace("_", " ").replace("&", " and ")
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(
        r"\b(wind\s*farm|solar\s*farm|power\s*station|energy\s*from\s*waste|"
        r"biomass|chp|plant|renewable|energy)\b", " ", text,
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def capacity_difference(station_mw: float, candidate_mw: float) -> float | None:
    if pd.isna(station_mw) or pd.isna(candidate_mw) or station_mw <= 0 or candidate_mw <= 0:
        return None
    return abs(station_mw - candidate_mw) / max(station_mw, candidate_mw)


def capacity_score(difference: float | None) -> float:
    if difference is None:
        return 0.0
    if difference <= 0.20 or difference * 100 <= 1.0:
        return 100.0
    return max(0.0, 100.0 * (1.0 - difference))


def date_score(station_date: pd.Timestamp, candidate_date: pd.Timestamp) -> float:
    if pd.isna(station_date) or pd.isna(candidate_date):
        return 0.0
    years = abs((station_date - candidate_date).days) / 365.25
    if years <= 1:
        return 100.0
    if years <= 3:
        return 75.0
    if years <= 5:
        return 50.0
    return 0.0


def _reconcile_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def load_repd_candidates(path: str) -> pd.DataFrame:
    repd = pd.read_csv(path, encoding="latin-1", low_memory=False)
    repd = repd.copy()
    repd["technology"] = repd["Technology Type"].map(REPD_TO_TECHNOLOGY)
    repd = repd.loc[repd["technology"].notna()].copy()
    repd["capacity_mw"] = pd.to_numeric(repd["Installed Capacity (MWelec)"], errors="coerce")
    repd["operational_date"] = pd.to_datetime(repd["Operational"], dayfirst=True, errors="coerce")
    result = pd.DataFrame({
        "candidate_source": "REPD",
        "candidate_id": repd["Ref ID"].astype(str),
        "candidate_name": repd["Site Name"],
        "candidate_technology": repd["technology"],
        "candidate_country": repd["Country"],
        "candidate_capacity_mw": repd["capacity_mw"],
        "candidate_date": repd["operational_date"],
        "candidate_address": repd["Address"].fillna(""),
    })
    return result.dropna(subset=["candidate_name", "candidate_technology"])


def load_pypsa_candidates(path: str) -> pd.DataFrame:
    generators = pd.read_csv(path, index_col=0, low_memory=False).copy()
    generators["technology"] = generators["carrier"].astype(str).map(CARRIER_TO_TECHNOLOGY)
    generators = generators.loc[generators["technology"].notna()].copy()
    names = generators.get("source_site_name", pd.Series(index=generators.index, dtype=object)).fillna("")
    names = names.where(names.astype(str).str.strip().ne(""), generators.index.to_series())
    dates = generators.get("operational_date", pd.Series(index=generators.index, dtype=object))
    result = pd.DataFrame({
        "candidate_source": "PyPSA_generator_register",
        "candidate_id": generators.index.astype(str),
        "candidate_name": names.astype(str),
        "candidate_technology": generators["technology"],
        "candidate_country": pd.NA,
        "candidate_capacity_mw": pd.to_numeric(generators["p_nom"], errors="coerce"),
        "candidate_date": pd.to_datetime(dates, dayfirst=True, errors="coerce"),
        "candidate_address": "",
    })
    return result.dropna(subset=["candidate_name", "candidate_technology"])


def candidate_rows(station: pd.Series, candidates: pd.DataFrame, limit: int) -> list[dict]:
    pool = candidates.loc[candidates["candidate_technology"].eq(station["model_technology"])].copy()
    if station.get("Country") in {"England", "Scotland", "Wales"}:
        country_pool = pool.loc[pool["candidate_country"].isna() | pool["candidate_country"].eq(station["Country"])]
        if not country_pool.empty:
            pool = country_pool
    if pool.empty:
        return []
    query = normalise(station["Station name"])
    station_date = pd.to_datetime(station.get("commissioning_date"), errors="coerce")
    station_capacity = pd.to_numeric(pd.Series([station.get("station_capacity_mw")]), errors="coerce").iloc[0]
    pool["name_score"] = pool["candidate_name"].map(lambda value: fuzz.ratio(query, normalise(value)))
    pool["capacity_relative_difference"] = pool["candidate_capacity_mw"].map(
        lambda value: capacity_difference(station_capacity, value)
    )
    pool["capacity_score"] = pool["capacity_relative_difference"].map(capacity_score)
    pool["date_score"] = pool["candidate_date"].map(lambda value: date_score(station_date, value))
    pool["joint_score"] = (
        0.65 * pool["name_score"] + 0.25 * pool["capacity_score"] + 0.10 * pool["date_score"]
    )
    pool["capacity_agrees"] = pool["capacity_relative_difference"].le(0.20) | (
        (pool["candidate_capacity_mw"] - station_capacity).abs().le(1.0)
    )
    return pool.sort_values(
        ["joint_score", "name_score", "capacity_score"], ascending=False
    ).head(limit).to_dict("records")


def triage(best: dict | None) -> str:
    if best is None:
        return "no_candidate"
    if best["name_score"] == 100.0 and best["capacity_agrees"]:
        return "automatic_candidate"
    if best["joint_score"] >= 75.0 and best["capacity_agrees"]:
        return "manual_review_high_priority"
    if best["joint_score"] >= 55.0:
        return "manual_review"
    return "no_plausible_candidate"


def build_worklist(ledger: pd.DataFrame, repd: pd.DataFrame, pypsa: pd.DataFrame, minimum_capacity_mw: float = 10.0):
    residual = ledger.loc[
        ledger["model_representation"].ne("named_station_evidenced")
        & pd.to_numeric(ledger["station_capacity_mw"], errors="coerce").ge(minimum_capacity_mw)
    ].copy()
    residual["commissioning_date"] = pd.to_datetime(residual["commissioning_date"], errors="coerce")
    all_candidates = pd.concat([repd, pypsa], ignore_index=True)
    long_rows, worklist_rows = [], []
    for _, station in residual.iterrows():
        matches = candidate_rows(station, all_candidates, limit=3)
        best = matches[0] if matches else None
        worklist_rows.append({
            "Accreditation reference": station["Accreditation reference"],
            "Station name": station["Station name"],
            "Technology": station["Technology"],
            "Country": station["Country"],
            "station_capacity_mw": station["station_capacity_mw"],
            "commissioning_date": station["commissioning_date"],
            "effective_date": station.get("effective_date"),
            "station_address": station.get("station_address", ""),
            "ofgem_roc_certificates": station["ofgem_roc_certificates"],
            "ofgem_eligible_generation_mwh": station["ofgem_eligible_generation_mwh"],
            "roc_expiry_date": station.get("roc_expiry_date"),
            "triage": triage(best),
            "best_candidate_source": best["candidate_source"] if best else "",
            "best_candidate_id": best["candidate_id"] if best else "",
            "best_candidate_name": best["candidate_name"] if best else "",
            "best_candidate_capacity_mw": best["candidate_capacity_mw"] if best else None,
            "best_name_score": best["name_score"] if best else None,
            "best_capacity_relative_difference": best["capacity_relative_difference"] if best else None,
            "best_date_score": best["date_score"] if best else None,
            "best_joint_score": best["joint_score"] if best else None,
        })
        for rank, match in enumerate(matches, start=1):
            long_rows.append({
                "Accreditation reference": station["Accreditation reference"],
                "Station name": station["Station name"],
                "Technology": station["Technology"],
                "Country": station["Country"],
                "station_capacity_mw": station["station_capacity_mw"],
                "ofgem_roc_certificates": station["ofgem_roc_certificates"],
                "candidate_rank": rank,
                **{key: match[key] for key in [
                    "candidate_source", "candidate_id", "candidate_name", "candidate_technology",
                    "candidate_country", "candidate_capacity_mw", "candidate_date", "candidate_address",
                    "name_score", "capacity_relative_difference", "capacity_score", "date_score",
                    "joint_score", "capacity_agrees",
                ]},
            })
    worklist = pd.DataFrame(worklist_rows).sort_values("ofgem_roc_certificates", ascending=False)
    candidates = pd.DataFrame(long_rows).sort_values(
        ["ofgem_roc_certificates", "candidate_rank"], ascending=[False, True]
    )
    summary = {
        "scope": f"ROC stations without direct model identity and Ofgem DNC >= {minimum_capacity_mw:g} MW",
        "station_count": int(len(worklist)),
        "roc_certificates": float(worklist["ofgem_roc_certificates"].sum()),
        "share_of_official_unmapped_rocs": float(
            worklist["ofgem_roc_certificates"].sum()
            / ledger.loc[ledger["model_representation"].ne("named_station_evidenced"), "ofgem_roc_certificates"].sum()
        ),
        "triage_counts": worklist["triage"].value_counts().to_dict(),
        "triage_roc_certificates": worklist.groupby("triage")["ofgem_roc_certificates"].sum().to_dict(),
        "rule": "Candidates are for improving direct identity evidence. Residual stations already retain an explicit assumed aggregate representation for complete accounting.",
    }
    return worklist, candidates, summary


def write_worklist(ledger_path, repd_path, generators_path, worklist_output, candidates_output, summary_output, minimum_capacity_mw=10.0):
    worklist, candidates, summary = build_worklist(
        pd.read_csv(ledger_path), load_repd_candidates(repd_path), load_pypsa_candidates(generators_path), minimum_capacity_mw
    )
    for output in [worklist_output, candidates_output, summary_output]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    worklist.to_csv(worklist_output, index=False)
    candidates.to_csv(candidates_output, index=False)
    Path(summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_worklist(
        snakemake.input.station_ledger_csv, snakemake.input.repd_csv, snakemake.input.generators_csv,
        snakemake.output.worklist_csv, snakemake.output.candidates_csv, snakemake.output.summary_json,
        snakemake.params.minimum_capacity_mw,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-ledger", required=True)
    parser.add_argument("--repd", required=True)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--worklist-output", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--minimum-capacity-mw", type=float, default=10.0)
    args = parser.parse_args()
    write_worklist(
        args.station_ledger, args.repd, args.generators, args.worklist_output,
        args.candidates_output, args.summary_output, args.minimum_capacity_mw,
    )
