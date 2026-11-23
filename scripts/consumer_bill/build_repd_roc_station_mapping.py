"""Create an auditable REPD-to-Ofgem RO accredited-station crosswalk.

The Ofgem station register proves accreditation, but does not supply a station's
historic ROC band or issued certificates.  This script therefore never invents
a band or changes a model input: it writes every candidate, score and capacity
check for review before the crosswalk is used for accounting.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


REPD_TO_RO_TECHNOLOGY = {
    "Wind Onshore": "onshore_wind",
    "Onshore Wind": "onshore_wind",  # legacy spelling
    "Wind Offshore": "offshore_wind",
    "Offshore Wind": "offshore_wind",  # legacy spelling
    "Solar Photovoltaics": "solar_pv",
    "Small Hydro": "hydro",
    "Large Hydro": "hydro",
    "Tidal Stream": "tidal_stream",
    "Shoreline Wave": "wave_power",
    "Biomass (dedicated)": "fuelled",
    "Biomass (co-firing)": "fuelled",
    "EfW Incineration": "fuelled",
    "ACT (not CHP)": "fuelled",
    "Anaerobic Digestion": "fuelled",
    "Advanced Conversion Technologies": "fuelled",
    "Landfill Gas": "landfill_gas",
    "Sewage Sludge Digestion": "sewage_gas",
}


def _normalise(value: object) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(wind\s*farm|solar\s*farm|power\s*station|windpark)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _ofgem_technology(value: object) -> str | None:
    text = str(value or "").lower()
    if "offshore wind" in text:
        return "offshore_wind"
    if "onshore wind" in text or text.startswith("wind with"):
        return "onshore_wind"
    if "photovoltaic" in text:
        return "solar_pv"
    if "landfill gas" in text:
        return "landfill_gas"
    if "sewage gas" in text:
        return "sewage_gas"
    if "tidal" in text:
        return "tidal_stream"
    if "wave" in text:
        return "wave_power"
    if "hydro" in text:
        return "hydro"
    if "fuelled" in text:
        return "fuelled"
    return None


def _capacity_agreement(repd_mw: float, station_kw: float) -> tuple[float | None, bool]:
    if pd.isna(repd_mw) or pd.isna(station_kw) or repd_mw <= 0 or station_kw <= 0:
        return None, False
    station_mw = station_kw / 1000.0
    relative_difference = abs(repd_mw - station_mw) / max(repd_mw, station_mw)
    return relative_difference, relative_difference <= 0.20 or abs(repd_mw - station_mw) <= 1.0


def build_mapping(repd: pd.DataFrame, stations: pd.DataFrame, modelled_year: int) -> tuple[pd.DataFrame, dict]:
    repd = repd.copy()
    stations = stations.copy()
    required_repd = {"Ref ID", "Site Name", "Technology Type", "Installed Capacity (MWelec)", "Country", "Operational"}
    required_stations = {"Accreditation reference", "Station name", "Station DNC in kW", "Technology", "Commissioning date", "Effective date", "Country"}
    if missing := required_repd.difference(repd.columns):
        raise ValueError(f"REPD is missing required columns: {sorted(missing)}")
    if missing := required_stations.difference(stations.columns):
        raise ValueError(f"Ofgem register is missing required columns: {sorted(missing)}")

    repd["operational"] = pd.to_datetime(repd["Operational"], dayfirst=True, errors="coerce")
    repd["repd_capacity_mw"] = pd.to_numeric(repd["Installed Capacity (MWelec)"], errors="coerce")
    repd["ro_technology"] = repd["Technology Type"].map(REPD_TO_RO_TECHNOLOGY)
    repd = repd.loc[
        repd["Country"].isin(["England", "Scotland", "Wales"])
        & repd["ro_technology"].notna()
        & repd["operational"].notna()
        & (repd["operational"] <= pd.Timestamp(f"{modelled_year}-12-31"))
    ].copy()

    stations["station_capacity_kw"] = pd.to_numeric(
        stations["Station DNC in kW"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    stations["ro_technology"] = stations["Technology"].map(_ofgem_technology)
    stations = stations.loc[
        stations["Country"].isin(["England", "Scotland", "Wales"])
        & stations["ro_technology"].notna()
    ].copy()
    stations["normalised_name"] = stations["Station name"].map(_normalise)

    rows = []
    for _, site in repd.iterrows():
        candidates = stations.loc[
            (stations["Country"] == site["Country"])
            & (stations["ro_technology"] == site["ro_technology"])
        ]
        query = _normalise(site["Site Name"])
        exact_candidates = candidates.loc[candidates["normalised_name"].eq(query)].copy() if query else candidates.iloc[0:0]
        exact_candidate_count = len(exact_candidates)
        choices = process.extract(query, candidates["normalised_name"], scorer=fuzz.ratio, limit=2) if query and not candidates.empty else []
        choice = choices[0] if choices else None
        match = candidates.loc[choice[2]] if choice else None
        score = float(choice[1]) if choice else None
        runner_up_score = float(choices[1][1]) if len(choices) > 1 else None

        # Resolve duplicate exact station names by the closest capacity before
        # deciding whether the identity is sufficiently unambiguous.
        if exact_candidate_count > 1:
            exact_candidates["_capacity_difference"] = exact_candidates["station_capacity_kw"].map(
                lambda station_kw: _capacity_agreement(site["repd_capacity_mw"], station_kw)[0]
            )
            exact_candidates = exact_candidates.sort_values("_capacity_difference", na_position="last")
            match = exact_candidates.iloc[0]
            score = 100.0
            runner_up_score = 100.0
        score_gap = score - runner_up_score if runner_up_score is not None else None
        capacity_diff, capacity_ok = _capacity_agreement(
            site["repd_capacity_mw"], match["station_capacity_kw"] if match is not None else None
        )
        exact = bool(score == 100.0)
        # An exact normalised station name, within technology and country, is a
        # stronger identity check than capacity.  DNC and REPD installed
        # capacity commonly describe different vintages or station boundaries;
        # rejecting these exact identities discarded millions of real ROCs.
        if exact and (exact_candidate_count == 1 or capacity_ok):
            status = "high_confidence_exact_name"
        elif score is not None and score >= 93.0 and capacity_ok:
            status = "high_confidence_fuzzy"
        elif score is not None and score >= 97.0 and (score_gap is None or score_gap >= 5.0):
            status = "high_confidence_strong_name"
        elif match is not None:
            status = "review_required"
        else:
            status = "unmatched"
        rows.append({
            "repd_ref_id": site["Ref ID"],
            "repd_site_name": site["Site Name"],
            "repd_technology": site["Technology Type"],
            "repd_capacity_mw": site["repd_capacity_mw"],
            "repd_operational_date": site["Operational"],
            "country": site["Country"],
            "ro_technology": site["ro_technology"],
            "mapping_status": status,
            "name_score": score,
            "exact_name_candidate_count": exact_candidate_count,
            "runner_up_name_score": runner_up_score,
            "name_score_gap": score_gap,
            "capacity_relative_difference": capacity_diff,
            "capacity_agrees": capacity_ok,
            "ro_accreditation_reference": match["Accreditation reference"] if match is not None else None,
            "ro_station_name": match["Station name"] if match is not None else None,
            "ro_station_capacity_kw": match["station_capacity_kw"] if match is not None else None,
            "ro_station_technology": match["Technology"] if match is not None else None,
            "ro_commissioning_date": match["Commissioning date"] if match is not None else None,
            "ro_effective_date": match["Effective date"] if match is not None else None,
        })
    result = pd.DataFrame(rows).sort_values(["mapping_status", "repd_site_name"]).reset_index(drop=True)
    high = result["mapping_status"].str.startswith("high_confidence")
    summary = {
        "modelled_year": modelled_year,
        "eligible_repd_sites": len(result),
        "high_confidence_matches": int(high.sum()),
        "high_confidence_capacity_mw": float(result.loc[high, "repd_capacity_mw"].sum()),
        "mapping_status_counts": result["mapping_status"].value_counts().to_dict(),
        "use_rule": "Only high-confidence rows may establish RO accreditation; this source does not provide historical ROC banding or issued certificates.",
    }
    return result, summary


def write_mapping(repd_path, stations_path, modelled_year, output_csv, output_summary):
    repd = pd.read_csv(repd_path, encoding="latin-1", low_memory=False)
    stations = pd.read_csv(stations_path, low_memory=False)
    result, summary = build_mapping(repd, stations, modelled_year)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_mapping(snakemake.input.repd_csv, snakemake.input.stations_csv, snakemake.params.modelled_year, snakemake.output.mapping_csv, snakemake.output.summary_json)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repd", required=True); parser.add_argument("--stations", required=True); parser.add_argument("--modelled-year", type=int, required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_mapping(args.repd, args.stations, args.modelled_year, args.output_csv, args.output_summary)
