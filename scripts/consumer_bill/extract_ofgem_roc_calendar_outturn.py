"""Extract a calendar-year GB ROC issuance outturn from Ofgem's public export."""

import argparse
import json
from pathlib import Path

import pandas as pd


USECOLS = [
    "Accreditation reference", "Station name", "Scheme", "Country", "Technology",
    "Output period", "Number of certificates", "MWh per certificate",
    "Start certificate number", "End certificate number",
]
GB_COUNTRIES = {"England", "Scotland", "Wales"}


def extract_calendar_outturn(
    certificates_path: str,
    year: int,
    period_start: str | None = None,
    period_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    start_month = pd.Timestamp(period_start).to_period("M") if period_start else pd.Period(f"{year}-01", freq="M")
    end_month = pd.Timestamp(period_end).to_period("M") if period_end else pd.Period(f"{year}-12", freq="M")
    rows = []
    for chunk in pd.read_csv(certificates_path, usecols=USECOLS, chunksize=200_000, low_memory=False):
        output = pd.to_datetime(chunk["Output period"], format="%b %Y", errors="coerce")
        output_month = output.dt.to_period("M")
        rows.append(chunk.loc[
            output_month.between(start_month, end_month)
            & chunk["Country"].isin(GB_COUNTRIES)
        ].copy())
    certificates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=USECOLS)
    certificates["number_of_certificates"] = pd.to_numeric(
        certificates["Number of certificates"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ).fillna(0.0)
    certificates["mwh_per_certificate"] = pd.to_numeric(certificates["MWh per certificate"], errors="coerce")
    certificates["eligible_generation_mwh"] = certificates["number_of_certificates"] * certificates["mwh_per_certificate"]
    certificates["implied_roc_band"] = certificates["number_of_certificates"] / certificates["eligible_generation_mwh"].where(certificates["eligible_generation_mwh"] > 0)
    certificate_key = ["Start certificate number", "End certificate number"]
    duplicate_count = int(certificates.duplicated(certificate_key, keep=False).sum())
    stations = certificates.groupby([
        "Accreditation reference", "Station name", "Scheme", "Country", "Technology"
    ], dropna=False, as_index=False).agg(
        ofgem_roc_certificates=("number_of_certificates", "sum"),
        ofgem_eligible_generation_mwh=("eligible_generation_mwh", "sum"),
        output_months=("Output period", "nunique"),
    )
    stations["implied_roc_band"] = stations["ofgem_roc_certificates"] / stations["ofgem_eligible_generation_mwh"].where(stations["ofgem_eligible_generation_mwh"] > 0)
    technology = stations.groupby(["Technology", "Country"], as_index=False).agg(
        ofgem_roc_certificates=("ofgem_roc_certificates", "sum"),
        ofgem_eligible_generation_mwh=("ofgem_eligible_generation_mwh", "sum"),
        station_count=("Accreditation reference", "nunique"),
    )
    technology["implied_roc_band"] = technology["ofgem_roc_certificates"] / technology["ofgem_eligible_generation_mwh"].where(technology["ofgem_eligible_generation_mwh"] > 0)
    summary = {
        "calendar_year": year,
        "period_start_month": str(start_month),
        "period_end_month": str(end_month),
        "scope": "Great Britain (England, Scotland and Wales)",
        "ofgem_roc_certificates": float(stations["ofgem_roc_certificates"].sum()),
        "ofgem_eligible_generation_mwh": float(stations["ofgem_eligible_generation_mwh"].sum()),
        "accredited_station_count": int(stations["Accreditation reference"].nunique()),
        "certificate_record_count": len(certificates),
        "duplicate_certificate_range_records": duplicate_count,
        "source_note": "Calendar-year filter is based on Ofgem's Output period, not issue date or compliance period.",
    }
    return stations, technology, summary


def write_outturn(
    certificates_path, year, output_stations, output_technology, output_summary,
    period_start=None, period_end=None,
):
    stations, technology, summary = extract_calendar_outturn(
        certificates_path, year, period_start, period_end
    )
    Path(output_stations).parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(output_stations, index=False)
    technology.to_csv(output_technology, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_outturn(
        snakemake.input.certificates_csv, snakemake.params.year,
        snakemake.output.stations_csv, snakemake.output.technology_csv,
        snakemake.output.summary_json, snakemake.params.period_start,
        snakemake.params.period_end,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", required=True); parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--period-start"); parser.add_argument("--period-end")
    parser.add_argument("--output-stations", required=True); parser.add_argument("--output-technology", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_outturn(
        args.certificates, args.year, args.output_stations, args.output_technology,
        args.output_summary, args.period_start, args.period_end,
    )
