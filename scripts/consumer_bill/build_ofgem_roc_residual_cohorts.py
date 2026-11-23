"""Build an evidence-led ROC ledger from Ofgem's station-level outturn.

The ledger deliberately distinguishes stations confidently represented by a
named PyPSA asset from stations represented through reviewed technology
aggregates. The latter remain Ofgem technology/commissioning cohorts, so the
historical certificate total and assumed 20-year ROC run-off stay visible.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


OFGEM_TO_MODEL_TECHNOLOGY = {
    "Fuelled": "fuelled",
    "Fuelled with a DNC of 50kW or less": "fuelled",
    "Hydro with a DNC greater than 20 MW": "hydro",
    "Hydro with a DNC of 20MW or less": "hydro",
    "Micro hydro": "hydro",
    "Landfill gas": "landfill_gas",
    "Offshore wind": "offshore_wind",
    "Offshore wind - demonstration turbines": "offshore_wind",
    "Offshore wind - floating turbines": "offshore_wind",
    "Onshore wind": "onshore_wind",
    "Photovoltaic": "solar_pv",
    "Sewage gas": "sewage_gas",
    "Tidal stream": "tidal",
}

STATION_COLUMNS = [
    "Accreditation reference", "Station name", "Scheme", "Country", "Technology",
    "ofgem_roc_certificates", "ofgem_eligible_generation_mwh", "output_months",
    "implied_roc_band", "station_capacity_mw", "commissioning_date", "effective_date",
    "station_address",
    "commissioning_year", "roc_expiry_date", "roc_expiry_year", "roc_expiry_basis",
    "model_technology", "model_representation", "direct_identity_matched",
    "representation_mapping_status", "assumed_model_representation",
    "representation_confidence", "mapped_model_generator_count",
    "mapped_model_generators", "evidence",
]

COHORT_COLUMNS = [
    "model_technology", "Technology", "Scheme", "Country", "commissioning_year",
    "roc_expiry_year", "station_count", "station_capacity_mw",
    "ofgem_eligible_generation_mwh", "ofgem_roc_certificates",
    "implied_roc_band", "evidence", "future_use",
]


def _clean_reference(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().removesuffix(".0")


def _station_capacity_mw(registry: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(
        registry["Station DNC in kW"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ) / 1000.0


def _matched_generators(mapping: pd.DataFrame) -> pd.DataFrame:
    """Return one record per Ofgem station with confidently mapped generators."""
    if mapping.empty:
        return pd.DataFrame(columns=[
            "Accreditation reference", "mapped_model_generator_count", "mapped_model_generators",
        ])
    required = {"mapping_status", "ro_accreditation_reference", "generator"}
    if missing := required.difference(mapping.columns):
        raise ValueError(f"PyPSA ROC mapping is missing required columns: {sorted(missing)}")
    mapping = mapping.copy()
    mapping["Accreditation reference"] = mapping["ro_accreditation_reference"].map(_clean_reference)
    matched = mapping.loc[
        mapping["mapping_status"].astype(str).str.startswith("high_confidence")
        & mapping["Accreditation reference"].ne("")
    ].copy()
    return matched.groupby("Accreditation reference", as_index=False).agg(
        mapped_model_generator_count=("generator", "nunique"),
        mapped_model_generators=("generator", lambda values: " | ".join(sorted(map(str, set(values))))),
    )


def _assumed_representations(mapping: pd.DataFrame) -> pd.DataFrame:
    """Return the explicit aggregate representation for non-direct stations."""
    required = {
        "mapping_status", "ro_accreditation_reference", "representation_id",
        "representation_confidence",
    }
    if mapping.empty or not required.issubset(mapping.columns):
        return pd.DataFrame(columns=[
            "Accreditation reference", "representation_mapping_status",
            "assumed_model_representation", "representation_confidence",
            "representation_assumption",
        ])
    assumed = mapping.loc[
        mapping["mapping_status"].astype(str).str.startswith("assumed_")
    ].copy()
    assumed["Accreditation reference"] = assumed["ro_accreditation_reference"].map(_clean_reference)
    assumed["representation_assumption"] = assumed.get(
        "representation_assumption",
        assumed.get("mapping_evidence", pd.Series("", index=assumed.index)),
    ).fillna("")
    assumed = assumed.drop_duplicates("Accreditation reference")
    return assumed.rename(columns={
        "mapping_status": "representation_mapping_status",
        "representation_id": "assumed_model_representation",
    })[[
        "Accreditation reference", "representation_mapping_status",
        "assumed_model_representation", "representation_confidence",
        "representation_assumption",
    ]]


def build_roc_ledgers(
    outturn: pd.DataFrame,
    registry: pd.DataFrame,
    mapping: pd.DataFrame,
    roc_eligibility_years: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return full official station ledger, unmapped cohorts and a checked summary."""
    required_outturn = {
        "Accreditation reference", "Station name", "Scheme", "Country", "Technology",
        "ofgem_roc_certificates", "ofgem_eligible_generation_mwh",
    }
    if missing := required_outturn.difference(outturn.columns):
        raise ValueError(f"Ofgem ROC outturn is missing required columns: {sorted(missing)}")
    required_registry = {
        "Accreditation reference", "Station DNC in kW", "Commissioning date", "Effective date", "Station address",
    }
    if missing := required_registry.difference(registry.columns):
        raise ValueError(f"Ofgem ROC registry is missing required columns: {sorted(missing)}")

    registry = registry.copy()
    registry["Accreditation reference"] = registry["Accreditation reference"].map(_clean_reference)
    registry["station_capacity_mw"] = _station_capacity_mw(registry)
    registry["commissioning_date"] = pd.to_datetime(
        registry["Commissioning date"], dayfirst=True, errors="coerce"
    )
    registry["effective_date"] = pd.to_datetime(
        registry["Effective date"], dayfirst=True, errors="coerce"
    )
    registry["station_address"] = registry["Station address"].fillna("")
    registry = registry.drop_duplicates("Accreditation reference")[[
        "Accreditation reference", "station_capacity_mw", "commissioning_date", "effective_date", "station_address",
    ]]

    ledger = outturn.copy()
    ledger["Accreditation reference"] = ledger["Accreditation reference"].map(_clean_reference)
    ledger = ledger.merge(registry, on="Accreditation reference", how="left")
    ledger = ledger.merge(_matched_generators(mapping), on="Accreditation reference", how="left")
    ledger = ledger.merge(_assumed_representations(mapping), on="Accreditation reference", how="left")
    ledger["mapped_model_generator_count"] = pd.to_numeric(
        ledger["mapped_model_generator_count"], errors="coerce"
    ).fillna(0).astype(int)
    ledger["mapped_model_generators"] = ledger["mapped_model_generators"].fillna("")
    for column in [
        "representation_mapping_status", "assumed_model_representation",
        "representation_confidence", "representation_assumption",
    ]:
        ledger[column] = ledger[column].fillna("")
    ledger["commissioning_year"] = ledger["commissioning_date"].dt.year.astype("Int64")
    # The plant commissioning date is useful cohort metadata, but the RO
    # duration starts from accreditation/effective date.  Some long-lived
    # hydro stations illustrate why commissioning-date expiry would be wrong.
    roc_start = ledger["effective_date"].fillna(ledger["commissioning_date"])
    ledger["roc_expiry_date"] = roc_start + pd.DateOffset(years=roc_eligibility_years)
    ledger["roc_expiry_year"] = ledger["roc_expiry_date"].dt.year.astype("Int64")
    ledger["roc_expiry_basis"] = ledger["effective_date"].notna().map({
        True: "Ofgem effective date + eligibility duration",
        False: "commissioning-date fallback + eligibility duration",
    })
    ledger["model_technology"] = ledger["Technology"].map(OFGEM_TO_MODEL_TECHNOLOGY).fillna("unclassified")
    ledger["direct_identity_matched"] = ledger["mapped_model_generator_count"].gt(0)
    ledger["model_representation"] = "official_unrepresented"
    ledger.loc[
        ledger["representation_mapping_status"].str.startswith("assumed_"),
        "model_representation",
    ] = ledger.loc[
        ledger["representation_mapping_status"].str.startswith("assumed_"),
        "representation_mapping_status",
    ]
    ledger.loc[ledger["direct_identity_matched"], "model_representation"] = "named_station_evidenced"
    ledger["evidence"] = ledger["representation_assumption"]
    ledger.loc[ledger["direct_identity_matched"], "evidence"] = (
        "Ofgem 2020 certificate outturn plus high-confidence PyPSA station match."
    )
    ledger.loc[ledger["model_representation"].eq("official_unrepresented"), "evidence"] = (
        "Ofgem 2020 certificate outturn; no direct or aggregate model representation."
    )
    for column in ["ofgem_roc_certificates", "ofgem_eligible_generation_mwh"]:
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce").fillna(0.0)
    ledger["implied_roc_band"] = (
        ledger["ofgem_roc_certificates"]
        / ledger["ofgem_eligible_generation_mwh"].where(ledger["ofgem_eligible_generation_mwh"] > 0)
    )
    ledger = ledger.reindex(columns=STATION_COLUMNS).sort_values(
        ["model_representation", "ofgem_roc_certificates"], ascending=[True, False]
    ).reset_index(drop=True)

    residual = ledger.loc[~ledger["direct_identity_matched"]].copy()
    cohorts = residual.groupby([
        "model_technology", "Technology", "Scheme", "Country", "commissioning_year", "roc_expiry_year",
    ], dropna=False, as_index=False).agg(
        station_count=("Accreditation reference", "nunique"),
        station_capacity_mw=("station_capacity_mw", "sum"),
        ofgem_eligible_generation_mwh=("ofgem_eligible_generation_mwh", "sum"),
        ofgem_roc_certificates=("ofgem_roc_certificates", "sum"),
    )
    cohorts["implied_roc_band"] = (
        cohorts["ofgem_roc_certificates"]
        / cohorts["ofgem_eligible_generation_mwh"].where(cohorts["ofgem_eligible_generation_mwh"] > 0)
    )
    cohorts["evidence"] = "Ofgem station-level 2020 certificate outturn, grouped only after identity matching."
    cohorts["future_use"] = (
        f"Historic residual cohort; assume ROC eligibility ends {roc_eligibility_years} years after Ofgem effective date "
        "subject to a scenario-specific future generation assumption."
    )
    cohorts = cohorts.reindex(columns=COHORT_COLUMNS).sort_values(
        ["roc_expiry_year", "model_technology"], na_position="last"
    ).reset_index(drop=True)

    official_certificates = float(ledger["ofgem_roc_certificates"].sum())
    named_certificates = float(ledger.loc[
        ledger["model_representation"].eq("named_station_evidenced"), "ofgem_roc_certificates"
    ].sum())
    assumed_certificates = float(ledger.loc[
        ledger["model_representation"].str.startswith("assumed_"), "ofgem_roc_certificates"
    ].sum())
    unrepresented_certificates = float(ledger.loc[
        ledger["model_representation"].eq("official_unrepresented"), "ofgem_roc_certificates"
    ].sum())
    residual_certificates = float(cohorts["ofgem_roc_certificates"].sum())
    if abs(official_certificates - named_certificates - residual_certificates) > 1e-6:
        raise AssertionError("Named and residual ROC ledgers do not reconcile to Ofgem's official total")
    summary = {
        "method": "Ofgem station-level certificate ledger with direct-identity and assumed-aggregate separation",
        "roc_eligibility_years_assumption": roc_eligibility_years,
        "official_roc_certificates": official_certificates,
        "official_eligible_generation_mwh": float(ledger["ofgem_eligible_generation_mwh"].sum()),
        "named_station_count": int(ledger.loc[ledger["model_representation"].eq("named_station_evidenced"), "Accreditation reference"].nunique()),
        "named_station_roc_certificates": named_certificates,
        "direct_identity_residual_station_count": int(residual["Accreditation reference"].nunique()),
        "direct_identity_residual_roc_certificates": residual_certificates,
        "named_station_coverage_of_official_certificates": named_certificates / official_certificates if official_certificates else None,
        "assumed_aggregate_station_count": int(ledger["model_representation"].str.startswith("assumed_").sum()),
        "assumed_aggregate_roc_certificates": assumed_certificates,
        "unrepresented_station_count": int(ledger["model_representation"].eq("official_unrepresented").sum()),
        "unrepresented_roc_certificates": unrepresented_certificates,
        "total_station_representation_coverage": float(ledger["model_representation"].ne("official_unrepresented").mean()) if len(ledger) else None,
        "total_certificate_representation_coverage": (official_certificates - unrepresented_certificates) / official_certificates if official_certificates else None,
        "cohort_count": int(len(cohorts)),
        "certificate_reconciliation_difference": official_certificates - named_certificates - residual_certificates,
        "historical_bill_rule": "Use all official certificates (named plus official residual) for the historical ROC ledger.",
        "roc_expiry_basis": "Ofgem effective (accreditation) date plus the configured eligibility duration; commissioning date is retained as cohort metadata only.",
        "future_rule": "Use named PyPSA stations directly; project assumed aggregate cohorts by technology and expire them by the stated effective-date assumption.",
    }
    return ledger, cohorts, summary


def _read_mapping(mapping_path) -> pd.DataFrame:
    paths = list(mapping_path) if isinstance(mapping_path, (list, tuple)) else [mapping_path]
    mappings = [pd.read_csv(path) for path in paths]
    return pd.concat(mappings, ignore_index=True) if mappings else pd.DataFrame()


def write_ledgers(outturn_path, registry_path, mapping_path, station_output, cohort_output, summary_output, roc_eligibility_years=20):
    ledger, cohorts, summary = build_roc_ledgers(
        pd.read_csv(outturn_path), pd.read_csv(registry_path), _read_mapping(mapping_path), roc_eligibility_years
    )
    for output in [station_output, cohort_output, summary_output]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(station_output, index=False)
    cohorts.to_csv(cohort_output, index=False)
    Path(summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_ledgers(
        snakemake.input.outturn_csv, snakemake.input.registry_csv, snakemake.input.mapping_csv,
        snakemake.output.station_ledger_csv, snakemake.output.residual_cohorts_csv,
        snakemake.output.summary_json, snakemake.params.roc_eligibility_years,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outturn", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--station-output", required=True)
    parser.add_argument("--cohort-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--roc-eligibility-years", type=int, default=20)
    args = parser.parse_args()
    write_ledgers(
        args.outturn, args.registry, args.mapping, args.station_output, args.cohort_output,
        args.summary_output, args.roc_eligibility_years,
    )
