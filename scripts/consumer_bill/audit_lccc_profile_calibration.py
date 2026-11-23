"""Audit raw renewable profiles against calendar-year LCCC CfD generation.

This is a pre-solve diagnostic: it uses official settled generation and the
weather profile before a scenario performance factor is applied.  It therefore
does not require, or modify, a market optimisation.
"""

import json
import re

import numpy as np
import pandas as pd


PROFILE_CARRIERS = {"wind_offshore", "wind_onshore", "solar_pv"}


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _read_profiles(paths, year):
    result = {}
    for path in paths:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        carrier = name.rsplit("_", 1)[0]
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame = frame.loc[str(year)]
        result[carrier] = frame.resample("D").sum()
    return result


def _first_present(group, column):
    values = group.get(column, pd.Series(dtype=object)).dropna()
    values = values[values.astype(str).str.strip().ne("")]
    return values.iloc[0] if not values.empty else None


def _build_assets(generator_paths):
    frames = [pd.read_csv(path, low_memory=False) for path in generator_paths]
    table = pd.concat(frames, ignore_index=True, sort=False)
    table["cfd_id"] = table.get("cfd_id", "").fillna("").astype(str).str.strip()
    table = table.loc[table["cfd_id"].ne("")].copy()
    rows = []
    for cfd_id, group in table.groupby("cfd_id", sort=True):
        rows.append({
            "CfD_ID": cfd_id,
            "source_site_name": _first_present(group, "source_site_name"),
            "carrier": _first_present(group, "carrier"),
            "resource_region": _first_present(group, "resource_region") or "GB",
            "allocation_share": pd.to_numeric(
                pd.Series([_first_present(group, "cfd_allocation_share_of_source")]),
                errors="coerce",
            ).fillna(1.0).iloc[0],
            "asset_rows_seen": int(len(group)),
        })
    return pd.DataFrame(rows)


def _profile_column(frame, source_name):
    if source_name in frame.columns:
        return source_name
    candidates = [column for column in frame.columns if _key(column) == _key(source_name)]
    return candidates[0] if len(candidates) == 1 else None


def _daily_actual(actuals, cfd_id, year):
    subset = actuals.loc[actuals["CfD_ID"].eq(cfd_id)].copy()
    subset["date"] = subset["Settlement_Date"].dt.normalize()
    subset = subset.loc[subset["date"].dt.year.eq(year)]
    return subset.groupby("date")["CFD_Generation_MWh"].sum().sort_index()


def _factor(actual, raw):
    return float(actual.sum() / raw.sum()) if raw.sum() > 0 else np.nan


def run(actuals_path, generator_paths, profile_paths, year):
    actuals = pd.read_csv(actuals_path, parse_dates=["Settlement_Date"])
    actuals["CfD_ID"] = actuals["CfD_ID"].astype(str).str.strip()
    actuals["CFD_Generation_MWh"] = pd.to_numeric(actuals["CFD_Generation_MWh"], errors="coerce").fillna(0.0)
    actuals = actuals.loc[actuals["Settlement_Date"].dt.year.eq(year)].copy()
    contract_metadata = actuals.groupby("CfD_ID", as_index=False).agg(
        cfd_unit=("Name_of_CfD_Unit", "first"), technology=("Technology", "first")
    )
    assets = _build_assets(generator_paths)
    profiles = _read_profiles(profile_paths, year)
    contracts = contract_metadata.merge(assets, on="CfD_ID", how="left")
    contract_rows, monthly_rows = [], []

    for record in contracts.to_dict("records"):
        actual = _daily_actual(actuals, record["CfD_ID"], year)
        carrier = record.get("carrier")
        source = record.get("source_site_name")
        profile = profiles.get(carrier)
        column = _profile_column(profile, source) if profile is not None and pd.notna(source) else None
        mapped = carrier in PROFILE_CARRIERS and column is not None
        row = dict(record)
        row.update({"profile_mapped": bool(mapped), "profile_column": column or ""})
        if mapped:
            raw = profile.loc[actual.index, column].astype(float) * float(record["allocation_share"])
            comparison = pd.concat([actual.rename("actual_generation_mwh"), raw.rename("raw_profile_mwh")], axis=1)
            row.update({
                "actual_generation_mwh": float(actual.sum()),
                "raw_profile_mwh": float(raw.sum()),
                "implied_settled_output_factor": _factor(actual, raw),
                "daily_correlation": float(comparison.corr().iloc[0, 1]) if len(comparison) > 2 else np.nan,
                "observed_days": int(len(comparison)),
                "profile_note": "Settled output includes real availability and curtailment; factor is validation evidence, not a future assumption.",
            })
            for month, values in comparison.groupby(comparison.index.month):
                monthly_rows.append({
                    **{key: row.get(key) for key in ["CfD_ID", "cfd_unit", "technology", "carrier", "resource_region", "allocation_share"]},
                    "calendar_month": int(month),
                    "actual_generation_mwh": float(values["actual_generation_mwh"].sum()),
                    "raw_profile_mwh": float(values["raw_profile_mwh"].sum()),
                    "implied_settled_output_factor": _factor(values["actual_generation_mwh"], values["raw_profile_mwh"]),
                    "daily_correlation": float(values.corr().iloc[0, 1]) if len(values) > 2 else np.nan,
                })
        else:
            row.update({
                "actual_generation_mwh": float(actual.sum()), "raw_profile_mwh": np.nan,
                "implied_settled_output_factor": np.nan, "daily_correlation": np.nan,
                "observed_days": int(len(actual)),
                "profile_note": "No site-level weather profile is expected for this carrier or the source profile could not be located.",
            })
        contract_rows.append(row)

    contract_table = pd.DataFrame(contract_rows).sort_values("actual_generation_mwh", ascending=False)
    monthly_table = pd.DataFrame(monthly_rows)
    if monthly_table.empty:
        cohort_table = pd.DataFrame()
    else:
        cohort_table = monthly_table.groupby(
            ["carrier", "resource_region", "calendar_month"], as_index=False
        ).agg(
            contract_count=("CfD_ID", "nunique"),
            actual_generation_mwh=("actual_generation_mwh", "sum"),
            raw_profile_mwh=("raw_profile_mwh", "sum"),
        )
        cohort_table["implied_settled_output_factor"] = (
            cohort_table["actual_generation_mwh"] / cohort_table["raw_profile_mwh"].where(cohort_table["raw_profile_mwh"] > 0)
        )
    mapped = contract_table.loc[contract_table["profile_mapped"]]
    summary = {
        "calendar_year": int(year),
        "treatment": "Pre-solve validation against LCCC settled output. It must not be copied directly into future physical availability assumptions.",
        "contracts_with_lccc_generation": int(len(contract_table)),
        "contracts_with_site_profile": int(len(mapped)),
        "lccc_generation_mwh": float(contract_table["actual_generation_mwh"].sum()),
        "profile_mapped_lccc_generation_mwh": float(mapped["actual_generation_mwh"].sum()),
        "profile_mapped_raw_generation_mwh": float(mapped["raw_profile_mwh"].sum()),
        "profile_mapped_implied_factor": _factor(mapped["actual_generation_mwh"], mapped["raw_profile_mwh"]),
        "unprofiled_lccc_generation_mwh": float(contract_table.loc[~contract_table["profile_mapped"], "actual_generation_mwh"].sum()),
    }
    return contract_table, monthly_table, cohort_table, summary


if "snakemake" in globals():
    contracts, monthly, cohorts, summary = run(
        snakemake.input.actuals_csv,
        list(snakemake.input.generators),
        list(snakemake.input.profiles),
        int(snakemake.params.year),
    )
    contracts.to_csv(snakemake.output.contracts_csv, index=False)
    monthly.to_csv(snakemake.output.monthly_csv, index=False)
    cohorts.to_csv(snakemake.output.cohorts_csv, index=False)
    with open(snakemake.output.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
elif __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actuals", required=True)
    parser.add_argument("--generators", nargs="+", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--contracts-output", required=True)
    parser.add_argument("--monthly-output", required=True)
    parser.add_argument("--cohorts-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    contracts, monthly, cohorts, summary = run(
        args.actuals, args.generators, args.profiles, args.year
    )
    contracts.to_csv(args.contracts_output, index=False)
    monthly.to_csv(args.monthly_output, index=False)
    cohorts.to_csv(args.cohorts_output, index=False)
    with open(args.summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
