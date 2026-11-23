"""Build an auditable LCCC-to-PyPSA CfD contract mapping.

The LCCC public register is the source of truth for historical CfD contracts,
generation and difference payments.  A project is never inferred solely from
technology or a generic strike price: it is either explicitly cross-walked to
one or more model generators, or retained as ``not_represented``.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_OVERRIDE_COLUMNS = {
    "cfd_id", "model_generator", "allocation_share", "mapping_method", "notes"
}


def calendar_year_contract_actuals(
    actual_path: str | Path,
    year: int,
    period_start: str | None = None,
    period_end: str | None = None,
) -> pd.DataFrame:
    """Aggregate LCCC records for a calendar year or an explicit model period."""
    actual = pd.read_csv(actual_path)
    actual["Settlement_Date"] = pd.to_datetime(actual["Settlement_Date"], errors="raise")
    actual = actual.loc[actual["Settlement_Date"].dt.year.eq(year)].copy()
    if period_start:
        actual = actual.loc[actual["Settlement_Date"].ge(pd.Timestamp(period_start))]
    if period_end:
        # A date-only end means the whole final day.
        end = pd.Timestamp(period_end)
        if len(str(period_end).strip()) <= 10:
            end += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        actual = actual.loc[actual["Settlement_Date"].le(end)]
    if actual.empty:
        raise ValueError(f"No LCCC CfD records found for calendar year {year}")
    numeric = ["CFD_Generation_MWh", "CFD_Payments_GBP"]
    actual[numeric] = actual[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    grouped = actual.groupby(
        ["CfD_ID", "Name_of_CfD_Unit", "Technology", "Allocation_round"], dropna=False, as_index=False
    ).agg(
        actual_generation_mwh=("CFD_Generation_MWh", "sum"),
        actual_payment_gbp=("CFD_Payments_GBP", "sum"),
        first_settlement_date=("Settlement_Date", "min"),
        last_settlement_date=("Settlement_Date", "max"),
    )
    return grouped


def _read_overrides(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=sorted(REQUIRED_OVERRIDE_COLUMNS))
    overrides = pd.read_csv(path).fillna({"notes": ""})
    missing = REQUIRED_OVERRIDE_COLUMNS - set(overrides.columns)
    if missing:
        raise ValueError(f"CfD override file is missing columns: {sorted(missing)}")
    overrides["allocation_share"] = pd.to_numeric(overrides["allocation_share"], errors="raise")
    if (overrides["allocation_share"] <= 0).any():
        raise ValueError("Every CfD mapping allocation_share must be positive")
    return overrides


def build_mapping(
    actual_path: str | Path,
    portfolio_path: str | Path,
    generator_path: str | Path,
    overrides_path: str | Path,
    year: int,
    cfd_bmu_path: str | Path | None = None,
    model_bmu_path: str | Path | None = None,
    source_candidates_path: str | Path | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return a full contract register and coverage summary for a model run."""
    actuals = calendar_year_contract_actuals(actual_path, year, period_start, period_end)
    portfolio = pd.read_csv(portfolio_path).rename(columns={"CFD_ID": "CfD_ID"})
    portfolio["Maximum_Contract_Capacity_MW"] = pd.to_numeric(
        portfolio["Maximum_Contract_Capacity_MW"], errors="coerce"
    )
    contracts = actuals.merge(
        portfolio[["CfD_ID", "Status", "Maximum_Contract_Capacity_MW"]], on="CfD_ID", how="left", validate="one_to_one"
    )
    if contracts["Status"].isna().any():
        missing = contracts.loc[contracts["Status"].isna(), "CfD_ID"].tolist()
        raise ValueError(f"LCCC portfolio has no status for: {missing}")

    generators = pd.read_csv(generator_path)
    if "name" not in generators.columns:
        raise ValueError("Model generator file needs a 'name' column")
    generator_columns = ["name", "carrier", "p_nom", "bus", "support_type", "cfd_round"]
    generator_columns = [c for c in generator_columns if c in generators.columns]
    generator_lookup = generators.set_index("name")[generator_columns[1:]]
    lccc_bmus: dict[str, str] = {}
    if cfd_bmu_path:
        cfd_bmu = pd.read_csv(cfd_bmu_path)
        cfd_bmu["Effective_From"] = pd.to_datetime(cfd_bmu["Effective_From"], errors="coerce")
        cfd_bmu["Effective_date_to"] = pd.to_datetime(cfd_bmu["Effective_date_to"], errors="coerce")
        start = pd.Timestamp(period_start or f"{year}-01-01")
        end = pd.Timestamp(period_end or f"{year}-12-31")
        active = cfd_bmu.loc[
            cfd_bmu["Effective_From"].le(end)
            & (cfd_bmu["Effective_date_to"].isna() | cfd_bmu["Effective_date_to"].ge(start))
        ]
        lccc_bmus = active.groupby("CFD_Id")["BMU_Id"].agg(lambda x: "; ".join(sorted(set(x.astype(str))))).to_dict()
    model_bmus: dict[str, str] = {}
    auto_bmu_candidates: dict[str, list[str]] = {}
    if model_bmu_path:
        model_bmu = pd.read_csv(model_bmu_path)
        required_bmu_columns = {"generator_name", "bmu_id"}
        if missing := required_bmu_columns.difference(model_bmu.columns):
            raise ValueError(f"Model BMU mapping is missing columns: {sorted(missing)}")
        model_bmus = model_bmu.groupby("generator_name")["bmu_id"].agg(
            lambda x: "; ".join(sorted(set(x.astype(str))))
        ).to_dict()
        valid_bmus = model_bmu.loc[
            model_bmu["generator_name"].isin(generator_lookup.index),
            ["generator_name", "bmu_id"],
        ].dropna().drop_duplicates()
        for bmu_id, group in valid_bmus.groupby("bmu_id"):
            auto_bmu_candidates[str(bmu_id)] = sorted(group["generator_name"].astype(str).unique())
    source_candidates = pd.DataFrame(columns=["cfd_id"])
    if source_candidates_path:
        source_candidates = pd.read_csv(source_candidates_path)
        if source_candidates["cfd_id"].duplicated().any():
            raise ValueError("Each pending source candidate must have one CfD contract row")
        source_candidates = source_candidates.set_index("cfd_id")
    overrides = _read_overrides(overrides_path)
    overrides = overrides.loc[overrides["cfd_id"].isin(contracts["CfD_ID"])].copy()

    # A rebuilt network may already contain explicit contract tranches.  In
    # that case map the contract to its own component and do not apply the old
    # allocation share a second time.
    for idx, row in overrides.iterrows():
        tranche = f"{row['model_generator']}__CfD__{row['cfd_id']}"
        if tranche in generator_lookup.index:
            overrides.at[idx, "model_generator"] = tranche
            overrides.at[idx, "allocation_share"] = 1.0
            overrides.at[idx, "mapping_method"] = f"{row['mapping_method']}+explicit_cfd_tranche"

    unknown = sorted(set(overrides["cfd_id"]) - set(contracts["CfD_ID"]))
    if unknown:
        raise ValueError(f"Overrides do not correspond to active {year} LCCC contracts: {unknown}")
    absent_generators = sorted(set(overrides["model_generator"]) - set(generator_lookup.index))

    # The public LCCC crosswalk can safely auto-map only an unambiguous active
    # CfD-contract -> BMU -> PyPSA-generator chain. A BMU shared by contracts
    # or mapped to several model assets remains unresolved for review.
    contract_auto_candidates: dict[str, list[str]] = {}
    for cfd_id, bmus in lccc_bmus.items():
        candidates = sorted({
            generator
            for bmu in str(bmus).split("; ")
            for generator in auto_bmu_candidates.get(bmu, [])
        })
        contract_auto_candidates[str(cfd_id)] = candidates
    candidate_use_count: dict[str, int] = {}
    for candidates in contract_auto_candidates.values():
        for generator in candidates:
            candidate_use_count[generator] = candidate_use_count.get(generator, 0) + 1

    rows: list[dict] = []
    for contract in contracts.to_dict("records"):
        contract_overrides = overrides.loc[overrides["cfd_id"].eq(contract["CfD_ID"])]
        if contract_overrides.empty:
            candidate = source_candidates.loc[contract["CfD_ID"]].to_dict() if contract["CfD_ID"] in source_candidates.index else {}
            auto_candidates = contract_auto_candidates.get(str(contract["CfD_ID"]), [])
            unique_candidates = [
                generator for generator in auto_candidates
                if candidate_use_count.get(generator, 0) == 1
            ]
            if len(unique_candidates) == 1:
                model_generator = unique_candidates[0]
                model = generator_lookup.loc[model_generator].to_dict()
                rows.append({
                    **contract,
                    "model_generator": model_generator,
                    "allocation_share": 1.0,
                    "mapped_capacity_mw": float(pd.to_numeric(model.get("p_nom"), errors="coerce")),
                    "mapping_method": "automatic_active_lccc_bmu_to_unique_model_bmu",
                    "mapping_status": "mapped",
                    "mapping_notes": "Automatic one-to-one active LCCC BMU and model BMU match; review before using as a permanent override.",
                    "lccc_bmu_ids": lccc_bmus.get(contract["CfD_ID"], pd.NA),
                    "model_bmu_ids": model_bmus.get(model_generator, pd.NA),
                    **{f"model_{k}": v for k, v in model.items()},
                })
                continue
            status = "ambiguous_bmu_match" if auto_candidates else "not_represented"
            if candidate:
                status = "source_mapped_pending_rebuild"
            rows.append({
                **contract,
                "model_generator": pd.NA,
                "allocation_share": 0.0,
                "mapped_capacity_mw": 0.0,
                "mapping_method": "source_candidate" if candidate else status,
                "mapping_status": status,
                "mapping_notes": candidate.get("notes", "No unique active LCCC BMU to model BMU match; retained explicitly for coverage accounting."),
                "lccc_bmu_ids": lccc_bmus.get(contract["CfD_ID"], pd.NA),
                "model_bmu_ids": pd.NA,
                **{k if k.startswith("source_") else f"source_{k}": v for k, v in candidate.items()},
            })
            continue
        for override in contract_overrides.to_dict("records"):
            if override["model_generator"] not in generator_lookup.index:
                rows.append({
                    **contract,
                    "model_generator": override["model_generator"],
                    "allocation_share": 0.0,
                    "mapped_capacity_mw": 0.0,
                    "mapping_method": override["mapping_method"],
                    "mapping_status": "override_generator_absent",
                    "mapping_notes": (
                        f"Declared override asset is absent from this model revision: "
                        f"{override['model_generator']}. No dispatch was imputed. {override['notes']}"
                    ).strip(),
                    "lccc_bmu_ids": lccc_bmus.get(contract["CfD_ID"], pd.NA),
                    "model_bmu_ids": pd.NA,
                })
                continue
            model = generator_lookup.loc[override["model_generator"]].to_dict()
            p_nom = float(pd.to_numeric(model.get("p_nom"), errors="coerce"))
            rows.append({
                **contract,
                "model_generator": override["model_generator"],
                "allocation_share": float(override["allocation_share"]),
                "mapped_capacity_mw": p_nom * float(override["allocation_share"]),
                "mapping_method": override["mapping_method"],
                "mapping_status": "mapped",
                "mapping_notes": override["notes"],
                "lccc_bmu_ids": lccc_bmus.get(contract["CfD_ID"], pd.NA),
                "model_bmu_ids": model_bmus.get(override["model_generator"], pd.NA),
                **{f"model_{k}": v for k, v in model.items()},
            })
    result = pd.DataFrame(rows)

    mapped = result.loc[result["mapping_status"].eq("mapped")]
    contract_shares = mapped.groupby("CfD_ID")["allocation_share"].sum()
    if (contract_shares > 1.000001).any():
        raise ValueError(f"Mapping exceeds 100% of model dispatch for contracts: {contract_shares[contract_shares > 1.000001].to_dict()}")

    contract_summary = result.groupby("CfD_ID", as_index=False).agg(
        cfd_unit=("Name_of_CfD_Unit", "first"),
        actual_generation_mwh=("actual_generation_mwh", "first"),
        actual_payment_gbp=("actual_payment_gbp", "first"),
        contract_capacity_mw=("Maximum_Contract_Capacity_MW", "first"),
        mapped_capacity_mw=("mapped_capacity_mw", "sum"),
        mapping_status=("mapping_status", "first"),
    )
    contract_summary["capacity_match_ratio"] = (
        contract_summary["mapped_capacity_mw"] / contract_summary["contract_capacity_mw"]
    )
    result = result.merge(contract_summary[["CfD_ID", "capacity_match_ratio"]], on="CfD_ID", how="left")

    mapped_contracts = contract_summary["mapping_status"].eq("mapped")
    source_mapped_contracts = contract_summary["mapping_status"].ne("not_represented")
    total_generation = float(contract_summary["actual_generation_mwh"].sum())
    total_payment = float(contract_summary["actual_payment_gbp"].sum())
    summary = {
        "calendar_year": year,
        "period_start": str(period_start) if period_start else f"{year}-01-01",
        "period_end": str(period_end) if period_end else f"{year}-12-31",
        "source": "LCCC public contract portfolio and actual CfD generation/payments datasets",
        "contracts_with_generation_or_payment": int(len(contract_summary)),
        "mapped_contracts": int(mapped_contracts.sum()),
        "not_represented_contracts": int(contract_summary["mapping_status"].eq("not_represented").sum()),
        "contract_register_complete": bool(len(contract_summary) == contracts["CfD_ID"].nunique()),
        "model_mapping_complete": bool(mapped_contracts.all()),
        "source_mapping_complete": bool(source_mapped_contracts.all()),
        "source_mapped_pending_rebuild_contracts": int(contract_summary["mapping_status"].eq("source_mapped_pending_rebuild").sum()),
        "automatic_bmu_mapped_contracts": int((result["mapping_method"] == "automatic_active_lccc_bmu_to_unique_model_bmu").sum()),
        "ambiguous_bmu_contracts": int(contract_summary["mapping_status"].eq("ambiguous_bmu_match").sum()),
        "override_generator_absent_contracts": int(
            contract_summary["mapping_status"].eq("override_generator_absent").sum()
        ),
        "absent_override_generators": absent_generators,
        "actual_generation_mwh_total": total_generation,
        "actual_payment_gbp_total": total_payment,
        "mapped_actual_generation_mwh": float(contract_summary.loc[mapped_contracts, "actual_generation_mwh"].sum()),
        "mapped_actual_payment_gbp": float(contract_summary.loc[mapped_contracts, "actual_payment_gbp"].sum()),
        "mapped_generation_coverage": float(contract_summary.loc[mapped_contracts, "actual_generation_mwh"].sum() / total_generation) if total_generation else None,
        "mapped_payment_coverage": float(contract_summary.loc[mapped_contracts, "actual_payment_gbp"].sum() / total_payment) if total_payment else None,
        "unrepresented_cfd_ids": contract_summary.loc[
            contract_summary["mapping_status"].eq("not_represented"), "CfD_ID"
        ].tolist(),
        "pending_model_rebuild_cfd_ids": contract_summary.loc[
            contract_summary["mapping_status"].eq("source_mapped_pending_rebuild"), "CfD_ID"
        ].tolist(),
        "capacity_mismatches": contract_summary.loc[
            mapped_contracts & (contract_summary["capacity_match_ratio"].sub(1).abs() > 0.02),
            ["CfD_ID", "contract_capacity_mw", "mapped_capacity_mw", "capacity_match_ratio"],
        ].to_dict("records"),
    }
    return result.sort_values(["actual_payment_gbp", "CfD_ID", "model_generator"], ascending=[False, True, True]), summary


def write_mapping(**kwargs) -> None:
    mapping, summary = build_mapping(
        kwargs["actual_path"],
        kwargs["portfolio_path"],
        kwargs["generator_path"],
        kwargs["overrides_path"],
        kwargs["year"],
        kwargs.get("cfd_bmu_path"),
        kwargs.get("model_bmu_path"),
        kwargs.get("source_candidates_path"),
        kwargs.get("period_start"),
        kwargs.get("period_end"),
    )
    Path(kwargs["output_csv"]).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(kwargs["output_csv"], index=False)
    Path(kwargs["output_summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    worklist_path = kwargs.get("output_worklist")
    if worklist_path:
        worklist = mapping.loc[~mapping["mapping_status"].eq("mapped")].copy()
        worklist.to_csv(worklist_path, index=False)


if "snakemake" in globals():
    write_mapping(
        actual_path=snakemake.input.actuals_csv,
        portfolio_path=snakemake.input.portfolio_csv,
        generator_path=snakemake.input.generators_csv,
        overrides_path=snakemake.input.overrides_csv,
        year=int(snakemake.params.year),
        cfd_bmu_path=snakemake.input.cfd_bmu_csv,
        model_bmu_path=snakemake.input.model_bmu_csv,
        source_candidates_path=snakemake.input.source_candidates_csv,
        period_start=snakemake.params.period_start,
        period_end=snakemake.params.period_end,
        output_csv=snakemake.output.mapping_csv,
        output_summary=snakemake.output.summary_json,
        output_worklist=snakemake.output.unresolved_worklist_csv,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actuals", required=True); parser.add_argument("--portfolio", required=True)
    parser.add_argument("--generators", required=True); parser.add_argument("--overrides")
    parser.add_argument("--year", type=int, required=True); parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True)
    parser.add_argument("--cfd-bmus"); parser.add_argument("--model-bmus")
    parser.add_argument("--source-candidates")
    parser.add_argument("--period-start"); parser.add_argument("--period-end")
    parser.add_argument("--output-worklist")
    args = parser.parse_args()
    write_mapping(
        actual_path=args.actuals,
        portfolio_path=args.portfolio,
        generator_path=args.generators,
        overrides_path=args.overrides,
        year=args.year,
        cfd_bmu_path=args.cfd_bmus,
        model_bmu_path=args.model_bmus,
        source_candidates_path=args.source_candidates,
        period_start=args.period_start,
        period_end=args.period_end,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        output_worklist=args.output_worklist,
    )
