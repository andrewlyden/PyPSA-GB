"""Audit renewable performance factors against named LCCC CfD outturn.

The renewable performance factors are deliberately carrier-wide physical
corrections.  They can make a national wind total plausible while still being
wrong for a particular support cohort or location.  This diagnostic keeps
those two questions separate:

* contract rows compare LCCC generation with the model's *available* energy,
  before any modelled curtailment; and
* cohort rows show where corrected availability and physical dispatch sit by
  carrier and support type across the complete model fleet.

It is a historical diagnostic only.  In particular, the implied contract
factor is not an instruction to apply that value to future scenarios.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


COMPONENT_NAMES = ("era5_bias", "turbine_curve", "wake_losses", "availability")


def resolve_performance_factors(config: dict[str, Any] | None) -> tuple[dict[str, float], str]:
    """Return effective factors using the same precedence as integration."""
    config = config or {}
    if not config.get("enabled", False):
        return {}, "disabled"
    flat = config.get("factors", {}) or {}
    if flat:
        return {str(carrier): float(value) for carrier, value in flat.items()}, "flat_override"

    components = [config.get(name, {}) or {} for name in COMPONENT_NAMES]
    carriers = set().union(*(set(component) for component in components))
    factors: dict[str, float] = {}
    for carrier in carriers:
        factor = 1.0
        for component in components:
            factor *= float(component.get(carrier, 1.0))
        factors[str(carrier)] = factor
    return factors, "stacked_components" if factors else "enabled_without_factors"


def parse_factor_arguments(values: list[str]) -> dict[str, Any]:
    """Build a flat factor config from repeatable ``carrier=value`` CLI values."""
    factors: dict[str, float] = {}
    for value in values:
        carrier, separator, raw_factor = value.partition("=")
        if not separator or not carrier or not raw_factor:
            raise ValueError(f"Expected carrier=value factor, got {value!r}")
        factors[carrier] = float(raw_factor)
    return {"enabled": bool(factors), "factors": factors}


def build_contract_factor_diagnostics(
    reconciliation: pd.DataFrame,
    mapping: pd.DataFrame,
    factors: dict[str, float],
    generator_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate an implied historical availability factor for each contract."""
    mapped = mapping.loc[mapping["mapping_status"].eq("mapped")].copy()
    carrier_by_contract = mapped.groupby("CfD_ID")["model_carrier"].first()
    result = reconciliation.copy()
    result["model_carrier"] = result["CfD_ID"].map(carrier_by_contract)
    result["configured_performance_factor"] = result["model_carrier"].map(factors).fillna(1.0)
    if generator_metadata is not None and not generator_metadata.empty:
        metadata = generator_metadata.copy()
        metadata.index.name = "model_generator"
        metadata = metadata.reset_index()[[
            "model_generator", "resource_region", "renewable_performance_factor",
            "renewable_performance_factor_source", "modelled_corrected_available_mwh",
        ]]
        allocated = mapped[["CfD_ID", "model_generator", "allocation_share", "model_carrier"]].merge(
            metadata, on="model_generator", how="left", validate="many_to_one"
        )
        # Assets outside the factor-managed carriers (for example biomass)
        # are absent from this metadata table and must retain the carrier
        # fallback below rather than being converted into a zero-energy row.
        allocated = allocated.loc[
            allocated["modelled_corrected_available_mwh"].notna()
        ].copy()
        if not allocated.empty:
            allocated["renewable_performance_factor"] = allocated[
                "renewable_performance_factor"
            ].fillna(allocated["model_carrier"].map(factors)).fillna(1.0)
            allocated["modelled_corrected_available_mwh"] = pd.to_numeric(
                allocated["modelled_corrected_available_mwh"], errors="coerce"
            ).fillna(0.0) * allocated["allocation_share"]
            allocated["inferred_pre_factor_available_mwh"] = allocated[
                "modelled_corrected_available_mwh"
            ].div(allocated["renewable_performance_factor"])
            metadata_by_contract = allocated.groupby("CfD_ID", as_index=False).agg(
                _metadata_corrected_available_mwh=("modelled_corrected_available_mwh", "sum"),
                _metadata_pre_factor_available_mwh=("inferred_pre_factor_available_mwh", "sum"),
                resource_regions=("resource_region", lambda values: "; ".join(sorted(set(values.dropna().astype(str))))),
                performance_factor_sources=("renewable_performance_factor_source", lambda values: "; ".join(sorted(set(values.dropna().astype(str))))),
            )
            result = result.merge(metadata_by_contract, on="CfD_ID", how="left")
            metadata_factor = result["_metadata_corrected_available_mwh"].div(
                result["_metadata_pre_factor_available_mwh"].where(
                    result["_metadata_pre_factor_available_mwh"] > 0
                )
            )
            result["configured_performance_factor"] = metadata_factor.fillna(
                result["configured_performance_factor"]
            )
    else:
        result["resource_regions"] = ""
        result["performance_factor_sources"] = "carrier_config_fallback"
    if "resource_regions" not in result:
        result["resource_regions"] = ""
    if "performance_factor_sources" not in result:
        result["performance_factor_sources"] = "carrier_config_fallback"
    result["resource_regions"] = result["resource_regions"].fillna("")
    result["performance_factor_sources"] = result[
        "performance_factor_sources"
    ].fillna("carrier_config_fallback")
    available = pd.to_numeric(
        result["modelled_contract_available_mwh_full_period"], errors="coerce"
    ).fillna(0.0)
    actual = pd.to_numeric(result["actual_generation_mwh"], errors="coerce").fillna(0.0)
    factor = result["configured_performance_factor"]
    result["inferred_pre_factor_available_mwh"] = available.div(factor.where(factor > 0))
    if "_metadata_pre_factor_available_mwh" in result:
        result["inferred_pre_factor_available_mwh"] = result[
            "_metadata_pre_factor_available_mwh"
        ].where(result["_metadata_pre_factor_available_mwh"].notna(), result["inferred_pre_factor_available_mwh"])
    result["implied_factor_to_match_actual_available_energy"] = actual.div(
        result["inferred_pre_factor_available_mwh"].where(
            result["inferred_pre_factor_available_mwh"] > 0
        )
    )
    result["availability_scaling_needed"] = actual.div(available.where(available > 0))
    result["factor_difference_from_configured"] = (
        result["implied_factor_to_match_actual_available_energy"] - factor
    )
    result["factor_ratio_to_configured"] = result[
        "implied_factor_to_match_actual_available_energy"
    ].div(factor.where(factor > 0))
    result["interpretation"] = "within_10_percent_of_configured_factor"
    result.loc[result["factor_ratio_to_configured"] > 1.10, "interpretation"] = (
        "modelled_available_energy_low_for_contract"
    )
    result.loc[result["factor_ratio_to_configured"] < 0.90, "interpretation"] = (
        "modelled_available_energy_high_for_contract"
    )
    result.loc[actual.eq(0), "interpretation"] = "no_actual_generation_to_calibrate"
    return result.sort_values("actual_generation_mwh", ascending=False)


def _network_energy_by_generator(network, names: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return dispatch, corrected availability and maximum profile per generator."""
    names = [name for name in names if name in network.generators.index]
    if not names:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    weights = network.snapshot_weightings.get(
        "generators", pd.Series(1.0, index=network.snapshots)
    )
    p_max_pu = network.generators_t.p_max_pu.reindex(index=network.snapshots, columns=names)
    static = pd.to_numeric(network.generators.loc[names, "p_max_pu"], errors="coerce").fillna(1.0)
    p_max_pu = p_max_pu.fillna(static).fillna(1.0)
    nominal = pd.to_numeric(network.generators.loc[names, "p_nom"], errors="coerce").fillna(0.0)
    available = p_max_pu.mul(nominal, axis=1).multiply(weights, axis=0).sum(axis=0)
    dispatch = network.generators_t.p.reindex(index=network.snapshots, columns=names).fillna(0.0)
    dispatch = dispatch.multiply(weights, axis=0).sum(axis=0)
    return dispatch, available, p_max_pu.max(axis=0)


def build_cohort_diagnostics(network, factors: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise corrected and inferred raw energy by support cohort."""
    generator_table = network.generators.copy()
    generator_table["carrier"] = generator_table["carrier"].astype(str)
    generator_table = generator_table.loc[generator_table["carrier"].isin(factors)].copy()
    generator_table["support_type"] = generator_table.get(
        "support_type", pd.Series("unlabelled", index=generator_table.index)
    ).fillna("unlabelled").astype(str)
    generator_table["resource_region"] = generator_table.get(
        "resource_region", pd.Series("GB", index=generator_table.index)
    ).fillna("GB").astype(str)
    generator_table["renewable_performance_factor"] = pd.to_numeric(
        generator_table.get(
            "renewable_performance_factor", generator_table["carrier"].map(factors)
        ), errors="coerce"
    ).fillna(generator_table["carrier"].map(factors)).fillna(1.0)
    generator_table["renewable_performance_factor_source"] = generator_table.get(
        "renewable_performance_factor_source",
        pd.Series("carrier_config_fallback", index=generator_table.index),
    ).fillna("carrier_config_fallback").astype(str)
    names = generator_table.index.tolist()
    dispatch, available, maximum = _network_energy_by_generator(network, names)
    generator_table["modelled_physical_dispatch_mwh"] = generator_table.index.map(dispatch).fillna(0.0)
    generator_table["modelled_corrected_available_mwh"] = generator_table.index.map(available).fillna(0.0)
    generator_table["max_corrected_p_max_pu"] = generator_table.index.map(maximum).fillna(0.0)
    generator_table["configured_performance_factor"] = generator_table[
        "renewable_performance_factor"
    ]
    generator_table["inferred_pre_factor_available_mwh"] = generator_table[
        "modelled_corrected_available_mwh"
    ].div(generator_table["configured_performance_factor"])
    generator_table["modelled_curtailment_mwh"] = (
        generator_table["modelled_corrected_available_mwh"]
        - generator_table["modelled_physical_dispatch_mwh"]
    ).clip(lower=0.0)
    # Profiles originate in [0, 1], so a corrected maximum above the factor
    # would signal either an additional transformation or a data issue.  This
    # is a consistency check, not proof of the complete integration history.
    generator_table["profile_upper_bound_consistent_with_single_factor"] = (
        generator_table["max_corrected_p_max_pu"]
        <= generator_table["configured_performance_factor"] + 1e-6
    )
    generator_table["profile_upper_bound_inconsistent"] = (
        ~generator_table["profile_upper_bound_consistent_with_single_factor"]
    )
    cohort = generator_table.groupby(
        ["carrier", "support_type", "configured_performance_factor"], dropna=False, as_index=False
    ).agg(
        generator_count=("carrier", "size"),
        capacity_mw=("p_nom", "sum"),
        inferred_pre_factor_available_mwh=("inferred_pre_factor_available_mwh", "sum"),
        modelled_corrected_available_mwh=("modelled_corrected_available_mwh", "sum"),
        modelled_physical_dispatch_mwh=("modelled_physical_dispatch_mwh", "sum"),
        modelled_curtailment_mwh=("modelled_curtailment_mwh", "sum"),
        profile_upper_bound_consistent=("profile_upper_bound_consistent_with_single_factor", "all"),
        profile_upper_bound_inconsistent_generator_count=("profile_upper_bound_inconsistent", "sum"),
    )
    cohort["curtailment_share_of_corrected_available"] = cohort["modelled_curtailment_mwh"].div(
        cohort["modelled_corrected_available_mwh"].where(
            cohort["modelled_corrected_available_mwh"] > 0
        )
    )
    return generator_table, cohort.sort_values(["carrier", "support_type"])


def audit(
    network_path: str | Path,
    mapping_path: str | Path,
    reconciliation_path: str | Path,
    performance_config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build contract and support-cohort factor diagnostics for one solved period."""
    import pypsa

    factors, factor_source = resolve_performance_factors(performance_config)
    network = pypsa.Network(network_path)
    assets, cohorts = build_cohort_diagnostics(network, factors)
    mapping = pd.read_csv(mapping_path)
    reconciliation = pd.read_csv(reconciliation_path)
    contracts = build_contract_factor_diagnostics(reconciliation, mapping, factors, assets)

    cfd_contracts = contracts.loc[contracts["actual_generation_mwh"] > 0].copy()
    contract_summary = cfd_contracts.groupby("model_carrier", dropna=False).agg(
        contract_count=("CfD_ID", "size"),
        actual_generation_mwh=("actual_generation_mwh", "sum"),
        modelled_available_mwh=("modelled_contract_available_mwh_full_period", "sum"),
        inferred_pre_factor_available_mwh=("inferred_pre_factor_available_mwh", "sum"),
    )
    contract_summary["available_energy_coverage"] = contract_summary["modelled_available_mwh"].div(
        contract_summary["actual_generation_mwh"]
    )
    contract_summary["portfolio_implied_factor"] = contract_summary["actual_generation_mwh"].div(
        contract_summary["inferred_pre_factor_available_mwh"].where(
            contract_summary["inferred_pre_factor_available_mwh"] > 0
        )
    )
    summary = {
        "treatment": "Historical factor diagnostic. Implied contract factors are evidence for profile investigation, not future assumptions.",
        "configured_factor_source": factor_source,
        "configured_performance_factors": factors,
        "network": str(network_path),
        "mapped_contracts": int(len(contracts)),
        "contracts_with_actual_generation": int(len(cfd_contracts)),
        "contract_portfolio_by_carrier": contract_summary.reset_index().to_dict("records"),
        "cohort_count": int(len(cohorts)),
        "cohort_profile_upper_bound_consistent": bool(
            cohorts["profile_upper_bound_consistent"].all()
        ) if not cohorts.empty else True,
        "profile_upper_bound_inconsistent_asset_count": int(
            assets["profile_upper_bound_inconsistent"].sum()
        ),
        "profile_upper_bound_inconsistent_assets": assets.loc[
            assets["profile_upper_bound_inconsistent"],
            ["carrier", "support_type", "max_corrected_p_max_pu", "configured_performance_factor"],
        ].sort_values("max_corrected_p_max_pu", ascending=False).head(25).reset_index().rename(
            columns={"index": "generator"}
        ).to_dict("records"),
        "interpretation": (
            "A material difference between a portfolio implied factor and its configured "
            "carrier factor means a carrier-wide correction cannot simultaneously match "
            "that CfD portfolio and the rest of the model fleet."
        ),
    }
    return contracts, assets, cohorts, summary


def write_audit(
    network_path: str | Path,
    mapping_path: str | Path,
    reconciliation_path: str | Path,
    performance_config: dict[str, Any] | None,
    contracts_output: str | Path,
    assets_output: str | Path,
    cohorts_output: str | Path,
    summary_output: str | Path,
) -> None:
    contracts, assets, cohorts, summary = audit(
        network_path, mapping_path, reconciliation_path, performance_config
    )
    Path(contracts_output).parent.mkdir(parents=True, exist_ok=True)
    contracts.to_csv(contracts_output, index=False)
    assets.to_csv(assets_output, index=True, index_label="generator")
    cohorts.to_csv(cohorts_output, index=False)
    Path(summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_audit(
        snakemake.input.network_nc,
        snakemake.input.mapping_csv,
        snakemake.input.reconciliation_csv,
        snakemake.params.performance_config,
        snakemake.output.contracts_csv,
        snakemake.output.assets_csv,
        snakemake.output.cohorts_csv,
        snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--reconciliation", required=True)
    parser.add_argument("--performance-config-json")
    parser.add_argument(
        "--factor", action="append", default=[], metavar="CARRIER=VALUE",
        help="Repeatable flat-factor alternative to --performance-config-json.",
    )
    parser.add_argument("--contracts-output", required=True)
    parser.add_argument("--assets-output", required=True)
    parser.add_argument("--cohorts-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    if args.performance_config_json:
        performance_config = json.loads(args.performance_config_json)
    else:
        performance_config = parse_factor_arguments(args.factor)
    write_audit(
        args.network,
        args.mapping,
        args.reconciliation,
        performance_config,
        args.contracts_output,
        args.assets_output,
        args.cohorts_output,
        args.summary_output,
    )
