"""Run one configured public-ownership financing bundle without rebuilding PyPSA.

This is the Snakemake-free path over the same completed market outputs.  It
uses the shared helpers in ``reference_runs.py`` so its scenario expansion and
configured decisions are identical to ``rules/public_ownership.smk``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

# Permit ``python scripts/public_ownership/run_financing.py`` from the project
# root as well as module execution used by tests/Snakemake.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.aggregate_reference_runs import write_annual_market_bundle
from scripts.public_ownership.build_asset_finance_register import (
    build_asset_finance_register, load_award_years, load_legacy_plant_rounds,
)
from scripts.public_ownership.build_cfd_strike_adjustments import write_strike_adjustments
from scripts.public_ownership.build_consumer_bill_financing_bridge import (
    build_consumer_bill_financing_bridge,
)
from scripts.public_ownership.calculate_financing_scenarios import (
    calculate_financing_scenarios, register_options_from_config, vintage_options_for,
)
from scripts.public_ownership.extract_workbook_v5 import file_sha256
from scripts.public_ownership.model_year import group_model_year
from scripts.public_ownership.reference_runs import (
    MONTHS, baseline_scenarios, check_balancing_group_scenarios, configured_case_specs, consumer_bill_groups,
    group_excluded_cases, group_forecast_case_id, group_market_cases, reference_run_scenarios,
    validate_public_ownership_config, vintaged_groups, wholesale_source,
)
from config.config_loader import read_config


def _load_snapshot(root: Path, config: dict) -> dict:
    """Load the assumptions snapshot and refuse one taken from a different workbook."""
    snapshot_path = root / config["assumptions_snapshot"]
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"{snapshot_path} is missing; run scripts/public_ownership/extract_workbook_v5.py "
            f"{config['source_workbook']} {config['source_variants']} {config['assumptions_snapshot']}"
        )
    with snapshot_path.open(encoding="utf-8") as handle:
        snapshot = yaml.safe_load(handle)
    workbook = root / config["source_workbook"]
    if workbook.exists() and snapshot.get("workbook_sha256") != file_sha256(workbook):
        raise ValueError(
            f"{snapshot_path} was extracted from a different workbook than {workbook}; "
            "re-run extract_workbook_v5.py before accounting"
        )
    return snapshot


def _forecast_case(raw_config: dict, config: dict, run_group: str) -> dict | None:
    """The group's consumer_bill.forecasts case (its own, else the bill link's), expanded like consumer_bill.smk."""
    cases = (((raw_config.get("consumer_bill") or {}).get("forecasts") or {}).get("cases")) or {}
    case_id = group_forecast_case_id(config, run_group)
    if not case_id or case_id not in cases:
        return None
    case = dict(cases[case_id])
    case["case_id"] = case_id
    case["source_scenarios"] = [case["source_scenario_pattern"].format(month=month) for month in MONTHS]
    return case


def run_bundle(config_path: str | Path, run_group: str, *, with_consumer_bill: bool = False) -> tuple[Path, Path]:
    """Build annual evidence, asset register and finance comparison from completed runs.

    ``with_consumer_bill`` also projects the per-case bill (WP-C) for a group
    that shares the bill-link forecast case's scenarios; it reads all twelve
    wholesale and balancing networks once per case.
    """
    root = Path.cwd()
    raw_config = read_config(config_path) or {}
    config_dir = Path(config_path) if Path(config_path).is_dir() else Path(config_path).parent
    if run_group not in ((raw_config.get("public_ownership") or {}).get("reference_run_groups") or {}):
        # Decision 31 (task D31): a study group is declared on its parent scenario entry,
        # in the scenarios.yaml beside the config; resolve it as the rules do.
        from scripts.public_ownership.study_config import load_studies

        scenarios_file = config_dir / "scenarios.yaml"
        if scenarios_file.exists():
            raw_config = dict(load_studies(config_dir, raw_config).config(run_group))
    forecast_cases = (((raw_config.get("consumer_bill") or {}).get("forecasts") or {}).get("cases")) or {}
    config = validate_public_ownership_config(raw_config.get("public_ownership") or {}, forecast_cases)
    if run_group not in config["reference_run_groups"]:
        raise KeyError(
            f"{run_group!r} is not a reference_run_group; configured: "
            f"{sorted(config['reference_run_groups'])}"
        )
    group = config["reference_run_groups"][run_group]
    if run_group in vintaged_groups(config):
        # Task SETTLE2050 (the reachability audit): the vintaged (2050) chain has stages this
        # bundle never had (fleet vintages, the contract book, the register-row settlement,
        # carried acquisition), so a 2050 group run here would be wrong, not just old.
        raise ValueError(
            f"{run_group} is a vintaged (model year {group_model_year(group)}) group: run_bundle is the 2030 path; "
            "build it through the Snakemake rules (rules/public_ownership.smk)")
    scenarios = reference_run_scenarios(group)
    summary_paths = [root / "resources" / "analysis" / f"{scenario}_market_summary.json" for scenario in scenarios]
    network_paths = [root / "resources" / "market" / f"{scenario}_balancing.nc" for scenario in scenarios]
    baseline_network_paths = [
        root / "resources" / "market" / f"{scenario}_balancing.nc"
        for scenario in baseline_scenarios(config, run_group)
    ]
    missing = [str(path) for path in summary_paths + network_paths + baseline_network_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Completed reference-run outputs are missing:\n" + "\n".join(missing))
    snapshot = _load_snapshot(root, config)

    output_dir = root / "resources" / "public_ownership"
    annual = write_annual_market_bundle(
        summary_paths,
        output_dir / f"{run_group}_monthly_market_metrics.csv",
        output_dir / f"{run_group}_annual_market_summary.json",
        output_dir / f"{run_group}_source_manifest.json",
        expected_scenarios=scenarios,
        run_group=run_group,
    )
    bill_link = config.get("consumer_bill") or {}
    repricing = bill_link.get("cfd_repricing") or {}
    labels = bill_link.get("label_corrections") or {}
    assets = build_asset_finance_register(
        network_paths, pd.read_csv(root / config["crosswalk"]),
        model_year=group_model_year(group), policy_start_year=int(config["policy_start_year"]),
        baseline_network_paths=baseline_network_paths,
        component_types=tuple(config["ownership_scope"]),
        support_type_overrides=labels.get("support_type") or {},
        award_years=load_award_years(root / repricing["award_years"]) if repricing.get("award_years") else None,
        eligible_award_years_from=repricing.get("eligible_award_years_from"),
        legacy_plant_rounds=(load_legacy_plant_rounds(root / repricing["award_years"])
                             if repricing.get("award_years") else frozenset()),
        existing_storage_sites=config.get("existing_storage_sites"),
    )
    assets_path = output_dir / f"{run_group}_asset_inputs.csv"
    assets.to_csv(assets_path, index=False)
    # WP-D: settle every asset against its own market and support revenue when
    # the group shares the bill-link forecast case (the settlement needs the
    # bill's strike table and RO value).
    forecast = _forecast_case(raw_config, config, run_group)
    is_bill_group = forecast is not None and run_group in consumer_bill_groups(config, forecast_cases)
    # Item BM1: a month sharing its parent's Stage 1 (wholesale_from) reads the parent's files.
    scenarios_path = config_dir / "scenarios.yaml"
    if scenarios_path.exists():
        from config.config_loader import load_scenario_entries

        scenario_configs = load_scenario_entries(scenarios_path.parent)
    else:
        scenario_configs = {}
    if is_bill_group:
        check_balancing_group_scenarios(config, run_group, scenario_configs)

    def _stage1(scenario, suffix):
        return root / "resources" / "market" / f"{wholesale_source(scenario, scenario_configs)}{suffix}"
    settlement_path = output_dir / f"{run_group}_asset_settlement.csv"
    if is_bill_group:
        from scripts.public_ownership.build_asset_settlement import write_asset_settlement
        from scripts.public_ownership.settlement import attach_settlement

        analysis = root / "resources" / "analysis"
        case_id = forecast["case_id"]
        if not with_consumer_bill and settlement_path.exists():
            settlement = pd.read_csv(settlement_path)
        else:
            settlement = write_asset_settlement(
                forecast["source_scenarios"],
                [_stage1(s, "_wholesale.nc") for s in forecast["source_scenarios"]],
                network_paths,
                [_stage1(s, "_wholesale_price.csv") for s in forecast["source_scenarios"]],
                [root / "resources" / "market" / f"{s}_redispatch_summary.csv" for s in forecast["source_scenarios"]],
                root / forecast["assumptions"], analysis / f"{case_id}_ro_scheme_summary.json",
                int(forecast["model_price_year"]), settlement_path,
                # Item B1: fuel and carbon from the build-time resource breakdown.
                marginal_cost_breakdowns=[
                    path for path in dict.fromkeys(
                        root / "resources" / "generators"
                        / f"{(scenario_configs.get(s) or {}).get('network_from') or s}_marginal_costs_breakdown.csv"
                        for s in forecast["source_scenarios"])
                    if path.exists()] or None,
            )
        assets = attach_settlement(assets, settlement)
    # Item MR4, as the Snakemake rule books it: CM revenue per kW on merchant and
    # storage rows, carried from the rates' 2025 money onto the register's year.
    cm_rates = root / "resources" / "market" / "capacity_market" / "cm_revenue_by_technology.csv"
    if cm_rates.exists() and forecast is not None:
        from scripts.market.capacity_market import attach_capacity_market_revenue
        from scripts.public_ownership.calculate_financing_scenarios import capacity_market_rates_in_price_year

        rebasing = json.loads((root / forecast["assumptions"]).read_text(encoding="utf-8"))["currency_rebasing"]
        assets = attach_capacity_market_revenue(assets, capacity_market_rates_in_price_year(
            pd.read_csv(cm_rates), int(config["price_year"]),
            (rebasing["from_price_year"], rebasing["to_price_year"], rebasing["factor"])))
    acquisition = config.get("acquisition") or {}
    # Task Y14: the group's own case list when it declares one, else the global list.
    specs = configured_case_specs(config, run_group)
    ro_stations = None
    if any(spec.acquires_legacy() for spec in specs):
        from scripts.public_ownership.legacy_valuation import settings_from_config as valuation_settings

        ro_stations = pd.read_csv(root / valuation_settings(acquisition.get("valuation"))["ro_register"], low_memory=False)
    register, summary = calculate_financing_scenarios(
        assets, pd.read_csv(root / config["technology_costs"]), snapshot,
        demand_mwh=annual["demand_MWh"], price_year=int(config["price_year"]),
        case_specs=specs, acquisition=acquisition, ro_stations=ro_stations,
        state_equity_share_value=config.get("state_stake_share"),
        state_operating_overhead=config.get("state_operating_overhead"),
        vintage_costs=vintage_options_for(config, run_group, group_model_year(group), root),
        **register_options_from_config(config, root),
    )
    register_path = output_dir / f"{run_group}_asset_finance_register.parquet"
    summary_path = output_dir / f"{run_group}_financing_summary.csv"
    from scripts.public_ownership.register_io import write_register

    write_register(register, register_path)
    summary.to_csv(summary_path, index=False)
    technology, bill_bridge, bridge_summary = build_consumer_bill_financing_bridge(register, summary)
    technology.to_csv(output_dir / f"{run_group}_financing_savings_by_technology.csv", index=False)
    bill_bridge.to_csv(output_dir / f"{run_group}_consumer_bill_financing_bridge.csv", index=False)
    (output_dir / f"{run_group}_consumer_bill_financing_bridge_summary.json").write_text(
        json.dumps(bridge_summary, indent=2), encoding="utf-8"
    )

    # WP-B, WP-D and WP-C apply only to a group that shares the forecast case's scenarios.
    if is_bill_group:
        from scripts.public_ownership.build_public_account import write_public_account

        adjustments_path = output_dir / f"{run_group}_cfd_strike_adjustments.csv"
        routes_path = output_dir / f"{run_group}_financing_saving_routes.csv"
        forecast_summary = analysis / f"{case_id}_consumer_bill_future_summary.json"
        write_strike_adjustments(
            register_path, summary_path, root / forecast["assumptions"], bill_link,
            model_price_year=int(forecast["model_price_year"]),
            adjustments_output=adjustments_path, routes_output=routes_path,
            forecast_summary_path=forecast_summary if forecast_summary.exists() else None,
            # The strike cut divides by the energy the bill actually pays.
            settlement_inputs={
                "case": forecast,
                "networks": [_stage1(s, "_wholesale.nc") for s in forecast["source_scenarios"]],
                "prices": [_stage1(s, "_wholesale_price.csv") for s in forecast["source_scenarios"]],
                "summaries": [analysis / f"{s}_market_summary.json" for s in forecast["source_scenarios"]],
                "physical_networks": network_paths,
            },
        )
        account_summary_path = output_dir / f"{run_group}_public_account_summary.json"
        # Review 29 Sep 2026: the account re-prices R1 rows exactly as the adjustments did.
        write_public_account(register_path, config, output_dir / f"{run_group}_public_account.csv", account_summary_path,
                             adjustments_path=adjustments_path, assumptions_path=root / forecast["assumptions"],
                             model_price_year=int(forecast["model_price_year"]))
        if with_consumer_bill:
            from scripts.public_ownership.project_public_ownership_consumer_bill import write_ownership_cases

            network_lines = output_dir / f"{run_group}_network_finance_account.csv"
            write_ownership_cases(
                forecast, tuple(group_market_cases(config, run_group)), root / forecast["assumptions"],
                [_stage1(s, "_wholesale.nc") for s in forecast["source_scenarios"]],
                [_stage1(s, "_wholesale_price.csv") for s in forecast["source_scenarios"]],
                [analysis / f"{s}_market_summary.json" for s in forecast["source_scenarios"]],
                network_paths,
                analysis / f"{forecast['annual_aggregate_study']}_annual_summary.json",
                analysis / f"{case_id}_future_balancing_costs_summary.json",
                root / forecast["recovery_demand_assumptions"],
                None,
                analysis / f"{case_id}_ro_scheme_summary.json",
                adjustments_path, routes_path, summary_path,
                analysis / f"{case_id}_consumer_bill_future_components.csv",
                output_dir / f"{run_group}_consumer_bill_by_ownership_case.csv",
                output_dir / f"{run_group}_consumer_bill_by_ownership_case_summary.json",
                output_dir / f"{run_group}_consumer_bill_by_ownership_case_reconciliation.csv",
                public_account_path=account_summary_path,
                network_lines_path=network_lines if network_lines.exists() else None,
                excluded_cases=group_excluded_cases(config, run_group),
            )
    return register_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config")
    parser.add_argument("--run-group", default="cp30_2030_cp30_network")
    parser.add_argument(
        "--with-consumer-bill", action="store_true",
        help="also project the per-case bill (WP-C); reads every monthly network once per case",
    )
    args = parser.parse_args()
    register, summary = run_bundle(args.config, args.run_group, with_consumer_bill=args.with_consumer_bill)
    print(json.dumps({"asset_finance_register": str(register), "financing_summary": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
