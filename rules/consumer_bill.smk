"""Consumer-bill accounting, Ofgem component extraction and forecast rules.

This module owns tariff, policy, network, retail and recovery accounting.  It
may consume wholesale, demand and redispatch artifacts from ``market.smk``,
but it does not own or require the market workflow for an official bill.
"""

import json
import re

_consumer_scenarios = scenarios
_consumer_run_ids = run_ids
_consumer_loaded_config = _full_config if "_full_config" in globals() else _loaded
_consumer_defaults = _consumer_loaded_config["defaults"]
_consumer_months = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

# Decision 31 (task D31): a later study's finance and bill settings (forecast cases,
# run groups) on its parent scenario entry in config/scenarios.yaml, each resolved as
# defaults with the entry's blocks deep-merged over it and validated here, at parse
# time. rules/public_ownership.smk reads the same object.
from scripts.public_ownership.study_config import load_studies as _consumer_load_studies


if "turn_down_month_inputs" not in globals():
    # Snakefile_consumer_bill does not load rules/turn_down_allocation.smk: every month reads as solved.
    def turn_down_month_inputs(scenario_ids):
        return []

_CONSUMER_STUDIES = _consumer_load_studies("config", _consumer_defaults)


def _consumer_bill_config(scenario_id):
    """The scenario's price-cap calculator settings (``consumer_bill.price_cap``)."""
    scenario = _consumer_scenarios.get(scenario_id, {})
    return scenario.get("consumer_bill", {}).get("price_cap", {})


def _consumer_scenario_params(wildcards):
    """Return the resolved scenario mapping expected by bill scripts.

    Not merged with ``config``: that holds the run_scenarios list, so every edit
    to it would be a params change, and it differs between this workflow and
    Snakefile_consumer_bill (which has no configfile).
    """
    return {
        **_consumer_scenarios[wildcards.scenario],
        "scenario_id": wildcards.scenario,
    }


if "_wholesale_source" not in globals():
    # Snakefile_consumer_bill does not include market.smk, which owns this.
    def _wholesale_source(scenario_id):
        """Scenario whose Stage 1 files a wholesale_from scenario reads (item BM1)."""
        return _consumer_scenarios.get(scenario_id, {}).get("wholesale_from") or scenario_id


if "_build_source" not in globals():
    # Snakefile_consumer_bill does not load the main Snakefile, which owns this.
    def _build_source(scenario_id):
        """Scenario whose build chain made this scenario's network (network_from)."""
        return _consumer_scenarios.get(scenario_id, {}).get("network_from") or scenario_id


def _consumer_support_bridge_input(wildcards):
    """Return the annual support bridge only when its same-year backtest is enabled."""
    bill = _consumer_bill_config(wildcards.scenario)
    if not bill.get("include_same_year_support_backtest", False):
        return []
    path = bill.get("support_to_bill_bridge")
    if not path:
        raise ValueError(f"{wildcards.scenario}: same-year support backtest needs support_to_bill_bridge")
    return [path]


def _consumer_network_bridge_input(wildcards):
    """Require the network bridge only for its explicit same-year backtest."""
    bill = _consumer_bill_config(wildcards.scenario)
    if not bill.get("include_same_year_network_backtest", False):
        return []
    return [f"{resources_path}/analysis/{wildcards.scenario}_network_cost_bridge.csv"]


def _consumer_procurement_bridge_input(wildcards):
    """Select observed reconstruction or the explicit PyPSA spot backtest."""
    procurement = _consumer_bill_config(wildcards.scenario).get("forward_procurement", {})
    if procurement.get("method") == "pypsa_spot_to_procurement_raw_backtest":
        return f"{resources_path}/analysis/{wildcards.scenario}_pypsa_spot_procurement_bridge.csv"
    return f"{resources_path}/analysis/{wildcards.scenario}_ofgem_forward_procurement_bridge.csv"


def _consumer_loss_proxy_input(wildcards):
    """Attach an explicitly configured annual physical-flow loss proxy."""
    audit = _consumer_bill_config(wildcards.scenario).get("loss_proxy_audit")
    if not audit:
        return []
    return [f"{resources_path}/analysis/{audit}_physical_network_loss_diagnostic.json"]


def _consumer_regional_weights_input(wildcards):
    """Use domestic account geography only when a scenario requests it."""
    path = _consumer_bill_config(wildcards.scenario).get("regional_weights_csv")
    return [path] if path else []


def _consumer_cost_stack_bundle_validation_input(wildcards):
    """Only cost-stack cases require the complete annual evidence bundle."""
    if not _consumer_bill_config(wildcards.scenario).get("cost_stack", {}).get("enabled", False):
        return []
    return [f"{resources_path}/analysis/{wildcards.scenario}_consumer_bill_input_bundle.json"]


def _consumer_support_outturn_inputs(wildcards):
    """Optional fixed RO issue evidence for a historic support screen."""
    path = _consumer_bill_config(wildcards.scenario).get(
        "historical_support_outturn", {}
    ).get("roc_reference_csv")
    return {"roc_reference_csv": path} if path else {}


def _consumer_support_outturn_lccc_actuals(wildcards):
    return _consumer_bill_config(wildcards.scenario).get(
        "historical_support_outturn", {}
    ).get("lccc_actuals_csv", "data/market/lccc/actual_cfd_generation_and_payments.csv")


def _consumer_detailed_support_config(scenario_id):
    return _consumer_bill_config(scenario_id).get("historical_support_outturn", {})


def _consumer_detailed_cfd_enabled(scenario_id):
    cfg = _consumer_detailed_support_config(scenario_id)
    # Preserve the established 2020 audit unless it is explicitly disabled.
    default = int(_consumer_scenarios[scenario_id].get("modelled_year", 0)) == 2020
    return bool(cfg.get("detailed_cfd_contract_reconciliation", default))


def _consumer_detailed_roc_enabled(scenario_id):
    cfg = _consumer_detailed_support_config(scenario_id)
    default = int(_consumer_scenarios[scenario_id].get("modelled_year", 0)) == 2020
    return bool(cfg.get("detailed_roc_station_reconciliation", default))


def _consumer_detailed_support_file(wildcards, key, legacy_path):
    return _consumer_detailed_support_config(wildcards.scenario).get(key, legacy_path)


def _consumer_cfd_mapping_overrides(wildcards):
    # This is a contract-to-physical-asset crosswalk, not a 2020-specific
    # calibration.  It applies to every historical year in which the contract
    # remains live, with each run still validating that the named asset exists.
    default = "data/market/lccc/cfd_pypsa_mapping_overrides.csv"
    return _consumer_detailed_support_file(wildcards, "cfd_mapping_overrides_csv", default)


def _consumer_cfd_source_candidates(wildcards):
    default = "data/market/lccc/cfd_pypsa_source_candidates.csv"
    return _consumer_detailed_support_file(wildcards, "cfd_source_candidates_csv", default)


def _consumer_bundle_provenance_input(wildcards):
    """Attach automated-source provenance when this bundle was manifest-built."""
    bill = _consumer_bill_config(wildcards.scenario)
    if not bill.get("source_manifest"):
        return []
    return [_consumer_snapshot_file(wildcards.scenario, "source_provenance.json")]


def _consumer_market_enabled(scenario_id):
    return bool(_consumer_scenarios.get(scenario_id, {}).get("market", {}).get("enabled", False))


def _consumer_regex(items):
    return r"(?:%s)" % "|".join(map(re.escape, items)) if items else r"(?!x)x"


def _consumer_bundle_dir(scenario_id):
    """Return a scenario's self-contained annual consumer-bill input bundle.

    ``data_bundle`` is preferred because it makes the provenance of a
    historical backtest or future projection explicit.  ``cap_snapshot`` is
    retained as a backwards-compatible shorthand for existing bundles.
    """
    bill = _consumer_bill_config(scenario_id)
    bundle = bill.get("data_bundle")
    if bundle:
        return str(bundle).rstrip("/\\")
    snapshot = bill.get("cap_snapshot", "2026_Q3")
    return f"data/market/ofgem_price_cap/{snapshot}"


def _consumer_snapshot_file(scenario_id, filename):
    return f"{_consumer_bundle_dir(scenario_id)}/{filename}"


def _consumer_wholesale_inputs(wildcards):
    bill = _consumer_bill_config(wildcards.scenario)
    if bill.get("mode", "official") == "counterfactual":
        return {
            "wholesale_network": f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale.nc",
            "wholesale_price_csv": f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_price.csv",
        }
    return {}


def _consumer_stack_model_inputs(wildcards):
    if _consumer_market_enabled(wildcards.scenario):
        return {
            "wholesale_network": f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale.nc",
            "wholesale_price_csv": f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_price.csv",
        }
    return {}


def _consumer_signal_inputs(wildcards):
    """Use audited final physical dispatch for the historic 2020 backtest."""
    inputs = {
        "network_nc": f"{resources_path}/market/{wildcards.scenario}_balancing.nc",
    }
    if wildcards.scenario in _consumer_roc_audit_ids:
        inputs["network_nc"] = f"{resources_path}/market/{wildcards.scenario}_balancing_roc_labelled.nc"
        inputs["roc_reconciliation_json"] = (
            f"{resources_path}/analysis/{wildcards.scenario}_roc_model_dispatch_reconciliation_summary.json"
        )
    if wildcards.scenario in _consumer_lccc_audit_ids:
        inputs["cfd_reconciliation_json"] = (
            f"{resources_path}/analysis/{wildcards.scenario}_lccc_cfd_model_dispatch_reconciliation_summary.json"
        )
    return inputs


_consumer_bill_ids = [
    rid for rid in _consumer_run_ids
    if _consumer_bill_config(rid).get("enabled", False)
]
_consumer_cost_stack_ids = [
    rid for rid in _consumer_bill_ids
    if _consumer_bill_config(rid).get("cost_stack", {}).get("enabled", False)
]
_consumer_audit_ids = [
    rid for rid in _consumer_cost_stack_ids
    if _consumer_market_enabled(rid)
    and not _consumer_scenarios[rid].get("market", {}).get("wholesale_only", False)
]
_consumer_backtest_ids = [
    rid for rid in _consumer_audit_ids
    if _consumer_bill_config(rid).get("backtest", {}).get(
        "enabled", int(_consumer_scenarios[rid].get("modelled_year", 9999)) <= 2024
    )
]
_consumer_historical_support_outturn_ids = [
    rid for rid in _consumer_audit_ids
    if int(_consumer_scenarios[rid].get("modelled_year", 9999)) <= 2024
    and _consumer_bill_config(rid).get("historical_support_outturn", {}).get("enabled", False)
]
_consumer_historical_replay_ids = [
    rid for rid in _consumer_cost_stack_ids
    if _consumer_bill_config(rid).get("input_mode") == "historical_backtest"
    and _consumer_bill_config(rid).get("source_manifest")
]
consumer_bill_historical_report_targets = expand(
    f"{resources_path}/consumer_bill/{{scenario}}/report.html",
    scenario=_consumer_historical_replay_ids,
)
if _consumer_historical_replay_ids:
    consumer_bill_historical_report_targets.append(
        f"{resources_path}/consumer_bill/index.html"
    )

# A historical bundle can be produced directly from a pinned source manifest.
# Keep this mapping at parse time so the downstream rules continue to consume
# ordinary files, while Snakemake can discover their producer automatically.
_consumer_automated_bundles = {}
for _consumer_id in _consumer_bill_ids:
    _consumer_bill = _consumer_bill_config(_consumer_id)
    _consumer_manifest = _consumer_bill.get("source_manifest")
    if not _consumer_manifest:
        continue
    _consumer_dir = _consumer_bundle_dir(_consumer_id).replace("\\", "/")
    _consumer_prefix = "data/market/ofgem_price_cap/"
    if not _consumer_dir.startswith(_consumer_prefix) or "/" in _consumer_dir[len(_consumer_prefix):]:
        raise ValueError(
            f"{_consumer_id}: an automated consumer_bill.price_cap source_manifest requires "
            "data_bundle under data/market/ofgem_price_cap/<bundle_id>"
        )
    _consumer_bundle_id = _consumer_dir[len(_consumer_prefix):]
    if _consumer_bundle_id in _consumer_automated_bundles and _consumer_automated_bundles[_consumer_bundle_id] != _consumer_manifest:
        raise ValueError(f"Conflicting source manifests declared for automated bundle {_consumer_bundle_id}")
    _consumer_automated_bundles[_consumer_bundle_id] = _consumer_manifest
CONSUMER_BILL_AUTOMATED_BUNDLE_REGEX = _consumer_regex(sorted(_consumer_automated_bundles))
_consumer_network_layer_ids = [
    rid for rid in _consumer_audit_ids
    if _consumer_bill_config(rid).get("network_layers", {}).get("enabled", False)
]
_support_monthly_ids = [
    rid for rid in _consumer_run_ids
    if _consumer_scenarios[rid].get("support", {}).get("audit", {}).get("enabled", False)
]
# Keep the complete declared groups separately from the scenarios selected for
# a normal model run.  This permits an explicit historical audit to reuse
# already-solved monthly networks without silently broadening ``snakemake``
# into a twelve-month optimisation.
_all_support_monthly_ids = [
    rid for rid, scenario in _consumer_scenarios.items()
    if scenario.get("support", {}).get("audit", {}).get("enabled", False)
]
_consumer_lccc_audit_ids = sorted(set(
    [
        rid for rid in _consumer_audit_ids
        if _consumer_detailed_cfd_enabled(rid)
    ]
    + [
        rid for rid in _support_monthly_ids
        if _consumer_detailed_cfd_enabled(rid)
    ]
))
_consumer_roc_audit_ids = sorted(set(
    [rid for rid in _consumer_audit_ids if _consumer_detailed_roc_enabled(rid)]
    + [rid for rid in _support_monthly_ids if _consumer_detailed_roc_enabled(rid)]
))

_all_support_group_ids = {}
for _support_id, _scenario in _consumer_scenarios.items():
    if not _scenario.get("support", {}).get("audit", {}).get("enabled", False):
        continue
    _support_cfg = _consumer_scenarios[_support_id]["support"]["audit"]
    _support_group = _support_cfg.get("annual_group")
    if not _support_group:
        raise ValueError(f"{_support_id}: support.audit needs annual_group")
    _all_support_group_ids.setdefault(_support_group, []).append(_support_id)
for _support_group, _support_ids in _all_support_group_ids.items():
    _support_ids.sort(
        key=lambda rid: _consumer_scenarios[rid].get("solve_period", {}).get("start", "")
    )

# A partial selection (for example only January) must never be reported as an
# annual comparison. Annual targets become active only when every group member
# is in the current run selection.
_active_support_ids = set(_support_monthly_ids)
_support_group_ids = {
    group: ids for group, ids in _all_support_group_ids.items()
    if set(ids).issubset(_active_support_ids)
}
for _support_group, _support_ids in _support_group_ids.items():
    _support_ids.sort(
        key=lambda rid: _consumer_scenarios[rid].get("solve_period", {}).get("start", "")
    )


def _support_sources(audit):
    return _support_group_ids[audit]


def _all_support_sources(audit):
    return _all_support_group_ids[audit]


def _support_year(audit):
    first = _support_sources(audit)[0]
    return int(
        _consumer_scenarios[first].get("support", {}).get("audit", {}).get(
            "calendar_year", _consumer_scenarios[first].get("modelled_year")
        )
    )


def _all_support_year(audit):
    first = _all_support_sources(audit)[0]
    return int(
        _consumer_scenarios[first].get("support", {}).get("audit", {}).get(
            "calendar_year", _consumer_scenarios[first].get("modelled_year")
        )
    )


def _support_first_source(audit):
    return _support_sources(audit)[0]


def _all_support_first_source(audit):
    return _all_support_sources(audit)[0]


def _support_bill_config(audit):
    return _consumer_scenarios[_all_support_first_source(audit)].get("support", {}).get(
        "audit", {}
    ).get("consumer_bill_bridge", {})
CONSUMER_BILL_SCENARIO_REGEX = _consumer_regex(_consumer_bill_ids)
CONSUMER_BILL_COST_STACK_REGEX = _consumer_regex(_consumer_cost_stack_ids)
CONSUMER_BILL_METHODOLOGY_AUDIT_REGEX = _consumer_regex(_consumer_audit_ids)
CONSUMER_BILL_BACKTEST_REGEX = _consumer_regex(_consumer_backtest_ids)
CONSUMER_HISTORICAL_SUPPORT_OUTTURN_REGEX = _consumer_regex(_consumer_historical_support_outturn_ids)
CONSUMER_BILL_NETWORK_LAYER_REGEX = _consumer_regex(_consumer_network_layer_ids)
CONSUMER_BILL_LCCC_AUDIT_REGEX = _consumer_regex(_consumer_lccc_audit_ids)
# Rules remain available for every declared support-audit month.  They execute
# only when explicitly requested or listed in ``market_targets``.
CONSUMER_BILL_ROC_AUDIT_REGEX = _consumer_regex(sorted(set(_consumer_roc_audit_ids + _all_support_monthly_ids)))
# The REPD-ROC station mapping is also a build input (add_thermal_generators_to_network,
# ``_historical_roc_crosswalk_input``) for any scenario whose historical ROC station labels
# are enabled, e.g. the 2020 validation months, which are not support-audit scenarios.
_historical_roc_label_ids = [
    rid for rid, scenario in _consumer_scenarios.items()
    if ((scenario.get("support") or {}).get("labels") or {}).get(
        "historical_roc_station_labels", {}
    ).get("enabled", False)
]
REPD_ROC_STATION_MAPPING_REGEX = _consumer_regex(
    sorted(set(_consumer_roc_audit_ids + _all_support_monthly_ids + _historical_roc_label_ids))
)
CONSUMER_SUPPORT_AUDIT_REGEX = _consumer_regex(_support_group_ids)
CONSUMER_ALL_SUPPORT_AUDIT_REGEX = _consumer_regex(_all_support_group_ids)
_support_bill_audits = [
    audit for audit in _support_group_ids
    if _support_bill_config(audit).get("enabled", False)
]
CONSUMER_SUPPORT_BILL_AUDIT_REGEX = _consumer_regex(_support_bill_audits)

consumer_support_audit_targets = []
for _support_group in _support_group_ids:
    consumer_support_audit_targets.extend([
        f"{resources_path}/analysis/{_support_group}_lccc_cfd_model_dispatch_reconciliation.csv",
        f"{resources_path}/analysis/{_support_group}_lccc_cfd_model_dispatch_reconciliation_summary.json",
        f"{resources_path}/analysis/{_support_group}_roc_model_dispatch_reconciliation.csv",
        f"{resources_path}/analysis/{_support_group}_roc_model_dispatch_reconciliation_summary.json",
        f"{resources_path}/analysis/{_support_group}_roc_official_station_ledger.csv",
        f"{resources_path}/analysis/{_support_group}_roc_official_residual_cohorts.csv",
        f"{resources_path}/analysis/{_support_group}_roc_official_residual_cohorts_summary.json",
    ])
    if _support_group in _support_bill_audits:
        consumer_support_audit_targets.extend([
            f"{resources_path}/analysis/{_support_group}_support_to_bill_bridge.csv",
            f"{resources_path}/analysis/{_support_group}_support_to_bill_bridge_summary.json",
            f"{resources_path}/analysis/{_support_group}_support_to_bill_bridge.html",
        ])


rule calculate_consumer_bill:
    """Report an official cap snapshot or an explicit wholesale sensitivity."""
    input:
        unpack(_consumer_wholesale_inputs),
        snapshot_rates_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "electricity_single_rate.csv"),
        snapshot_metadata_json=lambda wc: _consumer_snapshot_file(wc.scenario, "metadata.json"),
        regional_weights_csv=_consumer_regional_weights_input,
        bundle_validation=_consumer_cost_stack_bundle_validation_input,
    output:
        tariffs_csv=f"{resources_path}/analysis/{{scenario}}_consumer_tariff.csv",
        bills_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_SCENARIO_REGEX
    params: scenario_config=_consumer_scenario_params
    script: "../scripts/market/calculate_consumer_bill.py"


rule validate_consumer_bill_input_bundle:
    """Validate the portable annual input bundle before bill calculation."""
    input:
        rates_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "electricity_single_rate.csv"),
        metadata_json=lambda wc: _consumer_snapshot_file(wc.scenario, "metadata.json"),
        hedge_inputs_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "hedged_wholesale_inputs.csv"),
        policy_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "policy_components.csv"),
        network_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "network_components.csv"),
        smart_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "smart_meter_components.csv"),
        adjustment_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "adjustment_components.csv"),
        source_provenance_json=_consumer_bundle_provenance_input,
    output:
        validation_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_input_bundle.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    params:
        scenario_config=_consumer_scenario_params,
        bundle_dir=lambda wc: _consumer_bundle_dir(wc.scenario),
    script: "../scripts/consumer_bill/validate_consumer_bill_input_bundle.py"


rule build_historical_consumer_bill_bundle:
    """Fetch pinned Ofgem sources and compile a reusable annual input bundle."""
    input:
        manifest=lambda wc: _consumer_automated_bundles[wc.bundle],
    output:
        rates_csv="data/market/ofgem_price_cap/{bundle}/electricity_single_rate.csv",
        metadata_json="data/market/ofgem_price_cap/{bundle}/metadata.json",
        hedge_inputs_csv="data/market/ofgem_price_cap/{bundle}/hedged_wholesale_inputs.csv",
        wholesale_components_csv="data/market/ofgem_price_cap/{bundle}/wholesale_components.csv",
        policy_components_csv="data/market/ofgem_price_cap/{bundle}/policy_components.csv",
        network_components_csv="data/market/ofgem_price_cap/{bundle}/network_components.csv",
        smart_components_csv="data/market/ofgem_price_cap/{bundle}/smart_meter_components.csv",
        adjustment_components_csv="data/market/ofgem_price_cap/{bundle}/adjustment_components.csv",
        period_tariffs_csv="data/market/ofgem_price_cap/{bundle}/period_tariffs.csv",
        component_ledger_csv="data/market/ofgem_price_cap/{bundle}/component_ledger.csv",
        provenance_json="data/market/ofgem_price_cap/{bundle}/source_provenance.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: bundle=CONSUMER_BILL_AUTOMATED_BUNDLE_REGEX
    params:
        output_dir=lambda wc: f"data/market/ofgem_price_cap/{wc.bundle}",
        raw_dir=lambda wc: None,
    script: "../scripts/consumer_bill/build_historical_consumer_bill_bundle.py"


def _historical_report_dir(scenario_id):
    return f"{resources_path}/consumer_bill/{scenario_id}"


def _historical_consumption_profile(wildcards):
    path = _consumer_bill_config(wildcards.scenario).get("domestic_consumption_profile")
    return [path] if path else []


def _historical_report_evidence(wildcards, key):
    """Attach only explicitly configured saved model evidence to a report."""
    path = _consumer_bill_config(wildcards.scenario).get(
        "historical_report_evidence", {}
    ).get(key)
    return [path] if path else []


rule build_historical_consumer_bill_report:
    """Build published tariff accounting and annual Ofgem/model comparisons."""
    input:
        comparison_method="scripts/consumer_bill/compare_historical_components.py",
        period_tariffs=lambda wc: _consumer_snapshot_file(wc.scenario, "period_tariffs.csv"),
        component_ledger=lambda wc: _consumer_snapshot_file(wc.scenario, "component_ledger.csv"),
        bundle_validation=f"{resources_path}/analysis/{{scenario}}_consumer_bill_input_bundle.json",
        consumption_profile=_historical_consumption_profile,
        model_signals=lambda wc: _historical_report_evidence(wc, "model_signals_csv"),
        cfd_reconciliation=lambda wc: _historical_report_evidence(wc, "cfd_reconciliation_json"),
        cfd_mapping=lambda wc: _historical_report_evidence(wc, "cfd_mapping_json"),
        procurement_bridge=lambda wc: _historical_report_evidence(wc, "procurement_bridge_csv"),
        comparison_register=lambda wc: _historical_report_evidence(wc, "comparison_register_csv"),
        recovery_candidates=lambda wc: _historical_report_evidence(wc, "recovery_candidates_csv"),
        comparison_network_bridge=lambda wc: _historical_report_evidence(wc, "network_bridge_csv"),
    output:
        ofgem_model_comparison=f"{resources_path}/consumer_bill/{{scenario}}/ofgem_model_comparison.csv",
        ofgem_model_totals=f"{resources_path}/consumer_bill/{{scenario}}/ofgem_model_totals.csv",
        bill_components=f"{resources_path}/consumer_bill/{{scenario}}/bill_components.csv",
        period_bills=f"{resources_path}/consumer_bill/{{scenario}}/period_bills.csv",
        regional_bills=f"{resources_path}/consumer_bill/{{scenario}}/regional_bills.csv",
        model_validation=f"{resources_path}/consumer_bill/{{scenario}}/model_validation.csv",
        support_reconciliation=f"{resources_path}/consumer_bill/{{scenario}}/support_reconciliation.csv",
        recovery_bridge=f"{resources_path}/consumer_bill/{{scenario}}/recovery_bridge.csv",
        residuals=f"{resources_path}/consumer_bill/{{scenario}}/reconciliation_residuals.csv",
        provenance=f"{resources_path}/consumer_bill/{{scenario}}/provenance.json",
        checks=f"{resources_path}/consumer_bill/{{scenario}}/checks.json",
        report=f"{resources_path}/consumer_bill/{{scenario}}/report.html",
    params:
        bundle_dir=lambda wc: _consumer_bundle_dir(wc.scenario),
        output_dir=lambda wc: _historical_report_dir(wc.scenario),
        comparison_assumptions=lambda wc: _consumer_bill_config(wc.scenario).get("historical_comparison", {}),
        annual_consumption_kwh=lambda wc: float(
            _consumer_bill_config(wc.scenario).get("annual_consumption_kwh", [3100])[0]
        ),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=_consumer_regex(_consumer_historical_replay_ids)
    script: "../scripts/consumer_bill/build_historical_report.py"


rule consumer_bill_historical_sources:
    """Materialise and verify all configured pinned historical sources."""
    input:
        expand(
            "data/market/ofgem_price_cap/{bundle}/source_provenance.json",
            bundle=sorted(_consumer_automated_bundles),
        )


rule consumer_bill_historical_replay:
    """Build source-only tariff and component validation outputs."""
    input:
        expand(
            f"{resources_path}/consumer_bill/{{scenario}}/checks.json",
            scenario=_consumer_historical_replay_ids,
        )


rule consumer_bill_historical_audit:
    """Build model/settlement signal audits for configured historical runs."""
    input:
        expand(
            f"{resources_path}/analysis/{{scenario}}_pypsa_bill_signals_summary.json",
            scenario=[rid for rid in _consumer_audit_ids if rid in _consumer_historical_replay_ids],
        ),
        expand(
            f"{resources_path}/analysis/{{scenario}}_lccc_cfd_contract_mapping_summary.json",
            scenario=[rid for rid in _consumer_lccc_audit_ids if rid in _consumer_historical_replay_ids],
        ),
        expand(
            f"{resources_path}/analysis/{{scenario}}_lccc_cfd_model_dispatch_reconciliation_summary.json",
            scenario=[rid for rid in _consumer_lccc_audit_ids if rid in _consumer_historical_replay_ids],
        )


rule consumer_bill_historical_reports:
    """Build the maintained historical report index inputs."""
    input:
        consumer_bill_historical_report_targets


rule build_consumer_bill_historical_report_index:
    input:
        comparison_totals=expand(
            f"{resources_path}/consumer_bill/{{scenario}}/ofgem_model_totals.csv",
            scenario=_consumer_historical_replay_ids,
        ),
        reports=expand(
            f"{resources_path}/consumer_bill/{{scenario}}/report.html",
            scenario=_consumer_historical_replay_ids,
        ),
        checks=expand(
            f"{resources_path}/consumer_bill/{{scenario}}/checks.json",
            scenario=_consumer_historical_replay_ids,
        ),
    output:
        index=f"{resources_path}/consumer_bill/index.html",
    conda: "../envs/pypsa-gb.yaml"
    script: "../scripts/consumer_bill/build_historical_report_index.py"


rule analyze_consumer_bill_cost_stack:
    """Use market artifacts as inputs to the Ofgem retail cost-stack bridge."""
    input:
        unpack(_consumer_stack_model_inputs),
        bundle_validation=f"{resources_path}/analysis/{{scenario}}_consumer_bill_input_bundle.json",
        snapshot_metadata_json=lambda wc: _consumer_snapshot_file(wc.scenario, "metadata.json"),
        hedge_inputs_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "hedged_wholesale_inputs.csv"),
    output:
        wholesale_impact_csv=f"{resources_path}/analysis/{{scenario}}_wholesale_energy_impact.csv",
        forward_procurement_csv=f"{resources_path}/analysis/{{scenario}}_forward_procurement_periods.csv",
        retail_stack_csv=f"{resources_path}/analysis/{{scenario}}_retail_cost_stack.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_retail_cost_stack_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    params: scenario_config=_consumer_scenario_params
    script: "../scripts/market/analyze_consumer_bill_cost_stack.py"


rule extract_ofgem_2020_policy_components:
    input:
        jan_mar="data/market/ofgem_price_cap/2020_source/2019_10_to_2020_03_annex_4_v1.4.xlsx",
        apr_sep="data/market/ofgem_price_cap/2020_source/2020_04_to_2020_09_annex_4_v1.5.xlsx",
        oct_dec="data/market/ofgem_price_cap/2020_source/2020_10_to_2021_03_annex_4_v1.7.xlsx",
        hedge_inputs_csv="data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
    output:
        policy_components_csv="data/market/ofgem_price_cap/2020_annual/policy_components.csv",
        summary_json="data/market/ofgem_price_cap/2020_annual/policy_components_summary.json",
    params: source_dir="data/market/ofgem_price_cap/2020_source"
    conda: "../envs/pypsa-gb.yaml"
    script: "../scripts/market/extract_ofgem_2020_policy_components.py"


rule extract_ofgem_2020_network_components:
    input:
        jan_mar="data/market/ofgem_price_cap/2020_source/2019_10_to_2020_03_annex_3_network_v1.4.xlsx",
        apr_sep="data/market/ofgem_price_cap/2020_source/2020_04_to_2020_09_annex_3_network_v1.5.xlsx",
        oct_dec="data/market/ofgem_price_cap/2020_source/2020_10_to_2020_03_annex_3_network_v1.6.xlsx",
        hedge_inputs_csv="data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
    output:
        components_csv="data/market/ofgem_price_cap/2020_annual/network_components.csv",
        summary_json="data/market/ofgem_price_cap/2020_annual/network_components_summary.json",
    params: source_dir="data/market/ofgem_price_cap/2020_source"
    conda: "../envs/pypsa-gb.yaml"
    script: "../scripts/market/extract_ofgem_2020_network_components.py"


rule build_2020_domestic_pes_weights:
    """Map DESNZ domestic sales/accounts to the 14 Ofgem/PES regions."""
    input:
        domestic_consumption="data/market/consumer_bill_forecast/desnz_electricity_consumption_local_authority_2005_2024.csv",
        ons_centroids="data/network/GSP/ons_lad_2025_centroids.json",
        gsp_dno_lookup="data/network/GSP/neso_gsp_dno_lookup_20181031.csv",
    output:
        weights_csv="data/market/ofgem_price_cap/2020_annual/domestic_pes_weights.csv",
        summary_json="data/market/ofgem_price_cap/2020_annual/domestic_pes_weights_summary.json",
    params:
        year=2020,
    conda: "../envs/pypsa-gb.yaml"
    script: "../scripts/consumer_bill/build_domestic_pes_weights.py"


rule build_2020_regional_network_tariff_replay:
    """Replay regional Annex 3 network allowances using domestic PES weights."""
    input:
        domestic_weights="data/market/ofgem_price_cap/2020_annual/domestic_pes_weights.csv",
        jan_mar="data/market/ofgem_price_cap/2020_source/2019_10_to_2020_03_annex_3_network_v1.4.xlsx",
        apr_sep="data/market/ofgem_price_cap/2020_source/2020_04_to_2020_09_annex_3_network_v1.5.xlsx",
        oct_dec="data/market/ofgem_price_cap/2020_source/2020_10_to_2020_03_annex_3_network_v1.6.xlsx",
    output:
        replay_csv="data/market/ofgem_price_cap/2020_annual/regional_network_tariff_replay.csv",
        summary_json="data/market/ofgem_price_cap/2020_annual/regional_network_tariff_replay_summary.json",
    params:
        source_dir="data/market/ofgem_price_cap/2020_source",
    conda: "../envs/pypsa-gb.yaml"
    script: "../scripts/consumer_bill/build_regional_network_tariff_replay.py"


rule extract_ofgem_2020_smart_meter_components:
    input:
        jan_mar="data/market/ofgem_price_cap/2020_source/2019_10_to_2020_03_cap_model_v1.3.xlsx",
        apr_sep="data/market/ofgem_price_cap/2020_source/2020_04_to_2020_09_cap_model_v1.4.xlsx",
        oct_dec="data/market/ofgem_price_cap/2020_source/2020_10_to_2021_03_cap_model_v1.7.xlsx",
        hedge_inputs_csv="data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
    output:
        components_csv="data/market/ofgem_price_cap/2020_annual/smart_meter_components.csv",
        summary_json="data/market/ofgem_price_cap/2020_annual/smart_meter_components_summary.json",
    params: source_dir="data/market/ofgem_price_cap/2020_source"
    conda: "../envs/pypsa-gb.yaml"
    script: "../scripts/market/extract_ofgem_2020_smart_meter_components.py"


rule extract_ofgem_2020_adjustment_components:
    input:
        annex_8="data/market/ofgem_price_cap/2020_source/2020_10_to_2020_03_annex_8_adjustment_v1.2.xlsx",
        hedge_inputs_csv="data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
    output:
        components_csv="data/market/ofgem_price_cap/2020_annual/adjustment_components.csv",
        lineage_csv="data/market/ofgem_price_cap/2020_annual/adjustment_lineage.csv",
        summary_json="data/market/ofgem_price_cap/2020_annual/adjustment_components_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    script: "../scripts/market/extract_ofgem_2020_adjustment_components.py"


rule build_ofgem_benchmark_component_register:
    input:
        bundle_validation=f"{resources_path}/analysis/{{scenario}}_consumer_bill_input_bundle.json",
        snapshot_metadata_json=lambda wc: _consumer_snapshot_file(wc.scenario, "metadata.json"),
        retail_stack_csv=f"{resources_path}/analysis/{{scenario}}_retail_cost_stack.csv",
        policy_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "policy_components.csv"),
        network_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "network_components.csv"),
        smart_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "smart_meter_components.csv"),
        adjustment_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "adjustment_components.csv"),
    output:
        register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    script: "../scripts/market/build_consumer_bill_component_register.py"


rule build_consumer_bill_component_register:
    input:
        component_register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        bills_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill.csv",
    output:
        register_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_component_register.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_component_register_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    script: "../scripts/market/build_household_bill_component_register.py"


rule build_consumer_bill_forecast_registry:
    input: register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv"
    output:
        registry_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_forecast_registry.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_forecast_registry_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    script: "../scripts/market/build_consumer_bill_forecast_registry.py"


rule build_consumer_bill_recovery_bridges:
    """Expose the non-PyPSA recovery evidence needed for core bill components."""
    input:
        register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        wholesale_impact_csv=f"{resources_path}/analysis/{{scenario}}_wholesale_energy_impact.csv",
        market_summary_json=f"{resources_path}/analysis/{{scenario}}_market_summary.json",
        pypsa_signals_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_bill_signals.csv",
        loss_proxy_summary=_consumer_loss_proxy_input,
        assumptions_json=lambda wc: _consumer_bill_config(wc.scenario).get(
            "recovery_bridge_assumptions",
            "data/market/consumer_bill_forecast/recovery_bridge_assumptions_template.json",
        ),
    output:
        bridges_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_recovery_bridges.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_recovery_bridges_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_METHODOLOGY_AUDIT_REGEX
    script: "../scripts/consumer_bill/build_recovery_bridges.py"


rule aggregate_annual_physical_network_loss_proxy:
    """Build a calibrated I²R loss index from the complete physical monthly runs."""
    input:
        network_nc=lambda wc: [
            f"{resources_path}/market/{rid}_balancing.nc"
            for rid in _all_support_sources(wc.audit)
        ],
    output:
        losses_csv=f"{resources_path}/analysis/{{audit}}_physical_network_loss_diagnostic.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_physical_network_loss_diagnostic.json",
    params:
        expected_year=lambda wc: _all_support_year(wc.audit),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    script: "../scripts/consumer_bill/aggregate_physical_network_losses.py"


rule extract_pypsa_bill_signals:
    """Extract operational signals before translating them to bill recovery."""
    input:
        unpack(_consumer_signal_inputs),
        wholesale_price_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_price.csv",
        defaults=DEFAULTS_FILES,
        market_summary_json=f"{resources_path}/analysis/{{scenario}}_market_summary.json",
    output:
        signals_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_bill_signals.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_pypsa_bill_signals_summary.json",
        cfd_details_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_cfd_payment_proxy_details.csv",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_METHODOLOGY_AUDIT_REGEX
    params: scenario_config=lambda wc: {**_consumer_scenarios[wc.scenario], "scenario_id": wc.scenario}
    script: "../scripts/consumer_bill/extract_pypsa_bill_signals.py"


rule build_historical_support_outturn:
    """Screen CfD/RO evidence consistently before detailed historic reconciliation."""
    input:
        unpack(_consumer_support_outturn_inputs),
        market_summary_json=f"{resources_path}/analysis/{{scenario}}_market_summary.json",
        bill_register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        lccc_actuals_csv=_consumer_support_outturn_lccc_actuals,
        defaults=DEFAULTS_FILES,
    output:
        audit_csv=f"{resources_path}/analysis/{{scenario}}_historical_support_outturn.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_historical_support_outturn_summary.json",
    params:
        year=lambda wc: int(_consumer_scenarios[wc.scenario]["modelled_year"]),
        period_start=lambda wc: _consumer_scenarios[wc.scenario].get("solve_period", {}).get("start"),
        period_end=lambda wc: _consumer_scenarios[wc.scenario].get("solve_period", {}).get("end"),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_HISTORICAL_SUPPORT_OUTTURN_REGEX
    script: "../scripts/consumer_bill/build_historical_support_outturn.py"


rule reconcile_roc_model_dispatch:
    """Audit the station-labelled proxy and expose the aggregate ROC bridge."""
    input:
        network_nc=f"{resources_path}/market/{{scenario}}_balancing_roc_labelled.nc",
        reference_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_issued_by_technology.csv",
    output:
        reconciliation_csv=f"{resources_path}/analysis/{{scenario}}_roc_model_dispatch_reconciliation.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_roc_model_dispatch_reconciliation_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    script: "../scripts/consumer_bill/reconcile_roc_model_dispatch.py"


rule label_pypsa_roc_generators:
    """Recover REPD identities and apply actual calendar-year Ofgem ROC bands."""
    input:
        network_nc=f"{resources_path}/market/{{scenario}}_balancing.nc",
        outturn_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_issued_stations.csv",
        registry_csv="data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
        repd_crosswalk_csv=f"{resources_path}/analysis/{{scenario}}_repd_roc_station_mapping.csv",
        source_evidence_csv=f"{resources_path}/analysis/{{scenario}}_roc_station_source_evidence.csv",
        station_overrides_csv=lambda wc: _consumer_detailed_support_file(
            wc, "roc_station_overrides_csv",
            "data/market/ofgem_renewables_obligation/roc_pypsa_mapping_overrides_2020.csv",
        ),
        repd_sites=[
            f"{resources_path}/generators/sites/biomass_sites.csv",
            f"{resources_path}/generators/sites/waste_to_energy_sites.csv",
            f"{resources_path}/generators/sites/biogas_sites.csv",
            f"{resources_path}/generators/sites/landfill_gas_sites.csv",
            f"{resources_path}/generators/sites/sewage_gas_sites.csv",
            f"{resources_path}/generators/sites/advanced_biofuel_sites.csv",
            f"{resources_path}/generators/sites/large_hydro_sites.csv",
        ],
    output:
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_generator_roc_mapping.csv",
        network_nc=f"{resources_path}/market/{{scenario}}_balancing_roc_labelled.nc",
        summary_json=f"{resources_path}/analysis/{{scenario}}_pypsa_generator_roc_mapping_summary.json",
    params:
        buyout_price=lambda wc: float(_consumer_detailed_support_config(wc.scenario).get(
            "roc_buyout_price", 50.05
        ))
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    script: "../scripts/consumer_bill/map_pypsa_generators_to_ofgem_roc.py"


rule build_ofgem_roc_station_overlay:
    """Use the complete Ofgem inventory to specify safe ROC fleet actions."""
    input:
        outturn_csv=lambda wc: f"{resources_path}/analysis/ofgem_roc_{int(_consumer_scenarios[wc.scenario]['modelled_year'])}_calendar_issued_stations.csv",
        registry_csv="data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
        pre_solve_mapping_csv=f"{resources_path}/generators/{{scenario}}_pre_solve_roc_mapping.csv",
        physical_application_csv=f"{resources_path}/generators/{{scenario}}_pre_solve_roc_overlay_actions.csv",
        source_evidence_csv=f"{resources_path}/analysis/{{scenario}}_roc_station_source_evidence.csv",
        generators_csv=lambda wc: f"{resources_path}/generators/{_build_source(wc.scenario)}_generators_full.csv",
        action_overrides_csv=lambda wc: _consumer_detailed_support_file(
            wc, "roc_overlay_actions_csv",
            "data/market/ofgem_renewables_obligation/roc_historical_overlay_actions_2020.csv",
        ),
    output:
        overlay_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_station_overlay.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_station_overlay_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    shell:
        "python scripts/consumer_bill/build_ofgem_roc_station_overlay.py "
        "--outturn {input.outturn_csv} --registry {input.registry_csv} "
        "--pre-solve-mapping {input.pre_solve_mapping_csv} "
        "--physical-application {input.physical_application_csv} "
        "--source-evidence {input.source_evidence_csv} --generators {input.generators_csv} "
        "--action-overrides {input.action_overrides_csv} "
        "--overlay-output {output.overlay_csv} --summary-output {output.summary_json}"


rule complete_roc_station_representation:
    """Map every residual official station to an explicit assumed aggregate."""
    input:
        direct_mapping_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_generator_roc_mapping.csv",
        outturn_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_issued_stations.csv",
        registry_csv="data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
        ofgem_overlay_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_station_overlay.csv",
        assumptions_csv=lambda wc: _consumer_scenarios[wc.scenario].get("support", {}).get(
            "labels", {}
        ).get("historical_roc_station_labels", {}).get(
            "residual_representation_assumptions_file"
        ),
    output:
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_roc_station_representation_mapping.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_roc_station_representation_mapping_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    script: "../scripts/consumer_bill/complete_roc_station_representation.py"


rule build_ofgem_roc_residual_cohorts:
    """Preserve unmatched official ROC output as commissioning-year cohorts."""
    input:
        outturn_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_issued_stations.csv",
        registry_csv="data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_roc_station_representation_mapping.csv",
    output:
        station_ledger_csv=f"{resources_path}/analysis/{{scenario}}_roc_official_station_ledger.csv",
        residual_cohorts_csv=f"{resources_path}/analysis/{{scenario}}_roc_official_residual_cohorts.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_roc_official_residual_cohorts_summary.json",
    params:
        roc_eligibility_years=lambda wc: int(_consumer_scenarios[wc.scenario].get("support", {}).get(
            "labels", {}
        ).get("roc_eligibility_years", 20)),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    script: "../scripts/consumer_bill/build_ofgem_roc_residual_cohorts.py"


rule build_roc_mapping_priority_worklist:
    """Prioritise large official ROC residual stations for identity review."""
    input:
        station_ledger_csv=f"{resources_path}/analysis/{{scenario}}_roc_official_station_ledger.csv",
        repd_csv="data/renewables/repd-q2-jul-2025.csv",
        generators_csv=lambda wc: f"{resources_path}/generators/{_build_source(wc.scenario)}_generators_full.csv",
    output:
        worklist_csv=f"{resources_path}/analysis/{{scenario}}_roc_mapping_priority_worklist.csv",
        candidates_csv=f"{resources_path}/analysis/{{scenario}}_roc_mapping_priority_candidates.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_roc_mapping_priority_summary.json",
    params:
        minimum_capacity_mw=10.0,
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    script: "../scripts/consumer_bill/build_roc_mapping_priority_worklist.py"


rule audit_unresolved_roc_station_sources:
    """Search historical REPD and DUKES before treating a ROC station as absent."""
    input:
        worklist_csv=f"{resources_path}/analysis/{{scenario}}_roc_mapping_priority_worklist.csv",
        renewable_overrides_csv="data/market/support/renewable_support_overrides_2020.csv",
        generator_overrides_csv="data/market/support/generator_support_overrides_2020.csv",
        repd_2021_csv="data/renewables/renewable-energy-planning-database-q1-march-2021.csv",
        repd_2024_csv="data/renewables/repd-q2-july-2024.csv",
        repd_2025_csv="data/renewables/repd-q2-jul-2025.csv",
        dukes_xlsx="data/generators/DUKES_5.11_2025.xlsx",
    output:
        summary_csv=f"{resources_path}/analysis/{{scenario}}_roc_unresolved_source_audit.csv",
        candidates_csv=f"{resources_path}/analysis/{{scenario}}_roc_unresolved_source_audit_candidates.csv",
        metadata_json=f"{resources_path}/analysis/{{scenario}}_roc_unresolved_source_audit_summary.json",
    params:
        minimum_capacity_mw=10.0,
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    shell:
        "python scripts/consumer_bill/audit_unresolved_roc_station_sources.py "
        "--worklist {input.worklist_csv} "
        "--renewable-overrides {input.renewable_overrides_csv} "
        "--generator-overrides {input.generator_overrides_csv} "
        "--repd-2021 {input.repd_2021_csv} "
        "--repd-2024 {input.repd_2024_csv} "
        "--repd-2025 {input.repd_2025_csv} "
        "--dukes {input.dukes_xlsx} "
        "--summary-output {output.summary_csv} "
        "--candidates-output {output.candidates_csv} "
        "--metadata-output {output.metadata_json} "
        "--minimum-capacity-mw {params.minimum_capacity_mw}"


rule build_repd_roc_station_mapping:
    """Map historic REPD sites to Ofgem RO accreditations with explicit confidence."""
    input:
        repd_csv="data/renewables/repd-q2-jul-2025.csv",
        stations_csv="data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
    output:
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_repd_roc_station_mapping.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_repd_roc_station_mapping_summary.json",
    params:
        modelled_year=lambda wc: _consumer_scenarios[wc.scenario]["modelled_year"]
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=REPD_ROC_STATION_MAPPING_REGEX
    script: "../scripts/consumer_bill/build_repd_roc_station_mapping.py"


rule audit_pypsa_roc_capacity_coverage:
    """Separate direct station capacity from embedded/distributed model aggregates."""
    input:
        network_nc=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale.nc",
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_repd_roc_station_mapping.csv",
    output:
        coverage_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_roc_capacity_coverage.csv",
        generators_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_roc_capacity_by_representation.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_pypsa_roc_capacity_coverage_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    script: "../scripts/consumer_bill/audit_pypsa_roc_capacity_coverage.py"


rule extract_ofgem_roc_calendar_outturn:
    """Use Ofgem certificate output months to define the historic calendar-year target."""
    input:
        certificates_csv=lambda wc: _consumer_detailed_support_file(
            wc, "roc_certificates_csv",
            "data/market/ofgem_renewables_obligation/certificates_ro_current.csv",
        ),
    output:
        stations_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_issued_stations.csv",
        technology_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_issued_by_technology.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_ofgem_roc_issued_summary.json",
    params:
        year=lambda wc: _consumer_scenarios[wc.scenario]["modelled_year"],
        period_start=lambda wc: _consumer_scenarios[wc.scenario].get("solve_period", {}).get("start"),
        period_end=lambda wc: _consumer_scenarios[wc.scenario].get("solve_period", {}).get("end"),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_ROC_AUDIT_REGEX
    script: "../scripts/consumer_bill/extract_ofgem_roc_calendar_outturn.py"


rule extract_annual_ofgem_roc_calendar_outturn:
    """Build the full calendar-year ROC target for the twelve-month audit."""
    input:
        certificates_csv="data/market/ofgem_renewables_obligation/certificates_ro_current.csv",
    output:
        stations_csv=f"{resources_path}/analysis/{{audit}}_ofgem_roc_issued_stations.csv",
        technology_csv=f"{resources_path}/analysis/{{audit}}_ofgem_roc_issued_by_technology.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_ofgem_roc_issued_summary.json",
    params:
        year=lambda wc: _all_support_year(wc.audit),
        period_start=lambda wc: None,
        period_end=lambda wc: None,
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    script: "../scripts/consumer_bill/extract_ofgem_roc_calendar_outturn.py"


rule build_annual_ofgem_roc_residual_cohorts:
    """Create the annual official ROC cohort ledger from all monthly mappings."""
    input:
        outturn_csv=f"{resources_path}/analysis/{{audit}}_ofgem_roc_issued_stations.csv",
        registry_csv="data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
        mapping_csv=lambda wc: [
            f"{resources_path}/analysis/{rid}_roc_station_representation_mapping.csv"
            for rid in _all_support_sources(wc.audit)
        ],
    output:
        station_ledger_csv=f"{resources_path}/analysis/{{audit}}_roc_official_station_ledger.csv",
        residual_cohorts_csv=f"{resources_path}/analysis/{{audit}}_roc_official_residual_cohorts.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_roc_official_residual_cohorts_summary.json",
    params:
        roc_eligibility_years=lambda wc: int(_consumer_scenarios[_all_support_first_source(wc.audit)].get(
            "support", {}
        ).get("labels", {}).get("roc_eligibility_years", 20)),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    script: "../scripts/consumer_bill/build_ofgem_roc_residual_cohorts.py"


rule build_lccc_cfd_contract_mapping:
    """Map every LCCC CfD contract to model assets or an explicit gap."""
    input:
        actuals_csv="data/market/lccc/actual_cfd_generation_and_payments.csv",
        portfolio_csv="data/market/lccc/cfd_contract_portfolio_status.csv",
        cfd_bmu_csv="data/market/lccc/cfd_to_bm_unit_mapping.csv",
        generators_csv=lambda wc: f"{resources_path}/generators/{_build_source(wc.scenario)}_generators_full.csv",
        model_bmu_csv=f"{resources_path}/generators/{{scenario}}_bmu_mapping.csv",
        overrides_csv=_consumer_cfd_mapping_overrides,
        source_candidates_csv=_consumer_cfd_source_candidates,
    output:
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_contract_mapping.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_contract_mapping_summary.json",
        unresolved_worklist_csv=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_unresolved_mapping_worklist.csv",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_LCCC_AUDIT_REGEX
    params:
        year=lambda wc: _consumer_scenarios[wc.scenario]["modelled_year"],
        period_start=lambda wc: _consumer_scenarios[wc.scenario].get("solve_period", {}).get("start"),
        period_end=lambda wc: _consumer_scenarios[wc.scenario].get("solve_period", {}).get("end"),
    script: "../scripts/consumer_bill/build_lccc_cfd_mapping.py"


rule reconcile_lccc_cfd_model_dispatch:
    """Separate missing contracts from PyPSA dispatch differences in the CfD backtest."""
    input:
        network_nc=f"{resources_path}/market/{{scenario}}_balancing.nc",
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_contract_mapping.csv",
        actuals_csv="data/market/lccc/actual_cfd_generation_and_payments.csv",
    output:
        reconciliation_csv=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_model_dispatch_reconciliation.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_model_dispatch_reconciliation_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_LCCC_AUDIT_REGEX
    script: "../scripts/consumer_bill/reconcile_lccc_cfd_model_dispatch.py"


rule audit_renewable_performance_factors:
    """Expose carrier-factor bias separately for named CfD contracts and cohorts."""
    input:
        network_nc=f"{resources_path}/market/{{scenario}}_balancing.nc",
        mapping_csv=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_contract_mapping.csv",
        reconciliation_csv=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_model_dispatch_reconciliation.csv",
    output:
        contracts_csv=f"{resources_path}/analysis/{{scenario}}_renewable_performance_factor_contracts.csv",
        assets_csv=f"{resources_path}/analysis/{{scenario}}_renewable_performance_factor_assets.csv",
        cohorts_csv=f"{resources_path}/analysis/{{scenario}}_renewable_performance_factor_cohorts.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_renewable_performance_factor_summary.json",
    params:
        performance_config=lambda wc: _consumer_scenarios[wc.scenario].get("renewables", {}).get(
            "performance_factors", {}
        ),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_LCCC_AUDIT_REGEX
    script: "../scripts/consumer_bill/audit_renewable_performance_factors.py"


rule build_historical_alignment_scorecard:
    """Keep physical, CfD and ROC validation on explicit like-for-like scopes."""
    input:
        network_nc=f"{resources_path}/market/{{scenario}}_balancing.nc",
        espeni_csv="data/demand/espeni.csv",
        cfd_summary_json=f"{resources_path}/analysis/{{scenario}}_lccc_cfd_model_dispatch_reconciliation_summary.json",
        roc_summary_json=f"{resources_path}/analysis/{{scenario}}_roc_model_dispatch_reconciliation_summary.json",
        roc_reconciliation_csv=f"{resources_path}/analysis/{{scenario}}_roc_model_dispatch_reconciliation.csv",
    output:
        scorecard_csv=f"{resources_path}/analysis/{{scenario}}_historical_alignment_scorecard.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_historical_alignment_scorecard_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_LCCC_AUDIT_REGEX
    script: "../scripts/consumer_bill/build_historical_alignment_scorecard.py"


rule aggregate_monthly_support_generator_registers:
    """Union monthly asset registers so in-year commissioning remains visible."""
    input:
        # Deduplicated: months sharing one network_from parent share its register.
        generators=lambda wc: list(dict.fromkeys(
            f"{resources_path}/generators/{_build_source(rid)}_generators_full.csv"
            for rid in _all_support_sources(wc.audit)
        )),
        bmus=lambda wc: [
            f"{resources_path}/generators/{rid}_bmu_mapping.csv"
            for rid in _all_support_sources(wc.audit)
        ],
    output:
        generators_csv=f"{resources_path}/analysis/{{audit}}_annual_generator_register.csv",
        bmus_csv=f"{resources_path}/analysis/{{audit}}_annual_bmu_mapping.csv",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    script: "../scripts/consumer_bill/aggregate_monthly_generator_registers.py"


rule build_annual_lccc_cfd_contract_mapping:
    """Map the complete calendar-year CfD register to the union of monthly assets."""
    input:
        actuals_csv="data/market/lccc/actual_cfd_generation_and_payments.csv",
        portfolio_csv="data/market/lccc/cfd_contract_portfolio_status.csv",
        cfd_bmu_csv="data/market/lccc/cfd_to_bm_unit_mapping.csv",
        generators_csv=f"{resources_path}/analysis/{{audit}}_annual_generator_register.csv",
        model_bmu_csv=f"{resources_path}/analysis/{{audit}}_annual_bmu_mapping.csv",
        overrides_csv="data/market/lccc/cfd_pypsa_mapping_overrides.csv",
        source_candidates_csv="data/market/lccc/cfd_pypsa_source_candidates.csv",
    output:
        mapping_csv=f"{resources_path}/analysis/{{audit}}_lccc_cfd_contract_mapping.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_lccc_cfd_contract_mapping_summary.json",
        unresolved_worklist_csv=f"{resources_path}/analysis/{{audit}}_lccc_cfd_unresolved_mapping_worklist.csv",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    params:
        year=lambda wc: _all_support_year(wc.audit),
        period_start=lambda wc: None,
        period_end=lambda wc: None,
    script: "../scripts/consumer_bill/build_lccc_cfd_mapping.py"


rule reconcile_annual_lccc_cfd_model_dispatch:
    """Sum monthly solved dispatch before comparing with calendar-year LCCC actuals."""
    input:
        network_nc=lambda wc: [
            f"{resources_path}/market/{rid}_balancing.nc"
            for rid in _all_support_sources(wc.audit)
        ],
        mapping_csv=f"{resources_path}/analysis/{{audit}}_lccc_cfd_contract_mapping.csv",
        actuals_csv="data/market/lccc/actual_cfd_generation_and_payments.csv",
    output:
        reconciliation_csv=f"{resources_path}/analysis/{{audit}}_lccc_cfd_model_dispatch_reconciliation.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_lccc_cfd_model_dispatch_reconciliation_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    script: "../scripts/consumer_bill/reconcile_lccc_cfd_model_dispatch.py"


rule reconcile_annual_roc_model_dispatch:
    """Sum station-labelled monthly dispatch before the calendar-year Ofgem comparison."""
    input:
        network_nc=lambda wc: [
            f"{resources_path}/market/{rid}_balancing_roc_labelled.nc"
            for rid in _all_support_sources(wc.audit)
        ],
        reference_csv=f"{resources_path}/analysis/{{audit}}_ofgem_roc_issued_by_technology.csv",
    output:
        reconciliation_csv=f"{resources_path}/analysis/{{audit}}_roc_model_dispatch_reconciliation.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_roc_model_dispatch_reconciliation_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    script: "../scripts/consumer_bill/reconcile_roc_model_dispatch.py"


rule audit_annual_lccc_profile_calibration:
    """Compare raw site profiles with every LCCC CfD contract of the audit year before solving."""
    input:
        actuals_csv="data/market/lccc/actual_cfd_generation_and_payments.csv",
        generators=lambda wc: list(dict.fromkeys(
            f"{resources_path}/generators/{_build_source(rid)}_generators_full.csv"
            for rid in _all_support_sources(wc.audit)
        )),
        # The script slices each profile to the audit's calendar year, so the
        # profiles are that year's (it was hard-coded to 2020).
        profiles=lambda wc: [
            f"{resources_path}/renewable/profiles/{carrier}_{_all_support_year(wc.audit)}.csv"
            for carrier in ("wind_offshore", "wind_onshore", "solar_pv")
        ],
    output:
        contracts_csv=f"{resources_path}/analysis/{{audit}}_lccc_profile_calibration_contracts.csv",
        monthly_csv=f"{resources_path}/analysis/{{audit}}_lccc_profile_calibration_monthly.csv",
        cohorts_csv=f"{resources_path}/analysis/{{audit}}_lccc_profile_calibration_cohorts.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_lccc_profile_calibration_summary.json",
    params:
        year=lambda wc: _all_support_year(wc.audit),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_ALL_SUPPORT_AUDIT_REGEX
    script: "../scripts/consumer_bill/audit_lccc_profile_calibration.py"


rule build_annual_support_to_bill_bridge:
    """Keep annual CfD/ROC evidence current without assuming a levy pass-through."""
    input:
        policy_components_csv=lambda wc: (
            f"data/market/ofgem_price_cap/{_support_bill_config(wc.audit).get('cap_snapshot', '2020_annual')}/policy_components.csv"
        ),
        cfd_summary_json=f"{resources_path}/analysis/{{audit}}_lccc_cfd_model_dispatch_reconciliation_summary.json",
        cfd_mapping_summary_json=f"{resources_path}/analysis/{{audit}}_lccc_cfd_contract_mapping_summary.json",
        roc_summary_json=f"{resources_path}/analysis/{{audit}}_roc_model_dispatch_reconciliation_summary.json",
        roc_ledger_summary_json=f"{resources_path}/analysis/{{audit}}_roc_official_residual_cohorts_summary.json",
        cfd_levy_csv=lambda wc: _support_bill_config(wc.audit).get(
            "cfd_levy_recovery_file",
            "data/market/consumer_bill_forecast/2020_cfd_reconciled_levy_recovery.csv",
        ),
        monthly_roc_summaries=lambda wc: [
            f"{resources_path}/analysis/{rid}_roc_model_dispatch_reconciliation_summary.json"
            for rid in _all_support_sources(wc.audit)
        ],
        roc_scheme_parameters_json=lambda wc: _support_bill_config(wc.audit).get(
            "roc_scheme_parameters_file",
            "data/market/consumer_bill_forecast/2020_ro_scheme_recovery_parameters.json",
        ),
        assumptions_json=lambda wc: _support_bill_config(wc.audit).get(
            "recovery_bridge_assumptions",
            "data/market/consumer_bill_forecast/recovery_bridge_assumptions_template.json",
        ),
    output:
        bridge_csv=f"{resources_path}/analysis/{{audit}}_support_to_bill_bridge.csv",
        summary_json=f"{resources_path}/analysis/{{audit}}_support_to_bill_bridge_summary.json",
        report_html=f"{resources_path}/analysis/{{audit}}_support_to_bill_bridge.html",
    params:
        year=lambda wc: _all_support_year(wc.audit),
        expected_months=lambda wc: len(_all_support_sources(wc.audit)),
        monthly_starts=lambda wc: [
            _consumer_scenarios[rid].get("solve_period", {}).get("start")
            for rid in _all_support_sources(wc.audit)
        ],
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: audit=CONSUMER_SUPPORT_BILL_AUDIT_REGEX
    script: "../scripts/consumer_bill/build_annual_support_to_bill_bridge.py"


rule backtest_consumer_bill:
    """Compare a configured historical annual bill with named reconstruction paths."""
    input:
        register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        forecast_registry_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_forecast_registry.csv",
        wholesale_impact_csv=f"{resources_path}/analysis/{{scenario}}_wholesale_energy_impact.csv",
        recovery_bridges_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_recovery_bridges.csv",
        support_bridge_csv=_consumer_support_bridge_input,
        forward_procurement_bridge_csv=_consumer_procurement_bridge_input,
        network_bridge_csv=_consumer_network_bridge_input,
    output:
        backtest_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_backtest.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_backtest_summary.json",
        report_html=f"{resources_path}/analysis/{{scenario}}_consumer_bill_backtest.html",
    conda: "../envs/pypsa-gb.yaml"
    params:
        include_same_year_support_backtest=lambda wc: _consumer_bill_config(wc.scenario).get(
            "include_same_year_support_backtest", False
        ),
        include_same_year_network_backtest=lambda wc: _consumer_bill_config(wc.scenario).get(
            "include_same_year_network_backtest", False
        )
    wildcard_constraints: scenario=CONSUMER_BILL_BACKTEST_REGEX
    script: "../scripts/consumer_bill/build_2020_backtest.py"


rule build_ofgem_forward_procurement_bridge:
    """Reconstruct observed 2020 forward procurement; retain PyPSA only as a diagnostic."""
    input:
        register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        procurement_periods_csv=f"{resources_path}/analysis/{{scenario}}_forward_procurement_periods.csv",
        wholesale_impact_csv=f"{resources_path}/analysis/{{scenario}}_wholesale_energy_impact.csv",
    output:
        bridge_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_forward_procurement_bridge.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_ofgem_forward_procurement_bridge_summary.json",
        report_html=f"{resources_path}/analysis/{{scenario}}_ofgem_forward_procurement_bridge.html",
    conda: "../envs/pypsa-gb.yaml"
    # Needs no balancing-stage output, so wholesale_only cost stacks get it too.
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    script: "../scripts/consumer_bill/build_ofgem_forward_procurement_bridge.py"


rule build_pypsa_spot_procurement_bridge:
    """Use a PyPSA delivery spot price in an explicitly unfitted historic backtest."""
    input:
        register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        hedge_inputs_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "hedged_wholesale_inputs.csv"),
        wholesale_impact_csv=f"{resources_path}/analysis/{{scenario}}_wholesale_energy_impact.csv",
    output:
        bridge_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_spot_procurement_bridge.csv",
        periods_csv=f"{resources_path}/analysis/{{scenario}}_pypsa_spot_procurement_periods.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_pypsa_spot_procurement_bridge_summary.json",
        report_html=f"{resources_path}/analysis/{{scenario}}_pypsa_spot_procurement_bridge.html",
    conda: "../envs/pypsa-gb.yaml"
    params:
        forward_procurement=lambda wc: _consumer_bill_config(wc.scenario).get("forward_procurement", {})
    # Needs no balancing-stage output, so wholesale_only cost stacks get it too.
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    script: "../scripts/consumer_bill/build_pypsa_spot_procurement_bridge.py"


rule build_network_cost_to_bill_bridge:
    """Allocate named 2020 network allowances using valid PyPSA signals."""
    input:
        register_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        network_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "network_components.csv"),
        wholesale_impact_csv=f"{resources_path}/analysis/{{scenario}}_wholesale_energy_impact.csv",
        recovery_bridges_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_recovery_bridges.csv",
    output:
        bridge_csv=f"{resources_path}/analysis/{{scenario}}_network_cost_bridge.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_network_cost_bridge_summary.json",
        report_html=f"{resources_path}/analysis/{{scenario}}_network_cost_bridge.html",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_METHODOLOGY_AUDIT_REGEX
    script: "../scripts/consumer_bill/build_network_cost_to_bill_bridge.py"


rule build_network_cost_tariff_layers:
    """Expose regional tariff, loss and future allowed-revenue network layers."""
    input:
        regional_replay=lambda wc: _consumer_bill_config(wc.scenario).get(
            "network_layers", {}
        ).get("regional_replay_csv", _consumer_snapshot_file(wc.scenario, "regional_network_tariff_replay.csv")),
        domestic_weights=lambda wc: _consumer_bill_config(wc.scenario).get(
            "network_layers", {}
        ).get("domestic_weights_csv", _consumer_snapshot_file(wc.scenario, "domestic_pes_weights.csv")),
        loss_proxy_summary=_consumer_loss_proxy_input,
        future_assumptions=lambda wc: _consumer_bill_config(wc.scenario).get(
            "network_allowed_revenue_assumptions",
            "data/market/consumer_bill_forecast/network_allowed_revenue_projection_template.json",
        ),
        neso_tnuos="data/market/network_cost_sources/tnuos/neso_tnuos_demand_nhh.csv",
    output:
        tariffs_csv=f"{resources_path}/analysis/{{scenario}}_network_cost_tariff_layers.csv",
        projection_csv=f"{resources_path}/analysis/{{scenario}}_network_cost_allowed_revenue_projection.csv",
        historic_tnuos_csv=f"{resources_path}/analysis/{{scenario}}_network_cost_neso_tnuos_historic_comparator.csv",
        future_tnuos_csv=f"{resources_path}/analysis/{{scenario}}_network_cost_neso_tnuos_future_outlook.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_network_cost_layers_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_NETWORK_LAYER_REGEX
    params:
        historic_tnuos_financial_year=lambda wc: int(_consumer_bill_config(wc.scenario).get(
            "network_layers", {}
        ).get("historic_tnuos_financial_year", int(_consumer_scenarios[wc.scenario]["modelled_year"]) + 1)),
    script: "../scripts/consumer_bill/build_network_cost_layers.py"


rule create_consumer_bill_breakdown:
    input:
        bills_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill.csv",
        retail_stack_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_component_register.csv",
        policy_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "policy_components.csv"),
    output:
        breakdown_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_breakdown.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_breakdown_summary.json",
        report_html=f"{resources_path}/analysis/{{scenario}}_consumer_bill_breakdown.html",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    script: "../scripts/market/consumer_bill_breakdown.py"


rule review_consumer_bill:
    """One entry point for categories, granular charges and model evidence."""
    input:
        bills=f"{resources_path}/analysis/{{scenario}}_consumer_bill.csv",
        register=f"{resources_path}/analysis/{{scenario}}_consumer_bill_component_register.csv",
        bundle=f"{resources_path}/analysis/{{scenario}}_consumer_bill_input_bundle.json",
        bridges=lambda wc: [f"{resources_path}/analysis/{wc.scenario}_consumer_bill_recovery_bridges.csv"] if wc.scenario in _consumer_audit_ids else [],
        signals=lambda wc: [f"{resources_path}/analysis/{wc.scenario}_pypsa_bill_signals.csv"] if wc.scenario in _consumer_audit_ids else [],
        backtest=lambda wc: [f"{resources_path}/analysis/{wc.scenario}_consumer_bill_backtest_summary.json"] if wc.scenario in _consumer_backtest_ids else [],
    output:
        components=f"{resources_path}/analysis/{{scenario}}_consumer_bill_review.csv",
        summary=f"{resources_path}/analysis/{{scenario}}_consumer_bill_review_summary.json",
        report=f"{resources_path}/analysis/{{scenario}}_consumer_bill_review.html",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_COST_STACK_REGEX
    script: "../scripts/consumer_bill/review_consumer_bill.py"


rule consumer_bill_reports:
    """Build bill review pages for active scenarios with cost_stack enabled."""
    input:
        expand(f"{resources_path}/analysis/{{scenario}}_consumer_bill_review.html", scenario=_consumer_cost_stack_ids),
        consumer_bill_historical_report_targets,


# Future cases have explicit annual model inputs and assumptions. They remain an
# opt-in target because they can span twelve independently solved market runs.
_consumer_future_cases = _consumer_defaults.get("consumer_bill", {}).get("forecasts", {}).get("cases", {})


def _consumer_physical_scenarios(case):
    """The months whose physical (balancing) dispatch a case's CfD line and RO load factors settle on.

    The case's own months, except under central dispatch (P1/P2 under I3, 20 Oct 2026): a
    case with ``institution: central_dispatch`` keeps its market twin's months for Stage 1
    (the national price is the copperplate price, so the energy line is the twin's) and meters
    the CfD and the RO on the central-dispatch months (``central_scenario_pattern``).
    """
    if case.get("institution") == "central_dispatch":
        return [case["central_scenario_pattern"].format(month=month) for month in _consumer_months]
    return list(case["source_scenarios"])


for _case_id, _case in _consumer_future_cases.items():
    _case["case_id"] = _case_id
    _case["source_scenarios"] = [
        _case["source_scenario_pattern"].format(month=month)
        for month in _consumer_months
    ]
    _case["physical_scenarios"] = _consumer_physical_scenarios(_case)
# D31: the studies' forecast cases, expanded the same way. They answer the rules
# below like any case, but only defaults' cases join consumer_bill_future_reports
# (and so rule all): a study's months are requested by name.
_consumer_study_future_cases = {}
for _case_id in _CONSUMER_STUDIES.cases():
    _case = _CONSUMER_STUDIES.case(_case_id)
    _case["case_id"] = _case_id
    _case["source_scenarios"] = [
        _case["source_scenario_pattern"].format(month=month)
        for month in _consumer_months
    ]
    _case["physical_scenarios"] = _consumer_physical_scenarios(_case)
    _consumer_study_future_cases[_case_id] = _case
_consumer_all_future_cases = {**_consumer_future_cases, **_consumer_study_future_cases}
# P1/P2 under central dispatch (20 Oct 2026): a central-dispatch case's constraint line comes from
# the year FTI settlement (extract_central_dispatch_costs), every other case's from its balancing stage.
_consumer_central_dispatch_cases = {c: v for c, v in _consumer_all_future_cases.items()
                                    if v.get("institution") == "central_dispatch"}
_consumer_market_institution_cases = [c for c in _consumer_all_future_cases if c not in _consumer_central_dispatch_cases]

def _consumer_future_assumptions(bill_case):
    """Read a future case's assumptions file at DAG-construction time.

    The strike table, the domestic profile and the CPI indexation all live in
    that file, and the balancing-cost rule needs the indexation before any job
    runs, so the file is parsed here rather than only inside the scripts.

    FES 2050 decision 70: a central-dispatch strike case's assumptions are a rule output
    (build_central_dispatch_strike_inputs_2050), so they are built here in memory from its
    market case's file, exactly as the rule writes them; every other case reads its file.
    """
    case = _consumer_all_future_cases[bill_case]
    if case.get("central_dispatch_strikes"):
        from scripts.public_ownership.vintage_capital_2050 import (
            central_dispatch_assumptions, central_dispatch_files, central_strikes_spec)

        spec = central_strikes_spec(case, bill_case)
        files = central_dispatch_files(bill_case, resources_path)
        if str(case["assumptions"]) != files["assumptions"]:
            raise ValueError(f"{bill_case}: a central-dispatch strike case's assumptions are {files['assumptions']} "
                             f"(the generated file), not {case['assumptions']}")
        with open(spec["assumptions_from"], encoding="utf-8") as handle:
            return central_dispatch_assumptions(json.load(handle), case=bill_case,
                                                strike_table=files["strike_table"],
                                                contract_book=files["contract_book"])
    with open(case["assumptions"], encoding="utf-8") as handle:
        return json.load(handle)


def _consumer_bill_support_inputs(assumptions):
    """Task SETTLE2050: the 2050 contract book and DPA settlement a bill's rules name (none for a 2030 bill).

    The 2050 bill settles the vintaged register's own contracts
    (``model_linked_rules.cfd.contract_book``) and carries the support settlement's DPA
    total (``model_linked_rules.dpa.settlement_summary``); both are rule outputs, so the
    rules that read the bill's assumptions declare them. A 2030 bill names neither, so
    its rules' inputs are unchanged.
    """
    rules_ = assumptions["model_linked_rules"]
    out = [rules_["cfd"]["contract_book"]] if rules_["cfd"].get("contract_book") else []
    summary = (rules_.get("dpa") or {}).get("settlement_summary")
    out += [summary] if summary else []
    # Decision 41 (2050): a line booked from a register-backed public account
    # (sourced_externals.<line>.derivation.cost_from.public_account, e.g. Sizewell C's RAB)
    # reads that file, so the bill rules declare it; no 2030 line names one.
    for line in (assumptions.get("sourced_externals") or {}).values():
        cost_from = ((line or {}).get("derivation") or {}).get("cost_from") or {}
        if cost_from.get("public_account") and cost_from["public_account"] not in out:
            out.append(cost_from["public_account"])
    return out


rule extract_future_balancing_costs:
    """Constraint costs plus the CfD compensation the balancing LP prices at zero."""
    input:
        assumptions=lambda wc: _consumer_all_future_cases[wc.bill_case]["assumptions"],
        strike_table=lambda wc: _consumer_future_assumptions(wc.bill_case)["model_linked_rules"]["cfd"]["strike_table"],
        wholesale_networks=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale.nc" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        balancing_networks=lambda wc: [f"{resources_path}/market/{scenario}_balancing.nc" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        prices=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale_price.csv" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        summaries=lambda wc: [f"{resources_path}/analysis/{scenario}_market_summary.json" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        # Per-generator bid costs reconcile the recomputed CfD compensation against
        # the LP's own; the bid audit says which pricing the solve actually used.
        redispatch_summaries=lambda wc: [f"{resources_path}/market/{scenario}_redispatch_summary.csv" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        # Pro rata zonal turn-down (20 Oct 2026): each month is read on its network's basis.
        turn_down=lambda wc: turn_down_month_inputs(_consumer_all_future_cases[wc.bill_case]["source_scenarios"]),
        contract_value_bids=lambda wc: [f"{resources_path}/market/{scenario}_contract_value_bids.csv" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        # Task SETTLE2050: the 2050 contract book the compensation is recomputed on (none in 2030).
        support_book=lambda wc: [f for f in _consumer_bill_support_inputs(_consumer_future_assumptions(wc.bill_case))
                                 if f.endswith("_support_contract_book.csv")],
    output:
        monthly=f"{resources_path}/analysis/{{bill_case}}_future_balancing_costs_monthly.csv",
        carriers=f"{resources_path}/analysis/{{bill_case}}_future_balancing_costs_by_carrier.csv",
        summary=f"{resources_path}/analysis/{{bill_case}}_future_balancing_costs_summary.json",
    params:
        case=lambda wc: _consumer_all_future_cases[wc.bill_case],
        indexation=lambda wc: _consumer_future_assumptions(wc.bill_case)["cfd_strike_indexation"],
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: bill_case=_consumer_regex(_consumer_market_institution_cases)
    script: "../scripts/consumer_bill/extract_future_balancing_costs.py"


def _consumer_market_wholesale_flag(scenario_ids):
    """`` --market-wholesale-scenarios <parents>`` when the months take Stage 1 through ``wholesale_from``, else ''."""
    sources = [_wholesale_source(s) for s in scenario_ids]
    return " --market-wholesale-scenarios " + " ".join(sources) if sources != list(scenario_ids) else ""


rule extract_central_dispatch_costs:
    """P1/P2 under central dispatch (20 Oct 2026): a central-dispatch case's constraint line from the year FTI settlement.

    Cost recovery, firm access (with the lost CfD and ROC support), the storage and
    interconnector make-whole and the borrowed forecast-error charge, in the schema the
    projector reads; refuses a market year and a central-dispatch year from different
    solve runs (scripts/consumer_bill/extract_central_dispatch_costs.py).
    """
    input:
        script="scripts/consumer_bill/extract_central_dispatch_costs.py",
        settlement=lambda wc: _consumer_all_future_cases[wc.bill_case]["fti_settlement"],
        # The per-unit transfers fti_settle_year writes beside the year table.
        units=lambda wc: re.sub(r"\.csv$", "_units.csv", str(_consumer_all_future_cases[wc.bill_case]["fti_settlement"])),
        market_summary=lambda wc: (f"{resources_path}/analysis/"
                                   f"{_consumer_all_future_cases[wc.bill_case]['institution_parent']}"
                                   "_future_balancing_costs_summary.json"),
        # The two years' solve identities (file write times) for the mixed-vintage refusal.
        market_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc"
                                     for s in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        central_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc"
                                      for s in _consumer_all_future_cases[wc.bill_case]["physical_scenarios"]],
    output:
        monthly=f"{resources_path}/analysis/{{bill_case}}_future_balancing_costs_monthly.csv",
        carriers=f"{resources_path}/analysis/{{bill_case}}_future_balancing_costs_by_carrier.csv",
        summary=f"{resources_path}/analysis/{{bill_case}}_future_balancing_costs_summary.json",
    params:
        model_price_year=lambda wc: _consumer_all_future_cases[wc.bill_case]["model_price_year"],
        market=lambda wc: " ".join(_consumer_all_future_cases[wc.bill_case]["source_scenarios"]),
        market_wholesale=lambda wc: _consumer_market_wholesale_flag(
            _consumer_all_future_cases[wc.bill_case]["source_scenarios"]),
        central=lambda wc: " ".join(_consumer_all_future_cases[wc.bill_case]["physical_scenarios"]),
        # The same window the institution comparison judges two solve runs by (code default 24 h).
        gap=lambda wc: float(((_consumer_defaults.get("public_ownership") or {}).get("institution_comparison") or {})
                             .get("mixed_vintage_gap_hours", 24)),
        premium=lambda wc: ("" if _consumer_all_future_cases[wc.bill_case].get(
            "counter_trade_premium_on_constraint_line", True) else " --no-counter-trade-premium"),
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: bill_case=_consumer_regex(_consumer_central_dispatch_cases)
    shell:
        "python {input.script} {input.settlement} {input.market_summary} {output.monthly} {output.carriers}"
        " {output.summary} --model-price-year {params.model_price_year} --units {input.units}"
        " --market-scenarios {params.market}{params.market_wholesale} --central-scenarios {params.central}"
        " --mixed-vintage-gap-hours {params.gap}{params.premium}"


rule build_future_ro_scheme_cost:
    """Value the ROC-eligible fleet still accredited in the case's obligation year.

    The fleet comes from the Ofgem register rather than the network: the CP30
    renewable fleet is synthetic and carries no accreditation dates, so no
    generator can hold a ROC label. Only the load factors are model-derived, and
    they come from the balancing networks because ROCs are issued on metered
    output.
    """
    input:
        register="data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
        # Observed bandings and the fuelled-station activity test (item MR8).
        certificates="data/market/ofgem_renewables_obligation/certificates_ro_current.csv",
        parameters=lambda wc: _consumer_all_future_cases[wc.bill_case]["ro_scheme_parameters"],
        recovery_demand=lambda wc: _consumer_all_future_cases[wc.bill_case]["recovery_demand_assumptions"],
        # ROCs are issued on metered output: the physical months (a central-dispatch case's I3 year).
        networks=lambda wc: [f"{resources_path}/market/{scenario}_balancing.nc" for scenario in _consumer_all_future_cases[wc.bill_case]["physical_scenarios"]],
        turn_down=lambda wc: turn_down_month_inputs(_consumer_all_future_cases[wc.bill_case]["physical_scenarios"]),
    output:
        by_carrier=f"{resources_path}/analysis/{{bill_case}}_ro_eligible_fleet.csv",
        summary=f"{resources_path}/analysis/{{bill_case}}_ro_scheme_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: bill_case=_consumer_regex(_consumer_all_future_cases)
    script: "../scripts/consumer_bill/build_ro_eligible_fleet.py"


def _consumer_ldes_public_account(wildcards):
    """Item D4(c): the public account whose private LDES floor top-up the forecast bill carries.

    The top-up is built by the finance register from model market revenue
    (rule build_public_ownership_public_account), for the run group whose
    forecast case this is. That chain reads no bill output, so there is no cycle.
    """
    from scripts.public_ownership.reference_runs import consumer_bill_groups, group_forecast_case_id, register_bound_parent

    # Not named `rule`: Snakemake reserves the keyword even inside functions.
    ldes_rule = _consumer_future_assumptions(wildcards.bill_case)["model_linked_rules"].get("ldes_cap_and_floor") or {}
    if not ldes_rule.get("enabled", True):
        return []
    study = _CONSUMER_STUDIES.case_study.get(wildcards.bill_case)
    if study is None:
        po = _consumer_defaults.get("public_ownership") or {}
        # F28 (27 Oct 2026): a 2030 register bound (the legislation-year bound) re-accounts its parent's months.
        groups = ([g for g in consumer_bill_groups(po, _consumer_future_cases)
                   if group_forecast_case_id(po, g) == wildcards.bill_case
                   and register_bound_parent(po["reference_run_groups"][g]) is None]
                  if po.get("reference_run_groups") else [])
        # Item F12 (12 Oct 2026): a price-level case (price_level_parent) re-prices its parent's
        # months. A price-level twin run group projected through the case is its account (the
        # LDES top-up then settles at the case's level); without one it reads its parent's
        # account, the top-up held at the model's price.
        case = _consumer_all_future_cases[wildcards.bill_case]
        parent_case = case.get("price_level_parent")
        if parent_case and not groups:
            parent = _consumer_all_future_cases.get(parent_case)
            if parent is None or parent["source_scenario_pattern"] != case["source_scenario_pattern"]:
                raise ValueError(f"{wildcards.bill_case}: price_level_parent {parent_case!r} must be a forecast case "
                                 "on the same months")
            groups = ([g for g in consumer_bill_groups(po, _consumer_future_cases)
                       if group_forecast_case_id(po, g) == parent_case
                       and register_bound_parent(po["reference_run_groups"][g]) is None]
                      if po.get("reference_run_groups") else [])
    else:
        # D31: a study case's groups are its study's.  A register bound (task Y14)
        # re-accounts its parent's months and bill, so the parent's account is the one.
        from scripts.public_ownership.reference_runs import register_bound_parent

        po = _CONSUMER_STUDIES.configs[study]["public_ownership"]
        own = set(_CONSUMER_STUDIES.groups(study))
        study_cases = _CONSUMER_STUDIES.configs[study]["consumer_bill"]["forecasts"]["cases"]
        groups = [g for g in consumer_bill_groups(po, study_cases)
                  if g in own and group_forecast_case_id(po, g) == wildcards.bill_case
                  and register_bound_parent(po["reference_run_groups"][g]) is None]
        # FES 2050 decision 70: a central-dispatch case no run group projects through (FES25_HT_2050_cd, private
        # only) holds its market case's account: FTI's storage make-whole keeps storage at its self-dispatch
        # earnings, so the floor top-up is the market year's. Every other case, and every 2030 case, is as before.
        case = _consumer_all_future_cases[wildcards.bill_case]
        if not groups and case.get("institution") == "central_dispatch" and case.get("institution_parent"):
            groups = [g for g in consumer_bill_groups(po, study_cases)
                      if g in own and group_forecast_case_id(po, g) == case["institution_parent"]
                      and register_bound_parent(po["reference_run_groups"][g]) is None]
    if len(groups) != 1:
        raise ValueError(
            f"{wildcards.bill_case}: the LDES cap-and-floor line needs exactly one public-ownership run group "
            f"projected through this forecast case, found {groups}; disable model_linked_rules.ldes_cap_and_floor "
            "or configure the group"
        )
    return [f"{resources_path}/public_ownership/{groups[0]}_public_account_summary.json"]


def _consumer_register_line_accounts(bill_case):
    """Item F12 (12 Oct 2026): ``{line: account}`` for a price-level case's register-booked lines ({} otherwise).

    A line booked from a run group's public account (``sourced_externals.<line>.derivation.
    cost_from.public_account``, the battery top-up) reads that group's price-level twin at the
    case's level when one is configured, so the top-up is sized on the register settled at
    the bill's price. Without a twin, or at the model's level, the line reads the account
    its assumptions name (held at the model's price, and the summary says so).
    """
    case = _consumer_all_future_cases[bill_case]
    level = str(case.get("price_level", "model"))
    # Network bounds (18-19 Oct 2026): a case on another network names the run group whose account
    # its register-booked lines read (register_account_group: that network's market group), at any level.
    target = case.get("register_account_group")
    if (level == "model" and not target) or bill_case in _CONSUMER_STUDIES.case_study:
        return {}
    from scripts.public_ownership.reference_runs import group_forecast_case_id, price_level_twin_groups

    po = _consumer_defaults.get("public_ownership") or {}
    if target and target not in (po.get("reference_run_groups") or {}):
        raise ValueError(f"{bill_case}: register_account_group {target!r} is not a public_ownership run group")
    twins = price_level_twin_groups(po) if po.get("reference_run_groups") else {}
    out = {}
    for key, line in (_consumer_future_assumptions(bill_case).get("sourced_externals") or {}).items():
        account = (((line or {}).get("derivation") or {}).get("cost_from") or {}).get("public_account")
        named = re.search(r"([^/\\]+)_public_account_summary\.json$", str(account or ""))
        if not named:
            continue
        group = str(target or named.group(1))
        if level == "model":
            if group != named.group(1):
                out[key] = f"{resources_path}/public_ownership/{group}_public_account_summary.json"
            continue
        at_level = [twin for twin, parent in twins.items() if parent == group
                    and str((_consumer_all_future_cases.get(group_forecast_case_id(po, twin)) or {})
                            .get("price_level", "model")) == level]
        if len(at_level) == 1:
            out[key] = f"{resources_path}/public_ownership/{at_level[0]}_public_account_summary.json"
        elif group != named.group(1):
            out[key] = f"{resources_path}/public_ownership/{group}_public_account_summary.json"
    return out


rule project_future_consumer_bill:
    input:
        # Item D1: the electrolysers' Stage 1 draw is part of the metered-supply
        # denominators; the saved wholesale network carries no link flows.
        wholesale_links=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale_links.csv" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        # Item D4(c): the LDES floor top-up the forecast bill carries.
        ldes_public_account=_consumer_ldes_public_account,
        # Item F12: a price-level case's register-booked lines from its level's twin account ([] otherwise).
        register_line_accounts=lambda wc: sorted(set(_consumer_register_line_accounts(wc.bill_case).values())),
        assumptions=lambda wc: _consumer_all_future_cases[wc.bill_case]["assumptions"],
        ro_scheme_costs=f"{resources_path}/analysis/{{bill_case}}_ro_scheme_summary.json",
        # Change triggers only (the case arrives as params.case): a study case (D31) is
        # triggered by its study's resolved config instead. scenarios.yaml is ancient()
        # because the projector never reads it and a study's finance settings now live in
        # it, so an edit there must not rerun every forecast bill (a study case's own
        # changes reach it through the resolved config above, defaults' through config/defaults/).
        defaults=lambda wc: (DEFAULTS_FILES if wc.bill_case not in _CONSUMER_STUDIES.case_study else
                             f"{resources_path}/public_ownership/{_CONSUMER_STUDIES.case_study[wc.bill_case]}_finance_config.yaml"),
        scenarios=[ancient(path) for path in SCENARIO_FILES],
        strike_table=lambda wc: _consumer_future_assumptions(wc.bill_case)["model_linked_rules"]["cfd"]["strike_table"],
        # Task SETTLE2050: the 2050 contract book and the support settlement's DPA total (none in 2030).
        support_inputs=lambda wc: _consumer_bill_support_inputs(_consumer_future_assumptions(wc.bill_case)),
        domestic_profile=lambda wc: _consumer_future_assumptions(wc.bill_case)["model_linked_rules"]["direct_fuel"]["domestic_shaping"]["profile_file"],
        # F18 (27 Oct 2026): the peak block leaves out England and Wales bank holidays, and the
        # constraint line's split is set beside NESO's thermal constraint cost for the case's network.
        bank_holidays="data/market/consumer_bill_sources/england_wales_bank_holidays.csv",
        neso_benchmarks="data/CP30/validation_references/neso_balancing_benchmarks.csv",
        recovery_demand=lambda wc: _consumer_all_future_cases[wc.bill_case]["recovery_demand_assumptions"],
        annual_summary=lambda wc: f"{resources_path}/analysis/{_consumer_all_future_cases[wc.bill_case]['annual_aggregate_study']}_annual_summary.json",
        balancing_costs=f"{resources_path}/analysis/{{bill_case}}_future_balancing_costs_summary.json",
        networks=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale.nc" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        # The CfD line on metered output: the physical months (a central-dispatch case's I3 year, decision 1).
        physical_networks=lambda wc: [f"{resources_path}/market/{scenario}_balancing.nc" for scenario in _consumer_all_future_cases[wc.bill_case]["physical_scenarios"]],
        turn_down=lambda wc: turn_down_month_inputs(_consumer_all_future_cases[wc.bill_case]["physical_scenarios"]),
        prices=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale_price.csv" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
        summaries=lambda wc: [f"{resources_path}/analysis/{scenario}_market_summary.json" for scenario in _consumer_all_future_cases[wc.bill_case]["source_scenarios"]],
    output:
        components=f"{resources_path}/analysis/{{bill_case}}_consumer_bill_future_components.csv",
        summary=f"{resources_path}/analysis/{{bill_case}}_consumer_bill_future_summary.json",
        report=f"{resources_path}/analysis/{{bill_case}}_consumer_bill_future.html",
        hourly=f"{resources_path}/analysis/{{bill_case}}_consumer_bill_future_hourly_inputs.csv",
    params:
        case=lambda wc: _consumer_all_future_cases[wc.bill_case],
        register_line_accounts=lambda wc: _consumer_register_line_accounts(wc.bill_case)
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: bill_case=_consumer_regex(_consumer_all_future_cases)
    script: "../scripts/consumer_bill/project_future_bill.py"


rule consumer_bill_future_reports:
    input:
        expand(f"{resources_path}/analysis/{{bill_case}}_consumer_bill_future.html", bill_case=list(_consumer_future_cases))


# Decision 12 (task SETTLE2050, from the reachability audit): the composed heat-pump plus EV
# household's time-of-use cost-to-serve beside its single-rate bill, for every forecast case
# whose bill composes archetypes (household_archetypes.report_composition: the 2050 bills).
def _consumer_composite_archetypes(bill_case):
    block = _consumer_future_assumptions(bill_case).get("household_archetypes") or {}
    return [a["id"] for a in (block.get("archetypes") or []) if isinstance(a, dict) and a.get("added")]


_consumer_composite_cases = [c for c in _consumer_all_future_cases
                             if (_consumer_future_assumptions(c).get("household_archetypes") or {}).get("report_composition")
                             and _consumer_composite_archetypes(c)]


rule build_archetype_cost_to_serve_composite:
    """Decision 12: what the composed households' load shapes cost at the year's hourly prices (diagnostic).

    ``archetype_cost_to_serve.py --composite`` prices each composed household part by
    part (base on the domestic profile, heat pump on its temperature shape with the
    scaled COP, EV on its charging windows) on the forecast bill's hourly inputs and
    the weather year's cutout, beside the single-rate bill (the headline). A 2050-only
    target: no 2030 bill composes archetypes.
    """
    input:
        hourly=f"{resources_path}/analysis/{{bill_case}}_consumer_bill_future_hourly_inputs.csv",
        summary=f"{resources_path}/analysis/{{bill_case}}_consumer_bill_future_summary.json",
        cutout=lambda wc: f"{resources_path}/atlite/cutouts/uk-{_consumer_all_future_cases[wc.bill_case].get('weather_year', 2013)}.nc",
        script="scripts/consumer_bill/archetype_cost_to_serve.py"
    output:
        f"{resources_path}/analysis/{{bill_case}}_archetype_cost_to_serve_composite.json"
    params:
        archetypes=lambda wc: " ".join(_consumer_composite_archetypes(wc.bill_case))
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: bill_case=_consumer_regex(_consumer_composite_cases)
    shell:
        "python -m scripts.consumer_bill.archetype_cost_to_serve {input.hourly} {input.cutout} {input.summary} {output}"
        " --composite {params.archetypes}"


rule align_historical_procurement:
    input:
        script="scripts/consumer_bill/align_historical_procurement.py",
        manifest="data/market/consumer_bill_sources/procurement_alignment.json",
        new_sources=expand("data/market/consumer_bill_sources/raw/procurement_alignment/{period}_annex_2.xlsx", period=["2024_Q1", "2024_Q3", "2024_Q4"]),
        q2="data/market/consumer_bill_sources/raw/2024/2024_Q2_annex_2_wholesale.xlsx",
        old_sources=expand("data/market/ofgem_price_cap/2020_source/{name}", name=[
            "2019_10_to_2020_03_annex_2_wholesale_v1.4.xlsx",
            "2020_04_to_2020_09_annex_2_wholesale_v1.5.xlsx",
            "2020_10_to_2020_03_annex_2_wholesale_v1.6.xlsx"]),
        old_inputs="data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
        new_inputs="data/market/ofgem_price_cap/2024_annual/hedged_wholesale_inputs.csv",
        # F18 (27 Oct 2026): England and Wales bank holidays are off the peak block.
        bank_holidays="data/market/consumer_bill_sources/england_wales_bank_holidays.csv",
        parent="data/market/ofgem_price_cap/2020_historical_annual/wholesale_components.csv",
        old_spot="resources/analysis/Validation_2020_pypsa_spot_procurement_periods.csv",
        new_spot="resources/analysis/Historical_2024_ETYS_14zone_market_pypsa_spot_procurement_periods.csv",
    output:
        periods="resources/analysis/consumer_bill_evidence/aligned_procurement_periods.csv",
        summary="resources/analysis/consumer_bill_evidence/aligned_procurement_summary.json",
        report="resources/analysis/consumer_bill_evidence/aligned_procurement_report.md",
    conda: "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.consumer_bill.align_historical_procurement"


rule audit_consumer_bill_methodology:
    input:
        retail_stack_csv=f"{resources_path}/analysis/{{scenario}}_ofgem_benchmark_component_register.csv",
        market_summary_json=f"{resources_path}/analysis/{{scenario}}_market_summary.json",
        policy_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "policy_components.csv"),
    output:
        audit_csv=f"{resources_path}/analysis/{{scenario}}_consumer_bill_methodology_audit.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_consumer_bill_methodology_audit_summary.json",
        report_html=f"{resources_path}/analysis/{{scenario}}_consumer_bill_methodology_audit.html",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_METHODOLOGY_AUDIT_REGEX
    script: "../scripts/market/audit_consumer_bill_methodology.py"


rule reconcile_bsuos_redispatch:
    input:
        network_components_csv=lambda wc: _consumer_snapshot_file(wc.scenario, "network_components.csv"),
        wholesale_impact_csv=f"{resources_path}/analysis/{{scenario}}_wholesale_energy_impact.csv",
        market_summary_json=f"{resources_path}/analysis/{{scenario}}_market_summary.json",
    output:
        reconciliation_csv=f"{resources_path}/analysis/{{scenario}}_bsuos_redispatch_reconciliation.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_bsuos_redispatch_reconciliation_summary.json",
    conda: "../envs/pypsa-gb.yaml"
    wildcard_constraints: scenario=CONSUMER_BILL_METHODOLOGY_AUDIT_REGEX
    script: "../scripts/market/reconcile_bsuos_redispatch.py"
