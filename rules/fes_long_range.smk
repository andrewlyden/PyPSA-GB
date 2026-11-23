"""
The FES long-range chain: FES 2025 inputs, the declared station schedule and the post-solve
validation, for any pathway and year (FES25_HT_2050 was the first study).

Every output here is under resources/FES/ with the prefix fes2025_, or under
resources/analysis/ keyed by the study. No rule here is a target of `all`, and none feeds a
CP30 rule: the FES long-range scenario entries (HT50_* today) request these files.

  build_fes2025_energy_targets        ES1 supply, ED1 demand (fiscal -> calendar), Dem_BB
                                      components, WS1/FLX1 hydrogen for one pathway and year
  extract_fes2025_commodity_prices    CP1/CP2 base case in GBP real 2024 (standard layout)
  build_fes2025_future_costs_source   one year's prices as a marginal_costs.future_costs
                                      source_file, for a FES long-range scenario's future-cost case:

      marginal_costs:
        future_costs:
          enabled: true
          case: fes2025_ht
          carbon_price_support_end: "2028-04-01"
          cases:
            fes2025_ht:
              source_file: "resources/FES/fes2025_future_costs_2050.json"
              fallback_case: central     # biomass, waste, hydrogen, VOM: FES publishes none

  build_fes_station_schedule          the declared Direct stations of one pathway and year
                                      (data/generators/fes/station_list.csv, fes_supply.thermal)
"""

import re

from scripts.FES.fes2025_targets import FES2025_PATHWAYS
from scripts.sources.registry import pathway_by_code, pathway_slug

FES2025_WORKBOOK_PATH = f"{data_path}/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx"
FES2025_PATHWAY_REGEX = "|".join(FES2025_PATHWAYS)
FES_STATION_LIST = f"{data_path}/generators/fes/station_list.csv"


def fes_targets_path(scenario_config, kind="targets.json"):
    """A FES long-range scenario's own targets file: its edition, pathway and modelled year.

    ``kind`` is ``targets.json``, ``targets.csv`` or ``es1_lines.csv`` (rule
    build_fes2025_energy_targets). Only FES 2025 has the targets rule today.
    """
    targets = scenario_config["targets"]
    if targets["source"] != "FES" or int(targets["edition"]) != 2025:
        raise ValueError(f"{scenario_config.get('_scenario_id')}: the FES targets rule covers FES 2025 only, "
                         f"not {targets}")
    slug = pathway_slug(pathway_by_code("FES", 2025, targets["pathway"]).name)
    return f"{resources_path}/FES/fes2025_{slug}_{int(scenario_config['modelled_year'])}_{kind}"


rule build_fes2025_energy_targets:
    """FES 2025 energy targets of one pathway and year (the long-range counterpart of the CP30 energy targets)."""
    input:
        workbook=FES2025_WORKBOOK_PATH,
        logic=["scripts/FES/fes2025_targets.py", "scripts/FES/build_fes_inputs.py"],
    output:
        json=f"{resources_path}/FES/fes2025_{{pathway_slug}}_{{year}}_targets.json",
        csv=f"{resources_path}/FES/fes2025_{{pathway_slug}}_{{year}}_targets.csv",
        es1_lines=f"{resources_path}/FES/fes2025_{{pathway_slug}}_{{year}}_es1_lines.csv",
    params:
        pathway=lambda wc: FES2025_PATHWAYS[wc.pathway_slug],
        # Declared bases: ED1 converted to calendar years (Y1); the demand target is ED1 system
        # demand excluding electrolysis, including losses (Y6a). The consumer total is in the JSON beside it.
        year_basis="calendar",
        # Y6a (30 Sep): the HT50 loads are loss-inclusive like CP30 2030 ("GB Total System
        # Demand - Minus electrolysis"); the bill recovery file divides by the meter factor.
        demand_basis="system",
    wildcard_constraints:
        pathway_slug=FES2025_PATHWAY_REGEX,
        year="20[2-5][0-9]",
    log:
        "logs/FES/build_fes2025_energy_targets_{pathway_slug}_{year}.log",
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/build_fes_inputs.py"


rule extract_fes2025_commodity_prices:
    """FES 2025 CP1/CP2 base case, every year, in GBP real 2024 (fails rather than falls back)."""
    input:
        workbook=FES2025_WORKBOOK_PATH,
        logic=["scripts/FES/fes_commodity_prices.py", "scripts/FES/build_fes_inputs.py"],
    output:
        fuel_prices=f"{resources_path}/FES/fes2025_fuel_prices_gbp2024.csv",
        carbon_prices=f"{resources_path}/FES/fes2025_carbon_prices_gbp2024.csv",
    params:
        price_case="Base Case",
    log:
        "logs/FES/extract_fes2025_commodity_prices.log",
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/build_fes_inputs.py"


rule build_fes2025_future_costs_source:
    """One year's FES 2025 prices as a marginal_costs.future_costs source file (FES long-range scenarios)."""
    input:
        workbook=FES2025_WORKBOOK_PATH,
        logic=["scripts/FES/fes_commodity_prices.py", "scripts/FES/build_fes_inputs.py"],
    output:
        json=f"{resources_path}/FES/fes2025_future_costs_{{year}}.json",
    params:
        price_case="Base Case",
    wildcard_constraints:
        year="20[2-5][0-9]",
    log:
        "logs/FES/build_fes2025_future_costs_source_{year}.log",
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/build_fes_inputs.py"


rule build_fes_station_schedule:
    """The declared Direct stations of one FES 2025 pathway and year (stage 5: data, filtered by year).

    The stations open in the year (data/generators/fes/station_list.csv) take the FES Direct
    capacity of nuclear, gas CCS (at the year's ES1 CCS share of Gen_BB009), BECCS (its ES1
    capacity) and hydrogen power in each licence area; every total is conserved. A scenario
    names the output in fes_supply.thermal.station_schedule.
    """
    input:
        station_list=FES_STATION_LIST,
        fes_data=f"{resources_path}/FES/FES_2025_data.csv",
        es1_lines=f"{resources_path}/FES/fes2025_{{pathway_slug}}_{{year}}_es1_lines.csv",
        logic=["scripts/generators/build_fes_station_schedule.py", "scripts/generators/fes_supply.py"],
    output:
        schedule=f"{resources_path}/FES/fes2025_{{pathway_slug}}_{{year}}_station_schedule.csv",
    params:
        pathway=lambda wc: FES2025_PATHWAYS[wc.pathway_slug],
        pathway_code=lambda wc: pathway_by_code_name(wc.pathway_slug),
    wildcard_constraints:
        pathway_slug=FES2025_PATHWAY_REGEX,
        year="20[2-5][0-9]",
    log:
        "logs/FES/build_fes_station_schedule_{pathway_slug}_{year}.log",
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/generators/build_fes_station_schedule.py"


def pathway_by_code_name(slug):
    """The FES 2025 short code (HT, EE, ...) of a pathway slug."""
    from scripts.sources.registry import pathway_code
    return pathway_code("FES", 2025, FES2025_PATHWAYS[slug])


# ── Y8 post-solve validation of a solved FES long-range market year ──────────────────────
#
# Read-only on the solved months: no rule here solves anything, none is a target of `all`,
# and none is an input of a CP30 rule. A study is a `monthly_aggregates` entry (a study's study.yaml)
# (config/config.yaml) whose twelve months are FES long-range market scenarios (targets
# source FES, method gsp_block, market enabled) with their own Stage 1 (no wholesale_from):
# FES25_HT_2050 today. Run every such study with the target rule `fes_long_range_validation`.
#
#   validate_fes_dispatch           ES1 energy reconciliation (FES 2025, the study's calendar year)
#   diagnose_fes_ic1                IC1 interconnector diagnostics, a winter and a summer month
#   measure_fes_bm_storage_churn    BM simultaneous charge and discharge by storage carrier

FES_IC1_MONTHS = ["Jan", "Jul"]


def _gate_window(year):
    from scripts.market.bm_storage_churn import gate_window
    return gate_window(year)


def _is_fes_validation_study(study):
    """A twelve-month study whose months are FES long-range market scenarios with their own Stage 1."""
    months = (CP30_MONTHLY_AGGREGATES.get(study) or {}).get("scenarios") or []
    configs = [scenarios.get(m) for m in months]
    if len(months) != 12 or any(c is None for c in configs):
        return False
    return all(c["targets"]["source"] == "FES" and c["method"] == "gsp_block"
               and (c.get("market") or {}).get("enabled", False) and not c.get("wholesale_from")
               for c in configs)


FES_VALIDATION_STUDIES = sorted(s for s in CP30_MONTHLY_AGGREGATES if _is_fes_validation_study(s))
FES_VALIDATION_STUDY_REGEX = (
    "|".join(map(re.escape, FES_VALIDATION_STUDIES)) if FES_VALIDATION_STUDIES else r"(?!x)x"
)


def _fes_study_year(study):
    """The modelled year of a validation study (its months agree on it)."""
    years = {int(scenarios[m]["modelled_year"]) for m in _cp30_monthly_aggregate_config(study)["scenarios"]}
    if len(years) != 1:
        raise ValueError(f"{study}: its months model different years {sorted(years)}")
    return years.pop()


_FES_MONTH_RE = re.compile(r"_(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?=_|$)")


def _fes_month_of(scenario_id):
    return _FES_MONTH_RE.search(scenario_id).group(1)


def _fes_month_ids(study, months=None):
    """The study's month scenarios (its monthly_aggregates entry), or those of the given months."""
    scenario_ids = list(_cp30_monthly_aggregate_config(study)["scenarios"])
    if months is None:
        return scenario_ids
    by_month = {_fes_month_of(s): s for s in scenario_ids}
    return [by_month[m] for m in months]


def _fes_month_files(suffix, months=None):
    def _files(wildcards):
        return [f"{resources_path}/market/{s}_{suffix}" for s in _fes_month_ids(wildcards.study, months)]
    return _files


rule validate_fes_dispatch:
    """Final (BM) dispatch of a FES long-range market year against FES 2025 ES1; every WARN explained."""
    input:
        balancing=_fes_month_files("balancing.nc"),
        wholesale=_fes_month_files("wholesale.nc"),
        wholesale_links=_fes_month_files("wholesale_links.csv"),
        wholesale_price=_fes_month_files("wholesale_price.csv"),
        workbook=FES2025_WORKBOOK_PATH,
        logic=["scripts/FES/fes_dispatch_validation.py", "scripts/FES/fes2025_targets.py",
               "scripts/CP30/energy_validation.py"],
    output:
        csv=f"{resources_path}/analysis/{{study}}_fes_energy_validation.csv",
        json=f"{resources_path}/analysis/{{study}}_fes_energy_validation.json",
    params:
        scenario_config=lambda wc: scenario_param(scenarios[_cp30_monthly_aggregate_config(wc.study)["scenarios"][0]]),
    wildcard_constraints:
        study=FES_VALIDATION_STUDY_REGEX,
    log:
        "logs/FES/validate_fes_dispatch_{study}.log",
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/FES/fes_dispatch_validation.py"


rule diagnose_fes_ic1:
    """IC1 diagnostics (flows, hours at capacity, spreads, trade value) for a winter and a summer month."""
    input:
        balancing=_fes_month_files("balancing.nc", FES_IC1_MONTHS),
        wholesale_links=_fes_month_files("wholesale_links.csv", FES_IC1_MONTHS),
        wholesale_price=_fes_month_files("wholesale_price.csv", FES_IC1_MONTHS),
        logic=["scripts/market/ic1_month_diagnostics.py", "scripts/interconnectors/market_coupling.py"],
    output:
        summary=f"{resources_path}/analysis/{{study}}_ic1_diagnostics_summary.json",
        cables=f"{resources_path}/analysis/{{study}}_ic1_diagnostics_cables.csv",
        groups=f"{resources_path}/analysis/{{study}}_ic1_diagnostics_groups.csv",
        spreads=f"{resources_path}/analysis/{{study}}_ic1_diagnostics_spreads.csv",
        tranches=f"{resources_path}/analysis/{{study}}_ic1_diagnostics_tranches.csv",
    params:
        study=lambda wc: wc.study,
        months=FES_IC1_MONTHS,
        scenarios=lambda wc: _fes_month_ids(wc.study, FES_IC1_MONTHS),
    wildcard_constraints:
        study=FES_VALIDATION_STUDY_REGEX,
    log:
        "logs/FES/diagnose_fes_ic1_{study}.log",
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/market/ic1_month_diagnostics.py"


rule measure_fes_bm_storage_churn:
    """Simultaneous charge and discharge in the BM, by storage carrier and month, with the gate week."""
    input:
        networks=_fes_month_files("balancing.nc"),
        logic=["scripts/market/bm_storage_churn.py", "scripts/market/diagnose_storage_degeneracy.py"],
    output:
        csv=f"{resources_path}/analysis/{{study}}_bm_storage_churn.csv",
        json=f"{resources_path}/analysis/{{study}}_bm_storage_churn.json",
    params:
        months=lambda wc: [_fes_month_of(s) for s in _fes_month_ids(wc.study)],
        # The gate week of the study's year (10-16 January, the gate-week scenarios' solve_period).
        window=lambda wc: list(_gate_window(_fes_study_year(wc.study))),
    wildcard_constraints:
        study=FES_VALIDATION_STUDY_REGEX,
    log:
        "logs/FES/measure_fes_bm_storage_churn_{study}.log",
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/market/bm_storage_churn.py"


rule fes_long_range_validation:
    """The Y8 validation outputs of every FES long-range market year (not part of `all`)."""
    input:
        expand(f"{resources_path}/analysis/{{study}}_fes_energy_validation.csv", study=FES_VALIDATION_STUDIES),
        expand(f"{resources_path}/analysis/{{study}}_ic1_diagnostics_summary.json", study=FES_VALIDATION_STUDIES),
        expand(f"{resources_path}/analysis/{{study}}_bm_storage_churn.csv", study=FES_VALIDATION_STUDIES),


# ── Decision 70: the strike diagnostic on the central-dispatch year ─────────────────────
#
# The post-AR7 break-even strikes re-derived on the central year's output beside the market
# year's (scripts/public_ownership/central_strike_diagnostic_2050.py). Report only: no rule
# reads its outputs, and it is not a target of `all`. The post-solve pipeline runs it as its
# step 5 from the same command; standalone:
#
#   snakemake resources/public_ownership/fes_ht_<year>_central_dispatch_strike_diagnostic.csv \
#       --rerun-triggers mtime --allowed-rules build_fes_year_central_dispatch_strike_diagnostic
#
# (--allowed-rules keeps a stale solved month from pulling a re-solve.) User, 14 Nov 2026: one rule for every
# long-range year, 2038 to 2050; it replaced the 2050-only rule (identical inputs and command for 2050).
FES2050_STRIKE_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fes_year_strike_files(year, suffix, family="market"):
    return [f"{resources_path}/market/FES25_HT_{year}_{m}_{family}{suffix}" for m in FES2050_STRIKE_MONTHS]


rule build_fes_year_central_dispatch_strike_diagnostic:
    """Decision 70 on any long-range year: post-AR7 strikes on the central-dispatch year beside the market year's (uncompensated curtailment only)."""
    input:
        market_stage1=lambda wc: _fes_year_strike_files(wc.year, "_wholesale_dispatch.csv"),
        market_bm=lambda wc: _fes_year_strike_files(wc.year, "_balancing_dispatch.csv"),
        central_bm=lambda wc: _fes_year_strike_files(wc.year, "_balancing_dispatch.csv", family="central_dispatch"),
        prices=lambda wc: _fes_year_strike_files(wc.year, "_wholesale_price.csv"),
        networks=lambda wc: _fes_year_strike_files(wc.year, "_wholesale.nc"),
        audits=lambda wc: _fes_year_strike_files(wc.year, "_contract_value_bids.csv"),
        book=f"{resources_path}/public_ownership/fes_ht_{{year}}_support_contract_book.csv",
        calibration="data/public_ownership/financing/strike_calibration_{year}.csv",
        register=f"{resources_path}/public_ownership/fes_ht_{{year}}_asset_finance_register.parquet",
        bill_summary=f"{resources_path}/analysis/FES25_HT_{{year}}_consumer_bill_future_summary.json",
        logic=["scripts/public_ownership/central_strike_diagnostic_2050.py",
               "scripts/public_ownership/vintage_capital_2050.py"],
    output:
        csv=f"{resources_path}/public_ownership/fes_ht_{{year}}_central_dispatch_strike_diagnostic.csv",
        json=f"{resources_path}/public_ownership/fes_ht_{{year}}_central_dispatch_strike_diagnostic.json",
    wildcard_constraints:
        year=r"20(3[89]|4[0-9]|50)",
    log:
        "logs/FES/build_fes_year_central_dispatch_strike_diagnostic_{year}.log",
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.public_ownership.vintage_capital_2050 calibrate --dispatch-basis central"
        " --book {input.book} --register {input.register} --market-calibration {input.calibration} --study FES25_HT_{wildcards.year}"
        " --central-suffix _central_dispatch --bill-summary {input.bill_summary}"
        " --out {output.csv} --json-out {output.json} > {log} 2>&1"


# ── The 2050 results notebook (plan Part 5) ──────────────────────────────────────────────
#
# Not a target of `all`, and its only output is the notebook, so no 2030 rule reads anything
# it writes. It reads the final 2050 outputs only and quotes no number of its own; run it after
# the decision-49 re-solve and the 2050 finance chain:
#
#   snakemake build_fes2050_results_notebook --rerun-triggers mtime \
#       --allowed-rules build_fes2050_results_notebook
#
# (--allowed-rules stops a stale upstream file from pulling solves or 2030 jobs into the run;
# drop it only when the chain is meant to rebuild.) The 2030 comparison files and the per-month
# BM tables are read when present but are not declared inputs: the 2030 snapshot belongs to the
# 2030 session, and the solved months are behind the ledger already.
FES2050_PO = f"{resources_path}/public_ownership/fes_ht_2050"
FES2050_AN = f"{resources_path}/analysis/FES25_HT_2050"


rule build_fes2050_results_notebook:
    """The FES 2050 results notebook (bill by case, three parties, 2030 vs 2050, pathway, BM by month, sensitivities)."""
    input:
        script="scripts/analysis/build_fes2050_results_notebook.py",
        by_case=f"{FES2050_PO}_consumer_bill_by_ownership_case.csv",
        by_case_summary=f"{FES2050_PO}_consumer_bill_by_ownership_case_summary.json",
        state_bids=f"{FES2050_PO}_consumer_bill_balancing_cases.csv",
        state_bids_summary=f"{FES2050_PO}_consumer_bill_balancing_cases_summary.json",
        tariff=f"{FES2050_PO}_nationalised_bill_components.csv",
        institution=f"{FES2050_PO}_nationalised_bill_by_institution.csv",
        institution_summary=f"{FES2050_PO}_nationalised_bill_by_institution.json",
        ledger=f"{FES2050_PO}_exchequer_ledger.csv",
        pathway=f"{resources_path}/public_ownership/pathway_2030_2050.csv",
        pathway_summary=f"{resources_path}/public_ownership/pathway_2030_2050_summary.json",
        private_lines=f"{FES2050_AN}_consumer_bill_future_components.csv",
        bm_monthly=f"{FES2050_AN}_future_balancing_costs_monthly.csv",
        validation=f"{FES2050_AN}_fes_energy_validation.csv",
        validation_summary=f"{FES2050_AN}_fes_energy_validation.json",
        ic1=f"{FES2050_AN}_ic1_diagnostics_summary.json",
        churn=f"{FES2050_AN}_bm_storage_churn.json",
    output:
        "notebooks/fes_ht_2050_results.ipynb"
    log:
        "logs/FES/build_fes2050_results_notebook.log",
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "(python {input.script} && jupyter nbconvert --to notebook --execute --inplace"
        " --ExecutePreprocessor.timeout=1800 {output}) > {log} 2>&1"
