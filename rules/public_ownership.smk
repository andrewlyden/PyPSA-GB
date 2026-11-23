"""Public-ownership finance layer over completed market runs.

Every rule reads finished market artifacts (``*_market_summary.json`` and
``*_balancing.nc``) and never alters dispatch.  Configuration lives in
``config/defaults/`` under ``public_ownership``; scenario expansion, the
case list and the decision-register checks are shared with ``run_financing.py``
through ``scripts/public_ownership/reference_runs.py``.  A later study (FES 2050,
decision 31) declares its own groups and forecast cases on its parent scenario entry
in ``config/scenarios.yaml``; a group declared there resolves as defaults with that
entry's blocks deep-merged over it (``scripts/public_ownership/study_config.py``),
and its scripts read the resolved ``{study}_finance_config.yaml`` as ``--config``.

Item K1: three scenarios (``private``, ``public_control``, ``public_ownership``)
with sensitivities underneath.  ``reference_runs.expanded_cases`` is the one
place the case list is built; every post-solve stage loops over it inside its
script (the stages share the settled year and the register) and ``rule all``
asks for every case's bill: the market bill for cases that keep the market,
the acquisition tariff (item N5) for cases that acquire the legacy fleet, and
the institution comparison (item N10) for the central-dispatch case once its
second solved year is in ``run_scenarios``.

``enabled: true`` adds a reference-run group to ``rule all`` only when every
scenario in that group is in the active run selection, so enabling the layer
never forces an extra market solve.  Any group can still be requested
explicitly, e.g.::

    snakemake resources/public_ownership/historical_2024_consumer_bill_financing_bridge.csv

provided its market outputs and those of its ``baseline_run_group`` exist.
"""

import json as _po_json
import re as _po_re

from scripts.public_ownership.build_asset_finance_register import (
    existing_storage_site_flags as _po_existing_storage_site_flags,
)
from scripts.public_ownership.model_year import group_model_year as _po_group_model_year

from scripts.public_ownership.reference_runs import (
    baseline_scenarios as _po_baseline_scenarios,
    consumer_bill_groups as _po_consumer_bill_groups,
    expanded_cases as _po_expanded_cases,
    institution_cases as _po_institution_cases,
    institution_scenarios as _po_institution_scenarios,
    institution_block as _po_institution_block,
    market_bill_cases as _po_market_bill_cases,
    reference_run_scenarios as _po_reference_run_scenarios,
    required_scenarios as _po_required_scenarios,
    check_balancing_group_scenarios as _po_check_balancing_group,
    group_balancing_bids as _po_group_balancing_bids,
    group_bid_case as _po_group_bid_case,
    institution_market_twin as _po_institution_market_twin,
    group_excluded_cases as _po_group_excluded_cases,
    group_forecast_case_id as _po_group_forecast_case_id,
    group_network_case as _po_group_network_case,
    purchase_price_group as _po_purchase_price_group,
    group_market_cases as _po_group_market_cases,
    group_tariff_cases as _po_group_tariff_cases,
    is_tariff_balancing_group as _po_is_tariff_balancing_group,
    institution_twin_central_scenarios as _po_institution_twin_central_scenarios,
    institution_twin_groups as _po_institution_twin_groups,
    price_level_twin_groups as _po_price_level_twin_groups,
    tariff_cases as _po_tariff_cases,
    validate_public_ownership_config as _po_validate,
    vintaged_groups as _po_vintaged_groups,
    register_bound_groups as _po_register_bound_groups,
)
from scripts.public_ownership.model_year import is_vintaged_model_year as _po_is_vintaged_year

# F20 phase 0 (13 Oct 2026): the folder this layer writes and reads its own outputs in
# ({po_resources_path}/public_ownership).  It is resources_path unless a parity run sets
# --config po_resources_path=<root> (scripts/public_ownership/f20_parity/README.md), so a
# second tree can run the layer into its own root while reading every solved input from
# resources_path; without the key every path below is the one it always was.
# 14 Oct 2026: earlier rule files reset resources_path to "resources" after the Snakefile read the
# override, so it is read again here (the layer's input functions resolve it at call time).
resources_path = str(config.get("resources_path") or "resources")
po_resources_path = str(config.get("po_resources_path") or resources_path)


def _po_parity_path(path):
    """A literal ``resources/...`` path named in data, redirected for a parity run (unchanged otherwise)."""
    from scripts.public_ownership.f20_parity.run_parity import parity_path

    return parity_path(path, resources_path, po_resources_path)


PUBLIC_OWNERSHIP_CONFIG = _full_config["defaults"].get("public_ownership") or {}
PUBLIC_OWNERSHIP_ENABLED = bool(PUBLIC_OWNERSHIP_CONFIG.get("enabled", False))
# The forecast cases were expanded (case_id, source_scenarios) by consumer_bill.smk.
_PO_FORECAST_CASES = _full_config["defaults"].get("consumer_bill", {}).get("forecasts", {}).get("cases", {})

# Decision 31 (task D31): a later study's run groups and forecast cases live on its
# parent scenario entry in config/scenarios.yaml (FES25_HT_2050), and a group declared
# there resolves as the defaults (config/defaults/) with the study's blocks deep-merged over it
# (scripts/public_ownership/study_config.py; consumer_bill.smk, included first, loads
# the studies, validates each resolved config and expands their forecast cases). Every
# group declared in defaults resolves to PUBLIC_OWNERSHIP_CONFIG itself, so each 2030
# list, input, param and command below is unchanged.
if "_CONSUMER_STUDIES" in globals():
    _PO_STUDIES = _CONSUMER_STUDIES
else:
    from scripts.public_ownership.study_config import load_studies as _po_load_studies

    _PO_STUDIES = _po_load_studies("config", _full_config["defaults"])
_PO_STUDY_GROUPS = _PO_STUDIES.groups() if PUBLIC_OWNERSHIP_CONFIG else []
# Every forecast case a group can be projected through: defaults' and the studies'.
_PO_ALL_FORECAST_CASES = {**_PO_FORECAST_CASES,
                          **{case: _PO_STUDIES.case(case) for case in _PO_STUDIES.cases()}}


def _po_cfg(run_group):
    """The ``public_ownership`` block ``run_group`` resolves to: its study's, else defaults' (D31)."""
    if _PO_STUDIES.study_of(run_group) is None:
        return PUBLIC_OWNERSHIP_CONFIG
    return _PO_STUDIES.public_ownership(run_group)


def _po_cfg_arg(wildcards):
    """The config a group's scripts read (``--config``): the config directory (``read_config`` loads its
    defaults), or its study's resolved file (D31)."""
    study = _PO_STUDIES.study_of(wildcards.run_group)
    if study is None:
        return "config"
    return f"{po_resources_path}/public_ownership/{study}_finance_config.yaml"


def _po_cfg_inputs(wildcards):
    """The files behind ``_po_cfg_arg`` (rule inputs): every defaults file, or the study's resolved file."""
    if _PO_STUDIES.study_of(wildcards.run_group) is None:
        return list(DEFAULTS_FILES)
    return [_po_cfg_arg(wildcards)]


def _po_get(key, default=None):
    """An input or param reading ``public_ownership.<key>`` from the group's resolved block (D31)."""
    return lambda wc: _po_cfg(wc.run_group).get(key, default)


if PUBLIC_OWNERSHIP_CONFIG:
    _po_validate(PUBLIC_OWNERSHIP_CONFIG, _PO_FORECAST_CASES)
    PUBLIC_OWNERSHIP_GROUPS = PUBLIC_OWNERSHIP_CONFIG["reference_run_groups"]
    if _PO_STUDY_GROUPS:
        PUBLIC_OWNERSHIP_GROUPS = {**PUBLIC_OWNERSHIP_GROUPS, **{
            g: _po_cfg(g)["reference_run_groups"][g] for g in _PO_STUDY_GROUPS}}
    _PO_CONSUMER_BILL = PUBLIC_OWNERSHIP_CONFIG.get("consumer_bill") or {}
    _PO_BILL_GROUPS = _po_consumer_bill_groups(PUBLIC_OWNERSHIP_CONFIG, _PO_FORECAST_CASES) if _PO_CONSUMER_BILL else []
    _PO_BILL_GROUPS += [g for g in _PO_STUDY_GROUPS if _po_cfg(g).get("consumer_bill")
                        and g in _po_consumer_bill_groups(_po_cfg(g), _PO_STUDIES.forecast_cases(g))]
    # Item K1: the one case list, and its three destinations.
    PUBLIC_OWNERSHIP_CASES = _po_expanded_cases(PUBLIC_OWNERSHIP_CONFIG)
    _PO_MARKET_CASES = _po_market_bill_cases(PUBLIC_OWNERSHIP_CONFIG)
    _PO_TARIFF_CASES = _po_tariff_cases(PUBLIC_OWNERSHIP_CONFIG)
    _PO_INSTITUTION_CASES = _po_institution_cases(PUBLIC_OWNERSHIP_CONFIG)
    _PO_INSTITUTION_SCENARIOS = _po_institution_scenarios(PUBLIC_OWNERSHIP_CONFIG)
else:
    PUBLIC_OWNERSHIP_GROUPS = {}
    _PO_CONSUMER_BILL = {}
    _PO_BILL_GROUPS = []
    PUBLIC_OWNERSHIP_CASES, _PO_MARKET_CASES, _PO_TARIFF_CASES, _PO_INSTITUTION_CASES = [], [], [], []
    _PO_INSTITUTION_SCENARIOS = []
_PO_INSTITUTION = PUBLIC_OWNERSHIP_CONFIG.get("institution_comparison") or {}


# Task Y14 (patch 5): the tariff and institution lists, scenarios and settings of one
# group.  A group with its own case list (the 2050 groups) reads its own; every 2030
# group has none, so each value equals the global one above.
import functools as _po_functools


@_po_functools.lru_cache(maxsize=None)
def _po_group_lists(group):
    if not PUBLIC_OWNERSHIP_CONFIG:
        return (), (), (), {}
    cfg = _po_cfg(group)
    return (tuple(_po_tariff_cases(cfg, group)),
            tuple(_po_institution_cases(cfg, group)),
            tuple(_po_institution_scenarios(cfg, group)),
            _po_institution_block(cfg, group))


def _po_group_tariff_cases(group):
    return list(_po_group_lists(group)[0])


def _po_group_institution_cases(group):
    return list(_po_group_lists(group)[1])


def _po_group_institution_scenarios(group):
    return list(_po_group_lists(group)[2])


def _po_group_institution(group):
    return _po_group_lists(group)[3]


def _po_group_fe_ratio_flag(group):
    """Decision 18: the group's own ``institution_comparison.forecast_error_charge.fleet_ratio`` ('' without one)."""
    own = ((PUBLIC_OWNERSHIP_GROUPS.get(group) or {}).get("institution_comparison") or {}).get("forecast_error_charge") or {}
    ratio = own.get("fleet_ratio")
    return f" --forecast-error-fleet-ratio {float(ratio)}" if ratio is not None and own.get("enabled", True) else ""


def _po_group_runnable(group):
    """Whether every scenario of ``group`` is in the active run selection (``reference_runs.runnable_groups``).

    An institution twin (P1/P2 under central dispatch, 20 Oct 2026) also needs its parent's
    central-dispatch months, which its settlement, register and bill read.
    """
    own = set(_po_reference_run_scenarios(PUBLIC_OWNERSHIP_GROUPS[group]))
    if group in _PO_INSTITUTION_TWINS:
        own |= set(_po_institution_twin_central_scenarios(PUBLIC_OWNERSHIP_CONFIG, group))
    return own <= set(run_ids)


def _po_group_institution_ready(group):
    # Item F12: a price-level twin carries no institution comparison (the central year is solved at its own price).
    if group in _po_price_level_twin_groups(PUBLIC_OWNERSHIP_CONFIG or {}):
        return False
    return bool(_po_group_institution_cases(group)) and set(_po_group_institution_scenarios(group)) <= set(run_ids)

if PUBLIC_OWNERSHIP_ENABLED:
    _po_missing = {
        group: [s for s in _po_required_scenarios(_po_cfg(group), group) if s not in scenarios]
        for group in PUBLIC_OWNERSHIP_GROUPS
    }
    _po_missing = {group: names for group, names in _po_missing.items() if names}
    # D31: a study may declare a group before its months exist (the 2050 P2 share
    # groups); its rules cannot run until they do, so it is reported, not refused.
    # A group whose missing scenarios are all defined by studies config.yaml does not load is inactive, not an
    # error (the public config loads no study); a scenario defined nowhere is still refused.
    _po_unloaded = set()
    if _po_missing:
        from config_loader import read_scenario_sources as _po_read_scenario_sources
        _po_studies_dir = Path(workflow.basedir) / "config" / "studies"
        _po_all_studies = sorted(p.name for p in _po_studies_dir.iterdir() if p.is_dir() and p.name != "archive")
        _po_study_ids = set(_po_read_scenario_sources(Path(workflow.basedir) / "config",
                                                      {"studies": _po_all_studies}).entries)
        _po_unloaded = {group for group, names in _po_missing.items() if set(names) <= _po_study_ids}
    _po_pending = {group: names for group, names in _po_missing.items()
                   if group in _PO_STUDY_GROUPS or group in _po_unloaded}
    _po_missing = {group: names for group, names in _po_missing.items()
                   if group not in _PO_STUDY_GROUPS and group not in _po_unloaded}
    if _po_missing:
        raise ValueError(
            "public_ownership.reference_run_groups reference scenarios that are not defined "
            f"in config/scenarios.yaml: {_po_missing}"
        )
    # Groups of studies config.yaml does not load are simply inactive and say nothing (a new user's run printed a
    # screen of them, 19 Nov 2026); only a loaded study's group still waiting for its months is reported.
    _po_report = {group: names for group, names in _po_pending.items() if group not in _po_unloaded}
    if _po_report:
        print("public_ownership: study groups whose months are not yet in config/scenarios.yaml (their rules "
              "cannot run until they are): "
              + ", ".join(f"{g} ({len(names)}, e.g. {names[0]})" for g, names in _po_report.items()))

# Item BM1: a group's months must bid on the shares of the cases it projects, and a
# contract-value group's months on none.  Checked at parse time, on defined scenarios.
for _po_bill_group in _PO_BILL_GROUPS:
    if all(s in scenarios for s in _po_reference_run_scenarios(PUBLIC_OWNERSHIP_GROUPS[_po_bill_group])):
        _po_check_balancing_group(_po_cfg(_po_bill_group), _po_bill_group, scenarios)
# The acquisition tariff, the institution comparison and the workbook feedback read
# the contract-value group only.
_PO_MARKET_BID_GROUPS = [g for g in _PO_BILL_GROUPS
                         if _po_group_balancing_bids(PUBLIC_OWNERSHIP_GROUPS[g]) == "contract_value"]

# Item BM1, P3: a tariff balancing group names the acquisition-tariff case its
# months bid on (bid_case); its bill is that case's tariff, merged apart from the
# market-bill balancing groups (P2), whose merge and ledger inputs stay as they were.
_PO_TARIFF_BALANCING_GROUPS = [g for g in _PO_BILL_GROUPS if g not in _PO_MARKET_BID_GROUPS
                               and _po_is_tariff_balancing_group(_po_cfg(g), g)]
_PO_BALANCING_GROUPS = [g for g in _PO_BILL_GROUPS if g not in _PO_MARKET_BID_GROUPS
                        and g not in _PO_TARIFF_BALANCING_GROUPS]
PUBLIC_OWNERSHIP_BALANCING_GROUP_REGEX = "|".join(map(_po_re.escape, _PO_BALANCING_GROUPS)) or r"(?!x)x"

# Item F12 at the ownership layer (12 Oct 2026): a price-level twin
# (reference_run_groups.<twin>.price_level_of: <group>) re-accounts its parent's solved
# months at another wholesale price level (its forecast case's price_level, CP.32), so
# every output of the layer exists at both levels under the two groups' own names. A
# twin takes its parent's role at its own level: it is measured against the market
# group's twin (_po_market_bid_group_for), and it shares the parent's solve-side tables
# (state shares, commissioning years, their check and the per-unit bid decomposition,
# all solve quantities with no price in them: _po_solved_group). No 2030 group is a twin
# until the config declares one, so every list, regex and input below is then unchanged.
_PO_PRICE_LEVEL_TWINS = _po_price_level_twin_groups(PUBLIC_OWNERSHIP_CONFIG) if PUBLIC_OWNERSHIP_CONFIG else {}
PUBLIC_OWNERSHIP_PRICE_LEVEL_TWIN_REGEX = "|".join(map(_po_re.escape, _PO_PRICE_LEVEL_TWINS)) or r"(?!x)x"
# P1/P2 under central dispatch (user decisions of 14 and 20 Oct 2026): an institution twin
# (reference_run_groups.<twin>.institution_of: <market group>) re-projects its parent's market-bill
# cases on the parent's central-dispatch year (I3, FTI's settlement): its own asset settlement
# (build_public_ownership_institution_twin_settlement), register, account and bills under its own
# name, its register's dispatch and its ledger's carbon on the central months. It takes no market
# group's role (no balancing group is measured against it) and owns no solve-side table (it reads
# its parent's: _po_solved_group). merge_public_ownership_institution_groups sets it beside its parent.
_PO_INSTITUTION_TWINS = _po_institution_twin_groups(PUBLIC_OWNERSHIP_CONFIG) if PUBLIC_OWNERSHIP_CONFIG else {}
PUBLIC_OWNERSHIP_INSTITUTION_TWIN_REGEX = "|".join(map(_po_re.escape, _PO_INSTITUTION_TWINS)) or r"(?!x)x"


def _po_solved_group(group):
    """The group whose solve-side tables ``group`` reads: its price-level or institution parent, else itself."""
    return _PO_PRICE_LEVEL_TWINS.get(group, _PO_INSTITUTION_TWINS.get(group, group))


def _po_institution_twin_of(market_group):
    """The institution twin of a market group, or None."""
    twins = [twin for twin, parent in _PO_INSTITUTION_TWINS.items() if parent == market_group]
    return twins[0] if twins else None


def _po_mapped_prices(group):
    """A price-level twin's mapped Stage 1 GB price files, one per month in month order ([] for every other group)."""
    if group not in _PO_PRICE_LEVEL_TWINS:
        return []
    return [f"{po_resources_path}/public_ownership/{group}_mapped_prices/{month}_wholesale_price.csv"
            for month in _PO_MONTHS]


_PO_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# The per-unit decomposition runs on every balancing group, P2 and P3 alike (a twin reads its parent's).
PUBLIC_OWNERSHIP_ANY_BALANCING_GROUP_REGEX = (
    "|".join(map(_po_re.escape, [g for g in _PO_BALANCING_GROUPS + _PO_TARIFF_BALANCING_GROUPS
                                 if g not in _PO_PRICE_LEVEL_TWINS])) or r"(?!x)x")
PUBLIC_OWNERSHIP_MARKET_BID_GROUP_REGEX = "|".join(map(_po_re.escape, _PO_MARKET_BID_GROUPS)) or r"(?!x)x"
# The solve-side tables (state shares, commissioning years, their check, the auction
# calibration check) belong to the groups the solves bid on: never a price-level twin.
# An institution twin owns no solve-side table either (it re-projects its parent's cases, 20 Oct 2026).
PUBLIC_OWNERSHIP_SOLVED_GROUP_REGEX = "|".join(
    map(_po_re.escape, [g for g in PUBLIC_OWNERSHIP_GROUPS if g not in _PO_PRICE_LEVEL_TWINS
                        and g not in _PO_INSTITUTION_TWINS])) or r"(?!x)x"
PUBLIC_OWNERSHIP_SOLVED_MARKET_BID_GROUP_REGEX = "|".join(
    map(_po_re.escape, [g for g in _PO_MARKET_BID_GROUPS if g not in _PO_PRICE_LEVEL_TWINS
                        and g not in _PO_INSTITUTION_TWINS])) or r"(?!x)x"

PUBLIC_OWNERSHIP_GROUP_REGEX = "|".join(map(_po_re.escape, PUBLIC_OWNERSHIP_GROUPS)) or r"(?!x)x"
# FES 2050 plan, task Y10: groups whose registers the stock-flow vintages (a
# fleet_vintages block, model year 2040 on).  None in the 2030 config, so the
# unvintaged regex is the full group regex there, character for character.
_PO_VINTAGED_GROUPS = _po_vintaged_groups(PUBLIC_OWNERSHIP_CONFIG) if PUBLIC_OWNERSHIP_CONFIG else []
_PO_VINTAGED_GROUPS += [g for g in _PO_STUDY_GROUPS if g in _po_vintaged_groups(_po_cfg(g))]
PUBLIC_OWNERSHIP_UNVINTAGED_GROUP_REGEX = "|".join(
    map(_po_re.escape, [g for g in PUBLIC_OWNERSHIP_GROUPS if g not in _PO_VINTAGED_GROUPS])) or r"(?!x)x"
PUBLIC_OWNERSHIP_VINTAGED_GROUP_REGEX = "|".join(map(_po_re.escape, _PO_VINTAGED_GROUPS)) or r"(?!x)x"
# Task Y14: no-re-solve register bounds of a vintaged group (learning rate, retirement rule).
_PO_REGISTER_BOUND_GROUPS = _po_register_bound_groups(PUBLIC_OWNERSHIP_CONFIG) if PUBLIC_OWNERSHIP_CONFIG else []
_PO_REGISTER_BOUND_GROUPS += [g for g in _PO_STUDY_GROUPS if g in _po_register_bound_groups(_po_cfg(g))]
# Task Y12: groups after the 2030 acquisition carry its annuity (acquisition_carry_forward.py);
# a group names the group it was bought in with acquisition_from_group (default cp30_2030_cp30_network).
_PO_ACQUISITION_YEAR = 2030
_PO_CARRIED_GROUPS = [g for g, grp in PUBLIC_OWNERSHIP_GROUPS.items()
                      if _po_group_model_year(grp) > _PO_ACQUISITION_YEAR]
PUBLIC_OWNERSHIP_CARRIED_GROUP_REGEX = "|".join(map(_po_re.escape, _PO_CARRIED_GROUPS)) or r"(?!x)x"


def _po_carried_acquisition(wildcards):
    """Task Y12: ``{group}_carried_acquisition.json`` for a group after the purchase; nothing for a 2030 group."""
    if wildcards.run_group not in _PO_CARRIED_GROUPS:
        return []
    return [f"{po_resources_path}/public_ownership/{wildcards.run_group}_carried_acquisition.json"]


def _po_fleet_continuity(wildcards):
    """User, 14 Nov 2026: the later year's fleet-continuity check (verify_fleet_continuity); nothing for 2030."""
    if wildcards.run_group not in _PO_CARRIED_GROUPS:
        return []
    return [f"{po_resources_path}/public_ownership/{wildcards.run_group}_fleet_continuity.json"]


def _po_network_year_tables(wildcards):
    """Task Y12: the network account's model-year data (RAV roll-forward, pass-through), vintaged years only."""
    if wildcards.run_group not in PUBLIC_OWNERSHIP_GROUPS:
        return []
    year = _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[wildcards.run_group])
    if not _po_is_vintaged_year(year):
        return []
    # D31: the files the group's own network_account.model_years.<year> block names.
    from scripts.public_ownership.network_finance_account import year_override

    block = year_override(_po_cfg(wildcards.run_group).get("network_account"), year)
    return [block[key] for key in ("roll_forward_inputs", "guarantee_pass_through_table") if block.get(key)]


# D31: a study's resolved config, the file its groups' scripts read as --config.
PUBLIC_OWNERSHIP_STUDY_REGEX = "|".join(
    map(_po_re.escape, sorted({_PO_STUDIES.study_of(g) for g in _PO_STUDY_GROUPS}))) or r"(?!x)x"


def _po_study_scenario_files(study):
    """The scenario files that can define ``study``: config/scenarios.yaml and the study folder whose entries name it.

    Falls back to every scenario file when no single folder declares the id (a flat study file, say).
    """
    import yaml as _yaml

    folders = sorted({str(Path(p).parent) for p in SCENARIO_FILES if "/studies/" in str(p)})
    for folder in folders:
        ids = set()
        for name in ("scenarios.yaml", "families.yaml"):
            path = Path(folder) / name
            if path.exists():
                ids |= set((_yaml.safe_load(path.read_text(encoding="utf-8")) or {}).keys())
        if study in ids:
            return ["config/scenarios.yaml"] + [p for p in SCENARIO_FILES if str(Path(p).parent) == folder]
    return list(SCENARIO_FILES)


rule resolve_public_ownership_study_config:
    """Decision 31 (task D31): the defaults with a study's finance blocks deep-merged over it.

    The study is the parent scenario entry (``FES25_HT_2050``) whose ``public_ownership``
    and ``consumer_bill.forecasts`` blocks declare its run groups and forecast cases;
    the file has the defaults' layout, so a study group's scripts read it where a
    2030 group's read the config directory (``--config config``).
    """
    input:
        defaults=DEFAULTS_FILES,
        # 11 Nov 2026: the study's own folder only (plus the canonical scenarios), so adding or editing another
        # study (fes2040) does not re-resolve this one and re-run its whole bill layer.
        scenarios=lambda wc: _po_study_scenario_files(wc.study),
        script="scripts/public_ownership/study_config.py"
    output:
        config=f"{po_resources_path}/public_ownership/{{study}}_finance_config.yaml"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        study=PUBLIC_OWNERSHIP_STUDY_REGEX
    shell:
        "python scripts/public_ownership/study_config.py --study {wildcards.study} --out {output.config}"


PUBLIC_OWNERSHIP_BILL_GROUP_REGEX = "|".join(map(_po_re.escape, _PO_BILL_GROUPS)) or r"(?!x)x"
_PO_DEFAULT_SNAPSHOT = f"{po_resources_path}/public_ownership/model_assumptions.yaml"
_PO_SNAPSHOT = PUBLIC_OWNERSHIP_CONFIG.get("assumptions_snapshot", _PO_DEFAULT_SNAPSHOT)
_PO_NETWORK_BLOCKS = f"{resources_path}/analysis/network_price_control_building_blocks.json"
# Review 29 Sep 2026: the evidence tables the register, the network account and the
# ledger read (data, declared so a data edit reruns only what reads it) are read per
# group from its resolved block (D31): _po_group_register_tables, _po_group_network_tables,
# and _po_get for household_tax_incidence and contested_fiscal_costs (F29 retired guarantee_cost).
_PO_LINE_SOURCES = "data/market/consumer_bill_benchmarks/bill_line_sources.csv"
_PO_HPC_SCHEDULE = "data/public_ownership/financing/hpc_schedule_edf_2025.json"
_PO_NETWORK_SOURCES = "data/market/network_cost_sources/price_controls"


def _po_auction_margin_bound(run_group):
    """FES 2050 decision 71: a register bound's own auction-margin block, or None (every other group, all of 2030)."""
    return (PUBLIC_OWNERSHIP_GROUPS.get(run_group) or {}).get("auction_margin") or None


def _po_bound_own_lines(run_group):
    """Bill lines a register bound's own inputs set ([] for every other group, all of 2030).

    The LDES floor is read from the group's own public account (its register sizes the top-up), and a margin
    bound (decision 71) settles the CfD line at its own strikes; the private case need not reproduce the parent's
    forecast bill on these.
    """
    if not (PUBLIC_OWNERSHIP_GROUPS.get(run_group) or {}).get("register_bound_of"):
        return []
    return ["ldes_cap_and_floor"] + (["cfd"] if _po_auction_margin_bound(run_group) else [])


def _po_auction_margin_files(run_group):
    """Decision 71: the strike table and bill assumptions a margin bound settles on (its own, written by a rule)."""
    base = f"{po_resources_path}/public_ownership/{run_group}_auction_margin"
    return {"strike_table": f"{base}_strike_table.csv", "assumptions": f"{base}_bill_assumptions.json"}


def _po_parent_forecast_case(run_group):
    return _PO_ALL_FORECAST_CASES[_po_group_forecast_case_id(_po_cfg(run_group), run_group)]


def _po_forecast_case(wildcards):
    """The consumer_bill_forecasts case a group's per-case bill is projected through (its own, else the link's).

    Decision 71: a margin bound projects through its parent's case with its own bill assumptions (its
    strike table, contract book and DPA settlement); every other group's case is returned unchanged.
    """
    case = _po_parent_forecast_case(wildcards.run_group)
    if _po_auction_margin_bound(wildcards.run_group):
        case = {**case, "assumptions": _po_auction_margin_files(wildcards.run_group)["assumptions"]}
    return case


def _po_forecast_assumptions(wildcards):
    if _po_auction_margin_bound(wildcards.run_group):
        # Decision 71: built in memory from the parent's file, so the DAG resolves before the rule writes it.
        from scripts.public_ownership.vintage_capital_2050 import bound_assumptions as _po_bound_assumptions

        group = PUBLIC_OWNERSHIP_GROUPS[wildcards.run_group]
        with open(_po_parent_forecast_case(wildcards.run_group)["assumptions"], encoding="utf-8") as handle:
            return _po_bound_assumptions(_po_json.load(handle), parent=group["register_bound_of"],
                                         group=wildcards.run_group,
                                         strike_table=_po_auction_margin_files(wildcards.run_group)["strike_table"])
    with open(_po_forecast_case(wildcards)["assumptions"], encoding="utf-8") as handle:
        return _po_json.load(handle)


def _po_bill_support_inputs(wildcards):
    """Task SETTLE2050: the 2050 contract book and DPA settlement the group's bill names ([] for 2030).

    F20 parity (14 Oct 2026): the assumptions name these as literal ``resources/...`` paths (the 2030
    battery top-up reads ``resources/public_ownership/cp30_2030_cp30_network_public_account_summary.json``), so a
    parity run redirects them to its own root and the input tree; unchanged without one.
    """
    return [_po_parity_path(f) for f in _consumer_bill_support_inputs(_po_forecast_assumptions(wildcards))]


def _po_bill_contract_book(wildcards):
    """Task SETTLE2050: the contract book the group's bill settles ([] for 2030)."""
    return [f for f in _po_bill_support_inputs(wildcards) if f.endswith("_support_contract_book.csv")]


def _po_cm_rebasing(wildcards):
    """``from to factor`` of the bill's currency rebasing, which carries the 2025 CM rates onto the register's year."""
    rebasing = _po_forecast_assumptions(wildcards)["currency_rebasing"]
    return " ".join(str(rebasing[k]) for k in ("from_price_year", "to_price_year", "factor"))


# D31: settings read from a group's resolved block.  For every 2030 group the block is
# PUBLIC_OWNERSHIP_CONFIG itself, so each returns what the module-level reads did.
def _po_group_repricing(group):
    return (_po_cfg(group).get("consumer_bill") or {}).get("cfd_repricing") or {}


def _po_group_award_years(group):
    return _po_group_repricing(group).get("award_years", "data/public_ownership/financing/cfd_round_award_years.csv")


def _po_group_register_tables(group):
    cfg = _po_cfg(group)
    calibration = cfg.get("cost_calibration") or {}
    return [path for path in (calibration.get("table") if calibration.get("enabled") else None,
                              (cfg.get("acquisition") or {}).get("hpc_completion")) if path]


def _po_group_network_tables(group):
    account = _po_cfg(group).get("network_account") or {}
    return [path for path in (account.get("acquisition_premium_table"),
                              account.get("guarantee_pass_through_table")) if path]


def _po_group_ro_register(group):
    return ((_po_cfg(group).get("acquisition") or {}).get("valuation") or {}).get(
        "ro_register", "data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv")


def _po_cm_rates(wildcards):
    """The Capacity Market rates per kW of the group's model year: 2030's, or a vintaged year's own (Y11)."""
    year = _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[wildcards.run_group])
    suffix = f"_{year}" if _po_is_vintaged_year(year) else ""
    return f"{resources_path}/market/capacity_market/cm_revenue_by_technology{suffix}.csv"


def _po_cm_auction(wildcards):
    """F9: the 2030/31 T-4 the account re-clears per case (the CM builder's outputs); none for a vintaged year."""
    year = _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[wildcards.run_group])
    if _po_is_vintaged_year(year):
        return []
    return [f"{resources_path}/market/capacity_market/{name}" for name in
            ("cm_auction_model.json", "cm_t4_2030_31_supply.csv", "cm_calendar_2030_summary.json")]


def _po_cm_auction_flag(wildcards, input):
    return (" --cm-auction " + " ".join(str(p) for p in input.cm_auction)) if input.cm_auction else ""


def _po_register_flags(storage_sites=True, group=None):
    """The register's label, re-pricing and storage-site flags.

    ``storage_sites=False`` (the vintaged register, task E2E): the existing-site split
    is CP30's (its LAES carrier absorbs existing pumped hydro); a vintaged build carries
    pumped hydro as its own carrier, so the split does not apply.  ``group`` (D31) reads
    that group's resolved block; None reads defaults'.
    """
    cfg = _po_cfg(group)
    bill = cfg.get("consumer_bill") or {}
    repricing = bill.get("cfd_repricing") or {}
    flags = []
    overrides = (bill.get("label_corrections") or {}).get("support_type") or {}
    if overrides:
        flags.append("--support-type-override " + " ".join(f"{k}={v}" for k, v in overrides.items()))
    if repricing:
        flags.append(f"--award-years {_po_group_award_years(group)}")
        flags.append(f"--eligible-award-years-from {repricing.get('eligible_award_years_from', 2025)}")
    # Existing plant a model carrier absorbs (CP30's LAES includes existing pumped hydro).
    sites = _po_existing_storage_site_flags(cfg.get("existing_storage_sites")) if storage_sites else ""
    if sites:
        flags.append(sites)
    # F28: the legislation year (a group's own legislated_year, else the block's; the bound groups set 2025).
    legislation = cfg.get("legislation") or {}
    year = (PUBLIC_OWNERSHIP_GROUPS.get(group) or {}).get("legislated_year", legislation.get("year")) if group else         legislation.get("year")
    if year is not None:
        flags.append(f"--legislated-year {int(year)} --transfer-year {int(legislation.get('transfer_year', 2030))}"
                     f" --snapshot-year {int(legislation.get('snapshot_year', 2025))}"
                     f" --lead-times {legislation.get('lead_times', 'data/public_ownership/financing/legislation_lead_times.csv')}")
        # F28 (d): dated plant reaching the end of its life is rebuilt (false: today's private life extension).
        if legislation.get("rebuild_at_end_of_life", True):
            flags.append("--rebuild-lives data/public_ownership/financing/technology_costs.csv")
    # F28: small-scale plant no case owns (the ES1 lines file is the rule's small_scale_lines input).
    small = cfg.get("small_scale_exclusion") or {}
    if small.get("data_sources"):
        flags.append("--small-scale-data-source " + " ".join(str(s) for s in small["data_sources"]))
    if small.get("es1_lines"):
        flags.append(f"--micro-capacity-lines {small['es1_lines']}")
    return " ".join(flags)


def _po_small_scale_lines(wildcards):
    """F28: the group's ES1 lines table, whose micro capacity is small-scale plant ([] without one)."""
    lines = (_po_cfg(wildcards.run_group).get("small_scale_exclusion") or {}).get("es1_lines")
    return [lines] if lines else []


def _public_ownership_group(wildcards):
    return PUBLIC_OWNERSHIP_GROUPS[wildcards.run_group]


def _public_ownership_scenarios(wildcards):
    return _po_reference_run_scenarios(_public_ownership_group(wildcards))


def _public_ownership_summary_inputs(wildcards):
    return expand(
        f"{resources_path}/analysis/{{scenario}}_market_summary.json",
        scenario=_public_ownership_scenarios(wildcards),
    )


def _public_ownership_network_inputs(wildcards):
    # An institution twin's register takes its dispatch from its parent's central-dispatch months (20 Oct 2026).
    scenarios = (_po_institution_twin_central_scenarios(PUBLIC_OWNERSHIP_CONFIG, wildcards.run_group)
                 if wildcards.run_group in _PO_INSTITUTION_TWINS else _public_ownership_scenarios(wildcards))
    return expand(
        f"{resources_path}/market/{{scenario}}_balancing.nc",
        scenario=scenarios,
    )


def _po_physical_scenarios(wildcards):
    """The months a group's bill meters the CfD line on (and whose dispatch its ledger measures).

    Its forecast case's ``physical_scenarios`` (rules/consumer_bill.smk): the case's own months,
    the central-dispatch months for a central-dispatch case (P1/P2 under I3, 20 Oct 2026).
    """
    case = _po_forecast_case(wildcards)
    return list(case.get("physical_scenarios") or case["source_scenarios"])


def _public_ownership_baseline_network_inputs(wildcards):
    return expand(
        f"{resources_path}/market/{{scenario}}_balancing.nc",
        scenario=_po_baseline_scenarios(_po_cfg(wildcards.run_group), wildcards.run_group),
    )


_PO_WORKBOOK = PUBLIC_OWNERSHIP_CONFIG.get("source_workbook", "data/public_ownership/GB_Power_Model_v6.2.xlsx")
_PO_COMMITTED_SNAPSHOT = PUBLIC_OWNERSHIP_CONFIG.get(
    "committed_snapshot", "data/public_ownership/financing/workbook_v6_2_assumptions.yaml")


def _po_workbook_present() -> bool:
    """The financing workbook is not distributed with the public release; the committed snapshot stands in."""
    return Path(_PO_WORKBOOK).exists()


def _po_snapshot_inputs(wildcards):
    if _po_workbook_present():
        return {
            "workbook": _PO_WORKBOOK,
            "variants": PUBLIC_OWNERSHIP_CONFIG.get(
                "source_variants", "data/public_ownership/financing/workbook_v6_2_variants.csv"),
            "extractor": "scripts/public_ownership/extract_workbook_v5.py",
        }
    return {"committed": _PO_COMMITTED_SNAPSHOT}


def _po_snapshot_command(wildcards, input, output):
    if "workbook" in input.keys():
        return (f"python scripts/public_ownership/extract_workbook_v5.py {input.workbook} {input.variants}"
                f" {output.snapshot}")
    return (f'python -c "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])"'
            f" {input.committed} {output.snapshot}")


rule extract_public_ownership_workbook_assumptions:
    """Copy the financing workbook's cells (v6.2) and its Excel-recalculated alternatives into a YAML snapshot (schema 5).

    F29 (26 Oct 2026); v6.2 since 30 Oct 2026.  The variants CSV is committed data, written by hand on Windows with
    scripts/public_ownership/workbook_variants.py (Excel); the extractor refuses it unless its
    JSON names this workbook's SHA256 and its base variant reproduces the saved values.

    The workbook is not distributed with the public release: without it the rule copies the committed snapshot
    (``committed_snapshot``), which tests/unit/test_public_ownership_finance.py holds equal to a fresh extraction
    wherever the workbook is present.
    """
    input:
        unpack(_po_snapshot_inputs)
    output:
        snapshot=_PO_SNAPSHOT
    params:
        command=_po_snapshot_command
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "{params.command}"


rule aggregate_public_ownership_reference_run:
    """Sum one full-year or twelve monthly market summaries into an annual bundle."""
    input:
        summary_jsons=_public_ownership_summary_inputs
    output:
        monthly_csv=f"{po_resources_path}/public_ownership/{{run_group}}_monthly_market_metrics.csv",
        annual_json=f"{po_resources_path}/public_ownership/{{run_group}}_annual_market_summary.json",
        manifest_json=f"{po_resources_path}/public_ownership/{{run_group}}_source_manifest.json"
    params:
        scenarios=_public_ownership_scenarios,
        run_group=lambda wc: wc.run_group
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_GROUP_REGEX
    script:
        "../scripts/public_ownership/aggregate_reference_runs.py"


rule build_public_ownership_asset_register:
    """Register components, annual dispatch and legacy/new-build cohorts."""
    input:
        networks=_public_ownership_network_inputs,
        baseline_networks=_public_ownership_baseline_network_inputs,
        crosswalk=_po_get("crosswalk", "data/public_ownership/financing/pypsa_finance_technology_crosswalk.csv"),
        award_years=lambda wc: _po_group_award_years(wc.run_group),
        # F28 (d): the economic lives the end-of-life rebuild reads.
        rebuild_lives="data/public_ownership/financing/technology_costs.csv",
        # 15 Nov 2026 (E5): declared per-asset cost technologies (CP30's LAES-labelled pumped storage).
        technology_overrides="data/public_ownership/financing/finance_technology_overrides.csv",
        # F28: the legislation lead times.
        lead_times=lambda wc: (_po_cfg(wc.run_group).get("legislation") or {}).get(
            "lead_times", "data/public_ownership/financing/legislation_lead_times.csv")
    output:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_inputs.csv"
    params:
        model_year=lambda wc: _po_group_model_year(_public_ownership_group(wc)),
        policy_start_year=_po_get("policy_start_year", 2025),
        component_types=lambda wc: " ".join(_po_cfg(wc.run_group).get("ownership_scope", ["Generator", "StorageUnit"])),
        flags=lambda wc: _po_register_flags(group=wc.run_group)
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_UNVINTAGED_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_asset_finance_register.py {input.crosswalk} {output.register} {input.networks}"
        " --model-year {params.model_year} --policy-start-year {params.policy_start_year}"
        " --component-types {params.component_types} --baseline-networks {input.baseline_networks} {params.flags}"
        " --technology-overrides {input.technology_overrides}"


def _po_fleet_vintages(wildcards):
    return PUBLIC_OWNERSHIP_GROUPS[wildcards.run_group].get("fleet_vintages") or {}


def _po_refurbishment_flag(block):
    """F28 (d): a fleet_vintages block's refurbishment rule as ``--refurbishment ...`` ('' without one)."""
    from scripts.public_ownership.fleet_vintages import parse_refurbishment, refurbishment_flag

    return refurbishment_flag(parse_refurbishment((block or {}).get("refurbishment")))


rule build_public_ownership_fleet_vintages:
    """FES 2050 plan, task Y10: the stock-flow vintage mix of a vintaged group's model year.

    FES 2025 ES1 capacity 2024 to the model year by technology, the existing fleet's
    commissioning years (REPD, DUKES), Y2's economic lives and the declared
    retirement rule (``fleet_vintages.RETIREMENT_RULES``).  Only groups with a
    ``fleet_vintages`` block and a model year from 2040 match.
    """
    input:
        workbook=f"{data_path}/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx",
        repd="data/renewables/repd-q2-jul-2025.csv",
        dukes=f"{resources_path}/generators/DUKES/DUKES_2023_generators.csv",
        lives="data/public_ownership/financing/technology_economic_lives.csv",
        closures="data/generators/nuclear_phase_out_dates.csv",
        # FES 2050 decision 62: the REPD records left out of the evidence ([] without one).
        stale_records=lambda wc: _fleet_stale_records_input(_po_fleet_vintages(wc))
    output:
        mix=f"{po_resources_path}/public_ownership/{{run_group}}_fleet_vintages.csv",
        flows=f"{po_resources_path}/public_ownership/{{run_group}}_fleet_vintage_flows.csv",
        retired=f"{po_resources_path}/public_ownership/{{run_group}}_fleet_retirements_by_decade.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_fleet_vintages_summary.json"
    params:
        model_year=lambda wc: _po_group_model_year(_public_ownership_group(wc)),
        pathway=lambda wc: _po_fleet_vintages(wc).get("pathway", "Holistic Transition"),
        rule=lambda wc: _po_fleet_vintages(wc).get("retirement_rule", "economic_life_with_extension"),
        life_source=lambda wc: _po_fleet_vintages(wc).get("life_source", "central"),
        # FES 2050 decision 48: per-technology lives over the source (rules/market.smk helper).
        life_overrides=lambda wc: _life_override_flag(_po_fleet_vintages(wc).get("life_overrides")),
        # FES 2050 decision 62: the repowering rule and evidence exclusions ('' before it).
        decision62=lambda wc: _fleet_decision62_flags(_po_fleet_vintages(wc), labels=False),
        # F28 (d): hydro and pumped hydro refurbished under the strict rule ('' without a rule).
        refurbishment=lambda wc: _po_refurbishment_flag(_po_fleet_vintages(wc))
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_VINTAGED_GROUP_REGEX
    shell:
        "python scripts/public_ownership/fleet_vintages.py --workbook \"{input.workbook}\" --pathway \"{params.pathway}\""
        " --target-year {params.model_year} --rule {params.rule} --life-source {params.life_source}{params.life_overrides}"
        "{params.decision62}{params.refurbishment}"
        " --repd {input.repd} --dukes {input.dukes} --lives {input.lives} --closures {input.closures}"
        " --out-mix {output.mix} --out-flows {output.flows} --out-retirements {output.retired}"
        " --out-summary {output.summary}"


rule build_public_ownership_vintaged_asset_register:
    """Task Y10: the register of a vintaged group, every row given a build year by the stock-flow join.

    The same register as ``build_public_ownership_asset_register`` with the
    stock-flow mix in place of the baseline split (``--fleet-vintages``).
    """
    input:
        networks=_public_ownership_network_inputs,
        crosswalk=_po_get("crosswalk", "data/public_ownership/financing/pypsa_finance_technology_crosswalk.csv"),
        award_years=lambda wc: _po_group_award_years(wc.run_group),
        fleet_vintages=f"{po_resources_path}/public_ownership/{{run_group}}_fleet_vintages.csv",
        # Task Y11's live contracts in the model year (AR7/AR7a phases, HPC, Sizewell C).
        known_contracts="data/market/support/cfd_contracts.csv",
        # Task Y14-prep: 2050 crosswalk rows (CCS gas, SMR, hydrogen turbines, CAES, LAES 25 years).
        crosswalk_2050="data/public_ownership/financing/pypsa_finance_technology_crosswalk_2050.csv",
        # F28: the ES1 lines whose micro capacity is small-scale plant.
        small_scale_lines=_po_small_scale_lines,
        # F28: the legislation lead times.
        lead_times=lambda wc: (_po_cfg(wc.run_group).get("legislation") or {}).get(
            "lead_times", "data/public_ownership/financing/legislation_lead_times.csv")
    output:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_inputs.csv"
    params:
        model_year=lambda wc: _po_group_model_year(_public_ownership_group(wc)),
        policy_start_year=_po_get("policy_start_year", 2025),
        component_types=lambda wc: " ".join(_po_cfg(wc.run_group).get("ownership_scope", ["Generator", "StorageUnit"])),
        flags=lambda wc: _po_register_flags(storage_sites=False, group=wc.run_group),
        join_mode=lambda wc: _po_fleet_vintages(wc).get("join_mode", "pro_rata"),
        life_source=lambda wc: _po_fleet_vintages(wc).get("life_source", "central"),
        life_overrides=lambda wc: _life_override_flag(_po_fleet_vintages(wc).get("life_overrides")),
        site_vintages=lambda wc: (f"--site-vintages {_po_fleet_vintages(wc)['site_vintages']}"
                                  if _po_fleet_vintages(wc).get("site_vintages") else ""),
        # FES 2050 decision 62: the repowering rule (the join) and the no-CfD list ('' before it).
        decision62=lambda wc: _fleet_decision62_flags(_po_fleet_vintages(wc), stock_flow=False)
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_VINTAGED_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_asset_finance_register.py {input.crosswalk} {output.register} {input.networks}"
        " --model-year {params.model_year} --policy-start-year {params.policy_start_year}"
        " --component-types {params.component_types} {params.flags}"
        " --fleet-vintages {input.fleet_vintages} --fleet-vintage-mode {params.join_mode}"
        " --life-source {params.life_source}{params.life_overrides} {params.site_vintages}"
        " --crosswalk-supplement {input.crosswalk_2050}{params.decision62}"


def _po_settlement_input(wildcards):
    """WP-D settlement is built only for groups that share the bill-link forecast case."""
    if wildcards.run_group in _PO_BILL_GROUPS:
        return [f"{po_resources_path}/public_ownership/{wildcards.run_group}_asset_settlement.csv"]
    return []


def _po_settlement_flag(wildcards, input):
    return f"--settlement {input.settlement[0]}" if len(input.settlement) else ""


# Task Y14 (patch 2): the settlement's inputs, shared by the 2030 rule and its vintaged twin.
_PO_SETTLEMENT_INPUTS = {
    "wholesale": lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale.nc" for s in _po_forecast_case(wc)["source_scenarios"]],
    "balancing": lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_forecast_case(wc)["source_scenarios"]],
    "prices": lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_price.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
    "redispatch": lambda wc: [f"{resources_path}/market/{s}_redispatch_summary.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
    # Pro rata zonal turn-down (20 Oct 2026): each month is read on its network's basis.
    "turn_down": lambda wc: turn_down_month_inputs(_po_forecast_case(wc)["source_scenarios"]),
    # Read beside each Stage 1 network by the settlement's baseload reference (item D1); declared for the DAG.
    "wholesale_links": lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_links.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
    # Decides whether the curtailment compensation is already inside the
    # balancing settlement; the bill's constraint line reads the same audit.
    "contract_value_bids": lambda wc: [f"{resources_path}/market/{s}_contract_value_bids.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
    "assumptions": lambda wc: _po_forecast_case(wc)["assumptions"],
    "strike_table": lambda wc: _po_forecast_assumptions(wc)["model_linked_rules"]["cfd"]["strike_table"],
    "ro_scheme_costs": lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['case_id']}_ro_scheme_summary.json",
    # Item B1: fuel and carbon from the build-time resource breakdown, one per build source.
    "marginal_costs": lambda wc: sorted({f"{resources_path}/generators/{_build_source(s)}_marginal_costs_breakdown.csv"
                                      for s in _po_forecast_case(wc)["source_scenarios"]})
}
# Bill groups split by vintaging: the 2030 settlement rule takes the unvintaged ones
# (every 2030 bill group, so its regex matches the same groups), the twin the rest.
# An institution twin's settlement is its own rule's (build_public_ownership_institution_twin_settlement).
PUBLIC_OWNERSHIP_UNVINTAGED_BILL_GROUP_REGEX = "|".join(
    map(_po_re.escape, [g for g in _PO_BILL_GROUPS if g not in _PO_VINTAGED_GROUPS
                        and g not in _PO_PRICE_LEVEL_TWINS and g not in _PO_INSTITUTION_TWINS])) or r"(?!x)x"
PUBLIC_OWNERSHIP_VINTAGED_BILL_GROUP_REGEX = "|".join(
    map(_po_re.escape, [g for g in _PO_BILL_GROUPS if g in _PO_VINTAGED_GROUPS])) or r"(?!x)x"


def _po_hydrogen_fuel_flag(wildcards):
    """``--hydrogen-fuel-price BASIS`` for a vintaged group (group key ``hydrogen_fuel_price``; null books none)."""
    basis = PUBLIC_OWNERSHIP_GROUPS[wildcards.run_group].get("hydrogen_fuel_price", "bus0_marginal_price")
    return f"--hydrogen-fuel-price {basis}" if basis else ""


rule build_public_ownership_asset_settlement:
    """WP-D: market and support settlement per asset over the twelve monthly runs."""
    input:
        **_PO_SETTLEMENT_INPUTS
    output:
        settlement=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement.csv"
    params:
        scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_UNVINTAGED_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_asset_settlement.py {output.settlement} --scenarios {params.scenarios}"
        " --wholesale {input.wholesale} --balancing {input.balancing} --prices {input.prices} --redispatch {input.redispatch}"
        " --contract-value-bids {input.contract_value_bids}"
        " --assumptions {input.assumptions} --strike-table {input.strike_table} --ro-scheme-summary {input.ro_scheme_costs}"
        " --model-price-year {params.model_price_year} --marginal-cost-breakdown {input.marginal_costs}"


rule build_public_ownership_price_level_settlement:
    """F19 P3 decision 2: the asset settlement at CP30's own 2030 price level (CP.32).

    Each market hour's Stage 1 GB price mapped onto CP.32's duration curve
    (``consumer_bill.price_level_mapping``), the month price files written beside it, and every asset
    re-settled on them (Stage 1 positions and CfD top-ups; balancing, fuel and ROCs held).
    Read only by the legacy valuation's post-2030 price path (the financing step and the
    valuation); no bill reads it.
    """
    input:
        **_PO_SETTLEMENT_INPUTS,
        central=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement.csv",
        curve=lambda wc: ((_full_config["defaults"].get("consumer_bill") or {}).get("price_level_mapping") or {}).get(
            "source_file", "data/CP30/cp32_price_duration_curve_2030.csv"),
        defaults=_po_cfg_inputs
    output:
        settlement=f"{po_resources_path}/public_ownership/{{run_group}}_price_level_settlement.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_price_level_settlement_summary.json",
        prices=directory(f"{po_resources_path}/public_ownership/{{run_group}}_price_level_prices")
    params:
        config_arg=_po_cfg_arg,
        scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_UNVINTAGED_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/price_level_settlement.py {input.central} {output.settlement} {output.summary}"
        " --prices-dir {output.prices} --scenarios {params.scenarios}"
        " --wholesale {input.wholesale} --balancing {input.balancing} --prices {input.prices} --redispatch {input.redispatch}"
        " --assumptions {input.assumptions} --strike-table {input.strike_table} --ro-scheme-summary {input.ro_scheme_costs}"
        " --model-price-year {params.model_price_year} --marginal-cost-breakdown {input.marginal_costs}"
        " --config {params.config_arg}"


rule build_public_ownership_price_level_twin_settlement:
    """Item F12 at the ownership layer (12 Oct 2026): a price-level twin's asset settlement.

    The parent group's settlement re-settled at the twin's forecast-case price level,
    the year mapped exactly as that case's bill maps it (the case's mapping block and
    zero band): Stage 1 positions and CfD top-ups on the mapped price, and (item F24.2)
    each CfD unit's contract-value bid inside its balancing settlement, by the change the
    twin bill's extraction puts on the constraint line; the rest of the balancing
    settlement, fuel and carbon, ROCs, the DPA and every volume held as solved. Writes the twelve mapped month
    price files the twin's tariff and ledger value the Stage 1 quantities at.
    """
    input:
        **_PO_SETTLEMENT_INPUTS,
        central=lambda wc: f"{po_resources_path}/public_ownership/{_PO_PRICE_LEVEL_TWINS[wc.run_group]}_asset_settlement.csv",
        curve=lambda wc: (_po_forecast_case(wc).get("price_level_mapping") or {}).get(
            "source_file", "data/CP30/cp32_price_duration_curve_2030.csv"),
        defaults=_po_cfg_inputs
    output:
        settlement=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_price_level_twin_settlement_summary.json",
        prices=expand(f"{po_resources_path}/public_ownership/{{{{run_group}}}}_mapped_prices/{{month}}_wholesale_price.csv",
                      month=_PO_MONTHS)
    params:
        config_arg=_po_cfg_arg,
        scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"],
        case=lambda wc: _po_forecast_case(wc)["case_id"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_PRICE_LEVEL_TWIN_REGEX
    # The bid audits (input.contract_value_bids) are declared but not passed: the script reads each beside its
    # redispatch summary, and listing them puts the command over Windows' 8,191-character limit (24 Oct 2026).
    shell:
        "python scripts/public_ownership/price_level_settlement.py {input.central} {output.settlement} {output.summary}"
        " --price-level-case {params.case} --mapped-prices {output.prices} --scenarios {params.scenarios}"
        " --wholesale {input.wholesale} --balancing {input.balancing} --prices {input.prices} --redispatch {input.redispatch}"
        " --assumptions {input.assumptions} --strike-table {input.strike_table} --ro-scheme-summary {input.ro_scheme_costs}"
        " --model-price-year {params.model_price_year} --marginal-cost-breakdown {input.marginal_costs}"
        " --config {params.config_arg}"


def _po_twin_fti(wildcards):
    """An institution twin's year FTI settlement (its forecast case's ``fti_settlement``, its parent's)."""
    return str(_po_forecast_case(wildcards)["fti_settlement"])


rule build_public_ownership_institution_twin_settlement:
    """P1/P2 under central dispatch (20 Oct 2026): an institution twin's asset settlement on its parent's I3 year.

    The central schedule at the national price (the market twin's copperplate price),
    each unit's FTI transfers (cost recovery, firm access with the lost CfD and ROC
    support, the storage weekly make-whole, its real-time correction at cost) as its
    balancing settlement, CfD and ROC support on metered central output
    (scripts/public_ownership/settlement.py, settle_assets_central_dispatch).
    """
    input:
        # The market months' Stage 1 networks and prices: the national price and the demand weighting.
        wholesale=_PO_SETTLEMENT_INPUTS["wholesale"],
        prices=_PO_SETTLEMENT_INPUTS["prices"],
        wholesale_links=_PO_SETTLEMENT_INPUTS["wholesale_links"],
        central_wholesale=lambda wc: [f"{resources_path}/market/{s}_wholesale.nc" for s in _po_physical_scenarios(wc)],
        central_links=lambda wc: [f"{resources_path}/market/{s}_wholesale_links.csv" for s in _po_physical_scenarios(wc)],
        central_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_physical_scenarios(wc)],
        fti_units=lambda wc: _po_re.sub(r"\.csv$", "_units.csv", _po_twin_fti(wc)),
        assumptions=_PO_SETTLEMENT_INPUTS["assumptions"],
        strike_table=_PO_SETTLEMENT_INPUTS["strike_table"],
        ro_scheme_costs=_PO_SETTLEMENT_INPUTS["ro_scheme_costs"],
        marginal_costs=_PO_SETTLEMENT_INPUTS["marginal_costs"],
    output:
        settlement=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement.csv",
        transfers=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement_transfers.csv"
    params:
        scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        central=lambda wc: " ".join(_po_physical_scenarios(wc)),
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_INSTITUTION_TWIN_REGEX
    shell:
        "python scripts/public_ownership/build_asset_settlement.py {output.settlement} --scenarios {params.scenarios}"
        " --wholesale {input.wholesale} --prices {input.prices} --central-scenarios {params.central}"
        " --central-wholesale {input.central_wholesale} --central-balancing {input.central_balancing}"
        " --fti-units {input.fti_units} --transfers-output {output.transfers}"
        " --assumptions {input.assumptions} --strike-table {input.strike_table} --ro-scheme-summary {input.ro_scheme_costs}"
        " --model-price-year {params.model_price_year} --marginal-cost-breakdown {input.marginal_costs}"


def _po_price_level_settlement(wildcards):
    """F19 P3 decision 2: the CP30-level settlement a 2030 bill group's legacy valuation reads ([] otherwise).

    Item F12: a price-level twin reads its parent's. The post-2030 path is CP30's level
    whatever the 2030 year settled at; only the model year's own cash moves with the twin.
    """
    group = _po_solved_group(wildcards.run_group)
    if group in _PO_BILL_GROUPS and group not in _PO_VINTAGED_GROUPS:
        return [f"{po_resources_path}/public_ownership/{group}_price_level_settlement.csv"]
    return []


def _po_price_level_case_flag(flag):
    """`` FLAG CASE`` when the group's forecast case is off the model's own price level (item F12), else ''."""
    def _flag(wildcards):
        case = _po_forecast_case(wildcards)
        return f" {flag} {case['case_id']}" if str(case.get("price_level", "model")) != "model" else ""
    return _flag


def _po_price_level_value_flag(wildcards):
    """`` --price-level LEVEL`` for a price-level twin's public account (item F12), else ''."""
    level = str(_po_forecast_case(wildcards).get("price_level", "model"))
    return f" --price-level {level}" if level != "model" else ""


def _po_price_level_flag(wildcards, input):
    flag = f" --price-level-settlement {input.price_level_settlement[0]}" if len(input.price_level_settlement) else ""
    # User decision (10 Nov 2026): the anchored price path's table (2030 bill groups).
    anchors = getattr(input, "price_anchors", [])
    return flag + (f" --price-anchors {anchors[0]}" if len(anchors) else "")


#: User decision (10 Nov 2026): the legacy price path's anchors: today's forward level and every solved year
#: (2030 from the report group, 2050 from fes_ht_2050, and any intermediate snapshot year's group).
_PO_PRICE_ANCHORS = f"{po_resources_path}/public_ownership/legacy_price_anchors.csv"
_PO_ANCHOR_CFG = ((PUBLIC_OWNERSHIP_CONFIG.get("acquisition") or {}).get("valuation") or {}).get("price_anchors") or {}
_PO_ANCHOR_GROUPS = {2030: str(PUBLIC_OWNERSHIP_CONFIG.get("report_group") or "cp30_2030_no_acceleration"), 2050: "fes_ht_2050",
                     **{int(y): str(g) for y, g in ((_PO_ANCHOR_CFG.get("solved_groups")) or {}).items()}}


#: User, 11 Nov 2026: under acquisition.valuation.anchor_streams all, every stream per MW by technology from the
#: registers of the solved years whose groups never read the anchors (the vintaged years: the 2030 register reads
#: them, so its own streams come from the register being valued); every technology, so no carrier filter.
_PO_ANCHOR_ALL_STREAMS = str(((PUBLIC_OWNERSHIP_CONFIG.get("acquisition") or {}).get("valuation") or {}).get(
    "anchor_streams", "price")) == "all"
_PO_ANCHOR_REGISTERS = ({y: f"{po_resources_path}/public_ownership/{g}_asset_finance_register.parquet"
                         for y, g in sorted(_PO_ANCHOR_GROUPS.items()) if g in _PO_VINTAGED_GROUPS}
                        if _PO_ANCHOR_ALL_STREAMS else {})


def _po_price_anchors(wildcards):
    """The anchors table a 2030 bill group's legacy valuation reads (the anchored price path); [] otherwise."""
    group = _po_solved_group(wildcards.run_group)
    if group in _PO_BILL_GROUPS and group not in _PO_VINTAGED_GROUPS:
        return [_PO_PRICE_ANCHORS]
    return []


rule build_public_ownership_price_anchors:
    """The legacy price path's anchors (user decision, 10 Nov 2026): each solved year's capture price by carrier
    from its asset settlement, and today's forward level on the earliest year's capture ratios."""
    input:
        settlements=[f"{po_resources_path}/public_ownership/{g}_asset_settlement.csv"
                     for _, g in sorted(_PO_ANCHOR_GROUPS.items())],
        reference_prices=[str((_PO_ANCHOR_CFG.get("today") or {}).get("reference_price_files", "")).format(month=m)
                          for m in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
                         if (_PO_ANCHOR_CFG.get("today") or {}).get("reference_price_files") else [],
        registers=list(_PO_ANCHOR_REGISTERS.values()),
        script="scripts/public_ownership/price_anchors.py",
    output:
        anchors=_PO_PRICE_ANCHORS,
    params:
        settlement_args=" ".join(f"--settlement {y}={po_resources_path}/public_ownership/{g}_asset_settlement.csv"
                                 for y, g in sorted(_PO_ANCHOR_GROUPS.items())),
        today=(f" --today-year {_PO_ANCHOR_CFG['today']['year']}"
               f" --today-gbp-per-mwh {_PO_ANCHOR_CFG['today']['level_gbp_per_mwh_nominal']}"
               f" --today-deflator {_PO_ANCHOR_CFG['today']['deflator_to_2024']}"
               if _PO_ANCHOR_CFG.get("today") else ""),
        carriers=(" --carriers " + " ".join(map(str, _PO_ANCHOR_CFG["carriers"]))
                  if _PO_ANCHOR_CFG.get("carriers") and not _PO_ANCHOR_ALL_STREAMS else ""),
        register_args="".join(f" --register {y}={path}" for y, path in _PO_ANCHOR_REGISTERS.items()),
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} {params.settlement_args}{params.today} --reference-price-files {input.reference_prices}"
        "{params.carriers}{params.register_args} --out {output.anchors}"


# Task SETTLE2050: the run's fuel and carbon prices, the DPA reference CCGT's (group key dpa_future_costs).
_PO_DPA_FUTURE_COSTS = "resources/FES/fes2025_future_costs_2050.json"


def _po_vintage_capital_files(block):
    """FES 2050 decision 41: the data a study's capital basis reads (public_ownership.vintage_capital_2050).

    The capture prices the merchant tail earns and the break-even calibration; [] for every
    block without the key (every 2030 group), so no 2030 rule gains an input.
    """
    basis = (block or {}).get("vintage_capital_2050") or {}
    files = [basis[k] for k in ("capture_prices", "strike_calibration") if basis.get(k)]
    # Year-generic stage 5b: the strike path's capex calibration (AR7's fixed-bottom offshore factor).
    return files + ["data/market/support/cfd_strike_cost_calibration.csv"] if files else files


def _po_support_line_files(block):
    """Decisions 58, 63, 67: the enabled support lines' data files (public_ownership.support_lines; [] without it)."""
    lines = (block or {}).get("support_lines") or {}
    return [spec.get("file", spec.get("table")) for key, spec in lines.items()
            if isinstance(spec, dict) and spec.get("enabled", True) and (spec.get("file") or spec.get("table"))]


def _po_cm_design_files(run_group):
    """FES 2050 decision 73: the 2050 auction's builder outputs the CM design re-clears ([] without the block).

    ``public_ownership.capacity_market_design`` names them; only the FES 2050 study config does,
    so no 2030 rule gains an input or a flag.
    """
    block = (_po_cfg(run_group) or {}).get("capacity_market_design") or {}
    return [block[key] for key in ("fleet", "summary") if block.get(key)]


def _po_cm_design_financing_summary(wildcards):
    """Decision 73: the financing summary the tariff prices the adequacy contract from ([] in 2030)."""
    if not _po_cm_design_files(wildcards.run_group):
        return []
    return [f"{po_resources_path}/public_ownership/{wildcards.run_group}_financing_summary.csv"]


def _po_support_reference(run_group):
    """FES 2050 decisions 58 and 63: a balancing group's market-group register, whose per-asset support lines it holds.

    Only a balancing group (P2 or P3 bid months) of a study with support lines; [] for every other
    group and every 2030 group, so no 2030 rule gains an input or a flag.
    """
    if not ((_po_cfg(run_group) or {}).get("support_lines") or {}):
        return []
    if run_group not in _PO_BALANCING_GROUPS + _PO_TARIFF_BALANCING_GROUPS:
        return []
    return [f"{po_resources_path}/public_ownership/{_po_market_bid_group_for(run_group)}_asset_finance_register.parquet"]


def _po_vintaged_register_flags(run_group):
    """The financing step's 2050 flags: decision 54's rate-path offsets and decisions 58/63's support reference ('' in 2030)."""
    return ("".join(f" --rate-path-offsets {p}" for p in _po_rate_path_offsets(run_group))
            + "".join(f" --support-reference {p}" for p in _po_support_reference(run_group)))


def _po_rate_path_calibrations(block):
    """FES 2050 decision 54: each private rate path's calibration file that exists ([] before the first derive)."""
    basis = (block or {}).get("vintage_capital_2050") or {}
    return [p for p in (basis.get("strike_calibration_by_rate_path") or {}).values() if Path(p).exists()]


def _po_rate_path_offsets(run_group):
    """Decision 54: the study's rate-path strike offsets, for a vintaged group's register and adjustments ([] in 2030)."""
    study = _PO_STUDIES.study_of(run_group)
    if study is None or not ((_po_cfg(run_group).get("vintage_capital_2050") or {}).get("strike_calibration_by_rate_path")):
        return []
    return [f"{po_resources_path}/public_ownership/{study}_cfd_strike_rate_path_offsets.csv"]


rule build_public_ownership_cfd_strike_table_2050:
    """Task SETTLE2050: a study's 2050 CfD strikes in the published strike table's format.

    Y11's post-AR7 strikes (the build year's levelised cost at the private hurdle rate,
    decision 6; BECCS at its electricity-leg strike) and the AR7 and HPC contracts live
    in the model year (``contract_book_2050.strike_table_frame``), priced with the
    study's resolved insurance settings (D4(a)), never the defaults alone. The 2050
    bill names the file as ``model_linked_rules.cfd.strike_table``.
    """
    input:
        config=f"{po_resources_path}/public_ownership/{{study}}_finance_config.yaml",
        strike_inputs="data/market/support/cfd_strike_inputs_2050.csv",
        known="data/market/support/cfd_contracts.csv",
        published="data/market/lccc/cfd_strike_prices_by_round.csv",
        regime="data/market/support/support_regime_2050.json",
        vintage_costs="data/public_ownership/financing/technology_costs_vintage.csv",
        hurdles="data/public_ownership/financing/fes2025_hurdle_rates.csv",
        snapshot=_PO_SNAPSHOT,
        # Decision 41: the capture prices and the break-even calibration of the study's basis.
        vintage_capital=lambda wc: _po_vintage_capital_files(_PO_STUDIES.configs[wc.study].get("public_ownership")),
        # Decision 54: each private rate path's own calibration, once derived.
        rate_path_calibrations=lambda wc: _po_rate_path_calibrations(_PO_STUDIES.configs[wc.study].get("public_ownership"))
    output:
        table=f"{po_resources_path}/public_ownership/{{study}}_cfd_strike_table.csv",
        # Decision 54: the strike offset each private rate path's cases take (no rows without one).
        offsets=f"{po_resources_path}/public_ownership/{{study}}_cfd_strike_rate_path_offsets.csv",
        # Decision 41: the strikes the solved BM bid on (decision 6's economic-life central),
        # which the curtailment compensation is valued at; the contract strikes are above.
        bm_table=f"{po_resources_path}/public_ownership/{{study}}_cfd_bm_strike_table.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        study=PUBLIC_OWNERSHIP_STUDY_REGEX
    params:
        # Year-generic (2040, 11 Nov 2026): the study's modelled year (the post-AR7 cohorts and live contracts).
        model_year=lambda wc: int(((((_PO_STUDIES.configs[wc.study].get("consumer_bill") or {}).get("forecasts") or {})
                                    .get("cases") or {}).get(wc.study) or {}).get("modelled_year", 2050)),
    shell:
        "python scripts/public_ownership/contract_book_2050.py strike-table --out {output.table}"
        " --config {input.config} --snapshot {input.snapshot} --bm-out {output.bm_table}"
        " --rate-path-offsets-out {output.offsets} --model-year {params.model_year}"


# FES 2050 decision 71 (user, 4 Oct 2026): the auction-margin register bounds (margin none and high, the 2030
# lever's values) each settle their own strike table: the study's basis with the bound's calibration (the
# installed central calibration re-margined, vintage_capital_2050.py margin). Their bill assumptions are the
# parent forecast case's with that table, the bound's contract book and DPA settlement; the BM strike table
# stays the central one (the bound is re-accounted on the central months). No 2030 group has the key.
_PO_AUCTION_MARGIN_GROUPS = [g for g in _PO_REGISTER_BOUND_GROUPS if _po_auction_margin_bound(g)]
PUBLIC_OWNERSHIP_AUCTION_MARGIN_GROUP_REGEX = "|".join(map(_po_re.escape, _PO_AUCTION_MARGIN_GROUPS)) or r"(?!x)x"


rule build_public_ownership_auction_margin_inputs_2050:
    """Decision 71: a margin bound's own strike table and bill assumptions (no solve; the BM keeps the central labels)."""
    input:
        config=lambda wc: f"{po_resources_path}/public_ownership/{_PO_STUDIES.study_of(wc.run_group)}_finance_config.yaml",
        assumptions=lambda wc: _po_parent_forecast_case(wc.run_group)["assumptions"],
        calibration=lambda wc: _po_auction_margin_bound(wc.run_group)["strike_calibration"],
        # The central calibration the bound's must be a re-margin of, and the rest of the study's basis.
        vintage_capital=lambda wc: _po_vintage_capital_files(_po_cfg(wc.run_group)),
        strike_inputs="data/market/support/cfd_strike_inputs_2050.csv",
        known="data/market/support/cfd_contracts.csv",
        published="data/market/lccc/cfd_strike_prices_by_round.csv",
        regime="data/market/support/support_regime_2050.json",
        vintage_costs="data/public_ownership/financing/technology_costs_vintage.csv",
        hurdles="data/public_ownership/financing/fes2025_hurdle_rates.csv",
        snapshot=_PO_SNAPSHOT
    output:
        table=f"{po_resources_path}/public_ownership/{{run_group}}_auction_margin_strike_table.csv",
        assumptions=f"{po_resources_path}/public_ownership/{{run_group}}_auction_margin_bill_assumptions.json"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_AUCTION_MARGIN_GROUP_REGEX
    shell:
        "python -m scripts.public_ownership.vintage_capital_2050 bound-inputs --config {input.config}"
        " --run-group {wildcards.run_group} --assumptions {input.assumptions} --snapshot {input.snapshot}"
        " --strike-table-out {output.table} --assumptions-out {output.assumptions}"


# FES 2050 decision 70 (user, 4 Oct 2026; built 25 Oct 2026): the full private-under-central-dispatch case.
# A study forecast case with a central_dispatch_strikes block (FES25_HT_2050_cd) is the market bill on the central
# year with the post-AR7 CfDs at the strike diagnostic's central strikes. Three rules give it its inputs, all
# without a solve and keyed by the forecast case: the strike calibration, strike table and bill assumptions;
# the contract book on its market group's register; and its own year FTI settlement (the 2050 central year
# against the market months, lost CfD support on that book). No 2030 case has the block, so no 2030 rule or
# input changes; the consumer bill rules project it like any central-dispatch case.
from scripts.public_ownership.vintage_capital_2050 import (
    central_dispatch_files as _po_cd_files, central_strikes_spec as _po_cd_spec)

_PO_CD_STRIKE_CASES = {c: _po_cd_spec(v, c) for c, v in _consumer_all_future_cases.items()
                       if v.get("central_dispatch_strikes")}
for _po_cd_case, _po_cd in _PO_CD_STRIKE_CASES.items():
    _po_cd_block = _consumer_all_future_cases[_po_cd_case]
    if _po_cd_case not in _CONSUMER_STUDIES.case_study:
        raise ValueError(f"{_po_cd_case}: central_dispatch_strikes is a study (2050) forecast case's block")
    # F20 phase 2: compared through the parity redirect (identity outside a parity run).
    if _po_parity_path(str(_po_cd_block["assumptions"])) != _po_cd_files(_po_cd_case, po_resources_path)["assumptions"]:
        raise ValueError(f"{_po_cd_case}: assumptions must be {_po_cd_files(_po_cd_case, po_resources_path)['assumptions']}")
    if _po_parity_path(str(_po_cd_block.get("fti_settlement"))) != f"{resources_path}/market/{_po_cd_case}_year_fti_settlement_copperplate.csv":
        raise ValueError(f"{_po_cd_case}: fti_settlement must be "
                         f"{resources_path}/market/{_po_cd_case}_year_fti_settlement_copperplate.csv (its own settlement)")
    if _po_cd["market_group"] not in PUBLIC_OWNERSHIP_GROUPS:
        raise ValueError(f"{_po_cd_case}: central_dispatch_strikes.market_group {_po_cd['market_group']!r} is not a run group")
    _po_cd_fe = {bool((((scenarios[s].get("market") or {}).get("wholesale") or {}).get("forecast_error") or {})
                      .get("enabled", False)) for s in _po_cd_block["physical_scenarios"] if s in scenarios}
    if True in _po_cd_fe:
        raise ValueError(f"{_po_cd_case}: its central months take forecast-error Stage 1; the case's FTI settlement "
                         "settles a perfect-foresight central year only")
PUBLIC_OWNERSHIP_CD_STRIKE_CASE_REGEX = "|".join(map(_po_re.escape, _PO_CD_STRIKE_CASES)) or r"(?!x)x"


def _po_cd_case(wildcards):
    return _consumer_all_future_cases[wildcards.bill_case]


def _po_cd_market_group(wildcards):
    return _PO_CD_STRIKE_CASES[wildcards.bill_case]["market_group"]


rule build_central_dispatch_strike_inputs_2050:
    """Decision 70: a central-dispatch case's strike calibration, strike table and bill assumptions (no solve).

    The study's installed calibration with every post-AR7 CfD cohort at the diagnostic's
    central strike (refused if the diagnostic read another calibration), priced on the study's
    basis; the assumptions are the market case's with that table and the case's contract book.
    """
    input:
        config=lambda wc: f"{po_resources_path}/public_ownership/{_CONSUMER_STUDIES.case_study[wc.bill_case]}_finance_config.yaml",
        assumptions=lambda wc: _PO_CD_STRIKE_CASES[wc.bill_case]["assumptions_from"],
        diagnostic=lambda wc: _PO_CD_STRIKE_CASES[wc.bill_case]["diagnostic"],
        # The installed calibration the central one is derived from, and the rest of the study's basis.
        vintage_capital=lambda wc: _po_vintage_capital_files(
            _CONSUMER_STUDIES.configs[_CONSUMER_STUDIES.case_study[wc.bill_case]].get("public_ownership")),
        strike_inputs="data/market/support/cfd_strike_inputs_2050.csv",
        known="data/market/support/cfd_contracts.csv",
        published="data/market/lccc/cfd_strike_prices_by_round.csv",
        regime="data/market/support/support_regime_2050.json",
        vintage_costs="data/public_ownership/financing/technology_costs_vintage.csv",
        hurdles="data/public_ownership/financing/fes2025_hurdle_rates.csv",
        snapshot=_PO_SNAPSHOT,
        logic="scripts/public_ownership/vintage_capital_2050.py"
    output:
        calibration=f"{po_resources_path}/public_ownership/{{bill_case}}_central_strike_calibration.csv",
        table=f"{po_resources_path}/public_ownership/{{bill_case}}_cfd_strike_table.csv",
        assumptions=f"{po_resources_path}/public_ownership/{{bill_case}}_bill_assumptions.json"
    params:
        contract_book=lambda wc: _po_cd_files(wc.bill_case, po_resources_path)["contract_book"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        bill_case=PUBLIC_OWNERSHIP_CD_STRIKE_CASE_REGEX
    shell:
        "python -m scripts.public_ownership.vintage_capital_2050 central-inputs --config {input.config}"
        " --case {wildcards.bill_case} --assumptions {input.assumptions} --diagnostic {input.diagnostic}"
        " --snapshot {input.snapshot} --calibration-out {output.calibration} --strike-table-out {output.table}"
        " --assumptions-out {output.assumptions} --contract-book {params.contract_book}"


rule build_central_dispatch_contract_book_2050:
    """Decision 70: a central-dispatch case's contract book, its market group's register at the central strikes."""
    input:
        assets=lambda wc: f"{po_resources_path}/public_ownership/{_po_cd_market_group(wc)}_asset_inputs.csv",
        assumptions=f"{po_resources_path}/public_ownership/{{bill_case}}_bill_assumptions.json",
        strike_table=f"{po_resources_path}/public_ownership/{{bill_case}}_cfd_strike_table.csv",
        regime="data/market/support/support_regime_2050.json",
        vintage_costs="data/public_ownership/financing/technology_costs_vintage.csv",
        snapshot=_PO_SNAPSHOT,
        defaults=lambda wc: f"{po_resources_path}/public_ownership/{_CONSUMER_STUDIES.case_study[wc.bill_case]}_finance_config.yaml",
        vintage_capital=lambda wc: _po_vintage_capital_files(
            _CONSUMER_STUDIES.configs[_CONSUMER_STUDIES.case_study[wc.bill_case]].get("public_ownership"))
    output:
        book=f"{po_resources_path}/public_ownership/{{bill_case}}_support_contract_book.csv"
    params:
        market_group=_po_cd_market_group,
        model_price_year=lambda wc: _po_cd_case(wc)["model_price_year"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        bill_case=PUBLIC_OWNERSHIP_CD_STRIKE_CASE_REGEX
    shell:
        "python scripts/public_ownership/contract_book_2050.py book --assets {input.assets}"
        " --assumptions {input.assumptions} --out {output.book} --config {input.defaults}"
        " --run-group {params.market_group} --snapshot {input.snapshot} --model-price-year {params.model_price_year}"


rule build_central_dispatch_case_fti_settlement:
    """Decision 70: a central-dispatch strike case's year FTI settlement (central months against the market months).

    As build_public_ownership_fti_settlement for a 2030 market group, on the case's own
    months named outright (the 2050 stem is not CP30's), perfect foresight (no self-dispatch
    year), the borrowed forecast-error charge at its market group's settings (decision 18:
    fleet ratio 2.32), and firm access paying the lost CfD support on the case's contract
    book (each cohort at its central strike; the 2050 audit's mixed units carry no CfD
    terms of their own). Read by extract_central_dispatch_costs.
    """
    input:
        central_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_cd_case(wc)["physical_scenarios"]],
        central_constraint_costs=lambda wc: [f"{resources_path}/market/{s}_constraint_costs.csv"
                                             for s in _po_cd_case(wc)["physical_scenarios"]],
        central_storage=lambda wc: [f"{resources_path}/market/{s}_wholesale_storage.csv" for s in _po_cd_case(wc)["physical_scenarios"]],
        central_schedule=lambda wc: [f"{resources_path}/market/{s}_wholesale_dispatch.csv" for s in _po_cd_case(wc)["physical_scenarios"]],
        central_links=lambda wc: [f"{resources_path}/market/{s}_wholesale_links.csv" for s in _po_cd_case(wc)["physical_scenarios"]],
        market_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_cd_case(wc)["source_scenarios"]],
        market_schedule=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_dispatch.csv"
                                    for s in _po_cd_case(wc)["source_scenarios"]],
        market_links=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_links.csv"
                                 for s in _po_cd_case(wc)["source_scenarios"]],
        market_prices=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_price.csv"
                                  for s in _po_cd_case(wc)["source_scenarios"]],
        market_storage=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_storage.csv"
                                   for s in _po_cd_case(wc)["source_scenarios"]],
        market_summaries=lambda wc: [f"{resources_path}/analysis/{s}_market_summary.json" for s in _po_cd_case(wc)["source_scenarios"]],
        market_turn_down=lambda wc: turn_down_month_inputs(_po_cd_case(wc)["source_scenarios"]),
        market_cfd_bids=lambda wc: [f"{resources_path}/market/{s}_contract_value_bids.csv" for s in _po_cd_case(wc)["source_scenarios"]],
        # The interconnector cap-and-floor rule, from the case's own bill assumptions.
        assumptions=f"{po_resources_path}/public_ownership/{{bill_case}}_bill_assumptions.json",
        book=f"{po_resources_path}/public_ownership/{{bill_case}}_support_contract_book.csv",
        logic=["scripts/market/fti_settle_year.py", "scripts/market/fti_settlement.py"]
    output:
        settlement=f"{resources_path}/market/{{bill_case}}_year_fti_settlement_copperplate.csv",
        units=f"{resources_path}/market/{{bill_case}}_year_fti_settlement_copperplate_units.csv"
    params:
        central=lambda wc: " ".join(_po_cd_case(wc)["physical_scenarios"]),
        market=lambda wc: " ".join(_po_cd_case(wc)["source_scenarios"]),
        market_wholesale=lambda wc: _po_market_wholesale_flag(_po_cd_case(wc)["source_scenarios"]),
        forecast_error=lambda wc: _po_forecast_error_flags(_po_group_institution(_po_cd_market_group(wc))),
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        bill_case=PUBLIC_OWNERSHIP_CD_STRIKE_CASE_REGEX
    shell:
        "python -m scripts.market.fti_settle_year --central-scenarios {params.central} --market-scenarios {params.market}"
        "{params.market_wholesale} --selfdispatch-suffix none --national-price copperplate"
        " {params.forecast_error} --assumptions {input.assumptions} --cfd-book {input.book}"
        " --out {output.settlement} --units-out {output.units}"


rule build_public_ownership_contract_book:
    """Task SETTLE2050: a vintaged group's contract book, one row per register row holding a live CfD or DPA.

    The register's own labels (Y10/Y11: AR7 and HPC phases, ``PAR7_<year>`` CfDs,
    ``DPA_<year>`` agreements), with each unit's share of its network asset and its
    terms: the strike entry it settles at, or its DPA availability rate. The bill, the
    settlement, the strike adjustments and the constraint line all settle from it.
    """
    input:
        assets=f"{po_resources_path}/public_ownership/{{run_group}}_asset_inputs.csv",
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        strike_table=lambda wc: _po_forecast_assumptions(wc)["model_linked_rules"]["cfd"]["strike_table"],
        regime="data/market/support/support_regime_2050.json",
        vintage_costs="data/public_ownership/financing/technology_costs_vintage.csv",
        snapshot=_PO_SNAPSHOT,
        defaults=_po_cfg_inputs,
        # Decision 41: the DPA availability rates' basis and calibration.
        vintage_capital=lambda wc: _po_vintage_capital_files(_po_cfg(wc.run_group))
    output:
        book=f"{po_resources_path}/public_ownership/{{run_group}}_support_contract_book.csv"
    params:
        config_arg=_po_cfg_arg,
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_VINTAGED_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/contract_book_2050.py book --assets {input.assets}"
        " --assumptions {input.assumptions} --out {output.book} --config {params.config_arg}"
        " --run-group {wildcards.run_group} --snapshot {input.snapshot} --model-price-year {params.model_price_year}"


rule build_public_ownership_vintaged_asset_settlement:
    """Task Y14 (patch 2): the settlement of a vintaged bill group, with its hydrogen turbines' fuel booked.

    The 2030 rule's inputs and command, plus ``--hydrogen-fuel-price`` (group key
    ``hydrogen_fuel_price``, default ``bus0_marginal_price``: AC delivery /
    efficiency x the Stage 1 price at the hydrogen bus; null books none).

    Task SETTLE2050: the group's contract book settled per register row (CfD difference
    payments and curtailment compensation per unit, DPA availability and variable
    payments less gainshare), and the support totals the 2050 bill's DPA line reads.
    """
    input:
        **_PO_SETTLEMENT_INPUTS,
        contract_book=f"{po_resources_path}/public_ownership/{{run_group}}_support_contract_book.csv",
        register_assets=f"{po_resources_path}/public_ownership/{{run_group}}_asset_inputs.csv",
        # The group's own model year (11 Nov 2026: 2040 read 2050's fuel and carbon prices).
        dpa_future_costs=lambda wc: PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get(
            "dpa_future_costs", f"resources/FES/fes2025_future_costs_{_po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[wc.run_group])}.json"),
    output:
        settlement=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement.csv",
        support=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement_support.json"
    params:
        scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"],
        hydrogen_fuel=_po_hydrogen_fuel_flag
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_VINTAGED_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_asset_settlement.py {output.settlement} --scenarios {params.scenarios}"
        " --wholesale {input.wholesale} --balancing {input.balancing} --prices {input.prices} --redispatch {input.redispatch}"
        " --contract-value-bids {input.contract_value_bids}"
        " --assumptions {input.assumptions} --strike-table {input.strike_table} --ro-scheme-summary {input.ro_scheme_costs}"
        " --model-price-year {params.model_price_year} --marginal-cost-breakdown {input.marginal_costs}"
        " {params.hydrogen_fuel} --contract-book {input.contract_book} --register-assets {input.register_assets}"
        " --support-summary {output.support} --dpa-future-costs {input.dpa_future_costs}"


def _po_held_acquisition(run_group):
    """Network bounds (18-19 Oct 2026): ``[register]`` of the group whose purchase ``run_group`` holds, else []."""
    if run_group in _PO_INSTITUTION_TWINS:
        return []        # an institution twin acquires nothing (its cases are market-bill cases, 20 Oct 2026)
    held = _po_purchase_price_group(_po_cfg(run_group), run_group, _PO_ALL_FORECAST_CASES)
    return [] if held is None else [f"{po_resources_path}/public_ownership/{held}_asset_finance_register.parquet"]


def _po_comparator_reference(run_group):
    """F27 item 8 (23 Oct 2026): a 2030 bid-stack group's market-group register, else [].

    The forgone corporation tax judges each case's comparator on the market months there, not on
    private-on-the-state's-bid-stack (not a feasible case). Vintaged (2050) groups keep their own:
    the tax rule is off in a vintaged register.
    """
    if run_group not in _PO_BALANCING_GROUPS + _PO_TARIFF_BALANCING_GROUPS:
        return []
    if _po_is_vintaged_year(_po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[run_group])):
        return []
    return [f"{po_resources_path}/public_ownership/{_po_market_bid_group_for(run_group)}_asset_finance_register.parquet"]


rule calculate_public_ownership_financing:
    """Fixed-dispatch required-revenue account per asset, cohort and case (every scenario and sensitivity)."""
    input:
        assets=f"{po_resources_path}/public_ownership/{{run_group}}_asset_inputs.csv",
        costs=_po_get("technology_costs", "data/public_ownership/financing/technology_costs.csv"),
        snapshot=_PO_SNAPSHOT,
        annual_market_summary=f"{po_resources_path}/public_ownership/{{run_group}}_annual_market_summary.json",
        settlement=_po_settlement_input,
        # F19 P3 decision 2: the legacy valuation's price path (2030 bill groups only).
        price_level_settlement=_po_price_level_settlement,
        price_anchors=_po_price_anchors,
        # Item MR4: calendar-2030 Capacity Market revenue per kW, booked to merchant units
        # (a vintaged model year's own calendar year, task Y11).
        cm_rates=_po_cm_rates,
        # Cases that acquire the legacy fleet price it from the RO run-off (item N2).
        ro_register=lambda wc: _po_group_ro_register(wc.run_group),
        # Item D4(a): the NESO TNUoS tariff tables generator TNUoS is costed from.
        tnuos_tables=lambda wc: [v for k, v in ((_po_cfg(wc.run_group).get("generator_costs") or {}).get("tnuos") or {}).items()
                                 if k in ("wider_tariffs", "offshore_local_tariffs")],
        # Review 29 Sep 2026: the auction capex calibration (decision 1) and HPC's
        # completion cost (decision 10), read through the config.
        register_tables=lambda wc: _po_group_register_tables(wc.run_group),
        # FES 2050 plan, task Y10: a vintaged model year (2040 on) is costed by build
        # year from Y2's table, and the wind-ramp lever reads Y2's hurdle rates.  An
        # empty list for every 2030 group, so their inputs are unchanged.
        vintage_tables=lambda wc: ([(PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get("vintage_costs") or {}).get(
                                        "table", "data/public_ownership/financing/technology_costs_vintage.csv"),
                                    "data/public_ownership/financing/fes2025_hurdle_rates.csv"]
                                   if _po_is_vintaged_year(_po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[wc.run_group]))
                                   else []),
        # FES 2050 decision 41: the capital basis's capture prices ([] for every 2030 group).
        vintage_capital=lambda wc: _po_vintage_capital_files(_po_cfg(wc.run_group))[:1],
        # FES 2050 decision 54: the private rate paths' strike offsets ([] for every 2030 group).
        rate_path_offsets=lambda wc: _po_rate_path_offsets(wc.run_group),
        # FES 2050 decision 58: the support lines' data ([] for every 2030 group).
        support_lines=lambda wc: _po_support_line_files(_po_cfg(wc.run_group)),
        # Decisions 58 and 63: a balancing group holds its market group's support lines ([] in 2030).
        support_reference=lambda wc: _po_support_reference(wc.run_group),
        # FES 2050 decision 73: the 2050 auction the CM design re-clears ([] in 2030; read through --config).
        cm_design=lambda wc: _po_cm_design_files(wc.run_group),
        # Network bounds (18-19 Oct 2026): the register whose acquisition prices this group holds
        # (purchase_price_from: the CENTRAL network's market group, at this group's price level; [] else).
        held_acquisition=lambda wc: _po_held_acquisition(wc.run_group),
        # F27 item 8: a 2030 bid-stack group's market-group register ([] for every other group).
        comparator_reference=lambda wc: _po_comparator_reference(wc.run_group),
        defaults=_po_cfg_inputs
    output:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_financing_summary.csv"
    params:
        config_arg=_po_cfg_arg,
        price_year=_po_get("price_year", 2024),
        settlement_flag=_po_settlement_flag,
        price_level_flag=_po_price_level_flag,
        # The CM rates are in 2025 money; the register divides them by the bill's
        # own currency rebasing (a param, not an input: the assumptions JSON is
        # edited often and only this factor matters here).
        cm_rebasing=lambda wc: _po_cm_rebasing(wc),
        # Decision 54 (rate-path offsets) and decisions 58/63 (a balancing group's support reference):
        # '' for every 2030 group, so its command is unchanged.
        rate_path_offsets=lambda wc: _po_vintaged_register_flags(wc.run_group),
        held_acquisition=lambda wc, input: (f" --held-acquisition-register {input.held_acquisition[0]}"
                                            if len(input.held_acquisition) else ""),
        # D1 (user, 14 Nov 2026): a later group owns exactly what its acquisition group's purchase bought.
        acquisition_baseline=lambda wc: (
            f" --acquisition-baseline data/public_ownership/financing/acquisition_baseline_"
            f"{PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get('acquisition_from_group', 'cp30_2030_cp30_network')}.csv"
            if wc.run_group in _PO_CARRIED_GROUPS else ""),
        comparator_reference=lambda wc, input: (f" --comparator-reference-register {input.comparator_reference[0]}"
                                                if len(input.comparator_reference) else "")
    # Register compaction (29 Sep 2026): peak memory at 210 cases, extrapolated from reduced runs
    # (po_impl/README.md); declared so a --resources mem_mb run does not start two at once.
    resources:
        mem_mb=8000
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_GROUP_REGEX
    shell:
        "python scripts/public_ownership/calculate_financing_scenarios.py {input.assets} {input.costs} {input.snapshot}"
        " {output.register} {output.summary} --annual-market-summary {input.annual_market_summary}"
        " --price-year {params.price_year} {params.settlement_flag}"
        " --capacity-market-revenue {input.cm_rates} --capacity-market-rebasing {params.cm_rebasing}"
        " --config {params.config_arg}{params.rate_path_offsets}{params.price_level_flag}{params.held_acquisition}"
        "{params.comparator_reference}{params.acquisition_baseline}"


rule build_public_ownership_state_shares:
    """Item BM1: the per-generator state share the public-bids balancing solves bid on.

    Made from the run group's finance register: cfd_bm_bids.py treats a generator
    missing from the table as private without a warning. The register is read as
    ``ancient()``: every bill-layer run rewrites it, almost never changing who owns
    what, and a table following it by modification time re-solved all 26 bid months
    for nothing. It is rebuilt only when missing or forced (``-R``), and
    ``verify_public_ownership_ownership_tables`` fails any bill run whose register no
    longer reproduces it for the solved cases.
    """
    input:
        register=ancient(f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet"),
        case_settings=ancient(f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv")
    output:
        shares=f"{po_resources_path}/public_ownership/{{run_group}}_generator_state_shares.csv"
    params:
        # 6 Oct 2026: only the cases the bid months read (the full table was 1 GB at 5,330 cases).
        cases=lambda wc: _po_bid_cases_flag(wc.run_group, "shares")
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_SOLVED_GROUP_REGEX
    shell:
        "python scripts/public_ownership/ownership_shares.py {input.register} {output.shares}{params.cases}"


rule build_public_ownership_deployment_share_tables:
    """FES 2050 plan, task Y9: one BM1 share table per L4 deployment share (25, 50, 75, 100%).

    Each is the per-generator state share of ``public_ownership@deployment_share_<pct>``
    from the group's register, for the per-share balancing solves (decision 16).
    Vintaged groups only; the register must carry the four L4 cases.
    """
    input:
        register=ancient(f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet"),
        case_settings=ancient(f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv")
    output:
        tables=expand(f"{po_resources_path}/public_ownership/{{{{run_group}}}}_generator_state_shares_deployment_share_{{pct}}.csv",
                      pct=(25, 50, 75, 100))
    params:
        prefix=lambda wc: f"{po_resources_path}/public_ownership/{wc.run_group}_generator_state_shares.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_VINTAGED_GROUP_REGEX
    shell:
        "python scripts/public_ownership/ownership_shares.py {input.register} {params.prefix} --deployment-shares 25 50 75 100"


rule build_public_ownership_commissioning_years:
    """Decision 20: the commissioning year P3's LIFO/FIFO tie-break ranks each unit on.

    Dates the CfD pipeline the register leaves undated from the LCCC contract
    portfolio (unit expected start, else its round and technology median; the
    future round at the model year). A separate table so the register, the
    financing and the P2 share table are untouched; only the P3 solves read it. The
    register is ``ancient()`` for the reason given under the share table.
    """
    input:
        register=ancient(f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet"),
        case_settings=ancient(f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv"),
        portfolio="data/market/lccc/cfd_contract_portfolio_status.csv",
    output:
        years=f"{po_resources_path}/public_ownership/{{run_group}}_generator_commissioning_years.csv"
    params:
        # 6 Oct 2026: only the cases the tie-break months read (the full table passed 3 GB).
        cases=lambda wc: _po_bid_cases_flag(wc.run_group, "years")
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_SOLVED_GROUP_REGEX
    shell:
        "python scripts/public_ownership/commissioning_years.py {input.register} {input.portfolio} {output.years}{params.cases}"


def _po_bid_table_pairs(run_group, kind):
    """``"TABLE::CASE"`` for each public-bids BM scenario reading ``run_group``'s share or year table.

    Only the cases the solves bid on are checked: a table carries every case in
    the register, and a longer case list is not an ownership change.
    """
    prefix = f"{po_resources_path}/public_ownership/{run_group}_"
    pairs = []
    for sid in _scenarios:
        block = ((_scenarios.get(sid, {}).get("support", {}) or {}).get("bm_bids", {}) or {})
        block = (block.get("contract_value_bids", {}) or {})
        block = block.get("public_ownership_bids", {}) or {}
        if not block.get("enabled", False):
            continue
        tie = block.get("tie_break", {}) or {}
        table = block.get("shares_file") if kind == "shares" else (
            tie.get("years_file") if tie.get("enabled", False) else None)
        if table and str(table).replace("\\", "/").startswith(prefix):
            pair = f'"{table}::{block["case_id"]}"'
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def _po_bid_cases_flag(run_group, kind):
    """`` --cases A B`` for the cases the bid months read from ``run_group``'s share or year table.

    Empty when no bid scenario reads the table, so the table keeps every case as before.
    The verify rule checks exactly these cases (``_po_bid_table_pairs``).
    """
    cases = []
    for pair in _po_bid_table_pairs(run_group, kind):
        case = pair.strip('"').split("::", 1)[1]
        if case not in cases:
            cases.append(case)
    return (" --cases " + " ".join(f'"{c}"' for c in cases)) if cases else ""


def _po_rab_outside_propositions(run_group):
    """FES 2050 decision 32: the group's study takes Sizewell C's RAB out of P1 and P2 (never a 2030 group)."""
    return bool((_po_cfg(run_group).get("vintage_capital_2050") or {}).get("rab_outside_propositions"))


def _po_verify_script(run_group):
    """The 2030 groups keep the check as it was; a 2050 group with decision 32 on takes its one exemption."""
    if _po_rab_outside_propositions(run_group):
        return "scripts/public_ownership/verify_ownership_tables_2050.py"
    return "scripts/public_ownership/verify_ownership_tables.py"


#: User decision (10 Nov 2026): energy-from-waste left out of the P3 purchase (lever efw_acquisition) takes it out
#: of the state's ownership after the P3 bid months were solved; EfW is redispatched by 5 MWh a year in both the
#: market and the P3 bid months (all three networks), so the months stand. The check records the change in its stamp.
_PO_VERIFY_ACCEPTED_CARRIERS = ("waste_to_energy",)
_PO_VERIFY_ACCEPTED_REASON = ("energy-from-waste left out of the P3 purchase (efw_acquisition, user 10 Nov 2026); "
                              "EfW BM redispatch is 5 MWh a year in market and P3 bid months alike")


_PO_ACCEPTED_OWNERSHIP_CHANGES = "data/public_ownership/financing/accepted_ownership_changes.csv"


def _po_verify_exemption(run_group):
    """No support-type exemption since decision 49 (c): the P2 share tables are rebuilt with Sizewell C private
    and the bid months re-solved on them. The 2050 script keeps its ``--exempt-support-types`` path (and its drop
    of private rows that left the register). The 2030 script accepts the EfW purchase-scope change (10 Nov 2026)."""
    if _po_verify_script(run_group).endswith("_2050.py"):
        # User, 15 Nov 2026: declared ownership changes the solved P2 months are accepted with (capped, recorded).
        return f" --accepted-changes {_PO_ACCEPTED_OWNERSHIP_CHANGES} --run-group {run_group}"
    carriers = " ".join(_PO_VERIFY_ACCEPTED_CARRIERS)
    return f' --accept-carriers {carriers} --accept-reason "{_PO_VERIFY_ACCEPTED_REASON}"'


rule verify_public_ownership_ownership_tables:
    """The share and year tables the P2/P3 bid months were solved on still match the register.

    The other half of the ``ancient()`` inputs above: it recomputes both tables from
    the current register for the solved cases and fails, with the rebuild command,
    if either differs. The bid-group bill merges read its stamp, so a real ownership
    change cannot reach a bill through stale solves.
    """
    input:
        script=lambda wc: _po_verify_script(wc.run_group),
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        portfolio="data/market/lccc/cfd_contract_portfolio_status.csv",
        shares=f"{po_resources_path}/public_ownership/{{run_group}}_generator_state_shares.csv",
        years=f"{po_resources_path}/public_ownership/{{run_group}}_generator_commissioning_years.csv",
        accepted_changes=_PO_ACCEPTED_OWNERSHIP_CHANGES,
    output:
        stamp=f"{po_resources_path}/public_ownership/{{run_group}}_ownership_tables_check.json"
    params:
        shares=lambda wc: " ".join(_po_bid_table_pairs(wc.run_group, "shares")),
        years=lambda wc: " ".join(_po_bid_table_pairs(wc.run_group, "years")),
        exempt=lambda wc: _po_verify_exemption(wc.run_group),
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_SOLVED_MARKET_BID_GROUP_REGEX
    shell:
        "python {input.script} {input.register} {input.portfolio}"
        " {output.stamp} --shares {params.shares} --years {params.years}{params.exempt}"


rule build_public_ownership_legacy_valuation:
    """Item N0: the legacy fleet's cash flows as a declining profile, priced at the public rate and at owners' WACC.

    Reads the private-case legacy rows of the finance register; settings under
    ``public_ownership.acquisition.valuation``.  Consumed by the acquisition tariff.
    """
    input:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        snapshot=_PO_SNAPSHOT,
        ro_register=lambda wc: _po_group_ro_register(wc.run_group),
        # F19 P3 decisions 2 and 3: the price path, the contract dates and the check table.
        price_level_settlement=_po_price_level_settlement,
        price_anchors=_po_price_anchors,
        contract_starts="data/public_ownership/financing/legacy_cfd_contract_starts.csv",
        # F27 item 13: NESO's constraint-cost path for gas plant's balancing margin.
        balancing_path="data/public_ownership/financing/neso_constraint_cost_path.csv",
        multiples="data/public_ownership/financing/legacy_transaction_multiples.csv",
        defaults=_po_cfg_inputs
    output:
        assets=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_valuation.csv",
        profile=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_cash_flow_profile.csv",
        # F19 valuation decisions (14 Oct 2026; 27 Oct 2026): the streams at each point of the valuation
        # levers a case sits at (legacy_price_path, legacy_asset_lives, capacity_procurement), read per
        # case by the tariff.
        profile_by_settings=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_cash_flow_profile_by_valuation_settings.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_valuation_summary.json"
    params:
        config_arg=_po_cfg_arg,
        price_level_flag=_po_price_level_flag
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_GROUP_REGEX
    shell:
        "python scripts/public_ownership/legacy_valuation.py {input.register} {input.snapshot}"
        " {output.assets} {output.profile} {output.summary} --config {params.config_arg}"
        " --transaction-multiples {input.multiples}{params.price_level_flag}"
        " --out-profile-by-valuation-settings {output.profile_by_settings}"


rule build_public_ownership_repowering_sensitivity:
    """OW15: onshore repowering at 0.5, report-only (user, 3 Oct 2026).

    Revalues the legacy fleet at ``public_ownership.acquisition.repowering_sensitivity``'s
    share and carries the change in the market-value purchase price to the acquisition
    tariff's annuity and bill. Refuses a valuation file that a valuation on the current
    config and register would not reproduce.
    """
    input:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        snapshot=_PO_SNAPSHOT,
        valuation=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_valuation.csv",
        tariff_summary=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_summary.json",
        defaults=_po_cfg_inputs,
        # F19: the central valuation's price path (8 Oct 2026).
        price_level_settlement=_po_price_level_settlement,
        price_anchors=_po_price_anchors
    output:
        f"{po_resources_path}/public_ownership/{{run_group}}_repowering_sensitivity.json"
    conda:
        "../envs/pypsa-gb.yaml"
    params:
        config_arg=_po_cfg_arg,
        price_level_flag=_po_price_level_flag
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_GROUP_REGEX
    shell:
        "python scripts/public_ownership/repowering_sensitivity.py {input.register} {input.snapshot}"
        " {input.valuation} {input.tariff_summary} {output} --config {params.config_arg} --run-group {wildcards.run_group}{params.price_level_flag}"


rule extract_network_price_control_building_blocks:
    """Item N6 data: the RIIO-ED2 and RIIO-3 finance-annex building blocks (RAV, allowed WACC, gearing, tax)."""
    input:
        ed2=f"{_PO_NETWORK_SOURCES}/ofgem_riio_ed2_final_determinations_finance_annex_2022_11.pdf",
        riio3=f"{_PO_NETWORK_SOURCES}/ofgem_riio3_final_determinations_finance_annex_2025_12.pdf"
    output:
        licensee=f"{resources_path}/analysis/network_price_control_building_blocks.csv",
        summary=_PO_NETWORK_BLOCKS
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python scripts/consumer_bill/extract_network_price_control_building_blocks.py"
        " --ed2-finance-annex {input.ed2} --riio3-finance-annex {input.riio3}"
        " --out-licensee {output.licensee} --out-summary {output.summary}"


rule build_public_ownership_network_finance_account:
    """Item N6 / J6: the wires under the networks lever, both treatments (acquired at the public rate; debt guarantee)."""
    input:
        blocks=_PO_NETWORK_BLOCKS,
        snapshot=_PO_SNAPSHOT,
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        recovery_demand=lambda wc: _po_forecast_case(wc)["recovery_demand_assumptions"],
        # Review 29 Sep 2026: the premium over RAV (decision 7) and the roll-forward inputs, read
        # through the config.  F29 (26 Oct 2026): no debt guarantee, so no expected-loss table.
        network_tables=lambda wc: _po_group_network_tables(wc.run_group),
        # Task Y12: empty for every 2030 group.
        network_year_tables=_po_network_year_tables,
        defaults=_po_cfg_inputs
    output:
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_network_finance_account.json",
        lines=f"{po_resources_path}/public_ownership/{{run_group}}_network_finance_account.csv"
    params:
        config_arg=_po_cfg_arg
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/network_finance_account.py {input.blocks} {input.snapshot}"
        " {input.assumptions} {input.recovery_demand} {output.summary} {output.lines} --config {params.config_arg}"


rule build_public_ownership_public_account:
    """WP-D, WP-I, WP-E: the public account by case and group and what each rule routes to the bill.

    Review 29 Sep 2026: the account re-prices R1 rows exactly as the bill does,
    so it reads WP-B's strike adjustments (the paid energy each cut divides by)
    and the bill's strike table.
    """
    input:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        adjustments=f"{po_resources_path}/public_ownership/{{run_group}}_cfd_strike_adjustments.csv",
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        strike_table=lambda wc: _po_forecast_assumptions(wc)["model_linked_rules"]["cfd"]["strike_table"],
        defaults=_po_cfg_inputs,
        # F9: the 2030/31 T-4 re-cleared with each case's state plant bidding at its own cost (2030 only).
        cm_auction=_po_cm_auction
    output:
        account=f"{po_resources_path}/public_ownership/{{run_group}}_public_account.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_public_account_summary.json"
    params:
        config_arg=_po_cfg_arg,
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"],
        cm_auction_flag=_po_cm_auction_flag,
        # Item F12: '' for every model-price group, so its command is unchanged.
        price_level_flag=_po_price_level_value_flag
    # Register compaction (29 Sep 2026): peak memory at 210 cases, extrapolated from reduced runs
    # (po_impl/README.md); declared so a --resources mem_mb run does not start two at once.
    resources:
        mem_mb=10000
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_public_account.py {input.register} {output.account} {output.summary}"
        " --config {params.config_arg} --strike-adjustments {input.adjustments} --assumptions {input.assumptions}"
        " --model-price-year {params.model_price_year} --strike-table {input.strike_table}{params.cm_auction_flag}"
        "{params.price_level_flag}"


rule build_public_ownership_consumer_bill_bridge:
    """Trace financing savings to technologies and consumer-bill groups."""
    input:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_financing_summary.csv"
    output:
        technology=f"{po_resources_path}/public_ownership/{{run_group}}_financing_savings_by_technology.csv",
        bridge=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_financing_bridge.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_financing_bridge_summary.json"
    # F20 phase 1 (13 Oct 2026): estimated peak under 1 GB (private, a 16-case batch, small outputs),
    # to be replaced by the peak a timed run measures (stage_timing's peak_mb).
    resources:
        mem_mb=2000
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_consumer_bill_financing_bridge.py {input.register} {input.summary} {output.technology} {output.bridge} {output.summary}"


rule build_public_ownership_cfd_strike_adjustments:
    """WP-B: per-case CfD strike changes reconciling exactly to the register saving."""
    input:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        financing_summary=f"{po_resources_path}/public_ownership/{{run_group}}_financing_summary.csv",
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        strike_table=lambda wc: _po_forecast_assumptions(wc)["model_linked_rules"]["cfd"]["strike_table"],
        award_years=lambda wc: _po_group_award_years(wc.run_group),
        # The strike cut divides by the energy the bill's CfD line actually pays
        # (negative pricing rule, collars, economic curtailment), measured on the
        # same month files the bill settles.  Measured, it is used alone, so the
        # forecast bill's summary (the old fallback denominator) is not an input:
        # since review 29 Sep 2026 the public account reads these adjustments, and
        # the forecast bill reads the public account (its LDES floor), so reading
        # the forecast bill here would close a cycle.
        networks=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale.nc" for s in _po_forecast_case(wc)["source_scenarios"]],
        physical_networks=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_physical_scenarios(wc)],
        turn_down=lambda wc: turn_down_month_inputs(_po_physical_scenarios(wc)),
        prices=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_price.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
        summaries=lambda wc: [f"{resources_path}/analysis/{s}_market_summary.json" for s in _po_forecast_case(wc)["source_scenarios"]],
        # Item D1: the electrolysers' Stage 1 draw is part of the demand the bill weights its price on.
        links=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_links.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
        # Task SETTLE2050: the paid-energy probe settles the 2050 bill's contract book ([] in 2030).
        support_book=_po_bill_contract_book,
        # FES 2050 decision 54: the private rate paths' strike offsets ([] in 2030).
        rate_path_offsets=lambda wc: _po_rate_path_offsets(wc.run_group),
        defaults=_po_cfg_inputs
    output:
        adjustments=f"{po_resources_path}/public_ownership/{{run_group}}_cfd_strike_adjustments.csv",
        routes=f"{po_resources_path}/public_ownership/{{run_group}}_financing_saving_routes.csv"
    params:
        config_arg=_po_cfg_arg,
        model_price_year=lambda wc: _po_forecast_case(wc)["model_price_year"],
        modelled_year=lambda wc: _po_forecast_case(wc)["modelled_year"],
        scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        # Decision 54: '' in 2030, so its command is unchanged.
        rate_path_offsets=lambda wc: "".join(f" --rate-path-offsets {p}" for p in _po_rate_path_offsets(wc.run_group)),
        # Item F12: '' for every model-price group, so its command is unchanged.
        price_level_case=_po_price_level_case_flag("--forecast-case")
    # Register compaction (29 Sep 2026): peak memory at 210 cases, extrapolated from reduced runs
    # (po_impl/README.md); declared so a --resources mem_mb run does not start two at once.
    resources:
        mem_mb=8000
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_cfd_strike_adjustments.py {input.register} {input.financing_summary}"
        " {input.assumptions} {output.adjustments} {output.routes} --config {params.config_arg}"
        " --model-price-year {params.model_price_year} --strike-table {input.strike_table}"
        " --scenarios {params.scenarios} --modelled-year {params.modelled_year}"
        " --networks {input.networks} --physical-networks {input.physical_networks}"
        " --prices {input.prices} --summaries {input.summaries} --links {input.links}{params.rate_path_offsets}"
        "{params.price_level_case}"


rule project_public_ownership_consumer_bill:
    """WP-C: the forecast case's bill projected once per market case (scenarios and sensitivities) through the live projector."""
    input:
        adjustments=f"{po_resources_path}/public_ownership/{{run_group}}_cfd_strike_adjustments.csv",
        routes=f"{po_resources_path}/public_ownership/{{run_group}}_financing_saving_routes.csv",
        financing_summary=f"{po_resources_path}/public_ownership/{{run_group}}_financing_summary.csv",
        public_account=f"{po_resources_path}/public_ownership/{{run_group}}_public_account_summary.json",
        network_lines=f"{po_resources_path}/public_ownership/{{run_group}}_network_finance_account.csv",
        reference_components=lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['case_id']}_consumer_bill_future_components.csv",
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        ro_scheme_costs=lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['case_id']}_ro_scheme_summary.json",
        strike_table=lambda wc: _po_forecast_assumptions(wc)["model_linked_rules"]["cfd"]["strike_table"],
        domestic_profile=lambda wc: _po_forecast_assumptions(wc)["model_linked_rules"]["direct_fuel"]["domestic_shaping"]["profile_file"],
        recovery_demand=lambda wc: _po_forecast_case(wc)["recovery_demand_assumptions"],
        annual_summary=lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['annual_aggregate_study']}_annual_summary.json",
        balancing_costs=lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['case_id']}_future_balancing_costs_summary.json",
        networks=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale.nc" for scenario in _po_forecast_case(wc)["source_scenarios"]],
        physical_networks=lambda wc: [f"{resources_path}/market/{scenario}_balancing.nc" for scenario in _po_physical_scenarios(wc)],
        turn_down=lambda wc: turn_down_month_inputs(_po_physical_scenarios(wc)),
        prices=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale_price.csv" for scenario in _po_forecast_case(wc)["source_scenarios"]],
        summaries=lambda wc: [f"{resources_path}/analysis/{scenario}_market_summary.json" for scenario in _po_forecast_case(wc)["source_scenarios"]],
        # Read beside each Stage 1 network by run_projection (item D1); declared for the DAG.
        wholesale_links=lambda wc: [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale_links.csv" for scenario in _po_forecast_case(wc)["source_scenarios"]],
        # Review 29 Sep 2026: a plain-language source label and link per bill line.
        line_sources=_PO_LINE_SOURCES,
        # Task SETTLE2050: the 2050 contract book and DPA settlement the bill reads ([] in 2030).
        support_inputs=_po_bill_support_inputs,
    output:
        by_case=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_summary.json",
        reconciliation=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_reconciliation.csv"
    params:
        case=lambda wc: _po_forecast_case(wc),
        # Item BM1: each group projects the cases on its own balancing-bid stack.
        cases=lambda wc: _po_group_market_cases(_po_cfg(wc.run_group), wc.run_group),
        tariff_cases=lambda wc: _po_group_excluded_cases(_po_cfg(wc.run_group), wc.run_group),
        # FES 2050 (25 Oct 2026): lines a register bound's own inputs set, kept out of the private-case check.
        bound_lines=lambda wc: _po_bound_own_lines(wc.run_group)
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    script:
        "../scripts/public_ownership/project_public_ownership_consumer_bill.py"


def _po_market_bid_group_for(balancing_group):
    """The contract-value group on the same model year a balancing group is measured against."""
    year = _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[balancing_group])
    # Task Y14: a register bound (register_bound_of) re-accounts its parent's months, so it is never the
    # market group a balancing group is measured against.  No 2030 group is a bound.
    # Item F12: a price-level twin is measured against the market group's twin at its level, every
    # other group against a model-price market group (with no twin configured, exactly as before).
    twin = balancing_group in _PO_PRICE_LEVEL_TWINS
    level = str(_po_parent_forecast_case(balancing_group).get("price_level", "model"))
    # Network bounds (18-19 Oct 2026): and on the same network case (None for every group without one).
    network = _po_group_network_case(PUBLIC_OWNERSHIP_GROUPS[balancing_group])
    # P1/P2 under central dispatch (20 Oct 2026): an institution twin is never the group a bid stack is measured against.
    matches = [g for g in _PO_MARKET_BID_GROUPS if _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[g]) == year
               and g not in _PO_REGISTER_BOUND_GROUPS and (g in _PO_PRICE_LEVEL_TWINS) == twin
               and g not in _PO_INSTITUTION_TWINS
               and _po_group_network_case(PUBLIC_OWNERSHIP_GROUPS[g]) == network
               and (not twin or str(_po_parent_forecast_case(g).get("price_level", "model")) == level)]
    if len(matches) != 1:
        raise ValueError(f"balancing group {balancing_group} needs one contract-value bill group in {year}; got {matches}")
    return matches[0]


def _po_group_months(group, suffix):
    return [f"{resources_path}/market/{s}{suffix}" for s in _po_reference_run_scenarios(PUBLIC_OWNERSHIP_GROUPS[group])]


def _po_months_bid_table(group, kind="shares"):
    """The share (or commissioning-year) table a balancing group's months bid on, from their own config.

    Network bounds (18-19 Oct 2026): the delay networks' bid months bid on cp30_2030_cp30_network's tables
    (their ``public_ownership_bids.shares_file``), not on a table of their own market group, so the
    tables are read off the months rather than named after the market group. None when the group's
    months are not defined or name no table.
    """
    tables = set()
    for sid in _po_reference_run_scenarios(PUBLIC_OWNERSHIP_GROUPS[group]):
        block = (((_scenarios.get(sid) or {}).get("support") or {}).get("bm_bids") or {})
        block = (block.get("contract_value_bids") or {})
        block = block.get("public_ownership_bids") or {}
        if not block.get("enabled", False):
            continue
        table = block.get("shares_file") if kind == "shares" else (block.get("tie_break") or {}).get("years_file")
        if table:
            tables.add(str(table).replace("\\", "/"))
    if len(tables) > 1:
        raise ValueError(f"{group}: its months bid on more than one {kind} table: {sorted(tables)}")
    return tables.pop() if tables else None


def _po_bid_tables_group(market_group):
    """The group whose share and year tables (and so whose verify stamp) a market group's bid stacks read.

    The group the balancing groups' months name in their ``shares_file``
    (``{group}_generator_state_shares.csv``); the market group's solved group when it has no
    balancing group with defined months (every configuration before the network bounds: itself).
    """
    owners = set()
    for group in _po_balancing_groups_for(market_group) + _po_tariff_balancing_groups_for(market_group):
        table = _po_months_bid_table(_po_solved_group(group))
        named = _po_re.search(r"([^/]+)_generator_state_shares\.csv$", table or "")
        if named:
            owners.add(named.group(1))
    if len(owners) > 1:
        raise ValueError(f"{market_group}: its bid stacks bid on tables of several groups: {sorted(owners)}")
    return owners.pop() if owners else _po_solved_group(market_group)


rule build_public_bm_bid_decomposition:
    """Item BM1: per-unit split of the bid stack's effect, month by month, against the market group's months.

    The independent cross-check on the balancing group's bill: transfer to the
    state, payment to private counterparties, the offsetting CfD settlement and
    the turn-down volume, from the balancing LPs' own per-unit costs.
    """
    input:
        script="scripts/market/decompose_public_bm_bids.py",
        # Item Y4: the share table of the market group on the same model year (cp30_2030_cp30_network for 2030).
        # Network bounds (18-19 Oct 2026): the table the months bid on (cp30_2030_cp30_network's for every network).
        shares=lambda wc: f"{po_resources_path}/public_ownership/{_po_bid_tables_group(_po_market_bid_group_for(wc.run_group))}_generator_state_shares.csv",
        base_redispatch=lambda wc: _po_group_months(_po_market_bid_group_for(wc.run_group), "_redispatch_summary.csv"),
        base_bids=lambda wc: _po_group_months(_po_market_bid_group_for(wc.run_group), "_contract_value_bids.csv"),
        public_redispatch=lambda wc: _po_group_months(wc.run_group, "_redispatch_summary.csv"),
        public_bids=lambda wc: _po_group_months(wc.run_group, "_contract_value_bids.csv"),
        turn_down=lambda wc: turn_down_month_inputs(
            [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-len("_redispatch_summary.csv")]
             for g in (_po_market_bid_group_for(wc.run_group), wc.run_group)
             for p in _po_group_months(g, "_redispatch_summary.csv")]),
    output:
        f"{po_resources_path}/public_ownership/{{run_group}}_bm_bid_decomposition.json"
    params:
        case=lambda wc: _po_group_bid_case(_po_cfg(wc.run_group), wc.run_group),
        base=lambda wc: PUBLIC_OWNERSHIP_GROUPS[_po_market_bid_group_for(wc.run_group)]["scenario_pattern"],
        public=lambda wc: PUBLIC_OWNERSHIP_GROUPS[wc.run_group]["scenario_pattern"]
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_ANY_BALANCING_GROUP_REGEX
    shell:
        "python {input.script} --shares {input.shares} --case {params.case}"
        " --months {params.base} {params.public} --out {output}"


def _po_balancing_groups_for(market_group):
    return [g for g in _PO_BALANCING_GROUPS if _po_market_bid_group_for(g) == market_group]


def _po_balancing_merge_flags(wildcards, input):
    groups = _po_balancing_groups_for(wildcards.run_group)
    return " ".join(
        f"--group {g} {by_case} {summary} {decomposition}"
        for g, by_case, summary, decomposition in zip(groups, input.group_by_case, input.group_summary,
                                                      input.decomposition))


rule merge_public_ownership_balancing_groups:
    """Item BM1: the balancing-bid cases beside the market cases, split into bids and ownership."""
    input:
        market_by_case=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        market_summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_summary.json",
        group_by_case=lambda wc: [f"{po_resources_path}/public_ownership/{g}_consumer_bill_by_ownership_case.csv"
                                  for g in _po_balancing_groups_for(wc.run_group)],
        group_summary=lambda wc: [f"{po_resources_path}/public_ownership/{g}_consumer_bill_by_ownership_case_summary.json"
                                  for g in _po_balancing_groups_for(wc.run_group)],
        decomposition=lambda wc: [f"{po_resources_path}/public_ownership/{_po_solved_group(g)}_bm_bid_decomposition.json"
                                  for g in _po_balancing_groups_for(wc.run_group)],
        # Item F12: a twin shares its parent's solve-side tables, so it reads the parent's check.
        # Network bounds (18-19 Oct 2026): the check of the tables the bid months read (cp30_2030_cp30_network's).
        ownership_check=lambda wc: f"{po_resources_path}/public_ownership/{_po_bid_tables_group(wc.run_group)}_ownership_tables_check.json",
    output:
        csv=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_balancing_cases.csv",
        json=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_balancing_cases_summary.json"
    params:
        flags=_po_balancing_merge_flags
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_MARKET_BID_GROUP_REGEX
    shell:
        "python scripts/public_ownership/merge_balancing_groups.py {input.market_by_case} {input.market_summary}"
        " {output.csv} {output.json} {params.flags}"


def _po_tariff_balancing_groups_for(market_group):
    return [g for g in _PO_TARIFF_BALANCING_GROUPS if _po_market_bid_group_for(g) == market_group]


def _po_tariff_merge_flags(wildcards, input):
    groups = _po_tariff_balancing_groups_for(wildcards.run_group)
    return " ".join(
        f"--group {g} {_po_group_bid_case(_po_cfg(g), g)} {by_case} {summary} {components} {tariff}"
        f" {decomposition}"
        for g, by_case, summary, components, tariff, decomposition in zip(
            groups, input.group_by_case, input.group_summary, input.group_tariff_components,
            input.group_tariff_summary, input.decomposition))


rule merge_public_ownership_tariff_balancing_groups:
    """Item BM1, P3: a tariff case's acquisition tariff on its own bid stack beside the same case on the market group.

    The stack's effect is the case's tariff on the balancing group's months less
    its tariff on the market months, line by line, with the purchase-price lines
    split out (the price is struck on the market year) and the decomposition
    reported beside it.  P2's merge (merge_public_ownership_balancing_groups) is
    untouched.
    """
    input:
        script="scripts/public_ownership/merge_tariff_balancing_groups.py",
        market_by_case=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        market_summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_summary.json",
        market_tariff_components=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_components.csv",
        market_tariff_summary=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_summary.json",
        group_by_case=lambda wc: [f"{po_resources_path}/public_ownership/{g}_consumer_bill_by_ownership_case.csv"
                                  for g in _po_tariff_balancing_groups_for(wc.run_group)],
        group_summary=lambda wc: [f"{po_resources_path}/public_ownership/{g}_consumer_bill_by_ownership_case_summary.json"
                                  for g in _po_tariff_balancing_groups_for(wc.run_group)],
        group_tariff_components=lambda wc: [f"{po_resources_path}/public_ownership/{g}_nationalised_bill_components.csv"
                                            for g in _po_tariff_balancing_groups_for(wc.run_group)],
        group_tariff_summary=lambda wc: [f"{po_resources_path}/public_ownership/{g}_nationalised_bill_summary.json"
                                         for g in _po_tariff_balancing_groups_for(wc.run_group)],
        decomposition=lambda wc: [f"{po_resources_path}/public_ownership/{_po_solved_group(g)}_bm_bid_decomposition.json"
                                  for g in _po_tariff_balancing_groups_for(wc.run_group)],
        ownership_check=lambda wc: f"{po_resources_path}/public_ownership/{_po_bid_tables_group(wc.run_group)}_ownership_tables_check.json",
    output:
        csv=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_balancing_cases.csv",
        json=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_balancing_cases_summary.json"
    params:
        flags=_po_tariff_merge_flags
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_MARKET_BID_GROUP_REGEX
    shell:
        "python {input.script} {input.market_by_case} {input.market_summary} {input.market_tariff_components}"
        " {input.market_tariff_summary} {output.csv} {output.json} {params.flags}"


def _po_headline_bill_summary(wildcards):
    """The market group's per-case bill summary: the private bill every tariff headline is set against.

    A tariff balancing group's own private (on its bid-stack months) is not the
    headline private (review 29 Sep 2026, the P3 multi-year basis fix).
    """
    group = wildcards.run_group
    market = group if group in _PO_MARKET_BID_GROUPS else _po_market_bid_group_for(group)
    return f"{po_resources_path}/public_ownership/{market}_consumer_bill_by_ownership_case_summary.json"


def _po_tariff_flags(wildcards):
    """Item BM1, P3: a tariff balancing group tariffs its bid case's fleet family and reads Stage 1 from each parent.

    Empty for the market group, whose months are their own Stage 1.
    """
    flags = []
    if wildcards.run_group in _PO_TARIFF_BALANCING_GROUPS:
        # The script derives the list itself: at 133 cases it overflowed the Windows command line (30 Sep 2026).
        flags.append(f"--tariff-group {wildcards.run_group}")
    scenarios = list(_po_forecast_case(wildcards)["source_scenarios"])
    sources = [_wholesale_source(s) for s in scenarios]
    if sources != scenarios:
        flags.append("--wholesale-scenarios " + " ".join(sources))
    # Task Y12: a group after the 2030 purchase carries its annuity (empty for a 2030 group).
    for path in _po_carried_acquisition(wildcards):
        flags.append(f"--carried-acquisition {path}")
    # FES 2050 decision 73: the adequacy-planning contract's price (empty for a 2030 group).
    for path in _po_cm_design_financing_summary(wildcards):
        flags.append(f"--financing-summary {path}")
    # Item F12: a price-level twin's Stage 1 congestion rent at its mapped GB prices (none elsewhere).
    mapped = _po_mapped_prices(wildcards.run_group)
    if mapped:
        flags.append("--stage-1-prices " + " ".join(mapped))
    return " ".join(flags)


rule build_public_ownership_carried_acquisition:
    """Task Y12: the 2030 purchase carried to a later model year (the annuity, survivors and replacements).

    Reads the acquisition group's tariff summary and legacy valuation (default
    ``cp30_2030_cp30_network``; ``acquisition_from_group`` names another) and the snapshot.  Only
    groups whose model year is after 2030 match.
    """
    input:
        tariff_summary=lambda wc: f"{po_resources_path}/public_ownership/"
                                  f"{PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get('acquisition_from_group', 'cp30_2030_cp30_network')}"
                                  "_nationalised_bill_summary.json",
        valuation=lambda wc: f"{po_resources_path}/public_ownership/"
                             f"{PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get('acquisition_from_group', 'cp30_2030_cp30_network')}"
                             "_legacy_valuation.csv",
        snapshot=_PO_SNAPSHOT,
        # Task SETTLE2050: the model year's register, its legacy rows reported against the survival schedule.
        register_assets=f"{po_resources_path}/public_ownership/{{run_group}}_asset_inputs.csv"
    output:
        carried=f"{po_resources_path}/public_ownership/{{run_group}}_carried_acquisition.json"
    params:
        model_year=lambda wc: _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[wc.run_group]),
        acquisition_year=_PO_ACQUISITION_YEAR
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_CARRIED_GROUP_REGEX
    shell:
        "python scripts/public_ownership/acquisition_carry_forward.py {input.tariff_summary} {input.valuation}"
        " {input.snapshot} {output.carried} --model-year {params.model_year}"
        " --acquisition-year {params.acquisition_year} --register-assets {input.register_assets}"


rule verify_public_ownership_acquisition_baseline:
    """User decision D1 (14 Nov 2026): an acquisition group's committed bought-capacity table equals its register.

    ``data/public_ownership/financing/acquisition_baseline_{acquisition_group}.csv`` is what the later groups' registers
    own up to (``acquisition_baseline.py``); a register change that moves what the purchase bought fails here with
    the command that rewrites it.
    """
    input:
        register=f"{po_resources_path}/public_ownership/{{acquisition_group}}_asset_finance_register.parquet",
        baseline="data/public_ownership/financing/acquisition_baseline_{acquisition_group}.csv",
        code="scripts/public_ownership/acquisition_baseline.py"
    output:
        stamp=f"{po_resources_path}/public_ownership/{{acquisition_group}}_acquisition_baseline_check.txt"
    wildcard_constraints:
        acquisition_group=PUBLIC_OWNERSHIP_UNVINTAGED_GROUP_REGEX
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.public_ownership.acquisition_baseline --register {input.register} --out {input.baseline}"
        " --check --stamp {output.stamp}"


rule audit_public_ownership_purchase:
    """15 Nov 2026: every acquired row priced at its owners' value (row 5%, technology 1%), excluded plant unowned.

    A rule, not an ad-hoc check, so the register and the valuation it is priced from are always the current ones
    (``results_audit.purchase_agreement``). The valuation is the purchase's own group's (``purchase_price_from``).
    """
    input:
        register=f"{po_resources_path}/public_ownership/{{acquisition_group}}_asset_finance_register.parquet",
        valuation=lambda wc: (f"{po_resources_path}/public_ownership/"
                              f"{_po_purchase_price_group(_po_cfg(wc.acquisition_group), wc.acquisition_group, _PO_ALL_FORECAST_CASES) or wc.acquisition_group}"
                              "_legacy_valuation.csv"),
        code="scripts/public_ownership/results_audit.py"
    output:
        report=f"{po_resources_path}/public_ownership/{{acquisition_group}}_purchase_audit.json"
    wildcard_constraints:
        acquisition_group=PUBLIC_OWNERSHIP_UNVINTAGED_GROUP_REGEX
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.public_ownership.results_audit --purchase-register {input.register}"
        " --purchase-valuation {input.valuation} --out {output.report}"


rule verify_public_ownership_tariff_invariants:
    """15 Nov 2026: a buying case that owns DPA plant returns the DPA the kept remaining_ccs line pays
    (``verify_bill_invariants.dpa_failures``); the bug it guards against closed every ledger."""
    input:
        components=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_components.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_summary.json",
        code="scripts/public_ownership/verify_bill_invariants.py"
    output:
        report=f"{po_resources_path}/public_ownership/{{run_group}}_tariff_invariants.json"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.public_ownership.verify_bill_invariants --group {wildcards.run_group} --out {output.report}"
        " --tariff-components {input.components} --tariff-summary {input.summary}"


rule verify_public_ownership_bill_invariants:
    """15 Nov 2026: a case whose state does not hold Sizewell C's RAB carries its public-rate return on the levy
    (``verify_bill_invariants.sizewell_failures``); the bug it guards against closed every ledger."""
    input:
        by_case=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        account=f"{po_resources_path}/public_ownership/{{run_group}}_public_account_summary.json",
        code="scripts/public_ownership/verify_bill_invariants.py"
    output:
        report=f"{po_resources_path}/public_ownership/{{run_group}}_bill_invariants.json"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.public_ownership.verify_bill_invariants --group {wildcards.run_group} --out {output.report}"
        " --bill-by-case {input.by_case} --public-account {input.account}"


rule report_public_ownership_headline_moves:
    """15 Nov 2026: every headline move against the last accepted results has a declared reason
    (``headline_moves.py``; reference and declared moves in data/public_ownership/results_reference/)."""
    input:
        totals=f"{po_resources_path}/public_ownership/{{run_group}}_household_totals.csv",
        reference="data/public_ownership/results_reference/{run_group}_headline.csv",
        declared="data/public_ownership/results_reference/declared_moves.csv",
        code="scripts/public_ownership/headline_moves.py"
    output:
        report=f"{po_resources_path}/public_ownership/{{run_group}}_headline_moves.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.public_ownership.headline_moves --totals {input.totals} --group {wildcards.run_group}"
        " --declared {input.declared} --out {output.report}"


rule verify_fleet_continuity:
    """User, 14 Nov 2026: what a buying case owns in a later model year against what the purchase bought.

    By technology group (``fleet_continuity.py``): owned within min(bought, the year's pre-legislation plant), else the
    run fails; bought plant still repaid but no longer in the year's fleet is reported. Any later year (2035-2050).
    """
    input:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        baseline=lambda wc: ("data/public_ownership/financing/acquisition_baseline_"
                             f"{PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get('acquisition_from_group', 'cp30_2030_cp30_network')}.csv"),
        baseline_check=lambda wc: (f"{po_resources_path}/public_ownership/"
                                   f"{PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get('acquisition_from_group', 'cp30_2030_cp30_network')}"
                                   "_acquisition_baseline_check.txt"),
        valuation=lambda wc: f"{po_resources_path}/public_ownership/"
                             f"{PUBLIC_OWNERSHIP_GROUPS[wc.run_group].get('acquisition_from_group', 'cp30_2030_cp30_network')}"
                             "_legacy_valuation.csv",
        code="scripts/public_ownership/fleet_continuity.py"
    output:
        report=f"{po_resources_path}/public_ownership/{{run_group}}_fleet_continuity.json"
    params:
        model_year=lambda wc: _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[wc.run_group])
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_CARRIED_GROUP_REGEX
    shell:
        "python -m scripts.public_ownership.fleet_continuity --register {input.register}"
        " --model-year {params.model_year} --baseline {input.baseline} --valuation {input.valuation} --out {output.report}"


rule build_public_ownership_nationalised_bill:
    """Items N5, N7 and N8: the acquisition tariff for every case that acquires the legacy fleet.

    Replaces the market-construct lines of the private bill with the cost of
    generation and storage from that case's register rows; the network lines
    follow the case's networks lever (item N6).  Runs on the bill group only;
    joins ``all`` when a configured case acquires the legacy fleet.
    """
    input:
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        bill_by_case=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        valuation=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_valuation.csv",
        profile=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_cash_flow_profile.csv",
        profile_by_settings=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_cash_flow_profile_by_valuation_settings.csv",
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        recovery_demand=lambda wc: _po_forecast_case(wc)["recovery_demand_assumptions"],
        bill_summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_summary.json",
        headline_bill_summary=_po_headline_bill_summary,
        forecast_summary=lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['case_id']}_consumer_bill_future_summary.json",
        network_account=f"{po_resources_path}/public_ownership/{{run_group}}_network_finance_account.json",
        network_lines=f"{po_resources_path}/public_ownership/{{run_group}}_network_finance_account.csv",
        # Item N10: the net payment to foreign counterparties is measured on the links.
        market_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_forecast_case(wc)["source_scenarios"]],
        # Item D3: the cables' Stage 1 congestion rent (day-ahead schedule at the GB price).
        # A wholesale_from month (P3) reads its parent's Stage 1 (schedule, price and the link
        # tables of the shared build); a market month is its own parent.
        # A Stage-1-only parent (s1_zf, 13 Oct 2026) has no balancing network: its own Stage 1
        # network carries the same link tables (nationalised_institutions._stage_1_network_path).
        stage_1_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" if _wholesale_source(s) == s
                                      else f"{resources_path}/market/{_wholesale_source(s)}_wholesale.nc"
                                      for s in _po_forecast_case(wc)["source_scenarios"]],
        market_links=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_links.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
        market_prices=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_price.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
        # Item F12: a price-level twin's mapped month prices ([] for every other group).
        mapped_prices=lambda wc: _po_mapped_prices(wc.run_group),
        # Item D3: the interconnectors' Capacity Market revenue, whose GB halves leave the kept line
        # (a vintaged model year's own rates, task Y11).
        cm_rates=_po_cm_rates,
        # Task Y12: the carried 2030 purchase, for a group after it (empty for a 2030 group).
        carried_acquisition=_po_carried_acquisition,
        # User, 14 Nov 2026: a later year's tariff waits for its fleet-continuity check (D1), which fails the run.
        fleet_continuity=_po_fleet_continuity,
        # FES 2050 decision 73: the adequacy contract's price, from the financing summary ([] in 2030).
        cm_design_financing=_po_cm_design_financing_summary,
        defaults=_po_cfg_inputs
    output:
        components=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_components.csv",
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_summary.json",
        reconciliation=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_reconciliation.csv",
        multi_year=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_multi_year.csv"
    params:
        config_arg=_po_cfg_arg,
        market_scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        tariff_flags=_po_tariff_flags
    # F20 phase 1 (13 Oct 2026): estimated peak 3-4 GB (a batch of 9 cases, and every case's report held
    # for the ~470 MB summary), to be replaced by the peak a timed run measures (stage_timing's peak_mb).
    resources:
        mem_mb=6000
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/nationalised_tariff.py {input.register} {input.bill_by_case}"
        " {input.valuation} {input.profile} {input.assumptions} {input.recovery_demand}"
        " {output.components} {output.summary} {output.reconciliation} {output.multi_year} --config {params.config_arg}"
        " --bill-summary {input.bill_summary} --forecast-summary {input.forecast_summary} --institution I1"
        " --network-account {input.network_account} --market-scenarios {params.market_scenarios}"
        " --cm-rates {input.cm_rates} --headline-bill-summary {input.headline_bill_summary} {params.tariff_flags}"
        " --legacy-profile-by-valuation-settings {input.profile_by_settings}"


def _po_market_wholesale_flag(market_scenarios, flag="--market-wholesale-scenarios"):
    """`` <flag> <parents>`` when a market month takes its Stage 1 through ``wholesale_from``, else ''.

    18 Oct 2026: a market family (cp30_network) reads its Stage 1 family's (stage1) files; a market
    year that owns its Stage 1 keeps its command unchanged.
    """
    scenarios = list(market_scenarios)
    sources = [_wholesale_source(s) for s in scenarios]
    return f" {flag} " + " ".join(sources) if sources != scenarios else ""


rule build_public_ownership_institution_comparison:
    """Item N10: the nationalised bill under the market institution (I1) and central dispatch (I3).

    Spans two solved years, so it joins ``all`` only when the central-dispatch
    scenarios named in ``institution_comparison`` are in ``run_scenarios``.
    """
    input:
        tariff_summary=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_summary.json",
        market_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_forecast_case(wc)["source_scenarios"]],
        # 24 Oct 2026: the market year is measured on the turn-down basis its I1 tariff reads.
        turn_down=lambda wc: turn_down_month_inputs(_po_forecast_case(wc)["source_scenarios"]),
        central_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc"
                                      for s in _po_group_institution_scenarios(wc.run_group)],
        marginal_costs=lambda wc: _po_group_institution(wc.run_group).get(
            "marginal_costs", "resources/generators/marginal_costs_breakdown.csv"),
        # Declared so a re-solve of either year rebuilds the I3 real-time correction first.
        fti_settlement=lambda wc: _po_group_institution(wc.run_group).get("fti_settlement", []),
        # Item D3: the foreign halves' rent on the central schedule at the national
        # price (the market twin's copperplate price); GB shares from the bill's table.
        central_links=lambda wc: [f"{resources_path}/market/{s}_wholesale_links.csv"
                                  for s in _po_group_institution_scenarios(wc.run_group)],
        # Review 27 Sep 2026: the I3 real-time correction is measured on the moved
        # lines' basis, the central physical dispatch less its Stage 1 schedule.
        central_schedule=lambda wc: [f"{resources_path}/market/{s}_wholesale_dispatch.csv"
                                     for s in _po_group_institution_scenarios(wc.run_group)],
        # 18 Oct 2026: a wholesale_from market year (cp30_network on stage1) owns no Stage 1
        # files, so its copperplate price, schedule and links are its parent's, as in the tariff above.
        market_prices=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_price.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
        market_links=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_links.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
        market_schedule=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_dispatch.csv" for s in _po_forecast_case(wc)["source_scenarios"]],
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        # Review B13: the CHP heat credit (marginal_costs.byproduct_adjustments).
        defaults=_po_cfg_inputs,
    output:
        summary=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_by_institution.json",
        comparison=f"{po_resources_path}/public_ownership/{{run_group}}_nationalised_bill_by_institution.csv"
    params:
        config_arg=_po_cfg_arg,
        market_scenarios=lambda wc: " ".join(_po_forecast_case(wc)["source_scenarios"]),
        # Empty when every market month owns its Stage 1, so that command is unchanged.
        market_wholesale=lambda wc: _po_market_wholesale_flag(_po_forecast_case(wc)["source_scenarios"]),
        central_scenarios=lambda wc: " ".join(_po_group_institution_scenarios(wc.run_group)),
        fti_flag=lambda wc: (f"--fti-settlement {_po_group_institution(wc.run_group)['fti_settlement']}"
                             if _po_group_institution(wc.run_group).get("fti_settlement") else ""),
        # P3's central case is central dispatch on the market-value purchase: its I1 is that twin.
        # Decision 18: a group's own forecast-error fleet ratio (2050: 2.32) rescales the
        # settlement's charge; no 2030 group declares one, so its command is unchanged.
        fe_ratio=lambda wc: _po_group_fe_ratio_flag(wc.run_group),
        case_flags=lambda wc: ("--institution-case {} --case {}".format(
                                   *_po_institution_market_twin(_po_cfg(wc.run_group), wc.run_group))
                               if _po_institution_market_twin(_po_cfg(wc.run_group), wc.run_group) else "")
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/nationalised_institutions.py {input.tariff_summary} {output.summary} {output.comparison}"
        " --market-scenarios {params.market_scenarios}{params.market_wholesale} --central-scenarios {params.central_scenarios}"
        " --marginal-costs {input.marginal_costs} {params.fti_flag} {params.case_flags}"
        " --config {params.config_arg} --assumptions {input.assumptions}{params.fe_ratio}"


# The FTI settlement files follow scripts/market/fti_settle_year.py's naming: one
# stem per month with a scenario suffix, the market year it settles against, and,
# for a forecast-error central year only, the self-dispatch Stage 1 schedules on the
# same forecast (firm-access counterfactual). The central year is perfect foresight
# (user decision, 27 Sep 2026: forecast error is the borrowed FTI charge), so firm
# access settles against the market twin's own Stage 1 and no self-dispatch year runs.
# Whether one is needed is read from the central scenarios' own forecast_error, so the
# two cannot drift apart.
# Item Y4: the stem and both suffixes are read off the configured patterns (the
# central year's and the market group's), never written for one year.
# Network bounds (18-19 Oct 2026): one settlement per market group with an institution
# comparison, each on its own central-dispatch year and market months, keyed by the file
# its institution_comparison.fti_settlement names (central_dispatch against cp30_network on the
# CP30 network; central_dispatch_no_acceleration against no_acceleration; and
# central_dispatch_no_upgrades against no_upgrades).
_PO_FTI_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_PO_FTI_SUFFIX_RE = r"\{month\}(\d{4})?_(\w+)$"


def _po_forecast_error_flags(block=None):
    """CLI flags for the borrowed FTI forecast-error charge (audit 26 Sep 2026).

    Read from the group's ``institution_comparison.forecast_error_charge`` (the global
    block unless the group sets its own); a missing block takes the defaults in
    scripts/market/forecast_error_charge.py, and the settings are validated here so a
    bad basis fails at parse time.
    """
    from scripts.market.forecast_error_charge import settings_from_config

    fe = settings_from_config((block if block is not None else _PO_INSTITUTION).get("forecast_error_charge"))
    flags = [f"--forecast-error-basis {fe['basis']}", f"--forecast-error-fleet-ratio {fe['fleet_ratio']}",
             f"--forecast-error-price-year {fe['price_year']}"]
    if fe["as_simple_total"]:
        flags.append("--forecast-error-simple-total")
    if not fe["enabled"]:
        flags.append("--no-forecast-error-charge")
    return " ".join(flags)


def _po_fti_spec(group):
    """The FTI settlement a market group's institution comparison reads, or None without a complete block."""
    block = _po_group_institution(group)
    settlement = block.get("fti_settlement")
    central_pattern = str(block.get("central_scenario_pattern", ""))
    market_pattern = str(PUBLIC_OWNERSHIP_GROUPS[group].get("scenario_pattern", ""))
    central = _po_re.search(_PO_FTI_SUFFIX_RE, central_pattern)
    market = _po_re.search(_PO_FTI_SUFFIX_RE, market_pattern)
    basis = _po_re.search(r"_fti_settlement_(\w+)\.csv$", str(settlement or ""))
    if not (settlement and central and market and basis):
        return None
    stem = central_pattern[:central.start(2)] + "{suffix}"
    if market_pattern[:market.start(2)] + "{suffix}" != stem:
        raise ValueError(f"{group}: the central-dispatch and market months must share one stem "
                         f"({central_pattern!r} against {market_pattern!r})")
    central_months = [stem.format(month=m, suffix=central.group(2)) for m in _PO_FTI_MONTHS]
    fe = {bool((((scenarios[s].get("market") or {}).get("wholesale") or {}).get("forecast_error") or {})
               .get("enabled", False)) for s in central_months if s in scenarios}
    if len(fe) > 1:
        raise ValueError(f"{group}: central dispatch months mix forecast-error and perfect-foresight Stage 1")
    return {
        "group": group, "settlement": str(settlement), "stem": stem,
        "central": central.group(2), "market": market.group(2),
        "selfdispatch": "selfdispatch_fe" if True in fe else "none",
        "basis": basis.group(1),
        "market_months": [stem.format(month=m, suffix=market.group(2)) for m in _PO_FTI_MONTHS],
        "assumptions": _po_forecast_case(SimpleNamespace(run_group=group))["assumptions"],
        "forecast_error": _po_forecast_error_flags(block),
    }


from types import SimpleNamespace

_PO_FTI_SPECS = {}
for _po_fti_group in _PO_MARKET_BID_GROUPS:
    if _po_fti_group in _PO_PRICE_LEVEL_TWINS or _po_fti_group in _PO_STUDY_GROUPS:
        continue
    if not _po_group_institution_cases(_po_fti_group):
        continue
    _po_spec = _po_fti_spec(_po_fti_group)
    if _po_spec is None:
        continue
    _po_same = _PO_FTI_SPECS.get(_po_spec["settlement"])
    if _po_same is not None and {k: v for k, v in _po_same.items() if k != "group"} != {
            k: v for k, v in _po_spec.items() if k != "group"}:
        raise ValueError(f"{_po_same['group']} and {_po_fti_group} name one FTI settlement file "
                         f"({_po_spec['settlement']}) for different years")
    _PO_FTI_SPECS.setdefault(_po_spec["settlement"], _po_spec)
# 20 Oct 2026 (P1/P2 under central dispatch): the rule writes the per-unit transfers beside each year table
# (``<stem>_units.csv``), so its wildcard is the table's stem.
PUBLIC_OWNERSHIP_FTI_SETTLEMENT_REGEX = "|".join(
    map(_po_re.escape, [_po_re.sub(r"\.csv$", "", s) for s in _PO_FTI_SPECS])) or r"(?!x)x"


def _po_fti(wildcards):
    return _PO_FTI_SPECS[f"{wildcards.fti_stem}.csv"]


def _po_fti_files(spec, side, kind):
    suffix = spec[side]
    return [f"{resources_path}/market/{spec['stem'].format(month=m, suffix=suffix)}_{kind}" for m in _PO_FTI_MONTHS]


def _po_fti_market_stage_1_files(spec, kind):
    """18 Oct 2026: the market months' Stage 1 files (their wholesale_from parents; the CP30 families read stage1's)."""
    return [f"{resources_path}/market/{_wholesale_source(s)}_{kind}" for s in spec["market_months"]]


if _PO_FTI_SPECS:

    rule build_public_ownership_fti_settlement:
        """Item N10: the year FTI settlement whose real-time correction the I3 bill carries.

        Was a hand-run step (``python -m scripts.market.fti_settle_year``), so a
        fleet rebuild left the institution comparison reading the previous
        fleet's correction.  The self-dispatch scenarios, needed only for a
        forecast-error central year, contribute only their Stage 1 schedule,
        which is why they are ``wholesale_only``.  Network bounds (18-19 Oct 2026):
        one per market group's institution comparison (``_PO_FTI_SPECS``).
        """
        input:
            central_balancing=lambda wc: _po_fti_files(_po_fti(wc), "central", "balancing.nc"),
            central_constraint_costs=lambda wc: _po_fti_files(_po_fti(wc), "central", "constraint_costs.csv"),
            market_balancing=lambda wc: _po_fti_files(_po_fti(wc), "market", "balancing.nc"),
            # The market twin's Stage 1 is the firm-access counterfactual of a
            # perfect-foresight central year (and its price the D7 price); a
            # wholesale_from market month's is its parent's (18 Oct 2026).
            market_schedule=lambda wc: _po_fti_market_stage_1_files(_po_fti(wc), "wholesale_dispatch.csv"),
            market_links=lambda wc: _po_fti_market_stage_1_files(_po_fti(wc), "wholesale_links.csv"),
            market_prices=lambda wc: _po_fti_market_stage_1_files(_po_fti(wc), "wholesale_price.csv"),
            selfdispatch_schedule=lambda wc: (_po_fti_files(_po_fti(wc), "selfdispatch", "wholesale_dispatch.csv")
                                              if _po_fti(wc)["selfdispatch"] != "none" else []),
            # Decision D7: firm access and cost recovery are valued at the
            # self-dispatch-on-forecast copperplate price (forecast-error year only).
            selfdispatch_prices=lambda wc: (_po_fti_files(_po_fti(wc), "selfdispatch", "wholesale_price.csv")
                                            if _po_fti(wc)["selfdispatch"] != "none" else []),
            # settle() reads the market accounting for its re-trading cross-check and
            # double-count guard (review 26 Sep 2026), so it must wait for them.
            market_summaries=lambda wc: [f"{resources_path}/analysis/{s}_market_summary.json"
                                         for s in _po_fti(wc)["market_months"]],
            market_turn_down=lambda wc: turn_down_month_inputs(_po_fti(wc)["market_months"]),
            # Decisions of 14 Oct 2026 (FTI's method): firm access pays lost CfD support on
            # the strikes the market's balancing stage bid on; storage is made whole week by
            # week to its self-dispatch earnings; interconnector holders day by day to their
            # self-dispatch rent (the self-dispatch links of a forecast-error year).
            market_cfd_bids=lambda wc: [f"{resources_path}/market/{s}_contract_value_bids.csv"
                                        for s in _po_fti(wc)["market_months"]],
            central_storage=lambda wc: _po_fti_files(_po_fti(wc), "central", "wholesale_storage.csv"),
            market_storage=lambda wc: _po_fti_market_stage_1_files(_po_fti(wc), "wholesale_storage.csv"),
            selfdispatch_storage=lambda wc: (_po_fti_files(_po_fti(wc), "selfdispatch", "wholesale_storage.csv")
                                             if _po_fti(wc)["selfdispatch"] != "none" else []),
            selfdispatch_links=lambda wc: (_po_fti_files(_po_fti(wc), "selfdispatch", "wholesale_links.csv")
                                           if _po_fti(wc)["selfdispatch"] != "none" else []),
            # Item D4(b): each side's interconnector cap-and-floor pass-through
            # uses the forecast bill's regime table.
            assumptions=lambda wc: _po_fti(wc)["assumptions"],
        output:
            settlement="{fti_stem}.csv",
            # P1/P2 under central dispatch (20 Oct 2026): each unit's transfers, read by the twin's settlement.
            units="{fti_stem}_units.csv"
        params:
            central=lambda wc: _po_fti(wc)["central"],
            market=lambda wc: _po_fti(wc)["market"],
            selfdispatch=lambda wc: _po_fti(wc)["selfdispatch"],
            basis=lambda wc: _po_fti(wc)["basis"],
            forecast_error=lambda wc: _po_fti(wc)["forecast_error"],
            # Empty when every market month owns its Stage 1, so that command is unchanged.
            market_wholesale=lambda wc: _po_market_wholesale_flag(_po_fti(wc)["market_months"]),
        conda:
            "../envs/pypsa-gb.yaml"
        wildcard_constraints:
            fti_stem=PUBLIC_OWNERSHIP_FTI_SETTLEMENT_REGEX
        shell:
            "python -m scripts.market.fti_settle_year --suffix {params.central} --market-suffix {params.market}"
            "{params.market_wholesale}"
            " --selfdispatch-suffix {params.selfdispatch} --national-price {params.basis}"
            " {params.forecast_error} --assumptions {input.assumptions} --out {output.settlement}"
            " --units-out {output.units}"


def _po_ledger_optional(wildcards):
    """Item D2: the tariff, institution and balancing-bid outputs the ledger reads when their cases are configured.

    Review 29 Sep 2026: the investor column reads more files beside these, found
    by the workflow's naming (``exchequer_ledger.investor_inputs``); each is
    declared here so the ledger runs after them.
    """
    group = wildcards.run_group
    po = f"{po_resources_path}/public_ownership"
    inputs = {"network_account_json": f"{po}/{group}_network_finance_account.json",
              # 24 Oct 2026: the carbon measurement reads each month on its turn-down basis.
              "turn_down": turn_down_month_inputs(_po_physical_scenarios(wildcards))}
    if _po_group_tariff_cases(group):
        inputs["tariff"] = f"{po}/{group}_nationalised_bill_summary.json"
        inputs["legacy_valuation"] = f"{po}/{group}_legacy_valuation.csv"
    if _po_group_institution_ready(group):
        inputs["institution"] = f"{po}/{group}_nationalised_bill_by_institution.json"
        inputs["central_balancing"] = [f"{resources_path}/market/{s}_balancing.nc"
                                       for s in _po_group_institution_scenarios(group)]
    balancing = _po_balancing_groups_for(group)
    if balancing and all(_po_group_runnable(g) for g in balancing):
        inputs["balancing"] = f"{po}/{group}_consumer_bill_balancing_cases_summary.json"
        inputs["balancing_months"] = [f"{resources_path}/market/{s}_balancing.nc" for g in balancing
                                      for s in _po_reference_run_scenarios(PUBLIC_OWNERSHIP_GROUPS[g])]
        inputs["balancing_turn_down"] = turn_down_month_inputs(
            [s for g in balancing for s in _po_reference_run_scenarios(PUBLIC_OWNERSHIP_GROUPS[g])])
        # The bid stack on the private fleet and the ownership effect on it: the merged
        # lines, the market bill's lines (recovery rates), each group's register, bill
        # and account, and both forecast bills (interconnector re-trading).
        inputs["balancing_lines"] = f"{po}/{group}_consumer_bill_balancing_cases.csv"
        inputs["bill_components"] = f"{po}/{group}_consumer_bill_by_ownership_case.csv"
        inputs["balancing_registers"] = [f"{po}/{g}_asset_finance_register.parquet" for g in balancing]
        inputs["balancing_case_settings"] = [f"{po}/{g}_case_settings.csv" for g in balancing]
        inputs["balancing_bills"] = [f"{po}/{g}_consumer_bill_by_ownership_case_summary.json" for g in balancing]
        inputs["balancing_accounts"] = [f"{po}/{g}_public_account_summary.json" for g in balancing]
        inputs["forecast_summaries"] = [
            f"{resources_path}/analysis/{_po_group_forecast_case_id(_po_cfg(g), g)}"
            "_consumer_bill_future_summary.json"
            for g in [group, *balancing]]
    return inputs


def _po_ledger_flags(wildcards, input):
    flags = []
    if "tariff" in input.keys():
        flags.append(f"--tariff-summary {input.tariff}")
    if "institution" in input.keys():
        flags.append(f"--institution-summary {input.institution} --central-scenarios "
                     + " ".join(_po_group_institution_scenarios(wildcards.run_group)))
    if "balancing" in input.keys():
        months = [Path(p).name[:-len("_balancing.nc")] for p in input.balancing_months]
        flags.append(f"--balancing-summary {input.balancing} --balancing-scenarios " + " ".join(months))
    return " ".join(flags)


rule build_public_ownership_exchequer_ledger:
    """Item D2: the Exchequer ledger beside the bill headline, per case (the headline itself does not change).

    Carbon receipts, domestic VAT, corporation tax forgone on state-owned plant,
    the network tax allowance, state deficits funded, the financing saving split
    into resource and tax transfer, and a net-of-Exchequer figure per case.
    Refuses when the finance layer has not exposed corporation tax per row or the
    state deficit per case (no --fallback here: missing fields must fail loudly).
    Review 29 Sep 2026: every case also carries the measured private-investor
    column and the resource saving, and the ledger refuses unless consumers +
    Exchequer + investors = resource saving to GBP 1 (scripts/public_ownership/
    investor_ledger.py).  F29 (26 Oct 2026): the social price of the risk the
    state holds (socialised_risk_gbp) is an Exchequer and resource item, never on
    the bill; the network debt guarantee is retired.
    """
    input:
        unpack(_po_ledger_optional),
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        bill_summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_summary.json",
        public_account=f"{po_resources_path}/public_ownership/{{run_group}}_public_account_summary.json",
        financing_summary=f"{po_resources_path}/public_ownership/{{run_group}}_financing_summary.csv",
        network_lines=f"{po_resources_path}/public_ownership/{{run_group}}_network_finance_account.csv",
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        recovery_demand=lambda wc: _po_forecast_case(wc)["recovery_demand_assumptions"],
        snapshot=_PO_SNAPSHOT,
        marginal_costs=lambda wc: _po_group_institution(wc.run_group).get(
            "marginal_costs", "resources/generators/marginal_costs_breakdown.csv"),
        market_balancing=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _po_physical_scenarios(wc)],
        defaults=_po_cfg_inputs,
        script="scripts/public_ownership/exchequer_ledger.py",
        investor_script="scripts/public_ownership/investor_ledger.py",
        party_flows_script="scripts/public_ownership/party_flows.py",
        # User decision 4: the legal-incidence key of the household total (sensitivity).
        tax_incidence=_po_get("household_tax_incidence", "data/public_ownership/financing/household_tax_incidence.csv"),
        # The gilt-stock externality and the MCPF, in the worst-case household totals only.
        contested_costs=_po_get("contested_fiscal_costs", "data/public_ownership/financing/contested_fiscal_costs.csv"),
        # The per-scenario uncertainties (RO relief, forward premium, CM range, BICS).
        forecast_summary=lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['case_id']}_consumer_bill_future_summary.json",
        # Second round (29 Sep 2026): 'HPC on EDF's current schedule', priced on the market months' Stage 1.
        hpc_schedule=_PO_HPC_SCHEDULE,
        # Item F12: a price-level twin prices it on its mapped month prices.
        market_prices=lambda wc: (_po_mapped_prices(wc.run_group) or
                                  [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_price.csv"
                                   for s in _po_forecast_case(wc)["source_scenarios"]])
    output:
        csv=f"{po_resources_path}/public_ownership/{{run_group}}_exchequer_ledger.csv",
        json=f"{po_resources_path}/public_ownership/{{run_group}}_exchequer_ledger.json",
        # Review 29 Sep 2026: every flow between consumers, the Exchequer, investors and
        # foreign counterparties per case (closure-checked), and the household totals.
        party_flows=f"{po_resources_path}/public_ownership/{{run_group}}_party_flows.csv",
        household_totals=f"{po_resources_path}/public_ownership/{{run_group}}_household_totals.csv"
    params:
        config_arg=_po_cfg_arg,
        market_scenarios=lambda wc: " ".join(_po_physical_scenarios(wc)),
        flags=_po_ledger_flags
    # F20 phase 1 (13 Oct 2026): estimated peak 1.5-2.5 GB (48 cached cases of ~40 columns, every case's
    # rows and flows), to be replaced by the peak a timed run measures (stage_timing's peak_mb).
    resources:
        mem_mb=4000
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_MARKET_BID_GROUP_REGEX
    shell:
        "python {input.script} {input.register} {input.bill_summary} {input.public_account} {input.network_lines}"
        " {input.assumptions} {input.recovery_demand} {input.snapshot} {output.csv} {output.json}"
        " --marginal-costs {input.marginal_costs} --market-scenarios {params.market_scenarios}"
        " --config {params.config_arg} --financing-summary {input.financing_summary}"
        " --tax-incidence {input.tax_incidence}"
        " --contested-costs {input.contested_costs} --forecast-summary {input.forecast_summary}"
        " --hpc-schedule {input.hpc_schedule} --market-prices {input.market_prices}"
        " --party-flows {output.party_flows} --household-totals {output.household_totals} {params.flags}"


def _po_twin_file(suffix):
    """A market group's institution twin's ``{twin}{suffix}`` (the merge's inputs)."""
    return lambda wc: f"{po_resources_path}/public_ownership/{_po_institution_twin_of(wc.run_group)}{suffix}"


PUBLIC_OWNERSHIP_INSTITUTION_PARENT_REGEX = "|".join(
    map(_po_re.escape, sorted(set(_PO_INSTITUTION_TWINS.values())))) or r"(?!x)x"


rule merge_public_ownership_institution_groups:
    """P1/P2 under central dispatch (20 Oct 2026): each twin case's I3 bill beside the market year, in three parts.

    Per case: the I1 bill, the institution's step on private (institution given owner), the
    ownership effect under I3 (owner given institution) and the I3 bill, its headline against
    private-I1 (user decision 8), with the three-party identity closed on each step and on the
    sum. A saving measured under I1 is never added to one measured on private under I3
    (scripts/public_ownership/merge_institution_groups.py).
    """
    input:
        script="scripts/public_ownership/merge_institution_groups.py",
        market_by_case=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        market_summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_summary.json",
        market_ledger=f"{po_resources_path}/public_ownership/{{run_group}}_exchequer_ledger.json",
        market_register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        twin_by_case=_po_twin_file("_consumer_bill_by_ownership_case.csv"),
        twin_summary=_po_twin_file("_consumer_bill_by_ownership_case_summary.json"),
        twin_ledger=_po_twin_file("_exchequer_ledger.json"),
        twin_register=_po_twin_file("_asset_finance_register.parquet"),
        twin_transfers=_po_twin_file("_asset_settlement_transfers.csv"),
        market_forecast=lambda wc: f"{resources_path}/analysis/{_po_forecast_case(wc)['case_id']}_consumer_bill_future_summary.json",
        twin_forecast=lambda wc: (f"{resources_path}/analysis/"
                                  f"{_po_group_forecast_case_id(_po_cfg(wc.run_group), _po_institution_twin_of(wc.run_group))}"
                                  "_consumer_bill_future_summary.json"),
        fti_settlement=lambda wc: _po_group_institution(wc.run_group)["fti_settlement"],
        assumptions=lambda wc: _po_forecast_case(wc)["assumptions"],
        recovery_demand=lambda wc: _po_forecast_case(wc)["recovery_demand_assumptions"],
    output:
        csv=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_institution.csv",
        json=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_institution.json"
    params:
        twin=lambda wc: _po_institution_twin_of(wc.run_group),
        po_dir=f"{po_resources_path}/public_ownership"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_INSTITUTION_PARENT_REGEX
    shell:
        "python {input.script} {wildcards.run_group} {params.twin} {output.csv} {output.json}"
        " --po-dir {params.po_dir} --market-forecast-summary {input.market_forecast}"
        " --twin-forecast-summary {input.twin_forecast} --fti-settlement {input.fti_settlement}"
        " --assumptions {input.assumptions} --recovery-demand {input.recovery_demand}"


def _po_feedback_optional(wildcards):
    """The tariff and institution summaries feed the export when their cases are configured."""
    inputs = []
    if _po_group_tariff_cases(wildcards.run_group):
        inputs.append(f"{po_resources_path}/public_ownership/{wildcards.run_group}_nationalised_bill_summary.json")
    if _po_group_institution_ready(wildcards.run_group):
        inputs.append(f"{po_resources_path}/public_ownership/{wildcards.run_group}_nationalised_bill_by_institution.json")
    return inputs


def _po_feedback_flags(wildcards, input):
    flags = []
    for path in input.optional:
        if path.endswith("_nationalised_bill_summary.json"):
            flags.append(f"--tariff-summary {path}")
        elif path.endswith("_by_institution.json"):
            flags.append(f"--institution-summary {path}")
    return " ".join(flags)


rule build_public_ownership_workbook_esm_feedback:
    """Items J4 and J5: the realised price gap, CPI effect and yield shifts by case, for the workbook's ESM override."""
    input:
        bill_summary=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case_summary.json",
        bill_components=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        financing_summary=f"{po_resources_path}/public_ownership/{{run_group}}_financing_summary.csv",
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        snapshot=_PO_SNAPSHOT,
        optional=_po_feedback_optional
    output:
        csv=f"{po_resources_path}/public_ownership/{{run_group}}_workbook_esm_feedback.csv",
        json=f"{po_resources_path}/public_ownership/{{run_group}}_workbook_esm_feedback.json"
    params:
        flags=_po_feedback_flags
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    shell:
        "python scripts/public_ownership/build_workbook_esm_feedback.py {input.bill_summary} {input.financing_summary}"
        " {input.register} {input.snapshot} {output.csv} {output.json} --bill-components {input.bill_components} {params.flags}"


# ── Reader-facing reports (30 Sep 2026) ───────────────────────────────────────
# The explorer, the dispatch explainer and the bill notebook, built from the
# ownership outputs by rule rather than by hand. The scripts are written for the
# CP30 2030 market group and its bill case; publishing a page stays manual.
# The report group is public_ownership.report_group: the CENTRAL network's market group, the only one carrying
# the full case list the explorer's grid needs. The scripts take the group (and its forecast case,
# _PO_REPORT_CASE) as arguments.
PO_REPORT_GROUP = str(PUBLIC_OWNERSHIP_CONFIG.get("report_group") or "cp30_2030_no_acceleration")
# The auction calibration is derived from one market group's register and held for every group
# (cost_calibration.derived_from_group, the CENTRAL network's), so only that group's register can
# reproduce it; without the key every solved market group is checked, as before.
_PO_CALIBRATION_FROM = (PUBLIC_OWNERSHIP_CONFIG.get("cost_calibration") or {}).get("derived_from_group")
_PO_CALIBRATION_CHECK_GROUPS = (
    ([_PO_CALIBRATION_FROM] if _PO_CALIBRATION_FROM else
     [g for g in _PO_MARKET_BID_GROUPS if g not in _PO_PRICE_LEVEL_TWINS and g not in _PO_STUDY_GROUPS
      and g not in _PO_INSTITUTION_TWINS])
    # A study's (2050) market groups keep their own check, as before the network bounds.
    + [g for g in _PO_MARKET_BID_GROUPS if g in _PO_STUDY_GROUPS and g not in _PO_PRICE_LEVEL_TWINS])
PUBLIC_OWNERSHIP_CALIBRATION_CHECK_GROUP_REGEX = "|".join(map(_po_re.escape, _PO_CALIBRATION_CHECK_GROUPS)) or r"(?!x)x"
_PO_REPORT = f"{po_resources_path}/public_ownership/{PO_REPORT_GROUP}"
# The report group's forecast case (its bill, RO and balancing summaries) and its FTI settlement.
_PO_REPORT_CASE = (_po_group_forecast_case_id(_po_cfg(PO_REPORT_GROUP), PO_REPORT_GROUP)
                   if PO_REPORT_GROUP in PUBLIC_OWNERSHIP_GROUPS else "CP30_2030_cp30_network")
_PO_REPORT_BILL = f"{resources_path}/analysis/{_PO_REPORT_CASE}"
_PO_REPORT_FTI = (_po_group_institution(PO_REPORT_GROUP).get("fti_settlement")
                  if PO_REPORT_GROUP in PUBLIC_OWNERSHIP_GROUPS else None) or \
    f"{resources_path}/market/CP30_FFR_2030_year_cp30_network_central_dispatch_fti_settlement_copperplate.csv"
_PO_EXPLORER_DIR = "scripts/analysis/public_ownership_explorer"
PO_REPORT_TARGETS = [
    f"{po_resources_path}/public_ownership/{_PO_CALIBRATION_FROM or PO_REPORT_GROUP}_cost_calibration_check.json",
    f"{resources_path}/analysis/public_ownership_bill_explorer.html",
    f"{resources_path}/analysis/dispatch_and_wholesale_pricing_explainer.html",
    "notebooks/cp30_2030_consumer_bill_figures.ipynb",
]
# OW11 (3 Oct 2026): the case-settings manifest and the lever spreadsheet (scripts/
# public_ownership/lever_workbook.py), report targets of the CP30 2030 market group.
PO_REPORT_TARGETS += [
    f"{po_resources_path}/public_ownership/{PO_REPORT_GROUP}_case_settings_manifest.json",
    f"{resources_path}/analysis/{PO_REPORT_GROUP}_ownership_levers.xlsx",
]
# OW15: the onshore repowering-share sensitivity, report-only (off since F28 (d), 27 Oct 2026: no private
# life extension in the central valuation).
if ((PUBLIC_OWNERSHIP_CONFIG.get("acquisition") or {}).get("repowering_sensitivity") or {}).get("enabled", False):
    PO_REPORT_TARGETS.append(f"{po_resources_path}/public_ownership/{PO_REPORT_GROUP}_repowering_sensitivity.json")
# The bill chapter of the Sphinx docs quotes the CP30 2030 run: its worked-example
# tables and inline figures are generated from the bill outputs (rule
# build_bill_docs_values, 6 Oct 2026), so a bill rerun refreshes them.
from scripts.analysis.build_bill_docs_values import GENERATED_FILES as _BILL_DOCS_FILES

BILL_DOCS_DIR = "docs/source/model/after_the_solve/consumer_bill/_generated"
PO_REPORT_TARGETS += [f"{BILL_DOCS_DIR}/{name}" for name in _BILL_DOCS_FILES]
# The must-run support sensitivity (user, 8 Oct 2026), report-only.
PO_REPORT_TARGETS += [f"{_PO_REPORT}_must_run_support_sensitivity.json"]
# F27 items 3, 13 and 15 (23 Oct 2026): what ownership cannot reach and what is declared, sized.
PO_REPORT_TARGETS += [f"{po_resources_path}/public_ownership/{PO_REPORT_GROUP}_declared_items.json"]

#: F27 item 2: as-built capital costs of pre-2025 plant (the FIDeR return above cost uses them when present).
_PO_HISTORICAL_COSTS = "data/public_ownership/financing/technology_costs_historical.csv"


rule build_public_ownership_declared_items:
    """F27 items 3, 13 and 15: the out-of-reach list (FIDeR, RO, FiT) and the declared items, sized and dated."""
    input:
        script="scripts/public_ownership/declared_items.py",
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        bill_by_case=f"{po_resources_path}/public_ownership/{{run_group}}_consumer_bill_by_ownership_case.csv",
        network_account=f"{po_resources_path}/public_ownership/{{run_group}}_network_finance_account.csv",
        legacy_valuation=f"{po_resources_path}/public_ownership/{{run_group}}_legacy_valuation.csv",
        asset_settlement=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement.csv",
        historical=lambda wc: [_PO_HISTORICAL_COSTS] if Path(_PO_HISTORICAL_COSTS).exists() else []
    output:
        f"{po_resources_path}/public_ownership/{{run_group}}_declared_items.json"
    params:
        historical=lambda wc, input: f" --historical-costs {input.historical[0]}" if len(input.historical) else ""
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_MARKET_BID_GROUP_REGEX
    shell:
        "python {input.script} {input.register} {input.bill_by_case} {input.network_account} {output}"
        " --legacy-valuation {input.legacy_valuation} --asset-settlement {input.asset_settlement}{params.historical}"


rule build_public_ownership_case_manifest:
    """OW11: the source and config hashes the case settings were written under, beside case_settings.csv.

    The config and the sources are ancient() inputs, so the manifest is rewritten only
    when the case settings are: it records the state they were written in, and the
    lever spreadsheet compares it with today's to mark saved cases current or stale.
    """
    input:
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        script=ancient("scripts/public_ownership/lever_workbook.py"),
        defaults=lambda wc: [ancient(path) for path in _po_cfg_inputs(wc)]
    output:
        manifest=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings_manifest.json"
    params:
        config_arg=_po_cfg_arg
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_GROUP_REGEX
    shell:
        "python {input.script} manifest {input.case_settings} --config {params.config_arg}"
        " --run-group {wildcards.run_group} --out {output.manifest}"


rule build_public_ownership_lever_workbook:
    """OW11: the ownership lever spreadsheet from the case catalogue, with the lever audit's status columns."""
    input:
        script="scripts/public_ownership/lever_workbook.py",
        levers="scripts/public_ownership/levers.py",
        reference_runs="scripts/public_ownership/reference_runs.py",
        defaults=_po_cfg_inputs,
        case_settings=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings.csv",
        manifest=f"{po_resources_path}/public_ownership/{{run_group}}_case_settings_manifest.json",
        household_totals=f"{po_resources_path}/public_ownership/{{run_group}}_household_totals.csv"
    output:
        workbook=f"{resources_path}/analysis/{{run_group}}_ownership_levers.xlsx"
    params:
        config_arg=_po_cfg_arg
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_MARKET_BID_GROUP_REGEX
    shell:
        "python {input.script} workbook --config {params.config_arg} --run-group {wildcards.run_group}"
        " --case-settings {input.case_settings} --manifest {input.manifest}"
        " --household-totals {input.household_totals} --out {output.workbook}"


def _po_group_wanted(group):
    """Whether ``group``'s outputs join ``all``: its months are selected and, for a baseline group (another group's
    ``baseline_run_group``: historical_2024), a group it is the baseline of is selected too. Ownership outputs come on
    only for the studies that need them (user, 18 Nov 2026): running the canonical Historical_2024_ETYS_14zone_market
    alone builds no ownership output."""
    if not _po_group_runnable(group):
        return False
    served = [g for g, spec in PUBLIC_OWNERSHIP_GROUPS.items() if (spec or {}).get("baseline_run_group") == group]
    return not served or any(_po_group_runnable(g) for g in served)


public_ownership_targets = []
if PUBLIC_OWNERSHIP_ENABLED:
    for _po_group in [g for g in PUBLIC_OWNERSHIP_GROUPS if _po_group_wanted(g)]:
        _po_institution_ready = _po_group_institution_ready(_po_group)
        public_ownership_targets.extend([
            f"{po_resources_path}/public_ownership/{_po_group}_annual_market_summary.json",
            f"{po_resources_path}/public_ownership/{_po_group}_source_manifest.json",
            f"{po_resources_path}/public_ownership/{_po_group}_consumer_bill_financing_bridge.csv",
        ])
        if _po_group in _PO_BILL_GROUPS:
            public_ownership_targets.extend([
                f"{po_resources_path}/public_ownership/{_po_group}_cfd_strike_adjustments.csv",
                f"{po_resources_path}/public_ownership/{_po_group}_public_account.csv",
                f"{po_resources_path}/public_ownership/{_po_group}_network_finance_account.json",
                f"{po_resources_path}/public_ownership/{_po_group}_consumer_bill_by_ownership_case.csv",
            ])
        if _po_group in _PO_MARKET_BID_GROUPS:
            public_ownership_targets.append(
                f"{po_resources_path}/public_ownership/{_po_group}_workbook_esm_feedback.csv")
            _po_balancing = _po_balancing_groups_for(_po_group)
            if _po_balancing and all(_po_group_runnable(g) for g in _po_balancing):
                public_ownership_targets.append(
                    f"{po_resources_path}/public_ownership/{_po_group}_consumer_bill_balancing_cases.csv")
            if _po_group_tariff_cases(_po_group):
                public_ownership_targets.append(
                    f"{po_resources_path}/public_ownership/{_po_group}_nationalised_bill_summary.json")
                # Item BM1, P3: the tariff case on its own bid stack, once its months are selected.
                _po_tariff_balancing = _po_tariff_balancing_groups_for(_po_group)
                if _po_tariff_balancing and all(_po_group_runnable(g) for g in _po_tariff_balancing):
                    public_ownership_targets.append(
                        f"{po_resources_path}/public_ownership/{_po_group}_nationalised_bill_balancing_cases.csv")
            if _po_institution_ready:
                public_ownership_targets.append(
                    f"{po_resources_path}/public_ownership/{_po_group}_nationalised_bill_by_institution.json")
            # Item D2: the Exchequer ledger beside every case's bill headline.
            public_ownership_targets.append(
                f"{po_resources_path}/public_ownership/{_po_group}_exchequer_ledger.json")
            # P1/P2 under central dispatch (20 Oct 2026): the twin's cases beside this group's, once the
            # central-dispatch months are selected too (_po_group_runnable of the twin).
            _po_twin = _po_institution_twin_of(_po_group)
            if _po_twin and _po_group_runnable(_po_twin):
                public_ownership_targets.append(
                    f"{po_resources_path}/public_ownership/{_po_group}_consumer_bill_by_institution.json")
            if _po_group == PO_REPORT_GROUP and _po_institution_ready:
                public_ownership_targets.extend(PO_REPORT_TARGETS)


# F20 phase 0 (13 Oct 2026): the parity run's target (scripts/public_ownership/f20_parity/README.md).
# Every ownership output `all` asks for, less the reader-facing reports and a study's (2050) groups,
# plus each solved market-bid group's cost-calibration check.  Nothing asks for it, so `all` is
# unchanged; a parity run names it as its target.
_PO_PARITY_TARGETS = [t for t in public_ownership_targets if t not in PO_REPORT_TARGETS
                      and not any(t.rsplit("/", 1)[-1].startswith(f"{g}_") for g in _PO_STUDY_GROUPS)]
_PO_PARITY_TARGETS += [f"{po_resources_path}/public_ownership/{g}_cost_calibration_check.json"
                       for g in _PO_CALIBRATION_CHECK_GROUPS if _po_group_runnable(g)]


rule public_ownership_parity:
    """F20 phase 0: every 2030 ownership output, for a parity run into its own root (no reports)."""
    input:
        _PO_PARITY_TARGETS,
        # 11 Nov 2026 (user: hand-made steps into the workflow): the 2030 bill's Capacity Market pin agrees with
        # the auction builder (rule verify_capacity_market_pin).
        f"{resources_path}/market/capacity_market/cm_pin_check.json"


# 11 Nov 2026 (user: hand-made steps into the workflow): a target per long-range model year in place of the
# hand-kept runs/bills_<year>_targets.txt. Every bill group of the year: its Exchequer ledger (the end of its bill
# chain: bills, tariff, institution comparison, accounts), household totals and workbook feedback; a bid-stack
# (balancing) group ends in its market group's merge (consumer_bill_balancing_cases.csv).
def _po_year_bill_targets(year):
    po = f"{po_resources_path}/public_ownership"
    groups = [g for g in _PO_BILL_GROUPS if g in _PO_STUDY_GROUPS
              and _po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[g]) == int(year) and _po_group_runnable(g)]
    out = []
    for g in groups:
        if g in _PO_BALANCING_GROUPS:
            merged = f"{po}/{_po_market_bid_group_for(g)}_consumer_bill_balancing_cases.csv"
            if merged not in out:
                out.append(merged)
            continue
        out += [f"{po}/{g}_exchequer_ledger.json", f"{po}/{g}_household_totals.csv",
                f"{po}/{g}_workbook_esm_feedback.csv"]
    # The balancing months bid on support labels carrying the installed strike calibration.
    out.append(f"{po}/fes_ht_{int(year)}_labels_calibration_check.json")
    # User, 14 Nov 2026: the year's bill inputs carry the current model outputs (no figure pinned in a builder).
    out.append(f"{resources_path}/market/capacity_market/fes_bill_inputs_{int(year)}_check.txt")
    return out


rule verify_fes_bill_inputs:
    """A long-range year's committed bill inputs are the builder's output on the current model outputs (user,
    14 Nov 2026: no model figure is pinned in a builder). The builder reads the year's Capacity Market auction
    (cm_calendar_<Y>_summary.json), the network building blocks and the network account's roll-forward; the files
    are the bill's inputs, so not rule outputs. This fails, naming the rebuild, when a model output has moved:
    python -m scripts.consumer_bill.build_fes_bill_inputs --year <Y>."""
    input:
        builder="scripts/consumer_bill/build_fes_bill_inputs.py",
        figures="scripts/consumer_bill/fes_bill_figures.py",
        cm=f"{resources_path}/market/capacity_market/cm_calendar_{{fes_year}}_summary.json",
        blocks=f"{resources_path}/analysis/network_price_control_building_blocks.json",
        roll_forward="data/market/network_cost_sources/rav_roll_forward_{fes_year}.yaml",
        assumptions="data/market/consumer_bill_benchmarks/{fes_year}_pypsa_informed_assumptions.json",
        recovery="data/market/consumer_bill_forecast/{fes_year}_recovery_demand_assumptions.json",
        ro="data/market/consumer_bill_forecast/{fes_year}_ro_scheme_parameters.json",
    output:
        stamp=f"{resources_path}/market/capacity_market/fes_bill_inputs_{{fes_year}}_check.txt"
    wildcard_constraints:
        fes_year=r"20[3-9][0-9]"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python -m scripts.consumer_bill.build_fes_bill_inputs --year {wildcards.fes_year} --check --stamp {output.stamp}"


rule verify_support_labels_calibration:
    """Every calibrated CfD cohort in a long-range year's support labels carries the installed strike calibration
    (user, 11 Nov 2026). The labels read the calibration as ancient, so a re-derived calibration never rebuilds them;
    this fails, naming the rebuild, when they differ (scripts/public_ownership/post_solve_year.py --steps labels)."""
    input:
        script="scripts/public_ownership/post_solve_year.py",
        labels=f"{resources_path}/market/FES25_HT_{{year}}_support_labels.csv",
        calibration="data/public_ownership/financing/strike_calibration_{year}.csv",
    output:
        stamp=f"{po_resources_path}/public_ownership/fes_ht_{{year}}_labels_calibration_check.json"
    wildcard_constraints:
        year=r"20[3-5][0-9]",
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} --year {wildcards.year} --steps check --stamp {output.stamp}"


for _po_bill_year in sorted({_po_group_model_year(PUBLIC_OWNERSHIP_GROUPS[g]) for g in _PO_STUDY_GROUPS
                             if g in _PO_BILL_GROUPS} - {2030}):
    rule:
        name: f"public_ownership_bills_{_po_bill_year}"
        input: _po_year_bill_targets(_po_bill_year)


rule verify_public_ownership_cost_calibration:
    """The committed auction calibration table still reproduces from the register (decision 1).

    The table is the financing step's input, so it cannot be a rule output without a
    cycle; the register records each row's factor, and this recomputes the table
    with the calibration divided back out and fails, with the re-derive command, if
    a factor has drifted.
    """
    input:
        script="scripts/public_ownership/derive_cost_calibration.py",
        register=f"{po_resources_path}/public_ownership/{{run_group}}_asset_finance_register.parquet",
        table=lambda wc: _po_cfg(wc.run_group)["cost_calibration"]["table"],
    output:
        stamp=f"{po_resources_path}/public_ownership/{{run_group}}_cost_calibration_check.json"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_CALIBRATION_CHECK_GROUP_REGEX
    shell:
        "python {input.script} {input.register} --check {input.table} --stamp {output.stamp}"


#: The wholesale price tile's model-year prices (user decision, 10 Nov 2026): 2030 the anchors' reference (CP30
#: Stage 1), 2050 the market family's Stage 1 months.
_PO_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_PO_EXPLORER_PRICES_2030 = ([str(_PO_ANCHOR_CFG["today"]["reference_price_files"]).format(month=m) for m in _PO_MONTHS]
                            if (_PO_ANCHOR_CFG.get("today") or {}).get("reference_price_files") else [])


rule build_public_ownership_explorer:
    """The Public Ownership Bill Explorer page (scenario-first, who pays, exact switches)."""
    input:
        extract=f"{_PO_EXPLORER_DIR}/extract_explorer_data.py",
        build=f"{_PO_EXPLORER_DIR}/build_explorer.py",
        template=f"{_PO_EXPLORER_DIR}/explorer_template.html",
        household_totals=f"{_PO_REPORT}_household_totals.csv",
        party_flows=f"{_PO_REPORT}_party_flows.csv",
        bill_lines=f"{_PO_REPORT}_consumer_bill_by_ownership_case.csv",
        tariff_lines=f"{_PO_REPORT}_nationalised_bill_components.csv",
        strikes=f"{_PO_REPORT}_cfd_strike_adjustments.csv",
        tariff_summary=f"{_PO_REPORT}_nationalised_bill_summary.json",
        ledger=f"{_PO_REPORT}_exchequer_ledger.json",
        sources="data/market/consumer_bill_benchmarks/bill_line_sources.csv",
        bill_summary=f"{_PO_REPORT_BILL}_consumer_bill_future_summary.json",
        energy=f"{resources_path}/CP30/validation/CP30_FFR_2030_energy_totals.csv",
        dispatch_validation=f"{resources_path}/analysis/CP30_2030_reinforced_cp30_energy_validation.csv",
        # the case catalogue: every grid position's case id, through levers.py (imported read-only)
        config=DEFAULTS_FILES,
        levers="scripts/public_ownership/levers.py",
        # the extract's flag defaults (--assumptions, --network-premium, --workbook), read off the same config
        snapshot=_PO_SNAPSHOT,
        network_premium=(PUBLIC_OWNERSHIP_CONFIG.get("network_account") or {}).get(
            "acquisition_premium_table", "data/public_ownership/financing/network_acquisition_premium.csv"),
        workbook=lambda wildcards: _PO_WORKBOOK if _po_workbook_present() else [],
        # User decision (10 Nov 2026; wired 11 Nov): the model year's wholesale price beside today's (the tile).
        price_files=_PO_EXPLORER_PRICES_2030,
    output:
        data=f"{resources_path}/analysis/public_ownership_explorer_data.json",
        html=f"{resources_path}/analysis/public_ownership_bill_explorer.html",
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.extract} --po {po_resources_path}/public_ownership --sources {input.sources}"
        " --bill-summary {input.bill_summary} --energy {input.energy} --dispatch-validation {input.dispatch_validation}"
        " --config config --group " + PO_REPORT_GROUP + " --price-files {input.price_files} --out {output.data}"
        " && python {input.build} {output.data} {output.html} --template {input.template}"


# 10 Nov 2026: the same explorer on a FES long-range study's group (fes_ht_<year>); one rule for every solved year
# (11 Nov 2026: 2040 joins 2050). Its catalogue is the study's resolved finance config (the run's own cases fill what
# it does not list); NESO publishes no constraint figure for these years and the CP30 energy checks do not apply, so
# those trust tiles are left out. Not in `all`.
def _po_explorer_year(wc, suffix):
    return f"{po_resources_path}/public_ownership/fes_ht_{wc.year}{suffix}"


rule build_public_ownership_explorer_year:
    """The Public Ownership Bill Explorer for a FES 2025 Holistic Transition long-range year (2040, 2050)."""
    input:
        extract=f"{_PO_EXPLORER_DIR}/extract_explorer_data.py",
        build=f"{_PO_EXPLORER_DIR}/build_explorer.py",
        template=f"{_PO_EXPLORER_DIR}/explorer_template.html",
        household_totals=lambda wc: _po_explorer_year(wc, "_household_totals.csv"),
        # The law-in-2025 rows (user, 10 Nov 2026): no other target asks for this group's bills.
        household_totals_law_2025=lambda wc: _po_explorer_year(wc, "_legislated_2025_household_totals.csv"),
        party_flows=lambda wc: _po_explorer_year(wc, "_party_flows.csv"),
        bill_lines=lambda wc: _po_explorer_year(wc, "_consumer_bill_by_ownership_case.csv"),
        tariff_lines=lambda wc: _po_explorer_year(wc, "_nationalised_bill_components.csv"),
        strikes=lambda wc: _po_explorer_year(wc, "_cfd_strike_adjustments.csv"),
        tariff_summary=lambda wc: _po_explorer_year(wc, "_nationalised_bill_summary.json"),
        ledger=lambda wc: _po_explorer_year(wc, "_exchequer_ledger.json"),
        sources="data/market/consumer_bill_benchmarks/bill_line_sources.csv",
        bill_summary=lambda wc: f"{resources_path}/analysis/FES25_HT_{wc.year}_consumer_bill_future_summary.json",
        config_year=lambda wc: f"{po_resources_path}/public_ownership/FES25_HT_{wc.year}_finance_config.yaml",
        levers="scripts/public_ownership/levers.py",
        snapshot=_PO_SNAPSHOT,
        price_files=lambda wc: [f"{resources_path}/market/FES25_HT_{wc.year}_{m}_market_wholesale_price.csv"
                                for m in _PO_MONTHS],
    output:
        data=f"{resources_path}/analysis/public_ownership_explorer_data_{{year}}.json",
        html=f"{resources_path}/analysis/public_ownership_bill_explorer_{{year}}.html",
    wildcard_constraints:
        year=r"20[34][0-9]|2050",
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.extract} --po {po_resources_path}/public_ownership --sources {input.sources}"
        " --bill-summary {input.bill_summary} --energy none --dispatch-validation none"
        " --config {input.config_year} --group fes_ht_{wildcards.year} --price-files {input.price_files} --out {output.data}"
        " && python {input.build} {output.data} {output.html} --template {input.template}"


rule build_dispatch_pricing_explainer:
    """The standalone dispatch and wholesale-pricing explainer page."""
    input:
        script="scripts/analysis/build_dispatch_pricing_explainer.py",
        template="scripts/analysis/dispatch_pricing_explainer_template.html",
        settlement=_PO_REPORT_FTI,
        institution=f"{_PO_REPORT}_nationalised_bill_by_institution.json",
        institution_lines=f"{_PO_REPORT}_nationalised_bill_by_institution.csv",
        bill_by_case=f"{_PO_REPORT}_consumer_bill_by_ownership_case_summary.json",
    output:
        f"{resources_path}/analysis/dispatch_and_wholesale_pricing_explainer.html"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} --group " + PO_REPORT_GROUP + " --fti-settlement {input.settlement}"


rule build_cp30_bill_notebook:
    """The CP30 2030 bill figures notebook, generated and executed in place."""
    input:
        script="scripts/analysis/build_cp30_bill_notebook.py",
        bill_summary=f"{_PO_REPORT_BILL}_consumer_bill_future_summary.json",
        by_case=f"{_PO_REPORT}_consumer_bill_by_ownership_case_summary.json",
        financing=f"{_PO_REPORT}_financing_summary.csv",
        technology=f"{_PO_REPORT}_financing_savings_by_technology.csv",
        ledger=f"{_PO_REPORT}_exchequer_ledger.json",
    output:
        "notebooks/cp30_2030_consumer_bill_figures.ipynb"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} --group " + PO_REPORT_GROUP + " --case " + _PO_REPORT_CASE +
        # The pattern holds {month}: doubled braces, or Snakemake formats it as a wildcard (9 Nov 2026).
        " --pattern " + str(PUBLIC_OWNERSHIP_GROUPS[PO_REPORT_GROUP]["scenario_pattern"]).replace("{", "{{").replace("}", "}}") +
        " && jupyter nbconvert --to notebook --execute --inplace"
        " --ExecutePreprocessor.timeout=1800 {output}"


rule build_bill_docs_values:
    """The bill chapter's generated tables and MyST substitutions (scripts/analysis/build_bill_docs_values.py)."""
    input:
        script="scripts/analysis/build_bill_docs_values.py",
        groupings_module="scripts/consumer_bill/bill_groupings.py",
        components=f"{_PO_REPORT_BILL}_consumer_bill_future_components.csv",
        summary=f"{_PO_REPORT_BILL}_consumer_bill_future_summary.json",
        hourly=f"{_PO_REPORT_BILL}_consumer_bill_future_hourly_inputs.csv",
        ro_summary=f"{_PO_REPORT_BILL}_ro_scheme_summary.json",
        ro_fleet=f"{_PO_REPORT_BILL}_ro_eligible_fleet.csv",
        balancing=f"{_PO_REPORT_BILL}_future_balancing_costs_summary.json",
        cm_calendar=f"{resources_path}/market/capacity_market/cm_calendar_2030_summary.json",
        cm_auction=f"{resources_path}/market/capacity_market/cm_auction_model.json",
        public_account=f"{_PO_REPORT}_public_account_summary.json",
        recovery="data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json",
        line_sources="data/market/consumer_bill_benchmarks/bill_line_sources.csv",
        groupings="data/market/consumer_bill_benchmarks/bill_groupings.csv",
        grouping_views="data/market/consumer_bill_benchmarks/bill_grouping_views.csv",
    output:
        [f"{BILL_DOCS_DIR}/{name}" for name in _BILL_DOCS_FILES]
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} --components {input.components} --summary {input.summary}"
        " --hourly {input.hourly} --ro-summary {input.ro_summary} --ro-fleet {input.ro_fleet}"
        " --balancing {input.balancing} --cm-calendar {input.cm_calendar} --cm-auction {input.cm_auction}"
        " --public-account {input.public_account} --recovery {input.recovery}"
        " --line-sources {input.line_sources} --groupings {input.groupings}"
        " --grouping-views {input.grouping_views} --out-dir " + BILL_DOCS_DIR


# ── Capacity Market (item MR4) ────────────────────────────────────────────────
CM_OUT = f"{resources_path}/market/capacity_market"
#: User decision (6 Oct 2026): the 2030/31 T-4 clears against this group's fleet.
CM_FLEET_GROUP = "cp30_2030_cp30_network"


def _cm_fleet_inputs(wildcards=None):
    """The CP30 2030 fleet the 2030/31 T-4 clears against (user decision, 6 Oct 2026).

    The group's register asset inputs (support labels, MW, MWh), its build's mapped
    interconnectors (cable identity) and CP30 ES1 (the capacity path that dates the
    build). None depends on a CM output: the asset inputs come from the group's
    solved market months and the historical baseline, which bid on no ownership
    table, and the other two come before the network build. The CM outputs feed
    only the financing register and the nationalised bill, downstream of them all.

    The build's CP30 cable composition (interconnectors.cp30_composition, 15 Oct 2026)
    is an input too, so the auction sees the cables the network carries (CP30 nominal
    ratings, MaresConnect, no Isle of Man cable, no pro-rata uplift).
    """
    months = _po_reference_run_scenarios(PUBLIC_OWNERSHIP_GROUPS[CM_FLEET_GROUP])
    build = _build_source(months[0])
    if any(_build_source(m) != build for m in months):
        raise ValueError(f"{CM_FLEET_GROUP}: the CM fleet needs one network build across its months")
    inputs = {
        "fleet_assets": f"{po_resources_path}/public_ownership/{CM_FLEET_GROUP}_asset_inputs.csv",
        "fleet_interconnectors": f"{resources_path}/interconnectors/{build}_interconnectors_mapped.csv",
        "fleet_es1": f"{resources_path}/CP30/CP30_ES1.csv",
    }
    composition = get_cp30_interconnector_composition(scenarios.get(build, {}))
    if not composition:
        raise ValueError(f"{CM_FLEET_GROUP}: build {build} has no CP30 interconnector composition")
    inputs["fleet_composition"] = composition[0]
    return inputs


rule verify_capacity_market_pin:
    """The 2030 bill's Capacity Market pin equals the auction builder's summary (user, 11 Nov 2026).

    The assumptions file is the bill's input, so it is not a rule output; this fails, naming the re-pin command
    (scripts/consumer_bill/sync_capacity_market_pin.py --write), when a model-derived number has drifted.
    """
    input:
        script="scripts/consumer_bill/sync_capacity_market_pin.py",
        summary=f"{CM_OUT}/cm_calendar_2030_summary.json",
        assumptions="data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json",
    output:
        stamp=f"{CM_OUT}/cm_pin_check.json"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} --summary {input.summary} --assumptions {input.assumptions} --check --stamp {output.stamp}"


rule build_capacity_market:
    """Item MR4: the calendar-2030 Capacity Market contract book and auction model (no solve of its own).

    The bill's Capacity Market line pins ``cm_calendar_2030_summary.json``'s total
    (tests/unit/test_capacity_market.py checks they agree); the asset register
    takes ``cm_revenue_by_technology.csv``. User decision (6 Oct 2026): the
    2030/31 T-4 supply is the CP30 2030 fleet (``_cm_fleet_inputs``), so the rule
    now waits on the cp30_2030_cp30_network register's asset inputs.
    """
    input:
        unpack(_cm_fleet_inputs),
        register="data/market/capacity_market/cm_register_cmu_20260923.csv",
        components="data/market/capacity_market/cm_register_components_20260923.csv",
        static="data/market/capacity_market/cm_auction_static_data.csv",
        parameters="data/market/capacity_market/cm_auction_parameters.csv",
        cpi="data/market/capacity_market/ons_cpi_d7bt.csv",
        derating="data/market/scarcity/neso_cm_derating_factors.csv",
        gas_schedule="data/market/CP30/gas_station_schedule_2030.csv",
        # MR4.4: Ofgem's 2024 CM allowance over a flat share gives the household peak factor.
        ofgem_ledger="data/market/ofgem_price_cap/2024_annual/component_ledger.csv",
        espeni="data/demand/espeni.csv",
        # The dsr_high case: FES 2025 HT I&C and residential DSR at peak in 2030 (FLX1).
        dsr_fes="data/market/capacity_market/cm_2050_fleet_fes2025_ht.csv",
        library=["scripts/market/capacity_market.py", "scripts/market/capacity_market_2050.py",
                 "scripts/CP30/storage_interconnectors.py"],
        script="scripts/market/build_capacity_market.py"
    output:
        book=f"{CM_OUT}/cm_contract_book_2030.csv",
        model=f"{CM_OUT}/cm_auction_model.json",
        summary=f"{CM_OUT}/cm_calendar_2030_summary.json",
        rates=f"{CM_OUT}/cm_revenue_by_technology.csv",
        matches=f"{CM_OUT}/cm_station_matches.csv",
        supply=f"{CM_OUT}/cm_t4_2030_31_supply.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python scripts/market/build_capacity_market.py --out " + CM_OUT +
        " --fleet-assets {input.fleet_assets} --fleet-interconnectors {input.fleet_interconnectors}"
        " --fleet-es1 {input.fleet_es1} --dsr-fes {input.dsr_fes}"
        " --fleet-composition {input.fleet_composition}"


rule build_capacity_market_year:
    """The calendar Capacity Market of any long-range model year, 2031 to 2050 (FES 2040, 11 Nov 2026).

    The register's agreements live in the year (the 2025/26 to 2029/30 T-4s' 15-year terms in
    2040; none reach 2050), matched to the FES stock-flow fleet, plus the Y11 auction on the residual,
    V2G bidding in as DSR centrally (FES 2050 decision 30) (scripts/market/build_capacity_market_year.py).
    User, 14 Nov 2026: one rule for every long-range year; it replaced rule build_capacity_market_2050 after
    reproducing its outputs exactly. Requested by a vintaged group of that year (``_po_cm_rates``, the CM design
    files) and by the year's bill-input check.
    """
    input:
        register="data/market/capacity_market/cm_register_cmu_20260923.csv",
        static="data/market/capacity_market/cm_auction_static_data.csv",
        parameters="data/market/capacity_market/cm_auction_parameters.csv",
        cpi="data/market/capacity_market/ons_cpi_d7bt.csv",
        derating="data/market/scarcity/neso_cm_derating_factors.csv",
        fleet="data/market/capacity_market/cm_2050_fleet_fes2025_ht.csv",
        calibration="data/market/capacity_market/cm_2050_requirement_calibration.csv",
        library=["scripts/market/capacity_market.py", "scripts/market/capacity_market_2050.py",
                 "scripts/market/build_capacity_market.py", "scripts/market/build_capacity_market_2050.py"],
        script="scripts/market/build_capacity_market_year.py"
    output:
        summary=f"{CM_OUT}/cm_calendar_{{year}}_summary.json",
        rates=f"{CM_OUT}/cm_revenue_by_technology_{{year}}.csv",
        fleet=f"{CM_OUT}/cm_fleet_{{year}}.csv",
        book=f"{CM_OUT}/cm_contract_book_{{year}}.csv"
    wildcard_constraints:
        year=r"20(3[1-9]|4[0-9]|50)"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python scripts/market/build_capacity_market_year.py --model-year {wildcards.year} --out " + CM_OUT


rule build_must_run_support_sensitivity:
    """Bill sensitivity (user, 8 Oct 2026): the support CP30's must-run plant without a contract would need.

    Lynemouth (contract ended 2027) and the merchant biogas proxies run to the pathway's
    energy at a loss; this pays each unit's 2030 going-forward shortfall on the bill line
    its support would ride on (``must_run_support_sensitivity.json``). Report-only: the
    central bill is unchanged.
    """
    input:
        script="scripts/consumer_bill/must_run_support_sensitivity.py",
        register=f"{_PO_REPORT}_asset_finance_register.parquet",
        settings=f"{_PO_REPORT}_case_settings.csv",
        bill=f"{_PO_REPORT_BILL}_consumer_bill_future_summary.json",
        spec="data/market/consumer_bill_benchmarks/must_run_support_sensitivity.json"
    output:
        csv=f"{_PO_REPORT}_must_run_support_sensitivity.csv",
        json=f"{_PO_REPORT}_must_run_support_sensitivity.json"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} {input.register} {input.bill} {input.spec} {output.csv} {output.json}"


rule report_capacity_market_revenue_adequacy:
    """Item MR4: energy margin + Capacity Market revenue against gas fixed costs, per gas class."""
    input:
        settlement=f"{po_resources_path}/public_ownership/{{run_group}}_asset_settlement.csv",
        assets=f"{po_resources_path}/public_ownership/{{run_group}}_asset_inputs.csv",
        rates=f"{CM_OUT}/cm_revenue_by_technology.csv",
        costs="data/market/capacity_market/gas_plant_costs_2024.csv"
    output:
        adequacy=f"{po_resources_path}/public_ownership/{{run_group}}_merchant_revenue_adequacy.csv"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        run_group=PUBLIC_OWNERSHIP_BILL_GROUP_REGEX
    shell:
        "python scripts/market/build_capacity_market.py --out " + CM_OUT +
        " --adequacy {input.settlement} {input.assets} {output.adequacy}"


# 10 Nov 2026 (user): the 2030 anchor is the report group (CENTRAL network); a solved year between 2030 and
# 2050 (2040 when it is run) is one more anchor: public_ownership.pathway.intermediate_snapshots {year: run_group}
# names its run group, whose Exchequer ledger the pathway then passes through.
_PATHWAY_MID = {int(y): str(g) for y, g in
                ((PUBLIC_OWNERSHIP_CONFIG.get("pathway") or {}).get("intermediate_snapshots") or {}).items()}
_PATHWAY_MID_LEDGERS = {y: f"{po_resources_path}/public_ownership/{g}_exchequer_ledger.json"
                        for y, g in sorted(_PATHWAY_MID.items())}
_PATHWAY_MID_ARGS = "".join(f" --snapshot {y}={path}" for y, path in _PATHWAY_MID_LEDGERS.items())


rule build_public_ownership_pathway_2030_2050:
    """FES 2050 plan task Y15: the 2030-2050 pathway-lite view of the public ownership benefits.

    No network is solved: financing year by year from the 2050 vintaged register's rows
    (plus the stock-flow's cohorts retired before 2050), P3's legacy cash and carried
    annuity by year, and everything else linear between the two solved snapshots, on the
    private-vs-case difference. The 2030 and 2050 rows reproduce each snapshot's Exchequer
    ledger by case, mechanism and party to GBP 1 m. New outputs only; never in `all`.
    """
    input:
        ledger_2030=f"{po_resources_path}/public_ownership/{PO_REPORT_GROUP}_exchequer_ledger.json",
        ledger_2050=f"{po_resources_path}/public_ownership/fes_ht_2050_exchequer_ledger.json",
        ledgers_mid=list(_PATHWAY_MID_LEDGERS.values()),
        register_2050=f"{po_resources_path}/public_ownership/fes_ht_2050_asset_finance_register.parquet",
        case_settings=f"{po_resources_path}/public_ownership/fes_ht_2050_case_settings.csv",
        config_2050=f"{po_resources_path}/public_ownership/FES25_HT_2050_finance_config.yaml",
        legacy_profile_2030=f"{po_resources_path}/public_ownership/{PO_REPORT_GROUP}_legacy_cash_flow_profile.csv",
        carried=f"{po_resources_path}/public_ownership/fes_ht_2050_carried_acquisition.json",
        fleet_flows=f"{po_resources_path}/public_ownership/fes_ht_2050_fleet_vintage_flows.csv",
        fleet_vintages=f"{po_resources_path}/public_ownership/fes_ht_2050_fleet_vintages.csv",
        fes_data=f"{resources_path}/FES/FES_2025_data.csv",
        script="scripts/public_ownership/pathway_2030_2050.py"
    output:
        csv=f"{po_resources_path}/public_ownership/pathway_2030_2050.csv",
        summary=f"{po_resources_path}/public_ownership/pathway_2030_2050_summary.json",
        figure=f"{po_resources_path}/public_ownership/pathway_2030_2050.png"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} --ledger-2030 {input.ledger_2030} --ledger-2050 {input.ledger_2050}"
        " --register-2050 {input.register_2050} --case-settings {input.case_settings}"
        " --config-2050 {input.config_2050} --legacy-profile-2030 {input.legacy_profile_2030}"
        " --carried-acquisition {input.carried} --fleet-flows {input.fleet_flows}"
        " --fleet-vintages {input.fleet_vintages} --fes-data {input.fes_data}"
        " --out-csv {output.csv} --out-json {output.summary} --out-figure {output.figure}" + _PATHWAY_MID_ARGS


rule build_public_ownership_pathway_explorer:
    """The cumulative 2030-2050 page (user, 10 Nov 2026), from the pathway summary; the solved years come from it."""
    input:
        summary=f"{po_resources_path}/public_ownership/pathway_2030_2050_summary.json",
        build="scripts/analysis/public_ownership_pathway_explorer/build_pathway_explorer.py",
        template="scripts/analysis/public_ownership_pathway_explorer/pathway_explorer_template.html",
    output:
        html=f"{resources_path}/analysis/public_ownership_pathway_explorer.html",
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.build} {input.summary} {output.html} --template {input.template}"
