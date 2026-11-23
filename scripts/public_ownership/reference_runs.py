"""Shared reference-run expansion, case expansion and decision-register validation.

``rules/public_ownership.smk`` and ``run_financing.py`` both import from here so
that the Snakemake graph and the standalone runner expand identical scenario
lists, identical case lists and enforce identical configured decisions.

Item K1: the case list is built in exactly one place, :func:`configured_case_specs`,
from ``public_ownership.scenarios`` (three bundles) and
``public_ownership.sensitivities`` (one lever at a time, named positions, the
bounding pair).  Every stage and ``rule all`` read that list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from scripts.public_ownership.finance import STATE_OVERHEAD_RATE_KEYS
from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR, group_model_year
from scripts.public_ownership.levers import (
    BIDS_CONTRACT_VALUE, LEVER_ACQUISITION_TERM, LEVER_BALANCING_BIDS, PRIVATE_SCENARIO, INSTITUTION_MARKET, LEGACY_PRIVATE, LEVER_DISPATCH_INSTITUTION, LEVER_LEGACY_FLEET, LEVER_NETWORKS,
    LEVER_LEGACY_PRICE_PATH, LEVER_PURCHASE_PRICE_BASIS, LEVER_THERMAL_BASIS, LEVERS, NETWORKS_PRIVATE,
    OVERHEAD_CENTRAL, OVERHEAD_HIGH, OVERHEAD_LOW, LeverConfigError, CaseSpec, base_case, expand_cases,
    LEVER_DELIVERED_SHARE, parse_scenarios, parse_sensitivities,
)
from scripts.public_ownership.levers import (
    LEVER_CAPACITY_PROCUREMENT, GATED_LEVERS, LEVER_STATE_DEPLOYMENT_SHARE, VARIANT_SEPARATOR, gated_levers_in, lever_default,
)
from scripts.public_ownership.levers import LEVER_LEGACY_OFFSHORE_PRICE


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MODES = frozenset({"full_year", "twelve_months"})
SUPPORTED_COMPONENT_TYPES = frozenset({"Generator", "StorageUnit"})

# Keys of the item-D1/N2 configuration that item K1 retired.  Naming one fails
# the parse with a pointer at where the decision now lives.
RETIRED_KEYS: dict[str, str] = {
    "cases": "declare the three scenarios under public_ownership.scenarios (item K1, plan Part 6.3)",
    "sensitivity_axes": "declare sensitivities under public_ownership.sensitivities (item K1)",
    "existing_assets": "acquisition of the legacy fleet is the legacy_fleet lever, set in a bundle or a named sensitivity",
    "network_ownership": "the networks lever, set in a bundle or a named sensitivity",
    "yield_assumption": "the sovereign_yield lever; list it under sensitivities.one_at_a_time",
    "surplus_recycling": "the surplus_recycling lever, set per scenario bundle",
    "public_shortfall_funding": "the shortfall_funding lever, set per scenario bundle",
    "public_ownership_asset_boundary": "the pre_2025_cfd_capacity lever",
    "boundary_sensitivities": "the pre_2025_cfd_capacity lever under sensitivities.structural",
    # OW9 (3 Oct 2026): public_return: explicit is retired (levers.RETIRED_VALUES).
    "public_return_explicit_rate": ("retired 3 Oct 2026 with public_return: explicit (OW9); the public return is "
                                    "public_rate or cepa_decomposition"),
    # F29 (26 Oct 2026): workbook v2.0 sets the state's cash rate (real gilt + lending margin).
    "public_rate_sources": ("retired 26 Oct 2026 (F29) with public_rate_basis: gilt_30y_real; the cash rate's "
                            "alternatives are workbook v2.0's lending margins (public_rate_basis)"),
}
RETIRED_ACQUISITION_KEYS: dict[str, str] = {
    "enabled": "acquisition is switched on by any case whose legacy_fleet lever is not private",
    "purchase_price_basis": "the purchase_price_basis lever (a named sensitivity extending fully_nationalised)",
    "thermal_basis": "the thermal_basis lever",
    "term_years": "the acquisition_term_years lever",
    # Review 29 Sep 2026 (user decision 5): who pays the purchase is a lever now.
    "include_in_net_bill": ("the acquisition_financing lever (tariff, or exchequer: the annuity funded from "
                            "general taxation, taxpayer_funded_purchase), set in a bundle or a preset"),
    # OW9 (3 Oct 2026): the explicit purchase and thermal bases are retired (levers.RETIRED_VALUES).
    "purchase_price_gbp": "retired 3 Oct 2026 with purchase_price_basis: explicit (OW9)",
    "thermal_price_gbp": "retired 3 Oct 2026 with thermal_basis: explicit (OW9)",
    # F29 (26 Oct 2026): the GBP 100 bn placeholder; book value is workbook v2.0's A1 (purchase_price_basis: book_value).
    "workbook_placeholder_gbp": "retired 26 Oct 2026 with purchase_price_basis: workbook_placeholder (F29)",
}
RETIRED_REPRICING_KEYS: dict[str, str] = {
    "eligibility": "the cfd_repricing_eligibility lever",
    "amortisation": "the cfd_amortisation lever",
    "sensitivities": "sensitivities.one_at_a_time",
}
SUPPORTED_CONSUMER_BENEFIT = frozenset({"unallocated_financing_saving", "routed_via_consumer_bill_link"})
REQUIRED_PATH_KEYS = ("source_workbook", "assumptions_snapshot", "crosswalk", "technology_costs")
SUPPORTED_STRIKE_BASIS = frozenset({"settled_bill_strike_table"})
SUPPORTED_SUPPORT_TYPE_LABELS = frozenset({"CfD", "merchant"})
SUPPORTED_ROUTE_CLASSIFICATIONS = frozenset({"capacity_market", "dispatchable_cfd"})
SUPPORTED_SOLAR_CFD_SHARE = frozenset({"network_labels"})
# Kept for callers that imported them; the values are the levers' now.
SUPPORTED_SURPLUS_RECYCLING = frozenset(LEVERS["surplus_recycling"].values)
SUPPORTED_SHORTFALL_FUNDING = frozenset(LEVERS["shortfall_funding"].values)
SUPPORTED_PURCHASE_PRICE_BASES = frozenset(LEVERS[LEVER_PURCHASE_PRICE_BASIS].values)
SUPPORTED_THERMAL_BASES = frozenset(LEVERS[LEVER_THERMAL_BASIS].values)
# The eligibility lever was retired on 29 Sep 2026 (review C1): contract vintage is the rule.
SUPPORTED_CFD_ELIGIBILITY = frozenset({"contract_vintage"})
SUPPORTED_CFD_AMORTISATION = frozenset(LEVERS["cfd_amortisation"].values)


def _fail(message: str) -> None:
    raise ValueError(f"public_ownership config: {message}")


def _require_int(config: Mapping, key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{key} must be an integer year, got {value!r}")
    return value


def _group_year(group: Mapping) -> int:
    """A group's ``model_year``: an integer year when given, :data:`DEFAULT_MODEL_YEAR` when absent (item Y4)."""
    if group.get("model_year") is None:
        return DEFAULT_MODEL_YEAR
    return _require_int(group, "model_year")


#: FES 2050 plan task Y14: a run group's own case list.  ``sensitivities`` takes the
#: schema of ``public_ownership.sensitivities`` (read against the shared
#: ``public_ownership.scenarios``); ``sensitivities_from`` names another group whose list
#: it shares (one level: that group must declare ``sensitivities`` itself).  A group with
#: neither takes the global list, so every 2030 group is unchanged.
GROUP_CASE_LIST_KEY = "sensitivities"
GROUP_CASE_LIST_FROM_KEY = "sensitivities_from"
#: Task Y14: a P2 balancing group per L4 share is keyed by the share it bids on.
GROUP_DEPLOYMENT_SHARE_KEY = LEVER_STATE_DEPLOYMENT_SHARE
#: Task Y14: a no-re-solve register bound of another group, on the same solved months.
GROUP_REGISTER_BOUND_KEY = "register_bound_of"
#: The group blocks a register bound may change (Y10 put both at group level; FES 2050 decision 71 adds the
#: auction margin, ``{level: none | high, strike_calibration: <file>}``: the bound's own strike table).
GROUP_AUCTION_MARGIN_KEY = "auction_margin"
REGISTER_BOUND_BLOCKS = ("fleet_vintages", "vintage_costs", "co_ownership_finance", GROUP_AUCTION_MARGIN_KEY,
                         # F28 (a): the legislation year (2025, "in force since 2025", is the bound on 2028).
                         "legislated_year")
#: The keys that define a group's solved months and bill; a bound must match its parent on each.
REGISTER_BOUND_SAME_KEYS = ("mode", "scenario", "scenario_pattern", "baseline_run_group")
#: Keys a group's own ``institution_comparison`` block may override over the global one.
GROUP_INSTITUTION_KEYS = frozenset({"central_scenario_pattern", "marginal_costs", "fti_settlement",
                                    "forecast_error_charge"})
#: Item F12 at the ownership layer (12 Oct 2026): a run group that re-accounts another group's solved
#: months at another wholesale price level (``price_level_of: <group>``), a *price-level twin*. Its
#: forecast case is the parent's case at that level (``price_level`` and ``price_level_parent`` on
#: the case), so every output exists at both levels under the two groups' own names.
GROUP_PRICE_LEVEL_OF_KEY = "price_level_of"
#: A price-level twin's reduced case list (user decision, 12 Oct 2026): ``{kinds: [...], cases: [...],
#: comparators: true}`` selects, from the full configured list in its order, every case whose catalogue
#: ``kind`` is listed (``scenario``, ``named``, ``preset``, ``bound`` ...) and every case named, plus (by
#: default) each kept case's private comparator. The settings of every kept case are the full list's.
GROUP_CASE_SELECTION_KEY = "case_selection"
CASE_SELECTION_KEYS = frozenset({"kinds", "cases", "comparators"})
#: The catalogue kinds (levers.expand_cases, levers.case_catalogue).
CASE_KINDS = frozenset({"scenario", "one_at_a_time", "named", "preset", "bound", "bound_comparator", "grid",
                        "bounding_pair"})
#: The keys a twin sets for itself; every other key must equal its parent's (same months, stack, bid case).
PRICE_LEVEL_TWIN_OWN_KEYS = frozenset({"forecast_case", GROUP_PRICE_LEVEL_OF_KEY, GROUP_CASE_SELECTION_KEY})
#: Network bounds (user decision, 18-19 Oct 2026): the network-delivery case a group's months were
#: solved on (``cp30``, ``no_acceleration``, ``no_upgrades``). The one-group-per-role rules (one
#: contract-value market group per year and price level, one bill group per bid stack, one group per
#: tariff bid case) hold per network case, and a balancing group is measured against the market group
#: of its own network case. A group without the key is on the network case ``None``, so a config that
#: declares none (every config before the bounds) resolves exactly as before.
GROUP_NETWORK_CASE_KEY = "network_case"
#: The group whose acquisition prices a group's register holds (``purchase_price_from: <group>``): the
#: legacy fleet is bought at the CENTRAL network's market value whatever network the year is solved on
#: (the bounds show network risk, not a different purchase). It must name a contract-value market group
#: of the same model year whose case list carries every case this group acquires with; a price-level
#: twin holds the source's twin at its own level.
GROUP_PURCHASE_PRICE_FROM_KEY = "purchase_price_from"
#: P1 and P2 under central dispatch (user decisions of 14 and 20 Oct 2026): a run group that re-projects a
#: contract-value market group's market-bill cases on that group's central-dispatch year (institution I3,
#: FTI's settlement), an *institution twin* (``institution_of: <market group>``). Like a price-level twin it
#: repeats its parent on every key but its own (the same months, so the national price, the demand and the
#: energy line are the parent's), with its own ``forecast_case`` (the parent's case under central dispatch:
#: ``institution: central_dispatch``) and its own ``case_selection``. Its cases are the market-bill cases on
#: the contract-value stack only: P3 has its own I3 path (the institution comparison) and P2's state bids
#: have no meaning without bids (decision 7). ``levers.dispatch_institution`` stays P3's.
GROUP_INSTITUTION_OF_KEY = "institution_of"
#: The keys an institution twin sets for itself; every other key must equal its parent's.
INSTITUTION_TWIN_OWN_KEYS = frozenset({"forecast_case", GROUP_INSTITUTION_OF_KEY, GROUP_CASE_SELECTION_KEY})
#: The institution an institution twin's forecast case declares (``institution``), and the keys it must carry.
INSTITUTION_CENTRAL_DISPATCH = "central_dispatch"
INSTITUTION_CASE_KEYS = ("institution_parent", "central_scenario_pattern", "fti_settlement")


def group_network_case(group: Mapping | None) -> str | None:
    """The network-delivery case a group's months were solved on (``network_case``), or None."""
    value = (group or {}).get(GROUP_NETWORK_CASE_KEY)
    return None if value is None else str(value)


def case_list_owner(config: Mapping, run_group: str | None) -> str | None:
    """The group whose case list ``run_group`` takes, or None for the global list (task Y14).

    A group with ``sensitivities`` owns its list; ``sensitivities_from`` points at
    the owner; a register bound without either takes its parent's list; every other
    group (every 2030 group) takes the global list.
    """
    if run_group is None:
        return None
    groups = config.get("reference_run_groups") or {}
    if run_group not in groups:
        _fail(f"reference_run_groups has no group {run_group!r}; configured: {sorted(groups)}")
    group = groups[run_group] or {}
    if GROUP_CASE_LIST_KEY in group:
        return run_group
    if group.get(GROUP_CASE_LIST_FROM_KEY) is not None:
        return str(group[GROUP_CASE_LIST_FROM_KEY])
    parent = group.get(GROUP_REGISTER_BOUND_KEY)
    if parent is not None and parent != run_group and parent in groups:
        return case_list_owner(config, str(parent))
    return None


def configured_group(config: Mapping, run_group: str | None) -> str | None:
    """``run_group`` if the config defines it, else None (the global case list).

    The stage scripts read their run group off an output's file name; a name for no
    configured group (a scratch or test path) takes the global list, as before task Y14.

    Task SETTLE2050 guard: a run group that a study declares (a ``config/scenarios.yaml``
    parent entry, decision 31) but this config does not is refused, because the config
    handed in is the defaults (``config/defaults/``) instead of the study's resolved config, and the
    global (2030) case list would be used silently (:func:`refuse_unresolved_study_group`).
    """
    if run_group in (config.get("reference_run_groups") or {}):
        return run_group
    refuse_unresolved_study_group(run_group)
    return None


def refuse_unresolved_study_group(run_group: str | None, config_dir=None) -> None:
    """Raise if ``run_group`` is a study's group (decision 31) that the config in hand does not declare.

    Args:
        run_group: The group a script was asked to account.
        config_dir: Where ``scenarios.yaml`` and ``defaults/`` live (default the repository's
            ``config``); a missing directory (a test or scratch tree) is not checked.

    Raises:
        ValueError: The group is declared by a study, so the script was handed the wrong config.
    """
    if run_group is None:
        return
    study = _study_of_group(str(run_group), str(config_dir or (Path(__file__).resolve().parents[2] / "config")))
    if study is not None:
        _fail(f"run group {run_group!r} is declared by the study {study!r} (config/scenarios.yaml, decision 31) "
              "but the config handed to this script does not declare it: pass the study's resolved config "
              f"(resources/public_ownership/{study}_finance_config.yaml), not the defaults (config/defaults/), or the global "
              "case list would be used silently")


@lru_cache(maxsize=8)
def _study_of_group(run_group: str, config_dir: str) -> str | None:
    from config.config_loader import defaults_files

    directory = Path(config_dir)
    if not (directory / "scenarios.yaml").exists() or not all(p.exists() for p in defaults_files(directory)):
        return None
    from scripts.public_ownership.study_config import load_studies

    return load_studies(directory, validate=False).study_of(run_group)


#: A group's ``sensitivities`` block holding this key extends the global list (:func:`extended_case_list`).
EXTEND_GLOBAL_KEY = "extend_global"


def _entry_lever(entry) -> str:
    """The lever a structural or one-at-a-time entry names (a bare name or ``{name: [values]}``)."""
    return next(iter(entry)) if isinstance(entry, Mapping) else str(entry)


def _moves_dropped(spec, drop_levers: set) -> bool:
    return isinstance(spec, Mapping) and any(k in drop_levers for k in spec)


def extended_case_list(global_block: Mapping, group_block: Mapping) -> dict:
    """The global sensitivities with a group's year-specific edits (user decision, 10 Nov 2026).

    ``extend_global: {drop_levers: [...], drop_named: [...], drop_preset_parents: [...]}`` removes, from every
    part of the global list, the levers the group's year cannot carry (one at a time, structural, bounds and grid
    axes) and the named cases (and presets on parents) it cannot run. The group's own ``one_at_a_time`` entries
    replace a global entry on the same lever or are added; its ``named`` cases are added; any other key the group
    sets replaces the global one.
    """
    import copy

    edit = dict(group_block.get(EXTEND_GLOBAL_KEY) or {})
    drop_levers = set(edit.get("drop_levers") or [])
    drop_named = set(edit.get("drop_named") or [])
    drop_parents = set(edit.get("drop_preset_parents") or [])
    # {parent: [axis token or selector, ...]}: a bound grid kept to these axes only (user, 10 Nov 2026: the 2050
    # Scenario 3 bound grid on its four largest drivers, 24 positions instead of 3,456).
    keep_axes = {k: set(v) for k, v in dict(edit.get("bound_grid_axes") or {}).items()}
    out = copy.deepcopy(dict(global_block))
    for key in ("structural", "one_at_a_time"):
        out[key] = [e for e in (out.get(key) or []) if _entry_lever(e) not in drop_levers]
    named = {k: v for k, v in (out.get("named") or {}).items() if k not in drop_named and not _moves_dropped(v, drop_levers)}
    out["named"] = named
    presets = dict(out.get("presets") or {})
    for name, preset in list(presets.items()):
        if not isinstance(preset, Mapping):
            continue
        preset = dict(preset)
        if isinstance(preset.get("applies_to"), list):
            preset["applies_to"] = [c for c in preset["applies_to"] if c not in drop_parents]
        if isinstance(preset.get("levers_by_case"), Mapping):
            preset["levers_by_case"] = {k: v for k, v in preset["levers_by_case"].items() if k not in drop_parents}
        presets[name] = preset
    out["presets"] = presets
    bounds = dict(out.get("bounds") or {})
    for side in ("best", "worst"):
        if isinstance(bounds.get(side), Mapping):
            bounds[side] = {k: v for k, v in bounds[side].items() if k not in drop_levers}
    out["bounds"] = bounds
    for grids_key in ("toggle_grids",):
        grids = dict(out.get(grids_key) or {})
        for gkey in ("grids", "bound_grids"):
            if isinstance(grids.get(gkey), Mapping):
                grids[gkey] = {parent: [ax for ax in axes if ax.get("lever") not in drop_levers
                                        and (gkey != "bound_grids" or parent not in keep_axes
                                             or (ax.get("token") or ax.get("selector")) in keep_axes[parent])]
                               for parent, axes in grids[gkey].items()}
        out[grids_key] = grids
    own_one = list(group_block.get("one_at_a_time") or [])
    own_levers = {_entry_lever(e) for e in own_one}
    out["one_at_a_time"] = [e for e in out["one_at_a_time"] if _entry_lever(e) not in own_levers] + own_one
    out["structural"] = [e for e in out["structural"] if _entry_lever(e) not in own_levers]
    out["named"] = {**out["named"], **dict(group_block.get("named") or {})}
    for key, value in group_block.items():
        if key not in (EXTEND_GLOBAL_KEY, "one_at_a_time", "named"):
            out[key] = copy.deepcopy(value)
    return out


def configured_case_specs(config: Mapping, run_group: str | None = None) -> list[CaseSpec]:
    """The case list this config asks for: scenarios, sensitivities, bounding pair (item K1).

    ``run_group`` (task Y14) takes that group's own list when it declares one
    (:func:`case_list_owner`); without it, or for a group with no list of its own,
    the global ``public_ownership.sensitivities`` list, exactly as before.
    """
    owner = case_list_owner(config, run_group)
    if owner is None:
        block, where = config.get("sensitivities"), None
    else:
        groups = config.get("reference_run_groups") or {}
        if owner not in groups or GROUP_CASE_LIST_KEY not in (groups[owner] or {}):
            _fail(f"reference_run_groups.{run_group}.{GROUP_CASE_LIST_FROM_KEY}={owner!r} must name a group that "
                  f"declares its own {GROUP_CASE_LIST_KEY}")
        block, where = groups[owner][GROUP_CASE_LIST_KEY], f"reference_run_groups.{owner}.{GROUP_CASE_LIST_KEY}"
    if owner is not None and isinstance(block, Mapping) and EXTEND_GLOBAL_KEY in block:
        # User decision (10 Nov 2026, "full 2030 set" in 2050): the group's list is the global one, less the
        # levers and named cases its year cannot carry, plus its own.
        block = extended_case_list(config.get("sensitivities") or {}, block)
    try:
        bundles = parse_scenarios(config.get("scenarios"))
        sensitivities = parse_sensitivities(block, bundles)
        specs = expand_cases(bundles, sensitivities)
    except LeverConfigError as error:
        _fail(str(error) if where is None else f"{where}: {error}")
    selection = group_case_selection(config, run_group)
    if selection is not None:
        # Item F12 (12 Oct 2026): a price-level twin's reduced list (every model-price group: None).
        specs = select_cases(specs, selection, where=f"reference_run_groups.{run_group}.{GROUP_CASE_SELECTION_KEY}")
    if run_group is not None and institution_parent_group(config, run_group) is not None:
        # An institution twin (20 Oct 2026) projects the market-bill cases on the contract-value stack only.
        specs = institution_twin_cases(specs)
    return specs


def institution_twin_cases(specs: list[CaseSpec]) -> list[CaseSpec]:
    """The cases an institution twin re-projects under central dispatch: market-bill cases on the contract-value stack.

    Cases that acquire the legacy fleet (P3, the acquisition tariff) or take
    ``dispatch_institution`` other than the market have their own I3 path (the
    institution comparison, item N10); a case on another balancing-bid stack (P2's
    state bids at cost) has no meaning without bids (user decision 7, 20 Oct 2026).
    Each kept case's private comparator is a market-bill case too, so the list stays
    closed under its comparators.
    """
    return [spec for spec in specs if spec.market_bill()
            and spec.settings.get(LEVER_BALANCING_BIDS, BIDS_CONTRACT_VALUE) == BIDS_CONTRACT_VALUE]


def group_case_selection(config: Mapping, run_group: str | None) -> Mapping | None:
    """A configured group's ``case_selection`` (a price-level twin's reduced case list), else None."""
    if run_group is None:
        return None
    group = (config.get("reference_run_groups") or {}).get(run_group) or {}
    selection = group.get(GROUP_CASE_SELECTION_KEY)
    return None if selection is None else selection


def case_kind(spec: CaseSpec) -> str:
    """A case's catalogue kind (``levers.case_catalogue``)."""
    return str((spec.meta or {}).get("kind", "scenario" if spec.variant is None else "case"))


def select_cases(specs: list[CaseSpec], selection: Mapping, where: str = GROUP_CASE_SELECTION_KEY) -> list[CaseSpec]:
    """The cases ``selection`` keeps from ``specs``, in the full list's order, each with its full-list settings.

    ``kinds`` keeps every case of those catalogue kinds; ``cases`` names cases (an unknown id is
    refused); ``comparators`` (default true) adds each kept case's private comparator, found on the
    full list (house rule 3: ``comparator_for_settings``). With ``comparators: false`` a kept case whose
    comparator is not kept is refused, because ``comparator_for_settings`` would silently fall back to
    ``private``. On the kept list every case's comparator is the full list's.
    """
    from scripts.public_ownership.levers import comparator_for_settings

    if not isinstance(selection, Mapping):
        _fail(f"{where} must be a mapping with {sorted(CASE_SELECTION_KEYS)}")
    unknown_keys = sorted(set(selection) - CASE_SELECTION_KEYS)
    if unknown_keys:
        _fail(f"{where} has unknown keys {unknown_keys}; expected {sorted(CASE_SELECTION_KEYS)}")
    kinds = [str(k) for k in (selection.get("kinds") or [])]
    if unknown := sorted(set(kinds) - CASE_KINDS):
        _fail(f"{where}.kinds names unknown kinds {unknown}; supported: {sorted(CASE_KINDS)}")
    named = [str(c) for c in (selection.get("cases") or [])]
    by_id = {spec.case_id: spec for spec in specs}
    if unknown := [c for c in named if c not in by_id]:
        _fail(f"{where}.cases names cases the config does not expand: {unknown[:5]}")
    comparators = selection.get("comparators", True)
    if not isinstance(comparators, bool):
        _fail(f"{where}.comparators must be true or false")
    kept = {spec.case_id for spec in specs if case_kind(spec) in kinds} | set(named)
    if not kept:
        _fail(f"{where} keeps no case")
    settings = {spec.case_id: dict(spec.settings) for spec in specs}
    comparator_of = lambda case: comparator_for_settings(case, settings)  # noqa: E731
    if comparators:
        while True:
            wider = kept | {comparator_of(case) for case in kept}
            if wider == kept:
                break
            kept = wider
    missing = sorted({comparator_of(case) for case in kept} - kept)
    if missing:
        _fail(f"{where}: kept cases are measured against private comparators it drops ({missing[:5]}); keep them "
              "or set comparators: true")
    out = [spec for spec in specs if spec.case_id in kept]
    kept_settings = {spec.case_id: dict(spec.settings) for spec in out}
    drift = [case for case in kept if comparator_for_settings(case, kept_settings) != comparator_of(case)]
    if drift:
        _fail(f"{where}: the comparator of {drift[:5]} differs on the kept list")
    return out


def configured_case_catalogue(config: Mapping, run_group: str | None = None) -> dict[str, dict]:
    """Where every configured case came from (kind, parent, preset, bound, grid toggles), for the summaries."""
    from scripts.public_ownership.levers import case_catalogue

    return case_catalogue(configured_case_specs(config, run_group))


def expanded_cases(config: Mapping, run_group: str | None = None) -> list[str]:
    """Every case id, in accounting order.  The only place the list is built.

    ``run_group`` (task Y14) gives that group's own list when it declares one; the
    default, and every group without one, is the global list.
    """
    return [spec.case_id for spec in configured_case_specs(config, run_group)]


def market_bill_cases(config: Mapping, run_group: str | None = None) -> list[str]:
    """Cases projected through the market bill: no legacy acquisition, market dispatch."""
    return [spec.case_id for spec in configured_case_specs(config, run_group) if spec.market_bill()]


def tariff_cases(config: Mapping, run_group: str | None = None) -> list[str]:
    """Cases that acquire the legacy fleet and take the acquisition tariff (item N5) instead."""
    return [spec.case_id for spec in configured_case_specs(config, run_group)
            if spec.acquires_legacy() and spec.settings.get(LEVER_DISPATCH_INSTITUTION) == INSTITUTION_MARKET]


def institution_cases(config: Mapping, run_group: str | None = None) -> list[str]:
    """Cases under the nationalised-dispatch institution (item N10), a second solved run group."""
    return [spec.case_id for spec in configured_case_specs(config, run_group)
            if spec.settings.get(LEVER_DISPATCH_INSTITUTION) != INSTITUTION_MARKET]


def institution_market_twin(config: Mapping, run_group: str | None = None) -> tuple[str, str] | None:
    """The institution case and its I1 twin: the same case with the market machinery kept.

    The institution comparison moves four measured lines off one acquisition
    tariff, so there is exactly one institution case and its twin must be a
    configured tariff case that differs from it in ``dispatch_institution`` alone.
    ``run_group`` (task Y14) reads that group's own case list.
    """
    specs = configured_case_specs(config, run_group)
    institution = [spec for spec in specs
                   if spec.settings.get(LEVER_DISPATCH_INSTITUTION) != INSTITUTION_MARKET]
    if not institution:
        return None
    if len(institution) != 1:
        _fail(f"the institution comparison carries one I1/I3 pair; {len(institution)} cases take central dispatch: "
              f"{[spec.case_id for spec in institution]}")
    case = institution[0]
    wanted = dict(case.settings, **{LEVER_DISPATCH_INSTITUTION: INSTITUTION_MARKET})
    twins = [spec.case_id for spec in specs if dict(spec.settings) == wanted]
    if len(twins) != 1:
        _fail(f"{case.case_id} needs exactly one configured case identical but for dispatch_institution: market; "
              f"found {twins}")
    return case.case_id, twins[0]


def scenario_ids(config: Mapping) -> list[str]:
    return list((config.get("scenarios") or {}).keys())


def _validate_consumer_bill(block: Mapping, groups: Mapping, forecast_cases: Mapping | None) -> None:
    """Check the bill-link decisions (plan section 23: decisions 1, 5, 6 and 13)."""
    if not isinstance(block, Mapping) or not block:
        _fail("consumer_bill must be a mapping recording the Release 1 decisions")
    forecast_case = block.get("forecast_case")
    if not forecast_case or not isinstance(forecast_case, str):
        _fail("consumer_bill.forecast_case must name a consumer_bill.forecasts case")
    repricing = block.get("cfd_repricing") or {}
    for key, pointer in RETIRED_REPRICING_KEYS.items():
        if key in repricing:
            _fail(f"consumer_bill.cfd_repricing.{key} is retired; it is {pointer} (item K1)")
    _require_int(repricing, "eligible_award_years_from")
    if not repricing.get("award_years"):
        _fail("consumer_bill.cfd_repricing.award_years must point to the round award-year table")
    term = _require_int(repricing, "contract_term_years")
    if term <= 0:
        _fail("consumer_bill.cfd_repricing.contract_term_years must be positive")
    # Round-specific terms (audit 26 Sep 2026: AR7 onwards is 20 years), checked
    # at parse time by the same function the re-pricing uses.
    from scripts.public_ownership.cfd_repricing import contract_terms_by_round
    try:
        contract_terms_by_round(repricing)
    except ValueError as exc:
        _fail(f"consumer_bill.{exc}")
    if repricing.get("strike_basis") not in SUPPORTED_STRIKE_BASIS:
        _fail(f"consumer_bill.cfd_repricing.strike_basis={repricing.get('strike_basis')!r} is not implemented; "
              f"supported: {sorted(SUPPORTED_STRIKE_BASIS)}")
    ldes = block.get("ldes_cap_and_floor") or {}
    if ldes and not isinstance(ldes.get("enabled"), bool):
        _fail("consumer_bill.ldes_cap_and_floor.enabled must be true or false")
    if ldes.get("enabled") and not list(ldes.get("carriers") or []):
        _fail("consumer_bill.ldes_cap_and_floor.carriers must list the LDES carriers")
    if ldes.get("enabled") and ldes.get("recovery_denominator", "final_demand_mwh") != "final_demand_mwh":
        _fail("consumer_bill.ldes_cap_and_floor.recovery_denominator: only final_demand_mwh is implemented")
    floor = ldes.get("floor_return") or {}
    if floor and floor.get("basis") != "cost_of_debt_like":
        _fail("consumer_bill.ldes_cap_and_floor.floor_return.basis: only cost_of_debt_like is implemented")
    for case_id, rate in (floor.get("real_rate_by_case") or {}).items():
        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate < 1):
            _fail(f"consumer_bill.ldes_cap_and_floor.floor_return.real_rate_by_case.{case_id} must be a rate in (0, 1) or null")
    cm = block.get("capacity_market_sensitivity") or {}
    if cm and not isinstance(cm.get("enabled"), bool):
        _fail("consumer_bill.capacity_market_sensitivity.enabled must be true or false")
    corrections = block.get("label_corrections") or {}
    for carrier, label in (corrections.get("support_type") or {}).items():
        if label not in SUPPORTED_SUPPORT_TYPE_LABELS:
            _fail(f"consumer_bill.label_corrections.support_type.{carrier}={label!r}; "
                  f"supported: {sorted(SUPPORTED_SUPPORT_TYPE_LABELS)}")
    for carrier, route in (corrections.get("route_classification") or {}).items():
        if route not in SUPPORTED_ROUTE_CLASSIFICATIONS:
            _fail(f"consumer_bill.label_corrections.route_classification.{carrier}={route!r}; "
                  f"supported: {sorted(SUPPORTED_ROUTE_CLASSIFICATIONS)}")
    if corrections.get("solar_cfd_share") not in SUPPORTED_SOLAR_CFD_SHARE:
        _fail(f"consumer_bill.label_corrections.solar_cfd_share={corrections.get('solar_cfd_share')!r}; "
              f"supported: {sorted(SUPPORTED_SOLAR_CFD_SHARE)}")
    if forecast_cases is None:
        return
    case = forecast_cases.get(forecast_case)
    if case is None:
        _fail(f"consumer_bill.forecast_case={forecast_case!r} is not a consumer_bill.forecasts case; "
              f"configured: {sorted(forecast_cases)}")
    pattern = str(case.get("source_scenario_pattern", ""))
    matching = [
        name for name, group in groups.items()
        if group_model_year(group) == int(case["modelled_year"])
        and group.get("mode") == "twelve_months" and str(group.get("scenario_pattern")) == pattern
    ]
    if not matching:
        _fail(f"no reference_run_group shares model_year {case['modelled_year']} and "
              f"source_scenario_pattern {pattern!r} with forecast case {forecast_case!r}; the private "
              "case could not reproduce that projection")


def _positive_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _validate_acquisition(acquisition: Mapping, specs: list[CaseSpec]) -> None:
    """Items N2 and N4: the acquisition parameters the levers of the acquiring cases need."""
    if not isinstance(acquisition, Mapping) or not acquisition:
        _fail("a case acquires the legacy fleet, so public_ownership.acquisition must be set")
    for key, pointer in RETIRED_ACQUISITION_KEYS.items():
        if key in acquisition:
            _fail(f"acquisition.{key} is retired; {pointer} (item K1)")
    # User decision 10: HPC completed by the state needs its sourced completion cost.
    if any(spec.settings.get("in_construction") == "public_completion" for spec in specs) and not acquisition.get(
            "hpc_completion"):
        _fail("a case takes in_construction: public_completion, so acquisition.hpc_completion must name the HPC "
              "completion cost table")
    if any(spec.settings.get("interconnector_price", "central") != "central" for spec in specs):
        rates = ((acquisition.get("interconnectors") or {}).get("owner_real_rate_by_setting") or {})
        wanted = {spec.settings["interconnector_price"] for spec in specs} - {"central"}
        if not wanted <= set(rates):
            _fail(f"acquisition.interconnectors.owner_real_rate_by_setting must give {sorted(wanted)}")
    if any(spec.settings.get("acquisition_transaction_costs", "none") != "none" for spec in specs):
        shares = acquisition.get("transaction_cost_share") or {}
        wanted = {spec.settings["acquisition_transaction_costs"] for spec in specs} - {"none"}
        if not wanted <= set(shares):
            _fail(f"acquisition.transaction_cost_share must give {sorted(wanted)}")
    if any(spec.settings.get("legacy_asset_lives", "central") != "central" for spec in specs) and not acquisition.get(
            "extended_asset_lives"):
        _fail("a case takes legacy_asset_lives: extended, so acquisition.extended_asset_lives must be set")


def _validate_state_operating_overhead(block: Mapping | None) -> None:
    """Item OH1: the state's holding-company cost, its twins and its scope.

    Three rates and a component scope, all enforced.  ``applies_to`` must stay
    inside generation and storage: the acquired networks keep their RIIO allowed
    opex in the network finance account, so charging an overhead on them would
    double-count that cost against the same assets.
    """
    if block is None:
        return
    if not isinstance(block, Mapping):
        _fail("state_operating_overhead must be a mapping (item OH1)")
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        _fail("state_operating_overhead.enabled must be true or false")
    unknown = sorted(set(block) - ({"enabled", "applies_to", "unstaffed_storage"}
                                   | set(STATE_OVERHEAD_RATE_KEYS.values())))
    if unknown:
        _fail(f"state_operating_overhead has unknown keys {unknown}; expected enabled, applies_to, "
              f"unstaffed_storage and {sorted(STATE_OVERHEAD_RATE_KEYS.values())}")
    storage = block.get("unstaffed_storage")
    if storage is not None:
        # F19 decision 6: the unstaffed-storage rates, their twins and their carriers.
        if not isinstance(storage, Mapping):
            _fail("state_operating_overhead.unstaffed_storage must be a mapping (F19 decision 6)")
        allowed = {"carriers", "in_vintaged_registers"} | set(STATE_OVERHEAD_RATE_KEYS.values())
        if sorted(set(storage) - allowed):
            _fail(f"state_operating_overhead.unstaffed_storage has unknown keys {sorted(set(storage) - allowed)}")
        carriers = storage.get("carriers")
        if carriers is not None and (isinstance(carriers, str) or not list(carriers)):
            _fail("state_operating_overhead.unstaffed_storage.carriers must be a non-empty list of carriers")
        if not isinstance(storage.get("in_vintaged_registers", False), bool):
            _fail("state_operating_overhead.unstaffed_storage.in_vintaged_registers must be true or false")
        storage_rates = {value: storage.get(key) for value, key in STATE_OVERHEAD_RATE_KEYS.items()}
        for value, rate in storage_rates.items():
            if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0):
                _fail(f"state_operating_overhead.unstaffed_storage.{STATE_OVERHEAD_RATE_KEYS[value]} must be a "
                      f"non-negative GBP/kW/yr rate, got {rate!r}")
        if all(rate is not None for rate in storage_rates.values()) and not (
                storage_rates[OVERHEAD_LOW] <= storage_rates[OVERHEAD_CENTRAL] <= storage_rates[OVERHEAD_HIGH]):
            _fail("state_operating_overhead.unstaffed_storage: the twins must bracket the central")
    for value, key in STATE_OVERHEAD_RATE_KEYS.items():
        rate = block.get(key)
        if rate is None:
            continue
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0:
            _fail(f"state_operating_overhead.{key} must be a non-negative GBP/kW/yr rate, got {rate!r}")
    rates = {value: block.get(key) for value, key in STATE_OVERHEAD_RATE_KEYS.items()}
    if all(rate is not None for rate in rates.values()):
        low, central, high = rates[OVERHEAD_LOW], rates[OVERHEAD_CENTRAL], rates[OVERHEAD_HIGH]
        if not low <= central <= high:
            _fail(f"state_operating_overhead: the twins must bracket the central, got low={low}, "
                  f"central={central}, high={high}")
    applies_to = block.get("applies_to")
    if applies_to is not None:
        applies_to = list(applies_to)
        if not applies_to or not set(applies_to) <= SUPPORTED_COMPONENT_TYPES:
            _fail("state_operating_overhead.applies_to must be a non-empty subset of "
                  f"{sorted(SUPPORTED_COMPONENT_TYPES)}; the acquired networks keep their RIIO allowed opex "
                  "in network_account, so an overhead there would double-count")


def validate_public_ownership_config(config: Mapping, forecast_cases: Mapping | None = None) -> Mapping:
    """Check the ``public_ownership`` section against what the code implements.

    ``forecast_cases`` (``consumer_bill.forecasts.cases``) enables the
    parse-time check that the bill-link forecast case and a reference-run group
    share one ``source_scenario_pattern``.
    """
    if not isinstance(config, Mapping) or not config:
        _fail("section is missing or empty")
    for key in REQUIRED_PATH_KEYS:
        if not config.get(key):
            _fail(f"{key} must be set")
    _require_int(config, "price_year")
    policy_start_year = _require_int(config, "policy_start_year")

    for key, pointer in RETIRED_KEYS.items():
        if key in config:
            _fail(f"{key} is retired; {pointer}")
    if "existing_support_contracts" in config and config["existing_support_contracts"] != "unchanged":
        _fail("existing_support_contracts: only 'unchanged' is implemented")

    # Item K1: the scenarios and sensitivities.  An unsupported lever value
    # fails here, at Snakemake parse time, before any job starts.
    specs = configured_case_specs(config)
    share = config.get("state_stake_share")
    if share is not None and not (isinstance(share, (int, float)) and not isinstance(share, bool) and 0 < share <= 1):
        _fail("state_stake_share must be null (the workbook value) or a share in (0, 1]")

    scope = list(config.get("ownership_scope") or [])
    if not scope or not set(scope) <= SUPPORTED_COMPONENT_TYPES:
        _fail(f"ownership_scope must be a non-empty subset of {sorted(SUPPORTED_COMPONENT_TYPES)}")

    _validate_state_operating_overhead(config.get("state_operating_overhead"))
    # Review 29 Sep 2026: the explorer's lever parameters and the auction calibration.
    from scripts.public_ownership.calculate_financing_scenarios import lever_parameters

    try:
        lever_parameters(config.get("lever_parameters"))
    except ValueError as error:
        _fail(str(error))
    calibration = config.get("cost_calibration")
    if calibration is not None:
        if not isinstance(calibration, Mapping) or set(calibration) - {"enabled", "table", "note", "derived_from_group"}:
            _fail("cost_calibration must be {enabled, table[, derived_from_group]}")
        if calibration.get("enabled") and not calibration.get("table"):
            _fail("cost_calibration.enabled needs cost_calibration.table")
        derived = calibration.get("derived_from_group")
        if derived is not None and derived not in (config.get("reference_run_groups") or {}):
            _fail(f"cost_calibration.derived_from_group={derived!r} must name a reference run group")

    acquisition = config.get("acquisition") or {}
    acquiring = [spec for spec in specs if spec.acquires_legacy()]
    if acquiring:
        _validate_acquisition(acquisition, acquiring)
    # Item N0: the valuation settings are enforced like every other decision.
    from scripts.public_ownership.legacy_valuation import settings_from_config

    try:
        settings_from_config((acquisition or {}).get("valuation"))
    except ValueError as error:
        _fail(str(error))
    # Item N6: the network account settings are enforced whenever a case touches the wires.
    if any(spec.settings[LEVER_NETWORKS] != NETWORKS_PRIVATE for spec in specs):
        from scripts.public_ownership.network_finance_account import settings as network_settings

        try:
            network_settings(config.get("network_account"))
        except ValueError as error:
            _fail(str(error))
    # Item N10: the institution comparison spans a second solved run group.
    if institution_cases(config):
        institution_market_twin(config)
        block = config.get("institution_comparison") or {}
        pattern = str(block.get("central_scenario_pattern", ""))
        if "{month}" not in pattern:
            _fail("institution_comparison.central_scenario_pattern must contain '{month}' when a case takes "
                  "dispatch_institution: nationalised_dispatch")
        if not block.get("marginal_costs"):
            _fail("institution_comparison.marginal_costs must point at the marginal-cost breakdown CSV")

    treatment = (config.get("consumer_benefit") or {}).get("treatment")
    if treatment not in SUPPORTED_CONSUMER_BENEFIT:
        _fail(f"consumer_benefit.treatment={treatment!r} is not implemented; "
              f"supported: {sorted(SUPPORTED_CONSUMER_BENEFIT)}")
    if treatment == "routed_via_consumer_bill_link" and not config.get("consumer_bill"):
        _fail("consumer_benefit.treatment=routed_via_consumer_bill_link needs the consumer_bill block")

    groups = config.get("reference_run_groups") or {}
    if not isinstance(groups, Mapping) or not groups:
        _fail("reference_run_groups must define at least one group")
    for name, group in groups.items():
        if not isinstance(group, Mapping):
            _fail(f"reference_run_groups.{name} must be a mapping")
        # Item Y4: a group's model year is required to be an integer when given and
        # defaults to DEFAULT_MODEL_YEAR (the CP30 year) when absent.
        model_year = _group_year(group)
        mode = group.get("mode")
        if mode not in MODES:
            _fail(f"reference_run_groups.{name}.mode={mode!r}; supported: {sorted(MODES)}")
        if mode == "full_year" and not group.get("scenario"):
            _fail(f"reference_run_groups.{name} needs 'scenario' for full_year mode")
        if mode == "twelve_months" and "{month}" not in str(group.get("scenario_pattern", "")):
            _fail(f"reference_run_groups.{name} needs a scenario_pattern containing '{{month}}'")
        bids = group.get("balancing_bids", BIDS_CONTRACT_VALUE)
        if bids not in LEVERS[LEVER_BALANCING_BIDS].values:
            _fail(f"reference_run_groups.{name}.balancing_bids={bids!r}; "
                  f"supported: {list(LEVERS[LEVER_BALANCING_BIDS].values)}")
        own_case = group.get("forecast_case")
        if own_case is not None and forecast_cases is not None and own_case not in forecast_cases:
            _fail(f"reference_run_groups.{name}.forecast_case={own_case!r} is not a consumer_bill.forecasts case")
        if (GROUP_CASE_SELECTION_KEY in group and group.get(GROUP_PRICE_LEVEL_OF_KEY) is None
                and group.get(GROUP_INSTITUTION_OF_KEY) is None):
            # Network bounds (user decision, 18-19 Oct 2026): the LOWER and UPPER networks' groups carry
            # the headline cases only, at model price too. 2030 (unvintaged) groups only.
            from scripts.public_ownership.model_year import is_vintaged_model_year

            if is_vintaged_model_year(model_year):
                _fail(f"reference_run_groups.{name}.{GROUP_CASE_SELECTION_KEY}: a reduced case list is implemented "
                      "for unvintaged (2030) groups only")
            configured_case_specs(config, name)        # the selection parses, keeps known cases, and is closed
        if group.get(GROUP_NETWORK_CASE_KEY) is not None and not isinstance(group.get(GROUP_NETWORK_CASE_KEY), str):
            _fail(f"reference_run_groups.{name}.{GROUP_NETWORK_CASE_KEY} must be a name")
        if group.get("institution_comparison") is not None:
            override = group["institution_comparison"]
            if not isinstance(override, Mapping) or set(override) - GROUP_INSTITUTION_KEYS:
                _fail(f"reference_run_groups.{name}.institution_comparison may set only "
                      f"{sorted(GROUP_INSTITUTION_KEYS)}")
            if institution_cases(config, name):
                pattern = str(institution_block(config, name).get("central_scenario_pattern", ""))
                if "{month}" not in pattern:
                    _fail(f"reference_run_groups.{name}: institution_comparison.central_scenario_pattern must "
                          "contain '{month}'")
        if "bid_case" in group:
            _validate_bid_case(config, name, group, bids)
        _validate_vintaged_group(name, group, model_year)
        # Task Y14: a group's own case list, its L4 share key and its register bound.
        _validate_group_case_list(config, name, group, model_year)
        if GROUP_DEPLOYMENT_SHARE_KEY in group:
            _validate_share_group(config, name, group, model_year, bids)
        if GROUP_REGISTER_BOUND_KEY in group:
            _validate_register_bound(config, name, group)
        elif GROUP_AUCTION_MARGIN_KEY in group:
            _fail(f"reference_run_groups.{name}.{GROUP_AUCTION_MARGIN_KEY}: only a register bound may set its own "
                  "auction margin (FES 2050 decision 71); the central margin is the study's vintage_capital_2050")
        baseline = group.get("baseline_run_group")
        if baseline is not None:
            if baseline == name or baseline not in groups:
                _fail(f"reference_run_groups.{name}.baseline_run_group={baseline!r} must name another group")
            baseline_year = _group_year(groups[baseline])
            if baseline_year >= model_year:
                _fail(f"reference_run_groups.{name}: baseline {baseline!r} ({baseline_year}) "
                      f"must be an earlier model year than {model_year}")
            if baseline_year >= policy_start_year:
                _fail(f"reference_run_groups.{name}: baseline {baseline!r} ({baseline_year}) must "
                      f"predate policy_start_year {policy_start_year} to define the legacy fleet")
    if "consumer_bill" in config:
        _validate_consumer_bill(config["consumer_bill"], groups, forecast_cases)
        _validate_balancing_groups(config, specs, forecast_cases)
    if price_level_twin_groups(config):
        _validate_price_level_twins(config, forecast_cases)
    if institution_twin_groups(config):
        _validate_institution_twins(config, forecast_cases)
    _validate_purchase_price_sources(config, forecast_cases)
    return config


#: FES 2050 plan task Y10: the keys a vintaged group's ``fleet_vintages`` and ``vintage_costs`` blocks take.
FLEET_VINTAGE_KEYS = frozenset({"pathway", "retirement_rule", "life_source", "join_mode", "site_vintages",
                                # FES 2050 decision 48: {stock-flow technology: life} over life_source.
                                "life_overrides",
                                # FES 2050 decision 62: the repowering rule, the evidence exclusions, and
                                # the technologies whose repowered capacity takes no CfD.
                                "repowering", "evidence_exclude_countries", "evidence_stale_records",
                                "no_cfd_on_repowering",
                                # F28 (d): hydro and pumped hydro refurbished under the strict rule.
                                "refurbishment"})
VINTAGE_COST_KEYS = frozenset({"enabled", "table", "insurance", "learning",
                               # F28 (d): the refurbished share's capex, by stock-flow technology.
                               "refurbishment_capex_share"})


def _validate_vintaged_group(name: str, group: Mapping, model_year: int) -> None:
    """Task Y10: ``fleet_vintages`` and ``vintage_costs`` belong to vintaged model years (2040 on) only.

    Both are optional; a group below the vintaged years that names either is
    refused, so no 2030 register can be vintaged or vintage-costed.
    """
    from scripts.public_ownership.fleet_vintages import JOIN_MODES, LIFE_SOURCES, RETIREMENT_RULES
    from scripts.public_ownership.model_year import VINTAGED_REGISTER_FROM_MODEL_YEAR, is_vintaged_model_year
    from scripts.public_ownership.vintage_costs import INSURANCE_MODES

    for key, allowed in (("fleet_vintages", FLEET_VINTAGE_KEYS), ("vintage_costs", VINTAGE_COST_KEYS)):
        if key not in group:
            continue
        block = group[key]
        if not isinstance(block, Mapping):
            _fail(f"reference_run_groups.{name}.{key} must be a mapping")
        if not is_vintaged_model_year(model_year):
            _fail(f"reference_run_groups.{name}.{key} applies from model year {VINTAGED_REGISTER_FROM_MODEL_YEAR}; "
                  f"the group is {model_year}")
        unknown = sorted(set(block) - allowed)
        if unknown:
            _fail(f"reference_run_groups.{name}.{key} has unknown keys {unknown}; expected {sorted(allowed)}")
    fleet = group.get("fleet_vintages") or {}
    if fleet.get("retirement_rule", "economic_life_with_extension") not in RETIREMENT_RULES:
        _fail(f"reference_run_groups.{name}.fleet_vintages.retirement_rule; supported: {sorted(RETIREMENT_RULES)}")
    if fleet.get("life_source", "central") not in LIFE_SOURCES:
        _fail(f"reference_run_groups.{name}.fleet_vintages.life_source; supported: {list(LIFE_SOURCES)}")
    if fleet.get("join_mode", "pro_rata") not in JOIN_MODES:
        _fail(f"reference_run_groups.{name}.fleet_vintages.join_mode; supported: {list(JOIN_MODES)}")
    overrides = fleet.get("life_overrides")
    if overrides:  # only a vintaged group that sets it (none in 2030) is checked
        if not isinstance(overrides, Mapping):
            _fail(f"reference_run_groups.{name}.fleet_vintages.life_overrides must map a stock-flow technology "
                  "to a life")
        try:
            from scripts.public_ownership.fleet_vintages import load_lives

            load_lives(source=fleet.get("life_source", "central"), overrides=overrides)
        except (TypeError, ValueError) as exc:
            _fail(f"reference_run_groups.{name}.fleet_vintages.life_overrides: {exc}")
    if fleet.get("repowering"):  # decision 62; no 2030 group sets it
        try:
            from scripts.public_ownership.fleet_vintages import parse_repowering

            parse_repowering(fleet["repowering"])
        except (TypeError, ValueError) as exc:
            _fail(f"reference_run_groups.{name}.fleet_vintages.repowering: {exc}")
    if fleet.get("refurbishment"):  # F28 (d)
        from scripts.public_ownership.fleet_vintages import parse_refurbishment

        try:
            refurbishment = parse_refurbishment(fleet["refurbishment"])
        except (TypeError, ValueError, KeyError) as exc:
            _fail(f"reference_run_groups.{name}.fleet_vintages.refurbishment: {exc}")
        shares = (group.get("vintage_costs") or {}).get("refurbishment_capex_share") or {}
        if missing := sorted(set(refurbishment.technologies) - set(shares)):
            _fail(f"reference_run_groups.{name}: refurbished {missing} need vintage_costs.refurbishment_capex_share")
    costs = group.get("vintage_costs") or {}
    if costs.get("insurance", "exclude") not in INSURANCE_MODES:
        _fail(f"reference_run_groups.{name}.vintage_costs.insurance; supported: {list(INSURANCE_MODES)}")
    if costs.get("insurance", "exclude") == "include":
        _fail(f"reference_run_groups.{name}.vintage_costs.insurance: include would book insurance twice beside the "
              "D4(a) line (generator_costs.insurance); the declared rule is exclude")


def _validate_group_case_list(config: Mapping, name: str, group: Mapping, model_year: int) -> None:
    """Task Y14: a group's own case list is checked like the global one, at parse time.

    The list (``sensitivities``) parses against the shared scenarios with the
    group's name in every error; ``sensitivities_from`` must point at a group of
    the same model year that declares a list itself.  A list that moves a gated
    lever (``levers.GATED_LEVERS``) serves vintaged model years only, so no 2030
    group can carry one.  The decisions the list's cases need (acquisition terms,
    the explicit public rate, the network account at this model year, the
    institution pair) are enforced on its own cases, as the global list's are.
    """
    from scripts.public_ownership.model_year import is_vintaged_model_year

    groups = config.get("reference_run_groups") or {}
    own, shared = GROUP_CASE_LIST_KEY in group, group.get(GROUP_CASE_LIST_FROM_KEY)
    if own and shared is not None:
        _fail(f"reference_run_groups.{name}: give {GROUP_CASE_LIST_KEY} or {GROUP_CASE_LIST_FROM_KEY}, not both")
    # Network bounds (18-19 Oct 2026): any group may carry its own institution comparison (its own
    # central-dispatch year), not only a group with its own case list.
    override = group.get("institution_comparison")
    if override is not None and (not isinstance(override, Mapping) or set(override) - GROUP_INSTITUTION_KEYS):
        _fail(f"reference_run_groups.{name}.institution_comparison may set only {sorted(GROUP_INSTITUTION_KEYS)}")
    if shared is not None:
        shared = str(shared)
        if shared == name or shared not in groups or GROUP_CASE_LIST_KEY not in (groups[shared] or {}):
            _fail(f"reference_run_groups.{name}.{GROUP_CASE_LIST_FROM_KEY}={shared!r} must name another group "
                  f"that declares its own {GROUP_CASE_LIST_KEY}")
        if _group_year(groups[shared]) != model_year:
            _fail(f"reference_run_groups.{name}.{GROUP_CASE_LIST_FROM_KEY}={shared!r} is a different model year")
        return
    if not own:
        return
    block = group[GROUP_CASE_LIST_KEY]
    if block is not None and not isinstance(block, Mapping):
        _fail(f"reference_run_groups.{name}.{GROUP_CASE_LIST_KEY} must be a mapping (the sensitivities schema)")
    specs = configured_case_specs(config, name)
    gated = gated_levers_in(specs)
    if gated and not is_vintaged_model_year(model_year):
        _fail(f"reference_run_groups.{name}.{GROUP_CASE_LIST_KEY} moves gated lever(s) {gated}, which apply to the "
              f"vintaged model years only; the group is {model_year}")
    acquiring = [spec for spec in specs if spec.acquires_legacy()]
    if acquiring:
        _validate_acquisition(config.get("acquisition") or {}, acquiring)
    if any(spec.settings[LEVER_NETWORKS] != NETWORKS_PRIVATE for spec in specs):
        from scripts.public_ownership.network_finance_account import settings as network_settings

        try:
            network_settings(config.get("network_account"), model_year)
        except ValueError as error:
            _fail(f"reference_run_groups.{name}: {error}")
    if institution_cases(config, name):
        institution_market_twin(config, name)
        block = institution_block(config, name)
        if "{month}" not in str(block.get("central_scenario_pattern", "")):
            _fail(f"reference_run_groups.{name}: institution_comparison.central_scenario_pattern must contain "
                  "'{month}' when a case takes dispatch_institution: nationalised_dispatch")
        if not block.get("marginal_costs"):
            _fail(f"reference_run_groups.{name}: institution_comparison.marginal_costs must point at the "
                  "marginal-cost breakdown CSV")


def group_deployment_share(group: Mapping) -> str | None:
    """The L4 share a P2 balancing group's months bid on (task Y14), or None: the group takes every share."""
    value = (group or {}).get(GROUP_DEPLOYMENT_SHARE_KEY)
    return None if value is None else str(value)


def deployment_share_bid_case(scenario: str, share: str) -> str:
    """``public_ownership``, ``"0.25"`` -> ``public_ownership@deployment_share_25``: the case whose share table bids."""
    return f"{scenario}{VARIANT_SEPARATOR}{GATED_LEVERS[LEVER_STATE_DEPLOYMENT_SHARE].label(share)}"


def _validate_share_group(config: Mapping, name: str, group: Mapping, model_year: int, bids: str) -> None:
    """Task Y14 (decisions 13, 14 and 16): a P2 balancing group keyed by the L4 share its months bid on.

    The key is a vintaged-year device: the group must be a market-bill balancing
    group (a stack other than contract value, no ``bid_case``), its value an L4
    share (not ``all``), at least one case of its list must sit on its stack at
    that share, and the contract-value twin whose share table the months bid on
    (``<scenario>@deployment_share_<pct>``) must be in the same list, so the market
    group's register writes the table.
    """
    from scripts.public_ownership.model_year import VINTAGED_REGISTER_FROM_MODEL_YEAR, is_vintaged_model_year

    share = group_deployment_share(group)
    lever = GATED_LEVERS[LEVER_STATE_DEPLOYMENT_SHARE]
    if not is_vintaged_model_year(model_year):
        _fail(f"reference_run_groups.{name}.{GROUP_DEPLOYMENT_SHARE_KEY} applies from model year "
              f"{VINTAGED_REGISTER_FROM_MODEL_YEAR}; the group is {model_year}")
    if share not in lever.values or share == lever.default:
        _fail(f"reference_run_groups.{name}.{GROUP_DEPLOYMENT_SHARE_KEY}={share!r}; supported: "
              f"{[v for v in lever.values if v != lever.default]}")
    if bids == BIDS_CONTRACT_VALUE or "bid_case" in group:
        _fail(f"reference_run_groups.{name}.{GROUP_DEPLOYMENT_SHARE_KEY} keys a P2 balancing group: it needs a "
              f"balancing_bids stack other than {BIDS_CONTRACT_VALUE!r} and no bid_case")
    stack = [case for case in group_market_cases(config, name) if case != PRIVATE_SCENARIO]
    if not stack:
        _fail(f"reference_run_groups.{name}: no case of its list takes balancing_bids={bids!r} at "
              f"{GROUP_DEPLOYMENT_SHARE_KEY}={share!r}")
    twin = group_bid_case(config, name)
    if twin not in expanded_cases(config, name):
        _fail(f"reference_run_groups.{name}: its months bid on {twin!r}'s share table, which its case list does "
              "not configure")


def register_bound_parent(group: Mapping) -> str | None:
    """The group a register bound re-accounts (task Y14), or None."""
    value = (group or {}).get(GROUP_REGISTER_BOUND_KEY)
    return None if value is None else str(value)


def register_bound_groups(config: Mapping) -> list[str]:
    """Groups that re-account another group's solved months on a different register (task Y14)."""
    return [name for name, group in (config.get("reference_run_groups") or {}).items()
            if register_bound_parent(group) is not None]


def _resolved_bound_blocks(group: Mapping) -> dict:
    """The register-defining blocks with their documented defaults, so equal settings compare equal."""
    fleet = dict(group.get("fleet_vintages") or {})
    costs = dict(group.get("vintage_costs") or {})
    return {
        "fleet_vintages": {"pathway": fleet.get("pathway", "Holistic Transition"),
                           "retirement_rule": fleet.get("retirement_rule", "economic_life_with_extension"),
                           "life_source": fleet.get("life_source", "central"),
                           "join_mode": fleet.get("join_mode", "pro_rata"),
                           "site_vintages": fleet.get("site_vintages"),
                           "life_overrides": {str(k): float(v) for k, v in (fleet.get("life_overrides") or {}).items()},
                           "repowering": {str(k): dict(v) for k, v in (fleet.get("repowering") or {}).items()},
                           "evidence_exclude_countries": sorted(fleet.get("evidence_exclude_countries") or []),
                           "evidence_stale_records": fleet.get("evidence_stale_records"),
                           "no_cfd_on_repowering": sorted(fleet.get("no_cfd_on_repowering") or []),
                           "refurbishment": dict(fleet.get("refurbishment") or {})},
        "vintage_costs": {"enabled": bool(costs.get("enabled", True)), "table": costs.get("table"),
                          "insurance": costs.get("insurance", "exclude"), "learning": bool(costs.get("learning", False)),
                          "refurbishment_capex_share": {str(k): float(v) for k, v in
                                                        (costs.get("refurbishment_capex_share") or {}).items()}},
        # Decision 68 (3 Oct 2026): a group's own co-ownership financing (its sensitivities are bounds).
        "co_ownership_finance": dict(group.get("co_ownership_finance") or {}),
        # Decision 71 (4 Oct 2026): a group's own auction margin (its strike calibration).
        GROUP_AUCTION_MARGIN_KEY: dict(group.get(GROUP_AUCTION_MARGIN_KEY) or {}),
        # F28 (a): a group's own legislation year (None: the block's legislation.year).
        "legislated_year": group.get("legislated_year"),
    }


def _validate_register_bound(config: Mapping, name: str, group: Mapping) -> None:
    """Task Y14: the learning-rate and retirement-rule bounds, re-accounted on the parent's solved months.

    Y10 set the retirement rule (``fleet_vintages``) and the learning-rate capex
    (``vintage_costs``) per group, because they change the register's rows and its
    cost registry rather than a case's settings.  A bound is therefore a second
    contract-value group of a vintaged year on the parent's months, the same bill
    and forecast case, that changes one or both blocks and nothing else; it needs
    no solve.  Its case list is the parent's unless it declares its own.
    """
    from scripts.public_ownership.model_year import is_vintaged_model_year

    groups = config.get("reference_run_groups") or {}
    parent = register_bound_parent(group)
    if parent == name or parent not in groups:
        _fail(f"reference_run_groups.{name}.{GROUP_REGISTER_BOUND_KEY}={parent!r} must name another group")
    other = groups[parent] or {}
    if register_bound_parent(other) is not None:
        _fail(f"reference_run_groups.{name}: {parent!r} is itself a register bound; bound the central group")
    # F28 (a): a 2030 (unvintaged) bound may move the legislation year alone.
    unvintaged = not is_vintaged_model_year(_group_year(group)) and not is_vintaged_model_year(_group_year(other))
    if unvintaged:
        moved = sorted(k for k, v in _resolved_bound_blocks(group).items() if v != _resolved_bound_blocks(other)[k])
        if moved != ["legislated_year"] or "fleet_vintages" in group or "fleet_vintages" in other:
            _fail(f"reference_run_groups.{name}: a register bound of an unvintaged (2030) group moves only "
                  f"legislated_year (it moves {moved})")
    for which, block in ((name, group), (parent, other)):
        if unvintaged:
            continue
        if not is_vintaged_model_year(_group_year(block)) or "fleet_vintages" not in block:
            _fail(f"reference_run_groups.{which}: a register bound and its parent must be vintaged groups "
                  "(a fleet_vintages block, model year 2040 on)")
        if group_balancing_bids(block) != BIDS_CONTRACT_VALUE or "bid_case" in block or \
                GROUP_DEPLOYMENT_SHARE_KEY in block:
            _fail(f"reference_run_groups.{which}: a register bound and its parent take the contract-value stack")
    if _group_year(group) != _group_year(other):
        _fail(f"reference_run_groups.{name}: a register bound shares its parent's model year")
    for key in REGISTER_BOUND_SAME_KEYS:
        if group.get(key) != other.get(key):
            _fail(f"reference_run_groups.{name}.{key}={group.get(key)!r} differs from {parent!r}'s "
                  f"{other.get(key)!r}: a register bound re-accounts the same solved months")
    if group_forecast_case_id(config, name) != group_forecast_case_id(config, parent):
        _fail(f"reference_run_groups.{name}: a register bound projects through its parent's forecast case")
    if _resolved_bound_blocks(group) == _resolved_bound_blocks(other):
        _fail(f"reference_run_groups.{name} changes neither {' nor '.join(REGISTER_BOUND_BLOCKS)} off {parent!r}: "
              "it bounds nothing")
    if GROUP_AUCTION_MARGIN_KEY in other:
        _fail(f"reference_run_groups.{parent}.{GROUP_AUCTION_MARGIN_KEY}: a central group's margin is the study's "
              "vintage_capital_2050.auction_margin_share")
    if GROUP_AUCTION_MARGIN_KEY in group:
        # FES 2050 decision 71: none or high, with its own (re-margined) strike calibration.
        from scripts.public_ownership.vintage_capital_2050 import VintageCapitalError, bound_margin_spec

        try:
            bound_margin_spec(config, name)
        except VintageCapitalError as exc:
            _fail(f"reference_run_groups.{name}: {exc}")


def vintaged_groups(config: Mapping) -> list[str]:
    """The run groups whose registers are vintaged by the stock-flow (a ``fleet_vintages`` block, 2040 on)."""
    from scripts.public_ownership.model_year import is_vintaged_model_year

    return [name for name, group in (config.get("reference_run_groups") or {}).items()
            if "fleet_vintages" in group and is_vintaged_model_year(group_model_year(group))]


def _validate_bid_case(config: Mapping, name: str, group: Mapping, bids: str) -> None:
    """Item BM1, P3: ``bid_case`` names the tariff case a balancing group's months bid on.

    Only a tariff case (one that acquires the legacy fleet and keeps the market
    institution) can be named: a market-bill balancing group (P2) derives its bid
    case from the cases on its stack, and a central-dispatch case reads a
    different solved year.  The group must declare a stack other than contract
    value, because a contract-value group's months bid on nobody's shares.
    """
    case = group.get("bid_case")
    if not isinstance(case, str) or not case:
        _fail(f"reference_run_groups.{name}.bid_case must name a configured case")
    if bids == BIDS_CONTRACT_VALUE:
        _fail(f"reference_run_groups.{name}.bid_case={case!r} needs a balancing_bids stack other than "
              f"{BIDS_CONTRACT_VALUE!r}: a contract-value group's months bid on no case's shares")
    if case not in tariff_cases(config, name):
        _fail(f"reference_run_groups.{name}.bid_case={case!r} must be a configured acquisition-tariff case "
              "(legacy_fleet acquired, dispatch_institution: market); a market-bill group derives its bid case")


def price_level_parent_group(config: Mapping, group_name: str) -> str | None:
    """The group a price-level twin re-accounts (``price_level_of``); None for every other group."""
    group = (config.get("reference_run_groups") or {}).get(group_name) or {}
    parent = group.get(GROUP_PRICE_LEVEL_OF_KEY)
    return None if parent is None else str(parent)


def price_level_twin_groups(config: Mapping) -> dict[str, str]:
    """``{twin: parent}`` for every price-level twin, in configured order ({} without one)."""
    return {name: str(group[GROUP_PRICE_LEVEL_OF_KEY])
            for name, group in (config.get("reference_run_groups") or {}).items()
            if isinstance(group, Mapping) and group.get(GROUP_PRICE_LEVEL_OF_KEY) is not None}


def forecast_case_price_level(case: Mapping | None) -> str:
    """A forecast case's wholesale price level (item F12): ``model`` unless the case says otherwise."""
    from scripts.consumer_bill.project_future_bill import price_level_of

    return price_level_of(case or {})


def group_price_level(config: Mapping, group_name: str, forecast_cases: Mapping) -> str:
    """The price level a group's bill and register settle at: its forecast case's (``model`` without one)."""
    return forecast_case_price_level(forecast_cases.get(group_forecast_case_id(config, group_name)))


def _validate_price_level_twins(config: Mapping, forecast_cases: Mapping | None) -> None:
    """Item F12 (12 Oct 2026): each price-level twin re-accounts one model-price group, at one other level.

    A twin repeats its parent on every key but its own forecast case (the same months,
    stack, bid case and baseline), so its register and bills differ from the parent's only
    by the price the year settles on.  Its forecast case must name a non-model price level
    and the parent's forecast case as its ``price_level_parent``.  One twin per parent and
    level; a twin of a balancing group (P2, P3) needs the market group's twin at the same
    level, which it is measured against.  2030 (unvintaged) groups only.
    """
    from scripts.public_ownership.model_year import is_vintaged_model_year

    groups = config.get("reference_run_groups") or {}
    twins = price_level_twin_groups(config)
    seen: dict[tuple[str, str], str] = {}
    for name, parent in twins.items():
        group = groups[name]
        if parent == name or parent not in groups:
            _fail(f"reference_run_groups.{name}.{GROUP_PRICE_LEVEL_OF_KEY}={parent!r} must name another group")
        if parent in twins:
            _fail(f"reference_run_groups.{name}: {parent!r} is itself a price-level twin; twin the model-price group")
        parent_group = groups[parent]
        different = sorted(k for k in (set(group) | set(parent_group)) - PRICE_LEVEL_TWIN_OWN_KEYS
                           if group.get(k) != parent_group.get(k))
        if different:
            _fail(f"reference_run_groups.{name}: a price-level twin repeats its parent {parent!r} on every key but "
                  f"{sorted(PRICE_LEVEL_TWIN_OWN_KEYS)}; it differs on {different}")
        if group_case_selection(config, name) is not None:
            configured_case_specs(config, name)          # the selection parses, keeps known cases, and is closed
        if is_vintaged_model_year(_group_year(group)) or parent_group.get(GROUP_REGISTER_BOUND_KEY) is not None:
            _fail(f"reference_run_groups.{name}: price-level twins are implemented for unvintaged (2030) groups only")
        if not group.get("forecast_case") or group.get("forecast_case") == group_forecast_case_id(config, parent):
            _fail(f"reference_run_groups.{name}: a price-level twin names its own forecast_case (the parent's "
                  "case at the other price level)")
        if forecast_cases is None:
            continue
        case = forecast_cases.get(group["forecast_case"]) or {}
        level = forecast_case_price_level(case)
        if level == forecast_case_price_level({}):
            _fail(f"reference_run_groups.{name}: its forecast case {group['forecast_case']!r} is at the model's own "
                  "price level; a price-level twin's case declares price_level")
        parent_case = group_forecast_case_id(config, parent)
        if case.get("price_level_parent") != parent_case:
            _fail(f"reference_run_groups.{name}: forecast case {group['forecast_case']!r} must name "
                  f"price_level_parent: {parent_case!r} (the parent group's forecast case)")
        if forecast_case_price_level(forecast_cases.get(parent_case)) != forecast_case_price_level({}):
            _fail(f"reference_run_groups.{name}: the parent {parent!r} must settle at the model's own price level")
        bill_groups = consumer_bill_groups(config, forecast_cases)
        if name not in bill_groups or parent not in bill_groups:
            _fail(f"reference_run_groups.{name}: a price-level twin and its parent must both be bill groups")
        if (parent, level) in seen:
            _fail(f"reference_run_groups.{parent} has two twins at price level {level!r}: {seen[(parent, level)]}, {name}")
        seen[(parent, level)] = name
    if forecast_cases is None:
        return
    # A balancing twin is measured against the market group's twin at its level.
    for name, parent in twins.items():
        if group_balancing_bids(groups[parent]) == BIDS_CONTRACT_VALUE:
            continue
        level = group_price_level(config, name, forecast_cases)
        year = _group_year(groups[name])
        network = group_network_case(groups[name])
        market = [twin for twin, twin_parent in twins.items()
                  if group_balancing_bids(groups[twin_parent]) == BIDS_CONTRACT_VALUE
                  and _group_year(groups[twin]) == year and group_price_level(config, twin, forecast_cases) == level
                  and group_network_case(groups[twin]) == network]
        if len(market) != 1:
            _fail(f"reference_run_groups.{name}: a balancing group's price-level twin needs exactly one "
                  f"contract-value twin on {year} at {level!r} (network case {network!r}) to be measured against; "
                  f"found {market}")
        _validate_twin_case_coverage(config, name, market[0])


def _validate_twin_case_coverage(config: Mapping, name: str, market: str) -> None:
    """A balancing twin's cases have their market-side counterparts on the market twin's (reduced) list.

    The P2 merge measures each bid-stack case against its market twin (equal on every lever but
    ``balancing_bids``; ``merge_balancing_groups.market_twin``), which falls back silently to the
    scenario when the twin is missing; the P3 merge sets each tariff case against the same case on the
    market group. So every counterpart the full list has must be kept on the market twin's list.
    """
    groups = config["reference_run_groups"]
    if group_case_selection(config, name) is None and group_case_selection(config, market) is None:
        return
    market_cases = set(expanded_cases(config, market))
    if is_tariff_balancing_group(config, name):
        missing = [c for c in group_tariff_cases(config, name) if c not in market_cases]
        if missing:
            _fail(f"reference_run_groups.{name}: its tariff cases {missing[:5]} are not on {market}'s case list, "
                  "which the tariff merge sets them against")
        return
    on_stack = [c for c in group_market_cases(config, name) if c != PRIVATE_SCENARIO]
    if not on_stack:
        _fail(f"reference_run_groups.{name}: its case list keeps no case on its {group_balancing_bids(groups[name])!r} "
              "stack")
    full = {spec.case_id: spec for spec in configured_case_specs(config)}
    market_specs = [full[c] for c in market_cases if full[c].market_bill()
                    and full[c].settings.get(LEVER_BALANCING_BIDS, BIDS_CONTRACT_VALUE) == BIDS_CONTRACT_VALUE]
    for case in on_stack:
        wanted = {k: v for k, v in full[case].settings.items() if k != LEVER_BALANCING_BIDS}
        twins_full = [spec.case_id for spec in full.values() if spec.market_bill()
                      and spec.settings.get(LEVER_BALANCING_BIDS, BIDS_CONTRACT_VALUE) == BIDS_CONTRACT_VALUE
                      and {k: v for k, v in spec.settings.items() if k != LEVER_BALANCING_BIDS} == wanted]
        kept = [spec.case_id for spec in market_specs
                if {k: v for k, v in spec.settings.items() if k != LEVER_BALANCING_BIDS} == wanted]
        if twins_full and not kept:
            _fail(f"reference_run_groups.{name}: {case!r} is measured against its market twin {twins_full[0]!r}, which "
                  f"{market}'s case list drops")


def institution_parent_group(config: Mapping, group_name: str) -> str | None:
    """The market group an institution twin re-projects under central dispatch (``institution_of``); else None."""
    group = (config.get("reference_run_groups") or {}).get(group_name) or {}
    parent = group.get(GROUP_INSTITUTION_OF_KEY) if isinstance(group, Mapping) else None
    return None if parent is None else str(parent)


def institution_twin_groups(config: Mapping) -> dict[str, str]:
    """``{twin: parent}`` for every institution twin, in configured order ({} without one)."""
    return {name: str(group[GROUP_INSTITUTION_OF_KEY])
            for name, group in (config.get("reference_run_groups") or {}).items()
            if isinstance(group, Mapping) and group.get(GROUP_INSTITUTION_OF_KEY) is not None}


def institution_twin_central_scenarios(config: Mapping, group_name: str) -> list[str]:
    """The twelve central-dispatch months an institution twin's physical quantities come from (its parent's I3 year)."""
    parent = institution_parent_group(config, group_name)
    if parent is None:
        return []
    return institution_scenarios(config, parent)


def _validate_institution_twins(config: Mapping, forecast_cases: Mapping | None) -> None:
    """P1/P2 under central dispatch (20 Oct 2026): each institution twin re-projects one market group on its I3 year.

    The parent must be a 2030 (unvintaged) contract-value market group at the model's own price level, with
    its own central-dispatch year (``institution_comparison.central_scenario_pattern``) and FTI settlement;
    neither a price-level twin, an institution twin nor a register bound. The twin repeats it on every key but
    :data:`INSTITUTION_TWIN_OWN_KEYS` (the same market months: the national price is the parent's copperplate
    price, user decision 2), and its forecast case is the parent's case under central dispatch: ``institution:
    central_dispatch``, ``institution_parent`` the parent's case, the same months, and the central months and FTI
    settlement the parent's institution comparison names. One twin per parent; its case list keeps at least one
    case besides private.
    """
    from scripts.public_ownership.model_year import is_vintaged_model_year

    groups = config.get("reference_run_groups") or {}
    seen: dict[str, str] = {}
    for name, parent in institution_twin_groups(config).items():
        where = f"reference_run_groups.{name}"
        group = groups[name]
        if parent == name or parent not in groups:
            _fail(f"{where}.{GROUP_INSTITUTION_OF_KEY}={parent!r} must name another group")
        parent_group = groups[parent] or {}
        if group.get(GROUP_PRICE_LEVEL_OF_KEY) is not None:
            _fail(f"{where}: a group is a price-level twin or an institution twin, not both")
        if (parent_group.get(GROUP_INSTITUTION_OF_KEY) is not None or parent_group.get(GROUP_PRICE_LEVEL_OF_KEY) is not None
                or parent_group.get(GROUP_REGISTER_BOUND_KEY) is not None):
            _fail(f"{where}: {parent!r} is itself a twin or a register bound; twin the model-price market group")
        if group_balancing_bids(parent_group) != BIDS_CONTRACT_VALUE or "bid_case" in parent_group:
            _fail(f"{where}: {parent!r} must be a contract-value market group (P2 state bids have no meaning without "
                  "bids, user decision 7; P3 has its own I3 path, the institution comparison)")
        if is_vintaged_model_year(_group_year(parent_group)):
            _fail(f"{where}: institution twins are implemented for unvintaged (2030) groups only")
        different = sorted(k for k in (set(group) | set(parent_group)) - INSTITUTION_TWIN_OWN_KEYS
                           if group.get(k) != parent_group.get(k))
        if different:
            _fail(f"{where}: an institution twin repeats its parent {parent!r} on every key but "
                  f"{sorted(INSTITUTION_TWIN_OWN_KEYS)}; it differs on {different}")
        if parent in seen:
            _fail(f"reference_run_groups.{parent} has two institution twins: {seen[parent]}, {name}")
        seen[parent] = name
        block = institution_block(config, parent)
        if "{month}" not in str(block.get("central_scenario_pattern", "")) or not block.get("fti_settlement"):
            _fail(f"{where}: its parent {parent!r} needs institution_comparison.central_scenario_pattern (with "
                  "'{month}') and fti_settlement, the central-dispatch year and its FTI settlement")
        on_stack = [c for c in group_market_cases(config, name) if c != PRIVATE_SCENARIO]
        if not on_stack:
            _fail(f"{where}: its case list keeps no market-bill case besides private")
        if PRIVATE_SCENARIO not in expanded_cases(config, name):
            _fail(f"{where}: its case list must keep private (the comparator every step is measured against)")
        own_case = group.get("forecast_case")
        parent_case = group_forecast_case_id(config, parent)
        if not own_case or own_case == parent_case:
            _fail(f"{where}: an institution twin names its own forecast_case (the parent's case under central dispatch)")
        if forecast_cases is None:
            continue
        case = forecast_cases.get(own_case) or {}
        if case.get("institution") != INSTITUTION_CENTRAL_DISPATCH:
            _fail(f"{where}: forecast case {own_case!r} must declare institution: {INSTITUTION_CENTRAL_DISPATCH}")
        if missing := [k for k in INSTITUTION_CASE_KEYS if not case.get(k)]:
            _fail(f"{where}: forecast case {own_case!r} lacks {missing}")
        if case.get("institution_parent") != parent_case:
            _fail(f"{where}: forecast case {own_case!r} must name institution_parent: {parent_case!r} (the parent "
                  "group's forecast case)")
        parent_case_block = forecast_cases.get(parent_case) or {}
        if case.get("source_scenario_pattern") != parent_case_block.get("source_scenario_pattern"):
            _fail(f"{where}: forecast case {own_case!r} must settle the parent's months (source_scenario_pattern "
                  f"{parent_case_block.get('source_scenario_pattern')!r}): the national price and the energy line are "
                  "the market twin's (user decision 2)")
        if forecast_case_price_level(case) != forecast_case_price_level({}) or \
                forecast_case_price_level(parent_case_block) != forecast_case_price_level({}):
            _fail(f"{where}: institution twins settle at the model's own price level")
        for key, wanted in (("central_scenario_pattern", block.get("central_scenario_pattern")),
                            ("fti_settlement", block.get("fti_settlement"))):
            if str(case.get(key)) != str(wanted):
                _fail(f"{where}: forecast case {own_case!r} {key}={case.get(key)!r} is not its parent's institution "
                      f"comparison's {wanted!r}")
        if name not in consumer_bill_groups(config, forecast_cases):
            _fail(f"{where}: an institution twin must be a bill group (its forecast case on the parent's months)")


def is_tariff_balancing_group(config: Mapping, group_name: str) -> bool:
    """A balancing group whose months bid on a tariff case's shares (P3), so its bill is the acquisition tariff."""
    group = config["reference_run_groups"][group_name]
    return (group_balancing_bids(group) != BIDS_CONTRACT_VALUE and group.get("bid_case") is not None
            and group["bid_case"] in tariff_cases(config, group_name))


def _validate_balancing_groups(config: Mapping, specs: list[CaseSpec], forecast_cases: Mapping | None) -> None:
    """Item BM1: every balancing-bid stack a case takes is solved by exactly one bill group.

    A case whose ``balancing_bids`` is not the contract-value default reads a
    second solved year, so it needs a group declaring the same stack, with its own
    forecast case on that group's months.  The acquisition tariff and the
    institution comparison read the market group only, so a case that takes them
    cannot also take a new bid stack.  P3 is a tariff balancing group instead: it
    names its tariff case with ``bid_case``, projects only ``private`` on the
    market bill and tariffs that case on its own months, so it takes no part in
    the one-group-per-stack count, and exactly one group may name a bid case.

    Task Y14: the count is per case list (the global list for every group without
    its own) and per L4 share.  A vintaged group may key itself by the share its
    months bid on (``state_deployment_share``), so the 2050 list's P2 stack is
    solved by one group per share, each with its own share table; a group without
    the key takes every share, as every 2030 group does.  On the 2030 config the
    count is exactly the one it always was.
    """
    groups = config["reference_run_groups"]
    lists = {None: list(specs)}
    for name in groups:
        owner = case_list_owner(config, name)
        if owner is not None and owner not in lists:
            lists[owner] = configured_case_specs(config, owner)
    for owner_specs in lists.values():
        for spec in owner_specs:
            bids = spec.settings.get(LEVER_BALANCING_BIDS, BIDS_CONTRACT_VALUE)
            if bids != BIDS_CONTRACT_VALUE and not spec.market_bill():
                _fail(f"case {spec.case_id!r} takes balancing_bids={bids!r} and the acquisition tariff or central "
                      "dispatch; only market-bill cases can read a balancing run group so far")
    if forecast_cases is None:
        return
    bill_groups = consumer_bill_groups(config, forecast_cases)
    for name, group in groups.items():
        bids = group_balancing_bids(group)
        if bids != BIDS_CONTRACT_VALUE and name not in bill_groups:
            _fail(f"reference_run_groups.{name} declares balancing_bids={bids!r} but is not a bill group: its "
                  "forecast_case must be a consumer_bill.forecasts case on the same scenario_pattern")
    # Item F12 (12 Oct 2026): a price-level twin fills its parent's role at its own level
    # (_validate_price_level_twins), so the one-group-per-role counts below are on the
    # model-price groups, exactly as before any twin.
    twins = price_level_twin_groups(config)
    # P1/P2 under central dispatch (20 Oct 2026): an institution twin re-projects a market group's cases on its
    # I3 year; it takes no role in the counts below either.
    institution_twins = institution_twin_groups(config)
    bill_groups = [name for name in bill_groups if name not in twins and name not in institution_twins]
    tariff_groups = [name for name in groups if name not in twins and is_tariff_balancing_group(config, name)]
    # Network bounds (18-19 Oct 2026): every rule below holds per network case (group_network_case;
    # None for a group without one, so a config without network cases counts exactly as before).
    named = [(case_list_owner(config, name), group_network_case(groups[name]), groups[name]["bid_case"])
             for name in tariff_groups]
    repeated = sorted({case for owner, network, case in named if named.count((owner, network, case)) > 1})
    if repeated:
        _fail(f"more than one reference_run_group names bid_case {repeated} on one network case; the tariff merge "
              "needs one group per case")
    default_share = lever_default(LEVER_STATE_DEPLOYMENT_SHARE)
    for owner, owner_specs in lists.items():
        wanted = {(spec.settings.get(LEVER_BALANCING_BIDS, BIDS_CONTRACT_VALUE),
                   str(spec.settings.get(LEVER_STATE_DEPLOYMENT_SHARE, default_share))) for spec in owner_specs}
        # The network cases of the bill groups on this case list (a list no bill group uses is
        # checked once, on no network case, and so fails as it always did).
        networks = sorted({group_network_case(groups[name]) for name in bill_groups
                           if case_list_owner(config, name) == owner} or {None},
                          key=lambda n: (n is not None, n or ""))
        for network in networks:
            users = [name for name in bill_groups if case_list_owner(config, name) == owner
                     and name not in tariff_groups and group_network_case(groups[name]) == network]
            for bids, share in sorted(key for key in wanted if key[0] != BIDS_CONTRACT_VALUE):
                solving = [name for name in users if group_balancing_bids(groups[name]) == bids
                           and group_deployment_share(groups[name]) in (None, share)]
                if len(solving) != 1:
                    where = "" if owner is None else f" in reference_run_groups.{owner}'s case list"
                    keyed = "" if share == default_share else f" at {LEVER_STATE_DEPLOYMENT_SHARE}={share!r}"
                    on = "" if network is None else f" on network case {network!r}"
                    _fail(f"balancing_bids={bids!r} is taken by a case{where}{keyed}, so exactly one bill group{on} "
                          f"must declare it; found {solving}")
    # A balancing group is measured against its network's market group; with a reduced case list on
    # either side, every counterpart the merge sets a case against must be kept (twins: below).
    for name in bill_groups:
        if group_balancing_bids(groups[name]) == BIDS_CONTRACT_VALUE or group_network_case(groups[name]) is None:
            continue
        market = contract_value_group_for(config, name, forecast_cases)
        if group_case_selection(config, name) is not None or group_case_selection(config, market) is not None:
            _validate_twin_case_coverage(config, name, market)


def contract_value_group_for(config: Mapping, group_name: str, forecast_cases: Mapping) -> str:
    """The contract-value bill group a group is measured against: same model year, network case and price level.

    A price-level twin is measured against the market group's twin at its level; every other group against
    the model-price market group of its network case (register bounds are never a market group).
    """
    groups = config["reference_run_groups"]
    twins = price_level_twin_groups(config)
    group = groups[group_name]
    year, network, twin = _group_year(group), group_network_case(group), group_name in twins
    level = group_price_level(config, group_name, forecast_cases)
    # An institution twin (20 Oct 2026) is never the market group a balancing group is measured against.
    matches = [name for name in consumer_bill_groups(config, forecast_cases)
               if group_balancing_bids(groups[name]) == BIDS_CONTRACT_VALUE and _group_year(groups[name]) == year
               and groups[name].get(GROUP_REGISTER_BOUND_KEY) is None and (name in twins) == twin
               and groups[name].get(GROUP_INSTITUTION_OF_KEY) is None
               and group_network_case(groups[name]) == network
               and (not twin or group_price_level(config, name, forecast_cases) == level)]
    if len(matches) != 1:
        _fail(f"reference_run_groups.{group_name} needs exactly one contract-value bill group on {year}, network case "
              f"{network!r}{' at ' + repr(level) if twin else ''}; found {matches}")
    return matches[0]


def purchase_price_group(config: Mapping, group_name: str, forecast_cases: Mapping | None = None) -> str | None:
    """The group whose acquisition prices ``group_name``'s register holds (``purchase_price_from``), or None.

    A price-level twin repeats its parent's key; it holds the named group's twin at its own price level
    (the purchase is valued on the year settled at that level), or the named group itself if that group
    has no twin at the level.
    """
    groups = config.get("reference_run_groups") or {}
    source = (groups.get(group_name) or {}).get(GROUP_PURCHASE_PRICE_FROM_KEY)
    if source is None:
        return None
    source = str(source)
    twins = price_level_twin_groups(config)
    if group_name in twins and forecast_cases is not None:
        level = group_price_level(config, group_name, forecast_cases)
        at_level = [twin for twin, parent in twins.items()
                    if parent == source and group_price_level(config, twin, forecast_cases) == level]
        if len(at_level) == 1:
            return at_level[0]
    return source


def _validate_purchase_price_sources(config: Mapping, forecast_cases: Mapping | None) -> None:
    """Each ``purchase_price_from`` names a market group of the same year that prices every case the group acquires with."""
    groups = config.get("reference_run_groups") or {}
    for name, group in groups.items():
        source = group.get(GROUP_PURCHASE_PRICE_FROM_KEY)
        if source is None:
            continue
        if not isinstance(source, str) or source == name or source not in groups:
            _fail(f"reference_run_groups.{name}.{GROUP_PURCHASE_PRICE_FROM_KEY}={source!r} must name another group")
        if group_balancing_bids(group) != BIDS_CONTRACT_VALUE:
            _fail(f"reference_run_groups.{name}.{GROUP_PURCHASE_PRICE_FROM_KEY}: only a contract-value market group "
                  "holds another group's purchase (a bid-stack group's tariff is merged against its market group's)")
        held = purchase_price_group(config, name, forecast_cases)
        source_group = groups[held]
        if (group_balancing_bids(source_group) != BIDS_CONTRACT_VALUE
                or source_group.get(GROUP_PURCHASE_PRICE_FROM_KEY) is not None
                or _group_year(source_group) != _group_year(group)):
            _fail(f"reference_run_groups.{name}: {held!r} must be a contract-value group of the same model year that "
                  f"prices its own purchase (no {GROUP_PURCHASE_PRICE_FROM_KEY} of its own)")
        missing = sorted(set(tariff_cases(config, name)) - set(tariff_cases(config, held)))
        if missing:
            _fail(f"reference_run_groups.{name}: its acquisition cases {missing[:5]} are not on {held!r}'s case list, "
                  "whose register prices the purchase it holds")


def group_forecast_case_id(config: Mapping, group_name: str) -> str | None:
    """The forecast case a group's bill is projected through: its own, else the bill link's."""
    group = config["reference_run_groups"][group_name]
    return group.get("forecast_case") or (config.get("consumer_bill") or {}).get("forecast_case")


def group_balancing_bids(group: Mapping) -> str:
    """The balancing-bid stack a group's months were solved on (item BM1)."""
    return str(group.get("balancing_bids", BIDS_CONTRACT_VALUE))


def consumer_bill_groups(config: Mapping, forecast_cases: Mapping) -> list[str]:
    """Groups that produce a per-case bill: same model year and scenario pattern as their forecast case.

    A group's forecast case is its own ``forecast_case`` or, by default, the
    bill link's.  Historical groups never qualify, so they produce no bill.
    """
    names = []
    for name, group in config["reference_run_groups"].items():
        case = forecast_cases.get(group_forecast_case_id(config, name))
        if case is None:
            continue
        if (group_model_year(group) == int(case["modelled_year"]) and group.get("mode") == "twelve_months"
                and str(group.get("scenario_pattern")) == str(case.get("source_scenario_pattern", ""))):
            names.append(name)
    return names


def group_market_cases(config: Mapping, group_name: str) -> list[str]:
    """The market-bill cases a group projects: those on its balancing-bid stack.

    The contract-value group takes every market case on that stack.  A group on
    another stack takes those cases plus ``private``, which has to reproduce the
    group's own forecast case (house rule 3 within the group); private on those
    months is the comparator that isolates what the bid stack did.  A tariff
    balancing group (P3, ``bid_case``) projects ``private`` alone: its case takes
    the acquisition tariff (:func:`group_tariff_cases`), which replaces that bill.

    Task Y14: the cases come from the group's own list (:func:`case_list_owner`),
    and a group keyed by an L4 share (``state_deployment_share``) takes only the
    cases on its stack at that share.
    """
    if is_tariff_balancing_group(config, group_name):
        return [PRIVATE_SCENARIO]
    group = config["reference_run_groups"][group_name]
    bids = group_balancing_bids(group)
    share = group_deployment_share(group)
    default_share = lever_default(LEVER_STATE_DEPLOYMENT_SHARE)
    on_stack = [spec.case_id for spec in configured_case_specs(config, group_name)
                if spec.market_bill() and spec.settings.get(LEVER_BALANCING_BIDS, BIDS_CONTRACT_VALUE) == bids
                and (share is None or str(spec.settings.get(LEVER_STATE_DEPLOYMENT_SHARE, default_share)) == share)]
    if bids == BIDS_CONTRACT_VALUE:
        return on_stack
    return [PRIVATE_SCENARIO] + [case for case in on_stack if case != PRIVATE_SCENARIO]


def group_excluded_cases(config: Mapping, group_name: str) -> list[str]:
    """Every configured case a group's market bill does not project, reported as such."""
    projected = set(group_market_cases(config, group_name))
    return [case for case in expanded_cases(config, group_name) if case not in projected]


#: Levers that set what the state pays for the fleet, never which plant it owns:
#: tariff cases differing only on these own the same fleet, so they bid the same way.
#: F19 (6 Oct 2026): the legacy price path and the owners' rate move the purchase price, never the fleet.
#: F27 item 6 (24 Oct 2026): so does capacity procurement under an acquisition (the valuation's CM income).
#: User decision (24 Oct 2026): and the price of legacy offshore wind (owners' value or deal multiples).
ACQUISITION_TERM_LEVERS: tuple[str, ...] = (LEVER_PURCHASE_PRICE_BASIS, LEVER_THERMAL_BASIS, LEVER_ACQUISITION_TERM,
                                            LEVER_LEGACY_PRICE_PATH,
                                            LEVER_CAPACITY_PROCUREMENT, LEVER_LEGACY_OFFSHORE_PRICE)


def same_fleet_tariff_cases(config: Mapping, case_id: str, run_group: str | None = None) -> list[str]:
    """The tariff cases that own the same fleet as ``case_id``: equal on every lever but the acquisition terms.

    Under P3 these are the fully_nationalised family (the purchase-price basis,
    the thermal basis and the annuity term vary; ownership does not), so each has
    the same state share per unit and bids the same stack.  ``case_id`` first.
    ``run_group`` (task Y14) reads that group's own case list.
    """
    specs = {spec.case_id: spec for spec in configured_case_specs(config, run_group)}
    wanted = {k: v for k, v in specs[case_id].settings.items() if k not in ACQUISITION_TERM_LEVERS}
    family = [case for case in tariff_cases(config, run_group)
              if {k: v for k, v in specs[case].settings.items() if k not in ACQUISITION_TERM_LEVERS} == wanted]
    # Second round (29 Sep 2026): the toggle grid on a family case goes with it.  The
    # P3 toggles (rate, premiums, HPC completion, O&M, overhead) change what the fleet
    # costs, never which plant the state owns, so each bids the same stack.
    members = set(family)
    grid = [case for case in tariff_cases(config, run_group) if case not in members
            and any(m.get("grid") in members for m in (specs[case].meta.get("grid_memberships") or []))]
    if any(specs[case].settings.get(LEVER_DELIVERED_SHARE, "all") != specs[case_id].settings.get(
            LEVER_DELIVERED_SHARE, "all") for case in grid):
        _fail(f"a toggle grid on {case_id} moves the delivered share, which changes the fleet its months bid on")
    return [case_id] + [case for case in family if case != case_id] + grid


def group_tariff_cases(config: Mapping, group_name: str) -> list[str]:
    """The acquisition-tariff cases a group's nationalised bill prices (item BM1, P3).

    The contract-value group tariffs every configured tariff case, as it always
    has; a tariff balancing group tariffs its ``bid_case`` and every tariff case
    that owns the same fleet (:func:`same_fleet_tariff_cases`), since its months
    bid the same way for all of them; a market-bill balancing group (P2) tariffs none.
    """
    group = config["reference_run_groups"][group_name]
    if group_balancing_bids(group) == BIDS_CONTRACT_VALUE:
        return tariff_cases(config, group_name)
    if is_tariff_balancing_group(config, group_name):
        return same_fleet_tariff_cases(config, str(group["bid_case"]), group_name)
    return []


def group_bid_case(config: Mapping, group_name: str) -> str | None:
    """The ownership case whose state shares a balancing group's months bid on; None for contract value.

    A group keyed by an L4 share (task Y14) bids on that share's contract-value case,
    ``<scenario>@deployment_share_<pct>``, whose table ``ownership_shares.py
    --deployment-shares`` writes from the market group's register.
    """
    group = config["reference_run_groups"][group_name]
    if group_balancing_bids(group) == BIDS_CONTRACT_VALUE:
        return None
    if group.get("bid_case") is not None:
        return str(group["bid_case"])
    bases ={base_case(case) for case in group_market_cases(config, group_name) if case != PRIVATE_SCENARIO}
    if len(bases) != 1:
        _fail(f"reference_run_groups.{group_name}: the cases on its bid stack must share one scenario; "
              f"got {sorted(bases)}")
    share = group_deployment_share(group)
    if share is not None:
        return deployment_share_bid_case(bases.pop(), share)
    return bases.pop()


def check_balancing_group_scenarios(config: Mapping, group_name: str, scenario_configs: Mapping) -> None:
    """A balancing group's months bid on the shares its cases own; a contract-value group's months do not.

    ``scenario_configs`` maps scenario ids to their (merged or raw) definitions.
    """
    expected = group_bid_case(config, group_name)
    for scenario in reference_run_scenarios(config["reference_run_groups"][group_name]):
        block = ((((scenario_configs.get(scenario) or {}).get("support") or {}).get("bm_bids") or {})
                 .get("contract_value_bids") or {})
        bids = block.get("public_ownership_bids") or {}
        declared = bids.get("case_id") if bids.get("enabled") else None
        if declared != expected:
            _fail(f"reference_run_groups.{group_name}: scenario {scenario} bids on the shares of case "
                  f"{declared!r}, but the group's cases need {expected!r}")


def wholesale_source(scenario_id: str, scenario_configs: Mapping) -> str:
    """The scenario whose Stage 1 files a scenario reads (``wholesale_from``), as rules/market.smk does."""
    return (scenario_configs.get(scenario_id) or {}).get("wholesale_from") or scenario_id


def reference_run_scenarios(group: Mapping) -> list[str]:
    """Expand one group into its full-year or twelve monthly scenario IDs."""
    if group["mode"] == "full_year":
        return [group["scenario"]]
    return [group["scenario_pattern"].format(month=month) for month in MONTHS]


def institution_block(config: Mapping, run_group: str | None = None) -> dict:
    """The institution comparison settings: the global block, with the owning group's override (task Y14).

    A group with its own case list may set ``institution_comparison`` (its central
    scenario pattern, marginal-cost breakdown, FTI settlement, forecast-error
    charge) over the global block; every other group reads the global block.
    """
    block = dict(config.get("institution_comparison") or {})
    owner = case_list_owner(config, run_group)
    if owner is not None:
        block.update(dict((config["reference_run_groups"][owner] or {}).get("institution_comparison") or {}))
    if run_group is not None and run_group != owner:
        # Network bounds (18-19 Oct 2026): a group's own block (its own central-dispatch year) over both.
        block.update(dict(((config.get("reference_run_groups") or {}).get(run_group) or {})
                          .get("institution_comparison") or {}))
    return block


def institution_scenarios(config: Mapping, run_group: str | None = None) -> list[str]:
    """The twelve central-dispatch scenarios the institution comparison measures against."""
    block = institution_block(config, run_group)
    pattern = str(block.get("central_scenario_pattern", ""))
    return [pattern.format(month=month) for month in MONTHS] if "{month}" in pattern else []


def baseline_run_group(config: Mapping, group_name: str) -> str | None:
    return config["reference_run_groups"][group_name].get("baseline_run_group")


def baseline_scenarios(config: Mapping, group_name: str) -> list[str]:
    baseline = baseline_run_group(config, group_name)
    if not baseline:
        return []
    return reference_run_scenarios(config["reference_run_groups"][baseline])


def required_scenarios(config: Mapping, group_name: str) -> list[str]:
    """Own scenarios plus any baseline scenarios the group's register needs."""
    own = reference_run_scenarios(config["reference_run_groups"][group_name])
    return own + [s for s in baseline_scenarios(config, group_name) if s not in own]


def runnable_groups(config: Mapping, active_scenarios: Iterable[str]) -> list[str]:
    """Groups whose own scenarios are all in the active run selection.

    The baseline group is a declared input dependency rather than a gate: its
    completed market outputs must exist, but it need not be re-selected.
    """
    active = set(active_scenarios)
    return [
        name for name, group in config["reference_run_groups"].items()
        if set(reference_run_scenarios(group)) <= active
    ]
