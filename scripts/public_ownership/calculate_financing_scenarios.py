"""Build fixed-dispatch ownership/financing accounts from reviewed registers.

Item K1: every treatment is read from the lever settings a case carries
(:mod:`scripts.public_ownership.levers`), never from its name.  Who owns a row,
what the owner's capital costs, whether a legacy asset is bought and on what
terms, all follow from the settings; two cases that agree on every register
lever share every number and are accounted once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.finance import (
    TNUOS_CLASSES, TNUOS_CONVENTIONAL_CARBON, TNUOS_CONVENTIONAL_LOW_CARBON, capital_recovery_factor,
    coordination_at_stake, require_snapshot_schema, systematic_premium,
    corporation_tax_wedges, embedded_corporation_tax, nearest_zone, ownership_share_for_settings, private_rate,
    public_legacy_rate, rate_for_settings, state_overhead_carrier_rates, state_overhead_rate, with_model_year,
    co_ownership_rate, deployment_selection, deployment_share, neso_wind_ramp_deltas, private_rate_path_delta,
)
from scripts.public_ownership.model_year import cm_rate_columns, frame_model_year, is_vintaged_model_year
from scripts.public_ownership.legacy_valuation import (
    GROUP_EXCLUDED, GROUP_VALUED, PRICE_PATH_ANCHORED, PRICE_PATH_CP30, PRICE_PATH_MODEL, PRICE_PATH_TODAY, attach_price_path, extended_lives_block,
    settings_from_config as valuation_settings, value_legacy_fleet,
)
from scripts.public_ownership.levers import (
    CONTROL_PREMIUM_NONE, DEFAULT_SETTINGS, DELIVERED_HALF, IN_CONSTRUCTION_PRIVATE, IN_CONSTRUCTION_PUBLIC,
    IN_CONSTRUCTION_EXCLUDED, LEVER_EFW_ACQUISITION, EFW_EXCLUDED, EFW_INCLUDED,
    LEGACY_ACQUIRE_ALL, LEGACY_ACQUIRE_VALUED, LEGACY_PRIVATE, LIVES_CENTRAL, LIVES_EXTENDED,
    LEVER_ACQUISITION_TERM, LEVER_ASSET_LIVES, LEVER_DELIVERED_SHARE, TERM_REMAINING_LIFE,
    LEVER_GENERATION_CONTROL_PREMIUM, LEVER_IN_CONSTRUCTION, LEVER_LEGACY_CONTRACTS, LEVER_LEGACY_FLEET,
    LEVER_LEGACY_PRICE_PATH, PRICE_BOOK_VALUE, PRICE_MARKET_VALUE, PRICE_NO_SUPPORT, PRICE_RO_PASS_THROUGH,
    PRICE_STATE_PV,
    LEVER_NEW_BUILD_GENERATION, LEVER_PUBLIC_RETURN, LEVER_STATE_CAPEX_OVERRUN, LEVER_STATE_OM,
    LEVER_STATE_STAKE_SCALE, LEVER_TAX_EQUIVALENT, RETURN_CEPA_DECOMPOSITION, RETURN_PUBLIC_RATE,
    BIDS_CONTRACT_VALUE, PRIVATE_SCENARIO,
    LEVER_NEW_BUILD_STORAGE, LEVER_PRE_2025_CFD, LEVER_PURCHASE_PRICE_BASIS, LEVER_THERMAL_BASIS, LEVERS,
    NEW_BUILD_PRIVATE, NEW_BUILD_PUBLIC, NEW_BUILD_STATE_STAKE, OVERRUN_PLUS_15, PRE_2025_PUBLIC_OWNERS_VALUE,
    OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT,
    PRE_2025_PUBLIC_REPRICED, PRE_2025_STATE_OWNED, STAKE_FUNDED_SCALE, STATE_OM_PLUS_10,
    STORAGE_PRIVATE, TAX_EQUIVALENT_ON, THERMAL_MARKET_VALUE, VARIANT_SEPARATOR, CaseSpec, case_settings,
    comparator_for_settings, expand_axis_cases, expand_cases, spec_for_row,
    LEVER_PRIVATE_RATE_PATH, RATE_PATH_NESO_WIND_RAMP, gated_levers_in, lever_default,
    LEVER_AUCTION_MARGIN, AUCTION_MARGIN_CENTRAL, AUCTION_MARGIN_HIGH, AUCTION_MARGIN_NONE,
    LEVER_SELF_INSURANCE, SELF_INSURANCE_CENTRAL, SELF_INSURANCE_HIGH, SELF_INSURANCE_NONE,
    GBE_VALUES, NEW_BUILD_GBE_PARTNERSHIP,
)
# F27 item 1 (24 Oct 2026): Net Zero Teesside's pre-2025 DPA joins the pre-2025 option.
from scripts.public_ownership.levers import PRE_2025_DPA_CONTRACTS, is_pre_2025_dpa
# F27 item 6: capacity without an auction (the P3 purchase price values existing units on cost-based contracts).
from scripts.public_ownership import capacity_procurement
from scripts.public_ownership.levers import LEVER_CAPACITY_PROCUREMENT
# User decision (24 Oct 2026): legacy offshore wind at observed deal multiples (a named P3 case).
from scripts.public_ownership import offshore_deal_multiple
from scripts.public_ownership.levers import LEVER_LEGACY_OFFSHORE_PRICE
from config.config_loader import read_config


REQUIRED_ASSET_COLUMNS = {
    "model_year", "component_type", "asset_id", "carrier", "finance_technology",
    "finance_vintage", "capacity_mw", "dispatch_mwh", "evidence_status",
}
REQUIRED_COST_COLUMNS = {
    "technology", "capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh",
    "fixed_om_gbp_per_kw_year", "variable_om_gbp_per_mwh", "economic_life_years", "status",
}
LEVER_COLUMNS = tuple(LEVERS)
# Kept for callers written against the item-D1 vocabulary.
AXIS_COLUMNS = LEVER_COLUMNS
OWNERSHIP_BOUNDARIES = LEVERS[LEVER_PRE_2025_CFD].values
BOUNDARY_EXCLUDING_PRE_2025_CFD, BOUNDARY_ALL_CFD_REPRICED = OWNERSHIP_BOUNDARIES[:2]
PURCHASE_PRICE_BASES = LEVERS[LEVER_PURCHASE_PRICE_BASIS].values
THERMAL_BASES = LEVERS[LEVER_THERMAL_BASIS].values
DEFAULT_LEGACY_CASES = ("private", "public_coordination", "public_ownership")
#: Plant the state never owns (the legacy valuation's ``excluded_carriers``).
DEFAULT_EXCLUDED_CARRIERS = ("EU_market", "EU_import", "load_shedding", "load shedding", "voll", "unserved_energy")
#: Item OH1 applies to owned generation and storage only (config
#: ``public_ownership.state_operating_overhead.applies_to`` overrides it).
SUPPORTED_OVERHEAD_COMPONENTS = ("Generator", "StorageUnit")


def canonical_technology(value: object) -> str:
    """Normalise workbook typography without collapsing distinct technologies."""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text)


def _snapshot_at_stake(snapshot: dict, stake: float) -> dict:
    """The snapshot with the Scenario 2 rates recomputed at another state stake (lever state_stake_scale).

    Workbook v2.0's Sc2 formula (:func:`finance.coordination_at_stake`): the state's share of the
    equity requires the risk-free rate, so a smaller stake keeps more of the premium and the
    joint venture's rate rises; the guarantee's debt leg does not move.
    """
    return coordination_at_stake(snapshot, stake)


def _technology_rates(snapshot: dict) -> dict[str, dict]:
    return {
        canonical_technology(row["technology"]): row
        for row in snapshot.get("technologies", [])
    }


def _validated_costs(
    costs: pd.DataFrame, *, price_year: int | None = None, rate_basis: str | None = None,
) -> pd.DataFrame:
    """Keep reviewed/provisional rows and prove they share the configured price basis.

    Capital recovery multiplies these costs by real-CPI WACCs from the workbook
    snapshot, so every row must be in the configured real price year.
    """
    missing = REQUIRED_COST_COLUMNS - set(costs.columns)
    if missing:
        raise ValueError(f"Cost registry lacks columns: {sorted(missing)}")
    accepted_statuses = {"reviewed", "provisional"}
    accepted = costs.loc[
        costs["status"].astype(str).str.lower().isin(accepted_statuses)
    ].copy()
    if price_year is not None and "price_year" in accepted.columns:
        years = pd.to_numeric(accepted["price_year"], errors="coerce")
        wrong = accepted.loc[years.ne(int(price_year)), "technology"].tolist()
        if wrong:
            raise ValueError(
                f"Cost registry rows are not in configured price_year {price_year}: {wrong}"
            )
    if rate_basis is not None and "real_or_nominal" in accepted.columns:
        expected = "real" if str(rate_basis).lower().startswith("real") else "nominal"
        basis = accepted["real_or_nominal"].astype(str).str.strip().str.lower()
        wrong = accepted.loc[basis.ne(expected), "technology"].tolist()
        if wrong:
            raise ValueError(
                f"Cost registry rows are not on the snapshot's {expected} basis: {wrong}"
            )
    return accepted


# ── FES 2050 plan, task Y10: the vintaged register's cost lookup ────────────
#: A vintaged register's acquired survivors: bought in 2030, their annuity carried by Y12.
CARRIED_ACQUISITION_BASIS = "carried_from_2030"
#: Insurance on vintage-costed rows (Y2's finding).  The vintage table carries DESNZ
#: insurance inside fixed O&M (decision 26), and the finance layer books its own
#: insurance line (item D4(a): 0.4% of capex a year, 11.22 GBP/kW/yr offshore).
#: Declared rule: the table is read with ``insurance="exclude"`` and the D4(a) line
#: stays on, so every year carries one insurance line on one basis (the 2030 one).
VINTAGE_INSURANCE_RULE = "exclude"
#: The build-year column the Y10 join writes; ``commissioning_year`` stands in without it.
BUILD_YEAR_COLUMN = "build_year"


def build_year_vector(frame: pd.DataFrame) -> np.ndarray:
    """Each row's build year (float, NaN if undated): ``build_year`` where given, else ``commissioning_year``."""
    n = len(frame)
    out = np.full(n, np.nan)
    for column in ("commissioning_year", BUILD_YEAR_COLUMN):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            out = np.where(np.isfinite(values), values, out)
    return out


def vintage_cost_registry(table=None, *, insurance: str = VINTAGE_INSURANCE_RULE, learning: bool = False) -> pd.DataFrame:
    """Y2's vintage table as a cost registry keyed by technology and build year.

    Args:
        table: The loaded table, its path, or None for ``technology_costs_vintage.csv``.
        insurance: ``exclude`` (the declared rule, :data:`VINTAGE_INSURANCE_RULE`) takes
            DESNZ insurance out of fixed O&M; ``include`` keeps it (only with the
            D4(a) insurance line switched off, never both).
        learning: Capex from the CCC learning-rate column (decision 11's sensitivity).

    Returns:
        The registry schema (``technology_costs.csv`` columns) plus ``cost_build_year``
        and ``vintage_insurance_excluded_gbp_per_kw_year``, every technology and build year.
    """
    from scripts.public_ownership.vintage_costs import INSURANCE_MODES, TECHNOLOGY_COST_COLUMNS, load_vintage_costs

    if insurance not in INSURANCE_MODES:
        raise ValueError(f"vintage insurance {insurance!r}; supported: {INSURANCE_MODES}")
    frame = table if isinstance(table, pd.DataFrame) else load_vintage_costs(table)
    frame = frame.copy()
    if learning:
        frame["capex_power_gbp_per_kw"] = pd.to_numeric(frame["learning_capex_gbp_per_kw"], errors="coerce")
    excluded = pd.to_numeric(frame["insurance_gbp_per_kw_year"], errors="coerce").fillna(0.0)
    if insurance == "exclude":
        frame["fixed_om_gbp_per_kw_year"] = pd.to_numeric(frame["fixed_om_gbp_per_kw_year"], errors="coerce") - excluded
    out = frame[list(TECHNOLOGY_COST_COLUMNS)].copy()
    out["cost_build_year"] = frame["build_year"].astype(float).to_numpy()
    out["vintage_insurance_excluded_gbp_per_kw_year"] = excluded.to_numpy() if insurance == "exclude" else 0.0
    out["cost_basis"] = f"vintage_{'learning' if learning else 'central'}_insurance_{insurance}"
    return out


def vintage_options_for(block: Mapping, run_group: str | None, model_year: int, root: Path | str = "."):
    """The vintage cost registry a run group's register takes, or None (a 2030 register: technology_costs.csv).

    A register of a vintaged model year (:func:`model_year.is_vintaged_model_year`,
    2040 on) takes Y2's table unless its group sets ``vintage_costs: {enabled:
    false}``; ``vintage_costs`` may name ``table``, ``insurance`` and ``learning``.
    A group below 2040 that asks for vintage costs is refused, so no 2030 register
    can take them.
    """
    groups = dict((block or {}).get("reference_run_groups") or {})
    config = dict((groups.get(run_group) or {}).get("vintage_costs") or {}) if run_group else {}
    if not is_vintaged_model_year(model_year):
        if config and config.get("enabled", True):
            raise ValueError(f"reference_run_groups.{run_group}.vintage_costs is set but the register is for "
                             f"{model_year}; vintage costs apply from the vintaged model years only")
        return None
    if config.get("enabled", True) is False:
        return None
    table = config.get("table")
    registry = vintage_cost_registry(Path(root) / table if table else None,
                                     insurance=str(config.get("insurance", VINTAGE_INSURANCE_RULE)),
                                     learning=bool(config.get("learning", False)))
    # F28 (d): refurbished hydro and pumped hydro cost this share of new-build capex (declared).
    registry.attrs["refurbishment_capex_share"] = {str(k): float(v) for k, v in
                                                   dict(config.get("refurbishment_capex_share") or {}).items()}
    return registry


def _state_owned(asset: pd.Series, settings: Mapping[str, str], plan: pd.DataFrame | None, idx,
                 excluded_carriers=None) -> bool:
    """Is this row inside the state's ownership boundary under these lever settings?

    * A legacy row is owned only when the ``legacy_fleet`` lever acquires it:
      the valued group (renewables, hydro, marine, nuclear) under
      ``acquire_valued_group``, every acquirable GB row under ``acquire_all``;
      foreign plant is never bought.
    * Foreign plant (``excluded_carriers``, the valuation's list) is never owned
      whatever its vintage, with or without an acquisition plan (item B4: under
      P1 and P2 the guard ran only with a plan, so nine EU_market import rows
      were marked state-owned).
    * A new-build or in-construction row the register cannot cost (evidence not
      ``resolved``: CHP, oil and others the crosswalk holds as legacy-only) has no
      costed destination in a market case, so it is never owned either (item B4:
      465 such units were marked state-owned under P2 and their share was lost).
    * An in-construction row follows the ``in_construction`` lever.
    * A new-build row is owned when new build is public or state-staked, unless
      it is storage under ``new_build_storage: private`` or a CfD awarded before
      2025 (or of unknown round) under ``pre_2025_cfd_capacity: private``.
    * F27 item 1 (24 Oct 2026): Net Zero Teesside on its pre-2025 DPA
      (:func:`is_pre_2025_dpa`) follows ``pre_2025_cfd_capacity`` like a pre-2025
      CfD, but only in a case that does not acquire the legacy fleet.
    """
    vintage = str(asset.get("finance_vintage"))
    excluded = set(excluded_carriers if excluded_carriers is not None else DEFAULT_EXCLUDED_CARRIERS)
    if str(asset.get("carrier")) in excluded:
        return False
    if vintage == "legacy":
        legacy = settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE)
        if legacy == LEGACY_PRIVATE or plan is None:
            return False
        if not bool(plan.loc[idx, "acquirable"]):
            return False
        if legacy == LEGACY_ACQUIRE_VALUED:
            return str(plan.loc[idx, "legacy_scope_group"]) == GROUP_VALUED
        return legacy == LEGACY_ACQUIRE_ALL
    round_ = str(asset.get("cfd_round") or "")
    if (vintage == "in_construction" and plan is not None and str(asset.get("evidence_status", "resolved")) == "resolved"
            and settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE) != LEGACY_PRIVATE
            and settings.get(LEVER_IN_CONSTRUCTION) != IN_CONSTRUCTION_PUBLIC and round_ != HPC_CONTRACT_ROUND
            and not (round_ in CARRIED_IN_CONSTRUCTION_ROUNDS)):
        # User decision (15 Nov 2026): a case that buys the fleet owns plant in construction at the purchase as its
        # own new capacity (the vector path, _RowInputs.use_vintage, also owns the projects not operating at it).
        return True
    if plan is not None and not bool(plan.loc[idx, "acquirable"]):
        # Foreign plant is never bought, whatever its vintage (item N2).
        return False
    if str(asset.get("evidence_status", "resolved")) != "resolved":
        return False
    if (vintage == "in_construction" and round_ in CARRIED_IN_CONSTRUCTION_ROUNDS and plan is not None
            and carried_in_construction_mask(np.array([True]), np.array([round_ == HPC_CONTRACT_ROUND]),
                                             settings)[0]):
        # 13 Nov 2026: plant bought in 2030 (Hinkley Point C, AR7 offshore) is still the state's in a later year.
        return True
    if vintage == "in_construction":
        return settings.get(LEVER_IN_CONSTRUCTION) == IN_CONSTRUCTION_PUBLIC
    if settings.get(LEVER_NEW_BUILD_GENERATION, NEW_BUILD_PRIVATE) == NEW_BUILD_PRIVATE:
        return False
    if str(asset.get("component_type")) == "StorageUnit" and settings.get(LEVER_NEW_BUILD_STORAGE) == STORAGE_PRIVATE:
        return False
    pre_2025_dpa = is_pre_2025_dpa(asset.get("support_type"), asset.get("cfd_round"), asset.get("model_year"))
    if settings.get(LEVER_PRE_2025_CFD) in PRE_2025_STATE_OWNED:
        # F27 item 1: Net Zero Teesside joins the option where the bill is the market bill; a case that
        # acquires the legacy fleet keeps it on its DPA (the tariff pays it once, on the kept line).
        return not (pre_2025_dpa and settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE) != LEGACY_PRIVATE)
    if pre_2025_dpa:
        return False
    is_cfd = str(asset.get("support_type") or "").strip().lower() == "cfd"
    return not (is_cfd and str(asset.get("contract_vintage") or "") != "post_policy")


def _attach_lever_columns(register: pd.DataFrame, case_specs, levers: Sequence[str] = LEVER_COLUMNS) -> pd.DataFrame:
    """Every lever as a categorical column, mapped from the case (29 Sep 2026, register compaction).

    The settings are identical on every row of a case, so a code per row instead
    of a repeated string: the same values, a fraction of the memory.  ``levers``
    is :data:`LEVER_COLUMNS` plus any gated lever a configured case sets
    (:func:`levers.gated_levers_in`), so a 2030 register carries exactly the
    columns it always has.
    """
    settings = {spec.case_id: spec.settings for spec in case_specs}
    columns = lever_categoricals(register["case_id"], {case: s for case, s in settings.items()}, levers)
    out = pd.concat([register, pd.DataFrame(columns, index=register.index)], axis=1)
    # The columns in the order the per-row dicts gave them: the levers straight after case_id.
    before = list(register.columns[: register.columns.get_loc("case_id") + 1])
    after = [c for c in register.columns if c not in before]
    return out[before + list(columns) + after]


def lever_categoricals(case_ids: pd.Series, settings_by_case: Mapping[str, Mapping[str, str]],
                       levers: Sequence[str] = LEVER_COLUMNS) -> dict:
    """Each lever as ``register_io.lever_categorical`` of its per-row values, built from one value per case.

    The categorical of the mapped rows depends only on which values occur and in
    which order they first occur, and a case's rows all carry one value, so it
    is built from the cases in order of first appearance and expanded by code:
    the same categories and codes, without mapping every row of every lever.
    """
    from scripts.public_ownership.register_io import lever_categorical

    codes, cases = pd.factorize(case_ids.astype(object), use_na_sentinel=False)
    out = {}
    for lever in levers:
        per_case = lever_categorical(lever, [settings_by_case.get(case, {}).get(lever, lever_default(lever))
                                             if lever not in LEVER_COLUMNS else settings_by_case.get(case, {}).get(lever)
                                             for case in cases])
        out[lever] = pd.Categorical.from_codes(np.asarray(per_case.codes)[codes], dtype=per_case.dtype)
    return out


def _effective_capex_power(asset: pd.Series) -> tuple[float, bool]:
    """Capex per kW for the row, normalising a storage row to its own duration."""
    capex_power = float(asset["capex_power_gbp_per_kw"])
    cost_duration = pd.to_numeric(asset.get("duration_hours"), errors="coerce")
    energy = float(asset.get("energy_capacity_mwh", 0.0) or 0.0)
    capacity = float(asset["capacity_mw"])
    asset_duration = energy / capacity if capacity > 0 else None
    normalised = (
        asset["component_type"] == "StorageUnit"
        and pd.notna(cost_duration) and float(cost_duration) > 0
        and asset_duration is not None
    )
    if normalised:
        capex_power *= asset_duration / float(cost_duration)
    return capex_power, bool(normalised)


def _remaining_life(asset: pd.Series) -> float:
    life = float(asset["economic_life_years"])
    commissioning_year = asset.get("commissioning_year")
    if pd.isna(commissioning_year):
        return life
    return max(1.0, life - (float(asset["model_year"]) - float(commissioning_year)))


def acquisition_terms(acquisition: Mapping, settings: Mapping[str, str]) -> dict:
    """The acquisition block with the lever-controlled terms taken from the settings.

    ``purchase_price_basis``, ``thermal_basis`` and ``term_years`` are levers
    (Part 6.3, L14 to L16), and so (29 Sep 2026) is Hinkley Point C's treatment
    (``in_construction``); the block keeps only the parameters they need (the
    valuation settings).
    """
    terms = dict(acquisition or {})
    terms["purchase_price_basis"] = str(settings.get(LEVER_PURCHASE_PRICE_BASIS, DEFAULT_SETTINGS[LEVER_PURCHASE_PRICE_BASIS]))
    terms["thermal_basis"] = str(settings.get(LEVER_THERMAL_BASIS, DEFAULT_SETTINGS[LEVER_THERMAL_BASIS]))
    # F28 (e): "remaining_life" (each asset over its own remaining life) or one term in years.
    term = str(settings.get(LEVER_ACQUISITION_TERM, DEFAULT_SETTINGS[LEVER_ACQUISITION_TERM]))
    terms["term_years"] = term if term == TERM_REMAINING_LIFE else float(term)
    terms["legacy_contracts"] = str(settings.get(LEVER_LEGACY_CONTRACTS, DEFAULT_SETTINGS[LEVER_LEGACY_CONTRACTS]))
    terms["in_construction"] = str(settings.get(LEVER_IN_CONSTRUCTION, DEFAULT_SETTINGS[LEVER_IN_CONSTRUCTION]))
    # User decision (10 Nov 2026): energy-from-waste in the purchase or left with its owners (central).
    terms["efw_acquisition"] = str(settings.get(LEVER_EFW_ACQUISITION, DEFAULT_SETTINGS[LEVER_EFW_ACQUISITION]))
    terms["generation_control_premium"] = str(settings.get(LEVER_GENERATION_CONTROL_PREMIUM,
                                                           DEFAULT_SETTINGS[LEVER_GENERATION_CONTROL_PREMIUM]))
    terms["asset_lives"] = str(settings.get(LEVER_ASSET_LIVES, DEFAULT_SETTINGS[LEVER_ASSET_LIVES]))
    terms["legacy_price_path"] = str(settings.get(LEVER_LEGACY_PRICE_PATH, DEFAULT_SETTINGS[LEVER_LEGACY_PRICE_PATH]))
    # F27 item 6 (24 Oct 2026): existing units valued on cost-based contracts instead of the Capacity Market.
    terms["capacity_procurement"] = str(settings.get(LEVER_CAPACITY_PROCUREMENT,
                                                     DEFAULT_SETTINGS[LEVER_CAPACITY_PROCUREMENT]))
    # User decision (24 Oct 2026): legacy offshore wind at its owners' value or at observed deal multiples.
    terms["legacy_offshore_price"] = str(settings.get(LEVER_LEGACY_OFFSHORE_PRICE,
                                                      DEFAULT_SETTINGS[LEVER_LEGACY_OFFSHORE_PRICE]))
    return terms


def acquisition_row_terms(term_setting, retirement_year, model_year) -> np.ndarray:
    """Each acquired row's repayment term in years (F28 (e), user decision 26 Oct 2026).

    ``remaining_life``: the valuation's retirement year less the model year (the years the plant
    runs after the purchase, the horizon its price was valued over), at least one; a number:
    that term for every row.
    """
    retire = np.asarray(pd.to_numeric(pd.Series(retirement_year), errors="coerce"), dtype=float)
    if str(term_setting) != TERM_REMAINING_LIFE:
        term = float(term_setting)
        if term <= 0:
            raise ValueError("acquisition.term_years must be positive")
        return np.full(len(retire), term)
    model = np.asarray(pd.to_numeric(pd.Series(model_year), errors="coerce"), dtype=float)
    if (~np.isfinite(retire)).any():
        raise ValueError("a repayment over remaining life needs every acquired row's retirement year")
    return np.maximum(1.0, retire - model)


def crf_rows(rate: float, years) -> np.ndarray:
    """The capital recovery factor at one rate over each row's years (NaN where a row has none)."""
    years = np.asarray(years, dtype=float)
    return np.array([capital_recovery_factor(float(rate), float(n)) if np.isfinite(n) else np.nan for n in years])


def _plan_key(terms: Mapping, rate: float) -> tuple:
    return (terms["purchase_price_basis"], terms["thermal_basis"], terms["term_years"], terms["legacy_contracts"],
            terms["in_construction"], terms["generation_control_premium"],
            terms["asset_lives"], terms.get("legacy_price_path", PRICE_PATH_MODEL),
            terms.get("capacity_procurement", capacity_procurement.AUCTION),
            terms.get("legacy_offshore_price", offshore_deal_multiple.OWNERS_VALUE),
            terms.get("efw_acquisition", EFW_EXCLUDED), rate)


#: User decision (10 Nov 2026): the carriers the efw_acquisition lever keeps out of the purchase, and the asset
#: in_construction: left_private keeps out (Hinkley Point C, every row of it).
EFW_CARRIERS = ("waste_to_energy",)
HPC_ASSET_PREFIX = "Hinkley Point C"
#: Hinkley Point C's own contract round: a later model year's register carries the plant as pre-legislation
#: in-construction cohorts on this round (the 2030 register as one legacy row).
HPC_CONTRACT_ROUND = "HPC"
#: Contract rounds whose plant the 2030 purchase bought as operating (every pre-legislation round CP30 counts as built
#: by 2030). A later model year's FES vintages date part of a round after the transfer (in-construction: AR7's 3.3 GW
#: offshore in 2040), and a case that bought it in 2030 still owns it then (review finding 9, 13 Nov 2026). Post-AR7
#: FES cohorts (PAR7_*) and Sizewell C's RAB are not in the 2030 purchase and keep their in-construction treatment.
CARRIED_IN_CONSTRUCTION_ROUNDS: tuple[str, ...] = (
    HPC_CONTRACT_ROUND, "Investment Contract", "AR1", "AR2", "AR3", "AR4", "AR5", "AR6", "AR7", "AR7a")


def carries_acquired_legacy(settings: Mapping[str, str]) -> bool:
    """A case that buys the legacy fleet (S3 and its twins): its 2030 purchase is carried into later years."""
    return str(settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE)) != LEGACY_PRIVATE


def carried_in_construction_mask(in_construction_round_rows: np.ndarray, hpc_rows: np.ndarray,
                                 settings: Mapping[str, str]) -> np.ndarray:
    """Rows a buying case carries as acquired legacy plant: in-construction rows on a round the 2030 purchase bought,
    Hinkley Point C's only while the case keeps HPC in the purchase (``in_construction`` not ``left_private``)."""
    if not carries_acquired_legacy(settings):
        return np.zeros(len(in_construction_round_rows), dtype=bool)
    return in_construction_round_rows & (~hpc_rows | carries_acquired_in_construction(settings))


def carries_acquired_in_construction(settings: Mapping[str, str]) -> bool:
    """A case whose 2030 purchase bought Hinkley Point C still owns it in a later model year (13 Nov 2026).

    User decision (results review, finding 1): every 2030-acquired asset still operating in 2040/2050 is
    state-owned there, its income credited and its capital in the carried annuity. The case buys the legacy
    fleet and keeps HPC in the purchase (bought operating, ``in_construction: private``, or completed by the
    state, ``public_completion``); ``left_private`` leaves HPC out of the purchase and private in every year.
    """
    return (str(settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE)) != LEGACY_PRIVATE
            and str(settings.get(LEVER_IN_CONSTRUCTION, IN_CONSTRUCTION_PRIVATE)) != IN_CONSTRUCTION_EXCLUDED)


def case_purchase_exclusions(merged: pd.DataFrame, acquisition: Mapping) -> np.ndarray:
    """Legacy rows a case leaves out of the purchase (private, unpriced): energy-from-waste unless the case buys
    it (``efw_acquisition: included``), and Hinkley Point C under ``in_construction: left_private``."""
    legacy = merged["finance_vintage"].astype(str).eq("legacy").to_numpy()
    out = np.zeros(len(merged), dtype=bool)
    if str(acquisition.get("efw_acquisition", EFW_EXCLUDED)) != EFW_INCLUDED:
        out |= merged["carrier"].astype(str).isin(EFW_CARRIERS).to_numpy()
    if str(acquisition.get("in_construction", IN_CONSTRUCTION_PRIVATE)) == IN_CONSTRUCTION_EXCLUDED:
        out |= merged["asset_id"].astype(str).str.startswith(HPC_ASSET_PREFIX).to_numpy()
    return out & legacy


#: Columns of ``data/public_ownership/financing/hpc_completion_cost.csv``.
HPC_COMPLETION_COLUMNS = ("asset_match", "total_capacity_mw", "sunk_cost_gbp_2015", "remaining_capex_gbp_2015",
                          "price_index_2015_to_price_year", "finance_technology")


def hpc_completion_price(asset: Mapping, table: pd.DataFrame, *, public_rate: float, term_years: float,
                         premium: float) -> dict | None:
    """What the state pays to complete one in-construction asset itself (user decision 10, 29 Sep 2026).

    The state takes over the project: it pays the vendors the capital already spent
    (sunk cost, the natural compensation for an asset under construction), and
    finances the remaining capex at the public rate plus the graded construction
    premium.  Expressed as one price so it replaces the asset's purchase price and
    runs on the acquisition's own annuity (no double count, and the same basis as
    every other purchase): ``(sunk + remaining x CRF(r + p, T) / CRF(r, T)) x index
    x (modelled capacity / project capacity)``.  The capacity share scales the
    project to the capacity the register values (the legacy valuation's cash flows
    are the modelled unit's).  ``None`` when the asset is not in the table.
    """
    name = str(asset.get("asset_id", "")).split("::", 1)[0]
    rows = table.loc[table["asset_match"].astype(str).map(lambda token: name.startswith(token))]
    if rows.empty:
        return None
    row = rows.iloc[0]
    total_mw = float(row["total_capacity_mw"])
    share = min(1.0, float(asset["capacity_mw"]) / total_mw) if total_mw > 0 else 1.0
    index = float(row["price_index_2015_to_price_year"])
    sunk = float(row["sunk_cost_gbp_2015"]) * index * share
    remaining = float(row["remaining_capex_gbp_2015"]) * index * share
    uplift = capital_recovery_factor(public_rate + premium, term_years) / capital_recovery_factor(public_rate, term_years)
    return {"price_gbp": sunk + remaining * uplift, "sunk_cost_gbp": sunk, "remaining_capex_gbp": remaining,
            "construction_premium": premium, "construction_premium_capitalised_gbp": remaining * (uplift - 1.0),
            "capacity_share_of_project": share}


def load_cost_calibration(path: str | Path | None) -> pd.DataFrame | None:
    """The private cost calibration table (user decision 1, 29 Sep 2026): capex factor per carrier and round."""
    if not path:
        return None
    table = pd.read_csv(path)
    missing = {"carrier", "cfd_round", "capex_factor"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    factors = pd.to_numeric(table["capex_factor"], errors="coerce")
    if factors.isna().any() or not factors.between(0.05, 2.0).all():
        raise ValueError(f"{path}: every capex_factor must be a number in [0.05, 2]")
    # 6 Nov 2026 (user): one factor per round, every finance vintage pooled. A table keyed on the
    # vintage (before that date) is refused: under F28 the vintage follows the register's legislation year.
    if "finance_vintage" in table.columns:
        raise ValueError(
            f"{path} is keyed on finance_vintage; the calibration is one factor per (carrier, cfd_round) since "
            "6 Nov 2026. Re-derive it: python scripts/public_ownership/derive_cost_calibration.py REGISTER --out TABLE")
    if table.duplicated(["carrier", "cfd_round"]).any():
        raise ValueError(f"{path} repeats a (carrier, cfd_round) cohort")
    return table


def apply_cost_calibration(merged: pd.DataFrame, table: pd.DataFrame | None) -> pd.DataFrame:
    """Scale the capital cost of the calibrated cohorts, in every case (trust the auction).

    A cohort that settles at a cleared CfD strike is taken to break even there:
    the strike is revealed required revenue at the private cost of capital, and
    the capital cost is the part the model does not observe, so its capex is
    scaled by the table's factor (derivation in the table and in
    ``scripts/public_ownership/derive_cost_calibration.py``).  The plant costs what it
    costs whoever owns it, so every case takes the same factor; scaling the
    private case alone would make the state's identical plant dearer.  Insurance
    that is a share of capex follows (it is computed from the scaled capex), so
    the cohort's private rent is zero at the strike.  One factor per (carrier, round),
    applied to every finance vintage of the round (6 Nov 2026): under F28 the vintage is
    set by the register's legislation year, not by the plant, so the factor follows the
    plant into every group (the 2025 legislation bound included).  A cohort in the table
    that matches no row is refused: the table and the fleet have drifted apart.
    """
    merged = merged.copy()
    merged["capex_calibration_factor"] = 1.0
    # Follow-up F6 (4 Oct 2026): at an auction margin m the cohort's factor is
    # k(m) = k(0) - m x revenue / (capital charge + capex-scaled insurance), so a row's
    # capex at margin m is its calibrated capex x (1 - m x slope), slope = revenue /
    # ((capital charge + capex-scaled insurance) x k(0)).  Tables without the cohort sums
    # (older derivations) give slope 0: no margin.  The flag marks capex-based insurance.
    merged["capex_calibration_margin_slope"] = 0.0
    merged["capex_calibration_insurance_scales"] = False
    if table is None or table.empty:
        return merged
    carrier = merged["carrier"].astype(str)
    rounds = merged["cfd_round"].fillna("").astype(str) if "cfd_round" in merged.columns else pd.Series("", index=merged.index)
    # 28 Oct 2026: every drifted cohort is named at once, and how to re-derive (the derivation reads a
    # register this step cannot build while the table is stale).
    missing = []
    for row in table.itertuples(index=False):
        mask = carrier.eq(str(row.carrier)) & rounds.eq(str(row.cfd_round))
        if not mask.any():
            missing.append(f"row of {row.carrier} {row.cfd_round} (the fleet holds no such round)")
    if missing:
        raise ValueError(
            f"cost calibration: no {missing[0]}; the calibration table no longer matches the fleet (re-derive it). "
            f"Every cohort without rows: {'; '.join(missing)}. To re-derive: build this register once with "
            "public_ownership.cost_calibration.enabled: false, run derive_cost_calibration.py REGISTER --out TABLE "
            "(it divides the recorded factor, here 1, back out), then build again with the table enabled")
    for row in table.itertuples(index=False):
        mask = carrier.eq(str(row.carrier)) & rounds.eq(str(row.cfd_round))
        factor = float(row.capex_factor)
        for column in ("capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh"):
            if column in merged.columns:
                merged.loc[mask, column] = pd.to_numeric(merged.loc[mask, column], errors="coerce") * factor
        merged.loc[mask, "capex_calibration_factor"] = factor
        merged.loc[mask, "capex_calibration_margin_slope"] = calibration_margin_slope(row)
        merged.loc[mask, "capex_calibration_insurance_scales"] = (
            float(getattr(row, "cohort_capex_scaled_insurance_gbp", 0.0) or 0.0) > 0.0)
    return merged


def apply_refurbishment_costs(merged: pd.DataFrame, shares: Mapping[str, float]) -> pd.DataFrame:
    """F28 (d): the refurbished share of a cohort costs ``share`` of new-build capex (a vintaged register).

    ``cohort_refurbishment_share`` (the stock-flow mix's ``refurbishment_share``, hydro and pumped
    hydro under the strict rule) of the row is refurbished plant: its capex is
    ``1 - s + s x share`` of the vintage row's.  A refurbished technology without a declared share
    is refused.
    """
    if "cohort_refurbishment_share" not in merged.columns:
        return merged
    s = pd.to_numeric(merged["cohort_refurbishment_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if not (s > 0).any():
        return merged
    technology = (merged["stock_flow_technology"].astype(str) if "stock_flow_technology" in merged.columns
                  else merged["finance_technology"].astype(str))
    missing = sorted(set(technology[s > 0]) - set(shares))
    if missing:
        raise ValueError(f"refurbished cohorts of {missing} need vintage_costs.refurbishment_capex_share")
    scale = 1.0 - s + s * technology.map(shares).fillna(1.0)
    out = merged.copy()
    for column in ("capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * scale
    out["refurbishment_capex_scale"] = scale
    return out


def calibration_margin_slope(row) -> float:
    """``revenue / ((capital charge + capex-scaled insurance) x k(0))`` of one calibration row (0 without the sums)."""
    try:
        revenue = float(getattr(row, "cohort_contract_revenue_gbp"))
        capital = float(getattr(row, "cohort_capital_charge_private_gbp"))
        insurance = float(getattr(row, "cohort_capex_scaled_insurance_gbp", 0.0) or 0.0)
        factor = float(getattr(row, "capex_factor"))
    except (AttributeError, TypeError, ValueError):
        return 0.0
    denominator = (capital + insurance) * factor
    if not (np.isfinite(revenue) and np.isfinite(denominator)) or denominator <= 0.0:
        return 0.0
    return revenue / denominator


def auction_margin_capex_scale(slope, margin: float) -> np.ndarray:
    """``1 - margin x slope``: a calibrated row's capex at the auction margin, as a share of its k(0) capex."""
    scale = 1.0 - float(margin) * np.asarray(slope, dtype=float)
    if (scale <= 0.05).any():
        raise ValueError(f"auction margin {margin}: a calibrated cohort's capex would fall to {scale.min():.3f} of its "
                         "break-even value; the margin is larger than the cohort can carry")
    return scale


def _carbon_removal_split(register: pd.DataFrame, carriers: Sequence[str]) -> pd.Series:
    """Rows on a split (electricity plus carbon-removal) contract with a required revenue to split."""
    return (register["carrier"].astype(str).isin(carriers) & register["component_type"].astype(str).eq("Generator")
            & pd.to_numeric(register["annual_required_revenue_gbp"], errors="coerce").notna())


def carbon_removal_leg_share(register: pd.DataFrame, carriers: Sequence[str] | None) -> pd.Series | None:
    """The electricity-leg share per asset, read off the private rows of ``register`` (None when nothing splits).

    :func:`book_carbon_removal_cfd` applies it; computed once from the private
    case it can be applied to each register key's rows in turn (30 Sep 2026).
    """
    from scripts.public_ownership.cfd_repricing import electricity_leg_share

    carriers = [str(c) for c in (carriers or ())]
    if not carriers or register.empty or "producer_rent_gbp" not in register.columns:
        return None
    private = register.loc[register["case_id"].eq("private") & _carbon_removal_split(register, carriers)]
    private = private.set_index("asset_id")
    revenue = (pd.to_numeric(private["market_revenue_gbp"], errors="coerce").fillna(0.0)
               + pd.to_numeric(private["support_revenue_gbp"], errors="coerce").fillna(0.0)
               - pd.to_numeric(private["fuel_carbon_gbp"], errors="coerce").fillna(0.0))
    return electricity_leg_share(private, revenue, pd.to_numeric(private["annual_required_revenue_gbp"], errors="coerce"),
                                 carriers)


def book_carbon_removal_cfd(register: pd.DataFrame, carriers: Sequence[str] | None,
                            leg_share: pd.Series | None = None,
                            live_share: pd.Series | None = None,
                            merchant_removals: pd.Series | None = None) -> pd.DataFrame:
    """User decision 2 (29 Sep 2026): Power BECCS's carbon-removal leg is paid by its own CfD, never by the bill.

    DESNZ's dual CfD pays BECCS an electricity CfD on the levy and a carbon CfD
    (CfDc) from the Exchequer.  The register costs the whole plant, so its
    electricity strike covers only ``leg share`` of required revenue (the private
    contract revenue over the private required revenue, :func:`cfd_repricing.
    electricity_leg_share`) and the carbon CfD pays the rest at the owner's own
    cost: ``(1 - leg share) x required revenue`` of the case, booked to the row as
    ``carbon_removal_cfd_revenue_gbp`` and into its rent and the state's share of
    it.  So the private electricity-leg rent is zero by construction, a state owner
    carries no capture-leg deficit, and the capture leg's financing saving is the
    Exchequer's (a smaller carbon CfD), which the Exchequer ledger books in every
    case.  Who funds the CfDc (the Exchequer, or a levy on bills) is the
    ``beccs_removal_funding`` lever, read downstream.

    ``leg_share`` (:func:`carbon_removal_leg_share`) is the share per asset when
    the private rows are not in ``register`` (one register key at a time).

    FES 2050 decision 44 (user, 3 Oct 2026): ``live_share`` (asset id -> the share of
    the model year the carbon CfD is live on its own term, :func:`carbon_cfd_live_share`)
    scales the payment, so a carbon CfD that has ended pays nothing.  None (every 2030
    register) pays it in full, as before.

    FES 2050 (user, 3 Oct 2026): ``merchant_removals`` (asset id -> GBP,
    :func:`merchant_removal_value`) is what a plant earns for its removals once its carbon CfD
    has ended, at their market value (UK ETS with greenhouse gas removals integrated). It is
    booked in the same column: within a fixed cap a removal credit displaces an auctioned
    allowance, so its payer is the Exchequer, as for the carbon CfD. None adds nothing.
    """
    register = register.copy()
    register["carbon_removal_cfd_revenue_gbp"] = 0.0
    register["electricity_leg_share"] = np.nan
    carriers = [str(c) for c in (carriers or ())]
    if not carriers or register.empty or "producer_rent_gbp" not in register.columns:
        return register
    split = _carbon_removal_split(register, carriers)
    if not split.any():
        return register
    share = leg_share if leg_share is not None else carbon_removal_leg_share(register, carriers)
    leg = register.loc[split, "asset_id"].map(share).fillna(1.0)
    rr = pd.to_numeric(register.loc[split, "annual_required_revenue_gbp"], errors="coerce").fillna(0.0)
    cfdc = (1.0 - leg) * rr
    if live_share is not None:
        cfdc = cfdc * register.loc[split, "asset_id"].map(live_share).fillna(1.0).astype(float)
    if merchant_removals is not None:
        cfdc = cfdc + register.loc[split, "asset_id"].map(merchant_removals).fillna(0.0).astype(float)
    register.loc[split, "electricity_leg_share"] = leg
    register.loc[split, "carbon_removal_cfd_revenue_gbp"] = cfdc
    has_rent = split & register["producer_rent_gbp"].notna()
    ownership = pd.to_numeric(register["ownership_share"], errors="coerce").fillna(0.0)
    register.loc[has_rent, "producer_rent_gbp"] = register.loc[has_rent, "producer_rent_gbp"] + cfdc[has_rent]
    register.loc[has_rent, "public_surplus_before_recycling_gbp"] = (
        register.loc[has_rent, "public_surplus_before_recycling_gbp"] + ownership[has_rent] * cfdc[has_rent])
    return register


def carbon_cfd_live_share(merged: pd.DataFrame, carriers: Sequence[str] | None, term_years: int | None,
                          model_year: int | None = None) -> pd.Series | None:
    """FES 2050 decision 44: the share of the model year each split-contract row's carbon CfD is live.

    The carbon-removal CfD (CfDc) runs ``term_years`` from the row's build year, on its own
    term rather than the electricity CfD's; its live share is
    ``capacity_market_2050.live_fraction`` (expiries spread through the expiry year), the rule
    the register's contract labels use.  A row without a build year keeps 1.  None when no
    term is configured (every 2030 register).
    """
    if term_years is None or merged.empty:
        return None
    from scripts.market.capacity_market_2050 import live_fraction

    carriers = [str(c) for c in (carriers or ())]
    rows = merged.loc[merged["carrier"].astype(str).isin(carriers)]
    if rows.empty:
        return None
    year = int(model_year if model_year is not None else pd.to_numeric(rows["model_year"], errors="coerce").max())
    build = pd.to_numeric(rows["build_year"], errors="coerce") if "build_year" in rows.columns         else pd.Series(np.nan, index=rows.index)
    share = [live_fraction(float(b), int(term_years), year) if np.isfinite(b) else 1.0 for b in build]
    return pd.Series(share, index=rows["asset_id"].astype(str).to_numpy(), dtype=float)


def merchant_removal_value(merged: pd.DataFrame, live_share: pd.Series | None, price_gbp_per_mwh: float | None) -> pd.Series | None:
    """FES 2050 (user, 3 Oct 2026): each split-contract row's removals sold at market value after its carbon CfD.

    ``(1 - carbon CfD live share) x dispatch x price per MWh`` (captured tCO2 per MWh x the
    removal price, ``support_regime_2050.beccs_strike``'s ``removal_credit_gbp_per_mwh``).
    None without a price or a live share (every 2030 register).
    """
    if price_gbp_per_mwh is None or live_share is None or live_share.empty:
        return None
    rows = merged.assign(asset_id=merged["asset_id"].astype(str)).set_index("asset_id")
    dispatch = pd.to_numeric(rows["dispatch_mwh"], errors="coerce").fillna(0.0).reindex(live_share.index).fillna(0.0)
    return (1.0 - live_share.clip(0.0, 1.0)) * dispatch * float(price_gbp_per_mwh)


#: Register support type of Sizewell C's regulated asset base (Y10/Y11 ``known_contracts_2050``).
RAB_SUPPORT_TYPE = "RAB"


def support_lines_by_asset(merged: pd.DataFrame, private_frame: pd.DataFrame, support: Mapping | None, *,
                           book, reference: pd.DataFrame | None = None, ldes_carriers=()) -> dict:
    """FES 2050 decision 58: each support line per asset id (``support_lines_2050``), or ``{}`` without the block.

    Ancillary services and gate fees are asset properties (capacity, carrier, dispatch); the
    life-extension top-up is the private case's operating shortfall after every other income
    (``book`` books the carbon CfD and RAB support first), so it is sized after them; the battery
    top-up (decision 63) is sized last, on the full required revenue. ``reference`` (a balancing
    run group's market group, private rows): every line is held at its per-asset amounts there.
    """
    if not support:
        return {}
    from scripts.public_ownership import support_lines_2050 as sl

    out: dict = {}
    if support.get("ancillary_services"):
        spec = support["ancillary_services"]
        table, report = sl.ancillary_by_row(merged, sl.load_ancillary(spec["file"]), spec["case"])
        out[sl.COLUMN_ANCILLARY] = table[sl.COLUMN_ANCILLARY]
        out[sl.COLUMN_ANCILLARY_VOLTAGE] = table[sl.COLUMN_ANCILLARY_VOLTAGE]
        out["_ancillary_report"] = report
    if support.get("efw_gate_fee"):
        spec = support["efw_gate_fee"]
        out[sl.COLUMN_GATE_FEE] = sl.gate_fee_by_row(merged, sl.gate_fee_gbp_per_mwh(spec["file"], spec["case"]))
    uplift = support.get("battery_subhourly_uplift")
    if uplift:
        # Decision 67: the batteries' declared sub-hourly revenue, before any top-up is sized.
        out[sl.COLUMN_SUBHOURLY] = sl.subhourly_by_row(merged, uplift["carriers"], uplift["share"])
    if support.get("life_extension_top_up"):
        private = private_frame.loc[private_frame["case_id"].eq("private")]
        booked = sl.book_support(book(private), {k: v for k, v in out.items() if not k.startswith("_")})
        out[sl.COLUMN_TOP_UP] = sl.top_up_by_row(booked, support["life_extension_top_up"]["technologies"])
    if support.get("battery_top_up"):
        # Decision 63: the battery rows' shortfall on their full required revenue, after every other line.
        private = private_frame.loc[private_frame["case_id"].eq("private")]
        booked = sl.book_support(book(private), {k: v for k, v in out.items() if not k.startswith("_")})
        out[sl.COLUMN_BATTERY_TOP_UP] = sl.battery_top_up_by_row(
            booked, support["battery_top_up"]["carriers"], ldes_carriers)
        if uplift:
            # Decision 67's high case, a memo: the top-up the batteries would need at the high uplift.
            high = {**{k: v for k, v in out.items() if not k.startswith("_") and k != sl.COLUMN_BATTERY_TOP_UP},
                    sl.COLUMN_SUBHOURLY: sl.subhourly_by_row(merged, uplift["carriers"], uplift["high_share"])}
            out[sl.COLUMN_BATTERY_TOP_UP_HIGH] = sl.battery_top_up_by_row(
                sl.book_support(book(private), high), support["battery_top_up"]["carriers"], ldes_carriers)
    if reference is not None:
        # A balancing run group: its bid stack moves no support payment (held at the market group's).
        columns = [c for c in out if not c.startswith("_")]
        out.update(sl.support_from_reference(reference, columns, merged))
        out["_support_reference"] = True
    return out


#: The support lines that are properties of the asset, sized without the private case's account.
ASSET_SUPPORT_LINES = ("ancillary_services", "efw_gate_fee", "battery_subhourly_uplift")


def asset_support_lines(merged: pd.DataFrame, support: Mapping | None) -> dict:
    """The asset-property support lines per asset id (:data:`ASSET_SUPPORT_LINES`), for the legacy valuation.

    The top-ups (life extension, battery) are sized on the private case's account, so they
    are not here: the acquisition plans are priced after that account is built, with the
    top-ups added (:data:`PURCHASE_TOP_UP_COLUMNS`; F19 valuation decision, user 14 Oct
    2026; before it the purchase left out the legacy batteries' GBP 29-30 m a year).
    """
    if not support:
        return {}
    spec = {k: v for k, v in support.items() if k in ASSET_SUPPORT_LINES or k == "reference"}
    if not any(k in spec for k in ASSET_SUPPORT_LINES):
        return {}
    lines = support_lines_by_asset(merged, merged.iloc[:0], spec, book=None, reference=spec.get("reference"))
    return {k: v for k, v in lines.items() if not k.startswith("_")}


def book_support(frame: pd.DataFrame, support_by_asset: Mapping) -> pd.DataFrame:
    """Decision 58: :func:`support_lines_2050.book_support` (unchanged without support lines)."""
    if not support_by_asset:
        return frame
    from scripts.public_ownership.support_lines_2050 import book_support as _book

    return _book(frame, {k: v for k, v in support_by_asset.items() if not k.startswith("_")})


def rab_allowed_revenue_support(private: pd.DataFrame) -> pd.Series:
    """FES 2050 decision 41 (c): the RAB support each Sizewell C row is paid, read off the private case.

    The RAB pays its allowed revenue, so the support is the private case's required revenue
    less its market and Capacity Market revenue, net of fuel: what makes the private row's
    producer rent zero (at cost). It is the regulated allowance whoever holds the equity, so
    every case books the same amount (a state share then earns the allowance on its own,
    cheaper, finance). Indexed by asset id; empty when the register holds no RAB row (2030).
    """
    if private.empty or "support_type" not in private.columns or "producer_rent_gbp" not in private.columns:
        return pd.Series(dtype=float)
    rab = private["support_type"].astype(str).eq(RAB_SUPPORT_TYPE) & private["producer_rent_gbp"].notna()
    if not rab.any():
        return pd.Series(dtype=float)
    rows = private.loc[rab]
    return pd.Series(-pd.to_numeric(rows["producer_rent_gbp"], errors="coerce").to_numpy(dtype=float),
                     index=rows["asset_id"].astype(str).to_numpy())


def book_rab_allowed_revenue(register: pd.DataFrame, support: pd.Series | None) -> pd.DataFrame:
    """Book :func:`rab_allowed_revenue_support` to every case's RAB rows (support, rent and the state's share).

    A register with no RAB support (every 2030 register) is returned unchanged, with no new column.
    """
    if support is None or support.empty or register.empty:
        return register
    register = register.copy()
    rab = register["support_type"].astype(str).eq(RAB_SUPPORT_TYPE) & register["producer_rent_gbp"].notna()
    paid = register["asset_id"].astype(str).map(support).where(rab, 0.0).fillna(0.0).astype(float)
    register["rab_support_gbp"] = paid
    ownership = pd.to_numeric(register["ownership_share"], errors="coerce").fillna(0.0)
    for column in ("support_revenue_gbp", "producer_rent_gbp"):
        register.loc[rab, column] = pd.to_numeric(register.loc[rab, column], errors="coerce").fillna(0.0) + paid[rab]
    register.loc[rab, "public_surplus_before_recycling_gbp"] = (
        pd.to_numeric(register.loc[rab, "public_surplus_before_recycling_gbp"], errors="coerce").fillna(0.0)
        + ownership[rab] * paid[rab])
    return register


def _valuation_preview(legacy: pd.DataFrame, rates: Mapping[str, dict]) -> pd.DataFrame:
    """The legacy rows as the legacy valuation reads them, for pricing a purchase (:func:`plan_acquisition`).

    F28 (fix, 28 Oct 2026): ``ownership_exclusion`` is carried where the register has it, so the
    valuation's ``scope_group`` excludes small-scale plant here as it does in the scope plan
    (:func:`acquisition_scope_plan`) and in the legacy valuation itself. Without it the priced
    plan valued rooftop solar the scope plan excludes, and :func:`check_plan_scope` refused the
    plan on every acquiring register with a small-scale exclusion.
    """
    def _col(name, default=np.nan):
        return (pd.to_numeric(legacy[name], errors="coerce") if name in legacy.columns
                else pd.Series(default, index=legacy.index)).to_numpy()

    preview = pd.DataFrame({
        "case_id": "private", "asset_id": legacy["asset_id"].to_numpy(),
        "component_type": legacy["component_type"].to_numpy(), "carrier": legacy["carrier"].to_numpy(),
        "finance_vintage": "legacy",
        "support_type": (legacy["support_type"] if "support_type" in legacy.columns
                         else pd.Series("", index=legacy.index)).to_numpy(),
        "cfd_round": (legacy["cfd_round"] if "cfd_round" in legacy.columns
                      else pd.Series("", index=legacy.index)).to_numpy(),
        "cfd_award_year": _col("cfd_award_year"),
        "capacity_mw": legacy["capacity_mw"].to_numpy(),
        "commissioning_year": _col("commissioning_year"),
        # F19 P3 decision 3: the CfD contract share of a cohort row, the gate fees on its output,
        # and a new contract's strike against its wholesale revenue.
        "cohort_legacy_share": _col("cohort_legacy_share"),
        "dispatch_mwh": _col("dispatch_mwh", 0.0),
        "wholesale_mwh": np.nan_to_num(_col("wholesale_mwh", 0.0)),
        "wholesale_revenue_gbp": np.nan_to_num(_col("wholesale_revenue_gbp", 0.0)),
        # F27 item 13: the balancing margin gas plant's path scales (settlement less redispatched variable cost).
        "balancing_settlement_gbp": np.nan_to_num(_col("balancing_settlement_gbp", 0.0)),
        "balancing_mwh": np.nan_to_num(_col("balancing_mwh", 0.0)),
        "market_revenue_gbp": _col("market_revenue_gbp", 0.0),
        "cfd_support_gbp": _col("cfd_support_gbp", 0.0),
        # Tells the valuation whether the compensation sits in market revenue (contract-value bids).
        "support_revenue_gbp": _col("support_revenue_gbp", 0.0),
        "cfd_curtailment_compensation_gbp": np.nan_to_num(_col("cfd_curtailment_compensation_gbp", 0.0)),
        "roc_support_gbp": _col("roc_support_gbp", 0.0),
        "fuel_carbon_gbp": _col("fuel_carbon_gbp", 0.0),
        "capacity_market_revenue_gbp": np.nan_to_num(_col("capacity_market_revenue_gbp", 0.0)),
        # Item D4(a): generator TNUoS and insurance are fixed operating costs of
        # the plant, so the purchase price is valued net of them, exactly as the
        # register costs them (``_generator_fixed_charges_gbp``, zero when off).
        "annual_fixed_om_gbp": (legacy["capacity_mw"].astype(float) * 1_000.0
                                * pd.to_numeric(legacy["fixed_om_gbp_per_kw_year"], errors="coerce")
                                + (pd.to_numeric(legacy["_generator_fixed_charges_gbp"], errors="coerce").fillna(0.0)
                                   if "_generator_fixed_charges_gbp" in legacy.columns else 0.0)).to_numpy(),
        # Item B1: the settlement's fuel and carbon no longer carry the network's
        # variable O&M, so the valuation deducts the register's own.
        "annual_variable_om_gbp": np.nan_to_num(
            pd.to_numeric(legacy["dispatch_mwh"], errors="coerce").fillna(0.0).to_numpy()
            * pd.to_numeric(legacy["variable_om_gbp_per_mwh"], errors="coerce").fillna(0.0).to_numpy()),
        "wacc": [
            (rates.get(canonical_technology(t)) or {}).get("wacc", {}).get("private", np.nan)
            for t in legacy["finance_technology"]
        ],
        # F29: the owners' rate is the finance technology's (workbook v2.0, Legacy Fleet column AF), and
        # the book value of a row the workbook has no cohort for is its as-new capex over its life.
        "finance_technology": legacy["finance_technology"].to_numpy(),
        "as_new_capex_gbp": _as_new_capex_vector(legacy),
        "economic_life_years": _col("economic_life_years"),
        "model_year": legacy["model_year"].to_numpy(),
    }, index=legacy.index)
    if "ownership_exclusion" in legacy.columns:
        preview["ownership_exclusion"] = legacy["ownership_exclusion"].to_numpy()
    # Fix 15 Nov 2026: the contract income outside cfd_support_gbp that the legacy valuation counts (its 8 Nov fix,
    # user: P3 buys the CCS plants): Net Zero Teesside's DPA payments and Drax BECCS's carbon-removal CfD. Without
    # them the purchase priced NZT at GBP 0.56 bn and Drax BECCS at 0.35 bn against the valuation's 3.16 and 1.79 bn,
    # while the owners give the income up with the plant.
    for column in ("dpa_support_gbp", "carbon_removal_cfd_revenue_gbp"):
        if column in legacy.columns:
            preview[column] = np.nan_to_num(_col(column, 0.0))
    return preview


def plan_acquisition(
    merged: pd.DataFrame, rates: Mapping[str, dict], finance_snapshot: dict, acquisition: Mapping,
    ro_stations: pd.DataFrame, *, public_rate: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Price every legacy row for a case that acquires the legacy fleet (items N2 and N4).

    Returns a frame indexed like ``merged`` with ``acquirable``, ``legacy_scope_group``,
    ``acquisition_price_gbp`` and ``acquisition_basis`` per row, and a summary.

    * The valued group (renewables and nuclear) is priced on
      ``purchase_price_basis``: ``state_present_value`` takes each row at its present
      value to the state, ``market_value`` at its owners' value (F29: workbook v2.0's
      Approach 2, after tax, from :mod:`legacy_valuation`).  A row whose value is
      negative is priced at zero (nobody pays for a liability); its loss still shows
      in the account.
    * Thermal and storage rows take ``thermal_basis`` under those two bases:
      ``market_value`` (the central since F19 P3 decision 3, 6 Oct 2026) is each row
      at its owners' value, floored at zero, as the valued group is; or ``zero``.
    * F29 (user, 26 Oct 2026): the workbook's other acquisition bases price every
      acquired row, thermal and storage too: ``book_value`` (A1, the net book value),
      ``no_support`` (A2 without CfD and RO support) and ``ro_pass_through`` (A3: RO
      plant at its no-support value, the rest at owners' value).
    * Each acquired row carries its annuity term and factor (``acquisition_term_years``,
      ``acquisition_annuity_factor``): today one term for all, the hook a per-asset
      repayment (F28) fills.
    * The valuation's price after the model year is the case's
      ``legacy_price_path``: ``cp30_level`` puts the rows on
      ``acquisition["price_level_settlement"]`` (the asset settlement at CP30's
      price level, ``price_level_settlement.py``); without that table, or under
      ``model_2030``, every year keeps the model year's price, and the summary's
      ``legacy_price_path_applied`` says which was used.
    * Foreign plant (``EU_market``) is not acquirable.

    ``public_rate`` is the rate the purchase is valued and annuitised at (the
    public legacy rate under the case's basis and yield levers); the snapshot's
    real rate when not given.
    """
    if not acquisition:
        raise ValueError("acquiring the legacy fleet needs public_ownership.acquisition")
    valuation_block = dict(acquisition.get("valuation") or {})
    if str(acquisition.get("asset_lives", LIVES_CENTRAL)) == LIVES_EXTENDED:
        # Lever legacy_asset_lives (29 Sep 2026): renewables live longer and Sizewell B
        # runs to its extension date, so the flows (and a market-value price) run on.
        valuation_block = extended_lives_block(valuation_block, acquisition.get("extended_asset_lives"))
    settings = valuation_settings(valuation_block)
    legacy = merged.loc[merged["finance_vintage"].astype(str).eq("legacy")]
    required = {"market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp", "cfd_support_gbp", "roc_support_gbp"}
    if not required.issubset(merged.columns):
        raise ValueError(f"acquiring the legacy fleet needs the WP-D settlement columns on the register: {sorted(required)}")
    if public_rate is None:
        public_rate = float(finance_snapshot["assumptions"]["public_borrowing_real"]["value"])

    preview = _valuation_preview(legacy, rates)
    # F19 P3 decision 3: asset-property support lines (ancillary services, sub-hourly uplift)
    # are operating cash the owners keep, as in the legacy valuation's own rows.
    for column, by_asset in (acquisition.get("support_by_asset") or {}).items():
        preview[column] = (pd.Series(by_asset).reindex(legacy["asset_id"].astype(str)).fillna(0.0).to_numpy(dtype=float)
                           if len(pd.Series(by_asset)) else 0.0)
    # Purchase audit (15 Nov 2026): the owners' book value and fixed costs as the private register books them, which
    # the legacy valuation reads. The asset inputs carry calibrated capex before the auction-margin scale, so CfD
    # marine (the one calibrated cohort valued on its own capex) was priced 1.2% above the owners' value.
    for column, by_asset in (acquisition.get("owners_view_by_asset") or {}).items():
        held = pd.Series(by_asset, dtype=float).reindex(legacy["asset_id"].astype(str)).to_numpy()
        preview[column] = np.where(np.isnan(held), pd.to_numeric(preview[column], errors="coerce").to_numpy(), held)
    # F27 item 6 (24 Oct 2026): under cost-based capacity contracts announced before the purchase,
    # the owners value each existing unit on its cost-based payment, not its Capacity Market income.
    capacity_terms = None
    if str(acquisition.get("capacity_procurement", capacity_procurement.AUCTION)) == capacity_procurement.COST_BASED_EXISTING:
        terms_cb = capacity_procurement.cost_based_payments(preview, acquisition.get("capacity_cost_based_existing"))
        capacity_terms = {
            "capacity_market_receipt_existing_gbp": float(terms_cb.loc[terms_cb["existing"], "capacity_market_receipt_gbp"].sum()),
            "cost_based_payment_existing_gbp": float(terms_cb.loc[terms_cb["existing"], "cost_based_payment_gbp"].sum()),
        }
        preview["capacity_market_revenue_gbp"] = terms_cb["cost_based_payment_gbp"].to_numpy()
    # F19 P3 decision 2: the declared price path after the model year.
    path = str(acquisition.get("legacy_price_path", PRICE_PATH_MODEL))
    if path not in (PRICE_PATH_CP30, PRICE_PATH_MODEL, PRICE_PATH_ANCHORED, PRICE_PATH_TODAY):
        raise ValueError(f"legacy_price_path={path!r}; supported: "
                         f"{[PRICE_PATH_ANCHORED, PRICE_PATH_CP30, PRICE_PATH_MODEL, PRICE_PATH_TODAY]}")
    path_table = acquisition.get("price_level_settlement")
    applied = PRICE_PATH_MODEL
    if path in (PRICE_PATH_ANCHORED, PRICE_PATH_TODAY) and acquisition.get("price_anchors") is not None:
        # User decision (10 Nov 2026): each technology's capture price through today and every solved year.
        from scripts.public_ownership.legacy_valuation import with_model_year_streams

        # User, 11 Nov 2026: under anchor_streams all, the model year's streams per MW are this register's own.
        anchors = with_model_year_streams(acquisition["price_anchors"], merged, int(merged["model_year"].iloc[0]),
                                          settings) if "model_year" in merged.columns else acquisition["price_anchors"]
        settings = dict(settings, price_path=path, price_anchors=anchors)
        applied = path
    if path == PRICE_PATH_CP30 and path_table is not None:
        preview = attach_price_path(preview, path_table)
        applied = PRICE_PATH_CP30
    valuation, _, valuation_summary = value_legacy_fleet(preview, public_rate, settings, ro_stations, finance_snapshot)
    valuation.index = legacy.index
    # F28 (e): each row's repayment term (its remaining life central, or one term for all).
    term_setting = acquisition.get("term_years", 30)
    row_terms = pd.Series(acquisition_row_terms(term_setting, valuation["retirement_year"], legacy["model_year"]),
                          index=legacy.index)

    plan = pd.DataFrame(index=merged.index)
    plan["legacy_scope_group"] = None
    plan.loc[legacy.index, "legacy_scope_group"] = valuation["scope_group"].to_numpy()
    plan["acquirable"] = ~merged["carrier"].astype(str).isin(settings["excluded_carriers"])
    plan["acquisition_price_gbp"] = np.nan
    plan["acquisition_basis"] = None

    # Valued group: renewables and nuclear.
    basis = str(acquisition.get("purchase_price_basis", PRICE_STATE_PV))
    if basis not in PURCHASE_PRICE_BASES:
        from scripts.public_ownership.levers import RETIRED_VALUES

        note = RETIRED_VALUES.get((LEVER_PURCHASE_PRICE_BASIS, basis))
        raise ValueError(f"acquisition.purchase_price_basis={basis!r}: "
                         + (note or f"supported: {list(PURCHASE_PRICE_BASES)}"))
    valued = valuation.loc[valuation["scope_group"].eq(GROUP_VALUED)]
    price = workbook_basis_price(valued, basis)
    # Item G3, lever L19. A cancelled treatment takes the CfD top-up and ROC
    # income streams out of the price, and the former owners are paid a residual:
    # those streams at their own WACC. Honoured and cancelled then differ by the
    # rate differential on the streams alone, never by the streams twice.
    treatment = str(acquisition.get("legacy_contracts", "honoured"))
    cancelled_streams = ("cfd_top_up", "roc_income")
    streams_state = sum(valued[f"pv_state_{st}_gbp"] for st in cancelled_streams)
    streams_market = sum(valued[f"pv_market_{st}_gbp"] for st in cancelled_streams)
    plan["pv_state_cancelled_streams_gbp"] = np.nan
    plan["pv_market_cancelled_streams_gbp"] = np.nan
    plan["cancelled_contract_residual_pv_gbp"] = 0.0
    plan["cancelled_residual_route"] = ""
    if treatment != "honoured":
        if basis not in ("state_present_value", "market_value"):
            raise ValueError(
                f"legacy contracts {treatment!r} cancelled under purchase_price_basis {basis!r}: that price "
                "cannot say which part of it was the contracts; use state_present_value or market_value")
        if basis == "state_present_value":
            price = (valued["pv_state_gbp"] - streams_state).clip(lower=0.0)
        else:
            price = (valued["pv_market_gbp"] - streams_market).clip(lower=0.0)
        plan.loc[valued.index, "pv_state_cancelled_streams_gbp"] = streams_state.to_numpy()
        plan.loc[valued.index, "pv_market_cancelled_streams_gbp"] = streams_market.to_numpy()
        plan.loc[valued.index, "cancelled_contract_residual_pv_gbp"] = streams_market.clip(lower=0.0).to_numpy()
        plan.loc[valued.index, "cancelled_residual_route"] = (
            "bills" if treatment == "cancelled_residual_on_bills" else "exchequer")
    # User decision (24 Oct 2026): lever legacy_offshore_price.  Under transaction_multiple every
    # legacy offshore row of the valued group is priced at observed deal multiples per MW instead of
    # its owners' value (scripts/public_ownership/offshore_deal_multiple.py); nothing else moves.
    offshore_price = str(acquisition.get("legacy_offshore_price", offshore_deal_multiple.OWNERS_VALUE))
    offshore_terms = None
    basis_by_row = pd.Series(basis, index=valued.index, dtype=object)
    if offshore_price == offshore_deal_multiple.TRANSACTION_MULTIPLE:
        if treatment != "honoured":
            raise ValueError("legacy_offshore_price transaction_multiple prices the farms with their contracts (the "
                             f"deals bought them so); it cannot be combined with legacy contracts {treatment!r}")
        offshore_terms = offshore_deal_multiple.multiple(acquisition.get("offshore_transaction_multiple"))
        offshore = valued["carrier"].astype(str).eq(offshore_terms["carrier"])
        owners = price.loc[offshore]
        price = price.copy()
        price.loc[offshore] = offshore_deal_multiple.price_rows(valued.loc[offshore, "capacity_mw"],
                                                                offshore_terms["gbp_m_per_mw"])
        basis_by_row.loc[offshore] = offshore_deal_multiple.BASIS
        offshore_terms.update({
            "capacity_mw": float(pd.to_numeric(valued.loc[offshore, "capacity_mw"], errors="coerce").sum()),
            "owners_value_price_gbp": float(owners.sum()), "transaction_multiple_price_gbp": float(price.loc[offshore].sum()),
        })
    elif offshore_price != offshore_deal_multiple.OWNERS_VALUE:
        raise ValueError(f"legacy_offshore_price={offshore_price!r}; supported: "
                         f"{[offshore_deal_multiple.OWNERS_VALUE, offshore_deal_multiple.TRANSACTION_MULTIPLE]}")
    # Backlog AP1 (29 Sep 2026): a control premium over the valued group's price.
    control = str(acquisition.get("generation_control_premium", CONTROL_PREMIUM_NONE))
    control_share = float((acquisition.get("generation_control_premium_share") or {}).get(control, 0.0)) \
        if control != CONTROL_PREMIUM_NONE else 0.0
    price = price * (1.0 + control_share)
    plan.loc[valued.index, "acquisition_price_gbp"] = price.to_numpy()
    plan.loc[valued.index, "acquisition_basis"] = (basis_by_row if not control_share
                                                   else basis_by_row + f"+control_premium_{control}").to_numpy()
    plan["hpc_completion_sunk_cost_gbp"] = np.nan
    plan["hpc_completion_remaining_capex_gbp"] = np.nan
    if str(acquisition.get("in_construction", IN_CONSTRUCTION_PRIVATE)) == IN_CONSTRUCTION_PUBLIC:
        # User decision 10: the state completes Hinkley Point C itself.  Its completion
        # capital replaces its purchase price (never both), on the same annuity.
        table = acquisition.get("hpc_completion_table")
        if table is None or len(table) == 0:
            raise ValueError("in_construction: public_completion needs public_ownership.acquisition.hpc_completion "
                             "(data/public_ownership/financing/hpc_completion_cost.csv)")
        premium = float(acquisition.get("hpc_construction_premium", 0.0))
        completed = 0
        for idx in valued.index:
            terms_row = hpc_completion_price(merged.loc[idx], table, public_rate=public_rate,
                                             term_years=float(row_terms.loc[idx]), premium=premium)
            if terms_row is None:
                continue
            completed += 1
            plan.loc[idx, "acquisition_price_gbp"] = terms_row["price_gbp"]
            plan.loc[idx, "acquisition_basis"] = "hpc_public_completion"
            plan.loc[idx, "hpc_completion_sunk_cost_gbp"] = terms_row["sunk_cost_gbp"]
            plan.loc[idx, "hpc_completion_remaining_capex_gbp"] = terms_row["remaining_capex_gbp"]
        if not completed:
            raise ValueError("in_construction: public_completion matched no acquired legacy row; check asset_match "
                             "in the HPC completion table")

    # Thermal and storage: owners' value (central) or zero under the state- and market-value bases;
    # F29: the workbook's A1, A2 no-support and A3 bases price them as they price the valued group.
    thermal_basis = str(acquisition.get("thermal_basis", THERMAL_MARKET_VALUE))
    if thermal_basis not in THERMAL_BASES:
        from scripts.public_ownership.levers import RETIRED_VALUES

        note = RETIRED_VALUES.get((LEVER_THERMAL_BASIS, thermal_basis))
        raise ValueError(f"acquisition.thermal_basis={thermal_basis!r}: " + (note or f"supported: {list(THERMAL_BASES)}"))
    other = valuation.loc[~valuation["scope_group"].isin([GROUP_VALUED, GROUP_EXCLUDED])]
    if basis in (PRICE_STATE_PV, PRICE_MARKET_VALUE):
        thermal_price = (pd.Series(0.0, index=other.index) if thermal_basis == "zero"
                         else workbook_basis_price(other, PRICE_MARKET_VALUE))
        thermal_label = f"thermal_{thermal_basis}"
    else:
        thermal_price, thermal_label = workbook_basis_price(other, basis), f"thermal_{basis}"
    plan.loc[other.index, "acquisition_price_gbp"] = thermal_price.to_numpy()
    plan.loc[other.index, "acquisition_basis"] = thermal_label
    excluded_rows = valuation.index[valuation["scope_group"].eq(GROUP_EXCLUDED).to_numpy()]
    plan.loc[excluded_rows, "acquirable"] = False
    # User decision (10 Nov 2026): the case's own exclusions (energy-from-waste, Hinkley left private).
    dropped = case_purchase_exclusions(merged, acquisition)
    plan.loc[dropped, "acquirable"] = False
    plan.loc[dropped, "acquisition_price_gbp"] = np.nan
    plan.loc[dropped, "acquisition_basis"] = None

    # F28 (e): the repayment term and annuity factor per acquired row; _key_account reads the columns.
    priced = plan["acquisition_price_gbp"].notna()
    plan["acquisition_term_years"] = np.where(priced, row_terms.reindex(plan.index).to_numpy(dtype=float), np.nan)
    plan["acquisition_annuity_factor"] = np.where(priced, crf_rows(public_rate, plan["acquisition_term_years"]), np.nan)
    book_value = pd.to_numeric(valuation.get("net_book_value_gbp"), errors="coerce") if "net_book_value_gbp" in \
        valuation.columns else pd.Series(np.nan, index=valuation.index)
    summary = {
        "purchase_price_basis": basis, "thermal_basis": thermal_basis,
        **plan_term_summary(plan, public_rate, term_setting),
        "legacy_contracts": treatment,
        "in_construction": str(acquisition.get("in_construction", IN_CONSTRUCTION_PRIVATE)),
        "generation_control_premium": control, "generation_control_premium_share": control_share,
        "asset_lives": str(acquisition.get("asset_lives", LIVES_CENTRAL)),
        "legacy_price_path": path, "legacy_price_path_applied": applied,
        "owner_value_method": str(settings["owner_value_method"]),
        "purchase_basis_label": purchase_basis_label(basis, thermal_basis),
        # F29: the workbook's A1 net book value of thermal and storage, beside the owners' value.
        "thermal_and_storage_book_value_gbp": float(book_value.loc[other.index].clip(lower=0.0).sum()),
        "valued_book_value_gbp": float(book_value.loc[valued.index].clip(lower=0.0).sum()),
        "thermal_and_storage_owners_pv_gbp": float(
            pd.to_numeric(valuation.loc[other.index, "pv_market_gbp"], errors="coerce").fillna(0.0).clip(lower=0.0).sum()),
        "cancelled_contract_residual_pv_gbp": float(plan["cancelled_contract_residual_pv_gbp"].sum()),
        "public_real_rate": public_rate,
        "valued_purchase_price_gbp": float(plan.loc[valued.index, "acquisition_price_gbp"].sum()),
        "thermal_and_storage_purchase_price_gbp": float(plan.loc[other.index, "acquisition_price_gbp"].sum()),
        "valued_pv_state_gbp": valuation_summary["valued_pv_state_gbp"],
        "valued_pv_market_gbp": valuation_summary["valued_pv_market_gbp"],
        "excluded_capacity_mw": float(merged.loc[~plan["acquirable"], "capacity_mw"].sum()),
        "excluded_carriers": sorted(merged.loc[~plan["acquirable"], "carrier"].astype(str).unique()),
    }
    summary["annual_acquisition_charge_gbp"] = _annual_charge(
        plan, plan.index.isin(valued.index.union(other.index)))
    summary["cancelled_contract_residual_annual_gbp"] = float(
        (pd.to_numeric(plan["cancelled_contract_residual_pv_gbp"], errors="coerce").fillna(0.0)
         * pd.to_numeric(plan["acquisition_annuity_factor"], errors="coerce").fillna(0.0)).sum())
    # F27 item 6: the model year's Capacity Market income the valuation replaced with cost-based payments.
    summary["capacity_procurement"] = str(acquisition.get("capacity_procurement", capacity_procurement.AUCTION))
    if capacity_terms is not None:
        summary["capacity_procurement_valuation"] = capacity_terms
    # User decision (24 Oct 2026): legacy offshore wind's price basis, and the deals it was priced at.
    summary["legacy_offshore_price"] = offshore_price
    if offshore_terms is not None:
        summary["offshore_transaction_multiple"] = offshore_terms
    return plan, summary


def workbook_basis_price(valuation: pd.DataFrame, basis: str) -> pd.Series:
    """Each valued row's purchase price on one acquisition basis, floored at zero (F29, workbook v2.0).

    ``state_present_value`` (the state's own value, pre-tax at the public rate), ``market_value``
    (A2, the owners' after-tax value), ``book_value`` (A1, the net book value), ``no_support``
    (A2 without CfD and RO support) and ``ro_pass_through`` (A3: RO plant at no-support, the
    rest at A2).
    """
    def column(name):
        if name not in valuation.columns:
            raise ValueError(f"the legacy valuation has no {name} column for purchase_price_basis {basis!r}")
        return pd.to_numeric(valuation[name], errors="coerce").fillna(0.0).clip(lower=0.0)

    if basis == PRICE_STATE_PV:
        return column("pv_state_gbp")
    if basis == PRICE_MARKET_VALUE:
        return column("pv_market_gbp")
    if basis == PRICE_BOOK_VALUE:
        return column("net_book_value_gbp")
    if basis == PRICE_NO_SUPPORT:
        return column("pv_no_support_gbp")
    if basis == PRICE_RO_PASS_THROUGH:
        from scripts.public_ownership.legacy_valuation import is_ro_row

        return column("pv_no_support_gbp").where(is_ro_row(valuation["support_type"]), column("pv_market_gbp"))
    raise ValueError(f"unknown purchase_price_basis {basis!r}")


def _as_new_capex_vector(rows: pd.DataFrame) -> np.ndarray:
    """Each row's as-new capex (GBP) at its effective power capex plus any storage energy (F29: the book value
    of a legacy row the workbook has no cohort for)."""
    power, _ = _effective_capex_power_vector(rows)
    energy = (pd.to_numeric(rows.get("energy_capacity_mwh"), errors="coerce").fillna(0.0).to_numpy()
              * pd.to_numeric(rows.get("capex_energy_gbp_per_kwh"), errors="coerce").fillna(0.0).to_numpy()
              if {"energy_capacity_mwh", "capex_energy_gbp_per_kwh"}.issubset(rows.columns) else 0.0)
    return pd.to_numeric(rows["capacity_mw"], errors="coerce").to_numpy(dtype=float) * 1_000.0 * power + energy * 1_000.0


def _annual_charge(plan: pd.DataFrame, mask) -> float:
    """The first year's purchase annuity over the masked rows: each row's price at its own factor (F28 (e))."""
    price = pd.to_numeric(plan["acquisition_price_gbp"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    factor = pd.to_numeric(plan["acquisition_annuity_factor"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return float((price * factor)[np.asarray(mask, dtype=bool)].sum())


def plan_term_summary(plan: pd.DataFrame, public_rate: float, term_setting) -> dict:
    """The plan's repayment terms in summary (F28 (e)).

    ``term_basis`` is ``remaining_life`` or ``fixed``.  Under remaining life ``term_years`` is the
    price-weighted mean term and ``annuity_factor`` the price-weighted factor (a total price times
    it is the first year's annuity); ``last_payment_offset_years`` is the longest term.  Under a
    fixed term all three are that term's.
    """
    if str(term_setting) != TERM_REMAINING_LIFE:
        fixed = float(term_setting)
        return {"term_basis": "fixed", "term_years": fixed, "annuity_factor": capital_recovery_factor(public_rate, fixed),
                "last_payment_offset_years": fixed}
    price = pd.to_numeric(plan["acquisition_price_gbp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    term = pd.to_numeric(plan["acquisition_term_years"], errors="coerce")
    factor = pd.to_numeric(plan["acquisition_annuity_factor"], errors="coerce")
    priced = term.notna()
    weight = float(price[priced].sum())
    if weight > 0:
        mean_term = float((price[priced] * term[priced]).sum() / weight)
        mean_factor = float((price[priced] * factor[priced]).sum() / weight)
    else:
        mean_term = float(term[priced].mean()) if priced.any() else float("nan")
        mean_factor = capital_recovery_factor(public_rate, mean_term) if np.isfinite(mean_term) else float("nan")
    return {"term_basis": TERM_REMAINING_LIFE, "term_years": mean_term, "annuity_factor": mean_factor,
            "last_payment_offset_years": float(term[priced].max()) if priced.any() else float("nan")}


#: Network bounds (18-19 Oct 2026): the columns read from the register a group holds its purchase from.
HELD_ACQUISITION_COLUMNS = ("case_id", "asset_id", "finance_vintage", "acquisition_price_gbp",
                            "annual_cancelled_contract_residual_gbp")


#: Relative tolerance within which the held cases on one plan must agree on an asset's price (20 Oct 2026).
HELD_PRICE_RTOL = 1e-9


def _legacy_finance_eligible(merged: pd.DataFrame) -> np.ndarray:
    """Rows the register can cost: ``finance_eligible`` and ``evidence_status == 'resolved'`` where present.

    The register writes an acquisition price only on an eligible row a case acquires, so the held
    price check compares eligible rows only (OCGT, oil and gas engines are legacy, priced by the
    plan and never costed).
    """
    ok = np.ones(len(merged), dtype=bool)
    if "finance_eligible" in merged.columns:
        ok &= merged["finance_eligible"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).to_numpy()
    if "evidence_status" in merged.columns:
        ok &= merged["evidence_status"].astype(str).eq("resolved").to_numpy()
    return ok


def _held_by_asset(rows: pd.DataFrame, column: str, case_ids: Sequence[str]) -> pd.Series:
    """One value per asset from the held cases' rows, refused where two cases disagree."""
    values = pd.to_numeric(rows[column], errors="coerce")
    frame = pd.DataFrame({"asset_id": rows["asset_id"].astype(str).to_numpy(), "value": values.to_numpy()}).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    stats = frame.groupby("asset_id")["value"].agg(["min", "max"])
    spread = stats["max"] - stats["min"]
    bad = spread > HELD_PRICE_RTOL * (1.0 + stats["max"].abs())
    if bad.any():
        raise ValueError(f"the held cases {list(case_ids)[:5]} disagree on {column} for "
                         f"{int(bad.sum())} legacy assets (e.g. {sorted(stats.index[bad])[:5]}); cases on one "
                         "acquisition plan must be priced alike")
    return stats["max"]


def hold_acquisition_prices(plan: pd.DataFrame, summary: dict, merged: pd.DataFrame, held: pd.DataFrame,
                            case_ids: str | Sequence[str], source: str | None = None, *,
                            acquires_all: bool = True) -> tuple[pd.DataFrame, dict]:
    """The acquisition plan with each legacy row's price taken from another group's register (network bounds).

    User decision (18-19 Oct 2026): the P3 purchase is struck at the CENTRAL network's market value
    and held across the network bounds, which show network risk, not a different purchase. Each legacy
    row this plan's cases acquire takes the price (and, under cancelled legacy contracts, the residual
    owed to the former owners) the held register (``purchase_price_from``'s) carries for it; every other
    column of the plan, the scope groups and the plant the state never owns among them, is this group's own.

    The plan key leaves out the ``legacy_fleet`` lever (20 Oct 2026), so one plan serves the cases that
    buy only the valued group and those that buy everything, and the register writes a price only on the
    rows each case acquires. The held price of an asset is therefore taken from the union of
    ``case_ids`` (every case on this plan), and the cases must agree on it. The set check covers the
    finance-eligible legacy rows the plan's cases acquire (the valued group, or every acquirable row when
    ``acquires_all``); a row priced in scope on one side and not the other is refused, since the two
    fleets must be the same build. A legacy row outside that scope keeps this plan's own price: no case
    on the plan buys it, so no register row or per-case summary reads it.

    Args:
        plan: :func:`plan_acquisition`'s frame (indexed like ``merged``).
        summary: Its summary; the price totals and the annual charge are recomputed on the held prices,
            and ``purchase_price_held_from`` records the source and both totals.
        merged: The costed register rows (``asset_id`` and ``finance_vintage``; ``finance_eligible`` and
            ``evidence_status`` where present).
        held: The held register's legacy rows for the plan's cases (:data:`HELD_ACQUISITION_COLUMNS`).
        case_ids: Every case on this plan's acquisition terms (one id is accepted).
        source: The held register's path, for the record.
        acquires_all: Whether any of ``case_ids`` buys the whole legacy fleet (``legacy_fleet: acquire_all``);
            False when they all buy only the valued group.

    Returns:
        ``(plan, summary)`` on the held prices.
    """
    ids_wanted = [str(case_ids)] if isinstance(case_ids, str) else [str(c) for c in case_ids]
    rows = held.loc[held["case_id"].astype(str).isin(ids_wanted)
                    & held["finance_vintage"].astype(str).eq("legacy")]
    if rows.empty:
        raise ValueError(f"the held register {source or ''} carries no legacy rows for cases {ids_wanted[:5]}; "
                         "purchase_price_from must name a group whose case list includes them")
    if rows[["case_id", "asset_id"]].astype(str).duplicated().any():
        raise ValueError(f"the held register repeats a legacy asset for one of the cases {ids_wanted[:5]}")
    prices = _held_by_asset(rows, "acquisition_price_gbp", ids_wanted)
    legacy = merged["finance_vintage"].astype(str).eq("legacy").to_numpy()
    ids = merged["asset_id"].astype(str).to_numpy()
    own = pd.to_numeric(plan["acquisition_price_gbp"], errors="coerce").to_numpy(dtype=float)
    priced = legacy & np.isfinite(own)
    group = plan["legacy_scope_group"].astype(str).to_numpy()
    in_scope = _legacy_finance_eligible(merged)
    if "acquirable" in plan.columns:
        in_scope &= plan["acquirable"].fillna(False).astype(bool).to_numpy()
    if not acquires_all:
        in_scope &= group == GROUP_VALUED
    held_rows = priced & in_scope
    here, there = set(ids[held_rows]), set(prices.index)
    if here != there:
        raise ValueError(f"cases {ids_wanted[:5]}: the held register prices a different legacy fleet (only here: "
                         f"{sorted(here - there)[:5]}, only there: {sorted(there - here)[:5]}); the network bounds "
                         "must share one build")
    out = plan.copy()
    out.loc[held_rows, "acquisition_price_gbp"] = prices.reindex(ids[held_rows]).to_numpy(dtype=float)
    # F28 (e): each row's own annuity factor (the plan's; one factor for all under a fixed term).
    factor = (pd.to_numeric(out["acquisition_annuity_factor"], errors="coerce").to_numpy(dtype=float)
              if "acquisition_annuity_factor" in out.columns else np.full(len(out), float(summary["annuity_factor"])))
    if "annual_cancelled_contract_residual_gbp" in rows.columns:
        residual = _held_by_asset(rows, "annual_cancelled_contract_residual_gbp", ids_wanted)
        own_factor = factor[held_rows]
        positive = np.isfinite(own_factor) & (own_factor > 0)
        out.loc[held_rows, "cancelled_contract_residual_pv_gbp"] = np.where(
            positive, residual.reindex(ids[held_rows]).fillna(0.0).to_numpy(dtype=float)
            / np.where(positive, own_factor, 1.0), 0.0)
    valued = priced & (group == GROUP_VALUED)
    other = priced & (group != GROUP_VALUED) & (group != GROUP_EXCLUDED)
    price = pd.to_numeric(out["acquisition_price_gbp"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    kept = priced & ~held_rows
    held_summary = dict(summary)
    held_summary["purchase_price_held_from"] = {
        "register": source, "case_id": ids_wanted[0], "case_ids": ids_wanted, "acquires_all": bool(acquires_all),
        "own_valued_purchase_price_gbp": float(summary["valued_purchase_price_gbp"]),
        "own_thermal_and_storage_purchase_price_gbp": float(summary["thermal_and_storage_purchase_price_gbp"]),
        "held_rows": int(held_rows.sum()),
        "own_price_rows_outside_scope": int(kept.sum()),
        "own_price_outside_scope_gbp": float(price[kept].sum()),
        "rule": "user decision 18-19 Oct 2026: the purchase is struck at the CENTRAL network's market value and "
                "held across the network bounds",
    }
    held_summary["valued_purchase_price_gbp"] = float(price[valued].sum())
    held_summary["thermal_and_storage_purchase_price_gbp"] = float(price[other].sum())
    held_summary["annual_acquisition_charge_gbp"] = float(np.nansum((price * factor)[valued | other]))
    if held_summary.get("term_basis") == TERM_REMAINING_LIFE:
        held_summary.update(plan_term_summary(out, float(summary["public_real_rate"]), TERM_REMAINING_LIFE))
    held_summary["cancelled_contract_residual_pv_gbp"] = float(
        pd.to_numeric(out["cancelled_contract_residual_pv_gbp"], errors="coerce").fillna(0.0).sum())
    held_summary["cancelled_contract_residual_annual_gbp"] = float(np.nansum(
        pd.to_numeric(out["cancelled_contract_residual_pv_gbp"], errors="coerce").fillna(0.0).to_numpy() * factor))
    return out, held_summary


def purchase_basis_label(basis: str, thermal_basis: str) -> str:
    """The purchase basis in words, group by group (F19 P3 decision 3: label it honestly)."""
    # F29: the workbook's A1, A2 no-support and A3 bases price every acquired row alike.
    whole_fleet = {PRICE_BOOK_VALUE: "the whole fleet at net book value (workbook A1)",
                   PRICE_NO_SUPPORT: "the whole fleet at owners' value without CfD and RO support (workbook A2 no-support)",
                   PRICE_RO_PASS_THROUGH: ("RO plant at its no-support value, the rest at owners' value (workbook A3, "
                                           "a policy case embedding a transfer)")}
    if basis in whole_fleet:
        return whole_fleet[basis]
    valued = {PRICE_MARKET_VALUE: "renewables and nuclear at owners' value (after tax, workbook A2)",
              PRICE_STATE_PV: "renewables and nuclear at their value to the state"}.get(basis, basis)
    thermal = {THERMAL_MARKET_VALUE: "thermal and storage at owners' value", "zero": "thermal and storage at no price"}.get(
        thermal_basis, thermal_basis)
    return f"{valued}, {thermal}"


def carried_acquisition_plan(merged: pd.DataFrame, acquisition: Mapping, *, public_rate: float) -> tuple[pd.DataFrame, dict]:
    """The acquisition plan of a vintaged register (FES 2050 plan, tasks Y10 and Y12): the 2030 purchase carried.

    The state bought the legacy fleet in 2030 (decision 2) and pays that annuity in
    the model year whatever survives, in Y12's carried line
    (``acquisition_carry_forward.py``).  So a surviving acquired row is NOT re-priced
    here: its purchase price is zero and its basis ``carried_from_2030``, and its
    required revenue is operating cost only; pricing it again would count the fleet
    twice in the public account and the Exchequer ledger.  Every legacy row of a
    vintaged register is taken to be a 2030 survivor (Y12's
    ``check_register_survivors`` compares the two).  Scope groups and the plant the
    state never owns come from the valuation settings, as in 2030.

    Args:
        merged: The costed register rows.
        acquisition: ``public_ownership.acquisition`` (its ``valuation`` settings).
        public_rate: The case's public legacy rate (reported only).

    Returns:
        ``(plan, summary)`` in :func:`plan_acquisition`'s shape.
    """
    from scripts.public_ownership.legacy_valuation import scope_group

    settings = valuation_settings(dict((acquisition or {}).get("valuation") or {}))
    legacy = merged["finance_vintage"].astype(str).eq("legacy")
    plan = pd.DataFrame(index=merged.index)
    plan["legacy_scope_group"] = None
    plan.loc[legacy, "legacy_scope_group"] = scope_group(merged.loc[legacy], settings).to_numpy()
    plan["acquirable"] = ~merged["carrier"].astype(str).isin(settings["excluded_carriers"])
    plan.loc[legacy & plan["legacy_scope_group"].eq(GROUP_EXCLUDED), "acquirable"] = False
    # User decision (10 Nov 2026): the same case exclusions as plan_acquisition, so the scope plan matches.
    plan.loc[case_purchase_exclusions(merged, acquisition), "acquirable"] = False
    plan["acquisition_price_gbp"] = np.where(legacy & plan["acquirable"], 0.0, np.nan)
    plan["acquisition_basis"] = np.where(legacy & plan["acquirable"], CARRIED_ACQUISITION_BASIS, None)
    for column in ("pv_state_cancelled_streams_gbp", "pv_market_cancelled_streams_gbp",
                   "hpc_completion_sunk_cost_gbp", "hpc_completion_remaining_capex_gbp"):
        plan[column] = np.nan
    plan["cancelled_contract_residual_pv_gbp"] = 0.0
    plan["cancelled_residual_route"] = ""
    # Nothing is priced here: the 2030 purchase is repaid by Y12's carried line (under a remaining-life
    # term it follows each 2030 asset's own term, acquisition_carry_forward).
    term_setting = (acquisition or {}).get("term_years", 30)
    fixed = str(term_setting) != TERM_REMAINING_LIFE
    factor = (capital_recovery_factor(float(public_rate), float(term_setting))
              if fixed and float(public_rate) > 0 else 0.0)
    plan["acquisition_term_years"] = np.where(plan["acquisition_price_gbp"].notna(),
                                              float(term_setting) if fixed else np.nan, np.nan)
    plan["acquisition_annuity_factor"] = np.where(plan["acquisition_price_gbp"].notna(), factor, np.nan)
    summary = {
        "purchase_price_basis": CARRIED_ACQUISITION_BASIS, "thermal_basis": CARRIED_ACQUISITION_BASIS,
        "term_basis": "fixed" if fixed else TERM_REMAINING_LIFE,
        "term_years": float(term_setting) if fixed else float("nan"),
        "legacy_contracts": str((acquisition or {}).get("legacy_contracts", "honoured")),
        "in_construction": IN_CONSTRUCTION_PRIVATE, "generation_control_premium": CONTROL_PREMIUM_NONE,
        "generation_control_premium_share": 0.0, "asset_lives": str((acquisition or {}).get("asset_lives", LIVES_CENTRAL)),
        "thermal_and_storage_book_value_gbp": 0.0, "thermal_and_storage_owners_pv_gbp": 0.0,
        "cancelled_contract_residual_pv_gbp": 0.0, "public_real_rate": float(public_rate),
        "annuity_factor": factor,
        "valued_purchase_price_gbp": 0.0, "thermal_and_storage_purchase_price_gbp": 0.0,
        "valued_pv_state_gbp": 0.0, "valued_pv_market_gbp": 0.0,
        "excluded_capacity_mw": float(merged.loc[~plan["acquirable"], "capacity_mw"].sum()),
        "excluded_carriers": sorted(merged.loc[~plan["acquirable"], "carrier"].astype(str).unique()),
        "annual_acquisition_charge_gbp": 0.0, "cancelled_contract_residual_annual_gbp": 0.0,
        "carried_from_acquisition_year": True,
    }
    return plan, summary


#: F19 valuation decision (user, 14 Oct 2026): the support lines sized on the private case's account
#: that the purchase price counts as owner cash (the life-extension and battery top-ups).
PURCHASE_TOP_UP_COLUMNS = ("life_extension_top_up_gbp", "battery_revenue_top_up_gbp")
#: The plan columns that set which rows a case acquires (``_state_owned_vector``), fixed before pricing.
PLAN_SCOPE_COLUMNS = ("legacy_scope_group", "acquirable")


def acquisition_scope_plan(merged: pd.DataFrame, acquisition: Mapping) -> pd.DataFrame:
    """A plan's scope before it is priced: each row's legacy scope group and whether it is acquirable.

    The same rule :func:`plan_acquisition` applies (the valuation's scope groups; excluded carriers
    never acquirable), so a priced plan replaces it without moving which rows a case owns
    (:func:`check_plan_scope`).  It carries no price, so nothing can read one from it.
    """
    plan, _ = carried_acquisition_plan(merged, acquisition, public_rate=0.0)
    return plan[list(PLAN_SCOPE_COLUMNS)].copy()


def check_plan_scope(scope: pd.DataFrame, plan: pd.DataFrame) -> None:
    """Refuse a priced plan whose scope differs from the scope the register's layout was built on."""
    for column in PLAN_SCOPE_COLUMNS:
        before = scope[column].astype(str).to_numpy()
        after = plan[column].astype(str).to_numpy()
        if len(before) != len(after) or not np.array_equal(before, after):
            raise ValueError(f"the priced acquisition plan moves {column} against its scope plan: the register's "
                             "layout was built on the scope")


def _acquisition_summary_for(summary: dict, owned_legacy: pd.Series, plan: pd.DataFrame) -> dict:
    """The acquisition summary restricted to the rows a case actually buys."""
    price = pd.to_numeric(plan["acquisition_price_gbp"], errors="coerce").fillna(0.0)
    valued = plan["legacy_scope_group"].astype(str).eq(GROUP_VALUED)
    out = dict(summary)
    out["valued_purchase_price_gbp"] = float(price[owned_legacy & valued].sum())
    out["thermal_and_storage_purchase_price_gbp"] = float(price[owned_legacy & ~valued].sum())
    out["annual_acquisition_charge_gbp"] = (
        _annual_charge(plan, owned_legacy.to_numpy()) if "acquisition_annuity_factor" in plan.columns
        else (out["valued_purchase_price_gbp"] + out["thermal_and_storage_purchase_price_gbp"]) * out["annuity_factor"])
    return out


GENERATOR_CHARGE_COLUMNS = (
    "annual_generator_tnuos_gbp", "annual_insurance_gbp", "tnuos_zone", "tnuos_class",
    "tnuos_transmission_connected", "tnuos_annual_load_factor", "tnuos_wider_gbp_per_kw",
    "tnuos_offshore_local_gbp_per_kw", "tnuos_offshore_local_basis",
)


def load_generator_cost_tables(config: Mapping | None, root: Path | str = ".") -> dict:
    """Read the TNUoS tariff tables ``public_ownership.generator_costs.tnuos`` names."""
    tnuos = dict((config or {}).get("tnuos") or {})
    if not tnuos.get("enabled", False):
        return {}
    root = Path(root)
    return {
        "wider": pd.read_csv(root / tnuos["wider_tariffs"]),
        "offshore_local": pd.read_csv(root / tnuos["offshore_local_tariffs"]),
    }


def _normalised_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower())


def offshore_local_tariffs(asset_ids: pd.Series, table: pd.DataFrame, unmatched: str = "median") -> tuple[pd.Series, pd.Series]:
    """Offshore local TNUoS (substation + circuit + ETUoS, GBP/kW, table money) per asset, and its basis.

    A farm the NESO table names takes its own tariff (``register_match`` on the
    normalised base asset id, ``equals`` or ``prefix``); every other offshore
    farm, which has not transferred to an OFTO yet so has no published tariff,
    takes the table's median (``unmatched: median``).  Declared approximation:
    new OFTO tariffs are set on transfer and follow the transferred assets' cost.
    """
    totals = (pd.to_numeric(table["substation_gbp_per_kw"], errors="coerce").fillna(0.0)
              + pd.to_numeric(table["circuit_gbp_per_kw"], errors="coerce").fillna(0.0)
              + pd.to_numeric(table["etuos_gbp_per_kw"], errors="coerce").fillna(0.0))
    if unmatched != "median":
        raise ValueError(f"generator_costs.tnuos.offshore_local_unmatched={unmatched!r}; supported: median")
    fallback = float(totals.median())
    # Both parts of a partial-CfD site ("<site>" and "<site>_non_CfD", F19) are
    # the one farm and pay its tariff.
    from scripts.renewables.support_register_labels import partial_cfd_base_name

    values, basis = [], []
    for asset_id in asset_ids.astype(str):
        name = _normalised_name(partial_cfd_base_name(asset_id.split("::", 1)[0]))
        hit = None
        for token, rule, total in zip(table["register_match"].astype(str), table["match_rule"].astype(str), totals):
            if (rule == "equals" and name == token) or (rule == "prefix" and name.startswith(token)):
                hit = total
                break
        values.append(fallback if hit is None else float(hit))
        basis.append("table_median_untransferred" if hit is None else "neso_table")
    return pd.Series(values, index=asset_ids.index), pd.Series(basis, index=asset_ids.index)


def generator_charges(merged: pd.DataFrame, config: Mapping | None, tables: Mapping | None) -> pd.DataFrame:
    """Item D4(a): generator TNUoS and insurance per register row, in the register's price year.

    **TNUoS** (``generator_costs.tnuos``; NESO Five-Year View of TNUoS Tariffs
    2026/27 to 2030/31, v3.0, Table 6 for 2030/31 and Table 13 for offshore
    local): each transmission-connected row pays the wider tariff of its
    charging zone for its generator class at its own annual load factor
    (dispatch over capacity x hours), plus, for offshore wind, the offshore
    local substation, circuit and ETUoS tariffs that recover the OFTO's
    revenue.  The tariffs are nominal money of their year and are carried to
    the register's price year by the configured CPIH factors.  The zone is the
    one whose reference point is nearest the row's bus (``bus_lat``/``bus_lon``
    from the settlement); a row without a location pays no wider tariff and is
    flagged.  A row is transmission-connected when its carrier always is
    (offshore wind, nuclear, CCGT, CCS) or its site capacity reaches the
    region's large-station threshold (SHETL 10 MW, SPTL 30 MW, NGET 100 MW):
    smaller embedded plant pays no generator TNUoS (the small Embedded Export
    Tariff it earns is left out).  Onshore local substation and circuit tariffs
    (at most 0.92 GBP/kW for the substation) are left out.

    DESNZ's offshore capex (2,670 GBP/kW) excludes the offshore transmission
    assets: the developer builds them and transfers them to an OFTO, and the
    generator pays for them through the offshore local tariffs, which DESNZ
    carries under "connection and use of system charges".  So charging the
    offshore local tariff here is not a double count of capex.

    **Insurance** (``generator_costs.insurance``): a sourced GBP/kW/yr where one
    exists (offshore wind, BVG Associates 11,220 GBP/MW/yr), otherwise a share
    of as-new capex a year (GHD for DESNZ 2025, 0.4%), on the DESNZ/BEIS
    generation-cost rows whose fixed O&M excludes it; technologies whose cost
    source does not separate it are listed as excluded.

    Both are costs of the plant whoever owns it, identical in every case.
    """
    out = pd.DataFrame(index=merged.index)
    out["annual_generator_tnuos_gbp"] = 0.0
    out["annual_insurance_gbp"] = 0.0
    out["tnuos_zone"] = None
    out["tnuos_class"] = None
    out["tnuos_transmission_connected"] = False
    out["tnuos_annual_load_factor"] = np.nan
    out["tnuos_wider_gbp_per_kw"] = 0.0
    out["tnuos_offshore_local_gbp_per_kw"] = 0.0
    out["tnuos_offshore_local_basis"] = None
    config = dict(config or {})
    capacity = pd.to_numeric(merged["capacity_mw"], errors="coerce").fillna(0.0)

    insurance = dict(config.get("insurance") or {})
    if insurance.get("enabled", False):
        by_tech = {canonical_technology(k): float(v) for k, v in (insurance.get("gbp_per_kw_year_by_technology") or {}).items()}
        excluded = {canonical_technology(t) for t in (insurance.get("excluded_technologies") or [])}
        share = insurance.get("share_of_capex_per_year")
        # finance.annual_insurance for every row at once (30 Sep 2026; it was a row loop).
        technology = (merged["finance_technology"] if "finance_technology" in merged.columns
                      else pd.Series(None, index=merged.index, dtype=object)).map(canonical_technology)
        costed = ~(merged["technology"].isna() if "technology" in merged.columns else pd.Series(True, index=merged.index))
        costed &= ~technology.isin(excluded)
        costed &= (merged["capex_power_gbp_per_kw"].notna() if "capex_power_gbp_per_kw" in merged.columns
                   else pd.Series(False, index=merged.index))
        values = np.zeros(len(merged))
        if costed.any():
            rows = merged.loc[costed]
            capex_power, _ = _effective_capex_power_vector(rows)
            installed = rows["capacity_mw"].to_numpy(dtype=float)
            capex_total = (installed * 1_000.0 * capex_power
                           + _or_zero(rows, "energy_capacity_mwh") * 1_000.0 * _or_zero(rows, "capex_energy_gbp_per_kwh"))
            rate = technology[costed].map(by_tech)
            by_rate = installed * 1_000.0 * rate.to_numpy(dtype=float)
            by_share = capex_total * float(share) if share is not None else np.zeros(len(rows))
            values[costed.to_numpy()] = np.where(technology[costed].isin(list(by_tech)).to_numpy(), by_rate, by_share)
        out["annual_insurance_gbp"] = values

    tnuos = dict(config.get("tnuos") or {})
    if tnuos.get("enabled", False):
        if not tables or "wider" not in tables:
            raise ValueError("generator_costs.tnuos is enabled but its tariff tables were not loaded")
        zones = tables["wider"].copy()
        zones["zone"] = zones["zone"].astype(int)
        zone_rows = zones.set_index("zone")
        class_by_carrier = {str(k): str(v) for k, v in (tnuos.get("class_by_carrier") or {}).items()}
        unknown = sorted(set(class_by_carrier.values()) - set(TNUOS_CLASSES))
        if unknown:
            raise ValueError(f"generator_costs.tnuos.class_by_carrier uses unknown classes {unknown}")
        always = {str(c) for c in (tnuos.get("always_transmission_carriers") or [])}
        thresholds = {str(k): float(v) for k, v in (tnuos.get("transmission_threshold_mw_by_region") or {}).items()}
        offshore = {str(c) for c in (tnuos.get("offshore_carriers") or [])}
        hours = float(tnuos.get("hours_per_year", 8760.0))
        wider_factor = float(tnuos["wider_tariffs_to_price_year"])
        local_factor = float(tnuos["offshore_local_to_price_year"])
        carrier = merged["carrier"].astype(str)
        lat = pd.to_numeric(merged["bus_lat"], errors="coerce") if "bus_lat" in merged.columns else pd.Series(np.nan, index=merged.index)
        lon = pd.to_numeric(merged["bus_lon"], errors="coerce") if "bus_lon" in merged.columns else pd.Series(np.nan, index=merged.index)
        zone = pd.Series(nearest_zone(lat.tolist(), lon.tolist(), zones), index=merged.index, dtype=object)
        base = merged["asset_id"].astype(str).str.split("::", n=1).str[0]
        site_mw = capacity.groupby(base).transform("sum")
        region = zone.map(lambda z: zone_rows.loc[int(z), "transmission_region"] if z is not None else None)
        threshold = region.map(lambda r: thresholds.get(str(r), np.inf) if r is not None else np.inf)
        connected = carrier.isin(always) | (site_mw >= threshold.astype(float))
        tnuos_class = carrier.map(class_by_carrier)
        dispatch = pd.to_numeric(merged["dispatch_mwh"], errors="coerce").fillna(0.0)
        alf = (dispatch / (capacity * hours)).where(capacity > 0).clip(lower=0.0, upper=1.0)
        # finance.wider_tnuos_tariff for every charged row at once (30 Sep 2026; it was a row loop).
        wider = np.zeros(len(merged))
        charged = (connected.to_numpy(dtype=bool) & zone.map(lambda z: z is not None).to_numpy(dtype=bool)
                   & tnuos_class.map(lambda c: isinstance(c, str)).to_numpy(dtype=bool) & np.isfinite(alf.to_numpy()))
        if charged.any():
            classes = tnuos_class[charged].to_numpy(dtype=object)
            unknown = sorted(set(classes) - set(TNUOS_CLASSES))
            if unknown:
                raise ValueError(f"unknown TNUoS generator class {unknown[0]!r}; supported: {TNUOS_CLASSES}")
            tariffs = zone_rows.loc[[int(z) for z in zone[charged]]]
            peak = tariffs["system_peak_gbp_per_kw"].to_numpy(dtype=float)
            shared = tariffs["shared_year_round_gbp_per_kw"].to_numpy(dtype=float)
            not_shared = tariffs["not_shared_year_round_gbp_per_kw"].to_numpy(dtype=float)
            adjustment = tariffs["adjustment_gbp_per_kw"].to_numpy(dtype=float)
            a = np.minimum(np.maximum(alf.to_numpy(dtype=float)[charged], 0.0), 1.0)
            tariff = np.where(
                classes == TNUOS_CONVENTIONAL_CARBON, peak + a * (shared + not_shared) + adjustment,
                np.where(classes == TNUOS_CONVENTIONAL_LOW_CARBON, peak + a * shared + not_shared + adjustment,
                         a * shared + not_shared + adjustment))
            wider[charged] = tariff * wider_factor
        local = pd.Series(0.0, index=merged.index)
        local_basis = pd.Series(None, index=merged.index, dtype=object)
        is_offshore = carrier.isin(offshore)
        if is_offshore.any():
            values, basis = offshore_local_tariffs(merged.loc[is_offshore, "asset_id"], tables["offshore_local"],
                                                   str(tnuos.get("offshore_local_unmatched", "median")))
            local[is_offshore] = values * local_factor
            local_basis[is_offshore] = basis
        out["tnuos_zone"] = zone
        out["tnuos_class"] = tnuos_class.where(connected, None)
        out["tnuos_transmission_connected"] = connected.to_numpy()
        out["tnuos_annual_load_factor"] = alf
        out["tnuos_wider_gbp_per_kw"] = np.where(connected, wider, 0.0)
        out["tnuos_offshore_local_gbp_per_kw"] = np.where(connected, local, 0.0)
        out["tnuos_offshore_local_basis"] = local_basis
        out["annual_generator_tnuos_gbp"] = (out["tnuos_wider_gbp_per_kw"] + out["tnuos_offshore_local_gbp_per_kw"]) \
            * capacity * 1_000.0
    return out


def _corporation_tax_forgone(asset: pd.Series, capex_power: float, rate_assumption: Mapping, settings: Mapping,
                             wedge: Mapping | None, *, owned: bool, vintage: str, case_wacc: float,
                             remaining: float) -> float:
    """Corporation tax the state does not pay on one row: the tax wedge inside its financing saving.

    Method (decision D2): corporation tax is carried in a private owner's
    capital charge as the wedge between its pre-tax WACC and the same WACC with
    the tax on equity returns removed, so the tax embedded in an annuitised
    charge is ``capex x [CRF(r) - CRF(r - wedge)]`` over the row's remaining
    life (:func:`finance.embedded_corporation_tax`; wedges from
    :func:`finance.corporation_tax_wedges`, tau = workbook Assumptions C67).
    ``capex`` is the row's as-new capital cost (the base the register
    annuitises; an acquired legacy row uses it too, not its purchase price).

    * A row the state does not own: zero.
    * An owned row financed by the state itself (the public rate, or acquired
      legacy plant at the public legacy rate): the whole tax
      embedded at the private rate, since a public body pays none.
    * An owned row on the CEPA decomposition (the P1 30% stake): the JV remains
      a taxable company, so only the change is forgone, the embedded tax at the
      private rate less that at the Scenario 2 rate with the Scenario 2 wedge.

    ``NaN`` when the snapshot carries no wedge for the technology.
    """
    if wedge is None:
        return float("nan")
    if not owned:
        return 0.0
    capex_total = (float(asset["capacity_mw"]) * 1_000.0 * float(capex_power)
                   + float(asset.get("energy_capacity_mwh", 0.0) or 0.0) * 1_000.0
                   * float(asset.get("capex_energy_gbp_per_kwh", 0.0) or 0.0))
    r_private = private_rate(rate_assumption, settings)
    embedded_private = embedded_corporation_tax(capex_total, r_private, float(wedge["private"]), remaining)
    if vintage == "legacy":
        return embedded_private
    kind = str(settings.get(LEVER_PUBLIC_RETURN, DEFAULT_SETTINGS[LEVER_PUBLIC_RETURN]))
    if kind == RETURN_CEPA_DECOMPOSITION:
        return embedded_private - embedded_corporation_tax(
            capex_total, float(case_wacc), float(wedge["public_coordination"]), remaining)
    return embedded_private


# ── The per-key account, vectorised (30 Sep 2026) ────────────────────────────
# Until 30 Sep 2026 every register key walked the asset table with iterrows()
# and built one dict per row (about 2.9 million at 210 cases), then let pandas
# type the list of dicts.  The functions below compute the same numbers as
# arrays: every elementwise formula keeps the scalar code's operation order, the
# non-linear helpers (the capital recovery factor, the rate rules, the ownership
# share) are called once per distinct input, and each column is typed by the
# same routine pandas.DataFrame(list_of_dicts) uses, so a key's rows are the
# rows the loop built, value for value and dtype for dtype.

#: The keys a row's dict took, in the order it took them (the register's column
#: order is the union of these over the row kinds, in the order the kinds occur).
_BASE_KEYS = ("state_owned", "state_equity_share", "ownership_share", "legacy_scope_group", "acquisition_price_gbp",
              "acquisition_basis", "acquisition_term_years_applied", "capex_calibration_factor",
              *GENERATOR_CHARGE_COLUMNS)
_INELIGIBLE_KEYS = ("wacc", "annual_capital_recovery_gbp", "annual_fixed_om_gbp", "annual_variable_om_gbp",
                    "annual_site_fixed_om_gbp", "annual_generator_tnuos_gbp", "annual_insurance_gbp",
                    "corporation_tax_forgone_gbp", "annual_state_overhead_gbp", "annual_tax_equivalent_gbp",
                    "annual_required_revenue_gbp", "public_surplus_before_recycling_gbp", "evidence_status")
_ACQUIRED_KEYS = ("pv_state_cancelled_streams_gbp", "pv_market_cancelled_streams_gbp", "hpc_completion_sunk_cost_gbp",
                  "hpc_completion_remaining_capex_gbp", "annual_cancelled_contract_residual_gbp",
                  "cancelled_residual_route")
_ACCOUNT_KEYS = ("annual_capital_recovery_gbp", "annual_fixed_om_gbp", "annual_variable_om_gbp",
                 "annual_required_revenue_gbp", "capex_overrun_share_applied", "state_om_premium_share_applied",
                 "annual_state_overhead_gbp", "annual_site_fixed_om_gbp", "annual_generator_tnuos_gbp",
                 "annual_insurance_gbp", "corporation_tax_forgone_gbp", "annual_tax_equivalent_gbp",
                 "socialised_risk_gbp")
_RESULT_KEYS = ("cost_status", "effective_capex_power_gbp_per_kw", "storage_duration_normalised", "wacc",
                "remaining_life_years", "market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp",
                "producer_rent_gbp", "public_surplus_before_recycling_gbp")
#: Row kinds: unresolved evidence or cost (I), costed (E), costed and acquired legacy plant (A).
KIND_INELIGIBLE, KIND_ELIGIBLE, KIND_ACQUIRED = "I", "E", "A"
UNRESOLVED_EVIDENCE = "unresolved_cost_or_finance_mapping"


def _kind_keys(asset_fields: Sequence[str], kind: str) -> list[str]:
    """The keys one row of this kind carried, in dict order (a repeated key keeps its first place)."""
    extra = {KIND_INELIGIBLE: _INELIGIBLE_KEYS, KIND_ELIGIBLE: _ACCOUNT_KEYS + _RESULT_KEYS,
             KIND_ACQUIRED: _ACQUIRED_KEYS + _ACCOUNT_KEYS + _RESULT_KEYS}[kind]
    return list(dict.fromkeys([*asset_fields, *_BASE_KEYS, *extra, "case_id"]))


def _union_in_order(sequences) -> list[str]:
    """Keys in order of first appearance across the sequences (pandas' column order for a list of dicts)."""
    out: dict[str, None] = {}
    for sequence in sequences:
        for key in sequence:
            out.setdefault(key, None)
    return list(out)


def _records_column(values: np.ndarray):
    """The column ``pandas.DataFrame(list_of_dicts)`` builds from these Python objects, typed its way."""
    from pandas.core.internals.construction import convert_object_array

    return convert_object_array([np.asarray(values, dtype=object)], dtype=None)[0]


def _or_zero(frame: pd.DataFrame, column: str) -> np.ndarray:
    """``float(row.get(column, 0.0) or 0.0)`` for every row: a missing column or a false value is 0.0, NaN stays NaN."""
    if column not in frame.columns:
        return np.zeros(len(frame))
    values = frame[column]
    if values.dtype == object:
        return np.array([float(v or 0.0) for v in values], dtype=float)
    array = values.to_numpy(dtype=float)
    return np.where(array == 0.0, 0.0, array)


def _effective_capex_power_vector(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_effective_capex_power` for every row at once: (capex per kW, whether it was duration-normalised)."""
    n = len(merged)
    capex_power = merged["capex_power_gbp_per_kw"].to_numpy(dtype=float)
    cost_duration = (pd.to_numeric(merged["duration_hours"], errors="coerce").to_numpy(dtype=float)
                     if "duration_hours" in merged.columns else np.full(n, np.nan))
    energy = _or_zero(merged, "energy_capacity_mwh")
    capacity = merged["capacity_mw"].to_numpy(dtype=float)
    positive = capacity > 0
    asset_duration = np.full(n, np.nan)
    np.divide(energy, capacity, out=asset_duration, where=positive)
    normalised = (merged["component_type"].eq("StorageUnit").to_numpy() & ~np.isnan(cost_duration)
                  & (np.nan_to_num(cost_duration) > 0) & positive)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = capex_power * (asset_duration / cost_duration)
    return np.where(normalised, scaled, capex_power), normalised


def _remaining_life_vector(merged: pd.DataFrame) -> np.ndarray:
    """:func:`_remaining_life` for every row (``max(1, ...)`` ignores NaN, as the scalar ``max`` did)."""
    life = merged["economic_life_years"].to_numpy(dtype=float)
    if "commissioning_year" not in merged.columns:
        return life
    commissioning = pd.to_numeric(merged["commissioning_year"], errors="coerce").to_numpy(dtype=float)
    model_year = merged["model_year"].to_numpy(dtype=float)
    return np.where(np.isnan(commissioning), life, np.fmax(1.0, life - (model_year - commissioning)))


def _per_distinct(keys: list, function) -> np.ndarray:
    """``function(*key)`` for every key, evaluated once per distinct key."""
    cache = {key: function(*key) for key in dict.fromkeys(keys)}
    return np.array([cache[key] for key in keys], dtype=float)


class _RowInputs:
    """The key-independent inputs of the account, one array per quantity, computed once per run.

    Each is what the per-row loop read off ``asset`` (``float(asset[...])``,
    ``asset.get(..., 0.0) or 0.0`` and so on), for every row of the merged
    register.
    """

    def __init__(self, merged: pd.DataFrame, charges: pd.DataFrame, asset_fields: Sequence[str],
                 has_asset_revenue: bool, excluded_carriers, undelivered: set):
        n = len(merged)
        self.n = n
        self.asset_fields = list(asset_fields)
        self.has_asset_revenue = has_asset_revenue
        self.asset_values = {k: merged[k].to_numpy(dtype=object) for k in self.asset_fields}
        self.technology = merged["_technology_key"].astype(str).to_numpy()
        self.resolved = merged["evidence_status"].astype(str).eq("resolved").to_numpy()
        self.has_cost = (merged["technology"].notna().to_numpy() if "technology" in merged.columns
                         else np.zeros(n, dtype=bool))
        self.base_vintage = merged["finance_vintage"].astype(str).to_numpy()
        self._vintage_override = None
        component = merged["component_type"].astype(str)
        self.component = component.to_numpy()
        self.is_storage = component.eq("StorageUnit").to_numpy()
        self.is_generator = component.eq("Generator").to_numpy()
        excluded = set(excluded_carriers if excluded_carriers is not None else DEFAULT_EXCLUDED_CARRIERS)
        self.excluded = merged["carrier"].astype(str).isin(excluded).to_numpy()
        blank = pd.Series(None, index=merged.index, dtype=object)
        support = merged["support_type"] if "support_type" in merged.columns else blank
        contract = merged["contract_vintage"] if "contract_vintage" in merged.columns else blank
        is_cfd = support.map(lambda v: str(v or "").strip().lower() == "cfd").to_numpy(dtype=bool)
        post_policy = contract.map(lambda v: str(v or "") == "post_policy").to_numpy(dtype=bool)
        # F27 item 1 (24 Oct 2026): Net Zero Teesside's pre-2025 DPA is a pre-2025 contract too, so the
        # mask covers it (``pre_2025_dpa`` alone marks it, for the acquisition rule and its contract start).
        rounds = merged["cfd_round"] if "cfd_round" in merged.columns else blank
        years = merged["model_year"] if "model_year" in merged.columns else blank
        candidate = rounds.map(lambda r: str(r or "") in PRE_2025_DPA_CONTRACTS).to_numpy(dtype=bool)
        self.pre_2025_dpa = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(candidate):
            self.pre_2025_dpa[i] = is_pre_2025_dpa(support.iloc[i], rounds.iloc[i], years.iloc[i])
        self.pre_2025_dpa_start = np.array([float(PRE_2025_DPA_CONTRACTS[str(r)]) if dpa else np.nan
                                            for r, dpa in zip(rounds, self.pre_2025_dpa)], dtype=float)
        self.pre_policy_cfd = (is_cfd & ~post_policy) | self.pre_2025_dpa
        # 13 Nov 2026 (user, results review): Hinkley Point C's cohorts in a later year's register (in-construction
        # on its own round), which a case that bought it in 2030 reads as legacy (:meth:`use_vintage`).
        in_construction = self.base_vintage == "in_construction"
        self.in_construction_carried_round = in_construction & rounds.map(
            lambda r: str(r or "") in CARRIED_IN_CONSTRUCTION_ROUNDS).to_numpy(dtype=bool)
        self.in_construction_hpc = in_construction & rounds.map(
            lambda r: str(r or "") == HPC_CONTRACT_ROUND).to_numpy(dtype=bool)
        # The rows the current key carries (set by :meth:`use_vintage`; none until then).
        self.acquired_in_construction = np.zeros(n, dtype=bool)
        # User decision (15 Nov 2026, option a): projects not operating at the purchase
        # (acquisition.valuation.not_operating_at_acquisition: CP30's four pumped-storage projects without an FID) are
        # completed by the state in a case that buys the fleet, as new build at cost; set by the caller.
        self.completed_by_state = np.zeros(n, dtype=bool)
        # The rows the current key owns as its own new capacity (set by :meth:`use_vintage`).
        self.state_new_capacity = np.zeros(n, dtype=bool)
        # D1 (user, 14 Nov 2026): in a later model year, the rows the acquisition year's purchase bought
        # (:meth:`set_acquisition_baseline`); None in the acquisition year itself or without a baseline.
        self.baseline_acquired = None
        self.baseline_legacy_not_bought = None
        # F28 (a): contracted plant committed before the legislation year (legacy in the register). The
        # pre_2025_cfd_capacity switch's state ownership takes it back as new build (:meth:`use_vintage`).
        status = (merged["legislation_status"].astype(str).to_numpy() if "legislation_status" in merged.columns
                  else np.full(n, "", dtype=object))
        self.pre_legislation_contract = (status == "pre_legislation") & self.pre_policy_cfd
        self.undelivered =merged.index.isin(list(undelivered)) if undelivered else np.zeros(n, dtype=bool)
        self.capacity = merged["capacity_mw"].to_numpy(dtype=float)
        self.dispatch = merged["dispatch_mwh"].to_numpy(dtype=float)
        self.energy = _or_zero(merged, "energy_capacity_mwh")
        self.capex_energy = _or_zero(merged, "capex_energy_gbp_per_kwh")
        self.capex_power, self.normalised = _effective_capex_power_vector(merged)
        self.remaining = _remaining_life_vector(merged)
        self.fixed_om_rate = merged["fixed_om_gbp_per_kw_year"].to_numpy(dtype=float)
        self.variable_om_rate = merged["variable_om_gbp_per_mwh"].to_numpy(dtype=float)
        self.capex_calibration = (merged["capex_calibration_factor"].to_numpy(dtype=object)
                                  if "capex_calibration_factor" in merged.columns else np.full(n, 1.0, dtype=object))
        # Follow-up F6: the auction-margin slope of calibrated rows (0 elsewhere) and capex-based insurance.
        self.margin_slope = (pd.to_numeric(merged["capex_calibration_margin_slope"], errors="coerce").fillna(0.0)
                             .to_numpy(dtype=float) if "capex_calibration_margin_slope" in merged.columns else np.zeros(n))
        self.margin_insurance = (merged["capex_calibration_insurance_scales"].fillna(False).astype(bool).to_numpy()
                                 if "capex_calibration_insurance_scales" in merged.columns else np.zeros(n, dtype=bool))
        self.cost_status = (merged["status"].to_numpy(dtype=object) if "status" in merged.columns
                            else np.full(n, None, dtype=object))
        self.charges = {c: charges[c].to_numpy(dtype=object) for c in GENERATOR_CHARGE_COLUMNS}
        self.tnuos = charges["annual_generator_tnuos_gbp"].to_numpy(dtype=float)
        self.insurance = charges["annual_insurance_gbp"].to_numpy(dtype=float)
        if has_asset_revenue:
            self.market_revenue = _or_zero(merged, "market_revenue_gbp")
            self.support_revenue = _or_zero(merged, "support_revenue_gbp")
            self.fuel_carbon = _or_zero(merged, "fuel_carbon_gbp")
        self.capacity_market = (pd.to_numeric(merged["capacity_market_revenue_gbp"], errors="coerce").fillna(0.0)
                                .to_numpy(dtype=float) if "capacity_market_revenue_gbp" in merged.columns
                                else np.zeros(n))
        self.capex_total = self.capacity * 1_000.0 * self.capex_power + self.energy * 1_000.0 * self.capex_energy
        # FES 2050 plan (Y9, Y10): the build year and the technology-vintage cell, and the
        # asset ids, read only by the gated levers (a 2030 case never touches them).
        self.build_year = build_year_vector(merged)
        # F19 decision 3: the commissioning year a legacy row's tax written-down value runs from.
        self.commissioning_year = (pd.to_numeric(merged["commissioning_year"], errors="coerce").to_numpy(dtype=float)
                                   if "commissioning_year" in merged.columns else np.full(n, np.nan))
        self.asset_ids = merged["asset_id"].astype(str).to_numpy()
        carrier = merged["carrier"].astype(str).to_numpy() if "carrier" in merged.columns else np.full(n, "")
        self.carrier = carrier
        self._cell_technology = np.where(self.technology != "nan", self.technology, carrier)
        self._cells = None
        self._order = None
        self._ramp = {}

    @property
    def vintage(self) -> np.ndarray:
        """Each row's finance vintage under the register key in hand (:meth:`use_vintage`), else the register's."""
        return self.base_vintage if self._vintage_override is None else self._vintage_override

    def set_acquisition_baseline(self, merged: pd.DataFrame, bought: Mapping[str, float], *,
                                 acquisition_year: int = 2030,
                                 round_caps: Mapping[str, Mapping[str, float]] | None = None) -> None:
        """D1 (user, 14 Nov 2026): mark the rows a buying case owns in this later model year.

        By technology group, oldest first, up to the MW the acquisition year's purchase bought
        (:func:`acquisition_baseline.acquired_rows`): legacy rows, then in-construction rows on a round the purchase
        bought, then other in-construction rows (plant the purchase counted as running that this fleet builds
        later). Legacy rows beyond the purchase stay private in every case.
        """
        from scripts.public_ownership.acquisition_baseline import acquired_rows

        excluded = self.excluded.copy()
        if "ownership_exclusion" in merged.columns:
            excluded |= merged["ownership_exclusion"].fillna("").astype(str).str.strip().ne("").to_numpy()
        frame = pd.DataFrame({
            "carrier": merged["carrier"].astype(str).to_numpy(), "finance_vintage": self.base_vintage,
            "build_year": (pd.to_numeric(merged["build_year"], errors="coerce").to_numpy()
                           if "build_year" in merged.columns else np.nan),
            "capacity_mw": pd.to_numeric(merged["capacity_mw"], errors="coerce").fillna(0.0).to_numpy(),
            "cfd_round": (merged["cfd_round"].fillna("").astype(str).to_numpy()
                          if "cfd_round" in merged.columns else ""),
            "excluded": excluded})
        self.baseline_acquired = acquired_rows(frame, bought, round_caps=round_caps,
                                               carried_rounds=CARRIED_IN_CONSTRUCTION_ROUNDS,
                                               acquisition_year=acquisition_year)
        self.baseline_legacy_not_bought = (self.base_vintage == "legacy") & ~self.baseline_acquired & ~excluded
        # User decision (15 Nov 2026): plant built after the purchase is new capacity, which S3 owns at cost
        # (use_vintage); a row without a build year is refused by acquired_rows above when it could be bought.
        built = pd.to_numeric(frame["build_year"], errors="coerce").to_numpy(dtype=float)
        self.built_after_acquisition = (np.isfinite(built) & (built > float(acquisition_year))
                                        & np.isin(self.base_vintage, ("legacy", "in_construction")) & ~excluded)

    def use_vintage(self, settings: Mapping[str, str], plan) -> None:
        """Set the vintage the next key reads (F28 (a), user decision 26 Oct 2026).

        Plant contracted before the legislation year is legacy in the register: private under P1 and
        P2, bought by P3. Under ``pre_2025_cfd_capacity`` state ownership without a purchase (the
        switch on P1 or P2) the state owns and re-prices it as new build, as it did the pre-2025
        contracts commissioned after 2025 before F28; a case that buys the fleet still buys it.
        """
        flip = (plan is None and self.pre_legislation_contract.any()
                and settings.get(LEVER_PRE_2025_CFD) in PRE_2025_STATE_OWNED
                and settings.get(LEVER_NEW_BUILD_GENERATION, NEW_BUILD_PRIVATE) != NEW_BUILD_PRIVATE)
        self._vintage_override = (np.where(self.pre_legislation_contract, "new_build", self.base_vintage)
                                  if flip else None)
        # User decision (option E, 28 Oct 2026): GB Energy, dated from the GB Energy Act 2025, owns the rows of its
        # selection committed before the group's legislation year too; the state finances them as new build
        # (the key writes this view as the row's finance_vintage), and no case buys them.
        gbe_rows = self.gbe_owned_rows(settings, plan)
        if gbe_rows is not None:
            base = self._vintage_override if self._vintage_override is not None else self.base_vintage
            self._vintage_override = np.where(gbe_rows, "new_build", base)
        # 13 Nov 2026 (user, results review, finding 1): a case that bought Hinkley Point C in 2030 still owns it in
        # a later year, as acquired legacy plant (operating cost only; its capital is in the carried annuity), not as
        # private in-construction plant on its CfD.
        if plan is not None and getattr(self, "baseline_acquired", None) is not None:
            # D1 (user, 14 Nov 2026): the purchase's own capacity, by technology group (HPC follows its lever).
            self.acquired_in_construction = (
                (self.base_vintage == "in_construction") & self.baseline_acquired
                & carried_in_construction_mask(np.ones(len(self.base_vintage), dtype=bool), self.in_construction_hpc,
                                               settings))
        else:
            self.acquired_in_construction = (carried_in_construction_mask(
                self.in_construction_carried_round, self.in_construction_hpc, settings) if plan is not None
                else np.zeros(len(self.base_vintage), dtype=bool))
        if self.acquired_in_construction.any():
            base = self._vintage_override if self._vintage_override is not None else self.base_vintage
            self._vintage_override = np.where(self.acquired_in_construction, "legacy", base)
        # User decision (15 Nov 2026): S3 is full ownership; "if it's new capacity added then S3 should have it". A
        # case that buys the fleet owns every plant not yet operating at the purchase as its own new capacity, at
        # cost: the projects the purchase cannot buy (acquisition.valuation.not_operating_at_acquisition, CP30's
        # four pumped-storage projects in 2030) and, in a later year, everything built after the purchase that the
        # purchase did not carry (FES pumped storage, post-2030 nuclear and BECCS, Sizewell C, uncontracted plant
        # FES adds). Only plant running at the purchase that it did not buy (the fleet surplus) stays private. Hinkley Point C keeps its own rule (bought in 2030, carried);
        # under in_construction: public_completion the lever's own path owns in-construction plant.
        n = len(self.base_vintage)
        new_capacity = np.zeros(n, dtype=bool)
        if plan is not None:
            new_capacity = np.asarray(getattr(self, "completed_by_state", np.zeros(n, dtype=bool)), dtype=bool).copy()
            if str(settings.get(LEVER_IN_CONSTRUCTION, IN_CONSTRUCTION_PRIVATE)) != IN_CONSTRUCTION_PUBLIC:
                # In a later year, everything built after the purchase that the purchase did not carry; without a
                # carried baseline (an acquisition year's own register), the plant still in construction.
                later = getattr(self, "built_after_acquisition", None)
                bought = getattr(self, "baseline_acquired", None)
                later = (self.base_vintage == "in_construction") if later is None else later
                new_capacity |= (later & ~self.acquired_in_construction & ~self.in_construction_hpc
                                 & (~bought if bought is not None else True))
        self.state_new_capacity = new_capacity
        if new_capacity.any():
            base = self._vintage_override if self._vintage_override is not None else self.base_vintage
            self._vintage_override = np.where(new_capacity, "new_build", base)

    def gbe_owned_rows(self, settings: Mapping[str, str], plan) -> np.ndarray | None:
        """GB Energy's selection (offshore stake and local part) under a GBE case dated at its own year; else None."""
        if (plan is not None or getattr(self, "gbe_eligible_share", None) is None
                or settings.get(LEVER_NEW_BUILD_GENERATION) not in GBE_VALUES
                or getattr(self, "gbe_offshore", None) is None):
            return None
        return np.asarray(self.gbe_offshore, dtype=bool) | np.asarray(self.gbe_local, dtype=bool)

    def cells(self) -> list:
        """(technology, build year) per row, ``undated`` where the row has no build year (lever L4's cells)."""
        if self._cells is None:
            self._cells = [(t, int(y) if np.isfinite(y) else "undated")
                           for t, y in zip(self._cell_technology, self.build_year)]
        return self._cells

    def order_keys(self) -> list:
        """A fixed order within a cell with no location or name bias: the md5 of the asset id."""
        if self._order is None:
            import hashlib

            self._order = [hashlib.md5(a.encode("utf-8")).hexdigest() for a in self.asset_ids]
        return self._order

    def ramp_delta(self, settings: Mapping[str, str], deltas) -> np.ndarray:
        """The private-rate change the ``private_rate_path`` lever gives each row (decision 19)."""
        key = str(settings.get(LEVER_PRIVATE_RATE_PATH, lever_default(LEVER_PRIVATE_RATE_PATH)))
        if key not in self._ramp:
            self._ramp[key] = _per_distinct(
                list(zip(self.technology.tolist(), self.build_year.tolist())),
                lambda t, y: private_rate_path_delta(settings, t, y, deltas))
        return self._ramp[key]


class _KeyTerms:
    """What one register key's lever settings fix for the account (the per-key preamble of the old loop)."""

    def __init__(self, settings: Mapping[str, str], *, plan, plan_summary, public_rate, rates, wedges, share,
                 overhead_rate: float, overrun: float, om_premium: float, ramp_deltas=None,
                 auction_margin: float = 0.0, self_insurance: float = 0.0,
                 overhead_carrier_rates: Mapping[str, float] | None = None):
        self.settings = settings
        self.plan, self.plan_summary, self.public_rate = plan, plan_summary, public_rate
        self.rates, self.wedges, self.share = rates, wedges, share
        self.overhead_rate, self.overrun, self.om_premium = overhead_rate, overrun, om_premium
        # F19 decision 6: carriers whose overhead rate differs from the fleet rate (unstaffed storage).
        self.overhead_carrier_rates = dict(overhead_carrier_rates or {})
        self.tax_equivalent = settings.get(LEVER_TAX_EQUIVALENT) == TAX_EQUIVALENT_ON
        self.delivered_half = settings.get(LEVER_DELIVERED_SHARE) == DELIVERED_HALF
        # FES 2050 plan: lever L4 (state_deployment_share) and decision 19 (private_rate_path).
        self.deployment = deployment_share(settings)
        self.rate_ramp = (str(settings.get(LEVER_PRIVATE_RATE_PATH, lever_default(LEVER_PRIVATE_RATE_PATH)))
                          == RATE_PATH_NESO_WIND_RAMP)
        self.ramp_deltas = ramp_deltas
        # Follow-up F6: the share of contract revenue the calibrated cohorts keep above full cost.
        self.auction_margin = float(auction_margin)
        # Follow-up F7: the share of its insurance premium a wholly state-owned row saves.
        self.self_insurance = float(self_insurance)
        # F19 decision 3: the comparator's rent and capital charge per row (set before the key is
        # accounted, :func:`_comparator_rows`) and the corporation-tax basis; None keeps the old basis.
        self.comparator: dict[str, np.ndarray] | None = None
        self.tax_basis: Mapping | None = None
        # F10: GB Energy as announced (lever_parameters.gbe_announced; set by _terms_for).
        self.gbe: Mapping | None = None


def _state_owned_vector(rows: _RowInputs, settings: Mapping[str, str], plan: pd.DataFrame | None) -> np.ndarray:
    """:func:`_state_owned` for every row under one lever point."""
    legacy_rows = rows.vintage == "legacy"
    acquirable = plan["acquirable"].astype(bool).to_numpy() if plan is not None else np.ones(rows.n, dtype=bool)
    legacy = settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE)
    if legacy == LEGACY_PRIVATE or plan is None:
        legacy_owned = np.zeros(rows.n, dtype=bool)
    elif legacy == LEGACY_ACQUIRE_VALUED:
        # Hinkley Point C is in the valued group of the 2030 purchase (13 Nov 2026: its later-year cohorts too).
        legacy_owned = acquirable & (plan["legacy_scope_group"].astype(str).eq(GROUP_VALUED).to_numpy()
                                     | rows.acquired_in_construction)
    else:
        legacy_owned = acquirable & (legacy == LEGACY_ACQUIRE_ALL)
    if plan is not None and getattr(rows, "baseline_legacy_not_bought", None) is not None:
        # D1 (user, 14 Nov 2026): legacy plant beyond what the acquisition year's purchase bought is not the state's.
        legacy_owned = legacy_owned & ~rows.baseline_legacy_not_bought
    if settings.get(LEVER_NEW_BUILD_GENERATION, NEW_BUILD_PRIVATE) == NEW_BUILD_PRIVATE:
        new_build = np.zeros(rows.n, dtype=bool)
    else:
        new_build = np.ones(rows.n, dtype=bool)
        if settings.get(LEVER_NEW_BUILD_STORAGE) == STORAGE_PRIVATE:
            new_build &= ~rows.is_storage
        if settings.get(LEVER_PRE_2025_CFD) not in PRE_2025_STATE_OWNED:
            gbe_rows = rows.gbe_owned_rows(settings, plan) if hasattr(rows, "gbe_owned_rows") else None
            # Option E: a contract GB Energy's year makes its own is the state's (its CfD stays as signed).
            new_build &= ~(rows.pre_policy_cfd & ~gbe_rows) if gbe_rows is not None else ~rows.pre_policy_cfd
        elif legacy != LEGACY_PRIVATE:
            # F27 item 1: an acquiring case keeps Net Zero Teesside on its DPA (the kept remaining_ccs line).
            new_build &= ~getattr(rows, "pre_2025_dpa", np.zeros(rows.n, dtype=bool))
    other = np.where(rows.vintage == "in_construction",
                     settings.get(LEVER_IN_CONSTRUCTION) == IN_CONSTRUCTION_PUBLIC, new_build)
    other = other & acquirable & rows.resolved
    owned = ~rows.excluded & np.where(legacy_rows, legacy_owned, other)
    if plan is not None:
        # User decision (15 Nov 2026): S3 is full ownership, so the plant not yet operating at the purchase is the
        # case's own new capacity whatever the new-build levers say (use_vintage sets the rows).
        owned = owned | (getattr(rows, "state_new_capacity", np.zeros(rows.n, dtype=bool)) & ~rows.excluded
                         & rows.resolved)
    return owned


def _key_kinds(rows: _RowInputs, terms: _KeyTerms) -> dict[str, np.ndarray]:
    """Ownership, eligibility and the row kind of every row under one key (it sets the key's vintage view)."""
    settings = terms.settings
    rows.use_vintage(settings, terms.plan)
    rate_ok = np.array([bool(terms.rates.get(t)) for t in rows.technology], dtype=bool) if rows.n else np.zeros(0, bool)
    eligible = rows.resolved & rows.has_cost & rate_ok
    owned = _state_owned_vector(rows, settings, terms.plan)
    if terms.plan is not None and rows.acquired_in_construction.any():
        # 13 Nov 2026 (user, results review, finding 1), enforced: plant the 2030 purchase bought is never left
        # private in a later year while its annuity is still charged (it was: HPC paid through its CfD and again
        # through the annuity in 2040 and 2050).
        stranded = rows.acquired_in_construction & ~owned & ~rows.excluded
        if stranded.any():
            raise ValueError(f"{int(stranded.sum())} in-construction cohorts bought in 2030 (Hinkley Point C cohorts "
                             f"and other carried rounds, {rows.capacity[stranded].sum():,.0f} MW) are not state-owned "
                             f"under a case that still pays for them: {list(rows.asset_ids[stranded][:3])}")
    if terms.delivered_half:
        # The state delivers only half of the cohort; the rest is built privately.
        owned = owned & ~((rows.vintage == "new_build") & rows.undelivered)
    fraction = np.ones(rows.n)
    if terms.deployment is not None:
        # Lever L4 (FES 2050 decisions 13 and 17), fractional since decision 59 (user, 3 Oct 2026):
        # every new-build row inside P2's boundary is held at exactly the share, the rest by a
        # private co-owner (whole-row selection made the 25/50/75% shares lumpy).
        new_owned = owned & (rows.vintage == "new_build")
        fraction = np.where(new_owned, float(terms.deployment), 1.0)
        owned = owned & ~(new_owned & (fraction <= 0.0))
    gbe_stake = None
    if terms.gbe is not None:
        # F10: GB Energy as announced owns only its selection inside P2's boundary: the offshore
        # stake rows (P1's stake mechanics, the share set in _key_account) and the local solar and
        # onshore rows (P2's), the latter at the partnership share in that sensitivity (decision 59's
        # co-ownership); everything else, storage included, stays private.
        offshore, local = rows.gbe_offshore, rows.gbe_local
        owned = owned & (offshore | local) & (rows.vintage == "new_build")
        gbe_stake = owned & offshore
        local_share = float(terms.gbe["local_public_share"])
        gbe_share = getattr(rows, "gbe_eligible_share", None)
        if gbe_share is None:
            fraction = np.where(owned & local, local_share, fraction)
        else:
            # Option E: the state holds a row's part committed on or after GBE's year (decision 59's co-ownership
            # for the rest); the stake's 30% is of that part.
            gbe_share = np.asarray(gbe_share, dtype=float)
            fraction = np.where(owned & offshore, gbe_share, fraction)
            fraction = np.where(owned & local, gbe_share * local_share, fraction)
    rab = getattr(rows, "rab_outside_propositions", None)
    if (rab is not None and terms.plan is None
            and settings.get(LEVER_NEW_BUILD_GENERATION, NEW_BUILD_PRIVATE) in (NEW_BUILD_PUBLIC, NEW_BUILD_STATE_STAKE)):
        # FES 2050 decision 32 (user, 3 Oct 2026; switch ``vintage_capital_2050.rab_outside_propositions``):
        # Sizewell C's RAB is a regulated asset outside P1's stake and P2's boundary; the state takes
        # CfD, merchant and DPA new build only. Applied after L4's selection, so no other row's
        # selection moves (the P2 bid months were solved on those selections). P3 still acquires it.
        owned = owned & ~rab
    held = getattr(rows, "held_private", None)
    if held is not None:
        # FES 2050 (user, 3 Oct 2026): a carrier newly costed after the P2 bid months were solved
        # (geothermal) stays private in every case until the full rebuild re-solves on its ownership.
        owned = owned & ~held
    acquired = owned & (rows.vintage == "legacy") & (terms.plan is not None)
    kind = np.where(~eligible, KIND_INELIGIBLE, np.where(acquired, KIND_ACQUIRED, KIND_ELIGIBLE))
    out = {"eligible": eligible, "owned": owned, "acquired": acquired, "kind": kind,
           "deployment_fraction": np.where(owned, fraction, 1.0)}
    if gbe_stake is not None:
        out["gbe_stake"] = gbe_stake
        out["gbe_part"] = np.where(gbe_stake, GBE_PART_STAKE, np.where(owned, GBE_PART_LOCAL, None)).astype(object)
    return out


def _kinds_in_order(kind: np.ndarray) -> list[str]:
    """The row kinds of one key in the order they first occur."""
    if not len(kind):
        return []
    values, first = np.unique(kind, return_index=True)
    return [str(values[i]) for i in np.argsort(first)]


#: F27 item 1: the term of the pre-2025 CfDs (FIDeR and AR1-AR6: 15 years from start).
PRE_2025_CFD_TERM_YEARS = 15.0


def _owners_value_premium(rows: _RowInputs, terms: _KeyTerms, idx: np.ndarray, own: np.ndarray, vintage: np.ndarray,
                          share_e: np.ndarray, wacc: np.ndarray, remaining: np.ndarray,
                          finance_snapshot: Mapping) -> np.ndarray:
    """F27 item 1 (user, 20 Oct 2026): the annual premium over cost of pre-2025 CfD plant bought at its owners' value.

    Under ``pre_2025_cfd_capacity: public_bought_owners_value`` the state pays the owners what the
    plant is worth to them: its cost plus the present value, at their own rate, of the rent they
    earn above their required return (the comparator's register rent, judged as the corporation-tax
    rule judges it) over the contract's remaining years (15-year terms; after its contract a
    merchant plant is taken to earn about its required return, declared). The premium is financed
    with the plant at the state's rate over the remaining life, on the state's share. It is a
    payment to the former owners, so the register carries it apart (``owners_value_premium_gbp``)
    for the ledger. Zero for every other lever value. Net Zero Teesside's pre-2025 DPA (15 years
    from 2028, ``levers.PRE_2025_DPA_CONTRACTS``) is bought the same way (24 Oct 2026).
    """
    out = np.zeros(len(idx))
    if terms.settings.get(LEVER_PRE_2025_CFD) != PRE_2025_PUBLIC_OWNERS_VALUE:
        return out
    if terms.comparator is None or terms.tax_basis is None:
        raise ValueError("pre_2025_cfd_capacity: public_bought_owners_value needs the comparator's rows "
                         "(public_ownership.corporation_tax_basis enabled, a non-vintaged register)")
    bought = own & rows.pre_policy_cfd[idx] & (vintage != "legacy")
    if not bought.any():
        return out
    technology = rows.technology[idx]
    r_private = _per_distinct(list(zip(technology.tolist(), vintage.tolist())),
                              lambda t, v: rate_for_settings(terms.rates.get(t), finance_snapshot, terms.settings,
                                                             finance_vintage=v, owned=False))
    model_year = float(terms.tax_basis["model_year"])
    start = rows.commissioning_year[idx]
    # F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA runs 15 years from its start (2028), whatever
    # the register's commissioning year says (the overlay row carries none).
    dpa_start = getattr(rows, "pre_2025_dpa_start", None)
    if dpa_start is not None:
        start = np.where(np.isfinite(dpa_start[idx]), dpa_start[idx], start)
    left = np.where(np.isfinite(start), start + PRE_2025_CFD_TERM_YEARS - model_year, PRE_2025_CFD_TERM_YEARS)
    years = np.clip(np.minimum(left, remaining), 0.0, None)
    annuity = np.where((years > 0) & (r_private > 0), (1.0 - (1.0 + r_private) ** -years) / np.where(r_private > 0, r_private, 1.0),
                       years)
    present_value = terms.comparator["rent"][idx] * annuity
    crf = _per_distinct(list(zip(wacc.tolist(), remaining.tolist())), capital_recovery_factor)
    return np.where(bought, share_e * present_value * crf, 0.0)


def owners_value_premium_state_adjustment(premium, share) -> np.ndarray:
    """``-(1 - s) x premium`` per row: the premium is the state's purchase of its own share, so its part of the row
    carries all of it, while every split of the row (rent, at-cost saving) is pro rata on ``s``. Zero at full
    ownership and wherever no premium is paid."""
    premium = np.asarray(premium, dtype=float)
    share = np.clip(np.asarray(share, dtype=float), 0.0, 1.0)
    return np.where(premium != 0.0, -(1.0 - share) * premium, 0.0)


def _plan_annuity(plan: pd.DataFrame, plan_summary: Mapping) -> np.ndarray:
    """Each row's acquisition annuity factor: the plan's own per row (the F28 hook), else the summary's."""
    if "acquisition_annuity_factor" in plan.columns:
        own = pd.to_numeric(plan["acquisition_annuity_factor"], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(own), own, float(plan_summary["annuity_factor"]))
    return np.full(len(plan), float(plan_summary["annuity_factor"]))


def _plan_terms(plan: pd.DataFrame, plan_summary: Mapping) -> np.ndarray:
    """Each row's acquisition term (years): the plan's own per row, else the summary's."""
    if "acquisition_term_years" in plan.columns:
        own = pd.to_numeric(plan["acquisition_term_years"], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(own), own, float(plan_summary["term_years"]))
    return np.full(len(plan), float(plan_summary["term_years"]))


def _risk_charge_ratio(rate: np.ndarray, premium: np.ndarray, years: np.ndarray) -> np.ndarray:
    """CRF(r + premium, n) / CRF(r, n) - 1 per row: the share a capital charge rises at the economic rate (F29)."""
    return _per_distinct(
        list(zip(np.asarray(rate, dtype=float).tolist(), np.asarray(premium, dtype=float).tolist(),
                 np.asarray(years, dtype=float).tolist())),
        lambda r, p, n: (capital_recovery_factor(r + p, max(float(n), 1.0))
                         / capital_recovery_factor(r, max(float(n), 1.0)) - 1.0) if p > 0.0 and n == n else 0.0)


def _key_account(rows: _RowInputs, terms: _KeyTerms, kinds: Mapping[str, np.ndarray], *,
                 finance_snapshot: Mapping, overhead_applies_to, self_insurance_column: bool = False,
                 owners_value_column: bool = False,
                 capital_terms: Mapping[str, np.ndarray] | None = None,
                 co_ownership: Mapping | None = None) -> dict[str, np.ndarray]:
    """Every column of one key's asset rows as Python objects, exactly as the per-row dicts held them."""
    settings, plan = terms.settings, terms.plan
    n = rows.n
    eligible, owned, acquired = kinds["eligible"], kinds["owned"], kinds["acquired"]
    ineligible = ~eligible
    values = {k: v.copy() for k, v in rows.asset_values.items()}

    def put(key: str, array, mask=None) -> None:
        if key not in values:
            values[key] = np.full(n, np.nan, dtype=object)       # a key a row never took reads NaN
        column = values[key]
        if mask is None:
            column[:] = array
        elif np.ndim(array) == 0:
            column[mask] = array
        else:
            column[mask] = np.asarray(array, dtype=object)[mask]

    if rows.vintage is not rows.base_vintage:
        # F28 (a): the key's own vintage view (the pre_2025_cfd_capacity switch owns pre-legislation contracts).
        put("finance_vintage", rows.vintage.astype(object))

    share = _per_distinct(
        list(zip(rows.vintage.tolist(), owned.tolist())),
        lambda vintage, is_owned: ownership_share_for_settings(
            settings, finance_vintage=vintage, owned=is_owned, configured_share=terms.share))
    fraction = kinds.get("deployment_fraction")
    stake_rows = kinds.get("gbe_stake")
    if stake_rows is not None:
        # F10: the offshore stake holds the configured share of the row, as P1's stake does.
        share = np.where(stake_rows, float(terms.gbe["offshore_stake_share"]), share)
        put("gbe_part", kinds["gbe_part"])
        if getattr(rows, "gbe_eligible_share", None) is not None:
            # Option E: each GBE row's share committed on or after GB Energy's year (the GB Energy Act 2025).
            owned_part = np.array([part is not None for part in kinds["gbe_part"]], dtype=bool)
            put("gbe_eligible_share", np.where(owned_part, rows.gbe_eligible_share, np.nan).astype(object))
    co_owned = None
    if fraction is not None and (fraction < 1.0).any():
        # Decision 59: lever L4's rows are held at exactly the share (the state's part of the row).
        share = share * fraction
        co_owned = owned & (fraction < 1.0)
    put("state_owned", owned.astype(object))
    put("state_equity_share", share.astype(object))
    put("ownership_share", share.astype(object))
    put("legacy_scope_group", plan["legacy_scope_group"].to_numpy(dtype=object) if plan is not None else None)
    put("acquisition_price_gbp", None)
    put("acquisition_basis", None)
    put("acquisition_term_years_applied", None)
    put("capex_calibration_factor", rows.capex_calibration)
    for column in GENERATOR_CHARGE_COLUMNS:
        put(column, rows.charges[column])

    # Rows without a costed technology and a finance mapping: nothing is accounted.
    for key in _INELIGIBLE_KEYS[:-1]:
        put(key, None, ineligible)
    put("evidence_status", UNRESOLVED_EVIDENCE, ineligible)
    if not eligible.any():
        return values

    idx = np.flatnonzero(eligible)
    technology, vintage = rows.technology[idx], rows.vintage[idx]
    own, share_e = owned[idx], share[idx]
    stake_e = stake_rows[idx] if stake_rows is not None else np.zeros(len(idx), dtype=bool)
    # F10: the offshore stake is financed as P1's is (a joint venture on the CEPA decomposition).
    stake_settings = ({**settings, LEVER_NEW_BUILD_GENERATION: NEW_BUILD_STATE_STAKE,
                       LEVER_PUBLIC_RETURN: RETURN_CEPA_DECOMPOSITION} if stake_rows is not None else settings)
    wacc = _per_distinct(
        list(zip(technology.tolist(), vintage.tolist(), own.tolist(), share_e.tolist(), stake_e.tolist())),
        lambda t, v, o, s, k: rate_for_settings(terms.rates.get(t), finance_snapshot, stake_settings if k else settings,
                                                finance_vintage=v, owned=o, ownership_share=s))
    if terms.rate_ramp:
        # Decision 19's sensitivity: NESO's wind step-down on every privately financed wind vintage after 2030.
        wacc = np.where(~own, wacc + rows.ramp_delta(settings, terms.ramp_deltas)[idx], wacc)
    offsets = (getattr(rows, "strike_offsets", None) or {}).get(RATE_PATH_NESO_WIND_RAMP) if terms.rate_ramp else None
    remaining = rows.remaining[idx]
    co = co_owned[idx] if co_owned is not None else np.zeros(len(idx), dtype=bool)
    frac_e = fraction[idx] if co_owned is not None else np.ones(len(idx))
    co_basis = str((co_ownership or {}).get("basis", CO_OWNERSHIP_POOLED))
    co_separate = co & (co_basis != CO_OWNERSHIP_POOLED)
    wacc_state = wacc.copy()
    co_owner_rate = np.full(len(idx), np.nan)
    if co.any():
        # Decision 59: a co-owned row's rate is the one whose annuity is the share-weighted annuity
        # of its owners (the state at its rate, the private co-owner at the private rate).
        private_part = _per_distinct(list(zip(technology.tolist(), vintage.tolist())),
                                     lambda t, v: rate_for_settings(terms.rates.get(t), finance_snapshot, settings,
                                                                    finance_vintage=v, owned=False))
        if terms.rate_ramp:
            private_part = private_part + rows.ramp_delta(settings, terms.ramp_deltas)[idx]
        years_basis = remaining
        if capital_terms is not None and "level_row" in capital_terms:
            # Decision 66: a merchant row's annuity runs over its economic life.
            years_basis = np.where(capital_terms["level_row"][idx], capital_terms["level_life_years"][idx], remaining)
        # Decision 68 (user, 3 Oct 2026): the co-owner's own private rate; under the reduction sensitivity, cut
        # by bp_per_point basis points per percentage point of state share (on the co-owner's share only).
        co_owner_rate_unreduced = np.where(co, private_part, np.nan)
        reduction = float((co_ownership or {}).get("private_rate_reduction_bp_per_point", 0.0)) * frac_e * 100.0 / 1e4
        private_part = private_part - np.where(co, reduction, 0.0)
        co_owner_rate = np.where(co, private_part, np.nan)
        blended = _per_distinct(list(zip(wacc.tolist(), private_part.tolist(), frac_e.tolist(), years_basis.tolist())),
                                lambda pub, priv, f, n: co_ownership_rate(pub, priv, f, max(float(n), 1.0)))
        wacc = np.where(co, blended, wacc)
    capacity, energy = rows.capacity[idx], rows.energy[idx]
    capex_power, capex_energy = rows.capex_power[idx], rows.capex_energy[idx]
    # Follow-up F6: the auction margin lowers the calibrated cohorts' capex in every case alike.
    margin_scale = (auction_margin_capex_scale(rows.margin_slope[idx], terms.auction_margin)
                    if terms.auction_margin and rows.margin_slope.any() else np.ones(len(idx)))
    capex_power, capex_energy = capex_power * margin_scale, capex_energy * margin_scale
    # Levers state_capex_overrun and state_om_premium: costs the state bears on the
    # plant it builds or runs, never on the private comparator's.
    new_build_owned = own & (vintage == "new_build")
    # Decision 59: a co-owned row bears the state's overrun and O&M premium on the state's part only.
    capex_scale = 1.0 + np.where(new_build_owned, terms.overrun * frac_e, 0.0)
    om_scale = 1.0 + np.where(own & rows.is_generator[idx], terms.om_premium * frac_e, 0.0)
    crf = _per_distinct(list(zip(wacc.tolist(), remaining.tolist())), capital_recovery_factor)
    capital = (capacity * 1_000.0 * (capex_power * capex_scale) + energy * 1_000.0 * (capex_energy * capex_scale)) * crf
    if capital_terms is not None:
        # FES 2050 decision 41 (vintage_capital_2050): a row within its contract pays the term
        # annuity on capex less the tail's value, for its live share of the year; a row whose
        # contract has ended has recovered its capital (sunk), so its capital charge is zero.
        capital = _term_basis_capital(capital_terms, idx, wacc, capacity, energy, capex_power * capex_scale,
                                      capex_energy * capex_scale, capital)
    if co_separate.any():
        # Decision 59, ``co_ownership_finance: separate`` (review 2, 3 Oct 2026): each owner finances its own
        # part, the state at its rate with its overrun, the co-owner at the private rate (less any
        # reduction); the row's capital is the share-weighted sum, not the whole row at a blended rate.
        overrun_full = 1.0 + np.where(new_build_owned, terms.overrun, 0.0)
        capital_state_view = _co_view_capital(capital_terms, idx, wacc_state, remaining, capacity, energy,
                                              capex_power * overrun_full, capex_energy * overrun_full)
        capital_private_view = _co_view_capital(capital_terms, idx, np.where(co, co_owner_rate, wacc_state), remaining,
                                                capacity, energy, capex_power, capex_energy)
        capital = np.where(co_separate, frac_e * capital_state_view + (1.0 - frac_e) * capital_private_view, capital)
        # Decision 68: the co-owner's gain from the reduction (a transfer the Exchequer's implicit guarantee backs).
        co_investment_gain = np.zeros(len(idx))
        if float((co_ownership or {}).get("private_rate_reduction_bp_per_point", 0.0)) > 0.0:
            capital_private_unreduced = _co_view_capital(
                capital_terms, idx, np.where(co, co_owner_rate_unreduced, wacc_state), remaining, capacity, energy,
                capex_power, capex_energy)
            co_investment_gain = np.where(co_separate, (1.0 - frac_e) * (capital_private_unreduced
                                                                         - capital_private_view), 0.0)
    fixed_om = capacity * 1_000.0 * (rows.fixed_om_rate[idx] * om_scale)
    variable_om = rows.dispatch[idx] * (rows.variable_om_rate[idx] * om_scale)
    required = capital + fixed_om + variable_om
    # Item OH1: the state's holding company, on owned generation and storage capacity only.
    applies = np.isin(rows.component[idx], list(overhead_applies_to)) if overhead_applies_to is not None \
        else np.ones(len(idx), dtype=bool)
    overhead_rate = np.full(len(idx), float(terms.overhead_rate))
    for carrier_name, carrier_rate in terms.overhead_carrier_rates.items():
        overhead_rate = np.where(rows.carrier[idx] == carrier_name, float(carrier_rate), overhead_rate)
    overhead = np.where(applies & (share_e > 0.0), capacity * 1_000.0 * share_e * overhead_rate, 0.0)
    required = required + overhead
    # Item D4(a): generator TNUoS and insurance, folded into fixed O&M with the parts beside it.
    site_fixed_om = fixed_om
    tnuos, insurance = rows.tnuos[idx], rows.insurance[idx]
    insurance = np.where(rows.margin_insurance[idx], insurance * margin_scale, insurance)
    # Follow-up F7: a row the state owns outright self-insures and saves the insurer's loading.
    self_insured = own & (share_e >= 1.0 - 1e-9) if terms.self_insurance else np.zeros(len(idx), dtype=bool)
    self_insurance_saving = np.where(self_insured, insurance * terms.self_insurance, 0.0)
    insurance = insurance - self_insurance_saving
    fixed_om = fixed_om + (tnuos + insurance)
    required = required + (tnuos + insurance)
    # F29 (user, 26 Oct 2026): the social cost of the systematic risk the state holds on the plant it
    # finances at the cash rate, workbook v2.0's economic rate less its cash rate (rho x beta x P), on
    # the state's part of the row's own capital charge.  Off the bill: a taxpayers' cost in the
    # Exchequer ledger and out of the resource saving.  A stake (P1, the GB Energy offshore stake) books
    # none: the workbook treats Sc2's legs as contingent.
    state_financed = own & ~stake_e & (str(settings.get(LEVER_PUBLIC_RETURN, DEFAULT_SETTINGS[LEVER_PUBLIC_RETURN]))
                                       == RETURN_PUBLIC_RATE)
    premium = _per_distinct(list(zip(technology.tolist(), vintage.tolist())),
                            lambda t, v: systematic_premium(terms.rates.get(t) or {}, settings,
                                                            operating=(v == "legacy")))
    years_n = remaining
    if capital_terms is not None and "level_row" in capital_terms:
        years_n = np.where(capital_terms["level_row"][idx], capital_terms["level_life_years"][idx], remaining)
    rate_state = np.where(co_separate, wacc_state, wacc) if co.any() else wacc
    state_capital = capital * np.where(co, frac_e, 1.0)
    if co_separate.any():
        state_capital = np.where(co_separate, frac_e * capital_state_view, state_capital)
    socialised = np.where(state_financed & (premium > 0.0),
                          state_capital * _risk_charge_ratio(rate_state, premium, years_n), 0.0)
    owners_value_premium = _owners_value_premium(rows, terms, idx, own, vintage, share_e, wacc, remaining,
                                                 finance_snapshot)
    capital = capital + owners_value_premium
    required = required + owners_value_premium
    # F27 item 1 (user, 27 Oct 2026): the premium is the state's purchase of its own share, so the state's
    # part of the row carries all of it; pro rata on the whole row, co-owners bore (1 - s) of it.
    premium_state_adjustment = owners_value_premium_state_adjustment(owners_value_premium, share_e)
    is_acquired = acquired[idx]
    if is_acquired.any():
        # Item N2: a legacy asset costs what it costs to buy, annuitised at the
        # public rate over the acquisition term, plus its O&M.  The term and its factor are
        # per row (plan_acquisition's hook; one term for every row today).
        annuity = _plan_annuity(plan, terms.plan_summary)[idx]
        price = plan["acquisition_price_gbp"].to_numpy(dtype=float)[idx]
        # Review finding 9 (13 Nov 2026): a carried in-construction row (bought in 2030) has no price in this
        # year's plan; its capital is the carried annuity, so it costs its O&M here (it was left unpriced, so the
        # tariff paid it as today, CfD included).
        carried = rows.acquired_in_construction[idx] & ~np.isfinite(price)
        bought = np.where(carried, 0.0, price * annuity)
        capital = np.where(is_acquired, bought, capital)
        required = np.where(is_acquired, bought + fixed_om + variable_om + overhead, required)
        unpriced = is_acquired & rows.acquired_in_construction[idx] & ~np.isfinite(required)
        if unpriced.any():
            # Enforced (13 Nov 2026): an unpriced carried row is paid as today in the tariff (its CfD on the bill)
            # while the carried annuity pays for it again.
            raise ValueError(f"{int(unpriced.sum())} carried in-construction rows have no operating cost: "
                             f"{list(rows.asset_ids[idx][unpriced][:3])}")
        wacc = np.where(is_acquired, float(terms.public_rate), wacc)
        term_rows = _plan_terms(plan, terms.plan_summary)[idx]
        socialised = np.where(is_acquired & (premium > 0.0), np.nan_to_num(bought) * _risk_charge_ratio(
            np.full(len(idx), float(terms.public_rate)), premium, term_rows), socialised)
    # Decision D2, the corporation tax a private owner of the same (unscaled) plant would pay.
    tax = _corporation_tax_forgone_vector(rows, terms, idx, own, vintage, np.where(co_separate, wacc_state, wacc),
                                          remaining, capital_terms, capex_scale=margin_scale,
                                          stake=stake_e if stake_rows is not None else None)
    if co.any():
        # Decision 59: only the state's part of a co-owned row stops paying corporation tax.
        tax = np.where(co, tax * frac_e, tax)
    tax_equivalent = (np.where(new_build_owned & np.isfinite(tax), tax, 0.0) if terms.tax_equivalent
                      else np.zeros(len(idx)))
    required = required + tax_equivalent

    def put_eligible(key: str, array) -> None:
        full = np.full(n, None, dtype=object)
        full[idx] = np.asarray(array, dtype=object) if np.ndim(array) else array
        put(key, full, eligible)

    if co_ownership is not None:
        # Decision 59 (review 2): what moves the state's surplus from its share of the row's rent to its share
        # of the revenue less its own cost (zero on every row that is not co-owned under separate financing).
        co_adjustment = np.zeros(len(idx))
        co_capital_adjustment = np.zeros(len(idx))
        if co_separate.any():
            om_base = capacity * 1_000.0 * rows.fixed_om_rate[idx] + rows.dispatch[idx] * rows.variable_om_rate[idx]
            premium = np.where(own & rows.is_generator[idx], terms.om_premium, 0.0)
            gap = (capital_private_view - capital_state_view) - premium * om_base
            co_adjustment = np.where(co_separate, frac_e * (1.0 - frac_e) * gap - (1.0 - frac_e) * tax_equivalent, 0.0)
            co_capital_adjustment = np.where(
                co_separate, frac_e * (1.0 - frac_e) * (capital_private_view - capital_state_view), 0.0)
        put_eligible("co_ownership_basis", np.where(co, co_basis, None))
        put_eligible("co_owner_rate", co_owner_rate)
        put_eligible("state_part_rate", np.where(co, wacc_state, np.nan))
        if co_separate.any():
            put_eligible("capex_state_view_gbp", np.where(
                co_separate, capacity * 1_000.0 * capex_power * overrun_full + energy * 1_000.0 * capex_energy
                * overrun_full, np.nan))
            put_eligible("capex_private_view_gbp", np.where(
                co_separate, capacity * 1_000.0 * capex_power + energy * 1_000.0 * capex_energy, np.nan))
        put_eligible("co_ownership_state_adjustment_gbp", co_adjustment)
        put_eligible("co_ownership_state_capital_adjustment_gbp", co_capital_adjustment)
        put_eligible("co_investment_rate_gain_gbp", co_investment_gain if co_separate.any() else np.zeros(len(idx)))
    if capital_terms is not None:
        # Decision 66 (review 2): the years each row's capital is annuitised over, for the LDES floor.
        years_basis_all = remaining
        if "level_row" in capital_terms:
            years_basis_all = np.where(capital_terms["level_row"][idx], capital_terms["level_life_years"][idx], remaining)
        put_eligible("capital_annuity_years", years_basis_all)

    if is_acquired.any():
        mask = np.zeros(n, dtype=bool)
        mask[idx[is_acquired]] = True
        annuity = _plan_annuity(plan, terms.plan_summary)
        put("acquisition_price_gbp", plan["acquisition_price_gbp"].to_numpy(dtype=float).astype(object), mask)
        put("acquisition_basis", plan["acquisition_basis"].to_numpy(dtype=object), mask)
        put("acquisition_term_years_applied", _plan_terms(plan, terms.plan_summary).astype(object), mask)
        # Item G3: the residual owed for cancelled contracts, annuitised on the
        # purchase's own terms; the tariff carries it, required revenue does not.
        for column in _ACQUIRED_KEYS[:4]:
            if column in plan.columns:
                put(column, plan[column].to_numpy(dtype=float).astype(object), mask)
        residual = plan["cancelled_contract_residual_pv_gbp"].to_numpy(dtype=float) * annuity
        put("annual_cancelled_contract_residual_gbp", residual.astype(object), mask)
        put("cancelled_residual_route", plan["cancelled_residual_route"].to_numpy(dtype=object), mask)
    account = {
        "annual_capital_recovery_gbp": capital, "annual_fixed_om_gbp": fixed_om,
        "annual_variable_om_gbp": variable_om, "annual_required_revenue_gbp": required,
        "capex_overrun_share_applied": capex_scale - 1.0, "state_om_premium_share_applied": om_scale - 1.0,
        "annual_state_overhead_gbp": overhead, "annual_site_fixed_om_gbp": site_fixed_om,
        "annual_generator_tnuos_gbp": tnuos, "annual_insurance_gbp": insurance,
        "corporation_tax_forgone_gbp": tax, "annual_tax_equivalent_gbp": tax_equivalent,
        # F29: the socialised systematic risk (not in required revenue; the ledgers read it).
        "socialised_risk_gbp": socialised,
    }
    for key in _ACCOUNT_KEYS:
        put_eligible(key, account[key])
    if rows.margin_slope.any():
        # Follow-up F6: the share of its break-even capex a calibrated row carries at the case's margin.
        put_eligible("auction_margin_capex_scale", margin_scale)
    if self_insurance_column:
        # Follow-up F7: the premium the row saves by self-insuring (inside annual_insurance_gbp's reduction).
        put_eligible("self_insurance_saving_gbp", self_insurance_saving)
    if owners_value_column:
        # F27 item 1: the premium over cost paid to the former owners (inside the capital charge), and
        # (27 Oct 2026) what moves the co-owners' pro rata part of it onto the state's share.
        put_eligible("owners_value_premium_gbp", owners_value_premium)
        put_eligible(OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT, premium_state_adjustment)
    put_eligible("cost_status", rows.cost_status[idx])
    put_eligible("effective_capex_power_gbp_per_kw", capex_power * capex_scale)
    put_eligible("storage_duration_normalised", rows.normalised[idx])
    put_eligible("wacc", wacc)
    put_eligible("remaining_life_years", remaining)
    if offsets is not None:
        # Decision 54: on the NESO wind ramp every post-AR7 wind contract is struck on that path's own
        # calibrated strike, so its CfD support falls by the offset times the energy it is paid on,
        # whoever owns it; the offset is recorded for the calibration and the strike adjustments.
        put("cfd_strike_offset_gbp_per_mwh", offsets["per_mwh"].astype(object))
        if "cfd_support_gbp" in values:
            cfd = pd.to_numeric(pd.Series(values["cfd_support_gbp"]), errors="coerce").fillna(0.0).to_numpy()
            values["cfd_support_gbp"] = (cfd - offsets["gbp"]).astype(object)
    if rows.has_asset_revenue:
        market, support, fuel = rows.market_revenue[idx], rows.support_revenue[idx], rows.fuel_carbon[idx]
        if offsets is not None:
            support = support - offsets["gbp"][idx]
        # Fuel and carbon are deducted before any surplus is read; Capacity Market
        # revenue (item MR4) is the owner's income too, so it is in the rent.
        rent = market + support + rows.capacity_market[idx] - fuel - required
        # Item B5: the state overhead is already the state share of the centre.
        surplus = share_e * (rent + overhead) - overhead
        if co_ownership is not None:
            # Decision 59, separate financing: the state's surplus is its share of the revenue less its OWN
            # cost (its capital at its rate, its O&M premium, its tax equivalent), not its share of the
            # blended row's (which gave it f^2 of its own saving). The co-owner keeps the rest of the rent.
            surplus = surplus + co_adjustment
        # F27 item 1 (27 Oct 2026): the state's share carries the whole owners'-value premium.
        surplus = surplus + premium_state_adjustment
        for key, array in (("market_revenue_gbp", market), ("support_revenue_gbp", support),
                           ("fuel_carbon_gbp", fuel), ("producer_rent_gbp", rent),
                           ("public_surplus_before_recycling_gbp", surplus)):
            put_eligible(key, array)
    else:
        for key in ("market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp", "producer_rent_gbp",
                    "public_surplus_before_recycling_gbp"):
            put_eligible(key, None)
    return values


def rate_path_row_offsets(merged: pd.DataFrame, offsets: Mapping | None) -> dict | None:
    """FES 2050 decision 54: per row, the strike offset each private rate path takes (GBP/MWh) and its money (GBP).

    ``offsets`` is ``contract_book_2050.load_rate_path_offsets``: ``{path: {(cfd_round, table
    technology): GBP/MWh}}``. A row on a post-AR7 CfD (``support_type`` CfD, ``cfd_round``
    ``PAR7_<year>``) takes its entry's offset; the money is the offset times the energy the row is
    paid on (the settlement's ``cfd_paid_mwh``) times its live share already in that energy. None
    without offsets (every 2030 register).
    """
    if not offsets:
        return None
    from scripts.public_ownership.contract_book_2050 import TECHNOLOGY_TABLE_NAME

    n = len(merged)
    support = merged["support_type"].fillna("").astype(str).to_numpy() if "support_type" in merged.columns else np.full(n, "")
    rounds = merged["cfd_round"].fillna("").astype(str).to_numpy() if "cfd_round" in merged.columns else np.full(n, "")
    technology = (merged["stock_flow_technology"].fillna("").astype(str).map(TECHNOLOGY_TABLE_NAME).fillna("").to_numpy()
                  if "stock_flow_technology" in merged.columns else np.full(n, ""))
    paid = (pd.to_numeric(merged["cfd_paid_mwh"], errors="coerce").fillna(0.0).to_numpy()
            if "cfd_paid_mwh" in merged.columns else np.zeros(n))
    out = {}
    for path, table in offsets.items():
        per_mwh = np.array([float(table.get((r, t), 0.0)) if s == "CfD" and r.startswith("PAR7_") else 0.0
                            for s, r, t in zip(support, rounds, technology)])
        out[str(path)] = {"per_mwh": per_mwh, "gbp": per_mwh * paid}
    return out


#: Decision 59 (review 2, 3 Oct 2026): how a co-owned (L4) row is financed, ``public_ownership.co_ownership_finance``.
CO_OWNERSHIP_POOLED, CO_OWNERSHIP_SEPARATE, CO_OWNERSHIP_SEPARATE_REDUCED = (
    "pooled", "separate", "separate_with_private_reduction")
CO_OWNERSHIP_BASES = (CO_OWNERSHIP_POOLED, CO_OWNERSHIP_SEPARATE, CO_OWNERSHIP_SEPARATE_REDUCED)
#: Decision 68's sensitivity: the co-owner's WACC cut, basis points per percentage point of state share.
CO_REDUCTION_BP_PER_POINT = 0.5
#: Register columns a co-ownership setting adds (a register without the setting has none).
CO_OWNERSHIP_COLUMNS = ("co_ownership_basis", "co_owner_rate", "state_part_rate", "capex_state_view_gbp",
                        "capex_private_view_gbp", "co_ownership_state_adjustment_gbp",
                        "co_ownership_state_capital_adjustment_gbp", "co_ownership_ldes_state_adjustment_gbp",
                        "co_investment_rate_gain_gbp")


def co_ownership_settings(block: Mapping | None) -> dict | None:
    """``public_ownership.co_ownership_finance`` checked, or None (pooled, the original treatment, everywhere).

    ``basis`` (decision 68, user, 3 Oct 2026): ``separate`` (central: each owner finances its own
    part at its own rate), ``separate_with_private_reduction`` (the sensitivity: the co-owner's rate
    cut by ``private_rate_reduction_bp_per_point`` basis points per percentage point of state share,
    0.5 centrally, range 0 to 1, on the co-owner's share of co-owned rows only) or ``pooled`` (the
    whole row at the rate whose annuity is the owners' share-weighted annuity, then split by the
    share: a leakage stress test only). ``co_investment_genuine_share`` (0 centrally, at most 0.5) is
    the part of the co-owner's reduction counted as a resource saving; the rest is a transfer the
    Exchequer's implicit guarantee backs (``co_investment_implicit_guarantee_gbp``).
    """
    raw = (block or {}).get("co_ownership_finance")
    if not raw:
        return None
    raw = dict(raw)
    if unknown := sorted(set(raw) - {"basis", "private_rate_reduction_bp_per_point", "co_investment_genuine_share"}):
        raise ValueError(f"co_ownership_finance has unknown keys {unknown}")
    basis = str(raw.get("basis", CO_OWNERSHIP_POOLED))
    if basis not in CO_OWNERSHIP_BASES:
        raise ValueError(f"co_ownership_finance.basis={basis!r}; one of {CO_OWNERSHIP_BASES}")
    per_point = float(raw.get("private_rate_reduction_bp_per_point",
                              CO_REDUCTION_BP_PER_POINT if basis == CO_OWNERSHIP_SEPARATE_REDUCED else 0.0) or 0.0)
    if per_point and basis != CO_OWNERSHIP_SEPARATE_REDUCED:
        raise ValueError("co_ownership_finance.private_rate_reduction_bp_per_point needs basis "
                         "separate_with_private_reduction")
    if not 0.0 <= per_point <= 1.0:
        raise ValueError("co_ownership_finance.private_rate_reduction_bp_per_point must be in [0, 1]")
    genuine = float(raw.get("co_investment_genuine_share", 0.0) or 0.0)
    if not 0.0 <= genuine <= 0.5:
        raise ValueError("co_ownership_finance.co_investment_genuine_share must be in [0, 0.5]")
    return {"basis": basis, "private_rate_reduction_bp_per_point": per_point, "co_investment_genuine_share": genuine}


def _co_view_capital(capital_terms, idx, rate, remaining, capacity, energy, capex_power, capex_energy) -> np.ndarray:
    """A whole row's capital charge at one owner's rate and capex (the co-ownership views, decision 59)."""
    crf = _per_distinct(list(zip(rate.tolist(), remaining.tolist())), capital_recovery_factor)
    capital = (capacity * 1_000.0 * capex_power + energy * 1_000.0 * capex_energy) * crf
    if capital_terms is not None:
        capital = _term_basis_capital(capital_terms, idx, rate, capacity, energy, capex_power, capex_energy, capital)
    return capital


def _term_basis_capital(capital_terms: Mapping[str, np.ndarray], idx: np.ndarray, rate: np.ndarray,
                        capacity: np.ndarray, energy: np.ndarray, capex_power: np.ndarray, capex_energy: np.ndarray,
                        capital: np.ndarray) -> np.ndarray:
    """Decision 41's capital charge on the eligible rows ``idx`` (``vintage_capital_2050.register_capital_terms``)."""
    from scripts.public_ownership.vintage_capital_2050 import tail_value_vector

    term_row, expired = capital_terms["term_row"][idx], capital_terms["expired_row"][idx]
    level = capital_terms["level_row"][idx] if "level_row" in capital_terms else np.zeros(len(idx), dtype=bool)
    past = capital_terms["past_life_row"][idx] if "past_life_row" in capital_terms else np.zeros(len(idx), dtype=bool)
    if not (term_row.any() or expired.any() or level.any() or past.any()):
        return capital
    out = np.where(expired | past, 0.0, capital)
    at = np.flatnonzero(level)
    if len(at):
        # FES 2050 decision 66: a merchant row pays a level annuity over its economic life at its own rate.
        life = capital_terms["level_life_years"][idx][at]
        gross = capacity[at] * 1_000.0 * capex_power[at] + energy[at] * 1_000.0 * capex_energy[at]
        out[at] = gross * _per_distinct(list(zip(rate[at].tolist(), life.tolist())), capital_recovery_factor)
    live = np.flatnonzero(term_row)
    if len(live):
        years = capital_terms["term_years"][idx][live]
        tail = tail_value_vector(capital_terms["tail_margin_gbp_per_kw_year"][idx][live],
                                 capital_terms["life_years"][idx][live], years, rate[live]) * capacity[live] * 1_000.0
        gross = capacity[live] * 1_000.0 * capex_power[live] + energy[live] * 1_000.0 * capex_energy[live]
        crf = _per_distinct(list(zip(rate[live].tolist(), years.tolist())), capital_recovery_factor)
        out[live] = capital_terms["live_share"][idx][live] * np.maximum(gross - tail, 0.0) * crf
        if "merchant_share" in capital_terms:
            # FES 2050 decision 62: the merchant (repowered solar) share keeps the economic-life charge.
            out[live] = out[live] + capital_terms["merchant_share"][idx][live] * capital[live]
    return out


def _corporation_tax_forgone_vector(rows: _RowInputs, terms: _KeyTerms, idx: np.ndarray, owned: np.ndarray,
                                    vintage: np.ndarray, case_wacc: np.ndarray, remaining: np.ndarray,
                                    capital_terms: Mapping[str, np.ndarray] | None = None,
                                    capex_scale: np.ndarray | None = None, stake: np.ndarray | None = None) -> np.ndarray:
    """:func:`_corporation_tax_forgone` for the eligible rows ``idx`` of one key (NaN without a wedge, 0 unowned).

    Decision 41: on the term basis the tax a private owner embeds is on the same capital
    charge (capex less the tail's value at the private rate, over the term, for the live
    share); an expired row embeds none.
    """
    settings = terms.settings
    technology = rows.technology[idx]
    has_wedge = np.array([terms.wedges.get(t) is not None for t in technology], dtype=bool)
    out = np.where(has_wedge, 0.0, np.nan)
    live = np.flatnonzero(has_wedge & owned)
    if not len(live):
        return out
    tech = technology[live]
    capex = rows.capex_total[idx][live]
    if capex_scale is not None:
        capex = capex * np.asarray(capex_scale, dtype=float)[live]   # follow-up F6: the auction margin
    years = remaining[live]
    wacc = case_wacc[live]
    wedge_private = np.array([float(terms.wedges[t]["private"]) for t in tech])
    r_private = _per_distinct([(t,) for t in tech], lambda t: private_rate(terms.rates.get(t), settings))
    if terms.rate_ramp:
        # The tax a private owner would embed is at its own (ramped) rate.
        r_private = r_private + rows.ramp_delta(settings, terms.ramp_deltas)[idx][live]
    if capital_terms is not None:
        from scripts.public_ownership.vintage_capital_2050 import tail_value_vector

        at = idx[live]
        term_row, expired = capital_terms["term_row"][at], capital_terms["expired_row"][at]
        if "level_row" in capital_terms:
            # Decision 66: a merchant row embeds tax on its level annuity (over its life); past its life, none.
            capex = np.where(capital_terms["past_life_row"][at], 0.0, capex)
            years = np.where(capital_terms["level_row"][at], capital_terms["level_life_years"][at], years)
        if term_row.any() or expired.any():
            capex = np.where(expired, 0.0, capex)
            years = np.where(term_row, capital_terms["term_years"][at], years)
            tail = tail_value_vector(capital_terms["tail_margin_gbp_per_kw_year"][at], capital_terms["life_years"][at],
                                     capital_terms["term_years"][at], r_private) * rows.capacity[at] * 1_000.0
            capex = np.where(term_row, capital_terms["live_share"][at] * np.maximum(capex - tail, 0.0), capex)

    def embedded(rate: np.ndarray, wedge: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # finance.embedded_corporation_tax, its factor once per distinct (rate, wedge, years)
        factor = _per_distinct(list(zip(rate[mask].tolist(), wedge[mask].tolist(), years[mask].tolist())),
                               lambda r, w, y: 0.0 if y <= 0
                               else capital_recovery_factor(r, y) - capital_recovery_factor(r - w, y))
        result = np.zeros(len(tech))
        c = capex[mask]
        result[mask] = np.where((years[mask] <= 0) | (c == 0.0), 0.0, c * factor)
        return result

    everyone = np.ones(len(tech), dtype=bool)
    tax = embedded(r_private, wedge_private, everyone)
    kind = str(settings.get(LEVER_PUBLIC_RETURN, DEFAULT_SETTINGS[LEVER_PUBLIC_RETURN]))
    new = vintage[live] != "legacy"
    if kind == RETURN_CEPA_DECOMPOSITION and new.any():
        # The JV remains a taxable company: only the change is forgone.
        wedge_pc = np.array([float(terms.wedges[t]["public_coordination"]) if m else np.nan
                             for t, m in zip(tech, new)])
        tax = np.where(new, tax - embedded(wacc, wedge_pc, new), tax)
    if stake is not None and kind != RETURN_CEPA_DECOMPOSITION:
        # F10: GB Energy's offshore stake is a joint venture (P1's mechanics), a taxable company: only
        # the change is forgone, as under the CEPA decomposition.
        joint = new & np.asarray(stake, dtype=bool)[live]
        if joint.any():
            wedge_pc = np.array([float(terms.wedges[t]["public_coordination"]) if m else np.nan
                                 for t, m in zip(tech, joint)])
            tax = np.where(joint, embedded(r_private, wedge_private, everyone) - embedded(wacc, wedge_pc, joint), tax)
    if terms.comparator is not None and terms.tax_basis is not None:
        at = idx[live]
        # F19 decision 3 (a): the comparator's plant pays no tax where it earns below its required
        # return, so none is forgone on the state's new build there (judged on its cohort, below).
        loss_making = (terms.comparator["loss_making"][at] if "loss_making" in terms.comparator
                       else terms.comparator["rent"][at] < -LOSS_TOLERANCE_GBP)
        tax = np.where(new & loss_making, 0.0, tax)
        # (b): an acquired legacy row forgoes the tax on its 2030 operating margin, after
        # capital allowances on its tax written-down value and interest, not the as-new embedded tax.
        if (~new).any():
            margin = terms.comparator["rent"][at] + terms.comparator["capital"][at]
            legacy_tax = legacy_corporation_tax(margin, rows.capex_total[at], rows.commissioning_year[at],
                                                terms.tax_basis)
            tax = np.where(new, tax, legacy_tax)
    out[live] = tax
    return out


#: F19 decision 3: the corporation-tax basis of the forgone tax (config
#: ``public_ownership.corporation_tax_basis``; kept equal to it).
#: F19 decision 3: a comparator cohort's rent below this (GBP) is loss-making (float noise is not).
LOSS_TOLERANCE_GBP = 1.0

#: F27 item 8: the market-group register columns a bid-stack group's comparator is read from.
COMPARATOR_REFERENCE_COLUMNS = ("case_id", "asset_id", "producer_rent_gbp", "annual_capital_recovery_gbp",
                                "carrier", "support_type", "cfd_round", "finance_vintage")

CORPORATION_TAX_BASIS_DEFAULTS: dict = {
    "enabled": True,
    "in_vintaged_registers": False,
    "rate": 0.25,
    "special_rate_writing_down_allowance": 0.06,
    "special_rate_first_year_allowance": 0.50,
    "first_year_allowance_from_year": 2023,
    "gearing": 0.70,
    "cost_of_debt_nominal": 0.07,
}


def _comparator_rows(keyed, settings_by_spec, built, private_key, build, booked, rows,
                     reference: Callable[[str], pd.DataFrame] | None = None) -> dict[int, dict]:
    """F19 decision 3: ``{id(terms): {"rent", "capital"}}``, the comparator's final rows for each owning key.

    A key's comparator is its first case's (``comparator_for_settings``); every case in a key
    shares its register levers, so they share the comparator's register rows.  The comparator
    is accounted once, booked as the register books it (carbon CfD, RAB, support lines, LDES
    floor), and read positionally (each case's rows are the asset table in order).  Keys with
    no owned rows (the private comparators) get none.

    ``reference`` (F27 item 8, 23 Oct 2026) returns the comparator case's rows from the
    network's market group: a bid-stack group's own private rows are private on the state's
    bid stack, which is not a feasible case (the state's units are turned down more, so whole
    cohorts looked loss-making and P2's state-bids case forgave GBP 785m less tax). The
    comparator is then judged on the market months, as the bill's merge measures it.
    """
    key_of_case = {spec.case_id: i for i, (specs, _) in enumerate(keyed) for spec in specs}
    cache: dict[str, dict] = {}

    def comparator_arrays(case_id: str) -> dict:
        if case_id not in cache:
            if reference is not None:
                frame = reference(case_id)
                frame = frame.loc[frame["case_id"].astype(str).eq(case_id)].drop_duplicates("asset_id")
                frame = frame.set_index(frame["asset_id"].astype(str)).reindex(rows.asset_ids)
                missing = int(frame["case_id"].isna().sum())
                if missing:
                    raise ValueError(f"comparator {case_id}: {missing} assets of this register are not in the "
                                     "market group's register (it must hold the same fleet)")
                frame = frame.reset_index(drop=True)
            else:
                i = key_of_case[case_id]
                specs, terms = keyed[i]
                spec = next(s for s in specs if s.case_id == case_id)
                if i == private_key and i in built and case_id in set(built[i][0]["case_id"].astype(str)):
                    frame = built[i][0].copy()
                else:
                    frame = build([spec], terms)[0]
                frame = booked(frame, [spec])
                frame = frame.loc[frame["case_id"].astype(str).eq(case_id)]
            if len(frame) != rows.n or not np.array_equal(frame["asset_id"].astype(str).to_numpy(), rows.asset_ids):
                raise ValueError(f"comparator {case_id}: its rows are not the asset table in order")
            num = lambda c: pd.to_numeric(frame[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)  # noqa: E731
            rent = num("producer_rent_gbp")
            # Whether the comparator's plant earns its required return is judged on its cohort (carrier,
            # support, round and vintage): a calibrated CfD cohort breaks even (plus its auction margin)
            # as a whole, so a below-average site in it is not loss-making plant; LDES under its floor,
            # merchant solar and hydro are.
            keys = [frame[c].astype(str).fillna("") if c in frame.columns else pd.Series("", index=frame.index)
                    for c in ("carrier", "support_type", "cfd_round", "finance_vintage")]
            cohort_rent = pd.Series(rent, index=frame.index).groupby(keys, dropna=False).transform("sum")
            cache[case_id] = {"rent": rent, "capital": num("annual_capital_recovery_gbp"),
                              "loss_making": cohort_rent.to_numpy(dtype=float) < -LOSS_TOLERANCE_GBP}
        return cache[case_id]

    out = {}
    for specs, terms in keyed:
        if str(specs[0].case_id).split(VARIANT_SEPARATOR, 1)[0] == PRIVATE_SCENARIO:
            continue
        out[id(terms)] = comparator_arrays(comparator_for_settings(specs[0].case_id, settings_by_spec))
    return out


def corporation_tax_basis(block: Mapping | None) -> dict:
    """``public_ownership.corporation_tax_basis`` over the code defaults, checked (F19 decision 3)."""
    out = {**CORPORATION_TAX_BASIS_DEFAULTS, **dict(block or {})}
    unknown = sorted(set(out) - set(CORPORATION_TAX_BASIS_DEFAULTS))
    if unknown:
        raise ValueError(f"public_ownership.corporation_tax_basis has unknown keys {unknown}")
    for key in ("rate", "special_rate_writing_down_allowance", "special_rate_first_year_allowance", "gearing",
                "cost_of_debt_nominal"):
        if not 0.0 <= float(out[key]) <= 1.0:
            raise ValueError(f"public_ownership.corporation_tax_basis.{key} must be a share in [0, 1]")
    return out


def legacy_corporation_tax(margin, capex, commissioning_year, basis: Mapping) -> np.ndarray:
    """F19 decision 3 (b): ``rate x max(0, margin - allowances - interest)`` in the model year, per row.

    * ``margin``: the comparator row's operating margin (rent plus capital charge: revenue
      less every non-capital cost), the profit before capital allowances and interest;
    * allowances: generation plant in the special-rate pool (long-life assets, CAA 2001
      s.91 and s.104A), the as-new capex written down at the special rate from its
      commissioning year, after the 50% special-rate first-year allowance on expenditure
      from 1 April 2023 (CAA 2001 s.45S); the model year's allowance is the rate times the
      written-down value brought forward;
    * interest: debt at the workbook gearing on the written-down value (debt amortising with
      the plant's tax value, a declared proxy) at a nominal cost of debt.

    The written-down value is on today's as-new capex (the register's base), a declared
    approximation to the historical cost.  A row with no commissioning year is taken as
    commissioned in the model year less one.
    """
    margin = np.asarray(margin, dtype=float)
    capex = np.asarray(capex, dtype=float)
    year = int(basis["model_year"])
    built = np.where(np.isfinite(np.asarray(commissioning_year, dtype=float)),
                     np.asarray(commissioning_year, dtype=float), year - 1)
    built = np.minimum(built, year - 1)
    wda = float(basis["special_rate_writing_down_allowance"])
    fya = np.where(built >= float(basis["first_year_allowance_from_year"]),
                   float(basis["special_rate_first_year_allowance"]), 0.0)
    # Written-down value at the start of the model year: the first year takes the FYA (else the WDA).
    first = np.where(fya > 0.0, fya, wda)
    years_after_first = np.maximum(year - built - 1.0, 0.0)
    wdv = capex * (1.0 - first) * (1.0 - wda) ** years_after_first
    allowance = wda * wdv
    interest = float(basis["gearing"]) * wdv * float(basis["cost_of_debt_nominal"])
    taxable = np.maximum(np.nan_to_num(margin) - allowance - interest, 0.0)
    return float(basis["rate"]) * taxable


def _key_frame(rows: _RowInputs, terms: _KeyTerms, kinds: Mapping[str, np.ndarray], specs: Sequence[CaseSpec],
               columns: Sequence[str], **account_options) -> pd.DataFrame:
    """One register key's rows: each asset once per case sharing the key, typed as the per-row dicts were."""
    values = _key_account(rows, terms, kinds, **account_options)
    k = len(specs)
    take = np.repeat(np.arange(rows.n), k)
    data = {}
    for column in columns:
        if column == "case_id":
            data[column] = np.tile(np.array([spec.case_id for spec in specs], dtype=object), rows.n)
            continue
        source = values.get(column)
        typed = _records_column(source if source is not None else np.full(rows.n, np.nan, dtype=object))
        data[column] = typed[take] if k != 1 else typed
    return pd.DataFrame(data, columns=list(columns))


def _case_totals(register: pd.DataFrame) -> pd.DataFrame:
    """The financing summary's sums per case, over rows holding whole cases (one register key's, or all)."""
    return register.groupby("case_id", dropna=False, sort=False).agg(
        assets=("asset_id", "size"),
        resolved_assets=("annual_required_revenue_gbp", "count"),
        annual_required_revenue_gbp=("annual_required_revenue_gbp", "sum"),
        annual_capital_recovery_gbp=("annual_capital_recovery_gbp", "sum"),
        annual_state_overhead_gbp=("annual_state_overhead_gbp", "sum"),
        annual_generator_tnuos_gbp=("annual_generator_tnuos_gbp", "sum"),
        annual_insurance_gbp=("annual_insurance_gbp", "sum"),
        corporation_tax_forgone_gbp=("corporation_tax_forgone_gbp", "sum"),
        annual_tax_equivalent_gbp=("annual_tax_equivalent_gbp", "sum"),
        carbon_removal_cfd_revenue_gbp=("carbon_removal_cfd_revenue_gbp", "sum"),
        public_surplus_before_recycling_gbp=(
            "public_surplus_before_recycling_gbp", lambda values: values.sum(min_count=1)
        ),
    ).reset_index()


def _book_ldes_floor_to_rows(register: pd.DataFrame, settings: Mapping | None, has_revenue: bool,
                             market_by_case: Mapping[str, bool] | None = None) -> pd.DataFrame:
    """Item B2: the LDES floor top-up the bill pays, in the store's producer rent and the state's surplus.

    Only market-bill cases pay the floor (the acquisition tariff replaces it);
    the amount per row is :func:`public_account.ldes_floor_topups` on the case's
    ``ldes_floor_basis``, the same function the account and the bill use.
    ``market_by_case`` (each case's ``market_bill()``, when the caller has its
    specs) saves reading it back off the lever columns.
    """
    from scripts.public_ownership.public_account import ldes_floor_topups

    register = register.copy()
    register["ldes_floor_topup_paid_gbp"] = 0.0
    if not settings or not has_revenue or register.empty:
        return register
    market = (pd.Series(dict(market_by_case)) if market_by_case is not None
              else register.groupby("case_id", sort=False).apply(
                  lambda rows: spec_for_row(case_settings(rows)).market_bill(), include_groups=False))
    mask = register["case_id"].map(market).astype(bool)
    if not mask.any():
        return register
    floors = ldes_floor_topups(register.loc[mask], settings)
    paid = floors["ldes_floor_topup_paid_gbp"].astype(float)
    register.loc[mask, "ldes_floor_topup_paid_gbp"] = paid
    share = pd.to_numeric(register["ownership_share"], errors="coerce").fillna(0.0)
    booked = register["ldes_floor_topup_paid_gbp"]
    has_rent = register["producer_rent_gbp"].notna()
    register.loc[has_rent, "producer_rent_gbp"] = register.loc[has_rent, "producer_rent_gbp"] + booked[has_rent]
    register.loc[has_rent, "public_surplus_before_recycling_gbp"] = (
        register.loc[has_rent, "public_surplus_before_recycling_gbp"] + share[has_rent] * booked[has_rent])
    if "ldes_floor_topup_state_part_gbp" in floors.columns:
        # Decision 59 (review 2): a separately financed co-owned store's state part is floored on its own cost.
        from scripts.public_ownership.public_account import co_ownership_adjustments

        rows = register.loc[mask].assign(ldes_floor_topup_state_part_gbp=floors["ldes_floor_topup_state_part_gbp"])
        _, floor_adjustment = co_ownership_adjustments(rows)
        adjustment = pd.Series(0.0, index=register.index)
        adjustment.loc[mask] = floor_adjustment.to_numpy()
        register["co_ownership_ldes_state_adjustment_gbp"] = adjustment
        register.loc[has_rent, "public_surplus_before_recycling_gbp"] = (
            register.loc[has_rent, "public_surplus_before_recycling_gbp"] + adjustment[has_rent])
    return register


#: Lever parameters the register needs (config ``public_ownership.lever_parameters``);
#: these defaults equal the repository config so a missing block cannot move a number.
LEVER_PARAMETER_DEFAULTS: dict = {
    "state_stake_funded_scale": 0.15,          # GB Energy + NWF capital at a funded scale
    "delivered_share_half": 0.5,               # P2 delivers half of each technology's post-2025 cohort
    "capex_overrun_share": {OVERRUN_PLUS_15: 0.15},
    "state_om_premium_share": {STATE_OM_PLUS_10: 0.10},
    "generation_control_premium_share": {"twenty_pct": 0.20},
    # Follow-up F6 (user, 4 Oct 2026): the auction margin's share of contract revenue.
    "auction_margin_share": {AUCTION_MARGIN_CENTRAL: 0.05, AUCTION_MARGIN_NONE: 0.0, AUCTION_MARGIN_HIGH: 0.10},
    # Follow-up F7 (user, 4 Oct 2026): the insurer's loading a wholly state-owned row saves,
    # and whether vintaged (2040 on) registers take it (2050 has not decided: off).
    "self_insurance_share": {SELF_INSURANCE_CENTRAL: 0.35, SELF_INSURANCE_NONE: 0.0, SELF_INSURANCE_HIGH: 0.50},
    "self_insurance_in_vintaged_registers": False,
    # F10 (user, 6 Oct 2026): GB Energy as announced. A 30% stake in the post-2025 offshore wind
    # cohort (the Crown Estate partnership) and 8 GW of local solar and onshore wind (the Local
    # Power Plan), pro rata across that cohort; the partnership sensitivity holds the 8 GW at 50%.
    # User decision (option E, 28 Oct 2026): GB Energy's eligibility is dated from the GB Energy Act 2025
    # (Royal Assent 15 May 2025), not the group's legislation year; None judges it at the group's year.
    "gbe_announced": {"offshore_stake_share": 0.30, "offshore_carriers": ["wind_offshore"],
                      "local_capacity_mw": 8000.0, "local_carriers": ["solar_pv", "wind_onshore"],
                      "partnership_public_share": 0.5, "legislated_year": 2025},
    # F27 item 6 (24 Oct 2026): lever capacity_procurement's cost-based contracts for existing units.
    capacity_procurement.PARAMETER_KEY: dict(capacity_procurement.DEFAULT_PARAMETERS),
    # User decision (24 Oct 2026): lever legacy_offshore_price's deal multiples.
    offshore_deal_multiple.PARAMETER_KEY: dict(offshore_deal_multiple.DEFAULT_PARAMETERS),
}
#: F10: the register's gbe_part values (cfd_repricing gives the stake P1's contract form).
GBE_PART_STAKE, GBE_PART_LOCAL = "offshore_stake", "local"


def lever_parameters(block: Mapping | None) -> dict:
    """The lever parameters with the repository defaults filled in, checked."""
    out = {**LEVER_PARAMETER_DEFAULTS, **dict(block or {})}
    unknown = sorted(set(out) - set(LEVER_PARAMETER_DEFAULTS))
    if unknown:
        raise ValueError(f"public_ownership.lever_parameters has unknown keys {unknown}")
    for key in ("state_stake_funded_scale", "delivered_share_half"):
        if not 0.0 < float(out[key]) < 1.0:
            raise ValueError(f"public_ownership.lever_parameters.{key} must be a share in (0, 1)")
    for key in ("capex_overrun_share", "state_om_premium_share", "generation_control_premium_share",
                "auction_margin_share", "self_insurance_share"):
        for value, share in dict(out[key]).items():
            if not 0.0 <= float(share) <= 2.0:
                raise ValueError(f"public_ownership.lever_parameters.{key}.{value} must be a share in [0, 2]")
    defaults = LEVER_PARAMETER_DEFAULTS["gbe_announced"]
    gbe = {**defaults, **dict(out["gbe_announced"] or {})}
    if set(gbe) != set(defaults):
        raise ValueError(f"public_ownership.lever_parameters.gbe_announced takes {sorted(defaults)}")
    for key in ("offshore_stake_share", "partnership_public_share"):
        if not 0.0 < float(gbe[key]) <= 1.0:
            raise ValueError(f"public_ownership.lever_parameters.gbe_announced.{key} must be a share in (0, 1]")
    if float(gbe["local_capacity_mw"]) < 0.0:
        raise ValueError("public_ownership.lever_parameters.gbe_announced.local_capacity_mw must be >= 0")
    if gbe["legislated_year"] is not None:
        if isinstance(gbe["legislated_year"], bool) or int(gbe["legislated_year"]) != float(gbe["legislated_year"]):
            raise ValueError("public_ownership.lever_parameters.gbe_announced.legislated_year must be a year or null")
        gbe["legislated_year"] = int(gbe["legislated_year"])
    out["gbe_announced"] = gbe
    out[capacity_procurement.PARAMETER_KEY] = capacity_procurement.parameters(out[capacity_procurement.PARAMETER_KEY])
    out[offshore_deal_multiple.PARAMETER_KEY] = offshore_deal_multiple.parameters(out[offshore_deal_multiple.PARAMETER_KEY])
    return out


def gbe_selection(rows, gbe: Mapping) -> dict:
    """F10: the rows GB Energy as announced holds, inside P2's boundary (post-2025 new build, not a pre-2025 CfD).

    ``offshore``: every such row of the offshore carriers (the stake). ``local``: whole rows of
    the local carriers taken pro rata, the same share of every (technology, build year) cell
    (``finance.deployment_selection``, md5 order), until ``local_capacity_mw``. Returns the masks
    and the capacity each part holds.
    """
    eligible = getattr(rows, "gbe_eligible_share", None)
    if eligible is None:
        boundary = ((getattr(rows, "base_vintage", rows.vintage) == "new_build") & rows.resolved & ~rows.excluded
                    & ~rows.pre_policy_cfd)
        weighted = rows.capacity
    else:
        # User decision (option E, 28 Oct 2026): eligibility is GB Energy's own year (the GB Energy Act 2025),
        # by the legislation-year commitment rules; each row counts its share committed on or after it
        # (rows split at the group's year are not split again). Net Zero Teesside's pre-2025 DPA stays out.
        eligible = np.asarray(eligible, dtype=float)
        boundary = (eligible > 0.0) & rows.resolved & ~rows.excluded & ~getattr(rows, "pre_2025_dpa", False)
        weighted = rows.capacity * eligible
    carrier = np.asarray(rows.carrier, dtype=object).astype(str)
    offshore = boundary & np.isin(carrier, list(gbe["offshore_carriers"]))
    candidates = boundary & np.isin(carrier, list(gbe["local_carriers"]))
    total = float(weighted[candidates].sum())
    target = float(gbe["local_capacity_mw"])
    if target > total + 1e-6:
        raise ValueError(f"GB Energy's local capacity ({target:,.0f} MW) exceeds the post-2025 cohort it is drawn "
                         f"from ({total:,.0f} MW)")
    share = target / total if total > 0 else 0.0
    local = deployment_selection(weighted, rows.cells(), rows.order_keys(), candidates, share)
    return {"offshore": offshore, "local": local, "offshore_mw": float(weighted[offshore].sum()),
            "local_mw": float(weighted[local].sum()), "local_cohort_mw": total, "local_share": share}


def undelivered_rows(merged: pd.DataFrame, share: float) -> set:
    """Lever new_build_delivered_share = half: the new-build rows the state does NOT deliver.

    Each carrier's post-2025 cohort is split by capacity: rows in asset-id order,
    delivered while the cumulative capacity is within ``share`` of the carrier's
    total (the row that crosses the line is delivered when more than half of it
    lies inside).  Deterministic, and the same rows in every case that takes it.
    """
    new_build = merged.loc[merged["finance_vintage"].astype(str).eq("new_build")
                           & merged["evidence_status"].astype(str).eq("resolved")]
    out = set()
    for _, rows in new_build.groupby(new_build["carrier"].astype(str), sort=True):
        rows = rows.sort_values("asset_id")
        capacity = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0)
        total = float(capacity.sum())
        if total <= 0:
            continue
        midpoint = capacity.cumsum() - capacity / 2.0
        out.update(rows.index[midpoint > share * total])
    return out


def calculate_financing_scenarios(
    assets: pd.DataFrame, costs: pd.DataFrame, finance_snapshot: dict, *,
    cases: tuple[str, ...] | None = None,
    scenarios: Mapping[str, Mapping[str, str]] | None = None,
    sensitivities: Mapping | None = None,
    state_equity_share_value: float | None = None,
    demand_mwh: float | None = None,
    price_year: int | None = None,
    case_specs: Sequence[CaseSpec] | None = None,
    axes: Mapping[str, Mapping] | None = None,
    bounding_pair: bool = True,
    acquisition: Mapping | None = None,
    ro_stations: pd.DataFrame | None = None,
    state_operating_overhead: Mapping | None = None,
    generator_costs: Mapping | None = None,
    generator_cost_tables: Mapping | None = None,
    ldes_account_settings: Mapping | None = None,
    excluded_carriers: Sequence[str] | None = None,
    cost_calibration: pd.DataFrame | None = None,
    electricity_leg_carriers: Sequence[str] | None = None,
    hpc_completion: pd.DataFrame | None = None,
    lever_parameter_block: Mapping | None = None,
    register_sink=None,
    register_cases: Sequence[str] | None = None,
    vintage_costs: pd.DataFrame | None = None,
    hurdle_rates: pd.DataFrame | None = None,
    vintage_capital: Mapping | None = None,
    support_lines: Mapping | None = None,
    co_ownership: Mapping | None = None,
    corporation_tax_basis_block: Mapping | None = None,
    held_acquisition_prices: pd.DataFrame | None = None,
    held_acquisition_source: str | None = None,
    comparator_reference: Callable[[str], pd.DataFrame] | None = None,
    capacity_market_design: Mapping | None = None,
    legislation_settings: Mapping | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Calculate one account row per asset/cohort/case.

    ``legislation_settings`` (``policy_start_year``, ``snapshot_year``, ``lead_times``; option E, 28 Oct 2026)
    lets GB Energy's eligibility be judged at its own year (``lever_parameters.gbe_announced.legislated_year``)
    by the legislation-year rules (:func:`legislation_year.eligible_share_at`); defaults when absent.

    ``capacity_market_design`` (FES 2050 decision 73): the Capacity Market design options
    (:func:`capacity_market_design.settings`); each case's CM revenue, rent and state surplus
    are booked to its design and the financing summary gains its ``capacity_market_design_*``
    columns. Vintaged (2050) registers only, and never beside lever ``capacity_procurement``
    (F27 item 6, the 2030 mechanism on the same receipts): both are refused. None (every 2030
    config) changes nothing.

    ``register_sink`` (30 Sep 2026): a callable given each register key's rows
    in turn (every case sharing the key, lever columns attached, in register
    order).  With it the register is never held whole, the function returns
    ``(None, totals)`` and peak memory stays flat in the number of cases
    (``main`` streams to :class:`register_io.RegisterWriter` this way); without
    it the rows are concatenated and returned as before.

    ``register_cases`` (30 Sep 2026, :func:`run_group_case_specs`): only these
    cases are accounted and returned, each with exactly the rows, columns,
    order and totals it has in the register of every case (the register keys,
    the column order and the acquisition plans are still read over
    ``case_specs``).

    ``generator_costs`` (config ``public_ownership.generator_costs``) and its
    loaded ``generator_cost_tables`` add generator TNUoS and insurance to every
    row's fixed costs (item D4(a), :func:`generator_charges`); absent, neither
    is modelled.  ``ldes_account_settings`` (:func:`public_account.account_settings`)
    puts the LDES floor top-up the bill pays into the store's producer rent
    (item B2); absent, no floor is booked to rows.  ``excluded_carriers`` is the
    plant the state never owns (the legacy valuation's list).

    Review 29 Sep 2026: ``cost_calibration`` (:func:`load_cost_calibration`)
    scales the capex of cohorts calibrated to their cleared strike, in every case;
    ``electricity_leg_carriers`` books Power BECCS's carbon-removal CfD
    (:func:`book_carbon_removal_cfd`); ``hpc_completion`` prices Hinkley Point C's
    completion by the state; ``lever_parameter_block`` holds the stake-scale,
    delivered-share, overrun, public O&M and control-premium parameters.

    Assets with unresolved evidence or unreviewed costs are retained in the
    coverage output but intentionally excluded from financial headline totals.
    ``price_year`` (config ``public_ownership.price_year``) is enforced against
    the cost registry when given.

    The case set comes from one of three places: ``case_specs`` (already
    expanded); ``scenarios`` and ``sensitivities`` (the item-K1 config shape,
    expanded here); or ``cases`` with ``axes`` (the item-D1 shape over the
    legacy case names, kept for tests).  With none of them the three legacy
    market cases are accounted at the baseline point.  Each spec writes one row
    set labelled ``<scenario>@<label>`` and carries its lever settings as
    columns, which is how the later stages read the point they are accounting.

    ``acquisition`` (config ``public_ownership.acquisition``) and ``ro_stations``
    (the Ofgem RO register) are needed only when some case acquires the legacy
    fleet: its legacy rows are then costed at their annuitised acquisition
    price at the public rate instead of original capital.

    FES 2050 plan, task Y10: ``vintage_costs`` (:func:`vintage_cost_registry`)
    costs every row by technology AND build year (``build_year``, else
    ``commissioning_year``; a year before the table's first takes its first, one
    after its last its last) in place of ``costs``.  Only a vintaged model year
    (2040 on) may take it, and every costed row must then carry a build year.
    ``hurdle_rates`` (Y2's ``fes2025_hurdle_rates.csv``) feeds the
    ``private_rate_path`` lever; it is read from its default path when a case
    takes the NESO wind ramp and none is given.

    FES 2050 decision 41: ``vintage_capital`` (``vintage_capital_2050.basis_from_config``
    plus ``insurance_config`` and ``learning``) puts a vintaged register's contract rows on
    the term basis and its expired-contract rows at operating cost
    (:func:`_term_basis_capital`); off, or the economic-life basis, changes nothing. A
    vintaged register's Sizewell C RAB rows are paid their allowed revenue
    (:func:`book_rab_allowed_revenue`) in every case.
    """
    missing = REQUIRED_ASSET_COLUMNS - set(assets.columns)
    if missing:
        raise ValueError(f"Asset register lacks columns: {sorted(missing)}")
    # Item Y4: the snapshot is read for the register's own model year (the
    # sovereign-yield shift is the gilt path in that year).
    finance_snapshot = with_model_year(finance_snapshot, frame_model_year(assets))
    if case_specs is None:
        if scenarios is not None:
            sens = dict(sensitivities or {})
            sens.setdefault("bounding_pair", bounding_pair)
            case_specs = expand_cases(scenarios, sens)
        else:
            legacy_cases = tuple(cases) if cases is not None else DEFAULT_LEGACY_CASES
            if "private" not in legacy_cases:
                raise ValueError("cases must include the 'private' baseline")
            case_specs = expand_axis_cases(legacy_cases, axes or {}, bounding_pair=bounding_pair)
    case_specs = list(case_specs)
    if len({spec.case_id for spec in case_specs}) != len(case_specs):
        raise ValueError("case_specs repeats a case id")
    if not any(spec.case_id == "private" for spec in case_specs):
        raise ValueError("cases must include the 'private' baseline")
    params = lever_parameters(lever_parameter_block)
    # Only the register levers change the required-revenue account, so cases
    # that share a register key share every number: account once, label many.
    by_key: dict[tuple, list[CaseSpec]] = {}
    for spec in case_specs:
        by_key.setdefault(spec.register_key(), []).append(spec)
    # FES 2050 plan: gated levers get a register column only where a case sets one.
    lever_columns = LEVER_COLUMNS + tuple(gated_levers_in(case_specs))
    # Tasks Y10/Y12: a vintaged register (2040 on) carries the 2030 purchase of the legacy fleet.
    vintaged_register = is_vintaged_model_year(frame_model_year(assets))
    # FES 2050 decision 73: each case's Capacity Market design result, filled as its key is booked.
    cm_design_results: dict[str, dict] = {}
    if capacity_market_design:
        _check_capacity_market_design_scope(vintaged_register, frame_model_year(assets), case_specs)
    if vintage_costs is not None:
        # Task Y10: a vintaged register is costed by build year (a 2030 register never is).
        model_year = frame_model_year(assets)
        if not is_vintaged_model_year(model_year):
            raise ValueError(f"vintage costs apply to vintaged model years only; this register is for {model_year}")
        from scripts.public_ownership.vintage_costs import FIRST_BUILD_YEAR, LAST_BUILD_YEAR

        assets = assets.copy()
        assets["cost_build_year"] = np.clip(build_year_vector(assets), FIRST_BUILD_YEAR, LAST_BUILD_YEAR)
    asset_fields = list(assets.columns)
    reviewed_costs = _validated_costs(
        costs if vintage_costs is None else vintage_costs, price_year=price_year,
        rate_basis=finance_snapshot.get("rate_basis")
    )
    assets = assets.copy()
    # F27 item 5 (23 Oct 2026): the 2030 register may cost a row on another row than its rate
    # technology too (OCGT and gas engines: CCGT's rate, their own DESNZ costs).
    # ``_technology_key`` stays the rate technology (rows.technology reads it for the WACC).
    cost_names = (assets["cost_technology"].where(assets["cost_technology"].notna(), assets["finance_technology"])
                  if "cost_technology" in assets.columns else assets["finance_technology"])
    assets["_technology_key"] = assets["finance_technology"].map(canonical_technology)
    reviewed_costs = reviewed_costs.copy()
    reviewed_costs["_technology_key"] = reviewed_costs["technology"].map(canonical_technology)
    if vintage_costs is None:
        merged = assets.assign(_cost_key=cost_names.map(canonical_technology)).merge(
            reviewed_costs.rename(columns={"_technology_key": "_cost_key"}), how="left", on="_cost_key",
            suffixes=("", "_cost"), validate="many_to_one",
        ).drop(columns=["_cost_key"])
    else:
        # Task Y14-prep: a row may cost on a vintage row other than its rate technology.
        cost_key = cost_names.map(canonical_technology)
        costed = cost_key.isin(set(reviewed_costs["_technology_key"]))
        undated = costed & ~np.isfinite(assets["cost_build_year"].to_numpy(dtype=float))
        if undated.any():
            raise ValueError(f"a vintaged register needs a build year on every costed row; {int(undated.sum())} have "
                             f"none (e.g. {assets.loc[undated, 'asset_id'].astype(str).head(3).tolist()}): run the "
                             "Y10 join (fleet_vintages.assign_build_years) first")
        reviewed_costs = reviewed_costs.rename(columns={"cost_build_year": "_cost_build_year",
                                                        "_technology_key": "_cost_key"})
        merged = assets.assign(_cost_build_year=assets["cost_build_year"].astype(float), _cost_key=cost_key).merge(
            reviewed_costs, how="left", on=["_cost_key", "_cost_build_year"],
            suffixes=("", "_cost"), validate="many_to_one",
        ).drop(columns=["_cost_build_year", "_cost_key"])
    # User decision 1 (29 Sep 2026): trust the auction.  Calibrated cohorts' capex is
    # scaled in every case before any cost that depends on it is computed.
    # FES 2050 (task E2E): the calibration is a fit of the CP30 2030 fleet's rounds; a
    # vintaged register is costed by build year from Y2's table and its post-AR7 strikes
    # are that vintage cost at the private rate (decision 6), so no auction is trusted.
    merged = apply_cost_calibration(merged, cost_calibration if vintage_costs is None else None)
    if vintage_costs is not None:
        # Year-generic stage 5b: the strike path's capex factor (AR7's fixed-bottom offshore k), so an at-cost
        # strike equals the register's required revenue; private_cost_calibration.csv never reaches a vintaged
        # register (the line above passes None), so nothing is scaled twice.
        from scripts.market.cfd_strikes import vintage_capex_factors

        factor = vintage_capex_factors(cost_names.to_numpy(), snapshot=finance_snapshot,
                                       insurance_config=(generator_costs or {}).get("insurance"))
        for column in ("capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh"):
            if column in merged.columns:
                merged[column] = pd.to_numeric(merged[column], errors="coerce") * factor
        merged["capex_calibration_factor"] = factor
    if vintage_costs is not None:
        merged = apply_refurbishment_costs(merged, vintage_costs.attrs.get("refurbishment_capex_share") or {})
    # Item D4(a): TNUoS and insurance are properties of the plant, identical in
    # every case, so they are computed once per row here.
    charges = generator_charges(merged, generator_costs, generator_cost_tables)
    merged["_generator_fixed_charges_gbp"] = (charges["annual_generator_tnuos_gbp"]
                                              + charges["annual_insurance_gbp"]).to_numpy()
    # Decision D2: the corporation-tax wedge per technology, for the Exchequer ledger.
    tax_wedges = {canonical_technology(k): v for k, v in corporation_tax_wedges(finance_snapshot).items()}
    rates = _technology_rates(finance_snapshot)
    equity_share = state_equity_share_value
    if equity_share is None:
        equity_share = float(
            finance_snapshot.get("assumptions", {})
            .get("public_coordination_state_equity_share", {})
            .get("value", 0.30)
        )
    # Lever state_stake_scale: a funded-scale stake with the Scenario 2 WACC and
    # tax wedge recomputed at that stake (the workbook calculator, item J2).
    funded_rates, funded_wedges = rates, tax_wedges
    funded_share = float(params["state_stake_funded_scale"])
    if any(spec.settings.get(LEVER_STATE_STAKE_SCALE) == STAKE_FUNDED_SCALE for spec in case_specs):
        funded_snapshot = _snapshot_at_stake(finance_snapshot, funded_share)
        funded_rates = _technology_rates(funded_snapshot)
        funded_wedges = {canonical_technology(k): v for k, v in corporation_tax_wedges(funded_snapshot).items()}
    undelivered = (undelivered_rows(merged, float(params["delivered_share_half"]))
                   if any(spec.settings.get(LEVER_DELIVERED_SHARE) == DELIVERED_HALF for spec in case_specs) else set())
    has_asset_revenue = {"market_revenue_gbp", "support_revenue_gbp"}.issubset(asset_fields)

    # F19 P3 decision 3: the support lines that are properties of the asset (ancillary
    # services, sub-hourly uplift, gate fees) are operating income the owners keep, so the
    # purchase price values them as the legacy valuation does (none without support lines).
    plan_support = asset_support_lines(merged, support_lines) if acquisition else {}

    # One acquisition plan per distinct set of acquisition terms and public rate.
    acquiring = [specs for specs in by_key.values() if specs[0].acquires_legacy()]
    if acquiring and (not acquisition or (ro_stations is None and not vintaged_register)):
        raise ValueError("acquiring the legacy fleet needs the acquisition settings and the Ofgem RO register")

    def _terms(settings):
        terms = acquisition_terms(acquisition, settings)
        terms["hpc_completion_table"] = hpc_completion
        terms["generation_control_premium_share"] = params["generation_control_premium_share"]
        terms["support_by_asset"] = plan_support
        terms["capacity_cost_based_existing"] = params[capacity_procurement.PARAMETER_KEY]
        terms["offshore_transaction_multiple"] = params[offshore_deal_multiple.PARAMETER_KEY]
        # F29: the state completes Hinkley Point C at its cash rate (workbook v2.0 carries no
        # construction premium), so hpc_construction_premium stays at its default, zero.
        return terms

    # The plan key leaves out the legacy_fleet lever, so one plan can serve cases that buy only the
    # valued group and cases that buy everything (20 Oct 2026): every case id on a plan key, and
    # whether any of them buys the whole fleet, for the held purchase price.
    keyed_acquiring = []
    plan_cases: dict[tuple, list[str]] = {}
    plan_acquires_all: dict[tuple, bool] = {}
    for specs in acquiring:
        settings = specs[0].settings
        terms = _terms(settings)
        rate = public_legacy_rate(finance_snapshot, settings)
        plan_key = _plan_key(terms, rate)
        keyed_acquiring.append((terms, rate, plan_key))
        plan_cases.setdefault(plan_key, []).extend(str(spec.case_id) for spec in specs)
        plan_acquires_all[plan_key] = plan_acquires_all.get(plan_key, False) or any(
            spec.settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE) == LEGACY_ACQUIRE_ALL for spec in specs)

    plans: dict[tuple, tuple[pd.DataFrame, dict | None, float]] = {}
    for terms, rate, plan_key in keyed_acquiring:
        if plan_key not in plans:
            if vintaged_register:
                # Tasks Y10/Y12: a vintaged register carries the 2030 purchase, never re-prices it.
                plan, summary = carried_acquisition_plan(merged, terms, public_rate=rate)
            else:
                # F19 valuation decision (user, 14 Oct 2026): the purchase counts the top-ups sized on
                # the private case's account, so it is priced once that account is built (below,
                # _price_plans).  Until then a plan carries only its scope, which is all the
                # register's column layout and the private build read.
                plan, summary = acquisition_scope_plan(merged, terms), None
            plans[plan_key] = (plan, summary, rate)

    def _price_plans(top_ups: Mapping, owners_view: Mapping | None = None) -> None:
        """Price every pending plan with the purchase's support lines plus ``top_ups`` (by asset id), in place;
        ``owners_view`` (column -> asset id -> value) replaces the preview's own figures where given."""
        support = {**plan_support, **dict(top_ups)}
        for terms, rate, plan_key in keyed_acquiring:
            scope, summary, _ = plans[plan_key]
            if summary is not None:
                continue
            plan, summary = plan_acquisition(merged, rates, finance_snapshot,
                                             {**terms, "support_by_asset": support,
                                              "owners_view_by_asset": dict(owners_view or {})},
                                             ro_stations, public_rate=rate)
            if held_acquisition_prices is not None:
                # Network bounds (18-19 Oct 2026): the purchase held at the CENTRAL network's prices.
                plan, summary = hold_acquisition_prices(plan, summary, merged, held_acquisition_prices,
                                                        plan_cases[plan_key], held_acquisition_source,
                                                        acquires_all=plan_acquires_all[plan_key])
            check_plan_scope(scope, plan)
            if top_ups:
                summary["purchase_top_ups_gbp_model_year"] = {
                    column: float(pd.Series(by_asset, dtype=float).reindex(
                        merged.loc[merged["finance_vintage"].astype(str).eq("legacy"), "asset_id"].astype(str))
                        .fillna(0.0).sum()) for column, by_asset in top_ups.items()}
            plans[plan_key] = (plan, summary, rate)

    def _plan_for(settings):
        if settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE) == LEGACY_PRIVATE:
            return None, None, None
        terms = acquisition_terms(acquisition, settings)
        rate = public_legacy_rate(finance_snapshot, settings)
        return plans[_plan_key(terms, rate)]

    # Item OH1.  The scope rule is a config value, not a literal: the acquired
    # networks keep their RIIO allowed opex in the network finance account, so
    # charging an overhead on them here would double-count.
    overhead_cfg = dict(state_operating_overhead or {})
    overhead_applies_to = tuple(overhead_cfg.get("applies_to") or SUPPORTED_OVERHEAD_COMPONENTS)

    rows = _RowInputs(merged, charges, asset_fields, has_asset_revenue, excluded_carriers, undelivered)
    if acquisition:
        from scripts.public_ownership.legacy_valuation import not_operating_mask

        rows.completed_by_state = not_operating_mask(merged, valuation_settings(dict(acquisition.get("valuation") or {})))
    # D1 (user, 14 Nov 2026): a later year's buying cases own exactly what the acquisition year's purchase bought.
    baseline_path = (acquisition or {}).get("carried_baseline")
    if baseline_path and "model_year" in merged.columns and is_vintaged_model_year(frame_model_year(merged)):
        from scripts.public_ownership.acquisition_baseline import (
            baseline_acquisition_year, load_baseline, load_round_caps)

        path = Path(str(baseline_path))
        path = path if path.is_absolute() else PROJECT_ROOT / path
        rows.set_acquisition_baseline(merged, load_baseline(path), acquisition_year=baseline_acquisition_year(path),
                                      round_caps=load_round_caps(path))
    gbe_cases = [spec for spec in case_specs if spec.settings.get(LEVER_NEW_BUILD_GENERATION) in GBE_VALUES]
    gbe_params = params["gbe_announced"]
    if gbe_cases:
        if abs(float(gbe_params["offshore_stake_share"]) - float(equity_share)) > 1e-9:
            raise ValueError(f"GB Energy's offshore stake ({gbe_params['offshore_stake_share']}) must equal the stake the "
                             f"Scenario 2 WACC is computed at ({equity_share}): P1's mechanics are the snapshot's")
        if gbe_params["legislated_year"] is not None and "legislation_status" in merged.columns:
            # Option E (user decision, 28 Oct 2026): GB Energy's own year, the GB Energy Act 2025.
            from scripts.public_ownership.legislation_year import eligible_share_at, load_lead_times

            leg = dict(legislation_settings or {})
            rows.gbe_eligible_share = eligible_share_at(
                merged, int(gbe_params["legislated_year"]), policy_start_year=int(leg.get("policy_start_year", 2025)),
                model_year=int(frame_model_year(assets)), snapshot_year=int(leg.get("snapshot_year", 2025)),
                leads=load_lead_times(leg.get("lead_times")), vintaged=vintaged_register)
        selection = gbe_selection(rows, gbe_params)
        rows.gbe_offshore, rows.gbe_local = selection["offshore"], selection["local"]
        print(f"F10 GB Energy as announced: offshore stake on {selection['offshore_mw']:,.0f} MW "
              f"({gbe_params['offshore_stake_share']:.0%}, "
              f"{selection['offshore_mw'] * float(gbe_params['offshore_stake_share']):,.0f} MW); local "
              f"{selection['local_mw']:,.0f} MW of {selection['local_cohort_mw']:,.0f} MW "
              f"({selection['local_share']:.1%} of each cell)")
    # FES 2050 decision 32: a switch, off unless the study's vintage_capital_2050 block sets it.
    rows.rab_outside_propositions = (merged["support_type"].astype(str).eq(RAB_SUPPORT_TYPE).to_numpy()
                                     if vintaged_register and (vintage_capital or {}).get("rab_outside_propositions")
                                     else None)
    held_carriers = [str(c) for c in ((vintage_capital or {}).get("held_private_carriers") or [])]
    rows.held_private = (merged["carrier"].astype(str).isin(held_carriers).to_numpy()
                         if vintaged_register and held_carriers else None)
    # FES 2050 decision 54: the strike offset of each private rate path, per row (GBP/MWh and GBP).
    rows.strike_offsets = rate_path_row_offsets(merged, (vintage_capital or {}).get("rate_path_offsets"))
    ramp_deltas = None
    if any(str(spec.settings.get(LEVER_PRIVATE_RATE_PATH, lever_default(LEVER_PRIVATE_RATE_PATH)))
           == RATE_PATH_NESO_WIND_RAMP for spec in case_specs):
        ramp_deltas = {canonical_technology(k): v for k, v in neso_wind_ramp_deltas(hurdle_rates).items()}
    account_options = {"finance_snapshot": finance_snapshot, "overhead_applies_to": overhead_applies_to}
    if co_ownership is not None:
        # Decision 59 (review 2): the co-owned rows' financing basis (absent: pooled, no new column).
        account_options["co_ownership"] = co_ownership
    if vintage_capital is not None and vintage_capital.get("enabled"):
        if not vintaged_register:
            raise ValueError("vintage_capital_2050 applies to vintaged (2040 on) registers only")
        from scripts.public_ownership.vintage_capital_2050 import register_capital_terms

        account_options["capital_terms"] = register_capital_terms(
            merged, vintage_capital, insurance_config=vintage_capital.get("insurance_config"),
            learning=bool(vintage_capital.get("learning", False)))

    def _terms_for(settings: Mapping[str, str]) -> _KeyTerms:
        terms = _terms_for_base(settings)
        generation = settings.get(LEVER_NEW_BUILD_GENERATION)
        if generation in GBE_VALUES:
            terms.gbe = {"offshore_stake_share": float(gbe_params["offshore_stake_share"]),
                         "local_public_share": (float(gbe_params["partnership_public_share"])
                                                if generation == NEW_BUILD_GBE_PARTNERSHIP else 1.0)}
        return terms

    def _terms_for_base(settings: Mapping[str, str]) -> _KeyTerms:
        plan, plan_summary, public_rate = _plan_for(settings)
        funded = settings.get(LEVER_STATE_STAKE_SCALE) == STAKE_FUNDED_SCALE
        return _KeyTerms(
            settings, plan=plan, plan_summary=plan_summary, public_rate=public_rate,
            rates=funded_rates if funded else rates, wedges=funded_wedges if funded else tax_wedges,
            share=funded_share if funded else equity_share, overhead_rate=state_overhead_rate(overhead_cfg, settings),
            overhead_carrier_rates=state_overhead_carrier_rates(overhead_cfg, settings, vintaged=vintaged_register),
            overrun=float(params["capex_overrun_share"].get(settings.get(LEVER_STATE_CAPEX_OVERRUN), 0.0)),
            om_premium=float(params["state_om_premium_share"].get(settings.get(LEVER_STATE_OM), 0.0)),
            ramp_deltas=ramp_deltas,
            auction_margin=float(params["auction_margin_share"].get(
                str(settings.get(LEVER_AUCTION_MARGIN, lever_default(LEVER_AUCTION_MARGIN))), 0.0)),
            self_insurance=(0.0 if vintaged_register and not params["self_insurance_in_vintaged_registers"]
                            else float(params["self_insurance_share"].get(
                                str(settings.get(LEVER_SELF_INSURANCE, lever_default(LEVER_SELF_INSURANCE))), 0.0))))

    keyed = [(specs, _terms_for(dict(specs[0].settings))) for specs in by_key.values()]
    settings_by_spec = {spec.case_id: spec.settings for spec in case_specs}
    # The register's columns: the union of every row's keys in the order they
    # first occur, the order pandas gave the per-row dicts (a short first pass).
    order = _union_in_order(
        _kind_keys(asset_fields, kind) for _, terms in keyed for kind in _kinds_in_order(_key_kinds(rows, terms)["kind"]))
    if rows.strike_offsets:
        # FES 2050 decision 54: the rate path's strike offset each row was settled at (NaN off the path).
        order = order + ["cfd_strike_offset_gbp_per_mwh"]
    if co_ownership is not None:
        order = order + list(CO_OWNERSHIP_COLUMNS)
    if account_options.get("capital_terms") is not None:
        order = order + ["capital_annuity_years"]
    if rows.margin_slope.any():
        # Follow-up F6: written only where the calibration has cohort sums (the 2030 register).
        order = order + ["auction_margin_capex_scale"]
    if gbe_cases:
        # F10: which part of GB Energy's portfolio a row is (absent from every other register).
        order = order + ["gbe_part"] + (["gbe_eligible_share"] if getattr(rows, "gbe_eligible_share", None) is not None
                                        else [])
    if any(terms.self_insurance for _, terms in keyed):
        # Follow-up F7: written only when a case self-insures (no vintaged register by default).
        order = order + ["self_insurance_saving_gbp"]
        account_options["self_insurance_column"] = True
    if any(spec.settings.get(LEVER_PRE_2025_CFD) == PRE_2025_PUBLIC_OWNERS_VALUE for spec in case_specs):
        # F27 item 1: written only when a case buys the pre-2025 CfD plant at its owners' value.
        order = order + ["owners_value_premium_gbp", OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT]
        account_options["owners_value_column"] = True

    def _build(specs, terms) -> tuple[pd.DataFrame, dict]:
        kinds = _key_kinds(rows, terms)
        frame = _key_frame(rows, terms, kinds, specs, order, **account_options).infer_objects()
        return _attach_lever_columns(frame, specs, lever_columns), kinds

    # Streaming (30 Sep 2026): one register key's rows at a time, handed to
    # ``register_sink`` when given, so peak memory does not grow with the case
    # count.  The BECCS electricity-leg share is read off the private rows, so the
    # key holding the private case is built first.
    kept = None if register_cases is None else {str(c) for c in register_cases}
    if kept is not None and "private" not in kept:
        raise ValueError("register_cases must include the 'private' baseline")

    def _wanted(specs):
        return list(specs) if kept is None else [spec for spec in specs if spec.case_id in kept]

    private_key = next(i for i, (specs, _) in enumerate(keyed) if any(s.case_id == "private" for s in specs))
    built = {private_key: _build(_wanted(keyed[private_key][0]), keyed[private_key][1])}
    leg_share = carbon_removal_leg_share(built[private_key][0], electricity_leg_carriers)
    # FES 2050 decision 44: the carbon CfD on its own term (vintaged registers with a term only).
    carbon_live = (carbon_cfd_live_share(merged, electricity_leg_carriers,
                                         (vintage_capital or {}).get("beccs_carbon_cfd_term_years"))
                   if vintaged_register and vintage_capital is not None else None)
    merchant_removals = (merchant_removal_value(merged, carbon_live,
                                                (vintage_capital or {}).get("beccs_removal_value_gbp_per_mwh"))
                         if carbon_live is not None else None)
    # FES 2050 decision 41 (c): Sizewell C's RAB allowance, read off the private rows (none in 2030).
    rab_support = (rab_allowed_revenue_support(built[private_key][0].loc[built[private_key][0]["case_id"].eq("private")])
                   if vintaged_register else None)
    # FES 2050 decision 58 and the battery items (decisions 63, 67), any model year that configures them
    # (public_ownership.support_lines; none in 2030 unless a 2030 study turns them on).
    from scripts.public_ownership.support_lines_2050 import require_vintaged

    require_vintaged(support_lines, vintaged_register)
    support_by_asset = support_lines_by_asset(
        merged, built[private_key][0], support_lines,
        book=lambda f: book_rab_allowed_revenue(
            book_carbon_removal_cfd(f, electricity_leg_carriers, leg_share=leg_share, live_share=carbon_live,
                                    merchant_removals=merchant_removals), rab_support),
        reference=(support_lines or {}).get("reference"),
        ldes_carriers=((ldes_account_settings or {}).get("ldes_carriers") or ()
                       if (ldes_account_settings or {}).get("ldes_enabled") else ()))
    if rab_support is not None and not rab_support.empty and support_by_asset:
        # Review 2 (3 Oct 2026): the RAB pays its allowed revenue, so the support-line income Sizewell C earns
        # (its ancillary services) nets off the allowance instead of adding to its rent; the bill's RAB line
        # falls by it. CfD and DPA rows keep theirs, which decision 41 (b)'s calibration nets in its next pass.
        from scripts.public_ownership.support_lines_2050 import SUPPORT_COLUMNS

        income = sum((pd.Series(support_by_asset[c]).reindex(rab_support.index).fillna(0.0)
                      for c in SUPPORT_COLUMNS if c in support_by_asset), pd.Series(0.0, index=rab_support.index))
        rab_support = rab_support - income

    from scripts.public_ownership.support_lines_2050 import owner_cost_top_up

    owner_cost_carriers = (owner_cost_top_up(support_lines, vintaged_register)
                           if support_by_asset and "battery_revenue_top_up_gbp" in support_by_asset else None)

    if acquiring and not vintaged_register:
        # F19 valuation decision (user, 14 Oct 2026): the top-ups (the legacy batteries' revenue top-up,
        # GBP 30 m a year on the CENTRAL register) are owner cash, so the purchase values them, as the
        # valuation file and the streams the tariff reads already do; then every key takes its priced plan.
        top_ups = {column: support_by_asset[column] for column in PURCHASE_TOP_UP_COLUMNS
                   if column in support_by_asset}
        # Independent review (15 Nov 2026): Drax BECCS's carbon-removal CfD is booked only after the register is
        # built, so the plan's preview never saw it and the purchase valued the plant without it (GBP 225 m a year
        # on the 2030 private rows). The owners' contract income, as the legacy valuation counts it, at the
        # private case's booking.
        private_rows = built[private_key][0]
        private_rows = book_carbon_removal_cfd(
            private_rows.loc[private_rows["case_id"].astype(str).eq("private")], electricity_leg_carriers,
            leg_share=leg_share, live_share=carbon_live, merchant_removals=merchant_removals)
        cfdc = pd.to_numeric(private_rows["carbon_removal_cfd_revenue_gbp"], errors="coerce").fillna(0.0)
        if cfdc.ne(0.0).any():
            top_ups["carbon_removal_cfd_revenue_gbp"] = (
                cfdc.groupby(private_rows["asset_id"].astype(str)).sum().loc[lambda s: s.ne(0.0)].to_dict())
        from scripts.public_ownership.legacy_valuation import as_new_capex as _as_new_capex

        ids = private_rows["asset_id"].astype(str)
        owners_view = {
            "as_new_capex_gbp": pd.Series(_as_new_capex(private_rows), index=ids.to_numpy()).groupby(level=0).sum().to_dict(),
            "annual_fixed_om_gbp": pd.to_numeric(private_rows["annual_fixed_om_gbp"], errors="coerce")
            .groupby(ids).sum().to_dict(),
        }
        _price_plans(top_ups, owners_view)
        for _, key_terms in keyed:
            if key_terms.plan is not None:
                key_terms.plan, key_terms.plan_summary, _ = _plan_for(key_terms.settings)

    def _booked(frame: pd.DataFrame, specs) -> pd.DataFrame:
        if capacity_market_design:
            # FES 2050 decision 73: each case's CM design first, so the LDES floor below nets the
            # design's CM revenue (the support lines are sized on the private rows, which it never moves).
            from scripts.public_ownership.capacity_market_design import book as _book_cm_design

            frame, results = _book_cm_design(frame, capacity_market_design, specs)
            cm_design_results.update(results)
        frame = book_carbon_removal_cfd(frame, electricity_leg_carriers, leg_share=leg_share, live_share=carbon_live,
                                        merchant_removals=merchant_removals)
        frame = book_rab_allowed_revenue(frame, rab_support)
        frame = book_support(frame, support_by_asset)
        if owner_cost_carriers:
            # F19 decision 5: each case's battery top-up at its owner's cost (the private row unchanged).
            from scripts.public_ownership.support_lines_2050 import rebook_battery_top_up_at_owner_cost

            frame = rebook_battery_top_up_at_owner_cost(frame, owner_cost_carriers)
        # User decision (6 Oct 2026): the LDES floor is booked after the support lines, so it
        # nets every income the store earns (``public_account.ldes_floor_topups``), as the
        # account recomputes it; no support line is sized on a floored store's rent.
        return _book_ldes_floor_to_rows(frame, ldes_account_settings, has_asset_revenue,
                                        {spec.case_id: spec.market_bill() for spec in specs})

    # F19 decision 3: the forgone corporation tax is capped at what each case's comparator would
    # pay, so every key that owns plant reads its comparator's final rent and capital charge.
    tax_basis = corporation_tax_basis(corporation_tax_basis_block)
    if tax_basis["enabled"] and (not vintaged_register or tax_basis["in_vintaged_registers"]):
        tax_basis = {**tax_basis, "model_year": frame_model_year(assets)}
        comparator_rows = _comparator_rows(keyed, settings_by_spec, built, private_key, _build, _booked, rows,
                                           reference=comparator_reference)
        for specs, terms in keyed:
            terms.comparator, terms.tax_basis = comparator_rows.get(id(terms)), tax_basis
    frames: list[pd.DataFrame] = []
    totals_parts: list[pd.DataFrame] = []
    acquisition_by_case: dict[str, dict] = {}
    for i, (specs, terms) in enumerate(keyed):
        wanted = _wanted(specs)
        if not wanted:
            # A key none of whose cases this register holds: only its acquisition
            # summary is kept, so the summary has the columns the full register gave it.
            if terms.plan is not None:
                case_summary = _acquisition_summary_for(
                    terms.plan_summary, pd.Series(_key_kinds(rows, terms)["acquired"], index=merged.index), terms.plan)
                acquisition_by_case.update({spec.case_id: case_summary for spec in specs})
            continue
        frame, kinds = built.pop(i) if i in built else _build(wanted, terms)
        frame = _booked(frame, wanted)
        totals_parts.append(_case_totals(frame))
        if terms.plan is not None:
            case_summary = _acquisition_summary_for(
                terms.plan_summary, pd.Series(kinds["acquired"], index=merged.index), terms.plan)
            for spec in specs:
                acquisition_by_case[spec.case_id] = case_summary
        if register_sink is not None:
            register_sink(frame)
        else:
            frames.append(frame)
        del frame
    register = None
    if register_sink is None:
        # One DataFrame of every row, typed as a single DataFrame of the dicts would be.
        register = pd.concat(frames, ignore_index=True).infer_objects()
        del frames
        # Categories over the whole register, as one mapping of every case gave them.
        for lever, column in lever_categoricals(register["case_id"], settings_by_spec, lever_columns).items():
            register[lever] = column
    totals = pd.concat(totals_parts, ignore_index=True)
    order = {spec.case_id: i for i, spec in enumerate(case_specs)}
    totals = totals.sort_values("case_id", key=lambda s: s.map(order)).reset_index(drop=True)
    totals["unresolved_assets"] = totals["assets"] - totals["resolved_assets"]
    private_total = totals.loc[
        totals["case_id"].eq("private"), "annual_required_revenue_gbp"
    ]
    if len(private_total) != 1:
        raise ValueError("Financing comparison requires exactly one private baseline")
    # House rule 3: a case that moves a lever the private baseline also takes
    # (the WACC basis, merchant contracts at cost, the storage floor, the
    # pay-as-bid bound) is measured against the private case at the same point,
    # never against the central private case.
    settings_by_case = {spec.case_id: spec.settings for spec in case_specs}
    rr_by_case = totals.set_index("case_id")["annual_required_revenue_gbp"]
    totals["comparator_case_id"] = totals["case_id"].map(lambda c: comparator_for_settings(c, settings_by_case))
    totals["annual_financing_saving_vs_private_gbp"] = (
        totals["comparator_case_id"].map(rr_by_case).to_numpy() - totals["annual_required_revenue_gbp"].to_numpy()
    )
    # Decision D2: the financing saving split into the corporation tax the state
    # does not pay (a transfer from the Exchequer) and the resource saving.
    # Lever tax_equivalent: tax the state pays as a tax equivalent is inside its
    # required revenue and back with the Exchequer, so it is no transfer at all.
    net_tax = totals["corporation_tax_forgone_gbp"] - totals["annual_tax_equivalent_gbp"].fillna(0.0)
    ct_by_case = pd.Series(net_tax.to_numpy(), index=totals["case_id"])
    totals["annual_financing_saving_tax_transfer_gbp"] = (
        net_tax.to_numpy() - totals["comparator_case_id"].map(ct_by_case).to_numpy())
    totals["annual_financing_saving_resource_gbp"] = (
        totals["annual_financing_saving_vs_private_gbp"] - totals["annual_financing_saving_tax_transfer_gbp"])
    # This is intentionally not called a bill reduction.  It is the amount
    # that could be allocated to consumers under a later, explicit recovery
    # rule; no surplus-recycling or household-allocation assumption is made.
    totals["unallocated_consumer_benefit_gbp"] = totals[
        "annual_financing_saving_vs_private_gbp"
    ]
    totals["consumer_equivalent_saving_gbp_per_mwh_demand"] = (
        totals["unallocated_consumer_benefit_gbp"] / float(demand_mwh)
        if demand_mwh is not None and float(demand_mwh) > 0 else None
    )
    totals["public_surplus_status"] = (
        "asset_settlement_linked" if has_asset_revenue
        else "not_calculated_without_asset_settlement"
    )
    for lever in lever_columns:
        totals[lever] = totals["case_id"].map(lambda c: settings_by_case[c].get(lever, lever_default(lever)))
    totals["base_case"] = totals["case_id"].map(lambda c: str(c).split(VARIANT_SEPARATOR, 1)[0])
    totals["market_bill"] = totals["case_id"].map(
        lambda c: next(spec for spec in case_specs if spec.case_id == c).market_bill())
    if capacity_market_design and cm_design_results:
        # FES 2050 decision 73: the financing summary's capacity_market_design_* columns.
        from scripts.public_ownership.capacity_market_design import summary_columns

        design = summary_columns(cm_design_results, capacity_market_design.get("rebasing_factor"))
        totals = totals.merge(design, on="case_id", how="left", validate="one_to_one")
    if acquisition_by_case:
        def _acq(case_id, key):
            summary = acquisition_by_case.get(case_id)
            return summary[key] if summary else np.nan

        totals["acquisition_purchase_price_gbp"] = totals["case_id"].map(
            lambda c: (_acq(c, "valued_purchase_price_gbp") + _acq(c, "thermal_and_storage_purchase_price_gbp"))
            if c in acquisition_by_case else np.nan)
        totals["acquisition_annual_charge_gbp"] = totals["case_id"].map(lambda c: _acq(c, "annual_acquisition_charge_gbp"))
        totals["acquisition_purchase_price_basis"] = totals["case_id"].map(
            lambda c: acquisition_by_case[c]["purchase_price_basis"] if c in acquisition_by_case else None)
        totals["acquisition_term_years"] = totals["case_id"].map(lambda c: _acq(c, "term_years"))
        totals["acquisition_public_rate"] = totals["case_id"].map(lambda c: _acq(c, "public_real_rate"))
        totals.attrs["acquisition"] = {case: dict(summary) for case, summary in acquisition_by_case.items()}
    return register, totals


#: The price year of ``cm_revenue_by_technology.csv``: the Capacity Market builder
#: (scripts/market/build_capacity_market.py, PRICE_YEAR) carries every agreement
#: to 2025 money, the bill's report year.
CM_RATES_PRICE_YEAR = 2025
# Item Y4: the money columns are recognised by name for any year
# (model_year.is_cm_rate_column: the per-kW rate, every delivery-year and every
# calendar-year revenue), not listed for one year.


def capacity_market_rates_in_price_year(rates: pd.DataFrame, price_year: int | None, rebasing) -> pd.DataFrame:
    """The CM rates carried from their 2025 money onto the register's price year.

    The register is in ``price_year`` money (2024) and the bill turns it into
    report money (2025) with its ``currency_rebasing`` factor, so the rates are
    divided by that same factor: a CM payment then round-trips to the contract
    book's own 2025 total exactly (item MR4). Left unconverted they were ~3.8%
    high in every register use: the LDES floor netting, the nationalised
    tariff's Capacity Market split and the legacy valuation's operating cash.
    ``rebasing`` is ``(from_year, to_year, factor)``; without it the rates are
    refused unless the register is itself in 2025 money.
    """
    rates = rates.copy()
    if price_year is None or int(price_year) == CM_RATES_PRICE_YEAR:
        return rates
    if rebasing is None:
        raise ValueError(
            f"the Capacity Market rates are in {CM_RATES_PRICE_YEAR} money and the register in {price_year}; "
            "pass --capacity-market-rebasing FROM TO FACTOR (the bill's currency_rebasing)")
    from_year, to_year, factor = int(rebasing[0]), int(rebasing[1]), float(rebasing[2])
    if (from_year, to_year) != (int(price_year), CM_RATES_PRICE_YEAR):
        raise ValueError(f"--capacity-market-rebasing runs {from_year} to {to_year}; the register needs "
                         f"{price_year} to {CM_RATES_PRICE_YEAR}")
    if not factor > 0:
        raise ValueError("the rebasing factor must be positive")
    for column in cm_rate_columns(rates.columns):
        rates[column] = pd.to_numeric(rates[column], errors="coerce") / factor
    return rates


def _check_capacity_market_design_scope(vintaged_register: bool, model_year, case_specs) -> None:
    """FES 2050 decision 73 is a vintaged-register design; it never shares a register with the 2030 mechanisms.

    The 2030 Capacity Market under ownership is F9 (the 2030/31 T-4 re-cleared with the state's
    plant bidding at cost, in the public account; off in a vintaged register) and F27 item 6
    (lever ``capacity_procurement``: existing units on cost-based contracts). Decision 73
    re-clears the 2050 auction with the state's plant out, which re-prices the same receipts,
    so a register may carry one or the other, never both.
    """
    if not vintaged_register:
        raise ValueError(f"public_ownership.capacity_market_design (FES 2050 decision 73) applies to vintaged "
                         f"(2050) registers only; this register is for {model_year}, where the Capacity Market "
                         "under ownership is F9 (capacity_market_state_bids) and capacity_procurement")
    from scripts.public_ownership.capacity_procurement import AUCTION

    clash = [spec.case_id for spec in case_specs
             if str(spec.settings.get(LEVER_CAPACITY_PROCUREMENT, AUCTION)) != AUCTION]
    if clash:
        raise ValueError(f"cases {clash[:5]} set capacity_procurement (F27 item 6, a 2030 Capacity Market lever) "
                         "in a register under capacity_market_design (decision 73); the two re-price the same "
                         "Capacity Market receipts, so a register takes one or the other")


def register_options_from_config(block: Mapping, root: Path | str = ".") -> dict:
    """The register's config-driven options, shared by the Snakemake rule and ``run_financing.py``.

    Generator TNUoS and insurance (item D4(a)), the LDES floor booked to its
    payees (item B2) and the plant the state never owns (item B4).
    """
    from scripts.public_ownership.public_account import account_settings

    generator_costs = block.get("generator_costs") or {}
    acquisition = block.get("acquisition") or {}
    valuation = valuation_settings(acquisition.get("valuation"))
    calibration = dict(block.get("cost_calibration") or {})
    hpc = acquisition.get("hpc_completion")
    repricing = ((block.get("consumer_bill") or {}).get("cfd_repricing") or {})
    return {
        "generator_costs": generator_costs,
        "generator_cost_tables": load_generator_cost_tables(generator_costs, root),
        "ldes_account_settings": account_settings(block),
        "excluded_carriers": tuple(valuation["excluded_carriers"]),
        # Review 29 Sep 2026 (user decisions 1, 2 and 10, and the explorer's levers).
        "cost_calibration": (load_cost_calibration(Path(root) / calibration["table"])
                             if calibration.get("enabled") and calibration.get("table") else None),
        "electricity_leg_carriers": list(repricing.get("at_cost_electricity_leg_carriers") or []),
        "hpc_completion": pd.read_csv(Path(root) / hpc) if hpc else None,
        "lever_parameter_block": block.get("lever_parameters"),
        # FES 2050 decision 41: absent from every 2030 config, so None there.
        "vintage_capital": vintage_capital_options(block, root),
        # Decisions 58, 63 and 67: the support lines, only when the config names them (no key in 2030).
        **({"support_lines": support_line_settings(block, root)} if block.get("support_lines") else {}),
        # Decision 59 (review 2): the co-owned rows' financing basis, only when the config names it.
        **({"co_ownership": co_ownership_settings(block)} if block.get("co_ownership_finance") else {}),
        # FES 2050 decision 73: the Capacity Market design, only when the config names it (no key in 2030).
        **({"capacity_market_design": capacity_market_design_settings(block, root)}
           if block.get("capacity_market_design") else {}),
        # Option E (28 Oct 2026): the legislation rules GB Energy's own year is judged by.
        "legislation_settings": {
            "policy_start_year": int(block.get("policy_start_year", 2025)),
            "snapshot_year": int((block.get("legislation") or {}).get("snapshot_year", 2025)),
            "lead_times": str(Path(root) / ((block.get("legislation") or {}).get(
                "lead_times", "data/public_ownership/financing/legislation_lead_times.csv")))},
    }


def capacity_market_design_settings(block: Mapping, root: Path | str = ".") -> dict | None:
    """``public_ownership.capacity_market_design`` checked, its auction inputs loaded (decision 73), or None."""
    from scripts.public_ownership.capacity_market_design import settings

    return settings(block.get("capacity_market_design"), root)


def support_line_settings(block: Mapping, root: Path | str = ".") -> dict | None:
    """``public_ownership.support_lines`` checked (:func:`support_lines_2050.settings`), or None."""
    from scripts.public_ownership.support_lines_2050 import settings

    return settings(block.get("support_lines"), root)


def vintage_capital_options(block: Mapping, root: Path | str = ".") -> dict | None:
    """Decision 41's capital basis for the register (``public_ownership.vintage_capital_2050``), or None."""
    from scripts.public_ownership.vintage_capital_2050 import CONFIG_KEY, basis_from_config

    if not (block or {}).get(CONFIG_KEY):
        return None
    basis = basis_from_config(block, root)
    basis["insurance_config"] = dict(((block.get("generator_costs") or {}).get("insurance")) or {})
    return basis


def _with_vintage_learning(options: dict, block: Mapping, run_group: str | None) -> dict:
    """Decision 41: the capital basis's tail is priced on the group's own vintage costs (learning or not)."""
    if options.get("vintage_capital"):
        group = ((block.get("reference_run_groups") or {}).get(run_group) or {}) if run_group else {}
        options["vintage_capital"]["learning"] = bool((group.get("vintage_costs") or {}).get("learning", False))
    return options


def _with_rate_path_offsets(options: dict, path) -> dict:
    """Decision 54: the rate-path strike offsets reach the register through the capital basis."""
    if path:
        if not options.get("vintage_capital"):
            raise ValueError("--rate-path-offsets needs the study's vintage_capital_2050 block")
        from scripts.public_ownership.contract_book_2050 import load_rate_path_offsets

        options["vintage_capital"]["rate_path_offsets"] = load_rate_path_offsets(path)
    return options


def _with_support_reference(options: dict, path) -> dict:
    """FES 2050 decisions 58 and 63: a balancing run group's support lines, held at its market group's.

    ``path`` is the market group's register; its private rows' support columns are read (a few
    columns of one case). Without a path (every 2030 group) the options are unchanged.
    """
    if path:
        if not options.get("support_lines"):
            raise ValueError("--support-reference needs the study's public_ownership.support_lines block")
        from scripts.public_ownership.register_io import read_register
        from scripts.public_ownership.support_lines_2050 import MEMO_COLUMNS, SUPPORT_COLUMNS

        options["support_lines"]["reference"] = read_register(
            path, columns=["asset_id", *SUPPORT_COLUMNS, *MEMO_COLUMNS], cases=["private"])
    return options


REGISTER_FILE_SUFFIXES = ("_asset_finance_register.parquet", "_asset_finance_register.csv")


def run_group_from_register_path(path: str | Path) -> str | None:
    """``{run_group}_asset_finance_register.parquet`` -> ``run_group`` (None for any other name)."""
    name = Path(path).name
    for suffix in REGISTER_FILE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def run_group_case_specs(block: Mapping, run_group: str | None, specs: Sequence[CaseSpec]) -> list[CaseSpec]:
    """The cases a run group registers (30 Sep 2026): its own, not every configured case.

    A balancing group (one whose months were solved on another bid stack, P2 or
    P3) projects a handful of market cases (:func:`reference_runs.group_market_cases`)
    and prices its tariff cases (:func:`reference_runs.group_tariff_cases`); every
    later stage of the group reads only those.  So its register holds them, the
    private case and the private comparator of each (house rule 3: the first
    private-family case at the same point on every all_cases lever, which the
    later stages look up in the register itself), in configured order.  Every
    number a kept case carries is the one the full register gave it.  The
    contract-value group, and any group the config does not declare, registers
    every case as before: its register feeds the state shares the solves bid on,
    the institution comparison and the Exchequer ledger.
    """
    from scripts.public_ownership.reference_runs import group_balancing_bids, group_market_cases, group_tariff_cases

    groups = block.get("reference_run_groups") or {}
    if not run_group or run_group not in groups or group_balancing_bids(groups[run_group]) == BIDS_CONTRACT_VALUE:
        return list(specs)
    settings_by_case = {spec.case_id: spec.settings for spec in specs}
    own = {PRIVATE_SCENARIO, *group_market_cases(block, run_group), *group_tariff_cases(block, run_group)}
    while True:
        wider = own | {comparator_for_settings(case, settings_by_case) for case in own}
        if wider == own:
            break
        own = wider
    unknown = sorted(own - set(settings_by_case))
    if unknown:
        raise ValueError(f"run group {run_group!r} names cases the config does not expand: {unknown[:5]}")
    return [spec for spec in specs if spec.case_id in own]


def require_study_config(block: Mapping, run_group: str | None, model_year) -> None:
    """FES 2050 guard (user, 3 Oct 2026): a vintaged (2040 on) register needs its study's resolved config.

    Handed the defaults (``config/defaults/``) instead, a 2050 group is not in ``reference_run_groups`` and
    would fall back silently to the global (2030) case list and settings. Refuse it. A 2030
    register is never checked.
    """
    from scripts.public_ownership.model_year import is_vintaged_model_year

    if model_year is None or not is_vintaged_model_year(int(model_year)):
        return
    groups = (block or {}).get("reference_run_groups") or {}
    if not run_group or run_group not in groups:
        raise ValueError(f"a {int(model_year)} register (run group {run_group!r}) was handed a config without that "
                         "group: pass the study's resolved config (resources/public_ownership/<study>_finance_config.yaml, "
                         "study_config.py), not the defaults (config/defaults/)")


#: F20 phase 0: the per-key steps timed when PYPSA_GB_STAGE_TIMING is set (scripts/public_ownership/stage_timing.py):
#: the account and tiling per register key, the LDES floor booking and the per-key totals.
TIMED_FUNCTIONS = ("calculate_financing_scenarios", "plan_acquisition", "generator_charges", "_comparator_rows",
                   "_key_kinds", "_key_account", "_key_frame", "_records_column", "_attach_lever_columns",
                   "book_carbon_removal_cfd", "book_rab_allowed_revenue", "book_support", "support_lines_by_asset",
                   "_book_ldes_floor_to_rows", "_case_totals")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_register")
    parser.add_argument("cost_registry")
    parser.add_argument("finance_snapshot")
    parser.add_argument("output_register")
    parser.add_argument("output_summary")
    parser.add_argument("--annual-market-summary")
    parser.add_argument(
        "--settlement", default=None,
        help="WP-D per-asset settlement CSV; when given, revenue and fuel columns are attached to every cohort row",
    )
    parser.add_argument(
        "--capacity-market-revenue", default=None,
        help="calendar-year CM revenue per kW by technology (item MR4; the builder's model year); adds capacity_market_revenue_gbp",
    )
    parser.add_argument(
        "--capacity-market-rebasing", nargs=3, default=None, metavar=("FROM_YEAR", "TO_YEAR", "FACTOR"),
        help="the bill's currency_rebasing (from_price_year to_price_year factor): the CM rates are in "
             "CM_RATES_PRICE_YEAR money and are divided by FACTOR onto the register's price year",
    )
    parser.add_argument(
        "--price-year", type=int, default=None,
        help="Enforce this price year on the cost registry (config public_ownership.price_year)",
    )
    parser.add_argument("--config", required=True,
                        help="config file whose public_ownership.scenarios and .sensitivities declare the case set (item K1)")
    parser.add_argument("--price-level-settlement", default=None,
                        help="F19 P3 decision 2: the asset settlement at CP30's price level "
                             "(price_level_settlement.py); a case acquiring the legacy fleet on "
                             "legacy_price_path cp30_level values its years after the model year on it")
    parser.add_argument("--price-anchors", default=None,
                        help="legacy_price_anchors.csv (price_anchors.py; user, 10 Nov 2026): legacy_price_path "
                             "anchored runs each technology's capture price through today and every solved year")
    parser.add_argument("--run-group", default=None,
                        help="the run group this register is for (default: read off the register's file name); a "
                             "balancing group registers only its own cases and their comparators")
    parser.add_argument("--rate-path-offsets", default=None,
                        help="FES 2050 decision 54: the study's rate-path strike offsets (contract_book_2050.py strike-table)")
    parser.add_argument("--acquisition-baseline", default=None,
                        help="D1 (14 Nov 2026): the acquisition group's bought-capacity table (a later model year)")
    parser.add_argument("--held-acquisition-register", default=None,
                        help="network bounds (18-19 Oct 2026): the register (purchase_price_from's) whose legacy "
                             "acquisition prices this group's acquiring cases hold")
    parser.add_argument("--comparator-reference-register", default=None,
                        help="F27 item 8 (23 Oct 2026): a bid-stack group's market-group register; the forgone "
                             "corporation tax judges each case's comparator on its rows (the market months)")
    parser.add_argument("--support-reference", default=None,
                        help="FES 2050 decisions 58/63: a balancing group's market-group register; its support "
                             "lines are held at that register's private per-asset amounts")
    args = parser.parse_args()
    # F20 phase 0: opt-in stage timing (PYPSA_GB_STAGE_TIMING); without it nothing is wrapped.
    from scripts.public_ownership import public_account as _public_account, stage_timing

    stage_timing.instrument_register_io()
    stage_timing.instrument(globals(), TIMED_FUNCTIONS, prefix="financing")
    stage_timing.instrument(_public_account, ("ldes_floor_topups", "_floor_parts", "_floor_parts_original"),
                            prefix="public_account")
    lap = stage_timing.laps("financing.main")
    from scripts.public_ownership.reference_runs import configured_case_specs, configured_group

    assets = pd.read_csv(args.asset_register, low_memory=False)
    if args.settlement:
        from scripts.public_ownership.settlement import attach_settlement

        assets = attach_settlement(assets, pd.read_csv(args.settlement))
    if args.capacity_market_revenue:
        from scripts.market.capacity_market import attach_capacity_market_revenue

        rates = capacity_market_rates_in_price_year(
            pd.read_csv(args.capacity_market_revenue), args.price_year, args.capacity_market_rebasing)
        assets = attach_capacity_market_revenue(assets, rates)
    costs = pd.read_csv(args.cost_registry)
    with Path(args.finance_snapshot).open(encoding="utf-8") as handle:
        snapshot = yaml.safe_load(handle)
    demand_mwh = None
    if args.annual_market_summary:
        with Path(args.annual_market_summary).open(encoding="utf-8") as handle:
            demand_mwh = json.load(handle).get("demand_MWh")
    block = (read_config(args.config) or {}).get("public_ownership") or {}
    # F29: the rates are workbook v2.0's (schema 4); an older snapshot is refused.
    require_snapshot_schema(snapshot)
    run_group = args.run_group or run_group_from_register_path(args.output_register)
    # Task Y14: a group with its own case list (sensitivities / sensitivities_from) registers
    # that list; every other group, and a register named for no configured group, the global one.
    require_study_config(block, run_group, frame_model_year(assets))
    specs = configured_case_specs(block, configured_group(block, run_group))
    keep = run_group_case_specs(block, run_group, specs)
    # Task Y10: a vintaged register (2040 on) is costed by build year from Y2's table.
    vintage = vintage_options_for(block, run_group, frame_model_year(assets))
    acquisition = block.get("acquisition") or {}
    if acquisition.get("valuation"):
        # FES 2050 decision 62: a register bound's own repowering rule prices its legacy rows too
        # (unchanged without a valuation repowering rule, every 2030 run).
        from scripts.public_ownership.legacy_valuation import group_repowering_settings

        adjusted = group_repowering_settings(acquisition["valuation"], block, run_group)
        # F27 item 13: gas balancing margin on the NESO constraint-cost series of the group's network.
        from scripts.public_ownership.legacy_valuation import group_balancing_path_settings

        adjusted = group_balancing_path_settings(adjusted, block, run_group)
        if adjusted != dict(acquisition["valuation"]):
            acquisition = {**acquisition, "valuation": adjusted}
    if args.price_level_settlement:
        acquisition = {**acquisition, "price_level_settlement": pd.read_csv(args.price_level_settlement, low_memory=False)}
    if getattr(args, "price_anchors", None):
        acquisition = {**acquisition, "price_anchors": pd.read_csv(args.price_anchors)}
    if args.acquisition_baseline:
        # D1 (user, 14 Nov 2026): a later year owns exactly what its acquisition group's purchase bought.
        acquisition = {**acquisition, "carried_baseline": str(Path(args.acquisition_baseline).resolve())}
    if not args.price_level_settlement and not is_vintaged_model_year(int(frame_model_year(assets) or 0)) and any(
            spec.acquires_legacy() and spec.settings.get(LEVER_LEGACY_PRICE_PATH) == PRICE_PATH_CP30 for spec in specs):
        # A group with no consumer-bill months (the historical groups) has no CP30-level
        # settlement: its plans fall back to the model price and say so
        # (legacy_price_path_applied in the acquisition summary).
        print("warning: legacy_price_path cp30_level without --price-level-settlement; the legacy valuation "
              "runs on the model year's price (legacy_price_path_applied = model_2030)", file=sys.stderr)
    ro_stations = None
    if any(spec.acquires_legacy() for spec in specs):
        ro_stations = pd.read_csv(
            valuation_settings(acquisition.get("valuation"))["ro_register"], low_memory=False)
    from scripts.public_ownership.register_io import RegisterWriter

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    # 29 Sep 2026: Parquet plus a case-settings table (register_io); a .csv path keeps the old format.
    # 30 Sep 2026: written one register key at a time, so the register is never held whole.
    options = _with_support_reference(
        _with_rate_path_offsets(_with_vintage_learning(register_options_from_config(block), block, run_group),
                                args.rate_path_offsets),
        args.support_reference)
    if options.get("capacity_market_design") is not None:
        # Decision 73: the design's auction amounts are in the CM rates' money (2025), carried onto the
        # register's price year by the same factor as the rates (--capacity-market-rebasing).
        rebased = args.price_year is not None and int(args.price_year) != CM_RATES_PRICE_YEAR
        if rebased and not args.capacity_market_rebasing:
            raise ValueError("the Capacity Market design needs --capacity-market-rebasing (decision 73)")
        options["capacity_market_design"]["rebasing_factor"] = (
            float(args.capacity_market_rebasing[2]) if rebased else 1.0)
    group_co = (((block.get("reference_run_groups") or {}).get(run_group) or {}).get("co_ownership_finance")
                if run_group else None)
    if group_co:
        # Decision 68: a register bound's own co-ownership financing (the sensitivities) over the study's.
        options["co_ownership"] = co_ownership_settings({"co_ownership_finance": group_co})
    held_prices = None
    if args.held_acquisition_register:
        from scripts.public_ownership.register_io import read_register

        acquiring_ids = [spec.case_id for spec in specs if spec.acquires_legacy()]
        held_prices = read_register(args.held_acquisition_register, columns=list(HELD_ACQUISITION_COLUMNS),
                                    cases=acquiring_ids) if acquiring_ids else None
        if held_prices is not None:
            held_prices = held_prices.loc[held_prices["finance_vintage"].astype(str).eq("legacy")]
            missing = sorted(set(acquiring_ids) - set(held_prices["case_id"].astype(str)))
            if missing:
                raise ValueError(f"{args.held_acquisition_register} lacks acquiring cases {missing[:5]}")
    comparator_reference = None
    if args.comparator_reference_register:
        from scripts.public_ownership.register_io import read_register

        def comparator_reference(case_id: str, path=args.comparator_reference_register) -> pd.DataFrame:
            frame = read_register(path, cases=[case_id], columns=list(COMPARATOR_REFERENCE_COLUMNS))
            if frame.empty:
                raise ValueError(f"{path} lacks the comparator case {case_id!r}")
            return frame
    lap("setup")
    with RegisterWriter(args.output_register) as writer:
        _, summary = calculate_financing_scenarios(
            assets, costs, snapshot, case_specs=specs, demand_mwh=demand_mwh,
            price_year=args.price_year, acquisition=acquisition, ro_stations=ro_stations,
            state_equity_share_value=block.get("state_stake_share"),
            state_operating_overhead=block.get("state_operating_overhead"),
            corporation_tax_basis_block=block.get("corporation_tax_basis"),
            register_sink=writer.write,
            register_cases=None if len(keep) == len(specs) else [spec.case_id for spec in keep],
            vintage_costs=vintage,
            held_acquisition_prices=held_prices,
            held_acquisition_source=args.held_acquisition_register,
            comparator_reference=comparator_reference,
            **options,
        )
    lap("register")
    summary.to_csv(args.output_summary, index=False)
    lap("write_summary")


if __name__ == "__main__":
    main()
