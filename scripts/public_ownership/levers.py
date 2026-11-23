"""Levers, scenario bundles and the sensitivity expansion every stage shares (item K1).

A scenario is a named bundle of lever settings and nothing else.  No script may derive
a treatment from a case name: the ownership share, the cost of capital, the
acquisition of the legacy fleet, the recycling rule and every other treatment is
read from the lever columns the register carries, which are written from the
:class:`CaseSpec` this module expands.

Three kinds of case come out of :func:`expand_cases`:

* the **scenarios** themselves (``private``, ``public_control``,
  ``public_ownership`` in the repository config), each a full bundle;
* **one-at-a-time sensitivities**, ``<scenario>@<label>``: one lever moved off
  its bundle value, generated only for the scenarios where the lever bites
  (a lever's ``requires`` says when it does), plus a bounding case per public
  scenario with every *structural* lever on its first alternative at once;
* **named sensitivities**, ``<scenario>@<name>``: a declared multi-lever
  position (the publicly owned company, full nationalisation), allowed because
  each is a position from the workplan or the workbook rather than a grid point;
* **presets** (review 29 Sep 2026): one multi-lever *incidence* position (who
  pays) applied to several propositions, ``<parent>@<preset>`` or
  ``<parent>_<preset>``, with per-parent values where a preset needs them
  (``levers_by_case``, 30 Sep 2026: ``todays_rules`` is each scenario's own old
  incidence); they move only :data:`INCIDENCE_LEVERS`, so each leaves its
  parent's resource saving unchanged (the three-party ledger enforces it);
* **bounds**: the best and worst coherent *assumption* position of a case
  (``<parent>@best_case`` / ``_worst_case``), under its own incidence;
* **toggle grids**: every combination of every axis of a parent
  (``<parent>@grid_<piece>_<piece>``), for the explorer.  Since 30 Sep 2026 an
  axis is a binary switch, a multi-valued selector (``networks``, the strike
  basis) or a switch conditional on a selector option (the guarantee's
  pass-through only under the guarantee); each case carries its position vector
  in :func:`case_catalogue` and a position another case already accounts is that
  case's grid membership.

Since 30 Sep 2026 (the user's rebase) every public scenario's base is "bill-payers
first" (the retired ``consumer_first`` preset) with the wires private; the old
central incidence is the ``todays_rules`` preset and the wires are a grid axis.

Lever values are enforced, not documented (house rule 11): only the values in
:data:`LEVERS` are implemented, and anything else fails at Snakemake parse time
through :func:`scripts.public_ownership.reference_runs.validate_public_ownership_config`.

Three different things are called ``at_cost`` and they are not the same switch:
``public_cfd_at_cost`` is the surplus-recycling rule; ``cfd_strike_basis:
at_cost`` re-prices an owned CfD to the asset's cost; ``new_build_contracts:
at_cost`` treats merchant new build as contracted at required revenue in every
case.  The labels keep them apart.
"""

from __future__ import annotations

import itertools
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


VARIANT_SEPARATOR = "@"
BOUNDING_PAIR_LABEL = "all_alternatives"
PRIVATE_SCENARIO = "private"

# Which cases a lever can move.  ``all_cases`` includes the private baseline and
# so changes the comparator rather than the ownership effect (house rule 3).
SCOPE_PUBLIC = "public_cases"
SCOPE_ALL = "all_cases"
SCOPES = frozenset({SCOPE_PUBLIC, SCOPE_ALL})

# ── lever names ───────────────────────────────────────────────────────────────
# 1. Who owns what
LEVER_NEW_BUILD_GENERATION = "new_build_generation"
LEVER_PRE_2025_CFD = "pre_2025_cfd_capacity"
LEVER_LEGACY_FLEET = "legacy_fleet"
LEVER_NETWORKS = "networks"
LEVER_NEW_BUILD_STORAGE = "new_build_storage"
LEVER_IN_CONSTRUCTION = "in_construction"
#: User decision (10 Nov 2026): energy-from-waste plants left out of the purchase by default.
LEVER_EFW_ACQUISITION = "efw_acquisition"
# 2. Cost of capital
LEVER_PRIVATE_WACC_BASIS = "private_wacc_basis"
LEVER_PUBLIC_RETURN = "public_return"
LEVER_PUBLIC_RATE_BASIS = "public_rate_basis"
LEVER_CONSTRUCTION_PREMIUM = "construction_premium"
#: F29 (user, 26 Oct 2026): replaces construction_premium; moves the economic rate, never the bill.
LEVER_SYSTEMATIC_PREMIUM = "systematic_premium"
LEVER_SOVEREIGN_YIELD = "sovereign_yield"
LEVER_GUARANTEE_COST = "guarantee_cost"
# 3. Acquisition terms
LEVER_PURCHASE_PRICE_BASIS = "purchase_price_basis"
LEVER_THERMAL_BASIS = "thermal_basis"
LEVER_LEGACY_PRICE_PATH = "legacy_price_path"
LEVER_OWNER_DISCOUNT_RATE = "legacy_owner_discount_rate"
# User decision (24 Oct 2026): legacy offshore wind at its owners' value, or at observed deal multiples.
LEVER_LEGACY_OFFSHORE_PRICE = "legacy_offshore_price"
OFFSHORE_OWNERS_VALUE, OFFSHORE_DEAL_MULTIPLE = "owners_value", "transaction_multiple"
LEVER_ACQUISITION_TERM = "acquisition_term_years"
#: F28: the acquisition term value that repays each asset over its own remaining life.
TERM_REMAINING_LIFE = "remaining_life"
LEVER_LEGACY_CONTRACTS = "legacy_contracts_under_acquisition"
# 4. How public plant is paid
LEVER_CFD_STRIKE_BASIS = "cfd_strike_basis"
LEVER_CFD_AMORTISATION = "cfd_amortisation"
LEVER_SURPLUS_RECYCLING = "surplus_recycling"
LEVER_SHORTFALL_FUNDING = "shortfall_funding"
LEVER_ROUTE_BOOKING = "route_booking"
LEVER_NEW_BUILD_CONTRACTS = "new_build_contracts"
LEVER_LDES_FLOOR_BASIS = "ldes_floor_basis"
LEVER_STATE_OVERHEAD = "state_overhead"
# 5. Market and institution
LEVER_WHOLESALE_PRICING = "wholesale_pricing"
LEVER_DISPATCH_INSTITUTION = "dispatch_institution"
LEVER_BALANCING_BIDS = "balancing_bids"
# 6. Fiscal incidence: who pays, never how much (user decisions 3 and 5, 29 Sep 2026)
LEVER_STATE_DEFICIT_FUNDING = "state_deficit_funding"
LEVER_TAX_EQUIVALENT = "tax_equivalent"
LEVER_NETWORK_INCIDENCE = "network_incidence"
LEVER_ACQUISITION_FINANCING = "acquisition_financing"
LEVER_TARIFF_PRICING = "tariff_pricing"
# 7. Background: moves every case alike, private included (a change of comparator)
LEVER_LEGACY_LEVY_FUNDING = "legacy_levy_funding"
LEVER_BECCS_REMOVAL_FUNDING = "beccs_removal_funding"
# Network assumptions (user decisions 6 and 7)
LEVER_GUARANTEE_PASS_THROUGH = "guarantee_pass_through"
LEVER_NETWORK_PREMIUM = "network_acquisition_premium"
# 8. Assumptions the explorer's toggle grids and the bounds move (29 Sep 2026)
LEVER_STATE_STAKE_SCALE = "state_stake_scale"
LEVER_DELIVERED_SHARE = "new_build_delivered_share"
LEVER_STATE_CAPEX_OVERRUN = "state_capex_overrun"
LEVER_STATE_OM = "state_om_premium"
LEVER_GENERATION_CONTROL_PREMIUM = "generation_control_premium"
LEVER_INTERCONNECTOR_PRICE = "interconnector_price"
LEVER_TRANSACTION_COSTS = "acquisition_transaction_costs"
LEVER_ASSET_LIVES = "legacy_asset_lives"
#: Retired 29 Sep 2026 (review C1): structurally inert, its two cases reproduced P1
#: and P2 to the penny.  Kept as a name so older registers and code read cleanly;
#: commissioning vintage is reached through ``pre_2025_cfd_capacity: public_repriced``.
LEVER_CFD_ELIGIBILITY = "cfd_repricing_eligibility"

# Backwards-compatible names: the six structural axes of item D1 kept their names.
AXIS_OWNERSHIP_BOUNDARY = LEVER_PRE_2025_CFD
AXIS_CFD_STRIKE_BASIS = LEVER_CFD_STRIKE_BASIS
AXIS_ROUTE_BOOKING = LEVER_ROUTE_BOOKING
AXIS_NEW_BUILD_CONTRACTS = LEVER_NEW_BUILD_CONTRACTS
AXIS_LDES_FLOOR_BASIS = LEVER_LDES_FLOOR_BASIS
AXIS_CFD_AMORTISATION = LEVER_CFD_AMORTISATION

# Values referred to by name elsewhere.
NEW_BUILD_PRIVATE, NEW_BUILD_STATE_STAKE, NEW_BUILD_PUBLIC = "private", "state_stake", "public"
# F10 (user, 6 Oct 2026): GB Energy as announced. A stake in the post-2025 offshore wind cohort
# (the Crown Estate partnership, P1's stake mechanics) and the Local Power Plan's local solar and
# onshore wind owned at the public rate (P2's mechanics); everything else private. The partnership
# value holds the local capacity at a public share (lever_parameters.gbe_announced).
NEW_BUILD_GBE, NEW_BUILD_GBE_PARTNERSHIP = "gbe_announced", "gbe_announced_partnership_half"
GBE_VALUES = frozenset({NEW_BUILD_GBE, NEW_BUILD_GBE_PARTNERSHIP})
PRE_2025_PRIVATE, PRE_2025_PUBLIC_REPRICED = "private", "public_repriced"
#: F27 item 1 (user, 20 Oct 2026): the same plant bought at its owners' value (cost plus the present value
#: of their rent at their own rate), beside the at-cost version, so the gap shows.
PRE_2025_PUBLIC_OWNERS_VALUE = "public_bought_owners_value"
PRE_2025_STATE_OWNED = frozenset({PRE_2025_PUBLIC_REPRICED, PRE_2025_PUBLIC_OWNERS_VALUE})
#: The register column (user, 27 Oct 2026) moving the co-owners' pro rata part of the owners'-value premium
#: onto the state's share, ``-(1 - s) x premium`` (calculate_financing_scenarios writes it; the public
#: account, the at-cost re-pricing and the party flows read it).
OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT = "owners_value_premium_state_adjustment_gbp"
#: F27 item 1 (user, 24 Oct 2026: "Net Zero Teesside joins the option"): pre-2025 contracts that are not
#: CfDs, by contract id (the register's ``cfd_round``), with the year their 15-year term starts. Net Zero
#: Teesside Power's Dispatchable Power Agreement was signed in December 2024 (start-up targeted 2028,
#: data/market/CP30/committed_overlay_2030.csv). It follows ``pre_2025_cfd_capacity`` like a pre-2025 CfD
#: in the market-bill cases; its payments are the bill's ``remaining_ccs`` line.
PRE_2025_DPA_CONTRACTS = {"NZT_POWER_DPA": 2028}
#: Its term (NZT Power DPA: expiry at the 15th anniversary of the start; data/market/support/support_regime_2050.json).
PRE_2025_DPA_TERM_YEARS = 15


def is_pre_2025_dpa(support_type, cfd_round, model_year) -> bool:
    """F27 item 1 (24 Oct 2026): is a register row on a pre-2025 DPA (:data:`PRE_2025_DPA_CONTRACTS`)?

    Only while the agreement is live: from its start to the end of its
    :data:`PRE_2025_DPA_TERM_YEARS` term (NZT: 2028 to 2043). So 2030 and 2040 (11 Nov 2026) carry it and
    2050 does not, where a DPA is a modelled later agreement (2050 does not move).
    """
    if str(support_type or "").strip().upper() != "DPA" or str(cfd_round or "") not in PRE_2025_DPA_CONTRACTS:
        return False
    try:
        year = float(model_year)
    except (TypeError, ValueError):
        return True
    if not (year == year and abs(year) != float("inf")):
        return True
    return int(year) < PRE_2025_DPA_CONTRACTS[str(cfd_round)] + PRE_2025_DPA_TERM_YEARS


LEGACY_PRIVATE, LEGACY_ACQUIRE_VALUED, LEGACY_ACQUIRE_ALL = "private", "acquire_valued_group", "acquire_all"
NETWORKS_PRIVATE, NETWORKS_DEBT_GUARANTEE, NETWORKS_ACQUIRED = "private", "debt_guarantee", "acquired"
STORAGE_FOLLOWS, STORAGE_PRIVATE = "follows_generation", "private"
IN_CONSTRUCTION_PRIVATE, IN_CONSTRUCTION_PUBLIC = "private", "public_completion"
#: User decision (10 Nov 2026): Hinkley Point C left out of the purchase, private on its own CfD.
IN_CONSTRUCTION_EXCLUDED = "left_private"
EFW_EXCLUDED, EFW_INCLUDED = "excluded", "included"
#: v6.2 (user, 30 Oct 2026): the lead rate is tax-shielded at a notional 55% gearing for every
#: technology (RIIO-3); ``observed_gearing_shielded`` (v2.0's technology-specific gearing) is retired.
WACC_NOTIONAL_GEARING, WACC_CEPA_VANILLA, WACC_UNIFORM_GEARING = (
    "notional_gearing_shielded", "cepa_vanilla", "uniform_gearing_shielded")
WACC_OBSERVED_GEARING = "observed_gearing_shielded"  # retired value, kept so a stale declaration fails with a pointer
#: F29 (user, 26 Oct 2026): the workbook's co-equal ERP convention, Fixed TMR (CEPA / UKRN).
WACC_FIXED_TMR = "fixed_tmr_shielded"
RETURN_PUBLIC_RATE, RETURN_CEPA_DECOMPOSITION, RETURN_COMMERCIAL_BENCHMARK = (
    "public_rate", "cepa_decomposition", "commercial_benchmark")
#: Retired 3 Oct 2026 (OW9): the three ``explicit`` values, kept so a stale declaration
#: fails with a pointer (:data:`RETIRED_VALUES`).
RETURN_EXPLICIT = PRICE_EXPLICIT = "explicit"
#: User decision 9 (29 Sep 2026): the two real-rate stresses replace ``nominal``,
#: which applied 4.8% nominal to real cash flows (a unit mix the workbook itself
#: rules out: C28 is "on REAL basis").
RATE_BASIS_REAL, RATE_BASIS_PLUS_1PP, RATE_BASIS_GILT_30Y = "real_cpi", "real_plus_1pp", "gilt_30y_real"
RATE_BASIS_NOMINAL = "nominal"  # retired value, kept so a stale declaration fails with a pointer
#: F29 (user, 26 Oct 2026): the workbook's public lending margins over the real gilt (Sc4 Operative B5).
#: ``real_cpi`` is the central, the NLF margin (+11 bp); the values map to the snapshot's
#: ``public_rate_by_lending_margin`` keys.
RATE_BASIS_PWLB_HRA, RATE_BASIS_PWLB_CERTAINTY, RATE_BASIS_PWLB_STANDARD, RATE_BASIS_GILT_FLAT = (
    "pwlb_hra", "pwlb_certainty", "pwlb_standard", "gilt_flat")
RATE_BASIS_MARGIN_KEYS = {RATE_BASIS_REAL: "nlf", RATE_BASIS_PWLB_HRA: "pwlb_hra",
                          RATE_BASIS_PWLB_CERTAINTY: "pwlb_certainty", RATE_BASIS_PWLB_STANDARD: "pwlb_standard",
                          RATE_BASIS_GILT_FLAT: "gilt_flat"}
PREMIUM_GRADED, PREMIUM_FLAT, PREMIUM_NONE = "graded", "flat_50bps", "none"
#: F29: the price of the systematic risk the state holds (Sc4 Operative B6, B7, B10).  v6.2 (user,
#: 30 Oct 2026): a flat 1.9% central; none, a flat 1%, 2% per unit of CEPA asset beta and the market
#: ERP per unit of beta the alternatives.  Values map to the snapshot's ``economic_rate_by_premium`` keys.
SYSTEMATIC_FLAT_19, SYSTEMATIC_NONE, SYSTEMATIC_FLAT_10, SYSTEMATIC_BETA_20, SYSTEMATIC_MARKET_ERP = (
    "social_flat_19", "none", "social_flat_10", "social_beta_20", "market_erp")
#: Retired with v6.2 (kept so a stale declaration fails with a pointer).
SYSTEMATIC_SOCIAL_P20, SYSTEMATIC_SOCIAL_P15, SYSTEMATIC_SOCIAL_P30 = "social_p20", "social_p15", "social_p30"
YIELD_A, YIELD_B, YIELD_C = "A", "B", "C"
OVERHEAD_CENTRAL, OVERHEAD_LOW, OVERHEAD_HIGH = "central", "low", "high"
GUARANTEE_COST_CENTRAL, GUARANTEE_COST_LOW, GUARANTEE_COST_HIGH = "central", "low", "high"
PRICING_PAY_AS_CLEAR, PRICING_PAY_AS_BID_CEILING = "pay_as_clear", "pay_as_bid_ceiling"
INSTITUTION_MARKET, INSTITUTION_NATIONALISED_DISPATCH = "market", "nationalised_dispatch"
BIDS_CONTRACT_VALUE, BIDS_STATE_AT_COST = "contract_value", "state_at_cost"
CONTRACTS_HONOURED, CONTRACTS_CANCELLED_BILLS, CONTRACTS_CANCELLED_EXCHEQUER = (
    "honoured", "cancelled_residual_on_bills", "cancelled_residual_on_exchequer")
STRIKE_FINANCING_SAVING, STRIKE_AT_COST, STRIKE_AUCTION_CLEARING = "financing_saving_only", "at_cost", "auction_clearing"
RECYCLING_REBATE_GROSS = "rebate_gross"
RECYCLING_OFFSET_POLICY_LEVIES = "offset_policy_levies"
DEFICITS_EXCHEQUER, DEFICITS_BILL_LEVY = "exchequer", "bill_levy"
TAX_EQUIVALENT_NONE, TAX_EQUIVALENT_ON = "none", "on"
NETWORK_INCIDENCE_BILLS, NETWORK_INCIDENCE_FEE, NETWORK_INCIDENCE_RETAINED = (
    "bills", "guarantee_fee_on_bills", "return_retained")
ACQUISITION_ON_TARIFF, ACQUISITION_EXCHEQUER = "tariff", "exchequer"
TARIFF_AT_COST, TARIFF_AT_MARKET_BILL = "cost", "market_bill"
LEGACY_LEVIES_ON_BILLS, LEGACY_LEVIES_TO_TAXATION = "bills", "domestic_to_taxation"
BECCS_REMOVAL_EXCHEQUER, BECCS_REMOVAL_BILL_LEVY = "exchequer", "bill_levy"
PASS_THROUGH_NEW_DEBT, PASS_THROUGH_NEW_DEBT_LOW, PASS_THROUGH_FULL_STOCK = (
    "new_and_refinanced", "new_debt_low", "full_stock")
RAV_PREMIUM_CENTRAL, RAV_PREMIUM_LOW, RAV_PREMIUM_HIGH, RAV_PREMIUM_ZERO = "central", "low", "high", "statutory_at_rav"
#: User request (2 Oct 2026): a 25% premium over RAV, between the listed-market low and the central.
RAV_PREMIUM_QUARTER = "quarter"
STAKE_WORKBOOK, STAKE_FUNDED_SCALE = "workbook_stake", "funded_scale"
DELIVERED_ALL, DELIVERED_HALF = "all", "half"
OVERRUN_NONE, OVERRUN_PLUS_15 = "none", "plus_15pct"
STATE_OM_SAME, STATE_OM_PLUS_10 = "same", "plus_10pct"
CONTROL_PREMIUM_NONE, CONTROL_PREMIUM_20 = "none", "twenty_pct"
IC_PRICE_CENTRAL, IC_PRICE_LOW, IC_PRICE_HIGH = "central", "low", "high"
TRANSACTION_NONE, TRANSACTION_TWO_PCT = "none", "two_pct"
LIVES_CENTRAL, LIVES_EXTENDED = "central", "extended"
THERMAL_BOOK_VALUE, THERMAL_MARKET_VALUE = "book_value", "market_value"
#: F29 (user, 26 Oct 2026): the workbook's acquisition bases (Legacy A1/A2 tabs) on every acquired row:
#: A1 book value, A2 market value (``market_value``), A2 without support, A3 RO cohorts at no-support value.
PRICE_STATE_PV, PRICE_MARKET_VALUE = "state_present_value", "market_value"
PRICE_BOOK_VALUE, PRICE_NO_SUPPORT, PRICE_RO_PASS_THROUGH = "book_value", "no_support", "ro_pass_through"
PRICE_WORKBOOK_PLACEHOLDER = "workbook_placeholder"  # retired by F29, kept so a stale declaration fails with a pointer
#: F19 P3 decision 2 (user, 6 Oct 2026): the legacy valuation's price after the model year.
PRICE_PATH_CP30, PRICE_PATH_MODEL = "cp30_level", "model_2030"
#: User decision (10 Nov 2026): the central path runs through today's forward level and every solved year.
PRICE_PATH_ANCHORED = "anchored"
#: User, 11 Nov 2026: today's forward prices held for the fleet's life (a named P3 sensitivity).
PRICE_PATH_TODAY = "today_level"
#: F19 (user, 6 Oct 2026): the rate the owners' value discounts operating renewables at.  Retired by
#: F29 (26 Oct 2026): every row is discounted at the workbook's post-tax operating rate.
OWNER_RATE_OPERATING, OWNER_RATE_HURDLE = "operating_asset", "new_build_hurdle"

_PUBLIC_NEW_BUILD = frozenset({NEW_BUILD_STATE_STAKE, NEW_BUILD_PUBLIC, *GBE_VALUES})
#: Values under which the state develops plant itself at the public rate (P2's financing levers bite).
_STATE_DEVELOPER = frozenset({NEW_BUILD_PUBLIC, *GBE_VALUES})


@dataclass(frozen=True)
class Lever:
    """One declared lever: what it may take, who it applies to, how it is labelled.

    ``values[0]`` is the private baseline's value and the default for every
    bundle that does not name the lever.  ``requires`` maps other levers to the
    values a bundle must hold for a sensitivity on this lever to be generated;
    ``requires_by_value`` does the same for one value only.  ``register`` marks
    the levers that change the required-revenue account itself: two cases that
    agree on every register lever share every register number, so the account
    is computed once for them.
    """

    name: str
    values: tuple[str, ...]
    scope: str
    #: value -> variant label.  A label must be unique across every lever,
    #: because the case id carries the label alone.
    labels: Mapping[str, str] = field(default_factory=dict)
    register: bool = False
    requires: Mapping[str, frozenset] = field(default_factory=dict)
    requires_by_value: Mapping[str, Mapping[str, frozenset]] = field(default_factory=dict)
    description: str = ""

    @property
    def default(self) -> str:
        return self.values[0]

    def label(self, value: str) -> str:
        return self.labels.get(value, value)

    def applies(self, value: str, bundle: Mapping[str, str]) -> bool:
        """Can moving this lever to ``value`` change a number in ``bundle``?"""
        if bundle.get(self.name, self.default) == value:
            return False
        if self.scope == SCOPE_PUBLIC and bundle.get(LEVER_NEW_BUILD_GENERATION, NEW_BUILD_PRIVATE) == NEW_BUILD_PRIVATE:
            return False
        for lever, allowed in self.requires.items():
            if bundle.get(lever, ALL_LEVERS[lever].default) not in allowed:
                return False
        for lever, allowed in self.requires_by_value.get(value, {}).items():
            if bundle.get(lever, ALL_LEVERS[lever].default) not in allowed:
                return False
        return True


def _lever(name, values, scope, *, labels=None, register=False, requires=None, requires_by_value=None, description=""):
    return Lever(
        name=name, values=tuple(values), scope=scope, labels=dict(labels or {}), register=register,
        requires={k: frozenset(v) for k, v in (requires or {}).items()},
        requires_by_value={value: {k: frozenset(v) for k, v in reqs.items()}
                           for value, reqs in (requires_by_value or {}).items()},
        description=description,
    )


_REQUIRES_PUBLIC_NEW_BUILD = {LEVER_NEW_BUILD_GENERATION: _PUBLIC_NEW_BUILD}
_REQUIRES_FULL_NEW_BUILD = {LEVER_NEW_BUILD_GENERATION: {NEW_BUILD_PUBLIC}}
_REQUIRES_STATE_DEVELOPER = {LEVER_NEW_BUILD_GENERATION: _STATE_DEVELOPER}
_REQUIRES_ACQUISITION = {LEVER_LEGACY_FLEET: {LEGACY_ACQUIRE_VALUED, LEGACY_ACQUIRE_ALL}}

# Follow-up F6 and F7 (user, 4 Oct 2026).
LEVER_AUCTION_MARGIN = "auction_margin"
AUCTION_MARGIN_CENTRAL, AUCTION_MARGIN_NONE, AUCTION_MARGIN_HIGH = "central", "none", "high"
LEVER_SELF_INSURANCE = "self_insurance"
SELF_INSURANCE_CENTRAL, SELF_INSURANCE_NONE, SELF_INSURANCE_HIGH = "central", "none", "high"
# F27 item 6 (user, 20 and 24 Oct 2026): capacity without an auction (scripts/public_ownership/capacity_procurement.py).
LEVER_CAPACITY_PROCUREMENT = "capacity_procurement"
CAPACITY_AUCTION, CAPACITY_COST_BASED_EXISTING = "auction", "cost_based_existing"

LEVERS: dict[str, Lever] = {lever.name: lever for lever in (
    # ── 1. who owns what ────────────────────────────────────────────────────
    _lever(LEVER_NEW_BUILD_GENERATION, (NEW_BUILD_PRIVATE, NEW_BUILD_STATE_STAKE, NEW_BUILD_PUBLIC, NEW_BUILD_GBE,
                                        NEW_BUILD_GBE_PARTNERSHIP), SCOPE_ALL,
           labels={NEW_BUILD_PRIVATE: "new_build_private", NEW_BUILD_STATE_STAKE: "new_build_state_stake",
                   NEW_BUILD_PUBLIC: "new_build_public", NEW_BUILD_GBE: "new_build_gbe_announced",
                   NEW_BUILD_GBE_PARTNERSHIP: "new_build_gbe_partnership_half"},
           register=True,
           description="Who owns the post-2025 build: private; a state stake (share from the snapshot, 30%); the state; "
                       "GB Energy as announced (F10: a 30% stake in post-2025 offshore wind through the Crown Estate "
                       "partnership, on P1's stake mechanics, and 8 GW of local solar and onshore wind at the public "
                       "rate, on P2's; everything else private), or the same with the 8 GW at a 50% public share."),
    _lever(LEVER_PRE_2025_CFD, (PRE_2025_PRIVATE, PRE_2025_PUBLIC_REPRICED, PRE_2025_PUBLIC_OWNERS_VALUE), SCOPE_PUBLIC,
           labels={PRE_2025_PRIVATE: "pre_2025_cfd_private", PRE_2025_PUBLIC_REPRICED: "ownership_all_cfd_repriced",
                   PRE_2025_PUBLIC_OWNERS_VALUE: "ownership_all_cfd_bought_owners_value"},
           register=True,
           description="Capacity whose CfD was awarded before 2025: keeps its private owner and signed strike, or the "
                       "state owns it too and re-prices it, taking it at cost (the owners get their capital, not their "
                       "return above it) or buying it at the owners' value (cost plus the present value of their rent "
                       "at their own rate over the contract's remaining years; F27 item 1). Net Zero Teesside "
                       "Power's DPA (signed December 2024) follows it in the market-bill cases, re-priced on the "
                       "remaining_ccs line."),
    _lever(LEVER_LEGACY_FLEET, (LEGACY_PRIVATE, LEGACY_ACQUIRE_VALUED, LEGACY_ACQUIRE_ALL), SCOPE_PUBLIC,
           labels={LEGACY_PRIVATE: "legacy_private", LEGACY_ACQUIRE_VALUED: "legacy_acquired",
                   LEGACY_ACQUIRE_ALL: "legacy_and_thermal_acquired"},
           register=True, requires=_REQUIRES_FULL_NEW_BUILD,
           description="The legacy fleet: private; renewables, hydro, marine and nuclear bought at their valuation; "
                       "plus thermal and storage."),
    _lever(LEVER_NETWORKS, (NETWORKS_PRIVATE, NETWORKS_ACQUIRED), SCOPE_PUBLIC,
           labels={NETWORKS_PRIVATE: "networks_private", NETWORKS_ACQUIRED: "networks_acquired"},
           # User decision 2-3 (30 Sep 2026): the wires are out of every base and a
           # choice in every scenario, open to any public case (a stake, the state
           # developer, full acquisition of the fleet).  The network account prices each
           # treatment from the RAV alone, whoever owns the generation, so no stage
           # depends on the pairing.  F29 (user, 26 Oct 2026): the debt guarantee is
           # retired; its 75 bp came from v11-2's Cost of Capital sheet, and workbook v2.0
           # carries no network rate.
           requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="The wires: private, or the RAV bought and financed at the public cash rate (item N6)."),
    _lever(LEVER_NEW_BUILD_STORAGE, (STORAGE_FOLLOWS, STORAGE_PRIVATE), SCOPE_PUBLIC,
           labels={STORAGE_FOLLOWS: "storage_follows_generation", STORAGE_PRIVATE: "storage_private"}, register=True, requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="New-build storage follows the generation lever or stays private."),
    # User decision 10 (29 Sep 2026): a real P3 option for Hinkley Point C.  Under
    # ``private`` HPC is bought with the legacy fleet as an operating plant at its
    # CfD value (the purchase_price_basis); under ``public_completion`` the state
    # takes over the construction project, pays the vendors what has been spent,
    # finishes it at the public cash rate (F29: no construction premium; the workbook
    # carries expected overruns as capex), and that completion capital replaces HPC's
    # purchase price (one price, one annuity).
    _lever(LEVER_IN_CONSTRUCTION, (IN_CONSTRUCTION_PRIVATE, IN_CONSTRUCTION_PUBLIC, IN_CONSTRUCTION_EXCLUDED),
           SCOPE_PUBLIC,
           labels={IN_CONSTRUCTION_PRIVATE: "hpc_bought_operating", IN_CONSTRUCTION_PUBLIC: "hpc_public_completion",
                   IN_CONSTRUCTION_EXCLUDED: "hpc_left_private"},
           register=True, requires={**_REQUIRES_ACQUISITION},
           description="Hinkley Point C: bought as an operating plant at its CfD value with the legacy fleet "
                       "(central), completed by the state (sunk cost to the vendors, remaining capex at the public "
                       "cash rate), or left out of the purchase, private on its own CfD (user, 10 Nov 2026)."),
    # User decision (10 Nov 2026): energy-from-waste plants are waste infrastructure whose income is mostly gate
    # fees paid by councils; the purchase leaves them with their (private) owners unless this lever buys them.
    _lever(LEVER_EFW_ACQUISITION, (EFW_EXCLUDED, EFW_INCLUDED), SCOPE_PUBLIC,
           labels={EFW_EXCLUDED: "efw_left_private", EFW_INCLUDED: "efw_acquired"},
           register=True, requires={**_REQUIRES_ACQUISITION},
           description="Energy-from-waste plants in the purchase of the existing fleet: left with their owners "
                       "and their council contracts (central), or bought at their owners' value with the rest "
                       "(their gate fees then credited to the tariff)."),
    # ── 2. cost of capital ──────────────────────────────────────────────────
    # F29 (user, 26 Oct 2026): every rate is the financing workbook's (v6.2 since 30 Oct 2026,
    # GB_Power_Model_v6.2.xlsx, snapshot schema 5, scripts/public_ownership/extract_workbook_v5.py).
    _lever(LEVER_PRIVATE_WACC_BASIS, (WACC_NOTIONAL_GEARING, WACC_FIXED_TMR), SCOPE_ALL,
           labels={WACC_NOTIONAL_GEARING: "wacc_notional_gearing", WACC_FIXED_TMR: "wacc_fixed_tmr"},
           register=True,
           description="The workbook's two co-equal ERP conventions for the private (Sc1) and P1 (Sc2) lead rates, "
                       "tax-shielded at a notional 55% gearing (RIIO-3): Fixed ERP (the default, CEPA's 5.22% "
                       "held, so the refreshed risk-free rate passes through: offshore 8.48%) or Fixed TMR (CEPA / "
                       "UKRN: total market return held at 6.83%, the premium the residual, 4.23%: offshore 7.61%)."),
    _lever(LEVER_PUBLIC_RETURN, (RETURN_PUBLIC_RATE, RETURN_CEPA_DECOMPOSITION), SCOPE_PUBLIC,
           labels={RETURN_PUBLIC_RATE: "return_public_rate", RETURN_CEPA_DECOMPOSITION: "return_cepa_decomposition"},
           register=True,
           description="What state-financed new build earns: the public cash rate (real gilt plus the lending "
                       "margin, the workbook's Sc3/Sc4 rate, no construction premium), or the workbook's Sc2 rate "
                       "(a 30% state equity stake at the risk-free rate and a guarantee lifting the debt two "
                       "notches on the iBoxx ladder)."),
    # F29: the lending margin moves the state's cash rate; P1's Sc2 rate prices the state's
    # stake at the risk-free rate, so the margin never reaches a stake.
    _lever(LEVER_PUBLIC_RATE_BASIS, (RATE_BASIS_REAL, RATE_BASIS_PWLB_HRA, RATE_BASIS_PWLB_CERTAINTY,
                                     RATE_BASIS_PWLB_STANDARD, RATE_BASIS_GILT_FLAT), SCOPE_PUBLIC,
           labels={RATE_BASIS_REAL: "public_rate_real", RATE_BASIS_PWLB_HRA: "public_rate_pwlb_hra",
                   RATE_BASIS_PWLB_CERTAINTY: "public_rate_pwlb_certainty",
                   RATE_BASIS_PWLB_STANDARD: "public_rate_pwlb_standard", RATE_BASIS_GILT_FLAT: "public_rate_gilt_flat"},
           register=True, requires=_REQUIRES_STATE_DEVELOPER,
           description="The state's cash rate, real gilt (2.60%, CEPA's 10y/20y index-linked method at the "
                       "workbook's valuation date) plus a public lending margin (DMO, Aug 2026): the National Loans "
                       "Fund's 11 bp (central, 2.71%), PWLB HRA 40 bp, certainty 80 bp, standard 100 bp, or the gilt "
                       "flat."),
    _lever(LEVER_STATE_STAKE_SCALE, (STAKE_WORKBOOK, STAKE_FUNDED_SCALE), SCOPE_PUBLIC,
           labels={STAKE_WORKBOOK: "stake_workbook", STAKE_FUNDED_SCALE: "stake_funded_scale"},
           register=True, requires={LEVER_NEW_BUILD_GENERATION: {NEW_BUILD_STATE_STAKE}},
           description="The state's stake under P1: the workbook's 30% (Assumptions C30), or a funded scale (15%, "
                       "GB Energy and NWF capital), with the workbook's Sc2 WACC recomputed at that stake."),
    _lever(LEVER_DELIVERED_SHARE, (DELIVERED_ALL, DELIVERED_HALF), SCOPE_PUBLIC,
           labels={DELIVERED_ALL: "delivered_all", DELIVERED_HALF: "delivered_half"},
           register=True, requires=_REQUIRES_FULL_NEW_BUILD,
           description="How much of the post-2025 build the state delivers under P2: all of it (the assumption on "
                       "its face), or half, each technology's cohort split by capacity; the private remainder keeps "
                       "its strike (coherence rule i)."),
    _lever(LEVER_STATE_CAPEX_OVERRUN, (OVERRUN_NONE, OVERRUN_PLUS_15), SCOPE_PUBLIC,
           labels={OVERRUN_NONE: "overrun_none", OVERRUN_PLUS_15: "overrun_15pct"},
           register=True, requires=_REQUIRES_STATE_DEVELOPER,
           description="A construction overrun borne by the state on the new build it owns: none (the workbook's "
                       "central optimism-bias uplift is zero), or +15% of capex (Sovacool et al. 2014; HM Treasury "
                       "optimism bias)."),
    _lever(LEVER_STATE_OM, (STATE_OM_SAME, STATE_OM_PLUS_10), SCOPE_PUBLIC,
           labels={STATE_OM_SAME: "public_om_same", STATE_OM_PLUS_10: "public_om_10pct"},
           register=True, requires=_REQUIRES_STATE_DEVELOPER,
           description="Public operating efficiency on owned generation: the same site fixed and variable O&M as a "
                       "private owner, or 10% more (Newbery & Pollitt 1997, 6%; IMF 2020, about 15%)."),
    # F29 (user, 26 Oct 2026): the workbook's economic rate, cash + rho x asset beta x P.  The
    # state holds the systematic risk of the plant it owns and charges no premium for it on the
    # bill (bills pay the cash rate); its social cost is a taxpayers' cost, booked by the register
    # (``socialised_risk_gbp``), the Exchequer ledger and the resource saving.  A stake (P1) books
    # none: the workbook prices Sc2's legs as contingent, not as an economic rate.
    # Meeting notes (user, 10 Nov 2026): the central is gilts alone (none); gilts plus the 1.9% premium is a
    # Scenario 3 sensitivity (sensitivities.named fully_nationalised_market_value_systematic_premium).
    _lever(LEVER_SYSTEMATIC_PREMIUM, (SYSTEMATIC_NONE, SYSTEMATIC_FLAT_19, SYSTEMATIC_FLAT_10,
                                      SYSTEMATIC_BETA_20, SYSTEMATIC_MARKET_ERP), SCOPE_PUBLIC,
           labels={SYSTEMATIC_FLAT_19: "systematic_premium_flat_19", SYSTEMATIC_NONE: "systematic_premium_none",
                   SYSTEMATIC_FLAT_10: "systematic_premium_flat_10", SYSTEMATIC_BETA_20: "systematic_premium_beta_20",
                   SYSTEMATIC_MARKET_ERP: "systematic_premium_market_erp"},
           register=True, requires=_REQUIRES_STATE_DEVELOPER,
           description="The social price of the systematic risk the state holds on plant it owns (workbook v6.2 "
                       "Sc4 Operative): none (central since 10 Nov 2026: gilts alone, pure Arrow-Lind, the Green Book), a "
                       "flat 1.9% on every technology (the Green Book Discount Rate Review's social-opportunity-cost "
                       "estimate, Freeman & Groom 2026 para 54; the Scenario 3 sensitivity), a flat 1% (the review's recommendation, para 55), 2% per unit "
                       "of CEPA asset beta (France, Quinet 2013; v2.0's central), or the market ERP, 5.22%, per unit "
                       "of beta (the Market Economy Investor Principle bound).  It moves the resource saving and the "
                       "Exchequer, never the bill."),
    _lever(LEVER_SOVEREIGN_YIELD, (YIELD_A, YIELD_B), SCOPE_PUBLIC,
           labels={YIELD_A: "yield_a", YIELD_B: "yield_b"}, register=True,
           description="Sovereign yields invariant (A), or rising with the scenario's own debt and contingent "
                       "exposure over GDP (B: 2 bp per pp, Poghosyan 2014; the workbook's Gilt-Adjusted WACCs, P1 on "
                       "its Sc2 path, P2 on Sc3, an acquiring case on Sc4), applied in the model year to every public "
                       "and P1 rate."),
    # User decision 7 (29 Sep 2026): the premium over RAV a market-value purchase of the
    # wires pays, from UK network and water takeovers (network_acquisition_premium.csv).
    _lever(LEVER_NETWORK_PREMIUM, (RAV_PREMIUM_CENTRAL, RAV_PREMIUM_LOW, RAV_PREMIUM_HIGH, RAV_PREMIUM_ZERO,
                                   RAV_PREMIUM_QUARTER),
           SCOPE_PUBLIC,
           labels={RAV_PREMIUM_CENTRAL: "rav_premium_central", RAV_PREMIUM_LOW: "rav_premium_low",
                   RAV_PREMIUM_HIGH: "rav_premium_high", RAV_PREMIUM_ZERO: "rav_premium_zero",
                   RAV_PREMIUM_QUARTER: "rav_premium_25pct"},
           requires={LEVER_NETWORKS: {NETWORKS_ACQUIRED}},
           description="The premium over RAV the state pays for the wires, annuitised like the legacy purchase "
                       "(30 years at the public rate): central 50% (recent GB electricity distribution deals), low "
                       "20% (listed market-to-RAV, no control premium), high 75%, zero (a Parliament-set price), or 25% "
                       "(a sensitivity, user request 2 Oct 2026)."),
    # ── 3. acquisition terms ────────────────────────────────────────────────
    # F29 (user, 26 Oct 2026): the workbook's acquisition bases.  The owners' value
    # (market_value) is the workbook's A2 method on each register row: after-tax flows plus
    # the capital allowances still to come on the row's book value, at the workbook's
    # post-tax operating rate.  book_value, no_support and ro_pass_through are its A1, A2
    # no-support and A3 bases, and apply to every acquired row, thermal and storage too.
    _lever(LEVER_PURCHASE_PRICE_BASIS, (PRICE_STATE_PV, PRICE_MARKET_VALUE, PRICE_BOOK_VALUE, PRICE_NO_SUPPORT,
                                        PRICE_RO_PASS_THROUGH),
           SCOPE_PUBLIC,
           labels={PRICE_STATE_PV: "purchase_at_state_value", PRICE_MARKET_VALUE: "purchase_at_market_value",
                   PRICE_BOOK_VALUE: "purchase_at_book_value", PRICE_NO_SUPPORT: "purchase_at_no_support_value",
                   PRICE_RO_PASS_THROUGH: "purchase_ro_pass_through"},
           register=True, requires=_REQUIRES_ACQUISITION,
           description="What the state pays for the legacy fleet: its present value to the state (the public rate); "
                       "its owners' value (the workbook's A2: after-tax flows and capital allowances at the post-tax "
                       "operating rate); its net book value (A1: historic capex written down); its owners' value "
                       "without CfD and RO support (A2 no-support, the Northern Rock principle); or the RO plant at "
                       "that no-support value and the rest at owners' value (A3, a policy case that embeds a "
                       "transfer from RO shareholders)."),
    # F19 P3 decision 3 (user, 6 Oct 2026; was user decision 11 of 29 Sep, book value
    # central): thermal and storage are bought at their owners' value like renewables
    # and nuclear (each row at its owners' present value, floored at zero, energy from
    # waste with its gate fees).  F29: the as-new book value case is retired (book value is
    # now the workbook's A1 basis of purchase_price_basis, on every acquired row).
    _lever(LEVER_THERMAL_BASIS, (THERMAL_MARKET_VALUE, "zero"), SCOPE_PUBLIC,
           labels={"zero": "thermal_at_zero", THERMAL_MARKET_VALUE: "thermal_at_market_value"}, register=True,
           requires={LEVER_LEGACY_FLEET: {LEGACY_ACQUIRE_ALL}},
           description="What the state pays for legacy thermal and storage under the state- or market-value "
                       "purchase: their owners' value (central, as renewables and nuclear are), or zero."),
    # F19 P3 decision 2 (user, 6 Oct 2026): the price the legacy fleet's merchant tail and
    # CfD references earn after the model year.  Not an assumption lever (no bound or grid
    # moves it: adding it to the best/worst pair would double P3's bound grid).  F29 (user,
    # 26 Oct 2026): the model's own 2030 price is the central, CP30's level the sensitivity
    # (the workbook asks for the ESM's price in place of its GBP 52/MWh placeholder).
    _lever(LEVER_LEGACY_PRICE_PATH, (PRICE_PATH_ANCHORED, PRICE_PATH_MODEL, PRICE_PATH_CP30, PRICE_PATH_TODAY),
           SCOPE_PUBLIC,
           labels={PRICE_PATH_ANCHORED: "price_path_anchored", PRICE_PATH_CP30: "price_path_cp30",
                   PRICE_PATH_MODEL: "price_path_model_2030", PRICE_PATH_TODAY: "price_path_today"},
           register=True, requires=_REQUIRES_ACQUISITION,
           description="The legacy valuation's price after the model year: each technology's capture price on a "
                       "straight line through today's forward level and every solved year (2030, 2040 when solved, "
                       "2050), flat after the last (central since 10 Nov 2026, legacy_price_anchors.csv); the model's "
                       "own 2030 price held flat; today's forward price held for the fleet's life (today_level, "
                       "11 Nov 2026); or CP30's 2030 level (each hour mapped onto CP.32's Further Flex and "
                       "Renewables duration curve, GBP 69.9/MWh)."),
    # User decision (24 Oct 2026): the central purchase prices legacy offshore wind at the model's
    # cash-flow (owners') value, about 2.31 GBP m/MW; a named P3 case prices it at observed UK
    # operating-stake deal multiples (lever_parameters.legacy_offshore_transaction_multiple), every
    # other row still at owners' value.  Not an assumption lever (no bound or grid moves it).
    _lever(LEVER_LEGACY_OFFSHORE_PRICE, (OFFSHORE_OWNERS_VALUE, OFFSHORE_DEAL_MULTIPLE), SCOPE_PUBLIC,
           labels={OFFSHORE_OWNERS_VALUE: "offshore_at_owners_value", OFFSHORE_DEAL_MULTIPLE: "offshore_at_deal_multiples"},
           register=True, requires=_REQUIRES_ACQUISITION,
           description="What the state pays for legacy offshore wind: its owners' value from the model's cash flows "
                       "(central), or observed UK operating offshore wind stake sales, 2020-2025, per MW of "
                       "100%-equivalent capacity (the median, about GBP 4.6 m/MW in 2024 money; "
                       "data/public_ownership/financing/legacy_offshore_transaction_multiples.csv)."),
    _lever(LEVER_GENERATION_CONTROL_PREMIUM, (CONTROL_PREMIUM_NONE, CONTROL_PREMIUM_20), SCOPE_PUBLIC,
           labels={CONTROL_PREMIUM_NONE: "control_premium_none", CONTROL_PREMIUM_20: "control_premium_20pct"},
           register=True, requires=_REQUIRES_ACQUISITION,
           description="A control premium over the valued group's purchase price (backlog AP1): none, or 20% (EDF's "
                       "2022-23 buyout paid 34% over the 12-month average, 53% over the undisturbed price)."),
    _lever(LEVER_INTERCONNECTOR_PRICE, (IC_PRICE_CENTRAL, IC_PRICE_LOW, IC_PRICE_HIGH), SCOPE_PUBLIC,
           labels={IC_PRICE_CENTRAL: "ic_price_central", IC_PRICE_LOW: "ic_price_low", IC_PRICE_HIGH: "ic_price_high"},
           requires=_REQUIRES_ACQUISITION,
           description="The interconnectors' GB halves' purchase price: their rent capitalised at the owner rate "
                       "(central 6.7%), a lower price at 8.3%, or a higher one at 5%."),
    _lever(LEVER_TRANSACTION_COSTS, (TRANSACTION_NONE, TRANSACTION_TWO_PCT), SCOPE_PUBLIC,
           labels={TRANSACTION_NONE: "transaction_costs_none", TRANSACTION_TWO_PCT: "transaction_costs_2pct"},
           requires=_REQUIRES_ACQUISITION,
           description="Advisory and transaction costs of the acquisition: none, or 2% of the purchase price and the "
                       "networks' RAV, annuitised like the purchase."),
    _lever(LEVER_ASSET_LIVES, (LIVES_CENTRAL, LIVES_EXTENDED), SCOPE_PUBLIC,
           labels={LIVES_CENTRAL: "asset_lives_central", LIVES_EXTENDED: "asset_lives_extended"},
           register=True, requires=_REQUIRES_ACQUISITION,
           description="Legacy retirement lives in the valuation: central (solar 38, offshore wind 30, onshore wind "
                       "and marine 25 years, decision 48; Sizewell B 2035), or "
                       "extended (renewables +5 years, Sizewell B to 2055), which raises the market-value price."),
    # F28 (user, 26 Oct 2026): each acquired asset is repaid over its own remaining life (the valuation's
    # retirement year less the model year) at the public rate, so the purchase's lifetime present value is
    # unchanged and the 2030s carry more of it; one 30-year (or 15-year) term for every asset are toggles.
    _lever(LEVER_ACQUISITION_TERM, (TERM_REMAINING_LIFE, "30", "15"), SCOPE_PUBLIC,
           labels={TERM_REMAINING_LIFE: "term_remaining_life", "30": "term_30y", "15": "term_15y"}, register=True,
           requires=_REQUIRES_ACQUISITION,
           description="How the purchase is annuitised at the public rate: each asset over its own remaining life "
                       "(central), or every asset over one term of 30 or 15 years."),
    # Item G3 (decision 8): what happens to the acquired plant's support contracts.
    # Full nationalisation is exempt (the state bought the counterparty), which
    # parse_sensitivities enforces, because named cases skip ``applies``.
    _lever(LEVER_LEGACY_CONTRACTS, (CONTRACTS_HONOURED, CONTRACTS_CANCELLED_BILLS, CONTRACTS_CANCELLED_EXCHEQUER),
           SCOPE_PUBLIC,
           labels={CONTRACTS_HONOURED: "contracts_honoured", CONTRACTS_CANCELLED_BILLS: "contracts_cancelled_on_bills",
                   CONTRACTS_CANCELLED_EXCHEQUER: "contracts_cancelled_on_exchequer"},
           register=True,
           requires={**_REQUIRES_ACQUISITION, LEVER_PRE_2025_CFD: {PRE_2025_PRIVATE}},
           description="The acquired plant's CfD top-up and ROC income: honoured (bought with the plant at the public "
                       "rate), or cancelled with the former owners paid their value at their own WACC, on the bill or "
                       "by the Exchequer."),
    # ── 4. how public plant is paid ─────────────────────────────────────────
    # ``auction_clearing`` (user decision 5, 29 Sep 2026): the state keeps the auction
    # strike, so its owned CfDs are not re-priced on the bill at all; they join the
    # public account as R2 and their rent follows surplus_recycling (the Treasury
    # keeps its return).  Co-owners keep their share of the saving.
    _lever(LEVER_CFD_STRIKE_BASIS, (STRIKE_FINANCING_SAVING, STRIKE_AT_COST, STRIKE_AUCTION_CLEARING), SCOPE_PUBLIC,
           labels={STRIKE_FINANCING_SAVING: "strike_financing_saving", STRIKE_AT_COST: "strike_at_cost",
                   STRIKE_AUCTION_CLEARING: "strike_auction"},
           requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="A re-priced strike falls by the financing saving, or to the asset's cost; or the state keeps "
                       "the auction-cleared strike and its return (R1 rows join the public account as R2)."),
    _lever(LEVER_CFD_AMORTISATION, ("contract_term", "economic_life"), SCOPE_PUBLIC,
           labels={"contract_term": "amortised_over_term", "economic_life": "amortised_over_life"}, requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="The re-priced capital charge annuitised over the contract term or the economic life."),
    # Review 29 Sep 2026 (user decision 5): ``offset_policy_levies`` was a duplicate of the
    # rebate (the same code path).  It now means what it says: the routed surplus offsets
    # the Renewables Obligation, Feed-in Tariff and Warm Home Discount lines of DOMESTIC
    # customers, over domestic supply, capped at those lines (the rest is rebated over
    # eligible supply).  ``rebate_gross`` returns every positive state balance (merchant,
    # R2 and the R1 remainder) to the bill and leaves every negative one with the
    # Exchequer, the consumer-first reading.
    _lever(LEVER_SURPLUS_RECYCLING,
           ("public_cfd_at_cost", "cost_recovery_contracting", "rebate_per_mwh_eligible_supply",
            RECYCLING_OFFSET_POLICY_LEVIES, "retained_by_treasury", "no_assumption", RECYCLING_REBATE_GROSS), SCOPE_PUBLIC,
           labels={value: f"recycling_{value}" for value in (
               "public_cfd_at_cost", "cost_recovery_contracting", "rebate_per_mwh_eligible_supply",
               RECYCLING_OFFSET_POLICY_LEVIES, "retained_by_treasury", "no_assumption", RECYCLING_REBATE_GROSS)},
           requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="How the state's surplus reaches, or does not reach, the bill: the at-cost CfD, cost-recovery "
                       "contracting, a rebate over eligible supply, an offset to domestic customers' legacy levies, "
                       "retained by the Treasury, or every positive balance rebated and every deficit left with the "
                       "Exchequer."),
    _lever(LEVER_SHORTFALL_FUNDING, ("levy_on_eligible_supply", "exchequer"), SCOPE_PUBLIC,
           labels={"levy_on_eligible_supply": "shortfall_levy", "exchequer": "shortfall_exchequer"}, requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="A portfolio shortfall levied on eligible supply or carried by the Exchequer."),
    _lever(LEVER_ROUTE_BOOKING, ("sensitivity_only", "booked"), SCOPE_PUBLIC,
           labels={"sensitivity_only": "routes_reported", "booked": "routes_booked"}, requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="The Capacity Market and CCS-levy financing savings reported beside the bill or booked on it."),
    _lever(LEVER_NEW_BUILD_CONTRACTS, ("merchant", "at_cost"), SCOPE_ALL,
           labels={"merchant": "new_build_merchant", "at_cost": "new_build_at_cost"},
           description="Merchant new build left on merchant revenue, or contracted at required revenue in every case "
                       "(a change of comparator basis)."),
    _lever(LEVER_LDES_FLOOR_BASIS, ("cost_of_debt_like", "technology_wacc"), SCOPE_ALL,
           labels={"cost_of_debt_like": "ldes_floor_debt_like", "technology_wacc": "ldes_floor_at_wacc"},
           description="The long-duration storage floor at a debt-like rate or at the technology WACC (moves private too)."),
    # User decision, 4 Oct 2026 (follow-up F6): CfD auctions pay as clear, so the projects
    # cheaper than the marginal one earn more than their required return.  The auction
    # calibration (trust the auction) scales a cohort's capex to break even on average;
    # this lever keeps a margin of that share of contract revenue above full cost instead
    # (the shares are lever_parameters.auction_margin_share: central 5%, none 0, high 10%).
    # It moves the plant's cost in every case, private included, so it changes the
    # comparator.  2050 has no calibrated cohorts, so it moves nothing there.
    _lever(LEVER_AUCTION_MARGIN, (AUCTION_MARGIN_CENTRAL, AUCTION_MARGIN_NONE, AUCTION_MARGIN_HIGH), SCOPE_ALL,
           labels={AUCTION_MARGIN_CENTRAL: "auction_margin_central", AUCTION_MARGIN_NONE: "auction_margin_none",
                   AUCTION_MARGIN_HIGH: "auction_margin_high"},
           register=True,
           description="The margin the auction-calibrated CfD cohorts keep above full cost at their cleared strike: "
                       "5% of contract revenue centrally (pay-as-clear leaves infra-marginal profit), none (the "
                       "cohort just breaks even), or 10%.  The state's own plant needs no margin, so it is a "
                       "saving of ownership."),
    # User decision, 4 Oct 2026 (follow-up F7): central government carries its own risks
    # rather than buying commercial insurance (HM Treasury, Managing Public Money), so
    # plant the state owns outright saves the insurer's loading on its premium (the part
    # above expected claims, which the state still bears): lever_parameters.self_insurance_share,
    # central 35%, none 0, high 50%.  Only wholly state-owned rows: a minority stake (P1)
    # leaves the project company buying cover.  Vintaged (2040 on) registers ignore it
    # unless lever_parameters.self_insurance_in_vintaged_registers is set (2050 has not decided).
    _lever(LEVER_SELF_INSURANCE, (SELF_INSURANCE_CENTRAL, SELF_INSURANCE_NONE, SELF_INSURANCE_HIGH), SCOPE_PUBLIC,
           labels={SELF_INSURANCE_CENTRAL: "self_insurance_central", SELF_INSURANCE_NONE: "self_insurance_none",
                   SELF_INSURANCE_HIGH: "self_insurance_high"},
           register=True, requires=_REQUIRES_STATE_DEVELOPER,
           description="Plant the state owns outright self-insures: it saves 35% of its insurance premium centrally "
                       "(the insurer's costs and margin; it still bears the expected claims), none, or 50%."),
    _lever(LEVER_STATE_OVERHEAD, (OVERHEAD_CENTRAL, OVERHEAD_LOW, OVERHEAD_HIGH), SCOPE_PUBLIC,
           labels={OVERHEAD_CENTRAL: "overhead_central", OVERHEAD_LOW: "overhead_low", OVERHEAD_HIGH: "overhead_high"},
           register=True, requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="What it costs the state to run the thing (item OH1): the holding company's board, group "
                       "finance and treasury, legal, HR, IT, risk and regulatory functions, charged per kW of "
                       "state-owned generation and storage on top of the site-level fixed O&M both owners already "
                       "carry.  Central 3 GBP/kW/yr, twins 1 and 8; the rates are configured, not hard-coded."),
    # ── 5. market and institution ───────────────────────────────────────────
    _lever(LEVER_WHOLESALE_PRICING, (PRICING_PAY_AS_CLEAR, PRICING_PAY_AS_BID_CEILING), SCOPE_ALL,
           labels={PRICING_PAY_AS_BID_CEILING: "pay_as_bid_ceiling"},
           requires_by_value={PRICING_PAY_AS_BID_CEILING: {LEVER_NEW_BUILD_GENERATION: {NEW_BUILD_PRIVATE}}},
           description="Pay-as-clear, or the most pay-as-bid could remove computed from the settled positions "
                       "(a bounding line, never a headline; plan Part 7.3)."),
    _lever(LEVER_DISPATCH_INSTITUTION, (INSTITUTION_MARKET, INSTITUTION_NATIONALISED_DISPATCH), SCOPE_PUBLIC,
           labels={INSTITUTION_MARKET: "market_dispatch", INSTITUTION_NATIONALISED_DISPATCH: "central_dispatch"},
           requires={LEVER_LEGACY_FLEET: {LEGACY_ACQUIRE_ALL}},
           description="The market machinery kept (I1) or central dispatch at resource cost (I3, the second solved "
                       "run group); only meaningful once every asset is owned."),
    # Item BM1: who bids what in the balancing mechanism.  Not a register lever:
    # it selects the solved run group whose dispatch the case's bill reads (a
    # reference_run_groups entry declaring the same balancing_bids), and every
    # stage of that group accounts it on that dispatch.
    _lever(LEVER_BALANCING_BIDS, (BIDS_CONTRACT_VALUE, BIDS_STATE_AT_COST), SCOPE_PUBLIC,
           labels={BIDS_CONTRACT_VALUE: "bm_bids_contract_value", BIDS_STATE_AT_COST: "state_bids_at_cost"},
           requires={**_REQUIRES_FULL_NEW_BUILD, LEVER_LEGACY_FLEET: {LEGACY_PRIVATE}},
           description="Every CfD unit bids its contract value to turn down in the balancing mechanism, or the "
                       "state-owned share of each unit bids at cost (a second solved run group, P2's twelve months)."),
    # F27 item 6 (user, 20 Oct 2026; a lever since 24 Oct): a capacity-procurement choice ownership
    # makes easier, on P2 (one at a time) and P3 (named).  Not offered on P1: a 30% stake in new
    # build owns no existing plant and gives the state no hand in procuring it, so the choice there
    # would be a regulatory change open to the private case too (a change of comparator).  A register
    # lever because under P3 it moves the purchase price (the legacy valuation); on the market bill
    # it is its own row against the Capacity Market line, booked to private owners.
    _lever(LEVER_CAPACITY_PROCUREMENT, (CAPACITY_AUCTION, CAPACITY_COST_BASED_EXISTING), SCOPE_PUBLIC,
           labels={CAPACITY_AUCTION: "capacity_auction", CAPACITY_COST_BASED_EXISTING: "capacity_cost_based_existing"},
           register=True, requires=_REQUIRES_FULL_NEW_BUILD,
           description="How existing capacity needed for security is paid: the Capacity Market clearing price "
                       "(today), or a cost-based contract, its going-forward fixed cost less its own wholesale and "
                       "balancing margin, floored at zero and capped at its receipt; new capacity is still "
                       "auctioned.  A transfer from plant owners under P2; under P3 a lower purchase price (only if "
                       "announced before the purchase)."),
    # ── 6. fiscal incidence: who pays, never how much ─────────────────────────
    # Each moves money between consumers, the Exchequer and investors and leaves the
    # resource saving unchanged to the pound (the three-party ledger enforces it for
    # every preset case).  values[0] is today's behaviour, so every existing number
    # is reproduced.
    _lever(LEVER_STATE_DEFICIT_FUNDING, (DEFICITS_EXCHEQUER, DEFICITS_BILL_LEVY), SCOPE_PUBLIC,
           labels={DEFICITS_EXCHEQUER: "deficits_exchequer", DEFICITS_BILL_LEVY: "deficits_on_bills"},
           requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="Who funds the state's deficits outside the recycled routes (storage, the at-cost remainder, "
                       "the Capacity Market and dispatchable-CfD routes, a negative Capacity Market booking): the "
                       "Exchequer, or a levy on eligible supply."),
    _lever(LEVER_TAX_EQUIVALENT, (TAX_EQUIVALENT_NONE, TAX_EQUIVALENT_ON), SCOPE_PUBLIC,
           labels={TAX_EQUIVALENT_NONE: "tax_equivalent_none", TAX_EQUIVALENT_ON: "tax_equivalent_on"},
           register=True, requires=_REQUIRES_PUBLIC_NEW_BUILD,
           description="Competitive neutrality: state-owned new build pays the corporation tax a private owner "
                       "would (the register's corporation_tax_forgone_gbp, added to its required revenue and paid "
                       "to the Exchequer), and acquired networks keep the DUoS tax allowance, paid to the Exchequer. "
                       "Legacy plant is excluded: its as-new tax base is meaningless (house rule 10)."),
    _lever(LEVER_NETWORK_INCIDENCE, (NETWORK_INCIDENCE_BILLS, NETWORK_INCIDENCE_RETAINED),
           SCOPE_PUBLIC,
           labels={NETWORK_INCIDENCE_BILLS: "network_saving_to_bills", NETWORK_INCIDENCE_RETAINED: "network_return_retained"},
           requires={LEVER_NETWORKS: {NETWORKS_ACQUIRED}},
           description="Where the bought wires' saving lands: all on TNUoS and DUoS (today), or the return retained "
                       "by the Treasury (allowed revenue unchanged, the wedge and the tax allowance paid to the "
                       "Exchequer)."),
    _lever(LEVER_ACQUISITION_FINANCING, (ACQUISITION_ON_TARIFF, ACQUISITION_EXCHEQUER), SCOPE_PUBLIC,
           labels={ACQUISITION_ON_TARIFF: "acquisition_on_tariff", ACQUISITION_EXCHEQUER: "acquisition_taxpayer_funded"},
           requires=_REQUIRES_ACQUISITION,
           description="The acquisition annuity (legacy renewables and nuclear, thermal and storage, the "
                       "interconnectors' GB halves) on the tariff, or funded from general taxation."),
    # User answer Q5 (30 Sep 2026): P3's treasury_first.  The state charges households
    # the private comparator's market bill and the Treasury keeps the whole margin over
    # the tariff's cost (a tariff line paid to the Exchequer), so the bill is private's,
    # the Exchequer gains the margin and investors and the resource cost do not move.
    _lever(LEVER_TARIFF_PRICING, (TARIFF_AT_COST, TARIFF_AT_MARKET_BILL), SCOPE_PUBLIC,
           labels={TARIFF_AT_COST: "tariff_at_cost", TARIFF_AT_MARKET_BILL: "tariff_at_market_bill"},
           requires=_REQUIRES_ACQUISITION,
           description="What the acquisition tariff charges: its cost (the base), or the private comparator's "
                       "market bill, the state charging market prices and the Treasury keeping the margin."),
    # ── 7. background: every case alike, private included ─────────────────────
    _lever(LEVER_LEGACY_LEVY_FUNDING, (LEGACY_LEVIES_ON_BILLS, LEGACY_LEVIES_TO_TAXATION), SCOPE_ALL,
           labels={LEGACY_LEVIES_ON_BILLS: "legacy_levies_on_bills", LEGACY_LEVIES_TO_TAXATION: "legacy_levies_to_taxation"},
           description="The Renewables Obligation, Feed-in Tariff and Warm Home Discount of domestic customers on "
                       "bills (current law) or moved to general taxation; a background comparator, never an "
                       "ownership effect (the Exchequer ledger books the levies it takes on)."),
    _lever(LEVER_BECCS_REMOVAL_FUNDING, (BECCS_REMOVAL_EXCHEQUER, BECCS_REMOVAL_BILL_LEVY), SCOPE_ALL,
           labels={BECCS_REMOVAL_EXCHEQUER: "beccs_removal_exchequer", BECCS_REMOVAL_BILL_LEVY: "beccs_removal_on_bills"},
           description="Power BECCS's carbon-removal leg (the dual CfD's CfDc, DESNZ): paid by the Exchequer in every "
                       "case (user decision 2), or recovered by a levy on electricity bills in every case."),
)}

#: Levers a preset may move: incidence only, so the resource saving cannot change.
INCIDENCE_LEVERS: tuple[str, ...] = (
    LEVER_SURPLUS_RECYCLING, LEVER_SHORTFALL_FUNDING, LEVER_ROUTE_BOOKING, LEVER_CFD_STRIKE_BASIS,
    LEVER_STATE_DEFICIT_FUNDING, LEVER_TAX_EQUIVALENT, LEVER_NETWORK_INCIDENCE, LEVER_ACQUISITION_FINANCING,
    LEVER_TARIFF_PRICING,
)
#: Incidence levers that describe the market bill's contract form and routes, with no
#: meaning once the legacy fleet is bought (the acquisition tariff recovers every cost).
MARKET_ONLY_INCIDENCE_LEVERS: tuple[str, ...] = (
    LEVER_SURPLUS_RECYCLING, LEVER_SHORTFALL_FUNDING, LEVER_ROUTE_BOOKING, LEVER_CFD_STRIKE_BASIS,
    LEVER_STATE_DEFICIT_FUNDING,
)
#: Levers a best/worst bound or a toggle grid may move: assumptions on a scenario's
#: own cost, never incidence and never the comparator (no all_cases lever).
ASSUMPTION_LEVERS: tuple[str, ...] = (
    LEVER_STATE_OVERHEAD, LEVER_SOVEREIGN_YIELD, LEVER_PUBLIC_RATE_BASIS, LEVER_SYSTEMATIC_PREMIUM,
    LEVER_NETWORK_PREMIUM, LEVER_IN_CONSTRUCTION,
    LEVER_PURCHASE_PRICE_BASIS, LEVER_THERMAL_BASIS, LEVER_ACQUISITION_TERM,
    LEVER_STATE_STAKE_SCALE, LEVER_DELIVERED_SHARE, LEVER_STATE_CAPEX_OVERRUN, LEVER_STATE_OM,
    LEVER_GENERATION_CONTROL_PREMIUM, LEVER_INTERCONNECTOR_PRICE, LEVER_TRANSACTION_COSTS, LEVER_ASSET_LIVES,
)
#: Levers a toggle grid may move (user decisions 3-5, 30 Sep 2026): the assumption
#: levers, plus three structural choices the explorer shows as axes of every grid
#: (who owns the wires, whether the state takes the pre-2025 CfD projects, the strike
#: basis).  A grid is not an incidence position, so the resource invariance does not
#: apply to it; the presets alone carry that check.
GRID_LEVERS: tuple[str, ...] = ASSUMPTION_LEVERS + (LEVER_NETWORKS, LEVER_PRE_2025_CFD, LEVER_CFD_STRIKE_BASIS)
#: Levers a coupled selector option may set beside its own lever, never as an axis of
#: their own (user decision, 2 Oct 2026: P1's "cut to cost" also costs over the life).
GRID_COUPLED_LEVERS: tuple[str, ...] = (LEVER_CFD_AMORTISATION,)
#: Comparator-changing (all_cases) assumptions a bound may also move.  Each bound
#: kind then gets its own private comparator case at the same point on them, so the
#: bound is measured against a like-for-like private bill (house rule 3).
BOUND_COMPARATOR_LEVERS: tuple[str, ...] = (LEVER_LDES_FLOOR_BASIS, LEVER_PRIVATE_WACC_BASIS)
#: Levers a bound grid may move (user request, 2 Oct 2026): every assumption a best or
#: worst bound moves, so the explorer can switch them one by one between the two.
BOUND_GRID_LEVERS: tuple[str, ...] = ASSUMPTION_LEVERS + BOUND_COMPARATOR_LEVERS
#: A bound grid's name in ``grid_memberships`` is its parent's id plus this suffix, so
#: it never collides with the parent's own toggle grid (whose name is the parent id).
BOUND_GRID_SUFFIX = "#bounds"
#: Levers retired from the catalogue, and where the choice lives now.
RETIRED_LEVERS: dict[str, str] = {
    LEVER_CFD_ELIGIBILITY: ("retired 29 Sep 2026 (review C1): structurally inert, its two cases reproduced P1 and P2; "
                            "commissioning vintage is reached through pre_2025_cfd_capacity: public_repriced"),
    # F29 (user, 26 Oct 2026): workbook v2.0 (GB_Power_Model_Rebuild_v5.xlsx).
    LEVER_CONSTRUCTION_PREMIUM: ("retired 26 Oct 2026 (F29): workbook v2.0 charges the state the cash rate with no "
                                 "construction premium (expected overruns are capex: state_capex_overrun); the risk "
                                 "the state holds is priced by systematic_premium, off the bill"),
    LEVER_GUARANTEE_COST: ("retired 26 Oct 2026 (F29) with networks: debt_guarantee (workbook v2.0 has no network "
                           "rate)"),
    LEVER_GUARANTEE_PASS_THROUGH: ("retired 26 Oct 2026 (F29) with networks: debt_guarantee (workbook v2.0 has no "
                                   "network rate)"),
    LEVER_OWNER_DISCOUNT_RATE: ("retired 26 Oct 2026 (F29): every legacy row is discounted at workbook v2.0's post-tax "
                                "operating rate (Legacy Fleet column AF; Hinkley Point C at the new-build nuclear "
                                "post-tax rate)"),
}
#: Lever values retired from the catalogue, and what replaces them.
RETIRED_VALUES: dict[tuple[str, str], str] = {
    (LEVER_PUBLIC_RATE_BASIS, RATE_BASIS_NOMINAL): (
        "retired 29 Sep 2026 (user decision 9): 4.8% nominal on real cash flows is a unit mix; use real_plus_1pp or "
        "gilt_30y_real"),
    # OW9 (user, 2 Oct 2026; retired 3 Oct 2026): no case ever took them and none had a
    # sourced number behind it.  public_return_explicit_rate, acquisition.purchase_price_gbp
    # and acquisition.thermal_price_gbp went with them.
    (LEVER_PUBLIC_RETURN, RETURN_EXPLICIT): (
        "retired 3 Oct 2026 (OW9): a configured rate with no source; use public_rate, cepa_decomposition or "
        "commercial_benchmark (the 6.2% benchmark, Assumptions C29)"),
    (LEVER_PURCHASE_PRICE_BASIS, PRICE_EXPLICIT): (
        "retired 3 Oct 2026 (OW9): a configured total with no source; use market_value, state_present_value or "
        "workbook_placeholder (the workbook's book value)"),
    (LEVER_THERMAL_BASIS, PRICE_EXPLICIT): (
        "retired 3 Oct 2026 (OW9): a configured total with no source; use market_value or zero"),
    # F29 (user, 26 Oct 2026): replaced by workbook v2.0's own sensitivities.
    (LEVER_PRIVATE_WACC_BASIS, WACC_CEPA_VANILLA): (
        "retired 26 Oct 2026 (F29): workbook v2.0 reports vanilla rates for comparison only; its co-equal case is "
        "fixed_tmr_shielded"),
    (LEVER_PRIVATE_WACC_BASIS, WACC_UNIFORM_GEARING): (
        "retired 26 Oct 2026 (F29): workbook v2.0 has no uniform-gearing lead rate; use fixed_tmr_shielded"),
    # v6.2 (user, 30 Oct 2026).
    (LEVER_PRIVATE_WACC_BASIS, WACC_OBSERVED_GEARING): (
        "retired 30 Oct 2026 (workbook v6.2): the lead rate is shielded at a notional 55% gearing; use "
        "notional_gearing_shielded (or fixed_tmr_shielded)"),
    (LEVER_SYSTEMATIC_PREMIUM, SYSTEMATIC_SOCIAL_P20): (
        "retired 30 Oct 2026 (workbook v6.2): the central is a flat 1.9%, social_flat_19; 2% per unit of beta is "
        "social_beta_20"),
    (LEVER_SYSTEMATIC_PREMIUM, SYSTEMATIC_SOCIAL_P15): (
        "retired 30 Oct 2026 (workbook v6.2): use social_flat_19, none, social_flat_10, social_beta_20 or market_erp"),
    (LEVER_SYSTEMATIC_PREMIUM, SYSTEMATIC_SOCIAL_P30): (
        "retired 30 Oct 2026 (workbook v6.2): use social_flat_19, none, social_flat_10, social_beta_20 or market_erp"),
    (LEVER_PUBLIC_RETURN, RETURN_COMMERCIAL_BENCHMARK): (
        "retired 26 Oct 2026 (F29): the 6.2% benchmark (v11-2 Assumptions C29) has no source in workbook v2.0"),
    (LEVER_PUBLIC_RATE_BASIS, RATE_BASIS_PLUS_1PP): (
        "retired 26 Oct 2026 (F29): the risk the state holds is priced by systematic_premium (the economic rate); "
        "the cash rate's alternatives are the lending margins: pwlb_hra, pwlb_certainty, pwlb_standard, gilt_flat"),
    (LEVER_PUBLIC_RATE_BASIS, RATE_BASIS_GILT_30Y): (
        "retired 26 Oct 2026 (F29): workbook v2.0's risk-free rate is the index-linked gilt curve (CEPA's method); "
        "use the lending margins: pwlb_hra, pwlb_certainty, pwlb_standard, gilt_flat"),
    (LEVER_SOVEREIGN_YIELD, YIELD_C): (
        "retired 26 Oct 2026 (F29): workbook v2.0 leaves assumption C blank, pending its input-output model"),
    (LEVER_NETWORKS, NETWORKS_DEBT_GUARANTEE): (
        "retired 26 Oct 2026 (F29): v11-2's 75 bp network debt cut came from a sheet workbook v2.0 does not have"),
    (LEVER_NETWORK_INCIDENCE, NETWORK_INCIDENCE_FEE): (
        "retired 26 Oct 2026 (F29) with networks: debt_guarantee"),
    (LEVER_PURCHASE_PRICE_BASIS, PRICE_WORKBOOK_PLACEHOLDER): (
        "retired 26 Oct 2026 (F29): v11-2's GBP 100 bn placeholder; workbook v2.0's book value is book_value (A1)"),
    (LEVER_THERMAL_BASIS, THERMAL_BOOK_VALUE): (
        "retired 26 Oct 2026 (F29): book value is workbook v2.0's A1 basis, purchase_price_basis: book_value, on "
        "every acquired row"),
}
#: Presets retired from the catalogue, and where their position lives now.
RETIRED_PRESETS: dict[str, str] = {
    "consumer_first": ("retired 30 Sep 2026 (user decision 1): bill-payers first is every scenario's base now, so it "
                       "makes no case; the old central incidence is the todays_rules preset"),
}

#: label -> lever, checked unique at import: the case id carries the label alone.
LABELS: dict[str, str] = {}
for _lever_obj in LEVERS.values():
    for _value in _lever_obj.values:
        _label = _lever_obj.label(_value)
        if _label in LABELS:
            raise RuntimeError(f"variant label {_label!r} is claimed by both {LABELS[_label]!r} and {_lever_obj.name!r}")
        LABELS[_label] = _lever_obj.name
del _lever_obj, _value, _label

# ── Year-gated levers (FES 2050 plan, tasks Y9 and Y10, 30 Sep 2026) ─────────
# Enforced exactly like LEVERS (house rule 11: only these values are implemented,
# and anything else fails at parse time), but kept OUT of ``LEVERS`` until the 2030
# year closes: every stage that iterates ``LEVERS`` (the register's lever columns,
# the case settings, the explorer's labels and case data) would otherwise gain a
# column or key on every 2030 output.  A case carries a gated lever only when its
# bundle or sensitivity names it, so no 2030 case does; a stage reads one with
# ``settings.get(name, lever_default(name))``.  When 2030 closes they move into
# LEVERS in one edit, with their explorer labels.
LEVER_STATE_DEPLOYMENT_SHARE = "state_deployment_share"
LEVER_PRIVATE_RATE_PATH = "private_rate_path"
DEPLOYMENT_ALL = "all"
DEPLOYMENT_SHARES: dict[str, float] = {"0.25": 0.25, "0.50": 0.50, "0.75": 0.75, "1.00": 1.00}
RATE_PATH_FLAT, RATE_PATH_NESO_WIND_RAMP = "flat", "neso_wind_ramp"
#: FES 2050 decision 73 (user, 5 Oct 2026): the Capacity Market design under public ownership.
#: ``study_central`` takes the study's ``public_ownership.capacity_market_design.central``
#: (absent, as in every 2030 config: ``market_participant``, today's treatment).
LEVER_CM_DESIGN = "capacity_market_design"
CM_DESIGN_STUDY, CM_DESIGN_MARKET_PARTICIPANT, CM_DESIGN_STATE_PLANNED = (
    "study_central", "market_participant", "state_planned")

GATED_LEVERS: dict[str, Lever] = {lever.name: lever for lever in (
    # FES 2050 plan decision 13 and 17, lever L4 of the 2050 plan (not the 2030
    # catalogue's L4, which is ``networks``).  ``all`` is today's P2: the state owns
    # every post-2025 row inside its boundary.  A share owns that share of the
    # capacity P2 would own in every (technology, build year) cell, whole rows in a
    # fixed md5 order filled to the share by the capacity midpoint
    # (finance.deployment_selection); ``1.00`` takes that path and reproduces ``all``.
    _lever(LEVER_STATE_DEPLOYMENT_SHARE, (DEPLOYMENT_ALL, *DEPLOYMENT_SHARES), SCOPE_PUBLIC,
           labels={DEPLOYMENT_ALL: "deployment_share_all", "0.25": "deployment_share_25",
                   "0.50": "deployment_share_50", "0.75": "deployment_share_75", "1.00": "deployment_share_100"},
           register=True, requires={**_REQUIRES_FULL_NEW_BUILD, LEVER_DELIVERED_SHARE: {DELIVERED_ALL}},
           description="The state's share of the post-2025 build under P2 (FES 2050 decisions 13 and 17): every row "
                       "P2 owns (today), or 25, 50, 75 or 100% of that capacity in every technology and build year, "
                       "whole rows in a fixed order within each technology-vintage cell; the rest is private "
                       "new build on its strike."),
    # FES 2050 plan decision 19: flat real hurdle rates by build year centrally; the
    # NESO FES 2025 wind ramp (a linear step-down over 2031-2035: offshore 8.3 to
    # 6.3%, onshore 6.2 to 5.2%, floating 9.8 to 7.8%) as the spread sensitivity.
    # It moves every privately financed wind row built after 2030 by the ramp's delta
    # on 2030, in every case, private included, so it changes the comparator.
    _lever(LEVER_PRIVATE_RATE_PATH, (RATE_PATH_FLAT, RATE_PATH_NESO_WIND_RAMP), SCOPE_ALL,
           labels={RATE_PATH_FLAT: "private_rate_flat", RATE_PATH_NESO_WIND_RAMP: "private_rate_neso_wind_ramp"},
           register=True,
           description="The private cost of capital by build year: flat (the workbook, DESNZ and CEPA view), or "
                       "NESO's FES 2025 wind step-down (fes2025_hurdle_rates.csv) added to the private rate of "
                       "wind vintages after 2030 as its change on 2030."),
    # FES 2050 decision 73 (user, 5 Oct 2026): how the state's plant meets the Capacity Market.
    # ``state_planned``: under P2 the state's de-rated eligible capacity leaves the auction (the
    # target falls by it, the rest clears at the residual price); under an acquisition (P3) the CM
    # is replaced by adequacy planning, a targeted contract for what the state does not own priced
    # at that residual clearing. ``market_participant``: today's treatment (state plant paid like
    # any participant). Not a register lever: required revenue does not move, only the CM booking
    # (scripts/public_ownership/capacity_market_design.py). Vintaged (2050) registers only: the
    # 2030 Capacity Market mechanisms are F9 (capacity_market_state_bids) and F27 item 6
    # (capacity_procurement), and a register never takes both (calculate_financing_scenarios).
    _lever(LEVER_CM_DESIGN, (CM_DESIGN_STUDY, CM_DESIGN_MARKET_PARTICIPANT, CM_DESIGN_STATE_PLANNED), SCOPE_PUBLIC,
           labels={CM_DESIGN_STUDY: "cm_study_central", CM_DESIGN_MARKET_PARTICIPANT: "cm_market_participant",
                   CM_DESIGN_STATE_PLANNED: "cm_state_planned"},
           requires=_REQUIRES_FULL_NEW_BUILD,
           description="The Capacity Market design under public ownership (FES 2050 decision 73): the study's "
                       "central; the state's plant paid in the auction like any participant (today); or the "
                       "state's plant out of the auction (P2) and adequacy planning with a targeted contract "
                       "(acquisition)."),
)}
#: Every catalogued lever, gated ones included (validation and expansion read this).
ALL_LEVERS: dict[str, Lever] = {**LEVERS, **GATED_LEVERS}
#: label -> gated lever, unique against LABELS too.
GATED_LABELS: dict[str, str] = {}
for _lever_obj in GATED_LEVERS.values():
    if _lever_obj.name in LEVERS:
        raise RuntimeError(f"gated lever {_lever_obj.name!r} is also in LEVERS")
    for _value in _lever_obj.values:
        _label = _lever_obj.label(_value)
        if _label in LABELS or _label in GATED_LABELS:
            raise RuntimeError(f"variant label {_label!r} of gated lever {_lever_obj.name!r} is already claimed")
        GATED_LABELS[_label] = _lever_obj.name
del _lever_obj, _value, _label
#: Gated levers that change the required-revenue account.
GATED_REGISTER_LEVERS: tuple[str, ...] = tuple(name for name, lever in GATED_LEVERS.items() if lever.register)


def lever_default(name: str) -> str:
    """The baseline value of any catalogued lever, gated or not."""
    return ALL_LEVERS[name].default


def gated_levers_in(specs) -> list[str]:
    """The gated levers any of these cases sets off its default, in catalogue order (their register columns)."""
    specs = list(specs)
    return [name for name in GATED_LEVERS
            if any(str(spec.settings.get(name, lever_default(name))) != lever_default(name) for spec in specs)]

#: Levers that change the required-revenue account itself.
REGISTER_LEVERS: tuple[str, ...] = tuple(name for name, lever in LEVERS.items() if lever.register)
#: The six structural axes of item D1, the default set that takes a bounding pair.
STRUCTURAL_LEVERS: tuple[str, ...] = (
    LEVER_PRE_2025_CFD, LEVER_CFD_STRIKE_BASIS, LEVER_CFD_AMORTISATION, LEVER_ROUTE_BOOKING,
    LEVER_NEW_BUILD_CONTRACTS, LEVER_LDES_FLOOR_BASIS,
)
#: Every lever at the private baseline's value.
DEFAULT_SETTINGS: dict[str, str] = {name: lever.default for name, lever in LEVERS.items()}
# Kept for modules written against the axis vocabulary.
AXES = LEVERS
CENTRAL_SETTINGS = DEFAULT_SETTINGS
REGISTER_AXES = REGISTER_LEVERS


@dataclass(frozen=True)
class CaseSpec:
    """One account row set: a scenario bundle at one point on the levers."""

    case_id: str
    base_case: str
    settings: Mapping[str, str]
    variant: str | None = None
    #: Where the case came from, for the summaries and the explorer (review 29 Sep
    #: 2026): ``kind`` (scenario, one_at_a_time, named, preset, bound, grid,
    #: bounding_pair), ``parent`` (the case it moves levers off), ``preset`` or
    #: ``bound`` name, and for a grid case ``grid`` and ``toggles``, its position vector
    #: ({switch token: 0 | 1 | None, selector: option token}; None = a conditional switch
    #: outside its selector option).  ``grid_memberships`` lists every grid position a
    #: case stands for, the parent's own all-base position included.
    meta: Mapping = field(default_factory=dict)

    def register_key(self) -> tuple:
        """Cases sharing this key share every required-revenue number (gated register levers included)."""
        return (tuple(self.settings[name] for name in REGISTER_LEVERS)
                + tuple(self.settings.get(name, lever_default(name)) for name in GATED_REGISTER_LEVERS))

    def is_public(self) -> bool:
        return self.settings.get(LEVER_NEW_BUILD_GENERATION, NEW_BUILD_PRIVATE) != NEW_BUILD_PRIVATE

    def acquires_legacy(self) -> bool:
        return self.settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE) != LEGACY_PRIVATE

    def market_bill(self) -> bool:
        """Projected through the market bill (not the acquisition tariff)."""
        return (not self.acquires_legacy()
                and self.settings.get(LEVER_DISPATCH_INSTITUTION, INSTITUTION_MARKET) == INSTITUTION_MARKET)


def base_case(case_id) -> str:
    """``public_ownership@strike_at_cost`` -> ``public_ownership``."""
    return str(case_id).split(VARIANT_SEPARATOR, 1)[0]


def is_variant(case_id) -> bool:
    return VARIANT_SEPARATOR in str(case_id)


def variant_label(case_id) -> str | None:
    parts = str(case_id).split(VARIANT_SEPARATOR, 1)
    return parts[1] if len(parts) == 2 else None


class LeverConfigError(ValueError):
    """A scenario or sensitivity declaration the code does not implement."""


AxisConfigError = LeverConfigError


def _refuse_retired(where: str, names) -> None:
    """A retired lever fails with a pointer at where its choice lives now."""
    retired = sorted(set(map(str, names)) & set(RETIRED_LEVERS))
    if retired:
        raise LeverConfigError(f"{where} names retired lever(s): "
                               + "; ".join(f"{name} ({RETIRED_LEVERS[name]})" for name in retired))


def _check_value(where: str, name: str, value: str) -> str:
    """One lever value, refused with a pointer when retired and with the supported list when unknown."""
    value = str(value)
    if (name, value) in RETIRED_VALUES:
        raise LeverConfigError(f"{where}: {name}={value!r} is {RETIRED_VALUES[(name, value)]}")
    if value not in ALL_LEVERS[name].values:
        raise LeverConfigError(f"{where}: {name}={value!r} is not implemented; supported: {list(ALL_LEVERS[name].values)}")
    return value


def parse_scenarios(block: Mapping | None) -> dict[str, dict[str, str]]:
    """Read ``public_ownership.scenarios`` into full bundles.

    Every scenario names a value for every lever (unnamed levers take the
    private baseline's value); ``private`` must be present and must sit at the
    baseline on every lever, because it is the comparator everything else is
    measured against.
    """
    block = dict(block or {})
    if not block:
        raise LeverConfigError("public_ownership.scenarios must declare at least the 'private' scenario")
    if PRIVATE_SCENARIO not in block:
        raise LeverConfigError("public_ownership.scenarios must include the 'private' baseline")
    bundles: dict[str, dict[str, str]] = {}
    for scenario, declared in block.items():
        if VARIANT_SEPARATOR in str(scenario):
            raise LeverConfigError(f"scenario id {scenario!r} may not contain {VARIANT_SEPARATOR!r}")
        declared = dict(declared or {})
        _refuse_retired(f"scenarios.{scenario}", declared)
        unknown = sorted(set(declared) - set(ALL_LEVERS))
        if unknown:
            raise LeverConfigError(f"scenarios.{scenario} names unknown levers {unknown}; supported: {sorted(ALL_LEVERS)}")
        bundle = dict(DEFAULT_SETTINGS)
        for name, value in declared.items():
            bundle[name] = _check_value(f"scenarios.{scenario}", name, value)
        if scenario == PRIVATE_SCENARIO:
            moved = sorted(name for name, value in bundle.items() if value != lever_default(name))
            if moved:
                raise LeverConfigError(f"the private scenario must sit at the baseline on every lever; it moves {moved}")
        bundles[scenario] = bundle
    return bundles


def parse_sensitivities(block: Mapping | None, bundles: Mapping[str, Mapping[str, str]]) -> dict:
    """Read ``public_ownership.sensitivities``.

    Returns ``{"structural": [...], "one_at_a_time": [...], "named": {...},
    "presets": {...}, "bounds": {...}, "toggle_grids": {...}, "bounding_pair": bool}``.
    ``structural`` levers take a bounding pair; ``one_at_a_time`` entries are lever
    names (every other value) or ``{lever: [values]}`` (those values only);
    ``named`` entries are ``{name: {"scenario": id, "extends": name?, lever: value, ...}}``.

    Review 29 Sep 2026 (user decision 5) adds three multi-lever mechanisms, each
    applied to the cases it names rather than as a full cross:

    * ``presets``: ``{name: {"applies_to": [case ids], "levers": {lever: value},
      "levers_by_case": {case id: {lever: value}}}}``, one multi-lever INCIDENCE
      position applied to several propositions; only :data:`INCIDENCE_LEVERS` may be
      moved, so the resource saving cannot change.  ``levers_by_case`` (30 Sep 2026)
      overrides ``levers`` for one target, or for every target of one scenario.  A
      preset with no levers (``as_designed``) documents the base and makes no case; a
      retired preset name (:data:`RETIRED_PRESETS`) is refused with a pointer.
    * ``bounds``: ``{"applies_to": [...], "best": {lever: value}, "worst": {...}}``,
      the most and least favourable coherent ASSUMPTION positions of each case, under
      its own incidence; only :data:`ASSUMPTION_LEVERS`.  A mapping holds one value
      per lever, so two values of one lever (mutually exclusive settings) cannot be
      declared together.
    * ``toggle_grids``: ``{"enabled": bool, "max_cases": int, "grids": {parent case:
      [axis, ...]}}``, every combination of every axis on a parent, for the explorer;
      an axis is a switch ``{token, lever, value[, when]}`` or a selector ``{selector,
      lever, options}`` (:func:`parse_grid_axes`), on :data:`GRID_LEVERS` only.
    """
    block = dict(block or {})
    bounding_pair = bool(block.pop("bounding_pair", True))
    structural = [x if isinstance(x, Mapping) else str(x) for x in (block.pop("structural", None) or [])]
    one_at_a_time = list(block.pop("one_at_a_time", None) or [])
    named = dict(block.pop("named", None) or {})
    presets = dict(block.pop("presets", None) or {})
    bounds = dict(block.pop("bounds", None) or {})
    grids = dict(block.pop("toggle_grids", None) or {})
    if block:
        raise LeverConfigError(
            f"sensitivities has unknown keys {sorted(block)}; expected structural, one_at_a_time, named, presets, "
            "bounds, toggle_grids, bounding_pair")

    def _lever_values(entry, where):
        if isinstance(entry, Mapping):
            if len(entry) != 1:
                raise LeverConfigError(f"{where}: a lever entry must be a name or {{name: [values]}}, got {entry!r}")
            (name, values), = entry.items()
            values = [str(v) for v in (values or [])]
        else:
            name, values = str(entry), None
        _refuse_retired(where, [name])
        if name not in ALL_LEVERS:
            raise LeverConfigError(f"{where} names unknown lever {name!r}; supported: {sorted(ALL_LEVERS)}")
        if values is None:
            values = list(ALL_LEVERS[name].values)
        for value in values:
            _check_value(f"{where}.{name}", name, value)
        return name, values

    resolved_structural = [_lever_values(entry, "sensitivities.structural") for entry in structural]
    resolved_one = [_lever_values(entry, "sensitivities.one_at_a_time") for entry in one_at_a_time]
    seen = [name for name, _ in resolved_structural + resolved_one]
    if len(set(seen)) != len(seen):
        raise LeverConfigError("sensitivities lists a lever twice across structural and one_at_a_time")

    labels_seen = {**LABELS, **GATED_LABELS}

    resolved_named: dict[str, dict] = {}
    for name, spec in named.items():
        spec = dict(spec or {})
        if VARIANT_SEPARATOR in str(name) or str(name) == BOUNDING_PAIR_LABEL or str(name) in labels_seen:
            raise LeverConfigError(f"named sensitivity {name!r} collides with a lever label or the bounding label")
        extends = spec.pop("extends", None)
        scenario = spec.pop("scenario", None)
        if extends is not None:
            if extends not in resolved_named:
                raise LeverConfigError(f"named sensitivity {name!r} extends {extends!r}, which must be declared before it")
            parent = resolved_named[extends]
            scenario = scenario or parent["scenario"]
            settings = dict(parent["settings"])
        else:
            if scenario not in bundles:
                raise LeverConfigError(f"named sensitivity {name!r} must name a declared scenario; got {scenario!r}")
            settings = dict(bundles[scenario])
        if scenario not in bundles:
            raise LeverConfigError(f"named sensitivity {name!r} must name a declared scenario; got {scenario!r}")
        _refuse_retired(f"named sensitivity {name!r}", spec)
        unknown = sorted(set(spec) - set(ALL_LEVERS))
        if unknown:
            raise LeverConfigError(f"named sensitivity {name!r} names unknown levers {unknown}")
        for lever, value in spec.items():
            settings[lever] = _check_value(f"named sensitivity {name!r}", lever, value)
        if settings == dict(bundles[scenario]):
            raise LeverConfigError(f"named sensitivity {name!r} moves no lever off scenario {scenario!r}")
        contracts = settings.get(LEVER_LEGACY_CONTRACTS, CONTRACTS_HONOURED)
        if contracts != CONTRACTS_HONOURED:
            # Item G3: a cancelled treatment needs plant to have been bought, and
            # never applies under full nationalisation (pre-2025 CfDs re-priced).
            if settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE) == LEGACY_PRIVATE:
                raise LeverConfigError(
                    f"named sensitivity {name!r}: {LEVER_LEGACY_CONTRACTS}={contracts!r} needs legacy_fleet to "
                    "acquire the legacy fleet")
            if settings.get(LEVER_PRE_2025_CFD, PRE_2025_PRIVATE) in PRE_2025_STATE_OWNED:
                raise LeverConfigError(
                    f"named sensitivity {name!r}: {LEVER_LEGACY_CONTRACTS} does not apply under full nationalisation, "
                    "where the state has bought the counterparty")
        resolved_named[str(name)] = {"scenario": scenario, "settings": settings, "parent": extends or scenario}

    def _positions(where, declared, allowed):
        declared = dict(declared or {})
        _refuse_retired(where, declared)
        unknown = sorted(set(declared) - set(ALL_LEVERS))
        if unknown:
            raise LeverConfigError(f"{where} names unknown levers {unknown}")
        outside = sorted(set(declared) - set(allowed))
        if outside:
            raise LeverConfigError(f"{where} may move only {sorted(allowed)}; it names {outside}")
        return {lever: _check_value(where, lever, value) for lever, value in declared.items()}

    def _targets(where, entry):
        targets = [str(t) for t in (entry or [])]
        if len(set(targets)) != len(targets):
            raise LeverConfigError(f"{where}.applies_to repeats a case")
        return targets

    resolved_presets: dict[str, dict] = {}
    for name, spec in presets.items():
        spec = dict(spec or {})
        where = f"sensitivities.presets.{name}"
        if str(name) in RETIRED_PRESETS:
            raise LeverConfigError(f"{where} is {RETIRED_PRESETS[str(name)]}")
        if VARIANT_SEPARATOR in str(name) or str(name) in labels_seen or str(name) in resolved_named:
            raise LeverConfigError(f"{where}: the name collides with a lever label or a named sensitivity")
        extra = sorted(set(spec) - {"applies_to", "levers", "levers_by_case", "description"})
        if extra:
            raise LeverConfigError(f"{where} has unknown keys {extra}; expected applies_to, levers, levers_by_case, "
                                   "description")
        applies_to = _targets(where, spec.get("applies_to"))
        # User decision 7 (30 Sep 2026): a preset may hold per-parent values (todays_rules
        # is each scenario's own old incidence).  Keyed by case id; a target without an
        # entry takes its base scenario's, then the common ``levers`` alone.
        by_case = {}
        for case_id, declared in dict(spec.get("levers_by_case") or {}).items():
            by_case[str(case_id)] = _positions(f"{where}.levers_by_case.{case_id}", declared, INCIDENCE_LEVERS)
        resolved_presets[str(name)] = {
            "applies_to": applies_to,
            "levers": _positions(where, spec.get("levers"), INCIDENCE_LEVERS),
            "levers_by_case": by_case,
            "description": str(spec.get("description") or ""),
        }

    resolved_bounds: dict = {}
    if bounds:
        extra = sorted(set(bounds) - {"applies_to", "best", "worst", "rule", "exclude"})
        if extra:
            raise LeverConfigError(f"sensitivities.bounds has unknown keys {extra}; expected applies_to, best, worst, "
                                   "rule, exclude")
        exclude = {str(k): [str(v) for v in (vs or [])] for k, vs in dict(bounds.get("exclude") or {}).items()}
        unknown_ex = sorted({v for vs in exclude.values() for v in vs} - set(LEVERS))
        if unknown_ex:
            raise LeverConfigError(f"sensitivities.bounds.exclude names unknown levers {unknown_ex}")
        resolved_bounds = {
            "applies_to": _targets("sensitivities.bounds", bounds.get("applies_to")),
            "cases": {kind: _positions(f"sensitivities.bounds.{kind}", bounds.get(kind),
                                       ASSUMPTION_LEVERS + BOUND_COMPARATOR_LEVERS)
                      for kind in ("best", "worst") if bounds.get(kind)},
            "rule": str(bounds.get("rule") or ""),
            # Levers a named case does not take in either bound (P3: the +1 pp stress covers the rate, not yield B).
            "exclude": exclude,
        }

    resolved_grids: dict = {"enabled": False, "max_cases": GRID_WARN_CASES, "grids": {}, "bound_grids": {}}
    if grids:
        extra = sorted(set(grids) - {"enabled", "max_cases", "grids", "bound_grids", "note"})
        if extra:
            raise LeverConfigError(f"sensitivities.toggle_grids has unknown keys {extra}; expected enabled, max_cases, "
                                   "grids, bound_grids")
        resolved_grids["enabled"] = bool(grids.get("enabled", False))
        resolved_grids["max_cases"] = int(grids.get("max_cases", GRID_WARN_CASES))
        for parent, toggles in dict(grids.get("grids") or {}).items():
            resolved = parse_grid_axes(f"sensitivities.toggle_grids.grids.{parent}", toggles, _positions)
            if resolved:
                resolved_grids["grids"][str(parent)] = resolved
        for parent, toggles in dict(grids.get("bound_grids") or {}).items():
            resolved = parse_grid_axes(f"sensitivities.toggle_grids.bound_grids.{parent}", toggles, _positions,
                                       allowed=BOUND_GRID_LEVERS)
            if resolved:
                resolved_grids["bound_grids"][str(parent)] = resolved
    return {
        "structural": resolved_structural, "one_at_a_time": resolved_one,
        "named": resolved_named, "presets": resolved_presets, "bounds": resolved_bounds,
        "toggle_grids": resolved_grids, "bounding_pair": bounding_pair,
    }


#: A toggle grid above this many cases in total is reported (the explorer's owner asked
#: for a warning, not a refusal: the count is configurable per run).
GRID_WARN_CASES = 128
GRID_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: The two kinds of grid axis (30 Sep 2026).  A ``switch`` is one lever's one value,
#: off (0) or on (1); a ``selector`` is one lever over several values, the parent's
#: own value being the base option.  A switch may be conditional (``when``) on a
#: selector declared before it: it exists only in that selector's named option(s)
#: and is ``None`` in the position vector elsewhere, so it never repeats a case.
GRID_SWITCH, GRID_SELECTOR = "switch", "selector"


def parse_grid_axes(where: str, entries, positions=None, allowed: Sequence[str] = GRID_LEVERS) -> list[dict]:
    """Read one grid's axes, in declared order.

    ``{token, lever, value[, when]}`` is a switch; ``{selector, lever, options}`` is a
    selector whose ``options`` map an option token to a lever value.  ``when`` is
    ``{selector name: option token or [option tokens]}``.  Tokens and selector names
    share one namespace (they key the position vector); each lever appears once.
    ``positions`` is the value checker (``parse_sensitivities``' ``_positions``); left
    out, values are checked against :data:`LEVERS` directly.  ``allowed`` is the levers
    the grid may move: :data:`GRID_LEVERS` for a toggle grid, :data:`BOUND_GRID_LEVERS`
    for a bound grid.
    """
    allowed = tuple(allowed)

    def _check(lever, value, levers=None):
        levers = allowed if levers is None else levers
        if positions is not None:
            (lever_, value_), = positions(where, {lever: value}, levers).items()
            return lever_, value_
        if lever not in levers:
            raise LeverConfigError(f"{where} may move only {sorted(levers)}; it names {lever!r}")
        return lever, _check_value(where, lever, value)

    resolved: list[dict] = []
    for entry in list(entries or []):
        entry = dict(entry or {})
        if "selector" in entry:
            if set(entry) != {"selector", "lever", "options"}:
                raise LeverConfigError(f"{where}: a selector is {{selector, lever, options}}, got {sorted(entry)}")
            name = str(entry["selector"])
            if not GRID_TOKEN.fullmatch(name):
                raise LeverConfigError(f"{where}: selector {name!r} must be lower-case letters, digits and hyphens")
            options = dict(entry["options"] or {})
            if len(options) < 2:
                raise LeverConfigError(f"{where}: selector {name!r} needs at least two options")
            resolved_options, extra = {}, {}
            for option, value in options.items():
                option = str(option)
                if not GRID_TOKEN.fullmatch(option):
                    raise LeverConfigError(f"{where}: option {option!r} of {name!r} must be lower-case letters, "
                                           "digits and hyphens")
                if isinstance(value, Mapping):
                    # A coupled option (user decision, 2 Oct 2026: P1's "cut to cost" also costs
                    # over the plant's life): {selector's lever: value, other lever: value, ...}.
                    value = dict(value)
                    if str(entry["lever"]) not in value:
                        raise LeverConfigError(f"{where}: option {option!r} of {name!r} must set the selector's "
                                               f"own lever {entry['lever']!r}")
                    coupled = {}
                    for other, other_value in value.items():
                        other, other_value = _check(other, other_value,
                                                    allowed if other == str(entry["lever"]) else allowed + GRID_COUPLED_LEVERS)
                        coupled[other] = other_value
                    primary = coupled.pop(str(entry["lever"]))
                    if coupled:
                        extra[option] = coupled
                    resolved_options[option] = primary
                    continue
                lever, value = _check(entry["lever"], value)
                resolved_options[option] = value
            keys = [(v, tuple(sorted(extra.get(o, {}).items()))) for o, v in resolved_options.items()]
            if len(set(keys)) != len(keys):
                raise LeverConfigError(f"{where}: selector {name!r} lists one lever value twice")
            resolved.append({"kind": GRID_SELECTOR, "token": name, "lever": str(entry["lever"]),
                             "options": resolved_options, **({"extra": extra} if extra else {})})
            continue
        if not {"token", "lever", "value"} <= set(entry) or set(entry) - {"token", "lever", "value", "when"}:
            raise LeverConfigError(f"{where}: each toggle is {{token, lever, value[, when]}} or "
                                   f"{{selector, lever, options}}, got {sorted(entry)}")
        token = str(entry["token"])
        if not GRID_TOKEN.fullmatch(token):
            raise LeverConfigError(f"{where}: token {token!r} must be lower-case letters, digits and hyphens")
        lever, value = _check(entry["lever"], entry["value"])
        when = {}
        for selector, wanted in dict(entry.get("when") or {}).items():
            declared = next((a for a in resolved if a["kind"] == GRID_SELECTOR and a["token"] == str(selector)), None)
            if declared is None:
                raise LeverConfigError(f"{where}: switch {token!r} is conditional on {selector!r}, which must be a "
                                       "selector declared before it")
            wanted = [str(w) for w in (wanted if isinstance(wanted, (list, tuple)) else [wanted])]
            unknown = sorted(set(wanted) - set(declared["options"]))
            if unknown:
                raise LeverConfigError(f"{where}: switch {token!r} names unknown options {unknown} of {selector!r}")
            when[str(selector)] = wanted
        resolved.append({"kind": GRID_SWITCH, "token": token, "lever": lever, "value": value, "when": when})
    tokens = [a["token"] for a in resolved]
    levers_used = [a["lever"] for a in resolved]
    levers_used += sorted({lever for a in resolved for coupled in (a.get("extra") or {}).values() for lever in coupled})
    if len(set(tokens)) != len(tokens) or len(set(levers_used)) != len(levers_used):
        raise LeverConfigError(f"{where}: a token or a lever appears twice (a toggle is one lever's one value)")
    return resolved


def _grid_axes(toggles: Sequence[Mapping]) -> list[dict]:
    """Toggles as parsed, or in the pre-30-Sep shape ``{token, lever, value}`` (a plain switch)."""
    out = []
    for t in toggles:
        t = dict(t)
        t.setdefault("kind", GRID_SELECTOR if "options" in t else GRID_SWITCH)
        t.setdefault("when", {})
        t.setdefault("extra", {})
        out.append(t)
    return out


def grid_base_options(toggles: Sequence[Mapping], settings: Mapping[str, str]) -> dict[str, str]:
    """``{selector: option}``: the option each selector takes at the parent (its own lever value)."""
    out = {}
    for axis in _grid_axes(toggles):
        if axis["kind"] != GRID_SELECTOR:
            continue
        current = settings.get(axis["lever"], DEFAULT_SETTINGS[axis["lever"]])
        base = [option for option, value in axis["options"].items() if value == current
                and all(settings.get(k, DEFAULT_SETTINGS[k]) == v for k, v in (axis["extra"].get(option) or {}).items())]
        if len(base) != 1:
            raise LeverConfigError(f"selector {axis['token']!r} ({axis['lever']}) has no option at the parent's value "
                                   f"{current!r}; options: {axis['options']}")
        out[axis["token"]] = base[0]
    return out


def grid_positions(toggles: Sequence[Mapping], settings: Mapping[str, str] | None = None,
                   ) -> list[tuple[dict, list[str], dict]]:
    """Every position of a grid but the parent's own: ``[(vector, id pieces, {lever: value})]``.

    The vector holds every axis: a switch as 0/1 (``None`` where its ``when`` fails),
    a selector as its option token.  The id pieces are the switches that are on and
    ``<selector>-<option>`` for each selector off its base option, in declared order.
    Ordered by the number of pieces, then by each axis's choice in declared order
    (base first), so the case list is stable when the configuration is.
    """
    axes = _grid_axes(toggles)
    base = grid_base_options(axes, settings or DEFAULT_SETTINGS)
    partial: list[tuple[dict, list[int]]] = [({}, [])]
    for axis in axes:
        grown = []
        for vector, rank in partial:
            if axis["kind"] == GRID_SELECTOR:
                options = [base[axis["token"]]] + [o for o in axis["options"] if o != base[axis["token"]]]
                grown += [(dict(vector, **{axis["token"]: o}), rank + [i]) for i, o in enumerate(options)]
                continue
            live = all(vector.get(selector) in wanted for selector, wanted in axis["when"].items())
            if not live:
                grown.append((dict(vector, **{axis["token"]: None}), rank + [0]))
            else:
                grown += [(dict(vector, **{axis["token"]: bit}), rank + [bit]) for bit in (0, 1)]
        partial = grown
    out = []
    for vector, rank in partial:
        pieces, selectors, switches = [], {}, {}
        for axis in axes:
            choice = vector[axis["token"]]
            if axis["kind"] == GRID_SELECTOR:
                if choice != base[axis["token"]]:
                    pieces.append(f"{axis['token']}-{choice}")
                    selectors[axis["lever"]] = axis["options"][choice]
                    selectors.update(axis["extra"].get(choice) or {})
            elif choice:
                pieces.append(axis["token"])
                switches[axis["lever"]] = axis["value"]
        if pieces:
            # Selectors first: a switch's own requirement (the pass-through needs the
            # guarantee) is then met when apply_positions tests it.  The order key puts
            # the positions that move earlier axes first, as itertools.combinations
            # orders a plain binary grid (so the pre-30-Sep grids keep their order).
            key = (len(pieces), tuple((-int(choice > 0), choice) for choice in rank))
            out.append((vector, pieces, {**selectors, **switches}, key))
    out.sort(key=lambda item: item[3])
    return [(vector, pieces, positions) for vector, pieces, positions, _ in out]


def grid_case_count(grids: Mapping[str, Sequence[Mapping]], settings_by_parent: Mapping[str, Mapping] | None = None) -> int:
    """How many positions the grids hold beyond their parents (before existing cases are matched)."""
    settings_by_parent = settings_by_parent or {}
    return sum(len(grid_positions(toggles, settings_by_parent.get(parent))) for parent, toggles in grids.items())


def derived_case_id(parent_id: str, suffix: str) -> str:
    """``public_ownership`` + ``todays_rules`` -> ``public_ownership@todays_rules``;
    ``public_ownership@fully_nationalised_market_value`` + ``best_case`` ->
    ``public_ownership@fully_nationalised_market_value_best_case``."""
    return f"{parent_id}_{suffix}" if VARIANT_SEPARATOR in str(parent_id) else f"{parent_id}{VARIANT_SEPARATOR}{suffix}"


def apply_positions(settings: Mapping[str, str], positions: Mapping[str, str]) -> tuple[dict, list[str]]:
    """Move each lever to its position where it bites (:meth:`Lever.applies`); leave it at central where not.

    Levers are applied in the declared order against the settings as they
    stand, so a lever whose requirement another position satisfies sees it.
    Returns the settings and the levers actually moved.
    """
    out, moved = dict(settings), []
    for name, value in positions.items():
        if out.get(name, lever_default(name)) == value:
            continue
        if ALL_LEVERS[name].applies(value, out):
            out[name] = value
            moved.append(name)
    return out, moved


def grid_combinations(toggles: Sequence[Mapping]) -> list[tuple[dict, list[str]]]:
    """Every non-empty subset of the toggles, in a fixed order: ``[(toggle vector, tokens on)]``.

    Binary switches only (the pre-30-Sep grids); :func:`grid_positions` expands
    selectors and conditional switches and is what :func:`expand_cases` uses.

    Ordered by the number of toggles on, then by the declared order, so the case
    list is stable when the configuration is.
    """
    tokens = [str(t["token"]) for t in toggles]
    out = []
    for size in range(1, len(tokens) + 1):
        for on in itertools.combinations(range(len(tokens)), size):
            vector = {token: int(i in on) for i, token in enumerate(tokens)}
            out.append((vector, [tokens[i] for i in on]))
    return out


def expand_cases(
    scenarios: Mapping[str, Mapping[str, str]], sensitivities: Mapping | None = None,
) -> list[CaseSpec]:
    """Scenarios, one lever at a time, named positions, presets, bounds, toggle grids, then the bounding pair.

    The order is the order the stages account and report in.  A sensitivity is
    generated only where its lever bites (:meth:`Lever.applies`); the bounding
    pair puts every structural lever on its first alternative at once and is
    added for a scenario only when at least two of them bite there, because with
    one it would repeat a one-at-a-time variant.  A preset or a bound that moves
    no lever off its parent makes no case; a grid toggle that cannot move its
    parent is refused.
    """
    bundles = {scenario: dict(DEFAULT_SETTINGS, **dict(bundle)) for scenario, bundle in scenarios.items()}
    if PRIVATE_SCENARIO not in bundles:
        raise ValueError("scenarios must include the 'private' baseline")
    sens = dict(sensitivities or {})
    structural = list(sens.get("structural") or [])
    one_at_a_time = list(sens.get("one_at_a_time") or [])
    named = dict(sens.get("named") or {})
    presets = dict(sens.get("presets") or {})
    bounds = dict(sens.get("bounds") or {})
    grids = dict(sens.get("toggle_grids") or {})
    bounding_pair = bool(sens.get("bounding_pair", True))

    specs = [CaseSpec(scenario, scenario, dict(bundle), None, {"kind": "scenario", "parent": None})
             for scenario, bundle in bundles.items()]
    seen = {spec.case_id for spec in specs}

    def _add(case_id, scenario, settings, label, meta):
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r}")
        specs.append(CaseSpec(case_id, scenario, dict(settings), label, dict(meta)))
        seen.add(case_id)

    for name, values in structural + one_at_a_time:
        lever = ALL_LEVERS[name]
        for value in values:
            for scenario, bundle in bundles.items():
                if not lever.applies(value, bundle):
                    continue
                _add(f"{scenario}{VARIANT_SEPARATOR}{lever.label(value)}", scenario,
                     dict(bundle, **{name: value}), lever.label(value),
                     {"kind": "one_at_a_time", "parent": scenario, "lever": name, "value": value})

    for label, spec in named.items():
        parent = spec.get("parent") or spec["scenario"]
        _add(f"{spec['scenario']}{VARIANT_SEPARATOR}{label}", spec["scenario"], spec["settings"], label,
             {"kind": "named", "parent": parent if parent == spec["scenario"]
              else f"{spec['scenario']}{VARIANT_SEPARATOR}{parent}"})

    def _parent(case_id, where):
        by_id = {spec.case_id: spec for spec in specs}
        if case_id not in by_id:
            raise LeverConfigError(f"{where} applies to {case_id!r}, which is not a configured case")
        return by_id[case_id]

    for name, preset in presets.items():
        targets = list(preset.get("applies_to") or [])
        by_case = dict(preset.get("levers_by_case") or {})
        stray = sorted(set(by_case) - set(targets) - {base_case(t) for t in targets})
        if stray:
            raise LeverConfigError(f"sensitivities.presets.{name}.levers_by_case names {stray}, which it does not "
                                   "apply to (neither a target nor a target's scenario)")
        for target in targets:
            parent = _parent(target, f"sensitivities.presets.{name}")
            positions = dict(preset.get("levers") or {})
            # Per-parent values: the target's own entry, else its scenario's.
            positions.update(by_case.get(target, by_case.get(parent.base_case, {})))
            if parent.acquires_legacy():
                # Under an acquisition the state owns the counterparty: the contract form, the
                # recycling routes and the market deficits have no meaning, so they stay central.
                positions = {k: v for k, v in positions.items() if k not in MARKET_ONLY_INCIDENCE_LEVERS}
            settings, moved = apply_positions(parent.settings, positions)
            if not moved:
                continue
            case_id = derived_case_id(parent.case_id, name)
            _add(case_id, parent.base_case, settings, case_id.split(VARIANT_SEPARATOR, 1)[1],
                 {"kind": "preset", "parent": parent.case_id, "preset": name, "levers_moved": moved})

    for kind, positions in (bounds.get("cases") or {}).items():
        comparator_positions = {k: v for k, v in positions.items() if k in BOUND_COMPARATOR_LEVERS}
        if comparator_positions:
            # The bound's comparator: private at the same point on the all_cases levers it moves.
            settings, moved = apply_positions(bundles[PRIVATE_SCENARIO], comparator_positions)
            if moved:
                _add(f"{PRIVATE_SCENARIO}{VARIANT_SEPARATOR}{kind}_case_comparator", PRIVATE_SCENARIO, settings,
                     f"{kind}_case_comparator",
                     {"kind": "bound_comparator", "parent": PRIVATE_SCENARIO, "bound": kind, "levers_moved": moved})
        for target in bounds.get("applies_to") or []:
            parent = _parent(target, "sensitivities.bounds")
            excluded = set((bounds.get("exclude") or {}).get(target) or [])
            settings, moved = apply_positions(parent.settings, {k: v for k, v in positions.items() if k not in excluded})
            if not moved:
                continue
            case_id = derived_case_id(parent.case_id, f"{kind}_case")
            _add(case_id, parent.base_case, settings, case_id.split(VARIANT_SEPARATOR, 1)[1],
                 {"kind": "bound", "parent": parent.case_id, "bound": kind, "levers_moved": moved,
                  "rule": bounds.get("rule", "")})

    if grids.get("enabled"):
        # Two kinds of exact grid share one expansion: the toggle grids (named after
        # their parent) and the bound grids (user request, 2 Oct 2026: the best and
        # worst assumptions switched one by one, named parent + BOUND_GRID_SUFFIX,
        # case ids ``<parent>_bgrid_...``).
        kinds = [(target, toggles, "grids") for target, toggles in (grids.get("grids") or {}).items()]
        kinds += [(target, toggles, "bound_grids") for target, toggles in (grids.get("bound_grids") or {}).items()]
        grid_parents = {(target, kind): _parent(target, f"sensitivities.toggle_grids.{kind}")
                        for target, _, kind in kinds}
        total = sum(len(grid_positions(toggles, grid_parents[(target, kind)].settings))
                    for target, toggles, kind in kinds)
        if total > int(grids.get("max_cases", GRID_WARN_CASES)):
            warnings.warn(f"the toggle grids expand to {total} cases, above toggle_grids.max_cases "
                          f"({grids.get('max_cases', GRID_WARN_CASES)}); every one is projected", stacklevel=2)
        # A combination another stage already accounts (a one-at-a-time, named or preset
        # case at exactly these settings) is not projected twice: the existing case
        # carries the grid membership instead.  Indexed once, not searched per position.
        by_settings: dict = {}
        for spec in specs:
            by_settings.setdefault((spec.base_case, tuple(sorted(spec.settings.items()))), spec)
        for target, toggles, kind in kinds:
            parent = grid_parents[(target, kind)]
            bound_grid = kind == "bound_grids"
            grid_name = parent.case_id + (BOUND_GRID_SUFFIX if bound_grid else "")
            prefix = "bgrid_" if bound_grid else "grid_"
            axes = _grid_axes(toggles)
            where = f"sensitivities.toggle_grids.{kind}.{target}"
            base = grid_base_options(axes, parent.settings)
            # Every switch and every non-base option must move the parent, a conditional
            # switch under the selector option(s) it is conditional on.
            for axis in axes:
                if axis["kind"] == GRID_SELECTOR:
                    for option, value in axis["options"].items():
                        coupled = axis["extra"].get(option) or {}
                        if option != base[axis["token"]] and not all(
                                LEVERS[lever].applies(v, parent.settings) for lever, v in {axis["lever"]: value, **coupled}.items()):
                            raise LeverConfigError(f"{where}: option {axis['token']}={option} ({axis['lever']}={value}) "
                                                   "moves nothing on that case")
                    continue
                conditioned = dict(parent.settings)
                for selector, wanted in axis["when"].items():
                    selector_axis = next(a for a in axes if a["token"] == selector)
                    conditioned[selector_axis["lever"]] = selector_axis["options"][wanted[0]]
                    conditioned.update(selector_axis["extra"].get(wanted[0]) or {})
                if not LEVERS[axis["lever"]].applies(axis["value"], conditioned):
                    raise LeverConfigError(f"{where}: toggle {axis['token']!r} ({axis['lever']}={axis['value']}) "
                                           "moves nothing on that case")
            schema = {axis["token"]: ({"selector": axis["lever"], "options": dict(axis["options"]),
                                       "base": base[axis["token"]],
                                       **({"extra": dict(axis["extra"])} if axis["extra"] else {})}
                                      if axis["kind"] == GRID_SELECTOR
                                      else {axis["lever"]: axis["value"], **({"when": axis["when"]} if axis["when"] else {})})
                      for axis in axes}
            base_vector = {axis["token"]: (base[axis["token"]] if axis["kind"] == GRID_SELECTOR
                                           else (0 if all(base.get(s) in w for s, w in axis["when"].items()) else None))
                           for axis in axes}
            # The parent is the grid's all-base position: it carries that membership too,
            # so every position of the explorer's grid maps to exactly one case.
            parent.meta.setdefault("grid_memberships", []).append({"grid": grid_name, "toggles": base_vector})
            for vector, pieces, positions in grid_positions(axes, parent.settings):
                settings, moved = apply_positions(parent.settings, positions)
                if sorted(moved) != sorted(positions):
                    raise LeverConfigError(f"{where}: position {pieces} moves {sorted(moved)}, not "
                                           f"{sorted(positions)}; a grid must be exact")
                if bound_grid:
                    # House rule 3: a position that moves a comparator lever is measured
                    # against private at the same point on it, which must be a case.
                    comparator_positions = {k: v for k, v in positions.items() if k in BOUND_COMPARATOR_LEVERS}
                    if comparator_positions:
                        private_settings, private_moved = apply_positions(bundles[PRIVATE_SCENARIO],
                                                                          comparator_positions)
                        key = (PRIVATE_SCENARIO, tuple(sorted(private_settings.items())))
                        if private_moved and key not in by_settings:
                            label = "bgrid_comparator_" + "_".join(f"{k}-{v}" for k, v in sorted(comparator_positions.items()))
                            _add(f"{PRIVATE_SCENARIO}{VARIANT_SEPARATOR}{label}", PRIVATE_SCENARIO, private_settings,
                                 label, {"kind": "bound_comparator", "parent": PRIVATE_SCENARIO,
                                         "bound": "grid", "levers_moved": private_moved})
                            by_settings[key] = specs[-1]
                membership = {"grid": grid_name, "toggles": vector}
                twin = by_settings.get((parent.base_case, tuple(sorted(settings.items()))))
                if twin is not None:
                    twin.meta.setdefault("grid_memberships", []).append(membership)
                    continue
                case_id = derived_case_id(parent.case_id, prefix + "_".join(pieces))
                _add(case_id, parent.base_case, settings, case_id.split(VARIANT_SEPARATOR, 1)[1],
                     {"kind": "grid", "parent": parent.case_id, **membership,
                      "grid_memberships": [membership], "toggle_levers": schema})
                by_settings[(parent.base_case, tuple(sorted(settings.items())))] = specs[-1]

    if bounding_pair and structural:
        already = {(spec.base_case, tuple(sorted(spec.settings.items()))) for spec in specs}
        for scenario, bundle in bundles.items():
            settings = dict(bundle)
            moved = 0
            for name, values in structural:
                lever = LEVERS[name]
                # The bound is "every structural lever on its first alternative
                # to the PRIVATE baseline", never back towards it.  Since item L2
                # a scenario bundle may already hold a structural alternative
                # (P2 holds at_cost, economic_life, booked), and the baseline
                # value then "applies" in the one-at-a-time sense -- that is a
                # reverse sensitivity, not a bound, so it is skipped here.
                alternative = next(
                    (v for v in values if v != lever.default and lever.applies(v, bundle)), None)
                if alternative is not None:
                    settings[name] = alternative
                    moved += 1
            if moved < 2 or (scenario, tuple(sorted(settings.items()))) in already:
                continue
            _add(f"{scenario}{VARIANT_SEPARATOR}{BOUNDING_PAIR_LABEL}", scenario, settings, BOUNDING_PAIR_LABEL,
                 {"kind": "bounding_pair", "parent": scenario})
    return specs


def case_catalogue(specs: Iterable[CaseSpec]) -> dict[str, dict]:
    """``{case_id: {kind, parent, preset | bound | grid, toggles, levers_moved}}``: where each case came from.

    Written beside the ledger and the household totals so the explorer can look up
    a combination (a grid case's toggle vector) or a preset's parent without
    parsing case ids.
    """
    out = {}
    for spec in specs:
        meta = dict(spec.meta or {})
        meta.setdefault("kind", "scenario" if spec.variant is None else "case")
        meta.setdefault("parent", None if spec.variant is None else spec.base_case)
        meta["base_case"] = spec.base_case
        meta["market_bill"] = spec.market_bill()
        out[spec.case_id] = meta
    return out


def all_cases_levers() -> list[str]:
    """The levers that move the private baseline, so they change the comparator."""
    return [name for name, lever in ALL_LEVERS.items() if lever.scope == SCOPE_ALL and name != LEVER_NEW_BUILD_GENERATION]


all_cases_axes = all_cases_levers


def comparator_for_settings(case_id: str, settings_by_case: Mapping[str, Mapping[str, str]]) -> str:
    """The private case at the same point on *every* ``all_cases`` lever.

    All of them have to match. A private case that agrees on one and differs on
    another is the wrong column, and matching on only the first silently
    compares a case with a baseline it does not share.
    """
    settings = settings_by_case.get(case_id) or {}
    wanted = {name: settings.get(name, lever_default(name)) for name in all_cases_levers()}
    for other, other_settings in settings_by_case.items():
        if base_case(other) != PRIVATE_SCENARIO:
            continue
        if all(other_settings.get(name, lever_default(name)) == value for name, value in wanted.items()):
            return other
    return PRIVATE_SCENARIO


def comparator_case_id(spec: CaseSpec, specs: Iterable[CaseSpec]) -> str:
    """The private case this one is measured against (house rule 3)."""
    by_case = {candidate.case_id: dict(candidate.settings) for candidate in specs}
    by_case.setdefault(spec.case_id, dict(spec.settings))
    return comparator_for_settings(spec.case_id, by_case)


def settings_columns(specs: Iterable[CaseSpec]) -> dict[str, dict[str, str]]:
    """``{case_id: {lever: value}}``, the mapping the register writes as columns."""
    return {spec.case_id: dict(spec.settings) for spec in specs}


def lever_setting(row_or_map: Mapping, lever_name: str) -> str:
    """Read one lever from a register row, falling back to the baseline value.

    Registers written before a lever existed carry no column for it, so the
    baseline value is the only safe reading.
    """
    value = row_or_map.get(lever_name)
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return lever_default(lever_name)
    return str(value)


axis_setting = lever_setting


def case_settings(rows, lever_name: str | None = None):
    """The lever point a case's register rows sit at, or one lever of it.

    Every row of a case carries the same settings, so the first non-null wins;
    a register written before a lever existed has no column and falls back to
    the baseline value.  ``rows`` is a DataFrame of one case.
    """
    names = [lever_name] if lever_name else list(LEVERS)
    out = {}
    for name in names:
        if name not in rows.columns:
            out[name] = lever_default(name)
            continue
        values = _distinct_text(rows[name])
        if len(values) > 1:
            raise ValueError(f"a case carries more than one {name}: {sorted(values)}")
        out[name] = str(values[0]) if len(values) else lever_default(name)
    return out[lever_name] if lever_name else out


def _distinct_text(column) -> list:
    """``column.dropna().astype(str).unique()``, in order of appearance (the values :func:`case_settings` reads).

    F20 phase 1 (13 Oct 2026): a lever column read through ``register_io`` is a categorical
    of text, and its distinct values are read off the codes (the text of each category
    present, in the order the rows first carry it) instead of converting every row to text.
    """
    import numpy as np
    import pandas as pd

    if isinstance(column.dtype, pd.CategoricalDtype) and column.cat.categories.inferred_type == "string":
        codes = column.cat.codes.to_numpy()
        present = pd.unique(codes[codes >= 0])
        return list(np.asarray(column.cat.categories, dtype=object)[present])
    return list(column.dropna().astype(str).unique())


def spec_for_row(settings: Mapping[str, str]) -> CaseSpec:
    """A CaseSpec-like view of a settings mapping, for the ``is_public`` style predicates."""
    return CaseSpec("", "", {name: str(settings.get(name, DEFAULT_SETTINGS[name])) for name in LEVERS})


# ── compatibility with the item-D1 axis declaration ──────────────────────────

#: The D1 boundary axis and its values, as older declarations and tests spell them.
LEGACY_AXIS_ALIASES: dict[str, str] = {"ownership_boundary": LEVER_PRE_2025_CFD}
LEGACY_VALUE_ALIASES: dict[str, dict[str, str]] = {
    LEVER_PRE_2025_CFD: {"ownership_excluding_pre_2025_cfd": PRE_2025_PRIVATE,
                         "ownership_all_cfd_repriced": PRE_2025_PUBLIC_REPRICED},
}


def _unalias_axes(declared: Mapping) -> dict:
    out = {}
    for name, spec in dict(declared or {}).items():
        lever = LEGACY_AXIS_ALIASES.get(str(name), str(name))
        spec = dict(spec or {})
        values = LEGACY_VALUE_ALIASES.get(lever, {})
        if "central" in spec:
            spec["central"] = values.get(str(spec["central"]), spec["central"])
        if spec.get("alternatives"):
            spec["alternatives"] = [values.get(str(v), v) for v in spec["alternatives"]]
        out[lever] = spec
    return out


def parse_axes(block: Mapping | None) -> tuple[dict[str, dict], bool]:
    """Read the retired ``sensitivity_axes`` shape into ``{lever: {central, alternatives}}``.

    Kept so tests and older callers that build axis declarations keep working;
    the repository config uses ``scenarios`` and ``sensitivities`` (item K1).
    """
    block = dict(block or {})
    bounding_pair = bool(block.pop("bounding_pair", True))
    declared = _unalias_axes(block.pop("axes", None) or {})
    if block:
        raise LeverConfigError(f"sensitivity_axes has unknown keys {sorted(block)}; expected 'axes' and 'bounding_pair'")
    unknown = sorted(set(declared) - set(LEVERS))
    if unknown:
        raise LeverConfigError(f"unknown sensitivity axes {unknown}; supported: {sorted(LEVERS)}")
    resolved: dict[str, dict] = {}
    labels_seen: dict[str, str] = {}
    for name, lever in LEVERS.items():
        spec = dict(declared.get(name) or {})
        extra = sorted(set(spec) - {"central", "alternatives"})
        if extra:
            raise LeverConfigError(f"sensitivity_axes.axes.{name} has unknown keys {extra}")
        central = str(spec.get("central", lever.default))
        if central not in lever.values:
            raise LeverConfigError(f"sensitivity_axes.axes.{name}.central={central!r} is not implemented; supported: {list(lever.values)}")
        alternatives = [str(value) for value in (spec.get("alternatives") or [])]
        if len(set(alternatives)) != len(alternatives):
            raise LeverConfigError(f"sensitivity_axes.axes.{name}.alternatives repeats an entry")
        for value in alternatives:
            if value not in lever.values:
                raise LeverConfigError(f"sensitivity_axes.axes.{name}.alternatives entry {value!r} is not implemented; supported: {list(lever.values)}")
            if value == central:
                raise LeverConfigError(f"sensitivity_axes.axes.{name}.alternatives entry {value!r} is the central value")
            label = lever.label(value)
            if label in labels_seen:
                raise LeverConfigError(f"variant label {label!r} is claimed by both {labels_seen[label]!r} and {name!r}; "
                                       "the case id carries the label alone, so labels must be unique")
            labels_seen[label] = name
        resolved[name] = {"central": central, "alternatives": alternatives}
    return resolved, bounding_pair


LEGACY_CASE_BUNDLES: dict[str, dict[str, str]] = {
    # The three market cases of items D1 to D4, as bundles.
    "private": {},
    "public_coordination": {LEVER_NEW_BUILD_GENERATION: NEW_BUILD_STATE_STAKE,
                            LEVER_PUBLIC_RETURN: RETURN_CEPA_DECOMPOSITION},
    "public_control": {LEVER_NEW_BUILD_GENERATION: NEW_BUILD_STATE_STAKE,
                       LEVER_PUBLIC_RETURN: RETURN_CEPA_DECOMPOSITION},
    "public_ownership": {LEVER_NEW_BUILD_GENERATION: NEW_BUILD_PUBLIC, LEVER_PUBLIC_RETURN: RETURN_PUBLIC_RATE},
    # Item N2's fourth case: every GB asset state-owned.  Its in_construction value
    # was inert until 29 Sep 2026 (no register row carried the vintage), so the
    # bundle now takes the central reading, HPC bought as an operating plant.
    "public_nationalised": {LEVER_NEW_BUILD_GENERATION: NEW_BUILD_PUBLIC, LEVER_PUBLIC_RETURN: RETURN_PUBLIC_RATE,
                           LEVER_PRE_2025_CFD: PRE_2025_PUBLIC_REPRICED, LEVER_LEGACY_FLEET: LEGACY_ACQUIRE_ALL},
}


def expand_axis_cases(cases: Sequence[str], axes: Mapping[str, Mapping], *, bounding_pair: bool = True) -> list[CaseSpec]:
    """The item-D1 expansion over case names, mapped onto bundles.

    ``axes`` is the :func:`parse_axes` result: the central value of every lever
    is applied to every case (an axis central other than the baseline moved
    every case at once) and each alternative becomes a one-at-a-time
    sensitivity where it bites.  Only the case names in
    :data:`LEGACY_CASE_BUNDLES` are known.
    """
    cases = list(cases)
    if PRIVATE_SCENARIO not in cases:
        raise ValueError("cases must include the 'private' baseline")
    unknown = sorted(set(cases) - set(LEGACY_CASE_BUNDLES))
    if unknown:
        raise ValueError(f"unknown cases {unknown}; known: {sorted(LEGACY_CASE_BUNDLES)}")
    axes = _unalias_axes(axes)
    _refuse_retired("sensitivity axes", axes)
    unknown_axes = sorted(set(axes) - set(LEVERS))
    if unknown_axes:
        raise ValueError(f"unknown sensitivity axes {unknown_axes}; supported: {sorted(LEVERS)}")
    central = {name: str((axes.get(name) or {}).get("central", lever.default)) for name, lever in LEVERS.items()}
    for name, value in central.items():
        if value not in LEVERS[name].values:
            raise ValueError(f"axis {name} central {value!r} is not implemented; supported: {list(LEVERS[name].values)}")
    scenarios = {}
    for case_id in cases:
        bundle = dict(central)
        bundle.update(LEGACY_CASE_BUNDLES[case_id])
        if case_id == PRIVATE_SCENARIO:
            # The baseline carries an axis central like every other case, but
            # the scenario-level structure demands it sits at the default; the
            # register key still reflects the moved central through settings.
            bundle = dict(central)
        scenarios[case_id] = bundle
    structural = [(name, [v for v in (axes.get(name) or {}).get("alternatives", [])])
                  for name in LEVERS if (axes.get(name) or {}).get("alternatives")]
    specs = expand_cases(scenarios, {"structural": structural, "one_at_a_time": [], "named": {},
                                     "bounding_pair": bounding_pair})
    # Item N2's rule, kept for this shape: the nationalised case took no variant.
    return [spec for spec in specs if not (spec.variant and spec.acquires_legacy())]
