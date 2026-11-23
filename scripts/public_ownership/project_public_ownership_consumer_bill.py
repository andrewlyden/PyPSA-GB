"""WP-C: one physical run, several bills.

Each ownership case is projected through the live consumer-bill projector
(``scripts/consumer_bill/project_future_bill.py``) with its own CfD strike table
from WP-B and the extra lines the public account routes to it (WP-D, WP-E,
WP-I, WP-J):

* ``ldes_cap_and_floor``: floor top-up for long-duration storage, in every case
  including private.  Since item D4(c) (26 Sep 2026) it is a line of the
  forecast bill itself, so the private headline is one number; this script
  passes each case's own top-up to that line (``ldes_topup_for``) instead of
  adding it, and the private case must reproduce the forecast bill's value.
  It stays in ``NEW_LINES`` as a line that may move between cases;
* ``public_cfd_at_cost`` (W1 as the contract it is): the state's renewable and
  hydro new build on a two-way CfD with strike at cost, one policy line; or
  ``public_cost_recovery_contracting``, the same money on the wholesale line;
* ``public_surplus_rebate``: the rebate or levy-offset rules, a negative policy row;
* ``public_portfolio_shortfall_levy``: a negative amount to route, funded on the
  bill under ``levy_on_eligible_supply``.

The private case must reproduce the forecast case's saved components exactly on
every shared line; the CfD line must move by exactly the R1 strike adjustments;
each new line must move by exactly the public-account amount it carries; VAT is
checked once.  Gross = before recycling and shortfall rows; net = after.
Labelled sensitivities (``<case>@<eligibility>``) are projected gross only.

Outputs per reference-run group:

* ``{run_group}_consumer_bill_by_ownership_case.csv``: projector columns plus ``case_id``;
* ``{run_group}_consumer_bill_by_ownership_case_summary.json``;
* ``{run_group}_consumer_bill_by_ownership_case_reconciliation.csv``: saving by route
  beside the bill change it produced.

This file is used behind a Snakemake ``script:`` directive, so it must not
begin with a ``from __future__`` import.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.consumer_bill.line_recovery import entry_for, household_to_system_gbp, system_to_household_gbp
from scripts.consumer_bill.bill_groupings import group_of as bill_group_of
from scripts.consumer_bill.project_future_bill import SUPPLIER_LINE, conversion_denominators, run_projection
from scripts.public_ownership.build_cfd_strike_adjustments import load_bill_strike_table
from scripts.public_ownership.finance import NATIONALISED_CASE  # noqa: F401
from scripts.public_ownership.cfd_repricing import (
    IDENTITY_TOLERANCE_GBP, ROUTE_CAPACITY_MARKET, ROUTE_CFD_BILL, ROUTE_DISPATCHABLE_CFD,
    ROUTE_MERCHANT, ROUTE_STORAGE, ROUTES, base_case, case_strike_table, is_variant,
)
from scripts.public_ownership.cfd_repricing import ROUTE_DPA_BILL
from scripts.public_ownership.levers import (
    AXIS_NEW_BUILD_CONTRACTS, AXIS_ROUTE_BOOKING, BECCS_REMOVAL_BILL_LEVY, CENTRAL_SETTINGS,
    LEGACY_LEVIES_TO_TAXATION, LEVER_BECCS_REMOVAL_FUNDING, LEVER_LEGACY_LEVY_FUNDING, LEVER_NETWORKS,
    LEVER_PRE_2025_CFD, LEVER_WHOLESALE_PRICING, NETWORKS_PRIVATE, PRICING_PAY_AS_BID_CEILING,
    GATED_LEVERS, comparator_for_settings, lever_default, spec_for_row,
)
from scripts.public_ownership.network_finance_account import (
    adjustments_for_treatment, break_even_summary, incidence_amounts, select_rows,
)
from scripts.public_ownership import capacity_market_state_bids as cm_state_bids
from scripts.public_ownership import capacity_procurement
from scripts.public_ownership import sizewell_c_rab
from scripts.public_ownership import bill_recovery

VALUE = "annual_gbp_per_3100kwh_ex_vat"
LINE_LDES = "ldes_cap_and_floor"
LINE_W1 = "public_cost_recovery_contracting"
LINE_CFD_AT_COST = "public_cfd_at_cost"
LINE_REBATE = "public_surplus_rebate"
LINE_SHORTFALL = "public_portfolio_shortfall_levy"
# D2: the two route bookings, as their own rows rather than netted silently into
# an external line, so each still states its own quantity, price source and
# denominator (house rule 1).  The line they adjust is reported beside them.
LINE_CM_BOOKED = "capacity_market_public_financing_adjustment"
LINE_CCS_BOOKED = "allocated_ccs_public_financing_adjustment"
# D4: merchant new build on a contract at required revenue, in every case.
LINE_NEW_BUILD = "new_build_contracts_at_cost"
# Item J6 / N6: the networks lever's adjustment to the TNUoS and DUoS lines, as
# their own rows (the projector refuses a duplicate of an existing line).
LINE_TNUOS_ADJ = "tnuos_public_financing_adjustment"
LINE_DUOS_ADJ = "duos_public_financing_adjustment"
NETWORK_LINES = {LINE_TNUOS_ADJ: "tnuos", LINE_DUOS_ADJ: "duos"}
# Item K3: the pay-as-bid ceiling, a bound on the wholesale line under the private scenario alone.
LINE_PAY_AS_BID = "pay_as_bid_ceiling"
# Review 29 Sep 2026 (user decisions 2, 3 and 5): the repaired levy offset (domestic
# customers' RO, FiT and WHD, over domestic supply), the levy that recovers the
# state's own deficits under state_deficit_funding = bill_levy, the domestic legacy
# levies moved to taxation (a background lever, every case alike) and BECCS's
# carbon-removal CfD recovered on bills (a sensitivity, every case alike).
LINE_OFFSET = "public_surplus_offset_domestic_levies"
LINE_DEFICIT = "public_state_deficit_levy"
DOMESTIC_LEVY_LINES = {"renewables_obligation": "legacy_levy_ro_to_taxation",
                       "feed_in_tariffs": "legacy_levy_fit_to_taxation",
                       "warm_home_discount": "legacy_levy_whd_to_taxation"}
LINE_BECCS_LEVY = "beccs_carbon_removal_levy"
# OW10 (was G4): Sizewell C's RAB levy with its return at the public rate, in the public cases.
LINE_SIZEWELL = sizewell_c_rab.ADJUSTMENT_LINE
# F19 decision 5: the battery top-up at each case's own cost against the private case's the
# forecast bill carries (its own row beside battery_revenue_top_up, over final demand like it).
LINE_BATTERY_TOP_UP_ADJ = "battery_revenue_top_up_public_financing_adjustment"
# F9: the 2030/31 T-4 re-cleared with the state's own plant bidding at its own cost of capital (its own row
# beside capacity_market, recovered as that line is, with its household peak factor).
LINE_CM_STATE_BIDS = cm_state_bids.LINE
# F27 item 6: existing capacity on cost-based contracts instead of the clearing price (its own row beside
# capacity_market, recovered as that line is).
LINE_CM_COST_BASED = capacity_procurement.LINE
RECYCLING_LINES = (LINE_W1, LINE_CFD_AT_COST, LINE_REBATE, LINE_OFFSET, LINE_SHORTFALL, LINE_DEFICIT)
BOOKED_ROUTE_LINES = (LINE_CM_BOOKED, LINE_CCS_BOOKED)
BACKGROUND_LINES = tuple(DOMESTIC_LEVY_LINES.values()) + (LINE_BECCS_LEVY,)
NEW_LINES = ((LINE_LDES, LINE_NEW_BUILD) + RECYCLING_LINES + BOOKED_ROUTE_LINES + tuple(NETWORK_LINES)
             + (LINE_PAY_AS_BID,) + BACKGROUND_LINES + (LINE_SIZEWELL, LINE_BATTERY_TOP_UP_ADJ, LINE_CM_STATE_BIDS,
                                                        LINE_CM_COST_BASED))
# F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA re-priced to the state's cost where the state owns
# it under the pre-2025 option (route R1_dpa_repricing_bill), its own row beside remaining_ccs.
LINE_DPA_REPRICED = "remaining_ccs_public_financing_adjustment"
NEW_LINES = NEW_LINES + (LINE_DPA_REPRICED,)
#: FES 2050 decision 73: the Capacity Market design's move on the CM line (the residual auction, or under an
#: acquisition the targeted contract, against private's full auction). Kept out of NEW_LINES, which every
#: 2030 summary lists: it is added only to a case whose financing summary carries the design (2050 only).
LINE_CM_DESIGN = "capacity_market_design_adjustment"
#: Forecast-bill lines route R1 re-prices: the CfD line and the BECCS CfD carved
#: out of it (allocated_ccs, item MR7), which share the eligible-sales denominator.
R1_LINES = ("cfd", "allocated_ccs")
SOURCE_CLASS = "public_ownership_bill_link"
ROUTE_BOOKING_BOOKED = "booked"
NEW_BUILD_AT_COST = "at_cost"



def strike_basis_offset_saving(adjustments: pd.DataFrame, case_id: str) -> float:
    """FES 2050 decision 54: the CfD line saving a case's rate-path strike offsets give (system GBP 2024).

    ``offset x the energy the bill pays at the entry`` summed over the case's entries; zero when the
    adjustments carry no offsets (every 2030 file).
    """
    if "strike_basis_offset_gbp_per_mwh_2024" not in adjustments.columns:
        return 0.0
    rows = adjustments.loc[adjustments["case_id"].eq(case_id)]
    return float((pd.to_numeric(rows["strike_basis_offset_gbp_per_mwh_2024"], errors="coerce").fillna(0.0)
                  * pd.to_numeric(rows["eligible_dispatch_mwh"], errors="coerce").fillna(0.0)).sum())

def _two_volume(summary) -> bool:
    """True when a projector summary is on F18's two-volume recovery basis."""
    block = (summary or {}).get("recovery_basis")
    return isinstance(block, dict) and block.get("basis") == "loss_inclusive_two_volume"


def _household(system_gbp_2024, denominator_mwh, factor, mwh):
    return float(system_gbp_2024) * float(factor) / float(denominator_mwh) * float(mwh)


def _system(household_gbp_2025, denominator_mwh, factor, mwh):
    return float(household_gbp_2025) * float(denominator_mwh) / float(factor) / float(mwh)


BOOKED_ROUTE_SPEC = {
    LINE_CM_BOOKED: {
        "stack_group": "system_balancing", "line": "capacity_market", "route": ROUTE_CAPACITY_MARKET,
        # User decision 12: the booking the account actually takes (floored at the
        # route total under state_deficit_funding = exchequer); an account written
        # before 29 Sep 2026 has only the sensitivity, which was booked whole.
        "amount_key": "capacity_market_booked_gbp", "fallback_key": "capacity_market_sensitivity_gbp",
        "denominator": "final_demand",
        "label": "Capacity Market, state financing saving booked",
        "route_label": "the new-build gas and batteries the auction pays",
    },
    LINE_CCS_BOOKED: {
        "stack_group": "policy", "line": "allocated_ccs", "route": ROUTE_DISPATCHABLE_CFD,
        "amount_key": "ccs_levy_sensitivity_gbp", "denominator": "eligible",
        "label": "Allocated CCS levy, state financing saving booked",
        "route_label": "the BECCS capacity the dispatchable CfD pays",
    },
}


#: F27 item 1: the line the DPA re-pricing moves (its row is recovered as this line is).
DPA_LINE = "remaining_ccs"


def dpa_household(system_gbp, *, line_recovery, eligible_mwh, factor, mwh) -> float:
    """F27 item 1: a system amount on the DPA line as household GBP, through remaining_ccs's own recovery entry.

    On a D1 bill (no entries) the line is recovered over CfD eligible supplier sales, as the CfD levy.
    """
    if line_recovery is not None:
        return system_to_household_gbp(entry_for(line_recovery, DPA_LINE), float(system_gbp), mwh * 1000.0)
    return _household(float(system_gbp), eligible_mwh, factor, mwh)


def dpa_system(household_gbp, *, line_recovery, eligible_mwh, factor, mwh) -> float:
    """The inverse of :func:`dpa_household`."""
    if line_recovery is not None:
        return household_to_system_gbp(entry_for(line_recovery, DPA_LINE), float(household_gbp), mwh * 1000.0)
    return _system(float(household_gbp), eligible_mwh, factor, mwh)


def cm_household_allocation_factor(assumptions) -> float:
    """The Capacity Market line's household peak factor (item MR4.4), 1.0 where the bill declares none.

    The supplier charge falls on November-to-February weekday peak demand, where
    household demand is concentrated, so a household pays 1.348 times a flat
    energy share of the CM cost; the bill's CM line applies it, and a booking
    against that line must move it by the same rule (item B6).
    """
    derivation = (((assumptions or {}).get("sourced_externals") or {}).get("capacity_market") or {}).get("derivation") or {}
    return float(derivation.get("household_allocation_factor", 1.0))


def booked_route_amounts(account_case, *, factor, final_demand_mwh, eligible_mwh, mwh, cm_line, ccs_line,
                         cm_household_factor=1.0, line_recovery=None):
    """Household amounts the ``booked`` route setting takes off each route line.

    Returns ``{}`` unless the case sits on ``route_booking = booked``.  The
    booking is capped at the line it adjusts, because a route cannot give back
    more than the bill charges for it; the reconciliation applies the same cap,
    so the identity is against what was actually booked.  The Capacity Market
    booking is allocated to the household as the CM line itself is, with its
    peak factor ``cm_household_factor`` (item B6), not as a flat energy share.

    ``line_recovery`` (F19, 7 Oct 2026): the private bill's entries on the two-volume basis
    (F18). Each booking is then recovered on the entry of the line it adjusts: the Capacity
    Market's (GSP charging volume, its own allocation factor) and the allocated CCS levy's.
    Without it (a D1 bill: 2050) the old denominators, unchanged.
    """
    if not account_case or axis_settings(account_case)[AXIS_ROUTE_BOOKING] != ROUTE_BOOKING_BOOKED:
        return {}
    caps = {LINE_CM_BOOKED: cm_line, LINE_CCS_BOOKED: ccs_line}
    denominators = {"final_demand": final_demand_mwh, "eligible": eligible_mwh}
    household_factors = {LINE_CM_BOOKED: float(cm_household_factor), LINE_CCS_BOOKED: 1.0}
    amounts = {}
    for line_id, spec in BOOKED_ROUTE_SPEC.items():
        amount = account_case.get(spec["amount_key"])
        if amount is None and spec.get("fallback_key"):
            amount = account_case.get(spec["fallback_key"])
        if amount is None:
            continue
        if line_recovery is not None:
            household = system_to_household_gbp(entry_for(line_recovery, spec["line"]), float(amount), mwh * 1000.0)
        else:
            household = _household(float(amount), denominators[spec["denominator"]], factor, mwh) * household_factors[line_id]
        cap = caps[line_id]
        # Task SETTLE2050: a line that pays consumers back (a 2050 CfD line of paybacks) charges
        # nothing to give back, so the cap is never below zero; every 2030 line is positive.
        amounts[line_id] = min(household, max(float(cap), 0.0)) if cap is not None else household
    return amounts


def axis_settings(account_case):
    """The sensitivity-axis point this case sits at, central where unrecorded."""
    declared = (account_case or {}).get("sensitivity_axes") or {}
    settings = {axis: str(declared.get(axis, central)) for axis, central in CENTRAL_SETTINGS.items()}
    # Task Y14 (patch 3): a year-gated lever (2050) the case sets off its default joins its
    # levers, so an all_cases one finds its own comparator; no 2030 case declares one.
    for name in GATED_LEVERS:
        if name in declared and str(declared[name]) != lever_default(name):
            settings[name] = str(declared[name])
    return settings


def network_adjustment_rows(account_case, network_lines, *, account_source):
    """Item J6 / N6: TNUoS and DUoS adjustments for the case's networks lever, as their own rows.

    ``network_lines`` is the network finance account's lines table (one row per
    sector and treatment, GBP per household ex VAT, negative for a saving).
    """
    levers = axis_settings(account_case)
    treatment = levers.get(LEVER_NETWORKS, NETWORKS_PRIVATE)
    if treatment == NETWORKS_PRIVATE or network_lines is None:
        return []
    # The case's own rows (premium, rate basis, pass-through) and its network_incidence.
    amounts = adjustments_for_treatment(network_lines, treatment, levers)
    rows = []
    for line_id, bill_line in NETWORK_LINES.items():
        amount = float(amounts.get(bill_line, 0.0))
        if amount == 0.0:
            continue
        rows.append({
            "component_id": line_id, "stack_group": "network",
            "label": (f"{bill_line.upper()} adjustment: the wires under the state's {treatment.replace('_', ' ')}"),
            "annual_gbp_per_3100kwh_ex_vat": amount,
            "source_class": SOURCE_CLASS,
            "method": (f"networks = {treatment}, network_incidence = {levers.get('network_incidence')}: the "
                       f"{bill_line.upper()} line moves by the part of the network finance account's saving on the "
                       f"Regulatory Asset Value that reaches the bill ({'financing wedge at the public rate less the premium annuity, plus the tax allowance' if treatment == 'acquired' else 'notional gearing x the guarantee debt compression on the debt issued or refinanced 2026-2030'}), "
                       "recovered over final demand like the line it adjusts (item N6 / J6, user decisions 5-7)."),
            "source": account_source,
        })
    return rows


def pay_as_bid_row(account_case, *, factor, final_demand_mwh, mwh, account_source):
    """Item K3: the most pay-as-bid could remove from the wholesale line, as a bounding row."""
    if not account_case or not account_case.get("pay_as_bid_ceiling_applies"):
        return []
    ceiling = account_case.get("pay_as_bid_ceiling_gbp")
    if ceiling is None or float(ceiling) == 0.0:
        return []
    return [{
        "component_id": LINE_PAY_AS_BID, "stack_group": "hedged_wholesale",
        "label": "Pay-as-bid ceiling (every accepted GB generator paid its short-run cost)",
        "annual_gbp_per_3100kwh_ex_vat": -_household(float(ceiling), final_demand_mwh, factor, mwh),
        "source_class": SOURCE_CLASS,
        "method": (f"Sum over resolved non-CfD GB generators of max(0, wholesale revenue - fuel and carbon - variable O&M) "
                   f"= GBP{float(ceiling) / 1e6:,.0f}M in the private settlement (CfD plant is made whole by its top-up, "
                   "so its wholesale margin is not a consumer saving), removed from the wholesale line over "
                   "final demand. A ceiling, never a case (plan Part 7.3): offers do not stay at cost, the missing money "
                   "returns through the Capacity Market, and ownership confers no settlement rule."),
        "source": account_source,
    }]


def sizewell_terms(account_case, *, settings, network_lines):
    """OW10: ``(state_share, state_rate)`` for one case, or ``None`` when its Sizewell C levy does not move.

    The levy moves only in a public case whose share of its own new build
    (``state_new_build_share`` in the public account) exceeds the share of Sizewell C's
    capital the state already funds, only when the switch is on, and never where the
    case's own state holds Sizewell C's RAB rows (``rab_support_state_owned_gbp``; the levy is then replaced, as
    in the tariff). D4 (user, 14 Nov 2026; independent review, 15 Nov 2026): a later year's register books the RAB
    on private rows in S1 and S2, which keep the levy, so the adjustment applies there too; an account written before
    the held figure existed is read as holding it (the old rule).
    ``state_rate`` is the public rate plus the construction premium (F19 decision 1).
    """
    if not settings or not settings.get("enabled") or not account_case:
        return None
    if "rab_support_gbp" in account_case and float(
            account_case.get("rab_support_state_owned_gbp", account_case["rab_support_gbp"]) or 0.0) != 0.0:
        return None
    share = float(account_case.get("state_new_build_share") or 0.0)
    # F19 decision 1: only the capital beyond the state's existing share moves (none under P1).
    if sizewell_c_rab.refinanced_share(settings, share) <= 0.0:
        return None
    levers = axis_settings(account_case)
    return share, sizewell_c_rab.public_rate(network_lines, levers) + sizewell_c_rab.premium(settings, levers)


def sizewell_row(account_case, *, private_line, settings, network_lines, account_source):
    """OW10: the Sizewell C RAB adjustment as its own row (the projector refuses a duplicate of the line)."""
    terms = sizewell_terms(account_case, settings=settings, network_lines=network_lines)
    if terms is None:
        return []
    share, rate = terms
    amount = sizewell_c_rab.household_adjustment(private_line, settings, state_share=share, rate=rate)
    if amount == 0.0:
        return []
    return [{
        "component_id": LINE_SIZEWELL, "stack_group": "policy",
        "label": "Sizewell C RAB: the return on its asset base at the public rate",
        "annual_gbp_per_3100kwh_ex_vat": amount,
        "source_class": SOURCE_CLASS,
        "method": sizewell_c_rab.method_note(settings, state_share=share, rate=rate),
        "source": account_source,
    }]


def battery_top_up_change(account_case, private_account) -> float:
    """F19 decision 5: system GBP the case's battery top-up differs from the private case's (0 off the basis).

    Only where the register sized each case's top-up at its owner's cost (the account's
    ``battery_top_up_owner_cost``); the forecast bill's line is the private case's top-up.
    """
    if not account_case or not (account_case.get("battery_top_up_owner_cost")
                                and (private_account or {}).get("battery_top_up_owner_cost")):
        return 0.0
    return (float(account_case.get("battery_revenue_top_up_gbp") or 0.0)
            - float((private_account or {}).get("battery_revenue_top_up_gbp") or 0.0))


def battery_top_up_row(account_case, *, private_account, factor, final_demand_mwh, mwh, account_source):
    """F19 decision 5: the battery top-up at the case's own cost, as its own row (none when it does not move)."""
    change = battery_top_up_change(account_case, private_account)
    if abs(change) < 1e-6:
        return []
    return [{
        "component_id": LINE_BATTERY_TOP_UP_ADJ, "stack_group": "policy",
        "label": "Battery revenue top-up: at the owner's cost",
        "annual_gbp_per_3100kwh_ex_vat": _household(change, final_demand_mwh, factor, mwh),
        "source_class": SOURCE_CLASS,
        "method": ("F19 decision 5: each battery row is topped up to its own owner's required revenue (the state's "
                   "rows at the state's cost, a joint venture's at its rate) instead of the private case's "
                   "shortfall paid to every owner; the difference against the forecast bill's line, recovered over "
                   "final demand like it. The Capacity Market booking no longer carries batteries."),
        "source": account_source,
    }]


def cm_state_bids_share(account_case, private_account) -> float:
    """F9: the share of the calendar-2030 Capacity Market cost the case's re-cleared 2030/31 T-4 moves, against private's."""
    block = (account_case or {}).get(cm_state_bids.CONFIG_KEY)
    if not block:
        return 0.0
    base = ((private_account or {}).get(cm_state_bids.CONFIG_KEY) or {}).get("calendar_2030_cost_change_share", 0.0)
    return float(block["calendar_2030_cost_change_share"]) - float(base or 0.0)


def cm_state_bids_row(account_case, *, private_account, cm_line, account_source):
    """F9: the Capacity Market line moved by the case's re-cleared 2030/31 T-4, as its own row (none when nil)."""
    share = cm_state_bids_share(account_case, private_account)
    amount = float(cm_line) * share
    if abs(amount) < 1e-12:
        return []
    return [{
        "component_id": LINE_CM_STATE_BIDS, "stack_group": "system_balancing",
        "label": "Capacity Market: the 2030/31 T-4 with the state's plant bidding at its own cost",
        "annual_gbp_per_3100kwh_ex_vat": amount,
        "source_class": SOURCE_CLASS,
        "method": cm_state_bids.method_note(account_case[cm_state_bids.CONFIG_KEY]),
        "source": account_source,
    }]


def capacity_procurement_saving(account_case) -> float:
    """F27 item 6: system GBP (model price year) the case's cost-based contracts take off the Capacity Market line."""
    block = (account_case or {}).get(capacity_procurement.CONFIG_KEY)
    return float(block.get("saving_private_owners_gbp") or 0.0) if block else 0.0


def capacity_procurement_household(system_gbp, *, private_extra, cm_factor) -> float:
    """System GBP as a household amount on the Capacity Market line's own recovery (D1 bill: final demand x peak factor)."""
    recovery = private_extra.get("line_recovery") if private_extra.get("two_volume") else None
    if recovery is not None and "capacity_market" in recovery:
        return system_to_household_gbp(entry_for(recovery, "capacity_market"), float(system_gbp),
                                       float(private_extra["mwh"]) * 1000.0)
    return _household(float(system_gbp), private_extra["final_demand_mwh"], private_extra["factor"],
                      private_extra["mwh"]) * float(cm_factor)


def capacity_procurement_row(account_case, *, private_extra, cm_factor, account_source):
    """F27 item 6: the Capacity Market line less the cost-based contracts' saving, as its own row (none when nil)."""
    saving = capacity_procurement_saving(account_case)
    if abs(saving) < 1e-6:
        return []
    return [{
        "component_id": LINE_CM_COST_BASED, "stack_group": "system_balancing",
        "label": "Capacity Market: existing plant on cost-based contracts instead of the clearing price",
        "annual_gbp_per_3100kwh_ex_vat": -capacity_procurement_household(saving, private_extra=private_extra,
                                                                         cm_factor=cm_factor),
        "source_class": SOURCE_CLASS,
        "method": capacity_procurement.method_note(account_case[capacity_procurement.CONFIG_KEY]),
        "source": account_source,
    }]


def capacity_market_design_row(case_id, financing_summary, *, private_extra, cm_factor, account_source) -> list[dict]:
    """FES 2050 decision 73: the CM line's move under the case's Capacity Market design, as its own row.

    The private case's CM line pays the full auction; the case's pays its design's
    (:mod:`capacity_market_design`). The row is the design's cost change (system GBP, model
    money) on the Capacity Market line's own recovery, as F27 item 6's row is. Empty without
    the design columns (every 2030 financing summary) or where the cost does not move.
    """
    from scripts.public_ownership.capacity_market_design import case_moves

    moves = case_moves(financing_summary, case_id) if case_id != "private" else None
    if not moves or abs(moves["line_change_gbp"]) < 1e-6:
        return []
    return [{
        "component_id": LINE_CM_DESIGN, "stack_group": "system_balancing",
        "label": "Capacity Market under the public-ownership design (decision 73)",
        "annual_gbp_per_3100kwh_ex_vat": capacity_procurement_household(
            moves["line_change_gbp"], private_extra=private_extra, cm_factor=cm_factor),
        "source_class": SOURCE_CLASS,
        "method": (f"Design {moves['design']}: the state's de-rated capacity leaves the auction (or, under an "
                   f"acquisition, the CM becomes a targeted contract for what the state does not own); the rest "
                   f"clears at {moves['price_gbp_per_kw_year_2025']:.2f} GBP/kW/yr against "
                   f"{moves['price_private_gbp_per_kw_year_2025']:.2f}, so the auction pays GBP"
                   f"{moves['line_cost_gbp'] / 1e6:,.0f}M against GBP{moves['line_cost_private_gbp'] / 1e6:,.0f}M "
                   "(model money). Recovered as the Capacity Market line is."),
        "source": account_source,
    }]


def cm_recovery_options(cm_household_line: float, *, cm_factor: float, final_demand_mwh: float, factor: float,
                        mwh: float, cm_entry=None) -> dict:
    """F11 (user decision, 6 Oct 2026): the Capacity Market line under two other recovery rules, reported only.

    The headline keeps today's supplier charge on winter-peak demand (the household peak
    factor). (b) Flat per MWh across all supply: the household line divides by the factor
    and business pays the difference. (c) Taxation: the line leaves the bill and its whole
    system cost goes to the Exchequer (the taxpayer column of the household totals).
    Neither is an ownership saving.
    """
    line = float(cm_household_line)
    # F19 (7 Oct 2026): on the two-volume basis the line's system cost is through its own entry
    # (GSP charging volume and the peak factor); a D1 bill keeps final demand over the factor.
    if cm_entry is not None:
        system = household_to_system_gbp(cm_entry, line, float(mwh) * 1000.0)
    else:
        system = _system(line, final_demand_mwh, factor, mwh) / float(cm_factor) if cm_factor else 0.0
    flat = line / float(cm_factor) if cm_factor else line
    return {
        "household_line_peak_weighted_gbp_per_3100kwh_ex_vat": line,
        "household_peak_factor": float(cm_factor),
        "system_cost_gbp_2024": system,
        "b_flat_per_mwh": {"household_line_gbp_per_3100kwh_ex_vat": flat,
                           "bill_change_gbp_per_3100kwh_ex_vat": flat - line,
                           "household_share_change": (1.0 / float(cm_factor) - 1.0) if cm_factor else 0.0},
        "c_taxation": {"household_line_gbp_per_3100kwh_ex_vat": 0.0, "bill_change_gbp_per_3100kwh_ex_vat": -line,
                       "exchequer_cost_gbp_2024": system},
        "note": ("F11 reporting only: the headline keeps the peak-weighted supplier charge. (b) recovers the same total "
                 "flat per MWh of supply, so households pay less and business more; (c) moves it to taxation. "
                 "Neither counts towards an ownership saving."),
    }


def ldes_topup_for(account_case):
    """The LDES floor top-up (system GBP, model price year) one case's bill line carries (item D4(c)).

    The line belongs to the forecast bill (project_future_bill.py); this is the
    amount the projector is told to put on it for this case: the case's own
    account top-up, zero when the case has no LDES floor, and ``None`` without
    an account (the projector then refuses, rather than assume one).
    """
    if not account_case:
        return None
    if not account_case.get("ldes_cap_and_floor_enabled"):
        return 0.0
    return float(account_case.get("ldes_floor_topup_gbp", 0.0))


DOMESTIC_VOLUME_KEY = "domestic_electricity_supply_mwh"


def domestic_recovery(domestic_mwh) -> dict:
    """F19 (7 Oct 2026): the recovery a row built over domestic supply declares (scripts/consumer_bill/line_recovery.py).

    The levy offset and the levies moved to taxation are built over Ofgem's domestic
    supply volume, not the volume of the line they relieve, so their entries say so.
    """
    if not domestic_mwh:
        return {}
    return {"line_recovery": (DOMESTIC_VOLUME_KEY, 1.0, 1.0,
                              "domestic customers only: spread over domestic electricity supply (Ofgem's WHD volume), "
                              "no loss uplift"),
            "line_recovery_volumes": {DOMESTIC_VOLUME_KEY: float(domestic_mwh)}}


def background_rows(account_case, *, factor, eligible_mwh, mwh, domestic_mwh, levy_lines, account_source):
    """Two background levers that move every case alike, private included (a change of comparator).

    ``legacy_levy_funding = domestic_to_taxation``: domestic customers' Renewables
    Obligation, Feed-in Tariff and Warm Home Discount move to general taxation, so a
    household pays none of them (each line's household value comes off), and the
    Exchequer takes on the domestic share, ``line / household MWh x domestic supply``
    (the Exchequer ledger books it).  ``beccs_removal_funding = bill_levy``: BECCS's
    carbon-removal CfD is recovered over eligible supply instead of by the Exchequer.
    """
    axes = axis_settings(account_case)
    rows = []
    if axes.get(LEVER_LEGACY_LEVY_FUNDING) == LEGACY_LEVIES_TO_TAXATION:
        if domestic_mwh is None or levy_lines is None:
            raise ValueError("legacy_levy_funding = domestic_to_taxation needs the domestic supply volume and the "
                             "private levy lines")
        for line, row_id in DOMESTIC_LEVY_LINES.items():
            household_value = float(levy_lines.get(line, 0.0))
            if household_value == 0.0:
                continue
            rows.append({
                "component_id": row_id, "stack_group": "policy",
                "label": f"{line.replace('_', ' ').title()}: domestic share moved to general taxation",
                "annual_gbp_per_3100kwh_ex_vat": -household_value,
                "source_class": SOURCE_CLASS,
                "method": (f"A domestic customer pays no {line} (GBP{household_value:.2f} per household): the domestic "
                           f"share, GBP{_system(household_value, domestic_mwh, factor, mwh) / 1e6:,.0f}M over "
                           f"{domestic_mwh / 1e6:,.2f} TWh of domestic supply, is funded from general taxation "
                           "(a background choice, never an ownership effect)."),
                "source": account_source,
                **domestic_recovery(domestic_mwh),
            })
    if axes.get(LEVER_BECCS_REMOVAL_FUNDING) == BECCS_REMOVAL_BILL_LEVY:
        amount = float(account_case.get("carbon_removal_cfd_revenue_gbp") or 0.0)
        if amount:
            rows.append({
                "component_id": LINE_BECCS_LEVY, "stack_group": "policy",
                "label": "BECCS carbon-removal contract recovered on bills",
                "annual_gbp_per_3100kwh_ex_vat": _household(amount, eligible_mwh, factor, mwh),
                "source_class": SOURCE_CLASS,
                "method": (f"Power BECCS's carbon-removal CfD, (1 - electricity leg share) x required revenue = "
                           f"GBP{amount / 1e6:,.0f}M, recovered over CfD eligible supplier sales instead of by the "
                           "Exchequer (a sensitivity; the central case funds it from the Exchequer in every case)."),
                "source": account_source,
            })
    return rows


def offset_split(routed, *, factor, domestic_mwh, mwh, levy_lines, levies_on_bills=True):
    """The repaired offset rule: the routed surplus offsets domestic RO, FiT and WHD, the rest is rebated.

    Returns ``(offset_system, rebate_system)``: the offset is capped at the domestic
    customers' share of the three lines (system GBP over domestic supply), zero when
    they have already moved to taxation.
    """
    cap = 0.0
    if levies_on_bills and domestic_mwh and levy_lines:
        cap = sum(_system(float(levy_lines.get(line, 0.0)), domestic_mwh, factor, mwh) for line in DOMESTIC_LEVY_LINES)
    offset = min(max(float(routed), 0.0), cap)
    return offset, max(float(routed), 0.0) - offset


def extra_lines(case_id, account_case, *, factor, final_demand_mwh, eligible_mwh, mwh, account_source,
                cm_line=None, ccs_line=None, gross_only=False, network_lines=None, cm_household_factor=1.0,
                domestic_mwh=None, levy_lines=None, line_recovery=None, two_volume=False):
    """The public-account lines one case adds to the projector, as extra components.

    ``gross_only`` (sensitivity variants) drops the recycling rows (the LDES
    line, which does not depend on the eligibility rule, is the forecast bill's
    own line since item D4(c): see ``ldes_topup_for``).  ``cm_line``
    and ``ccs_line`` are the private case's Capacity Market and allocated CCS
    levy lines; a booking (D2) may not take more off a line than the line holds.
    ``network_lines`` is the network finance account's lines table (item J6).
    """
    rows = []
    if not account_case:
        return rows
    axes = axis_settings(account_case)
    rows.extend(pay_as_bid_row(account_case, factor=factor, final_demand_mwh=final_demand_mwh, mwh=mwh,
                               account_source=account_source))
    rows.extend(network_adjustment_rows(account_case, network_lines, account_source=account_source))
    # D4 applies in every case, private included: it changes the comparator
    # rather than measuring an ownership effect (house rule 3).
    if axes[AXIS_NEW_BUILD_CONTRACTS] == NEW_BUILD_AT_COST:
        shortfall = float(account_case.get("new_build_contract_bill_gbp", 0.0))
        gross = float(account_case.get("new_build_contract_shortfall_gbp", 0.0))
        if shortfall != 0.0:
            rows.append({
                "component_id": LINE_NEW_BUILD, "stack_group": "policy",
                "label": "New-build renewables contracted at cost (merchant onshore, solar and hydro)",
                "annual_gbp_per_3100kwh_ex_vat": _household(shortfall, eligible_mwh, factor, mwh),
                "source_class": SOURCE_CLASS,
                "method": (f"Required revenue at the case WACC less market revenue on merchant new-build generation "
                           f"= GBP{gross / 1e6:,.0f}M, of which GBP{shortfall / 1e6:,.0f}M is the privately owned share "
                           "left to contract; the state's share already returns through the surplus-recycling line. "
                           "Recovered over CfD eligible supplier sales. This is a change of comparator basis, not an "
                           "ownership effect: it removes a private baseline subsidised by loss-making investors."),
                "source": account_source,
            })
    # Item D4(c): the LDES line is no longer an extra component. It is a line of
    # the forecast bill itself; each case passes its own top-up to that line
    # (ldes_topup_for), so it can never be added twice.
    rows.extend(background_rows(account_case, factor=factor, eligible_mwh=eligible_mwh, mwh=mwh,
                                domestic_mwh=domestic_mwh, levy_lines=levy_lines, account_source=account_source))
    if case_id == "private" or gross_only:
        return rows
    # User decision 3: the state's own deficits recovered by a levy on eligible supply.
    deficit = float(account_case.get("state_deficit_on_bill_gbp") or 0.0)
    # 9 Nov 2026: a deficit within the identity tolerance is rounding (5e-18 GBP a household on the re-solved
    # months), and a line it emits on one stack and not the other broke the balancing merges' like-for-like check.
    if deficit > IDENTITY_TOLERANCE_GBP:
        rows.append({
            "component_id": LINE_DEFICIT, "stack_group": "policy",
            "label": "Levy recovering the state's own deficits",
            "annual_gbp_per_3100kwh_ex_vat": _household(deficit, eligible_mwh, factor, mwh),
            "source_class": SOURCE_CLASS,
            "method": (f"The negative balances the state keeps (storage, the at-cost remainder, the Capacity Market "
                       f"and dispatchable-CfD routes) = GBP{deficit / 1e6:,.0f}M, recovered over CfD eligible supplier "
                       "sales under state_deficit_funding = bill_levy instead of by the Exchequer."),
            "source": account_source,
        })
    # D2: the Capacity Market and CCS-levy routes booked as model-linked rows
    # instead of being reported beside the bill.
    for line_id, household in booked_route_amounts(
        account_case, factor=factor, final_demand_mwh=final_demand_mwh, eligible_mwh=eligible_mwh,
        mwh=mwh, cm_line=cm_line, ccs_line=ccs_line, cm_household_factor=cm_household_factor,
        line_recovery=line_recovery if two_volume else None,
    ).items():
        if household == 0.0:
            continue
        spec = BOOKED_ROUTE_SPEC[line_id]
        rows.append({
            "component_id": line_id, "stack_group": spec["stack_group"], "label": spec["label"],
            "annual_gbp_per_3100kwh_ex_vat": -household,
            "source_class": SOURCE_CLASS,
            "method": (f"Ownership share x the change in required revenue of {spec['route_label']}, booked against "
                       f"the {spec['line']} line rather than reported beside it (route_booking = booked); capped at "
                       "that line. The route is otherwise unallocated, so booking it moves no other pound."),
            "source": account_source,
        })
    # F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA re-priced to the state's cost (route R1_dpa_repricing_bill).
    dpa_cut = float(account_case.get("dpa_bill_cut_gbp") or 0.0)
    if dpa_cut != 0.0:
        rows.append({
            "component_id": LINE_DPA_REPRICED, "stack_group": "policy",
            "label": "Net Zero Teesside DPA re-priced to the state's cost (pre-2025 plant owned)",
            "annual_gbp_per_3100kwh_ex_vat": -dpa_household(dpa_cut, line_recovery=line_recovery if two_volume else None,
                                                            eligible_mwh=eligible_mwh, factor=factor, mwh=mwh),
            "source_class": SOURCE_CLASS,
            "method": (f"The state owns Net Zero Teesside Power under the pre-2025 option, so its DPA pays it what it "
                       f"costs: the bill's remaining_ccs line falls by GBP{dpa_cut / 1e6:,.0f}M, the plant's contract "
                       "revenue (market revenue plus the DPA, less fuel and carbon) less its cost at the case's rate "
                       "on the at-cost basis (the financing saving otherwise), as an R1 strike cut; capped at the DPA "
                       "payment. Recovered as remaining_ccs is."),
            "source": account_source,
        })
    rule = account_case.get("recycling_rule")
    routed = float(account_case.get("routed_to_bill_gbp", 0.0))
    if rule == "public_cfd_at_cost":
        # One two-way line: the state's renewable and hydro plant on a CfD whose
        # strike is its cost. Rent comes back as a negative row; a top-up is
        # positive, or off the bill when the shortfall is funded by the Exchequer.
        on_bill = routed if routed >= 0 else -float(account_case.get("shortfall_on_bill_gbp", 0.0))
        if on_bill != 0.0:
            rows.append({
                "component_id": LINE_CFD_AT_COST, "stack_group": "policy",
                "label": "Public CfD at cost (state renewables and hydro, two-way)",
                "annual_gbp_per_3100kwh_ex_vat": -_household(on_bill, eligible_mwh, factor, mwh),
                "source_class": SOURCE_CLASS,
                "method": (f"Ownership share x (market revenue - required revenue at the case WACC) of the state's new-build "
                           f"renewables and hydro = GBP{routed / 1e6:,.0f}M, settled as a two-way CfD with strike at cost and "
                           "recovered over CfD eligible supplier sales; gas and BECCS are outside it (Capacity Market and "
                           "dispatchable CfD routes, reported as sensitivities)."),
                "source": account_source,
            })
        return rows
    if routed >= 0:
        if rule == "cost_recovery_contracting" and routed > 0:
            rows.append({
                "component_id": LINE_W1, "stack_group": "hedged_wholesale",
                "label": "Public cost-recovery contracting (state merchant portfolio at cost, W1)",
                "annual_gbp_per_3100kwh_ex_vat": -_household(routed, final_demand_mwh, factor, mwh),
                "source_class": SOURCE_CLASS,
                "method": (f"Ownership share x (market revenue - fuel and carbon - required revenue at the case WACC) of the "
                           f"state's merchant generation = GBP{routed / 1e6:,.0f}M returned over final demand; the R2 CfD "
                           "surplus stays in the public account."),
                "source": account_source,
            })
        elif rule in ("rebate_per_mwh_eligible_supply", "rebate_gross") and routed > 0:
            rows.append({
                "component_id": LINE_REBATE, "stack_group": "policy",
                "label": "Public surplus rebate",
                "annual_gbp_per_3100kwh_ex_vat": -_household(routed, eligible_mwh, factor, mwh),
                "source_class": SOURCE_CLASS,
                "method": (f"Merchant and CfD public surplus GBP{routed / 1e6:,.0f}M returned over CfD eligible "
                           f"supplier sales under rule {rule}"
                           + (" (every positive balance; the negative ones stay with the state)" if rule == "rebate_gross"
                              else "") + "."),
                "source": account_source,
            })
        elif rule == "offset_policy_levies" and routed > 0:
            offset, rebate = offset_split(routed, factor=factor, domestic_mwh=domestic_mwh, mwh=mwh, levy_lines=levy_lines,
                                          levies_on_bills=axes.get(LEVER_LEGACY_LEVY_FUNDING) != LEGACY_LEVIES_TO_TAXATION)
            if offset > 0:
                rows.append({
                    "component_id": LINE_OFFSET, "stack_group": "policy",
                    "label": "Public surplus offsetting domestic customers' RO, FiT and WHD",
                    "annual_gbp_per_3100kwh_ex_vat": -_household(offset, domestic_mwh, factor, mwh),
                    "source_class": SOURCE_CLASS,
                    "method": (f"GBP{offset / 1e6:,.0f}M of the state's GBP{routed / 1e6:,.0f}M surplus offsets the "
                               "Renewables Obligation, Feed-in Tariff and Warm Home Discount of domestic customers, "
                               f"returned over {domestic_mwh / 1e6:,.2f} TWh of domestic supply and capped at those lines "
                               "(review 29 Sep 2026: the rule was a duplicate of the rebate)."),
                    "source": account_source,
                    **domestic_recovery(domestic_mwh),
                })
            if rebate > 0:
                rows.append({
                    "component_id": LINE_REBATE, "stack_group": "policy", "label": "Public surplus rebate (beyond the offset)",
                    "annual_gbp_per_3100kwh_ex_vat": -_household(rebate, eligible_mwh, factor, mwh),
                    "source_class": SOURCE_CLASS,
                    "method": (f"The GBP{rebate / 1e6:,.0f}M the domestic levy lines cannot absorb, rebated over CfD "
                               "eligible supplier sales."),
                    "source": account_source,
                })
    else:
        shortfall = float(account_case.get("shortfall_on_bill_gbp", 0.0))
        if shortfall > 0:
            rows.append({
                "component_id": LINE_SHORTFALL, "stack_group": "policy",
                "label": "Public portfolio shortfall levy",
                "annual_gbp_per_3100kwh_ex_vat": _household(shortfall, eligible_mwh, factor, mwh),
                "source_class": SOURCE_CLASS,
                "method": (f"The state's portfolio earns GBP{shortfall / 1e6:,.0f}M less than it costs under rule "
                           f"{rule}; funded by a levy on CfD eligible supply (public_shortfall_funding = levy_on_eligible_supply)."),
                "source": account_source,
            })
    return rows


#: Review 2 (3 Oct 2026): the register-booked top-up lines a group re-reads from its own public account.
OWN_ACCOUNT_LINE_KEYS = ("life_extension_top_up_gbp", "battery_revenue_top_up_gbp")


def own_account_top_up_lines(assumptions_path, own_account_path, public_account) -> dict:
    """``{bill line: own account}`` for each top-up line whose ``cost_from`` names another account ({} otherwise)."""
    if not own_account_path or not public_account:
        return {}
    assumptions_json = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    own = Path(own_account_path).resolve()
    private = ((public_account.get("cases") or {}).get("private") or {})
    out = {}
    for key, block in (assumptions_json.get("sourced_externals") or {}).items():
        cost_from = (block.get("derivation") or {}).get("cost_from") or {}
        if cost_from.get("key") not in OWN_ACCOUNT_LINE_KEYS or cost_from.get("key") not in private:
            continue
        named = Path(cost_from["public_account"])
        if not named.is_absolute():
            named = Path(__file__).resolve().parents[2] / named
        if named.resolve() != own:
            out[key] = str(own_account_path)
    return out


def project_ownership_cases(*args, own_account_path=None, **kwargs):
    """Return ``(components_by_case, summary_by_case, reconciliation)`` (:func:`_project_ownership_cases`).

    ``own_account_path`` (review 2, 3 Oct 2026): the run group's own public account. Where the
    forecast bill's top-up lines (``life_extension_operating_top_up``, ``battery_revenue_top_up``)
    read another group's account (a register bound shares its parent's forecast bill), they are
    re-read from this group's own, so the bill pays the top-ups this group's register sizes and the
    tariff's ledger closes on them; the private case is checked against the forecast bill on every
    other line. Without it, or on the account the bill names, nothing changes.
    """
    from scripts.consumer_bill.project_future_bill import register_lines_from

    assumptions_path = kwargs["assumptions_path"] if "assumptions_path" in kwargs else args[2]
    public_account = kwargs.get("public_account")
    own_lines = own_account_top_up_lines(assumptions_path, own_account_path, public_account)
    with register_lines_from(own_lines):
        return _project_ownership_cases(*args, own_lines=own_lines, **kwargs)


def _project_ownership_cases(
    case, cases, assumptions_path, networks, prices, summaries, physical_networks, annual_summary,
    balancing_costs, recovery_demand, strike_table_path, ro_scheme_costs, adjustments, routes,
    financing_summary, reference_components, public_account=None, public_account_source="public account",
    network_lines=None, excluded_cases=None, line_sources=None, own_lines=None, bound_lines=None,
):
    """Return ``(components_by_case, summary_by_case, reconciliation)``.

    ``cases`` is every case the market bill projects (item K1: scenarios,
    one-at-a-time sensitivities, named positions and bounding cases, in
    accounting order).  A case that acquires the legacy fleet pays no market
    price and holds no contract, so it takes the acquisition tariff instead;
    the caller names those in ``excluded_cases`` and they are reported.
    """
    cases = list(cases)
    excluded_cases = list(excluded_cases or [])
    account_cases = (public_account or {}).get("cases", {})
    sizewell_settings = ((public_account or {}).get("settings") or {}).get(sizewell_c_rab.CONFIG_KEY) or {}
    from scripts.public_ownership.model_year import is_vintaged_model_year as _vintaged

    # D4 (user, 14 Nov 2026): OW10 applies in every year. A later year's register keeps Sizewell C private on its RAB in
    # every case, so its return at the public rate (at the case's state share) is the bill's, as in 2030.
    # A case that reached the account but acquires the legacy fleet is never
    # projected here, whatever the caller passed.
    tariff_cases = [c for c in cases if c in account_cases
                    and not spec_for_row(axis_settings(account_cases[c])).market_bill()]
    excluded_cases = excluded_cases + [c for c in tariff_cases if c not in excluded_cases]
    cases = [c for c in cases if c not in tariff_cases]
    if "private" not in cases:
        raise ValueError("cases must include the private baseline")
    # Anything in the strike adjustments that the caller did not list is still
    # projected, so an older call shape keeps working.
    extra = [c for c in adjustments["case_id"].drop_duplicates() if c not in cases and c not in excluded_cases]
    all_cases = cases + extra
    cases = [c for c in all_cases if not is_variant(c)]
    variants = [c for c in all_cases if is_variant(c)]
    table = load_bill_strike_table(assumptions_path, int(case["model_price_year"]), strike_table_path)
    # Item B6: the CM booking is allocated to the household by the CM line's own peak factor.
    assumptions_json = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    cm_factor = cm_household_allocation_factor(assumptions_json)
    # Review 29 Sep 2026: the domestic supply volume the repaired levy offset and the
    # legacy levies moved to taxation are recovered over (Ofgem's WHD-obligated volume).
    domestic_mwh = domestic_supply_mwh(assumptions_json)
    reference = reference_components.set_index("component_id")[VALUE].astype(float)
    financing = financing_summary.set_index("case_id")["annual_financing_saving_vs_private_gbp"].astype(float)

    # Private first: its summary supplies the denominators every other case shares.
    components, summaries_by_case = {}, {}
    private_extra, private_route_lines = None, {}
    private_sizewell_line = 0.0
    private_margin_base = None
    # Cases with the same strikes settle to the same year; settle it once.
    settlement_cache: dict = {}
    for case_id in ["private"] + [c for c in all_cases if c != "private"]:
        case_table = case_strike_table(table, adjustments, case_id)
        extras = []
        if private_extra is not None:
            # A boundary variant is a full case with its own account; an
            # eligibility-only variant (no account entry) is projected gross.
            account_case = account_cases.get(case_id) or account_cases.get(base_case(case_id))
            extras = extra_lines(case_id, account_case, gross_only=case_id not in account_cases and is_variant(case_id),
                                 network_lines=network_lines, **private_extra, **private_route_lines)
            extras += sizewell_row(account_case, private_line=private_sizewell_line,
                                   settings=sizewell_settings, network_lines=network_lines,
                                   account_source=public_account_source)
            extras += [] if (case_id not in account_cases and is_variant(case_id)) else battery_top_up_row(
                account_case, private_account=account_cases.get("private"),
                factor=private_extra["factor"], final_demand_mwh=private_extra["final_demand_mwh"],
                mwh=private_extra["mwh"], account_source=public_account_source)
            extras += [] if (case_id not in account_cases and is_variant(case_id)) else cm_state_bids_row(
                account_case, private_account=account_cases.get("private"),
                cm_line=private_route_lines.get("cm_line", 0.0), account_source=public_account_source)
            # F27 item 6: existing capacity on cost-based contracts, against the Capacity Market line.
            extras += [] if case_id not in account_cases else capacity_procurement_row(
                account_cases[case_id], private_extra=private_extra,
                cm_factor=private_route_lines.get("cm_household_factor", 1.0), account_source=public_account_source)
            # FES 2050 decision 73 (none in 2030: the financing summary has no design columns).
            extras += capacity_market_design_row(
                case_id, financing_summary, private_extra=private_extra,
                cm_factor=private_route_lines.get("cm_household_factor", 1.0), account_source=public_account_source)
        elif account_cases.get("private"):
            # Denominators are not known before the first projection; the private
            # case's account lines are built from the recovery file the projector
            # itself reads (the LDES line is the forecast bill's own, item D4(c)).
            recovery = json.loads(Path(recovery_demand).read_text(encoding="utf-8"))
            assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
            # F18: the same denominators the projector's summary reports (on the two-volume basis
            # the CfD's is its eligible GSP volume over the household loss factor).
            conversion = conversion_denominators(recovery)
            bootstrap = {
                "factor": float(assumptions["currency_rebasing"]["factor"]),
                "final_demand_mwh": conversion["final_demand_mwh"],
                "eligible_mwh": conversion["eligible_mwh"],
                "mwh": float(assumptions["benchmark"]["annual_consumption_kwh"]) / 1000.0,
                "account_source": public_account_source,
            }
            extras = extra_lines("private", account_cases["private"], network_lines=network_lines, **bootstrap)
        # Item D4(c): the LDES line is the forecast bill's; this case's top-up goes on it.
        ldes_account = account_cases.get(case_id) or account_cases.get(base_case(case_id))
        _, frame, summary = run_projection(
            case, assumptions_path, networks, prices, summaries, physical_networks, annual_summary,
            balancing_costs, recovery_demand, strike_table_path, ro_scheme_costs,
            strike_table_object=case_table, extra_components=extras, settlement_cache=settlement_cache,
            ldes_topup_gbp=ldes_topup_for(ldes_account),
            # F27 item 10 (user, 20 Oct 2026): EBIT and headroom on each case's own costs, as the cap
            # formula would set them (supersedes G5's private margin for every case); a vintaged (2050)
            # year keeps the private case's.
            supplier_margin_base=(private_margin_base if _vintaged(int(case["modelled_year"])) else None),
            # Item F12 (12 Oct 2026): a price-level twin's account says its level (labels the LDES line only).
            ldes_account_price_level=(public_account or {}).get("price_level"),
        )
        frame.insert(0, "case_id", case_id)
        components[case_id] = frame
        summaries_by_case[case_id] = summary
        if private_extra is None:
            private_margin_base = (summary.get("supplier_margin") or {}).get("base")
            if private_margin_base is not None:
                private_margin_base = {**private_margin_base, "source": "the private case's bill lines (F18)"}
            # F19 (7 Oct 2026): the builders' two volumes through the private bill's own recovery
            # (scripts/public_ownership/bill_recovery.py): metered supply, the Exchequer's key, and the
            # volume over which the CfD line converts flat. On a D1 bill, its own two denominators.
            private_extra = {
                "factor": float(summary["model_rebasing_factor"]),
                "final_demand_mwh": bill_recovery.exchequer_key_mwh(summary),
                "eligible_mwh": bill_recovery.flat_equivalent_mwh(summary, "cfd"),
                "mwh": float(summary["annual_consumption_kwh"]) / 1000.0,
                "account_source": public_account_source,
                # F19 (7 Oct 2026): the private bill's per-line recovery (F18); on the two-volume
                # basis the bookings are recovered on the entries of the lines they adjust.
                "line_recovery": summary.get("line_recovery"),
                "two_volume": _two_volume(summary),
            }
            # The two lines a D2 booking adjusts, read once from the private
            # case: they are external lines and do not move between cases.
            private_lines = frame.set_index("component_id")[VALUE].astype(float)
            private_route_lines = {
                "cm_line": float(private_lines.get("capacity_market", 0.0)),
                "ccs_line": float(private_lines.get("allocated_ccs", 0.0)),
                "cm_household_factor": cm_factor,
                "domestic_mwh": domestic_mwh,
                "levy_lines": {line: float(private_lines.get(line, 0.0)) for line in DOMESTIC_LEVY_LINES},
            }
            # OW10: the levy the public cases' Sizewell C adjustment scales (an external line, never moved).
            private_sizewell_line = float(private_lines.get(sizewell_c_rab.LINE, 0.0))

    private = components["private"].set_index("component_id")[VALUE].astype(float)
    # The LDES line is compared too (item D4(c)): the forecast bill carries the
    # private top-up, so the private case must reproduce it, not add it.
    shared = private.drop(index=[c for c in NEW_LINES if c in private.index and c != LINE_LDES])
    # A register bound's own inputs set some lines (25 Oct 2026): the LDES floor from its own account, a margin
    # bound's CfD line at its own strikes (``bound_lines``, from the rule). Checked like the own top-up lines.
    check_lines = {*(own_lines or {}), *(bound_lines or ())}
    if check_lines:
        # Review 2: the top-up lines re-read from this group's own account are its own, not the forecast's.
        shared = shared.copy()
        moved = {line: float(shared[line] - reference[line]) for line in check_lines
                 if line in shared.index and line in reference.index}
        for line in moved:
            shared[line] = reference[line]
        # F18 sets the supplier line's EBIT and headroom as Ofgem's percentages of the bill's
        # costs, so an own top-up line moves it too (8 Oct 2026): take that move out exactly.
        margin = (summaries_by_case["private"].get("supplier_margin") or {})
        rule = margin.get("rule")
        if rule is not None and SUPPLIER_LINE in shared.index and moved:
            network = rule["network_group"]
            groups = bill_group_of(network["view"])
            d_costs = sum(moved.values())
            d_networks = sum(v for line, v in moved.items() if groups.get(line) == network["group"])
            d_ebit = float(rule["ebit_variable_share"]) * d_costs
            d_headroom = float(rule["headroom_share"]) * (d_costs - d_networks + d_ebit)
            shared[SUPPLIER_LINE] = shared[SUPPLIER_LINE] - d_ebit - d_headroom
    if not shared.index.equals(reference.index):
        raise ValueError("The private case does not carry the forecast case's component list")
    if not np.allclose(shared.to_numpy(), reference.to_numpy(), rtol=1e-12, atol=1e-9):
        worst = (shared - reference).abs().idxmax()
        differing = ~np.isclose(shared.to_numpy(), reference.to_numpy(), rtol=1e-12, atol=1e-9)
        listed = ", ".join(f"{line} {shared[line]:.6f} vs {reference[line]:.6f}" for line in shared.index[differing])
        raise ValueError(
            f"The private case does not reproduce the {case['case_id']} projection "
            f"(largest difference on {worst}: {shared[worst]:.9f} vs {reference[worst]:.9f}; every line that "
            f"differs: {listed}); regenerate the forecast bill or check the inputs are the same files"
        )

    ps = summaries_by_case["private"]
    factor, final_demand, eligible = private_extra["factor"], private_extra["final_demand_mwh"], private_extra["eligible_mwh"]
    mwh = private_extra["mwh"]
    two_volume = bool(private_extra.get("two_volume"))
    private_recovery = private_extra.get("line_recovery") if two_volume else None
    cm_entry = entry_for(private_recovery, "capacity_market") if private_recovery and "capacity_market" in private_recovery else None
    private_ldes = float(private.get(LINE_LDES, 0.0))
    private_account = account_cases.get("private", {})

    # An all_cases axis changes the comparator, so a case that moves one is
    # reported against the private case at the same point (house rule 3). The
    # matching lives in sensitivity_axes so this cannot drift from the expansion
    # that created the cases: every all_cases axis has to agree, not just one.
    settings_by_case = {
        case_id: axis_settings(account_cases.get(case_id) or account_cases.get(base_case(case_id)))
        for case_id in all_cases
    }
    comparator_case_ids = {
        case_id: comparator_for_settings(case_id, settings_by_case) for case_id in all_cases
    }

    reconciliation_rows, case_report = [], {}
    previous_gross = None
    # Task SETTLE2050: in a vintaged (2050) year the at-cost strike of a contract whose output
    # falls short of the bidder's expected load factor rises, so a case's gross bill can exceed
    # the preceding case's; that is recorded, not refused. Every 2030 year still refuses.
    gross_order_findings = []
    from scripts.public_ownership.model_year import is_vintaged_model_year

    vintaged_year = is_vintaged_model_year(int(case["modelled_year"]))
    for case_id in all_cases:
        frame = components[case_id].set_index("component_id")[VALUE].astype(float)
        summary = summaries_by_case[case_id]
        base = base_case(case_id)
        gross_only = is_variant(case_id) and case_id not in account_cases
        comparator = comparator_case_ids.get(case_id, "private")
        account_case = dict(account_cases.get(case_id) or account_cases.get(base, {}))
        if gross_only:
            # Eligibility-only variant: the LDES floor is kept, the recycling routes are not applied.
            account_case.update(routed_to_bill_gbp=0.0, shortfall_on_bill_gbp=0.0, shortfall_on_exchequer_gbp=0.0,
                                retained_in_public_account_gbp=0.0, recycling_rule="not_applied_sensitivity")
        delta = frame.sub(private, fill_value=0.0)
        moved = delta.index[~np.isclose(delta.to_numpy(), 0.0, atol=1e-9)].tolist()
        if case_id == "private" and moved:
            raise ValueError("The private case differs from itself")
        # allocated_ccs is the BECCS CfD carved out of the CfD settlement (item
        # MR7), so it moves with R1 exactly as the CfD line does.
        # F27 item 10: the supplier line's EBIT and headroom follow the case's own costs, so it moves with
        # them; its move is booked to suppliers below (supplier_margin_change_gbp_2024).
        allowed = {*R1_LINES, *NEW_LINES, SUPPLIER_LINE, LINE_CM_DESIGN}
        if set(moved) - allowed:
            raise ValueError(f"{case_id}: lines moved that no route explains: {sorted(set(moved) - allowed)}")

        # R1: the CfD line (with its allocated CCS carve-out) moves by exactly
        # the strike adjustments. Both lines share the eligible-sales denominator.
        cfd_delta = float(delta.get("cfd", 0.0)) + float(delta.get("allocated_ccs", 0.0))
        cfd_saving_system = -_system(cfd_delta, eligible, factor, mwh)
        case_routes = routes.loc[routes["case_id"].eq(case_id)].set_index("route")
        r1 = float(case_routes["bill_allocated_gbp_2024"].get(ROUTE_CFD_BILL, 0.0))
        # FES 2050 decision 54: a case on another private rate path settles every post-AR7 contract at
        # that path's calibrated strike; the CfD line carries that cut too (zero for every 2030 case).
        strike_basis = strike_basis_offset_saving(adjustments, case_id)
        residuals = {"cfd": cfd_saving_system - r1 - strike_basis}
        # R3: the LDES floor moves by exactly the change in top-up.
        ldes_delta = float(delta.get(LINE_LDES, 0.0))
        # F18: the LDES line is the projector's, a GSP-rate line; its change converts through the
        # line's own recovery entry (the D1 entry is metered supply, the same as final demand).
        ldes_change_system = (-household_to_system_gbp(entry_for(summary, LINE_LDES), ldes_delta, mwh * 1000.0)
                              if LINE_LDES in (summary.get("line_recovery") or {})
                              else -_system(ldes_delta, final_demand, factor, mwh))
        ldes_expected = (float(private_account.get("ldes_floor_topup_gbp", 0.0)) - float(account_case.get("ldes_floor_topup_gbp", 0.0))
                         if account_case else 0.0)
        residuals[LINE_LDES] = ldes_change_system - ldes_expected
        # W1 / rebate / shortfall: each line carries exactly the account amount.
        w1_system = -_system(float(frame.get(LINE_W1, 0.0)), final_demand, factor, mwh)
        cfd_at_cost_system = -_system(float(frame.get(LINE_CFD_AT_COST, 0.0)), eligible, factor, mwh)
        rebate_system = -_system(float(frame.get(LINE_REBATE, 0.0)), eligible, factor, mwh)
        offset_system = -_system(float(frame.get(LINE_OFFSET, 0.0)), domestic_mwh or 1.0, factor, mwh)
        shortfall_system = _system(float(frame.get(LINE_SHORTFALL, 0.0)), eligible, factor, mwh)
        deficit_system = _system(float(frame.get(LINE_DEFICIT, 0.0)), eligible, factor, mwh)
        routed = float(account_case.get("routed_to_bill_gbp", 0.0)) if account_case else 0.0
        rule = account_case.get("recycling_rule") if account_case else None
        shortfall_on_bill = float(account_case.get("shortfall_on_bill_gbp", 0.0) if account_case else 0.0)
        axes = axis_settings(account_case)
        levy_lines = {line: float(private.get(line, 0.0)) for line in DOMESTIC_LEVY_LINES}
        residuals[LINE_W1] = w1_system - (routed if rule == "cost_recovery_contracting" and routed > 0 else 0.0)
        residuals[LINE_CFD_AT_COST] = cfd_at_cost_system - (
            (routed if routed >= 0 else -shortfall_on_bill) if rule == "public_cfd_at_cost" else 0.0)
        expected_offset, expected_rebate = 0.0, 0.0
        if rule == "offset_policy_levies" and routed > 0:
            expected_offset, expected_rebate = offset_split(
                routed, factor=factor, domestic_mwh=domestic_mwh, mwh=mwh, levy_lines=levy_lines,
                levies_on_bills=axes.get(LEVER_LEGACY_LEVY_FUNDING) != LEGACY_LEVIES_TO_TAXATION)
        elif rule in ("rebate_per_mwh_eligible_supply", "rebate_gross") and routed > 0:
            expected_rebate = routed
        residuals[LINE_REBATE] = rebate_system - expected_rebate
        residuals[LINE_OFFSET] = offset_system - expected_offset
        residuals[LINE_SHORTFALL] = shortfall_system - (shortfall_on_bill if rule != "public_cfd_at_cost" else 0.0)
        deficit_expected = (float(account_case.get("state_deficit_on_bill_gbp") or 0.0)
                            if account_case and case_id != "private" and not gross_only else 0.0)
        residuals[LINE_DEFICIT] = deficit_system - deficit_expected
        # The background lines (every case alike): each carries exactly its moved amount.
        legacy_system = {}
        for line, row_id in DOMESTIC_LEVY_LINES.items():
            got = -_system(float(frame.get(row_id, 0.0)), domestic_mwh or 1.0, factor, mwh)
            expected = (_system(levy_lines[line], domestic_mwh, factor, mwh)
                        if axes.get(LEVER_LEGACY_LEVY_FUNDING) == LEGACY_LEVIES_TO_TAXATION else 0.0)
            legacy_system[row_id] = got
            residuals[row_id] = got - expected
        beccs_levy_system = _system(float(frame.get(LINE_BECCS_LEVY, 0.0)), eligible, factor, mwh)
        residuals[LINE_BECCS_LEVY] = beccs_levy_system - (
            float(account_case.get("carbon_removal_cfd_revenue_gbp") or 0.0)
            if account_case and axes.get(LEVER_BECCS_REMOVAL_FUNDING) == BECCS_REMOVAL_BILL_LEVY else 0.0)
        # D2: each booked route line carries exactly the amount the account
        # routes to it, after the same cap the booking applied.
        booked = booked_route_amounts(
            account_case, factor=factor, final_demand_mwh=final_demand, eligible_mwh=eligible, mwh=mwh,
            cm_line=float(private.get("capacity_market", 0.0)), ccs_line=float(private.get("allocated_ccs", 0.0)),
            cm_household_factor=cm_factor, line_recovery=private_recovery,
        )
        booked_system = {}
        for line_id, spec in BOOKED_ROUTE_SPEC.items():
            if private_recovery is not None:
                # F19 (7 Oct 2026): back to system money through the adjusted line's own entry.
                entry = entry_for(private_recovery, spec["line"])
                booked_system[line_id] = -household_to_system_gbp(entry, float(frame.get(line_id, 0.0)), mwh * 1000.0)
                residuals[line_id] = booked_system[line_id] - household_to_system_gbp(
                    entry, booked.get(line_id, 0.0), mwh * 1000.0)
                continue
            denominator = final_demand if spec["denominator"] == "final_demand" else eligible
            # Household back to system money through the same peak factor (item B6).
            peak = cm_factor if line_id == LINE_CM_BOOKED else 1.0
            booked_system[line_id] = -_system(float(frame.get(line_id, 0.0)), denominator, factor, mwh) / peak
            residuals[line_id] = booked_system[line_id] - _system(
                booked.get(line_id, 0.0), denominator, factor, mwh
            ) / peak
        # F27 item 1: the DPA row carries exactly the route's bill amount, which the account cut too.
        dpa_bill_system = -dpa_system(float(frame.get(LINE_DPA_REPRICED, 0.0)), line_recovery=private_recovery,
                                      eligible_mwh=eligible, factor=factor, mwh=mwh)
        dpa_expected = (0.0 if gross_only else
                        float(case_routes["bill_allocated_gbp_2024"].get(ROUTE_DPA_BILL, 0.0)) if len(case_routes) else 0.0)
        residuals[LINE_DPA_REPRICED] = dpa_bill_system - dpa_expected
        residuals["dpa_account_cut"] = (float(account_case.get("dpa_bill_cut_gbp") or 0.0) - dpa_expected
                                        if account_case and not gross_only else 0.0)
        # D4: a change of comparator basis, declared rather than routed; it must
        # still carry exactly the account amount.
        new_build_system = _system(float(frame.get(LINE_NEW_BUILD, 0.0)), eligible, factor, mwh)
        new_build_expected = (float(account_case.get("new_build_contract_bill_gbp", 0.0))
                              if account_case and axes[AXIS_NEW_BUILD_CONTRACTS] == NEW_BUILD_AT_COST else 0.0)
        residuals[LINE_NEW_BUILD] = new_build_system - new_build_expected
        # Item J6 / N6: each network adjustment row carries exactly the account's
        # amount for the case's networks lever (household GBP, converted over
        # final demand like the line it adjusts).
        treatment = axes.get(LEVER_NETWORKS, NETWORKS_PRIVATE)
        expected_network = (adjustments_for_treatment(network_lines, treatment, axes)
                            if treatment != NETWORKS_PRIVATE and network_lines is not None else {})
        # Where the networks' saving lands (user decision 5): the bill, a fee or the return the Exchequer keeps.
        network_incidence = ({line: {k: float(v) for k, v in amount.items() if k != "sector"}
                              for line, amount in incidence_amounts(
                                  select_rows(network_lines, treatment, axes), treatment, axes).items()}
                             if treatment != NETWORKS_PRIVATE and network_lines is not None else {})
        # User decision 7 (second round): the break-even premium beside the premium paid.
        network_break_even = (break_even_summary(select_rows(network_lines, treatment, axes))
                              if treatment == "acquired" and network_lines is not None else {})
        network_system = {}
        for line_id, bill_line in NETWORK_LINES.items():
            got = float(frame.get(line_id, 0.0))
            network_system[line_id] = _system(got, final_demand, factor, mwh)
            residuals[line_id] = _system(got - float(expected_network.get(bill_line, 0.0)), final_demand, factor, mwh)
        # Item K3: the pay-as-bid ceiling row carries exactly the account's ceiling.
        pay_as_bid_system = -_system(float(frame.get(LINE_PAY_AS_BID, 0.0)), final_demand, factor, mwh)
        pay_as_bid_expected = (float(account_case.get("pay_as_bid_ceiling_gbp") or 0.0)
                               if account_case and account_case.get("pay_as_bid_ceiling_applies") else 0.0)
        residuals[LINE_PAY_AS_BID] = pay_as_bid_system - pay_as_bid_expected
        # OW10: the Sizewell C row carries exactly the levy's return element moved to the public rate,
        # recovered over final demand like the levy.
        sizewell_system = _system(float(frame.get(LINE_SIZEWELL, 0.0)), final_demand, factor, mwh)
        sizewell = sizewell_terms(account_case, settings=sizewell_settings, network_lines=network_lines)
        sizewell_expected = (_system(sizewell_c_rab.household_adjustment(
            float(private.get(sizewell_c_rab.LINE, 0.0)), sizewell_settings, state_share=sizewell[0], rate=sizewell[1]),
            final_demand, factor, mwh) if sizewell else 0.0)
        residuals[LINE_SIZEWELL] = sizewell_system - sizewell_expected
        # F19 decision 5: the battery row carries exactly the case's top-up less the private case's.
        # F27 item 10: the supplier line's EBIT and headroom follow the case's own costs; its move
        # against private (system GBP, through the line's own recovery entry) is the suppliers'.
        supplier_delta = float(frame.get(SUPPLIER_LINE, 0.0)) - float(private.get(SUPPLIER_LINE, 0.0))
        supplier_system = (household_to_system_gbp(entry_for(summary, SUPPLIER_LINE), supplier_delta, mwh * 1000.0)
                           if SUPPLIER_LINE in (summary.get("line_recovery") or {})
                           else _system(supplier_delta, final_demand, factor, mwh))
        battery_system = _system(float(frame.get(LINE_BATTERY_TOP_UP_ADJ, 0.0)), final_demand, factor, mwh)
        battery_expected = (battery_top_up_change(account_case, private_account)
                            if case_id != "private" and not gross_only else 0.0)
        residuals[LINE_BATTERY_TOP_UP_ADJ] = battery_system - battery_expected
        # F9: the re-cleared T-4 row carries exactly its share of the private Capacity Market line (system GBP,
        # back through the line's own peak factor).
        # F19 (7 Oct 2026): on the two-volume basis through the Capacity Market line's own entry (GSP charging
        # volume and its allocation factor), not final demand over the factor.
        cm_to_system = ((lambda hh: household_to_system_gbp(cm_entry, hh, mwh * 1000.0)) if cm_entry is not None
                        else (lambda hh: _system(hh, final_demand, factor, mwh) / cm_factor))
        cm_bids_system = cm_to_system(float(frame.get(LINE_CM_STATE_BIDS, 0.0)))
        cm_bids_share = (cm_state_bids_share(account_case, private_account)
                         if case_id != "private" and not gross_only else 0.0)
        cm_bids_expected = cm_to_system(float(private.get("capacity_market", 0.0))) * cm_bids_share
        residuals[LINE_CM_STATE_BIDS] = cm_bids_system - cm_bids_expected
        cm_bids_block = (account_case or {}).get(cm_state_bids.CONFIG_KEY) if cm_bids_system else None
        # F27 item 6: the cost-based contracts' row carries exactly the account's saving on private owners' units.
        cm_cost_based_system = cm_to_system(float(frame.get(LINE_CM_COST_BASED, 0.0)))
        cm_cost_based_expected = (-capacity_procurement_saving(account_cases.get(case_id))
                                  if case_id != "private" else 0.0)
        residuals[LINE_CM_COST_BASED] = cm_cost_based_system - cm_cost_based_expected
        cm_cost_based_block = (account_cases.get(case_id) or {}).get(capacity_procurement.CONFIG_KEY)
        # FES 2050 decision 73: the design row carries exactly its auction's cost change (absent in 2030,
        # where no financing summary has the design columns and no residual is added).
        from scripts.public_ownership.capacity_market_design import case_moves

        cm_design = case_moves(financing_summary, case_id) if case_id != "private" else None
        if cm_design is not None:
            cm_design_system = cm_to_system(float(frame.get(LINE_CM_DESIGN, 0.0)))
            residuals[LINE_CM_DESIGN] = cm_design_system - cm_design["line_change_gbp"]
        bad = {k: v for k, v in residuals.items() if abs(v) > IDENTITY_TOLERANCE_GBP}
        if bad:
            raise ValueError(f"{case_id}: bill lines do not reconcile to their routes (system GBP 2024): {bad}")

        ex_vat, vat, total = float(summary["ex_vat_gbp"]), float(summary["vat_gbp"]), float(summary["total_gbp_inc_vat"])
        vat_rate = float(summary["vat_rate"])
        if not np.isclose(vat, ex_vat * vat_rate, rtol=0, atol=1e-9):
            raise ValueError(f"{case_id}: VAT is not the rate applied once to the ex-VAT subtotal")
        recycling_ex_vat = float(sum(frame.get(line, 0.0) for line in RECYCLING_LINES))
        gross_total = (ex_vat - recycling_ex_vat) * (1.0 + vat_rate)
        if not is_variant(case_id):
            if previous_gross is not None and gross_total > previous_gross + 1e-9:
                if not vintaged_year:
                    raise ValueError(f"{case_id}: gross bill {gross_total:,.2f} exceeds the preceding case {previous_gross:,.2f}")
                gross_order_findings.append({"case_id": case_id, "gross_bill_gbp_inc_vat": gross_total,
                                             "preceding_gross_bill_gbp_inc_vat": previous_gross})
            previous_gross = gross_total
        private_total = float(ps["total_gbp_inc_vat"])
        private_gross = (float(ps["ex_vat_gbp"])) * (1.0 + vat_rate)
        # The comparator column: the same private case under any all_cases axis
        # this variant moves.  A private comparator carries no recycling rows,
        # so its gross is its ex-VAT subtotal grossed up once.
        cs = summaries_by_case.get(comparator, ps)
        comparator_total = float(cs["total_gbp_inc_vat"])
        comparator_gross = float(cs["ex_vat_gbp"]) * (1.0 + vat_rate)

        cm_sensitivity = account_case.get("capacity_market_sensitivity_gbp") if account_case else None
        cm_line = float(frame.get("capacity_market", 0.0))
        cm_sensitivity_household = (min(max(cm_line, 0.0), _household(cm_sensitivity, final_demand, factor, mwh) * cm_factor)
                                    if cm_sensitivity is not None else None)
        ccs_sensitivity = account_case.get("ccs_levy_sensitivity_gbp") if account_case else None
        ccs_line = float(frame.get("allocated_ccs", 0.0))
        ccs_sensitivity_household = (min(max(ccs_line, 0.0), _household(ccs_sensitivity, eligible, factor, mwh))
                                     if ccs_sensitivity is not None else None)
        w2 = (float(account_case["w2_average_generation_cost_gbp"]) / float(account_case["w2_generation_mwh"])
              if account_case.get("w2_generation_mwh") else None)
        case_report[case_id] = {
            "base_case": base,
            "sensitivity": case_id.split("@", 1)[1] if is_variant(case_id) else None,
            "sensitivity_axes": axes,
            "comparator_case_id": comparator,
            "ex_vat_gbp": ex_vat, "vat_gbp": vat, "total_gbp_inc_vat": total,
            "gross_total_gbp_inc_vat": gross_total, "net_total_gbp_inc_vat": total,
            "delta_gross_vs_private_gbp_inc_vat": gross_total - private_gross,
            "delta_net_vs_private_gbp_inc_vat": total - private_total,
            "delta_gross_vs_comparator_gbp_inc_vat": gross_total - comparator_gross,
            "delta_net_vs_comparator_gbp_inc_vat": total - comparator_total,
            "delta_ex_vat_gbp": ex_vat - float(ps["ex_vat_gbp"]),
            "cfd_line_gbp_per_3100kwh_ex_vat": float(frame["cfd"]),
            "cfd_line_delta_gbp_per_3100kwh_ex_vat": cfd_delta,
            "new_lines_gbp_per_3100kwh_ex_vat": {line: float(frame.get(line, 0.0)) for line in NEW_LINES if line in frame.index},
            "lines_moved": moved,
            "ownership_boundary": account_case.get(LEVER_PRE_2025_CFD, account_case.get("ownership_boundary")) if account_case else None,
            "levers": axes,
            "network_treatment": treatment,
            "network_adjustments_gbp_per_3100kwh_ex_vat": {line: float(frame.get(line, 0.0)) for line in NETWORK_LINES},
            "network_adjustments_system_gbp_2024": network_system,
            "network_break_even_premium": network_break_even or None,
            "pay_as_bid_ceiling_gbp_2024": pay_as_bid_expected,
            # OW10: Sizewell C's RAB return at the public rate (system GBP, negative lowers the bill).
            "sizewell_c_rab_public_rate_gbp_2024": sizewell_system,
            # F27 item 10: the supplier margin's move against private (system GBP, positive raises the bill).
            "supplier_margin_change_gbp_2024": supplier_system,
            # F19 decision 5: the battery top-up at the case's own cost, and who it is no longer paid to.
            "battery_top_up_owner_cost_gbp_2024": battery_system,
            "battery_top_up_change_state_gbp_2024": (float(account_case.get("battery_top_up_change_state_gbp") or 0.0)
                                                     if account_case and battery_system else 0.0),
            "battery_top_up_change_coowners_gbp_2024": (
                float(account_case.get("battery_top_up_change_coowners_gbp") or 0.0)
                if account_case and battery_system else 0.0),
            # F9: the Capacity Market line moved by the re-cleared 2030/31 T-4, and the state's part of it
            # (the state's own winners' lower receipts; the rest is private winners').
            "capacity_market_state_bids_gbp_2024": cm_bids_system,
            "capacity_market_state_bids_state_part_gbp_2024": (
                cm_bids_system * float(cm_bids_block.get("state_part_share", 0.0)) if cm_bids_block else 0.0),
            "capacity_market_state_bids": cm_bids_block or None,
            # F27 item 6: existing capacity on cost-based contracts (system GBP, negative lowers the bill: a
            # transfer from the private owners of existing plant), and the account's block.
            "capacity_market_cost_based_existing_gbp_2024": cm_cost_based_system,
            capacity_procurement.CONFIG_KEY: cm_cost_based_block or None,
            # F11: two other recovery rules for the Capacity Market line, reported beside the headline only.
            "capacity_market_recovery_options": (None if vintaged_year else cm_recovery_options(
                cm_line + float(frame.get(LINE_CM_BOOKED, 0.0)) + float(frame.get(LINE_CM_STATE_BIDS, 0.0))
                + float(frame.get(LINE_CM_COST_BASED, 0.0)),
                cm_factor=cm_factor, final_demand_mwh=final_demand, factor=factor, mwh=mwh, cm_entry=cm_entry)),
            "sizewell_c_rab_public_rate": ({"state_share": sizewell[0], "public_real_rate": sizewell[1],
                                            "rab_real_rate": sizewell_settings["rab_real_rate"],
                                            "return_share": sizewell_settings["return_share"],
                                            "line_gbp_per_3100kwh_ex_vat": float(frame.get(LINE_SIZEWELL, 0.0))}
                                           if sizewell else None),
            "pay_as_bid_ceiling_line_gbp_per_3100kwh_ex_vat": float(frame.get(LINE_PAY_AS_BID, 0.0)),
            "financing_saving_gbp_2024": float(financing.get(case_id, financing.get(base, 0.0))),
            "bill_allocated_r1_gbp_2024": r1,
            # F27 item 1: Net Zero Teesside's DPA re-priced on the remaining_ccs line (system GBP, a cut).
            "bill_allocated_dpa_gbp_2024": dpa_bill_system,
            **({"strike_basis_offset_saving_gbp_2024": strike_basis}
               if "strike_basis_offset_gbp_per_mwh_2024" in adjustments.columns else {}),
            "bill_allocated_ldes_gbp_2024": ldes_expected,
            "bill_allocated_w1_gbp_2024": w1_system,
            "bill_allocated_public_cfd_at_cost_gbp_2024": cfd_at_cost_system,
            "bill_allocated_rebate_gbp_2024": rebate_system,
            "bill_allocated_offset_domestic_levies_gbp_2024": offset_system,
            "state_deficit_levy_gbp_2024": deficit_system,
            "state_deficit_funding": axes.get("state_deficit_funding"),
            "legacy_levies_to_taxation_gbp_2024": legacy_system,
            "legacy_levies_to_taxation_total_gbp_2024": float(sum(legacy_system.values())),
            "beccs_carbon_removal_levy_gbp_2024": beccs_levy_system,
            "carbon_removal_cfd_gbp_2024": float(account_case.get("carbon_removal_cfd_revenue_gbp") or 0.0)
            if account_case else 0.0,
            "network_incidence_gbp_2024": network_incidence,
            "capacity_market_booking_floored_gbp_2024": float(
                account_case.get("capacity_market_booking_floored_gbp") or 0.0) if account_case else 0.0,
            "ldes_floor_topup_gbp_2024": float(account_case.get("ldes_floor_topup_gbp", 0.0)) if account_case else 0.0,
            "ldes_floor_topup_at_wacc_gbp_2024": float(account_case.get("ldes_floor_topup_at_wacc_gbp", 0.0)) if account_case else 0.0,
            "ldes_floor_real_rate": account_case.get("ldes_floor_real_rate") if account_case else None,
            "ccs_levy_sensitivity_gbp_per_3100kwh_ex_vat": ccs_sensitivity_household,
            "capacity_market_route_surplus_not_recycled_gbp_2024": float(account_case.get("capacity_market_route_surplus_not_recycled_gbp", 0.0)) if account_case else 0.0,
            "dispatchable_cfd_route_surplus_not_recycled_gbp_2024": float(account_case.get("dispatchable_cfd_route_surplus_not_recycled_gbp", 0.0)) if account_case else 0.0,
            "shortfall_on_bill_gbp_2024": shortfall_system,
            "shortfall_on_exchequer_gbp_2024": float(account_case.get("shortfall_on_exchequer_gbp", 0.0)) if account_case else 0.0,
            "public_account_retained_gbp_2024": float(account_case.get("retained_in_public_account_gbp", 0.0)) if account_case else 0.0,
            "storage_surplus_not_recycled_gbp_2024": float(account_case.get("storage_surplus_not_recycled_gbp", 0.0)) if account_case else 0.0,
            "recycling_rule": account_case.get("recycling_rule") if account_case else None,
            "identity_residuals_gbp_2024": residuals,
            "routes_gbp_2024": {route: float(case_routes["saving_economic_life_gbp_2024"].get(route, 0.0)) for route in ROUTES},
            "route_identity_residual_gbp_2024": float(case_routes["identity_residual_gbp_2024"].iloc[0]) if len(case_routes) else 0.0,
            "capacity_market_sensitivity_gbp_per_3100kwh_ex_vat": cm_sensitivity_household,
            "route_booking_booked_gbp_2024": {line: booked_system[line] for line in BOOKED_ROUTE_SPEC},
            "capacity_market_line_net_gbp_per_3100kwh_ex_vat": cm_line + float(frame.get(LINE_CM_BOOKED, 0.0)),
            "allocated_ccs_line_net_gbp_per_3100kwh_ex_vat": ccs_line + float(frame.get(LINE_CCS_BOOKED, 0.0)),
            "new_build_contracts_line_gbp_per_3100kwh_ex_vat": float(frame.get(LINE_NEW_BUILD, 0.0)),
            "new_build_contracts_bill_gbp_2024": new_build_system,
            "new_build_contracts_shortfall_gbp_2024": (
                float(account_case.get("new_build_contract_shortfall_gbp", 0.0)) if account_case else 0.0),
            "w2_average_generation_cost_gbp_per_mwh_2024": w2,
            "strike_table_adjustment": summary.get("strike_table_adjustment"),
            # Item G6: every case's bill at the archetype volumes, on the declared
            # standing and unit split (a network saving lands mostly on the standing charge).
            "household_archetypes": (summary.get("diagnostics", {}).get("household_archetypes") or {}).get("bills"),
            # F18: one recovery entry per line of this case's bill (scripts/consumer_bill/line_recovery.py);
            # the lines this script adds carry a copy of the entry of the line they adjust.
            "line_recovery": summary.get("line_recovery"),
            "supplier_margin": summary.get("supplier_margin"),
        }
        if cm_design is not None:
            # FES 2050 decision 73: the CM line's move and who it pays (system GBP 2024), for the
            # three-party identity and the party flows (never in 2030).
            case_report[case_id].update({
                "capacity_market_design": cm_design,
                "capacity_market_design_line_gbp_2024": cm_design_system,
                "capacity_market_design_line_gbp_per_3100kwh_ex_vat": float(frame.get(LINE_CM_DESIGN, 0.0)),
                "capacity_market_line_net_gbp_per_3100kwh_ex_vat": (
                    case_report[case_id]["capacity_market_line_net_gbp_per_3100kwh_ex_vat"]
                    + float(frame.get(LINE_CM_DESIGN, 0.0))),
            })
        bill_changes = {
            ROUTE_CFD_BILL: cfd_delta,
            ROUTE_STORAGE: ldes_delta,
            ROUTE_MERCHANT: (float(frame.get(LINE_W1, 0.0)) + float(frame.get(LINE_CFD_AT_COST, 0.0))
                            + float(frame.get(LINE_REBATE, 0.0)) + float(frame.get(LINE_OFFSET, 0.0))
                            + float(frame.get(LINE_SHORTFALL, 0.0))),
            ROUTE_CAPACITY_MARKET: float(frame.get(LINE_CM_BOOKED, 0.0)),
            ROUTE_DISPATCHABLE_CFD: float(frame.get(LINE_CCS_BOOKED, 0.0)),
            ROUTE_DPA_BILL: float(frame.get(LINE_DPA_REPRICED, 0.0)),
        }
        allocated = {ROUTE_CFD_BILL: r1, ROUTE_DPA_BILL: dpa_bill_system, ROUTE_STORAGE: ldes_expected,
                     ROUTE_MERCHANT: w1_system + cfd_at_cost_system + rebate_system + offset_system - shortfall_system,
                     ROUTE_CAPACITY_MARKET: booked_system[LINE_CM_BOOKED],
                     ROUTE_DISPATCHABLE_CFD: booked_system[LINE_CCS_BOOKED]}
        for route in ROUTES:
            reconciliation_rows.append({
                "case_id": case_id, "route": route,
                "treatment": ("bill_allocated" if route in (ROUTE_CAPACITY_MARKET, ROUTE_DISPATCHABLE_CFD)
                              and abs(float(allocated.get(route, 0.0))) > 0.0
                              else case_routes["treatment"].get(route)),
                "assets": int(case_routes["assets"].get(route, 0)),
                "saving_economic_life_gbp_2024": float(case_routes["saving_economic_life_gbp_2024"].get(route, 0.0)),
                "bill_allocated_gbp_2024": float(allocated.get(route, 0.0)),
                "term_adjustment_gbp_2024": float(case_routes["term_adjustment_gbp_2024"].get(route, 0.0)),
                # Review 28 Sep 2026: the at-cost rent inside it is a transfer, labelled apart.
                "at_cost_rent_transfer_gbp_2024": float(
                    case_routes["at_cost_rent_transfer_gbp_2024"].get(route, 0.0)
                    if "at_cost_rent_transfer_gbp_2024" in case_routes else 0.0),
                "bill_change_gbp_per_3100kwh_ex_vat": float(bill_changes.get(route, 0.0)),
                "bill_change_gbp_per_3100kwh_inc_vat": float(bill_changes.get(route, 0.0)) * (1.0 + vat_rate),
                "financing_saving_total_gbp_2024": float(financing.get(case_id, financing.get(base, 0.0))),
                "route_identity_residual_gbp_2024": case_report[case_id]["route_identity_residual_gbp_2024"],
                "bill_identity_residual_gbp_2024": float(sum(abs(v) for v in residuals.values())) if route == ROUTE_CFD_BILL else 0.0,
            })

    by_case = pd.concat([components[c] for c in all_cases], ignore_index=True)
    report = {
        "forecast_case": case["case_id"],
        # P1/P2 under central dispatch (20 Oct 2026): the institution the bills settle under.
        "institution": case.get("institution", "market"),
        "cases": case_report,
        "central_cases": cases,
        "sensitivity_cases": variants,
        "cases_not_projected_through_the_market_bill": excluded_cases,
        "levers_by_case": {c: r["levers"] for c, r in case_report.items()},
        "private_reproduces_forecast_case": True,
        **({"gross_bill_order_findings": gross_order_findings} if vintaged_year else {}),
        "new_lines": list(NEW_LINES),
        "comparator_case_ids": comparator_case_ids,
        "sensitivity_axes_by_case": {c: r["sensitivity_axes"] for c, r in case_report.items()},
        "identity_tolerance_gbp": IDENTITY_TOLERANCE_GBP,
        "benchmark_kwh": float(ps["annual_consumption_kwh"]),
        "price_basis": ps["currency"],
        # DEPRECATED on the two-volume basis (F18): read each case's line_recovery
        # (scripts/public_ownership/bill_recovery.py); kept for the D1 (2050) bill.
        "cfd_recovery_denominator_mwh": eligible,
        "final_demand_recovery_denominator_mwh": final_demand,
        # F19 (7 Oct 2026): the private bill's recovery basis and the Exchequer's per-household key.
        "recovery_basis": ps.get("recovery_basis"),
        "exchequer_per_household_key": ps.get("exchequer_per_household_key"),
        "model_rebasing_factor": factor,
        "public_account_settings": (public_account or {}).get("settings"),
        # Review 29 Sep 2026: a plain-language source and link for every bill line.
        "line_sources": dict(line_sources or {}),
        "domestic_supply_mwh": domestic_mwh,
        "notes": [
            "Gross = bill before the public-account recycling and shortfall rows; net = after them. Boundary variants "
            "(<case>@<boundary>) are full cases with their own account; eligibility-only variants are gross only.",
            "R1 takes the capital-charge change annuitised over the configured term; the LDES line takes the change in floor top-up; "
            "W1 returns the state's merchant rent including the pre-existing rent it acquires (a transfer).",
            "Item K1: three scenarios (private, public_control, public_ownership) with sensitivities underneath, each "
            "<scenario>@<label> moving one lever off the scenario bundle, named multi-lever positions (company_model, "
            "fully_nationalised) and <scenario>@all_alternatives moving the six structural levers at once. A lever that "
            "moves the private baseline (new_build_contracts, ldes_floor_basis, private_wacc_basis, wholesale_pricing) "
            "is a change of comparator basis and has its own private column, named in comparator_case_ids.",
            "Cases that acquire the legacy fleet take the acquisition tariff (item N5), not this bill; they are listed in "
            "cases_not_projected_through_the_market_bill.",
            "route_booking = booked adds the Capacity Market and CCS-levy bookings as their own rows; the line they "
            "adjust is reported net in capacity_market_line_net_gbp_per_3100kwh_ex_vat and "
            "allocated_ccs_line_net_gbp_per_3100kwh_ex_vat.",
            "Within this group the constraint line is shared by every case (house rule 2): each case settles the "
            "group's own months. Item BM1 makes constraint costs respond to ownership through the balancing_bids "
            "lever: the state-at-cost bid stack is solved as its own run group and set beside these cases by "
            "merge_balancing_groups.py (cp30_2030_cp30_network_consumer_bill_balancing_cases), which splits it into market "
            "private, a private-on-stack decomposition term (not a feasible case, never a saving) and the ownership effect; its bid_stack_effect is the P2 increment. Never add any of them to a case here.",
            "Every line that moves reconciles to its route within the identity tolerance, or the script refuses to write.",
        ],
    }
    return by_case, report, pd.DataFrame(reconciliation_rows)


def domestic_supply_mwh(assumptions):
    """Ofgem's WHD-obligated domestic supply (the only domestic volume the bill declares), or None."""
    whd = (((assumptions or {}).get("sourced_externals") or {}).get("warm_home_discount") or {}).get("derivation") or {}
    value = whd.get("domestic_electricity_supply_mwh")
    return float(value) if value is not None else None


def load_line_sources(path):
    """``{component_id: {source_type, source_label, source_url}}`` from bill_line_sources.csv (plain language)."""
    if not path or not Path(path).exists():
        return {}
    frame = pd.read_csv(path).fillna("")
    return {str(r.component_id): {"source_type": str(r.source_type), "source_label": str(r.source_label),
                                  "source_url": str(r.source_url)} for r in frame.itertuples(index=False)}


def write_ownership_cases(
    case, cases, assumptions_path, networks, prices, summaries, physical_networks, annual_summary,
    balancing_costs, recovery_demand, strike_table_path, ro_scheme_costs, adjustments_path, routes_path,
    financing_summary_path, reference_components_path, output_csv, output_json, output_reconciliation,
    public_account_path=None, network_lines_path=None, excluded_cases=None, line_sources_path=None,
    bound_lines=None,
):
    public_account = (
        json.loads(Path(public_account_path).read_text(encoding="utf-8")) if public_account_path else None
    )
    network_lines = pd.read_csv(network_lines_path) if network_lines_path else None
    by_case, report, reconciliation = project_ownership_cases(
        case, list(cases), assumptions_path, list(networks), list(prices), list(summaries),
        list(physical_networks), annual_summary, balancing_costs, recovery_demand, strike_table_path,
        ro_scheme_costs, pd.read_csv(adjustments_path), pd.read_csv(routes_path),
        pd.read_csv(financing_summary_path), pd.read_csv(reference_components_path),
        public_account=public_account, public_account_source=str(public_account_path or "public account"),
        network_lines=network_lines, excluded_cases=excluded_cases, line_sources=load_line_sources(line_sources_path),
        own_account_path=public_account_path, bound_lines=bound_lines,
    )
    for path in (output_csv, output_json, output_reconciliation):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    by_case.to_csv(output_csv, index=False)
    Path(output_json).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    reconciliation.to_csv(output_reconciliation, index=False)
    return by_case, report, reconciliation


if "snakemake" in globals():
    write_ownership_cases(
        snakemake.params.case, snakemake.params.cases, snakemake.input.assumptions,
        snakemake.input.networks, snakemake.input.prices, snakemake.input.summaries,
        snakemake.input.physical_networks, snakemake.input.annual_summary, snakemake.input.balancing_costs,
        snakemake.input.recovery_demand, snakemake.input.strike_table, snakemake.input.ro_scheme_costs,
        snakemake.input.adjustments, snakemake.input.routes, snakemake.input.financing_summary,
        snakemake.input.reference_components, snakemake.output.by_case, snakemake.output.summary,
        snakemake.output.reconciliation, public_account_path=snakemake.input.public_account,
        network_lines_path=snakemake.input.network_lines, excluded_cases=snakemake.params.tariff_cases,
        line_sources_path=snakemake.input.line_sources,
        bound_lines=getattr(snakemake.params, "bound_lines", None),
    )
