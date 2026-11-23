"""Build the data file for the Public Ownership Bill Explorer from the public-ownership outputs.

Reads a run's public-ownership outputs and writes a compact JSON the page renders from. Nothing is
invented: every figure is read from a file or derived by an arithmetic rule declared here (VAT added at
the ledger's rate, system GBP converted to GBP per household with the ledger's own key, a change against
the base taken as the difference of two changes against private).

Rule ``build_public_ownership_explorer`` runs this and ``build_explorer.py``; publishing the page is a
manual step. Usage (defaults are this repository's outputs)::

    python scripts/analysis/public_ownership_explorer/extract_explorer_data.py [--po DIR] [--sources CSV]
        [--bill-summary JSON] [--energy CSV] [--dispatch-validation CSV] [--config YAML | --no-config]
        [--run-label TEXT] [--out JSON]

The case list (30 Sep 2026 rebase). Every scenario's base is "bill-payers first". The explorer's
switches are the exact toggle grids of ``public_ownership.sensitivities.toggle_grids``: switches,
selectors and conditional switches. Every combination maps to exactly one case through the position
vectors (``grid_memberships``; README of the rebase, section 2). The map is taken from the case catalogue
of ``--config`` (``scripts/public_ownership/levers.py``, imported read-only), which holds every case id
whether or not this run carries its numbers. It is cross-checked against the run's own memberships
(household totals and the ledger's ``case_catalogue``). Without ``--config`` the run's memberships alone
are used. A combination the run has no numbers for is shown as "not in this build"; a combination that maps to no
case id at all is an error.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[2]

# ---------------------------------------------------------------------------------------------------
# Declarations: the words on the page, keyed by case id, preset, grid token and lever value.
# ---------------------------------------------------------------------------------------------------
P1, P2 = "public_control", "public_ownership"
P3 = "public_ownership@fully_nationalised_market_value"
#: 10 Nov 2026: the pathway each modelled year's runs follow (None: any other year, the FES long-range chain).
PATHWAY_NAMES = {2030: ("Clean Power 2030", "NESO's Clean Power 2030 pathway"),
                 None: ("FES 2025 Holistic Transition {year}",
                        "NESO's Future Energy Scenarios 2025 Holistic Transition pathway")}

SCENARIOS = {
    P1: {"code": "S1", "label": "Public control",
         "short": "A 30% state stake in new plant",
         "sub": "GB Energy and the National Wealth Fund take a 30% stake in every power station committed once the law is passed in 2028. "
                "Private companies own the other 70%, and all the plant that exists today stays private.",
         "mechanism": "The state borrows more cheaply than private investors, so the contract price that new plant "
                      "needs falls. That saving goes on the bill; the state's own costs stay with the Exchequer."},
    P2: {"code": "S2", "label": "Public ownership, new build only",
         "short": "The state owns all new plant",
         "sub": "A state developer owns every power station committed once the law is passed in 2028, borrows at the government's rate and "
                "sells the power at what it costs. The plant that exists today stays private.",
         "mechanism": "New wind and solar farms are paid what they cost to build and run, not what a private "
                      "developer needs to earn. That saving goes on the bill; the state's own costs stay with "
                      "the Exchequer."},
    P3: {"code": "S3", "label": "New build + nationalisation",
         "short": "Existing plant bought at market value, plus all new plant",
         "sub": "On top of owning all new plant, the state buys the power stations that exist today at market value, "
                "and Britain's half of the interconnectors. Households then pay what the system costs to run, "
                "including the yearly repayments on the purchase.",
         "mechanism": "Households stop paying market prices and subsidy contracts. They pay the fleet's running costs "
                      "and the repayments on the purchase instead."},
}
NETWORKS_NOTE = ("The wires are a separate choice in every scenario. In each base the transmission and distribution "
                 "networks stay private. The switches let the state guarantee the network companies' debt, or buy "
                 "the wires outright.")
BASE_RULE = ("Every scenario's base is \"bill-payers first\": every public saving goes on the bill, and every public "
             "cost, and any shortfall, stays with the Exchequer. In P3 the repayments on the purchase stay on the bill.")

# Who pays. `as_designed` is the base itself (no case of its own).
PRESETS = [
    {"key": "as_designed", "label": "Bill-payers first (base)",
     "note": "Every public saving goes on the bill; every public cost stays with the Exchequer.",
     "note_by": {P3: "Households pay what the system costs, repayments on the purchase included; the state's other "
                     "costs stay with the Exchequer."}},
    {"key": "todays_rules", "label": "Today's rules",
     "note": "The rules the scenario had before bill-payers first became the base.",
     "note_by": {P1: "The Treasury keeps the state's return on its stake and carries any loss.",
                 P2: "The state's plant sells at cost through contracts on the bill, and any shortfall is charged to "
                     "bill-payers.",
                 P3: "For this scenario today's rules are the base itself: the purchase is repaid through the bill."}},
    {"key": "competitive_neutrality", "label": "State acts like a company",
     "note": "The state keeps its surplus at the Treasury as a dividend, pays the taxes a private owner would, and "
             "charges its own losses to bills.",
     "note_by": {P3: "The state's new plant pays the corporation tax a private owner would; the purchase stays on "
                     "the bill."}},
    {"key": "treasury_first", "label": "Treasury first",
     "note": "The state keeps the auction contract price and its financing margin; the Treasury keeps any surplus.",
     "note_by": {P3: "The state charges households the private market bill and the Treasury keeps the margin over "
                     "cost."}},
    {"key": "purchase_on_taxation", "label": "Purchase paid from taxation", "only": [P3],
     "note": "The repayments on the purchase are paid from general taxation instead of the bill."},
]

# The switches of each exact grid, keyed by the grid tokens of config/defaults/ (public_ownership/sensitivities.yaml). Every token of the
# live grid must be declared here (the extractor refuses otherwise), so no switch reaches the page unworded.
RATE = {"label": "State borrows at the PWLB standard rate",
        "explain": "The state's lending margin is the Public Works Loan Board's standard 100 basis points over gilts, "
                   "instead of the National Loans Fund's 11: 3.60% real instead of 2.71%."}
OVERHEAD = {"label": "Dearer to run the state company",
            "explain": "Head-office costs of £8 per kW of state plant a year instead of £3."}
OM = {"label": "State runs its plant 10% dearer",
      "explain": "Operation and maintenance cost 10% more than under private owners."}
NETWORK_OPTIONS = {
    "private": {"label": "Private (base)", "explain": "The network companies stay private and finance themselves as today."},
    "acquired": {"label": "State buys the wires",
                 "explain": "The state buys the transmission and distribution networks at their regulated value plus a "
                            "50% premium and finances them at the public rate."},
}
PRE_2025 = {"label": "State takes over the pre-2025 contract projects",
            "explain": "Wind and solar farms that won their subsidy contracts before 2025 come under the scenario's "
                       "ownership too, and their contract prices are reset by the same rule as new plant's."}
GRID_UI = {
    P1: {
        "stake-15": {"group": "other", "label": "A 15% stake instead of 30%",
                     "explain": "The stake GB Energy and the National Wealth Fund could fund from their own capital."},
        "overhead-high": {**OVERHEAD, "group": "other"},
        "networks": {"group": "wires", "label": "Who owns the wires", "options": NETWORK_OPTIONS},
        "pre-2025-repriced": {**PRE_2025, "group": "contracts"},
        "strike": {"group": "contracts", "label": "How the contract price is set",
                   "options": {
                       "saving": {"label": "Cut by the financing saving (base)",
                                  "explain": "The contract price falls by what cheaper state borrowing saves."},
                       "cost": {"label": "Cut to cost",
                                "explain": "The state's 30% is paid what it costs to build and run; the co-owners keep "
                                           "the auction price on their share."},
                       "auction": {"label": "Keep the auction price",
                                   "explain": "The contract keeps the price it won at auction; the state keeps its "
                                              "return, which under bill-payers first goes back to the bill."}}},
    },
    P2: {
        "delivered-half": {"group": "other", "label": "The state builds only half the new plant",
                           "explain": "The state builds half of each technology's new plant; private developers build "
                                      "the rest on normal contracts."},
        "rate-pwlb-standard": {**RATE, "group": "other"},
        "overrun-15": {"group": "other", "label": "State projects overrun by 15%",
                       "explain": "Construction costs 15% more than planned, and the state bears it."},
        "om-plus-10": {**OM, "group": "other"},
        "overhead-high": {**OVERHEAD, "group": "other"},
        "networks": {"group": "wires", "label": "Who owns the wires", "options": NETWORK_OPTIONS},
        "pre-2025-repriced": {**PRE_2025, "group": "contracts"},
    },
    P3: {
        "rate-pwlb-standard": {**RATE, "group": "other",
                               "explain": RATE["explain"] + " It applies to the purchase and to the new plant."},
        "control-premium-20": {"group": "other", "label": "20% premium to buy the existing plant",
                               "explain": "The state pays 20% over market value to take control, a premium of the kind "
                                          "paid in takeovers (EDF's 2022-23 buy-out paid 34%)."},
        "hpc-completion": {"group": "other", "label": "State completes Hinkley Point C",
                           "explain": "The state takes over the unfinished plant, pays EDF what it has spent and "
                                      "finishes it at the public cash rate, instead of buying it as a running plant on its "
                                      "contract."},
        "om-plus-10": {**OM, "group": "other", "explain": "Operation and maintenance cost 10% more across the whole fleet."},
        "overhead-high": {**OVERHEAD, "group": "other"},
        "networks": {"group": "wires", "label": "Who owns the wires", "options": NETWORK_OPTIONS},
        "rav-premium-high": {"group": "wires", "label": "Wires bought at a 75% premium",
                             "explain": "The price of the wires is 75% over their regulated value instead of 50%.",
                             "requires": "Only when the state buys the wires."},
    },
}
GRID_NOTES = {
    P2: "There is no choice of contract price rule here. The state owns 100% of its plant, so whichever rule sets the "
        "price, every pound it leaves with the state comes back to the bill: every rule gives the same bill.",
}

# The axes of the bound grids (OW7 item 14): every assumption the best and worst cases move, as an exact grid
# named "<parent>#bounds". Keyed by the bound-grid tokens of config/defaults/ (toggle_grids.bound_grids);
# every token must be declared here, so no axis reaches the page unworded. Which option is the best and which
# the worst is read from the best and worst cases' own settings, not declared.
BOUND_UI = {
    "wacc": {"label": "Private investors' cost of capital", "options": {
        "notional": {"label": "Fixed equity premium (base)",
                     "explain": "The equity risk premium held at 5.22% over today's gilt yields, a notional 55% share "
                                "of debt for every technology (the regulators' convention), with tax relief on the "
                                "interest: the workbook's lead basis."},
        "tmr": {"label": "Fixed total market return",
                "explain": "The total market return held at 6.83% (the regulators' convention, co-equal in the "
                           "workbook), so the premium is what is left. Private plant costs less to finance, so the "
                           "private bill falls too; positions on this option are measured against a private system "
                           "with the same change."}}},
    "ldes-wacc": {"label": "Long-duration storage support at its own cost of capital",
                  "explain": "The storage floor pays the technology's own cost of capital, not a debt-like rate. It "
                             "changes the private bill too, so it is measured against a private system with the same "
                             "change."},
    "overhead": {"label": "Cost of running the state company", "options": {
        "central": {"label": "£3 per kW a year (base)", "explain": "Head-office costs of £3 per kW of state plant a year."},
        "low": {"label": "£1 per kW a year", "explain": "Strong economies of scale: head-office costs of £1 per kW."},
        "high": {"label": "£8 per kW a year", "explain": "An offshore wind farm's onshore admin rate applied to the "
                                                         "whole fleet: £8 per kW."}}},
    "yield-b": {"label": "Government borrowing costs rise as state debt grows",
                "explain": "The workbook's yield path B: gilt yields, and every public rate, rise with state borrowing "
                           "and the guarantees the state stands behind."},
    "rate-pwlb-standard": dict(RATE),
    "premium": {"label": "Price of the risk the state takes on (off the bill)", "options": {
        "social": {"label": "1.9% a year",
                   "explain": "The workbook's social price, the Green Book discount rate review's estimate: the risk "
                              "the state carries on its own plant costs taxpayers 1.9% a year on every technology. "
                              "Bills do not move."},
        "none": {"label": "None (base)", "explain": "Pure Arrow-Lind, as in today's Green Book: the risk costs taxpayers "
                                                    "nothing; the central case since 10 Nov 2026 (gilts alone). Bills do not "
                                                    "move."},
        "erp": {"label": "The market's price",
                "explain": "The market equity premium, 5.22% per unit of risk: what a commercial investor would "
                           "charge. Bills do not move."}}},
    "ic": {"label": "Price of Britain's half of the interconnectors", "options": {
        "central": {"label": "Central (base)",
                    "explain": "Their after-tax income discounted at the owners' rate of 4.87%, the workbook's "
                               "interconnector rate."},
        "low": {"label": "Lower price", "explain": "Their income discounted at 5.87%."},
        "high": {"label": "Higher price", "explain": "Their income discounted at 3.87%."}}},
    "price-state-value": {"label": "Existing fleet bought at the state's own valuation",
                          "explain": "Its income discounted at the state's rate instead of its owners': a higher "
                                     "price."},
    "hpc-completion": {"label": "State completes Hinkley Point C",
                       "explain": "Instead of buying it as a running plant on its contract."},
    "txn-2pct": {"label": "Advisers' and transaction costs of 2%",
                 "explain": "2% of the purchase price, repaid like the purchase."},
    # 13 Nov 2026 (user, results review): 2% is in the central purchase, so the bound grid's switch removes it.
    "txn-none": {"label": "No advisers' or transaction costs on the purchase",
                 "explain": "The central purchase carries 2% of its price for advisers and transaction costs, repaid "
                            "like the purchase; this switch removes them (the best case)."},
    # 13 Nov 2026 (user, results review): Scenario 2's worst case prices the risk the state takes on.
    "premium-erp": {"label": "Risk the state takes on priced at the market's rate (off the bill)",
                    "explain": "The market equity premium, 5.22% per unit of risk, on the plant the state owns: what a "
                               "commercial investor would charge. A cost to taxpayers; bills do not move."},
    "overrun-15": {"label": "State build costs 15% over budget",
                   "explain": "A construction overrun of 15% of capex on the new build the state owns (Sovacool et al. "
                              "2014; HM Treasury optimism bias), borne by the state."},
    "om-plus-10": {"label": "State runs its plant at 10% higher cost",
                   "explain": "Fixed and variable O&M 10% above a private owner's on the plant the state owns "
                              "(Newbery & Pollitt 1997, 6%; IMF 2020, about 15%)."},
    "lives-extended": {"label": "Existing plant expected to run longer",
                       "explain": "Renewables 5 years longer and Sizewell B to 2055, which raises the market-value "
                                  "price."},
}

# P3's headline is the central-dispatch case (user decision 21); the explorer's switches, bounds and
# one-at-a-time rows stay on the market-machinery case, where they are computed.
# The "provisional (being checked)" status was dropped on 21 Oct 2026 (user decision): the market and
# central-dispatch years are one run and the central dispatch settlement is rerun. The note names what the dispatch
# step rests on; the page puts the step's size (headline less the market-machinery case) in front of it.
HEADLINE = {P3: {"case": f"{P3}_central_dispatch", "status": None,
                 "label": "with a public operator dispatching the system",
                 "alt_label": "with the market machinery kept",
                 "note": ("Most of the step is better use of the interconnectors. It rests on two assumptions. Under "
                          "the market, the balancing stage undoes interconnector flows at £20/MWh on top of pricing "
                          "foreign plant like GB gas plant in the balancing mechanism (offers at 1.6 times cost; set "
                          "so the model matches NESO's constraint cost), and central "
                          "dispatch never has to undo them. And because the model's central dispatch sees the whole "
                          "year perfectly, it carries a benchmark allowance for forecast errors, taken from "
                          "the 2026 REMA scheduling and dispatch assessment for NESO (about £13 a household).")}}

# One-at-a-time sensitivities (OW7 item 10, 2 Oct 2026): generated from the case catalogue, not typed. Every
# one-at-a-time and named case of a scenario's family is listed under it, except those already reachable as a
# position of the scenario's switch grid. A case that buys the existing fleet belongs to P3 (the scenario that
# buys it); any other case to its nearest scenario ancestor. The words are keyed on what the case moves:
#   SENS_BY_LEVER   (lever, value) for a case that moves one lever off its scenario (or a one-at-a-time case);
#   SENS_BY_NAME    the named case's own name, for a named case that moves several levers at once.
# Each entry is {group, label, explain[, by: {scenario: {overrides}}]}; text may carry {placeholders} filled from
# the workbook snapshot and the configuration (sens_constants). A catalogue case with no words fails the build,
# so the page never shows a lever id.
SENS_GROUPS = ["When the law is passed", "Purchase price and term", "How the system is run", "How much new plant the state builds",
               "Steps from P2 towards P3", "Borrowing",
               "Private investors' cost of capital", "Contracts", "The wires", "Storage", "Running costs",
               "Balancing the grid", "GB Energy as a company", "GB Energy as announced", "Old levies and other background",
               "Earlier, more cautious versions"]
COMPARATOR_NOTE = "It changes the private bill too, so it is measured against a private system with the same change."
SENS_BY_LEVER = {
    # 2050 (10 Nov 2026): the FES 2050 study's own one-at-a-time cases.
    **{("state_deployment_share", share): {
        "group": "How much new plant the state builds",
        "label": f"The state builds {int(round(float(share) * 100))}% of new plant",
        "explain": (f"The state builds and owns {int(round(float(share) * 100))}% of the plant built after the "
                    "legislation year, whole sites spread across every technology and build year; private developers "
                    "build the rest on today's terms.")}
       for share in ("0.25", "0.50", "0.75", "1.00")},
    ("private_rate_path", "neso_wind_ramp"): {
        "group": "Private investors' cost of capital",
        "label": "Private offshore wind finance gets cheaper (NESO's wind ramp)",
        "explain": "Private investors' cost of capital for offshore wind falls by 0.4 points in 2031 and 2 points "
                   "from 2035, as in NESO's wind ramp, and contract prices are struck on those cheaper rates."},
    ("capacity_market_design", "market_participant"): {
        "group": "How the system is run",
        "label": "State plant stays in the Capacity Market auction",
        "explain": "The state's power stations bid in the Capacity Market and are paid like any other plant, as today, "
                   "instead of leaving the auction while the market buys only the capacity the state does not own.",
        "by": {P3: {"explain": "Today's Capacity Market agreements carry on for the plant the state does not buy, "
                               "instead of a targeted contract for that capacity at the residual price."}}},
    # Borrowing
    ("public_rate_basis", "pwlb_hra"): {
        "group": "Borrowing", "label": "State borrows at the PWLB housing rate",
        "explain": "The state's cash rate with the Public Works Loan Board's housing revenue account margin, 40 basis "
                   "points over gilts ({rate_pwlb_hra}), instead of the National Loans Fund's 11 ({rate_central})."},
    ("public_rate_basis", "pwlb_certainty"): {
        "group": "Borrowing", "label": "State borrows at the PWLB certainty rate",
        "explain": "The PWLB certainty margin, 80 basis points over gilts ({rate_pwlb_certainty}), instead of "
                   "{rate_central}."},
    ("public_rate_basis", "pwlb_standard"): {
        "group": "Borrowing", "label": "State borrows at the PWLB standard rate",
        "explain": "The PWLB standard margin, 100 basis points over gilts ({rate_pwlb_standard}), instead of "
                   "{rate_central}."},
    ("public_rate_basis", "gilt_flat"): {
        "group": "Borrowing", "label": "State borrows at the gilt rate",
        "explain": "The real gilt with no lending margin ({rate_gilt_flat}), instead of {rate_central}."},
    ("systematic_premium", "none"): {
        "group": "Borrowing", "label": "No price on the risk the state takes on",
        "explain": "The risk the state carries on its own plant is not charged to taxpayers (pure Arrow-Lind, as in "
                   "today's Green Book), instead of {premium_p} a year. Bills do not move; "
                   "the Exchequer column and the saving to the country do."},
    # 13 Nov 2026: the central is gilts alone (10 Nov), so 1% is a price added to it, not a lower one.
    ("systematic_premium", "social_flat_10"): {
        "group": "Borrowing", "label": "Gilts plus a 1% premium for the risk taxpayers carry",
        "explain": "The state's plant is charged a flat 1% a year on top of the gilt-based borrowing rate (the rate "
                   "HM Treasury adopted in 2026 after the Green Book discount rate review, whose reviewers estimated "
                   "1.9%), as a cost to taxpayers (never on the bill). The central case uses gilts alone."},
    ("systematic_premium", "social_beta_20"): {
        "group": "Borrowing", "label": "The risk the state takes on, priced by technology",
        "explain": "2% per unit of the technology's market risk (France's rule), so riskier plant costs more, instead "
                   "of a flat {premium_p}. Bills do not move."},
    ("systematic_premium", "market_erp"): {
        "group": "Borrowing", "label": "The risk the state takes on, at the market's price",
        "explain": "The market's equity risk premium, 5.22% per unit of the technology's risk: what a commercial "
                   "investor would charge. Bills do not move; most of the saving to the country goes."},
    ("sovereign_yield", "B"): {
        "group": "Borrowing", "label": "Government borrowing costs rise as state debt grows",
        "explain": "The workbook's yield path B: gilt yields rise with the state's extra debt and the guarantees it "
                   "stands behind, and every public borrowing rate rises with them, by {yield_b_shift} by 2030.",
        "by": {P1: {"explain": "The workbook's yield path B: gilt yields rise with the extra state borrowing a 30% "
                               "stake needs, and every public borrowing rate rises with them, by {yield_b_shift_p1} "
                               "by 2030."},
               P2: {"explain": "The workbook's yield path B: gilt yields rise with the extra state borrowing, and "
                               "every public borrowing rate rises with them, by {yield_b_shift_p2} by 2030."}}},
    # Private investors' cost of capital (both move the private bill too)
    ("private_wacc_basis", "fixed_tmr_shielded"): {
        "group": "Private investors' cost of capital",
        "label": "Private investors' cost of capital on the regulators' convention",
        "explain": "The workbook's co-equal case: the total market return held at 6.83% and the equity premium taken "
                   "as what is left (4.23% at today's gilt yields), CEPA's and UK regulators' convention, instead of "
                   "holding the premium at 5.22% (offshore wind {off_tmr} instead of {off_obs})."},
    # Contracts
    ("cfd_amortisation", "economic_life"): {
        "group": "Contracts", "label": "Contract costs spread over the plant's life (the reverse of P1's base)",
        "explain": "P1's base already repays the re-priced contract over its 20-year term, so this row shows the "
                   "reverse: the same contract spread over the plant's economic life."},
    ("cfd_amortisation", "contract_term"): {
        "group": "Contracts", "label": "Contract costs repaid over the contract term",
        "explain": "The base spreads the at-cost contract over the plant's economic life; this repays it over the "
                   "20-year contract."},
    ("cfd_strike_basis", "financing_saving_only"): {
        "group": "Contracts", "label": "Contract price cut only by the financing saving",
        "explain": "The state's plant keeps a contract price cut by its cheaper borrowing, instead of one set at "
                   "cost. Under bill-payers first whatever the price leaves with the state is returned to the bill, "
                   "so the bill does not change."},
    ("cfd_strike_basis", "at_cost"): {
        "group": "Contracts", "label": "Contract price cut to cost",
        "explain": "The state's share is paid what it costs to build and run."},
    ("pre_2025_cfd_capacity", "public_repriced"): {
        "group": "Contracts", "label": "State takes over the pre-2025 contract projects",
        "explain": "Wind and solar farms that won their contracts before 2025 come under the scenario's ownership "
                   "too, and their contract prices are reset by the same rule as new plant's."},
    ("pre_2025_cfd_capacity", "public_bought_owners_value"): {
        "group": "Contracts", "label": "State buys the pre-2025 contract projects at their owners' value",
        "explain": "As the takeover above, but the owners are paid what the farms are worth to them: their cost plus "
                   "the profit above their required return that their contracts would still have earned. The gap "
                   "to the takeover at cost is what the owners keep."},
    ("route_booking", "sensitivity_only"): {
        "group": "Contracts", "label": "Capacity Market and carbon-capture savings reported, not put on the bill",
        "explain": "Cheaper state borrowing on new gas plant, batteries and carbon capture lowers what the Capacity "
                   "Market and the carbon-capture levy cost. In the base that saving goes on the bill; here it is "
                   "reported beside it instead."},
    ("new_build_contracts", "at_cost"): {
        "group": "Contracts", "label": "New merchant plant contracted at cost in every case",
        "explain": "New plant that would sell its power on the open market is put on a contract at its cost "
                   "instead, in the private system as much as the public one."},
    # The wires (one-lever forms; the grid carries them, but a named case may land here)
    ("networks", "acquired"): {
        "group": "The wires", "label": "State buys the wires",
        "explain": "The state buys the transmission and distribution networks at their regulated value plus a "
                   "{rav_central} premium and finances them at the public rate."},
    # Storage
    ("ldes_floor_basis", "technology_wacc"): {
        "group": "Storage", "label": "Long-duration storage support at its own cost of capital",
        "explain": "The floor that supports long-duration storage pays the technology's own cost of capital instead "
                   "of a debt-like rate ({ldes_private} for a private project)."},
    # Contracts: the auction margin (follow-up F6, 4 Oct 2026)
    ("auction_margin", "none"): {
        "group": "Contracts", "label": "No auction margin: CfD projects just break even",
        "explain": "The offshore wind, tidal, onshore wind and solar projects calibrated to their auction prices "
                   "keep no margin above "
                   "their full cost, instead of 5% of their contract revenue. In the private system as much as the "
                   "public one, so it is measured against private at the same margin."},
    ("auction_margin", "high"): {
        "group": "Contracts", "label": "Bigger auction margin (10%)",
        "explain": "Auctions pay every winner the clearing price, so cheaper projects keep a margin. Here they keep "
                   "10% of their contract revenue above full cost instead of 5%; the state's own plant needs none."},
    # Running costs
    ("self_insurance", "none"): {
        "group": "Running costs", "label": "State plant buys commercial insurance",
        "explain": "Plant the state owns outright pays the full commercial insurance premium, as a private owner "
                   "would, instead of saving the insurer's costs and margin (35% of the premium). " + "Managing Public Money does not bind a public corporation to self-insure (A7.3.13), and a body that charges for its output prices a notional premium into its charges (A4.4.4); the 'none' value is the public-corporation reading."},
    ("self_insurance", "high"): {
        "group": "Running costs", "label": "State plant saves half its insurance premiums",
        "explain": "Central government carries its own risks rather than insuring commercially. Here plant the "
                   "state owns outright saves 50% of its insurance premium instead of 35%; it still bears the "
                   "expected claims."},
    ("state_overhead", "low"): {
        "group": "Running costs", "label": "Cheaper to run the state company",
        "explain": "Head-office costs of {overhead_low} per kW of state plant a year instead of {overhead_central} "
                   "({overhead_storage_low} instead of {overhead_storage_central} on unstaffed storage)."},
    ("state_overhead", "high"): {
        "group": "Running costs", "label": "Dearer to run the state company",
        "explain": "Head-office costs of {overhead_high} per kW of state plant a year instead of {overhead_central} "
                   "({overhead_storage_high} instead of {overhead_storage_central} on unstaffed storage)."},
    # Balancing the grid
    ("balancing_bids", "state_at_cost"): {
        "group": "Balancing the grid", "label": "State plant asks only its costs to turn down",
        "explain": "When the grid is congested, the state's plant asks to be paid only its costs for reducing output, "
                   "not its contract value. Computed on its own solved year."},
    # P3: purchase price and term
    ("purchase_price_basis", "state_present_value"): {
        "group": "Purchase price and term", "label": "Priced at the state's own valuation",
        "explain": "The fleet's future income valued at the public borrowing rate, which gives a higher price than "
                   "market value."},
    ("purchase_price_basis", "book_value"): {
        "group": "Purchase price and term", "label": "Bought at book value",
        "explain": "Every plant bought at its net book value: the financing workbook's historic cost by cohort, "
                   "written down over its accounting life."},
    ("purchase_price_basis", "no_support"): {
        "group": "Purchase price and term", "label": "Bought at its value without subsidy contracts",
        "explain": "Every plant bought at what its owners would value it at without its CfD and Renewables "
                   "Obligation income, the value without the support government put in."},
    ("purchase_price_basis", "ro_pass_through"): {
        "group": "Purchase price and term", "label": "Renewables Obligation plant bought without its subsidy",
        "explain": "Plant on the Renewables Obligation bought at its value without that income, the rest at its "
                   "owners' value: a policy choice that takes value from the RO plant's shareholders."},
    ("acquisition_term_years", "15"): {
        "group": "Purchase price and term", "label": "Purchase repaid over 15 years",
        "explain": "The whole purchase is repaid over 15 years instead of each plant over its remaining life, so the "
                   "early years' repayments are larger."},
    ("acquisition_term_years", "30"): {
        "group": "Purchase price and term", "label": "Purchase repaid over 30 years",
        "explain": "The whole purchase is repaid evenly over 30 years instead of each plant over its remaining life: "
                   "the same cost in today's money, spread later, so the 2030s pay less of it."},
    # The purchase's scope (user decisions, 10 Nov 2026).
    ("in_construction", "left_private"): {
        "group": "Purchase price and term", "label": "Hinkley Point C left out of the purchase",
        "explain": "Hinkley Point C stays private on its own contract, instead of the state buying it as a running "
                   "plant in 2030: a lower purchase price, and its contract stays on the bill."},
    ("efw_acquisition", "included"): {
        "group": "Purchase price and term", "label": "Energy-from-waste plants bought too",
        "explain": "The purchase also buys the energy-from-waste plants at their owners' value. Their value is mostly "
                   "the gate fees councils pay to burn waste, which then go to the state; the central leaves them "
                   "with the waste companies and their council contracts."},
    ("systematic_premium", "social_flat_19"): {
        "group": "Borrowing", "label": "Gilts plus a premium for the risk taxpayers carry",
        "explain": "The state's plant is charged the workbook's systemic risk premium, a flat 1.9% a year on top of "
                   "the gilt-based borrowing rate, as a cost to taxpayers (never on the bill). The central case "
                   "uses gilts alone."},
    ("in_construction", "public_completion"): {
        "group": "Purchase price and term", "label": "State completes Hinkley Point C",
        "explain": "The state takes over the unfinished plant, pays EDF what it has spent and finishes it at the "
                   "public rate, instead of buying it as a running plant on its contract."},
    ("dispatch_institution", "nationalised_dispatch"): {
        "group": "How the system is run", "label": "A public operator dispatches the system",
        "explain": "A public operator schedules every plant centrally at cost, instead of keeping the market "
                   "machinery. Computed on its own solved year. This is P3's headline case."},
    ("legacy_fleet", "acquire_valued_group"): {
        "group": "Steps from P2 towards P3", "label": "P2 plus the existing renewables and nuclear",
        "explain": "Bought at the state's own valuation; gas, biomass and storage stay private."},
    ("legacy_fleet", "acquire_all"): {
        "group": "Steps from P2 towards P3", "label": "P2 plus the whole existing fleet",
        "explain": "Renewables, nuclear, gas, biomass and storage, bought at the state's own valuation."},
    # Background: every case alike, private included
    ("legacy_levy_funding", "domestic_to_taxation"): {
        "group": "Old levies and other background", "label": "Old levies moved to taxation",
        "explain": "Households' Renewables Obligation, Feed-in Tariff and Warm Home Discount costs are paid from "
                   "general taxation in private and public cases alike."},
    ("beccs_removal_funding", "bill_levy"): {
        "group": "Old levies and other background", "label": "Carbon removal at bioenergy plant paid through bills",
        "explain": "The carbon-removal half of the contract for bioenergy with carbon capture is paid by a levy on "
                   "bills instead of by the Exchequer, in private and public cases alike."},
    ("wholesale_pricing", "pay_as_bid_ceiling"): {
        "group": "Old levies and other background", "label": "Wholesale paid as bid (an upper bound)",
        "explain": "The most that paying each generator its own bid, instead of the clearing price, could take off "
                   "the private bill."},
    # OW9 (3 Oct 2026): the lever values no other case used, each a named case on P2.
    ("new_build_storage", "private"): {
        "group": "Storage", "label": "New batteries and storage stay private",
        "explain": "The state owns the new power plants but not the new storage, which private owners build and "
                   "finance as today."},
    ("surplus_recycling", "cost_recovery_contracting"): {
        "group": "Contracts", "label": "State plant's surplus returned through the wholesale price",
        "explain": "The state sells its plant's power under contracts at cost, so its market surplus comes off the "
                   "wholesale part of every bill; its contract surplus stays with the state."},
    ("surplus_recycling", "rebate_per_mwh_eligible_supply"): {
        "group": "Contracts", "label": "State's net surplus rebated per unit",
        "explain": "The state's net surplus, gains and losses together, is rebated on each unit of electricity "
                   "supplied, instead of only its gains."},
    ("surplus_recycling", "offset_policy_levies"): {
        "group": "Contracts", "label": "State's surplus pays households' old levies first",
        "explain": "The state's surplus first pays households' Renewables Obligation, Feed-in Tariff and Warm Home "
                   "Discount costs, and the rest is rebated on each unit supplied."},
    ("surplus_recycling", "no_assumption"): {
        "group": "Contracts", "label": "State's surplus not returned to bills",
        "explain": "Nothing the state's plant earns comes back to bills: the surplus stays with the state."},
    # F19 P3 named cases.
    ("legacy_price_path", "cp30_level"): {
        "group": "Purchase price and term", "label": "Fleet valued on CP30's 2030 prices",
        "explain": "The existing fleet's future income is valued on CP30's 2030 price level instead of the central "
                   "price path (today's forward prices, then this model's 2030 and 2050 prices): a higher purchase "
                   "price."},
    # User, 11 Nov 2026: today's forward prices held for the fleet's life.
    ("legacy_price_path", "today_level"): {
        "group": "Purchase price and term", "label": "Fleet valued on today's prices",
        "explain": "The existing fleet's future income is valued as if today's forward wholesale prices (Ofgem's "
                   "2026 price-cap wholesale index, about GBP 95/MWh) lasted its whole life, instead of the central "
                   "path that falls from today's prices to this model's 2030 and 2050 prices: a higher purchase "
                   "price, as owners would ask for if they expected prices to stay where they are."},
    # User decision (10 Nov 2026): the anchored path is central; holding 2030's prices is the named sensitivity.
    ("legacy_price_path", "model_2030"): {
        "group": "Purchase price and term", "label": "Fleet valued on 2030 prices held flat",
        "explain": "The existing fleet's future income is valued on this model's 2030 wholesale prices held flat for "
                   "every later year, instead of the central path, which runs in a straight line from today's "
                   "forward prices through this model's 2030 and 2050 prices."},
    # F27 item 6 (24 Oct 2026): P2 one at a time, P3 named.
    ("capacity_procurement", "cost_based_existing"): {
        "group": "Contracts", "label": "Existing plant paid its cost, not the Capacity Market price",
        "explain": "Existing plant needed for security is paid its going-forward fixed cost less what it earns in the "
                   "wholesale market and balancing, instead of the Capacity Market clearing price; new capacity is "
                   "still auctioned. Under P2 a transfer from the plant's private owners to bill payers; under P3 a "
                   "lower purchase price, if announced before the purchase."},
    # User decision (24 Oct 2026): P3 named.
    ("legacy_offshore_price", "transaction_multiple"): {
        "group": "Purchase price and term", "label": "Existing offshore wind bought at recent deal prices",
        "explain": "Existing offshore wind farms priced at the median price per megawatt of recent UK operating "
                   "offshore wind stake sales (about £4.6m in 2024 money), instead of the model's own value of their "
                   "cash flows (about £2.3m); every other plant at its owners' value: a higher purchase price."},
    # F10: GB Energy as announced (lever_parameters.gbe_announced).
    ("new_build_generation", "gbe_announced"): {
        "group": "GB Energy as announced", "label": "GB Energy as announced",
        "explain": "Only what GB Energy has announced: a 30% stake in the new offshore wind farms, on P1's terms, "
                   "and 8 GW of local solar and onshore wind wholly public; everything else, storage included, is "
                   "built and financed privately."},
    ("new_build_generation", "gbe_announced_partnership_half"): {
        "group": "GB Energy as announced", "label": "GB Energy as announced, local plant half public",
        "explain": "As announced, but the 8 GW of local solar and onshore wind is built in partnership: half public, "
                   "half private at the private rate."},
}
SENS_BY_NAME = {
    # 2050 (10 Nov 2026): P2 at a deployment share with the state's plant bidding at cost in balancing.
    **{f"deployment_share_{pct}_state_bids": {
        "group": "Balancing the grid",
        "label": f"The state builds {pct}% of new plant, and its plant bids at cost in balancing",
        "explain": (f"As the {pct}% deployment case, with the state's plant offering its turn-down in the balancing "
                    "mechanism at its own cost instead of at contract value (the twelve balancing months solved "
                    "again on that bid stack).")}
       for pct in ("25", "50", "75", "100")},
    # P1
    "gbe_value_to_bills": {
        "group": "GB Energy as a company", "label": "The stake's value to bills",
        "explain": "The 30% stake, with the state's position passed to bills through at-cost contracts and any "
                   "shortfall charged to them."},
    "conservative_stake": {
        "group": "Earlier, more cautious versions", "label": "P1 as first set up",
        "explain": "The stake as it was modelled before 22 September: the Capacity Market and carbon-capture savings "
                   "reported rather than put on the bill, the state's 30% selling through at-cost contracts, and any "
                   "shortfall charged to bill-payers."},
    "stake_networks_acquired_rav_premium_25pct": {
        "group": "The wires", "label": "State buys the wires at a 25% premium",
        "explain": "{wires_quarter}"},
    # P2
    "conservative_developer": {
        "group": "Earlier, more cautious versions", "label": "P2 as first set up",
        "explain": "The state developer as it was modelled before 22 September: the contract price cut only by the "
                   "financing saving and repaid over the 20-year contract, the Capacity Market and carbon-capture "
                   "savings reported rather than put on the bill, and the state's plant selling through at-cost "
                   "contracts with any shortfall charged to bill-payers."},
    "networks_acquired_rav_at_statute": {
        "group": "The wires", "label": "State buys the wires at their regulated value, set by Parliament",
        "explain": "{wires_statute}"},
    "networks_acquired_rav_premium_25pct": {
        "group": "The wires", "label": "State buys the wires at a 25% premium",
        "explain": "{wires_quarter}"},
    # OW9 (3 Oct 2026)
    "networks_acquired_rav_premium_low": {
        "group": "The wires", "label": "State buys the wires at the listed-market premium",
        "explain": "The networks bought at their regulated value plus {rav_low}, what the listed market pays for a "
                   "minority stake, instead of {rav_central}."},
    "networks_acquired_return_retained": {
        "group": "The wires", "label": "State buys the wires and the Treasury keeps their return",
        "explain": "The state buys the networks, but network charges stay as they are: the cheaper finance and the "
                   "tax allowance are paid to the Treasury instead of coming off bills."},
    # P3 and the steps towards it
    "fully_nationalised_thermal_at_zero": {
        "group": "Purchase price and term", "label": "Gas, biomass and storage for nothing",
        "explain": "The thermal and storage plant is taken over for nothing; the rest is priced at the state's own "
                   "valuation."},
    "fully_nationalised_term_15y": {
        "group": "Purchase price and term", "label": "Priced at the state's own valuation, repaid over 15 years",
        "explain": "The higher price of the state's own valuation, repaid over 15 years instead of 30."},
    "fully_nationalised_market_value_rav_at_statute": {
        "group": "The wires", "label": "State also buys the wires, at their regulated value set by Parliament",
        "explain": "{wires_statute}"},
    "fully_nationalised_market_value_rav_premium_25pct": {
        "group": "The wires", "label": "State also buys the wires, at a 25% premium",
        "explain": "{wires_quarter}"},
    "legacy_acquired_contracts_cancelled_on_bills": {
        "group": "Steps from P2 towards P3",
        "label": "Renewables and nuclear bought, contracts cancelled, owners compensated through bills",
        "explain": "The existing plant's subsidy contracts are cancelled rather than honoured, and bill-payers pay the "
                   "former owners their value."},
    "legacy_acquired_contracts_cancelled_on_exchequer": {
        "group": "Steps from P2 towards P3",
        "label": "Renewables and nuclear bought, contracts cancelled, owners compensated by the Exchequer",
        "explain": "As above, with the Exchequer paying the compensation."},
    "legacy_and_thermal_acquired_contracts_cancelled_on_bills": {
        "group": "Steps from P2 towards P3",
        "label": "Whole fleet bought, contracts cancelled, owners compensated through bills",
        "explain": "The whole existing fleet, with its subsidy contracts cancelled and bill-payers paying the "
                   "compensation."},
    "legacy_and_thermal_acquired_contracts_cancelled_on_exchequer": {
        "group": "Steps from P2 towards P3",
        "label": "Whole fleet bought, contracts cancelled, owners compensated by the Exchequer",
        "explain": "As above, with the Exchequer paying the compensation."},
}

# The assumptions best and worst move, in plain words. Keyed by (lever, value); a bound lever missing here
# fails the build, so the page never shows a lever id.
LEVER_WORDS = {
    ("auction_margin", "none"): "auction-calibrated CfD projects keep no margin above full cost, not 5%",
    ("auction_margin", "high"): "auction-calibrated CfD projects keep a 10% margin above full cost, not 5%",
    ("self_insurance", "none"): "state plant pays full commercial insurance premiums",
    ("self_insurance", "high"): "state plant saves 50% of its insurance premiums, not 35%",
    ("state_overhead", "low"): "the state company's head office costs £1 per kW of plant a year, not £3",
    ("state_overhead", "high"): "the state company's head office costs £8 per kW of plant a year, not £3",
    ("systematic_premium", "none"): "no price on the risk the state takes on (off the bill)",
    ("systematic_premium", "market_erp"): "the risk the state takes on priced at the market's equity premium (off the bill)",
    ("interconnector_price", "low"): "Britain's half of the interconnectors bought at a lower price",
    ("interconnector_price", "high"): "Britain's half of the interconnectors bought at a higher price",
    ("ldes_floor_basis", "technology_wacc"): "long-duration storage support priced at the technology's own cost of "
                                             "capital",
    ("private_wacc_basis", "fixed_tmr_shielded"): "private investors' cost of capital on the regulators' convention "
                                                  "(a fixed total market return), which lowers it",
    ("sovereign_yield", "B"): "government borrowing costs rise as state debt grows",
    ("public_rate_basis", "pwlb_standard"): "the state borrows at the PWLB standard rate, 100 basis points over gilts",
    ("purchase_price_basis", "state_present_value"): "the existing fleet bought at the state's own valuation, not "
                                                     "its owners' value",
    ("in_construction", "public_completion"): "the state finishes building Hinkley Point C itself",
    ("acquisition_transaction_costs", "two_pct"): "advisers' and transaction costs of 2% of the purchase price",
    ("acquisition_transaction_costs", "none"): "no advisers' or transaction costs on the purchase, not 2%",
    ("state_capex_overrun", "plus_15pct"): "the state's new build costs 15% more to build than a private developer's",
    ("state_om_premium", "plus_10pct"): "the state runs its plant at 10% higher operating cost than a private owner",
    ("legacy_asset_lives", "extended"): "existing plant expected to run longer, which raises its market value and so "
                                        "the price",
    ("capacity_procurement", "cost_based_existing"): "existing plant paid its cost less its own margins, not the "
                                                     "Capacity Market price",
    ("legacy_offshore_price", "transaction_multiple"): "existing offshore wind bought at recent deal prices per MW, "
                                                       "not the model's value of its cash flows",
}

# Plain names for the bill lines (the formal name and the source stay beside them).
LINE_PLAIN = {
    "capacity_market_design_adjustment": "Capacity Market under public ownership",
    "direct_energy": "Wholesale energy", "shaping_adjustment": "Matching supply to household demand",
    "procurement_losses": "Electricity lost in the wires", "procurement_transaction_cost": "Trading costs",
    "procurement_day_ahead_rehedging": "Day-ahead re-hedging", "procurement_imbalance": "Supplier imbalance charges",
    "procurement_additional_risk_and_basis": "Supplier risk allowance",
    "procurement_forward_premium": "Forward premium (zero in the central case)",
    "cfd": "Contracts for difference (renewable and nuclear contracts)",
    "allocated_ccs": "Bioenergy with carbon capture: its electricity contract",
    "bsuos_constraints": "Payments to manage grid bottlenecks", "renewables_obligation": "Renewables Obligation",
    "bsuos_other_services": "Other grid balancing services",
    "bsuos_voltage_and_stability_constraints": "Voltage and stability services",
    "supplier_costs_and_margin": "Supplier costs and margin", "tnuos": "Transmission network charges",
    "duos": "Distribution network charges", "capacity_market": "Capacity Market",
    "sizewell_c_rab": "Sizewell C construction charge", "bics_levy": "Network charging compensation levy",
    "nuclear_contract_counterparty": "Contract administration levies", "smart_meter_net_cost_change": "Smart meter cost change",
    "aahedc": "Support for network costs in the Highlands and islands", "feed_in_tariffs": "Feed-in Tariffs",
    "warm_home_discount": "Warm Home Discount", "eco": "Energy Company Obligation (ends April 2026)",
    "remaining_ccs": "Gas power with carbon capture (Net Zero Teesside)",
    "interconnector_cap_and_floor_pass_through": "Interconnector revenue sharing",
    "ldes_cap_and_floor": "Long-duration storage support", "public_surplus_rebate": "State surplus returned to bills",
    "public_cfd_at_cost": "State plant sold at cost: its earnings returned to bills",
    "public_state_deficit_levy": "Levy to cover the state's own losses",
    "capacity_market_public_financing_adjustment": "Capacity Market: the state's cheaper borrowing",
    "sizewell_c_rab_public_rate_adjustment": "Sizewell C construction charge: its return at the public rate",
    "tnuos_public_financing_adjustment": "Transmission charges: cheaper network borrowing",
    "duos_public_financing_adjustment": "Distribution charges: cheaper network borrowing",
    "nationalised_legacy_acquisition_renewables_nuclear": "Repayments: existing renewables and nuclear",
    "nationalised_legacy_acquisition_thermal_storage": "Repayments: existing gas, biomass and storage",
    "nationalised_new_build_capital": "New plant: capital at the public rate",
    "nationalised_fixed_om": "Plant running costs (fixed)", "nationalised_generator_tnuos": "Generators' transmission charges",
    "nationalised_insurance": "Plant insurance", "nationalised_variable_om": "Plant running costs (per unit generated)",
    "nationalised_fuel_carbon": "Fuel and carbon", "nationalised_state_operating_overhead": "State company head office",
    "nationalised_imports": "Imports", "nationalised_unacquired_paid_as_today": "Plant not bought, paid as today",
    "nationalised_realtime_correction": "Real-time correction of the schedule",
    "nationalised_interconnector_retrading": "Interconnector re-trading paid abroad",
    "nationalised_cancelled_contract_residual": "Compensation for cancelled contracts",
    "nationalised_beccs_carbon_removal_cfd": "Bioenergy with carbon capture: carbon removal (paid by the Exchequer)",
    "nationalised_tax_equivalent": "Tax equivalents paid by state plant",
    "nationalised_dpa_receipts_on_owned_plant": "Net Zero Teesside contract income returned (state owns the plant)",
    "nationalised_interconnector_foreign_half_rent": "Interconnector income to foreign owners",
    "nationalised_interconnector_gb_half_acquisition": "Repayments: Britain's half of the interconnectors",
    "nationalised_capacity_market_not_acquired": "Capacity Market contracts with plant not bought",
    "nationalised_acquisition_transaction_costs": "Purchase transaction costs",
    "nationalised_fit_domestic_to_taxation": "Feed-in Tariffs moved to taxation",
    "nationalised_whd_domestic_to_taxation": "Warm Home Discount moved to taxation",
    "nationalised_treasury_margin": "Margin kept by the Treasury (market prices charged)",
    "beccs_carbon_removal_levy": "Carbon removal at bioenergy plant, levied on bills",
    "legacy_levy_fit_to_taxation": "Feed-in Tariffs: households' share moved to taxation",
    "legacy_levy_ro_to_taxation": "Renewables Obligation: households' share moved to taxation",
    "legacy_levy_whd_to_taxation": "Warm Home Discount: households' share moved to taxation",
    "new_build_contracts_at_cost": "New merchant plant contracted at cost",
    "pay_as_bid_ceiling": "Wholesale paid as bid (upper bound)",
    "battery_revenue_top_up": "Battery revenue top-up",
    "battery_revenue_top_up_public_financing_adjustment": "Battery revenue top-up: the state's cheaper borrowing",
    "capacity_market_state_bids_at_cost": "Capacity Market: state plant bidding at its own cost",
    "capacity_market_cost_based_existing": "Capacity Market: existing plant on cost-based contracts",
    "nationalised_ancillary_services_income": "Balancing-service income, now received by the state",
    "nationalised_battery_subhourly_income": "Batteries' within-the-hour trading income, now received by the state",
    "nationalised_efw_gate_fee_income": "Energy-from-waste gate fees, now received by the state",
    "public_cost_recovery_contracting": "State surplus returned on the wholesale line",
    "public_surplus_offset_domestic_levies": "State surplus offsetting households' levies",
    "remaining_ccs_public_financing_adjustment": "Gas power with carbon capture (Net Zero Teesside): its contract at the state's cost",
    # 2050 lines (FES 2050 support regime)
    "allocated_ccs_public_financing_adjustment": "Bioenergy with carbon capture: the state's cheaper borrowing",
    "dpa": "Dispatchable power agreements (gas with carbon capture and hydrogen power)",
    "life_extension_operating_top_up": "Top-up for plant kept running past its contract",
}
# Plain labels for the party flows, by flow id (a tariff or institution flow id is matched on its prefix).
FLOW_PLAIN = {
    "cm_design_households": "Capacity Market payments to households' car batteries and demand response",
    "cm_design_no_rent": "Capacity Market payments to small-scale plant outside the ownership plan",
    "cm_design_outside_register": "Capacity Market payments to business demand response, interconnectors and other plant",
    "cm_design_register_private": "Capacity Market payments to private plant and co-owners, at the lower clearing price",
    "cm_design_state": "The state's plant leaves the Capacity Market auction",
    "socialised_systematic_risk": "The market risk taxpayers take on with the state's plant, at a social price",
    "strike_no_rent_private": "Contract prices on small-scale plant outside the ownership plan",
    "strike_rate_path_private": "Contract prices struck on the cheaper private rates: private owners",
    "strike_rate_path_state": "Contract prices struck on the cheaper private rates: the state's plant",
    "state_finance": "Cheaper borrowing on the state's share of new plant",
    "coowner_finance": "Cheaper borrowing on the private co-owners' share of joint projects",
    "state_operating": "Running the state company, and any extra running costs of state plant",
    "corporation_tax_forgone": "Corporation tax the state's plant does not pay (a transfer within the public sector; Great British Energy pays corporation tax as a company today)",
    "takeover_of_plant_rent": "The state takes over the private owners' rent above their required return",
    "takeover_relief": "The state takes on plant earning less than private owners would need",
    "battery_top_up_state": "Battery top-up at the state's own cost",
    "battery_top_up_coowners": "Battery top-up at a joint venture's own cost: the co-owners' share",
    "domestic_vat": "VAT on household bills",
    "capacity_market_public_financing_adjustment": "The state's cheaper borrowing passed through the Capacity Market",
    "carbon_removal_cfd": "Carbon-removal payments to private owners of bioenergy with carbon capture",
    "coowner_tax_equivalent": "Tax equivalents on the co-owners' share",
    "exchequer_to_investors:acquisition annuity": "Repayments on the purchase paid from taxation, to the sellers and lenders",
    "exchequer_to_investors:corporation tax the former owners": ("Corporation tax the former owners no longer pay on "
                                                                 "their 2030 margin, after allowances and interest"),
    "institution:cross-border": "Central dispatch: cross-border trade",
    "institution:fuel": "Central dispatch: fuel",
    "institution:interconnector foreign": "Central dispatch: interconnector income to foreign owners",
    "institution:real-time correction": "Central dispatch: allowance for forecast errors",
    "institution:variable O&M": "Central dispatch: running costs",
    "institution_carbon": "Central dispatch: carbon receipts",
    "ldes_floor_private": "Long-duration storage support paid to private stores",
    "ldes_floor_state": "Long-duration storage support paid to state stores",
    "network_financing_wedge": "Cheaper borrowing on the networks (the public rate against their allowed return)",
    "network_guarantee_compression": "Cheaper network borrowing under the state guarantee",
    "network_guarantee_expected_loss": "Expected losses the Exchequer bears on the guaranteed network debt",
    "network_purchase_premium": "Premium over regulated value paid to the sellers of the wires",
    "network_excess_return": "Return above the regulator's allowance that the wires' sellers no longer earn",
    "owners_value_premium": "Premium over cost paid to the former owners of pre-2025 contract plant (the state's share)",
    "supplier_costs_and_margin": "Suppliers' profit and headroom, which follow each case's own costs",
    "network_tax_allowance": "Corporation-tax allowance a public owner of the wires does not need",
    "public_cfd_at_cost": "State plant sold at cost: its earnings returned to bills",
    "public_state_deficit_levy": "Levy on bills to cover the state's own losses",
    "public_surplus_rebate": "The state's surplus returned to bills",
    "strike_cut_coowners": "Contract price cut borne by the private co-owners",
    "strike_cut_state": "Contract price cut on the state's plant",
    "tariff:Capacity Market": "Capacity Market payments to plant the state now owns, no longer made",
    "tariff:LDES floor": "Long-duration storage support no longer paid to the stores' owners",
    "tariff:Power BECCS capture": "Carbon capture at bioenergy plant, paid by the Exchequer's carbon contract",
    "tariff:RO line above": "Renewables Obligation charged beyond what the plant earns from its certificates",
    "tariff:carbon-removal CfD": "Carbon-removal payments the Exchequer no longer makes to private owners",
    "tariff:cross-border energy": ("Cross-border trade at foreign prices, interconnector re-trading and foreign "
                                   "owners' interconnector income, net"),
    "tariff:hedging and imbalance": "Suppliers' hedging and imbalance allowances, no longer paid",
    "tariff:EfW gate fees": "Energy-from-waste gate fees, now received by the state and credited to the tariff",
    "tariff:interconnector GB halves: acquisition": "Repayments on buying Britain's half of the interconnectors",
    "tariff:interconnector GB halves: cap-and-floor": "Interconnector revenue sharing, no longer paid",
    "tariff:interconnector GB halves: congestion": "Interconnector trading income the former owners give up",
    "tariff:interconnector re-trading": "Interconnector re-trading, no longer paid abroad",
    "tariff:legacy acquisition annuity": "Repayments on the purchase of the existing fleet, to the sellers and lenders",
    "tariff:legacy operating margin": "Operating profit the existing plant's owners give up",
    "tariff:new-build capital": "New plant financed at the public rate instead of private returns",
    "tariff:new-build producer rent": "Returns private owners of new plant no longer earn",
    "tariff:register payments": "Other payments to owners outside the replaced bill lines",
    "tariff:state operating overhead": "Running the state company",
    "tariff:tax equivalents": "Tax equivalents paid by state-owned new plant",
    "tariff:the state charges market prices": "Margin over cost kept by the Treasury (market prices charged)",
    "beccs_carbon_removal_levy": "Carbon-removal payments for bioenergy plant, levied on bills instead of paid by the Exchequer",
    "bid_stack:bsuos_constraints": "State plant bidding at cost: grid constraint payments",
    "bid_stack:cfd": "State plant bidding at cost: contract payments",
    "bid_stack:ldes_cap_and_floor": "State plant bidding at cost: long-duration storage support",
    "bid_stack:renewables_obligation": "State plant bidding at cost: Renewables Obligation",
    "bid_stack_carbon": "State plant bidding at cost: carbon receipts",
    "bid_stack_resource:fuel": "State plant bidding at cost: fuel, net of carbon",
    "bid_stack_resource:variable O&M": "State plant bidding at cost: running costs",
    "bid_stack_retrading": "State plant bidding at cost: interconnector re-trading paid abroad",
    "exchequer_to_investors:cancelled contracts": "Cancelled contracts: compensation to former owners, paid by the Exchequer",
    "legacy_levy_fit_to_taxation": "Households' share of Feed-in Tariffs moved to general taxation",
    "legacy_levy_ro_to_taxation": "Households' share of the Renewables Obligation moved to general taxation",
    "legacy_levy_whd_to_taxation": "Households' share of the Warm Home Discount moved to general taxation",
    "new_build_contracts_at_cost": "New merchant plant treated as contracted at cost",
    "pay_as_bid_ceiling": "Wholesale paid as bid rather than at the clearing price (an upper bound)",
    "tariff:Feed-in Tariffs: domestic share": "Households' share of Feed-in Tariffs moved to general taxation",
    "tariff:Warm Home Discount: domestic share": "Households' share of the Warm Home Discount moved to general taxation",
    "tariff:acquisition advisory": "Advisory and transaction costs of the purchase",
    "tariff:cancelled contracts: residual": "Cancelled contracts: compensation to former owners, on the bill",
    "bid_stack:battery_revenue_top_up": "State plant bidding at cost: battery revenue top-up",
    "bid_stack:supplier_costs_and_margin": "State plant bidding at cost: supplier costs and margin",
    "capacity_market_state_bids_private": "Capacity Market at the re-cleared price: private plant",
    "capacity_market_state_bids_state": "Capacity Market at the re-cleared price: the state's own plant",
    "capacity_market_cost_based_existing": "Capacity Market: existing plant paid its cost, not the clearing price",
    "coowner_operating": "Running-cost changes on the private co-owners' share",
    "network_guarantee_fee": "Guarantee fee the network companies pay the Exchequer",
    "public_cost_recovery_contracting": "The state's surplus returned on the wholesale line",
    "public_surplus_offset_domestic_levies": "The state's surplus offsetting households' levies",
    "sizewell_c_rab_public_rate_adjustment": "Cheaper finance on Sizewell C (the public rate instead of its regulated return)",
    "tariff:Sizewell C": "Cheaper finance on Sizewell C (the public rate instead of its regulated return)",
    "tariff:ancillary-service revenue": "Balancing-service income the former owners give up, now received by the state",
    "tariff:battery sub-hourly revenue": "Batteries' within-the-hour trading income the former owners give up",
    "tariff:battery revenue top-up": "Battery revenue top-up no longer paid to the owners of batteries the state buys",
    "allocated_ccs_public_financing_adjustment": "The state's cheaper borrowing on bioenergy with carbon capture",
    "tariff:life-extension top-up": "Top-up for plant kept past its contract, no longer paid to the owners the state bought out",
}
CATEGORY_PLAIN = {"financing": "cheaper borrowing", "tax": "tax", "VAT": "VAT", "other": "other",
                  "contract re-pricing": "contract prices", "guarantee": "network guarantee", "acquisition": "purchase",
                  "deficit": "state losses", "carbon": "carbon"}

# Plain-English fallback source for tariff lines that have no row in bill_line_sources.csv.
NATIONALISED_SOURCE = {"type": "modelled",
                       "label": "Modelled: the cost of the state-owned fleet (PyPSA-GB dispatch, the asset finance "
                                "register and the valuation of the existing fleet)", "url": ""}

# NESO's own 2030 constraint-cost figure for the same pathway (CP30 main report, Table 1, p. 36:
# Further Flex and Renewables, thermal constraints, 2024 prices). Cited as a constant.
NESO_CP30_CONSTRAINT_BN = 3.57
NESO_CP30_SOURCE = "NESO, Clean Power 2030 Advice, main report Table 1 (Further Flex and Renewables)"
# Network bounds (18-19 Oct 2026): each network case's market group is set against NESO's figure for
# that network (data/CP30/validation_references/neso_balancing_benchmarks.csv); a group not listed
# (the CP30 network's cp30_2030_cp30_network) keeps the CP30 main report's.
_NAO_SOURCE = ("NAO, Upgrading the electricity transmission network (Sep 2026), Figure 4 and para 2.8, quoting "
               "NESO's Clean Power 2030 analysis (upper end of the range)")
NESO_CONSTRAINT_BY_NETWORK = {
    "no_acceleration": (7.79, _NAO_SOURCE + ": Norwich-Tilbury and Sea Link on their published 2031-33 dates"),
    "no_upgrades": (12.70, _NAO_SOURCE + ": no network built after 2025"),
}


def _read_config(path) -> dict:
    """``config_loader.read_config``, imported on first use: this script puts the repo root on sys.path itself."""
    # 9 Nov 2026: main() reads the config before the later sys.path insertion, so the import failed when the
    # script is run as `python scripts/...` (the rule's way); the config's own repository goes first here.
    repo = str(_repo_of(path))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from config.config_loader import read_config

    return read_config(path)


def _repo_of(config_path) -> Path:
    """The repository a ``--config`` belongs to: the config directory's parent (``<repo>/config``), else as before
    the file's grandparent."""
    path = Path(config_path).resolve()
    # 10 Nov 2026: a resolved study file anywhere in the repository (the 2050 page reads
    # resources/public_ownership/FES25_HT_2050_finance_config.yaml): the nearest parent holding config/.
    for parent in ([path] if path.is_dir() else []) + list(path.parents):
        if (parent / "config" / "config_loader.py").exists():
            return parent
    return path.parent if path.is_dir() else path.parents[1]


def neso_constraint_comparator(group: str) -> tuple[float, str]:
    """NESO's constraint cost (GBP bn, thermal, 2024 prices) for the network of a market group."""
    for network, value in NESO_CONSTRAINT_BY_NETWORK.items():
        if network in group:
            return value
    return NESO_CP30_CONSTRAINT_BN, NESO_CP30_SOURCE


def three_parts_fields(parts: dict) -> dict:
    """F19 P3 decision 1: the explorer's view of the tariff's ``three_parts`` (empty without it)."""
    if not parts:
        return {}
    legacy = parts.get("legacy_purchase") or {}
    return {
        "three_parts": {k: num(v) for k, v in (parts.get("gbp_per_3100kwh_inc_vat") or {}).items()},
        "three_parts_labels": parts.get("labels") or {},
        "legacy_purchase_path": [{"year": int(y), "delta_inc_vat": num(v)}
                                 for y, v in (legacy.get("path_gbp_inc_vat") or {}).items()],
        "legacy_purchase_pv_bn": num((legacy.get("pv_public_rate_system_gbp") or 0.0) / 1e9, 2),
        "legacy_purchase_level_inc_vat": num(legacy.get("level_equivalent_gbp_inc_vat")),
    }


def num(x, nd=2):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


# User decision (24 Oct 2026): P3's sensitivities are computed with the market machinery kept (I1); the
# ledger restates every case owning the same fleet on central dispatch as its I1 figure plus the measured
# step (``i3_approx_*`` in the household totals). The page shows them as a view. User, 11 Nov 2026: the view is
# the page's default for Scenario 3 and carries no approximation label; the method stays in the note.
P3_I3_VIEW = {
    "label": "Central dispatch",
    "alt_label": "Market machinery kept",
    "note": ("Only the headline case is computed with a public operator dispatching the system. Every other case "
             "here is its market-machinery figure plus the central-dispatch step measured on the headline case: fuel "
             "and carbon, running costs, cross-border trade and the allowance for forecast errors. The step does not "
             "depend on the purchase price, its term or how the old plant is valued, so it is added unchanged. What "
             "drives the step differs by year: in 2030 it is mostly the premium the market's balancing mechanism "
             "pays to buy back interconnector flows, which central dispatch avoids; where the market barely "
             "re-trades the interconnectors, it is mostly the allowance for forecast errors, a benchmark cost taken from "
             "the 2026 REMA scheduling and dispatch assessment for NESO, because the model dispatches with perfect foresight."),
    "market_bill_note": ("Where the state charges households the private market bill, the bill does not move: the "
                         "step goes to the Treasury's margin instead."),
}
I3_PREFIX = "i3_approx_"
#: User decision (10 Nov 2026): a case of the law-in-2025 run group, beside the 2028 default.
LAW_2025_SUFFIX = "#legislated_2025"


def all_bills_bn(change, by_id, comp: str) -> float | None:
    """What every bill-payer saves, GBP bn a year inc VAT (the ledger's consumer column: each moved line at its own
    recovery volume, plus VAT), against the comparator ``comp`` as the difference of two changes against private;
    a saving is above zero. None without the column."""
    if change is None or pd.isna(change):
        return None
    base = 0.0
    if comp != "private" and comp in by_id.index:
        base = by_id.loc[comp].get("consumer_change_vs_private_gbp")
        if base is None or pd.isna(base):
            return None
    return num((float(change) - float(base)) / 1e9, 2)


# ---- topics (user, 11 Nov 2026): every assumption on the page under one question, the same list in every scenario ----
#: (id, question, one-line help, what to say where a scenario has nothing under it {code: text}).
TOPICS = [
    ("scope", "When, and how much, does the state own?",
     "When the law takes effect and how much of the new plant, contracts and storage the state takes on.", {}),
    ("borrowing", "What does it cost to borrow?",
     "The state's borrowing rate, the price of the risk taxpayers carry, and private investors' cost of capital.", {}),
    ("delivery", "How well does the state build and run its plant?",
     "Construction overruns, running and insurance costs, and the cost of running the state company.", {}),
    ("contracts", "How are contracts and prices set?",
     "How contract prices are set and repaid, auction margins, the Capacity Market and balancing, and storage "
     "support.", {}),
    ("purchase", "What does buying the existing fleet cost?",
     "The price the state pays for the plant that exists today, how long it takes to repay, and what it buys.",
     {"S1": "Scenario 1 buys no existing plant.", "S2": "Scenario 2 buys no existing plant."}),
    ("wires", "Who owns the wires?",
     "Whether the state also buys the transmission and distribution networks, and at what price.", {}),
    ("who_pays", "Who pays: bills or taxes?",
     "How the state's surpluses and costs are shared between bill-payers and taxpayers; the settlements of who "
     "pays below set them together.", {}),
]
TOPIC_IDS = [t[0] for t in TOPICS]
#: Levers that settle who pays (the incidence levers) and the levers each question holds.
LEVER_TOPIC = {
    "legislated_year": "scope", "state_deployment_share": "scope", "pre_2025_cfd_capacity": "scope",
    "new_build_generation": "scope", "new_build_storage": "scope", "state_stake_scale": "scope",
    "new_build_delivered_share": "scope",
    "public_rate_basis": "borrowing", "systematic_premium": "borrowing", "sovereign_yield": "borrowing",
    "private_wacc_basis": "borrowing", "private_rate_path": "borrowing",
    "state_overhead": "delivery", "self_insurance": "delivery", "state_capex_overrun": "delivery",
    "state_om_premium": "delivery",
    "cfd_strike_basis": "contracts", "auction_margin": "contracts", "cfd_amortisation": "contracts",
    "new_build_contracts": "contracts", "capacity_procurement": "contracts", "capacity_market_design": "contracts",
    "balancing_bids": "contracts", "dispatch_institution": "contracts", "ldes_floor_basis": "contracts",
    "purchase_price_basis": "purchase", "acquisition_term_years": "purchase", "legacy_price_path": "purchase",
    "legacy_offshore_price": "purchase", "efw_acquisition": "purchase", "in_construction": "purchase",
    "thermal_basis": "purchase", "legacy_contracts_under_acquisition": "purchase", "legacy_fleet": "purchase",
    "generation_control_premium": "purchase", "interconnector_price": "purchase",
    "acquisition_transaction_costs": "purchase", "legacy_asset_lives": "purchase",
    "networks": "wires", "network_acquisition_premium": "wires", "network_incidence": "wires",
    "surplus_recycling": "who_pays", "shortfall_funding": "who_pays", "state_deficit_funding": "who_pays",
    "tax_equivalent": "who_pays", "acquisition_financing": "who_pays", "tariff_pricing": "who_pays",
    "legacy_levy_funding": "background", "beccs_removal_funding": "background",
    "route_booking": "checks",
}
#: Switch and best-to-worst axes by token (they carry no lever on the page).
TOKEN_TOPIC = {
    "stake-15": "scope", "delivered-half": "scope", "pre-2025-repriced": "scope",
    "rate-pwlb-standard": "borrowing", "premium": "borrowing", "wacc": "borrowing",
    "overrun-15": "delivery", "om-plus-10": "delivery", "overhead-high": "delivery", "overhead": "delivery",
    "strike": "contracts", "ldes-wacc": "contracts",
    "control-premium-20": "purchase", "hpc-completion": "purchase", "price-state-value": "purchase",
    "txn-2pct": "purchase", "txn-none": "purchase", "lives-extended": "purchase", "ic": "purchase",
    "premium-erp": "borrowing",
    "networks": "wires", "rav-premium-high": "wires",
}
#: Policy choices among the switch tokens (the rest are uncertain assumptions, as every best-to-worst axis is).
CHOICE_TOKENS = {"networks", "pre-2025-repriced", "strike"}
#: Sensitivity groups that are model checks or earlier versions, shown apart (closed).
CHECK_GROUPS = ("Earlier, more cautious versions",)


def lever_tag(levers) -> str:
    """'uncertain' where every lever is an assumption on the scenario's own cost (levers.ASSUMPTION_LEVERS and the
    comparator assumptions a bound may move, plus the market assumptions below); otherwise 'choice'."""
    from scripts.public_ownership import levers as lv

    uncertain = set(lv.ASSUMPTION_LEVERS) | set(lv.BOUND_COMPARATOR_LEVERS) | {
        "private_rate_path", "auction_margin", "legacy_price_path", "legacy_offshore_price", "sovereign_yield"}
    return "uncertain" if levers and all(l in uncertain for l in levers) else "choice"


def sens_topic(row) -> str:
    """The question a one-at-a-time row sits under: its first lever with a topic (checks and background apart)."""
    if row.get("group") in CHECK_GROUPS:
        return "checks"
    found = [LEVER_TOPIC[l] for l in row.get("levers") or [] if l in LEVER_TOPIC]
    for special in ("background", "checks"):
        if special in found:
            return special
    # A step from Scenario 2 towards Scenario 3 changes what is bought: the purchase question.
    if "purchase" in found:
        return "purchase"
    return found[0] if found else "contracts"


def annotate_topics(scenario: dict) -> dict:
    """Give every switch, one-at-a-time row and best-to-worst axis its question and its tag."""
    for ax in scenario.get("axes") or []:
        ax["topic"] = TOKEN_TOPIC.get(ax["token"], "contracts")
        ax["tag"] = "choice" if ax["token"] in CHOICE_TOKENS else "uncertain"
    for r in scenario.get("sensitivities") or []:
        r["topic"] = sens_topic(r)
        r["tag"] = lever_tag(r.get("levers"))
    for ax in ((scenario.get("bounds") or {}).get("axes") or []):
        ax["topic"] = TOKEN_TOPIC.get(ax["token"], "contracts")
        ax["tag"] = "uncertain"
    return scenario


#: Meeting notes (user, 10 Nov 2026), renumbered 11 Nov 2026 (user): Scenario 0 is private; P1, P2, P3 in any
#: sentence the page shows become Scenarios 1, 2, 3 (the short codes on the page are S1, S2, S3; case ids are
#: unchanged). The financing workbook's own scenarios are written Sc1-Sc4 (its Sc1 is private), so the two
#: numberings are never confused.
SCENARIO_NUMBERS = {"1": "1", "2": "2", "3": "3"}
_P_CODE = re.compile(r"\bP([123])\b")


def scenario_names(obj):
    """Every string in the page's data with P1/P2/P3 written as Scenario 1/2/3 (keys and case ids untouched)."""
    if isinstance(obj, str):
        return _P_CODE.sub(lambda m: f"Scenario {SCENARIO_NUMBERS[m.group(1)]}", obj)
    if isinstance(obj, list):
        return [scenario_names(v) for v in obj]
    if isinstance(obj, dict):
        return {k: scenario_names(v) for k, v in obj.items()}
    return obj


def i3_entry(r, per_hh: float, by_id=None, comp: str = "private") -> dict | None:
    """A case's figures on central dispatch (the ledger's ``i3_approx_*`` columns), in the page's per-case shape."""
    bill = r.get(I3_PREFIX + "bill_inc_vat") if hasattr(r, "get") else None
    if bill is None or (isinstance(bill, float) and math.isnan(bill)):
        return None
    db, dh = r.get(I3_PREFIX + "bill_delta_vs_comparator_gbp_inc_vat"), r.get(
        I3_PREFIX + "household_total_delta_vs_comparator_gbp_inc_vat")
    return {
        "b": num(bill), "db": num(db), "dbp": num(r.get(I3_PREFIX + "bill_delta_vs_private_inc_vat")),
        "x": num(float(dh) - float(db)) if pd.notna(dh) and pd.notna(db) else None,
        "h": num(r.get(I3_PREFIX + "household_total_gbp_inc_vat")), "dh": num(dh),
        "dhl": num(r.get(I3_PREFIX + "household_total_legal_incidence_delta_vs_private_gbp_inc_vat")),
        "p": [num(float(r[I3_PREFIX + col]) * per_hh) for col in (
            "consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp",
            "investor_change_vs_private_gbp", "resource_saving_vs_private_gbp")],
        "rm": num(float(r[I3_PREFIX + "resource_saving_vs_private_gbp"]) / 1e6, 0),
        "cbn": all_bills_bn(r.get(I3_PREFIX + "consumer_change_vs_private_gbp"), by_id, comp),
        "step": num(r.get(I3_PREFIX + "step_bill_gbp_inc_vat")),
        "mb": bool("market_bill" in str(r.get(I3_PREFIX + "basis") or "")),
    }


def p3_i3_view(ledger: dict) -> dict | None:
    """The page's words and the ledger's step for P3's central-dispatch view, or None when the run has none."""
    block = ledger.get("institution_step_by_case") or {}
    if not block.get("applied"):
        return None
    basis = str(block.get("label") or "").replace("central dispatch, additive approximation: ", "")
    return {**P3_I3_VIEW, "basis": basis, "institution_case": block.get("institution_case_id"),
            "twin": block.get("twin_case_id"), "cases": len(block["applied"]),
            "step_range": [num(v) for v in (block.get("step_bill_gbp_inc_vat_range") or [])] or None,
            "excluded": sorted(block.get("excluded") or {})}


def flow_label(flow_id: str, fallback: str) -> str:
    if flow_id in FLOW_PLAIN:
        return FLOW_PLAIN[flow_id]
    hits = [k for k in FLOW_PLAIN if flow_id.startswith(k)]
    return FLOW_PLAIN[max(hits, key=len)] if hits else fallback


# ---------------------------------------------------------------------------------------------------
# Grids: axes, positions and the position -> case map.
# ---------------------------------------------------------------------------------------------------
def axes_from_schema(schema: dict) -> list[dict]:
    """``toggle_levers`` ({token: {selector, options, base} | {lever: value[, when]}}) as ordered axes."""
    axes = []
    for token, entry in schema.items():
        if "selector" in entry:
            axes.append({"token": token, "kind": "selector", "lever": entry["selector"],
                         "options": list(entry["options"]), "values": dict(entry["options"]), "base": entry["base"]})
        else:
            when = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in (entry.get("when") or {}).items()}
            (lever, value), = [(k, v) for k, v in entry.items() if k != "when"]
            axes.append({"token": token, "kind": "switch", "lever": lever, "value": value, "when": when})
    return axes


def live(axis: dict, vector: dict) -> bool:
    return all(vector.get(sel) in wanted for sel, wanted in axis.get("when", {}).items())


def all_positions(axes: list[dict]) -> list[dict]:
    """Every UI state of a grid, the base included, as position vectors."""
    out = [{}]
    for axis in axes:
        grown = []
        for v in out:
            if axis["kind"] == "selector":
                grown += [dict(v, **{axis["token"]: o}) for o in axis["options"]]
            elif live(axis, v):
                grown += [dict(v, **{axis["token"]: b}) for b in (0, 1)]
            else:
                grown.append(dict(v, **{axis["token"]: None}))
        out = grown
    return out


def position_key(axes: list[dict], vector: dict) -> str:
    """The page's key: one field per axis in declared order; a switch 0/1 or '-' where its condition fails."""
    parts = []
    for axis in axes:
        v = vector[axis["token"]]
        parts.append(v if axis["kind"] == "selector" else ("-" if v is None else str(int(v))))
    return "|".join(parts)


def normalise(axes: list[dict], vector: dict) -> dict | None:
    """A membership vector on these axes, missing axes at base (a reduced run's cut-down grids).

    Returns None when the vector names an axis this grid does not have, or an impossible value.
    """
    tokens = {a["token"] for a in axes}
    if set(vector) - tokens:
        return None
    out = {}
    for axis in axes:
        t = axis["token"]
        if axis["kind"] == "selector":
            v = vector.get(t, axis["base"])
            if v not in axis["options"]:
                return None
        else:
            v = vector.get(t, 0 if live(axis, out) else None)
            if live(axis, out):
                if v is None or int(v) not in (0, 1):
                    return None
                v = int(v)
            elif v not in (None, 0):
                return None
            else:
                v = None
        out[t] = v
    return out


def load_config_catalogue(config_path: Path):
    """Specs from the live configuration, through the repository's own expansion (read-only import)."""
    import yaml

    sys.dont_write_bytecode = True
    repo = str(_repo_of(config_path))  # <repo>/config
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts.public_ownership import reference_runs  # noqa: E402

    cfg = _read_config(config_path)["public_ownership"]
    specs = reference_runs.configured_case_specs(cfg)
    cat = {}
    for s in specs:
        meta = dict(s.meta or {})
        cat[s.case_id] = {"kind": meta.get("kind"), "parent": meta.get("parent"), "preset": meta.get("preset"),
                          "bound": meta.get("bound"), "grid_memberships": meta.get("grid_memberships") or [],
                          "toggle_levers": meta.get("toggle_levers"), "settings": dict(s.settings),
                          "grid": meta.get("grid"), "lever": meta.get("lever"), "value": meta.get("value")}
    return cat


def run_catalogue(hh: pd.DataFrame, ledger: dict) -> dict:
    """The run's own catalogue: the ledger's case_catalogue, with the household totals' membership column."""
    cat = {cid: dict(v) for cid, v in (ledger.get("case_catalogue") or {}).items()}
    for row in hh.itertuples():
        entry = cat.setdefault(row.case_id, {})
        entry.setdefault("kind", getattr(row, "case_kind", None))
        entry.setdefault("parent", getattr(row, "case_parent", None) if isinstance(getattr(row, "case_parent", None), str) else None)
        if isinstance(getattr(row, "case_preset", None), str):
            entry.setdefault("preset", row.case_preset)
        if isinstance(getattr(row, "case_bound", None), str):
            entry.setdefault("bound", row.case_bound)
        gm = getattr(row, "grid_memberships", None)
        if isinstance(gm, str) and gm.strip() and not entry.get("grid_memberships"):
            entry["grid_memberships"] = json.loads(gm)
    return cat


def grid_schema(cat: dict, grid_name: str):
    """The axes schema of the grid called ``grid_name`` (a parent id, or a parent id + '#bounds').

    A grid case carries its grid's name in ``grid``; an older catalogue without it is matched on the parent,
    which only ever named the toggle grid.
    """
    for v in cat.values():
        if v.get("kind") != "grid" or not v.get("toggle_levers"):
            continue
        name = v.get("grid") or v.get("parent")
        if name == grid_name:
            return v["toggle_levers"]
    return None


def membership_map(cat: dict, parent: str, axes: list[dict], report: list) -> dict:
    out, bad = {}, 0
    for cid, v in cat.items():
        for m in v.get("grid_memberships") or []:
            if m.get("grid") != parent:
                continue
            vec = normalise(axes, m.get("toggles") or {})
            if vec is None:
                bad += 1
                report.append(f"{parent}: {cid} carries a position on axes this grid does not have "
                              f"({m.get('toggles')}); ignored")
                continue
            key = position_key(axes, vec)
            if key in out and out[key] != cid:
                report.append(f"{parent}: position {key} is carried by both {out[key]} and {cid}")
            out.setdefault(key, cid)
    return out


# ---------------------------------------------------------------------------------------------------
# Generated one-at-a-time list, exact combinations, interaction error (OW7 item 10, OW13 item 1).
# ---------------------------------------------------------------------------------------------------
def pct_text(x: float, nd: int = 3) -> str:
    """0.0274509 -> '2.745%'; trailing zeros dropped."""
    s = f"{x * 100:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s + "%"


def gbp_text(x: float) -> str:
    return f"£{x:g}"


def variant_name(case_id: str) -> str:
    return case_id.split("@", 1)[1] if "@" in case_id else case_id


def scenario_of(cid: str, cat: dict, settings: dict | None) -> str | None:
    """The scenario a catalogue case is shown under: P3 if it buys the existing fleet, else its nearest
    scenario ancestor (P1 or P2). None for private cases (comparators)."""
    if settings and str(settings.get("legacy_fleet", "private")) != "private":
        return P3
    seen, cur = set(), cid
    while cur and cur not in seen:
        seen.add(cur)
        if cur in (P1, P2):
            return cur
        if cur == "private":
            return None
        cur = (cat.get(cur) or {}).get("parent")
    return None


def settings_diff(mine: dict, base: dict) -> dict:
    return {k: str(v) for k, v in mine.items() if str(base.get(k)) != str(v)}


def sens_words(cid: str, meta: dict, diff: dict, sid: str, consts: dict) -> dict | None:
    """The words for one listed case: by (lever, value) when it moves one lever, else by its name."""
    entry = None
    if meta.get("kind") == "one_at_a_time" and meta.get("lever"):
        entry = SENS_BY_LEVER.get((meta["lever"], str(meta["value"])))
    if entry is None and len(diff) == 1:
        entry = SENS_BY_LEVER.get(next(iter(diff.items())))
    if entry is None:
        entry = SENS_BY_NAME.get(variant_name(cid))
    if entry is None:
        return None
    out = {k: v for k, v in entry.items() if k != "by"}
    out.update((entry.get("by") or {}).get(sid, {}))
    return {k: (v.format_map(consts) if isinstance(v, str) else v) for k, v in out.items()}


def generate_sensitivities(sid: str, cat: dict, settings_of, all_cases_levers: set, consts: dict,
                           skip: set, report: list) -> list[dict]:
    """Every one-at-a-time and named case of ``sid``'s family, in group order, with the levers it moves."""
    base = settings_of(sid) or {}
    rows, unworded = [], []
    for cid, meta in cat.items():
        if meta.get("kind") not in ("one_at_a_time", "named") or cid == sid:
            continue
        st = settings_of(cid)
        if scenario_of(cid, cat, st) != sid:
            continue
        if any(m.get("grid") == sid for m in meta.get("grid_memberships") or []):
            continue  # reachable through the scenario's own switches
        if cid in skip:
            continue
        diff = settings_diff(st or {}, base)
        words = sens_words(cid, meta, diff, sid, consts)
        if words is None:
            unworded.append(f"{cid} (moves {diff})")
            continue
        moves_private = bool(set(diff) & all_cases_levers)
        explain = words["explain"]
        if moves_private and COMPARATOR_NOTE not in explain:
            explain = f"{explain} {COMPARATOR_NOTE}"
        rows.append({"group": words["group"], "case": cid, "label": words["label"], "explain": explain,
                     "levers": sorted(diff), "moves_private": moves_private, "_diff": diff})
    if unworded:
        raise SystemExit(f"{sid}: one-at-a-time cases with no words in SENS_BY_LEVER or SENS_BY_NAME: {unworded}")
    order = {g: i for i, g in enumerate(SENS_GROUPS)}
    unknown = sorted({r["group"] for r in rows} - set(order))
    if unknown:
        raise SystemExit(f"{sid}: sensitivity groups not in SENS_GROUPS: {unknown}")
    rows.sort(key=lambda r: order[r["group"]])  # stable: catalogue order within a group
    return rows


def combo_index(rows: list[dict], base: dict, all_settings: dict, has_numbers) -> dict:
    """{'i+j+...': case id} for every catalogue case whose settings are the base plus exactly the levers of two
    or more lever-disjoint rows. A case with numbers wins over one without."""
    items = [frozenset(r["_diff"].items()) for r in rows]
    levers_of = [frozenset(r["_diff"]) for r in rows]
    out: dict = {}
    for cid, st in all_settings.items():
        d = frozenset(settings_diff(st, base).items())
        if len(d) < 2:
            continue
        cands = [i for i, it in enumerate(items) if it and it <= d]
        if len(cands) < 2:
            continue
        found = []

        def search(start, covered, used, chosen):
            if covered == d:
                if len(chosen) >= 2:
                    found.append(tuple(chosen))
                return
            for j in range(start, len(cands)):
                i = cands[j]
                if levers_of[i] & used:
                    continue
                search(j + 1, covered | items[i], used | levers_of[i], chosen + [i])

        search(0, frozenset(), frozenset(), [])
        for sol in found:
            key = "+".join(str(i) for i in sorted(sol))
            if key not in out or (has_numbers(cid) and not has_numbers(out[key])):
                out[key] = cid
    return out


def interaction_error(axes: list[dict], gmap: dict, cases: dict, field: str = "db") -> dict | None:
    """How far the sum of single effects misses the exact answer, over every grid position that moves two or
    more independent axes (a conditional switch counts with the selector it depends on)."""
    if not axes:
        return None

    def value(moves):
        vec = normalise(axes, moves)
        if vec is None:
            return None
        c = cases.get(gmap.get(position_key(axes, vec)))
        return None if c is None else c.get(field)

    base = value({})
    if base is None:
        return None
    by_token = {a["token"]: a for a in axes}
    errors = []
    for pos in all_positions(axes):
        moved, groups = [], set()
        for a in axes:
            t, v = a["token"], pos[a["token"]]
            if a["kind"] == "selector" and v != a["base"]:
                moved.append(a)
                groups.add(t)
            elif a["kind"] == "switch" and v == 1:
                moved.append(a)
                groups.add(next(iter(a["when"])) if a.get("when") else t)
        if len(groups) < 2:
            continue
        exact = value({k: v for k, v in pos.items() if v is not None})
        if exact is None:
            continue
        total, ok = base, True
        for a in moved:
            t = a["token"]
            if a["kind"] == "selector":
                one = value({t: pos[t]})
                inc = None if one is None else one - base
            else:
                cond = {s: pos[s] for s in (a.get("when") or {})}
                on, off = value({**cond, t: 1}), (value(cond) if cond else base)
                inc = None if on is None or off is None else on - off
            if inc is None:
                ok = False
                break
            total += inc
        if ok:
            errors.append(abs(exact - total))
    if not errors:
        return None
    errors.sort()
    n = len(errors)
    median = errors[n // 2] if n % 2 else (errors[n // 2 - 1] + errors[n // 2]) / 2
    _ = by_token
    return {"n": n, "median": num(median), "max": num(errors[-1])}


def page_axis(ax: dict, words: dict) -> dict:
    entry = {"token": ax["token"], "kind": ax["kind"], "label": words["label"]}
    if "group" in words:
        entry["group"] = words["group"]
    if ax["kind"] == "selector":
        entry["base"] = ax["base"]
        entry["options"] = [{"token": o, **words["options"][o]} for o in ax["options"]]
    else:
        entry["explain"] = words["explain"]
        if ax.get("when"):
            entry["when"] = ax["when"]
            entry["requires"] = words.get("requires", "")
    if words.get("note"):
        entry["note"] = words["note"]
    return entry


def bound_tags(ax: dict, best: dict | None, worst: dict | None) -> None:
    """Mark the option (or the switch) the best and the worst case take, from their own settings."""
    for kind, st in (("best", best), ("worst", worst)):
        if not st:
            continue
        val = str(st.get(ax["lever"]))
        if ax["kind"] == "selector":
            for o in ax["page"]["options"]:
                if str(ax["values"][o["token"]]) == val and o["token"] != ax["base"]:
                    o[kind] = True
        elif str(ax["value"]) == val:
            ax["page"][kind] = True


# ---------------------------------------------------------------------------------------------------
# Cost of capital and sources (OW7 item 9): read from the workbook snapshot, the case list and the config.
# ---------------------------------------------------------------------------------------------------
WACC_TECHS = [("Offshore wind", "Offshore wind"), ("Onshore wind", "Onshore wind"), ("Solar PV", "Solar PV"),
              ("Batteries", "Lithium batteries"), ("Long-duration storage", "Novel long duration energy storage (LDES)"),
              ("Nuclear", "Large-scale nuclear"), ("Gas power stations", "Gas generation - unabated"),
              ("Biomass", "Biomass - unabated"), ("Biomass with carbon capture", "Biomass - with CCUS - mature"),
              ("Hydro", "Hydropower")]
# Workbook alternatives and the levers that select them. A value with no case is listed as "not run".
WACC_ALTERNATIVES = [
    ("private_wacc_basis", "Private investors' cost of capital", {
        "notional_gearing_shielded": "Fixed equity premium (5.22%), a notional 55% borrowing for every technology, "
                                     "with tax relief on interest (the base; WACC Scenarios columns D and F)",
        "fixed_tmr_shielded": "Fixed total market return (6.83%), the regulators' convention, co-equal in the workbook"}),
    ("public_return", "What state-financed new plant earns", {
        "public_rate": "The state's cash rate: the real gilt plus a lending margin, no construction premium (P2 and "
                       "P3; WACC Scenarios column H)",
        "cepa_decomposition": "The workbook's Sc2: the state's 30% of the equity at the risk-free rate and a "
                              "guarantee lifting the debt two notches (P1; column F)"}),
    ("public_rate_basis", "The state's lending margin", {
        "real_cpi": "{rate_central}: the National Loans Fund, 11 basis points over the real gilt (Sc4 Operative F11)",
        "pwlb_hra": "{rate_pwlb_hra}: PWLB housing revenue account, 40 basis points",
        "pwlb_certainty": "{rate_pwlb_certainty}: PWLB certainty rate, 80 basis points",
        "pwlb_standard": "{rate_pwlb_standard}: PWLB standard rate, 100 basis points",
        "gilt_flat": "{rate_gilt_flat}: the gilt with no margin"}),
    ("systematic_premium", "Price of the risk the state takes on (off the bill)", {
        "social_flat_19": "{premium_p} a year on every technology (the Green Book discount rate review; the "
                          "economic rate, columns I and J)",
        "none": "None (pure Arrow-Lind, today's Green Book)",
        "social_flat_10": "1% a year (the review's recommendation)",
        "social_beta_20": "2% per unit of the technology's risk (France)",
        "market_erp": "The market equity premium, 5.22% per unit of the technology's risk"}),
    ("sovereign_yield", "Gilt yields as state debt grows", {
        "A": "Unchanged (path A)",
        "B": "Rising with state debt and guarantees (path B: {yield_b_shift_p1} for P1, {yield_b_shift_p2} for P2, "
             "{yield_b_shift_p3} for P3 by 2030)"}),
    ("state_stake_scale", "The state's stake in P1", {
        "workbook_stake": "30% (Assumptions B22)",
        "funded_scale": "15%, what GB Energy and the National Wealth Fund could fund"}),
]


def cost_of_capital(snapshot_path: Path, workbook_path: Path | None, settings: pd.DataFrame | None,
                    cfg: dict | None, levers_mod, consts: dict, report: list) -> dict | None:
    import hashlib
    import yaml

    if not snapshot_path.exists():
        report.append(f"workbook snapshot {snapshot_path} not found; no cost-of-capital section")
        return None
    snap = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    techs = {t["technology"]: t for t in snap.get("technologies", []) if "wacc" in t}
    rows = []
    for plain, name in WACC_TECHS:
        t = techs.get(name)
        if t is None:
            report.append(f"cost of capital: workbook technology {name!r} not in the snapshot")
            continue
        w = t["wacc"]
        cells = t.get("wacc_source_cells", {})
        rows.append({"tech": plain, "workbook": name, "risk": t.get("risk_rating"),
                     "private": num(w["private"] * 100, 2), "p1": num(w["public_coordination"] * 100, 2),
                     "p2": num(w["public_ownership_new_build"] * 100, 3),
                     "p3": num(w["public_ownership_legacy"] * 100, 3),
                     "tmr": num(((t.get("wacc_by_basis") or {}).get("fixed_tmr_shielded") or {}).get("private", float("nan")) * 100, 2),
                     "cells": [cells.get(k, "") for k in ("private", "public_coordination", "public_ownership_new_build",
                                                          "public_ownership_legacy")]})
    # case counts per alternative: cases holding the value where the lever can bite
    alts = []
    for lever, title, values in WACC_ALTERNATIVES:
        lev = levers_mod.LEVERS.get(lever) if levers_mod else None
        opts = []
        for value, text in values.items():
            n = None
            if settings is not None and lever in settings.columns:
                mask = settings[lever].astype(str) == value
                if lev is not None and lev.scope != "all_cases":
                    mask &= settings["new_build_generation"].astype(str) != "private"
                if lever == "state_stake_scale":
                    mask &= settings["new_build_generation"].astype(str) == "state_stake"
                if lever in ("systematic_premium", "public_rate_basis"):
                    mask &= (settings["new_build_generation"].astype(str) == "public") & (
                        settings["public_return"].astype(str) == "public_rate")
                if lever == "public_return":
                    pass
                n = int(mask.sum())
            opts.append({"value": value, "text": text.format_map(consts), "cases": n})
        alts.append({"lever": lever, "title": title, "options": opts})
    switches = snap.get("switches", {})
    unused = [
        f"The workbook's other gearing bases (it uses {switches.get('gearing_basis', 'technology-specific')}): CEPA's "
        "uniform 50%, the RIIO-3 notional 55% and a custom uniform gearing.",
        "Its CEPA replication and pure-CAPM tabs, and the vanilla (no tax shield) rates: reported for comparison with "
        "CEPA and DESNZ only.",
        "Its Sc2 variants (flat 75 bp debt compression, flat equity haircut, no debt or equity benefit) and "
        "its 1- and 3-notch uplifts.",
        "Its fiscal tabs beyond the gilt path (additional debt interest, the yield-response sensitivity, the repricing "
        "of the gilt stock) and its own build-out: the model carries its own fleet.",
        "Its cost of the existing fleet at 2026 cohort level (Legacy Fleet): the model values its own 2030 rows on the "
        "workbook's method.",
    ]
    outside = []
    na = ((cfg or {}).get("network_account") or {})
    allowed = na.get("allowed_wacc_cpih_real") or {}
    srcs = na.get("allowed_wacc_sources") or {}
    for sector in ("transmission", "distribution"):
        if sector in allowed:
            outside.append({"what": f"The wires: {sector}", "rate": pct_text(float(allowed[sector]), 2) + " real (CPIH)",
                            "source": srcs.get(sector, "")})
    ldes = (((cfg or {}).get("consumer_bill") or {}).get("ldes_cap_and_floor") or {}).get("floor_return") or {}
    rbc = ldes.get("real_rate_by_case") or {}
    words = {"private": "a private project", "public_control": "P1", "public_ownership": "P2 and P3"}
    for k, v in rbc.items():
        outside.append({"what": f"Long-duration storage floor, {words.get(k, k)}",
                        "rate": "the state's own rate" if v is None else pct_text(float(v), 2) + " real",
                        "source": "A debt-like rate, because Ofgem's scheme floors at a return similar to the cost of "
                                  "debt (private: BBB-minus debt; P1: two notches better, the guarantee's uplift; the "
                                  "workbook's cost of debt by rating, Assumptions G120:G128)." if v is not None
                                  else "The case's own cost of capital."})
    sha = snap.get("workbook_sha256")
    check = None
    if workbook_path is not None and workbook_path.exists():
        h = hashlib.sha256(workbook_path.read_bytes()).hexdigest()
        check = h == sha
        if not check:
            report.append(f"workbook hash {h} does not match the snapshot's {sha}")
    return {"rows": rows, "alternatives": alts, "unused": unused, "outside": outside,
            "workbook": snap.get("source_workbook"), "sha256": sha, "hash_checked": check,
            "extracted_on": str(snap.get("extracted_on", "")), "rate_basis": snap.get("rate_basis"),
            "price_year": snap.get("price_year")}


def sens_constants(snapshot_path: Path, cfg: dict | None, premium: pd.DataFrame | None) -> dict:
    """Numbers the generated words quote, from the workbook snapshot, the config and the wires premium table."""
    import collections
    import yaml

    c = collections.defaultdict(lambda: "?")
    snap = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {}
    a = snap.get("assumptions", {})
    r = a.get("public_borrowing_real", {}).get("value")
    if r is not None:
        c["rate_central"] = pct_text(r)
    # F29 (26 Oct 2026): workbook v2.0's cash rate at each lending margin, and its social premium.
    for key, block in (snap.get("public_rate_by_lending_margin") or {}).items():
        c[f"rate_{key}"] = pct_text(float(block["value"]))
    if a.get("social_premium", {}).get("value") is not None:
        c["premium_p"] = pct_text(float(a["social_premium"]["value"]), 1)
    off = next((t for t in snap.get("technologies", []) if t.get("technology") == "Offshore wind" and "wacc" in t), None)
    if off:
        b = off.get("wacc_by_basis", {})
        c["off_obs"] = pct_text(off["wacc"]["private"], 2)
        c["off_tmr"] = pct_text(b.get("fixed_tmr_shielded", {}).get("private", float("nan")), 2)
    shifts = snap.get("sovereign_yield_shift_2030", {})
    for key, sc in (("yield_b_shift_p1", "Sc2"), ("yield_b_shift_p2", "Sc3"), ("yield_b_shift_p3", "Sc4")):
        try:
            c[key] = f"{float(shifts[sc]['B']['value']) * 100:.2f} points"
        except (KeyError, TypeError):
            pass
    c["yield_b_shift"] = c["yield_b_shift_p2"]
    oh = (cfg or {}).get("state_operating_overhead") or {}
    for k in ("central", "low", "high"):
        if f"{k}_gbp_per_kw_year" in oh:
            c[f"overhead_{k}"] = gbp_text(float(oh[f"{k}_gbp_per_kw_year"]))
    # F19 decision 6: unstaffed storage (batteries, LDES, pumped hydro) at its own rate.
    storage_oh = {"central_gbp_per_kw_year": 1.5, "low_gbp_per_kw_year": 0.5, "high_gbp_per_kw_year": 4.0,
                  **dict(oh.get("unstaffed_storage") or {})}
    for k in ("central", "low", "high"):
        c[f"overhead_storage_{k}"] = gbp_text(float(storage_oh[f"{k}_gbp_per_kw_year"]))
    ldes = ((((cfg or {}).get("consumer_bill") or {}).get("ldes_cap_and_floor") or {}).get("floor_return") or {})
    if (ldes.get("real_rate_by_case") or {}).get("private") is not None:
        c["ldes_private"] = pct_text(float(ldes["real_rate_by_case"]["private"]), 2)
    if premium is not None:
        p = premium.set_index("setting")["premium_over_rav"].astype(float)
        c["rav_central"] = pct_text(p.get("central", float("nan")), 0)
        c["rav_low"] = pct_text(p.get("low", float("nan")), 0)
        c["wires_quarter"] = (f"The networks bought at their regulated value plus {pct_text(p.get('quarter', float('nan')), 0)} "
                              f"instead of {c['rav_central']}: between what the listed market pays for a minority "
                              f"stake ({pct_text(p.get('low', float('nan')), 0)}) and recent full-control sales. Not "
                              "itself a deal precedent.")
        c["wires_statute"] = ("Compensation set by Parliament at the networks' regulated value, with no premium (the "
                              "Northern Rock precedent). Not a market value.")
    return c


def wires_note(premium: pd.DataFrame | None, nat: dict) -> str:
    """Where the wires price comes from, in plain words, from the premium table and the break-even premium."""
    import re

    if premium is None:
        return ""
    p = premium.set_index("setting")
    central = float(p.loc["central", "premium_over_rav"])
    basis = str(p.loc["central", "basis"]).split("Gas networks")[0]
    mults = [float(m) for m in re.findall(r"(\d\.\d+)x", basis)]
    text = (f"Buying the wires costs their regulated value plus a premium. The central {pct_text(central, 0)} is the "
            f"median of the three recent full-control sales of GB distribution networks")
    if mults:
        text += f" ({min(mults):.2f} to {max(mults):.2f} times regulated value)"
    text += "."
    alts = []
    if "quarter" in p.index:
        alts.append(f"{pct_text(float(p.loc['quarter', 'premium_over_rav']), 0)}")
    if "statutory_at_rav" in p.index:
        alts.append("no premium, if Parliament sets compensation at regulated value")
    if alts:
        text += " The other assumptions below show " + " and ".join(alts) + "."
    be = None
    for c in (nat.get("cases") or {}).values():
        b = ((c.get("network_account") or {}).get("break_even_premium") or {})
        lv = c.get("levers") or {}
        if b.get("break_even_premium_over_rav") is not None and str(lv.get("public_rate_basis", "real_cpi")) == "real_cpi":
            be = b
            break
    if be:
        whole = be.get("break_even_premium_over_rav_with_tax_allowance")
        wedge = float(be["break_even_premium_over_rav"])
        if whole is not None:
            text += (f" Buying them lowers costs only below a premium of about {pct_text(float(whole), 0)} "
                     f"({pct_text(wedge, 0)} if the corporation-tax allowance a public owner no longer needs is left "
                     "out of the saving).")
        else:
            text += f" Buying them lowers costs only below a premium of about {pct_text(wedge, 0)}."
    return text



# ---------------------------------------------------------------------------------------------------
#: User, 13 Nov 2026 (results review, decision 4): the cases shown beside Scenario 3's headline. Its result turns on
#: the purchase price and term and on the risk the state takes on (review findings 3 and 5).
S3_BASIS_CASES = (
    ("Offshore wind at UK deal prices", "public_ownership@fully_nationalised_market_value_offshore_deal_multiples"),
    ("Valued at today's prices", "public_ownership@fully_nationalised_market_value_price_path_today"),
    ("Repaid over 15 years", "public_ownership@fully_nationalised_market_value_term_15y"),
    ("With a 1% risk premium (HM Treasury)", "public_ownership@fully_nationalised_market_value_systematic_premium_1pc"),
    ("With a 1.9% risk premium", "public_ownership@fully_nationalised_market_value_systematic_premium"),
)
#: The pathway's cases set side by side for Scenario 3 (S2 at every new build state-owned, bidding at cost).
PATHWAY_S3 = "public_ownership@fully_nationalised_market_value"
PATHWAY_S2 = "public_ownership@deployment_share_100_state_bids"


def s3_pathway_view(po: Path) -> dict | None:
    """Scenario 3 and 2 over 2027-2050 (NPV at 3.5%, GBP bn): consumers, Exchequer, their sum, and after 2050.

    Read from the pathway summary when it exists (it is built after the year explorers on a fresh run; a page built
    before it carries None and says so).
    """
    path = Path(po) / "pathway_2030_2050_summary.json"
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    out = {}
    for key, cid in (("s3", PATHWAY_S3), ("s2", PATHWAY_S2)):
        c = (summary.get("cases") or {}).get(cid)
        if not c:
            continue
        npv = lambda parties, p: float(((parties.get(p) or {}).get("total") or {}).get("npv_stpr_3p5") or 0.0) / 1e9  # noqa: E731
        par, after = c.get("parties") or {}, (c.get("beyond_2050") or {}).get("parties") or {}
        out[key] = {"consumers_bn": num(npv(par, "consumers"), 1), "exchequer_bn": num(npv(par, "exchequer"), 1),
                    "public_bn": num(npv(par, "consumers") + npv(par, "exchequer"), 1),
                    "resource_bn": num(npv(par, "resource"), 1),
                    "after_2050_public_bn": num(npv(after, "consumers") + npv(after, "exchequer"), 1)}
    return out or None


def year_note(po: Path, group: str, model_year) -> dict | None:
    """Why Scenario 3 saves less than Scenario 2 in a year after the purchase (user, 11 Nov 2026), from the year's
    own outputs; None when it saves more (2030) or a file is missing.

    The purchase of 2030 is repaid in level instalments while most of what it bought has stopped earning, and the
    central-dispatch step carries a benchmark forecast-error charge (the model's own dispatch is cheaper).
    """
    try:
        if not model_year or int(model_year) <= 2030:
            return None
        ht = pd.read_csv(po / f"{group}_household_totals.csv", low_memory=False).set_index("case_id")
        delta = pd.to_numeric(ht["bill_delta_vs_private_inc_vat"], errors="coerce")
        s2, s3_i1, s3 = (float(delta.get(k)) for k in ("public_ownership", P3, f"{P3}_central_dispatch"))
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if not s3 > s2:
        return None
    gbp = lambda x: ("−" if x < 0 else "+") + f"£{abs(x):.2f}"
    paras = [f"In {int(model_year)} Scenario 3 saves less on the bill than Scenario 2 ({gbp(s3)} against {gbp(s2)} "
             "a year per household). Scenario 3 is Scenario 2 plus the purchase of the plant that exists before the "
             "law. The purchase is repaid in level annual instalments, each asset over its own remaining life, while "
             "most of what it bought earns little by then: its contracts have ended, much of it has retired and "
             "wholesale prices are lower. The price paid reflects mostly the fleet's earnings in the 2030s, so the "
             "purchase gives bill-payers its value early and the bill for it later; over 2027-2050 together "
             "Scenario 3 saves more (see the cumulative pathway page)."]
    try:
        carried = json.loads((po / f"{group}_carried_acquisition.json").read_text(encoding="utf-8"))
        annuity = float(carried["cases"][P3]["ledger"]["annuity_gbp"])
        paras.append(f"This year the instalments cost about £{annuity / 1e9:.1f} bn. Most of the gap is timing, "
                     "who pays when for the purchase, not a worse use of resources: the real saving to the economy "
                     "differs far less between the two scenarios than the bill does.")
    except (OSError, KeyError, TypeError, ValueError):
        pass
    try:
        inst = json.loads((po / f"{group}_nationalised_bill_by_institution.json").read_text(encoding="utf-8"))
        gross = float(inst["institution_change_gross_inc_vat_gbp"])
        charge = float(inst["forecast_error_charge_ex_vat_gbp"])
        ratio = float(inst["forecast_error_fleet_ratio"])
        step = s3 - s3_i1
        if step > 0 and gross < 0:
            paras.append(f"Central dispatch adds {gbp(step)} a year to Scenario 3's bill this year although the model's "
                         f"own dispatch under it is cheaper ({gbp(gross)} a household: less gas burnt, more wind "
                         "and solar used). The rise comes from a forecast-error cost the model cannot produce itself "
                         "(it dispatches with perfect foresight): the benchmark estimate for the 2030s from "
                         "the 2026 REMA scheduling and dispatch assessment for NESO, scaled to "
                         f"this year's wind and solar output ({ratio:.2f} times), adds about £{charge:.2f} a "
                         "household before VAT. That scaling is a borrowed assumption, and the result turns on it.")
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return {"title": f"Why Scenario 3 saves less than Scenario 2 in {int(model_year)}", "paragraphs": paras}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--po", default=str(PROJECT_ROOT / "resources" / "public_ownership"))
    ap.add_argument("--sources", default=str(PROJECT_ROOT / "data" / "market" / "consumer_bill_benchmarks" /
                                             "bill_line_sources.csv"))
    ap.add_argument("--bill-summary", default=str(PROJECT_ROOT / "resources" / "analysis" /
                                                  "CP30_2030_cp30_network_consumer_bill_future_summary.json"))
    ap.add_argument("--energy", default=str(PROJECT_ROOT / "resources" / "CP30" / "validation" /
                                            "CP30_FFR_2030_energy_totals.csv"))
    ap.add_argument("--dispatch-validation", default=str(PROJECT_ROOT / "resources" / "analysis" /
                                                         "CP30_2030_reinforced_cp30_energy_validation.csv"))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config"),
                    help="configuration whose case catalogue gives every grid position its case id")
    ap.add_argument("--no-config", action="store_true", help="use the run's own grid memberships only")
    ap.add_argument("--run-label", default="",
                    help="how the page names this run, e.g. 'bill-layer run of 30 September 2026'")
    ap.add_argument("--assumptions", default=None,
                    help="the workbook snapshot (workbook_v5_assumptions.yaml); default: the config's "
                         "public_ownership.assumptions_snapshot under the repository")
    ap.add_argument("--workbook", default=None, help="the workbook itself, to check the snapshot's hash; default: the "
                                                     "snapshot's source_workbook under the repository")
    ap.add_argument("--network-premium", default=None,
                    help="network_acquisition_premium.csv; default: the config's network_account table")
    ap.add_argument("--bound-detail", action="store_true",
                    help="also carry bill lines, flows and contract prices for every bound-grid position "
                         "(off by default: thousands of positions, figures only)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "resources" / "analysis" / "public_ownership_explorer_data.json"))
    # 10 Nov 2026: the page for any modelled year (the 2050 page reads fes_ht_2050); default the run's own year.
    ap.add_argument("--price-files", nargs="*", default=None,
                    help="the model year's hourly wholesale price files (column wholesale_price), for the today-vs-model "
                         "price tile (user, 10 Nov 2026)")
    ap.add_argument("--model-year", type=int, default=None,
                    help="the modelled year the page describes; default: the P3 tariff summary's model_year, else 2030")
    ap.add_argument("--pathway-short", default=None, help="the pathway as the page's eyebrow and footer name it")
    ap.add_argument("--pathway-long", default=None, help="the pathway as the page's 'About these figures' names it")
    ap.add_argument("--group", default="cp30_2030_cp30_network",
                    help="the run group whose outputs the page reads (network bounds, 18-19 Oct 2026: the CENTRAL "
                         "network's market group, the one carrying every case; public_ownership.report_group)")
    a = ap.parse_args()
    PO = Path(a.po)
    report: list[str] = []
    repo = _repo_of(a.config) if Path(a.config).exists() else PROJECT_ROOT
    cfg_po: dict = {}
    levers_mod = None
    if Path(a.config).exists():
        import yaml
        cfg_po = (_read_config(a.config) or {}).get("public_ownership") or {}
        try:
            sys.dont_write_bytecode = True
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from scripts.public_ownership import levers as levers_mod  # noqa: E402  (read-only)
        except Exception as error:
            report.append(f"levers.py not imported ({type(error).__name__}: {error})")
    snapshot_path = Path(a.assumptions) if a.assumptions else repo / str(
        cfg_po.get("assumptions_snapshot", "resources/public_ownership/workbook_v5_assumptions.yaml"))
    premium_path = Path(a.network_premium) if a.network_premium else repo / str(
        (cfg_po.get("network_account") or {}).get("acquisition_premium_table",
                                                  "data/public_ownership/financing/network_acquisition_premium.csv"))
    premium = pd.read_csv(premium_path) if premium_path.exists() else None
    if premium is None:
        report.append(f"wires premium table {premium_path} not found; the wires note is left out")
    all_cases_levers = set(levers_mod.all_cases_levers()) if levers_mod else set()

    hh = pd.read_csv(PO / f"{a.group}_household_totals.csv")
    flows = pd.read_csv(PO / f"{a.group}_party_flows.csv")
    market_lines = pd.read_csv(PO / f"{a.group}_consumer_bill_by_ownership_case.csv")
    nat_lines = pd.read_csv(PO / f"{a.group}_nationalised_bill_components.csv")
    strikes = pd.read_csv(PO / f"{a.group}_cfd_strike_adjustments.csv")
    nat = json.loads((PO / f"{a.group}_nationalised_bill_summary.json").read_text(encoding="utf-8"))
    model_year = int(a.model_year or next((c.get("model_year") for c in (nat.get("cases") or {}).values()
                                           if isinstance(c, dict) and c.get("model_year")), None) or 2030)
    pathway_short, pathway_long = PATHWAY_NAMES.get(model_year, PATHWAY_NAMES[None])
    pathway_short = a.pathway_short or pathway_short.format(year=model_year)
    pathway_long = a.pathway_long or pathway_long.format(year=model_year)
    ledger = json.loads((PO / f"{a.group}_exchequer_ledger.json").read_text(encoding="utf-8"))
    settings_path = PO / f"{a.group}_case_settings.csv"
    run_settings = pd.read_csv(settings_path, dtype=str).set_index("case_id") if settings_path.exists() else None
    sources = pd.read_csv(a.sources).fillna("")
    SRC = {r.component_id: {"type": r.source_type, "label": r.source_label, "url": r.source_url}
           for r in sources.itertuples()}

    by_id = hh.set_index("case_id")
    conv = ledger.get("conversion", {})
    # F19 (user decision, 6 Oct 2026): the institution step's two solved years, flagged when mixed.
    vintage = ledger.get("institution_solve_vintage") or {}
    vat_rate = float(conv.get("vat_rate", 0.05))
    # GBP (system, 2024 prices) -> GBP per household (2025 prices) on the ledger's electricity-use key: the
    # Exchequer's per-household key, metered supply (F18); the party columns are system amounts, so this is
    # their per-household reading on that key, not a bill line's conversion.
    key_mwh = conv.get("exchequer_key_mwh", conv.get("final_demand_mwh"))
    if {"factor", "mwh"} <= set(conv) and key_mwh:
        per_hh = float(conv["factor"]) * float(conv["mwh"]) / float(key_mwh)
        key_note = ledger.get("household_total", {}).get("definition", "")
    else:  # fall back to the flow table's own conversion
        r = (flows["gbp_m"] * 1e6 / flows["gbp_per_household"]).replace([float("inf")], float("nan")).dropna()
        per_hh = 1.0 / float(r.median())
        key_note = "from the party-flow table's own conversion"
    households_equiv_m = 1.0 / per_hh / 1e6

    # ---- the catalogue --------------------------------------------------------------------------
    rcat = run_catalogue(hh, ledger)
    ccat = None
    if not a.no_config and Path(a.config).exists():
        try:
            ccat = load_config_catalogue(Path(a.config))
        except Exception as error:  # the page still builds from the run's own memberships
            report.append(f"configuration catalogue not loaded ({type(error).__name__}: {error}); using the run's "
                          "memberships only")
    cat = ccat or rcat
    unknown_run = sorted(set(by_id.index) - set(cat)) if ccat else []
    if unknown_run:
        # 10 Nov 2026: a study's own cases (FES 2050's deployment shares, rate path, CM design) are in the run's
        # catalogue only; the page lists them from it.
        cat = {**cat, **{cid: rcat[cid] for cid in unknown_run if cid in rcat}}
        report.append(f"{len(unknown_run)} run cases are not in the configuration's case list, taken from the run's "
                      f"own catalogue: {unknown_run[:8]}")
    # Bound-grid positions carry their figures only (thousands of them), unless --bound-detail.
    bound_only = set() if a.bound_detail else {
        cid for cid, v in cat.items()
        if v.get("kind") == "grid" and v.get("grid_memberships")
        and all(str(m.get("grid", "")).endswith("#bounds") for m in v["grid_memberships"])}

    # ---- per-case numbers (compact) ---------------------------------------------------------------
    def comparator(cid):
        r = by_id.loc[cid]
        comp = r.comparator_case_id if isinstance(r.get("comparator_case_id"), str) else "private"
        return comp if comp in by_id.index else "private"

    cases = {}

    def case_entry(r, table, comp):
        db_raw = r.get("bill_delta_vs_comparator_gbp_inc_vat")
        dh_raw = r.get("household_total_delta_vs_comparator_gbp_inc_vat")
        entry = {
            "b": num(r.bill_inc_vat), "db": num(db_raw), "dbp": num(r.bill_delta_vs_private_inc_vat),
            # the Exchequer's change against the comparator, per household on the ledger's key (a cost > 0):
            # the household total's change less the bill's, i.e. the taxpayer share's change
            "x": num(float(dh_raw) - float(db_raw)) if pd.notna(dh_raw) and pd.notna(db_raw) else None,
            "h": num(r.household_total_gbp_inc_vat), "dh": num(dh_raw),
            "dhl": num(r.get("household_total_legal_incidence_delta_vs_private_gbp_inc_vat")),
            "p": [num(float(r[col]) * per_hh) for col in (
                "consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp",
                "investor_change_vs_private_gbp", "resource_saving_vs_private_gbp")],
            "rm": num(float(r.resource_saving_vs_private_gbp) / 1e6, 0),
            "cbn": all_bills_bn(r.get("consumer_change_vs_private_gbp"), table, comp),
        }
        if comp != "private":
            entry["c"] = comp
        # User, 13 Nov 2026: the electrified (heat pump + EV) household beside the 3,100 kWh benchmark, on its
        # own standing and unit split (scripts/public_ownership/electrified_household.py); against private.
        for key, prefix in (("e", "electrified"), ("eh", "electrified_high")):
            if pd.notna(r.get(f"{prefix}_bill_inc_vat")):
                entry[key] = {"b": num(r.get(f"{prefix}_bill_inc_vat")),
                              "db": num(r.get(f"{prefix}_bill_delta_vs_private_inc_vat")),
                              "dh": num(r.get(f"{prefix}_household_total_delta_vs_private_gbp_inc_vat")),
                              "dhl": num(r.get(f"{prefix}_household_total_legal_incidence_delta_vs_private_gbp_inc_vat"))}
        # 24 Oct 2026: a same-fleet P3 case's figures on central dispatch (additive approximation).
        i3 = i3_entry(r, per_hh, table, comp)
        if i3:
            for key, prefix in (("e", "electrified"), ("eh", "electrified_high")):
                if key not in entry:
                    continue
                # The central-dispatch step moves per-kWh lines only (wholesale, balancing, the forecast-error
                # allowance), and its taxpayer share is on the electricity-use key: both scale with use.
                k = float(r.get(f"{prefix}_kwh")) / 3100.0
                step_b = (i3["dbp"] - entry["dbp"]) if i3.get("dbp") is not None and entry.get("dbp") is not None else None
                step_h = (i3["dh"] - entry["dh"]) if i3.get("dh") is not None and entry.get("dh") is not None and comp == "private" else None
                e = entry[key]
                i3[key] = ({"b": num(e["b"] + step_b * k), "db": num(e["db"] + step_b * k),
                            "dh": num(e["dh"] + step_h * k) if step_h is not None and e.get("dh") is not None else None,
                            "dhl": None} if step_b is not None and e.get("b") is not None else None)
            entry["i3"] = i3
        return entry

    for cid in by_id.index:
        cases[cid] = case_entry(by_id.loc[cid], by_id, comparator(cid))
    # User decision (10 Nov 2026): the law-in-2025 run group's cases, keyed <case>#legislated_2025.
    law_path = PO / f"{a.group}_legislated_2025_household_totals.csv"
    if law_path.exists():
        law = pd.read_csv(law_path).set_index("case_id")
        for cid in law.index:
            r = law.loc[cid]
            comp = r.comparator_case_id if isinstance(r.get("comparator_case_id"), str) else "private"
            cases[f"{cid}{LAW_2025_SUFFIX}"] = case_entry(r, law, comp if comp in law.index else "private")
    else:
        report.append(f"{law_path.name} not found; the law-in-2025 rows are left out")

    # itemised flows for every case, labels held once
    flow_defs, flow_index, flows_by = [], {}, {}
    fl = flows.copy()
    fl["abs"] = fl.gbp_per_household.abs()
    for cid, g in fl.sort_values("abs", ascending=False).groupby("case_id", sort=False):
        if cid in bound_only:
            continue
        rows = []
        for f in g.itertuples():
            if abs(float(f.gbp_per_household)) < 0.005:
                continue
            key = (flow_label(str(f.flow_id), str(f.label)), f.payer, f.payee, CATEGORY_PLAIN.get(f.category, f.category))
            if key not in flow_index:
                flow_index[key] = len(flow_defs)
                flow_defs.append(list(key))
            rows.append([flow_index[key], num(f.gbp_per_household), num(f.gbp_m, 0)])
        flows_by[cid] = rows

    unworded = sorted({str(f) for f in flows.flow_id.unique()
                       if str(f) not in FLOW_PLAIN and not any(str(f).startswith(k) for k in FLOW_PLAIN)})
    if unworded:
        report.append(f"flows shown with the ledger's own label (add them to FLOW_PLAIN): {unworded}")

    # bill lines for every case, the line definitions held once. Each line carries two groups from
    # bill_groupings.csv (6 Oct 2026): what it pays for (the page's default, since the drawer compares a
    # case with private) and Ofgem's price-cap category at the current cap release.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.consumer_bill import bill_groupings  # noqa: E402  (read-only)
    line_meta = bill_groupings.annotate(
        pd.concat([market_lines, nat_lines]).drop_duplicates("component_id")).set_index("component_id")
    line_groupings = {view: {"title": bill_groupings.view_title(view), "order": bill_groupings.group_order(view)}
                      for view in bill_groupings.VIEWS}
    line_defs, line_index, lines_by = [], {}, {}
    tariff_ids = set(nat_lines.case_id)
    for table, ids in ((market_lines, set(market_lines.case_id) - tariff_ids), (nat_lines, tariff_ids)):
        sums = table[table.case_id.isin(ids)].groupby(["case_id", "component_id"])["annual_gbp_per_3100kwh_ex_vat"].sum()
        for (cid, comp), value in sums.items():
            if abs(value) < 0.005 or cid in bound_only:
                continue
            if comp not in line_index:
                src = SRC.get(comp) or (NATIONALISED_SOURCE if comp.startswith("nationalised_") else None) or {}
                line_index[comp] = len(line_defs)
                line_defs.append([comp, LINE_PLAIN.get(comp, str(line_meta.loc[comp, "component"])),
                                  str(line_meta.loc[comp, "functional_group"]), str(line_meta.loc[comp, "component"]),
                                  src.get("type", ""), src.get("label", ""), src.get("url", ""),
                                  str(line_meta.loc[comp, "ofgem_group"])])
            lines_by.setdefault(cid, []).append([line_index[comp], num(value)])

    unworded = sorted(d[0] for d in line_defs if d[0] not in LINE_PLAIN)
    if unworded:
        report.append(f"bill lines shown with their formal name (add them to LINE_PLAIN): {unworded}")

    # contract prices by technology (market cases)
    strikes_by = {}
    s_all = strikes[strikes.eligible_dispatch_mwh > 0]
    for cid, sc in s_all.groupby("case_id"):
        if cid in bound_only:
            continue
        rows = []
        for tech, g in sc.groupby("technology"):
            w = g.eligible_dispatch_mwh
            if w.sum() < 1e5:
                continue
            rows.append([tech, num((g.strike_private_gbp_per_mwh_2024 * w).sum() / w.sum(), 1),
                         num((g.strike_case_gbp_per_mwh_2024 * w).sum() / w.sum(), 1), num(w.sum() / 1e6, 1)])
        strikes_by[cid] = sorted(rows, key=lambda r: -r[3])

    unc_all = ledger.get("uncertainties_by_case", {})

    def uncertainties(cid):
        u = unc_all.get(cid, {})
        return [{"key": k, "label": v.get("label"), "line": v.get("line"),
                 "private": num(v.get("private_bill_change_gbp_inc_vat")),
                 "case": num(v.get("case_bill_change_gbp_inc_vat")),
                 "gap": num(v.get("gap_change_gbp_inc_vat"))} for k, v in u.items()]

    def settings_of(cid):
        if ccat and cid in ccat:
            # 10 Nov 2026: a study's gated levers (FES 2050) are in the run's settings only; the run's row fills
            # the levers the catalogue does not carry, so a catalogue case and a run-only case compare alike.
            if run_settings is not None and cid in run_settings.index:
                return {**run_settings.loc[cid].to_dict(), **ccat[cid]["settings"]}
            return ccat[cid]["settings"]
        if run_settings is not None and cid in run_settings.index:
            return run_settings.loc[cid].to_dict()
        return None

    def derived(sid, kind, name):
        """The case a preset or a bound makes on a scenario: the catalogue's, else the run's."""
        field = "preset" if kind == "preset" else "bound"
        for source in (ccat, rcat):
            for cid, v in (source or {}).items():
                if v.get("kind") == kind and v.get("parent") == sid and v.get(field) == name:
                    return cid
        return None

    def bound_words(sid, cid):
        """The levers a bound moves off its parent, in words; and the comparator's."""
        if cid is None:
            return None
        mine, base = settings_of(cid), settings_of(sid)
        if mine is None or base is None:
            return None
        moved = [(k, v) for k, v in mine.items() if str(base.get(k)) != str(v)]
        words, missing = [], []
        for k, v in moved:
            (words if (k, v) in LEVER_WORDS else missing).append(LEVER_WORDS.get((k, v), f"{k}={v}"))
        if missing:
            raise SystemExit(f"bound {cid} moves levers with no words in LEVER_WORDS: {missing}")
        # the comparator: the run's, else the configuration's private case at the same bound
        comp = cases.get(cid, {}).get("c")
        if comp is None and ccat:
            bound = ccat[cid].get("bound")
            comp = next((c for c, v in ccat.items() if v.get("kind") == "bound_comparator" and v.get("bound") == bound),
                        None)
        comp_words = []
        if comp and settings_of(comp) is not None:
            priv = settings_of("private") or {}
            comp_words = [LEVER_WORDS[(k, v)] for k, v in settings_of(comp).items()
                          if str(priv.get(k)) != str(v) and (k, v) in moved]
        # how much those assumptions change the private bill itself (the comparator against private)
        comp_change = (num(cases[comp]["b"] - cases["private"]["b"])
                       if comp and comp in cases and "private" in cases and comp_words else None)
        return {"moves": words, "comparator_moves": comp_words, "comparator_bill_change": comp_change}

    consts = sens_constants(snapshot_path, cfg_po, premium)
    wnote = wires_note(premium, nat)
    all_settings = ({cid: v["settings"] for cid, v in ccat.items()} if ccat else
                    ({cid: row.to_dict() for cid, row in run_settings.iterrows()} if run_settings is not None else {}))
    has_numbers = lambda cid: cid in cases  # noqa: E731

    def grid_axes_and_map(grid_name, words_for, where):
        """Axes (with page words) and the position -> case map of one exact grid, configuration first."""
        schema = grid_schema(ccat, grid_name) if ccat else None
        source = "configuration"
        if schema is None:
            schema, source = grid_schema(rcat, grid_name), "run"
        axes = axes_from_schema(schema) if schema else []
        undeclared = [ax["token"] for ax in axes if ax["token"] not in words_for]
        if undeclared:
            raise SystemExit(f"{grid_name}: grid axes {undeclared} have no words in {where}")
        for ax in axes:
            if ax["kind"] == "selector":
                missing = [o for o in ax["options"] if o not in words_for[ax["token"]].get("options", {})]
                if missing:
                    raise SystemExit(f"{grid_name}: selector {ax['token']} options {missing} have no words in {where}")
        gmap = membership_map(cat, grid_name, axes, report) if axes else {}
        if ccat and axes:  # cross-check against the run's own memberships
            rmap = membership_map(rcat, grid_name, axes, report)
            for key, cid in rmap.items():
                if gmap.get(key) not in (None, cid):
                    report.append(f"{grid_name}: position {key} is {gmap[key]} in the configuration and {cid} in the run")
        for ax in axes:
            ax["page"] = page_axis(ax, words_for[ax["token"]])
        return axes, gmap, source

    # ---- scenarios ------------------------------------------------------------------------------
    scenarios = []
    grid_checks = {}
    interaction = {}
    listed = set()
    for sid, meta in SCENARIOS.items():
        code = meta["code"]
        # the switch grid
        axes, gmap, source = grid_axes_and_map(sid, GRID_UI[sid], "GRID_UI")
        positions = all_positions(axes) if axes else [{}]
        keys = [position_key(axes, p) for p in positions] if axes else [""]
        if not axes:
            gmap = {"": sid}
        unmapped = [k for k in keys if k not in gmap]
        with_numbers = sum(1 for k in keys if gmap.get(k) in cases)
        grid_checks[code] = {"positions": len(keys), "mapped": len(keys) - len(unmapped), "unmapped": unmapped,
                             "with_numbers": with_numbers, "axes_from": source,
                             "distinct_cases": len({gmap[k] for k in keys if k in gmap})}
        for k in unmapped:
            report.append(f"{sid}: UI position {k} maps to no case id")
        page_axes = [ax["page"] for ax in axes]
        for entry in page_axes:  # where the wires are chosen, say where their price comes from
            if entry["token"] == "networks" and wnote:
                entry["note"] = wnote
        ierr = interaction_error(axes, gmap, cases) if axes else None
        interaction[code] = ierr

        presets = []
        for p in PRESETS:
            if p.get("only") and sid not in p["only"]:
                continue
            cid = sid if p["key"] == "as_designed" else derived(sid, "preset", p["key"])
            presets.append({"key": p["key"], "label": p["label"], "note": p["note_by"].get(sid, p["note"])
                            if "note_by" in p else p["note"], "case": cid})
        best, worst = derived(sid, "bound", "best"), derived(sid, "bound", "worst")
        contested = None
        if worst in by_id.index and pd.notna(by_id.loc[worst].get("household_total_worst_with_contested_gbp_inc_vat")):
            wr = by_id.loc[worst]
            comp = by_id.loc[comparator(worst)]
            contested = {"hh_delta": num(float(wr.household_total_worst_with_contested_gbp_inc_vat)
                                         - float(comp.household_total_gbp_inc_vat)),
                         "gilt": num(wr.get("contested_gilt_stock_externality_gbp_per_household")),
                         "mcpf": num(wr.get("contested_mcpf_deadweight_gbp_per_household"))}

        # the bound grid: best to worst, one assumption at a time (OW7 item 14)
        bname = sid + "#bounds"
        baxes, bmap, bsource = grid_axes_and_map(bname, BOUND_UI, "BOUND_UI")
        bounds = None
        if baxes:
            bpos = all_positions(baxes)
            bkeys = [position_key(baxes, p) for p in bpos]
            bunmapped = [k for k in bkeys if k not in bmap]
            if bunmapped:
                report.append(f"{bname}: {len(bunmapped)} positions map to no case id, e.g. {bunmapped[:3]}")
            for ax in baxes:
                bound_tags(ax, settings_of(best), settings_of(worst))

            def corner(kind):
                vec = {}
                for ax in baxes:
                    if ax["kind"] == "selector":
                        hit = next((o["token"] for o in ax["page"]["options"] if o.get(kind)), ax["base"])
                        vec[ax["token"]] = hit
                    elif ax["page"].get(kind):
                        vec[ax["token"]] = 1
                full = normalise(baxes, vec)
                return position_key(baxes, full) if full is not None else None

            bcorners = {"best": corner("best"), "worst": corner("worst")}
            for kind, cid in (("best", best), ("worst", worst)):
                at = bmap.get(bcorners[kind])
                if cid and at != cid:
                    report.append(f"{bname}: the all-{kind} position is {at}, not the {kind} case {cid}")
            bwith = {k: bmap[k] for k in bkeys if bmap.get(k) in cases}
            bounds = {"axes": [ax["page"] for ax in baxes], "grid": bwith, "positions": len(bkeys),
                      "mapped": len(bkeys) - len(bunmapped), "with_numbers": len(bwith), "axes_from": bsource,
                      "corners": bcorners, "interaction": interaction_error(baxes, bmap, cases)}
            grid_checks[code + " bounds"] = {k: bounds[k] for k in ("positions", "mapped", "with_numbers", "axes_from")}

        # the one-at-a-time list, generated from the catalogue (OW7 item 10)
        sens_rows = generate_sensitivities(sid, cat, settings_of, all_cases_levers, consts, set(), report)
        base_settings = settings_of(sid) or {}
        combos = combo_index(sens_rows, base_settings, all_settings, has_numbers) if all_settings else {}
        sens = []
        for i, r in enumerate(sens_rows):
            listed.add(r["case"])
            sens.append({k: v for k, v in r.items() if k != "_diff"} | {"i": i})
        # User decision (10 Nov 2026): the law in force from 2025 instead of 2028, from its own run group.
        law = f"{sid}{LAW_2025_SUFFIX}"
        if law in cases:
            sens.insert(0, {"group": "When the law is passed", "case": law, "label": "The law is passed in 2025, not 2028",
                            "explain": "Plant committed from 2025 counts as new state build rather than existing "
                                       "fleet, so the state builds more and buys less. Computed on its own run (the "
                                       "law-in-2025 register), for each scenario's base case only.",
                            "levers": ["legislated_year"], "moves_private": False, "i": len(sens_rows)})

        headline = None
        h = HEADLINE.get(sid)
        if h and (h["case"] in cat or h["case"] in cases):
            headline = dict(h)

        extra = {}
        if sid == P3 and sid in nat.get("cases", {}):
            c = nat["cases"][sid]
            head = c.get(f"headline_{model_year}_and_averages") or c["headline_2030_and_averages"]
            be = c["break_even_purchase_price"]
            diag = c["diagnostics"]
            tsp = c.get("thermal_and_storage_purchase", {})
            my = [m for m in c.get("multi_year", []) if m.get("case_id") == sid]
            series = [{"year": m["year"], "delta_inc_vat": num(m["bill_delta_gbp_per_3100kwh_ex_vat"] * (1 + vat_rate))}
                      for m in my]
            annuity = next((m["acquisition_annuity_gbp"] for m in my if str(m["year"]) == str(model_year)), None)
            ro = next((u for u in uncertainties(sid) if u["key"] == "ro_exchequer_relief_continued"), None)
            base_db = cases.get(sid, {}).get("db")
            # Integration (7 Oct 2026): the ledger's household-total break-even is the lifetime price too.
            be_hh = by_id.loc[sid].get("break_even_purchase_price_household_total_gbp") if sid in by_id.index else None
            be_hh_snapshot = (by_id.loc[sid].get("break_even_purchase_price_household_total_2030_snapshot_gbp")
                              if sid in by_id.index else None)
            extra = {
                "snapshot": num(head[f"{model_year} snapshot"]["delta_vs_private_gbp_inc_vat"]),
                "avg_15": num(head["15-year average"]["delta_vs_private_gbp_inc_vat"]),
                "avg_30": num(head["30-year average"]["delta_vs_private_gbp_inc_vat"]),
                "series": series,
                # F27 item 14: what the year path holds and what moves, shown with it.
                "series_basis": next((m.get("basis") for m in my if m.get("basis")), None),
                "purchase_bn": num(be["purchase_price_gbp"] / 1e9, 2),
                "purchase_renewables_nuclear_bn": num(diag["acquisition_price_renewables_nuclear_gbp"] / 1e9, 2),
                "purchase_thermal_storage_bn": num(diag["acquisition_price_thermal_storage_gbp"] / 1e9, 2),
                "interconnector_bn": num(diag["interconnector_acquisition_price_gbp"] / 1e9, 2),
                "annuity_bn": num(annuity / 1e9, 2) if annuity is not None else None,
                "term_years": c.get("acquisition_terms", {}).get("term_years"),
                # F28 (e): each plant repaid over its own remaining life (term_years is then the mean).
                "term_basis": c.get("acquisition_terms", {}).get("term_basis", "fixed"),
                "thermal_basis": c.get("acquisition_terms", {}).get("thermal_basis"),
                "per_10bn": num(be["bill_gbp_inc_vat_per_10bn_of_price"]),
                "break_even_bill_bn": num(be["break_even_purchase_price_bill_gbp"] / 1e9, 0),
                # F19 P3 decision 5: the headline is the lifetime break-even; the 2030 snapshot beside it.
                "break_even_snapshot_bill_bn": (num(be["break_even_purchase_price_2030_snapshot_bill_gbp"] / 1e9, 0)
                                                if be.get("break_even_purchase_price_2030_snapshot_bill_gbp") is not None
                                                else None),
                # F19 P3 decision 1: the bill change in three parts (inc VAT per household), the legacy
                # purchase's path and its present value at the public rate.
                **three_parts_fields(c.get("three_parts") or {}),
                "break_even_hh_bn": num(float(be_hh) / 1e9, 0) if be_hh is not None and pd.notna(be_hh) else None,
                "break_even_hh_snapshot_bn": (num(float(be_hh_snapshot) / 1e9, 0)
                                              if be_hh_snapshot is not None and pd.notna(be_hh_snapshot) else None),
                "thermal_at_market_value_effect": num(tsp.get("bill_effect_of_market_value_instead_gbp_inc_vat")),
                "ro_relief_gap": ro and ro["gap"],
                "ro_relief_share_of_saving": (num(ro["gap"] / abs(base_db), 3) if ro and base_db else None),
                "wires": (c.get("network_account") or {}).get("treatment"),
                # User, 13 Nov 2026 (results review, decision 4): S3 led by the 2027-2050 pathway on consumers plus
                # the Exchequer, with the purchase-basis and risk cases beside the headline.
                "pathway": s3_pathway_view(PO),
                "basis_cases": [
                    {"label": label, "case": cid, "db": cases[cid].get("db"), "dh": cases[cid].get("dh")}
                    for label, cid in S3_BASIS_CASES if cid in cases],
            }
        scenarios.append(annotate_topics({
            "id": sid, **meta,
            "axes": page_axes, "grid": gmap, "grid_note": GRID_NOTES.get(sid, ""),
            "presets": presets, "sensitivities": sens, "sens_combos": combos, "interaction": ierr,
            "bounds": bounds, "headline": headline,
            "best": best, "worst": worst,
            "best_words": bound_words(sid, best), "worst_words": bound_words(sid, worst),
            "contested": contested, "extra": extra, "uncertainties": uncertainties(sid),
        }))

    # ---- which catalogue cases the page shows nowhere, and why --------------------------------------
    shown = {"private"} | listed
    for s in scenarios:
        shown |= {s["id"], s["best"], s["worst"]} | {p["case"] for p in s["presets"]} | set(s["grid"].values())
        if s["headline"]:
            shown.add(s["headline"]["case"])
        if ccat and s["bounds"]:
            shown |= {cid for cid, v in ccat.items()
                      if any(m.get("grid") == s["id"] + "#bounds" for m in v.get("grid_memberships") or [])}
    comparators = {cases[c]["c"] for c in shown if c in cases and cases[c].get("c")}
    not_shown = {}
    for cid, v in cat.items():
        if cid in shown or cid is None:
            continue
        if cid.startswith("private@") and (cid in comparators or v.get("kind") in ("bound_comparator", "one_at_a_time")):
            why = ("a private comparator: it measures a case that also changes the private bill"
                   if v.get("lever") != "wholesale_pricing" else
                   "a bound on the private bill only (pay-as-bid ceiling); no public case to set it against")
        elif v.get("kind") == "preset":
            why = f"a who-pays preset of {v.get('parent')}, which the page has no panel for"
        else:
            why = f"{v.get('kind')} case under {v.get('parent')}"
        not_shown.setdefault(why, []).append(cid)

    # ---- trust panel ---------------------------------------------------------------------------
    # NESO's constraint figure is for 2030 (CP30); no NESO figure for another year.
    neso_bn, neso_source = neso_constraint_comparator(a.group) if model_year == 2030 else (None, None)
    trust = {"neso_cp30_constraint_bn": neso_bn, "neso_cp30_source": neso_source}
    bs = Path(a.bill_summary)
    if bs.exists():
        j = json.loads(bs.read_text(encoding="utf-8"))
        d = j.get("diagnostics", {})
        bal = j.get("balancing_costs", {})
        trust.update({
            "bill_summary_file": bs.name,
            "constraint_cost_bn": num(bal.get("constraint_cost_gbp", float("nan")) / 1e9, 2),
            "constraint_line_hh": num(d.get("bsuos_constraints_gbp_per_3100kwh")),
            "neso_outturn_hh": num(d.get("neso_constraint_outturn_gbp_per_3100kwh")),
            "neso_outturn_year": d.get("neso_comparator_regulatory_year"),
            "modelled_share": num(j.get("pypsa_derived_share_of_ex_vat"), 4),
            "informed_share": num(j.get("pypsa_informed_share_of_ex_vat"), 4),
            "bill_2500kwh": num((d.get("alternative_consumption_bills") or {}).get("2500", {}).get("total_gbp_inc_vat")),
        })
        # User, 13 Nov 2026: the electrified household (the bill's headline archetype) and what it is made of.
        arch = d.get("household_archetypes") or {}
        bills = {str(b_["archetype"]): b_ for b_ in arch.get("bills") or []}
        for tkey, akey in (("electrified", "headline_archetype"), ("electrified_high", "high_demand_archetype")):
            head = arch.get(akey)
            if head and head in bills:
                parts = (arch.get("composition") or {}).get(head) or []
                trust[tkey] = {
                    "archetype": head, "kwh": num(bills[head]["annual_consumption_kwh"], 0),
                    "standing_gbp_ex_vat": num(bills[head]["standing_charge_gbp_ex_vat"]),
                    "unit_p_per_kwh_ex_vat": num(bills[head]["unit_rate_p_per_kwh_ex_vat"], 2),
                    "parts": [{"part": str(p_.get("part")), "kwh": num(p_.get("kwh"), 0)} for p_ in parts]}
    en = Path(a.energy)
    if en.exists():
        e = pd.read_csv(en)
        avail = e[(e.component == "available_energy") & (e.status == "PASS")]
        dem = e[e.component == "demand"]
        trust["energy"] = {
            "file": en.name,
            "available_max_rel_diff": num(avail.relative_difference.abs().max(), 6),
            "demand_rel_diff": num(dem.relative_difference.abs().max(), 7) if len(dem) else None,
            "demand_twh": num(float(dem.cp30_gwh.iloc[0]) / 1e3, 2) if len(dem) else None,
        }
    # User decision (10 Nov 2026): today's wholesale price (the anchored price path's first anchor) beside the
    # model year's time-weighted price.
    today = (((cfg_po.get("acquisition") or {}).get("valuation") or {}).get("price_anchors") or {}).get("today") or {}
    if today and a.price_files:
        prices = pd.concat([pd.read_csv(f, index_col=0)["wholesale_price"] for f in a.price_files if Path(f).exists()])
        trust["wholesale"] = {
            "today_year": today.get("year"), "today_nominal": num(today.get("level_gbp_per_mwh_nominal")),
            "today_real": num(float(today["level_gbp_per_mwh_nominal"]) * float(today.get("deflator_to_2024", 1.0))),
            "model_year": model_year, "model_mean": num(float(prices.mean())), "hours": int(len(prices)),
            "source": "Ofgem default tariff cap direct-fuel electricity index, Jul-Dec 2026 (shaped to household demand)"}
    # Hinkley Point C on EDF's schedule: the per-case results sit in uncertainties_by_case.
    hp = (unc_all.get("private") or {}).get("hpc_edf_schedule")
    h3 = (unc_all.get(P3) or {}).get("hpc_edf_schedule")
    if hp and h3:
        rng = lambda e: sorted([num(e["range"]["low"]), num(e["range"]["high"])])
        trust["hpc_schedule"] = {
            "market_bill_change": num(hp["case_bill_change_gbp_inc_vat"]), "market_bill_range": rng(hp),
            "p3_bill_change": num(h3["case_bill_change_gbp_inc_vat"]), "p3_bill_range": rng(h3),
            "p3_saving_narrows_by": num(h3["gap_change_gbp_inc_vat"]),
        }
    dv = Path(a.dispatch_validation)
    if dv.exists():
        v = pd.read_csv(dv)
        disp = v[v.component == "dispatched_energy"].set_index("item")
        cur = v[v.component == "curtailment"]
        counted = v[v.status.isin(["PASS", "WARN", "FAIL"])]
        trust["dispatch"] = {
            "file": dv.name,
            "rel_diff": {k: num(disp.relative_difference.get(k), 4) for k in ("wind_offshore", "wind_onshore", "solar_pv")},
            "curtailment_model_twh": num(cur.model_gwh.sum() / 1e3, 1),
            "curtailment_cp30_twh": num(cur.cp30_gwh.sum() / 1e3, 1),
            "status": {k: int((counted.status == k).sum()) for k in ("PASS", "WARN", "FAIL")},
        }
        # Follow-up F15 (6 Oct 2026): a table on the market basis (ES1 rows judged on Stage 1)
        # carries the CP.08 constraint row; the page then words the comparison that way.
        con = v[v.component == "constraint_curtailment"]
        if len(con):
            trust["dispatch"].update({
                "basis": "market",
                "constraint_model_twh": num(con.model_gwh.sum() / 1e3, 1),
                "constraint_cp30_twh": num(con.cp30_gwh.sum() / 1e3, 1),
            })
            # CP.08 has a volume for NESO's recommended network and today's network only; on another
            # network case the page says which network NESO's figure is for.
            if any(network in a.group for network in NESO_CONSTRAINT_BY_NETWORK):
                trust["dispatch"]["constraint_cp30_note"] = (
                    "on its recommended network (NESO publishes no volume for this network case; "
                    "on today's network it is 62.6 TWh)")

    # ---- cost of capital and sources (OW7 item 9) -----------------------------------------------------
    workbook_path = Path(a.workbook) if a.workbook else None
    if workbook_path is None and snapshot_path.exists():
        import yaml
        src = (yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}).get("source_workbook")
        workbook_path = repo / src if src else None
    settings_table = run_settings.reset_index() if run_settings is not None else None
    wacc = cost_of_capital(snapshot_path, workbook_path, settings_table, cfg_po, levers_mod, consts, report)

    catalogue_count = len(cat)
    data = {
        "meta": {
            "generated": date.today().isoformat(),
            "run_label": a.run_label,
            "model_year": model_year, "pathway_short": pathway_short, "pathway_long": pathway_long,
            "consumption_kwh": 3100,
            "price_year": 2025,
            "vat_rate": vat_rate,
            "per_household_key": key_note,
            "households_equivalent_m": num(households_equiv_m, 2),
            "legal_incidence_theta": ledger.get("household_total", {}).get("legal_incidence", {}).get("theta"),
            "case_count": len(cases),
            "catalogue_count": catalogue_count,
            "base_rule": BASE_RULE, "networks_note": NETWORKS_NOTE,
            # User, 11 Nov 2026: the questions the page groups every assumption under, in order.
            "topics": [{"id": t, "title": q, "help": h, "none": na} for t, q, h, na in TOPICS],
            "grid_checks": grid_checks,
            "not_shown": {why: len(ids) for why, ids in not_shown.items()},
            "bound_detail": bool(a.bound_detail),
            "line_groupings": line_groupings,
            # F19: P3's institution step (I1 to I3) when its two solved years come from different solve runs.
            "institution_mixed_vintage": vintage.get("mixed_vintage"),
            "institution_solve_vintage": ({k: {kk: vintage[k].get(kk) for kk in ("first_written_utc", "last_written_utc",
                                                                                 "fingerprint")}
                                           for k in ("market_year", "central_dispatch_year") if vintage.get(k)}
                                          if vintage else None),
            # User decision (24 Oct 2026): every P3 case on central dispatch, an additive approximation.
            "p3_i3_view": p3_i3_view(ledger),
        },
        "private": {"id": "private", "uncertainties": uncertainties("private")},
        "scenarios": scenarios,
        "cases": cases,
        "flow_defs": flow_defs, "flows": flows_by,
        "line_defs": line_defs, "lines": lines_by,
        "strikes": {k: v for k, v in strikes_by.items() if k != "private"},
        "trust": trust,
        "wacc": wacc,
        "year_note": year_note(PO, a.group, model_year),
    }
    text = json.dumps(scenario_names(data), separators=(",", ":"), ensure_ascii=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out} ({len(text) / 1024:.0f} kB): {len(cases)} cases with numbers, "
          f"{catalogue_count} in the {'configuration' if ccat else 'run'} catalogue; GBP per household per system GBP {per_hh:.4e}")
    for s in scenarios:
        g = grid_checks[s["code"]]
        base = cases.get(s["id"], {})
        print(f"  {s['code']} bill {base.get('b')} ({base.get('db')}), Exchequer {base.get('x')} | best {s['best']}"
              f" worst {s['worst']} | grid {g['positions']} positions, {g['mapped']} mapped to "
              f"{g['distinct_cases']} cases, {g['with_numbers']} with numbers (axes from {g['axes_from']})"
              f" | {len(s['sensitivities'])} sensitivities, {len(s['sens_combos'])} exact combinations of them"
              f" | interaction error {s['interaction']}")
        b = s.get("bounds")
        if b:
            print(f"     bounds: {b['positions']} positions, {b['mapped']} mapped, {b['with_numbers']} with numbers"
                  f" (axes from {b['axes_from']}); corners {b['corners']}; interaction error {b['interaction']}")
        if s.get("headline"):
            hc = s["headline"]["case"]
            print(f"     headline {hc}: {cases.get(hc, {}).get('db')} ({s['headline']['status']})")
    print(f"  catalogue cases shown nowhere on the page: {sum(len(v) for v in not_shown.values())}")
    for why, ids in not_shown.items():
        print(f"    {len(ids)}: {why}")
        for cid in ids:
            print(f"        {cid}")
    if wacc:
        print(f"  cost of capital: {len(wacc['rows'])} technologies; workbook hash checked: {wacc['hash_checked']}")
    for line in report:
        print("  note:", line)
    if any(g.get("unmapped") for g in grid_checks.values()):
        raise SystemExit("some UI positions map to no case id (listed above)")


if __name__ == "__main__":
    main()
