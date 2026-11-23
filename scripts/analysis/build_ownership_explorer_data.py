"""Assemble the data the public-ownership explorer artifact embeds (item K1 reporting).

One JSON for every case the finance layer produces: its lever settings, the
bill it takes (market bill, acquisition tariff or institution comparison), its
comparator, the component breakdown, the annual savings and, for the acquisition
cases, the multi-year path.  Everything is read from the saved
``resources/public_ownership/cp30_2030_*`` outputs; nothing is computed here
beyond differences and labels.

    python scripts/analysis/build_ownership_explorer_data.py [--out resources/analysis/ownership_explorer_data.json]
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.levers import LEVERS, STRUCTURAL_LEVERS, case_settings  # noqa: E402
from scripts.public_ownership.nationalised_tariff import NEW_LINE_LABELS  # noqa: E402
from scripts.public_ownership.reference_runs import configured_case_specs  # noqa: E402
from config.config_loader import load_defaults

PO = PROJECT_ROOT / "resources" / "public_ownership"
GROUP = "cp30_2030_cp30_network"

SCENARIOS = {
    "private": {"label": "Private ownership", "short": "S1",
                "description": "Today's ownership and financing: private capital at CEPA hurdle rates, CfDs at their strikes, the wires under RIIO.",
                "guide": {
                    "change": "This is the central comparator. Post-2025 generation and storage remain privately owned; legacy assets, plant in construction and the regulated networks also remain private.",
                    "mechanism": "Technology-specific private WACCs recover capital through CfDs, merchant revenue and support mechanisms. Existing CfDs keep their signed strikes and networks recover their allowed RIIO revenue.",
                    "held_fixed": "The CP30 2030 demand, capacity, fuel and carbon costs, weather, network topology and solved dispatch are the common physical background used by the other central cases.",
                    "caution": "It is a modelled 2030 private-finance benchmark, not today's observed household bill. Sensitivities that alter the common comparator are compared with a private case carrying the same alteration."
                }},
    # Since item L2 the two public scenarios ARE the policy propositions P1 and
    # P2: a scenario that sits conservative on every lever at once is a floor,
    # not a design a government would legislate.  The bundles each scenario used
    # to hold are kept as the @conservative_stake and @conservative_developer
    # cases underneath them.
    "public_control": {"label": "Public control: GB Energy as legislated", "short": "S2 / P1",
                       "description": "GB Energy and the National Wealth Fund take a 30% stake in the post-2025 build; the Capacity Market and CCS savings are booked on the bill, the state's surplus is rebated to bills and any shortfall is carried by the Exchequer (bill-payers first). The wires stay private; guaranteeing or buying them is a choice beside it.",
                       "guide": {
                               "change": "The state takes a 30% equity stake in post-2025 generation and storage. The other 70% stays private; the legacy fleet, pre-2025 CfDs, the wires and Hinkley Point C remain private.",
                               "mechanism": "The financing calculation reduces debt cost by 75 basis points and removes the 4.7 percentage-point equity risk premium from the state's 30% share. The Capacity Market and CCS financing savings are booked as bill lines. Every positive balance of the state is rebated to bills and every deficit is carried by the Exchequer; the rules before 30 September 2026 (the state's return kept by the Exchequer, the guarantee on the wires) are the today's-rules case and the networks choice.",
                               "held_fixed": "Ownership does not change the wholesale dispatch, market-clearing rule or physical system. Balancing costs are held fixed across every ownership case. That is not a one-way understatement: item BM1 re-solved the balancing mechanism with the state's share bidding at cost under full new-build ownership and the bill moved by __BM1_BID_STACK__ a household, part of it the state paying itself and part returned as CfD payments on extra private output; a 30% stake moves less.",
                               "caution": "This is a co-investment and financing case, not full public ownership. The strike stays at the auction-cleared level reduced by the financing saving, because a minority stake leaves a private majority owner on a fixed-term CfD — at-cost is not the contract that would be written. The paired case routes the value to bills instead, and it comes out worse."
                       }},
    "public_ownership": {"label": "Public ownership: the state developer", "short": "S3 / P2",
                         "description": "A state-owned developer-owner of the post-2025 fleet at the real public borrowing rate, selling at cost over the assets' lives, with the Capacity Market and CCS routes booked; its surplus is rebated to bills and any shortfall carried by the Exchequer (bill-payers first); the legacy fleet and the wires stay private.",
                         "guide": {
                                 "change": "The state owns 100% of post-2025 generation and storage and contracts its output at cost over each asset's economic life. Existing plant, pre-2025 CfDs, regulated networks and Hinkley Point C remain private, so this stays on the market bill and is directly comparable with private ownership.",
                                 "mechanism": "New assets earn the 2.745% real public borrowing rate plus a construction-risk premium graded from 0.5 to 1.5 percentage points. The owned contracts are re-priced to the asset's cost rather than by the financing saving alone, amortised over economic life rather than the CfD contract term, because the state holds the asset for its life. The Capacity Market and CCS savings are booked as bill lines.",
                                 "held_fixed": "The common CP30 2030 physical run and pay-as-clear market remain in place. Buying existing assets, buying the networks and central dispatch are separate cases rather than hidden in the headline. Balancing costs are held fixed across every ownership case.",
                                 "caution": "New-build ownership at 100% by 2030 is a deliverability claim, not a financing one: 83 GW of the 203.8 GW eligible. It is also not full nationalisation — the acquisition cases use a different tariff calculation and must be read together with their purchase price."
                         }},
}
VALUE_LABELS = {
    "new_build_generation": {"private": "private", "state_stake": "30% state stake", "public": "state-owned",
                             "gbe_announced": "GB Energy as announced",
                             "gbe_announced_partnership_half": "GB Energy as announced, local power in partnership"},
    "pre_2025_cfd_capacity": {"private": "keeps its private owner and strike", "public_repriced": "state-owned and re-priced",
                              "public_bought_owners_value": "bought at its owners' value and re-priced"},
    "legacy_fleet": {"private": "private", "acquire_valued_group": "renewables and nuclear bought", "acquire_all": "all bought, thermal and storage too"},
    "networks": {"private": "private", "acquired": "RAV bought at the public rate"},
    "new_build_storage": {"follows_generation": "follows generation", "private": "private"},
    "in_construction": {"private": "bought as an operating plant", "public_completion": "completed by the state at the public rate",
                        "left_private": "left out of the purchase, private on its own CfD"},
    "efw_acquisition": {"excluded": "left with their owners (central)", "included": "bought with the rest of the fleet"},
    "private_wacc_basis": {"notional_gearing_shielded": "Fixed ERP, notional 55% gearing, tax shield (workbook lead)", "fixed_tmr_shielded": "Fixed TMR (CEPA / UKRN)"},
    "public_return": {"public_rate": "the state's cash rate (gilt + NLF margin)", "cepa_decomposition": "workbook Sc2 (30% stake at the risk-free rate, two-notch guarantee)"},
    "public_rate_basis": {"real_cpi": "NLF margin, +11 bp (2.76%)", "pwlb_hra": "PWLB HRA, +40 bp (3.05%)",
                          "pwlb_certainty": "PWLB certainty, +80 bp (3.45%)",
                          "pwlb_standard": "PWLB standard, +100 bp (3.65%)", "gilt_flat": "gilt flat (2.65%)"},
    "systematic_premium": {"social_flat_19": "a flat 1.9% (Scenario 4 sensitivity)", "none": "none: gilts alone (central)",
                           "social_flat_10": "a flat 1%", "social_beta_20": "2% per unit of beta",
                           "market_erp": "the market ERP, 5.22% per unit of beta"},
    "sovereign_yield": {"A": "A: yields unchanged", "B": "B: yields rise with debt and contingent exposure"},
    "purchase_price_basis": {"state_present_value": "state present value", "market_value": "owners' value (workbook A2, after tax)",
                             "book_value": "net book value (workbook A1)",
                             "no_support": "owners' value without support (workbook A2 no-support)",
                             "ro_pass_through": "RO plant at no-support value (workbook A3)"},
    "thermal_basis": {"zero": "zero", "market_value": "owners' value (market value)"},
    "acquisition_term_years": {"remaining_life": "Each asset's remaining life", "30": "30 years", "15": "15 years"},
    "legacy_contracts_under_acquisition": {"honoured": "honoured", "cancelled_residual_on_bills": "cancelled, residual on bills",
                                           "cancelled_residual_on_exchequer": "cancelled, residual on the Exchequer"},
    "cfd_strike_basis": {"financing_saving_only": "strike falls by the financing saving", "at_cost": "strike at cost",
                         "auction_clearing": "auction strike kept, return to the Treasury"},
    "cfd_amortisation": {"contract_term": "over the CfD contract term", "economic_life": "over economic life"},
    "surplus_recycling": {"public_cfd_at_cost": "public CfD at cost", "cost_recovery_contracting": "cost-recovery contracting", "rebate_per_mwh_eligible_supply": "rebate per MWh", "offset_policy_levies": "offsets domestic RO, FiT and WHD", "retained_by_treasury": "retained by the Treasury", "no_assumption": "no assumption", "rebate_gross": "every surplus rebated, deficits to the Exchequer"},
    "shortfall_funding": {"levy_on_eligible_supply": "levy on eligible supply", "exchequer": "Exchequer"},
    "route_booking": {"sensitivity_only": "reported beside the bill", "booked": "booked on the bill"},
    "new_build_contracts": {"merchant": "merchant", "at_cost": "contracted at cost"},
    "ldes_floor_basis": {"cost_of_debt_like": "debt-like rate", "technology_wacc": "technology WACC"},
    "state_overhead": {"central": "3 GBP/kW/yr (central)", "low": "1 GBP/kW/yr", "high": "8 GBP/kW/yr"},
    "wholesale_pricing": {"pay_as_clear": "pay as clear", "pay_as_bid_ceiling": "pay-as-bid ceiling"},
    "dispatch_institution": {"market": "market (I1)", "nationalised_dispatch": "central dispatch (I3)"},
    "balancing_bids": {"contract_value": "contract value, every unit", "state_at_cost": "state share at cost"},
    # Review 29 Sep 2026: the incidence, background and assumption levers.
    "state_deficit_funding": {"exchequer": "Exchequer", "bill_levy": "levy on bills"},
    "tax_equivalent": {"none": "none", "on": "tax equivalents paid"},
    "network_incidence": {"bills": "saving to bills", "return_retained": "return kept by the Treasury"},
    "acquisition_financing": {"tariff": "on the tariff", "exchequer": "from general taxation"},
    "tariff_pricing": {"cost": "at cost", "market_bill": "market prices, the Treasury keeps the margin"},
    "legacy_levy_funding": {"bills": "on bills", "domestic_to_taxation": "domestic share to taxation"},
    "beccs_removal_funding": {"exchequer": "Exchequer", "bill_levy": "levy on bills"},
    "network_acquisition_premium": {"central": "50% over RAV", "low": "20% over RAV", "high": "75% over RAV",
                                    "statutory_at_rav": "at RAV (Parliament-set)", "quarter": "25% over RAV"},
    "state_stake_scale": {"workbook_stake": "30% stake", "funded_scale": "15% stake"},
    "new_build_delivered_share": {"all": "all of it", "half": "half of it"},
    "state_capex_overrun": {"none": "none", "plus_15pct": "+15% capex"},
    "state_om_premium": {"same": "same as private", "plus_10pct": "+10% O&M"},
    "generation_control_premium": {"none": "none", "twenty_pct": "20% over market value"},
    "interconnector_price": {"central": "owner rate 4.87% (workbook)", "low": "owner rate 5.87%", "high": "owner rate 3.87%"},
    "acquisition_transaction_costs": {"none": "none", "two_pct": "2% of the consideration"},
    "legacy_asset_lives": {"central": "central lives", "extended": "extended lives"},
    # F19 (6 Oct 2026), F29 (26 Oct 2026): the legacy valuation's price path (the model's own 2030 price central).
    "legacy_price_path": {"anchored": "a path through today's prices, 2030 and 2050 (central)", "cp30_level": "CP30's 2030 price level", "model_2030": "the model's 2030 price held flat", "today_level": "today's forward prices held for the fleet's life"},
    "auction_margin": {"central": "5% of contract revenue", "none": "none (break-even)", "high": "10% of contract revenue"},
    "self_insurance": {"central": "35% of premiums saved", "none": "commercial insurance", "high": "50% of premiums saved"},
    # F27 item 6 (24 Oct 2026).
    "capacity_procurement": {"auction": "Capacity Market clearing price",
                             "cost_based_existing": "existing plant on cost-based contracts"},
    # User decision (24 Oct 2026).
    "legacy_offshore_price": {"owners_value": "owners' value (model cash flows)",
                              "transaction_multiple": "observed deal multiples"},
}
LEVER_LABELS = {
    "new_build_generation": "New-build generation", "pre_2025_cfd_capacity": "Pre-2025 CfD capacity", "legacy_fleet": "Legacy fleet",
    "networks": "Networks", "new_build_storage": "New-build storage", "in_construction": "Plant in construction", "efw_acquisition": "Energy-from-waste in the purchase",
    "private_wacc_basis": "Private WACC basis", "public_return": "Public return", "public_rate_basis": "Public rate basis",
    "systematic_premium": "Social price of the risk the state holds", "sovereign_yield": "Sovereign yields", "purchase_price_basis": "Purchase price basis",
    "thermal_basis": "Thermal purchase basis", "acquisition_term_years": "Acquisition term",
    "legacy_contracts_under_acquisition": "Acquired plant's support contracts",
    "cfd_strike_basis": "Strike basis", "cfd_amortisation": "Re-pricing amortisation", "surplus_recycling": "Surplus recycling",
    "shortfall_funding": "Shortfall funding", "route_booking": "CM and CCS routes", "new_build_contracts": "Merchant new build",
    "ldes_floor_basis": "LDES floor basis", "state_overhead": "State operating overhead",
    "wholesale_pricing": "Wholesale pricing", "dispatch_institution": "Dispatch institution",
    "balancing_bids": "Balancing-mechanism bids",
    "state_deficit_funding": "Who funds state deficits", "tax_equivalent": "Tax equivalents",
    "network_incidence": "Where the network saving lands", "acquisition_financing": "Who pays the purchase",
    "tariff_pricing": "What the public tariff charges",
    "legacy_levy_funding": "Legacy levies", "beccs_removal_funding": "BECCS carbon-removal contract",
    "network_acquisition_premium": "Network purchase premium", "state_stake_scale": "State stake",
    "new_build_delivered_share": "New build the state delivers", "state_capex_overrun": "Construction overrun",
    "state_om_premium": "Public operating costs", "generation_control_premium": "Generation control premium",
    "interconnector_price": "Interconnector price", "acquisition_transaction_costs": "Transaction costs",
    "legacy_asset_lives": "Legacy asset lives",
    "legacy_price_path": "Legacy fleet's price after 2030",
    "auction_margin": "Auction margin of calibrated CfDs",
    "self_insurance": "State plant self-insures",
    "capacity_procurement": "How existing capacity is paid",
    "legacy_offshore_price": "Price of existing offshore wind",
}
NAMED_LABELS = {
    "fully_nationalised": ("State buys the existing fleet", "The state buys the existing fleet as well as owning the new build: legacy plant bought at its state present value. The wires stay private unless the networks choice adds them."),
    "fully_nationalised_market_value": ("Public power system, market kept", "The state buys the existing fleet at what its owners would sell for and owns all generation and storage, dispatched through the market machinery, with a cost-reflective tariff. The wires stay private unless the networks choice adds them."),
    "fully_nationalised_book_value": ("State buys the existing fleet, book value", "As the state buying the existing fleet, every acquired plant at its net book value (the workbook's Approach 1)."),
    "fully_nationalised_no_support_value": ("State buys the existing fleet, without support", "As the state buying the existing fleet, every acquired plant at its owners' value without its CfD and RO income (the workbook's no-support value)."),
    "fully_nationalised_ro_pass_through": ("State buys the existing fleet, RO pass-through", "As the state buying the existing fleet, RO plant at its no-support value and the rest at owners' value (the workbook's Approach 3, a policy case)."),
    "fully_nationalised_thermal_at_zero": ("State buys the existing fleet, thermal at zero", "As the state buying the existing fleet, legacy thermal and storage taken at no price."),
    "fully_nationalised_term_15y": ("State buys the existing fleet, 15-year term", "As the state buying the existing fleet, the purchase annuitised over 15 years."),
    # F19 (6 Oct 2026): P3's purchase-price sensitivities.
    "fully_nationalised_market_value_price_path_today": ("Public power system, fleet valued on today's prices", "The public power system with the existing fleet valued as if today's forward wholesale prices lasted its whole life: a higher price."),
    "fully_nationalised_market_value_price_path_cp30": ("Public power system, CP30 price after 2030", "The public power system with the existing fleet valued on CP30's 2030 price level rather than the model's own 2030 price held flat: a higher price."),
    # F27 item 6 (24 Oct 2026).
    "fully_nationalised_market_value_capacity_cost_based": ("Public power system, capacity on cost-based contracts", "The public power system with existing plant valued on cost-based capacity contracts (its going-forward fixed cost less its own margin) instead of its Capacity Market income, announced before the purchase: a lower price."),
    # User decision (24 Oct 2026).
    "fully_nationalised_market_value_offshore_deal_multiples": ("Public power system, offshore at deal prices", "The public power system with existing offshore wind bought at what recent UK stake sales paid per megawatt (about £4.6m) rather than the model's own value of its cash flows (about £2.3m): a higher price."),
    "fully_nationalised_market_value_central_dispatch": ("Public power system", "All generation and storage in public hands, the existing fleet bought at market value, and a nationalised operator dispatching centrally at resource cost (institution I3), on the second solved year. The wires stay private unless the networks choice adds them."),
    # Group L policy propositions (22 September 2026), as item L2 left them:
    # P1 and P2 are the SCENARIOS, so only P1b and the P3 variant are named.
    "gbe_value_to_bills": ("GB Energy, value returned to bills", "The same institutions as the scenario, with the state's position routed to bills through a CfD at cost instead of to the Exchequer."),
    "fully_nationalised_market_value_term_15y": ("Public power system, 15-year term", "The public power system with the market-value purchase repaid over 15 years rather than asset by asset over each asset's remaining life."),
    # F28 (e), 26 Oct 2026: the central repays each asset over its remaining life; one 30-year annuity is the toggle.
    "fully_nationalised_market_value_term_30y": ("Public power system, one 30-year term", "The public power system with the whole market-value purchase repaid as one 30-year annuity rather than asset by asset over each asset's remaining life."),
    # Review 29 Sep 2026.
    "fully_nationalised_market_value_prices_held_at_2030": ("Public power system, fleet valued on 2030 prices held flat", "The existing fleet's income is valued on the model's 2030 prices held flat for its whole life, instead of the path through today's prices, 2030 and 2050."),
    "fully_nationalised_market_value_efw_acquired": ("Public power system, energy-from-waste bought too", "The purchase also buys the energy-from-waste plants at their owners' value (mostly their gate fees), which the central leaves with their owners."),
    "fully_nationalised_market_value_hpc_left_private": ("Public power system, Hinkley Point C left private", "Hinkley Point C is left out of the purchase and stays private on its own CfD, instead of bought as an operating plant."),
    "fully_nationalised_market_value_systematic_premium": ("Public power system, gilts plus the systemic risk premium", "The state's plant is charged the workbook's 1.9% systemic risk premium on top of the gilt-based cash rate, as a taxpayers' cost (off the bill)."),
    # 13 Nov 2026 (user, results review): the risk premium at HM Treasury's adopted 1%, and on Scenario 2.
    "fully_nationalised_market_value_systematic_premium_1pc": ("Public power system, gilts plus a 1% risk premium", "The state's plant is charged a flat 1% risk premium (the rate HM Treasury adopted in 2026) on top of the gilt-based cash rate, as a taxpayers' cost (off the bill)."),
    "systematic_premium_1pc": ("Public new build, gilts plus a 1% risk premium", "The plant the state builds is charged a flat 1% risk premium (the rate HM Treasury adopted in 2026) on top of the gilt-based cash rate, as a taxpayers' cost (off the bill)."),
    "systematic_premium_1pc9": ("Public new build, gilts plus a 1.9% risk premium", "The plant the state builds is charged the workbook's 1.9% risk premium on top of the gilt-based cash rate, as a taxpayers' cost (off the bill)."),
    "fully_nationalised_market_value_yield_b": ("Public power system, gilts rise with debt", "Gilt yields rise with the scenario's own debt and contingent exposure over GDP (2 bp per point, the workbook's yield path B)."),
    "fully_nationalised_market_value_hpc_completion": ("Public power system, HPC completed by the state", "Hinkley Point C bought at its sunk cost and finished at the public cash rate, instead of bought operating at its contract value."),
    "fully_nationalised_market_value_legacy_levies_to_taxation": ("Public power system, legacy levies taxed", "The domestic share of the legacy policy levies paid from taxation."),
    "fully_nationalised_market_value_beccs_removal_on_bills": ("Public power system, BECCS removal on bills", "The BECCS carbon-removal contract levied on bills rather than funded by the Exchequer."),
    "networks_acquired_rav_at_statute": ("State developer and the wires, compensation at RAV", "P2b with the wires bought at their regulated asset value by statute, with no premium."),
    "fully_nationalised_market_value_rav_at_statute": ("Public power system and the wires, at RAV", "P3 with the wires bought as well, at their regulated asset value by statute, with no premium."),
    # The bundles the public scenarios held before item L2, kept so the
    # conservative reading is still on the page.
    # Item G3, lever L19: the acquired plant's contracts cancelled rather than honoured.
    "legacy_acquired_contracts_cancelled_on_bills": ("Legacy bought, contracts cancelled (bills)", "Renewables and nuclear bought without their CfD and RO income; the former owners' residual is paid on the tariff."),
    "legacy_acquired_contracts_cancelled_on_exchequer": ("Legacy bought, contracts cancelled (Exchequer)", "The same, with the residual paid by the Exchequer and reported as a transfer."),
    "legacy_and_thermal_acquired_contracts_cancelled_on_bills": ("All legacy bought, contracts cancelled (bills)", "The whole legacy fleet bought, the renewables' and nuclear contracts cancelled, the residual on the tariff."),
    "legacy_and_thermal_acquired_contracts_cancelled_on_exchequer": ("All legacy bought, contracts cancelled (Exchequer)", "The same, with the residual paid by the Exchequer."),
    "conservative_stake": ("The stake and nothing else", "Public control as it was read before the propositions: the 30% stake, no guarantee on the wires, the Capacity Market and CCS savings reported rather than booked, and the state's position returned to bills."),
    "conservative_developer": ("State ownership on the auction strike", "Public ownership as it was read before the propositions: the state owns the new build but its contracts keep the auction-cleared strike over the CfD contract term, and the routes are not booked."),
    "strike_at_cost_over_life": ("Stake, contracts cut to cost over the plant's life", "Under the state stake, contracts are cut to cost, with the capital costed over the plant's economic life rather than the 20-year contract term."),
    "stake_networks_acquired_rav_premium_25pct": ("Stake and the wires, 25% premium", "The state stake with the wires bought at their regulated asset value plus a 25% premium."),
    "networks_acquired_rav_premium_25pct": ("State developer and the wires, 25% premium", "P2 with the wires bought at their regulated asset value plus a 25% premium."),
    "fully_nationalised_market_value_rav_premium_25pct": ("State buys everything, wires at a 25% premium", "P3 with the wires bought at their regulated asset value plus a 25% premium."),
    "gbe_announced": ("GB Energy as announced", "A 30% stake in post-2025 offshore wind (the Crown Estate partnership) and 8 GW of local solar and onshore wind at the public rate; everything else private."),
    "gbe_announced_partnership_half": ("GB Energy as announced, local power in partnership", "As GB Energy as announced, with the 8 GW of local power held 50% in partnership."),
    "developer_storage_private": ("State developer, storage left private", "P2 with new-build storage left to private owners."),
    "recycling_cost_recovery_contracting_on_developer": ("State developer, cost-recovery contracting", "P2 returns its surplus through cost-recovery contracting."),
    "recycling_rebate_per_mwh_on_developer": ("State developer, rebate per MWh", "P2 rebates its surplus over eligible supply."),
    "recycling_offset_policy_levies_on_developer": ("State developer, offsets levies", "P2 uses its surplus to offset domestic customers' legacy policy levies."),
    "recycling_no_assumption_on_developer": ("State developer, no recycling assumed", "P2 makes no assumption about where its surplus goes."),
    "networks_acquired_rav_premium_low": ("State developer and the wires, low premium", "P2 with the wires bought at the low premium over their regulated asset value."),
    "networks_acquired_return_retained": ("State developer and the wires, return retained", "P2 with the wires bought, the network saving retained by the state rather than passed to bills."),
}
#: The policy propositions, by CASE ID and in the order a reader meets them.
#: Since item L2 two of them are scenarios rather than sensitivities, so this
#: cannot be keyed on the variant suffix any more.  P1 and P2 keep the
#: "scenario" kind because they are the scenarios; the rest are surfaced as
#: their own kind so they are not buried among the analytical named positions.
PROPOSITIONS = {
    "public_control": "P1",
    "public_control@gbe_value_to_bills": "P1b",
    "public_ownership": "P2",
    "public_ownership@public_rate_pwlb_standard": "P2 stress",
    "public_ownership@yield_b": "P2 stress",
    "public_ownership@networks_acquired": "P2b",
    "public_ownership@fully_nationalised_market_value_central_dispatch": "P3",
    "public_ownership@fully_nationalised_market_value": "P3 alt",
    "public_ownership@fully_nationalised_market_value_term_15y": "P3 stress",
}
PROPOSITION_ORDER = tuple(PROPOSITIONS)
#: Reader-facing labels for the propositions that are NOT scenarios; a scenario
#: takes its label from SCENARIOS.
PROPOSITION_LABELS = {
    "public_ownership@networks_acquired": (
        "State developer and the wires",
        "The state developer, plus the regulated asset base bought and financed at the public rate."),
    "public_ownership@public_rate_pwlb_standard": (
        "State developer, PWLB standard margin",
        "The state developer stressed on the PWLB standard lending margin, 100 bp over gilts, instead of the National Loans Fund's 11 bp."),
    "public_ownership@yield_b": (
        "State developer, sovereign yield B",
        "The state developer stressed on sovereign yields rising with the debt stock."),
}
STACK_GROUPS = [
    ("hedged_wholesale", "Wholesale energy"), ("hedge_uplifts", "Hedging uplifts"), ("losses", "Losses"),
    ("system_balancing", "Balancing and capacity"), ("network", "Networks"), ("policy", "Policy costs"),
    ("supplier_operating", "Supplier costs"), ("nationalised_generation", "Cost of owned generation"),
]
WORKBOOK_LEVERS = ("private_wacc_basis", "public_rate_basis", "systematic_premium", "sovereign_yield")
OWNERSHIP_LEVERS = ("legacy_fleet", "networks", "wholesale_pricing")

LEVER_GROUPS = {
    "ownership": {"label": "Who owns what", "description": "The boundary between private, co-owned and publicly owned assets."},
    "capital": {"label": "Cost of capital", "description": "How private and public required returns are constructed."},
    "acquisition": {"label": "Acquisition terms", "description": "What is paid for existing assets and how that purchase is recovered."},
    "payment": {"label": "How public plant is paid", "description": "Which financing savings or portfolio cash flows reach the consumer bill."},
    "institution": {"label": "Market and institution", "description": "The settlement and dispatch arrangements around the assets."},
}

# Editorial explanations for the public artifact.  These describe the enforced
# values in LEVERS; they do not introduce any additional modelling assumption.
LEVER_GUIDES = {
    "new_build_generation": {
        "group": "ownership", "question": "Who owns generation commissioned from the 2025 policy boundary onward?",
        "mechanism": "Ownership selects the financing treatment and the share of required revenue or surplus assigned to the state. It does not change which generators are built or dispatched.",
        "held_fixed": "The installed capacity and hourly generation are inherited from the common CP30 2030 run.",
        "caution": "A public stake is not the same as 100% public ownership; the remaining share continues to require private returns.",
        "value_explanations": {
            "private": "All post-2025 build is privately financed at the selected technology-specific private WACC.",
            "state_stake": "The state owns 30% at the risk-free rate (no equity premium on its share) and guarantees the debt, lifting its rating two notches (workbook v2.0, Scenario 2).",
            "public": "The state owns 100% of post-2025 build and the selected public-return rule applies to the whole eligible asset.",
            "gbe_announced": "GB Energy as announced (F10): a 30% stake in post-2025 offshore wind through the Crown Estate partnership, financed and contracted as P1's stake, and 8 GW of local solar and onshore wind owned at the public rate as P2's; everything else private.",
            "gbe_announced_partnership_half": "As GB Energy as announced, with the 8 GW of local power held 50% in partnership with councils and cooperatives (the partner's half at the private rate).",
        }},
    "pre_2025_cfd_capacity": {
        "group": "ownership", "question": "Are assets with CfDs awarded before 2025 left with their owners and original contracts?",
        "mechanism": "Re-pricing older CfDs changes the policy-cost line because their signed strike is replaced by a lower public-finance strike for the owned capacity.",
        "held_fixed": "Output, wholesale prices and contract eligibility data are unchanged; only ownership and the strike paid on eligible output move.",
        "caution": "This assumes both acquisition and contractual re-pricing. It is broader than merely financing new assets publicly.",
        "value_explanations": {
            "private": "Pre-2025-award CfD capacity keeps its private owner and published strike.",
            "public_repriced": "The state also owns this capacity and re-prices its CfD using the case's strike and amortisation rules.",
            "public_bought_owners_value": ("As public_repriced, but the state pays the owners their cost plus the present value of "
                                           "their rent above their required return over the contract's remaining years (F27 item 1)."),
        }},
    "legacy_fleet": {
        "group": "ownership", "question": "Does the state buy operating assets that pre-date the new-build policy boundary?",
        "mechanism": "The market bill is replaced by an acquisition tariff: operating costs, capital recovery and the purchase annuity are charged directly instead of paying the acquired fleet through market and support revenues.",
        "held_fixed": "The central acquisition cases keep market dispatch; only the separately labelled central-dispatch case changes the operating institution.",
        "caution": "Large bill reductions are not free savings: they are conditional on a material upfront purchase price, shown beside every acquisition result.",
        "value_explanations": {
            "private": "The legacy fleet remains privately owned and is paid through the market and existing support arrangements.",
            "acquire_valued_group": "The state buys legacy renewables, hydro, marine and nuclear at the selected valuation basis.",
            "acquire_all": "The valued group plus legacy thermal generation and storage are bought; thermal and storage use their own purchase basis.",
        }},
    "networks": {
        "group": "ownership", "question": "Do transmission and distribution remain under private RIIO finance?",
        "mechanism": "Acquisition replaces allowed private financing and the corporation-tax allowance with public-rate finance on the Regulatory Asset Value.",
        "held_fixed": "Network topology, reinforcement and power flows are unchanged; this is a regulated-finance adjustment to TNUoS and DUoS.",
        "caution": "The distribution RAV for 2030 is rolled forward from ED2 because ED3 was not final; acquisition is at RAV plus the premium the networks premium choice sets. The debt guarantee was retired with workbook v2.0, which carries no network rate.",
        "value_explanations": {
            "private": "Networks keep their private RIIO allowed return and tax allowance.",
            "acquired": "The state buys the RAV and finances it at the public cash rate; the private financing wedge and tax allowance leave network charges.",
        }},
    "new_build_storage": {
        "group": "ownership", "question": "Does post-2025 storage follow the generation ownership boundary?",
        "mechanism": "The selected owner determines the financing rate used to calculate storage required revenue and public-account cash flow.",
        "held_fixed": "Storage power, energy capacity, dispatch and efficiency do not change.",
        "caution": "Battery and long-duration storage do not all have the same route to bills; the LDES floor is modelled separately.",
        "value_explanations": {
            "follows_generation": "New storage takes the same private, 30%-state or fully public ownership treatment as new generation.",
            "private": "New storage remains privately financed even when new generation has a public owner.",
        }},
    "in_construction": {
        "group": "ownership", "question": "Who finances completion of plant already under construction, represented by Hinkley Point C?",
        "mechanism": "Public completion replaces the private capital-recovery rate on the remaining construction requirement with the public rate.",
        "held_fixed": "Capacity, commissioning treatment and operating profile are not changed.",
        "caution": "This is an accounting counterfactual, not a project-level estimate of the legal cost of changing ownership during construction.",
        "value_explanations": {
            "private": "Hinkley Point C remains privately financed under its existing treatment.",
            "public_completion": "The state completes the in-construction asset at the selected public return.",
            "left_private": "Hinkley Point C is left out of the purchase and stays private on its own CfD.",
        }},
    "efw_acquisition": {
        "group": "ownership", "question": "Does the purchase of the existing fleet include energy-from-waste plants?",
        "mechanism": "Bought, their owners are paid their value (mostly the gate fees councils pay) and the gate fees are credited to the tariff; left out, they stay with their owners and sell their power as today.",
        "held_fixed": "The plants, their output and every other acquisition term are unchanged.",
        "caution": "Energy-from-waste plants are waste infrastructure, mostly owned by waste companies under long council contracts; their value is about 90% gate fees.",
        "value_explanations": {
            "excluded": "Central: left with their owners and their council contracts.",
            "included": "Bought at their owners' value with the rest of the existing fleet.",
        }},
    "private_wacc_basis": {
        "group": "capital", "question": "Which of the workbook's two equity-premium conventions sets private and public-control rates?",
        "mechanism": "Workbook v6.2 holds the equity risk premium at CEPA's 5.22% (Fixed ERP, its default) so the refreshed risk-free rate passes through to the cost of equity; Fixed TMR holds the total market return at 6.83% and takes the premium as the residual, 4.23%.",
        "held_fixed": "Technology costs, asset lives, output, gearing and public-ownership rates are unchanged.",
        "caution": "Because this also changes the private baseline, public cases are compared with private ownership on the same basis, not with the central private bill. Fixed TMR gives lower private rates, so a smaller saving.",
        "value_explanations": {
            "notional_gearing_shielded": "Central: Fixed ERP, a notional 55% gearing, net of the interest tax shield (offshore wind 8.48% real).",
            "fixed_tmr_shielded": "Fixed TMR, CEPA's and UK regulators' convention, co-equal in the workbook (offshore wind 7.53% real).",
        }},
    "public_return": {
        "group": "capital", "question": "What return does publicly financed new build earn?",
        "mechanism": "The return feeds the annuity used to recover upfront capital cost; capital-intensive technologies are therefore most sensitive.",
        "held_fixed": "Operating costs, fuel, carbon, capacity and dispatch remain fixed.",
        "caution": "The state's cash rate carries no premium for the risk the state takes on; that risk is priced off the bill (the social price of risk).",
        "value_explanations": {
            "public_rate": "The state's cash rate: the real gilt (2.60%) plus the National Loans Fund's 11 bp margin, 2.71% real.",
            "cepa_decomposition": "The workbook's Scenario 2: a 30% state equity stake requiring only the risk-free rate, and a guarantee lifting the project's debt two notches on the iBoxx ladder.",
        }},
    "public_rate_basis": {
        "group": "capital", "question": "What margin over gilts does the state borrow at?",
        "mechanism": "The rate changes the annual capital charge on every asset the state finances at its cash rate.",
        "held_fixed": "The risk-free rate, asset costs and the social price of risk are unchanged.",
        "caution": "The margins are the Debt Management Office's lending rates (August 2026); a public corporation borrowing through the National Loans Fund takes the 11 bp.",
        "value_explanations": {
            "real_cpi": "Central: the National Loans Fund margin, 11 bp over the real gilt (2.71%).",
            "pwlb_hra": "The PWLB Housing Revenue Account rate, 40 bp over gilts (3.00%).",
            "pwlb_certainty": "The PWLB certainty rate, 80 bp over gilts (3.40%).",
            "pwlb_standard": "The PWLB standard rate, 100 bp over gilts (3.60%).",
            "gilt_flat": "The real gilt with no margin (2.60%).",
        }},
    "systematic_premium": {
        "group": "capital", "question": "What does the systematic risk the state takes on cost society?",
        "mechanism": "The workbook's economic rate adds to the cash rate a premium for the systematic risk the state holds: a flat 1.9% in v6.2 (the Green Book discount rate review). The state charges no such premium on bills; the difference is a cost to taxpayers and comes out of the resource saving.",
        "held_fixed": "Bills, the cash rate and every physical assumption are unchanged; only the Exchequer column and the resource saving move.",
        "caution": "The premium is a convention calibrated on official practice (the review's 1.9% estimate, its 1% recommendation, France's 2% per unit of beta), not an estimate; at the market equity premium the state's financing advantage largely disappears.",
        "value_explanations": {
            "social_flat_19": "Scenario 4 sensitivity: a flat 1.9% (the Green Book discount rate review's social-opportunity-cost estimate, Freeman & Groom 2026).",
            "none": "Central since 10 November 2026: no premium, gilts alone (pure Arrow-Lind, as in the current Green Book).",
            "social_flat_10": "A flat 1% (the review's recommendation).",
            "social_beta_20": "2% per unit of beta (France, Quinet 2013; workbook v2.0's central).",
            "market_erp": "The market equity risk premium, 5.22% per unit of beta: what a commercial investor requires.",
        }},
    "sovereign_yield": {
        "group": "capital", "question": "Does additional public borrowing feed back into gilt yields?",
        "mechanism": "The workbook's gilt-yield shift in the model year is added to every public and public-control rate: P1 on its Scenario 2 path, the state developer on Scenario 3, a case that also buys the fleet on Scenario 4.",
        "held_fixed": "No macroeconomic re-solve is performed; the workbook shift is an exogenous adjustment.",
        "caution": "In 2030 B adds about 2 bp under P1, 16 bp under P2 and 20 bp under P3 (2 bp per point of debt and contingent exposure over GDP, Poghosyan 2014). The workbook's assumption C is blank pending its input-output model.",
        "value_explanations": {
            "A": "Central: gilt yields are invariant to the ownership programme.",
            "B": "Yields rise with the programme's debt and contingent exposure over GDP.",
        }},
    "purchase_price_basis": {
        "group": "acquisition", "question": "What does the state pay for the existing fleet?",
        "mechanism": "The purchase price is converted into an annuity at the public rate and included in the acquisition tariff.",
        "held_fixed": "The same forecast cash-flow profile, retirement dates and 30-year central recovery term are used unless another named sensitivity changes them.",
        "caution": "Owners' value follows the workbook's Approach 2: after-tax cash flows plus capital allowances, at post-tax operating rates calibrated on Brookfield's 2024 offshore purchase. None of the bases is a negotiated price.",
        "value_explanations": {
            "state_present_value": "Discount forecast legacy cash flows at the public real rate: the value to a low-cost public owner.",
            "market_value": "Owners' value: after corporation tax, with the capital allowances still to come, at the workbook's post-tax operating rates (offshore wind 6.1%).",
            "book_value": "Net book value: the workbook's historic capex by cohort written down over its accounting life (Approach 1).",
            "no_support": "Owners' value without CfD and Renewables Obligation income: the value without the support government put in.",
            "ro_pass_through": "RO plant at its no-support value, the rest at owners' value: a policy case that embeds a transfer from RO shareholders.",
        }},
    "thermal_basis": {
        "group": "acquisition", "question": "What is paid for legacy thermal generation and storage when all assets are acquired?",
        "mechanism": "This amount adds to the purchase annuity in the nationalised tariff.",
        "held_fixed": "Their fuel, carbon, variable costs, dispatch and remaining lives remain as modelled.",
        "caution": "A zero price is a deliberately extreme sensitivity, not an assumption that legal acquisition would be costless. Under the workbook's book-value, no-support and RO pass-through bases thermal and storage follow the purchase basis.",
        "value_explanations": {
            "market_value": "Central: their owners' value, as renewables and nuclear are priced.",
            "zero": "Assume thermal and storage transfer at no purchase price.",
        }},
    "acquisition_term_years": {
        "group": "acquisition", "question": "Over how many years is the acquisition price recovered from the tariff?",
        "mechanism": "A shorter term repays principal faster and therefore raises the annual bill during the recovery window.",
        "held_fixed": "The total purchase price and public discount rate do not change.",
        "caution": "The annual bill effect cannot be interpreted without the recovery horizon: 15 years is a cash-flow timing sensitivity, not a higher underlying purchase price.",
        "value_explanations": {
            "30": "Central: annuitise the purchase over 30 years at the public rate.",
            "15": "Recover the same purchase price over 15 years, increasing the annual acquisition charge.",
        }},
    "legacy_contracts_under_acquisition": {
        "group": "acquisition", "question": "What happens to the CfD and Renewables Obligation contracts of the legacy plant the state buys?",
        "mechanism": "Honoured, the purchase price includes the contracts' income, valued at the public rate. Cancelled, the price excludes it and the former owners are paid its value at their own cost of capital, annuitised on the same term: on the bill, or by the Exchequer as a transfer.",
        "held_fixed": "The plant, its output and every other acquisition term are unchanged.",
        "caution": "Cancelling saves only the difference between the two discount rates on the contract income, never the income itself a second time. At the market-value purchase basis it saves nothing. The Exchequer variant lowers the bill by moving the residual to taxpayers, which is a transfer, not a saving to the country.",
        "value_explanations": {
            "honoured": "Central: the contracts pass to the state with the plant and are paid for in the purchase price.",
            "cancelled_residual_on_bills": "Cancel the contracts; the former owners' residual is a line on the tariff.",
            "cancelled_residual_on_exchequer": "Cancel the contracts; the Exchequer pays the residual, reported as a transfer.",
        }},
    "cfd_repricing_eligibility": {
        "group": "payment", "question": "Is CfD re-pricing based on when the contract was awarded or when capacity commissioned?",
        "mechanism": "The rule changes how much owned CfD output receives the lower public-finance strike on the bill.",
        "held_fixed": "Ownership, strike formula, output and contract data remain fixed.",
        "caution": "In these outputs the two definitions produce the same rounded headline for the central public cases; that is a data result, not proof the distinction is generally immaterial.",
        "value_explanations": {
            "contract_vintage": "Central: re-price contracts awarded from 2025 onward; earlier awards keep their strike even if the plant commissions later.",
            "commissioning_vintage": "Re-price every owned cohort commissioned from 2025 onward, including capacity awarded earlier.",
        }},
    "cfd_strike_basis": {
        "group": "payment", "question": "How is the strike of an owned CfD reduced?",
        "mechanism": "The lower strike reduces levy-funded CfD top-ups on every eligible MWh.",
        "held_fixed": "The published starting strike, output and wholesale reference price remain fixed.",
        "caution": "“At cost” here means re-setting a CfD strike to modelled required revenue; it is different from the surplus-recycling and merchant-contracting levers that also use the phrase at cost.",
        "value_explanations": {
            "financing_saving_only": "Central: subtract only the modelled financing saving from the settled contract strike, preserving any other strike margin.",
            "at_cost": "Set the strike directly to modelled full cost, removing both the financing wedge and any remaining margin above cost.",
        }},
    "cfd_amortisation": {
        "group": "payment", "question": "Over what period is the re-priced capital charge calculated?",
        "mechanism": "An annuity over the contract term must recover capital faster than an economic-life annuity, so it leaves a smaller near-term strike reduction. It applies on both strike bases: on the at-cost basis the strike recovers the asset's capital within the term (until 26 Sep 2026 the at-cost strike ignored this lever).",
        "held_fixed": "Capex, WACC, generation and contract eligibility are unchanged.",
        "caution": "Economic-life amortisation lowers the 2030 bill but assumes capital can be recovered after the current CfD term through later revenues or contracts.",
        "value_explanations": {
            "contract_term": "Calculate the capital charge over the CfD contract term (config contract_term_years, by round where set).",
            "economic_life": "Spread capital recovery over the technology's full economic life.",
        }},
    "surplus_recycling": {
        "group": "payment", "question": "How does cash generated by the public portfolio reach—or not reach—households?",
        "mechanism": "The rule determines whether surplus reduces a contract strike, appears as a rebate or levy offset, or remains outside the consumer bill.",
        "held_fixed": "Asset costs, output and the gross portfolio cash position do not change; only its destination changes.",
        "caution": "Financing saving and bill saving are therefore not synonymous. A saving retained by the Treasury is a public-account benefit but not a household-bill reduction.",
        "value_explanations": {
            "public_cfd_at_cost": "Central public cases: renewable and hydro output uses a two-way CfD at cost, returning its rent through the policy-cost line.",
            "cost_recovery_contracting": "Contract each eligible public asset or portfolio to recover cost rather than market revenue.",
            "rebate_per_mwh_eligible_supply": "Return surplus as a uniform rebate over eligible electricity supply.",
            "offset_policy_levies": "Use surplus to reduce existing policy levies.",
            "retained_by_treasury": "Keep the return in the public account or Exchequer; it does not reduce the displayed bill.",
            "no_assumption": "Report the cash position without assuming a route back to consumers.",
        }},
    "shortfall_funding": {
        "group": "payment", "question": "Who covers the public portfolio when cost exceeds its routed revenue?",
        "mechanism": "A levy adds the shortfall to the consumer bill; Exchequer funding leaves it off the bill but transfers it to public finances.",
        "held_fixed": "The size of the underlying portfolio shortfall is unchanged.",
        "caution": "Moving a shortfall off-bill is not an economy-wide saving; taxpayers still finance it.",
        "value_explanations": {
            "levy_on_eligible_supply": "Central: recover the shortfall from eligible electricity supply and show it on the bill.",
            "exchequer": "The Exchequer covers the shortfall, so it is reported outside the household electricity bill.",
        }},
    "route_booking": {
        "group": "payment", "question": "Are financing savings on Capacity Market and CCS-supported assets placed on the bill?",
        "mechanism": "Booking subtracts the calculated financing adjustment from the capacity and policy-cost lines; reporting-only leaves the displayed bill unchanged.",
        "held_fixed": "Capacity obligations, CCS support, dispatch and the calculated financing saving stay fixed.",
        "caution": "The booked case assumes a settlement route capable of passing the saving through; the central case reports it beside the bill because that route is not fully specified.",
        "value_explanations": {
            "sensitivity_only": "Central: calculate and disclose CM and CCS financing effects but do not include them in the headline bill.",
            "booked": "Apply those financing effects directly to the corresponding bill lines.",
        }},
    "new_build_contracts": {
        "group": "payment", "question": "Is merchant new build left to market revenue or guaranteed modelled cost recovery?",
        "mechanism": "At-cost contracting replaces the merchant revenue position with required revenue; in these data some new merchant plant is under-recovering, so the contract can raise rather than lower bills.",
        "held_fixed": "Build, generation and wholesale prices are unchanged.",
        "caution": "This changes the basis of the private comparator too. It tests a market-design assumption, not a benefit unique to public ownership.",
        "value_explanations": {
            "merchant": "Central: new assets without a CfD receive the modelled merchant revenue available to them.",
            "at_cost": "Give merchant new build a contract paying modelled required revenue in every ownership case.",
        }},
    "state_overhead": {
        "group": "payment", "question": "What does it cost to run the public body that owns the assets?",
        "mechanism": "A per-kW charge on the state's share of owned generation and storage is added to required revenue, so it reduces the financing saving and, where the at-cost contract carries it, reaches the bill.",
        "held_fixed": "Plant-level operating costs are unchanged: the published fixed O&M is site-level for every owner, so this is the holding company on top of it, not a re-count of it.",
        "caution": "This is an assumption with a stated basis, not a measurement. No published source separates a generator's group overhead from its site costs, so the twins are deliberately wide. It is charged to the state and not to the private comparator.",
        "value_explanations": {
            "central": "3 GBP/kW/yr, where a bottom-up build from BVG Associates' disclosed onshore admin line over the mix the state actually owns (2.7) and a top-down scaling of Great British Energy's first accounts (2.2 to 4.3) both land.",
            "low": "1 GBP/kW/yr: strong scale economies, and most administration already inside the published fixed O&M.",
            "high": "8 GBP/kW/yr: BVG's offshore onshore-admin rate applied across the whole owned fleet.",
        }},
    "ldes_floor_basis": {
        "group": "payment", "question": "What return is protected by the long-duration electricity storage cap-and-floor?",
        "mechanism": "The floor top-up is the gap between protected required revenue and market revenue; a technology WACC generally creates a larger top-up than a debt-like floor.",
        "held_fixed": "Storage capacity, operation, efficiency and market revenue are unchanged.",
        "caution": "This policy line exists in the private case as well. Comparisons use a private baseline with the same floor basis.",
        "value_explanations": {
            "cost_of_debt_like": "Central: protect a debt-like real return, the workbook's BBB-minus debt (5.67%) private, two notches better (5.09%) under public control and the public rate under ownership.",
            "technology_wacc": "Protect the full technology WACC, increasing the revenue floor and likely the consumer top-up.",
        }},
    "wholesale_pricing": {
        "group": "institution", "question": "What is the maximum payment that a mechanical pay-as-bid calculation could remove?",
        "mechanism": "The bound replaces accepted non-CfD GB generators' wholesale revenue above short-run cost with cost, holding offers and dispatch fixed.",
        "held_fixed": "There is no strategic rebidding, investment response, replacement capacity payment or new dispatch solve.",
        "caution": "This is explicitly a ceiling, never a policy forecast or headline scenario. Real pay-as-bid offers would adapt and CfD plant is excluded because lower wholesale revenue would be offset by higher CfD top-ups.",
        "value_explanations": {
            "pay_as_clear": "Central: every accepted generator receives the market-clearing price.",
            "pay_as_bid_ceiling": "Bounding calculation: remove non-CfD inframarginal wholesale revenue above short-run cost with all behaviour frozen.",
        }},
    "dispatch_institution": {
        "group": "institution", "question": "Is the fully acquired system dispatched through the market or centrally at resource cost?",
        "mechanism": "The central-dispatch case substitutes four measured lines from the separately solved nationalised-dispatch run into the acquisition tariff, and adds FTI's forecast-error cost, borrowed and scaled, because the model produces almost none of its own.",
        "held_fixed": "It is only defined when every generation and storage asset is acquired; acquisition terms otherwise follow the fully nationalised case.",
        "caution": "This is the only case using a second physical solve. It should not be attributed to ownership alone because it combines ownership with a different operating institution.",
        "value_explanations": {
            "market": "Keep the market dispatch and settlement machinery used by the common CP30 run.",
            "nationalised_dispatch": "Use the separately solved central-dispatch institution at resource cost for the fully acquired system.",
        }},
    # Item BM1.
    "balancing_bids": {
        "group": "institution", "question": "Does the state's own plant bid at cost when the balancing mechanism turns it down?",
        "mechanism": "The balancing mechanism is re-solved with the state-owned share of each CfD unit bidding its avoided short-run cost instead of its contract value, and the bill reads that second run.",
        "held_fixed": "The wholesale market, the network, who owns what and every financing lever are those of the public developer case; only the balancing bids change.",
        "caution": "Part of the fall in accepted bids is the state paying itself, and part comes back as CfD payments on the extra private output. The network, not the bids, sets how much is turned down.",
        "value_explanations": {
            "contract_value": "Every CfD unit bids what it would lose on its contract, whoever owns it.",
            "state_at_cost": "The state's share of each unit bids at cost, so its plant is turned down first.",
        }},
    # ── Review 29 Sep 2026: assumption toggles, incidence levers and background switches ──
    "state_stake_scale": {
        "group": "ownership", "question": "How large is the state's stake in new build?",
        "mechanism": "The stake sets the share of each new asset financed at the public rate; the rest stays at the private rate.",
        "held_fixed": "The plant built, its output and every price are unchanged.",
        "caution": "The funded scale is what GB Energy and the National Wealth Fund can capitalise, not a policy ceiling.",
        "value_explanations": {
            "workbook_stake": "The workbook's 30% stake.",
            "funded_scale": "A 15% stake, the scale the announced public capital can fund.",
        }},
    "new_build_delivered_share": {
        "group": "ownership", "question": "How much of the post-2025 build does the state developer deliver?",
        "mechanism": "Half of each technology's new-build cohort by capacity stays private at the private rate; the state finances the rest.",
        "held_fixed": "The fleet and its output are the same in every case; only the owner of each asset moves.",
        "caution": "A delivery shortfall halves the financing saving; it does not remove the plant from the system.",
        "value_explanations": {
            "all": "The state delivers the whole post-2025 cohort.",
            "half": "The state delivers half of each technology's cohort; the rest is built privately.",
        }},
    "state_capex_overrun": {
        "group": "capital", "question": "Does the state build as cheaply as a private developer?",
        "mechanism": "The state's share of new-build capital cost is raised, so its capital charge rises with it.",
        "held_fixed": "Private developers' costs, the fleet and its output are unchanged.",
        "caution": "An assumption, not a measured overrun: public projects are not systematically dearer, but the risk is real.",
        "value_explanations": {
            "none": "The state builds at the same cost as a private developer.",
            "plus_15pct": "The state's new build costs 15% more.",
        }},
    "state_om_premium": {
        "group": "capital", "question": "Does the state run its plant as cheaply as a private owner?",
        "mechanism": "Fixed operating costs on the state's share rise by the premium.",
        "held_fixed": "Output, prices and private operating costs are unchanged.",
        "caution": "An assumption with no single published source.",
        "value_explanations": {
            "same": "The same operating cost as a private owner.",
            "plus_10pct": "Fixed operating costs 10% higher on the state's share.",
        }},
    "network_acquisition_premium": {
        "group": "acquisition", "question": "What premium over the regulated asset value is paid for the wires?",
        "mechanism": "The premium is annuitised at the public rate over 30 years and paid out of the network saving.",
        "held_fixed": "The regulated asset value, the allowed return and the public rate are unchanged.",
        "caution": "Recent control sales went for 1.4-1.7 times RAV; a price set by Parliament at RAV is a legal choice, not a market one.",
        "value_explanations": {
            "central": "50% over RAV, the median of recent full-control sales.",
            "low": "20% over RAV, what the listed market pays for a minority stake.",
            "high": "75% over RAV, the upper end of recent sales.",
            "statutory_at_rav": "Compensation set by Parliament at RAV.",
            "quarter": "25% over RAV, a sensitivity between the listed-market price and recent control sales.",
        }},
    "generation_control_premium": {
        "group": "acquisition", "question": "Does buying the legacy fleet cost a control premium?",
        "mechanism": "The purchase price of the legacy fleet rises by the premium, and the annuity with it.",
        "held_fixed": "The fleet's cash flows and output are unchanged.",
        "caution": "Takeover premiums vary widely; 20% is a round figure.",
        "value_explanations": {
            "none": "Bought at the valuation.",
            "twenty_pct": "A 20% control premium on the purchase price.",
        }},
    "interconnector_price": {
        "group": "acquisition", "question": "What is paid for the GB halves of the interconnectors?",
        "mechanism": "Their congestion rent after corporation tax is valued at an owner's discount rate; a higher rate gives a lower price.",
        "held_fixed": "The cables, their flows and their rent are unchanged.",
        "caution": "The rent depends on the foreign price assumption. The range is a declared one point either side of the workbook's rate.",
        "value_explanations": {
            "central": "Valued at 4.87%, the workbook's post-tax operating rate for interconnectors.",
            "low": "Valued at 5.87% (a lower price).",
            "high": "Valued at 3.87% (a higher price).",
        }},
    "acquisition_transaction_costs": {
        "group": "acquisition", "question": "Are the costs of the transaction counted?",
        "mechanism": "Advisers, legal and stamp costs are added as a share of everything bought and annuitised with the purchase.",
        "held_fixed": "The purchase prices themselves are unchanged.",
        "caution": "Two per cent is a round figure for large public acquisitions.",
        "value_explanations": {
            "none": "No transaction costs.",
            "two_pct": "Two per cent of everything bought.",
        }},
    "legacy_asset_lives": {
        "group": "acquisition", "question": "How long does the legacy fleet run?",
        "mechanism": "Longer lives lengthen the cash flows the purchase price is built on.",
        "held_fixed": "2030 output and costs are unchanged; only the valuation's horizon moves.",
        "caution": "At a market-value price, longer lives also raise what the owners are paid.",
        "value_explanations": {
            "central": "Onshore wind 25 years (30% repowered), offshore 30, solar 38; Sizewell B to 2055.",
            "extended": "Five more years for wind, solar and marine.",
        }},
    "legacy_price_path": {
        "group": "acquisition", "question": "What price does the existing fleet earn after 2030 in its valuation?",
        "mechanism": "A higher price raises merchant earnings after subsidies end, and so what the owners would sell for.",
        "held_fixed": "2030 output, costs and the 2030 bill's own prices are unchanged; only the purchase price moves.",
        "caution": "Since 10 November 2026 the central runs each technology's price through today's forward level (Ofgem's 2026 index), the model's 2030 and 2050 (and 2040 once solved), flat after 2050; the model's 2030 price held flat and CP30's 2030 level are the alternatives.",
        "value_explanations": {
            "anchored": "Central: each technology's capture price on a straight line through today, 2030 and 2050, flat after.",
            "model_2030": "The model's 2030 price held flat.",
            "cp30_level": "Each hour mapped onto CP30's own 2030 price duration curve (CP.32).",
            "today_level": "Today's forward prices (Ofgem's 2026 wholesale index on the 2030 capture ratios) held for the fleet's life.",
        }},
    "state_deficit_funding": {
        "group": "payment", "question": "Who pays when the state's plant earns less than it costs?",
        "mechanism": "The deficit is carried by the Exchequer or levied on bills.",
        "held_fixed": "The size of the deficit is unchanged; only who pays it moves.",
        "caution": "Moving a deficit onto bills raises the bill without changing what the system costs.",
        "value_explanations": {
            "exchequer": "The Exchequer carries it.",
            "bill_levy": "It is levied on bills.",
        }},
    "tax_equivalent": {
        "group": "payment", "question": "Does the state's plant pay the tax a private owner would?",
        "mechanism": "A tax equivalent is charged in the state's required revenue and paid to the Exchequer.",
        "held_fixed": "The plant, its financing and its output are unchanged.",
        "caution": "It is a transfer from bill payers to taxpayers, not a cost to the country.",
        "value_explanations": {
            "none": "No tax equivalent.",
            "on": "Corporation tax equivalent paid to the Exchequer.",
        }},
    "network_incidence": {
        "group": "payment", "question": "Who gets the network saving?",
        "mechanism": "The saving on the bought wires goes to bills, or the owner's return is kept by the Exchequer.",
        "held_fixed": "The size of the network saving is unchanged.",
        "caution": "A retained return is a transfer to taxpayers, not a lost saving.",
        "value_explanations": {
            "bills": "The whole saving reaches bills.",
            "return_retained": "The Exchequer keeps the owner's return, including the tax allowance.",
        }},
    "acquisition_financing": {
        "group": "payment", "question": "Who pays the annuity on the purchase?",
        "mechanism": "The annuity is recovered through the tariff or funded from general taxation.",
        "held_fixed": "The purchase price and its term are unchanged.",
        "caution": "Funding it from taxation lowers the bill by moving the cost to taxpayers.",
        "value_explanations": {
            "tariff": "The tariff recovers it.",
            "exchequer": "General taxation funds it.",
        }},
    "tariff_pricing": {
        "group": "payment", "question": "Does the public power system charge its cost or the market price?",
        "mechanism": "The state charges households what the private market bill would be; the margin between that and the tariff's cost goes to the Treasury.",
        "held_fixed": "What the fleet costs, who is paid for it and the purchase are unchanged: only who keeps the margin moves.",
        "caution": "The state charges market prices and the Treasury keeps the margin: the bill is about the private bill, and the saving appears in the Exchequer instead.",
        "value_explanations": {
            "cost": "The tariff recovers the cost of the system.",
            "market_bill": "The bill is the private comparator's; the Treasury keeps the margin over cost.",
        }},
    "legacy_levy_funding": {
        "group": "payment", "question": "Are the legacy policy levies paid from bills or from taxation?",
        "mechanism": "The domestic share of the Renewables Obligation, Feed-in Tariff and Warm Home Discount moves to the Exchequer, in every case including private.",
        "held_fixed": "The schemes and their total cost are unchanged.",
        "caution": "A background switch: it moves every case's bill by the same amount, so it does not change the ranking.",
        "value_explanations": {
            "bills": "Paid on bills, as today.",
            "domestic_to_taxation": "The domestic share paid from taxation.",
        }},
    "beccs_removal_funding": {
        "group": "payment", "question": "Who pays for the carbon capture at the BECCS plant?",
        "mechanism": "The carbon-removal contract pays for the capture; it is funded by the Exchequer or levied on bills.",
        "held_fixed": "The plant, its output and its electricity contract are unchanged.",
        "caution": "A background switch: it moves every case by the same amount.",
        "value_explanations": {
            "exchequer": "The Exchequer funds the carbon-removal contract.",
            "bill_levy": "The carbon-removal contract is levied on bills.",
        }},
    "auction_margin": {
        "group": "cost",
        "question": "Do the auction-calibrated CfD projects keep a margin above their full cost?",
        "mechanism": "CfD auctions pay as clear, so projects cheaper than the marginal one keep a margin above their required return. The calibration scales the cohort's capex so it keeps this share of contract revenue above full cost; the state's own plant needs no margin.",
        "held_fixed": "Strikes, dispatch and the private bill are unchanged; only the plant's capital cost, and so the state's cost of building it, moves.",
        "caution": "Individual bids are not published, so the size is a declared assumption; the direction is not. It moves every case, private included, so it is measured against private at the same margin.",
        "value_explanations": {"central": "5% of contract revenue.", "none": "The cohort just breaks even at its strike (the calibration as first decided).", "high": "10% of contract revenue."},
    },
    "self_insurance": {
        "group": "cost",
        "question": "Does plant the state owns outright buy commercial insurance?",
        "mechanism": "Central government carries its own risks rather than insuring commercially, so wholly state-owned plant saves the insurer's loading on its premium (its costs and margin); it still bears the expected claims.",
        "held_fixed": "Private plant, and plant under a minority state stake, keep their commercial insurance.",
        "caution": "The loading share is a declared assumption (Lloyd's 2025 expense ratio is 35.6%). The state then carries concentrated risks itself, such as a major offshore failure. Managing Public Money does not bind a public corporation to self-insure (A7.3.13), and a body that charges for its output prices a notional premium into its charges (A4.4.4); the 'none' value is the public-corporation reading.",
        "value_explanations": {"central": "35% of the premium saved.", "none": "Commercial insurance, as a private owner.", "high": "50% of the premium saved."},
    },
    # F27 item 6 (user, 20 and 24 Oct 2026).
    "capacity_procurement": {
        "group": "payment",
        "question": "Is existing capacity needed for security paid the Capacity Market price, or its cost?",
        "mechanism": "Each existing unit holding a Capacity Market agreement is paid a cost-based contract instead: its going-forward fixed cost (fixed O&M, network charges and insurance) less its own wholesale and balancing margin, floored at zero and never more than it cleared at. New capacity is still auctioned or built by the state.",
        "held_fixed": "The plant, its output, its costs and the auctions for new capacity are unchanged; only what existing plant is paid for being there moves.",
        "caution": "A capacity-procurement choice ownership makes easier, not a saving to the country: under P2 a transfer from the private owners of existing plant to bill payers; under P3 a lower purchase price, and only if the regime is announced before the purchase. Existing plant could close or leave the market rather than accept it, which the model does not test.",
        "value_explanations": {"auction": "The Capacity Market clearing price, as today.", "cost_based_existing": "Existing plant paid its going-forward cost less its own margin."},
    },
    # User decision (24 Oct 2026).
    "legacy_offshore_price": {
        "group": "acquisition",
        "question": "Is existing offshore wind bought at the model's value of its cash flows, or at what recent deals paid?",
        "mechanism": "Each existing offshore wind farm is priced at the median of recent UK operating offshore wind stake sales per megawatt of the whole farm (enterprise value, 2024 money) instead of its owners' value from the model's cash flows; every other plant stays at its owners' value.",
        "held_fixed": "The plant, its output, its costs, the tariff's other lines and dispatch are unchanged; only the purchase price, and so its annuity, moves.",
        "caution": "Deal prices reflect the buyers' own price forecasts, lives, financing and contract terms (several farms sold carry rich early CfDs or ROCs), and minority stakes; the model's value is about half of them because its offshore income after the contracts is low. The range of deals is wide (about £4.0m to £8.0m per MW).",
        "value_explanations": {"owners_value": "The model's own value of the farms' cash flows at the owners' rate.", "transaction_multiple": "The median price per MW of recent UK operating offshore wind stake sales."},
    },
}

NAMED_GUIDES = {
    # Review 29 Sep 2026.
    "fully_nationalised_market_value_prices_held_at_2030": {
        "change": "Each plant's market income is held at the model's 2030 prices for its whole life.",
        "mechanism": "The central runs each technology's price through today's level, 2030 and 2050; holding 2030 flat keeps wind and solar at their higher 2030 prices.",
        "held_fixed": "The fleet, its contracts and every other acquisition term are as in the public power system.",
        "caution": "This was the central before 10 November 2026."
    },
    "fully_nationalised_market_value_efw_acquired": {
        "change": "The purchase buys the energy-from-waste plants too, at their owners' value.",
        "mechanism": "Their value is mostly the gate fees councils pay; bought, those fees are credited to the tariff.",
        "held_fixed": "The fleet, the other acquisition terms and every other assumption are as in the public power system.",
        "caution": "Most plants are owned by waste companies under long council contracts, so buying them moves money mostly within the public sector."
    },
    "fully_nationalised_market_value_hpc_left_private": {
        "change": "Hinkley Point C is left out of the purchase and stays private on its own CfD.",
        "mechanism": "The purchase price falls by Hinkley's value and its CfD stays on the bill as a private contract.",
        "held_fixed": "The plant, its output and every other acquisition term are as in the public power system.",
        "caution": "The central buys it as an operating plant in 2030; the state completing it is a separate case."
    },
    "fully_nationalised_market_value_systematic_premium": {
        "change": "The risk taxpayers take on with the state's plant is priced at the workbook's systemic risk premium, a flat 1.9% a year on top of the gilt-based cash rate.",
        "mechanism": "Bills still pay the cash rate; the premium is a cost to the Exchequer and to the economy-wide saving, never to the bill.",
        "held_fixed": "The fleet, the purchase and every other assumption are as in the public power system.",
        "caution": "The 1.9% is the Green Book discount rate review's estimate of the social cost of public risk; the central case (gilts alone) follows the current Green Book."
    },
    "fully_nationalised_market_value_systematic_premium_1pc": {
        "change": "The risk taxpayers take on with the state's plant is priced at a flat 1% a year on top of the gilt-based cash rate.",
        "mechanism": "Bills still pay the cash rate; the premium is a cost to the Exchequer and to the economy-wide saving, never to the bill.",
        "held_fixed": "The fleet, the purchase and every other assumption are as in the public power system.",
        "caution": "1% is the rate HM Treasury adopted in its 2026 response to the Green Book discount rate review (the reviewers estimated 1.9%); the central case (gilts alone) follows the current Green Book."
    },
    "systematic_premium_1pc": {
        "change": "The risk taxpayers take on with the plant the state builds is priced at a flat 1% a year on top of the gilt-based cash rate.",
        "mechanism": "Bills still pay the cash rate; the premium is a cost to the Exchequer and to the economy-wide saving, never to the bill.",
        "held_fixed": "The new build, its delivery and every other assumption are as in public ownership.",
        "caution": "1% is the rate HM Treasury adopted in its 2026 response to the Green Book discount rate review; the central case (gilts alone) follows the current Green Book."
    },
    "systematic_premium_1pc9": {
        "change": "The risk taxpayers take on with the plant the state builds is priced at the workbook's 1.9% a year on top of the gilt-based cash rate.",
        "mechanism": "Bills still pay the cash rate; the premium is a cost to the Exchequer and to the economy-wide saving, never to the bill.",
        "held_fixed": "The new build, its delivery and every other assumption are as in public ownership.",
        "caution": "1.9% is the Green Book discount rate review's estimate of the social cost of public risk (HM Treasury adopted 1%); the central case (gilts alone) follows the current Green Book."
    },
    "fully_nationalised_market_value_yield_b": {
        "change": "Gilt yields rise with the extra state debt and contingent exposure, 2 basis points per point of GDP (the workbook's yield path B).",
        "mechanism": "Every public borrowing rate rises with the gilt yield, so the state's cash rate and the bill rise with it.",
        "held_fixed": "The fleet, the purchase and every other assumption are as in the public power system.",
        "caution": "The response is a literature estimate (Poghosyan 2014); a gilt path that follows CPI (assumption C) is pending the workbook's input-output model."
    },
    "fully_nationalised_market_value_hpc_completion": {
        "change": "Hinkley Point C is bought at what has been spent on it and finished by the state, financed at the public cash rate.",
        "mechanism": "The purchase price becomes the sunk cost plus the remaining capital at the public rate, instead of the value of its contract to the owners.",
        "held_fixed": "The plant, its output and every other acquisition term are as in the public power system.",
        "caution": "On EDF's own cost figures this is dearer than buying it operating, because the contract value is below the cost."
    },
    "fully_nationalised_market_value_legacy_levies_to_taxation": {
        "change": "The domestic share of the Renewables Obligation, Feed-in Tariff and Warm Home Discount paid from taxation.",
        "mechanism": "The levies leave the bill and the Exchequer carries them; the private comparator moves the same way.",
        "held_fixed": "The schemes, their total cost and the public power system are unchanged.",
        "caution": "A transfer from taxpayers to bill payers, not a saving."
    },
    "networks_acquired_rav_at_statute": {
        "change": "The wires are bought at their regulated asset value, the compensation Parliament would set by statute, instead of at the 50% premium recent sales fetched.",
        "mechanism": "With no premium to annuitise, the whole financing wedge and the tax allowance reach the network lines of the bill.",
        "held_fixed": "Everything else is the State developer and the wires (P2b).",
        "caution": "A statutory price below market value is a legal and political choice, and would be contested; the break-even premium is reported beside every networks-acquired case."
    },
    "fully_nationalised_market_value_rav_at_statute": {
        "change": "The public power system with the wires bought as well, at their regulated asset value by statute.",
        "mechanism": "With no premium to annuitise, the whole financing wedge and the tax allowance reach the network lines of the tariff.",
        "held_fixed": "The legacy fleet's market-value purchase and every other term of the public power system.",
        "caution": "A statutory price below market value is a legal and political choice, and would be contested."
    },
    "fully_nationalised_market_value_beccs_removal_on_bills": {
        "change": "The BECCS carbon-removal contract is levied on bills.",
        "mechanism": "The capture leg is paid by bill payers instead of the Exchequer; the private comparator moves the same way.",
        "held_fixed": "The plant, its output and its electricity contract are unchanged.",
        "caution": "A transfer from bill payers to taxpayers, not a cost to the country."
    },
    # Item G3, lever L19.
    "legacy_acquired_contracts_cancelled_on_bills": {
        "change": "Renewables and nuclear bought, with their CfD and Renewables Obligation contracts cancelled rather than honoured; the former owners are paid the contracts’ value at their own cost of capital as a tariff line.",
        "mechanism": "The purchase price drops the contract income and a residual line pays for it instead. Because owners discount at a higher rate than the state, the residual is smaller than what honouring would cost, and that difference is the whole saving.",
        "held_fixed": "The plant, its output, the purchase basis and term, the networks and dispatch are as in the legacy-acquired case.",
        "caution": "The saving is the rate differential on the contract income, not the income itself; at the market-value purchase basis it would be zero."
    },
    "legacy_acquired_contracts_cancelled_on_exchequer": {
        "change": "Renewables and nuclear bought, with their CfD and Renewables Obligation contracts cancelled rather than honoured; the former owners are paid the contracts’ value at their own cost of capital by the Exchequer.",
        "mechanism": "The purchase price drops the contract income and a residual line pays for it instead. Because owners discount at a higher rate than the state, the residual is smaller than what honouring would cost, and that difference is the whole saving.",
        "held_fixed": "The plant, its output, the purchase basis and term, the networks and dispatch are as in the legacy-acquired case.",
        "caution": "The bill falls by the whole residual because taxpayers pay it: a transfer from taxpayers to bill payers, not a saving to the country."
    },
    "legacy_and_thermal_acquired_contracts_cancelled_on_bills": {
        "change": "The whole legacy fleet bought, with the renewables’ and nuclear CfD and Renewables Obligation contracts cancelled rather than honoured; the former owners are paid the contracts’ value at their own cost of capital as a tariff line.",
        "mechanism": "The purchase price drops the contract income and a residual line pays for it instead. Because owners discount at a higher rate than the state, the residual is smaller than what honouring would cost, and that difference is the whole saving.",
        "held_fixed": "The plant, its output, the purchase basis and term, the networks and dispatch are as in the case that buys the whole legacy fleet.",
        "caution": "The saving is the rate differential on the contract income, not the income itself; thermal plant holds no such contracts."
    },
    "legacy_and_thermal_acquired_contracts_cancelled_on_exchequer": {
        "change": "The whole legacy fleet bought, with the renewables’ and nuclear CfD and Renewables Obligation contracts cancelled rather than honoured; the former owners are paid the contracts’ value at their own cost of capital by the Exchequer.",
        "mechanism": "The purchase price drops the contract income and a residual line pays for it instead. Because owners discount at a higher rate than the state, the residual is smaller than what honouring would cost, and that difference is the whole saving.",
        "held_fixed": "The plant, its output, the purchase basis and term, the networks and dispatch are as in the case that buys the whole legacy fleet.",
        "caution": "The bill falls by the whole residual because taxpayers pay it: a transfer, not a saving to the country."
    },
    "fully_nationalised": {
        "change": "All GB generation and storage are acquired, pre-2025 CfDs are publicly owned and re-priced, and Hinkley Point C is bought as an operating plant at its contract value. The wires stay private; buying them is the networks choice.",
        "mechanism": "The market bill is replaced by an acquisition tariff covering resource costs, public capital recovery and the purchase annuity.",
        "held_fixed": "Market dispatch remains the common CP30 run; central dispatch is a separate named case.",
        "caution": "Always read the annual bill together with the displayed acquisition price and the valuation basis."
    },
    "fully_nationalised_market_value": {
        "change": "All generation and storage in public hands, with the valued legacy group bought at owners' market value rather than at its value to a low-cost public owner. This is the policy proposition; the state-value case beside it is the alternative price. The wires stay private unless the networks choice adds them.",
        "mechanism": "The changed purchase price alters the annuity in the acquisition tariff; operations and annual resource costs do not move.",
        "held_fixed": "Every other full-nationalisation lever, including the 30-year term, stays fixed.",
        "caution": "Market value is modelled by the workbook's Approach 2 (after-tax cash flows and capital allowances at post-tax operating rates); it is not an observed negotiated price, and no premium over it is modelled. Note that it comes out CHEAPER than the state-value basis, because it discounts at the owner's higher required return. The contract-design levers do not appear here: once the state owns the counterparty there is no strike left to re-price."
    },
    "fully_nationalised_book_value": {
        "change": "The fully nationalised package buys every acquired plant at its net book value: the workbook's historic capex by cohort written down over its accounting life (its Approach 1).",
        "mechanism": "Book value replaces the value of the cash flows in the acquisition annuity.",
        "held_fixed": "Asset cash flows, dispatch, network treatment and the 30-year term are unchanged.",
        "caution": "Historic capex is a vintage-based proxy, not company accounts; the workbook presents book value as a sensitivity bounding the compensation question."
    },
    "fully_nationalised_no_support_value": {
        "change": "The fully nationalised package buys every acquired plant at its owners' value without its CfD and Renewables Obligation income.",
        "mechanism": "The support streams leave the price but stay in the tariff's revenue, so the purchase annuity falls by their after-tax value to the owners.",
        "held_fixed": "Asset cash flows, dispatch, network treatment and the 30-year term are unchanged.",
        "caution": "The Northern Rock principle (value without the support government put in): a legal lower bound, contested for contracts the state signed."
    },
    "fully_nationalised_ro_pass_through": {
        "change": "RO plant is bought at its no-support value and everything else at owners' value.",
        "mechanism": "The Renewables Obligation income stays with the tariff while its value leaves the price.",
        "held_fixed": "Asset cash flows, dispatch, network treatment and the 30-year term are unchanged.",
        "caution": "A policy case that embeds a transfer from RO shareholders; not comparable with market value."
    },
    "fully_nationalised_thermal_at_zero": {
        "change": "The fully nationalised package assigns a zero purchase price to legacy thermal generation and storage instead of their owners' value.",
        "mechanism": "Removing that part of the purchase price lowers the acquisition annuity.",
        "held_fixed": "The valued renewable and nuclear group, networks, operating costs and dispatch are unchanged.",
        "caution": "This is an extreme valuation sensitivity and should not be read as a legally costless transfer."
    },
    "fully_nationalised_market_value_price_path_cp30": {
        "change": "The existing fleet is valued on CP30's 2030 price level instead of the model's own 2030 price held flat.",
        "mechanism": "Higher post-subsidy earnings raise the owners' value and so the purchase annuity.",
        "held_fixed": "The 2030 bill's prices, output and costs are unchanged; only the purchase price moves.",
        "caution": "CP30's level sits above the model's own 2030 price; the workbook asks for the model's price."
    },
    # User (11 Nov 2026).
    "fully_nationalised_market_value_price_path_today": {
        "change": "The existing fleet is valued as if today's forward wholesale prices (Ofgem's 2026 index, about GBP 95/MWh) lasted its whole life, instead of the central path that falls to the model's 2030 and 2050 prices.",
        "mechanism": "Higher expected earnings raise the owners' value and so the purchase annuity; CfD top-ups fall against the higher reference price.",
        "held_fixed": "The bill year's prices, output and costs are unchanged; only the purchase price moves.",
        "caution": "An upper view of what owners might ask: today's prices are set by gas, which the modelled years burn far less of."
    },
    "fully_nationalised_market_value_capacity_cost_based": {
        "change": "Existing plant is valued on cost-based capacity contracts (going-forward fixed cost less its own margin, floored at zero) instead of its Capacity Market income.",
        "mechanism": "The owners lose the Capacity Market income above the plant's missing money, so their value, and the purchase annuity, falls.",
        "held_fixed": "The tariff already pays no Capacity Market agreement to plant the state owns; output, costs and dispatch are unchanged.",
        "caution": "A lower price only if the contract regime is announced before the purchase; announced after, the owners are paid for the income they had."
    },
    # User decision (24 Oct 2026).
    "fully_nationalised_market_value_offshore_deal_multiples": {
        "change": "Existing offshore wind is bought at the median price per MW of recent UK operating stake sales instead of the model's own value of its cash flows.",
        "mechanism": "A higher price for the 14.7 GW of existing offshore wind raises the purchase annuity on the tariff.",
        "held_fixed": "Every other plant stays at its owners' value; the cash flows, output, costs and dispatch are unchanged.",
        "caution": "Deal prices embed the buyers' price forecasts, lives and financing; the gap to the model's value is mostly the model's low offshore income after the contracts end."
    },
    "fully_nationalised_term_15y": {
        "change": "The same acquisition price is repaid over 15 years rather than 30.",
        "mechanism": "Faster principal recovery raises the annual acquisition charge during the repayment window.",
        "held_fixed": "Purchase valuation, public rate, asset scope, operations and dispatch are unchanged.",
        "caution": "The higher near-term bill is a timing effect; it does not mean the assets cost more in total."
    },
    "fully_nationalised_market_value_central_dispatch": {
        "change": "The public power system as proposed: all generation and storage in public hands, the existing fleet bought at market value, and a nationalised operator that dispatches centrally at resource cost instead of running the wholesale market and the balancing mechanism.",
        "mechanism": "The acquisition tariff of the market-value case, with four lines measured from the separately solved central-dispatch year: fuel and carbon, variable O&M, cross-border trade, and the real-time correction of the day-ahead schedule in place of the market's interconnector re-trading; plus FTI's forecast-error cost, borrowed and scaled, since the model's own is near zero.",
        "held_fixed": "The acquisition scope, the market-value purchase, the 30-year term, the networks and every financing assumption match the market-kept case; only the operating institution changes.",
        "caution": "This bundles ownership with a different operating institution and is evaluated on a second solved year. The constraint payments central dispatch removes are the same money ownership removes, so the two savings are substitutes and are never added."
    },
    "gbe_value_to_bills": {
        "change": "The state's position in the post-2025 build is routed to bills through a cost-based CfD instead of being retained by the Exchequer, and by the same coherence rule a shortfall is levied on eligible supply rather than carried by the Exchequer.",
        "mechanism": "Nothing about the assets, the stake or its financing changes; only the destination of the state's net position moves. At a 30% stake on the CEPA decomposition that position is a DEFICIT, so routing it to bills RAISES the net bill.",
        "held_fixed": "The 30% stake, the private wires, the booked Capacity Market and CCS routes, the contracts and the physical run all match the scenario.",
        "caution": "The pair with the scenario is the point: it prices the recycling decision, and the answer here is that recycling costs bill-payers money because the CP30 2030 market does not support new build at commercial returns. Read the gross and the net together."
    },
    "fully_nationalised_market_value_term_15y": {
        "change": "The public power system with its market-value purchase repaid over 15 years rather than asset by asset over each asset's remaining life.",
        "mechanism": "Faster principal recovery raises the annual acquisition charge inside the tariff during the repayment window.",
        "held_fixed": "Purchase valuation, asset scope, networks, public rate, operations and dispatch are unchanged.",
        "caution": "The higher near-term bill is a timing effect, not a higher total cost of the assets."
    },
    "fully_nationalised_market_value_term_30y": {
        "change": "The public power system with its whole market-value purchase repaid as one 30-year annuity rather than asset by asset over each asset's remaining life (F28 (e)).",
        "mechanism": "One even annuity spreads the repayment later than the per-asset schedule, so the 2030s carry less of the acquisition charge inside the tariff.",
        "held_fixed": "Purchase valuation, asset scope, networks, public rate, operations and dispatch are unchanged.",
        "caution": "The lower near-term bill is a timing effect: the lifetime present value at the public rate is the same."
    },
    "conservative_stake": {
        "change": "Public control is taken back to the reading it had before the policy propositions: no guarantee reaching the wires, the Capacity Market and CCS financing savings reported beside the bill rather than booked on it, and the state's position returned to bills through the at-cost CfD instead of retained by the Exchequer.",
        "mechanism": "Four levers move together, so this is the scenario's own earlier bundle rather than a one-at-a-time sensitivity. The stake, its financing and the physical run are untouched.",
        "held_fixed": "The 30% stake, the CEPA decomposition, the legacy fleet, the contracts and the CP30 physical background are exactly as in the scenario.",
        "caution": "This is the conservative floor, not a policy design: it is conservative on every lever at once, which is why the propositions replaced it as the headline."
    },
    "conservative_developer": {
        "change": "Public ownership is taken back to the reading it had before the policy propositions: owned contracts keep the auction-cleared strike reduced only by the financing saving, amortised over the CfD contract term, and the Capacity Market and CCS savings are reported rather than booked.",
        "mechanism": "Three levers move together. The state still owns the whole post-2025 cohort at the public rate; only the contract form and the route booking change.",
        "held_fixed": "Ownership, the public borrowing rate, the construction premium, the legacy fleet, the wires and the physical run are exactly as in the scenario.",
        "caution": "It holds a state-owned asset to a contract nobody would write for one, which is the reason the proposition was promoted over it. Keep it as the floor, not the answer."
    },
    "strike_at_cost_over_life": {"change": "Under the state stake, contracts are cut to cost, with the capital costed over the plant's economic life rather than the 20-year contract term.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "stake_networks_acquired_rav_premium_25pct": {"change": "The state stake with the wires bought at their regulated asset value plus a 25% premium.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "networks_acquired_rav_premium_25pct": {"change": "P2 with the wires bought at their regulated asset value plus a 25% premium.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "fully_nationalised_market_value_rav_premium_25pct": {"change": "P3 with the wires bought at their regulated asset value plus a 25% premium.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "gbe_announced": {"change": "The state holds what GB Energy has announced instead of the whole post-2025 cohort: a 30% stake in post-2025 offshore wind through the Crown Estate partnership (about 5.6 GW) and 8 GW of local solar and onshore wind (the Local Power Plan).", "mechanism": "The offshore stake is financed and contracted as P1's stake (the financing saving over the CfD term); the local power is owned at the public rate and contracted at cost over its life, as P2's. Storage and everything else stay private.", "held_fixed": "P2's financing and incidence levers; the physical run.", "caution": "A realism check on P2's scale, not a forecast of GB Energy's portfolio: the 8 GW is funded rather than owned by GB Energy, often in partnership."},
    "gbe_announced_partnership_half": {"change": "As GB Energy as announced, with the 8 GW of local power held 50% public.", "mechanism": "The partner's half of each local row is financed at the private rate (co-ownership, as the 2050 deployment shares).", "held_fixed": "Everything else is GB Energy as announced.", "caution": "A sensitivity, not a recommendation."},
    "developer_storage_private": {"change": "P2 with new-build storage left to private owners.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "recycling_cost_recovery_contracting_on_developer": {"change": "P2 returns its surplus through cost-recovery contracting.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "recycling_rebate_per_mwh_on_developer": {"change": "P2 rebates its surplus over eligible supply.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "recycling_offset_policy_levies_on_developer": {"change": "P2 uses its surplus to offset domestic customers' legacy policy levies.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "recycling_no_assumption_on_developer": {"change": "P2 makes no assumption about where its surplus goes.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "networks_guaranteed_passthrough_low": {"change": "The state guarantees the network companies' debt, with the low pass-through of the cheaper debt to the allowed return.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "networks_guaranteed_fee_on_bills": {"change": "The debt guarantee's fee is levied on bills.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "networks_acquired_rav_premium_low": {"change": "P2 with the wires bought at the low premium over their regulated asset value.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
    "networks_acquired_return_retained": {"change": "P2 with the wires bought, the network saving retained by the state rather than passed to bills.", "mechanism": "One named position on the lever catalogue.", "held_fixed": "Everything else is its parent scenario.", "caution": "A sensitivity, not a recommendation."},
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fill_bm1(text: str) -> None:
    """Put the measured BM1 bid-stack effect (the P2 increment) into the scenario guides."""
    for scenario in SCENARIOS.values():
        guide = scenario["guide"]
        for key, value in guide.items():
            guide[key] = value.replace("__BM1_BID_STACK__", text)


#: Item BM1, P3: the tariff balancing group P3 is priced on.
P3_GROUP = "cp30_2030_cp30_network_public_bids_p3"


def build(out_path: Path) -> dict:
    defaults = load_defaults()
    config = defaults["public_ownership"]
    specs = {s.case_id: s for s in configured_case_specs(config)}
    bundles = {sid: specs[sid].settings for sid in SCENARIOS}
    named_cfg = config["sensitivities"]["named"]

    bill = _read_json(PO / f"{GROUP}_consumer_bill_by_ownership_case_summary.json")
    tariff = _read_json(PO / f"{GROUP}_nationalised_bill_summary.json")
    institution = _read_json(PO / f"{GROUP}_nationalised_bill_by_institution.json")
    account = _read_json(PO / f"{GROUP}_public_account_summary.json")["cases"]
    financing = pd.read_csv(PO / f"{GROUP}_financing_summary.csv").set_index("case_id")
    feedback = pd.read_csv(PO / f"{GROUP}_workbook_esm_feedback.csv").set_index("case_id")
    comps = pd.read_csv(PO / f"{GROUP}_consumer_bill_by_ownership_case.csv")
    tariff_comps = pd.read_csv(PO / f"{GROUP}_nationalised_bill_components.csv")
    inst_lines = pd.read_csv(PO / f"{GROUP}_nationalised_bill_by_institution.csv").set_index("component_id")
    multi_year = pd.read_csv(PO / f"{GROUP}_nationalised_bill_multi_year.csv")
    # Review 29 Sep 2026: P3 is priced on its own bid months, so its multi-year series is
    # the P3 group's, which the tariff now measures against the headline private bill.
    p3_multi_year = PO / f"{P3_GROUP}_nationalised_bill_multi_year.csv"
    if p3_multi_year.exists():
        p3 = pd.read_csv(p3_multi_year)
        if "basis" not in p3.columns or not p3["basis"].astype(str).str.startswith("against the headline").all():
            raise ValueError(f"{p3_multi_year} is not on the headline private basis; rebuild the P3 tariff first")
        multi_year = pd.concat([multi_year.loc[~multi_year["case_id"].isin(p3["case_id"])], p3], ignore_index=True)
    network = _read_json(PO / f"{GROUP}_network_finance_account.json")
    # Item D2: the Exchequer ledger beside each bill headline (optional until the
    # rule has run; a case missing from it simply carries no ledger).
    ledger_path = PO / f"{GROUP}_exchequer_ledger.json"
    ledger = (_read_json(ledger_path).get("cases") or {}) if ledger_path.exists() else {}
    # Item BM1: the balancing-bid cases are projected by their own run group and set
    # beside the market cases by merge_balancing_groups.py.  Only bid_stack_effect
    # (the case against its own scenario on the market months) is ever presented;
    # the merge's private-on-stack term is not a feasible case.
    bm1_path = PO / f"{GROUP}_consumer_bill_balancing_cases_summary.json"
    bm1_cases = (_read_json(bm1_path).get("cases") or {}) if bm1_path.exists() else {}
    bm1_lines = (pd.read_csv(PO / f"{GROUP}_consumer_bill_balancing_cases.csv") if bm1_cases else pd.DataFrame())
    bm1 = bm1_cases.get("public_ownership@state_bids_at_cost")
    bm1_text = (f"{bm1['bid_stack_effect_gbp_inc_vat']:+.2f} GBP inc VAT" if bm1 else "under a pound")
    _fill_bm1(bm1_text)
    vat_rate = float(bill["cases"]["private"]["vat_gbp"]) / float(bill["cases"]["private"]["ex_vat_gbp"])
    # F19 (7 Oct 2026): household <-> system through each line's own recovery entry
    # (scripts/public_ownership/bill_recovery.py); the flat key is the Exchequer's (metered supply).
    from scripts.public_ownership import bill_recovery

    factor = float(bill["model_rebasing_factor"]); exchequer_key = bill_recovery.exchequer_key_mwh(bill)
    mwh = float(bill["benchmark_kwh"]) / 1000.0
    two_volume = bill_recovery.is_two_volume(bill)

    all_comps = pd.concat([comps, tariff_comps], ignore_index=True)
    component_meta = {}
    for r in all_comps.drop_duplicates("component_id").itertuples(index=False):
        component_meta[r.component_id] = {"label": str(r.component), "group": str(r.stack_group), "category": str(r.category)}
    comp_values = {c: g.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"].to_dict()
                   for c, g in all_comps.groupby("case_id")}
    # The institution case: the fully nationalised tariff with the measured lines moved to I3.
    inst_case = institution.get("institution_case_id") or "public_ownership@fully_nationalised_market_value_central_dispatch"
    i1_case = institution.get("i1_case_id") or "public_ownership@fully_nationalised_market_value"
    base = dict(comp_values[i1_case])
    for cid, row in inst_lines.iterrows():
        base[cid] = float(row["i3_gbp_per_3100kwh_ex_vat"])
        if cid not in component_meta:
            component_meta[cid] = {"label": NEW_LINE_LABELS.get(cid, cid), "group": "nationalised_generation",
                                   "category": "Cost of generation and storage (nationalised)"}
    comp_values[inst_case] = base

    def household_to_system(delta_hh, line_id=None):
        """A household amount as system GBP: through ``line_id``'s entry, else flat on the Exchequer's key."""
        if line_id is not None and two_volume:
            return bill_recovery.system_from_household(bill, line_id, delta_hh)
        return float(delta_hh) * exchequer_key / factor / mwh

    def on_bill_system(case_id, comparator, ex_vat_change):
        """The bill change at system level (each line at its own recovery); flat on the key on a D1 bill."""
        measured = (bill_recovery.bill_change_system_gbp(case_id, bill=bill, comparator=comparator, tariff=tariff,
                                                         institution=institution) if two_volume else None)
        return -measured if measured is not None else household_to_system(ex_vat_change)

    cases = []
    for case_id, spec in specs.items():
        scenario = spec.base_case
        levers = dict(spec.settings)
        moved = [n for n in LEVERS if levers[n] != bundles[scenario][n]]
        if spec.variant is None:
            kind, label, description = "scenario", SCENARIOS[scenario]["label"], SCENARIOS[scenario]["description"]
            assumption_explanation = dict(SCENARIOS[scenario]["guide"])
        elif spec.variant == "all_alternatives":
            kind, label = "bound", "All six structural levers at once"
            description = "The bounding case: every structural lever on its alternative at the same time."
            changed = "; ".join(
                f"{LEVER_LABELS[n]} → {VALUE_LABELS[n].get(levers[n], levers[n])}" for n in moved)
            assumption_explanation = {
                "change": f"Every structural lever that can affect this scenario is moved together: {changed}.",
                "mechanism": "The bill reflects the combined interaction of ownership boundary, CfD design, route booking, merchant contracting and the LDES floor; it is recalculated as one case rather than adding isolated effects.",
                "held_fixed": "All non-structural financing, acquisition and institutional levers stay at the scenario setting, and the CP30 2030 physical background is unchanged.",
                "caution": "This is an outer bound, not a coherent central policy package. Its result need not equal the sum of the one-at-a-time sensitivities because comparators and bill routes interact.",
            }
        elif spec.variant in named_cfg:
            kind = "named"
            label, description = NAMED_LABELS[spec.variant]
            assumption_explanation = dict(NAMED_GUIDES[spec.variant])
        else:
            lever = moved[0] if moved else None
            kind = "structural" if lever in STRUCTURAL_LEVERS else "workbook" if lever in WORKBOOK_LEVERS else "ownership"
            label = f"{LEVER_LABELS[lever]}: {VALUE_LABELS[lever][levers[lever]]}" if lever else spec.variant
            description = LEVERS[lever].description if lever else ""
            guide = LEVER_GUIDES[lever]
            base_value = bundles[scenario][lever]
            case_value = levers[lever]
            assumption_explanation = {
                "change": (
                    f"{LEVER_LABELS[lever]} moves from “{VALUE_LABELS[lever].get(base_value, base_value)}” "
                    f"to “{VALUE_LABELS[lever].get(case_value, case_value)}”. "
                    f"{guide['value_explanations'][case_value]}"
                ),
                "mechanism": guide["mechanism"],
                "held_fixed": (
                    f"{guide['held_fixed']} Every other lever remains at the {SCENARIOS[scenario]['label'].lower()} "
                    "setting unless stated otherwise."
                ),
                "caution": guide["caution"],
                "question": guide["question"],
            }
        # A proposition is surfaced as its own kind so a policy reader meets the
        # coherent packages first rather than hunting for them.  Since item L2
        # P1 and P2 are the scenarios themselves and keep the "scenario" kind;
        # the rest are re-kinded here whatever branch built them, because P2b is
        # now an ordinary one-lever case and P1b a named one.
        if case_id in PROPOSITIONS and spec.variant is not None:
            kind = "proposition"
            if case_id in PROPOSITION_LABELS:
                label, description = PROPOSITION_LABELS[case_id]
        stage = "market_bill" if spec.market_bill() else ("institution_comparison" if case_id == inst_case else "acquisition_tariff")
        if stage == "market_bill" and case_id not in bill["cases"] and case_id in bm1_cases:
            stage = "balancing_bids"
        entry = {}
        if stage == "balancing_bids":
            m = bm1_cases[case_id]
            comparator = m["base_case"]
            rows = bm1_lines.loc[bm1_lines["case_id"].eq(case_id)].set_index("component_id")
            ex_vat = float(rows["case_on_group"].sum())
            inc = float(m["total_gbp_inc_vat"])
            entry = {"ex_vat": ex_vat, "gross_inc_vat": inc, "net_inc_vat": inc}
            comp_b = bill["cases"][comparator]
            comparator_bill = {"ex_vat": comp_b["ex_vat_gbp"], "gross_inc_vat": comp_b["gross_total_gbp_inc_vat"],
                               "net_inc_vat": comp_b["net_total_gbp_inc_vat"]}
            # The P2 increment, never the private-on-stack decomposition term.
            delta_gross = delta_net = float(m["bid_stack_effect_gbp_inc_vat"])
            routes = {}
            comp_values[case_id] = rows["case_on_group"].astype(float).to_dict()
            purchase = None
            years = []
        elif stage == "market_bill":
            b = bill["cases"][case_id]
            comparator = b["comparator_case_id"]
            if scenario == "private":
                # A private variant is its own comparator in the accounts (house rule 3);
                # for the reader the useful reading is against the central private bill.
                comparator = "private"
            entry = {"ex_vat": b["ex_vat_gbp"], "gross_inc_vat": b["gross_total_gbp_inc_vat"], "net_inc_vat": b["net_total_gbp_inc_vat"]}
            comp_b = bill["cases"][comparator]
            comparator_bill = {"ex_vat": comp_b["ex_vat_gbp"], "gross_inc_vat": comp_b["gross_total_gbp_inc_vat"], "net_inc_vat": comp_b["net_total_gbp_inc_vat"]}
            delta_gross = b["gross_total_gbp_inc_vat"] - comp_b["gross_total_gbp_inc_vat"]
            delta_net = b["net_total_gbp_inc_vat"] - comp_b["net_total_gbp_inc_vat"]
            acct = account.get(case_id) or {}
            routes = {
                "cfd_repricing": b["bill_allocated_r1_gbp_2024"], "ldes_floor": b["bill_allocated_ldes_gbp_2024"],
                "public_cfd_at_cost": b["bill_allocated_public_cfd_at_cost_gbp_2024"], "w1_contracting": b["bill_allocated_w1_gbp_2024"],
                "rebate": b["bill_allocated_rebate_gbp_2024"], "shortfall_on_bill": b["shortfall_on_bill_gbp_2024"],
                "exchequer_shortfall": b["shortfall_on_exchequer_gbp_2024"], "retained_public_account": b["public_account_retained_gbp_2024"],
                "cm_booked": b["route_booking_booked_gbp_2024"].get("capacity_market_public_financing_adjustment", 0.0),
                "ccs_booked": b["route_booking_booked_gbp_2024"].get("allocated_ccs_public_financing_adjustment", 0.0),
                "network_adjustment": sum(b["network_adjustments_system_gbp_2024"].values()),
                "pay_as_bid_ceiling": b["pay_as_bid_ceiling_gbp_2024"],
            }
            purchase = None
            years = []
        elif stage == "acquisition_tariff":
            r = tariff["cases"][case_id]; b = r["bill"]
            comparator = "private"
            entry = {"ex_vat": b["ex_vat_gbp"], "gross_inc_vat": b["total_gbp_inc_vat"], "net_inc_vat": b["total_gbp_inc_vat"]}
            pb = bill["cases"]["private"]
            comparator_bill = {"ex_vat": pb["ex_vat_gbp"], "gross_inc_vat": pb["gross_total_gbp_inc_vat"], "net_inc_vat": pb["net_total_gbp_inc_vat"]}
            delta_gross = delta_net = b["delta_inc_vat_gbp"]
            routes = {"network_adjustment": sum(household_to_system(v, line_id)
                                                for line_id, v in b["network_adjustments_gbp_per_3100kwh_ex_vat"].items())}
            purchase = r["diagnostics"]["acquisition_price_gbp"]
            my = multi_year.loc[multi_year["case_id"].eq(case_id)]
            years = [{"year": str(y), "bill_delta_inc_vat": float(d) * (1 + vat_rate), "legacy_component_hh": float(l)}
                     for y, d, l in zip(my["year"], my["bill_delta_gbp_per_3100kwh_ex_vat"], my["legacy_component_gbp_per_3100kwh_ex_vat"])]
        else:
            comparator = "private"
            ex_vat = float(institution["i3_ex_vat_gbp"]); inc = float(institution["i3_inc_vat_gbp"])
            entry = {"ex_vat": ex_vat, "gross_inc_vat": inc, "net_inc_vat": inc}
            pb = bill["cases"]["private"]
            comparator_bill = {"ex_vat": pb["ex_vat_gbp"], "gross_inc_vat": pb["gross_total_gbp_inc_vat"], "net_inc_vat": pb["net_total_gbp_inc_vat"]}
            delta_gross = delta_net = inc - float(institution["private_inc_vat_gbp"])
            routes = {"institution_change_vs_i1_inc_vat": float(institution["institution_change_inc_vat_gbp"])}
            # The institution case buys what its I1 twin buys: P3 is the market-value
            # purchase (83.5 bn), not the state-PV fully_nationalised one (105.9 bn),
            # which is what this read until 26 Sep 2026.
            purchase = tariff["cases"][i1_case]["diagnostics"]["acquisition_price_gbp"]
            years = []
        fin = financing.loc[case_id] if case_id in financing.index else {"annual_financing_saving_vs_private_gbp": float("nan")}
        led = ledger.get(case_id)
        exchequer = None if led is None else {
            "net_vs_private_gbp": float(led["exchequer_net_vs_private_gbp"]),
            "net_vs_private_per_household": float(led["exchequer_net_vs_private_gbp_per_household"]),
            "net_of_exchequer_inc_vat": float(led["net_of_exchequer_gbp_inc_vat"]),
            "net_of_exchequer_delta_vs_private_inc_vat": float(led["net_of_exchequer_delta_vs_private_inc_vat"]),
            "parts_gbp": {k: float(v) for k, v in led["exchequer_net_vs_private_parts_gbp"].items()},
            "carbon_receipts_gbp": float(led["carbon_receipts_gbp"]),
            "vat_receipts_gbp": float(led["vat_receipts_gbp"]),
            "corporation_tax_forgone_gbp": float(led["corporation_tax_forgone_gbp"]),
            "network_tax_forgone_gbp": float(led["network_tax_forgone_gbp"]),
            "state_deficits_funded_gbp": float(led["state_deficits_funded_gbp"]),
            "financing_saving_tax_transfer_gbp": float(led["financing_saving_tax_transfer_gbp"]),
            "financing_saving_resource_gbp": float(led["financing_saving_resource_gbp"]),
        }
        fb = feedback.loc[case_id] if case_id in feedback.index else None
        cases.append({
            "id": case_id, "scenario": scenario, "variant": spec.variant, "kind": kind, "label": label, "description": description,
            "proposition": PROPOSITIONS.get(case_id),
            "assumption_explanation": assumption_explanation,
            "stage": stage, "comparator": comparator, "levers": levers, "levers_moved": moved,
            "bill": entry, "comparator_bill": comparator_bill,
            "delta_gross_inc_vat": float(delta_gross), "delta_net_inc_vat": float(delta_net),
            "delta_vs_central_private_inc_vat": float(entry["gross_inc_vat"] - bill["cases"]["private"]["gross_total_gbp_inc_vat"]),
            "financing_saving_gbp": (None if pd.isna(fin["annual_financing_saving_vs_private_gbp"])
                                     else float(fin["annual_financing_saving_vs_private_gbp"])),
            "on_bill_saving_gbp": on_bill_system(case_id, comparator, comparator_bill["ex_vat"] - entry["ex_vat"]),
            "realised_pass_through": (None if fb is None or pd.isna(fb["realised_pass_through"]) else float(fb["realised_pass_through"])),
            "purchase_price_gbp": purchase, "routes": {k: float(v) for k, v in routes.items()},
            "components": {k: float(v) for k, v in comp_values[case_id].items()},
            "multi_year": years,
            "exchequer": exchequer,
        })

    # Within their group the propositions read P1, P1b, P2, ... rather than in
    # config declaration order, which `extends` constrains.  Only the
    # proposition rows move, and only into each other's slots, so every other
    # case keeps its position in the list.
    _rank = {case_id: i for i, case_id in enumerate(PROPOSITION_ORDER)}
    _slots = [i for i, c in enumerate(cases) if c["kind"] == "proposition"]
    for _slot, _case in zip(_slots, sorted((cases[i] for i in _slots), key=lambda c: _rank[c["id"]])):
        cases[_slot] = _case

    payload = {
        "meta": {
            "generated": date.today().isoformat(), "model_year": 2030, "price_basis": bill["price_basis"],
            "benchmark_kwh": bill["benchmark_kwh"], "vat_rate": vat_rate,
            # F19 (user decision, 6 Oct 2026): P3's institution step from two solve runs, flagged.
            "institution_mixed_vintage": institution.get("mixed_vintage"),
            "institution_solve_vintage": institution.get("solve_vintage"),
            # The Exchequer's per-household key (metered supply); a bill line converts at its own entry.
            "final_demand_mwh": exchequer_key, "exchequer_key_mwh": exchequer_key,
            "system_gbp_per_household_gbp": exchequer_key / factor / mwh,
            "network_saving_per_household": {k: v["total_saving_gbp_per_3100kwh_ex_vat"] for k, v in network["treatments"].items()},
            "source": "PyPSA-GB CP30 2030, twelve reinforced months; finance layer item K1 (21 September 2026)",
            "scope_notes": [
                "One physical background: CP30 2030, using twelve reinforced solved months.",
                "No demand, fuel-price, weather-year, capacity-expansion, network-topology or FES-pathway sensitivity is varied here.",
                "Ownership cases normally reuse the same hourly dispatch; only the labelled central-dispatch institution case uses the second solved run.",
                "Financing savings change bills only through an explicit contract, levy, rebate, network or public-account route.",
                "Acquisition cases replace the market bill with an acquisition tariff and must be read with their purchase price and recovery term.",
                "Beside each bill is the Exchequer ledger (item D2): carbon receipts, domestic VAT, corporation tax the state's plant does not pay (a transfer within the public sector; Great British Energy pays corporation tax as a company today), the network tax allowance and state deficits, against private; the net-of-Exchequer figure is the bill plus what the case costs the Exchequer. The bill headline itself is unchanged.",
            ],
        },
        "scenarios": [{"id": k, **v, "bundle": {n: bundles[k][n] for n in LEVERS if bundles[k][n] != LEVERS[n].default}} for k, v in SCENARIOS.items()],
        "lever_groups": [{"id": k, **v} for k, v in LEVER_GROUPS.items()],
        "levers": {n: {"label": LEVER_LABELS[n], "description": LEVERS[n].description, "values": list(LEVERS[n].values),
                       "value_labels": VALUE_LABELS[n], "structural": n in STRUCTURAL_LEVERS,
                       **LEVER_GUIDES[n]} for n in LEVERS},
        "stack_groups": [{"id": g, "label": l} for g, l in STACK_GROUPS],
        "components": component_meta,
        "cases": cases,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="resources/analysis/ownership_explorer_data.json")
    args = parser.parse_args(argv)
    payload = build(Path(args.out))
    print(json.dumps({"cases": len(payload["cases"]), "components": len(payload["components"]), "out": args.out}))


if __name__ == "__main__":
    main()
