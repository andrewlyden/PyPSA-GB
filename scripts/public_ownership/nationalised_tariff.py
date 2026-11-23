"""The nationalised tariff.

Under full nationalisation the consumer does not pay a market price plus the levies
that top generators up to their contracts.  They pay what generation and storage
cost.  This module replaces the market-construct lines of the private bill with
that cost, built from the ``public_nationalised`` rows of the asset finance
register, and reconciles the two bills step by step so every pound that moves is
attributable.

**Replaced** (constructs of the market, gone by construction): the wholesale
purchase and its hedging uplifts, CfD difference payments, the Renewables
Obligation, balancing constraint payments, the Capacity Market (except the
agreements with capacity the state does not acquire, which stay as their own
line), the allocated CCS levy (BECCS is in the register) and the long-duration
storage floor.

**Kept and declared**: network charges, supplier costs, losses, NESO non-constraint
services, feed-in tariffs (small-scale plant is not bought, decision 7), the
Sizewell C RAB, the remaining CCS levy and the smaller policy lines.

**Added**: legacy acquisition charge (renewables and nuclear; thermal and storage),
new-build capital at the public rate, site fixed O&M, generator TNUoS and insurance
(item D4(a): three lines, the register's components of its fixed O&M), variable
O&M, fuel and carbon,
cross-border trade (measured on the links at the foreign price when the market
months are given, item N10; else the register's domestic import revenue, an
upper bound), plant the state does not acquire paid as today (carriers without a
cost mapping: CHP, OCGT, gas engines, oil), the Capacity Market agreements with
capacity it does not acquire, and the real-time correction cost under
institution I3 (zero for I1).

**Allocation, like for like (audit 26 Sep 2026).**  The tariff recovers each new
line by the allocation rules of the private lines it replaces, never flat over
final demand.  Until this audit every new line was spread flat over 318.3 TWh of
final demand, while the lines it replaces are not recovered that way: the CfD and
allocated CCS lines over CfD-eligible supplier sales (304.5 TWh, the EII-exempt
volume excluded), the Renewables Obligation over obligation-eligible supply, the
Capacity Market with its 1.348 household peak factor, and the wholesale purchase
with a domestic shaping line and hedging uplifts on the household's peak-heavy
energy cost.  Spreading the replacement flat moved part of the cost of generation
from households onto non-domestic users, so part of the reported household
"saving" was a reallocation, not a saving.  Now (:func:`recovery_rates`):

* each replaced line's system cost is recovered back out of its household value
  at its own rule (shaping is an allocation between consumers, zero in aggregate);
* the lines that are a *kept part* of a replaced line take that line's rule: the
  Capacity Market agreements the state does not acquire (peak factor), and the
  interconnector re-trading (I1), real-time correction (I3) and borrowed
  forecast-error charge, which are balancing costs recovered over final demand
  like the constraint line;
* every other new line, the cost of owned generation and storage, takes the
  *pool rate*: the household share of the replaced market-construct lines less
  those kept parts, so the cost that replaces them falls on households in the
  proportion they paid for what it replaces.

``tariff_allocation`` in each case's summary reports the pool rate against the
flat rate and the bill both ways, so the before and after stay visible.  There
is still no hourly price: the pool reproduces the shaping line's share, not a
time-of-use tariff.

**Interconnectors (item D3, 26 Sep 2026).**  With imports paid at the foreign
price on the links, the cables' congestion rent (the GB price value of the
day-ahead schedule less what was paid abroad, corrected export-loss convention)
stays with the public buyer unless it is owed to someone.  Every tariff case
(every case that acquires the legacy fleet, whatever its networks lever; user
decision 26 Sep 2026) also acquires the **GB halves** of the cables: their rent
accrues to the tariff, they leave the
cap-and-floor regime (the pass-through line stays dropped and no floor is owed),
their Capacity Market agreements leave the kept Capacity Market line, and the
tariff pays an **acquisition annuity** at the public rate on a placeholder price
(market value: the GB-half rent's present value at an owner's rate,
``public_ownership.acquisition.interconnectors``).  The **foreign halves**, and
whole cables owned and regulated in another jurisdiction (East West, Moyle, the
Isle of Man), keep their rent and are paid it on a line with a named payee.  GB
shares come from the bill's regime table
(:func:`scripts.market.fti_settlement.gb_half_shares`).  A cable commissioned in
the model year (MaresConnect in 2030, regime table ``public_ownership_treatment:
new_build_at_cost``, 15 Oct 2026) is new build, not a legacy acquisition: its GB
half is bought at cost on completion, the present value of its regime floor at the
public rate, on the same annuity line.

Item N4: the purchase-price basis, thermal basis and annuity term are reported
as one-at-a-time variants recomputed here from the per-asset legacy valuation,
without re-running the register.  Item N8: the multi-year view uses the legacy
cash-flow profile, holding everything except legacy contract expiry and
retirement at model-year levels (2030 on the CP30 year); the annuities on the
interconnectors' GB halves and the transaction fees run off at their own term.

**F19 P3 decisions (user, 6 Oct 2026).**

* The Net Zero Teesside DPA is paid once: the bill keeps its ``remaining_ccs`` line,
  so the plant paid as today is paid its market revenue only (its ``dpa_support_gbp``
  is a declared residual part, never a second payment).
* Under I1 the supplier imbalance allowance (``procurement_imbalance``) stays on the
  bill as I1's forecast-error cost; I3 replaces it with FTI's borrowed charge
  (``i1_forecast_error_cost`` in each case, read by ``nationalised_institutions``).
* Energy from waste: the state's gate fees, as the purchase price counts them (the
  legacy valuation's ``gate_fee_model_year_gbp``, capped at the UK's throughput), are
  credited on ``nationalised_efw_gate_fee_income``, at the pool rate on the two-volume
  basis (27 Oct 2026: income of owned plant, recovered as its cost is; flat before).
* The break-even price is the lifetime one: the price at which the legacy purchase's
  present value at the public rate is zero (the 2030 snapshot's is kept, renamed).
* ``three_parts``: the bill change against private split into new-build ownership
  (what P2 also does), the legacy purchase (its 2030 gain, its year path to the end of
  the annuity and its present value at the public rate) and the rest, each defined
  item by item on the register ledger, summing to the bill change exactly.

Usage::

    python scripts/public_ownership/nationalised_tariff.py <finance_register> <bill_by_case>
        <legacy_valuation> <legacy_profile> <assumptions.json> <recovery.json>
        <out_components.csv> <out_summary.json> <out_reconciliation.csv> <out_multi_year.csv>
        [--config config] [--realtime-correction-gbp 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.finance import NATIONALISED_CASE, capital_recovery_factor  # noqa: E402
from scripts.public_ownership.legacy_valuation import (  # noqa: E402
    GROUP_EXCLUDED, GROUP_VALUED, case_profile as legacy_case_profile,
)
from scripts.public_ownership import sizewell_c_rab  # noqa: E402
from scripts.consumer_bill.line_recovery import (  # noqa: E402
    entry_for, household_to_system_gbp, make_entry, system_to_household_gbp,
)
from scripts.public_ownership import bill_recovery  # noqa: E402
from scripts.public_ownership import fingerprints  # noqa: E402  (F20 phase 2)
from scripts.consumer_bill.bill_groupings import (  # noqa: E402
    annotate as annotate_bill_groups, group_of as bill_group_of, summary_block as bill_group_totals,
)
from scripts.public_ownership.model_year import (  # noqa: E402
    DEFAULT_MODEL_YEAR, cm_calendar_revenue_column, is_vintaged_model_year, year_offsets,
)
from scripts.public_ownership.levers import (  # noqa: E402
    INSTITUTION_MARKET, LEVER_DISPATCH_INSTITUTION, LEVER_LEGACY_PRICE_PATH, LEVER_NETWORKS, NETWORKS_PRIVATE,
    TERM_REMAINING_LIFE, case_settings,
    comparator_for_settings, spec_for_row,
)
from config.config_loader import read_config

VALUE = "annual_gbp_per_3100kwh_ex_vat"
IDENTITY_TOLERANCE_GBP = 1.0
STACK_GROUP = "nationalised_generation"
CATEGORY = "Cost of generation and storage (nationalised)"
SOURCE_CLASS = "public_ownership_nationalised_tariff"

REPLACED_LINES: dict[str, str] = {
    "direct_energy": "wholesale purchase: no market price under nationalisation",
    "shaping_adjustment": "hedging uplift on the wholesale purchase",
    "procurement_transaction_cost": "hedging uplift on the wholesale purchase",
    "procurement_day_ahead_rehedging": "hedging uplift on the wholesale purchase",
    "procurement_additional_risk_and_basis": "hedging uplift on the wholesale purchase",
    "procurement_forward_premium": "hedging uplift on the wholesale purchase",
    # Item D4(d): the supplier imbalance allowance is a rate on the domestic energy
    # cost like the hedging uplifts; a single public buyer holds no supplier
    # position to cash out.
    "procurement_imbalance": "supplier imbalance allowance on the wholesale purchase: no supplier position to cash out",
    # Flow-of-funds check, 27 Sep 2026: under metered recovery (item D1) the model's
    # loads are GSP-level, so energy line + losses line = price x GSP volume, and the
    # owned fleet's cost of generation already covers the energy lost between GSP
    # and meter. Keeping the line counted it twice (~1.1 bn a year).
    "procurement_losses": "losses energy: the owned fleet already generates for GSP-level load, losses included",
    "cfd": "CfD difference payments: no contracts to settle, the plant is owned",
    "bsuos_constraints": "balancing constraint payments: nobody is paid to deviate from a position they hold with themselves",
    "renewables_obligation": "Renewables Obligation: no certificates, the plant is owned",
    "capacity_market": ("Capacity Market: the acquired fleet's fixed cost is in the tariff; agreements with plant "
                        "the state does not acquire are put back as their own line"),
    "allocated_ccs": "allocated CCS levy: BECCS is in the register at the public rate",
    "ldes_cap_and_floor": "LDES floor: storage is simply owned and costed",
}
#: Task SETTLE2050: market-construct lines only a 2050 bill carries. Replaced like the rest
#: where the private bill has them, and reported among the replaced lines only then, so a
#: 2030 tariff's outputs are unchanged.
OPTIONAL_REPLACED_LINES: dict[str, str] = {
    "dpa": "dispatchable power agreements: the gas CCS and hydrogen plant is owned and costed",
}
ALL_REPLACED_LINES: dict[str, str] = {**REPLACED_LINES, **OPTIONAL_REPLACED_LINES}
#: FES 2050 decision 41 (c): Sizewell C's RAB levy. In 2030 Sizewell C is under construction
#: and outside the register, so the levy is kept; a 2050 register owns and costs the plant and
#: books its RAB allowance, so the tariff replaces the line, but only where the register says
#: so (:func:`register_replaced_lines`): the line is in every 2030 private bill.
RAB_LINE = "sizewell_c_rab"
TOP_UP_LINE = "life_extension_operating_top_up"
#: FES 2050 decision 63: the battery revenue top-up line (any bill that configures it).
BATTERY_TOP_UP_LINE = "battery_revenue_top_up"
REGISTER_REPLACED_LINES: dict[str, str] = {
    RAB_LINE: ("Sizewell C RAB levy: the plant is owned and costed in the tariff (a 2050 register books "
               "its RAB allowance, decision 41 (c))"),
    TOP_UP_LINE: ("life-extension operating top-up: the plant is owned and its operating cost is in the tariff "
                  "(decision 58)"),
    BATTERY_TOP_UP_LINE: ("battery revenue top-up: the batteries are owned and their cost, capital at the public "
                          "rate included, is in the tariff (decision 63)"),
}


def register_replaced_lines(diagnostics: Mapping) -> dict[str, str]:
    """Private lines the tariff replaces because the register books them to owned plant (2050 only)."""
    out = {}
    if float(diagnostics.get("private_rab_support_gbp", 0.0) or 0.0) > 0.0:
        out[RAB_LINE] = REGISTER_REPLACED_LINES[RAB_LINE]
    if float(diagnostics.get("private_top_up_gbp", 0.0) or 0.0) > 0.0:
        out[TOP_UP_LINE] = REGISTER_REPLACED_LINES[TOP_UP_LINE]
    if float(diagnostics.get("private_battery_top_up_gbp", 0.0) or 0.0) > 0.0:
        out[BATTERY_TOP_UP_LINE] = REGISTER_REPLACED_LINES[BATTERY_TOP_UP_LINE]
    return out
# Lines dropped without replacement when imports are priced at the foreign price
# on the links and the state acquires the GB halves of the cables (user decisions,
# 26 Sep 2026, item D3): the acquired halves leave their regimes, their rent
# accrues to the tariff and no floor is owed, so there is nothing to return
# through the cap-and-floor regime. Kept outside the pool, so no new line's rate
# moves. Under the register basis (imports at the GB price) the line stays.
INTERCONNECTOR_PASS_THROUGH_LINE = "interconnector_cap_and_floor_pass_through"
DROPPED_UNDER_FOREIGN_PRICE_IMPORTS: dict[str, str] = {
    INTERCONNECTOR_PASS_THROUGH_LINE: ("interconnector cap-and-floor pass-through: the GB halves are acquired and "
                                       "leave their regimes (their rent accrues to the tariff, no floor is owed); "
                                       "the foreign halves are paid their rent on its own line"),
}


def dropped_lines_for(diagnostics: Mapping) -> dict[str, str]:
    """The private lines the tariff drops without replacement, given its import basis."""
    if diagnostics.get("import_basis") == "measured_link_trade_at_the_foreign_price":
        return dict(DROPPED_UNDER_FOREIGN_PRICE_IMPORTS)
    return {}


# Item N6: when the state also owns the wires, these two lines fall by the
# financing wedge on the Regulatory Asset Value plus the corporation-tax
# allowance a public owner does not pay. Built by network_finance_account.py.
NETWORK_ADJUSTED_LINES = ("tnuos", "duos")
#: F19 P3 decision 6 (user, 6 Oct 2026): replaced lines a case keeps under institution I1.
#: The supplier imbalance allowance stands for the forecast error a market institution
#: still bears; I3 replaces it with FTI's borrowed forecast-error charge.
IMBALANCE_LINE = "procurement_imbalance"
KEPT_UNDER_I1: dict[str, str] = {
    IMBALANCE_LINE: ("F19 P3 decision 6: kept under I1 as I1's forecast-error cost (the market institution still "
                     "settles imbalance); I3 replaces it with FTI's borrowed forecast-error charge"),
}


def kept_replaced_lines(institution: str) -> dict[str, str]:
    """The replaced lines a tariff keeps under ``institution`` (F19 P3 decision 6: the imbalance allowance under I1)."""
    return dict(KEPT_UNDER_I1) if str(institution) == "I1" else {}
KEPT_DECLARED: dict[str, str] = {
    "bsuos_other_services": "NESO non-constraint services remain a real activity",
    "supplier_costs_and_margin": ("decision 6: cost to serve unchanged; F27 item 10 (20 Oct 2026): EBIT and headroom "
                                  "at Ofgem's percentages of the tariff's own costs, as the cap would set them"),
    "tnuos": "decision 4: unchanged unless network_ownership is acquired, when item N6 lowers it by the financing wedge on the transmission RAV",
    "duos": "decision 4: unchanged unless network_ownership is acquired, when item N6 lowers it by the wedge on the distribution RAV plus the tax allowance",
    "sizewell_c_rab": ("construction-period RAB payments kept; under consumer_bill.sizewell_c_rab_public_rate the "
                       "return element is financed at the public rate (OW10, was G4)"),
    "feed_in_tariffs": "decision 7: small-scale plant is not bought",
    "remaining_ccs": "CCS plant outside the register",
    "nuclear_contract_counterparty": "LCCC and ESC operating costs kept (small)",
    "bsuos_voltage_and_stability_constraints": ("NESO voltage and inertia actions remain a real activity (item D4(d)); "
                                                "only thermal constraint payments go with ownership"),
}
LINE_LEGACY_RENEWABLES = "nationalised_legacy_acquisition_renewables_nuclear"
LINE_LEGACY_THERMAL = "nationalised_legacy_acquisition_thermal_storage"
LINE_NEW_BUILD = "nationalised_new_build_capital"
LINE_FIXED_OM = "nationalised_fixed_om"
LINE_GENERATOR_TNUOS = "nationalised_generator_tnuos"
LINE_INSURANCE = "nationalised_insurance"
LINE_VARIABLE_OM = "nationalised_variable_om"
LINE_FUEL = "nationalised_fuel_carbon"
LINE_STATE_OVERHEAD = "nationalised_state_operating_overhead"
LINE_IMPORTS = "nationalised_imports"
LINE_UNACQUIRED = "nationalised_unacquired_paid_as_today"
LINE_REALTIME = "nationalised_realtime_correction"
LINE_IC_RETRADING = "nationalised_interconnector_retrading"
LINE_CANCELLED_RESIDUAL = "nationalised_cancelled_contract_residual"
LINE_CM_NOT_ACQUIRED = "nationalised_capacity_market_not_acquired"
LINE_FORECAST_ERROR = "nationalised_forecast_error_charge"
LINE_IC_FOREIGN_RENT = "nationalised_interconnector_foreign_half_rent"
LINE_IC_ACQUISITION = "nationalised_interconnector_gb_half_acquisition"
# Review 29 Sep 2026 (user decisions 2, 5 and the explorer's assumption levers).
LINE_BECCS_CAPTURE = "nationalised_beccs_carbon_removal_cfd"
LINE_TAX_EQUIVALENT = "nationalised_tax_equivalent"
#: Independent review (15 Nov 2026): the state that owns Net Zero Teesside keeps its DPA, which the bill's kept
#: remaining_ccs line still pays; the receipt comes back off the tariff, so the DPA is not paid twice.
LINE_DPA_RECEIPTS = "nationalised_dpa_receipts_on_owned_plant"
#: Owned carriers the register has no cost data for, paid as today by design (the module's rule for CHP and oil;
#: 15 Nov 2026: in the registers only 2030 legacy oil is owned, 165 MW, about GBP 0.1 m a year). Any other owned
#: uncosted carrier is refused: it has cost data and lost it (the 2040 OCGTs and gas engines did).
OWNED_UNCOSTED_DECLARED = {"oil": "legacy oil: no cost data, paid as today (2030: 165 MW, about GBP 0.1 m a year)",
                           "CHP": "CHP: no cost data, paid as today (owned in no register case on 15 Nov 2026)"}
#: The register-ledger attribution of LINE_DPA_RECEIPTS (the three-part split assigns it by this label).
DPA_RECEIPTS_STEP = "Net Zero Teesside DPA received by the state as owner, returned to the bill (the kept remaining_ccs line still pays it)"
LINE_TRANSACTION_COSTS = "nationalised_acquisition_transaction_costs"
LINE_LEGACY_FIT = "nationalised_fit_domestic_to_taxation"
LINE_LEGACY_WHD = "nationalised_whd_domestic_to_taxation"
#: User answer Q5 (30 Sep 2026): under tariff_pricing: market_bill the bill is set to the
#: private comparator's and this line, paid to the Treasury, is the margin over cost.
LINE_TREASURY_MARGIN = "nationalised_treasury_margin"
#: New lines paid by, or to, the Exchequer rather than by consumers to plant: the
#: three-party ledger books them to the Exchequer (never to investors or resource).
EXCHEQUER_LINES = (LINE_BECCS_CAPTURE, LINE_TAX_EQUIVALENT, LINE_LEGACY_FIT, LINE_LEGACY_WHD, LINE_TREASURY_MARGIN)
#: The acquisition annuity lines lever acquisition_financing can move off the tariff.
ANNUITY_LINES = (LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, LINE_IC_ACQUISITION)
NEW_LINE_LABELS: dict[str, str] = {
    LINE_LEGACY_RENEWABLES: "Legacy renewables and nuclear: acquisition charge",
    LINE_LEGACY_THERMAL: "Legacy thermal and storage: acquisition charge",
    LINE_NEW_BUILD: "New build: capital at the public rate",
    LINE_FIXED_OM: "Site fixed O&M, all owned plant",
    LINE_GENERATOR_TNUOS: "Generator TNUoS (wider and offshore local), all owned plant (item D4(a))",
    LINE_INSURANCE: "Insurance, all owned plant (item D4(a))",
    LINE_VARIABLE_OM: "Variable O&M, all owned plant",
    LINE_FUEL: "Fuel and carbon, all owned plant",
    LINE_STATE_OVERHEAD: "State operating overhead, all owned plant (item OH1)",
    LINE_IMPORTS: "Imports, valued at what GB paid them",
    LINE_UNACQUIRED: "Plant not acquired, paid as today (CHP, oil)",
    LINE_REALTIME: "Real-time correction of the day-ahead schedule (institution I3)",
    LINE_IC_RETRADING: "Interconnector re-trading, paid to foreign counterparties (institution I1)",
    LINE_CANCELLED_RESIDUAL: "Cancelled legacy CfD and RO contracts: residual paid to former owners (item G3)",
    LINE_CM_NOT_ACQUIRED: ("Capacity Market agreements with capacity the state does not acquire "
                           "(foreign halves of interconnectors, DSR, plant paid as today, capacity outside the model)"),
    LINE_FORECAST_ERROR: "Forecast-error charge borrowed from FTI (institution I3; the model produces almost none)",
    LINE_IC_FOREIGN_RENT: ("Interconnector congestion rent paid to foreign owners (foreign halves; whole cables "
                           "owned in another jurisdiction)"),
    LINE_IC_ACQUISITION: "Interconnector GB halves: acquisition annuity at the public rate (priced at the present value of their rent)",
    LINE_BECCS_CAPTURE: ("Power BECCS carbon-removal leg paid by its carbon CfD (the Exchequer), never by the bill "
                         "(user decision 2)"),
    LINE_TAX_EQUIVALENT: "Tax equivalents paid to the Exchequer by state-owned new build (competitive neutrality)",
    LINE_DPA_RECEIPTS: ("Net Zero Teesside DPA received by the state as owner, credited back (the bill's remaining_ccs "
                        "line still pays the DPA)"),
    LINE_TRANSACTION_COSTS: "Acquisition advisory and transaction costs, annuitised like the purchase",
    LINE_LEGACY_FIT: "Feed-in Tariffs: domestic share moved to general taxation (a background choice)",
    LINE_LEGACY_WHD: "Warm Home Discount: domestic share moved to general taxation (a background choice)",
    LINE_TREASURY_MARGIN: ("The state charges market prices: the margin over the tariff's cost, kept by the Treasury "
                           "(tariff_pricing: market_bill)"),
}
NEW_LINES = tuple(NEW_LINE_LABELS)
#: FES 2050 decision 58: income lines a 2050 register's support lines give owned plant. Kept out of
#: NEW_LINES and recovered flat over final demand, except the gate-fee credit on the two-volume
#: basis (POOL_INCOME_LINES).
LINE_ANCILLARY_INCOME = "nationalised_ancillary_services_income"
LINE_GATE_FEE_INCOME = "nationalised_efw_gate_fee_income"
#: FES 2050 decision 67: the owned batteries' declared sub-hourly revenue.
LINE_SUBHOURLY_INCOME = "nationalised_battery_subhourly_income"
#: The register's support-line columns (support_lines_2050.SUPPORT_COLUMNS), present only where configured.
SUPPORT_LINE_COLUMNS = ("ancillary_services_revenue_gbp", "efw_gate_fee_revenue_gbp", "battery_subhourly_revenue_gbp",
                        "life_extension_top_up_gbp", "battery_revenue_top_up_gbp")
OPTIONAL_NEW_LINE_LABELS: dict[str, str] = {
    LINE_ANCILLARY_INCOME: ("Ancillary services provided by owned plant, paid to the state out of the kept BSUoS "
                            "lines and credited to the tariff (decision 58)"),
    LINE_GATE_FEE_INCOME: ("Energy-from-waste gate fees the state receives from waste disposal, credited to the "
                           "tariff (decision 58)"),
    LINE_SUBHOURLY_INCOME: ("Sub-hourly trading revenue of owned batteries (intra-hour spreads the hourly dispatch "
                            "does not see), paid by other market participants and credited to the tariff (decision 67)"),
}


def line_label(line_id: str) -> str:
    """A tariff line's label, the 2050-only income lines included."""
    return NEW_LINE_LABELS.get(line_id) or OPTIONAL_NEW_LINE_LABELS[line_id]
# Bill lines recovered over CfD-eligible or RO-eligible supply rather than final
# demand.  allocated_ccs is settled like a CfD and recovered over the same
# eligible sales (project_future_bill.py); it was missing here until 26 Sep 2026.
ELIGIBLE_SUPPLY_LINES = {"cfd": "cfd_eligible_mwh", "allocated_ccs": "cfd_eligible_mwh",
                         "renewables_obligation": "ro_eligible_mwh", "dpa": "cfd_eligible_mwh"}
# How each replaced line is recovered in the private bill (project_future_bill.py):
#   final_demand      flat over final demand (metered supply)
#   gsp_rate          a GSP rate (F18): over the GSP charging volume, on the household's
#                     loss-uplifted 3.1 MWh (BSUoS, LDES); flat over final demand on a D1 bill
#   cfd_eligible      over CfD-eligible supplier sales (EII exempt)
#   ro_eligible       over RO obligation-eligible supply
#   capacity_market   over final demand times the household peak factor (MR4.4)
#   domestic_energy   a rate times the household's energy cost, which the shaping
#                     line makes peak-heavy (the hedging uplifts)
#   allocation        a transfer between consumers, zero in aggregate (the shaping
#                     line: domestic-weighted price less the demand-weighted price)
HEDGING_LINES = ("procurement_transaction_cost", "procurement_day_ahead_rehedging", "procurement_imbalance",
                 "procurement_additional_risk_and_basis", "procurement_forward_premium")
REPLACED_LINE_RECOVERY: dict[str, str] = {
    "direct_energy": "final_demand",
    "shaping_adjustment": "allocation",
    **{line: "domestic_energy" for line in HEDGING_LINES},
    # (loss factor - 1) x the household energy cost, like the hedging uplifts.
    "procurement_losses": "domestic_energy",
    "cfd": "cfd_eligible",
    "allocated_ccs": "cfd_eligible",
    "renewables_obligation": "ro_eligible",
    "bsuos_constraints": "gsp_rate",
    "capacity_market": "capacity_market",
    "ldes_cap_and_floor": "gsp_rate",
    # Task SETTLE2050 (a 2050 bill only): recovered over CfD-eligible sales, as the bill does.
    "dpa": "cfd_eligible",
}
# New lines that are a kept part of a replaced line, or a balancing cost of the
# same kind, take that line's rule; every other new line takes the pool rate.
NEW_LINE_RECOVERY: dict[str, str] = {
    LINE_CM_NOT_ACQUIRED: "capacity_market",
    LINE_IC_RETRADING: "gsp_rate",
    LINE_REALTIME: "gsp_rate",
    LINE_FORECAST_ERROR: "gsp_rate",
    # The domestic levies moved to taxation take the household's whole line off it:
    # system GBP over domestic supply, like the lines they relieve.
    LINE_LEGACY_FIT: "domestic",
    LINE_LEGACY_WHD: "domestic",
    # The DPA comes back over the volume the kept remaining_ccs line is levied on (the CfD supplier obligation).
    LINE_DPA_RECEIPTS: "cfd_eligible",
}
#: F19 (27 Oct 2026): income lines recovered at the pool rate on the two-volume (2030) basis. The
#: gate fees are income of owned plant, so they reduce the cost the pool recovers and are spread as
#: it is (on the GSP charging volume at the household loss factor, at the pool's weighting), not flat
#: over final demand; a D1 (2050) bill keeps them flat, as before.
POOL_INCOME_LINES = (LINE_GATE_FEE_INCOME,)
#: The rule name the margin line reports: it is recovered at the tariff's own effective
#: rate (its household change over its system change), set per case in main().
TREASURY_MARGIN_RULE = "tariff_saving"
POOL = "pool"

PURCHASE_BASES = ("state_present_value", "market_value", "book_value", "no_support", "ro_pass_through")
THERMAL_BASES = ("book_value", "zero")


def household(system_gbp_2024: float, denominator_mwh: float, factor: float, mwh: float) -> float:
    """System GBP (model price year) to GBP per benchmark household (report price year)."""
    return float(system_gbp_2024) * float(factor) / float(denominator_mwh) * float(mwh)


def system(household_gbp: float, denominator_mwh: float, factor: float, mwh: float) -> float:
    return float(household_gbp) * float(denominator_mwh) / float(factor) / float(mwh)


#: F19 (7 Oct 2026): on the two-volume basis (F18) each rule is the recovery entry of the
#: bill line that carries it; the tariff converts through the bill's own entries.
RULE_ENTRY_LINES = {"gsp_rate": "bsuos_constraints", "cfd_eligible": "cfd", "ro_eligible": "renewables_obligation",
                    "capacity_market": "capacity_market", "domestic_energy": "direct_energy"}


def _entry_rate(conv: Mapping, line_id: str) -> float:
    """GBP per household per system GBP of one bill line, through its recovery entry."""
    return system_to_household_gbp(entry_for(conv["line_recovery"], line_id), 1.0, float(conv["mwh"]) * 1000.0)


def _rule_rate(conv: Mapping, rule: str, domestic_energy_uplift: float = 1.0) -> float:
    """GBP per benchmark household (report year) per system GBP (model year) under one rule."""
    base = float(conv["factor"]) * float(conv["mwh"])
    if rule == "final_demand":
        return base / float(conv["final_demand_mwh"])
    if conv.get("line_recovery") and rule in RULE_ENTRY_LINES:
        rate = _entry_rate(conv, RULE_ENTRY_LINES[rule])
        return rate * float(domestic_energy_uplift) if rule == "domestic_energy" else rate
    if rule == "gsp_rate":
        # A D1 bill (2050): the GSP-rate lines are flat over final demand, as before F18.
        return base / float(conv["final_demand_mwh"])
    if rule == "cfd_eligible":
        return base / float(conv["cfd_eligible_mwh"])
    if rule == "ro_eligible":
        return base / float(conv["ro_eligible_mwh"])
    if rule == "capacity_market":
        # The bill's Capacity Market line divides the book by the GSP chargeable
        # volume (item D1), not metered supply, times the household peak factor.
        # conversion() puts the bill's own Capacity Market denominator here.
        return (base / float(conv.get("cm_volume_mwh", conv["final_demand_mwh"]))
                * float(conv.get("cm_household_allocation_factor", 1.0)))
    if rule == "domestic_energy":
        return base / float(conv["final_demand_mwh"]) * float(domestic_energy_uplift)
    if rule == "domestic":
        if not conv.get("domestic_mwh"):
            raise ValueError("the domestic recovery rule needs the domestic supply volume in the conversion")
        return base / float(conv["domestic_mwh"])
    raise ValueError(f"unknown recovery rule {rule!r}")


def recovery_rates(private_components: pd.DataFrame, conv: Mapping,
                   kept_parts_system_gbp: Mapping[str, float] | None = None,
                   extra_replaced: Mapping[str, str] | None = None,
                   kept_replaced: Mapping[str, str] | None = None) -> dict:
    """The like-for-like allocation: a rate per replaced and new line, and the pool rate.

    A rate converts system GBP (model price year) to GBP per 3,100 kWh household
    (report price year, ex VAT).  ``replaced_system_gbp`` is each replaced line's
    system cost recovered from its household value at its own rule (zero for the
    shaping line, an allocation between consumers).  The pool rate is the
    household total of the replaced lines over their system total, less the kept
    parts that are re-added at their own rule (``kept_parts_system_gbp``,
    ``{new_line: system GBP}``: the Capacity Market agreements not acquired and the
    I1 interconnector re-trading), so the cost of owned generation falls on
    households in the share they paid for the market-construct lines it replaces.
    """
    private = private_components.loc[private_components["case_id"].eq("private")].set_index("component_id")[VALUE]
    energy = float(private.get("direct_energy", 0.0))
    shaping = float(private.get("shaping_adjustment", 0.0))
    # The hedging uplifts are rates on the household's energy cost, which the
    # shaping line has already made peak-heavy.
    uplift = (energy + shaping) / energy if energy else 1.0
    rule_rates = {rule: _rule_rate(conv, rule, uplift)
                  for rule in ("final_demand", "gsp_rate", "cfd_eligible", "ro_eligible", "capacity_market",
                               "domestic_energy")}
    if conv.get("domestic_mwh"):
        rule_rates["domestic"] = _rule_rate(conv, "domestic", uplift)
    replaced_hh, replaced_system, rates = {}, {}, {}
    # Decision 41 (c): a register-replaced line (Sizewell C's levy) is recovered over final demand.
    recovery = {**REPLACED_LINE_RECOVERY, **{line: "final_demand" for line in (extra_replaced or {})}}
    # F19 P3 decision 6: a line the case keeps is not replaced, so it is outside the pool.
    recovery = {line: rule for line, rule in recovery.items() if line not in (kept_replaced or {})}
    for line_id, rule in recovery.items():
        if line_id not in private.index:
            continue
        hh = float(private[line_id])
        replaced_hh[line_id] = hh
        if rule == "allocation":
            replaced_system[line_id] = 0.0
            rates[line_id] = None
        else:
            # F19: a line with its own recovery entry converts through it (the same as its rule's
            # line on the two-volume basis; the uplift stays on the hedging lines).
            own = (conv.get("line_recovery") and line_id in conv["line_recovery"]
                   and rule not in ("domestic_energy", "domestic", "final_demand"))
            rates[line_id] = _entry_rate(conv, line_id) if own else rule_rates[rule]
            replaced_system[line_id] = hh / rates[line_id]
    kept = {k: float(v) for k, v in (kept_parts_system_gbp or {}).items()}
    kept_hh = sum(amount * rule_rates[NEW_LINE_RECOVERY[line]] for line, amount in kept.items())
    pool_hh = sum(replaced_hh.values()) - kept_hh
    pool_system = sum(replaced_system.values()) - sum(kept.values())
    flat = rule_rates["final_demand"]
    pool = pool_hh / pool_system if pool_system > 0 and pool_hh > 0 else flat
    new_rates = {line: (rule_rates.get(NEW_LINE_RECOVERY[line], pool) if line in NEW_LINE_RECOVERY else pool)
                 for line in NEW_LINES}
    new_rules = {line: NEW_LINE_RECOVERY.get(line, POOL) for line in NEW_LINES}
    if conv.get("line_recovery"):
        new_rates.update({line: pool for line in POOL_INCOME_LINES})
        new_rules.update({line: POOL for line in POOL_INCOME_LINES})
    return {
        "pool_rate": pool, "flat_rate": flat, "pool_over_flat": pool / flat,
        "rule_rates": rule_rates, "domestic_energy_uplift": uplift,
        "replaced_rates": rates, "replaced_system_gbp": replaced_system,
        "replaced_gbp_per_3100kwh_ex_vat": replaced_hh,
        "kept_parts_system_gbp": kept, "new_line_rates": new_rates, "new_line_rules": new_rules,
    }


def line_rate(allocation: Mapping | None, conv: Mapping, line_id: str) -> float:
    """The rate a tariff line is recovered at; flat over final demand without an allocation."""
    if allocation and line_id in (allocation.get("new_line_rates") or {}):
        return float(allocation["new_line_rates"][line_id])
    if allocation and line_id in (allocation.get("replaced_rates") or {}) and allocation["replaced_rates"][line_id]:
        return float(allocation["replaced_rates"][line_id])
    return household(1.0, conv["final_demand_mwh"], conv["factor"], conv["mwh"])


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


# ── interconnectors (item D3) ─────────────────────────────────────────────────

#: Terms for the GB halves' purchase.  Mirrors public_ownership.acquisition.interconnectors in
#: config/defaults/base/public_ownership/acquisition.yaml.  F29 (user, 26 Oct 2026): workbook v2.0's Approach 2, the rent after
#: corporation tax at the workbook's post-tax operating rate for interconnectors (Legacy Fleet
#: column AF on their Sc1 hurdle: 6.1% - (9.63% - 8.40%) = 4.87%); the workbook has no book
#: value for them, so no capital allowances (declared).  The lever's low and high prices are a
#: declared +/- 1 pp on that rate (the workbook gives no range for interconnectors).
IC_ACQUISITION_DEFAULTS = {
    "purchase_price_basis": "market_value",   # PV of the GB-half rent after tax at the owners' rate
    "owner_real_rate": 0.0487,                # workbook v2.0's post-tax operating rate, interconnectors
    "corporation_tax_rate": 0.25,             # workbook v2.0 Assumptions B45
    # F28 (f): each cable's remaining years (its regime or its asset life, whichever ends first).
    "regime_years": 25.0,                     # Ofgem's cap-and-floor regime
    "asset_life_years": 40.0,                 # declared (as capacity_market_2050's interconnector life)
    "commissioning_year_by_cable": {"Britned": 2011, "ElecLink": 2022},  # DUKES 5.13, where the regime table has none
    "remaining_life_years_by_cable": {"IFA": 10.0},  # declared: IFA (1986) runs past a 40-year life
    "price_gbp": None,                        # purchase_price_basis: explicit
    "owner_real_rate_by_setting": {"low": 0.0587, "high": 0.0387},  # lever interconnector_price, +/- 1 pp (declared)
}
IC_PURCHASE_BASES = ("market_value", "explicit")
#: Keys F28 (f) retired, with what replaced them.
IC_RETIRED_KEYS = {"remaining_life_years": "F28 (f): each cable's remaining years come from regime_years, "
                                            "asset_life_years and the cable's first operation (and "
                                            "remaining_life_years_by_cable overrides)"}


class AnnuitySchedule:
    """How a purchase annuity runs off after the acquisition year (F28 (e), user decision 26 Oct 2026).

    Each acquired asset is repaid at the public rate over its own term, so the first year's
    charge falls away asset by asset: ``live(offset)`` is the share of the first year's charge
    still payable ``offset`` years after the acquisition (1 before any term ends, 0 after the
    last).  One term for every asset (the 30- and 15-year toggles) is the old window, 1 then 0.
    Built from each asset's term and first-year charge.
    """

    def __init__(self, charge_by_term: Mapping[float, float]):
        charges = {float(term): float(charge) for term, charge in charge_by_term.items() if float(charge) > 0.0}
        if not charges:
            terms = [float(term) for term in charge_by_term] or [30.0]
            charges = {max(terms): 1.0}
        total = sum(charges.values())
        self.shares = {term: charge / total for term, charge in sorted(charges.items())}

    @classmethod
    def fixed(cls, term_years: float) -> "AnnuitySchedule":
        return cls({float(term_years): 1.0})

    @classmethod
    def from_rows(cls, terms, charges) -> "AnnuitySchedule":
        """From each asset's term (years) and first-year charge (GBP)."""
        frame = pd.DataFrame({"term": pd.to_numeric(pd.Series(list(terms)), errors="coerce"),
                              "charge": pd.to_numeric(pd.Series(list(charges)), errors="coerce").fillna(0.0)}).dropna()
        return cls(frame.groupby("term")["charge"].sum().to_dict())

    @classmethod
    def from_record(cls, record) -> "AnnuitySchedule":
        return cls({float(term): float(share) for term, share in record})

    @classmethod
    def combine(cls, parts) -> "AnnuitySchedule":
        """Several schedules weighted by their first-year charges: ``[(schedule, charge), ...]``."""
        out: dict[float, float] = {}
        for schedule, charge in parts:
            for term, share in schedule.shares.items():
                out[term] = out.get(term, 0.0) + share * float(charge)
        return cls(out)

    def live(self, offset: float) -> float:
        if offset < 0:
            return 0.0
        return float(sum(share for term, share in self.shares.items() if offset < term))

    @property
    def last_term(self) -> float:
        return max(self.shares)

    @property
    def mean_term(self) -> float:
        return float(sum(term * share for term, share in self.shares.items()))

    def pv_per_gbp_of_price(self, rate: float, first_offset: int) -> float:
        """Present value at ``rate`` (year 0 undiscounted at the model year) of the annuities a pound of price
        buys, its price split across the terms as the charges are, from ``first_offset`` (years since the
        acquisition at the model year) on."""
        r = float(rate)
        price = {term: share / capital_recovery_factor(r, term) for term, share in self.shares.items()}
        total = sum(price.values())
        pv = 0.0
        for term, p in price.items():
            paid = np.arange(max(0, int(first_offset)), int(round(term)))
            pv += (p / total) * capital_recovery_factor(r, term) * float(((1.0 + r) ** -(paid - int(first_offset))).sum())
        return pv

    def record(self) -> list:
        return [[term, share] for term, share in self.shares.items()]


def interconnector_remaining_years(cables: Mapping, settings: Mapping, model_year: int) -> dict[str, float]:
    """Each cable's remaining years in the model year (F28 (f), user decision 26 Oct 2026).

    A cap-and-floor cable runs to the end of its regime (``regime_years`` from first operation)
    or of its asset life (``asset_life_years``), whichever comes first; any other cable to the end
    of its asset life.  First operation is the regime table's ``operational``, else
    ``commissioning_year_by_cable``; ``remaining_life_years_by_cable`` overrides.  Never below zero.
    """
    regime_years, life = float(settings["regime_years"]), float(settings["asset_life_years"])
    years_by_cable = {str(k): float(v) for k, v in dict(settings.get("commissioning_year_by_cable") or {}).items()}
    overrides = {str(k): float(v) for k, v in dict(settings.get("remaining_life_years_by_cable") or {}).items()}
    out = {}
    for name, spec in (cables or {}).items():
        name = str(name)
        if name in overrides:
            out[name] = overrides[name]
            continue
        built = spec.get("operational")
        built = float(built) if built is not None else years_by_cable.get(name)
        if built is None:
            out[name] = float("nan")
            continue
        end = built + life
        if str(spec.get("regime")) == "cap_and_floor":
            end = min(end, built + regime_years)
        out[name] = max(0.0, end - float(model_year))
    return out


def interconnector_acquisition_settings(block: Mapping | None) -> dict:
    """Validated interconnector acquisition terms (missing keys take :data:`IC_ACQUISITION_DEFAULTS`)."""
    block = dict(block or {})
    for key, note in IC_RETIRED_KEYS.items():
        if key in block:
            raise ValueError(f"public_ownership.acquisition.interconnectors.{key} is retired: {note}")
    merged = {**IC_ACQUISITION_DEFAULTS, **block}
    unknown = sorted(set(merged) - set(IC_ACQUISITION_DEFAULTS))
    if unknown:
        raise ValueError(f"public_ownership.acquisition.interconnectors has unknown keys: {unknown}")
    if merged["purchase_price_basis"] not in IC_PURCHASE_BASES:
        raise ValueError(f"interconnector purchase_price_basis must be one of {IC_PURCHASE_BASES}")
    if merged["purchase_price_basis"] == "explicit" and merged["price_gbp"] is None:
        raise ValueError("interconnector purchase_price_basis explicit needs price_gbp")
    if not float(merged["owner_real_rate"]) > 0 or not float(merged["regime_years"]) > 0 \
            or not float(merged["asset_life_years"]) > 0:
        raise ValueError("interconnector owner_real_rate, regime_years and asset_life_years must be positive")
    if not 0.0 <= float(merged["corporation_tax_rate"]) < 1.0:
        raise ValueError("interconnector corporation_tax_rate must be in [0, 1)")
    for setting, rate in dict(merged["owner_real_rate_by_setting"] or {}).items():
        if not 0.0 < float(rate) < 0.2:
            raise ValueError(f"interconnector owner_real_rate_by_setting.{setting} must be a real rate in (0, 0.2)")
    return merged


#: Regime-table value of ``public_ownership_treatment`` for a cable commissioned in the model year
#: (MaresConnect, 2030): the state buys its GB half at cost on completion, not at the present value
#: of a rent history it does not have (15 Oct 2026).
IC_NEW_BUILD_TREATMENT = "new_build_at_cost"
IC_TREATMENTS = (IC_NEW_BUILD_TREATMENT,)


def interconnector_new_build_floors(rule: Mapping) -> dict[str, float]:
    """GB-half floor (model-price GBP a year) of each cable the regime table marks new build at cost.

    ``rule`` is the bill's ``model_linked_rules.interconnector_cap_and_floor`` block.  The floor is
    the regime's cost-based guaranteed revenue for the GB half, indexed to the model price year as
    the bill indexes it (:func:`scripts.consumer_bill.project_future_bill.interconnector_pass_through`).
    """
    from scripts.consumer_bill.project_future_bill import interconnector_pass_through

    names = []
    for name, spec in (rule.get("cables") or {}).items():
        treatment = spec.get("public_ownership_treatment")
        if treatment is None:
            continue
        if treatment not in IC_TREATMENTS:
            raise ValueError(f"{name}: public_ownership_treatment must be one of {IC_TREATMENTS}, not {treatment!r}")
        if str(spec.get("regime")) != "cap_and_floor":
            raise ValueError(f"{name}: a cable bought at cost needs a cap-and-floor regime to price its floor")
        names.append(str(name))
    if not names:
        return {}
    audit = interconnector_pass_through({n: 0.0 for n in names}, rule, int(rule["index"]["to_price_year"]))
    return {str(row["interconnector"]): float(row["floor_gbp"]) for row in audit["by_cable"]}


#: The interconnector capacity the 2030 purchase bought (CP30's cable set at its nominal ratings), the baseline a later
#: year's added capacity is measured against (D2, user 14 Nov 2026).
IC_ACQUISITION_COMPOSITION = PROJECT_ROOT / "data" / "interconnectors" / "cp30_interconnector_composition.csv"


def interconnector_post_acquisition_new_build(capacity_mw: Mapping[str, float], gb_shares: Mapping[str, float],
                                              new_build_floors: Mapping[str, float], *,
                                              composition: str | Path = IC_ACQUISITION_COMPOSITION) -> dict:
    """D2 (user, 14 Nov 2026): the GB halves of capacity added after the purchase, built by the state at cost.

    A later year's network gives a cable more capacity than the purchase bought (FES's 20-22 GW against CP30's
    12.45 GW in 2030). The state, which owns every GB half after the purchase, builds the added GB halves; they are
    costed as the purchase costs a cable commissioned in its year (MaresConnect in 2030): the regime's cost-based
    floor at the public rate, here per GB-half MW of the cables the regime table marks new build at cost. Their rent
    already accrues to the tariff with the rest of the cable's. Returns the annual cost and its audit.
    """
    table = pd.read_csv(composition, comment="#")
    base = {str(r["name"]): float(r["cp30_mw"]) for _, r in table.iterrows()
            if str(r.get("action", "")) in ("keep", "add") and pd.notna(r["cp30_mw"])}
    floor_per_mw = [float(new_build_floors[c]) / (base[c] * float(gb_shares[c]))
                    for c in new_build_floors if base.get(c, 0.0) > 0 and float(gb_shares.get(c, 0.0)) > 0]
    added = {c: max(0.0, float(mw) - base.get(c, 0.0)) * float(gb_shares.get(c, 0.0))
             for c, mw in capacity_mw.items()}
    added = {c: v for c, v in added.items() if v > 1e-6}
    if added and not floor_per_mw:
        raise ValueError("capacity added after the purchase needs a cost: no cable is marked new build at cost")
    rate = float(np.mean(floor_per_mw)) if floor_per_mw else 0.0
    return {"gb_half_mw_by_cable": added, "gbp_per_gb_half_mw_year": rate,
            "annual_gbp": float(sum(added.values()) * rate),
            "basis": ("GB halves of capacity added after the purchase, built by the state at the regime's cost-based "
                      "floor per GB-half MW (as the cables commissioned in the purchase year), at the public rate")}


def interconnector_lines(rent_by_cable: Mapping[str, float], gb_shares: Mapping[str, float], *,
                         settings: Mapping, public_rate: float, term_years,
                         new_build_floor_gbp: Mapping[str, float] | None = None,
                         remaining_years: Mapping[str, float] | None = None) -> tuple[dict[str, float], dict]:
    """The two interconnector tariff lines (system GBP, model year) and their audit.

    ``rent_by_cable`` is each cable's whole Stage 1 congestion rent over the year;
    ``gb_shares`` the GB half of each (:func:`scripts.market.fti_settlement.gb_half_shares`).
    The foreign halves are paid their rent.  The GB halves are acquired in every
    tariff case: their rent accrues to the tariff and their purchase is annuitised
    at the public rate over ``term_years``: the price is the GB-half rent (floored
    at zero per cable) times the annuity factor at the owner's rate over the
    remaining life (``market_value``), or ``price_gbp`` apportioned by GB-half rent
    (``explicit``); a placeholder like every other acquisition price.

    ``new_build_floor_gbp`` names cables commissioned in the model year
    (:func:`interconnector_new_build_floors`): their GB half is new build, bought at
    cost on completion rather than valued on its rent.  The cost stands in as the
    present value of the regime's floor at the public rate over the remaining life
    (the regime's 25 years): the floor is Ofgem's cost-based revenue at a cost of
    debt, close to the public rate.  They join neither the market value nor the
    ``explicit`` price, which are for the existing halves.

    F28 (f): ``remaining_years`` (:func:`interconnector_remaining_years`) gives each cable its own
    valuation horizon; ``term_years`` ``remaining_life`` (F28 (e)) repays each half over it, a
    number over that term for every cable.
    """
    rent = pd.Series({str(k): float(v) for k, v in rent_by_cable.items()}, dtype=float)
    unknown = sorted(set(rent.index) - set(gb_shares))
    if unknown:
        raise ValueError(f"interconnectors with congestion rent but no declared GB share: {unknown}")
    share = pd.Series({c: float(gb_shares[c]) for c in rent.index}, dtype=float)
    gb_half, foreign_half = rent * share, rent * (1.0 - share)
    r = float(settings["owner_real_rate"])
    years = pd.Series({c: float((remaining_years or {}).get(c, np.nan)) for c in rent.index}, dtype=float)
    unknown_years = sorted(years.index[years.isna() & (share > 0.0)])
    if unknown_years:
        raise ValueError(f"no remaining years for the GB halves of {unknown_years}: give their first operation "
                         "(acquisition.interconnectors.commissioning_year_by_cable) or an override")
    years = years.fillna(0.0)
    annuity_factor = (1.0 - (1.0 + r) ** -years) / r
    floors = {c: float(v) for c, v in (new_build_floor_gbp or {}).items() if c in rent.index}
    new_build = rent.index.isin(list(floors))
    if any(share[c] <= 0.0 for c in floors):
        raise ValueError(f"cables bought at cost need a GB half: {sorted(c for c in floors if share[c] <= 0.0)}")
    if str(settings["purchase_price_basis"]) == "explicit":
        weights = gb_half.clip(lower=0.0).where(~new_build, 0.0)
        price = (weights / weights.sum() * float(settings["price_gbp"])) if weights.sum() > 0 else weights
    else:
        # F29: the owners' value after corporation tax (workbook v2.0's Approach 2, no allowances).
        after_tax = 1.0 - float(settings.get("corporation_tax_rate", 0.0))
        price = (gb_half.clip(lower=0.0) * after_tax * annuity_factor).where(~new_build, 0.0)
    pr = float(public_rate)
    public_factor = {c: ((1.0 - (1.0 + pr) ** -years[c]) / pr if abs(pr) > 1e-12 else years[c]) for c in floors}
    at_cost = pd.Series({c: floors[c] * public_factor[c] for c in floors}, dtype=float)
    price = price.add(at_cost, fill_value=0.0)
    # F28 (e): each half over its own remaining years (at least one), or one term for every half.
    repay = (years.clip(lower=1.0) if str(term_years) == TERM_REMAINING_LIFE
             else pd.Series(float(term_years), index=years.index))
    charge = pd.Series({c: float(price.get(c, 0.0)) * capital_recovery_factor(pr, float(repay[c])) for c in price.index},
                       dtype=float)
    annuity = float(charge.sum())
    schedule = AnnuitySchedule.from_rows(repay.reindex(charge.index).to_numpy(), charge.to_numpy())
    lines = {LINE_IC_FOREIGN_RENT: float(foreign_half.sum()), LINE_IC_ACQUISITION: annuity}
    detail = {
        "interconnector_gb_halves_acquired": True,
        "interconnector_rent_gbp": float(rent.sum()),
        "interconnector_rent_by_cable_gbp": {k: float(v) for k, v in rent.items()},
        "interconnector_gb_share": {k: float(v) for k, v in share.items()},
        "interconnector_gb_half_rent_gbp": float(gb_half.sum()),
        "interconnector_foreign_half_rent_gbp": float(foreign_half.sum()),
        "interconnector_gb_half_rent_to_tariff_gbp": float(gb_half.sum()),
        "interconnector_acquisition_price_gbp": float(price.sum()),
        "interconnector_acquisition_price_by_cable_gbp": {k: float(v) for k, v in price.items()},
        "interconnector_remaining_years_by_cable": {k: float(v) for k, v in years.items()},
        "interconnector_annuity_schedule": schedule.record(),
        "interconnector_new_build_at_cost": {
            c: {"floor_gbp": floors[c], "price_gbp": float(at_cost[c]), "public_rate_annuity_factor": public_factor[c],
                "basis": "GB half new build in the model year, bought at cost on completion: the present value of "
                         "the regime's floor at the public rate over the remaining life"}
            for c in floors},
        "interconnector_acquisition_terms": {**{k: v for k, v in settings.items()}, "public_rate": float(public_rate),
                                             "term_years": (TERM_REMAINING_LIFE if str(term_years) == TERM_REMAINING_LIFE
                                                            else float(term_years)),
                                             "status": "estimated price (the GB-half rent's present value at an owner's rate), flagged like the other acquisition prices"},
    }
    return lines, detail


def interconnector_cm_acquired_share(capacity_by_cable: Mapping[str, float], gb_shares: Mapping[str, float]) -> float:
    """Capacity-weighted GB share of the interconnectors' Capacity Market agreements.

    The contract book carries interconnector revenue by technology, not by cable
    (``cm_revenue_by_technology.csv``), so the acquired share is apportioned by
    each cable's rating: a declared approximation.  Cables absent from the share
    table count as not acquired.
    """
    cap = pd.Series({str(k): float(v) for k, v in capacity_by_cable.items()}, dtype=float).clip(lower=0.0)
    if cap.sum() <= 0:
        return 0.0
    share = pd.Series({c: float(gb_shares.get(c, 0.0)) for c in cap.index}, dtype=float)
    return float((cap * share).sum() / cap.sum())


# ── the revenue requirement ───────────────────────────────────────────────────

def tariff_case_ids(register: pd.DataFrame) -> list[str]:
    """The cases in a register that take the acquisition tariff: legacy fleet acquired, market dispatch."""
    return tariff_case_ids_from_settings(
        {str(case_id): case_settings(rows) for case_id, rows in register.groupby("case_id", sort=False)})


def tariff_case_ids_from_settings(settings_by_case: Mapping[str, Mapping[str, str]]) -> list[str]:
    """:func:`tariff_case_ids` from each case's lever settings, in register order (the case-settings table)."""
    out = []
    for case_id, levers in settings_by_case.items():
        spec = spec_for_row(levers)
        if spec.acquires_legacy() and levers.get(LEVER_DISPATCH_INSTITUTION, INSTITUTION_MARKET) == INSTITUTION_MARKET:
            out.append(str(case_id))
    if not out and NATIONALISED_CASE in settings_by_case:
        out.append(NATIONALISED_CASE)
    return out


def select_tariff_cases(available: list[str], requested: list[str] | None) -> list[str]:
    """The tariff cases to price: every one in the register, or the ones asked for (item BM1, P3).

    A tariff balancing group prices its bid case's fleet family; a requested case the
    register does not tariff is refused rather than skipped.
    """
    if requested is None:
        return list(available)
    missing = [case for case in requested if case not in available]
    if missing or not requested:
        raise ValueError(f"--cases {list(requested)} must name tariff cases in the register; missing {missing}; "
                         f"the register tariffs {list(available)}")
    return [case for case in available if case in requested]


#: Levers that act only in the later (vintaged) register and leave the earlier purchase as it was: a case that
#: moves only these carries the purchase of the case it otherwise equals (FES 2050 decision 73's P3 sensitivity).
CARRY_NEUTRAL_LEVERS = frozenset({"capacity_market_design"})


def carried_case_id(case_id: str, carried_cases: Mapping, settings: pd.DataFrame | None) -> str | None:
    """The earlier-purchase case ``case_id`` carries: its own, else the one carried case at the same settings.

    ``settings`` is the register's case-settings table (``case_id`` plus every lever). Two settings are the same
    when every lever but :data:`CARRY_NEUTRAL_LEVERS` matches; more than one match is refused, none returns None.
    """
    if case_id in carried_cases:
        return case_id
    if settings is None or case_id not in set(settings["case_id"]):
        return None
    table = settings.set_index("case_id").astype(str)
    compared = [c for c in table.columns if c not in CARRY_NEUTRAL_LEVERS]
    own = table.loc[case_id, compared]
    matches = [c for c in carried_cases if c in table.index and table.loc[c, compared].equals(own)]
    if len(matches) > 1:
        raise ValueError(f"{case_id}: several carried cases share its settings ({matches}); name one")
    return matches[0] if matches else None


FIXED_OM_COMPONENTS =("annual_site_fixed_om_gbp", "annual_generator_tnuos_gbp", "annual_insurance_gbp")


def fixed_om_lines(rows: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    """Site fixed O&M, generator TNUoS and insurance on the masked rows (item D4(a)).

    The register's ``annual_fixed_om_gbp`` is their sum; the three parts must add
    up to it row by row, or this refuses.  A register written before item D4(a)
    carries no parts, and its fixed O&M is all site O&M.
    """
    total = _num(rows, "annual_fixed_om_gbp")[mask]
    if not all(c in rows.columns for c in FIXED_OM_COMPONENTS):
        return {LINE_FIXED_OM: float(total.sum()), LINE_GENERATOR_TNUOS: 0.0, LINE_INSURANCE: 0.0}
    site, tnuos, insurance = (_num(rows, c)[mask] for c in FIXED_OM_COMPONENTS)
    gap = (site + tnuos + insurance - total).abs()
    if float(gap.max() if len(gap) else 0.0) > 1.0:
        raise ValueError(f"site fixed O&M + generator TNUoS + insurance differ from annual_fixed_om_gbp by up to "
                         f"GBP {float(gap.max()):,.0f} on a row; the register's components do not add up")
    return {LINE_FIXED_OM: float(site.sum()), LINE_GENERATOR_TNUOS: float(tnuos.sum()),
            LINE_INSURANCE: float(insurance.sum())}


def _is_vintaged_register(rows: pd.DataFrame) -> bool:
    """True for a vintaged (2040 on) register's rows (task SETTLE2050)."""
    from scripts.public_ownership.model_year import is_vintaged_model_year

    if "model_year" not in rows.columns or rows.empty:
        return False
    years = pd.to_numeric(rows["model_year"], errors="coerce").dropna().unique()
    return len(years) == 1 and is_vintaged_model_year(int(years[0]))


def revenue_requirement(register: pd.DataFrame, *, realtime_correction_gbp: float = 0.0,
                        interconnector_retrading_gbp: float = 0.0,
                        interconnector_trade_gbp: float | None = None,
                        legacy_capital_override: pd.Series | None = None,
                        case_id: str = NATIONALISED_CASE,
                        interconnector: Mapping[str, float] | None = None,
                        interconnector_detail: Mapping | None = None,
                        carried_acquisition: Mapping | None = None,
                        carried_contracts=(), rab_at_rav: bool = False,
                        rav_rate: float | None = None,
                        carried_mw: Mapping[str, float] | None = None,
                        dpa_paid_by_kept_line: bool = False,
                        rab_paid_by_kept_line: bool = False,
                        legacy_gate_fee_gbp: pd.Series | None = None,
                        interconnector_post_acquisition_gbp: float = 0.0) -> tuple[dict[str, float], dict]:
    """System GBP (model price year) by new tariff line, and diagnostics.

    F19 P3 decision 4 (user, 6 Oct 2026): with ``dpa_paid_by_kept_line`` (the bill keeps
    its ``remaining_ccs`` line, the Net Zero Teesside DPA) the plant paid as today is paid
    its register revenue less its ``dpa_support_gbp``, so the DPA is paid once.

    F19 (user, 6 Oct 2026, "buy it, credit gate fees in the tariff"): ``legacy_gate_fee_gbp``
    (by asset_id, the legacy valuation's ``gate_fee_model_year_gbp``: the gate fees the
    purchase price counts, capped at the UK's throughput) is credited on owned, costed
    legacy rows as ``nationalised_efw_gate_fee_income`` where the register books no gate
    fee itself (a 2030 register; a 2050 one books ``efw_gate_fee_revenue_gbp``).

    FES 2050 decision 55, with a carried acquisition only: rows whose contract is in
    ``carried_contracts`` (``HPC_UNIT_1``: bought in the 2030 purchase) carry no capital (it is
    in the carried annuity), and with ``rab_at_rav`` the state acquires Sizewell C's RAB rows at
    their regulated asset value (the statutory route): their capital is the RAV annuitised at
    ``rav_rate`` over the remaining life, the RAV being the present value at the RAB's own rate
    of the capital its allowance pays (:func:`regulated_asset_value`). :func:`check_acquisition_pricing`
    then refuses any asset both acquired and paid as new build, and any state-held asset with
    no price.

    ``carried_acquisition`` (item Y12, decision 2; one case of
    ``acquisition_carry_forward.build``) is the purchase made in an earlier year: the
    register's legacy rows then carry no capital (operating cost only, their capital is in
    the carried annuity) and the acquisition lines are the carried ones, at their original
    terms, whatever survives.  Without it (the CP30 year) nothing changes.

    ``legacy_capital_override`` (indexed by asset_id) replaces the register's
    legacy acquisition charge row by row; item N4 uses it for the price-basis
    and term variants.

    **Cross-border energy.**  ``interconnector_trade_gbp``, when given, is the
    measured net payment to foreign counterparties: link flow times the foreign
    price, summed over the year, negative where the system is a net exporter.
    That is what a public buyer actually pays abroad, and it replaces both the
    register's import revenue (which values imported energy at the *domestic*
    wholesale price GB consumers paid, not at what left the country) and the
    day-ahead re-trading cost (a market artefact: a single buyer pays for the
    energy that physically flows, not for changing its own position).  Without
    it the two register-based lines stand, and both are declared approximations.

    ``interconnector`` (:func:`interconnector_lines`, item D3) adds the rent owed
    to cable owners the state does not buy and the GB halves' acquisition annuity;
    it needs the measured link trade, because under the register basis imports are
    already valued at the GB price, rent included.
    """
    nat = register.loc[register["case_id"].eq(case_id)].copy()
    if nat.empty:
        raise ValueError(f"register has no {case_id} rows")
    private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
    nat = nat.set_index("asset_id")
    if not nat.index.equals(private.index) and set(nat.index) != set(private.index):
        raise ValueError("nationalised and private rows do not cover the same assets")
    private = private.reindex(nat.index)
    owned = nat["state_owned"].astype(bool)
    resolved = nat["annual_required_revenue_gbp"].notna()
    legacy = nat["finance_vintage"].astype(str).eq("legacy")
    scope = nat["legacy_scope_group"].astype(str)
    capital = _num(nat, "annual_capital_recovery_gbp")
    if legacy_capital_override is not None:
        override = legacy_capital_override.reindex(nat.index)
        capital = capital.where(override.isna(), override)
    register_legacy_capital = None
    if carried_acquisition is not None:
        if legacy_capital_override is not None:
            raise ValueError("a carried acquisition replaces the legacy capital; it cannot take an override too")
        register_legacy_capital = float(capital[legacy & owned & resolved].sum())
        capital = capital.where(~legacy, 0.0)

    costed = owned & resolved
    # 15 Nov 2026: plant the case owns but cannot cost would be paid its market revenue as if private (the 2040
    # OCGTs and gas engines were, beside their carried purchase annuity). Refused unless declared negligible.
    uncosted = owned & ~resolved & ~nat["carrier"].astype(str).isin(["EU_market", "EU_import"])
    undeclared = sorted(set(nat.loc[uncosted, "carrier"].astype(str)) - set(OWNED_UNCOSTED_DECLARED))
    if undeclared:
        raise ValueError(f"{case_id}: owns plant it cannot cost ({undeclared}, "
                         f"{float(_num(nat, 'capacity_mw')[uncosted].sum()):,.0f} MW): it would be paid as today; give "
                         "the carrier a cost row, or declare it in OWNED_UNCOSTED_DECLARED with its size")
    decision55 = {}
    if carried_acquisition is not None and (carried_contracts or rab_at_rav):
        contract = (nat["cfd_contract_id"].fillna("").astype(str).str.split("#", n=1).str[0]
                    if "cfd_contract_id" in nat.columns else pd.Series("", index=nat.index))
        carried_rows = owned & ~legacy & contract.isin([str(c) for c in carried_contracts])
        # Review 2 (3 Oct 2026): ``carried_mw`` ({contract: MW the earlier purchase bought}) carries exactly
        # that capacity: the labels fill a contract with whole rows (HPC unit 1's tag holds 1,997 MW against
        # its 1,630), so a contract's rows carry the bought share of their capital and the rest stays new build.
        share = pd.Series(np.where(carried_rows, 1.0, 0.0), index=nat.index)
        if carried_mw:
            mw = pd.to_numeric(nat["capacity_mw"], errors="coerce").fillna(0.0)
            for name, bought in carried_mw.items():
                rows = carried_rows & contract.eq(str(name))
                tagged = float(mw[rows].sum())
                if tagged > 0:
                    share = share.where(~rows, min(1.0, float(bought) / tagged))
            decision55["carried_from_2030_tagged_mw"] = float(mw[carried_rows].sum())
            decision55["carried_from_2030_bought_mw"] = {str(k): float(v) for k, v in carried_mw.items()}
        decision55["carried_from_2030_mw"] = float((nat["capacity_mw"].astype(float) * share)[carried_rows].sum())
        decision55["carried_from_2030_capital_set_aside_gbp"] = float((capital * share)[carried_rows & resolved].sum())
        decision55["carried_from_2030_private_capital_gbp"] = float(
            (_num(private, "annual_capital_recovery_gbp") * share)[carried_rows & resolved].sum())
        capital = capital * (1.0 - share)
        carried_rows = carried_rows & share.ge(1.0)          # wholly carried: no capital at all
        rav_rows = pd.Series(False, index=nat.index)
        if rab_at_rav:
            rav_rows = owned & ~legacy & nat["support_type"].fillna("").astype(str).eq("RAB") & resolved
            if rav_rows.any():
                if rav_rate is None:
                    raise ValueError("Sizewell C at its RAV needs the public rate (rav_rate)")
                rav = regulated_asset_value(private.loc[rav_rows])
                remaining = pd.to_numeric(private.loc[rav_rows, "remaining_life_years"], errors="coerce").clip(lower=1.0)
                annuity = pd.Series([v * capital_recovery_factor(float(rav_rate), float(n))
                                     for v, n in zip(rav, remaining)], index=rav.index)
                decision55.update({"sizewell_c_rav_gbp": float(rav.sum()), "sizewell_c_rav_rate": float(rav_rate),
                                   "sizewell_c_rav_annuity_gbp": float(annuity.sum()),
                                   "sizewell_c_new_build_capital_replaced_gbp": float(capital[rav_rows].sum()),
                                   "sizewell_c_mw": float(nat.loc[rav_rows, "capacity_mw"].sum())})
                capital = capital.where(~rav_rows, annuity.reindex(capital.index))
        check_acquisition_pricing(nat, capital, owned=costed, legacy=legacy, carried=carried_rows,
                                  rav=rav_rows, case_id=case_id)
    imports = nat["carrier"].astype(str).isin(["EU_market", "EU_import"])
    # Owned rows without a cost mapping (CHP, oil) and unowned GB rows are paid
    # what they are paid today; imports are valued at what GB paid them.
    paid_as_today = ~costed & ~imports
    lines = {
        LINE_LEGACY_RENEWABLES: float(capital[costed & legacy & scope.eq(GROUP_VALUED)].sum()),
        LINE_LEGACY_THERMAL: float(capital[costed & legacy & ~scope.isin([GROUP_VALUED, GROUP_EXCLUDED])].sum()),
        LINE_NEW_BUILD: float(capital[costed & ~legacy].sum()),
        # Item D4(a): the register folds generator TNUoS and insurance into fixed
        # O&M and carries the parts beside it; the tariff shows the three apart.
        **fixed_om_lines(nat, costed),
        LINE_VARIABLE_OM: float(_num(nat, "annual_variable_om_gbp")[costed].sum()),
        LINE_FUEL: float(_num(nat, "fuel_carbon_gbp")[costed].sum()),
        # Item OH1.  The tariff is built from components, not from
        # annual_required_revenue_gbp, so the holding company needs its own
        # line or it would be silently dropped here.  Under P3 it covers the
        # acquired legacy fleet too, which is why it is summed over every
        # owned and costed row rather than over the new-build cohort.
        LINE_STATE_OVERHEAD: float(_num(nat, "annual_state_overhead_gbp")[costed].sum()),
        LINE_IMPORTS: float((_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp"))[imports].sum()),
        # Money audit (13 Nov 2026): plant the state does not own is paid as today, less the support a line the
        # bill keeps already pays: the DPA (remaining_ccs, F19 P3 decision 4) and Sizewell C's RAB (sizewell_c_rab,
        # kept when the state does not own Sizewell C). Before it, the RAB was paid twice in 2040 and 2050.
        LINE_UNACQUIRED: float((_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp")
                                - (_num(private, "dpa_support_gbp") if dpa_paid_by_kept_line else 0.0)
                                - (_num(private, "rab_support_gbp") if rab_paid_by_kept_line else 0.0))[paid_as_today].sum()),
        LINE_REALTIME: float(realtime_correction_gbp),
        LINE_IC_RETRADING: float(interconnector_retrading_gbp),
    }
    # Item G3, lever L19: the residual owed to former owners for cancelled
    # contracts. On the bill it is a tariff line; on the Exchequer it is a
    # transfer, reported but not charged.
    residual = _num(nat, "annual_cancelled_contract_residual_gbp").fillna(0.0)
    route = (nat["cancelled_residual_route"].fillna("").astype(str) if "cancelled_residual_route" in nat.columns
             else pd.Series("", index=nat.index))
    lines[LINE_CANCELLED_RESIDUAL] = float(residual[costed & legacy & route.eq("bills")].sum())
    exchequer_residual = float(residual[costed & legacy & route.eq("exchequer")].sum())
    # User decision 2: the BECCS capture leg is paid by the carbon CfD from the
    # Exchequer in every case, so it comes off the tariff (the register's own
    # carbon_removal_cfd_revenue_gbp on the owned rows).  Under beccs_removal_funding =
    # bill_levy the bill carries the whole plant, as it would the levy.
    carbon_cfd = _num(nat, "carbon_removal_cfd_revenue_gbp")[costed]
    beccs_on_bill = str(case_settings(register.loc[register["case_id"].eq(case_id)]).get(
        "beccs_removal_funding", "exchequer")) == "bill_levy"
    lines[LINE_BECCS_CAPTURE] = 0.0 if beccs_on_bill else -float(carbon_cfd.sum())
    # Independent review (15 Nov 2026): owned, costed plant on a DPA the bill's kept remaining_ccs line pays (Net Zero
    # Teesside in 2030) keeps its contract, so the state receives the DPA; the tariff costs the plant in full, so the
    # receipt is credited back. Without it the 2030 S3 bill paid the DPA twice (GBP 403 m a year).
    dpa_owned = _num(private, "dpa_support_gbp")[costed] if dpa_paid_by_kept_line else pd.Series(dtype=float)
    lines[LINE_DPA_RECEIPTS] = -float(dpa_owned.sum())
    # Lever tax_equivalent: the register folds it into required revenue, which the
    # tariff (built from components) does not read, so it is its own line.
    lines[LINE_TAX_EQUIVALENT] = float(_num(nat, "annual_tax_equivalent_gbp")[costed].sum())
    import_basis = "register_domestic_wholesale_revenue"
    if interconnector_trade_gbp is not None:
        # Measured link trade replaces both register-based cross-border lines.
        lines[LINE_IMPORTS] = float(interconnector_trade_gbp)
        lines[LINE_IC_RETRADING] = 0.0
        import_basis = "measured_link_trade_at_the_foreign_price"
    ic_lines = {LINE_IC_FOREIGN_RENT: 0.0, LINE_IC_ACQUISITION: 0.0}
    if interconnector:
        if interconnector_trade_gbp is None:
            raise ValueError("interconnector rent lines need the measured link trade: under the register basis "
                             "imports are valued at the GB price, rent included, so the rent would count twice")
        unknown = sorted(set(interconnector) - set(ic_lines))
        if unknown:
            raise ValueError(f"unknown interconnector lines: {unknown}")
        ic_lines.update({k: float(v) for k, v in interconnector.items()})
    lines.update(ic_lines)
    if carried_acquisition is not None:
        # Item Y12: the purchase is the earlier year's, at its own terms.
        gross = dict(carried_acquisition["gross_lines_model_year_gbp"])
        for line in (LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, LINE_CANCELLED_RESIDUAL):
            lines[line] = float(gross.get(line, 0.0))
        if interconnector:
            # D2 (user, 14 Nov 2026): plus the GB halves the state built after the purchase, at cost.
            lines[LINE_IC_ACQUISITION] = (float(gross.get(LINE_IC_ACQUISITION, 0.0))
                                          + float(interconnector_post_acquisition_gbp))
        exchequer_residual = float(carried_acquisition.get("cancelled_residual_exchequer_model_year_gbp", 0.0))
    diagnostics = {
        "import_basis": import_basis,
        # F19 P3 decision 4: the DPA the kept remaining_ccs line pays (not paid again as today).
        "dpa_paid_through_kept_line_gbp": (float(_num(private, "dpa_support_gbp")[paid_as_today].sum())
                                           if dpa_paid_by_kept_line else 0.0),
        "dpa_on_owned_costed_rows_gbp": float(_num(private, "dpa_support_gbp")[costed].sum()),
        "dpa_receipts_legacy_gbp": (float(_num(private, "dpa_support_gbp")[costed & legacy].sum())
                                    if dpa_paid_by_kept_line else 0.0),
        # Money audit (13 Nov 2026): the RAB the kept sizewell_c_rab line pays on plant paid as today.
        "rab_paid_through_kept_line_gbp": (float(_num(private, "rab_support_gbp")[paid_as_today].sum())
                                           if rab_paid_by_kept_line and "rab_support_gbp" in private.columns else 0.0),
        # The split of the lines summed over all owned plant, for the three-part split (F19 P3 decision 1).
        "state_overhead_legacy_gbp": float(_num(nat, "annual_state_overhead_gbp")[costed & legacy].sum()),
        "carbon_removal_cfd_legacy_gbp": float(_num(nat, "carbon_removal_cfd_revenue_gbp")[costed & legacy].sum()),
        "ancillary_services_legacy_gbp": float(_num(nat, "ancillary_services_revenue_gbp")[costed & legacy].sum()),
        "battery_subhourly_legacy_gbp": float(_num(nat, "battery_subhourly_revenue_gbp")[costed & legacy].sum()),
        **dict(interconnector_detail or {}),
        "cancelled_contract_residual_exchequer_transfer_gbp": exchequer_residual,
        "cancelled_contract_residual_on_bills_gbp": lines[LINE_CANCELLED_RESIDUAL],
        "import_register_domestic_revenue_gbp": float(
            (_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp"))[imports].sum()),
        "owned_costed_assets": int(costed.sum()), "owned_costed_capacity_mw": float(nat.loc[costed, "capacity_mw"].sum()),
        "paid_as_today_capacity_mw": float(nat.loc[paid_as_today, "capacity_mw"].sum()),
        "paid_as_today_carriers": sorted(nat.loc[paid_as_today, "carrier"].astype(str).unique()),
        "import_mwh": float(_num(private, "dispatch_mwh")[imports].sum()),
        # Register-side private payments by instrument, for the N7 bridge.
        "private_wholesale_revenue_gbp": float(_num(private, "wholesale_revenue_gbp").sum()),
        "private_balancing_settlement_gbp": float(_num(private, "balancing_settlement_gbp").sum()),
        "private_cfd_support_gbp": float(_num(private, "cfd_support_gbp").sum()),
        "private_roc_support_gbp": float(_num(private, "roc_support_gbp").sum()),
        "acquisition_price_gbp": float(_num(nat, "acquisition_price_gbp")[costed & legacy].sum()),
        "acquisition_price_renewables_nuclear_gbp": float(_num(nat, "acquisition_price_gbp")[costed & legacy & scope.eq(GROUP_VALUED)].sum()),
        "acquisition_price_thermal_storage_gbp": float(
            _num(nat, "acquisition_price_gbp")[costed & legacy & ~scope.isin([GROUP_VALUED, GROUP_EXCLUDED])].sum()),
        "legacy_thermal_fixed_om_gbp": float(_num(nat, "annual_fixed_om_gbp")[costed & legacy & scope.eq("thermal")].sum()),
        "legacy_thermal_fuel_carbon_gbp": float(_num(nat, "fuel_carbon_gbp")[costed & legacy & scope.eq("thermal")].sum()),
        "legacy_thermal_private_energy_margin_gbp": float(
            (_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp") - _num(private, "fuel_carbon_gbp")
             - _num(private, "annual_fixed_om_gbp") - _num(private, "annual_variable_om_gbp"))[costed & legacy & scope.eq("thermal")].sum()),
        # Register-side attribution (item N7), GB owned-and-costed rows only.
        "private_payments_to_owned_plant_gbp": float(
            (_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp"))[costed].sum()),
        "legacy_operating_margin_gbp": float(
            (_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp") - _num(private, "fuel_carbon_gbp")
             - _num(private, "annual_fixed_om_gbp") - _num(private, "annual_variable_om_gbp"))[costed & legacy].sum()),
        "new_build_private_rent_gbp": float(
            (_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp") - _num(private, "fuel_carbon_gbp")
             - _num(private, "annual_required_revenue_gbp"))[costed & ~legacy].sum()),
        "new_build_capital_private_gbp": float(_num(private, "annual_capital_recovery_gbp")[costed & ~legacy].sum()),
        # Item MR4: Capacity Market revenue the register books, split by whether the
        # state acquires and costs the plant (the rest stays on the Capacity Market).
        "capacity_market_revenue_register_gbp": float(_num(private, "capacity_market_revenue_gbp").sum()),
        "capacity_market_revenue_owned_costed_gbp": float(_num(private, "capacity_market_revenue_gbp")[costed].sum()),
        "capacity_market_revenue_paid_as_today_gbp": float(
            _num(private, "capacity_market_revenue_gbp")[paid_as_today].sum()),
        # For the register ledger's declared residual (item N7).
        "private_market_and_support_revenue_all_gbp": float(
            (_num(private, "market_revenue_gbp") + _num(private, "support_revenue_gbp")).sum()),
        # F19 P3 decision 1: the acquired fleet's share of it (self-insurance), for the three-part split.
        "owned_om_fuel_case_minus_private_legacy_gbp": float(
            ((_num(nat, "annual_fixed_om_gbp") + _num(nat, "annual_variable_om_gbp") + _num(nat, "fuel_carbon_gbp"))
             - (_num(private, "annual_fixed_om_gbp") + _num(private, "annual_variable_om_gbp")
                + _num(private, "fuel_carbon_gbp")))[costed & legacy].sum()),
        "owned_om_fuel_case_minus_private_gbp": float(
            ((_num(nat, "annual_fixed_om_gbp") + _num(nat, "annual_variable_om_gbp") + _num(nat, "fuel_carbon_gbp"))
             - (_num(private, "annual_fixed_om_gbp") + _num(private, "annual_variable_om_gbp")
                + _num(private, "fuel_carbon_gbp")))[costed].sum()),
        "private_new_build_state_overhead_gbp": float(_num(private, "annual_state_overhead_gbp")[costed & ~legacy].sum()),
        # User decision 2: the carbon CfD the private owners received from the Exchequer
        # (not a bill line), and the one the state company receives.
        "private_new_build_carbon_removal_cfd_gbp": float(
            _num(private, "carbon_removal_cfd_revenue_gbp")[costed & ~legacy].sum()),
        # 8 Nov 2026 (user: P3 buys Drax BECCS, the decision-2 booking mirrored): the carbon CfD the
        # Exchequer paid the acquired plant's former owners, outside legacy_operating_margin_gbp.
        "private_legacy_carbon_removal_cfd_gbp": float(
            _num(private, "carbon_removal_cfd_revenue_gbp")[costed & legacy].sum()),
        "carbon_removal_cfd_case_gbp": float(carbon_cfd.sum()),
        "beccs_removal_funding": "bill_levy" if beccs_on_bill else "exchequer",
    }
    if _is_vintaged_register(private) and "dpa_support_gbp" in private.columns:
        # Task SETTLE2050: the 2050 DPA payments the register books (the bill's dpa line), for the
        # register ledger. A 2030 register is never vintaged, so its diagnostics are unchanged.
        diagnostics["private_dpa_support_gbp"] = float(_num(private, "dpa_support_gbp").sum())
    support = {}
    # Any model year whose register books a support line (public_ownership.support_lines: 2050, or a 2030
    # study that turns on the ancillary and battery lines); a register without the columns is unchanged.
    if any(column in private.columns or column in nat.columns for column in SUPPORT_LINE_COLUMNS):
        # FES 2050 decision 58: the support lines the register books. Ancillary services (paid out of
        # the kept BSUoS lines) and gate fees (paid by waste disposal) on owned plant are the state's
        # income, credited to the tariff; the top-up line is replaced (register_replaced_lines).
        if "ancillary_services_revenue_gbp" in nat.columns:
            lines[LINE_ANCILLARY_INCOME] = -float(_num(nat, "ancillary_services_revenue_gbp")[costed].sum())
            support["private_ancillary_services_gbp"] = float(_num(private, "ancillary_services_revenue_gbp").sum())
        if "efw_gate_fee_revenue_gbp" in nat.columns:
            lines[LINE_GATE_FEE_INCOME] = -float(_num(nat, "efw_gate_fee_revenue_gbp")[costed].sum())
            support["private_efw_gate_fee_gbp"] = float(_num(private, "efw_gate_fee_revenue_gbp").sum())
            support["efw_gate_fee_legacy_gbp"] = float(_num(nat, "efw_gate_fee_revenue_gbp")[costed & legacy].sum())
        if "battery_subhourly_revenue_gbp" in nat.columns:
            # Decision 67: the owned batteries' sub-hourly revenue is the state's income, credited to the tariff.
            lines[LINE_SUBHOURLY_INCOME] = -float(_num(nat, "battery_subhourly_revenue_gbp")[costed].sum())
            support["private_battery_subhourly_gbp"] = float(_num(private, "battery_subhourly_revenue_gbp").sum())
        if "life_extension_top_up_gbp" in private.columns:
            support["private_top_up_gbp"] = float(_num(private, "life_extension_top_up_gbp").sum())
            support["private_top_up_owned_gbp"] = float(_num(private, "life_extension_top_up_gbp")[costed].sum())
        if "battery_revenue_top_up_gbp" in private.columns:
            # Decision 63: the battery top-up the private case pays, and the part on batteries the case owns.
            support["private_battery_top_up_gbp"] = float(_num(private, "battery_revenue_top_up_gbp").sum())
            support["private_battery_top_up_owned_gbp"] = float(
                _num(private, "battery_revenue_top_up_gbp")[costed].sum())
    if (legacy_gate_fee_gbp is not None and "efw_gate_fee_revenue_gbp" not in nat.columns
            and carried_acquisition is None):
        # F19: the gate fees the purchase price counts, now the state's income (a 2030 register).
        fees = pd.to_numeric(legacy_gate_fee_gbp.reindex(nat.index), errors="coerce").fillna(0.0)
        fees = fees.where(costed & legacy, 0.0)
        if float(fees.abs().sum()) > 0.0:
            lines[LINE_GATE_FEE_INCOME] = -float(fees.sum())
            support["efw_gate_fee_legacy_gbp"] = float(fees.sum())
            support["efw_gate_fee_basis"] = ("the legacy valuation's gate_fee_model_year_gbp (WRAP 2024/25 median "
                                             "fee, tonnes capped at the UK's 16.82 Mt), as the purchase price counts it")
    if _is_vintaged_register(private) and "rab_support_gbp" in private.columns:
        # FES 2050 decision 41 (c): Sizewell C's RAB allowance the register books, on plant the
        # case owns and costs (the levy the tariff replaces when it is positive).
        rab = _num(private, "rab_support_gbp")
        diagnostics["private_rab_support_gbp"] = float(rab[costed].sum())
        # Independent review (15 Nov 2026): the levy is replaced whole when the case costs any RAB row, so a case
        # that owns only part of Sizewell C would leave the rest of its RAB paid by nobody. Refuse it.
        if float(rab[costed].abs().sum()) > 0.0 and float(rab[~costed].abs().sum()) > 1.0:
            raise ValueError(
                f"{case_id}: the case owns part of Sizewell C's RAB (GBP {float(rab[costed].sum()):,.0f} of "
                f"{float(rab.sum()):,.0f}); the tariff replaces the levy whole, so the rest would be paid by nobody")
    if carried_acquisition is not None:
        # Item Y12: the prices the carried annuity was struck on, and what the register's
        # own (model-year) legacy capital would have been, set aside so it cannot count twice.
        diagnostics.update({
            "acquisition_price_gbp": float(carried_acquisition["purchase_price_gbp"]),
            "acquisition_price_renewables_nuclear_gbp": float(carried_acquisition["purchase_price_renewables_nuclear_gbp"]),
            "acquisition_price_thermal_storage_gbp": float(carried_acquisition["purchase_price_thermal_storage_gbp"]),
            "acquisition_carried_from_year": int(carried_acquisition["acquisition_year"]),
            "acquisition_carried_payable": bool(carried_acquisition["payable_in_model_year"]),
            "acquisition_carried_last_payment_year": int(carried_acquisition["last_payment_year"]),
            "register_legacy_capital_set_aside_gbp": register_legacy_capital,
        })
        if interconnector_detail is not None:
            diagnostics["interconnector_acquisition_price_gbp"] = float(
                carried_acquisition["interconnector_gb_halves_price_gbp"])
    if decision55:
        diagnostics["decision_55"] = decision55
    diagnostics.update(support)
    diagnostics["generation_and_storage_cost_gbp"] = float(sum(lines.values()))
    return lines, diagnostics


def regulated_asset_value(private_rows: pd.DataFrame) -> pd.Series:
    """FES 2050 decision 55: a RAB row's regulated asset value in the model year (the statutory price).

    The present value, at the RAB's allowed rate (the private row's ``wacc``), of the capital
    element its allowance pays over the remaining life (``annual_capital_recovery_gbp``, the
    register's own RAB allowance basis): the value the regulated return and depreciation are
    earned on, so former owners are paid what the allowance would have returned them.
    """
    capital = pd.to_numeric(private_rows["annual_capital_recovery_gbp"], errors="coerce").fillna(0.0)
    rate = pd.to_numeric(private_rows["wacc"], errors="coerce")
    years = pd.to_numeric(private_rows["remaining_life_years"], errors="coerce").clip(lower=1.0)
    out = [c / capital_recovery_factor(float(r), float(n)) if np.isfinite(r) and np.isfinite(n) else np.nan
           for c, r, n in zip(capital, rate, years)]
    return pd.Series(out, index=private_rows.index, dtype=float)


def check_acquisition_pricing(nat: pd.DataFrame, capital: pd.Series, *, owned: pd.Series, legacy: pd.Series,
                              carried: pd.Series, rav: pd.Series, case_id: str = "") -> None:
    """FES 2050 decision 55: the survivor check's enforced part, on a later year's tariff case.

    Refuses (1) any asset acquired in the earlier purchase (a legacy row, or a new-build row
    flagged ``carried``) that the tariff still charges capital for, as if new build; and (2) any
    state-held, costed asset with capex that has no price: no capital charge, not carried, not
    acquired at its RAV, and not on decision 41's sunk-capital rule (a contract that has ended,
    ``support_basis = contract_expired_merchant``, or a sunk legacy row, which is acquired).
    """
    capital = pd.to_numeric(capital, errors="coerce").fillna(0.0)
    acquired = (legacy | carried) & owned
    double = acquired & capital.gt(1e-6)
    if double.any():
        raise ValueError(f"{case_id}: {int(double.sum())} acquired assets are also paid capital as new build "
                         f"(e.g. {list(nat.index[double][:3])}): the earlier purchase already pays for them")
    capex = pd.to_numeric(nat["capex_power_gbp_per_kw"], errors="coerce") if "capex_power_gbp_per_kw" in nat.columns \
        else pd.Series(np.nan, index=nat.index)
    sunk = (nat["support_basis"].fillna("").astype(str).eq("contract_expired_merchant")
            if "support_basis" in nat.columns else pd.Series(False, index=nat.index))
    sized = pd.to_numeric(nat["capacity_mw"], errors="coerce").fillna(0.0).gt(0)
    unpriced = owned & ~acquired & ~rav & ~sunk & capital.le(1e-6) & capex.gt(0) & sized
    if unpriced.any():
        raise ValueError(f"{case_id}: {int(unpriced.sum())} state-held assets have no price (no capital charge, not "
                         f"acquired, not sunk; e.g. {list(nat.index[unpriced][:3])})")


def capacity_market_not_acquired(private_components: pd.DataFrame, diagnostics: Mapping, conv: Mapping,
                                 agreement_cost_gbp: float | None,
                                 interconnector_acquired_gbp: float = 0.0,
                                 contract_cost_gbp: float | None = None) -> tuple[float, dict]:
    """The part of the Capacity Market line that pays capacity the state does not acquire (item MR4).

    The line pays the whole contract book (``agreement_cost_gbp``, report-year
    money, times the household allocation factor).  Nationalisation removes only
    what the acquired and costed fleet was paid (``capacity_market_revenue_gbp``
    on those register rows, model-year money): interconnectors (328 m of 2025
    money) and DSR (110 m) are not bought, plant paid as today (CHP, oil) keeps
    its agreements, and book capacity the model does not carry is still
    contracted.  The kept share of the book is applied to the private household
    line, so the household allocation factor carries through, and returned as
    system GBP at the Capacity Market's own rule (the kept share of the book, in
    model-year money), which the tariff recovers at that same rule so the line
    reproduces it.  (Until 26 Sep 2026 the system amount was taken at the flat
    conversion, overstating it by the 1.348 peak factor.)  Returns
    ``(system_gbp, detail)``; zero without a book total.

    Item D3: ``interconnector_acquired_gbp`` (report-year money) is the part of the
    interconnectors' agreements held by GB halves the state acquires; it leaves the
    kept line with the rest of the acquired fleet.  The foreign halves and cables
    owned in another jurisdiction keep theirs.

    FES 2050 decision 73: with ``contract_cost_gbp`` (report-year money; only a vintaged
    register's financing summary carries it, :func:`adequacy_contract`) the kept line is the
    adequacy-planning contract for what the state does not own, less the GB halves' share of
    its interconnector payment. Without it (every 2030 case) nothing here changes.
    """
    private = private_components.loc[private_components["case_id"].eq("private")].set_index("component_id")[VALUE]
    line = float(private.get("capacity_market", 0.0))
    if not agreement_cost_gbp or line == 0.0:
        return 0.0, {"capacity_market_book_model_year_gbp": None, "capacity_market_kept_share": 0.0,
                     "capacity_market_kept_gbp_per_3100kwh_ex_vat": 0.0}
    book = float(agreement_cost_gbp) / float(conv["factor"])
    interconnector_owned = float(interconnector_acquired_gbp) / float(conv["factor"])
    if contract_cost_gbp is not None:
        # FES 2050 decision 73 (adequacy planning): the line is a targeted contract for what the state does
        # not own, priced off the residual auction (report-year money); the acquired fleet holds no part of
        # it, and the interconnectors' GB halves' share of its interconnector payment is the state's own.
        contract = float(contract_cost_gbp) / float(conv["factor"])
        kept_share = max(0.0, (contract - interconnector_owned) / book)
        kept_household = line * kept_share
        detail = {
            "capacity_market_book_model_year_gbp": book,
            "capacity_market_adequacy_contract_model_year_gbp": contract,
            "capacity_market_owned_costed_model_year_gbp": 0.0,
            "capacity_market_interconnector_gb_halves_acquired_model_year_gbp": interconnector_owned,
            "capacity_market_kept_share": kept_share,
            "capacity_market_kept_gbp_per_3100kwh_ex_vat": kept_household,
            "capacity_market_private_line_gbp_per_3100kwh_ex_vat": line,
        }
        return kept_household / _rule_rate(conv, "capacity_market"), detail
    owned = float(diagnostics.get("capacity_market_revenue_owned_costed_gbp", 0.0)) + interconnector_owned
    if owned > book * (1.0 + 1e-6):
        raise ValueError(f"the acquired fleet's Capacity Market revenue (GBP {owned:,.0f}) exceeds the contract book "
                         f"(GBP {book:,.0f} model-year money); the register and the bill disagree on the book")
    kept_share = max(0.0, 1.0 - owned / book)
    kept_household = line * kept_share
    detail = {
        "capacity_market_book_model_year_gbp": book,
        "capacity_market_owned_costed_model_year_gbp": owned,
        "capacity_market_interconnector_gb_halves_acquired_model_year_gbp": interconnector_owned,
        "capacity_market_kept_share": kept_share,
        "capacity_market_kept_gbp_per_3100kwh_ex_vat": kept_household,
        "capacity_market_private_line_gbp_per_3100kwh_ex_vat": line,
    }
    return kept_household / _rule_rate(conv, "capacity_market"), detail


def adequacy_contract(financing_summary: pd.DataFrame | None, case_id: str, conv: Mapping) -> dict | None:
    """FES 2050 decision 73: a tariff case's adequacy-planning contract, or None (today's kept CM line).

    Read off the financing summary's ``capacity_market_design_*`` columns (model money), carried to
    report money by the bill's rebasing factor: the contract's cost and its interconnector payment
    (whose GB halves the state owns). None without the columns (every 2030 run passes no summary),
    or when the case keeps the market-participant treatment (``cm_market_participant``).
    """
    from scripts.public_ownership.capacity_market_design import ADEQUACY_PLANNING, SUMMARY_PREFIX, case_moves

    moves = case_moves(financing_summary, case_id) if financing_summary is not None else None
    if moves is None or moves["design"] != ADEQUACY_PLANNING:
        return None
    table = financing_summary.set_index("case_id")
    factor = float(conv["factor"])
    return {"moves": moves, "line_cost_report_year_gbp": moves["line_cost_gbp"] * factor,
            "interconnector_payment_report_year_gbp":
                float(table.at[case_id, SUMMARY_PREFIX + "interconnector_payment_gbp"]) * factor}


# ── the bill ──────────────────────────────────────────────────────────────────

def conversion(assumptions: Mapping, recovery: Mapping, bill_summary: Mapping | None = None) -> dict:
    """Household conversion factors and the recovery denominators each replaced line uses.

    F19 (7 Oct 2026): a bill on F18's two-volume basis converts every line through its own
    recovery entry (``conv["line_recovery"]``, the private case's, from the bill summary);
    final demand is then metered supply, the Exchequer's key. A D1 bill (2050) keeps the
    recovery file's denominators, as before F18.
    """
    factor = float(assumptions["currency_rebasing"]["factor"])
    mwh = float(assumptions["benchmark"]["annual_consumption_kwh"]) / 1000.0
    vat_rate = float(assumptions["benchmark"]["vat_rate"])
    cm = ((assumptions.get("sourced_externals") or {}).get("capacity_market") or {}).get("derivation") or {}
    if bill_summary is not None and bill_recovery.is_two_volume(bill_summary):
        entries = bill_recovery.line_recovery_block(bill_summary, "private")
        if not entries:
            raise ValueError("the bill summary is on the two-volume basis but its private case carries no line_recovery")
        metered = bill_recovery.exchequer_key_mwh(bill_summary)
        rate = lambda line: system_to_household_gbp(entries[line], 1.0, mwh * 1000.0)  # noqa: E731
        # Each eligible volume as the flat denominator it equals (kept for readers of the
        # conversion block; the rates come from the entries).
        return {
            "cm_household_allocation_factor": float(entries["capacity_market"]["household_allocation_factor"])
            if "capacity_market" in entries else float(cm.get("household_allocation_factor", 1.0)),
            "factor": factor,
            "final_demand_mwh": metered,
            "cfd_eligible_mwh": factor * mwh / rate("cfd") if "cfd" in entries else metered,
            "ro_eligible_mwh": (factor * mwh / rate("renewables_obligation") if "renewables_obligation" in entries
                                else metered),
            "mwh": mwh,
            "vat_rate": vat_rate,
            "recovery_basis": bill_recovery.TWO_VOLUME_BASIS,
            "line_recovery": entries,
        }
    # A D1 bill (2050): the recovery file's volumes (item D1: the Capacity Market line over the
    # denominator the bill names for it, its household factor carrying the losses; every other
    # replaced line over metered supply or its eligible share), read in bill_recovery.
    volumes = bill_recovery.d1_recovery_file_volumes(recovery, cm.get("recovery_denominator", "final_demand_mwh"))
    final_demand, ro_eligible = volumes["final_demand_mwh"], volumes["ro_eligible_mwh"]
    cfd_eligible = float((bill_recovery.d1_denominators(bill_summary)["cfd_eligible_mwh"] if bill_summary else None)
                         or ro_eligible)
    return {
        "cm_household_allocation_factor": float(cm.get("household_allocation_factor", 1.0)),
        "cm_volume_mwh": volumes["cm_volume_mwh"],
        "factor": factor,
        "final_demand_mwh": final_demand,
        "cfd_eligible_mwh": cfd_eligible,
        "ro_eligible_mwh": ro_eligible,
        "mwh": mwh,
        "vat_rate": vat_rate,
    }


def _kept_line_system(conv: Mapping, line_id: str, household_gbp: float) -> float:
    """System GBP of a household amount on a kept line (a network or Sizewell C adjustment).

    F19 (7 Oct 2026): through the line's own recovery entry on the two-volume basis; flat
    over final demand on a D1 bill (both are metered supply for TNUoS, DUoS and Sizewell C).
    """
    if conv.get("line_recovery") and line_id in conv["line_recovery"]:
        return household_to_system_gbp(conv["line_recovery"][line_id], household_gbp, float(conv["mwh"]) * 1000.0)
    return system(float(household_gbp), conv["final_demand_mwh"], conv["factor"], conv["mwh"])


def _denominator(conv: Mapping, component_id: str) -> float:
    return float(conv.get(ELIGIBLE_SUPPLY_LINES.get(component_id, "final_demand_mwh"), conv["final_demand_mwh"]))


def _replaced_system(private_hh: float, line_id: str, conv: Mapping, allocation: Mapping | None) -> float:
    """System GBP a replaced line stands for: its household value at its own recovery rule.

    FES 2050 decision 61: the interconnector cap-and-floor pass-through is recovered through the
    TNUoS demand residual, which no tariff rule reproduces; where the conversion carries the
    bill's own system credit (``conv["interconnector_pass_through_system_gbp"]``, set for a
    vintaged year from the forecast summary) that is the line's system amount, scaled with the
    household value passed (the ledgers pass the line's own value, or its negative).
    """
    if line_id == INTERCONNECTOR_PASS_THROUGH_LINE and conv.get("interconnector_pass_through_household_gbp"):
        return (float(private_hh) / float(conv["interconnector_pass_through_household_gbp"])
                * float(conv["interconnector_pass_through_system_gbp"]))
    if allocation is not None and line_id in (allocation.get("replaced_rates") or {}):
        rate = allocation["replaced_rates"][line_id]
        return 0.0 if rate is None else float(private_hh) / float(rate)
    if conv.get("line_recovery") and line_id in conv["line_recovery"]:
        return household_to_system_gbp(conv["line_recovery"][line_id], private_hh, float(conv["mwh"]) * 1000.0)
    return system(private_hh, _denominator(conv, line_id), conv["factor"], conv["mwh"])


#: F19 (7 Oct 2026): the bill line whose entry a new tariff line's recovery is declared on, by rule.
#: The pool (and the Treasury margin, at the tariff's own rate) is declared on the GSP charging
#: volume at the household loss factor, with the allocation factor that reproduces its rate.
RULE_TEMPLATE_LINES = {**RULE_ENTRY_LINES, "final_demand": "direct_energy", POOL: "bsuos_constraints",
                       TREASURY_MARGIN_RULE: "bsuos_constraints", "flat": "direct_energy"}


def tariff_line_recovery(kept_ids, lines: Mapping[str, float], allocation: Mapping | None, conv: Mapping) -> dict:
    """``component_id -> entry`` for every line of a tariff bill on the two-volume basis.

    Kept lines carry the market bill's own entries. A new line is declared on the entry of
    the line its rule names (``RULE_TEMPLATE_LINES``; the domestic rule on domestic supply),
    with the household allocation factor set so the entry reproduces the rate the tariff
    charged it at: the pool's like-for-like weighting, the hedging uplift, the Capacity
    Market's peak factor. Checked to 1e-12 against that rate.
    """
    entries = conv["line_recovery"]
    kwh = float(conv["mwh"]) * 1000.0
    out = {}
    for component_id in kept_ids:
        if component_id not in entries:
            raise KeyError(f"kept tariff line {component_id!r} has no recovery entry in the market bill")
        out[component_id] = dict(entries[component_id])
    rules = (allocation or {}).get("new_line_rules") or {}
    for line_id in lines:
        rate = line_rate(allocation, conv, line_id)
        # A line outside the allocation (the optional income lines) is recovered flat (line_rate's fallback).
        rule = (rules.get(line_id, POOL) if allocation and line_id in (allocation.get("new_line_rates") or {})
                else "flat")
        if rule == "domestic":
            template = make_entry("domestic_electricity_supply_mwh", conv["domestic_mwh"],
                                  price_year_factor=entries["direct_energy"]["price_year_factor"],
                                  system_price_year=entries["direct_energy"]["system_price_year"],
                                  basis="domestic customers only: spread over domestic electricity supply")
        else:
            template = dict(entries[RULE_TEMPLATE_LINES[rule]])
        unit = system_to_household_gbp({**template, "household_allocation_factor": 1.0}, 1.0, kwh)
        entry = make_entry(template["volume_key"], template["volume_mwh"],
                           household_loss_factor=template["household_loss_factor"],
                           household_allocation_factor=rate / unit,
                           price_year_factor=template["price_year_factor"],
                           system_price_year=template["system_price_year"],
                           basis=(f"nationalised tariff line, rule {rule}: "
                                  + ("the pool of the replaced market-construct lines, recovered like for like "
                                     "(allocation factor = the pool's weighting on the GSP charging volume)"
                                     if rule in (POOL, TREASURY_MARGIN_RULE) else
                                     f"recovered like {RULE_TEMPLATE_LINES.get(rule, 'domestic supply')}")))
        if abs(system_to_household_gbp(entry, 1.0, kwh) / rate - 1.0) > 1e-12:
            raise ValueError(f"{line_id}: its recovery entry does not reproduce the tariff's rate")
        out[line_id] = entry
    return out


def tariff_cm_recovery_options(bill_summary: Mapping, lines: Mapping[str, float], conv: Mapping) -> dict | None:
    """F11 on P3: the tariff's kept Capacity Market line under (b) flat per MWh and (c) taxation.

    The same rule as the market cases (``project_public_ownership_consumer_bill.cm_recovery_options``),
    on the line the tariff keeps (``nationalised_capacity_market_not_acquired``, recovered at the Capacity
    Market's own rule), with its system cost the tariff's own amount for it, so (c) moves exactly that.
    """
    from scripts.public_ownership.project_public_ownership_consumer_bill import cm_recovery_options

    household = float((bill_summary.get("new_lines_gbp_per_3100kwh_ex_vat") or {}).get(LINE_CM_NOT_ACQUIRED, 0.0))
    if not household:
        return None
    factor = float(conv.get("cm_household_allocation_factor", 1.0))
    options = cm_recovery_options(household, cm_factor=factor, final_demand_mwh=float(conv["final_demand_mwh"]),
                                  factor=float(conv["factor"]), mwh=float(conv["mwh"]))
    system_gbp = float(lines.get(LINE_CM_NOT_ACQUIRED, 0.0))
    options["system_cost_gbp_2024"] = system_gbp
    options["c_taxation"]["exchequer_cost_gbp_2024"] = system_gbp
    options["line"] = LINE_CM_NOT_ACQUIRED
    options["note"] = ("F11 on the tariff (reporting only): the Capacity Market agreements the state does not acquire, "
                       "recovered (b) flat per MWh of supply or (c) from taxation instead of the supplier charge on "
                       "winter-peak demand. Neither counts towards an ownership saving.")
    return options


#: F27 item 10 (user, 20 Oct 2026): the supplier line, whose EBIT and headroom follow the tariff's costs.
SUPPLIER_LINE = "supplier_costs_and_margin"
SUPPLIER_MARGIN_STEP = "suppliers' EBIT and headroom on the tariff's own costs (F27 item 10)"


def supplier_margin_adjustment(components: pd.DataFrame, margin: Mapping | None) -> float:
    """GBP per household to add to the supplier line so its EBIT and headroom are on the tariff's costs.

    ``margin`` is the private bill's ``supplier_margin`` record (F18): Ofgem's EBIT and headroom
    percentages applied to the bill's own costs (every line ex VAT except EBIT, headroom and
    levelisation). ``components`` is the tariff before this adjustment, whose supplier line still
    carries the private case's EBIT and headroom. Zero without a margin rule.
    """
    rule = (margin or {}).get("rule")
    if rule is None:
        return 0.0
    old = float(margin["ebit_gbp"]) + float(margin["headroom_gbp"])
    costs = float(components[VALUE].sum()) - old - float(margin["levelisation_gbp"])
    network = rule["network_group"]
    groups = bill_group_of(network["view"])
    networks = float(components.loc[components["component_id"].map(groups).eq(network["group"]), VALUE].sum())
    ebit = float(rule["ebit_variable_share"]) * costs + float(margin["ebit_fixed_gbp"])
    headroom = float(rule["headroom_share"]) * (costs - networks + ebit)
    return ebit + headroom - old


def build_bill(private_components: pd.DataFrame, lines: Mapping[str, float], conv: Mapping, *,
               case_id: str = NATIONALISED_CASE, source: str = "asset finance register",
               network_adjustments: Mapping[str, float] | None = None,
               allocation: Mapping | None = None,
               dropped_lines: Mapping[str, str] | None = None,
               extra_replaced: Mapping[str, str] | None = None,
               kept_replaced: Mapping[str, str] | None = None,
               sizewell_adjustment: float = 0.0, sizewell_note: str = "",
               supplier_adjustment: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Kept private rows plus the new tariff rows, and the household summary.

    ``supplier_adjustment`` (F27 item 10, :func:`supplier_margin_adjustment`) is GBP per household
    added to the kept supplier line: its EBIT and headroom on the tariff's own costs.

    ``sizewell_adjustment`` (OW10, :mod:`sizewell_c_rab`) is GBP per household, negative
    for a saving, added to the kept Sizewell C levy: its return element at the public rate.

    ``extra_replaced`` (:func:`register_replaced_lines`) are lines replaced only where the
    register books them to owned plant (FES 2050 decision 41 (c)).

    ``dropped_lines`` (:func:`dropped_lines_for`) are private lines removed with
    nothing in their place and outside the pool.

    ``allocation`` (:func:`recovery_rates`) sets the rate each new line is
    recovered at; without it every line is flat over final demand, the pre-26 Sep
    2026 behaviour, kept only for callers that build a bill without a private
    breakdown to allocate from.

    ``network_adjustments`` (item N6) is ``{component_id: GBP per household}``, negative
    for a saving, applied to the kept TNUoS and DUoS rows when the state owns the wires
    too.  The adjustment is added to the row rather than replacing it, because operating
    costs, depreciation and pass-through are unchanged by ownership: only the cost of
    capital and the tax allowance move.
    """
    private = private_components.loc[private_components["case_id"].eq("private")].copy()
    if private.empty:
        raise ValueError("bill components carry no private case")
    dropped_ids = set(dropped_lines or {})
    kept_notes = {**KEPT_DECLARED, **dict(kept_replaced or {})}
    replaced_lines = {k: v for k, v in {**ALL_REPLACED_LINES, **dict(extra_replaced or {})}.items()
                      if k not in (kept_replaced or {})}
    kept = private.loc[~private["component_id"].isin(set(replaced_lines) | dropped_ids)].copy()
    kept["case_id"] = case_id
    adjustments = dict(network_adjustments or {})
    unknown = sorted(set(adjustments) - set(NETWORK_ADJUSTED_LINES))
    if unknown:
        raise ValueError(f"network adjustments name lines that are not network lines: {unknown}")
    if adjustments:
        missing = sorted(set(adjustments) - set(kept["component_id"]))
        if missing:
            raise ValueError(f"network adjustment has no line to apply to: {missing}")
        for component_id, amount in adjustments.items():
            row = kept["component_id"].eq(component_id)
            kept.loc[row, VALUE] = kept.loc[row, VALUE] + float(amount)
            kept.loc[row, "method"] = kept.loc[row, "method"].astype(str) + (
                f" [nationalised: {amount:+.2f} GBP from item N6, the financing wedge on the "
                "Regulatory Asset Value at the public borrowing rate plus the tax allowance]")
    if sizewell_adjustment:
        row = kept["component_id"].eq(sizewell_c_rab.LINE)
        if not row.any():
            raise ValueError("the Sizewell C RAB adjustment has no sizewell_c_rab line to apply to")
        kept.loc[row, VALUE] = kept.loc[row, VALUE] + float(sizewell_adjustment)
        kept.loc[row, "method"] = kept.loc[row, "method"].astype(str) + (
            f" [nationalised: {float(sizewell_adjustment):+.2f} GBP, OW10: {sizewell_note}]")
    if supplier_adjustment:
        row = kept["component_id"].eq(SUPPLIER_LINE)
        if not row.any():
            raise ValueError("the supplier margin adjustment has no supplier_costs_and_margin line to apply to")
        kept.loc[row, VALUE] = kept.loc[row, VALUE] + float(supplier_adjustment)
        kept.loc[row, "method"] = kept.loc[row, "method"].astype(str) + (
            f" [nationalised: {float(supplier_adjustment):+.2f} GBP, F27 item 10: EBIT and headroom on the "
            "tariff's own costs]")
    kept["method"] = [
        f"{m} [nationalised: {kept_notes.get(c, 'kept unchanged')}]" if isinstance(m, str) else m
        for m, c in zip(kept["method"], kept["component_id"])
    ]
    flat_rate = household(1.0, conv["final_demand_mwh"], conv["factor"], conv["mwh"])
    rows = []
    for line_id, amount in lines.items():
        rate = line_rate(allocation, conv, line_id)
        rule = (allocation or {}).get("new_line_rules", {}).get(line_id, "final_demand") if allocation else "flat"
        rows.append({
            "case_id": case_id, "scenario": private["scenario"].iloc[0], "modelled_year": private["modelled_year"].iloc[0],
            "component_id": line_id, "stack_group": STACK_GROUP, "category": CATEGORY,
            "component": line_label(line_id),
            VALUE: float(amount) * rate,
            "source_class": SOURCE_CLASS,
            "method": (f"GBP{amount / 1e6:,.0f}M (model price year) recovered "
                       + (f"like for like at the pool rate of the replaced market-construct lines, x{rate / flat_rate:.4f} "
                          f"their flat share over final demand {conv['final_demand_mwh'] / 1e6:,.1f} TWh"
                          if rule == POOL else
                          f"by the rule of the line it is part of ({rule}), x{rate / flat_rate:.4f} the flat share over "
                          f"final demand {conv['final_demand_mwh'] / 1e6:,.1f} TWh")
                       + f", x{conv['factor']:.4f} to report prices, x {conv['mwh']:.1f} MWh; no hourly price."),
            "source": source, "currency": private["currency"].iloc[0], "reference_gbp_per_3100kwh_ex_vat": np.nan,
        })
    # 6 Oct 2026: the Ofgem price-cap and functional groups of every line (bill_groupings.csv).
    components = annotate_bill_groups(pd.concat([kept, pd.DataFrame(rows)], ignore_index=True))
    recovery_entries = (tariff_line_recovery(list(kept["component_id"]), lines, allocation, conv)
                        if conv.get("line_recovery") else None)
    ex_vat = float(components[VALUE].sum())
    private_ex_vat = float(private[VALUE].sum())
    network_total = float(sum(adjustments.values()))
    vat = ex_vat * conv["vat_rate"]
    removed = private.loc[private["component_id"].isin(replaced_lines)].set_index("component_id")[VALUE]
    dropped = private.loc[private["component_id"].isin(dropped_ids)].set_index("component_id")[VALUE]
    # The before/after of the pool: the same bill with the pool lines recovered
    # flat over final demand (the kept parts stay at their own rule, which the
    # household value of the Capacity Market line always had), i.e. what the
    # tariff charged before 26 Sep 2026 on the same inputs.
    new_flat = float(sum(
        float(a) * (flat_rate if (allocation or {}).get("new_line_rules", {}).get(line, POOL) == POOL
                    else line_rate(allocation, conv, line))
        for line, a in lines.items()))
    new_total = float(sum(r[VALUE] for r in rows))
    summary = {
        "case_id": case_id,
        "ex_vat_gbp": ex_vat, "vat_gbp": vat, "total_gbp_inc_vat": ex_vat + vat,
        "private_ex_vat_gbp": private_ex_vat, "private_total_gbp_inc_vat": private_ex_vat * (1 + conv["vat_rate"]),
        "delta_ex_vat_gbp": ex_vat - private_ex_vat,
        "delta_inc_vat_gbp": (ex_vat - private_ex_vat) * (1 + conv["vat_rate"]),
        "replaced_lines_gbp_per_3100kwh_ex_vat": {k: float(v) for k, v in removed.items()},
        "replaced_lines_total_gbp_per_3100kwh_ex_vat": float(removed.sum()),
        "dropped_lines_gbp_per_3100kwh_ex_vat": {k: float(v) for k, v in dropped.items()},
        "dropped_lines_declared": {k: (dropped_lines or {})[k] for k in dropped.index},
        "new_lines_gbp_per_3100kwh_ex_vat": {
            r["component_id"]: float(r[VALUE]) for r in rows
        },
        "new_lines_total_gbp_per_3100kwh_ex_vat": float(sum(r[VALUE] for r in rows)),
        "groupings": bill_group_totals(components),
        # F19 (7 Oct 2026): one recovery entry per tariff line (scripts/consumer_bill/line_recovery.py), so a
        # reader converts household and system pounds as the bill does; absent on a D1 (2050) bill.
        **({"line_recovery": recovery_entries} if recovery_entries is not None else {}),
        "kept_lines_declared": {c: kept_notes.get(c, "kept unchanged") for c in kept["component_id"]},
        "network_adjustments_gbp_per_3100kwh_ex_vat": {k: float(v) for k, v in adjustments.items()},
        "network_adjustments_total_gbp_per_3100kwh_ex_vat": network_total,
        "sizewell_c_rab_public_rate_adjustment_gbp_per_3100kwh_ex_vat": float(sizewell_adjustment),
        "supplier_margin_adjustment_gbp_per_3100kwh_ex_vat": float(supplier_adjustment),
        "tariff_shape": ("like for like: each new line at the allocation rule of the lines it replaces "
                         "(pool rate for the cost of owned plant); no hourly price" if allocation else
                         "flat over final demand; no domestic time-of-use allocation"),
        "tariff_allocation": {
            "pool_rate_over_flat": (float(allocation["pool_over_flat"]) if allocation else 1.0),
            "new_lines_total_flat_gbp_per_3100kwh_ex_vat": new_flat,
            "new_lines_total_gbp_per_3100kwh_ex_vat": new_total,
            "like_for_like_less_flat_ex_vat_gbp": new_total - new_flat,
            "ex_vat_gbp_if_flat": ex_vat - new_total + new_flat,
            "delta_inc_vat_gbp_if_flat": (ex_vat - new_total + new_flat - private_ex_vat) * (1 + conv["vat_rate"]),
        },
    }
    return components, summary


# ── reconciliation (item N7) ──────────────────────────────────────────────────

def reconcile(private_components: pd.DataFrame, bill_components: pd.DataFrame, lines: Mapping[str, float],
              diagnostics: Mapping, conv: Mapping, bill_detail: Mapping | None = None,
              network_adjustments: Mapping[str, float] | None = None,
              allocation: Mapping | None = None,
              dropped_lines: Mapping[str, str] | None = None,
              extra_replaced: Mapping[str, str] | None = None,
              kept_replaced: Mapping[str, str] | None = None,
              sizewell_adjustment: float = 0.0, supplier_adjustment: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Two ledgers from the private bill to the nationalised bill.

    ``supplier_adjustment`` (as passed to :func:`build_bill`; F27 item 10): an adjustment step on
    the kept supplier line in the bill ledger, attributed to the suppliers' EBIT and headroom.

    ``sizewell_adjustment`` (as passed to :func:`build_bill`; OW10): an adjustment step
    on the kept Sizewell C levy in the bill ledger, attributed to its cheaper finance.

    ``extra_replaced`` (as passed to :func:`build_bill`; FES 2050 decision 41 (c)): Sizewell
    C's levy, removed in the bill ledger and set against the register's RAB support in the
    declared residual, as the DPA line is.

    ``dropped_lines`` (as passed to :func:`build_bill`) are removal steps in the
    bill ledger and named non-register payments in the register ledger.

    **Bill ledger**: one step per replaced line (removed, converted at the
    denominator the line is recovered over) and per new line (added); it sums
    to the bill change exactly, or the function refuses.

    **Register ledger**: the same change attributed economically.  First the
    replaced bill lines that are *not* payments to plant the register sees, each
    named: the LDES floor top-up, the Capacity Market, the allocated CCS levy,
    the hedging uplifts, the CfD economic-curtailment payment on pre-AR4 output,
    the RO value above the register's certificate income, and the interconnector
    re-trading inside the constraint line (``bill_detail`` supplies the last
    three from the forecast bill summary).  Then the register-side attribution
    on GB owned and costed plant: legacy operating margin captured less the
    acquisition annuity; new-build rent removed plus the WACC saving; the
    state's operating overhead (item OH1) and every other new line.

    What is left is the residual, and it is not a plug: it must equal the sum
    of its declared parts (each a named gap between a bill line and the register
    payments it stands for: the wholesale line against register wholesale
    revenue, the settled CfD line against register CfD support, the constraint
    line against the register balancing settlement, register revenue outside
    those instruments, and any case-versus-private O&M or fuel difference on
    owned plant) within the identity tolerance, or the function refuses.  A new
    tariff line with no attribution, or a register quantity that stops matching
    its bill line, therefore fails loudly instead of disappearing into the
    residual (the OH1 overhead did, at 0.59 bn, until 26 Sep 2026).

    Each line's system amount is its household value at its own recovery rule
    (``allocation``, :func:`recovery_rates`): the Capacity Market with its peak
    factor, the allocated CCS levy over CfD-eligible sales, and the shaping line
    at zero, being an allocation between consumers rather than a cost.
    """
    private = private_components.loc[private_components["case_id"].eq("private")].set_index("component_id")[VALUE]
    bill = bill_components.set_index("component_id")[VALUE]
    steps = []
    removed_system: dict[str, float] = {}
    for line_id, reason in {**ALL_REPLACED_LINES, **dict(extra_replaced or {})}.items():
        if line_id in private.index and line_id not in (kept_replaced or {}):
            hh = -float(private[line_id])
            sys_gbp = -_replaced_system(-hh, line_id, conv, allocation)
            removed_system[line_id] = -sys_gbp
            steps.append({"ledger": "bill", "step": f"remove {line_id}", "component_id": line_id, "reason": reason,
                          "gbp_per_3100kwh_ex_vat": hh, "system_gbp_model_year": sys_gbp})
    dropped_lines = dict(dropped_lines or {})
    for line_id, reason in dropped_lines.items():
        if line_id in private.index:
            hh = -float(private[line_id])
            sys_gbp = -_replaced_system(-hh, line_id, conv, allocation)
            removed_system[line_id] = -sys_gbp
            steps.append({"ledger": "bill", "step": f"drop {line_id}", "component_id": line_id, "reason": reason,
                          "gbp_per_3100kwh_ex_vat": hh, "system_gbp_model_year": sys_gbp})
    for line_id, amount in lines.items():
        steps.append({"ledger": "bill", "step": f"add {line_id}", "component_id": line_id, "reason": line_label(line_id),
                      "gbp_per_3100kwh_ex_vat": float(amount) * line_rate(allocation, conv, line_id),
                      "system_gbp_model_year": float(amount)})
    # Item N6: the network lines are kept, not replaced, so their adjustment is its
    # own step rather than a removal and an addition.
    network = dict(network_adjustments or {})
    for line_id, amount in network.items():
        steps.append({"ledger": "bill", "step": f"adjust {line_id}", "component_id": line_id,
                      "reason": "state owns the Regulatory Asset Value and finances it at the public rate",
                      "gbp_per_3100kwh_ex_vat": float(amount),
                      "system_gbp_model_year": _kept_line_system(conv, line_id, float(amount))})
    # OW10: the kept Sizewell C levy, its return element at the public rate, over final demand.
    sizewell_system = (_kept_line_system(conv, sizewell_c_rab.LINE, float(sizewell_adjustment))
                       if sizewell_adjustment else 0.0)
    if sizewell_adjustment:
        steps.append({"ledger": "bill", "step": f"adjust {sizewell_c_rab.LINE}", "component_id": sizewell_c_rab.LINE,
                      "reason": "Sizewell C's regulated asset base financed at the public rate (OW10)",
                      "gbp_per_3100kwh_ex_vat": float(sizewell_adjustment), "system_gbp_model_year": sizewell_system})
    # F27 item 10: the kept supplier line's EBIT and headroom on the tariff's own costs.
    supplier_system = (_kept_line_system(conv, SUPPLIER_LINE, float(supplier_adjustment))
                       if supplier_adjustment else 0.0)
    if supplier_adjustment:
        steps.append({"ledger": "bill", "step": f"adjust {SUPPLIER_LINE}", "component_id": SUPPLIER_LINE,
                      "reason": "suppliers' EBIT and headroom at Ofgem's percentages of the tariff's own costs (F27)",
                      "gbp_per_3100kwh_ex_vat": float(supplier_adjustment), "system_gbp_model_year": supplier_system})
    bill_change = float(bill.sum() - private.sum())
    ledger_sum = float(sum(s["gbp_per_3100kwh_ex_vat"] for s in steps))
    if abs(ledger_sum - bill_change) > 1e-6:
        raise ValueError(f"bill ledger does not close: steps {ledger_sum:.6f} vs change {bill_change:.6f} GBP/household")

    d = diagnostics
    detail = dict(bill_detail or {})
    get = lambda key: float(removed_system.get(key, 0.0))  # noqa: E731
    # Replaced lines that are not payments to register plant, each named.
    # The shaping line's system amount is zero under the like-for-like rules (an
    # allocation between consumers); it is kept in the sum for the flat fallback.
    hedging = sum(get(k) for k in ("shaping_adjustment",) + HEDGING_LINES)
    econ_curtailment = float(detail.get("cfd_economic_curtailment_payment_gbp", 0.0))
    ic_retrading = float(detail.get("interconnector_retrading_cost_gbp", 0.0))
    ro_above_register = get("renewables_obligation") - d["private_roc_support_gbp"]
    not_register_payments = [
        ("LDES floor top-up (a floor payment, not market revenue in the register)", -get("ldes_cap_and_floor")),
        ("Capacity Market payments (outside the register's market revenue; the acquired fleet's fixed cost is now "
         "in the tariff)", -get("capacity_market")),
        ("hedging and imbalance uplifts on the wholesale purchase (supplier costs of a market that no longer "
         "exists)", -hedging),
        ("CfD economic-curtailment payment on pre-AR4 output the wholesale LP curtailed (not in the register)", -econ_curtailment),
        # Review B16: this is a volume gap, not recycle or headroom value: the bill's
        # RO line pays for more certificates than the register books to the fleet.
        ("RO line above the register's certificate income (a volume gap: the bill's obligation pays for more "
         "certificates than the register books)", -ro_above_register),
        ("interconnector re-trading inside the constraint line (kept as its own tariff line under I1)", -ic_retrading),
    ] + [(reason, -get(line_id)) for line_id, reason in dropped_lines.items()]
    # Register-side attribution on GB owned and costed plant.
    attribution = [
        ("legacy operating margin captured (market + support - fuel - O&M on legacy plant)", -d["legacy_operating_margin_gbp"]),
        ("legacy acquisition annuity, renewables and nuclear", lines[LINE_LEGACY_RENEWABLES]),
        ("legacy acquisition annuity, thermal and storage", lines[LINE_LEGACY_THERMAL]),
        ("new-build producer rent removed (market + support - fuel - private required revenue)", -d["new_build_private_rent_gbp"]),
        ("new-build capital: private WACC to public rate", lines[LINE_NEW_BUILD] - d["new_build_capital_private_gbp"]),
        ("interconnector re-trading kept as a tariff line (I1)", lines[LINE_IC_RETRADING]),
        ("real-time correction (institution I3 only)", lines[LINE_REALTIME]),
        ("cancelled legacy contracts: residual paid to former owners on the bill (item G3)",
         lines.get(LINE_CANCELLED_RESIDUAL, 0.0)),
        ("state operating overhead on owned plant: the holding company the public owner runs (item OH1)",
         lines.get(LINE_STATE_OVERHEAD, 0.0)),
        ("Capacity Market agreements kept: capacity the state does not acquire (foreign halves of "
         "interconnectors, DSR, plant paid as today, capacity outside the model)", lines.get(LINE_CM_NOT_ACQUIRED, 0.0)),
        ("interconnector congestion rent paid to foreign owners: foreign halves and other-jurisdiction cables "
         "(item D3)", lines.get(LINE_IC_FOREIGN_RENT, 0.0)),
        ("interconnector GB halves: acquisition annuity at the public rate, priced at the present value of their rent (item D3)",
         lines.get(LINE_IC_ACQUISITION, 0.0)),
        ("Power BECCS capture leg paid by its carbon CfD from the Exchequer (user decision 2)",
         lines.get(LINE_BECCS_CAPTURE, 0.0)),
        ("tax equivalents paid to the Exchequer by state-owned new build (competitive neutrality)",
         lines.get(LINE_TAX_EQUIVALENT, 0.0)),
        (DPA_RECEIPTS_STEP, lines.get(LINE_DPA_RECEIPTS, 0.0)),
        ("acquisition advisory and transaction costs", lines.get(LINE_TRANSACTION_COSTS, 0.0)),
        ("Feed-in Tariffs: domestic share moved to general taxation", lines.get(LINE_LEGACY_FIT, 0.0)),
        ("Warm Home Discount: domestic share moved to general taxation", lines.get(LINE_LEGACY_WHD, 0.0)),
        ("the state charges market prices: margin over the tariff's cost kept by the Treasury",
         lines.get(LINE_TREASURY_MARGIN, 0.0)),
    ]
    if network:
        attribution.append((
            "networks: financing wedge on the Regulatory Asset Value at the public rate, plus the "
            "corporation-tax allowance a public owner does not pay (item N6)",
            sum(_kept_line_system(conv, line_id, float(a)) for line_id, a in network.items()),
        ))
    if sizewell_adjustment:
        attribution.append(("Sizewell C RAB: its return at the public rate instead of the construction-period WACC "
                            "(OW10)", sizewell_system))
    if supplier_adjustment:
        attribution.append((SUPPLIER_MARGIN_STEP, supplier_system))
    for line_id in (LINE_ANCILLARY_INCOME, LINE_GATE_FEE_INCOME, LINE_SUBHOURLY_INCOME):
        if line_id in lines:
            attribution.append((line_label(line_id), lines[line_id]))
    if TOP_UP_LINE in removed_system:
        attribution.append(("life-extension top-up on owned plant no longer paid to its owners (decision 58)",
                            -float(d.get("private_top_up_owned_gbp", 0.0))))
    if BATTERY_TOP_UP_LINE in removed_system:
        attribution.append(("battery revenue top-up on owned batteries no longer paid to their owners (decision 63)",
                            -float(d.get("private_battery_top_up_owned_gbp", 0.0))))
    if d.get("import_basis") == "measured_link_trade_at_the_foreign_price":
        # The register values imported energy at the domestic wholesale price GB
        # consumers paid for it; a public buyer pays the foreign price on the
        # link. Naming the difference keeps it out of the residual.
        attribution.append((
            "imports repriced from the register's domestic wholesale revenue to the foreign price paid abroad",
            lines[LINE_IMPORTS] - float(d.get("import_register_domestic_revenue_gbp", 0.0)),
        ))
    explained = sum(a for _, a in not_register_payments) + sum(a for _, a in attribution)
    change_system = sum(s["system_gbp_model_year"] for s in steps)
    # The residual's declared parts, each signed as its contribution to the change:
    # register payments to plant that the removed bill lines do not carry (or
    # carry more of), and anything the case rows cost differently from private.
    residual_parts = {
        "register wholesale revenue above the wholesale line (the line is demand-weighted; the register adds "
        "storage and load-shedding revenue)": d["private_wholesale_revenue_gbp"] - (
            get("direct_energy") + get("procurement_losses")),
        "register CfD support above the settled CfD line (CfD + allocated CCS - economic curtailment)":
            d["private_cfd_support_gbp"] - (get("cfd") + get("allocated_ccs") - econ_curtailment),
        "register balancing settlement above the constraint line net of interconnector re-trading":
            d["private_balancing_settlement_gbp"] - (get("bsuos_constraints") - ic_retrading),
        "register market and support revenue outside wholesale, balancing, CfD and RO payments":
            float(d.get("private_market_and_support_revenue_all_gbp", 0.0)) - (
                d["private_wholesale_revenue_gbp"] + d["private_balancing_settlement_gbp"]
                + d["private_cfd_support_gbp"] + d["private_roc_support_gbp"]
                + (float(d.get("private_dpa_support_gbp", 0.0)) if "dpa" in private.index else 0.0)
                + (float(d.get("private_rab_support_gbp", 0.0)) if RAB_LINE in removed_system else 0.0)),
        "case less private O&M, fuel and carbon on owned plant (zero when capacity and dispatch are shared)":
            float(d.get("owned_om_fuel_case_minus_private_gbp", 0.0)),
        "private-case state overhead on new build (zero unless the private case owns plant)":
            -float(d.get("private_new_build_state_overhead_gbp", 0.0)),
    }
    if float(d.get("dpa_paid_through_kept_line_gbp", 0.0) or 0.0):
        # F19 P3 decision 4: the DPA is register revenue outside the instruments (above), paid by the
        # kept remaining_ccs line and not by the tariff, so it nets out here (it was paid twice).
        residual_parts["NZT DPA paid through the kept remaining_ccs line, not by the tariff (F19 P3 decision 4)"] = (
            -float(d["dpa_paid_through_kept_line_gbp"]))
    if float(d.get("rab_paid_through_kept_line_gbp", 0.0) or 0.0):
        # Money audit (13 Nov 2026): Sizewell C's RAB is register revenue outside the instruments (above), paid by
        # the kept sizewell_c_rab line and not by the tariff, so it nets out here (it was paid twice).
        residual_parts["Sizewell C RAB paid through the kept sizewell_c_rab line, not by the tariff"] = (
            -float(d["rab_paid_through_kept_line_gbp"]))
    if "dpa" in private.index:
        # Task SETTLE2050: the 2050 bill's DPA line is the settlement's own total, so this is
        # the DPA support the register books above it (zero when the two read one settlement).
        residual_parts["register DPA support above the DPA line (task SETTLE2050)"] = (
            float(d.get("private_dpa_support_gbp", 0.0)) - get("dpa"))
    if TOP_UP_LINE in removed_system:
        # Decision 58: the line reads the register's private top-up; what the owned plant was paid less the
        # line removed (zero when every topped-up row is owned, the line's own recovery rate aside).
        residual_parts["register life-extension top-up on owned plant against the top-up line (decision 58)"] = (
            float(d.get("private_top_up_owned_gbp", 0.0)) - get(TOP_UP_LINE))
    if BATTERY_TOP_UP_LINE in removed_system:
        # Decision 63: the line reads the register's private battery top-up (zero when every topped-up
        # battery is owned, the line's own recovery rate aside).
        residual_parts["register battery top-up on owned batteries against the top-up line (decision 63)"] = (
            float(d.get("private_battery_top_up_owned_gbp", 0.0)) - get(BATTERY_TOP_UP_LINE))
    if RAB_LINE in removed_system:
        # Decision 41 (c): the levy is the register's RAB allowance on Sizewell C (zero when the
        # bill line reads the same public account; the levy's own recovery rate aside).
        residual_parts["register RAB support above the Sizewell C RAB line (decision 41 (c))"] = (
            float(d.get("private_rab_support_gbp", 0.0)) - get(RAB_LINE))
    residual = float(sum(residual_parts.values()))
    unexplained = change_system - explained - residual
    tolerance = max(IDENTITY_TOLERANCE_GBP, 1e-12 * float(sum(abs(v) for v in removed_system.values())))
    if abs(unexplained) > tolerance:
        raise ValueError(
            f"register ledger does not close: GBP {unexplained:,.2f} of the change is neither attributed nor in a "
            "declared residual part; a tariff line has no attribution or a register quantity no longer matches "
            "its bill line")
    register_steps = not_register_payments + attribution + [(
        "residual: register payments the removed bill lines do not carry, declared by part "
        "(see residual_parts_gbp in the summary)", residual)]
    # The bill ledger converts each line at its own recovery denominator, so its
    # household total is not system GBP over final demand.  The register ledger is
    # a system-GBP attribution; it is shown per household at the blended rate the
    # bill change itself realised, so the two ledgers close in both units.
    blended = bill_change / change_system if change_system else 0.0
    for step, amount in register_steps:
        steps.append({"ledger": "register", "step": step, "component_id": "", "reason": "",
                      "gbp_per_3100kwh_ex_vat": float(amount) * blended, "system_gbp_model_year": float(amount)})
    frame = pd.DataFrame(steps)
    replaced_total = sum(removed_system.values())
    summary = {
        "bill_change_gbp_per_3100kwh_ex_vat": bill_change,
        "bill_change_system_gbp": change_system,
        "replaced_lines_system_gbp": replaced_total,
        "replaced_lines_system_gbp_by_line": removed_system,
        "register_payments_to_owned_plant_gbp": d["private_payments_to_owned_plant_gbp"],
        "not_register_payments_gbp": {step: float(a) for step, a in not_register_payments},
        "attribution_gbp": {step: float(a) for step, a in attribution},
        "residual_gbp": residual,
        "residual_share_of_replaced_lines": residual / replaced_total if replaced_total else None,
        "residual_parts_gbp": residual_parts,
        "unexplained_gbp": float(unexplained),
        "register_ledger_tolerance_gbp": float(tolerance),
        "register_ledger": {step: float(amount) for step, amount in register_steps},
        "sizewell_c_rab_public_rate_system_gbp": sizewell_system,
        "supplier_margin_system_gbp": supplier_system,
    }
    return frame, summary


# ── item N4: one-at-a-time variants ───────────────────────────────────────────

def legacy_capital_variants(register: pd.DataFrame, valuation: pd.DataFrame, acquisition: Mapping,
                            public_rate: float, case_id: str = NATIONALISED_CASE) -> dict[str, dict]:
    """Per-asset legacy capital charge under each alternative basis and term.

    Central = the register.  Variants: purchase-price basis for the valued group
    (market value, state present value), the workbook's A1, A2 no-support and A3 bases on
    every acquired row (F29), thermal basis zero, and a 15-year term.  Returns
    ``{variant_label: {"override": Series by asset_id (NaN where the register value
    stands), "purchase_price_gbp", "term_years"}}``.
    """
    from scripts.public_ownership.legacy_valuation import is_ro_row

    nat = register.loc[register["case_id"].eq(case_id)].set_index("asset_id")
    legacy = nat.loc[nat["finance_vintage"].astype(str).eq("legacy") & nat["state_owned"].astype(bool)
                     & nat["annual_required_revenue_gbp"].notna()]
    valued_ids = legacy.index[legacy["legacy_scope_group"].astype(str).eq(GROUP_VALUED)]
    other_ids = legacy.index[~legacy["legacy_scope_group"].astype(str).isin([GROUP_VALUED, GROUP_EXCLUDED])]
    term = float(acquisition.get("term_years", 30))
    crf = capital_recovery_factor(public_rate, term)
    by_asset = valuation.set_index("asset_id")
    v = by_asset.reindex(valued_ids)

    def _floored(frame, column):
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)

    pv_state = _floored(v, "pv_state_gbp")
    pv_market = _floored(v, "pv_market_gbp")
    price = pd.to_numeric(legacy["acquisition_price_gbp"], errors="coerce").fillna(0.0)
    valued_price, other_price = float(price[valued_ids].sum()), float(price[other_ids].sum())
    variants: dict[str, dict] = {}

    def _add(label, override, total_price, term_years):
        variants[label] = {"override": override, "purchase_price_gbp": float(total_price), "term_years": float(term_years)}

    central_basis = str(acquisition.get("purchase_price_basis", "state_present_value"))
    if central_basis != "market_value":
        _add("price_market_value", pv_market * crf, pv_market.sum() + other_price, term)
    if central_basis != "state_present_value":
        _add("price_state_present_value", pv_state * crf, pv_state.sum() + other_price, term)
    # F29: the workbook's other bases price every acquired row (thermal and storage too).
    every = by_asset.reindex(valued_ids.append(other_ids))
    book, no_support, market_all = (_floored(every, c) for c in ("net_book_value_gbp", "pv_no_support_gbp",
                                                                 "pv_market_gbp"))
    ro = is_ro_row(every["support_type"]) if "support_type" in every.columns else pd.Series(False, index=every.index)
    for basis, prices, needs in (("book_value", book, ("net_book_value_gbp",)),
                                 ("no_support", no_support, ("pv_no_support_gbp",)),
                                 ("ro_pass_through", no_support.where(ro, market_all),
                                  ("pv_no_support_gbp", "pv_market_gbp"))):
        # A basis the valuation file does not carry (an older file, a fixture) is not offered.
        if central_basis != basis and all(c in every.columns for c in needs) and prices.notna().all():
            _add(f"price_{basis}", prices * crf, prices.sum(), term)
    if str(acquisition.get("thermal_basis", "market_value")) != "zero":
        _add("thermal_zero", pd.Series(0.0, index=other_ids), valued_price, term)
    alt_term = 15.0 if term != 15.0 else 30.0
    _add(f"term_{int(alt_term)}y", price * capital_recovery_factor(public_rate, alt_term), valued_price + other_price, alt_term)
    return variants


# ── item N8: multi-year ───────────────────────────────────────────────────────

#: The multi-year view's years as offsets from the model year (item Y4); on the
#: CP30 year they are 2030, 2032, 2035, 2037 (the RO closes), 2040, 2045, 2050, 2060.
MULTI_YEAR_OFFSETS = (0, 2, 5, 7, 10, 15, 20, 30)
#: F27 item 14 (user, 20 Oct 2026): what the multi-year view holds and what moves, stated beside every figure.
MULTI_YEAR_BASIS = (
    "Only the legacy purchase moves over the years: the acquired fleet's streams as its plant retires and its contracts "
    "end, against the purchase annuity over its term. Everything else (new build, demand, prices, the networks) is held "
    "at the model year on both sides, so this is not a forecast. Retiring plant would be replaced by new build, "
    "state-built at cost under P2 and P3, so the held part would grow: the 2050 study measures that growth. The legacy "
    "streams after the model year rest on the valuation's price path (lever legacy_price_path), and gas plant's "
    "balancing margin on NESO's constraint-cost projection relative to the model year (F27 item 13).")
#: Annuities paid over the purchase's term besides the legacy fleet's (F19 P3): they run off with it.
RUN_OFF_LINES = (LINE_IC_ACQUISITION, LINE_TRANSACTION_COSTS)
#: F28 (e): the wires have no remaining life to repay over; under a remaining-life term the fees on
#: them run 30 years, as the networks' acquisition premium does (declared).
WIRES_TERM_YEARS = 30.0
#: F19 valuation decision (user, 14 Oct 2026): how the legacy corporation tax is profiled, stated beside it.
LEGACY_CORPORATION_TAX_BASIS = (
    "Corporation tax the acquired legacy fleet no longer pays (a transfer from the Exchequer, never on the bill): the "
    "register's model-year figure per carrier (25% of the comparator's margin after capital allowances and interest, "
    "F19 decision 3), scaled each year by that carrier's legacy streams over its model-year streams (floored at zero; "
    "none after the model year for a carrier whose model-year streams are not positive). Capital allowances and "
    "interest fall with the plant's written-down value too, so this is a declared approximation, not a tax computation.")


def legacy_corporation_tax_by_carrier(case_rows: pd.DataFrame) -> dict[str, float]:
    """The model year's corporation tax forgone on a case's acquired legacy rows, by carrier (F19, 14 Oct 2026).

    Read off the register (``corporation_tax_forgone_gbp`` on the case's ``legacy`` rows, F19
    decision 3 (b)); empty for a register without the column.
    """
    if case_rows.empty or "corporation_tax_forgone_gbp" not in case_rows.columns:
        return {}
    rows = case_rows.loc[case_rows["finance_vintage"].astype(str).eq("legacy")]
    tax = pd.to_numeric(rows["corporation_tax_forgone_gbp"], errors="coerce").fillna(0.0)
    by_carrier = tax.groupby(rows["carrier"].astype(str)).sum()
    return {str(c): float(v) for c, v in by_carrier.items() if v != 0.0}


def legacy_corporation_tax_path(tax_by_carrier: Mapping[str, float] | None, profile: pd.DataFrame, years,
                                model_year: int) -> dict[int, float]:
    """Each year's legacy corporation tax forgone, profiled on the carriers' falling streams (F19, 14 Oct 2026).

    ``tax_c(y) = tax_c(model year) x max(streams_c(y), 0) / streams_c(model year)``, with the
    streams the cash-flow profile's (valued, thermal and storage groups) by carrier; see
    :data:`LEGACY_CORPORATION_TAX_BASIS`.  Zeros without a tax.
    """
    years = [int(y) for y in years]
    if not tax_by_carrier:
        return {y: 0.0 for y in years}
    valued = profile.loc[profile["scope_group"].ne(GROUP_EXCLUDED)]
    streams = valued.groupby([valued["carrier"].astype(str), valued["year"].astype(int)])["gbp"].sum()
    m0 = int(model_year)
    out = {}
    for year in years:
        total = 0.0
        for carrier, tax in tax_by_carrier.items():
            base = float(streams.get((carrier, m0), 0.0))
            if year == m0:
                share = 1.0
            elif base > 0.0:
                share = max(float(streams.get((carrier, year), 0.0)), 0.0) / base
            else:
                share = 0.0
            total += float(tax) * share
        out[year] = total
    return out


def multi_year(profile: pd.DataFrame, lines: Mapping[str, float], conv: Mapping, *, term_years: float | None = None,
               model_year: int, bill_delta_model_year_gbp_per_hh: float = 0.0,
               years=None,
               allocation: Mapping | None = None,
               acquisition_year: int | None = None,
               horizon_year: int | None = None,
               corporation_tax_by_carrier: Mapping[str, float] | None = None,
               schedules: Mapping[str, "AnnuitySchedule"] | None = None) -> pd.DataFrame:
    """The legacy-acquisition component of the saving by year, and the bill change it implies.

    Private payments to legacy plant equal its operating margin plus its support
    (the valuation streams) plus fuel and O&M; the nationalised tariff pays fuel
    and O&M plus the annuity.  Fuel and O&M fall away on both sides as plant
    retires, so the legacy component in year y is streams(y) - annuity(y), the
    annuity running for ``term_years`` from the model year.  The private bill
    falls as the RO closes and legacy CfDs expire while the annuity does not, so
    the component shrinks.  ``bill_delta_gbp_per_3100kwh_ex_vat`` is the model
    year's whole bill change shifted by the change in this component, with
    everything else (new build, demand, prices, replacement of retired plant)
    held at model-year levels on both sides: a declared limitation, not a
    forecast.  The component is converted at the rate the acquisition annuity is
    recovered at (the pool rate under ``allocation``).  ``years`` defaults to the
    model year plus :data:`MULTI_YEAR_OFFSETS` (item Y4).

    Item Y12: ``acquisition_year`` (default the model year) is when the annuity started,
    so a purchase carried from 2030 is charged to 2059, not for ``term_years`` from the
    model year; ``horizon_year`` (the legacy valuation's) refuses a view that would read
    years past the profile as silent zeros.

    F28 (e): ``schedules`` (``"legacy"``, and per run-off line) give how each annuity runs off,
    asset by asset (:class:`AnnuitySchedule`); without them every annuity runs ``term_years``.
    """
    if years is None:
        years = year_offsets(model_year, MULTI_YEAR_OFFSETS)
    start = int(model_year if acquisition_year is None else acquisition_year)
    schedules = _schedules(schedules, term_years)
    if horizon_year is not None:
        last = max(max(int(y) for y in years), int(model_year) + 29)
        if last > int(horizon_year):
            raise ValueError(f"the multi-year view reaches {last} but the legacy profile stops at {horizon_year}: "
                             "those years would read as silent zeros (extend acquisition.valuation.horizon_year)")
    valued = profile.loc[profile["scope_group"].ne(GROUP_EXCLUDED)]
    by_year = valued.groupby("year")["gbp"].sum()
    # Item G3: a residual for cancelled contracts runs on the purchase's term too.
    annuity = lines[LINE_LEGACY_RENEWABLES] + lines[LINE_LEGACY_THERMAL] + lines.get(LINE_CANCELLED_RESIDUAL, 0.0)
    rate = line_rate(allocation, conv, LINE_LEGACY_RENEWABLES)
    hh = lambda gbp: float(gbp) * rate  # noqa: E731
    # F19 P3 (trace section 2): the GB halves' annuity and the purchase's fees are paid over the same
    # term and stop with it (they were held flat in every year, past their own term, before).
    live = lambda line, year: schedules[line].live(int(year) - start)  # noqa: E731
    # The model year's lines are each annuity's first-year charge times its share still live then.
    annuity = _first_year(annuity, live("legacy", model_year))
    run_off = {line: _first_year(float(lines.get(line, 0.0)), live(line, model_year)) for line in RUN_OFF_LINES}
    run_off_rates = {line: line_rate(allocation, conv, line) for line in RUN_OFF_LINES}
    base_component = float(by_year.get(model_year, 0.0)) - annuity * live("legacy", model_year)

    def run_off_at(year_live) -> tuple[float, float]:
        """The run-off annuities (system GBP) and their household cost at the given live shares."""
        gbp = sum(run_off[line] * year_live(line) for line in RUN_OFF_LINES)
        return gbp, sum(run_off[line] * run_off_rates[line] * year_live(line) for line in RUN_OFF_LINES)

    _, base_run_off_hh = run_off_at(lambda line: live(line, model_year))
    rows = []
    for year in years:
        streams = float(by_year.get(year, 0.0))
        charge = annuity * live("legacy", year)
        component = streams - charge
        run_off_gbp, run_off_hh = run_off_at(lambda line, y=year: live(line, y))
        rows.append({"year": year, "legacy_streams_gbp": streams, "acquisition_annuity_gbp": charge,
                     "legacy_component_system_gbp": component,
                     "legacy_component_gbp_per_3100kwh_ex_vat": hh(component),
                     "interconnector_and_fees_annuity_gbp": run_off_gbp,
                     "bill_delta_gbp_per_3100kwh_ex_vat": (bill_delta_model_year_gbp_per_hh - hh(component - base_component)
                                                           + run_off_hh - base_run_off_hh)})
    frame = pd.DataFrame(rows)
    for horizon in (15, 30):
        span = [y for y in range(model_year, model_year + horizon)]
        streams = float(by_year.reindex(span).fillna(0.0).mean())
        mean_live = lambda line, s=span: float(np.mean([live(line, y) for y in s]))  # noqa: E731
        charge = annuity * mean_live("legacy")
        component = streams - charge
        run_off_gbp, run_off_hh = run_off_at(mean_live)
        frame = pd.concat([frame, pd.DataFrame([{
            "year": f"{horizon}-year average", "legacy_streams_gbp": streams, "acquisition_annuity_gbp": charge,
            "legacy_component_system_gbp": component,
            "legacy_component_gbp_per_3100kwh_ex_vat": hh(component),
            "interconnector_and_fees_annuity_gbp": run_off_gbp,
            "bill_delta_gbp_per_3100kwh_ex_vat": (bill_delta_model_year_gbp_per_hh - hh(component - base_component)
                                                  + run_off_hh - base_run_off_hh)}])],
            ignore_index=True)
    if corporation_tax_by_carrier is not None:
        # F19 valuation decision (user, 14 Oct 2026): the legacy corporation tax forgone on the falling
        # streams (an Exchequer transfer, not on the bill), not held at the model year's figure.
        span = range(int(model_year), int(model_year) + 30)
        path = legacy_corporation_tax_path(corporation_tax_by_carrier, profile, sorted({*map(int, years), *span}),
                                           model_year)
        tax = {int(y): path[int(y)] for y in years}
        for horizon in (15, 30):
            tax[f"{horizon}-year average"] = float(np.mean([path[y] for y in range(int(model_year),
                                                                                  int(model_year) + horizon)]))
        frame["legacy_corporation_tax_forgone_gbp"] = [
            tax[int(y)] if not isinstance(y, str) else tax[y] for y in frame["year"]]
        frame["legacy_corporation_tax_basis"] = LEGACY_CORPORATION_TAX_BASIS
    frame["basis"] = MULTI_YEAR_BASIS
    return frame


def _first_year(model_year_amount: float, live_share: float) -> float:
    """An annuity's first-year charge from its model-year amount and the share of it live in the model year."""
    return float(model_year_amount) / live_share if live_share > 0 else 0.0


def _schedules(schedules: Mapping | None, term_years) -> dict:
    """Every annuity's schedule: those given, the rest on one term (``term_years``, 30 when None)."""
    out = dict(schedules or {})
    fallback = out.get("legacy") or AnnuitySchedule.fixed(30.0 if term_years is None else float(term_years))
    out.setdefault("legacy", fallback)
    for line in RUN_OFF_LINES:
        out.setdefault(line, fallback)
    return out


# ── review 29 Sep 2026: levers and reporting ─────────────────────────────────

def interconnector_price_settings(central: Mapping, block: Mapping | None, setting: str) -> dict:
    """The GB halves' purchase terms at the ``interconnector_price`` lever's owner rate (central, low, high)."""
    out = dict(central)
    if setting and setting != "central":
        rates = dict((block or {}).get("owner_real_rate_by_setting") or {})
        if setting not in rates:
            raise ValueError(f"interconnector_price {setting!r} needs acquisition.interconnectors."
                             f"owner_real_rate_by_setting.{setting}")
        out["owner_real_rate"] = float(rates[setting])
    return out


def transaction_costs(diagnostics: Mapping, ic_detail: Mapping | None, network_incidence: Mapping, *, share: float,
                      public_rate: float, term_years, network_lines: pd.DataFrame | None, treatment: str,
                      levers: Mapping, legacy_annual_gbp: float | None = None,
                      ic_annual_gbp: float | None = None) -> float:
    """Lever acquisition_transaction_costs: ``share x (purchase + GB halves + wires at their price)``, annuitised.

    Advisory, legal and financing fees are a real cost of the transaction, paid on
    the whole consideration: the legacy fleet's price, the interconnectors' GB
    halves and, when the wires are bought, the RAV plus its premium.  F28 (e): under a
    remaining-life term each part's fees are repaid with it (``legacy_annual_gbp`` and
    ``ic_annual_gbp``, the parts' own annuities); the wires over :data:`WIRES_TERM_YEARS`.
    """
    if not share:
        return 0.0
    wires = 0.0
    if treatment == "acquired" and network_lines is not None:
        from scripts.public_ownership.network_finance_account import select_rows

        rows = select_rows(network_lines, treatment, levers)
        wires = float((_num(rows, "rav_gbp") + _num(rows, "premium_capital_gbp")).sum())
    if str(term_years) == TERM_REMAINING_LIFE:
        if legacy_annual_gbp is None or ic_annual_gbp is None:
            raise ValueError("a remaining-life term needs the legacy and interconnector annuities for the fees")
        return float(share) * (float(legacy_annual_gbp) + float(ic_annual_gbp)
                               + wires * capital_recovery_factor(float(public_rate), WIRES_TERM_YEARS))
    price = float(diagnostics.get("acquisition_price_gbp", 0.0) or 0.0)
    price += float((ic_detail or {}).get("interconnector_acquisition_price_gbp", 0.0) or 0.0)
    return float(share) * (price + wires) * capital_recovery_factor(float(public_rate), float(term_years))


def legacy_levy_lines(private_components: pd.DataFrame, conv: Mapping, levers: Mapping) -> tuple[float, float]:
    """Lever legacy_levy_funding: the domestic FiT and WHD a tariff household no longer pays (system GBP, negative).

    The tariff replaces the RO line already (the ROC streams are bought with the
    plant), so under the background switch only the two kept levies move: a
    household pays none of them and the Exchequer takes on the domestic share.
    """
    if str(levers.get("legacy_levy_funding", "bills")) != "domestic_to_taxation":
        return 0.0, 0.0
    if not conv.get("domestic_mwh"):
        raise ValueError("legacy_levy_funding = domestic_to_taxation needs the domestic supply volume")
    private = private_components.loc[private_components["case_id"].eq("private")].set_index("component_id")[VALUE]
    rate = _rule_rate(conv, "domestic")
    return (-float(private.get("feed_in_tariffs", 0.0)) / rate, -float(private.get("warm_home_discount", 0.0)) / rate)


def headline_years(years: pd.DataFrame, summary: Mapping, conv: Mapping, headline_private: Mapping | None,
                   model_year: int = DEFAULT_MODEL_YEAR) -> dict:
    """User decision 8: the model-year bill beside the 15- and 30-year averages, each labelled, on the headline basis."""
    vat = float(conv["vat_rate"])
    private_total = float(headline_private["total_gbp_inc_vat"]) if headline_private else float(
        summary["private_total_gbp_inc_vat"])
    out = {"private_bill_gbp_inc_vat": private_total,
           "basis": ("against the headline private bill" if headline_private else "against this group's private bill"),
           # The peak-year statement is the CP30 year's finding (item N8), stated only for that year.
           "note": ((f"{model_year} is the peak year of the legacy CfD and RO rent the purchase captures; the averages "
                     f"hold everything but legacy expiry, retirement and the annuity at {model_year} levels (item N8)")
                    if int(model_year) == DEFAULT_MODEL_YEAR else
                    (f"the averages hold everything but legacy expiry, retirement and the annuity at {model_year} "
                     "levels (item N8)"))}
    by_year = years.set_index(years["year"].astype(str))
    for key, label in ((str(model_year), f"{model_year} snapshot"), ("15-year average", "15-year average"),
                       ("30-year average", "30-year average")):
        if key not in by_year.index:
            continue
        delta = float(by_year.at[key, "bill_delta_gbp_per_3100kwh_ex_vat"]) * (1.0 + vat)
        out[label] = {"bill_gbp_inc_vat": private_total + delta, "delta_vs_private_gbp_inc_vat": delta}
    return out


def break_even_purchase_price(summary: Mapping, diagnostics: Mapping, allocation: Mapping | None, conv: Mapping, *,
                              public_rate: float, term_years: float | None = None, headline_private: Mapping | None,
                              exchequer_funded: bool, model_year: int = DEFAULT_MODEL_YEAR,
                              annuity_payable: bool = True, annuity_factor: float | None = None) -> dict:
    """The legacy purchase price at which the case's bill saving against private reaches zero.

    The annuity is recovered at the pool rate, so each pound of price adds
    ``CRF(public rate, term) x pool rate x (1 + VAT)`` to the household bill; the
    break-even price is the current price plus the saving over that slope.  On
    the 30-year average the annuity runs ``min(term, 30) / 30`` of the years, so
    the slope scales.  The household-total break-even (bill plus taxpayer share)
    is in the Exchequer ledger, which holds the Exchequer column.  Under
    ``acquisition_financing: exchequer`` the bill does not carry the price, so it
    has no bill break-even.  F28 (e): ``annuity_factor`` (the purchase's own, price-weighted over its
    assets' terms) replaces ``CRF(public rate, term)``.
    """
    price = float(diagnostics.get("acquisition_price_gbp", 0.0) or 0.0)
    rate = line_rate(allocation, conv, LINE_LEGACY_RENEWABLES)
    factor = (float(annuity_factor) if annuity_factor is not None
              else capital_recovery_factor(float(public_rate), float(term_years)))
    slope = factor * rate * (1.0 + float(conv["vat_rate"]))
    private_total = float(headline_private["total_gbp_inc_vat"]) if headline_private else float(
        summary["private_total_gbp_inc_vat"])
    delta = float(summary["total_gbp_inc_vat"]) - private_total
    out = {"purchase_price_gbp": price, "bill_gbp_inc_vat_per_10bn_of_price": slope * 1e10,
           "delta_vs_private_gbp_inc_vat": delta,
           "annuity_factor": factor}
    if not annuity_payable:
        # Item Y12: a purchase carried from an earlier year whose term has run: the price no
        # longer reaches this year's bill.
        out.update({"break_even_purchase_price_bill_gbp": None, "bill_gbp_inc_vat_per_10bn_of_price": 0.0,
                    "note": f"the carried purchase's annuity has ended before {model_year}: the price is not on its bill"})
        return out
    if exchequer_funded or slope <= 0:
        out.update({"break_even_purchase_price_bill_gbp": None,
                    "note": ("the bill is the private comparator's (tariff_pricing market_bill): the price moves the "
                             "Treasury's margin, not the bill" if diagnostics.get("treasury_margin_gbp")
                             else "the purchase is funded from general taxation, so the bill carries no price")})
        return out
    # F19 P3 decision 5 (user, 6 Oct 2026): the snapshot figure is renamed and labelled; the
    # lifetime break-even (main(), :func:`lifetime_break_even_price`) replaces it as the headline.
    out["break_even_purchase_price_2030_snapshot_bill_gbp"] = price - delta / slope
    out["break_even_purchase_price_bill_gbp"] = out["break_even_purchase_price_2030_snapshot_bill_gbp"]
    out["snapshot_note"] = ("a SNAPSHOT: the legacy fleet's price (renewables, nuclear, thermal and storage; not the "
                            f"interconnectors' GB halves or the wires) at which this case's {model_year} bill alone "
                            "equals private's; it credits the front-loaded legacy rent of one year against a level "
                            "annuity, so it is not a lifetime figure")
    out["note"] = out["snapshot_note"]
    return out


def legacy_streams_settings(levers: Mapping, profile_by_settings: pd.DataFrame | None) -> dict | str:
    """The valuation settings a case's legacy streams are on (F19, 27 Oct 2026), for its summary."""
    from scripts.public_ownership.legacy_valuation import PROFILE_SETTINGS

    if profile_by_settings is None:
        return "the valuation's central streams"
    return {column: (str(levers.get(lever)) if levers.get(lever) is not None else "central")
            for column, lever, _ in PROFILE_SETTINGS}


def lifetime_break_even_price(profile: pd.DataFrame, *, public_rate: float, term_years: float | None = None,
                              model_year: int, acquisition_year: int | None = None,
                              schedule: "AnnuitySchedule | None" = None) -> dict:
    """The purchase price at which the legacy purchase's present value at the public rate is zero (F19 P3 decision 5).

    The legacy component is the streams the state takes over (the cash-flow profile:
    valued, thermal and storage groups) less the acquisition annuity; at the public rate,
    year 0 undiscounted as :func:`legacy_valuation.present_value`, it is zero when
    ``price x CRF(r, term) x sum_{t < term} (1 + r)^-t`` equals the streams' present value.
    """
    valued = profile.loc[profile["scope_group"].ne(GROUP_EXCLUDED)]
    by_year = valued.groupby("year")["gbp"].sum()
    r = float(public_rate)
    years = np.array([int(y) for y in by_year.index])
    pv_streams = float((by_year.to_numpy() * (1.0 + r) ** -(years - int(model_year))).sum())
    start = int(model_year if acquisition_year is None else acquisition_year)
    schedule = schedule or AnnuitySchedule.fixed(float(term_years))
    # F28 (e): each asset's share of the price repaid over its own term.
    annuity_pv_per_gbp = schedule.pv_per_gbp_of_price(r, int(model_year) - start)
    return {"break_even_purchase_price_lifetime_gbp": pv_streams / annuity_pv_per_gbp if annuity_pv_per_gbp > 0 else None,
            "legacy_streams_pv_public_rate_gbp": pv_streams, "annuity_pv_per_gbp_of_price": annuity_pv_per_gbp,
            "profile_last_year": int(years.max()) if len(years) else None,
            "lifetime_note": ("the legacy fleet's price at which the purchase's present value at the public rate "
                              "(the streams the state takes over, to the valuation horizon, less the annuity over its "
                              "term) is zero; the interconnectors' GB halves and the wires are not in it")}


#: F19 P3 decision 1 (user, 6 Oct 2026): the three parts of P3's bill change against private.
PART_NEW_BUILD, PART_LEGACY, PART_REST = "new_build_ownership", "legacy_purchase", "rest"
PART_LABELS = {
    PART_NEW_BUILD: "New-build ownership: new plant financed and owned by the state (what P2 also does)",
    PART_LEGACY: "The legacy purchase: the existing fleet's cash flows bought for a level annuity (a timing effect)",
    PART_REST: ("The rest: pre-2025 CfD and RO settlement gaps, interconnectors, the balancing mechanism "
                "internalised, hedging, the Capacity Market and LDES floor, and the institution"),
}


RESIDUAL_OM_PART = "case less private O&M, fuel and carbon on owned plant (zero when capacity and dispatch are shared)"
RECONCILING_ITEM = ("the acquired fleet's 2030 register payments against the valuation's 2030 streams counted in the "
                    "legacy purchase (its Capacity Market income, whose line is removed above; Drax's declared margin; "
                    "battery top-ups)")


def _part_items(recon_summary: Mapping, lines: Mapping[str, float], d: Mapping,
                legacy_streams_model_year_gbp: float) -> dict[str, list[tuple[str, float]]]:
    """Every register-ledger item assigned to one part (system GBP, signed as its contribution to the change).

    The legacy purchase takes the valuation's own 2030 streams (``legacy_streams_model_year_gbp``,
    the cash-flow profile its year path and present value run on), not the register ledger's
    legacy margin; the gap between the two (chiefly the acquired fleet's Capacity Market income,
    whose line the ledger removes among the not-register payments) is one named item of the rest,
    so the parts still partition the ledger exactly.
    """
    attribution = dict(recon_summary.get("attribution_gbp") or {})
    residual_parts = dict(recon_summary.get("residual_parts_gbp") or {})
    line = lambda key: float(lines.get(key, 0.0) or 0.0)  # noqa: E731
    overhead_legacy = float(d.get("state_overhead_legacy_gbp", 0.0) or 0.0)
    income_legacy = {LINE_ANCILLARY_INCOME: -float(d.get("ancillary_services_legacy_gbp", 0.0) or 0.0),
                     LINE_GATE_FEE_INCOME: -float(d.get("efw_gate_fee_legacy_gbp", 0.0) or 0.0),
                     LINE_SUBHOURLY_INCOME: -float(d.get("battery_subhourly_legacy_gbp", 0.0) or 0.0)}
    income_legacy = {k: v for k, v in income_legacy.items() if k in lines}
    beccs_on_bill = d.get("beccs_removal_funding") == "bill_levy"
    beccs_legacy = 0.0 if beccs_on_bill else -float(d.get("carbon_removal_cfd_legacy_gbp", 0.0) or 0.0)
    dpa_legacy = -float(d.get("dpa_receipts_legacy_gbp", 0.0) or 0.0)
    om_all = float(residual_parts.get(RESIDUAL_OM_PART, 0.0) or 0.0)
    om_legacy = float(d.get("owned_om_fuel_case_minus_private_legacy_gbp", 0.0) or 0.0)
    new_build = [
        ("new-build producer rent removed", -float(d["new_build_private_rent_gbp"])),
        ("new-build capital: private WACC to public rate", line(LINE_NEW_BUILD) - float(d["new_build_capital_private_gbp"])),
        ("state operating overhead on owned new build (OH1)", line(LINE_STATE_OVERHEAD) - overhead_legacy),
        ("O&M on owned new build, case less private (self-insurance, any public O&M premium)", om_all - om_legacy),
        ("Sizewell C RAB at the public rate (OW10)", float(recon_summary.get("sizewell_c_rab_public_rate_system_gbp", 0.0) or 0.0)),
        ("Power BECCS capture leg paid by the Exchequer's carbon CfD (new build)", line(LINE_BECCS_CAPTURE) - beccs_legacy),
        ("tax equivalents paid by state-owned new build", line(LINE_TAX_EQUIVALENT)),
        ("Net Zero Teesside DPA received by the state on owned new build", line(LINE_DPA_RECEIPTS) - dpa_legacy),
    ] + [(f"{line_label(k)} (new build)", line(k) - v) for k, v in income_legacy.items()]
    # What the ledger books on the acquired fleet's own streams: its margin and the support lines' income.
    ledger_streams = -float(d["legacy_operating_margin_gbp"]) + sum(income_legacy.values())
    legacy = [
        ("legacy streams captured: the valuation's 2030 cash flows of the acquired fleet (market margin, CfD, RO, "
         "Capacity Market, support lines, gate fees)", -float(legacy_streams_model_year_gbp)),
        ("legacy acquisition annuity, renewables and nuclear", line(LINE_LEGACY_RENEWABLES)),
        ("legacy acquisition annuity, thermal and storage", line(LINE_LEGACY_THERMAL)),
        ("cancelled legacy contracts: residual on the bill (G3)", line(LINE_CANCELLED_RESIDUAL)),
        ("acquisition advisory and transaction costs", line(LINE_TRANSACTION_COSTS)),
        ("state operating overhead on the acquired fleet (OH1)", overhead_legacy),
        ("O&M on the acquired fleet, case less private (self-insurance, any public O&M premium)", om_legacy),
    ]
    assigned = {
        "legacy operating margin captured (market + support - fuel - O&M on legacy plant)",
        "legacy acquisition annuity, renewables and nuclear", "legacy acquisition annuity, thermal and storage",
        "new-build producer rent removed (market + support - fuel - private required revenue)",
        "new-build capital: private WACC to public rate",
        "cancelled legacy contracts: residual paid to former owners on the bill (item G3)",
        "state operating overhead on owned plant: the holding company the public owner runs (item OH1)",
        "Power BECCS capture leg paid by its carbon CfD from the Exchequer (user decision 2)",
        "tax equivalents paid to the Exchequer by state-owned new build (competitive neutrality)",
        DPA_RECEIPTS_STEP,
        "acquisition advisory and transaction costs",
        "Sizewell C RAB: its return at the public rate instead of the construction-period WACC (OW10)",
        *(line_label(k) for k in income_legacy),
    }
    rest = [(step, float(amount)) for step, amount in (recon_summary.get("not_register_payments_gbp") or {}).items()]
    rest += [(step, float(amount)) for step, amount in attribution.items() if step not in assigned]
    rest += [(f"residual: {part}", float(amount)) for part, amount in residual_parts.items() if part != RESIDUAL_OM_PART]
    rest.append((RECONCILING_ITEM, ledger_streams + float(legacy_streams_model_year_gbp)))
    # Independent review (15 Nov 2026): the valuation's streams (the legacy part's first item) already count the
    # acquired plant's carbon CfD and DPA, while the register ledger puts them in the reconciling item and the declared
    # residual (both here); the tariff's credits on acquired plant net those, so they sit here, not in the legacy part,
    # where they counted the two contracts twice (GBP 336 m and 403 m in 2030).
    rest.append(("Power BECCS capture leg on acquired plant, against the carbon CfD in the reconciling item", beccs_legacy))
    rest.append(("Net Zero Teesside DPA received by the state on acquired plant, against the DPA in the residual",
                 dpa_legacy))
    rest.append(("register ledger tolerance (unexplained)", float(recon_summary.get("unexplained_gbp", 0.0) or 0.0)))
    return {PART_NEW_BUILD: new_build, PART_LEGACY: legacy, PART_REST: rest}


def legacy_alive_share(valuation: pd.DataFrame | None, years) -> pd.Series:
    """The acquired fleet's capacity still running in each year, as a share of the model year's (valued, thermal, storage)."""
    years = list(years)
    if valuation is None or valuation.empty or "retirement_year" not in valuation.columns:
        return pd.Series(1.0, index=years)
    rows = valuation.loc[~valuation.get("scope_group", pd.Series("", index=valuation.index)).astype(str).eq(GROUP_EXCLUDED)]
    mw = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0).to_numpy()
    retire = pd.to_numeric(rows["retirement_year"], errors="coerce").fillna(np.inf).to_numpy()
    total = float(mw.sum())
    return pd.Series([float(mw[retire > y].sum()) / total if total > 0 else 1.0 for y in years], index=years)


def three_part_split(recon_summary: Mapping, lines: Mapping[str, float], diagnostics: Mapping, conv: Mapping,
                     allocation: Mapping | None, profile: pd.DataFrame, *, public_rate: float,
                     term_years: float | None = None,
                     model_year: int, acquisition_year: int | None = None,
                     headline_delta_ex_vat_gbp: float | None = None,
                     valuation: pd.DataFrame | None = None,
                     corporation_tax_by_carrier: Mapping[str, float] | None = None,
                     schedules: Mapping[str, "AnnuitySchedule"] | None = None) -> tuple[dict, pd.DataFrame]:
    """P3's bill change against private in three parts (F19 P3 decision 1), summing to it exactly.

    **Definitions.**  Every item of the tariff's register ledger (item N7, system GBP,
    which closes to the bill change) goes to exactly one part:

    * *new-build ownership* (what P2 also does): the producer rent removed and the capital
      re-priced from private WACC to the public rate on new build, the state's overhead,
      O&M difference (self-insurance) and ancillary or sub-hourly income on new build,
      Sizewell C's RAB at the public rate, the BECCS capture leg (the Exchequer's carbon CfD)
      and any tax equivalents;
    * *the legacy purchase*: the acquired fleet's 2030 cash flows as the legacy valuation
      counts them (market margin, CfD top-ups, RO income, Capacity Market income, support
      lines, gate fees: the cash-flow profile's model year) less the acquisition annuities,
      plus the transaction fees, the state's overhead and O&M difference on the acquired
      fleet and any cancelled-contract residual;
    * *the rest*: the replaced lines that are not payments to the register's plant (LDES
      floor, Capacity Market, hedging and imbalance, the CfD economic-curtailment payment on
      pre-AR4 output, the RO line's volume gap, re-trading inside the constraint line), the
      declared residual (the CfD, wholesale and constraint lines against the register's own
      payments: the balancing mechanism internalised; the NZT DPA kept on its own line), the
      interconnectors (cross-border energy at the foreign price, the foreign halves' rent,
      the GB halves' annuity), the Capacity Market agreements kept, the institution lines,
      the networks, and one reconciling item: the acquired fleet's 2030 register payments
      against the valuation's 2030 streams (chiefly its Capacity Market income, which the
      ledger removes with the Capacity Market line).

    The 2030 parts convert at the blended rate the bill change realised (bill change over
    its system change), as the register ledger does, so they add up to the bill change in
    both units.  Their year paths (to the later of the profile's horizon and the annuity's
    last year) move as the multi-year view does: the legacy purchase by the change in its
    streams less the annuity at the annuity's recovery rate, its fees with their term and
    its overhead and O&M with the acquired capacity still running; the rest drops the GB
    halves' annuity at its term; new-build ownership is held at 2030.  The legacy purchase's
    present value is at the public rate, year 0 undiscounted.
    """
    valued = profile.loc[profile["scope_group"].ne(GROUP_EXCLUDED)]
    by_year = valued.groupby("year")["gbp"].sum()
    streams_model_year = float(by_year.get(int(model_year), 0.0))
    items = _part_items(recon_summary, lines, diagnostics, streams_model_year)
    change_system = float(recon_summary["bill_change_system_gbp"])
    change_hh = float(recon_summary["bill_change_gbp_per_3100kwh_ex_vat"])
    blended = change_hh / change_system if change_system else 0.0
    vat = 1.0 + float(conv["vat_rate"])
    system_parts = {part: float(sum(a for _, a in rows)) for part, rows in items.items()}
    if abs(sum(system_parts.values()) - change_system) > max(IDENTITY_TOLERANCE_GBP, 1e-9 * abs(change_system)):
        raise ValueError(f"the three parts ({sum(system_parts.values()):,.2f}) do not add up to the tariff's system "
                         f"change ({change_system:,.2f}): a register-ledger item has no part")
    # Against the headline private bill (the market group's), the offset goes to the rest.
    offset_hh = (float(headline_delta_ex_vat_gbp) - change_hh) if headline_delta_ex_vat_gbp is not None else 0.0
    hh_parts = {part: amount * blended for part, amount in system_parts.items()}
    hh_parts[PART_REST] += offset_hh
    start = int(model_year if acquisition_year is None else acquisition_year)
    annuity = float(lines.get(LINE_LEGACY_RENEWABLES, 0.0)) + float(lines.get(LINE_LEGACY_THERMAL, 0.0)) + float(
        lines.get(LINE_CANCELLED_RESIDUAL, 0.0))
    rate = line_rate(allocation, conv, LINE_LEGACY_RENEWABLES)
    fees = float(lines.get(LINE_TRANSACTION_COSTS, 0.0))
    ic = float(lines.get(LINE_IC_ACQUISITION, 0.0))
    fees_rate, ic_rate = (line_rate(allocation, conv, LINE_TRANSACTION_COSTS), line_rate(allocation, conv, LINE_IC_ACQUISITION))
    legacy_items = dict(items[PART_LEGACY])
    running = sum(v for k, v in legacy_items.items() if k.startswith(("state operating overhead", "O&M on the acquired")))
    schedules = _schedules(schedules, term_years)
    last_term = max(schedule.last_term for schedule in schedules.values())
    last = int(max(int(by_year.index.max()) if len(by_year) else model_year, start + int(round(last_term)) - 1))
    years = list(range(int(model_year), last + 1))
    alive = legacy_alive_share(valuation, years)
    # F28 (e): each annuity runs off asset by asset (one window under a fixed term).
    live = lambda y, line="legacy": schedules[line].live(int(y) - start)  # noqa: E731
    m0 = int(model_year)
    # The model year's lines are each annuity's first-year charge times its share still live then.
    annuity, fees, ic = (_first_year(annuity, live(m0)), _first_year(fees, live(m0, LINE_TRANSACTION_COSTS)),
                         _first_year(ic, live(m0, LINE_IC_ACQUISITION)))
    component = lambda y: float(by_year.get(y, 0.0)) - annuity * live(y)  # noqa: E731
    rows = []
    for year in years:
        d_component = component(year) - component(m0)
        d_fees = fees * (live(year, LINE_TRANSACTION_COSTS) - live(m0, LINE_TRANSACTION_COSTS))
        d_ic = ic * (live(year, LINE_IC_ACQUISITION) - live(m0, LINE_IC_ACQUISITION))
        d_running = running * (float(alive[year]) - float(alive[m0]))
        legacy_system = system_parts[PART_LEGACY] - d_component + d_fees + d_running
        legacy_hh = hh_parts[PART_LEGACY] - rate * d_component + fees_rate * d_fees + rate * d_running
        rows.append({"year": year, "new_build_ownership_system_gbp": system_parts[PART_NEW_BUILD],
                     "legacy_purchase_system_gbp": legacy_system, "rest_system_gbp": system_parts[PART_REST] + d_ic,
                     "new_build_ownership_gbp_inc_vat": hh_parts[PART_NEW_BUILD] * vat,
                     "legacy_purchase_gbp_inc_vat": legacy_hh * vat,
                     "rest_gbp_inc_vat": (hh_parts[PART_REST] + ic_rate * d_ic) * vat,
                     "total_gbp_inc_vat": (hh_parts[PART_NEW_BUILD] + legacy_hh + hh_parts[PART_REST] + ic_rate * d_ic) * vat,
                     "legacy_streams_gbp": float(by_year.get(year, 0.0)), "legacy_annuity_gbp": annuity * live(year),
                     "acquired_capacity_running_share": float(alive[year])})
    path = pd.DataFrame(rows)
    r = float(public_rate)
    discount = (1.0 + r) ** -(path["year"].to_numpy() - m0)
    pv_legacy_system = float((path["legacy_purchase_system_gbp"].to_numpy() * discount).sum())
    pv_streams = float((path["legacy_streams_gbp"].to_numpy() * discount).sum())
    pv_annuity = float((path["legacy_annuity_gbp"].to_numpy() * discount).sum())
    level = capital_recovery_factor(r, len(years)) / (1.0 + r)   # year-0 undiscounted level payment over the span
    summary = {
        "definition": three_part_split.__doc__.split("**Definitions.**", 1)[1].strip(),
        "labels": PART_LABELS,
        "basis": ("against the headline private bill (the rest carries the offset between this group's private and "
                  "the headline's)" if headline_delta_ex_vat_gbp is not None else "against this group's private bill"),
        "years_basis": MULTI_YEAR_BASIS,
        "model_year": m0,
        "system_gbp": system_parts,
        "gbp_per_3100kwh_inc_vat": {part: v * vat for part, v in hh_parts.items()},
        "total_gbp_inc_vat": float(sum(hh_parts.values()) * vat),
        "blended_rate_gbp_per_household_per_system_gbp": blended,
        "items_system_gbp": {part: {step: float(a) for step, a in rows_ if abs(a) > 0.0} for part, rows_ in items.items()},
        "legacy_purchase": {
            "model_year_gbp_inc_vat": hh_parts[PART_LEGACY] * vat,
            "path_gbp_inc_vat": {int(y): float(v) for y, v in zip(path["year"], path["legacy_purchase_gbp_inc_vat"])},
            "pv_public_rate_system_gbp": pv_legacy_system,
            "pv_public_rate_streams_gbp": pv_streams, "pv_public_rate_annuity_gbp": pv_annuity,
            "level_equivalent_gbp_inc_vat": pv_legacy_system * level * blended * vat,
            "pv_span": [int(years[0]), int(years[-1])], "public_rate": r,
            "note": ("the bill change the purchase makes in each year (negative a saving), its present value at the "
                     "public rate over the span (system GBP, year 0 undiscounted), and the level yearly bill change "
                     "with the same present value"),
        },
    }
    if corporation_tax_by_carrier is not None:
        # F19 valuation decision (user, 14 Oct 2026): the corporation tax the acquired fleet no longer pays,
        # profiled on its falling streams (an Exchequer transfer beside the parts, never inside them).
        tax = legacy_corporation_tax_path(corporation_tax_by_carrier, profile, years, m0)
        path["legacy_corporation_tax_forgone_gbp"] = [tax[int(y)] for y in path["year"]]
        summary["legacy_purchase"]["corporation_tax_forgone"] = {
            "model_year_gbp": tax[m0], "by_carrier_model_year_gbp": dict(corporation_tax_by_carrier),
            "pv_public_rate_gbp": float((path["legacy_corporation_tax_forgone_gbp"].to_numpy() * discount).sum()),
            "held_flat_pv_public_rate_gbp": float(tax[m0] * discount.sum()),
            "basis": LEGACY_CORPORATION_TAX_BASIS,
        }
    return summary, path


def thermal_purchase_diagnostic(register: pd.DataFrame, valuation: pd.DataFrame, case_id: str,
                                allocation: Mapping | None, conv: Mapping, *, public_rate: float, term_years,
                                thermal_basis: str | None) -> dict:
    """What legacy thermal and storage cost on the case's basis, against their owners' own value.

    Since F19 P3 decision 3 (6 Oct 2026) the central is the owners' value (each row at
    its owners' present value, floored at zero), so the gap is zero there; under the
    book-value sensitivity (as-new capex scaled by remaining life, user decision 11's
    old central) the gap and the bill effect of the owners' value instead are reported.
    """
    rows = register.loc[register["case_id"].eq(case_id) & register["finance_vintage"].astype(str).eq("legacy")
                        & register["state_owned"].astype(bool)
                        & ~register["legacy_scope_group"].astype(str).isin([GROUP_VALUED, GROUP_EXCLUDED, "None", "nan"])]
    price = float(_num(rows, "acquisition_price_gbp").sum())
    owners = 0.0
    if len(rows) and valuation is not None and "pv_market_gbp" in valuation.columns:
        values = valuation.drop_duplicates("asset_id").set_index("asset_id").reindex(rows["asset_id"])
        owners = float(pd.to_numeric(values["pv_market_gbp"], errors="coerce").fillna(0.0).clip(lower=0.0).sum())
    # F28 (e): the rows' own price-weighted annuity factor under a remaining-life term.
    charged = float(_num(rows, "annual_capital_recovery_gbp").sum()) if "annual_capital_recovery_gbp" in rows.columns else 0.0
    factor = (charged / price if str(term_years) == TERM_REMAINING_LIFE and price > 0 else
              capital_recovery_factor(float(public_rate), 30.0 if str(term_years) == TERM_REMAINING_LIFE
                                      else float(term_years)))
    annual_gap = (price - owners) * factor
    rate = line_rate(allocation, conv, LINE_LEGACY_THERMAL)
    return {
        "thermal_basis": thermal_basis,
        "label": ("owners' value (each row at its owners' present value, floored at zero; F19 P3 decision 3, "
                  "6 Oct 2026)" if thermal_basis in (None, "market_value") else
                  f"{thermal_basis}: a sensitivity; the owners' own value (the central) is shown beside it"),
        "purchase_price_gbp": price, "owners_present_value_gbp": owners,
        "annual_premium_over_owners_value_gbp": annual_gap,
        "bill_effect_of_market_value_instead_gbp_inc_vat": -annual_gap * rate * (1.0 + float(conv["vat_rate"])),
    }


class RevenueRequirementMemo:
    """:func:`revenue_requirement` once per distinct case rows and arguments (F20 phase 1, 13 Oct 2026).

    The requirement depends on the case's own register rows, private's rows and the
    call's arguments, never on the case's name (it filters by it and names it in
    errors).  89% of cases share their rows with another case (the P3 bound grid
    almost all), and cases that share them are neighbours in register order, so the
    last ``size`` distinct calls are kept: a case whose rows equal one of them bit for
    bit (:func:`register_io.frames_identical`, every column but ``case_id``), against
    the same private rows and arguments, takes a deep copy of its lines and
    diagnostics; anything else is computed as before.  The arguments are compared
    through their JSON text (floats at full precision), ``legacy_gate_fee_gbp`` and
    ``legacy_capital_override`` by identity; one that is neither is never reused.

    F20 phase 2 (27 Oct 2026): a call given ``key`` (:func:`revenue_requirement_key`: the
    case's register body and the one lever the requirement reads off its rows) is matched on
    the key instead of on the rows, so cases sharing their rows but not every lever column
    (the P3 bound grid's LDES floor basis) share it too and no rows are compared; a sample of
    the shared calls (:class:`fingerprints.SampleCheck`) is computed in full and must match.
    """

    def __init__(self, size: int = 4):
        self.size = int(size)
        self._entries: list[tuple] = []          # (private version, signature, case rows, (lines, diagnostics))
        self._frame = None                       # the register frame private was last read from
        self._private = None                     # private's rows in it, and their version
        self._version = 0
        self.hits = self.misses = 0
        self._keyed: list[tuple] = []           # (private version, signature, held, key, result), newest last
        self.keyed_size = 64
        self.sample = fingerprints.SampleCheck("tariff.revenue_requirement")

    @staticmethod
    def _signature(kwargs: Mapping):
        identity = {k: id(kwargs.pop(k)) for k in ("legacy_gate_fee_gbp", "legacy_capital_override") if k in kwargs}
        try:
            text = json.dumps(kwargs, sort_keys=True, allow_nan=True, default=lambda v: f"{type(v).__name__}:{v!r}")
        except (TypeError, ValueError):
            return None
        return text, tuple(sorted(identity.items()))

    def _private_version(self, register: pd.DataFrame) -> int:
        """A number naming private's rows: the same while they are bit for bit the same, batch after batch."""
        from scripts.public_ownership.register_io import frames_identical

        if register is self._frame:
            return self._version
        private = register.loc[register["case_id"].eq("private")]
        if self._private is None or not frames_identical(private, self._private):
            self._version += 1
            self._entries.clear()
        self._frame, self._private = register, private
        return self._version

    def __call__(self, register: pd.DataFrame, *, case_id: str, key: tuple | None = None,
                 **kwargs) -> tuple[dict[str, float], dict]:
        import copy

        from scripts.public_ownership.register_io import frames_identical

        held = {k: kwargs[k] for k in ("legacy_gate_fee_gbp", "legacy_capital_override") if k in kwargs}
        signature = self._signature(dict(kwargs))
        if signature is None:
            self.misses += 1
            return revenue_requirement(register, case_id=case_id, **kwargs)
        version = self._private_version(register)
        if key is not None:
            return self._keyed_call(register, case_id, key, version, signature, held, kwargs)
        rows = register.loc[register["case_id"].eq(case_id)].drop(columns="case_id")
        for i in range(len(self._entries) - 1, -1, -1):
            entry_version, entry_signature, entry_held, entry_rows, result = self._entries[i]
            if (entry_version == version and entry_signature == signature
                    and all(entry_held.get(k) is v for k, v in held.items()) and frames_identical(entry_rows, rows)):
                self._entries.append(self._entries.pop(i))
                self.hits += 1
                return copy.deepcopy(result)
        self.misses += 1
        result = revenue_requirement(register, case_id=case_id, **kwargs)
        self._entries.append((version, signature, held, rows, copy.deepcopy(result)))
        del self._entries[:-self.size]
        return result

    def _keyed_call(self, register, case_id, key, version, signature, held, kwargs):
        """A call matched on its fingerprint (F20 phase 2): the requirement of an earlier case on it, or computed."""
        import copy

        for i in range(len(self._keyed) - 1, -1, -1):
            entry_version, entry_signature, entry_held, entry_key, result = self._keyed[i]
            if (entry_version == version and entry_signature == signature and entry_key == key
                    and all(entry_held.get(k) is v for k, v in held.items())):
                self._keyed.append(self._keyed.pop(i))
                self.hits += 1
                shared = copy.deepcopy(result)
                if self.sample.due():
                    full = revenue_requirement(register, case_id=case_id, **kwargs)
                    self.sample.check(case_id, _same_json(full, shared))
                return shared
        self.misses += 1
        result = revenue_requirement(register, case_id=case_id, **kwargs)
        self._keyed.append((version, signature, held, key, copy.deepcopy(result)))
        del self._keyed[:-self.keyed_size]
        return result


def _same_json(a, b) -> bool:
    """Whether two results are the same, value for value (their JSON text, floats at full precision)."""
    return (json.dumps(a, sort_keys=True, allow_nan=True, default=repr)
            == json.dumps(b, sort_keys=True, allow_nan=True, default=repr))


def revenue_requirement_key(case_id: str, settings: Mapping[str, str], *, vintaged: bool) -> tuple:
    """:func:`revenue_requirement` of a case is a function of this key and the call's arguments (F20 phase 2).

    It reads the case's register rows (:func:`fingerprints.body_key`) and one lever off them,
    ``beccs_removal_funding``; private's rows and the arguments are matched apart.
    """
    return (fingerprints.body_key(case_id, settings, vintaged=vintaged),
            str(settings.get("beccs_removal_funding", "exchequer")))


# ── CLI ───────────────────────────────────────────────────────────────────────

def purchase_schedules(case_rows: pd.DataFrame, ic_detail: Mapping | None, term_years) -> dict:
    """The legacy purchase's and the GB halves' annuity schedules for one case (F28 (e)).

    Under a remaining-life term the legacy schedule is the case's acquired rows' own terms
    (``acquisition_term_years_applied``) weighted by their first-year charges (capital recovery and
    any cancelled-contract residual), and the halves' is the interconnector lines' own; under a
    fixed term both are one window on it.
    """
    if term_years != TERM_REMAINING_LIFE:
        fixed = AnnuitySchedule.fixed(float(term_years))
        return {"legacy": fixed, LINE_IC_ACQUISITION: fixed}
    price = pd.to_numeric(case_rows.get("acquisition_price_gbp"), errors="coerce")
    owned = case_rows["state_owned"].astype(bool) if "state_owned" in case_rows.columns else pd.Series(True, index=case_rows.index)
    acquired = case_rows.loc[price.notna() & owned]
    charge = _num(acquired, "annual_capital_recovery_gbp") + (_num(acquired, "annual_cancelled_contract_residual_gbp")
                                                              if "annual_cancelled_contract_residual_gbp" in acquired.columns
                                                              else 0.0)
    terms = (pd.to_numeric(acquired["acquisition_term_years_applied"], errors="coerce")
             if "acquisition_term_years_applied" in acquired.columns else pd.Series(np.nan, index=acquired.index))
    if terms[charge > 0].isna().any():
        raise ValueError("a remaining-life term needs acquisition_term_years_applied on every acquired register row")
    legacy = AnnuitySchedule.from_rows(terms, charge)
    record = (ic_detail or {}).get("interconnector_annuity_schedule")
    return {"legacy": legacy, LINE_IC_ACQUISITION: AnnuitySchedule.from_record(record) if record else legacy}


def fees_schedule(schedules: Mapping, fees_gbp: float, share: float, legacy_annual: float,
                  ic_annual: float) -> AnnuitySchedule:
    """The transaction fees run off with the parts they were paid on (the wires' over :data:`WIRES_TERM_YEARS`)."""
    wires = max(0.0, float(fees_gbp) - float(share) * (float(legacy_annual) + float(ic_annual)))
    parts = [(schedules["legacy"], float(share) * float(legacy_annual)),
             (schedules[LINE_IC_ACQUISITION], float(share) * float(ic_annual))]
    if wires > 0.0:
        parts.append((AnnuitySchedule.fixed(WIRES_TERM_YEARS), wires))
    if sum(charge for _, charge in parts) <= 0.0:
        return schedules["legacy"]
    return AnnuitySchedule.combine(parts)


def carried_schedules(carried: Mapping) -> dict:
    """A carried purchase's schedules (F28 (e)): the earlier year's own, else one window on its term."""
    records = carried.get("annuity_schedules") or {}
    fixed = AnnuitySchedule.fixed(float(carried["term_years"]))
    out = {line: AnnuitySchedule.from_record(record) for line, record in records.items()}
    out.setdefault("legacy", fixed)
    for line in RUN_OFF_LINES:
        out.setdefault(line, out["legacy"])
    return out


def three_parts_path(multi_year_path) -> Path:
    """``{run_group}_nationalised_bill_multi_year.csv`` -> ``{run_group}_nationalised_bill_three_parts.csv``."""
    path = Path(multi_year_path)
    stem = path.stem[: -len("_multi_year")] if path.stem.endswith("_multi_year") else path.stem
    return path.with_name(f"{stem}_three_parts.csv")


def replaced_lines_reported(bill_by_case: pd.DataFrame) -> dict[str, str]:
    """The replaced lines a summary lists: every 2030 one, and a 2050-only line when the bill carries it."""
    present = set(bill_by_case["component_id"].astype(str)) if "component_id" in bill_by_case.columns else set()
    return {**REPLACED_LINES, **{k: v for k, v in OPTIONAL_REPLACED_LINES.items() if k in present}}


#: F20 phase 0: the per-case steps timed when PYPSA_GB_STAGE_TIMING is set (scripts/public_ownership/stage_timing.py).
TIMED_FUNCTIONS = ("revenue_requirement", "build_bill", "reconcile", "recovery_rates", "multi_year",
                   "three_part_split", "break_even_purchase_price", "lifetime_break_even_price",
                   "thermal_purchase_diagnostic", "capacity_market_not_acquired", "interconnector_lines",
                   "transaction_costs", "legacy_levy_lines", "headline_years", "tariff_cm_recovery_options")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("finance_register")
    parser.add_argument("bill_by_case")
    parser.add_argument("legacy_valuation")
    parser.add_argument("legacy_profile")
    parser.add_argument("assumptions")
    parser.add_argument("recovery")
    parser.add_argument("out_components")
    parser.add_argument("out_summary")
    parser.add_argument("out_reconciliation")
    parser.add_argument("out_multi_year")
    parser.add_argument("--config", default="config")
    parser.add_argument("--batch-cases", type=int, default=8,
                        help="tariff cases read at a time with private (memory is flat in the number of cases)")
    parser.add_argument("--forecast-summary", default=None,
                        help="The forecast bill summary JSON (CfD economic curtailment, balancing breakdown) for the N7 bridge")
    parser.add_argument("--bill-summary", default=None,
                        help="The per-case bill summary JSON (CfD recovery denominator)")
    parser.add_argument("--realtime-correction-gbp", type=float, default=0.0,
                        help="Resource cost of resolving day-ahead forecast error (institution I3); zero for I1")
    parser.add_argument("--network-account", default=None,
                        help="Network finance account summary JSON (item N6). When given and "
                             "network_ownership is acquired, the TNUoS and DUoS lines fall by its saving")
    parser.add_argument("--interconnector-trade-gbp", type=float, default=None,
                        help="Measured net payment to foreign counterparties (link flow x foreign price, negative "
                             "when a net exporter). Replaces the register's domestic import revenue and the "
                             "day-ahead re-trading line; see scripts/public_ownership/nationalised_institutions.py")
    parser.add_argument("--market-scenarios", nargs="*", default=None,
                        help="The solved market months; when given and --interconnector-trade-gbp is not, the net payment "
                             "to foreign counterparties is measured from their balancing networks (item N10)")
    parser.add_argument("--institution", default="I1", choices=("I1", "I2", "I3"),
                        help="Dispatch institution the register was built on. Under I1 the interconnector re-trading "
                             "cost stays as a tariff line; under I2 and I3 there is no position to unwind")
    parser.add_argument("--wholesale-scenarios", nargs="*", default=None,
                        help="The month whose Stage 1 files each market month reads (its wholesale_from parent), "
                             "paired with --market-scenarios; default the months themselves (item BM1, P3)")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Tariff only these cases (a tariff balancing group's bid case, item BM1 P3); "
                             "default every case in the register that acquires the legacy fleet")
    parser.add_argument("--tariff-group", default=None,
                        help="Tariff the cases this run group prices (reference_runs.group_tariff_cases on --config); "
                             "used instead of --cases, whose list outgrew the Windows command line at 133 cases")
    parser.add_argument("--headline-bill-summary", default=None,
                        help="The market group's per-case bill summary: the headline private bill every multi-year "
                             "series, headline and break-even is measured against (review 29 Sep 2026)")
    parser.add_argument("--cm-rates", default=None,
                        help="cm_revenue_by_technology.csv (item MR4); its interconnector row is the Capacity Market "
                             "revenue whose GB halves leave the kept line in every tariff case, which buys the GB "
                             "halves whether or not it buys the wires (item D3)")
    parser.add_argument("--stage-1-prices", nargs="*", default=None,
                        help="item F12 (12 Oct 2026): the GB price files the cables' Stage 1 congestion rent is valued "
                             "at, one per market month (a price-level twin run group's mapped prices); default each "
                             "Stage 1 month's own *_wholesale_price.csv")
    parser.add_argument("--financing-summary", default=None,
                        help="FES 2050 decision 73: the register's financing summary, whose capacity_market_design_* "
                             "columns price the adequacy-planning contract that replaces the CM under an acquisition; "
                             "without it (every 2030 run) the kept CM line is today's")
    parser.add_argument("--carried-acquisition", default=None,
                        help="acquisition_carry_forward.py output (item Y12, decision 2): the purchase made in an "
                             "earlier year, at its own terms; legacy rows then carry operating cost only. The CP30 "
                             "year never passes it")
    parser.add_argument("--legacy-profile-by-valuation-settings", "--legacy-profile-by-price-path",
                        dest="legacy_profile_by_settings", default=None,
                        help="legacy_valuation.py's profile at each point of the valuation levers (legacy_price_path, "
                             "legacy_asset_lives, capacity_procurement): each case's multi-year view, lifetime "
                             "break-even and three parts run on its own settings' streams (F19 valuation decisions, "
                             "14 Oct 2026; generalised 27 Oct 2026); default every case on legacy_profile")
    args = parser.parse_args(argv)
    # F20 phase 0: opt-in stage timing (PYPSA_GB_STAGE_TIMING); without it nothing is wrapped.
    from scripts.public_ownership import stage_timing

    stage_timing.instrument_register_io()
    stage_timing.instrument(globals(), TIMED_FUNCTIONS, prefix="tariff")
    stage_timing.instrument(RevenueRequirementMemo, ("__call__",), prefix="tariff.RevenueRequirementMemo")
    lap = stage_timing.laps("tariff.main")

    po = (read_config(args.config) or {}).get("public_ownership") or {}
    acquisition = po.get("acquisition") or {}
    carried_table = (json.loads(Path(args.carried_acquisition).read_text(encoding="utf-8"))
                     if args.carried_acquisition else None)
    # FES 2050 decision 73: the adequacy contract's price (no 2030 rule passes it).
    financing_summary = pd.read_csv(args.financing_summary) if args.financing_summary else None
    from scripts.public_ownership.register_io import RegisterReader, read_register, settings_path

    # The tariff prices the cases that acquire the legacy fleet, against private: read only those.
    # 30 Sep 2026: private, then the tariff cases a batch at a time (each case reads only its
    # own rows and private's), so memory is flat in the number of cases.
    streamed = settings_path(args.finance_register).exists() and not str(args.finance_register).endswith(".csv")
    if streamed:
        from scripts.public_ownership.cfd_repricing import register_settings_by_case

        settings_by_case = register_settings_by_case(args.finance_register)
        tariffed = tariff_case_ids_from_settings(settings_by_case)
        register = read_register(args.finance_register, cases=["private"])
    else:
        register = read_register(args.finance_register)
        tariffed = tariff_case_ids(register)
        settings_by_case = {case_id: case_settings(rows) for case_id, rows in register.groupby("case_id", sort=False)}
    # F20 phase 2: the revenue requirement once per fingerprint (a register keyed by its lever columns only).
    share_requirement = fingerprints.sharing_enabled() and (fingerprints.register_is_keyed(args.finance_register)
                                                           if streamed else fingerprints.frame_is_keyed(register))
    # A case moving only a later-register lever carries the earlier purchase at its settings (carried_case_id).
    carry_settings = (pd.read_csv(settings_path(args.finance_register))
                      if carried_table is not None and settings_path(args.finance_register).exists() else None)
    bill_by_case = pd.read_csv(args.bill_by_case)
    valuation = pd.read_csv(args.legacy_valuation)
    profile = pd.read_csv(args.legacy_profile)
    profile_by_settings = (pd.read_csv(args.legacy_profile_by_settings) if args.legacy_profile_by_settings else None)
    assumptions = json.loads(Path(args.assumptions).read_text(encoding="utf-8"))
    recovery = json.loads(Path(args.recovery).read_text(encoding="utf-8"))
    bill_summary = json.loads(Path(args.bill_summary).read_text(encoding="utf-8")) if args.bill_summary else None
    forecast = json.loads(Path(args.forecast_summary).read_text(encoding="utf-8")) if args.forecast_summary else {}
    balancing = forecast.get("balancing_costs") or {}
    bill_detail = {
        "cfd_economic_curtailment_payment_gbp": float(forecast.get("cfd_economic_curtailment_payment_gbp") or 0.0),
        "interconnector_retrading_cost_gbp": (
            float(balancing.get("interconnector_retrading_cost_gbp") or 0.0)
            if balancing.get("include_interconnector_retrading") else 0.0),
    }
    ic_retrading = bill_detail["interconnector_retrading_cost_gbp"] if args.institution == "I1" else 0.0
    conv = conversion(assumptions, recovery, bill_summary)
    ic_pass = (forecast.get("diagnostics") or {}).get(INTERCONNECTOR_PASS_THROUGH_LINE) or {}
    ic_pass_hh = (forecast.get("diagnostics") or {}).get("interconnector_cap_and_floor_credit_gbp_per_3100kwh")
    if ic_pass and ic_pass_hh:
        # FES 2050 decision 61, applied to 2030 too (user, 3 Oct 2026): the pass-through line's system
        # amount is the bill's own credit (it is a credit, so a negative line), not its household value
        # over final demand (TNUoS residual basis).
        conv["interconnector_pass_through_household_gbp"] = -float(ic_pass_hh)
        conv["interconnector_pass_through_system_gbp"] = -float(ic_pass["credit_to_consumers_gbp"])
    # Review 29 Sep 2026: the domestic supply the levies moved to taxation are recovered over.
    whd = ((assumptions.get("sourced_externals") or {}).get("warm_home_discount") or {}).get("derivation") or {}
    if whd.get("domestic_electricity_supply_mwh") is not None:
        conv["domestic_mwh"] = float(whd["domestic_electricity_supply_mwh"])
    cm_book_gbp = (((assumptions.get("sourced_externals") or {}).get("capacity_market") or {})
                   .get("derivation") or {}).get("agreement_cost_gbp")
    model_year = int(register["model_year"].iloc[0])
    public_rate = float(valuation["public_real_rate"].iloc[0])
    valuation_horizon = int(valuation["horizon_year"].max()) if "horizon_year" in valuation.columns else None
    if carried_table is not None and int(carried_table["model_year"]) != model_year:
        raise ValueError(f"--carried-acquisition is for {carried_table['model_year']}, the register for {model_year}")

    # Imported here: network_finance_account and nationalised_institutions import this module.
    from scripts.public_ownership.network_finance_account import (
        adjustments_for_treatment, break_even_summary, incidence_amounts, select_rows,
    )

    stage_1_scenarios = args.wholesale_scenarios or args.market_scenarios
    if args.market_scenarios and len(stage_1_scenarios) != len(args.market_scenarios):
        raise ValueError(f"--wholesale-scenarios names {len(stage_1_scenarios)} months for "
                         f"{len(args.market_scenarios)} market months; they must pair month by month")
    if args.interconnector_trade_gbp is None and args.market_scenarios:
        from scripts.public_ownership.nationalised_institutions import measure_link_trade

        args.interconnector_trade_gbp = float(measure_link_trade(
            args.market_scenarios, schedule_scenarios=stage_1_scenarios)["interconnector_cost_gbp"])

    # Item D3: the cables' Stage 1 congestion rent over the year (market months at
    # their own copperplate price) and the GB share of each, from the bill's regime table.
    ic_rent, ic_shares, ic_capacity, ic_cm_gbp, ic_new_build, ic_cables = None, {}, {}, 0.0, {}, {}
    ic_settings = interconnector_acquisition_settings(acquisition.get("interconnectors"))
    if args.interconnector_trade_gbp is not None and args.market_scenarios:
        from scripts.market.fti_settlement import gb_half_shares
        from scripts.public_ownership.nationalised_institutions import (
            interconnector_capacity_by_cable, measure_congestion_rent,
        )

        ic_rule = (assumptions.get("model_linked_rules") or {}).get("interconnector_cap_and_floor") or {}
        if not ic_rule.get("cables"):
            raise ValueError(f"{args.assumptions} declares no interconnector regimes; item D3 needs the GB shares")
        ic_shares = gb_half_shares(ic_rule)
        ic_cables = dict(ic_rule["cables"])
        # Cables commissioned in the model year (MaresConnect, 2030): GB half bought at cost, not at market value.
        ic_new_build = interconnector_new_build_floors(ic_rule)
        # A Stage 1 quantity: each month's parent carries the schedule, the price and the shared build.
        ic_rent = (measure_congestion_rent(stage_1_scenarios) if not args.stage_1_prices
                   else measure_congestion_rent(stage_1_scenarios, price_paths=args.stage_1_prices))
        ic_capacity = interconnector_capacity_by_cable(args.market_scenarios[0])
        if args.cm_rates:
            rates_cm = pd.read_csv(args.cm_rates).set_index("carrier")
            if "interconnector" not in rates_cm.index:
                raise ValueError(f"{args.cm_rates} has no interconnector row")
            ic_cm_gbp = float(rates_cm.at["interconnector", cm_calendar_revenue_column(model_year)])

    network_account, network_lines = None, None
    if args.network_account:
        network_account = json.loads(Path(args.network_account).read_text(encoding="utf-8"))
        network_lines = pd.read_csv(Path(args.network_account).with_suffix(".csv"))

    # Item K1: one tariff per case that acquires the legacy fleet, read off the
    # register's lever columns; the network lines move by the case's networks
    # lever (acquired: the RAV account of item N6).
    cases = tariffed
    if not cases:
        raise ValueError("the register carries no case that acquires the legacy fleet; nothing to tariff")
    requested = args.cases
    if args.tariff_group:
        if requested:
            raise ValueError("pass --cases or --tariff-group, not both")
        from scripts.public_ownership.reference_runs import group_tariff_cases
        requested = group_tariff_cases(po, args.tariff_group)
    cases = select_tariff_cases(cases, requested)
    primary = next((c for c in cases if c.endswith("@fully_nationalised") or c == NATIONALISED_CASE), cases[0])
    # Review 29 Sep 2026: every multi-year series and headline is measured against the
    # headline private bill, the market group's (1,024.25), never against private on a
    # balancing group's own months (the P3 group's private is 1,006.32): a 17.9 offset.
    headline_cases: dict = {}
    if args.headline_bill_summary:
        headline = json.loads(Path(args.headline_bill_summary).read_text(encoding="utf-8"))
        headline_cases = headline.get("cases") or {}
    lever_block = dict(po.get("lever_parameters") or {})
    # OW10: Sizewell C's RAB return at the public rate, on the kept levy (off unless configured).
    # D4 (user, 14 Nov 2026): carried into the later years too. Their registers keep Sizewell C private on its RAB in
    # every case (in construction, pre-legislation), so the public-rate return is applied as in 2030, at the case's
    # state share, once the plant runs on the bill's own sizewell_c_rab line (it was switched off from 2040).
    sizewell_settings = sizewell_c_rab.settings_from_config(po.get("consumer_bill"))
    private_bill = bill_by_case.loc[bill_by_case["case_id"].eq("private")].set_index("component_id")[VALUE]
    # F19 P3 decision 4: the kept remaining_ccs line is the NZT DPA (a 2050 bill replaces its dpa line).
    dpa_kept_line = "remaining_ccs" in private_bill.index and "dpa" not in private_bill.index
    # Money audit (13 Nov 2026): the bill's own Sizewell C RAB line (kept on every case that does not own Sizewell C;
    # where a case owns it, register_replaced_lines replaces the line and the RAB rows are costed, not paid as today).
    rab_kept_line = RAB_LINE in private_bill.index
    # F19 P3 decision 6: the lines kept under this institution (the imbalance allowance under I1).
    kept_replaced = {k: v for k, v in kept_replaced_lines(args.institution).items() if k in private_bill.index}
    reports, component_frames, recon_frames, year_frames, part_frames = {}, [], [], [], []
    # FES 2050 decision 55: plant the earlier purchase bought that the later register holds as new
    # build (HPC unit 1), and Sizewell C acquired at its RAV (vintage_capital_2050; absent in 2030).
    vintage_block = po.get("vintage_capital_2050") or {}
    decision55_contracts = [str(c) for c in (vintage_block.get("carried_from_earlier_purchase_contracts") or [])]
    decision55_rav = bool(vintage_block.get("rab_acquired_at_rav", False))
    # Review 2: the MW of each carried contract the earlier purchase bought (absent: every tagged row whole).
    decision55_mw = {str(k): float(v) for k, v in (vintage_block.get("carried_from_earlier_purchase_mw") or {}).items()}
    if set(decision55_mw) - set(decision55_contracts):
        raise ValueError("vintage_capital_2050.carried_from_earlier_purchase_mw names contracts not carried")

    def _case_registers():
        """Each case with a register holding its rows and private's (the whole register when not streamed)."""
        if not streamed:
            yield from ((case_id, register) for case_id in cases)
            return
        # F20 phase 1: private's row groups are read once, not with every batch (the frames are unchanged).
        reader = RegisterReader(args.finance_register, pinned=["private"])
        for start in range(0, len(cases), args.batch_cases):
            batch = cases[start:start + args.batch_cases]
            frame = reader.read(["private", *batch])
            yield from ((case_id, frame) for case_id in batch)

    # F19: the gate fees the purchase price counts (2030), by asset; the same for every case.
    legacy_gate_fees = (pd.to_numeric(valuation.drop_duplicates("asset_id").set_index("asset_id")["gate_fee_model_year_gbp"],
                                      errors="coerce")
                        if "gate_fee_model_year_gbp" in valuation.columns else None)
    # F20 phase 1: cases sharing their register rows share their revenue requirement.
    requirement = RevenueRequirementMemo()
    lap("setup")
    for case_id, register in _case_registers():
        case_rows = register.loc[register["case_id"].eq(case_id)]
        levers = case_settings(case_rows)
        # F19 valuation decisions (user, 14 Oct 2026; generalised 27 Oct 2026): the streams of the case's
        # own valuation settings (legacy_price_path, legacy_asset_lives, capacity_procurement), so its
        # multi-year view, lifetime break-even and three parts move with them.
        case_profile = legacy_case_profile(profile_by_settings, levers, profile)
        # ... and the corporation tax its acquired legacy rows no longer pay, profiled on those streams.
        legacy_tax = legacy_corporation_tax_by_carrier(case_rows)
        treatment = levers.get(LEVER_NETWORKS, NETWORKS_PRIVATE)
        # House rule 3: a case that moves a background (all_cases) lever, such as legacy
        # levies to taxation, is measured against the private case at the same point.
        comparator_id = comparator_for_settings(case_id, {
            **{c: (r.get("levers") or {}) for c, r in headline_cases.items() if c.split("@", 1)[0] == "private"},
            case_id: levers}) if headline_cases else "private"
        headline_private = headline_cases.get(comparator_id) or headline_cases.get("private")
        network_adjustments: dict[str, float] = {}
        network_incidence: dict = {}
        network_break_even: dict = {}
        if treatment != NETWORKS_PRIVATE:
            if network_lines is None:
                raise ValueError(f"{case_id} takes networks: {treatment} and needs --network-account")
            network_adjustments = adjustments_for_treatment(network_lines, treatment, levers)
            network_incidence = incidence_amounts(select_rows(network_lines, treatment, levers), treatment, levers)
            network_break_even = (break_even_summary(select_rows(network_lines, treatment, levers))
                                  if treatment == "acquired" else {})
        # F28 (e): each asset over its own remaining life (central), or one term for every asset.
        term_setting = str(levers.get("acquisition_term_years", acquisition.get("term_years", 30)))
        term_years = TERM_REMAINING_LIFE if term_setting == TERM_REMAINING_LIFE else float(term_setting)
        # FES 2050 decision 73: adequacy planning replaces the CM (none without --financing-summary's design).
        cm_design = adequacy_contract(financing_summary, case_id, conv)
        ic_lines, ic_detail, ic_cm_acquired = None, None, 0.0
        if ic_rent is not None:
            # Every tariff case buys the GB halves (user decision 26 Sep 2026), at the
            # interconnector_price lever's owner rate (29 Sep 2026).
            case_ic = interconnector_price_settings(ic_settings, acquisition.get("interconnectors"),
                                                    levers.get("interconnector_price", "central"))
            ic_lines, ic_detail = interconnector_lines(
                ic_rent, ic_shares, settings=case_ic, public_rate=public_rate, term_years=term_years,
                new_build_floor_gbp=ic_new_build,
                # F28 (f): each cable valued over its own remaining regime or asset life.
                remaining_years=interconnector_remaining_years(ic_cables, case_ic, int(model_year)))
            share = interconnector_cm_acquired_share(ic_capacity, ic_shares)
            ic_cm_case = ic_cm_gbp if cm_design is None else cm_design["interconnector_payment_report_year_gbp"]
            ic_cm_acquired = ic_cm_case * share
            ic_detail.update({"interconnector_capacity_by_cable_mw": ic_capacity,
                              "interconnector_cm_revenue_report_year_gbp": ic_cm_case,
                              "interconnector_cm_acquired_share": share,
                              "interconnector_cm_acquired_report_year_gbp": ic_cm_acquired})
        # D2 (user, 14 Nov 2026): a carried year's capacity added after the purchase, built by the state at cost.
        ic_post_cost, ic_post_detail = 0.0, None
        if carried_table is not None and ic_lines is not None and ic_capacity:
            ic_post_detail = interconnector_post_acquisition_new_build(ic_capacity, ic_shares, ic_new_build)
            ic_post_cost = float(ic_post_detail["annual_gbp"])
            ic_detail["interconnector_post_acquisition_new_build"] = ic_post_detail
        carried = None
        if carried_table is not None:
            carried_cases = carried_table.get("cases") or {}
            carried_id = carried_case_id(case_id, carried_cases, carry_settings)
            carried = carried_cases.get(carried_id) if carried_id else None
            if carried is None:
                raise ValueError(f"{case_id}: --carried-acquisition has no {carried_table.get('acquisition_year')} "
                                 "case with this id to carry")
        # F19: the gate fees the purchase price counts (2030), by asset (F20: read once, before the loop).
        gate_fees = legacy_gate_fees
        requirement_key = (revenue_requirement_key(case_id, settings_by_case[case_id],
                                                   vintaged=is_vintaged_model_year(model_year))
                           if share_requirement and case_id in settings_by_case else None)
        lines, diagnostics = requirement(register, realtime_correction_gbp=args.realtime_correction_gbp,
                                         key=requirement_key,
                                         interconnector_retrading_gbp=ic_retrading,
                                         interconnector_trade_gbp=args.interconnector_trade_gbp, case_id=case_id,
                                         interconnector=ic_lines, interconnector_detail=ic_detail,
                                         carried_acquisition=carried,
                                         # FES 2050 decision 55 (none in 2030: no carried purchase).
                                         carried_contracts=decision55_contracts, rab_at_rav=decision55_rav,
                                         rav_rate=public_rate, carried_mw=decision55_mw,
                                         # F19 P3 decision 4: the bill keeps the DPA's own line.
                                         dpa_paid_by_kept_line=dpa_kept_line,
                                         rab_paid_by_kept_line=rab_kept_line,
                                         legacy_gate_fee_gbp=gate_fees,
                                         interconnector_post_acquisition_gbp=ic_post_cost)
        # Item MR4: the Capacity Market agreements the state does not buy stay on the bill.
        if cm_design is None:
            lines[LINE_CM_NOT_ACQUIRED], cm_detail = capacity_market_not_acquired(bill_by_case, diagnostics, conv,
                                                                                  cm_book_gbp, ic_cm_acquired)
        else:
            lines[LINE_CM_NOT_ACQUIRED], cm_detail = capacity_market_not_acquired(
                bill_by_case, diagnostics, conv, cm_book_gbp, ic_cm_acquired,
                contract_cost_gbp=cm_design["line_cost_report_year_gbp"])
            cm_detail.update({"capacity_market_design": cm_design["moves"],
                              "capacity_market_household_payment_change_gbp": cm_design["moves"]["household_change_gbp"]})
        diagnostics.update(cm_detail)
        # F28 (e): how each purchase annuity runs off (asset by asset under a remaining-life term).
        legacy_annual = float(lines.get(LINE_LEGACY_RENEWABLES, 0.0)) + float(lines.get(LINE_LEGACY_THERMAL, 0.0))
        ic_annual = float((ic_lines or {}).get(LINE_IC_ACQUISITION, 0.0))
        schedules = purchase_schedules(case_rows, ic_detail, term_years)
        # Lever acquisition_transaction_costs: fees on the purchase, the GB halves and the wires.
        fee_share = float((acquisition.get("transaction_cost_share") or {}).get(
            levers.get("acquisition_transaction_costs", "none"), 0.0))
        lines[LINE_TRANSACTION_COSTS] = transaction_costs(
            diagnostics, ic_detail, network_incidence, share=fee_share,
            public_rate=public_rate, term_years=term_years, network_lines=network_lines, treatment=treatment,
            levers=levers, legacy_annual_gbp=legacy_annual, ic_annual_gbp=ic_annual)
        schedules[LINE_TRANSACTION_COSTS] = fees_schedule(schedules, lines[LINE_TRANSACTION_COSTS], fee_share,
                                                          legacy_annual, ic_annual)
        if carried is not None:
            # Item Y12: the fees were paid on the earlier purchase and annuitised with it.
            lines[LINE_TRANSACTION_COSTS] = float(carried["gross_lines_model_year_gbp"].get(LINE_TRANSACTION_COSTS, 0.0))
            # F28 (e): the earlier purchase's own schedules (one window on its term before them).
            schedules = carried_schedules(carried)
        # Lever legacy_levy_funding (background): domestic FiT and WHD move to taxation; the
        # RO has already gone with the tariff (the ROC streams are bought with the plant).
        lines[LINE_LEGACY_FIT], lines[LINE_LEGACY_WHD] = legacy_levy_lines(bill_by_case, conv, levers)
        # Lever acquisition_financing (user decision 5): the annuity funded from general
        # taxation instead of the tariff.  The vendors and lenders are paid either way.
        exchequer_annuity = 0.0
        if levers.get("acquisition_financing", "tariff") == "exchequer":
            exchequer_annuity = float(sum(lines.get(line, 0.0) for line in ANNUITY_LINES))
            diagnostics["acquisition_annuity_exchequer_funded_by_line_gbp"] = {
                line: float(lines.get(line, 0.0)) for line in ANNUITY_LINES}
            for line in ANNUITY_LINES:
                lines[line] = 0.0
        diagnostics["acquisition_annuity_exchequer_funded_gbp"] = exchequer_annuity
        diagnostics["network_incidence"] = {k: {kk: vv for kk, vv in v.items() if kk != "sector"}
                                            for k, v in network_incidence.items()}
        # Like-for-like allocation (audit 26 Sep 2026): the kept parts of replaced
        # lines come out of the pool and go back at their own rule.
        extra_replaced = register_replaced_lines(diagnostics)
        allocation = recovery_rates(bill_by_case, conv, {LINE_CM_NOT_ACQUIRED: lines[LINE_CM_NOT_ACQUIRED],
                                                         LINE_IC_RETRADING: lines[LINE_IC_RETRADING]},
                                    extra_replaced=extra_replaced, kept_replaced=kept_replaced)
        dropped = dropped_lines_for(diagnostics)
        # OW10: never where the register books Sizewell C's RAB itself (the levy is then replaced).
        sizewell_share = (sizewell_c_rab.state_new_build_share(register.loc[register["case_id"].eq(case_id)])
                          if sizewell_settings["enabled"] and sizewell_c_rab.LINE not in extra_replaced else 0.0)
        # F19 decision 1: the public rate plus the construction premium, on the capital beyond the existing share.
        sizewell_rate = (sizewell_c_rab.public_rate(network_lines, levers, fallback=public_rate)
                         + sizewell_c_rab.premium(sizewell_settings, levers) if sizewell_share > 0 else None)
        sizewell = {"sizewell_adjustment": (sizewell_c_rab.household_adjustment(
            float(private_bill.get(sizewell_c_rab.LINE, 0.0)), sizewell_settings, state_share=sizewell_share,
            rate=sizewell_rate) if sizewell_share > 0 else 0.0)}
        sizewell_note = (sizewell_c_rab.method_note(sizewell_settings, state_share=sizewell_share, rate=sizewell_rate)
                         if sizewell["sizewell_adjustment"] else "")
        diagnostics["sizewell_c_rab_public_rate"] = ({"state_share": sizewell_share, "state_rate": sizewell_rate,
                                                      "existing_state_capital_share":
                                                          sizewell_c_rab.existing_state_capital_share(sizewell_settings),
                                                      "refinanced_share":
                                                          sizewell_c_rab.refinanced_share(sizewell_settings, sizewell_share),
                                                      **sizewell_settings, **sizewell}
                                                     if sizewell["sizewell_adjustment"] else None)
        bill_args = dict(case_id=case_id, source=str(args.finance_register), network_adjustments=network_adjustments,
                         allocation=allocation, dropped_lines=dropped, extra_replaced=extra_replaced,
                         kept_replaced=kept_replaced, sizewell_note=sizewell_note, **sizewell)
        components, summary = build_bill(bill_by_case, lines, conv, **bill_args)
        # F27 item 10: EBIT and headroom on the tariff's own costs (2030; a vintaged year keeps private's).
        supplier = {"supplier_adjustment": (
            supplier_margin_adjustment(components, (bill_summary or {}).get("supplier_margin"))
            if not is_vintaged_model_year(int(model_year)) else 0.0)}
        if supplier["supplier_adjustment"]:
            components, summary = build_bill(bill_by_case, lines, conv, **bill_args, **supplier)
        recon, recon_summary = reconcile(bill_by_case, components, lines, diagnostics, conv, bill_detail,
                                         network_adjustments=network_adjustments, allocation=allocation,
                                         dropped_lines=dropped, extra_replaced=extra_replaced,
                                         kept_replaced=kept_replaced, **sizewell, **supplier)
        recon.insert(0, "case_id", case_id)
        headline_delta_ex_vat = (summary["ex_vat_gbp"] - float(headline_private["ex_vat_gbp"])
                                 if headline_private else summary["delta_ex_vat_gbp"])
        # Item Y12: a carried purchase runs from its own year and term; every view is
        # checked against the legacy profile's horizon (no silent zero years).
        years = multi_year(case_profile, lines, conv, schedules=schedules,
                           model_year=model_year, bill_delta_model_year_gbp_per_hh=headline_delta_ex_vat,
                           allocation=allocation, horizon_year=valuation_horizon,
                           acquisition_year=(int(carried["acquisition_year"]) if carried else None),
                           corporation_tax_by_carrier=legacy_tax if carried is None else None)
        years.insert(0, "case_id", case_id)
        years["basis"] = (f"against the headline private bill (the market group's {comparator_id})" if headline_private
                          else "against this group's own private bill")
        # User answer Q5 (30 Sep 2026): tariff_pricing: market_bill.  The state charges the
        # private comparator's bill (the headline private, else this group's) and the
        # Treasury keeps the margin over the tariff's cost: one line, paid to the
        # Exchequer, sized so the bill equals the comparator's.  The multi-year view above
        # stays on the cost basis (the margin is its gap to private in each year).
        # The margin is recovered in the proportions of the tariff's own saving (its household
        # change over its system change), so it takes back from every consumer exactly what
        # the cost tariff saved them: the benchmark household's bill is the comparator's and
        # the consumers' system change is zero at once.
        treasury_margin = 0.0
        if levers.get("tariff_pricing", "cost") == "market_bill":
            # The comparator is this group's own private bill, the one the reconciliation's system
            # change is measured against (on the market group it is also the headline private).
            target_ex_vat = float(summary["private_ex_vat_gbp"])
            gap_hh = target_ex_vat - summary["ex_vat_gbp"]
            gap_system = -float(recon_summary["bill_change_system_gbp"])
            if abs(gap_system) < 1.0 or gap_hh * gap_system <= 0:
                raise ValueError(f"{case_id}: tariff_pricing market_bill needs a tariff whose household and system "
                                 f"changes share a sign (household {gap_hh:.4f}, system {gap_system:,.0f})")
            treasury_margin = gap_system
            lines[LINE_TREASURY_MARGIN] = treasury_margin
            allocation = dict(allocation)
            allocation["new_line_rates"] = dict(allocation["new_line_rates"], **{LINE_TREASURY_MARGIN: gap_hh / gap_system})
            allocation["new_line_rules"] = dict(allocation["new_line_rules"], **{LINE_TREASURY_MARGIN: TREASURY_MARGIN_RULE})
            components, summary = build_bill(bill_by_case, lines, conv, case_id=case_id,
                                             source=str(args.finance_register),
                                             network_adjustments=network_adjustments, allocation=allocation,
                                             dropped_lines=dropped, extra_replaced=extra_replaced,
                                             kept_replaced=kept_replaced, sizewell_note=sizewell_note, **sizewell)
            recon, recon_summary = reconcile(bill_by_case, components, lines, diagnostics, conv, bill_detail,
                                             network_adjustments=network_adjustments, allocation=allocation,
                                             dropped_lines=dropped, extra_replaced=extra_replaced,
                                             kept_replaced=kept_replaced, **sizewell)
            recon.insert(0, "case_id", case_id)
            if abs(summary["ex_vat_gbp"] - target_ex_vat) > 1e-6 or abs(recon_summary["bill_change_system_gbp"]) > 1.0:
                raise ValueError(f"{case_id}: tariff_pricing market_bill left the bill at {summary['ex_vat_gbp']:.6f} "
                                 f"(comparator {target_ex_vat:.6f}) and a system change of "
                                 f"{recon_summary['bill_change_system_gbp']:,.2f}")
            years["basis"] = years["basis"] + ("; tariff at cost: under tariff_pricing market_bill the bill is the "
                                               "comparator's and the Treasury keeps the gap")
        diagnostics["treasury_margin_gbp"] = treasury_margin
        legacy_price = float(diagnostics.get("acquisition_price_gbp", 0.0) or 0.0)
        break_even = break_even_purchase_price(
            summary, diagnostics, allocation, conv, public_rate=public_rate,
            term_years=None if term_years == TERM_REMAINING_LIFE else term_years,
            annuity_factor=((legacy_annual / legacy_price if legacy_price > 0 else
                             capital_recovery_factor(public_rate, schedules["legacy"].mean_term))
                            if term_years == TERM_REMAINING_LIFE else None),
            headline_private=headline_private,
            exchequer_funded=bool(exchequer_annuity) or bool(treasury_margin), model_year=model_year,
            annuity_payable=bool(carried["payable_in_model_year"]) if carried else True)
        if carried is None:
            # F19 P3 decision 5: the lifetime break-even is the headline; the snapshot is kept, renamed.
            lifetime = lifetime_break_even_price(case_profile, public_rate=public_rate, model_year=model_year,
                                                 schedule=schedules["legacy"])
            break_even.update(lifetime)
            if break_even.get("break_even_purchase_price_bill_gbp") is not None:
                break_even["break_even_purchase_price_bill_gbp"] = lifetime["break_even_purchase_price_lifetime_gbp"]
                break_even["note"] = lifetime["lifetime_note"]
        # F19 P3 decision 1: the bill change in three parts, with the purchase's path and present value.
        parts, part_path = three_part_split(
            recon_summary, lines, diagnostics, conv, allocation, case_profile, public_rate=public_rate,
            schedules=schedules, model_year=model_year,
            acquisition_year=(int(carried["acquisition_year"]) if carried else None),
            headline_delta_ex_vat_gbp=headline_delta_ex_vat if headline_private else None, valuation=valuation,
            corporation_tax_by_carrier=legacy_tax if carried is None else None)
        part_path.insert(0, "case_id", case_id)
        part_frames.append(part_path)
        # F19 P3 decision 6: I1's forecast-error cost, which I3 replaces with FTI's charge.
        i1_forecast_error = None
        if IMBALANCE_LINE in kept_replaced:
            hh_imbalance = float(private_bill[IMBALANCE_LINE])
            rule_rate = float(allocation["rule_rates"][REPLACED_LINE_RECOVERY[IMBALANCE_LINE]])
            i1_forecast_error = {"component_id": IMBALANCE_LINE, "gbp_per_3100kwh_ex_vat": hh_imbalance,
                                 "system_gbp": hh_imbalance / rule_rate,
                                 "recovery_rule": REPLACED_LINE_RECOVERY[IMBALANCE_LINE],
                                 "note": KEPT_UNDER_I1[IMBALANCE_LINE]}
        # F11 on P3 (integration, 7 Oct 2026): the kept Capacity Market line (the agreements the state does
        # not acquire) under flat and taxation recovery, beside the headline only, as the market cases have.
        cm_options = (None if is_vintaged_model_year(model_year) else
                      tariff_cm_recovery_options(summary, lines, conv))
        reports[case_id] = {
            "case_id": case_id, "institution": args.institution, "model_year": model_year, "levers": levers,
            "capacity_market_recovery_options": cm_options,
            "acquisition_settings": {k: v for k, v in acquisition.items() if k not in ("valuation", "hpc_completion_table")},
            "acquisition_terms": {"purchase_price_basis": levers.get("purchase_price_basis"),
                                  "thermal_basis": levers.get("thermal_basis"),
                                  # F28 (e): the price-weighted mean term under remaining life, and the schedules.
                                  "term_basis": "remaining_life" if term_years == TERM_REMAINING_LIFE else "fixed",
                                  "term_years": (schedules["legacy"].mean_term if term_years == TERM_REMAINING_LIFE
                                                 else term_years),
                                  "last_payment_offset_years": max(s.last_term for s in schedules.values()),
                                  "annuity_schedules": {line: s.record() for line, s in schedules.items()},
                                  "in_construction": levers.get("in_construction"),
                                  "acquisition_financing": levers.get("acquisition_financing"),
                                  "tariff_pricing": levers.get("tariff_pricing", "cost")},
            "conversion": conv, "allocation": allocation, "bill_detail": bill_detail,
            "network_account": ({"treatment": treatment, "network_ownership": treatment,
                                 "total_saving_gbp": (network_account.get("treatments", {}).get(treatment, {}) or {}).get("total_saving_gbp")
                                 if network_account else None,
                                 "adjustments_gbp_per_3100kwh_ex_vat": network_adjustments,
                                 "break_even_premium": network_break_even or None}
                                if treatment != NETWORKS_PRIVATE else {"treatment": treatment, "network_ownership": "excluded"}),
            "revenue_requirement_gbp": lines, "diagnostics": diagnostics,
            "bill": summary, "reconciliation": recon_summary,
            "multi_year": years.to_dict(orient="records"),
            # User decision 8: the model year and the averages side by side, labelled.
            f"headline_{model_year}_and_averages": dict(headline_years(years, summary, conv, headline_private,
                                                                       model_year=model_year),
                                                        comparator_case_id=comparator_id if headline_private else None,
                                                        **({"pricing_note": (
                                                            "tariff_pricing market_bill: these are the tariff's COST "
                                                            "path; the bill charged is the comparator's and the "
                                                            "Treasury keeps the gap (treasury_margin_gbp in "
                                                            f"diagnostics for {model_year})")}
                                                           if treasury_margin else {})),
            # Item AP / review: the price at which the case's saving against private reaches zero
            # (F19 P3 decision 5: over the purchase's life; the 2030 snapshot beside it).
            "break_even_purchase_price": break_even,
            # F19 P3 decision 1: new-build ownership, the legacy purchase and the rest.
            "three_parts": parts,
            # F19 valuation decision (14 Oct 2026): the price path the case's legacy streams are on,
            # and (27 Oct 2026) every valuation setting they are on.
            "legacy_streams_price_path": (str(levers.get(LEVER_LEGACY_PRICE_PATH)) if profile_by_settings is not None
                                          and levers.get(LEVER_LEGACY_PRICE_PATH) is not None
                                          else "the valuation's central path"),
            "legacy_streams_settings": legacy_streams_settings(levers, profile_by_settings),
            "i1_forecast_error_cost": i1_forecast_error,
            # User decision 11: thermal and storage at book value, the conservative high price,
            # with the market-value alternative beside it.
            "thermal_and_storage_purchase": thermal_purchase_diagnostic(
                register, valuation, case_id, allocation, conv, public_rate=public_rate, term_years=term_years,
                thermal_basis=levers.get("thermal_basis")),
        }
        if carried is not None:
            # Item Y12: the purchase carried from its year, its ledger (survival and replacement).
            reports[case_id]["acquisition_carried"] = carried
        component_frames.append(components)
        recon_frames.append(recon)
        year_frames.append(years)

    lap("cases")
    print(requirement.sample.report())
    report = dict(reports[primary])
    report.update({
        "cases": reports,
        "primary_case_id": primary,
        "variants": {c: {"delta_inc_vat_gbp": r["bill"]["delta_inc_vat_gbp"], "levers": r["levers"],
                         "acquisition_price_gbp": r["diagnostics"]["acquisition_price_gbp"],
                         "term_years": r["acquisition_terms"]["term_years"]}
                     for c, r in reports.items() if c != primary},
        "replaced_lines": replaced_lines_reported(bill_by_case), "kept_declared": KEPT_DECLARED,
        "notes": [
            "Item K1: one tariff per case whose legacy_fleet lever acquires the fleet; the purchase-price basis, the "
            "thermal basis and the annuity term are levers on the register, so item N4's variants are the named "
            "sensitivities extending fully_nationalised rather than recomputed here.",
            "Institution I1 (market machinery kept, all counterparties public) unless stated: the register was built on the "
            "solved market year, so the resource cost is the market dispatch's. I2 and I3 need the register rebuilt on a "
            "central-dispatch reference group (items N9 and N10).",
            "Cross-border trade is the measured net payment on the links at the foreign price (item N10) when the market "
            "months are given (import_basis in diagnostics); otherwise imports are valued at what GB paid them, an upper "
            "bound on the foreign price a public buyer would pay.",
            "The thermal fleet loses its price-setting role: its fixed cost is in the tariff and its Capacity Market "
            "payments are removed; its fuel is paid at cost. Agreements with capacity the state does not acquire "
            "(interconnectors, DSR, plant paid as today, capacity outside the model) stay as their own line.",
            "Like-for-like allocation (audit 26 Sep 2026): each new line is recovered at the rule of the lines it "
            "replaces, the cost of owned plant at the pool rate of the replaced market-construct lines (CfD and CCS "
            "over CfD-eligible sales, RO over obligation-eligible supply, the Capacity Market with its household peak "
            "factor, the shaping and hedging on the household's energy cost). tariff_allocation in each bill reports "
            "the flat-recovery bill beside it; the difference is cost the flat tariff moved onto non-domestic users.",
            "Networks follow the networks lever (item N6): acquired lowers TNUoS and DUoS by the wedge and the tax allowance.",
            "Item D3: every tariff case also buys the GB halves of the interconnectors (acquisition annuity at the "
            "public rate on a placeholder market-value price); their rent accrues to the tariff, the cap-and-floor "
            "pass-through is dropped and their Capacity Market agreements leave the kept line. The foreign halves "
            "and other-jurisdiction cables are paid their rent (payee: foreign owners).",
            "Item D4(a): the register's fixed O&M includes generator TNUoS and insurance; the tariff shows site "
            "fixed O&M, generator TNUoS and insurance as three lines, each at the pool rate.",
            "Never quote the saving without the purchase price beside it (acquisition_price_gbp in diagnostics).",
        ],
    })
    if args.stage_1_prices:
        # Item F12 (12 Oct 2026): a price-level twin values the cables' Stage 1 rent on its mapped GB prices.
        report["stage_1_price_files"] = [str(p) for p in args.stage_1_prices]
    summary, diagnostics, recon = reports[primary]["bill"], reports[primary]["diagnostics"], pd.concat(recon_frames, ignore_index=True)
    years = pd.concat(year_frames, ignore_index=True)
    variant_summaries = report["variants"]
    out_components = pd.concat(component_frames, ignore_index=True)
    out_parts = three_parts_path(args.out_multi_year)
    for path in (args.out_components, args.out_summary, args.out_reconciliation, args.out_multi_year, out_parts):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    out_components.to_csv(args.out_components, index=False)
    recon.to_csv(args.out_reconciliation, index=False)
    years.to_csv(args.out_multi_year, index=False)
    # F19 P3 decision 1: the three parts by year, beside the multi-year view.
    pd.concat(part_frames, ignore_index=True).to_csv(out_parts, index=False)
    Path(args.out_summary).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    lap("write")
    print(json.dumps({
        "private_ex_vat": round(summary["private_ex_vat_gbp"], 2), "nationalised_ex_vat": round(summary["ex_vat_gbp"], 2),
        "delta_inc_vat": round(summary["delta_inc_vat_gbp"], 2),
        "purchase_price_gbp_bn": round(diagnostics["acquisition_price_gbp"] / 1e9, 1),
        "cases": {k: round(v["bill"]["delta_inc_vat_gbp"], 2) for k, v in reports.items()},
    }))


if __name__ == "__main__":
    main()
