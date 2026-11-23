"""The Exchequer ledger beside the bill headline, per ownership case (item D2, 26 Sep 2026).

The consumer bill is one of two ledgers the ownership cases move.  The other is the
Exchequer's: a state that owns plant stops receiving corporation tax on it, a
state that owns the wires stops receiving the tax the network allowance pays for,
a state account that runs a deficit is funded by the Exchequer, and a lower bill
carries less VAT.  House rule 9 says a transfer must be labelled; this module puts
the transfers in one table so the bill headline can be read with them.

**The bill headline does not change.**  Each case gets, beside it:

* **carbon receipts** (UK ETS plus carbon price support): electricity dispatch
  times the build-time carbon rate per generator
  (``{build_source}_marginal_costs_breakdown.csv``), split into ETS and CPS by
  its per-tonne rates, net of the CHP heat credit (the heat's share is paid by
  heat customers), measured on the solved year the case settles on: the market
  months for every market and tariff case, the P2 bid months for
  ``state_bids_at_cost``, the central-dispatch months for the institution case.
  Carbon is identical across cases that share dispatch: owned plant pays its
  carbon to the Exchequer like private plant.  Receipts are approximated by
  emissions times price (power has no free allocation; the Exchequer's auction
  revenue need not clear at the modelled price).  Energy from waste carries no
  carbon in the breakdown (its fuel and carbon are inside the gate fee).
* **VAT on domestic bills** at the bill's rate (5%): the benchmark household's
  ex-VAT bill per MWh times domestic electricity supply
  (``sourced_externals.warm_home_discount.derivation.domestic_electricity_supply_mwh``,
  Ofgem's WHD-obligated supply volume, the only domestic volume the bill
  declares; it treats every domestic MWh as billed at the benchmark's average
  rate, standing charge included).  Non-domestic VAT is *not* counted: a
  VAT-registered business recovers it, so it is not a net receipt (the
  unregistered and charity share at 5% is left out and said so).
* **corporation tax forgone** on state-owned plant: the tax wedge inside the
  financing saving, the register's own ``corporation_tax_forgone_gbp`` per row
  (``calculate_financing_scenarios._corporation_tax_forgone``): ``capex x [CRF(r) -
  CRF(r - w)]`` over the row's remaining life on its as-new capital cost, ``r``
  the private rate and ``w = (1 - g) x r_E x tau`` the corporation tax on the
  equity return at the technology's observed gearing ``g`` (workbook WACC
  calculator, ``tau = 0.25``).  A P1 joint venture still pays tax (only the change
  to the Scenario 2 rate and wedge is forgone); a state-owned row pays none.
  Legacy plant is on the same as-new capital base, which house rule 10 warns is
  meaningless in absolute terms: the legacy figure is reported apart.  The
  split of the financing saving is the financing summary's own
  (``annual_financing_saving_tax_transfer_gbp``) when it is given.
* the **network tax allowance** (item N6): when the state owns the wires,
  consumers stop paying DUoS's corporation-tax allowance (895.5 m) and the
  Exchequer stops receiving the tax it funds.  Transmission's allowance is not
  tabulated in the RIIO-3 annex and is not counted: said so on every row.
* the **expected loss on a network debt guarantee** (review 29 Sep 2026): under
  ``networks: debt_guarantee`` the Exchequer stands behind the networks' notional
  debt (RAV x notional gearing, from the network account) and carries its
  expected default loss, PD x LGD a year at the ``guarantee_cost`` lever's rating
  (:func:`guarantee_terms`; rates and sources in
  ``data/public_ownership/financing/network_guarantee_expected_loss.csv``).  HM
  Treasury prices a contingent liability at its expected cost, risk neutral at
  the margin (Contingent Liability Approval Framework, April 2023, paras 4.4-4.5).
  The low and high settings are reported beside the case's own.
* **state deficits the Exchequer funds** and **surpluses it retains**, from the
  public account (``state_deficit_exchequer_funded_gbp`` when the public account
  provides it); the cancelled-contract residual routed to the Exchequer (item G3).
  A tariff case recovers every cost on the bill, so it has no deficit by
  construction.

The **financing saving** is split into its tax transfer (the corporation tax
forgone on the rows it is measured on) and its resource part (the rest).

Every Exchequer item is shown in GBP m (model price year, 2024) and per 3,100 kWh
household in report-year money, converted flat over final demand (house rule 8:
never divided by households).  ``exchequer_net_vs_private`` is the change in the
Exchequer's position against the private case (receipts up is positive), and
``net_of_exchequer_gbp_inc_vat`` is the bill headline less that change per
household: the bill plus what the case costs the Exchequer.

**The third party (review 29 Sep 2026).**  Beside the Exchequer each case now
carries a measured private-investor column and the resource saving
(:mod:`investor_ledger`), and the ledger refuses to write unless

    consumer change + Exchequer change + investor change = resource saving

closes to GBP 1 at system level for every case against private.  The consumer
column there is the bill's system-level change (each line at its own recovery
denominator) plus domestic VAT, not the headline times final demand;
``consumer_and_exchequer_gain_vs_private_gbp`` and ``gain_beyond_resource_saving_gbp``
(the headline read flat, against the comparator's financing saving) are kept
beside it unchanged.

**Upstream fields.**  The corporation tax per row and the state deficit per case
are the finance layer's to expose.  When either is missing the ledger refuses,
naming the field, unless ``--fallback`` is passed, when it computes them here
(the method above for tax; for deficits the public account's own signed fields:
shortfall on the Exchequer plus every negative retained or not-recycled surplus)
and records ``provenance: computed_here`` on every row.

Usage::

    python scripts/public_ownership/exchequer_ledger.py <register> <bill_summary> <public_account_summary>
        <network_account.csv> <assumptions.json> <recovery.json> <snapshot.yaml> <out.csv> <out.json>
        --marginal-costs <breakdown.csv> --market-scenarios <12 months>
        [--tariff-summary ...] [--institution-summary ... --central-scenarios <12 months>]
        [--balancing-summary ... --balancing-scenarios <12 months>] [--config config] [--fallback]

The investor column also reads, beside these by the workflow's naming
(:func:`investor_inputs`): the network account's JSON, the legacy valuation
(tariff cases), and for a state-bids case the merged lines, the market bill's
per-line table, its run group's register and bill summary, and both forecast
bills' summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.electrified_household import (COLUMNS as ELECTRIFIED_COLUMNS,
    HIGH_PREFIX as ELECTRIFIED_HIGH,
    add_columns as add_electrified_columns, archetype_basis, market_standing, tariff_standing)
from scripts.public_ownership.finance import (  # noqa: E402
    capital_recovery_factor, corporation_tax_wedges, embedded_corporation_tax,
)
from scripts.public_ownership import investor_ledger as investors  # noqa: E402
from scripts.public_ownership.register_io import CaseRows  # noqa: E402
from scripts.public_ownership.levers import (  # noqa: E402
    DEFAULT_SETTINGS, LEVER_GUARANTEE_COST, LEVER_GUARANTEE_PASS_THROUGH, LEVER_NETWORKS, LEVERS,
    NETWORKS_DEBT_GUARANTEE, comparator_for_settings,
)
from config.config_loader import read_config

UPSTREAM_CT_COLUMN = "corporation_tax_forgone_gbp"
#: F19 decision 3 (6 Oct 2026): the Exchequer part's label.
CT_PART_LABEL = ("corporation tax the state's plant does not pay (a transfer within the public sector; Great British "
                 "Energy pays corporation tax as a company today)")
#: F29: the Exchequer's side of investor_ledger.SOCIALISED_RISK (the same amount, so the identity closes).
SOCIALISED_RISK_PART = ("systematic risk taxpayers hold on the state's plant, at the workbook's social price "
                        "(economic rate less cash rate; F29)")
UPSTREAM_DEFICIT_FIELD = "state_deficit_exchequer_funded_gbp"
HEADLINE_CASES = {
    "private": "private",
    "public_control": "P1",
    "public_ownership": "P2",
    "public_ownership@state_bids_at_cost": "P2 with state bids at cost",
    "public_ownership@fully_nationalised": "state buys the existing fleet (I1, state present value)",
    "public_ownership@fully_nationalised_market_value": "P3 alt (state buys the existing fleet, I1, market value)",
    "public_ownership@fully_nationalised_market_value_central_dispatch": "P3 (state buys the existing fleet, I3, central dispatch)",
}
REGISTER_COLUMNS = ["case_id", "asset_id", "finance_vintage", "finance_technology", "ownership_share", "state_owned",
                    "annual_capital_recovery_gbp", "wacc", "remaining_life_years", "acquisition_price_gbp",
                    "legacy_scope_group", "networks", "public_return", "dispatch_mwh", "annual_variable_om_gbp",
                    # F19 decision 7: the party flows split the takeover by cohort.
                    "carrier", "support_type", "cfd_round",
                    *investors.REGISTER_COLUMNS]
COVERAGE = [
    "VAT: domestic electricity bills at 5% only. Non-domestic VAT is recoverable by VAT-registered businesses and is "
    "not a net receipt; the unregistered and charity share is not counted.",
    "VAT base: the benchmark household's ex-VAT bill per MWh times Ofgem's domestic supply volume (97.48 TWh, WHD "
    "obligated suppliers); every domestic MWh is taken at the benchmark's average rate.",
    "Carbon: UK ETS plus carbon price support on electricity, dispatch times the build-time rate, net of the CHP heat "
    "credit; emissions times price stands in for auction receipts; energy from waste carries none in the breakdown.",
    "Corporation tax the state's plant does not pay (a transfer within the public sector; Great British Energy pays "
    "corporation tax as a company today): on new build the wedge (1 - g) r_E tau on each state-owned row's as-new "
    "capital charge (workbook WACC calculator, tau 0.25), none where the comparator's cohort earns below its "
    "required return; on acquired legacy plant 25% of its 2030 operating margin after special-rate capital "
    "allowances on its tax written-down value and interest (F19 decision 3; reported apart); P1 joint ventures "
    "still taxed at the Scenario 2 wedge; the GB halves of acquired interconnectors are not in the register and "
    "their owners' tax is not counted.",
    "Network tax allowance: distribution (ED2, 895.5 m) only; transmission's is not tabulated in RIIO-3 and not counted.",
    "Network debt guarantee: expected loss (PD x LGD) on RAV x notional gearing at the guarantee_cost setting; no fee "
    "is charged to the networks (the bill keeps the whole compression), so the Exchequer carries it unpriced to them; "
    "the guarantee is taken not to change the networks' own default probability.",
    "Not covered: income tax and national insurance on employment, business rates, dividend and capital-gains tax "
    "on private owners' distributions, and the macroeconomic feedback of item J4 (the workbook ESM).",
    "Review 29 Sep 2026: Power BECCS's carbon-removal CfD (the dual CfD's CfDc) is paid by the Exchequer in every case "
    "unless beccs_removal_funding = bill_levy; tax equivalents (competitive neutrality) are receipts; domestic legacy "
    "levies moved to taxation (a background lever) are costs; a network guarantee fee and a retained network return "
    "(network_incidence) are receipts; an acquisition annuity funded from taxation (acquisition_financing) is a cost.",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _norm(name) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


# ── corporation tax ─────────────────────────────────────────────────────────────

def tax_wedges(snapshot: Mapping) -> dict[str, dict[str, float]]:
    """Per technology (normalised name): the private and coordination corporation-tax wedges.

    The finance layer's own construction (:func:`finance.corporation_tax_wedges`:
    ``(1 - g) x r_E x tau`` at observed gearing, the coordination wedge on the
    Scenario 2 equity), so a fallback computed here matches the register.
    """
    out = {}
    for name, w in corporation_tax_wedges(snapshot).items():
        out[_norm(name)] = {"private": float(w["private"]), "coordination": float(w["public_coordination"]),
                            "tau": float(w["tax_rate"])}
    return out


def _embedded(capex: pd.Series, rate: pd.Series, wedge: pd.Series, life: pd.Series) -> pd.Series:
    """``capex x [CRF(rate) - CRF(rate - wedge)]`` row by row (:func:`finance.embedded_corporation_tax`)."""
    out = pd.Series(0.0, index=capex.index)
    ok = capex.notna() & rate.notna() & wedge.notna() & life.gt(0)
    for idx in capex.index[ok]:
        out[idx] = embedded_corporation_tax(float(capex[idx]), float(rate[idx]), float(wedge[idx]), float(life[idx]))
    return out


def _private_capex(private: pd.DataFrame) -> pd.Series:
    """The as-new capital cost the register annuitises: private capital charge over CRF(private rate, life)."""
    capital = pd.to_numeric(private["annual_capital_recovery_gbp"], errors="coerce")
    rate = pd.to_numeric(private["wacc"], errors="coerce")
    life = _num(private, "remaining_life_years")
    out = pd.Series(np.nan, index=private.index)
    ok = capital.notna() & rate.notna() & life.gt(0)
    for idx in private.index[ok]:
        out[idx] = float(capital[idx]) / capital_recovery_factor(float(rate[idx]), float(life[idx]))
    return out


def corporation_tax_forgone(register: pd.DataFrame, case_id: str, wedges: Mapping[str, Mapping[str, float]],
                            private_tax: pd.Series | None = None) -> dict:
    """Corporation tax the state does not pay on the plant a case owns, GBP (the fallback, computed here).

    Mirrors ``calculate_financing_scenarios._corporation_tax_forgone`` row by row:
    the tax embedded at the private rate on the as-new capital cost
    (``capex x [CRF(r) - CRF(r - w)]``) is forgone on a row the state owns and
    finances itself, legacy plant included; on the CEPA decomposition (P1's
    stake) the joint venture still pays tax, so only the change to the Scenario 2
    rate and wedge is forgone; a commercial-benchmark company pays tax at the
    private wedge scaled by the rate ratio.  ``private_tax`` (the private
    embedded tax per asset) may be passed in so it is computed once.
    """
    private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
    case = register.loc[register["case_id"].eq(case_id)].set_index("asset_id").reindex(private.index)
    tech = private["finance_technology"].map(_norm)
    w_private = tech.map({k: v["private"] for k, v in wedges.items()})
    w_coord = tech.map({k: v["coordination"] for k, v in wedges.items()})
    r_private = pd.to_numeric(private["wacc"], errors="coerce")
    life = _num(private, "remaining_life_years")
    capex = _private_capex(private)
    if private_tax is None:
        private_tax = _embedded(capex, r_private, w_private, life)
    owned = _num(case, "ownership_share").gt(0)
    legacy = private["finance_vintage"].astype(str).eq("legacy")
    case_rate = pd.to_numeric(case["wacc"], errors="coerce")
    kind = (case["public_return"].astype(str) if "public_return" in case.columns
            else pd.Series("public_rate", index=case.index))
    lost = pd.Series(0.0, index=private.index)
    lost[owned] = private_tax[owned]
    coordination = owned & ~legacy & kind.eq("cepa_decomposition")
    if coordination.any():
        lost[coordination] = (private_tax - _embedded(capex, case_rate, w_coord, life))[coordination]
    company = owned & ~legacy & kind.eq("commercial_benchmark")
    if company.any():
        scaled = (w_private * case_rate / r_private).where(r_private > 0, 0.0)
        lost[company] = (private_tax - _embedded(capex, case_rate, scaled, life))[company]
    unmapped = sorted(set(private.loc[owned & w_private.isna(), "finance_technology"].dropna().astype(str)))
    return {
        "corporation_tax_forgone_gbp": float(lost.sum()),
        "corporation_tax_forgone_new_build_gbp": float(lost[~legacy].sum()),
        "corporation_tax_forgone_legacy_gbp": float(lost[legacy].sum()),
        "corporation_tax_rows_still_taxed": int((coordination | company).sum()),
        "corporation_tax_unmapped_technologies": unmapped,
    }


def upstream_corporation_tax(register: pd.DataFrame, case_id: str) -> dict:
    """The finance layer's own per-row corporation tax forgone, summed by vintage."""
    rows = register.loc[register["case_id"].eq(case_id)]
    tax = _num(rows, UPSTREAM_CT_COLUMN)
    legacy = rows["finance_vintage"].astype(str).eq("legacy")
    return {"corporation_tax_forgone_gbp": float(tax.sum()),
            "corporation_tax_forgone_new_build_gbp": float(tax[~legacy].sum()),
            "corporation_tax_forgone_legacy_gbp": float(tax[legacy].sum()),
            "corporation_tax_rows_still_taxed": None, "corporation_tax_unmapped_technologies": []}


# ── the public account ─────────────────────────────────────────────────────────

def state_account(account_case: Mapping | None, *, fallback: bool, case_id: str) -> dict:
    """Deficits the Exchequer funds and surpluses it retains for one market case, GBP (both >= 0)."""
    if account_case is None:
        raise ValueError(f"the public account summary has no case {case_id}")
    signed = {k: float(account_case.get(k) or 0.0) for k in (
        "retained_in_public_account_gbp", "storage_surplus_not_recycled_gbp",
        "capacity_market_route_surplus_not_recycled_gbp", "dispatchable_cfd_route_surplus_not_recycled_gbp")}
    surplus = sum(max(v, 0.0) for v in signed.values())
    if UPSTREAM_DEFICIT_FIELD in account_case:
        return {"state_deficits_funded_gbp": float(account_case[UPSTREAM_DEFICIT_FIELD] or 0.0),
                "state_surplus_retained_gbp": float(account_case.get("state_surplus_retained_gbp", surplus) or 0.0),
                "state_account_provenance": "finance layer (public account exchequer_position)"}
    if not fallback:
        raise ValueError(
            f"the public account summary carries no {UPSTREAM_DEFICIT_FIELD} for {case_id}: the finance layer must "
            "expose the state deficit the Exchequer funds (after the B2/B3 storage corrections); pass --fallback to "
            "compute it here from the account's signed fields instead")
    deficit = float(account_case.get("shortfall_on_exchequer_gbp") or 0.0) + sum(max(-v, 0.0) for v in signed.values())
    return {"state_deficits_funded_gbp": deficit, "state_surplus_retained_gbp": surplus,
            "state_account_provenance": ("computed_here: shortfall on the Exchequer plus every negative retained or "
                                         "not-recycled surplus of the public account (before any B2/B3 correction)")}


# ── carbon and VAT ──────────────────────────────────────────────────────────────

def carbon_receipts(measurement: Mapping) -> dict:
    """Carbon receipts from a dispatch measurement (:func:`nationalised_institutions.measure_dispatch`)."""
    for key in ("carbon_gbp", "carbon_uk_ets_gbp", "carbon_price_support_gbp"):
        if key not in measurement:
            raise ValueError(f"dispatch measurement carries no {key}; it predates review B13")
    return {"carbon_receipts_gbp": float(measurement["carbon_gbp"]),
            "carbon_uk_ets_gbp": float(measurement["carbon_uk_ets_gbp"]),
            "carbon_price_support_gbp": float(measurement["carbon_price_support_gbp"]),
            "chp_heat_credit_carbon_gbp": float(measurement.get("chp_heat_credit_carbon_gbp", 0.0))}


def domestic_vat(ex_vat_gbp_per_household: float, conv: Mapping) -> float:
    """VAT on domestic bills, system GBP in model-year money."""
    per_mwh_report = float(ex_vat_gbp_per_household) / float(conv["mwh"])
    return float(conv["vat_rate"]) * per_mwh_report * float(conv["domestic_mwh"]) / float(conv["factor"])


def conversion(assumptions: Mapping, recovery: Mapping, bill_summary: Mapping | None = None) -> dict:
    """The ledger's household conversion: the Exchequer's per-household key (metered supply).

    F19 (7 Oct 2026): a bill on F18's two-volume basis states the key
    (``exchequer_per_household_key``); it is read there, never through the recovery file's
    deprecated ``final_demand_mwh`` alias. A D1 (2050) bill keeps the recovery file's volume.
    ``final_demand_mwh`` here names that key (metered supply) for the ledger's readers.
    """
    whd = ((assumptions.get("sourced_externals") or {}).get("warm_home_discount") or {}).get("derivation") or {}
    if "domestic_electricity_supply_mwh" not in whd:
        raise ValueError("the assumptions declare no domestic electricity supply volume "
                         "(sourced_externals.warm_home_discount.derivation.domestic_electricity_supply_mwh)")
    from scripts.public_ownership import bill_recovery

    key = (bill_recovery.exchequer_key_mwh(bill_summary) if bill_summary is not None
           and bill_recovery.is_two_volume(bill_summary)
           else bill_recovery.d1_recovery_file_volumes(recovery)["final_demand_mwh"])
    return {
        "factor": float(assumptions["currency_rebasing"]["factor"]),
        "final_demand_mwh": key,
        "exchequer_key_mwh": key,
        "mwh": float(assumptions["benchmark"]["annual_consumption_kwh"]) / 1000.0,
        "vat_rate": float(assumptions["benchmark"]["vat_rate"]),
        "domestic_mwh": float(whd["domestic_electricity_supply_mwh"]),
    }


def per_household(system_gbp: float, conv: Mapping) -> float:
    """System GBP (model year) to GBP per 3,100 kWh household (report year), flat over final demand."""
    return float(system_gbp) * float(conv["factor"]) / float(conv["final_demand_mwh"]) * float(conv["mwh"])


def household_to_system(gbp_per_household: float, conv: Mapping) -> float:
    """The inverse of :func:`per_household`: system GBP in the model year."""
    return float(gbp_per_household) / float(conv["mwh"]) * float(conv["final_demand_mwh"]) / float(conv["factor"])


# ── the ledger ──────────────────────────────────────────────────────────────────

def ledger_row(case: Mapping, private: Mapping, conv: Mapping) -> dict:
    """One case's ledger: its items, the change against private, and the net-of-Exchequer figure.

    ``case`` and ``private`` carry ``bill_ex_vat``, ``bill_inc_vat`` (per household),
    ``carbon_receipts_gbp``, ``corporation_tax_forgone_gbp``, ``network_tax_forgone_gbp``,
    ``state_deficits_funded_gbp``, ``state_surplus_retained_gbp``,
    ``exchequer_residual_gbp``, ``financing_saving_gbp`` and
    ``financing_saving_tax_transfer_gbp`` (system GBP, model year).
    """
    row = dict(case)
    row["vat_receipts_gbp"] = domestic_vat(case["bill_ex_vat"], conv)
    vat_private = domestic_vat(private["bill_ex_vat"], conv)
    row["vat_change_vs_private_gbp"] = row["vat_receipts_gbp"] - vat_private
    row["carbon_change_vs_private_gbp"] = case["carbon_receipts_gbp"] - private["carbon_receipts_gbp"]
    row["state_deficits_change_vs_private_gbp"] = case["state_deficits_funded_gbp"] - private["state_deficits_funded_gbp"]
    row["state_surplus_change_vs_private_gbp"] = case["state_surplus_retained_gbp"] - private["state_surplus_retained_gbp"]
    parts = {
        "carbon receipts": row["carbon_change_vs_private_gbp"],
        "VAT on domestic bills": row["vat_change_vs_private_gbp"],
        # F19 decision 3: the label (the size is the register's, capped at what the comparator would pay).
        CT_PART_LABEL: -float(case["corporation_tax_forgone_gbp"]),
        "network corporation-tax allowance forgone (distribution)": -float(case["network_tax_forgone_gbp"]),
        "expected loss on the network debt guarantee (J6)": -float(case.get("guarantee_expected_loss_gbp", 0.0)),
        # F29 (user, 26 Oct 2026): workbook v2.0's Four Accounts, "TAXPAYERS: socialised systematic risk".
        SOCIALISED_RISK_PART: -(_field(case, "socialised_risk_gbp") - _field(private, "socialised_risk_gbp")),
        # Decision 68: the co-investors' rate reduction the state's implicit guarantee backs (zero without it).
        "implicit guarantee on co-investment (decision 68)": -(
            _field(case, "co_investment_implicit_guarantee_gbp") - _field(private, "co_investment_implicit_guarantee_gbp")),
        "state deficits funded": -row["state_deficits_change_vs_private_gbp"],
        "state surplus retained": row["state_surplus_change_vs_private_gbp"],
        "cancelled-contract residual paid by the Exchequer (item G3)": -float(case["exchequer_residual_gbp"]),
        # Review 29 Sep 2026 (user decisions 2 to 7): each against private, which carries
        # the carbon CfD too (the Exchequer pays the private BECCS owners' CfDc).
        "tax equivalents received from state-owned plant": (
            _field(case, "tax_equivalent_received_gbp") - _field(private, "tax_equivalent_received_gbp")),
        "Power BECCS carbon-removal CfD paid by the Exchequer (dual CfD)": -(
            _field(case, "carbon_removal_cfd_exchequer_gbp") - _field(private, "carbon_removal_cfd_exchequer_gbp")),
        "domestic legacy levies moved to general taxation": -(
            _field(case, "legacy_levies_exchequer_gbp") - _field(private, "legacy_levies_exchequer_gbp")),
        "network guarantee fee received": _field(case, "network_fee_gbp") - _field(private, "network_fee_gbp"),
        "network return retained by the Treasury": (
            _field(case, "network_retained_gbp") - _field(private, "network_retained_gbp")),
        "acquisition annuity funded from general taxation": -(
            _field(case, "acquisition_annuity_exchequer_gbp") - _field(private, "acquisition_annuity_exchequer_gbp")),
        # User answer Q5 (30 Sep 2026): tariff_pricing market_bill, the margin over cost.
        "margin over the tariff's cost kept by the Treasury (market prices charged)": (
            _field(case, "treasury_margin_gbp") - _field(private, "treasury_margin_gbp")),
        # F9: the state's own Capacity Market winners paid the re-cleared 2030/31 price.
        "Capacity Market receipts of the state's plant at the re-cleared price (F9)": (
            _field(case, "capacity_market_state_bids_state_gbp") - _field(private, "capacity_market_state_bids_state_gbp")),
    }
    net = float(sum(parts.values()))
    row["exchequer_net_vs_private_parts_gbp"] = parts
    row["exchequer_net_vs_private_gbp"] = net
    row["exchequer_net_vs_private_gbp_per_household"] = per_household(net, conv)
    row["net_of_exchequer_gbp_inc_vat"] = float(case["bill_inc_vat"]) - row["exchequer_net_vs_private_gbp_per_household"]
    row["bill_delta_vs_private_inc_vat"] = float(case["bill_inc_vat"]) - float(private["bill_inc_vat"])
    row["net_of_exchequer_delta_vs_private_inc_vat"] = (row["net_of_exchequer_gbp_inc_vat"]
                                                        - float(private["bill_inc_vat"]))
    saving, tax = float(case["financing_saving_gbp"]), float(case["financing_saving_tax_transfer_gbp"])
    row["financing_saving_resource_gbp"] = saving - tax
    # Review 28 Sep 2026: who gains, beside the country's resource saving. Consumers'
    # gain is the inc-VAT bill change on final demand (VAT then sits in the Exchequer
    # parts); consumers + Exchequer is the domestic public side. Its gap to the
    # resource financing saving is what private owners (and foreign counterparties)
    # give up, plus any resource effect outside the financing saving (the state's
    # overhead, dispatch changes under another institution). It is the headline read
    # flat against the comparator's financing saving, so it is reported, never
    # checked; the measured split that closes is the three-party identity
    # (investor_ledger, review 29 Sep 2026), added by build().
    consumer_gain = -household_to_system(row["bill_delta_vs_private_inc_vat"], conv)
    row["consumer_gain_vs_private_gbp"] = consumer_gain
    row["consumer_and_exchequer_gain_vs_private_gbp"] = consumer_gain + net
    row["gain_beyond_resource_saving_gbp"] = consumer_gain + net - row["financing_saving_resource_gbp"]
    row["gain_beyond_resource_saving_label"] = (
        "consumers + Exchequer less the resource financing saving: transfer from private owners and foreign "
        "counterparties, plus resource effects outside the financing saving; measured and checked as the "
        "three-party identity (investor_change_vs_private_gbp, resource_saving_vs_private_gbp)")
    row.setdefault("guarantee_expected_loss_gbp", 0.0)
    row.setdefault("co_investment_implicit_guarantee_gbp", 0.0)
    for key in ("carbon_receipts_gbp", "vat_receipts_gbp", "corporation_tax_forgone_gbp", "network_tax_forgone_gbp",
                "guarantee_expected_loss_gbp",
                "state_deficits_funded_gbp", "state_surplus_retained_gbp", "exchequer_residual_gbp",
                "financing_saving_gbp", "financing_saving_tax_transfer_gbp", "financing_saving_resource_gbp",
                "consumer_and_exchequer_gain_vs_private_gbp", "gain_beyond_resource_saving_gbp"):
        row[key.replace("_gbp", "_gbp_per_household")] = per_household(row[key], conv)
    return row


def _field(entry: Mapping, key: str) -> float:
    return float(entry.get(key) or 0.0)


def guarantee_terms(table: pd.DataFrame, network_account: Mapping) -> dict:
    """The guaranteed network debt and the expected-loss rate at each ``guarantee_cost`` setting.

    ``table`` is ``network_guarantee_expected_loss.csv`` (one row per lever value:
    annual default probability, loss given default, their product in basis points
    and the sources); ``network_account`` is the network account's JSON, whose
    ``debt_guarantee`` treatment gives each sector's RAV and notional gearing.  The
    exposure is the notional debt, ``RAV x gearing``, and it must reproduce the
    account's own saving at its debt compression, or this refuses.

    Retired with ``networks: debt_guarantee`` (F29, 26 Oct 2026): no case can hold the
    guarantee, so the command line no longer prices it.
    """
    from scripts.public_ownership.levers import RETIRED_LEVERS

    if LEVER_GUARANTEE_COST not in LEVERS:
        raise ValueError(f"guarantee_cost {RETIRED_LEVERS.get(LEVER_GUARANTEE_COST, 'is retired')}")
    settings = LEVERS[LEVER_GUARANTEE_COST].values
    rows = table.set_index("setting")
    if sorted(rows.index) != sorted(settings):
        raise ValueError(f"the guarantee expected-loss table must carry one row per setting {list(settings)}, "
                         f"found {sorted(rows.index)}")
    rates = {}
    for setting in settings:
        pd_annual = float(rows.at[setting, "annual_default_probability"])
        lgd = float(rows.at[setting, "loss_given_default"])
        bps = float(rows.at[setting, "expected_loss_bps"])
        if abs(pd_annual * lgd * 1e4 - bps) > 1e-3:
            raise ValueError(f"guarantee_cost {setting}: {bps} bp is not PD {pd_annual} x LGD {lgd}")
        rates[setting] = bps
    treatment = ((network_account.get("treatments") or {}).get("debt_guarantee") or {})
    if not treatment:
        raise ValueError("the network account carries no debt_guarantee treatment to take the guaranteed debt from")

    def _debt(block_by_sector):
        exposure, by_sector = 0.0, {}
        for sector in ("transmission", "distribution"):
            block = block_by_sector[sector]
            # User decision 6: the guaranteed debt is the share issued or refinanced 2026-2030.
            debt = float(block.get("guaranteed_debt_gbp") or float(block["rav_gbp"]) * float(block["notional_gearing"]))
            if abs(debt * float(block["debt_compression"]) - float(block["financing_saving_gbp"])) > 1.0:
                raise ValueError(f"{sector}: RAV x gearing x compression does not reproduce the guarantee's saving")
            by_sector[sector] = debt
            exposure += debt
        return exposure, by_sector

    exposure, by_sector = _debt(treatment)
    by_setting = {}
    for variant in (network_account.get("variants") or {}).values():
        if variant.get("treatment") == "debt_guarantee":
            setting = (variant.get("variant") or {}).get("guarantee_pass_through")
            if setting:
                by_setting[setting] = _debt(variant)[0]
    return {"guaranteed_debt_gbp": exposure, "guaranteed_debt_by_sector_gbp": by_sector,
            "guaranteed_debt_by_pass_through_gbp": by_setting,
            "expected_loss_bps": rates, "sources": sorted(set(table["sources"].astype(str)))}


def guarantee_expected_loss(levers: Mapping, guarantee: Mapping | None, case_id: str) -> dict:
    """One case's expected-loss charge at its own ``guarantee_cost`` setting, and at the low and high ones."""
    if (levers or {}).get(LEVER_NETWORKS) != NETWORKS_DEBT_GUARANTEE:
        return {"guarantee_expected_loss_gbp": 0.0}
    if guarantee is None:
        raise ValueError(f"{case_id} guarantees the networks' debt: the ledger needs the guarantee terms "
                         "(--guarantee-cost and the network account's JSON)")
    setting = str((levers or {}).get(LEVER_GUARANTEE_COST, DEFAULT_SETTINGS[LEVER_GUARANTEE_COST]))
    pass_through = str((levers or {}).get(LEVER_GUARANTEE_PASS_THROUGH, DEFAULT_SETTINGS[LEVER_GUARANTEE_PASS_THROUGH]))
    debt = float((guarantee.get("guaranteed_debt_by_pass_through_gbp") or {}).get(pass_through,
                                                                                  guarantee["guaranteed_debt_gbp"]))
    out = {"guarantee_cost": setting, "guaranteed_network_debt_gbp": debt,
           "guarantee_expected_loss_bps": float(guarantee["expected_loss_bps"][setting]),
           "guarantee_expected_loss_gbp": debt * float(guarantee["expected_loss_bps"][setting]) / 1e4}
    for other, bps in guarantee["expected_loss_bps"].items():
        out[f"guarantee_expected_loss_at_{other}_gbp"] = debt * float(bps) / 1e4
    return out


def network_tax_allowance(network_lines: pd.DataFrame, levers: Mapping | None = None) -> float:
    """The distribution tax allowance an acquiring case forgoes, read at the case's own variant row.

    Since 29 Sep 2026 the account carries one acquired row per sector and variant
    (premium, public rate): summing every acquired row counted the allowance once
    per variant.
    """
    from scripts.public_ownership.network_finance_account import select_rows

    rows = select_rows(network_lines, "acquired", levers)
    rows = rows.loc[rows["sector"].eq("distribution")]
    if rows.empty:
        raise ValueError("the network finance account has no acquired distribution row")
    return float(rows["tax_allowance_saving_gbp"].sum())


def load_register(path: str | Path, cases: Sequence[str]):
    """The ledger's columns for its cases and private, from the Parquet register (or an old CSV).

    30 Sep 2026: a :class:`register_io.CaseRows`, read a case at a time as the
    ledger asks for it, so memory is flat in the number of cases.
    """
    cols = list(dict.fromkeys(REGISTER_COLUMNS + [UPSTREAM_CT_COLUMN]))
    return CaseRows(path, columns=cols, cases=set(cases) | {"private"}, share_key=register_share_key(path))


def register_share_key(path):
    """F20 phase 2: ``case_id -> fingerprints.body_key``, for the ledger's register requests; None to share nothing.

    Every request :func:`_register_requests` lists reads the case's register body alone (its
    rows, private's, and the ``public_return`` register lever), never its other levers, and
    returns no label, so cases with the same body share one result (the lever-read audit is
    ``tests/unit/test_f20_phase2_fingerprints.py``).
    """
    from scripts.public_ownership import fingerprints
    from scripts.public_ownership.cfd_repricing import register_settings_by_case

    if not (fingerprints.sharing_enabled() and fingerprints.register_is_keyed(path)):
        return None
    settings = register_settings_by_case(path)
    vintaged = fingerprints.register_vintaged(path)
    keys = {case_id: fingerprints.body_key(case_id, point, vintaged=vintaged) for case_id, point in settings.items()}
    return lambda case_id: keys.get(case_id, ("case", case_id))


def build(*, register: pd.DataFrame, bill: Mapping, account: Mapping, network_lines: pd.DataFrame, conv: Mapping,
          wedges: Mapping, carbon: Mapping[str, Mapping], tariff: Mapping | None = None,
          institution: Mapping | None = None, balancing: Mapping | None = None, fallback: bool = False,
          financing: pd.DataFrame | None = None,
          balancing_accounts: Mapping[str, Mapping] | None = None,
          investor: Mapping | None = None, guarantee: Mapping | None = None,
          catalogue: Mapping[str, Mapping] | None = None, incidence: Mapping | None = None,
          contested: Mapping | None = None, forecast_summary: Mapping | None = None,
          hpc_schedule: Mapping | None = None,
          electrified_basis: Mapping | None = None,
          electrified_high_basis: Mapping | None = None) -> tuple[pd.DataFrame, dict]:
    """The ledger for every case, and its summary.  ``carbon`` maps a dispatch key to its receipts:
    ``market`` always, ``balancing`` for the state-bids case and ``central`` for the institution case.

    ``investor`` (:func:`investor_inputs`) adds the private-investor column and the
    resource saving to every case and refuses unless the three-party identity
    closes (:func:`three_party_columns`); the command line always passes it.
    ``guarantee`` (:func:`guarantee_terms`) prices the network debt guarantee's
    expected loss for every case on ``networks: debt_guarantee``; such a case
    without it is refused.

    ``balancing_accounts`` maps a state-bids case to the public account of the run
    group it settles on (:func:`balancing_group_accounts`); without it the market
    group's account is read, which settles that case on the wrong months (review
    28 Sep 2026).

    ``financing`` (the financing summary) supplies each market case's financing
    saving and its tax-transfer split (``annual_financing_saving_tax_transfer_gbp``)
    when it carries them; otherwise the saving is the bill summary's and its tax
    transfer the corporation tax forgone on new build.  A tariff case's financing
    saving is its new-build capital saving only (the legacy purchase is a price,
    not a financing saving, and its register comparison is meaningless, house
    rule 10), with the new-build tax as its transfer.
    """
    split = None
    if financing is not None and "annual_financing_saving_tax_transfer_gbp" in financing.columns:
        split = financing.set_index("case_id")[["annual_financing_saving_vs_private_gbp",
                                                "annual_financing_saving_tax_transfer_gbp"]]

    def financing_split(case_id: str, entry: dict) -> None:
        if split is not None and case_id in split.index:
            entry["financing_saving_gbp"] = float(split.at[case_id, "annual_financing_saving_vs_private_gbp"])
            entry["financing_saving_tax_transfer_gbp"] = float(split.at[case_id, "annual_financing_saving_tax_transfer_gbp"])
            entry["financing_split_provenance"] = "financing summary"
        else:
            entry["financing_saving_tax_transfer_gbp"] = entry["corporation_tax_forgone_new_build_gbp"]
            entry["financing_split_provenance"] = "bill summary saving; tax transfer = corporation tax on new build"
    # 30 Sep 2026: the register is read a case at a time (a frame, as in the tests, is wrapped).
    register = CaseRows.of(register)
    has_upstream_ct = UPSTREAM_CT_COLUMN in register.columns
    if not has_upstream_ct and not fallback:
        raise ValueError(f"the finance register carries no {UPSTREAM_CT_COLUMN} column: the finance layer must "
                         "expose the corporation tax forgone per row; pass --fallback to compute it here from the "
                         "register's capital charges and the workbook WACC calculator (tau 0.25)")
    private_rows = register.rows("private").set_index("asset_id")
    private_tax = None
    if not has_upstream_ct:
        w_private = private_rows["finance_technology"].map(_norm).map({k: v["private"] for k, v in wedges.items()})
        private_tax = _embedded(_private_capex(private_rows), pd.to_numeric(private_rows["wacc"], errors="coerce"),
                                w_private, _num(private_rows, "remaining_life_years"))
    register_cases = register.cases
    # F20 phase 1: every register-derived quantity the passes below read, computed once per case in
    # register order (each run of row groups read once) and memoised on the register.
    register.clear_memo()
    register.prefetch(_register_requests(register, bill=bill, balancing=balancing, tariff=tariff,
                                         institution=institution, investor=investor, catalogue=catalogue,
                                         contested=contested, hpc=bool(hpc_schedule and forecast_summary
                                                                       and not hpc_schedule.get("skipped")),
                                         has_upstream_ct=has_upstream_ct, wedges=wedges, private_tax=private_tax))

    def tax_for(case_id: str, source: CaseRows | None = None) -> dict:
        if source is not None:
            # A state-bids case on its own run group's register (14 Oct 2026, below).
            out = upstream_corporation_tax(source.rows(case_id), case_id)
            out["corporation_tax_provenance"] = ("finance layer, the case's own run group's register "
                                                 "(column corporation_tax_forgone_gbp)")
            return out
        if case_id not in register_cases:
            raise ValueError(f"the finance register has no rows for {case_id}")
        out = register.memo(*_tax_request(case_id, has_upstream_ct, wedges, private_tax))
        out["corporation_tax_provenance"] = ("finance layer (register column corporation_tax_forgone_gbp)"
                                             if has_upstream_ct else "computed_here (workbook wedge, see module)")
        return out

    cases: dict[str, dict] = {}
    levers_by_case = bill.get("levers_by_case") or {}

    def register_sum(case_id: str, column: str, source: CaseRows | None = None) -> float:
        if source is not None:
            # The case's own run group's register (14 Oct 2026): read directly, not memoised on the market's.
            return float(_num(source.rows(case_id), column).sum()) if column in source.columns else 0.0
        # F20 phase 1: memoised; a column the register lacks is 0.0 without reading the rows (it was 0.0 after).
        return register.memo(*_sum_request(case_id, column)) if column in register.columns else 0.0

    def own_group_register(case_id: str) -> CaseRows | None:
        """A state-bids case's own run group's register, when the investor inputs carry it with the case's rows.

        14 Oct 2026: since F19 decision 3 the corporation tax forgone depends on settlement (new build
        forgoes it only where its comparator cohort earns its required return), so the case's rows on the
        market group's register (settled on the market months) are not the rows its bill, its public
        account and the investor ledger's ownership step read (settled on its own months). Like the state
        account (review 28 Sep 2026), every register item of the case is read on its own group's register.
        """
        frame = ((((investor or {}).get("balancing") or {}).get(case_id) or {}).get("register"))
        if frame is None:
            return None
        own = CaseRows.of(frame)
        if case_id not in own.cases or UPSTREAM_CT_COLUMN not in own.columns:
            return None
        return own

    def market_items(case_id: str, b: Mapping, levers: Mapping, source: CaseRows | None = None) -> dict:
        """Review 29 Sep 2026: the Exchequer items a market-bill case's own outputs carry."""
        incidence = b.get("network_incidence_gbp_2024") or {}
        genuine = float((investor or {}).get("co_investment_genuine_share", 0.0) or 0.0)
        return {
            # Decision 68: the co-investors' rate reduction is a transfer the Exchequer's implicit guarantee
            # backs, less the share counted as genuine (a resource saving); zero without the register column.
            "co_investment_implicit_guarantee_gbp": (register_sum(case_id, "co_investment_rate_gain_gbp", source)
                                                     * (1.0 - genuine)),
            "tax_equivalent_received_gbp": register_sum(case_id, "annual_tax_equivalent_gbp", source),
            # F29: the systematic risk the state holds on the plant it finances (register column).
            "socialised_risk_gbp": register_sum(case_id, "socialised_risk_gbp", source),
            "carbon_removal_cfd_exchequer_gbp": (register_sum(case_id, "carbon_removal_cfd_revenue_gbp", source)
                                                 if levers.get("beccs_removal_funding", "exchequer") == "exchequer" else 0.0),
            "legacy_levies_exchequer_gbp": float(b.get("legacy_levies_to_taxation_total_gbp_2024") or 0.0),
            "network_fee_gbp": float(sum(v.get("fee_to_exchequer_gbp", 0.0) for v in incidence.values())),
            "network_retained_gbp": float(sum(v.get("retained_by_exchequer_gbp", 0.0) for v in incidence.values())),
            "acquisition_annuity_exchequer_gbp": 0.0, "levers_for_flows": dict(levers),
            # F9: the state's own 2030/31 T-4 winners paid the re-cleared price (negative: less received).
            "capacity_market_state_bids_state_gbp": float(b.get("capacity_market_state_bids_state_part_gbp_2024") or 0.0)}

    for case_id, b in (bill.get("cases") or {}).items():
        levers = b.get("levers") or levers_by_case.get(case_id) or {}
        entry = {"case_id": case_id, "stage": "market_bill", "dispatch": "market",
                 "bill_ex_vat": float(b["ex_vat_gbp"]), "bill_inc_vat": float(b["total_gbp_inc_vat"]),
                 **carbon["market"], **tax_for(case_id),
                 **state_account((account.get("cases") or {}).get(case_id), fallback=fallback, case_id=case_id),
                 "network_tax_forgone_gbp": (network_tax_allowance(network_lines, levers)
                                             if levers.get("networks") == "acquired" else 0.0),
                 "exchequer_residual_gbp": 0.0, "financing_saving_gbp": float(b.get("financing_saving_gbp_2024") or 0.0)}
        financing_split(case_id, entry)
        entry["networks"] = levers.get("networks")
        entry.update(market_items(case_id, b, levers))
        # F11: the Capacity Market line under flat and taxation recovery (reported beside the headline only).
        entry["cm_recovery_options"] = b.get("capacity_market_recovery_options")
        # User, 13 Nov 2026: the electrified household's standing charge, from the case's own archetype bills.
        entry["electrified_standing_charge_gbp_ex_vat"] = market_standing(b)
        entry["electrified_standing_basis"] = "the case's own archetype bills"
        cases[case_id] = entry
    if "private" not in cases:
        raise ValueError("the bill summary carries no private case")
    for case_id, m in ((balancing or {}).get("cases") or {}).items():
        if "balancing" not in carbon and not carbon.get("balancing_by_group"):
            raise ValueError(f"{case_id} settles on its own balancing months; pass --balancing-scenarios")
        levers = m.get("levers") or {}
        inc = float(m["total_gbp_inc_vat"])
        base = cases.get(m.get("base_case"))
        own = own_group_register(case_id)
        # The case's rows on its own run group's register; without them (an older caller), the market
        # register's rows, or the base case's: the same assets and shares, settled on the market months.
        register_case = case_id if own is not None or case_id in register_cases else m["base_case"]
        entry = {"case_id": case_id, "stage": "balancing_bids", "dispatch": "state bids months",
                 "bill_ex_vat": inc / (1.0 + conv["vat_rate"]), "bill_inc_vat": inc,
                 **balancing_carbon(carbon, m.get("run_group")),
                 **tax_for(register_case, own),
                 **state_account((((balancing_accounts or {}).get(case_id) or account).get("cases") or {}).get(case_id),
                                 fallback=fallback, case_id=case_id),
                 "state_account_run_group": m.get("run_group") if case_id in (balancing_accounts or {}) else "market",
                 "network_tax_forgone_gbp": (network_tax_allowance(network_lines, levers)
                                             if levers.get("networks") == "acquired" else 0.0),
                 "exchequer_residual_gbp": 0.0,
                 "financing_saving_gbp": float(base["financing_saving_gbp"]) if base else 0.0,
                 "networks": levers.get("networks")}
        financing_split(case_id, entry)
        group_bill = ((((investor or {}).get("balancing") or {}).get(case_id) or {}).get("bill") or {}).get("cases") or {}
        entry["register_run_group"] = m.get("run_group") if own is not None else "market"
        entry.update(market_items(register_case, group_bill.get(case_id) or {}, levers, own))
        # The bid stack moves per-kWh (balancing) lines: the base case's standing charge.
        entry["electrified_standing_charge_gbp_ex_vat"] = (base or {}).get("electrified_standing_charge_gbp_ex_vat")
        entry["electrified_standing_basis"] = "the base case's (the bid stack moves per-kWh lines)"
        cases[case_id] = entry
    tariff_cases = (tariff or {}).get("cases") or {}
    for case_id, r in tariff_cases.items():
        levers = r.get("levers") or {}
        attribution = (r.get("reconciliation") or {}).get("attribution_gbp") or {}
        new_build = next((v for k, v in attribution.items() if k.startswith("new-build capital: private WACC")), 0.0)
        entry = {"case_id": case_id, "stage": "acquisition_tariff", "dispatch": "market",
                 "bill_ex_vat": float(r["bill"]["ex_vat_gbp"]), "bill_inc_vat": float(r["bill"]["total_gbp_inc_vat"]),
                 **carbon["market"], **tax_for(case_id),
                 # The tariff recovers every cost of owned plant on the bill: no state deficit by construction.
                 "state_deficits_funded_gbp": 0.0, "state_surplus_retained_gbp": 0.0,
                 "state_account_provenance": "tariff at cost: every cost of owned plant is on the bill",
                 "network_tax_forgone_gbp": (network_tax_allowance(network_lines, levers)
                                             if levers.get("networks") == "acquired" else 0.0),
                 "exchequer_residual_gbp": float((r.get("diagnostics") or {}).get(
                     "cancelled_contract_residual_exchequer_transfer_gbp") or 0.0),
                 # The new-build capital saving only: the legacy acquisition is a purchase, not a financing saving.
                 "financing_saving_gbp": -float(new_build), "networks": levers.get("networks"),
                 "financing_split_provenance": "tariff new-build capital saving; tax transfer = corporation tax on new build"}
        entry["financing_saving_tax_transfer_gbp"] = entry["corporation_tax_forgone_new_build_gbp"]
        # Review 29 Sep 2026: the tariff's own Exchequer items.
        lines = r.get("revenue_requirement_gbp") or {}
        diag = r.get("diagnostics") or {}
        net_inc = diag.get("network_incidence") or {}
        entry.update({
            "tax_equivalent_received_gbp": float(lines.get("nationalised_tax_equivalent", 0.0) or 0.0),
            # The tariff's owned plant takes its carbon CfD under the case's funding lever; plant the
            # purchase leaves private (F28: Drax BECCS is legacy, and CCS Biomass is in no acquisition
            # group) keeps it from the Exchequer as today. Counting only owned plant booked a saving
            # the Exchequer does not make (GBP 225 m on every 2030 tariff case, fix 8 Nov 2026).
            "carbon_removal_cfd_exchequer_gbp": (
                (float(diag.get("carbon_removal_cfd_case_gbp", 0.0) or 0.0)
                 if diag.get("beccs_removal_funding", "exchequer") == "exchequer" else 0.0)
                + register_sum(case_id, "carbon_removal_cfd_revenue_gbp")
                - float(diag.get("carbon_removal_cfd_case_gbp", 0.0) or 0.0)),
            "legacy_levies_exchequer_gbp": -float((lines.get("nationalised_fit_domestic_to_taxation", 0.0) or 0.0)
                                                  + (lines.get("nationalised_whd_domestic_to_taxation", 0.0) or 0.0)),
            "network_fee_gbp": float(sum(v.get("fee_to_exchequer_gbp", 0.0) for v in net_inc.values())),
            "network_retained_gbp": float(sum(v.get("retained_by_exchequer_gbp", 0.0) for v in net_inc.values())),
            "acquisition_annuity_exchequer_gbp": float(diag.get("acquisition_annuity_exchequer_funded_gbp", 0.0) or 0.0),
            "treasury_margin_gbp": float(lines.get("nationalised_treasury_margin", 0.0) or 0.0),
            # F29: the systematic risk the state holds on the plant it owns, new build and acquired alike.
            "socialised_risk_gbp": register_sum(case_id, "socialised_risk_gbp"),
            "levers_for_flows": dict(levers)})
        # F11 on P3 (integration, 7 Oct 2026): the tariff's kept Capacity Market line, flat or from taxation.
        entry["cm_recovery_options"] = r.get("capacity_market_recovery_options")
        entry["electrified_standing_charge_gbp_ex_vat"] = (tariff_standing(r["bill"], electrified_basis)
                                                           if electrified_basis else None)
        entry["electrified_standing_basis"] = "private's less the standing part of the lines the tariff replaces"
        cases[case_id] = entry
    if institution:
        i3 = institution.get("institution_case_id")
        i1 = institution.get("i1_case_id")
        if i3 and i1 in cases:
            if "central" not in carbon:
                raise ValueError("the institution case settles on the central-dispatch months; pass --central-scenarios")
            entry = {**cases[i1], "case_id": i3, "stage": "institution_comparison", "dispatch": "central dispatch",
                     "bill_ex_vat": float(institution["i3_ex_vat_gbp"]),
                     "bill_inc_vat": float(institution["i3_inc_vat_gbp"]), **carbon["central"],
                     "institution_twin": i1,
                     # F19 (user decision, 6 Oct 2026): the two solved years from different solve runs.
                     "mixed_vintage": institution.get("mixed_vintage")}
            cases[i3] = entry
    # Review 29 Sep 2026: the expected loss the Exchequer carries on a network debt guarantee.
    levers_of = {c: (b.get("levers") or levers_by_case.get(c) or {}) for c, b in (bill.get("cases") or {}).items()}
    levers_of.update({c: m.get("levers") or {} for c, m in ((balancing or {}).get("cases") or {}).items()})
    levers_of.update({c: r.get("levers") or {} for c, r in tariff_cases.items()})
    for case_id, entry in cases.items():
        levers = levers_of.get(entry.get("institution_twin", case_id), {})
        entry.update(guarantee_expected_loss(levers, guarantee, case_id))
    rows = [ledger_row(entry, cases["private"], conv) for entry in cases.values()]
    for row in rows:
        row["headline_label"] = HEADLINE_CASES.get(row["case_id"])
        row["transmission_tax_allowance"] = "not tabulated in the RIIO-3 annex; not counted"
        if cases[row["case_id"]].get("stage") == "institution_comparison":
            row["mixed_vintage"] = cases[row["case_id"]].get("mixed_vintage")
    flows, flow_checks, invariant = [], [], []
    kept: dict = {}
    if investor is not None:
        flows, flow_checks = three_party_columns(
            rows, cases, register=register, bill=bill, account=account, tariff=tariff, institution=institution,
            balancing=balancing, balancing_accounts=balancing_accounts, carbon=carbon, network_lines=network_lines,
            investor=investor, conv=conv, keep=kept)
        invariant = check_resource_invariant(rows, catalogue or {})
    # User decision 4: the household total beside every bill (central key and legal incidence).
    household_totals(rows, conv, catalogue=catalogue or {}, incidence=incidence, contested=contested,
                     register=register, tariff=tariff, network_lines=network_lines,
                     model_year=(forecast_summary or {}).get("modelled_year"),
                     comparator_three_party=_vintaged_forecast(forecast_summary))
    # User, 13 Nov 2026: the electrified (heat pump + EV) household beside the 3,100 kWh benchmark.
    add_electrified_columns(rows, electrified_basis, vat_rate=float(conv["vat_rate"]))
    add_electrified_columns(rows, electrified_high_basis, vat_rate=float(conv["vat_rate"]), prefix=ELECTRIFIED_HIGH)
    # User decision (24 Oct 2026): every same-fleet P3 case on I3 beside its I1 figures (additive approximation).
    step_block = (institution_step_columns(rows, cases, tariff=tariff, institution=institution, carbon=carbon,
                                           conv=conv, kept=kept, investor=investor, incidence=incidence)
                  if investor is not None and "central" in carbon else {})
    uncertainty =uncertainties_by_case(rows, forecast_summary, tariff, float(conv["vat_rate"])) if forecast_summary else {}
    frame = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows])
    summary = {
        "price_basis": ("system money GBP real 2024 (PyPSA model money); per household GBP real 2025 per 3,100 kWh, "
                        "converted on the Exchequer's per-household key, metered supply (house rule 8; F18); a bill "
                        "line converts at its own recovery entry"),
        "conversion": dict(conv),
        # F19 (user decision, 6 Oct 2026): the institution step's two solved years and whether they are one run.
        "institution_solve_vintage": (institution or {}).get("solve_vintage"),
        "definitions": {
            "exchequer_net_vs_private_gbp": "change in the Exchequer's position against the private case; positive "
                                             "is more money for the Exchequer",
            "net_of_exchequer_gbp_inc_vat": "bill headline (inc VAT) less the Exchequer change per household: the "
                                             "bill plus what the case costs the Exchequer",
            "financing_saving_tax_transfer_gbp": "corporation tax forgone on the new-build rows the financing saving "
                                                  "is measured on (a transfer, house rule 9)",
            "financing_saving_resource_gbp": "financing saving less its tax transfer (the resource saving)",
        },
        "coverage": COVERAGE,
        "headline_cases": {c: l for c, l in HEADLINE_CASES.items() if c in cases},
        "cases": {row["case_id"]: row for row in rows},
    }
    summary["household_total"] = {
        "definition": ("household_total_gbp_inc_vat = bill (inc VAT) + taxpayer share, the Exchequer change against "
                       "private per household on the electricity-use key (3.1 MWh over metered supply, F18), the bill's own "
                       "allocation, so a pure bill-to-Exchequer transfer nets to about zero"),
        "legal_incidence": (dict(incidence, definition=(
            "household_total_legal_incidence_gbp_inc_vat: the taxpayer share on legal incidence, theta x Exchequer "
            "change / UK households, report-year money")) if incidence else None),
        "contested_worst_case": (dict(contested, definition=(
            "worst cases only: household_total_worst_with_contested_gbp_inc_vat adds the gilt-stock externality and "
            "the marginal cost of public funds, both contested and never central")) if contested else None),
    }
    summary["resource_invariant"] = {
        "rule": "every incidence preset leaves its parent's resource saving unchanged to GBP 1 (user decision 5)",
        "cases": invariant}
    summary["party_flows"] = {"checks": flow_checks, "parties": ["consumers", "exchequer", "private_investors",
                                                                "foreign", "resource"]}
    # Second round (29 Sep 2026): HPC on EDF's schedule, beside the other uncertainties.
    # Task Y13: a year whose bill switches the sensitivity off records why instead.
    if hpc_schedule and hpc_schedule.get("skipped"):
        summary["hpc_edf_schedule_sensitivity"] = {"skipped": hpc_schedule["skipped"]}
    elif hpc_schedule and forecast_summary:
        hpc = hpc_schedule_by_case(rows, register=register, tariff=tariff, forecast_summary=forecast_summary,
                                   conv=conv, schedule=hpc_schedule["schedule"], prices=hpc_schedule["prices"],
                                   valuation=(investor or {}).get("legacy_valuation"))
        summary["hpc_edf_schedule_sensitivity"] = {k: v for k, v in hpc.items() if k != "by_case"}
        for case_id, entry in (hpc.get("by_case") or {}).items():
            uncertainty.setdefault(case_id, {})["hpc_edf_schedule"] = dict(entry["central"], range={
                k: entry[k]["case_bill_change_gbp_inc_vat"] for k in ("low", "high")},
                label="HPC on EDF's current schedule")
    summary["uncertainties_by_case"] = uncertainty
    # Task Y13: a year whose list drops 2030-named items says which and why (none for 2030).
    if not_applicable := uncertainty_list((forecast_summary or {}).get("modelled_year"))[2]:
        summary["uncertainties_not_applicable"] = dict(not_applicable)
    summary["case_catalogue"] = dict(catalogue or {})
    if step_block:
        summary["institution_step_by_case"] = step_block
    if guarantee is not None:
        summary["network_debt_guarantee"] = {
            **{k: v for k, v in guarantee.items()},
            "method": ("expected loss = annual default probability x loss given default on the guaranteed notional debt "
                       "(RAV x notional gearing); HM Treasury Contingent Liability Approval Framework (April 2023) "
                       "paras 4.4-4.5: at least the expected cost, government risk neutral at the margin"),
        }
    if investor is not None:
        summary["three_party_identity"] = {
            "identity": ("consumer_change_vs_private_gbp + exchequer_net_vs_private_gbp + investor_change_vs_private_gbp "
                         "= resource_saving_vs_private_gbp, every case, system GBP (model price year)"),
            "tolerance_gbp": investors.IDENTITY_TOLERANCE_GBP,
            "definitions": {
                "consumer_change_vs_private_gbp": ("minus the ex-VAT bill change at system level (each moved line at its "
                                                   "own recovery denominator) minus the domestic VAT change"),
                "investor_change_vs_private_gbp": ("change in private owners' net position: revenue less fuel, carbon, "
                                                   "O&M and the required return on their capital, their share only, plus "
                                                   "payments to private parties outside the register; acquisitions on a "
                                                   "cash basis"),
                "resource_saving_vs_private_gbp": ("fall in the real cost of supply: capital at the financing party's rate "
                                                   "net of corporation tax, O&M, the state's overhead, fuel net of carbon, "
                                                   "the networks' cost of capital and net payments abroad"),
                "investor_basis": "required_return: the headline investor column's basis (user decision, 29 Sep 2026)",
                "investor_change_required_return_basis_gbp": "the headline investor column, named by its basis",
                "investor_change_observed_return_basis_gbp": ("diagnostic: capital providers credited with the capital "
                                                              "charges paid, so cost_of_capital_saving_gbp is an investor "
                                                              "loss, not a resource saving; with "
                                                              "resource_saving_observed_return_basis_gbp it closes too"),
                "investor_change_acquisition_gap_as_resource_gbp": ("acquisition cases only: the cost-of-capital gap "
                                                                    "on the legacy fleet (state valuation less owners', "
                                                                    "annuitised) moved from investors to the resource "
                                                                    "saving"),
                "guarantee_expected_loss_gbp": ("the Exchequer's expected loss on the network debt guarantee, in the "
                                                "Exchequer column and out of the resource saving"),
            },
            "choices": dict(investors.CHOICES),
            "headline_gbp_m": investors.headline_table([r for r in rows if r["case_id"] in HEADLINE_CASES])
            .round(3).to_dict(orient="records"),
        }
    summary["_party_flow_rows"] = flows
    return frame, summary


def three_party_columns(rows: list[dict], cases: Mapping[str, Mapping], *, register: pd.DataFrame, bill: Mapping,
                        account: Mapping, tariff: Mapping | None, institution: Mapping | None,
                        balancing: Mapping | None, balancing_accounts: Mapping[str, Mapping] | None,
                        carbon: Mapping[str, Mapping], network_lines: pd.DataFrame, investor: Mapping,
                        conv: Mapping, keep: dict | None = None) -> None:
    """The private-investor column and the resource saving on every row, and the identity (review 29 Sep 2026).

    Each case is measured on its own stage's ledger (:mod:`investor_ledger`):
    market-bill cases on the register and the public account, the state-bids
    case as the bid stack's effect on the private fleet plus the ownership effect
    on its own run group, tariff cases on the tariff's register ledger, the
    institution case on its I1 twin plus the measured I3 lines.  Refuses, naming
    the case, unless consumers + Exchequer + investors = resource saving to GBP 1.
    """
    public_rate = investor.get("network_public_rate")
    from scripts.public_ownership import bill_recovery

    two_volume = bill_recovery.is_two_volume(bill)
    positions: dict[tuple, dict] = {}

    def position(frame: pd.DataFrame, key: str, case_id: str) -> dict:
        if (key, case_id) not in positions:
            positions[(key, case_id)] = CaseRows.of(frame).memo(*_position_request(case_id))
        return positions[(key, case_id)]

    bill_cases = bill.get("cases") or {}
    groups = investor.get("balancing") or {}
    parties: dict[str, dict] = {}
    memos: dict[str, dict] = {}
    networks: dict[str, dict] = {}
    steps: dict[str, dict] = {}
    for case_id, entry in cases.items():
        stage = entry["stage"]
        network = investors.network_parties(network_lines, entry.get("networks"), public_rate,
                                            guarantee_expected_loss_gbp=entry.get("guarantee_expected_loss_gbp", 0.0),
                                            levers=entry.get("levers_for_flows"))
        networks[case_id] = network
        if stage == "market_bill":
            parties[case_id] = investors.market_case_parties(
                bill_case=bill_cases[case_id], bill_private=bill_cases["private"],
                account_case=(account.get("cases") or {}).get(case_id),
                position=position(register, "market", case_id), position_private=position(register, "market", "private"),
                network=network, co_investment_transfer_gbp=float(entry.get("co_investment_implicit_guarantee_gbp", 0.0)))
        elif stage == "balancing_bids":
            merged = balancing["cases"][case_id]
            group = groups.get(case_id)
            if group is None:
                raise ValueError(f"{case_id} settles on run group {merged.get('run_group')}: the investor ledger needs "
                                 "that group's register, bill summary and merged lines")
            lines = group["lines"].loc[group["lines"]["case_id"].eq(case_id)]
            # The bid stack on the private fleet (item BM1's decomposition term, a step, never a case) ...
            dispatch = investors.dispatch_step_parties(
                line_changes=investors.dispatch_line_changes(lines, group["rates"]),
                position_group=position(group["register"], merged["run_group"], "private"),
                position_market=position(register, "market", "private"),
                carbon_change_gbp=float(balancing_carbon(carbon, merged.get("run_group"))["carbon_receipts_gbp"])
                - float(carbon["market"]["carbon_receipts_gbp"]),
                retrading_change_gbp=float(group["retrading_group_gbp"]) - float(group["retrading_market_gbp"]))
            # ... then the ownership effect on that stack, a market-bill case of its own group.
            group_bill = group["bill"]["cases"]
            own_account = ((balancing_accounts or {}).get(case_id) or account).get("cases") or {}
            ownership = investors.market_case_parties(
                bill_case=group_bill[case_id], bill_private=group_bill["private"],
                account_case=own_account.get(case_id),
                position=position(group["register"], merged["run_group"], case_id),
                position_private=position(group["register"], merged["run_group"], "private"), network=network,
                co_investment_transfer_gbp=float(entry.get("co_investment_implicit_guarantee_gbp", 0.0)))
            parties[case_id] = investors.combine(dispatch, ownership)
            steps[case_id] = {"dispatch": dispatch, "ownership": ownership}
        elif stage == "acquisition_tariff":
            parties[case_id] = investors.tariff_case_parties(
                tariff["cases"][case_id], corporation_tax_legacy_gbp=entry["corporation_tax_forgone_legacy_gbp"],
                corporation_tax_new_build_gbp=entry["corporation_tax_forgone_new_build_gbp"], network=network,
                socialised_risk_gbp=(float(entry.get("socialised_risk_gbp", 0.0) or 0.0)
                                     - float(cases["private"].get("socialised_risk_gbp", 0.0) or 0.0)),
                private_plant_carbon_cfd_change_gbp=(
                    CaseRows.of(register).memo(*_private_plant_carbon_cfd_request(case_id))
                    if "carbon_removal_cfd_revenue_gbp" in register.columns else 0.0))
            memos[case_id] = CaseRows.of(register).memo(*_legacy_memo_request(case_id, investor.get("legacy_valuation")))
        elif stage == "institution_comparison":
            twin = entry["institution_twin"]
            tariff_ex_vat = float(tariff["cases"][twin]["bill"]["ex_vat_gbp"])
            if abs(float(institution["i1_ex_vat_gbp"]) - tariff_ex_vat) > 1e-6:
                raise ValueError(
                    f"the institution comparison's I1 bill ({float(institution['i1_ex_vat_gbp']):.4f}) is not the "
                    f"tariff's {twin} bill ({tariff_ex_vat:.4f}): it was built on another tariff; rebuild it first")
            step = investors.institution_step_parties(
                institution, carbon_change_gbp=float(carbon["central"]["carbon_receipts_gbp"])
                - float(carbon["market"]["carbon_receipts_gbp"]))
            parties[case_id] = investors.combine(parties[twin], step)
            memos[case_id] = memos.get(twin, {})
            steps[case_id] = {"twin": parties[twin], "step": step}
        else:
            raise ValueError(f"{case_id}: no investor measurement for stage {stage!r}")
    for row in rows:
        fill_three_party(row, parties[row["case_id"]], memos.get(row["case_id"]), two_volume=two_volume,
                         investor=investor, conv=conv)
    if keep is not None:
        # 24 Oct 2026: the parties the derived I3 rows build on (institution_step_columns).
        keep.update(parties=parties, memos=memos, two_volume=two_volume)
    # Review 29 Sep 2026: the itemised flows, each case checked against its columns.
    return party_flow_rows(rows, cases, parties, register=register, bill=bill, account=account, balancing=balancing,
                           balancing_accounts=balancing_accounts, investor=investor, carbon=carbon,
                           network_lines=network_lines, conv=conv, networks=networks, step_parts=steps)


def fill_three_party(row: dict, parties: Mapping, memo: Mapping | None, *, two_volume: bool, investor: Mapping,
                     conv: Mapping) -> None:
    """One row's three-party columns from its parties (in place): the identity, the memo, the per-household reads."""
    case_id = row["case_id"]
    row.update(investors.three_party_row(
        parties, vat_change_gbp=row["vat_change_vs_private_gbp"],
        exchequer_change_gbp=row["exchequer_net_vs_private_gbp"], case_id=case_id,
        risk_adjusted=bool(investor.get("risk_adjusted_view", False))))
    if memo:
        investors.acquisition_gap_as_resource(row, memo)
    if two_volume:
        # F19 (7 Oct 2026): on the two-volume basis (F18) no single volume turns the headline into a
        # system amount, so consumers' gain is the measured one (each line at its own entry, plus VAT).
        gain = float(row["consumer_change_vs_private_gbp"])
        row["consumer_gain_vs_private_gbp"] = gain
        row["consumer_and_exchequer_gain_vs_private_gbp"] = gain + float(row["exchequer_net_vs_private_gbp"])
        row["gain_beyond_resource_saving_gbp"] = (row["consumer_and_exchequer_gain_vs_private_gbp"]
                                                  - float(row["financing_saving_resource_gbp"]))
        for key in ("consumer_and_exchequer_gain_vs_private_gbp", "gain_beyond_resource_saving_gbp"):
            row[key.replace("_gbp", "_gbp_per_household")] = per_household(row[key], conv)
    row.update(investors.household_columns(row, lambda gbp: per_household(gbp, conv)))


# ── P3's sensitivities on central dispatch (user decision, 24 Oct 2026) ──────────────────────────────

#: The prefix of the I3 columns a same-fleet tariff case carries beside its I1 figures.
I3_APPROX_PREFIX = "i3_approx_"
#: The I1 row's columns restated on I3 (each read as the I1 column of the same name).
I3_APPROX_COLUMNS = (
    "bill_inc_vat", "bill_delta_vs_private_inc_vat", "bill_delta_vs_comparator_gbp_inc_vat",
    "taxpayer_share_gbp_per_household", "household_total_gbp_inc_vat", "household_total_delta_vs_private_gbp_inc_vat",
    "household_total_delta_vs_comparator_gbp_inc_vat", "household_total_legal_incidence_delta_vs_private_gbp_inc_vat",
    "consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp", "investor_change_vs_private_gbp",
    "resource_saving_vs_private_gbp", "three_party_identity_residual_gbp",
    "consumer_change_vs_private_gbp_per_household", "exchequer_net_vs_private_gbp_per_household",
    "investor_change_vs_private_gbp_per_household", "resource_saving_vs_private_gbp_per_household")
I3_MARKET_BILL_NOTE = ("tariff_pricing market_bill: the state charges the comparator's bill under either institution, "
                       "so the step moves the Treasury's margin, not the bill")


def _household_reads(out: dict, *, private_row: Mapping, comparator_row: Mapping, conv: Mapping,
                     incidence: Mapping | None) -> None:
    """:func:`household_totals`'s per-row reads for one derived row (in place), against the I1 row's comparator."""
    private_bill = float(private_row["bill_inc_vat"])
    exchequer = float(out["exchequer_net_vs_private_gbp"])
    out["taxpayer_share_gbp_per_household"] = -per_household(exchequer, conv)
    out["household_total_gbp_inc_vat"] = float(out["bill_inc_vat"]) + out["taxpayer_share_gbp_per_household"]
    out["household_total_delta_vs_private_gbp_inc_vat"] = out["household_total_gbp_inc_vat"] - private_bill
    out["bill_delta_vs_comparator_gbp_inc_vat"] = float(out["bill_inc_vat"]) - float(comparator_row["bill_inc_vat"])
    out["household_total_delta_vs_comparator_gbp_inc_vat"] = (out["household_total_gbp_inc_vat"]
                                                               - float(comparator_row["household_total_gbp_inc_vat"]))
    if incidence:
        out["household_total_legal_incidence_delta_vs_private_gbp_inc_vat"] = (
            float(out["bill_inc_vat"]) - legal_incidence_per_household(exchequer, conv, incidence) - private_bill)


def institution_step_columns(rows: list[dict], cases: Mapping[str, Mapping], *, tariff: Mapping | None,
                             institution: Mapping | None, carbon: Mapping[str, Mapping], conv: Mapping,
                             kept: Mapping | None, investor: Mapping, incidence: Mapping | None) -> dict:
    """User decision (24 Oct 2026): every same-fleet P3 case beside its I1 figures on I3, as an additive approximation.

    Only P3's central case is solved under central dispatch (I3); its sensitivities (purchase price
    basis, term, thermal basis, owner rate, presets, bounds, grids) are tariffed on the market year
    (I1). Each tariff case owning the institution twin's fleet (:func:`nationalised_institutions.same_fleet_as`)
    gets the measured I1-to-I3 step re-run on its own I1 lines and rates
    (:func:`nationalised_institutions.derive_case_step`): its bill, Exchequer (carbon on the central year,
    VAT), the investor column (unchanged: plant paid as today is carried at its I1 payments) and the
    resource saving, every derived row checked to close the three-party identity, written as
    ``i3_approx_*`` columns on the I1 row. Under ``tariff_pricing: market_bill`` the bill is the
    comparator's under either institution and the step goes to the Treasury's margin. The measured
    I3 case is not derived again: its twin's derived figures must equal it, or this refuses (a stale
    institution summary). Returns the summary block (label, step, applied and excluded cases).
    """
    from scripts.public_ownership import nationalised_institutions as ni

    if not institution or not kept:
        return {}
    i3, twin = institution.get("institution_case_id"), institution.get("i1_case_id")
    tariff_cases = (tariff or {}).get("cases") or {}
    if not i3 or i3 not in cases or twin not in tariff_cases:
        return {}
    by_case = {row["case_id"]: row for row in rows}
    private = cases["private"]
    carbon_change = float(carbon["central"]["carbon_receipts_gbp"]) - float(carbon["market"]["carbon_receipts_gbp"])
    applied, excluded = {}, {}
    for case_id, report in tariff_cases.items():
        row = by_case.get(case_id)
        if row is None or (cases.get(case_id) or {}).get("stage") != "acquisition_tariff":
            continue
        same, why = ni.same_fleet_as(report, tariff_cases[twin])
        if not same:
            excluded[case_id] = why
            continue
        derived = ni.derive_case_step(report, institution)
        step = investors.institution_step_parties(derived, carbon_change_gbp=carbon_change)
        entry = {**cases[case_id], **carbon["central"]}
        market_bill = (report.get("levers") or {}).get("tariff_pricing", "cost") == "market_bill"
        if market_bill:
            entry["treasury_margin_gbp"] = (_field(entry, "treasury_margin_gbp")
                                            - float(derived["institution_change_system_gbp"]))
            step = dict(step, bill_change_system_gbp=0.0)
        else:
            entry["bill_ex_vat"] = float(derived["i3_ex_vat_gbp"])
            entry["bill_inc_vat"] = float(derived["i3_inc_vat_gbp"])
        out = ledger_row(entry, private, conv)
        fill_three_party(out, investors.combine(kept["parties"][case_id], step), (kept.get("memos") or {}).get(case_id),
                         two_volume=bool(kept.get("two_volume")), investor=investor, conv=conv)
        comparator = by_case.get(str(row.get("comparator_case_id") or "private"), by_case["private"])
        _household_reads(out, private_row=by_case["private"], comparator_row=comparator, conv=conv,
                         incidence=incidence)
        for column in I3_APPROX_COLUMNS:
            if column in out:
                row[I3_APPROX_PREFIX + column] = out[column]
        row[I3_APPROX_PREFIX + "step_bill_gbp_inc_vat"] = float(out["bill_inc_vat"]) - float(row["bill_inc_vat"])
        row[I3_APPROX_PREFIX + "step_household_total_gbp_inc_vat"] = (float(out["household_total_gbp_inc_vat"])
                                                                      - float(row["household_total_gbp_inc_vat"]))
        row[I3_APPROX_PREFIX + "basis"] = ni.DERIVED_STEP_LABEL + (f"; {I3_MARKET_BILL_NOTE}" if market_bill else "")
        row[I3_APPROX_PREFIX + "institution_case_id"] = i3
        applied[case_id] = {
            "step_bill_gbp_inc_vat": row[I3_APPROX_PREFIX + "step_bill_gbp_inc_vat"],
            "step_system_gbp": float(derived["institution_change_system_gbp"]),
            "step_household_total_gbp_inc_vat": row[I3_APPROX_PREFIX + "step_household_total_gbp_inc_vat"],
            "step_resource_gbp": float(out["resource_saving_vs_private_gbp"]) - float(row["resource_saving_vs_private_gbp"]),
            "market_bill_pricing": market_bill}
    # The measured case is its twin plus the step (N10): the derived twin must reproduce it.
    measured, twin_row = by_case[i3], by_case.get(twin)
    if twin_row is not None and twin in applied:
        for column in I3_APPROX_COLUMNS:
            if column not in measured or I3_APPROX_PREFIX + column not in twin_row:
                continue
            gap = float(twin_row[I3_APPROX_PREFIX + column]) - float(measured[column])
            tolerance = investors.IDENTITY_TOLERANCE_GBP if column.endswith("_gbp") else 1e-6
            if abs(gap) > max(tolerance, 1e-9 * abs(float(measured[column]))):
                raise ValueError(f"{twin} plus the institution step gives {column} {float(twin_row[I3_APPROX_PREFIX + column]):,.6f}, "
                                 f"the measured {i3} {float(measured[column]):,.6f}: the institution summary is not on "
                                 "this tariff; rebuild it")
    steps = [v["step_bill_gbp_inc_vat"] for k, v in applied.items() if not v["market_bill_pricing"]]
    return {
        "label": ni.DERIVED_STEP_LABEL,
        "institution_case_id": i3, "twin_case_id": twin,
        "method": ("each same-fleet tariff case's I1 lines with the measured step's moved lines (fuel and carbon, "
                   "variable O&M, cross-border trade, foreign halves' rent, re-trading, real-time correction and "
                   "forecast-error charge, the I1 imbalance allowance) re-run at the case's own recovery rates; "
                   "investors unchanged; carbon on the central year; three-party identity checked per row"),
        "columns": [I3_APPROX_PREFIX + c for c in I3_APPROX_COLUMNS],
        "step_bill_gbp_inc_vat_range": [min(steps), max(steps)] if steps else None,
        "applied": applied, "excluded": excluded,
        "turn_down_basis": ((institution.get("measurement") or {}).get("market") or {}).get("turn_down_basis_by_month"),
    }


def ledger_options(public_ownership: Mapping) -> dict:
    """Decisions 68 and 69: the investor-side options a config sets (none in 2030: an empty dict).

    ``co_investment_genuine_share`` from ``co_ownership_finance`` (the share of the co-investors'
    rate reduction counted as a resource saving) and ``risk_adjusted_view`` from
    ``risk_adjusted_view.enabled`` (decision 69, default off).
    """
    out = {}
    co = public_ownership.get("co_ownership_finance")
    if co:
        from scripts.public_ownership.calculate_financing_scenarios import co_ownership_settings

        out["co_investment_genuine_share"] = co_ownership_settings(public_ownership)["co_investment_genuine_share"]
    view = public_ownership.get("risk_adjusted_view")
    if view is not None:
        if not isinstance(view, Mapping) or set(view) - {"enabled"}:
            raise ValueError("risk_adjusted_view takes {enabled: true|false} only")
        out["risk_adjusted_view"] = bool(view.get("enabled", False))
    return out


def check_institution_fleet_ratio(institution: Mapping | None, block: Mapping, group: str | None) -> None:
    """FES 2050 decision 57: refuse an institution comparison built without the group's forecast-error ratio.

    A group that declares ``institution_comparison.forecast_error_charge.fleet_ratio`` (2050:
    2.32, decision 18) must read an I3 bill whose charge was rescaled to it
    (``forecast_error_fleet_ratio`` in the summary). A 2030 group declares none: unchecked.
    """
    if not institution or not group:
        return
    own = ((((block.get("reference_run_groups") or {}).get(group) or {}).get("institution_comparison") or {})
           .get("forecast_error_charge") or {})
    if own.get("fleet_ratio") is None or own.get("enabled", True) is False:
        return
    applied = institution.get("forecast_error_fleet_ratio")
    if applied is None or abs(float(applied) - float(own["fleet_ratio"])) > 1e-9:
        raise ValueError(f"{group} declares forecast-error fleet ratio {own['fleet_ratio']} (decision 18), but the "
                         f"institution comparison was rescaled to {applied}: rebuild it with --forecast-error-fleet-ratio")


def balancing_carbon(carbon: Mapping[str, Mapping], run_group: str | None) -> Mapping:
    """The carbon receipts of one balancing group's own months (FES 2050 decision 53).

    ``carbon["balancing_by_group"]`` (``main``: each group measured on its own twelve
    months) when the ledger was given more than one balancing group; else the single
    group's ``carbon["balancing"]`` (every 2030 ledger, unchanged).
    """
    by_group = carbon.get("balancing_by_group") or {}
    if run_group in by_group:
        return by_group[run_group]
    if len(by_group) > 1:
        raise ValueError(f"no carbon measured for balancing group {run_group!r}; measured: {sorted(by_group)}")
    return carbon["balancing"]


def balancing_scenarios_by_group(scenarios, patterns: Mapping[str, str]) -> dict[str, list[str]]:
    """FES 2050 decision 53: split the balancing months among their run groups by each group's scenario pattern.

    ``patterns`` is ``{run_group: scenario_pattern}`` (``"FES25_HT_2050_{month}_p2_bids_25pct"``).
    Every scenario must match exactly one group, or the split fails: carbon measured on
    another group's months would be counted in this group's state-bids case.
    """
    import re

    regex = {g: re.compile("^" + re.escape(str(p)).replace(re.escape("{month}"), "[A-Za-z]+") + "$")
             for g, p in patterns.items()}
    out: dict[str, list[str]] = {g: [] for g in patterns}
    for scenario in scenarios:
        hits = [g for g, r in regex.items() if r.match(str(scenario))]
        if len(hits) != 1:
            raise ValueError(f"balancing month {scenario!r} matches {hits or 'no'} run group(s) of {sorted(patterns)}")
        out[hits[0]].append(str(scenario))
    if empty := sorted(g for g, s in out.items() if not s):
        raise ValueError(f"balancing groups {empty} have no months among the scenarios given")
    return out


def investor_inputs(*, register_path, bill_summary_path, network_account_path,
                    assumptions: Mapping, recovery: Mapping, bill: Mapping, balancing: Mapping | None,
                    tariff: Mapping | None, balancing_summary_path=None) -> dict:
    """Everything :func:`three_party_columns` reads beyond the ledger's own inputs, found beside them.

    Like :func:`balancing_group_accounts`, each file is named by the workflow's
    convention next to the ones the ledger is given, and the rule declares every
    one: the network account's JSON (the public rate), the bill's per-line
    components (recovery rates), the legacy valuation (tariff cases), and for a
    state-bids case its own run group's register and bill summary, the merged
    lines and both forecast bills' re-trading.
    """
    from scripts.public_ownership.nationalised_tariff import conversion as tariff_conversion, recovery_rates

    network_json = Path(network_account_path).with_suffix(".json")
    out = {"network_public_rate": (json.loads(network_json.read_text(encoding="utf-8")).get("public_real_rate")
                                   if network_json.exists() else None)}
    if tariff and (tariff.get("cases") or {}):
        valuation = Path(register_path).with_name(
            Path(register_path).name.replace("_asset_finance_register" + Path(register_path).suffix,
                                             "_legacy_valuation.csv"))
        if not valuation.exists():
            raise FileNotFoundError(f"the tariff cases need the legacy valuation {valuation}")
        out["legacy_valuation"] = pd.read_csv(valuation)
    merged_cases = (balancing or {}).get("cases") or {}
    if merged_cases:
        components = Path(bill_summary_path).with_name(Path(bill_summary_path).name.replace("_summary.json", ".csv"))
        rates = recovery_rates(pd.read_csv(components), tariff_conversion(assumptions, recovery, bill))["replaced_rates"]
        # The forecast bills sit in the analysis directory beside public_ownership/.
        # F20 parity (14 Oct 2026): a parity run writes this layer into its own root and reads the
        # forecast bills from the input tree (PYPSA_GB_INPUT_RESOURCES); unset, beside public_ownership/.
        analysis = (Path(os.environ["PYPSA_GB_INPUT_RESOURCES"]) / "analysis"
                    if os.environ.get("PYPSA_GB_INPUT_RESOURCES")
                    else Path(bill_summary_path).resolve().parents[1] / "analysis")

        def retrading(forecast_case: str) -> float:
            path = analysis / f"{forecast_case}_consumer_bill_future_summary.json"
            if not path.exists():
                raise FileNotFoundError(f"the state-bids case needs the forecast bill summary {path}")
            costs = json.loads(path.read_text(encoding="utf-8")).get("balancing_costs") or {}
            return float(costs.get("interconnector_retrading_cost_gbp") or 0.0) \
                if costs.get("include_interconnector_retrading") else 0.0

        if balancing_summary_path is None:
            raise ValueError("the state-bids case needs the merged lines beside the balancing summary")
        # The merge writes its per-line table beside its summary (merge_balancing_groups.py).
        lines = pd.read_csv(Path(balancing_summary_path).with_name(
            Path(balancing_summary_path).name.replace("_summary.json", ".csv")))
        # F18 (8 Oct 2026): a moved line the tariff does not replace (the supplier line, whose
        # EBIT and headroom follow the bill's costs) converts at its own recovery entry.
        from scripts.public_ownership import bill_recovery

        rates = dict(rates)
        if bill_recovery.is_two_volume(bill):
            for line in lines["component_id"].astype(str).unique():
                if not rates.get(line) and line in (bill_recovery.line_recovery_block(bill) or {}):
                    rates[line] = bill_recovery.household_per_system(bill, line)
        groups = {}
        for case_id, merged in merged_cases.items():
            group = merged["run_group"]
            groups[case_id] = {
                "register": load_register(Path(register_path).with_name(
                    f"{group}_asset_finance_register{Path(register_path).suffix}"),
                                          [case_id, "private"]),
                "bill": json.loads(Path(bill_summary_path).with_name(
                    f"{group}_consumer_bill_by_ownership_case_summary.json").read_text(encoding="utf-8")),
                "lines": lines, "rates": rates,
                "retrading_market_gbp": retrading(balancing["market_forecast_case"]),
                "retrading_group_gbp": retrading(merged["forecast_case"]),
            }
        out["balancing"] = groups
    return out


def balancing_group_accounts(balancing: Mapping | None, market_account_path) -> dict:
    """Each state-bids case's own run group's public account, read beside the market group's.

    The balancing merge records the run group a case settles on; that group's
    finance layer writes ``{run_group}_public_account_summary.json`` in the same
    directory, before its bill (and so before the merge) can exist.
    """
    out = {}
    for case_id, entry in ((balancing or {}).get("cases") or {}).items():
        group = entry.get("run_group")
        if not group:
            continue
        path = Path(market_account_path).with_name(f"{group}_public_account_summary.json")
        if not path.exists():
            raise FileNotFoundError(f"{case_id} settles on run group {group}, but its public account {path} is missing")
        out[case_id] = json.loads(path.read_text(encoding="utf-8"))
    return out


# ── review 29 Sep 2026: household totals, the invariant, uncertainties, flows ──

#: Levers whose presets must leave the parent's resource saving unchanged (user decision 5).
RESOURCE_INVARIANT_TOLERANCE_GBP = 1.0


def load_tax_incidence(path) -> dict:
    """The legal-incidence key's parameters (household share of UK tax receipts, UK households)."""
    frame = pd.read_csv(path).set_index("parameter")
    return {"theta": float(frame.at["household_share_of_tax_receipts", "value"]),
            "theta_low": float(frame.at["household_share_of_tax_receipts", "low"]),
            "theta_high": float(frame.at["household_share_of_tax_receipts", "high"]),
            "households": float(frame.at["uk_households", "value"]),
            "year": str(frame.at["household_share_of_tax_receipts", "year"]),
            "sources": sorted(set(frame["sources"].astype(str)))}


def load_contested_costs(path) -> dict:
    frame = pd.read_csv(path).set_index("parameter")
    return {k: float(frame.at[k, "value"]) for k in frame.index} | {"sources": sorted(set(frame["sources"].astype(str)))}


def legal_incidence_per_household(system_gbp: float, conv: Mapping, incidence: Mapping, theta: float | None = None) -> float:
    """An Exchequer amount (model-year GBP) per household on legal incidence: theta x GBP / households, in report money."""
    return float(system_gbp) * float(conv["factor"]) * float(theta if theta is not None else incidence["theta"]) / float(
        incidence["households"])


def state_borrowing_gbp(register: pd.DataFrame | None, case_id: str, *, tariff_case: Mapping | None,
                        network_lines: pd.DataFrame | None, levers: Mapping,
                        register_part: tuple[float, float] | None = None) -> float:
    """What the state borrows for a case: its share of new-build capex, the purchase prices and the wires.

    For the contested gilt-stock externality only (worst-case household totals):
    capex is the capital charge over its CRF at the row's own rate and life.
    ``register_part`` (F20 phase 1) is :func:`state_borrowing_register_part` already
    computed off the register (``register`` is then not read).
    """
    capex, purchase = register_part if register_part is not None else state_borrowing_register_part(register, case_id)
    ic = float(((tariff_case or {}).get("diagnostics") or {}).get("interconnector_acquisition_price_gbp", 0.0) or 0.0)
    wires = 0.0
    if levers.get(LEVER_NETWORKS) == "acquired" and network_lines is not None:
        from scripts.public_ownership.network_finance_account import select_rows

        selected = select_rows(network_lines, "acquired", levers)
        wires = float((_num(selected, "rav_gbp") + _num(selected, "premium_capital_gbp")).sum())
    return capex + purchase + ic + wires


def state_borrowing_register_part(register: pd.DataFrame, case_id: str) -> tuple[float, float]:
    """The register's part of :func:`state_borrowing_gbp`: the state's new-build capex and the purchase prices."""
    rows = register.loc[register["case_id"].eq(case_id)]
    share = _num(rows, "ownership_share")
    capital = _num(rows, "annual_capital_recovery_gbp")
    rate = pd.to_numeric(rows["wacc"], errors="coerce")
    life = _num(rows, "remaining_life_years")
    new_build = rows["finance_vintage"].astype(str).eq("new_build") & share.gt(0) & rate.notna() & life.gt(0)
    capex = 0.0
    for idx in rows.index[new_build]:
        capex += float(share[idx]) * float(capital[idx]) / capital_recovery_factor(float(rate[idx]), float(life[idx]))
    legacy = rows["finance_vintage"].astype(str).eq("legacy") & share.gt(0)
    purchase = float(_num(rows, "acquisition_price_gbp")[legacy].sum())
    return capex, purchase


# -- F20 phase 1: the register-derived quantities the ledger's passes share ---------
# Each pass over the cases used to read every case's rows again (CaseRows keeps 48
# cases); each quantity is now one request, prefetched in register order by
# :func:`build` and read back by the pass through ``CaseRows.memo`` (a deep copy).
# A request is ``(label, case ids, compute(rows))``, the call the pass made before.

#: The register columns ``build``'s market items sum per case.
REGISTER_SUM_COLUMNS = ("co_investment_rate_gain_gbp", "annual_tax_equivalent_gbp", "carbon_removal_cfd_revenue_gbp")


def _tax_request(case_id: str, has_upstream_ct: bool, wedges: Mapping, private_tax) -> tuple:
    if has_upstream_ct:
        return "tax_upstream", (case_id,), lambda rows: upstream_corporation_tax(rows, case_id)
    return "tax_computed_here", (case_id, "private"), lambda rows: corporation_tax_forgone(rows, case_id, wedges, private_tax)


def _sum_request(case_id: str, column: str) -> tuple:
    return f"sum:{column}", (case_id,), lambda rows: float(_num(rows, column).sum())


def _private_plant_carbon_cfd_request(case_id: str) -> tuple:
    """A tariff case's carbon CfD on the rows it does not cost (they stay private and the Exchequer pays
    them as today), less private's on the same assets; the tariff's row rule (owned and resolved)."""
    column = "carbon_removal_cfd_revenue_gbp"

    def compute(rows):
        case = rows.loc[rows["case_id"].astype(str).eq(case_id)].set_index("asset_id")
        private = rows.loc[rows["case_id"].astype(str).eq("private")].set_index("asset_id").reindex(case.index)
        costed = case["state_owned"].astype(bool) & case["annual_required_revenue_gbp"].notna()
        return float((_num(case, column) - _num(private, column))[~costed].sum())
    return "private_plant_carbon_cfd_change", (case_id, "private"), compute


def _position_request(case_id: str) -> tuple:
    return "register_position", (case_id,), lambda rows: investors.register_position(rows, case_id)


def _legacy_memo_request(case_id: str, valuation: pd.DataFrame | None) -> tuple:
    def compute(case_rows):
        rows_case = case_rows.loc[case_rows["finance_vintage"].astype(str).eq("legacy")
                                  & pd.to_numeric(case_rows["acquisition_price_gbp"], errors="coerce").notna()]
        return investors.legacy_acquisition_memo(rows_case, valuation)
    return "legacy_acquisition_memo", (case_id,), compute


def _borrowing_request(case_id: str) -> tuple:
    return "state_borrowing_register_part", (case_id,), lambda rows: state_borrowing_register_part(rows, case_id)


def _hpc_request(case_id: str) -> tuple:
    def compute(case_rows):
        from scripts.public_ownership import hpc_schedule as hs

        owned = case_rows.loc[case_rows["asset_id"].astype(str).str.startswith(hs.HPC_ASSET_PREFIX)]
        return bool(len(owned)) and bool(owned["state_owned"].astype(bool).any())
    return "hpc_state_owns", (case_id,), compute


def _components_request(case_id: str) -> tuple:
    from scripts.public_ownership import party_flows as pf

    return "register_components", (case_id, "private"), lambda rows: pf.register_components(rows, case_id)


def _register_requests(register, *, bill: Mapping, balancing: Mapping | None, tariff: Mapping | None,
                       institution: Mapping | None, investor: Mapping | None, catalogue: Mapping | None,
                       contested: Mapping | None, hpc: bool, has_upstream_ct: bool, wedges: Mapping,
                       private_tax) -> list[tuple]:
    """Every request the passes of :func:`build` will make of the market register, for ``CaseRows.prefetch``.

    It mirrors the passes' own choice of cases; a request a pass never makes only
    costs its computation, and one a pass makes that is missing here is computed
    when asked (so the list affects speed only, never a number).
    """
    market = [str(c) for c in (bill.get("cases") or {})]
    held = register.cases
    bids = [str(c) if c in held else str(m.get("base_case"))
            for c, m in ((balancing or {}).get("cases") or {}).items()]
    tariffed = [str(c) for c in ((tariff or {}).get("cases") or {})]
    requests = [_tax_request(c, has_upstream_ct, wedges, private_tax)
                for c in dict.fromkeys(market + bids + tariffed) if c in held]
    requests += [_sum_request(c, column) for c in dict.fromkeys(market + bids) for column in REGISTER_SUM_COLUMNS
                 if column in register.columns]
    twin = {}
    if institution and institution.get("institution_case_id") and institution.get("i1_case_id") in tariffed:
        twin[str(institution["institution_case_id"])] = str(institution["i1_case_id"])
    every = list(dict.fromkeys(market + [str(c) for c in ((balancing or {}).get("cases") or {})] + tariffed
                               + list(twin)))
    if investor is not None:
        requests += [_position_request(c) for c in dict.fromkeys(["private", *market])]
        requests += [_legacy_memo_request(c, investor.get("legacy_valuation")) for c in tariffed]
        requests += [_components_request(c) for c in market]
    if contested:
        worst = [c for c in every if dict((catalogue or {}).get(c) or {}).get("bound") == "worst"]
        requests += [_borrowing_request(twin.get(c, c)) for c in worst]
    if hpc:
        requests += [_hpc_request(twin.get(c, c)) for c in every]
    return requests


def contested_fiscal_costs(row: Mapping, borrowing_gbp: float, contested: Mapping, conv: Mapping,
                           model_year: int) -> dict:
    """User decision C (29 Sep 2026): the gilt-stock externality and the marginal cost of public funds.

    Both are contested and enter each scenario's WORST-case household total only,
    never the bill or a central figure.  Gilt stock (user, 30 Oct 2026: the financing
    workbook v6.2's method, ``Build-Out`` section F): the state's borrowing plus its
    contingent exposure (a stress-case loss on it) as a share of the model year's GDP
    raises gilt yields by ``bp_per_pp`` basis points per point, net of the offset; the
    shift reprices the share of the marketable gilt stock (grown with GDP) refinanced
    since 2025 over the stock's average maturity.  MCPF: each pound of net Exchequer
    cost costs ``MCPF - 1`` more in deadweight (Kleven and Kreiner 2006, UK 1.26).
    """
    years = float(model_year) - 2024.0
    gdp = float(contested["gdp_2024_gbp"]) * (1.0 + float(contested["real_gdp_growth"])) ** years
    exposure = float(borrowing_gbp) * (1.0 + float(contested["stress_case_loss_share"]))
    shift_bp = (float(contested["yield_bp_per_pp_debt_to_gdp"]) * 100.0 * exposure / gdp
                * (1.0 - float(contested["counterfactual_offset_share"])))
    stock = float(contested["marketable_gilt_stock_2024_gbp"]) * gdp / float(contested["gdp_2024_gbp"])
    refinanced = min(1.0, max(0.0, (float(model_year) - float(contested["refinancing_from_year"]))
                              / float(contested["average_gilt_maturity_years"])))
    gilt = shift_bp / 1e4 * stock * refinanced
    exchequer_cost = max(-float(row["exchequer_net_vs_private_gbp"]), 0.0)
    mcpf = (float(contested["marginal_cost_of_public_funds"]) - 1.0) * exchequer_cost
    return {"contested_state_borrowing_gbp": float(borrowing_gbp), "contested_gilt_yield_shift_bp": shift_bp,
            "contested_gilt_stock_externality_gbp": gilt, "contested_mcpf_deadweight_gbp": mcpf,
            "contested_gilt_stock_externality_gbp_per_household": per_household(gilt, conv),
            "contested_mcpf_deadweight_gbp_per_household": per_household(mcpf, conv),
            "contested_gilt_stock_repriced_gbp": stock * refinanced,
            "contested_label": ("contested: the gilt-stock externality (financing workbook v6.2, Build-Out section F) "
                                "and the marginal cost of public funds (Kleven and Kreiner 2006) are in the worst "
                                "case's household total only")}


def cm_recovery_option_columns(row: dict, conv: Mapping) -> None:
    """F11 (user decision, 6 Oct 2026): beside a case's headline, its Capacity Market recovered (b) flat per MWh
    and (c) from taxation (in place on ``row``; nothing where the bill carries no options). A tariff case's
    options are on the Capacity Market agreements it keeps (``nationalised_capacity_market_not_acquired``).

    (b) moves only the household line (business pays the difference), so only the bill moves.
    (c) takes the line off the bill and puts its whole system cost, plus the domestic VAT
    no longer charged on it, on the Exchequer: the taxpayer share on the household
    total's own key. Neither is an ownership saving; the headline is unchanged.
    """
    options = row.get("cm_recovery_options")
    if not options:
        return
    vat = 1.0 + float(conv["vat_rate"])
    line = float(options["household_line_peak_weighted_gbp_per_3100kwh_ex_vat"])
    flat_change = float(options["b_flat_per_mwh"]["bill_change_gbp_per_3100kwh_ex_vat"]) * vat
    row["cm_household_line_gbp_inc_vat"] = line * vat
    row["cm_option_b_flat_bill_change_gbp_inc_vat"] = flat_change
    row["cm_option_b_flat_bill_gbp_inc_vat"] = float(row["bill_inc_vat"]) + flat_change
    taxation_cost = float(options["c_taxation"]["exchequer_cost_gbp_2024"]) + domestic_vat(line, conv)
    row["cm_option_c_taxation_bill_gbp_inc_vat"] = float(row["bill_inc_vat"]) - line * vat
    row["cm_option_c_taxation_exchequer_cost_gbp"] = taxation_cost
    row["cm_option_c_taxpayer_share_gbp_per_household"] = (float(row["taxpayer_share_gbp_per_household"])
                                                           + per_household(taxation_cost, conv))
    row["cm_option_c_household_total_gbp_inc_vat"] = (row["cm_option_c_taxation_bill_gbp_inc_vat"]
                                                      + row["cm_option_c_taxpayer_share_gbp_per_household"])
    row["cm_recovery_options_note"] = options.get("note")


def household_totals(rows: list[dict], conv: Mapping, *, catalogue: Mapping[str, Mapping], incidence: Mapping | None,
                     contested: Mapping | None, register: pd.DataFrame, tariff: Mapping | None,
                     network_lines: pd.DataFrame | None, comparator_three_party: bool = False,
                     model_year: int | None = None) -> None:
    """User decision 4: the household total, bill plus taxpayer share, beside every bill (in place on ``rows``).

    Central key (M1): the electricity-use key, the bill's own allocation, so a pure
    bill-to-Exchequer transfer nets to about zero: ``bill - Exchequer change x
    3.1 MWh / final demand``.  Sensitivity (M2): legal tax incidence, the
    household share of tax receipts over UK households.  Each against private
    and against the case's own comparator (a background lever moves private too).
    """
    by_case = {row["case_id"]: row for row in rows}
    private_bill = float(by_case["private"]["bill_inc_vat"])
    tariff_cases = (tariff or {}).get("cases") or {}
    for row in rows:
        meta = dict(catalogue.get(row["case_id"]) or {})
        row["case_kind"] = meta.get("kind")
        row["case_parent"] = meta.get("parent")
        row["case_preset"] = meta.get("preset")
        row["case_bound"] = meta.get("bound")
        row["grid_toggles"] = json.dumps(meta.get("toggles")) if meta.get("toggles") else None
        row["grid_memberships"] = json.dumps(meta.get("grid_memberships")) if meta.get("grid_memberships") else None
        exchequer = float(row["exchequer_net_vs_private_gbp"])
        row["taxpayer_share_gbp_per_household"] = -per_household(exchequer, conv)
        row["household_total_gbp_inc_vat"] = float(row["bill_inc_vat"]) + row["taxpayer_share_gbp_per_household"]
        row["household_total_delta_vs_private_gbp_inc_vat"] = row["household_total_gbp_inc_vat"] - private_bill
        row["household_total_key"] = "electricity use (3.1 MWh over final demand), the bill's own allocation"
        if incidence:
            share = -legal_incidence_per_household(exchequer, conv, incidence)
            row["taxpayer_share_legal_incidence_gbp_per_household"] = share
            row["household_total_legal_incidence_gbp_inc_vat"] = float(row["bill_inc_vat"]) + share
            row["household_total_legal_incidence_delta_vs_private_gbp_inc_vat"] = (
                row["household_total_legal_incidence_gbp_inc_vat"] - private_bill)
        if contested and meta.get("bound") == "worst":
            levers = row.get("levers_for_flows") or {}
            borrower = row.get("institution_twin") or row["case_id"]
            borrowing = state_borrowing_gbp(None, borrower, tariff_case=tariff_cases.get(row["case_id"]),
                                            network_lines=network_lines, levers=levers,
                                            register_part=CaseRows.of(register).memo(*_borrowing_request(borrower)))
            if model_year is None:
                raise ValueError("the contested gilt-stock externality needs the model year (the forecast summary's "
                                 "modelled_year)")
            row.update(contested_fiscal_costs(row, borrowing, contested, conv, int(model_year)))
            row["household_total_worst_with_contested_gbp_inc_vat"] = (
                row["household_total_gbp_inc_vat"] + row["contested_gilt_stock_externality_gbp_per_household"]
                + row["contested_mcpf_deadweight_gbp_per_household"])
    for row in rows:
        cm_recovery_option_columns(row, conv)
    # Against the comparator (house rule 3): the private case at the same point on every all_cases lever.
    settings = {row["case_id"]: (row.get("levers_for_flows") or {}) for row in rows}
    for row in rows:
        comparator = comparator_for_settings(row["case_id"], settings) if settings.get(row["case_id"]) else "private"
        comp = by_case.get(comparator, by_case["private"])
        row["comparator_case_id"] = comp["case_id"]
        row["bill_delta_vs_comparator_gbp_inc_vat"] = float(row["bill_inc_vat"]) - float(comp["bill_inc_vat"])
        row["household_total_delta_vs_comparator_gbp_inc_vat"] = (row["household_total_gbp_inc_vat"]
                                                                   - comp["household_total_gbp_inc_vat"])
        if comparator_three_party:
            # FES 2050 decision 54: each party's change against the comparator (private at the same
            # point on every all_cases lever, e.g. private@private_rate_neso_wind_ramp), not flat private.
            for column in THREE_PARTY_COLUMNS:
                if column in row and column in comp:
                    row[column.replace("_vs_private_", "_vs_comparator_")] = float(row[column]) - (
                        float(comp[column]) if comp["case_id"] != "private" else 0.0)
    # A tariff case's purchase price at which its household total equals private's.
    for case_id, report in tariff_cases.items():
        row = by_case.get(case_id)
        block = (report or {}).get("break_even_purchase_price") or {}
        if row is None or not block:
            continue
        crf = float(block.get("annuity_factor") or 0.0)
        per_gbp_bill = float(block.get("bill_gbp_inc_vat_per_10bn_of_price") or 0.0) / 1e10
        # Domestic VAT on the higher bill comes back to taxpayers on the same key.
        vat_back = (per_gbp_bill / (1.0 + float(conv["vat_rate"])) * float(conv["vat_rate"]) / float(conv["mwh"])
                    * float(conv["domestic_mwh"]) / float(conv["factor"])) * per_household(1.0, conv)
        exchequer_funded = block.get("break_even_purchase_price_bill_gbp") is None
        slope = (per_household(crf, conv) if exchequer_funded else per_gbp_bill - vat_back)
        delta = row["household_total_delta_vs_comparator_gbp_inc_vat"]
        snapshot = float(block["purchase_price_gbp"]) - delta / slope if slope > 0 else None
        lifetime = block.get("break_even_purchase_price_lifetime_gbp")
        # F19 P3 decision 5 (integration, 7 Oct 2026): the headline break-even is the tariff's LIFETIME
        # price (the purchase's present value at the public rate is zero). It does not depend on who
        # pays (bill or taxation, VAT on both legs alike), so the household total's is the same price.
        # The 2030 figures (the price at which this year's bill, or household total, equals the
        # comparator's) are kept, labelled as snapshots.
        row["break_even_purchase_price_household_total_2030_snapshot_gbp"] = snapshot
        row["break_even_purchase_price_bill_2030_snapshot_gbp"] = block.get(
            "break_even_purchase_price_2030_snapshot_bill_gbp")
        row["break_even_purchase_price_household_total_gbp"] = (float(lifetime) if lifetime is not None else snapshot)
        row["break_even_purchase_price_bill_gbp"] = (float(lifetime) if lifetime is not None
                                                     and not exchequer_funded
                                                     else block.get("break_even_purchase_price_bill_gbp"))
        row["break_even_purchase_price_basis"] = ("lifetime: the purchase's present value at the public rate is zero"
                                                  if lifetime is not None else "2030 snapshot")
        row["household_total_gbp_inc_vat_per_10bn_of_price"] = slope * 1e10


def check_resource_invariant(rows: list[dict], catalogue: Mapping[str, Mapping],
                             tolerance: float = RESOURCE_INVARIANT_TOLERANCE_GBP) -> list[dict]:
    """User decision 5: every incidence preset leaves its parent's resource saving unchanged to GBP 1.

    A preset moves only incidence levers (who pays), so the real cost of supply
    cannot move; a gap means a transfer has been booked as a cost somewhere.
    Refuses, naming the case; returns the check per preset case.
    """
    by_case = {row["case_id"]: row for row in rows}
    out = []
    for case_id, meta in catalogue.items():
        if meta.get("kind") != "preset" or case_id not in by_case:
            continue
        parent = by_case.get(meta.get("parent"))
        if parent is None or "resource_saving_vs_private_gbp" not in by_case[case_id]:
            continue
        gap = float(by_case[case_id]["resource_saving_vs_private_gbp"]) - float(parent["resource_saving_vs_private_gbp"])
        out.append({"case_id": case_id, "parent": parent["case_id"], "preset": meta.get("preset"),
                    "resource_saving_gap_gbp": gap})
        by_case[case_id]["resource_saving_gap_vs_parent_gbp"] = gap
        if abs(gap) > tolerance:
            raise ValueError(f"{case_id} (preset {meta.get('preset')}) moves the resource saving of {parent['case_id']} "
                             f"by GBP {gap:,.2f}: a preset may move money between the parties, never the real cost")
    return out


#: Cross-cutting uncertainties on the private bill (forecast summary diagnostics) and
#: whether the acquisition tariff replaces the line they move (so they move P3's gap).
UNCERTAINTIES = {
    "ro_exchequer_relief_continued": ("renewables_obligation", "Renewables Obligation relief for domestic customers "
                                      "continued beyond March 2029 (75%)"),
    "forward_premium_minus_3pct": ("procurement_forward_premium", "Forward premium on wholesale energy, -3%"),
    "forward_premium_plus_5pct": ("procurement_forward_premium", "Forward premium on wholesale energy, +5%"),
    "forward_premium_stress_plus_10pct": ("procurement_forward_premium", "Forward premium on wholesale energy, +10% stress"),
    "shaping_low_1pct": ("shaping_adjustment", "Wholesale shaping allowance, low"),
    "shaping_high_7pct": ("shaping_adjustment", "Wholesale shaping allowance, high"),
    "capacity_market_t4_at_2029_30_clearing": ("capacity_market", "Capacity Market T-4 2030/31 at the 2029/30 clearing price"),
    # User decision (6 Oct 2026): the 2030/31 T-4 clears on the CP30 fleet; its cases move the
    # uncertain supply inputs (the 0x and 2x new-build pipeline cases are retired).
    "capacity_market_t4_storage_efc_low": ("capacity_market", "Capacity Market T-4 with storage de-rated at "
                                           "NESO's 2027/28 level"),
    "capacity_market_t4_ldes_excluded": ("capacity_market", "Capacity Market T-4 without the new cap-and-floor LDES"),
    "capacity_market_t4_dsr_low": ("capacity_market", "Capacity Market T-4 with DSR at its 2028/29 entry"),
    "capacity_market_t4_dsr_high": ("capacity_market", "Capacity Market T-4 with FES DSR at peak"),
    "bics_levy_exemption_2027": ("bics_levy", "Network Charging Compensation levy exemption from 2027"),
}

#: The forecast-summary keys each levy/CM uncertainty is read from, for the 2030 year.
_LEVY_SOURCES_2030 = (("capacity_market_t4_at_2029_30_clearing", "capacity_market_t4_2030_31_at_2029_30_clearing_27_10"),
                      ("capacity_market_t4_storage_efc_low", "capacity_market_t4_case_storage_efc_low"),
                      ("capacity_market_t4_ldes_excluded", "capacity_market_t4_case_ldes_excluded"),
                      ("capacity_market_t4_dsr_low", "capacity_market_t4_case_dsr_low"),
                      ("capacity_market_t4_dsr_high", "capacity_market_t4_case_dsr_high"),
                      ("bics_levy_exemption_2027", "levy_exemption_bics_2027"))

#: Task Y13: the 2050 list. The procurement uncertainties carry over unchanged; the
#: 2030-named items are replaced by declared 2050 equivalents (their forecast-summary
#: keys come from the 2050 bill file's own sensitivities) or are not applicable.
UNCERTAINTIES_2050 = {
    "forward_premium_minus_3pct": UNCERTAINTIES["forward_premium_minus_3pct"],
    "forward_premium_plus_5pct": UNCERTAINTIES["forward_premium_plus_5pct"],
    "forward_premium_stress_plus_10pct": UNCERTAINTIES["forward_premium_stress_plus_10pct"],
    "shaping_low_1pct": UNCERTAINTIES["shaping_low_1pct"],
    "shaping_high_7pct": UNCERTAINTIES["shaping_high_7pct"],
    "capacity_market_2050_clearing_low": ("capacity_market", "Capacity Market 2050 at the low end of the modelled "
                                          "clearing range"),
    "capacity_market_2050_clearing_high": ("capacity_market", "Capacity Market 2050 at the high end of the modelled "
                                           "clearing range"),
    "levy_exemption_electrolysers_2050": ("cfd", "Electrolysers exempt from the supplier levies (CfD, CCS, Capacity "
                                          "Market) in 2050"),
}
_LEVY_SOURCES_2050 = (("capacity_market_2050_clearing_low", "capacity_market_clearing_low"),
                      ("capacity_market_2050_clearing_high", "capacity_market_clearing_high"),
                      ("levy_exemption_electrolysers_2050", "levy_exemption_electrolysers_2050"))
#: 2030 items with no 2050 counterpart, and why (reported in the ledger summary).
UNCERTAINTIES_NOT_APPLICABLE_2050 = {
    "ro_exchequer_relief_continued": "the Renewables Obligation closed on 31 March 2037, so there is no relief to "
                                     "continue in 2050",
    "capacity_market_t4_at_2029_30_clearing": "replaced by capacity_market_2050_clearing_low/high: no agreement "
                                              "signed by 2030 reaches 2050",
    "capacity_market_t4_storage_efc_low": "replaced by capacity_market_2050_clearing_low/high",
    "capacity_market_t4_ldes_excluded": "replaced by capacity_market_2050_clearing_low/high",
    "capacity_market_t4_dsr_low": "replaced by capacity_market_2050_clearing_low/high",
    "capacity_market_t4_dsr_high": "replaced by capacity_market_2050_clearing_low/high",
    "bics_levy_exemption_2027": "BICS is a 2027 scheme with no stated term; the 2050 exemption question is the "
                                "electrolysers' 88.8 TWh (levy_exemption_electrolysers_2050)",
}
#: The first model year whose bill this list fits: the RO has closed (support ended 31 March 2037) and the 2030
#: Capacity Market cases no longer apply (build_fes_bill_inputs refuses earlier years).
VINTAGED_UNCERTAINTIES_FROM = 2038


def vintaged_uncertainties(year: int) -> tuple[dict, tuple, dict]:
    """The 2050 list's structure on model year ``year``'s own keys (FES 2040, 11 Nov 2026; any long-range year).

    The keys follow the year's bill: ``capacity_market_<Y>_clearing_low/high`` and
    ``levy_exemption_electrolysers_<Y>`` (the levy sensitivity ``electrolysers_<Y>`` of
    build_fes_bill_inputs). 2050 keeps its committed list (:data:`UNCERTAINTIES_2050`).
    """
    y = int(year)
    listed = {
        **{k: UNCERTAINTIES[k] for k in ("forward_premium_minus_3pct", "forward_premium_plus_5pct",
                                         "forward_premium_stress_plus_10pct", "shaping_low_1pct", "shaping_high_7pct")},
        f"capacity_market_{y}_clearing_low": ("capacity_market", f"Capacity Market {y} at the low end of the modelled "
                                              "clearing range"),
        f"capacity_market_{y}_clearing_high": ("capacity_market", f"Capacity Market {y} at the high end of the modelled "
                                               "clearing range"),
        f"levy_exemption_electrolysers_{y}": ("cfd", "Electrolysers exempt from the supplier levies (CfD, CCS, Capacity "
                                              f"Market) in {y}"),
    }
    sources = ((f"capacity_market_{y}_clearing_low", "capacity_market_clearing_low"),
               (f"capacity_market_{y}_clearing_high", "capacity_market_clearing_high"),
               (f"levy_exemption_electrolysers_{y}", f"levy_exemption_electrolysers_{y}"))
    replaced = f"replaced by capacity_market_{y}_clearing_low/high"
    not_applicable = {
        "ro_exchequer_relief_continued": ("the Renewables Obligation closed on 31 March 2037, so there is no relief to "
                                          f"continue in {y}"),
        "capacity_market_t4_at_2029_30_clearing": replaced,
        "capacity_market_t4_storage_efc_low": replaced,
        "capacity_market_t4_ldes_excluded": replaced,
        "capacity_market_t4_dsr_low": replaced,
        "capacity_market_t4_dsr_high": replaced,
        "bics_levy_exemption_2027": (f"BICS is a 2027 scheme with no stated term; the {y} exemption question is the "
                                     f"electrolysers' demand (levy_exemption_electrolysers_{y})"),
    }
    return listed, sources, not_applicable


#: Committed uncertainty lists by model year (2050's text predates the generic list).
UNCERTAINTIES_BY_YEAR = {2050: (UNCERTAINTIES_2050, _LEVY_SOURCES_2050, UNCERTAINTIES_NOT_APPLICABLE_2050)}


def uncertainty_list(model_year: int | None) -> tuple[dict, tuple, dict]:
    """``(uncertainties, levy sources, not applicable)`` for a model year (task Y13; default the 2030 list)."""
    if model_year is not None and int(model_year) in UNCERTAINTIES_BY_YEAR:
        return UNCERTAINTIES_BY_YEAR[int(model_year)]
    if model_year is not None and int(model_year) >= VINTAGED_UNCERTAINTIES_FROM:
        return vintaged_uncertainties(int(model_year))
    return UNCERTAINTIES, _LEVY_SOURCES_2030, {}


def uncertainty_values(forecast_summary: Mapping) -> dict[str, float]:
    """Each uncertainty's change to the private bill, GBP per household ex VAT (from the forecast summary).

    The list is the forecast bill's own year's (``modelled_year``; task Y13).
    """
    d = (forecast_summary or {}).get("diagnostics") or {}
    wp = ((d.get("wholesale_procurement") or {}).get("sensitivities_gbp_per_3100kwh_ex_vat") or {})
    levy = d.get("levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat") or {}
    relief = d.get("ro_relief_case_values_gbp_per_3100kwh") or {}
    listed, levy_sources, _ = uncertainty_list((forecast_summary or {}).get("modelled_year"))
    out = {}
    if ("ro_exchequer_relief_continued" in listed
            and "relief_extended" in relief and "relief_ends_march_2029" in relief):
        out["ro_exchequer_relief_continued"] = float(relief["relief_extended"]) - float(relief["relief_ends_march_2029"])
    for key, source in (("forward_premium_minus_3pct", ("forward_premium", "minus_3pct")),
                        ("forward_premium_plus_5pct", ("forward_premium", "plus_5pct")),
                        ("forward_premium_stress_plus_10pct", ("forward_premium", "stress_plus_10pct")),
                        ("shaping_low_1pct", ("shaping", "low_1pct")), ("shaping_high_7pct", ("shaping", "high_7pct"))):
        value = (wp.get(source[0]) or {}).get(source[1])
        if value is not None and key in listed:
            out[key] = float(value)
    for key, source in levy_sources:
        if levy.get(source) is not None:
            out[key] = float(levy[source])
    return out


#: Round 4 (3 Oct 2026): sensitivities a tariff case carries on its own recovery (2050 only).
TARIFF_OWN_EXEMPTION = ("levy_exemption_electrolysers_2050",)


def tariff_policy_ratio(report: Mapping) -> float:
    """The tariff's cost per pound of the private lines it replaces (household, ex VAT): new lines over replaced."""
    bill = (report or {}).get("bill") or {}
    replaced = float(bill.get("replaced_lines_total_gbp_per_3100kwh_ex_vat") or 0.0)
    new = float(bill.get("new_lines_total_gbp_per_3100kwh_ex_vat") or 0.0)
    return new / replaced if replaced > 0 else 0.0


def uncertainties_by_case(rows: list[dict], forecast_summary: Mapping | None, tariff: Mapping | None,
                          vat_rate: float) -> dict[str, dict]:
    """Per case, each uncertainty's effect on private's bill, on the case's bill and on the gap (inc VAT).

    A market case carries every line private does, so its bill moves with private's
    and its gap does not (the Capacity Market booking is capped at the line, a
    second-order effect).  The acquisition tariff replaces the RO, the hedging and
    shaping lines and the Capacity Market (bar the agreements it does not acquire),
    so for a tariff case those uncertainties move the gap: continued RO relief
    shrinks P3's saving by the whole relief.
    """
    from scripts.public_ownership.nationalised_tariff import REPLACED_LINES

    values = uncertainty_values(forecast_summary or {})
    listed = uncertainty_list((forecast_summary or {}).get("modelled_year"))[0]
    tariff_cases = (tariff or {}).get("cases") or {}
    out = {}
    for row in rows:
        case_id = row["case_id"]
        report = tariff_cases.get(row.get("institution_twin") or case_id)
        kept_cm = float(((report or {}).get("diagnostics") or {}).get("capacity_market_kept_share", 0.0) or 0.0)
        entry = {}
        for key, value in values.items():
            line, label = listed[key]
            private_change = value * (1.0 + vat_rate)
            if report is None:
                case_change = private_change
            elif line == "capacity_market":
                case_change = private_change * kept_cm
            elif key in TARIFF_OWN_EXEMPTION and line in REPLACED_LINES:
                # Round 4 (3 Oct 2026): the tariff has an exemption of its own. It recovers its cost of the
                # replaced policy lines over final demand like those lines; with electrolysers out of that base,
                # each household pays more by the private effect scaled by the tariff's cost per pound of the
                # lines it replaces (new lines over replaced lines, like for like).
                case_change = private_change * tariff_policy_ratio(report)
            elif line in REPLACED_LINES:
                case_change = 0.0
            else:
                case_change = private_change
            entry[key] = {"label": label, "line": line, "private_bill_change_gbp_inc_vat": private_change,
                          "case_bill_change_gbp_inc_vat": case_change,
                          "gap_change_gbp_inc_vat": case_change - private_change}
        out[case_id] = entry
    return out


def hpc_schedule_by_case(rows: list[dict], *, register: pd.DataFrame, tariff: Mapping | None,
                         forecast_summary: Mapping, conv: Mapping, schedule: Mapping,
                         prices: Mapping[str, float], valuation: pd.DataFrame | None) -> dict:
    """Bill-level sensitivity 'HPC on EDF's current schedule' for every case (:mod:`hpc_schedule`).

    A background sensitivity: private and every scenario alike, central and range
    (EDF's unfavourable scenario and an early start).  A market case carries HPC on
    its CfD, so its bill moves exactly as private's; a tariff case that owns HPC
    moves on its own lines, so the gap moves.  Household totals move with the bill
    (neither the CfD nor the tariff touches the Exchequer).
    """
    from scripts.public_ownership import hpc_schedule as hs

    register = CaseRows.of(register)
    hpc = hs.hpc_inputs(register.rows("private"))
    d = (forecast_summary or {}).get("diagnostics") or {}
    settled = float(d.get("nuclear_cfd_settled_twh") or 0.0) * 1e6
    if not hpc or not settled:
        return {}
    payment = float(d.get("nuclear_cfd_settled_payment_gbp") or 0.0)
    strike = next((float(r["strike_price_gbp_per_mwh"]) for r in (forecast_summary.get("strike_table") or [])
                   if r.get("cfd_round") == "HPC" and r.get("technology") == "nuclear"), None)
    if strike is None:
        raise ValueError("the forecast bill's strike table has no HPC row")
    from scripts.public_ownership import bill_recovery

    vat = float(conv["vat_rate"])
    # F19 (7 Oct 2026): on the two-volume basis the CfD and wholesale rows convert through the
    # forecast bill's own entries; a D1 bill keeps its CfD denominator and final demand.
    two_volume = bill_recovery.is_two_volume(forecast_summary)
    line_rates = ({"cfd": bill_recovery.household_per_system(forecast_summary, "cfd"),
                   "wholesale": bill_recovery.household_per_system(forecast_summary, "direct_energy")}
                  if two_volume else None)
    eligible = (None if two_volume else float(
        bill_recovery.d1_denominators(forecast_summary, from_diagnostics=True)["cfd_eligible_mwh"]
        or conv["final_demand_mwh"]))
    first_power = {"central": None, "low": schedule["range"]["low"]["unit_1_first_power"],
                   "high": schedule["range"]["high"]["unit_1_first_power"]}
    shortfalls = {k: hs.schedule_shortfall(schedule, settled, prices, v) for k, v in first_power.items()}
    # Item Y4: the output key names the model year the prices are for.
    hpc_year_key = f"hpc_{hs.prices_model_year(prices)}_twh"
    markets = {k: hs.market_bill_change(s, cfd_payment_gbp=payment, settled_mwh=settled, strike_gbp_per_mwh=strike,
                                        conv=conv, eligible_mwh=eligible, line_rates=line_rates)
               for k, s in shortfalls.items()}
    private_inc = {k: (m["cfd_line_gbp_per_3100kwh_ex_vat"] + m["wholesale_line_gbp_per_3100kwh_ex_vat"]) * (1 + vat)
                   for k, m in markets.items()}
    hpc_row = (valuation.loc[valuation["asset_id"].astype(str).str.startswith(hs.HPC_ASSET_PREFIX)]
               if valuation is not None and "asset_id" in valuation.columns else pd.DataFrame())
    tariff_cases = (tariff or {}).get("cases") or {}
    out = {}
    for row in rows:
        case_id = row["case_id"]
        report = tariff_cases.get(row.get("institution_twin") or case_id)
        state_owns = register.memo(*_hpc_request(row.get("institution_twin") or case_id))
        entry = {}
        for key, shortfall in shortfalls.items():
            if report is not None and state_owns and len(hpc_row):
                levers = report.get("levers") or {}
                basis = levers.get("purchase_price_basis")
                rate = (None if levers.get("in_construction") == "public_completion"
                        else float(hpc_row["wacc"].iloc[0]) if basis == "market_value"
                        else float(hpc_row["public_real_rate"].iloc[0]) if basis == "state_present_value" else None)
                annuity = float(((report.get("break_even_purchase_price") or {}).get("annuity_factor")) or 0.0)
                # F19: the tariff's own pool rate (its fuel and annuity lines) on the two-volume basis.
                pool = (float((report.get("allocation") or {}).get("pool_rate") or 0.0) or None) if two_volume else None
                t = hs.tariff_change(shortfall, variable_cost_gbp_per_mwh=hpc["variable_cost_gbp_per_mwh"],
                                     annual_flow_gbp=float(hpc_row["annual_flow_model_year_gbp"].iloc[0]),
                                     settled_mwh=settled, basis_rate=rate, public_rate=0.0, term_years=1.0, conv=conv,
                                     line_rate=pool)
                t["annuity_system_gbp"] = t["purchase_price_change_gbp"] * annuity
                t["annuity_gbp_per_3100kwh_ex_vat"] = (t["annuity_system_gbp"] * pool if pool else
                                                        t["annuity_system_gbp"] * float(conv["factor"])
                                                        / float(conv["final_demand_mwh"]) * float(conv["mwh"]))
                case_inc = (t["energy_gbp_per_3100kwh_ex_vat"] + t["annuity_gbp_per_3100kwh_ex_vat"]) * (1 + vat)
                lines = {"tariff_replacement_energy_less_hpc_variable_cost": t["energy_gbp_per_3100kwh_ex_vat"],
                         "tariff_hpc_purchase_annuity": t["annuity_gbp_per_3100kwh_ex_vat"],
                         "hpc_purchase_price_change_gbp": t["purchase_price_change_gbp"]}
            else:
                m = markets[key]
                case_inc = private_inc[key]
                lines = {"cfd_hpc_settlement": m["cfd_line_gbp_per_3100kwh_ex_vat"],
                         "wholesale_replacement_energy": m["wholesale_line_gbp_per_3100kwh_ex_vat"]}
            entry[key] = {hpc_year_key: shortfall["edf_mwh"] / 1e6, "missing_twh": shortfall["missing_mwh"] / 1e6,
                          "replacement_price_gbp_per_mwh": shortfall["replacement_price_gbp_per_mwh"],
                          "lines_gbp_per_3100kwh_ex_vat": lines,
                          "private_bill_change_gbp_inc_vat": private_inc[key],
                          "case_bill_change_gbp_inc_vat": case_inc,
                          "household_total_change_gbp_inc_vat": case_inc,
                          "gap_change_gbp_inc_vat": case_inc - private_inc[key]}
        out[case_id] = entry
    return {
        "label": "HPC on EDF's current schedule (bill-level sensitivity, no re-solve)",
        "cp30_assumption": schedule["cp30_physical_assumption"],
        "edf_schedule": schedule["edf_schedule"], "range": schedule["range"],
        "settled_twh_cp30": settled / 1e6, "strike_gbp_per_mwh_model_prices": strike,
        "cfd_top_up_gbp_per_mwh": markets["central"]["cfd_top_up_gbp_per_mwh"],
        "cfd_reference_gbp_per_mwh": markets["central"]["cfd_reference_gbp_per_mwh"],
        "hpc_variable_cost_gbp_per_mwh": hpc["variable_cost_gbp_per_mwh"],
        "approximation": ("dispatch, emissions and constraint effects are not re-solved; the missing energy is priced at "
                          "the market-year Stage 1 system price over the months it is missing"),
        "by_case": out,
    }


def party_flow_rows(rows: list[dict], cases: Mapping[str, Mapping], parties: Mapping[str, Mapping], *,
                    register: pd.DataFrame, bill: Mapping, account: Mapping, balancing: Mapping | None,
                    balancing_accounts: Mapping | None, investor: Mapping, carbon: Mapping,
                    network_lines: pd.DataFrame, conv: Mapping, networks: Mapping[str, Mapping],
                    step_parts: Mapping[str, Mapping]) -> tuple[list[dict], list[dict]]:
    """The itemised flows for every case and their closure checks (:mod:`party_flows`)."""
    from scripts.public_ownership import party_flows as pf

    ph = lambda gbp: per_household(gbp, conv)  # noqa: E731
    all_rows, checks = [], []
    bill_cases = bill.get("cases") or {}
    for row in rows:
        case_id = row["case_id"]
        entry = cases[case_id]
        stage = entry["stage"]
        levers = dict(entry.get("levers_for_flows") or {})
        flows = pf.Flows(case_id, ph)
        if stage == "market_bill":
            pf.market_flows(flows, parties=parties[case_id],
                            components=CaseRows.of(register).memo(*_components_request(case_id)),
                            account_case=(account.get("cases") or {}).get(case_id), network=networks[case_id],
                            levers=levers)
        elif stage == "balancing_bids":
            group = (investor.get("balancing") or {}).get(case_id) or {}
            own_account = ((balancing_accounts or {}).get(case_id) or account).get("cases") or {}
            run_group = (((balancing or {}).get("cases") or {}).get(case_id) or {}).get("run_group")
            pf.dispatch_step_flows(flows, step_parts[case_id]["dispatch"],
                                   float(balancing_carbon(carbon, run_group)["carbon_receipts_gbp"])
                                   - float(carbon["market"]["carbon_receipts_gbp"]))
            pf.market_flows(flows, parties=step_parts[case_id]["ownership"],
                            components=pf.register_components(CaseRows.of(group["register"]).rows(case_id, "private"),
                                                              case_id),
                            account_case=own_account.get(case_id), network=networks[case_id], levers=levers)
        elif stage == "acquisition_tariff":
            pf.tariff_flows(flows, parties=parties[case_id], network=networks[case_id], levers=levers)
        elif stage == "institution_comparison":
            twin = entry["institution_twin"]
            pf.tariff_flows(flows, parties=step_parts[case_id]["twin"], network=networks[twin],
                            levers=dict(cases[twin].get("levers_for_flows") or {}))
            pf.institution_step_flows(flows, step_parts[case_id]["step"],
                                      float(carbon["central"]["carbon_receipts_gbp"])
                                      - float(carbon["market"]["carbon_receipts_gbp"]))
        pf.vat_flow(flows, float(row["vat_change_vs_private_gbp"]))
        checks.append(pf.check(flows.rows, row))
        all_rows.extend(flows.rows)
    return all_rows, checks


HOUSEHOLD_COLUMNS = [
    "case_id", "headline_label", "stage", "case_kind", "case_parent", "case_preset", "case_bound", "grid_toggles",
    "grid_memberships", "comparator_case_id", "bill_inc_vat", "bill_delta_vs_private_inc_vat",
    "bill_delta_vs_comparator_gbp_inc_vat", "taxpayer_share_gbp_per_household", "household_total_gbp_inc_vat",
    "household_total_delta_vs_private_gbp_inc_vat", "household_total_delta_vs_comparator_gbp_inc_vat",
    "taxpayer_share_legal_incidence_gbp_per_household", "household_total_legal_incidence_gbp_inc_vat",
    "household_total_legal_incidence_delta_vs_private_gbp_inc_vat", "household_total_worst_with_contested_gbp_inc_vat",
    "contested_gilt_stock_externality_gbp_per_household", "contested_mcpf_deadweight_gbp_per_household",
    "break_even_purchase_price_bill_gbp", "break_even_purchase_price_household_total_gbp",
    # F19 P3 decision 5: the two above are the lifetime price; the 2030 snapshots beside them.
    "break_even_purchase_price_bill_2030_snapshot_gbp", "break_even_purchase_price_household_total_2030_snapshot_gbp",
    "break_even_purchase_price_basis",
    "consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp", "investor_change_vs_private_gbp",
    "resource_saving_vs_private_gbp", "resource_saving_gap_vs_parent_gbp",
    # F11: the Capacity Market recovered flat (b) or from taxation (c), beside the headline only.
    "cm_household_line_gbp_inc_vat", "cm_option_b_flat_bill_change_gbp_inc_vat", "cm_option_b_flat_bill_gbp_inc_vat",
    "cm_option_c_taxation_bill_gbp_inc_vat", "cm_option_c_taxation_exchequer_cost_gbp",
    "cm_option_c_taxpayer_share_gbp_per_household", "cm_option_c_household_total_gbp_inc_vat",
    # User decision (24 Oct 2026): a same-fleet P3 case on central dispatch, an additive approximation.
    *(I3_APPROX_PREFIX + c for c in I3_APPROX_COLUMNS),
    I3_APPROX_PREFIX + "step_bill_gbp_inc_vat", I3_APPROX_PREFIX + "step_household_total_gbp_inc_vat",
    I3_APPROX_PREFIX + "basis", I3_APPROX_PREFIX + "institution_case_id",
    # User, 13 Nov 2026: the electrified household (scripts/public_ownership/electrified_household.py).
    *ELECTRIFIED_COLUMNS,
]


#: The three-party columns (against private) restated against the comparator in a vintaged year (decision 54).
THREE_PARTY_COLUMNS = ("consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp",
                       "investor_change_vs_private_gbp", "resource_saving_vs_private_gbp")


def _vintaged_forecast(forecast_summary: Mapping | None) -> bool:
    """Whether the forecast bill is a vintaged (2040 on) year's: the comparator columns are added there only."""
    from scripts.public_ownership.model_year import is_vintaged_model_year

    year = (forecast_summary or {}).get("modelled_year")
    return year is not None and is_vintaged_model_year(int(year))


def household_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The per-case household-total table (user decisions 4 and D): bill and household total side by side."""
    return frame[[c for c in HOUSEHOLD_COLUMNS if c in frame.columns]]


def consumer_flexibility_memo(assumptions: Mapping | None) -> dict | None:
    """FES 2050 decision 30: the Capacity Market paid to consumer-owned V2G and DSR, as a memo (None in 2030).

    V2G (decision 24) and residential DSR are owned by households, outside every ownership
    case: bill payers pay them through the Capacity Market line and the households that own
    them receive it.  Both legs are consumers', and the payment is the same in every case
    (an acquisition tariff keeps it on the bill as capacity the state does not buy), so it
    is zero in every column of the three-party identity against private.
    """
    derivation = ((((assumptions or {}).get("sourced_externals") or {}).get("capacity_market") or {})
                  .get("derivation") or {})
    payees = derivation.get("payments_by_payee_gbp")
    if not payees:
        return None
    households = float(payees.get("v2g_owners_households", 0.0)) + float(payees.get("dsr_residential", 0.0))
    return {
        "price_year": 2025,
        "agreement_cost_gbp": float(derivation.get("agreement_cost_gbp", 0.0) or 0.0),
        "payments_by_payee_gbp": {k: float(v) for k, v in payees.items()},
        "consumers_pay_gbp": households, "consumers_receive_gbp": households,
        "change_vs_private_gbp": {"consumers": 0.0, "exchequer": 0.0, "investors": 0.0, "resource": 0.0},
        "reading": ("Consumer pays and consumer receives: bill payers pay the households that own V2G (and "
                    "residential DSR) through the Capacity Market line. The payment is the same in every case, so "
                    "it nets to zero in each case's change against private; industrial DSR is paid to businesses, "
                    "and the rest of the line to the register's plant and interconnectors."),
    }


#: F20 phase 0: the passes timed when PYPSA_GB_STAGE_TIMING is set (scripts/public_ownership/stage_timing.py);
#: each of the per-case passes over CaseRows is its own label.
TIMED_FUNCTIONS = ("load_register", "build", "corporation_tax_forgone", "upstream_corporation_tax",
                   "three_party_columns", "investor_inputs", "balancing_group_accounts", "household_totals",
                   "state_borrowing_gbp", "check_resource_invariant", "uncertainties_by_case",
                   "hpc_schedule_by_case", "party_flow_rows", "_register_requests", "state_borrowing_register_part")


def main(argv=None) -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("register", "bill_summary", "public_account", "network_account", "assumptions", "recovery",
                 "snapshot", "out_csv", "out_json"):
        parser.add_argument(name)
    parser.add_argument("--marginal-costs", required=True, help="Build-time marginal-cost breakdown (carbon rates)")
    parser.add_argument("--market-scenarios", nargs="+", required=True)
    parser.add_argument("--tariff-summary", default=None)
    parser.add_argument("--institution-summary", default=None)
    parser.add_argument("--central-scenarios", nargs="*", default=None)
    parser.add_argument("--balancing-summary", default=None)
    parser.add_argument("--balancing-scenarios", nargs="*", default=None)
    parser.add_argument("--guarantee-cost", default=None,
                        help="network_guarantee_expected_loss.csv: the expected-loss rate at each guarantee_cost setting")
    parser.add_argument("--financing-summary", default=None,
                        help="Financing summary CSV (the financing saving and its tax-transfer split per case)")
    parser.add_argument("--config", default=None, help="the config directory (CHP heat credit, the case catalogue)")
    parser.add_argument("--tax-incidence", default=None,
                        help="household_tax_incidence.csv: the legal-incidence key (household share of tax, UK households)")
    parser.add_argument("--contested-costs", default=None,
                        help="contested_fiscal_costs.csv: the gilt-stock externality and the MCPF (worst cases only)")
    parser.add_argument("--forecast-summary", default=None,
                        help="the forecast bill summary: the cross-cutting uncertainties per scenario")
    parser.add_argument("--hpc-schedule", default=None,
                        help="hpc_schedule_edf_2025.json: the bill-level sensitivity 'HPC on EDF's current schedule'")
    parser.add_argument("--market-prices", nargs="*", default=None,
                        help="the twelve market months' Stage 1 *_wholesale_price.csv, January first")
    parser.add_argument("--party-flows", default=None, help="output: the itemised flows per case (CSV)")
    parser.add_argument("--household-totals", default=None, help="output: bill and household total per case (CSV)")
    parser.add_argument("--fallback", action="store_true",
                        help="Compute missing upstream fields (corporation tax per row, state deficits) here")
    parser.add_argument("--run-group", default=None,
                        help="Task Y14: the run group whose case catalogue the summary carries (default: read off "
                             "the register's file name; a group with no case list of its own takes the global one)")
    args = parser.parse_args(argv)
    # F20 phase 0: opt-in stage timing (PYPSA_GB_STAGE_TIMING); without it nothing is wrapped.
    from scripts.public_ownership import nationalised_institutions as _institutions, stage_timing
    from scripts.public_ownership import party_flows as _party_flows

    stage_timing.instrument_register_io()
    stage_timing.instrument(globals(), TIMED_FUNCTIONS, prefix="ledger")
    stage_timing.instrument(investors, ("register_position", "market_case_parties", "tariff_case_parties",
                                        "legacy_acquisition_memo"), prefix="investor_ledger")
    stage_timing.instrument(_party_flows, ("register_components",), prefix="party_flows")
    stage_timing.instrument(_institutions, ("measure_dispatch",), prefix="nationalised_institutions")
    lap = stage_timing.laps("ledger.main")

    from scripts.public_ownership.nationalised_institutions import (
        HEAT_CREDIT_DEFAULTS, config_heat_credit_settings, measure_dispatch,
    )

    read = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p else None  # noqa: E731
    bill, account = read(args.bill_summary), read(args.public_account)
    tariff, institution, balancing = read(args.tariff_summary), read(args.institution_summary), read(args.balancing_summary)
    assumptions, recovery = read(args.assumptions), read(args.recovery)
    conv = conversion(assumptions, recovery, bill)
    snapshot = yaml.safe_load(Path(args.snapshot).read_text(encoding="utf-8"))
    heat = (config_heat_credit_settings(args.config)
            if args.config else HEAT_CREDIT_DEFAULTS)
    rates = pd.read_csv(args.marginal_costs).drop_duplicates("generator").set_index("generator")
    carbon = {"market": carbon_receipts(measure_dispatch(args.market_scenarios, rates, heat))}
    if args.balancing_scenarios:
        balancing_groups = sorted({str(m.get("run_group")) for m in ((balancing or {}).get("cases") or {}).values()
                                   if m.get("run_group")})
        if len(balancing_groups) > 1:
            # FES 2050 decision 53: each state-bids case's carbon is measured on its own group's
            # months (the four P2 share groups solved 48 months; summing them counted carbon 4x).
            if not args.config:
                raise ValueError("several balancing groups need --config (each group's scenario pattern)")
            from scripts.public_ownership.reference_runs import configured_group

            po_block = (read_config(args.config) or {}).get("public_ownership") or {}
            groups_cfg = po_block.get("reference_run_groups") or {}
            for g in balancing_groups:
                configured_group(po_block, g)          # refuses a study group handed the defaults
                if g not in groups_cfg:
                    raise ValueError(f"balancing group {g} is not in {args.config}")
            patterns = {g: groups_cfg[g]["scenario_pattern"] for g in balancing_groups}
            split = balancing_scenarios_by_group(args.balancing_scenarios, patterns)
            carbon["balancing_by_group"] = {g: carbon_receipts(measure_dispatch(months, rates, heat))
                                            for g, months in split.items()}
        else:
            carbon["balancing"] = carbon_receipts(measure_dispatch(args.balancing_scenarios, rates, heat))
    if args.central_scenarios:
        carbon["central"] = carbon_receipts(measure_dispatch(args.central_scenarios, rates, heat))
    cases = list((bill.get("cases") or {})) + list(((tariff or {}).get("cases") or {})) + list(
        ((balancing or {}).get("cases") or {})) + [str(m.get("base_case")) for m in ((balancing or {}).get("cases") or {}).values()]
    register = load_register(args.register, cases)
    network_json = Path(args.network_account).with_suffix(".json")
    # F29 (26 Oct 2026): networks: debt_guarantee is retired, so no case carries the guarantee's expected
    # loss; --guarantee-cost is accepted and not read.
    guarantee = None
    # Review 29 Sep 2026: the private-investor column and the three-party identity, always checked here.
    investor = investor_inputs(register_path=args.register, bill_summary_path=args.bill_summary,
                               network_account_path=args.network_account,
                               assumptions=assumptions, recovery=recovery, bill=bill, balancing=balancing,
                               tariff=tariff, balancing_summary_path=args.balancing_summary)
    catalogue = {}
    if args.config:
        # Decisions 68 and 69: the co-investment guarantee's genuine share and the risk-adjusted view
        # (public_ownership.co_ownership_finance, public_ownership.risk_adjusted_view; both absent in 2030).
        po = (read_config(args.config) or {}).get("public_ownership") or {}
        from scripts.public_ownership.calculate_financing_scenarios import run_group_from_register_path as _rg

        own = ((po.get("reference_run_groups") or {}).get(args.run_group or _rg(args.register)) or {})
        # Decision 68: a register bound's own co-ownership financing over the study's.
        investor.update(ledger_options({**po, **({"co_ownership_finance": own["co_ownership_finance"]}
                                                  if own.get("co_ownership_finance") else {})}))
    if args.config:
        from scripts.public_ownership.calculate_financing_scenarios import run_group_from_register_path
        from scripts.public_ownership.reference_runs import configured_case_catalogue, configured_group

        block = (read_config(args.config) or {}).get("public_ownership") or {}
        # Task Y14 (patch 4): a group with its own case list (the 2050 groups) catalogues that list;
        # every 2030 group, and a register named for no configured group, the global list as before.
        group = configured_group(block, args.run_group or run_group_from_register_path(args.register))
        # FES 2050 guard (user, 3 Oct 2026): a later year's group handed the defaults fails loudly.
        from scripts.public_ownership.calculate_financing_scenarios import require_study_config

        require_study_config(block, group, _forecast_model_year(args.forecast_summary))
        catalogue = configured_case_catalogue(block, group)
        check_institution_fleet_ratio(institution, block, group)
    lap("inputs")
    frame, summary = build(register=register, bill=bill, account=account,
                           network_lines=pd.read_csv(args.network_account), conv=conv, wedges=tax_wedges(snapshot),
                           carbon=carbon, tariff=tariff, institution=institution, balancing=balancing,
                           fallback=args.fallback,
                           financing=pd.read_csv(args.financing_summary) if args.financing_summary else None,
                           balancing_accounts=balancing_group_accounts(balancing, args.public_account),
                           investor=investor, guarantee=guarantee, catalogue=catalogue,
                           incidence=load_tax_incidence(args.tax_incidence) if args.tax_incidence else None,
                           contested=load_contested_costs(args.contested_costs) if args.contested_costs else None,
                           forecast_summary=read(args.forecast_summary),
                           electrified_basis=(archetype_basis(read(args.forecast_summary), assumptions)
                                              if args.forecast_summary else None),
                           electrified_high_basis=(archetype_basis(read(args.forecast_summary), assumptions,
                                                                   key="high_demand_archetype")
                                                   if args.forecast_summary else None),
                           hpc_schedule=_hpc_schedule_for_year(assumptions, args.hpc_schedule, args.market_prices,
                                                               model_year=_forecast_model_year(args.forecast_summary)))
    lap("build")
    flow_rows = summary.pop("_party_flow_rows", [])
    summary["sources"] = {k: v for k, v in vars(args).items() if k not in ("out_csv", "out_json")}
    # FES 2050 decision 30: who the Capacity Market line pays (absent from every 2030 assumptions file).
    memo = consumer_flexibility_memo(assumptions)
    if memo is not None:
        summary["capacity_market_consumer_flexibility"] = memo
    for path in (args.out_csv, args.out_json):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    summary["party_flows"]["flows_by_case"] = {}
    for flow in flow_rows:
        summary["party_flows"]["flows_by_case"].setdefault(flow["case_id"], []).append(
            {k: flow[k] for k in ("flow_id", "label", "payer", "payee", "gbp_m", "gbp_per_household", "category")})
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    if args.party_flows:
        from scripts.public_ownership.party_flows import frame as flow_frame

        flow_frame(flow_rows).to_csv(args.party_flows, index=False)
    if args.household_totals:
        household_frame(frame).to_csv(args.household_totals, index=False)
    lap("write")
    show = frame.loc[frame["headline_label"].notna(), ["case_id", "bill_inc_vat", "exchequer_net_vs_private_gbp",
                                                        "net_of_exchequer_gbp_inc_vat"]]
    print(show.round(2).to_string(index=False))
    print(investors.headline_table([summary["cases"][c] for c in show["case_id"]]).round(1).to_string(index=False))


def _forecast_model_year(forecast_summary_path) -> int | None:
    """The forecast bill's ``modelled_year`` (item Y4), or None without a forecast summary."""
    if not forecast_summary_path or not Path(forecast_summary_path).exists():
        return None
    year = json.loads(Path(forecast_summary_path).read_text(encoding="utf-8")).get("modelled_year")
    return None if year is None else int(year)


def hpc_schedule_switch(assumptions: Mapping | None) -> tuple[bool, str | None]:
    """Whether the bill's year runs the HPC-schedule sensitivity (task Y13), and why not.

    ``exchequer_ledger.hpc_schedule_sensitivity`` in the forecast bill's assumptions file:
    ``{"enabled": false, "reason": ...}`` switches it off (the 2050 bill: both HPC units run
    long before 2050, so EDF's start dates move nothing). Absent, it is on, as in 2030.

    Args:
        assumptions: the forecast bill's assumptions file (the ledger's ``assumptions`` input).

    Returns:
        ``(enabled, reason)``; the reason is None when enabled.
    """
    block = ((assumptions or {}).get("exchequer_ledger") or {}).get("hpc_schedule_sensitivity") or {}
    if block.get("enabled", True) is not False:
        return True, None
    if not block.get("reason"):
        raise ValueError("exchequer_ledger.hpc_schedule_sensitivity.enabled is false without a reason")
    return False, str(block["reason"])


def _hpc_schedule_for_year(assumptions, schedule_path, price_paths, model_year: int | None = None) -> dict | None:
    """The HPC schedule inputs, or ``{"skipped": reason}`` when the bill's year switches it off."""
    enabled, reason = hpc_schedule_switch(assumptions)
    if not enabled:
        return {"skipped": reason} if schedule_path else None
    return _hpc_schedule_inputs(schedule_path, price_paths, model_year=model_year)


def _hpc_schedule_inputs(schedule_path, price_paths, model_year: int | None = None) -> dict | None:
    """The HPC schedule file and the twelve months' mean Stage 1 prices, or None when not asked for.

    The months are the model year's (item Y4): the forecast bill's ``modelled_year``,
    else the default year.
    """
    if not schedule_path:
        return None
    from scripts.public_ownership import hpc_schedule as hs

    if not price_paths or len(price_paths) != 12:
        raise ValueError("--hpc-schedule needs --market-prices for the twelve market months, January first")
    months = [str(m) for m in hs.model_year_months(model_year)]
    return {"schedule": hs.load_schedule(schedule_path), "prices": hs.monthly_prices(dict(zip(months, price_paths)))}


if __name__ == "__main__":
    main()
