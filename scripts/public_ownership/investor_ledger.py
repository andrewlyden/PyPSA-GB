"""The private-investor column and the three-party identity (review 29 Sep 2026).

The Exchequer ledger (:mod:`exchequer_ledger`) set each case's bill beside the
Exchequer's position and reported what was left over, unchecked.  This module
measures the third party, private owners, so that for every case against the
private case

    consumer change + Exchequer change + investor change = resource saving

holds as an adding-up identity, checked to GBP 1 at system level.  A transfer
between the parties (an acquisition payment, a CfD re-pricing, the balancing
bids the state stops paying itself, the corporation tax a public owner does not
pay) appears once with each sign and nets out; only the resource saving is left.
System GBP, model price year (2024), change against the private case,
positive when the party is better off.

**Consumers**: the ex-VAT bill change at system level, each moved line at its own
recovery denominator (the amounts the projector and the tariff reconcile their
lines to), plus the change in domestic VAT.  Non-domestic VAT is recovered by
VAT-registered businesses and is no consumer's cost, so VAT nets exactly against
the Exchequer ledger's domestic VAT receipt.  This is not the household headline
times final demand: lines recovered over CfD-eligible sales or with the Capacity
Market's peak factor do not convert flat.

**Exchequer**: the ledger's ``exchequer_net_vs_private_gbp``, unchanged.

**Investors**: the change in private owners' net position, revenue less fuel,
carbon, O&M and the **required return on their capital** at the rate the
register charges them (after the corporation tax embedded in that rate, which the
Exchequer receives), on their share of each row only; plus the payments to
private parties the register does not book, each named (the Capacity Market
agreements, the hedging allowances, the RO line above the register's certificate
income, the interconnectors' GB halves).  An acquisition is measured on a **cash
basis**: the vendors, and the lenders who fund the price, receive the tariff's
acquisition annuity in place of the fleet's 2030 cash.  The premium over the
owners' own valuation and the cost-of-capital gap (the state's valuation less the
owners', both annuitised at the tariff's own factor) are reported beside it, with
the reading that moves the gap from investors to the resource saving
(``*_acquisition_gap_as_resource_gbp``); both readings close.

**Two investor bases (user decision, 29 Sep 2026).**  The headline is the
required-return basis (``investor_change_vs_private_gbp``, repeated as
``investor_change_required_return_basis_gbp``, ``investor_basis`` names it).
Beside it, as a diagnostic, the **observed-return basis**
(``investor_change_observed_return_basis_gbp`` with
``resource_saving_observed_return_basis_gbp``): capital providers, private
lenders and owners and the state's own lenders alike, are credited with the
capital charges actually paid, so every cost-of-capital part of the resource
saving (``cost_of_capital_saving_gbp``, the parts in :data:`CAPITAL_PARTS`)
becomes an investor loss instead.  Both readings close, and both are checked.

**Resource saving**: the fall in what supply really costs: capital at the rate of
whoever finances it net of the corporation tax inside that rate, O&M and the
state's holding company (the register's required revenue), fuel net of carbon
(carbon is a receipt of the Exchequer, so it is a transfer inside the identity),
the networks' cost of capital on the RAV (under a debt guarantee net of the
expected loss the Exchequer now carries), and net payments abroad (link trade,
the foreign halves' rent, the re-trading the balancing mechanism pays foreign
counterparties).  The change in carbon receipts is the value of the emissions
change at the carbon price; it is reported by the Exchequer ledger and is not
part of this saving.

Each case type is measured on its own ledger:

* **market-bill cases** (:func:`market_case_parties`): register producer rent,
  split by ``ownership_share``, and the public account's R1 cut, LDES floor,
  bookings and recycling routes; the bill's moved lines at their reconciled
  system amounts.  Nothing is derived by difference, so the identity is a check
  that the bill, the public account and the register route every pound once.
* **acquisition-tariff cases** (:func:`tariff_case_parties`): the tariff's
  register ledger (item N7) re-derived from its structured fields, every item
  assigned to one party; the tariff's own closure is re-checked here, so a new
  tariff line without an assignment fails loudly.
* **the state-bids case** (item BM1): the P2 bid stack's effect on the private
  fleet (:func:`dispatch_step_parties`), then the ownership effect on the stack
  (a market-bill case of its own run group).  On the dispatch step the bill
  lines that move are compared instrument by instrument with the register
  revenue that moves; each gap is a payment to a party the register does not
  book (balancing counterparties outside the register, the RO buy-out fund) and
  is named.
* **the institution case** (item N10, :func:`institution_step_parties`): its I1
  twin, then the measured I3-less-I1 lines, which are all resource or carbon.

Definitional choices, each declared on the output: the required return (the
register's rate) rather than observed returns; the cash basis on acquisitions;
network acquisition at RAV earns investors their allowed return, so it is a wash
for them (a premium over RAV is theirs, annuitised at the rate consumers pay it);
a debt guarantee's expected loss (lever ``guarantee_cost``) is the Exchequer's
cost and comes out of the resource saving, the rest of the compression staying a
resource saving (user decision, 29 Sep 2026); the hedging and imbalance
allowances a nationalised buyer stops paying go to suppliers (investors), not to
resource; generator TNUoS is a cost of the plant, as the register folds it.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from scripts.public_ownership.nationalised_tariff import (
    HEDGING_LINES, INTERCONNECTOR_PASS_THROUGH_LINE, LINE_BECCS_CAPTURE, LINE_CANCELLED_RESIDUAL, LINE_DPA_RECEIPTS,
    LINE_CM_NOT_ACQUIRED, LINE_FORECAST_ERROR, LINE_IC_ACQUISITION, LINE_IC_FOREIGN_RENT, LINE_IC_RETRADING,
    LINE_IMPORTS, LINE_LEGACY_FIT, LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, LINE_LEGACY_WHD, LINE_NEW_BUILD,
    LINE_REALTIME, LINE_STATE_OVERHEAD, LINE_TAX_EQUIVALENT, LINE_TRANSACTION_COSTS, LINE_TREASURY_MARGIN,
    LINE_VARIABLE_OM, LINE_ANCILLARY_INCOME, LINE_GATE_FEE_INCOME, TOP_UP_LINE, BATTERY_TOP_UP_LINE,
    LINE_SUBHOURLY_INCOME,
)

IDENTITY_TOLERANCE_GBP = 1.0
#: Balancing-bid stacks: the cap-and-floor pass-through the moved congestion rent pays the GB halves (9 Nov 2026).
DISPATCH_PASS_THROUGH = ("interconnector cap-and-floor pass-through on the moved congestion rent (the GB halves' "
                         "owners, outside the register)")
#: Tariff cases: the carbon CfD on BECCS the purchase leaves private, case less private (fix 8 Nov 2026).
PRIVATE_PLANT_CARBON_CFD = ("carbon-removal CfD the Exchequer pays the owners of BECCS the purchase leaves private, "
                            "case less private")
#: Instrument columns the register books into producer rent outside REVENUE_COLUMNS.
RENT_ONLY_INSTRUMENTS = ("battery_revenue_top_up_gbp",)
#: Register columns the investor and resource measurement reads.
REGISTER_COLUMNS = ["producer_rent_gbp", "public_surplus_before_recycling_gbp", "annual_required_revenue_gbp",
                    "annual_capital_recovery_gbp", "market_revenue_gbp", "support_revenue_gbp",
                    "capacity_market_revenue_gbp", "ldes_floor_topup_paid_gbp", "fuel_carbon_gbp",
                    "wholesale_revenue_gbp", "balancing_settlement_gbp", "cfd_support_gbp", "roc_support_gbp",
                    # Review 29 Sep 2026: tax equivalents, BECCS's carbon CfD and the state overhead.
                    "annual_tax_equivalent_gbp", "carbon_removal_cfd_revenue_gbp", "annual_state_overhead_gbp",
                    # F27 item 1: the premium over cost paid for pre-2025 CfD plant bought at its owners' value.
                    "owners_value_premium_gbp", "owners_value_premium_state_adjustment_gbp",
                    # Decision 59 (review 2): a co-owned row's state part on its own cost (2050 registers only).
                    "co_ownership_state_adjustment_gbp", "co_ownership_state_capital_adjustment_gbp",
                    "co_ownership_ldes_state_adjustment_gbp", "co_investment_rate_gain_gbp",
                    # FES 2050 decision 73: the state's share of a co-owned row's CM revenue taken back.
                    "capacity_market_state_part_adjustment_gbp",
                    # The rent-only dispatch instrument (8 Oct 2026). Left unread, register_position saw a zero
                    # top-up and the stack's move reached investors twice (the rent and the line; 14 Oct 2026).
                    *RENT_ONLY_INSTRUMENTS,
                    # F29 (26 Oct 2026): the systematic risk the state holds, at the workbook's social price.
                    "socialised_risk_gbp"]
#: FES 2050 decision 73: the CM design's bill row and its payees outside the register.
LINE_CM_DESIGN = "capacity_market_design_adjustment"
CM_DESIGN_OUTSIDE = ("Capacity Market under the public-ownership design: payments outside the register (I&C DSR, "
                     "interconnectors, plant the register does not carry; decision 73)")
CM_DESIGN_HOUSEHOLDS = ("Capacity Market under the public-ownership design: payments to V2G owners and residential "
                        "DSR (households; decision 73)")
#: 10 Nov 2026: register rows with no rent (F28's small-scale plant, excluded from ownership) still carry CM
#: revenue, which the design moves; it is paid to their (private) owners and no rent carries it.
CM_DESIGN_NO_RENT = ("Capacity Market under the public-ownership design: register rows with no rent (small-scale plant "
                     "outside the ownership register, F28), private owners")
#: 10 Nov 2026: the same rows' CfD support moves with a case's strikes (a private rate path re-strikes every
#: post-AR7 contract, small-scale ones included); the CfD line carries it, no rent does.
CFD_NO_RENT = ("CfD support on register rows with no rent (small-scale plant outside the ownership register, F28), "
               "private owners")
#: Bill lines the balancing stack can move, grouped with the register instrument that pays them
#: (the pairing of the tariff's declared residual parts, item N7).
DISPATCH_INSTRUMENTS = {
    "wholesale line against register wholesale revenue": (("direct_energy", "procurement_losses"),
                                                           "wholesale_revenue_gbp"),
    "constraint line, net of re-trading, against the register balancing settlement": (("bsuos_constraints",),
                                                                                       "balancing_settlement_gbp"),
    "CfD and allocated CCS lines against register CfD support": (("cfd", "allocated_ccs"), "cfd_support_gbp"),
    "RO line against the register's certificate income": (("renewables_obligation",), "roc_support_gbp"),
    "LDES floor line against the floor the register pays": (("ldes_cap_and_floor",), "ldes_floor_topup_paid_gbp"),
    # F19 decision 5 with review 2 (8 Oct 2026): each group re-reads the private battery top-up from its own
    # account, so the stack moves it. The register books it into rent, not into REVENUE_COLUMNS.
    "battery top-up line against the top-up the register books": (("battery_revenue_top_up",),
                                                                   "battery_revenue_top_up_gbp"),
}
#: Lines that pay retail suppliers' EBIT and headroom (F18: Ofgem's percentages of the bill's
#: costs, so they move when the stack moves the costs). Suppliers are private parties outside the
#: register, and the margin is a transfer: investors, never resource (8 Oct 2026).
SUPPLIER_MARGIN_LINES = ("supplier_costs_and_margin",)
REVENUE_COLUMNS = ("market_revenue_gbp", "support_revenue_gbp", "capacity_market_revenue_gbp",
                   "ldes_floor_topup_paid_gbp", "carbon_removal_cfd_revenue_gbp")
CHOICES = {
    "investor_counterfactual": ("required return: private owners' position is revenue less fuel, carbon, O&M and "
                                "the register's capital charge at their rate (after the corporation tax in it), "
                                "on their ownership share"),
    "acquisition": ("cash basis: the vendors and the lenders who fund the price receive the tariff's acquisition "
                    "annuity in place of the fleet's 2030 cash; the acquisition_gap_as_resource reading moves the "
                    "cost-of-capital gap (state valuation less owners', annuitised) from investors to the resource "
                    "saving and is reported beside it"),
    "sizewell_c_rab": ("OW10: the state refinances Sizewell C's regulated asset base at its value, so investors are "
                       "made whole and the fall in its return is a resource saving (a cost of capital), as for the "
                       "wires bought at RAV; the corporation-tax block of the allowance is not separated (declared)"),
    "networks": ("acquired at RAV: owners earn their allowed return on the proceeds, so the sale is a wash for them; "
                 "a premium over RAV is theirs, annuitised at the public rate consumers pay it at; a debt guarantee's "
                 "saving is a resource saving, its contingent cost to the Exchequer unpriced"),
    "carbon": ("a receipt of the Exchequer paid by emitters: a transfer inside the identity; the resource saving is "
               "fuel, O&M, capital and payments abroad, and the value of the emissions change is the Exchequer "
               "ledger's carbon change"),
    "hedging_allowances": "paid to suppliers (investors), not a resource saving, when a nationalised buyer stops paying them",
    "generator_tnuos": "a cost of the plant, as the register folds it into fixed O&M",
    "debt_guarantee": ("the Exchequer carries the guaranteed network debt's expected loss (PD x LGD x RAV x notional "
                       "gearing, lever guarantee_cost): an Exchequer cost and, being the default loss lenders no longer "
                       "bear, out of the resource saving; the rest of the compression stays a resource saving"),
    "observed_return_diagnostic": ("beside the required-return headline: capital providers (private lenders and "
                                   "owners, and the state's own lenders) credited with the capital charges actually "
                                   "paid, so every cost-of-capital part of the resource saving becomes an investor "
                                   "loss instead; both readings close"),
}
#: Resource parts that are a cost of capital: the observed-return diagnostic moves them to investors.
CAPITAL_REGISTER = "capital charges at the financing party's rate (register)"
CAPITAL_TAX = "less the corporation tax inside the private rate the state does not pay (a transfer from the Exchequer)"
CAPITAL_NETWORKS = "networks: cost of capital on the RAV (N6 / J6), net of the guarantee's expected loss"
#: F27 item 10 (23 Oct 2026): the supplier line, whose EBIT and headroom follow each case's costs.
SUPPLIER_MARGIN_LINE = "supplier_costs_and_margin"
#: F27 item 6 (24 Oct 2026): existing capacity on cost-based contracts (capacity_procurement.LINE).
CM_COST_BASED_LINE = "capacity_market_cost_based_existing"
#: F27 item 7 (23 Oct 2026): bought wires earn the vendors' return above the allowed return for the state.
NETWORK_EXCESS_RETURN_LABEL = "networks: return above the allowed return the vendors no longer earn (F27)"
CAPITAL_TARIFF_NEW_BUILD = "new-build capital: private WACC to the public rate"
#: OW10: Sizewell C's RAB refinanced at the public rate, a cost-of-capital saving like the wires'.
CAPITAL_SIZEWELL = "Sizewell C: cost of capital on its RAB at the public rate (OW10)"
#: Decision 68: the co-owners' rate reduction the state's implicit guarantee backs (a transfer, not a saving).
CO_INVESTMENT_GUARANTEE = "co-investors' rate reduction backed by the state's implicit guarantee (a transfer; decision 68)"
CAPITAL_PARTS = frozenset({CAPITAL_REGISTER, CAPITAL_TAX, CAPITAL_NETWORKS, CAPITAL_TARIFF_NEW_BUILD, CAPITAL_SIZEWELL,
                           CO_INVESTMENT_GUARANTEE})
#: F29 (user, 26 Oct 2026): workbook v2.0's economic rate less its cash rate on the plant the state finances,
#: the social cost of the systematic risk taxpayers now hold.  Booked equally by the Exchequer ledger, so a
#: cost to the resource saving and to the Exchequer, never to the bill; not a capital part (no cash moves).
SOCIALISED_RISK = ("systematic risk the state holds on the plant it finances, at the workbook's social price "
                   "(economic rate less cash rate; F29)")


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _sum(parts: Mapping[str, float]) -> float:
    return float(sum(float(v) for v in parts.values()))


# ── the register ──────────────────────────────────────────────────────────────

def register_position(register: pd.DataFrame, case_id: str) -> dict:
    """Private owners' share of producer rent, and the case's required revenue and corporation tax.

    The register splits each row's rent between the state
    (``public_surplus_before_recycling_gbp``: its share before the holding
    company, less the holding company) and the private co-owner, whose share is
    the rest, ``(1 - share) x (rent + overhead)`` (item B5: the overhead is the
    state's cost alone).  Rows without a rent (costs unresolved) carry no
    required revenue either; their revenue less fuel and carbon is reported
    apart (``unresolved_net_cash_gbp``), and the instrument totals cover every
    row, as the bill lines do.
    """
    rows = register.loc[register["case_id"].eq(case_id)]
    if rows.empty:
        raise ValueError(f"the finance register has no rows for {case_id}")
    rent, state = _num(rows, "producer_rent_gbp"), _num(rows, "public_surplus_before_recycling_gbp")
    unresolved = (rows["producer_rent_gbp"].isna() if "producer_rent_gbp" in rows.columns
                  else pd.Series(True, index=rows.index))
    revenue = sum(_num(rows, c) for c in REVENUE_COLUMNS)
    return {
        "private_rent_gbp": float((rent - state).sum()),
        "producer_rent_gbp": float(rent.sum()),
        "required_revenue_gbp": float(_num(rows, "annual_required_revenue_gbp").sum()),
        "capital_recovery_gbp": float(_num(rows, "annual_capital_recovery_gbp").sum()),
        # F27 item 1: inside the capital charge, but a payment to the former owners, not a resource cost.
        "owners_value_premium_gbp": float(_num(rows, "owners_value_premium_gbp").sum()),
        "corporation_tax_forgone_gbp": float(_num(rows, "corporation_tax_forgone_gbp").sum()),
        # A tax equivalent is inside required revenue and paid to the Exchequer: never a resource cost.
        "tax_equivalent_gbp": float(_num(rows, "annual_tax_equivalent_gbp").sum()),
        "carbon_removal_cfd_gbp": float(_num(rows, "carbon_removal_cfd_revenue_gbp").sum()),
        "socialised_risk_gbp": float(_num(rows, "socialised_risk_gbp").sum()),
        "revenue_gbp": float(revenue.sum()),
        "unresolved_net_cash_gbp": float((revenue - _num(rows, "fuel_carbon_gbp"))[unresolved].sum()),
        # 10 Nov 2026: the CM revenue on those rows (F28 small-scale plant): the design moves it, no rent carries it.
        "capacity_market_revenue_no_rent_gbp": float(_num(rows, "capacity_market_revenue_gbp")[unresolved].sum()),
        "cfd_support_no_rent_gbp": float(_num(rows, "cfd_support_gbp")[unresolved].sum()),
        **{c: float(_num(rows, c).sum()) for c in ("fuel_carbon_gbp", "wholesale_revenue_gbp", "balancing_settlement_gbp",
                                                   "cfd_support_gbp", "roc_support_gbp", "ldes_floor_topup_paid_gbp",
                                                   "battery_revenue_top_up_gbp")},
    }


# ── networks ──────────────────────────────────────────────────────────────────

def network_parties(network_lines: pd.DataFrame | None, treatment: str | None, public_rate: float | None,
                    guarantee_expected_loss_gbp: float = 0.0, levers: Mapping | None = None) -> dict:
    """The wires under a networks lever: consumers' saving, the tax transfer, investors' premium, the resource wedge.

    ``financing_saving_gbp`` in the network account is ``RAV x (allowed WACC -
    public rate) - premium x RAV x CRF(public rate, 30)``: the resource wedge is
    the rate gap on the RAV, and the premium's annuity is paid to the vendors
    (who earn their allowed return on the proceeds, so the sale at RAV is
    otherwise a wash).  A debt guarantee leaves owner and tax in place; the
    lenders' lower cost of debt is partly the default loss they no longer bear,
    which the Exchequer now carries (``guarantee_expected_loss_gbp``, the
    Exchequer ledger's charge), so the resource wedge is the compression less
    that expected loss.
    """
    loss = float(guarantee_expected_loss_gbp)
    if loss and treatment != "debt_guarantee":
        raise ValueError("a guarantee's expected loss belongs to the debt_guarantee treatment only")
    empty = {"bill_gbp": 0.0, "tax_gbp": 0.0, "premium_gbp": 0.0, "wedge_gbp": 0.0,
             "guarantee_expected_loss_gbp": 0.0, "fee_gbp": 0.0, "retained_gbp": 0.0, "gross_wedge_gbp": 0.0,
             "excess_return_gbp": 0.0}
    if not treatment or treatment == "private" or network_lines is None:
        return empty
    from scripts.public_ownership.network_finance_account import incidence_amounts, select_rows

    try:
        rows = select_rows(network_lines, treatment, levers)
    except ValueError:
        rows = network_lines.loc[network_lines["treatment"].astype(str).eq(treatment)]
    if rows.empty:
        raise ValueError(f"the network finance account carries no rows for treatment {treatment!r}")
    saving = float(_num(rows, "financing_saving_gbp").sum())
    tax = float(_num(rows, "tax_allowance_saving_gbp").sum())
    # F27 item 7: the vendors' return above the allowed return, now the state's (a transfer from them).
    excess = float(_num(rows, "excess_return_saving_gbp").sum())
    premium = 0.0
    if treatment == "acquired":
        rates = (pd.to_numeric(rows["public_real_rate"], errors="coerce")
                 if "public_real_rate" in rows.columns else pd.Series(float("nan"), index=rows.index))
        if rates.isna().any():
            if public_rate is None:
                raise ValueError("an acquired network needs the network account's public_real_rate")
            rates = rates.fillna(float(public_rate))
        rav, allowed = _num(rows, "rav_gbp"), _num(rows, "allowed_wacc_cpih_real")
        premium = float((rav * (allowed - rates) - _num(rows, "financing_saving_gbp")).sum())
    # User decision 5 (network_incidence, tax_equivalent): what reaches the bill, what the
    # Exchequer is paid as a fee and what it keeps as the owner's return.
    amounts = incidence_amounts(rows, treatment, levers) if levers else {}
    on_bill = float(sum(v["saving_on_bill_gbp"] for v in amounts.values())) if amounts else saving + tax + excess
    fee = float(sum(v["fee_to_exchequer_gbp"] for v in amounts.values())) if amounts else 0.0
    retained = float(sum(v["retained_by_exchequer_gbp"] for v in amounts.values())) if amounts else 0.0
    return {"bill_gbp": -on_bill, "tax_gbp": tax, "premium_gbp": premium, "wedge_gbp": saving + premium - loss,
            "guarantee_expected_loss_gbp": loss, "fee_gbp": fee, "retained_gbp": retained,
            "gross_wedge_gbp": saving + premium, "excess_return_gbp": excess}


# ── market-bill cases ─────────────────────────────────────────────────────────

def bill_line_moves(bill_case: Mapping) -> dict:
    """System GBP each line the projector moves adds to the bill (positive raises it), from its reconciled amounts.

    Every amount is one the projector checks against its route to GBP 1
    (``identity_residuals_gbp_2024``), recovered at the line's own denominator.
    """
    booked = bill_case.get("route_booking_booked_gbp_2024") or {}
    network = bill_case.get("network_adjustments_system_gbp_2024") or {}
    get = lambda key: float(bill_case.get(key) or 0.0)  # noqa: E731
    return {
        # FES 2050 decision 54: a rate path's own strikes move the line too (the register's support
        # revenue carries the same cut; absent, so zero, in 2030 and on the flat path).
        "cfd": -get("bill_allocated_r1_gbp_2024") - get("strike_basis_offset_saving_gbp_2024"),
        # F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA re-priced (absent before: zero).
        "remaining_ccs_public_financing_adjustment": -get("bill_allocated_dpa_gbp_2024"),
        "ldes_cap_and_floor": -get("bill_allocated_ldes_gbp_2024"),
        "public_cost_recovery_contracting": -get("bill_allocated_w1_gbp_2024"),
        "public_cfd_at_cost": -get("bill_allocated_public_cfd_at_cost_gbp_2024"),
        "public_surplus_rebate": -get("bill_allocated_rebate_gbp_2024"),
        "public_portfolio_shortfall_levy": get("shortfall_on_bill_gbp_2024"),
        **{line: -float(v) for line, v in booked.items()},
        "new_build_contracts_at_cost": get("new_build_contracts_bill_gbp_2024"),
        **{line: float(v) for line, v in network.items()},
        "pay_as_bid_ceiling": -get("pay_as_bid_ceiling_gbp_2024"),
        # Review 29 Sep 2026: the repaired levy offset, the state-deficit levy and the two
        # background lines (every case alike).
        "public_surplus_offset_domestic_levies": -get("bill_allocated_offset_domestic_levies_gbp_2024"),
        "public_state_deficit_levy": get("state_deficit_levy_gbp_2024"),
        **{line: -float(v) for line, v in (bill_case.get("legacy_levies_to_taxation_gbp_2024") or {}).items()},
        "beccs_carbon_removal_levy": get("beccs_carbon_removal_levy_gbp_2024"),
        # OW10: Sizewell C's RAB return at the public rate (absent before 3 Oct 2026: zero).
        "sizewell_c_rab_public_rate_adjustment": get("sizewell_c_rab_public_rate_gbp_2024"),
        # F19 decision 5: the battery top-up at each case's own cost (absent before 6 Oct 2026: zero).
        "battery_revenue_top_up_public_financing_adjustment": get("battery_top_up_owner_cost_gbp_2024"),
        # F9: the re-cleared 2030/31 T-4 (absent before 7 Oct 2026: zero).
        "capacity_market_state_bids_at_cost": get("capacity_market_state_bids_gbp_2024"),
        # F27 item 10: EBIT and headroom on the case's own costs (absent before 23 Oct 2026: zero).
        SUPPLIER_MARGIN_LINE: get("supplier_margin_change_gbp_2024"),
        # F27 item 6: existing capacity on cost-based contracts (absent before 24 Oct 2026: zero).
        CM_COST_BASED_LINE: get("capacity_market_cost_based_existing_gbp_2024"),
        # FES 2050 decision 73: the CM design's move (only where the projector reports one; never in 2030).
        **({LINE_CM_DESIGN: get("capacity_market_design_line_gbp_2024")}
           if "capacity_market_design_line_gbp_2024" in bill_case else {}),
    }


def market_case_parties(*, bill_case: Mapping, bill_private: Mapping, account_case: Mapping | None,
                        position: Mapping, position_private: Mapping, network: Mapping,
                        co_investment_transfer_gbp: float = 0.0) -> dict:
    """Consumers' bill change, investors' change and the resource saving for one market-bill case.

    ``position`` and ``position_private`` are :func:`register_position` of the
    case and of the private case on the same register; ``network`` is
    :func:`network_parties` for the case's networks lever.  ``co_investment_transfer_gbp``
    (decision 68): the part of the co-owners' rate reduction the Exchequer's implicit guarantee
    backs, a transfer taken out of the resource saving (the Exchequer ledger books it).
    """
    moves_case, moves_private = bill_line_moves(bill_case), bill_line_moves(bill_private)
    lines = {k: moves_case.get(k, 0.0) - moves_private.get(k, 0.0) for k in dict.fromkeys([*moves_case, *moves_private])}
    account = account_case or {}
    get = lambda mapping, key: float(mapping.get(key) or 0.0)  # noqa: E731
    investor = {
        "producer rent, private owners' share (register, at their required return)":
            position["private_rent_gbp"] - position_private["private_rent_gbp"],
        # R1: the strike cut the state does not fund comes out of the co-owners' contract
        # revenue (pro rata on the financing-saving basis; nothing at cost). F27 item 1: the R1 cut
        # includes Net Zero Teesside's DPA re-priced on remaining_ccs (the account books it as R1).
        "CfD strike cut borne by private co-owners (R1)":
            -(get(account, "r1_bill_cut_gbp") - get(account, "r1_cut_borne_by_state_gbp")),
        "merchant new build contracted at cost, paid to its private owners (D4)":
            lines.get("new_build_contracts_at_cost", 0.0),
        # The bounding row takes the margin off the bill and off the generators alike.
        "pay-as-bid ceiling taken from private generators (K3)": lines.get("pay_as_bid_ceiling", 0.0),
        # F27 item 1: pre-2025 CfD plant bought at its owners' value: their rent's present value, annuitised.
        "pre-2025 CfD plant bought at the owners' value: premium over cost paid to the former owners (F27)":
            float(position.get("owners_value_premium_gbp", 0.0)) - float(position_private.get("owners_value_premium_gbp", 0.0)),
        # F27 item 10: the cap's EBIT and headroom follow the case's costs; suppliers are private.
        "supplier EBIT and headroom on the case's own costs, paid to suppliers (F27)": lines.get(SUPPLIER_MARGIN_LINE, 0.0),
        "networks: premium over RAV paid to the vendors (N6)": network["premium_gbp"],
        NETWORK_EXCESS_RETURN_LABEL: -float(network.get("excess_return_gbp", 0.0)),
        # F9: private winners of the 2030/31 T-4 paid the lower clearing price (the state's own part is
        # the Exchequer ledger's); a transfer, so no resource part.
        "Capacity Market 2030/31 T-4 at the re-cleared price, private winners (F9)": (
            lines.get("capacity_market_state_bids_at_cost", 0.0)
            - (get(bill_case, "capacity_market_state_bids_state_part_gbp_2024")
               - get(bill_private, "capacity_market_state_bids_state_part_gbp_2024"))),
        # F27 item 6: existing units paid their going-forward cost less their own margin instead of the
        # clearing price; the row is the privately owned share only, so all of it is private owners'.
        "Capacity Market: existing plant on cost-based contracts, private owners paid less (F27)":
            lines.get(CM_COST_BASED_LINE, 0.0),
    }
    # F27 item 1: the owners'-value premium is in the capital charge but is a transfer, so it leaves both parts.
    premium = float(position.get("owners_value_premium_gbp", 0.0)) - float(position_private.get("owners_value_premium_gbp", 0.0))
    capital = position_private["capital_recovery_gbp"] - (position["capital_recovery_gbp"] - premium)
    # A tax equivalent the state pays is inside its required revenue and back with the
    # Exchequer, so it is taken out of the O&M part: it is a transfer, not a cost.
    tax_equivalent = float(position.get("tax_equivalent_gbp", 0.0)) - float(position_private.get("tax_equivalent_gbp", 0.0))
    resource = {
        CAPITAL_REGISTER: capital,
        "O&M and the state's overhead (register required revenue less its capital charge)":
            position_private["required_revenue_gbp"] - (position["required_revenue_gbp"] - premium) - capital
            + tax_equivalent,
        CAPITAL_TAX: -(position["corporation_tax_forgone_gbp"] - position_private["corporation_tax_forgone_gbp"]),
        CAPITAL_NETWORKS: network["wedge_gbp"],
        # OW10: the bill's fall on Sizewell C's levy is the RAB's cheaper finance.
        CAPITAL_SIZEWELL: -lines.get("sizewell_c_rab_public_rate_adjustment", 0.0),
        # F29: the risk the state takes on is a real cost, priced socially (the Exchequer books it too).
        SOCIALISED_RISK: -(float(position.get("socialised_risk_gbp", 0.0))
                           - float(position_private.get("socialised_risk_gbp", 0.0))),
    }
    # 10 Nov 2026: the CfD line's move on rows with no rent (F28 small-scale plant) reaches their private
    # owners and no rent; the party flows carry it inside the strike-cut flows (the whole line).
    cfd_no_rent = (float(position.get("cfd_support_no_rent_gbp", 0.0))
                   - float(position_private.get("cfd_support_no_rent_gbp", 0.0)))
    if cfd_no_rent:
        investor[CFD_NO_RENT] = cfd_no_rent
    out = {"bill_change_system_gbp": _sum(lines), "bill_lines_gbp": lines,
           "investor_parts_gbp": investor, "resource_parts_gbp": resource}
    design = cm_design_split(bill_case, bill_private)
    if design is not None:
        # FES 2050 decision 73: the register rows' CM change is in the rent (co-owners) and the account
        # (the state); the rest of the line's move is paid outside the register: to households (V2G and
        # residential DSR, consumers) and to investors (I&C DSR, interconnectors, unmodelled plant).
        investor[CM_DESIGN_OUTSIDE] = design["outside_register_other_change_gbp"]
        # 10 Nov 2026: the part of the register's change on rows with no rent (F28 small-scale plant) is
        # in the line but in no rent; their private owners receive it.
        no_rent = (float(position.get("capacity_market_revenue_no_rent_gbp", 0.0))
                   - float(position_private.get("capacity_market_revenue_no_rent_gbp", 0.0)))
        if no_rent:
            investor[CM_DESIGN_NO_RENT] = no_rent
        out["consumer_parts_gbp"] = {CM_DESIGN_HOUSEHOLDS: design["household_change_gbp"]}
        out["capacity_market_design_split_gbp"] = design
    if co_investment_transfer_gbp:
        resource[CO_INVESTMENT_GUARANTEE] = -float(co_investment_transfer_gbp)
        out["co_investment_transfer_gbp"] = float(co_investment_transfer_gbp)
    return out


def cm_design_split(bill_case: Mapping, bill_private: Mapping) -> dict | None:
    """FES 2050 decision 73: the CM design's line move split by payee, against private (None without it)."""
    case = bill_case.get("capacity_market_design")
    if not case:
        return None
    private = bill_private.get("capacity_market_design") or {}
    get = lambda mapping, key: float(mapping.get(key) or 0.0)  # noqa: E731
    keys = ("line_change_gbp", "register_change_gbp", "household_change_gbp", "outside_register_other_change_gbp")
    return {key: get(case, key) - get(private, key) for key in keys}


# ── the state-bids case: the bid stack on the private fleet ──────────────────

def dispatch_step_parties(*, line_changes: Mapping[str, float], position_group: Mapping, position_market: Mapping,
                          carbon_change_gbp: float, retrading_change_gbp: float) -> dict:
    """The bid stack's effect on the private fleet: its months against the market months, everything private.

    ``line_changes`` is system GBP per moved bill line (the group's private bill
    less the market private bill, each at its recovery rate); the positions are
    :func:`register_position` of ``private`` on the two registers.  Investors get
    the register rent change plus, line by line, what the bill pays that the
    register's instrument does not (balancing counterparties outside the
    register, the RO scheme's volume gap); re-trading on the constraint line is
    paid abroad.  Resource: fuel net of the Exchequer's carbon, the register's
    dispatch-dependent costs (variable O&M, TNUoS on the load factor), and the
    re-trading.
    """
    known = ({line for lines, _ in DISPATCH_INSTRUMENTS.values() for line in lines} | set(SUPPLIER_MARGIN_LINES)
             | {INTERCONNECTOR_PASS_THROUGH_LINE})
    unknown = sorted(set(line_changes) - known)
    if unknown:
        raise ValueError(f"the balancing stack moved bill lines this ledger cannot assign to a party: {unknown}")
    def delta(column: str) -> float:
        # A rent-only instrument a position predates (8 Oct 2026) is zero; every other column must be there.
        if column in RENT_ONLY_INSTRUMENTS and column not in position_group and column not in position_market:
            return 0.0
        return float(position_group[column] - position_market[column])

    investor = {
        "producer rent on the private fleet (register, balancing stack against market stack)":
            delta("producer_rent_gbp"),
        "revenue less fuel and carbon of plant without resolved costs (register)": delta("unresolved_net_cash_gbp"),
    }
    instruments = 0.0
    for label, (lines, instrument) in DISPATCH_INSTRUMENTS.items():
        change = float(sum(float(line_changes.get(line, 0.0)) for line in lines))
        if "bsuos_constraints" in lines:
            change -= float(retrading_change_gbp)
        instruments += delta(instrument)
        # What the bill pays on the line that the register's instrument does not
        # (balancing counterparties outside the register, the RO scheme's volume gap).
        investor[f"{label}: paid to parties the register does not book"] = change - delta(instrument)
    supplier = float(sum(float(line_changes.get(line, 0.0)) for line in SUPPLIER_MARGIN_LINES))
    if supplier:
        investor["supplier EBIT and headroom on the moved costs (retail suppliers, outside the register)"] = supplier
    # 9 Nov 2026: the cap-and-floor pass-through, paid to the interconnectors' GB halves (private, outside the
    # register), as the institution step books it; a stack and its market twin on re-solved months need not
    # move the cables' congestion rent identically, and the line was unassigned.
    pass_through = float(line_changes.get(INTERCONNECTOR_PASS_THROUGH_LINE, 0.0))
    if pass_through:
        investor[DISPATCH_PASS_THROUGH] = pass_through
    # Register revenue outside these instruments (the DPA and the rest) that moves with dispatch.
    # Rent-only instruments are revenue here too, or the rent change would count them twice.
    revenue = delta("revenue_gbp") + sum(delta(column) for column in RENT_ONLY_INSTRUMENTS)
    investor["register revenue outside the instruments the moved lines pay"] = -(revenue - instruments)
    resource = {
        "fuel, net of the carbon the Exchequer receives (register fuel and carbon)":
            -(delta("fuel_carbon_gbp") - float(carbon_change_gbp)),
        "variable O&M and load-factor-dependent charges (register required revenue)":
            -delta("required_revenue_gbp"),
        "interconnector re-trading paid abroad": -float(retrading_change_gbp),
    }
    return {"bill_change_system_gbp": float(sum(float(v) for v in line_changes.values())),
            "bill_lines_gbp": dict(line_changes), "investor_parts_gbp": investor, "resource_parts_gbp": resource}


def dispatch_line_changes(lines: pd.DataFrame, rates: Mapping[str, float | None], *,
                          group_column: str = "private_on_group", market_column: str = "private_on_market",
                          tolerance: float = 1e-9) -> dict:
    """System GBP per bill line the balancing stack moves on the private case, at each line's recovery rate."""
    frame = lines.drop_duplicates("component_id").set_index("component_id")
    change = (pd.to_numeric(frame[group_column], errors="coerce").fillna(0.0)
              - pd.to_numeric(frame[market_column], errors="coerce").fillna(0.0))
    out = {}
    for line, hh in change.items():
        if abs(float(hh)) <= tolerance:
            continue
        rate = rates.get(line)
        if not rate:
            raise ValueError(f"the balancing stack moved {line}, which has no recovery rate to put it in system GBP")
        out[str(line)] = float(hh) / float(rate)
    return out


# ── acquisition-tariff cases ──────────────────────────────────────────────────

def _residual_om(reconciliation: Mapping) -> float:
    """The tariff reconciliation's O&M residual part (system GBP; zero when absent)."""
    from scripts.public_ownership.nationalised_tariff import RESIDUAL_OM_PART

    return float((reconciliation.get("residual_parts_gbp") or {}).get(RESIDUAL_OM_PART, 0.0) or 0.0)


def tariff_case_parties(case: Mapping, *, corporation_tax_legacy_gbp: float, corporation_tax_new_build_gbp: float,
                        network: Mapping, socialised_risk_gbp: float = 0.0,
                        private_plant_carbon_cfd_change_gbp: float = 0.0,
                        tolerance: float = IDENTITY_TOLERANCE_GBP) -> dict:
    """The tariff's register ledger (item N7), every item assigned to consumers' counterparties.

    ``case`` is one entry of the nationalised bill summary's ``cases``.  The
    items are rebuilt from the structured fields (revenue requirement, the
    register-side diagnostics, the replaced lines' system amounts, the bill
    detail and the declared residual parts) and must add up to the bill change
    the bill ledger gives, or this refuses: an unassigned tariff line would
    otherwise pass as nobody's money.  ``socialised_risk_gbp`` (F29) is the case's
    systematic risk at the workbook's social price against private's, off the bill: a
    resource cost the Exchequer ledger books too.  ``private_plant_carbon_cfd_change_gbp`` is the
    carbon CfD the Exchequer pays the owners of BECCS the purchase leaves private, case less
    private (fix 8 Nov 2026: Drax BECCS is legacy under F28 and in no acquisition group, and a
    background lever such as the private WACC basis moves its CfD); not a bill line, so it is
    not in the items.
    """
    lines = case["revenue_requirement_gbp"]
    d = case["diagnostics"]
    reconciliation = case["reconciliation"]
    removed = reconciliation.get("replaced_lines_system_gbp_by_line") or {}
    detail = case.get("bill_detail") or {}
    line = lambda key: float(lines.get(key, 0.0) or 0.0)  # noqa: E731
    gone = lambda key: float(removed.get(key, 0.0) or 0.0)  # noqa: E731
    gb_half_rent = float(d.get("interconnector_gb_half_rent_to_tariff_gbp", 0.0) or 0.0)
    retrading = float(detail.get("interconnector_retrading_cost_gbp", 0.0) or 0.0)
    imports_repriced = line(LINE_IMPORTS) - float(d.get("import_register_domestic_revenue_gbp", 0.0) or 0.0)
    # Each bill item, signed as its contribution to the bill change, with its party.
    carbon_cfd_private = float(d.get("private_new_build_carbon_removal_cfd_gbp", 0.0) or 0.0)
    carbon_cfd_legacy = float(d.get("private_legacy_carbon_removal_cfd_gbp", 0.0) or 0.0)
    items = [
        ("investor", "legacy operating margin the owners give up (market + support - fuel - O&M on acquired plant)",
         -float(d["legacy_operating_margin_gbp"])),
        ("investor", "legacy acquisition annuity, to the vendors and the lenders who fund the price (cash basis)",
         line(LINE_LEGACY_RENEWABLES) + line(LINE_LEGACY_THERMAL)),
        # With BECCS's carbon CfD (user decision 2) the private owners' true rent includes
        # the CfDc the Exchequer paid them; the diagnostic excludes it, so the item splits.
        ("investor", "new-build producer rent private owners no longer earn (at their required return)",
         -(float(d["new_build_private_rent_gbp"]) + carbon_cfd_private)),
        ("exchequer", "carbon-removal CfD the Exchequer no longer pays private BECCS owners", carbon_cfd_private),
        # 8 Nov 2026 (user): acquired legacy BECCS (Drax under F28) mirrors new build; the legacy margin
        # leaves the carbon CfD out, so the former owners' loss of it and the Exchequer's saving are a pair.
        ("investor", "carbon-removal CfD the former owners of acquired BECCS no longer receive", -carbon_cfd_legacy),
        ("exchequer", "carbon-removal CfD the Exchequer no longer pays the former owners of acquired BECCS",
         carbon_cfd_legacy),
        ("investor", "Capacity Market agreements of acquired plant and the interconnectors' GB halves",
         -gone("capacity_market") + line(LINE_CM_NOT_ACQUIRED)
         - float(d.get("capacity_market_household_payment_change_gbp", 0.0) or 0.0)),
        ("investor", "LDES floor top-up no longer paid to the stores' owners", -gone("ldes_cap_and_floor")),
        ("investor", "hedging and imbalance allowances no longer paid to suppliers",
         -sum(gone(k) for k in ("shaping_adjustment",) + HEDGING_LINES)),
        ("investor", "RO line above the register's certificate income (the scheme's volume gap)",
         -(gone("renewables_obligation") - float(d["private_roc_support_gbp"]))),
        ("investor", "CfD economic-curtailment payment on pre-AR4 output",
         -float(detail.get("cfd_economic_curtailment_payment_gbp", 0.0) or 0.0)),
        ("investor", "interconnector GB halves: congestion rent given up", -gb_half_rent),
        ("investor", "interconnector GB halves: cap-and-floor pass-through no longer paid",
         -gone(INTERCONNECTOR_PASS_THROUGH_LINE)),
        ("investor", "interconnector GB halves: acquisition annuity (priced at the present value of their rent)", line(LINE_IC_ACQUISITION)),
        ("investor", "cancelled contracts: residual paid to former owners on the bill (G3)", line(LINE_CANCELLED_RESIDUAL)),
        # Money audit (13 Nov 2026): the residual's O&M part (case less private O&M, fuel and carbon on owned plant:
        # the state's self-insurance saving, any public O&M premium) is a real resource change, as in S1 and S2;
        # the rest of the residual is register payments to owners.
        ("investor", "register payments the removed lines do not carry (the tariff's declared residual parts)",
         float(reconciliation["residual_gbp"]) - _residual_om(reconciliation)),
        ("resource", "O&M, fuel and carbon on owned plant, case less private (self-insurance, any public O&M premium)",
         _residual_om(reconciliation)),
        ("resource", CAPITAL_TARIFF_NEW_BUILD, line(LINE_NEW_BUILD) - float(d["new_build_capital_private_gbp"])),
        ("resource", "state operating overhead on owned plant (OH1)", line(LINE_STATE_OVERHEAD)),
        # F19 P3 decision 5 (user, 6 Oct 2026): one net item.  Under I1 the re-trading is not saved
        # abroad: the measured link trade (the import line) pays it, premium included, so the two
        # former items (re-trading -1.18 bn, cross-border +1.16 bn) were one flow shown twice.
        ("resource", "cross-border energy and interconnector re-trading, net of what private paid abroad "
                     "(foreign price on the links, the foreign halves' rent)",
         line(LINE_IC_RETRADING) - retrading + imports_repriced + gb_half_rent + line(LINE_IC_FOREIGN_RENT)),
        ("resource", "real-time correction and forecast-error charge (I3 only)",
         line(LINE_REALTIME) + line(LINE_FORECAST_ERROR)),
        ("network", "networks: acquisition (N6)", float(network["bill_gbp"])),
        # OW10: the kept Sizewell C levy with its return at the public rate.
        ("resource", CAPITAL_SIZEWELL, float(reconciliation.get("sizewell_c_rab_public_rate_system_gbp", 0.0) or 0.0)),
        # F27 item 10: the suppliers' EBIT and headroom on the tariff's own costs (suppliers are private).
        ("investor", "supplier EBIT and headroom on the tariff's own costs, paid to suppliers (F27)",
         float(reconciliation.get("supplier_margin_system_gbp", 0.0) or 0.0)),
        # Review 29 Sep 2026: lines paid by or to the Exchequer, and a real transaction cost.
        ("exchequer", "Power BECCS capture leg paid by its carbon CfD from the Exchequer", line(LINE_BECCS_CAPTURE)),
        ("exchequer", "tax equivalents paid to the Exchequer by state-owned new build", line(LINE_TAX_EQUIVALENT)),
        # Independent review (15 Nov 2026): the DPA the kept remaining_ccs line pays is in the declared residual
        # (booked to investors above) though the state now owns the plant; this credit, the state's receipt
        # returned to the bill, nets it, so the former owners give the DPA up with the plant.
        ("investor", "Net Zero Teesside DPA received by the state as owner, returned to the bill",
         line(LINE_DPA_RECEIPTS)),
        ("exchequer", "Feed-in Tariffs: domestic share moved to general taxation", line(LINE_LEGACY_FIT)),
        ("exchequer", "Warm Home Discount: domestic share moved to general taxation", line(LINE_LEGACY_WHD)),
        ("resource", "acquisition advisory and transaction costs", line(LINE_TRANSACTION_COSTS)),
        # User answer Q5 (30 Sep 2026): the state charges market prices, the Treasury keeps the margin.
        ("exchequer", "the state charges market prices: margin over the tariff's cost kept by the Treasury",
         line(LINE_TREASURY_MARGIN)),
    ]
    # FES 2050 decision 58 (2050 tariffs only): the support lines' income on owned plant and the top-up.
    for line_id, label in ((LINE_ANCILLARY_INCOME, "ancillary-service revenue the former owners no longer receive "
                                                   "(paid out of the kept BSUoS lines, now to the state)"),
                           (LINE_GATE_FEE_INCOME, "EfW gate fees the former owners no longer receive (paid by waste "
                                                  "disposal, now to the state)"),
                           (LINE_SUBHOURLY_INCOME, "battery sub-hourly revenue the former owners no longer receive "
                                                   "(paid by other market participants, now to the state)")):
        if line_id in lines:
            items.append(("investor", label, line(line_id)))
    if TOP_UP_LINE in removed:
        items.append(("investor", "life-extension top-up no longer paid to the owners of acquired plant",
                      -float(d.get("private_top_up_owned_gbp", 0.0) or 0.0)))
    if BATTERY_TOP_UP_LINE in removed:
        # FES 2050 decision 63 (2050 tariffs only): the battery top-up the former owners were paid.
        items.append(("investor", "battery revenue top-up no longer paid to the owners of acquired batteries",
                      -float(d.get("private_battery_top_up_owned_gbp", 0.0) or 0.0)))
    if "capacity_market_household_payment_change_gbp" in d:
        # FES 2050 decision 73: under adequacy planning the targeted contract pays V2G owners and
        # residential DSR at its own price; their change is households' (consumers'), not investors'.
        items.append(("household", CM_DESIGN_HOUSEHOLDS, float(d["capacity_market_household_payment_change_gbp"])))
    bill_change = float(reconciliation["bill_change_system_gbp"])
    explained = float(sum(a for _, _, a in items))
    if abs(explained - bill_change) > max(tolerance, 1e-12 * abs(bill_change)):
        raise ValueError(
            f"{case.get('case_id')}: the tariff's register ledger (GBP {explained:,.2f}) no longer adds up to its bill "
            f"change (GBP {bill_change:,.2f}); a tariff line has no party in the investor ledger")
    investor = {label: amount for party, label, amount in items if party == "investor"}
    investor["cancelled contracts: residual paid to former owners by the Exchequer (G3)"] = float(
        d.get("cancelled_contract_residual_exchequer_transfer_gbp", 0.0) or 0.0)
    # The ledger books the tax the state forgoes on acquired legacy plant; its former owners
    # stop paying it.  F19 decision 3: on the 2030 operating margin after capital allowances
    # and interest (calculate_financing_scenarios.legacy_corporation_tax), not the as-new base.
    investor["corporation tax the former owners no longer pay on legacy plant (on its 2030 margin after capital "
             "allowances and interest; a transfer within the public sector once the state owns it)"] = float(
        corporation_tax_legacy_gbp)
    investor[PRIVATE_PLANT_CARBON_CFD] = float(private_plant_carbon_cfd_change_gbp)
    investor["networks: premium over RAV paid to the vendors (N6)"] = float(network["premium_gbp"])
    investor[NETWORK_EXCESS_RETURN_LABEL] = -float(network.get("excess_return_gbp", 0.0))
    # Lever acquisition_financing (user decision 5): the vendors and lenders are paid the
    # annuity whoever funds it; funded from taxation it is off the tariff.
    investor["acquisition annuity paid from general taxation (to the vendors and lenders)"] = float(
        d.get("acquisition_annuity_exchequer_funded_gbp", 0.0) or 0.0)
    resource = {label: -amount for party, label, amount in items if party == "resource"}
    resource[CAPITAL_TAX] = -float(corporation_tax_new_build_gbp)
    resource[CAPITAL_NETWORKS] = float(network["wedge_gbp"])
    resource[SOCIALISED_RISK] = -float(socialised_risk_gbp)
    out = {"bill_change_system_gbp": bill_change, "bill_lines_gbp": {label: amount for _, label, amount in items},
           "bill_items": [(party, label, amount) for party, label, amount in items],
           "exchequer_parts_gbp": {label: amount for party, label, amount in items if party == "exchequer"},
           "investor_parts_gbp": investor, "resource_parts_gbp": resource}
    # FES 2050 decision 73: households' CM income under adequacy planning (no such item in 2030).
    households = {label: amount for party, label, amount in items if party == "household"}
    if households:
        out["consumer_parts_gbp"] = households
    return out


def legacy_acquisition_memo(rows: pd.DataFrame, valuation: pd.DataFrame | None) -> dict:
    """The acquisition annuity against the owners' and the state's valuations, both annuitised at its own factor.

    ``rows`` are the tariff case's acquired legacy register rows
    (``acquisition_price_gbp``, ``annual_capital_recovery_gbp``); ``valuation``
    is the legacy valuation (``pv_state_gbp`` at the public rate, ``pv_market_gbp``
    at the owners' rate).  The premium is a transfer to the vendors in either
    reading; the gap is what the acquisition_gap_as_resource reading counts as a resource
    saving (the state values the same streams higher because it discounts at
    the public rate) rather than as an investor loss.
    """
    price = _num(rows, "acquisition_price_gbp")
    annuity = _num(rows, "annual_capital_recovery_gbp")
    if valuation is None or rows.empty or float(price.sum()) <= 0.0:
        return {}
    factor = float(annuity.sum()) / float(price.sum())
    values = valuation.drop_duplicates("asset_id").set_index("asset_id").reindex(rows["asset_id"])
    missing = int(values["pv_market_gbp"].isna().sum())
    if missing:
        raise ValueError(f"{missing} acquired legacy rows have no legacy valuation")
    market = pd.to_numeric(values["pv_market_gbp"], errors="coerce").fillna(0.0).to_numpy()
    state = pd.to_numeric(values["pv_state_gbp"], errors="coerce").fillna(0.0).to_numpy()
    return {
        "legacy_acquisition_price_gbp": float(price.sum()),
        "legacy_owners_valuation_gbp": float(market.sum()),
        "legacy_state_valuation_gbp": float(state.sum()),
        "legacy_annuity_factor": factor,
        "legacy_acquisition_annuity_gbp": float(annuity.sum()),
        "legacy_premium_over_owners_valuation_annuitised_gbp": float((price.to_numpy() - market).sum()) * factor,
        "legacy_cost_of_capital_gap_annuitised_gbp": float((state - market).sum()) * factor,
    }


# ── the institution case ─────────────────────────────────────────────────────

def institution_step_parties(institution: Mapping, *, carbon_change_gbp: float,
                             tolerance: float = IDENTITY_TOLERANCE_GBP) -> dict:
    """I3 less its I1 twin (item N10): every moved line is resource or carbon; nobody private is paid differently.

    The institution comparison carries the plant paid as today across at its I1
    payments, so investors do not move.  ``carbon_change_gbp`` is the Exchequer
    ledger's own central-less-market carbon receipt; the audit's must agree, or
    the two stages read different dispatch.
    """
    audit = institution.get("audit") or {}
    deltas = audit.get("measured_deltas_gbp") or {}
    for key in ("fuel_delta_gbp", "carbon_delta_gbp"):
        if key not in audit:
            raise ValueError(f"the institution comparison's audit carries no {key}; it predates review B13")
    if abs(float(audit["carbon_delta_gbp"]) - float(carbon_change_gbp)) > tolerance:
        raise ValueError(
            f"the institution comparison's carbon change (GBP {float(audit['carbon_delta_gbp']):,.0f}) is not the "
            f"ledger's (GBP {float(carbon_change_gbp):,.0f}): the two read different months or rates")
    resource = {
        "fuel on both solved years, net of the CHP heat credit": -float(audit["fuel_delta_gbp"]),
        "variable O&M": -float(deltas.get(LINE_VARIABLE_OM, 0.0)),
        "cross-border energy on the links": -float(deltas.get(LINE_IMPORTS, 0.0)),
        "interconnector foreign halves' rent on the central schedule": -float(deltas.get(LINE_IC_FOREIGN_RENT, 0.0)),
        "interconnector re-trading removed": float(audit.get("interconnector_retrading_removed_gbp", 0.0) or 0.0),
        "real-time correction replaced by the borrowed forecast-error charge": (
            float(audit.get("realtime_correction_replaced_by_charge_gbp", 0.0) or 0.0)
            - float(audit.get("forecast_error_charge_added_gbp", 0.0) or 0.0)),
    }
    change = float(institution["institution_change_system_gbp"])
    explained = -_sum(resource) + float(carbon_change_gbp)
    if abs(explained - change) > max(tolerance, 1e-12 * abs(change)):
        raise ValueError(f"the institution comparison moves GBP {change:,.2f} of lines but its audit explains "
                         f"GBP {explained:,.2f}; a moved line has no party in the investor ledger")
    return {"bill_change_system_gbp": change, "investor_parts_gbp": {
        "plant paid as today, carried across at its I1 payments": 0.0}, "resource_parts_gbp": resource}


# ── P1/P2 under central dispatch: the institution step on the private fleet (20 Oct 2026) ──

#: The bill lines the central-dispatch year can move on the private case: the dispatch instruments, the
#: suppliers' margin and the interconnectors' cap-and-floor pass-through (its rent moves with the schedule).
INSTITUTION_STEP_LINES = frozenset({line for lines, _ in DISPATCH_INSTRUMENTS.values() for line in lines}
                                   | set(SUPPLIER_MARGIN_LINES) | {INTERCONNECTOR_PASS_THROUGH_LINE})
#: The measured quantities the step names, system GBP (central less market unless named).
INSTITUTION_STEP_KEYS = (
    "schedule_trade_change_gbp",          # the Stage 1 schedules' payment abroad for cross-border energy
    "rent_change_before_make_whole_gbp",  # the cables' Stage 1 congestion rent
    "make_whole_gbp",                     # the interconnector day-ahead make-whole (central only)
    "gb_half_revenue_change_gbp",         # the GB halves' share of rent + make-whole
    "foreign_half_revenue_change_gbp",    # the foreign halves' share
    "realtime_links_gbp",                 # central real-time link trade (physical less schedule, premium in)
    "market_retrading_gbp",               # the market year's re-trading (gone under central dispatch)
    "forecast_error_charge_gbp",          # the borrowed FTI charge on the constraint line
    "counter_trade_premium_gbp",          # the central stage's premium, when on the constraint line
    "realtime_at_cost_register_gbp",      # the real-time correction paid to register units at cost
    "exchequer_paid_revenue_change_gbp",  # register revenue the Exchequer pays (BECCS's carbon CfD)
)
INSTITUTION_STEP_ABROAD = ("cross-border energy and rent paid abroad (the Stage 1 trade, the foreign halves' rent and "
                           "make-whole, the central real-time link trade; the market year's re-trading removed)")
INSTITUTION_STEP_FORECAST_ERROR = ("the borrowed forecast-error charge (with the counter-trade premium) less the real-time "
                                   "correction it pays for (decision 9: a resource cost)")


def institution_step_parties_market(*, line_changes: Mapping[str, float], position_central: Mapping,
                                    position_market: Mapping, carbon_change_gbp: float,
                                    measured: Mapping[str, float]) -> dict:
    """The central-dispatch year against the market year on the private fleet: the institution step of P1/P2-I3.

    Like :func:`dispatch_step_parties` (one owner, two dispatches): ``line_changes`` is
    system GBP per moved bill line (private on the I3 twin's bill less private on the
    market bill, each at its recovery entry); the positions are :func:`register_position`
    of ``private`` on the two registers. Investors take the register rent change and, line
    by line, what the bill pays that the register's instrument does not; the parts the
    settlement measures are named rather than left in those residuals (``measured``,
    :data:`INSTITUTION_STEP_KEYS`):

    * the GB end of the Stage 1 cross-border trade on the wholesale line splits into the
      payment abroad and the cables' rent; the day-ahead make-whole on the constraint line is
      the cables' too; the GB halves' share (investors) and the foreign halves' (abroad) are
      named, as is the cap-and-floor pass-through line, which the GB halves pay;
    * the forecast-error charge (and the counter-trade premium) on the constraint line pays
      the real-time correction: at cost to register units (inside their balancing settlement)
      and at the trade prices abroad on the links; the rest is the resource the borrowed
      charge stands for (decision 9);
    * the market year's re-trading was paid abroad; it is gone;
    * register revenue the Exchequer pays (BECCS's carbon CfD) stays with its owners.

    Resource: fuel net of the Exchequer's carbon, the register's dispatch-dependent costs,
    the payments abroad and the forecast-error resource. The identity closes by
    construction; :func:`three_party_row` checks it with the Exchequer's step.
    """
    unknown = sorted(set(line_changes) - INSTITUTION_STEP_LINES)
    if unknown:
        raise ValueError(f"the central-dispatch year moved bill lines this ledger cannot assign to a party: {unknown}")
    missing = [k for k in INSTITUTION_STEP_KEYS if k not in measured]
    if missing:
        raise ValueError(f"the institution step needs the settlement's {missing}")
    m = {k: float(measured[k]) for k in INSTITUTION_STEP_KEYS}
    split = m["gb_half_revenue_change_gbp"] + m["foreign_half_revenue_change_gbp"]
    if abs(split - (m["rent_change_before_make_whole_gbp"] + m["make_whole_gbp"])) > IDENTITY_TOLERANCE_GBP:
        raise ValueError("the cables' GB and foreign halves do not add up to their rent and make-whole change")

    def delta(column: str) -> float:
        if column in RENT_ONLY_INSTRUMENTS and column not in position_central and column not in position_market:
            return 0.0
        return float(position_central[column] - position_market[column])

    # Named parts of the wholesale and constraint lines (the rest of each line is the residual below).
    named = {"direct_energy": m["schedule_trade_change_gbp"] + m["rent_change_before_make_whole_gbp"],
             "bsuos_constraints": (m["make_whole_gbp"] + m["forecast_error_charge_gbp"] + m["counter_trade_premium_gbp"]
                                   - m["realtime_at_cost_register_gbp"] - m["market_retrading_gbp"])}
    investor = {
        "producer rent on the private fleet (register, central-dispatch year against the market year)":
            delta("producer_rent_gbp"),
        "revenue less fuel and carbon of plant without resolved costs (register)": delta("unresolved_net_cash_gbp"),
    }
    instruments = 0.0
    for label, (lines, instrument) in DISPATCH_INSTRUMENTS.items():
        change = float(sum(float(line_changes.get(line, 0.0)) for line in lines))
        change -= float(sum(named.get(line, 0.0) for line in lines))
        instruments += delta(instrument)
        investor[f"{label}: paid to parties the register does not book"] = change - delta(instrument)
    investor["interconnector GB halves: Stage 1 rent and the day-ahead make-whole"] = m["gb_half_revenue_change_gbp"]
    investor["interconnector GB halves: the cap-and-floor pass-through they pay (bill line)"] = float(
        line_changes.get(INTERCONNECTOR_PASS_THROUGH_LINE, 0.0))
    investor["Power BECCS carbon-removal CfD (paid by the Exchequer)"] = m["exchequer_paid_revenue_change_gbp"]
    supplier = float(sum(float(line_changes.get(line, 0.0)) for line in SUPPLIER_MARGIN_LINES))
    if supplier:
        investor["supplier EBIT and headroom on the moved costs (retail suppliers, outside the register)"] = supplier
    # Register revenue no moved line pays comes out of the rent (the Exchequer's part is named above).
    revenue = delta("revenue_gbp") + sum(delta(column) for column in RENT_ONLY_INSTRUMENTS)
    investor["register revenue outside the instruments the moved lines pay"] = -(revenue - instruments)
    # The rent's other income (batteries' sub-hourly revenue, other services), read off the register: no
    # bill line and no modelled party pays it, so its change with dispatch is taken back out of the rent
    # (e2e 20 Oct 2026: -10.8 m of battery sub-hourly revenue on the CENTRAL network's year).
    other = (delta("producer_rent_gbp") - revenue + delta("fuel_carbon_gbp") + delta("required_revenue_gbp")
             + delta("unresolved_net_cash_gbp"))
    investor["register income no bill line pays (battery sub-hourly revenue, other services)"] = -other
    resource = {
        "fuel, net of the carbon the Exchequer receives (register fuel and carbon)":
            -(delta("fuel_carbon_gbp") - float(carbon_change_gbp)),
        "variable O&M and load-factor-dependent charges (register required revenue)": -delta("required_revenue_gbp"),
        INSTITUTION_STEP_ABROAD: -(m["schedule_trade_change_gbp"] + m["foreign_half_revenue_change_gbp"]
                                   + m["realtime_links_gbp"] - m["market_retrading_gbp"]),
        INSTITUTION_STEP_FORECAST_ERROR: -(m["forecast_error_charge_gbp"] + m["counter_trade_premium_gbp"]
                                           - m["realtime_at_cost_register_gbp"] - m["realtime_links_gbp"]),
    }
    return {"bill_change_system_gbp": float(sum(float(v) for v in line_changes.values())),
            "bill_lines_gbp": dict(line_changes), "investor_parts_gbp": investor, "resource_parts_gbp": resource,
            "measured_gbp": m}


# ── the identity ──────────────────────────────────────────────────────────────

def combine(*steps: Mapping) -> dict:
    """Stack measured steps (the state-bids case's two, the institution case on its twin): parts are summed by label."""
    out = {"bill_change_system_gbp": 0.0, "investor_parts_gbp": {}, "resource_parts_gbp": {}}
    for step in steps:
        out["bill_change_system_gbp"] += float(step["bill_change_system_gbp"])
        for key in ("investor_parts_gbp", "resource_parts_gbp"):
            for label, amount in step[key].items():
                out[key][label] = out[key].get(label, 0.0) + float(amount)
        # FES 2050 decision 73: households' CM income (a step carries it only under the design).
        for label, amount in (step.get("consumer_parts_gbp") or {}).items():
            parts = out.setdefault("consumer_parts_gbp", {})
            parts[label] = parts.get(label, 0.0) + float(amount)
    return out


def three_party_row(parties: Mapping, *, vat_change_gbp: float, exchequer_change_gbp: float,
                    case_id: str, tolerance: float = IDENTITY_TOLERANCE_GBP, risk_adjusted: bool = False) -> dict:
    """The four columns and the identity for one case; refuses when it does not close.

    ``vat_change_gbp`` is the change in domestic VAT (the consumer pays it, the
    Exchequer ledger receives it); ``exchequer_change_gbp`` is the ledger's net
    change, VAT included.  ``risk_adjusted`` (decision 69, a view beside the headline): the
    cost-of-capital saving is the price of risk the taxpayer now bears (the Lucas view), so it
    leaves the resource saving for a taxpayer line on the Exchequer's side; checked to close.
    """
    consumer = -float(parties["bill_change_system_gbp"]) - float(vat_change_gbp)
    # FES 2050 decision 73: households' own CM income (V2G, residential DSR) is a consumer's gain too.
    consumer_parts = dict(parties.get("consumer_parts_gbp") or {})
    if consumer_parts:
        consumer += _sum(consumer_parts)
    investor = _sum(parties["investor_parts_gbp"])
    resource = _sum(parties["resource_parts_gbp"])
    residual = consumer + float(exchequer_change_gbp) + investor - resource
    scale = max(1.0, abs(consumer), abs(investor), abs(resource))
    if abs(residual) > max(tolerance, 1e-12 * scale):
        raise ValueError(
            f"{case_id}: consumers ({consumer / 1e6:,.3f} m) + Exchequer ({float(exchequer_change_gbp) / 1e6:,.3f} m) "
            f"+ investors ({investor / 1e6:,.3f} m) differ from the resource saving ({resource / 1e6:,.3f} m) by "
            f"GBP {residual:,.2f}; a transfer is booked to one party and not the other")
    # The diagnostic pair (choice 1): capital providers credited with the capital charges paid.
    capital = float(sum(v for k, v in parties["resource_parts_gbp"].items() if k in CAPITAL_PARTS))
    # F29: the risk already priced at the workbook's social price (a cost, positive here).
    socialised = -float(parties["resource_parts_gbp"].get(SOCIALISED_RISK, 0.0))
    observed_residual = consumer + float(exchequer_change_gbp) + (investor - capital) - (resource - capital)
    if abs(observed_residual) > max(tolerance, 1e-12 * scale):
        raise ValueError(f"{case_id}: the observed-return reading does not close (GBP {observed_residual:,.2f})")
    return {
        "consumer_bill_change_system_gbp": float(parties["bill_change_system_gbp"]),
        "consumer_change_vs_private_gbp": consumer,
        "investor_change_vs_private_gbp": investor,
        "investor_basis": "required_return",
        "investor_change_required_return_basis_gbp": investor,
        "investor_change_observed_return_basis_gbp": investor - capital,
        "resource_saving_required_return_basis_gbp": resource,
        "resource_saving_observed_return_basis_gbp": resource - capital,
        "cost_of_capital_saving_gbp": capital,
        "investor_change_vs_private_parts_gbp": dict(parties["investor_parts_gbp"]),
        "resource_saving_vs_private_gbp": resource,
        "resource_saving_vs_private_parts_gbp": dict(parties["resource_parts_gbp"]),
        "three_party_identity_residual_gbp": residual,
        **({"consumer_change_vs_private_parts_gbp": consumer_parts} if consumer_parts else {}),
        "socialised_risk_vs_private_gbp": socialised,
        **(risk_adjusted_view(consumer, float(exchequer_change_gbp), investor, resource, capital, case_id, tolerance,
                              socialised=socialised)
           if risk_adjusted else {}),
    }


def risk_adjusted_view(consumer: float, exchequer: float, investor: float, resource: float, capital: float,
                       case_id: str, tolerance: float = IDENTITY_TOLERANCE_GBP, socialised: float = 0.0) -> dict:
    """Decision 69 (user, 3 Oct 2026): the risk-adjusted (Lucas) reading of one case, checked to close.

    The state's cheaper capital (``cost_of_capital_saving_gbp``: the capital parts of the resource
    saving, net of the corporation tax inside the private rate) is read as the market price of the
    risk the taxpayer takes on, not a saving: the resource saving loses it and the Exchequer's side
    carries it as ``taxpayer_risk_bearing_gbp``. The observed-return diagnostic moves the same wedge
    to investors (capital providers paid less); this view moves it to taxpayers. Consumers and
    investors are unchanged; the headline is unchanged.  F29: the headline already charges the
    risk at the workbook's social price (``socialised``, in both the resource saving and the
    Exchequer), so this view puts the market price in its place, never on top of it.
    """
    exchequer_adjusted = exchequer + socialised - capital
    resource_adjusted = resource + socialised - capital
    residual = consumer + exchequer_adjusted + investor - resource_adjusted
    if abs(residual) > max(tolerance, 1e-12 * max(1.0, abs(consumer), abs(investor), abs(resource))):
        raise ValueError(f"{case_id}: the risk-adjusted reading does not close (GBP {residual:,.2f})")
    return {"resource_saving_risk_adjusted_gbp": resource_adjusted, "taxpayer_risk_bearing_gbp": capital,
            "exchequer_change_risk_adjusted_gbp": exchequer_adjusted}


def household_columns(row: Mapping, per_household) -> dict:
    """Per-household values of the four columns (flat over final demand, like the rest of the ledger)."""
    keys = ("consumer_change_vs_private_gbp", "investor_change_vs_private_gbp", "resource_saving_vs_private_gbp",
            "three_party_identity_residual_gbp", "investor_change_required_return_basis_gbp",
            "investor_change_observed_return_basis_gbp", "resource_saving_required_return_basis_gbp",
            "resource_saving_observed_return_basis_gbp", "cost_of_capital_saving_gbp",
            "resource_saving_risk_adjusted_gbp", "taxpayer_risk_bearing_gbp", "exchequer_change_risk_adjusted_gbp",
            "socialised_risk_vs_private_gbp",
            "investor_change_acquisition_gap_as_resource_gbp", "resource_saving_acquisition_gap_as_resource_gbp")
    return {k.replace("_gbp", "_gbp_per_household"): per_household(row[k]) for k in keys if k in row}


def acquisition_gap_as_resource(row: dict, memo: Mapping) -> None:
    """Add the other reading of an acquisition: the cost-of-capital gap moved to the resource saving.

    Choice 2, not choice 1: the headline is the cash basis; this values the
    purchase as the owners' rate would (state valuation less owners',
    annuitised), a gap investors then do not lose and the country saves.
    """
    gap = float(memo.get("legacy_cost_of_capital_gap_annuitised_gbp", 0.0) or 0.0)
    row.update(memo)
    row["investor_change_acquisition_gap_as_resource_gbp"] = row["investor_change_vs_private_gbp"] + gap
    row["resource_saving_acquisition_gap_as_resource_gbp"] = row["resource_saving_vs_private_gbp"] + gap


def headline_table(rows: Sequence[Mapping]) -> pd.DataFrame:
    """Case, the four columns and the residual, system GBP m."""
    columns = ["case_id", "consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp",
               "investor_change_vs_private_gbp", "resource_saving_vs_private_gbp", "three_party_identity_residual_gbp"]
    frame = pd.DataFrame([{c: r.get(c) for c in columns} for r in rows if "investor_change_vs_private_gbp" in r])
    return frame.set_index("case_id").astype(float).div(1e6).reset_index() if not frame.empty else frame

