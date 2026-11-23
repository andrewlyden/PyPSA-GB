"""Itemised money flows between the parties, per case against private (review 29 Sep 2026).

The three-party ledger (:mod:`investor_ledger`) gives each case four totals
against private: consumers, the Exchequer, private investors and the resource
saving.  This module itemises them for the explorer's cash-flow breakdown: each
row is one flow of money for one case, from a ``payer`` to a ``payee``.

**Parties.** ``consumers``, ``exchequer`` (the Treasury and the state's own
companies together, as the Exchequer ledger counts them), ``private_investors``
(owners, co-owners, vendors, lenders and suppliers), ``foreign`` (counterparties
abroad) and ``resource``, the real cost of supply: a flow *from* resource is a
real saving that lands with its payee, a flow *to* resource a real cost the payer
bears.  Payments abroad are part of the real cost of supply, so ``foreign`` sits
on the resource side of the checks.

**Sign.**  ``gbp_m`` is never negative: a flow computed negative is written the
other way round.  ``gbp_per_household`` is on the ledger's central key (flat over
final demand, 2025 money per 3,100 kWh household).

**Closure, enforced** (:func:`check`), each to GBP 1:

1. every party's net (received less paid) equals its three-party column;
2. transfer flows (between consumers, the Exchequer and investors) net to zero
   across the parties;
3. the resource flows (including payments abroad) sum to the resource saving.

**How the market-bill flows are built.**  Per register row, with ``s`` the
state's share of the case row, ``dcap`` and ``dom`` the fall in capital charge
and in operating cost (O&M, the public O&M premium) against private, ``oh`` the
state's holding company, ``rent_p`` private's rent (net of the carbon-removal CfD)
and ``teq`` a tax equivalent:

* the state's share of the saving, ``s x dcap`` and ``s x dom - oh``, lands with
  the Exchequer; the co-owners' share with private investors;
* ``-s x rent_p`` moves from the Exchequer to private investors: the state takes
  on plant that earns less (or more) than private owners require;
* the corporation tax inside the private rate, which the state does not pay, is
  a flow from the Exchequer to resource (it is not a real saving);
* each bill line is a flow between consumers and the party the line pays or is
  paid by (the strike cut from the state and the co-owners, the surplus returned,
  the levies, the network wedge from resource), and domestic VAT follows the bill.

These are the same quantities the three-party identity adds up, so the checks
guard the itemisation against drift, not the identity itself.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.public_ownership.levers import OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT

TOLERANCE_GBP = 1.0
#: F19 decision 3 (6 Oct 2026): the forgone corporation tax's label.
#: F19 decision 7: the cohort a takeover's sign is judged on.
COHORT_KEYS = ("carrier", "support_type", "cfd_round", "finance_vintage")
CT_FLOW_LABEL = ("Corporation tax the state's plant does not pay (a transfer within the public sector; "
                 "Great British Energy pays corporation tax as a company today)")
PARTIES = ("consumers", "exchequer", "private_investors", "foreign", "resource")
REAL_PARTIES = ("consumers", "exchequer", "private_investors")
RESOURCE_SIDE = ("resource", "foreign")
CATEGORIES = ("financing", "tax", "deficit", "acquisition", "contract re-pricing", "VAT", "carbon", "guarantee",
              "levies", "other")
COLUMN_BY_PARTY = {"consumers": "consumer_change_vs_private_gbp", "exchequer": "exchequer_net_vs_private_gbp",
                   "private_investors": "investor_change_vs_private_gbp"}
FLOW_COLUMNS = ["case_id", "flow_id", "label", "payer", "payee", "gbp_m", "gbp_per_household", "category"]


class Flows:
    """An append-only list of flows for one case."""

    def __init__(self, case_id: str, per_household):
        self.case_id, self.per_household, self.rows = case_id, per_household, []

    def add(self, flow_id: str, label: str, payer: str, payee: str, gbp: float, category: str) -> None:
        if payer not in PARTIES or payee not in PARTIES or category not in CATEGORIES:
            raise ValueError(f"flow {flow_id}: unknown party or category ({payer}, {payee}, {category})")
        gbp = float(gbp)
        if not np.isfinite(gbp) or abs(gbp) < 1e-6:
            return
        if gbp < 0:
            payer, payee, gbp = payee, payer, -gbp
        self.rows.append({"case_id": self.case_id, "flow_id": flow_id, "label": label, "payer": payer, "payee": payee,
                          "gbp": gbp, "gbp_m": gbp / 1e6, "gbp_per_household": float(self.per_household(gbp)),
                          "category": category})


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


# ── the register, row by row ─────────────────────────────────────────────────

def register_components(register: pd.DataFrame, case_id: str, private_id: str = "private") -> dict:
    """The row-level split of the state's and the co-owners' positions against private (see the module)."""
    case = register.loc[register["case_id"].eq(case_id)].drop_duplicates("asset_id").set_index("asset_id")
    private = register.loc[register["case_id"].eq(private_id)].drop_duplicates("asset_id").set_index("asset_id")
    private = private.reindex(case.index)
    ok = case["producer_rent_gbp"].notna() & private["producer_rent_gbp"].notna() if "producer_rent_gbp" in case.columns \
        else pd.Series(False, index=case.index)
    c, p = case.loc[ok], private.loc[ok]
    s = _num(c, "ownership_share")
    oh = _num(c, "annual_state_overhead_gbp")
    teq_c, teq_p = _num(c, "annual_tax_equivalent_gbp"), _num(p, "annual_tax_equivalent_gbp")
    # F27 item 1: the owners'-value premium is inside the case's capital charge but is a payment to the
    # former owners, its own flow below, so it leaves the capital and operating parts.
    premium = _num(c, "owners_value_premium_gbp")
    premium_adjustment = _num(c, OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT)
    dcap = _num(p, "annual_capital_recovery_gbp") - (_num(c, "annual_capital_recovery_gbp") - premium)
    drr = _num(p, "annual_required_revenue_gbp") - (_num(c, "annual_required_revenue_gbp") - premium)
    dom = drr - dcap + oh + (teq_c - teq_p) - _num(p, "annual_state_overhead_gbp")
    cfdc_c, cfdc_p = _num(c, "carbon_removal_cfd_revenue_gbp"), _num(p, "carbon_removal_cfd_revenue_gbp")
    taken = -s * _num(p, "producer_rent_gbp")
    # F19 decision 7: the sign of the private rent is judged per cohort (carrier, support type, CfD
    # round, vintage) where the register carries them, so a calibrated cohort that breaks even as a
    # whole is not split into gross positive and negative halves; per row otherwise.
    keys = [c[k].astype(str) for k in COHORT_KEYS if k in c.columns]
    by_cohort = taken.groupby(keys).transform("sum") if len(keys) == len(COHORT_KEYS) else taken
    dldes = _num(c, "ldes_floor_topup_paid_gbp") - _num(p, "ldes_floor_topup_paid_gbp")
    # Decision 59 (review 2): on a separately financed co-owned row the state's part is on its own cost
    # and floor; the register's adjustments move it from the co-owner (zero without the columns).
    a_cost, a_cap = _num(c, "co_ownership_state_adjustment_gbp"), _num(c, "co_ownership_state_capital_adjustment_gbp")
    a_ldes = _num(c, "co_ownership_ldes_state_adjustment_gbp")
    # FES 2050 decision 73: the CM design moves the rows' CM revenue (zero without it: every 2030 row's
    # case and private revenue are the same and the adjustment column is absent). The takeover above
    # gives the state its share of private's rent, CM included; these correct both owners to the design.
    cm_c, cm_p = _num(c, "capacity_market_revenue_gbp"), _num(p, "capacity_market_revenue_gbp")
    a_cm = _num(c, "capacity_market_state_part_adjustment_gbp")
    # 10 Nov 2026: rows with no rent (F28 small-scale plant) carry CM revenue the design moves; their
    # private owners receive the change (investor_ledger.CM_DESIGN_NO_RENT).
    no_rent_case = case["producer_rent_gbp"].isna() if "producer_rent_gbp" in case.columns \
        else pd.Series(True, index=case.index)
    private_rows = register.loc[register["case_id"].eq(private_id)].drop_duplicates("asset_id")
    no_rent_private = private_rows["producer_rent_gbp"].isna() if "producer_rent_gbp" in private_rows.columns \
        else pd.Series(True, index=private_rows.index)
    out = {
        "cm_design_state": float((s * cm_c + a_cm - s * cm_p).sum()),
        "cm_design_private": float(((1 - s) * cm_c - a_cm - (1 - s) * cm_p).sum()),
        "cm_design_no_rent": float(_num(case, "capacity_market_revenue_gbp")[no_rent_case].sum()
                                   - _num(private_rows, "capacity_market_revenue_gbp")[no_rent_private].sum()),
        # The same rows' CfD support: a case's strikes move it (the CfD line), the account's cuts do not.
        "cfd_no_rent": float(_num(case, "cfd_support_gbp")[no_rent_case].sum()
                             - _num(private_rows, "cfd_support_gbp")[no_rent_private].sum()),
        "state_capital": float((s * dcap + a_cap).sum()), "private_capital": float(((1 - s) * dcap - a_cap).sum()),
        "state_operating": float((s * dom - oh + (a_cost - a_cap)).sum()),
        "private_operating": float(((1 - s) * dom - (a_cost - a_cap)).sum()),
        # F19 decision 7: the private rent the state takes on, on the whole rent (the BECCS carbon
        # CfD included, so the pair no longer appears gross), split by its sign: rows whose private
        # owners earned above their required return (mostly the R1 auction rent) and rows they ran
        # below it (the relief of their shortfall: LDES under its floor, merchant solar and hydro).
        "takeover_positive": float(taken[by_cohort < 0.0].sum()),
        "takeover_negative": float(taken[by_cohort >= 0.0].sum()),
        "ldes_state": float((s * dldes + a_ldes).sum()), "ldes_private": float(((1 - s) * dldes - a_ldes).sum()),
        # Only the co-owners' part of the carbon CfD's change remains a separate flow.
        "carbon_cfd_private": float(((1 - s) * (cfdc_c - cfdc_p)).sum()),
        "tax_equivalent_private": float(((1 - s) * teq_c).sum()),
        # F27 item 1 (fix, 24 Oct 2026): the premium sits in the whole row's capital charge, so the row's
        # owners carried it pro rata through their rent. User, 27 Oct 2026: it is booked on the state's
        # share only (the register's adjustment moves the co-owners' part onto the state), so the state's
        # flow to the former owners is the whole premium and the co-owners' part is nil. A register without
        # the adjustment keeps the pro rata split.
        "owners_value_premium": float((s * premium - premium_adjustment).sum()),
        "owners_value_premium_coowners": float(((1 - s) * premium + premium_adjustment).sum()),
        "corporation_tax": float(_num(case, "corporation_tax_forgone_gbp").sum()
                                 - _num(private.reindex(case.index), "corporation_tax_forgone_gbp").sum()),
        # F29 (fix 8 Nov 2026): the systematic risk the state holds at the workbook's social price, which the
        # Exchequer ledger and the resource saving both carry (investor_ledger.SOCIALISED_RISK); it had no flow,
        # so every case with state-financed plant failed the Exchequer closure by exactly this amount.
        "socialised_risk": float(_num(case, "socialised_risk_gbp").sum()
                                 - _num(private.reindex(case.index), "socialised_risk_gbp").sum()),
    }
    if BATTERY_TOP_UP_COLUMN in case.columns:
        # F19 decision 5: the battery top-up line's move split by who owns the row, measured as the bill
        # measures the line (this case against private on the same register). A balancing group holds
        # its support lines at the market group's amounts, so the account's private-basis split differs
        # from its own line there by private's own move between the two months (14 Oct 2026).
        top_up = _num(case, BATTERY_TOP_UP_COLUMN) - _num(private.reindex(case.index), BATTERY_TOP_UP_COLUMN)
        share = _num(case, "ownership_share")
        out["battery_top_up_state"] = float((share * top_up).sum())
        out["battery_top_up_private"] = float(((1 - share) * top_up).sum())
    return out


# ── networks ─────────────────────────────────────────────────────────────────

def network_flows(flows: Flows, network: Mapping, treatment: str | None, incidence: str | None) -> None:
    """The wires: the wedge from resource, the premium to the vendors, the tax, the fee, the expected loss."""
    if not treatment or treatment == "private":
        return
    gross = float(network.get("gross_wedge_gbp", network.get("wedge_gbp", 0.0) + network.get("guarantee_expected_loss_gbp", 0.0)))
    premium, tax = float(network.get("premium_gbp", 0.0)), float(network.get("tax_gbp", 0.0))
    fee, retained = float(network.get("fee_gbp", 0.0)), float(network.get("retained_gbp", 0.0))
    loss = float(network.get("guarantee_expected_loss_gbp", 0.0))
    keeps = incidence == "return_retained"
    beneficiary = "exchequer" if keeps else "consumers"
    if treatment == "debt_guarantee":
        flows.add("network_guarantee_compression", "Cheaper network debt under the state guarantee",
                  "resource", beneficiary, gross, "guarantee")
        flows.add("network_guarantee_fee", "Guarantee fee the networks pay the Exchequer", "consumers", "exchequer",
                  fee if not keeps else 0.0, "guarantee")
        flows.add("network_guarantee_expected_loss", "Expected loss the Exchequer carries on guaranteed network debt",
                  "exchequer", "resource", loss, "guarantee")
        return
    flows.add("network_financing_wedge", "Cheaper finance on the networks' asset base (public rate against allowed return)",
              "resource", beneficiary, gross, "financing")
    flows.add("network_purchase_premium", "Premium over the asset base paid to the network vendors",
              beneficiary, "private_investors", premium, "acquisition")
    # F27 item 7: the vendors' return above the allowed return, now earned for the state.
    flows.add("network_excess_return", "Return above the allowed return the network vendors no longer earn",
              "private_investors", beneficiary, float(network.get("excess_return_gbp", 0.0)), "acquisition")
    # The tax allowance: forgone by the Exchequer, and on the bill unless the state keeps it.
    kept = keeps or (tax > 0 and retained >= tax - 1e-6)
    flows.add("network_tax_allowance", "Corporation-tax allowance a public owner of the wires does not pay",
              "exchequer", "consumers", 0.0 if kept else tax, "tax")


# ── the bill lines of a market case ──────────────────────────────────────────

BILL_LINE_PARTY = {
    "public_cost_recovery_contracting": ("exchequer", "The state's surplus returned on the wholesale line", "other"),
    "public_cfd_at_cost": ("exchequer", "The state's renewables contracted at cost: rent returned to bills", "other"),
    "public_surplus_rebate": ("exchequer", "The state's surplus rebated on bills", "other"),
    "public_surplus_offset_domestic_levies": ("exchequer", "The state's surplus offsetting domestic customers' levies", "levies"),
    "public_portfolio_shortfall_levy": ("exchequer", "Levy making good a shortfall on the state's portfolio", "deficit"),
    "public_state_deficit_levy": ("exchequer", "Levy recovering the state's own deficits", "deficit"),
    "capacity_market_public_financing_adjustment": ("exchequer", "The state's cheaper finance booked on the Capacity Market line", "financing"),
    "allocated_ccs_public_financing_adjustment": ("exchequer", "The state's cheaper finance booked on the CCS levy", "financing"),
    "new_build_contracts_at_cost": ("private_investors", "Merchant new build contracted at its cost", "other"),
    "pay_as_bid_ceiling": ("private_investors", "Generators paid their short-run cost (a bound)", "other"),
    # F27 item 10: the price cap's EBIT and headroom on the case's own costs, paid to suppliers.
    "supplier_costs_and_margin": ("private_investors", "Suppliers' EBIT and headroom on the case's own costs", "other"),
    # F27 item 6: existing plant paid its going-forward cost less its own margin, not the clearing price
    # (negative: the private owners pay it back to consumers).
    "capacity_market_cost_based_existing": ("private_investors", "Capacity Market: existing plant on cost-based "
                                            "contracts instead of the clearing price", "other"),
    "legacy_levy_ro_to_taxation": ("exchequer", "Domestic Renewables Obligation paid from general taxation", "levies"),
    "legacy_levy_fit_to_taxation": ("exchequer", "Domestic Feed-in Tariffs paid from general taxation", "levies"),
    "legacy_levy_whd_to_taxation": ("exchequer", "Domestic Warm Home Discount paid from general taxation", "levies"),
    "beccs_carbon_removal_levy": ("exchequer", "BECCS carbon-removal contract recovered on bills", "levies"),
}
NETWORK_BILL_LINES = ("tnuos_public_financing_adjustment", "duos_public_financing_adjustment")
#: OW10: Sizewell C's RAB return at the public rate, a real saving that reaches consumers.
SIZEWELL_BILL_LINE = "sizewell_c_rab_public_rate_adjustment"
#: F19 decision 5: the battery top-up at each owner's cost, split by who no longer receives it.
BATTERY_TOP_UP_BILL_LINE = "battery_revenue_top_up_public_financing_adjustment"
#: The register column the battery top-up line pays (support_lines_2050.COLUMN_BATTERY_TOP_UP).
BATTERY_TOP_UP_COLUMN = "battery_revenue_top_up_gbp"
#: F9: the 2030/31 T-4 re-cleared with the state's plant bidding at its own cost, split by which winners lose it.
CM_STATE_BIDS_BILL_LINE = "capacity_market_state_bids_at_cost"
#: F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA re-priced, part of the account's R1 cut.
DPA_BILL_LINE = "remaining_ccs_public_financing_adjustment"
# The resource label :func:`investor_ledger.dispatch_step_parties` gives the re-trading change.
RETRADING_ABROAD = "interconnector re-trading paid abroad"


def market_flows(flows: Flows, *, parties: Mapping, components: Mapping, account_case: Mapping | None,
                 network: Mapping, levers: Mapping) -> None:
    """One market-bill case: the register split, the bill lines and the wires (see the module)."""
    account = account_case or {}
    flows.add("state_finance", "Cheaper finance on the state's share of new plant", "resource", "exchequer",
              components["state_capital"], "financing")
    flows.add("coowner_finance", "Cheaper finance on the co-owners' share of joint ventures", "resource",
              "private_investors", components["private_capital"], "financing")
    if parties.get("co_investment_transfer_gbp"):
        # Decision 68: the co-investors' rate reduction is backed by the state's implicit guarantee, a cost to
        # the Exchequer, not a resource saving.
        flows.add("co_investment_implicit_guarantee", "Co-investors' rate reduction backed by the state's implicit "
                  "guarantee", "exchequer", "resource", float(parties["co_investment_transfer_gbp"]), "financing")
    flows.add("state_operating", "Operating costs of state plant: the holding company and any public O&M premium",
              "resource", "exchequer", components["state_operating"], "other")
    flows.add("coowner_operating", "Operating cost changes on the co-owners' share", "resource", "private_investors",
              components["private_operating"], "other")
    flows.add("corporation_tax_forgone", CT_FLOW_LABEL,
              "exchequer", "resource", components["corporation_tax"], "tax")
    # F29: off the bill, no cash moves; a cost to the Exchequer and to the resource saving alike.
    flows.add("socialised_systematic_risk", "Systematic risk taxpayers hold on the state's plant, at the workbook's "
              "social price (economic rate less cash rate; F29)", "exchequer", "resource",
              components.get("socialised_risk", 0.0), "financing")
    # F19 decision 7: the takeover net of the BECCS carbon CfD, split into its two signs.
    flows.add("takeover_of_plant_rent", "The state takes over the private owners' rent above their required return "
              "(mostly the auction rent on new contracts)", "exchequer", "private_investors",
              components.get("takeover_positive", 0.0), "other")
    flows.add("takeover_relief", "The state takes on plant earning less than private owners require (relief of "
              "their shortfall)", "exchequer", "private_investors", components.get("takeover_negative", 0.0), "other")
    flows.add("owners_value_premium", "Pre-2025 contracted plant bought at its owners' value: the state's share of "
              "the premium over cost paid to the former owners", "exchequer", "private_investors", components.get("owners_value_premium", 0.0),
              "acquisition")
    flows.add("carbon_removal_cfd", "Carbon-removal contract payments: the co-owners' share", "exchequer",
              "private_investors", components["carbon_cfd_private"], "other")
    flows.add("coowner_tax_equivalent", "Tax equivalents on the co-owners' share", "private_investors", "exchequer",
              components["tax_equivalent_private"], "tax")
    flows.add("ldes_floor_state", "Long-duration storage floor paid to state stores", "consumers", "exchequer",
              components["ldes_state"], "other")
    flows.add("ldes_floor_private", "Long-duration storage floor paid to private stores", "consumers",
              "private_investors", components["ldes_private"], "other")
    lines = dict(parties.get("bill_lines_gbp") or {})
    cut = float(account.get("r1_bill_cut_gbp") or 0.0)
    state_cut = float(account.get("r1_cut_borne_by_state_gbp") or 0.0)
    # F27 item 1 (24 Oct 2026): the account's R1 cut includes Net Zero Teesside's DPA re-priced on its
    # own row, so the strike-cut flows below carry both lines.
    cfd_move = float(lines.get("cfd", 0.0)) + float(lines.get(DPA_BILL_LINE, 0.0))
    # FES 2050 decision 54: a case on another private rate path cuts every post-AR7 contract to that
    # path's strike too (zero in 2030 and in every flat-path case).
    path_cut = float(account.get("rate_path_cut_gbp") or 0.0)
    path_state = float(account.get("rate_path_cut_borne_by_state_gbp") or 0.0)
    # 10 Nov 2026: the CfD line also moves the support of rows with no rent (F28 small-scale plant, private),
    # which the account does not carry.
    cfd_no_rent = float(components.get("cfd_no_rent", 0.0))
    if abs(cfd_move + cut + path_cut - cfd_no_rent) > TOLERANCE_GBP:
        raise ValueError(f"{flows.case_id}: the CfD line moves {cfd_move:,.0f} but the account cuts {cut + path_cut:,.0f}"
                         f" and the rows with no rent move {cfd_no_rent:,.0f}")
    if cfd_no_rent:
        flows.add("strike_no_rent_private", "Contract strikes on small-scale plant outside the ownership register "
                  "(F28): private owners", "private_investors", "consumers", -cfd_no_rent, "contract re-pricing")
    if path_cut:
        flows.add("strike_rate_path_state", "Contracts struck at the rate path's own strike: the state's plant",
                  "exchequer", "consumers", path_state, "contract re-pricing")
        flows.add("strike_rate_path_private", "Contracts struck at the rate path's own strike: private owners",
                  "private_investors", "consumers", path_cut - path_state, "contract re-pricing")
    flows.add("strike_cut_state", "Contract strike cut on the state's plant", "exchequer", "consumers", state_cut,
              "contract re-pricing")
    flows.add("strike_cut_coowners", "Contract strike cut borne by private co-owners", "private_investors", "consumers",
              cut - state_cut, "contract re-pricing")
    # A line that lowers the bill is paid from resource to consumers (the flow reverses its sign).
    flows.add(SIZEWELL_BILL_LINE, "Cheaper finance on Sizewell C's asset base (public rate against its regulated "
              "return)", "consumers", "resource", float(lines.get(SIZEWELL_BILL_LINE, 0.0)), "financing")
    # F19 decision 5: a lower battery top-up comes off the state's batteries (the Exchequer) and off
    # the co-owners' share of a joint venture's (private investors), each at its own required revenue.
    battery = float(lines.get(BATTERY_TOP_UP_BILL_LINE, 0.0))
    battery_state = (float(components["battery_top_up_state"]) if "battery_top_up_state" in components
                     else float(account.get("battery_top_up_change_state_gbp") or 0.0)) if battery else 0.0
    flows.add("battery_top_up_state", "Battery top-up at the state's own cost (its batteries need less)",
              "consumers", "exchequer", battery_state, "financing")
    flows.add("battery_top_up_coowners", "Battery top-up at a joint venture's own cost (its co-owners' share)",
              "consumers", "private_investors", battery - battery_state, "financing")
    # F9: a lower clearing price is paid less to every 2030/31 winner: the state's own plant (the Exchequer)
    # and private winners, pro rata on the award (a transfer to consumers).
    cm_bids = float(lines.get(CM_STATE_BIDS_BILL_LINE, 0.0))
    cm_bids_state = (cm_bids * float(((account.get("capacity_market_state_bids") or {}).get("state_part_share")) or 0.0)
                     if cm_bids else 0.0)
    flows.add("capacity_market_state_bids_state", "Capacity Market at the re-cleared price: the state's own winners",
              "consumers", "exchequer", cm_bids_state, "other")
    flows.add("capacity_market_state_bids_private", "Capacity Market at the re-cleared price: private winners",
              "consumers", "private_investors", cm_bids - cm_bids_state, "other")
    handled = {"cfd", "ldes_cap_and_floor", SIZEWELL_BILL_LINE, BATTERY_TOP_UP_BILL_LINE, CM_STATE_BIDS_BILL_LINE,
               DPA_BILL_LINE, *NETWORK_BILL_LINES}
    if CM_DESIGN_LINE in lines:
        # FES 2050 decision 73 (2050 market cases only; F9's line above is off in a vintaged register).
        cm_design_flows(flows, float(lines[CM_DESIGN_LINE]), components, parties)
        handled.add(CM_DESIGN_LINE)
    for line, amount in lines.items():
        if line in handled or abs(float(amount)) < 1e-9:
            continue
        if line not in BILL_LINE_PARTY:
            raise ValueError(f"{flows.case_id}: bill line {line} has no party in the flow table")
        party, label, category = BILL_LINE_PARTY[line]
        # A line that raises the bill is paid by consumers to its party.
        flows.add(line, label, "consumers", party, float(amount), category)
    network_flows(flows, network, levers.get("networks"), levers.get("network_incidence"))


#: FES 2050 decision 73: the Capacity Market design's bill row.
CM_DESIGN_LINE = "capacity_market_design_adjustment"


def cm_design_flows(flows: Flows, amount: float, components: Mapping, parties: Mapping) -> None:
    """FES 2050 decision 73: the CM line's move, from consumers to each payee it no longer pays (or pays less).

    The state's plant (its CM revenue, to the Exchequer), the register's co-owners and private plant,
    the payees outside the register (I&C DSR, interconnectors, unmodelled plant: investors) and V2G
    owners and residential DSR (households: consumers paying consumers, shown so the transfer is
    visible; it nets to zero for consumers, whose column carries it as their income).
    """
    split = parties.get("capacity_market_design_split_gbp") or {}
    outside = float(split.get("outside_register_other_change_gbp", 0.0))
    households = float(split.get("household_change_gbp", 0.0))
    no_rent = components.get("cm_design_no_rent", 0.0)
    parts = (components.get("cm_design_state", 0.0) + components.get("cm_design_private", 0.0) + no_rent
             + outside + households)
    if abs(parts - amount) > max(TOLERANCE_GBP, 1e-9 * abs(amount)):
        raise ValueError(f"{flows.case_id}: the Capacity Market design line moves {amount:,.0f}; its payees "
                         f"{parts:,.0f}")
    flows.add("cm_design_state", "Capacity Market: the state's plant leaves the auction (decision 73)",
              "consumers", "exchequer", components.get("cm_design_state", 0.0), "other")
    flows.add("cm_design_register_private", "Capacity Market: co-owners and private plant at the residual price",
              "consumers", "private_investors", components.get("cm_design_private", 0.0), "other")
    if no_rent:
        flows.add("cm_design_no_rent", "Capacity Market: small-scale plant outside the ownership register (F28), "
                  "at the residual price", "consumers", "private_investors", no_rent, "other")
    flows.add("cm_design_outside_register", "Capacity Market: I&C DSR, interconnectors and plant outside the register",
              "consumers", "private_investors", outside, "other")
    flows.add("cm_design_households", "Capacity Market: V2G owners and residential DSR (households)",
              "consumers", "consumers", households, "other")


# ── acquisition-tariff cases ─────────────────────────────────────────────────

def _foreign(label: str) -> bool:
    return "abroad" in label or "foreign" in label


def tariff_flows(flows: Flows, *, parties: Mapping, network: Mapping, levers: Mapping) -> None:
    """One tariff case: each item of the tariff's register ledger, with its party, and the non-bill transfers."""
    items = list(parties.get("bill_items") or [])
    for party, label, amount in items:
        flow_id = "tariff:" + label[:60]
        if party == "investor":
            category = "acquisition" if ("acquisition" in label or "annuity" in label or "margin" in label) else "other"
            flows.add(flow_id, label, "consumers", "private_investors", amount, category)
        elif party == "resource":
            flows.add(flow_id, label, "consumers", "foreign" if _foreign(label) else "resource", amount,
                      "financing" if "capital" in label else "other")
        elif party == "exchequer":
            category = "tax" if "tax" in label else ("levies" if "taxation" in label else "other")
            flows.add(flow_id, label, "consumers", "exchequer", amount, category)
        elif party == "network":
            continue
        elif party == "household":
            # FES 2050 decision 73: the targeted contract's payment to V2G owners and residential DSR.
            flows.add(flow_id, label, "consumers", "consumers", amount, "other")
        else:
            raise ValueError(f"{flows.case_id}: tariff item {label!r} has party {party!r}")
    bill_labels = {label for _, label, _ in items}
    for label, amount in (parties.get("investor_parts_gbp") or {}).items():
        if label in bill_labels or label.startswith("networks:"):
            continue
        category = "tax" if "corporation tax" in label else "acquisition"
        flows.add("exchequer_to_investors:" + label[:50], label, "exchequer", "private_investors", amount, category)
    from scripts.public_ownership.investor_ledger import SOCIALISED_RISK  # local: keep this module light

    for label, amount in (parties.get("resource_parts_gbp") or {}).items():
        if label.startswith("less the corporation tax"):
            flows.add("corporation_tax_forgone", CT_FLOW_LABEL, "exchequer", "resource", -float(amount), "tax")
        elif label == SOCIALISED_RISK:
            # F29 (fix 8 Nov 2026): the ledger's resource part is minus the case's systematic risk against private;
            # the Exchequer bears it, as in market_flows.
            flows.add("socialised_systematic_risk", "Systematic risk taxpayers hold on the state's plant, at the "
                      "workbook's social price (economic rate less cash rate; F29)", "exchequer", "resource",
                      -float(amount), "financing")
    network_flows(flows, network, levers.get("networks"), levers.get("network_incidence"))


# ── balancing and institution steps ──────────────────────────────────────────

def dispatch_step_flows(flows: Flows, step: Mapping, carbon_change_gbp: float) -> None:
    """The bid stack on the private fleet: bill lines and resource to generators, carbon to the Exchequer.

    Interconnector re-trading sits inside the constraint line but is paid abroad, as
    :func:`investor_ledger.dispatch_step_parties` books it, so that part of the line
    goes to ``foreign`` rather than to generators.
    """
    resource_parts = step.get("resource_parts_gbp") or {}
    retrading = -float(resource_parts.get(RETRADING_ABROAD, 0.0))
    for line, amount in (step.get("bill_lines_gbp") or {}).items():
        if line == "bsuos_constraints" and retrading:
            flows.add("bid_stack_retrading", "Balancing-bid stack: interconnector re-trading paid abroad",
                      "consumers", "foreign", retrading, "other")
            amount = float(amount) - retrading
        flows.add("bid_stack:" + line, f"Balancing-bid stack: {line.replace('_', ' ')} paid to generators",
                  "consumers", "private_investors", amount, "other")
    for label, amount in resource_parts.items():
        if label == RETRADING_ABROAD:
            continue  # booked above, on the constraint line it rides on
        flows.add("bid_stack_resource:" + label[:40], f"Balancing-bid stack: {label}",
                  "resource", "foreign" if _foreign(label) else "private_investors", amount, "other")
    flows.add("bid_stack_carbon", "Balancing-bid stack: carbon paid by generators", "private_investors", "exchequer",
              carbon_change_gbp, "carbon")


def institution_step_flows(flows: Flows, step: Mapping, carbon_change_gbp: float) -> None:
    """Central dispatch against its I1 twin: every moved line is resource (or abroad) and carbon."""
    for label, amount in (step.get("resource_parts_gbp") or {}).items():
        flows.add("institution:" + label[:40], f"Central dispatch: {label}",
                  "foreign" if _foreign(label) else "resource", "consumers", amount, "other")
    flows.add("institution_carbon", "Central dispatch: carbon receipts", "consumers", "exchequer", carbon_change_gbp,
              "carbon")


def vat_flow(flows: Flows, vat_change_gbp: float) -> None:
    flows.add("domestic_vat", "VAT on domestic bills", "consumers", "exchequer", vat_change_gbp, "VAT")


# ── the checks ───────────────────────────────────────────────────────────────

def net_by_party(rows: Sequence[Mapping]) -> dict[str, float]:
    out = {party: 0.0 for party in PARTIES}
    for row in rows:
        out[row["payee"]] += float(row["gbp"])
        out[row["payer"]] -= float(row["gbp"])
    return out


def check(rows: Sequence[Mapping], ledger_row: Mapping, tolerance: float = TOLERANCE_GBP) -> dict:
    """The three closure checks for one case; refuses, naming the case and the gap, unless each is within GBP 1."""
    case_id = ledger_row["case_id"]
    net = net_by_party(rows)
    out = {"case_id": case_id}
    for party, column in COLUMN_BY_PARTY.items():
        gap = net[party] - float(ledger_row[column])
        out[f"{party}_gap_gbp"] = gap
        if abs(gap) > max(tolerance, 1e-12 * abs(float(ledger_row[column]))):
            raise ValueError(f"{case_id}: the itemised flows give {party} GBP {net[party]:,.2f}, the ledger "
                             f"GBP {float(ledger_row[column]):,.2f} (gap {gap:,.2f}); a flow is missing or doubled")
    # A transfer leaves one party and reaches another, so across the parties it nets to
    # zero; a flow written with an unknown or repeated party would break this.
    real = [r for r in rows if r["payer"] in REAL_PARTIES and r["payee"] in REAL_PARTIES]
    transfer_net = sum(net_by_party(real)[p] for p in REAL_PARTIES)
    out["transfer_net_gbp"] = transfer_net
    if abs(transfer_net) > tolerance:
        raise ValueError(f"{case_id}: transfers between the parties do not net to zero (GBP {transfer_net:,.2f})")
    resource = -(net["resource"] + net["foreign"])
    gap = resource - float(ledger_row["resource_saving_vs_private_gbp"])
    out["resource_gap_gbp"] = gap
    if abs(gap) > max(tolerance, 1e-12 * abs(resource)):
        raise ValueError(f"{case_id}: the resource flows sum to GBP {resource:,.2f}, the resource saving is "
                         f"GBP {float(ledger_row['resource_saving_vs_private_gbp']):,.2f}")
    return out


def frame(rows: Sequence[Mapping]) -> pd.DataFrame:
    return pd.DataFrame([{k: r[k] for k in FLOW_COLUMNS} for r in rows], columns=FLOW_COLUMNS)
