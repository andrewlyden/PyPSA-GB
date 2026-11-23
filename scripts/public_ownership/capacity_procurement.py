"""F27 item 6: capacity without an auction (lever ``capacity_procurement``, user decisions 20 and 24 Oct 2026).

Under ``cost_based_existing`` every existing unit that holds a calendar-2030 Capacity
Market receipt is taken to be needed for security and is paid a cost-based contract
instead of the clearing price: its going-forward fixed cost less its own wholesale and
balancing margin, floored at zero.  New capacity is still auctioned or state-built, so
the new-build rows keep their receipts.  It is a capacity-procurement choice ownership
makes easier, never a resource saving: the plant, its output and its costs are
unchanged, and only who keeps the clearing price above the unit's missing money moves.

* **Which units.** Register rows whose ``finance_vintage`` is in
  ``existing_finance_vintages`` (``legacy``) with ``capacity_market_revenue_gbp > 0``.
* **Going-forward fixed cost.** The register's own fixed charges on the row,
  ``annual_fixed_om_gbp``: site fixed O&M (``technology_costs.csv``, DESNZ Electricity
  Generation Costs and the crosswalk's sources), generator TNUoS and insurance (item
  D4(a)).  Every technology the register costs has one, so pumped hydro, hydro,
  energy from waste, nuclear and batteries are priced on the same basis as gas.  For
  CCGT it is 17.8 GBP/kW-yr (2024 money) against 18.6 (2025 money) in the Capacity
  Market builder's DESNZ gas table (``merchant_revenue_adequacy.csv``).  A row the
  register cannot cost (no fixed-cost row: oil) keeps its receipt.
* **Own margin.** ``market_revenue_gbp`` (wholesale revenue plus the balancing
  settlement) less fuel and carbon and variable O&M.  Support-line income
  (ancillary services, gate fees) is not in it, as the design says: the contract
  nets the unit's wholesale and balancing margins only, which is the conservative
  reading (energy from waste keeps its whole receipt).
* **Payment.** ``max(0, going-forward fixed cost - own margin)``, capped at the
  unit's own receipt (``cap_at_auction_receipt``): a unit that cleared the auction
  accepted that price, so a cost-based contract never pays it more.
* **P2 (market bill).** The bill's Capacity Market line falls by the saving on the
  privately owned share of each existing unit (a state-owned share keeps its
  receipt, which the state's own surplus rule already recycles), on its own row
  (``capacity_market_cost_based_existing``) recovered as the Capacity Market line is.
  The register keeps every receipt at the central price ("as now", as F9 does), so
  the saving is a named transfer from private owners in the investor ledger.  Where F9
  re-clears the 2030/31 T-4 in the same case, the units' receipts are first scaled by
  F9's calendar-2030 share change (``f9_scale``), so the part F9's own row already
  takes off them is not taken twice (pro rata, as F9 itself books its change).
* **P3 (acquisition tariff).** The tariff pays no Capacity Market agreement to plant
  the state owns, so the bill line does not move; the lever lowers the purchase
  price instead: the legacy valuation values each existing unit with the cost-based
  payment in place of its Capacity Market income (struck on the 2030 margin and held
  for the unit's life, as the income it replaces is).  That is a lower price for the
  vendors only if the contract regime is announced before the purchase.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

#: Lever name and values (scripts/public_ownership/levers.py declares the lever).
AUCTION, COST_BASED_EXISTING = "auction", "cost_based_existing"
#: Key of the per-case block in the public account summary.
CONFIG_KEY = "capacity_procurement"
#: The bill row (recovered as the Capacity Market line: INHERITED_LINE_RECOVERY).
LINE = "capacity_market_cost_based_existing"
#: ``public_ownership.lever_parameters`` key; calculate_financing_scenarios.LEVER_PARAMETER_DEFAULTS holds these.
PARAMETER_KEY = "capacity_cost_based_existing"
DEFAULT_PARAMETERS: dict = {
    # Register finance vintages that are "existing" capacity (the rest is new build, still auctioned).
    "existing_finance_vintages": ["legacy"],
    # A cost-based contract never pays a unit more than the receipt it cleared the auction at.
    "cap_at_auction_receipt": True,
}


def parameters(block: Mapping | None) -> dict:
    """``lever_parameters.capacity_cost_based_existing`` over the defaults, checked."""
    block = dict(block or {})
    unknown = sorted(set(block) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise ValueError(f"public_ownership.lever_parameters.{PARAMETER_KEY} takes {sorted(DEFAULT_PARAMETERS)}; "
                         f"unknown {unknown}")
    out = {**DEFAULT_PARAMETERS, **block}
    vintages = [str(v) for v in (out["existing_finance_vintages"] or [])]
    if not vintages:
        raise ValueError(f"public_ownership.lever_parameters.{PARAMETER_KEY}.existing_finance_vintages is empty")
    if not isinstance(out["cap_at_auction_receipt"], bool):
        raise ValueError(f"public_ownership.lever_parameters.{PARAMETER_KEY}.cap_at_auction_receipt must be true or false")
    return {"existing_finance_vintages": vintages, "cap_at_auction_receipt": out["cap_at_auction_receipt"]}


def _num(rows: pd.DataFrame, column: str, default=0.0) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(default, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def cost_based_payments(rows: pd.DataFrame, params: Mapping | None = None, receipt_scale: float = 1.0) -> pd.DataFrame:
    """Per row (indexed like ``rows``): the receipt, the cost-based payment and the saving.

    ``receipt_scale`` scales each receipt first (F9's share change on the same case;
    1.0 elsewhere).  Rows that are not existing, or hold no receipt, pay their receipt
    (saving zero); a row without a going-forward fixed cost keeps its receipt.
    """
    params = parameters(params)
    receipt = _num(rows, "capacity_market_revenue_gbp").fillna(0.0) * float(receipt_scale)
    vintage = (rows["finance_vintage"].astype(str) if "finance_vintage" in rows.columns
               else pd.Series("", index=rows.index))
    existing = vintage.isin(params["existing_finance_vintages"]) & receipt.gt(0.0)
    fixed = _num(rows, "annual_fixed_om_gbp", np.nan)
    margin = (_num(rows, "market_revenue_gbp").fillna(0.0) - _num(rows, "fuel_carbon_gbp").fillna(0.0)
              - _num(rows, "annual_variable_om_gbp").fillna(0.0))
    costed = existing & fixed.notna()
    missing = (fixed - margin).clip(lower=0.0)
    if params["cap_at_auction_receipt"]:
        missing = missing.clip(upper=receipt)
    payment = receipt.where(~costed, missing.fillna(0.0))
    return pd.DataFrame({
        "existing": existing, "costed": costed,
        "capacity_market_receipt_gbp": receipt,
        "going_forward_fixed_cost_gbp": fixed.where(existing),
        "own_margin_gbp": margin.where(existing),
        "cost_based_payment_gbp": payment,
        "saving_gbp": receipt - payment,
    }, index=rows.index)


def valuation_rows(rows: pd.DataFrame, params: Mapping | None = None) -> pd.DataFrame:
    """P3: the rows with each existing unit's Capacity Market income replaced by its cost-based payment."""
    out = rows.copy()
    out["capacity_market_revenue_gbp"] = cost_based_payments(rows, params)["cost_based_payment_gbp"].to_numpy()
    return out


def f9_scale(account_case: Mapping | None, private_account: Mapping | None) -> float:
    """``1 + F9's calendar-2030 cost change share`` on this case against private's (1.0 without F9)."""
    def share(block):
        return float(((block or {}).get("capacity_market_state_bids") or {}).get("calendar_2030_cost_change_share")
                     or 0.0)
    return 1.0 + share(account_case) - share(private_account)


def account_result(private_rows: pd.DataFrame, case_rows: pd.DataFrame, params: Mapping | None = None,
                   receipt_scale: float = 1.0) -> dict:
    """One market case's saving (system GBP, model price year) on the privately owned share of existing units.

    The payment is a property of the unit, so it is struck on the private case's rows
    (``private_rows``); ``case_rows`` give the state's ``ownership_share`` per asset.
    """
    params = parameters(params)
    private = private_rows.drop_duplicates("asset_id").set_index("asset_id")
    terms = cost_based_payments(private, params, receipt_scale)
    share = (_num(case_rows.drop_duplicates("asset_id").set_index("asset_id"), "ownership_share")
             .reindex(private.index).fillna(0.0).clip(lower=0.0, upper=1.0))
    terms["private_share"] = 1.0 - share
    existing = terms.loc[terms["existing"]]
    saving_private = float((existing["saving_gbp"] * existing["private_share"]).sum())
    carriers = private.loc[existing.index, "carrier"].astype(str) if "carrier" in private.columns else None
    by_carrier = {}
    if carriers is not None:
        for carrier, part in existing.groupby(carriers):
            by_carrier[str(carrier)] = {
                "capacity_market_receipt_gbp": float(part["capacity_market_receipt_gbp"].sum()),
                "cost_based_payment_gbp": float(part["cost_based_payment_gbp"].sum()),
                "saving_gbp": float((part["saving_gbp"] * part["private_share"]).sum()),
            }
    return {
        "lever_value": COST_BASED_EXISTING,
        "receipt_scale_f9": float(receipt_scale),
        "existing_units": int(len(existing)),
        "existing_units_without_fixed_cost": int((~existing["costed"]).sum()),
        "capacity_market_receipt_existing_gbp": float(existing["capacity_market_receipt_gbp"].sum()),
        "cost_based_payment_existing_gbp": float(existing["cost_based_payment_gbp"].sum()),
        "saving_gbp": float(existing["saving_gbp"].sum()),
        # The bill row: the privately owned share (a state share keeps its receipt).
        "saving_private_owners_gbp": saving_private,
        "saving_state_share_kept_gbp": float(existing["saving_gbp"].sum()) - saving_private,
        "by_carrier": by_carrier,
        "parameters": params,
    }


def method_note(result: Mapping) -> str:
    return (f"F27 item 6: the {result['existing_units']} existing units holding calendar-2030 Capacity Market "
            f"receipts (GBP {result['capacity_market_receipt_existing_gbp'] / 1e6:,.0f}m) are paid a cost-based "
            f"contract instead: going-forward fixed cost (the register's fixed O&M, TNUoS and insurance) less their "
            f"own wholesale and balancing margin, floored at zero and capped at the receipt "
            f"(GBP {result['cost_based_payment_existing_gbp'] / 1e6:,.0f}m). The privately owned share of the "
            f"difference, GBP {result['saving_private_owners_gbp'] / 1e6:,.0f}m, comes off the Capacity Market line, "
            "recovered as that line is: a transfer from plant owners, a capacity-procurement choice ownership makes "
            "easier. New capacity is still auctioned.")
