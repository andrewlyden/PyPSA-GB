"""The order in which the wholesale curtails zero-cost renewables (26 Sep 2026).

Every CfD unit and every merchant renewable reached the wholesale at a marginal
cost of exactly 0, so whenever supply exceeded demand the price cleared at 0 and
which of ~9,000 tied units gave way was left to the barrier solver. It spread the
curtailment smoothly but loaded it onto small units (January: 47% of output under
5 MW against 20% over 500 MW), and it decided how much CfD top-up is paid: 25.7
TWh of AR4+ and 12.4 TWh of pre-AR4 CfD output was curtailed at a zero price in
2030, where real plant would have kept generating.

``contract_aware`` gives each unit the price at which it would really back off
under its support terms (see revenue_tracking.negative_price_eligibility):

* merchant renewables (``merchant_carriers``: wind, solar, marine and hydro) at
  their as-built cost plus a small variable cost
  (``merchant_variable_cost_gbp_per_mwh``, +0.5): they lose only that cost when
  curtailed, so they give way first and the price they set stays positive.
  Hydro is included because at 0 it tied with AR4+ and spent reservoir water in
  surplus hours (a Dec 2030 two-day solve: +25 GWh of large hydro, displacing
  AR4+ wind);
* AR4 and later CfDs at ``ar4_plus_bid_gbp_per_mwh`` (0.0, decision D1 of 27 Sep
  2026): their contract pays the top-up whenever the price is not negative and
  nothing once it is, so zero is their true reservation price. They give way
  next, and an hour they set clears at zero, where the top-up is still paid;
* AR1-AR3 CfDs, Investment Contracts and Hinkley Point C at minus their strike:
  paid through negative prices (AR2-AR3 unless six hours run negative; AR1 always), so they
  give way last, alongside ROC plant at minus its certificate value.

The treatment before 27 Sep 2026 (AR4+ at -1.0, merchant at 0.0), which sent every
hour AR4+ set below zero and so stripped its top-up, is kept as a sensitivity:
set ``ar4_plus_bid_gbp_per_mwh: -1.0`` and ``merchant_variable_cost_gbp_per_mwh: 0.0``.

Thermal CfDs keep their fuel-cost treatment (``cfd_dispatch_cost_exempt_carriers``).
Applied to the market wholesale only: a network-constrained central dispatch
(``copperplate_mode: none``) dispatches on cost and is left as built. The
balancing stage reloads the built network, so its offers are unchanged.
Defaults must match config/defaults/ market.wholesale.renewable_curtailment_order.
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.market.cfd_strike_table import contract_technology, strike_lookup
from scripts.market.revenue_tracking import POST_AR4_ROUNDS

MODES = ("as_built", "contract_aware")
#: Carriers whose CfD terms decide their curtailment order.
ORDERED_CARRIERS = frozenset({"wind_offshore", "wind_onshore", "solar_pv", "marine", "nuclear"})
#: Defaults of market.wholesale.renewable_curtailment_order (must match config/defaults/).
DEFAULT_AR4_PLUS_BID = 0.0
DEFAULT_MERCHANT_VARIABLE_COST = 0.5
DEFAULT_MERCHANT_CARRIERS = ("wind_offshore", "wind_onshore", "solar_pv", "marine", "large_hydro", "small_hydro")


def apply_curtailment_order(network, block: dict | None, strike_table, logger: logging.Logger,
                            copperplate_mode: str | None = None) -> pd.DataFrame | None:
    """Set CfD and merchant renewables' wholesale marginal cost by their support terms; return an audit or ``None``."""
    block = dict(block or {})
    mode = str(block.get("mode", "as_built"))
    if mode not in MODES:
        raise ValueError(f"market.wholesale.renewable_curtailment_order.mode must be one of {MODES}, not {mode!r}")
    if mode == "as_built":
        return None
    if str(copperplate_mode or "").lower() == "none":
        logger.info("Curtailment order: network-constrained dispatch on cost, contract-aware order not applied")
        return None
    ar4_bid = float(block.get("ar4_plus_bid_gbp_per_mwh", DEFAULT_AR4_PLUS_BID))
    if not -10.0 <= ar4_bid <= 0.0:
        raise ValueError(f"ar4_plus_bid_gbp_per_mwh must be zero or small and negative (-10 to 0), not {ar4_bid}")
    merchant_cost = float(block.get("merchant_variable_cost_gbp_per_mwh", DEFAULT_MERCHANT_VARIABLE_COST))
    if not 0.0 <= merchant_cost <= 5.0:
        raise ValueError(f"merchant_variable_cost_gbp_per_mwh must be small and non-negative (0 to 5), not {merchant_cost}")
    merchant_carriers = frozenset(str(c).strip() for c in
                                  (block.get("merchant_carriers") or DEFAULT_MERCHANT_CARRIERS))
    gens = network.generators
    support = gens.get("support_type", pd.Series("", index=gens.index)).fillna("").astype(str).str.strip()
    carrier = gens["carrier"].fillna("").astype(str).str.strip()
    rounds = gens.get("cfd_round", pd.Series("", index=gens.index)).fillna("").astype(str).str.strip()
    technology = contract_technology(gens)
    ordered = gens.index[support.eq("CfD") & carrier.isin(ORDERED_CARRIERS)]
    rows = []
    for name in ordered:
        cfd_round = rounds[name]
        if cfd_round in POST_AR4_ROUNDS:
            cost, basis = ar4_bid, "post_ar4_reservation_price"
        else:
            strike, key, _ = strike_lookup(strike_table, carrier[name], cfd_round, technology[name])
            if strike is None:
                raise ValueError(f"Curtailment order: no strike for {name} ({carrier[name]}, {cfd_round!r})")
            cost, basis = -float(strike), f"minus_strike:{key}"
        rows.append({"generator": name, "carrier": carrier[name], "cfd_round": cfd_round,
                     "as_built_marginal_cost": float(gens.at[name, "marginal_cost"]),
                     "wholesale_marginal_cost": cost, "basis": basis})
    merchant = gens.index[support.eq("merchant") & carrier.isin(merchant_carriers)]
    varying = getattr(network.generators_t, "marginal_cost", None)
    varying_cols = varying.columns if varying is not None else pd.Index([])
    if merchant_cost != 0.0 and len(merchant):
        for name in merchant:
            as_built = float(gens.at[name, "marginal_cost"])
            rows.append({"generator": name, "carrier": carrier[name], "cfd_round": "",
                         "as_built_marginal_cost": as_built,
                         "wholesale_marginal_cost": as_built + merchant_cost,
                         "basis": "merchant_plus_variable_cost"})
        # A time-varying merchant cost keeps its shape and takes the same adder.
        tv_merchant = varying_cols.intersection(merchant)
        if len(tv_merchant):
            network.generators_t.marginal_cost.loc[:, tv_merchant] = varying[tv_merchant] + merchant_cost
    audit = pd.DataFrame(rows)
    if not audit.empty:
        network.generators.loc[audit["generator"], "marginal_cost"] = audit["wholesale_marginal_cost"].to_numpy()
        # A time-varying cost would override the static one, so drop any for the CfD units.
        cfd_units = audit.loc[audit["basis"] != "merchant_plus_variable_cost", "generator"]
        varying = getattr(network.generators_t, "marginal_cost", None)
        if varying is not None and len(varying.columns):
            overlap = varying.columns.intersection(cfd_units)
            if len(overlap):
                network.generators_t.marginal_cost = varying.drop(columns=overlap)
        summary = audit.groupby("basis")["generator"].size()
        logger.info("Curtailment order (contract_aware): %d units re-priced for the wholesale "
                    "(AR4+ at %.2f, merchant %+.2f GBP/MWh): %s",
                    len(audit), ar4_bid, merchant_cost, summary.to_dict())
    return audit
