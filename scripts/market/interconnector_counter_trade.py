"""Interconnector counter-trading in the balancing stage: its price and its limits.

What this prices
----------------
When the balancing stage moves a cross-border cable off its Stage 1 (day-ahead)
schedule, NESO is counter-trading: it pays a trader or a neighbouring system
operator to change the flow. The IC1 supply curve already charges the difference
in foreign trade value (``market_coupling.interconnector_trade_cost``), but a
counter-trade also has to pay whoever turns plant up or down on the far side.
FTI for Ofgem (2023, Appendix A1.21 and A1.38, pp.276-279) prices that at
GBP 130/MWh in 2025 and GBP 100/MWh from 2030 ("slightly above the SRMC of
CCGT plants", since changing a flow "would likely require constraining-on gas
generators in a neighbouring country"), and notes that this matches NESO's
historical treatment: interconnectors are constrained on only when nothing
domestic is cheaper. Here the premium is added on top of the trade-value
difference, per MWh of absolute deviation from the Stage 1 schedule.

Pricing rule (``offer_pricing`` for moves that raise GB's import position, i.e.
less export or more import; ``bid_pricing`` for moves that lower it), added
3 Oct 2026:

* ``additive`` (default, the rule above): trade-value change plus the premium.
* ``all_in``: the premium is the whole price of the move. FTI's own table
  (Table 5-5, p.118; Figure A1-1, p.277) gives the interconnector's BM offer and
  bid as "Cost of reversing flow EUR 130 / EUR 100" and nothing else: there is no
  foreign marginal-value term, unlike hydrogen and other storage in the same
  table. So FTI's figure already pays for the foreign gas turned on, and adding
  it to the trade-value change counts the foreign energy twice (an export
  reduction costs the lost export bid plus 100). Under ``all_in`` the hourly
  premium on a move is ``max(premium - m, 0)``, ``m`` being the move's marginal
  trade value at the Stage 1 schedule (``market_coupling.
  counter_trade_direction_premiums``), so a move costs ``max(premium, m)``: the
  all-in price, or the foreign market price when that is dearer. NESO's CP30
  redispatch works the same way (CP30 Annex 2, p.5: wind constrained off "is
  most frequently replaced by changing the position of interconnector flows ...
  usually a reduction in exports ... When prices are higher in the connected
  markets, it is cheaper to run gas plants than to reduce exports or increase
  imports"): gas wins only where the connected market is dearer than gas.

Foreign-plant uplift (16 Oct 2026, default off): NESO/ESO's Long-term Market and
Network Constraint Modelling (Jan 2017, section 12, p.22) prices an interconnector
move as the spread plus the foreign marginal plant's own offer or bid, with "the same
bid and offer multipliers ... as the GB plant":

* eq. 6 (offer, the move adds GB supply): ``P_for(1-L) - P_GB + PlantOffer(1-L)``;
* eq. 1 (bid, the move removes GB supply): ``P_GB - P_for(1+L) - PlantBid(1+L)``.

User decision, 16 Oct 2026: the foreign marginal plant takes GB gas's multipliers.
Here, in the ESO-cost convention (positive = NESO pays; ``market_utils.
DERIVED_BID_CONVENTIONS``), with ``m`` the move's marginal trade value at the Stage 1
schedule (``market_coupling.marginal_trade_cost``; the losses ``L`` are inside it as
the cable's efficiency, and ``P_GB`` is paid or earned by the GB units the BM moves):

* an **offer** move (import up, or export down) costs ``m_up`` (the foreign ask bought,
  or the ``eta^2 x bid`` export income given up) plus
  ``s_offer x max(m_up, 0)``. It is ``(1 + s_offer) x m_up``, as GB gas offers at
  ``(1 + offer_markup) x SRMC``.
* a **bid** move (import down, or export up) costs ``m_down`` (negative: the ask NESO
  saves, or the export bid it earns) plus ``s_bid x max(-m_down, 0)``. That is
  ``-(1 - s_bid) x |m_down|``: NESO receives the foreign value less the plant's bid
  discount, as a GB gas bid hands back ``-(mc - bid_discount x |mc|)`` under
  ``net_saving``. The foreign plant turned down keeps ``s_bid`` of its saving, so the
  discount RAISES NESO's cost, by ``s_bid x |m_down|``.

Both uplifts are non-negative additions to NESO's cost; a move whose foreign value is
negative (``m_up < 0`` or ``m_down > 0``) gets none. Settings
(``interconnector_counter_trade``):

* ``foreign_offer_uplift_share`` / ``foreign_bid_uplift_share``: explicit shares;
* ``foreign_plant_multipliers: gb_gas``: take them from the scenario's own CCGT
  ``carrier_overrides`` (``offer_markup``, ``bid_discount``, falling back to
  ``default_offer_markup`` / ``default_bid_discount``), so the 0.6 lives in one place;
  it may not be combined with explicit shares;
* ``foreign_uplift_scale`` (default 1): multiplies both shares, so a calibration moves
  them in proportion (1 = the multipliers as set);
* ``foreign_uplift_share`` (retired name, kept as an alias): one share on both sides;
  it may not be combined with the keys above.

With ``premium 0`` and both rules ``additive`` the scale is the one interconnector
calibration lever of the NESO-consistent CP30 families. The uplift is a payment like
the premium: in the objective and in ``counter_trade_premium_cost``.

The premium is a payment, not a regulariser. NESO's counter-trades are
Balancing Services Adjustment Data trades recovered through BSUoS alongside the
bid-offer acceptances, so the premium sits in the balancing objective (it steers
the solve) and in the reported interconnector re-trading cost
(``counter_trade_premium_cost``, read by ``analyze_market`` and
``fti_settlement``), which the bill adds to its constraint line when
``bsuos_constraints.include_interconnector_retrading`` is true. It is not in
``constraint_costs.csv``, which covers accepted bids and offers of GB units only,
so the two lines stay additive and never double count.

What this limits
----------------
LCP Delta (March 2026, pp.47-48) reproduces present-day interconnector redispatch
with two levers: a flow limit, the most a cable may move from its day-ahead
position as a share of its capacity (200% being full export to full import), and
a minimum spread of GBP 21/MWh a trader must see before counter-trading. Its
counterfactual limits are Belgium 33%, Denmark 22%, France 45%, Germany 22%,
Netherlands 30%, Ireland 0% and Norway 0%. Here the flow limit bounds each
cross-border link's deviation, in both directions, to ``share x p_nom``, and the
spread rule (``min_spread_rule``) is either

* ``stage1_gate`` (default): in an hour whose Stage 1 spread
  ``|GB price - foreign price|`` is below the minimum (the foreign price being
  the ask when Stage 1 imports, the bid when it exports and the nearer of the
  two when it is idle) no trader takes a counter-trade that INCREASES the
  cable's ``|flow|``; a move back towards zero, the network shedding a flow it
  cannot carry, stays allowed inside the flow limit (up to the Stage 1 flow); or
* ``premium_floor``: the minimum spread is a floor on the per-MWh premium (a
  trader's minimum margin), which the FTI premium already exceeds.

Feasibility: the band is soft
-----------------------------
A hard band made BM windows infeasible (27 Sep 2026, network C with N-1 derates
and the 2024 boundary outage replay): when the network cannot carry a cable's
Stage 1 flow to or from its landing bus, nothing else can stand in for the
export (load shedding adds no demand at the landing bus), so the cable itself
must move further than its flow limit. The band therefore binds only the
*priced* counter-trade. Movement beyond it goes to ``Link-counter-trade-excess``
variables priced at ``beyond_limit_penalty_gbp_per_mwh`` (default 300, on top
of the 100 premium such a move also pays): above every real bid and offer the
CP30 balancing stage prices (dearest: gas engines ~229, OCGT ~211 at 1.29 x
SRMC in the Dec 2030 week; Investment Contract offshore CfD bid 200.84 at a
zero reference), so a within-band counter-trade plus any domestic action is
always preferred, and far below the 6,000 value of lost load and the
excluded-carrier penalty prices. It was 1,000 until 27 Sep 2026, which made
the objective, not the network, decide how hard a forced move was avoided.
It is an emergency cross-border action the solve takes only when nothing
domestic can resolve the constraint. Above it sit only oil peakers (491 at
1.29 x SRMC, ~1 MWh a week in the Dec and Jun 2030 test weeks) and the nuclear
restart override (999). That penalty is
objective-only and stripped from the reported BM cost
(``counter_trade_excess_penalty``); the counter-trade premium is still paid on
every MWh moved, inside the band or beyond it, and the excess volume is logged.
Where the balancing stage's own availability (``links_t.p_min_pu`` /
``p_max_pu``, which limit both directions during an outage) no longer admits the
Stage 1 position, the band itself widens by exactly the forced movement, so an
outage costs no penalty.

No wheeling (17 Oct 2026, default off)
--------------------------------------
With the flow limits off (``max_deviation_share`` 2.0) the balancing stage can export more on a northern cable
(NSL, Moyle, EWIC, MaresConnect) and import more on a southern one in the same hour: a path round the B8 boundary
through Europe that NESO's CP30 redispatch does not take (CP.18: IC bids 1.53 TWh a year against 13.83 of offers).
``no_wheeling: true`` forbids it: in each hour the counter-trades move GB's supply in one direction only
(``add_no_wheeling_constraints``; ``no_wheeling_formulation`` ``binary``, exact, a MILP with one binary per hour,
or ``hull``, its LP relaxation, which removes little). Forced moves beyond the band are not constrained.

Central dispatch
----------------
The same rules apply to a central-dispatch Stage 2 correcting its own day-ahead
schedule. A foreign counterparty does not care which institution scheduled GB's
side of the trade: the premium is a resource cost abroad (foreign gas turned on)
and the flow limit describes cross-border trading arrangements, not GB's
dispatch. The gate then reads the Stage 1 price the central run reports (its
demand-bus mean of nodal prices), a proxy for the GB end of each cable.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from scripts.interconnectors.market_coupling import (
    COUNTER_TRADE_BID_PREMIUM_COLUMN,
    COUNTER_TRADE_BID_RULE_COLUMN,
    COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN,
    COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN,
    COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN,
    COUNTER_TRADE_OFFER_RULE_COLUMN,
    COUNTER_TRADE_PREMIUM_COLUMN,
    COUNTER_TRADE_PRICING_RULES,
    counter_trade_direction_premiums,
    get_piecewise_interconnectors,
    link_price_tables,
    piecewise_market_prices,
)
from scripts.market.pricing_basis import priced_balancing

#: Static link column recording the flow limit each cross-border link was solved with.
COUNTER_TRADE_SHARE_COLUMN = "market_counter_trade_max_share"

#: Code defaults; must match ``market.balancing.interconnector_counter_trade`` in
#: config/defaults/base/market/balancing.yaml (on for every scenario since decision D4, 27 Oct 2026;
#: the CP30 source sets its own premium and pricing rules).
DEFAULT_COUNTER_TRADE: dict = {
    "enabled": True,
    "premium_gbp_per_mwh": 100.0,
    "max_deviation_share": 0.45,
    "max_deviation_share_by_country": {
        "Belgium": 0.33,
        "Denmark": 0.22,
        "France": 0.45,
        "Germany": 0.22,
        "Netherlands": 0.30,
        "Ireland": 0.0,
        "Northern Ireland": 0.0,
        "Norway": 0.0,
    },
    "min_spread_gbp_per_mwh": 21.0,
    "min_spread_rule": "stage1_gate",
    "beyond_limit_penalty_gbp_per_mwh": 300.0,
    "offer_pricing": "additive",
    "bid_pricing": "additive",
    "bid_premium_gbp_per_mwh": None,
    "foreign_plant_multipliers": None,
    "foreign_offer_uplift_share": 0.0,
    "foreign_bid_uplift_share": 0.0,
    "foreign_uplift_scale": 1.0,
    "no_wheeling": False,
    "no_wheeling_formulation": "binary",
    "no_wheeling_mip_gap": None,
}
#: ``no_wheeling_formulation`` values (module docstring, "No wheeling").
NO_WHEELING_FORMULATIONS = ("binary", "hull")
#: Upper bound on each effective foreign-plant share (after the scale): GB gas offers
#: sit at 1.6 x SRMC, so a mark-up above 100% of the foreign price is outside any
#: evidence; a bid discount of 100% would hand NESO nothing for turning plant down.
MAX_FOREIGN_UPLIFT_SHARE = 1.0
#: ``foreign_plant_multipliers`` sources: name -> the carrier whose BM multipliers
#: (``carrier_overrides``) the foreign marginal plant takes (user, 16 Oct 2026).
FOREIGN_PLANT_MULTIPLIER_SOURCES = {"gb_gas": "CCGT"}
#: Fallbacks for a source carrier without its own multipliers: the code defaults of
#: ``market.balancing.default_offer_markup`` / ``default_bid_discount`` (market_utils).
_FALLBACK_OFFER_MARKUP = 0.10
_FALLBACK_BID_DISCOUNT = 0.10
#: Retired single-share name, still read as an alias that sets both sides.
LEGACY_FOREIGN_UPLIFT_KEY = "foreign_uplift_share"


def resolve_foreign_uplift(raw: Mapping, balancing_config: Mapping | None) -> dict:
    """Effective foreign-plant shares ``{offer, bid, scale, source}`` (module docstring).

    ``raw`` is the scenario's ``interconnector_counter_trade`` block as written (before
    the code defaults), so explicit and derived settings can be told apart. The borrowed
    multipliers are read on the balancing stage's pricing basis (zero under ``at_cost``).
    """
    balancing_config = priced_balancing(dict(balancing_config or {}))
    legacy = raw.get(LEGACY_FOREIGN_UPLIFT_KEY)
    source = raw.get("foreign_plant_multipliers")
    # A share at 0 is the config/defaults/ value every merged config carries: not "set".
    explicit = {k: raw[k] for k in ("foreign_offer_uplift_share", "foreign_bid_uplift_share")
                if raw.get(k) is not None and float(raw[k]) != 0.0}
    if legacy is not None and (source is not None or explicit):
        raise ValueError(
            f"interconnector_counter_trade.{LEGACY_FOREIGN_UPLIFT_KEY} (one share on both sides, retired) cannot "
            "be combined with foreign_plant_multipliers or foreign_offer/bid_uplift_share"
        )
    if source is not None and explicit:
        raise ValueError(
            "interconnector_counter_trade.foreign_plant_multipliers sets both shares; do not also set "
            f"foreign_offer_uplift_share / foreign_bid_uplift_share (got {explicit})"
        )
    if source is not None:
        key = str(source).strip().lower()
        if key not in FOREIGN_PLANT_MULTIPLIER_SOURCES:
            raise ValueError(
                f"interconnector_counter_trade.foreign_plant_multipliers must be one of "
                f"{sorted(FOREIGN_PLANT_MULTIPLIER_SOURCES)} or null, got {source!r}"
            )
        carrier = FOREIGN_PLANT_MULTIPLIER_SOURCES[key]
        override = dict((balancing_config.get("carrier_overrides") or {}).get(carrier) or {})
        if str(override.get("mode", "markup")).lower() == "absolute":
            raise ValueError(f"foreign_plant_multipliers {key}: {carrier} is priced in absolute mode, "
                             "so it has no multipliers to lend")
        offer = override.get("offer_markup", balancing_config.get("default_offer_markup", _FALLBACK_OFFER_MARKUP))
        bid = override.get("bid_discount", balancing_config.get("default_bid_discount", _FALLBACK_BID_DISCOUNT))
        source_label = f"{key} ({carrier} offer_markup {float(offer):g}, bid_discount {float(bid):g})"
    elif legacy is not None:
        offer = bid = legacy
        source_label = f"{LEGACY_FOREIGN_UPLIFT_KEY} (alias, both sides)"
    else:
        offer = raw.get("foreign_offer_uplift_share", DEFAULT_COUNTER_TRADE["foreign_offer_uplift_share"])
        bid = raw.get("foreign_bid_uplift_share", DEFAULT_COUNTER_TRADE["foreign_bid_uplift_share"])
        source_label = "explicit"
    scale = raw.get("foreign_uplift_scale", DEFAULT_COUNTER_TRADE["foreign_uplift_scale"])
    scale = DEFAULT_COUNTER_TRADE["foreign_uplift_scale"] if scale is None else float(scale)
    if scale < 0.0:
        raise ValueError(f"interconnector_counter_trade.foreign_uplift_scale must be non-negative, got {scale}")
    offer = 0.0 if offer is None else float(offer)
    bid = 0.0 if bid is None else float(bid)
    out = {"offer": offer * scale, "bid": bid * scale, "scale": scale, "source": source_label,
           "offer_multiplier": offer, "bid_multiplier": bid}
    for side in ("offer", "bid"):
        if not 0.0 <= out[side] <= MAX_FOREIGN_UPLIFT_SHARE:
            raise ValueError(
                f"interconnector_counter_trade foreign {side} uplift share must lie in [0, "
                f"{MAX_FOREIGN_UPLIFT_SHARE}] after the scale, got {out[side]:g} "
                f"({side} multiplier {out[side + '_multiplier']:g} x scale {scale:g}; source {source_label})"
            )
    return out
MIN_SPREAD_RULES = ("stage1_gate", "premium_floor", "none")
#: Stage 1 flows within this many MW of zero count as idle for the gate.
IDLE_FLOW_MW = 1.0


def resolve_counter_trade_config(balancing_config: Mapping | None) -> dict:
    """The effective counter-trade settings, code defaults filled in and checked."""
    raw = dict((balancing_config or {}).get("interconnector_counter_trade") or {})
    cfg = {**DEFAULT_COUNTER_TRADE, **raw}
    cfg["max_deviation_share_by_country"] = dict(
        raw.get("max_deviation_share_by_country",
                DEFAULT_COUNTER_TRADE["max_deviation_share_by_country"]) or {}
    )
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["premium_gbp_per_mwh"] = float(cfg["premium_gbp_per_mwh"])
    bid_premium = cfg["bid_premium_gbp_per_mwh"]
    cfg["bid_premium_gbp_per_mwh"] = None if bid_premium is None else float(bid_premium)
    if cfg["bid_premium_gbp_per_mwh"] is not None and cfg["bid_premium_gbp_per_mwh"] < 0:
        raise ValueError("interconnector_counter_trade.bid_premium_gbp_per_mwh must be non-negative or null")
    cfg["min_spread_gbp_per_mwh"] = float(cfg["min_spread_gbp_per_mwh"])
    # Foreign-plant uplift: the effective (scaled) shares replace the settings as written.
    uplift = resolve_foreign_uplift(raw, balancing_config)
    cfg.pop(LEGACY_FOREIGN_UPLIFT_KEY, None)
    cfg["foreign_offer_uplift_share"] = uplift["offer"]
    cfg["foreign_bid_uplift_share"] = uplift["bid"]
    cfg["foreign_uplift_scale"] = uplift["scale"]
    cfg["foreign_uplift_source"] = uplift["source"]
    penalty = cfg["beyond_limit_penalty_gbp_per_mwh"]
    cfg["beyond_limit_penalty_gbp_per_mwh"] = None if penalty is None else float(penalty)
    if cfg["beyond_limit_penalty_gbp_per_mwh"] is not None and cfg["beyond_limit_penalty_gbp_per_mwh"] < 0:
        raise ValueError("interconnector_counter_trade.beyond_limit_penalty_gbp_per_mwh must be non-negative or null")
    cfg["max_deviation_share"] = None if cfg["max_deviation_share"] is None else float(cfg["max_deviation_share"])
    cfg["min_spread_rule"] = str(cfg["min_spread_rule"]).lower()
    if cfg["min_spread_rule"] not in MIN_SPREAD_RULES:
        raise ValueError(
            f"interconnector_counter_trade.min_spread_rule must be one of {MIN_SPREAD_RULES}, "
            f"got {cfg['min_spread_rule']!r}"
        )
    for key in ("offer_pricing", "bid_pricing"):
        cfg[key] = str(cfg[key]).lower()
        if cfg[key] not in COUNTER_TRADE_PRICING_RULES:
            raise ValueError(
                f"interconnector_counter_trade.{key} must be one of {COUNTER_TRADE_PRICING_RULES}, got {cfg[key]!r}"
            )
    cfg["no_wheeling"] = bool(cfg["no_wheeling"])
    cfg["no_wheeling_formulation"] = str(cfg["no_wheeling_formulation"]).lower()
    if cfg["no_wheeling_formulation"] not in NO_WHEELING_FORMULATIONS:
        raise ValueError(
            f"interconnector_counter_trade.no_wheeling_formulation must be one of {NO_WHEELING_FORMULATIONS}, "
            f"got {cfg['no_wheeling_formulation']!r}"
        )
    gap = cfg["no_wheeling_mip_gap"]
    cfg["no_wheeling_mip_gap"] = None if gap is None else float(gap)
    if cfg["no_wheeling_mip_gap"] is not None and not 0.0 <= cfg["no_wheeling_mip_gap"] < 1.0:
        raise ValueError("interconnector_counter_trade.no_wheeling_mip_gap must lie in [0, 1) or be null")
    if cfg["premium_gbp_per_mwh"] < 0 or cfg["min_spread_gbp_per_mwh"] < 0:
        raise ValueError("interconnector_counter_trade premium and minimum spread must be non-negative")
    shares = list(cfg["max_deviation_share_by_country"].values())
    if cfg["max_deviation_share"] is not None:
        shares.append(cfg["max_deviation_share"])
    if any(float(s) < 0 or float(s) > 2.0 for s in shares):
        raise ValueError("interconnector_counter_trade shares must lie in [0, 2] (2 = full export to full import)")
    return cfg


def effective_premium(cfg: Mapping) -> float:
    """Per-MWh premium the objective and the reports use."""
    premium = float(cfg["premium_gbp_per_mwh"])
    if cfg.get("min_spread_rule") == "premium_floor":
        premium = max(premium, float(cfg["min_spread_gbp_per_mwh"]))
    return premium


def effective_bid_premium(cfg: Mapping) -> float:
    """Per-MWh premium on down moves (bids): ``bid_premium_gbp_per_mwh``, else the premium."""
    if cfg.get("bid_premium_gbp_per_mwh") is None:
        return effective_premium(cfg)
    premium = float(cfg["bid_premium_gbp_per_mwh"])
    if cfg.get("min_spread_rule") == "premium_floor":
        premium = max(premium, float(cfg["min_spread_gbp_per_mwh"]))
    return premium


def counter_trade_links(network, candidates: Iterable[str] | None = None) -> list[str]:
    """Cross-border links: the signed IC1 cables plus any link touching an external bus."""
    links = network.links
    names = set(get_piecewise_interconnectors(network).values())
    for name in links.index:
        if "External" in str(links.at[name, "bus0"]) or "External" in str(links.at[name, "bus1"]):
            names.add(str(name))
    ordered = [str(n) for n in links.index if str(n) in names]
    if candidates is not None:
        keep = set(map(str, candidates))
        ordered = [n for n in ordered if n in keep]
    return ordered


def link_shares(network, names: list[str], cfg: Mapping) -> pd.Series:
    """Flow limit per link (share of p_nom) from its counterparty country."""
    by_country = {str(k).strip().lower(): float(v) for k, v in cfg["max_deviation_share_by_country"].items()}
    default = cfg["max_deviation_share"]
    country = (
        network.links["counterparty_country"].reindex(names)
        if "counterparty_country" in network.links.columns
        else pd.Series(index=names, dtype=object)
    )
    shares = {}
    for name in names:
        key = str(country.get(name, "")).strip().lower()
        if key in by_country:
            shares[name] = by_country[key]
        elif default is not None:
            shares[name] = float(default)
        else:
            shares[name] = np.inf
    return pd.Series(shares, dtype=float)


def stamp_counter_trade_attributes(network, cfg: Mapping, names: list[str] | None = None) -> list[str]:
    """Write the premium and flow limit onto the links the balancing stage counter-trades.

    The saved balancing network then carries the premium it was solved with, so
    ``counter_trade_premium_cost`` values re-trading from the network alone.
    """
    names = counter_trade_links(network) if names is None else names
    premium = pd.Series(0.0, index=network.links.index)
    share = pd.Series(np.nan, index=network.links.index)
    if names:
        premium.loc[names] = effective_premium(cfg)
        share.loc[names] = link_shares(network, names, cfg)
    network.links[COUNTER_TRADE_PREMIUM_COLUMN] = premium
    network.links[COUNTER_TRADE_SHARE_COLUMN] = share
    # The pricing rule travels with the network only when it is not the default,
    # so a network solved with the additive rule is written exactly as before.
    if cfg.get("offer_pricing", "additive") != "additive" or cfg.get("bid_pricing", "additive") != "additive":
        for column, key in ((COUNTER_TRADE_OFFER_RULE_COLUMN, "offer_pricing"),
                            (COUNTER_TRADE_BID_RULE_COLUMN, "bid_pricing")):
            rule = pd.Series("", index=network.links.index, dtype=object)
            if names:
                rule.loc[names] = cfg[key]
            network.links[column] = rule
    else:
        network.links.drop(columns=[c for c in (COUNTER_TRADE_OFFER_RULE_COLUMN, COUNTER_TRADE_BID_RULE_COLUMN)
                                    if c in network.links.columns], inplace=True)
    if cfg.get("bid_premium_gbp_per_mwh") is not None:
        bid = pd.Series(0.0, index=network.links.index)
        if names:
            bid.loc[names] = effective_bid_premium(cfg)
        network.links[COUNTER_TRADE_BID_PREMIUM_COLUMN] = bid
    elif COUNTER_TRADE_BID_PREMIUM_COLUMN in network.links.columns:
        network.links.drop(columns=[COUNTER_TRADE_BID_PREMIUM_COLUMN], inplace=True)
    # Likewise the foreign-plant uplift, one column per side: both written when either
    # side is non-zero, neither otherwise. The retired single column is never written.
    offer_share, bid_share = foreign_uplift_shares(cfg)
    if COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN in network.links.columns:
        network.links.drop(columns=[COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN], inplace=True)
    if offer_share > 0.0 or bid_share > 0.0:
        for column, share_value in ((COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN, offer_share),
                                    (COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN, bid_share)):
            uplift = pd.Series(0.0, index=network.links.index)
            if names:
                uplift.loc[names] = share_value
            network.links[column] = uplift
    else:
        network.links.drop(columns=[c for c in (COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN,
                                                COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN)
                                    if c in network.links.columns], inplace=True)
    return names


def foreign_uplift_shares(cfg: Mapping) -> tuple[float, float]:
    """``(offer, bid)`` effective foreign-plant shares of a resolved config (0 when absent)."""
    return (float(cfg.get("foreign_offer_uplift_share", 0.0) or 0.0),
            float(cfg.get("foreign_bid_uplift_share", 0.0) or 0.0))


def stage1_spread(network, names: list[str], schedule: pd.DataFrame, gb_price: pd.Series) -> pd.DataFrame:
    """``|GB Stage 1 price - foreign price|`` per snapshot and signed link.

    The foreign price is the ask when Stage 1 imports, the bid when it exports and
    the nearer of the two when it is idle. Links without piecewise prices get NaN.
    """
    snapshots = schedule.index
    out = pd.DataFrame(np.nan, index=snapshots, columns=names)
    signed = set(get_piecewise_interconnectors(network).values())
    links, ask_t = link_price_tables(network)
    gb = pd.to_numeric(pd.Series(gb_price), errors="coerce").reindex(snapshots)
    for name in names:
        if name not in signed:
            continue
        ask, bid = piecewise_market_prices(links, ask_t, name, snapshots)
        flow = schedule[name].reindex(snapshots).fillna(0.0)
        to_ask = (gb - ask).abs()
        to_bid = (gb - bid).abs()
        spread = np.minimum(to_ask, to_bid)
        spread = spread.where(~(flow > IDLE_FLOW_MW), to_ask)
        spread = spread.where(~(flow < -IDLE_FLOW_MW), to_bid)
        out[name] = spread
    return out


def counter_trade_bounds(network, names: list[str], schedule: pd.DataFrame,
                         gb_price: pd.Series | None, cfg: Mapping):
    """Upper bounds on the up (more import) and down (more export) deviation, MW.

    Returns ``(up, down, audit)``: two snapshots x links frames and a frame of
    per-link-hour flags (``gated``, ``forced``) for the log.
    """
    snapshots = schedule.index
    ws = schedule.reindex(columns=names).fillna(0.0)
    p_nom = pd.to_numeric(network.links.loc[names, "p_nom"], errors="coerce").fillna(0.0)
    p_max = network.get_switchable_as_dense("Link", "p_max_pu", snapshots)[names].mul(p_nom, axis=1)
    p_min = network.get_switchable_as_dense("Link", "p_min_pu", snapshots)[names].mul(p_nom, axis=1)
    shares = link_shares(network, names, cfg)
    allow = pd.DataFrame(
        np.tile((shares * p_nom).to_numpy(dtype=float), (len(snapshots), 1)),
        index=snapshots, columns=names,
    )
    allow_up, allow_down = allow.copy(), allow.copy()
    gated = pd.DataFrame(False, index=snapshots, columns=names)
    if cfg["min_spread_rule"] == "stage1_gate" and cfg["min_spread_gbp_per_mwh"] > 0 and gb_price is not None:
        spread = stage1_spread(network, names, ws, gb_price)
        gated = spread.lt(cfg["min_spread_gbp_per_mwh"]).fillna(False)
        # The gate stops a trader taking a counter-trade the spread does not pay
        # for, i.e. a move that INCREASES |flow| (more import on an importing
        # cable, more export on an exporting one; any move on an idle one). A
        # move back towards zero is the network reducing a flow it cannot carry:
        # allowed inside the flow limit, up to the Stage 1 flow itself (beyond
        # zero it would build a flow the other way, which the gate again stops).
        # Before 27 Sep 2026 the gate zeroed both directions, so every
        # network-forced reduction in a gated hour went to the beyond-limit penalty.
        importing = ws.gt(IDLE_FLOW_MW)
        exporting = ws.lt(-IDLE_FLOW_MW)
        allow_up = allow_up.mask(gated & ~exporting, 0.0)
        allow_up = allow_up.mask(gated & exporting, np.minimum(allow, -ws))
        allow_down = allow_down.mask(gated & ~importing, 0.0)
        allow_down = allow_down.mask(gated & importing, np.minimum(allow, ws))
    # Never tighter than the movement the balancing stage's own limits force.
    forced_up = (p_min - ws).clip(lower=0.0)
    forced_down = (ws - p_max).clip(lower=0.0)
    up = np.maximum(allow_up, forced_up)
    down = np.maximum(allow_down, forced_down)
    # A bound of inf would be a free variable bound; cap at the physical swing.
    swing = (p_max - p_min).clip(lower=0.0) + forced_up + forced_down
    up = up.where(np.isfinite(up), swing)
    down = down.where(np.isfinite(down), swing)
    audit = {"gated": gated, "forced": (forced_up > 1e-6) | (forced_down > 1e-6),
             "up": up, "down": down}
    return up, down, audit


#: Relative MIP gap parameter name by solver (``no_wheeling_mip_gap``).
MIP_GAP_OPTION = {"gurobi": "MIPGap", "highs": "mip_rel_gap"}


def no_wheeling_solver_options(solver_name: str, solver_options: Mapping | None, balancing_config: Mapping | None,
                               logger: logging.Logger | None = None) -> dict:
    """``solver_options`` with the no-wheeling MILP's relative gap (``no_wheeling_mip_gap``) when the binary rule is on.

    The balancing objective carries the Stage 1 interconnector valuation (about -4 m GBP a day on the CP30 2030
    network), so a relative gap is a gap on that magnitude: 1e-3 is about GBP 4,000 a day, under 0.1% of a day's BM
    cost. Unchanged options otherwise (null keeps the solver's default, 1e-4 for Gurobi)."""
    options = dict(solver_options or {})
    cfg = resolve_counter_trade_config(balancing_config)
    if not (cfg["enabled"] and cfg["no_wheeling"] and cfg["no_wheeling_formulation"] == "binary"):
        return options
    gap = cfg["no_wheeling_mip_gap"]
    key = MIP_GAP_OPTION.get(str(solver_name).lower())
    if gap is None or key is None:
        return options
    options[key] = gap
    if logger:
        logger.info("Interconnector counter-trade no wheeling: %s = %g (no_wheeling_mip_gap)", key, gap)
    return options


def no_wheeling_caps(network, names: list[str], ws: pd.DataFrame, up: pd.DataFrame, down: pd.DataFrame):
    """The band bounds ``up`` / ``down`` tightened to each cable's physical headroom at its Stage 1 position.

    Up (more GB supply) is at most ``p_max - p_stage1``, down at most ``p_stage1 - p_min``; never below 0."""
    p_nom = pd.to_numeric(network.links.loc[names, "p_nom"], errors="coerce").fillna(0.0)
    p_max = network.get_switchable_as_dense("Link", "p_max_pu", ws.index)[names].mul(p_nom, axis=1)
    p_min = network.get_switchable_as_dense("Link", "p_min_pu", ws.index)[names].mul(p_nom, axis=1)
    head_up = (p_max - ws[names]).clip(lower=0.0)
    head_down = (ws[names] - p_min).clip(lower=0.0)
    return np.minimum(up[names], head_up), np.minimum(down[names], head_down)


def add_no_wheeling_constraints(model, up_var, down_var, up: pd.DataFrame, down: pd.DataFrame, dim: str,
                                formulation: str = "binary", tol_mw: float = 1e-6) -> int:
    """No same-hour opposite counter-trades: in an hour, IC up moves (more GB supply) on one cable and down moves
    (less GB supply) on another are not both allowed. Returns the number of hours constrained.

    With ``U_t`` / ``D_t`` the hour's summed priced up / down moves and ``CapU_t`` / ``CapD_t`` their summed bounds
    (``up`` / ``down``, MW), the rule is ``U_t x D_t = 0``, a complementarity no linear constraint states exactly:

    * ``binary`` (exact): a direction binary ``z_t`` per hour with ``U_t <= CapU_t z_t`` and
      ``D_t <= CapD_t (1 - z_t)``. The balancing LP becomes a MILP with one binary per hour where both directions
      have headroom.
    * ``hull``: ``U_t / CapU_t + D_t / CapD_t <= 1``, the convex hull of ``{U = 0} u {D = 0}`` in the box, i.e. the
      LP relaxation of ``binary`` and the tightest linear constraint that never cuts off a one-direction hour. It
      removes only wheeling that uses most of both headrooms at once (on the CP30 2030 test months about 5% of the
      wheeled energy, scratchpad icfix/hull_bite.py), so it is kept for small systems and tests.

    Hours where either direction has no headroom need no constraint. Movement beyond the band (the soft-band excess
    variables, priced at the penalty) is a forced move and is left out of the rule. ``up`` / ``down`` may be the
    band bounds tightened to each cable's physical headroom at its Stage 1 position (``no_wheeling_caps``): a cable
    moving further up than that must also move down, which every pricing rule here makes dearer, so the tighter caps
    cut off nothing an optimum uses and make the ``hull`` form tighter.
    """
    cap_up = up.sum(axis=1).astype(float)
    cap_down = down.sum(axis=1).astype(float)
    both = cap_up.gt(tol_mw) & cap_down.gt(tol_mw)
    hours = pd.Index(up.index[both.to_numpy()], name="snapshot")
    if not len(hours):
        return 0
    u_sum = up_var.sel(snapshot=hours).sum(dim)
    d_sum = down_var.sel(snapshot=hours).sum(dim)
    cu = xr.DataArray(cap_up.loc[hours].to_numpy(), dims=["snapshot"], coords={"snapshot": hours})
    cd = xr.DataArray(cap_down.loc[hours].to_numpy(), dims=["snapshot"], coords={"snapshot": hours})
    if formulation == "binary":
        z = model.add_variables(binary=True, coords=[hours], name="Link-counter-trade-direction")
        model.add_constraints(u_sum - cu * z <= 0.0, name="Link-counter-trade-no-wheeling-up")
        model.add_constraints(d_sum + cd * z <= cd, name="Link-counter-trade-no-wheeling-down")
    elif formulation == "hull":
        model.add_constraints(u_sum * (1.0 / cu) + d_sum * (1.0 / cd) <= 1.0, name="Link-counter-trade-no-wheeling-hull")
    else:
        raise ValueError(f"no_wheeling_formulation must be one of {NO_WHEELING_FORMULATIONS}, got {formulation!r}")
    return int(len(hours))


def build_counter_trade_callback(network, schedule: pd.DataFrame, gb_price: pd.Series | None,
                                 balancing_config: Mapping | None, logger: logging.Logger | None = None):
    """PyPSA ``extra_functionality`` pricing and bounding cross-border deviations.

    Adds ``Link-counter-trade-up`` / ``-down`` (MW, non-negative) with
    ``p == p_stage1 + up - down`` for every cross-border link, the band above as
    their upper bounds, and ``premium x (up + down) x weight`` to the objective.
    Must run after the callback that replaces the objective. Returns None when
    disabled or when the network has no cross-border link. The bounds used by
    each call are kept on ``callback.bounds`` (keyed by first snapshot).
    """
    logger = logger or logging.getLogger(__name__)
    cfg = resolve_counter_trade_config(balancing_config)
    if not cfg["enabled"]:
        return None
    names = counter_trade_links(network, schedule.columns)
    if not names:
        logger.info("Interconnector counter-trade: no cross-border links in the Stage 1 schedule")
        return None
    premium = effective_premium(cfg)
    penalty = cfg["beyond_limit_penalty_gbp_per_mwh"]
    if gb_price is None and cfg["min_spread_rule"] == "stage1_gate":
        logger.warning("Interconnector counter-trade: no Stage 1 price, spread gate off")
    shares = link_shares(network, names, cfg)
    bid_premium = effective_bid_premium(cfg)
    offer_uplift, bid_uplift = foreign_uplift_shares(cfg)
    uplift_on = offer_uplift > 0.0 or bid_uplift > 0.0
    # Hourly direction premiums whenever a side is all_in, the bid premium differs or
    # the foreign-plant uplift is on.
    all_in = ("all_in" in (cfg["offer_pricing"], cfg["bid_pricing"]) or bid_premium != premium
              or uplift_on)
    if all_in:
        logger.info(
            "Interconnector counter-trade pricing: offers (less export / more import) %s, bids %s; an all_in "
            "move costs max(premium, its marginal trade value) per MWh; premium %.1f on offers, %.1f on bids",
            cfg["offer_pricing"], cfg["bid_pricing"], premium, bid_premium,
        )

    def direction_premiums(n, sns, live):
        lk = n.links.copy()
        lk[COUNTER_TRADE_OFFER_RULE_COLUMN] = ""
        lk[COUNTER_TRADE_BID_RULE_COLUMN] = ""
        lk.loc[live, COUNTER_TRADE_OFFER_RULE_COLUMN] = cfg["offer_pricing"]
        lk.loc[live, COUNTER_TRADE_BID_RULE_COLUMN] = cfg["bid_pricing"]
        full = schedule.reindex(index=sns).fillna(0.0)
        return counter_trade_direction_premiums(full, lk, live, pd.Series(premium, index=live),
                                                link_price_tables(n)[1], pd.Series(bid_premium, index=live),
                                                pd.Series(offer_uplift, index=live) if uplift_on else None,
                                                pd.Series(bid_uplift, index=live) if uplift_on else None)

    logger.info(
        "Interconnector counter-trade on %d cross-border links: premium %.1f GBP/MWh on |deviation| "
        "from Stage 1, foreign-plant uplift %.3f (offers) / %.3f (bids) of the foreign marginal value "
        "(scale %.3f, source %s), flow limit %s of p_nom, spread rule %s (%.1f GBP/MWh), movement beyond "
        "the limit %s",
        len(names), premium, offer_uplift, bid_uplift, cfg.get("foreign_uplift_scale", 1.0),
        cfg.get("foreign_uplift_source", "explicit"),
        ", ".join(f"{n}={s:.2f}" for n, s in shares.items()),
        cfg["min_spread_rule"], cfg["min_spread_gbp_per_mwh"],
        "forbidden (hard band)" if penalty is None
        else f"at {penalty:.0f} GBP/MWh objective-only penalty (soft band)",
    )

    if cfg["no_wheeling"]:
        logger.info(
            "Interconnector counter-trade: no wheeling (no same-hour up and down moves on different cables), "
            "%s formulation%s", cfg["no_wheeling_formulation"],
            " (one direction binary per hour: the balancing solve is a MILP)"
            if cfg["no_wheeling_formulation"] == "binary" else "",
        )

    bounds: dict = {}

    def add_counter_trade(n, snapshots):
        model = n.model
        if "Link-p" not in model.variables:
            return
        link_p = model.variables["Link-p"]
        dims = [d for d in link_p.dims if d != "snapshot"]
        dim = dims[0] if dims else "Link"
        present = set(map(str, link_p.coords[dim].values))
        live = [name for name in names if name in present and name in n.links.index]
        if not live:
            return
        sns = pd.Index(snapshots)
        ws = schedule.reindex(index=sns, columns=live).fillna(0.0)
        price = None if gb_price is None else pd.Series(gb_price).reindex(sns)
        up, down, audit = counter_trade_bounds(n, live, ws, price, cfg)
        bounds[sns[0]] = audit
        coords = {"snapshot": sns, dim: pd.Index(live, name=dim)}
        zeros = xr.DataArray(np.zeros((len(sns), len(live))), dims=["snapshot", dim], coords=coords)
        up_var = model.add_variables(
            lower=zeros, upper=xr.DataArray(up.to_numpy(dtype=float), dims=["snapshot", dim], coords=coords),
            name="Link-counter-trade-up",
        )
        down_var = model.add_variables(
            lower=zeros, upper=xr.DataArray(down.to_numpy(dtype=float), dims=["snapshot", dim], coords=coords),
            name="Link-counter-trade-down",
        )
        if cfg["no_wheeling"]:
            cap_up, cap_down = no_wheeling_caps(n, live, ws, up, down)
            n_hours = add_no_wheeling_constraints(model, up_var, down_var, cap_up, cap_down, dim,
                                                  cfg["no_wheeling_formulation"])
            audit["no_wheeling_hours"] = n_hours
        moved = up_var - down_var
        paid_up, paid_down = up_var, down_var
        weights = n.snapshot_weightings.objective.reindex(sns).fillna(1.0)
        weight_da = xr.DataArray(weights.to_numpy(dtype=float), dims=["snapshot"], coords={"snapshot": sns})
        if penalty is not None:
            # Soft band: movement beyond it is allowed at the penalty (module docstring).
            ex_up = model.add_variables(lower=zeros, name="Link-counter-trade-excess-up")
            ex_down = model.add_variables(lower=zeros, name="Link-counter-trade-excess-down")
            moved = moved + ex_up - ex_down
            paid_up, paid_down = paid_up + ex_up, paid_down + ex_down
            if penalty > 0.0:
                model.objective = model.objective + ((ex_up + ex_down) * weight_da).sum() * penalty
        model.add_constraints(
            link_p.sel({dim: live}) - moved == ws.to_numpy(dtype=float),
            name="Link-counter-trade-anchor",
        )
        if premium > 0.0 or bid_premium > 0.0 or uplift_on:
            if not all_in:
                model.objective = model.objective + ((paid_up + paid_down) * weight_da).sum() * premium
            else:
                # Hourly direction premiums: the part of the all-in price the
                # trade-value change does not already charge (module docstring).
                prem_up, prem_down = direction_premiums(n, sns, live)
                up_da = xr.DataArray(prem_up.to_numpy(dtype=float), dims=["snapshot", dim], coords=coords)
                down_da = xr.DataArray(prem_down.to_numpy(dtype=float), dims=["snapshot", dim], coords=coords)
                model.objective = model.objective + ((paid_up * up_da + paid_down * down_da) * weight_da).sum()

    add_counter_trade.bounds = bounds
    add_counter_trade.links = names
    add_counter_trade.config = cfg
    return add_counter_trade


def counter_trade_excess_penalty(network, callback) -> tuple[float, float]:
    """``(penalty GBP, MWh)`` of movement beyond the band in the network's last solve.

    Read from the solved ``Link-counter-trade-excess-*`` variables. The penalty is
    objective-only: callers strip it from the reported BM cost, as they strip the
    storage guards. ``(0, 0)`` when the callback is off or the band is hard.
    """
    if callback is None:
        return 0.0, 0.0
    penalty = callback.config.get("beyond_limit_penalty_gbp_per_mwh")
    model = getattr(network, "model", None)
    if penalty is None or model is None:
        return 0.0, 0.0
    total = 0.0
    weights = None
    for name in ("Link-counter-trade-excess-up", "Link-counter-trade-excess-down"):
        if name not in model.variables:
            continue
        solution = model.variables[name].solution
        if weights is None:
            sns = pd.Index(solution.coords["snapshot"].values)
            weights = xr.DataArray(
                network.snapshot_weightings.objective.reindex(sns).fillna(1.0).to_numpy(dtype=float),
                dims=["snapshot"], coords={"snapshot": solution.coords["snapshot"].values},
            )
        total += float((solution.fillna(0.0).clip(min=0.0) * weights).sum())
    return total * float(penalty), total


def summarise_counter_trade(physical: pd.DataFrame, schedule: pd.DataFrame, links: pd.DataFrame,
                            bounds: Mapping | None = None, tol_mw: float = 1.0,
                            ask_t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-link deviation volume, premium cost and how often the band bound it.

    ``ask_t`` (the hourly ask table) values an ``all_in`` premium on the prices
    the LP saw; the additive premium does not need it.
    """
    from scripts.interconnectors.market_coupling import counter_trade_premium_cost

    if COUNTER_TRADE_SHARE_COLUMN not in links.columns:
        return pd.DataFrame()
    share = pd.to_numeric(links[COUNTER_TRADE_SHARE_COLUMN], errors="coerce")
    names = [n for n in links.index[share.notna()] if n in physical.columns]
    if not names:
        return pd.DataFrame()
    idx = physical.index
    phys = physical.reindex(index=idx, columns=names).fillna(0.0)
    sched = schedule.reindex(index=idx, columns=names).fillna(0.0)
    hours = pd.Series(1.0, index=idx)
    if len(idx) > 1:
        step = pd.Index(idx).to_series().diff().dt.total_seconds().dropna().median() / 3600
        if pd.notna(step) and step > 0:
            hours[:] = float(step)
    dev = phys - sched
    premium = counter_trade_premium_cost(phys, schedule.reindex(index=idx).fillna(0.0), links, ask_t).reindex(
        columns=names, fill_value=0.0)
    rows = []
    frames = {}
    if bounds:
        for key in ("up", "down", "gated"):
            parts = [audit[key] for audit in bounds.values() if key in audit]
            frames[key] = pd.concat(parts).groupby(level=0).last().reindex(idx) if parts else None
    for name in names:
        row = {
            "link": name,
            "up_MWh": float((dev[name].clip(lower=0.0) * hours).sum()),
            "down_MWh": float(((-dev[name]).clip(lower=0.0) * hours).sum()),
            "premium_GBP": float((premium[name] * hours).sum()),
            "max_share": float(share[name]),
        }
        if frames.get("up") is not None and name in frames["up"].columns:
            up_b, down_b = frames["up"][name], frames["down"][name]
            at_bound = ((dev[name] > tol_mw) & (dev[name] >= up_b - tol_mw)) | (
                (dev[name] < -tol_mw) & (-dev[name] >= down_b - tol_mw))
            row["hours_at_bound"] = int(at_bound.sum())
            row["hours_gated"] = int(frames["gated"][name].fillna(False).sum())
            beyond = (dev[name] - up_b).clip(lower=0.0) + (-dev[name] - down_b).clip(lower=0.0)
            row["beyond_limit_MWh"] = float((beyond.where(beyond > tol_mw, 0.0) * hours).sum())
        rows.append(row)
    return pd.DataFrame(rows)
