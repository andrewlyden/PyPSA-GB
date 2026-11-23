"""Shared constraints and helpers for directional interconnector links.

Future interconnectors are represented by two non-negative links per cable:
one from the foreign zone to GB and one from GB to the foreign zone.  PyPSA
constrains each link independently, so this module adds the physical shared
cable limit used by both directions.
"""

from __future__ import annotations

import logging
import re
import numpy as np

import pandas as pd
import xarray as xr


def get_directional_interconnector_pairs(network) -> dict[str, dict[str, str]]:
    """Return complete ``{pair_id: {import, export}}`` directional link pairs."""
    links = network.links
    required = {"interconnector_pair_id", "interconnector_direction"}
    if links.empty or not required.issubset(links.columns):
        return {}

    pairs: dict[str, dict[str, str]] = {}
    for link_name, row in links.iterrows():
        pair_id = row.get("interconnector_pair_id")
        direction = str(row.get("interconnector_direction", "")).lower()
        if pd.isna(pair_id) or direction not in {"import", "export"}:
            continue
        pairs.setdefault(str(pair_id), {})[direction] = str(link_name)

    return {
        pair_id: directions
        for pair_id, directions in pairs.items()
        if {"import", "export"}.issubset(directions)
    }


def build_interconnector_shared_capacity_callback(network, logger=None):
    """Build a PyPSA callback enforcing one capacity limit per cable.

    The constraint is ``p_import + p_export <= capacity * availability`` at
    every snapshot.  It prevents paired links from each using the full cable
    rating simultaneously.  Bid/ask spreads on the foreign counterparty then
    remove any economic incentive for simultaneous counter-flow.
    """
    logger = logger or logging.getLogger(__name__)
    pairs = get_directional_interconnector_pairs(network)
    if not pairs:
        return None

    logger.info(
        "Will enforce shared directional capacity on %d interconnector cables",
        len(pairs),
    )

    def shared_capacity(n, snapshots):
        model = n.model
        if "Link-p" not in model.variables:
            return

        live_pairs = get_directional_interconnector_pairs(n)
        link_p = model.variables["Link-p"]
        component_dims = [dim for dim in link_p.dims if dim != "snapshot"]
        link_dim = component_dims[0] if component_dims else "name"

        p_max_pu = n.get_switchable_as_dense("Link", "p_max_pu", snapshots)
        for pair_id, directions in live_pairs.items():
            import_link = directions["import"]
            export_link = directions["export"]
            if import_link not in link_p.coords[link_dim].values:
                continue
            if export_link not in link_p.coords[link_dim].values:
                continue

            capacity = min(
                float(n.links.at[import_link, "p_nom"]),
                float(n.links.at[export_link, "p_nom"]),
            )
            availability = p_max_pu[[import_link, export_link]].min(axis=1)
            rhs = xr.DataArray(
                capacity * availability.to_numpy(dtype=float),
                dims=["snapshot"],
                coords={"snapshot": snapshots},
            )
            expression = (
                link_p.sel({link_dim: import_link})
                + link_p.sel({link_dim: export_link})
            )
            safe_id = re.sub(r"[^A-Za-z0-9_]+", "_", pair_id).strip("_")
            model.add_constraints(
                expression <= rhs,
                name=f"Link-interconnector_shared_capacity_{safe_id}",
            )

    return shared_capacity


def lock_material_counterflows(
    network,
    link_dispatch: pd.DataFrame | None = None,
    tolerance_mw: float = 0.1,
    logger=None,
) -> int:
    """Disable the weaker direction where a directional cable counterflows.

    A pair of non-negative directional links is convenient for linear import
    and export pricing, but a shared-capacity constraint alone cannot impose
    the non-convex condition that only one direction may flow at a time.
    This helper supports a two-pass LP solution: after an initial solution,
    retain the stronger direction at each material counterflow snapshot and
    set the other direction's availability to zero before re-solving.

    It only changes affected link-hours and preserves the original outage
    availability for every other link-hour. The return value is the number
    of cable-hours locked, allowing callers to skip the second solve when
    the initial LP is already physically directional.
    """
    logger = logger or logging.getLogger(__name__)
    if tolerance_mw < 0:
        raise ValueError("tolerance_mw must be non-negative")

    pairs = get_directional_interconnector_pairs(network)
    if not pairs:
        return 0

    if link_dispatch is None:
        link_dispatch = getattr(network.links_t, "p0", pd.DataFrame())
    if link_dispatch is None or link_dispatch.empty:
        return 0

    snapshots = pd.Index(link_dispatch.index).intersection(network.snapshots)
    if snapshots.empty:
        return 0

    availability = network.get_switchable_as_dense("Link", "p_max_pu", snapshots)
    locks = 0
    for pair_id, directions in pairs.items():
        import_link = directions["import"]
        export_link = directions["export"]
        if (
            import_link not in link_dispatch.columns
            or export_link not in link_dispatch.columns
            or import_link not in availability.columns
            or export_link not in availability.columns
        ):
            continue

        imported = link_dispatch.loc[snapshots, import_link].clip(lower=0.0)
        exported = link_dispatch.loc[snapshots, export_link].clip(lower=0.0)
        counterflow = (imported > tolerance_mw) & (exported > tolerance_mw)
        if not counterflow.any():
            continue

        # Keep the larger sent flow. For exact ties retain import, giving a
        # deterministic result without introducing a numerical tolerance.
        import_wins = imported[counterflow] >= exported[counterflow]
        blocked_export = import_wins.index[import_wins]
        blocked_import = import_wins.index[~import_wins]
        availability.loc[blocked_export, export_link] = 0.0
        availability.loc[blocked_import, import_link] = 0.0
        locks += int(counterflow.sum())
        logger.warning(
            "Locking %d material counterflow hour(s) on %s for a second LP pass",
            int(counterflow.sum()),
            pair_id,
        )

    if locks:
        # Assign the fully resolved availability frame so existing outages and
        # time-varying limits are retained alongside the new direction locks.
        network.links_t.p_max_pu = availability
    return locks



def export_efficiency(efficiency) -> float:
    """The factor on the export bid in a signed link's cost: ``efficiency**2`` (item D2).

    A signed link (bus0 foreign, bus1 GB) running backwards takes ``efficiency x |p|``
    from GB and puts ``|p|`` on the foreign bus, a gain instead of a loss. The
    foreign bus is a neutral balance, so only the money matters: GB should earn
    ``efficiency x bid`` per MWh it sends, which is ``efficiency**2 x bid`` per MWh
    of link flow. Every valuation (the LP epigraph, ``interconnector_trade_cost``
    and the bid rebuilt in ``piecewise_market_prices``) uses this one factor.
    """
    return max(float(efficiency), 0.01) ** 2


#: Static link column (GBP/MWh) holding the single-basis export-bid floor; NaN or
#: absent means no floor (``floor_export_bid``).
EXPORT_BID_FLOOR_COLUMN = "market_export_bid_floor"


def floor_export_bid(bid, reference, floor):
    """Lift the export bid to ``min(reference, floor)`` where the foreign price is positive.

    The single basis's trading hurdle (item D2) prices exports at ``reference -
    hurdle/2``, so when the foreign price is below the half hurdle the bid is
    negative and GB curtails its own zero-cost surplus rather than export it. The
    floor removes the hurdle in those hours: ``bid = max(bid, min(reference,
    floor))`` wherever ``reference > 0``. The bid is never lifted above the
    foreign price, so the ask (``reference`` plus at least the half spread) still
    covers ``efficiency**2 x bid`` and the signed-link cost stays convex. A
    non-positive foreign price keeps its bid. ``floor`` None or NaN is a no-op.
    """
    if floor is None or not np.isfinite(float(floor)):
        return bid
    floor = float(floor)
    if isinstance(bid, pd.Series):
        reference = pd.Series(reference, index=bid.index, dtype=float)
        lifted = reference.clip(upper=floor)
        return bid.where(~(reference > 0.0), np.maximum(bid, lifted))
    reference = float(reference)
    if reference > 0.0:
        return max(float(bid), min(reference, floor))
    return bid


def get_piecewise_interconnectors(network) -> dict[str, str]:
    """Return ``{cable_name: signed_link_name}`` for direct LP pricing."""
    return _piecewise_links(network.links)


def _piecewise_links(links: pd.DataFrame) -> dict[str, str]:
    required = {"interconnector_name", "foreign_market_pricing"}
    if links.empty or not required.issubset(links.columns):
        return {}
    links = links[links["foreign_market_pricing"].astype(str).str.lower().eq("piecewise_bidirectional")]
    return {str(row["interconnector_name"]): str(link_name) for link_name, row in links.iterrows()}


# ── volume tranches (item IC1) ───────────────────────────────────────────────
#
# GB's summed signed flow F into a coupled foreign market moves the price it
# trades at. With boundaries T_k = k * width (k = 1 .. count-1) the extra cost is
#
#     sum_k import_step * max(0, F - T_k) + export_step * max(0, -F - T_k),
#
# on top of each cable's own bid/ask epigraph: a convex function of F, so the LP
# stays an LP. Flow is measured at bus0 (the foreign side), like the epigraph.
# The steps are written by ``scripts.interconnectors.volume_tranches``.

def get_volume_tranche_groups(links: pd.DataFrame) -> dict[str, dict]:
    """``{market: {links, width, count, import_step, export_step}}`` for grouped signed links."""
    if "market_tranche_group" not in links.columns:
        return {}
    signed = set(_piecewise_links(links).values())
    group = links["market_tranche_group"].fillna("").astype(str)
    groups = {}
    for name in sorted(set(group[group.ne("") & group.ne("nan")])):
        members = [link for link in group.index[group.eq(name)] if link in signed]
        if not members:
            continue
        params = {}
        for key, column in (("width", "market_tranche_width_mw"), ("count", "market_tranche_count"),
                            ("import_step", "market_tranche_import_step"),
                            ("export_step", "market_tranche_export_step")):
            values = pd.to_numeric(links.loc[members, column], errors="coerce")
            if values.isna().any() or not np.allclose(values, values.iloc[0], rtol=0.0, atol=1e-9):
                raise ValueError(f"tranche group {name!r} has missing or inconsistent {column}")
            params[key] = float(values.iloc[0])
        params["count"] = int(round(params["count"]))
        if params["width"] <= 0 or params["import_step"] < 0 or params["export_step"] < 0:
            raise ValueError(f"tranche group {name!r} needs a positive width and non-negative steps")
        groups[name] = {"links": members, **params}
    return groups


def tranche_boundaries(group: dict) -> list[float]:
    return [k * group["width"] for k in range(1, group["count"])]


def tranche_cost(net_flow, group: dict) -> pd.Series:
    """The tranche cost of a signed net flow (GBP per hour of flow), per snapshot."""
    flow = pd.Series(net_flow, dtype=float)
    cost = pd.Series(0.0, index=flow.index)
    for boundary in tranche_boundaries(group):
        cost += group["import_step"] * (flow - boundary).clip(lower=0.0)
        cost += group["export_step"] * (-flow - boundary).clip(lower=0.0)
    return cost


def allocate_tranche_cost(flows: pd.DataFrame, group: dict) -> pd.DataFrame:
    """Split a group's tranche cost across its cables by share of same-direction flow.

    Only net import in excess of a boundary costs an import step, so an import
    hour's cost is shared by the cables that import in it; export likewise.
    """
    members = [link for link in group["links"] if link in flows.columns]
    cols = flows[members].fillna(0.0)
    net = cols.sum(axis=1)
    cost = tranche_cost(net, group)
    importing = cols.clip(lower=0.0)
    exporting = (-cols).clip(lower=0.0)
    share = importing.div(importing.sum(axis=1).replace(0.0, np.nan), axis=0).where(net.gt(0.0), 0.0)
    share = share.where(net.gt(0.0), exporting.div(exporting.sum(axis=1).replace(0.0, np.nan), axis=0))
    return share.fillna(0.0).mul(cost, axis=0)


def interconnector_trade_cost(flows: pd.DataFrame, links: pd.DataFrame, ask_t: pd.DataFrame | None = None,
                              ) -> pd.DataFrame:
    """What the LP charges for a signed interconnector schedule, per snapshot and cable.

    ``max(ask * p, efficiency**2 * bid * p)`` per cable plus its share of the market's
    tranche cost: positive for net cost, negative for net export income. Unweighted;
    multiply by snapshot weights for money. This is the one valuation every
    post-solve consumer uses, so none can drift from the solve's own objective.
    """
    signed = [link for link in _piecewise_links(links).values() if link in flows.columns]
    snapshots = flows.index
    out = pd.DataFrame(0.0, index=snapshots, columns=signed)
    for link in signed:
        ask, bid = piecewise_market_prices(links, ask_t, link, snapshots)
        p = flows[link].fillna(0.0).astype(float)
        eff = export_efficiency(links.at[link, "efficiency"])
        out[link] = np.maximum(ask.to_numpy() * p.to_numpy(), eff * bid.to_numpy() * p.to_numpy())
    for group in get_volume_tranche_groups(links).values():
        allocated = allocate_tranche_cost(flows.reindex(columns=signed, fill_value=0.0), group)
        out[allocated.columns] += allocated
    return out


#: Static link column (GBP/MWh) the balancing stage stamps on each cross-border
#: link it counter-trades (scripts/market/interconnector_counter_trade.py).
COUNTER_TRADE_PREMIUM_COLUMN = "market_counter_trade_premium"

#: Static link columns (strings) recording how the premium was applied to moves
#: that raise GB's import position (offers: less export or more import) and to
#: moves that lower it (bids). ``additive`` (and an absent column, i.e. any
#: network solved before 3 Oct 2026): premium on top of the trade-value change.
#: ``all_in``: the premium is the whole price of the move, the trade-value change
#: included (``counter_trade_direction_premiums``).
COUNTER_TRADE_OFFER_RULE_COLUMN = "market_counter_trade_offer_rule"
COUNTER_TRADE_BID_RULE_COLUMN = "market_counter_trade_bid_rule"
COUNTER_TRADE_PRICING_RULES = ("additive", "all_in")
#: Static link column (GBP/MWh), written only when the bid side has its own
#: premium (``bid_premium_gbp_per_mwh``); absent, bids pay
#: ``COUNTER_TRADE_PREMIUM_COLUMN`` like offers.
COUNTER_TRADE_BID_PREMIUM_COLUMN = "market_counter_trade_bid_premium"
#: Static link columns (shares, e.g. 0.6 and 0.1), written together when either is
#: non-zero: the foreign-plant uplift on offer moves (import up / export down) and on bid
#: moves (import down / export up), as a share of the move's foreign marginal value, on
#: top of its trade-value change (``interconnector_counter_trade.foreign_offer_uplift_share``
#: / ``foreign_bid_uplift_share`` or ``foreign_plant_multipliers: gb_gas``, 16 Oct 2026).
#: NESO/ESO's Long-term Market and Network Constraint Modelling (Jan 2017, section 12,
#: eqs 6 and 1) prices an interconnector move at the spread plus the foreign plant's offer
#: or bid with GB plant's multipliers. Both raise NESO's cost (ESO-cost convention): see
#: ``counter_trade_direction_premiums`` and scripts/market/interconnector_counter_trade.py.
COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN = "market_counter_trade_foreign_offer_uplift"
COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN = "market_counter_trade_foreign_bid_uplift"
#: The retired single column (one share on both sides, 16 Oct 2026 morning). No network
#: was saved with it non-zero; a network carrying it is still valued, on both sides.
COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN = "market_counter_trade_foreign_uplift"


def marginal_trade_cost(schedule: pd.DataFrame, links: pd.DataFrame, ask_t: pd.DataFrame | None = None,
                        tol: float = 1e-6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marginal trade cost of moving each signed cable up and down from ``schedule``, GBP/MWh.

    ``(up, down)``: ``up`` is what one more MW of import (or one MW less export)
    adds to ``interconnector_trade_cost``; ``down`` what one MW less import (or
    one more export) adds, so it is usually negative (GB saves the ask or earns
    the export bid). From the same epigraph the LP charges: the ask above zero,
    ``efficiency**2 x bid`` below it, plus the right- or left-hand slope of the
    market's volume tranches at the schedule's net flow. Signed links only.
    """
    signed = [link for link in _piecewise_links(links).values() if link in schedule.columns]
    snapshots = schedule.index
    up = pd.DataFrame(0.0, index=snapshots, columns=signed)
    down = pd.DataFrame(0.0, index=snapshots, columns=signed)
    for link in signed:
        ask, bid = piecewise_market_prices(links, ask_t, link, snapshots)
        export_slope = export_efficiency(links.at[link, "efficiency"]) * bid
        s = schedule[link].fillna(0.0).astype(float)
        up[link] = np.where(s.to_numpy() >= -tol, ask.to_numpy(), export_slope.to_numpy())
        down[link] = -np.where(s.to_numpy() > tol, ask.to_numpy(), export_slope.to_numpy())
    for group in get_volume_tranche_groups(links).values():
        members = [link for link in group["links"] if link in signed]
        if not members:
            continue
        net = schedule.reindex(columns=members).fillna(0.0).sum(axis=1).to_numpy()
        slope_up = np.zeros(len(snapshots))
        slope_down = np.zeros(len(snapshots))
        for boundary in tranche_boundaries(group):
            slope_up += group["import_step"] * (net >= boundary - tol) - group["export_step"] * (-net > boundary + tol)
            slope_down += -group["import_step"] * (net > boundary + tol) + group["export_step"] * (-net >= boundary - tol)
        up[members] = up[members].add(slope_up, axis=0)
        down[members] = down[members].add(slope_down, axis=0)
    return up, down


def counter_trade_direction_premiums(schedule: pd.DataFrame, links: pd.DataFrame, names: list[str],
                                     premium: pd.Series, ask_t: pd.DataFrame | None = None,
                                     bid_premium: pd.Series | None = None,
                                     foreign_offer_uplift: pd.Series | None = None,
                                     foreign_bid_uplift: pd.Series | None = None,
                                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-MWh premium on an up move and on a down move, per snapshot and link.

    Under ``additive`` it is the link's premium in every hour. Under ``all_in``
    the premium is the whole price of a move (FTI 2023, Table 5-5 and A1.38: the
    cost of reversing flow, EUR 100/MWh from 2030, is the interconnector's BM
    offer and bid, with no foreign marginal value beside it), so only the part the
    trade-value change does not already charge is added: ``max(premium - m, 0)``
    with ``m`` the move's marginal trade cost at the schedule
    (``marginal_trade_cost``). A move then costs ``max(premium, m)`` per MWh: the
    all-in price, or the foreign market price when that is dearer (no trader
    counter-trades below the market). Exact at the schedule; a move that crosses
    zero or a tranche boundary pays the trade value beyond it, as before.
    Directional links carry no trade value on the link, so they stay additive.
    ``bid_premium`` (default ``premium``) is the premium on down moves.

    Foreign-plant uplift (shares per link, default none; under either rule; signed
    links only), in the ESO-cost convention (positive = NESO pays):

    * ``foreign_offer_uplift`` ``s_o`` on an up (offer) move, import up or export down:
      ``+ s_o x max(m_up, 0)``, ``m_up`` the foreign ask it buys or the export bid it
      gives up. The move costs ``(1 + s_o) x m_up``: the foreign plant turned up offers
      at its value marked up, as GB gas offers at ``(1 + offer_markup) x SRMC`` (LTMNC
      eq. 6).
    * ``foreign_bid_uplift`` ``s_b`` on a down (bid) move, import down or export up:
      ``+ s_b x max(-m_down, 0)``, ``-m_down`` the ask it saves or the export bid it
      earns. The move costs ``-(1 - s_b) x |m_down|``: NESO receives the foreign plant's
      saving less its bid discount, as a GB gas bid is ``-(mc - bid_discount x |mc|)``
      (``market_utils.DERIVED_BID_CONVENTIONS``, ``net_saving``; LTMNC eq. 1).

    Both are additions to NESO's cost. ``foreign_bid_uplift`` None with an offer uplift
    given takes the offer share (the retired one-share form).
    """
    idx = schedule.index
    base = premium.reindex(names).fillna(0.0).astype(float)
    base_down = base if bid_premium is None else bid_premium.reindex(names).fillna(0.0).astype(float)
    up = pd.DataFrame(np.tile(base.to_numpy(), (len(idx), 1)), index=idx, columns=names)
    down = pd.DataFrame(np.tile(base_down.to_numpy(), (len(idx), 1)), index=idx, columns=names)
    rules = {}
    for column, key in ((COUNTER_TRADE_OFFER_RULE_COLUMN, "up"), (COUNTER_TRADE_BID_RULE_COLUMN, "down")):
        rule = links[column].reindex(names) if column in links.columns else pd.Series(index=names, dtype=object)
        rules[key] = [n for n in names if str(rule.get(n, "")).strip().lower() == "all_in"]
    signed = set(_piecewise_links(links).values())
    def _shares(series):
        if series is None:
            return pd.Series(0.0, index=names)
        return pd.to_numeric(series.reindex(names), errors="coerce").fillna(0.0).astype(float)

    uplift_up = _shares(foreign_offer_uplift)
    uplift_down = uplift_up if foreign_bid_uplift is None else _shares(foreign_bid_uplift)
    lifted = [n for n in names if n in signed and (uplift_up[n] > 0.0 or uplift_down[n] > 0.0)]
    need = [n for n in dict.fromkeys(rules["up"] + rules["down"] + lifted) if n in signed]
    if not need:
        return up, down
    # Tranche slopes need every member of a group, so read the whole signed schedule.
    full = schedule.reindex(columns=[n for n in _piecewise_links(links).values() if n in schedule.columns]
                            ).fillna(0.0)
    for n in need:
        if n not in full.columns:
            full[n] = 0.0
    m_up, m_down = marginal_trade_cost(full, links, ask_t)
    for n in rules["up"]:
        if n in m_up.columns:
            up[n] = (base[n] - m_up[n]).clip(lower=0.0)
    for n in rules["down"]:
        if n in m_down.columns:
            down[n] = (base_down[n] - m_down[n]).clip(lower=0.0)
    for n in lifted:
        if n in m_up.columns:
            # Offer: + s_o x the foreign value bought or given up (the plant's offer mark-up).
            up[n] = up[n] + uplift_up[n] * m_up[n].clip(lower=0.0)
            # Bid: m_down < 0 is NESO's receipt; + s_b x |m_down| is the plant's bid discount.
            down[n] = down[n] + uplift_down[n] * (-m_down[n]).clip(lower=0.0)
    return up, down


def counter_trade_premium_cost(flows: pd.DataFrame, schedule: pd.DataFrame, links: pd.DataFrame,
                               ask_t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Counter-trade premium per snapshot and link: ``premium x |flow - schedule|``.

    The premium is what NESO pays a trader or neighbouring operator to move a
    cable off its day-ahead schedule, on top of the change in trade value that
    ``interconnector_trade_cost`` measures (FTI 2023, GBP 100/MWh from 2030). It
    is read from the premium the balancing network was solved with, so a network
    solved without one values at zero. Unweighted, like ``interconnector_trade_cost``.
    A link solved under the ``all_in`` rule (``COUNTER_TRADE_OFFER_RULE_COLUMN`` /
    ``COUNTER_TRADE_BID_RULE_COLUMN``) pays the hourly direction premium of
    ``counter_trade_direction_premiums`` instead; pass the hourly ask table
    (``link_price_tables``) so it is valued on the prices the LP saw. A link
    stamped with ``COUNTER_TRADE_BID_PREMIUM_COLUMN`` pays that on down moves, and one
    stamped with ``COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN`` /
    ``COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN`` its foreign-plant uplift on up and down
    moves too (the retired ``COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN``: one share, both sides).
    """
    if COUNTER_TRADE_PREMIUM_COLUMN not in links.columns:
        return pd.DataFrame(0.0, index=flows.index, columns=[])
    premium = pd.to_numeric(links[COUNTER_TRADE_PREMIUM_COLUMN], errors="coerce").fillna(0.0)
    bid_premium = None
    paid = premium > 0.0
    if COUNTER_TRADE_BID_PREMIUM_COLUMN in links.columns:
        bid_premium = pd.to_numeric(links[COUNTER_TRADE_BID_PREMIUM_COLUMN], errors="coerce")
        bid_premium = bid_premium.where(bid_premium.notna(), premium)
        paid = paid | (bid_premium > 0.0)
    uplift_up = uplift_down = None

    def _column(column):
        return pd.to_numeric(links[column], errors="coerce").fillna(0.0)

    if COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN in links.columns or COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN in links.columns:
        zero = pd.Series(0.0, index=links.index)
        uplift_up = (_column(COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN)
                     if COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN in links.columns else zero)
        uplift_down = (_column(COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN)
                       if COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN in links.columns else zero)
    elif COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN in links.columns:
        uplift_up = uplift_down = _column(COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN)
    if uplift_up is not None:
        paid = paid | (uplift_up > 0.0) | (uplift_down > 0.0)
    names = [name for name in premium.index[paid] if name in flows.columns or name in schedule.columns]
    p = flows.reindex(columns=names).fillna(0.0)
    s = schedule.reindex(index=flows.index, columns=names).fillna(0.0)
    if bid_premium is None and uplift_up is None and not any(
            column in links.columns for column in (COUNTER_TRADE_OFFER_RULE_COLUMN, COUNTER_TRADE_BID_RULE_COLUMN)):
        return (p - s).abs().mul(premium.reindex(names), axis=1)
    sched_full = schedule.reindex(index=flows.index).fillna(0.0)
    prem_up, prem_down = counter_trade_direction_premiums(sched_full, links, names, premium, ask_t, bid_premium,
                                                          uplift_up, uplift_down)
    dev = p - s
    return dev.clip(lower=0.0) * prem_up + (-dev).clip(lower=0.0) * prem_down


def get_piecewise_market_prices(network, link_name, snapshots):
    """Return import-ask and export-bid series for a signed interconnector."""
    links, ask_t = link_price_tables(network)
    return piecewise_market_prices(links, ask_t, link_name, snapshots)


def link_price_tables(network) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The static link table and the hourly ask table a valuation needs."""
    dynamic = network.links_t.get("marginal_cost", pd.DataFrame())
    return network.links, (dynamic if dynamic is not None else pd.DataFrame())


def link_price_tables_from_dataset(ds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same two tables read straight from a saved network's NetCDF.

    Reading the dataset with xarray is seconds; loading it as a PyPSA network
    is minutes on the full ETYS files.
    """
    names = pd.Index(ds["links_i"].values, name="Link")
    columns = {}
    for var in ds.data_vars:
        if var.startswith("links_") and not var.startswith("links_t_") and ds[var].dims == ("links_i",):
            columns[var[len("links_"):]] = ds[var].values
    links = pd.DataFrame(columns, index=names)
    ask_t = pd.DataFrame()
    if "links_t_marginal_cost" in ds:
        ask_t = ds["links_t_marginal_cost"].to_pandas()
        ask_t.columns = ask_t.columns.astype(str)
        ask_t.index = dataset_snapshots(ds, ask_t.index)
    return links, ask_t


def dataset_snapshots(ds, index) -> pd.Index:
    """The timestamps behind a saved network's snapshot index.

    PyPSA's NetCDF indexes snapshots 0..n and keeps the times in
    ``snapshots_snapshot``; an integer index joins nothing that is timestamped.
    """
    if "snapshots_snapshot" in ds and len(ds["snapshots_snapshot"]) == len(index):
        return pd.DatetimeIndex(ds["snapshots_snapshot"].values, name="snapshot")
    return pd.Index(index)


def piecewise_market_prices(links: pd.DataFrame, ask_t: pd.DataFrame, link_name, snapshots):
    """Import ask and export bid for one signed link from the link tables."""
    snapshots = pd.Index(snapshots)
    ask = pd.Series(float(links.at[link_name, "marginal_cost"]), index=snapshots, dtype=float)
    if ask_t is not None and link_name in ask_t.columns:
        ask.update(ask_t[link_name].reindex(snapshots).dropna())
    separate = links["market_separate_import_export_prices"] if "market_separate_import_export_prices" in links else None
    if separate is not None and bool(pd.Series(separate).fillna(False).astype(bool).get(link_name, False)):
        # CP.29 gives different annual import and export prices. Both use the
        # same ENTSO-E hourly deviation, so a static offset preserves the two
        # targets after the network is serialised to NetCDF (custom Link time
        # series are not reliably persisted by PyPSA).
        offset = float(links["market_export_bid_offset"].get(link_name, 0.0)) if "market_export_bid_offset" in links else 0.0
        return ask, ask - offset

    eta = export_efficiency(links.at[link_name, "efficiency"])
    base_half = float(links.at[link_name, "market_spread_half"])
    epsilon = float(links.at[link_name, "market_circulation_epsilon"])
    factor = (1.0 - eta) / (1.0 + eta)
    if factor <= 1e-12:
        bid = ask - 2.0 * max(base_half, epsilon)
        reference = ask - max(base_half, epsilon)
    else:
        threshold = (epsilon - base_half) / factor
        static_reference = ask - base_half
        linear_mask = static_reference >= threshold
        bid = ask - 2.0 * base_half
        negative_reference = (ask - epsilon) / (1.0 - factor)
        bid.loc[~linear_mask] = (
            negative_reference.loc[~linear_mask] * (1.0 + factor) - epsilon
        )
        reference = static_reference.where(linear_mask, negative_reference)
    if EXPORT_BID_FLOOR_COLUMN in links.columns:
        floor = pd.to_numeric(pd.Series([links.at[link_name, EXPORT_BID_FLOOR_COLUMN]]), errors="coerce").iloc[0]
        bid = floor_export_bid(bid, reference, floor)
    return ask, bid


def build_piecewise_interconnector_cost_callback(
    network,
    logger=None,
    baseline_dispatch: pd.DataFrame | None = None,
    subtract_native_link_cost: bool = True,
):
    """Add convex directional bid/ask costs to signed interconnector Links.

    Positive Link-p is an import; negative Link-p is an export. An epigraph
    variable represents ``max(import_ask * p, efficiency**2 * export_bid * p)``
    (``export_efficiency``: the export pays the cable loss).

    ``baseline_dispatch`` (the Stage 1 link schedule, item C9): the callback
    records the weighted trade value of that schedule over the solved snapshots
    on ``n._ic_baseline_trade_value_gbp``. It is a constant, so the LP is
    unchanged; a caller reports ``objective - stage1_interconnector_trade_value(n)``
    so the BM cost carries only the CHANGE in trade value, not the whole flow.
    The bid/ask construction guarantees the left-hand export slope does not
    exceed the right-hand import slope, making this exact piecewise cost
    convex and therefore an LP.
    """
    logger = logger or logging.getLogger(__name__)
    if not get_piecewise_interconnectors(network):
        return None

    def add_piecewise_cost(n, snapshots):
        links = get_piecewise_interconnectors(n)
        if not links or "Link-p" not in n.model.variables:
            return
        model = n.model
        link_p = model.variables["Link-p"]
        link_dims = [dim for dim in link_p.dims if dim != "snapshot"]
        link_dim = link_dims[0] if link_dims else "Link"
        link_names = [
            name for name in links.values()
            if name in link_p.coords[link_dim].values
        ]
        if not link_names:
            return

        asks = pd.DataFrame(index=snapshots)
        export_slopes = pd.DataFrame(index=snapshots)
        for link_name in link_names:
            ask, bid = get_piecewise_market_prices(n, link_name, snapshots)
            asks[link_name] = ask
            export_slopes[link_name] = bid * export_efficiency(
                n.links.at[link_name, "efficiency"]
            )
        if (asks < export_slopes - 1e-9).any().any():
            raise ValueError(
                "Interconnector bid/ask prices are non-convex; increase the "
                "transaction spread or correct the foreign price input"
            )

        coords = {"snapshot": snapshots, link_dim: link_names}
        ask_da = xr.DataArray(
            asks.to_numpy(dtype=float), dims=["snapshot", link_dim], coords=coords
        )
        export_da = xr.DataArray(
            export_slopes.to_numpy(dtype=float),
            dims=["snapshot", link_dim], coords=coords,
        )
        trade_cost = model.add_variables(
            lower=xr.full_like(ask_da, -np.inf),
            name="Link-interconnector-piecewise-cost",
        )
        flow = link_p.sel({link_dim: link_names})
        model.add_constraints(
            trade_cost >= flow * ask_da,
            name="Link-interconnector-import-cost",
        )
        model.add_constraints(
            trade_cost >= flow * export_da,
            name="Link-interconnector-export-cost",
        )
        weights = n.snapshot_weightings.objective.reindex(snapshots).fillna(1.0)
        weight_da = xr.DataArray(
            weights.to_numpy(dtype=float), dims=["snapshot"],
            coords={"snapshot": snapshots},
        )
        objective = (trade_cost * weight_da).sum()
        if subtract_native_link_cost:
            objective = objective - (flow * ask_da * weight_da).sum()
        for market, group in get_volume_tranche_groups(n.links).items():
            members = [name for name in group["links"] if name in link_names]
            if not members:
                continue
            net = link_p.sel({link_dim: members}).sum(link_dim)
            safe = re.sub(r"[^A-Za-z0-9_]+", "_", market).strip("_")
            for k, boundary in enumerate(tranche_boundaries(group), start=1):
                for direction, sign, step in (("import", 1.0, group["import_step"]),
                                              ("export", -1.0, group["export_step"])):
                    if step <= 0.0:
                        continue
                    excess = model.add_variables(
                        lower=0.0, coords=[pd.Index(snapshots, name="snapshot")],
                        name=f"Link-ic-tranche-{direction}-{safe}-{k}",
                    )
                    model.add_constraints(
                        excess >= sign * net - boundary,
                        name=f"Link-ic-tranche-{direction}-{safe}-{k}",
                    )
                    objective = objective + (excess * weight_da).sum() * step
            logger.info(
                "Added %d volume tranches on %s (%d cables, width %.0f MW, steps +%.2f / -%.2f GBP/MWh)",
                group["count"], market, len(members), group["width"], group["import_step"], group["export_step"],
            )
        if baseline_dispatch is not None:
            n._ic_baseline_trade_value_gbp = baseline_trade_value(
                baseline_dispatch, n.links, link_price_tables(n)[1], snapshots,
                weights=n.snapshot_weightings.objective,
            )
            logger.info(
                "Stage 1 interconnector trade value over these snapshots: %.0f GBP "
                "(reporting offset: BM cost = objective - this)", n._ic_baseline_trade_value_gbp,
            )
        model.objective = model.objective + objective
        logger.info(
            "Added convex piecewise bid/ask costs for %d signed interconnectors",
            len(link_names),
        )

    return add_piecewise_cost


def baseline_trade_value(baseline_dispatch: pd.DataFrame, links: pd.DataFrame, ask_t: pd.DataFrame | None,
                         snapshots, weights: pd.Series | None = None) -> float:
    """Weighted GBP the piecewise objective charges for a fixed schedule (item C9).

    ``interconnector_trade_cost`` of ``baseline_dispatch`` on ``snapshots`` (tranches
    included), times the objective weights: exactly the value the BM objective's
    interconnector terms would take if every cable stayed on that schedule.
    """
    snapshots = pd.Index(snapshots)
    signed = list(_piecewise_links(links).values())
    if baseline_dispatch is None or not signed:
        return 0.0
    flows = baseline_dispatch.reindex(index=snapshots, columns=signed).fillna(0.0)
    cost = interconnector_trade_cost(flows, links, ask_t)
    w = (pd.Series(1.0, index=snapshots) if weights is None
         else pd.Series(weights).reindex(snapshots).fillna(1.0))
    return float(cost.mul(w, axis=0).to_numpy().sum())


def stage1_interconnector_trade_value(network) -> float:
    """The Stage 1 trade value the last piecewise callback recorded on ``network`` (0 if none).

    BM reporting (item C9): ``bm_cost = network.objective - ... - stage1_interconnector_trade_value(network)``
    so the reported cost is the change in interconnector trade value (plus every
    other redispatch cost), not the whole-flow valuation of the BM schedule.
    """
    return float(getattr(network, "_ic_baseline_trade_value_gbp", 0.0) or 0.0)
