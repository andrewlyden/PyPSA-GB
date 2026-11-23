"""Stage 1 interconnector trading-hurdle override (16 Oct 2026, Stage 1 price-gap test, variant s1_nh).

The single foreign-price basis (item D2) prices each signed interconnector link from one hourly
foreign reference price: import ask = reference + hurdle/2, export bid = reference - hurdle/2 (exports
also pay the cable loss, ``market_coupling.export_efficiency``), with the export-bid floor of 27 Sep
2026 on top. The hurdle (``interconnectors.pricing.single_price_transaction_spread_gbp_per_mwh``, 20
for CP30 since OW1) is a calibrated trader friction fitted to CP30's gross volumes; NESO publishes no
hurdle and describes the flows as "based on the lowest system cost" (FOI/26/076 p.3).

The hurdle is built into the network (``links.marginal_cost``, ``links_t.marginal_cost``,
``market_spread_half``, ``market_export_bid_floor``), and every Stage 1 month of a family shares one
build (``network_from``), so a Stage 1 variant cannot change it through the interconnector config.
``market.wholesale.interconnector_hurdle_override`` rewrites those link attributes before the Stage 1
solve, as if the network had been built with the given spread and floor:

* the foreign reference price is recovered from each link's ask exactly as
  ``market_coupling.piecewise_market_prices`` recovers it (the builder's convexity widening for negative
  prices included);
* the new ask is the builder's own ``_foreign_bid_ask_prices(reference, efficiency, spread)``, so the
  link equals one built at that spread, and the bid is rebuilt from it by ``piecewise_market_prices``
  at solve time as usual;
* the IC1 volume tranches (``market_tranche_*``) are left alone: they are the foreign market's price
  response to GB's traded volume (elasticity 1.0 x CP.29 level / market demand, Swinand and O'Mahoney
  2014), the supply-curve shape an endogenous European dispatch (CP30's PLEXOS) also has, not a trader
  friction;
* the cable loss and the circulation epsilon (a 0.01 GBP/MWh numerical tie-break) are kept.

The saved Stage 1 network carries the rewritten attributes, so every valuation that reads the link
tables (``market_coupling.interconnector_trade_cost``) prices the Stage 1 flows as they were solved.
The balancing stage reads its own network and does not apply the override unless
``apply_in_balancing`` (default false, the 16 Oct behaviour: a Stage 1 test only). Off unless
``enabled``; the defaults below are those of ``config/defaults/`` (where the key is documented,
not set).

Level offset (17 Oct 2026: Stage 1 refitted to NESO's CP.28-30 run). ``level_offset_gbp_per_mwh``
moves every single-basis foreign reference price by a constant before the ask and bid are rebuilt, as
``interconnectors.pricing.single_price_level_offsets_gbp_per_mwh`` would at build (one offset for every
coupled market). With ``apply_in_balancing`` true the balancing stage re-prices its own network the same
way, so the BM values interconnector moves on the foreign prices Stage 1 traded at (the counter-trade
pricing reads the BM network's link tables). Default 0 and false: no change.
"""
import logging

import numpy as np
import pandas as pd

from scripts.interconnectors.add_to_network import _foreign_bid_ask_prices
from scripts.interconnectors.market_coupling import (
    EXPORT_BID_FLOOR_COLUMN,
    export_efficiency,
    get_piecewise_interconnectors,
)

#: Code defaults of ``market.wholesale.interconnector_hurdle_override``.
DEFAULT_HURDLE_OVERRIDE = {
    "enabled": False,
    # Full bid/ask spread (GBP/MWh) the single-basis links trade at; 0 = no trader hurdle.
    "spread_gbp_per_mwh": 0.0,
    # Export-bid floor (GBP/MWh); None = no floor (with no hurdle the bid is the foreign price).
    "export_bid_floor_gbp_per_mwh": None,
    # Constant added to every single-basis foreign reference price (GBP/MWh); 0 = the build's level.
    "level_offset_gbp_per_mwh": 0.0,
    # Re-price the balancing stage's network the same way (false = Stage 1 only, the 16 Oct behaviour).
    "apply_in_balancing": False,
}


def resolve_hurdle_override(wholesale_config) -> dict | None:
    """The override block over its defaults, or None when it is off. Refuses unknown keys."""
    block = (wholesale_config or {}).get("interconnector_hurdle_override") or {}
    unknown = sorted(set(block) - set(DEFAULT_HURDLE_OVERRIDE))
    if unknown:
        raise ValueError(f"market.wholesale.interconnector_hurdle_override: unknown keys {unknown}; "
                         f"supported {sorted(DEFAULT_HURDLE_OVERRIDE)}")
    cfg = {**DEFAULT_HURDLE_OVERRIDE, **block}
    if not bool(cfg["enabled"]):
        return None
    spread = float(cfg["spread_gbp_per_mwh"])
    if spread < 0:
        raise ValueError("interconnector_hurdle_override.spread_gbp_per_mwh must be non-negative")
    floor = cfg["export_bid_floor_gbp_per_mwh"]
    if floor is not None and float(floor) < 0:
        raise ValueError("interconnector_hurdle_override.export_bid_floor_gbp_per_mwh must be non-negative")
    offset = float(cfg["level_offset_gbp_per_mwh"] or 0.0)
    if not np.isfinite(offset):
        raise ValueError("interconnector_hurdle_override.level_offset_gbp_per_mwh must be finite")
    return {"spread_gbp_per_mwh": spread, "export_bid_floor_gbp_per_mwh": None if floor is None else float(floor),
            "level_offset_gbp_per_mwh": offset, "apply_in_balancing": bool(cfg["apply_in_balancing"])}


def recover_reference(ask, efficiency: float, half: float, epsilon: float):
    """The foreign reference price behind a single-basis ask (``piecewise_market_prices``' inversion).

    The builder sets ``ask = reference + max(half, max(-reference, 0) x f + epsilon)`` with
    ``f = (1 - eta) / (1 + eta)``, ``eta = efficiency**2``; above the threshold the half spread is the
    configured one, below it the convexity term.
    """
    eta = export_efficiency(efficiency)
    factor = (1.0 - eta) / (1.0 + eta)
    if factor <= 1e-12:
        return ask - max(half, epsilon)
    threshold = (epsilon - half) / factor
    static_reference = ask - half
    negative_reference = (ask - epsilon) / (1.0 - factor)
    if isinstance(ask, pd.Series):
        return static_reference.where(static_reference >= threshold, negative_reference)
    return float(static_reference) if float(static_reference) >= threshold else float(negative_reference)


def apply_interconnector_hurdle_override(network, wholesale_config, logger=None, stage: str = "wholesale") -> dict | None:
    """Re-price the single-basis interconnector links at the override's spread, floor and level (in place).

    ``stage`` "balancing" re-prices only when the override sets ``apply_in_balancing``. Returns a small
    report (links changed, mean ask before and after) or None when the override is off (or not applied in
    this stage). Raises on a separate-basis link: the override is defined for the single basis only.
    """
    logger = logger or logging.getLogger(__name__)
    if stage not in ("wholesale", "balancing"):
        raise ValueError(f"stage must be 'wholesale' or 'balancing', not {stage!r}")
    cfg = resolve_hurdle_override(wholesale_config)
    if cfg is None or (stage == "balancing" and not cfg["apply_in_balancing"]):
        return None
    offset = cfg["level_offset_gbp_per_mwh"]
    links = network.links
    names = list(get_piecewise_interconnectors(network).values())
    if not names:
        logger.warning("Interconnector hurdle override: no piecewise interconnector links; nothing changed")
        return {"links": 0}
    basis = links["market_price_basis"] if "market_price_basis" in links else pd.Series("separate", index=links.index)
    separate = [n for n in names if str(basis.get(n, "separate")) != "single"]
    if separate:
        raise ValueError(f"interconnector_hurdle_override needs the single foreign-price basis; "
                         f"separate-basis links: {separate[:5]}")
    ask_t = network.links_t.get("marginal_cost", pd.DataFrame())
    if ask_t is None:
        ask_t = pd.DataFrame()
    new_pricing = {"transaction_spread_gbp_per_mwh": cfg["spread_gbp_per_mwh"]}
    before, after = [], []
    for name in names:
        eta_link = float(links.at[name, "efficiency"])
        half = float(links.at[name, "market_spread_half"])
        epsilon = float(links.at[name, "market_circulation_epsilon"])
        pricing = dict(new_pricing, circulation_epsilon_gbp_per_mwh=epsilon)
        static_ask = float(links.at[name, "marginal_cost"])
        static_ref = recover_reference(static_ask, eta_link, half, epsilon) + offset
        new_static_ask, new_static_bid = _foreign_bid_ask_prices(static_ref, eta_link, pricing)
        if cfg["export_bid_floor_gbp_per_mwh"] is not None and static_ref > 0:
            new_static_bid = max(new_static_bid, min(static_ref, cfg["export_bid_floor_gbp_per_mwh"]))
        links.at[name, "marginal_cost"] = new_static_ask
        if "market_export_bid" in links:
            links.at[name, "market_export_bid"] = new_static_bid
        if "market_export_bid_offset" in links:
            links.at[name, "market_export_bid_offset"] = new_static_ask - new_static_bid
        if not ask_t.empty and name in ask_t.columns:
            ask = pd.to_numeric(ask_t[name], errors="coerce").astype(float)
            reference = recover_reference(ask, eta_link, half, epsilon) + offset
            new_ask, _ = _foreign_bid_ask_prices(reference, eta_link, pricing)
            before.append(float(ask.mean()))
            after.append(float(new_ask.mean()))
            ask_t[name] = new_ask.to_numpy(dtype=float)
        else:
            before.append(static_ask)
            after.append(new_static_ask)
    links.loc[names, "market_spread_half"] = cfg["spread_gbp_per_mwh"] / 2.0
    floor = cfg["export_bid_floor_gbp_per_mwh"]
    links.loc[names, EXPORT_BID_FLOOR_COLUMN] = np.nan if floor is None else floor
    if not ask_t.empty:
        network.links_t["marginal_cost"] = ask_t
    report = {"links": len(names), "spread_gbp_per_mwh": cfg["spread_gbp_per_mwh"],
              "export_bid_floor_gbp_per_mwh": floor, "level_offset_gbp_per_mwh": offset, "stage": stage,
              "mean_ask_before": float(np.mean(before)), "mean_ask_after": float(np.mean(after))}
    logger.info(
        "Interconnector hurdle override (%s): %d single-basis links re-priced at spread %.2f GBP/MWh, "
        "export-bid floor %s, level offset %+.2f GBP/MWh; mean import ask %.2f -> %.2f GBP/MWh; "
        "volume tranches unchanged",
        stage, len(names), cfg["spread_gbp_per_mwh"], "off" if floor is None else f"{floor:g}", offset,
        report["mean_ask_before"], report["mean_ask_after"],
    )
    return report
