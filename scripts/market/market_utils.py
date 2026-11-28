"""
Market Simulation Utilities for PyPSA-GB

Shared helper functions for the two-stage market dispatch simulation:
  - Bid/offer price calculation from marginal costs
  - Wholesale position extraction from solved networks
  - Redispatch volume and cost computation
  - Congestion identification on lines/transformers

Fixes (November 2026), marked "FIX-n" in the code
-------------------------------------------------------------
FIX-1  ``_load_elexon_bid_offer`` records each generator's price source
       (elexon_bmu / fitted / carrier_median / derived / zero) on
       ``network._bm_price_source``; ``compute_redispatch_volumes`` writes it
       as a ``price_source`` column in the redispatch summary, so the BM
       calibration scorecard can leave out prices copied from the ELEXON
       carrier median it compares against.
FIX-2  ``_load_elexon_bid_offer``: once sentinel prices are removed (by
       ``elexon_data`` or the defensive-bid mask) a BMU can have prices on
       one side only, or a column with no value at all. The old code assumed
       both sides exist and raised ``KeyError`` (Didcot GT, 7 Jan 2020).
       Generators are now directly matched only when both sides have a real
       price; storage aggregates each side over the columns it has, keeping
       the derived price for an empty side.
"""

import json
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from scripts.market.cfd_bm_bids import build_contract_value_bid_frames
from scripts.market.pricing_basis import priced_balancing


DEFAULT_STORAGE_BMU_PREFIXES = {
    "Pumped Storage Hydroelectricity": [
        "T_CRUA",
        "T_CRUAD",
        "T_FOYE",
        "T_FOYED",
        "T_DINO",
        "T_FFES",
        "T_FFESST",
    ],
    "pumped_hydro": [
        "T_CRUA",
        "T_CRUAD",
        "T_FOYE",
        "T_FOYED",
        "T_DINO",
        "T_FFES",
        "T_FFESST",
    ],
}


#: ELEXON bids at or above this (GBP/MWh the unit would pay to be turned down) are refusals, not prices.
DEFENSIVE_BID_GBP_PER_MWH = 1000.0


def carrier_medians_within_cap(carriers: pd.Series, gen_offer: pd.Series, gen_bid: pd.Series,
                               max_price: float) -> tuple[dict, dict, dict]:
    """Per-carrier median ELEXON offer and bid of the matched generators, each side kept only within the cap.

    ``gen_offer``/``gen_bid`` are NaN for unmatched generators. The cap applies to each side separately
    (18 Nov 2026): wind's "do not call" offers (median 1,500) used to reject its bids too (median 74, what wind
    actually bid), so every unmatched Scottish wind farm fell back to a flat 25 (CfD units 0) and the 2020
    validation paid half NESO's price per MWh. Returns (offers, bids, rejected): rejected maps a carrier with a
    side over the cap to its (offer, bid) medians.
    """
    offers, bids, rejected = {}, {}, {}
    matched = gen_offer.notna()
    for carrier in carriers[matched].unique():
        members = carriers.index[carriers.eq(carrier) & matched]
        med_offer, med_bid = gen_offer[members].median(), gen_bid[members].median()
        if abs(med_offer) <= max_price:
            offers[carrier] = med_offer
        if abs(med_bid) <= max_price:
            bids[carrier] = med_bid
        if abs(med_offer) > max_price or abs(med_bid) > max_price:
            rejected[carrier] = (med_offer, med_bid)
    return offers, bids, rejected


def _log_elexon_coverage_summary(
    network,
    matched_mask: pd.Series,
    fallback_mask: pd.Series,
    fallback_label: str,
    logger: logging.Logger,
) -> None:
    """Log how much of the fleet is priced by direct ELEXON matches vs fallback."""
    total = len(network.generators)
    matched = int(matched_mask.sum())
    fallback = int(fallback_mask.sum())
    if total == 0:
        return

    matched_pct = 100.0 * matched / total
    fallback_pct = 100.0 * fallback / total
    logger.info(
        f"ELEXON bid/offer: {matched}/{total} generators matched "
        f"({matched_pct:.1f}%), {fallback}/{total} filled via {fallback_label} "
        f"({fallback_pct:.1f}%)"
    )

    if matched_pct < 10.0:
        fallback_capacity = network.generators.loc[fallback_mask, "p_nom"].sum()
        logger.warning(
            "Low direct ELEXON pricing coverage: "
            f"{matched_pct:.1f}% of generators matched, "
            f"{fallback_capacity:,.0f} MW priced via fallback. "
            "BM redispatch will be dominated by fallback assumptions."
        )

    carrier_df = pd.DataFrame(
        {
            "carrier": network.generators["carrier"],
            "p_nom": network.generators["p_nom"].fillna(0.0),
            "matched": matched_mask.astype(int),
            "fallback": fallback_mask.astype(int),
        },
        index=network.generators.index,
    )
    carrier_summary = (
        carrier_df.groupby("carrier")
        .agg(
            generators=("carrier", "size"),
            p_nom_mw=("p_nom", "sum"),
            matched=("matched", "sum"),
            fallback=("fallback", "sum"),
        )
        .sort_values(["fallback", "p_nom_mw"], ascending=[False, False])
    )

    carrier_summary = carrier_summary[carrier_summary["fallback"] > 0].head(10)
    if not carrier_summary.empty:
        logger.info("Top fallback-priced carriers:")
        for carrier, row in carrier_summary.iterrows():
            logger.info(
                f"  {carrier}: fallback {int(row['fallback'])}/{int(row['generators'])} "
                f"generators, {row['p_nom_mw']:.0f} MW fleet, direct matches {int(row['matched'])}"
            )


#: ``market.balancing.derived_bid_convention``. Bids are in ESO-cost convention
#: (positive = NESO pays to turn the unit down). ``net_saving`` prices a markup-mode
#: bid as the unit's avoided cost handed back to NESO, the raw ELEXON bid negated:
#: ``-(mc - discount * |mc|)``, i.e. ``-0.9 x mc`` for a thermal unit, the sign the
#: ELEXON path (``_load_elexon_bid_offer``) and the future bid stack calibration
#: (``future_bid_stack._calibration_records``) already use. ``legacy_cost`` keeps
#: the pre-26 Sep 2026 ``+mc x (1 - discount)``, which charged NESO for turning gas
#: down: CCGT's implied bid was +107 GBP/MWh on CP30 December 2030 and Stage 2 turned
#: CCGT down by 0.06 MWh all month. Must match config/defaults/.
DERIVED_BID_CONVENTIONS = ("net_saving", "legacy_cost")
DEFAULT_DERIVED_BID_CONVENTION = "net_saving"


def _derived_bid_turn_down_saving(balancing: dict, logger: Optional[logging.Logger] = None) -> bool:
    """Whether markup-mode generator bids hand the avoided cost back to NESO (True) or charge it (legacy)."""
    convention = str(
        (balancing or {}).get("derived_bid_convention", DEFAULT_DERIVED_BID_CONVENTION)
    ).lower()
    if convention not in DERIVED_BID_CONVENTIONS:
        raise ValueError(
            f"market.balancing.derived_bid_convention must be one of {DERIVED_BID_CONVENTIONS}, "
            f"got {convention!r}"
        )
    if logger is not None and convention == "legacy_cost":
        logger.warning(
            "derived_bid_convention 'legacy_cost': markup-mode generator bids charge NESO "
            "for turning the unit down (pre-26 Sep 2026 behaviour)"
        )
    return convention == "net_saving"


def _compute_component_prices(
    components: pd.DataFrame,
    carrier_overrides: dict,
    default_offer_markup: float,
    default_bid_discount: float,
    turn_down_saving: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """Compute static bid/offer prices for generators or storage units.

    ``turn_down_saving`` (generators, see ``DERIVED_BID_CONVENTIONS``) makes a
    markup-mode bid ``-(mc - discount * |mc|)``: turning a unit down saves its
    marginal cost, most of which NESO receives, as a raw ELEXON bid of about
    0.9 x mc does. ``mode: absolute`` bids are configured in ESO-cost convention
    and are never changed. Storage keeps the legacy form (False): its bid is an
    opportunity cost set by ``_derive_storage_prices``, not a fuel saving.
    """
    offer_prices = pd.Series(index=components.index, dtype=float)
    bid_prices = pd.Series(index=components.index, dtype=float)

    for idx in components.index:
        carrier = components.loc[idx, "carrier"]
        mc = components.loc[idx, "marginal_cost"]

        overrides = carrier_overrides.get(carrier, {})
        mode = overrides.get("mode", "markup")
        default_offer = mc * (1.0 + overrides.get("offer_markup", default_offer_markup))
        discount = overrides.get("bid_discount", default_bid_discount)
        if turn_down_saving:
            default_bid = -(mc - discount * abs(mc))
        else:
            default_bid = mc * (1.0 - discount)

        if mode == "absolute":
            offer_prices[idx] = overrides.get("offer_price", default_offer)
            bid_prices[idx] = overrides.get("bid_price", default_bid)
        else:
            offer_prices[idx] = default_offer
            bid_prices[idx] = default_bid

    return offer_prices, bid_prices




#: ``market.balancing.storage_churn_guard_gbp_per_mwh``. The least a storage unit
#: must cost the BM for each MWh it destroys by charging and discharging at once.
#: With gross-leg anchors a simultaneous pair that leaves the state of charge
#: unchanged (charge x, discharge eta_rt * x) costs NESO
#: ``offer * eta_rt * x + bid * x`` (ESO-cost bid, negative for a unit that pays
#: to charge) and destroys ``(1 - eta_rt) * x``, so churn is a dump load whenever
#: that cost per MWh destroyed is below the turn-down bid it displaces. The
#: largest in the CP30 runs is the Investment Contract offshore CfD bid, strike
#: 200.84 GBP/MWh at a zero reference (contract_value_bids audit, Jan 2030);
#: 250 leaves a 25% margin. round_trip mode raises a unit's offer to
#: ``(guard * (1 - eta_rt) - bid) / eta_rt`` when the configured premium falls
#: short (the dispatch-cost floor is ignored, which errs on the safe side).
#: Must match config/defaults/; 0 disables.
DEFAULT_STORAGE_CHURN_GUARD = 250.0


def _derive_storage_prices(
    storage_units: pd.DataFrame,
    carrier_overrides: dict,
    default_offer_markup: float,
    default_bid_discount: float,
    logger: logging.Logger,
    churn_guard: float = DEFAULT_STORAGE_CHURN_GUARD,
) -> Tuple[pd.Series, pd.Series]:
    """Static bid/offer prices for storage units, round-trip aware.

    A storage unit's offer (discharge more) and bid (charge more / discharge
    less) are two halves of one round trip, so pricing them as independent
    absolutes lets the BM buy the cheap half and never pay for the dear one.
    ``mode: "round_trip"`` anchors on the configured ``bid_price`` and returns,
    in the model's ESO-cost convention (positive = NESO pays),

        bid   = -bid_price
        offer =  bid_price / (efficiency_store * efficiency_dispatch) + offer_premium

    ``bid_price`` is configured as the positive price the unit pays NESO per MWh
    it charges more or discharges less: the raw ELEXON sign. In 2024 BOD data
    battery first-tranche bids had a median of +13 GBP/MWh (+40 excluding zeros)
    and 56% were positive, i.e. the battery pays. Before 26 Sep 2026 the value
    was used unnegated, so NESO paid the battery on both legs of a round trip;
    now a round trip costs NESO only its losses plus the premium. The offer is
    raised where needed so that churn costs at least ``churn_guard`` per MWh
    destroyed (``DEFAULT_STORAGE_CHURN_GUARD``), and the prices are refused
    unless no storage unit can be charged for more than any storage unit's
    offer (``_check_storage_cross_carrier_arbitrage``). Other modes fall through
    to ``_compute_component_prices`` unchanged (``absolute`` bids are configured
    in ESO-cost convention); a WARNING is logged once per carrier whose offer
    sits below the round-trip-consistent floor.
    """
    offer_prices, bid_prices = _compute_component_prices(
        storage_units, carrier_overrides, default_offer_markup, default_bid_discount
    )
    if len(storage_units) == 0:
        return offer_prices, bid_prices

    eta_rt = (
        pd.to_numeric(storage_units["efficiency_store"], errors="coerce").fillna(1.0)
        * pd.to_numeric(storage_units["efficiency_dispatch"], errors="coerce").fillna(1.0)
    ).clip(lower=1e-6)
    churn_guard = float(churn_guard or 0.0)
    if churn_guard < 0.0:
        raise ValueError(
            f"market.balancing.storage_churn_guard_gbp_per_mwh must be >= 0, got {churn_guard}"
        )

    warned = set()
    derived_carriers = set()
    guarded = {}
    for idx in storage_units.index:
        carrier = storage_units.loc[idx, "carrier"]
        overrides = carrier_overrides.get(carrier, {})
        if overrides.get("mode") == "round_trip":
            if "bid_price" not in overrides:
                raise ValueError(
                    f"carrier_overrides['{carrier}'] uses mode 'round_trip' but has "
                    "no bid_price to anchor on"
                )
            paid = float(overrides["bid_price"])
            eta = float(eta_rt[idx])
            bid = -paid
            offer = paid / eta + float(overrides.get("offer_premium", 0.0))
            if churn_guard > 0.0 and eta < 1.0:
                floor = (churn_guard * (1.0 - eta) - bid) / eta
                if offer < floor - 1e-9:
                    guarded.setdefault(carrier, []).append(floor - offer)
                    offer = floor
            bid_prices[idx] = bid
            offer_prices[idx] = offer
            derived_carriers.add(carrier)
        elif (
            offer_prices[idx] < bid_prices[idx] / float(eta_rt[idx]) - 1e-9
            and carrier not in warned
        ):
            warned.add(carrier)
            logger.warning(
                f"Storage carrier '{carrier}': offer {offer_prices[idx]:.2f} is below "
                f"the round-trip-consistent floor bid/eta_rt = "
                f"{bid_prices[idx] / float(eta_rt[idx]):.2f} GBP/MWh; the BM can charge "
                "these units without paying for the energy the round trip loses. "
                "Set mode: round_trip for this carrier."
            )

    for carrier in sorted(derived_carriers):
        mask = storage_units["carrier"] == carrier
        logger.info(
            f"Storage carrier '{carrier}': bid {bid_prices[mask].iloc[0]:.2f} GBP/MWh "
            f"(ESO-cost; the unit pays {-bid_prices[mask].iloc[0]:.2f} to charge more "
            f"or discharge less), offer derived from round trip "
            f"{offer_prices[mask].min():.2f}-{offer_prices[mask].max():.2f} GBP/MWh "
            f"({int(mask.sum())} units)"
        )
    for carrier, raised in sorted(guarded.items()):
        logger.warning(
            f"Storage carrier '{carrier}': offer_premium too small for its round trip; "
            f"offer raised by up to {max(raised):.2f} GBP/MWh on {len(raised)} units so "
            f"simultaneous charge/discharge costs at least {churn_guard:.0f} GBP per MWh "
            "destroyed (market.balancing.storage_churn_guard_gbp_per_mwh)"
        )

    _check_storage_cross_carrier_arbitrage(storage_units, offer_prices, bid_prices)
    return offer_prices, bid_prices


#: ``market.balancing.storage_bid_reference``. ``stage1_price``: a round_trip
#: carrier with a ``bid_margin`` bids, per snapshot, what a storage operator
#: bids in GB: the price it would pay to charge sits a margin below the market
#: price, ``paid_t = lambda_t - bid_margin`` with lambda the Stage 1 clearing
#: price (``network._bm_reference_price``). 2024 ELEXON BOD, first bid tranche
#: against the hourly MID price: battery bids were below MID in 98.6% of
#: periods, median MID - 44; pumped storage median MID - 79. A fixed paid bid
#: is wrong in the hours that matter: Stage 1 clears at exactly zero in 41% of
#: 2030 hours, where a battery paying a fixed 40 GBP to charge made the BM a
#: money pump (turn curtailed wind up at ~0, charge for a 40 receipt; the CP30
#: 1 Jan 2030 probe booked a negative BM cost and 39 GWh of extra charging).
#: ``fixed`` keeps the static ``bid_price``. ``stage1_daily_mean`` (10 Oct 2026) references the same formula
#: to the centred 24-hour mean Stage 1 price instead of the hour's own: a store that charges less (or
#: discharges more) now gives up energy it would have sold later in the day, so its opportunity value is the
#: day's price level, not an hour that clears at or below zero (where ``stage1_price`` makes a battery offer
#: about GBP 3/MWh). Must match config/defaults/.
STORAGE_BID_REFERENCES = ("stage1_price", "fixed", "stage1_daily_mean")
DEFAULT_STORAGE_BID_REFERENCE = "stage1_price"


def _state_storage_at_cost_shares(balancing: dict, bm_bids: Optional[dict], units: pd.Index,
                                  logger: logging.Logger):
    """Item F8: state ownership shares for storage bidding at cost, and the churn guard; or (None, 0).

    Read from ``support.bm_bids.contract_value_bids.public_ownership_bids`` (``bm_bids`` is the
    scenario's ``support.bm_bids`` block) when its ``storage_at_cost`` is true: the same share
    table the state's CfD bids use, so the bid stack and the bill agree about who owns which store.
    """
    support = (bm_bids or {})
    block = ((support.get("contract_value_bids", {}) or {}).get("public_ownership_bids", {}) or {})
    if not (support.get("enabled", False) and block.get("enabled", False) and block.get("storage_at_cost", False)):
        return None, 0.0
    from scripts.market.cfd_bm_bids import resolve_public_ownership_bids

    shares, _ = resolve_public_ownership_bids(block, logger)
    shares = shares.reindex(units).fillna(0.0).clip(0.0, 1.0)
    guard = float(balancing.get("storage_churn_guard_gbp_per_mwh", DEFAULT_STORAGE_CHURN_GUARD) or 0.0)
    owned = shares[shares.gt(0.0)]
    logger.info("Item F8: %d storage units at least partly state-owned bid at cost (mean share %.2f): no bid "
                "margin, offer premium cut to the churn guard's minimum", len(owned),
                float(owned.mean()) if len(owned) else 0.0)
    return shares, guard


def _attach_storage_price_frames(
    network,
    balancing: dict,
    su_offer: pd.Series,
    su_bid: pd.Series,
    logger: logging.Logger,
    bm_bids: Optional[dict] = None,
) -> None:
    """Per-snapshot round_trip storage prices indexed to the Stage 1 price.

    ``bm_bids`` is the scenario's ``support.bm_bids`` block (item F8's storage-at-cost shares).

    For each round_trip unit whose carrier sets ``bid_margin``:
    ``bid_t = -(lambda_t - bid_margin)`` (ESO-cost) and
    ``offer_t = (lambda_t - bid_margin) / eta_rt + premium_eff``, where
    ``premium_eff`` is the unit's static offer less its static paid bid over
    eta_rt, i.e. the configured premium after the churn guard. The churn and
    cross-unit guards therefore hold in every snapshot (the cross-unit check is
    repeated per snapshot). Units without ``bid_margin`` keep their static
    prices. Leaves ``network._bm_su_offer_tv`` / ``_bm_su_bid_tv``; the static
    Series are unchanged and remain the fallback.
    """
    reference = str(balancing.get("storage_bid_reference", DEFAULT_STORAGE_BID_REFERENCE)).lower()
    if reference not in STORAGE_BID_REFERENCES:
        raise ValueError(
            f"market.balancing.storage_bid_reference must be one of {STORAGE_BID_REFERENCES}, got {reference!r}"
        )
    su = network.storage_units
    if reference == "fixed" or len(su) == 0:
        return
    price = getattr(network, "_bm_reference_price", None)
    if price is None:
        logger.warning(
            "storage_bid_reference stage1_price but no Stage 1 price on the network; "
            "storage keeps its static bid_price"
        )
        return
    overrides = balancing.get("carrier_overrides", {}) or {}
    indexed = [
        idx for idx in su.index
        if (overrides.get(su.at[idx, "carrier"], {}) or {}).get("mode") == "round_trip"
        and "bid_margin" in (overrides.get(su.at[idx, "carrier"], {}) or {})
    ]
    if not indexed:
        return
    snapshots = network.snapshots
    lam = pd.to_numeric(pd.Series(price), errors="coerce").reindex(snapshots)
    if lam.isna().any():
        logger.warning(f"Stage 1 price missing in {int(lam.isna().sum())} snapshots; filled forward/backward")
        lam = lam.ffill().bfill().fillna(0.0)
    if reference == "stage1_daily_mean":
        lam = lam.rolling(24, center=True, min_periods=1).mean()
    eta = (
        pd.to_numeric(su["efficiency_store"], errors="coerce").fillna(1.0)
        * pd.to_numeric(su["efficiency_dispatch"], errors="coerce").fillna(1.0)
    ).clip(lower=1e-6)
    offer_tv = pd.DataFrame(np.tile(su_offer.reindex(su.index).to_numpy(), (len(snapshots), 1)),
                            index=snapshots, columns=su.index)
    bid_tv = pd.DataFrame(np.tile(su_bid.reindex(su.index).to_numpy(), (len(snapshots), 1)),
                          index=snapshots, columns=su.index)
    state_shares, churn_guard = _state_storage_at_cost_shares(balancing, bm_bids, su.index, logger)
    for idx in indexed:
        spec = overrides[su.at[idx, "carrier"]]
        margin = float(spec["bid_margin"])
        premium = float(su_offer[idx]) + float(su_bid[idx]) / float(eta[idx])   # static offer - paid/eta
        if state_shares is not None:
            # Item F8: the state-owned share bids at the market value of the energy (no
            # margin) and offers its round trip plus only the churn guard's minimum.
            share = float(state_shares.get(idx, 0.0))
            guard_premium = churn_guard * (1.0 - float(eta[idx])) / float(eta[idx]) if churn_guard > 0.0 else 0.0
            margin = (1.0 - share) * margin
            premium = (1.0 - share) * premium + share * min(guard_premium, premium)
        paid = lam.to_numpy() - margin
        bid_tv[idx] = -paid
        offer_tv[idx] = paid / float(eta[idx]) + premium
        # offer_floor (11 Oct 2026): GB storage offers do not follow the market price down. 2024 ELEXON BOD,
        # first offer tranche in hours with MID below GBP 20: batteries 69, pumped storage 106 (against 102
        # and 123 overall), so in a near-zero 2030 hour the indexed formula (about 15-20) undercuts every real
        # offer. The private share keeps the floor; a state-owned share (item F8) still offers at cost.
        floor = float(spec.get("offer_floor", 0.0) or 0.0)
        if floor > 0.0:
            share = float(state_shares.get(idx, 0.0)) if state_shares is not None else 0.0
            offer_tv[idx] = np.maximum(offer_tv[idx].to_numpy(), (1.0 - share) * floor)
    for snapshot in snapshots[
        (offer_tv.min(axis=1) + bid_tv.min(axis=1)).to_numpy() < -1e-9
    ][:1]:
        _check_storage_cross_carrier_arbitrage(su, offer_tv.loc[snapshot], bid_tv.loc[snapshot])
    network._bm_su_offer_tv = offer_tv
    network._bm_su_bid_tv = bid_tv
    by_carrier = su.loc[indexed].groupby("carrier").size().to_dict()
    logger.info(
        f"Storage bids indexed to the Stage 1 price for {len(indexed)} units {by_carrier}: "
        f"paid bid = price - bid_margin, offer = paid / eta_rt + premium; Stage 1 price "
        f"mean {lam.mean():.2f} GBP/MWh, unit-mean ESO-cost bid "
        f"{float(bid_tv[indexed].mean().mean()):.2f}, offer {float(offer_tv[indexed].mean().mean()):.2f}"
    )


def _check_storage_cross_carrier_arbitrage(
    storage_units: pd.DataFrame, offer_prices: pd.Series, bid_prices: pd.Series
) -> None:
    """Refuse storage prices under which the BM profits from moving energy between units.

    Unit X discharging one MWh more (NESO pays ``offer_X``) while unit Y charges
    that MWh (NESO pays ``bid_Y``, negative when Y pays to charge) leaves the
    system balance unchanged, so it is a pure money pump unless
    ``offer_X + bid_Y >= 0`` for every pair: the cheapest storage offer must be
    at least the most any storage unit pays to charge. Checked over every
    storage unit priced here, before participation penalties are applied,
    except consumer flexibility (F5): it is demand, held at its Stage 1 schedule
    in the BM (``_apply_participation_policy``), so it never trades there and its
    carrier has no BM prices (it would otherwise fail every CP30 2030 month on
    DH_flexibility's 0 offer against the battery's 40 GBP/MWh bid).
    """
    flex = _demand_flex_units(storage_units)
    keep = storage_units.index[~flex.to_numpy()]
    offers = pd.to_numeric(offer_prices.reindex(keep), errors="coerce").dropna()
    bids = pd.to_numeric(bid_prices.reindex(keep), errors="coerce").dropna()
    if offers.empty or bids.empty:
        return
    cheapest, keenest = offers.idxmin(), bids.idxmin()
    if float(offers[cheapest]) + float(bids[keenest]) < -1e-9:
        carrier = storage_units["carrier"]
        raise ValueError(
            "Storage BM prices allow cross-unit arbitrage: the cheapest storage offer "
            f"({carrier.get(cheapest, cheapest)}, {float(offers[cheapest]):.2f} GBP/MWh) is below "
            f"what {carrier.get(keenest, keenest)} pays to charge "
            f"({-float(bids[keenest]):.2f} GBP/MWh), so the BM would discharge one unit into "
            "another for profit. Lower that bid_price or raise the offer_premium in "
            "market.balancing.carrier_overrides."
        )


def apply_storage_dispatch_cost_floor(
    network,
    stage_config: dict,
    logger: logging.Logger,
    key: str = "storage_marginal_cost_floor",
    default: float = 1.0,
) -> int:
    """Enforce a minimum storage dispatch cost before a market solve.

    With a zero marginal cost every storage unit is indifferent between charging
    and discharging in hours that clear at exactly zero, and a barrier solve
    without crossover (the configured default) returns interior solutions that
    charge and discharge the same unit simultaneously, dissipating energy
    through round-trip losses. The CP30 2030 run lost ~3.5 TWh of battery
    energy this way in Stage 1 (15.3 TWh charged against 9.7 TWh discharged at
    85% round-trip efficiency), and 3.6 TWh in Stage 2 before the BM objective
    priced gross ``p_dispatch`` (see solve_balancing.py). A small strictly
    positive cost breaks the tie. Shared by ``market.wholesale.<key>`` and
    ``market.balancing.<key>``; 0 disables.
    """
    floor = float(stage_config.get(key, default) or 0.0)
    if floor <= 0.0 or len(network.storage_units) == 0:
        return 0
    current = pd.to_numeric(network.storage_units["marginal_cost"], errors="coerce").fillna(0.0)
    below = current < floor
    if below.any():
        network.storage_units.loc[below, "marginal_cost"] = floor
        by_carrier = network.storage_units.loc[below].groupby("carrier").size().to_dict()
        logger.info(
            f"Storage dispatch cost floor {floor:.2f} GBP/MWh applied to "
            f"{int(below.sum())} units below it: {by_carrier}"
        )
    return int(below.sum())


def apply_storage_churn_share(network, stage_config: dict, logger: logging.Logger) -> int:
    """Limit how much storage may churn (charge and discharge at once) in a market solve.

    CP30's PLEXOS lets storage charge and discharge simultaneously, so its market burns
    part of the surplus through round-trip losses instead of curtailing it (follow-ups
    F16, F22). ``market.wholesale.storage_churn_share`` (0 to 1, absent = no change)
    calibrates how much of that a NESO-comparable case reproduces: that share of each
    storage carrier's power, by unit in name order, keeps its cost; every other unit's
    dispatch cost is raised to where churning to save a curtailed AR4+ unit (worth the
    AR4+ bid, ``renewable_curtailment_order.ar4_plus_bid_gbp_per_mwh``) no longer pays:
    bid x (1 - round trip) / round trip + 1 GBP/MWh. Returns the number of units raised.
    """
    share = stage_config.get("storage_churn_share")
    if share is None or len(network.storage_units) == 0:
        return 0
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"storage_churn_share must lie in [0, 1], not {share}")
    order = stage_config.get("renewable_curtailment_order", {}) or {}
    gain = abs(float(order.get("ar4_plus_bid_gbp_per_mwh", 0.0) or 0.0))
    # Consumer flexibility (demand, F5) is never part of the storage fleet's churn share.
    su = network.storage_units[~_demand_flex_units(network.storage_units)]
    rte = (pd.to_numeric(su["efficiency_store"], errors="coerce").fillna(1.0)
           * pd.to_numeric(su["efficiency_dispatch"], errors="coerce").fillna(1.0)).clip(lower=1e-6, upper=1.0)
    no_churn = gain * (1.0 - rte) / rte + 1.0
    raised = []
    for carrier, units in su.groupby("carrier"):
        units = units.sort_index()
        cum = units["p_nom"].cumsum() / max(float(units["p_nom"].sum()), 1e-9)
        blocked = units.index[(cum - units["p_nom"] / max(float(units["p_nom"].sum()), 1e-9)) >= share - 1e-12]
        cost = pd.to_numeric(su.loc[blocked, "marginal_cost"], errors="coerce").fillna(0.0)
        target = no_churn.loc[blocked]
        up = blocked[cost < target]
        network.storage_units.loc[up, "marginal_cost"] = target.loc[up]
        raised += list(up)
        logger.info(f"Storage churn share {share:.2f}, {carrier}: {len(units) - len(blocked)} of {len(units)} units "
                    f"({1 - float(units.loc[blocked, 'p_nom'].sum()) / max(float(units['p_nom'].sum()), 1e-9):.0%} of "
                    f"{float(units['p_nom'].sum()):,.0f} MW) may churn; {len(up)} raised to "
                    f"{float(target.mean()) if len(target) else 0:.2f} GBP/MWh")
    return len(raised)


def _demand_flex_units(storage_units: pd.DataFrame) -> pd.Series:
    """True for consumer-flexibility storage units (``demand_carriers.DEMAND_FLEX_STORAGE_CARRIERS``)."""
    from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS

    return storage_units["carrier"].astype(str).isin(DEMAND_FLEX_STORAGE_CARRIERS)


#: Load carrier of the exogenous hydrogen offtake (scripts/hydrogen/add_hydrogen_system.py).
HYDROGEN_OFFTAKE_CARRIER = "H2_demand"


def apply_hydrogen_offtake_scale(network, market_config: dict, logger: logging.Logger) -> int:
    """Scale the exogenous hydrogen offtake at solve time (``market.hydrogen_offtake_scale``).

    The offtake (``hydrogen.exogenous_demand``) is a flat Load on the GB hydrogen bus written at build, and every
    month of a family shares one build (``network_from``), so a Stage 1 variant cannot change it through the
    hydrogen config. CP.28-30 calibration (17 Oct 2026) aims electrolysis at NESO's CP.28 (14.22 TWh) rather than ES1 (14.86):
    this multiplies every ``H2_demand`` load's ``p_set`` (static and hourly) by the scale. Applied by both stages
    straight after loading the network, as the storage round trips are, so the balancing stage balances the
    hydrogen bus Stage 1 solved. Absent or None: no change (the code default; ``config/defaults/`` documents
    the key and does not set it). Returns the number of loads scaled.
    """
    scale = (market_config or {}).get("hydrogen_offtake_scale")
    if scale is None:
        return 0
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"market.hydrogen_offtake_scale must be a finite number >= 0, not {scale}")
    loads = network.loads.index[network.loads["carrier"] == HYDROGEN_OFFTAKE_CARRIER]
    if len(loads) == 0:
        raise ValueError("market.hydrogen_offtake_scale is set but the network has no hydrogen offtake "
                         f"(no Load with carrier {HYDROGEN_OFFTAKE_CARRIER!r})")
    before = float(network.loads.loc[loads, "p_set"].sum())
    network.loads.loc[loads, "p_set"] = network.loads.loc[loads, "p_set"] * scale
    p_set_t = network.loads_t.get("p_set")
    timed = [n for n in loads if p_set_t is not None and n in p_set_t.columns]
    if timed:
        network.loads_t.p_set[timed] = p_set_t[timed] * scale
    logger.info(f"Hydrogen offtake scaled x{scale:.4f} (market.hydrogen_offtake_scale): {len(loads)} load(s), "
                f"static {before:,.1f} -> {before * scale:,.1f} MW_H2" + (f", {len(timed)} hourly" if timed else ""))
    return len(loads)


#: The FES long-range hydrogen system's gas reformer with CCS (scripts/hydrogen/fes_hydrogen.REFORMER).
HYDROGEN_REFORMER_LINK = "GB_gas_reforming_ccs"


def apply_hydrogen_supply_price(network, market_config: dict, logger: logging.Logger) -> dict | None:
    """Price the reformer's hydrogen at an external long-run cost (``market.hydrogen_supply_price``).

    2050 price calibration (user, 25 Oct 2026): the external GB 2050 price studies take hydrogen at a price set
    outside the power model, DESNZ's blue hydrogen cost (LCP Delta for DESNZ 2024 p.63; FTI for NESO 2025),
    where our build prices the reformer at its short-run cost, so the endogenous hydrogen value never carries the
    capital cost. ``{enabled, gbp_per_mwh_h2, link, source}``: the reformer link's marginal cost is set so that its
    cost per MWh of hydrogen out, (gas supply cost + link cost) / efficiency, equals ``gbp_per_mwh_h2``. Applied by
    the annual pass, Stage 1 and the balancing stage straight after loading the network, as the hydrogen offtake
    scale is. Off unless ``enabled`` (absent everywhere but the 2050 study). Returns a report, or None when off.
    """
    cfg = dict((market_config or {}).get("hydrogen_supply_price") or {})
    if not cfg.get("enabled", False):
        return None
    target = float(cfg["gbp_per_mwh_h2"])
    name = str(cfg.get("link", HYDROGEN_REFORMER_LINK))
    if name not in network.links.index:
        raise ValueError(f"market.hydrogen_supply_price: the network has no link {name!r}")
    eta = float(network.links.at[name, "efficiency"])
    gas_bus = network.links.at[name, "bus0"]
    supply = network.generators.index[network.generators["bus"].eq(gas_bus)]
    if len(supply) == 0:
        raise ValueError(f"market.hydrogen_supply_price: no gas supply generator on {gas_bus!r}")
    gas_cost = pd.to_numeric(network.generators.loc[supply, "marginal_cost"], errors="coerce")
    if gas_cost.nunique() != 1 or (getattr(network.generators_t, "marginal_cost", pd.DataFrame()).columns
                                   .isin(supply).any()):
        raise ValueError("market.hydrogen_supply_price needs one flat gas supply cost on the reformer's gas bus")
    gas = float(gas_cost.iloc[0])
    link_cost = float(network.links.at[name, "marginal_cost"])
    short_run = (gas + link_cost) / eta
    if target < short_run - 1e-9:
        raise ValueError(f"market.hydrogen_supply_price {target} is below the reformer's short-run cost {short_run:.2f}")
    timed = getattr(network.links_t, "marginal_cost", pd.DataFrame())
    if name in timed.columns:
        network.links_t.marginal_cost = timed.drop(columns=[name])
    network.links.at[name, "marginal_cost"] = target * eta - gas
    report = {"link": name, "short_run_gbp_per_mwh_h2": short_run, "gbp_per_mwh_h2": target,
              "efficiency": eta, "gas_gbp_per_mwh": gas, "source": str(cfg.get("source", ""))}
    logger.info(f"Hydrogen supply priced at {target:.2f} GBP/MWh_H2 (short-run {short_run:.2f}; "
                f"market.hydrogen_supply_price): {name}")
    return report


def apply_storage_round_trip_override(network, market_config: dict, logger: logging.Logger) -> int:
    """Set storage round-trip efficiencies by carrier at solve time (``market.storage_round_trip_override``).

    ``{carrier: round_trip}``; each leg becomes ``sqrt(round_trip)``. Applied by both stages straight after
    loading the network, so they see the same storage. CP30's own round trips, implied by its data workbook
    (output / charging, FFR 2030): batteries 0.70 (ES1 17,946 / 25,660 GWh; N2 21,784 / 31,120), pumped hydro
    0.74 (N2 9,362 / 12,684), liquid air 0.55 (N2 2,020 / 3,673), long-duration storage overall 0.70 (ES1).
    Absent or empty: no change. Returns the number of units changed.
    """
    override = (market_config or {}).get("storage_round_trip_override") or {}
    if not override or len(network.storage_units) == 0:
        return 0
    changed = 0
    for carrier, rte in override.items():
        rte = float(rte)
        if not 0.0 < rte <= 1.0:
            raise ValueError(f"storage_round_trip_override[{carrier!r}] must lie in (0, 1], not {rte}")
        units = network.storage_units.index[network.storage_units["carrier"] == carrier]
        if len(units) == 0:
            continue
        leg = rte ** 0.5
        network.storage_units.loc[units, "efficiency_store"] = leg
        network.storage_units.loc[units, "efficiency_dispatch"] = leg
        changed += len(units)
        logger.info(f"Storage round trip for '{carrier}' set to {rte:.3f} ({leg:.4f} per leg) on {len(units)} units "
                    "(market.storage_round_trip_override)")
    return changed


#: ``market.wholesale.churn_sink`` (11 Oct 2026): CP30's market burns surplus through simultaneous storage
#: charge/discharge (F16, F22). ``storage_churn_share`` reproduces that by leaving part of the storage fleet free
#: to cycle, but the same free storage then arbitrages every day and pins shoulder prices near zero (storage sets
#: 48% of churn_neso's hours, at a mean GBP 58, against CP30's 38% of hours above GBP 100). The sink separates the
#: two: a negative-only generator at each storage bus, ``share_of_storage_power`` x that bus's storage power, priced
#: just above the AR4+ curtailment bid, absorbs surplus only where the market would otherwise curtail, so storage
#: can carry its own cycling cost (``storage_marginal_cost_floor``) without changing curtailment. The balancing
#: stage holds it at its Stage 1 output (``fix_churn_sink``).
CHURN_SINK_CARRIER = "churn_sink"


def _churn_sink_frame(network, stage_config: dict):
    """(names, buses, p_nom, price) of the sink generators a config asks for, or None."""
    cfg = (stage_config or {}).get("churn_sink") or {}
    if not cfg.get("enabled", False) or len(network.storage_units) == 0:
        return None
    share = float(cfg.get("share_of_storage_power", 0.0))
    if share <= 0.0:
        return None
    order = (stage_config or {}).get("renewable_curtailment_order", {}) or {}
    default_price = float(order.get("ar4_plus_bid_gbp_per_mwh", 0.0) or 0.0) + 0.01
    price = float(cfg.get("price_gbp_per_mwh", default_price))
    # Sized on storage plant only: consumer flexibility (demand, F5) carries no sink.
    plant = network.storage_units[~_demand_flex_units(network.storage_units)]
    per_bus = plant.groupby("bus")["p_nom"].sum() * share
    per_bus = per_bus[per_bus > 0.0]
    names = pd.Index([f"churn_sink {bus}" for bus in per_bus.index])
    return names, pd.Index(per_bus.index), per_bus.to_numpy(), price


def apply_churn_sink(network, stage_config: dict, logger: logging.Logger) -> int:
    """Add the market stage's churn sink (``market.wholesale.churn_sink``); returns the number of sink units."""
    frame = _churn_sink_frame(network, stage_config)
    if frame is None:
        return 0
    names, buses, p_nom, price = frame
    if CHURN_SINK_CARRIER not in network.carriers.index:
        network.add("Carrier", CHURN_SINK_CARRIER)
    network.add("Generator", names, bus=buses, carrier=CHURN_SINK_CARRIER, p_nom=p_nom,
                p_min_pu=-1.0, p_max_pu=0.0, marginal_cost=price)
    logger.info(f"Churn sink: {len(names)} units, {p_nom.sum():,.0f} MW in all, absorbing surplus at or below "
                f"{price:.2f} GBP/MWh (market.wholesale.churn_sink)")
    return len(names)


def churn_sink_columns(columns) -> list:
    """The churn sink units among ``columns`` (named ``churn_sink <bus>``)."""
    return [c for c in columns if str(c).startswith(f"{CHURN_SINK_CARRIER} ")]


def align_redispatch_generators(wholesale_gen: pd.DataFrame, physical_gen: pd.DataFrame, snapshots,
                                logger: logging.Logger | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Wholesale and physical generator tables for the redispatch accounting, without the churn sink.

    Item F22 (13 Oct 2026). ``fix_churn_sink`` pins the sink to its Stage 1 output and drops its columns from
    the wholesale table (it is no anchor), so aligning that table on the physical one filled the sink's Stage 1
    position with zero and its held absorption (about 4.4 TWh a year, at zero cost) showed as a balancing
    turn-down. The sink is not a balancing action: it is left out of the volumes, and its held energy is logged
    on its own line. Returns ``(wholesale, physical, sink_mwh)`` with ``sink_mwh`` >= 0 the energy it absorbed.
    """
    sink = churn_sink_columns(physical_gen.columns)
    physical = physical_gen.drop(columns=sink)
    wholesale = wholesale_gen.reindex(snapshots).reindex(columns=physical.columns, fill_value=0.0)
    sink_mwh = 0.0
    if sink:
        index = pd.DatetimeIndex(physical_gen.index)
        if len(index) > 1:
            step = pd.Series(index, index=index).diff().dt.total_seconds().div(3600.0)
            hours = step.fillna(step.median()).to_numpy()
        else:
            hours = np.ones(len(index))
        sink_mwh = float(-(physical_gen[sink].clip(upper=0.0).to_numpy() * hours[:, None]).sum())
        if logger is not None:
            logger.info(f"Churn sink held at its Stage 1 output: {sink_mwh / 1e3:,.1f} GWh absorbed by "
                        f"{len(sink)} units; not a balancing action, so left out of the redispatch volumes")
    return wholesale, physical, sink_mwh


#: ``market.balancing.churn_sink`` (24 Oct 2026): ``held`` (default) pins the sink to its Stage 1 output;
#: ``free`` is the consistency check: the BM may move each unit anywhere between zero and its Stage 1 peak
#: absorption, anchored on Stage 1 like any generator and priced by ``carrier_overrides.churn_sink``. It is a
#: balancing key so a family can share its parent's Stage 1 (``wholesale_from``).
CHURN_SINK_BALANCING_MODES = ("held", "free")


def _rehomed_storage_buses(network) -> dict:
    """Dropped bus -> bus its storage moved to, from the network-delivery record (largest unit's move wins)."""
    raw = (network.meta or {}).get("network_delivery")
    if not raw:
        return {}
    record = json.loads(raw) if isinstance(raw, str) else raw
    moves = [m for m in (record.get("appendix_b") or {}).get("rehomed", []) if m.get("component") == "storage_unit"]
    out = {}
    for m in sorted(moves, key=lambda m: m.get("p_nom", 0.0)):
        out[m["from_bus"]] = m["to_bus"]
    return out


def fix_churn_sink(network, stage_config: dict, wholesale_gen: pd.DataFrame, logger: logging.Logger,
                   balancing_mode: str = "held") -> pd.DataFrame:
    """Balancing stage: add the sink held at its Stage 1 output; returns ``wholesale_gen`` without its columns.

    The sink stands for losses CP30's market burnt, so the balancing stage neither redispatches nor prices it:
    its bounds are pinned to the Stage 1 output hour by hour and it is left out of the redispatch anchors.
    The units are Stage 1's own (named ``churn_sink <bus>``): Stage 1 solves on a collapsed copy of the network,
    whose storage sits on fewer buses than the full network's, so the layout is read off Stage 1's output rather
    than rebuilt here. A config with the sink on and a Stage 1 without it (or the reverse) is refused, as is a sink
    bus the balancing network lacks.

    Under ``balancing_mode="free"`` the bounds are not pinned and the columns stay in ``wholesale_gen``, so the
    sink is an anchored unit the BM may redispatch (a check on what holding it costs, never a central case).
    """
    frame = _churn_sink_frame(network, stage_config)
    sink_cols = churn_sink_columns(wholesale_gen.columns)
    if frame is None:
        if sink_cols:
            raise ValueError("Stage 1 carries a churn sink but market.wholesale.churn_sink is off")
        return wholesale_gen
    price = frame[3]
    if not sink_cols:
        raise ValueError("Stage 1 has no output for any churn sink unit: re-solve Stage 1 with market.wholesale.churn_sink")
    names = pd.Index(sink_cols)
    buses = pd.Index([str(c)[len("churn_sink "):] for c in sink_cols])
    # A sink sits with Stage 1's storage, so it follows the storage the network-delivery revert re-homed off a
    # dropped bus (network_delivery.rehome_stranded_max_mw).
    rehomed = _rehomed_storage_buses(network)
    moved = {b: rehomed[b] for b in buses if b not in network.buses.index and b in rehomed}
    if moved:
        buses = pd.Index([moved.get(b, b) for b in buses])
        logger.info(f"Churn sink re-homed with its storage: {len(moved)} units "
                    f"({', '.join(f'{a} -> {b}' for a, b in sorted(moved.items()))})")
    absent = [b for b in buses if b not in network.buses.index]
    if absent:
        raise ValueError(f"{len(absent)} Stage 1 churn sink buses are not in the balancing network (e.g. {absent[0]})")
    mode = str(balancing_mode or "held").lower()
    if mode not in CHURN_SINK_BALANCING_MODES:
        raise ValueError(f"market.balancing.churn_sink must be one of {CHURN_SINK_BALANCING_MODES}, got {mode!r}")
    position = wholesale_gen[sink_cols].reindex(network.snapshots).fillna(0.0)
    p_nom = (-position.min()).clip(lower=1.0).to_numpy()      # enough to carry the fixed Stage 1 output
    if CHURN_SINK_CARRIER not in network.carriers.index:
        network.add("Carrier", CHURN_SINK_CARRIER)
    network.add("Generator", names, bus=buses, carrier=CHURN_SINK_CARRIER, p_nom=p_nom,
                p_min_pu=-1.0, p_max_pu=0.0, marginal_cost=price)
    if mode == "free":
        logger.info(f"Churn sink free in the balancing stage: {len(names)} units, {p_nom.sum():,.0f} MW (each "
                    f"unit's Stage 1 peak), anchored on {-float(position.sum().sum()) / 1e3:,.1f} GWh of Stage 1 "
                    "absorption (market.balancing.churn_sink: free)")
        return wholesale_gen
    pu = position.div(pd.Series(p_nom, index=names), axis=1).clip(lower=-1.0, upper=0.0)
    network.generators_t.p_min_pu = pd.concat([network.generators_t.p_min_pu.drop(columns=list(names), errors="ignore"), pu], axis=1)
    network.generators_t.p_max_pu = pd.concat([network.generators_t.p_max_pu.drop(columns=list(names), errors="ignore"), pu], axis=1)
    logger.info(f"Churn sink held at Stage 1: {len(names)} units, {-float(position.sum().sum()) / 1e3:,.1f} GWh absorbed "
                "over the period")
    return wholesale_gen.drop(columns=sink_cols)


# ── Storage state of charge at the month boundary ────────────────────────
#
# Each CP30 month is an independent run that starts every storage unit at the
# network's state_of_charge_initial, half its energy capacity (add_storage.py).
# Nothing asked for it back: the rolling day-ahead stage has no terminal value,
# so day 1 sold the 74 GWh (24 battery + 50 LAES) the run was handed, 0.89 TWh a
# year of free energy. Measured on the 2030 months (wholesale storage CSVs,
# 26 Sep 2026) the fleet holds very different levels at midnight - mean 9% of
# battery capacity in July, 65% in December, 0-99% day to day - so no single
# starting level is "right". ``month_end: at_least_initial`` makes each month
# energy-neutral instead: the window that holds the solve period's last snapshot
# must leave every unit with at least what it started with (cyclic per month in
# rolling form), so whatever the month starts with it pays for. The target is
# capped at what the unit could charge within that window from empty, so the
# constraint can never make the window infeasible. ``initial_fraction`` (null
# keeps the network's value) sets the starting level; the balancing stage applies
# the same helper so both stages start from one state. Defaults must match
# config/defaults/ market.storage_soc.
STORAGE_SOC_MONTH_END_MODES = ("at_least_initial", "free")


def resolve_storage_soc_config(market_config: dict) -> dict:
    """``market.storage_soc`` with defaults filled and validated."""
    cfg = dict((market_config or {}).get("storage_soc", {}) or {})
    fraction = cfg.get("initial_fraction")
    if fraction is not None:
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"market.storage_soc.initial_fraction must be in [0, 1], got {fraction}")
    month_end = str(cfg.get("month_end", "at_least_initial")).lower()
    if month_end not in STORAGE_SOC_MONTH_END_MODES:
        raise ValueError(
            f"market.storage_soc.month_end must be one of {STORAGE_SOC_MONTH_END_MODES}, got {month_end!r}"
        )
    if _seasonal_storage_units_cfg(market_config) is not None:
        # Decision 33: units start on the annual pass and their period end is valued there.
        if month_end != "free":
            raise ValueError(
                "market.hydrogen_store.seasonal_trajectory.storage_units needs market.storage_soc.month_end "
                "'free': at_least_initial would hold the end hard at the trajectory's start state")
        if fraction is not None:
            raise ValueError(
                "market.storage_soc.initial_fraction conflicts with seasonal_trajectory.storage_units "
                "(the start comes from the annual pass); set it to null")
    carriers = cfg.get("carriers")
    return {
        "initial_fraction": fraction,
        "month_end": month_end,
        "carriers": None if not carriers else tuple(str(c) for c in carriers),
    }


def _soc_units(network, carriers) -> pd.Index:
    su = network.storage_units
    if len(su) == 0:
        return pd.Index([])
    energy = pd.to_numeric(su["p_nom"], errors="coerce").fillna(0.0) * pd.to_numeric(
        su["max_hours"], errors="coerce"
    ).fillna(0.0)
    keep = energy > 0
    if carriers is not None:
        keep &= su["carrier"].isin(set(carriers))
    if "cyclic_state_of_charge" in su.columns:
        keep &= ~su["cyclic_state_of_charge"].fillna(False).astype(bool)
    return su.index[keep]


def apply_initial_storage_soc(network, market_config: dict, logger: logging.Logger) -> pd.Series:
    """Set the run's starting state of charge (if configured) and return it per unit (MWh)."""
    cfg = resolve_storage_soc_config(market_config)
    su = network.storage_units
    if len(su) == 0:
        return pd.Series(dtype=float)
    units = _soc_units(network, cfg["carriers"])
    if cfg["initial_fraction"] is not None and len(units):
        energy = (su.loc[units, "p_nom"] * su.loc[units, "max_hours"]).astype(float)
        network.storage_units.loc[units, "state_of_charge_initial"] = cfg["initial_fraction"] * energy
        logger.info(
            f"Storage starts at {100 * cfg['initial_fraction']:.0f}% of energy capacity "
            f"({len(units)} units, {cfg['initial_fraction'] * energy.sum() / 1e3:,.1f} GWh)"
        )
    if len(units) and _seasonal_storage_units_cfg(market_config) is not None:
        # Decision 33 (seasonal_trajectory.storage_units): start where the annual pass has them.
        apply_trajectory_storage_soc(network, market_config, units, logger)
    initial =pd.to_numeric(network.storage_units["state_of_charge_initial"], errors="coerce").fillna(0.0)
    logger.info(
        f"Storage initial state of charge: {initial.loc[units].sum() / 1e3:,.1f} GWh over "
        f"{len(units)} units; month end {cfg['month_end']}"
    )
    return initial


def build_month_end_soc_callback(network, market_config: dict, final_snapshot, logger: logging.Logger):
    """``extra_functionality`` holding each unit's final state of charge at or above its start.

    Acts only in the optimisation whose snapshots contain ``final_snapshot`` (the
    last window of a rolling solve, or the single full-period solve). Returns
    None when ``month_end`` is ``free`` or there is no storage.
    """
    cfg = resolve_storage_soc_config(market_config)
    if _seasonal_storage_units_cfg(market_config) is not None:
        # Decision 33 (seasonal_trajectory.storage_units): the end is valued, not held.
        units = _soc_units(network, cfg["carriers"])
        if not len(units):
            return None
        return _build_trajectory_storage_end_value_callback(network, market_config, units, final_snapshot, logger)
    if cfg["month_end"] == "free":
        logger.info("Storage month end: free (no terminal state-of-charge condition)")
        return None
    units = _soc_units(network, cfg["carriers"])
    if not len(units):
        return None
    su = network.storage_units.loc[units]
    start = pd.to_numeric(su["state_of_charge_initial"], errors="coerce").fillna(0.0)

    def month_end_soc(n, sns):
        if final_snapshot not in pd.Index(sns):
            return
        model = n.model
        if "StorageUnit-state_of_charge" not in model.variables:
            return
        soc = model.variables["StorageUnit-state_of_charge"]
        dim = [d for d in soc.dims if d != "snapshot"][0]
        names = [u for u in units if u in set(soc.coords[dim].values)]
        if not names:
            return
        steps = len(sns)
        hours = 1.0
        if steps > 1:
            hours = float((pd.Index(sns)[1] - pd.Index(sns)[0]).total_seconds() / 3600.0)
        elapsed = steps * hours
        weights = n.snapshot_weightings.stores.reindex(pd.Index(sns)).fillna(1.0)
        if weights.nunique() > 1:
            # A coarse look-ahead (blocks weighted by their hours): the window's span is the weighted sum.
            elapsed = float(weights.sum())
        units_df = n.storage_units.loc[names]
        charge_mw = (
            pd.to_numeric(units_df["p_nom"], errors="coerce").fillna(0.0)
            * pd.to_numeric(units_df["p_min_pu"], errors="coerce").fillna(-1.0).abs()
            * pd.to_numeric(units_df["efficiency_store"], errors="coerce").fillna(1.0)
        )
        retention = (1.0 - pd.to_numeric(units_df["standing_loss"], errors="coerce").fillna(0.0)) ** elapsed
        reachable = 0.95 * charge_mw * elapsed * retention
        target = np.minimum(start.reindex(names).to_numpy(), reachable.to_numpy())
        import xarray as xr

        target_da = xr.DataArray(target, dims=[dim], coords={dim: pd.Index(names, name=dim)})
        model.add_constraints(
            soc.sel({dim: names}).sel(snapshot=final_snapshot) >= target_da,
            name="StorageUnit-month_end_soc",
        )
        capped = int((reachable.to_numpy() < start.reindex(names).to_numpy() - 1e-6).sum())
        logger.info(
            f"Storage month end: {len(names)} units held at or above their starting state of charge "
            f"({target.sum() / 1e3:,.1f} GWh) at {final_snapshot}"
            + (f"; {capped} targets capped at what the window can recharge" if capped else "")
        )
        # Consumer flexibility (F5) is demand: it must also end no fuller than it started, or a
        # zero-price last evening lets it buy free surplus it never uses (F5 test, 20 Oct 2030 week:
        # 59 GWh, +1.1% of the week's demand; up to ~160 GWh a month, the units' whole energy).
        flex_names = [u for u in names if str(n.storage_units.at[u, "carrier"]) in _DEMAND_FLEX_CARRIERS()]
        if flex_names:
            # Soft (a hard cap is infeasible when a drained unit enters the last window holding more
            # than that window's drain): the excess is priced at FLEX_MONTH_END_EXCESS_GBP_PER_MWH.
            flex_idx = pd.Index(flex_names, name=dim)
            cap_da = xr.DataArray(start.reindex(flex_names).to_numpy(), dims=[dim], coords={dim: flex_idx})
            excess = model.add_variables(lower=xr.DataArray(np.zeros(len(flex_idx)), dims=[dim], coords={dim: flex_idx}),
                                         name="StorageUnit-month_end_flex_excess")
            model.add_constraints(
                soc.sel({dim: flex_names}).sel(snapshot=final_snapshot) - excess <= cap_da,
                name="StorageUnit-month_end_soc_consumer_flex",
            )
            model.objective = model.objective + (excess * FLEX_MONTH_END_EXCESS_GBP_PER_MWH).sum()
            logger.info(f"Storage month end: {len(flex_names)} consumer flexibility units end at or below their start")

    return month_end_soc


#: Objective-only price on consumer-flexibility energy left above its start at the period end.
FLEX_MONTH_END_EXCESS_GBP_PER_MWH = 1000.0


def _DEMAND_FLEX_CARRIERS():
    from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS

    return set(DEMAND_FLEX_STORAGE_CARRIERS)


# ── Hydrogen store level across rolling windows (decision D5, 27 Sep 2026) ──
#
# The CP30 hydrogen store (GB_H2_storage, ~93 GWh) is built ``e_cyclic``, which a
# rolling solve applies per window: every day (Stage 1) or every hour (a 1 h BM
# window) had to end where it began, so hydrogen made on a windy day could never
# be burnt on a still one, and a single-snapshot BM window could not move the
# store at all. ``market.hydrogen_store.carry_level`` treats the store like a
# storage unit's state of charge instead: the run starts it at
# ``initial_fraction`` of ``e_nom``, each window starts where the previous one
# ended (``e_initial``), and with ``month_end: at_least_initial`` the window
# holding the solve period's last snapshot must end at or above the run's start,
# capped at that window's starting level plus what the electrolysers can make in
# it net of the offtake, so the condition can never be infeasible. Both stages call the same helpers:
#
#     start = prepare_carried_stores(network, market_config, logger)  # once, full network
#     callback = build_month_end_store_callback(network, market_config, start, last_snapshot, logger)
#     for each window:  apply_carried_store_levels(window_net, levels)
#                       solve(..., extra_functionality includes callback)
#                       levels = final_store_levels(window_net, start.index)
#
# End-of-window value (user decision, 27 Sep 2026). A myopic 24 h window puts no
# value on hydrogen left at its end, so Stage 1 ran the store down and made only
# the day's offtake even in curtailed surplus hours (Dec 8-14 2030 week: 1.83 of
# 3.65 GW electrolysing at a 0-0.5 GBP/MWh price, the store at 22.9 of 92.8 GWh).
# ``end_of_window_value_gbp_per_mwh_h2`` (V) adds ``-V x e(store, last snapshot)``
# to each Stage 1 rolling window's objective (build_end_of_window_store_value_callback),
# so electrolysers bank hydrogen whenever electricity costs less than
# ``efficiency x V``. Objective only: the caller adds ``store_end_of_window_credit``
# back to the reported objective. The BM does not use it (it keeps its band).
#
# Defaults must match config/defaults/ market.hydrogen_store.
# With market.hydrogen_store.seasonal_trajectory (decision 33, task Y8a) month_end must
# be "free": the period end is valued (and optionally soft-floored) on the trajectory,
# never held hard (scripts/market/seasonal_storage.py).
HYDROGEN_STORE_MONTH_END_MODES = ("at_least_initial", "free")
DEFAULT_CARRIED_STORE_CARRIERS = ("H2_gas",)


def resolve_store_level_config(market_config: dict) -> dict:
    """``market.hydrogen_store`` with defaults filled and validated."""
    cfg = dict((market_config or {}).get("hydrogen_store", {}) or {})
    fraction = cfg.get("initial_fraction", 0.5)
    if fraction is not None:
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"market.hydrogen_store.initial_fraction must be in [0, 1], got {fraction}")
    month_end = str(cfg.get("month_end", "at_least_initial")).lower()
    if month_end == "trajectory":
        raise ValueError(
            "market.hydrogen_store.month_end 'trajectory' is retired (a hard end pin shed 815 GWh in the Y8a "
            "gate week): use month_end 'free' with seasonal_trajectory.end_condition 'value' or 'soft_floor'"
        )
    if month_end not in HYDROGEN_STORE_MONTH_END_MODES:
        raise ValueError(
            f"market.hydrogen_store.month_end must be one of {HYDROGEN_STORE_MONTH_END_MODES}, got {month_end!r}"
        )
    if month_end != "free" and _seasonal_trajectory_cfg(market_config) is not None:
        raise ValueError(
            "market.hydrogen_store.seasonal_trajectory needs month_end 'free': the start is pinned to the "
            "trajectory, so at_least_initial would hold the end hard at the trajectory's start level"
        )
    value = cfg.get("end_of_window_value_gbp_per_mwh_h2", None)
    if value is not None:
        value = float(value)
        if not value >= 0.0:
            raise ValueError(
                f"market.hydrogen_store.end_of_window_value_gbp_per_mwh_h2 must be >= 0 or null, got {value}"
            )
    return {
        "carry_level": bool(cfg.get("carry_level", True)),
        "initial_fraction": fraction,
        "month_end": month_end,
        "carriers": tuple(str(c) for c in (cfg.get("carriers") or DEFAULT_CARRIED_STORE_CARRIERS)),
        "end_of_window_value_gbp_per_mwh_h2": value,
    }


def _seasonal_trajectory_cfg(market_config: dict):
    """The enabled ``hydrogen_store.seasonal_trajectory`` block (decision 33), else None.

    None without the key, so every scenario that does not set it takes the code paths
    exactly as before (scripts/market/seasonal_storage.py).
    """
    block = ((market_config or {}).get("hydrogen_store", {}) or {}).get("seasonal_trajectory")
    if not block:
        return None
    from scripts.market.seasonal_storage import resolve_seasonal_trajectory_config

    cfg = resolve_seasonal_trajectory_config(market_config)
    return cfg if cfg["enabled"] else None


def carried_store_names(network, market_config: dict) -> pd.Index:
    """Stores whose level is carried between windows (empty when ``carry_level`` is off)."""
    cfg = resolve_store_level_config(market_config)
    stores = getattr(network, "stores", None)
    if not cfg["carry_level"] or stores is None or len(stores) == 0:
        return pd.Index([])
    e_nom = pd.to_numeric(stores["e_nom"], errors="coerce").fillna(0.0)
    return stores.index[stores["carrier"].isin(set(cfg["carriers"])) & (e_nom > 0)]


def prepare_carried_stores(network, market_config: dict, logger: logging.Logger) -> pd.Series:
    """Switch carried stores from per-solve cyclic to a carried level; return each one's start (MWh).

    Call once on the full-period network before the windows are cut (both
    stages, so both start from one level). Returns an empty Series when there is
    nothing to carry, in which case the stores keep their built behaviour.
    """
    cfg = resolve_store_level_config(market_config)
    names = carried_store_names(network, market_config)
    if not len(names):
        if not cfg["carry_level"] and len(getattr(network, "stores", [])):
            logger.info("Hydrogen store: carry_level off, stores keep their built (per-window cyclic) level rule")
        return pd.Series(dtype=float)
    stores = network.stores
    e_nom = pd.to_numeric(stores.loc[names, "e_nom"], errors="coerce").fillna(0.0)
    if cfg["initial_fraction"] is not None:
        network.stores.loc[names, "e_initial"] = cfg["initial_fraction"] * e_nom
    network.stores.loc[names, "e_cyclic"] = False
    trajectory_cfg = _seasonal_trajectory_cfg(market_config)
    if trajectory_cfg is not None:
        # Decision 33: start where the annual pass has the store at this period's start.
        from scripts.market.seasonal_storage import load_seasonal_trajectory, store_shares, trajectory_start_level

        trajectory = load_seasonal_trajectory(trajectory_cfg["file"])
        level = trajectory_start_level(trajectory, network.snapshots[0])
        shares = store_shares(stores, names)
        network.stores.loc[names, "e_initial"] = np.minimum(level * shares.reindex(names).to_numpy(), e_nom.to_numpy())
        logger.info(
            f"Hydrogen store: seasonal trajectory {trajectory_cfg['file']} sets the start at "
            f"{level / 1e3:,.1f} GWh ({100 * level / max(float(e_nom.sum()), 1e-9):.1f}% of e_nom) at "
            f"{network.snapshots[0]}"
        )
    start = pd.to_numeric(network.stores.loc[names, "e_initial"], errors="coerce").fillna(0.0)
    logger.info(
        f"Hydrogen store: level carried between windows for {len(names)} store(s), starting at "
        f"{start.sum() / 1e3:,.1f} GWh of {e_nom.sum() / 1e3:,.1f} GWh; month end {cfg['month_end']}"
    )
    return start.astype(float)


def apply_carried_store_levels(window_network, levels) -> None:
    """Start each carried store of a window network at ``levels`` (MWh; dict or Series, None = run start)."""
    if levels is None:
        return
    items = levels.items() if hasattr(levels, "items") else []
    for name, level in items:
        if name in window_network.stores.index:
            window_network.stores.loc[name, "e_initial"] = float(level)
            window_network.stores.loc[name, "e_cyclic"] = False


def final_store_levels(window_network, names, at=None) -> dict:
    """Each carried store's level at the window's last snapshot (MWh), for the next window.

    ``at`` (None = the last snapshot) reads the level at that snapshot instead:
    the last kept hour of a Stage 1 window solved with a look-ahead.
    """
    e = getattr(getattr(window_network, "stores_t", None), "e", None)
    if e is None or e.empty:
        return {}
    cols = [n for n in names if n in e.columns]
    row = e[cols].iloc[-1] if at is None else e[cols].loc[at]
    return {n: float(v) for n, v in row.items()}


def _store_refill_rate(network, store: str) -> tuple:
    """(MW the links feeding a store's bus can deliver to it, that bus)."""
    bus = network.stores.at[store, "bus"]
    links = network.links
    feeding = links.index[(links["bus1"] == bus) & (links["bus0"] != bus)] if len(links) else pd.Index([])
    make = float((
        pd.to_numeric(links.loc[feeding, "p_nom"], errors="coerce").fillna(0.0)
        * pd.to_numeric(links.loc[feeding, "efficiency"], errors="coerce").fillna(1.0)
    ).sum()) if len(feeding) else 0.0
    return make, bus


def build_month_end_store_callback(network, market_config: dict, start: pd.Series, final_snapshot,
                                   logger: logging.Logger):
    """``extra_functionality`` holding each carried store's final level at or above ``start``.

    Acts only in the optimisation whose snapshots contain ``final_snapshot``.
    The target is capped at the window's starting level plus what the feeding
    links (electrolysers) can make in that window net of the offtake at the
    store's bus, so it never makes the window infeasible. None when ``month_end`` is ``free`` or
    nothing is carried.
    """
    cfg = resolve_store_level_config(market_config)
    trajectory_cfg = _seasonal_trajectory_cfg(market_config)
    if (trajectory_cfg is not None and trajectory_cfg["end_condition"] == "soft_floor"
            and start is not None and len(start)):
        # Decision 33: no hard end; a penalised shortfall below the trajectory.
        return _build_trajectory_soft_floor_callback(network, trajectory_cfg, list(start.index), final_snapshot,
                                                     logger, cfg["end_of_window_value_gbp_per_mwh_h2"])
    if start is None or not len(start) or cfg["month_end"] == "free":
        if start is not None and len(start):
            logger.info("Hydrogen store month end: free (no terminal level condition)")
        return None
    names = list(start.index)

    def month_end_store(n, sns):
        if final_snapshot not in pd.Index(sns):
            return
        model = n.model
        if "Store-e" not in model.variables:
            return
        e = model.variables["Store-e"]
        dim = [d for d in e.dims if d != "snapshot"][0]
        present = [s for s in names if s in set(e.coords[dim].values)]
        if not present:
            return
        weights = n.snapshot_weightings.stores.reindex(pd.Index(sns)).fillna(1.0)
        hours = float(weights.sum())
        targets = []
        for s in present:
            make_mw, bus = _store_refill_rate(n, s)
            offtake = 0.0
            loads = n.loads.index[n.loads["bus"] == bus]
            if len(loads):
                varying = n.loads_t.p_set.reindex(index=pd.Index(sns), columns=loads) if len(n.loads_t.p_set.columns) else None
                static = pd.to_numeric(n.loads.loc[loads, "p_set"], errors="coerce").fillna(0.0)
                profile = pd.DataFrame(np.repeat(static.to_numpy()[None, :], len(sns), axis=0),
                                       index=pd.Index(sns), columns=loads)
                if varying is not None:
                    profile = varying.fillna(profile)
                offtake = float(profile.mul(weights, axis=0).sum().sum())
            loss = float(pd.to_numeric(n.stores.at[s, "standing_loss"], errors="coerce") or 0.0)
            retention = (1.0 - loss) ** hours
            # The window starts at its carried level; from there the feeding
            # links can add at most (make - offtake) over the window. Unlike a
            # battery, the store cannot refill from empty in a day, so capping
            # "from empty" (as storage_soc does) would let it end well below start.
            window_start = float(pd.to_numeric(n.stores.at[s, "e_initial"], errors="coerce") or 0.0)
            reachable = max(0.0, window_start * retention + 0.95 * (make_mw * hours - offtake) * retention)
            targets.append(min(float(start[s]), reachable))
        import xarray as xr

        target_da = xr.DataArray(np.asarray(targets), dims=[dim], coords={dim: pd.Index(present, name=dim)})
        model.add_constraints(
            e.sel({dim: present}).sel(snapshot=final_snapshot) >= target_da,
            name="Store-month_end_level",
        )
        capped = int(sum(t < float(start[s]) - 1e-6 for s, t in zip(present, targets)))
        logger.info(
            f"Hydrogen store month end: {len(present)} store(s) held at or above "
            f"{sum(targets) / 1e3:,.1f} GWh (run start {start.reindex(present).sum() / 1e3:,.1f} GWh) at {final_snapshot}"
            + (f"; {capped} capped at what the window can make" if capped else "")
        )

    return month_end_store


def build_end_of_window_store_value_callback(market_config: dict, start: pd.Series, logger: logging.Logger):
    """``extra_functionality`` crediting each carried store's level at the window's last snapshot.

    Adds ``-V x e(store, sns[-1])`` to the objective, V the configured
    ``end_of_window_value_gbp_per_mwh_h2`` (GBP/MWh_H2). Objective only: it adds
    no constraint, and ``e`` is bounded by ``e_nom``, so it cannot make a window
    infeasible or unbounded. In the window holding the month's last snapshot it
    sits beside the month-end floor (``build_month_end_store_callback``): the
    floor still binds whenever V alone would leave the store below it, and above
    it V decides, as in every other window. None when V is null or nothing is
    carried (a per-window cyclic store has a fixed end, so a credit would be a
    constant).
    """
    cfg = resolve_store_level_config(market_config)
    trajectory_cfg = _seasonal_trajectory_cfg(market_config)
    if trajectory_cfg is not None and trajectory_cfg["end_value"] == "dual" and start is not None and len(start):
        return _build_trajectory_end_value_callback(trajectory_cfg, list(start.index), logger)
    value = cfg["end_of_window_value_gbp_per_mwh_h2"]
    if value is None or value <= 0.0:
        return None
    if start is None or not len(start):
        logger.info("Hydrogen store end-of-window value: configured but no store level is carried; not applied")
        return None
    names = list(start.index)
    logger.info(
        f"Hydrogen store end-of-window value: {value:.2f} GBP/MWh_H2 on {len(names)} store(s) at each "
        f"window's last snapshot (objective only, stripped from the reported objective)"
    )

    def end_of_window_value(n, sns):
        model = n.model
        if "Store-e" not in model.variables:
            return
        e = model.variables["Store-e"]
        dim = [d for d in e.dims if d != "snapshot"][0]
        present = [s for s in names if s in set(e.coords[dim].values)]
        if not present:
            return
        final = e.sel({dim: present}).sel(snapshot=pd.Index(sns)[-1])
        model.objective = model.objective + (final * -float(value)).sum()

    end_of_window_value.value = float(value)
    end_of_window_value.stores = tuple(names)
    return end_of_window_value


def store_end_of_window_credit(window_network, callback) -> tuple:
    """``(MWh credited, GBP credited)``: what the end-of-window value took off a solved window's objective.

    ``(0.0, 0.0)`` without the callback. The caller adds the GBP back to the
    reported objective (the credit is not a cost anyone pays).
    """
    if callback is None:
        return 0.0, 0.0
    e = getattr(getattr(window_network, "stores_t", None), "e", None)
    if e is None or e.empty:
        return 0.0, 0.0
    cols = [s for s in callback.stores if s in e.columns]
    mwh = float(e[cols].iloc[-1].sum()) if cols else 0.0
    # A time-varying (seasonal trajectory) value records the one it applied last.
    value = getattr(callback, "last_value", None)
    return mwh, mwh * (callback.value if value is None else value)


# ── Seasonal trajectory (decision 33, task Y8a) ─────────────────────────────
#
# The annual pass (scripts/market/seasonal_storage.py, rule
# solve_seasonal_storage_pass) gives the hydrogen store's level for every hour of
# the year and its value, the store-balance dual, and the same per storage-unit
# carrier. A month that sets ``market.hydrogen_store.seasonal_trajectory``
# starts the store there (hard, ``prepare_carried_stores``), values each Stage 1
# window's end level at the dual for that hour (``end_value: dual``) and never
# pins the end: ``end_condition: value`` leaves it to the value alone,
# ``soft_floor`` adds a penalised shortfall below the trajectory's level at the
# period end (``build_month_end_store_callback``). With ``storage_units: true``
# storage units start on the pass too (``apply_initial_storage_soc``) and the
# period's last window values their end state at the pass's storage dual
# (``build_month_end_soc_callback``). Every objective-only term is stripped from
# the reported objective (``objective_only_adjustment``).


def _seasonal_storage_units_cfg(market_config: dict):
    """The enabled trajectory block when it also carries storage units, else None."""
    cfg = _seasonal_trajectory_cfg(market_config)
    return cfg if cfg is not None and cfg["storage_units"] else None


def _voll_equivalent_gbp_per_mwh_h2(network, store_bus: str) -> tuple:
    """``(VOLL x best turbine efficiency, VOLL, efficiency)`` for the store's bus.

    VOLL is the highest load-shedding cost on the network (6000 when there is none);
    the efficiency is the best of the links that take hydrogen off ``store_bus`` to
    another bus. A shortfall penalty at or above the product could make a window shed
    load to keep hydrogen in the store.
    """
    g = network.generators
    shed = g.index[g["carrier"].astype(str).isin(["load_shedding", "load shedding"])] if len(g) else pd.Index([])
    voll = float(pd.to_numeric(g.loc[shed, "marginal_cost"], errors="coerce").max()) if len(shed) else 6000.0
    if not np.isfinite(voll) or voll <= 0:
        voll = 6000.0
    links = network.links
    leaving = links.index[(links["bus0"] == store_bus) & (links["bus1"] != store_bus)] if len(links) else pd.Index([])
    eff = float(pd.to_numeric(links.loc[leaving, "efficiency"], errors="coerce").max()) if len(leaving) else 1.0
    if not np.isfinite(eff) or eff <= 0:
        eff = 1.0
    return voll * eff, voll, eff


def _build_trajectory_soft_floor_callback(network, trajectory_cfg: dict, names, final_snapshot,
                                          logger: logging.Logger, flat_value: Optional[float] = None):
    """``end_condition: soft_floor``: a window pays P per MWh_H2 it ends below the trajectory.

    Adds ``shortfall >= 0`` per store with ``e(T) + shortfall >= trajectory level(T) x share``
    and ``P x shortfall`` to the objective, T the window's last snapshot. P is a premium
    on top of the end-of-window value V, so the store's value is kinked at the path: V
    above it, ``V + P`` below it. With ``soft_floor_windows: every_window`` (default)
    every Stage 1 window carries it; with ``period_end`` only the window holding
    ``final_snapshot`` does. No hard condition: the window is always feasible.
    ``V_max + P`` must be below the VOLL-equivalent (``_voll_equivalent_gbp_per_mwh_h2``;
    ``V_max`` the trajectory's highest value), checked here, so no window sheds load to
    keep hydrogen. Objective only: ``objective_only_gbp`` returns what to add back to
    the window's objective (minus the penalty paid).

    ``surplus_discount_gbp_per_mwh_h2`` (D, null by default) also prices the level above
    the path: ``e(T) + shortfall - surplus = level``, ``surplus >= 0``, and the window's
    hydrogen terms become ``(V + P) x shortfall - (V - D) x surplus``, the end-of-window
    credit ``-V x e(T)`` restated about the path (the same objective less the constant
    ``V x level``, with D taken off the value above the path). Two reasons (9 Nov 2026):
    the annual dual V sits within the solver's optimality tolerance of the reformer's
    hydrogen cost, so without D a window is indifferent between reforming gas into the
    store and not, and central dispatch's solver profile reformed flat out; and the
    constant (about GBP 0.5 bn a window) loosened the barrier's relative gap. Objective
    only: ``objective_only_gbp`` also returns ``-D x surplus - V x level``.
    """
    from scripts.market.seasonal_storage import (
        load_seasonal_trajectory, store_shares, trajectory_end_level, trajectory_value,
    )

    trajectory = load_seasonal_trajectory(trajectory_cfg["file"])
    penalty = float(trajectory_cfg["shortfall_penalty_gbp_per_mwh_h2"])
    discount = trajectory_cfg.get("surplus_discount_gbp_per_mwh_h2")
    discount = None if discount is None else float(discount)
    dual_value = trajectory_cfg["end_value"] == "dual"
    every_window = trajectory_cfg["soft_floor_windows"] == "every_window"
    if trajectory_cfg["end_value"] == "dual":
        v_max = float(pd.to_numeric(trajectory["value_gbp_per_mwh_h2"], errors="coerce").clip(lower=0.0).max())
    else:
        v_max = float(flat_value or 0.0)
    v_max = v_max if np.isfinite(v_max) else 0.0
    for store in names:
        bus = network.stores.at[store, "bus"]
        limit, voll, eff = _voll_equivalent_gbp_per_mwh_h2(network, bus)
        if not v_max + penalty < limit:
            raise ValueError(
                f"seasonal_trajectory: end value {v_max:g} + shortfall_penalty_gbp_per_mwh_h2 {penalty:g} is not "
                f"below the VOLL-equivalent {limit:,.0f} GBP/MWh_H2 (VOLL {voll:,.0f} x turbine efficiency "
                f"{eff:.3f}) for {store}: the window could shed load to keep hydrogen")
    shares = store_shares(network.stores, names)
    logger.info(
        f"Hydrogen store soft floor (seasonal trajectory): {'every window end' if every_window else 'the period end'}"
        f" below the trajectory's level priced at {penalty:.2f} GBP/MWh_H2 over the end value (objective only); period end "
        f"{trajectory_end_level(trajectory, final_snapshot) / 1e3:,.1f} GWh at {final_snapshot}")
    if discount is not None:
        logger.info(
            f"Hydrogen store above the trajectory: valued {discount:.2f} GBP/MWh_H2 under the end value "
            f"(surplus_discount_gbp_per_mwh_h2; objective only), the end-of-window credit restated about the path")

    def soft_floor(n, sns):
        soft_floor.active = False
        soft_floor.path_credit_gbp = 0.0
        last = pd.Index(sns)[-1]
        if not every_window and final_snapshot not in pd.Index(sns):
            return
        model = n.model
        if "Store-e" not in model.variables:
            return
        e = model.variables["Store-e"]
        dim = [d for d in e.dims if d != "snapshot"][0]
        present = [s for s in names if s in set(e.coords[dim].values)]
        if not present:
            return
        import xarray as xr

        level = trajectory_end_level(trajectory, last)
        coord = pd.Index(present, name=dim)
        target_da = xr.DataArray(np.asarray([float(level * shares[s]) for s in present]), dims=[dim],
                                 coords={dim: coord})
        shortfall = model.add_variables(lower=0.0, coords=[coord], name="Store-trajectory_shortfall")
        final = e.sel({dim: present}).sel(snapshot=last)
        if discount is None:
            model.add_constraints(final + shortfall >= target_da, name="Store-trajectory_soft_floor")
            model.objective = model.objective + (shortfall * penalty).sum()
            soft_floor.active = True
            return
        surplus = model.add_variables(lower=0.0, coords=[coord], name="Store-trajectory_surplus")
        model.add_constraints(final + shortfall - surplus == target_da, name="Store-trajectory_soft_floor")
        # The end-of-window value the store callback credits on e(T) in this window
        # (build_end_of_window_store_value_callback): the annual dual, else the flat value.
        if dual_value:
            value = max(0.0, trajectory_value(trajectory, last))
        else:
            value = float(flat_value) if flat_value is not None and float(flat_value) > 0.0 else 0.0
        # -V e(T) = -V level + V shortfall - V surplus: cancel the credit on e(T) and
        # carry it on the two deviations, so the objective holds no V x level constant.
        model.objective = (
            model.objective
            + (shortfall * (penalty + value)).sum()
            + (surplus * (discount - value)).sum()
            + (final * value).sum()
        )
        soft_floor.path_credit_gbp = value * float(target_da.sum())
        soft_floor.active = True

    def shortfall_mwh(window_network) -> float:
        """MWh_H2 the solved window ended below the trajectory (0 when it carried no floor)."""
        model = getattr(window_network, "model", None)
        if not getattr(soft_floor, "active", False) or model is None                 or "Store-trajectory_shortfall" not in model.variables:
            return 0.0
        return float(np.nansum(model.variables["Store-trajectory_shortfall"].solution.values))

    def surplus_mwh(window_network) -> float:
        """MWh_H2 the solved window ended above the trajectory (0 without a surplus discount)."""
        model = getattr(window_network, "model", None)
        if not getattr(soft_floor, "active", False) or model is None \
                or "Store-trajectory_surplus" not in model.variables:
            return 0.0
        return float(np.nansum(model.variables["Store-trajectory_surplus"].solution.values))

    def objective_only_gbp(window_network) -> float:
        """GBP to add to the window objective: minus the penalty and discount paid and the path's credit."""
        if not getattr(soft_floor, "active", False):
            return 0.0
        out = -penalty * shortfall_mwh(window_network)
        if discount is not None:
            out -= discount * surplus_mwh(window_network) + float(getattr(soft_floor, "path_credit_gbp", 0.0))
        return out

    soft_floor.active = False
    soft_floor.penalty = penalty
    soft_floor.discount = discount
    soft_floor.path_credit_gbp = 0.0
    soft_floor.objective_only_gbp = objective_only_gbp
    soft_floor.shortfall_mwh = shortfall_mwh
    soft_floor.surplus_mwh = surplus_mwh
    return soft_floor


def _build_trajectory_end_value_callback(trajectory_cfg: dict, names, logger: logging.Logger):
    """End-of-window value from the annual pass: ``-V(T) x e(store, T)``, V(T) its dual at the window's last hour.

    Objective only, as the flat value. V is floored at 0 (as the flat value must be).
    ``callback.last_value`` is the value the last call applied, for
    ``store_end_of_window_credit``.
    """
    from scripts.market.seasonal_storage import load_seasonal_trajectory, trajectory_value

    trajectory = load_seasonal_trajectory(trajectory_cfg["file"])
    logger.info(
        f"Hydrogen store end-of-window value: the seasonal trajectory's hourly store value "
        f"({trajectory_cfg['file']}) on {len(names)} store(s) at each window's last snapshot "
        f"(objective only, stripped from the reported objective)"
    )

    def end_of_window_value(n, sns):
        model = n.model
        if "Store-e" not in model.variables:
            return
        e = model.variables["Store-e"]
        dim = [d for d in e.dims if d != "snapshot"][0]
        present = [s for s in names if s in set(e.coords[dim].values)]
        if not present:
            return
        last = pd.Index(sns)[-1]
        value = max(0.0, trajectory_value(trajectory, last))
        end_of_window_value.last_value = value
        end_of_window_value.applied.append((last, value))
        final = e.sel({dim: present}).sel(snapshot=last)
        model.objective = model.objective + (final * -float(value)).sum()

    end_of_window_value.value = float("nan")
    end_of_window_value.last_value = None
    end_of_window_value.applied = []
    end_of_window_value.time_varying = True
    end_of_window_value.stores = tuple(names)
    return end_of_window_value


def apply_trajectory_storage_soc(network, market_config: dict, units, logger: logging.Logger) -> None:
    """Start ``units`` at the annual pass's state for their carrier, pro rata to energy capacity.

    Capped at each unit's energy capacity. Carriers the trajectory does not carry
    keep their state.
    """
    from scripts.market.seasonal_storage import load_seasonal_trajectory, trajectory_storage_start

    cfg = _seasonal_storage_units_cfg(market_config)
    trajectory = load_seasonal_trajectory(cfg["file"])
    level = trajectory_storage_start(trajectory, network.snapshots[0])
    su = network.storage_units.loc[units]
    energy = (pd.to_numeric(su["p_nom"], errors="coerce").fillna(0.0)
              * pd.to_numeric(su["max_hours"], errors="coerce").fillna(0.0))
    set_gwh, missing = {}, []
    for carrier, members in su.groupby("carrier").groups.items():
        if carrier not in level.index:
            missing.append(str(carrier))
            continue
        e = energy.loc[members]
        total = float(e.sum())
        share = e / total if total > 0 else pd.Series(1.0 / len(e), index=e.index)
        soc = np.minimum(float(level[carrier]) * share, e)
        network.storage_units.loc[members, "state_of_charge_initial"] = soc.to_numpy()
        set_gwh[str(carrier)] = round(float(soc.sum()) / 1e3, 1)
    logger.info(
        f"Storage starts on the seasonal trajectory ({cfg['file']}) at {network.snapshots[0]}: "
        f"{sum(set_gwh.values()):,.1f} GWh, by carrier {set_gwh}"
        + (f"; not in the trajectory, kept: {missing}" if missing else "")
    )


def _build_trajectory_storage_end_value_callback(network, market_config: dict, units, final_snapshot,
                                                 logger: logging.Logger):
    """The period's last window values each unit's end state at the pass's dual for its carrier.

    ``-V_c(T) x soc(unit, T)`` in the window holding ``final_snapshot`` only (earlier
    windows stay myopic); V floored at 0. Objective only: ``objective_only_gbp``
    returns the credit to add back.
    """
    from scripts.market.seasonal_storage import load_seasonal_trajectory, trajectory_storage_value

    cfg = _seasonal_storage_units_cfg(market_config)
    trajectory = load_seasonal_trajectory(cfg["file"])
    value = trajectory_storage_value(trajectory, final_snapshot).clip(lower=0.0)
    carrier = network.storage_units.loc[units, "carrier"]
    unit_value = carrier.map(value)
    valued = list(unit_value.index[unit_value.notna()])
    logger.info(
        "Storage period end: free, the last window values each unit's end state at the seasonal "
        f"trajectory's storage value for {final_snapshot} (objective only): "
        + ", ".join(f"{c} {v:.2f}" for c, v in value.items()) + " GBP/MWh")

    def storage_end_value(n, sns):
        if final_snapshot not in pd.Index(sns):
            return
        model = n.model
        if "StorageUnit-state_of_charge" not in model.variables:
            return
        soc = model.variables["StorageUnit-state_of_charge"]
        dim = [d for d in soc.dims if d != "snapshot"][0]
        present = [u for u in valued if u in set(soc.coords[dim].values)]
        if not present:
            return
        import xarray as xr

        coord = pd.Index(present, name=dim)
        v = xr.DataArray(unit_value.loc[present].to_numpy(dtype=float), dims=[dim], coords={dim: coord})
        final = soc.sel({dim: present}).sel(snapshot=final_snapshot)
        model.objective = model.objective + (final * -v).sum()
        storage_end_value.applied = present

    def objective_only_gbp(window_network) -> float:
        """GBP to add to the window objective: the credit taken (0 outside the last window)."""
        present = getattr(storage_end_value, "applied", None)
        soc = getattr(window_network.storage_units_t, "state_of_charge", pd.DataFrame())
        if not present or soc is None or soc.empty or final_snapshot not in soc.index:
            return 0.0
        end = soc.loc[final_snapshot, [u for u in present if u in soc.columns]]
        return float((end * unit_value.reindex(end.index)).sum())

    storage_end_value.applied = None
    storage_end_value.value = value
    storage_end_value.objective_only_gbp = objective_only_gbp
    return storage_end_value


def objective_only_adjustment(window_network, *callbacks) -> float:
    """GBP to add to a solved window's objective for the callbacks' objective-only terms.

    Callbacks without ``objective_only_gbp`` (every one outside the seasonal
    trajectory) contribute 0.0.
    """
    total = 0.0
    for cb in callbacks:
        fn = getattr(cb, "objective_only_gbp", None)
        if fn is not None:
            total += float(fn(window_network))
    return total


def trajectory_period_end_report(window_network, store_callback, storage_callback) -> dict:
    """One solved window's seasonal-trajectory terms: soft-floor shortfall (MWh_H2) and penalty, storage end credit (GBP)."""
    out = {}
    if store_callback is not None and hasattr(store_callback, "shortfall_mwh"):
        out["h2_shortfall_mwh"] = store_callback.shortfall_mwh(window_network)
        out["h2_shortfall_penalty_gbp"] = store_callback.penalty * out["h2_shortfall_mwh"]
        if getattr(store_callback, "discount", None) is not None:
            out["h2_surplus_mwh"] = store_callback.surplus_mwh(window_network)
            out["h2_surplus_discount_gbp"] = store_callback.discount * out["h2_surplus_mwh"]
    if storage_callback is not None and hasattr(storage_callback, "objective_only_gbp"):
        out["storage_end_credit_gbp"] = storage_callback.objective_only_gbp(window_network)
    return out


# ── Minimum output as an energy budget (user decision, 27 Sep 2026) ─────────
#
# ``thermal.availability_factors.min_output`` becomes an hourly p_min_pu at the
# thermal build. For CP30 biomass (0.35) that forced 1.24 GW through every
# Stage 1 surplus hour (GB price 0-0.5, renewables curtailed). The floor exists
# for energy, not timing: Drax, Lynemouth and MGT Teesside (2.7 GW) cost 128.6
# GBP/MWh against 7.9 for the small ROC units (0.83 GW), so without it biomass
# falls from CP30's 12.2 TWh to ~4 TWh. For carriers listed in
# ``min_output_energy_window_hours`` ({carrier: hours}) the solve replaces the
# hourly floor with an energy budget: in every block of that many hours (cut
# from the start of each optimisation, a shorter last block keeping its own
# hours) each unit delivers at least the energy the floor would have forced,
# ``sum_t w_t p[g,t] >= sum_t w_t p_min_pu[g,t] p_nom[g]`` (floor as clipped to
# availability), with p_min_pu zeroed so timing is free up to p_max_pu. Stage 1
# (rolling or single, copperplate or central) carries the budget; the BM by
# default only drops the hourly floor (its Stage 1 anchor holds the energy, and it
# can bid the units down at their normal bid), or keeps the budget too under
# market.balancing.min_output_energy_budget: hold (below). Solve time only: the built networks keep the
# floor. Committable units (a Stage 1 UC carrier) keep their commitment floor.
#
#     floor = prepare_min_output_energy_budget(network, scenario_config, logger)  # once
#     callback = build_min_output_energy_callback(floor)
#     each window: solve(..., extra_functionality includes callback)
#                  budget, delivered = min_output_energy_delivered(window_net, callback)


def resolve_min_output_energy_windows(scenario_config: dict) -> dict:
    """``thermal.availability_factors.min_output_energy_window_hours`` as {carrier: hours} (default {})."""
    block = ((((scenario_config or {}).get("thermal", {}) or {}).get("availability_factors", {})) or {})
    raw = block.get("min_output_energy_window_hours", {}) or {}
    out = {}
    for carrier, hours in raw.items():
        hours = float(hours)
        if not hours > 0.0:
            raise ValueError(
                f"thermal.availability_factors.min_output_energy_window_hours.{carrier} must be > 0, got {hours}"
            )
        out[str(carrier)] = hours
    return out


def _min_output_energy_units(network, windows: dict) -> pd.Index:
    """Generators of the listed carriers with a floor to convert (committable units excluded)."""
    gens = getattr(network, "generators", None)
    if not windows or gens is None or not len(gens):
        return pd.Index([])
    mask = gens["carrier"].isin(set(windows))
    if "committable" in gens.columns:
        mask &= ~gens["committable"].fillna(False).astype(bool)
    return gens.index[mask]


def _zero_min_output(network, names) -> None:
    """Set static and time-varying p_min_pu to 0 for ``names``."""
    network.generators.loc[names, "p_min_pu"] = 0.0
    tv = network.generators_t.p_min_pu
    cols = [g for g in names if g in tv.columns]
    if cols:
        network.generators_t["p_min_pu"] = tv.drop(columns=cols)


def prepare_min_output_energy_budget(network, scenario_config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Hourly floor (MW, snapshots x units) of the budget carriers; their p_min_pu is then zeroed.

    Call once on the full solve-period network after the floor has been clipped
    to availability (``enforce_feasible_generator_operating_bounds``) and after
    any Stage 1 unit commitment. Empty when nothing is configured.
    """
    windows = resolve_min_output_energy_windows(scenario_config)
    names = _min_output_energy_units(network, windows)
    if not len(names):
        if windows:
            logger.info(f"Minimum-output energy budget: no eligible units for {sorted(windows)}")
        return pd.DataFrame(index=pd.Index(network.snapshots))
    p_min = network.get_switchable_as_dense("Generator", "p_min_pu", network.snapshots)[names]
    p_max = network.get_switchable_as_dense("Generator", "p_max_pu", network.snapshots)[names]
    p_nom = pd.to_numeric(network.generators.loc[names, "p_nom"], errors="coerce").fillna(0.0)
    floor = p_min.clip(upper=p_max).clip(lower=0.0).mul(p_nom, axis=1)
    floor = floor.loc[:, floor.sum() > 0.0]
    _zero_min_output(network, names)
    if floor.shape[1]:
        weights = network.snapshot_weightings.generators.reindex(floor.index).fillna(1.0)
        carriers = network.generators.loc[floor.columns, "carrier"]
        for carrier, hours in windows.items():
            cols = carriers.index[carriers == carrier]
            if len(cols):
                logger.info(
                    f"Minimum-output energy budget: {carrier}, {len(cols)} units "
                    f"({p_nom.reindex(cols).sum():,.0f} MW): hourly floor replaced by an energy budget "
                    f"per {hours:g} h, {floor[cols].mul(weights, axis=0).sum().sum() / 1e3:,.1f} GWh over the period"
                )
    floor.attrs["hours"] = {g: windows[network.generators.at[g, "carrier"]] for g in floor.columns}
    return floor


# ``market.balancing.min_output_energy_budget`` (OW2, user decision 3 Oct 2026):
# ``drop`` (default, the behaviour above) or ``hold``. Under ``hold`` the BM keeps
# each budget unit's Stage 1 energy budget, the floor's energy per block, with its
# timing free: prepare_min_output_energy_budget on the BM network and
# build_min_output_energy_callback in every BM optimisation. Why: at cost (the
# central-dispatch BM bids biomass at -128.6 against gas offers ~107) the BM
# re-optimised Drax, Lynemouth and MGT's 0.35 floor away, 8.8 TWh of biomass for
# gas in the perfect-foresight year (scratchpad ow2/ow2_report.md); the anchor
# alone holds the energy only while the bid/offer spread does. Blocks are cut from
# the start of each optimisation, as in Stage 1, so a BM window as long as the
# block (24 h) holds exactly the Stage 1 blocks; a shorter window holds its own
# share of the floor. Not in config/defaults/ (adding a market key there changes every
# scenario's solve params); the code default is ``drop``.
MIN_OUTPUT_ENERGY_BUDGET_MODES = ("drop", "hold")


def resolve_min_output_energy_budget_mode(balancing_config: Optional[dict]) -> str:
    """``market.balancing.min_output_energy_budget`` validated (default ``drop``)."""
    mode = str((balancing_config or {}).get("min_output_energy_budget", "drop")).lower()
    if mode not in MIN_OUTPUT_ENERGY_BUDGET_MODES:
        raise ValueError(
            f"market.balancing.min_output_energy_budget must be one of {MIN_OUTPUT_ENERGY_BUDGET_MODES}, "
            f"got {mode!r}"
        )
    return mode


def prepare_bm_min_output_energy_budget(network, scenario_config: dict, balancing_config: Optional[dict],
                                        logger: logging.Logger) -> Optional[pd.DataFrame]:
    """BM side of the energy budget: ``drop`` zeroes the hourly floor and returns None;
    ``hold`` also returns the floor (MW, as Stage 1 builds it) for build_min_output_energy_callback.

    Call where drop_min_output_floor_for_energy_budget was called: after the
    floor is clipped to availability (enforce_feasible_generator_operating_bounds).
    """
    if resolve_min_output_energy_budget_mode(balancing_config) == "drop":
        drop_min_output_floor_for_energy_budget(network, scenario_config, logger)
        return None
    floor = prepare_min_output_energy_budget(network, scenario_config, logger)
    if floor is None or not floor.shape[1]:
        return None
    window = (balancing_config or {}).get("window_hours")
    hours = pd.Series(floor.attrs.get("hours", {}), dtype=float)
    if str((balancing_config or {}).get("mode", "full_period")) == "rolling" and window is not None             and len(hours) and float(window) < float(hours.max()):
        logger.warning(
            f"Minimum-output energy budget held in the BM, but BM windows ({float(window):g} h) are shorter than "
            f"the budget blocks ({float(hours.max()):g} h): each window holds its own share of the floor, "
            f"so timing is only free inside a window"
        )
    logger.info(
        f"Minimum-output energy budget: held in the BM for {floor.shape[1]} units "
        f"(market.balancing.min_output_energy_budget: hold)"
    )
    return floor


def drop_min_output_floor_for_energy_budget(network, scenario_config: dict, logger: logging.Logger) -> pd.Index:
    """BM side: zero the hourly floor of the budget carriers (the Stage 1 anchor carries their energy)."""
    names = _min_output_energy_units(network, resolve_min_output_energy_windows(scenario_config))
    if not len(names):
        return names
    _zero_min_output(network, names)
    logger.info(
        f"Minimum-output energy budget: hourly floor dropped for {len(names)} units "
        f"({network.generators.loc[names, 'p_nom'].sum():,.0f} MW) in the BM; Stage 1 carries their energy"
    )
    return names


def _energy_blocks(sns, hours: float, weights: pd.Series) -> list:
    """Consecutive blocks of ``sns`` spanning ``hours`` each (by snapshot weight); the last may be shorter."""
    blocks, current, span = [], [], 0.0
    for t in pd.Index(sns):
        current.append(t)
        span += float(weights.get(t, 1.0))
        if span >= hours - 1e-9:
            blocks.append(current)
            current, span = [], 0.0
    if current:
        blocks.append(current)
    return blocks


def _floor_on_snapshots(floor: pd.DataFrame, sns, weights: pd.Series) -> pd.DataFrame:
    """The hourly floor on an optimisation's snapshots: the row itself, or a coarse block's mean.

    With uniform weights (every solve except a coarse look-ahead) this is
    ``floor.reindex(sns)`` exactly. A snapshot weighted more than the finest one
    stands for the floor rows from it up to the next snapshot (the last one: as
    many rows as its weight in units of the finest weight), and takes their mean,
    so mean x weight is the block's floor energy.
    """
    sns = pd.Index(sns)
    if weights.nunique() <= 1:
        return floor.reindex(index=sns)
    pos = floor.index.get_indexer(sns)
    unit = float(weights.min())
    rows = []
    for i, t in enumerate(sns):
        if pos[i] < 0:
            rows.append(pd.Series(np.nan, index=floor.columns))
            continue
        end = pos[i + 1] if i + 1 < len(sns) and pos[i + 1] > pos[i] else pos[i] + max(1, int(round(float(weights.iloc[i]) / unit)))
        rows.append(floor.iloc[pos[i]:end].mean())
    return pd.DataFrame(rows, index=sns)


def build_min_output_energy_callback(floor: pd.DataFrame):
    """``extra_functionality`` holding each budget unit's energy per block at or above its floor's energy.

    None when ``floor`` is empty. The budget of every block is at most the
    unit's available energy (the floor is clipped to p_max_pu hour by hour), so
    the constraint cannot make a window infeasible. The budgets of the last
    optimisation are kept on ``callback.last`` for ``min_output_energy_delivered``.
    """
    if floor is None or not floor.shape[1]:
        return None
    hours_by_unit = pd.Series(floor.attrs.get("hours", {}), dtype=float).reindex(floor.columns).fillna(24.0)

    def min_output_energy(n, sns):
        callback.last = None
        model = n.model
        if "Generator-p" not in model.variables:
            return
        import xarray as xr

        p = model.variables["Generator-p"]
        dim = [d for d in p.dims if d != "snapshot"][0]
        sns = pd.Index(sns)
        weights = n.snapshot_weightings.generators.reindex(sns).fillna(1.0)
        present = [g for g in floor.columns if g in set(p.coords[dim].values)]
        window_floor = _floor_on_snapshots(floor, sns, weights).reindex(columns=present).fillna(0.0)
        budgets = []
        for hours, units in hours_by_unit.reindex(present).groupby(hours_by_unit.reindex(present)):
            units = list(units.index)
            for k, block in enumerate(_energy_blocks(sns, float(hours), weights)):
                w = weights.reindex(block)
                budget = window_floor.loc[block, units].mul(w, axis=0).sum()
                budget = budget[budget > 1e-6]
                if not len(budget):
                    continue
                idx = pd.Index(list(budget.index), name=dim)
                w_da = xr.DataArray(w.to_numpy(dtype=float), dims=["snapshot"], coords={"snapshot": pd.Index(block, name="snapshot")})
                energy = (p.sel({dim: list(idx), "snapshot": block}) * w_da).sum("snapshot")
                model.add_constraints(
                    energy >= xr.DataArray(budget.to_numpy(dtype=float), dims=[dim], coords={dim: idx}),
                    name=f"Generator-min_output_energy_{int(hours)}h_{k}",
                )
                budgets.append(pd.DataFrame({"block": k, "start": block[0], "hours": float(w.sum()),
                                             "budget_mwh": budget}).rename_axis("generator").reset_index())
        callback.last = pd.concat(budgets, ignore_index=True) if budgets else None

    callback = min_output_energy
    callback.last = None
    callback.units = tuple(floor.columns)
    return callback


def min_output_energy_delivered(window_network, callback, snapshots=None) -> tuple:
    """``(budget MWh, delivered MWh)`` of the last optimisation's budget units ((0.0, 0.0) without one).

    ``snapshots`` (None = the whole optimisation) counts only the blocks that
    start in them and the energy delivered in them: the kept hours of a Stage 1
    window solved with a look-ahead.
    """
    if callback is None or getattr(callback, "last", None) is None:
        return 0.0, 0.0
    budgets = callback.last
    p = window_network.generators_t.p
    if snapshots is not None:
        kept = pd.Index(snapshots)
        budgets = budgets[pd.Index(budgets["start"]).isin(kept)]
        p = p.loc[p.index.isin(kept)]
    weights = window_network.snapshot_weightings.generators.reindex(p.index).fillna(1.0)
    units = [g for g in budgets["generator"].unique() if g in p.columns]
    delivered = float(p[units].mul(weights, axis=0).sum().sum()) if units else 0.0
    return float(budgets["budget_mwh"].sum()), delivered


# ── Solver fallback ladder ───────────────────────────────────────────────
#
# The configured Gurobi profile is the fast one (auto scaling, default numeric
# focus, plain barrier without crossover). The conservative settings that used
# to be the default (NumericFocus 3, BarHomogeneous 1, ScaleFlag 2,
# DualReductions 0) were a workaround for the 1e6 MVA copperplate relaxation
# and roughly doubled every solve; they are kept as the first fallback rung,
# and dual simplex (a vertex solution, robust to barrier misclassification) is
# the last. A window that needs a rung is logged as a WARNING so the timing
# gate can see it.

BARRIER_ONLY_SOLVER_KEYS = ("BarHomogeneous", "BarConvTol", "BarIterLimit")
CONSERVATIVE_BARRIER_OVERRIDES = {
    "NumericFocus": 3,
    "BarHomogeneous": 1,
    "ScaleFlag": 2,
    "DualReductions": 0,
}
ACCEPTED_TERMINATION_CONDITIONS = ("optimal",)


class SolveOutcome(NamedTuple):
    """Result of :func:`solve_with_fallback`."""

    status: str
    condition: str
    rung: int            # 1-based index of the rung that produced the solution
    rung_name: str
    solver_options: dict  # the options that rung used
    elapsed: float        # seconds spent inside optimize() over every rung tried
    attempts: tuple       # ((rung_name, status, condition, seconds), ...)
    bound_violation: float = float("nan")  # MW, the accepted solution's largest (max_bound_violation)


# ── Post-solve bound check (review 27 Sep 2026, item C11) ────────────────
#
# A barrier stopped without crossover, and an "optimal" whose uncrush Gurobi
# downgrades to "suboptimal" (seen on the central-dispatch LP: "big uncrush
# violation 4.2"), can return a point outside the variables' own bounds. The
# status alone does not say by how much, so every accepted rung is measured and
# the figure logged per window; above the threshold the rung counts as failed
# and the ladder moves on (dual simplex returns a vertex inside the bounds).

#: Largest bound violation (MW; MWh for storage energy) an accepted solution may carry.
#: 1 MW since 13 Oct 2026 (items F3/F14; was 50). In the 6,845 checks of the 400
#: latest market logs every violation of 1 MW or more (96) was the free loop on
#: the twin Sofia HVDC poles, which the build now merges
#: (network_build/parallel_links.py); the largest elsewhere was 0.4 MW and the
#: 90th percentile 2e-7 MW.
MAX_BOUND_VIOLATION_MW = 1.0


def max_bound_violation(network, snapshots=None) -> Tuple[float, str]:
    """Largest amount by which a solved network's dispatch breaks its own bounds.

    Checks generator output against ``p_min_pu``/``p_max_pu`` x ``p_nom`` (the
    lower bound only for non-committable units), link flow against its
    ``p_min_pu``/``p_max_pu`` range, storage-unit dispatch, charging and state of
    charge against their ranges, store energy against ``e_min_pu``/``e_max_pu``,
    and line and transformer flow against ``s_max_pu`` x ``s_nom``. Extendable
    assets use their optimised capacity. Returns ``(violation, "Component name")``
    with violation >= 0, or ``(nan, "")`` for an object that is not a solved
    PyPSA network (the ladder's test stubs).
    """
    if not hasattr(network, "get_switchable_as_dense") or not hasattr(network, "generators_t"):
        return float("nan"), ""
    sns = network.snapshots if snapshots is None else pd.Index(snapshots)
    worst = [0.0, ""]

    def capacity(df, nom):
        opt = f"{nom}_opt"
        cap = pd.to_numeric(df[nom], errors="coerce").fillna(0.0)
        if f"{nom}_extendable" in df.columns and opt in df.columns:
            ext = df[f"{nom}_extendable"].fillna(False).astype(bool)
            cap = cap.where(~ext, pd.to_numeric(df[opt], errors="coerce").fillna(0.0))
        return cap

    def record(excess: pd.DataFrame, component: str):
        if excess is None or excess.empty:
            return
        values = excess.to_numpy(dtype=float, na_value=0.0)
        if not values.size:
            return
        flat = int(np.nanargmax(values))
        value = float(values.flat[flat])
        if value > worst[0]:
            worst[0] = value
            worst[1] = f"{component} {excess.columns[flat % values.shape[1]]}"

    def dense(component, attr, names):
        return network.get_switchable_as_dense(component, attr, sns)[names]

    def solved(table, attr, names):
        frame = getattr(table, attr, pd.DataFrame())
        return frame.reindex(index=sns, columns=names).astype(float).fillna(0.0)

    gens = network.generators
    if len(gens):
        names = gens.index
        p = solved(network.generators_t, "p", names)
        cap = capacity(gens, "p_nom")
        upper = dense("Generator", "p_max_pu", names).mul(cap, axis=1)
        lower = dense("Generator", "p_min_pu", names).mul(cap, axis=1)
        if "committable" in gens.columns:
            lower.loc[:, gens["committable"].fillna(False).astype(bool).to_numpy()] = -np.inf
        record(np.maximum(p - upper, lower - p), "Generator")

    links = network.links
    if len(links):
        names = links.index
        p0 = solved(network.links_t, "p0", names)
        cap = capacity(links, "p_nom")
        upper = dense("Link", "p_max_pu", names).mul(cap, axis=1)
        lower = dense("Link", "p_min_pu", names).mul(cap, axis=1)
        record(np.maximum(p0 - upper, lower - p0), "Link")

    units = network.storage_units
    if len(units):
        names = units.index
        cap = capacity(units, "p_nom")
        dispatch = solved(network.storage_units_t, "p_dispatch", names)
        store = solved(network.storage_units_t, "p_store", names)
        soc = solved(network.storage_units_t, "state_of_charge", names)
        upper = dense("StorageUnit", "p_max_pu", names).mul(cap, axis=1)
        lower = dense("StorageUnit", "p_min_pu", names).mul(cap, axis=1)
        record(np.maximum(dispatch - upper.clip(lower=0.0), -dispatch), "StorageUnit dispatch")
        record(np.maximum(store + lower.clip(upper=0.0), -store), "StorageUnit store")
        soc_cap = cap * pd.to_numeric(units["max_hours"], errors="coerce").fillna(0.0)
        record(np.maximum(soc - soc_cap, -soc), "StorageUnit state_of_charge")

    stores = network.stores
    if len(stores):
        names = stores.index
        e = solved(network.stores_t, "e", names)
        cap = capacity(stores, "e_nom")
        upper = dense("Store", "e_max_pu", names).mul(cap, axis=1)
        lower = dense("Store", "e_min_pu", names).mul(cap, axis=1)
        record(np.maximum(e - upper, lower - e), "Store")

    for component, table, series in (("Line", network.lines, network.lines_t),
                                     ("Transformer", network.transformers, network.transformers_t)):
        if not len(table):
            continue
        names = table.index
        flow = solved(series, "p0", names).abs()
        limit = dense(component, "s_max_pu", names).mul(capacity(table, "s_nom"), axis=1)
        record(flow - limit, component)
    return worst[0], worst[1]


def _check_bound_violation(network, logger, context: str, snapshots=None) -> float:
    """Measure and log the accepted solution's bound violation; nan when it cannot be measured."""
    try:
        violation, where = max_bound_violation(network, snapshots)
    except Exception as exc:  # the check must never be what fails a solve
        logger.warning(f"{context}: post-solve bound check could not run ({type(exc).__name__}: {exc})")
        return float("nan")
    if np.isfinite(violation):
        (logger.warning if violation > MAX_BOUND_VIOLATION_MW else logger.info)(
            f"{context}: max post-solve bound violation {violation:.3g} MW"
            + (f" ({where})" if where else "")
        )
    return float(violation)


class SolveFailedError(RuntimeError):
    """Every rung of the fallback ladder failed."""

    def __init__(self, context: str, attempts: Sequence[tuple]):
        self.context = context
        self.attempts = tuple(attempts)
        self.status = self.attempts[-1][1]
        self.condition = self.attempts[-1][2]
        tried = "; ".join(
            f"{name}: {status} ({cond}, {secs:.1f}s)"
            for name, status, cond, secs in self.attempts
        )
        super().__init__(f"{context} failed on every solver rung [{tried}]")


def _option_key(options: dict, key: str) -> str:
    """Return the existing spelling of ``key`` in ``options`` (Gurobi is case-insensitive)."""
    for existing in options:
        if str(existing).lower() == key.lower():
            return existing
    return key


def _set_option(options: dict, key: str, value) -> None:
    options[_option_key(options, key)] = value


def _get_option(options: dict, key: str, default=None):
    return options.get(_option_key(options, key), default)


def _drop_option(options: dict, key: str) -> None:
    options.pop(_option_key(options, key), None)


def conservative_barrier_options(solver_options: dict) -> dict:
    """Configured options plus the conservative numerics (ladder rung 2)."""
    options = dict(solver_options or {})
    for key, value in CONSERVATIVE_BARRIER_OVERRIDES.items():
        _set_option(options, key, value)
    return options


def dual_simplex_options(solver_options: dict) -> dict:
    """Dual simplex with crossover, barrier-only keys dropped (ladder rung 3)."""
    options = dict(solver_options or {})
    for key in BARRIER_ONLY_SOLVER_KEYS:
        _drop_option(options, key)
    _set_option(options, "method", 1)
    _set_option(options, "crossover", -1)
    _set_option(options, "DualReductions", 1)
    _set_option(
        options,
        "NumericFocus",
        max(1, int(_get_option(options, "NumericFocus", 0) or 0)),
    )
    return options


def build_solver_fallback_ladder(
    solver_name: str, solver_options: dict
) -> List[Tuple[str, dict]]:
    """Ordered ``(rung_name, solver_options)`` pairs to try in turn.

    Rung 1 is always the configured options. Gurobi adds the conservative
    barrier profile and then dual simplex; a rung identical to an earlier one
    is skipped (so a run already configured conservatively has two rungs).
    Other solvers have the single configured rung.
    """
    configured = dict(solver_options or {})
    ladder: List[Tuple[str, dict]] = [("configured", configured)]
    if str(solver_name).lower() != "gurobi":
        return ladder
    for name, options in (
        ("conservative_barrier", conservative_barrier_options(configured)),
        ("dual_simplex", dual_simplex_options(configured)),
    ):
        if all(options != existing for _, existing in ladder):
            ladder.append((name, options))
    return ladder


def solve_with_fallback(
    network,
    solver_name: str,
    solver_options: dict,
    extra_functionality: Optional[Callable],
    logger: logging.Logger,
    context: str,
    optimize: Optional[Callable] = None,
    accepted_conditions: Sequence[str] = ACCEPTED_TERMINATION_CONDITIONS,
    max_bound_violation_mw: float = MAX_BOUND_VIOLATION_MW,
    **optimize_kwargs,
) -> SolveOutcome:
    """Run ``network.optimize`` down the fallback ladder until a rung is optimal.

    An accepted rung is then measured with :func:`max_bound_violation` and the
    figure logged (item C11); a solution more than ``max_bound_violation_mw``
    outside its own bounds counts as a failed rung.

    A rung is accepted when the returned status is ``"ok"`` *and* the
    termination condition is in ``accepted_conditions`` (``optimal`` only by
    default: linopy maps ``iteration_limit`` and ``suboptimal`` to status
    ``ok``, which the old ``status == "ok"`` check silently accepted). Anything
    else advances to the next rung. Success on a rung after the first is logged
    as a WARNING naming the rung; failure on every rung raises
    :class:`SolveFailedError` carrying the last status and condition so the
    caller can run its own diagnostics (IIS) before giving up.

    ``optimize`` defaults to ``network.optimize``; pass a wrapper (or a stub in
    tests) taking ``(network, **kwargs)``.
    """
    if optimize is None:
        def optimize(n, **kwargs):
            return n.optimize(**kwargs)

    ladder = build_solver_fallback_ladder(solver_name, solver_options)
    attempts = []
    total = 0.0
    for rung, (rung_name, options) in enumerate(ladder, start=1):
        started = time.perf_counter()
        status, condition = optimize(
            network,
            solver_name=solver_name,
            solver_options=options,
            extra_functionality=extra_functionality,
            **optimize_kwargs,
        )
        elapsed = time.perf_counter() - started
        total += elapsed
        attempts.append((rung_name, str(status), str(condition), elapsed))
        accepted = str(status) == "ok" and str(condition).lower() in {
            c.lower() for c in accepted_conditions
        }
        violation = float("nan")
        if accepted:
            # Item C11: an accepted status can still carry a point outside the
            # bounds; above the threshold the rung counts as failed.
            violation = _check_bound_violation(
                network, logger, context, optimize_kwargs.get("snapshots"))
            if np.isfinite(violation) and violation > max_bound_violation_mw:
                accepted = False
                condition = f"{condition}; bound violation {violation:.1f} MW"
                attempts[-1] = (rung_name, str(status), str(condition), elapsed)
        if accepted:
            if rung > 1:
                logger.warning(
                    f"{context}: solved on fallback rung {rung} ({rung_name}) "
                    f"after {', '.join(f'{n}: {s} ({c})' for n, s, c, _ in attempts[:-1])}"
                )
            return SolveOutcome(
                status=str(status),
                condition=str(condition),
                rung=rung,
                rung_name=rung_name,
                solver_options=options,
                elapsed=total,
                attempts=tuple(attempts),
                bound_violation=violation,
            )
        if rung < len(ladder):
            logger.warning(
                f"{context}: rung {rung} ({rung_name}) returned {status} "
                f"({condition}) after {elapsed:.1f}s; retrying with rung "
                f"{rung + 1} ({ladder[rung][0]})"
            )
    raise SolveFailedError(context, attempts)


def solve_with_counterflow_lock(
    network,
    solver_name: str,
    solver_options: dict,
    extra_functionality: Optional[Callable],
    logger: logging.Logger,
    context: str,
    lock_counterflows: Optional[Callable] = None,
    optimize: Optional[Callable] = None,
    **optimize_kwargs,
) -> SolveOutcome:
    """Solve down the ladder, then lock material interconnector counterflows and re-solve.

    ``lock_counterflows(network, logger=...)`` is
    ``scripts.interconnectors.market_coupling.lock_material_counterflows`` (passed
    in to keep this module free of that import); it returns a truthy value when
    it changed the network and a second solve is needed. The returned outcome
    sums the optimize seconds of both solves and reports the highest rung either
    needed, so a per-window log line describes the whole window.
    """
    outcome = solve_with_fallback(
        network, solver_name, solver_options, extra_functionality, logger,
        context, optimize=optimize, **optimize_kwargs,
    )
    if lock_counterflows is None:
        return outcome
    locks = lock_counterflows(network, logger=logger)
    if not locks:
        return outcome
    second = solve_with_fallback(
        network, solver_name, solver_options, extra_functionality, logger,
        f"{context} [counterflow re-solve]", optimize=optimize, **optimize_kwargs,
    )
    worst = second if second.rung >= outcome.rung else outcome
    return SolveOutcome(
        status=second.status,
        condition=second.condition,
        rung=worst.rung,
        rung_name=worst.rung_name,
        solver_options=second.solver_options,
        elapsed=outcome.elapsed + second.elapsed,
        attempts=outcome.attempts + second.attempts,
        bound_violation=second.bound_violation,
    )


SOLVER_IO_APIS = ("direct", "lp", "mps")
DEFAULT_SOLVER_IO_API = "direct"


def solver_io_kwargs(scenario_config: dict, solver_name: str, logger: Optional[logging.Logger] = None) -> dict:
    """Extra ``optimize()`` kwargs from ``solve.solver.io_api`` (default ``direct``).

    ``direct`` builds the solver model through its Python API (gurobipy /
    highspy) instead of writing an LP file and reading it back, which was 5-7 s
    of a ~21 s wholesale window and 15-17 s of a ~62 s BM window on the 1-week
    network (18 Sep 2026), with an identical objective. ``lp`` / ``mps`` force
    the file route; anything else (or a solver without a direct interface)
    leaves linopy's default.
    """
    io_api = str(((scenario_config or {}).get("solve", {}) or {}).get("solver", {}).get(
        "io_api", DEFAULT_SOLVER_IO_API)).lower()
    if io_api not in SOLVER_IO_APIS:
        return {}
    if io_api == "direct" and str(solver_name).lower() not in {"gurobi", "highs"}:
        if logger:
            logger.info(f"solve.solver.io_api 'direct' is not available for {solver_name}; using the file interface")
        return {}
    return {"io_api": io_api}


def log_fallback_summary(logger: logging.Logger, stage: str, fallback_windows: list, n_windows: int) -> None:
    """One line per stage saying whether any window needed a fallback rung."""
    if fallback_windows:
        shown = ", ".join(f"{w} ({name})" for w, name in fallback_windows[:20])
        more = "" if len(fallback_windows) <= 20 else f" ... and {len(fallback_windows) - 20} more"
        logger.warning(
            f"{stage}: fallback solver rungs used in {len(fallback_windows)} of "
            f"{n_windows} windows: {shown}{more}"
        )
    else:
        logger.info(f"{stage}: no fallback solver rungs used in any of {n_windows} windows")


def _apply_renewable_support_pricing(
    network,
    scenario_config: Optional[dict],
    balancing: dict,
    gen_offer: pd.Series,
    gen_bid: pd.Series,
    logger: logging.Logger,
) -> Tuple[pd.Series, pd.Series]:
    """Apply unit-level CfD/ROC BM prices using the same support labels as MCs.

    This applies only to modelled (derived or future-stack) prices. Direct
    ELEXON prices remain observed historical evidence and are never replaced.
    CfD generators use the configured low curtailment cost; ROC generators use
    their own banding times the same model-year buyout price used in wholesale
    marginal costs and post-solve ROC revenue accounting. Merchant assets retain
    their carrier-specific fallback price because their opportunity cost is not
    a subsidy payment.

    The scalar CfD price this assigns is the pre-B1 behaviour and is kept for
    scenarios that have not opted in. When
    ``support.bm_bids.contract_value_bids`` is enabled, a later pass
    in ``calculate_bid_offer_prices`` replaces it with a per-snapshot
    ``max(strike - reference, 0)`` and drops merchant renewables to short-run
    cost; see ``scripts/market/cfd_bm_bids.py``. One flat CfD bid makes
    balancing curtailment indifferent to strike, which is why that pass exists.
    """
    cfg = ((scenario_config or {}).get("support", {}) or {}).get("bm_bids", {}) or {}
    if not cfg.get("enabled", False):
        return gen_offer, gen_bid
    mode = str(cfg.get("mode", "fixed")).lower()
    if mode == "empirical":
        from scripts.market.support_bid_stack import apply_empirical_support_prices
        return apply_empirical_support_prices(network, cfg, gen_offer, gen_bid, logger)
    if mode != "fixed":
        raise ValueError("support.bm_bids.mode must be fixed or empirical")
    if "support_type" not in network.generators.columns:
        logger.warning("Renewable support-aware BM pricing is enabled but no support_type attributes exist; retaining carrier-level prices")
        return gen_offer, gen_bid

    scenario_config = scenario_config or {}
    subsidy_cfg = (scenario_config.get("support", {}) or {}).get("labels", {}) or {}
    mc_subsidies = (scenario_config.get("support", {}) or {}).get("dispatch_costs", {}) or {}
    if not subsidy_cfg.get("enabled", False) or not mc_subsidies.get("enabled", False):
        logger.warning("Renewable support-aware BM pricing requires support.labels.enabled and support.dispatch_costs.enabled; retaining carrier-level prices")
        return gen_offer, gen_bid

    from scripts.generators.apply_marginal_costs import _get_roc_buyout_price

    modelled_year = int(scenario_config.get("modelled_year", 2024))
    roc_buyout = _get_roc_buyout_price(mc_subsidies.get("roc_buyout_prices", {}), modelled_year)
    cfd_cfg = cfg.get("cfd", {}) or {}
    roc_cfg = cfg.get("roc", {}) or {}
    cfd_offer = float(cfd_cfg.get("offer_price", 0.0))
    cfd_bid = float(cfd_cfg.get("bid_price", 0.0))
    roc_offer = float(roc_cfg.get("offer_price", 0.0))
    roc_multiplier = float(roc_cfg.get("bid_multiplier", 1.0))
    # Carriers whose support label must NOT override the explicit carrier
    # bid/offer prices. Nuclear is the case that matters: Hinkley Point C is a
    # CfD unit, but a zero CfD offer let the balancing LP turn it up for free
    # (6.9 TWh/yr in the CP30 run) against the ``nuclear`` override of 999.
    # Defaults here must match config/defaults/base/support.yaml
    # support.bm_bids.exclude_carriers.
    exclude_carriers = {
        str(c).strip().lower()
        for c in (cfg.get("exclude_carriers", ["nuclear", "CCS Biomass", "biomass"]) or [])
    }

    gen_offer = gen_offer.copy()
    gen_bid = gen_bid.copy()
    summary = []
    excluded = []
    for name, generator in network.generators.iterrows():
        support_type = str(generator.get("support_type", "merchant")).strip().lower()
        carrier = str(generator.get("carrier", "")).strip().lower()
        if carrier in exclude_carriers and support_type in ("cfd", "roc"):
            excluded.append((carrier, float(generator.get("p_nom", 0.0))))
            continue
        if support_type == "cfd":
            gen_offer.at[name] = cfd_offer
            gen_bid.at[name] = cfd_bid
            summary.append(("CfD", float(generator.get("p_nom", 0.0))))
        elif support_type == "roc":
            banding = pd.to_numeric(pd.Series([generator.get("ro_banding")]), errors="coerce").iloc[0]
            if pd.notna(banding) and banding > 0 and roc_buyout > 0:
                gen_offer.at[name] = roc_offer
                gen_bid.at[name] = float(banding) * roc_buyout * roc_multiplier
                summary.append(("ROC", float(generator.get("p_nom", 0.0))))

    if summary:
        report = pd.DataFrame(summary, columns=["support_type", "p_nom"])
        grouped = report.groupby("support_type").agg(units=("support_type", "size"), capacity_mw=("p_nom", "sum"))
        details = ", ".join(f"{support}: {int(row.units)} units / {row.capacity_mw:.0f} MW" for support, row in grouped.iterrows())
        logger.info("Applied unit-level renewable support BM pricing (%s); ROC buyout %.2f GBP/ROC in %d", details, roc_buyout, modelled_year)
    else:
        logger.info("No CfD/ROC renewable units eligible for support-aware BM pricing")
    if excluded:
        ex = pd.DataFrame(excluded, columns=["carrier", "p_nom"]).groupby("carrier")["p_nom"].agg(["size", "sum"])
        logger.info(
            "Support-aware BM pricing skipped for excluded carriers (explicit carrier "
            "bid/offer prices retained): %s",
            ", ".join(f"{c}: {int(r['size'])} units / {r['sum']:.0f} MW" for c, r in ex.iterrows()),
        )
    return gen_offer, gen_bid


def _attach_contract_value_bid_frames(
    network,
    scenario_config: Optional[dict],
    balancing: dict,
    gen_offer: pd.Series,
    gen_bid: pd.Series,
    logger: logging.Logger,
) -> None:
    """Replace the scalar CfD and merchant-renewable bids with contract value.

    A scalar CfD bid makes balancing curtailment indifferent to strike, which is
    the defect backlog item B1 exists to remove. When
    ``support.bm_bids.contract_value_bids`` is enabled this builds the
    snapshots-by-generators frames the balancing solver already accepts and
    attaches them to the network. Every generator the contract-value module does
    not touch keeps the price the rest of the pricing chain gave it.

    Called before ``_apply_participation_policy`` on purpose: the participation
    policy overwrites the time-varying frames for non-participants and for
    penalty overrides, and must have the last word.
    """
    cfg = ((((scenario_config or {}).get("support", {}) or {}).get("bm_bids", {}) or {}).get(
        "contract_value_bids", {}
    ) or {})
    if not cfg.get("enabled", False):
        return
    frames = build_contract_value_bid_frames(
        network,
        scenario_config or {},
        balancing,
        gen_offer,
        gen_bid,
        getattr(network, "_bm_reference_price", None),
        logger,
    )
    network._bm_offer_tv = frames.offer_tv
    network._bm_bid_tv = frames.bid_tv
    network._bm_contract_value_audit = frames.audit
    # Objective-only (decision 20); solve_balancing keeps it out of reported costs.
    network._bm_offer_regulariser = frames.offer_regulariser
    network._bm_bid_regulariser = frames.bid_regulariser


def _overlay_time_varying_markup_prices(
    network,
    carrier_overrides: dict,
    default_offer_markup: float,
    default_bid_discount: float,
    turn_down_saving: bool,
    gen_offer: pd.Series,
    gen_bid: pd.Series,
    logger: logging.Logger,
) -> int:
    """Price markup-mode generators with a time-varying marginal cost per snapshot.

    Heat-led CHP carries a seasonal ``generators_t.marginal_cost``
    (apply_marginal_costs.apply_seasonal_heat_profile); pricing its bids and
    offers on the static annual mean would have the BM see a July unit at its
    January cost. Writes into ``network._bm_offer_tv`` / ``_bm_bid_tv``, creating
    them from the static prices when no earlier pass built them. Absolute-mode
    carriers keep their configured prices. Returns the number of generators.
    """
    varying = getattr(network.generators_t, "marginal_cost", None)
    if varying is None or varying.empty or len(network.snapshots) == 0:
        return 0
    generators = network.generators
    names = [
        g for g in varying.columns
        if g in generators.index
        and (carrier_overrides.get(generators.at[g, "carrier"], {}) or {}).get("mode", "markup") == "markup"
    ]
    if not names:
        return 0
    snapshots = network.snapshots
    mc = varying[names].reindex(snapshots).fillna(generators.loc[names, "marginal_cost"])
    carriers = generators.loc[names, "carrier"]
    markup = carriers.map(lambda c: (carrier_overrides.get(c, {}) or {}).get("offer_markup", default_offer_markup))
    discount = carriers.map(lambda c: (carrier_overrides.get(c, {}) or {}).get("bid_discount", default_bid_discount))
    offer = mc * (1.0 + markup.astype(float))
    if turn_down_saving:
        bid = -(mc - mc.abs() * discount.astype(float))
    else:
        bid = mc * (1.0 - discount.astype(float))

    for attr, static, frame in (("_bm_offer_tv", gen_offer, offer), ("_bm_bid_tv", gen_bid, bid)):
        current = getattr(network, attr, None)
        if current is None:
            current = pd.DataFrame(
                np.tile(static.reindex(generators.index).to_numpy(dtype=float), (len(snapshots), 1)),
                index=snapshots, columns=generators.index,
            )
        current = current.reindex(index=snapshots)
        current.loc[:, names] = frame[names].to_numpy()
        setattr(network, attr, current)
    logger.info(
        f"Time-varying marginal cost priced per snapshot for {len(names)} markup-mode generators "
        f"({', '.join(sorted(carriers.unique()))})"
    )
    return len(names)


def _resolve_bid_offer_source(
    network,
    balancing: dict,
    logger: logging.Logger,
    scenario_id: str = "",
) -> str:
    """
    Resolve the effective bid/offer source.

    ``auto`` prefers ELEXON data for historical scenarios when the required
    files are already available, otherwise it falls back to derived pricing.
    """
    source = balancing.get("bid_offer_source", "auto")
    if source != "auto":
        return source

    snap_year = network.snapshots[0].year if len(network.snapshots) > 0 else None
    if snap_year is None or snap_year > 2024:
        logger.info(
            "bid_offer_source=auto: using derived pricing "
            "(future scenario or no snapshot year)"
        )
        return "derived"

    elexon_cfg = balancing.get("elexon", {})
    data_dir_raw = elexon_cfg.get("data_dir", "resources/market/{scenario}/elexon")
    if "{scenario}" in data_dir_raw and scenario_id:
        data_dir_raw = data_dir_raw.replace("{scenario}", scenario_id)
    data_dir = Path(data_dir_raw)
    bmu_mapping_raw = elexon_cfg.get(
        "bmu_mapping", "resources/generators/{scenario}_bmu_mapping.csv"
    )
    if "{scenario}" in bmu_mapping_raw and scenario_id:
        bmu_mapping_raw = bmu_mapping_raw.replace("{scenario}", scenario_id)
    bmu_mapping_path = Path(bmu_mapping_raw)
    offer_file = data_dir / "elexon_offers.csv"
    bid_file = data_dir / "elexon_bids.csv"

    if bmu_mapping_path.exists() and offer_file.exists() and bid_file.exists():
        logger.info("bid_offer_source=auto: using ELEXON bid/offer data")
        return "elexon"

    logger.info(
        "bid_offer_source=auto: ELEXON files not available, "
        "falling back to derived pricing"
    )
    return "derived"


def _price_ladders_enabled(balancing: dict) -> bool:
    """Return True when ELEXON price-ladder dispatch is enabled."""
    return bool(
        balancing.get("elexon", {})
        .get("price_ladders", {})
        .get("enabled", False)
    )


def _load_elexon_ladder_file(
    path: Path,
    bmu_to_gen: dict,
    generator_index: pd.Index,
    logger: logging.Logger,
    label: str,
) -> pd.DataFrame:
    """
    Load a long-format ELEXON ladder file and map BMUs to model generators.

    Output columns: snapshot, generator, block, price, volume_mw.
    If multiple BMUs map to the same generator, their blocks are stacked and
    re-ranked by price for each generator-hour.
    """
    cols = ["snapshot", "generator", "block", "price", "volume_mw"]
    if not path.exists():
        logger.warning(f"ELEXON {label} ladder file not found: {path}")
        return pd.DataFrame(columns=cols)

    ladders = pd.read_csv(path)
    if ladders.empty:
        return pd.DataFrame(columns=cols)

    required = {"datetime", "bmu_id", "price", "volume_mw"}
    missing = required.difference(ladders.columns)
    if missing:
        raise ValueError(
            f"ELEXON {label} ladder file {path} is missing columns: {sorted(missing)}"
        )

    ladders["snapshot"] = pd.to_datetime(ladders["datetime"])
    ladders["generator"] = ladders["bmu_id"].map(bmu_to_gen)
    ladders["price"] = pd.to_numeric(ladders["price"], errors="coerce")
    ladders["volume_mw"] = pd.to_numeric(ladders["volume_mw"], errors="coerce")
    ladders = ladders[
        ladders["generator"].isin(generator_index)
        & ladders["snapshot"].notna()
        & ladders["price"].notna()
        & ladders["volume_mw"].notna()
        & (ladders["volume_mw"] > 0)
    ].copy()
    if ladders.empty:
        logger.warning(f"ELEXON {label} ladders loaded but no rows mapped to generators")
        return pd.DataFrame(columns=cols)

    # Merge identical price blocks after BMU->generator mapping, then re-rank.
    ladders = (
        ladders.groupby(["snapshot", "generator", "price"], as_index=False)[
            "volume_mw"
        ]
        .sum()
        .sort_values(["snapshot", "generator", "price"])
    )
    ladders["block"] = (
        ladders.groupby(["snapshot", "generator"]).cumcount() + 1
    ).astype(int)

    logger.info(
        f"Loaded ELEXON {label} ladders: {len(ladders):,} blocks, "
        f"{ladders['generator'].nunique()} generators"
    )
    return ladders[cols]


def _resolve_component_participants(
    components: pd.DataFrame,
    participation_cfg: dict,
    matched_names: pd.Index | None,
    logger: logging.Logger,
    component_label: str,
) -> tuple[pd.Index, pd.Index, str, float, float]:
    """Resolve which components participate in BM and how others are handled."""
    cfg = participation_cfg or {}
    mode = str(cfg.get("mode", "all")).lower()
    behavior = str(cfg.get("behavior", "priced_out")).lower()
    penalty_offer = float(cfg.get("penalty_offer_price", 5000.0))
    penalty_bid = float(cfg.get("penalty_bid_price", 5000.0))

    if behavior not in {"priced_out", "fixed", "fallback_priced"}:
        raise ValueError(
            f"Unsupported {component_label} participation behavior='{behavior}'. "
            "Use 'priced_out', 'fixed', or 'fallback_priced'."
        )

    component_index = components.index
    component_set = set(component_index)
    matched_set = set((matched_names if matched_names is not None else pd.Index([])).tolist())

    if mode == "all":
        participants = component_set.copy()
    elif mode == "none":
        participants = set()
    elif mode == "elexon_mapped":
        participants = component_set.intersection(matched_set)
        if len(participants) == 0 and len(component_index) > 0:
            logger.warning(
                f"{component_label} participation mode 'elexon_mapped' selected, "
                f"but no directly matched ELEXON {component_label} were found."
            )
    else:
        raise ValueError(
            f"Unsupported {component_label} participation mode='{mode}'. "
            "Use 'all', 'none', or 'elexon_mapped'."
        )

    min_p_nom_mw = float(cfg.get("min_p_nom_mw", 0.0) or 0.0)
    if min_p_nom_mw > 0.0 and "p_nom" in components.columns:
        participants &= set(
            components.index[components["p_nom"].fillna(0.0) >= min_p_nom_mw]
        )

    include_carriers = set(cfg.get("include_carriers", []) or [])
    if include_carriers and "carrier" in components.columns:
        participants &= set(
            components.index[components["carrier"].isin(include_carriers)]
        )

    exclude_carriers = set(cfg.get("exclude_carriers", []) or [])
    if exclude_carriers and "carrier" in components.columns:
        participants -= set(
            components.index[components["carrier"].isin(exclude_carriers)]
        )

    include_names = set(str(v) for v in (cfg.get("include_names", []) or []))
    if include_names:
        participants &= include_names

    exclude_names = set(str(v) for v in (cfg.get("exclude_names", []) or []))
    if exclude_names:
        participants -= exclude_names

    participant_index = pd.Index(
        [name for name in component_index if name in participants]
    )
    non_participant_index = component_index.difference(participant_index)
    return (
        participant_index,
        non_participant_index,
        behavior,
        penalty_offer,
        penalty_bid,
    )


def _apply_participation_policy(
    network,
    market_config: dict,
    logger: logging.Logger,
    gen_offer: pd.Series,
    gen_bid: pd.Series,
    su_offer: pd.Series,
    su_bid: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Apply optional BM participation filters and attach solver metadata."""
    balancing = market_config.get("balancing", {})
    participation_cfg = balancing.get("participation", {}) or {}

    gen_cfg = participation_cfg.get("generators", {}) or {}
    direct_matched_gens = getattr(network, "_bm_direct_matched_generators", pd.Index([]))
    (
        gen_participants,
        gen_non_participants,
        gen_behavior,
        gen_penalty_offer,
        gen_penalty_bid,
    ) = _resolve_component_participants(
        network.generators,
        gen_cfg,
        direct_matched_gens,
        logger,
        "generator",
    )

    gen_penalty_override = pd.Index([])
    if gen_behavior == "fixed" and "carrier" in network.generators.columns:
        gen_penalty_override = gen_non_participants.intersection(
            network.generators.index[
                network.generators["carrier"] == "load_shedding"
            ]
        )
        if len(gen_penalty_override) > 0:
            gen_participants = pd.Index(
                [
                    name
                    for name in network.generators.index
                    if name in set(gen_participants).union(gen_penalty_override)
                ]
            )
            gen_non_participants = gen_non_participants.difference(gen_penalty_override)
            logger.info(
                f"Keeping {len(gen_penalty_override)} load-shedding generators "
                "available as priced-out feasibility slack"
            )

    su_cfg = participation_cfg.get("storage_units", {}) or {}
    (
        su_participants,
        su_non_participants,
        su_behavior,
        su_penalty_offer,
        su_penalty_bid,
    ) = _resolve_component_participants(
        network.storage_units,
        su_cfg,
        matched_names=pd.Index([]),
        logger=logger,
        component_label="storage unit",
    )

    network._bm_eligible_generators = gen_participants
    network._bm_participating_generators = (
        network.generators.index
        if gen_behavior in {"priced_out", "fallback_priced"}
        else gen_participants
    )
    network._bm_fixed_generators = (
        gen_non_participants if gen_behavior == "fixed" else pd.Index([])
    )
    network._bm_eligible_storage_units = su_participants
    network._bm_participating_storage_units = (
        network.storage_units.index
        if su_behavior in {"priced_out", "fallback_priced"}
        else su_participants
    )
    network._bm_fixed_storage_units = (
        su_non_participants if su_behavior == "fixed" else pd.Index([])
    )
    network._bm_generator_participation_behavior = gen_behavior
    network._bm_storage_participation_behavior = su_behavior

    # hold_carriers: generators of these carriers are not balancing-mechanism units (in 2020, most
    # embedded wind and solar). They submit no bids or offers: both are set to the penalty prices, so
    # the BM keeps them at their Stage 1 position unless the network leaves no other way, and any such
    # turn-down is reported apart (market_accounting non_participant_turn_down). Under
    # ``fallback_priced`` an ``exclude_carriers`` entry alone keeps its fallback prices, so the BM still
    # turned it down (2020 validation, 19 Nov 2026: 3.1 TWh of embedded wind). Fixing them outright
    # left 22 May 2020 infeasible: a node that cannot export its embedded output has no other relief.
    hold_carriers = set(gen_cfg.get("hold_carriers", []) or [])
    if hold_carriers and "carrier" in network.generators.columns:
        held = network.generators.index[network.generators["carrier"].isin(hold_carriers)]
        if len(held):
            network._bm_eligible_generators = pd.Index(
                [g for g in network._bm_eligible_generators if g not in set(held)])
            gen_offer.loc[held] = gen_penalty_offer
            gen_bid.loc[held] = gen_penalty_bid
            if getattr(network, "_bm_offer_tv", None) is not None:
                network._bm_offer_tv.loc[:, network._bm_offer_tv.columns.intersection(held)] = gen_penalty_offer
            if getattr(network, "_bm_bid_tv", None) is not None:
                network._bm_bid_tv.loc[:, network._bm_bid_tv.columns.intersection(held)] = gen_penalty_bid
            for attr in ("_bm_offer_ladders", "_bm_bid_ladders"):
                ladders = getattr(network, attr, None)
                if ladders is not None:
                    setattr(network, attr, ladders[~ladders["generator"].isin(held)].copy())
            logger.info(f"Holding {len(held)} generators of carriers {sorted(hold_carriers)} at their Stage 1 "
                        "position: no bids or offers, turn-down only at the penalty price "
                        "(participation.generators.hold_carriers)")

    # Consumer flexibility (F5) is demand dispatched by its owners' tariffs in Stage 1: the BM holds
    # it at its Stage 1 schedule whatever the participation lists say (both years; a family that
    # rewrites exclude_carriers cannot drop it by omission). A sensitivity that lets it bid sets
    # participation.storage_units.hold_consumer_flexibility: false.
    if bool(su_cfg.get("hold_consumer_flexibility", True)) and "carrier" in network.storage_units.columns:
        from scripts.demand.demand_carriers import CONSUMER_FLEX_CARRIERS

        flex = set(network.storage_units.index[network.storage_units["carrier"].isin(CONSUMER_FLEX_CARRIERS)])
        if flex:
            network._bm_eligible_storage_units = pd.Index(
                [s for s in network._bm_eligible_storage_units if s not in flex])
            network._bm_participating_storage_units = pd.Index(
                [s for s in network._bm_participating_storage_units if s not in flex])
            held = set(network._bm_fixed_storage_units) | flex
            network._bm_fixed_storage_units = pd.Index([s for s in network.storage_units.index if s in held])
            su_non_participants = su_non_participants.difference(pd.Index(sorted(flex)))
            logger.info(f"Holding {len(flex)} consumer flexibility units at their Stage 1 schedule")

    if gen_behavior == "priced_out" and len(gen_non_participants) > 0:
        gen_offer.loc[gen_non_participants] = gen_penalty_offer
        gen_bid.loc[gen_non_participants] = gen_penalty_bid
        if hasattr(network, "_bm_offer_tv") and network._bm_offer_tv is not None:
            network._bm_offer_tv.loc[:, gen_non_participants] = gen_penalty_offer
        if hasattr(network, "_bm_bid_tv") and network._bm_bid_tv is not None:
            network._bm_bid_tv.loc[:, gen_non_participants] = gen_penalty_bid
        if hasattr(network, "_bm_offer_ladders") and network._bm_offer_ladders is not None:
            network._bm_offer_ladders = network._bm_offer_ladders[
                ~network._bm_offer_ladders["generator"].isin(gen_non_participants)
            ].copy()
        if hasattr(network, "_bm_bid_ladders") and network._bm_bid_ladders is not None:
            network._bm_bid_ladders = network._bm_bid_ladders[
                ~network._bm_bid_ladders["generator"].isin(gen_non_participants)
            ].copy()
    elif gen_behavior == "fixed" and len(gen_penalty_override) > 0:
        gen_offer.loc[gen_penalty_override] = gen_penalty_offer
        gen_bid.loc[gen_penalty_override] = gen_penalty_bid
        if hasattr(network, "_bm_offer_tv") and network._bm_offer_tv is not None:
            network._bm_offer_tv.loc[:, gen_penalty_override] = gen_penalty_offer
        if hasattr(network, "_bm_bid_tv") and network._bm_bid_tv is not None:
            network._bm_bid_tv.loc[:, gen_penalty_override] = gen_penalty_bid
        if hasattr(network, "_bm_offer_ladders") and network._bm_offer_ladders is not None:
            network._bm_offer_ladders = network._bm_offer_ladders[
                ~network._bm_offer_ladders["generator"].isin(gen_penalty_override)
            ].copy()
        if hasattr(network, "_bm_bid_ladders") and network._bm_bid_ladders is not None:
            network._bm_bid_ladders = network._bm_bid_ladders[
                ~network._bm_bid_ladders["generator"].isin(gen_penalty_override)
            ].copy()
    elif gen_behavior == "fallback_priced" and len(gen_non_participants) > 0:
        logger.info(
            f"Keeping fallback bid/offer prices for {len(gen_non_participants)} "
            "non-eligible generators"
        )

    if su_behavior == "priced_out" and len(su_non_participants) > 0:
        su_offer.loc[su_non_participants] = su_penalty_offer
        su_bid.loc[su_non_participants] = su_penalty_bid
        if hasattr(network, "_bm_su_offer_tv") and network._bm_su_offer_tv is not None:
            network._bm_su_offer_tv.loc[:, su_non_participants] = su_penalty_offer
        if hasattr(network, "_bm_su_bid_tv") and network._bm_su_bid_tv is not None:
            network._bm_su_bid_tv.loc[:, su_non_participants] = su_penalty_bid
    elif su_behavior == "fallback_priced" and len(su_non_participants) > 0:
        logger.info(
            f"Keeping fallback bid/offer prices for {len(su_non_participants)} "
            "non-eligible storage units"
        )

    logger.info(
        f"BM generator participation: {len(gen_participants)}/{len(network.generators)} "
        f"eligible, {len(gen_non_participants)} non-participants "
        f"({gen_behavior})"
    )
    logger.info(
        f"BM storage participation: {len(su_participants)}/{len(network.storage_units)} "
        f"eligible, {len(su_non_participants)} non-participants "
        f"({su_behavior})"
    )

    return gen_offer, gen_bid, su_offer, su_bid


def calculate_bid_offer_prices(
    network,
    market_config: dict,
    logger: logging.Logger,
    scenario_id: str = "",
    time_varying: bool = False,
    scenario_config: Optional[dict] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate bid and offer prices for generators and storage units.

    Supports two modes per carrier (set in carrier_overrides):
      "markup"    offer = mc  (1 + markup),
                  bid = -(mc - discount * |mc|)  (derived_bid_convention
                  "net_saving"; "legacy_cost" restores bid = mc  (1 - discount))
      "absolute"  offer/bid are fixed /MWh values from config
      "round_trip" (storage) bid = -bid_price, offer = bid_price / eta_rt + premium
                  (``_derive_storage_prices``)

    Bid prices use ESO-cost convention: positive = costs ESO to decrease.
    ELEXON raw bids (negative = ESO pays) are negated at load time, and a
    markup-mode generator bid is the raw bid a unit that saves its marginal
    cost would submit, negated the same way (see DERIVED_BID_CONVENTIONS).
    A round_trip storage ``bid_price`` is configured in the raw sign too (what
    the unit pays to charge) and negated the same way.

    Parameters
    ----------
    network : pypsa.Network
        Network with generators and storage units (marginal_cost must be set).
    market_config : dict
        Market configuration dict containing 'balancing' sub-dict.
    logger : logging.Logger
        Logger instance.
    scenario_id : str, optional
        Scenario identifier for file path expansion.
    time_varying : bool, optional
        If True and source is ELEXON, also populate
        ``network._bm_offer_tv`` and ``network._bm_bid_tv`` with
        per-snapshot DataFrames (snapshots  generators), and populate
        ``network._bm_su_offer_tv`` / ``network._bm_su_bid_tv`` for storage
        carriers when ELEXON storage price traces are configured.

    Returns
    -------
    gen_offer_prices : pd.Series
        Offer prices per generator (/MWh), indexed by generator name.
    gen_bid_prices : pd.Series
        Bid prices per generator (/MWh), indexed by generator name.
    su_offer_prices : pd.Series
        Offer prices per storage unit (/MWh), indexed by storage unit name.
    su_bid_prices : pd.Series
        Bid prices per storage unit (/MWh), indexed by storage unit name.
    """
    # pricing_basis at_cost (central dispatch) reads every markup and discount as zero.
    balancing = priced_balancing(market_config.get("balancing", {}))
    # Support-aware bids (CfD/ROC/merchant, F8 storage at cost) are the scenario's support.bm_bids.
    bm_bids = ((scenario_config or {}).get("support", {}) or {}).get("bm_bids", {}) or {}
    default_offer_markup = balancing.get("default_offer_markup", 0.10)
    default_bid_discount = balancing.get("default_bid_discount", 0.10)
    carrier_overrides = balancing.get("carrier_overrides", {})
    turn_down_saving = _derived_bid_turn_down_saving(balancing, logger)
    storage_churn_guard = float(
        balancing.get("storage_churn_guard_gbp_per_mwh", DEFAULT_STORAGE_CHURN_GUARD)
    )
    bid_offer_source = _resolve_bid_offer_source(
        network, balancing, logger, scenario_id=scenario_id
    )

    #  Source: ELEXON 
    if bid_offer_source == "elexon":
        # Guard: ELEXON data only exists for historical periods (2024)
        snap_year = network.snapshots[0].year if len(network.snapshots) > 0 else None
        if snap_year and snap_year > 2024:
            raise ValueError(
                f"bid_offer_source='elexon' is not valid for future scenarios "
                f"(network year {snap_year} > 2024). Use 'derived' or 'csv' instead."
            )
        gen_offer, gen_bid, su_offer, su_bid = _load_elexon_bid_offer(
            network, balancing, default_offer_markup, default_bid_discount,
            carrier_overrides, logger,
            scenario_id=scenario_id,
            time_varying=time_varying,
            turn_down_saving=turn_down_saving,
        )
        return _apply_participation_policy(
            network, market_config, logger, gen_offer, gen_bid, su_offer, su_bid
        )

    #  Source: CSV 
    if bid_offer_source == "csv":
        gen_offer, gen_bid, su_offer, su_bid = _load_csv_bid_offer(
            network, balancing, logger,
        )
        return _apply_participation_policy(
            network, market_config, logger, gen_offer, gen_bid, su_offer, su_bid
        )

    #  Source: derived (default) 
    # Source: empirical future bid stack.
    if bid_offer_source == "future_bid_stack":
        # Start with existing technology and subsidy-aware derived prices. The
        # empirical module replaces only carriers with sufficient historic
        # ELEXON evidence, rebased to this future network's own marginal cost.
        gen_offer, gen_bid = _compute_component_prices(
            network.generators,
            carrier_overrides,
            default_offer_markup,
            default_bid_discount,
            turn_down_saving=turn_down_saving,
        )
        if len(network.storage_units) > 0:
            su_offer, su_bid = _derive_storage_prices(
                network.storage_units,
                carrier_overrides,
                default_offer_markup,
                default_bid_discount,
                logger,
                churn_guard=storage_churn_guard,
            )
        else:
            su_offer = pd.Series(dtype=float)
            su_bid = pd.Series(dtype=float)

        from scripts.market.future_bid_stack import apply_future_bid_stack

        gen_offer, gen_bid, su_offer, su_bid = apply_future_bid_stack(
            network,
            balancing,
            gen_offer,
            gen_bid,
            su_offer,
            su_bid,
            logger,
        )
        gen_offer, gen_bid = _apply_renewable_support_pricing(
            network, scenario_config, balancing, gen_offer, gen_bid, logger
        )
        if time_varying:
            _attach_contract_value_bid_frames(
                network, scenario_config, balancing, gen_offer, gen_bid, logger
            )
            _attach_storage_price_frames(network, balancing, su_offer, su_bid, logger, bm_bids=bm_bids)
        return _apply_participation_policy(
            network, market_config, logger, gen_offer, gen_bid, su_offer, su_bid
        )
    if bid_offer_source != "derived":
        raise ValueError(
            f"Unknown bid_offer_source='{bid_offer_source}'. "
            "Supported: 'auto', 'derived', 'elexon', 'csv', 'future_bid_stack'."
        )

    # Generators
    gen_offer, gen_bid = _compute_component_prices(
        network.generators,
        carrier_overrides,
        default_offer_markup,
        default_bid_discount,
        turn_down_saving=turn_down_saving,
    )
    logger.info(
        f"Bid/offer prices calculated for {len(gen_offer)} generators "
        f"(offer range: {gen_offer.min():.2f}-{gen_offer.max():.2f}/MWh, "
        f"bid range: {gen_bid.min():.2f}-{gen_bid.max():.2f}/MWh)"
    )

    # Storage units
    if len(network.storage_units) > 0:
        su_offer, su_bid = _derive_storage_prices(
            network.storage_units,
            carrier_overrides,
            default_offer_markup,
            default_bid_discount,
            logger,
            churn_guard=storage_churn_guard,
        )
        logger.info(
            f"Bid/offer prices calculated for {len(su_offer)} storage units "
            f"(offer range: {su_offer.min():.2f}-{su_offer.max():.2f}/MWh, "
            f"bid range: {su_bid.min():.2f}-{su_bid.max():.2f}/MWh)"
        )
    else:
        su_offer = pd.Series(dtype=float)
        su_bid = pd.Series(dtype=float)

    gen_offer, gen_bid = _apply_renewable_support_pricing(
        network, scenario_config, balancing, gen_offer, gen_bid, logger
    )
    if time_varying:
        _attach_contract_value_bid_frames(
            network, scenario_config, balancing, gen_offer, gen_bid, logger
        )
        _overlay_time_varying_markup_prices(
            network, carrier_overrides, default_offer_markup, default_bid_discount,
            turn_down_saving, gen_offer, gen_bid, logger,
        )
        _attach_storage_price_frames(network, balancing, su_offer, su_bid, logger, bm_bids=bm_bids)
    return _apply_participation_policy(
        network, market_config, logger, gen_offer, gen_bid, su_offer, su_bid
    )


def _load_elexon_bid_offer(
    network, balancing: dict,
    default_offer_markup: float, default_bid_discount: float,
    carrier_overrides: dict,
    logger: logging.Logger,
    scenario_id: str = "",
    time_varying: bool = False,
    turn_down_saving: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load bid/offer prices from pre-fetched ELEXON BMRS data.

    Reads per-BMU bid/offer CSVs, maps BMU IDs to PyPSA generator names,
    and fills unmatched generators using carrier-group averages from the
    ELEXON dataset. Storage units fall back to derived pricing.

    When ``time_varying=True``, also attaches per-snapshot price DataFrames
    to the network as ``network._bm_offer_tv`` and ``network._bm_bid_tv``
    (snapshots  generators).  ELEXON-matched generators get their actual
    hourly prices; unmatched generators get their static price broadcast.

    Parameters
    ----------
    network : pypsa.Network
    balancing : dict
        Balancing config with 'elexon' sub-dict.
    default_offer_markup, default_bid_discount : float
        Fallback markups for derived pricing.
    carrier_overrides : dict
    logger : logging.Logger

    Returns
    -------
    gen_offer, gen_bid, su_offer, su_bid : pd.Series
    """
    elexon_cfg = balancing.get("elexon", {})
    data_dir_raw = elexon_cfg.get("data_dir", "resources/market/elexon")
    # Expand {scenario} template if present
    if "{scenario}" in data_dir_raw and scenario_id:
        data_dir_raw = data_dir_raw.replace("{scenario}", scenario_id)
    data_dir = Path(data_dir_raw)
    bmu_mapping_raw = elexon_cfg.get(
        "bmu_mapping", "resources/generators/{scenario}_bmu_mapping.csv"
    )
    if "{scenario}" in bmu_mapping_raw and scenario_id:
        bmu_mapping_raw = bmu_mapping_raw.replace("{scenario}", scenario_id)
    bmu_mapping_path = Path(bmu_mapping_raw)
    fallback = elexon_cfg.get("fallback", "derived")

    # Load BMU  generator mapping
    if not bmu_mapping_path.exists():
        raise FileNotFoundError(
            f"BMU mapping file not found: {bmu_mapping_path}. "
            "Run the ELEXON data retrieval rule first."
        )
    bmu_map = pd.read_csv(bmu_mapping_path)
    # Expect columns: bmu_id, generator_name (at minimum)
    bmu_to_gen = dict(zip(bmu_map["bmu_id"], bmu_map["generator_name"]))
    logger.info(f"Loaded BMU mapping: {len(bmu_to_gen)} BMUgenerator entries")

    # Load ELEXON offer/bid CSVs
    offer_file = data_dir / "elexon_offers.csv"
    bid_file = data_dir / "elexon_bids.csv"
    for f in [offer_file, bid_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"ELEXON data file not found: {f}. "
                "Run the retrieve_elexon_market_data rule first."
            )

    elexon_offers = pd.read_csv(offer_file, index_col=0)
    elexon_bids = pd.read_csv(bid_file, index_col=0)
    elexon_offers.index = pd.to_datetime(elexon_offers.index)
    elexon_bids.index = pd.to_datetime(elexon_bids.index)
    # Remove duplicate timestamps (can occur at DST transitions or data glitches).
    # Keep first occurrence so reindex() against network snapshots works cleanly.
    if elexon_offers.index.duplicated().any():
        n_dup = elexon_offers.index.duplicated().sum()
        logger.warning(f"ELEXON offers: dropping {n_dup} duplicate timestamps")
        elexon_offers = elexon_offers[~elexon_offers.index.duplicated(keep="first")]
    if elexon_bids.index.duplicated().any():
        n_dup = elexon_bids.index.duplicated().sum()
        logger.warning(f"ELEXON bids: dropping {n_dup} duplicate timestamps")
        elexon_bids = elexon_bids[~elexon_bids.index.duplicated(keep="first")]
    # A bid at or above +1,000 GBP/MWh (the unit would pay that to be turned down: ELEXON's +9,998 "do not
    # accept" sentinel and its kin) is a refusal, not a price. Taken as a price it made turning the unit down a
    # revenue: Kype Muir earned the system operator GBP 3.7 m in January 2020 (18 Nov 2026). Those hours fall
    # back like hours with no bid.
    defensive = elexon_bids >= DEFENSIVE_BID_GBP_PER_MWH
    if defensive.any().any():
        logger.info(f"ELEXON bids: {int(defensive.sum().sum())} BMU-hours at or above "
                    f"+{DEFENSIVE_BID_GBP_PER_MWH:.0f} GBP/MWh treated as no bid")
        elexon_bids = elexon_bids.mask(defensive)
    elexon_offers_raw = elexon_offers.copy()
    elexon_bids_raw = elexon_bids.copy()
    logger.info(f"Loaded ELEXON offers: {elexon_offers.shape}, bids: {elexon_bids.shape}")

    # Map BMU columns to generator names
    elexon_offers = elexon_offers.rename(columns=bmu_to_gen)
    elexon_bids = elexon_bids.rename(columns=bmu_to_gen)
    if elexon_offers.columns.has_duplicates:
        elexon_offers = elexon_offers.T.groupby(level=0).mean().T
    if elexon_bids.columns.has_duplicates:
        elexon_bids = elexon_bids.T.groupby(level=0).mean().T

    # Build per-generator average offer/bid prices (mean across snapshots)
    gen_offer = pd.Series(index=network.generators.index, dtype=float)
    gen_bid = pd.Series(index=network.generators.index, dtype=float)

    matched = 0
    direct_matched_generators = []
    one_sided = []
    for gen_name in network.generators.index:
        # FIX-2: once sentinel prices are dropped (elexon_data FIX-BOD-1) a BMU
        # can have real offers but no bids, or the reverse (e.g. Didcot GT:
        # every bid was a 9999 sentinel). The old code assumed both columns
        # exist and raised KeyError. A generator is directly matched only
        # when both sides have a real price; otherwise it uses the fallback.
        has_offer = gen_name in elexon_offers.columns and elexon_offers[gen_name].notna().any()
        has_bid = gen_name in elexon_bids.columns and elexon_bids[gen_name].notna().any()
        if has_offer != has_bid:
            one_sided.append(gen_name)
        if has_offer and has_bid:
            gen_offer[gen_name] = elexon_offers[gen_name].mean()
            # ELEXON sign convention: negative bid = ESO must PAY generator to reduce.
            # Our objective uses: min  offerinc + biddec, where bid must be
            # positive when it costs the ESO money.  Negate ELEXON bids so that
            # the downstream convention is: positive bid = cost to ESO to decrease,
            # negative bid = ESO is paid to turn the unit down.
            gen_bid[gen_name] = -elexon_bids[gen_name].mean()
            matched += 1
            direct_matched_generators.append(gen_name)

    network._bm_direct_matched_generators = pd.Index(direct_matched_generators)
    if one_sided:
        logger.info(
            f"FIX-2: {len(one_sided)} generator(s) have ELEXON prices on one side only "
            f"(after removing sentinels) and use the fallback: {one_sided[:10]}"
        )
    # FIX-1: record where each generator's BM price comes from, so the
    # calibration scorecard can leave out prices copied from the ELEXON
    # carrier median (comparing those with that median is circular).
    price_source = pd.Series("unpriced", index=network.generators.index, dtype=object)
    price_source.loc[pd.Index(direct_matched_generators)] = "elexon_bmu"

    ladder_cfg = elexon_cfg.get("price_ladders", {}) or {}
    if bool(ladder_cfg.get("enabled", False)):
        offer_ladder_file = Path(
            ladder_cfg.get("offer_file", data_dir / "elexon_offer_ladders.csv")
        )
        bid_ladder_file = Path(
            ladder_cfg.get("bid_file", data_dir / "elexon_bid_ladders.csv")
        )
        if not offer_ladder_file.is_absolute() and offer_ladder_file.parent == Path("."):
            offer_ladder_file = data_dir / offer_ladder_file
        if not bid_ladder_file.is_absolute() and bid_ladder_file.parent == Path("."):
            bid_ladder_file = data_dir / bid_ladder_file

        network._bm_offer_ladders = _load_elexon_ladder_file(
            offer_ladder_file,
            bmu_to_gen,
            network.generators.index,
            logger,
            label="offer",
        )
        network._bm_bid_ladders = _load_elexon_ladder_file(
            bid_ladder_file,
            bmu_to_gen,
            network.generators.index,
            logger,
            label="bid",
        )
        network._bm_ladder_fallback_volume_mw = float(
            ladder_cfg.get("fallback_volume_mw", 1.0e6)
        )
        network._bm_ladder_missing_hour_fallback = int(
            bool(ladder_cfg.get("missing_hour_fallback", True))
        )
    else:
        network._bm_offer_ladders = None
        network._bm_bid_ladders = None

    #  Fitted fallback (opt-in) 
    # Fit per-carrier offer/bid = MC + , MC +  from ELEXON-matched
    # generators, apply to unmatched ones in the same carrier. Runs BEFORE
    # carrier_average / derived so it takes precedence when enabled.
    fitted_cfg = balancing.get("fitted_fallback", {})
    fitted_enabled = bool(fitted_cfg.get("enabled", False))
    fitted_applied_idx = pd.Index([])
    carrier_fits: dict = {}
    if fitted_enabled:
        from scripts.market.bm_elexon_fit import (
            fit_carrier_prices,
            apply_fitted_fallback,
        )
        min_matched_gens = int(fitted_cfg.get("min_matched_gens", 3))
        min_r2 = float(fitted_cfg.get("min_r2", 0.1))
        fitted_max_price = float(fitted_cfg.get("max_price", 500.0))

        mapped_gen_names = set(bmu_to_gen.values())
        # Negate ELEXON bids for the fit so coefficients are in ESO-cost
        # convention, matching gen_bid downstream.
        logger.info("Fitting per-carrier ELEXON price regressions...")
        carrier_fits = fit_carrier_prices(
            generators=network.generators,
            elexon_offers=elexon_offers,
            elexon_bids=-elexon_bids,
            mapped_gen_names=mapped_gen_names,
            min_matched_gens=min_matched_gens,
            logger=logger,
        )

        unmatched_idx_pre = network.generators.index[gen_offer.isna()]
        fit_offer, fit_bid, fitted_applied_idx = apply_fitted_fallback(
            generators=network.generators,
            unmatched_index=unmatched_idx_pre,
            fits=carrier_fits,
            min_r2=min_r2,
            max_price=fitted_max_price,
            logger=logger,
        )
        if len(fitted_applied_idx) > 0:
            gen_offer.loc[fitted_applied_idx] = fit_offer.loc[fitted_applied_idx]
            gen_bid.loc[fitted_applied_idx] = fit_bid.loc[fitted_applied_idx]
            price_source.loc[fitted_applied_idx] = "fitted"  # FIX-1

    # Fill remaining unmatched generators with the configured fallback
    unmatched = gen_offer.isna()
    if unmatched.any():
        if fallback == "carrier_average":
            # Use median ELEXON prices from matched generators of the same
            # carrier.  Median is robust to strategic outliers (e.g., nuclear
            # 5000 defensive bids, wind 99999 offers).  Falls back to
            # derived pricing when the median exceeds a sanity cap or no
            # ELEXON-matched generators exist for the carrier.
            max_price = float(elexon_cfg.get("carrier_average_max_price", 500.0))
            carrier_avg_offer, carrier_avg_bid, carrier_avg_rejected = carrier_medians_within_cap(
                network.generators["carrier"], gen_offer, gen_bid, max_price)

            # Compute derived prices as ultimate fallback
            derived_offer, derived_bid = _compute_component_prices(
                network.generators,
                carrier_overrides,
                default_offer_markup,
                default_bid_discount,
                turn_down_saving=turn_down_saving,
            )

            filled_carrier_avg = 0
            filled_derived = 0
            for idx in network.generators.index[unmatched]:
                carrier = network.generators.loc[idx, "carrier"]
                gen_offer[idx] = carrier_avg_offer.get(carrier, derived_offer[idx])
                gen_bid[idx] = carrier_avg_bid.get(carrier, derived_bid[idx])
                if carrier in carrier_avg_offer or carrier in carrier_avg_bid:
                    filled_carrier_avg += 1
                    price_source[idx] = "carrier_median"  # FIX-1
                else:
                    filled_derived += 1
                    price_source[idx] = "derived"  # FIX-1

            logger.info(
                f"Carrier-average fallback: {filled_carrier_avg} generators filled "
                f"from ELEXON carrier medians, {filled_derived} from derived pricing"
            )
            for c in sorted(set(carrier_avg_offer) | set(carrier_avg_bid)):
                offer_text = f"{carrier_avg_offer[c]:.1f}" if c in carrier_avg_offer else "derived"
                bid_text = f"{carrier_avg_bid[c]:.1f}" if c in carrier_avg_bid else "derived"
                logger.info(f"  {c}: ELEXON median offer={offer_text}/MWh, bid={bid_text}/MWh")
            if carrier_avg_rejected:
                for c, (mo, mb) in sorted(carrier_avg_rejected.items()):
                    logger.info(
                        f"  {c}: an ELEXON median exceeded the cap (offer={mo:.0f}, "
                        f"bid={mb:.0f}, cap={max_price:.0f}); that side uses derived pricing"
                    )

        elif fallback == "derived":
            derived_offer, derived_bid = _compute_component_prices(
                network.generators,
                carrier_overrides,
                default_offer_markup,
                default_bid_discount,
                turn_down_saving=turn_down_saving,
            )
            gen_offer.loc[unmatched] = derived_offer.loc[unmatched]
            gen_bid.loc[unmatched] = derived_bid.loc[unmatched]
            price_source.loc[unmatched] = "derived"  # FIX-1
        else:
            gen_offer[unmatched] = 0.0
            gen_bid[unmatched] = 0.0
            price_source.loc[unmatched] = "zero"  # FIX-1
    network._bm_price_source = price_source  # FIX-1

    matched_mask = ~unmatched
    _log_elexon_coverage_summary(
        network,
        matched_mask=matched_mask,
        fallback_mask=unmatched,
        fallback_label=fallback,
        logger=logger,
    )

    #  Build time-varying price matrices 
    # For ELEXON-matched generators: per-snapshot prices from the CSVs.
    # For unmatched: static price broadcast to all snapshots.
    if time_varying and len(network.snapshots) > 0:
        snaps = network.snapshots
        gen_names = network.generators.index

        # Start with static prices broadcast across snapshots
        offer_tv = pd.DataFrame(
            np.tile(gen_offer.values, (len(snaps), 1)),
            index=snaps, columns=gen_names,
        )
        bid_tv = pd.DataFrame(
            np.tile(gen_bid.values, (len(snaps), 1)),
            index=snaps, columns=gen_names,
        )

        # Overlay ELEXON per-snapshot data for matched generators
        for gen_name in gen_names:
            # FIX-2: overlay only generators directly matched on both sides
            if gen_name in network._bm_direct_matched_generators:
                offer_aligned = elexon_offers[gen_name].reindex(snaps)
                bid_aligned = elexon_bids[gen_name].reindex(snaps)
                # Fill gaps with the static mean
                offer_tv[gen_name] = offer_aligned.fillna(gen_offer[gen_name])
                # Negate ELEXON bids  ESO cost convention with no clipping.
                bid_tv[gen_name] = (-bid_aligned).fillna(
                    gen_bid[gen_name]
                )

        # Apply carrier-mean-normalised ELEXON hourly shape to generators that
        # got fitted-fallback prices, so unmatched gens get peak/offpeak
        # structure instead of a flat line.
        if fitted_enabled and len(fitted_applied_idx) > 0 \
                and fitted_cfg.get("time_varying_shape", True):
            from scripts.market.bm_elexon_fit import build_carrier_shape
            mapped_gen_names = set(bmu_to_gen.values())
            min_matched_gens = int(fitted_cfg.get("min_matched_gens", 3))
            offer_shape = build_carrier_shape(
                network.generators, elexon_offers,
                mapped_gen_names, min_matched_gens,
            )
            bid_shape = build_carrier_shape(
                network.generators, -elexon_bids,
                mapped_gen_names, min_matched_gens,
            )
            if not offer_shape.empty:
                offer_shape = offer_shape.reindex(snaps).fillna(1.0)
            if not bid_shape.empty:
                bid_shape = bid_shape.reindex(snaps).fillna(1.0)

            shaped_gens = 0
            for idx in fitted_applied_idx:
                carrier = network.generators.loc[idx, "carrier"]
                if not offer_shape.empty and carrier in offer_shape.columns:
                    offer_tv[idx] = gen_offer[idx] * offer_shape[carrier].values
                if not bid_shape.empty and carrier in bid_shape.columns:
                    bid_tv[idx] = gen_bid[idx] * bid_shape[carrier].values
                if (not offer_shape.empty and carrier in offer_shape.columns) \
                        or (not bid_shape.empty and carrier in bid_shape.columns):
                    shaped_gens += 1
            if shaped_gens > 0:
                logger.info(
                    f"Time-varying shape applied to {shaped_gens} fitted-fallback generators"
                )

        network._bm_offer_tv = offer_tv
        network._bm_bid_tv = bid_tv
        logger.info(
            f"Time-varying bid/offer prices: {offer_tv.shape} "
            f"(ELEXON per-snapshot for {matched} generators)"
        )

    # Storage units  default to derived pricing, but use carrier-level ELEXON
    # traces where configured (e.g. pumped storage BMUs in historical runs).
    su_offer = pd.Series(dtype=float)
    su_bid = pd.Series(dtype=float)
    if len(network.storage_units) > 0:
        su_offer, su_bid = _compute_component_prices(
            network.storage_units,
            carrier_overrides,
            default_offer_markup,
            default_bid_discount,
        )

        storage_prefix_map = elexon_cfg.get(
            "storage_bmu_prefixes", DEFAULT_STORAGE_BMU_PREFIXES
        )
        storage_price_aggregation = str(
            elexon_cfg.get("storage_price_aggregation", "median")
        ).lower()

        def _aggregate_storage_prices(df: pd.DataFrame, columns: list[str]) -> pd.Series:
            if storage_price_aggregation == "mean":
                return df[columns].mean(axis=1, skipna=True)
            return df[columns].median(axis=1, skipna=True)

        su_offer_tv = None
        su_bid_tv = None
        if time_varying and len(network.snapshots) > 0:
            snaps = network.snapshots
            su_offer_tv = pd.DataFrame(
                np.tile(su_offer.values, (len(snaps), 1)),
                index=snaps,
                columns=network.storage_units.index,
            )
            su_bid_tv = pd.DataFrame(
                np.tile(su_bid.values, (len(snaps), 1)),
                index=snaps,
                columns=network.storage_units.index,
            )

        matched_storage_carriers = []
        for carrier, prefixes in storage_prefix_map.items():
            su_names = network.storage_units.index[
                network.storage_units["carrier"] == carrier
            ]
            if len(su_names) == 0:
                continue

            prefix_list = prefixes if isinstance(prefixes, list) else [prefixes]
            matched_cols = [
                col for col in elexon_offers_raw.columns
                if any(str(col).startswith(prefix) for prefix in prefix_list)
            ]
            if not matched_cols:
                continue

            # FIX-2: a storage BMU can be missing from one side after the
            # sentinel filter; aggregate each side over the columns it has,
            # and keep the derived price for a side with no ELEXON data.
            offer_cols = [c for c in matched_cols if c in elexon_offers_raw.columns]
            bid_cols = [c for c in matched_cols if c in elexon_bids_raw.columns]
            offer_series = (_aggregate_storage_prices(elexon_offers_raw, offer_cols)
                            if offer_cols else pd.Series(dtype=float))
            bid_series = (-_aggregate_storage_prices(elexon_bids_raw, bid_cols)
                          if bid_cols else pd.Series(dtype=float))

            static_offer = (float(offer_series.mean()) if offer_series.notna().any()
                            else float(su_offer.loc[su_names].mean()))
            static_bid = (float(bid_series.mean()) if bid_series.notna().any()
                          else float(su_bid.loc[su_names].mean()))
            if not offer_series.notna().any():
                offer_series = pd.Series(static_offer, index=elexon_offers_raw.index)
            if not bid_series.notna().any():
                bid_series = pd.Series(static_bid, index=elexon_bids_raw.index)
            su_offer.loc[su_names] = static_offer
            su_bid.loc[su_names] = static_bid

            if su_offer_tv is not None and su_bid_tv is not None:
                offer_aligned = offer_series.reindex(snaps).fillna(static_offer)
                bid_aligned = bid_series.reindex(snaps).fillna(static_bid)
                for su_name in su_names:
                    su_offer_tv[su_name] = offer_aligned.values
                    su_bid_tv[su_name] = bid_aligned.values

            matched_storage_carriers.append(
                f"{carrier} ({len(matched_cols)} BMUs  {len(su_names)} units)"
            )

        if su_offer_tv is not None and su_bid_tv is not None:
            network._bm_su_offer_tv = su_offer_tv
            network._bm_su_bid_tv = su_bid_tv

        if matched_storage_carriers:
            logger.info(
                "ELEXON storage bid/offer applied for carriers: "
                + ", ".join(matched_storage_carriers)
            )
        else:
            logger.info(
                "No ELEXON storage carrier traces matched; storage falls back to derived pricing"
            )

    return gen_offer, gen_bid, su_offer, su_bid


def _load_csv_bid_offer(
    network, balancing: dict,
    logger: logging.Logger,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load bid/offer prices from user-supplied CSV files.

    CSV format: rows = snapshots (or a single row for static prices),
    columns = generator/storage unit names. Values in /MWh.
    If a single row, prices are broadcast to all snapshots.

    Parameters
    ----------
    network : pypsa.Network
    balancing : dict
        Balancing config with 'csv' sub-dict containing 'offer_file', 'bid_file'.
    logger : logging.Logger

    Returns
    -------
    gen_offer, gen_bid, su_offer, su_bid : pd.Series
    """
    csv_cfg = balancing.get("csv", {})
    offer_path = csv_cfg.get("offer_file")
    bid_path = csv_cfg.get("bid_file")

    if not offer_path or not bid_path:
        raise ValueError(
            "bid_offer_source='csv' requires both 'csv.offer_file' and 'csv.bid_file' "
            "to be set in market.balancing config."
        )

    offer_df = pd.read_csv(offer_path, index_col=0)
    bid_df = pd.read_csv(bid_path, index_col=0)
    logger.info(f"Loaded CSV offers: {offer_df.shape}, bids: {bid_df.shape}")

    # Use mean across snapshots if time-varying, or single row
    def _extract_prices(df, components, label):
        prices = pd.Series(index=components.index, dtype=float, data=0.0)
        for name in components.index:
            if name in df.columns:
                prices[name] = df[name].mean()
        return prices

    gen_offer = _extract_prices(offer_df, network.generators, "generator")
    gen_bid = _extract_prices(bid_df, network.generators, "generator")

    su_offer = pd.Series(dtype=float)
    su_bid = pd.Series(dtype=float)
    if len(network.storage_units) > 0:
        su_offer = _extract_prices(offer_df, network.storage_units, "storage_unit")
        su_bid = _extract_prices(bid_df, network.storage_units, "storage_unit")

    logger.info(
        f"CSV bid/offer loaded: {len(gen_offer)} generators, {len(su_offer)} storage units"
    )

    return gen_offer, gen_bid, su_offer, su_bid


# ── Hydrogen links in the balancing stage ────────────────────────────────
#
# Every CP30 electrolyser (204 links) and H2 turbine (209) meets on one
# copperplate GB_H2 bus. The balancing objective replaces PyPSA's, and only
# cross-border links were anchored, so Stage 2 moved electrolysis for nothing:
# 3.72 TWh up and 4.49 TWh down over the 2030 year, about 14% of gross GB
# redispatch, worth ~165 m a year at nodal prices (audit of 26 Sep 2026). The
# user decision is to price that movement at the hydrogen value Stage 1 implies:
# the GB_H2 bus dual, which the wholesale stage now writes as the
# ``hydrogen_price`` column of its price CSV. A link's owner values one more MW
# of p0 at ``c`` (its marginal cost of running): an electrolyser gains
# ``efficiency x lambda_H2`` of hydrogen, so ``c = mc - efficiency x lambda``; a
# turbine burns ``lambda_H2`` of it, so ``c = mc + lambda``. Turning p0 up is then
# priced ``c + markup x |c|`` and turning it down ``-c + discount x |c|`` (ESO-cost
# convention, the same shape as a derived generator offer and net-saving bid), so
# a shift of electrolysis within a day costs NESO the bid-offer spread rather
# than nothing. Defaults below must match config/defaults/
# market.balancing.hydrogen_links.
DEFAULT_HYDROGEN_LINK_CARRIERS = ("electrolysis", "H2_turbine")
DEFAULT_HYDROGEN_BUS_CARRIERS = ("H2_gas", "H2", "hydrogen")
HYDROGEN_PRICE_COLUMN = "hydrogen_price"


def resolve_hydrogen_link_config(balancing: dict) -> dict:
    """``market.balancing.hydrogen_links`` with defaults filled (markups from the balancing defaults)."""
    cfg = dict((balancing or {}).get("hydrogen_links", {}) or {})
    offer_markup = cfg.get("offer_markup")
    bid_discount = cfg.get("bid_discount")
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "carriers": tuple(cfg.get("carriers") or DEFAULT_HYDROGEN_LINK_CARRIERS),
        "bus_carriers": tuple(cfg.get("bus_carriers") or DEFAULT_HYDROGEN_BUS_CARRIERS),
        "offer_markup": float(
            (balancing or {}).get("default_offer_markup", 0.10) if offer_markup is None else offer_markup
        ),
        "bid_discount": float(
            (balancing or {}).get("default_bid_discount", 0.10) if bid_discount is None else bid_discount
        ),
    }


def hydrogen_bus_names(network, bus_carriers: Sequence[str] = DEFAULT_HYDROGEN_BUS_CARRIERS) -> pd.Index:
    """Buses carrying hydrogen (by bus carrier)."""
    if len(network.buses) == 0 or "carrier" not in network.buses.columns:
        return pd.Index([])
    wanted = {str(c).strip().lower() for c in bus_carriers}
    carriers = network.buses["carrier"].fillna("").astype(str).str.strip().str.lower()
    return network.buses.index[carriers.isin(wanted)]


def hydrogen_links(network, carriers: Sequence[str], bus_carriers: Sequence[str]) -> pd.Index:
    """Links of the given carriers that have exactly one hydrogen end."""
    if len(network.links) == 0:
        return pd.Index([])
    h2 = set(hydrogen_bus_names(network, bus_carriers))
    links = network.links
    one_end = links["bus0"].isin(h2) ^ links["bus1"].isin(h2)
    return links.index[links["carrier"].isin(set(carriers)) & one_end]


def wholesale_hydrogen_price(prices: pd.DataFrame, network,
                             bus_carriers: Sequence[str] = DEFAULT_HYDROGEN_BUS_CARRIERS) -> Optional[pd.Series]:
    """Hourly Stage 1 hydrogen value (GBP/MWh_H2): the mean dual over hydrogen buses.

    ``None`` when the network has no hydrogen bus with a price. The CP30 system
    has one bus (GB_H2), so the mean is that bus's dual.

    Only hydrogen buses that a link from a non-hydrogen bus feeds (the
    electrolysers' buses) count when there are any: the value electrolysers are
    priced at in the BM is the hydrogen they make. A bus that only supplies
    turbines with bought fuel (``GB_H2_power_fuel``, item OW3) has the fuel
    price as its dual and is left out. Networks without such a bus are unchanged.
    """
    if prices is None or len(prices) == 0:
        return None
    h2 = hydrogen_bus_names(network, bus_carriers)
    links = getattr(network, "links", None)
    if links is not None and len(links):
        fed = set(links.loc[links["bus1"].isin(h2) & ~links["bus0"].isin(h2), "bus1"])
        if fed:
            h2 = pd.Index([b for b in h2 if b in fed])
    columns = [b for b in h2 if b in prices.columns]
    if not columns:
        return None
    return prices[columns].mean(axis=1).rename(HYDROGEN_PRICE_COLUMN)


def estimate_hydrogen_value_from_schedule(
    electricity_price: pd.Series,
    wholesale_links: pd.DataFrame,
    network,
    link_names: Sequence[str],
    window_hours: int,
    logger: logging.Logger,
    bus_carriers: Sequence[str] = DEFAULT_HYDROGEN_BUS_CARRIERS,
) -> pd.Series:
    """Fallback hydrogen value for a Stage 1 run that predates the ``hydrogen_price`` column.

    Per day-ahead window: the electrolysis-weighted electricity price divided by
    the electrolysers' capacity-weighted efficiency, i.e. what the window's
    hydrogen cost to make on average. It understates the dual when the marginal
    production hour is dearer than the average one, so a re-run of Stage 1 is
    preferred; the caller warns.
    """
    snapshots = pd.Index(electricity_price.index)
    links = network.links.loc[[l for l in link_names if l in network.links.index]]
    h2 = set(hydrogen_bus_names(network, bus_carriers))
    electrolysers = [l for l in links.index if links.at[l, "bus1"] in h2 and links.at[l, "bus0"] not in h2]
    electrolysers = [l for l in electrolysers if l in wholesale_links.columns]
    value = pd.Series(np.nan, index=snapshots, name=HYDROGEN_PRICE_COLUMN)
    if not electrolysers:
        return value.fillna(0.0)
    p_nom = pd.to_numeric(links.loc[electrolysers, "p_nom"], errors="coerce").fillna(0.0)
    eff = pd.to_numeric(links.loc[electrolysers, "efficiency"], errors="coerce").fillna(1.0)
    eta = float((p_nom * eff).sum() / p_nom.sum()) if p_nom.sum() > 0 else float(eff.mean())
    draw = wholesale_links[electrolysers].reindex(snapshots).fillna(0.0).clip(lower=0.0).sum(axis=1)
    steps = max(1, int(window_hours))
    overall = (
        float((draw * electricity_price).sum() / draw.sum()) if draw.sum() > 0 else float(electricity_price.mean())
    )
    for start in range(0, len(snapshots), steps):
        window = snapshots[start:start + steps]
        w = draw.loc[window]
        mean_price = float((w * electricity_price.loc[window]).sum() / w.sum()) if w.sum() > 1e-9 else overall
        value.loc[window] = max(mean_price, 0.0) / max(eta, 1e-6)
    logger.warning(
        "Stage 1 price CSV has no hydrogen_price column; hydrogen links in the balancing stage "
        f"are priced on an estimate (electrolysis-weighted electricity price / {eta:.2f} per "
        f"{steps} h window, mean {value.mean():.2f} GBP/MWh_H2). Re-run Stage 1 for the bus dual."
    )
    return value


def hydrogen_link_prices(
    network,
    link_names: Sequence[str],
    hydrogen_price: pd.Series,
    offer_markup: float,
    bid_discount: float,
    bus_carriers: Sequence[str] = DEFAULT_HYDROGEN_BUS_CARRIERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-snapshot prices (GBP per MWh of p0) for turning each hydrogen link up and down.

    Returns ``(increase, decrease)`` frames in ESO-cost convention, snapshots x
    links. See the section comment above for the formula.
    """
    snapshots = pd.Index(hydrogen_price.index)
    names = [l for l in link_names if l in network.links.index]
    h2 = set(hydrogen_bus_names(network, bus_carriers))
    links = network.links.loc[names]
    mc = pd.to_numeric(links["marginal_cost"], errors="coerce").fillna(0.0)
    eff = pd.to_numeric(links["efficiency"], errors="coerce").fillna(1.0)
    # +1 for a link that burns hydrogen at bus0, -efficiency for one that makes it at bus1.
    weight = pd.Series(
        np.where(links["bus0"].isin(h2), 1.0, np.where(links["bus1"].isin(h2), -eff, 0.0)),
        index=names,
    )
    lam = pd.to_numeric(hydrogen_price, errors="coerce").fillna(0.0).to_numpy()[:, None]
    cost = mc.to_numpy()[None, :] + weight.to_numpy()[None, :] * lam
    increase = cost + float(offer_markup) * np.abs(cost)
    decrease = -cost + float(bid_discount) * np.abs(cost)
    return (
        pd.DataFrame(increase, index=snapshots, columns=names),
        pd.DataFrame(decrease, index=snapshots, columns=names),
    )


def compute_link_redispatch_volumes(
    wholesale_links: pd.DataFrame,
    physical_links: pd.DataFrame,
    increase_prices: pd.DataFrame,
    decrease_prices: pd.DataFrame,
    network,
    logger: logging.Logger,
    bus_carriers: Sequence[str] = DEFAULT_HYDROGEN_BUS_CARRIERS,
) -> pd.DataFrame:
    """Redispatch rows for anchored hydrogen links, in electrical-injection terms.

    Same columns as ``compute_redispatch_volumes`` (``type`` = ``link``). A p0
    move is costed at the prices the balancing LP used; volumes are MWh at the
    electrical end. An electrolyser drawing more takes power off the system, so
    it is booked as a decrease (with its cost in ``bid_cost``) and drawing less
    as an increase (``offer_cost``); a turbine's p0 is hydrogen in, so its
    electrical move is ``efficiency x`` the p0 move.
    """
    names = [l for l in increase_prices.columns
             if l in physical_links.columns and l in network.links.index]
    columns = ["component", "type", "carrier", "increase_MWh", "decrease_MWh",
               "offer_cost", "bid_cost", "net_cost"]
    if not names or physical_links.empty:
        return pd.DataFrame(columns=columns)
    index = physical_links.index
    hours = pd.Series(1.0, index=index)
    if len(index) > 1:
        spacing = index.to_series().diff().dt.total_seconds().median() / 3600.0
        if np.isfinite(spacing) and spacing > 0:
            hours[:] = spacing
    h2 = set(hydrogen_bus_names(network, bus_carriers))
    schedule = wholesale_links.reindex(index=index, columns=names).fillna(0.0)
    moved = physical_links[names] - schedule
    up = moved.clip(lower=0.0).mul(hours, axis=0)
    down = (-moved).clip(lower=0.0).mul(hours, axis=0)
    up_cost = (up * increase_prices.reindex(index=index, columns=names).fillna(0.0)).sum()
    down_cost = (down * decrease_prices.reindex(index=index, columns=names).fillna(0.0)).sum()
    records = []
    for name in names:
        link = network.links.loc[name]
        if link["bus1"] in h2:  # electrolyser: p0 is electricity drawn
            record = dict(increase_MWh=float(down[name].sum()), decrease_MWh=float(up[name].sum()),
                          offer_cost=float(down_cost[name]), bid_cost=float(up_cost[name]))
        else:  # turbine: p0 is hydrogen burned, efficiency x p0 is electricity
            eff = float(pd.to_numeric(pd.Series([link.get("efficiency", 1.0)]), errors="coerce").fillna(1.0).iloc[0])
            record = dict(increase_MWh=eff * float(up[name].sum()), decrease_MWh=eff * float(down[name].sum()),
                          offer_cost=float(up_cost[name]), bid_cost=float(down_cost[name]))
        record.update(component=name, type="link", carrier=link.get("carrier", "unknown"))
        record["net_cost"] = record["offer_cost"] + record["bid_cost"]
        records.append(record)
    summary = pd.DataFrame(records, columns=columns)
    logger.info(
        f"Hydrogen link redispatch: {len(summary)} links, increase "
        f"{summary['increase_MWh'].sum():,.0f} MWh, decrease {summary['decrease_MWh'].sum():,.0f} MWh "
        f"(electrical), cost {summary['net_cost'].sum():,.0f}"
    )
    return summary


def extract_wholesale_positions(
    network,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract dispatch positions from a solved wholesale (copperplate) network.

    Parameters
    ----------
    network : pypsa.Network
        Solved wholesale network (after network.optimize()).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    gen_dispatch : pd.DataFrame
        Generator dispatch (MW) indexed by snapshot, columns = generator names.
    su_dispatch : pd.DataFrame
        Storage unit dispatch (MW) indexed by snapshot, columns = storage unit names.
    link_dispatch : pd.DataFrame
        Link dispatch at bus0 (MW) indexed by snapshot, columns = link names.
    """
    gen_dispatch = network.generators_t.p.copy()
    logger.info(f"Extracted wholesale generator dispatch: {gen_dispatch.shape}")

    if len(network.storage_units_t.p) > 0:
        su_dispatch = network.storage_units_t.p.copy()
        logger.info(f"Extracted wholesale storage dispatch: {su_dispatch.shape}")
    else:
        su_dispatch = pd.DataFrame(index=network.snapshots)

    if len(network.links_t.p0) > 0:
        link_dispatch = network.links_t.p0.copy()
        logger.info(f"Extracted wholesale link dispatch: {link_dispatch.shape}")
    else:
        link_dispatch = pd.DataFrame(index=network.snapshots)

    return gen_dispatch, su_dispatch, link_dispatch


def compute_redispatch_volumes(
    wholesale_gen: pd.DataFrame,
    physical_gen: pd.DataFrame,
    wholesale_su: pd.DataFrame,
    physical_su: pd.DataFrame,
    gen_offer_prices: pd.Series,
    gen_bid_prices: pd.Series,
    su_offer_prices: pd.Series,
    su_bid_prices: pd.Series,
    network,
    logger: logging.Logger,
    gen_offer_prices_tv: pd.DataFrame | None = None,
    gen_bid_prices_tv: pd.DataFrame | None = None,
    su_offer_prices_tv: pd.DataFrame | None = None,
    su_bid_prices_tv: pd.DataFrame | None = None,
    gen_offer_ladders: pd.DataFrame | None = None,
    gen_bid_ladders: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-asset redispatch volumes and costs.

    Increase (offer accepted) = physical - wholesale > 0   paid offer_price
    Decrease (bid accepted)   = wholesale - physical > 0   paid bid_price

    Parameters
    ----------
    wholesale_gen, physical_gen : pd.DataFrame
        Generator dispatch (MW), snapshots  generators.
    wholesale_su, physical_su : pd.DataFrame
        Storage unit dispatch (MW), snapshots  storage units.
    gen_offer_prices, gen_bid_prices : pd.Series
        Per-generator offer/bid prices (/MWh).
    su_offer_prices, su_bid_prices : pd.Series
        Per-storage-unit offer/bid prices (/MWh).
    network : pypsa.Network
        Solved network (for carrier lookup).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    gen_summary : pd.DataFrame
        Per-generator summary with columns: carrier, increase_MWh, decrease_MWh,
        offer_cost, bid_cost, net_cost.
    su_summary : pd.DataFrame
        Per-storage-unit summary with same columns.
    """

    def _snapshot_hours(index: pd.Index) -> pd.Series:
        """Return per-snapshot durations in hours for MW-to-MWh conversion."""
        if len(index) == 0:
            return pd.Series(dtype=float, index=index)
        if len(index) == 1:
            return pd.Series(1.0, index=index, dtype=float)

        dt = pd.Series(index=index, data=index.to_series().diff().dt.total_seconds() / 3600)
        inferred = dt.dropna().median()
        if not np.isfinite(inferred) or inferred <= 0:
            inferred = 1.0
        dt.iloc[0] = inferred
        dt = dt.fillna(inferred).clip(lower=0.0)
        return dt.astype(float)

    def _ladder_cost(
        volume: float,
        snapshot,
        component: str,
        ladder_lookup: dict | None,
        fallback_price: float,
    ) -> float:
        if volume <= 0 or not ladder_lookup:
            return volume * fallback_price
        blocks = ladder_lookup.get((snapshot, component))
        if blocks is None or len(blocks) == 0:
            return volume * fallback_price

        remaining = float(volume)
        cost = 0.0
        for price, volume_mw in blocks:
            take = min(remaining, float(volume_mw))
            if take > 0:
                cost += take * float(price)
                remaining -= take
            if remaining <= 1.0e-9:
                break
        if remaining > 1.0e-9:
            cost += remaining * fallback_price
        return cost

    def _prepare_ladder_lookup(ladders: pd.DataFrame | None) -> dict:
        if ladders is None or ladders.empty:
            return {}
        lookup = {}
        for key, group in ladders.groupby(["snapshot", "generator"], sort=False):
            blocks = (
                group.sort_values("price")[["price", "volume_mw"]]
                .to_numpy(dtype=float)
            )
            lookup[key] = blocks
        return lookup

    def _summarize(
        wholesale_df,
        physical_df,
        offer_prices,
        bid_prices,
        components,
        component_type,
        offer_prices_tv=None,
        bid_prices_tv=None,
        offer_ladders=None,
        bid_ladders=None,
    ):
        records = []
        # Align columns
        common = wholesale_df.columns.intersection(physical_df.columns)
        snapshot_hours = _snapshot_hours(wholesale_df.index)

        offer_prices_tv_aligned = None
        bid_prices_tv_aligned = None
        if offer_prices_tv is not None:
            offer_prices_tv_aligned = offer_prices_tv.reindex(
                index=wholesale_df.index, columns=common
            )
        if bid_prices_tv is not None:
            bid_prices_tv_aligned = bid_prices_tv.reindex(
                index=wholesale_df.index, columns=common
            )
        offer_ladder_lookup = _prepare_ladder_lookup(offer_ladders)
        bid_ladder_lookup = _prepare_ladder_lookup(bid_ladders)

        for name in common:
            diff = physical_df[name] - wholesale_df[name]  # MW per snapshot
            increase_mw = diff.clip(lower=0)
            decrease_mw = (-diff).clip(lower=0)
            increase_series = increase_mw * snapshot_hours  # MWh per snapshot
            decrease_series = decrease_mw * snapshot_hours
            increase = increase_series.sum()
            decrease = decrease_series.sum()

            if offer_ladder_lookup:
                if offer_prices_tv_aligned is not None and name in offer_prices_tv_aligned:
                    fallback_offer = offer_prices_tv_aligned[name].fillna(
                        offer_prices.get(name, 0.0)
                    )
                else:
                    fallback_offer = pd.Series(
                        offer_prices.get(name, 0.0), index=increase_series.index
                    )
                offer_cost = sum(
                    _ladder_cost(
                        mw, snap, name, offer_ladder_lookup, fallback_offer.loc[snap]
                    )
                    * snapshot_hours.loc[snap]
                    for snap, mw in increase_mw.items()
                )
            elif offer_prices_tv_aligned is not None and name in offer_prices_tv_aligned:
                offer_cost = (
                    increase_series * offer_prices_tv_aligned[name].fillna(0.0)
                ).sum()
            else:
                offer_cost = increase * offer_prices.get(name, 0.0)

            if bid_ladder_lookup:
                if bid_prices_tv_aligned is not None and name in bid_prices_tv_aligned:
                    fallback_bid = bid_prices_tv_aligned[name].fillna(
                        bid_prices.get(name, 0.0)
                    )
                else:
                    fallback_bid = pd.Series(
                        bid_prices.get(name, 0.0), index=decrease_series.index
                    )
                bid_cost = sum(
                    _ladder_cost(
                        mw, snap, name, bid_ladder_lookup, fallback_bid.loc[snap]
                    )
                    * snapshot_hours.loc[snap]
                    for snap, mw in decrease_mw.items()
                )
            elif bid_prices_tv_aligned is not None and name in bid_prices_tv_aligned:
                bid_cost = (
                    decrease_series * bid_prices_tv_aligned[name].fillna(0.0)
                ).sum()
            else:
                bid_cost = decrease * bid_prices.get(name, 0.0)

            records.append({
                "component": name,
                "type": component_type,
                "carrier": components.loc[name, "carrier"] if name in components.index else "unknown",
                # FIX-1: origin of the BM price ("" when not recorded)
                "price_source": (
                    str(getattr(network, "_bm_price_source", {}).get(name, ""))
                    if component_type == "generator" else "storage"
                ),
                "increase_MWh": increase,
                "decrease_MWh": decrease,
                "offer_cost": offer_cost,
                "bid_cost": bid_cost,
                "net_cost": offer_cost + bid_cost,
            })
        return pd.DataFrame(records)

    gen_summary = _summarize(
        wholesale_gen, physical_gen, gen_offer_prices, gen_bid_prices,
        network.generators, "generator",
        offer_prices_tv=gen_offer_prices_tv,
        bid_prices_tv=gen_bid_prices_tv,
        offer_ladders=gen_offer_ladders,
        bid_ladders=gen_bid_ladders,
    )
    logger.info(
        f"Generator redispatch: "
        f"total increase={gen_summary['increase_MWh'].sum():,.0f} MWh, "
        f"total decrease={gen_summary['decrease_MWh'].sum():,.0f} MWh, "
        f"total cost={gen_summary['net_cost'].sum():,.0f}"
    )

    if len(wholesale_su.columns) > 0 and len(physical_su.columns) > 0:
        su_summary = _summarize(
            wholesale_su, physical_su, su_offer_prices, su_bid_prices,
            network.storage_units, "storage_unit",
            offer_prices_tv=su_offer_prices_tv,
            bid_prices_tv=su_bid_prices_tv,
        )
        logger.info(
            f"Storage redispatch: "
            f"total increase={su_summary['increase_MWh'].sum():,.0f} MWh, "
            f"total decrease={su_summary['decrease_MWh'].sum():,.0f} MWh, "
            f"total cost={su_summary['net_cost'].sum():,.0f}"
        )
    else:
        su_summary = pd.DataFrame()

    return gen_summary, su_summary


def identify_congested_boundaries(
    network,
    threshold: float = 0.95,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Identify congested lines and transformers in a solved network.

    A line/transformer is considered congested at a snapshot if its loading
    exceeds ``threshold`` fraction of its ``s_nom``.

    Parameters
    ----------
    network : pypsa.Network
        Solved network with lines_t.p0 populated.
    threshold : float
        Fraction of s_nom above which a component is congested (default 0.95).
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    congestion_df : pd.DataFrame
        Columns: component, type, s_nom, max_loading_fraction,
        hours_congested, mean_loading_fraction.
    """
    records = []

    # Lines
    if len(network.lines_t.p0) > 0 and len(network.lines) > 0:
        for line in network.lines.index:
            if line not in network.lines_t.p0.columns:
                continue
            s_nom = network.lines.loc[line, "s_nom"]
            if s_nom <= 0:
                continue
            loading = network.lines_t.p0[line].abs() / s_nom
            hours_congested = (loading >= threshold).sum()
            if hours_congested > 0:
                records.append({
                    "component": line,
                    "type": "line",
                    "s_nom_MVA": s_nom,
                    "max_loading_fraction": loading.max(),
                    "hours_congested": int(hours_congested),
                    "mean_loading_fraction": loading.mean(),
                })

    # Transformers
    if len(network.transformers_t.p0) > 0 and len(network.transformers) > 0:
        for trafo in network.transformers.index:
            if trafo not in network.transformers_t.p0.columns:
                continue
            s_nom = network.transformers.loc[trafo, "s_nom"]
            if s_nom <= 0:
                continue
            loading = network.transformers_t.p0[trafo].abs() / s_nom
            hours_congested = (loading >= threshold).sum()
            if hours_congested > 0:
                records.append({
                    "component": trafo,
                    "type": "transformer",
                    "s_nom_MVA": s_nom,
                    "max_loading_fraction": loading.max(),
                    "hours_congested": int(hours_congested),
                    "mean_loading_fraction": loading.mean(),
                })

    congestion_df = pd.DataFrame(records)
    if logger:
        if len(congestion_df) > 0:
            n_lines = (congestion_df["type"] == "line").sum()
            n_trafos = (congestion_df["type"] == "transformer").sum()
            logger.info(
                f"Congestion analysis: {n_lines} congested lines, "
                f"{n_trafos} congested transformers (threshold={threshold:.0%})"
            )
        else:
            logger.info(f"No congested components found (threshold={threshold:.0%})")

    return congestion_df
