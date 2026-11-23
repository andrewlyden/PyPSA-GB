"""A linear proxy for unit commitment in the real-time stage.

Why
---
In this model any thermal unit can move from nothing to full output within the hour
at its marginal cost, so correcting a forecast error is symmetric: being short costs
the fuel of the next unit, being long saves it. Nothing makes being caught out
expensive, and a symmetric error nets to roughly nothing whatever its size. That is
why the year's real-time correction came out *negative*, and why this model cannot
by itself reproduce the forecast-error cost FTI identify as the main consumer cost of
central dispatch. FTI attribute it to plant with longer lead times, which is a
commitment effect, and a mixed-integer commitment over 13,981 generators is out of
reach at year scale.

What
----
Two data edits before the real-time solve, both keyed on the day-ahead schedule:

1. **Start premium.** A thermal unit the schedule left *off* in an hour can still be
   brought on in real time, but its offer carries a premium for the unplanned start.
2. **Committed floor.** A thermal unit the schedule ran at or above its minimum
   stable level cannot be taken below that level in real time; the operator must
   spill renewables instead. A unit the schedule ran below minimum stable is treated
   as uncommitted, because a linear schedule cannot represent a unit that is on but
   below its floor.

Being short now means paying for unplanned starts; being long means keeping
committed plant running while spilling free energy. That is the asymmetry a real
operator faces, and it applies to both institutions' real-time stages alike, so it
does not tilt the comparison: the central case simply has far more forecast error
to correct because, following FTI, self-dispatch clears its error through intra-day
trading first.

Config: ``market.balancing.commitment_proxy`` in ``config/defaults/``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_CARRIERS = [
    "CCGT", "OCGT", "gas_engine", "coal", "oil", "biomass", "CHP", "waste_to_energy", "CCS Biomass",
    "biogas", "landfill_gas", "sewage_gas", "advanced_biofuel",
]
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "carriers": DEFAULT_CARRIERS,
    "min_stable_fraction": 0.4,
    "start_premium_gbp_mwh": 50.0,
    "committed_threshold_mw": 1.0,
}

_LOG = logging.getLogger(__name__)


def _broadcast(static: pd.Series, snapshots: pd.Index, columns: pd.Index) -> pd.DataFrame:
    vals = static.reindex(columns).fillna(0.0).to_numpy(dtype=float)
    return pd.DataFrame(np.tile(vals, (len(snapshots), 1)), index=snapshots, columns=columns)


def apply_commitment_proxy(
    network,
    wholesale_gen: pd.DataFrame,
    gen_offer: pd.Series,
    gen_bid: pd.Series,
    gen_offer_tv: pd.DataFrame | None,
    gen_bid_tv: pd.DataFrame | None,
    cfg: Mapping[str, Any] | None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    """Apply the proxy in place on ``network`` and return the offer and bid frames to use.

    Returns the frames unchanged (possibly ``None``) and an empty summary when the
    proxy is disabled. When enabled and the frames are ``None`` they are created by
    broadcasting the static prices over every generator, because the real-time stage
    treats a missing column as a zero price.
    """
    log = logger or _LOG
    c = dict(DEFAULTS)
    c.update(cfg or {})
    if not c.get("enabled", False):
        return gen_offer_tv, gen_bid_tv, {}

    gens = network.generators
    snapshots = pd.DatetimeIndex(network.snapshots)
    carriers = set(c["carriers"])
    units = [g for g in gens.index if gens.at[g, "carrier"] in carriers and g in wholesale_gen.columns]
    if not units:
        log.info("Commitment proxy: no eligible units in the schedule; nothing applied")
        return gen_offer_tv, gen_bid_tv, {}

    msf = float(c["min_stable_fraction"])
    premium = float(c["start_premium_gbp_mwh"])
    thr = float(c["committed_threshold_mw"])
    sched = wholesale_gen[units].reindex(snapshots).fillna(0.0)
    p_nom = gens.loc[units, "p_nom"].astype(float)
    static_min = gens.loc[units, "p_min_pu"].astype(float) if "p_min_pu" in gens.columns else pd.Series(0.0, index=units)

    committed = sched.ge(thr)
    at_or_above_msg = sched.ge(p_nom * msf, axis=1)

    # 1. committed floor as a time-varying p_min_pu, never below the unit's own
    # effective floor: its hourly p_min_pu where it has one (seasonal CHP minimum
    # output, 26 Sep 2026), else its static floor. The hourly floor replaces the
    # static one rather than being clipped to it: a summer CHP floor of ~0.18
    # must not be lifted back to the annual 0.3 (review 26 Sep 2026).
    floor = pd.DataFrame(np.where(committed & at_or_above_msg, msf, 0.0), index=snapshots, columns=units)
    tv_min = network.generators_t.p_min_pu
    if tv_min is None or tv_min.empty:
        tv_min = pd.DataFrame(index=snapshots)
    else:
        tv_min = tv_min.reindex(snapshots)
    base = tv_min.reindex(columns=units).fillna(_broadcast(static_min, snapshots, pd.Index(units)))
    floor = floor.where(floor >= base, base)
    for u in units:
        tv_min[u] = floor[u].values
    network.generators_t["p_min_pu"] = tv_min

    # 2. start premium on the offers of units the schedule left off
    if gen_offer_tv is None or gen_bid_tv is None:
        gen_offer_tv = _broadcast(gen_offer, snapshots, gens.index)
        gen_bid_tv = _broadcast(gen_bid, snapshots, gens.index)
        created = True
    else:
        gen_offer_tv = gen_offer_tv.reindex(index=snapshots)
        gen_bid_tv = gen_bid_tv.reindex(index=snapshots)
        for u in units:
            if u not in gen_offer_tv.columns:
                gen_offer_tv[u] = float(gen_offer.get(u, 0.0))
            if u not in gen_bid_tv.columns:
                gen_bid_tv[u] = float(gen_bid.get(u, 0.0))
        created = False
    uncommitted = ~committed
    gen_offer_tv.loc[:, units] = gen_offer_tv[units].values + premium * uncommitted.values.astype(float)

    unit_hours = committed.size
    summary = {
        "units": len(units),
        "share_unit_hours_uncommitted": float(uncommitted.values.mean()),
        "share_unit_hours_floored": float((committed & at_or_above_msg).values.mean()),
        "min_stable_fraction": msf,
        "start_premium_gbp_mwh": premium,
        "frames_created": created,
    }
    log.info("=" * 80)
    log.info("COMMITMENT PROXY: START PREMIUM ON UNSCHEDULED UNITS, FLOOR ON SCHEDULED ONES")
    log.info("=" * 80)
    log.info(
        f"  {len(units)} thermal units, {unit_hours:,} unit-hours: "
        f"{100*summary['share_unit_hours_uncommitted']:.1f}% uncommitted (offer +{premium:.0f}/MWh), "
        f"{100*summary['share_unit_hours_floored']:.1f}% floored at {msf:.0%} of capacity"
        + ("; time-varying price frames created from static prices" if created else "")
    )
    return gen_offer_tv, gen_bid_tv, summary
