#!/usr/bin/env python3
"""Pro rata zonal turn-down re-pricing of the balancing stage's wind bids (user decision, 20 Oct 2026).

**Why.** The balancing LP turns wind down cheapest first: inside a congested zone it takes the units
whose bids are lowest (CfD contract value in a high-price hour, merchant at 0) and leaves the
expensive ones (ROC at banding x buyout, high-strike CfDs) running. NESO's CP30 constraint model
(CP.08) shares a zone's turn-down across its wind fleet, and BM practice spreads acceptances over
many BMUs. So the headline constraint cost re-prices the model's own turn-down as NESO would share it:

* For each BM month, each hour and each NESO CP.08 zone (17 zones, :mod:`scripts.market.neso_zones`
  on the month's own network), the turn-down of the allocation's technologies (``wind_onshore`` and
  ``wind_offshore``; decrease = Stage 1 p - BM p, clipped at 0) is re-shared among the zone's units of
  those technologies in proportion to each unit's Stage 1 output in that hour. The zone-hour volume
  is unchanged; only which units take it changes. A unit on a bus with no zone keeps its own volume.
* Each unit's turn-down is priced at its own bid in that hour, exactly as the BM priced it, rebuilt
  from the month's ``{scenario}_contract_value_bids.csv`` (CfD: the contract value on the Stage 1
  reference with the audit's basis, floor and negative-price rule, blended with the state share as
  the BM blended it; merchant and state-at-cost rows: their bid) and the network (ROC: banding x
  buyout x multiplier). The family's own bids, so the P2/P3 bid families are re-priced on theirs.
  A long-range (2040/2050) unit bids from the support label table (audit ``pricing`` ``support_labels_2050``):
  the hourly CfD part of its cohorts is rebuilt from the table exactly as ``support_bids`` built it (since the
  11 Nov 2026 alignment it is the contract value, so it moves hour by hour), plus the unit's time-invariant
  merchant, static, DPA, state and RO parts taken from the audit's mean bid (13 Nov 2026).
* Check, every month: re-pricing the model's OWN turn-down with the rebuilt bids reproduces the
  solve's wind bid cost (``{scenario}_redispatch_summary.csv``) to ``check_tolerance_rel`` (0.1%) of
  the gross bid money, or the month fails.

Outputs per month: ``{scenario}_pro_rata_turn_down.csv`` (one row per unit: zone, support, state share,
our MWh and cost, pro rata MWh and cost) and ``{scenario}_pro_rata_turn_down.json`` (totals, uplift,
check). Per ``monthly_aggregates`` study: ``{study}_pro_rata_turn_down_summary.json`` with
``neso_comparable_redispatch_cost_pro_rata_GBP`` = the NESO-comparable cost + (pro rata - our) wind
bid cost, beside the as-solved figure, and ``headline_*`` per the mode in force.

Config ``market.balancing.turn_down_allocation`` (read by the re-pricing and bill rules only, never by
the solves): ``mode`` ``pro_rata_zonal`` (the headline) or ``cheapest_first`` (as solved);
``technologies``; ``zone_basis`` ``neso_cp08``. :func:`allocation_in_force` is the one helper the bill
and ownership layers use to read per-unit turn-down volumes and bid costs under the mode in force.

    python scripts/market/turn_down_allocation.py --scenario CP30_FFR_2030_Jan_no_acceleration
    python scripts/market/turn_down_allocation.py --study CP30_2030_no_acceleration [--months]
"""

# No `from __future__` import: Snakemake's script directive prepends a preamble.

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

#: The two bases every month and family is reported on.
BASES = ("cheapest_first", "pro_rata_zonal")
#: ``mode``: a basis, or ``closest_to_neso`` (user, 20 Oct 2026): the basis whose NESO-comparable cost is
#: nearer NESO's published cost for the family's network (3,570 central, 7,790 no_acceleration, 10,240
#: seven_late, 12,700 no_upgrades; ``neso_balancing_benchmarks.csv``).
MODES = ("closest_to_neso", *BASES)
ZONE_BASES = ("neso_cp08",)
#: Code defaults of ``market.balancing.turn_down_allocation`` (config/defaults/ must match;
#: tests/unit/test_turn_down_allocation.py).
DEFAULT_CONFIG: dict = {
    "mode": "closest_to_neso",
    "technologies": ["wind_onshore", "wind_offshore"],
    "zone_basis": "neso_cp08",
    "check_tolerance_rel": 0.001,
    "check_tolerance_abs_gbp": 1000.0,
}
UNIT_COLUMNS = ["unit", "carrier", "bus", "zone", "support_type", "pricing", "state_share", "stage1_MWh",
                "our_MWh", "pro_rata_MWh", "our_cost_GBP", "pro_rata_cost_GBP", "solve_bid_cost_GBP",
                "solve_decrease_MWh"]


def allocation_config(block: dict | None) -> dict:
    """``market.balancing.turn_down_allocation`` over :data:`DEFAULT_CONFIG`, validated."""
    block = dict(block or {})
    if unknown := sorted(set(block) - set(DEFAULT_CONFIG)):
        raise ValueError(f"turn_down_allocation has unknown keys {unknown}; supported: {sorted(DEFAULT_CONFIG)}")
    cfg = {**copy.deepcopy(DEFAULT_CONFIG), **block}
    if cfg["mode"] not in MODES:
        raise ValueError(f"turn_down_allocation.mode must be one of {MODES}, got {cfg['mode']!r}")
    if cfg["zone_basis"] not in ZONE_BASES:
        raise ValueError(f"turn_down_allocation.zone_basis must be one of {ZONE_BASES}, got {cfg['zone_basis']!r}")
    cfg["technologies"] = [str(t) for t in (cfg["technologies"] or [])]
    if not cfg["technologies"]:
        raise ValueError("turn_down_allocation.technologies is empty")
    return cfg


def scenario_allocation_config(scenario: dict | None) -> dict:
    """The block from a scenario's merged config (absent: the code defaults)."""
    block = (((scenario or {}).get("market") or {}).get("balancing") or {}).get("turn_down_allocation")
    return allocation_config(block)


def roc_bid_rate(scenario: dict | None) -> float | None:
    """GBP/MWh per unit of ROC banding the BM bid (buyout x ``roc.bid_multiplier``), or None."""
    from scripts.generators.apply_marginal_costs import _get_roc_buyout_price

    scenario = scenario or {}
    subs = ((scenario.get("support") or {}).get("dispatch_costs") or {})
    buyout = _get_roc_buyout_price(subs.get("roc_buyout_prices", {}) or {}, int(scenario.get("modelled_year", 2024)))
    rsp = (((scenario.get("support") or {}).get("bm_bids")) or {})
    mult = float(((rsp.get("roc") or {}).get("bid_multiplier", 1.0)))
    return float(buyout) * mult if buyout else None


# ═══════════════════════════════════════════════════════════════════════════
# Pure pieces (the tests drive these directly)
# ═══════════════════════════════════════════════════════════════════════════
def pro_rata_turn_down(stage1: np.ndarray, decrease: np.ndarray, groups) -> np.ndarray:
    """Re-share each group's hourly turn-down pro rata to Stage 1 output (snapshots x units arrays).

    ``groups``: one label per unit; a missing label (None/NaN) keeps the unit's own volume. The group
    volume is conserved in every hour; a unit never takes more than its Stage 1 output.
    """
    stage1 = np.clip(np.asarray(stage1, dtype=float), 0.0, None)
    decrease = np.asarray(decrease, dtype=float)
    labels = pd.Series(list(groups), dtype=object)
    out = decrease.copy()
    for g in labels.dropna().unique():
        k = (labels == g).to_numpy()
        need = decrease[:, k].sum(axis=1)
        cap = stage1[:, k]
        total = cap.sum(axis=1)
        share = np.divide(cap, total[:, None], out=np.zeros_like(cap), where=total[:, None] > 0)
        moved = share * need[:, None]
        # An hour with turn-down but no Stage 1 output in the group cannot be re-shared: keep it as solved.
        stuck = (total <= 0) & (need > 0)
        if stuck.any():
            moved[stuck] = decrease[:, k][stuck]
        out[:, k] = moved
    return out


#: The audit ``pricing`` tag of a unit priced from the support label table (``support_bids._audit``).
LABEL_PRICING = "support_labels_2050"


def label_cfd_bids(names: pd.Index, audit: pd.DataFrame, labels: pd.DataFrame,
                   reference: pd.Series) -> pd.DataFrame:
    """Hourly CfD part (GBP/MWh) of each label-priced unit's bid, as ``support_bids.build_label_bid_frames`` built it.

    ``audit``: the month's audit indexed by generator (its ``state_share``, ``cfd_weight``, ``bid_basis``,
    ``negative_price_rule``). ``labels``: the build's support label table. The state's share comes out of
    each unit's post-2025 cohorts pro rata (item BM1), as in the solve.

    Raises:
        ValueError: a unit's rebuilt CfD weight differs from the audit's (the label table or the state
            carriers are not the ones the month bid on), or a wind cohort references the baseload price.
    """
    from scripts.market.cfd_bm_bids import DEFAULT_MERCHANT_CARRIERS
    from scripts.market.support_bids import POLICY_START_YEAR, _live, cfd_bid_profiles

    a = audit.reindex(names)
    lab = labels.loc[labels["component_type"].astype(str).eq("Generator") & labels["component"].isin(names)]
    weight = pd.to_numeric(lab["weight"], errors="coerce").fillna(0.0)
    year = pd.to_numeric(lab["build_year"], errors="coerce")
    state = pd.to_numeric(a["state_share"], errors="coerce").fillna(0.0)
    if (state > 0).any():
        eligible = lab["carrier"].isin(set(DEFAULT_MERCHANT_CARRIERS) | {"marine"})
        post_row = year.ge(POLICY_START_YEAR) & eligible
        post = weight.where(post_row, 0.0).groupby(lab["component"]).sum()
        taken = state.reindex(post.index).fillna(0.0)
        scale = (1.0 - taken / post.where(post > 0, 1.0)).where(post > 0, 1.0)
        weight = weight * lab["component"].map(scale).where(post_row, 1.0)
    is_cfd = lab["support_type"].eq("cfd")
    cfd_w = weight.where(is_cfd, 0.0) * _live(lab)
    keys = ["strike_gbp_per_mwh_2024", "negative_price_rule", "reference_price"]
    rows = lab.loc[is_cfd, keys].fillna({"negative_price_rule": "ar4_plus", "reference_price": "intermittent_hourly"})
    if rows["reference_price"].eq("baseload_seasonal").any():
        raise ValueError("a label-priced wind unit has a baseload-reference CfD cohort; the re-pricing rebuilds "
                         "hourly-reference contracts only")
    out = pd.DataFrame(0.0, index=reference.index, columns=names)
    got = cfd_w.groupby(lab["component"]).sum().reindex(names).fillna(0.0)
    gap = (got - pd.to_numeric(a["cfd_weight"], errors="coerce").fillna(0.0)).abs()
    if float(gap.max() if len(gap) else 0.0) > 1e-6:
        raise ValueError(f"the label table's CfD weight differs from the audit's on {int((gap > 1e-6).sum())} units "
                         f"(max {float(gap.max()):.6f}, first {gap.idxmax()}): not the table or state carriers the "
                         "month bid on")
    if rows.empty:
        return out
    groups = rows.drop_duplicates().reset_index(drop=True)
    basis = str(a["bid_basis"].dropna().iloc[0]) if a["bid_basis"].notna().any() else "contract_value"
    rule = str(a["negative_price_rule"].dropna().iloc[0]) if a["negative_price_rule"].notna().any() else "none"
    profiles = cfd_bid_profiles(groups, reference.astype(float), None, basis, rule != "none")
    merged = rows.reset_index().merge(groups.reset_index().rename(columns={"index": "group"}), on=keys, how="left")
    w = (pd.DataFrame({"component": lab.loc[merged["index"], "component"].to_numpy(),
                       "group": merged["group"].to_numpy(), "w": cfd_w.loc[merged["index"]].to_numpy()})
         .groupby(["group", "component"])["w"].sum().unstack("component", fill_value=0.0)
         .reindex(index=profiles.columns, columns=names, fill_value=0.0))
    out.loc[:, :] = profiles.to_numpy() @ w.to_numpy()
    return out


def rebuild_bids(units: pd.DataFrame, audit: pd.DataFrame, reference: pd.Series, roc_rate: float | None,
                 implied: pd.Series | None = None,
                 labels: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Hourly bid (GBP/MWh, ESO-cost convention) of each unit as the BM priced it, and its pricing label.

    ``units``: index unit, columns ``support_type``, ``ro_banding``, ``cfd_round``. ``audit``: the month's
    ``contract_value_bids.csv``. ``reference``: the Stage 1 wholesale price on the BM snapshots.
    ``implied``: per-unit ``bid_cost / decrease`` from the solve, the last resort for a unit no rule
    prices (constant bid assumed; labelled ``implied_from_solve``). ``labels``: the build's support label
    table, needed when the audit has label-priced units (:data:`LABEL_PRICING`): their bid is the hourly CfD
    part (:func:`label_cfd_bids`) plus a constant that returns the audit's mean bid (every other part of a
    label bid is time-invariant).
    """
    from scripts.market.cfd_bm_bids import foregone_rate
    from scripts.market.revenue_tracking import negative_price_eligibility

    idx = units.index
    bids = pd.DataFrame(0.0, index=reference.index, columns=idx)
    label = pd.Series("unpriced", index=idx, dtype=object)
    a = audit.drop_duplicates("generator").set_index("generator").reindex(idx)
    has = a["pricing"].notna()
    cfd = has & a["pricing"].eq("cfd_contract_value")
    ref = reference.astype(float)
    for (basis, floor, rule, refb, rnd), grp in a[cfd].groupby(
            [a["bid_basis"].fillna("strike"), a["contract_value_floor"].fillna("zero"),
             a["negative_price_rule"].fillna("none"), a["reference_price_basis"].fillna("hourly_wholesale"),
             a["cfd_round"].fillna("").astype(str)], dropna=False):
        r = ref if refb == "hourly_wholesale" else pd.Series(float(grp["mean_reference_price_gbp_per_mwh"].iloc[0]),
                                                              index=ref.index)
        for strike, sub in grp.groupby("strike_price_gbp_per_mwh"):
            cv = foregone_rate(float(strike), r, str(basis), str(floor) if str(floor) in ("zero", "none") else "zero")
            if str(rule) != "none":
                cv = cv.where(negative_price_eligibility(r.to_numpy(), [rnd], str(rule))[:, 0], 0.0)
            s = sub["state_share"].fillna(0.0).to_numpy()
            const = sub["mean_bid_gbp_per_mwh"].to_numpy() - (1.0 - s) * float(cv.mean())
            bids.loc[:, sub.index] = np.outer(cv.to_numpy(), 1.0 - s) + const[None, :]
            label.loc[sub.index] = "cfd_contract_value"
    by_label = has & a["pricing"].eq(LABEL_PRICING)
    if by_label.any():
        if labels is None:
            raise ValueError(f"{int(by_label.sum())} units bid from the support label table "
                             f"(pricing {LABEL_PRICING}); pass the build's label table (input support_labels)")
        names = a.index[by_label]
        part = label_cfd_bids(names, a.loc[names], labels, ref)
        const = pd.to_numeric(a.loc[names, "mean_bid_gbp_per_mwh"], errors="coerce").to_numpy() - part.mean().to_numpy()
        bids.loc[:, names] = part.to_numpy() + const[None, :]
        label.loc[names] = LABEL_PRICING
    flat = has & ~cfd & ~by_label
    if flat.any():
        bids.loc[:, a.index[flat]] = a.loc[flat, "mean_bid_gbp_per_mwh"].astype(float).to_numpy()[None, :]
        label.loc[flat] = a.loc[flat, "pricing"].astype(str)
    sup = units["support_type"].astype(str).str.strip().str.lower()
    band = pd.to_numeric(units["ro_banding"], errors="coerce")
    roc = ~has & sup.eq("roc") & band.gt(0)
    if roc.any() and roc_rate:
        bids.loc[:, idx[roc]] = (band[roc] * float(roc_rate)).to_numpy()[None, :]
        label.loc[roc] = "roc_banding_x_buyout"
    rest = label.eq("unpriced")
    if rest.any() and implied is not None:
        imp = implied.reindex(idx[rest])
        ok = imp.notna()
        if ok.any():
            bids.loc[:, imp.index[ok]] = imp[ok].to_numpy()[None, :]
            label.loc[imp.index[ok]] = "implied_from_solve"
    return bids, label


# ═══════════════════════════════════════════════════════════════════════════
# One month
# ═══════════════════════════════════════════════════════════════════════════
def _snapshot_index(d) -> pd.DatetimeIndex:
    v = d.variables["snapshots_snapshot"]
    units = str(getattr(v, "units", ""))
    match = re.match(r"\s*(\w+)\s+since\s+(.+)", units)
    if not match:
        raise ValueError(f"snapshots_snapshot has no 'since' units: {units!r}")
    unit = {"hours": "h", "minutes": "min", "seconds": "s", "days": "D"}[match.group(1)]
    return pd.DatetimeIndex(pd.Timestamp(match.group(2).strip()) + pd.to_timedelta(np.asarray(v[:]), unit=unit))


def _read_generators(path, technologies, with_p: bool = True) -> dict:
    import netCDF4

    d = netCDF4.Dataset(str(path))
    d.set_auto_mask(False)
    try:
        s = lambda v: np.asarray(d[v][:]).astype(str)  # noqa: E731
        f = lambda v: np.asarray(d[v][:], dtype=float)  # noqa: E731
        names = s("generators_i")
        g = pd.DataFrame({"carrier": s("generators_carrier"), "bus": s("generators_bus"),
                          "support_type": s("generators_support_type") if "generators_support_type" in d.variables
                          else "merchant",
                          "ro_banding": f("generators_ro_banding") if "generators_ro_banding" in d.variables else np.nan,
                          "cfd_round": s("generators_cfd_round") if "generators_cfd_round" in d.variables else ""},
                         index=names)
        g = g[g["carrier"].isin(technologies)]
        snaps = _snapshot_index(d)
        weights = (pd.Series(f("snapshots_generators"), index=snaps) if "snapshots_generators" in d.variables
                   else pd.Series(1.0, index=snaps))
        p = None
        if with_p:
            cols = s("generators_t_p_i")
            pos = pd.Series(np.arange(len(cols)), index=cols)
            take = pos.reindex(g.index).dropna().astype(int)
            # Whole-variable read, then slice: netCDF4 fancy indexing on thousands of columns takes minutes.
            raw = (np.asarray(d["generators_t_p"][:], dtype=float)[:, np.sort(take.to_numpy())] if len(take)
                   else np.zeros((len(snaps), 0)))
            order = cols[np.sort(take.to_numpy())]
            # Units absent from the saved time series held zero output throughout.
            p = pd.DataFrame(raw, index=snaps, columns=order).reindex(columns=g.index, fill_value=0.0)
    finally:
        d.close()
    return {"generators": g, "snapshots": snaps, "weights": weights, "p": p}


#: Seconds of mtime slack (OneDrive and copies can reorder writes made in the same minute).
VINTAGE_SLACK_S = 60.0


def check_month_vintage(name: str, balancing_nc, stage1_files, month_files_=()) -> None:
    """Refuse a balancing month older than its Stage 1 (it was solved on a previous Stage 1).

    Snakemake re-solves such a month before re-pricing it (the Stage 1 files are inputs of the solve), so this
    fires only outside the DAG or on a stale file kept with ``ancient``/``--rerun-triggers``; the volume check in
    :func:`reprice_month` catches the same thing from the content.
    """
    t_bm = Path(balancing_nc).stat().st_mtime
    for f in list(stage1_files) + list(month_files_):
        if f and Path(f).exists() and Path(f).stat().st_mtime > t_bm + VINTAGE_SLACK_S:
            raise RuntimeError(
                f"{name}: the balancing month ({Path(balancing_nc).name}) is older than {Path(f).name}; it was solved "
                "on a previous Stage 1 or its outputs were rewritten since. Re-solve the month "
                "(solve_balancing_mechanism) before re-pricing its turn-down; nothing was written.")


def reprice_month(name: str, balancing_nc, stage1_nc, price_csv, audit_csv, redispatch_csv, bus_zone: pd.Series,
                  cfg: dict, roc_rate: float | None, log=None, labels_csv=None) -> tuple[pd.DataFrame, dict]:
    """The per-unit table and the month's totals. Raises if the month predates its Stage 1 or the as-solved
    check fails. ``labels_csv``: the build's support label table (long-range months)."""
    log = log or logger
    check_month_vintage(name, balancing_nc, [stage1_nc, price_csv])
    cfg = allocation_config({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG})
    tech = cfg["technologies"]
    bm = _read_generators(balancing_nc, tech)
    s1 = _read_generators(stage1_nc, tech)
    units = bm["generators"]
    snaps = bm["snapshots"]
    p_s1 = s1["p"].reindex(index=snaps, columns=units.index)
    if p_s1.isna().any().any():
        units_missing = int(p_s1.isna().all(axis=0).sum())
        hours_missing = int(p_s1.isna().all(axis=1).sum())
        raise ValueError(f"{name}: {units_missing} of its {len(units)} wind units and {hours_missing} of its "
                         f"{len(snaps)} hours are missing from the Stage 1 network {Path(stage1_nc).name}: the month "
                         "was solved on another Stage 1 (or another fleet). Re-solve it; nothing was written.")
    hours = bm["weights"].reindex(snaps).fillna(1.0).to_numpy()[:, None]
    a = p_s1.to_numpy()
    dec_mw = np.clip(a - bm["p"].to_numpy(), 0.0, None)
    price = pd.read_csv(price_csv, index_col=0, parse_dates=True)["wholesale_price"]
    price.index = pd.DatetimeIndex(price.index)
    reference = price.reindex(snaps)
    if reference.isna().any():
        raise ValueError(f"{name}: Stage 1 price missing for {int(reference.isna().sum())} balancing snapshots")
    rs = pd.read_csv(redispatch_csv)
    rs = rs[rs["component"].isin(units.index)].drop_duplicates("component").set_index("component")
    implied = (rs["bid_cost"] / rs["decrease_MWh"]).where(rs["decrease_MWh"] > 1e-9)
    audit = pd.read_csv(audit_csv)
    labels = None
    if labels_csv:
        from scripts.market.support_bids import load_support_labels

        labels = load_support_labels(labels_csv)
    bids, label = rebuild_bids(units, audit, reference, roc_rate, implied, labels)
    # Balancing non-participants (turned down at the penalty bid, market_accounting's rule; P3 leaves a few
    # MWh of solver noise on them) are outside the BM: they keep their own volume at their penalty price,
    # take no share of the zone's turn-down, and the bill nets them out as it always has.
    from scripts.market.market_accounting import _non_participant_turn_down

    penalty = units.index.intersection(_non_participant_turn_down(rs).index)
    if len(penalty):
        bids.loc[:, penalty] = implied.reindex(penalty).to_numpy()[None, :]
        label.loc[penalty] = "non_participant_penalty"
    zone = units["bus"].map(bus_zone)
    groups = zone.where(~units.index.isin(penalty))
    pr_mw = pro_rata_turn_down(a, dec_mw, groups.to_numpy())
    b = bids.to_numpy()
    ours = (dec_mw * hours)
    pro = (pr_mw * hours)
    share = audit.drop_duplicates("generator").set_index("generator")["state_share"].reindex(units.index).fillna(0.0)
    table = pd.DataFrame({
        "unit": units.index, "carrier": units["carrier"].to_numpy(), "bus": units["bus"].to_numpy(),
        "zone": zone.to_numpy(), "support_type": units["support_type"].to_numpy(), "pricing": label.to_numpy(),
        "state_share": share.to_numpy(), "stage1_MWh": (np.clip(a, 0, None) * hours).sum(0),
        "our_MWh": ours.sum(0), "pro_rata_MWh": pro.sum(0), "our_cost_GBP": (ours * b).sum(0),
        "pro_rata_cost_GBP": (pro * b).sum(0),
        "solve_bid_cost_GBP": rs["bid_cost"].reindex(units.index).fillna(0.0).to_numpy(),
        "solve_decrease_MWh": rs["decrease_MWh"].reindex(units.index).fillna(0.0).to_numpy()}, columns=UNIT_COLUMNS)
    solve_cost = float(table["solve_bid_cost_GBP"].sum())
    our_cost = float(table["our_cost_GBP"].sum())
    gross = float(table["solve_bid_cost_GBP"].abs().sum())
    tol = max(float(cfg["check_tolerance_rel"]) * gross, float(cfg["check_tolerance_abs_gbp"]))
    vol_gap = float(table["our_MWh"].sum() - table["solve_decrease_MWh"].sum())
    unit_gap = float((table["our_MWh"] - table["solve_decrease_MWh"]).abs().max()) if len(table) else 0.0
    if unit_gap > 1.0:
        # The solve's own turn-down (Stage 1 it anchored to - BM) is not this Stage 1 - BM: a different Stage 1.
        raise AssertionError(
            f"{name}: Stage 1 - BM turn-down differs from the solve's own by up to {unit_gap:,.1f} MWh on one unit "
            f"(total {vol_gap:,.1f} MWh): the balancing month was solved on another Stage 1 than {Path(stage1_nc).name}. "
            "Re-solve the month; nothing was written.")
    ok = abs(our_cost - solve_cost) <= tol and abs(vol_gap) <= max(1e-6 * float(table["our_MWh"].sum()), 1.0)
    zone_check = (pd.DataFrame({"z": groups.fillna("outside").to_numpy()}).assign(o=ours.sum(0), p=pro.sum(0))
                  .groupby("z")[["o", "p"]].sum())
    conserved = float((zone_check["o"] - zone_check["p"]).abs().max()) if len(zone_check) else 0.0
    totals = {
        "scenario": name, "bases": list(BASES), "technologies": tech, "zone_basis": cfg["zone_basis"],
        "hours": float(hours.sum()), "units": int(len(units)),
        "turn_down_MWh": float(table["our_MWh"].sum()), "pro_rata_turn_down_MWh": float(table["pro_rata_MWh"].sum()),
        "unzoned_turn_down_MWh": float(table.loc[table["zone"].isna(), "our_MWh"].sum()),
        "non_participant_turn_down_MWh": float(table.loc[table["pricing"].eq("non_participant_penalty"), "our_MWh"].sum()),
        "wind_bid_cost_as_solved_GBP": solve_cost, "wind_bid_cost_as_solved_rebuilt_GBP": our_cost,
        "wind_bid_cost_pro_rata_GBP": float(table["pro_rata_cost_GBP"].sum()),
        "pro_rata_uplift_GBP": float(table["pro_rata_cost_GBP"].sum()) - our_cost,
        "check": {"rebuilt_minus_solve_GBP": our_cost - solve_cost, "tolerance_GBP": tol, "max_unit_volume_gap_MWh": unit_gap,
                  "relative_to_gross": (our_cost - solve_cost) / gross if gross else 0.0,
                  "volume_gap_MWh": vol_gap, "max_zone_volume_gap_MWh": conserved, "passed": bool(ok)},
        "pricing_counts": {k: int(v) for k, v in label.value_counts().items()},
        "by_zone": {str(z): {"our_MWh": float(r.our_MWh), "pro_rata_MWh": float(r.pro_rata_MWh),
                             "our_cost_GBP": float(r.our_cost_GBP), "pro_rata_cost_GBP": float(r.pro_rata_cost_GBP)}
                    for z, r in table.assign(zone=table["zone"].fillna("unzoned")).groupby("zone")[
                        ["our_MWh", "pro_rata_MWh", "our_cost_GBP", "pro_rata_cost_GBP"]].sum().iterrows()},
        "by_support": {str(z): {"our_MWh": float(r.our_MWh), "pro_rata_MWh": float(r.pro_rata_MWh),
                                "our_cost_GBP": float(r.our_cost_GBP), "pro_rata_cost_GBP": float(r.pro_rata_cost_GBP)}
                       for z, r in table.groupby("support_type")[
                           ["our_MWh", "pro_rata_MWh", "our_cost_GBP", "pro_rata_cost_GBP"]].sum().iterrows()},
    }
    log.info("%s: wind turn-down %.3f TWh; bid cost as solved %.2f m (rebuilt %.2f m, gap %.4f%% of gross); "
             "pro rata %.2f m; uplift %+.2f m", name, totals["turn_down_MWh"] / 1e6, solve_cost / 1e6, our_cost / 1e6,
             100 * totals["check"]["relative_to_gross"], totals["wind_bid_cost_pro_rata_GBP"] / 1e6,
             totals["pro_rata_uplift_GBP"] / 1e6)
    if not ok:
        raise AssertionError(
            f"{name}: the rebuilt bids do not reproduce the solve's wind bid cost: rebuilt {our_cost:,.0f} vs solve "
            f"{solve_cost:,.0f} GBP (tolerance {tol:,.0f}), volume gap {vol_gap:,.1f} MWh; pricing "
            f"{totals['pricing_counts']}")
    return table, totals


# ═══════════════════════════════════════════════════════════════════════════
# A family (monthly_aggregates study)
# ═══════════════════════════════════════════════════════════════════════════
#: The NESO-comparable cost NESO published for each network (benchmark metric), by delay case.
NETWORK_BENCHMARKS = {None: "neso_constraint_cost_gbpm", "no_acceleration": "neso_cost_no_acceleration_gbpm",
                      "seven_late": "neso_cost_seven_late_gbpm", "no_upgrades": "neso_cost_no_upgrades_gbpm"}


def network_benchmark(delay_case: str | None, benchmarks_file=None) -> dict:
    """NESO's published cost (GBP) for a network: the central/CP30 network, or a delay case."""
    from scripts.CP30.balancing_validation import BENCHMARKS_FILE, read_benchmarks

    metric = NETWORK_BENCHMARKS.get(delay_case)
    if metric is None:
        raise ValueError(f"no NESO cost benchmark for network {delay_case!r}; known: {sorted(map(str, NETWORK_BENCHMARKS))}")
    bench = read_benchmarks(benchmarks_file or BENCHMARKS_FILE)
    return {"network": delay_case or "central", "metric": metric, "value_GBP": float(bench.at[metric, "value"]) * 1e6,
            "source": f"{bench.at[metric, 'sheet']}: {bench.at[metric, 'cell']}"}


def select_basis(mode: str, costs: dict, neso_gbp: float | None) -> tuple[str, str]:
    """(basis, reason). ``costs``: {basis: NESO-comparable GBP}. A tie goes to cheapest_first (as solved)."""
    if mode in BASES:
        return mode, "configured"
    if mode != "closest_to_neso":
        raise ValueError(f"turn_down_allocation.mode must be one of {MODES}, got {mode!r}")
    if neso_gbp is None or any(costs.get(b) is None for b in BASES):
        raise ValueError("closest_to_neso needs both NESO-comparable costs and NESO's figure for the network")
    gaps = {b: abs(float(costs[b]) - float(neso_gbp)) for b in BASES}
    basis = min(BASES, key=lambda b: (gaps[b], BASES.index(b)))
    return basis, (f"closest to NESO's {neso_gbp / 1e6:,.0f} m: as solved {costs['cheapest_first'] / 1e6:,.0f} m "
                   f"({100 * (costs['cheapest_first'] / neso_gbp - 1):+.1f}%), pro rata "
                   f"{costs['pro_rata_zonal'] / 1e6:,.0f} m ({100 * (costs['pro_rata_zonal'] / neso_gbp - 1):+.1f}%)")


def summarise_family(study: str, month_totals: list[dict], annual_summary: dict | None, cfg: dict,
                     network: dict | None = None) -> dict:
    """The study's annual totals on both bases, and the headline basis (configured, or closest to NESO).

    ``network``: :func:`network_benchmark` of the family's network (needed for ``closest_to_neso``).
    """
    cfg = allocation_config({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG})
    keys = ("turn_down_MWh", "wind_bid_cost_as_solved_GBP", "wind_bid_cost_as_solved_rebuilt_GBP",
            "wind_bid_cost_pro_rata_GBP", "pro_rata_uplift_GBP", "unzoned_turn_down_MWh", "hours")
    tot = {k: float(sum(m.get(k, 0.0) for m in month_totals)) for k in keys}
    agg = (annual_summary or {}).get("aggregate") or {}
    neso = agg.get("neso_comparable_redispatch_cost_GBP")
    neso = float(neso) if neso is not None else None
    pro = neso + tot["pro_rata_uplift_GBP"] if neso is not None else None
    costs = {"cheapest_first": neso, "pro_rata_zonal": pro}
    by_zone: dict = {}
    for m in month_totals:
        for z, r in (m.get("by_zone") or {}).items():
            acc = by_zone.setdefault(z, {k: 0.0 for k in r})
            for k, v in r.items():
                acc[k] += float(v)
    neso_figure = (network or {}).get("value_GBP")
    basis, reason = select_basis(cfg["mode"], costs, neso_figure)
    wind = {"cheapest_first": tot["wind_bid_cost_as_solved_GBP"],
            "pro_rata_zonal": tot["wind_bid_cost_as_solved_GBP"] + tot["pro_rata_uplift_GBP"]}
    return {
        "study": study, "mode": cfg["mode"], "technologies": cfg["technologies"], "zone_basis": cfg["zone_basis"],
        "months": [m.get("scenario") for m in month_totals], **tot,
        "neso_comparable_redispatch_cost_as_solved_GBP": neso,
        "neso_comparable_redispatch_cost_pro_rata_GBP": pro,
        "neso_comparable_redispatch_cost_by_basis_GBP": costs,
        "wind_bid_cost_by_basis_GBP": wind,
        "network": network,
        "neso_network_cost_GBP": neso_figure,
        "gap_to_neso_by_basis": ({b: (c / neso_figure - 1.0 if c is not None else None) for b, c in costs.items()}
                                 if neso_figure else None),
        "headline_basis": basis,
        "headline_basis_reason": reason,
        "headline_neso_comparable_redispatch_cost_GBP": costs[basis],
        "headline_wind_bid_cost_GBP": wind[basis],
        "basis_labels": {"cheapest_first": "as solved (the balancing LP's cheapest-first turn-down)",
                         "pro_rata_zonal": "pro rata zonal turn-down (NESO CP.08 zones, Stage 1 output shares)"},
        "checks_passed": all(bool((m.get("check") or {}).get("passed")) for m in month_totals),
        "by_zone": by_zone,
    }


# ═══════════════════════════════════════════════════════════════════════════
# The helper the bill and ownership layers read turn-down through
# ═══════════════════════════════════════════════════════════════════════════
#: Suffixes of a month's files; the pro rata table and totals sit beside them.
MONTH_FILE_SUFFIXES = ("_redispatch_summary.csv", "_balancing.nc", "_balancing_dispatch.csv", "_constraint_costs.csv",
                       "_contract_value_bids.csv", "_pro_rata_turn_down.csv", "_pro_rata_turn_down.json",
                       "_turn_down_basis.json")


def month_files(path) -> tuple[Path, Path]:
    """Any file of a BM month (redispatch summary, balancing network, ...) -> its pro rata (csv, json)."""
    p = Path(path)
    for suffix in MONTH_FILE_SUFFIXES:
        if p.name.endswith(suffix):
            stem = p.name[: -len(suffix)]
            return p.with_name(f"{stem}_pro_rata_turn_down.csv"), p.with_name(f"{stem}_pro_rata_turn_down.json")
    raise ValueError(f"{path}: not a balancing-month file (expected one of {MONTH_FILE_SUFFIXES})")


def pro_rata_path(redispatch_path) -> Path:
    """``..._redispatch_summary.csv`` (or any month file) -> its ``..._pro_rata_turn_down.csv``."""
    return month_files(redispatch_path)[0]


def month_allocation(path) -> dict | None:
    """The month's re-pricing totals (``*_pro_rata_turn_down.json``), or None when it was not re-priced."""
    if path is None:
        return None
    _, js = month_files(path)
    if not js.exists():
        return None
    return json.loads(js.read_text(encoding="utf-8"))


# ── The basis in force ───────────────────────────────────────────────────────
# Each month's basis is a file beside its outputs, ``{scenario}_turn_down_basis.json`` (rule
# select_turn_down_basis): the headline basis of the month's network, chosen on that network's market family
# (``*_pro_rata_turn_down_summary.json``). Every reader below takes the basis from it, so the bill and
# ownership scripts need no argument: their rules declare the month files (turn_down_month_inputs).


def month_basis_path(path) -> Path:
    """Any month file -> its ``*_turn_down_basis.json``."""
    csv, _ = month_files(path)
    return csv.with_name(csv.name.replace("_pro_rata_turn_down.csv", "_turn_down_basis.json"))


def selected_basis(selection) -> str:
    """The headline basis of a family summary or a month basis file (path or dict)."""
    data = selection if isinstance(selection, dict) else json.loads(Path(selection).read_text(encoding="utf-8"))
    basis = str(data.get("headline_basis") or "")
    if basis not in BASES:
        raise ValueError(f"turn-down selection names no basis in {BASES}: {basis!r}")
    return basis


def network_basis_study(scenario: str, cases: dict) -> str | None:
    """The market family whose headline basis a month's bill reads: its network's.

    ``cases``: the consumer bill forecast cases (each with ``source_scenarios``). A month's network group is
    its cases' ``register_account_group`` (``cp30_2030_cp30_network`` when absent: the CP30 network); the group's base
    case is the case of that name (``CP30_2030_cp30_network``, ``CP30_2030_no_acceleration``, ...), and its
    ``annual_aggregate_study`` is the network's market family. P2/P3 bid families and price-level twins thus
    follow their network. None: no bill case reads the month.
    """
    studies = {}
    lower = {str(c).lower(): c for c in cases}
    for cid, case in cases.items():
        if scenario not in (case.get("source_scenarios") or []):
            continue
        group = str(case.get("turn_down_network_case") or case.get("register_account_group") or "cp30_2030_cp30_network")
        base = lower.get(group.lower())
        if base is None or not cases[base].get("annual_aggregate_study"):
            raise ValueError(f"{cid}: network group {group!r} has no base bill case with an annual_aggregate_study")
        studies[cases[base]["annual_aggregate_study"]] = cid
    if len(studies) > 1:
        raise ValueError(f"{scenario} is read by bill cases on different networks: {studies}")
    return next(iter(studies), None)


def month_basis_record(scenario: str, network_summary: dict, network_summary_path: str | None = None) -> dict:
    """The month basis file's content: the network's selection, both costs and why."""
    return {"scenario": scenario, "headline_basis": selected_basis(network_summary),
            "network_study": network_summary.get("study"), "network": network_summary.get("network"),
            "mode": network_summary.get("mode"), "reason": network_summary.get("headline_basis_reason"),
            "neso_comparable_redispatch_cost_by_basis_GBP": network_summary.get(
                "neso_comparable_redispatch_cost_by_basis_GBP"),
            "neso_network_cost_GBP": network_summary.get("neso_network_cost_GBP"),
            "source": network_summary_path}


def basis_in_force(basis: str | None = None) -> str:
    if basis is not None and basis not in BASES:
        raise ValueError(f"turn-down basis must be one of {BASES}, got {basis!r}")
    return basis or "cheapest_first"


_FORCED: dict = {"basis": None}


class force_basis:
    """Context manager for tests and diagnostics: read every re-priced month on ``basis``, whatever its basis
    file says (``with force_basis("pro_rata_zonal"): ...``). Never used by a rule."""

    def __init__(self, basis: str | None):
        self.basis = basis_in_force(basis) if basis else None

    def __enter__(self):
        self._previous = _FORCED["basis"]
        _FORCED["basis"] = self.basis
        _HOURLY_CACHE.clear()
        return self

    def __exit__(self, *exc):
        _FORCED["basis"] = self._previous
        _HOURLY_CACHE.clear()
        return False


def month_basis(path, basis: str | None = None) -> str:
    """The basis one month's file is read on.

    ``basis`` given: that basis (tests, diagnostics). Otherwise the month's ``*_turn_down_basis.json``; as
    solved for a Stage 1 (wholesale) file and for a month without one (2050, historical, central dispatch,
    or a month no bill group reads). A pro rata basis needs the month's re-pricing; its absence is an error.
    """
    if path is None:
        return "cheapest_first"
    try:
        month_files(path)
    except ValueError:
        return "cheapest_first"
    if basis is None and _FORCED["basis"] is not None:
        basis = _FORCED["basis"]
    if basis is None:
        bp = month_basis_path(path)
        basis = selected_basis(bp) if bp.exists() else "cheapest_first"
    b = basis_in_force(basis)
    if b != "cheapest_first":
        if month_allocation(path) is None:
            raise FileNotFoundError(f"turn-down basis {b} for {Path(path).name} but {month_files(path)[1].name} does "
                                    "not exist (rule reprice_turn_down_pro_rata)")
        _check_repricing_current(path)
    return b


def _check_repricing_current(path) -> None:
    """Refuse a re-pricing older than the month's solve outputs (the month was re-solved since)."""
    csv, js = month_files(path)
    stem = js.name[: -len("_pro_rata_turn_down.json")]
    t = min(csv.stat().st_mtime if csv.exists() else 0.0, js.stat().st_mtime)
    for suffix in ("_balancing.nc", "_redispatch_summary.csv"):
        f = js.with_name(stem + suffix)
        if f.exists() and f.stat().st_mtime > t + VINTAGE_SLACK_S:
            raise RuntimeError(f"{js.name} is older than {f.name}: the month was re-solved after its turn-down was "
                               "re-priced; re-run reprice_turn_down_pro_rata before reading it on the pro rata basis")


_month_basis = month_basis


def basis_record(paths) -> dict:
    """What an output records about the bases its months were read on: {basis: [month file names]}."""
    out: dict = {}
    for p in paths or []:
        out.setdefault(month_basis(p), []).append(Path(p).name)
    return out


def bid_cost_uplift_in_force(path, basis: str | None = None) -> float:
    """GBP added to the month's accepted bid cost by the basis in force (0 as solved)."""
    if _month_basis(path, basis) == "cheapest_first":
        return 0.0
    return float(month_allocation(path)["pro_rata_uplift_GBP"])


def apply_allocation_to_redispatch(redispatch: pd.DataFrame, pro_rata: pd.DataFrame | None, basis: str) -> pd.DataFrame:
    """A redispatch summary on ``basis``: under ``pro_rata_zonal`` the units of the re-priced technologies
    carry their pro rata ``decrease_MWh`` and ``bid_cost`` (``net_cost`` moved by the same amount;
    ``bid_cost`` = solve bid cost + (pro rata - rebuilt as-solved) cost, so the summed move is exactly
    the month's uplift); everything else, and every unit under ``cheapest_first``, is as solved. Works
    on a frame with ``component`` as a column or as the index."""
    if basis not in BASES:
        raise ValueError(f"turn-down basis must be one of {BASES}, got {basis!r}")
    if basis == "cheapest_first":
        return redispatch
    if pro_rata is None:
        raise FileNotFoundError("turn-down basis pro_rata_zonal but no pro rata table was given")
    out = redispatch.copy()
    names_all = (out["component"] if "component" in out.columns else pd.Series(out.index, index=out.index)).astype(str)
    pr = pro_rata.drop_duplicates("unit").set_index("unit")
    hit = names_all.isin(pr.index).to_numpy()
    if "type" in out.columns:
        hit &= out["type"].astype(str).eq("generator").to_numpy()
    names = names_all[hit].to_numpy()
    old_bid = pd.to_numeric(out.loc[hit, "bid_cost"], errors="coerce").fillna(0.0).to_numpy()
    new_bid = old_bid + (pr["pro_rata_cost_GBP"].reindex(names).to_numpy()
                         - pr["our_cost_GBP"].reindex(names).to_numpy())
    out.loc[hit, "decrease_MWh"] = pr["pro_rata_MWh"].reindex(names).to_numpy()
    out.loc[hit, "bid_cost"] = new_bid
    if "net_cost" in out.columns:
        out.loc[hit, "net_cost"] = pd.to_numeric(out.loc[hit, "net_cost"], errors="coerce").fillna(0.0).to_numpy() + (
            new_bid - old_bid)
    return out


def read_redispatch_in_force(path, basis: str | None = None, **read_csv_kwargs) -> pd.DataFrame:
    """Read a month's ``*_redispatch_summary.csv`` on the turn-down basis in force.

    The one place the bill and ownership layers read per-unit BM turn-down volumes and bid costs.
    ``read_csv_kwargs`` pass through (``index_col=0`` keeps ``component`` as the index).
    """
    rs = pd.read_csv(path, **read_csv_kwargs)
    b = _month_basis(path, basis)
    if b == "cheapest_first":
        return rs
    return apply_allocation_to_redispatch(rs, pd.read_csv(month_files(path)[0]), b)


def allocation_in_force(redispatch_path, basis: str | None = None) -> pd.DataFrame:
    """Alias of :func:`read_redispatch_in_force` with ``component`` as a column."""
    return read_redispatch_in_force(redispatch_path, basis)


def allocation_groups(table: pd.DataFrame) -> pd.Series:
    """unit -> the zone it shares turn-down in (NaN: keeps its own; non-participants and unzoned units)."""
    t = table.set_index("unit") if "unit" in table.columns else table
    return t["zone"].where(t["pricing"].astype(str).ne("non_participant_penalty"))


def reallocated_dispatch(stage1_mw: np.ndarray, balancing_mw: np.ndarray, groups) -> np.ndarray:
    """BM output (MW) once the turn-down is re-shared: Stage 1 - pro rata turn-down + the unit's own turn-up."""
    s1 = np.asarray(stage1_mw, dtype=float)
    bm = np.asarray(balancing_mw, dtype=float)
    dec = np.clip(s1 - bm, 0.0, None)
    up = np.clip(bm - s1, 0.0, None)
    return s1 - pro_rata_turn_down(s1, dec, groups) + up


_HOURLY_CACHE: dict = {}


def hourly_dispatch_in_force(balancing_path, basis: str | None = None) -> pd.DataFrame | None:
    """Hourly BM output (MW, BM snapshots x re-priced units) on the basis in force, or None as solved."""
    if _month_basis(balancing_path, basis) == "cheapest_first":
        return None
    key = str(Path(balancing_path).resolve())
    if key not in _HOURLY_CACHE:
        csv, _ = month_files(balancing_path)
        totals = month_allocation(balancing_path)
        table = pd.read_csv(csv).drop_duplicates("unit").set_index("unit")
        tech = list((totals.get("config") or {}).get("technologies") or DEFAULT_CONFIG["technologies"])
        bm = _read_generators(balancing_path, tech)
        # F20 parity harness: a path recorded in the month's JSON is redirected to the input tree in a
        # parity run (PYPSA_GB_INPUT_RESOURCES); unchanged otherwise.
        from scripts.public_ownership.f20_parity.run_parity import parity_path

        s1 = _read_generators(parity_path(totals["inputs"]["stage1_network"]), tech)
        units = bm["generators"].index.intersection(table.index)
        p_s1 = s1["p"].reindex(index=bm["snapshots"], columns=units).fillna(0.0)
        p_bm = bm["p"].reindex(columns=units)
        _HOURLY_CACHE.clear()
        _HOURLY_CACHE[key] = pd.DataFrame(
            reallocated_dispatch(p_s1.to_numpy(), p_bm.to_numpy(), allocation_groups(table).reindex(units).to_numpy()),
            index=bm["snapshots"], columns=units)
    return _HOURLY_CACHE[key]


def balancing_dispatch_in_force(frame: pd.DataFrame, balancing_path, basis: str | None = None) -> pd.DataFrame:
    """``frame`` (snapshots x generators, BM output read from ``balancing_path``) on the basis in force.

    The re-priced units' columns are replaced by their re-shared output; others are untouched. The frame
    may be indexed by timestamps or by position (0..n-1); rows are matched by position.
    """
    alloc = hourly_dispatch_in_force(balancing_path, basis)
    if alloc is None:
        return frame
    cols = frame.columns.intersection(alloc.columns)
    if not len(cols):
        return frame
    if len(frame.index) != len(alloc.index):
        raise ValueError(f"{balancing_path}: {len(frame.index)} rows against {len(alloc.index)} BM snapshots")
    out = frame.copy()
    out.loc[:, cols] = alloc[cols].to_numpy()
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Entry points
# ═══════════════════════════════════════════════════════════════════════════
def run_month(name: str, paths: dict, cfg: dict, roc_rate: float | None, secured: dict | None,
              out_csv: str, out_json: str, zones_file: str | None = None, log=None) -> dict:
    from scripts.market.neso_zones import ZONES_FILE, network_bus_zones

    log = log or logger
    bz = network_bus_zones(paths["network"], zones_file or ZONES_FILE, secured)
    counts = {k: int(v) for k, v in bz["mapping"].astype(str).str.split(":").str[0].value_counts().items()}
    log.info("%s: buses zoned on this network: %s", name, counts)
    table, totals = reprice_month(name, paths["network"], paths["stage1_network"], paths["stage1_price"],
                                  paths["contract_value_bids"], paths["redispatch"], bz["neso_zone"], cfg, roc_rate, log,
                                  paths.get("support_labels"))
    totals["zone_mapping_counts"] = counts
    totals["config"] = {k: v for k, v in allocation_config({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG}).items()
                        if k != "mode"}
    totals["roc_bid_rate_gbp_per_mwh_per_banding"] = roc_rate
    totals["inputs"] = {k: str(v) for k, v in paths.items()}
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    Path(out_json).write_text(json.dumps(totals, indent=2, default=str), encoding="utf-8")
    return totals


def run_family(study: str, month_jsons: list[str], annual_summary_path: str | None, cfg: dict, out_json: str,
               log=None, delay_case: str | None = None, benchmarks_file=None) -> dict:
    log = log or logger
    months = [json.loads(Path(p).read_text(encoding="utf-8")) for p in month_jsons]
    annual = (json.loads(Path(annual_summary_path).read_text(encoding="utf-8"))
              if annual_summary_path and Path(annual_summary_path).exists() else None)
    out = summarise_family(study, months, annual, cfg, network_benchmark(delay_case, benchmarks_file))
    out["inputs"] = {"months": list(month_jsons), "annual_summary": annual_summary_path}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("%s: wind turn-down %.2f TWh; wind bid cost as solved %.0f m, pro rata %.0f m (uplift %+.0f m); "
             "NESO-comparable %s -> %s m (headline: %s)", study, out["turn_down_MWh"] / 1e6,
             out["wind_bid_cost_as_solved_GBP"] / 1e6, out["wind_bid_cost_pro_rata_GBP"] / 1e6,
             out["pro_rata_uplift_GBP"] / 1e6,
             f"{out['neso_comparable_redispatch_cost_as_solved_GBP'] / 1e6:,.0f}"
             if out["neso_comparable_redispatch_cost_as_solved_GBP"] is not None else "n/a",
             f"{out['neso_comparable_redispatch_cost_pro_rata_GBP'] / 1e6:,.0f}"
             if out["neso_comparable_redispatch_cost_pro_rata_GBP"] is not None else "n/a", out["headline_basis"])
    log.info("%s: headline basis %s (mode %s; %s)", study, out["headline_basis"], out["mode"],
             out["headline_basis_reason"])
    return out


def month_paths(name: str, stage1_source: str, resources: str = "resources") -> dict:
    m = f"{resources}/market"
    return {"network": f"{m}/{name}_balancing.nc", "redispatch": f"{m}/{name}_redispatch_summary.csv",
            "contract_value_bids": f"{m}/{name}_contract_value_bids.csv",
            "stage1_network": f"{m}/{stage1_source}_wholesale.nc",
            "stage1_price": f"{m}/{stage1_source}_wholesale_price.csv"}


def scenario_settings(scenario: dict) -> dict:
    """What the month re-pricing reads from a month's scenario config (the rule passes only this as params).

    The mode is left out: a month is always priced on both bases, so changing the mode re-runs only the
    family summary.
    """
    secured = dict(((((scenario.get("network") or {}).get("transmission")) or {}).get("outage_schedule") or {})
                   .get("secured_capability") or {})
    zones_file = ((((scenario.get("validation") or {}).get("balancing")) or {}).get("zones_file"))
    cfg = scenario_allocation_config(scenario)
    cfg.pop("mode")
    return {"turn_down_allocation": cfg,
            "roc_bid_rate": roc_bid_rate(scenario),
            "zones_file": zones_file,
            "secured_capability": {k: secured.get(k) for k in ("geometry_file", "include", "exclude") if secured.get(k)}}


def family_settings(scenario: dict) -> dict:
    """What the family summary reads from its first month's scenario config: the mode and the network."""
    from scripts.CP30.balancing_validation import delay_case_of

    delivery = (((scenario.get("market") or {}).get("balancing") or {}).get("network_delivery") or {})
    return {"turn_down_allocation": scenario_allocation_config(scenario),
            "delay_case": delay_case_of(delivery.get("case")),
            "benchmarks_file": (((scenario.get("validation") or {}).get("balancing")) or {}).get("benchmarks_file")}


def _load_scenarios():
    import yaml

    sys.path.insert(0, str(PROJECT_ROOT / "config"))
    from config_loader import load_config

    full = load_config(PROJECT_ROOT / "config")
    return full["study_settings"], full["scenarios"]


if "snakemake" in globals():
    from scripts.utilities.logging_config import setup_logging

    _log = setup_logging(snakemake.log[0] if snakemake.log else "turn_down_allocation")
    _s = dict(snakemake.params.settings)
    if snakemake.rule == "select_turn_down_basis":
        _summary = json.loads(Path(snakemake.input.summary).read_text(encoding="utf-8"))
        _rec = month_basis_record(snakemake.wildcards.scenario, _summary, str(snakemake.input.summary))
        Path(snakemake.output.json).write_text(json.dumps(_rec, indent=2, default=str), encoding="utf-8")
        _log.info("%s: turn-down basis %s (network family %s: %s)", snakemake.wildcards.scenario,
                  _rec["headline_basis"], _rec["network_study"], _rec["reason"])
    elif snakemake.rule == "reprice_turn_down_pro_rata":
        _labels = snakemake.input.get("support_labels", None) or []
        _labels = [_labels] if isinstance(_labels, str) else list(_labels)
        _paths = {"network": snakemake.input.network, "redispatch": snakemake.input.redispatch,
                  "contract_value_bids": snakemake.input.contract_value_bids,
                  "stage1_network": snakemake.input.stage1_network, "stage1_price": snakemake.input.stage1_price}
        if _labels:
            _paths["support_labels"] = _labels[0]
        run_month(snakemake.wildcards.scenario, _paths,
                  _s["turn_down_allocation"], _s.get("roc_bid_rate"), _s.get("secured_capability"),
                  snakemake.output.csv, snakemake.output.json, _s.get("zones_file"), _log)
    else:
        run_family(snakemake.wildcards.study, list(snakemake.input.months), snakemake.input.annual_summary,
                   _s["turn_down_allocation"], snakemake.output.json, _log, _s.get("delay_case"),
                   _s.get("benchmarks_file"))
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="one balancing month")
    parser.add_argument("--study", help="a monthly_aggregates study: every month, then the summary")
    parser.add_argument("--resources", default="resources")
    parser.add_argument("--out-dir", default=None, help="default: resources/market (months), resources/analysis")
    parser.add_argument("--mode", choices=MODES, default=None, help="override the configured mode (summary only)")
    parser.add_argument("--summary-only", action="store_true", help="re-use the months' JSONs; write the summary")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _main, _scen = _load_scenarios()
    _aggs = _main.get("monthly_aggregates") or {}
    if args.scenario:
        _names = [args.scenario]
    elif args.study in _aggs:
        _names = _aggs[args.study]["scenarios"]
    else:  # a study declared only in a queue's --configfile: the CP30 month naming
        _tag = args.study.removeprefix("CP30_2030_")
        _names = [f"CP30_FFR_2030_{m}_{_tag}" for m in
                  ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
    _jsons = []
    for _n in _names:
        _od = Path(args.out_dir or f"{args.resources}/market")
        if args.summary_only:
            _jsons.append(str(_od / f"{_n}_pro_rata_turn_down.json"))
            continue
        _st = scenario_settings(_scen[_n])
        _src = _scen[_n].get("wholesale_from") or _n
        _od = Path(args.out_dir or f"{args.resources}/market")
        _o_csv, _o_json = _od / f"{_n}_pro_rata_turn_down.csv", _od / f"{_n}_pro_rata_turn_down.json"
        run_month(_n, month_paths(_n, _src, args.resources), _st["turn_down_allocation"], _st["roc_bid_rate"],
                  _st["secured_capability"], str(_o_csv), str(_o_json), _st["zones_file"])
        _jsons.append(str(_o_json))
    if args.study:
        _fs = family_settings(_scen[_names[0]])
        if args.mode:
            _fs["turn_down_allocation"]["mode"] = args.mode
        _oa = Path(args.out_dir or f"{args.resources}/analysis")
        run_family(args.study, _jsons, f"{args.resources}/analysis/{args.study}_annual_summary.json",
                   _fs["turn_down_allocation"], str(_oa / f"{args.study}_pro_rata_turn_down_summary.json"),
                   delay_case=_fs["delay_case"], benchmarks_file=_fs["benchmarks_file"])
