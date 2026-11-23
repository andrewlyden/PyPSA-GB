"""Settle a central dispatch case the way FTI settle it, and set it beside its market twin.

FTI's REMA scheduling and dispatch assessment for NESO (February 2026) has consumers
under central dispatch pay a *national* price for energy, plus two make-whole
transfers to units that nodal scheduling strands under that price, plus whatever the
operator spends resolving day-ahead forecast error in real time. Under self-dispatch
consumers pay the wholesale price plus the balancing mechanism. This module computes
both sides from outputs that already exist, so no re-solve is needed.

Two choices are made explicit rather than buried:

**Which national price.** FTI's national price is the unconstrained one, so the default
is the market twin's copperplate price, which makes energy identical on both sides and
lets the comparison isolate dispatch. ``--national-price nodal`` instead uses the
demand-weighted average of the central schedule's own nodal prices; that is the
locational-pricing institution, reported for contrast, and it is far higher because GB
demand sits where nodal prices are high.

**Constrained-off units and must-run units.** Today a curtailed unit keeps the national
price it earned day-ahead for output it never delivers, and a CfD unit recovers its lost
top-up through its balancing bid; the market's energy line plus balancing cost already
contains all of that. Under central dispatch the unit is never scheduled, so FTI's firm
access pays its lost margin instead. Symmetrically, cost recovery is charged only on output
central dispatch runs *above* what self-dispatch would have run: must-run plant carried at
its floor under both institutions is paid the national price under both, and nobody charges
consumers its cost above that price today. Gross cost recovery and the market's
constrained-off lost margin are reported for information and enter no total.

Per central dispatch case, over the chosen dates:

- **national price**: as above;
- **energy**: that price times outturn GB demand;
- **cost recovery**: for every unit whose marginal cost exceeds the national price in
  that hour, the shortfall times the output central dispatch runs above self-dispatch;
- **firm access**: for every GB unit the self-dispatch (copperplate) schedule ran that
  the central schedule did not, the national price less its marginal cost, floored at
  zero, times the lost output; interconnector counterparties are excluded and the
  amount they would have drawn is reported for information;
- **the price of those two transfers** (decision D7, 27 Sep 2026): with a
  self-dispatch-on-forecast case, cost recovery and firm access are valued at *its*
  copperplate price, the price the forecast counterfactual cleared at, because both
  compare two schedules made on the forecast; the energy line keeps the outturn
  national price. ``central_fa_cr_price_basis`` names the price used and the two
  transfers at the energy line's price are reported for information;
- **real-time correction**: the resource cost the outturn dispatch adds over the
  Stage 1 schedule, generators at marginal cost floored at zero *plus* link flows
  at their marginal cost *plus* the interconnector counter-trade premium the
  balancing network was solved with (C11, 27 Sep 2026: the market side carries the
  same premium inside its re-trading line). The link term matters: the import generators carry zero
  marginal cost and the foreign price sits on the interconnector links, so
  counting generators alone values every import at nothing. Supported
  renewables carry negative marginal costs in the network to represent
  subsidy-backed bidding; at cost basis that let the balancing stage book a
  turn-down of wind as income, so the balancing objective's own net is kept
  only for reference and every valuation here uses floored costs.

For the market twin: energy at the copperplate price, the Stage 2 constraint cost, the
**interconnector re-trading cost**, and the lost-margin quantity above.

**Interconnector re-trading (audit 26 Sep 2026).** The market twin's Stage 2 moves the
interconnectors off their day-ahead schedule, and that change is bought and sold at the
foreign bid and ask: 1,484 m a year on the CP30 run, which the consumer bill carries in its
constraint line (``bsuos_constraints.include_interconnector_retrading: true``) and which
NESO's constraint category carries too. It is *not* in the constraint-cost file (the
redispatch net cost counts generators and storage only; the ``EU_market`` carrier's net cost
is zero), so until this audit the market's consumer cost left it out while the central side
counted its own real-time link changes inside the real-time correction, which values
``physical - schedule`` at the same bid and ask (:func:`link_trade_cost`). The market side
was therefore understated by the whole re-trading cost and the saving with it. It is now
its own line, ``market_interconnector_retrading_gbp_m``, valued exactly as
``analyze_market`` values it (:func:`interconnector_retrading_cost`) and computed from the
window's own link flows, so a week settles its own re-trading rather than a pro-rata share.
Nothing counts it twice: the central side's link term is its own schedule change, never the
market's; the resource-cost lines value physical flows, not changes; and :func:`settle`
refuses a constraint cost that already matches the accounting's
``modelled_redispatch_cost_including_retrading_GBP``. On a whole-period run the value is
checked against that accounting figure and the accounting figure is reported beside it.

**Forecast-error charge and the headline.** Our model produces almost no forecast-error
cost (see :mod:`scripts.market.forecast_error_charge`), so the raw saving corresponds to
FTI's *gross* figure. The settlement reports ``consumer_saving_gross_gbp_m``, the borrowed
``forecast_error_charge_gbp_m`` for the same window, and
``consumer_saving_fti_comparable_net_gbp_m``, **the headline** (user decision, 26 Sep 2026;
``headline_metric`` names it): the gross with the model's own real-time correction taken
back out (``central_realtime_correction_replaced_gbp_m``, about -53 m a year) and the
charge deducted, so forecast error is counted once, from FTI's estimate. The gross keeps
the model's correction.

**Interconnector pass-through and the rent transfer (item D4(b), 26 Sep 2026).** The bill
returns part of the cables' congestion rent to consumers through the cap-and-floor regimes
(and tops up a floor). Each side now receives that pass-through on its *own* Stage 1 flows:
the market's day-ahead schedule at the copperplate price, the central schedule at the
national price (``*_ic_pass_through_credit_gbp_m``, a credit lowering the consumer cost).
The rent difference between the two schedules is a transfer from (or to) the cable owners,
not a saving: ``ic_congestion_rent_transfer_info_gbp_m`` with its GB-half and foreign-half
recipients, entering no total. A month's cap and floor are its share of the annual levels;
the year settlement reassesses once on the summed per-cable rents (``market_ic_rent_gbp_m::*``).
Rent uses the corrected export-loss convention: GB gives up ``efficiency x |p0|`` on an export.

**FTI's method, decisions of 14 Oct 2026.** Firm access also pays a turned-down CfD unit
max(0, strike - reference) and a ROC unit its ROC value on the lost day-ahead volume
(``central_firm_access_cfd_support_gbp_m`` / ``..._roc_support_gbp_m``, inside
``central_firm_access_gbp_m``; ``--firm-access-support metered_only`` is the sensitivity).
Storage is made whole week by week to its self-dispatch earnings
(``central_storage_make_whole_gbp_m``) and interconnector capacity holders day by day to
their self-dispatch rent (``central_ic_make_whole_gbp_m``, which joins the central rent
before the regimes). All three are central consumer costs.

**Market constraint basis (user decision, 20 Oct 2026).** ``market_constraint_cost_gbp_m``
is the balancing stage as solved, so the headline compares two least-cost dispatches. Where
a month's turn-down basis re-prices it (pro rata zonal turn-down), the uplift is reported
beside it (``market_constraint_pro_rata_uplift_gbp_m``) with the headline on that basis
(``consumer_saving_fti_comparable_net_in_force_basis_gbp_m``).

**Price basis.** Every money figure here is PyPSA money, which the consumer bill treats as
**2024 prices** and rebases x1.0384 to its 2025 report year; ``price_basis`` says so. The resource cost of each *physical* dispatch is reported alongside, because the
money numbers are transfers and the resource number is not. It covers generators
at floored marginal cost and link flows as the LP priced them (:func:`link_trade_cost`),
the latter not floored because an export earns the foreign price as genuinely as an
import pays it; generator-only and interconnector-only figures are given for information.

Usage::

    python -m scripts.market.fti_settlement --central <scenario> --market <scenario>
        [--start 2030-01-03 --end 2030-01-09] [--national-price copperplate|nodal]
        [--forecast-error-basis renewable_capacity|demand|gross_share] [--no-forecast-error-charge]
        [--assumptions data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json]
        [--out path.csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from scripts.market import forecast_error_charge as fec

_LOG = logging.getLogger(__name__)
MARKET_DIR = Path("resources/market")
ANALYSIS_DIR = Path("resources/analysis")
#: PyPSA money; the consumer bill rebases it x1.0384 (CPIH) to its 2025 report year.
PRICE_BASIS = "GBP real 2024 (PyPSA model money; the consumer bill rebases x1.0384 to 2025 prices)"
HEADLINE_METRIC = "consumer_saving_fti_comparable_net_gbp_m"
#: The same saving with the market's constraint cost on its in-force turn-down basis (20 Oct 2026).
HEADLINE_METRIC_IN_FORCE = "consumer_saving_fti_comparable_net_in_force_basis_gbp_m"
#: Relative gap between the settlement's own re-trading figure and the market
#: accounting's above which the whole-period run warns (the two value the same
#: flows at the same prices, so any gap is a wiring fault, not rounding).
RETRADING_CHECK_TOLERANCE = 0.01
NATIONAL_PRICE_BASES = ("copperplate", "nodal")
# Interconnector counterparties are not GB assets with firm access rights: an
# import the central schedule does not take is a trade that did not happen, and
# the seller sells elsewhere. FTI treat interconnector repositioning as a
# balancing cost, which the real-time correction and the energy line already
# carry, so these carriers earn no firm access. Cost recovery still applies to
# them, because an import taken above the national price is a real purchase.
FIRM_ACCESS_EXCLUDED_CARRIERS = ("EU_market", "EU_import", "EU_export")
# Decision of 14 Oct 2026 (FTI 2025 slides 43-44): firm access also pays a turned-down
# CfD unit its lost top-up, max(0, strike - reference), and a ROC unit its lost ROC
# value, on the lost day-ahead volume. "metered_only" (Ofgem 2023: no constrained-off
# payments) is the sensitivity, a transfer from owners rather than a saving.
FIRM_ACCESS_SUPPORT_BASES = ("lost_support", "metered_only")
DEFAULT_FIRM_ACCESS_SUPPORT = "lost_support"
#: The per-unit CfD terms the market's balancing stage priced its bids on.
CFD_AUDIT_KIND = "contract_value_bids"
#: The resource saving with each side's counter-trade premium counted as paid abroad.
GB_RESOURCE_SAVING = "resource_saving_gb_perspective_gbp_m"
#: Each side's CfD and RO lines on its metered output (decision of 20 Oct 2026).
SUPPORT_LINE_ROWS = ("market_support_lines_cfd_gbp_m", "market_support_lines_roc_gbp_m",
                     "central_support_lines_cfd_gbp_m", "central_support_lines_roc_gbp_m")


# ── pure functions ───────────────────────────────────────────────────────────

def national_price(
    marginal_price: pd.DataFrame, load: pd.DataFrame, load_bus: pd.Series
) -> pd.Series:
    """Demand-weighted average nodal price per snapshot over the given loads."""
    cols = [c for c in load.columns if c in load_bus.index and load_bus[c] in marginal_price.columns]
    if not cols:
        raise ValueError("no loads map onto priced buses")
    w = load[cols].clip(lower=0.0)
    p = marginal_price[load_bus[cols].values]
    p.columns = cols
    num = (w * p).sum(axis=1)
    den = w.sum(axis=1).replace(0.0, np.nan)
    return (num / den).fillna(0.0)


def cost_recovery(dispatch: pd.DataFrame, mc: pd.Series, price: pd.Series) -> pd.Series:
    """Per unit: sum over hours of max(0, mc - price) x output. Units missing a cost are skipped."""
    cols = [c for c in dispatch.columns if c in mc.index]
    short = pd.DataFrame(
        np.maximum(0.0, mc[cols].values[None, :] - price.reindex(dispatch.index).values[:, None]),
        index=dispatch.index, columns=cols,
    )
    return (short * dispatch[cols].clip(lower=0.0)).sum()


def cost_recovery_increment(
    central: pd.DataFrame, selfdispatch: pd.DataFrame, mc: pd.Series, price: pd.Series
) -> pd.Series:
    """Per unit: sum over hours of max(0, mc - price) x max(0, central output - self-dispatch output).

    Only the output central dispatch runs *above* what self-dispatch would have run
    is made whole. Must-run plant carried at its floor under both institutions is
    paid the national price under both, and nobody charges consumers its cost above
    that price today, so it must not be charged here either. Symmetric with
    :func:`firm_access` on the shortfall.
    """
    cols = [c for c in central.columns if c in selfdispatch.columns and c in mc.index]
    idx = central.index.intersection(selfdispatch.index)
    extra = (central.loc[idx, cols] - selfdispatch.loc[idx, cols]).clip(lower=0.0)
    short = pd.DataFrame(
        np.maximum(0.0, mc[cols].values[None, :] - price.reindex(idx).values[:, None]),
        index=idx, columns=cols,
    )
    return (short * extra).sum()


def firm_access(
    central: pd.DataFrame, market: pd.DataFrame, mc: pd.Series, price: pd.Series
) -> pd.Series:
    """Per unit: sum over hours of max(0, price - mc) x max(0, market output - central output)."""
    cols = [c for c in market.columns if c in central.columns and c in mc.index]
    idx = market.index.intersection(central.index)
    lost = (market.loc[idx, cols] - central.loc[idx, cols]).clip(lower=0.0)
    margin = pd.DataFrame(
        np.maximum(0.0, price.reindex(idx).values[:, None] - mc[cols].values[None, :]),
        index=idx, columns=cols,
    )
    return (margin * lost).sum()


def support_terms(generators: pd.DataFrame, cfd_audit: pd.DataFrame | None, roc_buyout: float) -> dict:
    """Per unit: the CfD strike, whether its negative-price rule applies, and the ROC value per MWh.

    ``generators`` is the market network's generator table (``support_type``,
    ``ro_banding``); ``cfd_audit`` the market month's contract-value bid audit
    (``generator``, ``strike_price_gbp_per_mwh``, ``negative_price_rule``), the strikes
    the balancing stage priced its CfD bids on. A ROC unit forgoes ``ro_banding x
    buyout`` per MWh it does not generate, the value its balancing bid recovers.
    """
    strike = pd.Series(dtype=float)
    npr = pd.Series(dtype=bool)
    if cfd_audit is not None and not cfd_audit.empty:
        a = cfd_audit.drop_duplicates("generator").set_index("generator")
        cfd = a[a.get("support_type", pd.Series("CfD", index=a.index)).astype(str).eq("CfD")]
        strike = pd.to_numeric(cfd["strike_price_gbp_per_mwh"], errors="coerce").dropna().astype(float)
        rule = cfd.get("negative_price_rule", pd.Series("none", index=cfd.index)).astype(str).str.lower()
        npr = ~rule.reindex(strike.index).fillna("none").isin(("none", "", "nan"))
    roc = pd.Series(dtype=float)
    if "support_type" in generators.columns and roc_buyout > 0:
        is_roc = generators["support_type"].astype(str).eq("ROC")
        banding = pd.to_numeric(generators.get("ro_banding", pd.Series(np.nan, index=generators.index)),
                                errors="coerce")
        missing = is_roc & ~(banding > 0)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} ROC generators carry no ro_banding, e.g. "
                             f"{list(generators.index[missing][:3])}")
        roc = (banding[is_roc] * float(roc_buyout)).astype(float)
    return {"cfd_strike": strike, "cfd_negative_price_rule": npr, "roc_value": roc}


def book_cfd_terms(terms: Mapping, book: pd.DataFrame, cfd_audit: pd.DataFrame | None = None) -> dict:
    """``terms`` with the CfD part taken from a 2050 contract book (FES 2050 decision 70, 25 Oct 2026).

    A 2050 network generator carries several CfD cohorts at once (``PAR7_<build year>`` rows
    of the vintaged register, each with its own strike and ``energy_factor``, the share of the
    asset's output it is paid on), and the market audit labels such a unit ``mixed`` with one
    capacity-weighted strike, which :func:`support_terms` skips. With a book the CfD terms are
    the book's: ``cfd_cohorts`` (assets x strikes, the summed ``energy_factor`` of each asset's
    rows at each strike), ``cfd_weight`` (its row sum) and ``cfd_strike`` (the weighted strike,
    an index of the paid assets). The negative-price rule is the audit's for the asset, whatever
    its support type; the ROC terms are kept. The strikes are the book's, so a book built on
    another strike table (the central-dispatch twin's) values the lost support at those strikes.
    """
    cfd = book.loc[book["scheme"].astype(str).eq("CfD") & book["component_type"].astype(str).eq("Generator")]
    factor = pd.to_numeric(cfd["energy_factor"], errors="coerce").fillna(0.0)
    strike_values = pd.to_numeric(cfd["strike_gbp_per_mwh_2024"], errors="coerce")
    if strike_values[factor > 0].isna().any():
        raise ValueError("the contract book has CfD rows without a strike (strike_gbp_per_mwh_2024)")
    frame = pd.DataFrame({"asset": cfd["network_asset"].astype(str).to_numpy(), "strike": strike_values.to_numpy(),
                          "factor": factor.to_numpy()})
    frame = frame.loc[frame["factor"] > 0]
    cohorts = (frame.groupby(["asset", "strike"])["factor"].sum().unstack("strike", fill_value=0.0)
               if len(frame) else pd.DataFrame(dtype=float))
    weight = cohorts.sum(axis=1) if len(cohorts) else pd.Series(dtype=float)
    strike = ((cohorts * cohorts.columns.to_numpy(dtype=float)[None, :]).sum(axis=1) / weight
              if len(cohorts) else pd.Series(dtype=float))
    npr = pd.Series(False, index=strike.index)
    if cfd_audit is not None and not cfd_audit.empty and "negative_price_rule" in cfd_audit.columns:
        rule = cfd_audit.drop_duplicates("generator").set_index("generator")["negative_price_rule"]
        rule = rule.astype(str).str.lower().reindex(strike.index).fillna("none")
        npr = ~rule.isin(("none", "", "nan"))
    return {**dict(terms), "cfd_strike": strike.astype(float), "cfd_negative_price_rule": npr.astype(bool),
            "cfd_cohorts": cohorts.astype(float), "cfd_weight": weight.astype(float)}


def _cohort_top_up(cohorts: pd.DataFrame, cols: list, price: pd.Series) -> pd.DataFrame:
    """Hours x ``cols``: sum over each asset's cohorts of ``energy_factor x max(0, strike - price)``."""
    strikes = cohorts.columns.to_numpy(dtype=float)
    per_strike = np.maximum(0.0, strikes[None, :] - price.to_numpy(dtype=float)[:, None])
    return pd.DataFrame(per_strike @ cohorts.loc[cols].to_numpy(dtype=float).T, index=price.index, columns=cols)


def lost_support(central: pd.DataFrame, market: pd.DataFrame, price: pd.Series, terms: Mapping) -> pd.DataFrame:
    """Per unit, the support firm access adds on the lost day-ahead volume (decision of 14 Oct 2026).

    The volume is :func:`firm_access`'s, max(0, market output - central output). A CfD
    unit is paid max(0, strike - price) per lost MWh, nothing in hours below zero where
    its negative-price rule applies; a ROC unit its ROC value. Columns ``cfd`` and
    ``roc``, GBP; units with neither are absent.
    """
    idx = market.index.intersection(central.index)
    price = pd.Series(price, dtype=float).reindex(idx)
    out = {}
    strike, npr = terms["cfd_strike"], terms["cfd_negative_price_rule"]
    cols = [c for c in strike.index if c in market.columns and c in central.columns]
    if cols:
        lost = (market.loc[idx, cols] - central.loc[idx, cols]).clip(lower=0.0)
        if terms.get("cfd_cohorts") is not None:
            # A 2050 contract book (book_cfd_terms): each cohort's own strike, on its share of the output.
            top_up = _cohort_top_up(terms["cfd_cohorts"], cols, price)
        else:
            top_up = pd.DataFrame(np.maximum(0.0, strike[cols].values[None, :] - price.values[:, None]),
                                  index=idx, columns=cols)
        ruled = [c for c in cols if bool(npr.get(c, False))]
        if ruled:
            top_up.loc[price.values < 0, ruled] = 0.0
        out["cfd"] = (top_up * lost).sum()
    roc = terms["roc_value"]
    cols = [c for c in roc.index if c in market.columns and c in central.columns]
    if cols:
        lost = (market.loc[idx, cols] - central.loc[idx, cols]).clip(lower=0.0)
        out["roc"] = lost.mul(roc[cols], axis=1).sum()
    frame = pd.DataFrame(out).fillna(0.0)
    for column in ("cfd", "roc"):
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[["cfd", "roc"]]


def support_on_output(output: pd.DataFrame, price: pd.Series, terms: Mapping) -> pd.DataFrame:
    """Per unit, the CfD top-up and ROC value paid on metered ``output`` (decision of 20 Oct 2026).

    The consumer bill's CfD and RO lines: a CfD unit is paid strike - price per MWh it
    generates (negative when the price is above the strike; nothing in hours below zero
    where its negative-price rule applies), a ROC unit its ROC value. Each side of the
    comparison carries these on its own output at its own energy price, so turn-down paid
    through :func:`lost_support` (or, in the market, through contract-value bids) is offset
    by the top-ups it no longer earns. Columns ``cfd`` and ``roc``, GBP.
    """
    idx = output.index
    price = pd.Series(price, dtype=float).reindex(idx)
    out = {}
    strike, npr = terms["cfd_strike"], terms["cfd_negative_price_rule"]
    cols = [c for c in strike.index if c in output.columns]
    if cols:
        top_up = pd.DataFrame(strike[cols].values[None, :] - price.values[:, None], index=idx, columns=cols)
        if terms.get("cfd_weight") is not None:
            # A 2050 contract book: strike - price is linear, so the cohorts sum to the weighted strike on the paid share.
            top_up = top_up * terms["cfd_weight"][cols].to_numpy(dtype=float)[None, :]
        ruled = [c for c in cols if bool(npr.get(c, False))]
        if ruled:
            top_up.loc[price.values < 0, ruled] = 0.0
        out["cfd"] = (top_up * output[cols].fillna(0.0).clip(lower=0.0)).sum()
    roc = terms["roc_value"]
    cols = [c for c in roc.index if c in output.columns]
    if cols:
        out["roc"] = output[cols].fillna(0.0).clip(lower=0.0).mul(roc[cols], axis=1).sum()
    frame = pd.DataFrame(out).fillna(0.0)
    for column in ("cfd", "roc"):
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[["cfd", "roc"]]


def lost_volume_mwh(central: pd.DataFrame, market: pd.DataFrame, units, weight: pd.Series | None = None) -> float:
    """MWh the central schedule runs ``units`` below self-dispatch, :func:`lost_support`'s volume (unweighted hours).

    ``weight`` (a 2050 contract book's ``cfd_weight``) counts only each unit's paid share.
    """
    cols = [c for c in units if c in market.columns and c in central.columns]
    if not cols:
        return 0.0
    idx = market.index.intersection(central.index)
    lost = (market.loc[idx, cols] - central.loc[idx, cols]).clip(lower=0.0)
    if weight is not None:
        lost = lost * weight.reindex(cols).fillna(0.0).to_numpy(dtype=float)[None, :]
    return float(lost.sum().sum())


def storage_make_whole(central: pd.DataFrame, selfdispatch: pd.DataFrame, price_c: pd.Series,
                       price_sd: pd.Series) -> pd.Series:
    """Per storage unit: the weekly make-whole to its self-dispatch earnings (FTI 2025 slide 42).

    Under central dispatch storage settles at the national price on the operator's
    schedule and is paid, week by week, up to what it would have earned self-dispatching:
    sum over weeks of max(0, earnings on the self-dispatch schedule at its price - earnings
    on the central schedule at the national price). Dispatch is positive, charging
    negative. Weeks are calendar weeks inside the window, so a week a month boundary
    splits is assessed in two parts (a declared approximation).
    """
    cols = [c for c in central.columns if c in selfdispatch.columns]
    idx = central.index.intersection(selfdispatch.index)
    if not cols or not len(idx):
        return pd.Series(dtype=float)
    e_c = central.loc[idx, cols].fillna(0.0).mul(pd.Series(price_c, dtype=float).reindex(idx), axis=0)
    e_sd = selfdispatch.loc[idx, cols].fillna(0.0).mul(pd.Series(price_sd, dtype=float).reindex(idx), axis=0)
    week = pd.DatetimeIndex(idx).to_period("W")
    gap = (e_sd.groupby(week).sum() - e_c.groupby(week).sum()).clip(lower=0.0)
    return gap.sum()


#: The per-unit transfer columns :func:`unit_transfers` writes (GBP), and the settlement row each sums to (GBP m).
UNIT_TRANSFER_ROWS = {
    "cost_recovery_gbp": "central_cost_recovery_gbp_m",
    "firm_access_margin_gbp": "central_firm_access_margin_gbp_m",
    "firm_access_cfd_support_gbp": "central_firm_access_cfd_support_gbp_m",
    "firm_access_roc_support_gbp": "central_firm_access_roc_support_gbp_m",
    "storage_make_whole_gbp": "central_storage_make_whole_gbp_m",
}
UNIT_TRANSFER_COLUMNS = ["component_type", "asset_id", "carrier", *UNIT_TRANSFER_ROWS]


def unit_transfers(cost_recovery_by_unit: pd.Series, firm_access_margin: pd.Series, support: pd.DataFrame,
                   storage_make_whole_by_unit: pd.Series, generator_carrier: pd.Series,
                   storage_carrier: pd.Series | None = None) -> pd.DataFrame:
    """Per unit, the central-dispatch transfers the settlement pays it, GBP (20 Oct 2026).

    Generators carry cost recovery, the firm-access margin and the lost CfD and ROC
    support (``support``, :func:`lost_support`, columns ``cfd`` and ``roc``); storage
    units the weekly make-whole. Each column sums to the settlement row
    :data:`UNIT_TRANSFER_ROWS` names (times 1e6), so an asset settlement on the
    central-dispatch year books exactly the money the bill's constraint line pays.
    Units with nothing are left out.
    """
    support = support if support is not None else pd.DataFrame(columns=["cfd", "roc"], dtype=float)
    generators = pd.DataFrame({
        "cost_recovery_gbp": pd.Series(cost_recovery_by_unit, dtype=float),
        "firm_access_margin_gbp": pd.Series(firm_access_margin, dtype=float),
        "firm_access_cfd_support_gbp": pd.Series(support.get("cfd", pd.Series(dtype=float)), dtype=float),
        "firm_access_roc_support_gbp": pd.Series(support.get("roc", pd.Series(dtype=float)), dtype=float),
    }).fillna(0.0)
    generators["storage_make_whole_gbp"] = 0.0
    generators.insert(0, "component_type", "Generator")
    generators.insert(1, "carrier", pd.Series(generator_carrier).reindex(generators.index).fillna("").astype(str))
    storage = pd.DataFrame({"storage_make_whole_gbp": pd.Series(storage_make_whole_by_unit, dtype=float)}).fillna(0.0)
    for column in ("cost_recovery_gbp", "firm_access_margin_gbp", "firm_access_cfd_support_gbp",
                   "firm_access_roc_support_gbp"):
        storage[column] = 0.0
    storage.insert(0, "component_type", "StorageUnit")
    carriers = pd.Series(storage_carrier) if storage_carrier is not None else pd.Series(dtype=str)
    storage.insert(1, "carrier", carriers.reindex(storage.index).fillna("").astype(str))
    out = pd.concat([generators, storage])
    out.index.name = "asset_id"
    out = out.reset_index()
    amounts = list(UNIT_TRANSFER_ROWS)
    out = out.loc[out[amounts].abs().sum(axis=1).gt(0.0)]
    return out[UNIT_TRANSFER_COLUMNS].reset_index(drop=True)


def check_unit_transfers(units: pd.DataFrame, rows: Mapping[str, float], where: str = "",
                         tolerance_gbp: float = 1.0) -> None:
    """Refuse per-unit transfers that do not sum to the settlement's own rows (GBP m) to ``tolerance_gbp``."""
    for column, row in UNIT_TRANSFER_ROWS.items():
        total = float(pd.to_numeric(units[column], errors="coerce").fillna(0.0).sum()) if column in units else 0.0
        wanted = float(rows[row]) * 1e6
        if abs(total - wanted) > max(tolerance_gbp, 1e-9 * abs(wanted)):
            raise ValueError(f"{where}: per-unit {column} sums to GBP {total:,.2f}, the settlement's {row} to "
                             f"GBP {wanted:,.2f}")


def fa_cr_price(
    energy_price: pd.Series, national_price_basis: str, selfdispatch: str | None,
    selfdispatch_price: pd.Series | None,
) -> tuple[pd.Series, str]:
    """The price firm access and cost recovery are valued at, and its label (decision D7).

    Both transfers compare the central schedule with what self-dispatch would have
    scheduled on the *same forecast*, so they are valued at the price that forecast
    counterfactual cleared at: the self-dispatch-on-forecast copperplate price. The
    outturn copperplate price (the energy line's) belongs to a schedule neither side
    made. Without a self-dispatch-on-forecast case (a perfect-foresight central case,
    whose counterfactual is the market's own copperplate on the outturn) and under the
    nodal basis (the central schedule's own forecast-based prices) the energy line's
    price is used.
    """
    if selfdispatch is None or selfdispatch_price is None:
        return energy_price, "energy line price (no forecast counterfactual)"
    if national_price_basis != "copperplate":
        return energy_price, f"energy line price ({national_price_basis})"
    price = pd.Series(selfdispatch_price, dtype=float).reindex(energy_price.index)
    if price.isna().any():
        raise ValueError(f"{selfdispatch}: wholesale price missing for {int(price.isna().sum())} of "
                         f"{len(price)} settled hours")
    return price, f"{selfdispatch} copperplate price (self-dispatch on the forecast)"


def resource_cost(dispatch: pd.DataFrame, mc: pd.Series) -> float:
    cols = [c for c in dispatch.columns if c in mc.index]
    return float((dispatch[cols].clip(lower=0.0) * mc[cols]).sum().sum())


def unplanned_start_cost(
    phys: pd.DataFrame, sched: pd.DataFrame, premium: float, units: list, threshold_mw: float
) -> float:
    """Cost of starting plant the day-ahead schedule left off: premium x output brought on.

    The commitment proxy prices an unplanned start into the real-time offer; the
    settlement values the central side at cost, so that premium would otherwise
    vanish. It is a real cost (start-up fuel and wear), so it is counted here as
    premium x max(0, outturn - schedule) on unit-hours where the schedule was
    below the committed threshold.
    """
    cols = [u for u in units if u in phys.columns and u in sched.columns]
    if not cols or premium <= 0:
        return 0.0
    idx = phys.index.intersection(sched.index)
    s, p = sched.loc[idx, cols], phys.loc[idx, cols]
    off = s.lt(threshold_mw)
    brought_on = (p - s).clip(lower=0.0).where(off, 0.0)
    return float(premium * brought_on.sum().sum())


def link_trade_cost(p0: pd.DataFrame, links: pd.DataFrame, ask_t: pd.DataFrame | None = None) -> float:
    """Cost of link flows over the hours given.

    Signed interconnectors are valued exactly as the LP charged them
    (``market_coupling.interconnector_trade_cost``): an import pays the foreign
    ask, an export earns efficiency times the foreign bid, and the volume
    tranches are added (item IC1). Before IC1 every signed flow was valued at the
    import ask in both directions, so an export earned 89.93 GBP/MWh where the
    LP paid about 38.6. Other links are flow times their marginal cost. Flow is
    not floored at zero: an export's negative cost is real income. Generators
    alone miss this, because the import generators carry zero marginal cost and
    the trade price sits on the links.
    """
    from scripts.interconnectors.market_coupling import _piecewise_links, interconnector_trade_cost

    signed = [c for c in _piecewise_links(links).values() if c in p0.columns]
    other = [c for c in p0.columns if c in links.index and c not in signed]
    total = float((p0[other] * links.loc[other, "marginal_cost"].astype(float)).sum().sum())
    if signed:
        total += float(interconnector_trade_cost(p0[signed], links, ask_t).to_numpy().sum())
    return total


def interconnector_retrading_cost(phys: pd.DataFrame, sched: pd.DataFrame, links: pd.DataFrame,
                                  ask_t: pd.DataFrame | None = None) -> float:
    """Cost of moving the signed interconnectors off their day-ahead schedule.

    Both schedules are valued as the balancing LP priced trade
    (``market_coupling.interconnector_trade_cost``: import at the ask, export at
    efficiency times the bid, plus the volume tranches) and the day-ahead value is
    taken from the physical one, exactly as ``analyze_market`` does for the
    accounting's ``interconnector_retrading_cost_GBP``. Tranche cost depends on
    the whole market's flow, so each schedule is valued whole. Positive when
    re-trading costs consumers money. The counter-trade premium the balancing
    network was solved with (``market_coupling.counter_trade_premium_cost``) is
    added, as ``analyze_market`` adds it.
    """
    from scripts.interconnectors.market_coupling import (
        _piecewise_links, counter_trade_premium_cost, interconnector_trade_cost,
    )

    signed = [c for c in _piecewise_links(links).values() if c in phys.columns or c in sched.columns]
    if not signed:
        return 0.0
    idx = phys.index
    p = phys.reindex(index=idx, columns=signed).fillna(0.0)
    s = sched.reindex(index=idx, columns=signed).fillna(0.0)
    return float(interconnector_trade_cost(p, links, ask_t).to_numpy().sum()
                 - interconnector_trade_cost(s, links, ask_t).to_numpy().sum()
                 + counter_trade_premium_cost(p, s, links, ask_t).to_numpy().sum())


# ── interconnector congestion rent and the regulated pass-through (item D4(b)) ──

#: Hours in calendar 2030; a window's cap and floor are this share of the annual levels.
YEAR_HOURS = 8760.0
from scripts.utilities.calendar_year import hours_in_year  # noqa: E402  (2040: leap-year windows)
#: The forecast bill's assumptions, whose interconnector_cap_and_floor rule sets each regime.
DEFAULT_ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
MODEL_PRICE_YEAR = 2024
IC_RENT_MARKET_PREFIX = "market_ic_rent_gbp_m::"
IC_RENT_CENTRAL_PREFIX = "central_ic_rent_gbp_m::"


def gb_half_shares(rule: Mapping) -> dict[str, float]:
    """The GB owner's share of each declared cable, from the bill's interconnector rule.

    Cap-and-floor cables and IFA take their ``gb_share_of_rent`` (the 50:50 split of
    each regulated cable); an exempt merchant cable has a GB half too (BritNed,
    ElecLink), at ``default_gb_share_of_rent``; a cable owned and regulated in another
    jurisdiction (East West, Moyle and its Auchencrosh converter, the Isle of Man
    cable) has no GB half. A cable may declare ``gb_share_of_ownership`` to override.
    Used for the GB/foreign split of the congestion-rent transfer (item D4(b)) and
    for the GB halves a nationalising state acquires (item D3).
    """
    default = float(rule.get("default_gb_share_of_rent", 0.5))
    out: dict[str, float] = {}
    for name, spec in (rule.get("cables") or {}).items():
        regime = str(spec.get("regime"))
        if "gb_share_of_ownership" in spec:
            share = float(spec["gb_share_of_ownership"])
        elif regime in ("cap_and_floor", "use_of_revenues_share"):
            share = float(spec.get("gb_share_of_rent", default))
        elif regime == "exempt_merchant":
            share = default
        elif regime == "other_jurisdiction":
            share = 0.0
        else:
            raise ValueError(f"{name}: unknown interconnector regime {regime!r}")
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"{name}: GB share {share} outside [0, 1]")
        out[str(name)] = share
    return out


def congestion_rent_hourly(flows: pd.DataFrame, gb_price: pd.Series, links: pd.DataFrame,
                           ask_t: pd.DataFrame | None = None, weights: pd.Series | None = None) -> pd.DataFrame:
    """Hourly whole-cable congestion rent per signed interconnector, GBP, columns by cable name.

    Rent is the GB price times the energy GB gains less what the LP paid abroad
    (``market_coupling.interconnector_trade_cost``, tranches included). PyPSA's
    link delivers ``efficiency x p0`` at the GB end in *either* direction, so an
    export of ``|p0|`` takes ``efficiency x |p0|`` from GB and the energy term is
    ``efficiency x p0`` both ways (review B9: ``analyze_market`` valued exports at
    ``|p0|``). Columns are cable names, the key the bill's regime table uses.
    """
    from scripts.interconnectors.market_coupling import _piecewise_links, interconnector_trade_cost

    cables = _piecewise_links(links)
    signed = [link for link in cables.values() if link in flows.columns]
    if not signed:
        return pd.DataFrame(index=flows.index)
    f = flows.reindex(columns=signed).fillna(0.0).astype(float)
    price = pd.Series(gb_price, dtype=float).reindex(f.index)
    if price.isna().any():
        raise ValueError(f"GB price missing for {int(price.isna().sum())} of {len(price)} hours of link flow")
    eff = pd.to_numeric(links.loc[signed, "efficiency"], errors="coerce").astype(float)
    trade = interconnector_trade_cost(f, links, ask_t)[signed]
    hourly = f.mul(eff, axis=1).mul(price, axis=0) - trade
    if weights is not None:
        hourly = hourly.mul(pd.Series(weights, dtype=float).reindex(f.index).fillna(1.0), axis=0)
    names = {link: cable for cable, link in cables.items()}
    return hourly.rename(columns=names).astype(float)


def signed_trade_cost(flows: pd.DataFrame, links: pd.DataFrame, ask_t: pd.DataFrame | None = None) -> float:
    """What the signed interconnectors' flows pay abroad over the hours given, GBP (tranches included).

    The trade :func:`congestion_rent_hourly` nets from the GB end's value, so for a Stage 1
    schedule ``GB price x efficiency x p0 = this + the congestion rent``, cable by cable.
    """
    from scripts.interconnectors.market_coupling import _piecewise_links, interconnector_trade_cost

    signed = [link for link in _piecewise_links(links).values() if link in flows.columns]
    if not signed:
        return 0.0
    f = flows.reindex(columns=signed).fillna(0.0).astype(float)
    return float(interconnector_trade_cost(f, links, ask_t)[signed].to_numpy().sum())


def congestion_rent_by_cable(flows: pd.DataFrame, gb_price: pd.Series, links: pd.DataFrame,
                             ask_t: pd.DataFrame | None = None, weights: pd.Series | None = None) -> pd.Series:
    """Whole-cable congestion rent per signed interconnector over the hours given, GBP
    (:func:`congestion_rent_hourly` summed)."""
    hourly = congestion_rent_hourly(flows, gb_price, links, ask_t, weights)
    if hourly.columns.empty:
        return pd.Series(dtype=float)
    return hourly.sum().astype(float)


def interconnector_make_whole(rent_selfdispatch_hourly: pd.DataFrame, rent_central_hourly: pd.DataFrame) -> pd.Series:
    """Per cable: the day-ahead make-whole to capacity holders (FTI 2025 slide 45; decision of 14 Oct 2026).

    Each day, a cable whose central schedule earns less congestion rent than its
    self-dispatch schedule would have is paid the difference: sum over days of
    max(0, self-dispatch rent - central rent). It is paid to the cable's holders, so it
    joins the cable's rent before the regulated pass-through, and consumers pay it.
    """
    cables = sorted(set(rent_selfdispatch_hourly.columns) | set(rent_central_hourly.columns))
    if not cables:
        return pd.Series(dtype=float)
    idx = rent_selfdispatch_hourly.index.union(rent_central_hourly.index)
    sd = rent_selfdispatch_hourly.reindex(index=idx, columns=cables).fillna(0.0)
    c = rent_central_hourly.reindex(index=idx, columns=cables).fillna(0.0)
    day = pd.DatetimeIndex(idx).normalize()
    return (sd.groupby(day).sum() - c.groupby(day).sum()).clip(lower=0.0).sum().astype(float)


def scaled_regime_rule(rule: Mapping, year_share: float) -> dict:
    """The regime rule with every cap and floor scaled to a window's share of the year.

    The regimes assess a year; a month settled on its own gets a twelfth-sized cap
    and floor, a declared approximation that the year settlement replaces with one
    annual assessment on the summed rents (:func:`year_pass_through`).
    """
    import copy

    out = copy.deepcopy(dict(rule))
    for spec in (out.get("cables") or {}).values():
        for key in ("cap_gbp_m", "floor_gbp_m"):
            if key in spec and spec[key] is not None:
                spec[key] = float(spec[key]) * float(year_share)
    return out


def interconnector_regime_pass_through(rent_by_cable: Mapping[str, float], rule: Mapping,
                                       year_share: float = 1.0, model_price_year: int = MODEL_PRICE_YEAR) -> dict:
    """What the regulated regimes hand back to (positive) or take from GB consumers, GBP.

    Wraps ``project_future_bill.interconnector_pass_through`` (the bill's own rule)
    with the cap and floor scaled to ``year_share`` of a year.
    """
    from scripts.consumer_bill.project_future_bill import interconnector_pass_through

    return interconnector_pass_through({k: float(v) for k, v in rent_by_cable.items()},
                                       scaled_regime_rule(rule, min(1.0, float(year_share))), model_price_year)


def load_interconnector_rule(assumptions_path: str | Path | None = None) -> dict:
    """The bill's ``model_linked_rules.interconnector_cap_and_floor`` block."""
    path = Path(assumptions_path or DEFAULT_ASSUMPTIONS)
    rule = json.loads(path.read_text(encoding="utf-8"))["model_linked_rules"]["interconnector_cap_and_floor"]
    if not rule.get("cables"):
        raise ValueError(f"{path}: interconnector_cap_and_floor declares no cables")
    return rule


def interconnector_rent_rows(rent_m: pd.Series, rent_c: pd.Series, rule: Mapping, hours: float,
                            year_hours: float = YEAR_HOURS) -> dict:
    """The D4(b) settlement rows: rents, each side's pass-through, and the rent transfer.

    Each side's pass-through is its own Stage 1 rent put through the regimes: the
    market's day-ahead schedule at the copperplate price, central dispatch's
    schedule at the national price. The rent difference is a transfer between
    cable owners and consumers, not a saving: the part the regimes return enters
    the consumer costs through the two credits, the rest stays with the owners and
    is split here between the GB and the foreign halves. It enters no total.
    """
    shares = gb_half_shares(rule)
    unknown = sorted((set(rent_m.index) | set(rent_c.index)) - set(shares))
    if unknown:
        raise ValueError(f"interconnectors with rent but no declared regime: {unknown}")
    share = min(1.0, float(hours) / float(year_hours))
    pt_m = interconnector_regime_pass_through(rent_m, rule, share)
    pt_c = interconnector_regime_pass_through(rent_c, rule, share)
    cables = sorted(set(rent_m.index) | set(rent_c.index))
    delta = pd.Series({c: float(rent_m.get(c, 0.0)) - float(rent_c.get(c, 0.0)) for c in cables}, dtype=float)
    gb = pd.Series({c: shares[c] for c in cables}, dtype=float)
    out = {
        "market_ic_congestion_rent_gbp_m": float(rent_m.sum()) / 1e6,
        "central_ic_congestion_rent_gbp_m": float(rent_c.sum()) / 1e6,
        "market_ic_pass_through_credit_gbp_m": pt_m["credit_to_consumers_gbp"] / 1e6,
        "central_ic_pass_through_credit_gbp_m": pt_c["credit_to_consumers_gbp"] / 1e6,
        "ic_pass_through_year_share": share,
        # Transfer, not saving: market day-ahead rent less the central schedule's.
        "ic_congestion_rent_transfer_info_gbp_m": float(delta.sum()) / 1e6,
        "ic_congestion_rent_transfer_gb_half_info_gbp_m": float((delta * gb).sum()) / 1e6,
        "ic_congestion_rent_transfer_foreign_half_info_gbp_m": float((delta * (1.0 - gb)).sum()) / 1e6,
        "ic_congestion_rent_transfer_via_regimes_to_consumers_info_gbp_m": (
            pt_m["credit_to_consumers_gbp"] - pt_c["credit_to_consumers_gbp"]) / 1e6,
    }
    for cable in cables:
        out[f"{IC_RENT_MARKET_PREFIX}{cable}"] = float(rent_m.get(cable, 0.0)) / 1e6
        out[f"{IC_RENT_CENTRAL_PREFIX}{cable}"] = float(rent_c.get(cable, 0.0)) / 1e6
    return out


def year_pass_through(table: pd.DataFrame, rule: Mapping, hours: float, year_hours: float = YEAR_HOURS) -> dict:
    """One annual regime assessment on the per-cable rents summed over the settled months."""
    def summed(prefix):
        rows = [r for r in table.index if str(r).startswith(prefix)]
        return pd.Series({r[len(prefix):]: pd.to_numeric(table.loc[r], errors="coerce").sum() * 1e6 for r in rows},
                         dtype=float)

    return interconnector_rent_rows(summed(IC_RENT_MARKET_PREFIX), summed(IC_RENT_CENTRAL_PREFIX), rule, hours,
                                    year_hours=year_hours)


def check_retrading_not_in_constraint_cost(constraint_net_gbp: float, accounting: Mapping | None) -> None:
    """Refuse a constraint cost that already carries the re-trading cost.

    The settlement adds re-trading to the market's consumer cost as its own line
    because the constraint-cost file holds generator and storage redispatch
    only. If that file ever starts to include re-trading (its total matching the
    accounting's including-re-trading figure rather than the excluding one), adding
    it again would count it twice, so this raises instead.
    """
    if not accounting:
        return
    excl = accounting.get("modelled_redispatch_cost_GBP")
    incl = accounting.get("modelled_redispatch_cost_including_retrading_GBP")
    retrading = accounting.get("interconnector_retrading_cost_GBP")
    if excl is None or incl is None or not retrading:
        return
    scale = max(1.0, abs(float(retrading)))
    if abs(constraint_net_gbp - float(incl)) < 1e-6 * scale and abs(constraint_net_gbp - float(excl)) > 1e-6 * scale:
        raise ValueError("the constraint cost already includes interconnector re-trading; adding the "
                         "re-trading line would count it twice")


def market_accounting(scenario: str) -> dict | None:
    """The market run's accounting block, or None when the summary is not there."""
    path = ANALYSIS_DIR / f"{scenario}_market_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("accounting") or None


# ── IO ───────────────────────────────────────────────────────────────────────

def _csv(scenario: str, kind: str, start, end) -> pd.DataFrame:
    df = pd.read_csv(MARKET_DIR / f"{scenario}_{kind}.csv", index_col=0, parse_dates=True)
    return df.loc[start:end]


def _network(scenario: str, stage: str):
    import pypsa  # local import: slow, and the pure functions must not need it

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pypsa.Network(str(MARKET_DIR / f"{scenario}_{stage}.nc"))


def _constraint_total(scenario: str) -> pd.Series:
    """The TOTAL row, on the month's turn-down basis (scripts/market/turn_down_allocation.py): the pro rata
    zonal uplift on the wind bid cost added to ``bid_cost`` and ``net_cost`` where that basis is in force (a
    central-dispatch year has no re-pricing and reads as solved). The hourly dispatch is left as solved:
    re-sharing conserves every zone-hour's wind output, so the resource cost does not move."""
    from scripts.market.turn_down_allocation import bid_cost_uplift_in_force

    path = MARKET_DIR / f"{scenario}_constraint_costs.csv"
    total = pd.read_csv(path, index_col=0).loc["TOTAL"].copy()
    uplift = bid_cost_uplift_in_force(path)
    total["net_cost_as_solved"] = float(total["net_cost"])
    if uplift:
        for column in ("bid_cost", "net_cost"):
            total[column] = float(total[column]) + uplift
    return total


def _gb_loads(network) -> tuple[pd.DataFrame, pd.Series]:
    """GB demand by Load, plus what consumer flexibility draws (by unit, at its bus; F5, 27 Oct 2026)."""
    from scripts.market.market_accounting import _network_flex_draw

    load = network.loads_t.p_set
    bus = network.loads["bus"]
    flex = _network_flex_draw(network, network.snapshots)
    if len(flex.columns):
        load = pd.concat([load, flex], axis=1)
        bus = pd.concat([bus, network.storage_units.loc[flex.columns, "bus"]])
    if "country" in network.buses.columns:
        gb = set(network.buses.index[network.buses["country"].eq("GB")])
        keep = [c for c in load.columns if bus.get(c) in gb]
        load = load[keep]
    return load, bus


def _window_share(scen: str, sched: pd.DataFrame, phys: pd.DataFrame, stage_1: str | None = None) -> float:
    """Share of the scenario's whole-period redispatch volume that falls in the window.

    ``stage_1`` names the month whose Stage 1 schedule the scenario's balancing stage read
    (its ``wholesale_from`` parent); default the scenario itself.
    """
    full = _csv(scen, "balancing_dispatch", None, None)
    full_s = _csv(stage_1 or scen, "wholesale_dispatch", None, None).reindex(full.index)
    cols = [c for c in full.columns if c in full_s.columns and c in sched.columns]
    tot = (full[cols] - full_s[cols]).abs().sum().sum()
    win = (phys[cols] - sched[cols]).abs().sum().sum()
    return win / tot if tot else 1.0


def settle(
    central: str,
    market: str,
    start=None,
    end=None,
    national_price_basis="copperplate",
    selfdispatch: str | None = None,
    logger=None,
    forecast_error: Mapping | None = None,
    assumptions: str | Path | None = None,
    market_wholesale: str | None = None,
    firm_access_support: str = DEFAULT_FIRM_ACCESS_SUPPORT,
    units: bool = False,
    cfd_book: pd.DataFrame | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Settle ``central`` against ``market`` over the window.

    ``cfd_book`` (FES 2050 decision 70, 25 Oct 2026): a 2050 contract book whose CfD cohorts and
    strikes replace the audit's (:func:`book_cfd_terms`); ``None`` (every 2030 settlement) keeps them.

    ``units`` (20 Oct 2026) also returns :func:`unit_transfers`, each GB unit's cost
    recovery, firm access (margin and lost support) and storage make-whole, whose sums
    are the settlement's own rows.

    ``firm_access_support`` (decision of 14 Oct 2026): ``lost_support`` adds each
    turned-down CfD unit's lost top-up and each ROC unit's lost ROC value to firm
    access (:func:`lost_support`); ``metered_only`` leaves them out, the sensitivity in
    which owners absorb them.

    ``market_wholesale`` names the month whose Stage 1 files (schedule, links and
    copperplate price) the market month's balancing stage read: its ``wholesale_from``
    parent (18 Oct 2026: the ``cp30_network`` months own no Stage 1 files, ``stage1``'s
    are theirs). Default the market month itself.

    ``assumptions`` is the forecast bill's assumptions JSON, whose interconnector
    cap-and-floor rule gives each side its regulated pass-through (item D4(b));
    default :data:`DEFAULT_ASSUMPTIONS`.

    ``forecast_error`` is the borrowed forecast-error charge's settings
    (:func:`scripts.market.forecast_error_charge.settings_from_config`); ``None``
    takes the defaults and ``{"enabled": False}`` reports a zero charge, so the net
    equals the gross.

    ``selfdispatch`` names a scenario whose Stage 1 schedule stands in for what
    self-dispatch would have done *on the same forecast* the central case saw;
    FTI's firm-access transfer is defined against that, and without it a
    forecast-error case is paid for forecast differences rather than network
    stranding. When omitted the market twin's copperplate schedule (on the
    outturn) is used, which is exact for a perfect-foresight case.
    """
    if national_price_basis not in NATIONAL_PRICE_BASES:
        raise ValueError(f"national_price_basis must be one of {NATIONAL_PRICE_BASES}")
    if firm_access_support not in FIRM_ACCESS_SUPPORT_BASES:
        raise ValueError(f"firm_access_support must be one of {FIRM_ACCESS_SUPPORT_BASES}")
    log = logger or _LOG
    fe = fec.settings_from_config(forecast_error)
    start = pd.Timestamp(start) if start else None
    end = pd.Timestamp(end) + pd.Timedelta(hours=23) if end else None
    # Captured before anything else is named: until 26 Sep 2026 the unplanned
    # start cost below reused the name ``start``, so "is this a window?" was
    # answered by whether any start cost had been incurred.
    windowed = bool(start is not None or end is not None)

    log.info("loading Stage 1 network for %s", central)
    n_c = _network(central, "wholesale")
    mp = n_c.buses_t.marginal_price.loc[start:end]
    # Supported renewables carry negative marginal costs in this network, which
    # represents subsidy-backed bidding, not resource cost. Every valuation here
    # is a resource or lost-margin quantity, so costs are floored at zero:
    # spilling wind costs nothing real and earns nothing real.
    mc = n_c.generators["marginal_cost"].clip(lower=0.0)

    log.info("loading Stage 2 networks (outturn demand and physical link flows)")
    n_m = _network(market, "balancing")
    n_cb = _network(central, "balancing")
    from scripts.interconnectors.market_coupling import link_price_tables

    # Each case is valued on its own network's link prices; both share one build.
    tables_m, tables_c = link_price_tables(n_m), link_price_tables(n_cb)
    is_ic = (n_m.links["bus0"].astype(str).str.contains("External")
             | n_m.links["bus1"].astype(str).str.contains("External"))
    ic_links = set(n_m.links.index[is_ic])
    load, bus = _gb_loads(n_m)
    load = load.loc[start:end]
    demand = load.sum(axis=1)
    idx = mp.index.intersection(demand.index)
    mp, load, demand = mp.loc[idx], load.loc[idx], demand.loc[idx]

    stage_1_m = market_wholesale or market
    price_m = _csv(stage_1_m, "wholesale_price", start, end)["wholesale_price"].reindex(idx)
    price_nodal = national_price(mp, load, bus)
    price_c = price_m if national_price_basis == "copperplate" else price_nodal

    sched_c = _csv(central, "wholesale_dispatch", start, end).reindex(idx)
    sched_m = _csv(stage_1_m, "wholesale_dispatch", start, end).reindex(idx)
    phys_c = _csv(central, "balancing_dispatch", start, end).reindex(idx)
    phys_m = _csv(market, "balancing_dispatch", start, end).reindex(idx)

    # Link flows: schedules from the links CSV, physical from the Stage 2 network
    lk_sched_c = _csv(central, "wholesale_links", start, end).reindex(idx)
    lk_sched_m = _csv(stage_1_m, "wholesale_links", start, end).reindex(idx)
    lk_phys_c = n_cb.links_t.p0.loc[start:end].reindex(idx)
    lk_phys_m = n_m.links_t.p0.loc[start:end].reindex(idx)
    def ltc(df, tables):
        return link_trade_cost(df.fillna(0.0), *tables)

    def ltc_ic(df, tables):
        return link_trade_cost(df[[c for c in df.columns if c in ic_links]].fillna(0.0), *tables)

    # what self-dispatch would have scheduled: on the same forecast when we
    # have it, otherwise the market's copperplate schedule on the outturn
    sched_sd = _csv(selfdispatch, "wholesale_dispatch", start, end).reindex(idx) if selfdispatch else sched_m
    # Decision D7 (27 Sep 2026): firm access and cost recovery value a schedule
    # made on the forecast, so they are priced at the price that forecast
    # counterfactual cleared at (the self-dispatch-on-forecast copperplate price),
    # not at the outturn copperplate price the energy line uses.
    price_fa, fa_price_basis = fa_cr_price(price_c, national_price_basis, selfdispatch,
                                           _csv(selfdispatch, "wholesale_price", start, end)["wholesale_price"]
                                           if selfdispatch else None)

    cr = cost_recovery_increment(sched_c, sched_sd, mc, price_fa)

    # Unplanned starts under the commitment proxy, read from the central
    # scenario's own configuration so the settlement cannot be run with a
    # premium the solve did not use.
    start_cost = 0.0
    start_premium = 0.0
    config_all = None
    try:
        # config/ on the path, as the Snakefile has it. Without it (when
        # config_loader still imported a sibling module) every run logged an
        # import error and read no proxy, so a proxy-enabled scenario would have
        # settled with no start cost.
        config_dir = str(Path(__file__).resolve().parents[2] / "config")
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
        from config.config_loader import get_scenario, load_config
        config_all = load_config()
        cp = (get_scenario(central, config_all).get("market", {}).get("balancing", {})
              .get("commitment_proxy", {}) or {})
    except Exception as exc:  # pragma: no cover - configuration lookup only
        log.warning("commitment proxy config not readable for %s: %s", central, exc)
        cp = {}
    if cp.get("enabled", False):
        units = [g for g in n_c.generators.index if n_c.generators.at[g, "carrier"] in set(cp.get("carriers", []))]
        start_premium = float(cp.get("start_premium_gbp_mwh", 0.0))
        start_cost = unplanned_start_cost(phys_c, sched_c, start_premium, units,
                                          float(cp.get("committed_threshold_mw", 1.0)))
    cr_gross = cost_recovery(sched_c, mc, price_fa)
    fa_all = firm_access(sched_c, sched_sd, mc, price_fa)
    carrier = n_c.generators["carrier"].reindex(fa_all.index)
    ic_mask = carrier.isin(FIRM_ACCESS_EXCLUDED_CARRIERS) | carrier.astype(str).str.startswith("EU_")
    fa_margin = fa_all[~ic_mask]
    fa_ic = fa_all[ic_mask]
    # Decision of 14 Oct 2026: firm access also pays the support the lost volume forgoes
    # (GB units only; the strikes the market's balancing stage bid on, the ROC rate its
    # ROC bids used).
    support = pd.DataFrame(columns=["cfd", "roc"], dtype=float)
    audit_path = MARKET_DIR / f"{market}_{CFD_AUDIT_KIND}.csv"
    terms = None
    if firm_access_support == "lost_support" or audit_path.exists():
        if not audit_path.exists():
            raise FileNotFoundError(f"{audit_path}: firm access pays lost CfD support on the strikes the market's "
                                    "balancing stage bid on; pass firm_access_support='metered_only' to settle without")
        if config_all is None:
            raise RuntimeError("the configuration is unreadable, so the ROC value firm access pays is unknown")
        from config.config_loader import get_scenario
        from scripts.market.turn_down_allocation import roc_bid_rate

        audit = pd.read_csv(audit_path)
        terms = support_terms(n_m.generators, audit, roc_bid_rate(get_scenario(market, config_all)) or 0.0)
        if cfd_book is not None:
            terms = book_cfd_terms(terms, cfd_book, audit)

    def gb_units(frame: pd.DataFrame) -> pd.DataFrame:
        units_carrier = n_c.generators["carrier"].reindex(frame.index).astype(str)
        return frame[~(units_carrier.isin(FIRM_ACCESS_EXCLUDED_CARRIERS) | units_carrier.str.startswith("EU_"))]

    if firm_access_support == "lost_support":
        support = gb_units(lost_support(sched_c, sched_sd, price_fa, terms))
        # The lost day-ahead volume those payments are on (20 Oct 2026: the I3 bill's CfD line reports it).
        lost_cfd_mwh = lost_volume_mwh(sched_c, sched_sd, terms["cfd_strike"].index.intersection(support.index),
                                       terms.get("cfd_weight"))
        lost_roc_mwh = lost_volume_mwh(sched_c, sched_sd, terms["roc_value"].index.intersection(support.index))
    else:
        lost_cfd_mwh = lost_roc_mwh = 0.0
    fa = fa_margin.add(support.sum(axis=1), fill_value=0.0)
    # Decision of 20 Oct 2026: each side's CfD and RO lines on its own metered output, so
    # the comparison credits the top-ups turned-down output no longer earns.
    no_support = pd.DataFrame(columns=["cfd", "roc"], dtype=float)
    support_lines_m = gb_units(support_on_output(phys_m, price_m, terms)) if terms is not None else no_support
    support_lines_c = gb_units(support_on_output(phys_c, price_c, terms)) if terms is not None else no_support
    # Decision of 14 Oct 2026 (FTI 2025 slide 42): storage settles at the national price on
    # the central schedule and is made whole, week by week, to its self-dispatch earnings.
    store_c = _csv(central, "wholesale_storage", start, end).reindex(idx)
    store_sd = _csv(selfdispatch or stage_1_m, "wholesale_storage", start, end).reindex(idx)
    # Consumer flexibility is demand, not storage plant: no make-whole (F5, 27 Oct 2026).
    from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS
    flex_units = set(n_m.storage_units.index[n_m.storage_units["carrier"].astype(str).isin(DEMAND_FLEX_STORAGE_CARRIERS)])
    store_c = store_c[[c for c in store_c.columns if c not in flex_units]]
    store_sd = store_sd[[c for c in store_sd.columns if c not in flex_units]]
    storage_mw = storage_make_whole(store_c, store_sd, price_c, price_fa)
    # The same two transfers at the energy line's price, for continuity with
    # settlements written before D7; information only.
    fa_at_c = firm_access(sched_c, sched_sd, mc, price_c)
    fa_at_c = fa_at_c[~ic_mask.reindex(fa_at_c.index).fillna(False).astype(bool)]
    cr_at_c = cost_recovery_increment(sched_c, sched_sd, mc, price_c)
    # C11: the counter-trade premium the central balancing network was solved with
    # is part of what correcting the schedule costs (a resource cost, as on the
    # market side, where it sits inside the re-trading line).
    from scripts.interconnectors.market_coupling import counter_trade_premium_cost

    ct_premium_c = float(counter_trade_premium_cost(lk_phys_c.fillna(0.0), lk_sched_c.fillna(0.0),
                                                    tables_c[0], tables_c[1]).to_numpy().sum())
    # The market's premium on the same rule (it sits inside its re-trading line).
    ct_premium_m = float(counter_trade_premium_cost(lk_phys_m.fillna(0.0), lk_sched_m.fillna(0.0),
                                                    tables_m[0], tables_m[1]).to_numpy().sum())
    # Information only: the lost-margin quantity on the units the market's
    # balancing stage turns down. It is NOT added to the market's consumer cost,
    # because the market's energy line already pays those units the national
    # price for output they never deliver; that is how they are made whole today.
    off_m = firm_access(phys_m, sched_m, mc, price_m)
    cc_c, cc_m = _constraint_total(central), _constraint_total(market)
    # The market's interconnector re-trading, from the window's own flows (see the
    # module docstring). On a whole-period run it is checked against the market
    # accounting, which values the same flows at the same prices.
    retrading_m = interconnector_retrading_cost(lk_phys_m, lk_sched_m, *tables_m)
    accounting_m = market_accounting(market)
    check_retrading_not_in_constraint_cost(float(cc_m["net_cost_as_solved"]), accounting_m)
    retrading_accounting = (accounting_m or {}).get("interconnector_retrading_cost_GBP")
    if not windowed and retrading_accounting is not None:
        gap = retrading_m - float(retrading_accounting)
        if abs(gap) > RETRADING_CHECK_TOLERANCE * max(1.0, abs(float(retrading_accounting))):
            log.warning("%s: re-trading %.1f m here against %.1f m in the market accounting",
                        market, retrading_m / 1e6, float(retrading_accounting) / 1e6)
    share_c = _window_share(central, sched_c, phys_c) if windowed else 1.0
    share_m = _window_share(market, sched_m, phys_m, stage_1_m) if windowed else 1.0

    dsum = float(demand.sum())
    r = {
        "price_basis": PRICE_BASIS,
        "headline_metric": HEADLINE_METRIC,
        "national_price_basis": national_price_basis,
        "firm_access_counterfactual": selfdispatch or f"{stage_1_m} (copperplate on outturn)",
        "hours": len(idx),
        "gb_demand_twh": dsum / 1e6,
        "market_copperplate_price_gbp_mwh": float((price_m * demand).sum() / dsum),
        "central_nodal_demand_weighted_price_gbp_mwh": float((price_nodal * demand).sum() / dsum),
        "central_national_price_used_gbp_mwh": float((price_c * demand).sum() / dsum),
        "market_energy_gbp_m": float((price_m * demand).sum() / 1e6),
        # User decision, 20 Oct 2026: as solved here; the in-force (pro rata) basis beside it.
        "market_constraint_cost_gbp_m": float(cc_m["net_cost_as_solved"] * share_m / 1e6),
        "market_constraint_cost_in_force_gbp_m": float(cc_m["net_cost"] * share_m / 1e6),
        "market_constraint_pro_rata_uplift_gbp_m": float(
            (float(cc_m["net_cost"]) - float(cc_m["net_cost_as_solved"])) * share_m / 1e6),
        "market_interconnector_retrading_gbp_m": retrading_m / 1e6,
        "market_interconnector_retrading_accounting_info_gbp_m": (
            float(retrading_accounting) / 1e6 if retrading_accounting is not None and not windowed else np.nan),
        "market_constrained_off_lost_margin_info_gbp_m": float(off_m.sum() / 1e6),
        "market_redispatch_volume_twh": float((cc_m["increase_MWh"] + cc_m["decrease_MWh"]) * share_m / 2e6),
        "central_energy_gbp_m": float((price_c * demand).sum() / 1e6),
        "central_fa_cr_price_basis": fa_price_basis,
        "central_fa_cr_price_gbp_mwh": float((price_fa * demand).sum() / dsum),
        "central_cost_recovery_gbp_m": float(cr.sum() / 1e6),
        "central_cost_recovery_gross_info_gbp_m": float(cr_gross.sum() / 1e6),
        "central_firm_access_gbp_m": float(fa.sum() / 1e6),
        "firm_access_support_basis": firm_access_support,
        "central_firm_access_margin_gbp_m": float(fa_margin.sum() / 1e6),
        "central_firm_access_cfd_support_gbp_m": float(support["cfd"].sum() / 1e6),
        "central_firm_access_roc_support_gbp_m": float(support["roc"].sum() / 1e6),
        "central_firm_access_cfd_lost_twh": lost_cfd_mwh / 1e6,
        "central_firm_access_roc_lost_twh": lost_roc_mwh / 1e6,
        "central_storage_make_whole_gbp_m": float(storage_mw.sum() / 1e6),
        "support_lines_basis": "metered output" if terms is not None else "not available (no CfD bid audit)",
        "market_support_lines_cfd_gbp_m": float(support_lines_m["cfd"].sum() / 1e6),
        "market_support_lines_roc_gbp_m": float(support_lines_m["roc"].sum() / 1e6),
        "central_support_lines_cfd_gbp_m": float(support_lines_c["cfd"].sum() / 1e6),
        "central_support_lines_roc_gbp_m": float(support_lines_c["roc"].sum() / 1e6),
        "central_firm_access_interconnectors_excluded_info_gbp_m": float(fa_ic.sum() / 1e6),
        "central_firm_access_at_energy_price_info_gbp_m": float(fa_at_c.sum() / 1e6),
        "central_cost_recovery_at_energy_price_info_gbp_m": float(cr_at_c.sum() / 1e6),
        # The cost of correcting the schedule in real time is the resource cost
        # the outturn dispatch adds over the schedule, both at floored marginal
        # cost. It does not depend on the balancing stage's bid and offer sign
        # conventions, which under cost-basis pricing can make a turn-down of
        # negative-cost wind look like income. The balancing objective's own net
        # is kept beside it for reference.
        "central_realtime_correction_gbp_m": (
            (resource_cost(phys_c, mc) + ltc(lk_phys_c, tables_c) + start_cost + ct_premium_c)
            - (resource_cost(sched_c, mc) + ltc(lk_sched_c, tables_c))
        ) / 1e6,
        "central_unplanned_start_cost_gbp_m": start_cost / 1e6,
        "central_counter_trade_premium_gbp_m": ct_premium_c / 1e6,
        "market_counter_trade_premium_gbp_m": ct_premium_m / 1e6,
        "central_start_premium_gbp_mwh": start_premium,
        "central_realtime_bm_net_gbp_m": float(cc_c["net_cost"] * share_c / 1e6),
        "central_realtime_volume_twh": float((cc_c["increase_MWh"] + cc_c["decrease_MWh"]) * share_c / 2e6),
        "market_resource_cost_gbp_m": (resource_cost(phys_m, mc) + ltc(lk_phys_m, tables_m)) / 1e6,
        "central_resource_cost_gbp_m": (resource_cost(phys_c, mc) + ltc(lk_phys_c, tables_c) + start_cost) / 1e6,
        "central_schedule_resource_cost_gbp_m": (resource_cost(sched_c, mc) + ltc(lk_sched_c, tables_c)) / 1e6,
        "market_resource_cost_generators_only_info_gbp_m": resource_cost(phys_m, mc) / 1e6,
        "central_resource_cost_generators_only_info_gbp_m": resource_cost(phys_c, mc) / 1e6,
        "market_link_trade_cost_gbp_m": ltc(lk_phys_m, tables_m) / 1e6,
        "central_link_trade_cost_gbp_m": ltc(lk_phys_c, tables_c) / 1e6,
        "central_schedule_link_trade_cost_gbp_m": ltc(lk_sched_c, tables_c) / 1e6,
        "market_interconnector_trade_cost_info_gbp_m": ltc_ic(lk_phys_m, tables_m) / 1e6,
        "central_interconnector_trade_cost_info_gbp_m": ltc_ic(lk_phys_c, tables_c) / 1e6,
        "window_share_of_market_redispatch": share_m,
        "window_share_of_central_redispatch": share_c,
    }
    # Item D4(b): each side's regulated interconnector pass-through on its own
    # Stage 1 flows, and the congestion-rent difference as a transfer line.
    rule = load_interconnector_rule(assumptions)
    rent_m_h = congestion_rent_hourly(lk_sched_m, price_m, *tables_m)
    rent_c_h = congestion_rent_hourly(lk_sched_c, price_c, *tables_c)
    # Decision of 14 Oct 2026 (FTI 2025 slide 45): capacity holders are made whole day
    # by day to the rent their self-dispatch schedule would have earned; the payment is
    # the cables' revenue, so it joins the central rent before the regimes.
    rent_sd_h = (congestion_rent_hourly(_csv(selfdispatch, "wholesale_links", start, end).reindex(idx),
                                        price_fa, *tables_m) if selfdispatch else rent_m_h)
    ic_mw = interconnector_make_whole(rent_sd_h, rent_c_h)
    rent_m = rent_m_h.sum().astype(float) if not rent_m_h.columns.empty else pd.Series(dtype=float)
    rent_c = rent_c_h.sum().astype(float) if not rent_c_h.columns.empty else pd.Series(dtype=float)
    r["central_ic_make_whole_gbp_m"] = float(ic_mw.sum() / 1e6)
    r["central_ic_congestion_rent_before_make_whole_info_gbp_m"] = float(rent_c.sum() / 1e6)
    # P1/P2 under central dispatch (20 Oct 2026): what each schedule pays abroad for its cross-border
    # energy, on exactly the trade the congestion rent nets (signed cables, tranches included), so the
    # GB end of the Stage 1 trade splits into the payment abroad and the rent. Information only.
    r["market_ic_schedule_trade_cost_info_gbp_m"] = signed_trade_cost(lk_sched_m, *tables_m) / 1e6
    r["central_ic_schedule_trade_cost_info_gbp_m"] = signed_trade_cost(lk_sched_c, *tables_c) / 1e6
    r["central_ic_physical_trade_cost_info_gbp_m"] = signed_trade_cost(lk_phys_c, *tables_c) / 1e6
    # 2040 (11 Nov 2026): a window's share of its own calendar year (8,784 hours in a leap year).
    year_hours = float(hours_in_year(idx[0].year)) if len(idx) else YEAR_HOURS
    r.update(interconnector_rent_rows(rent_m, rent_c.add(ic_mw, fill_value=0.0), rule, hours=len(idx),
                                      year_hours=year_hours))
    r.update(consumer_totals(r, fe, hours=len(idx), year_hours=year_hours))
    r["resource_saving_gbp_m"] = r["market_resource_cost_gbp_m"] - r["central_resource_cost_gbp_m"]
    r[GB_RESOURCE_SAVING] = gb_resource_saving(r)

    out = pd.DataFrame.from_dict(r, orient="index", columns=["value"])
    out.index.name = "metric"
    log.info("largest cost-recovery recipients (GBP m): %s", (cr.sort_values(ascending=False).head(6) / 1e6).round(2).to_dict())
    log.info("largest firm-access recipients (GBP m): %s", (fa.sort_values(ascending=False).head(6) / 1e6).round(2).to_dict())
    log.info("largest market constrained-off lost margins, information only (GBP m): %s", (off_m.sort_values(ascending=False).head(6) / 1e6).round(2).to_dict())
    if not units:
        return out
    # P1/P2 under central dispatch (20 Oct 2026): who each transfer is paid to, so an asset settlement
    # on the central-dispatch year (scripts/public_ownership/settlement.py) books exactly these totals.
    return out, unit_transfers(cr, fa_margin, support, storage_mw, n_c.generators["carrier"],
                               n_cb.storage_units["carrier"] if "carrier" in n_cb.storage_units else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--central", required=True)
    ap.add_argument("--market", required=True)
    ap.add_argument("--market-wholesale", default=None,
                    help="the market month's Stage 1 month (its wholesale_from parent); default --market")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--national-price", default="copperplate", choices=NATIONAL_PRICE_BASES)
    ap.add_argument("--selfdispatch", default=None,
                    help="scenario whose Stage 1 schedule is self-dispatch on the same forecast (firm-access counterfactual)")
    add_forecast_error_arguments(ap)
    ap.add_argument("--assumptions", default=str(DEFAULT_ASSUMPTIONS),
                    help="forecast bill assumptions JSON (interconnector cap-and-floor rule, item D4(b))")
    add_firm_access_support_argument(ap)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = settle(a.central, a.market, a.start, a.end, a.national_price, a.selfdispatch,
                 forecast_error=forecast_error_settings(a), assumptions=a.assumptions,
                 market_wholesale=a.market_wholesale, firm_access_support=a.firm_access_support)
    suffix = a.national_price + ("_sdfe" if a.selfdispatch else "")
    path = Path(a.out) if a.out else MARKET_DIR / f"{a.central}_fti_settlement_{suffix}.csv"
    out.to_csv(path)
    pd.set_option("display.width", 120)
    print(out.to_string())
    print(f"\nwritten {path}")


def consumer_totals(r: Mapping, fe: Mapping, hours: float, year_hours: float = YEAR_HOURS) -> dict:
    """The two consumer costs, the gross saving, the borrowed charge and the headline net saving.

    Like for like: the market pays the national price for all scheduled output,
    curtailed or not, plus the balancing mechanism, plus the cost of re-trading
    its interconnector positions in real time (the bill carries it in the
    constraint line). Central dispatch pays the national price for the central
    schedule, makes whole the output it runs above self-dispatch (cost recovery)
    and the output it leaves off (firm access), and pays to correct its forecast
    in real time, its own link changes included. Each line enters exactly one
    total once; the information-only lines enter none.

    Item D4(b): each side also receives the regulated interconnector pass-through
    on its own Stage 1 flows (a credit when rent above a cap is returned, a charge
    when a floor is topped up), exactly as the consumer bill carries it; the
    congestion-rent difference itself is a transfer (``*_info``) and enters no total.

    Decisions of 14 Oct 2026: central dispatch also pays storage its weekly make-whole
    and interconnector capacity holders their day-ahead make-whole; firm access carries
    the lost CfD and ROC support (inside ``central_firm_access_gbp_m``).

    Decision of 20 Oct 2026: each side also carries the CfD and RO lines on its own metered
    output (:func:`support_on_output`). Paying turned-down output its lost support while
    leaving these out charged central dispatch for the support and never credited the
    top-ups that output no longer earns. The headline without them is kept as
    ``consumer_saving_fti_comparable_net_excluding_support_lines_info_gbp_m``.

    User decision of 20 Oct 2026: the market's constraint cost here is the balancing
    stage **as solved** (both sides are least-cost dispatches, so the gap is the
    institution). Where a month's headline turn-down basis re-prices it (pro rata
    zonal turn-down, ``turn_down_allocation``), that uplift is reported beside it:
    ``consumer_saving_fti_comparable_net_in_force_basis_gbp_m`` is the headline with
    the market on its in-force basis, so the two stay reconcilable.
    """
    missing = [k for k in ("market_ic_pass_through_credit_gbp_m", "central_ic_pass_through_credit_gbp_m",
                           "central_storage_make_whole_gbp_m", "central_ic_make_whole_gbp_m",
                           "market_constraint_pro_rata_uplift_gbp_m", *SUPPORT_LINE_ROWS) if k not in r]
    if missing:
        raise KeyError(f"consumer_totals needs the interconnector pass-through, the make-whole lines, the "
                       f"turn-down uplift and the support lines: {missing}")
    out = {}
    support_m = r["market_support_lines_cfd_gbp_m"] + r["market_support_lines_roc_gbp_m"]
    support_c = r["central_support_lines_cfd_gbp_m"] + r["central_support_lines_roc_gbp_m"]
    out["market_consumer_cost_gbp_m"] = (r["market_energy_gbp_m"] + r["market_constraint_cost_gbp_m"]
                                         + r["market_interconnector_retrading_gbp_m"]
                                         - r["market_ic_pass_through_credit_gbp_m"] + support_m)
    out["central_consumer_cost_gbp_m"] = (
        r["central_energy_gbp_m"] + r["central_cost_recovery_gbp_m"]
        + r["central_firm_access_gbp_m"] + r["central_realtime_correction_gbp_m"]
        + r["central_storage_make_whole_gbp_m"] + r["central_ic_make_whole_gbp_m"]
        - r["central_ic_pass_through_credit_gbp_m"] + support_c
    )
    out["consumer_saving_gross_gbp_m"] = out["market_consumer_cost_gbp_m"] - out["central_consumer_cost_gbp_m"]
    # The borrowed FTI forecast-error charge for this window, and the FTI-comparable
    # net saving, which is the headline (scripts.market.forecast_error_charge).
    out["forecast_error_charge_basis"] = forecast_error_label(fe)
    out["forecast_error_charge_gbp_m"] = (
        fec.charge_for_period_gbp_m(out["consumer_saving_gross_gbp_m"], r["gb_demand_twh"], hours,
                                    basis=fe["basis"], as_simple_total=fe["as_simple_total"],
                                    fleet_ratio=fe["fleet_ratio"], price_year=fe["price_year"],
                                    year_hours=year_hours)
        if fe["enabled"] else 0.0)
    # The borrowed charge REPLACES the model's own real-time correction rather than
    # adding to it (user decision, 26 Sep 2026): forecast error is counted once,
    # entirely from FTI's estimate. With the charge off the net equals the gross.
    replaced = r["central_realtime_correction_gbp_m"] if fe["enabled"] else 0.0
    out["central_realtime_correction_replaced_gbp_m"] = replaced
    out[HEADLINE_METRIC] = (out["consumer_saving_gross_gbp_m"] + replaced
                            - out["forecast_error_charge_gbp_m"])
    out["consumer_saving_fti_comparable_net_excluding_support_lines_info_gbp_m"] = (
        out[HEADLINE_METRIC] - (support_m - support_c))
    uplift = r["market_constraint_pro_rata_uplift_gbp_m"]
    out["market_consumer_cost_in_force_basis_gbp_m"] = out["market_consumer_cost_gbp_m"] + uplift
    out[HEADLINE_METRIC_IN_FORCE] = out[HEADLINE_METRIC] + uplift
    return out


def gb_resource_saving(r: Mapping) -> float:
    """The resource saving from GB's side: the counter-trade premium is paid abroad (decision of 20 Oct 2026).

    ``resource_saving_gbp_m`` values both sides' physical dispatch: GB fuel at floored
    marginal cost and cross-border energy at the trade value. The premium each side pays
    to move a cable off its day-ahead flow (``counter_trade_premium_cost``: the GBP/MWh
    premium plus any foreign-plant offer and bid uplift the network was solved with) sits
    in neither, yet goes to traders and plant abroad, so from GB's side it is lost. Adding each side's
    premium gives the saving the ownership ledger measures (its "paid abroad" line),
    before the borrowed forecast-error charge.
    """
    return (r["resource_saving_gbp_m"] + r["market_counter_trade_premium_gbp_m"]
            - r["central_counter_trade_premium_gbp_m"])


def forecast_error_label(fe: Mapping) -> str:
    if not fe["enabled"]:
        return "disabled"
    reading = "simple total" if fe["as_simple_total"] else "NPV reading"
    return f"{fe['basis']}, {reading}, fleet ratio {fe['fleet_ratio']:g}, {fe['price_year']} prices"


def add_forecast_error_arguments(ap: argparse.ArgumentParser) -> None:
    """CLI for the borrowed forecast-error charge (config ``...institution_comparison.forecast_error_charge``)."""
    d = fec.DEFAULT_SETTINGS
    ap.add_argument("--forecast-error-basis", default=d["basis"], choices=fec.BASES)
    ap.add_argument("--forecast-error-simple-total", action="store_true",
                    help="read FTI's 7.9 bn as a simple total over eleven years, not an NPV")
    ap.add_argument("--forecast-error-fleet-ratio", type=float, default=d["fleet_ratio"])
    ap.add_argument("--forecast-error-price-year", type=int, default=d["price_year"])
    ap.add_argument("--no-forecast-error-charge", action="store_true",
                    help="report a zero charge, so the FTI-comparable net equals the gross")


def add_firm_access_support_argument(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--firm-access-support", default=DEFAULT_FIRM_ACCESS_SUPPORT, choices=FIRM_ACCESS_SUPPORT_BASES,
                    help="lost_support (decision of 14 Oct 2026): firm access pays lost CfD and ROC support; "
                         "metered_only: owners absorb it (the sensitivity)")


def forecast_error_settings(a) -> dict:
    return fec.settings_from_config({
        "enabled": not a.no_forecast_error_charge, "basis": a.forecast_error_basis,
        "as_simple_total": bool(a.forecast_error_simple_total), "fleet_ratio": a.forecast_error_fleet_ratio,
        "price_year": a.forecast_error_price_year})


if __name__ == "__main__":
    sys.exit(main())
