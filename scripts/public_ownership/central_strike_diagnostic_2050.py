"""FES 2050 decision 70: the post-AR7 break-even strikes recalibrated on the central-dispatch year (a diagnostic).

**The question (user, 4 October 2026).** If central dispatch cuts curtailment, and curtailment risk
is priced into CfD strikes, is there a bigger future benefit? In 2050 the post-AR7 private strikes
break even on the MARKET year's delivered output (decision 41 (b)), and the central-dispatch year
(``FES25_HT_2050_{month}_central_dispatch``) feeds only P3 I3, where the tariff replaces CfDs. So no
case shows strikes under central dispatch. This diagnostic re-derives each post-AR7 CfD cohort's
break-even strike on the central year's output, beside the market year's, before any full private
central-dispatch case is built (that case follows only if the effect is material).

**The rule: only uncompensated curtailment counts against a strike, under both bases.**

* *Market year (how the calibration already treats it).* A CfD unit's day-ahead (Stage 1) position
  is firm: when the BM turns it down it keeps the day-ahead price and its bid restores the lost
  difference payment (decision 37: CfD bids at strike; ``settlement.settle_contract_book`` books the
  restoration as ``cfd_curtailment_compensation_gbp``, inside the producer rent the calibration
  divides). BM turn-down is therefore income, not lost output, in the market-year calibration. What
  a developer loses is the output the day-ahead market never schedules (national surplus) and the
  difference payments the negative-pricing rule withholds.
* *Central year (FTI firm access and cost recovery, as ``scripts.market.fti_settlement`` settles
  the central year).* Firm access pays every GB unit the output the self-dispatch schedule would
  have run and the central schedule did not; for a perfect-foresight central year the self-dispatch
  schedule is the market twin's own Stage 1. So CfD output in the market Stage 1 schedule that
  central dispatch turns down is compensated, and does not count against the strike.
* *One measure for both.* Per unit and hour, the output that earns contract value is
  ``max(physical output, market Stage 1 schedule)``: delivered output, plus turn-down below the firm
  schedule (BM turn-down in the market year, firm-access turn-down in the central year), both
  valued at contract value, ``p + e (s - p)`` per MWh (``s`` in hours the contract pays, the price
  ``p`` in hours the negative-pricing rule withholds, ``e``). Uncompensated curtailment is available
  output less that. The two bases differ only where central dispatch runs a unit ABOVE the market's
  day-ahead schedule (less uncompensated curtailment: a lower strike), and through the price in
  withheld hours.
* *Declared simplifications.* (a) The market year's BM pays a turned-down CfD unit its strike on
  top of the kept day-ahead price (bids at strike, decision 37), which exceeds contract value by the
  day-ahead price; that excess is a feature of the bid rule, not of curtailment risk, and is held
  at the market year's (it sits in the anchor below, the same under both bases). (b) FTI floors
  firm access at zero in negative-price hours; 2050 HT has none, and the rule above is applied
  unfloored on both bases. (c) Reference price: the FTI national price, the market twin's
  copperplate Stage 1 price (``--central-price market_copperplate``, the default, as
  ``fti_settle_year --national-price copperplate``); ``central_stage1`` uses the central year's
  own Stage 1 price instead. (d) Every other income (Capacity Market, ancillary and support lines,
  the merchant tail credit, the capture-vs-reference spread of baseload contracts) is held at the
  market year's: the diagnostic moves volume, not the rest of the account.

**Anchor.** The market strike is the installed break-even calibration (``strike_calibration_2050.csv``,
``calibrated_value``: the register's rent at the strike in use, over its paid energy). The cohort's
contract-value income at that strike on the market basis, less the variable cost of what it
delivered, ``I_m(s_m) - v D_m``, is what it needs from the contract; the central strike solves
``I_c(s_c) - v D_c = I_m(s_m) - v D_m``. Linear in the strike, so

    s_c = (s_m x B_m + A_m - A_c + v x (D_c - D_m)) / B_c,
    B = sum(e x E),  A = sum((1 - e) x E x p),  E = max(D, F),  D = sum(delivered).

``v`` is the cohort's variable cost per delivered MWh as the register books it (variable O&M plus
fuel and carbon over dispatch, the private case's rows; ``--register``): output a basis delivers
costs it, compensated output does not. The market basis therefore reproduces ``s_m`` exactly, and
every income term the two bases share cancels.

**Auction margin (decision 71, user, 4 October 2026).** The installed calibration keeps a margin ``m``
of contract revenue as rent (``auction_margin_share``, 0.05 centrally; 0 breaks even). The central
strike earns the same margin on its own contract revenue, ``rent_c = m x s_c x Q_c``, not the market
year's rent: ``I_c(s_c) - v D_c - m s_c Q_c = I_m(s_m) - v D_m - m s_m Q_m``, so

    s_c = (s_m x (B_m - m Q_m) + A_m - A_c + v x (D_c - D_m)) / (B_c - m Q_c),

with ``Q`` the paid energy (delivered output in hours the contract pays, ``sum(e x D)``), the contract
revenue's own measure (strike x paid MWh), on each basis. ``m`` is read from the calibration, so the
diagnostic is on the margin the chain used; at ``m = 0`` it is the formula above, and the market basis
still reproduces ``s_m`` exactly. The paid energy the market basis computes (``sum(e x D)``) is reconciled against the
calibration's own denominator (the settlement's ``cfd_paid_mwh``).

**Scope (decision 70).** This matters for central dispatch as a reform under PRIVATE ownership. Under
P2 and P3 the state builds at cost and bears curtailment itself (decision 39); the channel there is
the cost of capital, already modelled. The hurdle-rate (risk-premium) channel, by which lower
volume risk could lower the private WACC itself, is not modelled.

**GBP effect on the 2050 CfD line.** The strike difference times the central year's paid energy
(difference payments on metered output), converted per household with the 2050 bill's own CfD-line
conversion (``diagnostics.cfd_line_gbp_per_3100kwh`` over ``cfd_payment_gbp``, which carries the
recovery denominator and the 2024 -> 2025 rebase); the strike difference on the compensated
turn-down is reported beside it, recovered over final demand like the constraint line.

Usage (through the calibration CLI, decision 54's pattern)::

    python -m scripts.public_ownership.vintage_capital_2050 calibrate --dispatch-basis central \\
        --book resources/public_ownership/fes_ht_2050_support_contract_book.csv \\
        --market-calibration data/public_ownership/financing/strike_calibration_2050.csv \\
        --out resources/public_ownership/fes_ht_2050_central_dispatch_strike_diagnostic.csv
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("central_strike_diagnostic_2050")

DEFAULT_STUDY = "FES25_HT_2050"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MARKET_SUFFIX = "_market"
CENTRAL_SUFFIX = "_central_dispatch"
MARKET_DIR = "resources/market"
DEFAULT_BOOK = "resources/public_ownership/fes_ht_2050_support_contract_book.csv"
DEFAULT_CALIBRATION = "data/public_ownership/financing/strike_calibration_2050.csv"
DEFAULT_BILL_SUMMARY = "resources/analysis/FES25_HT_2050_consumer_bill_future_summary.json"
DEFAULT_REGISTER = "resources/public_ownership/fes_ht_2050_asset_finance_register.parquet"
REGISTER_COLUMNS = ("case_id", "asset_id", "dispatch_mwh", "annual_variable_om_gbp", "fuel_carbon_gbp")
#: Decision 71: the calibration's auction-margin column (``vintage_capital_2050.AUCTION_MARGIN_COLUMN``).
MARGIN_COLUMN = "auction_margin_share"
DEFAULT_OUT = "resources/public_ownership/fes_ht_2050_central_dispatch_strike_diagnostic.csv"
CENTRAL_PRICE_BASES = ("market_copperplate", "central_stage1")
#: Report groups (decision 70: offshore apart from onshore and solar; the signs may differ).
TECHNOLOGY_GROUPS = {"offshore": ("offshore",), "onshore": ("onshore",), "solar": ("solar",),
                     "onshore_and_solar": ("onshore", "solar"), "tidal": ("tidal",), "nuclear": ("nuclear",)}
#: Proposed, for the user to confirm: the per-household CfD-line effect (GBP ex VAT a year, 3,100 kWh)
#: at or above which the full private central-dispatch case is worth building.
DEFAULT_MATERIALITY_GBP_PER_3100KWH = 1.0
#: Paid energy recomputed here against the calibration's own denominator: warn above this gap.
RECONCILIATION_TOLERANCE = 0.01
ENERGY_KEYS = ("available", "delivered", "compensated", "contract_value", "uncompensated", "gross_curtailed",
               "paid", "contract_value_paid", "contract_value_unpaid")
RULE_TEXT = (
    "Only uncompensated curtailment counts against a strike, on both bases. Per unit and hour the "
    "output earning contract value is max(physical output, market Stage 1 schedule): delivered output "
    "plus turn-down below the firm day-ahead schedule (BM turn-down in the market year, restored by the "
    "CfD bid; firm-access turn-down in the central year, FTI's firm access against the market twin's "
    "Stage 1), valued at contract value p + e(s - p). Uncompensated curtailment = available - that "
    "(day-ahead surplus) and the withheld negative-price hours. The market strike is the installed "
    "calibration; the central strike solves I_c(s) - v D_c - m s Q_c = I_m(s_m) - v D_m - m s_m Q_m (v: the "
    "register's variable cost per delivered MWh; m: the calibration's auction margin, decision 71, so the "
    "central strike earns the same share of its own contract revenue, strike x paid MWh Q); every other "
    "income is held at the market year's.")


class StrikeDiagnosticError(ValueError):
    """Inputs the diagnostic cannot use."""


# ── cohorts ──────────────────────────────────────────────────────────────────

def market_strikes(calibration: pd.DataFrame) -> pd.DataFrame:
    """The calibrated post-AR7 CfD cohorts: ``key``, ``cfd_round``, ``technology``, ``build_year``,
    ``market_strike`` (the calibrated value), ``calibration_paid_mwh`` (its denominator) and
    ``auction_margin`` (decision 71: the share of contract revenue the calibration keeps as rent; 0 for a
    file from before the decision)."""
    rows = calibration.loc[calibration["scheme"].astype(str).eq("CfD")
                           & calibration["cfd_round"].astype(str).str.startswith("PAR7_")]
    if rows.empty:
        raise StrikeDiagnosticError("the calibration has no post-AR7 CfD cohort")
    if rows["key"].duplicated().any():
        raise StrikeDiagnosticError("the calibration repeats a cohort")
    margin = (pd.to_numeric(rows[MARGIN_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
              if MARGIN_COLUMN in rows.columns else np.zeros(len(rows)))
    if len(margin) and float(margin.max() - margin.min()) > 1e-9:
        raise StrikeDiagnosticError(f"the calibration mixes auction margins {sorted(set(np.round(margin, 6)))}")
    if ((margin < 0) | (margin >= 0.5)).any():
        raise StrikeDiagnosticError("the calibration's auction margin must be in [0, 0.5)")
    return pd.DataFrame({"key": rows["key"].astype(str).to_numpy(), "cfd_round": rows["cfd_round"].astype(str).to_numpy(),
                         "technology": rows["technology"].astype(str).to_numpy(),
                         "build_year": pd.to_numeric(rows["build_year"]).astype(int).to_numpy(),
                         "market_strike": pd.to_numeric(rows["calibrated_value"]).to_numpy(dtype=float),
                         "calibration_paid_mwh": pd.to_numeric(rows["denominator"]).to_numpy(dtype=float),
                         "auction_margin": margin})


def cohort_weights(book: pd.DataFrame, cohorts: Iterable[str]) -> pd.DataFrame:
    """Network asset x cohort: the share of the asset's output each calibrated cohort's contract settles on.

    The contract book's CfD units (generators) whose ``strike_key`` is a calibrated cohort, summed
    over units by ``energy_factor`` (settlement share x live share), exactly as
    ``contract_book_2050.cfd_units`` scales each unit's dispatch for the settlement.
    """
    cohorts = list(cohorts)
    cfd = book.loc[book["scheme"].astype(str).eq("CfD") & book["component_type"].astype(str).eq("Generator")]
    if "strike_key" not in cfd.columns:
        raise StrikeDiagnosticError("the contract book carries no strike_key")
    cfd = cfd.loc[cfd["strike_key"].astype(str).isin(cohorts)]
    missing = sorted(set(cohorts) - set(cfd["strike_key"].astype(str)))
    if missing:
        raise StrikeDiagnosticError(f"{len(missing)} calibrated cohorts have no unit in the contract book "
                                    f"(e.g. {missing[:3]}): the book and the calibration disagree")
    factor = pd.to_numeric(cfd["energy_factor"], errors="coerce")
    if factor.isna().any() or (factor <= 0).any():
        raise StrikeDiagnosticError("contract-book CfD units need a positive energy_factor")
    weights = (pd.DataFrame({"asset": cfd["network_asset"].astype(str).to_numpy(),
                             "key": cfd["strike_key"].astype(str).to_numpy(), "w": factor.to_numpy()})
               .pivot_table(index="asset", columns="key", values="w", aggfunc="sum", fill_value=0.0))
    return weights.reindex(columns=cohorts, fill_value=0.0)


# ── volumes ──────────────────────────────────────────────────────────────────

def basis_energy(delivered: pd.DataFrame, firm: pd.DataFrame, available: pd.DataFrame,
                 weights: pd.DataFrame, hours: pd.Series) -> dict[str, pd.DataFrame]:
    """Hourly cohort energy (MWh) on one basis: snapshots x cohorts, per :data:`ENERGY_KEYS` part.

    Args:
        delivered: Physical output, MW (snapshots x network assets): the market BM or the central outturn.
        firm: The firm schedule, MW: the market twin's Stage 1 (the same frame on both bases).
        available: Available output, MW (p_nom x p_max_pu).
        weights: :func:`cohort_weights`.
        hours: Snapshot weightings.

    Returns:
        ``available``, ``delivered``, ``compensated`` (turn-down below the firm schedule),
        ``contract_value`` (= delivered + compensated = max(D, F)), ``uncompensated``
        (available - max(D, F), floored at zero per unit and hour) and ``gross_curtailed``
        (available - delivered, floored), each per cohort.
    """
    assets = weights.index
    for name, frame in (("delivered", delivered), ("firm", firm), ("available", available)):
        if missing := sorted(set(assets) - set(frame.columns)):
            raise StrikeDiagnosticError(f"{len(missing)} contract-book assets missing from the {name} table "
                                        f"(e.g. {missing[:3]})")
        if not frame.index.equals(delivered.index):
            raise StrikeDiagnosticError(f"the {name} table's snapshots differ from the dispatch's")
    d = delivered[assets].to_numpy(dtype=float).clip(min=0.0)
    f = firm[assets].to_numpy(dtype=float).clip(min=0.0)
    a = available[assets].to_numpy(dtype=float).clip(min=0.0)
    h = hours.reindex(delivered.index).to_numpy(dtype=float)[:, None]
    if not np.isfinite(h).all():
        raise StrikeDiagnosticError("snapshot weightings do not cover the dispatch")
    e = np.maximum(d, f)
    parts = {"available": a, "delivered": d, "compensated": e - d, "contract_value": e,
             "uncompensated": np.maximum(a - e, 0.0), "gross_curtailed": np.maximum(a - d, 0.0)}
    w = weights.to_numpy(dtype=float)
    return {k: pd.DataFrame((v * h) @ w, index=delivered.index, columns=weights.columns) for k, v in parts.items()}


def concat_energy(months: Sequence[Mapping[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """The months' :func:`basis_energy` frames stacked into the year."""
    if not months:
        raise StrikeDiagnosticError("no month to diagnose")
    return {k: pd.concat([m[k] for m in months]) for k in months[0]}


# ── strikes ──────────────────────────────────────────────────────────────────

def income_terms(contract_value: np.ndarray, price: np.ndarray, eligible: np.ndarray) -> tuple[float, float]:
    """``(A, B)`` of a cohort's contract-value income ``A + s x B`` over hours.

    ``B = sum(e x E)`` (the energy the strike is paid on), ``A = sum((1 - e) x E x p)`` (the
    energy the negative-pricing rule leaves at the reference price).
    """
    e = np.asarray(eligible, dtype=float)
    cv = np.asarray(contract_value, dtype=float)
    return float(((1.0 - e) * cv * np.asarray(price, dtype=float)).sum()), float((e * cv).sum())


def central_strike(market_strike: float, market_terms: tuple[float, float],
                   central_terms: tuple[float, float], cost_shift_gbp: float = 0.0, *, margin: float = 0.0,
                   market_paid_mwh: float = 0.0, central_paid_mwh: float = 0.0) -> float:
    """The strike at which the central basis earns the margin the market basis earns at the market strike.

    ``cost_shift_gbp`` is the variable cost the central basis's output adds, ``v x (D_c - D_m)``.
    Decision 71: with an auction margin ``m`` each basis keeps ``m x strike x paid MWh`` as rent, so
    ``s_c = (s_m (B_m - m Q_m) + A_m - A_c + shift) / (B_c - m Q_c)``; at ``m = 0`` the break-even
    formula, and with identical bases ``s_m`` exactly.
    """
    a_m, b_m = market_terms
    a_c, b_c = central_terms
    m = float(margin)
    denominator = b_c - m * float(central_paid_mwh)
    if b_c <= 0.0 or denominator <= 0.0:
        return float("nan")
    return (market_strike * (b_m - m * float(market_paid_mwh)) + a_m - a_c + cost_shift_gbp) / denominator


def eligibility(price: pd.Series, rounds: Sequence[str], rule: str) -> np.ndarray:
    """Hours x cohorts: does the difference payment accrue (``revenue_tracking.negative_price_eligibility``)."""
    from scripts.market.revenue_tracking import negative_price_eligibility

    return np.asarray(negative_price_eligibility(price.to_numpy(dtype=float), list(rounds), rule), dtype=bool)


def diagnose(cohorts: pd.DataFrame, market: Mapping[str, pd.DataFrame], central: Mapping[str, pd.DataFrame],
             market_price: pd.Series, central_price: pd.Series, *, negative_price_rule: str,
             variable_cost: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Per cohort: both strikes, the energy on each basis, and the strike difference's money.

    Args:
        cohorts: :func:`market_strikes`.
        market, central: :func:`concat_energy` on each basis (same cohorts and snapshots).
        market_price, central_price: The reference price each basis settles on.
        negative_price_rule: The CfD negative-pricing rule (the bill's, ``gb_contract_terms`` in 2050).
        variable_cost: Cohort -> variable cost per delivered MWh (:func:`register_variable_costs`;
            None or a missing cohort: zero).
    """
    keys = cohorts["key"].tolist()
    index = market["contract_value"].index
    if not central["contract_value"].index.equals(index):
        raise StrikeDiagnosticError("the market and central years cover different snapshots")
    p_m = market_price.reindex(index)
    p_c = central_price.reindex(index)
    if p_m.isna().any() or p_c.isna().any():
        raise StrikeDiagnosticError("a reference price does not cover the dispatch")
    rounds = cohorts["cfd_round"].tolist()
    e_m = eligibility(p_m, rounds, negative_price_rule)
    e_c = eligibility(p_c, rounds, negative_price_rule)
    rows = []
    for j, c in enumerate(cohorts.itertuples(index=False)):
        k = c.key
        out = {"key": k, "cfd_round": c.cfd_round, "technology": c.technology, "build_year": int(c.build_year)}
        terms = {}
        for tag, energy, price, elig in (("market", market, p_m, e_m[:, j]), ("central", central, p_c, e_c[:, j])):
            cv = energy["contract_value"][k].to_numpy()
            terms[tag] = income_terms(cv, price.to_numpy(), elig)
            # Decision 70 (info): the central strike if central turn-down were NOT compensated (no firm access).
            if tag == "central":
                terms["central_no_firm_access"] = income_terms(energy["delivered"][k].to_numpy(), price.to_numpy(), elig)
            for part in ("available", "delivered", "compensated", "contract_value", "uncompensated", "gross_curtailed"):
                out[f"{tag}_{part}_mwh"] = float(energy[part][k].sum())
            out[f"{tag}_paid_mwh"] = float((elig * energy["delivered"][k].to_numpy()).sum())
            out[f"{tag}_contract_value_paid_mwh"] = terms[tag][1]
            out[f"{tag}_contract_value_unpaid_mwh"] = float(((~elig) * cv).sum())
        s_m = float(c.market_strike)
        v = float((variable_cost or {}).get(k, 0.0))
        shift = v * (out["central_delivered_mwh"] - out["market_delivered_mwh"])
        # Decision 71: each basis keeps the calibration's margin of its own contract revenue (strike x paid MWh).
        m = float(getattr(c, "auction_margin", 0.0) or 0.0)
        paid = {"margin": m, "market_paid_mwh": out["market_paid_mwh"], "central_paid_mwh": out["central_paid_mwh"]}
        s_c = central_strike(s_m, terms["market"], terms["central"], shift, **paid)
        out.update({"market_strike_gbp_per_mwh": s_m, "central_strike_gbp_per_mwh": s_c,
                    "strike_difference_gbp_per_mwh": s_c - s_m, "variable_cost_gbp_per_mwh": v,
                    "variable_cost_shift_gbp": shift, "auction_margin_share": m,
                    "market_margin_rent_gbp": m * s_m * out["market_paid_mwh"],
                    "central_margin_rent_gbp": (m * s_c * out["central_paid_mwh"]) if np.isfinite(s_c) else float("nan"),
                    "central_strike_without_firm_access_info": central_strike(s_m, terms["market"],
                                                                              terms["central_no_firm_access"], shift,
                                                                              **paid),
                    "calibration_paid_mwh": float(c.calibration_paid_mwh)})
        out["paid_mwh_reconciliation_ratio"] = (out["market_paid_mwh"] / out["calibration_paid_mwh"]
                                                if out["calibration_paid_mwh"] > 0 else float("nan"))
        delta = s_c - s_m if np.isfinite(s_c) else 0.0
        # The CfD line: difference payments on the central year's metered output in paid hours.
        out["cfd_line_effect_gbp"] = delta * out["central_paid_mwh"]
        # The same strike difference on the compensated turn-down (firm access), recovered like the constraint line.
        out["compensation_effect_gbp"] = delta * (out["central_contract_value_paid_mwh"] - out["central_paid_mwh"])
        out["cfd_and_compensation_effect_gbp"] = out["cfd_line_effect_gbp"] + out["compensation_effect_gbp"]
        out["cfd_line_effect_on_market_volume_gbp_info"] = delta * out["market_paid_mwh"]
        rows.append(out)
    return pd.DataFrame(rows)


def register_variable_costs(register: pd.DataFrame, book: pd.DataFrame, cohorts: Iterable[str],
                            case_id: str = "private") -> dict[str, float]:
    """Cohort -> variable cost per delivered MWh as the register books it (variable O&M + fuel and carbon).

    The private case's rows (``asset_id`` = the contract book's ``unit``) of each calibrated cohort:
    ``sum(annual_variable_om_gbp + fuel_carbon_gbp) / sum(dispatch_mwh)``, the rate
    ``vintage_capital_2050.declared_output_strike`` scales by. A cohort that dispatched nothing is zero.
    """
    rows = register.loc[register["case_id"].astype(str).eq(str(case_id))]
    cfd = book.loc[book["scheme"].astype(str).eq("CfD")]
    key = pd.Series(cfd["strike_key"].astype(str).to_numpy(), index=cfd["unit"].astype(str).to_numpy())
    num = lambda c: pd.to_numeric(rows[c], errors="coerce").fillna(0.0).to_numpy()  # noqa: E731
    frame = pd.DataFrame({"key": key.reindex(rows["asset_id"].astype(str)).to_numpy(), "dispatch": num("dispatch_mwh"),
                          "cost": num("annual_variable_om_gbp") + num("fuel_carbon_gbp")})
    g = frame.dropna(subset=["key"]).groupby("key")[["dispatch", "cost"]].sum()
    out = {}
    for k in cohorts:
        if k in g.index and g.loc[k, "dispatch"] > 0:
            out[str(k)] = float(g.loc[k, "cost"] / g.loc[k, "dispatch"])
    return out


# ── the bill's conversion ────────────────────────────────────────────────────

def bill_conversion(summary: Mapping) -> dict:
    """GBP a year -> GBP per household, from the 2050 bill summary's own CfD line.

    ``cfd_per_gbp``: the CfD line per 3,100 kWh over the CfD payment (recovery over eligible
    supplier sales, rebased 2024 -> 2025); ``final_demand_per_gbp``: the same rebase over final
    demand (the constraint line's denominator). The headline household (the bill's headline
    archetype) pays both on its unit rate, so it scales by its consumption.
    """
    diag = summary.get("diagnostics") or {}
    payment = float(summary.get("cfd_payment_gbp") or diag.get("cfd_payment_gbp") or 0.0)
    per_3100 = diag.get("cfd_line_gbp_per_3100kwh")
    cfd_denominator = float(summary.get("cfd_recovery_denominator_mwh") or diag.get("cfd_recovery_denominator_mwh") or 0.0)
    final = float(summary.get("final_demand_recovery_denominator_mwh")
                  or (summary.get("recovery_denominators") or {}).get("final_demand_mwh") or 0.0)
    if payment <= 0 or per_3100 is None or cfd_denominator <= 0 or final <= 0:
        raise StrikeDiagnosticError("the bill summary lacks the CfD line, its payment or its recovery denominators")
    cfd_per_gbp = float(per_3100) / payment
    rebase = cfd_per_gbp * cfd_denominator / 3.1
    archetypes = diag.get("household_archetypes") or {}
    headline = str(archetypes.get("headline_archetype") or "benchmark")
    kwh = next((float(b["annual_consumption_kwh"]) for b in archetypes.get("bills") or []
                if str(b.get("archetype")) == headline), 3100.0)
    return {"cfd_per_gbp_per_3100kwh": cfd_per_gbp, "final_demand_per_gbp_per_3100kwh": rebase * 3.1 / final,
            "price_rebase": rebase, "headline_archetype": headline, "headline_kwh": kwh,
            "vat_rate": float(summary.get("vat_rate", 0.05) or 0.0),
            "report_price_year": summary.get("report_price_year"), "model_price_year": summary.get("model_price_year")}


def add_household_columns(frame: pd.DataFrame, conversion: Mapping) -> pd.DataFrame:
    """Per-household columns (ex VAT, the bill's report prices): 3,100 kWh and the headline household."""
    out = frame.copy()
    per_3100 = (out["cfd_line_effect_gbp"] * conversion["cfd_per_gbp_per_3100kwh"]
                + out["compensation_effect_gbp"] * conversion["final_demand_per_gbp_per_3100kwh"])
    out["cfd_line_effect_gbp_per_3100kwh_ex_vat"] = out["cfd_line_effect_gbp"] * conversion["cfd_per_gbp_per_3100kwh"]
    out["cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat"] = per_3100
    out["cfd_and_compensation_effect_gbp_per_headline_household_ex_vat"] = per_3100 * conversion["headline_kwh"] / 3100.0
    return out


# ── summary ──────────────────────────────────────────────────────────────────

def _group_summary(rows: pd.DataFrame, conversion: Mapping | None) -> dict:
    paid_c = float(rows["central_paid_mwh"].sum())
    paid_m = float(rows["market_paid_mwh"].sum())
    out = {"cohorts": int(len(rows))}
    for tag in ("market", "central"):
        for part in ("available", "delivered", "compensated", "contract_value", "uncompensated", "gross_curtailed",
                     "paid"):
            out[f"{tag}_{part}_twh"] = float(rows[f"{tag}_{part}_mwh"].sum()) / 1e6
    out["market_strike_paid_weighted"] = (float((rows["market_strike_gbp_per_mwh"] * rows["market_paid_mwh"]).sum()) / paid_m
                                          if paid_m > 0 else None)
    out["central_strike_paid_weighted"] = (float((rows["central_strike_gbp_per_mwh"] * rows["central_paid_mwh"]).sum()) / paid_c
                                           if paid_c > 0 else None)
    out["strike_difference_paid_weighted"] = (float(rows["cfd_line_effect_gbp"].sum()) / paid_c if paid_c > 0 else None)
    for key in ("cfd_line_effect_gbp", "compensation_effect_gbp", "cfd_and_compensation_effect_gbp",
                "cfd_line_effect_on_market_volume_gbp_info"):
        out[key.replace("_gbp", "_gbp_m")] = float(rows[key].sum()) / 1e6
    if conversion is not None:
        for key in ("cfd_line_effect_gbp_per_3100kwh_ex_vat", "cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat",
                    "cfd_and_compensation_effect_gbp_per_headline_household_ex_vat"):
            out[key] = float(rows[key].sum())
        out["cfd_and_compensation_effect_gbp_per_3100kwh_inc_vat"] = (
            out["cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat"] * (1.0 + conversion["vat_rate"]))
    return out


def summarise(frame: pd.DataFrame, *, conversion: Mapping | None, settings: Mapping,
              materiality_gbp_per_3100kwh: float = DEFAULT_MATERIALITY_GBP_PER_3100KWH) -> dict:
    """The JSON summary: the rule, settings, totals, by technology, the report groups and reconciliation."""
    by_tech = {t: _group_summary(g, conversion) for t, g in frame.groupby("technology", sort=True)}
    groups = {name: _group_summary(frame.loc[frame["technology"].isin(techs)], conversion)
              for name, techs in TECHNOLOGY_GROUPS.items() if frame["technology"].isin(techs).any()}
    total = _group_summary(frame, conversion)
    ratio = frame["paid_mwh_reconciliation_ratio"]
    worst = (ratio - 1.0).abs()
    recon = {"total_market_paid_mwh": float(frame["market_paid_mwh"].sum()),
             "total_calibration_paid_mwh": float(frame["calibration_paid_mwh"].sum()),
             "worst_cohort_gap": float(worst.max()) if worst.notna().any() else None,
             "worst_cohort": (str(frame.loc[worst.idxmax(), "key"]) if worst.notna().any() else None),
             "tolerance": RECONCILIATION_TOLERANCE}
    recon["within_tolerance"] = bool(recon["worst_cohort_gap"] is not None
                                     and recon["worst_cohort_gap"] <= RECONCILIATION_TOLERANCE)
    headline = total.get("cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat")
    return {
        "decision": "FES 2050 plan, decision 70 (user, 4 October 2026): strike diagnostic on the central-dispatch year",
        "written": dt.datetime.now().isoformat(timespec="seconds"),
        "rule": RULE_TEXT,
        "scope": ("Central dispatch as a reform under private ownership. Under P2 and P3 the state builds at cost "
                  "and bears curtailment itself (decision 39); the channel there is the cost of capital, already "
                  "modelled. The hurdle-rate (risk-premium) channel is not modelled. The full private "
                  "central-dispatch case follows only if this diagnostic is material."),
        "price_basis": "GBP real 2024 (model money) for strikes and GBP totals; per-household figures in the bill's "
                       "report prices (rebased by the bill's own conversion), ex VAT unless named",
        "settings": dict(settings),
        "conversion": dict(conversion) if conversion is not None else None,
        "total": total, "groups": groups, "by_technology": by_tech, "reconciliation": recon,
        "materiality": {"threshold_gbp_per_3100kwh_ex_vat": materiality_gbp_per_3100kwh,
                        "threshold_status": "proposed by the implementing agent; for the user to confirm",
                        "headline_gbp_per_3100kwh_ex_vat": headline,
                        "material": (None if headline is None else bool(abs(headline) >= materiality_gbp_per_3100kwh))},
    }


# ── reading the solved months ────────────────────────────────────────────────

def read_dispatch(path, assets: Sequence[str]) -> pd.DataFrame:
    """A ``*_dispatch.csv`` (snapshot x generator, MW) restricted to ``assets`` (all must be present)."""
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(assets) - set(header))
    if missing:
        raise StrikeDiagnosticError(f"{path}: {len(missing)} contract-book assets missing (e.g. {missing[:3]})")
    frame = pd.read_csv(path, usecols=[header[0], *assets], index_col=0, parse_dates=True)
    if frame.index.has_duplicates:
        raise StrikeDiagnosticError(f"{path} repeats a snapshot")
    return frame[list(assets)]


def read_price(path) -> pd.Series:
    """The Stage 1 ``wholesale_price`` (the settlement's reference price, ``project_future_bill.read_hourly_inputs``)."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)["wholesale_price"]
    if prices.index.has_duplicates:
        raise StrikeDiagnosticError(f"{path} repeats a snapshot")
    return prices.astype(float)


def read_availability(path, assets: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Available output (p_nom x p_max_pu, MW) of ``assets`` and the snapshot weightings, from a solved network.

    The whole variable is read and subset in pandas (label selection on a saved network is slow).
    """
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4") as ds:
        snapshots = pd.DatetimeIndex(ds["snapshots_snapshot"].values)
        hours = pd.Series(np.asarray(ds["snapshots_generators"].values, dtype=float), index=snapshots)
        names = pd.Index(ds["generators_i"].values).astype(str)
        p_nom = pd.Series(np.asarray(ds["generators_p_nom"].values, dtype=float), index=names)
        static = pd.Series(np.asarray(ds["generators_p_max_pu"].values, dtype=float), index=names) \
            if "generators_p_max_pu" in ds else pd.Series(1.0, index=names)
        dynamic = None
        if "generators_t_p_max_pu" in ds:
            dynamic = pd.DataFrame(np.asarray(ds["generators_t_p_max_pu"].values, dtype=float), index=snapshots,
                                   columns=pd.Index(ds["generators_t_p_max_pu_i"].values).astype(str))
    if missing := sorted(set(assets) - set(names)):
        raise StrikeDiagnosticError(f"{path}: {len(missing)} contract-book assets not in the network (e.g. {missing[:3]})")
    pu = pd.DataFrame(np.broadcast_to(static.reindex(assets).to_numpy()[None, :], (len(snapshots), len(assets))).copy(),
                      index=snapshots, columns=list(assets))
    if dynamic is not None:
        varying = [a for a in assets if a in dynamic.columns]
        pu[varying] = dynamic[varying].to_numpy()
    return pu.mul(p_nom.reindex(assets).to_numpy(), axis=1), hours


def month_paths(study: str, month: str, *, central_suffix: str = CENTRAL_SUFFIX, market_dir: str = MARKET_DIR,
                root: Path = ROOT) -> dict[str, Path]:
    """The solved files one month reads: the market twin's Stage 1 and BM, the central outturn, prices, network."""
    m, c = f"{study}_{month}{MARKET_SUFFIX}", f"{study}_{month}{central_suffix}"
    d = Path(root) / market_dir
    return {"market_stage1": d / f"{m}_wholesale_dispatch.csv", "market_bm": d / f"{m}_balancing_dispatch.csv",
            "central_bm": d / f"{c}_balancing_dispatch.csv", "market_price": d / f"{m}_wholesale_price.csv",
            "central_price": d / f"{c}_wholesale_price.csv", "network": d / f"{m}_wholesale.nc",
            "audit": d / f"{m}_contract_value_bids.csv"}


def month_energy(paths: Mapping[str, Path], weights: pd.DataFrame, *, central_price_basis: str) -> dict:
    """One month: each basis's cohort energy and the two reference prices."""
    assets = list(weights.index)
    firm = read_dispatch(paths["market_stage1"], assets)
    market_d = read_dispatch(paths["market_bm"], assets).reindex(firm.index)
    central_d = read_dispatch(paths["central_bm"], assets).reindex(firm.index)
    if market_d.isna().any().any() or central_d.isna().any().any():
        raise StrikeDiagnosticError(f"{paths['market_bm']} or {paths['central_bm']} does not cover the Stage 1 snapshots")
    available, hours = read_availability(paths["network"], assets)
    available = available.reindex(firm.index)
    if available.isna().any().any():
        raise StrikeDiagnosticError(f"{paths['network']} does not cover the dispatch snapshots")
    market_price = read_price(paths["market_price"]).reindex(firm.index)
    central_price = (market_price if central_price_basis == "market_copperplate"
                     else read_price(paths["central_price"]).reindex(firm.index))
    return {"market": basis_energy(market_d, firm, available, weights, hours),
            "central": basis_energy(central_d, firm, available, weights, hours),
            "market_price": market_price, "central_price": central_price}


def audit_rules(paths: Sequence[Path]) -> dict:
    """The market months' CfD bid basis and negative-pricing rule (the balancing audit), checked to agree."""
    from scripts.consumer_bill.extract_future_balancing_costs import bid_basis_from_audit, negative_price_rule_from_audit

    bases = {bid_basis_from_audit(p) for p in paths}
    rules = {negative_price_rule_from_audit(p) for p in paths}
    if len(bases) > 1 or len(rules) > 1:
        raise StrikeDiagnosticError(f"the market months disagree: bid bases {sorted(bases)}, rules {sorted(rules)}")
    return {"bid_basis": bases.pop(), "negative_price_rule": rules.pop()}


def run(*, study: str = DEFAULT_STUDY, months: Sequence[str] = MONTHS, book_path=DEFAULT_BOOK,
        calibration_path=DEFAULT_CALIBRATION, bill_summary_path=DEFAULT_BILL_SUMMARY, out_csv=DEFAULT_OUT,
        register_path=DEFAULT_REGISTER,
        out_json=None, central_suffix: str = CENTRAL_SUFFIX, central_price_basis: str = "market_copperplate",
        negative_price_rule: str | None = None, market_dir: str = MARKET_DIR, root: Path = ROOT,
        materiality_gbp_per_3100kwh: float = DEFAULT_MATERIALITY_GBP_PER_3100KWH) -> tuple[pd.DataFrame, dict]:
    """Read the solved months, diagnose every calibrated cohort, write the CSV and JSON summary."""
    if central_price_basis not in CENTRAL_PRICE_BASES:
        raise StrikeDiagnosticError(f"central_price_basis={central_price_basis!r}; one of {CENTRAL_PRICE_BASES}")
    root = Path(root)

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    from scripts.public_ownership.contract_book_2050 import read_contract_book

    cohorts = market_strikes(pd.read_csv(_abs(calibration_path)))
    book = read_contract_book(_abs(book_path))
    weights = cohort_weights(book, cohorts["key"])
    variable_cost, cost_basis = {}, "none: no register (variable cost taken as zero)"
    if register_path and _abs(register_path).exists():
        from scripts.public_ownership.register_io import read_register

        register = read_register(_abs(register_path), columns=list(REGISTER_COLUMNS), cases=["private"])
        variable_cost = register_variable_costs(register, book, cohorts["key"])
        cost_basis = f"register private rows: (variable O&M + fuel and carbon) / dispatch ({Path(register_path).name})"
        del register
    else:
        LOGGER.warning("no register at %s: variable cost taken as zero", register_path)
    paths = [month_paths(study, m, central_suffix=central_suffix, market_dir=market_dir, root=root) for m in months]
    missing = [str(p[k]) for p in paths for k in ("market_stage1", "market_bm", "central_bm", "market_price", "network")
               if not p[k].exists()]
    if central_price_basis == "central_stage1":
        missing += [str(p["central_price"]) for p in paths if not p["central_price"].exists()]
    if missing:
        raise StrikeDiagnosticError(f"{len(missing)} solved files missing (e.g. {missing[:3]}); run after the "
                                    "decision-49 market and central years")
    audit = audit_rules([p["audit"] for p in paths])
    rule = negative_price_rule or audit["negative_price_rule"]
    parts = []
    for m, p in zip(months, paths):
        LOGGER.info("decision 70: %s", m)
        parts.append(month_energy(p, weights, central_price_basis=central_price_basis))
    market = concat_energy([x["market"] for x in parts])
    central = concat_energy([x["central"] for x in parts])
    frame = diagnose(cohorts, market, central, pd.concat([x["market_price"] for x in parts]),
                     pd.concat([x["central_price"] for x in parts]), negative_price_rule=rule,
                     variable_cost=variable_cost)
    conversion = None
    summary_path = _abs(bill_summary_path) if bill_summary_path else None
    if summary_path is not None and summary_path.exists():
        conversion = bill_conversion(json.loads(summary_path.read_text(encoding="utf-8")))
        frame = add_household_columns(frame, conversion)
    else:
        LOGGER.warning("no bill summary at %s: no per-household figures", summary_path)
    inputs = {str(x): (dt.datetime.fromtimestamp(_abs(x).stat().st_mtime).isoformat(timespec="seconds")
                       if _abs(x).exists() else None)
              for x in [calibration_path, book_path, bill_summary_path, register_path] if x}
    settings = {"study": study, "months": list(months), "central_suffix": central_suffix,
                "central_price_basis": central_price_basis, "negative_price_rule": rule,
                # Decision 71: the calibration's auction margin, earned on each basis's own contract revenue.
                "auction_margin_share": float(cohorts["auction_margin"].iloc[0]),
                "market_bm_bid_basis": audit["bid_basis"], "firm_schedule": "market twin's Stage 1 (wholesale_dispatch)",
                "variable_cost_basis": cost_basis,
                "inputs": inputs}
    summary = summarise(frame, conversion=conversion, settings=settings,
                        materiality_gbp_per_3100kwh=materiality_gbp_per_3100kwh)
    if not summary["reconciliation"]["within_tolerance"]:
        LOGGER.warning("market-basis paid energy differs from the calibration's by up to %.2f%% (%s): the "
                       "calibration and the solved months may not match", 100 * (summary["reconciliation"]
                                                                                 ["worst_cohort_gap"] or 0.0),
                       summary["reconciliation"]["worst_cohort"])
    out_csv = _abs(out_csv)
    out_json = _abs(out_json) if out_json else out_csv.with_suffix(".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    LOGGER.info("wrote %s and %s", out_csv, out_json)
    return frame, summary


def print_summary(summary: Mapping) -> None:
    """The groups on stdout (strikes in GBP/MWh 2024, energy in TWh, money in GBP m)."""
    print(f"decision 70 strike diagnostic ({summary['settings']['central_price_basis']}, "
          f"rule {summary['settings']['negative_price_rule']})")
    for name, g in {**summary["groups"], "total": summary["total"]}.items():
        print(f"  {name:18s} strike {g['market_strike_paid_weighted'] or float('nan'):8.2f} -> "
              f"{g['central_strike_paid_weighted'] or float('nan'):8.2f}  uncompensated "
              f"{g['market_uncompensated_twh']:7.2f} -> {g['central_uncompensated_twh']:7.2f} TWh  "
              f"CfD line {g['cfd_line_effect_gbp_m']:9.1f} m  per 3,100 kWh "
              f"{g.get('cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat', float('nan')):7.2f}")
    r = summary["reconciliation"]
    print(f"  paid-energy reconciliation: worst cohort gap {r['worst_cohort_gap']}, within tolerance {r['within_tolerance']}")
