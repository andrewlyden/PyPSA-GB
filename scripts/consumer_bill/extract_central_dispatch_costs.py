"""The bill's constraint line under central dispatch (institution I3), from the year FTI settlement.

P1 and P2 under central dispatch (user decisions of 14 and 20 Oct 2026). A forecast case
with ``institution: central_dispatch`` is the market case's bill re-projected on its
central-dispatch year, settled the way FTI settle it (:mod:`scripts.market.fti_settlement`,
:mod:`scripts.market.fti_settle_year`). This writes the balancing-cost summary the projector
reads (``{bill_case}_future_balancing_costs_summary.json``, the schema
``extract_future_balancing_costs`` writes for a market year), with the constraint line

    cost recovery + firm access (its margin and the lost CfD and ROC support)
    + storage weekly make-whole + interconnector day-ahead make-whole
    + the borrowed FTI forecast-error charge (+ the central balancing stage's counter-trade premium)

and interconnector re-trading at zero: there is no balancing market to re-trade in. The
model's own real-time correction is taken back out, because the borrowed charge replaces it
(user decision of 26 Sep 2026; it is recorded beside, ``realtime_correction_replaced_gbp``).
The forecast-error charge is recovered on this line, as for P3 (decision 9).

The other lines the institution moves are the projector's, from the case's own inputs:

* **wholesale and hedging**: unchanged. The national price is the market twin's copperplate
  price (decision 2), so the case reads the market months' Stage 1 files;
* **CfD**: on metered central output (decision 1): the case's physical networks are the
  central-dispatch months (``central_scenario_pattern``), the reference price the national price;
* **interconnector cap-and-floor pass-through**: on the central schedule's rent including the
  day-ahead make-whole (decision 4). The bill's rent is the market summaries'; the change each
  cable's rent undergoes, central less market on the settlement's own convention, is written
  here (``interconnector_rent_change_by_cable_gbp``) and added to it, so the institution moves
  the rent and nothing else does;
* **storage**: the weekly make-whole counts as storage revenue before the LDES floor test and in
  P2's storage route (decision 3): the twin run group's asset settlement books it.

The extraction refuses a market year and a central-dispatch year from different solve runs
(``nationalised_institutions.solve_vintage``, ``mixed_vintage``), a settlement short of twelve
months, any national price but the copperplate one and any firm-access basis but lost support.

Money is PyPSA money, GBP real 2024 (the settlement's), as the market extraction's is.

Usage::

    python scripts/consumer_bill/extract_central_dispatch_costs.py SETTLEMENT MARKET_SUMMARY
        OUT_MONTHLY OUT_CARRIERS OUT_SUMMARY --model-price-year 2024
        [--units SETTLEMENT_UNITS] [--market-scenarios ... --central-scenarios ... --market-wholesale-scenarios ...]
        [--mixed-vintage-gap-hours 24] [--no-counter-trade-premium]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.market.fti_settlement import (  # noqa: E402
    IC_RENT_CENTRAL_PREFIX, IC_RENT_MARKET_PREFIX, UNIT_TRANSFER_ROWS,
)

INSTITUTION = "central_dispatch"
YEAR = "YEAR"
#: The constraint line's components under I3, GBP, and the settlement row (YEAR, GBP m) each is read from.
CONSTRAINT_PARTS = {
    "cost_recovery_gbp": "central_cost_recovery_gbp_m",
    "firm_access_gbp": "central_firm_access_gbp_m",
    "storage_make_whole_gbp": "central_storage_make_whole_gbp_m",
    "interconnector_make_whole_gbp": "central_ic_make_whole_gbp_m",
    "forecast_error_charge_gbp": "forecast_error_charge_gbp_m",
    "counter_trade_premium_gbp": "central_counter_trade_premium_gbp_m",
}
#: Settlement rows carried for information (YEAR, GBP m unless named otherwise).
INFO_ROWS = {
    "firm_access_margin_gbp": "central_firm_access_margin_gbp_m",
    "firm_access_cfd_support_gbp": "central_firm_access_cfd_support_gbp_m",
    "firm_access_roc_support_gbp": "central_firm_access_roc_support_gbp_m",
    "realtime_correction_replaced_gbp": "central_realtime_correction_replaced_gbp_m",
    "realtime_correction_gbp": "central_realtime_correction_gbp_m",
    "central_ic_pass_through_credit_settlement_gbp": "central_ic_pass_through_credit_gbp_m",
    "market_ic_pass_through_credit_settlement_gbp": "market_ic_pass_through_credit_gbp_m",
    "market_constraint_cost_settlement_gbp": "market_constraint_cost_gbp_m",
    "market_interconnector_retrading_settlement_gbp": "market_interconnector_retrading_gbp_m",
    "fti_market_consumer_cost_gbp": "market_consumer_cost_gbp_m",
    "fti_central_consumer_cost_gbp": "central_consumer_cost_gbp_m",
    "fti_consumer_saving_headline_gbp": "consumer_saving_fti_comparable_net_gbp_m",
    "fti_resource_saving_gbp": "resource_saving_gbp_m",
    "market_ic_schedule_trade_cost_gbp": "market_ic_schedule_trade_cost_info_gbp_m",
    "central_ic_schedule_trade_cost_gbp": "central_ic_schedule_trade_cost_info_gbp_m",
    "central_ic_physical_trade_cost_gbp": "central_ic_physical_trade_cost_info_gbp_m",
}
FULL_YEAR_HOURS = (8760.0, 8784.0)
#: Decision 1 (20 Oct 2026): firm access pays the lost support; decision 2: the copperplate national price.
REQUIRED_FIRM_ACCESS_BASIS = "lost_support"
REQUIRED_NATIONAL_PRICE = "copperplate"


def _value(table: pd.DataFrame, row: str, column: str = YEAR, required: bool = True) -> float:
    if row not in table.index:
        if required:
            raise KeyError(f"the FTI settlement has no row {row!r}; rebuild it (scripts.market.fti_settle_year)")
        return float("nan")
    value = pd.to_numeric(pd.Series([table.at[row, column]]), errors="coerce").iloc[0]
    if required and not np.isfinite(value):
        raise ValueError(f"the FTI settlement's {row} ({column}) is not a number")
    return float(value)


def month_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c != YEAR]


def check_settlement(table: pd.DataFrame) -> None:
    """Refuse a settlement this bill cannot be built on (decisions 1 and 2, a whole calendar year)."""
    months = month_columns(table)
    settled = _value(table, "months_settled")
    if len(months) != 12 or settled != 12:
        raise ValueError(f"the FTI settlement covers {int(settled)} months ({months}); the I3 bill needs all twelve")
    hours = _value(table, "hours")
    if hours not in FULL_YEAR_HOURS:
        raise ValueError(f"the FTI settlement covers {hours:g} hours, not one calendar year")
    for row, wanted in (("firm_access_support_basis", REQUIRED_FIRM_ACCESS_BASIS),
                        ("national_price_basis", REQUIRED_NATIONAL_PRICE)):
        if row not in table.index:
            raise KeyError(f"the FTI settlement has no row {row!r}; it predates the decisions of 14 Oct 2026")
        found = sorted({str(table.at[row, m]) for m in months})
        if found != [wanted]:
            raise ValueError(f"the I3 bill is built on {row} {wanted!r} (user decisions of 14 and 20 Oct 2026); the "
                             f"settlement has {found}")


def rent_change_by_cable(table: pd.DataFrame) -> dict[str, float]:
    """Per cable, the central schedule's rent (with the day-ahead make-whole) less the market's, GBP over the year."""
    cables = sorted({str(r)[len(p):] for r in table.index for p in (IC_RENT_MARKET_PREFIX, IC_RENT_CENTRAL_PREFIX)
                     if str(r).startswith(p)})
    out = {}
    for cable in cables:
        central = _value(table, f"{IC_RENT_CENTRAL_PREFIX}{cable}", required=False)
        market = _value(table, f"{IC_RENT_MARKET_PREFIX}{cable}", required=False)
        out[cable] = (np.nan_to_num(central) - np.nan_to_num(market)) * 1e6
    return out


def constraint_parts(table: pd.DataFrame, column: str = YEAR, counter_trade_premium: bool = True) -> dict:
    """The I3 constraint line's components in GBP for one column of the settlement (a month or YEAR)."""
    parts = {key: _value(table, row, column) * 1e6 for key, row in CONSTRAINT_PARTS.items()}
    if not counter_trade_premium:
        parts["counter_trade_premium_gbp"] = 0.0
    return parts


def fti_counter_trade_treatment(parts: Mapping, market_constraint_gbp: float, on_line: bool) -> dict:
    """The I3 line and its consumer saving with the counter-trade premium treated as FTI treat it (27 Oct 2026).

    The premium stays on the I3 constraint line (user decision); FTI count it inside the real-time
    correction (the settlement's ``central_realtime_correction_gbp_m`` includes it), which their
    forecast-error charge replaces, so on their reading it is not charged on top. A sensitivity, no
    headline change: the line and the institution step without the premium, beside the headline.
    """
    premium = float(parts.get("counter_trade_premium_gbp", 0.0)) if on_line else 0.0
    line = float(sum(parts.values()))
    fti_line = line - premium
    return {
        "counter_trade_premium_gbp": premium,
        "headline_constraint_cost_gbp": line,
        "constraint_cost_gbp": fti_line,
        "headline_consumer_saving_on_constraint_line_gbp": float(market_constraint_gbp) - line,
        "consumer_saving_on_constraint_line_gbp": float(market_constraint_gbp) - fti_line,
        "institution_step_on_constraint_line_gbp": fti_line - float(market_constraint_gbp),
        "basis": ("FTI's treatment: the central balancing stage's counter-trade premium is part of the real-time "
                  "correction (central_realtime_correction_gbp_m) that the borrowed forecast-error charge replaces, "
                  "so it is not charged on top; the headline keeps it on the line (user decision)"
                  if on_line else "the line already leaves the premium off (--no-counter-trade-premium)"),
    }


def monthly_frame(table: pd.DataFrame, counter_trade_premium: bool = True) -> pd.DataFrame:
    """Each settled month's components (GBP). A month's forecast-error charge is its share of the year's."""
    rows = []
    for month in month_columns(table):
        parts = constraint_parts(table, month, counter_trade_premium)
        rows.append({"month": month, "hours": _value(table, "hours", month), **parts,
                     "constraint_cost_gbp": float(sum(parts.values())),
                     **{key: _value(table, row, month, required=False) * 1e6 for key, row in INFO_ROWS.items()}})
    return pd.DataFrame(rows)


def carrier_frame(units: pd.DataFrame | None) -> pd.DataFrame:
    """The transfers by carrier, GBP (from the settlement's per-unit file), largest first; empty without it."""
    columns = ["carrier", *UNIT_TRANSFER_ROWS]
    if units is None or units.empty:
        return pd.DataFrame(columns=columns)
    out = units.groupby("carrier", as_index=False)[list(UNIT_TRANSFER_ROWS)].sum()
    return out.assign(_total=out[list(UNIT_TRANSFER_ROWS)].sum(axis=1)).sort_values(
        "_total", ascending=False).drop(columns="_total")[columns]


def central_dispatch_costs(settlement: pd.DataFrame, market_summary: Mapping, *, model_price_year: int,
                           units: pd.DataFrame | None = None, vintage: Mapping | None = None,
                           counter_trade_premium: bool = True, allow_mixed_vintage: bool = False):
    """The I3 balancing-cost summary: ``(monthly, by_carrier, annual)``.

    ``settlement`` is the year FTI settlement (metric rows, the months and ``YEAR``);
    ``market_summary`` the market twin's own balancing-cost summary (``annual``), read for
    the I1 line beside the I3 one and for the balancing bids' negative-price rule, which
    the lost CfD support honours too (FTI 2025 slide 44); ``units`` the per-unit transfers
    (the by-carrier table); ``vintage`` :func:`nationalised_institutions.solve_vintage` of
    the two years, refused when ``mixed_vintage``.
    """
    check_settlement(settlement)
    if vintage is not None and vintage.get("mixed_vintage") and not allow_mixed_vintage:
        raise ValueError(
            "the market year and the central-dispatch year come from different solve runs (mixed_vintage, "
            f"{vintage.get('gap_between_runs_hours', float('nan')):.0f} h apart): the institution step would compare "
            "two vintages. Re-solve the central-dispatch year on the market year's build first")
    market = dict(market_summary.get("annual") or market_summary)
    if int(market["model_price_year"]) != int(model_price_year):
        raise ValueError(f"the market extraction is on {market['model_price_year']}, not {model_price_year}")
    if market.get("institution") == INSTITUTION:
        raise ValueError("the market twin's balancing costs are themselves a central-dispatch extraction")
    if units is not None:
        from scripts.market.fti_settlement import check_unit_transfers

        rows = {row: _value(settlement, row) for row in UNIT_TRANSFER_ROWS.values()}
        check_unit_transfers(units, rows, where="the per-unit transfers against the year settlement")
    parts = constraint_parts(settlement, YEAR, counter_trade_premium)
    constraint = float(sum(parts.values()))
    info = {key: _value(settlement, row, required=False) * 1e6 for key, row in INFO_ROWS.items()}
    info = {key: (value if np.isfinite(value) else None) for key, value in info.items()}
    info["firm_access_cfd_support_gbp"] = _value(settlement, INFO_ROWS["firm_access_cfd_support_gbp"]) * 1e6
    demand_mwh = _value(settlement, "gb_demand_twh") * 1e6
    lost_cfd_mwh = _value(settlement, "central_firm_access_cfd_lost_twh", required=False) * 1e6
    monthly = monthly_frame(settlement, counter_trade_premium)
    by_carrier = carrier_frame(units)
    i1 = float(market["constraint_cost_gbp"])
    annual = {
        "institution": INSTITUTION,
        "reported_period_hours": _value(settlement, "hours"),
        "demand_mwh": demand_mwh,
        "model_price_year": int(model_price_year),
        **parts,
        "constraint_cost_gbp": constraint,
        "constraint_cost_parts": list(CONSTRAINT_PARTS),
        "counter_trade_premium_on_constraint_line": bool(counter_trade_premium),
        **info,
        # The schema the projector reads, filled for central dispatch: no balancing market, so no
        # accepted offers or bids, no feasibility slack or penalty turn-down inside them, no re-trading.
        "constraint_cost_excludes_slack_and_penalty": True,
        "accepted_offer_cost_gbp": 0.0,
        "accepted_bid_cost_gbp": 0.0,
        "feasibility_slack_cost_gbp": 0.0,
        "feasibility_slack_energy_mwh": 0.0,
        "feasibility_slack_cost_in_accepted_costs_gbp": 0.0,
        "non_participant_turn_down_cost_at_penalty_gbp": 0.0,
        "non_participant_turn_down_mwh": 0.0,
        "tie_break_step_cost_gbp": 0.0,
        "interconnector_retrading_cost_gbp": 0.0,
        "include_interconnector_retrading": False,
        # The lost CfD top-up on the turned-down day-ahead volume is paid as firm access (decision 1), not
        # through a bid; the CfD line stays on metered central output.
        "cfd_curtailment_compensation_gbp": info["firm_access_cfd_support_gbp"],
        "cfd_curtailed_mwh": lost_cfd_mwh if np.isfinite(lost_cfd_mwh) else 0.0,
        "cfd_curtailment_compensation_treatment": "firm_access_lost_support",
        "balancing_negative_price_rule": market.get("balancing_negative_price_rule"),
        "interconnector_rent_change_by_cable_gbp": rent_change_by_cable(settlement),
        "solve_vintage": dict(vintage) if vintage is not None else None,
        "mixed_vintage_allowed": bool(allow_mixed_vintage),
        # The market twin's line, and the institution's step on it.
        "market_constraint_cost_gbp": i1,
        "market_interconnector_retrading_cost_gbp": float(market.get("interconnector_retrading_cost_gbp") or 0.0),
        "market_include_interconnector_retrading": bool(market.get("include_interconnector_retrading")),
        "institution_step_on_constraint_line_gbp": constraint - i1,
        # 27 Oct 2026: the sensitivity with the counter-trade premium inside the replaced real-time correction.
        "counter_trade_premium_fti_treatment": fti_counter_trade_treatment(parts, i1, counter_trade_premium),
        "unlabelled_support": market.get("unlabelled_support", []),
        "constraint_cost_gbp_per_mwh_demand": constraint / demand_mwh if demand_mwh else None,
        "price_basis": "GBP real 2024 (PyPSA model money), as the FTI settlement",
        "method": (
            "Central dispatch on FTI's method (user decisions of 14 and 20 Oct 2026): constraint cost = cost "
            "recovery (output run above self-dispatch, its cost above the national price) + firm access (the "
            "national price less cost on output self-dispatch would have run, with the lost CfD top-up and ROC "
            "value) + the storage weekly make-whole + the interconnector day-ahead make-whole + the borrowed FTI "
            "forecast-error charge"
            + (" + the counter-trade premium the central balancing stage was solved with" if counter_trade_premium
               else "")
            + ", summed over twelve settled months. The model's own real-time correction is taken back out: the "
            "borrowed charge replaces it. Interconnector re-trading is zero: there is no balancing market to re-trade "
            "in."),
        "limitations": [
            "The forecast-error charge is FTI's, scaled to this fleet (scripts/market/forecast_error_charge.py); the "
            "model is perfect foresight and produces almost none itself.",
            "Firm access and cost recovery are valued against the market twin's copperplate schedule on the outturn "
            "(a perfect-foresight central year has no separate self-dispatch-on-forecast schedule).",
            "Storage is made whole week by week inside each month; a week a month boundary splits is assessed in two "
            "parts.",
        ],
    }
    return monthly, by_carrier, annual


def write_central_dispatch_costs(settlement_path, market_summary_path, output_monthly, output_carriers,
                                 output_summary, *, model_price_year: int, units_path=None, vintage=None,
                                 counter_trade_premium: bool = True, allow_mixed_vintage: bool = False) -> dict:
    settlement = pd.read_csv(settlement_path, index_col=0)
    units = pd.read_csv(units_path) if units_path else None
    market = json.loads(Path(market_summary_path).read_text(encoding="utf-8"))
    monthly, by_carrier, annual = central_dispatch_costs(
        settlement, market, model_price_year=model_price_year, units=units, vintage=vintage,
        counter_trade_premium=counter_trade_premium, allow_mixed_vintage=allow_mixed_vintage)
    annual["fti_settlement"] = str(settlement_path)
    annual["market_balancing_costs"] = str(market_summary_path)
    for path in (output_monthly, output_carriers, output_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_monthly, index=False)
    by_carrier.to_csv(output_carriers, index=False)
    Path(output_summary).write_text(json.dumps({"annual": annual, "by_carrier": by_carrier.to_dict("records")},
                                               indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    return annual


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def solve_vintage_of(market_scenarios: Sequence[str], central_scenarios: Sequence[str],
                     market_wholesale_scenarios: Sequence[str] | None = None,
                     gap_hours: float | None = None) -> dict:
    """The two years' solve identities and whether they are one run (``nationalised_institutions``)."""
    from scripts.public_ownership.nationalised_institutions import (
        MIXED_VINTAGE_GAP_HOURS_DEFAULT, solve_identity, solve_vintage,
    )

    return solve_vintage(solve_identity(list(market_scenarios), market_wholesale_scenarios),
                         solve_identity(list(central_scenarios)),
                         MIXED_VINTAGE_GAP_HOURS_DEFAULT if gap_hours is None else float(gap_hours))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("settlement", "market_summary", "out_monthly", "out_carriers", "out_summary"):
        parser.add_argument(name)
    parser.add_argument("--model-price-year", type=int, required=True)
    parser.add_argument("--units", default=None, help="the settlement's per-unit transfers (..._units.csv)")
    parser.add_argument("--market-scenarios", nargs="*", default=None)
    parser.add_argument("--market-wholesale-scenarios", nargs="*", default=None)
    parser.add_argument("--central-scenarios", nargs="*", default=None)
    parser.add_argument("--mixed-vintage-gap-hours", type=float, default=None)
    parser.add_argument("--allow-mixed-vintage", action="store_true",
                        help="build on two solve runs' years anyway (a diagnostic; the summary records the gap)")
    parser.add_argument("--no-counter-trade-premium", action="store_true",
                        help="leave the central balancing stage's counter-trade premium off the constraint line")
    args = parser.parse_args(argv)
    vintage = None
    if args.market_scenarios and args.central_scenarios:
        vintage = solve_vintage_of(args.market_scenarios, args.central_scenarios,
                                   args.market_wholesale_scenarios or None, args.mixed_vintage_gap_hours)
    annual = write_central_dispatch_costs(
        args.settlement, args.market_summary, args.out_monthly, args.out_carriers, args.out_summary,
        model_price_year=args.model_price_year, units_path=args.units, vintage=vintage,
        counter_trade_premium=not args.no_counter_trade_premium, allow_mixed_vintage=args.allow_mixed_vintage)
    print(f"I3 constraint line GBP {annual['constraint_cost_gbp'] / 1e6:,.1f}M against the market's "
          f"GBP {annual['market_constraint_cost_gbp'] / 1e6:,.1f}M (step {annual['institution_step_on_constraint_line_gbp'] / 1e6:+,.1f}M)")


if __name__ == "__main__":
    main()
