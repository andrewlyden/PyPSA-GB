"""Settle every month of a central dispatch year against the market year, and total it.

Runs :mod:`scripts.market.fti_settlement` for each month, using the self-dispatch
schedule on the same forecast when present, and writes one table with the twelve
months and their sum. Months whose outputs are missing are reported and skipped, so
the script can run while a year is still finishing.

The headline is ``consumer_saving_fti_comparable_net_gbp_m``: the gross saving (the
market's consumer cost now carrying its interconnector re-trading), with the model's own
real-time correction taken back out, less the borrowed FTI forecast-error charge that
replaces it, each month's charge being its share of the year so the
twelve add to the annual figure (audit 26 Sep 2026; see
:mod:`scripts.market.fti_settlement`). Money is PyPSA money, 2024 prices.

Item D4(b): each side's interconnector cap-and-floor pass-through is part of its consumer
cost. A month is assessed against a twelfth of the annual cap and floor; YEAR is assessed
once on the per-cable rents summed over the months, and the consumer totals, gross and
headline for YEAR are rebuilt from the summed components (:func:`reassess_year`), so YEAR
is not the plain sum of the monthly pass-through rows.

The central year is perfect foresight (user decision, 27 Sep 2026): forecast error is
the borrowed FTI charge, so firm access settles against the market twin's own Stage 1
and ``--selfdispatch-suffix`` defaults to ``none``. A forecast-error central year
(``--suffix central_dispatch_fe``) needs ``--selfdispatch-suffix selfdispatch_fe``.

A market year that takes its Stage 1 from another family through ``wholesale_from``
(18 Oct 2026: the ``cp30_network`` months read ``stage1``'s) owns no Stage 1 files;
``--market-wholesale-scenarios`` names each month's parent, January to December, and the
market side's schedule, links and copperplate price are read from it.

Another study's year (FES 2050 decision 70, 25 Oct 2026: ``FES25_HT_2050_<Mon>_market`` against
``FES25_HT_2050_<Mon>_central_dispatch``) names its months outright with ``--central-scenarios``
and ``--market-scenarios`` (twelve each, January to December) in place of the CP30 stem, and
``--cfd-book`` settles the lost CfD support on a 2050 contract book's cohorts and strikes
(:func:`scripts.market.fti_settlement.book_cfd_terms`).

Usage::

    python -m scripts.market.fti_settle_year --suffix cp30_network_central_dispatch
        [--market-suffix cp30_network]
        [--selfdispatch-suffix none] [--forecast-error-basis renewable_capacity]
        [--out path.csv]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.market.fti_settlement import (
    DEFAULT_ASSUMPTIONS, GB_RESOURCE_SAVING, HEADLINE_METRIC, HEADLINE_METRIC_IN_FORCE, IC_RENT_CENTRAL_PREFIX,
    IC_RENT_MARKET_PREFIX, MARKET_DIR, PRICE_BASIS, SUPPORT_LINE_ROWS, UNIT_TRANSFER_COLUMNS, YEAR_HOURS,
    add_firm_access_support_argument, add_forecast_error_arguments, check_unit_transfers, consumer_totals,
    forecast_error_label, forecast_error_settings, load_interconnector_rule, settle, year_pass_through,
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STEM = "CP30_FFR_2030_{m}_{suffix}"
SUM_ROWS = [
    "gb_demand_twh", "market_energy_gbp_m", "market_constraint_cost_gbp_m",
    "market_interconnector_retrading_gbp_m", "market_interconnector_retrading_accounting_info_gbp_m",
    "market_redispatch_volume_twh", "central_energy_gbp_m", "central_cost_recovery_gbp_m",
    "central_cost_recovery_gross_info_gbp_m", "central_firm_access_gbp_m",
    "central_firm_access_interconnectors_excluded_info_gbp_m", "central_realtime_correction_gbp_m",
    # Decision D7: the two transfers at the energy line's price (information), and C11's premium.
    "central_firm_access_at_energy_price_info_gbp_m", "central_cost_recovery_at_energy_price_info_gbp_m",
    "central_counter_trade_premium_gbp_m", "central_unplanned_start_cost_gbp_m",
    "central_realtime_bm_net_gbp_m", "central_realtime_volume_twh", "market_resource_cost_gbp_m",
    "central_resource_cost_gbp_m", "central_schedule_resource_cost_gbp_m",
    "market_consumer_cost_gbp_m", "central_consumer_cost_gbp_m", "consumer_saving_gross_gbp_m",
    "central_realtime_correction_replaced_gbp_m", "forecast_error_charge_gbp_m", HEADLINE_METRIC,
    "resource_saving_gbp_m", "market_constrained_off_lost_margin_info_gbp_m", "hours",
    # Item D4(b): the interconnector pass-through (reassessed once for the year below)
    # and the congestion-rent transfer, which enters no total.
    "market_ic_congestion_rent_gbp_m", "central_ic_congestion_rent_gbp_m",
    "market_ic_pass_through_credit_gbp_m", "central_ic_pass_through_credit_gbp_m",
    "ic_congestion_rent_transfer_info_gbp_m", "ic_congestion_rent_transfer_gb_half_info_gbp_m",
    "ic_congestion_rent_transfer_foreign_half_info_gbp_m",
    "ic_congestion_rent_transfer_via_regimes_to_consumers_info_gbp_m",
    # Decisions of 14 Oct 2026 (lost support, storage and interconnector make-whole) and the
    # market constraint bases of 20 Oct 2026.
    "central_firm_access_margin_gbp_m", "central_firm_access_cfd_support_gbp_m",
    "central_firm_access_roc_support_gbp_m", "central_storage_make_whole_gbp_m", "central_ic_make_whole_gbp_m",
    "central_ic_congestion_rent_before_make_whole_info_gbp_m", "market_constraint_cost_in_force_gbp_m",
    "market_constraint_pro_rata_uplift_gbp_m", "market_consumer_cost_in_force_basis_gbp_m", HEADLINE_METRIC_IN_FORCE,
    # Decision of 20 Oct 2026: each side's CfD and RO lines on its metered output.
    *SUPPORT_LINE_ROWS, "consumer_saving_fti_comparable_net_excluding_support_lines_info_gbp_m",
    # The market's counter-trade premium and the saving with both premiums paid abroad.
    "market_counter_trade_premium_gbp_m", GB_RESOURCE_SAVING,
    # P1/P2 under central dispatch (20 Oct 2026): each schedule's payment abroad (information) and the
    # lost day-ahead volume firm access pays support on.
    "market_ic_schedule_trade_cost_info_gbp_m", "central_ic_schedule_trade_cost_info_gbp_m",
    "central_ic_physical_trade_cost_info_gbp_m", "central_firm_access_cfd_lost_twh", "central_firm_access_roc_lost_twh",
]
#: Rows the annual regime assessment and the consumer totals recompute for YEAR
#: (a month's cap and floor were its share of the year; the year is assessed once).
YEAR_RECOMPUTED = [
    "market_ic_pass_through_credit_gbp_m", "central_ic_pass_through_credit_gbp_m",
    "ic_congestion_rent_transfer_via_regimes_to_consumers_info_gbp_m", "ic_pass_through_year_share",
    "market_consumer_cost_gbp_m", "central_consumer_cost_gbp_m", "consumer_saving_gross_gbp_m",
    "central_realtime_correction_replaced_gbp_m", "forecast_error_charge_gbp_m", HEADLINE_METRIC,
    "market_consumer_cost_in_force_basis_gbp_m", HEADLINE_METRIC_IN_FORCE,
    "consumer_saving_fti_comparable_net_excluding_support_lines_info_gbp_m",
]


def reassess_year(table, rule, fe, hours: float) -> dict:
    """YEAR values for :data:`YEAR_RECOMPUTED`: one regime assessment on the summed per-cable rents.

    ``table`` holds the monthly columns (metric rows); the per-cable rent rows carry
    the prefixes :data:`IC_RENT_MARKET_PREFIX` / :data:`IC_RENT_CENTRAL_PREFIX`.
    The consumer totals are then rebuilt from the summed components, which is
    exact because every other component is additive over months.
    """
    months = table.drop(columns=[c for c in ("YEAR",) if c in table.columns])
    # 2040 (11 Nov 2026): a full leap year has 8,784 hours and is one year, not 1.0027 of one.
    year_hours = max(float(hours), YEAR_HOURS)
    ic = year_pass_through(months, rule, hours, year_hours=year_hours)
    summed = lambda k: float(pd.to_numeric(months.loc[k], errors="coerce").sum())  # noqa: E731
    r = {k: summed(k) for k in (
        "market_energy_gbp_m", "market_constraint_cost_gbp_m", "market_interconnector_retrading_gbp_m",
        "central_energy_gbp_m", "central_cost_recovery_gbp_m", "central_firm_access_gbp_m",
        "central_realtime_correction_gbp_m", "gb_demand_twh", "central_storage_make_whole_gbp_m",
        "central_ic_make_whole_gbp_m", "market_constraint_pro_rata_uplift_gbp_m", *SUPPORT_LINE_ROWS)}
    r.update(ic)
    r.update(consumer_totals(r, fe, hours=hours, year_hours=year_hours))
    return {k: r[k] for k in YEAR_RECOMPUTED}


def units_path(settlement_path) -> Path:
    """The per-unit transfers file written beside a year settlement (``..._units.csv``)."""
    path = Path(settlement_path)
    return path.with_name(f"{path.stem}_units.csv")


def month_scenarios(names, flag: str) -> dict | None:
    """Month -> scenario for twelve names listed January to December (``None`` when not given)."""
    if names is None:
        return None
    names = list(names)
    if len(names) != len(MONTHS):
        raise ValueError(f"{flag} needs {len(MONTHS)} names (January to December), got {len(names)}")
    for m, name in zip(MONTHS, names):
        if f"_{m}" not in name:
            raise ValueError(f"{flag}: {name} is not a {m} month; list them January to December")
    return dict(zip(MONTHS, names))


def market_stage_1(market_suffix: str, wholesale_scenarios=None, market_scenarios=None) -> dict:
    """Month -> the scenario whose Stage 1 files the market month read (its ``wholesale_from`` parent).

    ``wholesale_scenarios`` lists the twelve parents January to December; ``None`` means every
    market month owns its Stage 1.
    """
    if wholesale_scenarios is None:
        if market_scenarios is not None:
            return dict(market_scenarios)
        return {m: STEM.format(m=m, suffix=market_suffix) for m in MONTHS}
    names = list(wholesale_scenarios)
    if len(names) != len(MONTHS):
        raise ValueError(f"--market-wholesale-scenarios needs {len(MONTHS)} names (January to December), got {len(names)}")
    for m, name in zip(MONTHS, names):
        if f"_{m}" not in name:
            raise ValueError(f"--market-wholesale-scenarios: {name} is not a {m} month; list them January to December")
    return dict(zip(MONTHS, names))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suffix", default="cp30_network_central_dispatch")
    ap.add_argument("--market-suffix", default="cp30_network")
    ap.add_argument("--market-wholesale-scenarios", nargs=12, default=None, metavar="SCENARIO",
                    help="each market month's Stage 1 month (its wholesale_from parent), January to December; "
                         "default the market months themselves")
    ap.add_argument("--selfdispatch-suffix", default="none",
                    help="self-dispatch-on-forecast scenario suffix; 'none' for a perfect-foresight central year")
    ap.add_argument("--national-price", default="copperplate")
    add_forecast_error_arguments(ap)
    ap.add_argument("--assumptions", default=str(DEFAULT_ASSUMPTIONS),
                    help="forecast bill assumptions JSON (interconnector cap-and-floor rule, item D4(b))")
    add_firm_access_support_argument(ap)
    ap.add_argument("--central-scenarios", nargs=12, default=None, metavar="SCENARIO",
                    help="the central-dispatch months, January to December, in place of the CP30 stem and --suffix")
    ap.add_argument("--market-scenarios", nargs=12, default=None, metavar="SCENARIO",
                    help="the market months, January to December, in place of the CP30 stem and --market-suffix")
    ap.add_argument("--cfd-book", default=None,
                    help="a 2050 contract book: firm access pays lost CfD support on its cohorts and strikes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--units-out", default=None,
                    help="per-unit transfers (cost recovery, firm access, storage make-whole); default beside --out")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    fe = forecast_error_settings(a)
    rule = load_interconnector_rule(a.assumptions)

    sd_suffix = None if str(a.selfdispatch_suffix or "").strip().lower() in ("", "none") else a.selfdispatch_suffix
    central_named = month_scenarios(a.central_scenarios, "--central-scenarios")
    market_named = month_scenarios(a.market_scenarios, "--market-scenarios")
    if (central_named is None) != (market_named is None):
        ap.error("--central-scenarios and --market-scenarios go together")
    if central_named is not None and sd_suffix:
        ap.error("named months settle a perfect-foresight central year (--selfdispatch-suffix none)")
    cfd_book = None
    if a.cfd_book:
        from scripts.public_ownership.contract_book_2050 import read_contract_book

        cfd_book = read_contract_book(a.cfd_book)
    cols, units = {}, []
    stage_1 = market_stage_1(a.market_suffix, a.market_wholesale_scenarios, market_named)
    for m in MONTHS:
        central = central_named[m] if central_named else STEM.format(m=m, suffix=a.suffix)
        market = market_named[m] if market_named else STEM.format(m=m, suffix=a.market_suffix)
        sd = STEM.format(m=m, suffix=sd_suffix) if sd_suffix else None
        needed = [MARKET_DIR / f"{central}_balancing.nc", MARKET_DIR / f"{central}_constraint_costs.csv",
                  MARKET_DIR / f"{market}_balancing.nc"]
        if stage_1[m] != market:
            needed.append(MARKET_DIR / f"{stage_1[m]}_wholesale_dispatch.csv")
        if sd and not (MARKET_DIR / f"{sd}_wholesale_dispatch.csv").exists():
            print(f"{m}: no self-dispatch schedule on the forecast; using the market copperplate schedule")
            sd = None
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            print(f"{m}: skipped, missing {missing}")
            continue
        print(f"{m}: settling", flush=True)
        month, month_units = settle(central, market, national_price_basis=a.national_price, selfdispatch=sd,
                                    forecast_error=fe, assumptions=a.assumptions,
                                    market_wholesale=None if stage_1[m] == market else stage_1[m],
                                    firm_access_support=a.firm_access_support, units=True, cfd_book=cfd_book)
        cols[m] = month["value"]
        check_unit_transfers(month_units, cols[m], where=f"{central}")
        units.append(month_units.assign(month=m, central_scenario=central, market_scenario=market))

    if not cols:
        print("nothing to settle yet")
        return 1
    table = pd.DataFrame(cols)
    rent_rows = [r for r in table.index if str(r).startswith((IC_RENT_MARKET_PREFIX, IC_RENT_CENTRAL_PREFIX))]
    num = table.loc[[r for r in SUM_ROWS + rent_rows if r in table.index]].astype(float)
    hours = float(pd.to_numeric(table.loc["hours"], errors="coerce").sum())
    year = reassess_year(table, rule, fe, hours)
    table["YEAR"] = ""
    # min_count: a row every month leaves blank (the accounting cross-check on a
    # windowed month) stays blank rather than summing to zero.
    table.loc[num.index, "YEAR"] = num.sum(axis=1, min_count=1)
    for key, value in year.items():
        table.loc[key, "YEAR"] = value
    for label, value in (("price_basis", PRICE_BASIS), ("headline_metric", HEADLINE_METRIC),
                         ("forecast_error_charge_basis", forecast_error_label(fe))):
        table.loc[label, "YEAR"] = value
    table.loc["months_settled", :] = ""
    table.loc["months_settled", "YEAR"] = len(cols)
    out = Path(a.out) if a.out else MARKET_DIR / f"CP30_FFR_2030_year_{a.suffix}_fti_settlement_{a.national_price}.csv"
    table.to_csv(out)
    # P1/P2 under central dispatch (20 Oct 2026): each unit's transfers, month by month, beside the table.
    units_out = Path(a.units_out) if a.units_out else units_path(out)
    pd.concat(units, ignore_index=True)[["month", "central_scenario", "market_scenario", *UNIT_TRANSFER_COLUMNS]] \
        .to_csv(units_out, index=False)

    show = ["market_energy_gbp_m", "market_constraint_cost_gbp_m", "market_interconnector_retrading_gbp_m",
            "market_ic_pass_through_credit_gbp_m", "central_ic_pass_through_credit_gbp_m",
            "ic_congestion_rent_transfer_info_gbp_m",
            "central_cost_recovery_gbp_m", "central_firm_access_gbp_m", "central_firm_access_cfd_support_gbp_m",
            "central_firm_access_roc_support_gbp_m", "central_storage_make_whole_gbp_m",
            "central_ic_make_whole_gbp_m", "central_realtime_correction_gbp_m", *SUPPORT_LINE_ROWS,
            "market_consumer_cost_gbp_m", "central_consumer_cost_gbp_m", "consumer_saving_gross_gbp_m",
            "central_realtime_correction_replaced_gbp_m", "forecast_error_charge_gbp_m", HEADLINE_METRIC,
            "market_constraint_pro_rata_uplift_gbp_m", HEADLINE_METRIC_IN_FORCE, "resource_saving_gbp_m",
            "market_counter_trade_premium_gbp_m", GB_RESOURCE_SAVING]
    pd.set_option("display.width", 200)
    print()
    shown = num.loc[[s for s in show if s in num.index]].round(1)
    shown["YEAR"] = [round(float(table.loc[s, "YEAR"]), 1) for s in shown.index]
    print(shown.to_string())
    print(f"\n{len(cols)} months settled; written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
