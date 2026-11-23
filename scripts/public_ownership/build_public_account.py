"""WP-D command line: the public account by case and group, and what each rule routes.

Writes ``{run_group}_public_account.csv`` and ``{run_group}_public_account_summary.json``
from the settled finance register, the WP-B routes and the configured
recycling, shortfall, LDES cap-and-floor and Capacity Market decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.cfd_repricing import (
    DEFAULT_BATCH_CASES, IDENTITY_TOLERANCE_GBP, ROUTE_CFD_BILL, SavingsStream, limit_at_cost_to_deliverable,
    repricing_settings, settled_energy_by_key,
)
from scripts.public_ownership import capacity_market_state_bids as cm_state_bids
from scripts.public_ownership import capacity_procurement
from scripts.public_ownership.levers import LEVER_CAPACITY_PROCUREMENT
from scripts.public_ownership.public_account import account_settings, build_public_account, pay_as_bid_ceiling_gbp
from config.config_loader import read_config


def paid_energy_from_adjustments(adjustments: pd.DataFrame) -> dict[str, float]:
    """WP-B's paid energy per strike entry, carried on its adjustments (the same for every case)."""
    paid = (adjustments.loc[adjustments["case_id"].eq("private")].drop_duplicates("strike_key")
            .set_index("strike_key")["eligible_dispatch_mwh"])
    return {str(k): float(v) for k, v in paid.items()}


def check_all_cases_repriced(adjustments: pd.DataFrame, repriced: set) -> None:
    """Every case the adjustments cut must have been re-priced here (a case never streamed took no cut)."""
    expected = adjustments.groupby("case_id")["r1_saving_gbp_2024"].sum()
    missing = expected.loc[~expected.index.isin(list(repriced))]
    if (missing.abs() > IDENTITY_TOLERANCE_GBP).any():
        worst = missing.abs().idxmax()
        raise ValueError(f"{worst}: the account's R1 cut differs from the strike adjustments' by GBP "
                         f"{-missing[worst]:,.2f}; the two stages re-price differently (rebuild the strike "
                         "adjustments on this register)")


def reprice_as_the_bill_does(savings: pd.DataFrame, adjustments: pd.DataFrame, table,
                             settled_energy: pd.Series | None = None, cases=None) -> pd.DataFrame:
    """The R1 cuts the bill actually takes, row by row (review 29 Sep 2026).

    WP-B puts an at-cost contract whose rent its strike cannot carry back on the
    financing-saving basis (``limit_at_cost_to_deliverable``); the bill's CfD
    line then moves by the smaller cut and the rest stays with the public owner.
    Without the same step here the account took the whole at-cost cut off the
    state's rent, so the state appeared to fund GBP 4.05 m a year it keeps (the
    all-CfD-repriced boundary and the bounding pair), which the three-party
    identity caught.  The paid energy each strike cut divides by is WP-B's own,
    carried on the adjustments (``eligible_dispatch_mwh``, a physical quantity of
    the run, the same for every case).  Refuses unless each case's R1 cut then
    equals the adjustments' to GBP 1.

    30 Sep 2026: ``savings`` may be a batch of cases (``cases``, whose cuts are
    checked) without the private case, given ``settled_energy``
    (:func:`cfd_repricing.settled_energy_by_key` of the private savings); a
    case never passed here is checked by :func:`check_all_cases_repriced`.
    """
    paid = paid_energy_from_adjustments(adjustments)
    savings, limited = limit_at_cost_to_deliverable(savings, table, None, paid, settled_energy=settled_energy)
    cut = savings.loc[savings["route"].eq(ROUTE_CFD_BILL)].groupby("case_id")["saving_bill_term_gbp"].sum()
    expected = adjustments.groupby("case_id")["r1_saving_gbp_2024"].sum()
    if cases is not None:
        expected = expected.loc[expected.index.isin(list(cases))]
    gap = cut.reindex(expected.index, fill_value=0.0) - expected
    if (gap.abs() > IDENTITY_TOLERANCE_GBP).any():
        worst = gap.abs().idxmax()
        raise ValueError(f"{worst}: the account's R1 cut differs from the strike adjustments' by GBP {gap[worst]:,.2f}; "
                         "the two stages re-price differently (rebuild the strike adjustments on this register)")
    for row in limited.itertuples(index=False):
        print(f"public account: {row.strike_key} in {row.case_id} kept on the financing saving, as the bill keeps it")
    return savings


def merge_accounts(accounts: list[pd.DataFrame], summaries: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Batches of :func:`public_account.build_public_account` as one account: sorted by case and group, as one call sorts."""
    account = pd.concat(accounts, ignore_index=True).sort_values(["case_id", "group"]).reset_index(drop=True)
    merged = {}
    for summary in summaries:
        merged.update(summary)
    return account, {case_id: merged[case_id] for case_id in sorted(merged)}


def capacity_market_line_is_nil(assumptions_path) -> bool:
    """Whether the bill's Capacity Market line is a declared nil (``value_basis: nil``), so it books nothing."""
    if assumptions_path is None:
        return False
    document = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    block = (document.get("sourced_externals") or {}).get("capacity_market") or {}
    return block.get("value_basis") == "nil"


def capacity_market_state_bids(summary: dict, rows: pd.DataFrame, private_rows: pd.DataFrame, auction, settings) -> None:
    """F9: each market case's 2030/31 T-4 re-cleared with its state plant bidding at its own cost (in place)."""
    for case_id, case_rows in rows.groupby("case_id", sort=False):
        if case_id not in summary:
            continue
        result = cm_state_bids.reclear(auction, cm_state_bids.row_bid_factors(case_rows, private_rows), settings)
        summary[case_id][cm_state_bids.CONFIG_KEY] = result


def capacity_procurement_accounts(summary: dict, rows: pd.DataFrame, private_rows: pd.DataFrame,
                                  settings_by_case, params) -> None:
    """F27 item 6: each market case under ``capacity_procurement: cost_based_existing`` carries its saving (in place).

    Runs after F9, so a case F9 re-clears has its existing units' receipts scaled by F9's own
    calendar-2030 share change first (:func:`capacity_procurement.f9_scale`); private's F9 block
    reproduces the builder (no change), so its absence from a batch reads as no change.
    """
    for case_id, case_rows in rows.groupby("case_id", sort=False):
        if case_id not in summary:
            continue
        value = str((settings_by_case.get(case_id) or {}).get(LEVER_CAPACITY_PROCUREMENT, capacity_procurement.AUCTION))
        if value != capacity_procurement.COST_BASED_EXISTING:
            continue
        scale = capacity_procurement.f9_scale(summary[case_id], summary.get("private"))
        summary[case_id][capacity_procurement.CONFIG_KEY] = capacity_procurement.account_result(
            private_rows, case_rows, params, receipt_scale=scale)


def stamp_price_level(report: dict, price_level=None) -> dict:
    """Item F12 (12 Oct 2026): report with price_level when it is not the model's own (a twin's account)."""
    if price_level is None or str(price_level) == "model":
        return report
    return {**report, "price_level": str(price_level)}


def write_public_account(register_path, public_ownership_config, account_output, summary_output,
                         adjustments_path=None, assumptions_path=None, model_price_year=None, strike_table_path=None,
                         batch_cases: int = DEFAULT_BATCH_CASES, cm_auction_paths=None, price_level=None):
    """Write the account and its summary.

    ``price_level`` (item F12, 12 Oct 2026): the wholesale price level the register was
    settled at, written as the summary's ``price_level`` when it is not the model's own
    (a price-level twin run group's account), so the bills reading the account can say
    which of their lines it re-prices. None or ``model`` writes the summary as before.

    F9: with ``cm_auction_paths`` (the Capacity Market builder's auction model, 2030/31
    T-4 supply table and calendar-2030 summary) and
    ``public_ownership.capacity_market_state_bids.enabled``, each market case carries
    its re-cleared 2030/31 T-4 (:mod:`capacity_market_state_bids`).

    30 Sep 2026: the register is read ``batch_cases`` market cases at a time
    (:class:`cfd_repricing.SavingsStream`) and each batch accounted on its own
    (:func:`public_account.build_public_account` with the whole run's settings,
    comparators and pay-as-bid ceiling), so memory is flat in the number of
    cases; the outputs are those of the whole-register account.
    """
    from scripts.public_ownership.register_io import market_bill_cases

    settings = account_settings(public_ownership_config)
    if settings["cm_sensitivity_enabled"] and capacity_market_line_is_nil(assumptions_path):
        # Task E2E (FES 2050): the bill's Capacity Market line is a declared nil (the 2050
        # auction clears at zero, Y11), and the bill caps a booking at its line, so nothing
        # can be booked against it: the state keeps that saving in its account.  Every 2030
        # bill has a positive CM line, so this never applies there.
        settings = {**settings, "cm_sensitivity_enabled": False}
        print("Capacity Market line is a declared nil: no route booking against it (the state keeps the saving)")
    repricing = repricing_settings(public_ownership_config.get("consumer_bill") or {})
    # The market-bill cases only: the account skips every acquiring case (register compaction).
    # Boundary variants are full cases in the register (``<case>@<boundary>``) and get their own account.
    stream = SavingsStream(register_path, repricing, cases=market_bill_cases(register_path), batch_cases=batch_cases)
    adjustments = table = energy = None
    if adjustments_path:
        from scripts.public_ownership.build_cfd_strike_adjustments import load_bill_strike_table

        if assumptions_path is None or model_price_year is None:
            raise ValueError("the strike adjustments need the bill's assumptions and model price year for its table")
        table = load_bill_strike_table(assumptions_path, int(model_price_year), strike_table_path)
        adjustments = pd.read_csv(adjustments_path)
        energy = settled_energy_by_key(stream.private_savings, table, None, paid_energy_from_adjustments(adjustments))
    ceiling = pay_as_bid_ceiling_gbp(stream.private_rows)
    cm_settings = cm_state_bids.settings_from_config(public_ownership_config)
    auction = None
    from scripts.public_ownership.model_year import is_vintaged_model_year

    years = pd.to_numeric(stream.private_rows.get("model_year"), errors="coerce").dropna().unique()         if "model_year" in stream.private_rows.columns else []
    if cm_settings["enabled"] and any(is_vintaged_model_year(int(y)) for y in years):
        # F9 re-clears the 2030/31 T-4: a vintaged (2050) register has its own auction (Y11), untouched.
        cm_settings = {**cm_settings, "enabled": False}
    if cm_settings["enabled"]:
        if not cm_auction_paths:
            raise ValueError("capacity_market_state_bids is enabled: pass the Capacity Market auction model, its "
                             "2030/31 supply table and the calendar-2030 summary (--cm-auction)")
        auction = cm_state_bids.load_auction(*cm_auction_paths, cm_settings)
    # F27 item 6: the cost-based contracts' parameters (lever_parameters.capacity_cost_based_existing).
    capacity_params = capacity_procurement.parameters(
        (public_ownership_config.get("lever_parameters") or {}).get(capacity_procurement.PARAMETER_KEY))
    comparator_rows: dict[str, pd.DataFrame] = {}

    def rows_for(case_id: str) -> pd.DataFrame:
        if case_id not in comparator_rows:
            # The battery top-up too: the account splits its change against private's (14 Oct 2026).
            rows = stream.context.rows_for(case_id)
            comparator_rows[case_id] = rows[[c for c in ("case_id", "asset_id", "annual_required_revenue_gbp",
                                                         "battery_revenue_top_up_gbp") if c in rows.columns]]
        return comparator_rows[case_id]

    accounts, summaries, repriced = [], [], set()
    for rows, savings in stream:
        if adjustments is not None:
            batch = set(savings["case_id"].astype(str))
            savings = reprice_as_the_bill_does(savings, adjustments, table, settled_energy=energy, cases=batch)
            repriced |= batch
        account, summary = build_public_account(rows, savings, settings, settings_by_case=stream.context.settings_by_case,
                                                rows_for=rows_for, pay_as_bid_ceiling=ceiling)
        if auction is not None:
            capacity_market_state_bids(summary, rows, stream.private_rows, auction, cm_settings)
        capacity_procurement_accounts(summary, rows, stream.private_rows, stream.context.settings_by_case,
                                      capacity_params)
        accounts.append(account)
        summaries.append(summary)
    if adjustments is not None:
        check_all_cases_repriced(adjustments, repriced)
    # F20 phase 2: the savings once per fingerprint (cfd_repricing.SavingsContext.case_savings).
    print(f"F20 phase 2: {stream.context.computed} savings computed; {stream.context.sample.report()}")
    account, summary = merge_accounts(accounts, summaries)
    report = {
        "settings": {**settings, cm_state_bids.CONFIG_KEY: cm_settings},
        "cases": summary,
        "notes": [
            "Rows re-priced under R1 are excluded from the account: an asset is re-priced or its saving goes to surplus, never both.",
            "The state's contracted-at-cost portfolio is renewables and hydro. Gas is paid by the Capacity Market or a strategic reserve and BECCS by a dispatchable CfD, outside this account; their financing savings are the Capacity Market and CCS-levy sensitivities.",
            "Storage surplus is never recycled (arbitrage-only revenue in the model); LDES is routed through the cap-and-floor line at a debt-like floor and batteries through the Capacity Market sensitivity.",
            "Surplus splits into the financing saving and the pre-existing rent the state acquires with the asset (a recipient change, a transfer). The financing saving is not all a resource saving: part of it is corporation tax the state does not pay (corporation_tax_forgone_gbp on the register, annual_financing_saving_tax_transfer_gbp in the financing summary), a transfer from the Exchequer.",
            "Items B2, B3, B5 and B12 (26 Sep 2026): the LDES floor the bill pays is in the stores' rent; the Capacity Market and CCS amounts booked on the bill are taken off the owning groups' surplus (booked_to_bill_gbp); the state overhead is deducted once from the state's share; the rent the state keeps on its re-priced (R1) rows after the strike cut is reported (r1_retained_rent_gbp) and retained with the R2 surplus.",
            "state_deficit_exchequer_funded_gbp is the sum of the negative balances left with the state (retained CfD and merchant surplus by rule, the Capacity Market and dispatchable-CfD routes after booking, the two storage groups) plus a routed shortfall the Exchequer carries; state_surplus_retained_gbp is the positive balances. Both feed the Exchequer ledger.",
        ],
    }
    report = stamp_price_level(report, price_level)
    for path in (account_output, summary_output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    account.to_csv(account_output, index=False)
    Path(summary_output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return account, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register")
    parser.add_argument("account_output")
    parser.add_argument("summary_output")
    parser.add_argument("--config", default="config")
    parser.add_argument("--strike-adjustments", default=None,
                        help="WP-B strike adjustments: the account re-prices R1 rows exactly as the bill does")
    parser.add_argument("--assumptions", default=None, help="consumer-bill assumptions JSON naming the strike table")
    parser.add_argument("--model-price-year", type=int, default=None)
    parser.add_argument("--strike-table", default=None)
    parser.add_argument("--batch-cases", type=int, default=DEFAULT_BATCH_CASES,
                        help="market cases read and accounted at a time (memory is flat in the number of cases)")
    parser.add_argument("--cm-auction", nargs=3, default=None, metavar=("MODEL_JSON", "SUPPLY_CSV", "CALENDAR_JSON"),
                        help="F9: the Capacity Market builder's auction model, 2030/31 T-4 supply and calendar summary")
    parser.add_argument("--price-level", default=None,
                        help="item F12: the price level the register was settled at (a price-level twin run group); "
                             "recorded in the summary when not the model's own")
    args = parser.parse_args()
    # F20 phase 0: opt-in stage timing (PYPSA_GB_STAGE_TIMING); without it nothing is wrapped.
    from scripts.public_ownership import public_account as _public_account, stage_timing

    stage_timing.instrument_register_io()
    stage_timing.instrument_cfd_repricing()
    stage_timing.instrument(globals(), ("write_public_account", "build_public_account", "reprice_as_the_bill_does",
                                        "capacity_market_state_bids", "merge_accounts", "pay_as_bid_ceiling_gbp",
                                        "settled_energy_by_key", "limit_at_cost_to_deliverable"),
                            prefix="public_account_stage")
    stage_timing.instrument(_public_account, ("ldes_floor_topups", "_floor_parts", "_floor_parts_original",
                                              "co_ownership_adjustments", "exchequer_position"), prefix="public_account")
    config = read_config(args.config) or {}
    write_public_account(args.register, config.get("public_ownership") or {}, args.account_output, args.summary_output,
                         adjustments_path=args.strike_adjustments, assumptions_path=args.assumptions,
                         model_price_year=args.model_price_year, strike_table_path=args.strike_table,
                         batch_cases=args.batch_cases, cm_auction_paths=args.cm_auction,
                         price_level=args.price_level)


if __name__ == "__main__":
    main()
