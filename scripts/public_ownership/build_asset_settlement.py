"""WP-D command line: annual market and support settlement per asset.

Reads the twelve monthly wholesale and balancing networks, wholesale prices and
per-asset redispatch summaries, settles CfD contracts through the bill's own
strike table and writes ``{run_group}_asset_settlement.csv`` in GBP real 2024.

A vintaged (2050) group passes ``--contract-book`` and ``--register-assets`` (task
SETTLE2050): the settlement is then one row per register row with each row's own CfD
and DPA support (``settlement.register_row_settlement``), and ``--support-summary``
writes the totals the 2050 bill's DPA line reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.market.cfd_strike_table import load_bm_strike_table
from scripts.public_ownership.build_cfd_strike_adjustments import load_bill_strike_table
from scripts.public_ownership.settlement import HYDROGEN_FUEL_PRICE_BASES, read_marginal_cost_breakdowns, settle_assets


def roc_value_model_prices(ro_scheme_summary_path, model_rebasing_factor: float) -> float:
    """The RO line's notional ROC value, carried back from report to model prices."""
    summary = json.loads(Path(ro_scheme_summary_path).read_text(encoding="utf-8"))
    value = summary["notional_roc_value"]
    return float(value["notional_roc_value_gbp_per_roc_report_prices"]) / float(model_rebasing_factor)


def dpa_support_model_prices(assumptions: dict) -> float | None:
    """The bill's DPA line (``sourced_externals.remaining_ccs``) as a system amount in the model's price year.

    ``cost_gbp`` is nominal money of its block; the line's own rebasing carries
    it to the report year and the bill's currency rebasing back to the model
    year, the conversions the projector applies (item B11).  ``None`` when the
    bill carries no such line.
    """
    block = (assumptions.get("sourced_externals") or {}).get("remaining_ccs") or {}
    cost = (block.get("derivation") or {}).get("cost_gbp")
    if cost is None:
        return None
    to_report = float((block.get("rebasing") or {}).get("factor", 1.0))
    return float(cost) * to_report / float(assumptions["currency_rebasing"]["factor"])


def dpa_prices_from_future_costs(path) -> dict | None:
    """Gas and carbon (GBP 2024) from the run's future-cost file, the reference CCGT's prices (task SETTLE2050)."""
    if not path:
        return None
    prices = json.loads(Path(path).read_text(encoding="utf-8"))["assumptions"]
    return {"gas": float(prices["fuel_prices"]["gas"]), "carbon": float(prices["carbon"]["uk_ets"])}


def write_asset_settlement(scenarios, wholesale, balancing, prices, redispatch, assumptions_path,
                           ro_scheme_summary, model_price_year, output, strike_table_path=None,
                           contract_value_bids=None, marginal_cost_breakdowns=None, hydrogen_fuel_price=None,
                           contract_book=None, register_assets=None, support_summary=None, dpa_future_costs=None):
    """Settle every asset over the case's months and write the settlement CSV.

    ``hydrogen_fuel_price`` (FES 2050 task Y14, vintaged groups only) prices the
    hydrogen turbines' fuel at their hydrogen bus (``settlement.HYDROGEN_FUEL_PRICE_BASES``);
    None, the default, books none, as every 2030 settlement always has.

    ``contract_book`` and ``register_assets`` (paths; task SETTLE2050, vintaged groups only)
    settle the register's own contracts per register row; ``support_summary`` is then
    written; ``dpa_future_costs`` is the run's future-cost JSON (the DPA reference CCGT).
    """
    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    table = load_bill_strike_table(assumptions_path, model_price_year, strike_table_path)
    if (contract_book is None) != (register_assets is None):
        raise ValueError("--contract-book and --register-assets go together")
    book = register = None
    if contract_book is not None:
        import pandas as pd

        from scripts.public_ownership.contract_book_2050 import read_contract_book

        book = read_contract_book(contract_book)
        register = pd.read_csv(register_assets, low_memory=False)
    roc_value = roc_value_model_prices(ro_scheme_summary, assumptions["currency_rebasing"]["factor"])
    settlement = settle_assets(
        list(scenarios), list(wholesale), list(balancing), list(prices), list(redispatch),
        strike_table=table,
        reference_price_basis=assumptions["model_linked_rules"]["cfd"].get("reference_price_basis"),
        roc_value_gbp_per_roc=roc_value,
        contract_value_bids_paths=list(contract_value_bids) if contract_value_bids else None,
        volume_caps={str(k): float(v["annual_mwh"])
                     for k, v in (assumptions["model_linked_rules"]["cfd"].get("volume_collars") or {}).items()},
        # The CfD bill line's rule, so support revenue settles as the bill does.
        negative_price_rule=str(assumptions["model_linked_rules"]["cfd"].get("negative_price_rule", "none")),
        # Item B1: resource fuel and carbon; item B11: the DPA line to its plant.
        marginal_cost_breakdown=read_marginal_cost_breakdowns(marginal_cost_breakdowns),
        dpa_support_gbp=dpa_support_model_prices(assumptions),
        hydrogen_fuel_price=hydrogen_fuel_price,
        contract_book=book, register_assets=register,
        bm_strike_table=load_bm_strike_table(assumptions, model_price_year),
        dpa_prices=dpa_prices_from_future_costs(dpa_future_costs),
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    settlement.to_csv(output, index=False)
    if book is not None and support_summary:
        from scripts.public_ownership.contract_book_2050 import write_support_summary

        write_support_summary(support_summary, settlement, settlement.attrs["dpa_units"],
                              treatment=settlement.attrs["compensation_treatment"], book_path=contract_book,
                              period_share=settlement.attrs["period_share"])
    elif support_summary:
        raise ValueError("--support-summary is written for a contract-book settlement only")
    return settlement


def write_central_dispatch_settlement(scenarios, central_scenarios, wholesale, central_wholesale, central_balancing,
                                      prices, fti_units, assumptions_path, ro_scheme_summary, model_price_year, output,
                                      strike_table_path=None, marginal_cost_breakdowns=None, transfers_output=None,
                                      realtime_at_cost: bool = True):
    """P1/P2 under central dispatch (20 Oct 2026): an institution twin's asset settlement on its parent's I3 year.

    ``wholesale`` and ``prices`` are the market months' Stage 1 networks and copperplate prices (the
    national price and the demand weighting, as the bill's); ``central_*`` the central-dispatch months;
    ``fti_units`` the year FTI settlement's per-unit transfers. ``transfers_output`` writes each asset's
    balancing settlement split into its transfers (``settlement.CENTRAL_TRANSFER_COLUMNS``).
    """
    import pandas as pd

    from scripts.public_ownership.settlement import settle_assets_central_dispatch

    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    cfd = assumptions["model_linked_rules"]["cfd"]
    settlement = settle_assets_central_dispatch(
        list(scenarios), list(central_scenarios), list(wholesale), list(central_wholesale), list(central_balancing),
        list(prices), pd.read_csv(fti_units),
        strike_table=load_bill_strike_table(assumptions_path, model_price_year, strike_table_path),
        reference_price_basis=cfd.get("reference_price_basis"),
        roc_value_gbp_per_roc=roc_value_model_prices(ro_scheme_summary, assumptions["currency_rebasing"]["factor"]),
        volume_caps={str(k): float(v["annual_mwh"]) for k, v in (cfd.get("volume_collars") or {}).items()},
        negative_price_rule=str(cfd.get("negative_price_rule", "none")),
        marginal_cost_breakdown=read_marginal_cost_breakdowns(marginal_cost_breakdowns),
        dpa_support_gbp=dpa_support_model_prices(assumptions), realtime_at_cost=realtime_at_cost)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    settlement.to_csv(output, index=False)
    if transfers_output:
        settlement.attrs["transfers"].to_csv(transfers_output, index=False)
    return settlement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--wholesale", nargs="+", required=True)
    parser.add_argument("--balancing", nargs="+", default=None)
    parser.add_argument("--prices", nargs="+", required=True)
    parser.add_argument("--redispatch", nargs="+", default=None)
    # P1/P2 under central dispatch (20 Oct 2026): an institution twin settles on its parent's I3 year.
    parser.add_argument("--central-scenarios", nargs="+", default=None,
                        help="the central-dispatch months (institution twin); --wholesale and --prices stay the market's")
    parser.add_argument("--central-wholesale", nargs="+", default=None)
    parser.add_argument("--central-balancing", nargs="+", default=None)
    parser.add_argument("--fti-units", default=None, help="the year FTI settlement's per-unit transfers")
    parser.add_argument("--transfers-output", default=None)
    parser.add_argument("--no-realtime-at-cost", action="store_true",
                        help="leave each generator's real-time correction unpaid (owners bear it)")
    parser.add_argument("--assumptions", required=True)
    parser.add_argument("--ro-scheme-summary", required=True)
    parser.add_argument("--model-price-year", type=int, required=True)
    parser.add_argument("--strike-table", default=None)
    parser.add_argument(
        "--contract-value-bids", nargs="+", default=None,
        help="Per-month contract-value bid audits from the balancing solve; they decide "
             "whether the curtailment compensation is already in the balancing settlement.",
    )
    parser.add_argument(
        "--marginal-cost-breakdown", nargs="+", default=None,
        help="Build-time {build_source}_marginal_costs_breakdown.csv files; fuel and carbon are read from them (item B1).",
    )
    parser.add_argument(
        "--hydrogen-fuel-price", default=None, choices=HYDROGEN_FUEL_PRICE_BASES,
        help="Task Y14 (vintaged groups only): price the hydrogen turbines' fuel at their hydrogen bus; "
             "omitted, no hydrogen fuel is booked (every 2030 group).",
    )
    parser.add_argument("--contract-book", default=None,
                        help="Task SETTLE2050 (vintaged groups): the 2050 contract book (contract_book_2050.py)")
    parser.add_argument("--register-assets", default=None,
                        help="Task SETTLE2050: the vintaged register's asset inputs the book was built on")
    parser.add_argument("--support-summary", default=None,
                        help="Task SETTLE2050: write the support totals by scheme (the 2050 bill's DPA line reads it)")
    parser.add_argument("--dpa-future-costs", default=None,
                        help="Task SETTLE2050: the run's future-cost JSON (gas and carbon for the DPA reference CCGT)")
    args = parser.parse_args()
    if args.central_scenarios:
        if not (args.central_wholesale and args.central_balancing and args.fti_units):
            parser.error("--central-scenarios needs --central-wholesale, --central-balancing and --fti-units")
        write_central_dispatch_settlement(
            args.scenarios, args.central_scenarios, args.wholesale, args.central_wholesale, args.central_balancing,
            args.prices, args.fti_units, args.assumptions, args.ro_scheme_summary, args.model_price_year, args.output,
            args.strike_table, marginal_cost_breakdowns=args.marginal_cost_breakdown,
            transfers_output=args.transfers_output, realtime_at_cost=not args.no_realtime_at_cost)
        return
    if not (args.balancing and args.redispatch):
        parser.error("--balancing and --redispatch are required (without --central-scenarios)")
    write_asset_settlement(
        args.scenarios, args.wholesale, args.balancing, args.prices, args.redispatch, args.assumptions,
        args.ro_scheme_summary, args.model_price_year, args.output, args.strike_table,
        contract_value_bids=args.contract_value_bids,
        marginal_cost_breakdowns=args.marginal_cost_breakdown,
        hydrogen_fuel_price=args.hydrogen_fuel_price,
        contract_book=args.contract_book, register_assets=args.register_assets,
        support_summary=args.support_summary, dpa_future_costs=args.dpa_future_costs,
    )


if __name__ == "__main__":
    main()
