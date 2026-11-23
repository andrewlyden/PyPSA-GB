"""WP-B: per-case CfD strike adjustments reconciling exactly to the finance register.

Reads the asset finance register (one row per asset, cohort and case), the
financing summary and the consumer-bill assumptions that name the strike table
the bill settles against, and writes

* ``{run_group}_cfd_strike_adjustments.csv``: ``strike_case = strike_private -
  delta`` for every table entry a CfD contract settles at, per case;
* ``{run_group}_financing_saving_routes.csv``: the saving by route (R1 bill,
  R2 public account, storage, Capacity Market, merchant, unresolved) with the
  identity against the financing summary.
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

from scripts.market.cfd_strike_table import load_strike_table, strike_overrides
from scripts.public_ownership.register_io import market_bill_cases
from scripts.public_ownership.cfd_repricing import (
    DEFAULT_BATCH_CASES, SavingsStream, StrikeAdjustments, assert_single_route, limit_at_cost_to_deliverable,
    repricing_settings, route_summary, settled_energy_by_key,
)
from config.config_loader import read_config


def load_bill_strike_table(assumptions_path: str | Path, model_price_year: int, strike_table_path=None):
    """The table the forecast bill settles against, indexed to the model price year."""
    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    path = strike_table_path or assumptions["model_linked_rules"]["cfd"]["strike_table"]
    return load_strike_table(path, assumptions["cfd_strike_indexation"], int(model_price_year),
                             overrides=strike_overrides(assumptions))


def paid_curtailment_by_key(forecast_summary_path) -> dict[str, float]:
    """Curtailed energy the bill pays at strike minus price, by strike entry (pre-AR4 rounds).

    The fallback denominator only: the summary carries every curtailed MWh, not
    the ones the negative pricing rule leaves paid, so this is the paid volume
    only under ``negative_price_rule: none``. :func:`bill_paid_energy_by_key`
    measures the paid volume itself and supersedes it.
    """
    if not forecast_summary_path:
        return {}
    summary = json.loads(Path(forecast_summary_path).read_text(encoding="utf-8"))
    rows = summary.get("cfd_economic_curtailment_by_strike") or []
    out: dict[str, float] = {}
    for row in rows:
        out[str(row["strike_key"])] = out.get(str(row["strike_key"]), 0.0) + float(row["curtailed_mwh"])
    return out


#: The strike step the paid-energy probe settles the year at. The CfD line is
#: linear in every strike (payment = (strike - reference) x paid energy; the
#: economic-curtailment uplift max(strike - price, 0) is linear too wherever the
#: strike is above the near-zero price it is paid at), so any step returns the
#: same energy; one pound keeps the probe on the linear side of that clip.
PAID_ENERGY_PROBE_GBP_PER_MWH = 1.0


def _payments_by_key(annual: dict) -> pd.Series:
    """The CfD line's payment per strike entry: settled dispatch plus economic curtailment."""
    rows = list((annual.get("cfd_strike_provenance") or {}).get("by_strike") or [])
    rows += list(annual.get("cfd_economic_curtailment_by_strike") or [])
    if not rows:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(rows)
    return frame.groupby(frame["strike_key"].astype(str))["payment_gbp"].sum().astype(float)


def bill_paid_energy_by_key(case: dict, assumptions: dict, networks, prices, summaries, physical_networks,
                            table, links=None) -> dict[str, float]:
    """Energy the bill's CfD line pays a difference on, per strike entry, measured on its own settlement.

    The year is settled twice through the projector's own settlement
    (``aggregate_inputs``, with exactly the arguments ``run_projection`` passes):
    once at the bill's strike table and once with every strike raised by
    :data:`PAID_ENERGY_PROBE_GBP_PER_MWH`. The payment change per strike entry,
    divided by the step, is the energy the line pays at that strike: balancing
    dispatch after the support start, in the hours the round's negative pricing
    rule leaves paid (``gb_contract_terms``), scaled by any volume collar, plus
    the pre-AR4 curtailed energy the economic-curtailment rule pays on the same
    terms. A strike cut of ``delta`` then moves the bill by exactly ``delta`` x
    this energy, which is what the route identity checks.

    ``links`` are the Stage 1 link schedules (``*_wholesale_links.csv``, one per
    month): the electrolysers' draw is part of the metered-supply demand the bill
    weights its price on (item D1). Without them the schedule written beside each
    wholesale network is used, exactly as ``run_projection`` does.

    A ``case`` at another price level (item F12: ``price_level: cp32`` with its
    ``price_level_mapping``; a price-level twin run group, 12 Oct 2026) is settled at
    that level, as the bill settles it: the negative pricing rule then reads the mapped
    hours. A case without ``price_level`` settles at the model's price, as before.
    """
    from scripts.consumer_bill.project_future_bill import (
        DEFAULT_PRICE_LEVEL, DEFAULT_ZERO_PRICE_THRESHOLD, aggregate_inputs, contract_book_path, price_level_of,
        sibling_link_schedules,
    )

    cfd_rule = assumptions["model_linked_rules"]["cfd"]
    scenarios = list(case["source_scenarios"])
    lengths = {len(scenarios), len(networks), len(prices), len(summaries), len(physical_networks)}
    if len(lengths) != 1:
        raise ValueError("scenarios, networks, prices, summaries and physical networks must align month by month")
    if links:
        links = list(links)
    elif all(str(n).endswith("_wholesale.nc") for n in networks):
        links = sibling_link_schedules(networks)
    else:
        links = None   # the projector refuses a month with electrolysers and no schedule
    if links is not None and len(links) != len(scenarios):
        raise ValueError("the Stage 1 link schedules must align with the scenarios month by month")
    level = price_level_of(case)
    # Item F12: passed only off the model's level, so a model-price call is exactly the one it was.
    price_level = ({} if level == DEFAULT_PRICE_LEVEL
                   else {"price_level": level, "price_level_mapping": case.get("price_level_mapping")})

    def settle(strikes):
        # Kept identical to run_projection's call: the paid energy is only the
        # bill's denominator if it is settled exactly as the bill settles.
        _, annual = aggregate_inputs(
            scenarios, list(networks), list(prices), list(summaries), int(case["modelled_year"]),
            list(physical_networks), strikes, None, cfd_rule["negative_price_rule"], None,
            reference_price_basis=cfd_rule.get("reference_price_basis", {}),
            economic_curtailment={
                "rule": cfd_rule.get("economic_curtailment_rule", "none"),
                "pre_ar4_rounds": cfd_rule.get("pre_ar4_rounds"),
                "zero_price_threshold_gbp_per_mwh": cfd_rule.get(
                    "zero_price_threshold_gbp_per_mwh", DEFAULT_ZERO_PRICE_THRESHOLD),
            },
            volume_caps={str(k): float(v["annual_mwh"]) for k, v in (cfd_rule.get("volume_collars") or {}).items()},
            wholesale_links=links, contract_book=contract_book_path(cfd_rule), **price_level,
        )
        return _payments_by_key(annual)

    step = PAID_ENERGY_PROBE_GBP_PER_MWH
    base = settle(table)
    raised = settle(table.adjusted({key: -step for key in table.strikes}, label="paid_energy_probe"))
    keys = base.index.union(raised.index)
    paid = (raised.reindex(keys, fill_value=0.0) - base.reindex(keys, fill_value=0.0)) / step
    if (paid < -1e-6).any():
        raise ValueError(f"raising every strike lowered the CfD line at {sorted(paid.index[paid < -1e-6])}; "
                         "the settlement is not linear in the strike")
    return {str(k): float(max(v, 0.0)) for k, v in paid.items()}


def price_level_settings(config: dict, case_id: str) -> dict:
    """``{price_level, price_level_mapping}`` of a ``consumer_bill.forecasts`` case (item F12; {} at the model's level)."""
    from scripts.consumer_bill.project_future_bill import DEFAULT_PRICE_LEVEL, price_level_of

    cases = (((config.get("consumer_bill") or {}).get("forecasts") or {}).get("cases")) or {}
    if case_id not in cases:
        raise ValueError(f"--forecast-case {case_id!r} is not a consumer_bill.forecasts case; configured: {sorted(cases)}")
    case = cases[case_id]
    level = price_level_of(case)
    if level == DEFAULT_PRICE_LEVEL:
        return {}
    return {"price_level": level, "price_level_mapping": case.get("price_level_mapping")}


def rate_path_case_offsets(settings_by_case, offsets_path, *, private_rows=None, table=None,
                           energy: pd.Series | None = None) -> dict | None:
    """FES 2050 decision 54: ``{case_id: {strike_key: offset}}`` for every case on a private rate path with offsets.

    The register cuts each contract by ``offset x the row's cfd_paid_mwh``; so that the bill's CfD
    line moves by exactly the same money, an entry's offset is rescaled to the energy the bill pays
    at it (``energy``): ``offset x sum of the private rows' cfd_paid_mwh / energy``. Without the
    private rows or the energy it is the offset itself. None without an offsets file (every 2030
    call), so the adjustments keep their columns.
    """
    if not offsets_path:
        return None
    from scripts.public_ownership.cfd_repricing import _priced_rows
    from scripts.public_ownership.contract_book_2050 import load_rate_path_offsets
    from scripts.public_ownership.levers import LEVER_PRIVATE_RATE_PATH, lever_default

    offsets = load_rate_path_offsets(offsets_path)
    scale: dict[str, float] = {}
    if private_rows is not None and table is not None and energy is not None and "cfd_paid_mwh" in private_rows.columns:
        rows = private_rows.assign(case_id="private",
                                   support_type=private_rows["support_type"].fillna("").astype(str),
                                   cfd_round=private_rows["cfd_round"].fillna("").astype(str))
        priced = _priced_rows(rows, table)
        paid = priced.groupby("strike_key")["cfd_paid_mwh"].apply(
            lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum()))
        for key, mwh in paid.items():
            e = float(energy.get(key, 0.0))
            scale[str(key)] = mwh / e if e > 0 else 0.0
    out = {}
    for case_id, settings in settings_by_case.items():
        path = str(settings.get(LEVER_PRIVATE_RATE_PATH, lever_default(LEVER_PRIVATE_RATE_PATH)))
        if path in offsets:
            out[case_id] = {f"{r}_{t}": float(v) * scale.get(f"{r}_{t}", 1.0) for (r, t), v in offsets[path].items()}
    return out


def write_strike_adjustments(
    register_path, financing_summary_path, assumptions_path, consumer_bill_config, *,
    model_price_year, adjustments_output, routes_output, strike_table_path=None, forecast_summary_path=None,
    settlement_inputs: dict | None = None, batch_cases: int = DEFAULT_BATCH_CASES,
    rate_path_offsets_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the strike adjustments and the routes.

    ``settlement_inputs`` (``case``, ``networks``, ``prices``, ``summaries``,
    ``physical_networks``: the forecast case's month files) lets the strike cut
    divide by the energy the bill actually pays (:func:`bill_paid_energy_by_key`).
    Without them the register's dispatch stands in, which the bill pays in full
    only under ``negative_price_rule: none``; any other rule is refused.

    30 Sep 2026: the register is read ``batch_cases`` market cases at a time
    (:class:`cfd_repricing.SavingsStream`) and each batch's adjustments and
    routes are kept, so memory is flat in the number of cases; the outputs are
    those of the whole-register calculation.

    F20 phase 2 (27 Oct 2026): the savings are computed once per savings fingerprint and
    only the first case of each is read (:meth:`cfd_repricing.SavingsStream.savings_batches`);
    every case still gets its own adjustments and routes.
    """
    settings = repricing_settings(consumer_bill_config)
    summary = pd.read_csv(financing_summary_path)
    table = load_bill_strike_table(assumptions_path, model_price_year, strike_table_path)
    # The market-bill cases only: every acquiring case is skipped by the re-pricing (register compaction).
    stream = SavingsStream(register_path, settings, cases=market_bill_cases(register_path), batch_cases=batch_cases)
    assert_single_route(stream.private_savings)
    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    rule = str(assumptions["model_linked_rules"]["cfd"].get("negative_price_rule", "none"))
    paid_curtailment = paid_curtailment_by_key(forecast_summary_path)
    paid_energy = None
    if settlement_inputs:
        paid_energy = bill_paid_energy_by_key(
            settlement_inputs["case"], assumptions, settlement_inputs["networks"], settlement_inputs["prices"],
            settlement_inputs["summaries"], settlement_inputs["physical_networks"], table,
            links=settlement_inputs.get("links"))
    elif rule != "none":
        raise ValueError(
            f"the bill applies negative_price_rule {rule!r}, so it leaves some CfD hours unpaid and the register's "
            "dispatch is not the strike cut's denominator; pass the forecast case's month files "
            "(--scenarios/--networks/--prices/--summaries/--physical-networks) so the paid energy is measured")
    energy = settled_energy_by_key(stream.private_savings, table, paid_curtailment, paid_energy)
    builder = StrikeAdjustments(stream.private_savings, table, settings,
                                paid_curtailment_mwh_by_key=paid_curtailment, paid_energy_mwh_by_key=paid_energy,
                                case_offsets=rate_path_case_offsets(stream.context.settings_by_case, rate_path_offsets_path,
                                                                    private_rows=stream.private_rows, table=table,
                                                                    energy=energy))
    route_parts = []
    for savings in stream.savings_batches():
        assert_single_route(savings)
        # A contract whose rent cannot pass through its strike keeps the ordinary
        # financing-saving treatment; this runs before both the adjustments and the
        # routes so the strike table and the route amounts describe the same thing.
        savings, limited = limit_at_cost_to_deliverable(savings, table, paid_curtailment, paid_energy,
                                                        settled_energy=energy)
        for row in limited.itertuples(index=False):
            print(
                f"at-cost re-pricing not deliverable for {row.strike_key} in {row.case_id}: "
                f"GBP {row.at_cost_gbp_per_mwh_2024:,.2f}/MWh of rent against a "
                f"GBP {row.strike_gbp_per_mwh_2024:,.2f}/MWh strike over {row.settled_mwh:,.0f} MWh; "
                f"kept on the financing saving (GBP {row.financing_saving_gbp_2024:,.0f} instead of "
                f"GBP {row.at_cost_saving_gbp_2024:,.0f})"
            )
        builder.add(savings)
        route_parts.append(route_summary(savings, summary))
    print(f"F20 phase 2: {stream.context.computed} savings computed, {stream.rows_read} cases read; "
          f"{stream.context.sample.report()}")
    adjustments = builder.frame()
    routes = pd.concat(route_parts, ignore_index=True)
    if not routes["identity_within_tolerance"].all():
        bad = routes.loc[~routes["identity_within_tolerance"], ["case_id", "identity_residual_gbp_2024"]]
        raise ValueError(f"route savings do not reconcile to the financing summary:\n{bad.drop_duplicates()}")
    for path in (adjustments_output, routes_output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    adjustments.to_csv(adjustments_output, index=False)
    routes.to_csv(routes_output, index=False)
    return adjustments, routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register")
    parser.add_argument("financing_summary")
    parser.add_argument("assumptions", help="consumer-bill forecast assumptions JSON naming the strike table")
    parser.add_argument("adjustments_output")
    parser.add_argument("routes_output")
    parser.add_argument("--config", default="config")
    parser.add_argument("--model-price-year", type=int, required=True)
    parser.add_argument("--strike-table", default=None)
    parser.add_argument("--forecast-summary", default=None,
                        help="forecast bill summary JSON; its economic-curtailment diagnostics extend the fallback denominator")
    # The forecast case's month files: the strike cut divides by the energy the
    # bill's CfD line actually pays, measured on the same settlement.
    parser.add_argument("--scenarios", nargs="*", default=None, help="the forecast case's source scenarios")
    parser.add_argument("--modelled-year", type=int, default=None)
    parser.add_argument("--networks", nargs="*", default=None, help="wholesale networks, one per scenario")
    parser.add_argument("--prices", nargs="*", default=None, help="wholesale price CSVs, one per scenario")
    parser.add_argument("--summaries", nargs="*", default=None, help="market summary JSONs, one per scenario")
    parser.add_argument("--physical-networks", nargs="*", default=None, help="balancing networks, one per scenario")
    parser.add_argument("--batch-cases", type=int, default=DEFAULT_BATCH_CASES,
                        help="market cases read and re-priced at a time (memory is flat in the number of cases)")
    parser.add_argument("--rate-path-offsets", default=None,
                        help="FES 2050 decision 54: the study's rate-path strike offsets (cases on that path settle there)")
    parser.add_argument("--links", nargs="*", default=None,
                        help="Stage 1 link schedules (*_wholesale_links.csv), one per scenario (electrolyser draw, "
                             "item D1); default the schedule beside each wholesale network")
    parser.add_argument("--forecast-case", default=None,
                        help="item F12: the consumer_bill.forecasts case (in --config) whose price level the paid "
                             "energy is settled at (a price-level twin run group); default the model's price")
    args = parser.parse_args()
    # F20 phase 0: opt-in stage timing (PYPSA_GB_STAGE_TIMING); without it nothing is wrapped.
    from scripts.public_ownership import stage_timing

    stage_timing.instrument_register_io()
    stage_timing.instrument_cfd_repricing()
    stage_timing.instrument(globals(), ("write_strike_adjustments", "bill_paid_energy_by_key", "load_bill_strike_table",
                                        "limit_at_cost_to_deliverable", "assert_single_route", "rate_path_case_offsets"),
                            prefix="strike_adjustments")
    config = read_config(args.config) or {}
    consumer_bill = (config.get("public_ownership") or {}).get("consumer_bill") or {}
    settlement_inputs = None
    if args.scenarios:
        if args.modelled_year is None:
            parser.error("--scenarios needs --modelled-year")
        case = {"source_scenarios": args.scenarios, "modelled_year": args.modelled_year}
        if args.forecast_case:
            case.update(price_level_settings(config, args.forecast_case))
        settlement_inputs = {
            "case": case,
            "networks": args.networks or [], "prices": args.prices or [], "summaries": args.summaries or [],
            "physical_networks": args.physical_networks or [], "links": args.links or None,
        }
    write_strike_adjustments(
        args.register, args.financing_summary, args.assumptions, consumer_bill,
        model_price_year=args.model_price_year, adjustments_output=args.adjustments_output,
        routes_output=args.routes_output, strike_table_path=args.strike_table,
        forecast_summary_path=args.forecast_summary, settlement_inputs=settlement_inputs,
        batch_cases=args.batch_cases, rate_path_offsets_path=args.rate_path_offsets,
    )


if __name__ == "__main__":
    main()
