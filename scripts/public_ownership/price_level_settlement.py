"""The asset settlement at CP30's 2030 price level, CP.32 (F19 P3 decision 2).

Maps the market year's Stage 1 GB price onto CP.32's duration curve
(:mod:`scripts.consumer_bill.price_level_mapping`), writes the mapped month price files,
and re-settles every asset on them through the WP-D settlement itself
(:func:`settlement.settle_assets`): the Stage 1 positions at the mapped price, and the
CfD difference payments on balancing dispatch against the mapped reference (the
baseload carriers' seasonal reference recomputed from it).  A higher reference lowers
the top-ups.  Everything else is held as solved: the balancing settlement (and so the
CfD curtailment compensation inside it under contract-value bids), fuel and carbon,
ROC income (buy-out value, no price in it) and the DPA.  Volumes never move.

Output ``{run_group}_price_level_settlement.csv`` has the central settlement's columns
and rows with ``wholesale_revenue_gbp``, ``cfd_support_gbp``, ``market_revenue_gbp`` and
``support_revenue_gbp`` on the mapped price, plus ``wholesale_revenue_model_price_gbp``
and ``cfd_support_model_price_gbp`` carried from the central settlement.  Its only reader
is the legacy valuation's post-model-year price path (lever ``legacy_price_path``); it
never replaces the central settlement and no bill reads it (the F12 bill sensitivity was
dropped, user 6 Oct 2026).  The summary JSON states both price levels and each carrier's
capture price on each: a static bracket, volumes held as solved.

A price-level twin run group (item F12 at the ownership layer, 12 Oct 2026;
``reference_run_groups.<twin>.price_level_of``) takes its asset settlement from here
with ``--price-level-case CASE``: the year mapped exactly as that forecast case's bill
maps it (``project_future_bill.map_year_to_price_level``: the case's mapping block and
the zero band), the twelve mapped month price files written to ``--mapped-prices``, and
the output is the central settlement's own columns (:data:`settlement.SETTLEMENT_COLUMNS`)
with the same repriced and held columns as above (:func:`build_at_bill_price_level`), and
one more (item F24.2, user decision 24 Oct 2026): each CfD unit's contract-value bid inside
``balancing_settlement_gbp`` moves to the mapped reference by exactly the change the twin's
bill puts on its constraint line (:func:`cfd_bid_changes`), so the bill-to-register
identities close. The legacy valuation's price path (without the flag) holds it as before.
Without the flag nothing changes.

Usage::

    python scripts/public_ownership/price_level_settlement.py <central_settlement.csv> <out.csv> <out_summary.json>
        --prices-dir DIR --scenarios ... --wholesale ... --balancing ... --prices ... --redispatch ...
        --assumptions A --ro-scheme-summary R --model-price-year 2024 [--strike-table T]
        [--marginal-cost-breakdown ...] [--config config]
        [--price-level-case CASE --mapped-prices FILE ...]   (a price-level twin; replaces --prices-dir)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.consumer_bill import price_level_mapping as plm  # noqa: E402
from config.config_loader import read_config

#: Columns re-settled on the mapped price; every other settlement column is held.
REPRICED_COLUMNS = ("wholesale_revenue_gbp", "cfd_support_gbp")
MODEL_PRICE_COLUMNS = {c: c.replace("_gbp", "_model_price_gbp") for c in REPRICED_COLUMNS}


def combine(central: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    """The central settlement with its price-dependent columns taken from ``mapped``.

    Rows are matched on (component type, asset id); every central row must have a
    mapped twin with the same Stage 1 energy (the volumes are held, so a mismatch
    means the two settlements read different months).
    """
    key = ["component_type", "asset_id"]
    left = central.set_index(key)
    right = mapped.set_index(key).reindex(left.index)
    if right[list(REPRICED_COLUMNS)].isna().any().any():
        missing = right.index[right[list(REPRICED_COLUMNS)].isna().any(axis=1)][:5].tolist()
        raise ValueError(f"the mapped settlement lacks central rows, e.g. {missing}")
    energy = pd.to_numeric(left["wholesale_mwh"], errors="coerce").fillna(0.0)
    gap = (pd.to_numeric(right["wholesale_mwh"], errors="coerce").fillna(0.0) - energy).abs()
    if (gap > 1e-6 * energy.abs().clip(lower=1.0)).any():
        raise ValueError("the mapped and central settlements disagree on Stage 1 energy: different months?")
    out = left.copy()
    for column in REPRICED_COLUMNS:
        out[MODEL_PRICE_COLUMNS[column]] = pd.to_numeric(left[column], errors="coerce").fillna(0.0)
        out[column] = pd.to_numeric(right[column], errors="coerce").fillna(0.0)
    delta_cfd = out["cfd_support_gbp"] - out[MODEL_PRICE_COLUMNS["cfd_support_gbp"]]
    out["market_revenue_gbp"] = out["wholesale_revenue_gbp"] + pd.to_numeric(left["balancing_settlement_gbp"],
                                                                              errors="coerce").fillna(0.0)
    out["support_revenue_gbp"] = pd.to_numeric(left["support_revenue_gbp"], errors="coerce").fillna(0.0) + delta_cfd
    return out.reset_index()


def carrier_capture(settlement: pd.DataFrame) -> pd.DataFrame:
    """Capture price by carrier on both levels (wholesale revenue over Stage 1 energy, generators only)."""
    rows = settlement.loc[settlement["component_type"].eq("Generator")]
    grouped = rows.groupby("carrier")[["wholesale_mwh", "wholesale_revenue_gbp",
                                       MODEL_PRICE_COLUMNS["wholesale_revenue_gbp"]]].sum()
    energy = grouped["wholesale_mwh"].where(grouped["wholesale_mwh"] > 0)
    return pd.DataFrame({
        "wholesale_twh": grouped["wholesale_mwh"] / 1e6,
        "capture_model_gbp_per_mwh": grouped[MODEL_PRICE_COLUMNS["wholesale_revenue_gbp"]] / energy,
        "capture_mapped_gbp_per_mwh": grouped["wholesale_revenue_gbp"] / energy,
    })


def build(central_path, out_path, summary_path, prices_dir, *, scenarios, wholesale, balancing, prices, redispatch,
          assumptions, ro_scheme_summary, model_price_year, strike_table=None, marginal_cost_breakdowns=None,
          settings: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Map the year, re-settle on the mapped price files and write the combined settlement and summary."""
    from scripts.public_ownership.build_asset_settlement import write_asset_settlement

    settings = plm.settings_from_config(settings)
    source = Path(settings["source_file"])
    curve = plm.load_duration_curve(source if source.exists() else PROJECT_ROOT / source, settings["pathway"],
                                    float(settings["rebasing_to_model_price_year"]))
    hourly = plm.read_market_year(wholesale, prices)
    mapped = plm.map_market_year(hourly, curve)
    mapped_files = plm.write_mapped_price_files(hourly, mapped, prices_dir)
    by_name = {p.name: p for p in mapped_files}
    mapped_prices = [by_name[Path(p).name] for p in prices]
    with tempfile.TemporaryDirectory() as tmp:
        # No contract-value audits: the compensation recomputed on the mapped price is not
        # the bids the balancing stage accepted, and it is held from the central settlement.
        resettled = write_asset_settlement(
            scenarios, wholesale, balancing, mapped_prices, redispatch, assumptions, ro_scheme_summary,
            model_price_year, Path(tmp) / "mapped.csv", strike_table, contract_value_bids=None,
            marginal_cost_breakdowns=marginal_cost_breakdowns)
    central = pd.read_csv(central_path, low_memory=False)
    combined = combine(central, resettled)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    summary = plm.mapping_summary(hourly, mapped, curve, settings)
    capture = carrier_capture(combined)
    summary.update({
        "central_settlement": str(central_path),
        "mapped_price_files": [str(p) for p in mapped_prices],
        "held_as_solved": ["balancing_settlement_gbp (and the CfD curtailment compensation inside it)",
                           "fuel_carbon_gbp", "roc_support_gbp", "dpa_support_gbp", "every volume"],
        "totals_gbp": {
            "wholesale_revenue_model_price": float(combined[MODEL_PRICE_COLUMNS["wholesale_revenue_gbp"]].sum()),
            "wholesale_revenue_mapped": float(combined["wholesale_revenue_gbp"].sum()),
            "cfd_support_model_price": float(combined[MODEL_PRICE_COLUMNS["cfd_support_gbp"]].sum()),
            "cfd_support_mapped": float(combined["cfd_support_gbp"].sum()),
        },
        "capture_by_carrier": {c: {k: (None if not np.isfinite(v) else float(v)) for k, v in row.items()}
                               for c, row in capture.iterrows()},
    })
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return combined, summary


def audit_beside(redispatch_path) -> Path:
    """The month's contract-value bid audit, written beside its redispatch summary by the balancing solve."""
    name = Path(redispatch_path).name
    if not name.endswith("_redispatch_summary.csv"):
        raise ValueError(f"{redispatch_path} is not a redispatch summary; pass the bid audits explicitly")
    return Path(redispatch_path).with_name(name[: -len("_redispatch_summary.csv")] + "_contract_value_bids.csv")


def cfd_bid_changes(wholesale, balancing, prices, redispatch, contract_value_bids, mapped: pd.Series, *,
                    assumptions, strike_table, model_price_year) -> pd.DataFrame:
    """Item F24.2: each CfD unit's bid change at the price level, summed over the months (GBP).

    Month by month through ``extract_future_balancing_costs.cfd_bid_change_at_price_level``,
    with the strike table the extraction values compensation on
    (``compensation_strike_source``), the Stage 1 price files as the model's reference and
    ``mapped`` as the level's, so the total is the twin bill's own. Indexed by generator:
    ``change_gbp`` (the money the bid moves) and ``gross_change_gbp`` (on the whole turn-down).
    """
    from scripts.consumer_bill.extract_future_balancing_costs import (
        cfd_bid_change_at_price_level, compensation_strike_source,
    )
    from scripts.market.cfd_strike_table import load_strike_table

    lengths = {len(wholesale), len(balancing), len(prices), len(redispatch), len(contract_value_bids)}
    if len(lengths) != 1:
        raise ValueError("the months' networks, prices, redispatch summaries and bid audits must align")
    document = json.loads(Path(assumptions).read_text(encoding="utf-8"))
    if strike_table is None:
        strike_table = document["model_linked_rules"]["cfd"]["strike_table"]
    path, overrides = compensation_strike_source(document, strike_table)
    table = load_strike_table(path, document["cfd_strike_indexation"], int(model_price_year), overrides=overrides)
    frames = []
    for w, b, price, r, audit in zip(wholesale, balancing, prices, redispatch, contract_value_bids):
        model = pd.read_csv(price, index_col=0, parse_dates=True)["wholesale_price"]
        frames.append(cfd_bid_change_at_price_level(w, b, model, mapped, table, audit, r))
    rows = (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["generator", "change_gbp", "gross_change_gbp"]))
    return rows.groupby(rows["generator"].astype(str))[["change_gbp", "gross_change_gbp"]].sum()


def apply_cfd_bid_changes(combined: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """The settlement with each CfD unit's bid change on its balancing settlement (item F24.2).

    ``balancing_settlement_gbp`` takes ``change_gbp`` (the private share's contract-value bid;
    the state-owned share bids its public price and does not move) and
    ``cfd_curtailment_compensation_gbp`` ``gross_change_gbp``; ``market_revenue_gbp`` follows.
    The bids priced the compensation (contract value), so support revenue never held it and
    does not move. Every unit must be a generator row of the settlement.
    """
    out = combined.copy()
    if changes.empty:
        return out
    rows = out["component_type"].eq("Generator")
    position = pd.Series(out.index[rows], index=out.loc[rows, "asset_id"].astype(str))
    if position.index.has_duplicates:
        raise ValueError("the settlement has more than one generator row per asset")
    if missing := sorted(set(changes.index) - set(position.index)):
        raise ValueError(f"CfD units with a bid change have no settlement row, e.g. {missing[:5]}")
    at = position.reindex(changes.index).to_numpy()
    for column, source in (("balancing_settlement_gbp", "change_gbp"),
                           ("cfd_curtailment_compensation_gbp", "gross_change_gbp")):
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        values.loc[at] += changes[source].to_numpy()
        out[column] = values
    out["market_revenue_gbp"] = (pd.to_numeric(out["wholesale_revenue_gbp"], errors="coerce").fillna(0.0)
                                 + out["balancing_settlement_gbp"])
    return out


def build_at_bill_price_level(central_path, out_path, summary_path, mapped_price_paths, *, case: dict, scenarios,
                              wholesale, balancing, prices, redispatch, assumptions, ro_scheme_summary,
                              model_price_year, strike_table=None, marginal_cost_breakdowns=None,
                              contract_value_bids=None) -> tuple[pd.DataFrame, dict]:
    """A price-level twin's asset settlement: the central one re-settled at ``case``'s price level (item F12).

    The year is mapped as the forecast case's bill maps it
    (:func:`project_future_bill.map_year_to_price_level`), so the register, the strike cut's
    paid energy and the bill all settle on one hourly series. The Stage 1 positions and the
    CfD difference payments move to the mapped price. Item F24.2 (user decision, 24 Oct 2026):
    so do the CfD contract-value bids inside the balancing settlement, each unit by exactly
    the change the twin's bill puts on its constraint line
    (:func:`apply_cfd_bid_changes`; ``contract_value_bids`` are the months' bid audits, by
    default beside each redispatch summary). Every other part of the balancing settlement,
    fuel and carbon, ROCs, the DPA and every volume are the central settlement's
    (:func:`combine`). Writes the settlement with :data:`settlement.SETTLEMENT_COLUMNS` only,
    rows in the central order, so the register reads it exactly as it reads the central one.
    """
    from scripts.consumer_bill.project_future_bill import DEFAULT_PRICE_LEVEL, map_year_to_price_level, price_level_of
    from scripts.public_ownership.build_asset_settlement import write_asset_settlement
    from scripts.public_ownership.settlement import SETTLEMENT_COLUMNS

    level = price_level_of(case)
    if level == DEFAULT_PRICE_LEVEL:
        raise ValueError("a price-level settlement needs a forecast case off the model's own price level")
    if len(mapped_price_paths) != len(prices) or len(set(map(str, prices))) != len(prices):
        raise ValueError("one mapped price file per month, and one distinct Stage 1 price file per month")
    hourly = plm.read_market_year(wholesale, prices)
    mapped, record = map_year_to_price_level(hourly, case.get("price_level_mapping"))
    # The targets are given in the months' order; the writer walks the sources in time order.
    target_of = {str(p): Path(t) for p, t in zip(prices, mapped_price_paths)}
    sources = list(dict.fromkeys(hourly["price_file"]))
    plm.write_mapped_price_files(hourly, mapped, targets=[target_of[s] for s in sources])
    mapped_prices = month_prices = [target_of[str(p)] for p in prices]
    with tempfile.TemporaryDirectory() as tmp:
        # No contract-value audits: the compensation is the balancing stage's, held from the central settlement.
        resettled = write_asset_settlement(
            scenarios, wholesale, balancing, month_prices, redispatch, assumptions, ro_scheme_summary,
            model_price_year, Path(tmp) / "mapped.csv", strike_table, contract_value_bids=None,
            marginal_cost_breakdowns=marginal_cost_breakdowns)
    central = pd.read_csv(central_path, low_memory=False)
    combined = combine(central, resettled)
    # Item F24.2: the CfD contract-value bids on the mapped reference, as the twin's bill moves them.
    audits = (list(contract_value_bids) if contract_value_bids is not None
              else [audit_beside(path) for path in redispatch])
    changes = cfd_bid_changes(wholesale, balancing, prices, redispatch, audits, mapped, assumptions=assumptions,
                              strike_table=strike_table, model_price_year=model_price_year)
    combined = apply_cfd_bid_changes(combined, changes)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    combined[SETTLEMENT_COLUMNS].to_csv(out_path, index=False)
    summary = dict(record)
    summary.update({
        "price_level": level,
        "central_settlement": str(central_path),
        "mapped_price_files": [str(p) for p in mapped_prices],
        "repriced_columns": list(REPRICED_COLUMNS) + [
            "balancing_settlement_gbp (the CfD contract-value bids only, item F24.2)",
            "cfd_curtailment_compensation_gbp (the same units)", "market_revenue_gbp", "support_revenue_gbp"],
        "held_as_solved": ["balancing_settlement_gbp apart from the CfD contract-value bids (the Stage 1-indexed "
                           "storage bids and offers among it, user decision 24 Oct 2026)",
                           "fuel_carbon_gbp", "roc_support_gbp", "dpa_support_gbp", "every volume"],
        "totals_gbp": {
            "wholesale_revenue_model_price": float(combined[MODEL_PRICE_COLUMNS["wholesale_revenue_gbp"]].sum()),
            "wholesale_revenue_at_price_level": float(combined["wholesale_revenue_gbp"].sum()),
            "cfd_support_model_price": float(combined[MODEL_PRICE_COLUMNS["cfd_support_gbp"]].sum()),
            "cfd_support_at_price_level": float(combined["cfd_support_gbp"].sum()),
            # Item F24.2: equals the twin bill's extraction's cfd_bid_price_level_change_gbp.
            "cfd_bid_change_at_price_level": float(changes["change_gbp"].sum()),
        },
        "cfd_bid_change_units": int(len(changes)),
    })
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return combined[SETTLEMENT_COLUMNS], summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("central_settlement")
    parser.add_argument("output")
    parser.add_argument("summary")
    parser.add_argument("--prices-dir", default=None, help="where the mapped month price files are written")
    for name in ("scenarios", "wholesale", "balancing", "prices", "redispatch"):
        parser.add_argument(f"--{name}", nargs="+", required=True)
    parser.add_argument("--assumptions", required=True)
    parser.add_argument("--ro-scheme-summary", required=True)
    parser.add_argument("--model-price-year", type=int, required=True)
    parser.add_argument("--strike-table", default=None)
    parser.add_argument("--marginal-cost-breakdown", nargs="+", default=None)
    parser.add_argument("--config", default="config")
    parser.add_argument("--price-level-case", default=None,
                        help="item F12: settle at this consumer_bill.forecasts case's price level, mapped as its bill "
                             "maps (a price-level twin run group's asset settlement)")
    parser.add_argument("--mapped-prices", nargs="+", default=None,
                        help="with --price-level-case: where each month's mapped price file is written, in order")
    parser.add_argument("--contract-value-bids", nargs="+", default=None,
                        help="with --price-level-case: the months' bid audits (item F24.2; default: beside each "
                             "redispatch summary)")
    args = parser.parse_args(argv)
    config = read_config(args.config) or {}
    if args.price_level_case:
        cases = (((config.get("consumer_bill") or {}).get("forecasts") or {}).get("cases")) or {}
        if args.price_level_case not in cases or not args.mapped_prices:
            parser.error("--price-level-case needs a consumer_bill.forecasts case in --config and --mapped-prices")
        _, summary = build_at_bill_price_level(
            args.central_settlement, args.output, args.summary, args.mapped_prices,
            case=cases[args.price_level_case], scenarios=args.scenarios, wholesale=args.wholesale,
            balancing=args.balancing, prices=args.prices, redispatch=args.redispatch, assumptions=args.assumptions,
            ro_scheme_summary=args.ro_scheme_summary, model_price_year=args.model_price_year,
            strike_table=args.strike_table, marginal_cost_breakdowns=args.marginal_cost_breakdown,
            contract_value_bids=args.contract_value_bids)
        print(json.dumps({"price_level": summary["price_level"],
                          "model_gbp_per_mwh": round(summary["model"]["time_weighted_gbp_per_mwh"], 2),
                          "mapped_gbp_per_mwh": round(summary["mapped"]["time_weighted_gbp_per_mwh"], 2)}))
        return
    if not args.prices_dir:
        parser.error("--prices-dir is required")
    block = (config.get("consumer_bill") or {}).get("price_level_mapping")
    _, summary = build(
        args.central_settlement, args.output, args.summary, args.prices_dir, scenarios=args.scenarios,
        wholesale=args.wholesale, balancing=args.balancing, prices=args.prices, redispatch=args.redispatch,
        assumptions=args.assumptions, ro_scheme_summary=args.ro_scheme_summary,
        model_price_year=args.model_price_year, strike_table=args.strike_table,
        marginal_cost_breakdowns=args.marginal_cost_breakdown, settings=block)
    print(json.dumps({"model_gbp_per_mwh": round(summary["model"]["time_weighted_gbp_per_mwh"], 2),
                      "mapped_gbp_per_mwh": round(summary["mapped"]["time_weighted_gbp_per_mwh"], 2)}))


if __name__ == "__main__":
    main()
