#!/usr/bin/env python3
"""Validate a CP30 calendar year's Stage 1 (wholesale) dispatch against CP30 (16 Oct 2026).

``validate_CP30_dispatch`` judges ES1 only once all twelve balancing months exist, and
its energy gaps only warn. This check runs as soon as the twelve Stage 1 months of a
family are solved, on Stage 1 alone, so a calibration run can stop before any
balancing solve when Stage 1 is off CP30:

* **Energy** (CP30 ES1, through ``scripts/CP30/energy_validation.py`` unchanged:
  ``cp30_energy_targets``, ``dataset_energy_tables``, ``build_energy_validation``):
  generation by technology, curtailment, demand, interconnector imports and exports,
  electrolysis and hydrogen turbines (the Stage 1 files keep link flows since the
  rebuild batch, F14), storage discharge. NZT's ``CCS Gas`` is counted in ES1's Gas
  line (the declared carve-out, ``fleet.spatial_proxy.committed_overlay_carve_out``).
* **Price** (CP30's CP.32 price duration curve, ``data/CP30/cp32_price_duration_curve_2030.csv``,
  the pathway's column): the mean (the curve's trapezoid mean,
  ``price_level_mapping.curve_mean``), the price exceeded in 1, 10, 30, 50, 70 and 90%
  of hours, and the share of hours at or below GBP 0.5/MWh, from the months'
  ``*_wholesale_price.csv``.

Statuses (``DEFAULT_CONFIG``; scenario block ``validation.stage1``):

* FAIL: a main generation line (wind, solar, nuclear, gas incl. CCS gas, biomass, waste,
  gas CHP, hydro) more than 15% off ES1; interconnector imports or exports more than
  20% off; demand more than 0.5% off; an ES1 line with no model mapping.
* WARN: any generation line above 10%; curtailment above 25%; interconnectors and
  electrolysis above 10%; price mean and percentiles above 25% (differences under
  GBP 5/MWh never warn); the zero-price share more than 0.10 off.
* INFO: storage discharge (CP30's PLEXOS allows simultaneous charge and discharge, so
  its throughput includes churn: F15/F16), availability, model-only carriers, load
  shedding.
* A line whose ES1 value is under 1 TWh is judged on its absolute difference, as a
  share of 1 TWh, not of its own small value.
* Stage 1 files solved before the rebuild batch keep no link flows: their electrolysis
  and hydrogen-turbine rows are INFO, not judged.
* ``price_share_warn_rel`` overrides the price band for one share (e.g. ``{0.5: 0.5}``).
* ``energy_basis: info`` (a Stage 1 fitted to another target, e.g. the stage1 family): generation, curtailment,
  interconnector and electrolysis rows are INFO; demand, unmapped lines and price are judged as usual.

On a FAIL the outputs are written, a ``*_failed_run.csv`` copy is kept (Snakemake
deletes a failed rule's outputs), and the rule exits non-zero.

    python scripts/CP30/stage1_validation.py --es1 resources/CP30/CP30_ES1.csv \\
        --pathway "Further Flex and Renewables" --modelled-year 2030 \\
        --networks resources/market/<month>_wholesale.nc ... \\
        --prices resources/market/<month>_wholesale_price.csv ... \\
        --carve-out "CCS Gas=gas" --output-csv out.csv --output-json out.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.CP30.energy_validation import (  # noqa: E402
    build_energy_validation,
    cp30_energy_targets,
    dataset_energy_tables,
    overlay_carved_into,
    sum_energy_tables,
)
from scripts.consumer_bill.price_level_mapping import (  # noqa: E402
    PATHWAYS as CP32_PATHWAYS,
    curve_mean,
    load_duration_curve,
)

#: Code defaults of the scenario block ``validation.stage1`` (config/defaults/method/brownfield.yaml,
#: added 16 Oct 2026; the two must match: tests/unit/test_neso_consistent_package.py).
DEFAULT_CONFIG: dict = {
    "enabled": True,                 # in rule all's CP30 validation targets
    "fail_on_fail": True,            # a FAIL row fails the rule (after the outputs are written)
    # Energy (ES1).
    "main_generation_carriers": ["wind_offshore", "wind_onshore", "solar_pv", "nuclear", "gas",
                                 "biomass", "waste_to_energy", "CHP", "large_hydro"],
    "generation_warn_rel": 0.10,     # every generation line
    "generation_fail_rel": 0.15,     # main generation lines only
    "curtailment_warn_rel": 0.25,
    "interconnector_warn_rel": 0.10,
    "interconnector_fail_rel": 0.20,
    "electrolysis_warn_rel": 0.10,
    "demand_fail_rel": 0.005,
    "small_line_gwh": 1000.0,        # ES1 below this: judged on |difference| / this
    # "es1": generation, curtailment, interconnector and electrolysis rows judged against ES1. "info": reported
    # as INFO, for a Stage 1 fitted to another target (the stage1 family: NESO's CP.28-30 balancing start, 18 Oct
    # 2026). Demand and unmapped ES1 lines are judged either way; price rows are unaffected.
    "energy_basis": "es1",
    # Price (CP.32).
    "price_curve_file": "data/CP30/cp32_price_duration_curve_2030.csv",
    "price_curve_pathway": None,     # None: the scenario's CP30_pathway as a CP.32 column
    "price_shares": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
    "price_warn_rel": 0.25,          # mean and every percentile
    "price_share_warn_rel": {},      # per-share override, e.g. {0.5: 0.50}
    "price_min_abs_gbp_per_mwh": 5.0,
    "zero_price_threshold_gbp_per_mwh": 0.5,
    "zero_price_share_warn_abs": 0.10,
}

STAGE = "stage1"
STAGE1_NOTE = "Stage 1 (market, before network constraints): ES1's basis"
STORAGE_NOTE = ("INFO only: CP30's PLEXOS model allows simultaneous charge and discharge, so ES1 storage "
                "throughput includes churn")
FITTED_BASIS_NOTE = ("INFO (energy_basis info): this Stage 1 is fitted to NESO's balancing start (CP.28-30), "
                     "not to ES1; ES1 reported for context")
COLUMNS = ["stage", "component", "item", "es1_lines", "unit", "cp30", "model", "difference",
           "relative_difference", "judged_difference", "basis", "warn_tolerance", "fail_tolerance",
           "status", "note"]


def stage1_config(block: dict | None) -> dict:
    """``validation.stage1`` over :data:`DEFAULT_CONFIG`, refusing unknown keys."""
    block = dict(block or {})
    if unknown := sorted(set(block) - set(DEFAULT_CONFIG)):
        raise ValueError(f"validation.stage1 has unknown keys {unknown}; supported: {sorted(DEFAULT_CONFIG)}")
    return {**copy.deepcopy(DEFAULT_CONFIG), **block}


def curve_column(pathway: str, override: str | None = None) -> str:
    """The CP.32 column of a CP30 pathway ("Further Flex and Renewables" -> further_flex_and_renewables)."""
    column = override or str(pathway).strip().lower().replace(" ", "_")
    if column not in CP32_PATHWAYS:
        raise ValueError(f"No CP.32 price curve for pathway {pathway!r} (column {column!r}); "
                         f"columns: {list(CP32_PATHWAYS)}")
    return column


def _carrier_of(item: str) -> str:
    """``build_energy_validation`` labels a multi-carrier row "gas (CCGT + OCGT + ...)"."""
    return str(item).split(" (", 1)[0]


def _judge(difference: float, reference: float, warn: float | None, fail: float | None,
           small_line: float | None = None, min_abs: float = 0.0) -> tuple[float, str, str]:
    """(judged difference, basis, status) for one row.

    Relative to the reference, or, where ``|reference| < small_line``, the absolute
    difference as a share of ``small_line``. Differences below ``min_abs`` always pass.
    """
    if small_line is not None and abs(reference) < small_line:
        judged, basis = difference / small_line, f"absolute: difference / {small_line:g} (CP30 below it)"
    else:
        judged = difference / reference if reference else (np.inf if difference else 0.0)
        basis = "relative"
    if abs(difference) < min_abs:
        return judged, f"absolute: |difference| below {min_abs:g}", "PASS"
    if fail is not None and abs(judged) > fail:
        return judged, basis, "FAIL"
    if warn is not None and abs(judged) > warn:
        return judged, basis, "WARN"
    return judged, basis, "PASS"


def _row(component, item, unit, cp30, model, warn=None, fail=None, note="", es1_lines="",
         small_line=None, min_abs=0.0, status=None) -> dict:
    difference = rel = judged = np.nan
    basis = ""
    if cp30 is not None and model is not None and not (pd.isna(cp30) or pd.isna(model)):
        difference = float(model) - float(cp30)
        rel = difference / float(cp30) if cp30 else (np.inf if difference else 0.0)
        if status is None and (warn is not None or fail is not None):
            judged, basis, status = _judge(difference, float(cp30), warn, fail, small_line, min_abs)
    return {"stage": STAGE, "component": component, "item": item, "es1_lines": es1_lines, "unit": unit,
            "cp30": cp30, "model": model, "difference": difference, "relative_difference": rel,
            "judged_difference": judged, "basis": basis, "warn_tolerance": warn, "fail_tolerance": fail,
            "status": status or "INFO", "note": note}


NO_LINK_FLOWS_NOTE = ("INFO only: these Stage 1 files keep no link flows (solved before the rebuild batch, F14), "
                      "so electrolysis and hydrogen turbines read as zero")


def _is_link_row(component: str, item: str) -> bool:
    return component == "electrolysis" or (component == "dispatched_energy" and item.startswith("Hydrogen"))


def judge_energy(table: pd.DataFrame, cfg: dict, link_flows: bool = True) -> pd.DataFrame:
    """Re-judge a ``build_energy_validation`` dispatch table on the Stage 1 tolerances.

    ``link_flows`` False (Stage 1 files without ``links_t_p0``) reports the electrolysis and
    hydrogen-turbine rows as INFO: the files cannot show them.
    """
    main = set(cfg["main_generation_carriers"])
    small = float(cfg["small_line_gwh"])
    basis = cfg.get("energy_basis", "es1")
    if basis not in ("es1", "info"):
        raise ValueError(f"validation.stage1.energy_basis must be 'es1' or 'info', not {basis!r}")
    rows = []
    for _, r in table.iterrows():
        comp, item = r["component"], str(r["item"])
        note = r["note"] if isinstance(r["note"], str) and r["note"] else ""
        kw = dict(note=f"{STAGE1_NOTE}; {note}" if note else STAGE1_NOTE,
                  es1_lines=r["es1_lines"] if isinstance(r["es1_lines"], str) else "")
        cp30 = None if pd.isna(r["cp30_gwh"]) else float(r["cp30_gwh"])
        model = None if pd.isna(r["model_gwh"]) else float(r["model_gwh"])
        if comp == "unmapped_es1_line":
            rows.append(_row(comp, item, "GWh", cp30, model, status="FAIL", **kw))
        elif not link_flows and _is_link_row(comp, item):
            kw["note"] = f"{kw['note']}; {NO_LINK_FLOWS_NOTE}"
            rows.append(_row(comp, item, "GWh", cp30, model, **kw))
        elif basis == "info" and comp in ("dispatched_energy", "curtailment", "interconnectors", "electrolysis"):
            kw["note"] = f"{kw['note']}; {FITTED_BASIS_NOTE}"
            rows.append(_row(comp, item, "GWh", cp30, model, **kw))
        elif comp == "dispatched_energy":
            fail = cfg["generation_fail_rel"] if _carrier_of(item) in main else None
            rows.append(_row(comp, item, "GWh", cp30, model, cfg["generation_warn_rel"], fail,
                             small_line=small, **kw))
        elif comp == "curtailment":
            rows.append(_row(comp, item, "GWh", cp30, model, cfg["curtailment_warn_rel"], None,
                             small_line=small, **kw))
        elif comp == "interconnectors":
            rows.append(_row(comp, item, "GWh", cp30, model, cfg["interconnector_warn_rel"],
                             cfg["interconnector_fail_rel"], small_line=small, **kw))
        elif comp == "electrolysis":
            rows.append(_row(comp, item, "GWh", cp30, model, cfg["electrolysis_warn_rel"], None,
                             small_line=small, **kw))
        elif comp == "demand" and cp30 is not None:
            rows.append(_row(comp, item, "GWh", cp30, model, None, cfg["demand_fail_rel"], **kw))
        elif comp == "storage_discharge":
            kw["note"] = f"{kw['note']}; {STORAGE_NOTE}"
            rows.append(_row(comp, item, "GWh", cp30, model, **kw))
        else:   # available_energy, model_only_carrier, load shedding: context
            rows.append(_row(comp, item, "GWh", cp30, model, **kw))
    return pd.DataFrame(rows, columns=COLUMNS)


def curve_value(curve: pd.Series, share: float) -> float:
    """CP.32 price exceeded in ``share`` of hours (linear between its 1% points)."""
    return float(np.interp(share, curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float)))


def curve_share_at_or_below(curve: pd.Series, threshold: float, resolution: int = 100_000) -> float:
    """Share of the year in which the (piecewise-linear) CP.32 curve is at or below ``threshold``."""
    grid = np.linspace(0.0, 1.0, resolution + 1)
    values = np.interp(grid, curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float))
    return float(np.mean(values <= threshold))


def model_price_at_share(prices: pd.Series, share: float) -> float:
    """Model price exceeded in ``share`` of hours: the (1 - share) quantile of the hourly prices."""
    return float(np.quantile(prices.to_numpy(dtype=float), 1.0 - share))


def price_rows(prices: pd.Series, curve: pd.Series, cfg: dict) -> list[dict]:
    """Mean, percentiles and the zero-price share of the Stage 1 price against CP.32."""
    warn, min_abs = float(cfg["price_warn_rel"]), float(cfg["price_min_abs_gbp_per_mwh"])
    by_share = {float(k): float(v) for k, v in (cfg.get("price_share_warn_rel") or {}).items()}
    source = f"CP.32 {curve.name}"
    note = (f"Stage 1 wholesale price, {len(prices)} h, against {source}; differences under GBP {min_abs:g}/MWh "
            "never warn")
    rows = [_row("price", "mean", "GBP/MWh", curve_mean(curve), float(prices.mean()), warn, None,
                 note=note, es1_lines=f"{source} (trapezoid mean)", min_abs=min_abs)]
    for share in cfg["price_shares"]:
        rows.append(_row("price", f"price exceeded in {100 * float(share):g}% of hours", "GBP/MWh",
                         curve_value(curve, float(share)), model_price_at_share(prices, float(share)),
                         by_share.get(float(share), warn), None,
                         note=note, es1_lines=f"{source} at share {float(share):g}", min_abs=min_abs))
    threshold = float(cfg["zero_price_threshold_gbp_per_mwh"])
    cp30_share = curve_share_at_or_below(curve, threshold)
    model_share = float((prices <= threshold).mean())
    diff = model_share - cp30_share
    tol = float(cfg["zero_price_share_warn_abs"])
    rows.append({**_row("price", f"share of hours at or below GBP {threshold:g}/MWh", "share of hours", cp30_share,
                        model_share, note=f"absolute tolerance {tol:g}", es1_lines=f"{source} (interpolated)"),
                 "judged_difference": diff, "basis": "absolute", "warn_tolerance": tol,
                 "status": "WARN" if abs(diff) > tol else "PASS"})
    return rows


def has_link_flows(paths: list[str]) -> bool:
    """Whether every Stage 1 file keeps its link flows (``links_t_p0``; saved since the rebuild batch, F14)."""
    import xarray as xr

    for path in paths:
        with xr.open_dataset(path) as ds:
            if "links_t_p0" not in ds:
                return False
    return True


def read_stage1_prices(paths: list[str]) -> pd.Series:
    """The months' hourly Stage 1 prices (``wholesale_price``), in order."""
    parts = []
    for path in paths:
        frame = pd.read_csv(path)
        if "wholesale_price" not in frame.columns:
            raise ValueError(f"{path} has no wholesale_price column")
        series = pd.to_numeric(frame["wholesale_price"], errors="coerce")
        if series.isna().any():
            raise ValueError(f"{path}: {int(series.isna().sum())} hours without a price")
        parts.append(series)
    return pd.concat(parts, ignore_index=True).rename("wholesale_price")


def build_stage1_validation(targets: dict, tables: dict, prices: pd.Series, curve: pd.Series,
                            cfg: dict | None = None, carved_into: dict | None = None,
                            link_flows: bool = True) -> pd.DataFrame:
    """The Stage 1 validation table: ES1 energy rows re-judged, then the CP.32 price rows."""
    cfg = stage1_config(cfg)
    energy = build_energy_validation(targets, tables, stage="dispatch", carved_into=carved_into)
    rows = judge_energy(energy, cfg, link_flows).to_dict("records") + price_rows(prices, curve, cfg)
    return pd.DataFrame(rows, columns=COLUMNS)


def format_stage1_validation(table: pd.DataFrame) -> str:
    view = table[["component", "item", "unit", "cp30", "model", "judged_difference", "status"]].copy()
    energy = view["unit"].eq("GWh")
    view.loc[energy, ["cp30", "model"]] = view.loc[energy, ["cp30", "model"]] / 1e3
    view.loc[energy, "unit"] = "TWh"
    return view.to_string(index=False, float_format=lambda v: f"{v:,.3f}")


def run(es1_path: str, pathway: str, modelled_year: int, networks: list[str], price_files: list[str],
        output_csv: str, output_json: str, cfg: dict | None = None, hydro_config: dict | None = None,
        carved_into: dict | None = None, curve_file: str | None = None, log_path: str | None = None,
        family: str = "") -> pd.DataFrame:
    from scripts.utilities.logging_config import setup_logging

    logger = setup_logging(log_path or "validate_cp30_stage1")
    cfg = stage1_config(cfg)
    if len(price_files) != len(networks):
        raise ValueError(f"{len(price_files)} price files for {len(networks)} Stage 1 networks")
    parts = []
    for path in networks:
        parts.append(dataset_energy_tables(path, hydro_config))
        logger.info("Read %s (%.0f h)", Path(path).name, parts[-1]["hours"])
    tables = sum_energy_tables(parts)
    if abs(tables["hours"] - 8760.0) > 0.5 and abs(tables["hours"] - 8784.0) > 0.5:
        logger.warning("Stage 1 months cover %.0f h, not a calendar year", tables["hours"])
    prices = read_stage1_prices(price_files)
    if abs(len(prices) - tables["hours"]) > 0.5:
        raise ValueError(f"{len(prices)} price hours for {tables['hours']:.0f} network hours")
    column = curve_column(pathway, cfg.get("price_curve_pathway"))
    curve = load_duration_curve(curve_file or cfg["price_curve_file"], column)
    link_flows = has_link_flows(networks)
    if not link_flows:
        logger.warning("Stage 1 files keep no link flows: electrolysis and hydrogen turbines reported as INFO")
    table = build_stage1_validation(cp30_energy_targets(es1_path, pathway, modelled_year), tables, prices, curve,
                                    cfg, carved_into, link_flows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    summary = {"family": family, "pathway": pathway, "modelled_year": modelled_year, "months": len(networks),
               "hours": tables["hours"], "price_curve": column, "carved_into": dict(carved_into or {}),
               "link_flows": link_flows,
               "config": cfg, "status_counts": table["status"].value_counts().to_dict(),
               "networks": [str(p) for p in networks],
               "rows": json.loads(table.to_json(orient="records"))}
    Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("CP30 Stage 1 validation (%d months):\n%s", len(networks), format_stage1_validation(table))
    for _, row in table[table["status"].isin(["WARN", "FAIL"])].iterrows():
        logger.warning("CP30 Stage 1 %s %s %s: model %.3f vs CP30 %.3f %s (judged %+.3f)", row["status"],
                       row["component"], row["item"], row["model"], row["cp30"], row["unit"], row["judged_difference"])
    fails = table[table["status"].eq("FAIL")]
    if len(fails) and cfg["fail_on_fail"]:
        # Snakemake deletes a failed rule's outputs: keep the table beside it for the review.
        kept = Path(output_csv).with_name(Path(output_csv).stem + "_failed_run.csv")
        table.to_csv(kept, index=False)
        raise ValueError(f"CP30 Stage 1 validation failed on {len(fails)} rows (full table kept at {kept}):\n"
                         + fails[["component", "item", "unit", "cp30", "model", "judged_difference"]]
                         .to_string(index=False))
    return table


if "snakemake" in globals():
    _config = dict(snakemake.params.scenario_config)
    run(
        es1_path=snakemake.input.cp30_es1,
        pathway=_config.get("CP30_pathway"),
        modelled_year=int(_config.get("modelled_year")),
        networks=list(snakemake.input.wholesale_networks),
        price_files=list(snakemake.input.wholesale_prices),
        output_csv=snakemake.output.csv,
        output_json=snakemake.output.json,
        cfg=_config.get("validation", {}).get("stage1"),
        hydro_config=_config.get("renewables", {}).get("hydro", {}),
        carved_into=overlay_carved_into(_config),
        curve_file=snakemake.input.price_curve,
        log_path=snakemake.log[0] if snakemake.log else None,
        family=snakemake.wildcards.study,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--es1", required=True)
    parser.add_argument("--pathway", required=True)
    parser.add_argument("--modelled-year", type=int, required=True)
    parser.add_argument("--networks", nargs="+", required=True)
    parser.add_argument("--prices", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--inflow-profile-factor", type=float, default=0.55,
                        help="large-hydro reservoir inflow factor; 0 = reservoir model off")
    parser.add_argument("--carve-out", action="append", default=[], metavar="CARRIER=LINE",
                        help="overlay carrier counted in a CP30 line, e.g. 'CCS Gas=gas' (the FFR 2030 build)")
    parser.add_argument("--report-only", action="store_true", help="never exit non-zero on a FAIL")
    args = parser.parse_args()
    hydro = ({"enabled": True, "large_hydro": {"reservoir_model": True,
                                              "inflow_profile_factor": args.inflow_profile_factor}}
             if args.inflow_profile_factor else {})
    run(args.es1, args.pathway, args.modelled_year, args.networks, args.prices, args.output_csv, args.output_json,
        {"fail_on_fail": not args.report_only}, hydro, dict(item.split("=", 1) for item in args.carve_out))
