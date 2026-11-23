#!/usr/bin/env python3
"""Validate a solved CP30 calendar year's FINAL dispatch against CP30 energy.

Reads the balancing-stage NetCDF of every month in a
``monthly_aggregates`` study (a study's study.yaml) (the physical dispatch after redispatch),
sums them, and reconciles with CP30 ES1 (scripts/CP30/energy_validation.py):
dispatched energy by technology against ES1 generation, curtailment against
ES1 curtailment, demand, storage, interconnectors and electrolysis, plus every
ES1 line with no model mapping.

Energy differences are model results and only WARN; an unmapped ES1 line or a
demand mismatch FAILS (the rule then errors, after writing its outputs).

Market basis (follow-up F15, 6 Oct 2026). ES1 is CP30's unconstrained dispatch
("no generation output is adjusted to overcome network or operability
constraints"), and its curtailment is market curtailment (CP.28 "Renewable
curtailment (market)"). Given the Stage 1 (wholesale) networks, the generation,
curtailment, storage and interconnector rows are therefore judged on Stage 1.
The final (balancing) values stay as ``*_final`` INFO rows. Hydrogen turbines and
electrolysis stay on the final dispatch: the Stage 1 files keep no link flows,
and the balancing stage holds those links at Stage 1. Constraint curtailment is
a separate CP30 figure (CP.08, clean-power bid volumes in redispatch), set
against the gross clean-power turn-down from Stage 1 to Stage 2.

    python scripts/CP30/validate_cp30_dispatch.py --es1 resources/CP30/CP30_ES1.csv \
        --pathway "Further Flex and Renewables" --modelled-year 2030 \
        --networks resources/market/<month>_balancing.nc ... \
        [--wholesale-networks resources/market/<month>_wholesale.nc ... --cp08 <file>] \
        --output-csv out.csv --output-json out.json
"""


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.CP30.energy_validation import (
    DEFAULT_TOLERANCES,
    build_energy_validation,
    cp30_energy_targets,
    dataset_energy_tables,
    energy_validation_failures,
    format_energy_validation,
    overlay_carved_into,
    sum_energy_tables,
)
from scripts.utilities.logging_config import setup_logging

#: Rows ES1 states on its unconstrained (market) basis, judged on Stage 1 when it is given.
MARKET_BASIS_COMPONENTS = ("dispatched_energy", "curtailment", "available_energy", "model_only_carrier",
                           "storage_discharge", "interconnectors")
#: Of those, the final rows kept as INFO beside the Stage 1 ones, renamed ``<component>_final``.
FINAL_MEMO_COMPONENTS = ("dispatched_energy", "curtailment", "storage_discharge", "interconnectors")
#: Clean power whose balancing-stage turn-down CP.08 counts (renewables and nuclear).
CLEAN_POWER_CARRIERS = ("wind_offshore", "wind_onshore", "solar_pv", "large_hydro", "small_hydro", "marine",
                        "nuclear")
#: CP.08's network case for the CP30 model (the network CP30 recommends, which the build carries).
CP08_NETWORK = "Network capacity we are recommending for CP30"
STAGE1_NOTE = "Stage 1 (market, before network constraints): ES1's basis"
FINAL_NOTE = "after the balancing stage; ES1 is unconstrained, so the Stage 1 row is the one judged"


def _hydrogen_turbine_row(table: pd.DataFrame) -> pd.Series:
    return table["component"].eq("dispatched_energy") & table["item"].astype(str).str.startswith("Hydrogen")


def market_basis_table(stage1: pd.DataFrame, final: pd.DataFrame,
                       constraint: dict | None = None) -> pd.DataFrame:
    """The validation table with the ES1 rows judged on Stage 1 (follow-up F15).

    ``stage1`` and ``final`` are :func:`build_energy_validation` dispatch tables of
    the Stage 1 and the balancing networks. Stage 1 supplies the market-basis rows
    (the hydrogen turbines excepted); the final table supplies demand, load
    shedding, electrolysis, the turbines and unmapped lines, plus INFO memo rows
    ``<component>_final``. ``constraint`` is an optional CP.08 row.
    """
    s1 = stage1[stage1["component"].isin(MARKET_BASIS_COMPONENTS) & ~_hydrogen_turbine_row(stage1)].copy()
    s1["stage"] = "dispatch_stage1"
    s1["note"] = [f"{STAGE1_NOTE}; {n}" if isinstance(n, str) and n else STAGE1_NOTE for n in s1["note"]]
    kept = final[~final["component"].isin(MARKET_BASIS_COMPONENTS) | _hydrogen_turbine_row(final)].copy()
    kept["stage"] = "dispatch_final"
    memo = final[final["component"].isin(FINAL_MEMO_COMPONENTS) & ~_hydrogen_turbine_row(final)].copy()
    memo["component"] = memo["component"] + "_final"
    memo["stage"] = "dispatch_final"
    memo["status"] = "INFO"
    memo["tolerance"] = np.nan
    memo["note"] = FINAL_NOTE
    parts = [s1, kept, memo]
    if constraint is not None:
        parts.append(pd.DataFrame([{**constraint, "stage": "dispatch_balancing"}]))
    out = pd.concat(parts, ignore_index=True)
    return out[list(final.columns)]


def clean_power_turn_down_mwh(wholesale_path: str | Path, balancing_path: str | Path,
                              carriers=CLEAN_POWER_CARRIERS) -> pd.Series:
    """Gross clean-power turn-down from Stage 1 to Stage 2 in one month, MWh by carrier.

    Sum over generators and hours of max(Stage 1 - Stage 2, 0): the volume the
    balancing stage bid off, as CP.08 counts bid volumes. Matched by generator
    name; generators only in Stage 2 (balancing-only units) are left out.
    """
    import xarray as xr

    def read(path):
        with xr.open_dataset(path) as ds:
            names = ds["generators_i"].values.astype(str)
            carrier = pd.Series(ds["generators_carrier"].values.astype(str), index=names)
            p = pd.DataFrame(ds["generators_t_p"].values, columns=ds["generators_t_p_i"].values.astype(str))
            weights = (ds["snapshots_generators"].values.astype(float) if "snapshots_generators" in ds
                       else np.ones(len(p)))
        return carrier, p, weights

    _, p1, _ = read(wholesale_path)
    carrier, p2, weights = read(balancing_path)
    if len(p1) != len(p2):
        raise ValueError(f"{Path(wholesale_path).name} has {len(p1)} snapshots but "
                         f"{Path(balancing_path).name} has {len(p2)}")
    keep = [c for c in p2.columns if c in p1.columns and carrier.get(c) in set(carriers)]
    down = np.clip(p1[keep].to_numpy() - p2[keep].to_numpy(), 0.0, None) * weights[:, None]
    return pd.Series(down.sum(axis=0), index=keep).groupby(carrier.reindex(keep).to_numpy()).sum()


def cp08_bid_volume_gwh(path: str | Path, pathway: str, network: str = CP08_NETWORK) -> float:
    """CP.08's clean-power constraint bid volume for a pathway and network, GWh."""
    table = pd.read_csv(path)
    row = table[table["network"].eq(network) & table["pathway"].eq(pathway)]
    if len(row) != 1:
        raise ValueError(f"{path}: expected one CP.08 row for {network!r} / {pathway!r}, found {len(row)}")
    return float(row["bid_volume_twh"].iloc[0]) * 1e3


def constraint_curtailment_row(target_gwh: float, turn_down_mwh: pd.Series,
                               tolerances: dict | None = None) -> dict:
    """The CP.08 row: gross clean-power turn-down in the balancing stage against CP30's bid volume."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    model = float(turn_down_mwh.sum()) / 1e3
    difference = model - target_gwh
    rel = difference / target_gwh if target_gwh else np.nan
    within = abs(rel) <= tol["curtailment_rel_tol"] or abs(difference) < tol["min_abs_gwh"]
    parts = ", ".join(f"{c} {v / 1e6:.1f}" for c, v in turn_down_mwh.sort_values(ascending=False).items() if v > 0)
    return {"component": "constraint_curtailment", "item": "clean power turned down in the balancing stage",
            "es1_lines": "CP.08 bid volume, recommended CP30 network", "cp30_gwh": target_gwh, "model_gwh": model,
            "difference_gwh": difference, "relative_difference": rel, "tolerance": tol["curtailment_rel_tol"],
            "status": "PASS" if within else "WARN",
            "note": f"gross Stage 1 minus Stage 2 by generator and hour (TWh: {parts})"}


def run(es1_path: str, pathway: str, modelled_year: int, networks: list[str], output_csv: str,
        output_json: str, hydro_config: dict | None = None, tolerances: dict | None = None,
        log_path: str | None = None, wholesale_networks: list[str] | None = None,
        cp08_path: str | None = None, carved_into: dict | None = None) -> None:
    """``carved_into``: {overlay carrier: CP30 line} (``energy_validation.overlay_carved_into``), e.g. NZT's
    CCS Gas counted in the Gas line (user decision 14 Oct 2026)."""
    logger = setup_logging(log_path or "validate_cp30_dispatch")
    parts = []
    for path in networks:
        parts.append(dataset_energy_tables(path, hydro_config))
        logger.info("Read %s (%.0f h)", Path(path).name, parts[-1]["hours"])
    tables = sum_energy_tables(parts)
    if abs(tables["hours"] - 8760.0) > 0.5 and abs(tables["hours"] - 8784.0) > 0.5:
        logger.warning("Solved months cover %.0f h, not a calendar year: CP30 annual lines are "
                       "compared with a partial year", tables["hours"])
    targets = cp30_energy_targets(es1_path, pathway, modelled_year)
    table = build_energy_validation(targets, tables, stage="dispatch", tolerances=tolerances, carved_into=carved_into)
    if wholesale_networks:
        if len(wholesale_networks) != len(networks):
            raise ValueError(f"{len(wholesale_networks)} Stage 1 networks for {len(networks)} balancing networks")
        stage1 = sum_energy_tables([dataset_energy_tables(path, hydro_config) for path in wholesale_networks])
        stage1_table = build_energy_validation(targets, stage1, stage="dispatch", tolerances=tolerances,
                                               carved_into=carved_into)
        constraint = None
        if cp08_path:
            turn_down = pd.concat([clean_power_turn_down_mwh(w, b) for w, b in zip(wholesale_networks, networks)])
            constraint = constraint_curtailment_row(cp08_bid_volume_gwh(cp08_path, pathway),
                                                    turn_down.groupby(level=0).sum(), tolerances)
        table = market_basis_table(stage1_table, table, constraint)
        logger.info("ES1 rows judged on Stage 1 (%d months); final values kept as *_final INFO rows",
                    len(wholesale_networks))

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    summary = {
        "pathway": pathway, "modelled_year": modelled_year, "months": len(networks),
        "hours": tables["hours"], "status_counts": table["status"].value_counts().to_dict(),
        "rows": json.loads(table.to_json(orient="records")),
    }
    Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("CP30 final-dispatch energy validation (%d months):\n%s", len(networks),
                format_energy_validation(table))
    warnings = table[table["status"].eq("WARN")]
    for _, row in warnings.iterrows():
        logger.warning("CP30 %s %s: model %.2f TWh vs CP30 %.2f TWh (%+.1f%%)", row["component"], row["item"],
                       row["model_gwh"] / 1e3, row["cp30_gwh"] / 1e3, 100 * row["relative_difference"])
    failures = energy_validation_failures(table)
    if not failures.empty:
        raise ValueError("CP30 dispatch energy validation failed:\n"
                         + failures[["component", "item", "cp30_gwh", "model_gwh", "note"]].to_string(index=False))


if "snakemake" in globals():
    config = dict(snakemake.params.scenario_config)
    run(
        es1_path=snakemake.input.cp30_es1,
        pathway=config.get("CP30_pathway"),
        modelled_year=int(config.get("modelled_year")),
        networks=list(snakemake.input.networks),
        output_csv=snakemake.output.csv,
        output_json=snakemake.output.json,
        hydro_config=config.get("renewables", {}).get("hydro", {}),
        tolerances=(config.get("cp30_validation", {}) or {}).get("energy"),
        log_path=snakemake.log[0] if snakemake.log else None,
        wholesale_networks=list(getattr(snakemake.input, "wholesale_networks", []) or []) or None,
        cp08_path=getattr(snakemake.input, "cp30_cp08", None) or None,
        carved_into=overlay_carved_into(config),
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--es1", required=True)
    parser.add_argument("--pathway", required=True)
    parser.add_argument("--modelled-year", type=int, required=True)
    parser.add_argument("--networks", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--inflow-profile-factor", type=float, default=0.55,
                        help="large-hydro reservoir inflow factor; 0 = reservoir model off")
    parser.add_argument("--wholesale-networks", nargs="*", default=None,
                        help="the months' Stage 1 networks, in the order of --networks (market basis, F15)")
    parser.add_argument("--cp08", default=None, help="data/CP30/cp30_cp08_constraint_bid_volumes_2030.csv")
    parser.add_argument("--carve-out", action="append", default=[], metavar="CARRIER=LINE",
                        help="overlay carrier counted in a CP30 line, e.g. 'CCS Gas=gas' (the FFR 2030 build)")
    args = parser.parse_args()
    hydro = ({"enabled": True, "large_hydro": {"reservoir_model": True,
                                              "inflow_profile_factor": args.inflow_profile_factor}}
             if args.inflow_profile_factor else {})
    run(args.es1, args.pathway, args.modelled_year, args.networks, args.output_csv, args.output_json, hydro,
        wholesale_networks=args.wholesale_networks, cp08_path=args.cp08,
        carved_into=dict(item.split("=", 1) for item in args.carve_out))
