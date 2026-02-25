"""
Solve Wholesale Market — Stage 1 of Two-Stage Market Dispatch

Solves a copperplate (unconstrained) dispatch representing the GB wholesale
electricity market. All line and transformer thermal limits are relaxed to a
very large value so that power flows freely across the network, producing a
single uniform clearing price.

Inputs:
  - Finalized network: {scenario}.nc

Outputs:
  - Solved wholesale network: {scenario}_wholesale.nc
  - Wholesale generator dispatch CSV: {scenario}_wholesale_dispatch.csv
  - Wholesale storage dispatch CSV: {scenario}_wholesale_storage.csv
  - Wholesale link dispatch CSV: {scenario}_wholesale_links.csv
  - Wholesale price CSV: {scenario}_wholesale_price.csv

See Also:
  - scripts/market/solve_balancing.py — Stage 2 (BM redispatch)
  - scripts/market/market_utils.py — Shared utilities
  - scripts/solve/solve_network.py — Standard (constrained) solve
"""

import pypsa
import logging
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network, save_network
from scripts.solve.solve_network import (
    validate_network_costs,
    improve_numerical_conditioning,
    configure_solver,
    get_solve_mode_from_config,
)
from scripts.market.market_utils import extract_wholesale_positions


def apply_copperplate_relaxation(network, wholesale_config, logger):
    """
    Remove all network constraints by setting line/transformer ratings to
    a very large value, creating a copperplate (single-price) system.

    The original ratings are stored in ``s_nom_original`` columns so they
    can be referenced later (e.g., for reporting).

    Parameters
    ----------
    network : pypsa.Network
        Network to modify in-place.
    wholesale_config : dict
        Wholesale market config containing 'transmission_relaxation' (MVA).
    logger : logging.Logger
        Logger instance.
    """
    relaxation = float(wholesale_config.get("transmission_relaxation", 1.0e6))

    logger.info("=" * 80)
    logger.info("APPLYING COPPERPLATE RELAXATION (WHOLESALE MARKET)")
    logger.info("=" * 80)

    # Store originals
    if len(network.lines) > 0:
        network.lines["s_nom_original"] = network.lines["s_nom"].copy()
        network.lines["s_nom"] = relaxation
        logger.info(
            f"Lines: set s_nom = {relaxation:.0e} MVA for {len(network.lines)} lines "
            f"(original range: {network.lines['s_nom_original'].min():.0f}-"
            f"{network.lines['s_nom_original'].max():.0f} MVA)"
        )

    if len(network.transformers) > 0:
        network.transformers["s_nom_original"] = network.transformers["s_nom"].copy()
        network.transformers["s_nom"] = relaxation
        logger.info(
            f"Transformers: set s_nom = {relaxation:.0e} MVA for "
            f"{len(network.transformers)} transformers"
        )

    logger.info("Network is now effectively copperplate (no transmission constraints)")
    logger.info("=" * 80)


def apply_solve_period(network, scenario_config, logger):
    """
    Restrict network snapshots to the configured solve period.

    Mirrors the solve-period logic from solve_network.py.

    Parameters
    ----------
    network : pypsa.Network
        Network to modify in-place.
    scenario_config : dict
        Scenario configuration (may contain 'solve_period' key).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    bool
        True if snapshots were restricted.
    """
    solve_period_config = scenario_config.get("solve_period", {})
    if not solve_period_config.get("enabled", False):
        logger.info("Solving full year (no period restriction)")
        return False

    logger.info("=" * 80)
    logger.info("APPLYING SOLVE PERIOD RESTRICTION")
    logger.info("=" * 80)

    if "auto_select" in solve_period_config:
        auto_mode = solve_period_config["auto_select"]
        logger.info(f"Auto-selecting period: {auto_mode}")

        if auto_mode == "peak_demand_week":
            if len(network.loads_t.p_set) > 0:
                total_demand = network.loads_t.p_set.sum(axis=1)
                weekly_demand = total_demand.resample("W").sum()
                peak_week_end = weekly_demand.idxmax()
                solve_start = peak_week_end - pd.Timedelta(days=6)
                solve_end = peak_week_end
            else:
                raise ValueError("Cannot auto-select peak demand week: no load data")
        elif auto_mode == "peak_wind_week":
            wind_gens = network.generators[
                network.generators.carrier.str.contains("wind", case=False, na=False)
            ]
            if len(wind_gens) > 0 and len(network.generators_t.p_max_pu) > 0:
                wind_cols = [g for g in wind_gens.index if g in network.generators_t.p_max_pu.columns]
                total_wind_cf = network.generators_t.p_max_pu[wind_cols].mean(axis=1)
                weekly_wind = total_wind_cf.resample("W").mean()
                peak_week_end = weekly_wind.idxmax()
                solve_start = peak_week_end - pd.Timedelta(days=6)
                solve_end = peak_week_end
            else:
                raise ValueError("Cannot auto-select peak wind week: no wind data")
        elif auto_mode == "low_wind_week":
            wind_gens = network.generators[
                network.generators.carrier.str.contains("wind", case=False, na=False)
            ]
            if len(wind_gens) > 0 and len(network.generators_t.p_max_pu) > 0:
                wind_cols = [g for g in wind_gens.index if g in network.generators_t.p_max_pu.columns]
                total_wind_cf = network.generators_t.p_max_pu[wind_cols].mean(axis=1)
                weekly_wind = total_wind_cf.resample("W").mean()
                low_week_end = weekly_wind.idxmin()
                solve_start = low_week_end - pd.Timedelta(days=6)
                solve_end = low_week_end
            else:
                raise ValueError("Cannot auto-select low wind week: no wind data")
        else:
            raise ValueError(f"Unknown auto_select mode: {auto_mode}")
    else:
        solve_start = pd.Timestamp(solve_period_config["start"])
        solve_end = pd.Timestamp(solve_period_config["end"])
        logger.info(f"Explicit solve period: {solve_start} to {solve_end}")

    original_snapshots = len(network.snapshots)
    mask = (network.snapshots >= solve_start) & (network.snapshots <= solve_end)
    selected = network.snapshots[mask]

    if len(selected) == 0:
        raise ValueError(f"No snapshots in period {solve_start} to {solve_end}")

    network.set_snapshots(selected)
    logger.info(
        f"Snapshots: {original_snapshots} → {len(network.snapshots)} "
        f"({100 * (1 - len(network.snapshots) / original_snapshots):.1f}% reduction)"
    )
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────
    log_path = (
        snakemake.log[0]
        if hasattr(snakemake, "log") and snakemake.log
        else "solve_wholesale"
    )
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("STAGE 1: SOLVING WHOLESALE MARKET (COPPERPLATE DISPATCH)")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # ── Load network ─────────────────────────────────────────────────
        input_path = snakemake.input.network
        logger.info(f"Loading finalized network from: {input_path}")
        network = load_network(input_path, custom_logger=logger)

        scenario_config = snakemake.params.scenario_config
        market_config = scenario_config.get("market", {})
        wholesale_config = market_config.get("wholesale", {})
        solver_name = snakemake.params.solver
        solver_options = snakemake.params.solver_options
        scenario_id = scenario_config.get("scenario_id", snakemake.wildcards.scenario)

        logger.info(f"Scenario: {scenario_id}")
        logger.info(
            f"Network: {len(network.buses)} buses, "
            f"{len(network.generators)} generators, "
            f"{len(network.storage_units)} storage units, "
            f"{len(network.links)} links"
        )

        # ── Pre-solve validation ─────────────────────────────────────────
        validate_network_costs(network, logger)

        # ── Apply copperplate relaxation ─────────────────────────────────
        apply_copperplate_relaxation(network, wholesale_config, logger)

        # ── Numerical conditioning ───────────────────────────────────────
        improve_numerical_conditioning(network, logger)

        # ── Apply solve period ───────────────────────────────────────────
        apply_solve_period(network, scenario_config, logger)

        logger.info(
            f"Optimization will run for {len(network.snapshots)} snapshots "
            f"(copperplate — no line constraints)"
        )

        # ── Solve mode (LP / MILP) ───────────────────────────────────────
        global_solve_mode = get_solve_mode_from_config()
        logger.info(f"Solve mode: {global_solve_mode}")

        if global_solve_mode == "LP":
            if "committable" in network.generators.columns:
                network.generators["committable"] = False
            logger.info("LP mode: unit commitment disabled")

        # Remove must-run if configured
        remove_must_run = scenario_config.get("optimization", {}).get("remove_must_run", False)
        if remove_must_run and "p_min_pu" in network.generators.columns:
            network.generators["p_min_pu"] = 0.0
            logger.info("Removed must-run constraints (p_min_pu = 0)")

        # ── Configure solver ─────────────────────────────────────────────
        solver_name, solver_options = configure_solver(
            network, solver_name, solver_options, logger
        )

        # ── Optimize ─────────────────────────────────────────────────────
        logger.info("Starting wholesale market optimization...")
        solve_start = time.time()

        status, termination_condition = network.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
        )

        solve_time = time.time() - solve_start
        logger.info(f"Wholesale solve completed in {solve_time:.2f}s")
        logger.info(f"Status: {status}, Condition: {termination_condition}")

        if status != "ok":
            raise RuntimeError(
                f"Wholesale optimization failed: {status} ({termination_condition})"
            )

        if hasattr(network, "objective"):
            logger.info(f"Total wholesale system cost: £{network.objective:,.2f}")

        # ── Extract and export wholesale positions ───────────────────────
        gen_dispatch, su_dispatch, link_dispatch = extract_wholesale_positions(
            network, logger
        )

        # Save dispatch CSVs
        gen_dispatch.to_csv(snakemake.output.wholesale_dispatch_csv)
        logger.info(f"Saved: {snakemake.output.wholesale_dispatch_csv}")

        su_dispatch.to_csv(snakemake.output.wholesale_storage_csv)
        logger.info(f"Saved: {snakemake.output.wholesale_storage_csv}")

        link_dispatch.to_csv(snakemake.output.wholesale_links_csv)
        logger.info(f"Saved: {snakemake.output.wholesale_links_csv}")

        # Wholesale price (should be uniform across all AC buses in copperplate)
        if hasattr(network, "buses_t") and len(network.buses_t.marginal_price) > 0:
            prices = network.buses_t.marginal_price

            # Filter to GB-only demand (load) buses.
            #
            # Two sources of price contamination are excluded:
            #   1. DC link buses (H2 turbines, electrolysers, internal HVDC)
            #      retain directional cost differentials even in copperplate.
            #   2. External (non-GB) interconnector buses (HVDC_External_*)
            #      carry EU_demand loads with p_set=0.  When historical
            #      p_set forces GB exports, those buses have no viable sink
            #      and the LP assigns extreme negative shadow prices
            #      (e.g. −£445,000/MWh) which corrupt the GB wholesale price.
            load_buses = set(network.loads["bus"].unique())
            if "country" in network.buses.columns:
                gb_buses = set(network.buses.index[network.buses["country"] == "GB"])
                demand_price_cols = [
                    c for c in prices.columns if c in load_buses and c in gb_buses
                ]
                n_excluded = len([c for c in prices.columns if c in load_buses and c not in gb_buses])
                if n_excluded:
                    logger.info(
                        f"Excluded {n_excluded} non-GB demand buses from price "
                        f"calculation (EU_demand loads on external interconnector buses)"
                    )
            else:
                demand_price_cols = [c for c in prices.columns if c in load_buses]
            if demand_price_cols:
                prices_demand = prices[demand_price_cols]
                logger.info(
                    f"Wholesale price computed from {len(demand_price_cols)} "
                    f"GB demand buses (of {len(prices.columns)} total)"
                )
            else:
                prices_demand = prices
                logger.warning(
                    "No GB demand buses found in marginal_price columns; "
                    "using all buses (spread may be inflated)"
                )

            uniform_price = prices_demand.mean(axis=1)
            price_spread = prices_demand.max(axis=1) - prices_demand.min(axis=1)
            max_spread = price_spread.max()

            price_df = pd.DataFrame({
                "wholesale_price": uniform_price,
                "price_spread": price_spread,
            })
            price_df.to_csv(snakemake.output.wholesale_price_csv)
            logger.info(f"Saved: {snakemake.output.wholesale_price_csv}")
            logger.info(
                f"Wholesale price: mean £{uniform_price.mean():.2f}/MWh, "
                f"max spread £{max_spread:.4f}/MWh "
                f"(should be ~0 for copperplate)"
            )
        else:
            # Create empty price file
            pd.DataFrame(columns=["wholesale_price", "price_spread"]).to_csv(
                snakemake.output.wholesale_price_csv
            )
            logger.warning("No marginal price data available from wholesale solve")

        # ── Save solved wholesale network ────────────────────────────────
        save_network(network, snakemake.output.network, custom_logger=logger)
        logger.info(f"Saved wholesale network: {snakemake.output.network}")

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(
            f"STAGE 1 COMPLETE — WHOLESALE MARKET SOLVED "
            f"(Total: {total_time:.2f}s)"
        )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"FATAL ERROR in wholesale market solve: {e}", exc_info=True)
        raise
