"""
Solve Wholesale Market — Stage 1 of Two-Stage Market Dispatch

import pypsa
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
from scripts.solve.hydro_constraints import (
    build_hydro_constraints_callback,
    combine_extra_functionalities,
    initialise_large_hydro_storage_state,
    log_hydro_constraint_setup,
    update_large_hydro_storage_state,
)
from scripts.market.market_utils import extract_wholesale_positions


def _identify_interconnector_links(network):
    """Return list of link names that are cross-border interconnectors.

    Interconnectors are identified by having a bus connected to an external
    node (bus name containing 'External').
    """
    ic_names = []
    for lk in network.links.index:
        bus0 = str(network.links.loc[lk, "bus0"])
        bus1 = str(network.links.loc[lk, "bus1"])
        if "External" in bus0 or "External" in bus1:
            ic_names.append(lk)
    return ic_names


def _build_fix_interconnectors_callback(network, logger):
    """Build an extra_functionality callback that fixes IC link flows to p_set.

    Returns None if there are no IC links with p_set data.
    """
    ic_names = _identify_interconnector_links(network)
    if not ic_names:
        return None

    # Check which IC links have p_set data
    if network.links_t.p_set.empty:
        logger.info("No links_t.p_set data — interconnector flows will be optimised")
        return None

    ic_with_pset = [lk for lk in ic_names if lk in network.links_t.p_set.columns]
    if not ic_with_pset:
        logger.info("No p_set data for IC links — interconnector flows will be optimised")
        return None

    logger.info(
        f"Will fix {len(ic_with_pset)} interconnector flows to historical p_set: "
        f"{ic_with_pset}"
    )

    def fix_interconnectors(n, sns):
        model = n.model
        if "Link-p" not in model.variables:
            return
        link_p = model.variables["Link-p"]
        for lk in ic_with_pset:
            if lk in link_p.coords["name"].values:
                p_set_vals = n.links_t.p_set[lk].reindex(sns).values
                model.add_constraints(
                    link_p.sel(name=lk) == p_set_vals,
                    name=f"Link-fix_ic_{lk}",
                )

    return fix_interconnectors


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


def solve_rolling_day_ahead(
    network, wholesale_config, solver_name, solver_options, scenario_config, logger
):
    """
    Solve wholesale market in rolling day-ahead windows.

    Each window (default 24 hours) is solved independently as a copperplate
    LP. Storage state-of-charge optionally carries over between consecutive
    windows.  This produces more realistic day-ahead scheduling because
    generators and storage can only see ``window_hours`` ahead.

    Parameters
    ----------
    network : pypsa.Network
        Copperplate-relaxed network with full snapshots already set.
    wholesale_config : dict
        Wholesale config (``window_hours``, ``carry_soc``).
    solver_name : str
    solver_options : dict
    scenario_config : dict
    logger : logging.Logger

    Returns
    -------
    gen_dispatch : pd.DataFrame
        Concatenated generator dispatch (snapshots x generators).
    su_dispatch : pd.DataFrame
        Concatenated storage unit dispatch (snapshots x storage units).
    link_dispatch : pd.DataFrame
        Concatenated link dispatch (snapshots x links).
    all_prices : pd.DataFrame
        Concatenated bus marginal prices (snapshots x buses).
    objective_total : float
        Sum of objectives across all windows.
    """
    window_hours = int(wholesale_config.get("window_hours", 24))
    carry_soc = wholesale_config.get("carry_soc", True)

    all_snapshots = network.snapshots.copy()
    n_total = len(all_snapshots)

    # Determine timestep resolution
    if n_total > 1:
        dt_hours = (all_snapshots[1] - all_snapshots[0]).total_seconds() / 3600
    else:
        dt_hours = 1.0
    steps_per_window = max(1, int(round(window_hours / dt_hours)))

    # Build list of window snapshot slices
    windows = []
    for start_idx in range(0, n_total, steps_per_window):
        end_idx = min(start_idx + steps_per_window, n_total)
        windows.append(all_snapshots[start_idx:end_idx])

    logger.info(
        f"Rolling day-ahead: {len(windows)} windows × {window_hours}h "
        f"({steps_per_window} timesteps each, {dt_hours}h resolution)"
    )

    # Build callback to fix interconnector flows at historical values
    ic_callback = _build_fix_interconnectors_callback(network, logger)
    large_hydro_state = initialise_large_hydro_storage_state(
        network, scenario_config.get("hydro", {})
    )
    log_hydro_constraint_setup(network, scenario_config, logger, context="Wholesale hydro")

    # Wholesale SoC floor for pumped hydro — preserves headroom for BM increases
    pumped_cfg = scenario_config.get("hydro", {}).get("pumped_hydro", {})
    ws_min_soc = pumped_cfg.get("wholesale_min_soc_fraction")
    if ws_min_soc is not None:
        logger.info(
            f"Wholesale pumped-hydro min SoC override: {100 * float(ws_min_soc):.0f}% "
            f"(BM uses {100 * float(pumped_cfg.get('min_soc_fraction', 0.15)):.0f}%)"
        )

    gen_dispatches = []
    su_dispatches = []
    link_dispatches = []
    price_dfs = []
    line_flow_dfs = []
    objective_total = 0.0
    prev_final_soc = None

    for i, window_snaps in enumerate(windows):
        window_start = window_snaps[0]
        window_end = window_snaps[-1]

        if (i + 1) % 30 == 0 or i == 0 or i == len(windows) - 1:
            logger.info(
                f"Window {i + 1}/{len(windows)}: "
                f"{window_start} → {window_end} ({len(window_snaps)} snapshots)"
            )

        # Create a copy restricted to this window
        day_net = network.copy()
        day_net.set_snapshots(window_snaps)

        # Carry over SoC from previous window
        if carry_soc and prev_final_soc is not None:
            for su_name, soc in prev_final_soc.items():
                if su_name in day_net.storage_units.index:
                    day_net.storage_units.loc[su_name, "state_of_charge_initial"] = soc

        hydro_callback = build_hydro_constraints_callback(
            day_net,
            scenario_config,
            large_hydro_storage_state=large_hydro_state,
            pumped_min_soc_override=ws_min_soc,
        )

        # Solve
        status, cond = day_net.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=combine_extra_functionalities(
                ic_callback, hydro_callback
            ),
        )
        if status != "ok":
            logger.error(
                f"Window {i + 1} ({window_start}) failed: {status} ({cond})"
            )
            raise RuntimeError(
                f"Wholesale window {i + 1} ({window_start}) failed: "
                f"{status} ({cond})"
            )

        objective_total += day_net.objective

        # Extract positions
        gen_d, su_d, link_d = extract_wholesale_positions(day_net, logger)
        gen_dispatches.append(gen_d)
        su_dispatches.append(su_d)
        link_dispatches.append(link_d)

        # Extract marginal prices
        if hasattr(day_net, "buses_t") and len(day_net.buses_t.marginal_price) > 0:
            price_dfs.append(day_net.buses_t.marginal_price.copy())

        # Extract line flows for congestion analysis
        if hasattr(day_net, "lines_t") and len(day_net.lines_t.p0) > 0:
            line_flow_dfs.append(day_net.lines_t.p0.copy())

        # Save final SoC for next window
        if carry_soc and len(day_net.storage_units) > 0:
            if (
                hasattr(day_net, "storage_units_t")
                and len(day_net.storage_units_t.state_of_charge) > 0
            ):
                prev_final_soc = (
                    day_net.storage_units_t.state_of_charge.iloc[-1].to_dict()
                )

        large_hydro_state = update_large_hydro_storage_state(
            day_net,
            window_snaps,
            scenario_config,
            large_hydro_state,
        )

    # Concatenate results
    gen_dispatch = pd.concat(gen_dispatches) if gen_dispatches else pd.DataFrame()
    su_dispatch = pd.concat(su_dispatches) if su_dispatches else pd.DataFrame()
    link_dispatch = pd.concat(link_dispatches) if link_dispatches else pd.DataFrame()
    all_prices = pd.concat(price_dfs) if price_dfs else pd.DataFrame()
    all_line_flows = pd.concat(line_flow_dfs) if line_flow_dfs else pd.DataFrame()

    # Write results back to the main network so the saved .nc file
    # contains solve outputs (generator dispatch, prices, line flows).
    if not gen_dispatch.empty:
        network.generators_t.p = gen_dispatch.reindex(
            columns=network.generators.index, fill_value=0.0
        )
    if not su_dispatch.empty:
        network.storage_units_t.p = su_dispatch.reindex(
            columns=network.storage_units.index, fill_value=0.0
        )
    if not all_prices.empty:
        network.buses_t.marginal_price = all_prices.reindex(
            columns=network.buses.index, fill_value=0.0
        )
    if not all_line_flows.empty:
        network.lines_t.p0 = all_line_flows.reindex(
            columns=network.lines.index, fill_value=0.0
        )

    logger.info(
        f"Rolling day-ahead complete: {len(windows)} windows, "
        f"total objective £{objective_total:,.2f}"
    )

    return gen_dispatch, su_dispatch, link_dispatch, all_prices, objective_total


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

        # ── Dispatch based on wholesale mode ─────────────────────────────
        wholesale_mode = wholesale_config.get("mode", "single")
        logger.info(f"Wholesale mode: {wholesale_mode}")

        solve_start = time.time()

        if wholesale_mode == "rolling_day_ahead":
            # ── Rolling day-ahead: solve in windows ──────────────────────
            logger.info("Starting rolling day-ahead wholesale optimization...")
            gen_dispatch, su_dispatch, link_dispatch, prices, objective = (
                solve_rolling_day_ahead(
                    network=network,
                    wholesale_config=wholesale_config,
                    solver_name=solver_name,
                    solver_options=solver_options,
                    scenario_config=scenario_config,
                    logger=logger,
                )
            )

            solve_time = time.time() - solve_start
            logger.info(f"Rolling day-ahead solve completed in {solve_time:.2f}s")
            logger.info(f"Total wholesale system cost: £{objective:,.2f}")

            # Save dispatch CSVs
            gen_dispatch.to_csv(snakemake.output.wholesale_dispatch_csv)
            logger.info(f"Saved: {snakemake.output.wholesale_dispatch_csv}")
            su_dispatch.to_csv(snakemake.output.wholesale_storage_csv)
            logger.info(f"Saved: {snakemake.output.wholesale_storage_csv}")
            link_dispatch.to_csv(snakemake.output.wholesale_links_csv)
            logger.info(f"Saved: {snakemake.output.wholesale_links_csv}")

            # Compute wholesale price from concatenated bus marginal prices
            if len(prices) > 0:
                load_buses = set(network.loads["bus"].unique())
                if "country" in network.buses.columns:
                    gb_buses = set(
                        network.buses.index[network.buses["country"] == "GB"]
                    )
                    demand_price_cols = [
                        c
                        for c in prices.columns
                        if c in load_buses and c in gb_buses
                    ]
                else:
                    demand_price_cols = [
                        c for c in prices.columns if c in load_buses
                    ]
                if demand_price_cols:
                    prices_demand = prices[demand_price_cols]
                else:
                    prices_demand = prices

                uniform_price = prices_demand.mean(axis=1)
                price_spread = prices_demand.max(axis=1) - prices_demand.min(axis=1)

                price_df = pd.DataFrame(
                    {
                        "wholesale_price": uniform_price,
                        "price_spread": price_spread,
                    }
                )
                price_df.to_csv(snakemake.output.wholesale_price_csv)
                logger.info(f"Saved: {snakemake.output.wholesale_price_csv}")
                logger.info(
                    f"Wholesale price: mean £{uniform_price.mean():.2f}/MWh, "
                    f"max spread £{price_spread.max():.4f}/MWh"
                )
            else:
                pd.DataFrame(
                    columns=["wholesale_price", "price_spread"]
                ).to_csv(snakemake.output.wholesale_price_csv)
                logger.warning(
                    "No marginal price data from rolling day-ahead solve"
                )

            # Save the last window's network as the wholesale network
            # (network object still has full snapshots — individual window
            # results are captured in the CSVs above.)
            save_network(network, snakemake.output.network, custom_logger=logger)
            logger.info(f"Saved wholesale network: {snakemake.output.network}")

        else:
            # ── Single solve (original behaviour) ────────────────────────
            logger.info("Starting wholesale market optimization...")

            ic_callback = _build_fix_interconnectors_callback(network, logger)
            log_hydro_constraint_setup(
                network,
                scenario_config,
                logger,
                context="Wholesale hydro",
            )
            # Apply wholesale SoC floor for pumped hydro
            pumped_cfg_single = scenario_config.get("hydro", {}).get("pumped_hydro", {})
            ws_min_soc_single = pumped_cfg_single.get("wholesale_min_soc_fraction")
            hydro_callback = build_hydro_constraints_callback(
                network,
                scenario_config,
                pumped_min_soc_override=ws_min_soc_single,
            )

            status, termination_condition = network.optimize(
                solver_name=solver_name,
                solver_options=solver_options,
                extra_functionality=combine_extra_functionalities(
                    ic_callback, hydro_callback
                ),
            )

            solve_time = time.time() - solve_start
            logger.info(f"Wholesale solve completed in {solve_time:.2f}s")
            logger.info(f"Status: {status}, Condition: {termination_condition}")

            if status != "ok":
                raise RuntimeError(
                    f"Wholesale optimization failed: "
                    f"{status} ({termination_condition})"
                )

            if hasattr(network, "objective"):
                logger.info(
                    f"Total wholesale system cost: £{network.objective:,.2f}"
                )

            # Extract and export wholesale positions
            gen_dispatch, su_dispatch, link_dispatch = (
                extract_wholesale_positions(network, logger)
            )

            gen_dispatch.to_csv(snakemake.output.wholesale_dispatch_csv)
            logger.info(f"Saved: {snakemake.output.wholesale_dispatch_csv}")
            su_dispatch.to_csv(snakemake.output.wholesale_storage_csv)
            logger.info(f"Saved: {snakemake.output.wholesale_storage_csv}")
            link_dispatch.to_csv(snakemake.output.wholesale_links_csv)
            logger.info(f"Saved: {snakemake.output.wholesale_links_csv}")

            # Wholesale price
            if (
                hasattr(network, "buses_t")
                and len(network.buses_t.marginal_price) > 0
            ):
                prices = network.buses_t.marginal_price
                load_buses = set(network.loads["bus"].unique())
                if "country" in network.buses.columns:
                    gb_buses = set(
                        network.buses.index[network.buses["country"] == "GB"]
                    )
                    demand_price_cols = [
                        c
                        for c in prices.columns
                        if c in load_buses and c in gb_buses
                    ]
                    n_excluded = len(
                        [
                            c
                            for c in prices.columns
                            if c in load_buses and c not in gb_buses
                        ]
                    )
                    if n_excluded:
                        logger.info(
                            f"Excluded {n_excluded} non-GB demand buses from "
                            f"price calculation"
                        )
                else:
                    demand_price_cols = [
                        c for c in prices.columns if c in load_buses
                    ]
                if demand_price_cols:
                    prices_demand = prices[demand_price_cols]
                    logger.info(
                        f"Wholesale price computed from "
                        f"{len(demand_price_cols)} GB demand buses "
                        f"(of {len(prices.columns)} total)"
                    )
                else:
                    prices_demand = prices
                    logger.warning(
                        "No GB demand buses found in marginal_price; "
                        "using all buses"
                    )

                uniform_price = prices_demand.mean(axis=1)
                price_spread = (
                    prices_demand.max(axis=1) - prices_demand.min(axis=1)
                )
                max_spread = price_spread.max()

                price_df = pd.DataFrame(
                    {
                        "wholesale_price": uniform_price,
                        "price_spread": price_spread,
                    }
                )
                price_df.to_csv(snakemake.output.wholesale_price_csv)
                logger.info(f"Saved: {snakemake.output.wholesale_price_csv}")
                logger.info(
                    f"Wholesale price: mean £{uniform_price.mean():.2f}/MWh, "
                    f"max spread £{max_spread:.4f}/MWh "
                    f"(should be ~0 for copperplate)"
                )
            else:
                pd.DataFrame(
                    columns=["wholesale_price", "price_spread"]
                ).to_csv(snakemake.output.wholesale_price_csv)
                logger.warning(
                    "No marginal price data available from wholesale solve"
                )

            # Save solved wholesale network
            save_network(
                network, snakemake.output.network, custom_logger=logger
            )
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
