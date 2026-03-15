"""
Solve Balancing Mechanism — Stage 2 of Two-Stage Market Dispatch

Starting from the wholesale (copperplate) dispatch positions, re-solves the
network with full transmission constraints. Generators and storage are
anchored to their wholesale positions via increase/decrease variables with
separate bid/offer prices, mimicking the GB Balancing Mechanism.

The optimisation minimises total redispatch cost:
    min  Σ_{g,t}  offer_price[g] · increase[g,t] + bid_price[g] · decrease[g,t]

Subject to:
    p[g,t] = p_wholesale[g,t] + increase[g,t] - decrease[g,t]   ∀ g,t
    increase[g,t] ≥ 0,  decrease[g,t] ≥ 0                        ∀ g,t
    All standard PyPSA constraints (power balance, line limits, gen limits, ...)

Inputs:
  - Finalized network: {scenario}.nc (with original transmission constraints)
  - Wholesale dispatch CSVs from Stage 1

Outputs:
  - Solved balancing network: {scenario}_balancing.nc
  - Balancing dispatch CSV: {scenario}_balancing_dispatch.csv
  - Redispatch summary CSV: {scenario}_redispatch_summary.csv
  - Constraint costs CSV: {scenario}_constraint_costs.csv
  - Congestion CSV: {scenario}_congestion.csv
  - Price comparison CSV: {scenario}_price_comparison.csv

See Also:
  - scripts/market/solve_wholesale.py — Stage 1 (copperplate)
  - scripts/market/market_utils.py — Shared utilities
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
    apply_transmission_relaxation,
    improve_numerical_conditioning,
    configure_solver,
    get_solve_mode_from_config,
)
from scripts.market.market_utils import (
    calculate_bid_offer_prices,
    compute_redispatch_volumes,
    identify_congested_boundaries,
)
from scripts.market.solve_wholesale import apply_solve_period


def _build_balancing_extra_functionality(
    wholesale_gen: pd.DataFrame,
    wholesale_su: pd.DataFrame,
    wholesale_links: pd.DataFrame,
    gen_offer_prices: pd.Series,
    gen_bid_prices: pd.Series,
    su_offer_prices: pd.Series,
    su_bid_prices: pd.Series,
    fix_interconnectors: bool,
    logger: logging.Logger,
):
    """
    Build the ``extra_functionality`` callback for PyPSA's ``network.optimize()``.

    The callback injects:
      1. Increase/decrease variables for generators and storage units
      2. Linking constraints:  p[g,t] == p_wholesale[g,t] + inc[g,t] - dec[g,t]
      3. A new objective:  min Σ offer·inc + bid·dec
      4. (Optional) Fix interconnector flows to wholesale values

    Parameters
    ----------
    wholesale_gen : pd.DataFrame
        Wholesale generator dispatch (snapshots × generators).
    wholesale_su : pd.DataFrame
        Wholesale storage dispatch (snapshots × storage units).
    wholesale_links : pd.DataFrame
        Wholesale link dispatch (snapshots × links).
    gen_offer_prices, gen_bid_prices : pd.Series
        Per-generator offer/bid prices.
    su_offer_prices, su_bid_prices : pd.Series
        Per-storage-unit offer/bid prices.
    fix_interconnectors : bool
        If True, fix Link flows to wholesale values.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    callable
        Function with signature ``extra_functionality(network, snapshots)``
        suitable for ``network.optimize(extra_functionality=...)``.
    """

    def extra_functionality(network, snapshots):
        """Inject BM redispatch variables, constraints, and objective."""
        import linopy
        import xarray as xr

        model = network.model
        n_snapshots = len(snapshots)

        logger.info("Injecting BM extra_functionality into optimisation model...")

        # Helper to create a zero lower-bound DataArray with correct dims.
        # PyPSA's linopy model uses ('snapshot', 'name') for all component
        # variables, so custom variables must match to allow arithmetic.
        def _zero_lower(sns, names):
            idx = pd.Index(names, name="name")
            return xr.DataArray(
                0, dims=["snapshot", "name"],
                coords={"snapshot": sns, "name": idx},
            )

        def _make_upper(sns, names, arr):
            idx = pd.Index(names, name="name")
            return xr.DataArray(
                arr, dims=["snapshot", "name"],
                coords={"snapshot": sns, "name": idx},
            )

        # ── 1. Generator increase/decrease variables ─────────────────────
        gen_names = [g for g in wholesale_gen.columns if g in network.generators.index]
        if gen_names:
            lower_gen = _zero_lower(snapshots, gen_names)

            # Upper bounds prevent the LP going unbounded when offer prices are
            # negative (e.g. ROC generators with negative marginal costs).
            # increase ≤ p_max[g,t] - p_wholesale[g,t]  (headroom to ceiling)
            # decrease ≤ p_wholesale[g,t] - p_min[g,t]  (headroom to floor)
            p_nom_gen = network.generators.loc[gen_names, "p_nom"].values  # (G,)
            p_max_pu_static = network.generators.loc[gen_names, "p_max_pu"].values
            p_min_pu_static = network.generators.loc[gen_names, "p_min_pu"].values

            p_max_arr = np.outer(np.ones(n_snapshots), p_nom_gen * p_max_pu_static)
            p_min_arr = np.outer(np.ones(n_snapshots), p_nom_gen * p_min_pu_static)

            # Apply time-varying p_max_pu where present (renewables).
            if not network.generators_t.p_max_pu.empty:
                tv_cols = [g for g in gen_names
                           if g in network.generators_t.p_max_pu.columns]
                if tv_cols:
                    tv_idx = [gen_names.index(g) for g in tv_cols]
                    tv_vals = (
                        network.generators_t.p_max_pu[tv_cols]
                        .reindex(snapshots)
                        .fillna(1.0)
                        .values
                    ) * p_nom_gen[tv_idx][np.newaxis, :]
                    p_max_arr[:, tv_idx] = tv_vals

            ws_gen_arr = wholesale_gen[gen_names].reindex(snapshots).values
            upper_inc_arr = np.maximum(0.0, p_max_arr - ws_gen_arr)
            upper_dec_arr = np.maximum(0.0, ws_gen_arr - p_min_arr)

            model.add_variables(
                lower=lower_gen,
                upper=_make_upper(snapshots, gen_names, upper_inc_arr),
                name="Generator-increase",
            )
            model.add_variables(
                lower=lower_gen,
                upper=_make_upper(snapshots, gen_names, upper_dec_arr),
                name="Generator-decrease",
            )

            # Linking constraint: p == p_wholesale + increase - decrease
            gen_p = model.variables["Generator-p"].sel({"name": gen_names})
            gen_inc = model.variables["Generator-increase"]
            gen_dec = model.variables["Generator-decrease"]

            lhs = gen_p - gen_inc + gen_dec
            model.add_constraints(
                lhs == ws_gen_arr,
                name="Generator-bm_anchor",
            )

            logger.info(
                f"Added BM variables + anchor constraints for {len(gen_names)} generators"
            )

        # ── 2. Storage unit increase/decrease variables ──────────────────
        su_names = [s for s in wholesale_su.columns if s in network.storage_units.index]
        if su_names:
            lower_su = _zero_lower(snapshots, su_names)

            # Upper bounds: net dispatch can swing ±p_nom from wholesale position.
            # increase ≤ p_nom - ws_su  (headroom above current net dispatch)
            # decrease ≤ ws_su + p_nom  (headroom below current net dispatch)
            p_nom_su = network.storage_units.loc[su_names, "p_nom"].values  # (S,)
            ws_su_arr = wholesale_su[su_names].reindex(snapshots).values    # (T, S)
            su_upper_inc_arr = np.maximum(0.0, p_nom_su[np.newaxis, :] - ws_su_arr)
            su_upper_dec_arr = np.maximum(0.0, ws_su_arr + p_nom_su[np.newaxis, :])

            model.add_variables(
                lower=lower_su,
                upper=_make_upper(snapshots, su_names, su_upper_inc_arr),
                name="StorageUnit-increase",
            )
            model.add_variables(
                lower=lower_su,
                upper=_make_upper(snapshots, su_names, su_upper_dec_arr),
                name="StorageUnit-decrease",
            )

            su_p = model.variables["StorageUnit-p_dispatch"].sel({"name": su_names})
            su_store = model.variables["StorageUnit-p_store"].sel({"name": su_names})
            su_inc = model.variables["StorageUnit-increase"]
            su_dec = model.variables["StorageUnit-decrease"]

            # Storage net dispatch = p_dispatch - p_store
            # Anchor: (p_dispatch - p_store) == p_wholesale + increase - decrease
            lhs = su_p - su_store - su_inc + su_dec
            model.add_constraints(
                lhs == ws_su_arr,
                name="StorageUnit-bm_anchor",
            )

            logger.info(
                f"Added BM variables + anchor constraints for {len(su_names)} storage units"
            )

        # ── 3. Fix interconnector flows (if configured) ──────────────────
        # Only fix actual cross-border interconnectors, NOT internal links
        # (e.g., H2 turbines, electrolysis, internal HVDC).
        # Interconnectors are identified by having a bus connected to an
        # external/foreign node (bus name containing "External").
        if fix_interconnectors and len(wholesale_links.columns) > 0:
            # Identify actual cross-border interconnectors
            ic_names = []
            for lk in wholesale_links.columns:
                if lk not in network.links.index:
                    continue
                bus0 = network.links.loc[lk, "bus0"]
                bus1 = network.links.loc[lk, "bus1"]
                if "External" in str(bus0) or "External" in str(bus1):
                    ic_names.append(lk)

            if ic_names and "Link-p" in model.variables:
                link_p = model.variables["Link-p"].sel({"name": ic_names})
                ws_link_vals = wholesale_links[ic_names].reindex(snapshots).values

                model.add_constraints(
                    link_p == ws_link_vals,
                    name="Link-bm_fix_interconnector",
                )
                logger.info(
                    f"Fixed {len(ic_names)} cross-border interconnectors at "
                    f"wholesale positions (skipped "
                    f"{len(wholesale_links.columns) - len(ic_names)} internal links)"
                )
            else:
                logger.info("No cross-border interconnectors found to fix")

        # ── 4. Replace objective with BM redispatch cost ─────────────────
        # Build objective expression: min Σ offer·inc + bid·dec
        obj_expr = None

        if gen_names:
            gen_inc = model.variables["Generator-increase"]
            gen_dec = model.variables["Generator-decrease"]

            # Build offer/bid coefficient arrays (snapshots × generators)
            offer_coeffs = np.tile(
                gen_offer_prices.reindex(gen_names).values, (n_snapshots, 1)
            )
            bid_coeffs = np.tile(
                gen_bid_prices.reindex(gen_names).values, (n_snapshots, 1)
            )

            gen_obj = (gen_inc * offer_coeffs).sum() + (gen_dec * bid_coeffs).sum()
            obj_expr = gen_obj

        if su_names:
            su_inc = model.variables["StorageUnit-increase"]
            su_dec = model.variables["StorageUnit-decrease"]

            su_offer_coeffs = np.tile(
                su_offer_prices.reindex(su_names).values, (n_snapshots, 1)
            )
            su_bid_coeffs = np.tile(
                su_bid_prices.reindex(su_names).values, (n_snapshots, 1)
            )

            su_obj = (su_inc * su_offer_coeffs).sum() + (su_dec * su_bid_coeffs).sum()
            obj_expr = obj_expr + su_obj if obj_expr is not None else su_obj

        if obj_expr is not None:
            model.objective = obj_expr
            logger.info("Replaced objective with BM redispatch cost minimisation")
        else:
            logger.warning(
                "No generators or storage units matched wholesale positions — "
                "objective unchanged"
            )

    return extra_functionality


if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────
    log_path = (
        snakemake.log[0]
        if hasattr(snakemake, "log") and snakemake.log
        else "solve_balancing"
    )
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("STAGE 2: SOLVING BALANCING MECHANISM (ANCHORED REDISPATCH)")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # ── Load finalized network (original transmission constraints) ───
        input_path = snakemake.input.network
        logger.info(f"Loading finalized network from: {input_path}")
        network = load_network(input_path, custom_logger=logger)

        scenario_config = snakemake.params.scenario_config
        market_config = scenario_config.get("market", {})
        balancing_config = market_config.get("balancing", {})
        solver_name = snakemake.params.solver
        solver_options = snakemake.params.solver_options
        scenario_id = scenario_config.get("scenario_id", snakemake.wildcards.scenario)

        # Override ELEXON paths when Snakemake provides them as inputs
        if hasattr(snakemake.input, 'bmu_mapping'):
            market_config.setdefault("balancing", {}).setdefault("elexon", {})
            market_config["balancing"]["elexon"]["bmu_mapping"] = snakemake.input.bmu_mapping
        if hasattr(snakemake.input, 'elexon_offers'):
            elexon_dir = str(Path(snakemake.input.elexon_offers).parent)
            market_config.setdefault("balancing", {}).setdefault("elexon", {})
            market_config["balancing"]["elexon"]["data_dir"] = elexon_dir

        logger.info(f"Scenario: {scenario_id}")
        logger.info(
            f"Network: {len(network.buses)} buses, "
            f"{len(network.generators)} generators, "
            f"{len(network.storage_units)} storage units, "
            f"{len(network.links)} links, "
            f"{len(network.lines)} lines"
        )

        # ── Load wholesale dispatch from Stage 1 ────────────────────────
        logger.info("Loading wholesale dispatch positions from Stage 1...")
        wholesale_gen = pd.read_csv(
            snakemake.input.wholesale_dispatch_csv, index_col=0, parse_dates=True
        )
        wholesale_su = pd.read_csv(
            snakemake.input.wholesale_storage_csv, index_col=0, parse_dates=True
        )
        wholesale_links = pd.read_csv(
            snakemake.input.wholesale_links_csv, index_col=0, parse_dates=True
        )

        logger.info(
            f"Wholesale positions loaded: "
            f"{wholesale_gen.shape[1]} generators, "
            f"{wholesale_su.shape[1]} storage units, "
            f"{wholesale_links.shape[1]} links"
        )

        # Load wholesale price for comparison
        wholesale_price = pd.read_csv(
            snakemake.input.wholesale_price_csv, index_col=0, parse_dates=True
        )

        # ── Pre-solve steps ──────────────────────────────────────────────
        validate_network_costs(network, logger)

        # Apply standard transmission relaxation if configured (floors/scales)
        # These are the real physics-based relaxations, NOT copperplate
        apply_transmission_relaxation(network, scenario_config, logger)

        improve_numerical_conditioning(network, logger)
        apply_solve_period(network, scenario_config, logger)

        # ── Solve mode ───────────────────────────────────────────────────
        global_solve_mode = get_solve_mode_from_config()
        logger.info(f"Solve mode: {global_solve_mode}")

        if global_solve_mode == "LP":
            if "committable" in network.generators.columns:
                network.generators["committable"] = False
            logger.info("LP mode: unit commitment disabled")

        remove_must_run = scenario_config.get("optimization", {}).get(
            "remove_must_run", False
        )
        if remove_must_run and "p_min_pu" in network.generators.columns:
            network.generators["p_min_pu"] = 0.0
            logger.info("Removed must-run constraints (p_min_pu = 0)")

        # ── Calculate bid/offer prices ───────────────────────────────────
        logger.info("=" * 80)
        logger.info("CALCULATING BID/OFFER PRICES")
        logger.info("=" * 80)
        gen_offer, gen_bid, su_offer, su_bid = calculate_bid_offer_prices(
            network, market_config, logger, scenario_id=scenario_id
        )

        # ── Build extra_functionality callback ───────────────────────────
        fix_ics = balancing_config.get("fix_interconnectors", True)
        extra_func = _build_balancing_extra_functionality(
            wholesale_gen=wholesale_gen,
            wholesale_su=wholesale_su,
            wholesale_links=wholesale_links,
            gen_offer_prices=gen_offer,
            gen_bid_prices=gen_bid,
            su_offer_prices=su_offer,
            su_bid_prices=su_bid,
            fix_interconnectors=fix_ics,
            logger=logger,
        )

        # ── Configure solver ─────────────────────────────────────────────
        solver_name, solver_options = configure_solver(
            network, solver_name, solver_options, logger
        )

        # ── Optimize ─────────────────────────────────────────────────────
        logger.info("Starting BM optimization (full network constraints)...")
        solve_start = time.time()

        status, termination_condition = network.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_func,
        )

        solve_time = time.time() - solve_start
        logger.info(f"BM solve completed in {solve_time:.2f}s")
        logger.info(f"Status: {status}, Condition: {termination_condition}")

        if status != "ok":
            raise RuntimeError(
                f"BM optimization failed: {status} ({termination_condition})"
            )

        bm_cost = network.objective
        logger.info(f"Total BM redispatch cost: £{bm_cost:,.2f}")

        # ── Export BM dispatch ───────────────────────────────────────────
        physical_gen = network.generators_t.p.copy()
        physical_gen.to_csv(snakemake.output.balancing_dispatch_csv)
        logger.info(f"Saved: {snakemake.output.balancing_dispatch_csv}")

        physical_su = network.storage_units_t.p.copy() if len(network.storage_units_t.p) > 0 else pd.DataFrame(index=network.snapshots)

        # ── Compute redispatch summary ───────────────────────────────────
        logger.info("=" * 80)
        logger.info("COMPUTING REDISPATCH VOLUMES AND COSTS")
        logger.info("=" * 80)

        # Align wholesale dispatch to current snapshots
        ws_gen_aligned = wholesale_gen.reindex(network.snapshots).reindex(
            columns=physical_gen.columns, fill_value=0.0
        )
        ws_su_aligned = wholesale_su.reindex(network.snapshots).reindex(
            columns=physical_su.columns, fill_value=0.0
        ) if len(physical_su.columns) > 0 else pd.DataFrame(index=network.snapshots)

        gen_summary, su_summary = compute_redispatch_volumes(
            wholesale_gen=ws_gen_aligned,
            physical_gen=physical_gen,
            wholesale_su=ws_su_aligned,
            physical_su=physical_su,
            gen_offer_prices=gen_offer,
            gen_bid_prices=gen_bid,
            su_offer_prices=su_offer,
            su_bid_prices=su_bid,
            network=network,
            logger=logger,
        )

        # Combine and save redispatch summary
        redispatch_summary = pd.concat([gen_summary, su_summary], ignore_index=True)
        redispatch_summary.to_csv(snakemake.output.redispatch_summary_csv, index=False)
        logger.info(f"Saved: {snakemake.output.redispatch_summary_csv}")

        # ── Constraint costs by carrier and region ───────────────────────
        if not redispatch_summary.empty:
            cost_by_carrier = (
                redispatch_summary.groupby("carrier")
                .agg(
                    offer_cost=("offer_cost", "sum"),
                    bid_cost=("bid_cost", "sum"),
                    net_cost=("net_cost", "sum"),
                    increase_MWh=("increase_MWh", "sum"),
                    decrease_MWh=("decrease_MWh", "sum"),
                )
                .sort_values("net_cost", ascending=False)
            )
            cost_by_carrier.loc["TOTAL"] = cost_by_carrier.sum()
            cost_by_carrier.to_csv(snakemake.output.constraint_costs_csv)
            logger.info(f"Saved: {snakemake.output.constraint_costs_csv}")

            total_offer = cost_by_carrier.loc["TOTAL", "offer_cost"]
            total_bid = cost_by_carrier.loc["TOTAL", "bid_cost"]
            total_net = cost_by_carrier.loc["TOTAL", "net_cost"]
            logger.info(
                f"Constraint costs: offers=£{total_offer:,.0f}, "
                f"bids=£{total_bid:,.0f}, total=£{total_net:,.0f}"
            )
        else:
            pd.DataFrame().to_csv(snakemake.output.constraint_costs_csv)
            logger.warning("No redispatch — zero constraint costs")

        # ── Congestion analysis ──────────────────────────────────────────
        congestion = identify_congested_boundaries(network, threshold=0.95, logger=logger)
        congestion.to_csv(snakemake.output.congestion_csv, index=False)
        logger.info(f"Saved: {snakemake.output.congestion_csv}")

        # ── Price comparison ─────────────────────────────────────────────
        if len(network.buses_t.marginal_price) > 0:
            nodal_prices = network.buses_t.marginal_price

            # Filter to GB-only demand (load) buses.
            # Exclude non-GB external buses (HVDC_External_*) that carry
            # EU_demand loads — these have extreme shadow prices when
            # historical p_set forces GB exports to an unsinkable node.
            load_buses = set(network.loads["bus"].unique())
            if "country" in network.buses.columns:
                gb_buses = set(network.buses.index[network.buses["country"] == "GB"])
                demand_cols = [
                    c for c in nodal_prices.columns
                    if c in load_buses and c in gb_buses
                ]
            else:
                demand_cols = [c for c in nodal_prices.columns if c in load_buses]
            if demand_cols:
                nodal_prices_demand = nodal_prices[demand_cols]
                logger.info(
                    f"BM nodal price computed from {len(demand_cols)} GB demand buses "
                    f"(of {len(nodal_prices.columns)} total)"
                )
            else:
                nodal_prices_demand = nodal_prices
                logger.warning(
                    "No GB demand buses found in BM marginal_price; "
                    "using all buses (spread may be inflated)"
                )

            mean_nodal = nodal_prices_demand.mean(axis=1)
            price_spread = nodal_prices_demand.max(axis=1) - nodal_prices_demand.min(axis=1)

            ws_price_aligned = wholesale_price["wholesale_price"].reindex(
                network.snapshots, fill_value=np.nan
            )

            price_comparison = pd.DataFrame({
                "wholesale_price": ws_price_aligned,
                "mean_nodal_price": mean_nodal,
                "min_nodal_price": nodal_prices_demand.min(axis=1),
                "max_nodal_price": nodal_prices_demand.max(axis=1),
                "nodal_spread": price_spread,
            })
            price_comparison.to_csv(snakemake.output.price_comparison_csv)
            logger.info(f"Saved: {snakemake.output.price_comparison_csv}")
            logger.info(
                f"Price comparison: wholesale mean £{ws_price_aligned.mean():.2f}/MWh, "
                f"nodal mean £{mean_nodal.mean():.2f}/MWh, "
                f"max spread £{price_spread.max():.2f}/MWh"
            )
        else:
            pd.DataFrame(
                columns=[
                    "wholesale_price",
                    "mean_nodal_price",
                    "min_nodal_price",
                    "max_nodal_price",
                    "nodal_spread",
                ]
            ).to_csv(snakemake.output.price_comparison_csv)
            logger.warning("No marginal price data from BM solve")

        # ── Save solved BM network ───────────────────────────────────────
        save_network(network, snakemake.output.network, custom_logger=logger)
        logger.info(f"Saved BM network: {snakemake.output.network}")

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(
            f"STAGE 2 COMPLETE — BALANCING MECHANISM SOLVED "
            f"(Total: {total_time:.2f}s, BM cost: £{bm_cost:,.2f})"
        )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"FATAL ERROR in BM solve: {e}", exc_info=True)
        raise
