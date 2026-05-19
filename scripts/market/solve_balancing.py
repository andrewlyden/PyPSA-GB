"""
Solve Balancing Mechanism — Stage 2 of Two-Stage Market Dispatch

Starting from the wholesale (copperplate) dispatch positions, re-solves the
network with full transmission constraints. Generators and storage are
anchored to their wholesale positions via increase/decrease variables with
separate bid/offer prices, mimicking the GB Balancing Mechanism.

The optimisation minimises total redispatch cost:
    min  Σ_{g,t}  offer_price[g] · increase[g,t] + bid_price[g] · decrease[g,t]

where bid_price is always in "ESO cost" convention: positive means it costs
the ESO money to turn the generator down.  ELEXON raw bids (negative = ESO
pays generator) are negated at load time in market_utils._load_elexon_bid_offer.

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
  - Balancing storage CSV: {scenario}_balancing_storage.csv
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
    apply_load_shedding_limits,
    apply_transmission_relaxation,
    apply_line_rating_overrides,
    apply_outage_schedule,
    _build_neso_boundary_constraints_callback,
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
from scripts.solve.hydro_constraints import (
    build_hydro_constraints_callback,
    combine_extra_functionalities,
    initialise_large_hydro_storage_state,
    log_hydro_constraint_setup,
    update_large_hydro_storage_state,
)


def _clear_runtime_bm_attrs(network):
    """Remove transient BM helper attributes before exporting the network."""
    for attr in [
        "_bm_offer_ladders",
        "_bm_bid_ladders",
        "_bm_ladder_fallback_volume_mw",
        "_bm_ladder_missing_hour_fallback",
    ]:
        if hasattr(network, attr):
            delattr(network, attr)


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
    gen_offer_tv: pd.DataFrame = None,
    gen_bid_tv: pd.DataFrame = None,
    su_offer_tv: pd.DataFrame = None,
    su_bid_tv: pd.DataFrame = None,
    gen_offer_ladders: pd.DataFrame = None,
    gen_bid_ladders: pd.DataFrame = None,
    ladder_fallback_volume_mw: float = 1.0e6,
    ladder_missing_hour_fallback: bool = True,
    participating_generators=None,
    fixed_generators=None,
    participating_storage_units=None,
    fixed_storage_units=None,
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

        # Helpers to create DataArrays with the same component dimension name
        # as PyPSA/linopy uses for each variable (e.g. Generator, StorageUnit).
        def _component_dim(variable_name):
            dims = [dim for dim in model.variables[variable_name].dims if dim != "snapshot"]
            return dims[0] if dims else "name"

        def _zero_lower(sns, names, dim_name="name"):
            idx = pd.Index(names, name=dim_name)
            return xr.DataArray(
                0, dims=["snapshot", dim_name],
                coords={"snapshot": sns, dim_name: idx},
            )

        def _make_upper(sns, names, arr, dim_name="name"):
            idx = pd.Index(names, name=dim_name)
            return xr.DataArray(
                arr, dims=["snapshot", dim_name],
                coords={"snapshot": sns, dim_name: idx},
            )

        def _build_ladder_arrays(ladders, sns, names, fallback_prices, side_label):
            """Return ladder generator names plus price/volume arrays."""
            if ladders is None or ladders.empty or not names:
                return [], None, None

            work = ladders[
                ladders["generator"].isin(names)
                & ladders["snapshot"].isin(pd.Index(sns))
            ].copy()
            if work.empty:
                return [], None, None

            ladder_names = [
                name for name in names if name in set(work["generator"].unique())
            ]
            max_block = max(1, int(work["block"].max()))
            name_pos = {name: i for i, name in enumerate(ladder_names)}
            snap_pos = {snap: i for i, snap in enumerate(pd.Index(sns))}

            prices = np.zeros((len(sns), len(ladder_names), max_block), dtype=float)
            volumes = np.zeros_like(prices)
            has_ladder = np.zeros((len(sns), len(ladder_names)), dtype=bool)

            for row in work.itertuples(index=False):
                snap_i = snap_pos.get(row.snapshot)
                name_i = name_pos.get(row.generator)
                if snap_i is None or name_i is None:
                    continue
                block_i = int(row.block) - 1
                if block_i < 0 or block_i >= max_block:
                    continue
                prices[snap_i, name_i, block_i] = float(row.price)
                volumes[snap_i, name_i, block_i] += float(row.volume_mw)
                has_ladder[snap_i, name_i] = True

            if ladder_missing_hour_fallback:
                fallback = fallback_prices.reindex(ladder_names).fillna(0.0).values
                missing = ~has_ladder
                for snap_i, name_i in zip(*np.where(missing)):
                    prices[snap_i, name_i, 0] = fallback[name_i]
                    volumes[snap_i, name_i, 0] = ladder_fallback_volume_mw

            active = volumes.sum(axis=(0, 2)) > 0
            if not active.any():
                return [], None, None
            ladder_names = [name for name, keep in zip(ladder_names, active) if keep]
            prices = prices[:, active, :]
            volumes = volumes[:, active, :]

            logger.info(
                f"Using {side_label} price ladders for {len(ladder_names)} generators "
                f"({max_block} blocks max)"
            )
            return ladder_names, prices, volumes

        # ── 1. Generator increase/decrease variables ─────────────────────
        # Generator participation is configurable. Non-participants can either
        # be priced out upstream or fixed at wholesale position here.
        all_gen_names = [g for g in wholesale_gen.columns if g in network.generators.index]
        participant_gen_set = set(
            all_gen_names if participating_generators is None else participating_generators
        )
        fixed_gen_set = set([] if fixed_generators is None else fixed_generators)
        gen_names = [g for g in all_gen_names if g in participant_gen_set]
        fixed_gen_names = [
            g for g in all_gen_names if g in fixed_gen_set and g not in participant_gen_set
        ]
        gen_dim = _component_dim("Generator-p") if all_gen_names else "name"

        if gen_names:
            lower_gen = _zero_lower(snapshots, gen_names, gen_dim)

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
                upper=_make_upper(snapshots, gen_names, upper_inc_arr, gen_dim),
                name="Generator-increase",
            )
            model.add_variables(
                lower=lower_gen,
                upper=_make_upper(snapshots, gen_names, upper_dec_arr, gen_dim),
                name="Generator-decrease",
            )

            # Linking constraint: p == p_wholesale + increase - decrease
            gen_p = model.variables["Generator-p"].sel({gen_dim: gen_names})
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
        if fixed_gen_names:
            gen_p_fixed = model.variables["Generator-p"].sel({gen_dim: fixed_gen_names})
            ws_gen_fixed = wholesale_gen[fixed_gen_names].reindex(snapshots).values
            model.add_constraints(
                gen_p_fixed == ws_gen_fixed,
                name="Generator-bm_fix_nonparticipant",
            )
            logger.info(
                f"Fixed {len(fixed_gen_names)} non-participating generators "
                "at wholesale positions"
            )

        # ── 2. Storage unit increase/decrease variables ──────────────────
        all_su_names = [s for s in wholesale_su.columns if s in network.storage_units.index]
        participant_su_set = set(
            all_su_names
            if participating_storage_units is None else participating_storage_units
        )
        fixed_su_set = set([] if fixed_storage_units is None else fixed_storage_units)
        su_names = [s for s in all_su_names if s in participant_su_set]
        fixed_su_names = [
            s for s in all_su_names if s in fixed_su_set and s not in participant_su_set
        ]
        su_dim = _component_dim("StorageUnit-p_dispatch") if all_su_names else "name"
        if su_names:
            lower_su = _zero_lower(snapshots, su_names, su_dim)

            # Upper bounds: net dispatch can swing ±p_nom from wholesale position.
            # increase ≤ p_nom - ws_su  (headroom above current net dispatch)
            # decrease ≤ ws_su + p_nom  (headroom below current net dispatch)
            p_nom_su = network.storage_units.loc[su_names, "p_nom"].values  # (S,)
            ws_su_arr = wholesale_su[su_names].reindex(snapshots).values    # (T, S)
            su_upper_inc_arr = np.maximum(0.0, p_nom_su[np.newaxis, :] - ws_su_arr)
            su_upper_dec_arr = np.maximum(0.0, ws_su_arr + p_nom_su[np.newaxis, :])

            model.add_variables(
                lower=lower_su,
                upper=_make_upper(snapshots, su_names, su_upper_inc_arr, su_dim),
                name="StorageUnit-increase",
            )
            model.add_variables(
                lower=lower_su,
                upper=_make_upper(snapshots, su_names, su_upper_dec_arr, su_dim),
                name="StorageUnit-decrease",
            )

            su_p = model.variables["StorageUnit-p_dispatch"].sel({su_dim: su_names})
            su_store = model.variables["StorageUnit-p_store"].sel({su_dim: su_names})
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
        if fixed_su_names:
            su_p_fixed = model.variables["StorageUnit-p_dispatch"].sel({su_dim: fixed_su_names})
            su_store_fixed = model.variables["StorageUnit-p_store"].sel({su_dim: fixed_su_names})
            ws_su_fixed = wholesale_su[fixed_su_names].reindex(snapshots).values
            model.add_constraints(
                su_p_fixed - su_store_fixed == ws_su_fixed,
                name="StorageUnit-bm_fix_nonparticipant",
            )
            logger.info(
                f"Fixed {len(fixed_su_names)} non-participating storage units "
                "at wholesale positions"
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
                link_dim = _component_dim("Link-p")
                link_p = model.variables["Link-p"].sel({link_dim: ic_names})
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
        # Build objective expression: min Σ hours·(offer·inc + bid·dec)
        obj_expr = None
        snapshot_hours = pd.Series(1.0, index=pd.Index(snapshots))
        if len(snapshot_hours) > 1:
            inferred_hours = (
                pd.Index(snapshots)
                .to_series()
                .diff()
                .dt.total_seconds()
                .dropna()
                .median()
                / 3600
            )
            if pd.notna(inferred_hours) and inferred_hours > 0:
                snapshot_hours.iloc[:] = float(inferred_hours)
        snapshot_hours_da = xr.DataArray(
            snapshot_hours.values,
            dims=["snapshot"],
            coords={"snapshot": snapshots},
        )

        if gen_names:
            gen_inc = model.variables["Generator-increase"]
            gen_dec = model.variables["Generator-decrease"]
            gen_obj_parts = []

            # Build offer/bid coefficient arrays (snapshots × generators).
            # Use per-snapshot time-varying prices if available (ELEXON),
            # otherwise broadcast static prices across all snapshots.
            if gen_offer_tv is not None and gen_bid_tv is not None:
                offer_coeffs = (
                    gen_offer_tv.reindex(index=snapshots, columns=gen_names)
                    .fillna(0.0).values
                )
                bid_coeffs = (
                    gen_bid_tv.reindex(index=snapshots, columns=gen_names)
                    .fillna(0.0).values
                )
            else:
                offer_coeffs = np.tile(
                    gen_offer_prices.reindex(gen_names).values, (n_snapshots, 1)
                )
                bid_coeffs = np.tile(
                    gen_bid_prices.reindex(gen_names).values, (n_snapshots, 1)
                )

            offer_ladder_names, offer_ladder_prices, offer_ladder_volumes = (
                _build_ladder_arrays(
                    gen_offer_ladders,
                    snapshots,
                    gen_names,
                    gen_offer_prices,
                    "offer",
                )
            )
            bid_ladder_names, bid_ladder_prices, bid_ladder_volumes = (
                _build_ladder_arrays(
                    gen_bid_ladders,
                    snapshots,
                    gen_names,
                    gen_bid_prices,
                    "bid",
                )
            )

            if offer_ladder_names:
                block_idx = pd.Index(
                    range(1, offer_ladder_prices.shape[2] + 1), name="block"
                )
                offer_upper = xr.DataArray(
                    offer_ladder_volumes,
                    dims=["snapshot", gen_dim, "block"],
                    coords={
                        "snapshot": snapshots,
                        gen_dim: pd.Index(offer_ladder_names, name=gen_dim),
                        "block": block_idx,
                    },
                )
                model.add_variables(
                    lower=xr.zeros_like(offer_upper),
                    upper=offer_upper,
                    name="Generator-offer-block",
                )
                offer_blocks = model.variables["Generator-offer-block"]
                model.add_constraints(
                    offer_blocks.sum("block")
                    == gen_inc.sel({gen_dim: offer_ladder_names}),
                    name="Generator-offer_ladder_balance",
                )
                offer_coeffs_da = xr.DataArray(
                    offer_ladder_prices,
                    dims=["snapshot", gen_dim, "block"],
                    coords=offer_upper.coords,
                )
                gen_obj_parts.append(
                    (offer_blocks * offer_coeffs_da * snapshot_hours_da).sum()
                )

            if bid_ladder_names:
                block_idx = pd.Index(
                    range(1, bid_ladder_prices.shape[2] + 1), name="block"
                )
                bid_upper = xr.DataArray(
                    bid_ladder_volumes,
                    dims=["snapshot", gen_dim, "block"],
                    coords={
                        "snapshot": snapshots,
                        gen_dim: pd.Index(bid_ladder_names, name=gen_dim),
                        "block": block_idx,
                    },
                )
                model.add_variables(
                    lower=xr.zeros_like(bid_upper),
                    upper=bid_upper,
                    name="Generator-bid-block",
                )
                bid_blocks = model.variables["Generator-bid-block"]
                model.add_constraints(
                    bid_blocks.sum("block") == gen_dec.sel({gen_dim: bid_ladder_names}),
                    name="Generator-bid_ladder_balance",
                )
                bid_coeffs_da = xr.DataArray(
                    bid_ladder_prices,
                    dims=["snapshot", gen_dim, "block"],
                    coords=bid_upper.coords,
                )
                gen_obj_parts.append(
                    (bid_blocks * bid_coeffs_da * snapshot_hours_da).sum()
                )

            offer_ladder_set = set(offer_ladder_names)
            bid_ladder_set = set(bid_ladder_names)
            non_ladder_offer_names = [
                name for name in gen_names if name not in offer_ladder_set
            ]
            non_ladder_bid_names = [
                name for name in gen_names if name not in bid_ladder_set
            ]
            if non_ladder_offer_names:
                offer_idx = [gen_names.index(name) for name in non_ladder_offer_names]
                gen_obj_parts.append(
                    (
                        gen_inc.sel({gen_dim: non_ladder_offer_names})
                        * offer_coeffs[:, offer_idx]
                        * snapshot_hours_da
                    ).sum()
                )
            if non_ladder_bid_names:
                bid_idx = [gen_names.index(name) for name in non_ladder_bid_names]
                gen_obj_parts.append(
                    (
                        gen_dec.sel({gen_dim: non_ladder_bid_names})
                        * bid_coeffs[:, bid_idx]
                        * snapshot_hours_da
                    ).sum()
                )

            gen_obj = (
                sum(gen_obj_parts[1:], gen_obj_parts[0]) if gen_obj_parts else None
            )
            obj_expr = gen_obj

        if su_names:
            su_inc = model.variables["StorageUnit-increase"]
            su_dec = model.variables["StorageUnit-decrease"]

            if su_offer_tv is not None and su_bid_tv is not None:
                su_offer_coeffs = (
                    su_offer_tv.reindex(index=snapshots, columns=su_names)
                    .fillna(0.0).values
                )
                su_bid_coeffs = (
                    su_bid_tv.reindex(index=snapshots, columns=su_names)
                    .fillna(0.0).values
                )
            else:
                su_offer_coeffs = np.tile(
                    su_offer_prices.reindex(su_names).values, (n_snapshots, 1)
                )
                su_bid_coeffs = np.tile(
                    su_bid_prices.reindex(su_names).values, (n_snapshots, 1)
                )

            su_obj = (
                su_inc * su_offer_coeffs * snapshot_hours_da
            ).sum() + (
                su_dec * su_bid_coeffs * snapshot_hours_da
            ).sum()
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


def solve_rolling_balancing(
    network,
    wholesale_gen,
    wholesale_su,
    wholesale_links,
    gen_offer,
    gen_bid,
    su_offer,
    su_bid,
    balancing_config,
    solver_name,
    solver_options,
    scenario_config,
    logger,
    gen_offer_tv=None,
    gen_bid_tv=None,
    su_offer_tv=None,
    su_bid_tv=None,
    gen_offer_ladders=None,
    gen_bid_ladders=None,
    ladder_fallback_volume_mw=1.0e6,
    ladder_missing_hour_fallback=True,
    participating_generators=None,
    fixed_generators=None,
    participating_storage_units=None,
    fixed_storage_units=None,
):
    """
    Solve balancing mechanism in rolling windows.

    Each window (default 1 hour = per-timestep) is solved independently
    with full network constraints, anchored to the wholesale positions
    for that window.  Storage state-of-charge carries between windows.

    Parameters
    ----------
    network : pypsa.Network
        Full network with original transmission constraints and snapshots set.
    wholesale_gen, wholesale_su, wholesale_links : pd.DataFrame
        Wholesale dispatch positions from Stage 1 (snapshots x components).
    gen_offer, gen_bid, su_offer, su_bid : pd.Series
        Per-component bid/offer prices.
    balancing_config : dict
        Balancing config with ``window_hours``.
    solver_name : str
    solver_options : dict
    scenario_config : dict
    logger : logging.Logger

    Returns
    -------
    physical_gen : pd.DataFrame
        Concatenated physical generator dispatch.
    physical_su : pd.DataFrame
        Concatenated physical storage dispatch.
    all_nodal_prices : pd.DataFrame
        Concatenated bus marginal prices.
    objective_total : float
        Sum of BM objectives across all windows.
    """
    def _build_retry_solver_options(current_solver_name, current_solver_options):
        if str(current_solver_name).lower() != "gurobi":
            return None

        retry_options = dict(current_solver_options or {})
        for barrier_only_key in ["BarHomogeneous", "BarConvTol", "BarIterLimit"]:
            retry_options.pop(barrier_only_key, None)

        retry_options["method"] = 1
        retry_options["crossover"] = -1
        retry_options["DualReductions"] = 1
        retry_options["NumericFocus"] = max(
            1, int(retry_options.get("NumericFocus", 0) or 0)
        )
        return retry_options

    # Configure a silent logger for subsequent windows to avoid log spam.
    # First window uses the main logger so messages appear once.
    bm_quiet = logging.getLogger("bm_quiet")
    bm_quiet.propagate = False
    if not bm_quiet.handlers:
        bm_quiet.addHandler(logging.NullHandler())

    window_hours = float(balancing_config.get("window_hours", 1))
    fix_ics = balancing_config.get("fix_interconnectors", True)

    all_snapshots = network.snapshots.copy()
    n_total = len(all_snapshots)

    # Determine timestep resolution
    if n_total > 1:
        dt_hours = (all_snapshots[1] - all_snapshots[0]).total_seconds() / 3600
    else:
        dt_hours = 1.0
    steps_per_window = max(1, int(round(window_hours / dt_hours)))

    # Build windows
    windows = []
    for start_idx in range(0, n_total, steps_per_window):
        end_idx = min(start_idx + steps_per_window, n_total)
        windows.append(all_snapshots[start_idx:end_idx])

    logger.info(
        f"Rolling BM: {len(windows)} windows × {window_hours}h "
        f"({steps_per_window} timesteps each, {dt_hours}h resolution)"
    )

    gen_dispatches = []
    su_dispatches = []
    price_dfs = []
    line_flow_dfs = []
    link_flow_dfs = []
    objective_total = 0.0
    prev_final_soc = None
    large_hydro_state = initialise_large_hydro_storage_state(
        network, scenario_config.get("hydro", {})
    )
    log_hydro_constraint_setup(network, scenario_config, logger, context="Balancing hydro")
    neso_boundary_callback = _build_neso_boundary_constraints_callback(
        network, scenario_config, logger
    )

    for i, window_snaps in enumerate(windows):
        window_start = window_snaps[0]
        window_end = window_snaps[-1]

        if (i + 1) % 100 == 0 or i == 0 or i == len(windows) - 1:
            logger.info(
                f"BM window {i + 1}/{len(windows)}: "
                f"{window_start} → {window_end} ({len(window_snaps)} snapshots)"
            )

        # Create a copy restricted to this window
        win_net = network.copy()
        win_net.set_snapshots(window_snaps)

        # Carry over SoC from previous window
        if prev_final_soc is not None:
            for su_name, soc in prev_final_soc.items():
                if su_name in win_net.storage_units.index:
                    win_net.storage_units.loc[su_name, "state_of_charge_initial"] = soc

        # Build extra_functionality for this window's wholesale positions
        ws_gen_window = wholesale_gen.reindex(window_snaps).reindex(
            columns=[
                g for g in wholesale_gen.columns if g in win_net.generators.index
            ],
            fill_value=0.0,
        )
        ws_su_window = wholesale_su.reindex(window_snaps).reindex(
            columns=[
                s for s in wholesale_su.columns if s in win_net.storage_units.index
            ],
            fill_value=0.0,
        )
        ws_links_window = wholesale_links.reindex(window_snaps).reindex(
            columns=[
                lk for lk in wholesale_links.columns if lk in win_net.links.index
            ],
            fill_value=0.0,
        )

        # Slice time-varying prices to this window's snapshots
        win_offer_tv = None
        win_bid_tv = None
        if gen_offer_tv is not None and gen_bid_tv is not None:
            win_offer_tv = gen_offer_tv.reindex(index=window_snaps).fillna(0.0)
            win_bid_tv = gen_bid_tv.reindex(index=window_snaps).fillna(0.0)

        win_su_offer_tv = None
        win_su_bid_tv = None
        if su_offer_tv is not None and su_bid_tv is not None:
            win_su_offer_tv = su_offer_tv.reindex(index=window_snaps).fillna(0.0)
            win_su_bid_tv = su_bid_tv.reindex(index=window_snaps).fillna(0.0)

        extra_func = _build_balancing_extra_functionality(
            wholesale_gen=ws_gen_window,
            wholesale_su=ws_su_window,
            wholesale_links=ws_links_window,
            gen_offer_prices=gen_offer,
            gen_bid_prices=gen_bid,
            su_offer_prices=su_offer,
            su_bid_prices=su_bid,
            fix_interconnectors=fix_ics,
            logger=logger if i == 0 else logging.getLogger("bm_quiet"),
            gen_offer_tv=win_offer_tv,
            gen_bid_tv=win_bid_tv,
            su_offer_tv=win_su_offer_tv,
            su_bid_tv=win_su_bid_tv,
            gen_offer_ladders=gen_offer_ladders,
            gen_bid_ladders=gen_bid_ladders,
            ladder_fallback_volume_mw=ladder_fallback_volume_mw,
            ladder_missing_hour_fallback=ladder_missing_hour_fallback,
            participating_generators=participating_generators,
            fixed_generators=fixed_generators,
            participating_storage_units=participating_storage_units,
            fixed_storage_units=fixed_storage_units,
        )
        hydro_callback = build_hydro_constraints_callback(
            win_net,
            scenario_config,
            large_hydro_storage_state=large_hydro_state,
        )
        # Solve
        status, cond = win_net.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=combine_extra_functionalities(
                extra_func, hydro_callback, neso_boundary_callback
            ),
        )

        if status != "ok" and str(cond).lower() == "unbounded":
            retry_options = _build_retry_solver_options(solver_name, solver_options)
            if retry_options is not None and retry_options != dict(solver_options or {}):
                logger.warning(
                    f"BM window {i + 1} ({window_start}) returned unbounded with "
                    "the configured Gurobi barrier settings; retrying with "
                    "dual-simplex fallback options"
                )
                status, cond = win_net.optimize(
                    solver_name=solver_name,
                    solver_options=retry_options,
                    extra_functionality=combine_extra_functionalities(
                        extra_func, hydro_callback, neso_boundary_callback
                    ),
                )

        if status != "ok":
            logger.error(
                f"BM window {i + 1} ({window_start}) failed: {status} ({cond})"
            )
            raise RuntimeError(
                f"BM window {i + 1} ({window_start}) failed: {status} ({cond})"
            )

        objective_total += win_net.objective

        # Collect physical dispatch
        gen_dispatches.append(win_net.generators_t.p.copy())
        if len(win_net.storage_units_t.p) > 0:
            su_dispatches.append(win_net.storage_units_t.p.copy())

        # Collect nodal prices
        if len(win_net.buses_t.marginal_price) > 0:
            price_dfs.append(win_net.buses_t.marginal_price.copy())

        # Collect line flows for congestion analysis
        if hasattr(win_net, "lines_t") and len(win_net.lines_t.p0) > 0:
            line_flow_dfs.append(win_net.lines_t.p0.copy())

        # Collect link flows (HVDC, etc.)
        if hasattr(win_net, "links_t") and len(win_net.links_t.p0) > 0:
            link_flow_dfs.append(win_net.links_t.p0.copy())

        # Save final SoC for next window
        if len(win_net.storage_units) > 0:
            if (
                hasattr(win_net, "storage_units_t")
                and len(win_net.storage_units_t.state_of_charge) > 0
            ):
                prev_final_soc = (
                    win_net.storage_units_t.state_of_charge.iloc[-1].to_dict()
                )

        large_hydro_state = update_large_hydro_storage_state(
            win_net,
            window_snaps,
            scenario_config,
            large_hydro_state,
        )

    # Concatenate
    physical_gen = pd.concat(gen_dispatches) if gen_dispatches else pd.DataFrame()
    physical_su = pd.concat(su_dispatches) if su_dispatches else pd.DataFrame()
    all_nodal_prices = pd.concat(price_dfs) if price_dfs else pd.DataFrame()
    all_line_flows = pd.concat(line_flow_dfs) if line_flow_dfs else pd.DataFrame()
    all_link_flows = pd.concat(link_flow_dfs) if link_flow_dfs else pd.DataFrame()

    # Write results back to the main network so the saved .nc file
    # contains solve outputs (dispatch, prices, line flows, link flows).
    if not physical_gen.empty:
        network.generators_t.p = physical_gen.reindex(
            columns=network.generators.index, fill_value=0.0
        )
    if not physical_su.empty:
        network.storage_units_t.p = physical_su.reindex(
            columns=network.storage_units.index, fill_value=0.0
        )
    if not all_nodal_prices.empty:
        network.buses_t.marginal_price = all_nodal_prices.reindex(
            columns=network.buses.index, fill_value=0.0
        )
    if not all_line_flows.empty:
        network.lines_t.p0 = all_line_flows.reindex(
            columns=network.lines.index, fill_value=0.0
        )
    if not all_link_flows.empty:
        network.links_t.p0 = all_link_flows.reindex(
            columns=network.links.index, fill_value=0.0
        )

    logger.info(
        f"Rolling BM complete: {len(windows)} windows, "
        f"total BM cost £{objective_total:,.2f}"
    )

    return physical_gen, physical_su, all_nodal_prices, objective_total


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

        # Apply per-line s_nom overrides (fix known ETYS data errors)
        apply_line_rating_overrides(network, scenario_config, logger)

        # Apply transmission outage schedule (if enabled)
        apply_outage_schedule(network, scenario_config, logger)

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
        if hasattr(snakemake.input, 'elexon_offer_ladders'):
            market_config.setdefault("balancing", {}).setdefault("elexon", {})
            market_config["balancing"]["elexon"].setdefault("price_ladders", {})
            market_config["balancing"]["elexon"]["price_ladders"][
                "offer_file"
            ] = snakemake.input.elexon_offer_ladders
        if hasattr(snakemake.input, 'elexon_bid_ladders'):
            market_config.setdefault("balancing", {}).setdefault("elexon", {})
            market_config["balancing"]["elexon"].setdefault("price_ladders", {})
            market_config["balancing"]["elexon"]["price_ladders"][
                "bid_file"
            ] = snakemake.input.elexon_bid_ladders

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
        apply_load_shedding_limits(network, logger)

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

        # ── Apply MC floors for zero-cost dispatchable generators ────────
        # Generators with MC=0 that should have non-zero costs create
        # unrealistic dispatch in both wholesale and BM stages.
        mc_floors = balancing_config.get("mc_floors", {})
        if mc_floors:
            total_fixed = 0
            for carrier, floor_mc in mc_floors.items():
                if floor_mc <= 0:
                    continue
                mask = (
                    (network.generators["carrier"] == carrier)
                    & (network.generators["marginal_cost"] <= 0)
                )
                n_fixed = mask.sum()
                if n_fixed > 0:
                    network.generators.loc[mask, "marginal_cost"] = floor_mc
                    cap_mw = network.generators.loc[mask, "p_nom"].sum()
                    logger.info(
                        f"  MC floor: {carrier}: {n_fixed} generators "
                        f"({cap_mw:.0f} MW) set to £{floor_mc:.1f}/MWh"
                    )
                    total_fixed += n_fixed
            if total_fixed > 0:
                logger.info(
                    f"MC floors applied to {total_fixed} zero-cost generators"
                )

        # ── Calculate bid/offer prices ───────────────────────────────────
        logger.info("=" * 80)
        logger.info("CALCULATING BID/OFFER PRICES")
        logger.info("=" * 80)
        gen_offer, gen_bid, su_offer, su_bid = calculate_bid_offer_prices(
            network, market_config, logger, scenario_id=scenario_id,
            time_varying=True,
        )

        # Retrieve time-varying DataFrames (set by calculate_bid_offer_prices
        # when time_varying=True and ELEXON data is available)
        gen_offer_tv = getattr(network, "_bm_offer_tv", None)
        gen_bid_tv = getattr(network, "_bm_bid_tv", None)
        su_offer_tv = getattr(network, "_bm_su_offer_tv", None)
        su_bid_tv = getattr(network, "_bm_su_bid_tv", None)
        gen_offer_ladders = getattr(network, "_bm_offer_ladders", None)
        gen_bid_ladders = getattr(network, "_bm_bid_ladders", None)
        ladder_fallback_volume_mw = getattr(
            network, "_bm_ladder_fallback_volume_mw", 1.0e6
        )
        ladder_missing_hour_fallback = bool(
            getattr(network, "_bm_ladder_missing_hour_fallback", 1)
        )
        participating_generators = getattr(
            network, "_bm_participating_generators", network.generators.index
        )
        fixed_generators = getattr(network, "_bm_fixed_generators", pd.Index([]))
        participating_storage_units = getattr(
            network, "_bm_participating_storage_units", network.storage_units.index
        )
        fixed_storage_units = getattr(network, "_bm_fixed_storage_units", pd.Index([]))
        if gen_offer_tv is not None:
            logger.info(
                f"Time-varying bid/offer prices: "
                f"{gen_offer_tv.shape[1]} generators × {gen_offer_tv.shape[0]} snapshots"
            )
        if su_offer_tv is not None:
            logger.info(
                f"Time-varying storage bid/offer prices: "
                f"{su_offer_tv.shape[1]} units × {su_offer_tv.shape[0]} snapshots"
            )

        # ── Configure solver ─────────────────────────────────────────────
        if gen_offer_ladders is not None and not gen_offer_ladders.empty:
            logger.info(
                f"Offer price ladders active: {len(gen_offer_ladders):,} blocks"
            )
        if gen_bid_ladders is not None and not gen_bid_ladders.empty:
            logger.info(
                f"Bid price ladders active: {len(gen_bid_ladders):,} blocks"
            )

        solver_name, solver_options = configure_solver(
            network, solver_name, solver_options, logger
        )

        # ── Dispatch based on balancing mode ─────────────────────────────
        bm_mode = balancing_config.get("mode", "full_period")
        logger.info(f"Balancing mode: {bm_mode}")

        solve_start = time.time()

        if bm_mode == "rolling":
            # ── Rolling BM: solve in windows ─────────────────────────────
            logger.info("Starting rolling BM optimization...")
            physical_gen, physical_su, nodal_prices_all, bm_cost = (
                solve_rolling_balancing(
                    network=network,
                    wholesale_gen=wholesale_gen,
                    wholesale_su=wholesale_su,
                    wholesale_links=wholesale_links,
                    gen_offer=gen_offer,
                    gen_bid=gen_bid,
                    su_offer=su_offer,
                    su_bid=su_bid,
                    balancing_config=balancing_config,
                    solver_name=solver_name,
                    solver_options=solver_options,
                    scenario_config=scenario_config,
                    logger=logger,
                    gen_offer_tv=gen_offer_tv,
                    gen_bid_tv=gen_bid_tv,
                    su_offer_tv=su_offer_tv,
                    su_bid_tv=su_bid_tv,
                    gen_offer_ladders=gen_offer_ladders,
                    gen_bid_ladders=gen_bid_ladders,
                    ladder_fallback_volume_mw=ladder_fallback_volume_mw,
                    ladder_missing_hour_fallback=ladder_missing_hour_fallback,
                    participating_generators=participating_generators,
                    fixed_generators=fixed_generators,
                    participating_storage_units=participating_storage_units,
                    fixed_storage_units=fixed_storage_units,
                )
            )

            solve_time = time.time() - solve_start
            logger.info(f"Rolling BM solve completed in {solve_time:.2f}s")
            logger.info(f"Total BM redispatch cost: £{bm_cost:,.2f}")

            # Export BM dispatch (generators)
            physical_gen.to_csv(snakemake.output.balancing_dispatch_csv)
            logger.info(f"Saved: {snakemake.output.balancing_dispatch_csv}")

            if physical_su.empty:
                physical_su = pd.DataFrame(
                    index=physical_gen.index if len(physical_gen) > 0
                    else network.snapshots
                )

            # Export BM dispatch (storage)
            physical_su.to_csv(snakemake.output.balancing_storage_csv)
            logger.info(f"Saved: {snakemake.output.balancing_storage_csv}")

        else:
            # ── Full-period solve (original behaviour) ───────────────────
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
                gen_offer_tv=gen_offer_tv,
                gen_bid_tv=gen_bid_tv,
                su_offer_tv=su_offer_tv,
                su_bid_tv=su_bid_tv,
                gen_offer_ladders=gen_offer_ladders,
                gen_bid_ladders=gen_bid_ladders,
                ladder_fallback_volume_mw=ladder_fallback_volume_mw,
                ladder_missing_hour_fallback=ladder_missing_hour_fallback,
                participating_generators=participating_generators,
                fixed_generators=fixed_generators,
                participating_storage_units=participating_storage_units,
                fixed_storage_units=fixed_storage_units,
            )
            log_hydro_constraint_setup(
                network,
                scenario_config,
                logger,
                context="Balancing hydro",
            )
            hydro_callback = build_hydro_constraints_callback(
                network,
                scenario_config,
            )
            neso_boundary_callback = _build_neso_boundary_constraints_callback(
                network, scenario_config, logger
            )

            logger.info("Starting BM optimization (full network constraints)...")

            status, termination_condition = network.optimize(
                solver_name=solver_name,
                solver_options=solver_options,
                extra_functionality=combine_extra_functionalities(
                    extra_func, hydro_callback, neso_boundary_callback
                ),
            )

            solve_time = time.time() - solve_start
            logger.info(f"BM solve completed in {solve_time:.2f}s")
            logger.info(f"Status: {status}, Condition: {termination_condition}")

            if status != "ok":
                raise RuntimeError(
                    f"BM optimization failed: "
                    f"{status} ({termination_condition})"
                )

            bm_cost = network.objective
            logger.info(f"Total BM redispatch cost: £{bm_cost:,.2f}")

            # Export BM dispatch (generators)
            physical_gen = network.generators_t.p.copy()
            physical_gen.to_csv(snakemake.output.balancing_dispatch_csv)
            logger.info(f"Saved: {snakemake.output.balancing_dispatch_csv}")

            # Export BM dispatch (storage)
            physical_su = (
                network.storage_units_t.p.copy()
                if len(network.storage_units_t.p) > 0
                else pd.DataFrame(index=network.snapshots)
            )
            physical_su.to_csv(snakemake.output.balancing_storage_csv)
            logger.info(f"Saved: {snakemake.output.balancing_storage_csv}")

            nodal_prices_all = (
                network.buses_t.marginal_price.copy()
                if len(network.buses_t.marginal_price) > 0
                else pd.DataFrame()
            )

        # ── Compute redispatch summary ───────────────────────────────────
        logger.info("=" * 80)
        logger.info("COMPUTING REDISPATCH VOLUMES AND COSTS")
        logger.info("=" * 80)

        # Use physical_gen index as the canonical snapshot index
        snaps = physical_gen.index if len(physical_gen) > 0 else network.snapshots

        ws_gen_aligned = wholesale_gen.reindex(snaps).reindex(
            columns=physical_gen.columns, fill_value=0.0
        )
        ws_su_aligned = (
            wholesale_su.reindex(snaps).reindex(
                columns=physical_su.columns, fill_value=0.0
            )
            if len(physical_su.columns) > 0
            else pd.DataFrame(index=snaps)
        )

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
            gen_offer_prices_tv=gen_offer_tv,
            gen_bid_prices_tv=gen_bid_tv,
            su_offer_prices_tv=su_offer_tv,
            su_bid_prices_tv=su_bid_tv,
            gen_offer_ladders=gen_offer_ladders,
            gen_bid_ladders=gen_bid_ladders,
        )

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
        # In rolling mode, results are written back to the network object
        # after the rolling solve completes, so congestion analysis works.
        congestion = identify_congested_boundaries(
            network, threshold=0.95, logger=logger
        )
        congestion.to_csv(snakemake.output.congestion_csv, index=False)
        logger.info(f"Saved: {snakemake.output.congestion_csv}")

        # ── Price comparison ─────────────────────────────────────────────
        if len(nodal_prices_all) > 0:
            # Filter to GB-only demand (load) buses
            load_buses = set(network.loads["bus"].unique())
            if "country" in network.buses.columns:
                gb_buses = set(
                    network.buses.index[network.buses["country"] == "GB"]
                )
                demand_cols = [
                    c
                    for c in nodal_prices_all.columns
                    if c in load_buses and c in gb_buses
                ]
            else:
                demand_cols = [
                    c for c in nodal_prices_all.columns if c in load_buses
                ]
            if demand_cols:
                nodal_prices_demand = nodal_prices_all[demand_cols]
                logger.info(
                    f"BM nodal price computed from {len(demand_cols)} "
                    f"GB demand buses (of {len(nodal_prices_all.columns)} total)"
                )
            else:
                nodal_prices_demand = nodal_prices_all
                logger.warning(
                    "No GB demand buses found in BM marginal_price; "
                    "using all buses"
                )

            mean_nodal = nodal_prices_demand.mean(axis=1)
            price_spread = (
                nodal_prices_demand.max(axis=1) - nodal_prices_demand.min(axis=1)
            )

            ws_price_aligned = wholesale_price["wholesale_price"].reindex(
                mean_nodal.index, fill_value=np.nan
            )

            price_comparison = pd.DataFrame(
                {
                    "wholesale_price": ws_price_aligned,
                    "mean_nodal_price": mean_nodal,
                    "min_nodal_price": nodal_prices_demand.min(axis=1),
                    "max_nodal_price": nodal_prices_demand.max(axis=1),
                    "nodal_spread": price_spread,
                }
            )
            price_comparison.to_csv(snakemake.output.price_comparison_csv)
            logger.info(f"Saved: {snakemake.output.price_comparison_csv}")
            logger.info(
                f"Price comparison: wholesale mean "
                f"£{ws_price_aligned.mean():.2f}/MWh, "
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
        _clear_runtime_bm_attrs(network)
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
