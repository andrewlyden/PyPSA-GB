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
import os
import time
import yaml
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network, save_network
from scripts.network_build.branch_units import (
    require_pypsa_branch_units,
    set_transformer_s_nom_preserving_impedance,
)
from scripts.solve.solve_network import (
    _build_neso_boundary_constraints_callback,
    validate_network_costs,
    apply_line_rating_overrides,
    apply_load_shedding_limits,
    apply_outage_schedule,
    apply_seasonal_line_ratings,
    apply_transmission_relaxation,
    enforce_feasible_generator_operating_bounds,
    improve_numerical_conditioning,
    configure_solver,
    get_solve_mode_from_config,
)
from scripts.solve.hydro_constraints import (
    build_hydro_constraints_callback,
    combine_extra_functionalities,
    initialise_large_hydro_storage_state,
    large_hydro_end_floor,
    large_hydro_net_inflow,
    log_hydro_constraint_setup,
    update_large_hydro_storage_state,
)
from scripts.market.market_utils import (
    HYDROGEN_PRICE_COLUMN,
    SolveFailedError,
    apply_carried_store_levels,
    apply_initial_storage_soc,
    apply_churn_sink,
    apply_storage_churn_share,
    apply_storage_dispatch_cost_floor,
    build_end_of_window_store_value_callback,
    build_min_output_energy_callback,
    build_month_end_soc_callback,
    build_month_end_store_callback,
    carried_store_names,
    final_store_levels,
    prepare_carried_stores,
    resolve_store_level_config,
    store_end_of_window_credit,
    extract_wholesale_positions,
    log_fallback_summary,
    min_output_energy_delivered,
    objective_only_adjustment,
    prepare_min_output_energy_budget,
    trajectory_period_end_report,
    solve_with_counterflow_lock,
    solver_io_kwargs,
    wholesale_hydrogen_price,
)
from scripts.market.copperplate import EXTERNAL_BUS_PATTERN, collapse_mainland, expand_results, identify_mainland
from scripts.market.forecast_error import apply_day_ahead_forecast_error
from scripts.market.internal_dc_tie_break import (
    META_KEY as INTERNAL_DC_TIE_BREAK_META_KEY,
    build_internal_dc_tie_break_callback,
    internal_dc_tie_break_cost,
    record_internal_dc_tie_break,
    resolve_internal_dc_tie_break,
)
from scripts.market.marginal_cost_basis import apply_marginal_cost_basis, needs_strike_table
from scripts.market.temporal_resolution import (
    aggregate_frame as aggregate_frame_to_blocks,
    aggregate_to_blocks,
    check_compatible as check_temporal_resolution,
    expand_rows as expand_block_rows,
    record_meta as record_temporal_resolution,
    resolve_temporal_resolution_hours,
    restore_hourly,
)
from scripts.market.wholesale_commitment import CommitmentOptimizer, capture_bounds, configure_commitment
from scripts.interconnectors.market_coupling import (
    build_interconnector_shared_capacity_callback,
    build_piecewise_interconnector_cost_callback,
    lock_material_counterflows,
)


def build_wholesale_scarcity_callback(network, wholesale_config, logger):
    """The Stage 1 scarcity curve (item MR3(b)) when ``scarcity_pricing.enabled``."""
    cfg = (wholesale_config or {}).get("scarcity_pricing") or {}
    if not cfg.get("enabled", False):
        return None
    from scripts.market.scarcity_pricing import (
        DEFAULT_EXCLUDED_CARRIERS, build_scarcity_pricing_callback, load_scarcity_curve,
    )
    curve = load_scarcity_curve(cfg["curve_file"], cfg.get("max_drm_threshold_mw"))
    return build_scarcity_pricing_callback(
        network, curve, logger,
        excluded_carriers=tuple(cfg.get("excluded_carriers", DEFAULT_EXCLUDED_CARRIERS)),
    )


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


def _attach_hydrogen_price(price_df, prices, network, logger):
    """Add the hourly hydrogen-bus dual to the price table (in place), when there is one.

    The balancing stage prices electrolysers and H2 turbines at this value
    (market_utils.hydrogen_link_prices); without it Stage 2 moved them for free.
    """
    hydrogen = wholesale_hydrogen_price(prices, network)
    if hydrogen is None:
        return price_df
    price_df[HYDROGEN_PRICE_COLUMN] = hydrogen.reindex(price_df.index)
    logger.info(
        f"Hydrogen value (Stage 1 hydrogen-bus dual): mean {hydrogen.mean():.2f} GBP/MWh_H2, "
        f"range {hydrogen.min():.2f}-{hydrogen.max():.2f}"
    )
    return price_df


STORAGE_CHURN_WARN_MWH = 1000.0  # per window; diagnostic threshold (1 GWh)
STORAGE_CHURN_META_KEY = "stage1_storage_churn"


def measure_storage_churn(network, logger, label="", warn_mwh=STORAGE_CHURN_WARN_MWH, snapshots=None):
    """Energy dissipated by simultaneous charge and discharge (diagnostic only).

    For each storage unit and snapshot the churned power is
    ``m = min(p_store, p_dispatch)``: m MW flows in and out at once, so the
    grid sees no net flow but the state of charge falls by
    ``m * (1 / efficiency_dispatch - efficiency_store)``. That loss, summed with
    the snapshot weightings, is the energy the solve burnt (a degenerate or
    zero-price optimum; see the storage zero-price degeneracy note). Logs a
    WARNING when the window total exceeds ``warn_mwh``. No behaviour change.
    ``snapshots`` (None = all) limits the measure to those rows, the kept
    hours of a window solved with a look-ahead.

    Returns a dict with the dissipated and churned MWh and the worst units.
    """
    su = network.storage_units
    out = {"label": str(label), "dissipated_mwh": 0.0, "churned_mwh": 0.0, "top_units": {}}
    st = getattr(network, "storage_units_t", None)
    if st is None or not len(su):
        return out
    p_store = getattr(st, "p_store", None)
    p_dispatch = getattr(st, "p_dispatch", None)
    if p_store is None or p_dispatch is None or p_store.empty or p_dispatch.empty:
        return out
    cols = p_store.columns.intersection(p_dispatch.columns).intersection(su.index)
    if not len(cols):
        return out
    both = np.minimum(p_store[cols].clip(lower=0.0), p_dispatch[cols].clip(lower=0.0))
    if snapshots is not None:
        both = both.loc[both.index.isin(pd.Index(snapshots))]
    weights = network.snapshot_weightings.generators.reindex(both.index).fillna(1.0)
    eta_s = pd.to_numeric(su.loc[cols, "efficiency_store"], errors="coerce").fillna(1.0)
    eta_d = pd.to_numeric(su.loc[cols, "efficiency_dispatch"], errors="coerce").fillna(1.0).clip(lower=1e-6)
    loss_factor = (1.0 / eta_d - eta_s).clip(lower=0.0)
    churned = both.mul(weights, axis=0).sum()
    dissipated = churned * loss_factor
    out["dissipated_mwh"] = float(dissipated.sum())
    out["churned_mwh"] = float(churned.sum())
    top = dissipated[dissipated > 0].sort_values(ascending=False).head(5)
    out["top_units"] = {str(k): round(float(v), 1) for k, v in top.items()}
    message = (
        f"Stage 1 storage churn {label}: {out['dissipated_mwh']:,.0f} MWh dissipated by "
        f"simultaneous charge/discharge ({out['churned_mwh']:,.0f} MWh churned)"
    )
    if out["dissipated_mwh"] > warn_mwh:
        logger.warning(message + f" > {warn_mwh:,.0f} MWh threshold; worst units {out['top_units']}")
    else:
        logger.debug(message)
    return out


def record_storage_churn(network, windows, logger):
    """Log the run total and store the per-window results in ``network.meta``."""
    total = float(sum(w["dissipated_mwh"] for w in windows))
    churned = float(sum(w["churned_mwh"] for w in windows))
    flagged = [w["label"] for w in windows if w["dissipated_mwh"] > STORAGE_CHURN_WARN_MWH]
    logger.info(
        f"Stage 1 storage churn total: {total:,.0f} MWh dissipated, {churned:,.0f} MWh churned "
        f"over {len(windows)} window(s); {len(flagged)} above {STORAGE_CHURN_WARN_MWH:,.0f} MWh"
        + (f" ({', '.join(flagged[:10])})" if flagged else "")
    )
    meta = dict(getattr(network, "meta", None) or {})
    meta[STORAGE_CHURN_META_KEY] = {
        "dissipated_mwh": round(total, 3),
        "churned_mwh": round(churned, 3),
        "windows_above_threshold": len(flagged),
        "threshold_mwh_per_window": STORAGE_CHURN_WARN_MWH,
        "per_window_dissipated_mwh": {w["label"]: round(w["dissipated_mwh"], 3) for w in windows},
    }
    network.meta = meta
    return total


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
        # Transformer x is per unit on s_nom: rescale it with the rating so the
        # physical impedance (x_pu) is unchanged.
        set_transformer_s_nom_preserving_impedance(
            network, network.transformers.index, relaxation
        )
        logger.info(
            f"Transformers: set s_nom = {relaxation:.0e} MVA for "
            f"{len(network.transformers)} transformers"
        )

    logger.info("Network is now effectively copperplate (no transmission constraints)")
    logger.info("=" * 80)


ISLAND_HVDC_ORIGINAL_COLUMN = "p_nom_original"


def relax_island_hvdc_links(network, wholesale_config, logger, external_pattern=EXTERNAL_BUS_PATTERN):
    """Stage 1 copperplate: lift the rating of every GB-internal HVDC link that serves an island.

    Collapsing the mainland keeps every link's rating, so the GB day-ahead market
    still saw the 600 MW Shetland HVDC (KERG1J - Central DC Substation) and the
    Caithness-Moray hub legs as limits, capping Shetland's ~870 MW of wind at
    the link in Stage 1 (review 27 Sep 2026, item C3). The day-ahead market is national: a
    network limit inside GB is the balancing stage's to resolve, and the BM
    reloads the built network, so it still sees the real ratings. Internal links
    are those of ``island_hvdc_carriers`` with neither end on an External (foreign)
    bus and at least one end off the mainland synchronous area. Each is raised to
    at least what the islands these links serve could push through it together
    (generation and storage power plus peak load, load shedding excluded), 5%
    over, which cannot bind
    and keeps the LP well scaled (no 1e6 bounds). Interconnectors and the
    hydrogen links are untouched. The original rating is kept in
    ``p_nom_original`` and restored by ``restore_island_hvdc_links``.

    Returns the relaxed link names.
    """
    if not bool(wholesale_config.get("relax_island_hvdc", True)) or len(network.links) == 0:
        return []
    carriers = {str(c) for c in (wholesale_config.get("island_hvdc_carriers") or ["DC"])}
    mainland, _ = identify_mainland(network, external_pattern)
    links = network.links
    external = (links["bus0"].astype(str).str.contains(external_pattern, regex=False)
                | links["bus1"].astype(str).str.contains(external_pattern, regex=False))
    off_mainland = ~links["bus0"].isin(mainland) | ~links["bus1"].isin(mainland)
    names = links.index[links["carrier"].astype(str).isin(carriers) & ~external & off_mainland]
    if not len(names):
        logger.info("Island HVDC: no GB-internal HVDC link serves an island; nothing to relax")
        return []
    # The islands these links serve: the synchronous sub-networks of their
    # off-mainland ends (identify_mainland has just set buses.sub_network).
    ends = pd.concat([links.loc[names, "bus0"], links.loc[names, "bus1"]])
    ends = ends[~ends.isin(mainland)]
    sub = network.buses["sub_network"]
    island_buses = set(sub.index[sub.isin(set(sub.reindex(ends.unique())))])
    gens = network.generators
    gen_mw = float(pd.to_numeric(
        gens.loc[gens["bus"].isin(island_buses) & (gens["carrier"] != "load_shedding"), "p_nom"],
        errors="coerce").fillna(0.0).sum())
    su = network.storage_units
    su_mw = float(pd.to_numeric(su.loc[su["bus"].isin(island_buses), "p_nom"], errors="coerce").fillna(0.0).sum()) if len(su) else 0.0
    loads = network.loads.index[network.loads["bus"].isin(island_buses)]
    load_mw = 0.0
    if len(loads):
        varying = network.loads_t.p_set.reindex(columns=loads)
        static = pd.to_numeric(network.loads.loc[loads, "p_set"], errors="coerce").fillna(0.0)
        load_mw = float(varying.fillna(static).sum(axis=1).max()) if len(varying) else float(static.sum())
    headroom = max(1.0, 1.05 * (gen_mw + su_mw + load_mw))
    original = pd.to_numeric(links.loc[names, "p_nom"], errors="coerce").fillna(0.0)
    if ISLAND_HVDC_ORIGINAL_COLUMN not in network.links.columns:
        network.links[ISLAND_HVDC_ORIGINAL_COLUMN] = np.nan
    network.links.loc[names, ISLAND_HVDC_ORIGINAL_COLUMN] = original
    network.links.loc[names, "p_nom"] = np.maximum(original.to_numpy(), headroom)
    logger.info(
        f"Island HVDC relaxed for the national day-ahead market: {len(names)} internal link(s) lifted to "
        f">= {headroom:,.0f} MW (islands served: generation {gen_mw:,.0f} MW, storage {su_mw:,.0f} MW, peak load "
        f"{load_mw:,.0f} MW); was " + ", ".join(f"{n} {original[n]:,.0f} MW" for n in names)
        + ". The balancing stage keeps the real ratings."
    )
    return list(names)


def restore_island_hvdc_links(network, names, logger):
    """Put back the ratings ``relax_island_hvdc_links`` lifted (the saved Stage 1 network keeps real ratings).

    ``p_nom_original`` stays on the links as a record, like ``s_nom_original`` in relax mode.
    """
    names = [n for n in (names or []) if n in network.links.index]
    if not names or ISLAND_HVDC_ORIGINAL_COLUMN not in network.links.columns:
        return
    network.links.loc[names, "p_nom"] = network.links.loc[names, ISLAND_HVDC_ORIGINAL_COLUMN].to_numpy()
    logger.info(f"Island HVDC: restored the real ratings of {len(names)} link(s) on the saved Stage 1 network")


def apply_central_network_delivery(network, scenario_config, logger, projects_file=None):
    """A central dispatch Stage 1's network delivery case; returns the scenario config to solve with.

    ``market.balancing.network_delivery`` (the outage replay of NESO's day-ahead boundary
    limits, delayed projects, the BM's transmission settings) is applied by the balancing
    stage alone. A copperplate Stage 1 has no network, so that is right for a market; a
    central dispatch schedules on the real network, and a day-ahead operator knows the
    day-ahead limits, so it gets the same case (18 Oct 2026). The case edits the scenario's
    ``transmission`` block, so a copy is returned and the caller's config is left alone.
    Without a case this is a no-op and returns the config unchanged.
    """
    import copy

    from scripts.market.network_delivery import apply_network_delivery, network_delivery_config

    balancing = ((scenario_config.get("market") or {}).get("balancing")) or {}
    if network_delivery_config(balancing) is None:
        return scenario_config
    scenario_config = copy.deepcopy(scenario_config)
    apply_network_delivery(network, scenario_config, (scenario_config.get("market") or {}).get("balancing"),
                           logger, projects_file=projects_file)
    logger.info("Central dispatch Stage 1: the balancing stage's network delivery case applied")
    return scenario_config


def build_central_boundary_callback(network, scenario_config, wholesale_config, logger):
    """Secured boundary limits for a central dispatch Stage 1 (``copperplate_mode: none``), else None.

    A central operator schedules on the real network, so it must respect the same
    aggregate boundary limits (SC1, B2, SC1.5, ...) the balancing stage enforces;
    without them Stage 1 moved power across boundaries the BM then had to undo
    (review 27 Sep 2026, item C1). The copperplate modes have no network, so no
    boundaries.
    """
    if str(wholesale_config.get("copperplate_mode", "collapse")).lower() != "none":
        return None
    callback = _build_neso_boundary_constraints_callback(network, scenario_config, logger)
    logger.info(
        "Central dispatch Stage 1: "
        + ("secured boundary limits enforced (as in the balancing stage)" if callback is not None
           else "no aggregate boundary limits configured for this scenario")
    )
    return callback


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


def _normalise_ramp_limit(value, default):
    """Return a PyPSA ramp limit fraction, accepting either percent or pu."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(value, 1.0))


def _commitment_initial_state(network, eligible, uc_config, carrier_params, logger):
    """Set generator initial up/down state for wholesale unit commitment."""
    mode = str(uc_config.get("initial_status", "off")).lower()
    if "up_time_before" not in network.generators.columns:
        network.generators["up_time_before"] = 0
    if "down_time_before" not in network.generators.columns:
        network.generators["down_time_before"] = 0

    if mode == "on":
        network.generators.loc[eligible, ["up_time_before", "down_time_before"]] = 0
        for carrier, params in carrier_params.items():
            carrier_mask = eligible & (network.generators["carrier"] == carrier)
            min_up = int(params.get("min_up_time", 0))
            network.generators.loc[carrier_mask, "up_time_before"] = max(min_up, 1)
        logger.info("Wholesale UC initial status: all committed units initially on")
    elif mode == "preserve":
        logger.info("Wholesale UC initial status: preserving network up/down state")
    else:
        network.generators.loc[eligible, ["up_time_before", "down_time_before"]] = 0
        for carrier, params in carrier_params.items():
            carrier_mask = eligible & (network.generators["carrier"] == carrier)
            min_down = int(params.get("min_down_time", 0))
            network.generators.loc[carrier_mask, "down_time_before"] = max(min_down, 1)
        logger.info("Wholesale UC initial status: all committed units initially off")


def apply_wholesale_unit_commitment(network, wholesale_config, logger):
    """
    Configure unit commitment for the wholesale stage only.

    The base network is normally built in LP mode, so thermal generators arrive
    with ``committable=False`` and no ramp/start parameters.  This helper applies
    a market-stage UC overlay immediately before the wholesale solve without
    changing the balancing-stage input network.
    """
    uc_config = wholesale_config.get("unit_commitment", {})
    enabled = bool(uc_config.get("enabled", False))

    if "committable" not in network.generators.columns:
        network.generators["committable"] = False

    if not enabled:
        n_before = int(network.generators["committable"].sum())
        network.generators["committable"] = False
        logger.info(
            "Wholesale unit commitment disabled"
            + (f" ({n_before} committable flags cleared)" if n_before else "")
        )
        return False

    params_by_carrier = uc_config.get("carrier_parameters", {})
    if not params_by_carrier:
        raise ValueError(
            "market.wholesale.unit_commitment.enabled is true but no "
            "carrier_parameters are configured"
        )

    logger.info("=" * 80)
    logger.info("APPLYING WHOLESALE UNIT COMMITMENT OVERLAY")
    logger.info("=" * 80)

    # Start from a clean slate so renewables/load shedding never inherit stale
    # committable flags from input data.
    network.generators["committable"] = False

    min_p_nom_mw = float(uc_config.get("min_p_nom_mw", 50.0))
    exclude_carriers = set(uc_config.get("exclude_carriers", []))
    default_ramp_up = _normalise_ramp_limit(
        uc_config.get("default_ramp_limit_up", 1.0), 1.0
    )
    default_ramp_down = _normalise_ramp_limit(
        uc_config.get("default_ramp_limit_down", 1.0), 1.0
    )
    default_start_ramp = _normalise_ramp_limit(
        uc_config.get("default_ramp_limit_start_up", 1.0), 1.0
    )
    default_shut_ramp = _normalise_ramp_limit(
        uc_config.get("default_ramp_limit_shut_down", 1.0), 1.0
    )

    required_cols = [
        "p_min_pu",
        "min_up_time",
        "min_down_time",
        "start_up_cost",
        "shut_down_cost",
        "ramp_limit_up",
        "ramp_limit_down",
        "ramp_limit_start_up",
        "ramp_limit_shut_down",
    ]
    for col in required_cols:
        if col not in network.generators.columns:
            network.generators[col] = np.nan if col.startswith("ramp_") else 0.0

    if "active" in network.generators.columns:
        active = network.generators["active"].astype(bool)
    else:
        active = pd.Series(True, index=network.generators.index)
    if "p_nom_extendable" in network.generators.columns:
        extendable = network.generators["p_nom_extendable"].astype(bool)
    else:
        extendable = pd.Series(False, index=network.generators.index)

    eligible_any = pd.Series(False, index=network.generators.index)
    applied_params = {}

    for carrier, params in params_by_carrier.items():
        if not params or not bool(params.get("enabled", True)):
            continue
        if carrier in exclude_carriers:
            continue

        mask = (
            (network.generators["carrier"] == carrier)
            & (network.generators["p_nom"].astype(float) >= min_p_nom_mw)
            & active
            & ~extendable
        )
        if not mask.any():
            logger.info(f"Wholesale UC carrier {carrier}: no eligible generators")
            continue

        p_min_pu = float(params.get("p_min_pu", 0.0))
        min_up = int(round(float(params.get("min_up_time", 0))))
        min_down = int(round(float(params.get("min_down_time", 0))))
        ramp_up = _normalise_ramp_limit(params.get("ramp_limit_up"), default_ramp_up)
        ramp_down = _normalise_ramp_limit(
            params.get("ramp_limit_down"), default_ramp_down
        )
        start_ramp = _normalise_ramp_limit(
            params.get("ramp_limit_start_up"), default_start_ramp
        )
        shut_ramp = _normalise_ramp_limit(
            params.get("ramp_limit_shut_down"), default_shut_ramp
        )
        start_cost_per_mw = float(params.get("start_up_cost_per_mw", 0.0))
        shut_cost_per_mw = float(params.get("shut_down_cost_per_mw", 0.0))

        network.generators.loc[mask, "committable"] = True
        network.generators.loc[mask, "p_min_pu"] = p_min_pu
        network.generators.loc[mask, "min_up_time"] = min_up
        network.generators.loc[mask, "min_down_time"] = min_down
        network.generators.loc[mask, "ramp_limit_up"] = ramp_up
        network.generators.loc[mask, "ramp_limit_down"] = ramp_down
        network.generators.loc[mask, "ramp_limit_start_up"] = start_ramp
        network.generators.loc[mask, "ramp_limit_shut_down"] = shut_ramp
        network.generators.loc[mask, "start_up_cost"] = (
            start_cost_per_mw * network.generators.loc[mask, "p_nom"].astype(float)
        )
        network.generators.loc[mask, "shut_down_cost"] = (
            shut_cost_per_mw * network.generators.loc[mask, "p_nom"].astype(float)
        )

        eligible_any |= mask
        applied_params[carrier] = {"min_up_time": min_up, "min_down_time": min_down}
        logger.info(
            f"Wholesale UC carrier {carrier}: {int(mask.sum())} generators, "
            f"{network.generators.loc[mask, 'p_nom'].sum():,.0f} MW, "
            f"p_min_pu={p_min_pu:.2f}, min_up={min_up}h, min_down={min_down}h, "
            f"ramp={ramp_up:.2f}/{ramp_down:.2f}"
        )

    if not eligible_any.any():
        raise ValueError("Wholesale unit commitment enabled but no generators matched")

    _commitment_initial_state(
        network, eligible_any, uc_config, applied_params, logger
    )

    n_committable = int(network.generators["committable"].sum())
    capacity_committable = network.generators.loc[
        network.generators["committable"], "p_nom"
    ].sum()
    logger.info(
        f"Wholesale UC enabled: {n_committable} generators, "
        f"{capacity_committable:,.0f} MW. Problem is now MILP for Stage 1."
    )
    logger.info("=" * 80)
    return True


def apply_wholesale_solver_overrides(solver_options, wholesale_config, logger):
    """Apply optional wholesale-stage solver overrides."""
    overrides = wholesale_config.get("solver_options", {})
    if not overrides:
        return solver_options

    merged = dict(solver_options or {})
    merged.update(overrides)
    logger.info(f"Applied wholesale solver option overrides: {overrides}")
    return merged


def _optimize_wholesale(network, **kwargs):
    """Run PyPSA optimize while hiding a known Linopy/xarray UC warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "In a future version of xarray the default value for join "
                "will change from join='outer' to join='exact'.*"
            ),
            category=FutureWarning,
            module=r"linopy\.expressions",
        )
        return network.optimize(**kwargs)


# ── Look-ahead for the rolling day-ahead stage (OW1, 3 Oct 2026) ─────────
#
# The rolling Stage 1 carries each storage unit's state of charge from one 24 h
# window to the next (carry_soc) but gives it no value at the window's end, so
# every day sells what it holds by midnight at any price above the 1.0 cost floor
# and starts the next day by charging again. The 2030 CP30 year shows the
# signature: the hour-23 price averages 36.6 GBP/MWh with the fleet discharging
# 1.33 GW, against 45.6 at 00:00 with 2.97 GW charging (OW1 hourly series,
# scratchpad ow1/hourly.csv); storage's opportunity value sets 31% of the
# 2,374 shoulder hours (1-60 GBP/MWh). ``look_ahead_hours`` N optimises each
# window over its own hours plus the next N (cut at the solve period's end, where
# the month-end conditions take over) and keeps only its own hours: the dispatch
# and prices of the kept hours, and the state of charge, hydrogen store level and
# large-hydro reservoir at the last kept snapshot, which the next window starts
# from. It is PyPSA's own rolling-horizon overlap
# (Network.optimize.optimize_with_rolling_horizon(horizon, overlap)).
# Chosen over the alternatives:
#   - a terminal value V x SoC(T): one V per carrier cannot follow the
#     day-to-day swing between surplus days (energy worth ~0) and gas-set days
#     (100-150), so it would itself set the shoulder's prices;
#   - a terminal SoC target: forces the end level whatever the next day holds,
#     which is the opposite of what a multi-day glut needs.
# The look-ahead values end-of-day energy at the next day's own prices from the
# same LP. 24 h exceeds the duration of every 2030 unit that cycles daily
# (batteries 1.8 h, LAES 12.5 h), so the horizon's own end condition (energy
# worth nothing at its last hour) no longer reaches the kept day. It does not
# reopen the zero-price storage degeneracy (solve.solver.crossover note): the storage
# cost floor still makes simultaneous charge and discharge strictly costly, and
# stored energy now has a positive value, so a zero-price hour gives no reason to
# discharge. Cost: each LP is (24 + N) / 24 times longer (about 2x at N = 24).
# 0 (the default) solves exactly as before. Defaults must match
# config/defaults/ market.wholesale.look_ahead_hours.
#
# Coarse look-ahead (``look_ahead_resolution_hours`` R, follow-up of 3 Oct 2026).
# The look-ahead only has to value the end-of-day state, so it need not be
# hourly. With R > 1 the kept hours stay hourly and the look-ahead hours become
# blocks of R hours (``coarsen_look_ahead``): every time-varying input (loads,
# p_max_pu / p_min_pu, marginal-cost series, inflows, link limits, line ratings)
# is the block's weighted mean and the block's snapshot weightings are the sum
# of its hours, so the energy balance, the generators' costs and PyPSA's
# state-of-charge update (soc += w x (eta_s p_store - p_dispatch / eta_d), with
# standing loss to the power w) carry the block's true duration. Each window is
# then 24 + N / R snapshots: at N = 24, R = 6 it is 28 (1.17x) against 48 (2x)
# hourly, and on the offline price taker (scratchpad
# shoulder/storage_lookahead_coarse.py) it keeps 98% of the hourly look-ahead's
# arbitrage gain. Known approximation inside the look-ahead only: ramp limits
# act per snapshot, so a block may move by one hour's ramp. 1 is the hourly
# look-ahead exactly; the default is 6. Defaults must match
# market.wholesale.look_ahead_resolution_hours.


def resolve_look_ahead_hours(wholesale_config) -> float:
    """``market.wholesale.look_ahead_hours`` (default 24, the base's; 0 or null = no look-ahead), validated."""
    value = (wholesale_config or {}).get("look_ahead_hours", 24)
    value = 0.0 if value is None else float(value)
    if not value >= 0.0:
        raise ValueError(f"market.wholesale.look_ahead_hours must be >= 0, got {value}")
    return value


def resolve_look_ahead_resolution_hours(wholesale_config) -> float:
    """``market.wholesale.look_ahead_resolution_hours`` (default 6, the base's; null = hourly), validated."""
    value = (wholesale_config or {}).get("look_ahead_resolution_hours", 6)
    value = 1.0 if value is None else float(value)
    if not value > 0.0:
        raise ValueError(f"market.wholesale.look_ahead_resolution_hours must be > 0, got {value}")
    return value


def look_ahead_block_starts(kept, horizon, resolution_steps: int) -> pd.Series:
    """Each look-ahead snapshot of ``horizon`` (after ``kept``) mapped to the first snapshot of its block."""
    tail = pd.Index(horizon)[len(kept):]
    steps = max(1, int(resolution_steps))
    starts = [tail[(i // steps) * steps] for i in range(len(tail))]
    return pd.Series(starts, index=tail)


def coarsen_look_ahead(network, kept, horizon, resolution_steps: int):
    """Put ``network`` (snapshots = ``horizon``) onto the kept hours plus one snapshot per look-ahead block.

    Time-varying inputs become the block's mean weighted by the objective
    weighting (non-numeric frames take the block's first row); the block's
    snapshot weightings are the sums of its hours. Returns the new snapshot
    index; the network is unchanged (and ``horizon`` returned) when there is
    nothing to coarsen.
    """
    kept = pd.Index(kept)
    horizon = pd.Index(horizon)
    if int(resolution_steps) <= 1 or len(horizon) <= len(kept):
        return horizon
    group = look_ahead_block_starts(kept, horizon, resolution_steps)
    tail = group.index
    starts = pd.Index(pd.unique(group.to_numpy()), name=horizon.name)
    weights = network.snapshot_weightings.reindex(horizon).fillna(1.0)
    w_obj = weights["objective"].reindex(tail)
    frames = []
    for component in network.components:
        for attr, frame in component.dynamic.items():
            if frame is None or frame.empty or not len(frame.columns):
                continue
            head = frame.reindex(kept)
            body = frame.reindex(tail)
            numeric = body.select_dtypes(include="number")
            if numeric.shape[1] == body.shape[1]:
                agg = body.mul(w_obj, axis=0).groupby(group).sum().div(w_obj.groupby(group).sum(), axis=0)
            else:
                agg = body.groupby(group).first()
            frames.append((component, attr, pd.concat([head, agg.reindex(starts)])))
    block_weights = weights.reindex(tail).groupby(group).sum().reindex(starts)
    new_index = kept.append(starts)
    network.set_snapshots(new_index)
    network.snapshot_weightings.loc[starts, :] = block_weights.to_numpy()
    for component, attr, frame in frames:
        frame.index = new_index
        component.dynamic[attr] = frame
    return new_index


def rolling_window_horizons(snapshots, steps_per_window: int, look_ahead_steps: int = 0):
    """``(windows, horizons)``: the kept snapshots of each rolling window and the snapshots it is optimised over.

    ``horizons[i]`` is ``windows[i]`` followed by the next ``look_ahead_steps``
    snapshots, cut at the end of ``snapshots``; with no look-ahead the two lists
    are the same slices.
    """
    snapshots = pd.Index(snapshots)
    n_total = len(snapshots)
    steps_per_window = max(1, int(steps_per_window))
    look_ahead_steps = max(0, int(look_ahead_steps))
    windows, horizons = [], []
    for start_idx in range(0, n_total, steps_per_window):
        end_idx = min(start_idx + steps_per_window, n_total)
        windows.append(snapshots[start_idx:end_idx])
        horizons.append(snapshots[start_idx:min(end_idx + look_ahead_steps, n_total)])
    return windows, horizons


def keep_window_rows(frame, kept):
    """``frame`` restricted to the kept snapshots of a look-ahead window (unchanged when ``kept`` is None)."""
    if kept is None or frame is None or not len(frame):
        return frame
    return frame.loc[frame.index.isin(pd.Index(kept))]


#: Link ports past bus0 whose flows a rolling Stage 1 keeps window by window.
LINK_OUTPUT_PORTS = ("p1", "p2", "p3")


def write_link_flows(network, p0: pd.DataFrame, ports: dict | None = None) -> None:
    """Put Stage 1 link flows on ``network.links_t`` (item F14).

    ``p0`` is the flow at bus0 (the ``*_wholesale_links.csv`` table); ``ports``
    gives the solved ``p1``, ``p2``... where the solve returned them. A two-port
    link without a solved ``p1`` takes ``-efficiency x p0``, which is what PyPSA
    writes for one. Missing links get zero flow.
    """
    if p0 is None or p0.empty or not len(network.links):
        return
    names = network.links.index
    flows = p0.reindex(index=network.snapshots, columns=names).fillna(0.0)
    network.links_t.p0 = flows
    ports = dict(ports or {})
    if "p1" not in ports:
        efficiency = pd.to_numeric(network.links["efficiency"], errors="coerce").fillna(1.0)
        ports["p1"] = -flows.mul(efficiency, axis=1)
    for port, frame in ports.items():
        bus = f"bus{port[1:]}"
        if bus not in network.links.columns:
            continue
        connected = network.links[bus].fillna("").astype(str).str.strip().ne("")
        frame = frame.reindex(index=network.snapshots, columns=names).fillna(0.0)
        frame.loc[:, ~connected.to_numpy()] = 0.0
        network.links_t[port] = frame


def solve_rolling_day_ahead(
    network, wholesale_config, solver_name, solver_options, scenario_config, logger,
    optimize_kwargs=None, energy_floor=None, optimize=None,
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
        Wholesale config (``window_hours``, ``carry_soc``, ``look_ahead_hours``).
        With ``look_ahead_hours`` N > 0 each window is optimised over its
        ``window_hours`` plus the next N hours (truncated at the solve period's
        end) and only the first ``window_hours`` are kept (``resolve_look_ahead_hours``).
    solver_name : str
    solver_options : dict
    scenario_config : dict
    logger : logging.Logger
    energy_floor : pd.DataFrame, optional
        Hourly floor (MW) of the minimum-output energy-budget units from
        ``prepare_min_output_energy_budget``; each window holds their energy
        per block at or above it. None or empty: no budget.
    optimize : callable, optional
        What each window solves through (default ``_optimize_wholesale``); a
        ``wholesale_commitment.CommitmentOptimizer`` for priced unit commitment.

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
        Sum of objectives across all windows. With a look-ahead each window's
        objective covers its whole optimisation horizon, so the sum counts the
        overlapping hours twice (logged; a diagnostic, not a cost).
    """
    window_hours = int(wholesale_config.get("window_hours", 24))
    carry_soc = wholesale_config.get("carry_soc", True)
    look_ahead_hours = resolve_look_ahead_hours(wholesale_config)
    look_ahead_resolution = resolve_look_ahead_resolution_hours(wholesale_config)

    all_snapshots = network.snapshots.copy()
    n_total = len(all_snapshots)

    # Determine timestep resolution
    if n_total > 1:
        dt_hours = (all_snapshots[1] - all_snapshots[0]).total_seconds() / 3600
    else:
        dt_hours = 1.0
    steps_per_window = max(1, int(round(window_hours / dt_hours)))
    look_ahead_steps = int(round(look_ahead_hours / dt_hours))
    resolution_steps = max(1, int(round(look_ahead_resolution / dt_hours)))

    # Build list of window snapshot slices (the kept hours) and, with a
    # look-ahead, the longer horizon each window is optimised over.
    windows, horizons = rolling_window_horizons(all_snapshots, steps_per_window, look_ahead_steps)

    logger.info(
        f"Rolling day-ahead: {len(windows)} windows × {window_hours}h "
        f"({steps_per_window} timesteps each, {dt_hours}h resolution)"
        + (f", each optimised with a {look_ahead_hours:g}h look-ahead ({look_ahead_steps} timesteps, "
           f"truncated at the period end) and only its first {window_hours}h kept"
           + (f"; the look-ahead in blocks of {look_ahead_resolution:g}h "
              f"({steps_per_window + -(-look_ahead_steps // resolution_steps)} snapshots a window)"
              if resolution_steps > 1 else "")
           if look_ahead_steps else "")
    )
    if look_ahead_steps and energy_floor is not None and energy_floor.shape[1]:
        # Budget blocks are cut from each optimisation's start: they line up with
        # the kept hours only when window_hours is a whole number of blocks.
        misaligned = sorted({h for h in energy_floor.attrs.get("hours", {}).values()
                             if abs(window_hours / h - round(window_hours / h)) > 1e-9})
        if misaligned:
            logger.warning(
                f"Look-ahead with minimum-output energy blocks of {misaligned} h that do not divide "
                f"window_hours {window_hours}: a block spanning kept and look-ahead hours can defer "
                "its energy into the look-ahead"
            )

    # Build callback to fix interconnector flows at historical values
    ic_callback = _build_fix_interconnectors_callback(network, logger)
    ic_capacity_callback = build_interconnector_shared_capacity_callback(
        network, logger
    )
    ic_piecewise_cost_callback = build_piecewise_interconnector_cost_callback(
        network, logger
    )
    scarcity_callback = build_wholesale_scarcity_callback(network, wholesale_config, logger)
    # Central dispatch (copperplate_mode none): the secured boundary limits the
    # balancing stage enforces (item C1). Built once on the solve period.
    boundary_callback = build_central_boundary_callback(network, scenario_config, wholesale_config, logger)
    market_config = scenario_config.get("market", {})
    # Month-end storage condition (market.storage_soc): only the window holding
    # the last snapshot carries it, so the month hands back what it was given.
    month_end_soc_callback = build_month_end_soc_callback(
        network, market_config, all_snapshots[-1], logger
    )
    # Hydrogen store level carried between windows (market.hydrogen_store, D5);
    # the caller has already applied prepare_carried_stores to this network.
    store_names = carried_store_names(network, market_config)
    store_start = pd.to_numeric(network.stores.loc[store_names, "e_initial"], errors="coerce").fillna(0.0)
    month_end_store_callback = build_month_end_store_callback(
        network, market_config, store_start, all_snapshots[-1], logger
    )
    # A myopic window values hydrogen left at its end at V (market.hydrogen_store
    # .end_of_window_value_gbp_per_mwh_h2); objective only, stripped below. Rolling
    # Stage 1 only (copperplate and central dispatch alike); the BM keeps its band.
    store_value_callback = build_end_of_window_store_value_callback(
        market_config, store_start, logger
    )
    # Minimum output as energy per block, not per hour (min_output_energy_window_hours).
    energy_budget_callback = build_min_output_energy_callback(energy_floor)
    # Tie-break on |flow| of GB-internal DC links (16 Oct 2026): objective only,
    # stripped from each window's objective below (internal_dc_tie_break.py).
    tie_break_cost = resolve_internal_dc_tie_break(market_config)
    tie_break_callback = build_internal_dc_tie_break_callback(
        network, tie_break_cost, logger, stage="Stage 1 rolling day-ahead"
    )
    tie_break_gbp_total = 0.0
    tie_break_mwh_total = 0.0
    # Central dispatch only, off by default: thermal limits screened iteratively,
    # same optimum (market.wholesale.central_constraint_screening;
    # scripts/market/central_constraint_screening.py). None leaves the loop as it was.
    screen = None
    if (wholesale_config.get("central_constraint_screening") or {}).get("enabled"):
        from scripts.market.central_constraint_screening import build_central_constraint_screen
        screen = build_central_constraint_screen(
            network, wholesale_config, logger,
            accepted_conditions=(optimize_kwargs or {}).get("accepted_conditions", ("optimal",)),
        )
    large_hydro_state = initialise_large_hydro_storage_state(
        network, scenario_config.get("renewables", {}).get("hydro", {})
    )
    # Large-hydro water lent at the start is handed back by the run's end
    # (renewables.hydro.large_hydro.month_end); each window ends at or above the floor.
    hydro_run_initial = None if large_hydro_state is None else large_hydro_state.copy()
    hydro_net_inflow = large_hydro_net_inflow(network, scenario_config.get("renewables", {}).get("hydro", {}))

    # Wholesale SoC floor for pumped hydro (existing sites included, item C2):
    # preserves headroom for BM increases; the BM applies min_soc_fraction.
    pumped_cfg = scenario_config.get("renewables", {}).get("hydro", {}).get("pumped_hydro", {})
    ws_min_soc = pumped_cfg.get("wholesale_min_soc_fraction")
    log_hydro_constraint_setup(network, scenario_config, logger, context="Wholesale hydro",
                               min_soc_override=ws_min_soc)

    gen_dispatches = []
    su_dispatches = []
    su_store_legs = []      # gross charge and discharge legs, saved on the network
    su_dispatch_legs = []   # (market.balancing.storage_anchor: stage1_legs)
    link_dispatches = []
    link_port_flows = {}    # links_t p1, p2, ... per window (F14: saved on the network)
    price_dfs = []
    line_flow_dfs = []
    store_level_dfs = []
    objective_total = 0.0
    store_credit_mwh_total = 0.0
    store_credit_gbp_total = 0.0
    energy_budget_total = 0.0
    energy_delivered_total = 0.0
    prev_final_soc = None
    prev_store_levels = None
    period_end_report = {}  # seasonal trajectory (decision 33): soft-floor shortfall, storage end credit

    fallback_windows = []
    churn_windows = []  # Stage 1 storage churn diagnostic, one entry per window

    for i, (window_snaps, horizon_snaps) in enumerate(zip(windows, horizons)):
        window_started = time.perf_counter()
        window_start = window_snaps[0]
        window_end = window_snaps[-1]
        # The optimisation's last snapshot: window_end without a look-ahead.
        horizon_end = horizon_snaps[-1]

        if (i + 1) % 30 == 0 or i == 0 or i == len(windows) - 1:
            logger.info(
                f"Window {i + 1}/{len(windows)}: "
                f"{window_start} → {window_end} ({len(window_snaps)} snapshots)"
                + (f", optimised to {horizon_end} ({len(horizon_snaps)} snapshots)"
                   if len(horizon_snaps) > len(window_snaps) else "")
            )

        # Create a copy restricted to this window (plus its look-ahead)
        day_net = network.copy()
        day_net.set_snapshots(horizon_snaps)
        # Coarse look-ahead: the look-ahead hours as blocks (resolution 1: unchanged).
        horizon_snaps = coarsen_look_ahead(day_net, window_snaps, horizon_snaps, resolution_steps)

        # Carry over SoC from previous window
        if carry_soc and prev_final_soc is not None:
            for su_name, soc in prev_final_soc.items():
                if su_name in day_net.storage_units.index:
                    day_net.storage_units.loc[su_name, "state_of_charge_initial"] = soc
        # Hydrogen store: start where the previous window ended (D5).
        apply_carried_store_levels(day_net, prev_store_levels)

        hydro_callback = build_hydro_constraints_callback(
            day_net,
            scenario_config,
            large_hydro_storage_state=large_hydro_state,
            pumped_min_soc_override=ws_min_soc,
            large_hydro_end_storage=large_hydro_end_floor(
                hydro_net_inflow, hydro_run_initial, horizon_end
            ),
            logger=logger if i in (0, len(windows) - 1) else None,
        )

        # Solve down the fallback ladder (configured -> conservative barrier
        # -> dual simplex), re-solving once if material counterflows are locked.
        context = f"Wholesale window {i + 1}/{len(windows)} ({window_start})"
        if screen is not None:
            screen.start_window(f"window {i + 1}/{len(windows)} {window_start:%Y-%m-%d %H:%M}")
        try:
            outcome = solve_with_counterflow_lock(
                day_net,
                solver_name,
                solver_options,
                combine_extra_functionalities(
                    ic_callback, ic_capacity_callback, hydro_callback,
                    ic_piecewise_cost_callback, scarcity_callback,
                    month_end_soc_callback, month_end_store_callback,
                    boundary_callback, store_value_callback,
                    energy_budget_callback, tie_break_callback,
                ),
                logger,
                context,
                lock_counterflows=lock_material_counterflows,
                optimize=(optimize or _optimize_wholesale) if screen is None
                else screen.wrap(optimize or _optimize_wholesale),
                **(optimize_kwargs or {}),
            )
        except SolveFailedError as exc:
            logger.error(f"{context} failed: {exc}")
            # As in the BM: an IIS on request (wholesale.diagnostics.compute_iis,
            # or PYPSA_GB_COMPUTE_IIS=1 for a one-off run without a config edit).
            compute_iis = bool(
                (wholesale_config.get("diagnostics", {}) or {}).get("compute_iis", False)
            ) or os.environ.get("PYPSA_GB_COMPUTE_IIS") == "1"
            if compute_iis and str(solver_name).lower() == "gurobi" and str(exc.condition).lower() in {
                "infeasible", "infeasible_or_unbounded", "inf_or_unbd",
            }:
                from scripts.market.solve_balancing import _log_gurobi_iis
                _log_gurobi_iis(day_net, logger, context=context)
            raise RuntimeError(
                f"Wholesale window {i + 1} ({window_start}) failed: "
                f"{exc.status} ({exc.condition})"
            ) from exc
        if outcome.rung > 1:
            fallback_windows.append((i + 1, outcome.rung_name))
        if screen is not None:
            screen.end_window(day_net)

        # The end-of-window hydrogen credit is a valuation, not a cost: add it
        # back so the reported objective is what the window's dispatch costs.
        store_credit_mwh, store_credit_gbp = store_end_of_window_credit(day_net, store_value_callback)
        window_objective = float(day_net.objective) + store_credit_gbp
        # Seasonal trajectory (decision 33): the period-end storage credit and the
        # hydrogen soft-floor penalty are objective only too (0 for every other run).
        objective_only = objective_only_adjustment(day_net, month_end_soc_callback, month_end_store_callback)
        if objective_only:
            window_objective += objective_only
            for key, amount in trajectory_period_end_report(
                    day_net, month_end_store_callback, month_end_soc_callback).items():
                period_end_report[key] = period_end_report.get(key, 0.0) + amount
        store_credit_mwh_total += store_credit_mwh
        store_credit_gbp_total += store_credit_gbp
        # With a look-ahead only the kept hours count: the budget blocks that
        # start in them, the churn in them, and every result extracted below.
        kept = window_snaps if len(horizon_snaps) > len(window_snaps) else None
        # Internal HVDC tie-break: the whole horizon's term leaves the window's
        # objective; the kept hours' term is what the run reports.
        tie_break_gbp, _ = internal_dc_tie_break_cost(day_net, tie_break_callback)
        window_objective -= tie_break_gbp
        kept_tie_break_gbp, kept_tie_break_mwh = internal_dc_tie_break_cost(
            day_net, tie_break_callback, snapshots=kept)
        tie_break_gbp_total += kept_tie_break_gbp
        tie_break_mwh_total += kept_tie_break_mwh
        energy_budget, energy_delivered = min_output_energy_delivered(
            day_net, energy_budget_callback, snapshots=kept)
        energy_budget_total += energy_budget
        energy_delivered_total += energy_delivered
        objective_total += window_objective
        churn_windows.append(measure_storage_churn(
            day_net, logger, label=f"{window_start:%Y-%m-%d %H:%M}", snapshots=kept))

        # Extract positions
        gen_d, su_d, link_d = extract_wholesale_positions(day_net, logger)
        gen_dispatches.append(keep_window_rows(gen_d, kept))
        su_dispatches.append(keep_window_rows(su_d, kept))
        if len(day_net.storage_units) and len(day_net.storage_units_t.p_store):
            su_store_legs.append(keep_window_rows(day_net.storage_units_t.p_store.copy(), kept))
            su_dispatch_legs.append(keep_window_rows(day_net.storage_units_t.p_dispatch.copy(), kept))
        link_dispatches.append(keep_window_rows(link_d, kept))
        for port in LINK_OUTPUT_PORTS:
            frame = day_net.links_t.get(port) if hasattr(day_net.links_t, "get") else None
            if frame is not None and frame.shape[1]:
                link_port_flows.setdefault(port, []).append(keep_window_rows(frame.copy(), kept))

        # Extract marginal prices
        if hasattr(day_net, "buses_t") and len(day_net.buses_t.marginal_price) > 0:
            price_dfs.append(keep_window_rows(day_net.buses_t.marginal_price.copy(), kept))

        # Extract line flows for congestion analysis
        if hasattr(day_net, "lines_t") and len(day_net.lines_t.p0) > 0:
            line_flow_dfs.append(keep_window_rows(day_net.lines_t.p0.copy(), kept))

        # Save final SoC for next window (at the last KEPT snapshot)
        if carry_soc and len(day_net.storage_units) > 0:
            if (
                hasattr(day_net, "storage_units_t")
                and len(day_net.storage_units_t.state_of_charge) > 0
            ):
                prev_final_soc = (
                    day_net.storage_units_t.state_of_charge.loc[window_end].to_dict()
                    if kept is not None
                    else day_net.storage_units_t.state_of_charge.iloc[-1].to_dict()
                )
        if len(store_names):
            prev_store_levels = final_store_levels(
                day_net, store_names, at=window_end if kept is not None else None)
        if len(day_net.stores) and not day_net.stores_t.e.empty:
            store_level_dfs.append(keep_window_rows(day_net.stores_t.e.copy(), kept))

        large_hydro_state = update_large_hydro_storage_state(
            day_net,
            window_snaps,
            scenario_config,
            large_hydro_state,
        )

        # One timing line per window so a gate run can read solve seconds,
        # the rung used and the non-solve overhead (copy, build, extraction).
        logger.info(
            f"Window {i + 1}/{len(windows)} {window_start:%Y-%m-%d %H:%M}: "
            f"optimize {outcome.elapsed:.1f}s (rung {outcome.rung} "
            f"{outcome.rung_name}, {outcome.condition}), objective "
            f"£{window_objective:,.0f}"
            + (f" (hydrogen end-of-window credit {store_credit_mwh / 1e3:,.1f} GWh, "
               f"£{store_credit_gbp:,.0f} stripped)" if store_value_callback is not None else "")
            + (f", horizon {len(horizon_snaps)} snapshots" if kept is not None else "")
            + f", window total {time.perf_counter() - window_started:.1f}s"
        )

    log_fallback_summary(logger, "Wholesale rolling day-ahead", fallback_windows, len(windows))
    record_storage_churn(network, churn_windows, logger)
    if screen is not None:
        screening_summary = screen.summary()
        logger.info(f"Central constraint screening summary: {screening_summary}")
        network.meta = {**dict(getattr(network, "meta", None) or {}),
                        "central_constraint_screening": screening_summary}

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
    if su_store_legs:
        # Both legs, not only the net: with a calibrated churn share
        # (market.wholesale.storage_churn_share) Stage 1 charges and discharges a
        # unit at once, which net p hides, and a BM anchored on the net then
        # starts from energy Stage 1 burnt (stage1_legs in solve_balancing.py).
        network.storage_units_t.p_store = pd.concat(su_store_legs).reindex(
            columns=network.storage_units.index, fill_value=0.0
        )
        network.storage_units_t.p_dispatch = pd.concat(su_dispatch_legs).reindex(
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
    # Item F14 (13 Oct 2026): link flows on the saved network too, so a Stage 1
    # file balances on its own. They were only in *_wholesale_links.csv.
    write_link_flows(network, link_dispatch,
                     {port: pd.concat(frames) for port, frames in link_port_flows.items()})
    if tie_break_callback is not None:
        record_internal_dc_tie_break(
            network, "wholesale", tie_break_cost, tie_break_callback.links,
            tie_break_gbp_total, tie_break_mwh_total, logger,
        )
    if store_level_dfs:
        # Store levels across the run (the carried hydrogen store's trajectory).
        network.stores_t.e = pd.concat(store_level_dfs).reindex(
            columns=network.stores.index, fill_value=0.0
        )
        if len(store_names):
            level = network.stores_t.e[list(store_names)].sum(axis=1)
            logger.info(
                f"Hydrogen store over the run: start {store_start.sum() / 1e3:,.1f} GWh, "
                f"min {level.min() / 1e3:,.1f}, max {level.max() / 1e3:,.1f}, "
                f"end {level.iloc[-1] / 1e3:,.1f} GWh"
            )
    if store_value_callback is not None and getattr(store_value_callback, "time_varying", False):
        # Seasonal trajectory (decision 33): the value differs window by window.
        applied = [v for _, v in getattr(store_value_callback, "applied", [])]
        logger.info(
            f"Hydrogen store end-of-window value over the run (seasonal trajectory): {len(windows)} windows "
            f"credited at the annual store value, "
            + (f"{min(applied):.2f}-{max(applied):.2f} GBP/MWh_H2 (mean {np.mean(applied):.2f}), " if applied else "")
            + f"{store_credit_mwh_total / 1e3:,.1f} GWh, £{store_credit_gbp_total:,.0f} stripped from the objective"
        )
    if period_end_report:
        logger.info(
            "Seasonal trajectory over the run (objective-only terms, stripped): "
            + ", ".join(f"{k} {v:,.1f}" for k, v in period_end_report.items())
        )
    elif store_value_callback is not None:
        logger.info(
            f"Hydrogen store end-of-window value over the run: {len(windows)} windows credited at "
            f"{store_value_callback.value:.2f} GBP/MWh_H2, {store_credit_mwh_total / 1e3:,.1f} GWh "
            f"(mean {store_credit_mwh_total / max(len(windows), 1) / 1e3:,.1f} GWh a window end), "
            f"£{store_credit_gbp_total:,.0f} stripped from the objective"
        )
    if energy_budget_callback is not None:
        logger.info(
            f"Minimum-output energy budget over the run: {len(energy_budget_callback.units)} units, "
            f"budget {energy_budget_total / 1e3:,.1f} GWh, delivered {energy_delivered_total / 1e3:,.1f} GWh"
        )

    logger.info(
        f"Rolling day-ahead complete: {len(windows)} windows, "
        f"total objective £{objective_total:,.2f}"
        + (f" (each window's objective covers its {look_ahead_hours:g}h look-ahead too, "
           "so overlapping hours are counted twice)" if look_ahead_steps else "")
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
        require_pypsa_branch_units(network, "Stage 1 wholesale solve",
                                   scenario_config.get("network_model"), logger)
        # Decision 29 (2050 only): a TYNDP price sensitivity re-prices the shared
        # network's interconnectors; a no-op unless market.foreign_price_override is set.
        from scripts.interconnectors.foreign_price_override import apply_foreign_price_override
        apply_foreign_price_override(network, scenario_config, snakemake.input, logger)
        # CP30's storage round trips, both stages alike (market.storage_round_trip_override).
        from scripts.market.market_utils import apply_storage_round_trip_override
        apply_storage_round_trip_override(network, scenario_config.get("market", {}), logger)
        # CP.28-30 calibration (17 Oct 2026): the exogenous hydrogen offtake scaled at solve time, both stages alike
        # (market.hydrogen_offtake_scale; absent = the build's offtake).
        from scripts.market.market_utils import apply_hydrogen_offtake_scale
        apply_hydrogen_offtake_scale(network, scenario_config.get("market", {}), logger)
        # 2050 price calibration (25 Oct 2026): hydrogen at an external long-run cost, all three stages alike
        # (market.hydrogen_supply_price; absent = the build's reformer cost).
        from scripts.market.market_utils import apply_hydrogen_supply_price
        apply_hydrogen_supply_price(network, scenario_config.get("market", {}), logger)
        market_config = scenario_config.get("market", {})
        wholesale_config = market_config.get("wholesale", {})
        # Stage 1 test variant s1_nh (16 Oct 2026): the single-basis interconnector trading hurdle
        # re-priced on the shared build; a no-op unless
        # market.wholesale.interconnector_hurdle_override.enabled (scripts/market/interconnector_hurdle.py).
        from scripts.market.interconnector_hurdle import apply_interconnector_hurdle_override
        apply_interconnector_hurdle_override(network, wholesale_config, logger)
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
        # Controlled adequacy comparisons may replay a previously saved
        # copperplate schedule, preventing LP tie-breaking/storage choices from
        # contaminating a topology comparison.
        position_source = wholesale_config.get("position_source", {}) or {}
        if position_source:
            # Normal wholesale solves restrict snapshots below; replay must do it
            # before validating the saved schedule against this network.
            apply_solve_period(network, scenario_config, logger)
            required = (
                "reference_wholesale_dispatch_csv",
                "reference_wholesale_storage_csv",
                "reference_wholesale_links_csv",
                "reference_wholesale_price_csv",
            )
            missing = [name for name in required if not hasattr(snakemake.input, name)]
            if missing:
                raise ValueError("Wholesale position source is configured but inputs are missing: " + ", ".join(missing))
            reference_gen = pd.read_csv(snakemake.input.reference_wholesale_dispatch_csv, index_col=0, parse_dates=True)
            reference_su = pd.read_csv(snakemake.input.reference_wholesale_storage_csv, index_col=0, parse_dates=True)
            reference_links = pd.read_csv(snakemake.input.reference_wholesale_links_csv, index_col=0, parse_dates=True)
            reference_price = pd.read_csv(snakemake.input.reference_wholesale_price_csv, index_col=0, parse_dates=True)
            expected = pd.DatetimeIndex(network.snapshots)
            for label, frame, columns in (
                ("generators", reference_gen, network.generators.index),
                ("storage units", reference_su, network.storage_units.index),
                ("links", reference_links, network.links.index),
            ):
                if not frame.index.equals(expected):
                    raise ValueError(f"Reference wholesale {label} snapshots do not match this scenario")
                unknown = set(frame.columns) - set(columns)
                if unknown:
                    raise ValueError(
                        f"Reference wholesale {label} contains components absent "
                        f"from this scenario: {', '.join(sorted(unknown)[:10])}"
                    )
                target_only = set(columns) - set(frame.columns)
                if target_only:
                    logger.warning(
                        "Reference wholesale %s omits %s target-only components; their replayed wholesale position is zero.",
                        label,
                        len(target_only),
                    )
            if not reference_price.index.equals(expected):
                raise ValueError("Reference wholesale price snapshots do not match this scenario")
            replay_gen = reference_gen.reindex(columns=network.generators.index, fill_value=0.0)
            replay_su = reference_su.reindex(columns=network.storage_units.index, fill_value=0.0)
            replay_links = reference_links.reindex(columns=network.links.index, fill_value=0.0)
            replay_gen.to_csv(snakemake.output.wholesale_dispatch_csv)
            replay_su.to_csv(snakemake.output.wholesale_storage_csv)
            replay_links.to_csv(snakemake.output.wholesale_links_csv)
            reference_price.to_csv(snakemake.output.wholesale_price_csv)
            network.generators_t.p = replay_gen
            network.storage_units_t.p = replay_su
            write_link_flows(network, replay_links)
            save_network(network, snakemake.output.network, custom_logger=logger)
            logger.warning("Replayed fixed wholesale positions from scenario '%s'; no wholesale optimisation was run.", position_source.get("scenario"))
            raise SystemExit(0)

        validate_network_costs(network, logger)

        # ── Marginal-cost basis (as_built by default) ────────────────────
        # Item N1: a public operator dispatches on fuel, carbon and variable
        # O&M, not on certificate income it pays to itself. Applied before the
        # forecast error and the copperplate step so both institutions see one
        # basis. See scripts/market/marginal_cost_basis.py.
        mc_audit = None
        if hasattr(snakemake.input, "marginal_costs_csv") and Path(str(snakemake.input.marginal_costs_csv)).exists():
            mc_audit = pd.read_csv(snakemake.input.marginal_costs_csv)
        basis_block = wholesale_config.get("marginal_cost_basis", {})
        basis_strikes = None
        if needs_strike_table(basis_block):
            from scripts.market.cfd_bm_bids import load_scenario_strike_table

            basis_strikes = load_scenario_strike_table(scenario_config)
        apply_marginal_cost_basis(network, basis_block, logger, audit=mc_audit, strike_table=basis_strikes)

        # ── Curtailment order among zero-cost renewables (26 Sep 2026) ───
        # contract_aware prices each CfD unit where its support terms make it
        # back off (merchant 0, AR4+ just below 0, older contracts at minus
        # strike). See scripts/market/curtailment_order.py.
        order_block = wholesale_config.get("renewable_curtailment_order", {}) or {}
        basis_mode = str((wholesale_config.get("marginal_cost_basis", {}) or {}).get("mode", "as_built"))
        if str(order_block.get("mode", "as_built")) != "as_built" and basis_mode != "as_built":
            logger.info("Curtailment order: marginal_cost_basis %s dispatches on cost; contract-aware order not applied",
                        basis_mode)
        elif str(order_block.get("mode", "as_built")) != "as_built":
            from scripts.market.cfd_bm_bids import load_scenario_strike_table
            from scripts.market.curtailment_order import apply_curtailment_order

            apply_curtailment_order(network, order_block, load_scenario_strike_table(scenario_config), logger,
                                    copperplate_mode=wholesale_config.get("copperplate_mode"))

        # ── Day-ahead forecast error (off by default) ────────────────────
        # Stage 1 schedules on a forecast; Stage 2 reloads the network and so
        # keeps the outturn, and its redispatch against this schedule is the
        # cost of resolving the error. See scripts/market/forecast_error.py.
        forecast_error_summary = apply_day_ahead_forecast_error(
            network, wholesale_config.get("forecast_error", {}), logger
        )
        if forecast_error_summary:
            logger.info(
                "Forecast error applied to "
                f"{len(forecast_error_summary.get('technologies', {}))} technologies"
                + (" and demand" if forecast_error_summary.get("demand") else "")
            )

        # ── Copperplate ──────────────────────────────────────────────────
        # "collapse" (default): solve a copy with the mainland synchronous
        # sub-network collapsed onto one bus (exact; scripts/market/copperplate.py).
        # The collapse happens just before the solve so every generator-level
        # pre-solve step below still sees the real buses (apply_load_shedding_limits
        # caps load shedding to same-bus demand).
        # "relax": legacy 1e6 MVA ratings on every line and transformer.
        # "none": no copperplate at all. Stage 1 meets demand at least cost on
        # the real network, which makes it a central dispatch rather than a
        # market: there is no position for Stage 2 to be paid to deviate from,
        # so the balancing stage should find almost nothing to redispatch and
        # its constraint cost falls out near zero rather than being asserted.
        copperplate_mode = str(
            wholesale_config.get("copperplate_mode", "collapse")
        ).lower()
        if copperplate_mode == "relax":
            apply_copperplate_relaxation(network, wholesale_config, logger)
        elif copperplate_mode == "none":
            # The two stages must see the same network or the balancing stage
            # will redispatch against ratings Stage 1 never had. These are the
            # preparations solve_balancing applies; the scenario's 220 kV
            # constraint floor and its explicit line overrides live here.
            logger.info("=" * 80)
            logger.info("CENTRAL DISPATCH: LEAST COST ON THE REAL NETWORK (NO COPPERPLATE)")
            logger.info("=" * 80)
            # 18 Oct 2026: the balancing stage's network delivery case (outage replay of NESO's
            # day-ahead boundary limits, delayed projects) applies here too, in the BM's order,
            # so a central schedule is made on the network the BM enforces.
            scenario_config = apply_central_network_delivery(
                network, scenario_config, logger,
                projects_file=getattr(snakemake.input, "appendix_b_projects", None))
            apply_line_rating_overrides(network, scenario_config, logger)
            apply_outage_schedule(network, scenario_config, logger)
            apply_transmission_relaxation(network, scenario_config, logger)
        elif copperplate_mode != "collapse":
            raise ValueError(
                "market.wholesale.copperplate_mode must be 'collapse', 'relax' or "
                f"'none', got {copperplate_mode!r}"
            )
        apply_storage_dispatch_cost_floor(network, wholesale_config, logger)
        apply_storage_churn_share(network, wholesale_config, logger)
        apply_churn_sink(network, wholesale_config, logger)

        # ── Numerical conditioning ───────────────────────────────────────
        improve_numerical_conditioning(network, logger)

        # ── Apply solve period ───────────────────────────────────────────
        apply_solve_period(network, scenario_config, logger)
        if copperplate_mode == "none":
            # Central dispatch on the real network: ETYS seasonal ratings apply
            # (the copperplate modes have no binding branch limits).
            apply_seasonal_line_ratings(network, scenario_config, logger)
        apply_load_shedding_limits(network, logger)
        # Must-take units (market.must_take_generation, e.g. CP30 rooftop solar) run at
        # their availability: never curtailed. scripts/market/must_take.py.
        from scripts.market.must_take import apply_must_take_stage1
        apply_must_take_stage1(network, market_config, logger)
        enforce_feasible_generator_operating_bounds(network, logger)
        # One starting state of charge for both stages (solve_balancing applies
        # the same helper); see market_utils.resolve_storage_soc_config.
        apply_initial_storage_soc(network, market_config, logger)
        # The hydrogen store's level is carried across windows from one starting
        # level (decision D5); solve_balancing applies the same helper.
        store_start = prepare_carried_stores(network, market_config, logger)
        # A national day-ahead market does not see GB-internal island HVDC
        # limits (item C3); the BM keeps them.
        relaxed_island_links = (
            relax_island_hvdc_links(network, wholesale_config, logger)
            if copperplate_mode in ("collapse", "relax") else []
        )

        logger.info(
            f"Optimization will run for {len(network.snapshots)} snapshots "
            f"(copperplate mode: {copperplate_mode})"
        )

        # ── Solve mode (LP / MILP) ───────────────────────────────────────
        global_solve_mode = get_solve_mode_from_config()
        logger.info(f"Solve mode: {global_solve_mode}")

        uc_original_bounds = capture_bounds(network)
        wholesale_uc_enabled = apply_wholesale_unit_commitment(
            network, wholesale_config, logger
        )
        # unit_commitment.method milp_fixed / linearized (16 Oct 2026, variant s1_uc): each window
        # solves through a wrapper that yields prices (scripts/market/wholesale_commitment.py);
        # method milp (the default) is the overlay as before, one MILP and no prices.
        commitment_plan = configure_commitment(network, wholesale_config, uc_original_bounds, logger)
        wholesale_optimize = _optimize_wholesale
        if commitment_plan is not None:
            step_h = ((network.snapshots[1] - network.snapshots[0]).total_seconds() / 3600
                      if len(network.snapshots) > 1 else 1.0)
            wholesale_optimize = CommitmentOptimizer(
                commitment_plan, _optimize_wholesale,
                max(1, int(round(float(wholesale_config.get("window_hours", 24)) / step_h))), logger,
            )
        if wholesale_uc_enabled and global_solve_mode == "LP":
            logger.info(
                "Global solve_mode is LP, but wholesale unit commitment is "
                "enabled explicitly for Stage 1"
            )

        # Remove must-run if configured
        remove_must_run = scenario_config.get("optimization", {}).get("remove_must_run", False)
        if remove_must_run and "p_min_pu" in network.generators.columns:
            network.generators["p_min_pu"] = 0.0
            logger.info("Removed must-run constraints (p_min_pu = 0)")

        # Minimum output as an energy budget for the listed carriers: the hourly
        # floor (already clipped to availability) becomes energy per block and
        # p_min_pu is zeroed, before the collapse so the copy inherits it.
        energy_floor = prepare_min_output_energy_budget(network, scenario_config, logger)

        # ── Configure solver ─────────────────────────────────────────────
        solver_name, solver_options = configure_solver(
            network, solver_name, solver_options, logger
        )
        solver_options = apply_wholesale_solver_overrides(
            solver_options, wholesale_config, logger
        )
        optimize_kwargs = solver_io_kwargs(scenario_config, solver_name, logger)
        # solve.solver.accepted_conditions widens what the fallback ladder takes from
        # a rung. The central dispatch LP with barrier and no crossover often
        # ends "suboptimal" (Gurobi: optimality tolerances not met) within
        # 0.1% of the fully converged objective in a third of the time; a
        # scenario may choose to accept that, and every window still logs the
        # condition it stopped on. Default is optimal only, as before.
        accepted = scenario_config.get("solve", {}).get("solver", {}).get("accepted_conditions")
        if accepted:
            optimize_kwargs["accepted_conditions"] = tuple(str(c) for c in accepted)
            logger.info(f"Solver: accepting termination conditions {optimize_kwargs['accepted_conditions']}")
        if optimize_kwargs:
            logger.info(f"Solver model interface: {optimize_kwargs}")

        # ── Exact copperplate: collapse the mainland onto one bus ────────
        # Solve on the collapsed copy; expand the results onto the original
        # afterwards so the saved network keeps every bus, line and transformer.
        original_network = None
        collapse_mapping = None
        if copperplate_mode == "collapse":
            original_network = network
            network, collapse_mapping = collapse_mainland(network, logger)

        # ── Temporal resolution (market.wholesale.temporal_resolution_hours) ──
        # the calibration to NESO's CP.28-30 market run (17 Oct 2026): NESO's CP.28-30 market run is a
        # 3-hour PLEXOS run. With R > 1 the solve network goes onto R-hour blocks
        # (block-mean inputs, snapshot weightings = block hours) and back to hourly
        # after the solve, outputs held within each block (scripts/market/temporal_resolution.py).
        # 1 (the default): nothing changes.
        temporal_hours = resolve_temporal_resolution_hours(wholesale_config)
        temporal_blocks = None
        if temporal_hours > 1:
            check_temporal_resolution(temporal_hours, wholesale_config,
                                      commitment=bool(wholesale_uc_enabled or commitment_plan is not None))
            temporal_blocks = aggregate_to_blocks(network, temporal_hours, logger)
            energy_floor = aggregate_frame_to_blocks(energy_floor, temporal_blocks)

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
                    optimize_kwargs=optimize_kwargs,
                    energy_floor=energy_floor,
                    optimize=wholesale_optimize,
                )
            )

            solve_time = time.time() - solve_start
            logger.info(f"Rolling day-ahead solve completed in {solve_time:.2f}s")
            logger.info(f"Total wholesale system cost: £{objective:,.2f}")

            if temporal_blocks is not None:
                # Block results back on the hourly snapshots (power held within each block).
                gen_dispatch, su_dispatch, link_dispatch, prices = (
                    expand_block_rows(frame, temporal_blocks)
                    for frame in (gen_dispatch, su_dispatch, link_dispatch, prices)
                )
                restore_hourly(network, temporal_blocks, logger)
            if original_network is not None:
                expand_results(original_network, network, collapse_mapping, logger)
                collapsed_meta = dict(getattr(network, "meta", None) or {})
                network = original_network
                for key in (STORAGE_CHURN_META_KEY, INTERNAL_DC_TIE_BREAK_META_KEY):
                    if collapsed_meta.get(key) is not None:
                        network.meta = {**dict(network.meta or {}), key: collapsed_meta[key]}
                prices = network.buses_t.marginal_price
            record_temporal_resolution(network, temporal_blocks)
            restore_island_hvdc_links(network, relaxed_island_links, logger)
            if commitment_plan is not None:
                # The saved Stage 1 network keeps the LP inputs; the commitment run is in its meta.
                commitment_plan.restore(network, logger)
                network.meta = {**dict(getattr(network, "meta", None) or {}),
                                "stage1_unit_commitment": wholesale_optimize.summary()}
                logger.info(f"Unit commitment summary: {wholesale_optimize.summary()}")

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
                _attach_hydrogen_price(price_df, prices, network, logger)
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
            ic_capacity_callback = build_interconnector_shared_capacity_callback(
                network, logger
            )
            ic_piecewise_cost_callback = build_piecewise_interconnector_cost_callback(
                network, logger
            )
            # Apply wholesale SoC floor for pumped hydro (existing sites included)
            pumped_cfg_single = scenario_config.get("renewables", {}).get("hydro", {}).get("pumped_hydro", {})
            ws_min_soc_single = pumped_cfg_single.get("wholesale_min_soc_fraction")
            log_hydro_constraint_setup(
                network,
                scenario_config,
                logger,
                context="Wholesale hydro",
                min_soc_override=ws_min_soc_single,
            )
            hydro_run_initial_single = initialise_large_hydro_storage_state(
                network, scenario_config.get("renewables", {}).get("hydro", {})
            )
            hydro_callback = build_hydro_constraints_callback(
                network,
                scenario_config,
                pumped_min_soc_override=ws_min_soc_single,
                large_hydro_end_storage=large_hydro_end_floor(
                    large_hydro_net_inflow(network, scenario_config.get("renewables", {}).get("hydro", {})),
                    hydro_run_initial_single,
                    network.snapshots[-1],
                ),
                logger=logger,
            )
            scarcity_callback = build_wholesale_scarcity_callback(
                network, wholesale_config, logger
            )
            month_end_soc_callback = build_month_end_soc_callback(
                network, market_config, network.snapshots[-1], logger
            )
            month_end_store_callback = build_month_end_store_callback(
                network, market_config, store_start, network.snapshots[-1], logger
            )
            boundary_callback = build_central_boundary_callback(
                network, scenario_config, wholesale_config, logger
            )
            if resolve_store_level_config(market_config)["end_of_window_value_gbp_per_mwh_h2"]:
                # One window sees the whole period, so the month-end rule is its terminal condition.
                logger.info("Hydrogen store end-of-window value: not applied in single mode (no window ends)")
            # Seasonal trajectory (decision 33) in single mode: the storage end value and the
            # soft floor act; their objective-only terms are logged, not stripped (declared).
            energy_budget_callback = build_min_output_energy_callback(energy_floor)
            # Tie-break on |flow| of GB-internal DC links (internal_dc_tie_break.py).
            tie_break_cost = resolve_internal_dc_tie_break(market_config)
            tie_break_callback = build_internal_dc_tie_break_callback(
                network, tie_break_cost, logger, stage="Stage 1 full period"
            )

            try:
                outcome = solve_with_counterflow_lock(
                    network,
                    solver_name,
                    solver_options,
                    combine_extra_functionalities(
                        ic_callback, ic_capacity_callback, hydro_callback,
                        ic_piecewise_cost_callback, scarcity_callback,
                        month_end_soc_callback, month_end_store_callback,
                        boundary_callback, energy_budget_callback,
                        tie_break_callback,
                    ),
                    logger,
                    "Wholesale full-period solve",
                    lock_counterflows=lock_material_counterflows,
                    optimize=wholesale_optimize,
                    **optimize_kwargs,
                )
            except SolveFailedError as exc:
                logger.error(f"Wholesale optimization failed: {exc}")
                raise RuntimeError(
                    f"Wholesale optimization failed: "
                    f"{exc.status} ({exc.condition})"
                ) from exc

            solve_time = time.time() - solve_start
            logger.info(f"Wholesale solve completed in {solve_time:.2f}s")
            logger.info(
                f"Status: {outcome.status}, Condition: {outcome.condition}, "
                f"solver rung {outcome.rung} ({outcome.rung_name}), "
                f"optimize {outcome.elapsed:.1f}s"
            )
            single_objective_only = objective_only_adjustment(
                network, month_end_soc_callback, month_end_store_callback)
            if single_objective_only:
                logger.info(
                    f"Seasonal trajectory: objective-only terms in this objective £{-single_objective_only:,.0f} "
                    f"(not stripped in single mode)")
            if energy_budget_callback is not None:
                budget, delivered = min_output_energy_delivered(network, energy_budget_callback)
                logger.info(
                    f"Minimum-output energy budget: {len(energy_budget_callback.units)} units, "
                    f"budget {budget / 1e3:,.1f} GWh, delivered {delivered / 1e3:,.1f} GWh"
                )

            churn = measure_storage_churn(network, logger, label="full period")
            tie_break_gbp, tie_break_mwh = internal_dc_tie_break_cost(network, tie_break_callback)
            # Block results back on the hourly snapshots (temporal_resolution_hours > 1 only).
            restore_hourly(network, temporal_blocks, logger)
            if original_network is not None:
                expand_results(original_network, network, collapse_mapping, logger)
                network = original_network
            record_temporal_resolution(network, temporal_blocks)
            restore_island_hvdc_links(network, relaxed_island_links, logger)
            if commitment_plan is not None:
                # The saved Stage 1 network keeps the LP inputs; the commitment run is in its meta.
                commitment_plan.restore(network, logger)
                network.meta = {**dict(getattr(network, "meta", None) or {}),
                                "stage1_unit_commitment": wholesale_optimize.summary()}
                logger.info(f"Unit commitment summary: {wholesale_optimize.summary()}")
            record_storage_churn(network, [churn], logger)
            if tie_break_callback is not None:
                # Objective only: out of the reported (and saved) Stage 1 cost.
                if getattr(network, "_objective", None) is not None:
                    network._objective = float(network._objective) - tie_break_gbp
                record_internal_dc_tie_break(
                    network, "wholesale", tie_break_cost, tie_break_callback.links,
                    tie_break_gbp, tie_break_mwh, logger,
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
                _attach_hydrogen_price(price_df, prices, network, logger)
                price_df.to_csv(snakemake.output.wholesale_price_csv)
                logger.info(f"Saved: {snakemake.output.wholesale_price_csv}")
                logger.info(
                    f"Wholesale price: mean £{uniform_price.mean():.2f}/MWh, "
                    f"max spread £{max_spread:.4f}/MWh "
                    + ("(nodal: central dispatch prices the real network)"
                       if copperplate_mode == "none" else "(should be ~0 for copperplate)")
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
