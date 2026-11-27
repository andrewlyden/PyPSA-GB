"""
Solve Balancing Mechanism  Stage 2 of Two-Stage Market Dispatch

Starting from the wholesale (copperplate) dispatch positions, re-solves the
network with full transmission constraints. Generators and storage are
anchored to their wholesale positions via increase/decrease variables with
separate bid/offer prices, mimicking the GB Balancing Mechanism.

The optimisation minimises total redispatch cost:
    min  _{g,t}  offer_price[g]  increase[g,t] + bid_price[g]  decrease[g,t]

where bid_price is always in "ESO cost" convention: positive means it costs
the ESO money to turn the generator down.  ELEXON raw bids (negative = ESO
pays generator) are negated at load time in market_utils._load_elexon_bid_offer,
and derived thermal bids are negative (the ESO receives the fuel the unit saves;
market_utils.DERIVED_BID_CONVENTIONS).  Hydrogen links (electrolysers, H2
turbines) are anchored the same way and priced at the Stage 1 hydrogen value
(``_hydrogen_link_prices``).

Subject to:
    p[g,t] = p_wholesale[g,t] + increase[g,t] - decrease[g,t]    g,t
    increase[g,t]  0,  decrease[g,t]  0                         g,t
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
  - scripts/market/solve_wholesale.py  Stage 1 (copperplate)
  - scripts/market/market_utils.py  Shared utilities
"""

import pypsa
import logging
import re
import copy
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network, save_network
from scripts.network_build.branch_units import require_pypsa_branch_units
from scripts.solve.solve_network import (
    validate_network_costs,
    apply_load_shedding_limits,
    enforce_feasible_generator_operating_bounds,
    apply_transmission_relaxation,
    apply_line_rating_overrides,
    apply_outage_schedule,
    apply_seasonal_line_ratings,
    _build_neso_boundary_constraints_callback,
    improve_numerical_conditioning,
    configure_solver,
    get_solve_mode_from_config,
)
from scripts.market.commitment_proxy import apply_commitment_proxy
from scripts.generators.thermal_ramping import widen_ramp_limits_to_availability
from scripts.market.market_utils import (
    HYDROGEN_PRICE_COLUMN,
    SolveFailedError,
    align_redispatch_generators,
    apply_carried_store_levels,
    apply_initial_storage_soc,
    apply_storage_dispatch_cost_floor,
    calculate_bid_offer_prices,
    fix_churn_sink,
    compute_link_redispatch_volumes,
    compute_redispatch_volumes,
    build_min_output_energy_callback,
    estimate_hydrogen_value_from_schedule,
    final_store_levels,
    hydrogen_link_prices,
    hydrogen_links,
    identify_congested_boundaries,
    log_fallback_summary,
    min_output_energy_delivered,
    prepare_bm_min_output_energy_budget,
    prepare_carried_stores,
    resolve_hydrogen_link_config,
    solve_with_counterflow_lock,
    solver_io_kwargs,
)
from scripts.market.diagnose_storage_degeneracy import reconstruct_state_of_charge
from scripts.market.cfd_bm_bids import AUDIT_COLUMNS as CONTRACT_VALUE_AUDIT_COLUMNS
from scripts.market.cfd_bm_bids import resolve_general_turn_up_floor
from scripts.market.market_accounting import storage_simultaneous_burn
from scripts.market.solve_wholesale import apply_solve_period
from scripts.market.adequacy_controls import apply_adequacy_relaxation
from scripts.solve.hydro_constraints import (
    build_hydro_constraints_callback,
    combine_extra_functionalities,
    initialise_large_hydro_storage_state,
    large_hydro_end_floor,
    large_hydro_net_inflow,
    log_hydro_constraint_setup,
    update_large_hydro_storage_state,
)
from scripts.interconnectors.market_coupling import (
    build_piecewise_interconnector_cost_callback,
    lock_material_counterflows,
    stage1_interconnector_trade_value,
    build_interconnector_shared_capacity_callback,
)
from scripts.market.interconnector_counter_trade import (
    build_counter_trade_callback,
    no_wheeling_solver_options,
    counter_trade_excess_penalty,
    counter_trade_links,
    resolve_counter_trade_config,
    stamp_counter_trade_attributes,
    summarise_counter_trade,
)
from scripts.market.internal_dc_tie_break import (
    build_internal_dc_tie_break_callback,
    internal_dc_tie_break_cost,
    record_internal_dc_tie_break,
    resolve_internal_dc_tie_break,
)


def prepare_interconnector_counter_trade(network, balancing_config: dict, wholesale_links: pd.DataFrame,
                                         logger) -> list:
    """Stamp the counter-trade premium and flow limit on the links the BM may re-trade.

    Only when re-trading is on (``interconnector_retrading``) and
    ``interconnector_counter_trade.enabled``; the saved balancing network then
    carries the premium the re-trading valuation reads. Returns the link names.
    """
    cfg = resolve_counter_trade_config(balancing_config)
    if not cfg["enabled"] or resolve_fix_interconnectors(balancing_config):
        return []
    names = counter_trade_links(network, wholesale_links.columns)
    stamp_counter_trade_attributes(network, cfg, names)
    return names


def _log_counter_trade_summary(network, physical_links, wholesale_links, callback, logger) -> None:
    """One log block: counter-trade volume, premium and how often the band bound."""
    if callback is None or physical_links is None or physical_links.empty:
        return
    summary = summarise_counter_trade(
        physical_links, wholesale_links.reindex(physical_links.index), network.links,
        getattr(callback, "bounds", None), ask_t=network.links_t.get("marginal_cost"),
    )
    if summary.empty:
        return
    logger.info(
        "Interconnector counter-trade: %.1f GWh up (more import / less export), %.1f GWh down, "
        "premium GBP %.0f, %d link-hours at the flow limit, %d link-hours gated by the spread rule",
        summary["up_MWh"].sum() / 1e3, summary["down_MWh"].sum() / 1e3, summary["premium_GBP"].sum(),
        int(summary.get("hours_at_bound", pd.Series(dtype=float)).sum()),
        int(summary.get("hours_gated", pd.Series(dtype=float)).sum()),
    )
    for row in summary.itertuples(index=False):
        logger.info(
            "  %s: up %.1f MWh, down %.1f MWh, premium GBP %.0f, share %.2f%s",
            row.link, row.up_MWh, row.down_MWh, row.premium_GBP, row.max_share,
            f", at bound {row.hours_at_bound} h, gated {row.hours_gated} h"
            if hasattr(row, "hours_at_bound") else "",
        )


def _log_gurobi_iis(network, logger, context: str, max_items: int = 200) -> None:
    """Compute and log a readable IIS for a failed Gurobi optimisation."""
    model = getattr(network, "model", None)
    solver_model = getattr(model, "solver_model", None)
    if solver_model is None or not hasattr(solver_model, "computeIIS"):
        logger.warning("IIS unavailable for %s: no compatible solver model", context)
        return

    try:
        logger.info("Computing Gurobi IIS for %s...", context)
        solver_model.computeIIS()

        constraints = [
            constr for constr in solver_model.getConstrs() if constr.IISConstr
        ]
        bound_vars = [
            var
            for var in solver_model.getVars()
            if bool(var.IISLB) or bool(var.IISUB)
        ]
        logger.error(
            "IIS for %s contains %d constraints and %d variable bounds",
            context,
            len(constraints),
            len(bound_vars),
        )

        def _label(raw_name: str, prefix: str):
            match = re.fullmatch(rf"{prefix}(\d+)", str(raw_name))
            return int(match.group(1)) if match else None

        for constr in constraints[:max_items]:
            label = _label(constr.ConstrName, "c")
            position = (
                model.constraints.get_label_position(label)
                if label is not None
                else (None, None)
            )
            logger.error(
                "  IIS constraint %s -> %s; sense=%s rhs=%s",
                constr.ConstrName,
                position,
                constr.Sense,
                constr.RHS,
            )
        if len(constraints) > max_items:
            logger.error("  ... %d further IIS constraints omitted", len(constraints) - max_items)

        for var in bound_vars[:max_items]:
            label = _label(var.VarName, "x")
            position = (
                model.variables.get_label_position(label)
                if label is not None
                else (None, None)
            )
            logger.error(
                "  IIS bound %s -> %s; lb=%s%s ub=%s%s",
                var.VarName,
                position,
                var.LB,
                " [IIS]" if var.IISLB else "",
                var.UB,
                " [IIS]" if var.IISUB else "",
            )
        if len(bound_vars) > max_items:
            logger.error("  ... %d further IIS bounds omitted", len(bound_vars) - max_items)
    except Exception:
        logger.exception("Failed to compute Gurobi IIS for %s", context)

def resolve_fix_interconnectors(balancing_config: dict, logger=None) -> bool:
    """Resolve fixed versus re-tradable BM interconnector schedules."""

    retrading = balancing_config.get("interconnector_retrading")
    if retrading is None:
        return bool(balancing_config.get("fix_interconnectors", True))
    if not isinstance(retrading, bool):
        raise ValueError(
            "market.balancing.interconnector_retrading must be true, false or null"
        )
    if logger:
        logger.info(
            "BM interconnector re-trading is %s",
            "enabled" if retrading else "disabled",
        )
    return not retrading




def _foreign_market_generator_names(network, generator_names):
    """Return only the generator names representing external market positions."""
    # ``foreign_market_role`` is a sparse custom PyPSA component attribute.
    # NetCDF round-trips serialise its missing values as empty strings, which
    # are non-null in pandas.  Test the explicit role values instead of just
    # ``notna`` so domestic generators are not mistaken for foreign assets.
    foreign_market_roles = {"import_supply", "export_demand", "market_balance"}
    names = []
    has_role = "foreign_market_role" in network.generators.columns
    for name in generator_names:
        bus_is_external = "External" in str(network.generators.at[name, "bus"])
        role_is_foreign = has_role and str(
            network.generators.at[name, "foreign_market_role"]
        ).strip().lower() in foreign_market_roles
        if bus_is_external or role_is_foreign:
            names.append(name)
    return set(names)


def _clear_runtime_bm_attrs(network):
    """Remove transient BM helper attributes before exporting the network."""
    for attr in [
        "_bm_offer_ladders",
        "_bm_bid_ladders",
        "_bm_ladder_fallback_volume_mw",
        "_bm_ladder_missing_hour_fallback",
        "_bm_counter_trade_callback",
    ]:
        if hasattr(network, attr):
            delattr(network, attr)


def _infer_snapshot_hours(snapshots) -> float:
    """Hours per snapshot from the first interval; 1.0 for a single snapshot."""
    if len(snapshots) > 1:
        return float((snapshots[1] - snapshots[0]).total_seconds() / 3600)
    return 1.0


def _storage_energy_cost(network, hours: float = None) -> float:
    """Storage dispatch cost the BM objective carries on gross ``p_dispatch``.

    This is the tie-breaking term restored in
    ``_build_balancing_extra_functionality``. It is not a redispatch payment,
    so it is stripped back out of the reported BM cost.
    """
    if len(network.storage_units) == 0 or len(network.storage_units_t.p_dispatch) == 0:
        return 0.0
    if hours is None:
        hours = _infer_snapshot_hours(network.snapshots)
    mc = pd.to_numeric(network.storage_units["marginal_cost"], errors="coerce").fillna(0.0)
    dispatch = network.storage_units_t.p_dispatch.reindex(columns=mc.index, fill_value=0.0)
    return float((dispatch * mc).sum().sum() * hours)


#: Must match config/defaults/ market.balancing.storage_end_soc.
DEFAULT_STORAGE_END_SOC_PENALTY = 200.0


def resolve_storage_end_soc_config(balancing_config: dict) -> tuple[bool, float]:
    """``market.balancing.storage_end_soc`` as ``(enabled, penalty GBP/MWh)``."""
    cfg = dict((balancing_config or {}).get("storage_end_soc", {}) or {})
    enabled = bool(cfg.get("enabled", True))
    penalty = float(cfg.get("penalty_gbp_per_mwh", DEFAULT_STORAGE_END_SOC_PENALTY) or 0.0)
    if penalty < 0.0:
        raise ValueError(
            f"market.balancing.storage_end_soc.penalty_gbp_per_mwh must be >= 0, got {penalty}"
        )
    return enabled and penalty > 0.0, penalty


#: ``market.balancing.storage_anchor`` (10 Oct 2026). ``net_sign``: the BM
#: anchors each storage leg on Stage 1's net output split by sign (p > 0
#: discharge, p < 0 charge), right while Stage 1 burns almost nothing.
#: ``stage1_legs``: it anchors on the gross legs Stage 1 saved on its network
#: (``{wholesale}_wholesale.nc``). Needed with a calibrated churn share
#: (``market.wholesale.storage_churn_share``): the CP30 2030 churn_neso Stage 1
#: burns 9.2 TWh a year in simultaneous charge and discharge, and a BM anchored
#: on the net started each hour without that churn and sold 5.5 TWh of energy
#: Stage 1 never had as storage offers. On the legs, undoing churn pays the
#: storage offer on the charge leg and its bid on the discharge leg, as any other
#: storage redispatch does. Absent from config/defaults/ (adding it would
#: invalidate every solve), so the code default is the one in force.
STORAGE_ANCHORS = ("net_sign", "stage1_legs")
DEFAULT_STORAGE_ANCHOR = "net_sign"


def resolve_storage_anchor(balancing_config: dict) -> str:
    """``market.balancing.storage_anchor``, validated."""
    anchor = str((balancing_config or {}).get("storage_anchor", DEFAULT_STORAGE_ANCHOR)).lower()
    if anchor not in STORAGE_ANCHORS:
        raise ValueError(f"market.balancing.storage_anchor must be one of {STORAGE_ANCHORS}, got {anchor!r}")
    return anchor


def read_stage1_storage_legs(path, wholesale_su: pd.DataFrame, logger, tolerance_mw: float = 1.0):
    """Stage 1's gross storage legs ``(p_store, p_dispatch)`` from its saved network.

    Aligned to ``wholesale_su`` (index and columns, missing units zero: PyPSA
    saves only non-zero series). Refuses a network without the legs (a Stage 1
    solved before 10 Oct 2026) and legs whose net differs from the saved net
    schedule by more than ``tolerance_mw`` in any unit-hour.
    """
    import xarray as xr
    from scripts.interconnectors.market_coupling import dataset_snapshots

    with xr.open_dataset(path) as ds:
        if "storage_units_t_p_store" not in ds or "storage_units_t_p_dispatch" not in ds:
            raise ValueError(
                f"{path} has no storage legs (storage_units_t_p_store / p_dispatch): market.balancing."
                "storage_anchor stage1_legs needs a Stage 1 solved since 10 Oct 2026; re-solve it"
            )
        legs = []
        for var in ("storage_units_t_p_store", "storage_units_t_p_dispatch"):
            frame = ds[var].to_pandas()
            frame.index = dataset_snapshots(ds, frame.index)
            frame.columns = frame.columns.astype(str)
            legs.append(frame.reindex(index=wholesale_su.index, columns=wholesale_su.columns).fillna(0.0))
    store, dispatch = legs
    gap = (dispatch - store - wholesale_su.fillna(0.0)).abs()
    if float(gap.to_numpy().max(initial=0.0)) > tolerance_mw:
        unit = gap.max().idxmax()
        raise ValueError(
            f"{path}: storage legs do not net to the saved Stage 1 schedule (worst {unit}, "
            f"{float(gap[unit].max()):.3f} MW)"
        )
    overlap = np.minimum(store.to_numpy(), dispatch.to_numpy()).sum() * float(
        np.median(_infer_snapshot_hours(wholesale_su.index)) if len(wholesale_su.index) > 1 else 1.0)
    logger.info(
        f"Storage anchored on Stage 1's gross legs ({store.shape[1]} units): simultaneous charge and "
        f"discharge {overlap / 1e3:,.1f} GWh over the period (market.balancing.storage_anchor stage1_legs)"
    )
    return store, dispatch


def stage1_storage_leg_arrays(wholesale_su: pd.DataFrame, names, snapshots, legs=None):
    """Stage 1 ``(dispatch, store)`` arrays (T, S): the saved legs, or net p split by sign."""
    if legs is not None:
        store, dispatch = legs
        return (
            dispatch.reindex(index=snapshots, columns=names).fillna(0.0).clip(lower=0.0).to_numpy(),
            store.reindex(index=snapshots, columns=names).fillna(0.0).clip(lower=0.0).to_numpy(),
        )
    net = wholesale_su.reindex(index=snapshots, columns=names).fillna(0.0).to_numpy()
    return np.maximum(net, 0.0), np.maximum(-net, 0.0)


def slice_storage_legs(legs, snapshots, names):
    """The legs restricted to a window's snapshots and units (None stays None)."""
    if legs is None:
        return None
    return tuple(leg.reindex(index=snapshots, columns=names).fillna(0.0) for leg in legs)


def stage1_state_of_charge(network, wholesale_su: pd.DataFrame, legs=None) -> pd.DataFrame:
    """State of charge Stage 1 scheduled, per snapshot and unit (MWh).

    Rebuilt from the saved net storage schedule and the network's starting state
    (``diagnose_storage_degeneracy.reconstruct_state_of_charge``); the BM starts
    from the same state (``apply_initial_storage_soc``). Net ``p`` cannot show
    simultaneous charge and discharge, so without ``legs`` the rebuilt path sits
    above the true one wherever Stage 1 churns (~0.3% of charge with the cost
    floor, far more with a churn share); with Stage 1's gross ``legs``
    (``read_stage1_storage_legs``) it is exact. Clipped to ``[0, p_nom * max_hours]``.
    """
    su = network.storage_units
    names = [s for s in wholesale_su.columns if s in su.index]
    if not names or wholesale_su.empty:
        return pd.DataFrame(index=wholesale_su.index)

    def _col(name, default):
        if name in su.columns:
            return pd.to_numeric(su.loc[names, name], errors="coerce").fillna(default)
        return pd.Series(default, index=names, dtype=float)

    attributes = pd.DataFrame({
        "efficiency_store": _col("efficiency_store", 1.0),
        "efficiency_dispatch": _col("efficiency_dispatch", 1.0),
        "standing_loss": _col("standing_loss", 0.0),
        "state_of_charge_initial": _col("state_of_charge_initial", 0.0),
    }, index=names)
    dispatch = wholesale_su[names].sort_index().fillna(0.0)
    hours = pd.Series(_infer_snapshot_hours(dispatch.index), index=dispatch.index)
    # Drains (consumer flexibility, F5: inflow < 0) move the level too; without them a drained
    # unit's rebuilt level only rises and its end-of-window floor is unreachable once it bids in
    # the BM (hold_consumer_flexibility: false). Positive (hydro) inflow is left out as before, so
    # no central result moves (it can spill, which the saved schedule does not show).
    inflow_t = getattr(network.storage_units_t, "inflow", None)
    inflow = (inflow_t.reindex(index=dispatch.index, columns=names).fillna(0.0).clip(upper=0.0).to_numpy()
              if inflow_t is not None and len(inflow_t.columns) else np.zeros(dispatch.shape))
    if legs is None and not np.any(inflow):
        soc = reconstruct_state_of_charge(dispatch, hours, attributes)
    else:
        leg_d, leg_s = stage1_storage_leg_arrays(wholesale_su, names, dispatch.index, legs)
        h = hours.to_numpy()[:, np.newaxis]
        delta = (leg_s * h * attributes["efficiency_store"].to_numpy()
                 - leg_d * h / attributes["efficiency_dispatch"].to_numpy() + inflow * h)
        decay = 1.0 - attributes["standing_loss"].to_numpy()
        level = attributes["state_of_charge_initial"].to_numpy().astype(float).copy()
        path = np.empty_like(delta)
        for step in range(delta.shape[0]):
            level = level * decay + delta[step]
            path[step] = level
        soc = pd.DataFrame(path, index=dispatch.index, columns=names)
    capacity = (_col("p_nom", 0.0) * _col("max_hours", 0.0)).astype(float)
    return soc.clip(lower=0.0).clip(upper=capacity, axis=1)


def held_consumer_flex_draw_by_bus(network, wholesale_su: pd.DataFrame, legs=None):
    """Snapshots x bus MW that consumer flexibility (F5) draws at its Stage 1 schedule (p_store - p_dispatch).

    The BM holds these units at Stage 1, so this is demand the BM cannot move; the load-shedding cap
    (``solve_network.apply_load_shedding_limits(extra_local_load=)``) adds it. None without such units.
    """
    from scripts.demand.demand_carriers import CONSUMER_FLEX_CARRIERS

    su = network.storage_units
    if len(su) == 0 or "carrier" not in su.columns:
        return None
    units = [s for s in wholesale_su.columns if s in su.index and str(su.at[s, "carrier"]) in CONSUMER_FLEX_CARRIERS]
    if not units:
        return None
    held_d, held_s = stage1_storage_leg_arrays(wholesale_su, units, network.snapshots, legs)
    draw = pd.DataFrame(held_s - held_d, index=network.snapshots, columns=units)
    return draw.T.groupby(su.loc[units, "bus"].astype(str).to_numpy()).sum().T


def _storage_end_soc_shortfall(network, target: pd.Series, units=None) -> pd.Series:
    """MWh each unit ended below its end-of-window target (the slack the solve used)."""
    soc = network.storage_units_t.state_of_charge
    if target is None or len(soc) == 0:
        return pd.Series(dtype=float)
    names = [s for s in target.index if s in soc.columns and (units is None or s in set(units))]
    final = soc[names].iloc[-1]
    return (target.reindex(names) - final).clip(lower=0.0).fillna(0.0)


#: ``market.balancing.storage_reverse_leg_guard_gbp_per_mwh``: objective-only
#: cost per MWh burnt by moving the leg that opposes the Stage 1 direction.
#: Tuned on the CP30 1 Jan 2030 probe (see config/defaults/). Separate from
#: the offer-side ``storage_churn_guard_gbp_per_mwh``. Must match config/defaults/.
DEFAULT_STORAGE_REVERSE_LEG_GUARD = 250.0


def storage_reverse_leg_prices(network, balancing_config: dict) -> pd.Series:
    """Objective-only reverse-leg price per storage unit: guard x (1/eta_d - eta_s).

    One MWh moved on the leg opposing the Stage 1 direction burns
    ``1/eta_dispatch - eta_store`` MWh relative to moving the Stage 1 leg, so
    this price makes every burnt MWh cost the reverse-leg guard
    (``market.balancing.storage_reverse_leg_guard_gbp_per_mwh``). Empty when the
    guard is 0 or there is no storage.
    """
    guard = float((balancing_config or {}).get(
        "storage_reverse_leg_guard_gbp_per_mwh", DEFAULT_STORAGE_REVERSE_LEG_GUARD) or 0.0)
    su = network.storage_units
    if guard <= 0.0 or len(su) == 0:
        return pd.Series(dtype=float)
    eta_s = pd.to_numeric(su["efficiency_store"], errors="coerce").fillna(1.0)
    eta_d = pd.to_numeric(su["efficiency_dispatch"], errors="coerce").fillna(1.0).clip(lower=1e-6)
    return (guard * (1.0 / eta_d - eta_s)).clip(lower=0.0)


def _storage_reverse_leg_cost(network, wholesale_su: pd.DataFrame, prices: pd.Series, legs=None) -> float:
    """What the reverse-leg guard added to the objective (stripped from the BM cost).

    The guard prices the discharge increase in Stage 1 charging hours and the
    charge increase in discharging hours; on net-sign anchors the Stage 1 leg
    opposing the direction is zero, so the increase is the whole leg.
    """
    if prices is None or len(prices) == 0 or len(network.storage_units_t.p_dispatch) == 0:
        return 0.0
    snaps = network.snapshots
    names = [s for s in prices.index if s in network.storage_units_t.p_dispatch.columns
             and s in wholesale_su.columns]
    if not names:
        return 0.0
    ws_d, ws_s = stage1_storage_leg_arrays(wholesale_su, names, snaps, legs)
    hours = _infer_snapshot_hours(snaps)
    dispatch = network.storage_units_t.p_dispatch.reindex(index=snaps, columns=names).fillna(0.0).to_numpy()
    store = network.storage_units_t.p_store.reindex(index=snaps, columns=names).fillna(0.0).to_numpy()
    reverse = (np.where(ws_s > 1e-6, np.maximum(dispatch - ws_d, 0.0), 0.0)
               + np.where(ws_d > 1e-6, np.maximum(store - ws_s, 0.0), 0.0))
    return float((reverse * prices.reindex(names).to_numpy()[np.newaxis, :]).sum() * hours)


def _log_storage_end_soc_summary(log: pd.DataFrame, penalty: float, logger) -> None:
    """One line on how much end-of-window storage slack the BM used."""
    if log is None or log.empty:
        return
    short = log[log["shortfall_mwh"] > 1e-3]
    message = (
        f"End-of-window storage floor: slack used in {len(short)} of {len(log)} windows, "
        f"{log['shortfall_mwh'].sum():,.1f} MWh summed over windows "
        f"(max {log['shortfall_mwh'].max():,.1f} MWh, last window "
        f"{log['shortfall_mwh'].iloc[-1]:,.1f} MWh of {log['target_mwh'].iloc[-1]:,.0f} MWh target), "
        f"penalty {log['penalty_gbp'].sum():,.0f} GBP at {penalty:.0f} GBP/MWh stripped from the BM cost"
    )
    logger.info(message)


def _apply_offer_regulariser(gen_offer_tv, regulariser, logger, side: str = "offers"):
    """The price frame the BM objective sees: ``gen_offer_tv`` plus an objective-only regulariser.

    Used for the decision-20 turn-up floor (on offers) and the BM1 state-first
    step (on bids). Returns the frame itself when there is no regulariser. The
    reported costs must keep using the unmodified frame: a regulariser is not a payment.
    """
    if regulariser is None or len(regulariser) == 0:
        return gen_offer_tv
    if gen_offer_tv is None:
        raise ValueError(f"An objective-only regulariser on {side} needs the per-snapshot price frame "
                         "contract-value bids build")
    columns = pd.Index(regulariser.index).intersection(gen_offer_tv.columns)
    objective = gen_offer_tv.copy()
    objective.loc[:, columns] = gen_offer_tv[columns].to_numpy() + regulariser[columns].to_numpy()[None, :]
    logger.info(f"Objective-only regulariser added to the {side} of {len(columns)} generators "
                f"(max {float(regulariser.max()):.2f} GBP/MWh); reported costs are unaffected")
    return objective


def _offer_regulariser_cost(physical_gen, wholesale_gen, regulariser, network, direction: str = "up") -> float:
    """What a regulariser added to the BM objective: its price x MWh moved in ``direction``.

    ``up`` (the turn-up floor, on offers) or ``down`` (the state-first step, on
    bids), measured against the wholesale position as the LP's increase and
    decrease variables are; stripped from the reported BM cost like
    ``_storage_energy_cost``.
    """
    if regulariser is None or len(regulariser) == 0 or physical_gen is None or physical_gen.empty:
        return 0.0
    columns = pd.Index(regulariser.index).intersection(physical_gen.columns)
    if not len(columns):
        return 0.0
    schedule = wholesale_gen.reindex(index=physical_gen.index, columns=columns).fillna(0.0)
    moved = physical_gen[columns] - schedule
    increase = (moved if direction == "up" else -moved).clip(lower=0.0)
    weights = getattr(network, "snapshot_weightings", None)
    hours = (pd.to_numeric(weights["generators"], errors="coerce").reindex(physical_gen.index).fillna(1.0)
             if weights is not None and "generators" in weights else pd.Series(1.0, index=physical_gen.index))
    return float(increase.mul(regulariser[columns], axis=1).sum(axis=1).mul(hours).sum())


def _prepare_hydrogen_link_pricing(network, balancing_config, wholesale_price, wholesale_links,
                                   scenario_config, logger):
    """Anchored hydrogen links, their BM prices and the held ones: ``(names, increase, decrease, held)``.

    ``names`` is empty (and the prices None) when ``hydrogen_links.enabled`` is
    false or the network has no hydrogen link. The hydrogen value is the Stage 1
    GB_H2 dual (``hydrogen_price`` column); a Stage 1 run from before that column
    existed falls back to ``estimate_hydrogen_value_from_schedule`` with a WARNING.
    ``held`` are the links of ``hydrogen_links.hold_carriers`` (the hydrogen
    turbines, decision D5): fixed at their Stage 1 position, so not priced.
    """
    cfg = resolve_hydrogen_link_config(balancing_config)
    if not cfg["enabled"]:
        logger.info("Hydrogen links: not anchored in the balancing stage (hydrogen_links.enabled false)")
        return [], None, None, []
    names = list(hydrogen_links(network, cfg["carriers"], cfg["bus_carriers"]))
    names = [lk for lk in names if lk in wholesale_links.columns]
    hold = set(resolve_held_hydrogen_carriers(balancing_config))
    release = set(resolve_released_hydrogen_links(balancing_config))
    held = [lk for lk in names if str(network.links.at[lk, "carrier"]) in hold and lk not in release]
    released = sorted(lk for lk in names if lk in release and str(network.links.at[lk, "carrier"]) in hold)
    if released:
        logger.info(f"Hydrogen links released from the hold (priced in the BM like the other hydrogen links): "
                    f"{', '.join(released)} (hydrogen_links.release_links)")
    names = [lk for lk in names if lk not in set(held)]
    if held:
        logger.info(
            f"Hydrogen links held at their Stage 1 position in the BM: {len(held)} "
            f"({', '.join(sorted(hold))}; decision D5)"
        )
    if not names:
        return [], None, None, held
    snapshots = pd.Index(network.snapshots)
    if HYDROGEN_PRICE_COLUMN in wholesale_price.columns:
        hydrogen_value = pd.to_numeric(
            wholesale_price[HYDROGEN_PRICE_COLUMN], errors="coerce"
        ).reindex(snapshots)
        missing = int(hydrogen_value.isna().sum())
        if missing:
            logger.warning(f"Hydrogen price missing in {missing} snapshots; filled forward/backward")
            hydrogen_value = hydrogen_value.ffill().bfill().fillna(0.0)
    else:
        window = int((scenario_config.get("market", {}).get("wholesale", {}) or {}).get("window_hours", 24))
        hydrogen_value = estimate_hydrogen_value_from_schedule(
            wholesale_price["wholesale_price"].reindex(snapshots).fillna(0.0),
            wholesale_links, network, names, window, logger, cfg["bus_carriers"],
        )
    increase, decrease = hydrogen_link_prices(
        network, names, hydrogen_value, cfg["offer_markup"], cfg["bid_discount"], cfg["bus_carriers"],
    )
    # Decision 35 (task BM2050): hydrogen turbines on their DPA, from the 2050
    # support label table. No 2030 scenario sets hydrogen_links.dpa_pricing.
    from scripts.market.support_bids import dpa_link_block

    dpa_cfg = dpa_link_block(balancing_config)
    if dpa_cfg.get("enabled", False):
        from scripts.market.support_bids import (
            DEFAULT_DPA_LINK_CARRIERS, dpa_link_prices, load_support_labels, support_labels_block)

        bm_bids = ((scenario_config or {}).get("support") or {}).get("bm_bids") or {}
        labels = load_support_labels(dpa_cfg.get("file") or support_labels_block(bm_bids).get("file"))
        increase, decrease, _ = dpa_link_prices(
            network, names, increase, decrease, hydrogen_value, labels, logger,
            carriers=tuple(dpa_cfg.get("carriers", DEFAULT_DPA_LINK_CARRIERS)),
        )
    carriers = network.links.loc[names, "carrier"].value_counts().to_dict()
    logger.info(
        f"Hydrogen links anchored in the BM: {carriers}; hydrogen value mean "
        f"{hydrogen_value.mean():.2f} GBP/MWh_H2 (range {hydrogen_value.min():.2f}-"
        f"{hydrogen_value.max():.2f}), offer markup {cfg['offer_markup']:.2f}, "
        f"bid discount {cfg['bid_discount']:.2f}"
    )
    return names, increase, decrease, held


def _log_storage_energy_balance(network, logger) -> dict:
    """Log the storage energy the solved dispatch destroyed (decision D6; WARNING above 1 MWh).

    Read from the gross legs (``market_accounting.storage_simultaneous_burn``):
    ``min(p_dispatch, p_store) x (1/eta_dispatch - eta_store)`` per unit-hour,
    counting overlaps above a tolerance relative to ``p_nom``. The old check
    counted any overlap above 1e-3 MW (45% of unit-hours on the Dec 2030 week,
    barrier noise) and rebuilt the state of charge from net ``p``, which cannot
    see a simultaneous pair at all. The same figure is reported as curtailed
    energy in the market accounting; nothing here changes the dispatch.
    """
    burn = storage_simultaneous_burn(network)
    result = {"simultaneous_unit_hours": burn["unit_hours"], "burn_MWh": burn["burn_MWh"]}
    if not burn["total_unit_hours"]:
        return result
    share = burn["unit_hours"] / burn["total_unit_hours"]
    by_carrier = ", ".join(f"{c} {v:,.1f}" for c, v in sorted(burn["burn_by_carrier_MWh"].items()))
    message = (
        f"Storage simultaneous charge+discharge (both legs above {burn['tolerance_pu']:g} x p_nom): "
        f"{burn['unit_hours']:,} of {burn['total_unit_hours']:,} unit-hours ({100 * share:.3f}%), "
        f"{burn['burn_MWh']:,.1f} MWh destroyed{' (' + by_carrier + ')' if by_carrier else ''}, "
        f"reported as curtailed energy; sub-tolerance noise {burn['noise_MWh']:,.2f} MWh"
    )
    (logger.warning if burn["burn_MWh"] > 1.0 else logger.info)(message)
    return result


#: Hydrogen-link carriers held at their Stage 1 position in the BM (decision D5).
#: Must match config/defaults/ market.balancing.hydrogen_links.hold_carriers.
DEFAULT_HELD_HYDROGEN_CARRIERS = ("H2_turbine",)


def resolve_held_hydrogen_carriers(balancing_config: dict) -> tuple:
    """``market.balancing.hydrogen_links.hold_carriers``: hydrogen links fixed at Stage 1."""
    cfg = dict((balancing_config or {}).get("hydrogen_links", {}) or {})
    held = cfg.get("hold_carriers", DEFAULT_HELD_HYDROGEN_CARRIERS)
    return tuple(str(c) for c in (held or ()))


def resolve_released_hydrogen_links(balancing_config: dict) -> tuple:
    """``market.balancing.hydrogen_links.release_links``: held-carrier links the BM may move (default none).

    19 Oct 2026 (user): on a network that cannot import a held electrolyser's schedule (no_upgrades reverts
    the 203 MVA Skye line to today's 99 MVA one, and the 94 MW Ardmore electrolyser then forced load
    shedding on Skye), that link is priced like any other hydrogen link instead of being held at Stage 1.
    """
    cfg = dict((balancing_config or {}).get("hydrogen_links", {}) or {})
    return tuple(str(lk) for lk in (cfg.get("release_links") or ()))


#: ``market.balancing.hydrogen_links.store_band``: ``stage1`` (decision D5's band,
#: the default) or ``none`` (task BM2050, decision 35: the store's own level only).
STORE_BAND_MODES = ("stage1", "none")


def resolve_store_band_mode(balancing_config: dict) -> str:
    """``market.balancing.hydrogen_links.store_band`` (default ``stage1``)."""
    cfg = dict((balancing_config or {}).get("hydrogen_links", {}) or {})
    mode = str(cfg.get("store_band", "stage1")).strip().lower()
    if mode not in STORE_BAND_MODES:
        raise ValueError(f"hydrogen_links.store_band must be one of {STORE_BAND_MODES}, got {mode!r}")
    return mode


def resolve_hydrogen_link_config_tolerance(balancing_config: dict) -> float:
    """``market.balancing.hydrogen_links.store_month_end_tolerance_pu`` (share of e_nom)."""
    cfg = dict((balancing_config or {}).get("hydrogen_links", {}) or {})
    tol = float(cfg.get("store_month_end_tolerance_pu", DEFAULT_STORE_MONTH_END_TOLERANCE_PU))
    if not 0.0 <= tol <= 0.1:
        raise ValueError(f"hydrogen_links.store_month_end_tolerance_pu must be in [0, 0.1], got {tol}")
    return tol


def stage1_store_levels(network, wholesale_links: pd.DataFrame, start: pd.Series, logger=None) -> pd.DataFrame:
    """Stage 1 level of each carried store per snapshot (MWh), rebuilt from the saved link schedule.

    Stage 1 saves link flows, not store levels. With one store on its bus the
    bus balance fixes the store's flow: what the links deliver to the bus
    (``efficiency x p0`` for links ending there) minus what they take
    (``p0`` for links starting there) minus the loads there, stepped with
    PyPSA's store recursion ``e_t = (1 - standing_loss)^w e_(t-1) + w x net``
    from the run's start (``prepare_carried_stores``). Stores sharing a bus
    are skipped (their split is not in the schedule). Clipped to [0, e_nom].
    """
    snapshots = pd.Index(network.snapshots)
    out = {}
    if start is None or not len(start) or not len(snapshots):
        return pd.DataFrame(index=snapshots)
    stores, links = network.stores, network.links
    weights = network.snapshot_weightings.stores.reindex(snapshots).fillna(1.0).to_numpy(dtype=float)
    flows = wholesale_links.reindex(index=snapshots).fillna(0.0)
    try:
        demand = network.get_switchable_as_dense("Load", "p_set", snapshots)
    except (AttributeError, KeyError):
        demand = pd.DataFrame(index=snapshots)
    for name in start.index:
        bus = stores.at[name, "bus"]
        if int((stores["bus"] == bus).sum()) != 1:
            if logger:
                logger.warning(f"Hydrogen store {name}: shares bus {bus}; no Stage 1 level path, no BM floor")
            continue
        into = links.index[links["bus1"] == bus]
        out_of = links.index[links["bus0"] == bus]
        eff = pd.to_numeric(links.loc[into, "efficiency"], errors="coerce").fillna(1.0)
        inflow = flows.reindex(columns=into, fill_value=0.0).mul(eff, axis=1).sum(axis=1)
        outflow = flows.reindex(columns=out_of, fill_value=0.0).sum(axis=1)
        loads = network.loads.index[network.loads["bus"] == bus]
        offtake = demand.reindex(columns=loads, fill_value=0.0).sum(axis=1) if len(loads) else 0.0
        net = (inflow - outflow - offtake).to_numpy(dtype=float)
        loss = float(pd.to_numeric(stores.at[name, "standing_loss"], errors="coerce") or 0.0)
        e_nom = float(pd.to_numeric(stores.at[name, "e_nom"], errors="coerce") or 0.0)
        level, path = float(start[name]), np.empty(len(snapshots))
        for t in range(len(snapshots)):
            level = (1.0 - loss) ** weights[t] * level + weights[t] * net[t]
            path[t] = level
        out[name] = np.clip(path, 0.0, e_nom if e_nom > 0 else np.inf)
    return pd.DataFrame(out, index=snapshots)


#: ``market.balancing.hydrogen_links.store_month_end_tolerance_pu``: how far (share
#: of e_nom) the BM store may end the solve period from Stage 1's month-end level
#: before the band's penalty applies. Must match config/defaults/.
DEFAULT_STORE_MONTH_END_TOLERANCE_PU = 1.0e-3


def stage1_store_band(network, wholesale_links: pd.DataFrame, start: pd.Series, tolerance_pu: float,
                      logger=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The BM hydrogen store band per snapshot: ``(lower, upper)`` in MWh (user decision "shift timing only").

    Electrolysers may run harder in the BM to absorb constrained wind, but only
    by bringing hydrogen forward: the month's output must match Stage 1's. So at
    the end of every window the store must lie in

    * ``lower = S1(t)``: the Stage 1 level (``stage1_store_levels``), the floor
      kept from before: the BM never uses hydrogen Stage 1 needed later; and
    * ``upper = S1(t) + R(t)``, where ``R(t)`` is what the rest of the period can
      still take out of the store by turning the electrolysers down from their
      Stage 1 schedule (``efficiency x (p0 - p_min)`` summed over the snapshots
      after ``t``; turbines are held at Stage 1 and cannot help). A store above
      ``upper`` could no longer be brought back by month end, so the ceiling is
      exactly what keeps the month-end target reachable, and it is no tighter:
      mid-month the BM stays free to bank hydrogen (the store's own e_nom
      bounds it), and each MWh banked is paid back later at the electrolysers'
      turn-down price, so the solve sees the shift's cost.

    At the last snapshot ``R = 0``, so the band closes on Stage 1's month-end
    level, widened by ``tolerance_pu x e_nom`` each side. Both sides are soft
    (``build_store_band_callback``). Standing loss (1e-4/h on GB_H2) is ignored
    in ``R``; it only shrinks a banked surplus, so ``upper`` is conservative.
    """
    path = stage1_store_levels(network, wholesale_links, start, logger)
    snapshots = pd.Index(network.snapshots)
    if path.empty or not len(path.columns):
        return pd.DataFrame(index=snapshots), pd.DataFrame(index=snapshots)
    links = network.links
    weights = network.snapshot_weightings.stores.reindex(snapshots).fillna(1.0)
    flows = wholesale_links.reindex(index=snapshots).fillna(0.0)
    lower, upper = path.copy(), path.copy()
    for name in path.columns:
        bus = network.stores.at[name, "bus"]
        makers = [lk for lk in links.index[(links["bus1"] == bus) & (links["bus0"] != bus)]]
        eff = pd.to_numeric(links.loc[makers, "efficiency"], errors="coerce").fillna(1.0)
        p_nom = pd.to_numeric(links.loc[makers, "p_nom"], errors="coerce").fillna(0.0)
        p_min = network.get_switchable_as_dense("Link", "p_min_pu", snapshots)[makers].mul(p_nom, axis=1) \
            if makers else pd.DataFrame(index=snapshots)
        reducible = (flows.reindex(columns=makers, fill_value=0.0) - p_min).clip(lower=0.0).mul(eff, axis=1)
        per_snapshot = reducible.sum(axis=1).mul(weights) if makers else pd.Series(0.0, index=snapshots)
        # R(t): what the snapshots after t can still take out.
        remaining = per_snapshot[::-1].cumsum()[::-1].shift(-1).fillna(0.0)
        e_nom = float(pd.to_numeric(network.stores.at[name, "e_nom"], errors="coerce") or 0.0)
        tol = float(tolerance_pu) * e_nom
        upper[name] = path[name] + remaining.to_numpy()
        lower.loc[snapshots[-1], name] = path.loc[snapshots[-1], name] - tol
        upper.loc[snapshots[-1], name] = path.loc[snapshots[-1], name] + tol
    return lower, upper


def build_store_band_callback(lower: pd.Series, upper: pd.Series, penalty: float, logger=None):
    """Soft band on each carried store's level at the window's last snapshot (decision D5, BM side).

    ``e[s, T] + slack_lo[s] >= lower[s]`` and ``e[s, T] - slack_hi[s] <= upper[s]``,
    both slacks priced at ``penalty`` per MWh in the objective only (the caller
    strips them, ``store_band_violation``). Soft so no window can be made
    infeasible; the band itself (``stage1_store_band``) is always reachable from
    a window that started inside it. None without a band or a penalty.
    """
    if lower is None or not len(lower) or penalty <= 0.0:
        return None

    def store_band(n, sns):
        model = n.model
        if "Store-e" not in model.variables:
            return
        import xarray as xr

        e = model.variables["Store-e"]
        dim = [d for d in e.dims if d != "snapshot"][0]
        present = [s for s in lower.index if s in set(e.coords[dim].values) and np.isfinite(lower[s])]
        if not present:
            return
        idx = pd.Index(present, name=dim)
        zeros = xr.DataArray(np.zeros(len(idx)), dims=[dim], coords={dim: idx})
        final = e.sel({dim: present}).sel(snapshot=pd.Index(sns)[-1])
        lo = model.add_variables(lower=zeros, name="Store-band_lower_slack")
        model.add_constraints(
            final + lo >= xr.DataArray(lower.reindex(present).to_numpy(dtype=float), dims=[dim], coords={dim: idx}),
            name="Store-band_lower",
        )
        penalty_terms = (lo * float(penalty)).sum()
        up_values = (upper.reindex(present) if upper is not None else pd.Series(np.inf, index=present))
        bounded = [s for s in present if np.isfinite(up_values[s])]
        if bounded:
            bidx = pd.Index(bounded, name=dim)
            hi = model.add_variables(lower=xr.DataArray(np.zeros(len(bidx)), dims=[dim], coords={dim: bidx}),
                                     name="Store-band_upper_slack")
            model.add_constraints(
                e.sel({dim: bounded}).sel(snapshot=pd.Index(sns)[-1]) - hi
                <= xr.DataArray(up_values.reindex(bounded).to_numpy(dtype=float), dims=[dim], coords={dim: bidx}),
                name="Store-band_upper",
            )
            penalty_terms = penalty_terms + (hi * float(penalty)).sum()
        model.objective = model.objective + penalty_terms
        if logger:
            logger.info(
                f"Hydrogen store band at {pd.Index(sns)[-1]}: {len(present)} store(s) between "
                f"{lower.reindex(present).sum() / 1e3:,.1f} and {up_values.sum() / 1e3:,.1f} GWh "
                f"(Stage 1 level to Stage 1 + what the rest of the period can take back), "
                f"violation at {penalty:.0f} GBP/MWh (objective only)"
            )

    return store_band


#: Objective-only price (GBP/MWh) on moving a held link off its Stage 1 position
#: (5 Oct 2026). The hold was an equality, so a carried store that Stage 1 runs
#: to its floor could make a window infeasible: rebuilt from Stage 1's saved
#: flows, the GB_H2 store sits 44-207 MWh below Stage 1's own level by mid-month
#: (solver tolerance accumulated over the rolling windows), and on a day Stage 1
#: empties it exactly the held electrolysers left no way to stay above zero
#: (IIS on the Aug 2030 P3 window 8: GB_H2 nodal balance, store energy balance,
#: store lower bound, Link-bm_hold). Above the 6,000 VoLL, so the slack is never
#: a cheaper balancing action than anything real; it is used only where the hold
#: is infeasible, and is stripped from the reported cost (held_link_slack).
HELD_LINK_SLACK_PENALTY_GBP_PER_MWH = 10_000.0

#: Held consumer flexibility (F5) moves off its Stage 1 schedule at this price instead (4 Nov 2026). Stage 1 has
#: no network, so it pre-heats all of London's flexible heating at the midday price; with B14 at its limit and
#: every London unit at full output, the hard hold left the BM only load shedding (about 3.4 GWh in each of
#: March and April 2030, at West Weybridge, Chessington and Amersham, on every network case). Above every
#: generator offer and below VoLL, so the BM reschedules flexible demand only where the alternative is
#: shedding; objective-only, stripped from the BM cost and reported apart (consumer_flex_rescheduled).
CONSUMER_FLEX_RESCHEDULE_GBP_PER_MWH = 1_000.0

_LINK_HOLD_VARIABLES = ("Link-bm_hold_up", "Link-bm_hold_down")
_STORAGE_HOLD_VARIABLES = ("StorageUnit-bm_hold_dispatch_up", "StorageUnit-bm_hold_dispatch_down",
                           "StorageUnit-bm_hold_store_up", "StorageUnit-bm_hold_store_down")


def _hold_slack_frames(network, names):
    """Weighted MWh frames (snapshot x unit) of the named hold-slack variables in the network's last solve."""
    model = getattr(network, "model", None)
    if model is None:
        return
    for name in names:
        if name not in model.variables:
            continue
        solution = model.variables[name].solution
        if solution is None:
            continue
        values = solution.to_pandas()
        if isinstance(values, pd.Series):
            values = values.to_frame()
        weights = network.snapshot_weightings.generators.reindex(values.index).fillna(1.0)
        yield values.clip(lower=0.0).mul(weights, axis=0)


def _consumer_flex_mask(network, units) -> np.ndarray:
    from scripts.demand.demand_carriers import CONSUMER_FLEX_CARRIERS

    carrier = network.storage_units["carrier"] if "carrier" in network.storage_units.columns else pd.Series(dtype=str)
    return carrier.reindex(pd.Index(units)).isin(CONSUMER_FLEX_CARRIERS).to_numpy()


def held_link_slack(network, penalty: float = HELD_LINK_SLACK_PENALTY_GBP_PER_MWH) -> tuple[float, float]:
    """``(penalty GBP, MWh)`` of held-link and held-storage movement off Stage 1 in the network's last solve.

    Consumer flexibility is not counted here: it moves at its own price (``consumer_flex_rescheduled``).
    """
    total = sum(float(f.to_numpy().sum()) for f in _hold_slack_frames(network, _LINK_HOLD_VARIABLES))
    for f in _hold_slack_frames(network, _STORAGE_HOLD_VARIABLES):
        total += float(f.loc[:, ~_consumer_flex_mask(network, f.columns)].to_numpy().sum())
    return total * float(penalty), total


def consumer_flex_rescheduled(network, price: float = CONSUMER_FLEX_RESCHEDULE_GBP_PER_MWH) -> tuple[float, float]:
    """``(penalty GBP, MWh)`` of held consumer flexibility moved off its Stage 1 schedule in the last solve."""
    total = 0.0
    for f in _hold_slack_frames(network, _STORAGE_HOLD_VARIABLES):
        total += float(f.loc[:, _consumer_flex_mask(network, f.columns)].to_numpy().sum())
    return total * float(price), total


def record_consumer_flex_rescheduled(network, gbp: float, mwh: float, logger) -> None:
    """Store the BM's consumer-flexibility rescheduling in ``network.meta['consumer_flex_rescheduled']`` and log it."""
    meta = dict(getattr(network, "meta", None) or {})
    meta["consumer_flex_rescheduled"] = {"mwh": float(mwh), "penalty_gbp": float(gbp),
                                         "price_gbp_per_mwh": CONSUMER_FLEX_RESCHEDULE_GBP_PER_MWH}
    network.meta = meta
    logger.info(f"Held consumer flexibility rescheduled in the BM: {mwh:,.1f} MWh at "
                f"{CONSUMER_FLEX_RESCHEDULE_GBP_PER_MWH:,.0f} GBP/MWh (penalty {gbp:,.0f} stripped from the BM cost)")


def store_band_violation(network, lower: pd.Series, upper: pd.Series) -> tuple[float, float]:
    """``(MWh below lower, MWh above upper)`` of the carried stores at the network's last snapshot."""
    e = getattr(getattr(network, "stores_t", None), "e", None)
    if lower is None or not len(lower) or e is None or e.empty:
        return 0.0, 0.0
    names = [s for s in lower.index if s in e.columns]
    final = e[names].iloc[-1]
    below = float((lower.reindex(names) - final).clip(lower=0.0).fillna(0.0).sum())
    above = 0.0
    if upper is not None:
        above = float((final - upper.reindex(names)).clip(lower=0.0).fillna(0.0).sum())
    return below, above


def resolve_offer_regulariser(network, bm_bids: dict, logger):
    """The objective-only offer regulariser, with the general turn-up floor applied in every case (item C8).

    ``bm_bids`` is the scenario's ``support.bm_bids`` block.

    ``cfd_bm_bids`` folds the general floor (``contract_value_bids
    .turn_up_floor_gbp_per_mwh``) into ``network._bm_offer_regulariser`` only
    when contract-value bids are on; windbid_low and central dispatch price
    renewables another way and so had free renewable-for-renewable swaps. The
    floor is a regulariser on reported volumes, not a price, so it applies
    whichever bid stack is in use.
    """
    existing = getattr(network, "_bm_offer_regulariser", None)
    cvb = ((bm_bids or {}).get("contract_value_bids") or {})
    if cvb.get("enabled", False):
        return existing
    general = resolve_general_turn_up_floor(cvb, network.generators["carrier"], logger)
    if general is None:
        return existing
    if existing is None or not len(existing):
        return general
    return pd.concat([general, existing], axis=1).max(axis=1)


def ensure_price_frames(network, gen_offer: pd.Series, gen_bid: pd.Series, offer_tv, bid_tv):
    """Per-snapshot offer and bid frames, broadcast from the static prices where the pricing built none.

    A regulariser needs the offer frame, and the balancing objective uses the
    frames only when BOTH exist, so the two are built together from the same
    static prices the LP would otherwise tile: the coefficients are unchanged.
    """
    snapshots = pd.Index(network.snapshots)
    names = network.generators.index

    def _tile(prices):
        return pd.DataFrame(np.tile(prices.reindex(names).to_numpy(dtype=float), (len(snapshots), 1)),
                            index=snapshots, columns=names)

    return (offer_tv if offer_tv is not None else _tile(gen_offer),
            bid_tv if bid_tv is not None else _tile(gen_bid))


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
    anchored_links=None,
    link_increase_prices: pd.DataFrame = None,
    link_decrease_prices: pd.DataFrame = None,
    storage_end_soc_target: pd.Series = None,
    storage_end_soc_penalty: float = 0.0,
    storage_reverse_leg_price: pd.Series = None,
    fixed_links=None,
    wholesale_su_legs=None,
):
    """
    Build the ``extra_functionality`` callback for PyPSA's ``network.optimize()``.

    ``wholesale_su_legs``: Stage 1's gross ``(p_store, p_dispatch)``
    (``read_stage1_storage_legs``) for the storage anchors; None splits the net
    ``wholesale_su`` by sign (``market.balancing.storage_anchor``).

    The callback injects:
      1. Increase/decrease variables for generators and storage units
      2. Linking constraints:  p[g,t] == p_wholesale[g,t] + inc[g,t] - dec[g,t]
      3. A new objective:  min  offerinc + biddec
      4. (Optional) Fix interconnector flows to wholesale values

    Parameters
    ----------
    wholesale_gen : pd.DataFrame
        Wholesale generator dispatch (snapshots  generators).
    wholesale_su : pd.DataFrame
        Wholesale storage dispatch (snapshots  storage units).
    wholesale_links : pd.DataFrame
        Wholesale link dispatch (snapshots  links).
    gen_offer_prices, gen_bid_prices : pd.Series
        Per-generator offer/bid prices.
    su_offer_prices, su_bid_prices : pd.Series
        Per-storage-unit offer/bid prices.
    fix_interconnectors : bool
        If True, fix Link flows to wholesale values.
    logger : logging.Logger
        Logger instance.
    anchored_links : list, optional
        Links (the hydrogen links) anchored to their wholesale p0 with priced
        increase/decrease variables, like generators.
    link_increase_prices, link_decrease_prices : pd.DataFrame, optional
        Per-snapshot prices (GBP per MWh of p0, ESO-cost convention) for those
        links; from ``market_utils.hydrogen_link_prices``.
    storage_end_soc_target : pd.Series, optional
        Per-unit state of charge (MWh) Stage 1 scheduled at the last snapshot
        of this solve (``stage1_state_of_charge``). Participating units must
        end at or above it, softly: a shortfall costs
        ``storage_end_soc_penalty`` per MWh in the objective only
        (``_storage_end_soc_shortfall`` recovers it for stripping).
    storage_reverse_leg_price : pd.Series, optional
        Objective-only price (GBP/MWh) per unit on the leg that opposes the
        Stage 1 direction: extra discharge in an hour Stage 1 charges, extra
        charge in an hour it discharges (``storage_reverse_leg_prices``).
        Stripped from the reported cost by ``_storage_reverse_leg_cost``.
    fixed_links : list, optional
        Links held at their wholesale p0 (``Link-p == p_wholesale``): the
        hydrogen turbines under decision D5
        (``market.balancing.hydrogen_links.hold_carriers``).

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

        def _build_ladder_arrays(
            ladders,
            sns,
            names,
            fallback_prices,
            side_label,
            movement_upper,
            fallback_prices_tv=None,
        ):
            """Return price/volume arrays without restricting physical redispatch.

            ELEXON BOD volumes are measured relative to the historical FPN.  The
            BM in this workflow is instead anchored to a modelled wholesale
            position, so the observed ladder depth is useful for pricing but is
            not a valid hard limit on movement from that different baseline.
            Append a fallback-priced residual block up to the physical
            increase/decrease bound.  This also matches ``_ladder_cost`` in
            ``market_utils``, which already applies the fallback price after
            the observed ladder has been exhausted.
            """
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

            # Reserve one final block for physical headroom beyond the observed
            # ELEXON ladder.  It will have zero volume where the empirical
            # ladder already covers the feasible movement.
            prices = np.zeros(
                (len(sns), len(ladder_names), max_block + 1), dtype=float
            )
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

            name_indices = [names.index(name) for name in ladder_names]
            physical_upper = np.asarray(movement_upper, dtype=float)[:, name_indices]

            if ladder_missing_hour_fallback:
                fallback = fallback_prices.reindex(ladder_names).fillna(0.0).values
                missing = ~has_ladder
                for snap_i, name_i in zip(*np.where(missing)):
                    prices[snap_i, name_i, 0] = fallback[name_i]
                    # The physical inc/dec variable is already the binding
                    # limit.  Avoid a nominal 1e6-MW block, which needlessly
                    # expands the bound range and can make Gurobi barrier
                    # misclassify a feasible ladder model as infeasible.
                    volumes[snap_i, name_i, 0] = min(
                        ladder_fallback_volume_mw,
                        physical_upper[snap_i, name_i],
                    )

            observed_volume = volumes[:, :, :max_block].sum(axis=2)
            residual_volume = np.maximum(0.0, physical_upper - observed_volume)

            if fallback_prices_tv is not None:
                residual_prices = np.asarray(fallback_prices_tv, dtype=float)[
                    :, name_indices
                ]
            else:
                residual_prices = np.tile(
                    fallback_prices.reindex(ladder_names).fillna(0.0).values,
                    (len(sns), 1),
                )
            prices[:, :, max_block] = residual_prices
            volumes[:, :, max_block] = residual_volume

            residual_hours = int((residual_volume > 1.0e-9).sum())
            residual_capacity = float(residual_volume.sum())

            active = volumes.sum(axis=(0, 2)) > 0
            if not active.any():
                return [], None, None
            ladder_names = [name for name, keep in zip(ladder_names, active) if keep]
            prices = prices[:, active, :]
            volumes = volumes[:, active, :]

            logger.info(
                f"Using {side_label} price ladders for {len(ladder_names)} generators "
                f"({max_block} observed blocks plus fallback residual); "
                f"fallback headroom added in {residual_hours:,} generator-hours "
                f"({residual_capacity:,.1f} MW summed across hours)"
            )
            return ladder_names, prices, volumes

        #  1. Generator increase/decrease variables 
        # Generator participation is configurable. Non-participants can either
        # be priced out upstream or fixed at wholesale position here.
        all_gen_names = [g for g in wholesale_gen.columns if g in network.generators.index]
        participant_gen_set = set(
            all_gen_names if participating_generators is None else participating_generators
        )
        fixed_gen_set = set([] if fixed_generators is None else fixed_generators)
        if fix_interconnectors:
            foreign_market_names = _foreign_market_generator_names(
                network, all_gen_names
            )
            participant_gen_set -= foreign_market_names
            fixed_gen_set |= foreign_market_names
        gen_names = [g for g in all_gen_names if g in participant_gen_set]
        fixed_gen_names = [
            g for g in all_gen_names if g in fixed_gen_set and g not in participant_gen_set
        ]
        gen_dim = _component_dim("Generator-p") if all_gen_names else "name"

        if gen_names:
            lower_gen = _zero_lower(snapshots, gen_names, gen_dim)

            # Upper bounds prevent the LP going unbounded when offer prices are
            # negative (e.g. ROC generators with negative marginal costs).
            # increase  p_max[g,t] - p_wholesale[g,t]  (headroom to ceiling)
            # decrease  p_wholesale[g,t] - p_min[g,t]  (headroom to floor)
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

            # Apply time-varying p_min_pu where present (the commitment proxy
            # floors units the Stage 1 schedule committed; heat-led CHP floors
            # follow heat demand, apply_marginal_costs.apply_seasonal_heat_profile).
            # As in PyPSA, a time-varying column replaces the static floor; the
            # proxy already clips its floor at the static one. Taking the max of
            # the two held CHP at its annual 30% floor in the BM in summer, when
            # Stage 1 had it at 20%. Missing hours fall back to the static floor.
            tv_min = getattr(network.generators_t, "p_min_pu", None)
            if tv_min is not None and not tv_min.empty:
                tv_cols = [g for g in gen_names if g in tv_min.columns]
                if tv_cols:
                    tv_idx = [gen_names.index(g) for g in tv_cols]
                    static_min = pd.Series(p_min_pu_static, index=gen_names)[tv_cols]
                    tv_vals = (
                        tv_min[tv_cols]
                        .reindex(snapshots)
                        .fillna(static_min)
                        .values
                    ) * p_nom_gen[tv_idx][np.newaxis, :]
                    p_min_arr[:, tv_idx] = tv_vals

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

        #  2. Storage unit increase/decrease variables 
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

            # Anchor each gross leg separately. The wholesale stage saves only
            # net p, but it dissipates ~0.3% of charge, so its gross split is
            # recovered by sign: p > 0 is discharge, p < 0 is charging.
            # Anchoring the net alone left the split behind any net position
            # free, and a unit that charges and discharges in the same hour
            # destroys energy for no more than its dispatch-cost floor - far
            # cheaper than paying a wind farm to curtail - so the BM used
            # storage as a dump load (13.7% of charging burnt on the 2030
            # months even with the floor). With both legs anchored a
            # simultaneous pair pays offer + bid and is never worth it.
            # With storage_anchor stage1_legs the legs are Stage 1's own, so a
            # unit that churned in Stage 1 starts the BM churning too.
            p_nom_su = network.storage_units.loc[su_names, "p_nom"].values  # (S,)
            ws_dispatch_arr, ws_store_arr = stage1_storage_leg_arrays(
                wholesale_su, su_names, snapshots, wholesale_su_legs)       # (T, S) each
            headroom_dispatch = np.maximum(0.0, p_nom_su[np.newaxis, :] - ws_dispatch_arr)
            headroom_store = np.maximum(0.0, p_nom_su[np.newaxis, :] - ws_store_arr)

            for var_name, upper in (
                ("StorageUnit-dispatch-increase", headroom_dispatch),
                ("StorageUnit-dispatch-decrease", ws_dispatch_arr),
                ("StorageUnit-store-increase", headroom_store),
                ("StorageUnit-store-decrease", ws_store_arr),
            ):
                model.add_variables(
                    lower=lower_su,
                    upper=_make_upper(snapshots, su_names, upper, su_dim),
                    name=var_name,
                )

            su_p = model.variables["StorageUnit-p_dispatch"].sel({su_dim: su_names})
            su_store = model.variables["StorageUnit-p_store"].sel({su_dim: su_names})
            model.add_constraints(
                su_p
                - model.variables["StorageUnit-dispatch-increase"]
                + model.variables["StorageUnit-dispatch-decrease"]
                == ws_dispatch_arr,
                name="StorageUnit-bm_anchor_dispatch",
            )
            model.add_constraints(
                su_store
                - model.variables["StorageUnit-store-increase"]
                + model.variables["StorageUnit-store-decrease"]
                == ws_store_arr,
                name="StorageUnit-bm_anchor_store",
            )

            logger.info(
                f"Added BM variables + gross-leg anchor constraints for "
                f"{len(su_names)} storage units"
            )
        su_hold_penalty_obj = None
        if fixed_su_names:
            su_p_fixed = model.variables["StorageUnit-p_dispatch"].sel({su_dim: fixed_su_names})
            su_store_fixed = model.variables["StorageUnit-p_store"].sel({su_dim: fixed_su_names})
            fixed_d, fixed_s = stage1_storage_leg_arrays(
                wholesale_su, fixed_su_names, snapshots, wholesale_su_legs)
            # Pin both legs, not the net, so a non-participant cannot churn either. Soft, as the held links'
            # (3 Nov 2026): a leg moves off Stage 1 only at HELD_LINK_SLACK_PENALTY_GBP_PER_MWH (objective only,
            # stripped by held_link_slack), so a near-empty unit whose carried state of charge drifted below
            # Stage 1's cannot make the window infeasible (IIS on 30 Jun 2030: CEAN1Q EV_smart_charging energy
            # balance, both leg pins, state of charge >= 0). Consumer flexibility moves at
            # CONSUMER_FLEX_RESCHEDULE_GBP_PER_MWH instead (4 Nov 2026), so the BM reschedules it before it sheds load.
            import xarray as xr

            hold_price = xr.DataArray(
                np.where(_consumer_flex_mask(network, fixed_su_names), CONSUMER_FLEX_RESCHEDULE_GBP_PER_MWH,
                         HELD_LINK_SLACK_PENALTY_GBP_PER_MWH),
                coords={su_dim: fixed_su_names}, dims=[su_dim])
            legs = {}
            for leg in ("dispatch", "store"):
                for side in ("up", "down"):
                    legs[leg, side] = model.add_variables(
                        lower=_zero_lower(snapshots, fixed_su_names, su_dim),
                        name=f"StorageUnit-bm_hold_{leg}_{side}")
            model.add_constraints(
                su_p_fixed - legs["dispatch", "up"] + legs["dispatch", "down"] == fixed_d,
                name="StorageUnit-bm_fix_nonparticipant_dispatch",
            )
            model.add_constraints(
                su_store_fixed - legs["store", "up"] + legs["store", "down"] == fixed_s,
                name="StorageUnit-bm_fix_nonparticipant_store",
            )
            su_hold_penalty_obj = sum((v * hold_price).sum() for v in legs.values())
            logger.info(
                f"Fixed {len(fixed_su_names)} non-participating storage units "
                f"at wholesale positions (both legs; soft at {HELD_LINK_SLACK_PENALTY_GBP_PER_MWH:,.0f} GBP/MWh, "
                f"consumer flexibility at {CONSUMER_FLEX_RESCHEDULE_GBP_PER_MWH:,.0f})"
            )

        #  3. Fix interconnector flows (if configured) 
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

        #  3b. Anchor hydrogen links to their wholesale p0
        # Internal links are otherwise free in this objective; electrolysers on
        # the copperplate GB_H2 bus moved 8.2 TWh a year for nothing. Anchored
        # like generators, with the Stage 1 hydrogen value as the price
        # (market_utils.hydrogen_link_prices).
        link_names = []
        if anchored_links is not None and len(anchored_links) and "Link-p" in model.variables:
            link_names = [
                lk for lk in anchored_links
                if lk in network.links.index and lk in wholesale_links.columns
                and link_increase_prices is not None and lk in link_increase_prices.columns
            ]
        if link_names:
            link_dim = _component_dim("Link-p")
            links = network.links.loc[link_names]
            p_nom_link = pd.to_numeric(links["p_nom"], errors="coerce").fillna(0.0).to_numpy()

            def _link_pu(attr, default):
                static = pd.to_numeric(links[attr], errors="coerce").fillna(default)
                frame = pd.DataFrame(
                    np.tile(static.to_numpy(), (n_snapshots, 1)),
                    index=pd.Index(snapshots), columns=link_names,
                )
                varying = getattr(network.links_t, attr, None)
                if varying is not None and not varying.empty:
                    cols = [lk for lk in link_names if lk in varying.columns]
                    if cols:
                        frame[cols] = varying[cols].reindex(snapshots).fillna(static[cols]).to_numpy()
                return frame.to_numpy() * p_nom_link[np.newaxis, :]

            link_max = _link_pu("p_max_pu", 1.0)
            link_min = _link_pu("p_min_pu", 0.0)
            ws_link_arr = wholesale_links[link_names].reindex(snapshots).fillna(0.0).values
            lower_link = _zero_lower(snapshots, link_names, link_dim)
            model.add_variables(
                lower=lower_link,
                upper=_make_upper(snapshots, link_names, np.maximum(0.0, link_max - ws_link_arr), link_dim),
                name="Link-increase",
            )
            model.add_variables(
                lower=lower_link,
                upper=_make_upper(snapshots, link_names, np.maximum(0.0, ws_link_arr - link_min), link_dim),
                name="Link-decrease",
            )
            model.add_constraints(
                model.variables["Link-p"].sel({link_dim: link_names})
                - model.variables["Link-increase"]
                + model.variables["Link-decrease"]
                == ws_link_arr,
                name="Link-bm_anchor",
            )
            logger.info(
                f"Added BM variables + anchor constraints for {len(link_names)} hydrogen links"
            )

        # 3c. Links held at their wholesale position (decision D5: hydrogen
        # turbines). A held turbine's hydrogen burn is Stage 1's, so the store
        # the balancing stage carries moves only with the electrolysers.
        held_links = []
        if fixed_links is not None and len(fixed_links) and "Link-p" in model.variables:
            link_dim = _component_dim("Link-p")
            present = set(model.variables["Link-p"].coords[link_dim].values)
            held_links = [
                lk for lk in fixed_links
                if lk in present and lk in wholesale_links.columns and lk not in set(link_names)
            ]
        hold_penalty_obj = None
        if held_links:
            link_dim = _component_dim("Link-p")
            # Soft hold (5 Oct 2026): a held link moves off Stage 1 only at
            # HELD_LINK_SLACK_PENALTY_GBP_PER_MWH (objective only), so a carried
            # store Stage 1 runs to its floor cannot make the window infeasible.
            hold_up = model.add_variables(
                lower=_zero_lower(snapshots, held_links, link_dim), name="Link-bm_hold_up")
            hold_down = model.add_variables(
                lower=_zero_lower(snapshots, held_links, link_dim), name="Link-bm_hold_down")
            model.add_constraints(
                model.variables["Link-p"].sel({link_dim: held_links}) - hold_up + hold_down
                == wholesale_links[held_links].reindex(snapshots).fillna(0.0).values,
                name="Link-bm_hold",
            )
            hold_penalty_obj = ((hold_up + hold_down) * float(HELD_LINK_SLACK_PENALTY_GBP_PER_MWH)).sum()
            logger.info(f"Held {len(held_links)} links at their wholesale position (hydrogen turbines and "
                        f"electrolysers; soft at {HELD_LINK_SLACK_PENALTY_GBP_PER_MWH:,.0f} GBP/MWh)")

        #  4. Replace objective with BM redispatch cost
        # Build objective expression: min  hours(offerinc + biddec)
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

            # Build offer/bid coefficient arrays (snapshots  generators).
            # Use per-snapshot time-varying prices where the pricing stage built
            # them - ELEXON traces for historical runs, contract-value CfD bids
            # for the CP30 ones (scripts/market/cfd_bm_bids.py) - otherwise
            # broadcast the static prices across all snapshots. The frames cover
            # every generator, so the fillna below is a guard, not a path: a
            # missing column would silently restore the zero bid B1 removed.
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
                    upper_inc_arr,
                    offer_coeffs,
                )
            )
            bid_ladder_names, bid_ladder_prices, bid_ladder_volumes = (
                _build_ladder_arrays(
                    gen_bid_ladders,
                    snapshots,
                    gen_names,
                    gen_bid_prices,
                    "bid",
                    upper_dec_arr,
                    bid_coeffs,
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
            inc_d = model.variables["StorageUnit-dispatch-increase"]
            dec_d = model.variables["StorageUnit-dispatch-decrease"]
            inc_s = model.variables["StorageUnit-store-increase"]
            dec_s = model.variables["StorageUnit-store-decrease"]

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

            # Net output up (more discharge, or less charging) is an accepted
            # offer; net output down (less discharge, or more charging) is an
            # accepted bid. This reproduces the old net pricing for every real
            # move and makes a simultaneous pair (inc_d + inc_s) pay both.
            su_obj = (
                (inc_d * su_offer_coeffs * snapshot_hours_da).sum()
                + (dec_s * su_offer_coeffs * snapshot_hours_da).sum()
                + (dec_d * su_bid_coeffs * snapshot_hours_da).sum()
                + (inc_s * su_bid_coeffs * snapshot_hours_da).sum()
            )
            obj_expr = obj_expr + su_obj if obj_expr is not None else su_obj

            # Reverse-leg guard. In an hour Stage 1 charges, moving net output up
            # by discharging (inc_d, offer) instead of charging less (dec_s,
            # offer) costs the same but burns (1/eta_d - eta_s) per MWh, and in
            # a discharging hour charging more (inc_s) instead of discharging
            # less (dec_d) does the same at the bid. Those ties are a free sink
            # whenever the BM would rather burn energy than curtail (CP30 1 Jan
            # 2030 probe: 840 MWh of overlap, 140 MWh burnt), and a real unit
            # cannot charge and discharge at once. The objective-only price
            # (storage_reverse_leg_prices) makes each burnt MWh cost at least
            # the churn guard; the reported cost excludes it.
            if storage_reverse_leg_price is not None:
                rev = (
                    pd.to_numeric(storage_reverse_leg_price, errors="coerce")
                    .reindex(su_names).fillna(0.0).to_numpy()
                )
                if (rev > 0).any():
                    charging_hours = (ws_store_arr > 1e-6).astype(float) * rev[np.newaxis, :]
                    discharging_hours = (ws_dispatch_arr > 1e-6).astype(float) * rev[np.newaxis, :]
                    obj_expr = obj_expr + (
                        (inc_d * charging_hours * snapshot_hours_da).sum()
                        + (inc_s * discharging_hours * snapshot_hours_da).sum()
                    )
                    logger.info(
                        f"Storage reverse-leg guard: {int((rev > 0).sum())} units, "
                        f"{rev[rev > 0].min():.1f}-{rev.max():.1f} GBP/MWh on discharge in "
                        "Stage 1 charging hours and charge in discharging hours (objective only)"
                    )

            # End-of-window state of charge (market.balancing.storage_end_soc):
            # soc[s, T] + slack[s] >= Stage 1 soc[s, T], slack priced at the
            # penalty. Objective-only; the caller strips penalty x shortfall from
            # the reported BM cost. Soft, so it can never make a window infeasible.
            end_soc_units = []
            if (
                storage_end_soc_target is not None
                and storage_end_soc_penalty > 0.0
                and "StorageUnit-state_of_charge" in model.variables
            ):
                soc_var = model.variables["StorageUnit-state_of_charge"]
                soc_dim = [d for d in soc_var.dims if d != "snapshot"][0]
                soc_names = set(soc_var.coords[soc_dim].values)
                target = pd.to_numeric(storage_end_soc_target, errors="coerce")
                end_soc_units = [
                    s for s in su_names
                    if s in soc_names and s in target.index and np.isfinite(target.get(s, np.nan))
                ]
            if end_soc_units:
                final = snapshots[-1]
                unit_idx = pd.Index(end_soc_units, name=soc_dim)
                model.add_variables(
                    lower=xr.DataArray(np.zeros(len(unit_idx)), dims=[soc_dim], coords={soc_dim: unit_idx}),
                    name="StorageUnit-end_soc_slack",
                )
                slack = model.variables["StorageUnit-end_soc_slack"]
                target_da = xr.DataArray(
                    target.reindex(end_soc_units).to_numpy(dtype=float),
                    dims=[soc_dim], coords={soc_dim: unit_idx},
                )
                model.add_constraints(
                    soc_var.sel({soc_dim: end_soc_units}).sel(snapshot=final) + slack >= target_da,
                    name="StorageUnit-end_soc_floor",
                )
                obj_expr = obj_expr + (slack * float(storage_end_soc_penalty)).sum()
                logger.info(
                    f"End-of-window storage floor: {len(end_soc_units)} units held at or above "
                    f"their Stage 1 state of charge at {final} ({float(target_da.sum()) / 1e3:,.1f} GWh), "
                    f"shortfall priced at {storage_end_soc_penalty:.0f} GBP/MWh (objective only)"
                )

        if link_names:
            inc_coeffs = link_increase_prices.reindex(index=snapshots, columns=link_names).fillna(0.0).values
            dec_coeffs = link_decrease_prices.reindex(index=snapshots, columns=link_names).fillna(0.0).values
            link_obj = (
                (model.variables["Link-increase"] * inc_coeffs * snapshot_hours_da).sum()
                + (model.variables["Link-decrease"] * dec_coeffs * snapshot_hours_da).sum()
            )
            obj_expr = obj_expr + link_obj if obj_expr is not None else link_obj

        # Price gross storage discharge. The redispatch terms above see only
        # the net deviation from the wholesale position, and the anchor
        # constrains only p_dispatch - p_store, so the split behind any net
        # position is otherwise free: a barrier solve without crossover then
        # charges and discharges the same unit at once and burns the round-trip
        # loss (3.6 TWh across the 2030 months). PyPSA's native objective prices
        # p_dispatch at marginal_cost; replacing the objective dropped that, so
        # it is restored here for every storage unit, participant or not. The
        # cost is stripped back out of the reported BM cost (_storage_energy_cost).
        if len(network.storage_units) > 0:
            su_p_var = model.variables["StorageUnit-p_dispatch"]
            p_dim = _component_dim("StorageUnit-p_dispatch")
            p_names = list(su_p_var.coords[p_dim].values)
            su_mc = (
                pd.to_numeric(network.storage_units["marginal_cost"], errors="coerce")
                .reindex(p_names)
                .fillna(0.0)
            )
            mc_da = xr.DataArray(
                su_mc.values,
                dims=[p_dim],
                coords={p_dim: pd.Index(p_names, name=p_dim)},
            )
            storage_energy_obj = (su_p_var * mc_da * snapshot_hours_da).sum()
            obj_expr = (
                obj_expr + storage_energy_obj
                if obj_expr is not None else storage_energy_obj
            )
            logger.info(
                f"Storage dispatch cost term added to BM objective for "
                f"{len(p_names)} units ({su_mc.min():.2f}-{su_mc.max():.2f} GBP/MWh "
                "on gross p_dispatch)"
            )

        if obj_expr is not None and hold_penalty_obj is not None:
            obj_expr = obj_expr + hold_penalty_obj
        if obj_expr is not None and su_hold_penalty_obj is not None:
            obj_expr = obj_expr + su_hold_penalty_obj
        if obj_expr is not None:
            model.objective = obj_expr
            logger.info("Replaced objective with BM redispatch cost minimisation")
        else:
            logger.warning(
                "No generators or storage units matched wholesale positions  "
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
    optimize_kwargs=None,
    anchored_links=None,
    link_increase_prices=None,
    link_decrease_prices=None,
    storage_end_soc_target=None,
    storage_end_soc_penalty=0.0,
    storage_reverse_leg_price=None,
    stage1_price=None,
    fixed_links=None,
    store_start=None,
    store_band=None,
    store_penalty=0.0,
    energy_floor=None,
    wholesale_su_legs=None,
):
    """
    Solve balancing mechanism in rolling windows.

    ``wholesale_su_legs``: Stage 1's gross storage legs for the anchors
    (``market.balancing.storage_anchor`` stage1_legs), sliced per window.

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
    storage_end_soc_target : pd.DataFrame, optional
        Stage 1 state of charge per snapshot and unit (``stage1_state_of_charge``).
        Each window's participating units must end at or above the row for the
        window's last snapshot, softly, at ``storage_end_soc_penalty`` GBP/MWh
        (objective only: stripped from the returned BM cost). The slack used per
        window is left on ``network._bm_storage_end_soc_log``.
    stage1_price : pd.Series, optional
        Stage 1 GB clearing price per snapshot, for the interconnector
        counter-trade spread rule (``interconnector_counter_trade.py``). The
        counter-trade premium stays in the returned BM cost: it is a payment.
    fixed_links : list, optional
        Links held at their wholesale position (decision D5: hydrogen turbines).
    store_start : pd.Series, optional
        Carried stores and their run start (``prepare_carried_stores``, already
        applied to ``network``): each window starts where the previous ended.
    store_band, store_penalty : (pd.DataFrame, pd.DataFrame), float
        ``(lower, upper)`` store level per snapshot (``stage1_store_band``): each
        window should end inside the band at its last snapshot, softly at
        ``store_penalty`` per MWh (objective only, stripped from the returned BM
        cost); the band closes on Stage 1's month-end level at the last snapshot.
    solver_name : str
    solver_options : dict
    scenario_config : dict
    logger : logging.Logger

    The returned BM cost is also net of the Stage 1 valuation of the cable
    schedule (``market_coupling.stage1_interconnector_trade_value``, item C9): the objective's
    interconnector term values the whole BM flow, most of which Stage 1 settled.

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
    # Configure a silent logger for subsequent windows to avoid log spam.
    # First window uses the main logger so messages appear once.
    bm_quiet = logging.getLogger("bm_quiet")
    bm_quiet.propagate = False
    if not bm_quiet.handlers:
        bm_quiet.addHandler(logging.NullHandler())

    window_hours = float(balancing_config.get("window_hours", 1))
    fix_ics = resolve_fix_interconnectors(balancing_config, logger)

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
        f"Rolling BM: {len(windows)} windows  {window_hours}h "
        f"({steps_per_window} timesteps each, {dt_hours}h resolution)"
    )

    gen_dispatches = []
    su_dispatches = []
    su_dispatch_gross = []
    su_store_gross = []
    su_soc = []
    price_dfs = []
    line_flow_dfs = []
    transformer_flow_dfs = []
    link_flow_dfs = []
    objective_total = 0.0
    storage_energy_cost_total = 0.0
    end_soc_penalty_total = 0.0
    reverse_leg_cost_total = 0.0
    counter_trade_excess_total = 0.0
    counter_trade_excess_mwh = 0.0
    ic_baseline_total = 0.0
    store_penalty_total = 0.0
    store_shortfall_total = 0.0
    flex_gbp_total = 0.0
    flex_mwh_total = 0.0
    end_soc_log = []
    prev_final_soc = None
    prev_store_levels = None
    store_levels = []
    all_generators = network.generators.index
    large_hydro_state = initialise_large_hydro_storage_state(
        network, scenario_config.get("renewables", {}).get("hydro", {})
    )
    # Same large-hydro month-end floor as Stage 1 (renewables.hydro.large_hydro.month_end).
    hydro_run_initial = None if large_hydro_state is None else large_hydro_state.copy()
    hydro_net_inflow = large_hydro_net_inflow(network, scenario_config.get("renewables", {}).get("hydro", {}))
    log_hydro_constraint_setup(network, scenario_config, logger, context="Balancing hydro")
    # market.balancing.min_output_energy_budget: hold (OW2): each budget unit keeps
    # its Stage 1 energy per block, timing free. None under the default drop.
    energy_budget_callback = build_min_output_energy_callback(energy_floor)
    energy_budget_total = energy_delivered_total = 0.0
    neso_boundary_callback = _build_neso_boundary_constraints_callback(
        network, scenario_config, logger
    )
    ic_capacity_callback = build_interconnector_shared_capacity_callback(
        network, logger
    )
    ic_piecewise_cost_callback = (
        None if fix_ics else build_piecewise_interconnector_cost_callback(
            network,
            logger,
            baseline_dispatch=wholesale_links,
            subtract_native_link_cost=False,
        )
    )
    counter_trade_callback = (
        None if fix_ics else build_counter_trade_callback(
            network, wholesale_links, stage1_price, balancing_config, logger,
        )
    )
    network._bm_counter_trade_callback = counter_trade_callback
    if counter_trade_callback is not None:
        # The no-wheeling MILP's gap (17 Oct 2026; unchanged options unless no_wheeling_mip_gap is set).
        solver_options = no_wheeling_solver_options(solver_name, solver_options, balancing_config, logger)
    # Tie-break on |flow| of GB-internal DC links (16 Oct 2026): objective only,
    # stripped from the BM cost below (internal_dc_tie_break.py).
    tie_break_cost = resolve_internal_dc_tie_break(scenario_config.get("market", {}))
    tie_break_callback = build_internal_dc_tie_break_callback(
        network, tie_break_cost, logger, stage="balancing, rolling"
    )
    tie_break_gbp_total = 0.0
    tie_break_mwh_total = 0.0

    fallback_windows = []

    for i, window_snaps in enumerate(windows):
        window_started = time.perf_counter()
        window_start = window_snaps[0]
        window_end = window_snaps[-1]

        if (i + 1) % 100 == 0 or i == 0 or i == len(windows) - 1:
            logger.info(
                f"BM window {i + 1}/{len(windows)}: "
                f"{window_start}  {window_end} ({len(window_snaps)} snapshots)"
            )

        # Create a copy restricted to this window. Item C12 (27 Sep 2026)
        # considered network.copy(snapshots=window): the same network (only the
        # int64/int32 dtype of build_year differs) but 2.7 s against 0.1 s for
        # copy() + set_snapshots() on the CP30 December network. It also tried
        # leaving the ~2,070 generators that are zero throughout a window (load
        # shedding at buses without load) out of the LP: optimize 108.2 s against
        # 109.6 s on 8 Dec 2030, objective equal to 1.7e-5, but the barrier then
        # landed on a different vertex of the degenerate ties (wind-for-wind and
        # parallel internal HVDC splits: 22 GWh of generator and 53 GWh of DC link
        # dispatch moved at equal cost), so neither is used.
        win_net = network.copy()
        win_net.set_snapshots(window_snaps)

        # Carry the hydrogen store level from the previous window (decision D5).
        if store_start is not None and len(store_start):
            apply_carried_store_levels(win_net, prev_store_levels)

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
        ws_su_legs_window = slice_storage_legs(wholesale_su_legs, window_snaps, ws_su_window.columns)
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

        window_end_target = None
        if (
            storage_end_soc_target is not None
            and storage_end_soc_penalty > 0.0
            and window_end in storage_end_soc_target.index
        ):
            window_end_target = storage_end_soc_target.loc[window_end]

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
            anchored_links=anchored_links,
            link_increase_prices=(
                None if link_increase_prices is None
                else link_increase_prices.reindex(index=window_snaps)
            ),
            link_decrease_prices=(
                None if link_decrease_prices is None
                else link_decrease_prices.reindex(index=window_snaps)
            ),
            storage_end_soc_target=window_end_target,
            storage_end_soc_penalty=storage_end_soc_penalty,
            storage_reverse_leg_price=storage_reverse_leg_price,
            fixed_links=fixed_links,
            wholesale_su_legs=ws_su_legs_window,
        )
        window_lower = window_upper = None
        if store_band is not None and store_penalty > 0.0 and window_end in store_band[0].index:
            window_lower, window_upper = store_band[0].loc[window_end], store_band[1].loc[window_end]
        store_callback = build_store_band_callback(
            window_lower, window_upper, store_penalty, logger if i in (0, len(windows) - 1) else None,
        )
        hydro_callback = build_hydro_constraints_callback(
            win_net,
            scenario_config,
            large_hydro_storage_state=large_hydro_state,
            large_hydro_end_storage=large_hydro_end_floor(
                hydro_net_inflow, hydro_run_initial, window_end
            ),
            logger=logger if i in (0, len(windows) - 1) else None,
        )
        # Solve down the fallback ladder (configured -> conservative barrier
        # -> dual simplex; the last rung is what distinguishes a numerical
        # misclassification from a structural infeasibility), re-solving once
        # if material counterflows are locked.
        context = f"BM window {i + 1}/{len(windows)} ({window_start})"
        try:
            outcome = solve_with_counterflow_lock(
                win_net,
                solver_name,
                solver_options,
                combine_extra_functionalities(
                    extra_func, tie_break_callback, counter_trade_callback, hydro_callback,
                    neso_boundary_callback,
                    ic_capacity_callback,
                    ic_piecewise_cost_callback,
                    store_callback,
                    energy_budget_callback,
                ),
                logger,
                context,
                lock_counterflows=lock_material_counterflows,
                **(optimize_kwargs or {}),
            )
        except SolveFailedError as exc:
            logger.error(f"{context} failed: {exc}")
            compute_iis = bool(
                (balancing_config.get("diagnostics", {}) or {}).get(
                    "compute_iis", False
                )
            )
            if compute_iis and str(solver_name).lower() == "gurobi" and exc.condition.lower() in {
                "infeasible",
                "infeasible_or_unbounded",
                "inf_or_unbd",
            }:
                _log_gurobi_iis(win_net, logger, context=context)
            raise RuntimeError(
                f"BM window {i + 1} ({window_start}) failed: "
                f"{exc.status} ({exc.condition})"
            ) from exc
        if outcome.rung > 1:
            fallback_windows.append((i + 1, outcome.rung_name))

        # The objective carries the storage dispatch cost term on gross
        # p_dispatch as a tie-breaker; it is not a redispatch payment.
        storage_energy_cost = _storage_energy_cost(win_net, dt_hours)
        # The end-of-window storage floor is a penalty, not a payment either.
        end_soc_penalty = 0.0
        if window_end_target is not None:
            shortfall = _storage_end_soc_shortfall(
                win_net, window_end_target, participating_storage_units
            )
            end_soc_penalty = float(shortfall.sum()) * float(storage_end_soc_penalty)
            end_soc_log.append({
                "window_end": window_end,
                "target_mwh": float(window_end_target.reindex(shortfall.index).sum()),
                "shortfall_mwh": float(shortfall.sum()),
                "units_short": int((shortfall > 1e-3).sum()),
                "penalty_gbp": end_soc_penalty,
            })
        reverse_leg_cost = _storage_reverse_leg_cost(win_net, ws_su_window, storage_reverse_leg_price,
                                                     ws_su_legs_window)
        end_soc_penalty += reverse_leg_cost   # both objective-only; reported together below
        # Beyond-limit interconnector movement: objective-only penalty, stripped too.
        excess_penalty, excess_mwh = counter_trade_excess_penalty(win_net, counter_trade_callback)
        if excess_mwh > 1.0:
            logger.warning(
                f"{context}: {excess_mwh:,.1f} MWh of interconnector movement beyond the "
                f"counter-trade flow limit (penalty {excess_penalty:,.0f} stripped from the BM cost)"
            )
        end_soc_penalty += excess_penalty
        counter_trade_excess_total += excess_penalty
        counter_trade_excess_mwh += excess_mwh
        # Hydrogen store band: objective-only, stripped.
        store_below, store_above = (
            store_band_violation(win_net, window_lower, window_upper) if store_callback is not None else (0.0, 0.0)
        )
        store_short = store_below + store_above
        end_soc_penalty += store_short * float(store_penalty)
        store_penalty_total += store_short * float(store_penalty)
        store_shortfall_total += store_short
        # Soft hold on held links: objective-only, stripped.
        hold_cost, hold_mwh = held_link_slack(win_net)
        if hold_mwh > 0.1:
            logger.warning(
                f"{context}: held links and storage moved {hold_mwh:,.1f} MWh off their Stage 1 position to keep the "
                f"window feasible (penalty {hold_cost:,.0f} stripped from the BM cost)"
            )
        end_soc_penalty += hold_cost
        # Consumer flexibility rescheduled: objective-only, stripped (reported separately).
        flex_gbp, flex_mwh = consumer_flex_rescheduled(win_net)
        if flex_mwh > 0.1:
            logger.info(f"{context}: held consumer flexibility rescheduled {flex_mwh:,.1f} MWh "
                        f"(penalty {flex_gbp:,.0f} stripped from the BM cost)")
        end_soc_penalty += flex_gbp
        flex_gbp_total += flex_gbp
        flex_mwh_total += flex_mwh
        # Internal HVDC tie-break: objective-only, stripped (reported separately).
        tie_break_gbp, tie_break_mwh = internal_dc_tie_break_cost(win_net, tie_break_callback)
        end_soc_penalty += tie_break_gbp
        tie_break_gbp_total += tie_break_gbp
        tie_break_mwh_total += tie_break_mwh
        # Item C9: the Stage 1 valuation of this window's cable schedule is trade
        # the day-ahead market already settled, not a balancing purchase. The
        # piecewise callback records it (market_coupling.baseline_trade_value, the
        # same interconnector_trade_cost the objective and the reports use); the
        # raw objective carries the whole-flow valuation (-123.8 m on the Dec 2030
        # week against a 79.8 m constraint cost).
        ic_baseline = (
            stage1_interconnector_trade_value(win_net)
            if ic_piecewise_cost_callback is not None else 0.0
        )
        ic_baseline_total += ic_baseline
        end_soc_penalty += ic_baseline
        objective_total += win_net.objective - storage_energy_cost - end_soc_penalty
        budget_mwh, delivered_mwh = min_output_energy_delivered(win_net, energy_budget_callback)
        energy_budget_total += budget_mwh
        energy_delivered_total += delivered_mwh
        storage_energy_cost_total += storage_energy_cost
        end_soc_penalty_total += end_soc_penalty
        reverse_leg_cost_total += reverse_leg_cost

        # Collect physical dispatch. Keep the gross storage split and the state
        # of charge as well as net p so the saved network can show whether a
        # unit charged and discharged in the same hour.
        gen_dispatches.append(
            win_net.generators_t.p.reindex(columns=all_generators, fill_value=0.0)
        )
        if len(win_net.storage_units_t.p) > 0:
            su_dispatches.append(win_net.storage_units_t.p.copy())
        if len(win_net.storage_units_t.p_dispatch) > 0:
            su_dispatch_gross.append(win_net.storage_units_t.p_dispatch.copy())
        if len(win_net.storage_units_t.p_store) > 0:
            su_store_gross.append(win_net.storage_units_t.p_store.copy())
        if len(win_net.storage_units_t.state_of_charge) > 0:
            su_soc.append(win_net.storage_units_t.state_of_charge.copy())

        # Collect nodal prices
        if len(win_net.buses_t.marginal_price) > 0:
            price_dfs.append(win_net.buses_t.marginal_price.copy())

        # Collect line flows for congestion analysis
        if hasattr(win_net, "lines_t") and len(win_net.lines_t.p0) > 0:
            line_flow_dfs.append(win_net.lines_t.p0.copy())

        # Collect transformer flows as well as line flows. Rolling balancing
        # previously discarded them, which made transformer congestion and
        # internal-topology diagnostics impossible in the saved network.
        if (
            hasattr(win_net, "transformers_t")
            and len(win_net.transformers_t.p0) > 0
        ):
            transformer_flow_dfs.append(win_net.transformers_t.p0.copy())

        # Collect link flows (HVDC, etc.)
        if hasattr(win_net, "links_t") and len(win_net.links_t.p0) > 0:
            link_flow_dfs.append(win_net.links_t.p0.copy())

        # Carried hydrogen store level for the next window, and its path.
        if store_start is not None and len(store_start):
            prev_store_levels = final_store_levels(win_net, store_start.index)
            e_window = getattr(win_net.stores_t, "e", pd.DataFrame())
            if e_window is not None and not e_window.empty:
                store_levels.append(e_window.reindex(columns=store_start.index).copy())

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

        # One timing line per window: optimize seconds, rung, and the
        # non-solve overhead (copy, anchor build, extraction).
        logger.info(
            f"BM window {i + 1}/{len(windows)} {window_start:%Y-%m-%d %H:%M}: "
            f"optimize {outcome.elapsed:.1f}s (rung {outcome.rung} "
            f"{outcome.rung_name}, {outcome.condition}), BM cost "
            f"{win_net.objective - storage_energy_cost - end_soc_penalty:,.0f}, window total "
            f"{time.perf_counter() - window_started:.1f}s"
        )

    log_fallback_summary(logger, "Rolling BM", fallback_windows, len(windows))

    # Concatenate
    physical_gen = pd.concat(gen_dispatches) if gen_dispatches else pd.DataFrame()
    physical_su = pd.concat(su_dispatches) if su_dispatches else pd.DataFrame()
    physical_su_dispatch = (
        pd.concat(su_dispatch_gross) if su_dispatch_gross else pd.DataFrame()
    )
    physical_su_store = pd.concat(su_store_gross) if su_store_gross else pd.DataFrame()
    physical_su_soc = pd.concat(su_soc) if su_soc else pd.DataFrame()
    all_nodal_prices = pd.concat(price_dfs) if price_dfs else pd.DataFrame()
    all_line_flows = pd.concat(line_flow_dfs) if line_flow_dfs else pd.DataFrame()
    all_transformer_flows = (
        pd.concat(transformer_flow_dfs)
        if transformer_flow_dfs else pd.DataFrame()
    )
    all_link_flows = pd.concat(link_flow_dfs) if link_flow_dfs else pd.DataFrame()

    # Write results back to the main network so the saved .nc file
    # contains solve outputs (dispatch, prices, line, transformer and link flows).
    if not physical_gen.empty:
        network.generators_t.p = physical_gen.reindex(
            columns=network.generators.index, fill_value=0.0
        )
    if not physical_su.empty:
        network.storage_units_t.p = physical_su.reindex(
            columns=network.storage_units.index, fill_value=0.0
        )
    if not physical_su_dispatch.empty:
        network.storage_units_t.p_dispatch = physical_su_dispatch.reindex(
            columns=network.storage_units.index, fill_value=0.0
        )
    if not physical_su_store.empty:
        network.storage_units_t.p_store = physical_su_store.reindex(
            columns=network.storage_units.index, fill_value=0.0
        )
    if not physical_su_soc.empty:
        network.storage_units_t.state_of_charge = physical_su_soc.reindex(
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
    if not all_transformer_flows.empty:
        network.transformers_t.p0 = all_transformer_flows.reindex(
            columns=network.transformers.index, fill_value=0.0
        )
    if not all_link_flows.empty:
        network.links_t.p0 = all_link_flows.reindex(
            columns=network.links.index, fill_value=0.0
        )
    if tie_break_callback is not None:
        record_internal_dc_tie_break(
            network, "balancing", tie_break_cost, tie_break_callback.links,
            tie_break_gbp_total, tie_break_mwh_total, logger,
        )
    record_consumer_flex_rescheduled(network, flex_gbp_total, flex_mwh_total, logger)

    logger.info(
        f"Rolling BM complete: {len(windows)} windows, "
        f"total BM cost {objective_total:,.2f} "
        f"(storage dispatch cost term {storage_energy_cost_total:,.2f} "
        f"and objective-only penalties {end_soc_penalty_total - ic_baseline_total:,.2f} "
        f"(of which reverse-leg {reverse_leg_cost_total:,.2f}, interconnector beyond-limit "
        f"{counter_trade_excess_total:,.2f} on {counter_trade_excess_mwh:,.1f} MWh, hydrogen store floor "
        f"{store_penalty_total:,.2f}, internal HVDC tie-break {tie_break_gbp_total:,.2f} on "
        f"{tie_break_mwh_total / 1e3:,.1f} GWh; the rest the storage end-of-window floor) and the Stage 1 "
        f"interconnector valuation {ic_baseline_total:,.2f} stripped from the objective)"
    )
    if store_levels:
        network.stores_t.e = pd.concat(store_levels).reindex(columns=network.stores.index)
        final = network.stores_t.e.iloc[-1].reindex(store_start.index)
        logger.info(
            f"Hydrogen store carried through {len(windows)} windows: "
            f"{store_start.sum() / 1e3:,.1f} GWh at the start, {final.sum() / 1e3:,.1f} GWh at the end"
            + (f", month-end target {store_band[0].iloc[-1].sum() / 1e3:,.2f}-"
               f"{store_band[1].iloc[-1].sum() / 1e3:,.2f} GWh"
               if store_band is not None and not store_band[0].empty else "")
            + f"; band violation {store_shortfall_total:,.1f} MWh "
              f"(penalty {store_penalty_total:,.0f} stripped)"
        )
    logger.info(
        f"Stage 1 interconnector valuation stripped from the BM cost: {ic_baseline_total:,.2f} "
        "(trade the day-ahead market settled; item C9)"
    )
    if energy_budget_callback is not None:
        logger.info(
            f"Minimum-output energy budget held in the BM: {energy_budget_total / 1e3:,.1f} GWh budgeted, "
            f"{energy_delivered_total / 1e3:,.1f} GWh delivered by the budget units"
        )
    network._bm_stage1_ic_valuation = ic_baseline_total
    network._bm_storage_end_soc_log = pd.DataFrame(end_soc_log)
    _log_storage_end_soc_summary(network._bm_storage_end_soc_log, storage_end_soc_penalty, logger)

    return physical_gen, physical_su, all_nodal_prices, objective_total


if __name__ == "__main__":
    # 
    # Setup
    # 
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
        #  Load finalized network (original transmission constraints) 
        input_path = snakemake.input.network
        logger.info(f"Loading finalized network from: {input_path}")
        network = load_network(input_path, custom_logger=logger)

        scenario_config = snakemake.params.scenario_config
        require_pypsa_branch_units(network, "Stage 2 balancing solve",
                                   scenario_config.get("network_model"), logger)
        # Decision 29 (2050 only): the same re-pricing as Stage 1; a no-op unless
        # market.foreign_price_override is set.
        from scripts.interconnectors.foreign_price_override import apply_foreign_price_override
        apply_foreign_price_override(network, scenario_config, snakemake.input, logger)
        # CP30's storage round trips, both stages alike (market.storage_round_trip_override).
        from scripts.market.market_utils import apply_storage_round_trip_override
        apply_storage_round_trip_override(network, scenario_config.get("market", {}), logger)
        # CP.28-30 calibration (17 Oct 2026): Stage 1's hydrogen offtake scale (market.hydrogen_offtake_scale), both stages
        # alike, and its interconnector re-pricing when the override sets apply_in_balancing (a no-op
        # otherwise, the 16 Oct behaviour), so the BM sees the hydrogen demand and foreign prices Stage 1 solved on.
        from scripts.market.market_utils import apply_hydrogen_offtake_scale
        apply_hydrogen_offtake_scale(network, scenario_config.get("market", {}), logger)
        from scripts.market.market_utils import apply_hydrogen_supply_price
        apply_hydrogen_supply_price(network, scenario_config.get("market", {}), logger)
        from scripts.market.interconnector_hurdle import apply_interconnector_hurdle_override
        apply_interconnector_hurdle_override(
            network, (scenario_config.get("market", {}) or {}).get("wholesale", {}), logger, stage="balancing")

        # Network delivery case (8 Oct 2026): delayed projects and their boundary years,
        # balancing stage only. Before the outage schedule, which builds the boundary limits,
        # and (15 Oct 2026) before the rating overrides, so circuits the case restores take
        # the BM voltage floor like every other branch.
        from scripts.market.network_delivery import apply_network_delivery
        scenario_config = copy.deepcopy(scenario_config)
        apply_network_delivery(network, scenario_config,
                               (scenario_config.get("market") or {}).get("balancing"), logger,
                               projects_file=getattr(snakemake.input, "appendix_b_projects", None))

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
        if hasattr(snakemake.input, 'future_bid_stack_parameters'):
            market_config.setdefault("balancing", {}).setdefault("future_bid_stack", {})[
                "parameters_file"
            ] = snakemake.input.future_bid_stack_parameters
        if hasattr(snakemake.input, 'support_bid_stack_parameters'):
            scenario_config.setdefault("support", {}).setdefault("bm_bids", {})[
                "parameters_file"
            ] = snakemake.input.support_bid_stack_parameters
        if hasattr(snakemake.input, 'support_labels'):
            # Task BM2050: the long-range label table the rule declared.
            support_block = (scenario_config.setdefault("support", {})
                             .setdefault("bm_bids", {})
                             .setdefault("contract_value_bids", {}))
            support_block["support_labels"] = dict(support_block.get("support_labels") or {},
                                                   file=snakemake.input.support_labels)
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

        #  Load wholesale dispatch from Stage 1 
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

        # The market stage's churn sink, held at its Stage 1 output (market.wholesale.churn_sink), or free to
        # move under market.balancing.churn_sink: free (the consistency check).
        wholesale_gen = fix_churn_sink(network, scenario_config.get("market", {}).get("wholesale", {}),
                                       wholesale_gen, logger,
                                       balancing_mode=scenario_config.get("market", {}).get("balancing", {})
                                       .get("churn_sink", "held"))

        # Stage 1's gross storage legs (market.balancing.storage_anchor).
        wholesale_su_legs = None
        if resolve_storage_anchor(scenario_config.get("market", {}).get("balancing", {})) == "stage1_legs":
            wholesale_su_legs = read_stage1_storage_legs(snakemake.input.wholesale_network, wholesale_su, logger)

        # Load wholesale price for comparison
        wholesale_price = pd.read_csv(
            snakemake.input.wholesale_price_csv, index_col=0, parse_dates=True
        )

        #  Pre-solve steps 
        validate_network_costs(network, logger)

        # Apply standard transmission relaxation if configured (floors/scales)
        # These are the real physics-based relaxations, NOT copperplate
        apply_transmission_relaxation(network, scenario_config, logger)

        # Explicit adequacy experiments are applied only after all standard
        # ratings/overrides. They are diagnostic, never a default reinforcement.
        adequacy_relaxation_audit = apply_adequacy_relaxation(
            network, scenario_config.get("network", {}).get("topology_adequacy", {}), logger
        )
        Path(snakemake.output.adequacy_relaxation_csv).parent.mkdir(
            parents=True, exist_ok=True
        )
        adequacy_relaxation_audit.to_csv(
            snakemake.output.adequacy_relaxation_csv, index=False
        )

        improve_numerical_conditioning(network, logger)
        apply_solve_period(network, scenario_config, logger)
        # ETYS seasonal circuit ratings by month (network-constrained stage)
        apply_seasonal_line_ratings(network, scenario_config, logger)
        # Documented outages of DC links in a historical year (network.transmission.declared_link_outages)
        from scripts.solve.link_outages import apply_declared_link_outages
        apply_declared_link_outages(network, scenario_config, logger)
        # Consumer flexibility (F5) is demand held at its Stage 1 draw: the load-shedding cap at its
        # bus includes that draw, or a node short of import could shed only its fixed loads.
        apply_load_shedding_limits(network, logger,
                                   extra_local_load=held_consumer_flex_draw_by_bus(network, wholesale_su,
                                                                                   wholesale_su_legs))
        enforce_feasible_generator_operating_bounds(network, logger)
        # CP.28-30 calibration (17 Oct 2026): the BM at Stage 1's block resolution
        # (market.balancing.temporal_resolution_hours, default 1 = hourly inputs). Every time-varying
        # input is held at its block mean, so a 3-hour Stage 1's held positions meet 3-hour wind and
        # demand, not hourly ones (scripts/market/temporal_resolution.py).
        from scripts.market.temporal_resolution import (
            check_balancing_compatible, flatten_to_blocks, resolve_balancing_temporal_resolution_hours,
        )
        bm_temporal_hours = resolve_balancing_temporal_resolution_hours(balancing_config)
        if bm_temporal_hours > 1:
            check_balancing_compatible(bm_temporal_hours, balancing_config)
            flatten_to_blocks(network, bm_temporal_hours, logger)
        # Carriers whose Stage 1 floor is an energy budget
        # (thermal.availability_factors.min_output_energy_window_hours) lose the
        # hourly floor here too: Stage 1 may have moved their energy out of an hour.
        # Under market.balancing.min_output_energy_budget: hold (OW2) the BM keeps
        # their energy budget as well (bm_energy_floor); drop (default) does not.
        bm_energy_floor = prepare_bm_min_output_energy_budget(
            network, scenario_config, balancing_config, logger
        )
        # Start storage where Stage 1 did (market.storage_soc), so the anchored
        # wholesale storage schedule is reachable from the same state of charge.
        apply_initial_storage_soc(network, market_config, logger)
        # Decision D5: the hydrogen store starts where Stage 1's did and its level
        # is carried between windows (market.hydrogen_store), not cyclic per window.
        store_start = prepare_carried_stores(network, market_config, logger)

        #  Solve mode
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
        widen_ramp_limits_to_availability(network, logger)

        #  Apply MC floors for zero-cost dispatchable generators
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
                        f"({cap_mw:.0f} MW) set to {floor_mc:.1f}/MWh"
                    )
                    total_fixed += n_fixed
            if total_fixed > 0:
                logger.info(
                    f"MC floors applied to {total_fixed} zero-cost generators"
                )

        #  Calculate bid/offer prices 
        logger.info("=" * 80)
        logger.info("CALCULATING BID/OFFER PRICES")
        logger.info("=" * 80)
        # Contract-value CfD bids need the Stage 1 clearing price as their
        # reference: a CfD unit turned down forgoes strike minus reference, so
        # its bid is per-snapshot rather than the scalar that made balancing
        # curtailment indifferent to strike. See scripts/market/cfd_bm_bids.py.
        network._bm_reference_price = wholesale_price["wholesale_price"]
        gen_offer, gen_bid, su_offer, su_bid = calculate_bid_offer_prices(
            network, market_config, logger, scenario_id=scenario_id,
            time_varying=True, scenario_config=scenario_config,
        )
        # Must-take units (market.must_take_generation) leave the BM, held at Stage 1.
        from scripts.market.must_take import apply_must_take_balancing, hold_at_stage1
        must_take = apply_must_take_balancing(network, market_config, logger)
        wholesale_gen = hold_at_stage1(network, wholesale_gen, must_take, logger)
        contract_value_audit = getattr(network, "_bm_contract_value_audit", None)
        audit_path = getattr(snakemake.output, "contract_value_bids_csv", None)
        if audit_path is not None:
            Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
            (contract_value_audit if contract_value_audit is not None
             else pd.DataFrame(columns=list(CONTRACT_VALUE_AUDIT_COLUMNS))
             ).to_csv(audit_path, index=False)
            logger.info(f"Contract-value bid audit written: {audit_path}")

        # Retrieve time-varying DataFrames (set by calculate_bid_offer_prices
        # when time_varying=True and ELEXON data is available)
        gen_offer_tv = getattr(network, "_bm_offer_tv", None)
        gen_bid_tv = getattr(network, "_bm_bid_tv", None)
        su_offer_tv = getattr(network, "_bm_su_offer_tv", None)
        su_bid_tv = getattr(network, "_bm_su_bid_tv", None)
        gen_offer_ladders = getattr(network, "_bm_offer_ladders", None)
        gen_bid_ladders = getattr(network, "_bm_bid_ladders", None)

        # Linear proxy for unit commitment, keyed on the Stage 1 schedule: a
        # start premium on units it left off, a floor on units it committed.
        # See scripts/market/commitment_proxy.py. Off unless configured.
        gen_offer_tv, gen_bid_tv, commitment_summary = apply_commitment_proxy(
            network, wholesale_gen, gen_offer, gen_bid, gen_offer_tv, gen_bid_tv,
            balancing_config.get("commitment_proxy", {}), logger,
        )

        # Decision 20: an objective-only turn-up floor on state-owned renewables
        # (cfd_bm_bids.resolve_turn_up_floor). The LP sees gen_offer_tv_objective;
        # reported offer costs keep gen_offer_tv, and the floor's cost is stripped
        # from bm_cost below.
        # Item C8: the general turn-up floor applies whether or not contract-value
        # bids built the frames (windbid_low, central dispatch had free swaps).
        offer_regulariser = resolve_offer_regulariser(
            network, (scenario_config.get("support") or {}).get("bm_bids") or {}, logger)
        bid_regulariser = getattr(network, "_bm_bid_regulariser", None)
        if ((offer_regulariser is not None and len(offer_regulariser))
                or (bid_regulariser is not None and len(bid_regulariser))):
            gen_offer_tv, gen_bid_tv = ensure_price_frames(network, gen_offer, gen_bid, gen_offer_tv, gen_bid_tv)
        gen_offer_tv_objective = _apply_offer_regulariser(gen_offer_tv, offer_regulariser, logger)
        # Item BM1: the objective-only state-first step on private merchant bids
        # (cfd_bm_bids.resolve_state_first_step); stripped from bm_cost the same way.
        gen_bid_tv_objective = _apply_offer_regulariser(gen_bid_tv, bid_regulariser, logger, side="bids")

        # Hydrogen links anchored at the Stage 1 hydrogen value (26 Sep 2026);
        # their movement is a real accepted offer or bid and stays in bm_cost.
        h2_link_names, h2_link_increase, h2_link_decrease, h2_held_links = _prepare_hydrogen_link_pricing(
            network, balancing_config, wholesale_price, wholesale_links, scenario_config, logger,
        )

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
                f"{gen_offer_tv.shape[1]} generators  {gen_offer_tv.shape[0]} snapshots"
            )
        if su_offer_tv is not None:
            logger.info(
                f"Time-varying storage bid/offer prices: "
                f"{su_offer_tv.shape[1]} units  {su_offer_tv.shape[0]} snapshots"
            )

        #  Configure solver 
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
        optimize_kwargs = solver_io_kwargs(scenario_config, solver_name, logger)
        if optimize_kwargs:
            logger.info(f"Solver model interface: {optimize_kwargs}")

        #  Dispatch based on balancing mode
        # Same tie-breaker as Stage 1: the BM objective prices gross storage
        # discharge at marginal_cost (see _build_balancing_extra_functionality),
        # so that cost must be strictly positive here too.
        apply_storage_dispatch_cost_floor(network, balancing_config, logger)

        # End-of-window state of charge: the Stage 1 path, rebuilt from the
        # saved net storage schedule from the state this stage starts in.
        end_soc_enabled, end_soc_penalty = resolve_storage_end_soc_config(balancing_config)
        stage1_soc = None
        if end_soc_enabled and len(network.storage_units) > 0 and len(wholesale_su.columns) > 0:
            stage1_soc = stage1_state_of_charge(
                network, wholesale_su.reindex(network.snapshots).fillna(0.0), legs=wholesale_su_legs
            )
            logger.info(
                f"End-of-window storage floor on: Stage 1 state of charge rebuilt for "
                f"{stage1_soc.shape[1]} units, penalty {end_soc_penalty:.0f} GBP/MWh"
            )
        else:
            end_soc_penalty = 0.0
            logger.info("End-of-window storage floor off (market.balancing.storage_end_soc)")

        reverse_leg_price = storage_reverse_leg_prices(network, balancing_config)

        # Decision D5 ("shift timing only"): the hydrogen store's band, Stage 1
        # path to Stage 1 + what the rest of the month can take back, closing on
        # Stage 1's month-end level (same switch and penalty as the storage floor).
        store_band = None
        # Decision 35 (task BM2050): hydrogen_links.store_band "none" leaves the
        # store bounded only by its own level once the turbines take part in the
        # BM; "stage1" (the default, every 2030 scenario) is decision D5's band.
        band_mode = resolve_store_band_mode(balancing_config)
        if band_mode == "none" and len(store_start):
            logger.info("Hydrogen store: no Stage 1 band (hydrogen_links.store_band none); the BM's hydrogen "
                        "use is limited by the store level only (decision 35)")
        if len(store_start) and end_soc_penalty > 0.0 and band_mode == "stage1":
            store_band = stage1_store_band(
                network, wholesale_links, store_start,
                float(resolve_hydrogen_link_config_tolerance(balancing_config)), logger,
            )
            if not store_band[0].empty:
                lower, upper = store_band
                logger.info(
                    f"Hydrogen store: Stage 1 level rebuilt from its link schedule, "
                    f"{lower.iloc[0].sum() / 1e3:,.1f} to {lower.iloc[-1].sum() / 1e3:,.1f} GWh "
                    f"(min {lower.sum(axis=1).min() / 1e3:,.1f}); month-end band "
                    f"{lower.iloc[-1].sum() / 1e3:,.2f}-{upper.iloc[-1].sum() / 1e3:,.2f} GWh; soft at "
                    f"{end_soc_penalty:.0f} GBP/MWh"
                )

        # Interconnector counter-trading (premium on |deviation| from Stage 1,
        # flow limit, spread rule): scripts/market/interconnector_counter_trade.py.
        counter_trade_names = prepare_interconnector_counter_trade(
            network, balancing_config, wholesale_links, logger,
        )
        if not counter_trade_names:
            logger.info(
                "Interconnector counter-trade pricing off "
                "(interconnectors fixed, or market.balancing.interconnector_counter_trade.enabled false)"
            )

        bm_mode = balancing_config.get("mode", "full_period")
        logger.info(f"Balancing mode: {bm_mode}")

        solve_start = time.time()

        if bm_mode == "rolling":
            #  Rolling BM: solve in windows 
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
                    gen_offer_tv=gen_offer_tv_objective,
                    gen_bid_tv=gen_bid_tv_objective,
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
                    optimize_kwargs=optimize_kwargs,
                    anchored_links=h2_link_names,
                    link_increase_prices=h2_link_increase,
                    link_decrease_prices=h2_link_decrease,
                    storage_end_soc_target=stage1_soc,
                    storage_end_soc_penalty=end_soc_penalty,
                    storage_reverse_leg_price=reverse_leg_price,
                    stage1_price=wholesale_price["wholesale_price"],
                    fixed_links=h2_held_links,
                    store_start=store_start,
                    store_band=store_band,
                    store_penalty=end_soc_penalty,
                    energy_floor=bm_energy_floor,
                    wholesale_su_legs=wholesale_su_legs,
                )
            )

            solve_time = time.time() - solve_start
            logger.info(f"Rolling BM solve completed in {solve_time:.2f}s")
            logger.info(f"Total BM redispatch cost: {bm_cost:,.2f}")

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
            #  Full-period solve (original behaviour) 
            fix_ics = resolve_fix_interconnectors(balancing_config, logger)
            final_target = (
                stage1_soc.loc[network.snapshots[-1]]
                if stage1_soc is not None and network.snapshots[-1] in stage1_soc.index
                else None
            )
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
                gen_offer_tv=gen_offer_tv_objective,
                gen_bid_tv=gen_bid_tv_objective,
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
                anchored_links=h2_link_names,
                link_increase_prices=h2_link_increase,
                link_decrease_prices=h2_link_decrease,
                storage_end_soc_target=final_target,
                storage_end_soc_penalty=end_soc_penalty,
                storage_reverse_leg_price=reverse_leg_price,
                fixed_links=h2_held_links,
                wholesale_su_legs=wholesale_su_legs,
            )
            final_lower = final_upper = None
            if store_band is not None and network.snapshots[-1] in store_band[0].index:
                final_lower = store_band[0].loc[network.snapshots[-1]]
                final_upper = store_band[1].loc[network.snapshots[-1]]
            store_callback = build_store_band_callback(final_lower, final_upper, end_soc_penalty, logger)
            log_hydro_constraint_setup(
                network,
                scenario_config,
                logger,
                context="Balancing hydro",
            )
            hydro_callback = build_hydro_constraints_callback(
                network,
                scenario_config,
                large_hydro_end_storage=large_hydro_end_floor(
                    large_hydro_net_inflow(network, scenario_config.get("renewables", {}).get("hydro", {})),
                    initialise_large_hydro_storage_state(
                        network, scenario_config.get("renewables", {}).get("hydro", {})),
                    network.snapshots[-1],
                ),
                logger=logger,
            )
            neso_boundary_callback = _build_neso_boundary_constraints_callback(
                network, scenario_config, logger
            )
            ic_capacity_callback = build_interconnector_shared_capacity_callback(
                network, logger
            )
            ic_piecewise_cost_callback = (
                None if fix_ics else build_piecewise_interconnector_cost_callback(
                    network,
                    logger,
                    baseline_dispatch=wholesale_links,
                    subtract_native_link_cost=False,
                )
            )
            counter_trade_callback = (
                None if fix_ics else build_counter_trade_callback(
                    network, wholesale_links, wholesale_price["wholesale_price"],
                    balancing_config, logger,
                )
            )
            network._bm_counter_trade_callback = counter_trade_callback
            if counter_trade_callback is not None:
                solver_options = no_wheeling_solver_options(solver_name, solver_options, balancing_config, logger)
            # Tie-break on |flow| of GB-internal DC links: objective only, stripped below.
            tie_break_cost = resolve_internal_dc_tie_break(market_config)
            tie_break_callback = build_internal_dc_tie_break_callback(
                network, tie_break_cost, logger, stage="balancing, full period"
            )

            logger.info("Starting BM optimization (full network constraints)...")

            try:
                outcome = solve_with_counterflow_lock(
                    network,
                    solver_name,
                    solver_options,
                    combine_extra_functionalities(
                        extra_func, tie_break_callback, counter_trade_callback, hydro_callback,
                        neso_boundary_callback,
                        ic_capacity_callback,
                        ic_piecewise_cost_callback,
                        store_callback,
                        build_min_output_energy_callback(bm_energy_floor),
                    ),
                    logger,
                    "BM full-period solve",
                    lock_counterflows=lock_material_counterflows,
                    **optimize_kwargs,
                )
            except SolveFailedError as exc:
                logger.error(f"BM optimization failed: {exc}")
                compute_iis = bool(
                    (balancing_config.get("diagnostics", {}) or {}).get(
                        "compute_iis", False
                    )
                )
                if compute_iis and str(solver_name).lower() == "gurobi" and exc.condition.lower() in {
                    "infeasible",
                    "infeasible_or_unbounded",
                    "inf_or_unbd",
                }:
                    _log_gurobi_iis(network, logger, context="BM full-period solve")
                raise RuntimeError(
                    f"BM optimization failed: {exc.status} ({exc.condition})"
                ) from exc

            solve_time = time.time() - solve_start
            logger.info(f"BM solve completed in {solve_time:.2f}s")
            logger.info(
                f"Status: {outcome.status}, Condition: {outcome.condition}, "
                f"solver rung {outcome.rung} ({outcome.rung_name}), "
                f"optimize {outcome.elapsed:.1f}s"
            )

            storage_energy_cost = _storage_energy_cost(network)
            end_soc_penalty_cost = 0.0
            if final_target is not None and end_soc_penalty > 0.0:
                shortfall = _storage_end_soc_shortfall(
                    network, final_target, participating_storage_units
                )
                end_soc_penalty_cost = float(shortfall.sum()) * end_soc_penalty
                _log_storage_end_soc_summary(pd.DataFrame([{
                    "window_end": network.snapshots[-1],
                    "target_mwh": float(final_target.reindex(shortfall.index).sum()),
                    "shortfall_mwh": float(shortfall.sum()),
                    "units_short": int((shortfall > 1e-3).sum()),
                    "penalty_gbp": end_soc_penalty_cost,
                }]), end_soc_penalty, logger)
            reverse_leg_cost = _storage_reverse_leg_cost(network, wholesale_su, reverse_leg_price,
                                                         wholesale_su_legs)
            end_soc_penalty_cost += reverse_leg_cost
            excess_penalty, excess_mwh = counter_trade_excess_penalty(network, counter_trade_callback)
            if excess_mwh > 1.0:
                logger.warning(
                    f"{excess_mwh:,.1f} MWh of interconnector movement beyond the counter-trade "
                    f"flow limit (penalty {excess_penalty:,.0f} stripped from the BM cost)"
                )
            end_soc_penalty_cost += excess_penalty
            if store_callback is not None:
                end_soc_penalty_cost += sum(store_band_violation(network, final_lower, final_upper)) * end_soc_penalty
            # Item C9: net of the Stage 1 valuation of the cable schedule.
            ic_baseline = (
                stage1_interconnector_trade_value(network)
                if ic_piecewise_cost_callback is not None else 0.0
            )
            network._bm_stage1_ic_valuation = ic_baseline
            end_soc_penalty_cost += ic_baseline
            # Soft hold on held links (5 Oct 2026): objective-only, stripped.
            hold_cost, hold_mwh = held_link_slack(network)
            if hold_mwh > 0.1:
                logger.warning(f"BM full-period solve: held links and storage moved {hold_mwh:,.1f} MWh off Stage 1 "
                               f"(penalty {hold_cost:,.0f} stripped from the BM cost)")
            end_soc_penalty_cost += hold_cost
            flex_gbp, flex_mwh = consumer_flex_rescheduled(network)
            record_consumer_flex_rescheduled(network, flex_gbp, flex_mwh, logger)
            end_soc_penalty_cost += flex_gbp
            # Internal HVDC tie-break: objective-only, stripped (reported separately).
            tie_break_gbp, tie_break_mwh = internal_dc_tie_break_cost(network, tie_break_callback)
            end_soc_penalty_cost += tie_break_gbp
            if tie_break_callback is not None:
                record_internal_dc_tie_break(
                    network, "balancing", tie_break_cost, tie_break_callback.links,
                    tie_break_gbp, tie_break_mwh, logger,
                )
            bm_cost = network.objective - storage_energy_cost - end_soc_penalty_cost
            logger.info(
                f"Total BM redispatch cost: {bm_cost:,.2f} "
                f"(storage dispatch cost term {storage_energy_cost:,.2f} and end-of-window "
                f"storage penalty plus reverse-leg guard {end_soc_penalty_cost - ic_baseline:,.2f} "
                f"(reverse-leg {reverse_leg_cost:,.2f}, internal HVDC tie-break {tie_break_gbp:,.2f}) "
                f"and the Stage 1 interconnector valuation "
                f"{ic_baseline:,.2f} stripped from the objective)"
            )

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

        #  Compute redispatch summary 
        # The pure redispatch cost is what analyze_market reports as
        # total_redispatch_cost; the rolling path never set it before.
        # Network.objective is a read-only property over _objective, which is
        # what optimize() writes and export_to_netcdf persists.
        for label, regulariser, direction in (
            ("Turn-up floor (decision 20)", offer_regulariser, "up"),
            ("State-first step (item BM1)", bid_regulariser, "down"),
        ):
            regulariser_cost = _offer_regulariser_cost(physical_gen, wholesale_gen, regulariser, network, direction)
            if regulariser_cost:
                bm_cost -= regulariser_cost
                logger.info(
                    f"{label} term {regulariser_cost:,.2f} stripped from the BM cost "
                    f"(objective only), leaving {bm_cost:,.2f}"
                )
        network._objective = bm_cost
        _log_storage_energy_balance(network, logger)
        _log_counter_trade_summary(
            network,
            network.links_t.p0 if len(network.links_t.p0) > 0 else None,
            wholesale_links,
            getattr(network, "_bm_counter_trade_callback", None),
            logger,
        )

        logger.info("=" * 80)
        logger.info("COMPUTING REDISPATCH VOLUMES AND COSTS")
        logger.info("=" * 80)

        # Use physical_gen index as the canonical snapshot index
        snaps = physical_gen.index if len(physical_gen) > 0 else network.snapshots

        # The churn sink is held at Stage 1, not redispatched: out of the volumes (F22).
        ws_gen_aligned, physical_gen_accounted, _ = align_redispatch_generators(
            wholesale_gen, physical_gen, snaps, logger)
        ws_su_aligned = (
            wholesale_su.reindex(snaps).reindex(
                columns=physical_su.columns, fill_value=0.0
            )
            if len(physical_su.columns) > 0
            else pd.DataFrame(index=snaps)
        )

        gen_summary, su_summary = compute_redispatch_volumes(
            wholesale_gen=ws_gen_aligned,
            physical_gen=physical_gen_accounted,
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

        link_summary = pd.DataFrame()
        if h2_link_names:
            physical_links = (
                network.links_t.p0.reindex(index=snaps)
                if len(network.links_t.p0) > 0 else pd.DataFrame(index=snaps)
            )
            link_summary = compute_link_redispatch_volumes(
                wholesale_links, physical_links, h2_link_increase, h2_link_decrease,
                network, logger,
                bus_carriers=resolve_hydrogen_link_config(balancing_config)["bus_carriers"],
            )

        redispatch_summary = pd.concat(
            [frame for frame in (gen_summary, su_summary, link_summary) if not frame.empty]
            or [gen_summary],
            ignore_index=True,
        )
        redispatch_summary.to_csv(snakemake.output.redispatch_summary_csv, index=False)
        logger.info(f"Saved: {snakemake.output.redispatch_summary_csv}")

        #  Constraint costs by carrier and region 
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
                f"Constraint costs: offers={total_offer:,.0f}, "
                f"bids={total_bid:,.0f}, total={total_net:,.0f}"
            )
        else:
            pd.DataFrame().to_csv(snakemake.output.constraint_costs_csv)
            logger.warning("No redispatch  zero constraint costs")

        #  Congestion analysis 
        # In rolling mode, results are written back to the network object
        # after the rolling solve completes, so congestion analysis works.
        congestion = identify_congested_boundaries(
            network, threshold=0.95, logger=logger
        )
        congestion.to_csv(snakemake.output.congestion_csv, index=False)
        logger.info(f"Saved: {snakemake.output.congestion_csv}")

        #  Price comparison 
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
                f"{ws_price_aligned.mean():.2f}/MWh, "
                f"nodal mean {mean_nodal.mean():.2f}/MWh, "
                f"max spread {price_spread.max():.2f}/MWh"
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

        #  Save solved BM network 
        _clear_runtime_bm_attrs(network)
        save_network(network, snakemake.output.network, custom_logger=logger)
        logger.info(f"Saved BM network: {snakemake.output.network}")

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(
            f"STAGE 2 COMPLETE  BALANCING MECHANISM SOLVED "
            f"(Total: {total_time:.2f}s, BM cost: {bm_cost:,.2f})"
        )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"FATAL ERROR in BM solve: {e}", exc_info=True)
        raise
