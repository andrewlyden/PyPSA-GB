"""Hydro and pumped-hydro operational constraints for PyPSA solves."""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd


LARGE_HYDRO_CARRIERS = {"large_hydro"}
PUMPED_HYDRO_CARRIERS = {"Pumped Storage Hydroelectricity", "pumped_hydro"}

# ── Large hydro: power ceiling apart from inflow (OW3, 3 Oct 2026) ──────────
#
# The reservoir model read inflow off the power ceiling, inflow = p_max_pu x
# p_nom x inflow_profile_factor, so the ceiling was only 1 / 0.55 = 1.82 x
# inflow. On the CP30 2030 fleet (1,896 MW) that is 0.38 of p_nom on average,
# 0.15 in July: Stage 1 swung between zero and that ceiling and could not catch
# up after a glut, spilling 345 GWh, and the BM had no headroom to put back the
# 777 GWh it turned down (3,491 GWh of inflow, 2,366 GWh dispatched; scratchpad
# ow3/ow3_ow5_report.md). ``renewables.hydro.large_hydro.power_ceiling: availability``
# writes the inflow at build into its own series (LARGE_HYDRO_INFLOW_SERIES, MW
# per MW of p_nom) and sets p_max_pu to ``availability_pu`` (see
# scripts/renewables/large_hydro_reservoir.py, run by finalize_network).
# Every reader of the inflow goes through large_hydro_inflow_pu, which takes the
# series where a unit has one and the old p_max_pu x inflow_profile_factor
# otherwise, so networks built before keep their meaning. Default
# ``inflow_scaled`` is today's behaviour. Changing it requires a rebuild.
LARGE_HYDRO_INFLOW_SERIES = "hydro_inflow_pu"
LARGE_HYDRO_TRAJECTORY_START = "hydro_trajectory_start_h"
LARGE_HYDRO_TRAJECTORY_END = "hydro_trajectory_end_h"
LARGE_HYDRO_CEILING_MODES = ("inflow_scaled", "availability")
LARGE_HYDRO_TRAJECTORY_KEYS = ("mean_level_hours", "smoothing_days")
# Where large_hydro_net_inflow leaves the run's end level for large_hydro_end_floor.
_RUN_END_ATTR = "large_hydro_run_end_mwh"


def combine_extra_functionalities(*callbacks: Optional[Callable]) -> Optional[Callable]:
    """Combine multiple PyPSA extra_functionality callbacks into one."""
    active_callbacks = [callback for callback in callbacks if callback is not None]
    if not active_callbacks:
        return None

    def combined(network, snapshots):
        for callback in active_callbacks:
            callback(network, snapshots)

    return combined


def get_large_hydro_names(network) -> list[str]:
    """Return large-hydro generator names present in the network."""
    if len(network.generators) == 0:
        return []

    carriers = network.generators.get(
        "carrier", pd.Series(index=network.generators.index, dtype=object)
    )
    return [
        name for name, carrier in carriers.items()
        if str(carrier) in LARGE_HYDRO_CARRIERS
    ]


def _existing_pumped_hydro_site_names(hydro_config: Optional[dict]) -> list[str]:
    """Names of the configured existing pumped-hydro sites (``renewables.hydro.pumped_hydro.existing_sites``)."""
    sites = ((hydro_config or {}).get("pumped_hydro", {}) or {}).get("existing_sites") or []
    names = []
    for site in sites:
        name = site.get("name") if isinstance(site, dict) else site
        if name:
            names.append(str(name).strip())
    return [n for n in names if n]


def get_pumped_hydro_names(network, hydro_config: Optional[dict] = None) -> list[str]:
    """Return pumped-hydro storage-unit names present in the network.

    A unit is pumped hydro if its carrier says so or if it is one of the
    existing pumped-hydro stations (``renewables.hydro.pumped_hydro.existing_sites``). CP30
    books those inside its LDES total, so they carry carrier ``LAES`` (fix S3,
    27 Sep 2026) and the carrier test alone missed Dinorwig, Cruachan,
    Ffestiniog and Foyers: the pumped-hydro SoC floors never applied to them
    (review 27 Sep 2026, item C2). They are recognised by an ``existing_site``
    attribute when the network carries one, else by the name add_storage gives
    them (``..._existing_<site>_<n>``).
    """
    if len(network.storage_units) == 0:
        return []

    su = network.storage_units
    carriers = su.get("carrier", pd.Series(index=su.index, dtype=object))
    selected = {name for name, carrier in carriers.items() if str(carrier) in PUMPED_HYDRO_CARRIERS}
    sites = _existing_pumped_hydro_site_names(hydro_config)
    if sites:
        if "existing_site" in su.columns:
            labelled = su["existing_site"].fillna("").astype(str).str.strip()
            selected |= set(su.index[labelled.isin(sites)])
        for name in su.index:
            if any(f"_existing_{site}_" in str(name) or str(name).endswith(f"_existing_{site}") for site in sites):
                selected.add(name)
    return [name for name in su.index if name in selected]


def initialise_large_hydro_storage_state(network, hydro_config: dict) -> Optional[pd.Series]:
    """Create initial reservoir energy state for large-hydro generators.

    ``initial_storage_hours`` x p_nom, or under ``month_end: trajectory`` the
    annual path's level just before the run's first snapshot.
    """
    names = get_large_hydro_names(network)
    large_cfg = hydro_config.get("large_hydro", {})

    if not names or not large_cfg.get("reservoir_model", False):
        return None

    if resolve_large_hydro_month_end(hydro_config) == "trajectory":
        start, _ = large_hydro_trajectory_levels(network, hydro_config, names)
        return start

    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    reservoir_capacity = p_nom * float(large_cfg.get("reservoir_capacity_hours", 168.0))
    initial_storage = p_nom * float(large_cfg.get("initial_storage_hours", 48.0))
    return initial_storage.clip(lower=0.0, upper=reservoir_capacity)


def log_hydro_constraint_setup(network, scenario_config: dict, logger, context: str = "Hydro",
                               min_soc_override: Optional[float] = None) -> None:
    """Log hydro policy summaries from the outer solve workflow.

    ``min_soc_override`` is the pumped-hydro floor the solve passes to
    ``build_hydro_constraints_callback`` (Stage 1: wholesale_min_soc_fraction), so
    the log states the floor actually applied.
    """
    hydro_config = scenario_config.get("renewables", {}).get("hydro", {})
    if not hydro_config.get("enabled", False):
        return

    large_hydro_names = get_large_hydro_names(network)
    large_cfg = hydro_config.get("large_hydro", {})
    if large_hydro_names and large_cfg.get("reservoir_model", False):
        inflow_series = _generator_series(network, LARGE_HYDRO_INFLOW_SERIES)
        separate = [] if inflow_series is None else [n for n in large_hydro_names if n in inflow_series.columns]
        logger.info(
            "%s: large-hydro reservoir constraints enabled for %d generators "
            "(initial storage %.1fh, reservoir capacity %.1fh, inflow factor %.2f; "
            "%d with the inflow series %s apart from the power ceiling)",
            context,
            len(large_hydro_names),
            float(large_cfg.get("initial_storage_hours", 48.0)),
            float(large_cfg.get("reservoir_capacity_hours", 168.0)),
            float(large_cfg.get("inflow_profile_factor", 0.55)),
            len(separate),
            LARGE_HYDRO_INFLOW_SERIES,
        )
        if resolve_large_hydro_month_end(hydro_config) == "trajectory":
            start, end = large_hydro_trajectory_levels(network, hydro_config, large_hydro_names)
            logger.info(
                "%s: large-hydro month_end trajectory: the run starts at %.1f GWh and ends at or "
                "above %.1f GWh (annual path, %s)",
                context, float(start.sum()) / 1e3, float(end.sum()) / 1e3,
                "built series" if _generator_series(network, LARGE_HYDRO_TRAJECTORY_START) is not None
                else "no built series: flat at mean_level_hours",
            )

    pumped_hydro_names = get_pumped_hydro_names(network, hydro_config)
    pumped_cfg = hydro_config.get("pumped_hydro", {})
    policy_parts = []

    # The floor this solve actually applies: the override (Stage 1's
    # wholesale_min_soc_fraction) when given, else min_soc_fraction.
    min_soc_fraction = min_soc_override if min_soc_override is not None else pumped_cfg.get("min_soc_fraction")
    if min_soc_fraction is not None:
        policy_parts.append(
            f"min SoC {100 * float(min_soc_fraction):.0f}%"
            + (" (wholesale override)" if min_soc_override is not None else "")
        )

    final_soc_fraction = pumped_cfg.get("final_soc_fraction")
    if final_soc_fraction is not None:
        policy_parts.append(f"final SoC {100 * float(final_soc_fraction):.0f}%")

    if pumped_hydro_names and policy_parts:
        su = network.storage_units.loc[pumped_hydro_names]
        energy = (pd.to_numeric(su["p_nom"], errors="coerce").fillna(0.0)
                  * pd.to_numeric(su["max_hours"], errors="coerce").fillna(0.0))
        logger.info(
            "%s: pumped-hydro SOC policy enabled for %d storage units, %.0f MW / %.1f GWh "
            "(carriers %s) (%s)",
            context,
            len(pumped_hydro_names),
            float(su["p_nom"].sum()),
            float(energy.sum()) / 1e3,
            sorted(su["carrier"].astype(str).unique()),
            ", ".join(policy_parts),
        )
    elif policy_parts:
        logger.info("%s: pumped-hydro SOC policy configured (%s) but no pumped-hydro storage units found",
                    context, ", ".join(policy_parts))


def _get_snapshot_weights(network, snapshots) -> pd.Series:
    """Return per-snapshot energy weights in hours."""
    if hasattr(network, "snapshot_weightings") and hasattr(network.snapshot_weightings, "generators"):
        return network.snapshot_weightings.generators.reindex(snapshots).fillna(1.0)
    return pd.Series(1.0, index=pd.Index(snapshots))


def _generator_series(network, attr: str) -> Optional[pd.DataFrame]:
    """``network.generators_t[attr]`` when it exists and has columns, else None."""
    series = getattr(network, "generators_t", None)
    if series is None:
        return None
    try:
        frame = series[attr]
    except (KeyError, AttributeError):
        return None
    return frame if isinstance(frame, pd.DataFrame) and frame.shape[1] else None


def large_hydro_inflow_pu(network, snapshots, hydro_config: dict, names: list[str]) -> pd.DataFrame:
    """Inflow per MW of p_nom (snapshots x units) for large-hydro generators.

    A unit with a column in ``generators_t[LARGE_HYDRO_INFLOW_SERIES]`` (power
    ceiling apart from inflow, built under ``power_ceiling: availability``)
    takes it as is; every other unit keeps the reservoir model's convention,
    ``p_max_pu x inflow_profile_factor``.
    """
    large_cfg = (hydro_config or {}).get("large_hydro", {}) or {}
    inflow_factor = float(large_cfg.get("inflow_profile_factor", 0.55))
    snapshots = pd.Index(snapshots)

    static_p_max = network.generators.loc[names, "p_max_pu"].fillna(1.0).astype(float)

    if getattr(network.generators_t, "p_max_pu", pd.DataFrame()).empty:
        p_max = pd.DataFrame(
            {name: static_p_max[name] for name in names},
            index=snapshots,
        )
    else:
        p_max = network.generators_t.p_max_pu.reindex(index=snapshots).copy()
        for name in names:
            if name in p_max.columns:
                p_max[name] = p_max[name].fillna(static_p_max[name])
            else:
                p_max[name] = static_p_max[name]
        p_max = p_max[names]

    inflow = p_max * inflow_factor
    series = _generator_series(network, LARGE_HYDRO_INFLOW_SERIES)
    if series is not None:
        present = [name for name in names if name in series.columns]
        if present:
            inflow[present] = series.reindex(index=snapshots)[present].fillna(0.0).astype(float)
    return inflow


def large_hydro_inflow_power(network, snapshots, hydro_config: dict, names: list[str]) -> pd.DataFrame:
    """Inflow (MW, snapshots x units) for large-hydro generators (``large_hydro_inflow_pu`` x p_nom)."""
    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    return large_hydro_inflow_pu(network, snapshots, hydro_config, names).mul(p_nom, axis=1)


def _get_large_hydro_inflow_energy(network, snapshots, hydro_config: dict, names: list[str]) -> pd.DataFrame:
    """Return per-snapshot inflow energy (MWh) for each large-hydro generator."""
    inflow_power = large_hydro_inflow_power(network, snapshots, hydro_config, names)
    weights = _get_snapshot_weights(network, snapshots)
    return inflow_power.mul(weights, axis=0)


def _get_large_hydro_min_dispatch_energy(network, snapshots, names: list[str]) -> pd.DataFrame:
    """Per-snapshot energy (MWh) each large-hydro generator must produce at its floor."""
    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    static = pd.to_numeric(
        network.generators.loc[names, "p_min_pu"]
        if "p_min_pu" in network.generators.columns else pd.Series(0.0, index=names),
        errors="coerce",
    ).fillna(0.0)
    floor = pd.DataFrame({name: static[name] for name in names}, index=pd.Index(snapshots))
    varying = getattr(network.generators_t, "p_min_pu", pd.DataFrame())
    if not varying.empty:
        present = [name for name in names if name in varying.columns]
        if present:
            floor[present] = varying.reindex(index=snapshots)[present].fillna(floor[present])
    weights = _get_snapshot_weights(network, snapshots)
    return floor.clip(lower=0.0).mul(p_nom, axis=1).mul(weights, axis=0)


# ── Large-hydro water returned by the end of the run ─────────────────────
#
# Each month's Stage 1 and BM start the reservoir at initial_storage_hours x
# p_nom (~91 GWh on the CP30 fleet) and nothing asked for it back: Stage 1
# dispatch exceeded inflow by 26-30% every month (Jan 2030 393.8 against 310.3
# GWh), about 1 TWh a year of free water. ``month_end: at_least_initial`` (the
# large-hydro twin of market.storage_soc) makes the store end the run at or
# above where it started. A rolling solve is myopic, so a condition in the last
# window alone could be unreachable: inflow cannot refill a buffer drawn down
# over the month in one day. Every window therefore ends at or above
#     floor = run_initial - sum over later snapshots of (inflow - minimum dispatch),
# the lowest store from which the run can still hand the buffer back by
# producing only its minimum afterwards. The floor rises to run_initial exactly
# in the window holding the run's last snapshot, and holds by induction from
# the start (the first window starts at run_initial). Each window's target is
# also capped at what that window can reach at minimum output, so it can never
# be infeasible. Inter-day use of the buffer is still allowed; the water is
# only lent. Defaults must match config/defaults/ renewables.hydro.large_hydro.
#
# ``month_end: trajectory`` (OW3, 3 Oct 2026) replaces the fixed start and end
# by an annual path built from the inflow alone: the release is the inflow's
# centred moving average over ``trajectory.smoothing_days`` (cyclic over the
# built year, so it releases exactly the inflow), and the store follows the
# cumulative difference around ``trajectory.mean_level_hours`` x p_nom. Each
# month (or week) starts at the path's level before its first snapshot and every
# window ends at or above floor = path level after the run's last snapshot -
# later net inflow, the same induction as above with the run's end level in
# place of run_initial. The twelve months are solved independently on one
# network, so carrying the store month to month would chain them; the path makes
# their boundaries agree instead: month m ends where month m + 1 starts, so the
# year releases the inflow and the monthly step of the proxy inflow profile is
# released across each month boundary rather than at it. Built into the network
# by prepare_large_hydro_reservoir (series LARGE_HYDRO_TRAJECTORY_START/END, hours
# of p_nom); without them the path is flat at mean_level_hours.
LARGE_HYDRO_MONTH_END_MODES = ("at_least_initial", "free", "trajectory")


def resolve_large_hydro_month_end(hydro_config: dict) -> str:
    """``renewables.hydro.large_hydro.month_end`` validated (default ``at_least_initial``)."""
    mode = str((hydro_config or {}).get("large_hydro", {}).get("month_end", "at_least_initial")).lower()
    if mode not in LARGE_HYDRO_MONTH_END_MODES:
        raise ValueError(
            f"renewables.hydro.large_hydro.month_end must be one of {LARGE_HYDRO_MONTH_END_MODES}, got {mode!r}"
        )
    return mode


def resolve_large_hydro_ceiling(hydro_config: dict) -> tuple[str, Optional[float]]:
    """``(renewables.hydro.large_hydro.power_ceiling, availability_pu)`` validated.

    ``inflow_scaled`` (default, today's behaviour): the ceiling is the inflow
    profile, inflow = p_max_pu x inflow_profile_factor. ``availability``: the
    ceiling is ``availability_pu`` (required, in (0, 1]) and the inflow has its
    own series; it needs the reservoir model, which is what budgets the inflow.
    """
    large = (hydro_config or {}).get("large_hydro", {}) or {}
    mode = str(large.get("power_ceiling", "inflow_scaled")).lower()
    if mode not in LARGE_HYDRO_CEILING_MODES:
        raise ValueError(
            f"renewables.hydro.large_hydro.power_ceiling must be one of {LARGE_HYDRO_CEILING_MODES}, got {mode!r}"
        )
    if mode == "inflow_scaled":
        return mode, None
    if not ((hydro_config or {}).get("enabled", False) and large.get("reservoir_model", False)):
        raise ValueError(
            "renewables.hydro.large_hydro.power_ceiling 'availability' needs renewables.hydro.enabled and "
            "large_hydro.reservoir_model: without the water budget the ceiling would be the energy"
        )
    value = large.get("availability_pu")
    if value is None or not 0.0 < float(value) <= 1.0:
        raise ValueError(
            f"renewables.hydro.large_hydro.power_ceiling 'availability' needs availability_pu in (0, 1], got {value!r}"
        )
    return mode, float(value)


def resolve_large_hydro_trajectory(hydro_config: dict) -> Optional[dict]:
    """``renewables.hydro.large_hydro.trajectory`` validated when ``month_end`` is ``trajectory``, else None.

    Both keys are required (there is no config/defaults/ entry to fall back on):
    ``mean_level_hours`` in [0, reservoir_capacity_hours], ``smoothing_days`` >= 0.
    """
    if resolve_large_hydro_month_end(hydro_config) != "trajectory":
        return None
    large = (hydro_config or {}).get("large_hydro", {}) or {}
    block = large.get("trajectory") or {}
    unknown = sorted(set(block) - set(LARGE_HYDRO_TRAJECTORY_KEYS))
    if unknown:
        raise ValueError(f"renewables.hydro.large_hydro.trajectory: unknown keys {unknown}")
    missing = [key for key in LARGE_HYDRO_TRAJECTORY_KEYS if block.get(key) is None]
    if missing:
        raise ValueError(f"renewables.hydro.large_hydro.month_end 'trajectory' needs trajectory.{missing}")
    capacity = float(large.get("reservoir_capacity_hours", 168.0))
    mean = float(block["mean_level_hours"])
    smoothing = float(block["smoothing_days"])
    if not 0.0 <= mean <= capacity:
        raise ValueError(
            f"renewables.hydro.large_hydro.trajectory.mean_level_hours must lie in [0, {capacity:g}], got {mean:g}"
        )
    if smoothing < 0.0:
        raise ValueError(f"renewables.hydro.large_hydro.trajectory.smoothing_days must be >= 0, got {smoothing:g}")
    return {"mean_level_hours": mean, "smoothing_days": smoothing, "reservoir_capacity_hours": capacity}


def large_hydro_trajectory_levels(network, hydro_config: dict, names: list[str]) -> tuple[pd.Series, pd.Series]:
    """``(store before the run's first snapshot, store after its last)`` in MWh on the annual path.

    Reads the built path (LARGE_HYDRO_TRAJECTORY_START / END, hours of p_nom) on
    the run's own snapshots; a unit without it sits at ``mean_level_hours``.
    Clipped to the reservoir.
    """
    traj = resolve_large_hydro_trajectory(hydro_config)
    if traj is None:
        raise ValueError("large_hydro_trajectory_levels needs renewables.hydro.large_hydro.month_end 'trajectory'")
    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    start_h = pd.Series(traj["mean_level_hours"], index=names, dtype=float)
    end_h = start_h.copy()
    start_series = _generator_series(network, LARGE_HYDRO_TRAJECTORY_START)
    end_series = _generator_series(network, LARGE_HYDRO_TRAJECTORY_END)
    snapshots = network.snapshots
    if start_series is not None and end_series is not None and len(snapshots):
        present = [n for n in names if n in start_series.columns and n in end_series.columns]
        if present:
            first = start_series.reindex(index=[snapshots[0]])[present].iloc[0]
            last = end_series.reindex(index=[snapshots[-1]])[present].iloc[-1]
            start_h[present] = first.fillna(start_h[present])
            end_h[present] = last.fillna(end_h[present])
    capacity = traj["reservoir_capacity_hours"]
    return (start_h.clip(0.0, capacity) * p_nom, end_h.clip(0.0, capacity) * p_nom)


def large_hydro_net_inflow(network, hydro_config: dict) -> Optional[pd.DataFrame]:
    """Inflow minus minimum-output energy per snapshot of the whole run (MWh), or None.

    Under ``month_end: trajectory`` the frame also carries the run's end level
    (``attrs``), which ``large_hydro_end_floor`` holds every window to.
    """
    hydro_config = hydro_config or {}
    names = get_large_hydro_names(network)
    large_cfg = hydro_config.get("large_hydro", {})
    if (
        not hydro_config.get("enabled", False)
        or not names
        or not large_cfg.get("reservoir_model", False)
        or resolve_large_hydro_month_end(hydro_config) == "free"
    ):
        return None
    snapshots = network.snapshots
    net = (
        _get_large_hydro_inflow_energy(network, snapshots, hydro_config, names)
        - _get_large_hydro_min_dispatch_energy(network, snapshots, names)
    )
    if resolve_large_hydro_month_end(hydro_config) == "trajectory":
        _, end = large_hydro_trajectory_levels(network, hydro_config, names)
        # A plain dict: pandas compares attrs with == when it combines frames.
        net.attrs[_RUN_END_ATTR] = {str(k): float(v) for k, v in end.items()}
    return net


def large_hydro_end_floor(
    net_inflow: Optional[pd.DataFrame], run_initial: Optional[pd.Series], window_end
) -> Optional[pd.Series]:
    """Store each large-hydro unit must hold at ``window_end`` (see the section note).

    The run's end level is ``run_initial`` (``at_least_initial``) or, under
    ``month_end: trajectory``, the path's level the net-inflow frame carries.
    """
    if net_inflow is None:
        return None
    run_end = net_inflow.attrs.get(_RUN_END_ATTR)
    target = pd.Series(run_end, dtype=float) if run_end is not None else run_initial
    if target is None:
        return None
    remaining = net_inflow.loc[net_inflow.index > window_end].sum(axis=0)
    return target.reindex(net_inflow.columns).fillna(0.0) - remaining


def build_hydro_constraints_callback(
    network,
    scenario_config: dict,
    large_hydro_storage_state: Optional[pd.Series] = None,
    pumped_min_soc_override: Optional[float] = None,
    large_hydro_end_storage: Optional[pd.Series] = None,
    logger=None,
) -> Optional[Callable]:
    """Build an extra_functionality callback for hydro-specific constraints.

    Parameters
    ----------
    pumped_min_soc_override : float, optional
        If provided, overrides ``pumped_hydro.min_soc_fraction`` from config.
        Used by the wholesale solve to enforce a higher SoC floor, preserving
        headroom for BM increases.
    large_hydro_end_storage : pd.Series, optional
        Store (MWh) each large-hydro unit must hold after this solve's last
        snapshot (``large_hydro_end_floor``). Capped at what the solve can reach
        from ``large_hydro_storage_state`` at minimum output, so never infeasible.
    """
    hydro_config = scenario_config.get("renewables", {}).get("hydro", {})
    if not hydro_config.get("enabled", False):
        return None

    large_hydro_names = get_large_hydro_names(network)
    pumped_hydro_names = get_pumped_hydro_names(network, hydro_config)
    if not large_hydro_names and not pumped_hydro_names:
        return None

    large_cfg = hydro_config.get("large_hydro", {})
    pumped_cfg = hydro_config.get("pumped_hydro", {})
    if large_hydro_storage_state is None and large_hydro_names:
        large_hydro_storage_state = initialise_large_hydro_storage_state(network, hydro_config)

    def hydro_extra_functionality(n, snapshots):
        import xarray as xr

        model = n.model

        if large_hydro_names and large_cfg.get("reservoir_model", False) and "Generator-p" in model.variables:
            names = [name for name in large_hydro_names if name in n.generators.index]
            if names:
                if large_hydro_storage_state is None:
                    initial_storage = pd.Series(0.0, index=names)
                else:
                    initial_storage = large_hydro_storage_state.reindex(names).fillna(0.0)
                inflow_energy = _get_large_hydro_inflow_energy(n, snapshots, hydro_config, names)
                inflow_cumsum = inflow_energy.cumsum(axis=0)

                gen_p = model.variables["Generator-p"].sel({"name": names})
                weights = _get_snapshot_weights(n, snapshots)
                weight_da = xr.DataArray(
                    weights.values,
                    dims=["snapshot"],
                    coords={"snapshot": snapshots},
                )
                weighted_dispatch = gen_p * weight_da

                for idx in range(len(snapshots)):
                    lhs = weighted_dispatch.isel(snapshot=slice(0, idx + 1)).sum("snapshot")
                    rhs = initial_storage.values + inflow_cumsum.iloc[idx].values
                    model.add_constraints(
                        lhs <= rhs,
                        name=f"Generator-large_hydro_water_budget_{idx}",
                    )

                if large_hydro_end_storage is not None and len(snapshots) > 0:
                    # Store after the last snapshot = initial + inflow - dispatch
                    # >= target, the target capped at the minimum-output store.
                    inflow_total = inflow_cumsum.iloc[-1].reindex(names).fillna(0.0)
                    min_total = _get_large_hydro_min_dispatch_energy(n, snapshots, names).sum(axis=0)
                    reachable = initial_storage + inflow_total - min_total
                    wanted = large_hydro_end_storage.reindex(names).fillna(-float("inf"))
                    target = pd.concat([wanted, reachable], axis=1).min(axis=1)
                    # A floor at or below zero is already implied by the water budget.
                    binding = (wanted > 0.0) & (target > 0.0)
                    if binding.any():
                        keep = list(target.index[binding])
                        allowance = (initial_storage + inflow_total - target).reindex(keep)
                        allowance_da = xr.DataArray(
                            allowance.values, dims=["name"], coords={"name": keep}
                        )
                        model.add_constraints(
                            weighted_dispatch.sel({"name": keep}).sum("snapshot") <= allowance_da,
                            name="Generator-large_hydro_end_storage",
                        )
                        if logger is not None:
                            capped = int((reachable.reindex(keep) < wanted.reindex(keep) - 1e-6).sum())
                            logger.info(
                                "Large-hydro end storage: %d units held at or above %.1f GWh "
                                "(run-initial floor %.1f GWh)%s",
                                len(keep),
                                float(target.reindex(keep).clip(lower=0.0).sum()) / 1e3,
                                float(wanted.reindex(keep).clip(lower=0.0).sum()) / 1e3,
                                f"; {capped} capped at the minimum-output store" if capped else "",
                            )

        if pumped_hydro_names and "StorageUnit-state_of_charge" in model.variables:
            names = [name for name in pumped_hydro_names if name in n.storage_units.index]
            if names:
                energy_cap = (
                    n.storage_units.loc[names, "p_nom"].astype(float)
                    * n.storage_units.loc[names, "max_hours"].astype(float)
                )
                state_of_charge = model.variables["StorageUnit-state_of_charge"].sel({"name": names})
                energy_cap_da = xr.DataArray(
                    energy_cap.reindex(names).values,
                    dims=["name"],
                    coords={"name": names},
                )

                min_soc_fraction = pumped_min_soc_override if pumped_min_soc_override is not None else pumped_cfg.get("min_soc_fraction")
                if min_soc_fraction is not None:
                    model.add_constraints(
                        state_of_charge >= energy_cap_da * float(min_soc_fraction),
                        name="StorageUnit-pumped_hydro_min_soc",
                    )

                final_soc_fraction = pumped_cfg.get("final_soc_fraction")
                if final_soc_fraction is not None and len(snapshots) > 0:
                    model.add_constraints(
                        state_of_charge.isel(snapshot=-1) >= energy_cap_da * float(final_soc_fraction),
                        name="StorageUnit-pumped_hydro_final_soc",
                    )

    return hydro_extra_functionality


def update_large_hydro_storage_state(
    network,
    snapshots,
    scenario_config: dict,
    current_storage: Optional[pd.Series],
) -> Optional[pd.Series]:
    """Advance large-hydro reservoir state after a solved window."""
    hydro_config = scenario_config.get("renewables", {}).get("hydro", {})
    large_cfg = hydro_config.get("large_hydro", {})
    names = get_large_hydro_names(network)

    if current_storage is None or not names or not large_cfg.get("reservoir_model", False):
        return current_storage

    reservoir_capacity = (
        network.generators.loc[names, "p_nom"].astype(float)
        * float(large_cfg.get("reservoir_capacity_hours", 168.0))
    )
    inflow_energy = _get_large_hydro_inflow_energy(network, snapshots, hydro_config, names).sum(axis=0)

    dispatch_df = getattr(network.generators_t, "p", pd.DataFrame())
    if dispatch_df.empty:
        dispatch_energy = pd.Series(0.0, index=names)
    else:
        dispatch = dispatch_df.reindex(index=snapshots, columns=names, fill_value=0.0)
        weights = _get_snapshot_weights(network, snapshots)
        dispatch_energy = dispatch.mul(weights, axis=0).sum(axis=0)

    next_storage = current_storage.reindex(names).fillna(0.0) + inflow_energy - dispatch_energy
    return next_storage.clip(lower=0.0, upper=reservoir_capacity)