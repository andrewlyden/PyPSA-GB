"""Large-hydro reservoir at build: the power ceiling apart from inflow, and the annual path.

OW3 (3 Oct 2026; scratchpad ow3/ow3_ow5_report.md). The reservoir model
(``scripts/solve/hydro_constraints.py``) read the inflow off the power ceiling,
inflow = p_max_pu x p_nom x inflow_profile_factor, and the CP30 proxy profile is
written in that convention (``scripts/renewables/large_hydro_inflow.py``), so the
ceiling was only 1.82 x mean inflow. Two build switches, both no-ops under the
defaults:

* ``hydro.large_hydro.power_ceiling: availability`` keeps that same inflow in its
  own series (``generators_t.hydro_inflow_pu``) and sets ``p_max_pu`` to
  ``availability_pu``;
* ``hydro.large_hydro.month_end: trajectory`` writes the annual reservoir path the
  months start and end on (``generators_t.hydro_trajectory_start_h`` / ``_end_h``,
  hours of p_nom), from the full built year of inflow.

Kept apart from ``large_hydro_inflow.py``, a declared input of the renewable
integration rule, so these steps do not re-trigger it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Called once on the finished network (scripts/solve/finalize_network.py), so it
# sees every large-hydro unit whatever built it, after the p_max_pu cleaning there.


def large_hydro_release_path(inflow_pu: pd.DataFrame, smoothing_snapshots: int) -> pd.DataFrame:
    """Cyclic centred moving average of the inflow over ``smoothing_snapshots``.

    The series is wrapped at both ends (the built year is treated as cyclic), so
    with equal snapshot weights the release sums to the inflow exactly. Zero or
    one snapshot returns the inflow itself; a window longer than the series is
    cut to its length (the flat annual mean).
    """
    n = len(inflow_pu)
    k = int(min(max(int(smoothing_snapshots), 0), n))
    if k <= 1:
        return inflow_pu.astype(float).copy()
    values = inflow_pu.to_numpy(dtype=float)
    extended = np.concatenate([values[-k:], values, values[:k]], axis=0)
    rolled = pd.DataFrame(extended).rolling(k, center=True, min_periods=k).mean().to_numpy()[k:k + n]
    return pd.DataFrame(rolled, index=inflow_pu.index, columns=inflow_pu.columns)


def large_hydro_trajectory(
    inflow_pu: pd.DataFrame,
    weights: pd.Series,
    mean_level_hours: float,
    smoothing_days: float,
    reservoir_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annual reservoir path in hours of p_nom: ``(level before each snapshot, level after it)``.

    level_after(t) = mean_level_hours + D(t) - mean(D), with D(t) the cumulative
    sum up to t of w x (inflow - release) and the release the inflow's cyclic
    centred moving average over ``smoothing_days``. Each snapshot's level before
    is the previous snapshot's level after. Clipped to [0, reservoir_hours].
    """
    weights = pd.Series(weights, index=inflow_pu.index).astype(float).fillna(1.0)
    step_hours = float(weights.median()) if len(weights) else 1.0
    smoothing = int(round(float(smoothing_days) * 24.0 / step_hours)) if step_hours > 0 else 0
    release = large_hydro_release_path(inflow_pu, smoothing)
    change = (inflow_pu.astype(float) - release).mul(weights, axis=0)
    cumulative = change.cumsum()
    level_after = cumulative - cumulative.mean() + float(mean_level_hours)
    level_before = level_after - change
    return (level_before.clip(0.0, float(reservoir_hours)), level_after.clip(0.0, float(reservoir_hours)))


def _set_generator_series(network, attr: str, frame: pd.DataFrame) -> None:
    """Write the columns of ``frame`` into ``generators_t[attr]``, replacing those units."""
    from scripts.solve.hydro_constraints import _generator_series

    existing = _generator_series(network, attr)
    if existing is not None:
        frame = pd.concat([existing.drop(columns=list(frame.columns), errors="ignore"), frame], axis=1)
    network.generators_t[attr] = frame.reindex(index=network.snapshots)


def separate_large_hydro_ceiling(network, hydro_config: dict, logger=None) -> dict:
    """``power_ceiling: availability``: inflow into its own series, p_max_pu to availability.

    The inflow written is exactly what the reservoir model read before
    (``large_hydro_inflow_pu``: p_max_pu x inflow_profile_factor, or an existing
    series), so the water budget is unchanged and only the power ceiling rises.
    Idempotent. Returns a summary (inflow energy before and after, MWh).
    """
    from scripts.solve.hydro_constraints import (
        LARGE_HYDRO_INFLOW_SERIES, get_large_hydro_names, large_hydro_inflow_pu, resolve_large_hydro_ceiling,
    )

    mode, availability = resolve_large_hydro_ceiling(hydro_config)
    names = get_large_hydro_names(network)
    summary = {"power_ceiling": mode, "units": 0}
    if mode != "availability" or not names:
        return summary
    snapshots = network.snapshots
    weights = network.snapshot_weightings.generators.reindex(snapshots).fillna(1.0)
    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    inflow = large_hydro_inflow_pu(network, snapshots, hydro_config, names)
    before_mwh = float(inflow.mul(p_nom, axis=1).mul(weights, axis=0).to_numpy().sum())
    _set_generator_series(network, LARGE_HYDRO_INFLOW_SERIES, inflow)
    p_max_t = network.generators_t.p_max_pu
    varying = [n for n in names if n in p_max_t.columns]
    if varying:
        network.generators_t["p_max_pu"] = p_max_t.drop(columns=varying)
    network.generators.loc[names, "p_max_pu"] = availability
    after = large_hydro_inflow_pu(network, snapshots, hydro_config, names)
    after_mwh = float(after.mul(p_nom, axis=1).mul(weights, axis=0).to_numpy().sum())
    mean_cf = before_mwh / float((p_nom.sum() * weights.sum()) or 1.0)
    summary.update({"units": len(names), "p_nom_mw": float(p_nom.sum()), "availability_pu": availability,
                    "inflow_mwh_before": before_mwh, "inflow_mwh_after": after_mwh,
                    "inflow_capacity_factor": mean_cf})
    if abs(after_mwh - before_mwh) > 1e-6 * max(abs(before_mwh), 1.0):
        raise RuntimeError(f"Large-hydro inflow changed when the ceiling was separated: "
                           f"{before_mwh:.3f} -> {after_mwh:.3f} MWh")
    if logger is not None:
        logger.info(
            "Large hydro: power ceiling apart from inflow on %d units (%.0f MW): p_max_pu %.3f "
            "(was the inflow profile / inflow_profile_factor); inflow %.1f GWh kept in generators_t.%s "
            "(CF %.3f, ceiling %.1f x mean inflow)",
            len(names), float(p_nom.sum()), availability, after_mwh / 1e3, LARGE_HYDRO_INFLOW_SERIES,
            mean_cf, availability / mean_cf if mean_cf > 0 else float("nan"),
        )
    return summary


def attach_large_hydro_trajectory(network, hydro_config: dict, logger=None) -> dict:
    """``month_end: trajectory``: write the annual path (hours of p_nom) into the network.

    Series ``LARGE_HYDRO_TRAJECTORY_START`` (level before each snapshot) and
    ``LARGE_HYDRO_TRAJECTORY_END`` (after it), from the full built year of
    inflow, so a month solved on the shared network reads its own start and end.
    """
    from scripts.solve.hydro_constraints import (
        LARGE_HYDRO_TRAJECTORY_END, LARGE_HYDRO_TRAJECTORY_START, get_large_hydro_names,
        large_hydro_inflow_pu, resolve_large_hydro_trajectory,
    )

    traj = resolve_large_hydro_trajectory(hydro_config)
    names = get_large_hydro_names(network)
    large = (hydro_config or {}).get("large_hydro", {}) or {}
    reservoir = bool((hydro_config or {}).get("enabled", False) and large.get("reservoir_model", False))
    if traj is None or not names or not reservoir:
        return {"trajectory": False}
    snapshots = network.snapshots
    weights = network.snapshot_weightings.generators.reindex(snapshots).fillna(1.0)
    inflow = large_hydro_inflow_pu(network, snapshots, hydro_config, names)
    before, after = large_hydro_trajectory(inflow, weights, traj["mean_level_hours"], traj["smoothing_days"],
                                           traj["reservoir_capacity_hours"])
    _set_generator_series(network, LARGE_HYDRO_TRAJECTORY_START, before)
    _set_generator_series(network, LARGE_HYDRO_TRAJECTORY_END, after)
    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    total = float(p_nom.sum() or 1.0)
    fleet_after = after.mul(p_nom, axis=1).sum(axis=1) / total
    fleet_before = before.mul(p_nom, axis=1).sum(axis=1) / total
    month_start = fleet_before.groupby(fleet_before.index.month).first()
    summary = {"trajectory": True, "mean_level_hours": traj["mean_level_hours"],
               "smoothing_days": traj["smoothing_days"],
               "min_hours": float(fleet_after.min()), "max_hours": float(fleet_after.max()),
               "month_start_hours": {int(m): round(float(v), 2) for m, v in month_start.items()}}
    if logger is not None:
        logger.info(
            "Large hydro: annual reservoir path (mean %.0f h of p_nom, release = %g-day centred mean of "
            "inflow): %.1f-%.1f h; month starts %s",
            traj["mean_level_hours"], traj["smoothing_days"], summary["min_hours"], summary["max_hours"],
            ", ".join(f"{m}:{v:.1f}" for m, v in summary["month_start_hours"].items()),
        )
    return summary


def prepare_large_hydro_reservoir(network, hydro_config: dict, logger=None) -> dict:
    """Both build steps above (each a no-op under the defaults); returns their summaries."""
    return {"ceiling": separate_large_hydro_ceiling(network, hydro_config, logger),
            "trajectory": attach_large_hydro_trajectory(network, hydro_config, logger)}
