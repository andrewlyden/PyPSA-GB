"""Stage 1 at a coarser temporal resolution (``market.wholesale.temporal_resolution_hours``, 17 Oct 2026).

NESO's CP30 market outputs behind CP.28, CP.29, CP.30 and CP.32 (and the start of its
redispatch, CP.18 / N2 / CP.08) come from "a simplified Plexos run using three-hour rather
than one-hour intervals" (CP30 Annex 4). the calibration to NESO's CP.28-30 market run reproduces that method:
Stage 1 solves on blocks of R hours, the BM runs hourly on the held block positions.

Method (PyPSA-native: snapshot weightings carry the block's duration, as the coarse
look-ahead of ``solve_wholesale.coarsen_look_ahead`` already does for its tail):

* ``aggregate_to_blocks`` puts the network on one snapshot per block of R hours, counted
  from the period's first snapshot (00:00, 03:00, ... for R = 3). Every time-varying input
  (loads, p_max_pu / p_min_pu, marginal-cost series, inflows, link limits, IC p_set ...)
  is the block's mean weighted by the objective weighting, and each block's snapshot
  weightings are the sums of its hours. PyPSA then counts generator costs, the energy
  balance and the storage / store energy update (soc += w x (eta_s p_store - p_dispatch /
  eta_d), standing loss to the power w) over the block's true duration, and divides the
  bus duals by the objective weighting, so block prices are in GBP/MWh. Ramp limits act
  per snapshot in PyPSA (not weighted), so they are scaled by R (capped at 1) for the
  solve: a unit may move R hours of ramp in a block. Every custom constraint in the Stage 1
  path already reads the snapshot weightings (hydro water budget, month-end storage and
  store conditions, minimum-output energy blocks, scarcity curve, IC piecewise cost,
  internal DC tie-break, must-take).
* ``restore_hourly`` puts the network back on its hourly snapshots, inputs and weightings
  (and the original ramp limits) and expands every solved output: power-like series
  (generator p, storage p / p_store / p_dispatch, link p0..p3, line flows, prices) hold the
  block's value in each of its hours (power constant within the block), and levels
  (storage state_of_charge, store e) are interpolated linearly inside the block from the
  previous block's end (the unit's starting level for the first block), which is exactly
  the level path constant power gives (standing loss aside).

Storage in the BM: the balancing stage rebuilds Stage 1's state of charge hour by hour
from the saved hourly legs and the starting state (``solve_balancing.stage1_state_of_charge``).
Constant power over the block's hours gives the same energy as the block's w x p, so the
rebuilt level equals Stage 1's own at every block end and is linear inside the block;
the only difference is standing loss applied hourly to intra-block energy (batteries
~0). The BM therefore anchors on a schedule Stage 1 could actually deliver hourly.

``1`` (the default, equal to config/defaults/ market.wholesale.temporal_resolution_hours)
solves hourly exactly as before: nothing here is called.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Code default; must equal config/defaults/ market.wholesale.temporal_resolution_hours.
DEFAULT_TEMPORAL_RESOLUTION_HOURS = 1
#: Network meta key recording the resolution Stage 1 was solved at.
META_KEY = "stage1_temporal_resolution"
#: Dynamic outputs that are levels (interpolated inside a block), with their static start attribute.
LEVEL_ATTRS = {("StorageUnit", "state_of_charge"): "state_of_charge_initial", ("Store", "e"): "e_initial"}
#: Components with per-snapshot ramp limits (PyPSA does not weight them by the snapshot duration).
RAMP_COMPONENTS = ("Generator", "Link")
RAMP_ATTRS = ("ramp_limit_up", "ramp_limit_down")


def resolve_temporal_resolution_hours(wholesale_config) -> int:
    """``market.wholesale.temporal_resolution_hours`` (default 1 = hourly), validated as a whole number >= 1."""
    value = (wholesale_config or {}).get("temporal_resolution_hours", DEFAULT_TEMPORAL_RESOLUTION_HOURS)
    value = DEFAULT_TEMPORAL_RESOLUTION_HOURS if value is None else value
    as_float = float(value)
    if as_float < 1.0 or abs(as_float - round(as_float)) > 1e-9:
        raise ValueError(f"market.wholesale.temporal_resolution_hours must be a whole number >= 1, got {value!r}")
    return int(round(as_float))


def check_compatible(resolution_hours: int, wholesale_config, commitment: bool = False) -> None:
    """Refuse settings the block solve cannot honour (rolling windows that split a block; unit commitment)."""
    if resolution_hours <= 1:
        return
    cfg = wholesale_config or {}
    if str(cfg.get("mode", "single")) == "rolling_day_ahead":
        window = float(cfg.get("window_hours", 24))
        if abs(window / resolution_hours - round(window / resolution_hours)) > 1e-9:
            raise ValueError(f"market.wholesale.window_hours {window:g} is not a whole number of "
                             f"temporal_resolution_hours {resolution_hours} blocks")
        look = float(cfg.get("look_ahead_hours", 0) or 0)
        if abs(look / resolution_hours - round(look / resolution_hours)) > 1e-9:
            raise ValueError(f"market.wholesale.look_ahead_hours {look:g} is not a whole number of "
                             f"temporal_resolution_hours {resolution_hours} blocks")
        coarse = float(cfg.get("look_ahead_resolution_hours", 1) or 1)
        if coarse > resolution_hours and abs(coarse / resolution_hours - round(coarse / resolution_hours)) > 1e-9:
            raise ValueError(f"market.wholesale.look_ahead_resolution_hours {coarse:g} is not a multiple of "
                             f"temporal_resolution_hours {resolution_hours}")
    if commitment:
        raise ValueError("market.wholesale.temporal_resolution_hours > 1 is not supported with Stage 1 unit "
                         "commitment (minimum up/down times are counted in snapshots)")


def _step_hours(index) -> float:
    index = pd.DatetimeIndex(index)
    if len(index) < 2:
        return 1.0
    return float(pd.Series(index).diff().dt.total_seconds().div(3600.0).median())


def block_starts(snapshots, steps: int) -> pd.Series:
    """Each snapshot mapped to the first snapshot of its block of ``steps`` snapshots (from the first)."""
    snapshots = pd.Index(snapshots)
    steps = max(1, int(steps))
    return pd.Series([snapshots[(i // steps) * steps] for i in range(len(snapshots))], index=snapshots)


@dataclass
class TemporalBlocks:
    """What ``restore_hourly`` needs to put a block-resolution network back on its hourly snapshots."""

    resolution_hours: int
    steps: int
    hourly_index: pd.Index
    group: pd.Series                      # hourly snapshot -> block start
    weightings: pd.DataFrame              # hourly snapshot weightings
    inputs: dict = field(default_factory=dict)        # (component, attr) -> hourly input frame
    static_ramps: dict = field(default_factory=dict)  # (component, attr) -> static Series

    @property
    def block_index(self) -> pd.Index:
        return pd.Index(pd.unique(self.group.to_numpy()), name=self.hourly_index.name)


def _output_attrs(component) -> set:
    defaults = getattr(component, "defaults", None)
    if defaults is None or "status" not in getattr(defaults, "columns", []):
        return set()
    status = defaults["status"].astype(str)
    return set(defaults.index[status.str.startswith("Output")])


def _scale_ramps(network, steps: int, blocks: TemporalBlocks) -> int:
    """Ramp limits x steps, capped at 1 (static and time-varying), for the block solve. Returns units touched."""
    touched = 0
    for name in RAMP_COMPONENTS:
        component = network.components[name]
        static = component.static
        for attr in RAMP_ATTRS:
            if attr in static.columns:
                values = pd.to_numeric(static[attr], errors="coerce")
                blocks.static_ramps[(name, attr)] = static[attr].copy()
                limited = values.notna() & (values < 1.0)
                touched += int(limited.sum())
                static[attr] = values.mul(steps).clip(upper=1.0).where(values.notna(), values)
            frame = component.dynamic.get(attr) if hasattr(component.dynamic, "get") else None
            if frame is not None and not frame.empty and frame.shape[1]:
                component.dynamic[attr] = frame.mul(steps).clip(upper=1.0)
    return touched


def aggregate_to_blocks(network, resolution_hours: int, logger: logging.Logger | None = None) -> TemporalBlocks | None:
    """Put ``network`` on one snapshot per block of ``resolution_hours`` (in place); None when nothing changes.

    Inputs become block means weighted by the objective weighting (non-numeric frames
    take the block's first row), the blocks' weightings the sums of their hours, ramp
    limits x the block's step count (capped at 1). The returned record restores it.
    """
    logger = logger or logging.getLogger(__name__)
    hourly = pd.Index(network.snapshots)
    step_h = _step_hours(hourly)
    steps = int(round(resolution_hours / step_h))
    if abs(resolution_hours / step_h - steps) > 1e-9:
        raise ValueError(f"temporal_resolution_hours {resolution_hours} is not a whole number of the network's "
                         f"{step_h:g} h snapshots")
    if steps <= 1 or len(hourly) <= 1:
        return None
    group = block_starts(hourly, steps)
    starts = pd.Index(pd.unique(group.to_numpy()), name=hourly.name)
    weights = network.snapshot_weightings.reindex(hourly).fillna(1.0)
    blocks = TemporalBlocks(int(resolution_hours), steps, hourly, group, weights.copy())
    w_obj = weights["objective"]
    w_sum = w_obj.groupby(group).sum()
    frames = []
    for component in network.components:
        outputs = _output_attrs(component)
        for attr, frame in component.dynamic.items():
            if frame is None or frame.empty or not len(frame.columns) or attr in outputs:
                continue
            body = frame.reindex(hourly)
            blocks.inputs[(component.name, attr)] = frame.copy()
            if body.select_dtypes(include="number").shape[1] == body.shape[1]:
                agg = body.mul(w_obj, axis=0).groupby(group).sum().div(w_sum, axis=0)
            else:
                agg = body.groupby(group).first()
            frames.append((component, attr, agg.reindex(starts)))
    block_weights = weights.groupby(group).sum().reindex(starts)
    network.set_snapshots(starts)
    network.snapshot_weightings.loc[starts, :] = block_weights.to_numpy()
    for component, attr, frame in frames:
        frame.index = starts
        component.dynamic[attr] = frame
    touched = _scale_ramps(network, steps, blocks)
    logger.info(
        f"Stage 1 temporal resolution {resolution_hours} h (market.wholesale.temporal_resolution_hours): "
        f"{len(hourly)} snapshots -> {len(starts)} blocks (block-mean inputs, snapshot weightings = block "
        f"hours); ramp limits x{steps} (capped at 1) on {touched} limited unit-attributes"
    )
    return blocks


def expand_rows(frame, blocks: TemporalBlocks):
    """A block-indexed frame (or Series) on the hourly snapshots: each hour holds its block's row."""
    if frame is None or blocks is None or not len(frame):
        return frame
    return frame.reindex(blocks.group.to_numpy()).set_axis(blocks.hourly_index, axis=0)


def expand_levels(frame: pd.DataFrame, blocks: TemporalBlocks, initial: pd.Series | None = None) -> pd.DataFrame:
    """A level series (value at each block's end) on the hourly snapshots, linear inside each block.

    Hour k (0-based) of a block of L hours takes ``prev + (k + 1) / L x (end - prev)``, with ``prev``
    the previous block's end level (``initial`` for the first block, else the first block's own value).
    """
    if frame is None or blocks is None or not len(frame):
        return frame
    ends = frame.reindex(blocks.block_index)
    if initial is None:
        first = ends.iloc[0]
    else:
        first = pd.to_numeric(initial.reindex(ends.columns), errors="coerce").fillna(ends.iloc[0])
    prev = ends.shift(1)
    prev.iloc[0] = first.to_numpy()
    group = blocks.group
    position = group.groupby(group).cumcount().to_numpy()
    length = group.map(group.value_counts()).to_numpy()
    frac = (position + 1) / length
    end_rows = ends.reindex(group.to_numpy()).to_numpy()
    prev_rows = prev.reindex(group.to_numpy()).to_numpy()
    values = prev_rows + frac[:, None] * (end_rows - prev_rows)
    return pd.DataFrame(values, index=blocks.hourly_index, columns=frame.columns)


def aggregate_frame(frame: pd.DataFrame, blocks: TemporalBlocks | None) -> pd.DataFrame:
    """An hourly frame (e.g. the minimum-output energy floor) as block means, attrs kept."""
    if blocks is None or frame is None:
        return frame
    attrs = dict(getattr(frame, "attrs", {}) or {})
    if not frame.shape[1]:
        out = pd.DataFrame(index=blocks.block_index)
    else:
        w = blocks.weightings["objective"].reindex(frame.index).fillna(1.0)
        grp = blocks.group.reindex(frame.index)
        out = frame.mul(w, axis=0).groupby(grp).sum().div(w.groupby(grp).sum(), axis=0).reindex(blocks.block_index)
    out.attrs.update(attrs)
    return out


def restore_hourly(network, blocks: TemporalBlocks | None, logger: logging.Logger | None = None) -> list:
    """Put ``network`` back on its hourly snapshots, inputs, weightings and ramp limits; expand the outputs.

    Returns the expanded outputs as ``"Component.attr"`` strings.
    """
    if blocks is None:
        return []
    logger = logger or logging.getLogger(__name__)
    outputs = []
    for component in network.components:
        out_attrs = _output_attrs(component)
        for attr, frame in component.dynamic.items():
            if attr in out_attrs and frame is not None and not frame.empty and frame.shape[1]:
                outputs.append((component, attr, frame.copy()))
    network.set_snapshots(blocks.hourly_index)
    network.snapshot_weightings.loc[:, :] = blocks.weightings.reindex(blocks.hourly_index).to_numpy()
    for (name, attr), frame in blocks.inputs.items():
        network.components[name].dynamic[attr] = frame
    for (name, attr), values in blocks.static_ramps.items():
        network.components[name].static[attr] = values
    written = []
    for component, attr, frame in outputs:
        key = (component.name, attr)
        if key in LEVEL_ATTRS:
            start_attr = LEVEL_ATTRS[key]
            initial = component.static[start_attr] if start_attr in component.static.columns else None
            expanded = expand_levels(frame, blocks, initial)
        else:
            expanded = expand_rows(frame, blocks)
        component.dynamic[attr] = expanded
        written.append(f"{component.name}.{attr}")
    logger.info(
        f"Stage 1 solved at {blocks.resolution_hours} h: outputs expanded to {len(blocks.hourly_index)} hourly "
        f"snapshots (power held within each block, levels linear): {', '.join(written) or 'none'}"
    )
    return written


# ── The balancing stage at the same resolution (market.balancing.temporal_resolution_hours) ──────────────
#
# A 3-hour Stage 1 holds every position constant over its block, while the hourly BM sees each hour's own wind and
# demand. The BM then has to turn down wind held above the hour's availability and follow demand inside the block:
# redispatch that is an artefact of the resolution mismatch, not of the network (17 Oct 2026, cal_3hnm Jan: 0.58 TWh
# of wind held above its hourly availability, 0.53 TWh of demand moving inside blocks). NESO's redispatch pairs with
# its 3-hour run (N2 - CP.18 = CP.30's net exchange), so the calibration runs the BM at the same resolution: every
# time-varying input is held at its block mean over the block's hours (snapshots and weightings unchanged, so every
# hourly Stage 1 table, price frame and anchor lines up as before) and ramp limits are scaled as in Stage 1. The BM
# then solves the 3-hour problem on hourly snapshots; any within-block variation it chooses is free of cost.


def resolve_balancing_temporal_resolution_hours(balancing_config) -> int:
    """``market.balancing.temporal_resolution_hours`` (default 1 = hourly inputs), validated as for Stage 1."""
    value = (balancing_config or {}).get("temporal_resolution_hours", DEFAULT_TEMPORAL_RESOLUTION_HOURS)
    return resolve_temporal_resolution_hours({"temporal_resolution_hours": value})


def flatten_to_blocks(network, resolution_hours: int, logger: logging.Logger | None = None) -> int:
    """Hold every time-varying input of ``network`` at its block mean over the block's hours (in place).

    Snapshots and weightings are unchanged. Blocks of ``resolution_hours`` from the first snapshot; the mean is
    weighted by the objective weighting; ramp limits x the block's step count (capped at 1), as in Stage 1. Outputs
    (already-solved series) are left alone. Returns the number of frames flattened (0 at hourly resolution).
    """
    logger = logger or logging.getLogger(__name__)
    hourly = pd.Index(network.snapshots)
    step_h = _step_hours(hourly)
    steps = int(round(resolution_hours / step_h))
    if abs(resolution_hours / step_h - steps) > 1e-9:
        raise ValueError(f"temporal_resolution_hours {resolution_hours} is not a whole number of the network's "
                         f"{step_h:g} h snapshots")
    if steps <= 1 or len(hourly) <= 1:
        return 0
    group = block_starts(hourly, steps)
    w = network.snapshot_weightings.reindex(hourly).fillna(1.0)["objective"]
    w_sum = w.groupby(group).transform("sum")
    n = 0
    for component in network.components:
        outputs = _output_attrs(component)
        for attr, frame in list(component.dynamic.items()):
            if frame is None or frame.empty or not len(frame.columns) or attr in outputs:
                continue
            body = frame.reindex(hourly)
            if body.select_dtypes(include="number").shape[1] != body.shape[1]:
                continue
            flat = body.mul(w, axis=0).groupby(group).transform("sum").div(w_sum, axis=0)
            flat.index = frame.index if len(frame.index) == len(hourly) else hourly
            component.dynamic[attr] = flat
            n += 1
    blocks = TemporalBlocks(int(resolution_hours), steps, hourly, group, network.snapshot_weightings.copy())
    touched = _scale_ramps(network, steps, blocks)
    logger.info(
        f"Balancing inputs at {resolution_hours} h resolution (market.balancing.temporal_resolution_hours): {n} "
        f"time-varying frames held at their block means over {len(pd.unique(group.to_numpy()))} blocks of "
        f"{steps} snapshots; ramp limits x{steps} (capped at 1) on {touched} limited unit-attributes"
    )
    return n


def check_balancing_compatible(resolution_hours: int, balancing_config) -> None:
    """Refuse a rolling BM whose windows split a block."""
    if resolution_hours <= 1:
        return
    cfg = balancing_config or {}
    if str(cfg.get("mode", "full_period")) == "rolling":
        window = float(cfg.get("window_hours", 1))
        if abs(window / resolution_hours - round(window / resolution_hours)) > 1e-9:
            raise ValueError(f"market.balancing.window_hours {window:g} is not a whole number of "
                             f"temporal_resolution_hours {resolution_hours} blocks")


def record_meta(network, blocks: TemporalBlocks | None) -> None:
    """Record the Stage 1 resolution on the saved network's meta (nothing at hourly resolution)."""
    if blocks is None:
        return
    network.meta = {**dict(getattr(network, "meta", None) or {}),
                    META_KEY: {"resolution_hours": int(blocks.resolution_hours),
                               "blocks": int(len(blocks.block_index)),
                               "hourly_snapshots": int(len(blocks.hourly_index))}}
