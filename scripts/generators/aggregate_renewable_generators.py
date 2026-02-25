"""
Aggregate renewable generators by (bus, carrier) to reduce problem size.

Unlike the identical-component aggregation in network_clustering/aggregate_components.py
(which requires bit-identical time series), this module performs **capacity-weighted
averaging** of p_max_pu profiles. This correctly combines renewable generators that
have unique, location-specific output profiles into a single representative generator
per (bus, carrier) group.

The aggregation preserves:
  - Total installed capacity (p_nom summed exactly)
  - Expected energy output (capacity-weighted average profile is energy-conserving)
  - Spatial resolution (one generator per bus per carrier)
  - Non-renewable generators (completely untouched)

Usage:
    from scripts.generators.aggregate_renewable_generators import aggregate_renewables_by_bus

    network, removed = aggregate_renewables_by_bus(network, carriers, logger)

Author: PyPSA-GB Team
Date: 2026-02
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pypsa


# Default set of renewable carriers eligible for aggregation
DEFAULT_RENEWABLE_CARRIERS = [
    "wind_onshore",
    "wind_offshore",
    "solar_pv",
    "large_hydro",
    "small_hydro",
    "tidal_stream",
    "shoreline_wave",
    "tidal_lagoon",
]

# Capacity fields to sum when merging generators
_CAPACITY_FIELDS = ["p_nom", "p_nom_min", "p_nom_max"]

# Time-varying fields to compute capacity-weighted averages for
_TIME_FIELDS = ["p_max_pu", "p_min_pu"]

# Static columns to ignore (not meaningful for the aggregated generator)
_IGNORE_COLUMNS = {
    "name",
    "build_year",
    "lifetime",
    "lon",
    "lat",
    "longitude",
    "latitude",
    "x",
    "y",
    "x_osgb36",
    "y_osgb36",
    "data_source",
    "country",
    "source",
}


def aggregate_renewables_by_bus(
    network: pypsa.Network,
    carriers: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pypsa.Network, int]:
    """
    Aggregate renewable generators per (bus, carrier) group.

    For each group of renewable generators sharing the same bus and carrier:
      - p_nom is summed
      - p_max_pu is capacity-weighted averaged:
            p_max_pu_agg(t) = sum(p_nom_i * p_max_pu_i(t)) / sum(p_nom_i)
      - Static attributes are taken from the largest-capacity member
      - lat/lon are capacity-weighted centroids (for plotting)

    Non-renewable generators are passed through unchanged.

    Args:
        network: PyPSA network (modified in-place).
        carriers: List of carrier names to aggregate. Defaults to all
            standard renewable carriers.
        logger: Logger instance. If None, a module-level logger is used.

    Returns:
        Tuple of (network, removed_count) where removed_count is the number
        of individual generators replaced by aggregated ones.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if carriers is None:
        carriers = list(DEFAULT_RENEWABLE_CARRIERS)

    carriers_set = set(carriers)
    gen_df = network.generators

    if gen_df.empty:
        logger.info("No generators in network — nothing to aggregate.")
        return network, 0

    # Split generators into renewable (to aggregate) and other (to keep)
    is_renewable = gen_df["carrier"].isin(carriers_set)
    renewable_df = gen_df[is_renewable].copy()
    other_df = gen_df[~is_renewable].copy()

    if renewable_df.empty:
        logger.info("No renewable generators matching specified carriers — nothing to aggregate.")
        return network, 0

    n_before = len(renewable_df)
    logger.info(f"Aggregating {n_before} renewable generators across {renewable_df['carrier'].nunique()} carriers")

    # Get existing time series
    p_max_pu = network.generators_t.p_max_pu.copy() if hasattr(network.generators_t, "p_max_pu") and not network.generators_t.p_max_pu.empty else pd.DataFrame(index=network.snapshots)
    p_min_pu = network.generators_t.p_min_pu.copy() if hasattr(network.generators_t, "p_min_pu") and not network.generators_t.p_min_pu.empty else pd.DataFrame(index=network.snapshots)

    # Collect results
    aggregated_rows = []
    new_p_max_pu = {}
    new_p_min_pu = {}

    # Group by (bus, carrier)
    groups = renewable_df.groupby(["bus", "carrier"])
    n_groups = len(groups)

    for (bus, carrier), group in groups:
        members = list(group.index)
        n_members = len(members)

        if n_members == 1:
            # Single generator — keep as-is (no rename, no change)
            gen_name = members[0]
            aggregated_rows.append(group.iloc[0])
            # Preserve time series if they exist
            if gen_name in p_max_pu.columns:
                new_p_max_pu[gen_name] = p_max_pu[gen_name]
            if gen_name in p_min_pu.columns:
                new_p_min_pu[gen_name] = p_min_pu[gen_name]
            continue

        # Multiple generators — aggregate
        capacities = group["p_nom"].values.astype(float)
        total_capacity = capacities.sum()

        if total_capacity <= 0:
            # Degenerate case: all zero capacity — take first member only
            logger.warning(f"Zero total capacity for ({bus}, {carrier}) group of {n_members} — keeping first member only")
            gen_name = members[0]
            aggregated_rows.append(group.iloc[0])
            if gen_name in p_max_pu.columns:
                new_p_max_pu[gen_name] = p_max_pu[gen_name]
            if gen_name in p_min_pu.columns:
                new_p_min_pu[gen_name] = p_min_pu[gen_name]
            continue

        weights = capacities / total_capacity

        # Choose the largest-capacity member as the template for static attrs
        largest_idx = group["p_nom"].idxmax()
        base = group.loc[largest_idx].copy()

        # Create aggregated name
        agg_name = f"{bus}_{carrier}__agg{n_members}"
        base.name = agg_name

        # Sum capacity fields
        for field in _CAPACITY_FIELDS:
            if field in group.columns:
                base[field] = group[field].sum()

        # Capacity-weighted centroid for lat/lon (for plotting)
        for coord in ["lat", "lon", "latitude", "longitude", "y", "x"]:
            if coord in group.columns:
                vals = group[coord].values
                if np.all(pd.notna(vals)):
                    base[coord] = np.average(vals.astype(float), weights=weights)

        aggregated_rows.append(base)

        # Capacity-weighted average of p_max_pu
        _aggregate_time_series(
            members, capacities, total_capacity,
            p_max_pu, agg_name, new_p_max_pu,
            default_value=1.0, snapshots=network.snapshots,
        )

        # Capacity-weighted average of p_min_pu
        _aggregate_time_series(
            members, capacities, total_capacity,
            p_min_pu, agg_name, new_p_min_pu,
            default_value=0.0, snapshots=network.snapshots,
        )

    # Build new generators DataFrame
    new_gen_df = pd.concat([other_df, pd.DataFrame(aggregated_rows)], ignore_index=False)
    # Ensure index name matches original
    new_gen_df.index.name = gen_df.index.name or "Generator"

    # Build new time series DataFrames
    # Keep columns for non-renewable generators that had time series
    other_names = set(other_df.index)
    other_p_max_cols = {col: p_max_pu[col] for col in p_max_pu.columns if col in other_names}
    other_p_min_cols = {col: p_min_pu[col] for col in p_min_pu.columns if col in other_names}

    # Merge other + aggregated time series
    all_p_max = {**other_p_max_cols, **new_p_max_pu}
    all_p_min = {**other_p_min_cols, **new_p_min_pu}

    # Apply to network
    network.generators = new_gen_df
    if all_p_max:
        network.generators_t.p_max_pu = pd.DataFrame(all_p_max, index=network.snapshots)
    else:
        network.generators_t.p_max_pu = pd.DataFrame(index=network.snapshots)

    if all_p_min:
        network.generators_t.p_min_pu = pd.DataFrame(all_p_min, index=network.snapshots)
    else:
        network.generators_t.p_min_pu = pd.DataFrame(index=network.snapshots)

    n_after = len(aggregated_rows)
    removed = n_before - n_after

    # Log summary
    logger.info(f"Renewable aggregation complete:")
    logger.info(f"  Before: {n_before} renewable generators")
    logger.info(f"  After:  {n_after} renewable generators ({n_groups} bus-carrier groups)")
    logger.info(f"  Removed: {removed} generators")

    # Per-carrier breakdown
    for c in sorted(carriers_set & set(renewable_df["carrier"].unique())):
        before_c = len(renewable_df[renewable_df["carrier"] == c])
        after_c = sum(1 for row in aggregated_rows if row.get("carrier", getattr(row, "carrier", None)) == c)
        cap_before = renewable_df.loc[renewable_df["carrier"] == c, "p_nom"].sum()
        cap_after = sum(
            row.get("p_nom", getattr(row, "p_nom", 0))
            for row in aggregated_rows
            if row.get("carrier", getattr(row, "carrier", None)) == c
        )
        logger.info(
            f"  {c}: {before_c} → {after_c} generators, "
            f"{cap_before:.1f} → {cap_after:.1f} MW"
        )

    return network, removed


def _aggregate_time_series(
    members: List[str],
    capacities: np.ndarray,
    total_capacity: float,
    ts_df: pd.DataFrame,
    agg_name: str,
    output_dict: dict,
    default_value: float,
    snapshots: pd.DatetimeIndex,
) -> None:
    """
    Compute capacity-weighted average time series for a group of generators.

    For generators that have entries in ts_df, their series are used directly.
    For generators without entries, a constant ``default_value`` is assumed
    (PyPSA uses 1.0 for p_max_pu and 0.0 for p_min_pu when not explicitly set).

    Args:
        members: Generator names in the group.
        capacities: Corresponding p_nom values (same order as members).
        total_capacity: Sum of capacities (pre-computed).
        ts_df: Full time series DataFrame (e.g., network.generators_t.p_max_pu).
        agg_name: Name for the aggregated generator.
        output_dict: Dict to store the result series (mutated in-place).
        default_value: Default constant value for generators without time series.
        snapshots: Network snapshots index.
    """
    weighted_sum = pd.Series(0.0, index=snapshots, dtype=float)
    any_ts = False

    for member, cap in zip(members, capacities):
        if member in ts_df.columns:
            weighted_sum += ts_df[member].values * cap
            any_ts = True
        else:
            # Generator has no explicit time series — use PyPSA default
            weighted_sum += default_value * cap

    if any_ts or default_value != 1.0:
        # Only store the series if there's something non-trivial
        result = weighted_sum / total_capacity
        # Clip to valid range for safety
        result = result.clip(0.0, 1.0)
        output_dict[agg_name] = result
