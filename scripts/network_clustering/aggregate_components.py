"""
Aggregate identical network components to reduce problem size without
changing optimization results.

This merges generators/storage with identical attributes and time series,
summing their capacities. Intended to run after clustering as an optional
clean-up step.
"""

import logging
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from pandas.util import hash_pandas_object
import pypsa

from scripts.utilities.network_io import load_network, save_network
from scripts.utilities.logging_config import setup_logging, log_execution_summary

logger = setup_logging("aggregate_components", log_level="INFO")

DEFAULT_IGNORE_COLUMNS = {
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


def _aggregate_loads_by_bus(network: pypsa.Network) -> int:
    """Aggregate loads on the same bus by summing their time series."""
    if network.loads.empty:
        return 0

    original_index_name = network.loads.index.name
    removed = 0
    new_rows = []
    new_p_set = {}
    new_q_set = {}

    p_set = getattr(network.loads_t, "p_set", None)
    q_set = getattr(network.loads_t, "q_set", None)

    for bus, group in network.loads.groupby("bus"):
        names = list(group.index)
        base = group.iloc[0].copy()
        new_name = names[0] if len(names) == 1 else f"{bus}_load"
        base.name = new_name
        new_rows.append(base)

        if p_set is not None and not p_set.empty:
            cols = [c for c in names if c in p_set.columns]
            if cols:
                new_p_set[new_name] = p_set[cols].sum(axis=1)
        if q_set is not None and not q_set.empty:
            cols = [c for c in names if c in q_set.columns]
            if cols:
                new_q_set[new_name] = q_set[cols].sum(axis=1)

        removed += len(names) - 1

    network.loads = pd.DataFrame(new_rows)
    network.loads.index.name = original_index_name or "load"
    if new_p_set:
        network.loads_t.p_set = pd.DataFrame(new_p_set, index=p_set.index)
    if new_q_set:
        network.loads_t.q_set = pd.DataFrame(new_q_set, index=q_set.index)

    return removed


def _hash_series(series: pd.Series) -> int:
    """Stable hash for time series columns."""
    if series is None or series.empty:
        return 0
    return int(hash_pandas_object(series, index=True).sum())


def _normalize_value(value, tol: float) -> float:
    """Round floats to reduce false negatives when comparing attributes."""
    if pd.isna(value):
        return None
    if isinstance(value, float):
        if abs(value) < tol:
            return 0.0
        return round(value, 9)
    return value


def _build_signature(
    df: pd.DataFrame,
    name: str,
    static_cols: List[str],
    time_fields: Iterable[str],
    time_data: Dict[str, pd.DataFrame],
    tol: float,
    skip_committable: bool = True,
) -> Tuple:
    """Generate a comparison signature for one component."""
    committable = bool(df.at[name, "committable"]) if "committable" in df.columns else False
    if skip_committable and committable:
        # Keep committable units distinct to avoid altering UC formulations
        return ("committable", name)

    static_part = tuple(
        (col, _normalize_value(df.at[name, col], tol))
        for col in static_cols
    )

    time_part = []
    for field in time_fields:
        tdf = time_data.get(field)
        if tdf is not None and name in tdf.columns:
            time_part.append((field, _hash_series(tdf[name])))
        else:
            time_part.append((field, None))

    return (static_part, tuple(time_part))


def aggregate_identical_components(
    network: pypsa.Network,
    component: str,
    capacity_fields: List[str],
    time_fields: Iterable[str],
    include_committable: bool,
    tol: float,
    ignore_columns: Iterable[str] = DEFAULT_IGNORE_COLUMNS,
    signature_columns: Iterable[str] = None,
) -> int:
    """
    Aggregate identical components (e.g., generators) by summing capacities.

    Args:
        network: PyPSA network to modify in-place.
        component: Component name (e.g., "generators", "storage_units").
        capacity_fields: Fields to sum when merging (e.g., p_nom).
        time_fields: Time-dependent fields to compare (e.g., p_max_pu).
        include_committable: Whether to allow committable units to merge.
        tol: Float comparison tolerance.

    Returns:
        Number of components removed through aggregation.
    """
    df = getattr(network, component)
    if df.empty:
        return 0

    time_container = getattr(network, f"{component}_t", None)
    time_data: Dict[str, pd.DataFrame] = {}
    for field in time_fields:
        if time_container is not None and hasattr(time_container, field):
            time_data[field] = getattr(time_container, field)
        else:
            time_data[field] = None

    ignore = set(ignore_columns or [])
    if signature_columns:
        static_cols = [c for c in signature_columns if c in df.columns]
    else:
        static_cols = [
            c for c in df.columns
            if c not in capacity_fields + ["name"] and c not in ignore
        ]
    groups: Dict[Tuple, List[str]] = {}
    for name in df.index:
        sig = _build_signature(
            df,
            name,
            static_cols,
            time_fields,
            time_data,
            tol,
            skip_committable=not include_committable,
        )
        groups.setdefault(sig, []).append(name)

    if all(len(members) == 1 for members in groups.values()):
        logger.info(f"No identical {component} found to aggregate.")
        return 0

    new_rows = []
    aggregation_map: List[Tuple[List[str], str]] = []
    for members in groups.values():
        primary = members[0]
        base = df.loc[primary].copy()
        for field in capacity_fields:
            if field in df.columns:
                base[field] = df.loc[members, field].sum()
        new_name = primary if len(members) == 1 else f"{primary}__agg{len(members)}"
        base.name = new_name
        new_rows.append(base)
        aggregation_map.append((members, new_name))

    new_df = pd.DataFrame(new_rows)
    new_df.index.name = df.index.name
    setattr(network, component, new_df)

    # Rebuild time series tables with aggregated names
    if time_container is not None:
        for field in time_fields:
            tdf = time_data.get(field)
            if tdf is None or tdf.empty:
                continue
            new_ts = {}
            for members, new_name in aggregation_map:
                source = members[0]
                if source in tdf.columns:
                    new_ts[new_name] = tdf[source]
            if new_ts:
                setattr(time_container, field, pd.DataFrame(new_ts, index=tdf.index))

    removed = sum(len(members) - 1 for members in groups.values())
    logger.info(
        f"Aggregated {removed} {component} into {len(new_rows)} groups "
        f"(from {len(df)})."
    )
    return removed


def main():
    global logger
    if "snakemake" in globals() and hasattr(snakemake, "log") and snakemake.log:
        logger = setup_logging(snakemake.log[0])

    start_time = time()
    start_network_path = Path(snakemake.input.network)
    output_path = Path(snakemake.output.aggregated_network)
    agg_config = getattr(snakemake.params, "aggregation_config", {}) or {}

    if not agg_config.get("enabled", False):
        logger.info("Component aggregation disabled; copying clustered network.")
        network = load_network(start_network_path, custom_logger=logger)
        save_network(network, output_path, custom_logger=logger)
        return

    include_generators = agg_config.get("include_generators", True)
    include_storage_units = agg_config.get("include_storage_units", True)
    include_stores = agg_config.get("include_stores", False)
    include_loads = agg_config.get("include_loads", True)
    include_committable = agg_config.get("include_committable", False)
    tol = agg_config.get("tolerance", 1e-9)

    logger.info(f"Loading clustered network from {start_network_path}")
    network = load_network(start_network_path, custom_logger=logger)

    removed_total = 0
    if include_generators:
        removed_total += aggregate_identical_components(
            network,
            "generators",
            capacity_fields=["p_nom", "p_nom_min", "p_nom_max"],
            time_fields=["p_max_pu", "p_min_pu"],
            include_committable=include_committable,
            tol=tol,
        )

    if include_storage_units:
        removed_total += aggregate_identical_components(
            network,
            "storage_units",
            capacity_fields=["p_nom", "p_nom_min", "p_nom_max"],
            time_fields=["p_max_pu", "p_min_pu"],
            include_committable=include_committable,
            tol=tol,
        )

    if include_stores:
        removed_total += aggregate_identical_components(
            network,
            "stores",
            capacity_fields=["e_nom", "e_nom_min", "e_nom_max"],
            time_fields=["e_max_pu", "e_min_pu"],
            include_committable=include_committable,
            tol=tol,
        )

    if include_loads:
        removed_total += _aggregate_loads_by_bus(network)

    logger.info(f"Total components removed via aggregation: {removed_total}")
    save_network(network, output_path, custom_logger=logger)

    log_execution_summary(
        logger,
        "Aggregate Identical Components",
        start_time,
        inputs={"network": str(start_network_path)},
        outputs={"aggregated_network": str(output_path)},
        context={
            "removed_components": removed_total,
            "include_generators": include_generators,
            "include_storage_units": include_storage_units,
            "include_stores": include_stores,
            "include_loads": include_loads,
        },
    )


if __name__ == "__main__":
    main()
