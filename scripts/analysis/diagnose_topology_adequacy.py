"""Diagnose clustered FES topology and adequacy outcomes without changing capacity."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network


def snapshot_hours(network) -> pd.Series:
    snapshots = pd.Index(network.snapshots)
    weights = getattr(network, "snapshot_weightings", None)
    if weights is not None:
        for column in ("generators", "objective"):
            if column in weights.columns:
                result = pd.to_numeric(weights[column], errors="coerce")
                if result.reindex(snapshots).notna().all():
                    return result.reindex(snapshots).astype(float)
    result = pd.Series(1.0, index=snapshots)
    if len(snapshots) > 1:
        seconds = snapshots.to_series().diff().dt.total_seconds().median()
        if pd.notna(seconds) and seconds > 0:
            result[:] = seconds / 3600.0
    return result


def component_envelope(network, busmap: pd.Series | None = None) -> pd.DataFrame:
    """Aggregate AC line and transformer thermal ratings by endpoint pair."""
    rows = []
    for component, table_name in (("line", "lines"), ("transformer", "transformers")):
        table = getattr(network, table_name)
        for name, row in table.iterrows():
            bus0, bus1 = str(row.bus0), str(row.bus1)
            if busmap is not None:
                bus0 = str(busmap.get(bus0, bus0))
                bus1 = str(busmap.get(bus1, bus1))
            if bus0 == bus1:
                continue
            left, right = sorted((bus0, bus1))
            rows.append(
                {
                    "component_type": component,
                    "component": name,
                    "bus0": left,
                    "bus1": right,
                    "capacity_MW": float(row.s_nom),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["bus0", "bus1", "capacity_MW", "n_components", "component_types"]
        )
    data = pd.DataFrame(rows)
    return (
        data.groupby(["bus0", "bus1"], as_index=False)
        .agg(
            capacity_MW=("capacity_MW", "sum"),
            n_components=("component", "count"),
            component_types=("component_type", lambda values: ",".join(sorted(set(values)))),
        )
        .sort_values("capacity_MW", ascending=False)
    )


def corridor_loading(network, threshold: float) -> pd.DataFrame:
    """Report actual constrained-network AC corridor loading."""
    hours = snapshot_hours(network)
    rows = []
    for component, table_name, time_name in (
        ("line", "lines", "lines_t"),
        ("transformer", "transformers", "transformers_t"),
    ):
        table = getattr(network, table_name)
        dispatch = getattr(network, time_name).p0
        for name, row in table.iterrows():
            if name not in dispatch:
                continue
            capacity = float(row.s_nom)
            if capacity <= 0:
                continue
            flow = dispatch[name].reindex(network.snapshots).fillna(0.0).abs()
            loading = flow / capacity
            rows.append(
                {
                    "component_type": component,
                    "component": name,
                    "bus0": str(row.bus0),
                    "bus1": str(row.bus1),
                    "capacity_MW": capacity,
                    "mean_loading_fraction": float(loading.mean()),
                    "max_loading_fraction": float(loading.max()),
                    "hours_at_or_above_threshold": float(hours[loading >= threshold].sum()),
                    "energy_MWh_abs": float((flow * hours).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["hours_at_or_above_threshold", "max_loading_fraction"],
        ascending=False,
    ) if rows else pd.DataFrame()


def cluster_balance(network, focus_clusters: set[str]) -> pd.DataFrame:
    """Trace demand, potential supply and feasibility slack by clustered bus."""
    hours = snapshot_hours(network)
    demand = network.get_switchable_as_dense("Load", "p_set", network.snapshots)
    rows = []
    load_by_bus = network.loads.bus
    for bus, names in load_by_bus.groupby(load_by_bus).groups.items():
        values = demand[list(names)].sum(axis=1)
        rows.append(
            {
                "bus": str(bus),
                "metric": "demand",
                "energy_MWh": float((values * hours).sum()),
                "peak_MW": float(values.max()),
                "capacity_MW": np.nan,
            }
        )

    generators = network.generators
    for bus, group in generators.groupby("bus"):
        is_slack = group.carrier.astype(str).str.lower().eq("load_shedding")
        slack_names = group.index[is_slack]
        if len(slack_names):
            values = network.generators_t.p[slack_names].sum(axis=1).clip(lower=0.0)
            rows.append(
                {
                    "bus": str(bus),
                    "metric": "feasibility_slack",
                    "energy_MWh": float((values * hours).sum()),
                    "peak_MW": float(values.max()),
                    "capacity_MW": float(group.loc[is_slack, "p_nom"].sum()),
                }
            )
        supply = group.loc[~is_slack, "p_nom"].sum()
        rows.append(
            {
                "bus": str(bus),
                "metric": "generator_capacity",
                "energy_MWh": np.nan,
                "peak_MW": np.nan,
                "capacity_MW": float(supply),
            }
        )

    result = pd.DataFrame(rows)
    result["focus_cluster"] = result["bus"].isin(focus_clusters)
    return result.sort_values(["focus_cluster", "bus", "metric"], ascending=[False, True, True])


def compare_corridors(unclustered, clustered, busmap, focus_clusters) -> pd.DataFrame:
    raw = component_envelope(unclustered, busmap)
    grouped = component_envelope(clustered)
    merged = raw.merge(grouped, on=["bus0", "bus1"], how="outer", suffixes=("_unclustered", "_clustered"))
    for column in ("capacity_MW_unclustered", "capacity_MW_clustered"):
        merged[column] = merged[column].fillna(0.0)
    merged["capacity_ratio_clustered_to_unclustered"] = np.where(
        merged["capacity_MW_unclustered"] > 0,
        merged["capacity_MW_clustered"] / merged["capacity_MW_unclustered"],
        np.nan,
    )
    merged["capacity_difference_MW"] = (
        merged["capacity_MW_clustered"] - merged["capacity_MW_unclustered"]
    )
    merged["touches_focus_cluster"] = (
        merged["bus0"].isin(focus_clusters) | merged["bus1"].isin(focus_clusters)
    )
    return merged.sort_values(
        ["touches_focus_cluster", "capacity_MW_clustered"],
        ascending=[False, False],
    )


def create_diagnosis(unclustered, clustered, balancing, busmap, config: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    focus_clusters = set(config.get("focus_clusters", []))
    threshold = float(config.get("loading_threshold", 0.95))
    corridors = compare_corridors(unclustered, clustered, busmap, focus_clusters)
    loading = corridor_loading(balancing, threshold)
    cluster_report = cluster_balance(balancing, focus_clusters)

    slack = cluster_report.loc[cluster_report.metric == "feasibility_slack"]
    slack_energy = float(slack.energy_MWh.sum()) if not slack.empty else 0.0
    slack_names = balancing.generators.index[
        balancing.generators.carrier.astype(str).str.lower().eq("load_shedding")
    ]
    max_slack = float(
        balancing.generators_t.p[slack_names].sum(axis=1).clip(lower=0.0).max()
    ) if len(slack_names) else 0.0
    focus_corridors = corridors[corridors.touches_focus_cluster]
    materially_changed = focus_corridors[
        focus_corridors.capacity_ratio_clustered_to_unclustered.notna()
        & (focus_corridors.capacity_ratio_clustered_to_unclustered.sub(1.0).abs() > 0.05)
    ]
    binding = loading[loading.hours_at_or_above_threshold > 0.0]

    conclusion = (
        "No material feasibility slack in the reported period."
        if slack_energy <= 0.1
        else (
            "Feasibility slack is present. This report identifies affected clustered "
            "buses and thermal corridors, but cannot establish a genuine adequacy "
            "event without an otherwise identical unclustered constrained solve."
        )
    )
    summary = {
        "diagnostic": "WP4_FES_topology_and_adequacy",
        "reported_period_hours": float(snapshot_hours(balancing).sum()),
        "focus_clusters": sorted(focus_clusters),
        "loading_threshold": threshold,
        "feasibility_slack_energy_MWh": slack_energy,
        "peak_feasibility_slack_MW": max_slack,
        "affected_buses": sorted(slack.loc[slack.energy_MWh > 0.1, "bus"].unique().tolist()),
        "binding_ac_components": int(len(binding)),
        "material_cluster_envelope_changes": int(len(materially_changed)),
        "conclusion": conclusion,
        "required_next_test": (
            "Run the paired unclustered WP4 scenario before changing ETYS capacity "
            "or interpreting balancing costs as a realistic FES outcome."
        ),
    }
    return summary, corridors, cluster_report


if __name__ == "__main__":
    logger = setup_logging(snakemake.log[0])
    config = snakemake.params.topology_adequacy
    unclustered = load_network(snakemake.input.unclustered_network, custom_logger=logger)
    clustered = load_network(snakemake.input.clustered_network, custom_logger=logger)
    balancing = load_network(snakemake.input.balancing_network, custom_logger=logger)
    busmap_data = pd.read_csv(snakemake.input.busmap_csv)
    busmap = pd.Series(busmap_data.cluster.values, index=busmap_data.bus_id.values)

    summary, corridors, clusters = create_diagnosis(
        unclustered, clustered, balancing, busmap, config
    )
    loading = corridor_loading(balancing, float(config.get("loading_threshold", 0.95)))
    output_dir = Path(snakemake.output.summary_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    corridors.to_csv(snakemake.output.corridors_csv, index=False)
    clusters.to_csv(snakemake.output.cluster_balance_csv, index=False)
    loading.to_csv(snakemake.output.loading_csv, index=False)
    with open(snakemake.output.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("WP4 topology diagnosis: %s", json.dumps(summary))

