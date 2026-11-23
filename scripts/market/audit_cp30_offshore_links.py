"""Audit capacity and realised use of CP30 explicit offshore export links."""

import json
from pathlib import Path

import pandas as pd

from scripts.utilities.network_io import load_network


def _component_name(prefix: str, source_bus: str) -> str:
    return f"{prefix}{str(source_bus).strip().replace('-', '_')}"


def _snapshot_hours(network) -> pd.Series:
    weights = network.snapshot_weightings
    for column in ("generators", "objective", "stores"):
        if column in weights.columns:
            return weights[column].astype(float)
    return pd.Series(1.0, index=network.snapshots, dtype=float)


def _incident_component_loading(network, component: str, landing_buses: pd.Series):
    """Return max loading and threshold exposure for components at each landing bus."""
    table = getattr(network, component)
    dispatch = getattr(network, f"{component}_t").p0
    capacity = table["s_nom"].where(table["s_nom"] > 0.0)
    loading = dispatch.abs().divide(capacity, axis="columns")
    rows = []
    for landing_bus in landing_buses:
        names = table.index[(table["bus0"] == landing_bus) | (table["bus1"] == landing_bus)]
        local = loading.reindex(columns=names)
        if local.empty or local.notna().sum().sum() == 0:
            rows.append({
                f"incident_{component}_count": int(len(names)),
                f"max_incident_{component}_loading": None,
                f"max_incident_{component}_name": None,
                f"incident_{component}_component_hours_at_or_above_95pct": 0.0,
            })
            continue
        maxima = local.max()
        max_name = maxima.idxmax()
        rows.append({
            f"incident_{component}_count": int(len(names)),
            f"max_incident_{component}_loading": float(maxima.loc[max_name]),
            f"max_incident_{component}_name": str(max_name),
            f"incident_{component}_component_hours_at_or_above_95pct": float(
                (local >= 0.95).sum().sum()
            ),
        })
    return pd.DataFrame(rows, index=landing_buses.index)


def audit_offshore_export_links(
    base_network_path: str | Path,
    balancing_network_path: str | Path,
    mapping_path: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    scenario_id: str,
) -> dict:
    """Write one transparent row per CP30 offshore export link.

    Link capacity is compared with the attached offshore generator capacity and
    realised balancing-stage export. Mapping evidence is joined by the stable
    link naming convention used in CP30.connection_allocation.
    """
    base = load_network(base_network_path)
    balancing = load_network(balancing_network_path)
    mapping = pd.read_csv(mapping_path)
    required = {"carrier", "source_bus", "landing_bus", "source"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Offshore-link audit mapping lacks columns: {sorted(missing)}")

    links = balancing.links[balancing.links.carrier.eq("offshore_export")].copy()
    if links.empty:
        raise ValueError(f"{scenario_id} has no offshore_export links to audit")
    links.index.name = "link_name"
    links = links.reset_index()
    links["source_bus"] = links["link_name"].str.removeprefix("CP30_EXPORT_")

    mapping = mapping.copy()
    mapping["link_name"] = mapping["source_bus"].map(
        lambda source_bus: _component_name("CP30_EXPORT_", source_bus)
    )
    mapping = mapping.rename(columns={
        "source": "mapping_evidence",
        "landing_bus": "mapping_landing_bus",
        "source_bus": "mapping_source_bus",
    })
    links = links.merge(
        mapping[["link_name", "mapping_source_bus", "mapping_landing_bus", "mapping_evidence"]],
        on="link_name", how="left", validate="one_to_one",
    )
    if links["mapping_evidence"].isna().any():
        missing_links = links.loc[links["mapping_evidence"].isna(), "link_name"].tolist()
        raise ValueError(f"Offshore-link audit has no source mapping for: {missing_links}")
    links["source_bus"] = links["mapping_source_bus"]

    base_gen_capacity = base.generators.groupby("bus")["p_nom"].sum()
    links["attached_generator_capacity_MW"] = links["bus0"].map(base_gen_capacity).fillna(0.0)
    links["export_capacity_factor_of_connected_generation"] = (
        links["p_nom"] / links["attached_generator_capacity_MW"].replace(0.0, pd.NA)
    )
    if not (links["bus1"].astype(str) == links["mapping_landing_bus"].astype(str)).all():
        raise ValueError("Offshore-link audit found a link whose landing bus differs from its mapping")
    links["zero_displacement_mapping"] = links["source_bus"].astype(str).eq(
        links["mapping_landing_bus"].astype(str)
    )
    links["landing_bus_voltage_kV"] = links["bus1"].map(balancing.buses["v_nom"])
    for component in ("lines", "transformers"):
        context = _incident_component_loading(balancing, component, links["bus1"])
        links = links.join(context)

    hours = _snapshot_hours(balancing)
    p0 = balancing.links_t.p0.reindex(columns=links["link_name"], fill_value=0.0)
    positive = p0.clip(lower=0.0)
    capacity = links.set_index("link_name")["p_nom"].reindex(p0.columns)
    utilisation = positive.divide(capacity, axis="columns")
    links = links.set_index("link_name")
    links["balancing_export_MWh"] = positive.mul(hours, axis="index").sum()
    links["mean_export_MW"] = positive.mul(hours, axis="index").sum() / hours.sum()
    links["max_export_MW"] = positive.max()
    links["mean_utilisation"] = utilisation.mul(hours, axis="index").sum() / hours.sum()
    links["max_utilisation"] = utilisation.max()
    links["hours_at_or_above_95pct"] = (utilisation >= 0.95).mul(hours, axis="index").sum()
    links["hours_at_or_above_99pct"] = (utilisation >= 0.99).mul(hours, axis="index").sum()
    links["balancing_export_share_of_attached_capacity_hours"] = (
        links["balancing_export_MWh"] /
        (links["attached_generator_capacity_MW"] * hours.sum()).replace(0.0, pd.NA)
    )
    links["capacity_balance_check_MW"] = links["p_nom"] - links["attached_generator_capacity_MW"] * links[
        "export_capacity_factor_of_connected_generation"
    ]

    columns = [
        "carrier", "source_bus", "bus0", "bus1", "mapping_landing_bus", "mapping_evidence",
        "zero_displacement_mapping", "landing_bus_voltage_kV",
        "p_nom", "attached_generator_capacity_MW", "export_capacity_factor_of_connected_generation",
        "balancing_export_MWh", "mean_export_MW", "max_export_MW", "mean_utilisation",
        "max_utilisation", "hours_at_or_above_95pct", "hours_at_or_above_99pct",
        "balancing_export_share_of_attached_capacity_hours", "capacity_balance_check_MW",
        "incident_lines_count", "max_incident_lines_loading", "max_incident_lines_name",
        "incident_lines_component_hours_at_or_above_95pct", "incident_transformers_count",
        "max_incident_transformers_loading", "max_incident_transformers_name",
        "incident_transformers_component_hours_at_or_above_95pct",
    ]
    links = links[columns].rename(columns={"p_nom": "export_link_capacity_MW"})
    output_csv = Path(output_csv)
    output_json = Path(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    links.reset_index().to_csv(output_csv, index=False)
    summary = {
        "scenario": scenario_id,
        "links": int(len(links)),
        "period_hours": float(hours.sum()),
        "export_link_capacity_MW": float(links["export_link_capacity_MW"].sum()),
        "attached_generator_capacity_MW": float(links["attached_generator_capacity_MW"].sum()),
        "balancing_export_MWh": float(links["balancing_export_MWh"].sum()),
        "capacity_weighted_mean_utilisation": float(
            (links["mean_utilisation"] * links["export_link_capacity_MW"]).sum() /
            links["export_link_capacity_MW"].sum()
        ),
        "links_at_or_above_95pct_for_any_hour": int((links["hours_at_or_above_95pct"] > 0).sum()),
        "links_at_or_above_99pct_for_any_hour": int((links["hours_at_or_above_99pct"] > 0).sum()),
        "zero_displacement_mappings": int(links["zero_displacement_mapping"].sum()),
        "zero_displacement_export_link_capacity_MW": float(
            links.loc[links["zero_displacement_mapping"], "export_link_capacity_MW"].sum()
        ),
        "landing_buses_with_incident_line_loading_at_or_above_95pct": int(
            (links["max_incident_lines_loading"] >= 0.95).sum()
        ),
        "landing_buses_with_incident_transformer_loading_at_or_above_95pct": int(
            (links["max_incident_transformers_loading"] >= 0.95).sum()
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    audit_offshore_export_links(
        snakemake.input.base_network,
        snakemake.input.balancing_network,
        snakemake.input.mapping_csv,
        snakemake.output.audit_csv,
        snakemake.output.summary_json,
        snakemake.wildcards.scenario,
    )
