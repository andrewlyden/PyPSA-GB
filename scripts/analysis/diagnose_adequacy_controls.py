"""Post-solve controlled-adequacy diagnosis (no topology changes)."""
import json
from pathlib import Path
import pandas as pd
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network


def _hours(network):
    return pd.Series(1.0, index=network.snapshots)


def diagnose(network, relaxation: pd.DataFrame, threshold: float):
    hours = _hours(network)
    shed = network.generators.index[network.generators.carrier.astype(str).str.lower().eq("load_shedding")]
    shed_by_bus = pd.DataFrame(columns=["bus", "energy_MWh", "peak_MW"])
    if len(shed):
        dispatch = network.generators_t.p.reindex(columns=shed, fill_value=0.0).clip(lower=0.0)
        buses = network.generators.loc[shed, "bus"]
        shed_by_bus = pd.DataFrame([{"bus": str(bus), "energy_MWh": float((dispatch[names].sum(axis=1) * hours).sum()), "peak_MW": float(dispatch[names].sum(axis=1).max())} for bus, names in buses.groupby(buses).groups.items()])
        shed_by_bus = shed_by_bus.sort_values("energy_MWh", ascending=False)
    rows = []
    for typ, table_name, time_name in (("line", "lines", "lines_t"), ("transformer", "transformers", "transformers_t")):
        table, dispatch = getattr(network, table_name), getattr(network, time_name).p0
        for name, row in table.iterrows():
            if name not in dispatch or float(row.s_nom) <= 0:
                continue
            load = dispatch[name].abs() / float(row.s_nom)
            status = str(row.get("etys_topology_status", ""))
            source = str(row.get("etys_topology_source", ""))
            topology_class = "unverified_topology_sensitivity" if status == "sensitivity" else ("documented_topology_correction" if source and source != "nan" else "existing_or_ETYS_asset")
            rows.append({"component_type": typ, "component": str(name), "bus0": str(row.bus0), "bus1": str(row.bus1), "s_nom_MW": float(row.s_nom), "max_loading_fraction": float(load.max()), "hours_at_threshold": float((load >= threshold).sum()), "mean_loading_fraction": float(load.mean()), "topology_class": topology_class, "topology_source": source if source != "nan" else ""})
    ranking = pd.DataFrame(rows).sort_values(["hours_at_threshold", "max_loading_fraction"], ascending=False)
    slack = float(shed_by_bus.energy_MWh.sum()) if not shed_by_bus.empty else 0.0
    summary = {"diagnostic": "controlled_adequacy", "feasibility_slack_energy_MWh": slack, "peak_feasibility_slack_MW": float(network.generators_t.p.reindex(columns=shed, fill_value=0.0).clip(lower=0.0).sum(axis=1).max()) if len(shed) else 0.0, "binding_ac_components": int((ranking.hours_at_threshold > 0).sum()) if not ranking.empty else 0, "relaxation_mode": str(relaxation.iloc[0]["mode"]) if not relaxation.empty else "none", "relaxed_components": int(relaxation["changed"].sum()) if not relaxation.empty else 0, "conclusion": "Aggregate capacity is adequate under the tested controls." if slack <= 1.0 else "Feasibility slack remains; use the ranked components for the next controlled relaxation case."}
    return summary, ranking, shed_by_bus

if __name__ == "__main__":
    logger = setup_logging(snakemake.log[0])
    network = load_network(snakemake.input.balancing_network, custom_logger=logger)
    relaxation = pd.read_csv(snakemake.input.relaxation_csv)
    config = snakemake.params.topology_adequacy or {}
    summary, ranking, shed = diagnose(network, relaxation, float(config.get("loading_threshold", 0.95)))
    Path(snakemake.output.summary_json).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(snakemake.output.binding_components_csv, index=False)
    shed.to_csv(snakemake.output.shedding_by_bus_csv, index=False)
    with open(snakemake.output.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Controlled adequacy diagnosis: %s", json.dumps(summary))
