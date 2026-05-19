"""
Diagnose whether mapped NESO boundaries behave like the NESO DA flow data.

This is intended for validation scenarios where BM results already exist.  It
checks whether each mapped boundary:

  * has the same high-utilisation hours as NESO,
  * has a sensible signed-flow construction,
  * forms a meaningful cut in the PyPSA topology,
  * isolates an area with enough generation/load to explain the NESO flow.

Outputs are written to resources/analysis/<scenario>_boundary_diagnostics/.

Example:
    python scripts/analysis/diagnose_neso_boundary_mapping.py --scenario Validation_Jun2020
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = ROOT / "data" / "network" / "neso_boundary_mapping.yaml"
DEFAULT_NESO_DA = ROOT / "data" / "validation" / "day_ahead_constraint_flows_limits.csv"
DEFAULT_MARKET = ROOT / "resources" / "market"
DEFAULT_ANALYSIS = ROOT / "resources" / "analysis"


def _read_mapping(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("boundaries", data)


def _iter_boundary_specs(mapping: dict):
    """Yield top-level boundaries plus validation subconstraints.

    A subconstraint is compared against its parent NESO boundary data but is
    reported with a distinct diagnostic name, e.g. ``SSHARN::SSHARN3``.
    """
    for parent_boundary, boundary_def in mapping.items():
        yield parent_boundary, parent_boundary, boundary_def, "boundary"
        for name, sub_def in (boundary_def.get("subconstraints") or {}).items():
            yield f"{parent_boundary}::{name}", parent_boundary, sub_def, "subconstraint"


def _read_da_flows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    date_col = next((c for c in df.columns if "Date" in c), None)
    if date_col is None:
        raise ValueError(f"Could not find date column in {path}")

    df = df.rename(
        columns={
            "Constraint Group": "boundary",
            date_col: "datetime",
            "Limit (MW)": "limit_mw",
            "Flow (MW)": "flow_mw",
        }
    )
    required = {"boundary", "datetime", "limit_mw", "flow_mw"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required NESO DA columns in {path}: {sorted(missing)}")

    df = df.loc[:, ["boundary", "datetime", "limit_mw", "flow_mw"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
    df["limit_mw"] = pd.to_numeric(df["limit_mw"], errors="coerce")
    df["flow_mw"] = pd.to_numeric(df["flow_mw"], errors="coerce")
    return df.dropna(subset=["boundary", "datetime", "limit_mw", "flow_mw"])


def _line_groups(boundary_def: dict, network: pypsa.Network) -> tuple[list[str], list[str]]:
    line_ids = list(
        dict.fromkeys(boundary_def.get("lines") or boundary_def.get("constraint_lines") or [])
    )
    flow_groups = boundary_def.get("flow_groups") or {}
    positive = list(dict.fromkeys(flow_groups.get("positive") or line_ids))
    negative = list(dict.fromkeys(flow_groups.get("negative") or []))
    positive = [line for line in positive if line in network.lines_t.p0.columns]
    negative = [line for line in negative if line in network.lines_t.p0.columns]
    return positive, negative


def _link_groups(boundary_def: dict, network: pypsa.Network) -> list[tuple[str, float]]:
    links: list[tuple[str, float]] = []
    for entry in boundary_def.get("links", []) or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            sign = float(entry.get("sign", 1.0))
        else:
            name = entry
            sign = 1.0
        if name in network.links_t.p0.columns:
            links.append((name, sign))
    return links


def _signed_flow(
    network: pypsa.Network, boundary_def: dict
) -> tuple[pd.Series, pd.Series, list[dict]]:
    positive, negative = _line_groups(boundary_def, network)
    links = _link_groups(boundary_def, network)

    signed = pd.Series(0.0, index=network.snapshots)
    gross = pd.Series(0.0, index=network.snapshots)
    components: list[dict] = []

    for line in positive:
        series = network.lines_t.p0[line].astype(float)
        signed = signed.add(series, fill_value=0.0)
        gross = gross.add(series.abs(), fill_value=0.0)
        components.append({"component": line, "kind": "line", "sign": 1.0, "series": series})

    for line in negative:
        series = -network.lines_t.p0[line].astype(float)
        signed = signed.add(series, fill_value=0.0)
        gross = gross.add(series.abs(), fill_value=0.0)
        components.append({"component": line, "kind": "line", "sign": -1.0, "series": series})

    for link, sign in links:
        series = sign * network.links_t.p0[link].astype(float)
        signed = signed.add(series, fill_value=0.0)
        gross = gross.add(series.abs(), fill_value=0.0)
        components.append({"component": link, "kind": "link", "sign": sign, "series": series})

    return signed, gross, components


def _build_ac_graph(network: pypsa.Network, remove_lines: set[str] | None = None) -> nx.Graph:
    remove_lines = remove_lines or set()
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)

    for name, row in network.lines.iterrows():
        if name not in remove_lines:
            graph.add_edge(row.bus0, row.bus1, key=name, kind="line")

    for name, row in network.transformers.iterrows():
        graph.add_edge(row.bus0, row.bus1, key=name, kind="transformer")

    return graph


def _carrier_capacity(network: pypsa.Network, buses: set[str]) -> pd.Series:
    gens = network.generators.loc[network.generators.bus.isin(buses)].copy()
    if gens.empty:
        return pd.Series(dtype=float)
    gens = gens.loc[gens.carrier != "load_shedding"]
    if gens.empty:
        return pd.Series(dtype=float)
    return gens.groupby("carrier")["p_nom"].sum().sort_values(ascending=False)


def _component_net_injection(network: pypsa.Network, buses: set[str]) -> pd.Series:
    idx = network.snapshots
    gens = network.generators.index[network.generators.bus.isin(buses)]
    loads = network.loads.index[network.loads.bus.isin(buses)]
    storage = network.storage_units.index[network.storage_units.bus.isin(buses)]

    gens = gens.intersection(network.generators_t.p.columns)
    loads = loads.intersection(network.loads_t.p_set.columns)
    storage = storage.intersection(network.storage_units_t.p.columns)

    out = pd.Series(0.0, index=idx)
    if len(gens):
        out = out.add(network.generators_t.p.loc[:, gens].sum(axis=1), fill_value=0.0)
    if len(storage):
        out = out.add(network.storage_units_t.p.loc[:, storage].sum(axis=1), fill_value=0.0)
    if len(loads):
        out = out.sub(network.loads_t.p_set.loc[:, loads].sum(axis=1), fill_value=0.0)
    return out


def _top_generators(network: pypsa.Network, buses: set[str], n: int = 8) -> str:
    gens = network.generators.loc[network.generators.bus.isin(buses)].copy()
    gens = gens.loc[gens.carrier != "load_shedding"]
    if gens.empty:
        return ""
    parts = []
    for name, row in gens.sort_values("p_nom", ascending=False).head(n).iterrows():
        parts.append(f"{name} ({row.carrier}, {row.p_nom:.0f} MW, {row.bus})")
    return "; ".join(parts)


def _boundary_lines(boundary_def: dict, network: pypsa.Network) -> list[str]:
    line_ids = list(
        dict.fromkeys(boundary_def.get("lines") or boundary_def.get("constraint_lines") or [])
    )
    return [line for line in line_ids if line in network.lines.index]


def _component_candidates(
    network: pypsa.Network,
    boundary: str,
    boundary_def: dict,
    common: pd.DatetimeIndex,
    neso_high: pd.Series,
) -> tuple[list[dict], list[dict]]:
    mapped_lines = _boundary_lines(boundary_def, network)
    graph = _build_ac_graph(network, set(mapped_lines))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest_size = len(components[0]) if components else 0

    component_rows: list[dict] = []
    generation_rows: list[dict] = []

    for component in components:
        buses = set(component)
        crossing = []
        for line in mapped_lines:
            row = network.lines.loc[line]
            if (row.bus0 in buses) != (row.bus1 in buses):
                crossing.append(line)
        if not crossing:
            continue

        capacity = _carrier_capacity(network, buses)
        non_load_p_nom = float(capacity.sum()) if not capacity.empty else 0.0
        net_inj = _component_net_injection(network, buses).reindex(common)
        high_values = net_inj.loc[neso_high.index[neso_high]]

        component_id = f"{boundary}_component_{len(component_rows) + 1}"
        component_rows.append(
            {
                "boundary": boundary,
                "component_id": component_id,
                "buses": len(buses),
                "is_largest_component": len(buses) == largest_size,
                "crossing_mapped_lines": len(crossing),
                "crossing_line_ids": "; ".join(crossing),
                "non_load_p_nom_mw": non_load_p_nom,
                "mean_net_injection_mw": float(net_inj.mean()),
                "mean_net_injection_on_neso_high_mw": (
                    float(high_values.mean()) if len(high_values) else np.nan
                ),
                "bus_ids": "; ".join(sorted(buses)[:80]),
                "top_generators": _top_generators(network, buses),
            }
        )

        for carrier, value in capacity.items():
            generation_rows.append(
                {
                    "boundary": boundary,
                    "component_id": component_id,
                    "carrier": carrier,
                    "p_nom_mw": float(value),
                }
            )

    component_rows.sort(
        key=lambda row: (
            row["is_largest_component"],
            -row["non_load_p_nom_mw"],
            -row["buses"],
        )
    )
    return component_rows, generation_rows


def _diagnosis(row: dict, component_rows: list[dict]) -> str:
    notes: list[str] = []
    if row["neso_high_hours"] and row["bm_high_hours_on_neso_high"] == 0:
        notes.append("model never reaches NESO high-utilisation state")
    if row["bm_mean_util_on_neso_high"] < 0.6:
        notes.append("model transfer is too low on NESO high hours")
    if row["bm_mean_util_on_neso_high"] > 1.25:
        notes.append("model/proxy transfer is too high on NESO high hours")
    if row["mean_abs_gap_bm_minus_neso_mw"] > 0.25 * row["neso_max_flow_mw"]:
        notes.append("model/proxy overstates NESO flow magnitude")
    if row["corr_abs_flow"] < 0:
        notes.append("model and NESO flow timing are negatively correlated")
    elif row["corr_abs_flow"] < 0.3:
        notes.append("weak model/NESO flow timing correlation")
    if row["net_over_gross_on_neso_high"] < 0.7:
        notes.append("signed branches cancel, so mapping may not be a clean cut")

    small_candidates = [
        comp for comp in component_rows if not comp["is_largest_component"]
    ]
    high_overlap = (
        row["bm_high_hours_on_neso_high"] / row["neso_high_hours"]
        if row["neso_high_hours"]
        else 1.0
    )
    if small_candidates and (high_overlap < 0.75 or row["bm_mean_util_on_neso_high"] < 0.75):
        best = max(small_candidates, key=lambda comp: comp["non_load_p_nom_mw"])
        if best["non_load_p_nom_mw"] < 0.5 * row["neso_max_flow_mw"]:
            notes.append(
                "largest isolated area has too little generation capacity "
                "for NESO flow magnitude"
            )

    return "; ".join(notes) if notes else "mapping broadly consistent"


def run(args: argparse.Namespace) -> None:
    scenario = args.scenario
    market_dir = Path(args.market_dir)
    output_dir = Path(args.output_dir) / f"{scenario}_boundary_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    bm_network_path = market_dir / f"{scenario}_balancing.nc"
    ws_network_path = market_dir / f"{scenario}_wholesale.nc"
    if not bm_network_path.exists():
        raise FileNotFoundError(f"Missing balancing network: {bm_network_path}")
    if not ws_network_path.exists():
        raise FileNotFoundError(f"Missing wholesale network: {ws_network_path}")

    bm = pypsa.Network(bm_network_path)
    ws = pypsa.Network(ws_network_path)
    mapping = _read_mapping(Path(args.mapping))
    da = _read_da_flows(Path(args.neso_da))

    start = pd.Timestamp(bm.snapshots[0])
    end = pd.Timestamp(bm.snapshots[-1])
    da = da.loc[
        (da["datetime"] >= start)
        & (da["datetime"] <= end)
        & (da["datetime"].dt.minute == 0)
    ].copy()

    summary_rows: list[dict] = []
    all_component_rows: list[dict] = []
    all_generation_rows: list[dict] = []
    component_flow_rows: list[dict] = []

    for boundary, neso_boundary, boundary_def, spec_type in _iter_boundary_specs(mapping):
        neso = da.loc[da["boundary"] == neso_boundary].set_index("datetime").sort_index()
        if neso.empty:
            continue

        common = pd.DatetimeIndex(neso.index).intersection(bm.snapshots).intersection(ws.snapshots)
        if len(common) == 0:
            continue

        limit = neso.loc[common, "limit_mw"].astype(float)
        neso_flow = neso.loc[common, "flow_mw"].astype(float)
        neso_util = (neso_flow.abs() / limit).replace([np.inf, -np.inf], np.nan)
        neso_high = neso_util >= args.high_utilisation_threshold

        bm_signed, bm_gross, bm_components = _signed_flow(bm, boundary_def)
        ws_signed, _ws_gross, _ws_components = _signed_flow(ws, boundary_def)
        bm_flow = bm_signed.reindex(common).abs()
        ws_flow = ws_signed.reindex(common).abs()
        bm_util = (bm_flow / limit).replace([np.inf, -np.inf], np.nan)
        ws_util = (ws_flow / limit).replace([np.inf, -np.inf], np.nan)

        if len(common) > 2:
            corr = float(
                np.corrcoef(
                    bm_flow.fillna(0.0).to_numpy(),
                    neso_flow.abs().fillna(0.0).to_numpy(),
                )[0, 1]
            )
        else:
            corr = np.nan

        top_component_parts = []
        for component in bm_components:
            series = component["series"].reindex(common)
            subset = series.loc[neso_high.index[neso_high]] if neso_high.any() else series
            top_component_parts.append(
                {
                    "boundary": boundary,
                    "neso_boundary": neso_boundary,
                    "spec_type": spec_type,
                    "component": component["component"],
                    "kind": component["kind"],
                    "sign": component["sign"],
                    "mean_abs_flow_on_neso_high_mw": float(subset.abs().mean()),
                    "mean_signed_flow_on_neso_high_mw": float(subset.mean()),
                }
            )
        component_flow_rows.extend(top_component_parts)

        net_over_gross = (bm_flow / bm_gross.reindex(common).replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        high_index = neso_high.index[neso_high]

        component_rows, generation_rows = _component_candidates(
            bm, boundary, boundary_def, common, neso_high
        )
        all_component_rows.extend(component_rows)
        all_generation_rows.extend(generation_rows)

        row = {
            "boundary": boundary,
            "neso_boundary": neso_boundary,
            "spec_type": spec_type,
            "hours_compared": int(len(common)),
            "neso_high_hours": int(neso_high.sum()),
            "ws_high_hours_on_neso_high": int((ws_util.loc[high_index] >= args.high_utilisation_threshold).sum()),
            "bm_high_hours_on_neso_high": int((bm_util.loc[high_index] >= args.high_utilisation_threshold).sum()),
            "neso_mean_util": float(neso_util.mean()),
            "ws_mean_util": float(ws_util.mean()),
            "bm_mean_util": float(bm_util.mean()),
            "ws_mean_util_on_neso_high": (
                float(ws_util.loc[high_index].mean()) if len(high_index) else np.nan
            ),
            "bm_mean_util_on_neso_high": (
                float(bm_util.loc[high_index].mean()) if len(high_index) else np.nan
            ),
            "bm_max_util_on_neso_high": (
                float(bm_util.loc[high_index].max()) if len(high_index) else np.nan
            ),
            "neso_mean_flow_mw": float(neso_flow.abs().mean()),
            "neso_max_flow_mw": float(neso_flow.abs().max()),
            "ws_mean_flow_mw": float(ws_flow.mean()),
            "bm_mean_flow_mw": float(bm_flow.mean()),
            "bm_mean_flow_on_neso_high_mw": (
                float(bm_flow.loc[high_index].mean()) if len(high_index) else np.nan
            ),
            "mean_abs_gap_bm_minus_neso_mw": float((bm_flow - neso_flow.abs()).mean()),
            "corr_abs_flow": corr,
            "net_over_gross_mean": float(net_over_gross.mean()),
            "net_over_gross_on_neso_high": (
                float(net_over_gross.loc[high_index].mean()) if len(high_index) else np.nan
            ),
            "mapped_lines": len(_boundary_lines(boundary_def, bm)),
            "mapped_links": len(_link_groups(boundary_def, bm)),
            "cut_components_with_mapped_edges": len(component_rows),
        }
        row["diagnosis"] = _diagnosis(row, component_rows)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    components = pd.DataFrame(all_component_rows)
    generation = pd.DataFrame(all_generation_rows)
    component_flows = pd.DataFrame(component_flow_rows)

    summary_path = output_dir / "boundary_diagnostic_summary.csv"
    components_path = output_dir / "boundary_component_candidates.csv"
    generation_path = output_dir / "boundary_component_generation.csv"
    component_flows_path = output_dir / "boundary_component_flows.csv"
    report_path = output_dir / "boundary_diagnostic_report.md"

    summary.to_csv(summary_path, index=False)
    components.to_csv(components_path, index=False)
    generation.to_csv(generation_path, index=False)
    component_flows.to_csv(component_flows_path, index=False)

    _write_report(
        report_path=report_path,
        scenario=scenario,
        summary=summary,
        components=components,
        generation=generation,
        component_flows=component_flows,
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {components_path}")
    print(f"Wrote {generation_path}")
    print(f"Wrote {component_flows_path}")


def _fmt(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def _write_report(
    report_path: Path,
    scenario: str,
    summary: pd.DataFrame,
    components: pd.DataFrame,
    generation: pd.DataFrame,
    component_flows: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append(f"# NESO Boundary Mapping Diagnostic: {scenario}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Boundary | NESO group | Type | NESO high h | BM high on same h | BM util on NESO high | "
        "NESO max flow MW | BM mean flow MW | Corr | Diagnosis |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.boundary} | {row.neso_boundary} | {row.spec_type} | "
            f"{int(row.neso_high_hours)} | "
            f"{int(row.bm_high_hours_on_neso_high)} | "
            f"{_fmt(row.bm_mean_util_on_neso_high, 2)} | "
            f"{_fmt(row.neso_max_flow_mw)} | {_fmt(row.bm_mean_flow_mw)} | "
            f"{_fmt(row.corr_abs_flow, 2)} | {row.diagnosis} |"
        )

    lines.append("")
    lines.append("## Boundary Components")
    lines.append("")
    lines.append(
        "These are topology components created by removing the mapped boundary "
        "lines. Small isolated areas with too little capacity relative to NESO "
        "flows are strong evidence that the mapping is not the real NESO cut."
    )
    lines.append("")
    lines.append(
        "| Boundary | Component | Buses | Largest? | Mapped crossing lines | "
        "Non-load p_nom MW | Net injection on NESO high MW | Top generators |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    if not components.empty:
        display = components.loc[~components["is_largest_component"]].copy()
        display = display.sort_values(
            ["boundary", "non_load_p_nom_mw"], ascending=[True, False]
        )
        for _, row in display.iterrows():
            lines.append(
                f"| {row.boundary} | {row.component_id} | {int(row.buses)} | "
                f"{bool(row.is_largest_component)} | {int(row.crossing_mapped_lines)} | "
                f"{_fmt(row.non_load_p_nom_mw)} | "
                f"{_fmt(row.mean_net_injection_on_neso_high_mw)} | "
                f"{row.top_generators} |"
            )

    lines.append("")
    lines.append("## Largest Boundary Branch Flows On NESO High Hours")
    lines.append("")
    lines.append("| Boundary | Component | Kind | Sign | Mean abs MW | Mean signed MW |")
    lines.append("|---|---|---|---:|---:|---:|")
    if not component_flows.empty:
        flows = component_flows.sort_values(
            ["boundary", "mean_abs_flow_on_neso_high_mw"],
            ascending=[True, False],
        )
        flows = flows.groupby("boundary", as_index=False).head(8)
        for _, row in flows.iterrows():
            lines.append(
                f"| {row.boundary} | {row.component} | {row.kind} | "
                f"{_fmt(row.sign, 0)} | "
                f"{_fmt(row.mean_abs_flow_on_neso_high_mw)} | "
                f"{_fmt(row.mean_signed_flow_on_neso_high_mw)} |"
            )

    lines.append("")
    lines.append("## Carrier Capacity In Candidate Components")
    lines.append("")
    lines.append("| Boundary | Component | Carrier | p_nom MW |")
    lines.append("|---|---|---|---:|")
    if not generation.empty:
        gen = generation.sort_values(
            ["boundary", "component_id", "p_nom_mw"], ascending=[True, True, False]
        )
        gen = gen.groupby(["boundary", "component_id"], as_index=False).head(8)
        for _, row in gen.iterrows():
            lines.append(
                f"| {row.boundary} | {row.component_id} | {row.carrier} | "
                f"{_fmt(row.p_nom_mw)} |"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="Validation_Jun2020")
    parser.add_argument("--market-dir", default=str(DEFAULT_MARKET))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--neso-da", default=str(DEFAULT_NESO_DA))
    parser.add_argument("--output-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--high-utilisation-threshold", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
