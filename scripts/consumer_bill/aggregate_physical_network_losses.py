"""Aggregate AC transmission losses from a complete set of physical PyPSA runs.

This intentionally excludes links: in PyPSA-GB they also represent
interconnectors and non-network conversions. The result is a physical
transmission-loss signal, not a supplier loss-factor charge.
"""


import argparse
import json
from pathlib import Path

import pandas as pd
import pypsa


def _weights(network: pypsa.Network) -> pd.Series:
    if "generators" in network.snapshot_weightings:
        return network.snapshot_weightings["generators"].astype(float)
    return pd.Series(1.0, index=network.snapshots)


def represented_ac_metrics_mwh(network: pypsa.Network) -> tuple[float, float]:
    """Return represented AC losses and gross line/transformer flow in MWh.

    A standard linear PyPSA network is lossless: its terminal flows are equal and
    opposite, even though line resistance is present in the static data.  Gross
    flow is retained so that an all-zero loss result can be identified as a
    modelling-property rather than mistaken for an observed physical outcome.
    """
    weights = _weights(network)
    losses = pd.Series(0.0, index=network.snapshots)
    gross_flow = pd.Series(0.0, index=network.snapshots)
    for component in ("lines", "transformers"):
        static = getattr(network, component)
        dynamic = getattr(network, f"{component}_t")
        if static.empty or not hasattr(dynamic, "p0") or not hasattr(dynamic, "p1"):
            continue
        losses = losses.add(dynamic.p0.add(dynamic.p1).sum(axis=1), fill_value=0.0)
        gross_flow = gross_flow.add(dynamic.p0.abs().sum(axis=1), fill_value=0.0)
    return max(float((losses * weights).sum()), 0.0), float((gross_flow * weights).sum())


def resistance_weighted_i2r_loss_proxy_mwh(network: pypsa.Network) -> tuple[float, float, float]:
    """Estimate line/transformer I²R losses from lossless dispatch flows.

    The LP dispatch has active-power flows but no voltage magnitudes or reactive
    power.  For a three-phase line, ``P² R / V²`` is therefore an approximate
    loss in MW. Transformer loss is similarly approximated as
    ``P² r_pu / s_nom``.  The result is deliberately an *index*: it must be
    calibrated to a published supplier-loss allowance and does not include
    distribution losses or regional supplier loss factors.
    """
    weights = _weights(network)

    def weighted_sum(frame: pd.DataFrame) -> float:
        return float((frame.sum(axis=1) * weights).sum()) if not frame.empty else 0.0

    line_loss = 0.0
    if not network.lines.empty and hasattr(network.lines_t, "p0"):
        lines = network.lines
        flow = network.lines_t.p0.reindex(columns=lines.index).fillna(0.0)
        voltage_kv = lines.bus0.map(network.buses.v_nom).astype(float)
        resistance_ohm = pd.to_numeric(lines.r, errors="coerce")
        valid = (voltage_kv > 0) & (resistance_ohm >= 0)
        if valid.any():
            coefficients = (resistance_ohm.loc[valid] / voltage_kv.loc[valid].pow(2))
            line_loss = weighted_sum(flow.loc[:, valid].pow(2).mul(coefficients, axis=1))

    transformer_loss = 0.0
    if not network.transformers.empty and hasattr(network.transformers_t, "p0"):
        transformers = network.transformers
        flow = network.transformers_t.p0.reindex(columns=transformers.index).fillna(0.0)
        resistance_pu = pd.to_numeric(transformers.r, errors="coerce")
        capacity_mva = pd.to_numeric(transformers.s_nom, errors="coerce")
        valid = (capacity_mva > 0) & (resistance_pu >= 0)
        if valid.any():
            coefficients = resistance_pu.loc[valid] / capacity_mva.loc[valid]
            transformer_loss = weighted_sum(flow.loc[:, valid].pow(2).mul(coefficients, axis=1))

    return line_loss + transformer_loss, line_loss, transformer_loss


def ac_transmission_losses_mwh(network: pypsa.Network) -> float:
    """Return represented line and transformer losses, excluding links."""
    return represented_ac_metrics_mwh(network)[0]


def loss_representation_status(loss_mwh: float, gross_flow_mwh: float) -> str:
    """Classify whether a saved network contains a usable AC-loss signal."""
    tolerance_mwh = max(gross_flow_mwh * 1e-12, 1e-6)
    if gross_flow_mwh > tolerance_mwh and loss_mwh <= tolerance_mwh:
        return "lossless_linear_network"
    if gross_flow_mwh <= tolerance_mwh:
        return "no_ac_line_or_transformer_flow"
    return "represented_ac_losses_available"


def _network_metrics(path: str | Path) -> dict:
    network = pypsa.Network(path)
    snapshots = pd.DatetimeIndex(network.snapshots)
    if snapshots.empty:
        raise ValueError(f"{path}: no snapshots")
    weights = _weights(network)
    demand_mwh = float((network.loads_t.p_set.sum(axis=1) * weights).sum())
    if demand_mwh <= 0:
        raise ValueError(f"{path}: model demand must be positive")
    loss_mwh, gross_flow_mwh = represented_ac_metrics_mwh(network)
    i2r_loss_proxy_mwh, line_i2r_loss_proxy_mwh, transformer_i2r_loss_proxy_mwh = (
        resistance_weighted_i2r_loss_proxy_mwh(network)
    )
    representation_status = loss_representation_status(loss_mwh, gross_flow_mwh)
    return {
        "network_file": str(path),
        "start": snapshots.min().isoformat(),
        "end": snapshots.max().isoformat(),
        "snapshot_count": len(snapshots),
        "demand_mwh": demand_mwh,
        "ac_line_transformer_gross_flow_mwh": gross_flow_mwh,
        "ac_transmission_loss_mwh": loss_mwh,
        "ac_transmission_loss_rate": loss_mwh / demand_mwh,
        "i2r_loss_proxy_mwh": i2r_loss_proxy_mwh,
        "line_i2r_loss_proxy_mwh": line_i2r_loss_proxy_mwh,
        "transformer_i2r_loss_proxy_mwh": transformer_i2r_loss_proxy_mwh,
        "i2r_loss_proxy_rate": i2r_loss_proxy_mwh / demand_mwh,
        "representation_status": representation_status,
    }


def aggregate_losses(network_paths: list[str | Path], expected_year: int | None = None) -> tuple[pd.DataFrame, dict]:
    if not network_paths:
        raise ValueError("At least one physical balancing network is required")
    rows = pd.DataFrame([_network_metrics(path) for path in network_paths])
    starts = pd.to_datetime(rows["start"])
    ends = pd.to_datetime(rows["end"])
    rows = rows.assign(_start=starts, _end=ends).sort_values("_start").reset_index(drop=True)
    if rows["_start"].duplicated().any() or (rows["_start"].iloc[1:].to_numpy() <= rows["_end"].iloc[:-1].to_numpy()).any():
        raise ValueError("Physical loss networks overlap or contain duplicate start times")
    if expected_year is not None and not ((rows["_start"].dt.year == expected_year) & (rows["_end"].dt.year == expected_year)).all():
        raise ValueError(f"Every physical loss network must lie within {expected_year}")
    demand = float(rows["demand_mwh"].sum())
    loss = float(rows["ac_transmission_loss_mwh"].sum())
    gross_flow = float(rows["ac_line_transformer_gross_flow_mwh"].sum())
    i2r_loss_proxy = float(rows["i2r_loss_proxy_mwh"].sum())
    statuses = sorted(rows["representation_status"].unique())
    signal_available = statuses == ["represented_ac_losses_available"]
    summary = {
        "purpose": "Physical AC transmission-loss diagnostic from completed balancing-network runs",
        "network_count": len(rows),
        "modelled_demand_mwh": demand,
        "ac_line_transformer_gross_flow_mwh": gross_flow,
        "ac_transmission_loss_mwh": loss,
        "ac_transmission_loss_rate": loss / demand,
        "i2r_loss_proxy_mwh": i2r_loss_proxy,
        "i2r_loss_proxy_rate": i2r_loss_proxy / demand,
        "i2r_loss_proxy_available": bool(i2r_loss_proxy > 0),
        "i2r_loss_proxy_formula": "sum(line P_MW^2 * R_ohm / V_kV^2) + sum(transformer P_MW^2 * r_pu / s_nom_MVA), weighted by snapshot duration",
        "representation_status": statuses,
        "loss_signal_available": signal_available,
        "treatment": (
            "Use only an available physical-loss signal as a calibrated supplier-loss bridge. "
            "The resistance-weighted flow-squared proxy is calibrated, not a physical AC power-flow result. "
            "This diagnostic excludes distribution losses, regional supplier loss factors and non-network link conversions. "
            "A lossless_linear_network result means the optimisation did not represent AC losses, so its zero must not be used as a loss estimate."
        ),
    }
    return rows.drop(columns=["_start", "_end"]), summary


def write_losses(network_paths, expected_year, output_csv, output_summary) -> None:
    """Aggregate a set of completed balancing networks and persist the diagnostic."""
    rows, summary = aggregate_losses(list(network_paths), expected_year)
    for output in (output_csv, output_summary):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="append", required=True)
    parser.add_argument("--expected-year", type=int)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_losses(args.network, args.expected_year, args.output_csv, args.output_summary)


if "snakemake" in globals():
    write_losses(snakemake.input.network_nc, snakemake.params.expected_year,
                 snakemake.output.losses_csv, snakemake.output.summary_json)
elif __name__ == "__main__":
    main()
