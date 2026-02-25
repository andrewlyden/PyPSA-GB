"""
Market Simulation Analysis — Post-processing for Two-Stage Dispatch

Produces a market-specific dashboard and JSON summary by comparing
wholesale (copperplate) and balancing mechanism (constrained) results.

Inputs:
  - Wholesale solved network (.nc)
  - Balancing solved network (.nc)
  - Redispatch summary CSV
  - Constraint costs CSV
  - Congestion CSV
  - Price comparison CSV
  - Wholesale dispatch CSV
  - Balancing dispatch CSV

Outputs:
  - Market dashboard (HTML): multi-panel interactive Plotly visualisation
  - Market summary (JSON): machine-readable key metrics

See Also:
  - scripts/market/solve_wholesale.py — Stage 1
  - scripts/market/solve_balancing.py — Stage 2
  - scripts/analysis/analyze_solved_network.py — Standard post-solve analysis
"""

import pypsa
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
import time
from pathlib import Path

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network


# ─────────────────────────────────────────────────────────────────────────────
# Carrier colour mapping (consistent with main analysis)
# ─────────────────────────────────────────────────────────────────────────────
CARRIER_COLORS = {
    "wind_onshore": "#4DAF4A",
    "wind_offshore": "#377EB8",
    "solar_pv": "#FFD700",
    "nuclear": "#E41A1C",
    "CCGT": "#FF7F00",
    "OCGT": "#A65628",
    "battery": "#984EA3",
    "pumped_hydro": "#00CED1",
    "large_hydro": "#00BFFF",
    "small_hydro": "#87CEEB",
    "biomass": "#228B22",
    "Bioenergy": "#228B22",
    "coal": "#333333",
    "Coal": "#333333",
    "oil": "#8B4513",
    "Oil": "#8B4513",
    "load_shedding": "#FF0000",
    "interconnector": "#808080",
    "DC": "#808080",
    "tidal_stream": "#006400",
    "shoreline_wave": "#2E8B57",
}


def _get_color(carrier: str) -> str:
    """Return colour for carrier, falling back to grey."""
    return CARRIER_COLORS.get(carrier, "#AAAAAA")


def _aggregate_by_carrier(dispatch_df: pd.DataFrame, network) -> pd.DataFrame:
    """Aggregate generator dispatch by carrier for stacked-area plotting."""
    carrier_map = {}
    for col in dispatch_df.columns:
        gen_name = col
        if gen_name in network.generators.index:
            carrier_map[col] = network.generators.loc[gen_name, "carrier"]
        else:
            carrier_map[col] = "unknown"

    dispatch_by_carrier = dispatch_df.rename(columns=carrier_map)
    # Group columns by carrier and sum
    return dispatch_by_carrier.T.groupby(level=0).sum().T


def create_market_dashboard(
    wholesale_network,
    balancing_network,
    wholesale_dispatch: pd.DataFrame,
    balancing_dispatch: pd.DataFrame,
    redispatch_summary: pd.DataFrame,
    constraint_costs: pd.DataFrame,
    congestion: pd.DataFrame,
    price_comparison: pd.DataFrame,
    output_path: str,
    logger: logging.Logger,
):
    """
    Create a multi-panel Plotly dashboard comparing wholesale vs BM dispatch.

    Panels:
      1. Wholesale vs Physical dispatch (stacked area, side-by-side)
      2. Redispatch volumes by carrier (bar chart: up/down)
      3. Constraint cost time series
      4. Line loading / congestion summary
      5. Wholesale vs nodal price spread
      6. Top redispatched assets table
    """
    logger.info("Creating market simulation dashboard...")

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Wholesale Dispatch (Copperplate)",
            "Physical Dispatch (BM Constrained)",
            "Redispatch Volumes by Carrier",
            "Constraint Costs by Carrier",
            "Price Comparison (Wholesale vs Nodal)",
            "Top 15 Congested Components",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # ── Panel 1 & 2: Wholesale vs Physical stacked area ──────────────────
    for panel_idx, (dispatch_df, net, title) in enumerate(
        [
            (wholesale_dispatch, wholesale_network, "Wholesale"),
            (balancing_dispatch, balancing_network, "Physical"),
        ],
        start=1,
    ):
        by_carrier = _aggregate_by_carrier(dispatch_df, net)
        col = panel_idx
        for carrier in sorted(by_carrier.columns):
            fig.add_trace(
                go.Scatter(
                    x=by_carrier.index,
                    y=by_carrier[carrier].clip(lower=0),
                    mode="lines",
                    stackgroup=f"dispatch_{panel_idx}",
                    name=f"{carrier}" if panel_idx == 1 else None,
                    line=dict(width=0),
                    fillcolor=_get_color(carrier),
                    showlegend=(panel_idx == 1),
                    legendgroup=carrier,
                ),
                row=1,
                col=col,
            )

    # ── Panel 3: Redispatch volumes by carrier (bar chart) ───────────────
    if not redispatch_summary.empty and "carrier" in redispatch_summary.columns:
        rd_by_carrier = (
            redispatch_summary.groupby("carrier")
            .agg(increase_MWh=("increase_MWh", "sum"), decrease_MWh=("decrease_MWh", "sum"))
        )
        rd_by_carrier = rd_by_carrier[
            (rd_by_carrier["increase_MWh"] > 0.1) | (rd_by_carrier["decrease_MWh"] > 0.1)
        ].sort_values("increase_MWh", ascending=False)

        fig.add_trace(
            go.Bar(
                x=rd_by_carrier.index,
                y=rd_by_carrier["increase_MWh"],
                name="Increase (offers)",
                marker_color="#2ca02c",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=rd_by_carrier.index,
                y=-rd_by_carrier["decrease_MWh"],
                name="Decrease (bids)",
                marker_color="#d62728",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="MWh", row=2, col=1)

    # ── Panel 4: Constraint costs by carrier ─────────────────────────────
    if not constraint_costs.empty and "net_cost" in constraint_costs.columns:
        # Drop the TOTAL row if present
        cc = constraint_costs.drop("TOTAL", errors="ignore")
        cc = cc[cc["net_cost"] > 0.1].sort_values("net_cost", ascending=True)

        fig.add_trace(
            go.Bar(
                x=cc["net_cost"],
                y=cc.index,
                orientation="h",
                name="Constraint Cost (£)",
                marker_color=[_get_color(c) for c in cc.index],
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="£", row=2, col=2)

    # ── Panel 5: Price comparison ────────────────────────────────────────
    if not price_comparison.empty and "wholesale_price" in price_comparison.columns:
        fig.add_trace(
            go.Scatter(
                x=price_comparison.index,
                y=price_comparison["wholesale_price"],
                mode="lines",
                name="Wholesale Price",
                line=dict(color="#1f77b4", width=2),
            ),
            row=3,
            col=1,
        )
        if "mean_nodal_price" in price_comparison.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_comparison.index,
                    y=price_comparison["mean_nodal_price"],
                    mode="lines",
                    name="Mean Nodal Price",
                    line=dict(color="#ff7f0e", width=2),
                ),
                row=3,
                col=1,
            )
        if "min_nodal_price" in price_comparison.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_comparison.index,
                    y=price_comparison["min_nodal_price"],
                    mode="lines",
                    name="Min Nodal",
                    line=dict(color="#ff7f0e", width=1, dash="dot"),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=price_comparison.index,
                    y=price_comparison["max_nodal_price"],
                    mode="lines",
                    name="Max Nodal",
                    line=dict(color="#ff7f0e", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(255,127,14,0.15)",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        fig.update_yaxes(title_text="£/MWh", row=3, col=1)

    # ── Panel 6: Top congested components ────────────────────────────────
    if not congestion.empty:
        top_congested = congestion.nlargest(15, "hours_congested")
        fig.add_trace(
            go.Bar(
                x=top_congested["hours_congested"],
                y=top_congested["component"],
                orientation="h",
                name="Hours Congested",
                marker_color="#e377c2",
                showlegend=False,
            ),
            row=3,
            col=2,
        )
        fig.update_xaxes(title_text="Hours", row=3, col=2)

    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=1200,
        width=1600,
        title_text="PyPSA-GB Market Simulation Dashboard",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
        template="plotly_white",
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Dashboard saved to {output_path}")


def create_market_summary(
    wholesale_network,
    balancing_network,
    redispatch_summary: pd.DataFrame,
    constraint_costs: pd.DataFrame,
    congestion: pd.DataFrame,
    price_comparison: pd.DataFrame,
    scenario_id: str,
    logger: logging.Logger,
) -> dict:
    """
    Create a machine-readable JSON summary of the market simulation.

    Returns
    -------
    dict
        Summary dictionary with key metrics.
    """
    summary = {
        "scenario": scenario_id,
        "market_mode": "two_stage",
        "network_size": {
            "buses": len(balancing_network.buses),
            "generators": len(balancing_network.generators),
            "storage_units": len(balancing_network.storage_units),
            "lines": len(balancing_network.lines),
            "links": len(balancing_network.links),
        },
        "snapshots": len(balancing_network.snapshots),
    }

    # Wholesale results
    if hasattr(wholesale_network, "objective") and wholesale_network.objective is not None:
        summary["wholesale"] = {
            "total_cost": float(wholesale_network.objective),
        }

    # BM results
    if hasattr(balancing_network, "objective") and balancing_network.objective is not None:
        summary["balancing"] = {
            "total_redispatch_cost": float(balancing_network.objective),
        }

    # Redispatch volumes
    if not redispatch_summary.empty:
        total_inc = redispatch_summary["increase_MWh"].sum()
        total_dec = redispatch_summary["decrease_MWh"].sum()
        total_offer_cost = redispatch_summary["offer_cost"].sum()
        total_bid_cost = redispatch_summary["bid_cost"].sum()

        summary["redispatch"] = {
            "total_increase_MWh": float(total_inc),
            "total_decrease_MWh": float(total_dec),
            "total_offer_cost": float(total_offer_cost),
            "total_bid_cost": float(total_bid_cost),
            "total_net_cost": float(total_offer_cost + total_bid_cost),
            "assets_redispatched": int((redispatch_summary["increase_MWh"] + redispatch_summary["decrease_MWh"] > 0.1).sum()),
        }

        # Per-carrier breakdown
        by_carrier = (
            redispatch_summary.groupby("carrier")
            .agg(
                increase_MWh=("increase_MWh", "sum"),
                decrease_MWh=("decrease_MWh", "sum"),
                net_cost=("net_cost", "sum"),
            )
            .to_dict(orient="index")
        )
        summary["redispatch"]["by_carrier"] = {
            k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_carrier.items()
        }

    # Congestion
    if not congestion.empty:
        summary["congestion"] = {
            "congested_lines": int((congestion["type"] == "line").sum()),
            "congested_transformers": int((congestion["type"] == "transformer").sum()),
            "most_congested": congestion.nlargest(5, "hours_congested")[
                ["component", "type", "hours_congested", "max_loading_fraction"]
            ].to_dict(orient="records"),
        }
    else:
        summary["congestion"] = {"congested_lines": 0, "congested_transformers": 0}

    # Price comparison
    if not price_comparison.empty and "wholesale_price" in price_comparison.columns:
        ws_mean = price_comparison["wholesale_price"].mean()
        summary["prices"] = {
            "mean_wholesale_price": float(ws_mean),
        }
        if "mean_nodal_price" in price_comparison.columns:
            summary["prices"]["mean_nodal_price"] = float(
                price_comparison["mean_nodal_price"].mean()
            )
        if "nodal_spread" in price_comparison.columns:
            summary["prices"]["max_nodal_spread"] = float(
                price_comparison["nodal_spread"].max()
            )
            summary["prices"]["mean_nodal_spread"] = float(
                price_comparison["nodal_spread"].mean()
            )

    logger.info(f"Market summary: {json.dumps(summary, indent=2, default=str)}")
    return summary


if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────
    log_path = (
        snakemake.log[0]
        if hasattr(snakemake, "log") and snakemake.log
        else "analyze_market"
    )
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("MARKET SIMULATION ANALYSIS")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        scenario_config = snakemake.params.scenario_config
        scenario_id = scenario_config.get("scenario_id", snakemake.wildcards.scenario)

        # ── Load networks ────────────────────────────────────────────────
        logger.info("Loading wholesale and balancing networks...")
        wholesale_network = load_network(
            snakemake.input.wholesale_network, custom_logger=logger
        )
        balancing_network = load_network(
            snakemake.input.balancing_network, custom_logger=logger
        )

        # ── Load CSVs ────────────────────────────────────────────────────
        wholesale_dispatch = pd.read_csv(
            snakemake.input.wholesale_dispatch_csv, index_col=0, parse_dates=True
        )
        balancing_dispatch = pd.read_csv(
            snakemake.input.balancing_dispatch_csv, index_col=0, parse_dates=True
        )
        redispatch_summary = pd.read_csv(snakemake.input.redispatch_summary_csv)
        constraint_costs = pd.read_csv(
            snakemake.input.constraint_costs_csv, index_col=0
        )
        congestion = pd.read_csv(snakemake.input.congestion_csv)
        price_comparison = pd.read_csv(
            snakemake.input.price_comparison_csv, index_col=0, parse_dates=True
        )

        # ── Create dashboard ─────────────────────────────────────────────
        create_market_dashboard(
            wholesale_network=wholesale_network,
            balancing_network=balancing_network,
            wholesale_dispatch=wholesale_dispatch,
            balancing_dispatch=balancing_dispatch,
            redispatch_summary=redispatch_summary,
            constraint_costs=constraint_costs,
            congestion=congestion,
            price_comparison=price_comparison,
            output_path=snakemake.output.dashboard,
            logger=logger,
        )

        # ── Create JSON summary ──────────────────────────────────────────
        summary = create_market_summary(
            wholesale_network=wholesale_network,
            balancing_network=balancing_network,
            redispatch_summary=redispatch_summary,
            constraint_costs=constraint_costs,
            congestion=congestion,
            price_comparison=price_comparison,
            scenario_id=scenario_id,
            logger=logger,
        )

        with open(snakemake.output.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved: {snakemake.output.summary_json}")

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"ANALYSIS COMPLETE (Total: {total_time:.2f}s)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"FATAL ERROR in market analysis: {e}", exc_info=True)
        raise
