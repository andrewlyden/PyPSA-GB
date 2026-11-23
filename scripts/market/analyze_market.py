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
from scripts.market.market_accounting import (
    attach_interconnector_retrading,
    create_market_accounting_report,
)

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network
from scripts.interconnectors.market_coupling import (
    get_piecewise_interconnectors,
    get_directional_interconnector_pairs,
    counter_trade_premium_cost,
    interconnector_trade_cost,
    link_price_tables,
)


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
    accounting: dict | None = None,
):
    """
    Create a multi-panel Plotly dashboard comparing wholesale vs BM dispatch.

    Panels:
      1. Wholesale vs Physical dispatch (stacked area, side-by-side)
      2. Redispatch volumes by carrier (bar chart: up/down)
      3. Modelled redispatch cost by carrier
      4. Line loading / congestion summary
      5. Wholesale price vs balancing nodal optimisation duals
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
            "Modelled Redispatch Cost by Carrier",
            "Wholesale Price vs BM Nodal Optimisation Dual",
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
                    name="Mean BM Nodal Dual",
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
                    name="Min BM Nodal Dual",
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
                    name="Max BM Nodal Dual",
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
        title_text=(
            "PyPSA-GB Market Simulation Dashboard<br><sup>BM nodal values are optimisation duals, not imbalance prices."
            + (" Short-run diagnostic: do not annualise this report." if accounting and accounting.get("is_short_run") else "")
            + "</sup>"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
        template="plotly_white",
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Dashboard saved to {output_path}")


def _snapshot_hours(index):
    '''Return duration in hours for each reported snapshot.'''
    hours = pd.Series(1.0, index=index, dtype=float)
    if len(index) > 1:
        duration = pd.Index(index).to_series().diff().dt.total_seconds().median()
        if pd.notna(duration) and duration > 0:
            hours[:] = float(duration) / 3600.0
    return hours


def _foreign_market_price(network, bus, role, snapshots):
    '''Return the foreign counterparty marginal price for one market role.'''
    generators = network.generators[network.generators['bus'] == bus]
    if 'foreign_market_role' in generators.columns:
        selected = generators[generators['foreign_market_role'] == role]
    else:
        prefix = 'EU_supply_' if role == 'import_supply' else 'EU_export_sink_'
        selected = generators[generators.index.str.startswith(prefix)]
    if selected.empty:
        return pd.Series(np.nan, index=snapshots)
    name = selected.index[0]
    prices = pd.Series(
        float(selected.at[name, 'marginal_cost']), index=snapshots, dtype=float
    )
    time_varying = network.generators_t.get('marginal_cost', pd.DataFrame())
    if name in time_varying.columns:
        prices.update(time_varying[name].reindex(snapshots).dropna())
    return prices

def _direct_link_market_price(network, link_name, direction, snapshots):
    """Return the foreign price encoded in a directly priced link."""
    costs = pd.Series(
        float(network.links.at[link_name, "marginal_cost"]),
        index=snapshots,
        dtype=float,
    )
    time_varying = network.links_t.get("marginal_cost", pd.DataFrame())
    if link_name in time_varying.columns:
        costs.update(time_varying[link_name].reindex(snapshots).dropna())
    if direction == "export":
        efficiency = float(network.links.at[link_name, "efficiency"])
        return -costs / efficiency
    return costs


def _calculate_signed_interconnector_metrics(network, link_dispatch, wholesale_prices):
    """Report one signed physical Link per interconnector cable."""
    links = get_piecewise_interconnectors(network)
    snapshots = pd.Index(link_dispatch.index)
    hours = _snapshot_hours(snapshots)
    gb_prices = pd.Series(wholesale_prices, index=snapshots).reindex(snapshots)
    p_max_pu = network.get_switchable_as_dense("Link", "p_max_pu", snapshots)
    # What the LP charged each cable, tranche share included (item IC1).
    signed = [name for name in links.values() if name in link_dispatch]
    trade_cost = interconnector_trade_cost(
        link_dispatch.reindex(index=snapshots, columns=signed).fillna(0.0), *link_price_tables(network)
    )
    rows = []
    for cable, link_name in links.items():
        if link_name not in link_dispatch:
            continue
        flow = link_dispatch[link_name].reindex(snapshots).fillna(0.0)
        imported = flow.clip(lower=0.0)
        exported = (-flow).clip(lower=0.0)
        efficiency = float(network.links.at[link_name, "efficiency"])
        import_delivered = imported * efficiency
        # PyPSA books a link as p1 = -efficiency * p0 in either direction, so on
        # an export (p0 < 0, bus1 = GB) GB gives up efficiency * |p0| and the far
        # end receives |p0|. The rent values exports at that GB-side flow (item
        # B9, 26 Sep 2026; valuing them at |p0| understated it by about GBP13m on
        # the CP30 year).
        export_gb_side = exported * efficiency
        capacity = float(network.links.at[link_name, "p_nom"])
        available = capacity * p_max_pu[link_name]
        # GB receives the delivered import and gives up the GB-side export; the
        # foreign side is what the LP charged for the trade. Methodology note
        # (item B9, reported, not changed): the tranche part of trade_cost is the
        # area under the stepped supply curve, what the LP charges, not a uniform
        # clearing price on the whole flow, so a coupled-market rent would be lower.
        rent = (gb_prices * (import_delivered - export_gb_side) - trade_cost[link_name]) * hours
        rows.append({
            "interconnector": cable,
            "counterparty_country": network.links.at[link_name, "counterparty_country"],
            "capacity_MW": capacity,
            "gross_import_MWh": float((import_delivered * hours).sum()),
            # |p0| on exports: what the far end receives (kept for continuity).
            "gross_export_MWh": float((exported * hours).sum()),
            "gross_export_gb_side_MWh": float((export_gb_side * hours).sum()),
            "net_import_MWh": float(((import_delivered - export_gb_side) * hours).sum()),
            # (1 - efficiency) x flow each way; on exports PyPSA books it at the far end.
            "losses_MWh": float(((imported - import_delivered + exported - export_gb_side) * hours).sum()),
            "mean_utilization": float(flow.abs().div(available.replace(0.0, np.nan)).mean()),
            "max_utilization": float(flow.abs().div(available.replace(0.0, np.nan)).max()),
            "simultaneous_counterflow_hours": 0.0,
            "estimated_congestion_rent_GBP": float(rent.sum()),
        })
    return pd.DataFrame(rows)


def _calculate_signed_retrading_metrics(network, day_ahead, balancing):
    """Compare day-ahead and balancing schedules for signed cables.

    ``network`` must be the balancing network: the bid/ask valuation reads the
    foreign prices that the balancing LP itself optimised against.
    """
    links = get_piecewise_interconnectors(network)
    snapshots = pd.Index(balancing.index)
    hours = _snapshot_hours(snapshots)
    signed = [name for name in links.values() if name in day_ahead and name in balancing]
    # Both schedules are valued exactly as the balancing LP priced trade,
    # the volume tranches included (item IC1). Tranche cost is a function of the
    # whole market's flow, so each schedule is valued as a whole, then per cable.
    tables = link_price_tables(network)
    da_cost = interconnector_trade_cost(
        day_ahead.reindex(index=snapshots, columns=signed).fillna(0.0), *tables).mul(hours, axis=0).sum()
    bm_cost = interconnector_trade_cost(
        balancing.reindex(index=snapshots, columns=signed).fillna(0.0), *tables).mul(hours, axis=0).sum()
    # The counter-trade premium the balancing stage paid on each MWh it moved a
    # cable off its day-ahead schedule (interconnector_counter_trade.py). It is a
    # payment to traders and foreign operators, so it belongs to re-trading.
    premium = counter_trade_premium_cost(
        balancing.reindex(index=snapshots, columns=signed).fillna(0.0),
        day_ahead.reindex(index=snapshots, columns=signed).fillna(0.0),
        tables[0], tables[1],
    ).mul(hours, axis=0).sum()
    rows = []
    for cable, link_name in links.items():
        if link_name not in signed:
            continue
        da = day_ahead[link_name].reindex(snapshots).fillna(0.0)
        bm = balancing[link_name].reindex(snapshots).fillna(0.0)
        da_trade_cost = float(da_cost[link_name])
        bm_trade_cost = float(bm_cost[link_name])
        premium_cost = float(premium.get(link_name, 0.0))
        rows.append({
            "interconnector": cable,
            "day_ahead_import_MWh": float((da.clip(lower=0.0) * hours).sum()),
            "balancing_import_MWh": float((bm.clip(lower=0.0) * hours).sum()),
            "day_ahead_export_MWh": float(((-da).clip(lower=0.0) * hours).sum()),
            "balancing_export_MWh": float(((-bm).clip(lower=0.0) * hours).sum()),

            "gross_schedule_adjustment_MWh": float(((bm - da).abs() * hours).sum()),
            "day_ahead_trade_cost_GBP": da_trade_cost,
            "balancing_trade_cost_GBP": bm_trade_cost,
            "counter_trade_premium_GBP": premium_cost,
            "retrading_cost_GBP": bm_trade_cost - da_trade_cost + premium_cost,
        })
    return pd.DataFrame(rows)

def calculate_interconnector_metrics(
    network,
    link_dispatch,
    wholesale_prices,
    logger,
    simultaneous_tolerance_mw=0.1,
):
    '''Report bidirectional energy, utilization and estimated rent per cable.'''
    if get_piecewise_interconnectors(network):
        return _calculate_signed_interconnector_metrics(network, link_dispatch, wholesale_prices)
    pairs = get_directional_interconnector_pairs(network)
    if not pairs:
        logger.info('No directional interconnector pairs available for reporting')
        return pd.DataFrame()
    snapshots = pd.Index(link_dispatch.index)
    hours = _snapshot_hours(snapshots)
    gb_prices = pd.Series(wholesale_prices, index=snapshots).reindex(snapshots)
    p_max_pu = network.get_switchable_as_dense('Link', 'p_max_pu', snapshots)
    rows = []
    for pair_id, directions in pairs.items():
        import_link = directions['import']
        export_link = directions['export']
        if import_link not in link_dispatch or export_link not in link_dispatch:
            continue
        import_sent = link_dispatch[import_link].reindex(snapshots).fillna(0.0).clip(lower=0.0)
        export_sent = link_dispatch[export_link].reindex(snapshots).fillna(0.0).clip(lower=0.0)
        import_eta = float(network.links.at[import_link, 'efficiency'])
        export_eta = float(network.links.at[export_link, 'efficiency'])
        import_delivered = import_sent * import_eta
        export_received = export_sent * export_eta
        capacity = min(
            float(network.links.at[import_link, 'p_nom']),
            float(network.links.at[export_link, 'p_nom']),
        )
        availability = p_max_pu[[import_link, export_link]].min(axis=1)
        available_capacity = capacity * availability
        utilization = (import_sent + export_sent).div(
            available_capacity.replace(0.0, np.nan)
        )
        pricing_model = str(
            network.links.at[import_link, 'foreign_market_pricing']
            if 'foreign_market_pricing' in network.links.columns
            else ''
        ).lower()
        if pricing_model == 'direct_link':
            import_price = _direct_link_market_price(
                network, import_link, 'import', snapshots
            )
            export_price = _direct_link_market_price(
                network, export_link, 'export', snapshots
            )
        else:
            foreign_bus = network.links.at[import_link, 'bus0']
            import_price = _foreign_market_price(
                network, foreign_bus, 'import_supply', snapshots
            )
            export_price = _foreign_market_price(
                network, foreign_bus, 'export_demand', snapshots
            )
        import_rent = (
            gb_prices * import_delivered - import_price * import_sent
        ) * hours
        export_rent = (
            export_price * export_received - gb_prices * export_sent
        ) * hours
        simultaneous = (
            (import_sent > simultaneous_tolerance_mw)
            & (export_sent > simultaneous_tolerance_mw)
        )
        link = network.links.loc[import_link]
        rows.append({
            'interconnector': pair_id,
            'counterparty_country': link.get('counterparty_country', 'Unknown'),
            'capacity_MW': capacity,
            'gross_import_MWh': float((import_delivered * hours).sum()),
            'gross_export_MWh': float((export_sent * hours).sum()),
            'net_import_MWh': float(
                ((import_delivered - export_sent) * hours).sum()
            ),
            'losses_MWh': float(
                ((import_sent - import_delivered + export_sent - export_received)
                 * hours).sum()
            ),
            'mean_utilization': float(utilization.mean()),
            'max_utilization': float(utilization.max()),
            'simultaneous_counterflow_hours': float(hours[simultaneous].sum()),
            'estimated_congestion_rent_GBP': float(
                import_rent.sum() + export_rent.sum()
            ),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        counterflow = result['simultaneous_counterflow_hours'].sum()
        if counterflow > 0:
            direct_pricing = (
                'foreign_market_pricing' in network.links.columns
                and network.links.loc[
                    [directions['import'] for directions in pairs.values()],
                    'foreign_market_pricing',
                ].astype(str).str.lower().eq('direct_link').any()
            )
            message = (
                f'Directional interconnectors show {counterflow:.1f} material '
                f'counterflow hours above {simultaneous_tolerance_mw:.3f} MW'
            )
            if direct_pricing:
                raise ValueError(message)
            logger.warning(message)
    return result


def calculate_interconnector_retrading_metrics(
    network, wholesale_link_dispatch, balancing_link_dispatch
) -> pd.DataFrame:
    """Measure physical BM deviation from each day-ahead cable schedule."""
    if get_piecewise_interconnectors(network):
        return _calculate_signed_retrading_metrics(network, wholesale_link_dispatch, balancing_link_dispatch)
    pairs = get_directional_interconnector_pairs(network)
    if not pairs or balancing_link_dispatch is None or balancing_link_dispatch.empty:
        return pd.DataFrame()
    snapshots = pd.Index(balancing_link_dispatch.index)
    hours = _snapshot_hours(snapshots)
    premium = counter_trade_premium_cost(
        balancing_link_dispatch.reindex(snapshots).fillna(0.0),
        wholesale_link_dispatch.reindex(snapshots).fillna(0.0),
        network.links,
    ).mul(hours, axis=0).sum()
    rows = []
    for pair_id, directions in pairs.items():
        import_link = directions["import"]
        export_link = directions["export"]
        required = [import_link, export_link]
        if any(name not in wholesale_link_dispatch or name not in balancing_link_dispatch for name in required):
            continue
        pair_premium = {}
        if premium.get(import_link, 0.0) or premium.get(export_link, 0.0):
            # Only the counter-trade premium is valued here: directional cables
            # price trade on their foreign generators, not on the links.
            pair_premium = {
                "counter_trade_premium_GBP": float(premium.get(import_link, 0.0) + premium.get(export_link, 0.0)),
                "retrading_cost_GBP": float(premium.get(import_link, 0.0) + premium.get(export_link, 0.0)),
            }
        da_import = wholesale_link_dispatch[import_link].reindex(snapshots).fillna(0.0).clip(lower=0.0)
        da_export = wholesale_link_dispatch[export_link].reindex(snapshots).fillna(0.0).clip(lower=0.0)
        bm_import = balancing_link_dispatch[import_link].reindex(snapshots).fillna(0.0).clip(lower=0.0)
        bm_export = balancing_link_dispatch[export_link].reindex(snapshots).fillna(0.0).clip(lower=0.0)
        rows.append({
            "interconnector": pair_id,
            "day_ahead_import_MWh": float((da_import * hours).sum()),
            "balancing_import_MWh": float((bm_import * hours).sum()),
            "day_ahead_export_MWh": float((da_export * hours).sum()),
            "balancing_export_MWh": float((bm_export * hours).sum()),
            "gross_schedule_adjustment_MWh": float((
                (bm_import.sub(da_import).abs() + bm_export.sub(da_export).abs())
                .mul(hours).sum()
            )),
            **pair_premium,
        })
    return pd.DataFrame(rows)


def _demand_weighted_wholesale_price(wholesale_price, wholesale_network, logger):
    """Return sum(price * demand) / sum(demand) over hours, or None if unavailable.

    The simple mean of hourly prices over-weights low-demand (typically
    cheap) hours relative to what consumers actually pay; this weights every
    hour by the modelled total load in that hour.
    """
    try:
        loads_p_set = getattr(wholesale_network.loads_t, "p_set", None)
        if loads_p_set is None or loads_p_set.empty:
            logger.info(
                "Demand-weighted wholesale price skipped: no hourly load p_set "
                "in the wholesale network"
            )
            return None
        price = pd.to_numeric(pd.Series(wholesale_price), errors="coerce")
        # Consumer flexibility draws carved-out load when it charges (F5, 27 Oct 2026): demand too.
        from scripts.market.market_accounting import _network_flex_draw

        flex = _network_flex_draw(wholesale_network, loads_p_set.index)
        demand = (loads_p_set.sum(axis=1) + flex.sum(axis=1)).reindex(price.index)
        valid = price.notna() & demand.notna()
        total_demand = float(demand[valid].sum())
        if not valid.any() or total_demand <= 0.0:
            logger.info(
                "Demand-weighted wholesale price skipped: no overlapping "
                "positive demand for the price index"
            )
            return None
        return float((price[valid] * demand[valid]).sum() / total_demand)
    except Exception as exc:  # diagnostic only; never fail the summary for it
        logger.warning("Demand-weighted wholesale price skipped: %s", exc)
        return None


def create_market_summary(
    wholesale_network,
    balancing_network,
    redispatch_summary: pd.DataFrame,
    constraint_costs: pd.DataFrame,
    congestion: pd.DataFrame,
    price_comparison: pd.DataFrame,
    scenario_id: str,
    logger: logging.Logger,
    interconnector_metrics: pd.DataFrame = None,
    accounting: dict | None = None,
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

    if accounting is not None:
        summary["accounting"] = accounting

    # Wholesale results
    if hasattr(wholesale_network, "objective") and wholesale_network.objective is not None:
        summary["wholesale"] = {
            "total_cost": float(wholesale_network.objective),
        }

    # BM results. The solve stores its reported BM cost as the objective: every
    # objective-only regulariser stripped and, since item C9, net of the Stage 1
    # valuation of the cable schedule (so GB bids/offers + re-trading).
    if hasattr(balancing_network, "objective") and balancing_network.objective is not None:
        summary["balancing"] = {
            "total_redispatch_cost": float(balancing_network.objective),
            "total_redispatch_cost_basis": (
                "accepted GB bids and offers plus interconnector re-trading (trade-value change "
                "and counter-trade premium); objective-only terms and the Stage 1 cable valuation stripped"
            ),
        }

    # Internal HVDC tie-break (16 Oct 2026): objective only in both stages, so it is
    # out of every cost above and reported here on its own (internal_dc_tie_break.py).
    from scripts.market.internal_dc_tie_break import tie_break_summary
    tie_break = tie_break_summary(wholesale_network, balancing_network)
    if tie_break:
        summary["internal_dc_tie_break"] = tie_break
        for stage, record in tie_break.items():
            summary.setdefault(stage, {})["internal_dc_tie_break_cost_GBP"] = float(record.get("cost_GBP", 0.0))

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
            "mean_wholesale_price_basis": "simple mean of hourly copperplate prices",
        }
        demand_weighted = _demand_weighted_wholesale_price(
            price_comparison["wholesale_price"], wholesale_network, logger
        )
        if demand_weighted is not None:
            summary["prices"]["demand_weighted_wholesale_price"] = demand_weighted
        if "mean_nodal_price" in price_comparison.columns:
            mean_nodal_dual = float(
                price_comparison["mean_nodal_price"].mean()
            )
            # Keep the legacy key while exposing the correct interpretation.
            summary["prices"]["mean_nodal_price"] = mean_nodal_dual
            summary["prices"]["mean_balancing_nodal_dual"] = mean_nodal_dual
            summary["prices"]["balancing_nodal_dual_definition"] = (
                "Optimisation dual at demand buses; not an imbalance price."
            )
        if "nodal_spread" in price_comparison.columns:
            summary["prices"]["max_nodal_spread"] = float(
                price_comparison["nodal_spread"].max()
            )
            summary["prices"]["mean_nodal_spread"] = float(
                price_comparison["nodal_spread"].mean()
            )

    if interconnector_metrics is not None and not interconnector_metrics.empty:
        summary['interconnectors'] = {
            'gross_import_MWh': float(
                interconnector_metrics['gross_import_MWh'].sum()
            ),
            'gross_export_MWh': float(
                interconnector_metrics['gross_export_MWh'].sum()
            ),
            'net_import_MWh': float(
                interconnector_metrics['net_import_MWh'].sum()
            ),
            'losses_MWh': float(interconnector_metrics['losses_MWh'].sum()),
            'estimated_congestion_rent_GBP': float(
                interconnector_metrics['estimated_congestion_rent_GBP'].sum()
            ),
            'simultaneous_counterflow_hours': float(
                interconnector_metrics[
                    'simultaneous_counterflow_hours'
                ].sum()
            ),
            'by_cable': interconnector_metrics.to_dict(orient='records'),
        }
        if 'gross_schedule_adjustment_MWh' in interconnector_metrics.columns:
            summary['interconnectors']['gross_schedule_adjustment_MWh'] = float(
                interconnector_metrics['gross_schedule_adjustment_MWh'].sum()
            )
        if 'retrading_cost_GBP' in interconnector_metrics.columns:
            # Bid/ask valuation of the balancing-stage change in cable
            # schedules; part of the balancing objective, not of accepted
            # offers/bids.
            summary['interconnectors']['retrading_cost_GBP'] = float(
                interconnector_metrics['retrading_cost_GBP'].fillna(0.0).sum()
            )
        # Gross bid/ask valuations of each stage's schedule. The balancing LP
        # objective carries the full balancing_trade_cost_GBP (its native
        # marginal_cost*p link term is cancelled), but the reported BM cost
        # (balancing.total_redispatch_cost) is net of day_ahead_trade_cost_GBP
        # (item C9, 27 Sep 2026), so it minus accepted offers/bids reconciles to
        # retrading_cost_GBP, the increment over the day-ahead schedule plus the
        # counter-trade premium.
        for column in ('day_ahead_trade_cost_GBP', 'balancing_trade_cost_GBP',
                       'counter_trade_premium_GBP'):
            if column in interconnector_metrics.columns:
                summary['interconnectors'][column] = float(
                    interconnector_metrics[column].fillna(0.0).sum()
                )
    logger.info(f"Market summary: {json.dumps(summary, indent=2, default=str)}")
    return summary


def _add_revenue_tracking_to_summary(
    summary: dict,
    wholesale_network,
    balancing_network,
    scenario_config: dict,
    logger,
) -> dict:
    """Add CfD/ROC revenue tracking data to market summary if enabled."""
    rev_config = scenario_config.get('support', {}).get('revenue', {})
    if not rev_config.get('enabled', False):
        return summary

    sub_config = scenario_config.get('support', {}).get('labels', {})
    if not sub_config.get('enabled', False):
        logger.info("Revenue tracking enabled but support.labels disabled — skipping")
        return summary

    try:
        from scripts.market.revenue_tracking import compute_revenue_tracking
        load_buses = pd.Index(wholesale_network.loads.bus.dropna().unique())
        price_columns = [
            bus for bus in load_buses
            if bus in wholesale_network.buses_t.marginal_price.columns
        ]
        wholesale_reference_price = (
            wholesale_network.buses_t.marginal_price[price_columns].mean(axis=1)
            if price_columns else None
        )
        results = compute_revenue_tracking(
            balancing_network,
            scenario_config,
            logger,
            wholesale_reference_price=wholesale_reference_price,
        )
        summary['revenue'] = results.get('summary', {})
        logger.info(f"Revenue tracking added to summary: {summary['revenue']}")
    except Exception as e:
        logger.warning(f"Revenue tracking failed: {e}")
        summary['revenue'] = {'error': str(e)}

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
        try:
            congestion = pd.read_csv(snakemake.input.congestion_csv)
        except pd.errors.EmptyDataError:
            logger.warning("Congestion CSV is empty — no congested elements found.")
            congestion = pd.DataFrame()
        price_comparison = pd.read_csv(
            snakemake.input.price_comparison_csv, index_col=0, parse_dates=True
        )

        # ── Create dashboard ─────────────────────────────────────────────
        accounting, accounting_report = create_market_accounting_report(
            balancing_network, redispatch_summary
        )
        for warning in accounting["warnings"]:
            logger.warning("MARKET ACCOUNTING: %s", warning)

        wholesale_links = pd.read_csv(
            snakemake.input.wholesale_links_csv,
            index_col=0,
            parse_dates=True,
        )
        wholesale_price = price_comparison['wholesale_price'].reindex(
            wholesale_links.index
        )
        interconnector_metrics = calculate_interconnector_metrics(
            wholesale_network,
            wholesale_links,
            wholesale_price,
            logger,
        )
        retrading_metrics = calculate_interconnector_retrading_metrics(
            balancing_network, wholesale_links, balancing_network.links_t.p0
        )
        if not retrading_metrics.empty:
            interconnector_metrics = interconnector_metrics.merge(
                retrading_metrics, on="interconnector", how="left"
            )

        # The interconnector re-trading cost sits in the balancing objective
        # but not in accepted offers/bids, so attach it to the accounting block
        # before the accounting table is written.
        retrading_cost_total = None
        if (
            not retrading_metrics.empty
            and "retrading_cost_GBP" in retrading_metrics.columns
        ):
            retrading_cost_total = float(retrading_metrics["retrading_cost_GBP"].sum())
            premium_total = (
                float(retrading_metrics["counter_trade_premium_GBP"].fillna(0.0).sum())
                if "counter_trade_premium_GBP" in retrading_metrics.columns else 0.0
            )
            logger.info(
                "Interconnector re-trading cost (bid/ask valuation of BM minus "
                "day-ahead schedules, plus counter-trade premium £%.0f): £%.0f",
                premium_total,
                retrading_cost_total,
            )
        accounting_report = attach_interconnector_retrading(
            accounting, accounting_report, retrading_cost_total
        )
        Path(snakemake.output.accounting_csv).parent.mkdir(parents=True, exist_ok=True)
        accounting_report.to_csv(snakemake.output.accounting_csv, index=False)

        Path(snakemake.output.interconnector_flows_csv).parent.mkdir(
            parents=True, exist_ok=True
        )
        interconnector_metrics.to_csv(
            snakemake.output.interconnector_flows_csv, index=False
        )

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
            accounting=accounting,
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
            interconnector_metrics=interconnector_metrics,
            accounting=accounting,

        )

        # ── Add revenue tracking if enabled ──────────────────────────────
        summary = _add_revenue_tracking_to_summary(
            summary, wholesale_network, balancing_network, scenario_config, logger
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
