"""
Revenue tracking for CfD and ROC subsidy schemes.

Post-solve module that calculates:
- CfD difference payments: (strike_price - wholesale_price) × dispatch × dt
- ROC income: ro_banding × roc_buyout_price × dispatch × dt

Input:
    - Solved network (with bus marginal prices from wholesale/balancing solve)
    - Generator attributes: support_type, ro_banding, cfd_round

Output:
    - Per-generator revenue breakdown CSV
    - System-level subsidy summary

Called after solve_wholesale or solve_balancing in the market pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network

logger = setup_logging("revenue_tracking")


def compute_cfd_payments(
    network,
    strike_prices: dict,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Compute CfD difference payments for generators with support_type == 'CfD'.

    CfD payment per timestep = (strike_price - reference_price) × dispatch_MW × dt_hours
    If reference_price > strike_price, the generator pays back the difference.

    Parameters
    ----------
    network : pypsa.Network
        Solved network with generators_t.p (dispatch) and buses_t.marginal_price.
    strike_prices : dict
        Mapping of CfD round / carrier to strike price (£/MWh).
        Keys can be: 'AR1_offshore', 'AR2_offshore', ..., 'future_offshore', etc.
    logger : Logger

    Returns
    -------
    DataFrame with columns: generator, carrier, bus, support_type, cfd_round,
        strike_price, total_dispatch_MWh, avg_reference_price, total_cfd_payment,
        net_cfd_cost (positive = cost to consumer, negative = generator pays back)
    """
    if 'support_type' not in network.generators.columns:
        logger.info("No support_type column — skipping CfD revenue tracking")
        return pd.DataFrame()

    cfd_gens = network.generators[network.generators['support_type'] == 'CfD']
    if cfd_gens.empty:
        logger.info("No CfD generators found")
        return pd.DataFrame()

    # Get snapshot weighting (hours per snapshot)
    if hasattr(network, 'snapshot_weightings') and 'generators' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['generators']
    elif hasattr(network, 'snapshot_weightings') and 'objective' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['objective']
    else:
        dt = pd.Series(1.0, index=network.snapshots)

    results = []
    for gen_name, gen in cfd_gens.iterrows():
        carrier = gen.get('carrier', '')
        bus = gen.get('bus', '')
        cfd_round = gen.get('cfd_round', '')

        # Look up strike price
        strike = _lookup_strike_price(carrier, cfd_round, strike_prices)
        if strike is None:
            logger.debug(f"No strike price for {gen_name} (carrier={carrier}, round={cfd_round})")
            continue

        # Get dispatch (MW) and reference price at generator bus
        if gen_name in network.generators_t.p.columns:
            dispatch = network.generators_t.p[gen_name]
        else:
            logger.debug(f"No dispatch data for {gen_name}")
            continue

        if bus in network.buses_t.marginal_price.columns:
            ref_price = network.buses_t.marginal_price[bus]
        else:
            ref_price = pd.Series(0.0, index=network.snapshots)

        # CfD payment per timestep: (strike - ref_price) × dispatch × dt
        payment = (strike - ref_price) * dispatch * dt
        total_dispatch = (dispatch * dt).sum()
        avg_ref_price = (ref_price * dispatch * dt).sum() / max(total_dispatch, 1e-6)

        results.append({
            'generator': gen_name,
            'carrier': carrier,
            'bus': bus,
            'support_type': 'CfD',
            'cfd_round': cfd_round,
            'strike_price': strike,
            'total_dispatch_MWh': total_dispatch,
            'avg_reference_price': avg_ref_price,
            'total_cfd_payment': payment.sum(),
            'net_cfd_cost': payment.sum(),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        total_cost = df['net_cfd_cost'].sum()
        total_mwh = df['total_dispatch_MWh'].sum()
        logger.info(f"CfD payments: {len(df)} generators, "
                     f"{total_mwh:,.0f} MWh, £{total_cost:,.0f} net cost")
    return df


def compute_roc_income(
    network,
    roc_buyout_price: float,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Compute ROC income for generators with support_type == 'ROC'.

    ROC income per timestep = ro_banding × roc_buyout_price × dispatch_MW × dt_hours

    Parameters
    ----------
    network : pypsa.Network
        Solved network.
    roc_buyout_price : float
        ROC buyout price for the modelled year (£/ROC).
    logger : Logger

    Returns
    -------
    DataFrame with columns: generator, carrier, bus, support_type, ro_banding,
        total_dispatch_MWh, roc_income_total
    """
    if 'support_type' not in network.generators.columns:
        logger.info("No support_type column — skipping ROC revenue tracking")
        return pd.DataFrame()

    roc_gens = network.generators[network.generators['support_type'] == 'ROC']
    if roc_gens.empty:
        logger.info("No ROC generators found")
        return pd.DataFrame()

    if hasattr(network, 'snapshot_weightings') and 'generators' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['generators']
    elif hasattr(network, 'snapshot_weightings') and 'objective' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['objective']
    else:
        dt = pd.Series(1.0, index=network.snapshots)

    results = []
    for gen_name, gen in roc_gens.iterrows():
        carrier = gen.get('carrier', '')
        bus = gen.get('bus', '')
        ro_banding = gen.get('ro_banding', 0.0)

        if not pd.notna(ro_banding) or ro_banding <= 0:
            continue

        if gen_name in network.generators_t.p.columns:
            dispatch = network.generators_t.p[gen_name]
        else:
            continue

        total_dispatch = (dispatch * dt).sum()
        roc_income = float(ro_banding) * roc_buyout_price * total_dispatch

        results.append({
            'generator': gen_name,
            'carrier': carrier,
            'bus': bus,
            'support_type': 'ROC',
            'ro_banding': ro_banding,
            'total_dispatch_MWh': total_dispatch,
            'roc_income_total': roc_income,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        total_income = df['roc_income_total'].sum()
        total_mwh = df['total_dispatch_MWh'].sum()
        logger.info(f"ROC income: {len(df)} generators, "
                     f"{total_mwh:,.0f} MWh, £{total_income:,.0f} total")
    return df


def _lookup_strike_price(carrier: str, cfd_round, strike_prices: dict):
    """Look up CfD strike price by round + carrier, with fallbacks."""
    if not strike_prices:
        return None

    # Try exact round+carrier match (e.g., "AR2_offshore")
    if pd.notna(cfd_round) and cfd_round:
        round_str = str(cfd_round).strip()
        # Map carrier to technology suffix
        tech_suffix = _carrier_to_tech_suffix(carrier)
        key = f"{round_str}_{tech_suffix}"
        if key in strike_prices:
            return strike_prices[key]
        # Try just the round
        if round_str in strike_prices:
            return strike_prices[round_str]

    # Fallback: try "future_{tech}" for FES generators
    tech_suffix = _carrier_to_tech_suffix(carrier)
    future_key = f"future_{tech_suffix}"
    if future_key in strike_prices:
        return strike_prices[future_key]

    return None


def _carrier_to_tech_suffix(carrier: str) -> str:
    """Map PyPSA carrier to CfD technology suffix."""
    mapping = {
        'wind_offshore': 'offshore',
        'wind_onshore': 'onshore',
        'solar_pv': 'solar',
        'nuclear': 'nuclear',
        'tidal_stream': 'tidal',
        'shoreline_wave': 'wave',
    }
    return mapping.get(carrier, carrier)


def compute_revenue_tracking(
    network,
    scenario_config: dict,
    logger: logging.Logger = logger,
) -> dict:
    """
    Run full revenue tracking computation.

    Parameters
    ----------
    network : pypsa.Network
        Solved network.
    scenario_config : dict
        Full scenario configuration.
    logger : Logger

    Returns
    -------
    dict with keys: 'cfd_df', 'roc_df', 'summary'
    """
    rev_config = scenario_config.get('market', {}).get('revenue_tracking', {})
    sub_config = scenario_config.get('subsidy_tracking', {})
    mc_sub = scenario_config.get('marginal_costs', {}).get('subsidies', {})

    results = {'cfd_df': pd.DataFrame(), 'roc_df': pd.DataFrame(), 'summary': {}}

    if rev_config.get('include_cfd', True):
        strike_prices = sub_config.get('cfd_strike_prices', {})
        results['cfd_df'] = compute_cfd_payments(network, strike_prices, logger)

    if rev_config.get('include_roc', True):
        modelled_year = scenario_config.get('modelled_year', 2024)
        buyout_prices = mc_sub.get('roc_buyout_prices', {})
        # Reuse the interpolation helper from apply_marginal_costs
        from scripts.generators.apply_marginal_costs import _get_roc_buyout_price
        roc_price = _get_roc_buyout_price(buyout_prices, modelled_year)
        results['roc_df'] = compute_roc_income(network, roc_price, logger)

    # Build summary
    summary = {}
    if not results['cfd_df'].empty:
        cfd = results['cfd_df']
        summary['cfd_total_cost'] = float(cfd['net_cfd_cost'].sum())
        summary['cfd_total_dispatch_MWh'] = float(cfd['total_dispatch_MWh'].sum())
        summary['cfd_generator_count'] = len(cfd)
    if not results['roc_df'].empty:
        roc = results['roc_df']
        summary['roc_total_income'] = float(roc['roc_income_total'].sum())
        summary['roc_total_dispatch_MWh'] = float(roc['total_dispatch_MWh'].sum())
        summary['roc_generator_count'] = len(roc)
    results['summary'] = summary

    return results
