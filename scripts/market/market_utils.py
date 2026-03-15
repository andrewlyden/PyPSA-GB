"""
Market Simulation Utilities for PyPSA-GB

Shared helper functions for the two-stage market dispatch simulation:
  - Bid/offer price calculation from marginal costs
  - Wholesale position extraction from solved networks
  - Redispatch volume and cost computation
  - Congestion identification on lines/transformers
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional


def calculate_bid_offer_prices(
    network,
    market_config: dict,
    logger: logging.Logger,
    scenario_id: str = "",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate bid and offer prices for generators and storage units.

    Offer price (to increase output) = marginal_cost * (1 + offer_markup)
    Bid price   (to decrease output) = marginal_cost * (1 - bid_discount)

    For renewables with zero marginal cost, a small absolute floor is used
    to ensure they are still economically curtailable in the BM.

    Parameters
    ----------
    network : pypsa.Network
        Network with generators and storage units (marginal_cost must be set).
    market_config : dict
        Market configuration dict containing 'balancing' sub-dict with
        'default_offer_markup', 'default_bid_discount', 'carrier_overrides'.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    gen_offer_prices : pd.Series
        Offer prices per generator (£/MWh), indexed by generator name.
    gen_bid_prices : pd.Series
        Bid prices per generator (£/MWh), indexed by generator name.
    su_offer_prices : pd.Series
        Offer prices per storage unit (£/MWh), indexed by storage unit name.
    su_bid_prices : pd.Series
        Bid prices per storage unit (£/MWh), indexed by storage unit name.
    """
    balancing = market_config.get("balancing", {})
    default_offer_markup = balancing.get("default_offer_markup", 0.10)
    default_bid_discount = balancing.get("default_bid_discount", 0.10)
    carrier_overrides = balancing.get("carrier_overrides", {})
    bid_offer_source = balancing.get("bid_offer_source", "derived")

    # Minimum absolute bid price for zero-marginal-cost generators (£/MWh)
    # Ensures curtailment has a small positive cost in the BM objective
    MIN_BID_FLOOR = balancing.get("min_bid_floor", 0.50)

    # ── Source: ELEXON ────────────────────────────────────────────────────────
    if bid_offer_source == "elexon":
        # Guard: ELEXON data only exists for historical periods (≤2024)
        snap_year = network.snapshots[0].year if len(network.snapshots) > 0 else None
        if snap_year and snap_year > 2024:
            raise ValueError(
                f"bid_offer_source='elexon' is not valid for future scenarios "
                f"(network year {snap_year} > 2024). Use 'derived' or 'csv' instead."
            )
        gen_offer, gen_bid, su_offer, su_bid = _load_elexon_bid_offer(
            network, balancing, default_offer_markup, default_bid_discount,
            carrier_overrides, MIN_BID_FLOOR, logger,
            scenario_id=scenario_id,
        )
        return gen_offer, gen_bid, su_offer, su_bid

    # ── Source: CSV ───────────────────────────────────────────────────────────
    if bid_offer_source == "csv":
        gen_offer, gen_bid, su_offer, su_bid = _load_csv_bid_offer(
            network, balancing, MIN_BID_FLOOR, logger,
        )
        return gen_offer, gen_bid, su_offer, su_bid

    # ── Source: derived (default) ─────────────────────────────────────────────
    if bid_offer_source != "derived":
        raise ValueError(
            f"Unknown bid_offer_source='{bid_offer_source}'. "
            "Supported: 'derived', 'elexon', 'csv'."
        )

    def _compute_prices(components: pd.DataFrame, component_type: str):
        """Compute offer/bid prices for a DataFrame of generators or storage units."""
        offer_prices = pd.Series(index=components.index, dtype=float)
        bid_prices = pd.Series(index=components.index, dtype=float)

        for idx in components.index:
            carrier = components.loc[idx, "carrier"]
            mc = components.loc[idx, "marginal_cost"]

            # Look up carrier-specific overrides
            overrides = carrier_overrides.get(carrier, {})
            offer_markup = overrides.get("offer_markup", default_offer_markup)
            bid_discount = overrides.get("bid_discount", default_bid_discount)

            offer_prices[idx] = mc * (1.0 + offer_markup)
            bid_prices[idx] = max(mc * (1.0 - bid_discount), MIN_BID_FLOOR)

        return offer_prices, bid_prices

    # Generators
    gen_offer, gen_bid = _compute_prices(network.generators, "generator")
    logger.info(
        f"Bid/offer prices calculated for {len(gen_offer)} generators "
        f"(offer range: £{gen_offer.min():.2f}-£{gen_offer.max():.2f}/MWh, "
        f"bid range: £{gen_bid.min():.2f}-£{gen_bid.max():.2f}/MWh)"
    )

    # Storage units
    if len(network.storage_units) > 0:
        su_offer, su_bid = _compute_prices(network.storage_units, "storage_unit")
        logger.info(
            f"Bid/offer prices calculated for {len(su_offer)} storage units "
            f"(offer range: £{su_offer.min():.2f}-£{su_offer.max():.2f}/MWh, "
            f"bid range: £{su_bid.min():.2f}-£{su_bid.max():.2f}/MWh)"
        )
    else:
        su_offer = pd.Series(dtype=float)
        su_bid = pd.Series(dtype=float)

    return gen_offer, gen_bid, su_offer, su_bid


def _load_elexon_bid_offer(
    network, balancing: dict,
    default_offer_markup: float, default_bid_discount: float,
    carrier_overrides: dict, min_bid_floor: float,
    logger: logging.Logger,
    scenario_id: str = "",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load bid/offer prices from pre-fetched ELEXON BMRS data.

    Reads per-BMU bid/offer CSVs, maps BMU IDs to PyPSA generator names,
    and fills unmatched generators using carrier-group averages from the
    ELEXON dataset. Storage units fall back to derived pricing.

    Parameters
    ----------
    network : pypsa.Network
    balancing : dict
        Balancing config with 'elexon' sub-dict.
    default_offer_markup, default_bid_discount : float
        Fallback markups for derived pricing.
    carrier_overrides : dict
    min_bid_floor : float
    logger : logging.Logger

    Returns
    -------
    gen_offer, gen_bid, su_offer, su_bid : pd.Series
    """
    elexon_cfg = balancing.get("elexon", {})
    data_dir_raw = elexon_cfg.get("data_dir", "resources/market/elexon")
    # Expand {scenario} template if present
    if "{scenario}" in data_dir_raw and scenario_id:
        data_dir_raw = data_dir_raw.replace("{scenario}", scenario_id)
    data_dir = Path(data_dir_raw)
    bmu_mapping_path = Path(elexon_cfg.get("bmu_mapping", "data/generators/bmus_prepared.csv"))
    fallback = elexon_cfg.get("fallback", "derived")

    # Load BMU → generator mapping
    if not bmu_mapping_path.exists():
        raise FileNotFoundError(
            f"BMU mapping file not found: {bmu_mapping_path}. "
            "Run the ELEXON data retrieval rule first."
        )
    bmu_map = pd.read_csv(bmu_mapping_path)
    # Expect columns: bmu_id, generator_name (at minimum)
    bmu_to_gen = dict(zip(bmu_map["bmu_id"], bmu_map["generator_name"]))
    logger.info(f"Loaded BMU mapping: {len(bmu_to_gen)} BMU→generator entries")

    # Load ELEXON offer/bid CSVs
    offer_file = data_dir / "elexon_offers.csv"
    bid_file = data_dir / "elexon_bids.csv"
    for f in [offer_file, bid_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"ELEXON data file not found: {f}. "
                "Run the retrieve_elexon_market_data rule first."
            )

    elexon_offers = pd.read_csv(offer_file, index_col=0)
    elexon_bids = pd.read_csv(bid_file, index_col=0)
    logger.info(f"Loaded ELEXON offers: {elexon_offers.shape}, bids: {elexon_bids.shape}")

    # Map BMU columns to generator names
    elexon_offers = elexon_offers.rename(columns=bmu_to_gen)
    elexon_bids = elexon_bids.rename(columns=bmu_to_gen)

    # Build per-generator average offer/bid prices (mean across snapshots)
    gen_offer = pd.Series(index=network.generators.index, dtype=float)
    gen_bid = pd.Series(index=network.generators.index, dtype=float)

    matched = 0
    for gen_name in network.generators.index:
        if gen_name in elexon_offers.columns:
            gen_offer[gen_name] = elexon_offers[gen_name].mean()
            gen_bid[gen_name] = max(elexon_bids[gen_name].mean(), min_bid_floor)
            matched += 1

    # Fill unmatched generators
    unmatched = gen_offer.isna()
    if unmatched.any():
        if fallback == "derived":
            # Use carrier-group average from matched ELEXON data, else formula
            for gen_name in network.generators.index[unmatched]:
                carrier = network.generators.loc[gen_name, "carrier"]
                mc = network.generators.loc[gen_name, "marginal_cost"]
                overrides = carrier_overrides.get(carrier, {})
                offer_markup = overrides.get("offer_markup", default_offer_markup)
                bid_discount = overrides.get("bid_discount", default_bid_discount)
                gen_offer[gen_name] = mc * (1.0 + offer_markup)
                gen_bid[gen_name] = max(mc * (1.0 - bid_discount), min_bid_floor)
        else:
            gen_offer[unmatched] = 0.0
            gen_bid[unmatched] = min_bid_floor

    logger.info(
        f"ELEXON bid/offer: {matched}/{len(network.generators)} generators matched, "
        f"{unmatched.sum()} filled via {fallback}"
    )

    # Storage units — always derived (no ELEXON BMU mapping for storage)
    su_offer = pd.Series(dtype=float)
    su_bid = pd.Series(dtype=float)
    if len(network.storage_units) > 0:
        su_offer = pd.Series(index=network.storage_units.index, dtype=float)
        su_bid = pd.Series(index=network.storage_units.index, dtype=float)
        for idx in network.storage_units.index:
            carrier = network.storage_units.loc[idx, "carrier"]
            mc = network.storage_units.loc[idx, "marginal_cost"]
            overrides = carrier_overrides.get(carrier, {})
            offer_markup = overrides.get("offer_markup", default_offer_markup)
            bid_discount = overrides.get("bid_discount", default_bid_discount)
            su_offer[idx] = mc * (1.0 + offer_markup)
            su_bid[idx] = max(mc * (1.0 - bid_discount), min_bid_floor)

    return gen_offer, gen_bid, su_offer, su_bid


def _load_csv_bid_offer(
    network, balancing: dict, min_bid_floor: float,
    logger: logging.Logger,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load bid/offer prices from user-supplied CSV files.

    CSV format: rows = snapshots (or a single row for static prices),
    columns = generator/storage unit names. Values in £/MWh.
    If a single row, prices are broadcast to all snapshots.

    Parameters
    ----------
    network : pypsa.Network
    balancing : dict
        Balancing config with 'csv' sub-dict containing 'offer_file', 'bid_file'.
    min_bid_floor : float
    logger : logging.Logger

    Returns
    -------
    gen_offer, gen_bid, su_offer, su_bid : pd.Series
    """
    csv_cfg = balancing.get("csv", {})
    offer_path = csv_cfg.get("offer_file")
    bid_path = csv_cfg.get("bid_file")

    if not offer_path or not bid_path:
        raise ValueError(
            "bid_offer_source='csv' requires both 'csv.offer_file' and 'csv.bid_file' "
            "to be set in market.balancing config."
        )

    offer_df = pd.read_csv(offer_path, index_col=0)
    bid_df = pd.read_csv(bid_path, index_col=0)
    logger.info(f"Loaded CSV offers: {offer_df.shape}, bids: {bid_df.shape}")

    # Use mean across snapshots if time-varying, or single row
    def _extract_prices(df, components, label):
        prices = pd.Series(index=components.index, dtype=float, data=0.0)
        for name in components.index:
            if name in df.columns:
                prices[name] = df[name].mean()
        return prices

    gen_offer = _extract_prices(offer_df, network.generators, "generator")
    gen_bid = _extract_prices(bid_df, network.generators, "generator").clip(lower=min_bid_floor)

    su_offer = pd.Series(dtype=float)
    su_bid = pd.Series(dtype=float)
    if len(network.storage_units) > 0:
        su_offer = _extract_prices(offer_df, network.storage_units, "storage_unit")
        su_bid = _extract_prices(bid_df, network.storage_units, "storage_unit").clip(lower=min_bid_floor)

    logger.info(
        f"CSV bid/offer loaded: {len(gen_offer)} generators, {len(su_offer)} storage units"
    )

    return gen_offer, gen_bid, su_offer, su_bid


def extract_wholesale_positions(
    network,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract dispatch positions from a solved wholesale (copperplate) network.

    Parameters
    ----------
    network : pypsa.Network
        Solved wholesale network (after network.optimize()).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    gen_dispatch : pd.DataFrame
        Generator dispatch (MW) indexed by snapshot, columns = generator names.
    su_dispatch : pd.DataFrame
        Storage unit dispatch (MW) indexed by snapshot, columns = storage unit names.
    link_dispatch : pd.DataFrame
        Link dispatch at bus0 (MW) indexed by snapshot, columns = link names.
    """
    gen_dispatch = network.generators_t.p.copy()
    logger.info(f"Extracted wholesale generator dispatch: {gen_dispatch.shape}")

    if len(network.storage_units_t.p) > 0:
        su_dispatch = network.storage_units_t.p.copy()
        logger.info(f"Extracted wholesale storage dispatch: {su_dispatch.shape}")
    else:
        su_dispatch = pd.DataFrame(index=network.snapshots)

    if len(network.links_t.p0) > 0:
        link_dispatch = network.links_t.p0.copy()
        logger.info(f"Extracted wholesale link dispatch: {link_dispatch.shape}")
    else:
        link_dispatch = pd.DataFrame(index=network.snapshots)

    return gen_dispatch, su_dispatch, link_dispatch


def compute_redispatch_volumes(
    wholesale_gen: pd.DataFrame,
    physical_gen: pd.DataFrame,
    wholesale_su: pd.DataFrame,
    physical_su: pd.DataFrame,
    gen_offer_prices: pd.Series,
    gen_bid_prices: pd.Series,
    su_offer_prices: pd.Series,
    su_bid_prices: pd.Series,
    network,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-asset redispatch volumes and costs.

    Increase (offer accepted) = physical - wholesale > 0  → paid offer_price
    Decrease (bid accepted)   = wholesale - physical > 0  → paid bid_price

    Parameters
    ----------
    wholesale_gen, physical_gen : pd.DataFrame
        Generator dispatch (MW), snapshots × generators.
    wholesale_su, physical_su : pd.DataFrame
        Storage unit dispatch (MW), snapshots × storage units.
    gen_offer_prices, gen_bid_prices : pd.Series
        Per-generator offer/bid prices (£/MWh).
    su_offer_prices, su_bid_prices : pd.Series
        Per-storage-unit offer/bid prices (£/MWh).
    network : pypsa.Network
        Solved network (for carrier lookup).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    gen_summary : pd.DataFrame
        Per-generator summary with columns: carrier, increase_MWh, decrease_MWh,
        offer_cost, bid_cost, net_cost.
    su_summary : pd.DataFrame
        Per-storage-unit summary with same columns.
    """

    def _summarize(wholesale_df, physical_df, offer_prices, bid_prices, components, component_type):
        records = []
        # Align columns
        common = wholesale_df.columns.intersection(physical_df.columns)
        for name in common:
            diff = physical_df[name] - wholesale_df[name]  # MW per snapshot
            increase = diff.clip(lower=0).sum()  # MWh (hourly snapshots)
            decrease = (-diff).clip(lower=0).sum()

            offer_cost = increase * offer_prices.get(name, 0.0)
            bid_cost = decrease * bid_prices.get(name, 0.0)

            records.append({
                "component": name,
                "type": component_type,
                "carrier": components.loc[name, "carrier"] if name in components.index else "unknown",
                "increase_MWh": increase,
                "decrease_MWh": decrease,
                "offer_cost": offer_cost,
                "bid_cost": bid_cost,
                "net_cost": offer_cost + bid_cost,
            })
        return pd.DataFrame(records)

    gen_summary = _summarize(
        wholesale_gen, physical_gen, gen_offer_prices, gen_bid_prices,
        network.generators, "generator"
    )
    logger.info(
        f"Generator redispatch: "
        f"total increase={gen_summary['increase_MWh'].sum():,.0f} MWh, "
        f"total decrease={gen_summary['decrease_MWh'].sum():,.0f} MWh, "
        f"total cost=£{gen_summary['net_cost'].sum():,.0f}"
    )

    if len(wholesale_su.columns) > 0 and len(physical_su.columns) > 0:
        su_summary = _summarize(
            wholesale_su, physical_su, su_offer_prices, su_bid_prices,
            network.storage_units, "storage_unit"
        )
        logger.info(
            f"Storage redispatch: "
            f"total increase={su_summary['increase_MWh'].sum():,.0f} MWh, "
            f"total decrease={su_summary['decrease_MWh'].sum():,.0f} MWh, "
            f"total cost=£{su_summary['net_cost'].sum():,.0f}"
        )
    else:
        su_summary = pd.DataFrame()

    return gen_summary, su_summary


def identify_congested_boundaries(
    network,
    threshold: float = 0.95,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Identify congested lines and transformers in a solved network.

    A line/transformer is considered congested at a snapshot if its loading
    exceeds ``threshold`` fraction of its ``s_nom``.

    Parameters
    ----------
    network : pypsa.Network
        Solved network with lines_t.p0 populated.
    threshold : float
        Fraction of s_nom above which a component is congested (default 0.95).
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    congestion_df : pd.DataFrame
        Columns: component, type, s_nom, max_loading_fraction,
        hours_congested, mean_loading_fraction.
    """
    records = []

    # Lines
    if len(network.lines_t.p0) > 0 and len(network.lines) > 0:
        for line in network.lines.index:
            if line not in network.lines_t.p0.columns:
                continue
            s_nom = network.lines.loc[line, "s_nom"]
            if s_nom <= 0:
                continue
            loading = network.lines_t.p0[line].abs() / s_nom
            hours_congested = (loading >= threshold).sum()
            if hours_congested > 0:
                records.append({
                    "component": line,
                    "type": "line",
                    "s_nom_MVA": s_nom,
                    "max_loading_fraction": loading.max(),
                    "hours_congested": int(hours_congested),
                    "mean_loading_fraction": loading.mean(),
                })

    # Transformers
    if len(network.transformers_t.p0) > 0 and len(network.transformers) > 0:
        for trafo in network.transformers.index:
            if trafo not in network.transformers_t.p0.columns:
                continue
            s_nom = network.transformers.loc[trafo, "s_nom"]
            if s_nom <= 0:
                continue
            loading = network.transformers_t.p0[trafo].abs() / s_nom
            hours_congested = (loading >= threshold).sum()
            if hours_congested > 0:
                records.append({
                    "component": trafo,
                    "type": "transformer",
                    "s_nom_MVA": s_nom,
                    "max_loading_fraction": loading.max(),
                    "hours_congested": int(hours_congested),
                    "mean_loading_fraction": loading.mean(),
                })

    congestion_df = pd.DataFrame(records)
    if logger:
        if len(congestion_df) > 0:
            n_lines = (congestion_df["type"] == "line").sum()
            n_trafos = (congestion_df["type"] == "transformer").sum()
            logger.info(
                f"Congestion analysis: {n_lines} congested lines, "
                f"{n_trafos} congested transformers (threshold={threshold:.0%})"
            )
        else:
            logger.info(f"No congested components found (threshold={threshold:.0%})")

    return congestion_df
