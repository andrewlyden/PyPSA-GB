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


DEFAULT_STORAGE_BMU_PREFIXES = {
    "Pumped Storage Hydroelectricity": [
        "T_CRUA",
        "T_CRUAD",
        "T_FOYE",
        "T_FOYED",
        "T_DINO",
        "T_FFES",
        "T_FFESST",
    ],
    "pumped_hydro": [
        "T_CRUA",
        "T_CRUAD",
        "T_FOYE",
        "T_FOYED",
        "T_DINO",
        "T_FFES",
        "T_FFESST",
    ],
}


def _log_elexon_coverage_summary(
    network,
    matched_mask: pd.Series,
    fallback_mask: pd.Series,
    fallback_label: str,
    logger: logging.Logger,
) -> None:
    """Log how much of the fleet is priced by direct ELEXON matches vs fallback."""
    total = len(network.generators)
    matched = int(matched_mask.sum())
    fallback = int(fallback_mask.sum())
    if total == 0:
        return

    matched_pct = 100.0 * matched / total
    fallback_pct = 100.0 * fallback / total
    logger.info(
        f"ELEXON bid/offer: {matched}/{total} generators matched "
        f"({matched_pct:.1f}%), {fallback}/{total} filled via {fallback_label} "
        f"({fallback_pct:.1f}%)"
    )

    if matched_pct < 10.0:
        fallback_capacity = network.generators.loc[fallback_mask, "p_nom"].sum()
        logger.warning(
            "Low direct ELEXON pricing coverage: "
            f"{matched_pct:.1f}% of generators matched, "
            f"{fallback_capacity:,.0f} MW priced via fallback. "
            "BM redispatch will be dominated by fallback assumptions."
        )

    carrier_df = pd.DataFrame(
        {
            "carrier": network.generators["carrier"],
            "p_nom": network.generators["p_nom"].fillna(0.0),
            "matched": matched_mask.astype(int),
            "fallback": fallback_mask.astype(int),
        },
        index=network.generators.index,
    )
    carrier_summary = (
        carrier_df.groupby("carrier")
        .agg(
            generators=("carrier", "size"),
            p_nom_mw=("p_nom", "sum"),
            matched=("matched", "sum"),
            fallback=("fallback", "sum"),
        )
        .sort_values(["fallback", "p_nom_mw"], ascending=[False, False])
    )

    carrier_summary = carrier_summary[carrier_summary["fallback"] > 0].head(10)
    if not carrier_summary.empty:
        logger.info("Top fallback-priced carriers:")
        for carrier, row in carrier_summary.iterrows():
            logger.info(
                f"  {carrier}: fallback {int(row['fallback'])}/{int(row['generators'])} "
                f"generators, {row['p_nom_mw']:.0f} MW fleet, direct matches {int(row['matched'])}"
            )


def _compute_component_prices(
    components: pd.DataFrame,
    carrier_overrides: dict,
    default_offer_markup: float,
    default_bid_discount: float,
) -> Tuple[pd.Series, pd.Series]:
    """Compute static bid/offer prices for generators or storage units."""
    offer_prices = pd.Series(index=components.index, dtype=float)
    bid_prices = pd.Series(index=components.index, dtype=float)

    for idx in components.index:
        carrier = components.loc[idx, "carrier"]
        mc = components.loc[idx, "marginal_cost"]

        overrides = carrier_overrides.get(carrier, {})
        mode = overrides.get("mode", "markup")
        default_offer = mc * (1.0 + overrides.get("offer_markup", default_offer_markup))
        default_bid = mc * (1.0 - overrides.get("bid_discount", default_bid_discount))

        if mode == "absolute":
            offer_prices[idx] = overrides.get("offer_price", default_offer)
            bid_prices[idx] = overrides.get("bid_price", default_bid)
        else:
            offer_prices[idx] = default_offer
            bid_prices[idx] = default_bid

    return offer_prices, bid_prices


def _resolve_bid_offer_source(
    network,
    balancing: dict,
    logger: logging.Logger,
    scenario_id: str = "",
) -> str:
    """
    Resolve the effective bid/offer source.

    ``auto`` prefers ELEXON data for historical scenarios when the required
    files are already available, otherwise it falls back to derived pricing.
    """
    source = balancing.get("bid_offer_source", "auto")
    if source != "auto":
        return source

    snap_year = network.snapshots[0].year if len(network.snapshots) > 0 else None
    if snap_year is None or snap_year > 2024:
        logger.info(
            "bid_offer_source=auto: using derived pricing "
            "(future scenario or no snapshot year)"
        )
        return "derived"

    elexon_cfg = balancing.get("elexon", {})
    data_dir_raw = elexon_cfg.get("data_dir", "resources/market/{scenario}/elexon")
    if "{scenario}" in data_dir_raw and scenario_id:
        data_dir_raw = data_dir_raw.replace("{scenario}", scenario_id)
    data_dir = Path(data_dir_raw)
    bmu_mapping_raw = elexon_cfg.get(
        "bmu_mapping", "resources/generators/{scenario}_bmu_mapping.csv"
    )
    if "{scenario}" in bmu_mapping_raw and scenario_id:
        bmu_mapping_raw = bmu_mapping_raw.replace("{scenario}", scenario_id)
    bmu_mapping_path = Path(bmu_mapping_raw)
    offer_file = data_dir / "elexon_offers.csv"
    bid_file = data_dir / "elexon_bids.csv"

    if bmu_mapping_path.exists() and offer_file.exists() and bid_file.exists():
        logger.info("bid_offer_source=auto: using ELEXON bid/offer data")
        return "elexon"

    logger.info(
        "bid_offer_source=auto: ELEXON files not available, "
        "falling back to derived pricing"
    )
    return "derived"


def _price_ladders_enabled(balancing: dict) -> bool:
    """Return True when ELEXON price-ladder dispatch is enabled."""
    return bool(
        balancing.get("elexon", {})
        .get("price_ladders", {})
        .get("enabled", False)
    )


def _load_elexon_ladder_file(
    path: Path,
    bmu_to_gen: dict,
    generator_index: pd.Index,
    logger: logging.Logger,
    label: str,
) -> pd.DataFrame:
    """
    Load a long-format ELEXON ladder file and map BMUs to model generators.

    Output columns: snapshot, generator, block, price, volume_mw.
    If multiple BMUs map to the same generator, their blocks are stacked and
    re-ranked by price for each generator-hour.
    """
    cols = ["snapshot", "generator", "block", "price", "volume_mw"]
    if not path.exists():
        logger.warning(f"ELEXON {label} ladder file not found: {path}")
        return pd.DataFrame(columns=cols)

    ladders = pd.read_csv(path)
    if ladders.empty:
        return pd.DataFrame(columns=cols)

    required = {"datetime", "bmu_id", "price", "volume_mw"}
    missing = required.difference(ladders.columns)
    if missing:
        raise ValueError(
            f"ELEXON {label} ladder file {path} is missing columns: {sorted(missing)}"
        )

    ladders["snapshot"] = pd.to_datetime(ladders["datetime"])
    ladders["generator"] = ladders["bmu_id"].map(bmu_to_gen)
    ladders["price"] = pd.to_numeric(ladders["price"], errors="coerce")
    ladders["volume_mw"] = pd.to_numeric(ladders["volume_mw"], errors="coerce")
    ladders = ladders[
        ladders["generator"].isin(generator_index)
        & ladders["snapshot"].notna()
        & ladders["price"].notna()
        & ladders["volume_mw"].notna()
        & (ladders["volume_mw"] > 0)
    ].copy()
    if ladders.empty:
        logger.warning(f"ELEXON {label} ladders loaded but no rows mapped to generators")
        return pd.DataFrame(columns=cols)

    # Merge identical price blocks after BMU->generator mapping, then re-rank.
    ladders = (
        ladders.groupby(["snapshot", "generator", "price"], as_index=False)[
            "volume_mw"
        ]
        .sum()
        .sort_values(["snapshot", "generator", "price"])
    )
    ladders["block"] = (
        ladders.groupby(["snapshot", "generator"]).cumcount() + 1
    ).astype(int)

    logger.info(
        f"Loaded ELEXON {label} ladders: {len(ladders):,} blocks, "
        f"{ladders['generator'].nunique()} generators"
    )
    return ladders[cols]


def _resolve_component_participants(
    components: pd.DataFrame,
    participation_cfg: dict,
    matched_names: pd.Index | None,
    logger: logging.Logger,
    component_label: str,
) -> tuple[pd.Index, pd.Index, str, float, float]:
    """Resolve which components participate in BM and how others are handled."""
    cfg = participation_cfg or {}
    mode = str(cfg.get("mode", "all")).lower()
    behavior = str(cfg.get("behavior", "priced_out")).lower()
    penalty_offer = float(cfg.get("penalty_offer_price", 5000.0))
    penalty_bid = float(cfg.get("penalty_bid_price", 5000.0))

    if behavior not in {"priced_out", "fixed", "fallback_priced"}:
        raise ValueError(
            f"Unsupported {component_label} participation behavior='{behavior}'. "
            "Use 'priced_out', 'fixed', or 'fallback_priced'."
        )

    component_index = components.index
    component_set = set(component_index)
    matched_set = set((matched_names if matched_names is not None else pd.Index([])).tolist())

    if mode == "all":
        participants = component_set.copy()
    elif mode == "none":
        participants = set()
    elif mode == "elexon_mapped":
        participants = component_set.intersection(matched_set)
        if len(participants) == 0 and len(component_index) > 0:
            logger.warning(
                f"{component_label} participation mode 'elexon_mapped' selected, "
                f"but no directly matched ELEXON {component_label} were found."
            )
    else:
        raise ValueError(
            f"Unsupported {component_label} participation mode='{mode}'. "
            "Use 'all', 'none', or 'elexon_mapped'."
        )

    min_p_nom_mw = float(cfg.get("min_p_nom_mw", 0.0) or 0.0)
    if min_p_nom_mw > 0.0 and "p_nom" in components.columns:
        participants &= set(
            components.index[components["p_nom"].fillna(0.0) >= min_p_nom_mw]
        )

    include_carriers = set(cfg.get("include_carriers", []) or [])
    if include_carriers and "carrier" in components.columns:
        participants &= set(
            components.index[components["carrier"].isin(include_carriers)]
        )

    exclude_carriers = set(cfg.get("exclude_carriers", []) or [])
    if exclude_carriers and "carrier" in components.columns:
        participants -= set(
            components.index[components["carrier"].isin(exclude_carriers)]
        )

    include_names = set(str(v) for v in (cfg.get("include_names", []) or []))
    if include_names:
        participants &= include_names

    exclude_names = set(str(v) for v in (cfg.get("exclude_names", []) or []))
    if exclude_names:
        participants -= exclude_names

    participant_index = pd.Index(
        [name for name in component_index if name in participants]
    )
    non_participant_index = component_index.difference(participant_index)
    return (
        participant_index,
        non_participant_index,
        behavior,
        penalty_offer,
        penalty_bid,
    )


def _apply_participation_policy(
    network,
    market_config: dict,
    logger: logging.Logger,
    gen_offer: pd.Series,
    gen_bid: pd.Series,
    su_offer: pd.Series,
    su_bid: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Apply optional BM participation filters and attach solver metadata."""
    balancing = market_config.get("balancing", {})
    participation_cfg = balancing.get("participation", {}) or {}

    gen_cfg = participation_cfg.get("generators", {}) or {}
    direct_matched_gens = getattr(network, "_bm_direct_matched_generators", pd.Index([]))
    (
        gen_participants,
        gen_non_participants,
        gen_behavior,
        gen_penalty_offer,
        gen_penalty_bid,
    ) = _resolve_component_participants(
        network.generators,
        gen_cfg,
        direct_matched_gens,
        logger,
        "generator",
    )

    gen_penalty_override = pd.Index([])
    if gen_behavior == "fixed" and "carrier" in network.generators.columns:
        gen_penalty_override = gen_non_participants.intersection(
            network.generators.index[
                network.generators["carrier"] == "load_shedding"
            ]
        )
        if len(gen_penalty_override) > 0:
            gen_participants = pd.Index(
                [
                    name
                    for name in network.generators.index
                    if name in set(gen_participants).union(gen_penalty_override)
                ]
            )
            gen_non_participants = gen_non_participants.difference(gen_penalty_override)
            logger.info(
                f"Keeping {len(gen_penalty_override)} load-shedding generators "
                "available as priced-out feasibility slack"
            )

    su_cfg = participation_cfg.get("storage_units", {}) or {}
    (
        su_participants,
        su_non_participants,
        su_behavior,
        su_penalty_offer,
        su_penalty_bid,
    ) = _resolve_component_participants(
        network.storage_units,
        su_cfg,
        matched_names=pd.Index([]),
        logger=logger,
        component_label="storage unit",
    )

    network._bm_eligible_generators = gen_participants
    network._bm_participating_generators = (
        network.generators.index
        if gen_behavior in {"priced_out", "fallback_priced"}
        else gen_participants
    )
    network._bm_fixed_generators = (
        gen_non_participants if gen_behavior == "fixed" else pd.Index([])
    )
    network._bm_eligible_storage_units = su_participants
    network._bm_participating_storage_units = (
        network.storage_units.index
        if su_behavior in {"priced_out", "fallback_priced"}
        else su_participants
    )
    network._bm_fixed_storage_units = (
        su_non_participants if su_behavior == "fixed" else pd.Index([])
    )
    network._bm_generator_participation_behavior = gen_behavior
    network._bm_storage_participation_behavior = su_behavior

    if gen_behavior == "priced_out" and len(gen_non_participants) > 0:
        gen_offer.loc[gen_non_participants] = gen_penalty_offer
        gen_bid.loc[gen_non_participants] = gen_penalty_bid
        if hasattr(network, "_bm_offer_tv") and network._bm_offer_tv is not None:
            network._bm_offer_tv.loc[:, gen_non_participants] = gen_penalty_offer
        if hasattr(network, "_bm_bid_tv") and network._bm_bid_tv is not None:
            network._bm_bid_tv.loc[:, gen_non_participants] = gen_penalty_bid
        if hasattr(network, "_bm_offer_ladders") and network._bm_offer_ladders is not None:
            network._bm_offer_ladders = network._bm_offer_ladders[
                ~network._bm_offer_ladders["generator"].isin(gen_non_participants)
            ].copy()
        if hasattr(network, "_bm_bid_ladders") and network._bm_bid_ladders is not None:
            network._bm_bid_ladders = network._bm_bid_ladders[
                ~network._bm_bid_ladders["generator"].isin(gen_non_participants)
            ].copy()
    elif gen_behavior == "fixed" and len(gen_penalty_override) > 0:
        gen_offer.loc[gen_penalty_override] = gen_penalty_offer
        gen_bid.loc[gen_penalty_override] = gen_penalty_bid
        if hasattr(network, "_bm_offer_tv") and network._bm_offer_tv is not None:
            network._bm_offer_tv.loc[:, gen_penalty_override] = gen_penalty_offer
        if hasattr(network, "_bm_bid_tv") and network._bm_bid_tv is not None:
            network._bm_bid_tv.loc[:, gen_penalty_override] = gen_penalty_bid
        if hasattr(network, "_bm_offer_ladders") and network._bm_offer_ladders is not None:
            network._bm_offer_ladders = network._bm_offer_ladders[
                ~network._bm_offer_ladders["generator"].isin(gen_penalty_override)
            ].copy()
        if hasattr(network, "_bm_bid_ladders") and network._bm_bid_ladders is not None:
            network._bm_bid_ladders = network._bm_bid_ladders[
                ~network._bm_bid_ladders["generator"].isin(gen_penalty_override)
            ].copy()
    elif gen_behavior == "fallback_priced" and len(gen_non_participants) > 0:
        logger.info(
            f"Keeping fallback bid/offer prices for {len(gen_non_participants)} "
            "non-eligible generators"
        )

    if su_behavior == "priced_out" and len(su_non_participants) > 0:
        su_offer.loc[su_non_participants] = su_penalty_offer
        su_bid.loc[su_non_participants] = su_penalty_bid
        if hasattr(network, "_bm_su_offer_tv") and network._bm_su_offer_tv is not None:
            network._bm_su_offer_tv.loc[:, su_non_participants] = su_penalty_offer
        if hasattr(network, "_bm_su_bid_tv") and network._bm_su_bid_tv is not None:
            network._bm_su_bid_tv.loc[:, su_non_participants] = su_penalty_bid
    elif su_behavior == "fallback_priced" and len(su_non_participants) > 0:
        logger.info(
            f"Keeping fallback bid/offer prices for {len(su_non_participants)} "
            "non-eligible storage units"
        )

    logger.info(
        f"BM generator participation: {len(gen_participants)}/{len(network.generators)} "
        f"eligible, {len(gen_non_participants)} non-participants "
        f"({gen_behavior})"
    )
    logger.info(
        f"BM storage participation: {len(su_participants)}/{len(network.storage_units)} "
        f"eligible, {len(su_non_participants)} non-participants "
        f"({su_behavior})"
    )

    return gen_offer, gen_bid, su_offer, su_bid


def calculate_bid_offer_prices(
    network,
    market_config: dict,
    logger: logging.Logger,
    scenario_id: str = "",
    time_varying: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate bid and offer prices for generators and storage units.

    Supports two modes per carrier (set in carrier_overrides):
      "markup"   — offer = mc × (1 + markup), bid = mc × (1 - discount)
      "absolute" — offer/bid are fixed £/MWh values from config

    Bid prices use ESO-cost convention: positive = costs ESO to decrease.
    ELEXON raw bids (negative = ESO pays) are negated at load time.

    Parameters
    ----------
    network : pypsa.Network
        Network with generators and storage units (marginal_cost must be set).
    market_config : dict
        Market configuration dict containing 'balancing' sub-dict.
    logger : logging.Logger
        Logger instance.
    scenario_id : str, optional
        Scenario identifier for file path expansion.
    time_varying : bool, optional
        If True and source is ELEXON, also populate
        ``network._bm_offer_tv`` and ``network._bm_bid_tv`` with
        per-snapshot DataFrames (snapshots × generators), and populate
        ``network._bm_su_offer_tv`` / ``network._bm_su_bid_tv`` for storage
        carriers when ELEXON storage price traces are configured.

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
    bid_offer_source = _resolve_bid_offer_source(
        network, balancing, logger, scenario_id=scenario_id
    )

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
            carrier_overrides, logger,
            scenario_id=scenario_id,
            time_varying=time_varying,
        )
        return _apply_participation_policy(
            network, market_config, logger, gen_offer, gen_bid, su_offer, su_bid
        )

    # ── Source: CSV ───────────────────────────────────────────────────────────
    if bid_offer_source == "csv":
        gen_offer, gen_bid, su_offer, su_bid = _load_csv_bid_offer(
            network, balancing, logger,
        )
        return _apply_participation_policy(
            network, market_config, logger, gen_offer, gen_bid, su_offer, su_bid
        )

    # ── Source: derived (default) ─────────────────────────────────────────────
    if bid_offer_source != "derived":
        raise ValueError(
            f"Unknown bid_offer_source='{bid_offer_source}'. "
            "Supported: 'auto', 'derived', 'elexon', 'csv'."
        )

    # Generators
    gen_offer, gen_bid = _compute_component_prices(
        network.generators,
        carrier_overrides,
        default_offer_markup,
        default_bid_discount,
    )
    logger.info(
        f"Bid/offer prices calculated for {len(gen_offer)} generators "
        f"(offer range: £{gen_offer.min():.2f}-£{gen_offer.max():.2f}/MWh, "
        f"bid range: £{gen_bid.min():.2f}-£{gen_bid.max():.2f}/MWh)"
    )

    # Storage units
    if len(network.storage_units) > 0:
        su_offer, su_bid = _compute_component_prices(
            network.storage_units,
            carrier_overrides,
            default_offer_markup,
            default_bid_discount,
        )
        logger.info(
            f"Bid/offer prices calculated for {len(su_offer)} storage units "
            f"(offer range: £{su_offer.min():.2f}-£{su_offer.max():.2f}/MWh, "
            f"bid range: £{su_bid.min():.2f}-£{su_bid.max():.2f}/MWh)"
        )
    else:
        su_offer = pd.Series(dtype=float)
        su_bid = pd.Series(dtype=float)

    return _apply_participation_policy(
        network, market_config, logger, gen_offer, gen_bid, su_offer, su_bid
    )


def _load_elexon_bid_offer(
    network, balancing: dict,
    default_offer_markup: float, default_bid_discount: float,
    carrier_overrides: dict,
    logger: logging.Logger,
    scenario_id: str = "",
    time_varying: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load bid/offer prices from pre-fetched ELEXON BMRS data.

    Reads per-BMU bid/offer CSVs, maps BMU IDs to PyPSA generator names,
    and fills unmatched generators using carrier-group averages from the
    ELEXON dataset. Storage units fall back to derived pricing.

    When ``time_varying=True``, also attaches per-snapshot price DataFrames
    to the network as ``network._bm_offer_tv`` and ``network._bm_bid_tv``
    (snapshots × generators).  ELEXON-matched generators get their actual
    hourly prices; unmatched generators get their static price broadcast.

    Parameters
    ----------
    network : pypsa.Network
    balancing : dict
        Balancing config with 'elexon' sub-dict.
    default_offer_markup, default_bid_discount : float
        Fallback markups for derived pricing.
    carrier_overrides : dict
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
    bmu_mapping_raw = elexon_cfg.get(
        "bmu_mapping", "resources/generators/{scenario}_bmu_mapping.csv"
    )
    if "{scenario}" in bmu_mapping_raw and scenario_id:
        bmu_mapping_raw = bmu_mapping_raw.replace("{scenario}", scenario_id)
    bmu_mapping_path = Path(bmu_mapping_raw)
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
    elexon_offers.index = pd.to_datetime(elexon_offers.index)
    elexon_bids.index = pd.to_datetime(elexon_bids.index)
    # Remove duplicate timestamps (can occur at DST transitions or data glitches).
    # Keep first occurrence so reindex() against network snapshots works cleanly.
    if elexon_offers.index.duplicated().any():
        n_dup = elexon_offers.index.duplicated().sum()
        logger.warning(f"ELEXON offers: dropping {n_dup} duplicate timestamps")
        elexon_offers = elexon_offers[~elexon_offers.index.duplicated(keep="first")]
    if elexon_bids.index.duplicated().any():
        n_dup = elexon_bids.index.duplicated().sum()
        logger.warning(f"ELEXON bids: dropping {n_dup} duplicate timestamps")
        elexon_bids = elexon_bids[~elexon_bids.index.duplicated(keep="first")]
    elexon_offers_raw = elexon_offers.copy()
    elexon_bids_raw = elexon_bids.copy()
    logger.info(f"Loaded ELEXON offers: {elexon_offers.shape}, bids: {elexon_bids.shape}")

    # Map BMU columns to generator names
    elexon_offers = elexon_offers.rename(columns=bmu_to_gen)
    elexon_bids = elexon_bids.rename(columns=bmu_to_gen)
    if elexon_offers.columns.has_duplicates:
        elexon_offers = elexon_offers.T.groupby(level=0).mean().T
    if elexon_bids.columns.has_duplicates:
        elexon_bids = elexon_bids.T.groupby(level=0).mean().T

    # Build per-generator average offer/bid prices (mean across snapshots)
    gen_offer = pd.Series(index=network.generators.index, dtype=float)
    gen_bid = pd.Series(index=network.generators.index, dtype=float)

    matched = 0
    direct_matched_generators = []
    for gen_name in network.generators.index:
        if gen_name in elexon_offers.columns:
            gen_offer[gen_name] = elexon_offers[gen_name].mean()
            # ELEXON sign convention: negative bid = ESO must PAY generator to reduce.
            # Our objective uses: min Σ offer·inc + bid·dec, where bid must be
            # positive when it costs the ESO money.  Negate ELEXON bids so that
            # the downstream convention is: positive bid = cost to ESO to decrease,
            # negative bid = ESO is paid to turn the unit down.
            gen_bid[gen_name] = -elexon_bids[gen_name].mean()
            matched += 1
            direct_matched_generators.append(gen_name)

    network._bm_direct_matched_generators = pd.Index(direct_matched_generators)

    ladder_cfg = elexon_cfg.get("price_ladders", {}) or {}
    if bool(ladder_cfg.get("enabled", False)):
        offer_ladder_file = Path(
            ladder_cfg.get("offer_file", data_dir / "elexon_offer_ladders.csv")
        )
        bid_ladder_file = Path(
            ladder_cfg.get("bid_file", data_dir / "elexon_bid_ladders.csv")
        )
        if not offer_ladder_file.is_absolute() and offer_ladder_file.parent == Path("."):
            offer_ladder_file = data_dir / offer_ladder_file
        if not bid_ladder_file.is_absolute() and bid_ladder_file.parent == Path("."):
            bid_ladder_file = data_dir / bid_ladder_file

        network._bm_offer_ladders = _load_elexon_ladder_file(
            offer_ladder_file,
            bmu_to_gen,
            network.generators.index,
            logger,
            label="offer",
        )
        network._bm_bid_ladders = _load_elexon_ladder_file(
            bid_ladder_file,
            bmu_to_gen,
            network.generators.index,
            logger,
            label="bid",
        )
        network._bm_ladder_fallback_volume_mw = float(
            ladder_cfg.get("fallback_volume_mw", 1.0e6)
        )
        network._bm_ladder_missing_hour_fallback = int(
            bool(ladder_cfg.get("missing_hour_fallback", True))
        )
    else:
        network._bm_offer_ladders = None
        network._bm_bid_ladders = None

    # ── Fitted fallback (opt-in) ──────────────────────────────────────────
    # Fit per-carrier offer/bid = α·MC + β, γ·MC + δ from ELEXON-matched
    # generators, apply to unmatched ones in the same carrier. Runs BEFORE
    # carrier_average / derived so it takes precedence when enabled.
    fitted_cfg = balancing.get("fitted_fallback", {})
    fitted_enabled = bool(fitted_cfg.get("enabled", False))
    fitted_applied_idx = pd.Index([])
    carrier_fits: dict = {}
    if fitted_enabled:
        from scripts.market.bm_elexon_fit import (
            fit_carrier_prices,
            apply_fitted_fallback,
        )
        min_matched_gens = int(fitted_cfg.get("min_matched_gens", 3))
        min_r2 = float(fitted_cfg.get("min_r2", 0.1))
        fitted_max_price = float(fitted_cfg.get("max_price", 500.0))

        mapped_gen_names = set(bmu_to_gen.values())
        # Negate ELEXON bids for the fit so coefficients are in ESO-cost
        # convention, matching gen_bid downstream.
        logger.info("Fitting per-carrier ELEXON price regressions...")
        carrier_fits = fit_carrier_prices(
            generators=network.generators,
            elexon_offers=elexon_offers,
            elexon_bids=-elexon_bids,
            mapped_gen_names=mapped_gen_names,
            min_matched_gens=min_matched_gens,
            logger=logger,
        )

        unmatched_idx_pre = network.generators.index[gen_offer.isna()]
        fit_offer, fit_bid, fitted_applied_idx = apply_fitted_fallback(
            generators=network.generators,
            unmatched_index=unmatched_idx_pre,
            fits=carrier_fits,
            min_r2=min_r2,
            max_price=fitted_max_price,
            logger=logger,
        )
        if len(fitted_applied_idx) > 0:
            gen_offer.loc[fitted_applied_idx] = fit_offer.loc[fitted_applied_idx]
            gen_bid.loc[fitted_applied_idx] = fit_bid.loc[fitted_applied_idx]

    # Fill remaining unmatched generators with the configured fallback
    unmatched = gen_offer.isna()
    if unmatched.any():
        if fallback == "carrier_average":
            # Use median ELEXON prices from matched generators of the same
            # carrier.  Median is robust to strategic outliers (e.g., nuclear
            # £5000 defensive bids, wind £99999 offers).  Falls back to
            # derived pricing when the median exceeds a sanity cap or no
            # ELEXON-matched generators exist for the carrier.
            max_price = float(elexon_cfg.get("carrier_average_max_price", 500.0))
            carrier_avg_offer = {}
            carrier_avg_bid = {}
            carrier_avg_rejected = {}  # carriers where median exceeded cap
            matched_mask = ~gen_offer.isna()
            for carrier in network.generators.loc[matched_mask, "carrier"].unique():
                carrier_gens = network.generators.index[
                    (network.generators["carrier"] == carrier) & matched_mask
                ]
                if len(carrier_gens) == 0:
                    continue
                med_offer = gen_offer[carrier_gens].median()
                med_bid = gen_bid[carrier_gens].median()
                # Sanity cap: reject carriers with extreme strategic pricing
                if abs(med_offer) > max_price or abs(med_bid) > max_price:
                    carrier_avg_rejected[carrier] = (med_offer, med_bid)
                    continue
                carrier_avg_offer[carrier] = med_offer
                carrier_avg_bid[carrier] = med_bid

            # Compute derived prices as ultimate fallback
            derived_offer, derived_bid = _compute_component_prices(
                network.generators,
                carrier_overrides,
                default_offer_markup,
                default_bid_discount,
            )

            filled_carrier_avg = 0
            filled_derived = 0
            for idx in network.generators.index[unmatched]:
                carrier = network.generators.loc[idx, "carrier"]
                if carrier in carrier_avg_offer:
                    gen_offer[idx] = carrier_avg_offer[carrier]
                    gen_bid[idx] = carrier_avg_bid[carrier]
                    filled_carrier_avg += 1
                else:
                    gen_offer[idx] = derived_offer[idx]
                    gen_bid[idx] = derived_bid[idx]
                    filled_derived += 1

            logger.info(
                f"Carrier-average fallback: {filled_carrier_avg} generators filled "
                f"from ELEXON carrier medians, {filled_derived} from derived pricing"
            )
            if carrier_avg_offer:
                for c in sorted(carrier_avg_offer):
                    logger.info(
                        f"  {c}: ELEXON median offer=£{carrier_avg_offer[c]:.1f}/MWh, "
                        f"bid=£{carrier_avg_bid[c]:.1f}/MWh"
                    )
            if carrier_avg_rejected:
                for c, (mo, mb) in sorted(carrier_avg_rejected.items()):
                    logger.info(
                        f"  {c}: ELEXON median exceeded cap (offer=£{mo:.0f}, "
                        f"bid=£{mb:.0f}, cap=£{max_price:.0f}) → derived fallback"
                    )

        elif fallback == "derived":
            derived_offer, derived_bid = _compute_component_prices(
                network.generators,
                carrier_overrides,
                default_offer_markup,
                default_bid_discount,
            )
            gen_offer.loc[unmatched] = derived_offer.loc[unmatched]
            gen_bid.loc[unmatched] = derived_bid.loc[unmatched]
        else:
            gen_offer[unmatched] = 0.0
            gen_bid[unmatched] = 0.0

    matched_mask = ~unmatched
    _log_elexon_coverage_summary(
        network,
        matched_mask=matched_mask,
        fallback_mask=unmatched,
        fallback_label=fallback,
        logger=logger,
    )

    # ── Build time-varying price matrices ──────────────────────────────────
    # For ELEXON-matched generators: per-snapshot prices from the CSVs.
    # For unmatched: static price broadcast to all snapshots.
    if time_varying and len(network.snapshots) > 0:
        snaps = network.snapshots
        gen_names = network.generators.index

        # Start with static prices broadcast across snapshots
        offer_tv = pd.DataFrame(
            np.tile(gen_offer.values, (len(snaps), 1)),
            index=snaps, columns=gen_names,
        )
        bid_tv = pd.DataFrame(
            np.tile(gen_bid.values, (len(snaps), 1)),
            index=snaps, columns=gen_names,
        )

        # Overlay ELEXON per-snapshot data for matched generators
        for gen_name in gen_names:
            if gen_name in elexon_offers.columns:
                offer_aligned = elexon_offers[gen_name].reindex(snaps)
                bid_aligned = elexon_bids[gen_name].reindex(snaps)
                # Fill gaps with the static mean
                offer_tv[gen_name] = offer_aligned.fillna(gen_offer[gen_name])
                # Negate ELEXON bids → ESO cost convention with no clipping.
                bid_tv[gen_name] = (-bid_aligned).fillna(
                    gen_bid[gen_name]
                )

        # Apply carrier-mean-normalised ELEXON hourly shape to generators that
        # got fitted-fallback prices, so unmatched gens get peak/offpeak
        # structure instead of a flat line.
        if fitted_enabled and len(fitted_applied_idx) > 0 \
                and fitted_cfg.get("time_varying_shape", True):
            from scripts.market.bm_elexon_fit import build_carrier_shape
            mapped_gen_names = set(bmu_to_gen.values())
            min_matched_gens = int(fitted_cfg.get("min_matched_gens", 3))
            offer_shape = build_carrier_shape(
                network.generators, elexon_offers,
                mapped_gen_names, min_matched_gens,
            )
            bid_shape = build_carrier_shape(
                network.generators, -elexon_bids,
                mapped_gen_names, min_matched_gens,
            )
            if not offer_shape.empty:
                offer_shape = offer_shape.reindex(snaps).fillna(1.0)
            if not bid_shape.empty:
                bid_shape = bid_shape.reindex(snaps).fillna(1.0)

            shaped_gens = 0
            for idx in fitted_applied_idx:
                carrier = network.generators.loc[idx, "carrier"]
                if not offer_shape.empty and carrier in offer_shape.columns:
                    offer_tv[idx] = gen_offer[idx] * offer_shape[carrier].values
                if not bid_shape.empty and carrier in bid_shape.columns:
                    bid_tv[idx] = gen_bid[idx] * bid_shape[carrier].values
                if (not offer_shape.empty and carrier in offer_shape.columns) \
                        or (not bid_shape.empty and carrier in bid_shape.columns):
                    shaped_gens += 1
            if shaped_gens > 0:
                logger.info(
                    f"Time-varying shape applied to {shaped_gens} fitted-fallback generators"
                )

        network._bm_offer_tv = offer_tv
        network._bm_bid_tv = bid_tv
        logger.info(
            f"Time-varying bid/offer prices: {offer_tv.shape} "
            f"(ELEXON per-snapshot for {matched} generators)"
        )

    # Storage units — default to derived pricing, but use carrier-level ELEXON
    # traces where configured (e.g. pumped storage BMUs in historical runs).
    su_offer = pd.Series(dtype=float)
    su_bid = pd.Series(dtype=float)
    if len(network.storage_units) > 0:
        su_offer, su_bid = _compute_component_prices(
            network.storage_units,
            carrier_overrides,
            default_offer_markup,
            default_bid_discount,
        )

        storage_prefix_map = elexon_cfg.get(
            "storage_bmu_prefixes", DEFAULT_STORAGE_BMU_PREFIXES
        )
        storage_price_aggregation = str(
            elexon_cfg.get("storage_price_aggregation", "median")
        ).lower()

        def _aggregate_storage_prices(df: pd.DataFrame, columns: list[str]) -> pd.Series:
            if storage_price_aggregation == "mean":
                return df[columns].mean(axis=1, skipna=True)
            return df[columns].median(axis=1, skipna=True)

        su_offer_tv = None
        su_bid_tv = None
        if time_varying and len(network.snapshots) > 0:
            snaps = network.snapshots
            su_offer_tv = pd.DataFrame(
                np.tile(su_offer.values, (len(snaps), 1)),
                index=snaps,
                columns=network.storage_units.index,
            )
            su_bid_tv = pd.DataFrame(
                np.tile(su_bid.values, (len(snaps), 1)),
                index=snaps,
                columns=network.storage_units.index,
            )

        matched_storage_carriers = []
        for carrier, prefixes in storage_prefix_map.items():
            su_names = network.storage_units.index[
                network.storage_units["carrier"] == carrier
            ]
            if len(su_names) == 0:
                continue

            prefix_list = prefixes if isinstance(prefixes, list) else [prefixes]
            matched_cols = [
                col for col in elexon_offers_raw.columns
                if any(str(col).startswith(prefix) for prefix in prefix_list)
            ]
            if not matched_cols:
                continue

            offer_series = _aggregate_storage_prices(
                elexon_offers_raw, matched_cols
            )
            bid_series = -_aggregate_storage_prices(elexon_bids_raw, matched_cols)

            static_offer = float(offer_series.mean())
            static_bid = float(bid_series.mean())
            su_offer.loc[su_names] = static_offer
            su_bid.loc[su_names] = static_bid

            if su_offer_tv is not None and su_bid_tv is not None:
                offer_aligned = offer_series.reindex(snaps).fillna(static_offer)
                bid_aligned = bid_series.reindex(snaps).fillna(static_bid)
                for su_name in su_names:
                    su_offer_tv[su_name] = offer_aligned.values
                    su_bid_tv[su_name] = bid_aligned.values

            matched_storage_carriers.append(
                f"{carrier} ({len(matched_cols)} BMUs → {len(su_names)} units)"
            )

        if su_offer_tv is not None and su_bid_tv is not None:
            network._bm_su_offer_tv = su_offer_tv
            network._bm_su_bid_tv = su_bid_tv

        if matched_storage_carriers:
            logger.info(
                "ELEXON storage bid/offer applied for carriers: "
                + ", ".join(matched_storage_carriers)
            )
        else:
            logger.info(
                "No ELEXON storage carrier traces matched; storage falls back to derived pricing"
            )

    return gen_offer, gen_bid, su_offer, su_bid


def _load_csv_bid_offer(
    network, balancing: dict,
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
    gen_bid = _extract_prices(bid_df, network.generators, "generator")

    su_offer = pd.Series(dtype=float)
    su_bid = pd.Series(dtype=float)
    if len(network.storage_units) > 0:
        su_offer = _extract_prices(offer_df, network.storage_units, "storage_unit")
        su_bid = _extract_prices(bid_df, network.storage_units, "storage_unit")

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
    gen_offer_prices_tv: pd.DataFrame | None = None,
    gen_bid_prices_tv: pd.DataFrame | None = None,
    su_offer_prices_tv: pd.DataFrame | None = None,
    su_bid_prices_tv: pd.DataFrame | None = None,
    gen_offer_ladders: pd.DataFrame | None = None,
    gen_bid_ladders: pd.DataFrame | None = None,
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

    def _snapshot_hours(index: pd.Index) -> pd.Series:
        """Return per-snapshot durations in hours for MW-to-MWh conversion."""
        if len(index) == 0:
            return pd.Series(dtype=float, index=index)
        if len(index) == 1:
            return pd.Series(1.0, index=index, dtype=float)

        dt = pd.Series(index=index, data=index.to_series().diff().dt.total_seconds() / 3600)
        inferred = dt.dropna().median()
        if not np.isfinite(inferred) or inferred <= 0:
            inferred = 1.0
        dt.iloc[0] = inferred
        dt = dt.fillna(inferred).clip(lower=0.0)
        return dt.astype(float)

    def _ladder_cost(
        volume: float,
        snapshot,
        component: str,
        ladder_lookup: dict | None,
        fallback_price: float,
    ) -> float:
        if volume <= 0 or not ladder_lookup:
            return volume * fallback_price
        blocks = ladder_lookup.get((snapshot, component))
        if blocks is None or len(blocks) == 0:
            return volume * fallback_price

        remaining = float(volume)
        cost = 0.0
        for price, volume_mw in blocks:
            take = min(remaining, float(volume_mw))
            if take > 0:
                cost += take * float(price)
                remaining -= take
            if remaining <= 1.0e-9:
                break
        if remaining > 1.0e-9:
            cost += remaining * fallback_price
        return cost

    def _prepare_ladder_lookup(ladders: pd.DataFrame | None) -> dict:
        if ladders is None or ladders.empty:
            return {}
        lookup = {}
        for key, group in ladders.groupby(["snapshot", "generator"], sort=False):
            blocks = (
                group.sort_values("price")[["price", "volume_mw"]]
                .to_numpy(dtype=float)
            )
            lookup[key] = blocks
        return lookup

    def _summarize(
        wholesale_df,
        physical_df,
        offer_prices,
        bid_prices,
        components,
        component_type,
        offer_prices_tv=None,
        bid_prices_tv=None,
        offer_ladders=None,
        bid_ladders=None,
    ):
        records = []
        # Align columns
        common = wholesale_df.columns.intersection(physical_df.columns)
        snapshot_hours = _snapshot_hours(wholesale_df.index)

        offer_prices_tv_aligned = None
        bid_prices_tv_aligned = None
        if offer_prices_tv is not None:
            offer_prices_tv_aligned = offer_prices_tv.reindex(
                index=wholesale_df.index, columns=common
            )
        if bid_prices_tv is not None:
            bid_prices_tv_aligned = bid_prices_tv.reindex(
                index=wholesale_df.index, columns=common
            )
        offer_ladder_lookup = _prepare_ladder_lookup(offer_ladders)
        bid_ladder_lookup = _prepare_ladder_lookup(bid_ladders)

        for name in common:
            diff = physical_df[name] - wholesale_df[name]  # MW per snapshot
            increase_mw = diff.clip(lower=0)
            decrease_mw = (-diff).clip(lower=0)
            increase_series = increase_mw * snapshot_hours  # MWh per snapshot
            decrease_series = decrease_mw * snapshot_hours
            increase = increase_series.sum()
            decrease = decrease_series.sum()

            if offer_ladder_lookup:
                if offer_prices_tv_aligned is not None and name in offer_prices_tv_aligned:
                    fallback_offer = offer_prices_tv_aligned[name].fillna(
                        offer_prices.get(name, 0.0)
                    )
                else:
                    fallback_offer = pd.Series(
                        offer_prices.get(name, 0.0), index=increase_series.index
                    )
                offer_cost = sum(
                    _ladder_cost(
                        mw, snap, name, offer_ladder_lookup, fallback_offer.loc[snap]
                    )
                    * snapshot_hours.loc[snap]
                    for snap, mw in increase_mw.items()
                )
            elif offer_prices_tv_aligned is not None and name in offer_prices_tv_aligned:
                offer_cost = (
                    increase_series * offer_prices_tv_aligned[name].fillna(0.0)
                ).sum()
            else:
                offer_cost = increase * offer_prices.get(name, 0.0)

            if bid_ladder_lookup:
                if bid_prices_tv_aligned is not None and name in bid_prices_tv_aligned:
                    fallback_bid = bid_prices_tv_aligned[name].fillna(
                        bid_prices.get(name, 0.0)
                    )
                else:
                    fallback_bid = pd.Series(
                        bid_prices.get(name, 0.0), index=decrease_series.index
                    )
                bid_cost = sum(
                    _ladder_cost(
                        mw, snap, name, bid_ladder_lookup, fallback_bid.loc[snap]
                    )
                    * snapshot_hours.loc[snap]
                    for snap, mw in decrease_mw.items()
                )
            elif bid_prices_tv_aligned is not None and name in bid_prices_tv_aligned:
                bid_cost = (
                    decrease_series * bid_prices_tv_aligned[name].fillna(0.0)
                ).sum()
            else:
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
        network.generators, "generator",
        offer_prices_tv=gen_offer_prices_tv,
        bid_prices_tv=gen_bid_prices_tv,
        offer_ladders=gen_offer_ladders,
        bid_ladders=gen_bid_ladders,
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
            network.storage_units, "storage_unit",
            offer_prices_tv=su_offer_prices_tv,
            bid_prices_tv=su_bid_prices_tv,
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
