"""Solve-time foreign price sensitivity on a shared network (decision 29, FES 2050 plan).

A 2050 scenario that solves on its parent's network (``network_from``) can differ
from the parent only in solve-time keys, but the foreign prices are written into the
network at build (``add_to_network``: the signed interconnector links' hourly ask,
static bid/ask, annual levels and IC1 volume-tranche steps). This module re-prices
those links on the loaded network from another TYNDP variant's shape, using the
build's own functions, so a European price sensitivity solves on the shared network
without rebuilding it.

Opt-in per scenario, under ``market`` (a solve-time key):

    market:
      foreign_price_override:
        enabled: true
        tyndp: {scenario: GA, route: tyndp_shape}   # DE | GA | NTplus2026

Only a network built on TYNDP prices (``interconnectors.pricing.foreign_price_source:
tyndp``, the ``HT50_*`` scenarios) may be re-priced. Absent or disabled, nothing is
read or changed. Re-pricing with the parent's own variant reproduces the built links
exactly (checked by ``verify``).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

PRICE_COLUMNS = (
    "marginal_cost",
    "market_export_bid",
    "market_export_bid_offset",
    "market_import_level",
    "market_export_level",
    "market_price_basis",
    "market_spread_half",
    "market_circulation_epsilon",
)


def override_block(scenario_config) -> dict:
    """The scenario's ``market.foreign_price_override`` block, or ``{}`` when absent or disabled."""
    block = ((scenario_config or {}).get("market") or {}).get("foreign_price_override") or {}
    if not block or not block.get("enabled", False):
        return {}
    unknown = sorted(set(block) - {"enabled", "tyndp"})
    if unknown:
        raise ValueError(f"market.foreign_price_override: unknown keys {unknown}")
    return block


def effective_pricing(scenario_config) -> dict:
    """``interconnectors.pricing`` with the override's TYNDP variant in place of the build's."""
    from scripts.interconnectors.tyndp_prices import is_tyndp_pricing, tyndp_settings

    block = override_block(scenario_config)
    pricing = copy.deepcopy(((scenario_config or {}).get("interconnectors") or {}).get("pricing") or {})
    if not block:
        return pricing
    if not is_tyndp_pricing(pricing):
        raise ValueError(
            "market.foreign_price_override needs a network built on TYNDP prices "
            "(interconnectors.pricing.foreign_price_source: tyndp)"
        )
    pricing["tyndp"] = tyndp_settings({"tyndp": block.get("tyndp") or {}})
    return pricing


def shape_path(scenario_config, resources_path: str = "resources") -> str | None:
    """The override's TYNDP shape file (rule ``build_tyndp_price_shape``), or None."""
    from scripts.interconnectors.tyndp_prices import tyndp_variant

    if not override_block(scenario_config):
        return None
    year = scenario_config.get("modelled_year") or scenario_config.get("year")
    weather_year = scenario_config.get("renewables_year", year)
    variant = tyndp_variant(effective_pricing(scenario_config))
    return f"{resources_path}/interconnectors/tyndp_{variant}_{year}_shape_wind{weather_year}.csv"


def market_demand_path(scenario_config) -> str | None:
    """The volume-tranche market demand file the build read, when tranches are on and overridden."""
    if not override_block(scenario_config):
        return None
    tranches = effective_pricing(scenario_config).get("volume_tranches") or {}
    return tranches.get("demand_file") if tranches.get("enabled", False) else None


def _signed_links(network) -> pd.DataFrame:
    links = network.links
    mode = links.get("foreign_market_pricing", pd.Series("", index=links.index)).astype(str).str.lower()
    signed = links[mode.eq("piecewise_bidirectional")]
    if signed.empty:
        raise ValueError("foreign_price_override: the network has no signed interconnector links")
    return signed


def _interconnectors_frame(signed: pd.DataFrame) -> pd.DataFrame:
    """The cable table ``add_interconnector_links`` was built from, recovered from the links."""
    prefix = "IC_"
    names = [str(i)[len(prefix):] if str(i).startswith(prefix) else str(i) for i in signed.index]
    frame = pd.DataFrame({
        "name": names,
        "from_bus": signed["bus1"].to_numpy(),
        "to_bus": signed["bus0"].to_numpy(),
        "capacity_mw": signed["p_nom"].astype(float).to_numpy(),
        "losses_percent": ((1.0 - signed["efficiency"].astype(float)) * 100.0).to_numpy(),
        "counterparty_country": signed["counterparty_country"].to_numpy(),
        "commissioning_year": signed.get("commissioning_year", pd.Series(np.nan, index=signed.index)).to_numpy(),
        "source": signed.get("source", pd.Series("Unknown", index=signed.index)).to_numpy(),
    })
    rebuilt = [f"{prefix}{n}" for n in frame["name"]]
    if rebuilt != list(signed.index):
        raise ValueError("foreign_price_override: interconnector link names do not follow IC_<name>")
    return frame


def reprice(network, pricing: dict, prices: pd.DataFrame, market_demand_file: str | None,
            modelled_year, logger=None) -> pypsa.Network:
    """Price the signed interconnectors as ``add_to_network`` would, on a scratch network.

    Returns the scratch network: its ``IC_*`` links carry the build's price columns
    and hourly ask for ``pricing`` and ``prices``.
    """
    from scripts.interconnectors import add_to_network as atn

    logger = logger or logging.getLogger(__name__)
    signed = _signed_links(network)
    frame = _interconnectors_frame(signed)
    scratch = pypsa.Network()
    scratch.set_snapshots(network.snapshots)
    for bus in pd.unique(pd.concat([frame["from_bus"], frame["to_bus"]])):
        scratch.add("Bus", bus, carrier=str(network.buses.at[bus, "carrier"]) if bus in network.buses.index else "AC")
    # The external buses carry their foreign market's balancing generator in the build;
    # add_external_market_balancers adds them again here (scratch only).
    atn.add_interconnector_links(
        scratch, frame, pd.DataFrame(), None,
        modelled_year=modelled_year, fes_pathway=None,
        entsoe_prices_df=prices, pricing_config=pricing,
    )
    if len(scratch.links) != len(signed):
        raise ValueError(
            f"foreign_price_override: re-priced {len(scratch.links)} of {len(signed)} interconnectors"
        )
    tranches = pricing.get("volume_tranches") or {}
    if tranches.get("enabled", False):
        from scripts.interconnectors.volume_tranches import apply_volume_tranches, load_market_demand_twh

        if not market_demand_file:
            raise ValueError("foreign_price_override: volume tranches are on but no market demand file was given")
        demand_twh = load_market_demand_twh(
            market_demand_file, tranches["eu_scenario"], tranches["pathway_alignment"], int(tranches["demand_year"]))
        apply_volume_tranches(scratch, tranches, demand_twh, logger=logger)
    return scratch


def _columns_to_copy(scratch_links: pd.DataFrame) -> list[str]:
    from scripts.interconnectors.market_coupling import EXPORT_BID_FLOOR_COLUMN
    from scripts.interconnectors.volume_tranches import TRANCHE_COLUMNS

    wanted = list(PRICE_COLUMNS) + [EXPORT_BID_FLOOR_COLUMN] + list(TRANCHE_COLUMNS)
    return [c for c in wanted if c in scratch_links.columns]


def apply_foreign_price_override(network, scenario_config, inputs, logger=None) -> dict | None:
    """Re-price the network's interconnectors to the scenario's override variant, in place,
    then re-fit the IC1 tranche steps if ``market.ic1_tranche_refit`` is set.

    ``inputs`` is the rule's ``snakemake.input`` (``foreign_price_shape`` and, with
    volume tranches, ``foreign_price_market_demand``; ``ic1_refit_market_demand`` for the
    refit). A no-op returning None when the scenario sets neither key. Returns a summary of
    the annual levels before and after the override (None if only the refit is set).
    """
    logger = logger or logging.getLogger(__name__)
    summary = _apply_override(network, scenario_config, inputs, logger)
    refit_tranche_steps(network, scenario_config, _input(inputs, "ic1_refit_market_demand"), logger)
    return summary


def _input(inputs, key):
    if inputs is None:
        return None
    if hasattr(inputs, "get"):
        return inputs.get(key)
    return getattr(inputs, key, None)


def _apply_override(network, scenario_config, inputs, logger) -> dict | None:
    block = override_block(scenario_config)
    if not block:
        return None
    from scripts.interconnectors.add_to_network import load_entsoe_prices

    pricing = effective_pricing(scenario_config)
    shape_file = inputs.get("foreign_price_shape") if hasattr(inputs, "get") else getattr(inputs, "foreign_price_shape")
    if not shape_file or not Path(str(shape_file)).exists():
        raise FileNotFoundError(f"foreign_price_override: shape file missing: {shape_file}")
    demand_file = inputs.get("foreign_price_market_demand") if hasattr(inputs, "get") else None
    prices = load_entsoe_prices(str(shape_file))
    if prices.empty:
        raise ValueError(f"foreign_price_override: could not read {shape_file}")
    modelled_year = scenario_config.get("modelled_year") or scenario_config.get("year")
    scratch = reprice(network, pricing, prices, demand_file, modelled_year, logger)

    signed = _signed_links(network).index
    before = network.links.loc[signed, "market_import_level"].astype(float).copy()
    for column in _columns_to_copy(scratch.links):
        if column not in network.links.columns:
            network.links[column] = scratch.links[column].iloc[:0].reindex(network.links.index)
        network.links.loc[signed, column] = scratch.links.loc[signed, column].to_numpy()
    hourly = scratch.links_t.marginal_cost.reindex(columns=signed)
    if hourly.isna().any().any():
        raise ValueError("foreign_price_override: the re-priced hourly ask has gaps")
    table = network.links_t.marginal_cost
    for link in signed:
        table[link] = hourly[link].to_numpy(dtype=float)
    network.links_t["marginal_cost"] = table
    after = network.links.loc[signed, "market_import_level"].astype(float)
    by_country = pd.DataFrame({
        "country": network.links.loc[signed, "counterparty_country"],
        "before": before, "after": after,
    }).groupby("country")[["before", "after"]].mean()
    logger.info("Foreign price override (decision 29): TYNDP %s, shape %s", pricing["tyndp"], shape_file)
    for country, row in by_country.iterrows():
        logger.info("  %s annual level %.2f -> %.2f GBP/MWh", country, row["before"], row["after"])
    network.meta = dict(getattr(network, "meta", {}) or {})
    network.meta["foreign_price_override"] = {"tyndp": pricing["tyndp"], "shape": str(shape_file)}
    return {"tyndp": pricing["tyndp"], "levels": by_country.round(4).to_dict(orient="index")}


# ── IC1 tranche refit (Y8 follow-up, 3 October 2026; 2050 only, off by default) ──────────
#
# The build already derives the steps by IC1's formula on 2050 inputs (Y3 step 5):
# step = elasticity x level / market demand x width, width = 2050 GB capacity / tranches,
# demand = ES2 2050, level = the TYNDP annual level. ``level_basis: annual`` re-derives
# exactly that (a check: drift 0). The other two bases follow IC1's 2030 convention, where
# the level was the price GB trades at on each side (CP.29 import price for imports, export
# price for exports), on the 2050 single-price basis:
#   traded_annual  the year's mean import ask and mean export bid
#   traded_window  the same over the scenario's solve period (the constant-elasticity curve
#                  dP/dQ = elasticity x P / Q evaluated at the prices of the hours solved)
#
#     market:
#       ic1_tranche_refit: {enabled: true, level_basis: traded_window, elasticity: null}

REFIT_LEVEL_BASES = ("annual", "traded_annual", "traded_window")


def tranche_refit_block(scenario_config) -> dict:
    """The scenario's ``market.ic1_tranche_refit`` block, or ``{}`` when absent or disabled."""
    block = ((scenario_config or {}).get("market") or {}).get("ic1_tranche_refit") or {}
    if not block or not block.get("enabled", False):
        return {}
    unknown = sorted(set(block) - {"enabled", "level_basis", "elasticity"})
    if unknown:
        raise ValueError(f"market.ic1_tranche_refit: unknown keys {unknown}")
    if block.get("level_basis", "annual") not in REFIT_LEVEL_BASES:
        raise ValueError(f"market.ic1_tranche_refit.level_basis must be one of {REFIT_LEVEL_BASES}")
    return block


def refit_market_demand_path(scenario_config) -> str | None:
    """The ES2 market demand file the refit reads (the build's tranche demand file)."""
    if not tranche_refit_block(scenario_config):
        return None
    tranches = effective_pricing(scenario_config).get("volume_tranches") or {}
    if not tranches.get("enabled", False):
        raise ValueError("market.ic1_tranche_refit needs interconnectors.pricing.volume_tranches enabled")
    return tranches.get("demand_file")


def _refit_snapshots(network, scenario_config, basis: str) -> pd.Index:
    snapshots = pd.Index(network.snapshots)
    period = (scenario_config or {}).get("solve_period") or {}
    if basis == "traded_window" and period.get("enabled", False):
        stamps = pd.DatetimeIndex(snapshots)
        mask = (stamps >= pd.Timestamp(period["start"])) & (stamps <= pd.Timestamp(period["end"]))
        if not mask.any():
            raise ValueError("market.ic1_tranche_refit: the solve period selects no snapshot")
        snapshots = snapshots[mask]
    return snapshots


def refit_levels(network, scenario_config, basis: str) -> pd.DataFrame:
    """Import and export level per signed link for a level basis (GBP/MWh)."""
    from scripts.interconnectors.market_coupling import piecewise_market_prices

    signed = _signed_links(network)
    if basis == "annual":
        return pd.DataFrame({"import_level": signed["market_import_level"].astype(float),
                             "export_level": signed["market_export_level"].astype(float)})
    snapshots = _refit_snapshots(network, scenario_config, basis)
    ask_t = network.links_t.get("marginal_cost", pd.DataFrame())
    rows = {}
    for link in signed.index:
        ask, bid = piecewise_market_prices(network.links, ask_t, link, snapshots)
        rows[link] = {"import_level": float(ask.mean()), "export_level": float(bid.mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def refit_tranche_steps(network, scenario_config, market_demand_file: str | None,
                        logger=None) -> pd.DataFrame | None:
    """Re-derive the IC1 tranche steps on the loaded network, in place (module note above).

    A no-op returning None unless ``market.ic1_tranche_refit`` is enabled. Only the two step
    columns change; groups, widths and counts are re-derived and must equal the build's.
    """
    from scripts.interconnectors.volume_tranches import derive_market_tranches, load_market_demand_twh

    block = tranche_refit_block(scenario_config)
    if not block:
        return None
    logger = logger or logging.getLogger(__name__)
    tranches = effective_pricing(scenario_config).get("volume_tranches") or {}
    demand_file = market_demand_file or refit_market_demand_path(scenario_config)
    if not demand_file or not Path(str(demand_file)).exists():
        raise FileNotFoundError(f"market.ic1_tranche_refit: market demand file missing: {demand_file}")
    demand_twh = load_market_demand_twh(demand_file, tranches["eu_scenario"], tranches["pathway_alignment"],
                                        int(tranches["demand_year"]))
    basis = block.get("level_basis", "annual")
    elasticity = float(block["elasticity"]) if block.get("elasticity") is not None else float(tranches["elasticity"])
    signed = _signed_links(network)
    levels = refit_levels(network, scenario_config, basis)
    frame = signed[["p_nom", "counterparty_country"]].copy()
    frame["market_import_level"] = levels["import_level"]
    frame["market_export_level"] = levels["export_level"]
    table = derive_market_tranches(frame, tranches["markets"], demand_twh, elasticity=elasticity,
                                   n_tranches=int(tranches["tranches"]))
    links = network.links
    for market, row in table.iterrows():
        members = row["links"].split(";")
        built_width = pd.to_numeric(links.loc[members, "market_tranche_width_mw"], errors="coerce")
        if not np.allclose(built_width, float(row["width_mw"]), rtol=1e-9, atol=1e-6) or \
                not links.loc[members, "market_tranche_group"].astype(str).eq(market).all():
            raise ValueError(f"market.ic1_tranche_refit: {market} widths or members differ from the build")
        before = (float(links.loc[members[0], "market_tranche_import_step"]),
                  float(links.loc[members[0], "market_tranche_export_step"]))
        links.loc[members, "market_tranche_import_step"] = float(row["import_step"])
        links.loc[members, "market_tranche_export_step"] = float(row["export_step"])
        logger.info("IC1 tranche refit (%s, elasticity %.2f) %s: levels %.2f / %.2f, steps import %.3f -> %.3f, "
                    "export %.3f -> %.3f GBP/MWh per tranche", basis, elasticity, market, row["import_level"],
                    row["export_level"], before[0], row["import_step"], before[1], row["export_step"])
    network.meta = dict(getattr(network, "meta", {}) or {})
    network.meta["ic1_tranche_refit"] = {
        "level_basis": basis, "elasticity": elasticity,
        "steps": {m: {"import": float(r["import_step"]), "export": float(r["export_step"])}
                  for m, r in table.iterrows()}}
    return table


def verify(network, scenario_config, prices: pd.DataFrame, market_demand_file: str | None) -> dict:
    """Max absolute differences between the network's built prices and a re-price at ``scenario_config``'s
    own pricing (no override). Zero everywhere means the re-pricing reproduces the build."""
    pricing = copy.deepcopy(((scenario_config or {}).get("interconnectors") or {}).get("pricing") or {})
    modelled_year = scenario_config.get("modelled_year") or scenario_config.get("year")
    scratch = reprice(network, pricing, prices, market_demand_file, modelled_year)
    signed = _signed_links(network).index
    diffs = {}
    for column in _columns_to_copy(scratch.links):
        a, b = network.links.loc[signed, column], scratch.links.loc[signed, column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            d = (a.astype(float) - b.astype(float)).abs()
            both_nan = a.isna() & b.isna()
            diffs[column] = float(d[~both_nan].max()) if (~both_nan).any() else 0.0
        else:
            diffs[column] = int((a.astype(str) != b.astype(str)).sum())
    built = network.links_t.marginal_cost.reindex(columns=signed)
    diffs["links_t.marginal_cost"] = float((built - scratch.links_t.marginal_cost.reindex(columns=signed)).abs().max().max())
    return diffs
