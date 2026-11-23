#!/usr/bin/env python3
"""
Add hydrogen system components to PyPSA network.

This script implements a simplified hydrogen sector coupling:
1. Adds a single GB-wide hydrogen bus
2. Adds electrolysis (Link: electricity → H2 bus)
3. Adds hydrogen storage (Store at H2 bus)
4. Converts H2 generators to Links (H2 bus → electricity)

This is a simplified "copper-plate" hydrogen network - all H2 assets share
a single storage pool. Future work will model regional hydrogen networks.

Key Parameters (based on literature):
- Electrolysis efficiency: 70% (LHV basis, including auxiliary loads)
- H2 turbine efficiency: 50% (combined cycle H2-ready turbines)
- Storage: Sized for 168h of generation capacity

Turbine fuel and VOM (item OW3, 3 Oct 2026; ``hydrogen.turbine_fuel`` and
``hydrogen.turbine_variable_om``, set in config/defaults/method/brownfield.yaml only). With
``turbine_fuel.source: external_supply`` the turbines burn hydrogen bought from
the wider hydrogen system at a fixed price instead of the GB_H2 store: they take
it off their own bus (``GB_H2_power_fuel``) from a full tank priced per MWh
drawn. The flat offtake already equals all of CP30's electrolytic hydrogen, so
turbine fuel electrolysed on top of it had made electrolysis 4.9% too high and
priced the fuel at the store's dual (mean 53.4, P90 147 GBP/MWh_H2 in the
2030 run), running the turbines 126 GWh against CP30's 338. The tank is a Store,
not a Generator, so no generator-level account (margin, settlement, register,
energy validation, BM participation) sees hydrogen as electricity.
``turbine_variable_om.basis: electricity_output`` puts the VOM on the electricity
leg (``marginal_cost = VOM x efficiency`` on p0, which is hydrogen in); the
default ``hydrogen_input`` keeps it on p0, where 5 per MWh_H2 is 10 per MWh_el.

Author: PyPSA-GB Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import logging
import warnings
from typing import Optional

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network
from scripts.CP30.assumptions import load_or_derive_cp30_electrolysis_capacity_mw

# Suppress PyPSA warnings
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Import logging configuration
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("add_hydrogen_system")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


# =============================================================================
# HYDROGEN SYSTEM PARAMETERS
# =============================================================================

# Electrolysis parameters
ELECTROLYSIS_EFFICIENCY = 0.70  # 70% electrical → H2 (LHV basis)
ELECTROLYSIS_MARGINAL_COST = 0.0  # £/MWh_el (operating cost, excl. electricity)

# H2 power generation parameters
H2_TURBINE_EFFICIENCY = 0.50  # 50% H2 → electricity (combined cycle)
H2_TURBINE_MARGINAL_COST = 5.0  # £/MWh_el (maintenance, water, etc.)

# Hydrogen storage parameters
H2_STORAGE_HOURS = 168  # 1 week of H2 generation capacity (for seasonal flexibility)
H2_STORAGE_EFFICIENCY = 0.98  # 98% round-trip (compression/decompression losses)
H2_STORAGE_STANDING_LOSS = 0.0001  # 0.01% per hour (minimal for cavern storage)

# Turbine fuel and VOM (item OW3). Code defaults reproduce the build before the
# item (turbines burn GB_H2 hydrogen; VOM on p0); config/defaults/ sets
# other values in config/defaults/method/brownfield.yaml hydrogen only.
TURBINE_FUEL_SOURCES = ("electrolysis", "external_supply")
TURBINE_VOM_BASES = ("hydrogen_input", "electricity_output")
DEFAULT_TURBINE_FUEL_SOURCE = "electrolysis"
DEFAULT_TURBINE_VOM_BASIS = "hydrogen_input"
H2_POWER_FUEL_BUS = "GB_H2_power_fuel"
H2_POWER_FUEL_STORE = "GB_H2_power_fuel_supply"
H2_POWER_FUEL_CARRIER = "H2_power_fuel"
# The tank holds a leap year of full-load burn, so it never binds in any solve
# period (months, rolling windows or a whole year) whether or not a solve
# carries store levels between windows (it is not a carried store: its
# carrier is not market.hydrogen_store.carriers).
H2_POWER_FUEL_TANK_HOURS = 8784
# Turbine availability (gas_cal, 14 Oct 2026): ``hydrogen.turbine_availability``
# is the turbine links' static p_max_pu, an output calibration that leaves
# capacity (p_nom) at CP30's. Code default 1.0: no derating, the build before it.
DEFAULT_TURBINE_AVAILABILITY = 1.0

# Electrolyser siting (15 Oct 2026; ``hydrogen.electrolyser_siting``). The code default
# ``h2_turbine_pro_rata`` is the build before: capacity pro rata to H2-turbine capacity by bus,
# which put 82% of CP30's 3.65 GW north of B8 at Humber/Teesside/Mersey turbine sites and 26 MW
# north of B4. ``fes_gsp`` places it by FES networked electrolysis (``Dem_BB009``) by GSP node,
# FES's Direct rows pro rata to their licence area's GSP rows, then relieves every node whose
# peak load plus electrolysis exceeds ``max_loading`` x what it can import at the BM's ratings
# (``relieve_electrolyser_import_headroom``, 16 Oct 2026; 0 switches the relief off).
ELECTROLYSER_SITING_METHODS = ("h2_turbine_pro_rata", "fes_gsp")
DEFAULT_ELECTROLYSER_SITING = {
    "method": "h2_turbine_pro_rata",
    "fes_year": 2024,
    "fes_scenario": "Holistic Transition",
    "max_loading": 1.0,
    "min_target_voltage_kv": 275.0,
    "boundaries": ["B4", "B6", "B8"],
}
#: Keys of ``hydrogen.electrolyser_siting`` retired with the import relief (16 Oct 2026).
RETIRED_SITING_KEYS = {
    "search_radius_km": "the import relief moves excess to a higher-voltage node of the same substation, else the "
                        "nearest node at or above min_target_voltage_kv in the same CP30 ES5 region and on the same "
                        "side of every siting boundary; it has no search radius",
}


def resolve_electrolyser_siting(hydrogen_config: Optional[dict]) -> dict:
    """``hydrogen.electrolyser_siting`` with code defaults filled and validated."""
    siting = dict(DEFAULT_ELECTROLYSER_SITING)
    given = dict((hydrogen_config or {}).get("electrolyser_siting") or {})
    retired = sorted(set(given) & set(RETIRED_SITING_KEYS))
    if retired:
        raise ValueError("hydrogen.electrolyser_siting: retired key(s) " + "; ".join(
            f"{k} ({RETIRED_SITING_KEYS[k]})" for k in retired))
    siting.update(given)
    if siting["method"] not in ELECTROLYSER_SITING_METHODS:
        raise ValueError(f"hydrogen.electrolyser_siting.method must be one of {ELECTROLYSER_SITING_METHODS}, "
                         f"got {siting['method']!r}")
    for key in ("max_loading", "min_target_voltage_kv"):
        siting[key] = float(siting[key])
    siting["boundaries"] = [str(b) for b in (siting.get("boundaries") or [])]
    return siting


def fes_electrolyser_rows(fes: pd.DataFrame, buses: pd.DataFrame, pathway: str, year: int) -> pd.DataFrame:
    """FES networked electrolysis (``Dem_BB009``) as placement rows: site_name, bus, power_mw, area, gsp.

    GSP rows sit at their FES ``Node ID`` (resolved on the network by the FES 2050 rule: exact bus,
    same site and voltage digit, else the site's highest voltage). FES names no site for its Direct
    (transmission-connected) rows and the TEC register lists no electrolyser by 2030, so each Direct
    row is split pro rata over the placed GSP rows of its own licence area (SHETL, SPTL, NGET),
    one row per part, so the split stays visible.
    """
    from scripts.hydrogen.fes_hydrogen import electrolyser_sites

    sites = electrolyser_sites(fes, buses, pathway, int(year))
    placed = sites[sites["bus"].notna()]
    if placed.empty:
        raise ValueError(f"FES electrolysis ({pathway} {year}): no Dem_BB009 row maps to a network bus")
    rows = [{"site_name": f"electrolysis_FES_{gsp}", "bus": bus, "power_mw": float(mw), "area": area,
             "gsp": gsp, "fes_row": "gsp"}
            for gsp, bus, area, mw in zip(placed["GSP"], placed["bus"], placed["area"], placed["mw"])]
    for gsp, area, mw in zip(sites.loc[sites["bus"].isna(), "GSP"], sites.loc[sites["bus"].isna(), "area"],
                             sites.loc[sites["bus"].isna(), "mw"]):
        pool = placed[placed["area"].eq(area)]
        if pool.empty:
            pool = placed
        for pool_gsp, bus, pool_mw in zip(pool["GSP"], pool["bus"], pool["mw"]):
            rows.append({"site_name": f"electrolysis_FES_{gsp}_at_{pool_gsp}", "bus": bus,
                         "power_mw": float(mw) * float(pool_mw) / float(pool["mw"].sum()), "area": area,
                         "gsp": gsp, "fes_row": "direct"})
    return pd.DataFrame(rows)


def electrolyser_relief_geography(network: pypsa.Network, gsp_polygon_file, gsp_group_region_file,
                                  boundary_file, boundaries, locational_config: Optional[dict] = None) -> dict:
    """The CP30 ES5 region and boundary sides of every GB bus, as the locational checks read them.

    Regions are ``locational_validation.assign_bus_regions`` (``spatial_allocation.bus_cp30_regions``
    on the scenario's GSP polygons and group map, with the transmission-site overrides); sides are
    ``locational_validation.oriented_sides`` on the checks' boundary file with their remote points
    (``validation.locational.boundary_remote_points``), so a move the relief allows cannot change
    check 4's count north of a boundary. Returns ``{'bus_region', 'bus_sides'}``.
    """
    from scripts.CP30.locational_validation import _gb_mask, assign_bus_regions, oriented_sides, resolve_config
    from scripts.CP30.spatial_allocation import load_gsp_region_map
    from scripts.solve.secured_boundaries import load_boundary_geometries

    polygons, group_region = load_gsp_region_map(gsp_polygon_file, gsp_group_region_file)
    cfg = resolve_config(locational_config)
    geoms = load_boundary_geometries(boundary_file)
    missing = [b for b in boundaries if b not in geoms]
    if missing:
        raise ValueError(f"electrolyser relief: boundaries {missing} not in {boundary_file}")
    buses = network.buses
    gb = _gb_mask(buses)
    return {"bus_region": assign_bus_regions(buses, polygons, group_region),
            "bus_sides": oriented_sides(buses[gb], geoms, list(boundaries), cfg["boundary_remote_points"])}


#: Moves the import relief makes off one node before it reports the node and leaves it.
MAX_RELIEF_MOVES_PER_NODE = 25


def relieve_electrolyser_import_headroom(rows: pd.DataFrame, network: pypsa.Network, siting: dict,
                                         transmission: Optional[dict], bus_region: pd.Series, bus_sides: pd.DataFrame,
                                         logger: logging.Logger, exclusions=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move electrolysis off nodes that cannot import it with their peak load (16 Oct 2026).

    CP30's electrolysers are held at their Stage 1 schedule in the balancing stage
    (``market.balancing.hydrogen_links.hold_carriers``), so a node whose electrolysers and load
    exceed what its circuits can bring in can be balanced only by shedding load. The FES-sited
    placement put 390 MW at BUMU1Q and 888 MW at ABNE1Q/1R/3- (Perth) behind 2 x 132 MVA each,
    and 194 MW at BAIN behind 2 x 114 MVA; the calibration solve of 15 Oct shed 215 GWh there.
    The relief before (``add_storage.spread_storage_over_branch_capacity``) judged each busbar by
    the sum of its branches, the 132/33 kV transformers into its own load busbar and the
    9,999 MVA bus coupler included, so none of them moved.

    The rule: every node's peak load plus electrolysis, in every month, at most ``max_loading`` x
    its import capacity at the ratings the BM enforces (``locational_validation.DemandHeadroom``,
    the model of check 11: bus-section nodes with their substation's lower-voltage busbars, the
    DC share of their withdrawal on each circuit, the first circuit to reach its rating; voltage
    floor, rating overrides, seasonal ratings). A withdrawal at ABNE goes 68% via CHAR1- and
    BAIN1Q takes 97% of its own on its own circuit, so the sum of a node's ratings overstates what
    it can import. ``max_loading`` is at most 1.0 (code default 1.0; CP30 sets 0.9 since 16 Oct
    2026, a margin for through-flow), not the 1.5 of
    generation and storage weak-bus relief (``network.spatial_mapping.weak_bus_relief``): that margin is
    diversity (wind, solar and batteries are seldom all at full output together), and a held
    electrolyser has none: Stage 1 runs it flat out in cheap hours, peak-load hours included.
    Storage charging is not counted: the BM prices storage and can stop it charging (at ABNE it
    discharged 150 MW into the shedding hours). No credit is taken for local generation.

    For each node over the limit (lowest voltage first, then largest excess) electrolysis moves off the node, then off
    the substation's lower-voltage busbars that pass demand into it (largest rows first, each row
    by exactly the MW that clears the excess, given the share of its demand that reaches the node),
    to the first of:

    1. a higher-voltage node of the same substation (four-character site code), highest first;
    2. the nearest node at or above ``min_target_voltage_kv`` in the same CP30 ES5 region
       (``bus_region``) and on the same side of every boundary of ``siting['boundaries']``
       (``bus_sides``), so FES's split behind B4, B6 and B8 (check 4) survives;

    taking at most the target's headroom (least over the months) and accepting a move only when
    no other node then exceeds its limit by more than before. A row split in two keeps its labels;
    the moved part is named ``{site_name}_relief_{bus}`` (the original site name and the final bus). ``exclusions`` (buses, e.g.
    ``spatial_utils.generation_placement_exclusions``) are never targets. Returns (rows, moves).
    """
    from scripts.CP30.locational_validation import DemandHeadroom, NetworkTables

    k = float(siting["max_loading"])
    min_kv = float(siting["min_target_voltage_kv"])
    boundaries = [b for b in siting.get("boundaries", []) if b in bus_sides.columns]
    tol = 1e-3
    tables = NetworkTables.from_network(network)
    model = DemandHeadroom(tables, transmission)
    buses = network.buses
    v_nom = pd.to_numeric(buses["v_nom"], errors="coerce").fillna(0.0)
    xy = buses[["x", "y"]].astype(float)
    allowed = pd.Index(model.gb_buses).difference(pd.Index(list(exclusions)))
    result = rows.reset_index(drop=True).copy()
    result["power_mw"] = pd.to_numeric(result["power_mw"], errors="coerce").fillna(0.0)
    sides = bus_sides.reindex(columns=boundaries)
    moves, stuck, moves_here = [], set(), {}

    def state():
        el = result.groupby("bus")["power_mw"].sum()
        worst = model.worst(el, max_loading=k, all_nodes=True)
        return worst, model.group

    worst, group = state()
    node_v = v_nom.groupby(group).max()
    for _ in range(1000):
        over = worst[(worst["excess_mw"] > tol) & ~worst.index.isin(list(stuck))]
        if over.empty:
            break
        # Lowest voltage first: a higher-voltage node of the same substation carries the same excess in its
        # pocket, and relieving the busbar the demand sits on clears both. Then the largest excess.
        node = min(over.index, key=lambda n: (node_v.get(n, 0.0), -float(over.at[n, "excess_mw"]), str(n)))
        excess = float(over.at[node, "excess_mw"])
        # Rows at the node or on its substation's lower-voltage busbars, by how much of a MW moved
        # comes off the node's binding circuit (its own buses first), then largest first.
        behind = set(model.model.behind[node])
        binding, binding_share = over.at[node, "binding"], over.at[node, "binding_share"]
        moves_here[node] = moves_here.get(node, 0) + 1
        if moves_here[node] > MAX_RELIEF_MOVES_PER_NODE:
            stuck.add(node)
            logger.warning(f"Electrolyser import relief: {node} still {excess:,.2f} MW over after "
                           f"{MAX_RELIEF_MOVES_PER_NODE} moves; left")
            continue
        share = {}
        for i in result.index[result["bus"].isin(behind) & result["power_mw"].gt(tol)]:
            e = model.effect(node, binding, binding_share, result.at[i, "bus"])
            if e > 0.01:
                share[i] = e
        order = sorted(share, key=lambda i: (-round(share[i], 3), -result.at[i, "power_mw"], i))
        moved = False
        for idx in order:
            src = result.at[idx, "bus"]
            # 0.01 MW over the excess, so the node ends strictly within its limit.
            amount = min(float(result.at[idx, "power_mw"]), excess / share[idx] + 0.01)
            # A higher-voltage node of the same substation may itself show the excess (its pocket holds
            # this node's demand); the trial below accepts the move only if no node ends worse.
            same_site = [n for n in node_v.index if str(n)[:4] == str(node)[:4] and node_v[n] > node_v[node]]
            same_site = sorted(same_site, key=lambda n: (-node_v[n], str(n)))
            region = bus_region.get(src)
            near = allowed[(v_nom.reindex(allowed) >= min_kv).to_numpy()]
            if region is not None and not pd.isna(region):
                near = near[(bus_region.reindex(near) == region).to_numpy()]
            if boundaries and src in sides.index:
                near = near[(sides.reindex(near) == sides.loc[src]).all(axis=1).to_numpy()]
            dist = np.hypot(xy.loc[near, "x"] - xy.at[src, "x"], xy.loc[near, "y"] - xy.at[src, "y"]) / 1000.0
            near_nodes = []
            for bus in dist.sort_values(kind="stable").index:
                n = group.get(bus)
                if n is not None and n != node and n not in over.index and n not in [m for m, _ in near_nodes]:
                    near_nodes.append((n, bus))
            candidates = [(n, sorted(str(b) for b in group.index[group.eq(n)] if b in allowed)[0], "same_substation")
                          for n in same_site if any(b in allowed for b in group.index[group.eq(n)])]
            candidates += [(n, bus, f"nearest_{dist[bus]:.0f}km") for n, bus in near_nodes]
            for target_node, target_bus, method in candidates:
                room = float(worst.at[target_node, "headroom_mw"]) if target_node in worst.index else 0.0
                x = amount if method == "same_substation" else min(amount, room)
                if not x > tol:
                    continue
                before = result.copy()
                moved_name = f"{str(result.at[idx, 'site_name']).split('_relief_')[0]}_relief_{target_bus}"
                if x >= float(result.at[idx, "power_mw"]) - tol:
                    result.at[idx, "bus"] = target_bus
                    result.at[idx, "site_name"] = moved_name
                else:
                    piece = result.loc[idx].copy()
                    piece["bus"], piece["power_mw"] = target_bus, x
                    piece["site_name"] = moved_name
                    result.at[idx, "power_mw"] = float(result.at[idx, "power_mw"]) - x
                    result.loc[len(result)] = piece
                trial, _ = state()
                if float(trial.at[node, "excess_mw"]) >= excess - tol:
                    result = before
                    if method == "same_substation":
                        continue             # that busbar's withdrawal loads the same circuit; try further away
                    break                    # this row's demand does not reach the node: no target helps
                worse = trial["excess_mw"].sub(worst["excess_mw"].reindex(trial.index).fillna(0.0)) > tol
                worse[node] = False
                if worse.any():
                    result = before
                    continue
                moves.append({"site_name": str(before.at[idx, "site_name"]), "capacity_mw": x,
                              "from_bus": src, "from_node": node, "to_bus": target_bus, "to_node": target_node,
                              "method": method, "region": region,
                              "from_kv": float(v_nom.get(src, np.nan)), "to_kv": float(v_nom.get(target_bus, np.nan))})
                worst = trial
                moved = True
                break
            if moved:
                break
        if not moved:
            stuck.add(node)
            logger.warning(
                f"Electrolyser import relief: {node} still {excess:,.1f} MW over {k:.2f} x its import capacity "
                "(no electrolysis to move, or no target in its region and boundary sides with headroom)")
    moves = pd.DataFrame(moves)
    if len(moves):
        logger.info(f"Electrolyser import relief: {moves['capacity_mw'].sum():,.1f} MW moved off "
                    f"{moves['from_node'].nunique()} node(s) above {k:.2f} x their import capacity at BM ratings")
        for (frm, to, method), mw in moves.groupby(["from_bus", "to_bus", "method"])["capacity_mw"].sum().items():
            logger.info(f"  {frm} -> {to} [{method}]: {mw:,.1f} MW")
    return result, moves


def place_fes_electrolysers(network: pypsa.Network, fes: pd.DataFrame, pathway: str, year: int,
                            total_mw: float, siting: dict, logger: logging.Logger,
                            relief: Optional[dict] = None) -> pd.Series:
    """Electrolyser MW by bus on the FES GSP basis, scaled to ``total_mw`` (the CP30 capacity), exact.

    With ``siting['max_loading'] > 0`` the import relief runs (:func:`relieve_electrolyser_import_headroom`);
    ``relief`` gives it ``transmission`` (the scenario's block), ``bus_region`` and ``bus_sides``
    (:func:`electrolyser_relief_geography`) and optionally ``exclusions``.
    """
    rows = fes_electrolyser_rows(fes, network.buses, pathway, year)
    fes_total = float(rows["power_mw"].sum())
    rows["power_mw"] *= float(total_mw) / fes_total
    rows["technology"] = "electrolysis"
    logger.info(
        f"Electrolyser siting fes_gsp: FES {pathway} {year} Dem_BB009 {fes_total:,.1f} MW scaled to "
        f"{total_mw:,.1f} MW (x{total_mw / fes_total:.6f}); by licence area "
        + ", ".join(f"{a} {v:,.0f}" for a, v in rows.groupby("area")["power_mw"].sum().items())
    )
    if siting["max_loading"] > 0:
        if not relief or "bus_region" not in relief or "bus_sides" not in relief:
            raise ValueError("electrolyser siting max_loading > 0 needs the relief geography (bus_region, bus_sides; "
                             "electrolyser_relief_geography) and the scenario's transmission block")
        rows, _ = relieve_electrolyser_import_headroom(
            rows, network, siting, relief.get("transmission"), relief["bus_region"], relief["bus_sides"], logger,
            exclusions=relief.get("exclusions", ()),
        )
    by_bus = rows.groupby("bus")["power_mw"].sum()
    by_bus = by_bus[by_bus > 0.0]
    if not np.isclose(by_bus.sum(), total_mw, rtol=1e-9):
        raise ValueError(f"electrolyser siting changed the total ({by_bus.sum():,.3f} vs {total_mw:,.3f} MW)")
    logger.info("Electrolyser siting: largest buses " + ", ".join(
        f"{b} {v:,.0f}" for b, v in by_bus.sort_values(ascending=False).head(10).items()))
    return by_bus


def resolve_turbine_settings(hydrogen_config: Optional[dict]) -> dict:
    """``hydrogen.turbine_fuel`` and ``hydrogen.turbine_variable_om`` with defaults filled and validated.

    Returns ``fuel_source``, ``fuel_price_gbp_per_mwh_h2`` (None unless the fuel is
    an external supply), ``vom_basis``, ``vom_gbp_per_mwh`` and ``availability``
    (``hydrogen.turbine_availability``, the links' p_max_pu). An empty config
    gives the build before item OW3.
    """
    cfg = dict(hydrogen_config or {})
    fuel = dict(cfg.get("turbine_fuel", {}) or {})
    source = str(fuel.get("source", DEFAULT_TURBINE_FUEL_SOURCE))
    if source not in TURBINE_FUEL_SOURCES:
        raise ValueError(f"hydrogen.turbine_fuel.source must be one of {TURBINE_FUEL_SOURCES}, got {source!r}")
    price = fuel.get("price_gbp_per_mwh_h2")
    if source == "external_supply":
        if price is None:
            raise ValueError("hydrogen.turbine_fuel.source external_supply needs price_gbp_per_mwh_h2")
        price = float(price)
        if not price >= 0.0:
            raise ValueError(f"hydrogen.turbine_fuel.price_gbp_per_mwh_h2 must be >= 0, got {price}")
    else:
        price = None
    vom = dict(cfg.get("turbine_variable_om", {}) or {})
    basis = str(vom.get("basis", DEFAULT_TURBINE_VOM_BASIS))
    if basis not in TURBINE_VOM_BASES:
        raise ValueError(f"hydrogen.turbine_variable_om.basis must be one of {TURBINE_VOM_BASES}, got {basis!r}")
    value = float(vom.get("gbp_per_mwh", H2_TURBINE_MARGINAL_COST))
    if not value >= 0.0:
        raise ValueError(f"hydrogen.turbine_variable_om.gbp_per_mwh must be >= 0, got {value}")
    availability = float(cfg.get("turbine_availability", DEFAULT_TURBINE_AVAILABILITY))
    if not 0.0 < availability <= 1.0:
        raise ValueError(f"hydrogen.turbine_availability must be in (0, 1], got {availability}")
    return {
        "fuel_source": source,
        "fuel_price_gbp_per_mwh_h2": price,
        "vom_basis": basis,
        "vom_gbp_per_mwh": value,
        "availability": availability,
    }


def turbine_link_marginal_cost(vom_gbp_per_mwh: float, basis: str, efficiency: float) -> float:
    """The turbine link's ``marginal_cost``, GBP per MWh of p0 (hydrogen in).

    ``electricity_output``: the VOM is per MWh of electricity, and one MWh of
    p0 makes ``efficiency`` MWh of it. ``hydrogen_input``: the VOM sits on p0 as given.
    """
    if basis == "electricity_output":
        return float(vom_gbp_per_mwh) * float(efficiency)
    if basis == "hydrogen_input":
        return float(vom_gbp_per_mwh)
    raise ValueError(f"turbine VOM basis must be one of {TURBINE_VOM_BASES}, got {basis!r}")


def load_fes_electrolysis_capacity(
    fes_file: str,
    year: int,
    fes_scenario: str,
    logger: logging.Logger
) -> float:
    """
    Load electrolysis capacity from FES data.
    
    Args:
        fes_file: Path to FES data CSV
        year: Modelled year
        fes_scenario: FES pathway name (e.g., "Holistic Transition")
        logger: Logger instance
        
    Returns:
        Total electrolysis capacity in MW
    """
    logger.info(f"Loading FES electrolysis data for {fes_scenario} {year}")
    
    fes = pd.read_csv(fes_file)
    year_col = str(year)
    
    if year_col not in fes.columns:
        logger.warning(f"Year {year} not in FES data, using nearest available")
        year_cols = [c for c in fes.columns if c.isdigit()]
        year_col = min(year_cols, key=lambda x: abs(int(x) - year))
        logger.info(f"Using year column: {year_col}")
    
    # Filter for electrolysis in the specified pathway
    electrolysis = fes[
        (fes['Technology Detail'] == 'Hydrogen electrolysis') &
        (fes['FES Pathway'] == fes_scenario)
    ]
    
    if len(electrolysis) == 0:
        logger.warning(f"No electrolysis data found for {fes_scenario}")
        return 0.0
    
    total_capacity = electrolysis[year_col].sum()
    logger.info(f"FES electrolysis capacity for {fes_scenario} {year}: {total_capacity:,.1f} MW")
    
    return total_capacity


def add_hydrogen_system(
    network: pypsa.Network,
    electrolysis_capacity_mw: float,
    h2_generation_capacity_mw: float,
    logger: logging.Logger,
    exogenous_h2_demand_mwh: float = 0.0,
    turbine_settings: Optional[dict] = None,
    electrolyser_mw_by_bus: Optional[pd.Series] = None,
) -> pypsa.Network:
    """
    Add hydrogen system components to the network.

    Creates a simplified "copper-plate" hydrogen network:
    - Single GB-wide H2 bus
    - Electrolysis links from electricity buses to H2 bus
    - H2 storage at the H2 bus
    - H2 power generation links from H2 bus to electricity buses (or, with an
      external turbine fuel supply, from their own priced fuel bus)

    Args:
        network: PyPSA network
        electrolysis_capacity_mw: Total electrolysis capacity (MW)
        h2_generation_capacity_mw: Total H2 generation capacity (MW)
        logger: Logger instance
        exogenous_h2_demand_mwh: Annual flat hydrogen offtake on the H2 bus (MWh_H2)
        turbine_settings: ``resolve_turbine_settings`` output; None gives the
            code defaults (turbines on GB_H2, VOM on p0)
        electrolyser_mw_by_bus: electrolyser MW by electricity bus (``place_fes_electrolysers``);
            None places ``electrolysis_capacity_mw`` pro rata to H2-turbine capacity (the default)

    Returns:
        Modified network with hydrogen system
    """
    turbine_settings = turbine_settings or resolve_turbine_settings({})
    logger.info("=" * 60)
    logger.info("ADDING HYDROGEN SYSTEM")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Add hydrogen carrier
    # -------------------------------------------------------------------------
    if 'H2_gas' not in network.carriers.index:
        network.add("Carrier", "H2_gas", 
                    color="#FF69B4",  # Hot pink for hydrogen gas
                    nice_name="Hydrogen Gas",
                    co2_emissions=0.0)
        logger.info("Added H2_gas carrier")
    
    if 'electrolysis' not in network.carriers.index:
        network.add("Carrier", "electrolysis",
                    color="#8A2BE2",  # Blue violet
                    nice_name="Electrolysis",
                    co2_emissions=0.0)
        logger.info("Added electrolysis carrier")
    
    if 'H2_turbine' not in network.carriers.index:
        network.add("Carrier", "H2_turbine",
                    color="#FF1493",  # Deep pink
                    nice_name="H2 Power Generation",
                    co2_emissions=0.0)
        logger.info("Added H2_turbine carrier")
    
    # -------------------------------------------------------------------------
    # Step 2: Find existing H2 generators and their buses
    # -------------------------------------------------------------------------
    h2_generators = network.generators[network.generators.carrier.isin(['H2', 'Hydrogen'])].copy()
    
    if len(h2_generators) == 0:
        logger.warning("No H2 generators found in network - skipping hydrogen system")
        return network
    
    logger.info(f"Found {len(h2_generators)} H2 generators with {h2_generators.p_nom.sum():,.1f} MW total capacity")
    
    # Get unique electricity buses where H2 generators are located
    h2_buses = h2_generators['bus'].unique()
    logger.info(f"H2 generators located at {len(h2_buses)} unique electricity buses")
    
    # -------------------------------------------------------------------------
    # Step 3: Create single GB-wide hydrogen bus
    # -------------------------------------------------------------------------
    h2_bus_name = "GB_H2"
    
    # Use centroid of H2 generator locations for visualization
    h2_gen_coords = network.buses.loc[h2_buses, ['x', 'y']].mean()
    
    network.add("Bus", h2_bus_name,
                carrier="H2_gas",
                x=h2_gen_coords['x'],
                y=h2_gen_coords['y'],
                v_nom=1.0)  # Notional voltage for hydrogen
    
    logger.info(f"Added hydrogen bus: {h2_bus_name}")
    
    # -------------------------------------------------------------------------
    # Step 4: Add electrolysis links (electricity → H2)
    # -------------------------------------------------------------------------
    if electrolyser_mw_by_bus is not None:
        # Placed by the configured siting (hydrogen.electrolyser_siting, e.g. FES GSP rows).
        allocation = pd.Series(electrolyser_mw_by_bus, dtype=float)
        unknown = sorted(set(allocation.index) - set(network.buses.index))
        if unknown:
            raise ValueError(f"electrolyser siting names buses absent from the network: {unknown[:10]}")
        if not np.isclose(allocation.sum(), electrolysis_capacity_mw, rtol=1e-9):
            raise ValueError(f"electrolyser siting totals {allocation.sum():,.3f} MW, "
                             f"not the {electrolysis_capacity_mw:,.3f} MW electrolysis capacity")
        min_mw = 0.0
    else:
        # Distribute electrolysis capacity proportionally to H2 generator capacity at each bus
        h2_gen_by_bus = h2_generators.groupby('bus')['p_nom'].sum()
        allocation = electrolysis_capacity_mw * h2_gen_by_bus / h2_gen_by_bus.sum()
        min_mw = 0.1  # Skip tiny allocations

    electrolysis_added = 0
    for elec_bus, elec_capacity in allocation.items():
        if elec_capacity < min_mw or elec_capacity <= 0.0:
            continue
        
        link_name = f"electrolysis_{elec_bus}"
        
        network.add("Link", link_name,
                    bus0=elec_bus,          # Electricity input
                    bus1=h2_bus_name,       # Hydrogen output
                    carrier="electrolysis",
                    efficiency=ELECTROLYSIS_EFFICIENCY,
                    p_nom=elec_capacity,
                    p_nom_extendable=False,
                    marginal_cost=ELECTROLYSIS_MARGINAL_COST,
                    capital_cost=0.0)
        
        electrolysis_added += elec_capacity
    
    logger.info(f"Added {len(network.links[network.links.carrier == 'electrolysis'])} electrolysis links")
    logger.info(f"Total electrolysis capacity: {electrolysis_added:,.1f} MW")
    
    # -------------------------------------------------------------------------
    # Step 5: Add hydrogen storage
    # -------------------------------------------------------------------------
    # Size storage for H2_STORAGE_HOURS of full H2 generation. The capacity is
    # electrical, so the hydrogen it burns in an hour is capacity / efficiency.
    h2_storage_energy = h2_generation_capacity_mw / H2_TURBINE_EFFICIENCY * H2_STORAGE_HOURS  # MWh (H2 LHV)
    
    network.add("Store", "GB_H2_storage",
                bus=h2_bus_name,
                carrier="H2_gas",
                e_nom=h2_storage_energy,
                e_nom_extendable=False,
                e_cyclic=True,  # SOC at end = SOC at start
                standing_loss=H2_STORAGE_STANDING_LOSS,
                marginal_cost=0.0,
                capital_cost=0.0)
    
    logger.info(f"Added H2 storage: {h2_storage_energy:,.0f} MWh ({H2_STORAGE_HOURS}h of generation)")

    # -------------------------------------------------------------------------
    # Step 5b: Exogenous hydrogen offtake (industry, transport, heat)
    # -------------------------------------------------------------------------
    # Without an external sink, electrolysis only runs to refuel the H2
    # turbines, so a pathway's electrolysis demand is never reached. The
    # offtake is a flat Load on the H2 bus in MW of hydrogen (LHV).
    if exogenous_h2_demand_mwh and exogenous_h2_demand_mwh > 0.0:
        if 'H2_demand' not in network.carriers.index:
            network.add("Carrier", "H2_demand", color="#C71585",
                        nice_name="Hydrogen offtake", co2_emissions=0.0)
        offtake_mw = float(exogenous_h2_demand_mwh) / 8760.0
        network.add("Load", "GB_H2_exogenous_demand",
                    bus=h2_bus_name,
                    carrier="H2_demand",
                    p_set=offtake_mw)
        logger.info(
            f"Added exogenous H2 offtake: {exogenous_h2_demand_mwh / 1e6:,.2f} TWh_H2/yr "
            f"({offtake_mw:,.0f} MW_H2 flat), needing "
            f"{exogenous_h2_demand_mwh / ELECTROLYSIS_EFFICIENCY / 1e6:,.2f} TWh of electricity"
        )
    
    # -------------------------------------------------------------------------
    # Step 6: Convert H2 generators to Links (H2 → electricity)
    # -------------------------------------------------------------------------
    # Remove existing H2 generators and replace with links
    h2_gen_data = h2_generators.copy()
    
    # Remove the generators
    network.remove("Generator", h2_generators.index.tolist())
    logger.info(f"Removed {len(h2_generators)} H2 generators (converting to links)")
    
    # Turbine fuel (item OW3). external_supply: the turbines take hydrogen off
    # their own bus, fed by a full tank priced per MWh drawn. The bus carries
    # H2_gas so the BM still treats the turbines as hydrogen links (held at
    # Stage 1, market.balancing.hydrogen_links), and nothing links it to GB_H2,
    # so the fuel can neither come from electrolysis nor serve the offtake.
    fuel_source = turbine_settings["fuel_source"]
    turbine_bus0 = h2_bus_name
    if fuel_source == "external_supply":
        turbine_bus0 = H2_POWER_FUEL_BUS
        if H2_POWER_FUEL_CARRIER not in network.carriers.index:
            network.add("Carrier", H2_POWER_FUEL_CARRIER, color="#DB7093",
                        nice_name="Hydrogen for power (bought)", co2_emissions=0.0)
        network.add("Bus", H2_POWER_FUEL_BUS,
                    carrier="H2_gas",
                    x=h2_gen_coords['x'],
                    y=h2_gen_coords['y'],
                    v_nom=1.0)
    turbine_mc = turbine_link_marginal_cost(
        turbine_settings["vom_gbp_per_mwh"], turbine_settings["vom_basis"], H2_TURBINE_EFFICIENCY
    )
    turbine_availability = float(turbine_settings.get("availability", DEFAULT_TURBINE_AVAILABILITY))

    # Add as links (H2 → electricity). A Link's p_nom limits its bus0 flow,
    # which here is hydrogen in, so the generator's electrical capacity is
    # divided by the efficiency; output at full load is then that capacity.
    for idx, gen in h2_gen_data.iterrows():
        link_name = f"H2_turbine_{idx}"

        network.add("Link", link_name,
                    bus0=turbine_bus0,     # Hydrogen input
                    bus1=gen['bus'],       # Electricity output
                    carrier="H2_turbine",
                    efficiency=H2_TURBINE_EFFICIENCY,
                    p_nom=gen['p_nom'] / H2_TURBINE_EFFICIENCY,   # MW of hydrogen in
                    p_nom_extendable=False,
                    p_max_pu=turbine_availability,
                    marginal_cost=turbine_mc,
                    capital_cost=0.0)

    turbines = network.links[network.links.carrier == 'H2_turbine']
    h2_turbine_count = len(turbines)
    h2_turbine_capacity = (turbines['p_nom'] * turbines['efficiency']).sum()
    logger.info(f"Added {h2_turbine_count} H2 turbine links with {h2_turbine_capacity:,.1f} MW electrical capacity")
    if turbine_availability < 1.0:
        logger.info(f"H2 turbine availability {turbine_availability:.3f} (links' p_max_pu; capacity unchanged): "
                    f"{h2_turbine_capacity * turbine_availability:,.1f} MW available")
    logger.info(
        f"H2 turbine VOM {turbine_settings['vom_gbp_per_mwh']:.2f} GBP/MWh on the "
        f"{turbine_settings['vom_basis']} basis: link marginal cost {turbine_mc:.2f} GBP per MWh of hydrogen in"
    )

    if fuel_source == "external_supply":
        price = float(turbine_settings["fuel_price_gbp_per_mwh_h2"])
        intake_mw = float(turbines['p_nom'].sum())
        tank_mwh = intake_mw * H2_POWER_FUEL_TANK_HOURS
        network.add("Store", H2_POWER_FUEL_STORE,
                    bus=H2_POWER_FUEL_BUS,
                    carrier=H2_POWER_FUEL_CARRIER,
                    e_nom=tank_mwh,
                    e_nom_extendable=False,
                    e_initial=tank_mwh,
                    e_cyclic=False,
                    standing_loss=0.0,
                    marginal_cost=price,
                    capital_cost=0.0)
        hurdle = (price + turbine_mc) / H2_TURBINE_EFFICIENCY
        logger.info(
            f"H2 turbine fuel: external supply at {price:.2f} GBP/MWh_H2 on {H2_POWER_FUEL_BUS} "
            f"(tank {tank_mwh / 1e6:,.2f} TWh_H2, {H2_POWER_FUEL_TANK_HOURS} h of {intake_mw:,.1f} MW intake); "
            f"turbines in merit above {hurdle:.2f} GBP/MWh; none of the fuel is electrolysed"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("HYDROGEN SYSTEM SUMMARY")
    logger.info("-" * 60)
    logger.info(f"  Electrolysis capacity:  {electrolysis_added:,.1f} MW (η={ELECTROLYSIS_EFFICIENCY*100:.0f}%)")
    logger.info(f"  H2 storage capacity:    {h2_storage_energy:,.0f} MWh ({H2_STORAGE_HOURS}h)")
    logger.info(f"  H2 turbine capacity:    {h2_turbine_capacity:,.1f} MW (η={H2_TURBINE_EFFICIENCY*100:.0f}%)")
    logger.info(f"  Round-trip efficiency:  {ELECTROLYSIS_EFFICIENCY * H2_TURBINE_EFFICIENCY * 100:.1f}%")
    if fuel_source == "external_supply":
        logger.info(f"  H2 turbine fuel:        bought at {turbine_settings['fuel_price_gbp_per_mwh_h2']:.2f} "
                    f"GBP/MWh_H2 ({H2_POWER_FUEL_BUS}), not electrolysed")
    else:
        logger.info("  H2 turbine fuel:        the GB_H2 store (electrolysed)")
    logger.info("-" * 60)
    
    return network


def main():
    """Main entry point when run as script."""
    
    # Check if running under Snakemake
    if 'snakemake' in globals():
        # Get inputs/outputs from Snakemake
        network_file = snakemake.input.network
        fes_file = snakemake.input.fes_data
        output_file = snakemake.output.network
        
        # Get parameters
        scenario = snakemake.params.scenario
        modelled_year = snakemake.params.modelled_year
        fes_scenario = snakemake.params.fes_scenario
        is_historical = snakemake.params.is_historical
        scenario_source = getattr(snakemake.params, 'scenario_source', None)
        cp30_pathway = getattr(snakemake.params, 'cp30_pathway', None)
        cp30_es1 = getattr(snakemake.input, 'cp30_es1', None)
        cp30_ed1 = getattr(snakemake.input, 'cp30_ed1', None)
        electrolysis_full_load_hours = float(getattr(snakemake.params, 'electrolysis_full_load_hours', 4000.0))
        hydrogen_config = dict(getattr(snakemake.params, 'hydrogen_config', {}) or {})
        
        logger.info(f"Adding hydrogen system for scenario: {scenario}")
        logger.info(f"Modelled year: {modelled_year}, FES scenario: {fes_scenario}")
        
    else:
        # Standalone testing
        import argparse
        parser = argparse.ArgumentParser(description="Add hydrogen system to PyPSA network")
        parser.add_argument("--network", required=True, help="Input network file")
        parser.add_argument("--fes-data", required=True, help="FES data CSV")
        parser.add_argument("--output", required=True, help="Output network file")
        parser.add_argument("--year", type=int, default=2035, help="Modelled year")
        parser.add_argument("--fes-scenario", default="Holistic Transition", help="FES scenario")
        parser.add_argument("--scenario-source", default="FES", help="Scenario source, e.g. FES or CP30")
        parser.add_argument("--cp30-es1", default=None, help="CP30 ES1 CSV for CP30 scenarios")
        parser.add_argument("--cp30-ed1", default=None, help="CP30 ED1 CSV for CP30 scenarios")
        parser.add_argument("--cp30-pathway", default=None, help="CP30 pathway name")
        parser.add_argument("--electrolysis-full-load-hours", type=float, default=4000.0, help="Full-load hours used to convert CP30 electrolysis GWh to MW")
        args = parser.parse_args()
        
        network_file = args.network
        fes_file = args.fes_data
        output_file = args.output
        modelled_year = args.year
        fes_scenario = args.fes_scenario
        is_historical = modelled_year <= 2024
        scenario_source = args.scenario_source
        cp30_pathway = args.cp30_pathway
        cp30_es1 = args.cp30_es1
        cp30_ed1 = args.cp30_ed1
        electrolysis_full_load_hours = args.electrolysis_full_load_hours
        hydrogen_config = {}
    
    # Skip for historical scenarios (no hydrogen system in historical data)
    if is_historical:
        logger.info("Historical scenario - copying network without hydrogen system")
        import shutil
        shutil.copy(network_file, output_file)
        return
    
    # Load network
    logger.info(f"Loading network from: {network_file}")
    network = load_network(network_file)
    
    # Task Y6b (FES 2050 plan, decision 15): FES long-range scenarios build the FES
    # hydrogen system of their pathway and year (FES-sited electrolysers, FES efficiencies,
    # the WS1 store, gas reforming with CCS, annual non-power demand). hydrogen.fes is
    # absent elsewhere.
    from scripts.hydrogen.fes_hydrogen import fes_hydrogen_config
    fes_h2 = fes_hydrogen_config(hydrogen_config)
    if fes_h2.get('enabled', False):
        import json
        from scripts.hydrogen.fes_hydrogen import add_fes_hydrogen_system, future_cost_prices
        targets_file = getattr(snakemake.input, 'fes_targets', None) if 'snakemake' in globals() else None
        costs_file = getattr(snakemake.input, 'fes_future_costs', None) if 'snakemake' in globals() else None
        if not targets_file or not costs_file:
            raise ValueError("hydrogen.fes needs the fes_targets and fes_future_costs inputs")
        with open(targets_file, encoding='utf-8') as handle:
            targets = json.load(handle)
        fes = pd.read_csv(fes_file, encoding='utf-8-sig', low_memory=False)
        audit = add_fes_hydrogen_system(
            network, fes, targets, future_cost_prices(costs_file), config=fes_h2,
            pathway=fes_scenario, year=int(modelled_year), log=logger,
        )
        logger.info(f"FES {modelled_year} hydrogen audit: {audit}")
        save_network(network, output_file)
        logger.info(f"Hydrogen system integration complete (FES {modelled_year})")
        return

    # Get H2 generation capacity from network (already added by thermal generator integration)
    h2_gen_capacity = network.generators[network.generators.carrier.isin(['H2', 'Hydrogen'])]['p_nom'].sum()
    
    if h2_gen_capacity == 0:
        logger.warning("No H2 generators in network - skipping hydrogen system")
        save_network(network, output_file)
        return
    if scenario_source == "CP30":
        if not cp30_ed1 or not cp30_es1 or not cp30_pathway:
            raise ValueError("CP30 hydrogen integration requires cp30_ed1, cp30_es1 and cp30_pathway")
        electrolysis_capacity, electrolysis_source = load_or_derive_cp30_electrolysis_capacity_mw(
            cp30_ed1,
            cp30_es1,
            cp30_pathway,
            modelled_year,
            full_load_hours=electrolysis_full_load_hours,
        )
        logger.info(
            f"CP30 electrolysis capacity: {electrolysis_capacity:,.1f} MW "
            f"from {electrolysis_source}"
        )
    else:
        # Load electrolysis capacity from FES for non-CP30 future scenarios.
        electrolysis_capacity = load_fes_electrolysis_capacity(
            fes_file, modelled_year, fes_scenario, logger
        )
        if electrolysis_capacity == 0:
            logger.warning("No electrolysis capacity in FES - using 2x H2 generation as default")
            electrolysis_capacity = h2_gen_capacity * 2

    if electrolysis_capacity <= 0:
        raise ValueError("Hydrogen generation exists but electrolysis capacity is zero")    
    # Add hydrogen system
    # Exogenous hydrogen offtake (config: hydrogen.exogenous_demand)
    exogenous_h2_demand_mwh = 0.0
    exo_cfg = (hydrogen_config or {}).get('exogenous_demand', {}) or {}
    if exo_cfg.get('enabled', False):
        source = str(exo_cfg.get('source', 'cp30_es1_electrolysis_demand'))
        share = float(exo_cfg.get('share', 1.0))
        if source == 'cp30_es1_electrolysis_demand':
            if scenario_source != "CP30" or not cp30_es1 or not cp30_pathway:
                raise ValueError(
                    "hydrogen.exogenous_demand.source cp30_es1_electrolysis_demand needs a "
                    "CP30 scenario with cp30_es1 and cp30_pathway"
                )
            from scripts.CP30.assumptions import load_cp30_electrolysis_demand_gwh
            demand_gwh = load_cp30_electrolysis_demand_gwh(cp30_es1, cp30_pathway, modelled_year)
            exogenous_h2_demand_mwh = float(demand_gwh) * 1e3 * ELECTROLYSIS_EFFICIENCY * share
            logger.info(
                f"CP30 electrolysis demand {demand_gwh:,.0f} GWh_el x eta {ELECTROLYSIS_EFFICIENCY} "
                f"x share {share} -> {exogenous_h2_demand_mwh / 1e6:,.2f} TWh_H2 exogenous offtake"
            )
        elif source == 'fixed_mwh':
            exogenous_h2_demand_mwh = float(exo_cfg.get('mwh_h2', 0.0)) * share
        else:
            raise ValueError(f"Unknown hydrogen.exogenous_demand.source {source!r}")

    # Electrolyser siting (hydrogen.electrolyser_siting, 15 Oct 2026).
    siting = resolve_electrolyser_siting(hydrogen_config)
    electrolyser_mw_by_bus = None
    if siting["method"] == "fes_gsp":
        siting_fes_file = getattr(snakemake.input, 'electrolyser_fes_data', None) if 'snakemake' in globals() else None
        if isinstance(siting_fes_file, (list, tuple)):
            siting_fes_file = siting_fes_file[0] if siting_fes_file else None
        if not siting_fes_file:
            raise ValueError("hydrogen.electrolyser_siting.method fes_gsp needs the electrolyser_fes_data input")
        siting_fes = pd.read_csv(siting_fes_file, encoding='utf-8-sig', low_memory=False)
        relief = None
        if siting["max_loading"] > 0:
            # The import relief's geography: the scenario's GSP polygons and region map and the
            # locational checks' boundary file (rule inputs), read as the checks read them.
            def _one(name):
                value = getattr(snakemake.input, name, None) if 'snakemake' in globals() else None
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                if not value:
                    raise ValueError(f"electrolyser import relief needs the {name} input")
                return str(value)
            from scripts.utilities.spatial_utils import generation_placement_exclusions
            polygon_file = _one('electrolyser_gsp_polygons')
            relief = electrolyser_relief_geography(
                network, polygon_file, _one('electrolyser_gsp_group_regions'), _one('electrolyser_boundaries'),
                siting["boundaries"], getattr(snakemake.params, 'locational_config', None),
            )
            relief["transmission"] = dict(getattr(snakemake.params, 'transmission_config', {}) or {})
            relief["exclusions"] = list(generation_placement_exclusions(network, land_boundary_file=polygon_file).index)
        electrolyser_mw_by_bus = place_fes_electrolysers(
            network, siting_fes, str(siting["fes_scenario"]), int(modelled_year),
            float(electrolysis_capacity), siting, logger, relief=relief,
        )

    network = add_hydrogen_system(
        network=network,
        electrolysis_capacity_mw=electrolysis_capacity,
        h2_generation_capacity_mw=h2_gen_capacity,
        logger=logger,
        exogenous_h2_demand_mwh=exogenous_h2_demand_mwh,
        turbine_settings=resolve_turbine_settings(hydrogen_config),
        electrolyser_mw_by_bus=electrolyser_mw_by_bus,
    )
    
    # Save network
    logger.info(f"Saving network to: {output_file}")
    save_network(network, output_file)
    logger.info("Hydrogen system integration complete")


if __name__ == "__main__":
    main()

