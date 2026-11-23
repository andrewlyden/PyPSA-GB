"""
Exact copperplate for the wholesale stage: collapse the mainland to one bus.

The wholesale market is a copperplate (single GB price). The legacy way to get
one, ``apply_copperplate_relaxation``, keeps every line and transformer and
sets its rating to 1e6 MVA. That leaves 1,913 lines and 1,622 transformers
with their DC power-flow equations in the LP, and bounds spanning 1e-3 to 1e6
MW, which is what made the wholesale solve slow and numerically fragile
(152-160 s per 24 h window with the conservative Gurobi settings that were
needed to solve it at all).

The GB network is 26 synchronous sub-networks: one mainland (2,378 buses,
carrying every load) and 25 islands (offshore hubs, KERG, the H2 bus, ...)
joined to it by DC links with finite ratings, plus the interconnector links to
the External buses. Collapsing *everything* onto one bus changes the economics
because the link limits vanish. Collapsing *only the mainland* is exact: with
infinite internal ratings every mainland bus already shares one price, so the
mainland is a single node in the relaxed LP too. Islands, External buses and
every link keep their identity and limits. Measured on one window of the 1-week
network: objective -23,439,407 collapsed vs -23,439,417 relaxed (4e-7
relative), 16-18 s instead of 152-160 s.

Pipeline (``solve_wholesale.py``): run every generator-level pre-solve step on
the original network (``apply_load_shedding_limits`` caps load shedding to
same-bus demand, so it must see the real buses), then ``collapse_mainland``,
solve the rolling windows on the collapsed network, and ``expand_results`` back
onto the original so ``{scenario}_wholesale.nc`` keeps its shape for
``analyze_market.py`` and the bill chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

MAINLAND_HUB = "GB_mainland"
EXTERNAL_BUS_PATTERN = "External"
REBUSED_COMPONENTS = ("generators", "loads", "storage_units", "stores", "shunt_impedances")


@dataclass
class CollapseMapping:
    """What ``collapse_mainland`` did, so ``expand_results`` can undo it."""

    hub: str
    mainland_buses: List[str]
    removed_lines: List[str]
    removed_transformers: List[str]
    removed_links: List[str]
    rebused: Dict[str, Dict[str, str]] = field(default_factory=dict)  # component -> {name: original bus}
    n_sub_networks: int = 0

    @property
    def mainland_bus_set(self) -> Set[str]:
        return set(self.mainland_buses)


def _gb_bus_mask(network, external_pattern: str) -> pd.Series:
    return ~network.buses.index.to_series().str.contains(external_pattern, regex=False)


def identify_mainland(network, external_pattern: str = EXTERNAL_BUS_PATTERN) -> Tuple[Set[str], int]:
    """Return the bus names of the largest synchronous sub-network and the sub-network count.

    Sub-networks are defined by passive branches (lines and transformers) only;
    links do not join them. External (foreign) buses are excluded from the
    choice of the largest sub-network.
    """
    network.determine_network_topology()
    gb = _gb_bus_mask(network, external_pattern)
    counts = network.buses.loc[gb, "sub_network"].value_counts()
    if counts.empty:
        raise ValueError("No non-External buses found; cannot identify the mainland")
    main_id = counts.idxmax()
    main_buses = set(network.buses.index[(network.buses["sub_network"] == main_id) & gb])
    return main_buses, int(counts.size)


def collapse_mainland(
    network,
    logger: Optional[logging.Logger] = None,
    hub: str = MAINLAND_HUB,
    external_pattern: str = EXTERNAL_BUS_PATTERN,
):
    """Return ``(collapsed, mapping)``: a copy with the mainland collapsed onto one bus.

    Steps: find the largest sub-network among non-External buses; add bus
    ``hub`` (v_nom 400, carrier AC, country GB, coordinates at the mainland
    mean); re-bus generators, loads, storage units, stores and shunts on
    mainland buses to the hub; remove links with both ends on the mainland
    (internal HVDC is redundant on one bus) and re-bus the mainland end of
    every other link; remove mainland lines and transformers; remove the
    mainland buses. The input network is not modified.
    """
    logger = logger or logging.getLogger(__name__)
    n = network.copy()
    if hub in n.buses.index:
        raise ValueError(f"Bus {hub!r} already exists; the network looks collapsed already")

    main_buses, n_sub = identify_mainland(n, external_pattern)
    main_list = sorted(main_buses)
    main_df = n.buses.loc[main_list]

    hub_kwargs = dict(v_nom=400.0, carrier="AC")
    for coord in ("x", "y", "lon", "lat"):
        if coord in main_df.columns:
            values = pd.to_numeric(main_df[coord], errors="coerce")
            if values.notna().any():
                hub_kwargs[coord] = float(values.mean())
    if "country" in main_df.columns:
        hub_kwargs["country"] = "GB"
    n.add("Bus", hub, **hub_kwargs)

    rebused: Dict[str, Dict[str, str]] = {}
    for comp in REBUSED_COMPONENTS:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or "bus" not in df.columns:
            continue
        mask = df["bus"].isin(main_buses)
        if mask.any():
            rebused[comp] = df.loc[mask, "bus"].to_dict()
            df.loc[mask, "bus"] = hub

    links = n.links
    both = links["bus0"].isin(main_buses) & links["bus1"].isin(main_buses)
    removed_links = list(links.index[both])
    if removed_links:
        internal = links.loc[removed_links]
        costed = internal.index[(pd.to_numeric(internal["marginal_cost"], errors="coerce").fillna(0.0) != 0.0)]
        if len(costed):
            logger.warning(
                f"Collapse removes {len(costed)} mainland-internal links with a non-zero "
                f"marginal cost (their flow would be zero on one bus anyway): {list(costed)[:10]}"
            )
        n.remove("Link", removed_links)
    links = n.links
    for end in ("bus0", "bus1"):
        mask = links[end].isin(main_buses)
        if mask.any():
            rebused.setdefault(f"links.{end}", links.loc[mask, end].to_dict())
            links.loc[mask, end] = hub

    lines = n.lines
    removed_lines = list(lines.index[lines["bus0"].isin(main_buses) & lines["bus1"].isin(main_buses)])
    if removed_lines:
        n.remove("Line", removed_lines)
    trafos = n.transformers
    removed_trafos = list(trafos.index[trafos["bus0"].isin(main_buses) & trafos["bus1"].isin(main_buses)])
    if removed_trafos:
        n.remove("Transformer", removed_trafos)

    for comp, name in (("lines", "Line"), ("transformers", "Transformer")):
        df = getattr(n, comp)
        touching = df["bus0"].isin(main_buses) | df["bus1"].isin(main_buses)
        if touching.any():
            raise AssertionError(
                f"{int(touching.sum())} {name}s still touch a mainland bus after the collapse; "
                "a passive branch cannot cross a sub-network boundary, so the topology is inconsistent"
            )
    for comp in REBUSED_COMPONENTS:
        df = getattr(n, comp, None)
        if df is not None and len(df) and "bus" in df.columns and df["bus"].isin(main_buses).any():
            raise AssertionError(f"{comp} still reference a mainland bus after the collapse")

    n.remove("Bus", main_list)
    n.determine_network_topology()

    mapping = CollapseMapping(
        hub=hub,
        mainland_buses=main_list,
        removed_lines=removed_lines,
        removed_transformers=removed_trafos,
        removed_links=removed_links,
        rebused=rebused,
        n_sub_networks=n_sub,
    )
    logger.info("=" * 80)
    logger.info("EXACT COPPERPLATE: MAINLAND COLLAPSED TO ONE BUS")
    logger.info("=" * 80)
    logger.info(
        f"{n_sub} synchronous sub-networks; mainland {len(main_list)} buses -> bus {hub!r}; "
        f"removed {len(removed_lines)} lines, {len(removed_trafos)} transformers, "
        f"{len(removed_links)} mainland-internal links"
    )
    logger.info(
        f"Collapsed network: {len(n.buses)} buses, {len(n.lines)} lines, "
        f"{len(n.transformers)} transformers, {len(n.links)} links (islands and "
        f"interconnectors keep their ratings), {len(n.generators)} generators, "
        f"{len(n.storage_units)} storage units, {len(n.loads)} loads"
    )
    try:
        n.consistency_check()
    except Exception as exc:  # pragma: no cover - PyPSA raises only on hard errors
        logger.warning(f"consistency_check after collapse: {exc}")
    logger.info("=" * 80)
    return n, mapping


def _assign_by_name(original, collapsed, comp: str, attr: str) -> bool:
    """Copy ``collapsed.<comp>_t.<attr>`` onto ``original`` by component name (missing -> 0)."""
    src = getattr(collapsed, f"{comp}_t")
    frame = src.get(attr) if hasattr(src, "get") else getattr(src, attr, None)
    if frame is None or len(frame) == 0 or frame.shape[1] == 0:
        return False
    index = getattr(original, comp).index
    getattr(original, f"{comp}_t")[attr] = frame.reindex(columns=index, fill_value=0.0)
    return True


def expand_results(original, collapsed, mapping: CollapseMapping, logger: Optional[logging.Logger] = None) -> None:
    """Write the collapsed network's solve results onto the original network.

    Generator, storage, store and link dispatch map by name (links removed by
    the collapse get zero columns). Every mainland bus takes the hub's marginal
    price, island and External buses keep their own, so the system price (mean
    over GB load buses) keeps its meaning. Removed lines and transformers get
    zero flow (they carried meaningless flows at 1e6 MVA before). The objective
    is copied when the collapsed network has one.
    """
    logger = logger or logging.getLogger(__name__)
    if not original.snapshots.equals(collapsed.snapshots):
        raise ValueError("Original and collapsed networks have different snapshots; cannot expand results")

    written = []
    for comp, attrs in (
        ("generators", ("p",)),
        ("storage_units", ("p", "p_dispatch", "p_store", "state_of_charge", "spill")),
        ("stores", ("p", "e")),
        ("links", ("p0", "p1")),
        ("lines", ("p0", "p1")),
        ("transformers", ("p0", "p1")),
    ):
        for attr in attrs:
            if _assign_by_name(original, collapsed, comp, attr):
                written.append(f"{comp}_t.{attr}")

    # Removed passive branches carry zero flow. When the collapsed network kept
    # no line (or transformer) at all, nothing above wrote the frame, so fill
    # it here rather than leave the original's flows empty after a solve.
    solved = len(collapsed.generators_t.p) > 0 and collapsed.generators_t.p.shape[1] > 0
    for comp in ("lines", "transformers"):
        index = getattr(original, comp).index
        if solved and len(index) and f"{comp}_t.p0" not in written:
            getattr(original, f"{comp}_t")["p0"] = pd.DataFrame(
                0.0, index=original.snapshots, columns=index
            )
            written.append(f"{comp}_t.p0 (zeros)")

    prices = collapsed.buses_t.marginal_price
    if len(prices) > 0 and prices.shape[1] > 0:
        expanded = prices.reindex(columns=original.buses.index)
        if mapping.hub in prices.columns:
            hub_price = prices[mapping.hub].to_numpy()
            main_cols = [b for b in mapping.mainland_buses if b in expanded.columns]
            expanded.loc[:, main_cols] = np.repeat(hub_price[:, None], len(main_cols), axis=1)
        missing = expanded.columns[expanded.isna().all()]
        if len(missing):
            logger.warning(f"{len(missing)} buses have no price after expansion; set to 0: {list(missing)[:10]}")
        original.buses_t.marginal_price = expanded.fillna(0.0)
        written.append("buses_t.marginal_price")

    if hasattr(collapsed, "_objective") and getattr(collapsed, "_objective", None) is not None:
        original._objective = collapsed._objective
        if hasattr(collapsed, "_objective_constant"):
            original._objective_constant = collapsed._objective_constant

    logger.info(
        f"Expanded collapsed results onto the original network ({len(original.buses)} buses): "
        f"{', '.join(written)}"
    )
