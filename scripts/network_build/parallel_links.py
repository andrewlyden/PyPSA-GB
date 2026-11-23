"""Merge identical parallel internal DC links into one (items F3 and F14, 13 Oct 2026).

Two links joining the same pair of GB buses with the same dispatch attributes are
one asset in the LP. Kept apart, a lossless, free, two-way pair can carry any flow
pair p_A = -p_B at no cost, so the solver returns a loop: the twin Sofia poles
(``SOFW4A_SOFWD_ETYS_UPG_2025_HVDC`` and ``..._1_HVDC``, 660 MW each, efficiency
1, marginal cost 0) ran about +/-1,140 MW in every market hour, which showed in
``*_wholesale_links.csv`` and in the bound check. The merged link carries the
summed rating, so every feasible dispatch of the pair is still feasible; only the
loop goes. The F14 probe found the merge moves no price.

Only internal links are merged (no bus of an External, non-GB node), and only
when nothing about them could differ in the LP: carrier, efficiency, marginal
cost, per-unit limits, extendability and capital cost agree, and neither has a
time series. A pair drawn the other way round (bus0 and bus1 swapped) merges
only when it is symmetric and lossless, where swapping the ends changes nothing.
The kept link is the first in the network's order; ``merged_parallel_links``
lists every name merged into it.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Static attributes that must agree for two links to be the same LP asset.
DISPATCH_ATTRIBUTES = (
    "carrier", "efficiency", "marginal_cost", "marginal_cost_quadratic", "p_min_pu", "p_max_pu",
    "p_nom_extendable", "capital_cost", "committable", "ramp_limit_up", "ramp_limit_down",
)
MERGED_COLUMN = "merged_parallel_links"


def _same(a, b) -> bool:
    if isinstance(a, (float, int, np.floating, np.integer)) and isinstance(b, (float, int, np.floating, np.integer)):
        return (np.isnan(a) and np.isnan(b)) or bool(np.isclose(a, b, rtol=0.0, atol=1e-9))
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def _time_varying_links(network) -> set:
    names = set()
    for frame in network.links_t.values():
        if isinstance(frame, pd.DataFrame) and frame.shape[1]:
            names.update(frame.columns.astype(str))
    return names


def _external_buses(network) -> set:
    buses = network.buses
    external = set(buses.index[buses.index.astype(str).str.contains("External", case=False)])
    if "country" in buses.columns:
        country = buses["country"].fillna("GB").astype(str).replace("", "GB")
        external |= set(buses.index[~country.eq("GB")])
    return external


def merge_parallel_internal_links(network, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Merge identical parallel internal links in place; return one row per merge.

    The audit has ``kept``, ``merged`` (the names removed, ``;``-joined),
    ``bus0``, ``bus1``, ``p_nom_before`` (the kept link's) and ``p_nom_after``.
    """
    log = logger or LOGGER
    links = network.links
    columns = ["kept", "merged", "bus0", "bus1", "p_nom_before", "p_nom_after"]
    if len(links) < 2:
        return pd.DataFrame(columns=columns)
    external = _external_buses(network)
    varying = _time_varying_links(network)
    attributes = [a for a in DISPATCH_ATTRIBUTES if a in links.columns]
    candidates = links.index[
        ~links["bus0"].isin(external) & ~links["bus1"].isin(external)
        & ~links.index.astype(str).isin(varying)
    ]
    if "bus2" in links.columns:
        # Multi-output links (bus2 set) are converters, not plain branches.
        bus2 = links["bus2"].fillna("").astype(str).str.strip()
        candidates = candidates[bus2.reindex(candidates).eq("").to_numpy()]
    pair = pd.Series(
        [tuple(sorted((str(a), str(b)))) for a, b in zip(links.loc[candidates, "bus0"], links.loc[candidates, "bus1"])],
        index=candidates,
    )
    rows, remove = [], []
    for _, group in pair.groupby(pair, sort=False):
        names = list(group.index)
        while len(names) > 1:
            keep, rest = names[0], names[1:]
            ref = links.loc[keep]
            matched, unmatched = [], []
            for name in rest:
                row = links.loc[name]
                same = all(_same(ref[a], row[a]) for a in attributes)
                if same and row["bus0"] != ref["bus0"]:
                    # Reversed: equivalent only when two-way symmetric and lossless.
                    same = (_same(float(ref["efficiency"]), 1.0)
                            and _same(float(ref["p_min_pu"]), -float(ref["p_max_pu"]))
                            and _same(float(ref.get("marginal_cost", 0.0)), 0.0))
                (matched if same else unmatched).append(name)
            if matched:
                before = float(ref["p_nom"])
                total = before + float(links.loc[matched, "p_nom"].sum())
                network.links.at[keep, "p_nom"] = total
                for bound in ("p_nom_max", "p_nom_min"):
                    if bound in links.columns and np.isfinite(float(links.at[keep, bound])):
                        network.links.at[keep, bound] = float(links.loc[[keep] + matched, bound].sum())
                if MERGED_COLUMN not in network.links.columns:
                    network.links[MERGED_COLUMN] = ""
                network.links.at[keep, MERGED_COLUMN] = ";".join(matched)
                remove.extend(matched)
                rows.append({"kept": keep, "merged": ";".join(matched), "bus0": ref["bus0"],
                             "bus1": ref["bus1"], "p_nom_before": before, "p_nom_after": total})
            names = unmatched
    if remove:
        network.remove("Link", remove)
    audit = pd.DataFrame(rows, columns=columns)
    if len(audit):
        log.info("Merged %d identical parallel internal links into %d: %s", len(remove), len(audit),
                 "; ".join(f"{r.kept} <- {r.merged} ({r.p_nom_after:.0f} MW)" for r in audit.itertuples()))
    else:
        log.info("No identical parallel internal links to merge")
    return audit
