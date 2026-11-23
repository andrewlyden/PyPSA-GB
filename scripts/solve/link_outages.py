"""Declared outages of DC links in a historical year (network.transmission.declared_link_outages).

A link that was out of service for a documented period carries no flow in that period: ``p_max_pu`` and
``p_min_pu`` are set to zero for the snapshots inside it. Each entry names its source. First use: the
Western HVDC link (FLIB41_HUCS4-_0) tripped on 10 January 2020 and returned on 8 February 2020 (Ofgem's
investigation of the link; Cornwall Insight's record constraint cost of GBP 30.9m), but the 2020 validation
had it at its full 2,250 MW (19 Nov 2026).

Entry: ``{link: <name>, start: 'YYYY-MM-DD[ HH:MM]', end: 'YYYY-MM-DD[ HH:MM]', source: <text>}``; ``end`` is
exclusive. A link missing from the network is refused, so a typo cannot pass silently.
"""

from __future__ import annotations

import logging

import pandas as pd


def apply_declared_link_outages(network, scenario_config: dict, logger: logging.Logger) -> int:
    """Zero the declared links' flow limits inside their outage windows. Returns the entries applied."""
    transmission = ((scenario_config or {}).get("network", {}) or {}).get("transmission", {}) or {}
    entries = transmission.get("declared_link_outages") or []
    applied = 0
    snapshots = pd.DatetimeIndex(network.snapshots)
    for entry in entries:
        link = str(entry["link"])
        if link not in network.links.index:
            raise ValueError(f"network.transmission.declared_link_outages names {link!r}, which is not in the network")
        start, end = pd.Timestamp(entry["start"]), pd.Timestamp(entry["end"])
        mask = (snapshots >= start) & (snapshots < end)
        if not mask.any():
            continue
        for attr in ("p_max_pu", "p_min_pu"):
            frame = network.links_t[attr]
            series = (frame[link].reindex(network.snapshots) if link in frame.columns
                      else pd.Series(float(network.links.at[link, attr]), index=network.snapshots))
            series = series.copy()
            series.loc[mask] = 0.0
            network.links_t[attr] = pd.concat([frame.drop(columns=[link], errors="ignore"),
                                               series.rename(link).to_frame()], axis=1)
        applied += 1
        logger.info("Declared link outage: %s out from %s to %s (%d snapshots; %s)", link, start, end,
                    int(mask.sum()), entry.get("source", "no source given"))
    return applied
