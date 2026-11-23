"""Declared connection buses for historical builds: the 2030 placement audit applied to past years (19 Nov 2026).

The CP30 2030 study corrected where several stations and offshore farms connect (the gas station schedule's
``BUS_OVERRIDES``, the REPD offshore landing crosswalk). Historical builds placed the same plant by the older
name-prefix heuristic and nearest bus, and in the 2020 validation that put Grain (1,517 MW, Kent) at Fleet in
Hampshire, Keadby behind a 500 MVA busbar link, and Robin Rigg (Cumbria) at Tongland in Scotland: the South East
import boundary bound in 24% of August's hours against 2.7% in NESO's day-ahead flows.

* Stations: ``data/generators/historical_station_buses.csv`` (``generator_pattern`` a regular expression matched
  in full, case-insensitive, to the generator name; ``bus``; ``source``).
* Offshore wind: ``data/network/ETYS/repd_offshore_landing_crosswalk.csv`` by ``repd_ref_id``.

A row whose bus is not in the network is skipped (logged). Returns the moves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

STATION_FILE = "data/generators/historical_station_buses.csv"
LANDING_CROSSWALK = "data/network/ETYS/repd_offshore_landing_crosswalk.csv"


def _ref(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def apply_declared_buses(network, station_file: str | Path | None = STATION_FILE,
                         landing_crosswalk: str | Path | None = LANDING_CROSSWALK, logger=None) -> pd.DataFrame:
    gens = network.generators
    moves = []
    if station_file and Path(station_file).is_file():
        table = pd.read_csv(station_file)
        for row in table.to_dict("records"):
            bus = str(row["bus"])
            pattern = re.compile(str(row["generator_pattern"]), re.IGNORECASE)
            names = [g for g in gens.index if pattern.fullmatch(str(g))]
            if bus not in network.buses.index:
                if logger and names:
                    logger.warning("Declared bus %s for %s is not in the network; left in place", bus, names)
                continue
            for name in names:
                moves.append((name, gens.at[name, "bus"], bus, "station", row.get("source", "")))
    if landing_crosswalk and Path(landing_crosswalk).is_file() and "repd_ref_id" in gens.columns:
        cw = pd.read_csv(landing_crosswalk)
        landing = {_ref(r): b for r, b in zip(cw["repd_ref_id"], cw["landing_bus"])}
        offshore = gens.index[gens["carrier"].astype(str).eq("wind_offshore")]
        for name in offshore:
            bus = landing.get(_ref(gens.at[name, "repd_ref_id"]))
            if bus and bus in network.buses.index:
                moves.append((name, gens.at[name, "bus"], bus, "offshore landing", "repd_offshore_landing_crosswalk"))
    frame = pd.DataFrame(moves, columns=["generator", "from_bus", "to_bus", "rule", "source"])
    frame = frame[frame["from_bus"] != frame["to_bus"]]
    for row in frame.itertuples():
        network.generators.at[row.generator, "bus"] = row.to_bus
    if logger:
        p_nom = network.generators["p_nom"].reindex(frame["generator"]).sum() if len(frame) else 0.0
        logger.info("Declared connection buses: %d generators moved (%.0f MW)", len(frame), p_nom)
        for row in frame.itertuples():
            logger.info("  %s: %s -> %s (%s)", row.generator, row.from_bus, row.to_bus, row.rule)
    return frame
