"""Network delivery cases for the balancing stage (user decision, 8 Oct 2026).

NESO's 2030 constraint-cost forecasts differ by how much of the planned network is
delivered (NAO, *Upgrading the electricity transmission network*, Sep 2026, Figure 4
and para 2.8): all CP30 upgrades (2.84-3.57 bn, our central case), no acceleration of
Norwich-Tilbury and Sea Link (6.58-7.79 bn), the seven most critical projects late (up
to +6.7 bn) and no upgrades at all (10.9-12.7 bn). Stage 1 is a copperplate market, so
the network does not move it: a case changes only the balancing stage and shares its
parent's Stage 1 through ``wholesale_from``.

``market.balancing.network_delivery`` (absent: nothing changes):

* ``remove_components``: ``{component, name | prefix}`` entries. Lines and transformers
  are removed; links are kept with ``p_nom = 0`` (the Stage 1 link schedule names
  them). A line or transformer whose removal would cut a bus carrying generation,
  load, storage or a link off the main grid is kept and reported (a connection asset:
  NESO's figures assume every 2030 generator connects).
* ``revert_appendix_b`` (15 Oct 2026): undo a set of ETYS Appendix B rows the build
  applied, chosen by any of (a row is reverted if any selector picks it):

  - ``after_year: Y``: every row dated after ``Y`` (the CP30-dated works and the HVDC
    enabling works pulled in from the 2032-2035 snapshots included);
  - ``projects: [name, ...]``: the rows ``data/network/ETYS/appendix_b_projects.csv``
    lists for each project (``projects_file`` overrides the path);
  - ``cp30_dated_works: true``: the rows the build pulled in as CP30-dated works;
  - ``rows: [id, ...]``: row ids as the upgrade audit's ``appendix_b_row_id`` gives them.

  The build records what every row did (``scripts/network_build/appendix_b_record.py``).
  Each component a chosen row touched is replayed through its recorded operations
  without the chosen rows, so the result is the state the other rows alone give: two
  changes on one circuit, a removal of a circuit a later row added and chained
  re-terminations included. In order: branches the chosen rows removed are put back on
  their original buses (with any bus, or dead island the build pruned because of the
  removal); re-rated branches get their earlier rating, impedance and seasonal ratings;
  the branches the chosen rows added are removed under the connection rule (a generator,
  load, storage unit or live link the removals would cut off keeps the fewest added
  branches that reach the main grid, the lowest-rated among equals, reported under
  ``kept_as_connection`` with their row ids, rating and what they connect); added links
  are kept with ``p_nom = 0``. Before the connection rule, small embedded plant on new buses
  the revert strands is re-homed (F25.5, 27 Oct 2026): a pocket of buses the Appendix B rows
  created, with no load, store or live link and at most ``rehome_stranded_max_mw`` (default
  50 MW) of generation and storage, moves to the nearest main-grid bus of the same voltage
  (else the nearest voltage), listed under ``rehomed`` (pockets left alone under
  ``stranded_pockets_kept``), and its circuit goes; new buses left without a branch, or in an island with
  nothing on it, are dropped with their load-shedding units; and the class-median
  seasonal ratings are recomputed on the reverted network. Impedances are recorded in pu
  on 100 MVA and written back in the network's own units (lines in ohm at bus0
  ``v_nom``; transformers pu on their own ``s_nom``).
* ``secured_capability``: keys that replace those of
  ``network.transmission.outage_schedule.secured_capability`` (``capability_year``,
  ``capability_year_overrides``, and since 9 Oct 2026 ``historic_derate`` and ``phase_in``: the
  NESO-style outage derate and the capability phase-in of scripts/solve/secured_boundaries.py, which
  act on the balancing stage only), and since 16 Oct 2026 ``boundaries`` (the list of
  secured boundaries; absent = every boundary with a capability row). A delayed
  project's boundary capability is held at the year before it, as NESO's own
  boundary-based studies do.
* ``transmission`` (16 Oct 2026): keys that replace those of ``network.transmission`` for the
  balancing stage only: ``min_bm_constraint_voltage_kv`` and ``bm_voltage_floor``
  (merged key by key). The boundaries-only sensitivity (NESO review S2-15) sets the
  floor to 1000 kV and ``bm_voltage_floor.scottish_transmission_constrained: false``,
  so no branch limit binds, and keeps NESO's 11 zone boundaries (B1a, B2, B4, B6, B7a,
  B8, B9, EC5, LE1, SW1, B13) in ``secured_capability.boundaries``: transfer is
  limited by boundary capability only, as in NESO's 17-zone model. Relaxed branches
  keep their place in the network, so their flows still count in the boundary sums.

``revert_cp30_dated_works`` (the first version of the revert, 15 Oct 2026) is retired:
it is ``revert_appendix_b: {cp30_dated_works: true}``.

The applied case is logged and stored in ``network.meta['network_delivery']``. Run it
before the line rating overrides and the outage schedule, so restored branches get the
BM voltage floor and the secured boundary specs see the case's branches.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

COMPONENT_TABLES = {"line": "lines", "transformer": "transformers", "link": "links"}
REVERT_KEY = "revert_appendix_b"
REVERT_SELECTORS = ("after_year", "projects", "cp30_dated_works", "rows", "projects_file")
# capability_file (19 Oct 2026): a case may read its boundary capabilities from another table (e.g. ETYS 2024's
# CP30-era values), resolved by secured_boundaries.secured_boundary_config like the base outage config's own key.
SECURED_CAPABILITY_KEYS = ("capability_year", "capability_year_overrides", "historic_derate", "phase_in",
                           "boundaries", "capability_file")
TRANSMISSION_KEYS = ("min_bm_constraint_voltage_kv", "bm_voltage_floor")
#: ``network_delivery.rehome_stranded_max_mw`` (F25.5, 27 Oct 2026): a pocket of new buses the revert
#: strands, holding at most this much generation plus storage and no load, store or live link, is
#: re-homed to the nearest surviving bus instead of keeping its circuit. 0 turns it off.
REHOME_KEY = "rehome_stranded_max_mw"
DEFAULT_REHOME_STRANDED_MAX_MW = 50.0
#: A re-homed component keeps its bus's voltage when a main-grid bus of that voltage is this close.
REHOME_SAME_VOLTAGE_KM = 25.0
APPENDIX_B_PROJECTS_FILE = Path(__file__).resolve().parents[2] / "data/network/ETYS/appendix_b_projects.csv"
PROJECT_COLUMNS = ("project", "project_name", "kind", "operation", "operator", "node1", "node2",
                   "appendix_b_year", "occurrence", "winter_rating_mva", "source")
RETIRED_KEYS = {
    "revert_cp30_dated_works": "use revert_appendix_b: {cp30_dated_works: true} (15 Oct 2026); the "
                               "no_upgrades case is revert_appendix_b: {after_year: 2025}",
}


def network_delivery_config(balancing_config: dict | None) -> dict | None:
    """The case block, validated, or ``None`` when no case is set."""
    block = (balancing_config or {}).get("network_delivery")
    if not block:
        return None
    if not block.get("case"):
        raise ValueError("market.balancing.network_delivery needs a case name")
    for key, pointer in RETIRED_KEYS.items():
        if key in block:
            raise ValueError(f"network_delivery.{key} is retired: {pointer}")
    for entry in block.get("remove_components", []) or []:
        if entry.get("component") not in COMPONENT_TABLES:
            raise ValueError(f"network_delivery: unknown component {entry.get('component')!r}")
        if bool(entry.get("name")) == bool(entry.get("prefix")):
            raise ValueError(f"network_delivery: give exactly one of name or prefix in {entry}")
    if block.get(REVERT_KEY) is not None:
        validate_revert_selector(block[REVERT_KEY])
    rehome_stranded_max_mw(block)
    unknown = set(block.get("secured_capability", {}) or {}) - set(SECURED_CAPABILITY_KEYS)
    if unknown:
        raise ValueError(f"network_delivery.secured_capability: unknown keys {sorted(unknown)}")
    boundaries = (block.get("secured_capability", {}) or {}).get("boundaries")
    if boundaries is not None and (not isinstance(boundaries, list) or not boundaries
                                   or not all(isinstance(b, str) and b for b in boundaries)):
        raise ValueError("network_delivery.secured_capability.boundaries must be a non-empty list of boundary names")
    unknown = set(block.get("transmission", {}) or {}) - set(TRANSMISSION_KEYS)
    if unknown:
        raise ValueError(f"network_delivery.transmission: unknown keys {sorted(unknown)}; "
                         f"supported: {list(TRANSMISSION_KEYS)}")
    return block


def rehome_stranded_max_mw(block: dict | None) -> float:
    """``network_delivery.rehome_stranded_max_mw``, validated (default ``DEFAULT_REHOME_STRANDED_MAX_MW``)."""
    value = (block or {}).get(REHOME_KEY, DEFAULT_REHOME_STRANDED_MAX_MW)
    if value is None:
        return DEFAULT_REHOME_STRANDED_MAX_MW
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"network_delivery.{REHOME_KEY} must be a number of MW >= 0 (0 turns it off), "
                         f"got {value!r}")
    return float(value)


def validate_revert_selector(selector) -> dict:
    """``revert_appendix_b``: a mapping of at least one selector (see the module docstring)."""
    if not isinstance(selector, dict) or not (set(selector) - {"projects_file"}):
        raise ValueError(f"network_delivery.{REVERT_KEY} must be a mapping with at least one of "
                         "after_year, projects, cp30_dated_works or rows")
    unknown = set(selector) - set(REVERT_SELECTORS)
    if unknown:
        raise ValueError(f"network_delivery.{REVERT_KEY}: unknown keys {sorted(unknown)}")
    year = selector.get("after_year")
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        raise ValueError(f"network_delivery.{REVERT_KEY}.after_year must be a year, got {year!r}")
    for key in ("projects", "rows"):
        value = selector.get(key)
        if value is not None and (not isinstance(value, list) or not value
                                  or not all(isinstance(v, str) and v for v in value)):
            raise ValueError(f"network_delivery.{REVERT_KEY}.{key} must be a non-empty list of names")
    if not isinstance(selector.get("cp30_dated_works", False), bool):
        raise ValueError(f"network_delivery.{REVERT_KEY}.cp30_dated_works must be true or false")
    return selector


def _matches(network, entry: dict) -> list[str]:
    table = getattr(network, COMPONENT_TABLES[entry["component"]])
    if entry.get("name"):
        if entry["name"] not in table.index:
            raise ValueError(f"network_delivery: {entry['component']} {entry['name']!r} is not in the network")
        return [entry["name"]]
    names = [n for n in table.index if str(n).startswith(entry["prefix"])]
    if not names:
        raise ValueError(f"network_delivery: no {entry['component']} starts with {entry['prefix']!r}")
    return names


#: Bus carriers a link joins as grid: HVDC between electrical buses (23 Oct 2026).
ELECTRICAL_BUS_CARRIERS = frozenset({"AC", "DC"})


def _graph(network) -> nx.MultiGraph:
    """Buses joined by lines, transformers and links that can carry power between electrical buses.

    A link into a non-electrical bus is not a route (23 Oct 2026): every electrolyser and H2
    turbine shares the GB_H2 buses, so counting them made a pocket with one look connected
    through the others, and the revert kept no circuit for it. Under no_upgrades that left
    Emmock (the 1.16 GW electrolyser) and An Suidhe (259 MW) as islands.
    """
    graph = nx.MultiGraph()
    graph.add_nodes_from(network.buses.index)
    for table in ("lines", "transformers"):
        frame = getattr(network, table)
        graph.add_edges_from((r.bus0, r.bus1, f"{table}:{name}") for name, r in frame[["bus0", "bus1"]].iterrows())
    links = network.links
    carrier = network.buses["carrier"]
    electrical = (links["bus0"].map(carrier).isin(ELECTRICAL_BUS_CARRIERS)
                  & links["bus1"].map(carrier).isin(ELECTRICAL_BUS_CARRIERS))
    live = links[(links["p_nom"] > 0) & electrical]
    graph.add_edges_from((r.bus0, r.bus1, f"links:{name}") for name, r in live[["bus0", "bus1"]].iterrows())
    return graph


#: Slack carriers every bus carries; they never make a bus a connection point (8 Oct 2026: the
#: load-shedding units kept two empty East Anglia stubs as "connection assets").
SLACK_CARRIERS = frozenset({"load_shedding"})


def _occupied_buses(network, ignore_links=()) -> set:
    generators = network.generators[~network.generators["carrier"].isin(SLACK_CARRIERS)]
    buses = set(generators["bus"]) | set(network.loads["bus"]) | set(network.storage_units["bus"])
    live = network.links[(network.links["p_nom"] > 0) & ~network.links.index.isin(list(ignore_links))]
    return buses | set(live["bus0"]) | set(live["bus1"])


def _stranded(graph: nx.MultiGraph, bus0, bus1, occupied: set) -> set:
    """Occupied buses the edge just removed cut off from the component holding the other end."""
    if nx.has_path(graph, bus0, bus1):
        return set()
    side0, side1 = nx.node_connected_component(graph, bus0), nx.node_connected_component(graph, bus1)
    smaller = side0 if len(side0) <= len(side1) else side1
    return smaller & occupied


def _strands_occupied_bus(graph: nx.MultiGraph, bus0, bus1, occupied: set) -> bool:
    """True when the edge just removed left an occupied bus off the component holding the other end."""
    return bool(_stranded(graph, bus0, bus1, occupied))


def _occupants(network, buses) -> list[str]:
    """What sits on ``buses`` (for the report): non-slack generators, loads, storage, stores, live links."""
    buses = set(buses)
    out = []
    gens = network.generators[network.generators["bus"].isin(buses)
                              & ~network.generators["carrier"].isin(SLACK_CARRIERS)]
    out += [f"generator:{n} ({c}, {p:.0f} MW)" for n, c, p in zip(gens.index, gens["carrier"], gens["p_nom"])]
    out += [f"load:{n}" for n in network.loads.index[network.loads["bus"].isin(buses)]]
    su = network.storage_units[network.storage_units["bus"].isin(buses)]
    out += [f"storage_unit:{n} ({c}, {p:.0f} MW)" for n, c, p in zip(su.index, su["carrier"], su["p_nom"])]
    if len(network.stores):
        out += [f"store:{n}" for n in network.stores.index[network.stores["bus"].isin(buses)]]
    live = network.links[(network.links["p_nom"] > 0)
                         & (network.links["bus0"].isin(buses) | network.links["bus1"].isin(buses))]
    out += [f"link:{n}" for n in live.index]
    return out


def _occupant_mw(network, buses) -> float:
    """Non-slack generation plus storage power on ``buses`` (MW), for judging a kept connection."""
    buses = set(buses)
    gens = network.generators[network.generators["bus"].isin(buses)
                              & ~network.generators["carrier"].isin(SLACK_CARRIERS)]
    su = network.storage_units[network.storage_units["bus"].isin(buses)]
    return float(gens["p_nom"].sum() + su["p_nom"].sum())


# ── Appendix B rows: selection ───────────────────────────────────────────────


def load_appendix_b_projects(path=None) -> pd.DataFrame:
    """The project-to-rows table (``data/network/ETYS/appendix_b_projects.csv``), validated."""
    path = Path(path) if path else APPENDIX_B_PROJECTS_FILE
    table = pd.read_csv(path)
    missing = set(PROJECT_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Appendix B projects file {path} is missing columns {sorted(missing)}")
    if table["source"].isna().any() or table["source"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Every row of {path} needs a source")
    bad = set(table["kind"]) - {"circuit", "transformer", "hvdc"}
    bad |= set(table["operation"]) - {"addition", "removal", "change"}
    if bad:
        raise ValueError(f"Appendix B projects file {path}: unknown kind/operation {sorted(bad)}")
    key = ["kind", "operator", "appendix_b_year", "operation", "node1", "node2", "occurrence"]
    if table.duplicated(key).any():
        raise ValueError(f"Appendix B projects file {path}: rows must be unique by {', '.join(key)}")
    return table


def project_row_id(entry) -> str:
    from scripts.network_build.appendix_b_record import row_id
    return row_id(entry["kind"], entry["operator"], entry["appendix_b_year"], entry["operation"],
                  entry["node1"], entry["node2"], int(entry["occurrence"]))


def select_appendix_b_rows(record: dict, selector: dict, projects_file=None) -> tuple[dict, dict]:
    """``({row id: reason}, summary)`` for the rows ``selector`` picks from the build's record."""
    rows = {r["id"]: r for r in record["rows"]}
    chosen: dict = {}
    after = selector.get("after_year")
    if after is not None:
        for rid, r in rows.items():
            if r.get("source_year") is not None and float(r["source_year"]) > after:
                chosen.setdefault(rid, f"after_{after}")
    if selector.get("cp30_dated_works"):
        for rid, r in rows.items():
            if r.get("cp30_dated_work"):
                chosen.setdefault(rid, f"cp30_dated_work:{r['cp30_dated_work']}")
    if selector.get("projects"):
        table = load_appendix_b_projects(selector.get("projects_file") or projects_file)
        for name in selector["projects"]:
            entries = table[table["project"] == name]
            if entries.empty:
                raise ValueError(f"revert_appendix_b: project {name!r} is not in the Appendix B projects file")
            for _, entry in entries.iterrows():
                rid = project_row_id(entry)
                if rid not in rows:
                    raise ValueError(f"revert_appendix_b: project {name} row {rid} is not in this build's Appendix "
                                     "B record (out of scope for the modelled year, or not in the workbook)")
                rating = rows[rid].get("winter_rating_mva")
                if pd.notna(entry["winter_rating_mva"]) and (
                        rating is None or abs(float(rating) - float(entry["winter_rating_mva"])) > 1e-6):
                    raise ValueError(f"revert_appendix_b: project {name} row {rid} is rated "
                                     f"{entry['winter_rating_mva']} MVA in the projects file but {rating} in the build")
                chosen.setdefault(rid, f"project:{name}")
    for rid in selector.get("rows") or []:
        if rid not in rows:
            raise ValueError(f"revert_appendix_b: row {rid} is not in this build's Appendix B record")
        chosen.setdefault(rid, "row")
    years = pd.Series([rows[r].get("source_year") for r in chosen], dtype=float)
    summary = {
        "selector": {k: v for k, v in selector.items() if k != "projects_file"},
        "rows_selected": len(chosen),
        "by_reason": pd.Series(list(chosen.values()), dtype=object).str.split(":").str[0].value_counts().to_dict()
        if chosen else {},
        "by_year": {str(int(k)): int(v) for k, v in years.value_counts().sort_index().items()} if len(years) else {},
        "not_applied": sorted(rid for rid in chosen
                              if not str(rows[rid].get("audit_status") or "").startswith("applied")),
    }
    return chosen, summary


# ── Appendix B rows: replay ──────────────────────────────────────────────────


def _fields(component: str) -> tuple:
    from scripts.network_build.appendix_b_record import CHANGE_FIELDS, LINK_FIELDS
    return LINK_FIELDS if component == "link" else CHANGE_FIELDS


def _replay(seq: list, skip, upto=None) -> tuple[bool, dict, list]:
    """``(exists, fields, conflicts)`` of one component after its ops, leaving out ``skip(op)``.

    ``upto``: stop before the op with this sequence number. A change on an absent component
    does nothing (the change row finds no branch); so does a prune of one, or a replayed
    missed removal. A recorded removal of an absent component is a conflict: without the
    left-out rows the build would have removed another branch.
    """
    component = seq[0]["component"]
    fields = _fields(component)
    first = seq[0]
    exists = first["op"] != "add"
    values = {f: (first.get("before") or {}).get(f) for f in fields} if exists else {}
    conflicts = []
    for op in seq:
        if upto is not None and op["seq"] >= upto:
            break
        if skip(op):
            continue
        if op["op"] == "add":
            exists, values = True, dict(op["after"])
        elif op["op"] == "change":
            if exists:
                values.update(op["after"])
        elif exists:                            # remove, prune
            exists = False
        elif op["op"] == "remove" and not op.get("missed"):
            conflicts.append({"op": op["op"], "row": op.get("row"), "reason": "component already absent"})
    return exists, values, conflicts


def _op_buses(op) -> frozenset:
    buses = op.get("buses") or [(op.get("before") or {}).get("bus0"), (op.get("before") or {}).get("bus1")]
    return frozenset(b for b in buses if b)


def plan_appendix_b_revert(record: dict, selected) -> tuple[list, set]:
    """One plan per component a selected row touched: its state without the selected rows.

    Returns ``(plans, islands_restored)``. A plan has ``component``, ``name``, ``rows``
    (the selected rows that touched it), ``target_exists`` and ``target`` (the recorded
    fields without them), ``built`` (with every row, for spotting later edits), ``full``
    (the last full row recorded, for re-adding it), ``replayed`` (rows that missed the
    branch in the build and apply to it without the selected rows) and ``conflicts``.

    The build applies every removal, of any year, before any change, so a kept change or
    removal row can miss its branch because a selected row removed it first (in the CP30
    2030 build: the 2025 changes on LAMB2T--WIYH2- and HAKB4---HARK41, whose circuits 2029
    and 2027 rows remove). The build records those misses (``change_missed``,
    ``remove_missed``) in sequence; here they are replayed on the branch the revert puts back.
    """
    selected = set(selected)
    restore = {i["island"] for i in record.get("islands", []) if set(i.get("caused_by", [])) & selected}

    def chosen(op) -> bool:
        if op["op"] == "prune":
            return op["island"] in restore
        return op.get("row") in selected

    by_component = defaultdict(list)
    missed = []
    for op in record["ops"]:
        if op["op"] in ("change_missed", "remove_missed"):
            if op.get("row") not in selected:
                missed.append(op)
            continue
        by_component[(op["component"], op["name"])].append(op)
    touched = {}
    for key, seq in by_component.items():
        seq.sort(key=lambda o: o["seq"])
        if any(chosen(o) for o in seq):
            touched[key] = seq
    by_pair = defaultdict(list)
    for key, seq in touched.items():
        pairs = {_op_buses(o) for o in seq} - {frozenset()}
        for pair in pairs:
            by_pair[(key[0], pair)].append(key)

    def target_skip(op) -> bool:
        return chosen(op)

    replayed = defaultdict(list)
    for op in sorted(missed, key=lambda o: o["seq"]):
        keys = by_pair.get((op["component"], _op_buses(op)), [])
        synthetic = dict(op, op=op["op"].replace("_missed", ""), missed=True)
        if op["op"] == "change_missed":
            targets = keys                       # a change row sets every parallel circuit
        else:
            alive = []
            for key in keys:
                exists, values, _ = _replay(touched[key], target_skip, upto=op["seq"])
                if exists:
                    rating = values.get("s_nom")
                    gap = abs(float(rating) - float(op["rating"])) if rating is not None and op.get("rating") \
                        is not None else 0.0
                    alive.append((gap, key))
            targets = [min(alive)[1]] if alive else []       # one branch, the closest rating
        for key in targets:
            touched[key] = sorted(touched[key] + [synthetic], key=lambda o: o["seq"])
            replayed[key].append(op.get("row"))

    plans = []
    for (component, name), seq in touched.items():
        target_exists, target, conflicts = _replay(seq, target_skip)
        built_exists, built, _ = _replay(seq, lambda o: bool(o.get("missed")))
        full = None
        for op in seq:
            if op["op"] in ("remove", "prune") and not op.get("missed"):
                full = op["before"]
        plans.append({"component": component, "name": name,
                      "rows": sorted({o["row"] for o in seq if chosen(o) and o.get("row") and not o.get("missed")}),
                      "islands": sorted({o["island"] for o in seq if chosen(o) and o["op"] == "prune"}),
                      "replayed": replayed.get((component, name), []),
                      "target_exists": target_exists, "target": target, "built_exists": built_exists,
                      "built": built, "full": full, "conflicts": conflicts})
    return plans, restore


def _restorable(network, component: str, row: dict) -> dict:
    """The recorded row's values the target table can take (no output attributes, no new columns)."""
    from scripts.network_build.appendix_b_record import (
        COMPONENT_CLASS, COMPONENT_TABLE, CP30_DATED_WORK_COLUMN, output_attributes,
    )
    table = getattr(network, COMPONENT_TABLE[component])
    row = dict(row)
    if component == "line" and "v_nom" in row and "v_nom" not in table.columns:
        # network_io moves the legacy lines.v_nom to voltage_kv on save
        row.setdefault("voltage_kv", row["v_nom"])
        row.pop("v_nom")
    skip = output_attributes(network, component) | {CP30_DATED_WORK_COLUMN}
    attrs = network.components[COMPONENT_CLASS[component]].defaults
    keep = set(table.columns) | set(attrs.index[attrs["static"].astype(bool)])
    return {k: v for k, v in row.items() if k in keep and k not in skip}


def _in_network_units(network, component: str, values: dict, units: str) -> dict:
    """Recorded pu-on-100-MVA impedances in the network's own units."""
    from scripts.network_build.branch_units import UNITS_PU100, branch_values_for_network
    if units != UNITS_PU100:
        raise ValueError(f"revert_appendix_b: recorded impedances are in {units!r}, expected {UNITS_PU100!r}")
    if component == "link" or not any(values.get(k) is not None for k in ("r", "x", "b", "g")):
        return dict(values)
    cls = "Line" if component == "line" else "Transformer"
    present = {k: v for k, v in values.items() if v is not None}
    return dict(values, **branch_values_for_network(network, cls, present, values.get("bus0"),
                                                    s_nom=values.get("s_nom")))


def _check_tags(network, record: dict) -> None:
    """Every component tagged ``cp30_dated_work`` must be one a recorded dated row added or changed."""
    from scripts.network_build.appendix_b_record import COMPONENT_TABLE, CP30_DATED_WORK_COLUMN, is_tagged
    dated = {r["id"] for r in record["rows"] if r.get("cp30_dated_work")}
    recorded = {(o["component"], o["name"]) for o in record["ops"]
                if o.get("row") in dated and o["op"] in ("add", "change")}
    tagged = set()
    for component in ("line", "transformer", "link"):
        table = getattr(network, COMPONENT_TABLE[component])
        if CP30_DATED_WORK_COLUMN in table.columns:
            tagged |= {(component, n) for n, v in table[CP30_DATED_WORK_COLUMN].items() if is_tagged(v)}
    extra = tagged - recorded
    if extra:
        raise ValueError("revert_appendix_b: components tagged cp30_dated_work but not in the recorded "
                         f"Appendix B works (network changed after the build?): {sorted(extra)[:10]}")


def _set_branch_fields(network, component: str, name: str, target: dict, units: str) -> None:
    """Write the recorded rating, impedance and seasonal ratings onto a present branch."""
    from scripts.network_build.branch_units import set_transformer_s_nom_preserving_impedance
    table = network.lines if component == "line" else network.transformers
    s_nom = target.get("s_nom")
    if s_nom is not None:
        if component == "transformer":
            set_transformer_s_nom_preserving_impedance(network, [name], float(s_nom))
        else:
            table.at[name, "s_nom"] = float(s_nom)
    impedance = {k: target[k] for k in ("r", "x", "b", "g") if target.get(k) is not None}
    if impedance:
        converted = _in_network_units(network, component, dict(
            impedance, bus0=table.at[name, "bus0"], s_nom=float(table.at[name, "s_nom"])), units)
        for k in impedance:
            table.at[name, k] = converted[k]
    if component == "line":
        for col in ("s_nom_spring", "s_nom_summer", "s_nom_autumn"):
            if col in table.columns:
                table.at[name, col] = np.nan if target.get(col) is None else float(target[col])


def _filled_season_cells(network) -> dict:
    """Line seasonal cells the build filled from the class-median ratio (refilled after a revert)."""
    from scripts.network_build.branch_attributes import RELAXED_RATING_MVA, SEASON_COLUMNS
    lines = network.lines
    if "seasonal_rating_source" not in lines.columns or not len(lines):
        return {}
    source = lines["seasonal_rating_source"].astype(object)
    v0 = lines["bus0"].map(network.buses["v_nom"]).astype(float)
    s_nom = lines["s_nom"].astype(float)
    ref = source.eq("etys") & (s_nom > 0) & (s_nom < RELAXED_RATING_MVA)
    candidates = source.eq("class_median_ratio") & (s_nom < RELAXED_RATING_MVA)
    out = {}
    for col in SEASON_COLUMNS.values():
        if col not in lines.columns:
            continue
        ratio = (lines.loc[ref, col].astype(float) / s_nom[ref]).clip(upper=1.5)
        by_class = ratio.groupby(v0[ref]).median()
        overall = float(ratio.median()) if len(ratio) else 1.0
        fill = s_nom * v0.map(by_class).fillna(overall)
        value = lines[col].astype(float)
        out[col] = list(lines.index[candidates & np.isclose(value, fill, rtol=1e-9, atol=1e-6)])
    return out


def _refill_seasonal(network, filled: dict, targets: list, log) -> None:
    """Recompute the class-median seasonal ratings on the reverted network, as the build does.

    ``filled`` are the cells the build filled (:func:`_filled_season_cells`, taken before the
    revert); ``targets`` the lines the revert restored or re-rated, whose recorded seasons
    (NaN where Appendix B published none) are completed here.
    """
    from scripts.network_build.branch_attributes import RELAXED_RATING_MVA, SEASON_COLUMNS
    lines = network.lines
    targets = [n for n in targets if n in lines.index]
    if not len(lines) or (not targets and not any(filled.values())):
        return
    for col in list(SEASON_COLUMNS.values()) + ["s_nom_winter"]:
        if col not in lines.columns:
            lines[col] = np.nan
    if "seasonal_rating_source" not in lines.columns:
        lines["seasonal_rating_source"] = pd.Series(np.nan, index=lines.index, dtype=object)
    s_nom = lines["s_nom"].astype(float)
    v0 = lines["bus0"].map(network.buses["v_nom"]).astype(float)
    is_target = lines.index.isin(targets)
    target_idx = pd.Index(targets)
    had_all = lines.loc[target_idx, list(SEASON_COLUMNS.values())].notna().all(axis=1)
    relaxed = s_nom >= RELAXED_RATING_MVA
    published = lines["seasonal_rating_source"].astype(object).eq("etys") & ~is_target
    published[target_idx] = had_all.to_numpy()
    ref = published & ~relaxed & (s_nom > 0)
    for col in SEASON_COLUMNS.values():
        ratio = (lines.loc[ref, col].astype(float) / s_nom[ref]).clip(upper=1.5)
        by_class = ratio.groupby(v0[ref]).median()
        overall = float(ratio.median()) if len(ratio) else 1.0
        fill = s_nom * v0.map(by_class).fillna(overall)
        cells = [n for n in filled.get(col, []) if n in lines.index and n not in set(targets)]
        lines.loc[cells, col] = fill[cells]
        missing = target_idx[(lines.loc[target_idx, col].isna() & ~relaxed[target_idx]).to_numpy()]
        lines.loc[missing, col] = fill[missing]
        lines.loc[target_idx[relaxed[target_idx].to_numpy()], col] = s_nom[target_idx][relaxed[target_idx]]
    lines.loc[target_idx, "s_nom_winter"] = s_nom[target_idx]
    lines.loc[target_idx, "seasonal_rating_source"] = np.where(
        relaxed[target_idx], "relaxed", np.where(had_all, "etys", "class_median_ratio"))
    log.info(f"  Seasonal ratings recomputed: {len(targets)} restored or re-rated lines, "
             f"{sum(len(v) for v in filled.values())} class-median cells refilled")


def _complete_transformers_and_licensee(network, lines: list, transformers: list) -> None:
    """Restored transformers take their one rating in every season; missing licensees by site majority."""
    from scripts.network_build.branch_attributes import SEASON_COLUMNS
    trafos = [n for n in transformers if n in network.transformers.index]
    table = network.transformers
    for col in list(SEASON_COLUMNS.values()) + ["s_nom_winter"]:
        if trafos and col in table.columns:
            table.loc[trafos, col] = table.loc[trafos, "s_nom"].astype(float)
    if trafos and "seasonal_rating_source" in table.columns:
        table.loc[trafos, "seasonal_rating_source"] = "winter_only"
    for names, table in ((lines, network.lines), (trafos, network.transformers)):
        names = [n for n in names if n in table.index]
        if not names or "etys_licensee" not in table.columns:
            continue
        lic = table["etys_licensee"].astype(object)
        known = lic.notna() & ~lic.astype(str).str.strip().isin(["", "nan"])
        site_lic = {}
        for t in (network.lines, network.transformers):
            if "etys_licensee" not in t.columns:
                continue
            k = t[t["etys_licensee"].notna() & ~t["etys_licensee"].astype(str).str.strip().isin(["", "nan", "OFTO"])]
            for end in ("bus0", "bus1"):
                for site, value in zip(k[end].astype(str).str[:4], k["etys_licensee"]):
                    site_lic.setdefault(site, []).append(value)
        site_lic = {s: pd.Series(v).value_counts().index[0] for s, v in site_lic.items()}
        has_source = "etys_licensee_source" in table.columns
        for name in names:
            if known[name]:
                if has_source and not isinstance(table.at[name, "etys_licensee_source"], str):
                    table.at[name, "etys_licensee_source"] = "etys_sheet"
                continue
            ends = [str(table.at[name, "bus0"])[:4], str(table.at[name, "bus1"])[:4]]
            found = [site_lic[e] for e in ends if e in site_lic]
            if found:
                table.at[name, "etys_licensee"] = found[0]
                if has_source:
                    table.at[name, "etys_licensee_source"] = "site_majority"


def _connection_keeps(graph: nx.MultiGraph, added: list, occupied: set, weight_of: dict) -> dict:
    """The fewest added branches that keep every occupied bus on the main grid.

    ``added`` is ``(bus0, bus1, key)`` per live added branch, ``weight_of[key]`` its rating.
    Without them the grid splits; each occupied bus left off the largest part is joined
    back along the path that needs the fewest added branches (existing branches cost
    nothing; among paths with as few, the lowest total rating), so a generator at a new
    substation keeps one route, not whichever branches a removal order happened to
    leave. Returns ``{key: occupied buses it serves}``.
    """
    current = graph.copy()
    current.remove_edges_from(added)
    main = max(nx.connected_components(current), key=len) if current.number_of_nodes() else set()
    by_pair = defaultdict(list)
    for u, v, key in added:
        by_pair[frozenset((u, v))].append(key)
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    for u, v in current.edges():
        weighted.add_edge(u, v, weight=0.0, key=None)
    for pair, keys in by_pair.items():
        u, v = tuple(pair) if len(pair) == 2 else (next(iter(pair)),) * 2
        if weighted.has_edge(u, v):
            continue
        key = min(keys, key=lambda k: (weight_of.get(k, 0.0), k))      # the lowest-rated parallel
        weighted.add_edge(u, v, weight=1.0 + 1e-7 * weight_of.get(key, 0.0), key=key)
    hub = ("__main_grid__",)
    weighted.add_edges_from(((hub, b) for b in main), weight=0.0, key=None)
    keeps: dict = {}
    anchor = next(iter(main)) if main else None
    for bus in sorted(occupied & set(graph.nodes), key=str):
        if anchor is None or nx.has_path(current, bus, anchor):
            continue
        try:
            path = nx.dijkstra_path(weighted, bus, hub)
        except nx.NetworkXNoPath:
            continue                    # not reachable even with the additions: not ours to fix
        served = nx.node_connected_component(current, bus) & occupied
        for u, v in zip(path[:-2], path[1:-1]):
            key = weighted[u][v]["key"]
            if key is not None:
                keeps.setdefault(key, set()).update(served)
                current.add_edge(u, v, key=key)
                weighted[u][v]["weight"] = 0.0
                weighted[u][v]["key"] = None
    return keeps


def _live_additions(network, graph: nx.MultiGraph, todo: dict) -> list:
    """``(bus0, bus1, key)`` of each selected addition still in the graph."""
    from scripts.network_build.appendix_b_record import COMPONENT_TABLE
    live = []
    for p in todo["additions"]:
        table = getattr(network, COMPONENT_TABLE[p["component"]])
        key = f"{COMPONENT_TABLE[p['component']]}:{p['name']}"
        b0, b1 = table.at[p["name"], "bus0"], table.at[p["name"], "bus1"]
        if graph.has_edge(b0, b1, key=key):
            live.append((b0, b1, key))
    return live


def _rehome_target_buses(network, main: set, pocket: set, bus) -> tuple[pd.Index, float]:
    """Main-grid AC buses a component on ``bus`` may move to, and the voltage chosen.

    The voltage is the bus's own where a main-grid bus of that voltage lies within
    ``REHOME_SAME_VOLTAGE_KM``, else the nearest voltage that has one, else any voltage.
    """
    from scripts.utilities.spatial_utils import _bus_osgb_xy
    buses = network.buses
    carrier = buses["carrier"] if "carrier" in buses.columns else pd.Series("AC", index=buses.index)
    names = buses.index.astype(str)
    candidates = buses.index[buses.index.isin(list(main)) & ~buses.index.isin(list(pocket))
                             & carrier.fillna("AC").replace("", "AC").eq("AC").to_numpy()
                             & ~names.str.contains("External", case=False)]
    xy = _bus_osgb_xy(buses)
    candidates = candidates[xy.loc[candidates].notna().all(axis=1).to_numpy()]
    if not len(candidates) or xy.loc[bus].isna().any():
        return candidates, float("nan")
    v_nom = pd.to_numeric(buses["v_nom"], errors="coerce")
    distance = np.hypot(xy.loc[candidates, "x"] - xy.at[bus, "x"], xy.loc[candidates, "y"] - xy.at[bus, "y"]) / 1e3
    v = float(v_nom.at[bus])
    for level in sorted(v_nom.loc[candidates].dropna().unique(), key=lambda lv: (abs(lv - v), -lv)):
        at_level = candidates[v_nom.loc[candidates].eq(level).to_numpy()]
        if (distance.loc[at_level] <= REHOME_SAME_VOLTAGE_KM).any():
            return at_level, float(level)
    return candidates, float("nan")


def _rehome_stranded_embedded(network, graph: nx.MultiGraph, occupied: set, todo: dict, record: dict,
                              out: dict, max_mw: float, log) -> None:
    """Move small embedded generation and storage off new buses the revert strands (F25.5, 27 Oct 2026).

    Without the selected additions, a pocket of buses the Appendix B rows created (``buses_added``)
    may hold plant the spatial mapping put on a new 2027/2029 substation. Keeping a 400 kV circuit for
    a few MW of embedded solar is not what a no-upgrades network has; such plant connects at the
    nearest existing substation. A pocket is re-homed when every occupied bus in it is new, it holds no
    load, no store and no live link, nothing on it is an evidenced placement (``_PROTECTED_PLACEMENTS``
    or a documented onshore connection), and its generation plus storage is at most ``max_mw``. Each
    generator and storage unit goes to the nearest main-grid bus of the same voltage (else the nearest
    voltage; ``_rehome_target_buses``), with ``spatial_utils.reassign_excluded_placements``. Its
    capacity, profile and labels are unchanged; only ``bus`` moves, and the pocket's added branches are
    then removed like any other. Every move is listed under ``rehomed``; pockets left alone under
    ``stranded_pockets_kept`` with the reason.
    """
    from scripts.utilities.spatial_utils import (_PROTECTED_PLACEMENTS, _explicit_onshore_connection,
                                                 reassign_excluded_placements)
    out["rehome_max_mw"] = float(max_mw)
    if not max_mw or max_mw <= 0:
        return
    current = graph.copy()
    current.remove_edges_from(_live_additions(network, graph, todo))
    if not current.number_of_nodes():
        return
    main = max(nx.connected_components(current), key=len)
    anchor = next(iter(main))
    new_buses = set(record.get("buses_added", []))
    gens = network.generators[~network.generators["carrier"].isin(SLACK_CARRIERS)]
    su = network.storage_units
    stores = network.stores if len(network.stores) else pd.DataFrame(columns=["bus"])
    loads = set(network.loads["bus"])
    live_links = network.links[network.links["p_nom"] > 0]
    link_buses = set(live_links["bus0"]) | set(live_links["bus1"])
    link_buses &= occupied          # added links (disabled by the revert) do not occupy a bus
    for pocket in sorted(nx.connected_components(current), key=lambda c: sorted(map(str, c))[0]):
        if anchor in pocket:
            continue
        held = pocket & occupied
        if not held or not nx.has_path(graph, next(iter(held)), anchor):
            continue                    # nothing to connect, or not ours to connect
        mw = _occupant_mw(network, pocket)
        on = {"generator": gens.index[gens["bus"].isin(pocket)], "storage_unit": su.index[su["bus"].isin(pocket)]}
        source = gens["data_source"].reindex(on["generator"]).fillna("").astype(str) \
            if "data_source" in gens.columns else pd.Series("", index=on["generator"])
        reason = None
        if not held <= new_buses:
            reason = "an existing bus"
        elif pocket & loads:
            reason = "load"
        elif stores["bus"].isin(pocket).any():
            reason = "a store"
        elif pocket & link_buses:
            reason = "a live link"
        elif mw > max_mw + 1e-9:
            reason = "above the threshold"
        elif source.isin(_PROTECTED_PLACEMENTS).any() or any(
                _explicit_onshore_connection(g, network.buses.index) for g in on["generator"]):
            reason = "an evidenced connection"
        if reason is not None:
            out["stranded_pockets_kept"].append({"buses": sorted(held), "occupant_mw": round(mw, 1),
                                                 "reason": reason})
            continue
        moves = []
        for component, names in on.items():
            table = network.generators if component == "generator" else network.storage_units
            for bus in sorted(set(table.loc[names, "bus"])):
                targets, level = _rehome_target_buses(network, main, pocket, bus)
                if not len(targets):
                    continue
                rows = pd.DataFrame({"bus": table.loc[names, "bus"]})
                rows = rows[rows["bus"].eq(bus)]
                rows["site_name"] = rows.index
                rows["technology"] = table.loc[rows.index, "carrier"].astype(str)
                excluded = network.buses.index.difference(targets)
                moved, log_moves = reassign_excluded_placements(rows, network, excluded, protected_sources=(),
                                                                log=log)
                for name, r in log_moves.set_index("site_name").iterrows() if len(log_moves) else []:
                    moves.append({"component": component, "name": name, "carrier": r["carrier"],
                                  "p_nom": float(table.at[name, "p_nom"]), "from_bus": r["from_bus"],
                                  "to_bus": r["to_bus"], "distance_km": round(float(r["distance_km"]), 2),
                                  "v_nom_from": float(network.buses.at[r["from_bus"], "v_nom"]),
                                  "v_nom_to": float(network.buses.at[r["to_bus"], "v_nom"]),
                                  "same_voltage": bool(np.isclose(level, network.buses.at[r["from_bus"], "v_nom"]))})
        moved_names = {m["name"] for m in moves}
        if moved_names != set(on["generator"]) | set(on["storage_unit"]):
            out["stranded_pockets_kept"].append({"buses": sorted(held), "occupant_mw": round(mw, 1),
                                                 "reason": "no main-grid bus to move to"})
            continue
        for m in moves:
            table = network.generators if m["component"] == "generator" else network.storage_units
            table.at[m["name"], "bus"] = m["to_bus"]
            occupied.add(m["to_bus"])
        occupied -= held
        out["rehomed"].extend(moves)


def _restore_appendix_b(network, record: dict, plans: list, out: dict) -> dict:
    """Put back what the selected rows removed and changed; return the additions to remove."""
    from scripts.network_build.appendix_b_record import COMPONENT_CLASS, COMPONENT_TABLE
    units = record["units"]
    readd = [p for p in plans if p["target_exists"]
             and p["name"] not in getattr(network, COMPONENT_TABLE[p["component"]]).index]
    # 1. buses at the ends of the branches to restore, where the build or a later step removed them
    needed = []
    for p in readd:
        if p["full"] is None:
            out["conflicts"].append({"component": p["component"], "name": p["name"],
                                     "reason": "absent with no recorded row to restore"})
            continue
        for end in ("bus0", "bus1"):
            bus = p["full"].get(end)
            if bus not in network.buses.index and bus not in needed:
                needed.append(bus)
    for bus in needed:
        row = record["bus_rows"].get(bus)
        if row is None:
            raise ValueError(f"revert_appendix_b: bus {bus} of a restored branch is not recorded")
        network.add("Bus", bus, **_restorable(network, "bus", row))
        out["buses_restored"].append(bus)

    # 2. branches the selected rows removed (re-terminations undone) or the build pruned with them
    restored = {"line": [], "transformer": [], "link": []}
    for p in readd:
        if p["full"] is None:
            continue
        component, name = p["component"], p["name"]
        values = _restorable(network, component, p["full"])
        for field, value in p["target"].items():
            if value is None:
                values.pop(field, None)
            else:
                values[field] = value
        values = {k: v for k, v in _in_network_units(network, component, values, units).items() if v is not None}
        network.add(COMPONENT_CLASS[component], name, **values)
        restored[component].append(name)
        out["restored"].append({"component": component, "name": name, "rows": p["rows"],
                                "bus0": values["bus0"], "bus1": values["bus1"],
                                "s_nom": values.get("s_nom", values.get("p_nom")),
                                **({"pruned_island": p["islands"]} if p["islands"] else {}),
                                **({"replayed_rows": p["replayed"]} if p["replayed"] else {})})

    # 3. ratings and impedances without the selected change rows
    rerated = {"line": [], "transformer": []}
    for p in plans:
        component, name = p["component"], p["name"]
        table = getattr(network, COMPONENT_TABLE[component])
        if not p["target_exists"] or name not in table.index or name in restored[component]:
            continue
        if p["built_exists"] and p["target"] == p["built"]:
            continue                    # the selected rows left this component's fields as they were
        if component == "link":
            if p["target"].get("p_nom") is not None and float(table.at[name, "p_nom"]) != float(p["target"]["p_nom"]):
                table.at[name, "p_nom"] = float(p["target"]["p_nom"])
            continue
        s_nom_now = float(table.at[name, "s_nom"])
        if p["built"].get("s_nom") is not None and abs(s_nom_now - float(p["built"]["s_nom"])) > 1e-6:
            out["modified_after_build"].append({"component": component, "name": name, "s_nom_now": s_nom_now,
                                                "s_nom_built": p["built"]["s_nom"]})
        x_now = float(table.at[name, "x"])
        _set_branch_fields(network, component, name, p["target"], units)
        rerated[component].append(name)
        if abs(float(table.at[name, "s_nom"]) - s_nom_now) > 1e-9 or abs(float(table.at[name, "x"]) - x_now) > 1e-12:
            out["rating_reverted"].append({"component": component, "name": name, "rows": p["rows"],
                                           "s_nom_after": s_nom_now, "s_nom_before": float(table.at[name, "s_nom"]),
                                           **({"replayed_rows": p["replayed"]} if p["replayed"] else {})})
    additions = [p for p in plans if not p["target_exists"]
                 and p["name"] in getattr(network, COMPONENT_TABLE[p["component"]]).index]
    return {"additions": additions,
            "complete_lines": restored["line"] + rerated["line"],
            "complete_transformers": restored["transformer"] + rerated["transformer"]}


def _remove_appendix_b_additions(network, graph: nx.MultiGraph, occupied: set, plans: list, todo: dict,
                                 out: dict, touched: set) -> None:
    """Remove the selected rows' new branches (connection rule), then the buses left empty."""
    from scripts.network_build.appendix_b_record import COMPONENT_CLASS, COMPONENT_TABLE
    live_added, weight_of = [], {}
    for p in todo["additions"]:
        table = getattr(network, COMPONENT_TABLE[p["component"]])
        key = f"{COMPONENT_TABLE[p['component']]}:{p['name']}"
        b0, b1 = table.at[p["name"], "bus0"], table.at[p["name"], "bus1"]
        touched |= {b0, b1}
        if graph.has_edge(b0, b1, key=key):
            live_added.append((b0, b1, key))
            weight_of[key] = float(table.at[p["name"], "p_nom" if p["component"] == "link" else "s_nom"])
    keeps = _connection_keeps(graph, live_added, occupied, weight_of)
    for p in todo["additions"]:
        component, name = p["component"], p["name"]
        table = getattr(network, COMPONENT_TABLE[component])
        bus0, bus1 = table.loc[name, ["bus0", "bus1"]]
        key = f"{COMPONENT_TABLE[component]}:{name}"
        live = graph.has_edge(bus0, bus1, key=key)
        rating = float(table.at[name, "p_nom" if component == "link" else "s_nom"])

        def kept(served):
            out["kept_as_connection"].append({
                "component": component, "name": name, "rows": p["rows"], "bus0": bus0, "bus1": bus1,
                "s_nom": rating, "v_nom": float(network.buses.at[bus0, "v_nom"]),
                "connects": sorted(served), "occupant_mw": round(_occupant_mw(network, served), 1),
                "occupants": _occupants(network, served)[:20]})

        if live and key in keeps:
            kept(keeps[key])
            continue
        if live:
            graph.remove_edge(bus0, bus1, key=key)
            stranded = _stranded(graph, bus0, bus1, occupied)
            if stranded:                # a safety net: the planned keeps should leave none
                graph.add_edge(bus0, bus1, key=key)
                kept(stranded)
                continue
        if component == "link":
            # kept with p_nom = 0, as remove_components does (the Stage 1 schedule names links)
            network.links.at[name, "p_nom"] = 0.0
            if "p_nom_max" in network.links.columns:
                network.links.at[name, "p_nom_max"] = 0.0
            out["links_disabled"].append({"name": name, "rows": p["rows"]})
            continue
        network.remove(COMPONENT_CLASS[component], name)
        out["removed"].append({"component": component, "name": name, "rows": p["rows"],
                               "bus0": bus0, "bus1": bus1, "s_nom": rating})

    # New buses (or pockets) the removals left off the main grid with nothing on them
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    one_ports = [t for t in ("generators", "loads", "storage_units", "stores", "shunt_impedances")
                 if len(getattr(network, t))]
    left = []
    for comp in components[1:]:
        if not (comp & touched):
            continue
        if comp & occupied:
            left.append(comp)
            continue
        links = network.links[network.links["bus0"].isin(comp) | network.links["bus1"].isin(comp)]
        attached = {t: getattr(network, t).index[getattr(network, t)["bus"].isin(comp)] for t in one_ports}
        non_slack = [n for t, names in attached.items() for n in names
                     if t != "generators" or network.generators.at[n, "carrier"] not in SLACK_CARRIERS]
        if (links["p_nom"] > 0).any() or non_slack:
            left.append(comp)           # something other than slack sits there: leave it and say so
            continue
        for component, table in (("line", network.lines), ("transformer", network.transformers)):
            inside = table.index[table["bus0"].isin(comp) | table["bus1"].isin(comp)]
            for name in inside:
                network.remove(COMPONENT_CLASS[component], name)
                out["removed"].append({"component": component, "name": name, "rows": [], "dead_island": True})
        # a disabled link (kept for the Stage 1 schedule) still names its buses: they stay, isolated
        named = set(links["bus0"]) | set(links["bus1"])
        drop = sorted(set(comp) - named)
        if named & set(comp):
            out.setdefault("buses_kept_for_disabled_links", []).extend(sorted(named & set(comp)))
        gens = network.generators.index[network.generators["bus"].isin(drop)]
        if len(gens):
            network.remove("Generator", list(gens))
            out["slack_generators_removed"].extend(gens.tolist())
        if drop:
            network.remove("Bus", drop)
            out["buses_removed"].extend(drop)
        graph.remove_nodes_from(drop)
    if left:
        out["islands_left"] = [sorted(c) for c in left]


def revert_appendix_b(network, selector: dict, graph_hook=None, log: logging.Logger | None = None,
                      projects_file=None, rehome_max_mw: float = DEFAULT_REHOME_STRANDED_MAX_MW) -> dict:
    """Undo the Appendix B rows ``selector`` picks (see the module docstring); return the record.

    ``graph_hook(graph, occupied)``, when given, runs after the restores and before the
    selected additions are removed (``apply_network_delivery`` removes ``remove_components``
    there, on the same graph). Small embedded plant on new buses the revert strands is then
    re-homed (``_rehome_stranded_embedded``; ``rehome_max_mw`` 0 turns it off) before the
    connection rule keeps any added branch.
    """
    from scripts.network_build.appendix_b_record import read_appendix_b_record
    log = log or logging.getLogger(__name__)
    record = read_appendix_b_record(network)
    if record is None:
        raise ValueError("network_delivery.revert_appendix_b: the network records no Appendix B works "
                         "(network.meta['appendix_b_works']); rebuild it with the recording build "
                         "(ETYS_upgrades.record_appendix_b_works, 15 Oct 2026)")
    _check_tags(network, record)
    chosen, summary = select_appendix_b_rows(record, selector, projects_file)
    plans, islands = plan_appendix_b_revert(record, chosen)
    out = {**summary, "islands_restored": sorted(islands), "buses_restored": [], "restored": [],
           "rating_reverted": [], "removed": [], "links_disabled": [], "kept_as_connection": [],
           "buses_removed": [], "slack_generators_removed": [], "conflicts": [], "modified_after_build": [],
           "rehomed": [], "stranded_pockets_kept": []}
    for p in plans:
        for c in p["conflicts"]:
            out["conflicts"].append({"component": p["component"], "name": p["name"], **c})
    filled = _filled_season_cells(network)
    todo = _restore_appendix_b(network, record, plans, out)

    graph = _graph(network)
    added_links = [p["name"] for p in todo["additions"] if p["component"] == "link"]
    occupied = _occupied_buses(network, ignore_links=added_links)
    if graph_hook is not None:
        graph_hook(graph, occupied)
    _rehome_stranded_embedded(network, graph, occupied, todo, record, out, rehome_max_mw, log)
    touched = set(record.get("buses_added", [])) | set(out["buses_restored"])
    for p in plans:
        row = p["full"] or {}
        touched |= {row.get("bus0"), row.get("bus1")} - {None}
    _remove_appendix_b_additions(network, graph, occupied, plans, todo, out, touched)
    _refill_seasonal(network, filled, todo["complete_lines"], log)
    _complete_transformers_and_licensee(network, todo["complete_lines"], todo["complete_transformers"])
    return out


def apply_network_delivery(network, scenario_config: dict, balancing_config: dict | None,
                           logger: logging.Logger | None = None, projects_file=None) -> dict | None:
    """Apply the case to ``network`` and ``scenario_config`` (in place); return the record, or ``None``.

    Run before the line rating overrides and the outage schedule, so restored branches
    take the BM voltage floor and the secured boundary specs see the case's capability
    years and branches.
    """
    log = logger or logging.getLogger(__name__)
    block = network_delivery_config(balancing_config)
    if block is None:
        return None
    record = {"case": block["case"], "description": block.get("description", ""), "removed": [],
              "links_disabled": [], "kept_as_connection": [], "secured_capability": {}}

    # Links first: a disabled link no longer connects anything for the island check.
    for entry in block.get("remove_components", []) or []:
        if entry["component"] != "link":
            continue
        for name in _matches(network, entry):
            network.links.at[name, "p_nom"] = 0.0
            if "p_nom_max" in network.links.columns:
                network.links.at[name, "p_nom_max"] = 0.0
            record["links_disabled"].append(name)

    def remove_listed(graph, occupied):
        for entry in block.get("remove_components", []) or []:
            if entry["component"] == "link":
                continue
            table = COMPONENT_TABLES[entry["component"]]
            for name in _matches(network, entry):
                bus0, bus1 = getattr(network, table).loc[name, ["bus0", "bus1"]]
                graph.remove_edge(bus0, bus1, key=f"{table}:{name}")
                if _strands_occupied_bus(graph, bus0, bus1, occupied):
                    graph.add_edge(bus0, bus1, key=f"{table}:{name}")
                    record["kept_as_connection"].append({"component": entry["component"], "name": name})
                    continue
                network.remove(entry["component"].capitalize(), name)
                record["removed"].append({"component": entry["component"], "name": name})

    # Appendix B rows: what they removed and changed goes back before the island check, so
    # the old circuits connect what the new ones did wherever they can.
    selector = block.get(REVERT_KEY)
    if selector:
        record["appendix_b"] = revert_appendix_b(
            network, selector, remove_listed, log, projects_file,
            rehome_max_mw=rehome_stranded_max_mw(block))
    else:
        remove_listed(_graph(network), _occupied_buses(network))

    cap_block = block.get("secured_capability") or {}
    if cap_block:
        outage = (scenario_config.setdefault("network", {}).setdefault("transmission", {})
                  .setdefault("outage_schedule", {}))
        secured = copy.deepcopy(outage.get("secured_capability") or {})
        for key, value in cap_block.items():
            secured[key] = copy.deepcopy(value)
        outage["secured_capability"] = secured
        record["secured_capability"] = {k: secured.get(k) for k in cap_block}

    tx_block = block.get("transmission") or {}
    if tx_block:
        transmission = scenario_config.setdefault("network", {}).setdefault("transmission", {})
        for key, value in tx_block.items():
            if key == "bm_voltage_floor":
                transmission["bm_voltage_floor"] = {**(transmission.get("bm_voltage_floor") or {}),
                                                    **copy.deepcopy(value or {})}
            else:
                transmission[key] = copy.deepcopy(value)
        record["transmission"] = {k: copy.deepcopy(transmission.get(k)) for k in tx_block}
        log.info(f"Network delivery case '{record['case']}': balancing-stage transmission {record['transmission']}")

    log.info(f"Network delivery case '{record['case']}': removed {len(record['removed'])} branches, "
             f"disabled {len(record['links_disabled'])} links, kept {len(record['kept_as_connection'])} "
             f"as connection assets; secured capability {record['secured_capability']}")
    for item in record["kept_as_connection"]:
        log.info(f"  kept {item['component']} {item['name']}: removing it would strand generation, load or storage")
    if selector:
        d = record["appendix_b"]
        log.info(f"  Appendix B rows reverted ({d['selector']}): {d['rows_selected']} rows {d['by_year']}; "
                 f"{len(d['restored'])} branches restored ({len(d['buses_restored'])} buses), "
                 f"{len(d['rating_reverted'])} ratings reverted, {len(d['removed'])} branches removed, "
                 f"{len(d['links_disabled'])} links disabled, {len(d['buses_removed'])} buses dropped, "
                 f"{len(d['kept_as_connection'])} kept as connections "
                 f"({sum(k['s_nom'] for k in d['kept_as_connection']):,.0f} MVA), "
                 f"{len(d['conflicts'])} replay conflicts")
        if d["rehomed"]:
            log.info(f"  re-homed {len(d['rehomed'])} components ({sum(m['p_nom'] for m in d['rehomed']):,.1f} MW) "
                     f"off {len({m['from_bus'] for m in d['rehomed']})} stranded new buses "
                     f"(pockets up to {d['rehome_max_mw']:g} MW)")
        for m in d["rehomed"]:
            log.info(f"  re-homed {m['component']} {m['name']} ({m['carrier']}, {m['p_nom']:.1f} MW): "
                     f"{m['from_bus']} -> {m['to_bus']} ({m['distance_km']:.1f} km, "
                     f"{m['v_nom_from']:.0f} -> {m['v_nom_to']:.0f} kV)")
        for item in d["kept_as_connection"]:
            log.info(f"  kept {item['component']} {item['name']} ({', '.join(item['rows'])}; {item['s_nom']:.0f} MVA): "
                     f"connects {', '.join(item['connects'])} ({item['occupant_mw']:.0f} MW; "
                     f"{'; '.join(item['occupants'][:5])})")
        for item in d["conflicts"]:
            log.warning(f"  Appendix B replay conflict: {item}")
        for item in d["modified_after_build"]:
            log.warning(f"  Appendix B branch re-rated after the build, reverted from its record: {item}")
        for island in d.get("islands_left", []):
            log.warning(f"  Appendix B revert left an island with components on it: {island[:8]}")
    network.meta = dict(network.meta or {}, network_delivery=json.dumps(record))
    return record
