"""Scenario-source registries as data: pathways, GSP aliases and ES1 line maps.

Three tables, each read here and nowhere else:

* ``data/scenario_sources/pathways.csv``: every pathway of every source edition (FES 2020 to
  2025, CP30), with its short code, role (net_zero, counterfactual, forecast), whether the
  building blocks carry it, its successor in the next edition, and the other spellings the
  source uses for it (``aliases``, ``;``-separated: FES 2020's building blocks print "LW" for
  "Leading the Way", FES 2021's "Leading The Way").
* ``data/FES/gsp_aliases.csv``: the GSP name clean-up per FES edition, applied in ``step`` order
  (``rename`` exact or substring, ``drop`` exact or contains, ``strip`` whitespace, ``none`` for an
  edition that needs nothing). An edition with no rows is unknown and fails.
* ``data/scenario_sources/es1_line_maps.csv``: ES1 supply and storage lines to model carriers, per
  source and edition.

Lookups fail with the known values listed; nothing falls back to another edition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHWAYS_FILE = PROJECT_ROOT / "data" / "scenario_sources" / "pathways.csv"
GSP_ALIASES_FILE = PROJECT_ROOT / "data" / "FES" / "gsp_aliases.csv"
ES1_LINE_MAPS_FILE = PROJECT_ROOT / "data" / "scenario_sources" / "es1_line_maps.csv"

PATHWAY_ROLES = ("net_zero", "counterfactual", "forecast")
SOURCES = ("FES", "CP30")
GSP_ALIAS_ACTIONS = {"rename": ("exact", "substring"), "drop": ("exact", "contains"), "strip": ("",), "none": ("",)}
ES1_LINE_ROLES = ("supply", "storage")


class UnknownPathway(KeyError):
    """No registry row for this source, edition and pathway."""


class UnknownEdition(KeyError):
    """The registry has no rows for this source edition."""


def _norm(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _source(source: str) -> str:
    key = _norm(source).upper()
    if key not in SOURCES:
        raise ValueError(f"Unknown scenario source {source!r}; expected one of {SOURCES}")
    return key


def pathway_slug(name: str) -> str:
    """File-name slug of a pathway name: 'Holistic Transition' -> 'holistic_transition'."""
    return re.sub(r"[^a-z0-9]+", "_", _norm(name).lower()).strip("_")


# ───────────────────────────────── pathways ─────────────────────────────────

@dataclass(frozen=True)
class Pathway:
    """One pathway of one source edition."""

    source: str
    edition: int
    name: str
    code: str
    role: str
    in_building_blocks: bool
    successor: str
    aliases: tuple[str, ...]
    note: str

    @property
    def slug(self) -> str:
        return pathway_slug(self.name)

    @property
    def names(self) -> tuple[str, ...]:
        """The name and every alias, as the source prints them."""
        return (self.name, *self.aliases)


def _as_bool(value: object) -> bool:
    text = _norm(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Not a boolean: {value!r}")


@lru_cache(maxsize=4)
def _pathways(path: str) -> tuple[Pathway, ...]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"source", "edition", "pathway_name", "code", "role", "in_building_blocks", "successor",
                "aliases", "note"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    rows = []
    for _, row in table.iterrows():
        role = _norm(row["role"])
        if role not in PATHWAY_ROLES:
            raise ValueError(f"{path}: pathway {row['pathway_name']!r} has role {role!r}; expected {PATHWAY_ROLES}")
        rows.append(Pathway(
            source=_source(row["source"]), edition=int(row["edition"]), name=_norm(row["pathway_name"]),
            code=_norm(row["code"]), role=role, in_building_blocks=_as_bool(row["in_building_blocks"]),
            successor=_norm(row["successor"]),
            aliases=tuple(a.strip() for a in _norm(row["aliases"]).split(";") if a.strip()),
            note=_norm(row["note"])))
    seen_codes: set[tuple[str, int, str]] = set()
    seen_names: dict[tuple[str, int, str], str] = {}
    for p in rows:
        key = (p.source, p.edition, p.code)
        if key in seen_codes:
            raise ValueError(f"{path}: code {p.code!r} twice in {p.source} {p.edition}")
        seen_codes.add(key)
        for name in p.names:
            name_key = (p.source, p.edition, name.casefold())
            if name_key in seen_names and seen_names[name_key] != p.code:
                raise ValueError(f"{path}: {name!r} names two pathways in {p.source} {p.edition}")
            seen_names[name_key] = p.code
    return tuple(rows)


def pathways(source: Optional[str] = None, edition: Optional[int] = None,
             path: str | Path = PATHWAYS_FILE) -> list[Pathway]:
    """Registry rows, optionally for one source and edition.

    Args:
        source: "FES" or "CP30"; None for all.
        edition: Edition (publication year); None for all.
        path: The pathways table.

    Returns:
        The matching pathways, in table order.

    Raises:
        UnknownEdition: ``edition`` given and the source has no rows for it.
    """
    rows = list(_pathways(str(path)))
    if source is not None:
        source = _source(source)
        rows = [p for p in rows if p.source == source]
    if edition is not None:
        known = sorted({p.edition for p in rows})
        rows = [p for p in rows if p.edition == int(edition)]
        if not rows:
            raise UnknownEdition(f"No {source or 'source'} edition {edition} in {path}; known editions {known}")
    return rows


def editions(source: str, path: str | Path = PATHWAYS_FILE) -> list[int]:
    """Editions of a source with at least one pathway."""
    return sorted({p.edition for p in pathways(source, path=path)})


def pathway_by_name(source: str, edition: int, name: str, path: str | Path = PATHWAYS_FILE) -> Pathway:
    """The pathway a source prints as ``name`` (the name or an alias; case and outer spaces ignored).

    Raises:
        UnknownEdition: The edition is not in the registry.
        UnknownPathway: The edition has no pathway of that name.
    """
    wanted = _norm(name).casefold()
    rows = pathways(source, edition, path=path)
    for p in rows:
        if wanted in {n.casefold() for n in p.names}:
            return p
    raise UnknownPathway(f"{source} {edition} has no pathway {name!r}; known: "
                         f"{sorted(n for p in rows for n in p.names)}")


def pathway_by_code(source: str, edition: int, code: str, path: str | Path = PATHWAYS_FILE) -> Pathway:
    """The pathway with short code ``code`` (e.g. "HT") in a source edition.

    Raises:
        UnknownEdition: The edition is not in the registry.
        UnknownPathway: The edition has no pathway with that code.
    """
    wanted = _norm(code).upper()
    rows = pathways(source, edition, path=path)
    for p in rows:
        if p.code.upper() == wanted:
            return p
    raise UnknownPathway(f"{source} {edition} has no pathway code {code!r}; known: {[p.code for p in rows]}")


def pathway_code(source: str, edition: int, name: str, path: str | Path = PATHWAYS_FILE) -> str:
    """Short code of a pathway name: ``pathway_code("FES", 2025, "Holistic Transition") == "HT"``."""
    return pathway_by_name(source, edition, name, path=path).code


def pathway_names(source: str, edition: int, path: str | Path = PATHWAYS_FILE) -> dict[str, str]:
    """{slug: name} for a source edition, the shape of ``fes2025_targets.FES2025_PATHWAYS``."""
    return {p.slug: p.name for p in pathways(source, edition, path=path)}


def successor(source: str, edition: int, code: str, path: str | Path = PATHWAYS_FILE) -> Optional[Pathway]:
    """The pathway that continues ``code`` in the next edition of the source, or None if none is declared."""
    current = pathway_by_code(source, edition, code, path=path)
    if not current.successor:
        return None
    later = [e for e in editions(source, path=path) if e > int(edition)]
    if not later:
        return None
    return pathway_by_code(source, later[0], current.successor, path=path)


# ─────────────────────────────── GSP aliases ────────────────────────────────

@dataclass(frozen=True)
class GspAlias:
    """One GSP clean-up step of one FES edition."""

    edition: int
    step: int
    action: str
    match: str
    raw: str
    canonical: str
    reason: str


@lru_cache(maxsize=4)
def _gsp_aliases(path: str) -> tuple[GspAlias, ...]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for _, row in table.iterrows():
        action, match = _norm(row["action"]), _norm(row["match"])
        if action not in GSP_ALIAS_ACTIONS or match not in GSP_ALIAS_ACTIONS[action]:
            raise ValueError(f"{path}: unsupported GSP alias action/match {action!r}/{match!r} "
                             f"(edition {row['edition']} step {row['step']})")
        # Raw and canonical are kept unstripped: a GSP name's spaces are part of the match.
        rows.append(GspAlias(edition=int(row["edition"]), step=int(row["step"]), action=action, match=match,
                             raw=str(row["raw"]), canonical=str(row["canonical"]), reason=_norm(row["reason"])))
    return tuple(sorted(rows, key=lambda a: (a.edition, a.step)))


def gsp_aliases(edition: int, path: str | Path = GSP_ALIASES_FILE) -> list[GspAlias]:
    """The GSP clean-up steps of one FES edition, in order.

    Raises:
        UnknownEdition: The table has no rows for the edition (add them, or one ``none`` row).
    """
    rows = [a for a in _gsp_aliases(str(path)) if a.edition == int(edition)]
    if not rows:
        known = sorted({a.edition for a in _gsp_aliases(str(path))})
        raise UnknownEdition(f"No GSP aliases for FES {edition} in {path}; known editions {known}. "
                             "Add the edition's rows (a single 'none' row if it needs no clean-up).")
    return rows


def apply_gsp_aliases(frame: pd.DataFrame, edition: int, column: str = "GSP",
                      path: str | Path = GSP_ALIASES_FILE) -> pd.DataFrame:
    """Rename and drop GSP names of one FES edition as the alias table says.

    Args:
        frame: Table with a GSP name column.
        edition: FES edition.
        column: The GSP name column.
        path: The alias table.

    Returns:
        A copy with the names renamed and the dropped rows removed (index kept).
    """
    out = frame.copy()
    for alias in gsp_aliases(edition, path=path):
        names = out[column]
        if alias.action == "rename" and alias.match == "substring":
            out[column] = names.str.replace(alias.raw, alias.canonical, regex=False)
        elif alias.action == "rename":
            out[column] = names.where(names != alias.raw, alias.canonical)
        elif alias.action == "drop" and alias.match == "contains":
            out = out[~names.str.contains(alias.raw, regex=False, na=False)]
        elif alias.action == "drop":
            out = out[names != alias.raw]
        elif alias.action == "strip":
            out[column] = names.str.strip()
    return out


# ────────────────────────────── ES1 line maps ───────────────────────────────

@dataclass(frozen=True)
class Es1Line:
    """One ES1 line (Category, Type, SubType) and its model carrier."""

    source: str
    edition: int
    role: str
    category: str
    type: str
    subtype: str
    carrier: str
    note: str


@lru_cache(maxsize=4)
def _es1_lines(path: str) -> tuple[Es1Line, ...]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for _, row in table.iterrows():
        role = _norm(row["role"])
        if role not in ES1_LINE_ROLES:
            raise ValueError(f"{path}: ES1 line role {role!r}; expected {ES1_LINE_ROLES}")
        rows.append(Es1Line(source=_source(row["source"]), edition=int(row["edition"]), role=role,
                            category=_norm(row["category"]), type=_norm(row["type"]),
                            subtype=_norm(row["subtype"]), carrier=_norm(row["carrier"]), note=_norm(row["note"])))
    return tuple(rows)


def es1_line_map(source: str, edition: int, role: str = "supply",
                 path: str | Path = ES1_LINE_MAPS_FILE) -> list[Es1Line]:
    """ES1 lines of one source edition and role, in table order.

    Raises:
        UnknownEdition: No map for the source edition and role.
    """
    source = _source(source)
    if role not in ES1_LINE_ROLES:
        raise ValueError(f"ES1 line role {role!r}; expected {ES1_LINE_ROLES}")
    rows = [line for line in _es1_lines(str(path))
            if line.source == source and line.edition == int(edition) and line.role == role]
    if not rows:
        known = sorted({(line.source, line.edition) for line in _es1_lines(str(path))})
        raise UnknownEdition(f"No ES1 {role} line map for {source} {edition} in {path}; known {known}")
    return rows


def es1_carrier_lookup(source: str, edition: int, role: str = "supply",
                       path: str | Path = ES1_LINE_MAPS_FILE) -> dict[tuple[str, ...], str]:
    """{(category, type, subtype): carrier} for supply; {(type, subtype): carrier} for storage."""
    lines = es1_line_map(source, edition, role, path=path)
    if role == "storage":
        return {(line.type, line.subtype): line.carrier for line in lines}
    return {(line.category, line.type, line.subtype): line.carrier for line in lines}


def clear_caches() -> None:
    """Forget the tables read (tests that write their own)."""
    _pathways.cache_clear()
    _gsp_aliases.cache_clear()
    _es1_lines.cache_clear()


def known_names(source: str, edition: int, path: str | Path = PATHWAYS_FILE) -> set[str]:
    """Every spelling (names and aliases) of a source edition's pathways."""
    return {n for p in pathways(source, edition, path=path) for n in p.names}


def resolve_all(source: str, edition: int, names: Iterable[str],
                path: str | Path = PATHWAYS_FILE) -> dict[str, str]:
    """{name: code} for every name; the first unknown name fails."""
    return {name: pathway_code(source, edition, name, path=path) for name in names}
