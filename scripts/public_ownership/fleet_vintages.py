"""The vintaged 2050 fleet: a stock-flow build from the FES annual trajectory (FES 2050 plan, task Y10).

Decision 1 of the FES 2050 plan makes the 2050 analysis a calendar-year snapshot
in which every asset carries its build year, so its capital cost and its rate
come from that vintage.  This module decides the build years:

1. **The trajectory.** FES 2025 ES1 installed capacity by technology for every
   year 2024-2050 (:func:`fes_capacity_trajectory`, through Y1's reader
   ``scripts/FES/fes2025_targets.py``).  Every positive ES1 capacity line maps to
   one stock-flow technology (:data:`STOCK_FLOW_TECHNOLOGIES`) or the build fails.
2. **The base stock.** The 2024 fleet is FES 2024 capacity, given an age profile
   by the commissioning years of the existing fleet: REPD operational dates for
   renewables and storage, DUKES 5.11 commissioning years for thermal, nuclear and
   hydro (:func:`existing_fleet_evidence`).  The evidence gives the shape, FES the
   level (every cohort is scaled by FES 2024 / evidence total, declared).  A
   technology with no dated evidence takes a uniform profile over the economic life
   before the base year.
3. **The flows** (:func:`stock_flow`), year by year to the target year, under the
   declared retirement rule (:data:`RETIREMENT_RULES`):

   * a cohort built in year ``b`` with economic life ``L`` (Y2's
     ``technology_economic_lives.csv``) is in the stock for years ``b .. b+L-1``
     and retires in year ``b + L`` (*end of life*);
   * *central rule* (``economic_life_with_extension``): an existing cohort is
     life-extended (no scheduled retirement; it leaves only when FES capacity
     falls, below) when it is already at or past its life in the base year, or
     when FES runs its technology down (target-year capacity below base-year
     capacity), since FES does not replace like for like a fleet it is retiring;
   * existing nuclear: a published closure date later than the base year
     (``data/generators/nuclear_phase_out_dates.csv``) holds the station through
     its closure year, so it retires the year after; a date at or before the base
     year is stale (FES 2024 still has the plant) and is left to the FES decline;
   * *bound rule* (``strict_economic_life``): no life extension, so every existing
     cohort already past its life retires in the first horizon year and every other
     retires at its life; it maximises the post-2025 build and is the
     retirement-rule bound of task Y14;
   * when FES capacity falls below the stock left after scheduled retirements, the
     difference retires early, oldest build year first (*FES decline*);
   * gross additions in year ``t`` = FES(t) - stock after retirements, i.e. the
     net FES change plus that year's retirements; they are new build of year ``t``;
   * the part of a year's additions that replaces that year's end-of-life
     retirements is *repowering*: a repowered site is new build of the repowering
     year, with the new build's cost and rate.

   So the stock in the target year sums to FES capacity in that year for every
   technology, exactly (the conservation test).
4. **The join** (:func:`assign_build_years`): the build years reach the rows of a
   (future) network finance register by technology, then by bus, then pro rata.
   The HT50 network is not built yet, so the join is written and tested on a
   synthetic register.

Outputs: the vintage mix (technology, build year, capacity), the flows by year,
capacity retired by decade, and a summary with every declaration in it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINANCING = PROJECT_ROOT / "data" / "public_ownership" / "financing"
DEFAULT_LIVES = FINANCING / "technology_economic_lives.csv"
DEFAULT_REPD = PROJECT_ROOT / "data" / "renewables" / "repd-q2-jul-2025.csv"
DEFAULT_DUKES = PROJECT_ROOT / "resources" / "generators" / "DUKES" / "DUKES_2023_generators.csv"
DEFAULT_NUCLEAR_CLOSURES = PROJECT_ROOT / "data" / "generators" / "nuclear_phase_out_dates.csv"
DEFAULT_PATHWAY = "Holistic Transition"
BASE_YEAR = 2024
TARGET_YEAR = 2050
#: Build years before this are the legacy fleet in the finance register (config policy_start_year).
POLICY_START_YEAR = 2025

RULE_CENTRAL = "economic_life_with_extension"
RULE_STRICT = "strict_economic_life"
RETIREMENT_RULES: dict[str, str] = {
    RULE_CENTRAL: ("each cohort retires at build year + economic life; an existing cohort is life-extended (no "
                   "scheduled retirement) when it is already at or past its life in the base year or when FES runs "
                   "its technology down (target-year below base-year capacity); existing nuclear with a published "
                   "closure date after the base year runs through that year; when FES capacity falls below the "
                   "surviving stock, the difference retires oldest first; gross additions = FES(t) - surviving "
                   "stock, new build of year t; additions replacing end-of-life retirements are repowering"),
    RULE_STRICT: ("as the central rule, with no life extension: every existing cohort past its economic life retires "
                  "in the first horizon year and every other at its life (the retirement-rule bound, task Y14)"),
}
LIFE_SOURCES = ("central", "fes_e02")

REASON_END_OF_LIFE = "end_of_life"
REASON_PAST_LIFE_AT_BASE = "past_life_at_base_year"
REASON_PUBLISHED_CLOSURE = "published_closure_date"
REASON_FES_DECLINE = "fes_decline_oldest_first"
#: FES 2050 decision 62 (user, 3 Oct 2026; research option 3): the repowering rule's exits.
REASON_REPOWERED_AT_LIFE = "repowered_at_decision_age"
REASON_LIFE_EXTENSION_END = "life_extension_end"
REASON_PAST_MAX_LIFE_AT_BASE = "past_max_life_at_base_year"
#: Exits the year's gross additions replace (repowering).
REPLACED_REASONS = (REASON_END_OF_LIFE, REASON_PAST_LIFE_AT_BASE, REASON_REPOWERED_AT_LIFE,
                    REASON_LIFE_EXTENSION_END, REASON_PAST_MAX_LIFE_AT_BASE)


@dataclass(frozen=True)
class RepoweringRule:
    """FES 2050 decision 62: at the decision age a share of each cohort repowers; the rest runs to the cap.

    ``share`` (0.3 central, 0.5 the named sensitivity) of every cohort leaves at the
    technology's life (the decision age: onshore 25 years, decision 48, the CfD
    end-of-life test for repowering) and is replaced as new build of that year; the
    remainder is life-extended (merchant, operating cost only: decisions 41 (a) and 42)
    and leaves at ``max_life_years`` (35), replaced as new build then. A base-year cohort
    already past the decision age has taken it (its repowered part is gone from the
    operational record) and runs to the cap; one past the cap leaves in the first year.
    """

    share: float
    max_life_years: float

    def __post_init__(self):
        if not 0.0 <= float(self.share) <= 1.0:
            raise ValueError(f"repowering share {self.share} is not in [0, 1]")
        if not float(self.max_life_years) > 0:
            raise ValueError("repowering max_life_years must be positive")


def parse_repowering(items) -> dict[str, RepoweringRule]:
    """``["Onshore wind=0.3:35"]`` (the command line) or ``{"Onshore wind": {"share": 0.3, "max_life_years": 35}}``.

    Empty or None: no repowering rule (every technology on its life, as before).
    """
    if not items:
        return {}
    out: dict[str, RepoweringRule] = {}
    if isinstance(items, Mapping):
        for name, spec in items.items():
            spec = dict(spec or {})
            unknown = sorted(set(spec) - {"share", "max_life_years"})
            if unknown or "share" not in spec or "max_life_years" not in spec:
                raise ValueError(f"repowering.{name}: needs share and max_life_years only (got {sorted(spec)})")
            out[str(name)] = RepoweringRule(float(spec["share"]), float(spec["max_life_years"]))
    else:
        for item in items:
            name, sep, value = str(item).rpartition("=")
            share, colon, cap = value.partition(":")
            if not sep or not colon or not name.strip():
                raise ValueError(f"repowering {item!r}: expected TECHNOLOGY=SHARE:MAX_LIFE")
            out[name.strip()] = RepoweringRule(float(share), float(cap))
    if unknown := sorted(set(out) - set(TECHNOLOGIES)):
        raise ValueError(f"repowering names unknown stock-flow technologies {unknown}")
    return out


def repowering_flag(rules: Mapping[str, RepoweringRule] | None) -> str:
    """The command-line form of :func:`parse_repowering` (``''`` without a rule)."""
    items = [f'"{name}={rule.share:g}:{rule.max_life_years:g}"' for name, rule in sorted((rules or {}).items())]
    return f" --repowering {' '.join(items)}" if items else ""


@dataclass(frozen=True)
class RefurbishmentRule:
    """F28 (d): technologies refurbished, not rebuilt, at the end of their life (strict rule only).

    Under the strict rule every cohort leaves at its life and its site is repowered as new
    build of that year. Hydro and pumped hydro are refurbished instead: the replacement is
    the same plant refurbished (the register costs it at a refurbishment share of new-build
    capex, the mix's ``refurbishment_share``), and a cohort already past its life in the base
    year is refurbished on a schedule, oldest first and evenly by capacity over ``first_year``
    to ``last_year``, rather than all at once in the first horizon year. ``windows`` gives a
    technology its own years: the pumped-storage stations are being refurbished now (ENGIE's
    Dinorwig and Ffestiniog programme to about 2036; Cruachan's units 2026-27), while SSE
    refurbishes its conventional hydro one station at a time (Tummel Bridge 2022-24, Lochay
    2025-28), which fits the even 2026-2045 schedule (sources 27 Oct 2026, docs F28).
    """

    technologies: tuple[str, ...]
    first_year: int
    last_year: int
    windows: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self):
        if unknown := sorted(set(self.technologies) - set(TECHNOLOGIES)):
            raise ValueError(f"refurbishment names unknown stock-flow technologies {unknown}")
        if int(self.last_year) < int(self.first_year):
            raise ValueError("refurbishment last_year precedes first_year")
        for name, first, last in self.windows:
            if name not in self.technologies:
                raise ValueError(f"refurbishment window for {name!r}, which is not a refurbished technology")
            if int(last) < int(first):
                raise ValueError(f"refurbishment window for {name!r}: last year precedes first year")

    def window(self, technology: str) -> tuple[int, int]:
        """The (first, last) refurbishment years of one technology."""
        for name, first, last in self.windows:
            if name == technology:
                return int(first), int(last)
        return int(self.first_year), int(self.last_year)


def parse_refurbishment(spec) -> RefurbishmentRule | None:
    """``{"technologies": [...], "first_year": 2026, "last_year": 2045, "windows": {TECH: [2026, 2036]}}``
    (config; ``windows`` optional) or ``"TECH;TECH=2026:2045|TECH=2026:2036"`` (the command line, each
    ``|`` segment one technology's window); None or empty: no refurbishment rule."""
    if not spec:
        return None
    if isinstance(spec, Mapping):
        unknown = sorted(set(spec) - {"technologies", "first_year", "last_year", "windows"})
        if unknown:
            raise ValueError(f"refurbishment has unknown keys {unknown}")
        windows = tuple((str(name), int(years[0]), int(years[1]))
                        for name, years in dict(spec.get("windows") or {}).items())
        return RefurbishmentRule(tuple(str(t) for t in spec["technologies"]), int(spec["first_year"]),
                                 int(spec["last_year"]), windows)
    base, *segments = str(spec).split("|")
    names, sep, years = base.rpartition("=")
    first, colon, last = years.partition(":")
    if not sep or not colon:
        raise ValueError(f"refurbishment {spec!r}: expected TECH;TECH=FIRST:LAST")
    windows = []
    for segment in segments:
        name, sep, years = segment.rpartition("=")
        w_first, colon, w_last = years.partition(":")
        if not sep or not colon:
            raise ValueError(f"refurbishment window {segment!r}: expected TECH=FIRST:LAST")
        windows.append((name.strip(), int(w_first), int(w_last)))
    return RefurbishmentRule(tuple(n.strip() for n in names.split(";") if n.strip()), int(first), int(last),
                             tuple(windows))


def refurbishment_flag(rule: RefurbishmentRule | None) -> str:
    """The command-line form of :func:`parse_refurbishment` (``''`` without a rule)."""
    if rule is None:
        return ""
    windows = "".join(f"|{name}={first}:{last}" for name, first, last in rule.windows)
    return f' --refurbishment "{";".join(rule.technologies)}={rule.first_year}:{rule.last_year}{windows}"'


def refurbishment_years(cohorts: Iterable[tuple[int, float]], rule: RefurbishmentRule,
                        technology: str | None = None) -> dict[int, int]:
    """Each past-life build year's refurbishment year: oldest first, evenly by capacity over the
    rule's years (``technology``'s own window when it has one)."""
    first, last = rule.window(technology) if technology is not None else (int(rule.first_year), int(rule.last_year))
    cells = sorted((int(y), float(c)) for y, c in cohorts)
    total = sum(c for _, c in cells)
    span = last - first + 1
    out, before = {}, 0.0
    for year, capacity in cells:
        mid = (before + capacity / 2.0) / total if total > 0 else 0.0
        out[year] = first + min(span - 1, int(mid * span))
        before += capacity
    return out


def _split_cohort(build_year: int, capacity: float, life: float, rule: RepoweringRule, source: str) -> list:
    """A new cohort under the repowering rule: the share leaving at the decision age, the rest at the cap."""
    parts = []
    if rule.share > 0:
        parts.append(Cohort(build_year, capacity * rule.share, int(build_year + life), source, REASON_REPOWERED_AT_LIFE))
    if rule.share < 1:
        parts.append(Cohort(build_year, capacity * (1.0 - rule.share), int(build_year + rule.max_life_years), source,
                            REASON_LIFE_EXTENSION_END))
    return parts


@dataclass(frozen=True)
class StockFlowTechnology:
    """One stock-flow technology: its ES1 capacity lines, its life, its evidence and its model carriers.

    ``name`` is the vintage-cost technology (``technology_costs_vintage.csv``) where
    one exists, so the finance register can cost a cohort by (technology, build year).
    """

    name: str
    es1_lines: tuple[tuple[str, str], ...]
    carriers: tuple[str, ...]
    life_technology: str | None = None
    declared_life: float | None = None
    repd_types: tuple[str, ...] = ()
    dukes_fuels: tuple[str, ...] = ()
    note: str = ""

    @property
    def life_key(self) -> str:
        return self.life_technology or self.name


#: Declared.  ES1 (Type, SubType) capacity lines are summed over connection
#: (transmission, distributed, micro); the off-grid offshore block is ES1's own
#: ``excluded`` role and never enters.  Evidence sources: REPD for renewables and
#: storage, DUKES for thermal, nuclear and hydro (DUKES has 1,466 MW of dated hydro
#: against REPD's 670 MW).
STOCK_FLOW_TECHNOLOGIES: tuple[StockFlowTechnology, ...] = (
    StockFlowTechnology("Solar PV", (("Solar PV", "Solar PV"),), ("solar_pv", "embedded_solar"),
                        repd_types=("Solar Photovoltaics",)),
    StockFlowTechnology("Onshore wind", (("Onshore Wind", "Onshore Wind"),), ("wind_onshore", "embedded_wind"),
                        repd_types=("Wind Onshore",)),
    StockFlowTechnology("Offshore wind", (("Offshore Wind", "Offshore Wind"),), ("wind_offshore",),
                        repd_types=("Wind Offshore",),
                        note="ES1 does not split floating from fixed; every offshore cohort is fixed-bottom"),
    StockFlowTechnology("Hydropower", (("Hydro", "Hydro"),), ("large_hydro", "small_hydro", "hydro"),
                        dukes_fuels=("Hydro",)),
    StockFlowTechnology("Tidal stream", (("Other Renewable", "Marine"),),
                        ("marine", "tidal_stream", "shoreline_wave", "tidal_lagoon"),
                        repd_types=("Tidal Stream", "Shoreline Wave", "Tidal Lagoon")),
    StockFlowTechnology("Geothermal", (("Other Renewable", "Geothermal CHP"),), ("geothermal",),
                        declared_life=25.0, note="no Y2 row; life declared at 25 years, as DESNZ's other thermal plant"),
    StockFlowTechnology("Other renewables", (("Other Renewables", "Other Renewables"),), (),
                        declared_life=20.0, repd_types=("Anaerobic Digestion", "Landfill Gas", "Sewage Sludge Digestion"),
                        note="no Y2 row; life declared at 20 years (anaerobic digestion and landfill gas engines). "
                             "FES 2025 ES1 has no Other Renewables line, so no carrier takes it: the HT50 build's "
                             "biogas units pool with Energy from waste (task E2E, 1 Oct 2026)"),
    StockFlowTechnology("Large-scale nuclear", (("Nuclear", "Nuclear - Large"),), ("nuclear",),
                        dukes_fuels=("Nuclear",)),
    StockFlowTechnology("Small modular reactors (SMRs)", (("Nuclear", "Nuclear - Small"),), ("nuclear", "SMR")),
    StockFlowTechnology("Biomass - unabated", (("Biomass", "Biomass"), ("Biomass", "Biomass CHP")), ("biomass",),
                        repd_types=("Biomass (dedicated)", "Biomass (Dedicated)", "Biomass (co-firing)")),
    StockFlowTechnology("Biomass - with CCUS - mature", (("CCS Biomass", "CCS Biomass"),), ("CCS Biomass",)),
    StockFlowTechnology("Energy from waste", (("Waste", "Waste"), ("Waste", "Waste CHP")), ("waste_to_energy", "biogas"),
                        repd_types=("EfW Incineration", "Advanced Conversion Technologies"),
                        note="FES 2025 ES1's Waste line covers waste and biogas (the energy validation compares "
                             "waste plus biogas with it, task Y14): the HT50 biogas units take their vintages "
                             "from this line (declared, task E2E)"),
    StockFlowTechnology("Gas generation - CCUS - mature", (("CCS Gas", "CCS Gas"),), ("CCS Gas",)),
    StockFlowTechnology("Gas generation - unabated",
                        (("Gas", "CCGT"), ("Gas", "OCGT"), ("Gas", "Gas Reciprocating Engines")),
                        ("CCGT", "OCGT", "gas_engine", "gas"), dukes_fuels=("CCGT", "OCGT", "Natural Gas")),
    StockFlowTechnology("Gas CHP", (("Gas", "Gas CHP"), ("Gas", "Gas CCGT CHP")), ("CHP",),
                        life_technology="Gas generation - unabated",
                        note="no DUKES dates for CHP: uniform age profile; life of unabated gas"),
    StockFlowTechnology("Hydrogen CCHT", (("Hydrogen", "Hydrogen"),),
                        ("Hydrogen", "hydrogen_turbine", "H2_CCHT", "H2_turbine", "H2"),
                        note="ES1 transmission hydrogen turbines; the HT50 build's H2_turbine Links (task Y14-prep) "
                             "pool with OCHT, as the Hydrogen carrier"),
    StockFlowTechnology("Hydrogen OCHT", (("Hydrogen", "Hydrogen Peaking"), ("Hydrogen", "Hydrogen CHP")),
                        ("Hydrogen", "hydrogen_peaking", "H2_OCHT", "H2_turbine", "H2"),
                        note="ES1 hydrogen peaking and hydrogen CHP"),
    StockFlowTechnology("Oil", (("Other Thermal", "Diesel"), ("Other Thermal", "Fuel Oil")), ("oil",),
                        declared_life=25.0, dukes_fuels=("Diesel/Gas Oil",),
                        note="no Y2 row; life declared at 25 years, as unabated gas"),
    StockFlowTechnology("Coal", (("Coal", "Coal"),), ("coal",), declared_life=25.0, dukes_fuels=("Coal",),
                        note="no Y2 row; zero in FES 2025 HT from 2024"),
    StockFlowTechnology("Lithium batteries", (("Battery", "Battery"),), ("Battery", "battery"),
                        repd_types=("Battery",)),
    StockFlowTechnology("Liquid air energy storage", (("Long Duration Energy Storage", "Liquid Air"),), ("LAES",),
                        life_technology="Liquid air energy storage - 6h", repd_types=("Liquid Air Energy Storage",),
                        note="life 25 years (FES E.02 liquid air); the 2050 register costs LAES on the same row "
                             "(pypsa_finance_technology_crosswalk_2050.csv, task Y14-prep), so both agree"),
    StockFlowTechnology("Compressed air energy storage", (("Long Duration Energy Storage", "Compressed Air"),), ("CAES",),
                        life_technology="Compressed air energy storage - 6h",
                        repd_types=("Compressed Air Energy Storage",)),
    StockFlowTechnology("Pumped hydro energy storage (PHES)", (("Long Duration Energy Storage", "Pumped Hydro"),),
                        ("pumped_hydro", "Pumped Storage Hydroelectricity", "PHS"),
                        life_technology="Pumped hydro energy storage (PHES) - 6h",
                        repd_types=("Pumped Storage Hydroelectricity",)),
)
TECHNOLOGIES: dict[str, StockFlowTechnology] = {t.name: t for t in STOCK_FLOW_TECHNOLOGIES}
#: ES1 roles that hold installed capacity the stock-flow counts.
CAPACITY_ROLES = ("supply", "storage")


def _technology_of_line() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for tech in STOCK_FLOW_TECHNOLOGIES:
        for line in tech.es1_lines:
            if line in out:
                raise RuntimeError(f"ES1 line {line} is mapped twice")
            out[line] = tech.name
    return out


ES1_LINE_TECHNOLOGY = _technology_of_line()


# ── inputs ────────────────────────────────────────────────────────────────────

def parse_life_overrides(items: Iterable[str] | Mapping[str, float] | None) -> dict[str, float]:
    """``["Onshore wind=25"]`` (the command line) or ``{"Onshore wind": 25}`` (config) -> ``{name: life}``."""
    if not items:
        return {}
    if isinstance(items, Mapping):
        pairs = list(items.items())
    else:
        pairs = []
        for item in items:
            name, sep, value = str(item).rpartition("=")
            if not sep or not name.strip():
                raise ValueError(f"life override {item!r}: expected TECHNOLOGY=YEARS")
            pairs.append((name.strip(), value))
    return {str(name): float(value) for name, value in pairs}


def load_lives(path: str | Path | None = None, *, source: str = "central",
               overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """Economic life by stock-flow technology.

    Args:
        path: Y2's ``technology_economic_lives.csv``.
        source: ``central`` (its ``economic_life_years``, the 2030-parity lives) or
            ``fes_e02`` (FES 2025 E.02 asset lifetimes, falling back to central where blank).
        overrides: ``{stock-flow technology: life}`` applied over ``source`` for those
            technologies only (FES 2050 decision 48: onshore wind at 25 years, the rest
            central). Empty or None: the source's lives unchanged.

    Returns:
        ``{stock-flow technology: life in years}``.

    Raises:
        ValueError: A technology has neither a table row nor a declared life, or an
            override names an unknown technology or a non-positive life.
    """
    if source not in LIFE_SOURCES:
        raise ValueError(f"life source {source!r}; supported: {LIFE_SOURCES}")
    overrides = dict(overrides or {})
    if unknown := sorted(set(overrides) - set(TECHNOLOGIES)):
        raise ValueError(f"life overrides name unknown stock-flow technologies {unknown}")
    if bad := sorted(k for k, v in overrides.items() if not float(v) > 0):
        raise ValueError(f"life overrides must be positive: {bad}")
    table = pd.read_csv(path or DEFAULT_LIVES)
    central = dict(zip(table["technology"], pd.to_numeric(table["economic_life_years"], errors="coerce")))
    fes = dict(zip(table["technology"], pd.to_numeric(table.get("fes_e02_life"), errors="coerce")))
    out = {}
    for tech in STOCK_FLOW_TECHNOLOGIES:
        life = None
        if tech.declared_life is not None and tech.life_technology is None and tech.life_key not in central:
            life = float(tech.declared_life)
        else:
            value = fes.get(tech.life_key) if source == "fes_e02" else None
            if value is None or not np.isfinite(value):
                value = central.get(tech.life_key)
            if value is not None and np.isfinite(value):
                life = float(value)
            elif tech.declared_life is not None:
                life = float(tech.declared_life)
        if tech.name in overrides:
            life = float(overrides[tech.name])
        if life is None or not life > 0:
            raise ValueError(f"no economic life for stock-flow technology {tech.name!r} ({tech.life_key!r})")
        out[tech.name] = life
    return out


def fes_capacity_trajectory(es1: pd.DataFrame, *, pathway: str = DEFAULT_PATHWAY,
                            years: Iterable[int] = range(BASE_YEAR, TARGET_YEAR + 1)) -> pd.DataFrame:
    """FES installed capacity by stock-flow technology and year, from the ES1 table.

    Args:
        es1: ``scripts.FES.fes2025_targets.read_es1()``.
        pathway: The FES pathway.
        years: The years to read.

    Returns:
        ``technology, year, capacity_mw`` (every technology, zero where FES has none).

    Raises:
        ValueError: A positive ES1 capacity line maps to no stock-flow technology.
    """
    from scripts.FES.fes2025_targets import classify_es1

    rows = []
    for year in years:
        classified = classify_es1(es1, pathway, int(year))
        capacity = classified.loc[classified["quantity"].eq("capacity_mw") & classified["role"].isin(CAPACITY_ROLES)]
        keys = list(zip(capacity["type"].astype(str), capacity["subtype"].astype(str)))
        unmapped = sorted({k for k, v in zip(keys, capacity["value_std"]) if k not in ES1_LINE_TECHNOLOGY and v > 0})
        if unmapped:
            raise ValueError(f"ES1 {pathway} {year}: capacity lines with no stock-flow technology: {unmapped}")
        tech = pd.Series([ES1_LINE_TECHNOLOGY.get(k) for k in keys], index=capacity.index)
        totals = pd.to_numeric(capacity["value_std"], errors="coerce").fillna(0.0).groupby(tech).sum()
        for name in TECHNOLOGIES:
            rows.append({"technology": name, "year": int(year), "capacity_mw": float(totals.get(name, 0.0))})
    return pd.DataFrame(rows)


def load_stale_records(path: str | Path | None) -> list[int]:
    """REPD ``Ref ID``s to leave out of the evidence (decision 62: stale originals of repowered sites, duplicates)."""
    if not path:
        return []
    table = pd.read_csv(path)
    if "ref_id" not in table.columns:
        raise ValueError(f"{path} needs a ref_id column")
    return [int(v) for v in pd.to_numeric(table["ref_id"], errors="raise")]


def existing_fleet_evidence(repd: pd.DataFrame | None, dukes: pd.DataFrame | None, *,
                            base_year: int = BASE_YEAR, exclude_countries: Iterable[str] = (),
                            exclude_ref_ids: Iterable[int] = ()) -> pd.DataFrame:
    """Commissioning years of the existing fleet, from REPD and DUKES.

    REPD: rows with development status ``Operational`` and an ``Operational`` date
    in or before the base year.  DUKES 5.11 (processed): the commissioning year
    column (``status``).  Each technology takes one source (:data:`STOCK_FLOW_TECHNOLOGIES`).

    Args:
        repd: The REPD table (as published).
        dukes: ``resources/generators/DUKES/DUKES_<year>_generators.csv``.
        base_year: The last year counted.
        exclude_countries: REPD ``Country`` values left out (FES 2050 decision 62: Northern
            Ireland, which FES's GB capacity does not hold). Empty: every row, as before.
        exclude_ref_ids: REPD ``Ref ID``s left out (stale originals of repowered sites still
            listed as operational, and duplicate records; :func:`load_stale_records`).

    Returns:
        ``technology, commissioning_year, capacity_mw, source, name`` per site.
    """
    frames = []
    if repd is not None and len(repd):
        operational = repd.loc[repd["Development Status (short)"].astype(str).str.strip().eq("Operational")]
        countries = {str(c).strip() for c in exclude_countries}
        if countries and "Country" in operational.columns:
            operational = operational.loc[~operational["Country"].astype(str).str.strip().isin(countries)]
        refs = {int(r) for r in exclude_ref_ids}
        if refs and "Ref ID" in operational.columns:
            operational = operational.loc[~pd.to_numeric(operational["Ref ID"], errors="coerce").isin(refs)]
        year = pd.to_datetime(operational["Operational"], dayfirst=True, errors="coerce").dt.year
        capacity = pd.to_numeric(operational["Installed Capacity (MWelec)"], errors="coerce")
        kinds = operational["Technology Type"].astype(str).str.strip()
        for tech in STOCK_FLOW_TECHNOLOGIES:
            if not tech.repd_types:
                continue
            mask = kinds.isin(tech.repd_types) & year.notna() & (year <= base_year) & (capacity > 0)
            frames.append(pd.DataFrame({
                "technology": tech.name, "commissioning_year": year[mask].astype(int).to_numpy(),
                "capacity_mw": capacity[mask].to_numpy(dtype=float), "source": "REPD",
                "name": operational.loc[mask, "Site Name"].astype(str).to_numpy()}))
    if dukes is not None and len(dukes):
        year = pd.to_numeric(dukes["status"], errors="coerce")
        capacity = pd.to_numeric(dukes["capacity_mw"], errors="coerce")
        fuel = dukes["fuel_type"].astype(str).str.strip()
        for tech in STOCK_FLOW_TECHNOLOGIES:
            if not tech.dukes_fuels:
                continue
            mask = fuel.isin(tech.dukes_fuels) & year.notna() & (year <= base_year) & (capacity > 0)
            frames.append(pd.DataFrame({
                "technology": tech.name, "commissioning_year": year[mask].astype(int).to_numpy(),
                "capacity_mw": capacity[mask].to_numpy(dtype=float), "source": "DUKES",
                "name": dukes.loc[mask, "station_name"].astype(str).str.strip().to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["technology", "commissioning_year", "capacity_mw", "source", "name"])
    return pd.concat(frames, ignore_index=True)


# ── the stock-flow ───────────────────────────────────────────────────────────

@dataclass
class Cohort:
    """Capacity of one build year of one technology, and when it is scheduled to retire (None: not in horizon)."""

    build_year: int
    capacity_mw: float
    retirement_year: int | None
    source: str
    schedule_reason: str = ""


def load_closure_dates(path: str | Path | None = None) -> dict[str, int]:
    """Published closure years of existing nuclear stations (``station name -> year``)."""
    table = pd.read_csv(path or DEFAULT_NUCLEAR_CLOSURES)
    dates = pd.to_datetime(table["date"], dayfirst=True, errors="coerce")
    return {str(n).strip(): int(d.year) for n, d in zip(table["name"], dates) if pd.notna(d)}


def base_stock(fes_base_mw: Mapping[str, float], evidence: pd.DataFrame, lives: Mapping[str, float], *,
               base_year: int = BASE_YEAR, rule: str = RULE_CENTRAL, run_down: Iterable[str] = (),
               closure_years: Mapping[str, int] | None = None,
               repowering: Mapping[str, RepoweringRule] | None = None,
               refurbishment: RefurbishmentRule | None = None) -> dict[str, list[Cohort]]:
    """The base-year fleet by technology and build year, scaled to FES base-year capacity.

    Args:
        fes_base_mw: FES capacity in the base year by technology.
        evidence: :func:`existing_fleet_evidence`.
        lives: Economic life by technology.
        base_year: The base year (2024).
        rule: A key of :data:`RETIREMENT_RULES`.
        run_down: Technologies FES runs down (target below base); life-extended under the central rule.
        closure_years: Published closure years by station name (existing nuclear, ``evidence.name``).
        repowering: FES 2050 decision 62: :class:`RepoweringRule` by technology, central rule only
            (the strict bound ignores it: every cohort leaves at its life). A cohort younger than
            the decision age splits (the share leaves at its life, the rest at the cap); one at or
            past the decision age in the base year runs to the cap; one past the cap leaves in the
            first year. None or empty: unchanged.
        refurbishment: F28 (d): under the strict rule, a refurbished technology's cohorts past
            their life in the base year leave on :func:`refurbishment_years`' schedule, not all
            in the first horizon year. None: unchanged.

    Returns:
        ``{technology: [Cohort]}``, oldest first; each technology's cohorts sum to its
        FES base-year capacity.
    """
    if rule not in RETIREMENT_RULES:
        raise ValueError(f"retirement rule {rule!r}; supported: {sorted(RETIREMENT_RULES)}")
    run_down = set(run_down)
    closure_years = {str(k).strip(): int(v) for k, v in dict(closure_years or {}).items()}
    out: dict[str, list[Cohort]] = {}
    for name in TECHNOLOGIES:
        level = float(fes_base_mw.get(name, 0.0))
        if level <= 0:
            out[name] = []
            continue
        life = float(lives[name])
        rows = evidence.loc[evidence["technology"].eq(name)] if len(evidence) else evidence
        cohorts: dict[tuple[int, int | None], dict] = {}
        if len(rows) and rows["capacity_mw"].sum() > 0:
            scale = level / float(rows["capacity_mw"].sum())
            names = rows["name"].astype(str).str.strip() if "name" in rows.columns else pd.Series("", index=rows.index)
            for row, station in zip(rows.itertuples(index=False), names):
                closure = closure_years.get(station)
                # A date after the base year holds the station through its closure year; a stale one is ignored.
                closure = closure + 1 if closure is not None and closure > base_year else None
                cell = cohorts.setdefault((int(row.commissioning_year), closure),
                                          {"capacity": 0.0, "source": f"base_stock_{str(row.source).lower()}"})
                cell["capacity"] += float(row.capacity_mw) * scale
        else:
            # Declared fallback: every age within the economic life equally represented.
            span = max(1, int(round(life)))
            first = base_year - span + 1
            for year in range(first, base_year + 1):
                cohorts[(year, None)] = {"capacity": level / span, "source": "base_stock_uniform_fallback"}
        stock = []
        repower = (repowering or {}).get(name) if rule == RULE_CENTRAL else None
        # F28 (d): past-life cohorts of a refurbished technology leave on the schedule (per technology where a window is given).
        schedule = (refurbishment_years([(y, cell["capacity"]) for (y, closure), cell in cohorts.items()
                                         if closure is None and int(y + life) <= base_year], refurbishment, name)
                    if rule == RULE_STRICT and refurbishment is not None and name in refurbishment.technologies
                    else {})
        for (year, closure), cell in sorted(cohorts.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
            end = int(year + life)
            if repower is not None and closure is None:
                cap = int(year + repower.max_life_years)
                if cap <= base_year:
                    stock.append(Cohort(int(year), float(cell["capacity"]), base_year + 1, cell["source"],
                                        REASON_PAST_MAX_LIFE_AT_BASE))
                elif end <= base_year:
                    stock.append(Cohort(int(year), float(cell["capacity"]), cap, cell["source"],
                                        REASON_LIFE_EXTENSION_END))
                else:
                    stock.extend(_split_cohort(int(year), float(cell["capacity"]), life, repower, cell["source"]))
                continue
            if closure is not None:
                retirement, reason = closure, REASON_PUBLISHED_CLOSURE
            elif rule == RULE_CENTRAL and (end <= base_year or name in run_down):
                retirement, reason = None, REASON_PAST_LIFE_AT_BASE
            elif end <= base_year:
                retirement, reason = schedule.get(int(year), base_year + 1), REASON_PAST_LIFE_AT_BASE
            else:
                retirement, reason = end, REASON_END_OF_LIFE
            stock.append(Cohort(int(year), float(cell["capacity"]), retirement, cell["source"], reason))
        out[name] = stock
    return out


def stock_flow(trajectory: pd.DataFrame, stock: Mapping[str, list[Cohort]], lives: Mapping[str, float], *,
               base_year: int = BASE_YEAR, target_year: int = TARGET_YEAR,
               tolerance_mw: float = 1e-6, repowering: Mapping[str, RepoweringRule] | None = None,
               refurbishment: RefurbishmentRule | None = None,
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the stock-flow from the base year to the target year.

    Args:
        trajectory: ``technology, year, capacity_mw`` (:func:`fes_capacity_trajectory`).
        stock: The base stock (:func:`base_stock`); copied, not changed.
        lives: Economic life by technology.
        base_year: The base year.
        target_year: The snapshot year.
        tolerance_mw: Below this a flow is zero.
        repowering: FES 2050 decision 62: each gross addition of a technology with a rule is
            split as :func:`base_stock` splits a young cohort, so no cohort outlives the cap.
            The flows gain ``repowered_at_decision_age_mw`` and ``life_extension_end_mw`` (the
            year's exits by kind) and the mix ``repowering_share`` (the share of the build
            year's capacity that leaves at the decision age). None or empty: unchanged.
        refurbishment: F28 (d): the mix gains ``refurbishment_share``, the share of a refurbished
            technology's build year that replaced its retired plant (refurbished, not new). None:
            unchanged.

    Returns:
        ``(vintage_mix, flows, retirements)``: the target-year stock by technology and
        build year; the flows by technology and year; and every retirement event
        (technology, year, build year, capacity, reason).

    Raises:
        ValueError: A trajectory year is missing, or FES capacity is negative.
    """
    wide = trajectory.pivot_table(index="technology", columns="year", values="capacity_mw", aggfunc="sum")
    missing = [y for y in range(base_year, target_year + 1) if y not in wide.columns]
    if missing:
        raise ValueError(f"the trajectory lacks years {missing}")
    if (wide.fillna(0.0) < -tolerance_mw).any().any():
        raise ValueError("FES capacity is negative somewhere in the trajectory")
    mix_rows, flow_rows, events = [], [], []
    for name in TECHNOLOGIES:
        life = float(lives[name])
        cohorts = [Cohort(c.build_year, c.capacity_mw, c.retirement_year, c.source, c.schedule_reason)
                   for c in stock.get(name, [])]
        fes = wide.loc[name] if name in wide.index else pd.Series(0.0, index=wide.columns)
        for year in range(base_year + 1, target_year + 1):
            start = sum(c.capacity_mw for c in cohorts)
            scheduled = 0.0
            kept = []
            for c in cohorts:
                if c.retirement_year is not None and c.retirement_year <= year:
                    scheduled += c.capacity_mw
                    events.append({"technology": name, "year": year, "build_year": c.build_year,
                                   "capacity_mw": c.capacity_mw, "reason": c.schedule_reason or REASON_END_OF_LIFE})
                else:
                    kept.append(c)
            cohorts = kept
            target = max(0.0, float(fes.get(year, 0.0)))
            surviving = sum(c.capacity_mw for c in cohorts)
            early = 0.0
            if surviving > target + tolerance_mw:
                excess = surviving - target
                cohorts.sort(key=lambda c: c.build_year)
                for c in cohorts:
                    if excess <= tolerance_mw:
                        break
                    cut = min(c.capacity_mw, excess)
                    c.capacity_mw -= cut
                    excess -= cut
                    early += cut
                    events.append({"technology": name, "year": year, "build_year": c.build_year,
                                   "capacity_mw": cut, "reason": REASON_FES_DECLINE})
                cohorts = [c for c in cohorts if c.capacity_mw > tolerance_mw]
            surviving = sum(c.capacity_mw for c in cohorts)
            addition = max(0.0, target - surviving)
            repower = (repowering or {}).get(name)
            if addition > tolerance_mw:
                if repower is not None:
                    cohorts.extend(_split_cohort(year, addition, life, repower, "gross_addition"))
                else:
                    cohorts.append(Cohort(year, addition, int(year + life), "gross_addition", REASON_END_OF_LIFE))
            else:
                addition = 0.0
            # End-of-life (and past-life) retirements the year's additions replace are repowering.
            year_events = [e for e in events if e["technology"] == name and e["year"] == year]
            end_of_life = sum(e["capacity_mw"] for e in year_events if e["reason"] in REPLACED_REASONS)
            row = {
                "technology": name, "year": year, "stock_start_mw": start, "scheduled_retirement_mw": scheduled,
                "early_retirement_mw": early, "gross_addition_mw": addition,
                "net_fes_change_mw": target - float(fes.get(year - 1, 0.0) if year - 1 in fes.index else 0.0),
                "repowered_mw": min(end_of_life, addition),
                "stock_end_mw": sum(c.capacity_mw for c in cohorts), "fes_capacity_mw": target}
            if repowering:
                for reason in (REASON_REPOWERED_AT_LIFE, REASON_LIFE_EXTENSION_END):
                    row[f"{reason}_mw"] = sum(e["capacity_mw"] for e in year_events if e["reason"] == reason)
            flow_rows.append(row)
        for c in sorted(cohorts, key=lambda c: c.build_year):
            entry = {"technology": name, "build_year": c.build_year, "capacity_mw": c.capacity_mw,
                     "source": c.source, "economic_life_years": life,
                     "scheduled_retirement_year": c.retirement_year,
                     "finance_vintage": "legacy" if c.build_year < POLICY_START_YEAR else "new_build"}
            if repowering:
                entry["repowered_capacity_mw"] = c.capacity_mw if c.schedule_reason == REASON_REPOWERED_AT_LIFE else 0.0
            mix_rows.append(entry)
    columns = ["technology", "build_year", "capacity_mw", "source", "economic_life_years",
               "scheduled_retirement_year", "finance_vintage"]
    extra = {"repowered_capacity_mw": ("repowered_capacity_mw", "sum")} if repowering else {}
    mix = pd.DataFrame(mix_rows, columns=columns + list(extra))
    # Cohorts of one build year from different sources (a base-stock year repeated by a
    # published closure, or the two parts of a repowering split) are one cell of the mix.
    mix = (mix.groupby(["technology", "build_year"], sort=False, as_index=False)
              .agg(capacity_mw=("capacity_mw", "sum"), source=("source", "first"),
                   economic_life_years=("economic_life_years", "first"),
                   scheduled_retirement_year=("scheduled_retirement_year", "first"),
                   finance_vintage=("finance_vintage", "first"), **extra))
    if repowering:
        # Decision 62: the share of the cell that leaves at the decision age (the rest at the cap).
        mix["repowering_share"] = (mix["repowered_capacity_mw"] / mix["capacity_mw"].where(mix["capacity_mw"] > 0)).fillna(0.0)
        mix = mix.drop(columns=["repowered_capacity_mw"])
        # ... and the share of the build year's gross addition that replaced retired plant
        # (its repowering, which a technology without a CfD route for repowering, solar, does not
        # get a contract on: ``label_support_contracts(no_cfd_on_repowering=)``).
        added = {(r["technology"], r["year"]): (r["repowered_mw"] / r["gross_addition_mw"]
                                                if r["gross_addition_mw"] > tolerance_mw else 0.0) for r in flow_rows}
        mix["replacement_share"] = [float(added.get((t, int(y)), 0.0)) for t, y in zip(mix["technology"], mix["build_year"])]
    if refurbishment is not None and "replacement_share" not in mix.columns:
        # F28 (d): under the strict rule every end-of-life exit is replaced, so the replaced share of each
        # build year is written here too (solar's repowering holds no CfD: no_cfd_on_repowering).
        replaced = {(r["technology"], r["year"]): (r["repowered_mw"] / r["gross_addition_mw"]
                                                   if r["gross_addition_mw"] > tolerance_mw else 0.0) for r in flow_rows}
        mix["replacement_share"] = [float(replaced.get((t, int(y)), 0.0))
                                    for t, y in zip(mix["technology"], mix["build_year"])]
    if refurbishment is not None:
        refurbished = {(r["technology"], r["year"]): (r["repowered_mw"] / r["gross_addition_mw"]
                                                      if r["gross_addition_mw"] > tolerance_mw else 0.0)
                       for r in flow_rows if r["technology"] in refurbishment.technologies}
        mix["refurbishment_share"] = [float(refurbished.get((t, int(y)), 0.0))
                                      for t, y in zip(mix["technology"], mix["build_year"])]
    return mix, pd.DataFrame(flow_rows), pd.DataFrame(events, columns=["technology", "year", "build_year",
                                                                        "capacity_mw", "reason"])


def decade_label(year: int, base_year: int = BASE_YEAR) -> str:
    """``2025-2030``, ``2031-2040``, ``2041-2050``: the horizon's first stretch, then decades."""
    year = int(year)
    if year <= 2030:
        return f"{base_year + 1}-2030"
    start = ((year - 1) // 10) * 10 + 1
    return f"{start}-{start + 9}"


def retirements_by_decade(events: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    """Capacity retired by technology, decade and reason (MW), with a total row per decade."""
    if events.empty:
        return pd.DataFrame(columns=["technology", "decade", "reason", "capacity_mw"])
    frame = events.assign(decade=events["year"].map(lambda y: decade_label(y, base_year)))
    table = frame.groupby(["technology", "decade", "reason"], as_index=False)["capacity_mw"].sum()
    total = frame.groupby(["decade", "reason"], as_index=False)["capacity_mw"].sum().assign(technology="TOTAL")
    return pd.concat([table, total[table.columns]], ignore_index=True)


def vintage_summary(mix: pd.DataFrame, target_year: int = TARGET_YEAR) -> pd.DataFrame:
    """The target-year mix by technology: capacity pre-2025, 2025-2030, then by decade to the target year, and in
    total (MW). For 2050: pre_2025, 2025_2030, 2031_2040, 2041_2050; for 2040 the last band is 2031_2040."""
    edges, labels = [-np.inf, POLICY_START_YEAR - 1, 2030], [f"pre_{POLICY_START_YEAR}", f"{POLICY_START_YEAR}_2030"]
    while edges[-1] < target_year:
        top = min(edges[-1] + 10, target_year)
        labels.append(f"{int(edges[-1]) + 1}_{int(top)}")
        edges.append(top)
    edges[-1] = np.inf
    bands = pd.cut(mix["build_year"], bins=edges, labels=labels)
    table = mix.assign(band=bands).pivot_table(index="technology", columns="band", values="capacity_mw",
                                               aggfunc="sum", fill_value=0.0, observed=False)
    table["total_mw"] = table.sum(axis=1)
    return table.reset_index()


def build_fleet_vintages(trajectory: pd.DataFrame, evidence: pd.DataFrame, lives: Mapping[str, float], *,
                         base_year: int = BASE_YEAR, target_year: int = TARGET_YEAR, rule: str = RULE_CENTRAL,
                         closure_years: Mapping[str, int] | None = None,
                         repowering: Mapping[str, RepoweringRule] | None = None,
                         refurbishment: RefurbishmentRule | None = None) -> dict:
    """The whole Y10 build: base stock, flows, the target-year mix and the reports.

    ``repowering`` (FES 2050 decision 62) applies under the central rule only; ``refurbishment``
    (F28 (d)) under the strict rule only.

    Returns:
        ``{"mix", "flows", "retirements", "retired_by_decade", "summary"}``.
    """
    repowering = dict(repowering or {}) if rule == RULE_CENTRAL else {}
    refurbishment = refurbishment if rule == RULE_STRICT else None
    base = trajectory.loc[trajectory["year"].eq(base_year)].set_index("technology")["capacity_mw"].to_dict()
    final = trajectory.loc[trajectory["year"].eq(target_year)].set_index("technology")["capacity_mw"].to_dict()
    run_down = sorted(n for n in TECHNOLOGIES if float(final.get(n, 0.0)) < float(base.get(n, 0.0)))
    stock = base_stock(base, evidence, lives, base_year=base_year, rule=rule, run_down=run_down,
                       closure_years=closure_years, repowering=repowering or None, refurbishment=refurbishment)
    mix, flows, events = stock_flow(trajectory, stock, lives, base_year=base_year, target_year=target_year,
                                    repowering=repowering or None, refurbishment=refurbishment)
    fes_target = trajectory.loc[trajectory["year"].eq(target_year)].set_index("technology")["capacity_mw"]
    by_tech = mix.groupby("technology")["capacity_mw"].sum().reindex(fes_target.index, fill_value=0.0)
    gap = float((by_tech - fes_target).abs().max()) if len(fes_target) else 0.0
    evidence_sources = (evidence.groupby("technology")["source"].first().to_dict() if len(evidence) else {})
    summary = {
        "base_year": base_year, "target_year": target_year, "retirement_rule": rule,
        "retirement_rule_text": RETIREMENT_RULES[rule],
        "conservation_max_abs_gap_mw": gap,
        "run_down_technologies": run_down if rule == RULE_CENTRAL else [],
        "fes_target_capacity_mw": float(fes_target.sum()),
        "mix_capacity_mw": float(mix["capacity_mw"].sum()),
        "post_policy_capacity_mw": float(mix.loc[mix["build_year"] >= POLICY_START_YEAR, "capacity_mw"].sum()),
        "gross_additions_mw": float(flows["gross_addition_mw"].sum()),
        "repowered_mw": float(flows["repowered_mw"].sum()),
        "lives_years": {k: float(v) for k, v in lives.items()},
        "evidence_source_by_technology": {name: evidence_sources.get(name, "uniform_fallback")
                                          for name in TECHNOLOGIES if float(base.get(name, 0.0)) > 0},
        "declarations": [
            "the FES level and the evidence shape: every base-year cohort is evidence capacity x FES 2024 / evidence total",
            "a technology without dated evidence takes a uniform age profile over its economic life",
            "a cohort built in b with life L is in the stock for b..b+L-1 and retires in b+L",
            "repowered sites are new build of the repowering year",
            f"build years before {POLICY_START_YEAR} are the legacy fleet in the finance register",
        ],
    }
    if refurbishment is not None:
        summary["refurbishment"] = {"technologies": list(refurbishment.technologies),
                                    "first_year": refurbishment.first_year, "last_year": refurbishment.last_year,
                                    "windows": {n: [f, l] for n, f, l in refurbishment.windows}}
        summary["declarations"].append(
            "F28 (d): hydro and pumped hydro are refurbished, not rebuilt, at the end of their life (the mix's "
            "refurbishment_share); cohorts past their life in the base year are refurbished oldest first, evenly by "
            f"capacity over {refurbishment.first_year}-{refurbishment.last_year}"
            + "".join(f" ({n}: {f}-{l})" for n, f, l in refurbishment.windows)
            + " (sourced 27 Oct 2026: SSE's station-by-station hydro works, ENGIE's and Drax's pumped-storage programmes)")
    if repowering:
        summary["repowering"] = {name: {"share_at_decision_age": r.share, "decision_age_years": float(lives[name]),
                                        "max_life_years": r.max_life_years} for name, r in repowering.items()}
        summary["declarations"].append(
            "decision 62: at the decision age (the technology's life) a share of each cohort repowers as new build of "
            "that year; the rest is life-extended at operating cost and leaves at the cap; no cohort outlives the cap")
    return {"mix": mix, "flows": flows, "retirements": events,
            "retired_by_decade": retirements_by_decade(events, base_year), "summary": summary,
            "vintage_summary": vintage_summary(mix, target_year)}


# ── the join onto a register ─────────────────────────────────────────────────

JOIN_PRO_RATA, JOIN_WHOLE_UNITS = "pro_rata", "whole_units"
JOIN_MODES = (JOIN_PRO_RATA, JOIN_WHOLE_UNITS)
METHOD_DATED = "asset_date"
METHOD_REPOWERED = "asset_date_repowered"
METHOD_BUS = "bus_located_survivor"
METHOD_RESIDUAL = "stock_flow_pro_rata"
METHOD_WHOLE = "stock_flow_whole_unit"
METHOD_NONE = "no_stock_flow_technology"
COHORT_SEPARATOR = "::"


def _technology_slug(technology: str) -> str:
    """``Small modular reactors (SMRs)`` -> ``small_modular_reactors_smrs``: a technology in a cohort id."""
    return "_".join("".join(ch if ch.isalnum() else " " for ch in str(technology).lower()).split())


def carrier_technologies() -> dict[str, tuple[str, ...]]:
    """``carrier -> stock-flow technologies`` (a carrier naming two, such as ``nuclear``, pools them)."""
    out: dict[str, list[str]] = {}
    for tech in STOCK_FLOW_TECHNOLOGIES:
        for carrier in tech.carriers:
            out.setdefault(carrier, []).append(tech.name)
    return {k: tuple(v) for k, v in out.items()}


def _join_group(row: Mapping, by_carrier: Mapping[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """The technologies a register row may take its vintage from.

    A ``stock_flow_technology`` column wins, then a ``finance_technology`` naming a
    stock-flow technology, then the carrier (pooled where a carrier maps to two).
    """
    for column in ("stock_flow_technology", "finance_technology"):
        value = row.get(column)
        if isinstance(value, str) and value in TECHNOLOGIES:
            return (value,)
    group = by_carrier.get(str(row.get("carrier")))
    return group or None


def _order_key(asset_id: str) -> str:
    """A deterministic order within a cell that carries no location or name bias (md5 of the id)."""
    return hashlib.md5(str(asset_id).encode("utf-8")).hexdigest()


def repowering_year(commissioning_year: float, life: float, model_year: int) -> int:
    """The build year of the cohort standing on a dated site in the model year.

    The original build if it still stands (``b + L > model year``), else the latest
    repowering ``b + kL`` at or before the model year.
    """
    b = int(commissioning_year)
    if b + life > model_year:
        return b
    k = math.floor((model_year - b) / life)
    return int(b + k * life)


def standing_cohorts(commissioning_year: float, life: float, rule: RepoweringRule, model_year: int, *,
                     base_year: int = BASE_YEAR) -> list[tuple[int, float]]:
    """FES 2050 decision 62: the build years standing on a dated site in the model year, with their shares.

    The repowering rule's tree (:class:`RepoweringRule`): a cohort younger than the decision
    age at the base year splits at its life (the share repowers as new build then) and the
    rest runs to the cap, then repowers; a cohort past the decision age at the base year runs
    to the cap (or leaves in the first horizon year when already past it). Each new build
    splits the same way. Shares sum to one.
    """
    out: dict[int, float] = {}

    def walk(year: int, share: float, existing: bool) -> None:
        if share <= 0:
            return
        decision, cap = int(year + life), int(year + rule.max_life_years)
        if existing and decision <= base_year:
            end = max(cap, base_year + 1)
            if model_year < end:
                out[year] = out.get(year, 0.0) + share
            else:
                walk(end, share, False)
            return
        if model_year < decision:
            out[year] = out.get(year, 0.0) + share
            return
        walk(decision, share * rule.share, False)
        if model_year < cap:
            out[year] = out.get(year, 0.0) + share * (1.0 - rule.share)
        else:
            walk(cap, share * (1.0 - rule.share), False)

    walk(int(commissioning_year), 1.0, int(commissioning_year) <= base_year)
    return sorted(out.items())


def assign_build_years(register: pd.DataFrame, mix: pd.DataFrame, *, model_year: int,
                       lives: Mapping[str, float], site_vintages: pd.DataFrame | None = None,
                       mode: str = JOIN_PRO_RATA, policy_start_year: int = POLICY_START_YEAR,
                       tolerance: float = 1e-9,
                       repowering: Mapping[str, RepoweringRule] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give every register row a build year from the vintage mix (task Y10, the join).

    The join, declared, per technology (or pooled carrier group):

    1. **Dated rows** (``commissioning_year`` given, from the network's operational
       date or build year): the cohort standing on the site in the model year, the
       original build if it survives, else its repowering year (:func:`repowering_year`),
       up to what that cell of the mix still holds.
    2. **Bus-located survivors** (``site_vintages``: technology, bus, build_year,
       capacity_mw, the dated existing fleet placed on the network's buses): the
       undated rows at that bus take those cells first, pro rata to capacity.
    3. **The rest**, from the cells left: ``pro_rata`` splits every row's remaining
       capacity over them in proportion (exact); ``whole_units`` gives each row one
       build year, rows in md5 order filling the cells in build-year order by the
       capacity midpoint (whole units, error at most half a unit per cell).

    Where the register's capacity of a technology differs from the mix's (the
    network is not FES exactly), the mix gives the shares and the register keeps its
    capacity (``mix_scale`` in the report).

    Args:
        register: Rows with ``component_type, asset_id, carrier, capacity_mw`` and
            optionally ``bus, commissioning_year, energy_capacity_mwh, dispatch_mwh,
            stock_flow_technology, finance_technology``.
        mix: The target-year vintage mix (:func:`stock_flow`).
        model_year: The register's model year.
        lives: Economic life by technology.
        site_vintages: Optional bus-located survivors.
        mode: ``pro_rata`` or ``whole_units``.
        policy_start_year: Build years before it are ``legacy``.
        tolerance: Capacity below this is zero.
        repowering: FES 2050 decision 62's rules; a dated row of a technology with one takes
            each standing build year's share (:func:`standing_cohorts`) instead of one
            repowering year. None: unchanged.

    Returns:
        ``(cohorts, report)``: the register as cohort rows (``asset_id`` gains
        ``::b<year>`` when a row is split), each with ``build_year``,
        ``commissioning_year`` (= build year), ``site_commissioning_year``,
        ``finance_vintage``, ``vintage_method``, ``stock_flow_technology``,
        ``cohort_share`` and ``cohort_legacy_share``; and a per-technology report
        (register MW, mix MW, scale, assigned MW by method).
    """
    if mode not in JOIN_MODES:
        raise ValueError(f"join mode {mode!r}; supported: {JOIN_MODES}")
    register = register.reset_index(drop=True).copy()
    by_carrier = carrier_technologies()
    groups = [_join_group(row, by_carrier) for row in register.to_dict("records")]
    capacity = pd.to_numeric(register["capacity_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dated_year = (pd.to_numeric(register["commissioning_year"], errors="coerce").to_numpy(dtype=float)
                  if "commissioning_year" in register.columns else np.full(len(register), np.nan))
    buses = register["bus"].astype(str).to_numpy() if "bus" in register.columns else np.full(len(register), "")
    # pieces[i] = [(technology, build_year, capacity, method)]
    pieces: list[list[tuple]] = [[] for _ in range(len(register))]
    report = []
    group_keys = sorted({g for g in groups if g is not None})
    for group in group_keys:
        rows = [i for i, g in enumerate(groups) if g == group]
        cells_frame = mix.loc[mix["technology"].isin(group)]
        cells: dict[tuple[str, int], float] = {
            (str(t), int(b)): float(c) for t, b, c in
            zip(cells_frame["technology"], cells_frame["build_year"], cells_frame["capacity_mw"]) if c > tolerance}
        register_mw = float(capacity[rows].sum())
        mix_mw = float(sum(cells.values()))
        scale = register_mw / mix_mw if mix_mw > tolerance else np.nan
        if mix_mw > tolerance:
            cells = {k: v * scale for k, v in cells.items()}
        remaining = capacity.copy()
        by_method: dict[str, float] = {}

        def take(i: int, key: tuple[str, int], amount: float, method: str) -> None:
            amount = min(amount, remaining[i], cells.get(key, 0.0))
            if amount <= tolerance:
                return
            pieces[i].append((key[0], key[1], amount, method))
            remaining[i] -= amount
            cells[key] -= amount
            by_method[method] = by_method.get(method, 0.0) + amount

        if not cells:
            report.append({"technologies": "|".join(group), "register_mw": register_mw, "mix_mw": mix_mw,
                           "mix_scale": scale, "unassigned_mw": register_mw})
            continue
        # 1. dated rows
        for i in rows:
            if not np.isfinite(dated_year[i]):
                continue
            options = []
            for tech in group:
                rule = (repowering or {}).get(tech)
                if rule is not None:
                    # Decision 62: the site's standing build years and their shares of the row.
                    whole = remaining[i]
                    for year, part in standing_cohorts(dated_year[i], float(lives[tech]), rule, model_year):
                        options.append((tech, year, whole * part))
                    continue
                year = repowering_year(dated_year[i], float(lives[tech]), model_year)
                options.append((tech, year, None))
            for tech, year, amount in options:
                method = METHOD_DATED if year == int(dated_year[i]) else METHOD_REPOWERED
                take(i, (tech, year), remaining[i] if amount is None else amount, method)
        # 2. bus-located survivors
        if site_vintages is not None and len(site_vintages):
            located = site_vintages.loc[site_vintages["technology"].isin(group)]
            for (tech, bus), block in located.groupby(["technology", "bus"], sort=True):
                at_bus = [i for i in rows if buses[i] == str(bus) and not np.isfinite(dated_year[i])]
                if not at_bus:
                    continue
                for year, amount in block.groupby("build_year")["capacity_mw"].sum().sort_index().items():
                    key = (str(tech), int(year))
                    want = min(float(amount) * (scale if np.isfinite(scale) else 1.0), cells.get(key, 0.0))
                    room = float(sum(remaining[i] for i in at_bus))
                    if want <= tolerance or room <= tolerance:
                        continue
                    for i in at_bus:
                        take(i, key, want * remaining[i] / room if room > 0 else 0.0, METHOD_BUS)
        # 3. the rest
        open_rows = [i for i in rows if remaining[i] > tolerance]
        if mode == JOIN_PRO_RATA:
            pool = float(sum(v for v in cells.values() if v > tolerance))
            need = float(sum(remaining[i] for i in open_rows))
            if pool > tolerance and need > tolerance:
                fraction = {k: v / pool for k, v in cells.items() if v > tolerance}
                for i in open_rows:
                    left = remaining[i]
                    for key, share in fraction.items():
                        amount = left * share
                        pieces[i].append((key[0], key[1], amount, METHOD_RESIDUAL))
                        cells[key] -= amount
                        by_method[METHOD_RESIDUAL] = by_method.get(METHOD_RESIDUAL, 0.0) + amount
                    remaining[i] = 0.0
        else:
            order = sorted(open_rows, key=lambda i: _order_key(register.at[i, "asset_id"]))
            keys = sorted((k for k, v in cells.items() if v > tolerance), key=lambda k: (k[1], k[0]))
            bounds = np.cumsum([cells[k] for k in keys])
            position = 0.0
            for i in order:
                midpoint = position + remaining[i] / 2.0
                slot = int(np.searchsorted(bounds, midpoint, side="left")) if len(keys) else 0
                key = keys[min(slot, len(keys) - 1)] if keys else None
                position += remaining[i]
                if key is None:
                    continue
                pieces[i].append((key[0], key[1], remaining[i], METHOD_WHOLE))
                cells[key] -= remaining[i]
                by_method[METHOD_WHOLE] = by_method.get(METHOD_WHOLE, 0.0) + remaining[i]
                remaining[i] = 0.0
        report.append({"technologies": "|".join(group), "register_mw": register_mw, "mix_mw": mix_mw,
                       "mix_scale": scale, **{f"assigned_{k}_mw": v for k, v in by_method.items()},
                       "unassigned_mw": float(sum(remaining[i] for i in rows))})
    out_rows = []
    scaled_columns = [c for c in ("capacity_mw", "energy_capacity_mwh", "dispatch_mwh") if c in register.columns]
    for i, record in enumerate(register.to_dict("records")):
        parts = [p for p in pieces[i] if p[2] > tolerance]
        total = capacity[i]
        if not parts:
            row = dict(record)
            row.update({"build_year": np.nan, "site_commissioning_year": record.get("commissioning_year"),
                        "vintage_method": METHOD_NONE if groups[i] is None else "unassigned",
                        "stock_flow_technology": None, "cohort_share": 1.0,
                        "cohort_legacy_share": np.nan})
            out_rows.append(row)
            continue
        # One piece per (technology, build year).
        merged: dict[tuple[str, int], list] = {}
        for tech, year, amount, method in parts:
            cell = merged.setdefault((tech, year), [0.0, method])
            cell[0] += amount
        legacy_share = sum(v[0] for (t, y), v in merged.items() if y < policy_start_year) / total if total > 0 else 0.0
        # A pooled carrier (nuclear: large and SMR; H2_turbine: CCHT and OCHT) can give one
        # row two technologies of the same build year: the id then names the technology
        # too, so every cohort id is unique (task E2E, 1 Oct 2026).
        pooled = len({tech for tech, _ in merged}) > 1
        for (tech, year), (amount, method) in sorted(merged.items(), key=lambda item: (item[0][1], item[0][0])):
            share = amount / total if total > 0 else 1.0
            row = dict(record)
            if len(merged) > 1:
                row["asset_id"] = f"{record['asset_id']}{COHORT_SEPARATOR}b{year}"
                if pooled:
                    row["asset_id"] += f"{COHORT_SEPARATOR}{_technology_slug(tech)}"
            for column in scaled_columns:
                value = pd.to_numeric(record.get(column), errors="coerce")
                row[column] = float(value) * share if pd.notna(value) else value
            row.update({
                "build_year": int(year), "site_commissioning_year": record.get("commissioning_year"),
                "commissioning_year": float(year),
                "finance_vintage": "legacy" if year < policy_start_year else "new_build",
                "vintage_method": method, "stock_flow_technology": tech,
                "cohort_share": share, "cohort_legacy_share": legacy_share})
            out_rows.append(row)
    return pd.DataFrame(out_rows), pd.DataFrame(report)


# ── support contracts on the vintaged register (with task Y11) ───────────────
#: Declared (Y11, decision 4): which vintages hold which contract in the model year.
#: Low-carbon generation built after the last held round holds a post-AR7 CfD
#: (``PAR7_<build year>``, 20 years, so in 2050 every such build from 2031 is live);
#: gas CCS and hydrogen turbines hold a dispatchable power agreement (15 years, live
#: in 2050 from build year 2036). The model year is the caller's (``model_year``); the
#: known contracts are every round's (``cfd_strikes.known_contracts(model_year)``), so in
#: 2040 AR3 to AR6 are partly live.  Build years before these with no live known
#: contract are merchant.
CFD_TECHNOLOGIES = ("Offshore wind", "Onshore wind", "Solar PV", "Tidal stream", "Large-scale nuclear",
                    "Small modular reactors (SMRs)", "Biomass - with CCUS - mature")
DPA_TECHNOLOGIES = ("Gas generation - CCUS - mature", "Hydrogen CCHT", "Hydrogen OCHT")
POST_AR7_FIRST_BUILD_YEAR = 2031
#: User decision (13 Nov 2026, results review finding 9): renewables built in 2029-30 and not on a known contract
#: hold the 2030 chain's modelled AR8 CfD (``cfd_strike_table.MODELLED_ROUND``, strikes from
#: ``build_ar8_modelled_strikes.py``), on the CfD shares the CP30 2030 chain gives the uncontracted increment
#: (``support.labels.fes_cfd_fractions`` with ``source/CP30.yaml``: offshore and onshore all, solar half; tidal
#: stream all, as every tidal project is contract-dependent), so the 2030, 2040 and 2050 years agree on who holds
#: a contract. Before it these build years were merchant in 2040 and 2050 (about 14 GW of offshore in 2030).
AR8_MODELLED_ROUND = "AR8_modelled"
AR8_MODELLED_BUILD_YEARS = (2029, 2030)
AR8_MODELLED_CFD_SHARE = {"Offshore wind": 1.0, "Onshore wind": 1.0, "Solar PV": 0.5, "Tidal stream": 1.0}
#: Register component types that hold generation contracts (Links: the hydrogen turbines).
CONTRACT_COMPONENT_TYPES = ("Generator", "Link")
POST_AR7_TERM_YEARS = 20
DPA_TERM_YEARS = 15
#: Task SETTLE2050: a modelled contract's term is its technology's (Y11's declared table,
#: ``cfd_strikes_2050.strike_inputs_for``: 20 years for wind and solar, 15 for tidal stream
#: and Power BECCS, 35 for nuclear) and its live share in the model year is
#: ``capacity_market_2050.live_fraction`` (expiries spread through the expiry year), so the
#: register, the 2050 support book and the bill agree on who holds a contract. Before it,
#: every CfD vintage took 20 years and a contract expiring in the model year was merchant.
#: Known-contract technology names (Y11's table) -> stock-flow technology.
CONTRACT_TECHNOLOGY = {"Floating offshore wind": "Offshore wind"}
SUPPORT_BASIS_KNOWN, SUPPORT_BASIS_PAR7, SUPPORT_BASIS_DPA, SUPPORT_BASIS_EXPIRED = (
    "known_contract", "post_ar7_modelled", "dpa_modelled", "contract_expired_merchant")
#: Review 2 (3 Oct 2026): a build year wholly repowered without a CfD route (decision 62) is merchant new
#: build with its capital still to recover, never an expired contract (whose capital is sunk).
SUPPORT_BASIS_REPOWERED = "repowered_merchant"
#: Review 2 (3 Oct 2026; rebuilt 24 Oct 2026): a known contract's last row is split so the contract holds
#: exactly its MW (HPC 1,630 + 1,630, Sizewell C 3,260, every AR7 phase). The row's extensive columns are
#: shared between the two parts, so capacity and each component's cohort shares are unchanged.
SPLIT_COLUMNS = ("capacity_mw", "energy_capacity_mwh", "dispatch_mwh", "cohort_share")
SPLIT_TOLERANCE_MW = 1e-6


def label_support_contracts(cohorts: pd.DataFrame, contracts: pd.DataFrame, *, model_year: int,
                            award_years: Mapping[str, int] | None = None,
                            eligible_award_years_from: int = POLICY_START_YEAR,
                            no_cfd_on_repowering: Iterable[str] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the vintaged register's generation rows the contract each holds in the model year.

    Known contracts first (``cfd_strikes.known_contracts``: every live round's phases, HPC,
    Sizewell C's RAB), each live phase on the rows of its technology and start year
    (``floor(start_year)``; the nearest other build year when that cell is short),
    rows in md5 order filled to exactly the phase's MW: whole rows while they fit, then
    the next row split (:data:`SPLIT_COLUMNS` scaled; the labelled part is
    ``<asset_id>::k<contract_id>_<phase>``, the rest keeps the row's id and stays open
    for the next phase), with
    ``cfd_live_share`` the phase's share of the model year (a phase expiring in the
    year is part-live).  Then the modelled vintages: ``PAR7_<year>`` CfDs and
    ``DPA_<year>`` agreements by :data:`CFD_TECHNOLOGIES` / :data:`DPA_TECHNOLOGIES`,
    live on their technology's term (:func:`modelled_contract_term`) with
    ``cfd_live_share`` = ``live_fraction(build year, term, model year)``, and merchant
    where the term has run out.  Storage and technologies outside both lists keep the
    network's labels.

    Task SETTLE2050: a labelled CfD or DPA row's ``network_support_type`` is set to its
    label, because the 2050 bill and settlement settle the register's labels themselves
    (``contract_book_2050``): route R1 re-prices a contract the bill settles, and here the
    bill settles every labelled one.

    Args:
        cohorts: :func:`assign_build_years` output (``stock_flow_technology``, ``build_year``).
        contracts: ``scripts.market.cfd_strikes.known_contracts(model_year)``.
        model_year: The register's model year.
        award_years: ``cfd_round -> award year`` (``cfd_round_award_years.csv``); a
            modelled round's award year is its build year (declared; post-policy either way).
        eligible_award_years_from: First award year that is post-policy (config
            cfd_repricing.eligible_award_years_from): HPC (2016) is pre-policy, AR7 (2026) post.
        no_cfd_on_repowering: FES 2050 decision 62: technologies with no CfD route for
            repowering (solar). A ``PAR7_<year>`` row of one holds its CfD on the share of its
            build year that is new capacity only: ``cfd_live_share`` is scaled by ``1 -
            cohort_replacement_share`` (the mix's ``replacement_share``), and the row records
            the merchant share as ``repowered_merchant_share``. Empty: unchanged.

    Returns:
        ``(cohorts, report)``: the labelled rows, and per known phase the MW asked and assigned
        (``split_asset_id``: the row split to make it exact, if any). With no split the rows and
        their order are the input's.
    """
    out = cohorts.copy()
    for column, default in (("support_type", None), ("cfd_round", None), ("cfd_contract_id", None),
                            ("cfd_live_share", np.nan), ("contract_vintage", None), ("cfd_award_year", np.nan),
                            ("support_basis", None), ("contract_round_at_build", None)):
        if column not in out.columns:
            out[column] = default
    for column in ("support_type", "cfd_round", "cfd_contract_id", "contract_vintage", "support_basis",
                   "contract_round_at_build"):
        out[column] = out[column].astype(object)
    for column in ("cfd_live_share", "cfd_award_year"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    award_years = dict(award_years or {})
    tech = out["stock_flow_technology"].astype(object).to_numpy()
    year = pd.to_numeric(out["build_year"], errors="coerce").to_numpy(dtype=float)
    capacity = pd.to_numeric(out["capacity_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # Hydrogen turbines are Links in the HT50 build (task Y14-prep); they hold DPAs like generators.
    is_generator = out["component_type"].astype(str).isin(CONTRACT_COMPONENT_TYPES).to_numpy()
    order = np.array([_order_key(a) for a in out["asset_id"].astype(str)], dtype=object)
    taken = np.zeros(len(out), dtype=bool)
    col = {name: out.columns.get_loc(name) for name in out.columns}

    def put(i, **values):
        for name, value in values.items():
            out.iat[i, col[name]] = value

    def split(i: int, part_mw: float, part_id: str) -> None:
        """Row ``i`` becomes ``part_mw`` of itself under ``part_id``; the rest, under the row's id, follows it."""
        nonlocal out, tech, year, capacity, is_generator, order, taken
        fraction = part_mw / capacity[i]
        rest = out.iloc[[i]].copy()
        for name in SPLIT_COLUMNS:
            if name in out.columns:
                value = pd.to_numeric(out.iat[i, col[name]], errors="coerce")
                if pd.notna(value):
                    out.iat[i, col[name]] = float(value) * fraction
                    rest.iat[0, col[name]] = float(value) * (1.0 - fraction)
        out.iat[i, col["asset_id"]] = part_id
        out = pd.concat([out.iloc[:i + 1], rest, out.iloc[i + 1:]], ignore_index=True)
        tech, year, is_generator, order = (np.insert(a, i + 1, a[i]) for a in (tech, year, is_generator, order))
        capacity = np.insert(capacity, i + 1, capacity[i] - part_mw)
        capacity[i] = part_mw
        taken = np.insert(taken, i + 1, False)

    report = []
    live = contracts.loc[pd.to_numeric(contracts["live_share"], errors="coerce").fillna(0.0) > 0]
    for c in live.sort_values(["start_year", "contract_id", "phase"]).itertuples(index=False):
        technology = CONTRACT_TECHNOLOGY.get(str(c.technology), str(c.technology))
        start = int(math.floor(float(c.start_year)))
        want = float(c.capacity_mw)
        candidates = np.flatnonzero((tech == technology) & is_generator & ~taken & np.isfinite(year))
        ranked = sorted(candidates, key=lambda i: (abs(year[i] - start), year[i] < start, order[i]))
        assigned, split_id = 0.0, None
        scheme = str(c.scheme) if str(c.scheme) not in ("", "nan") else "CfD"
        awarded = float(award_years.get(str(c.cfd_round), np.nan))
        rab = "RAB" in str(c.cfd_round)
        vintage = ("not_applicable" if rab else "unknown_round" if not np.isfinite(awarded)
                   else "post_policy" if awarded >= eligible_award_years_from else "pre_policy")
        for i in ranked:
            need = want - assigned
            if need <= SPLIT_TOLERANCE_MW:
                break
            if capacity[i] > need + SPLIT_TOLERANCE_MW:
                # The last row is split, so the phase holds exactly its MW (review 2, 3 Oct 2026). The
                # rest is inserted after it, so ``i`` (and every earlier row) keeps its position.
                split_id = str(out.iat[i, col["asset_id"]])
                split(i, need, f"{split_id}{COHORT_SEPARATOR}k{c.contract_id}_{int(c.phase)}")
            taken[i] = True
            assigned += capacity[i]
            put(i, support_type="RAB" if rab else scheme, cfd_round=str(c.cfd_round),
                cfd_contract_id=f"{c.contract_id}#{int(c.phase)}", cfd_live_share=float(c.live_share),
                contract_vintage=vintage, cfd_award_year=awarded, support_basis=SUPPORT_BASIS_KNOWN)
            if split_id is not None:
                break
        report.append({"contract_id": c.contract_id, "phase": int(c.phase), "technology": technology,
                       "start_build_year": start, "contract_mw": want, "assigned_mw": assigned,
                       "live_share": float(c.live_share), "split_asset_id": split_id})
    from scripts.market.capacity_market_2050 import live_fraction

    no_cfd = set(no_cfd_on_repowering or ())
    replaced = (pd.to_numeric(out["cohort_replacement_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
                if no_cfd and "cohort_replacement_share" in out.columns else np.zeros(len(out)))
    if no_cfd:
        out["repowered_merchant_share"] = 0.0
        col["repowered_merchant_share"] = out.columns.get_loc("repowered_merchant_share")
    for i in np.flatnonzero(~taken & is_generator & np.isfinite(year)):
        technology, b = tech[i], int(year[i])
        if technology in CFD_TECHNOLOGIES:
            ar8 = b in AR8_MODELLED_BUILD_YEARS and technology in AR8_MODELLED_CFD_SHARE
            if ar8:
                share = (live_fraction(b, modelled_contract_term(technology, b), model_year)
                         * AR8_MODELLED_CFD_SHARE[technology])
            else:
                share = (live_fraction(b, modelled_contract_term(technology, b), model_year)
                         if b >= POST_AR7_FIRST_BUILD_YEAR else 0.0)
            repowered_whole = False
            if technology in no_cfd and share > 0 and replaced[i] > 0:
                # Decision 62: the repowered share of a solar build year is merchant.
                put(i, repowered_merchant_share=float(replaced[i]))
                share = share * (1.0 - float(replaced[i]))
                repowered_whole = share <= 0
            # The AR8 cohort is a modelled contract on its technology's term, as PAR7 (the same support basis:
            # its financing term, legislation status and strike lookup follow the modelled rule).
            round_name = AR8_MODELLED_ROUND if ar8 else f"PAR7_{b}"
            label, basis, kind = (round_name, SUPPORT_BASIS_PAR7, "CfD") if share > 0 else (
                None, SUPPORT_BASIS_REPOWERED if repowered_whole else SUPPORT_BASIS_EXPIRED, "merchant")
            # S1/S2 audit (14 Nov 2026): the contract the row was built on, whatever its state in this model year, so
            # its legislation status (committed at the award) is the same in every year.
            built_on = round_name if (ar8 or b >= POST_AR7_FIRST_BUILD_YEAR) else None
        elif technology in DPA_TECHNOLOGIES:
            share = live_fraction(b, modelled_contract_term(technology, b), model_year)
            label, basis, kind = (f"DPA_{b}", SUPPORT_BASIS_DPA, "DPA") if share > 0 else (
                None, SUPPORT_BASIS_EXPIRED, "merchant")
            built_on = f"DPA_{b}"
        else:
            continue
        award = (float(award_years.get(AR8_MODELLED_ROUND, np.nan)) if label == AR8_MODELLED_ROUND
                 else float(b) if label else np.nan)
        put(i, contract_round_at_build=built_on)
        put(i, support_type=kind, cfd_round=label, cfd_contract_id=None, cfd_live_share=share if label else np.nan,
            contract_vintage=("not_applicable" if not label
                              else "post_policy" if not np.isfinite(award) or award >= eligible_award_years_from
                              else "pre_policy"),
            cfd_award_year=award, support_basis=basis)
    # SETTLE2050: the bill settles these labels (contract_book_2050), so they are network-settled.
    if "network_support_type" not in out.columns:
        out["network_support_type"] = None
    out["network_support_type"] = out["network_support_type"].astype(object)
    settled = out["support_type"].isin(["CfD", "DPA"]) & out["cfd_round"].notna()
    out.loc[settled, "network_support_type"] = out.loc[settled, "support_type"]
    return out, pd.DataFrame(report)


@lru_cache(maxsize=256)
def modelled_contract_term(technology: str, build_year: int) -> int:
    """A modelled 2050 contract's term in years (task SETTLE2050).

    CfD technologies take Y11's declared term (``cfd_strike_inputs_2050.csv`` through
    ``cfd_strikes_2050.strike_inputs_for``: 20 years for wind and solar, 15 for tidal
    stream and Power BECCS, 35 for nuclear); DPAs the scheme's 15 years
    (``support_regime_2050.json`` ``dpa.term_years``).
    """
    if technology in DPA_TECHNOLOGIES:
        from scripts.public_ownership.support_regime_2050 import _value, load_parameters

        return int(_value(load_parameters()["dpa"]["term_years"]))
    from scripts.market.cfd_strikes import strike_inputs_for

    return int(strike_inputs_for(technology, int(build_year))["term_years"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _read_repd(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1", low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", default=None, help="the FES 2025 data workbook (default: Y1's)")
    parser.add_argument("--pathway", default=DEFAULT_PATHWAY)
    parser.add_argument("--base-year", type=int, default=BASE_YEAR)
    parser.add_argument("--target-year", type=int, default=TARGET_YEAR)
    parser.add_argument("--rule", default=RULE_CENTRAL, choices=sorted(RETIREMENT_RULES))
    parser.add_argument("--life-source", default="central", choices=LIFE_SOURCES)
    parser.add_argument("--life-override", nargs="*", default=[],
                        help="TECHNOLOGY=YEARS over the life source (FES 2050 decision 48: 'Onshore wind=25')")
    parser.add_argument("--refurbishment", default=None,
                        help="F28 (d): TECH;TECH=FIRST:LAST, strict rule only (hydro and pumped hydro refurbished)")
    parser.add_argument("--repowering", nargs="*", default=[],
                        help="TECHNOLOGY=SHARE:MAX_LIFE (FES 2050 decision 62: 'Onshore wind=0.3:35')")
    parser.add_argument("--exclude-country", nargs="*", default=[],
                        help="REPD countries left out of the evidence (decision 62: 'Northern Ireland')")
    parser.add_argument("--stale-records", default=None,
                        help="CSV of REPD ref_ids left out of the evidence (decision 62: stale repowered originals)")
    parser.add_argument("--repd", default=str(DEFAULT_REPD))
    parser.add_argument("--dukes", default=str(DEFAULT_DUKES))
    parser.add_argument("--lives", default=str(DEFAULT_LIVES))
    parser.add_argument("--closures", default=str(DEFAULT_NUCLEAR_CLOSURES))
    parser.add_argument("--out-mix", required=True)
    parser.add_argument("--out-flows", required=True)
    parser.add_argument("--out-retirements", required=True)
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()
    from scripts.FES.fes2025_targets import FES2025_WORKBOOK, read_es1

    es1 = read_es1(args.workbook or FES2025_WORKBOOK)
    trajectory = fes_capacity_trajectory(es1, pathway=args.pathway, years=range(args.base_year, args.target_year + 1))
    evidence = existing_fleet_evidence(_read_repd(args.repd), pd.read_csv(args.dukes), base_year=args.base_year,
                                       exclude_countries=args.exclude_country,
                                       exclude_ref_ids=load_stale_records(args.stale_records))
    overrides = parse_life_overrides(args.life_override)
    lives = load_lives(args.lives, source=args.life_source, overrides=overrides)
    repowering = parse_repowering(args.repowering)
    result = build_fleet_vintages(trajectory, evidence, lives, base_year=args.base_year, target_year=args.target_year,
                                  rule=args.rule, closure_years=load_closure_dates(args.closures),
                                  repowering=repowering, refurbishment=parse_refurbishment(args.refurbishment))
    for key, path in (("mix", args.out_mix), ("flows", args.out_flows), ("retired_by_decade", args.out_retirements)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        result[key].to_csv(path, index=False)
    summary = dict(result["summary"], pathway=args.pathway, life_source=args.life_source,
                   vintage_summary=result["vintage_summary"].to_dict("records"))
    if overrides:
        summary["life_overrides"] = overrides
    if args.exclude_country or args.stale_records:
        summary["evidence_exclusions"] = {"countries": list(args.exclude_country),
                                          "stale_records_file": args.stale_records,
                                          "ref_ids": load_stale_records(args.stale_records)}
    Path(args.out_summary).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(result["vintage_summary"].round(0).to_string(index=False))
    print(result["retired_by_decade"].loc[result["retired_by_decade"]["technology"].eq("TOTAL")].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
