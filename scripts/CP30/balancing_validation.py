#!/usr/bin/env python3
"""Compare a solved CP30 2030 balancing family against NESO's published figures (18 Oct 2026).

Report only: every row is PASS, WARN or INFO, and the rule never fails a run. A family is a
``monthly_aggregates`` study (a study's study.yaml) (``CP30_2030_<tag>``): twelve balancing months
``CP30_FFR_2030_<Mon>_<tag>``, each with its Stage 1 from its ``wholesale_from``
scenario. Five sections, each against sourced, committed benchmarks
(``data/CP30/validation_references/neso_balancing_benchmarks.csv``, written by
:func:`extract_benchmarks` from the CP30 workbook with sheet and cell references; a test re-reads
the workbook):

1. **volumes**: balancing actions in CP.18's groups (offers and bids of interconnectors, unabated
   gas incl. CHP, wind, and "Other", storage included), TWh, against CP.18 as published (FFR,
   the recommended network). Interconnector actions are each cable's deviation from its Stage 1
   flow (import orientation: positive is more import or less export, an offer). The churn sink,
   load shedding and the foreign market generators are not BM actions. The ES1-start restatement
   of CP.18 (CP.18 moved by NESO's start-point gap, N2 - CP.18 net - ES1) is reported as INFO.
2. **cost**: the NESO-comparable redispatch cost on the family's headline turn-down basis
   (``{study}_pro_rata_turn_down_summary.json``, ``scripts/market/turn_down_allocation.py``: as solved
   ``cheapest_first`` or ``pro_rata_zonal``, named in the row; the other basis is INFO). Before the
   re-pricing, the as-solved cost: the NESO-comparable redispatch cost (``aggregate.neso_comparable_redispatch_cost_GBP``
   of the family's annual summary: ordinary redispatch + interconnector re-trading, VoLL slack and
   non-participant turn-down out) against NESO's GBP 3,570 m (main report Table 1, p.36; thermal;
   2024 prices, as the model). The split by technology group (``*_constraint_costs.csv`` without
   the TOTAL row, which double counts; IC re-trading from the market summaries) is INFO.
3. **final**: the balancing (final) dispatch against N2 (CP30's physical dispatch with network
   constraints): wind, gas incl. CHP, unabated gas (CCGT + OCGT + gas engines), interconnector
   imports, exports and net export, and the unabated gas share of generation on N2's definition
   (18,539 / 373,988 GWh = 4.96%: generation excluding storage; ours, every generator except the
   foreign market, load shedding and the churn sink).
4. **zones**: clean-power turn-down (bids of wind, solar, hydro and marine) by NESO's 17 economic
   zones against CP.08 (recommended network, FFR): shares and TWh, and Scotland (zones 1-4). CP.08
   exceeds every CP.18 bid (another run vintage), so its shares are judged, its TWh are INFO.
   Buses are zoned at run time on the family's own network
   (``data/CP30/validation_references/neso_cp08_zones.geojson``, decoded from CP.08's heat maps):
   polygon (nearest polygon offshore), the network graph for buses more than 2 km outside every
   polygon, then reconciled with the bus's side of the zone-defining ETYS boundaries.
5. **score**: the pre-stated central-case score (user, 18 Oct 2026): mean |gap| over cost, final
   wind, final gas incl. CHP and final net export; a run more than 15% off on cost cannot be
   central.
6. **delay** (a family whose ``network_delivery`` case is a delay case): its cost against NESO's
   figure for that network (no_acceleration 7.79 bn; seven_late 3.57 + 6.67 bn of single-project
   impacts; no_upgrades 12.70 bn) and, for no_upgrades, the clean turn-down against CP.08's
   "network we have today". Sections 1-5 are then INFO (they are the recommended network's
   figures).

Tolerances: :data:`DEFAULT_CONFIG` (scenario block ``validation.balancing``). Which families
join ``rule all``: ``balancing_validation_families`` in config/studies/cp30_2030/study.yaml (rules/CP30.smk).

    python scripts/CP30/balancing_validation.py --study CP30_2030_cp30_network --out-dir <dir>
    python scripts/CP30/balancing_validation.py --extract-benchmarks   # rewrite the benchmark CSV
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WORKBOOK = "data/CP30/Clean Power 2030 Data Workbook_V003.xlsx"
BENCHMARKS_FILE = "data/CP30/validation_references/neso_balancing_benchmarks.csv"
ZONES_FILE = "data/CP30/validation_references/neso_cp08_zones.geojson"
SOURCE_URL = "https://www.neso.energy/publications/clean-power-2030"

#: Code defaults of the scenario block ``validation.balancing`` (in
#: config/defaults/method/brownfield.yaml; the two must match, tests/unit/test_cp30_balancing_validation.py).
DEFAULT_CONFIG: dict = {
    "benchmarks_file": BENCHMARKS_FILE,
    "zones_file": ZONES_FILE,
    "volume_basis": "published",     # CP.18 as published (the CP.28-30 calibration); "es1_start": CP.18 restated to an ES1 start
    "volume_warn_rel": 0.15,         # every CP.18 volume
    "small_volume_twh": 1.0,         # CP.18 below this: judged on |difference| / this
    "cost_warn_rel": 0.15,           # NESO-comparable cost against GBP 3,570 m
    "final_warn_rel": 0.05,          # final positions against N2
    "gas_share_warn_abs": 0.01,      # unabated gas share of generation against N2's 4.96%
    "zone_share_warn_abs": 0.05,     # each zone's share of clean turn-down against CP.08
    "scotland_share_warn_rel": 0.15,  # zones 1-4's share against CP.08's 0.532
    "central_cost_max_rel": 0.15,    # a run more than this off on cost cannot be central
    "delay_cost_warn_rel": 0.20,     # a delay family's cost against NESO's figure for its network
    "remote_bus_km": 2.0,            # buses further outside every zone polygon are zoned by the graph
    "graph_cutoff": 6,               # branches searched for the nearest on-land bus
}

# ── Carrier groups ──────────────────────────────────────────────────────────
#: Not balancing actions: the foreign market generators (their moves are the IC cable deviations),
#: VoLL slack and the churn sink (F22).
NOT_ACTIONS = {"load_shedding", "EU_market", "churn_sink"}
CP18_GAS = {"CCGT", "OCGT", "gas_engine", "CHP"}
CP18_WIND = {"wind_onshore", "wind_offshore", "embedded_wind"}
STORAGE = {"Battery", "battery", "Domestic Battery", "LAES", "CAES", "pumped_hydro",
           "Pumped Storage Hydroelectricity"}
#: Clean power in CP.08's sense (bid volumes of clean power sources constrained).
CLEAN = {"wind_offshore", "wind_onshore", "embedded_wind", "solar_pv", "embedded_solar", "large_hydro",
         "small_hydro", "marine", "tidal_stream", "shoreline_wave"}
FINAL_WIND = {"wind_onshore", "wind_offshore", "embedded_wind"}
FINAL_GAS = {"CCGT", "OCGT", "gas_engine", "CHP"}
UNABATED_GAS = {"CCGT", "OCGT", "gas_engine"}
GENERATION_EXCLUDED = {"EU_market", "load_shedding", "churn_sink"}
COST_GROUP = {"wind_offshore": "wind", "wind_onshore": "wind", "embedded_wind": "wind", "solar_pv": "solar",
              "embedded_solar": "solar", "CCGT": "gas", "OCGT": "gas", "gas_engine": "gas", "CHP": "gas CHP",
              "Battery": "storage", "battery": "storage", "Domestic Battery": "storage", "LAES": "storage",
              "Pumped Storage Hydroelectricity": "storage", "pumped_hydro": "storage", "CAES": "storage",
              "nuclear": "nuclear", "biomass": "biomass", "CCS Biomass": "biomass", "large_hydro": "hydro",
              "small_hydro": "hydro", "EU_market": "foreign market (cost in IC re-trading)", "CCS Gas": "gas CCS",
              "load_shedding": "load shedding (VoLL slack)", "churn_sink": "churn sink"}

# ── NESO zones ──────────────────────────────────────────────────────────────
from scripts.market.neso_zones import SCOTLAND, ZONE_BOUNDARIES, ZONES, is_external_bus  # noqa: E402,F401

#: network_delivery case keyword -> (benchmark metric of NESO's cost for that network, label).
DELAY_CASES = {
    "no_acceleration": ("neso_cost_no_acceleration_gbpm", "Norwich-Tilbury and Sea Link on their published dates"),
    "seven_late": ("neso_cost_seven_late_gbpm", "the seven critical projects late"),
    "no_upgrades": ("neso_cost_no_upgrades_gbpm", "no network upgrades after 2025"),
}

COLUMNS = ["section", "metric", "unit", "neso", "model", "difference", "relative_difference", "judged_difference",
           "basis", "tolerance", "status", "neso_metric", "neso_source", "note"]
BENCHMARK_COLUMNS = ["metric", "value", "unit", "sheet", "cell", "source_url", "note"]


def balancing_config(block: dict | None) -> dict:
    """``validation.balancing`` over :data:`DEFAULT_CONFIG`, refusing unknown keys."""
    block = dict(block or {})
    if unknown := sorted(set(block) - set(DEFAULT_CONFIG)):
        raise ValueError(f"validation.balancing has unknown keys {unknown}; supported: {sorted(DEFAULT_CONFIG)}")
    cfg = {**copy.deepcopy(DEFAULT_CONFIG), **block}
    if cfg["volume_basis"] not in ("published", "es1_start"):
        raise ValueError(f"validation.balancing.volume_basis must be published or es1_start, "
                         f"got {cfg['volume_basis']!r}")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks from the CP30 workbook
# ═══════════════════════════════════════════════════════════════════════════
FFR = "Further Flex and Renewables"
#: Figures not in the workbook: the main report and the NAO's reading of NESO's delay analysis.
REPORT_BENCHMARKS = [
    {"metric": "neso_constraint_cost_gbpm", "value": 3570.0, "unit": "GBP m", "sheet": "main report",
     "cell": "Table 1, p.36 ('network required for clean power, incl. 3 critical projects', FFR)",
     "source_url": SOURCE_URL,
     "note": "NESO, Clean Power 2030 (5 Nov 2024): thermal constraints only (Annex 4 fn 13); 2024 prices (Annex 4 "
             "s.2), as the model; 2013 weather; interconnector counter-trades inside by inference (CP.18's IC actions)"},
    {"metric": "neso_cost_no_acceleration_gbpm", "value": 7790.0, "unit": "GBP m", "sheet": "NAO report",
     "cell": "Figure 4 and para 2.8",
     "source_url": "",
     "note": "NAO, Upgrading the electricity transmission network (Sep 2026), quoting NESO CP30 (2024 prices, "
             "thermal): Norwich-Tilbury and Sea Link on their published 2031-33 dates, 6.58-7.79 bn; the upper "
             "end"},
    {"metric": "neso_cost_seven_late_single_projects_gbpm", "value": 6670.0, "unit": "GBP m", "sheet": "NAO report",
     "cell": "Figure 4 and para 2.8",
     "source_url": "",
     "note": "NAO (Sep 2026), quoting NESO CP30: sum of the single-project impacts of the seven critical projects "
             "being late (up to +6.7 bn over the central network)"},
    {"metric": "neso_cost_no_upgrades_gbpm", "value": 12700.0, "unit": "GBP m", "sheet": "NAO report",
     "cell": "Figure 4 and para 2.8",
     "source_url": "",
     "note": "NAO (Sep 2026), quoting NESO CP30: nothing after 2025, 10.9-12.7 bn; the upper end"},
]


def _num(value) -> float:
    if value is None:
        raise ValueError("empty benchmark cell")
    return float(value)


def extract_benchmarks(workbook: str | Path = WORKBOOK) -> pd.DataFrame:
    """Every benchmark row: read from the workbook (each label cell checked), derived, or from the reports.

    Energy in TWh (CP.18 and N2 are GWh in the sheet; ES1's energy lines too), cost in GBP m.
    """
    import openpyxl

    wb = openpyxl.load_workbook(str(workbook), read_only=True, data_only=True)
    rows: list[dict] = []

    def add(metric, value, unit, sheet, cell, note=""):
        rows.append({"metric": metric, "value": float(value), "unit": unit, "sheet": sheet, "cell": cell,
                     "source_url": SOURCE_URL, "note": note})

    def check(sheet, ref, want):
        got = str(wb[sheet][ref].value).replace("\xa0", " ")
        if want.lower() not in got.lower():
            raise ValueError(f"{sheet}!{ref} is {got!r}, expected {want!r}: the workbook layout moved")

    # CP.18: bid-offer volumes, FFR (columns E offers, F bids), the recommended network (CP.18!B14).
    s = "CP.18"
    check(s, "E5", FFR)
    check(s, "E6", "Offers")
    check(s, "F6", "Bids")
    check(s, "B14", "network scenario recommended")
    cp18 = {}
    for row, group, label in ((7, "ic", "IC"), (8, "gas", "Unabated Gas"), (9, "wind", "Wind"),
                              (10, "other", "Other"), (11, "total", "Total")):
        check(s, f"B{row}", label)
        for col, side in (("E", "offer"), ("F", "bid")):
            gwh = _num(wb[s][f"{col}{row}"].value)
            cp18[f"{side}_{group}"] = gwh
            note = f"CP.18 '{label}' {side}s, FFR, recommended network; GWh in the sheet"
            if group == "other":
                note += ("; 'Other' is undefined by NESO: everything not IC, unabated gas or wind (storage, solar, "
                         "biomass, nuclear, hydro)")
            if group == "gas":
                note += "; CHP inclusion not stated (the model counts CHP in this group)"
            add(f"cp18_{side}_{group}_twh", gwh / 1e3, "TWh", s, f"{col}{row}", note)

    # N2: redispatch data (physical dispatch with network constraints), FFR rows, column I = 2030.
    s = "N2"
    check(s, "D6", "Pathway")
    check(s, "I6", "2030")
    n2 = []
    for i, row in enumerate(wb[s].iter_rows(min_row=7, max_row=400, min_col=3, max_col=9, values_only=True), start=7):
        if row[1] == FFR:
            n2.append({"row": i, "variable": row[2], "category": row[3], "type": row[4], "subtype": row[5],
                       "value": _num(row[6])})
    n2 = pd.DataFrame(n2)
    if n2.empty:
        raise ValueError("N2 has no FFR rows")

    def n2_sum(mask, metric, note):
        sel = n2[mask]
        if sel.empty:
            raise ValueError(f"N2: no rows for {metric}")
        add(metric, sel["value"].sum() / 1e3, "TWh", s, "+".join(f"I{r}" for r in sel["row"]), note)
        return sel["value"].sum()

    gen = n2["variable"].eq("Generation (GWh)")
    wind = n2_sum(gen & n2["type"].isin(["Offshore Wind", "Onshore Wind"]), "n2_wind_twh",
                  "N2 wind generation (offshore + onshore, Tx + Dx), FFR 2030")
    unabated = n2_sum(gen & n2["type"].eq("Gas") & n2["subtype"].isin(["CCGT", "OCGT", "Gas Reciprocating Engines"]),
                      "n2_unabated_gas_twh", "N2 CCGT + OCGT + gas reciprocating engines (CHP outside)")
    n2_sum(gen & n2["type"].eq("Gas"), "n2_gas_incl_chp_twh", "N2 unabated gas incl. gas CHP")
    total = n2_sum(gen & n2["category"].ne("Storage"), "n2_generation_excl_storage_twh",
                   "N2 'Generation (GWh)' rows except storage: the denominator of the unabated gas share")
    imp = n2_sum(n2["variable"].eq("Import (GWh)"), "n2_ic_import_twh", "N2 interconnector imports")
    exp = n2_sum(n2["variable"].eq("Export (GWh)"), "n2_ic_export_twh", "N2 interconnector exports")
    imp_row = int(n2.loc[n2["variable"].eq("Import (GWh)"), "row"].iloc[0])
    exp_row = int(n2.loc[n2["variable"].eq("Export (GWh)"), "row"].iloc[0])
    add("n2_net_export_twh", (exp - imp) / 1e3, "TWh", s, f"I{exp_row}-I{imp_row}", "N2 exports minus imports")
    unabated_rows = n2[gen & n2["type"].eq("Gas") & n2["subtype"].isin(["CCGT", "OCGT", "Gas Reciprocating Engines"])]
    total_rows = n2[gen & n2["category"].ne("Storage")]
    add("n2_unabated_gas_share", unabated / total, "share", s,
        f"({'+'.join(f'I{r}' for r in unabated_rows['row'])}) / ({'+'.join(f'I{r}' for r in total_rows['row'])})",
        "N2's headline test: unabated gas share of generation excluding storage (CHP outside: its emissions count "
        "to industry)")

    # ES1 (FFR 2030 = column P): the economic run CP.18's start-point gap is measured against.
    s = "ES1"
    check(s, "D10", "Pathway")
    check(s, "P10", "2030")
    es1 = {}
    for key, refs, want in (("wind", (21, 57, 22, 58), "Wind"), ("gas", (26, 27), "Gas"),
                            ("import", (78,), "Import"), ("export", (77,), "Export")):
        for r in refs:
            label = " / ".join(str(wb[s][f"{c}{r}"].value) for c in ("C", "D", "E", "F", "G", "H"))
            if want.lower() not in label.lower():
                raise ValueError(f"ES1 row {r} is {label!r}, expected {want!r}")
            if FFR.lower() not in label.lower():
                raise ValueError(f"ES1 row {r} is not an FFR row: {label!r}")
        es1[key] = sum(_num(wb[s][f"P{r}"].value) for r in refs)
        metric = {"wind": "es1_wind_twh", "gas": "es1_gas_incl_chp_twh", "import": "es1_ic_import_twh",
                  "export": "es1_ic_export_twh"}[key]
        add(metric, es1[key] / 1e3, "TWh", s, "+".join(f"P{r}" for r in refs),
            "ES1 (CP30's unconstrained market dispatch), FFR 2030")

    # The ES1-start restatement (Run A basis): NESO's redispatch started from its 3-hour economic run, not
    # ES1; start - ES1 = N2 - CP.18 net - ES1, put on the dominant side of each group.
    n2v = {"wind": wind, "gas": n2[gen & n2["type"].eq("Gas")]["value"].sum(), "ic": imp - exp}
    es1v = {"wind": es1["wind"], "gas": es1["gas"], "ic": es1["import"] - es1["export"]}
    net = {g: cp18[f"offer_{g}"] - cp18[f"bid_{g}"] for g in ("wind", "gas", "ic")}
    gap = {g: n2v[g] - net[g] - es1v[g] for g in ("wind", "gas", "ic")}
    for metric, value, cell, note in (
            ("es1_start_bid_wind_twh", cp18["bid_wind"] - gap["wind"], "CP.18!F9 - (N2 wind - (CP.18!E9-F9) - ES1 wind)",
             f"start - ES1 = {gap['wind'] / 1e3:+.3f} TWh of wind"),
            ("es1_start_offer_ic_twh", cp18["offer_ic"] + gap["ic"], "CP.18!E7 + (N2 IC net import - (CP.18!E7-F7) - ES1)",
             f"start - ES1 = {gap['ic'] / 1e3:+.3f} TWh of net import"),
            ("es1_start_bid_ic_twh", cp18["bid_ic"], "CP.18!F7", "the start-point gap is on the offer side"),
            ("es1_start_offer_gas_twh", cp18["offer_gas"] + gap["gas"],
             "CP.18!E8 + (N2 gas incl. CHP - (CP.18!E8-F8) - ES1 gas incl. CHP)",
             f"start - ES1 = {gap['gas'] / 1e3:+.3f} TWh of gas")):
        rows.append({"metric": metric, "value": value / 1e3, "unit": "TWh", "sheet": "derived", "cell": cell,
                     "source_url": SOURCE_URL,
                     "note": "CP.18 restated to an ES1 start (N2, CP.18, ES1 cells as listed); " + note})

    # CP.08: clean-power bid volumes by zone (M network, N pathway, O zone, P TWh), rows 11-78.
    s = "CP.08"
    check(s, "P10", "Total TWh")
    totals = {}
    for r in range(11, 79):
        net_label = str(wb[s][f"M{r}"].value).replace("\xa0", " ")
        path = str(wb[s][f"N{r}"].value)
        if path != FFR:
            continue
        net_key = "today" if "today" in net_label else "recommended" if "recommending" in net_label else None
        if net_key is None:
            raise ValueError(f"CP.08!M{r} is {net_label!r}")
        zone = int(_num(wb[s][f"O{r}"].value))
        twh = _num(wb[s][f"P{r}"].value)
        totals.setdefault(net_key, []).append((r, twh))
        add(f"cp08_{net_key}_ffr_z{zone}_twh", twh, "TWh", s, f"P{r}",
            f"CP.08 clean-power bid volume, zone {zone}, FFR, network {'we have today' if net_key == 'today' else 'recommended for CP30'}")
    for net_key, items in totals.items():
        if len(items) != 17:
            raise ValueError(f"CP.08 {net_key} FFR has {len(items)} zones, not 17")
        add(f"cp08_{net_key}_ffr_total_twh", sum(v for _, v in items), "TWh", s,
            f"P{items[0][0]}:P{items[-1][0]}", "CP.08 summed over the 17 zones")
        scot = sum(v for r, v in items[:4])
        add(f"cp08_{net_key}_ffr_scotland_twh", scot, "TWh", s, f"P{items[0][0]}:P{items[3][0]}",
            "CP.08 zones 1-4 (Scotland)")

    # CP.28-30: NESO's 3-hour economic run (the start of its redispatch), context for Stage 1.
    check("CP.28", "H5", FFR)
    check("CP.28", "F9", "Net export")
    add("cp28_net_export_twh", _num(wb["CP.28"]["H9"].value), "TWh", "CP.28", "H9",
        "net export of the 3-hour economic run (the start of NESO's redispatch)")
    check("CP.30", "L6", FFR)
    check("CP.30", "J8", "Imports")
    check("CP.30", "J9", "Exports")
    add("cp30_ic_import_twh", _num(wb["CP.30"]["L8"].value), "TWh", "CP.30", "L8",
        "imports of the economic run without network constraints (FOI/26/076)")
    add("cp30_ic_export_twh", -_num(wb["CP.30"]["L9"].value), "TWh", "CP.30", "-L9", "exports of the economic run")
    # CP.31: constraint cost including voltage and stability (an earlier run; context only).
    check("CP.31", "E9", "Constraint Costs")
    add("cp31_constraint_cost_incl_voltage_gbpm", _num(wb["CP.31"]["G9"].value) * 1e3, "GBP m", "CP.31", "G9",
        "thermal plus voltage and stability (Annex 4 Fig 10; earlier model runs, so G9 - 3,570 is not the V&S cost); "
        "GBP bn in the sheet")
    wb.close()

    for r in REPORT_BENCHMARKS:
        rows.append(dict(r))
    base = 3570.0
    rows.append({"metric": "neso_cost_seven_late_gbpm", "value": base + 6670.0, "unit": "GBP m", "sheet": "derived",
                 "cell": "neso_constraint_cost_gbpm + neso_cost_seven_late_single_projects_gbpm", "source_url": "",
                 "note": "central 3,570 plus the sum of the single-project impacts (an upper bound: impacts overlap)"})
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)


def read_benchmarks(path: str | Path = BENCHMARKS_FILE) -> pd.Series:
    """{metric: value} from the committed benchmark file."""
    table = pd.read_csv(path)
    if table["metric"].duplicated().any():
        raise ValueError(f"{path}: duplicate metrics {sorted(table.loc[table['metric'].duplicated(), 'metric'])}")
    return table.set_index("metric")


# ═══════════════════════════════════════════════════════════════════════════
# Reading the solved months
# ═══════════════════════════════════════════════════════════════════════════
def _strings(d, name) -> np.ndarray:
    return np.asarray(d.variables[name][:]).astype(str)


def _floats(d, name) -> np.ndarray:
    return np.asarray(d.variables[name][:], dtype=float)


def _snapshot_index(d) -> pd.DatetimeIndex:
    v = d.variables["snapshots_snapshot"]
    units = str(getattr(v, "units", ""))
    match = re.match(r"\s*(\w+)\s+since\s+(.+)", units)
    if not match:
        raise ValueError(f"snapshots_snapshot has no 'since' units: {units!r}")
    unit = {"hours": "h", "minutes": "min", "seconds": "s", "days": "D"}[match.group(1)]
    return pd.DatetimeIndex(pd.Timestamp(match.group(2).strip()) + pd.to_timedelta(np.asarray(v[:]), unit=unit))


def read_balancing_network(path: str | Path) -> dict:
    """Static tables, per-generator weighted energy (MWh) and the link flows of a saved balancing network.

    Read with netCDF4 (a label ``.sel`` on these files is slow). Snapshot weights: ``snapshots_generators``.
    """
    import netCDF4

    d = netCDF4.Dataset(str(path))
    d.set_auto_mask(False)
    try:
        bus_i = _strings(d, "buses_i")
        buses = pd.DataFrame({"x": _floats(d, "buses_x"), "y": _floats(d, "buses_y"),
                              "lon": _floats(d, "buses_lon") if "buses_lon" in d.variables else np.nan,
                              "lat": _floats(d, "buses_lat") if "buses_lat" in d.variables else np.nan,
                              "country": (_strings(d, "buses_country") if "buses_country" in d.variables
                                          else np.full(len(bus_i), "GB"))}, index=bus_i)

        def table(prefix, cols):
            if f"{prefix}_i" not in d.variables:
                return pd.DataFrame(columns=list(cols))
            out = pd.DataFrame(index=_strings(d, f"{prefix}_i"))
            for col, kind in cols.items():
                name = f"{prefix}_{col}"
                if name in d.variables:
                    out[col] = _strings(d, name) if kind == "s" else _floats(d, name)
                else:
                    out[col] = "" if kind == "s" else np.nan
            return out

        lines = table("lines", {"bus0": "s", "bus1": "s", "s_nom": "f"})
        transformers = table("transformers", {"bus0": "s", "bus1": "s", "s_nom": "f"})
        links = table("links", {"bus0": "s", "bus1": "s", "carrier": "s", "p_nom": "f"})
        generators = table("generators", {"bus": "s", "carrier": "s"})
        storage_units = table("storage_units", {"bus": "s", "carrier": "s"})
        snapshots = _snapshot_index(d)
        p = _floats(d, "generators_t_p")
        weights = _floats(d, "snapshots_generators") if "snapshots_generators" in d.variables else np.ones(p.shape[0])
        energy = pd.Series((p * weights[:, None]).sum(axis=0), index=_strings(d, "generators_t_p_i"))
        if "links_t_p0" in d.variables:
            link_p0 = pd.DataFrame(_floats(d, "links_t_p0"), index=snapshots, columns=_strings(d, "links_t_p0_i"))
        else:
            link_p0 = pd.DataFrame(index=snapshots)
    finally:
        d.close()
    return {"buses": buses, "lines": lines, "transformers": transformers, "links": links,
            "generators": generators, "storage_units": storage_units, "snapshots": snapshots,
            "weights": pd.Series(weights, index=snapshots), "generator_energy": energy, "link_p0": link_p0}


def interconnector_import_signs(links: pd.DataFrame, buses: pd.DataFrame) -> pd.Series:
    """+1 where p0 > 0 is an import into GB (bus0 foreign), -1 where bus1 is the foreign end."""
    country = buses["country"] if "country" in buses else pd.Series("GB", index=buses.index)
    signs = {}
    for name, r in links.iterrows():
        if not str(name).startswith("IC_"):
            continue
        ext0 = is_external_bus(r["bus0"], country.get(r["bus0"], "GB"))
        ext1 = is_external_bus(r["bus1"], country.get(r["bus1"], "GB"))
        if ext0 == ext1:
            raise ValueError(f"interconnector {name}: cannot tell its foreign end ({r['bus0']} - {r['bus1']})")
        signs[name] = 1.0 if ext0 else -1.0
    return pd.Series(signs, dtype=float)


def read_stage1_links(path: str | Path, links: list[str], snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    """Stage 1 flows of ``links`` on ``snapshots`` (a cable missing from the file holds zero, logged by the caller)."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    missing_hours = snapshots.difference(frame.index)
    if len(missing_hours):
        raise ValueError(f"{path}: {len(missing_hours)} balancing snapshots missing from the Stage 1 links")
    return frame.reindex(index=snapshots, columns=links).fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Measures (pure functions on tables; the tests drive these directly)
# ═══════════════════════════════════════════════════════════════════════════
def cp18_group(carrier: str) -> str:
    if carrier in CP18_GAS:
        return "gas"
    if carrier in CP18_WIND:
        return "wind"
    if carrier in STORAGE:
        return "storage"
    return "other"


def unit_volumes(redispatch: pd.DataFrame) -> dict:
    """Offers and bids (MWh) by CP.18 group from a ``*_redispatch_summary.csv`` (non-actions out)."""
    rs = redispatch[~redispatch["carrier"].isin(NOT_ACTIONS)]
    group = rs["carrier"].map(cp18_group)
    out = {}
    for g in ("gas", "wind", "storage", "other"):
        sel = rs[group == g]
        out[f"offer_{g}"] = float(sel["increase_MWh"].sum())
        out[f"bid_{g}"] = float(sel["decrease_MWh"].sum())
    clean = rs[rs["carrier"].isin(CLEAN)]
    out["clean_down"] = float(clean["decrease_MWh"].sum())
    return out


def interconnector_volumes(balancing_p0: pd.DataFrame, stage1_p0: pd.DataFrame, import_sign: pd.Series,
                           weights: pd.Series | None = None) -> dict:
    """IC offers and bids (MWh): each cable's deviation from Stage 1 in import orientation.

    Positive deviation (more import or less export) is an offer, negative a bid.
    """
    ic = list(import_sign.index)
    dev = (balancing_p0.reindex(columns=ic).fillna(0.0) - stage1_p0.reindex(columns=ic).fillna(0.0)) * import_sign
    w = (weights.reindex(dev.index).to_numpy() if weights is not None else np.ones(len(dev)))[:, None]
    return {"offer_ic": float((dev.clip(lower=0.0).to_numpy() * w).sum()),
            "bid_ic": float(((-dev).clip(lower=0.0).to_numpy() * w).sum())}


def interconnector_energy(p0: pd.DataFrame, import_sign: pd.Series, weights: pd.Series | None = None) -> dict:
    """Imports and exports (MWh) over the cables."""
    ic = list(import_sign.index)
    flow = p0.reindex(columns=ic).fillna(0.0) * import_sign
    w = (weights.reindex(flow.index).to_numpy() if weights is not None else np.ones(len(flow)))[:, None]
    return {"ic_import": float((flow.clip(lower=0.0).to_numpy() * w).sum()),
            "ic_export": float(((-flow).clip(lower=0.0).to_numpy() * w).sum())}


def final_generation(energy: pd.Series, carriers: pd.Series) -> dict:
    """Final (balancing) generation (MWh) on N2's groups."""
    c = carriers.reindex(energy.index)
    return {"wind": float(energy[c.isin(FINAL_WIND)].sum()),
            "gas_incl_chp": float(energy[c.isin(FINAL_GAS)].sum()),
            "unabated_gas": float(energy[c.isin(UNABATED_GAS)].sum()),
            "generation": float(energy[~c.isin(GENERATION_EXCLUDED) & c.notna()].sum())}


def cost_by_group(constraint_costs: pd.DataFrame) -> pd.DataFrame:
    """Offer, bid and net cost (GBP) and volumes (MWh) by technology group; the TOTAL row dropped."""
    c = constraint_costs[~constraint_costs["carrier"].astype(str).str.upper().eq("TOTAL")].copy()
    c["group"] = c["carrier"].map(lambda x: COST_GROUP.get(x, "other"))
    return c.groupby("group")[["offer_cost", "bid_cost", "net_cost", "increase_MWh", "decrease_MWh"]].sum()


def zone_turndown(redispatch: pd.DataFrame, bus_of: pd.Series, bus_zone: pd.Series) -> tuple[pd.Series, float]:
    """Clean turn-down (MWh) by NESO zone, and the clean turn-down left unzoned (external or unknown bus)."""
    rs = redispatch[~redispatch["carrier"].isin(["churn_sink", "load_shedding"])]
    rs = rs[rs["carrier"].isin(CLEAN)]
    zone = rs["component"].map(bus_of).map(bus_zone)
    by_zone = rs.groupby(zone)["decrease_MWh"].sum().reindex(list(ZONES)).fillna(0.0)
    return by_zone, float(rs.loc[zone.isna(), "decrease_MWh"].sum())


# ── Bus -> NESO zone: scripts/market/neso_zones.py (shared with the pro rata turn-down re-pricing) ──
from scripts.market.neso_zones import (  # noqa: E402,F401  (re-exported)
    allowed_zones, boundary_crossings, boundary_sides, bus_zone_map, load_zone_polygons, polygon_zones,
)


# ═══════════════════════════════════════════════════════════════════════════
# The family
# ═══════════════════════════════════════════════════════════════════════════
def apply_turn_down_basis(measures: dict, turn_down: dict | None) -> dict:
    """Put the family's headline turn-down basis on the cost (both bases kept; no summary: as solved)."""
    solved = measures["neso_comparable_cost_GBP"]
    if not turn_down:
        measures.update(headline_basis="cheapest_first", cost_by_basis_GBP={"cheapest_first": solved},
                        turn_down_basis_source="none: as solved (no pro rata turn-down summary)")
        return measures
    by_basis = dict(turn_down.get("neso_comparable_redispatch_cost_by_basis_GBP") or {})
    basis = str(turn_down["headline_basis"])
    if by_basis.get("cheapest_first") is not None and np.isfinite(solved) and             abs(float(by_basis["cheapest_first"]) - solved) > 1.0:
        raise ValueError(f"turn-down summary's as-solved cost {by_basis['cheapest_first']:,.0f} differs from the "
                         f"annual summary's {solved:,.0f}: the summary is stale")
    measures.update(headline_basis=basis, cost_by_basis_GBP={k: float(v) for k, v in by_basis.items() if v is not None},
                    neso_comparable_cost_GBP=float(by_basis[basis]),
                    turn_down_basis_source=f"{turn_down.get('mode')}: {turn_down.get('headline_basis_reason', '')}",
                    wind_bid_cost_by_basis_GBP=turn_down.get("wind_bid_cost_by_basis_GBP"))
    return measures


def measure_family(months: list[dict], annual_summary: dict | None, polys, secured: dict | None, cfg: dict,
                   logger=None) -> dict:
    """Sum the family's months. Each month: {"name", "network", "redispatch", "constraint_costs",
    "stage1_links", "market_summary"} (paths)."""
    log = logger.info if logger else (lambda *a: None)
    warn = logger.warning if logger else (lambda *a: None)
    vol: dict[str, float] = {}
    s1: dict[str, float] = {}
    fin: dict[str, float] = {}
    costs = []
    acc: dict[str, float] = {}
    zones = pd.Series(0.0, index=list(ZONES))
    unzoned = 0.0
    hours = 0.0
    mapping_cache: dict = {}
    mapping_counts: dict = {}
    for m in months:
        net = read_balancing_network(m["network"])
        weights = net["weights"]
        hours += float(weights.sum())
        rs = pd.read_csv(m["redispatch"])
        for k, v in unit_volumes(rs).items():
            vol[k] = vol.get(k, 0.0) + v
        sign = interconnector_import_signs(net["links"], net["buses"])
        missing = [c for c in sign.index if c not in net["link_p0"].columns]
        if missing:
            warn("%s: interconnectors without saved flows %s (held at zero)", m["name"], missing)
        stage1 = read_stage1_links(m["stage1_links"], list(sign.index), net["snapshots"])
        absent = [c for c in sign.index if c not in pd.read_csv(m["stage1_links"], nrows=0).columns]
        if absent:
            warn("%s: interconnectors absent from the Stage 1 links %s (held at zero)", m["name"], absent)
        for k, v in interconnector_volumes(net["link_p0"], stage1, sign, weights).items():
            vol[k] = vol.get(k, 0.0) + v
        for k, v in interconnector_energy(stage1, sign, weights).items():
            s1[k] = s1.get(k, 0.0) + v
        for k, v in interconnector_energy(net["link_p0"], sign, weights).items():
            fin[k] = fin.get(k, 0.0) + v
        for k, v in final_generation(net["generator_energy"], net["generators"]["carrier"]).items():
            fin[k] = fin.get(k, 0.0) + v
        costs.append(cost_by_group(pd.read_csv(m["constraint_costs"])))
        a = json.loads(Path(m["market_summary"]).read_text(encoding="utf-8")).get("accounting", {})
        for k in ("neso_comparable_redispatch_cost_GBP", "ordinary_redispatch_cost_excluding_feasibility_slack_GBP",
                  "interconnector_retrading_cost_GBP", "feasibility_slack_cost_GBP", "feasibility_slack_energy_MWh",
                  "non_participant_turn_down_cost_at_penalty_GBP", "churn_sink_held_MWh"):
            acc[k] = acc.get(k, 0.0) + float(a.get(k) or 0.0)
        key = (tuple(net["buses"].index), tuple(net["lines"].index), tuple(net["transformers"].index),
               tuple(net["links"].index[net["links"]["p_nom"].fillna(0) > 0]))
        if key not in mapping_cache:
            crossings = boundary_crossings(net, ZONE_BOUNDARIES, secured)
            mapping_cache[key] = bus_zone_map(net, polys, crossings, float(cfg["remote_bus_km"]),
                                              int(cfg["graph_cutoff"]))
            counts = {k: int(v) for k, v in
                      mapping_cache[key]["mapping"].str.split(":").str[0].value_counts().items()}
            mapping_counts[m["name"]] = counts
            log("%s: buses zoned on this network: %s", m["name"], counts)
        bz = mapping_cache[key]["neso_zone"]
        bus_of = pd.concat([net["generators"]["bus"], net["storage_units"]["bus"]])
        z, u = zone_turndown(rs, bus_of, bz)
        zones += z
        unzoned += u
        log("Read %s (%.0f h)", m["name"], float(weights.sum()))
    cost = pd.concat(costs).groupby(level=0).sum() if costs else pd.DataFrame()
    neso_cost = None
    if annual_summary is not None:
        neso_cost = (annual_summary.get("aggregate") or {}).get("neso_comparable_redispatch_cost_GBP")
    cost_source = "annual summary aggregate.neso_comparable_redispatch_cost_GBP"
    if neso_cost is None:
        neso_cost = acc.get("neso_comparable_redispatch_cost_GBP", np.nan)
        cost_source = "sum of the months' market summaries (no annual summary value)"
    return {"volumes": vol, "stage1": s1, "final": fin, "cost_by_group": cost, "accounting": acc,
            "neso_comparable_cost_GBP": float(neso_cost), "cost_source": cost_source, "zones": zones,
            "unzoned_clean_MWh": unzoned, "hours": hours, "months": len(months),
            "zone_mapping_counts": mapping_counts, "distinct_topologies": len(mapping_cache)}


def delay_case_of(case: str | None, study: str = "") -> str | None:
    """The delay case keyword of a ``network_delivery`` case (``no_acceleration_outage_neso`` -> no_acceleration)."""
    for text in (str(case or ""), str(study or "")):
        for key in DELAY_CASES:
            if key in text:
                return key
    return None


# ═══════════════════════════════════════════════════════════════════════════
# The table
# ═══════════════════════════════════════════════════════════════════════════
def _row(section, metric, unit, neso, model, tol=None, *, rel=True, small=None, neso_metric="", source="",
         note="", status=None, info=False) -> dict:
    diff = rel_diff = judged = np.nan
    basis = ""
    if neso is not None and model is not None and np.isfinite(neso) and np.isfinite(model):
        diff = float(model) - float(neso)
        rel_diff = diff / float(neso) if neso else (np.inf if diff else 0.0)
        if tol is not None and status is None:
            if not rel:
                judged, basis = diff, "absolute"
            elif small is not None and abs(neso) < small:
                judged, basis = diff / small, f"absolute: difference / {small:g} (NESO below it)"
            else:
                judged, basis = rel_diff, "relative"
            status = "WARN" if abs(judged) > tol else "PASS"
    if info and status in ("PASS", "WARN"):
        note = f"{note}; INFO: {info}" if isinstance(info, str) else note
        status = "INFO"
    return {"section": section, "metric": metric, "unit": unit, "neso": neso, "model": model, "difference": diff,
            "relative_difference": rel_diff, "judged_difference": judged, "basis": basis, "tolerance": tol,
            "status": status or "INFO", "neso_metric": neso_metric, "neso_source": source, "note": note}


def build_balancing_validation(measures: dict, bench: pd.DataFrame, cfg: dict | None = None,
                               delay_case: str | None = None) -> tuple[pd.DataFrame, dict]:
    """The validation rows and the score. ``bench``: :func:`read_benchmarks` (indexed by metric)."""
    cfg = balancing_config(cfg)

    def b(metric):
        return float(bench.at[metric, "value"])

    def src(metric):
        r = bench.loc[metric]
        return f"{r['sheet']}!{r['cell']}" if r["sheet"] not in ("main report", "NAO report", "derived") \
            else f"{r['sheet']}: {r['cell']}"

    central_info = (f"delay case {delay_case}: the recommended network's benchmark, shown for context"
                    if delay_case else False)
    rows = []
    v = {k: val / 1e6 for k, val in measures["volumes"].items()}
    v["offer_other_incl_storage"] = v.get("offer_other", 0.0) + v.get("offer_storage", 0.0)
    v["bid_other_incl_storage"] = v.get("bid_other", 0.0) + v.get("bid_storage", 0.0)
    v["offer_total"] = sum(v.get(f"offer_{g}", 0.0) for g in ("ic", "gas", "wind", "other", "storage"))
    v["bid_total"] = sum(v.get(f"bid_{g}", 0.0) for g in ("ic", "gas", "wind", "other", "storage"))
    tol, small = float(cfg["volume_warn_rel"]), float(cfg["small_volume_twh"])
    published = cfg["volume_basis"] == "published"
    for side in ("offer", "bid"):
        for g, model_key, label in (("ic", f"{side}_ic", "interconnectors"), ("gas", f"{side}_gas", "unabated gas incl. CHP"),
                                    ("wind", f"{side}_wind", "wind"),
                                    ("other", f"{side}_other_incl_storage", "other incl. storage"),
                                    ("total", f"{side}_total", "total")):
            metric = f"cp18_{side}_{g}_twh"
            rows.append(_row("volumes", f"{side}s: {label}", "TWh", b(metric), v.get(model_key, 0.0),
                             tol if published else None, small=small, neso_metric=metric, source=src(metric),
                             note="CP.18 as published (FFR, recommended network)", info=central_info))
        rows.append(_row("volumes", f"{side}s: storage (inside other)", "TWh", None, v.get(f"{side}_storage", 0.0),
                         note="CP.18 has no storage row"))
    for key, label in (("offer_ic", "offers: interconnectors"), ("bid_ic", "bids: interconnectors"),
                       ("offer_gas", "offers: unabated gas incl. CHP"), ("bid_wind", "bids: wind")):
        metric = f"es1_start_{key}_twh"
        rows.append(_row("volumes", f"{label} (ES1-start basis)", "TWh", b(metric), v.get(key, 0.0),
                         tol if not published else None, small=small, neso_metric=metric, source=src(metric),
                         note="CP.18 restated to an ES1 start (Run A basis)", info=central_info))
    for g, label in (("ic", "interconnectors"), ("gas", "unabated gas incl. CHP"), ("wind", "wind")):
        neso_net = b(f"cp18_offer_{g}_twh") - b(f"cp18_bid_{g}_twh")
        rows.append(_row("volumes", f"net (offers - bids): {label}", "TWh", neso_net,
                         v.get(f"offer_{g}", 0.0) - v.get(f"bid_{g}", 0.0), neso_metric=f"cp18_offer_{g}_twh - bid",
                         source=f"CP.18 offers - bids ({g})", note="context"))

    # Cost.
    cost = measures["neso_comparable_cost_GBP"] / 1e6
    neso_cost = b("neso_constraint_cost_gbpm")
    basis = measures.get("headline_basis", "cheapest_first")
    label = {"cheapest_first": "as solved", "pro_rata_zonal": "pro rata zonal turn-down"}
    rows.append(_row("cost", f"NESO-comparable redispatch cost (headline basis: {basis})", "GBP m", neso_cost, cost,
                     float(cfg["cost_warn_rel"]),
                     neso_metric="neso_constraint_cost_gbpm", source=src("neso_constraint_cost_gbpm"),
                     note=f"{measures.get('cost_source', '')}: ordinary redispatch + IC re-trading; VoLL slack and "
                          f"non-participant turn-down out; 2024 prices both; wind turn-down {label.get(basis, basis)} "
                          f"({measures.get('turn_down_basis_source', '')})", info=central_info))
    for other, value in (measures.get("cost_by_basis_GBP") or {}).items():
        if other != basis:
            rows.append(_row("cost", f"NESO-comparable redispatch cost ({other})", "GBP m", neso_cost, value / 1e6,
                             neso_metric="neso_constraint_cost_gbpm", source=src("neso_constraint_cost_gbpm"),
                             note=f"INFO: the other turn-down basis ({label.get(other, other)})"))
    acc = measures.get("accounting", {})
    for k, label in (("ordinary_redispatch_cost_excluding_feasibility_slack_GBP", "ordinary redispatch (units)"),
                     ("interconnector_retrading_cost_GBP", "interconnector re-trading"),
                     ("feasibility_slack_cost_GBP", "VoLL slack (not in the comparable cost)"),
                     ("non_participant_turn_down_cost_at_penalty_GBP", "non-participant turn-down at penalty (out)")):
        rows.append(_row("cost", label, "GBP m", None, acc.get(k, 0.0) / 1e6, note="sum of the market summaries"))
    rows.append(_row("cost", "load shed (VoLL slack energy)", "GWh", None,
                     acc.get("feasibility_slack_energy_MWh", 0.0) / 1e3, note="sum of the market summaries"))
    cbg = measures.get("cost_by_group")
    if isinstance(cbg, pd.DataFrame) and not cbg.empty:
        for group, r in cbg.sort_values("net_cost", ascending=False).iterrows():
            rows.append(_row("cost", f"net cost by group: {group}", "GBP m", None, r["net_cost"] / 1e6,
                             note=(f"offers {r['increase_MWh'] / 1e6:.2f} TWh for {r['offer_cost'] / 1e6:,.1f} GBP m; "
                                   f"bids {r['decrease_MWh'] / 1e6:.2f} TWh for {r['bid_cost'] / 1e6:,.1f} GBP m "
                                   "(constraint_costs.csv, TOTAL row dropped)")))

    # Final positions against N2.
    f = {k: val / 1e6 for k, val in measures["final"].items()}
    f["net_export"] = f.get("ic_export", 0.0) - f.get("ic_import", 0.0)
    ftol = float(cfg["final_warn_rel"])
    for key, metric, label in (("wind", "n2_wind_twh", "wind"), ("gas_incl_chp", "n2_gas_incl_chp_twh", "gas incl. CHP"),
                               ("unabated_gas", "n2_unabated_gas_twh", "unabated gas (CCGT + OCGT + engines)"),
                               ("ic_import", "n2_ic_import_twh", "interconnector imports"),
                               ("ic_export", "n2_ic_export_twh", "interconnector exports"),
                               ("net_export", "n2_net_export_twh", "net export"),
                               ("generation", "n2_generation_excl_storage_twh", "generation excl. storage")):
        rows.append(_row("final", label, "TWh", b(metric), f.get(key, 0.0), ftol, neso_metric=metric, source=src(metric),
                         note="balancing (final) dispatch against N2", info=central_info))
    share = f["unabated_gas"] / f["generation"] if f.get("generation") else np.nan
    rows.append(_row("final", "unabated gas share of generation", "share", b("n2_unabated_gas_share"), share,
                     float(cfg["gas_share_warn_abs"]), rel=False, neso_metric="n2_unabated_gas_share",
                     source=src("n2_unabated_gas_share"),
                     note="ours: CCGT + OCGT + gas engines over every generator except the foreign market, load "
                          "shedding and the churn sink", info=central_info))
    s1 = {k: val / 1e6 for k, val in measures.get("stage1", {}).items()}
    if s1:
        for key, metric in (("ic_import", "cp30_ic_import_twh"), ("ic_export", "cp30_ic_export_twh")):
            rows.append(_row("final", f"Stage 1 {key.replace('ic_', 'interconnector ')}s", "TWh", b(metric),
                             s1.get(key, 0.0), neso_metric=metric, source=src(metric),
                             note="context: Stage 1 against NESO's economic run (CP.30)"))

    # Zones against CP.08 (recommended network, FFR).
    zones = measures["zones"].reindex(list(ZONES)).fillna(0.0) / 1e6
    total = float(zones.sum())
    cp08 = pd.Series({z: b(f"cp08_recommended_ffr_{z.lower()}_twh") for z in ZONES})
    cp08_total = float(cp08.sum())
    ztol = float(cfg["zone_share_warn_abs"])
    for z in ZONES:
        metric = f"cp08_recommended_ffr_{z.lower()}_twh"
        rows.append(_row("zones", f"{z} share of clean turn-down", "share", cp08[z] / cp08_total,
                         zones[z] / total if total else np.nan, ztol, rel=False, neso_metric=metric,
                         source=src(metric), note=f"{zones[z]:.3f} TWh against CP.08 {cp08[z]:.3f} TWh",
                         info=central_info))
    scot, cp08_scot = float(zones[list(SCOTLAND)].sum()), float(cp08[list(SCOTLAND)].sum())
    rows.append(_row("zones", "Scotland (Z1-4) share of clean turn-down", "share", cp08_scot / cp08_total,
                     scot / total if total else np.nan, float(cfg["scotland_share_warn_rel"]),
                     neso_metric="cp08_recommended_ffr_scotland_twh", source=src("cp08_recommended_ffr_scotland_twh"),
                     note="relative tolerance", info=central_info))
    rows.append(_row("zones", "Scotland (Z1-4) clean turn-down", "TWh", cp08_scot, scot,
                     neso_metric="cp08_recommended_ffr_scotland_twh", source=src("cp08_recommended_ffr_scotland_twh"),
                     note="INFO: CP.08 exceeds every CP.18 bid (another run vintage); its shares are judged"))
    rows.append(_row("zones", "clean turn-down, all zones", "TWh", cp08_total, total,
                     neso_metric="cp08_recommended_ffr_total_twh", source=src("cp08_recommended_ffr_total_twh"),
                     note="INFO, as above"))
    rows.append(_row("zones", "clean turn-down not zoned (foreign or unknown bus)", "TWh", None,
                     measures.get("unzoned_clean_MWh", 0.0) / 1e6))

    # Central-case score.
    gaps = {"cost": cost / neso_cost - 1, "wind": f["wind"] / b("n2_wind_twh") - 1,
            "gas_incl_chp": f["gas_incl_chp"] / b("n2_gas_incl_chp_twh") - 1,
            "net_export": f["net_export"] / b("n2_net_export_twh") - 1}
    mean_gap = float(np.mean([abs(x) for x in gaps.values()]))
    eligible = abs(gaps["cost"]) <= float(cfg["central_cost_max_rel"])
    rows.append(_row("score", "mean |gap| over cost, final wind, gas incl. CHP, net export", "share", None, mean_gap,
                     note="pre-stated central-case score (18 Oct 2026); smaller is better; gaps "
                          + ", ".join(f"{k} {100 * x:+.1f}%" for k, x in gaps.items())))
    rows.append(_row("score", f"central-case eligible (cost within {100 * float(cfg['central_cost_max_rel']):g}%)",
                     "flag", 1.0, float(eligible), status="PASS" if eligible else "WARN",
                     note=f"cost gap {100 * gaps['cost']:+.1f}%", info=central_info))
    score = {"gaps": gaps, "mean_abs_gap": mean_gap, "central_eligible": bool(eligible)}

    # Delay families.
    if delay_case:
        metric, label = DELAY_CASES[delay_case]
        rows.append(_row("delay", f"NESO-comparable cost, {delay_case} ({label})", "GBP m", b(metric), cost,
                         float(cfg["delay_cost_warn_rel"]), neso_metric=metric, source=src(metric),
                         note=str(bench.at[metric, "note"])))
        rows.append(_row("delay", f"increase over the central network, {delay_case}", "GBP m", b(metric) - neso_cost,
                         cost - neso_cost, note="context: NESO's increase against ours over NESO's central 3,570"))
        if delay_case == "no_upgrades":
            cp08t = pd.Series({z: b(f"cp08_today_ffr_{z.lower()}_twh") for z in ZONES})
            rows.append(_row("delay", "clean turn-down, all zones (network we have today)", "TWh", float(cp08t.sum()),
                             total, float(cfg["volume_warn_rel"]), neso_metric="cp08_today_ffr_total_twh",
                             source=src("cp08_today_ffr_total_twh"), note="CP.08 'network capacity we have today'"))
            for z in ZONES:
                metric_z = f"cp08_today_ffr_{z.lower()}_twh"
                rows.append(_row("delay", f"{z} share of clean turn-down (today's network)", "share",
                                 cp08t[z] / cp08t.sum(), zones[z] / total if total else np.nan, ztol, rel=False,
                                 neso_metric=metric_z, source=src(metric_z),
                                 note=f"{zones[z]:.3f} TWh against CP.08 today {cp08t[z]:.3f} TWh"))
    table = pd.DataFrame(rows, columns=COLUMNS)
    if table["status"].eq("FAIL").any():
        raise AssertionError("report-only check produced a FAIL row")
    return table, score


def format_table(table: pd.DataFrame) -> str:
    view = table[["section", "metric", "unit", "neso", "model", "judged_difference", "status"]]
    return view.to_string(index=False, float_format=lambda x: f"{x:,.4g}" if abs(x) < 1 else f"{x:,.2f}")


def run(study: str, months: list[dict], annual_summary_path: str | None, output_csv: str, output_json: str,
        cfg: dict | None = None, secured: dict | None = None, network_delivery_case: str | None = None,
        log_path: str | None = None, turn_down_summary_path: str | None = None) -> pd.DataFrame:
    from scripts.utilities.logging_config import setup_logging

    logger = setup_logging(log_path or "validate_cp30_balancing")
    cfg = balancing_config(cfg)
    if len(months) != 12:
        logger.warning("%s: %d months, not twelve: the totals are not a calendar year", study, len(months))
    bench = read_benchmarks(cfg["benchmarks_file"])
    polys = load_zone_polygons(cfg["zones_file"])
    annual = (json.loads(Path(annual_summary_path).read_text(encoding="utf-8"))
              if annual_summary_path and Path(annual_summary_path).exists() else None)
    measures = measure_family(months, annual, polys, secured, cfg, logger)
    turn_down = (json.loads(Path(turn_down_summary_path).read_text(encoding="utf-8"))
                 if turn_down_summary_path and Path(turn_down_summary_path).exists() else None)
    apply_turn_down_basis(measures, turn_down)
    if abs(measures["hours"] - 8760.0) > 0.5 and abs(measures["hours"] - 8784.0) > 0.5:
        logger.warning("%s: the months cover %.0f h, not a calendar year", study, measures["hours"])
    delay = delay_case_of(network_delivery_case, study)
    table, score = build_balancing_validation(measures, bench, cfg, delay)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    summary = {"study": study, "months": [m["name"] for m in months], "hours": measures["hours"],
               "delay_case": delay, "network_delivery_case": network_delivery_case, "config": cfg,
               "status_counts": table["status"].value_counts().to_dict(), "score": score,
               "cost_source": measures["cost_source"], "zone_mapping": measures["zone_mapping_counts"],
               "headline_basis": measures["headline_basis"], "cost_by_basis_GBP": measures["cost_by_basis_GBP"],
               "turn_down_basis_source": measures["turn_down_basis_source"],
               "distinct_topologies": measures["distinct_topologies"],
               "inputs": {"months": months, "annual_summary": annual_summary_path,
                          "turn_down_summary": turn_down_summary_path},
               "rows": json.loads(table.to_json(orient="records"))}
    Path(output_json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("CP30 balancing validation, %s (%d months, report only):\n%s", study, len(months), format_table(table))
    for _, r in table[table["status"].eq("WARN")].iterrows():
        logger.warning("CP30 balancing WARN %s %s: model %.4g vs NESO %.4g %s (judged %+.3f)", r["section"],
                       r["metric"], r["model"], r["neso"], r["unit"], r["judged_difference"])
    return table


# ── Resolving a study outside Snakemake ─────────────────────────────────────
def study_months(study: str, resources: str = "resources") -> tuple[list[dict], dict]:
    """The months of a ``monthly_aggregates`` study (a study's study.yaml) with their input paths, and the first month's config."""
    import yaml

    sys.path.insert(0, str(PROJECT_ROOT / "config"))
    from config_loader import load_config

    full = load_config(PROJECT_ROOT / "config")
    scenarios = full["scenarios"]
    names = full["study_settings"]["monthly_aggregates"][study]["scenarios"]
    months = []
    for name in names:
        source = scenarios.get(name, {}).get("wholesale_from") or name
        months.append(month_paths(name, source, resources))
    return months, scenarios[names[0]]


def month_paths(name: str, stage1_source: str, resources: str = "resources") -> dict:
    return {"name": name, "network": f"{resources}/market/{name}_balancing.nc",
            "redispatch": f"{resources}/market/{name}_redispatch_summary.csv",
            "constraint_costs": f"{resources}/market/{name}_constraint_costs.csv",
            "stage1_links": f"{resources}/market/{stage1_source}_wholesale_links.csv",
            "market_summary": f"{resources}/analysis/{name}_market_summary.json"}


def scenario_settings(scenario: dict) -> dict:
    """What the check reads from a month's scenario config (the rule passes only this as params)."""
    balancing = ((scenario.get("market") or {}).get("balancing") or {})
    delivery = balancing.get("network_delivery") or {}
    secured = dict((((scenario.get("network") or {}).get("transmission") or {}).get("outage_schedule") or {})
                   .get("secured_capability") or {})
    return {"cp30_balancing_validation": (scenario.get("validation") or {}).get("balancing") or {},
            "network_delivery_case": delivery.get("case"),
            "secured_capability": {k: secured.get(k) for k in ("geometry_file", "include", "exclude") if secured.get(k)}}


if "snakemake" in globals():
    _s = dict(snakemake.params.settings)
    _months = [dict(zip(("name", "network", "redispatch", "constraint_costs", "stage1_links", "market_summary"), t))
               for t in zip(snakemake.params.months, snakemake.input.networks, snakemake.input.redispatch,
                            snakemake.input.constraint_costs, snakemake.input.stage1_links,
                            snakemake.input.market_summaries)]
    run(study=snakemake.wildcards.study, months=_months, annual_summary_path=snakemake.input.annual_summary,
        output_csv=snakemake.output.csv, output_json=snakemake.output.json,
        cfg=_s.get("cp30_balancing_validation"), secured=_s.get("secured_capability"),
        network_delivery_case=_s.get("network_delivery_case"),
        log_path=snakemake.log[0] if snakemake.log else None,
        turn_down_summary_path=getattr(snakemake.input, "turn_down_summary", None))
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", help="a monthly_aggregates study, e.g. CP30_2030_cp30_network")
    parser.add_argument("--out-dir", default="resources/analysis")
    parser.add_argument("--resources", default="resources")
    parser.add_argument("--turn-down-summary", default=None,
                        help="the family's pro rata turn-down summary (default <resources>/analysis/<study>_...)")
    parser.add_argument("--log", default=None, help="log file (default <out-dir>/<study>_cp30_balancing_validation.log)")
    parser.add_argument("--extract-benchmarks", nargs="?", const=BENCHMARKS_FILE, metavar="CSV",
                        help=f"re-read the CP30 workbook and write the benchmark file (default {BENCHMARKS_FILE})")
    args = parser.parse_args()
    if args.extract_benchmarks:
        out = extract_benchmarks(WORKBOOK)
        out.to_csv(args.extract_benchmarks, index=False)
        print(f"wrote {len(out)} benchmarks to {args.extract_benchmarks}")
    if args.study:
        _months, _scenario = study_months(args.study, args.resources)
        _s = scenario_settings(_scenario)
        out_dir = Path(args.out_dir)
        run(args.study, _months, f"{args.resources}/analysis/{args.study}_annual_summary.json",
            str(out_dir / f"{args.study}_cp30_balancing_validation.csv"),
            str(out_dir / f"{args.study}_cp30_balancing_validation.json"),
            _s["cp30_balancing_validation"], _s["secured_capability"], _s["network_delivery_case"],
            args.log or str(out_dir / f"{args.study}_cp30_balancing_validation.log"),
            args.turn_down_summary or f"{args.resources}/analysis/{args.study}_pro_rata_turn_down_summary.json")
