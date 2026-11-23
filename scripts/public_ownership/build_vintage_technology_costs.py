"""Build the 2025-2050 vintage technology cost table (FES 2050 plan, task Y2).

Run by hand, not by Snakemake::

    python scripts/public_ownership/build_vintage_technology_costs.py

It reads the staged source workbooks in ``data/fes_2050_sources/`` and the
2030 cost table ``data/public_ownership/financing/technology_costs.csv`` (read
only), and writes three files beside the latter:

* ``technology_costs_vintage.csv``: technology x build year 2025-2050, in real
  2024 GBP, with the columns of ``technology_costs.csv`` plus ``build_year``,
  the declared basis adjustment, the learning-rate sensitivity and provenance;
* ``fes2025_hurdle_rates.csv``: FES 2025 hurdle rates by year (the decision 19
  spread sensitivity) beside the DESNZ Annex A rates by commissioning year;
* ``technology_economic_lives.csv``: the economic life the vintage table uses
  and what each source says.

Method (plan task Y2, decisions 11 and 26):

* **Full DESNZ basis (decision 26).** For the technologies DESNZ Electricity
  Generation Costs 2025 Annex A covers, capex is pre-development + construction
  + infrastructure (which includes the grid connection; Annex A note 5) and
  fixed O&M includes insurance.  Connection and use-of-system charges are left
  out: the finance layer charges TNUoS on its own line (item D4(a)).  The tidal
  row is taken from Annex A.
* **Anchor and ratio.**  For each technology in ``technology_costs.csv`` the
  2030 anchor is the corrected 2030 value, and a later year is the anchor x
  the source's own year / 2030 ratio.  The declared adjustment is corrected
  minus the ``technology_costs.csv`` basis carried forward on the same source
  ratio, so the table's 2030 row minus the adjustment reproduces
  ``technology_costs.csv`` exactly.  Technologies absent from
  ``technology_costs.csv`` take the source's own level (adjustment zero), except
  the lithium battery durations, which are anchored on the four-hour row so the
  duration curve stays internally consistent.
* **Years.**  DESNZ publishes 2030, 2035, ..., 2050: annual values are linear
  between them.  Before 2030 a DESNZ technology follows the FES 2025 shape of
  the same technology.  FES is annual.  A value is held flat after the last
  source year (decision 11).
* **Price base.**  FES 2025 GBP to 2024 GBP by the FES EA.01 GDP deflator
  (2024 = 97.72656 on 2025 = 100, i.e. x 0.9773), the same GDP-deflator basis
  DESNZ uses for Annex A.  It is NOT the 1.0384 CPI-type factor the plan uses
  between the bill (2025) and finance (2024) price bases.
* **Learning sensitivity.**  CCC Seventh Carbon Budget Methodology Report
  (May 2025) Table 10.2 learning rates per doubling of global deployment.  No
  public global deployment path could be used (the IEA NZE 2023 series the CCC
  uses needs an IEA licence), so the rate is applied as its annual-decline
  equivalent at a declared one global doubling per decade after 2030, on top of
  the central trajectory.
"""

import argparse
import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "fes_2050_sources"
FINANCING = ROOT / "data" / "public_ownership" / "financing"
DEFAULT_ANNEX_A = SOURCES / "annexA_2025.xlsx"
DEFAULT_FES_INPUTS = SOURCES / "fes_econ_inputs.xlsx"
DEFAULT_TECHNOLOGY_COSTS = FINANCING / "technology_costs.csv"
#: 2030-only rows of ``technology_costs.csv`` (F27 items 1 and 5, 23-24 Oct 2026): the 2050 crosswalk
#: supplement keeps their carriers out of finance, so the vintage table does not carry them.  "Large hydro" (IRENA 2023) is 2030-only too: the 2050 supplement costs large_hydro on Hydropower.
TWENTY_THIRTY_ONLY = frozenset({"Gas CCUS - Net Zero Teesside", "Large hydro"})
DEFAULT_OUTPUT = FINANCING / "technology_costs_vintage.csv"
DEFAULT_HURDLE_OUTPUT = FINANCING / "fes2025_hurdle_rates.csv"
DEFAULT_LIVES_OUTPUT = FINANCING / "technology_economic_lives.csv"

YEARS = list(range(2025, 2051))
ANCHOR_YEAR = 2030
PRICE_YEAR = 2024
DESNZ_YEARS = (2030, 2035, 2040, 2045, 2050)

ANNEX_A_NAME = "DESNZ Electricity Generation Costs 2025 Annex A (updated 18 Mar 2026)"
ANNEX_A_SHEET = "Technical and Cost Assumptions"
FES_NAME = "NESO FES 2025 Economics Inputs Data Workbook (Dec 2025)"
DEA_NAME = ("Danish Energy Agency, Technology Data for Energy Storage (Nov 2018), "
            "Vanadium Redox Battery data sheet")
DEA_PAGE = "p.125 (chapter pp.113-146; data/fes_2050_sources/dea_storage_vanadium_redox_flow_battery.pdf)"

#: FES 2025 Data Workbook WS1, HT 2050: networked electrolysis 72.27 TWh H2 (row 466) from
#: 88.71 TWh of electricity (row 593); the same 81.5% holds in 2030 (3.12 / 3.83).
#: FES H.02 quotes capex per kW of hydrogen output (HHV), so per kW of electricity
#: input = per kW H2 x efficiency.
ELECTROLYSER_EFFICIENCY_HHV = 72.27 / 88.71
ELECTROLYSER_EFFICIENCY_SOURCE = ("FES 2025 Data Workbook V005 WS1 rows 466 and 593, HT 2050: "
                                  "72.27 TWh H2 / 88.71 TWh electricity = 0.8147 (HHV)")
#: DESNZ Hydrogen Production Costs 2021 p.14: 30-year lifetime for PEM electrolysis; p.19:
#: electrolysis variable OPEX is annuitised stack replacement (a time-based annual cost).
ELECTROLYSER_LIFE_YEARS = 30
#: No source life for salt caverns in FES; declared equal to the electrolysis technical life.
SALT_CAVERN_LIFE_YEARS = 30

#: CCC 7CB Methodology Report (May 2025) p.294 Table 10.2, central learning rate per doubling.
CCC_LEARNING_RATES = {"offshore": 0.12, "onshore": 0.07, "solar": 0.20, "battery": 0.16}
CCC_SOURCE = "CCC Seventh Carbon Budget Methodology Report (May 2025) p.294 Table 10.2"
#: Declared: one doubling of global deployment per decade after 2030.
LEARNING_DOUBLING_TIME_YEARS = 10.0
LEARNING_START_YEAR = 2030


# ── Source readers ───────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def read_fes_deflators(path: Path) -> dict:
    """GDP deflator and EUR/GBP rate from FES EA.01.

    Args:
        path: The FES 2025 Economics Inputs workbook.

    Returns:
        ``{"to_2024": float, "deflator": {year: index}, "eur_per_gbp": {year: rate}}``.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb["EA.01"].iter_rows(values_only=True, max_col=60))
    header = next(r for r in rows if r and "Variable" in r)
    years = {i: v for i, v in enumerate(header) if isinstance(v, int)}
    out = {"deflator": {}, "eur_per_gbp": {}}
    for row in rows:
        if row and row[2] == "GDP deflator":
            out["deflator"] = {years[i]: float(row[i]) for i in years if row[i] is not None}
        if row and row[2] == "Euro to GBP":
            out["eur_per_gbp"] = {years[i]: float(row[i]) for i in years if row[i] is not None}
    out["to_2024"] = out["deflator"][2024] / out["deflator"][2025]
    return out


@functools.lru_cache(maxsize=None)
def read_fes_series(path: Path) -> dict:
    """FES E.03, H.02 and ER.01 time series and the E.02 lives and H.01 static values.

    Args:
        path: The FES 2025 Economics Inputs workbook.

    Returns:
        ``{"series": {(sheet, variable, type): (row, {year: value})}, "lives": {type: (row, life)},
        "static": {variable: (row, value)}}``; values in FES 2025 GBP.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    series = {}
    for sheet in ("E.03", "H.02", "ER.01"):
        rows = list(wb[sheet].iter_rows(values_only=True, max_col=40))
        header = next(r for r in rows if r and 2025 in r and 2050 in r)
        years = {i: v for i, v in enumerate(header) if isinstance(v, int)}
        for number, row in enumerate(rows, start=1):
            if row is header or not row or row[3] is None or row[5] is None:
                continue
            values = {years[i]: float(row[i]) for i in years if isinstance(row[i], (int, float))}
            if values:
                series[(sheet, str(row[3]).strip(), str(row[5]).strip())] = (number, values)
    lives = {}
    for number, row in enumerate(wb["E.02"].iter_rows(values_only=True, max_col=6), start=1):
        if row and isinstance(row[2], (int, float)) and row[1]:
            lives[str(row[1]).strip()] = (number, float(row[2]))
    static = {}
    for number, row in enumerate(wb["H.01"].iter_rows(values_only=True, max_col=8), start=1):
        if row and row[2] and row[4] is not None:
            try:
                static[str(row[2]).strip()] = (number, float(row[4]))
            except (TypeError, ValueError):
                continue
    return {"series": series, "lives": lives, "static": static}


@functools.lru_cache(maxsize=None)
def read_annex_a(path: Path) -> dict:
    """Annex A technical and cost assumptions, one frame per technology block.

    Args:
        path: The DESNZ EGC 2025 Annex A workbook.

    Returns:
        ``{block name: DataFrame}`` indexed by commissioning year (2030..2050), with
        columns such as ``Pre-development Cost|Medium`` (GBP/kW), ``Infrastructure|Medium``
        (GBP thousand per plant), ``Fixed O&M`` and ``Insurance`` (GBP/MW/yr),
        ``Reference plant size`` (MW) and ``Hurdle Rate|Medium``; plus ``_rows``,
        the sheet row of each field.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[ANNEX_A_SHEET].iter_rows(values_only=True))
    labels, parent = [], None
    for row in rows:
        if row[0] is not None:
            parent = str(row[0]).strip()
        labels.append(parent if row[1] is None else f"{parent}|{str(row[1]).strip()}")
    names, years = rows[0], rows[1]
    blocks = {}
    starts = [c for c in range(3, len(names)) if names[c] is not None]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(names)
        name = " ".join(str(names[start]).split())
        columns = {}
        for c in range(start, end):
            year = int(str(years[c]).split("\n")[0])
            if year in columns.values():
                continue  # geothermal repeats 2035 (FOAK and NOAK); keep the first
            columns[c] = year
        data, field_rows = {}, {}
        for r in range(2, len(rows)):
            data[labels[r]] = {columns[c]: rows[r][c] for c in columns}
            field_rows[labels[r]] = r + 1
        frame = pd.DataFrame(data)
        frame.attrs["rows"] = field_rows
        frame.attrs["columns"] = (start + 1, end)
        blocks[name] = frame
    return blocks


def _num(value) -> float:
    """Annex A cell to float: 'N/A' and blanks are zero (Annex A uses N/A for 'not included')."""
    if value is None or (isinstance(value, str) and value.strip().upper() in {"N/A", ""}):
        return 0.0
    return float(value)


def _column_letter(index: int) -> str:
    return openpyxl.utils.get_column_letter(index)


# ── Year handling ────────────────────────────────────────────────────────────


def interpolate_years(points: dict, years=YEARS) -> pd.Series:
    """Linear interpolation between published years, held flat outside them.

    Args:
        points: ``{year: value}`` at the published years.
        years: The years to return.

    Returns:
        A float Series indexed by ``years``.
    """
    known = pd.Series({int(k): float(v) for k, v in points.items()}).sort_index()
    return pd.Series(np.interp(years, known.index, known.values), index=list(years))


def year_basis(published: set, years=YEARS, pre_shape: str | None = None) -> pd.Series:
    """How each year's value was obtained, for the ``year_basis`` column."""
    first, last = min(published), max(published)
    out = {}
    for year in years:
        if year in published:
            out[year] = "published"
        elif year > last:
            out[year] = "held_flat_after_last_source_year"
        elif year < first:
            out[year] = pre_shape or "held_back_before_first_source_year"
        else:
            out[year] = "interpolated"
    return pd.Series(out)


def fes_ratio(values: dict, years=YEARS, anchor=ANCHOR_YEAR) -> pd.Series:
    """The FES series divided by its own anchor-year value (flat where FES has no value)."""
    series = pd.Series(values).reindex(years)
    series = series.ffill().bfill()
    base = series.loc[anchor]
    if not base:
        return pd.Series(1.0, index=list(years))
    return series / base


# ── Technology specifications ────────────────────────────────────────────────


@dataclass
class Spec:
    """One vintage-table technology and where its numbers come from."""

    technology: str
    kind: str  # desnz | fes | fes_family | fes_h2 | salt | dea
    wacc_technology: str
    family: str
    desnz_block: str | None = None
    fes_capex: str | None = None
    fes_fom: str | None = None
    fes_vom: str | None = None
    fes_life: str | None = None
    fes_sheet: str = "E.03"
    duration_hours: float | None = None
    family_anchor: str | None = None
    fes_interp: tuple | None = None  # (lower type, upper type, weight on upper)
    learning: str | None = None
    status: str = "reviewed"
    notes: str = ""
    extra: dict = field(default_factory=dict)


CAPEX, FOM, VOM = "CapEx (£/kw)", "Fixed OpEx (£/kw/year)", "Variable OpEx (£/MWh)"

SPECS = [
    # DESNZ Annex A technologies (full DESNZ basis).
    Spec("Solar PV", "desnz", "Solar PV", "solar", desnz_block="Large Scale Solar PV",
         fes_capex="Solar PV", fes_fom="Solar PV", fes_life="Solar PV", learning="solar"),
    Spec("Onshore wind", "desnz", "Onshore wind", "onshore", desnz_block="Onshore Wind",
         fes_capex="Onshore wind", fes_fom="Onshore wind", fes_life="Onshore wind", learning="onshore"),
    Spec("Offshore wind", "desnz", "Offshore wind", "offshore", desnz_block="Fixed Offshore Wind",
         fes_capex="Offshore wind", fes_fom="Offshore wind", fes_life="Offshore wind - fixed", learning="offshore",
         notes="Annex A excludes offshore connection (OFTO, charged through TNUoS): infrastructure is N/A"),
    Spec("Floating offshore wind", "desnz", "Floating offshore wind", "offshore",
         desnz_block="Floating Offshore Wind", fes_capex="Floating offshore wind",
         fes_fom="Floating offshore wind", fes_life="Offshore wind - floating", learning="offshore",
         notes="Annex A 2030 is FOAK; insurance 0 in Annex A (not separated)"),
    Spec("Gas generation - unabated", "desnz", "Gas generation - unabated", "gas",
         desnz_block="Gas CCGT 30% Load Factor", fes_capex="Gas CCGT", fes_fom="Gas CCGT", fes_life="CCGT",
         notes="CCGT 30% load factor block, as technology_costs.csv"),
    # 15 Nov 2026: the peakers the 2030 purchase bought still run in 2040 (OCGT 2.5 GW, engines 0.5 GW).
    Spec("Gas OCGT", "desnz", "Gas generation - unabated", "gas", desnz_block="Gas OCGT 300MW 5% Load Factor",
         notes="300MW 5% load factor block, as technology_costs.csv (F27 item 5)"),
    Spec("Gas reciprocating engine", "desnz", "Gas generation - unabated", "gas",
         desnz_block="Gas Reciprocating Engine 5%", notes="5% load factor block, as technology_costs.csv (F27 item 5)"),
    Spec("Gas generation - CCUS - mature", "desnz", "Gas generation - CCUS - mature", "gas_ccs",
         desnz_block="Gas CCUS 900MW 88% Load Factor", fes_capex="CCS gas", fes_fom="CCS gas", fes_life="CCS gas",
         notes="Annex A note 9: only capex updated in 2025, other components from EGC 2020"),
    Spec("Hydrogen CCHT", "desnz", "Hydrogen CCHT/OCHT - mature", "hydrogen_power",
         desnz_block="CCHT 900MW 30% Load Factor", fes_capex="Hydrogen CCHT", fes_fom="Hydrogen CCHT", fes_life="CCGT hydrogen",
         notes="Annex A 2030 is FOAK on a 50% H2 blend; 100% H2 from 2035, so capex rises 2030 to 2035"),
    Spec("Hydrogen OCHT", "desnz", "Hydrogen CCHT/OCHT - mature", "hydrogen_power",
         desnz_block="OCHT 30% Load Factor", fes_capex="H2 OCGT", fes_fom="H2 OCGT", fes_life="Hydrogen peaking plants",
         notes="Annex A 2030 is FOAK"),
    Spec("Tidal stream", "desnz", "Tidal stream", "marine", desnz_block="Tidal Stream Energy",
         fes_capex="Tidal", fes_fom="Tidal",
         notes="Decision 26: taken from Annex A (2030 FOAK 230 + 4,200 = 4,430 GBP/kW)"),
    # User, 3 Oct 2026 (FES 2050 item 3): FES ES1's 11 MW of geothermal (Geothermal CHP) had no
    # cost row. Annex A's Deep Granite Geothermal (FOAK 2030 and 2035, NOAK 2035 on); FES has no
    # geothermal series, so years before 2030 hold the 2030 (FOAK) values flat, declared.
    Spec("Deep geothermal", "desnz", "Deep geothermal", "geothermal", desnz_block="Deep Granite Geothermal",
         status="provisional",
         notes="Annex A Deep Granite Geothermal; FOAK 2030 (2.2 MW reference plant) held flat before 2030 (no FES "
               "series); NOAK 15 MW plant from 2040. Electricity costs only: the CHP heat credit is not modelled"),
    # technology_costs.csv rows on an FES ratio (no DESNZ 2025 coverage).
    Spec("Large-scale nuclear", "fes", "Large-scale nuclear", "nuclear", fes_capex="Nuclear large",
         fes_fom="Nuclear large", fes_life="Nuclear large",
         notes="FES (EGC 2016) is flat, so the workbook anchor is held flat (decision 11)"),
    Spec("Hydropower", "fes", "Hydropower", "hydro", fes_capex="Hydro", fes_fom="Hydro",
         fes_vom="Hydro", fes_life="Small-scale hydro"),
    Spec("Biomass - unabated", "fes", "Biomass - unabated", "biomass", fes_capex="Biomass",
         fes_fom="Biomass", fes_vom="Biomass", fes_life="Biomass"),
    Spec("Biomass - with CCUS - mature", "fes", "Biomass - with CCUS - mature", "beccs",
         fes_sheet="ER.01", fes_capex="CCS biomass", fes_fom="CCS biomass", fes_vom="CCS biomass",
         fes_life="CCS biomass", notes="FES ER.01 power BECCS (EGC 2020 basis)"),
    Spec("Energy from waste", "fes", "Energy from waste", "waste", fes_capex="Waste", fes_fom="Waste",
         fes_vom="Waste", fes_life="Waste"),
    Spec("Lithium batteries", "fes", "Lithium batteries", "battery", duration_hours=4,
         fes_capex="Battery storage-4hr", fes_fom="Battery storage-4hr", fes_life="Battery 4hr",
         learning="battery"),
    Spec("Novel long duration energy storage (LDES)", "fes", "Novel long duration energy storage (LDES)",
         "ldes", duration_hours=4, fes_capex="LAES-6hr", fes_fom="LAES-6hr", fes_life="Liquid air 6h",
         status="provisional",
         notes="Continuity row: the 2030 thermal-storage proxy on the FES LAES 6 h trajectory; "
               "2050 LAES assets should take the LAES rows"),
    # New technologies at the source's own level.
    Spec("Small modular reactors (SMRs)", "fes", "Small modular reactors (SMRs)", "nuclear",
         fes_capex="Nuclear SMR", fes_fom="Nuclear SMR", fes_vom="Nuclear SMR", fes_life="Nuclear SMR",
         status="provisional",
         notes="FES literature review; vendor costs confidential (EGC 2025); no FES value before 2030"),
]
for hours in (1, 2, 6, 8, 9):
    interp = ("Battery storage-6hr", "Battery storage-9hr", 2.0 / 3.0) if hours == 8 else None
    fes_type = None if hours == 8 else f"Battery storage-{hours}hr"
    SPECS.append(Spec(
        f"Lithium batteries - {hours}h", "fes_family", "Lithium batteries", "battery",
        duration_hours=hours, fes_capex=fes_type, fes_fom=fes_type, fes_life="Battery 4hr",
        family_anchor="Lithium batteries", fes_interp=interp, learning="battery",
        notes=("Anchored on the Lithium batteries 4 h row x FES duration and year shape"
               + ("; 8 h interpolated linearly between FES 6 h and 9 h on duration" if hours == 8 else ""))))
for label, prefix, life_prefix, wacc, family, hours_list in (
        ("Liquid air energy storage", "LAES-", "Liquid air ", "Novel long duration energy storage (LDES)",
         "ldes", (6, 9, 12)),
        ("Compressed air energy storage", "CAES-", "Compressed air ", "Novel long duration energy storage (LDES)",
         "ldes", (6, 20, 72)),
        ("Pumped hydro energy storage (PHES)", "Hydro pumped storage-", "Pumped hydro ",
         "Pumped hydro energy storage (PHES)", "phes", (6, 20, 72))):
    for hours in hours_list:
        life_key = f"{life_prefix}{hours}h"
        if prefix.startswith("Hydro") and hours == 6:
            life_key = "Pumped hydro 20h"  # E.02 lists 20 h and 72 h only; all pumped hydro lives are 41
        SPECS.append(Spec(f"{label} - {hours}h", "fes", wacc, family, duration_hours=hours,
                          fes_capex=f"{prefix}{hours}hr", fes_fom=f"{prefix}{hours}hr", fes_life=life_key))
SPECS += [
    Spec("Hydrogen electrolyser", "fes_h2", "Hydrogen electrolyser", "hydrogen", fes_sheet="H.02",
         fes_capex="PEM", fes_fom="PEM"),
    Spec("Hydrogen storage - salt cavern", "salt", "", "hydrogen", status="provisional",
         notes="No workbook WACC category; FES costs hydrogen at the 3.5% social rate (EA.02)"),
    Spec("Vanadium redox flow battery", "dea", "Novel long duration energy storage (LDES)", "flow",
         status="provisional", notes="Not in FES 2025; DEA two-part cost, provisional"),
]


# ── Row builders ─────────────────────────────────────────────────────────────


def _fes(fes: dict, sheet: str, variable: str, fes_type: str) -> tuple[int, dict]:
    key = (sheet, variable, fes_type)
    if key not in fes["series"]:
        raise KeyError(f"FES series not found: {key}")
    return fes["series"][key]


def _fes_ref(sheet: str, row_numbers: list) -> str:
    rows = ", ".join(str(r) for r in row_numbers if r)
    return f"sheet {sheet} row(s) {rows}"


def build_desnz(spec: Spec, blocks: dict, fes: dict, to_2024: float, anchors: pd.DataFrame) -> pd.DataFrame:
    """A DESNZ Annex A technology on the full DESNZ basis, with the 2030 adjustment."""
    block = blocks[spec.desnz_block]
    rows = block.attrs["rows"]
    pts = {}
    for year in DESNZ_YEARS:
        size = _num(block.loc[year, "Reference plant size"])
        predev = _num(block.loc[year, "Pre-development Cost|Medium"])
        construction = _num(block.loc[year, "Construction Cost|Medium"])
        infra = _num(block.loc[year, "Infrastructure|Medium"]) / size if size else 0.0
        fixed = _num(block.loc[year, "Fixed O&M"]) / 1000.0
        insurance = _num(block.loc[year, "Insurance"]) / 1000.0
        pts[year] = {
            "excl_capex": predev + construction, "infra": infra, "full_capex": predev + construction + infra,
            "excl_fom": fixed, "insurance": insurance, "full_fom": fixed + insurance,
            "vom": _num(block.loc[year, "Variable O&M"]), "life": _num(block.loc[year, "Operating lifetime"]),
        }
    frame = pd.DataFrame({k: interpolate_years({y: v[k] for y, v in pts.items()}) for k in pts[2030]})
    # Before 2030: the FES 2025 shape of the same technology (capex and fixed O&M); flat at
    # the 2030 value for a technology FES does not cost (geothermal).
    if spec.fes_capex is None:
        capex_row = fom_row = None
        capex_shape = fom_shape = pd.Series(1.0, index=YEARS)
    else:
        capex_row, capex_series = _fes(fes, spec.fes_sheet, CAPEX, spec.fes_capex)
        fom_row, fom_series = _fes(fes, spec.fes_sheet, FOM, spec.fes_fom)
        capex_shape, fom_shape = fes_ratio(capex_series), fes_ratio(fom_series)
    pre = [y for y in YEARS if y < ANCHOR_YEAR]
    for column in ("excl_capex", "infra", "full_capex"):
        frame.loc[pre, column] = frame.loc[ANCHOR_YEAR, column] * capex_shape.loc[pre]
    for column in ("excl_fom", "insurance", "full_fom"):
        frame.loc[pre, column] = frame.loc[ANCHOR_YEAR, column] * fom_shape.loc[pre]
    first_col, last_col = block.attrs["columns"]
    page = (f"Annex A, sheet '{ANNEX_A_SHEET}', columns {_column_letter(first_col)}:{_column_letter(last_col - 1)} "
            f"('{spec.desnz_block}'), rows {rows['Pre-development Cost|Medium']} (pre-development, Medium), "
            f"{rows['Construction Cost|Medium']} (construction, Medium), {rows['Infrastructure|Medium']} "
            f"(infrastructure, Medium, / plant size row {rows['Reference plant size']}), {rows['Fixed O&M']} "
            f"(fixed O&M), {rows['Insurance']} (insurance), {rows['Variable O&M']} (variable O&M); "
            + (f"before 2030 FES {_fes_ref(spec.fes_sheet, [capex_row, fom_row])} shape" if capex_row is not None
               else "before 2030 the 2030 values held flat (no FES series)"))
    out = pd.DataFrame(index=YEARS)
    out["capex_power_gbp_per_kw"] = frame["full_capex"]
    out["infrastructure_capex_gbp_per_kw"] = frame["infra"]
    out["fixed_om_gbp_per_kw_year"] = frame["full_fom"]
    out["insurance_gbp_per_kw_year"] = frame["insurance"]
    out["variable_om_gbp_per_mwh"] = frame["vom"]
    out["source_level_capex_power_gbp_per_kw"] = frame["full_capex"]
    out["economic_life_years"] = frame["life"]
    basis = year_basis(set(DESNZ_YEARS), pre_shape="fes_shape_before_first_desnz_year")
    out["year_basis"] = basis
    out["source"] = f"{ANNEX_A_NAME}, Medium, full basis (pre-development + construction + infrastructure; fixed O&M + insurance)"
    out["source_page"] = page
    if spec.technology in anchors.index:
        anchor = anchors.loc[spec.technology]
        # technology_costs.csv basis carried forward on the source's own (excluded-basis) ratio.
        tc_capex = float(anchor["capex_power_gbp_per_kw"]) * frame["excl_capex"] / frame.loc[ANCHOR_YEAR, "excl_capex"]
        tc_fom = float(anchor["fixed_om_gbp_per_kw_year"]) * frame["excl_fom"] / frame.loc[ANCHOR_YEAR, "excl_fom"]
        tc_vom = pd.Series(float(anchor["variable_om_gbp_per_mwh"]), index=YEARS)
        out["basis_adjustment_capex_power_gbp_per_kw"] = out["capex_power_gbp_per_kw"] - tc_capex
        out["basis_adjustment_fixed_om_gbp_per_kw_year"] = out["fixed_om_gbp_per_kw_year"] - tc_fom
        out["basis_adjustment_variable_om_gbp_per_mwh"] = out["variable_om_gbp_per_mwh"] - tc_vom
        out["economic_life_years"] = float(anchor["economic_life_years"])
        parts = []
        if spec.technology == "Tidal stream":
            parts.append("tidal row replaced by Annex A (technology_costs.csv 5,598.98 does not reproduce Annex A)")
        if (frame["infra"] > 0).any():
            parts.append("capex adds Annex A infrastructure (includes grid connection)")
        if (frame["insurance"] > 0).any():
            parts.append("fixed O&M adds Annex A insurance")
        out["basis_adjustment_note"] = "Decision 26 full DESNZ basis: " + "; ".join(parts or ["no change"])
        out["anchor_rule"] = "anchor = corrected 2030 (full DESNZ basis); later years x Annex A year/2030 ratio"
    else:
        out["anchor_rule"] = "source level (not in technology_costs.csv)"
        out["basis_adjustment_note"] = "not in technology_costs.csv: source level on the full DESNZ basis"
    return out


def build_fes(spec: Spec, fes: dict, to_2024: float, anchors: pd.DataFrame, built: dict) -> pd.DataFrame:
    """An FES 2025 technology: ratio-anchored on technology_costs.csv or at the FES level."""
    def series(variable, fes_type):
        if spec.fes_interp and fes_type is None:
            low_type, high_type, weight = spec.fes_interp
            low_row, low = _fes(fes, spec.fes_sheet, variable, low_type)
            high_row, high = _fes(fes, spec.fes_sheet, variable, high_type)
            return [low_row, high_row], {y: low[y] + weight * (high[y] - low[y]) for y in low}
        row, values = _fes(fes, spec.fes_sheet, variable, fes_type)
        return [row], values

    capex_rows, capex = series(CAPEX, spec.fes_capex)
    fom_rows, fom = series(FOM, spec.fes_fom)
    vom_rows, vom = ([], {})
    if spec.fes_vom:
        vom_rows, vom = series(VOM, spec.fes_vom)
    capex_level = pd.Series(capex).reindex(YEARS).ffill().bfill() * to_2024
    fom_level = pd.Series(fom).reindex(YEARS).ffill().bfill() * to_2024
    vom_level = (pd.Series(vom).reindex(YEARS).ffill().bfill() * to_2024) if vom else pd.Series(0.0, index=YEARS)
    out = pd.DataFrame(index=YEARS)
    out["source_level_capex_power_gbp_per_kw"] = capex_level
    anchor_name = spec.family_anchor or (spec.technology if spec.technology in anchors.index else None)
    if anchor_name is not None:
        anchor = anchors.loc[anchor_name]
        if spec.family_anchor:
            base_row, base_capex = _fes(fes, spec.fes_sheet, CAPEX, anchors_fes_type(anchor_name))
            _, base_fom = _fes(fes, spec.fes_sheet, FOM, anchors_fes_type(anchor_name))
            capex_ratio = pd.Series(capex).reindex(YEARS).ffill().bfill() / base_capex[ANCHOR_YEAR]
            fom_ratio = pd.Series(fom).reindex(YEARS).ffill().bfill() / base_fom[ANCHOR_YEAR]
            capex_rows = capex_rows + [base_row]
        else:
            capex_ratio, fom_ratio = fes_ratio(capex), fes_ratio(fom)
        vom_ratio = fes_ratio(vom) if vom else pd.Series(1.0, index=YEARS)
        out["capex_power_gbp_per_kw"] = float(anchor["capex_power_gbp_per_kw"]) * capex_ratio
        out["fixed_om_gbp_per_kw_year"] = float(anchor["fixed_om_gbp_per_kw_year"]) * fom_ratio
        out["variable_om_gbp_per_mwh"] = float(anchor["variable_om_gbp_per_mwh"]) * vom_ratio
        out["economic_life_years"] = float(anchor["economic_life_years"])
        if spec.family_anchor:
            out["anchor_rule"] = (f"anchored on '{anchor_name}' 2030 x FES {spec.duration_hours:g} h / FES 4 h "
                                  "2030 ratio (duration and year shape)")
            out["basis_adjustment_note"] = "not in technology_costs.csv; anchored on the 4 h row (no basis change)"
        else:
            out["anchor_rule"] = "anchor = technology_costs.csv 2030 (no DESNZ basis to correct); later years x FES year/2030 ratio"
            out["basis_adjustment_note"] = "no adjustment: not a DESNZ Annex A technology"
        if spec.fes_life and spec.family_anchor:
            out["economic_life_years"] = fes["lives"][spec.fes_life][1]
    else:
        out["capex_power_gbp_per_kw"] = capex_level
        out["fixed_om_gbp_per_kw_year"] = fom_level
        out["variable_om_gbp_per_mwh"] = vom_level
        out["economic_life_years"] = fes["lives"][spec.fes_life][1]
        out["anchor_rule"] = "source level (not in technology_costs.csv), FES 2025 GBP x GDP deflator to 2024"
        out["basis_adjustment_note"] = "not in technology_costs.csv: FES 2025 level (FES capex includes infrastructure)"
    published = {y for y, v in capex.items() if v is not None}
    out["year_basis"] = year_basis(published)
    life_row = fes["lives"][spec.fes_life][0] if spec.fes_life else None
    out["source"] = f"{FES_NAME}, all pathways (base case)"
    out["source_page"] = (f"FES {_fes_ref(spec.fes_sheet, capex_rows)} (capex), {_fes_ref(spec.fes_sheet, fom_rows)} "
                          f"(fixed O&M)" + (f", {_fes_ref(spec.fes_sheet, vom_rows)} (variable O&M)" if vom_rows else "")
                          + (f", E.02 row {life_row} (life)" if life_row else ""))
    return out


_ANCHOR_FES_TYPES = {"Lithium batteries": "Battery storage-4hr"}


def anchors_fes_type(technology: str) -> str:
    """The FES series a family anchor row follows."""
    return _ANCHOR_FES_TYPES[technology]


def build_electrolyser(spec: Spec, fes: dict, to_2024: float) -> pd.DataFrame:
    """PEM electrolysis from FES H.02, converted from per kW H2 (HHV) to per kW electricity."""
    capex_row, capex = _fes(fes, "H.02", "CapEx (£/kW)", "PEM")
    fom_row, fom = _fes(fes, "H.02", "Fixed OpEx (£/kW)", "PEM")
    vom_row, vom = _fes(fes, "H.02", "Variable OpEx (£/kW)", "PEM")
    eta = ELECTROLYSER_EFFICIENCY_HHV
    level = pd.Series(capex).reindex(YEARS) * to_2024 * eta
    out = pd.DataFrame(index=YEARS)
    out["capex_power_gbp_per_kw"] = level
    out["source_level_capex_power_gbp_per_kw"] = level
    out["fixed_om_gbp_per_kw_year"] = (pd.Series(fom) + pd.Series(vom)).reindex(YEARS) * to_2024 * eta
    out["variable_om_gbp_per_mwh"] = 0.0
    out["economic_life_years"] = ELECTROLYSER_LIFE_YEARS
    out["year_basis"] = year_basis(set(capex))
    out["source"] = f"{FES_NAME}, PEM; life from DESNZ Hydrogen Production Costs 2021"
    out["source_page"] = (f"FES sheet H.02 rows {capex_row} (capex), {fom_row} (fixed O&M), {vom_row} (variable O&M, "
                          "folded into fixed O&M); per kW H2 (HHV) x efficiency; DESNZ HPC 2021 p.14 (30-year life), "
                          "p.19 (variable OPEX = annuitised stack replacement)")
    out["anchor_rule"] = "source level (not in technology_costs.csv), per kW electrical input"
    out["basis_adjustment_note"] = "not in technology_costs.csv: FES 2025 level"
    out["notes_extra"] = (f"Capex and O&M per kW electrical = FES per kW H2 HHV x {eta:.4f} ({ELECTROLYSER_EFFICIENCY_SOURCE}); "
                          "FES variable OpEx (0.9% of capex a year, H.01) is a per-kW annual cost and is folded into fixed O&M")
    return out


def build_salt_cavern(spec: Spec, fes: dict, to_2024: float) -> pd.DataFrame:
    """Salt-cavern hydrogen storage: a flat energy cost from FES H.01 (DNV 2019)."""
    capex_row, capex = fes["static"]["Salt cavern CapEx"]
    opex_row, opex_share = fes["static"]["Salt cavern OpEx"]
    out = pd.DataFrame(index=YEARS)
    out["capex_power_gbp_per_kw"] = 0.0
    out["source_level_capex_power_gbp_per_kw"] = 0.0
    out["capex_energy_gbp_per_kwh"] = capex * to_2024
    out["fixed_om_gbp_per_kw_year"] = 0.0
    out["fixed_om_gbp_per_kwh_year"] = capex * to_2024 * opex_share
    out["variable_om_gbp_per_mwh"] = 0.0
    out["economic_life_years"] = SALT_CAVERN_LIFE_YEARS
    out["year_basis"] = "held_flat_no_trajectory"
    out["source"] = f"{FES_NAME} H.01 (DNV 2019, Hydrogen in the electricity value chain, 2018 prices rebased by FES)"
    out["source_page"] = (f"FES sheet H.01 rows {capex_row} (capex per kWh working gas) and {opex_row} "
                          "(OpEx 4% of capex a year)")
    out["anchor_rule"] = "source level, flat (no source trajectory; decision 11)"
    out["basis_adjustment_note"] = "not in technology_costs.csv: FES 2025 level"
    out["notes_extra"] = (f"Per kWh of working gas (HHV); life {SALT_CAVERN_LIFE_YEARS} years is a declared assumption "
                          "(no FES value), equal to the electrolysis technical life")
    return out


def build_flow(spec: Spec, fes_meta: dict) -> pd.DataFrame:
    """Vanadium redox flow: DEA 2020/2030/2050 two-part cost, EUR2015 to GBP2024."""
    # DEA data sheet p.125, central values (EUR2015).
    energy = {2020: 0.45, 2030: 0.24, 2050: 0.22}  # MEUR/MWh, energy component
    other = {2020: 0.05, 2030: 0.03, 2050: 0.03}  # MEUR/MWh, other project costs (8% of capex)
    power = {2020: 0.41, 2030: 0.33, 2050: 0.33}  # MEUR/MW, capacity component
    fixed_share = {2020: 0.020, 2030: 0.015, 2050: 0.015}  # of total investment a year
    variable = {2020: 0.9, 2030: 0.9, 2050: 0.9}  # EUR2015/MWh
    rate = eur2015_to_gbp2024(fes_meta)
    years = list(range(2020, 2051))
    energy_s = interpolate_years({y: (energy[y] + other[y]) * 1000.0 * rate for y in energy}, years)
    power_s = interpolate_years({y: power[y] * 1000.0 * rate for y in power}, years)
    share_s = interpolate_years(fixed_share, years)
    out = pd.DataFrame(index=YEARS)
    out["capex_power_gbp_per_kw"] = power_s.loc[YEARS]
    out["source_level_capex_power_gbp_per_kw"] = power_s.loc[YEARS]
    out["capex_energy_gbp_per_kwh"] = energy_s.loc[YEARS]
    out["fixed_om_gbp_per_kw_year"] = (power_s * share_s).loc[YEARS]
    out["fixed_om_gbp_per_kwh_year"] = (energy_s * share_s).loc[YEARS]
    out["variable_om_gbp_per_mwh"] = variable[2030] * rate
    out["economic_life_years"] = 20.0
    out["year_basis"] = year_basis({2020, 2030, 2050}, years=years).loc[YEARS]
    out["source"] = DEA_NAME
    out["source_page"] = DEA_PAGE + "; central values, energy component + other project costs per MWh, capacity component per MW"
    out["anchor_rule"] = "source level (not in technology_costs.csv), two-part: power per kW + energy per kWh"
    out["basis_adjustment_note"] = "not in technology_costs.csv: DEA level (capacity component includes grid connection)"
    out["notes_extra"] = (f"EUR2015 to GBP2024 x {rate:.4f} (FES EA.01: 2015 EUR/GBP and GDP deflator); "
                          "duration left blank because the cost is two-part (energy capex per kWh)")
    return out


def eur2015_to_gbp2024(fes_meta: dict) -> float:
    """EUR 2015 prices to GBP 2024 prices: 2015 exchange rate, then the GBP GDP deflator."""
    return (1.0 / fes_meta["eur_per_gbp"][2015]) * fes_meta["deflator"][2024] / fes_meta["deflator"][2015]


# ── Assembly ─────────────────────────────────────────────────────────────────


OUTPUT_COLUMNS = [
    "technology", "build_year", "capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh", "duration_hours",
    "fixed_om_gbp_per_kw_year", "variable_om_gbp_per_mwh", "economic_life_years", "price_year",
    "real_or_nominal", "source", "source_page", "status",
    "wacc_technology", "family", "fixed_om_gbp_per_kwh_year",
    "infrastructure_capex_gbp_per_kw", "insurance_gbp_per_kw_year",
    "basis_adjustment_capex_power_gbp_per_kw", "basis_adjustment_capex_energy_gbp_per_kwh",
    "basis_adjustment_fixed_om_gbp_per_kw_year", "basis_adjustment_variable_om_gbp_per_mwh",
    "basis_adjustment_note", "source_level_capex_power_gbp_per_kw",
    "learning_rate_per_doubling", "learning_capex_gbp_per_kw", "learning_basis",
    "anchor_rule", "year_basis", "notes",
]
NUMERIC_DEFAULT_ZERO = [
    "capex_energy_gbp_per_kwh", "fixed_om_gbp_per_kwh_year", "infrastructure_capex_gbp_per_kw",
    "insurance_gbp_per_kw_year", "basis_adjustment_capex_power_gbp_per_kw",
    "basis_adjustment_capex_energy_gbp_per_kwh", "basis_adjustment_fixed_om_gbp_per_kw_year",
    "basis_adjustment_variable_om_gbp_per_mwh",
]


def learning_factor(rate: float, year: int) -> float:
    """Extra capex factor for the learning sensitivity: one global doubling per declared decade after 2030."""
    doublings = max(0.0, (year - LEARNING_START_YEAR) / LEARNING_DOUBLING_TIME_YEARS)
    return (1.0 - rate) ** doublings



def read_anchors(technology_costs: Path) -> pd.DataFrame:
    """``technology_costs.csv`` by technology, without its 2030-only rows."""
    anchors = pd.read_csv(technology_costs).set_index("technology")
    return anchors.drop(index=[t for t in TWENTY_THIRTY_ONLY if t in anchors.index])


def build_vintage_table(annex_a: Path = DEFAULT_ANNEX_A, fes_inputs: Path = DEFAULT_FES_INPUTS,
                        technology_costs: Path = DEFAULT_TECHNOLOGY_COSTS) -> pd.DataFrame:
    """The vintage table, technology x build year.

    Args:
        annex_a: DESNZ EGC 2025 Annex A workbook.
        fes_inputs: NESO FES 2025 Economics Inputs workbook.
        technology_costs: The 2030 cost table (read only; the anchors).

    Returns:
        A DataFrame with ``OUTPUT_COLUMNS``.
    """
    blocks = read_annex_a(annex_a)
    fes = read_fes_series(fes_inputs)
    fes_meta = read_fes_deflators(fes_inputs)
    to_2024 = fes_meta["to_2024"]
    if abs(to_2024 - 0.9773) > 5e-4:
        raise ValueError(f"FES EA.01 2025->2024 GDP deflator factor {to_2024:.5f} is not ~0.9773")
    anchors = read_anchors(technology_costs)
    frames, built = [], {}
    for spec in SPECS:
        if spec.kind == "desnz":
            frame = build_desnz(spec, blocks, fes, to_2024, anchors)
        elif spec.kind in {"fes", "fes_family"}:
            frame = build_fes(spec, fes, to_2024, anchors, built)
        elif spec.kind == "fes_h2":
            frame = build_electrolyser(spec, fes, to_2024)
        elif spec.kind == "salt":
            frame = build_salt_cavern(spec, fes, to_2024)
        elif spec.kind == "dea":
            frame = build_flow(spec, fes_meta)
        else:
            raise ValueError(spec.kind)
        frame = frame.copy()
        frame["technology"] = spec.technology
        frame["build_year"] = frame.index
        frame["duration_hours"] = spec.duration_hours if spec.duration_hours is not None else np.nan
        frame["wacc_technology"] = spec.wacc_technology
        frame["family"] = spec.family
        frame["status"] = spec.status
        frame["price_year"] = PRICE_YEAR
        frame["real_or_nominal"] = "real"
        for column in NUMERIC_DEFAULT_ZERO:
            if column not in frame.columns:
                frame[column] = 0.0
        if spec.technology in anchors.index and spec.kind != "desnz":
            anchor = anchors.loc[spec.technology]
            frame["capex_energy_gbp_per_kwh"] = float(anchor["capex_energy_gbp_per_kwh"])
        if spec.learning:
            rate = CCC_LEARNING_RATES[spec.learning]
            frame["learning_rate_per_doubling"] = rate
            frame["learning_capex_gbp_per_kw"] = [
                capex * learning_factor(rate, year)
                for capex, year in zip(frame["capex_power_gbp_per_kw"], frame["build_year"])]
            annual = 1.0 - (1.0 - rate) ** (1.0 / LEARNING_DOUBLING_TIME_YEARS)
            frame["learning_basis"] = (f"{CCC_SOURCE}: {rate:.0%} per doubling at a declared one global doubling "
                                       f"per {LEARNING_DOUBLING_TIME_YEARS:g} years after {LEARNING_START_YEAR}, "
                                       f"i.e. {annual:.2%} a year on top of the central trajectory")
        else:
            frame["learning_rate_per_doubling"] = np.nan
            frame["learning_capex_gbp_per_kw"] = frame["capex_power_gbp_per_kw"]
            frame["learning_basis"] = "no CCC Table 10.2 rate: learning column equals central"
        notes = [spec.notes] if spec.notes else []
        if "notes_extra" in frame.columns:
            notes.append(str(frame["notes_extra"].iloc[0]))
        frame["notes"] = "; ".join(n for n in notes if n)
        frames.append(frame[OUTPUT_COLUMNS])
        built[spec.technology] = frame
    table = pd.concat(frames, ignore_index=True)
    float_columns = [c for c in OUTPUT_COLUMNS if c not in {
        "technology", "build_year", "price_year", "real_or_nominal", "source", "source_page", "status",
        "wacc_technology", "family", "basis_adjustment_note", "learning_basis", "anchor_rule", "year_basis", "notes"}]
    table[float_columns] = table[float_columns].astype(float).round(6)
    # Exact parity: re-derive the stored 2030 adjustment from the rounded values.
    for tech in anchors.index:
        mask = table["technology"].eq(tech)
        for value_col, adj_col in (
                ("capex_power_gbp_per_kw", "basis_adjustment_capex_power_gbp_per_kw"),
                ("capex_energy_gbp_per_kwh", "basis_adjustment_capex_energy_gbp_per_kwh"),
                ("fixed_om_gbp_per_kw_year", "basis_adjustment_fixed_om_gbp_per_kw_year"),
                ("variable_om_gbp_per_mwh", "basis_adjustment_variable_om_gbp_per_mwh")):
            anchor_mask = mask & table["build_year"].eq(ANCHOR_YEAR)
            table.loc[anchor_mask, adj_col] = round(
                float(table.loc[anchor_mask, value_col].iloc[0]) - float(anchors.loc[tech, value_col]), 6)
    return table


# ── Hurdle rates and lives ───────────────────────────────────────────────────


#: FES E.03 / ER.01 type -> vintage-table technology.
FES_TO_VINTAGE = {
    "Solar PV": "Solar PV", "Onshore wind": "Onshore wind", "Offshore wind": "Offshore wind",
    "Floating offshore wind": "Floating offshore wind", "Gas CCGT": "Gas generation - unabated",
    "CCS gas": "Gas generation - CCUS - mature", "Hydrogen CCHT": "Hydrogen CCHT", "H2 OCGT": "Hydrogen OCHT",
    "Tidal": "Tidal stream", "Nuclear large": "Large-scale nuclear", "Nuclear SMR": "Small modular reactors (SMRs)",
    "Biomass": "Biomass - unabated", "CCS biomass": "Biomass - with CCUS - mature", "Waste": "Energy from waste",
    "Small-scale hydro": "Hydropower", "Battery storage-4hr": "Lithium batteries",
    "Battery storage-1hr": "Lithium batteries - 1h", "Battery storage-2hr": "Lithium batteries - 2h",
    "Battery storage-6hr": "Lithium batteries - 6h", "Battery storage-9hr": "Lithium batteries - 9h",
    "LAES-6hr": "Liquid air energy storage - 6h", "LAES-9hr": "Liquid air energy storage - 9h",
    "LAES-12hr": "Liquid air energy storage - 12h", "CAES-6hr": "Compressed air energy storage - 6h",
    "CAES-20hr": "Compressed air energy storage - 20h", "CAES-72hr": "Compressed air energy storage - 72h",
    "Hydro pumped storage-6hr": "Pumped hydro energy storage (PHES) - 6h",
    "Hydro pumped storage-20hr": "Pumped hydro energy storage (PHES) - 20h",
    "Hydro pumped storage-72hr": "Pumped hydro energy storage (PHES) - 72h",
}
DESNZ_TO_VINTAGE = {spec.desnz_block: spec.technology for spec in SPECS if spec.desnz_block}


def build_hurdle_table(annex_a: Path = DEFAULT_ANNEX_A, fes_inputs: Path = DEFAULT_FES_INPUTS) -> pd.DataFrame:
    """FES 2025 hurdle rates by year beside the DESNZ Annex A rates by commissioning year.

    Args:
        annex_a: DESNZ EGC 2025 Annex A workbook.
        fes_inputs: NESO FES 2025 Economics Inputs workbook.

    Returns:
        One row per source, source technology and case, with a ``hurdle_<year>`` column for
        each build year 2025-2050 (blank where the source publishes no value: DESNZ gives
        commissioning years 2030, 2035, ..., 2050 only).
    """
    fes = read_fes_series(fes_inputs)
    records = []
    for (sheet, variable, fes_type), (row, values) in sorted(fes["series"].items()):
        if not variable.startswith("Hurdle rate") or fes_type not in FES_TO_VINTAGE:
            continue
        record = {
            "source": f"{FES_NAME}, all pathways", "source_technology": fes_type,
            "technology": FES_TO_VINTAGE[fes_type], "case": "central", "basis": "pre-tax real",
            "source_page": f"FES sheet {sheet} row {row}",
            "role": ("decision 19 spread sensitivity (NESO 2035 wind step-down); "
                     "not used by the finance layer"),
            "status": "reviewed",
        }
        record.update({f"hurdle_{year}": values.get(year, np.nan) for year in YEARS})
        records.append(record)
    for name, frame in read_annex_a(annex_a).items():
        if name not in DESNZ_TO_VINTAGE:
            continue
        rows = frame.attrs["rows"]
        for case, label in (("central", "Medium"), ("low", "Low"), ("high", "High")):
            record = {
                "source": ANNEX_A_NAME, "source_technology": name, "technology": DESNZ_TO_VINTAGE[name],
                "case": case, "basis": "pre-tax real",
                "source_page": f"Annex A, sheet '{ANNEX_A_SHEET}', row {rows[f'Hurdle Rate|{label}']} ({label})",
                "role": ("reference: DESNZ/CEPA flat-by-vintage norm with FOAK/NOAK steps; "
                         "not used by the finance layer"),
                "status": "reviewed",
            }
            record.update({f"hurdle_{year}": (_num(frame.loc[year, f"Hurdle Rate|{label}"])
                                              if year in DESNZ_YEARS else np.nan) for year in YEARS})
            records.append(record)
    return pd.DataFrame(records)


def build_lives_table(vintage: pd.DataFrame, annex_a: Path = DEFAULT_ANNEX_A,
                      fes_inputs: Path = DEFAULT_FES_INPUTS,
                      technology_costs: Path = DEFAULT_TECHNOLOGY_COSTS) -> pd.DataFrame:
    """The economic life the vintage table uses, beside each source's value.

    Args:
        vintage: The vintage table from :func:`build_vintage_table`.
        annex_a: DESNZ EGC 2025 Annex A workbook.
        fes_inputs: NESO FES 2025 Economics Inputs workbook.
        technology_costs: The 2030 cost table.

    Returns:
        One row per technology.
    """
    fes = read_fes_series(fes_inputs)
    blocks = read_annex_a(annex_a)
    anchors = read_anchors(technology_costs)
    records = []
    for spec in SPECS:
        rows = vintage[vintage["technology"].eq(spec.technology)]
        record = {
            "technology": spec.technology,
            "economic_life_years": float(rows["economic_life_years"].iloc[0]),
            "technology_costs_csv_life": (float(anchors.loc[spec.technology, "economic_life_years"])
                                          if spec.technology in anchors.index else np.nan),
            "desnz_annex_a_life": (_num(blocks[spec.desnz_block].loc[2030, "Operating lifetime"])
                                   if spec.desnz_block else np.nan),
            "fes_e02_life": fes["lives"][spec.fes_life][1] if spec.fes_life else np.nan,
            "fes_e02_row": fes["lives"][spec.fes_life][0] if spec.fes_life else np.nan,
        }
        if spec.technology in anchors.index:
            record["rule"] = "technology_costs.csv life kept (2030 parity)"
        elif spec.desnz_block:
            record["rule"] = "DESNZ Annex A operating lifetime"
        elif spec.kind == "fes_h2":
            record["rule"] = "DESNZ Hydrogen Production Costs 2021 p.14 (PEM, 30 years)"
        elif spec.kind == "salt":
            record["rule"] = "declared assumption (no source value)"
        elif spec.kind == "dea":
            record["rule"] = "DEA data sheet p.125 technical lifetime"
        else:
            record["rule"] = "FES 2025 E.02 asset lifetime"
        records.append(record)
    return pd.DataFrame(records)


def main(argv=None) -> None:
    """Build and write the three tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annex-a", type=Path, default=DEFAULT_ANNEX_A)
    parser.add_argument("--fes-inputs", type=Path, default=DEFAULT_FES_INPUTS)
    parser.add_argument("--technology-costs", type=Path, default=DEFAULT_TECHNOLOGY_COSTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hurdle-output", type=Path, default=DEFAULT_HURDLE_OUTPUT)
    parser.add_argument("--lives-output", type=Path, default=DEFAULT_LIVES_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = build_vintage_table(args.annex_a, args.fes_inputs, args.technology_costs)
    table.to_csv(args.output, index=False)
    LOGGER.info("wrote %s (%d rows, %d technologies)", args.output, len(table), table["technology"].nunique())
    hurdles = build_hurdle_table(args.annex_a, args.fes_inputs)
    hurdles.to_csv(args.hurdle_output, index=False)
    LOGGER.info("wrote %s (%d rows)", args.hurdle_output, len(hurdles))
    lives = build_lives_table(table, args.annex_a, args.fes_inputs, args.technology_costs)
    lives.to_csv(args.lives_output, index=False)
    LOGGER.info("wrote %s (%d rows)", args.lives_output, len(lives))


if __name__ == "__main__":
    main()
