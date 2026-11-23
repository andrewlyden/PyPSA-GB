"""Export the financing inputs of the financing workbook (v6.2, ``GB_Power_Model_v6.2.xlsx``) to a YAML snapshot.

Schema 5 (30 Oct 2026, workbook v6.2, which keeps v2.0's layout: ``GB_Power_Model_Rebuild_v5.xlsx``
was schema 4).  v6.2 sets private gearing to a notional 55% (``wacc_by_basis.notional_gearing_shielded``),
reads the 15-year gilt directly, prices the state's systematic risk at a flat 1.9% (the
``economic_rate_by_premium`` keys follow the ``systematic_premium`` lever), values the gas cohort
from an asset register and reprices the marketable gilt stock (the contested gilt-stock externality's
inputs, ``assumptions.gilt_*``).

v2.0 replaced v11-2's single ``WACC Calculator`` with one tab
per scenario (Sc1 private, Sc2 30% state equity plus a 20% guarantee, Sc3/Sc4
public new build without/with the existing fleet) and a hand-off tab for PyPSA,
``WACC Scenarios``.  openpyxl cannot calculate, so this reads the cached results
of the workbook as saved, plus the workbook's own alternative cases recalculated
in Excel by ``workbook_variants.py`` (a committed CSV whose ``base`` variant must
reproduce the cached values, and whose JSON must name this workbook's SHA256).

Rates, per technology (``WACC Scenarios`` A7:J39, real CPI):

* ``wacc.private``                 Sc1 lead, tax-shielded (column D)
* ``wacc.public_coordination``     Sc2 lead, tax-shielded (F): P1
* ``wacc.public_ownership_new_build`` and ``wacc.public_ownership_legacy``:
  the Sc3/Sc4 cash rate (H), real gilt plus the NLF margin, what bills pay;
  the workbook carries no construction premium
* ``wacc.public_ownership_economic`` / ``..._operating_economic``: cash plus
  the systematic premium (I, J; v6.2 central a flat 1.9%), the social cost of the
  systematic risk taxpayers hold; for the resource account, never for bills
* ``corporation_tax_wedge``: the tax content of the lead rate (Sc4 Operative H for
  Sc1; the same construction on the Sc2 tab for P1)
* ``legacy_owner_rate``: the post-tax operating-asset rate the existing fleet is
  valued at, the workbook's ``Legacy Fleet`` column AF formula
  (calibrated offshore rate less the Sc1 vanilla hurdle spread to offshore)

Scalars come from ``Assumptions``, ``Sc2/Sc4 Operative``, ``Build-Out`` and
``Legacy Fleet``; the gilt-yield shifts by year from ``Gilt-Adjusted WACCs``
(Sc2 and Sc4 as saved, Sc3 from the ``fiscal_sc3`` variant), written as the
``sovereign_yield_shift_<year>`` blocks the finance layer reads; the legacy
cohort table (capacity, dates, historic capex, accounting life, values by basis)
from the three legacy tabs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Mapping

import openpyxl
import yaml

SCHEMA_VERSION = 5
WS, SC1, SC2, SC4 = "WACC Scenarios", "Sc1 Operative", "Sc2 Operative", "Sc4 Operative"
ASSUMPTIONS, BUILD_OUT, GILT = "Assumptions", "Build-Out", "Gilt-Adjusted WACCs"
LEGACY, LEGACY_A1, LEGACY_A2 = "Legacy Fleet", "Legacy A1 Book Value", "Legacy A2 Market DCF"
#: First technology row of each per-technology tab (33 rows, the same order; checked on read).
FIRST_ROW = {WS: 7, SC1: 15, SC2: 14, SC4: 15}
N_TECHNOLOGIES = 33
#: WACC Scenarios columns.
WS_COLUMNS = {"risk_rating": "B", "asset_beta": "C", "sc1_lead": "D", "sc1_vanilla": "E", "sc2_lead": "F",
              "sc2_vanilla": "G", "cash": "H", "economic": "I", "operating_economic": "J"}
SCALAR_CELLS = {
    # The state's cash rate (real gilt + public lending margin): bills, new build and acquisitions.
    "public_borrowing_real": (SC4, "F11"),
    "public_lending_margin": (SC4, "B11"),
    # The social premium: flat in v6.2 (Sc4 Operative B10 "Flat premium (no beta)"), per unit of beta otherwise.
    "social_premium": (SC4, "J11"),
    "risk_free_real": (ASSUMPTIONS, "D63"),
    "risk_free_nominal": (ASSUMPTIONS, "D62"),
    "expected_cpi": (ASSUMPTIONS, "D60"),
    "equity_risk_premium": (ASSUMPTIONS, "B64"),
    "total_market_return": (ASSUMPTIONS, "B43"),
    "corporation_tax_rate": (ASSUMPTIONS, "B45"),
    "debt_transaction_costs": (ASSUMPTIONS, "B46"),
    "public_coordination_state_equity_share": (ASSUMPTIONS, "B22"),
    "public_coordination_notches": (ASSUMPTIONS, "B20"),
    "public_coordination_retained_premium": (ASSUMPTIONS, "B23"),
    "public_coordination_guarantee_floor": (ASSUMPTIONS, "B25"),
    "public_coordination_guarantee_cover": (BUILD_OUT, "C181"),
    "optimism_bias_capex_uplift": (BUILD_OUT, "C180"),
    "yield_response_bp_per_pp": (BUILD_OUT, "C19"),
    "gas_fuel_cost_gbp_per_mwh": (LEGACY, "D89"),
    "legacy_owner_rate_calibrated": (LEGACY, "F21"),
    "legacy_owner_rate_pivot": (LEGACY, "D63"),
    "legacy_capital_allowance_rate": (LEGACY, "D93"),
    "legacy_reference_price_gbp_per_mwh": (LEGACY, "D80"),
    "legacy_acquisition_year": (LEGACY, "C7"),
    "legacy_gas_register_value_gbp_bn": (LEGACY, "D99"),
    # The contested gilt-stock externality (Build-Out section F): the yield response above, the
    # stress-case loss on the public asset base, the offset, the marketable stock and its maturity.
    "gdp_2024_gbp_bn": (BUILD_OUT, "C17"),
    "real_gdp_growth": (BUILD_OUT, "C18"),
    "gilt_stress_case_loss_share": (BUILD_OUT, "C174"),
    "gilt_counterfactual_offset_share": (BUILD_OUT, "C177"),
    "gilt_marketable_stock_2024_gbp_bn": (BUILD_OUT, "C178"),
    "gilt_average_maturity_years": (BUILD_OUT, "C179"),
}
#: The active switch settings (text), for provenance.
SWITCH_CELLS = {
    "gearing_basis": (ASSUMPTIONS, "B6"), "risk_free_vintage": (ASSUMPTIONS, "B8"),
    "inflation_vintage": (ASSUMPTIONS, "B9"), "erp_convention": (ASSUMPTIONS, "B10"),
    "risk_free_term_method": (ASSUMPTIONS, "B26"),
    "debt_beta_basis": (ASSUMPTIONS, "B11"), "cost_of_debt_basis": (ASSUMPTIONS, "B12"),
    "asset_beta_case": (ASSUMPTIONS, "B13"), "credit_rating_case": (ASSUMPTIONS, "B14"),
    "cfd_term_deduction": (ASSUMPTIONS, "B16"), "merchant_uplift": (ASSUMPTIONS, "B17"),
    "tax_shield": (ASSUMPTIONS, "B18"), "sc2_debt_mechanism": (ASSUMPTIONS, "B19"),
    "sc2_equity_treatment": (ASSUMPTIONS, "B21"), "public_lending_margin_basis": (SC4, "B5"),
    "systematic_premium_basis": (SC4, "B6"), "operating_asset_beta": (SC4, "B8"),
    "socialised_risk_holder": (SC4, "B9"), "social_premium_beta_basis": (SC4, "B10"),
    "fiscal_path_scenario": (BUILD_OUT, "C216"), "under_construction_capital": (BUILD_OUT, "C200"),
    "legacy_valuation_basis": (LEGACY, "C20"), "legacy_gas_valuation_basis": (LEGACY, "D98"),
    "gilt_debt_channel": (BUILD_OUT, "C175"), "gilt_contingent_exposure_channel": (BUILD_OUT, "C176"),
    "gilt_yield_response_basis": (BUILD_OUT, "C182"),
}
#: Variant -> the scalar it moves (cash rate under another lending margin).
MARGIN_VARIANTS = {"nlf": "base", "pwlb_hra": "margin_pwlb_hra", "pwlb_certainty": "margin_pwlb_certainty",
                   "pwlb_standard": "margin_pwlb_standard", "gilt_flat": "margin_gilt_flat"}
#: ``systematic_premium`` lever value -> the variant whose economic rate it reads (v6.2: a flat 1.9% is the base).
PREMIUM_VARIANTS = {"social_flat_19": "base", "none": "premium_none", "social_flat_10": "premium_flat_10",
                    "social_beta_20": "premium_beta_20", "market_erp": "premium_market_erp_beta"}
#: Gilt shift rows by year (bp): Sc2 B on row 7, the selected fiscal scenario's B on row 8.
GILT_YEAR_ROW, GILT_SC2_ROW, GILT_SC4_ROW = 5, 7, 8
GILT_FIRST_COLUMN, GILT_LAST_COLUMN = 4, 29  # D..AC = 2025..2050
#: Technologies workbook v2.0 renamed, written under the names the model's crosswalks and cost tables
#: key on (v11-2's, dashes normalised); the workbook's own name is kept as ``workbook_technology``.
MODEL_TECHNOLOGY_NAMES = {
    "Gas generation - with CCUS - mature": "Gas generation - CCUS - mature",
    "Gas generation - with CCUS - maturing": "Gas generation - CCUS - maturing",
    "Hydrogen CCHT / OCHT - mature": "Hydrogen CCHT/OCHT - mature",
    "Hydrogen CCHT / OCHT - emerging": "Hydrogen CCHT/OCHT - emerging",
}
#: The real CPI cost of debt by credit rating: Assumptions C5's extended ladder (rows 120-128, column G),
#: CEPA's iBoxx promised-yield route at the operative vintage (C3) plus the two rungs above BBB+.
DEBT_LADDER_ROWS, DEBT_LADDER_COLUMN = range(120, 129), "G"
LEGACY_ROWS = range(24, 39)  # Legacy Fleet cohorts 1-15; A1/A2 rows are these less 19
#: Legacy Fleet / A1 / A2 cohort columns.
LEGACY_COLUMNS = {"cohort": "B", "technology": "C", "support_regime": "D", "capacity_mw": "E", "cod_year": "F",
                  "design_life_years": "G", "closure_override_year": "H", "end_of_life_year": "I",
                  "load_factor": "K", "cfd_strike_2012_gbp_per_mwh": "O", "roc_band": "P", "capture_factor": "Q",
                  "owner_rate": "AH", "policy_cost_class": "AK"}
LEGACY_A1_COLUMNS = {"historic_capex_gbp_per_kw": "G", "accounting_life_years": "H", "net_book_value_gbp_m": "K",
                     "debt_assumed_gbp_m": "O"}
LEGACY_A2_COLUMNS = {"market_value_gbp_m": "V", "market_value_pre_tax_gbp_m": "U", "no_support_value_gbp_m": "Z",
                     "ro_pass_through_value_gbp_m": "AM", "capital_allowance_pv_factor": "AN"}
LEGACY_TOTAL_CELLS = {"book_value": ("E53", "C53", "D53"), "market_value": ("E54", "C54", "D54"),
                      "no_support": ("E55", "C55", "D55"), "ro_pass_through": ("E56", "C56", "D56")}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _formula_text(formula):
    if not isinstance(formula, (str, int, float, bool, type(None))):
        return getattr(formula, "text", str(formula))
    return formula


class _Book:
    """Cached values and formulas side by side."""

    def __init__(self, path: Path):
        self.values = openpyxl.load_workbook(path, data_only=True)
        self.formulas = openpyxl.load_workbook(path, data_only=False)

    def value(self, sheet: str, cell: str):
        return self.values[sheet][cell].value

    def cell(self, sheet: str, cell: str) -> dict:
        value = self.value(sheet, cell)
        if value is None:
            raise ValueError(f"{sheet}!{cell} has no cached value; open the workbook in Excel and save it")
        return {"value": value, "source_cell": f"{sheet}!{cell}",
                "formula": _formula_text(self.formulas[sheet][cell].value)}


def load_variants(csv_path: Path, workbook_sha256: str) -> dict[str, dict[tuple[str, str], object]]:
    """``{variant: {(sheet, cell): value}}`` from ``workbook_variants.py``'s output, checked against the workbook."""
    meta = json.loads(Path(csv_path).with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("workbook_sha256") != workbook_sha256:
        raise ValueError(f"{csv_path} was computed on another workbook (SHA256 {meta.get('workbook_sha256')}); "
                         "re-run scripts/public_ownership/workbook_variants.py")
    out: dict[str, dict[tuple[str, str], object]] = {}
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row["value"]
            try:
                value = float(value)
            except ValueError:
                pass
            out.setdefault(row["variant"], {})[(row["sheet"], row["cell"])] = value
    return out


def check_base_variant(book: _Book, variants: Mapping) -> int:
    """The Excel-recalculated ``base`` variant must equal the saved workbook's cached values."""
    base = variants.get("base") or {}
    if not base:
        raise ValueError("the variants file has no 'base' variant")
    bad = []
    for (sheet, cell), value in base.items():
        cached = book.value(sheet, cell)
        if isinstance(value, float):
            if not isinstance(cached, (int, float)) or abs(float(cached) - value) > 1e-9 * max(1.0, abs(value)):
                bad.append(f"{sheet}!{cell}: cached {cached!r}, recalculated {value!r}")
        elif str(cached) != str(value):
            bad.append(f"{sheet}!{cell}: cached {cached!r}, recalculated {value!r}")
    if bad:
        raise ValueError("the saved workbook's cached values differ from Excel's recalculation (save it from Excel "
                         "after a full calculation): " + "; ".join(bad[:10]))
    return len(base)


def _technology_names(book: _Book) -> list[str]:
    names = {}
    for sheet, first in FIRST_ROW.items():
        names[sheet] = [str(book.value(sheet, f"A{first + i}")).strip() for i in range(N_TECHNOLOGIES)]
    reference = names[WS]
    for sheet, listed in names.items():
        if listed != reference:
            raise ValueError(f"{sheet} lists the technologies in another order than {WS}: {listed} vs {reference}")
    return reference


def _technology_rows(book: _Book, variants: Mapping) -> list[dict]:
    a = {k: float(book.value(*c)) for k, c in SCALAR_CELLS.items() if k in
         ("legacy_owner_rate_calibrated", "legacy_owner_rate_pivot")}
    rows = []
    for i, name in enumerate(_technology_names(book)):
        r_ws, r1, r2, r4 = (FIRST_ROW[s] + i for s in (WS, SC1, SC2, SC4))
        ws = {key: book.value(WS, f"{col}{r_ws}") for key, col in WS_COLUMNS.items()}
        g = float(book.value(SC1, f"D{r1}"))
        sc1_vanilla = float(book.value(SC1, f"Q{r1}"))
        # Sc2 tax content, built as Sc4 Operative H builds Sc1's: (1 - g)(pre-tax - post-tax CoE) - shield.
        sc2_wedge = (1.0 - g) * (float(book.value(SC2, f"O{r2}")) - float(book.value(SC2, f"N{r2}"))) \
            - float(book.value(SC2, f"U{r2}"))
        cells = {k: f"{WS}!{WS_COLUMNS[c]}{r_ws}" for k, c in
                 (("private", "sc1_lead"), ("public_coordination", "sc2_lead"), ("public_ownership_new_build", "cash"),
                  ("public_ownership_legacy", "cash"), ("public_ownership_economic", "economic"),
                  ("public_ownership_operating_economic", "operating_economic"))}
        tmr = variants["erp_fixed_tmr"]
        rows.append({
            "technology": MODEL_TECHNOLOGY_NAMES.get(name, name),
            "workbook_technology": name,
            "risk_rating": ws["risk_rating"],
            "asset_beta": float(ws["asset_beta"]),
            "equity_beta": float(book.value(SC1, f"G{r1}")),
            "debt_beta": float(book.value(SC1, f"F{r1}")),
            "credit_rating": book.value(SC1, f"L{r1}"),
            "credit_rating_public_coordination": book.value(SC2, f"H{r2}"),
            "cost_of_debt_real": float(book.value(SC1, f"M{r1}")),
            "cost_of_debt_real_public_coordination": float(book.value(SC2, f"K{r2}")),
            "cost_of_equity_post_tax": float(book.value(SC1, f"J{r1}")),
            "cost_of_equity_pre_tax": float(book.value(SC1, f"K{r1}")),
            "post_tax_wacc": float(book.value(SC1, f"T{r1}")),
            "merchant_uplift": float(book.value(SC1, f"O{r1}")),
            # Stored as a positive deduction, as in schema 3 (the workbook's column P is negative).
            "cfd_term_deduction": -float(book.value(SC1, f"P{r1}")),
            "interest_tax_shield": float(book.value(SC1, f"R{r1}")),
            "gearing": book.cell(SC1, f"D{r1}"),
            "wacc": {
                "private": float(ws["sc1_lead"]),
                "public_coordination": float(ws["sc2_lead"]),
                "public_ownership_new_build": float(ws["cash"]),
                "public_ownership_legacy": float(ws["cash"]),
                "public_ownership_economic": float(ws["economic"]),
                "public_ownership_operating_economic": float(ws["operating_economic"]),
            },
            "wacc_source_cells": cells,
            "wacc_by_basis": {
                "notional_gearing_shielded": {"private": float(ws["sc1_lead"]),
                                              "public_coordination": float(ws["sc2_lead"])},
                "fixed_tmr_shielded": {"private": float(tmr[(WS, f"D{r_ws}")]),
                                       "public_coordination": float(tmr[(WS, f"F{r_ws}")])},
                "vanilla": {"private": float(ws["sc1_vanilla"]), "public_coordination": float(ws["sc2_vanilla"])},
            },
            "wacc_by_basis_source_cells": {
                "notional_gearing_shielded": {"private": cells["private"], "public_coordination": cells["public_coordination"]},
                "fixed_tmr_shielded": {"private": f"{WS}!D{r_ws} (variant erp_fixed_tmr)",
                                       "public_coordination": f"{WS}!F{r_ws} (variant erp_fixed_tmr)"},
                "vanilla": {"private": f"{WS}!E{r_ws}", "public_coordination": f"{WS}!G{r_ws}"},
            },
            "economic_rate_by_premium": {key: float(variants[v][(WS, f"I{r_ws}")]) for key, v in PREMIUM_VARIANTS.items()},
            "operating_economic_rate_by_premium": {key: float(variants[v][(WS, f"J{r_ws}")])
                                                   for key, v in PREMIUM_VARIANTS.items()},
            "systematic_premium": float(book.value(SC4, f"C{r4}")),
            "resource_rate_private": float(ws["sc1_lead"]) - float(book.value(SC4, f"H{r4}")),
            "corporation_tax_wedge": {"private": float(book.value(SC4, f"H{r4}")), "public_coordination": sc2_wedge},
            "corporation_tax_wedge_source": {"private": f"{SC4}!H{r4}",
                                             "public_coordination": f"(1 - {SC1}!D{r1}) x ({SC2}!O{r2} - {SC2}!N{r2}) - {SC2}!U{r2}"},
            "sc1_vanilla_hurdle": sc1_vanilla,
            # Legacy Fleet!AF: r_p = calibrated offshore rate - (offshore Sc1 vanilla - this technology's).
            "legacy_owner_rate": a["legacy_owner_rate_calibrated"] - (a["legacy_owner_rate_pivot"] - sc1_vanilla),
            "legacy_owner_rate_source": (f"{LEGACY}!F21 - ({LEGACY}!D63 - {SC1}!Q{r1}), the {LEGACY}!AF formula "
                                         "applied to this technology"),
            "source_rows": {"wacc": f"{WS}!A{r_ws}:J{r_ws}", "sc1": f"{SC1}!A{r1}:T{r1}", "sc2": f"{SC2}!A{r2}:V{r2}",
                            "sc4": f"{SC4}!A{r4}:H{r4}"},
        })
    return rows


def _yield_shifts(book: _Book, variants: Mapping) -> dict[int, dict]:
    """``{year: {Sc2: {B: cell}, Sc3: {B: cell}, Sc4: {B: cell}}}`` in rate units (the workbook is in bp)."""
    selected = str(book.value(BUILD_OUT, "C216"))
    if not selected.startswith("Sc4"):
        raise ValueError(f"{BUILD_OUT}!C216 is {selected!r}; the saved workbook must carry the Sc4 fiscal path")
    sc3 = variants["fiscal_sc3"]
    out = {}
    for col in range(GILT_FIRST_COLUMN, GILT_LAST_COLUMN + 1):
        letter = openpyxl.utils.get_column_letter(col)
        year = int(book.value(GILT, f"{letter}{GILT_YEAR_ROW}"))
        note = "gilt-yield shift in the investment year (bp / 10,000); applied additively, locked in for the asset's life"
        out[year] = {
            "Sc2": {"B": {"value": float(book.value(GILT, f"{letter}{GILT_SC2_ROW}")) / 1e4,
                          "source_cell": f"{GILT}!{letter}{GILT_SC2_ROW}", "note": note}},
            "Sc3": {"B": {"value": float(sc3[(GILT, f"{letter}{GILT_SC4_ROW}")]) / 1e4,
                          "source_cell": f"{GILT}!{letter}{GILT_SC4_ROW} (variant fiscal_sc3: {BUILD_OUT}!C216 = Sc3)",
                          "note": note}},
            "Sc4": {"B": {"value": float(book.value(GILT, f"{letter}{GILT_SC4_ROW}")) / 1e4,
                          "source_cell": f"{GILT}!{letter}{GILT_SC4_ROW}", "note": note}},
        }
    return out


def _legacy_cohorts(book: _Book) -> list[dict]:
    rows = []
    for row in LEGACY_ROWS:
        record = {key: book.value(LEGACY, f"{col}{row}") for key, col in LEGACY_COLUMNS.items()}
        record.update({key: book.value(LEGACY_A1, f"{col}{row - 19}") for key, col in LEGACY_A1_COLUMNS.items()})
        record.update({key: book.value(LEGACY_A2, f"{col}{row - 19}") for key, col in LEGACY_A2_COLUMNS.items()})
        record["source_rows"] = {"cohort": f"{LEGACY}!A{row}:AK{row}", "book_value": f"{LEGACY_A1}!A{row - 19}:W{row - 19}",
                                 "market_value": f"{LEGACY_A2}!A{row - 19}:AP{row - 19}"}
        rows.append(record)
    return rows


def extract(workbook_path: str | Path, variants_path: str | Path) -> dict:
    path = Path(workbook_path)
    sha = file_sha256(path)
    book = _Book(path)
    variants = load_variants(Path(variants_path), sha)
    checked = check_base_variant(book, variants)
    assumptions = {key: book.cell(sheet, cell) for key, (sheet, cell) in SCALAR_CELLS.items()}
    public_rate_by_margin = {key: {"value": float(variants[v][(SC4, "F11")]),
                                   "margin": float(variants[v][(SC4, "B11")]),
                                   "source": f"{SC4}!F11 (variant {v})"} for key, v in MARGIN_VARIANTS.items()}
    legacy_totals = {basis: {"total_gbp_bn": book.cell(LEGACY_A2, t), "equity_gbp_bn": book.cell(LEGACY_A2, e),
                             "debt_gbp_bn": book.cell(LEGACY_A2, d)}
                     for basis, (t, e, d) in LEGACY_TOTAL_CELLS.items()}
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "source_workbook": path.as_posix(),
        "workbook_sha256": sha,
        "variants_file": Path(variants_path).as_posix(),
        "variants_checked_cells": checked,
        "extracted_on": date.today().isoformat(),
        "rate_basis": "real_CPI",
        "price_year": 2024,
        "switches": {key: book.value(sheet, cell) for key, (sheet, cell) in SWITCH_CELLS.items()},
        "assumptions": assumptions,
        "public_rate_by_lending_margin": public_rate_by_margin,
        "cost_of_debt_by_rating": {str(book.value(ASSUMPTIONS, f"A{row}")): book.cell(ASSUMPTIONS, f"{DEBT_LADDER_COLUMN}{row}")
                                   for row in DEBT_LADDER_ROWS},
        "technologies": _technology_rows(book, variants),
        "legacy_cohorts": _legacy_cohorts(book),
        "legacy_acquisition_by_basis": legacy_totals,
        "variants": json.loads(Path(variants_path).with_suffix(".json").read_text(encoding="utf-8"))["variants"],
    }
    for year, block in _yield_shifts(book, variants).items():
        snapshot[f"sovereign_yield_shift_{year}"] = block
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workbook")
    parser.add_argument("variants", help="workbook_variants.py's CSV (its JSON beside it)")
    parser.add_argument("output")
    args = parser.parse_args()
    snapshot = extract(args.workbook, args.variants)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
