"""Recalculate the financing workbook's own alternative cases in Excel (Windows only).

The workbook's sensitivities are dropdown switches (``Sc4 Operative!B5`` lending
margin, ``Assumptions!B10`` ERP convention, ``Build-Out!C216`` Sc3/Sc4 fiscal
path, ...).  openpyxl reads only the cached results of the switches as saved, so
this script drives Excel through COM: for each variant it opens the workbook
read-only, sets the switches, recalculates, reads a fixed set of output ranges and
discards the workbook.  The results go to a committed CSV (one row per cell) with
a JSON beside it naming the workbook's SHA256 and the variants; the extractor
(``extract_workbook_v5.py``) reads those files, so no Snakemake rule needs Excel.

The ``base`` variant sets nothing; the extractor checks it against the saved
workbook's cached values, which proves the saved file was calculated.

Run by hand after the workbook changes::

    python scripts/public_ownership/workbook_variants.py data/public_ownership/GB_Power_Model_v6.2.xlsx \
        data/public_ownership/financing/workbook_v6_2_variants.csv

A workbook downloaded from the web carries Windows' mark of the web and Excel
refuses to open it through COM; clear it first (PowerShell ``Unblock-File``).
A workbook saved by LibreOffice (v6.2 was) opens in Excel only with repair:
pass ``--repair``.  The ``base`` parity check then proves the repair changed
none of the outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path

#: The variants: name -> (description, {(sheet, cell): value}).  Values are the
#: dropdown strings exactly as the workbook's data validation lists them.
VARIANTS: dict[str, tuple[str, dict[tuple[str, str], object]]] = {
    "base": ("the workbook as saved (parity check against its cached values)", {}),
    "erp_fixed_tmr": ("private rates at Fixed TMR (CEPA / UKRN), the workbook's co-equal ERP case",
                      {("Assumptions", "B10"): "Fixed TMR (CEPA / UKRN)"}),
    "margin_pwlb_hra": ("state cash rate at the PWLB HRA margin (+40 bp)",
                        {("Sc4 Operative", "B5"): "PWLB HRA (+40bp)"}),
    "margin_pwlb_certainty": ("state cash rate at the PWLB certainty margin (+80 bp)",
                              {("Sc4 Operative", "B5"): "PWLB certainty (+80bp)"}),
    "margin_pwlb_standard": ("state cash rate at the PWLB standard margin (+100 bp)",
                             {("Sc4 Operative", "B5"): "PWLB standard (+100bp)"}),
    "margin_gilt_flat": ("state cash rate at the gilt yield (no margin)",
                         {("Sc4 Operative", "B5"): "Gilt flat (0bp)"}),
    # v6.2 (user, 30 Oct 2026): the central premium is a flat 1.9% (Sc4 Operative B7, B10 "Flat premium
    # (no beta)"); the sensitivities are none, the Green Book review's own 1% flat, 2% per unit of CEPA
    # asset beta (v5's central, France) and the market ERP per unit of beta (the upper bound).
    "premium_none": ("economic rate without a systematic-risk premium (pure Arrow-Lind, Green Book)",
                     {("Sc4 Operative", "B6"): "None (pure Arrow-Lind)"}),
    "premium_flat_10": ("economic rate at a flat 1% (the Green Book Discount Rate Review's recommendation, para 55)",
                        {("Sc4 Operative", "B7"): 0.01}),
    "premium_beta_20": ("economic rate at 2% per unit of CEPA asset beta (France, Quinet 2013; v5's central)",
                        {("Sc4 Operative", "B7"): 0.02, ("Sc4 Operative", "B10"): "CEPA asset beta"}),
    "premium_market_erp_beta": ("economic rate at the market ERP per unit of CEPA asset beta (the Market Economy "
                                "Investor Principle bound)",
                                {("Sc4 Operative", "B6"): "Market ERP (operative)",
                                 ("Sc4 Operative", "B10"): "CEPA asset beta"}),
    "fiscal_sc3": ("fiscal paths for Scenario 3 (public new build, existing fleet private)",
                   {("Build-Out", "C216"): "Sc3: new build only"}),
}

#: Output ranges read in every variant: (sheet, range, what it is).
OUTPUTS: tuple[tuple[str, str, str], ...] = (
    ("WACC Scenarios", "A7:J39", "static rates by technology (Sc1/Sc2 lead and vanilla, Sc3/Sc4 cash, economic)"),
    ("Gilt-Adjusted WACCs", "D5:AC8", "gilt-yield shift by year (bp): years, A, B Sc2, B Sc4/Sc3"),
    ("Gilt-Adjusted WACCs", "D14:AC16", "state cash rate by investment year: years, A, B"),
    ("Sc4 Operative", "B11:J11", "public lending margin, cash rate and P"),
    ("Sc4 Operative", "A15:H47", "Scenario 4 rates by technology"),
)
_CELL = re.compile(r"([A-Z]+)(\d+)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_letter(index: int) -> str:
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _column_index(letters: str) -> int:
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch) - 64
    return index


def _cells(range_address: str, values) -> list[tuple[str, object]]:
    """``[(cell address, value)]`` for an Excel ``Value2`` block read from ``range_address``."""
    first = range_address.split(":")[0]
    col0, row0 = _CELL.fullmatch(first).groups()
    col0, row0 = _column_index(col0), int(row0)
    if not isinstance(values, tuple):
        values = ((values,),)
    out = []
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if value is None or value == "":
                continue
            out.append((f"{_column_letter(col0 + j)}{row0 + i}", value))
    return out


def run_variants(workbook: Path, variants=None, *, repair: bool = False) -> list[dict]:
    """Every variant's output cells, recalculated in Excel (``repair``: open with xlRepairFile)."""
    import pythoncom
    import win32com.client

    variants = VARIANTS if variants is None else variants
    pythoncom.CoInitialize()
    excel = win32com.client.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    excel.AutomationSecurity = 3  # msoAutomationSecurityForceDisable: no macros
    rows = []
    try:
        for name, (_description, switches) in variants.items():
            # Workbooks.Open(Filename, UpdateLinks=0, ReadOnly=True, ..., CorruptLoad=1 xlRepairFile).
            book = (excel.Workbooks.Open(str(Path(workbook).resolve()), 0, True, None, None, None, True, None, None,
                                         False, False, None, False, None, 1) if repair
                    else excel.Workbooks.Open(str(Path(workbook).resolve()), 0, True))
            try:
                for (sheet, cell), value in switches.items():
                    book.Worksheets(sheet).Range(cell).Value = value
                excel.CalculateFull()
                for sheet, address, _what in OUTPUTS:
                    for cell, value in _cells(address, book.Worksheets(sheet).Range(address).Value2):
                        rows.append({"variant": name, "sheet": sheet, "cell": cell, "value": value})
            finally:
                book.Close(False)
    finally:
        excel.Quit()
        pythoncom.CoUninitialize()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path, help="CSV; a JSON of the same stem is written beside it")
    parser.add_argument("--repair", action="store_true", help="open with Excel's repair (a LibreOffice-saved workbook)")
    args = parser.parse_args()
    rows = run_variants(args.workbook, repair=args.repair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "sheet", "cell", "value"])
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "workbook": args.workbook.as_posix(),
        "workbook_sha256": file_sha256(args.workbook),
        "written_on": date.today().isoformat(),
        "opened_with_repair": bool(args.repair),
        "outputs": [{"sheet": s, "range": r, "what": w} for s, r, w in OUTPUTS],
        "variants": {name: {"description": d, "switches": [{"sheet": s, "cell": c, "value": v}
                                                           for (s, c), v in sw.items()]}
                     for name, (d, sw) in VARIANTS.items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
