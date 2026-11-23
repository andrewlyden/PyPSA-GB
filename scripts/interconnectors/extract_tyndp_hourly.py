"""Extract the hourly TYNDP market data the 2050 foreign-price shape needs (task Y3).

One-off data preparation, run by hand, not by Snakemake. It reads the ENTSO-E /
ENTSOG TYNDP output workbooks and writes compact CSVs under
``data/interconnectors/tyndp2024/`` (see the README there for provenance and
licence, CC-BY 4.0).

TYNDP 2024 (final): ``MMStandardOutputFile_{DE|GA}2050_Plexos_CY{year}_v11_SoS.xlsb``,
sheet ``Hourly Market Data emarket``. Rows 11 and 12 (0-based 10 and 11) hold the
category and the node of each column; the data start on row 14. Per workbook the
extract keeps, for 8,736 hours (``01JAN00:00`` to ``30DEC23:00``):

- ``Marginal Cost`` of the nine market nodes GB trades with or is (EUR 2022/MWh);
- UK00 ``Wind Offshore`` generation potential (MW; generation plus curtailment),
  the shape-year regressor of the wind conditioning;
- UK00 ``Wind Onshore`` potential (MW) and UK00, FR00 and DE00 ``Demand`` (MW), kept
  so the weekday alignment of the TYNDP calendar can be checked.

TYNDP 2026 (draft) NT+: ``UK_TimeSeriesDashboard_TY2050_WS{nnn}_SoS.xlsx``, sheet
``E-Market``: UK00 ``Wind Offshore`` generation and ``RES curtailment`` (MW), 8,760
hours. The NT+ dashboards publish dispatched generation only, so the regressor is
offshore generation plus total RES curtailment (declared in the README).

Usage::

    python scripts/interconnectors/extract_tyndp_hourly.py tyndp2024 <xlsb> <out_csv>
    python scripts/interconnectors/extract_tyndp_hourly.py tyndp2026_uk <xlsx> <out_csv>
"""

import argparse
from pathlib import Path

import pandas as pd

MARKET_NODES = ("UK00", "UKNI", "IE00", "FR00", "BE00", "NL00", "DE00", "DKW1", "NOS0")
DEMAND_NODES = ("UK00", "FR00", "DE00")
SHEET_2024 = "Hourly Market Data emarket"


def _column_name(category: str, node: str):
    """The output column for one workbook column, or None to skip it."""
    if not isinstance(category, str) or not isinstance(node, str):
        return None
    if category.startswith("Marginal Cost") and node in MARKET_NODES:
        return f"mc_{node}"
    if node == "UK00" and category.startswith("Wind Offshore"):
        return "UK00_wind_offshore_potential_mw"
    if node == "UK00" and category.startswith("Wind Onshore"):
        return "UK00_wind_onshore_potential_mw"
    if node in DEMAND_NODES and category.startswith("Demand"):
        return f"demand_{node}_mw"
    return None


def extract_tyndp2024(workbook: str) -> pd.DataFrame:
    """Hourly market data of one TYNDP 2024 electricity workbook (see module docstring)."""
    from pyxlsb import open_workbook

    rows, columns = [], None
    with open_workbook(workbook) as wb:
        with wb.get_sheet(SHEET_2024) as sheet:
            category = node = None
            for i, row in enumerate(sheet.rows()):
                values = [c.v for c in row]
                if i == 10:
                    category = values
                elif i == 11:
                    node = values
                    columns = {}
                    for j, (cat, nd) in enumerate(zip(category, node)):
                        name = _column_name(cat, nd)
                        if name is not None:
                            columns.setdefault(name, []).append(j)
                elif i > 12:
                    if values[0] is None:
                        break
                    record = {"time": values[1]}
                    for name, idx in columns.items():
                        record[name] = sum(float(values[j] or 0.0) for j in idx)
                    rows.append(record)
    frame = pd.DataFrame(rows)
    missing = [f"mc_{n}" for n in MARKET_NODES if f"mc_{n}" not in frame]
    if missing or "UK00_wind_offshore_potential_mw" not in frame:
        raise ValueError(f"{workbook}: columns not found: {missing or 'UK00 offshore wind'}")
    if len(frame) != 8736:
        raise ValueError(f"{workbook}: expected 8,736 hours, found {len(frame)}")
    return frame


def extract_tyndp2026_uk(workbook: str) -> pd.DataFrame:
    """UK00 offshore wind generation and RES curtailment from one NT+ 2050 UK dashboard."""
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    sheet = wb["E-Market"]
    rows = list(sheet.iter_rows(values_only=True))
    category = rows[6]
    wanted = {"Wind Offshore": "UK00_wind_offshore_generation_mw", "RES curtailment": "UK00_res_curtailment_mw",
              "Marginal Cost": "mc_UK00"}
    idx = {}
    for j, cat in enumerate(category):
        for prefix, name in wanted.items():
            if isinstance(cat, str) and cat.startswith(prefix):
                idx.setdefault(name, []).append(j)
    data = [r for r in rows[10:] if r[0] is not None]
    frame = pd.DataFrame({"hour_of_year": [r[0] for r in data], "time": [r[1] for r in data]})
    for name, cols in idx.items():
        frame[name] = [sum(float(r[j] or 0.0) for j in cols) for r in data]
    if len(frame) != 8760:
        raise ValueError(f"{workbook}: expected 8,760 hours, found {len(frame)}")
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["tyndp2024", "tyndp2026_uk"])
    ap.add_argument("workbook")
    ap.add_argument("out_csv")
    a = ap.parse_args(argv)
    frame = extract_tyndp2024(a.workbook) if a.kind == "tyndp2024" else extract_tyndp2026_uk(a.workbook)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out_csv, index=False, float_format="%.6g")
    print(a.out_csv, frame.shape)


if __name__ == "__main__":
    main()
