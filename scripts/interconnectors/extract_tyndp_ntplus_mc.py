"""Extract a TYNDP 2026 National Trends+ year's hourly marginal costs from the hourly dashboard archive (10 Nov 2026).

The 2050 table (``data/interconnectors/tyndp2024/tyndp2026_NTplus2050_mc.csv``) was built by hand from
``NT+_TimeSeriesDashboard.zip``; this script builds the same table for any target year: for each weather scenario
in ``{year}/Weather scenario NNN/``, every country workbook's ``E-Market`` sheet is read and each market node's
``Marginal Cost [EUR/MWh_e]`` column kept (row 6 the node, row 7 the label, hours from row 11; 8,760 TYNDP hours).
Columns ``{node}_WS{NNN}``, as the 2050 table.

    python scripts/interconnectors/extract_tyndp_ntplus_mc.py --zip <NT+_TimeSeriesDashboard.zip> --year 2040 \\
        --out data/interconnectors/tyndp2024/tyndp2026_NTplus2040_mc.csv
"""
from __future__ import annotations

import argparse
import io
import zipfile

import numpy as np
import openpyxl
import pandas as pd

#: The 2050 table's nodes (GB's neighbours and GB itself).
NODES = ("UK00", "UKNI", "IE00", "FR00", "BE00", "NL00", "DE00", "DKW1", "NOS2")
HOURS = 8760
ROOT = "NT+_TimeSeriesDashboard"


def node_columns(blob: bytes, nodes) -> dict[str, np.ndarray]:
    """Each wanted node's hourly marginal cost in one country workbook."""
    ws = openpyxl.load_workbook(io.BytesIO(blob), read_only=True, data_only=True)["E-Market"]
    rows = list(ws.iter_rows(values_only=True))
    out = {}
    for c in range(len(rows[0])):
        node, label = rows[5][c], str(rows[6][c] or "")
        if node in nodes and label.startswith("Marginal Cost") and node not in out:
            values = [rows[r][c] for r in range(10, 10 + HOURS)]
            out[str(node)] = np.asarray([float(v) if v is not None else np.nan for v in values], dtype=float)
    return out


def extract(zip_path: str, year: int, nodes=NODES) -> pd.DataFrame:
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if n.startswith(f"{ROOT}/{year}/Weather scenario ") and n.endswith(".xlsx")
             and "__MACOSX" not in n]
    scenarios = sorted({n.split("/")[2].split()[-1] for n in names})
    table = {"hour_of_year": np.arange(1, HOURS + 1)}
    for ws in scenarios:
        found: dict[str, np.ndarray] = {}
        # Country workbooks whose code starts a wanted node first (UK00 in UK_, ...), then the rest.
        files = [n for n in names if f"Weather scenario {ws}/" in n]
        prefixes = {node[:2] for node in nodes}
        files.sort(key=lambda n: (n.split("/")[-1][:2] not in prefixes, n))
        for name in files:
            missing = [n for n in nodes if n not in found]
            if not missing:
                break
            found.update(node_columns(z.read(name), missing))
        lost = [n for n in nodes if n not in found]
        if lost:
            raise SystemExit(f"{year} WS{ws}: no marginal cost for {lost}")
        for node in nodes:
            table[f"{node}_WS{ws}"] = found[node]
        print(f"{year} WS{ws}: {len(nodes)} nodes")
    return pd.DataFrame(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = extract(a.zip, a.year)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {out.shape}; mean by column:", out.drop(columns="hour_of_year").mean().round(1).to_dict())


if __name__ == "__main__":
    main()
