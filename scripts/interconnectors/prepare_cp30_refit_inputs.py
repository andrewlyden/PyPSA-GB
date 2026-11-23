"""One-off data preparation for the 2013-weather refit of TYNDP prices (item F13, 7 Oct 2026).

Writes the two inputs of ``cp30_foreign_prices.build_cp30_refit_prices`` into
``data/interconnectors/tyndp2024/``:

1. ``tyndp2024_NT2030CY2009_zonal_hourly.csv`` (8,736 h): per European zone, TYNDP 2024 National Trends
   2030 (CY2009) demand (``{zone}|demand``) and wind onshore, wind offshore and solar potential
   (``{zone}|onwind`` ...; generation plus the zone's dumped energy shared by annual generation), and the
   marginal cost of the price nodes (``price|{node}``, EUR).
   Source: ``MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx``, sheet ``Hourly Market Data``.
2. ``renewables_ninja_2013_cf.csv`` (8,760 h, 2013 UTC): per zone the 2013 capacity factor of onshore wind
   (current fleet), offshore wind (near-term-future fleet, else current, else a neighbour) and solar PV (SARAH),
   unscaled. renewables.ninja v1.3 country data (Pfenninger and Staffell 2016; Staffell and Pfenninger 2016),
   CC BY-NC 4.0.

    python scripts/interconnectors/prepare_cp30_refit_inputs.py --tyndp-xlsx <file> --ninja-dir <dir>
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.interconnectors.cp30_foreign_prices import PRICE_NODES, REFIT_ZONES  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "interconnectors" / "tyndp2024"
TECH_CATEGORIES = {"onwind": ("Wind Onshore",), "offwind": ("Wind Offshore",),
                   "solar": ("Solar (Photovoltaic)", "Solar (Thermal)")}
NINJA_ISO = {"AT": "AT", "BE": "BE", "CZ": "CZ", "DKW": "DK", "DKE": "DK", "FI": "FI", "FR": "FR", "DE": "DE",
             "IT": "IT", "NL": "NL", "NO_S": "NO", "NO_MN": "NO", "PL": "PL", "PT": "PT", "SEM": "IE", "SI": "SI",
             "ES": "ES", "SE1": "SE", "SE2": "SE", "SE3": "SE", "SE4": "SE", "CH": "CH"}
OFFSHORE_PROXY = {"PL": "SE", "PT": "FR", "ES": "FR"}   # zones with offshore wind in TYNDP but none in ninja


def read_tyndp_zonal(xlsx: Path) -> pd.DataFrame:
    import openpyxl
    warnings.filterwarnings("ignore")
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = wb["Hourly Market Data"].iter_rows(values_only=True)
    cat = node = None
    for r in rows:
        first = [str(x).strip() if x is not None else "" for x in (r[:3] if r else [])]
        if "Category" in first:
            cat = [str(c).replace("\n", " ") if c is not None else "" for c in r]
        elif "Country" in first:
            node = [str(n) if n is not None else "" for n in r]
        elif "Hour" in first:
            break
    zone_of = {n: z for z, nodes in REFIT_ZONES.items() for n in nodes}
    price_nodes = {n for nodes in PRICE_NODES.values() for n in nodes}
    want = {}
    for j, (c, n) in enumerate(zip(cat, node)):
        if n in price_nodes and c.startswith("Marginal Cost"):
            want[j] = ("price", n)
        if n not in zone_of:
            continue
        z = zone_of[n]
        if c.startswith("Demand"):
            want[j] = ("demand", z)
        elif c.startswith("Dumped Energy"):
            want[j] = ("dumped", z)
        for tech, prefixes in TECH_CATEGORIES.items():
            if any(c.startswith(p) for p in prefixes):
                want[j] = (tech, z)
    data = [[r[j] for j in want] for r in rows if r is not None and any(x is not None for x in r[:3])]
    raw = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(list(want.values()))).apply(pd.to_numeric, errors="coerce")
    raw = raw.T.groupby(level=[0, 1]).sum(min_count=1).T.fillna(0.0)
    out = {}
    for z in REFIT_ZONES:
        out[f"{z}|demand"] = raw.get(("demand", z), pd.Series(0.0, index=raw.index))
        gens = {t: raw.get((t, z), pd.Series(0.0, index=raw.index)) for t in TECH_CATEGORIES}
        total = sum(float(g.sum()) for g in gens.values())
        dumped = raw.get(("dumped", z), pd.Series(0.0, index=raw.index))
        for t, g in gens.items():
            out[f"{z}|{t}"] = g + (float(g.sum()) / total if total > 0 else 0.0) * dumped
    for n in sorted(price_nodes):
        out[f"price|{n}"] = raw[("price", n)]
    frame = pd.DataFrame(out).round(3)
    if len(frame) != 8736:
        raise ValueError(f"expected 8,736 TYNDP hours, got {len(frame)}")
    return frame


def _ninja(path: Path, col: str):
    df = pd.read_csv(path, skiprows=2, index_col=0, parse_dates=True)
    if col not in df.columns:
        return None
    s = df[col].loc["2013-01-01":"2013-12-31 23:00"]
    if len(s) != 8760:
        raise ValueError(f"{path}: {len(s)} hours of 2013")
    return s.to_numpy(dtype=float)


def _wind(ninja_dir: Path, iso: str, kind: str):
    cur = ninja_dir / iso / f"ninja_wind_country_{iso}_current-merra-2_corrected.csv"
    nt = ninja_dir / iso / f"ninja_wind_country_{iso}_near-termfuture-merra-2_corrected.csv"
    if kind == "onwind":
        s = _ninja(cur, "onshore")
        return s if s is not None else _ninja(cur, "national")
    for path in (nt, cur):
        if path.exists():
            s = _ninja(path, "offshore")
            if s is not None and s.mean() > 0.05:
                return s
    if iso in OFFSHORE_PROXY:
        return _wind(ninja_dir, OFFSHORE_PROXY[iso], "offwind")
    return None


def read_ninja_2013(ninja_dir: Path, zonal: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for z, iso in NINJA_ISO.items():
        for tech in TECH_CATEGORIES:
            if zonal[f"{z}|{tech}"].sum() <= 0:
                continue
            s = (_ninja(ninja_dir / iso / f"ninja_pv_country_{iso}_sarah_corrected.csv", "national")
                 if tech == "solar" else _wind(ninja_dir, iso, tech))
            if s is None:
                raise ValueError(f"no renewables.ninja 2013 series for {z} {tech}")
            out[f"{z}|{tech}"] = s
    frame = pd.DataFrame(out, index=pd.date_range("2013-01-01", periods=8760, freq="h", name="datetime")).round(4)
    return frame.clip(lower=0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tyndp-xlsx", required=True, type=Path)
    ap.add_argument("--ninja-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    zonal = read_tyndp_zonal(args.tyndp_xlsx)
    zonal.to_csv(OUT_DIR / "tyndp2024_NT2030CY2009_zonal_hourly.csv", index_label="tyndp_hour")
    read_ninja_2013(args.ninja_dir, zonal).to_csv(OUT_DIR / "renewables_ninja_2013_cf.csv")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
