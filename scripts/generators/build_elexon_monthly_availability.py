"""Monthly station availability for a historical validation year from ELEXON physical notifications.

The 2020 validation's Stage 1 ran every nuclear station on one carrier-wide factor, so Hunterston B (both reactors
shut from 2018 until August 2020 for graphite inspections) ran at 840 MW in January while the fleet total matched
ESPENI; and it ran Peterhead at 874 MW against about 540 MW notified, which handed the balancing stage a cheap way to
relieve the B4 (SSE-SP) boundary that NESO did not have (user decision, 18 Nov 2026).

This writes rows into the year's ``thermal.availability_factors.generator_monthly_file``:

* **Nuclear** (every station): the monthly mean of its notified output over its capacity, a cap. The stations' sum
  matches ESPENI's metered nuclear to 0.5% in every month of 2020.
* **Held stations** (``--hold``, Peterhead): fixed at the monthly mean notified output (``fixed`` true: the minimum
  equals the maximum, so neither stage moves it). Peterhead stands in for the north of Scotland's voltage need,
  which a thermal-limits-only network does not represent.

Source: the calibration cache's ``pn_data_<year>.parquet`` (ELEXON PN, two samples a day, 09:30 and 18:30 UTC, in
MWh per settlement period, so MW is twice the value). Rows this script wrote before (``evidence`` starting
``ELEXON physical notifications``) are replaced; other rows are kept.

    python scripts/generators/build_elexon_monthly_availability.py --year 2020 \
        --mapping resources/generators/Validation_Jan2020_bmu_mapping.csv \
        --capacities resources/generators/Validation_Jan2020_marginal_costs_breakdown.csv \
        --out data/market/support/generator_monthly_availability_2020.csv --hold Peterhead
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import pandas as pd

EVIDENCE = "ELEXON physical notifications"
#: The power-station dictionary's partial match puts Heysham 1's reactors on Heysham 2; ELEXON's BMU ids are
#: unambiguous (HEYM1x Heysham 1, HEYM2x Heysham 2).
BMU_OVERRIDES = {
    "T_HEYM11": "Heysham 1", "T_HEYM12": "Heysham 1",
    "T_HEYM27": "Heysham 2", "T_HEYM28": "Heysham 2",
}


def read_pn(cache: Path, year: int) -> pd.DataFrame:
    """The year's sampled physical notifications, MW (wide: one column per BMU)."""
    with tarfile.open(cache) as archive:
        pn = pd.read_parquet(io.BytesIO(archive.extractfile(f"pn_data_{year}.parquet").read()))
    pn["datetime"] = pd.to_datetime(pn["datetime"])
    return pn.set_index("datetime") * 2.0   # MWh per half-hour -> MW


def station_bmus(mapping: pd.DataFrame, pn_columns) -> dict[str, list[str]]:
    rows = mapping[["bmu_id", "generator_name"]].copy()
    rows["generator_name"] = rows["bmu_id"].map(BMU_OVERRIDES).fillna(rows["generator_name"])
    rows = rows[rows["bmu_id"].isin(pn_columns)]
    return rows.groupby("generator_name")["bmu_id"].apply(list).to_dict()


def monthly_rows(pn: pd.DataFrame, bmus: dict[str, list[str]], capacities: pd.Series, year: int,
                 stations: list[str], fixed: bool) -> pd.DataFrame:
    rows = []
    for station in stations:
        if station not in bmus:
            raise ValueError(f"No ELEXON BMU mapped to {station!r}")
        if station not in capacities.index:
            raise ValueError(f"No capacity for {station!r}")
        output = pn[bmus[station]].sum(axis=1).clip(lower=0.0)
        for month, mean_mw in output.groupby(output.index.month).mean().items():
            rows.append({
                "generator": station, "year": year, "month": int(month),
                "availability_factor": round(min(1.0, float(mean_mw) / float(capacities[station])), 6),
                "evidence": f"{EVIDENCE} (monthly mean of the 09:30 and 18:30 samples; BMUs {' '.join(bmus[station])})",
                "notes": ("held at its notified output (minimum equals maximum): stands in for the voltage need a "
                          "thermal-limits-only network does not represent" if fixed else
                          "nuclear station's notified output; replaces the carrier-wide factor"),
                "fixed": fixed,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/market/elexon_calibration_cache.tar.gz"))
    parser.add_argument("--mapping", type=Path, required=True, help="a month's {scenario}_bmu_mapping.csv")
    parser.add_argument("--capacities", type=Path, required=True, help="a month's marginal-costs breakdown (p_nom_MW)")
    parser.add_argument("--out", type=Path, required=True, help="the year's generator_monthly_file")
    parser.add_argument("--hold", nargs="*", default=[], help="stations fixed at their notified output")
    args = parser.parse_args()

    pn = read_pn(args.cache, args.year)
    mapping = pd.read_csv(args.mapping)
    breakdown = pd.read_csv(args.capacities)
    capacities = breakdown.drop_duplicates("generator").set_index("generator")["p_nom_MW"]
    bmus = station_bmus(mapping, pn.columns)
    nuclear = sorted(breakdown.loc[breakdown["carrier"].eq("nuclear"), "generator"].unique())
    new = pd.concat([monthly_rows(pn, bmus, capacities, args.year, nuclear, fixed=False),
                     monthly_rows(pn, bmus, capacities, args.year, list(args.hold), fixed=True)], ignore_index=True)

    existing = pd.read_csv(args.out)
    if "fixed" not in existing.columns:
        existing["fixed"] = False
    kept = existing[~existing["evidence"].astype(str).str.startswith(EVIDENCE)]
    clash = set(kept["generator"]) & set(new["generator"])
    if clash:
        raise ValueError(f"Rows from another source already cover {sorted(clash)}")
    out = pd.concat([kept, new], ignore_index=True)
    out["fixed"] = out["fixed"].fillna(False).astype(bool)
    out.to_csv(args.out, index=False)
    print(f"{args.out}: {len(kept)} rows kept, {len(new)} written for {sorted(new['generator'].unique())}")


if __name__ == "__main__":
    main()
