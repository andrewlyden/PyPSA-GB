"""What the acquisition year's purchase bought, by technology group (user decision D1, 14 Nov 2026).

In a later model year (2035, 2040, 2045, 2050, ...) a case that bought the legacy fleet owns exactly the capacity the
purchase bought that the year's fleet still runs, and nothing more (independent review, 15 Nov 2026). A
pre-legislation row is eligible when it was built by the acquisition year on no contract or on a contract the
purchase bought, or built later on a contract the purchase bought (AR7 phases, the 2029-30 cohort's later twin,
Hinkley Point C's second unit); a row on a contract the purchase did not buy is never owned. Eligible rows are owned,
those on a bought contract first, then built ones, then by build year, up to each group's bought MW and each contract's bought MW
(``bought_mw_by_round``; Net Zero Teesside's DPA as ``ROUND_EQUIVALENTS``), within half a row; the rest stays
private. A plant the valuation steps up after the purchase year counts at its stepped capacity (:func:`valued_capacity`).

The table is written from the acquisition year's register (``--write``) and checked against it in every bill run
(``--check``), as the auction calibration table is, so the later years read a committed input rather than a
dependency on the acquisition year's run.

    python -m scripts.public_ownership.acquisition_baseline --register <group register.parquet> --out <table> --write
    python -m scripts.public_ownership.acquisition_baseline --register <group register.parquet> --out <table> --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: One table per acquisition run group (the group a later group's acquisition_from_group names): the central 2028 law
#: and the 2025-law bound buy different fleets.
BASELINE_DIR = ROOT / "data" / "public_ownership" / "financing"
BASELINE = BASELINE_DIR / "acquisition_baseline_cp30_2030_no_acceleration.csv"


def baseline_path(acquisition_group: str) -> Path:
    """The bought-capacity table of an acquisition run group."""
    return BASELINE_DIR / f"acquisition_baseline_{acquisition_group}.csv"


#: The case whose purchase the table records: every acquirable legacy row, energy-from-waste included (a case that
#: leaves a group out of its purchase applies its own exclusions on top).
BASELINE_CASE = "public_ownership@fully_nationalised_market_value_efw_acquired"
FALLBACK_CASE = "public_ownership@fully_nationalised_market_value"
#: Technology groups: a carrier's capacity in one year counts against the same group's purchase in another, where the
#: CP30 2030 fleet and the FES fleet name plant differently (CP30's "LAES" holds pumped-storage projects).
CARRIER_GROUP = {
    "wind_offshore": "offshore_wind", "wind_onshore": "onshore_wind", "solar_pv": "solar",
    "nuclear": "nuclear", "Battery": "battery", "battery": "battery",
    "CCGT": "gas", "OCGT": "gas_peaking", "gas_engine": "gas_peaking",
    "biomass": "biomass", "CCS Biomass": "biomass_ccs", "CCS Gas": "gas_ccs",
    "large_hydro": "hydro", "small_hydro": "hydro",
    "pumped_hydro": "long_duration_storage", "Pumped Storage Hydroelectricity": "long_duration_storage",
    "LAES": "long_duration_storage", "CAES": "long_duration_storage",
    "marine": "marine", "waste_to_energy": "energy_from_waste", "oil": "oil", "CHP": "chp",
}
COLUMNS = ("group", "bought_mw", "carriers", "acquisition_year", "source_case", "source")
#: Each group's bought MW by contract round (``round=MW;...``; ``none`` for plant on no contract), written beside
#: the group total (15 Nov 2026): a later year owns a contract's plant only up to what the purchase bought of it.
ROUND_COLUMN = "bought_mw_by_round"
NO_ROUND = "none"
#: The acquisition year's contract labels under the names a later year's register gives the same plant (one CCS
#: power plant in CP30 2030, Net Zero Teesside, is FES 2025's 2029 DPA cohort).
ROUND_EQUIVALENTS = {"NZT_POWER_DPA": ("DPA_2029",)}
TOLERANCE_MW = 1.0


def carrier_group(carrier) -> str:
    """The technology group of a carrier ('other:<carrier>' for one outside the table)."""
    return CARRIER_GROUP.get(str(carrier), f"other:{carrier}")


def baseline_from_register(register: pd.DataFrame, *, acquisition_year: int = 2030,
                           case_id: str | None = None) -> pd.DataFrame:
    """Bought MW by group: the acquisition year's state-owned legacy rows under the baseline case."""
    cases = set(register["case_id"].astype(str))
    case = case_id or (BASELINE_CASE if BASELINE_CASE in cases else FALLBACK_CASE)
    rows = register.loc[register["case_id"].astype(str).eq(case)]
    if rows.empty:
        raise ValueError(f"the register has no {case} rows")
    owned = rows["state_owned"].astype(str).str.lower().isin(["true", "1"])
    legacy = rows["finance_vintage"].astype(str).eq("legacy")
    bought = rows.loc[owned & legacy].copy()
    bought["group"] = bought["carrier"].map(carrier_group)
    bought["round"] = (bought["cfd_round"].fillna("").astype(str).replace("", NO_ROUND)
                       if "cfd_round" in bought.columns else NO_ROUND)
    bought["mw"] = pd.to_numeric(bought["capacity_mw"], errors="coerce").fillna(0.0)
    bought["mw"] = valued_capacity(bought, acquisition_year)
    out = (bought.groupby("group")
           .agg(bought_mw=("mw", "sum"),
                carriers=("carrier", lambda s: ";".join(sorted(set(map(str, s)))))).reset_index())
    by_round = bought.groupby(["group", "round"])["mw"].sum()
    out[ROUND_COLUMN] = out["group"].map(
        lambda g: ";".join(f"{r}={v:.6f}" for r, v in by_round.loc[g].sort_index().items()))
    out["acquisition_year"] = int(acquisition_year)
    out["source_case"] = case
    out["source"] = "the acquisition year's asset finance register: state-owned legacy rows of the baseline case"
    return out[list(COLUMNS) + [ROUND_COLUMN]].sort_values("group").reset_index(drop=True)


def valued_capacity(rows: pd.DataFrame, acquisition_year: int, capacity_from_year: Mapping | None = None) -> pd.Series:
    """Each bought row's MW as the purchase valued it: a named plant the valuation steps up after the purchase year
    (``acquisition.valuation.capacity_from_year``: Hinkley Point C, 2,317 MW in 2030, 3,260 MW from 2031) counts at
    its stepped capacity, shared over its rows pro rata (independent review, 15 Nov 2026: capping Hinkley at its 2030
    MW left 1 GW the purchase paid for private in 2040 and 2050)."""
    if capacity_from_year is None:
        from scripts.public_ownership.legacy_valuation import DEFAULT_SETTINGS

        capacity_from_year = DEFAULT_SETTINGS["capacity_from_year"]
    mw = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0).astype(float)
    base = rows["asset_id"].astype(str).str.split("::").str[0] if "asset_id" in rows.columns else None
    if base is None:
        return mw
    for name, step in (capacity_from_year or {}).items():
        hit = base.eq(str(name))
        if not hit.any() or int(step.get("from_year", 0)) <= int(acquisition_year):
            continue
        total = float(mw[hit].sum())
        target = float(step["capacity_mw"])
        if total > 0 and target > total:
            mw[hit] = mw[hit] * (target / total)
    return mw


def load_round_caps(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """``group -> {round: bought MW}`` with each 2030 label also under its later names (``ROUND_EQUIVALENTS``)."""
    table = pd.read_csv(path or BASELINE)
    if ROUND_COLUMN not in table.columns:
        raise ValueError(f"{path or BASELINE} has no {ROUND_COLUMN}; rewrite it with --write")
    out: dict[str, dict[str, float]] = {}
    for _, row in table.iterrows():
        caps: dict[str, float] = {}
        for item in str(row[ROUND_COLUMN] or "").split(";"):
            if "=" not in item:
                continue
            name, value = item.rsplit("=", 1)
            caps[name] = float(value)
            for later in ROUND_EQUIVALENTS.get(name, ()):
                caps[later] = float(value)
        out[str(row["group"])] = caps
    return out


def load_baseline(path: str | Path | None = None) -> dict[str, float]:
    """``group -> bought MW`` from the committed table."""
    table = pd.read_csv(path or BASELINE)
    missing = set(COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path or BASELINE} lacks columns {sorted(missing)}")
    return {str(r["group"]): float(r["bought_mw"]) for _, r in table.iterrows()}


def baseline_acquisition_year(path: str | Path | None = None) -> int:
    """The year the table's purchase was made (one value per table), so a later year needs no code change."""
    years = pd.to_numeric(pd.read_csv(path or BASELINE)["acquisition_year"], errors="coerce").dropna().unique()
    if len(years) != 1:
        raise ValueError(f"{path or BASELINE} names {len(years)} acquisition years; one table is one purchase")
    return int(years[0])


def acquired_rows(rows: pd.DataFrame, bought: Mapping[str, float], *,
                  round_caps: Mapping[str, Mapping[str, float]] | None = None,
                  carried_rounds: tuple[str, ...] = (), acquisition_year: int = 2030) -> np.ndarray:
    """Rows a buying case owns in a later year: only plant the purchase bought, oldest first, within what it bought.

    ``rows`` carries ``carrier``, ``finance_vintage`` (the register's base view), ``build_year``, ``capacity_mw``,
    ``cfd_round`` and, where present, ``excluded`` (rows outside every purchase: small-scale plant, evidence gaps).
    Eligible (15 Nov 2026; the earlier third tier, "any other later build", handed the MW of retired bought plant to
    plant the purchase never bought: Sizewell C, post-AR7 offshore, Hinkley's unbought part):

    * plant built by the acquisition year, on no contract or on a contract the purchase bought;
    * plant built later only on a contract the purchase bought (AR7 phases, Hinkley Point C).

    ``round_caps`` (``load_round_caps``) are the bought MW per contract; ``carried_rounds`` stand in for them where a
    table has none. A row is acquired while its group's and its contract's cumulative MW, the row included, stay within
    the purchase (half a row's MW of tolerance, so the boundary row goes the nearer way).
    """
    eligible = eligible_rows(rows, round_caps=round_caps, carried_rounds=carried_rounds,
                             acquisition_year=acquisition_year)
    n = len(rows)
    mw = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0).to_numpy()
    year = _build_years(rows, acquisition_year, eligible)
    built = year <= float(acquisition_year)
    key = _round_keys(rows)
    group = rows["carrier"].map(carrier_group).to_numpy()
    out = np.zeros(n, dtype=bool)
    for g in pd.unique(group[eligible]):
        cap = float(bought.get(str(g), 0.0))
        if cap <= 0.0:
            continue                                   # a group the purchase did not buy acquires nothing
        caps = dict((round_caps or {}).get(str(g), {})) if round_caps is not None else None
        idx = np.flatnonzero(eligible & (group == g))
        # Independent review (15 Nov 2026): rows on a bought contract first (each capped by its own contract's MW),
        # then plant on no contract; uncontracted FES plant the later fleet adds must not push out contract plant
        # the purchase bought (2040 marine: 79 MW of uncontracted rows had displaced 104 MW of AR4-AR8 rows).
        # Only with per-contract caps: without them a contract row is bounded by the group alone, and putting it first
        # could push out the plant that was running at the purchase.
        uncontracted_last = ((key[idx] == NO_ROUND) if caps is not None else np.zeros(len(idx), dtype=bool)).astype(int)
        order = idx[np.lexsort((year[idx], (~built[idx]).astype(int), uncontracted_last))]
        cum = 0.0
        used: dict[str, float] = {}
        for i in order:
            if cum + 0.5 * mw[i] > cap + TOLERANCE_MW:
                continue
            k = key[i]
            if caps is not None and k != NO_ROUND:
                if used.get(k, 0.0) + 0.5 * mw[i] > float(caps.get(k, 0.0)) + TOLERANCE_MW:
                    continue
                used[k] = used.get(k, 0.0) + mw[i]
            out[i] = True
            cum += mw[i]
    return out


def _build_years(rows: pd.DataFrame, acquisition_year: int, candidates: np.ndarray | None = None) -> np.ndarray:
    """Build years; a row without one is refused (independent review, 15 Nov 2026: it used to count as built by the
    acquisition year, a silent default)."""
    if "build_year" not in rows.columns:
        raise ValueError("acquired_rows needs build_year on every row")
    year = pd.to_numeric(rows["build_year"], errors="coerce").to_numpy(dtype=float)
    vintage = rows["finance_vintage"].astype(str).to_numpy()
    pool = candidates if candidates is not None else np.isin(vintage, ("legacy", "in_construction"))
    missing = np.isnan(year) & pool
    if missing.any():
        raise ValueError(f"{int(missing.sum())} pre-legislation rows have no build_year "
                         f"(e.g. {rows.loc[missing, 'carrier'].astype(str).iloc[0]})")
    return np.nan_to_num(year, nan=9999.0)


def _round_keys(rows: pd.DataFrame) -> np.ndarray:
    rounds = rows["cfd_round"].fillna("").astype(str).to_numpy() if "cfd_round" in rows.columns \
        else np.full(len(rows), "")
    return np.where(rounds == "", NO_ROUND, rounds)


def eligible_rows(rows: pd.DataFrame, *, round_caps: Mapping[str, Mapping[str, float]] | None = None,
                  carried_rounds: tuple[str, ...] = (), acquisition_year: int = 2030) -> np.ndarray:
    """Rows the purchase could own in a later year (the rule :func:`acquired_rows` and the continuity check share).

    Pre-legislation rows outside every exclusion that are either built by the acquisition year, on no contract or on
    a contract the purchase bought, or built later on a contract the purchase bought.
    """
    vintage = rows["finance_vintage"].astype(str).to_numpy()
    cand = np.isin(vintage, ("legacy", "in_construction"))
    if "excluded" in rows.columns:
        cand &= ~rows["excluded"].fillna(False).astype(bool).to_numpy()
    year = _build_years(rows, acquisition_year, cand)
    built = year <= float(acquisition_year)
    key = _round_keys(rows)
    group = rows["carrier"].map(carrier_group).to_numpy()
    out = np.zeros(len(rows), dtype=bool)
    for g in pd.unique(group[cand]):
        bought_rounds = (set((round_caps or {}).get(str(g), {})) if round_caps is not None
                         else set(carried_rounds) | {NO_ROUND})
        idx = np.flatnonzero(cand & (group == g))
        on_contract = key[idx] != NO_ROUND
        listed = np.isin(key[idx], list(bought_rounds))
        out[idx] = np.where(built[idx], ~on_contract | listed, on_contract & listed)
    return out


def main(argv=None) -> None:
    from scripts.public_ownership.register_io import read_register

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--register", required=True, help="the acquisition year's asset finance register")
    parser.add_argument("--acquisition-year", type=int, default=2030)
    parser.add_argument("--out", default=str(BASELINE))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--stamp", default=None, help="with --check: write this file when the table holds")
    args = parser.parse_args(argv)
    columns = ["case_id", "asset_id", "carrier", "finance_vintage", "state_owned", "capacity_mw", "cfd_round"]
    cases = [BASELINE_CASE, FALLBACK_CASE]
    register = read_register(args.register, columns=columns, cases=cases)
    fresh = baseline_from_register(register, acquisition_year=args.acquisition_year)
    if args.write:
        fresh.to_csv(args.out, index=False)
        print(fresh.to_string(index=False))
        return
    committed_table = pd.read_csv(args.out)
    if ROUND_COLUMN not in committed_table.columns:
        raise SystemExit(f"{args.out} has no {ROUND_COLUMN}; rewrite it with --write")
    committed_rounds = load_round_caps(args.out)
    fresh_rounds = {str(r["group"]): {k: float(v) for k, v in (i.rsplit("=", 1) for i in str(r[ROUND_COLUMN]).split(";") if "=" in i)}
                    for _, r in fresh.iterrows()}
    round_bad = {f"{g}/{k}": round(v - committed_rounds.get(g, {}).get(k, 0.0), 1)
                 for g, caps in fresh_rounds.items() for k, v in caps.items()
                 if abs(v - committed_rounds.get(g, {}).get(k, 0.0)) > TOLERANCE_MW}
    if round_bad:
        raise SystemExit(f"{args.out} has drifted from {args.register} by contract round (MW): {round_bad}; rewrite it "
                         "with --write and re-run the later years' registers")
    committed = committed_table.set_index("group")["bought_mw"]
    now = fresh.set_index("group")["bought_mw"]
    diff = (now.reindex(committed.index.union(now.index)).fillna(0.0)
            - committed.reindex(committed.index.union(now.index)).fillna(0.0)).abs()
    bad = diff[diff > TOLERANCE_MW]
    if len(bad):
        raise SystemExit(f"{args.out} has drifted from {args.register} (MW): {bad.round(1).to_dict()}; rewrite it with "
                         "--write and re-run the later years' registers")
    if args.stamp:
        Path(args.stamp).write_text("acquisition baseline holds\n", encoding="utf-8")
    print("acquisition baseline holds")


if __name__ == "__main__":
    main()
