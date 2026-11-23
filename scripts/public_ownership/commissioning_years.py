"""Decision 20: the commissioning year P3's balancing tie-break ranks each generator on.

The LIFO/FIFO tie-break (``cfd_bm_bids.resolve_tie_break``) orders state-owned CfD
units by commissioning year, but the register dates only assets that carry an
operational date: on the CP30 2030 fleet 5,423 of 5,476 state-owned CfD units had
none (the REPD pipeline and the CP30 increments), so the P3 solves could not start.
This table dates them from the evidence there is, in order:

1. ``asset_date``: the register's own ``commissioning_year`` (operational date or
   build year on the network);
2. ``lccc_unit_expected_start``: the LCCC contract portfolio's ``Expected_Start_Date``
   for the CfD unit of the same name and allocation round;
3. ``lccc_round_technology_median``: the median expected start of that round's
   units of the same technology in the portfolio;
4. ``ar8_round_model_year``: capacity on the modelled AR8 round (``AR8_modelled``,
   awarded in AR8 and built by the model year) is dated to the model year. A declared
   assumption; it makes that capacity the newest, so LIFO turns it down first.
5. ``lccc_round_median``: the median expected start of the round's dated units of
   any technology (18 Oct 2026). For CfD units whose technology has no dated unit
   in its round: LCCC lists live units without an expected start, so AR1's waste
   units (Energy Works (Hull), K3 CHP Facility) had no date and stopped the P3 solves.

Every year is capped at the model year, since the unit runs in that year's fleet.

Only CfD units are dated beyond (1), because only CfD bids carry the tie step.
The table is kept apart from the register and the share table on purpose: the
register's ``commissioning_year`` also sets remaining life in the financing and
the legacy valuation, and the share table is an input to every P2 solve, so
dating the pipeline there would move those results for a P3-only need.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.market.cfd_strike_table import MODELLED_ROUND as _MODELLED_ROUND  # noqa: E402
from scripts.public_ownership.ownership_shares import generator_name

# The register's CfD round -> the portfolio's Allocation_Round label. AR7 (the
# offshore pot) and AR7a (onshore, solar and tidal) are one round in the portfolio.
LCCC_ROUNDS = {
    "AR1": "Allocation Round 1", "AR2": "Allocation Round 2", "AR3": "Allocation Round 3",
    "AR4": "Allocation Round 4", "AR5": "Allocation Round 5", "AR6": "Allocation Round 6",
    "AR7": "Allocation Round 7", "AR7a": "Allocation Round 7",
    "Investment Contract": "Investment Contract",
}
# Model carrier -> the portfolio's Technology_Type values it covers.
LCCC_TECHNOLOGIES = {
    "wind_offshore": {"Offshore Wind", "Floating Offshore Wind"},
    "wind_onshore": {"Onshore Wind", "Remote Island Wind"},
    "solar_pv": {"Solar PV"},
    "marine": {"Tidal Stream"},
}
MODELLED_ROUND = _MODELLED_ROUND
REGISTER_COLUMNS = ("case_id", "asset_id", "carrier", "capacity_mw", "ownership_share",
                    "commissioning_year", "support_type", "cfd_round", "model_year")
OUTPUT_COLUMNS = ("case_id", "generator", "state_capacity_mw", "state_commissioning_year", "year_source")


def _name_key(name) -> str:
    """``Moor_House_Wind_Farm_CfD`` and ``Moor House Wind Farm`` -> ``moor house wind farm``."""
    words = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split()
    while words and words[-1] in {"cfd", "merchant"}:
        words.pop()
    return " ".join(words)


def load_portfolio(path) -> pd.DataFrame:
    """The LCCC portfolio's dated units: round label, technology, name key, start year."""
    frame = pd.read_csv(path)
    required = {"Name_of_CFD_Unit", "Allocation_Round", "Technology_Type", "Status", "Expected_Start_Date"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    frame = frame.loc[~frame["Status"].astype(str).str.strip().eq("Terminated")]
    year = pd.to_datetime(frame["Expected_Start_Date"], errors="coerce", format="mixed").dt.year
    return pd.DataFrame({
        "round": frame["Allocation_Round"].astype(str).str.strip(),
        "technology": frame["Technology_Type"].astype(str).str.strip(),
        "name_key": frame["Name_of_CFD_Unit"].map(_name_key),
        "year": year,
    }).dropna(subset=["year"])


def _lookups(portfolio: pd.DataFrame):
    by_unit = portfolio.groupby(["round", "name_key"])["year"].median().to_dict()
    by_technology = {}
    for carrier, technologies in LCCC_TECHNOLOGIES.items():
        rows = portfolio.loc[portfolio["technology"].isin(technologies)]
        for label, years in rows.groupby("round")["year"]:
            by_technology[(label, carrier)] = float(np.floor(years.median()))
    by_round = {label: float(np.floor(years.median())) for label, years in portfolio.groupby("round")["year"]}
    return by_unit, by_technology, by_round


def asset_years(register: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """Each register row's year and the evidence it rests on (``none`` if undated)."""
    missing = set(REGISTER_COLUMNS) - set(register.columns)
    if missing:
        raise ValueError(f"register lacks columns {sorted(missing)}")
    by_unit, by_technology, by_round = _lookups(portfolio)
    years, sources = [], []
    for row in register.itertuples(index=False):
        year = pd.to_numeric(row.commissioning_year, errors="coerce")
        if pd.notna(year):
            years.append(float(year)); sources.append("asset_date"); continue
        if str(row.support_type).strip().lower() != "cfd":
            years.append(np.nan); sources.append("none"); continue
        cfd_round = str(row.cfd_round).strip()
        if cfd_round == MODELLED_ROUND:
            years.append(float(row.model_year)); sources.append("ar8_round_model_year"); continue
        label = LCCC_ROUNDS.get(cfd_round)
        unit = by_unit.get((label, _name_key(generator_name(row.asset_id)))) if label else None
        if unit is not None and pd.notna(unit):
            years.append(float(np.floor(unit))); sources.append("lccc_unit_expected_start"); continue
        median = by_technology.get((label, str(row.carrier))) if label else None
        if median is not None:
            years.append(median); sources.append("lccc_round_technology_median"); continue
        round_median = by_round.get(label) if label else None
        if round_median is not None:
            years.append(round_median); sources.append("lccc_round_median"); continue
        years.append(np.nan); sources.append("none")
    # A unit in the model year's fleet runs in that year, so no later date can
    # hold for it (the AR7 offshore median is 2031); capping also keeps later
    # rounds from ranking older than the AR8 round.
    model_year = pd.to_numeric(register["model_year"], errors="coerce").to_numpy(dtype=float)
    # np.minimum, not np.fmin: an undated unit must stay undated, not become the model year.
    return register.assign(tie_break_year=np.minimum(np.asarray(years, dtype=float), model_year),
                           year_source=sources)


def generator_commissioning_years(register: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """State-owned capacity-weighted year per (case, generator), as the share table weights it."""
    frame = asset_years(register, portfolio)
    frame["generator"] = frame["asset_id"].map(generator_name)
    capacity = pd.to_numeric(frame["capacity_mw"], errors="coerce").fillna(0.0)
    state = capacity * pd.to_numeric(frame["ownership_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    dated = state.where(frame["tie_break_year"].notna(), 0.0)
    frame = frame.assign(_state=state, _dated=dated, _year_mw=(frame["tie_break_year"] * dated).fillna(0.0))
    keys = ["case_id", "generator"]
    grouped = frame.groupby(keys, sort=False)[["_state", "_dated", "_year_mw"]].sum()
    grouped["state_commissioning_year"] = grouped["_year_mw"].div(grouped["_dated"]).where(grouped["_dated"] > 0.0)
    # The evidence carrying the most state-owned capacity names the row's source.
    source = (frame.assign(_w=dated).groupby(keys + ["year_source"], sort=False)["_w"].sum()
              .reset_index().sort_values("_w", ascending=False).drop_duplicates(keys).set_index(keys)["year_source"])
    grouped["year_source"] = source.reindex(grouped.index).where(grouped["_dated"] > 0.0, "none")
    out = grouped.reset_index().rename(columns={"_state": "state_capacity_mw"})
    return out[list(OUTPUT_COLUMNS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register", help="the per-case asset finance register CSV")
    parser.add_argument("portfolio", help="data/market/lccc/cfd_contract_portfolio_status.csv")
    parser.add_argument("output")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="restrict to these case ids (default: every case in the register)")
    args = parser.parse_args()
    from scripts.public_ownership.register_io import iter_register_cases

    # 30 Sep 2026: whole register keys at a time, in register order (memory flat in the case
    # count); a (case, generator) pair never spans two keys, so the rows come out in the same order.
    # 6 Oct 2026: --cases keeps the table to the cases the bid months read; at 5,330 cases the
    # full table passed 3 GB.
    portfolio = load_portfolio(args.portfolio)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cases_by_source: dict[str, set] = {}
    owned_rows = undated = 0
    found: set = set()
    frames = iter_register_cases(args.register, columns=list(REGISTER_COLUMNS), batch_cases=None,
                                 cases=args.cases or None)
    for i, frame in enumerate(frames):
        if args.cases:
            frame = frame.loc[frame["case_id"].astype(str).isin(args.cases)]
            found.update(frame["case_id"].astype(str))
        # Written as it is made: the table is about a row per generator per case.
        table = generator_commissioning_years(frame, portfolio)
        table.to_csv(args.output, index=False, mode="w" if i == 0 else "a", header=i == 0)
        owned = table.loc[table["state_capacity_mw"] > 0.0]
        for source, cases in owned.groupby("year_source")["case_id"]:
            cases_by_source.setdefault(source, set()).update(cases)
        owned_rows += len(owned)
        undated += int(owned["state_commissioning_year"].isna().sum())
    unknown = sorted(set(args.cases or ()) - found)
    if unknown:
        raise SystemExit(f"register has no rows for {unknown}")
    counts =pd.Series({s: len(c) for s, c in sorted(cases_by_source.items())}, name="cases", dtype="int64")
    print(counts.rename_axis("year_source").to_string())
    print(f"{owned_rows:,} state-owned (case, generator) rows; "
          f"{undated:,} undated (non-CfD plant is left undated)")


if __name__ == "__main__":
    main()
