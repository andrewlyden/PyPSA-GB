"""Check that the ownership tables the public-bids BM solves read still match the register.

The P2 and P3 balancing months (item BM1) bid on two tables made from the finance
register: the per-generator state share (``ownership_shares.py``) and the P3
tie-break's commissioning year (``commissioning_years.py``).  Every bill-layer
run rewrites the register, although a lever change there almost never moves who
owns what.  If the tables followed the register by modification time, each such
run would re-solve all 26 bid months for no change, so the two table rules read
the register as ``ancient()`` and are rebuilt only on request.

This check is the other half of that bargain.  It recomputes both tables from the
current register, **for the cases the solves read only** (a table carries every
case in the register, and a longer case list must not look like an ownership
change), and fails if either differs from the table on disk.  The bid-group bill
merges take its stamp as an input, so a real ownership change cannot reach a bill
through stale solves.

Usage::

    python scripts/public_ownership/verify_ownership_tables.py REGISTER PORTFOLIO STAMP \\
        --shares TABLE::CASE [TABLE::CASE ...] [--years TABLE::CASE ...] \\
        [--accept-carriers waste_to_energy --accept-reason TEXT]

Accepted changes (user decision, 10 Nov 2026): ``--accept-carriers`` names carriers whose generators may differ,
but only by the state owning less than the solve read (a purchase-scope lever that takes plant out of the state's
purchase: energy-from-waste). Those generators are left out of the comparison and listed in the stamp with their
MW and ``--accept-reason``; every other generator is checked exactly as before. The P3 bid months were solved with
EfW state-owned; EfW is redispatched by 5 MWh a year in both the market and the P3 bid months, so re-solving them
without it would change nothing.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership import commissioning_years as years_mod  # noqa: E402
from scripts.public_ownership import ownership_shares as shares_mod  # noqa: E402
from scripts.public_ownership.register_io import read_register  # noqa: E402

#: Separates a table path from a case id on the command line (case ids contain ``@``).
PAIR_SEPARATOR = "::"
#: Float tolerance: the tables are written through CSV, which rounds the last bit.
ABS_TOLERANCE = 1e-9
REL_TOLERANCE = 1e-9
SHARE_COLUMNS = ("capacity_mw", "state_capacity_mw", "state_share", "state_commissioning_year")
YEAR_COLUMNS = ("state_capacity_mw", "state_commissioning_year")

REBUILD_HINT = (
    "Ownership of a solved bid case has changed, so the P2/P3 balancing months are stale. Rebuild the "
    "tables, which re-solves those months:\n"
    "  python -m snakemake <targets> --rerun-triggers mtime -R build_public_ownership_state_shares "
    "build_public_ownership_commissioning_years\n"
    "If the change is intended to leave ownership alone, find the lever that moved it before re-solving."
)


def parse_pairs(values: list[str] | None) -> dict[str, list[str]]:
    """``["path::case", ...]`` -> ``{path: [case, ...]}``, in the order given."""
    pairs: dict[str, list[str]] = {}
    for value in values or []:
        if PAIR_SEPARATOR not in value:
            raise SystemExit(f"expected TABLE{PAIR_SEPARATOR}CASE, got {value!r}")
        path, case = value.split(PAIR_SEPARATOR, 1)
        pairs.setdefault(path, [])
        if case not in pairs[path]:
            pairs[path].append(case)
    return pairs


def compare_tables(expected: pd.DataFrame, saved: pd.DataFrame, columns: tuple[str, ...],
                   text_columns: tuple[str, ...] = ()) -> dict:
    """Compare two per-(case, generator) tables; return the differences found.

    Args:
        expected: the table recomputed from the current register.
        saved: the table on disk, restricted to the same cases.
        columns: numeric columns compared to :data:`ABS_TOLERANCE` / :data:`REL_TOLERANCE`.
        text_columns: columns compared exactly.

    Returns:
        ``{"rows": n, "missing": [...], "extra": [...], "max_abs_diff": {col: x}, "text_mismatches": {col: n}}``.
    """
    keys = ["case_id", "generator"]
    left = expected.assign(case_id=expected["case_id"].astype(str), generator=expected["generator"].astype(str))
    right = saved.assign(case_id=saved["case_id"].astype(str), generator=saved["generator"].astype(str))
    merged = left.merge(right, on=keys, how="outer", suffixes=("_expected", "_saved"), indicator=True)
    report = {
        "rows": int(len(left)),
        "missing": merged.loc[merged["_merge"].eq("left_only"), keys].head(10).values.tolist(),
        "extra": merged.loc[merged["_merge"].eq("right_only"), keys].head(10).values.tolist(),
        "max_abs_diff": {}, "text_mismatches": {},
    }
    both = merged.loc[merged["_merge"].eq("both")]
    for column in columns:
        a = pd.to_numeric(both[f"{column}_expected"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(both[f"{column}_saved"], errors="coerce").to_numpy(dtype=float)
        same_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=REL_TOLERANCE, atol=ABS_TOLERANCE) | same_nan
        diff = np.where(same_nan, 0.0, np.abs(a - b))
        report["max_abs_diff"][column] = float(np.nanmax(diff)) if len(diff) else 0.0
        if not close.all():
            report.setdefault("failed_columns", []).append(column)
    for column in text_columns:
        mismatches = int((both[f"{column}_expected"].astype(str) != both[f"{column}_saved"].astype(str)).sum())
        report["text_mismatches"][column] = mismatches
        if mismatches:
            report.setdefault("failed_columns", []).append(column)
    return report


def accepted_generators(register_path: str, cases: list[str], carriers) -> set[str]:
    """Generators of ``cases`` whose register rows are all of an accepted carrier."""
    if not carriers:
        return set()
    register = read_register(register_path, columns=["case_id", "asset_id", "carrier"], cases=cases)
    register = register.loc[register["case_id"].astype(str).isin(cases)]
    generator = register["asset_id"].map(shares_mod.generator_name)
    accepted = register["carrier"].astype(str).isin([str(c) for c in carriers])
    by_generator = accepted.groupby(generator).all()
    return set(by_generator.index[by_generator].astype(str))


def split_accepted(expected: pd.DataFrame, saved: pd.DataFrame, accepted: set[str]):
    """Take the accepted generators out of both tables; refuse any whose state capacity grew."""
    if not accepted:
        return expected, saved, []
    keys = ["case_id", "generator"]
    e = expected.loc[expected["generator"].astype(str).isin(accepted), keys + ["state_capacity_mw"]]
    s = saved.loc[saved["generator"].astype(str).isin(accepted), keys + ["state_capacity_mw"]]
    both = e.merge(s, on=keys, how="outer", suffixes=("_now", "_solved"))
    both = both.fillna({"state_capacity_mw_now": 0.0, "state_capacity_mw_solved": 0.0})
    grew = both.loc[both["state_capacity_mw_now"] > both["state_capacity_mw_solved"] + ABS_TOLERANCE]
    if len(grew):
        raise SystemExit("accepted carriers may only lose state ownership; these gained it: "
                         f"{grew[keys].head(10).values.tolist()}")
    moved = both.loc[(both["state_capacity_mw_solved"] - both["state_capacity_mw_now"]).abs() > ABS_TOLERANCE]
    listing = [{"case_id": str(r.case_id), "generator": str(r.generator),
                "state_mw_solved": float(r.state_capacity_mw_solved), "state_mw_now": float(r.state_capacity_mw_now)}
               for r in moved.itertuples(index=False)]

    def keep(table):
        return table.loc[~table["generator"].astype(str).isin(accepted)]
    return keep(expected), keep(saved), listing


def check_table(kind: str, table_path: str, cases: list[str], register_path: str, portfolio_path: str,
                accept_carriers=(), accept_reason: str = "") -> dict:
    """Recompute one table for ``cases`` and compare it with the file on disk."""
    saved = pd.read_csv(table_path)
    saved = saved.loc[saved["case_id"].astype(str).isin(cases)]
    accepted = accepted_generators(register_path, cases, accept_carriers)
    if kind == "shares":
        columns = list(shares_mod.REGISTER_COLUMNS) + list(shares_mod.OPTIONAL_REGISTER_COLUMNS)
        register = read_register(register_path, columns=columns, cases=cases)
        expected = shares_mod.generator_state_shares(register.loc[register["case_id"].astype(str).isin(cases)])
        expected, saved, moved = split_accepted(expected, saved, accepted)
        report = compare_tables(expected, saved, SHARE_COLUMNS)
    else:
        register = read_register(register_path, columns=list(years_mod.REGISTER_COLUMNS), cases=cases)
        register = register.loc[register["case_id"].astype(str).isin(cases)]
        expected = years_mod.generator_commissioning_years(register, years_mod.load_portfolio(portfolio_path))
        expected, saved, moved = split_accepted(expected, saved, accepted)
        report = compare_tables(expected, saved, YEAR_COLUMNS, text_columns=("year_source",))
    absent = sorted(set(cases) - set(expected["case_id"].astype(str)))
    if absent:
        report.setdefault("failed_columns", []).append("case_missing_from_register")
        report["absent_cases"] = absent
    report.update(kind=kind, table=table_path, cases=cases)
    if moved:
        report["accepted_changes"] = {"carriers": list(accept_carriers), "reason": accept_reason,
                                      "generators": len(moved),
                                      "state_mw_solved": round(sum(m["state_mw_solved"] for m in moved), 3),
                                      "state_mw_now": round(sum(m["state_mw_now"] for m in moved), 3),
                                      "rows": moved}
    report["ok"] = not report.get("failed_columns") and not report["missing"] and not report["extra"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("register", help="the run group's asset finance register (.parquet)")
    parser.add_argument("portfolio", help="data/market/lccc/cfd_contract_portfolio_status.csv")
    parser.add_argument("stamp", help="the JSON stamp written when every table matches")
    parser.add_argument("--shares", nargs="*", default=[], help=f"TABLE{PAIR_SEPARATOR}CASE pairs")
    parser.add_argument("--years", nargs="*", default=[], help=f"TABLE{PAIR_SEPARATOR}CASE pairs")
    parser.add_argument("--accept-carriers", nargs="*", default=[],
                        help="carriers whose generators may only lose state ownership (recorded in the stamp)")
    parser.add_argument("--accept-reason", default="", help="why the accepted change leaves the solves valid")
    args = parser.parse_args()
    if args.accept_carriers and not args.accept_reason.strip():
        raise SystemExit("--accept-carriers needs --accept-reason")
    accept = dict(accept_carriers=tuple(args.accept_carriers), accept_reason=args.accept_reason)

    reports = [check_table("shares", table, cases, args.register, args.portfolio, **accept)
               for table, cases in parse_pairs(args.shares).items()]
    reports += [check_table("years", table, cases, args.register, args.portfolio, **accept)
                for table, cases in parse_pairs(args.years).items()]
    for report in reports:
        status = "matches" if report["ok"] else "DIFFERS"
        print(f"{report['kind']:6s} {report['table']} for {report['cases']}: {status} "
              f"({report['rows']:,} rows; max abs diff {report['max_abs_diff']})")
        if report.get("accepted_changes"):
            a = report["accepted_changes"]
            print(f"       accepted: {a['generators']} {a['carriers']} generators, state MW "
                  f"{a['state_mw_solved']:,.1f} as solved -> {a['state_mw_now']:,.1f} now ({a['reason']})")
    if not all(report["ok"] for report in reports):
        print(json.dumps([r for r in reports if not r["ok"]], indent=1, default=str), file=sys.stderr)
        raise SystemExit(REBUILD_HINT)
    Path(args.stamp).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stamp).write_text(json.dumps({"register": args.register, "tables": reports}, indent=1, default=str),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
