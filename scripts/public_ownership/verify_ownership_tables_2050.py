"""The ownership-table check for vintaged (2050) groups, with decision 32's one declared exception.

``verify_ownership_tables.py`` refuses any change in who owns what in a case the P2 bid months
were solved on. Decision 32 (user, 3 October 2026) takes Sizewell C's RAB out of P1's stake and
P2's boundary in the finance layer **without re-solving**: the 48 P2 bid months bid with its
cohorts state-owned. Its balancing-mechanism effect is 9-18 GWh a year of nuclear turn-down and
is declared negligible.

This check is the original one (its functions, unchanged) plus exactly that exception and no
other: a row whose ``support_type`` is in ``--exempt-support-types`` (``RAB``: Sizewell C) may
have left the state since the solve. For a generator holding such rows only,

* its capacity must match exactly, and
* the state capacity the solve read may exceed today's by no more than the exempt rows'
  capacity (and never fall below it);

its state share and commissioning year then follow and are not compared. Every other generator,
column and case is checked exactly as before. The stamp lists the exempt generators and MW.

A saved row whose generator has left the register with no state capacity (domestic batteries,
taken out of every ownership case on 3 Oct 2026) is dropped: the bids read a missing generator
as private, so it is the same bid. A row with any state capacity is never dropped.

Usage::

    python scripts/public_ownership/verify_ownership_tables_2050.py REGISTER PORTFOLIO STAMP \\
        --shares TABLE::CASE [...] [--years TABLE::CASE ...] --exempt-support-types RAB
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

from scripts.public_ownership import ownership_shares as shares_mod  # noqa: E402
from scripts.public_ownership import verify_ownership_tables as base  # noqa: E402
from scripts.public_ownership.register_io import read_register  # noqa: E402

#: The only support type decision 32 exempts (Sizewell C's regulated asset base).
ALLOWED_EXEMPTIONS = ("RAB",)


def exempt_capacity(register: pd.DataFrame, support_types) -> pd.DataFrame:
    """Per (case, generator): the capacity of rows whose support type is exempt."""
    rows = register.loc[register["support_type"].astype(str).isin(list(support_types))]
    if rows.empty:
        return pd.DataFrame(columns=["case_id", "generator", "exempt_mw"])
    rows = rows.assign(generator=rows["asset_id"].map(shares_mod.generator_name),
                       exempt_mw=pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0))
    return rows.groupby(["case_id", "generator"], as_index=False)["exempt_mw"].sum()


def compare_with_exemption(expected: pd.DataFrame, saved: pd.DataFrame, exempt: pd.DataFrame) -> dict:
    """``verify_ownership_tables.compare_tables`` on the non-exempt generators, and the bound on the exempt ones."""
    keys = ["case_id", "generator"]
    tag = lambda f: f.assign(case_id=f["case_id"].astype(str), generator=f["generator"].astype(str))  # noqa: E731
    expected, saved, exempt = tag(expected), tag(saved), tag(exempt)
    marked = set(map(tuple, exempt[keys].to_numpy().tolist()))
    is_exempt = lambda f: np.array([(c, g) in marked for c, g in zip(f["case_id"], f["generator"])], dtype=bool)  # noqa: E731
    report = base.compare_tables(expected.loc[~is_exempt(expected)], saved.loc[~is_exempt(saved)], base.SHARE_COLUMNS)
    e = expected.loc[is_exempt(expected)].merge(saved.loc[is_exempt(saved)], on=keys, suffixes=("_expected", "_saved"),
                                                how="outer", indicator="where").merge(exempt, on=keys, how="left")
    failures = []
    for r in e.itertuples(index=False):
        if r.where != "both":
            failures.append({"generator": r.generator, "case_id": r.case_id, "why": "missing from one table"})
            continue
        if not np.isclose(r.capacity_mw_expected, r.capacity_mw_saved, rtol=base.REL_TOLERANCE, atol=base.ABS_TOLERANCE):
            failures.append({"generator": r.generator, "case_id": r.case_id, "why": "capacity differs"})
        gap = float(r.state_capacity_mw_saved) - float(r.state_capacity_mw_expected)
        if gap < -base.ABS_TOLERANCE or gap > float(r.exempt_mw) + 1e-6:
            failures.append({"generator": r.generator, "case_id": r.case_id, "why": f"state MW moved by {gap:.6f}, "
                             f"outside [0, {float(r.exempt_mw):.6f}] (the exempt rows)"})
    both = e.loc[e["where"].eq("both")]
    report["exempt"] = {
        "generators": int(len(both)),
        "exempt_rows_mw": float(both["exempt_mw"].sum()),
        "state_mw_the_solve_read_above_today": float((both["state_capacity_mw_saved"] - both["state_capacity_mw_expected"]).sum()),
        "failures": failures,
        "reason": ("decision 32 (3 Oct 2026): Sizewell C's RAB left the state in the finance layer after the P2 "
                   "bid months were solved; its BM effect (9-18 GWh/yr) is declared negligible, no re-solve"),
    }
    if failures:
        report.setdefault("failed_columns", []).append("exempt_bound")
    return report


def drop_private_rows_left_the_register(expected: pd.DataFrame, saved: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Saved rows of generators the register no longer holds, with no state capacity, are dropped.

    The balancing bids read a generator missing from the share table as private
    (``ownership_shares``), so a row with zero state capacity and no register row is the same
    bid as no row: no ownership moved. The user's decision of 3 Oct 2026 took domestic
    batteries out of every ownership case's register (consumer flexibility). A saved row with
    any state capacity is never dropped, so a real ownership change still fails.
    """
    keys = ["case_id", "generator"]
    present = set(map(tuple, expected[keys].astype(str).to_numpy().tolist()))
    mine = np.array([(c, g) not in present for c, g in zip(saved["case_id"].astype(str), saved["generator"].astype(str))],
                    dtype=bool)
    zero = pd.to_numeric(saved["state_capacity_mw"], errors="coerce").fillna(0.0).abs() <= base.ABS_TOLERANCE
    drop = mine & zero.to_numpy()
    return saved.loc[~drop], int(drop.sum())


ACCEPTED_COLUMNS = ("run_group", "table", "carrier", "cap_gross_mw", "decided", "reason")


def load_accepted_changes(path: str | None, run_group: str | None) -> pd.DataFrame:
    """The declared ownership changes a run group's solved bid months are accepted with (user, 15 Nov 2026)."""
    if not path:
        return pd.DataFrame(columns=list(ACCEPTED_COLUMNS))
    table = pd.read_csv(path)
    missing = set(ACCEPTED_COLUMNS) - set(table.columns)
    if missing:
        raise SystemExit(f"{path} lacks columns {sorted(missing)}")
    if table[["decided", "reason"]].isna().any().any():
        raise SystemExit(f"{path}: every accepted change needs who decided it and why")
    return table.loc[table["run_group"].astype(str).eq(str(run_group))]


def split_declared(expected: pd.DataFrame, saved: pd.DataFrame, register: pd.DataFrame, table_path: str,
                   accepted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Take generators of a declared carrier out of both tables, within the declared cap of state MW moved.

    A generator qualifies only when every register row of it is of the declared carrier. A carrier whose gross
    state MW moved exceeds its cap (or any undeclared generator) still fails the check; the moves are recorded.
    """
    rows = accepted.loc[[str(t) in Path(table_path).stem for t in accepted["table"]]] if len(accepted) else accepted
    if rows.empty:
        return expected, saved, []
    carrier = register.assign(generator=register["asset_id"].map(shares_mod.generator_name)).groupby("generator")[
        "carrier"].agg(lambda c: "|".join(sorted(set(map(str, c)))))
    keys = ["case_id", "generator"]
    both = expected[keys + ["state_capacity_mw"]].merge(saved[keys + ["state_capacity_mw"]], on=keys, how="outer",
                                                         suffixes=("_now", "_solved")).fillna(
        {"state_capacity_mw_now": 0.0, "state_capacity_mw_solved": 0.0})
    both["carrier"] = both["generator"].map(carrier)
    both["moved"] = (both["state_capacity_mw_now"] - both["state_capacity_mw_solved"]).abs()
    records, take = [], set()
    for row in rows.itertuples(index=False):
        hit = both.loc[both["carrier"].eq(str(row.carrier))]
        gross = float(hit["moved"].sum())
        if gross > float(row.cap_gross_mw) + 1e-6:
            raise SystemExit(f"{Path(table_path).name}: {row.carrier} moved {gross:,.1f} state MW, above its declared cap "
                             f"{float(row.cap_gross_mw):,.0f}; the accepted change no longer describes the register")
        take |= set(hit["generator"].astype(str))
        records.append({"carrier": str(row.carrier), "gross_state_mw_moved": round(gross, 3),
                        "net_state_mw_change": round(float((hit["state_capacity_mw_now"]
                                                            - hit["state_capacity_mw_solved"]).sum()), 3),
                        "cap_gross_mw": float(row.cap_gross_mw), "decided": str(row.decided), "reason": str(row.reason)})

    def keep(frame):
        return frame.loc[~frame["generator"].astype(str).isin(take)]
    return keep(expected), keep(saved), records


def check_shares(table_path: str, cases: list[str], register_path: str, support_types,
                 accepted: pd.DataFrame | None = None) -> dict:
    """:func:`verify_ownership_tables.check_table` for a share table, with the exemption and declared changes."""
    saved = pd.read_csv(table_path)
    saved = saved.loc[saved["case_id"].astype(str).isin(cases)]
    columns = (list(shares_mod.REGISTER_COLUMNS) + list(shares_mod.OPTIONAL_REGISTER_COLUMNS)
               + ["support_type", "carrier"])
    register = read_register(register_path, columns=columns, cases=cases)
    register = register.loc[register["case_id"].astype(str).isin(cases)]
    expected = shares_mod.generator_state_shares(register)
    expected, saved, declared = split_declared(expected, saved, register, table_path,
                                               accepted if accepted is not None else load_accepted_changes(None, None))
    saved, dropped = drop_private_rows_left_the_register(expected, saved)
    report = compare_with_exemption(expected, saved, exempt_capacity(register, support_types))
    report["private_rows_no_longer_in_the_register"] = dropped
    if declared:
        report["accepted_changes"] = declared
    report["exempt"]["support_types"] = list(support_types)
    absent = sorted(set(cases) - set(expected["case_id"].astype(str)))
    if absent:
        report.setdefault("failed_columns", []).append("case_missing_from_register")
        report["absent_cases"] = absent
    report.update(kind="shares", table=table_path, cases=cases)
    report["ok"] = not report.get("failed_columns") and not report["missing"] and not report["extra"]
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("register")
    parser.add_argument("portfolio")
    parser.add_argument("stamp")
    parser.add_argument("--shares", nargs="*", default=[])
    parser.add_argument("--years", nargs="*", default=[])
    parser.add_argument("--exempt-support-types", nargs="*", default=[])
    parser.add_argument("--accepted-changes", default=None,
                        help="declared ownership changes the solved months are accepted with (CSV, user decisions)")
    parser.add_argument("--run-group", default=None, help="the run group whose rows of --accepted-changes apply")
    args = parser.parse_args(argv)
    accepted = load_accepted_changes(args.accepted_changes, args.run_group)
    unknown = sorted(set(args.exempt_support_types) - set(ALLOWED_EXEMPTIONS))
    if unknown:
        raise SystemExit(f"only {ALLOWED_EXEMPTIONS} may be exempted (decision 32); refused {unknown}")
    reports = [check_shares(table, cases, args.register, args.exempt_support_types, accepted)
               for table, cases in base.parse_pairs(args.shares).items()]
    reports += [base.check_table("years", table, cases, args.register, args.portfolio)
                for table, cases in base.parse_pairs(args.years).items()]
    for report in reports:
        status = "matches" if report["ok"] else "DIFFERS"
        extra = f"; exempt {report['exempt']['generators']} generators" if "exempt" in report else ""
        print(f"{report['kind']:6s} {report['table']} for {report['cases']}: {status}{extra}")
    if not all(report["ok"] for report in reports):
        print(json.dumps([r for r in reports if not r["ok"]], indent=1, default=str), file=sys.stderr)
        raise SystemExit(base.REBUILD_HINT)
    Path(args.stamp).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stamp).write_text(json.dumps({"register": args.register, "tables": reports}, indent=1, default=str),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
