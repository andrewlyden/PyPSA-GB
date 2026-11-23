"""Every headline move against the last accepted results has a declared reason (user, 15 Nov 2026).

The reference is ``data/public_ownership/results_reference/{group}_headline.csv``: the headline cases' bill and
household total of the last results the user accepted. ``declared_moves.csv`` beside it lists the changes made since,
each matching groups and cases by pattern (``fnmatch``) with its reason and decision date. A move of more than
``TOLERANCE_GBP`` in a case no declared row covers fails the rule; every move is written to the report with the
reason that covers it.

    python -m scripts.public_ownership.headline_moves --totals G_household_totals.csv --group G --out report.csv
    python -m scripts.public_ownership.headline_moves --totals G_household_totals.csv --group G --write-reference

After the user accepts a set of results: write the references from them and empty ``declared_moves.csv``.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path("data/public_ownership/results_reference")
DECLARED = REFERENCE_DIR / "declared_moves.csv"
S3 = "public_ownership@fully_nationalised_market_value"
HEADLINE_CASES = ("private", "public_control", "public_ownership", S3, S3 + "_central_dispatch")
COLUMNS = ("bill_inc_vat", "household_total_delta_vs_private_gbp_inc_vat")
TOLERANCE_GBP = 0.05


def headline(totals: pd.DataFrame) -> pd.DataFrame:
    rows = totals.loc[totals["case_id"].isin(HEADLINE_CASES), ["case_id", *[c for c in COLUMNS if c in totals]]]
    return rows.set_index("case_id")


def declared_reason(declared: pd.DataFrame, group: str, case: str) -> str:
    reasons = [f"{r.reason} ({r.decided})" for r in declared.itertuples(index=False)
               if fnmatch.fnmatch(group, str(r.group_pattern)) and fnmatch.fnmatch(case, str(r.case_pattern))]
    return "; ".join(reasons)


def compare(now: pd.DataFrame, reference: pd.DataFrame, declared: pd.DataFrame, group: str) -> pd.DataFrame:
    out = []
    for case in sorted(set(now.index) | set(reference.index)):
        for column in COLUMNS:
            new = float(now.at[case, column]) if case in now.index and column in now else float("nan")
            old = float(reference.at[case, column]) if case in reference.index and column in reference else float("nan")
            move = new - old
            if pd.isna(move) or abs(move) > TOLERANCE_GBP:
                reason = declared_reason(declared, group, case)
                out.append({"group": group, "case_id": case, "column": column, "reference": old, "now": new,
                            "move": move, "declared_reason": reason, "explained": bool(reason)})
    return pd.DataFrame(out, columns=["group", "case_id", "column", "reference", "now", "move", "declared_reason",
                                      "explained"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--totals", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    parser.add_argument("--declared", default=str(DECLARED))
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args(argv)
    now = headline(pd.read_csv(args.totals))
    reference_path = Path(args.reference_dir) / f"{args.group}_headline.csv"
    if args.write_reference:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        now.reset_index().to_csv(reference_path, index=False)
        print(f"reference written: {reference_path}")
        return
    if not reference_path.exists():
        raise SystemExit(f"no reference for {args.group}: {reference_path} (write it from accepted results)")
    declared = (pd.read_csv(args.declared) if Path(args.declared).exists()
                else pd.DataFrame(columns=["group_pattern", "case_pattern", "reason", "decided"]))
    report = compare(now, pd.read_csv(reference_path).set_index("case_id"), declared, args.group)
    unexplained = report.loc[~report["explained"]]
    for r in report.itertuples(index=False):
        print(f"{r.group} {r.case_id} {r.column}: {r.reference:.2f} -> {r.now:.2f} ({r.move:+.2f}) "
              f"{'[' + r.declared_reason + ']' if r.explained else 'UNDECLARED'}")
    if not unexplained.empty:
        raise SystemExit(f"{len(unexplained)} headline move(s) in {args.group} have no declared reason; add a row to "
                         f"{args.declared} naming the change, or find the bug")
    report.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
