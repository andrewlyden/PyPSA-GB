"""Money invariants of a group's bills that a ledger closure cannot see (user, 15 Nov 2026: checks in the workflow).

Both bugs below closed every ledger, because the same money was counted consistently twice or skipped consistently:

* ``dpa``: a buying case that owns Net Zero Teesside while the bill keeps its ``remaining_ccs`` line must return the
  DPA it receives (``nationalised_dpa_receipts_on_owned_plant`` = minus the kept line, within 1%).
* ``sizewell``: a case whose state takes on a share of new build beyond the state's existing share of Sizewell C, and
  whose own state does not hold Sizewell C's RAB, must carry the public-rate return (``sizewell_c_rab_public_rate_
  adjustment`` non-zero) wherever the private bill has a Sizewell C RAB line and the adjustment is switched on.

    python -m scripts.public_ownership.verify_bill_invariants --group G --out X.json
        [--tariff-components ..._nationalised_bill_components.csv --tariff-summary ..._nationalised_bill_summary.json]
        [--bill-by-case ..._consumer_bill_by_ownership_case.csv --public-account ..._public_account_summary.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VALUE = "annual_gbp_per_3100kwh_ex_vat"
DPA_LINE = "nationalised_dpa_receipts_on_owned_plant"
KEPT_DPA_LINE = "remaining_ccs"
SIZEWELL_RAB_LINE = "sizewell_c_rab"
SIZEWELL_ADJUSTMENT = "sizewell_c_rab_public_rate_adjustment"


def dpa_failures(components: pd.DataFrame, summary: dict) -> list[str]:
    out = []
    for case, block in (summary.get("cases") or {}).items():
        diagnostics = (block or {}).get("diagnostics") or {}
        owned = float(diagnostics.get("dpa_on_owned_costed_rows_gbp", 0.0) or 0.0)
        # The DPA on plant the case leaves private is paid through the kept line as today (the 2025-law bound buys
        # only the pre-legislation share of Net Zero Teesside), so the credit is the owned share of the kept line.
        private = float(diagnostics.get("dpa_paid_through_kept_line_gbp", 0.0) or 0.0)
        rows = components.loc[components["case_id"].eq(case)].set_index("component_id")[VALUE]
        kept = float(rows.get(KEPT_DPA_LINE, 0.0))
        if owned <= 0.0 or kept == 0.0:
            continue
        share = owned / (owned + private)
        credit = float(rows.get(DPA_LINE, 0.0))
        if abs(credit + share * kept) > 0.01 * abs(kept):
            out.append(f"{case}: owns {share:.0%} of the DPA plant (GBP {owned / 1e6:,.0f} m a year) but returns "
                       f"{credit:+.2f} against {share:.0%} of the kept remaining_ccs line's {kept:+.2f} GBP a household")
    return out


def sizewell_failures(by_case: pd.DataFrame, account: dict) -> list[str]:
    from scripts.public_ownership import sizewell_c_rab

    settings = (account.get("settings") or {}).get(sizewell_c_rab.CONFIG_KEY) or {}
    if not settings.get("enabled"):
        return []
    private = by_case.loc[by_case["case_id"].eq("private")].set_index("component_id")[VALUE]
    if float(private.get(SIZEWELL_RAB_LINE, 0.0)) == 0.0:
        return []
    out = []
    for case, block in (account.get("cases") or {}).items():
        if case == "private" or case not in set(by_case["case_id"]):
            continue
        share = float((block or {}).get("state_new_build_share") or 0.0)
        if sizewell_c_rab.refinanced_share(settings, share) <= 0.0:
            continue
        if float((block or {}).get("rab_support_state_owned_gbp", 0.0) or 0.0) != 0.0:
            continue                                        # the case's state holds the RAB: the levy is replaced
        line = float(by_case.loc[by_case["case_id"].eq(case) & by_case["component_id"].eq(SIZEWELL_ADJUSTMENT),
                                 VALUE].sum())
        if line == 0.0:
            out.append(f"{case}: state share {share:.0%} of new build, Sizewell C's RAB private, but no public-rate "
                       "return on its levy")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tariff-components", default=None)
    parser.add_argument("--tariff-summary", default=None)
    parser.add_argument("--bill-by-case", default=None)
    parser.add_argument("--public-account", default=None)
    args = parser.parse_args(argv)
    report = {"group": args.group, "checks": {}}
    if args.tariff_components:
        failures = dpa_failures(pd.read_csv(args.tariff_components),
                                json.loads(Path(args.tariff_summary).read_text(encoding="utf-8")))
        report["checks"]["dpa"] = {"ok": not failures, "failures": failures}
    if args.bill_by_case:
        failures = sizewell_failures(pd.read_csv(args.bill_by_case),
                                     json.loads(Path(args.public_account).read_text(encoding="utf-8")))
        report["checks"]["sizewell"] = {"ok": not failures, "failures": failures}
    report["ok"] = all(c["ok"] for c in report["checks"].values())
    for name, check in report["checks"].items():
        print(f"{args.group} {name}: {'ok' if check['ok'] else 'FAILS'}" + "".join(f"\n  {f}" for f in check["failures"]))
    if not report["ok"]:
        raise SystemExit(f"bill invariants fail for {args.group}")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
