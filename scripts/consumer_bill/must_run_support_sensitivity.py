"""Bill sensitivity: the support CP30's must-run plant without a contract would need (user decision, 8 Oct 2026).

The CP30 2030 scenario holds biomass and biogas to the pathway's energy, so Lynemouth
(whose investment contract ends in 2027) and the merchant biogas proxies run at a loss
the central bill never pays for. This sensitivity pays each such unit's calendar-2030
going-forward shortfall and recovers it on the bill line its support would ride on:

    shortfall = max(0, fuel and carbon + fixed and variable O&M + insurance + TNUoS
                       - market revenue - support - Capacity Market)

per asset, from the private case of the market run group's register. Each group's
system total is converted to GBP per household through its recovery line's own entry
(:mod:`scripts.consumer_bill.line_recovery`, item F18), and VAT at the bill's rate.
The groups, the columns and the exclusions are data
(``data/market/consumer_bill_benchmarks/must_run_support_sensitivity.json``).

Usage::

    python scripts/consumer_bill/must_run_support_sensitivity.py REGISTER BILL_SUMMARY SPEC OUT_CSV OUT_JSON
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.bill_recovery import consumption_kwh, household_per_system  # noqa: E402
from scripts.public_ownership.register_io import read_register  # noqa: E402

ID_COLUMNS = ["asset_id", "carrier", "support_type", "finance_vintage", "capacity_mw", "dispatch_mwh"]


def asset_shortfalls(register: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """One row per asset in a spec group: its revenue, going-forward cost and shortfall (GBP a year)."""
    basis = spec["cost_basis"]
    rows = []
    for group in spec["groups"]:
        take = register.loc[register["carrier"].astype(str).eq(group["carrier"])
                            & register["support_type"].astype(str).isin(group["support_types"])].copy()
        if take.empty:
            continue
        num = lambda column: pd.to_numeric(take[column], errors="coerce").fillna(0.0)
        take["revenue_gbp"] = sum(num(column) for column in basis["revenue_columns"])
        take["going_forward_cost_gbp"] = sum(num(column) for column in basis["cost_columns"])
        take["going_forward_cash_gbp"] = take["revenue_gbp"] - take["going_forward_cost_gbp"]
        take["shortfall_gbp"] = (-take["going_forward_cash_gbp"]).clip(lower=0.0)
        take["group"] = group["id"]
        take["recovery_line"] = group["recovery_line"]
        rows.append(take[["group", "recovery_line", *ID_COLUMNS, "revenue_gbp", "going_forward_cost_gbp",
                          "going_forward_cash_gbp", "shortfall_gbp"]])
    if not rows:
        raise ValueError("no register rows match the sensitivity's groups")
    return pd.concat(rows, ignore_index=True)


def summarise(assets: pd.DataFrame, bill: dict, spec: dict) -> dict:
    """System and household totals per group, each on its own recovery line."""
    vat_rate = float(bill["vat_rate"])
    groups = {}
    for group_id, rows in assets.groupby("group", sort=False):
        line = str(rows["recovery_line"].iloc[0])
        system_gbp = float(rows["shortfall_gbp"].sum())
        ex_vat = system_gbp * household_per_system(bill, line, case_id=spec.get("case_id", "private"))
        groups[group_id] = {
            "recovery_line": line,
            "assets": int(len(rows)),
            "assets_with_shortfall": int(rows["shortfall_gbp"].gt(0).sum()),
            "capacity_mw": round(float(rows["capacity_mw"].astype(float).sum()), 1),
            "dispatch_mwh": round(float(rows["dispatch_mwh"].astype(float).sum()), 0),
            "shortfall_system_gbp": round(system_gbp, 0),
            "gbp_per_household_ex_vat": round(ex_vat, 2),
            "gbp_per_household_inc_vat": round(ex_vat * (1 + vat_rate), 2),
        }
    ex_vat = sum(g["gbp_per_household_ex_vat"] for g in groups.values())
    central = float(bill["total_gbp_inc_vat"])
    return {
        "component_id": spec["component_id"],
        "label": spec["label"],
        "decided": spec["decided"],
        "cost_basis": spec["cost_basis"]["description"],
        "groups": groups,
        "shortfall_system_gbp": round(sum(g["shortfall_system_gbp"] for g in groups.values()), 0),
        "gbp_per_household_ex_vat": round(ex_vat, 2),
        "gbp_per_household_inc_vat": round(ex_vat * (1 + vat_rate), 2),
        "central_bill_gbp_inc_vat": round(central, 2),
        "sensitivity_bill_gbp_inc_vat": round(central + ex_vat * (1 + vat_rate), 2),
        "benchmark_kwh": consumption_kwh(bill),
        "vat_rate": vat_rate,
        "excluded": spec["excluded"],
        "scope_note": spec["scope_note"],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("register", help="the market run group's asset finance register")
    parser.add_argument("bill_summary", help="the forecast bill summary (two-volume basis)")
    parser.add_argument("spec", help="must_run_support_sensitivity.json")
    parser.add_argument("out_csv")
    parser.add_argument("out_json")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    bill = json.loads(Path(args.bill_summary).read_text(encoding="utf-8"))
    register = read_register(args.register, cases=[spec.get("case_id", "private")])
    assets = asset_shortfalls(register, spec)
    summary = summarise(assets, bill, spec)
    assets.sort_values(["group", "shortfall_gbp"], ascending=[True, False]).to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for group_id, group in summary["groups"].items():
        print(f"{group_id}: GBP {group['shortfall_system_gbp'] / 1e6:,.1f} m on {group['recovery_line']}, "
              f"GBP {group['gbp_per_household_inc_vat']:.2f}/household inc VAT")
    print(f"bill {summary['central_bill_gbp_inc_vat']:,.2f} -> {summary['sensitivity_bill_gbp_inc_vat']:,.2f} "
          f"GBP/household inc VAT (+{summary['gbp_per_household_inc_vat']:.2f})")


if __name__ == "__main__":
    main()
