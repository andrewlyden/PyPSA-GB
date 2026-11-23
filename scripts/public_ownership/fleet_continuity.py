"""Fleet continuity between the acquisition year and a later model year (user, 14 Nov 2026).

For any later model year Y (2035, 2040, 2045, 2050, ...): by technology group, the MW a buying case owns in Y's
register against what the acquisition year's purchase bought (``acquisition_baseline``) and against the plant Y's
fleet has that the purchase could have bought (pre-legislation rows, outside every exclusion). The run fails when a
buying case owns more than its purchase bought, or less than both the purchase and Y's fleet allow (beyond one row's
rounding), so plant paid for is never left private while Y's fleet still runs it, and plant never paid for is never
owned. Bought plant Y's fleet no longer runs is reported (``bought_not_running_mw``): the purchase's lives against Y's
fleet lives, which the life table aligns.

    python -m scripts.public_ownership.fleet_continuity --register <Y register> --model-year Y --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.public_ownership.acquisition_baseline import (  # noqa: E402
    BASELINE, NO_ROUND, carrier_group, load_baseline, load_round_caps,
)

BUYING_CASE = "public_ownership@fully_nationalised_market_value"
COLUMNS = ["case_id", "asset_id", "carrier", "finance_vintage", "state_owned", "capacity_mw", "ownership_exclusion",
           "cfd_round", "build_year"]
TOLERANCE_MW = 1.0


class ContinuityError(ValueError):
    """A later year owns plant its purchase did not buy, or leaves bought plant its fleet still runs private."""


def continuity_table(register: pd.DataFrame, bought: Mapping[str, float], *, case_id: str = BUYING_CASE,
                     excluded_groups=("energy_from_waste",), valuation: pd.DataFrame | None = None,
                     model_year: int | None = None, round_caps: Mapping[str, Mapping[str, float]] | None = None,
                     acquisition_year: int = 2030) -> pd.DataFrame:
    """One row per technology group: bought, eligible, owned, and the gaps (MW).

    Eligibility is :func:`acquisition_baseline.eligible_rows`, the rule the financing step owns rows by (independent
    review, 15 Nov 2026: a looser "available" count here failed correct runs and pushed toward owning unbought plant).
    ``excluded_groups``: groups the case leaves out of its purchase (energy from waste centrally), not checked.
    ``valuation``: the acquisition year's legacy valuation; with it, ``repaid_not_running_mw`` is the bought plant
    whose repayment still runs in ``model_year`` beyond what the case owns.
    """
    from scripts.public_ownership.acquisition_baseline import NO_ROUND, _round_keys, eligible_rows

    bought = {g: v for g, v in bought.items() if g not in set(excluded_groups)}
    rows = register.loc[register["case_id"].astype(str).eq(case_id)].copy()
    if rows.empty:
        raise ContinuityError(f"the register has no {case_id} rows")
    rows["group"] = rows["carrier"].map(carrier_group)
    rows["mw"] = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0)
    excluded = (rows["ownership_exclusion"].fillna("").astype(str).str.strip().ne("")
                if "ownership_exclusion" in rows.columns else pd.Series(False, index=rows.index))
    rows["excluded"] = excluded | rows["carrier"].astype(str).isin(["EU_market", "EU_import", "load_shedding"])
    eligible = pd.Series(eligible_rows(rows, round_caps=round_caps, acquisition_year=acquisition_year),
                         index=rows.index)
    owned = rows["state_owned"].astype(str).str.lower().isin(["true", "1"])
    pre = rows["finance_vintage"].astype(str).isin(["legacy", "in_construction"])
    key = pd.Series(_round_keys(rows), index=rows.index)
    # Independent review (15 Nov 2026): a private eligible row is ownable only while its own contract has room left
    # (half the row of slack, as acquired_rows allows), so a binding contract cap never fails a correct run.
    owned_by_round = rows.loc[owned & pre].groupby([rows["group"], key])["mw"].sum()
    caps = {str(g): {str(k): float(v) for k, v in c.items()} for g, c in (round_caps or {}).items()}

    def _ownable(i) -> bool:
        k = key.at[i]
        if round_caps is None or k == NO_ROUND:
            return True
        g = rows.at[i, "group"]
        return (float(owned_by_round.get((g, k), 0.0)) + 0.5 * float(rows.at[i, "mw"])
                <= float(caps.get(str(g), {}).get(k, 0.0)) + TOLERANCE_MW)

    out = []
    alive = {}
    if valuation is not None and model_year is not None:
        v = valuation.assign(group=valuation["carrier"].map(carrier_group))
        live = pd.to_numeric(v["retirement_year"], errors="coerce") > float(model_year)
        alive = v.loc[live].groupby("group")["capacity_mw"].sum().to_dict()
    for group in sorted((set(rows.loc[pre, "group"]) | set(bought)) - set(excluded_groups)):
        g = rows["group"].eq(group)
        eligible_mw = float(rows.loc[g & eligible, "mw"].sum())
        own = float(rows.loc[g & owned & pre, "mw"].sum())
        b = float(bought.get(group, 0.0))
        largest = float(rows.loc[g & eligible, "mw"].max()) if (g & eligible).any() else 0.0
        out.append({"group": group, "bought_mw": b, "eligible_mw": eligible_mw, "owned_mw": own,
                    "owned_not_eligible_mw": float(rows.loc[g & owned & pre & ~eligible, "mw"].sum()),
                    "owned_above_bought_mw": max(0.0, own - b),
                    "eligible_left_private_mw": float(rows.loc[[i for i in rows.index[g & eligible & ~owned]
                                                                if _ownable(i)], "mw"].sum()),
                    "headroom_mw": max(0.0, b - own),
                    "bought_not_running_mw": max(0.0, b - eligible_mw),
                    "repaid_not_running_mw": (max(0.0, float(alive.get(group, 0.0)) - own) if alive else np.nan),
                    "largest_row_mw": largest})
    return pd.DataFrame(out)


def contract_ownership(register: pd.DataFrame, round_caps: Mapping[str, Mapping[str, float]], *,
                       case_id: str = BUYING_CASE) -> pd.DataFrame:
    """Owned MW by group and contract round against what the purchase bought of that contract (15 Nov 2026).

    Plant on a contract the purchase never bought must be unowned; a bought contract is owned within its MW (half the
    largest row of slack). Plant on no contract (expired, merchant) is covered by the group check.
    """
    rows = register.loc[register["case_id"].astype(str).eq(case_id)].copy()
    rows = rows.loc[rows["state_owned"].astype(str).str.lower().isin(["true", "1"])
                    & rows["finance_vintage"].astype(str).isin(["legacy", "in_construction"])]
    rows["group"] = rows["carrier"].map(carrier_group)
    rows["round"] = rows["cfd_round"].fillna("").astype(str).replace("", NO_ROUND) if "cfd_round" in rows.columns \
        else NO_ROUND
    rows["mw"] = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0)
    rows = rows.loc[rows["round"].ne(NO_ROUND)]
    out = []
    for (group, contract), part in rows.groupby(["group", "round"]):
        bought = round_caps.get(str(group), {}).get(str(contract))
        owned = float(part["mw"].sum())
        out.append({"group": group, "round": contract, "owned_mw": owned,
                    "bought_mw": float(bought) if bought is not None else 0.0, "bought": bought is not None,
                    "largest_row_mw": float(part["mw"].max())})
    return pd.DataFrame(out, columns=["group", "round", "owned_mw", "bought_mw", "bought", "largest_row_mw"])


def contract_failures(table: pd.DataFrame) -> list[str]:
    out = []
    for r in table.itertuples(index=False):
        if not r.bought and r.owned_mw > TOLERANCE_MW:
            out.append(f"{r.group}: owns {r.owned_mw:,.0f} MW on {r.round}, a contract the purchase did not buy")
        elif r.bought and r.owned_mw > r.bought_mw + max(TOLERANCE_MW, 0.5 * r.largest_row_mw):
            out.append(f"{r.group}: owns {r.owned_mw:,.0f} MW on {r.round} against {r.bought_mw:,.0f} MW bought")
    return out


def failures(table: pd.DataFrame) -> list[str]:
    """Groups that break continuity: owning plant outside the purchase, owning more than was bought (beyond half the
    largest row), or leaving eligible plant private while the purchase still has headroom for it."""
    out = []
    for r in table.itertuples(index=False):
        slack = max(TOLERANCE_MW, 0.5 * r.largest_row_mw)
        if r.owned_not_eligible_mw > TOLERANCE_MW:
            out.append(f"{r.group}: owns {r.owned_not_eligible_mw:,.0f} MW the purchase did not buy")
        if r.owned_above_bought_mw > slack:
            out.append(f"{r.group}: owns {r.owned_mw:,.0f} MW against {r.bought_mw:,.0f} MW bought")
        if r.headroom_mw > slack and r.eligible_left_private_mw > slack:
            out.append(f"{r.group}: leaves {r.eligible_left_private_mw:,.0f} MW of bought plant private with "
                       f"{r.headroom_mw:,.0f} MW of the purchase unused")
    return out


def main(argv=None) -> None:
    from scripts.public_ownership.register_io import read_register

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--register", required=True)
    parser.add_argument("--model-year", type=int, required=True)
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--case", default=BUYING_CASE)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-only", action="store_true", help="write the table, never fail")
    parser.add_argument("--exclude-groups", nargs="*", default=["energy_from_waste"],
                        help="groups the case leaves out of its purchase")
    parser.add_argument("--valuation", default=None, help="the acquisition year's legacy valuation CSV")
    args = parser.parse_args(argv)
    register = read_register(args.register, columns=COLUMNS, cases=[args.case])
    valuation = pd.read_csv(args.valuation) if args.valuation else None
    from scripts.public_ownership.acquisition_baseline import baseline_acquisition_year

    table = continuity_table(register, load_baseline(args.baseline), case_id=args.case,
                             excluded_groups=tuple(args.exclude_groups), valuation=valuation,
                             model_year=args.model_year, round_caps=load_round_caps(args.baseline),
                             acquisition_year=baseline_acquisition_year(args.baseline))
    problems = failures(table)
    contracts = contract_ownership(register, load_round_caps(args.baseline), case_id=args.case)
    problems += contract_failures(contracts)
    doc = {"model_year": args.model_year, "case_id": args.case, "baseline": args.baseline,
           "passes": not problems, "failures": problems,
           "groups": table.round(1).to_dict(orient="records"),
           "contracts": contracts.round(1).to_dict(orient="records"),
           "rule": ("a buying case owns only plant its purchase bought (built by the purchase year on no or a bought "
                    "contract, or later on a bought contract), within each group's and contract's bought MW (half a "
                    "row's slack), and leaves no such plant private while the purchase has headroom")}
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(table.round(0).to_string(index=False))
    if problems and not args.report_only:
        raise SystemExit("fleet continuity fails:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
