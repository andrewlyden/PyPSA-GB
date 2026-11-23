"""P1 and P2 under central dispatch beside the market year: three parts per case and the headline (20 Oct 2026).

An institution twin (``reference_run_groups.<twin>.institution_of: <market group>``) projects
private, P1, P2 and their presets on the market group's central-dispatch year (institution I3,
FTI's settlement). Each case's I3 bill against the private *market* bill (I1) splits exactly
into two measured steps, each a difference of two projected bills:

* **institution given owner**: private-I3 less private-I1, the institution's step on the
  private fleet (a feasible reform: central dispatch with today's owners);
* **owner given institution**: the case less private, both under I3, the ownership
  accounting the twin group does on the central-dispatch year.

The headline (user decision 8, 20 Oct 2026) is the case under I3 against private-I1, with the
two increments beside it. The P1/P2-I1 saving (the market group's) is reported for information
only: **a saving measured under I1 and a saving measured on private-I3 are never added** (the
two start from the same bill, so their sum describes no case: ownership and the dispatch
institution are substitutes for the constraint payments, nationalised_institutions). Every sum
here is a chain of steps whose ends meet (:func:`chain`), which refuses any other.

The three-party identity (``investor_ledger``) closes on each step and on their sum, to GBP 1:
the institution step is :func:`investor_ledger.institution_step_parties_market` on the two
registers' private rows, the bills' moved lines, the FTI settlement's cross-border and real-time
quantities and the two ledgers' Exchequer positions (carbon, VAT, BECCS's carbon CfD); the
ownership step is the twin group's own Exchequer ledger row.

Usage::

    python scripts/public_ownership/merge_institution_groups.py MARKET_GROUP TWIN_GROUP OUT_CSV OUT_JSON
        --po-dir resources/public_ownership --market-forecast-summary ... --twin-forecast-summary ...
        --fti-settlement ... --assumptions ... --recovery-demand ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership import investor_ledger as investors  # noqa: E402
from scripts.public_ownership.levers import PRIVATE_SCENARIO  # noqa: E402

VALUE = "annual_gbp_per_3100kwh_ex_vat"
I1, I3 = "I1", "I3"
HEADLINE_METRIC = "i3_bill_vs_private_i1_gbp_inc_vat"
LINE_TOLERANCE_GBP = 1e-6
#: The forbidden sum, stated on every output.
SUBSTITUTES_RULE = ("Never add a saving measured under I1 (P1-I1, P2-I1, P3) to a saving measured on private under "
                    "I3: both start from the private market bill, so their sum is no case. Ownership and the dispatch "
                    "institution are substitutes for the constraint payments; the headline is one chain of two steps.")


# ── chains: the only sums this module makes ───────────────────────────────────

@dataclass(frozen=True)
class Step:
    """A change between two (owner, institution) bills, e.g. (private, I1) -> (private, I3)."""
    start: tuple[str, str]
    end: tuple[str, str]
    value: float
    label: str = ""


def chain(*steps: Step) -> Step:
    """The sum of steps whose ends meet; refuses anything else (a P2-I1 saving plus a private-I3 saving).

    Two savings that start from the same bill (private-I1) cannot be added: the second does not
    start where the first ends. This is the code form of the substitutes rule.
    """
    if not steps:
        raise ValueError("an empty chain")
    for first, second in zip(steps, steps[1:]):
        if first.end != second.start:
            raise ValueError(f"refused: {first.label or first.end} ends at {first.end} but {second.label or second.start} "
                             f"starts at {second.start}. {SUBSTITUTES_RULE}")
    return Step(steps[0].start, steps[-1].end, float(sum(s.value for s in steps)),
                " + ".join(s.label for s in steps if s.label))


# ── inputs ────────────────────────────────────────────────────────────────────

def _lines(by_case: pd.DataFrame, case_id: str, where: str) -> pd.Series:
    rows = by_case.loc[by_case["case_id"].eq(case_id)]
    if rows.empty:
        raise ValueError(f"{where} has no rows for case {case_id!r}")
    if rows["component_id"].duplicated().any():
        raise ValueError(f"{where}: case {case_id!r} repeats a component")
    return rows.set_index("component_id")[VALUE].astype(float)


def check_twin(market_summary: Mapping, twin_summary: Mapping, twin: str) -> None:
    """The two bills share a household, prices and recovery volumes, and the twin's bill is central dispatch."""
    from scripts.public_ownership.merge_balancing_groups import check_comparable

    check_comparable(market_summary, twin_summary, twin)
    if twin_summary.get("institution") != "central_dispatch":
        raise ValueError(f"{twin}: its bill is not a central-dispatch bill ({twin_summary.get('institution')!r})")
    if market_summary.get("institution", "market") != "market":
        raise ValueError(f"the market group's bill settles under {market_summary.get('institution')!r}, not the market")


def step_line_changes(market_by_case: pd.DataFrame, twin_by_case: pd.DataFrame, market_summary: Mapping,
                      tolerance: float = 1e-9) -> dict[str, float]:
    """System GBP per bill line private-I3 moves against private-I1, each at its own recovery entry."""
    from scripts.public_ownership import bill_recovery

    i1 = _lines(market_by_case, PRIVATE_SCENARIO, "the market group")
    i3 = _lines(twin_by_case, PRIVATE_SCENARIO, "the institution twin")
    if set(i1.index) != set(i3.index):
        raise ValueError(f"private carries different lines on the two bills: {sorted(set(i1.index) ^ set(i3.index))}")
    out = {}
    for line, hh in (i3 - i1.reindex(i3.index)).items():
        if abs(float(hh)) <= tolerance:
            continue
        out[str(line)] = bill_recovery.system_from_household(market_summary, str(line), float(hh))
    return out


def measured_quantities(settlement: pd.DataFrame, twin_forecast: Mapping, market_forecast: Mapping,
                        gb_shares: Mapping[str, float], realtime_at_cost_register_gbp: float,
                        exchequer_paid_revenue_change_gbp: float) -> dict:
    """The quantities :func:`investor_ledger.institution_step_parties_market` names, system GBP.

    From the year FTI settlement (YEAR, GBP m): the Stage 1 schedules' payment abroad, the cables'
    rent per cable (central with the make-whole) and the central real-time link trade; from the
    bills' balancing summaries: the I3 line's make-whole, forecast-error charge and premium, and the
    I1 line's re-trading (as the bills carry them).
    """
    from scripts.market.fti_settlement import IC_RENT_CENTRAL_PREFIX, IC_RENT_MARKET_PREFIX

    def year(row: str) -> float:
        if row not in settlement.index:
            raise KeyError(f"the FTI settlement has no row {row!r}; rebuild it (scripts.market.fti_settle_year)")
        return float(pd.to_numeric(pd.Series([settlement.at[row, "YEAR"]]), errors="coerce").iloc[0]) * 1e6

    cables = sorted({str(r)[len(p):] for r in settlement.index for p in (IC_RENT_MARKET_PREFIX, IC_RENT_CENTRAL_PREFIX)
                     if str(r).startswith(p)})
    if unknown := [c for c in cables if c not in gb_shares]:
        raise ValueError(f"cables with rent but no GB share in the bill's regime table: {unknown}")
    change = {c: (year(f"{IC_RENT_CENTRAL_PREFIX}{c}") if f"{IC_RENT_CENTRAL_PREFIX}{c}" in settlement.index else 0.0)
              - (year(f"{IC_RENT_MARKET_PREFIX}{c}") if f"{IC_RENT_MARKET_PREFIX}{c}" in settlement.index else 0.0)
              for c in cables}
    i3 = twin_forecast.get("balancing_costs") or {}
    i1 = market_forecast.get("balancing_costs") or {}
    if i3.get("institution") != "central_dispatch":
        raise ValueError("the twin's forecast bill carries no central-dispatch constraint line")
    retrading = float(i1.get("interconnector_retrading_cost_gbp") or 0.0) if i1.get("include_interconnector_retrading") else 0.0
    make_whole = float(i3["interconnector_make_whole_gbp"])
    if abs(make_whole - year("central_ic_make_whole_gbp_m")) > 1.0:
        raise ValueError("the I3 bill's interconnector make-whole is not the FTI settlement's: rebuild the extraction")
    return {
        "schedule_trade_change_gbp": year("central_ic_schedule_trade_cost_info_gbp_m")
        - year("market_ic_schedule_trade_cost_info_gbp_m"),
        "rent_change_before_make_whole_gbp": year("central_ic_congestion_rent_before_make_whole_info_gbp_m")
        - year("market_ic_congestion_rent_gbp_m"),
        "make_whole_gbp": make_whole,
        "gb_half_revenue_change_gbp": float(sum(gb_shares[c] * v for c, v in change.items())),
        "foreign_half_revenue_change_gbp": float(sum((1.0 - gb_shares[c]) * v for c, v in change.items())),
        "realtime_links_gbp": (year("central_ic_physical_trade_cost_info_gbp_m")
                               - year("central_ic_schedule_trade_cost_info_gbp_m")
                               + year("central_counter_trade_premium_gbp_m")),
        "market_retrading_gbp": retrading,
        "forecast_error_charge_gbp": float(i3["forecast_error_charge_gbp"]),
        "counter_trade_premium_gbp": float(i3.get("counter_trade_premium_gbp") or 0.0),
        "realtime_at_cost_register_gbp": float(realtime_at_cost_register_gbp),
        "exchequer_paid_revenue_change_gbp": float(exchequer_paid_revenue_change_gbp),
    }


def realtime_at_cost_on_register(transfers: pd.DataFrame | None) -> float:
    """The real-time correction the twin's settlement pays register units at cost (imports excluded)."""
    if transfers is None or transfers.empty or "realtime_at_cost_gbp" not in transfers:
        return 0.0
    from scripts.public_ownership.calculate_financing_scenarios import DEFAULT_EXCLUDED_CARRIERS

    kept = ~transfers["carrier"].astype(str).isin(DEFAULT_EXCLUDED_CARRIERS)
    return float(pd.to_numeric(transfers.loc[kept, "realtime_at_cost_gbp"], errors="coerce").fillna(0.0).sum())


def own_parties(row: Mapping) -> dict:
    """A case's ownership step as the twin group's Exchequer ledger measured it (against private-I3)."""
    for key in ("consumer_bill_change_system_gbp", "investor_change_vs_private_parts_gbp",
                "resource_saving_vs_private_parts_gbp"):
        if key not in row:
            raise ValueError(f"the twin's Exchequer ledger row {row.get('case_id')!r} carries no {key}")
    return {"bill_change_system_gbp": float(row["consumer_bill_change_system_gbp"]),
            "investor_parts_gbp": dict(row["investor_change_vs_private_parts_gbp"]),
            "resource_parts_gbp": dict(row["resource_saving_vs_private_parts_gbp"])}


# ── the merge ─────────────────────────────────────────────────────────────────

def merge(*, market_group: str, twin_group: str, market_by_case: pd.DataFrame, market_summary: Mapping,
          twin_by_case: pd.DataFrame, twin_summary: Mapping, market_ledger: Mapping, twin_ledger: Mapping,
          position_market: Mapping, position_central: Mapping, measured: Mapping[str, float],
          conv: Mapping) -> tuple[pd.DataFrame, dict]:
    """Every twin case beside the market year: the lines, the three parts, the headline and the identity per step."""
    from scripts.public_ownership.exchequer_ledger import ledger_row, per_household

    check_twin(market_summary, twin_summary, twin_group)
    ledger_market = market_ledger["cases"]
    ledger_twin = twin_ledger["cases"]
    # The institution step on private (a feasible reform: central dispatch with today's owners).
    exch = ledger_row(ledger_twin[PRIVATE_SCENARIO], ledger_market[PRIVATE_SCENARIO], conv)
    carbon = float(ledger_twin[PRIVATE_SCENARIO]["carbon_receipts_gbp"]) - float(
        ledger_market[PRIVATE_SCENARIO]["carbon_receipts_gbp"])
    step = investors.institution_step_parties_market(
        line_changes=step_line_changes(market_by_case, twin_by_case, market_summary),
        position_central=position_central, position_market=position_market, carbon_change_gbp=carbon,
        measured=measured)
    step_vat, step_exchequer = float(exch["vat_change_vs_private_gbp"]), float(exch["exchequer_net_vs_private_gbp"])
    step_row = investors.three_party_row(step, vat_change_gbp=step_vat, exchequer_change_gbp=step_exchequer,
                                         case_id="private (the institution step, I3 against I1)")
    total = lambda summary, case: float(summary["cases"][case]["total_gbp_inc_vat"])  # noqa: E731
    private_i1, private_i3 = total(market_summary, PRIVATE_SCENARIO), total(twin_summary, PRIVATE_SCENARIO)
    institution = Step((PRIVATE_SCENARIO, I1), (PRIVATE_SCENARIO, I3), private_i3 - private_i1,
                       "institution given owner (private-I3 against private-I1)")
    frames, cases = [], {}
    m_private = _lines(market_by_case, PRIVATE_SCENARIO, "the market group")
    t_private = _lines(twin_by_case, PRIVATE_SCENARIO, "the institution twin")
    for case_id in twin_summary["cases"]:
        case_i3 = total(twin_summary, case_id)
        own = Step((PRIVATE_SCENARIO, I3), (case_id, I3), case_i3 - private_i3,
                   "owner given institution (the case against private, both I3)")
        headline = chain(institution, own)
        if abs(headline.value - (case_i3 - private_i1)) > LINE_TOLERANCE_GBP:
            raise ValueError(f"{case_id}: the two steps do not add to the I3 bill against private-I1")
        row = ledger_twin.get(case_id)
        if row is None:
            raise ValueError(f"the twin's Exchequer ledger has no case {case_id!r}")
        parties = investors.combine(step, own_parties(row))
        identity = investors.three_party_row(
            parties, vat_change_gbp=step_vat + float(row["vat_change_vs_private_gbp"]),
            exchequer_change_gbp=step_exchequer + float(row["exchequer_net_vs_private_gbp"]), case_id=case_id)
        i1_case = case_id in market_summary["cases"]
        entry = {
            "run_group": twin_group, "market_group": market_group,
            "levers": (twin_summary["cases"][case_id] or {}).get("levers"),
            "headline_metric": HEADLINE_METRIC,
            HEADLINE_METRIC: headline.value,
            "institution_given_owner_gbp_inc_vat": institution.value,
            "owner_given_institution_gbp_inc_vat": own.value,
            "i3_bill_gbp_inc_vat": case_i3,
            "private_i3_bill_gbp_inc_vat": private_i3,
            "private_i1_bill_gbp_inc_vat": private_i1,
            # The same case under I1, for information: its saving is never added to the institution step.
            "i1_bill_gbp_inc_vat": total(market_summary, case_id) if i1_case else None,
            "ownership_under_i1_info_gbp_inc_vat": (total(market_summary, case_id) - private_i1) if i1_case else None,
            "institution_given_this_owner_info_gbp_inc_vat": (case_i3 - total(market_summary, case_id))
            if i1_case else None,
            "three_party_vs_private_i1": identity,
            "three_party_vs_private_i1_per_household": investors.household_columns(
                identity, lambda gbp: per_household(gbp, conv)),
            "three_party_ownership_step": {k: row.get(k) for k in (
                "consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp", "investor_change_vs_private_gbp",
                "resource_saving_vs_private_gbp", "three_party_identity_residual_gbp")},
        }
        cases[case_id] = entry
        lines = _lines(twin_by_case, case_id, "the institution twin")
        frame = pd.DataFrame({
            "private_on_market": m_private.reindex(lines.index, fill_value=0.0),
            "private_on_twin": t_private.reindex(lines.index, fill_value=0.0),
            "case_on_twin": lines,
        })
        frame["case_on_market"] = (_lines(market_by_case, case_id, "the market group").reindex(lines.index, fill_value=0.0)
                                   if i1_case else np.nan)
        frame["institution_given_owner"] = frame["private_on_twin"] - frame["private_on_market"]
        frame["owner_given_institution"] = frame["case_on_twin"] - frame["private_on_twin"]
        frame["i3_vs_private_i1"] = frame["case_on_twin"] - frame["private_on_market"]
        labels = twin_by_case.loc[twin_by_case["case_id"].eq(case_id)].drop_duplicates("component_id").set_index(
            "component_id")
        frame.insert(0, "component", labels["component"].reindex(frame.index) if "component" in labels else "")
        frame.insert(0, "case_id", case_id)
        check_lines(frame, twin_summary["cases"][case_id], market_summary["cases"][PRIVATE_SCENARIO], case_id)
        frames.append(frame.reset_index())
    components = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = {
        "market_group": market_group, "institution_twin": twin_group,
        "market_forecast_case": market_summary.get("forecast_case"),
        "twin_forecast_case": twin_summary.get("forecast_case"),
        "headline_metric": HEADLINE_METRIC,
        "substitutes_rule": SUBSTITUTES_RULE,
        "institution_step": {
            "bill_gbp_inc_vat": institution.value,
            "three_party": step_row,
            "three_party_per_household": investors.household_columns(step_row, lambda gbp: per_household(gbp, conv)),
            "bill_lines_system_gbp": step["bill_lines_gbp"],
            "measured_gbp": step["measured_gbp"],
            "exchequer_parts_gbp": exch["exchequer_net_vs_private_parts_gbp"],
        },
        "cases": cases,
        "price_basis": ("bills GBP per 3,100 kWh household inc VAT, report-year money; the identity in system GBP, "
                        "model price year (2024), as the Exchequer ledger"),
        "notes": ("Each case's I3 bill against private-I1 = institution given owner (private-I3 against private-I1) + "
                  "owner given institution (the case against private, both I3), a chain of two measured steps; the "
                  "three-party identity closes on each to GBP 1 and so on the sum. " + SUBSTITUTES_RULE),
    }
    return components, summary


def check_lines(frame: pd.DataFrame, case: Mapping, private_market: Mapping, case_id: str) -> None:
    """Each bill's lines sum to its summary's ex-VAT total (the lines and the totals are one projection)."""
    for column, bill in (("case_on_twin", case), ("private_on_market", private_market)):
        lines = float(frame[column].sum())
        if abs(lines - float(bill["ex_vat_gbp"])) > LINE_TOLERANCE_GBP:
            raise ValueError(f"{case_id}: {column} lines sum to {lines:.6f}, its summary to {float(bill['ex_vat_gbp']):.6f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("market_group", "twin_group", "out_csv", "out_json"):
        parser.add_argument(name)
    parser.add_argument("--po-dir", default="resources/public_ownership")
    parser.add_argument("--market-forecast-summary", required=True)
    parser.add_argument("--twin-forecast-summary", required=True)
    parser.add_argument("--fti-settlement", required=True)
    parser.add_argument("--assumptions", required=True, help="the bill assumptions (interconnector regimes, VAT)")
    parser.add_argument("--recovery-demand", required=True)
    args = parser.parse_args(argv)
    from scripts.market.fti_settlement import gb_half_shares, load_interconnector_rule
    from scripts.public_ownership.exchequer_ledger import conversion
    from scripts.public_ownership.register_io import read_register

    po = Path(args.po_dir)
    read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    market_summary = read(po / f"{args.market_group}_consumer_bill_by_ownership_case_summary.json")
    twin_summary = read(po / f"{args.twin_group}_consumer_bill_by_ownership_case_summary.json")
    positions = {}
    for group in (args.market_group, args.twin_group):
        rows = read_register(po / f"{group}_asset_finance_register.parquet", cases=[PRIVATE_SCENARIO])
        positions[group] = investors.register_position(rows, PRIVATE_SCENARIO)
    transfers_path = po / f"{args.twin_group}_asset_settlement_transfers.csv"
    transfers = pd.read_csv(transfers_path) if transfers_path.exists() else None
    measured = measured_quantities(
        pd.read_csv(args.fti_settlement, index_col=0), read(args.twin_forecast_summary),
        read(args.market_forecast_summary), gb_half_shares(load_interconnector_rule(args.assumptions)),
        realtime_at_cost_on_register(transfers),
        positions[args.twin_group]["carbon_removal_cfd_gbp"] - positions[args.market_group]["carbon_removal_cfd_gbp"])
    components, summary = merge(
        market_group=args.market_group, twin_group=args.twin_group,
        market_by_case=pd.read_csv(po / f"{args.market_group}_consumer_bill_by_ownership_case.csv"),
        market_summary=market_summary,
        twin_by_case=pd.read_csv(po / f"{args.twin_group}_consumer_bill_by_ownership_case.csv"),
        twin_summary=twin_summary, market_ledger=read(po / f"{args.market_group}_exchequer_ledger.json"),
        twin_ledger=read(po / f"{args.twin_group}_exchequer_ledger.json"),
        position_market=positions[args.market_group], position_central=positions[args.twin_group],
        measured=measured, conv=conversion(read(args.assumptions), read(args.recovery_demand), market_summary))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    step = summary["institution_step"]
    print(f"institution given owner (private-I3 against private-I1): {step['bill_gbp_inc_vat']:+.2f} GBP a household")
    for case_id, entry in summary["cases"].items():
        print(f"{case_id}: I3 {entry['i3_bill_gbp_inc_vat']:.2f}, against private-I1 {entry[HEADLINE_METRIC]:+.2f} "
              f"(institution {entry['institution_given_owner_gbp_inc_vat']:+.2f}, owner "
              f"{entry['owner_given_institution_gbp_inc_vat']:+.2f})")


if __name__ == "__main__":
    main()
