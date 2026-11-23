"""Put the balancing-bid cases beside the market cases, with the bid stack's effect split out (item BM1).

A case on a new balancing-bid stack (``balancing_bids: state_at_cost``) is
projected by its own run group, on that group's months, beside a ``private``
that reproduces the group's own forecast case. Its bill therefore splits exactly
into three parts, each a difference of two projected bills:

* **market private**, the bill everyone is compared with;
* **private-on-stack decomposition term** (``private_bid_stack_decomposition_term``),
  private on the group's months less private on the market months;
* **ownership effect on the stack**, the case less private, both on the group's
  months: the same ownership accounting the market group does, on this dispatch.

**The decomposition term is not a saving and never a result to present**
(audit 26 Sep 2026).  "Private on the P2 stack" is not a feasible case: the bid
stack exists only because the state owns the plant and bids it at cost, so a bill
with that stack and no public owner describes nothing that could happen.  Its
-14.84 GBP (22 Sep outputs) is an intermediate term of an exact telescoping sum,
kept under a name that says so; until this audit it was called "balancing effect"
and printed as "(bids)", which invited reading it as what the bids save.

The comparison a reader wants is the case against its own scenario on the market
months (``public_ownership@state_bids_at_cost`` against ``public_ownership``):
the same owner, the bids changed. That is ``bid_stack_effect``, **the P2
increment and the only headline** (``headline_metric`` names it; -0.47 on the
22 Sep outputs). Ownership and the balancing institution are substitutes; this
is one case on one stack, never a saving to add to another case's.

The per-unit decomposition (``scripts/market/decompose_public_bm_bids.py``) is an
independent estimate of the bid-stack effect built from the balancing LPs'
own per-unit costs. It is converted here on the bill's own denominators, never on
a constant: bid and offer changes are recovered over final demand like the
BSUoS constraint line, and the offsetting CfD settlement over CfD-eligible supply
like the CfD line. The gap between the two is reported with the terms the
decomposition leaves out.

Usage::

    python scripts/public_ownership/merge_balancing_groups.py MARKET_BY_CASE MARKET_SUMMARY OUT_CSV OUT_JSON
        --group GROUP BY_CASE SUMMARY DECOMPOSITION [--group ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.levers import PRIVATE_SCENARIO, base_case  # noqa: E402

VALUE = "annual_gbp_per_3100kwh_ex_vat"
#: Two bills are comparable only on the same household, prices and recovery volumes (the recovery
#: volumes through bill_recovery.recovery_signature: each line's entry on the two-volume basis, F18).
COMPARABLE_KEYS = ("benchmark_kwh", "price_basis", "model_rebasing_factor")
#: What the decomposition does not see, and so what the cross-check gap can hold.
DECOMPOSITION_OMITS = (
    "the RO line, whose load factors come from the balancing networks",
    "the re-priced state CfD strike, which recovers required revenue over a changed delivered volume",
    "interconnector re-trading and any change in the CfD negative-price and economic-curtailment rules",
)


def check_comparable(market_summary: dict, group_summary: dict, group: str) -> None:
    """Refuse a merge of two bills that do not share a household, price basis or recovery volume."""
    for summary, name in ((market_summary, "the market group"), (group_summary, group)):
        if summary.get("private_reproduces_forecast_case") is not True:
            raise ValueError(f"{name}: private does not reproduce its forecast case, so its cases cannot be compared")
    for key in COMPARABLE_KEYS:
        a, b = market_summary.get(key), group_summary.get(key)
        same = (np.isclose(float(a), float(b), rtol=1e-9, atol=0.0)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b)
        if not same:
            raise ValueError(f"{group}: {key} is {b!r} against the market group's {a!r}; the bills are not comparable")
    from scripts.public_ownership import bill_recovery

    a, b = bill_recovery.recovery_signature(market_summary), bill_recovery.recovery_signature(group_summary)
    if a != b:
        differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise ValueError(f"{group}: recovery volumes differ from the market group's ({differ[:5]}); "
                         "the bills are not comparable")


def _lines(by_case: pd.DataFrame, case_id: str, where: str) -> pd.Series:
    rows = by_case.loc[by_case["case_id"].eq(case_id)]
    if rows.empty:
        raise ValueError(f"{where} has no rows for case {case_id!r}")
    if rows["component_id"].duplicated().any():
        raise ValueError(f"{where}: case {case_id!r} repeats a component")
    return rows.set_index("component_id")[VALUE].astype(float)


def market_twin(case_id: str, case_levers: dict | None, market_summary: dict) -> str:
    """The market-group case a balancing-group case is measured against: equal on every lever but the bids.

    A preset or a toggle-grid case on the bid stack (29 Sep 2026) has its own twin on
    the market months (``public_ownership@grid_rate-plus-1pp`` for
    ``public_ownership@state_bids_at_cost_grid_rate-plus-1pp``), so the bid-stack
    effect isolates the bids.  Without levers, or without a twin, the scenario.

    Task Y14: more than one market case on the same levers is refused rather than
    resolved by order.  The FES 2050 list's P2 shares differ only on the gated L4
    lever, so a summary whose ``levers`` leave gated levers out would make every
    share look like ``public_ownership``; the merge must not pick one silently.
    (On the CP30 2030 outputs every balancing case has exactly one twin.)
    """
    if case_levers:
        wanted = {k: v for k, v in case_levers.items() if k != "balancing_bids"}
        matches = []
        for other, entry in (market_summary.get("cases") or {}).items():
            levers = {k: v for k, v in (entry.get("levers") or {}).items() if k != "balancing_bids"}
            if levers and levers == wanted:
                matches.append(other)
        if len(matches) > 1:
            raise ValueError(f"case {case_id!r} has {len(matches)} market twins on the same levers {matches[:4]}: the "
                             "summaries do not carry the lever that tells them apart (a gated lever such as "
                             "state_deployment_share must be in each case's 'levers')")
        if matches:
            return matches[0]
    return base_case(case_id)


def balancing_components(market: pd.DataFrame, group: pd.DataFrame, case_id: str, twin: str | None = None) -> pd.DataFrame:
    """The case's bill line by line: market private, the private-on-stack decomposition term, ownership effect."""
    columns = {
        "case_on_group": _lines(group, case_id, "the balancing group"),
        "private_on_group": _lines(group, PRIVATE_SCENARIO, "the balancing group"),
        "private_on_market": _lines(market, PRIVATE_SCENARIO, "the market group"),
        "base_on_market": _lines(market, twin or base_case(case_id), "the market group"),
    }
    # Like bills must match exactly: the case against its base, private against
    # private. Private may lack an ownership-only line the case carries (the
    # Capacity Market public financing adjustment, public_cfd_at_cost): it owns
    # nothing, so that line is zero on its bill, not a sign of a different bill.
    ids = {name: set(series.index) for name, series in columns.items()}
    pairs = (("case_on_group", "base_on_market"), ("private_on_group", "private_on_market"))
    for first, second in pairs:
        if ids[first] != ids[second]:
            raise ValueError(f"case {case_id!r}: {first} and {second} do not carry the same lines; "
                             f"differing lines {sorted(ids[first] ^ ids[second])}")
    if not ids["private_on_group"] <= ids["case_on_group"]:
        raise ValueError(f"case {case_id!r}: private carries lines the case does not: "
                         f"{sorted(ids['private_on_group'] - ids['case_on_group'])}")
    lines = columns["case_on_group"].index
    frame = pd.DataFrame({name: series.reindex(lines, fill_value=0.0) for name, series in columns.items()})
    # A decomposition term, not a feasible case: never present it as a saving.
    frame["private_bid_stack_decomposition_term"] = frame["private_on_group"] - frame["private_on_market"]
    frame["ownership_effect_on_stack"] = frame["case_on_group"] - frame["private_on_group"]
    frame["bid_stack_effect"] = frame["case_on_group"] - frame["base_on_market"]
    labels = group.loc[group["case_id"].eq(case_id)].drop_duplicates("component_id").set_index("component_id")
    frame.insert(0, "component", labels["component"].reindex(frame.index))
    frame.insert(1, "stack_group", labels["stack_group"].reindex(frame.index))
    frame.insert(0, "case_id", case_id)
    return frame.reset_index()


def _total(summary: dict, case_id: str, where: str) -> dict:
    case = (summary.get("cases") or {}).get(case_id)
    if case is None:
        raise ValueError(f"{where} summary has no case {case_id!r}")
    return case


#: How far a bill's lines may sit from its summary total (GBP a household).
LINE_TOTAL_TOLERANCE_GBP = 1e-6


def check_lines_match_totals(components: pd.DataFrame, bills: dict[str, dict], case_id: str) -> None:
    """Each of the four bills' lines, grossed up by its VAT, must sum to its summary total.

    The three-part split of a case's total (market private + balancing effect +
    ownership effect) is a telescoping sum and holds whatever the totals are, so
    checking it proves nothing. What can go wrong is that the line-by-line
    decomposition and the headline totals come from different projections (a
    stale CSV beside a fresh summary, a line dropped on the reindex, a
    different VAT rate), and that is what this checks, per bill.
    """
    for column, bill in bills.items():
        ex_vat, vat, total = float(bill["ex_vat_gbp"]), float(bill["vat_gbp"]), float(bill["total_gbp_inc_vat"])
        if abs(ex_vat + vat - total) > LINE_TOTAL_TOLERANCE_GBP:
            raise ValueError(f"{case_id}: {column} summary total is not ex VAT plus VAT")
        lines = float(components[column].sum())
        if abs(lines - ex_vat) > LINE_TOTAL_TOLERANCE_GBP:
            raise ValueError(f"{case_id}: {column} lines sum to {lines:.6f} but its summary ex-VAT total is "
                             f"{ex_vat:.6f}; the by-case CSV and summary JSON are not the same projection")


def _household_rates(summary: dict, vat_rate: float) -> dict:
    """GBP a household inc VAT per GBP of system cost, on each recovery the bill uses.

    F19 (7 Oct 2026): through the bill's own recovery entries (scripts/public_ownership/bill_recovery.py):
    a balancing payment is recovered through the BSUoS constraint line (a GSP rate on the two-volume
    basis), the CfD offset through the CfD line. The keys keep their old names; on a D1 (2050) bill they
    are final demand and CfD-eligible sales, as before.
    """
    from scripts.public_ownership import bill_recovery

    vat = 1.0 + vat_rate
    return {"final_demand": bill_recovery.household_per_system(summary, "bsuos_constraints") * vat,
            "cfd_eligible_supply": bill_recovery.household_per_system(summary, "cfd") * vat}


#: The state strike bases that hold its required revenue whatever it delivers, so its
#: balancing transfer returns through the strike and its CfD settlement does not move.
REVENUE_HOLDING_STRIKE_BASES = ("at_cost",)
#: Where a fixed-strike state's balance can come back to the bill: through the routes.
STATE_BALANCE_RETAINED = "retained_by_treasury"


def cross_check(decomposition: dict, bid_stack_effect_inc_vat: float, summary: dict, vat_rate: float,
                levers: dict | None = None) -> dict:
    """The decomposition's estimate of the bid-stack effect, on the bill's denominators, against the bill's.

    Which terms reach the bill depends on how the state's own CfDs are paid (30 Sep
    2026). At cost, the state's required revenue is held, so neither its balancing
    transfer nor its CfD settlement moves the bill. On a fixed strike (the
    treasury_first preset's ``auction_clearing``) the state is settled on its
    delivered volume: the BSUoS line loses the transfer to the state and the CfD
    line loses the state's strike - reference on the volume the stack moves off it,
    and both land on the state's balance (the Exchequer's, under retention). That is
    an incidence effect: investors and the resource saving do not move.
    """
    year = decomposition["year"]
    rates = _household_rates(summary, vat_rate)
    levers = levers or {}
    strike_basis = levers.get("cfd_strike_basis") or REVENUE_HOLDING_STRIKE_BASES[0]
    balancing = (float(year["payment_to_private_counterparties_gbp"]) + float(year["delta_offer_cost_gbp"]))
    offset = float(year["offsetting_private_cfd_settlement_gbp"])
    to_state = float(year["transfer_to_the_state_gbp"])
    state_cfd = year.get("state_cfd_settlement_at_fixed_strike_gbp")
    estimate = balancing * rates["final_demand"] + offset * rates["cfd_eligible_supply"]
    gap_can_hold = list(DECOMPOSITION_OMITS)
    if strike_basis in REVENUE_HOLDING_STRIKE_BASES:
        basis = "state strike at cost: its balancing transfer and CfD settlement stay off the bill"
    elif state_cfd is None:
        basis = (f"state strike {strike_basis}: the decomposition predates its state-side CfD term; "
                 "rerun build_public_bm_bid_decomposition")
        gap_can_hold.insert(0, "the state's own CfD settlement and balancing transfer on a fixed strike")
    else:
        estimate += to_state * rates["final_demand"] + float(state_cfd) * rates["cfd_eligible_supply"]
        basis = (f"state strike {strike_basis}: the state's balancing transfer and its CfD settlement on the "
                 "volume the stack moves off it both reach the bill")
        if levers.get("surplus_recycling") != STATE_BALANCE_RETAINED:
            gap_can_hold.insert(0, "the share of the state's changed balance that the surplus routes return")
        if strike_basis != "auction_clearing":
            gap_can_hold.insert(0, "the financing saving taken off the state's strike (the state term is valued "
                                   "at the auction top-up)")
    return {
        "decomposition_basis": basis,
        "state_cfd_strike_basis": strike_basis,
        "decomposition_private_bids_and_offers_gbp": balancing,
        "decomposition_offsetting_private_cfd_settlement_gbp": offset,
        "decomposition_transfer_to_the_state_gbp": to_state,
        "decomposition_state_cfd_settlement_at_fixed_strike_gbp": (None if state_cfd is None else float(state_cfd)),
        "decomposition_estimate_gbp_per_household_inc_vat": estimate,
        "bill_bid_stack_effect_gbp_per_household_inc_vat": bid_stack_effect_inc_vat,
        "gap_gbp_per_household_inc_vat": bid_stack_effect_inc_vat - estimate,
        "gap_can_hold": gap_can_hold,
        "household_rate_per_gbp_inc_vat": rates,
    }


def merge(market_by_case: pd.DataFrame, market_summary: dict, groups: list[dict]) -> tuple[pd.DataFrame, dict]:
    """``groups``: ``{"name", "by_case", "summary", "decomposition"}`` per balancing group."""
    frames, cases = [], {}
    for group in groups:
        name, summary = group["name"], group["summary"]
        check_comparable(market_summary, summary, name)
        for case_id in summary["cases"]:
            if case_id == PRIVATE_SCENARIO:
                continue
            # Task Y14: one group per P2 share, so a case must come from exactly one of them.
            if case_id in cases:
                raise ValueError(f"case {case_id!r} is projected by both {cases[case_id]['run_group']!r} and "
                                 f"{name!r}; each balancing case belongs to one group")
            twin = market_twin(case_id, (summary["cases"][case_id] or {}).get("levers"), market_summary)
            components = balancing_components(market_by_case, group["by_case"], case_id, twin)
            frames.append(components)
            case = _total(summary, case_id, name)
            private_group = _total(summary, PRIVATE_SCENARIO, name)
            private_market = _total(market_summary, PRIVATE_SCENARIO, "the market group")
            base_market = _total(market_summary, twin, "the market group")
            # The meaningful check: the line-level split and the totals come from the same four bills.
            # (The three parts below add to the total by construction: a telescoping sum.)
            check_lines_match_totals(components, {
                "case_on_group": case, "private_on_group": private_group,
                "private_on_market": private_market, "base_on_market": base_market,
            }, case_id)
            total = float(case["total_gbp_inc_vat"])
            decomposition_term = float(private_group["total_gbp_inc_vat"]) - float(private_market["total_gbp_inc_vat"])
            ownership = total - float(private_group["total_gbp_inc_vat"])
            vat_rate = float(case["vat_gbp"]) / float(case["ex_vat_gbp"])
            bid_stack = total - float(base_market["total_gbp_inc_vat"])
            entry = {
                "run_group": name,
                "forecast_case": summary.get("forecast_case"),
                "base_case": twin,
                "scenario_base_case": base_case(case_id),
                "total_gbp_inc_vat": total,
                "private_on_market_gbp_inc_vat": float(private_market["total_gbp_inc_vat"]),
                "private_on_group_gbp_inc_vat": float(private_group["total_gbp_inc_vat"]),
                "base_on_market_gbp_inc_vat": float(base_market["total_gbp_inc_vat"]),
                # The P2 increment: the only figure to present.
                "headline_metric": "bid_stack_effect_gbp_inc_vat",
                # Not a feasible case (private cannot bid the state's stack); a term
                # of the telescoping sum only, never a saving.
                "private_bid_stack_decomposition_term_gbp_inc_vat": decomposition_term,
                "ownership_effect_on_stack_gbp_inc_vat": ownership,
                "bid_stack_effect_gbp_inc_vat": bid_stack,
                "delta_vs_private_gbp_inc_vat": total - float(private_market["total_gbp_inc_vat"]),
                "levers": case.get("levers"),
            }
            if group.get("decomposition") is not None:
                entry["cross_check"] = cross_check(group["decomposition"], bid_stack, summary, vat_rate,
                                                   case.get("levers"))
            cases[case_id] = entry
    components = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return components, {
        "market_forecast_case": market_summary.get("forecast_case"),
        "cases": cases,
        "notes": ("Each case = market private + private-on-stack decomposition term + ownership effect on the "
                  "stack, exactly (a telescoping sum of the four bills; the merge checks instead that each bill's "
                  "lines sum to its total). The decomposition term is private on a bid stack that only exists "
                  "because the state owns and bids the plant: not a feasible case, never a saving, never presented. "
                  "bid_stack_effect is the case against its own scenario on the market months, the P2 increment "
                  "and the headline. Ownership and the balancing institution are substitutes: never add this "
                  "effect to another case's saving."),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("market_by_case")
    parser.add_argument("market_summary")
    parser.add_argument("out_csv")
    parser.add_argument("out_json")
    parser.add_argument("--group", nargs=4, action="append", required=True,
                        metavar=("NAME", "BY_CASE", "SUMMARY", "DECOMPOSITION"))
    args = parser.parse_args(argv)
    groups = [{"name": name, "by_case": pd.read_csv(by_case),
               "summary": json.loads(Path(summary).read_text(encoding="utf-8")),
               "decomposition": json.loads(Path(decomposition).read_text(encoding="utf-8"))}
              for name, by_case, summary, decomposition in args.group]
    components, summary = merge(pd.read_csv(args.market_by_case),
                                json.loads(Path(args.market_summary).read_text(encoding="utf-8")), groups)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for case_id, entry in summary["cases"].items():
        check = entry.get("cross_check") or {}
        print(f"{case_id}: P2 increment against {entry['base_case']} {entry['bid_stack_effect_gbp_inc_vat']:+.2f} "
              f"(bill {entry['total_gbp_inc_vat']:.2f}; decomposition {entry['private_on_market_gbp_inc_vat']:.2f} "
              f"{entry['private_bid_stack_decomposition_term_gbp_inc_vat']:+.2f} [private on the stack, not a case] "
              f"{entry['ownership_effect_on_stack_gbp_inc_vat']:+.2f} [ownership])"
              + (f", decomposition {check['decomposition_estimate_gbp_per_household_inc_vat']:+.2f}" if check else ""))


if __name__ == "__main__":
    main()
