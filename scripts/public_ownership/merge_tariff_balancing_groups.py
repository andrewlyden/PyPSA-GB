"""Put a tariff case on its own balancing-bid stack beside the same case on the market months (item BM1, P3).

P3 ("the public power system", market machinery kept) acquires the legacy fleet,
so its bill is the acquisition tariff (``nationalised_tariff.py``), not the
market bill. A tariff balancing group (``reference_run_groups.<group>.bid_case``)
solves the year again with every unit the case owns bidding at cost, curtailment
ordered by the decision-20 LIFO tie-break, and tariffs that case on those months,
with every tariff case that owns the same fleet (the fully_nationalised family:
only the purchase price basis, thermal basis or annuity term differ, so each bids
the same stack). This merge sets each beside the same case's tariff on the market
months, one entry per case, and names the bid case's figure as the headline:

* **market private**, the bill everyone is compared with;
* **private-on-stack decomposition term** (``private_bid_stack_decomposition_term``),
  private on the group's months less private on the market months. As in the P2
  merge this is **not a feasible case and never a saving**: nobody but the state
  bids that stack. It is a term of the telescoping sum only;
* **ownership effect on the stack**, the tariff less private, both on the group's
  months;
* **bid-stack effect** (``bid_stack_effect``), the tariff on the group's months
  less the same case's tariff on the market months: the same owner, the same
  purchase, only the balancing bids changed. It moves through the physical
  dispatch only (fuel and carbon, variable O&M, cross-border trade and re-trading),
  because the tariff replaces every market-construct line.

**The purchase price is struck on the market year.** The tariff's acquisition
lines annuitise a price valued from each legacy asset's cash flows on the group's
own settlement, and on the P3 months that settlement pays the legacy fleet at P3's
bids, a world that exists only after the purchase. So the price lines differ
between the two tariffs for a reason that has nothing to do with the bid stack.
They are split out (``purchase_price_line_effect``), with both prices beside them,
and the headline (``bid_stack_effect_purchase_price_held``) holds them at the
market year's value. Ownership and the balancing institution are substitutes:
never add this effect to another case's saving.

The per-unit decomposition (``decompose_public_bm_bids.py``) is reported beside
it as is. It measures bid money, which the tariff does not carry as a line, so it
is context and a volume check, never converted onto the bill. The decision-20 tie
steps are already netted out of every money total it and the two bills report.

Usage::

    python scripts/public_ownership/merge_tariff_balancing_groups.py MARKET_BY_CASE MARKET_SUMMARY
        MARKET_TARIFF_COMPONENTS MARKET_TARIFF_SUMMARY OUT_CSV OUT_JSON
        --group GROUP BID_CASE BY_CASE BILL_SUMMARY TARIFF_COMPONENTS TARIFF_SUMMARY DECOMPOSITION [--group ...]
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

from scripts.public_ownership.levers import PRIVATE_SCENARIO  # noqa: E402
from scripts.public_ownership.merge_balancing_groups import (  # noqa: E402
    LINE_TOTAL_TOLERANCE_GBP, VALUE, _lines, check_comparable, check_lines_match_totals,
)
from scripts.public_ownership.nationalised_tariff import (  # noqa: E402
    LINE_IC_ACQUISITION, LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL,
)

#: The tariff lines that annuitise a purchase price, struck before the bid stack changes.
PURCHASE_PRICE_LINES = (LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, LINE_IC_ACQUISITION)
#: Diagnostics carrying the purchase price, reported for both tariffs.
PRICE_DIAGNOSTICS = ("acquisition_price_gbp", "acquisition_price_renewables_nuclear_gbp",
                     "acquisition_price_thermal_storage_gbp")
#: The decomposition's own year totals, reported as they are (GBP and MWh, net of tie steps).
DECOMPOSITION_FIELDS = (
    "delta_bid_cost_gbp", "delta_offer_cost_gbp", "transfer_to_the_state_gbp",
    "payment_to_private_counterparties_gbp", "offsetting_private_cfd_settlement_gbp",
    "base_decrease_MWh", "public_decrease_MWh", "delta_decrease_MWh", "tie_break_step_cost_gbp",
)
#: Conversion keys the two tariffs must share (household, rebasing, recovery volumes).
CONVERSION_TOLERANCE = 1e-9


def _tariff_case(summary: dict, case_id: str, where: str) -> dict:
    case = (summary.get("cases") or {}).get(case_id)
    if case is None:
        raise ValueError(f"{where} tariff summary has no case {case_id!r}; tariffed: {sorted(summary.get('cases') or {})}")
    return case


def _bill_total(summary: dict, case_id: str, where: str) -> dict:
    case = (summary.get("cases") or {}).get(case_id)
    if case is None:
        raise ValueError(f"{where} bill summary has no case {case_id!r}")
    return case


def check_tariffs_comparable(market: dict, group: dict, case_id: str, name: str) -> None:
    """The two tariffs must share the household conversion and each must replace its own group's private bill."""
    a, b = market.get("conversion") or {}, group.get("conversion") or {}
    # 9 Nov 2026: the interconnector pass-through keys are each group's own measured credit (household and system),
    # which a bid stack moves; the conversion they carry is their ratio, which the groups must share.
    hh, sy = "interconnector_pass_through_household_gbp", "interconnector_pass_through_system_gbp"
    if all(k in c and c[k] for c in (a, b) for k in (hh, sy)):
        ra, rb = float(a[sy]) / float(a[hh]), float(b[sy]) / float(b[hh])
        if not np.isclose(ra, rb, rtol=CONVERSION_TOLERANCE, atol=0.0):
            raise ValueError(f"{name}: {case_id} interconnector pass-through system per household is {rb!r} against "
                             f"the market group's {ra!r}")
    for key in sorted((set(a) | set(b)) - {hh, sy}):
        x, y = a.get(key), b.get(key)
        same = (np.isclose(float(x), float(y), rtol=CONVERSION_TOLERANCE, atol=0.0)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) else x == y)
        if not same:
            raise ValueError(f"{name}: {case_id} tariff conversion {key} is {y!r} against the market group's {x!r}")
    if market.get("institution") != group.get("institution"):
        raise ValueError(f"{name}: {case_id} is tariffed under institution {group.get('institution')!r} here and "
                         f"{market.get('institution')!r} on the market group")


def check_tariff_replaces_bill(tariff: dict, bill_private: dict, where: str) -> None:
    """A tariff's private comparator is the private bill it was built on, to the penny."""
    tariff_private = float(tariff["bill"]["private_total_gbp_inc_vat"])
    if abs(tariff_private - float(bill_private["total_gbp_inc_vat"])) > LINE_TOTAL_TOLERANCE_GBP:
        raise ValueError(f"{where}: the tariff replaced a private bill of {tariff_private:.6f} but the group's private "
                         f"bill is {float(bill_private['total_gbp_inc_vat']):.6f}; they are not the same projection")


def tariff_components(market_tariff: pd.DataFrame, group_tariff: pd.DataFrame,
                      market_bill: pd.DataFrame, group_bill: pd.DataFrame, case_id: str) -> pd.DataFrame:
    """The case's tariff line by line on both stacks, with both privates beside it."""
    columns = {
        "case_on_group": _lines(group_tariff, case_id, "the balancing group's tariff"),
        "case_on_market": _lines(market_tariff, case_id, "the market group's tariff"),
        "private_on_group": _lines(group_bill, PRIVATE_SCENARIO, "the balancing group's bill"),
        "private_on_market": _lines(market_bill, PRIVATE_SCENARIO, "the market group's bill"),
    }
    ids = {name: set(series.index) for name, series in columns.items()}
    for first, second in (("case_on_group", "case_on_market"), ("private_on_group", "private_on_market")):
        if ids[first] != ids[second]:
            raise ValueError(f"case {case_id!r}: {first} and {second} do not carry the same lines; "
                             f"differing lines {sorted(ids[first] ^ ids[second])}")
    # The tariff replaces some private lines and adds its own, so the union carries both.
    lines = columns["case_on_group"].index.append(
        columns["private_on_group"].index.difference(columns["case_on_group"].index))
    frame = pd.DataFrame({name: series.reindex(lines, fill_value=0.0) for name, series in columns.items()})
    frame["private_bid_stack_decomposition_term"] = frame["private_on_group"] - frame["private_on_market"]
    frame["ownership_effect_on_stack"] = frame["case_on_group"] - frame["private_on_group"]
    frame["bid_stack_effect"] = frame["case_on_group"] - frame["case_on_market"]
    frame["purchase_price_line"] = frame.index.isin(PURCHASE_PRICE_LINES)
    labels = pd.concat([
        group_tariff.loc[group_tariff["case_id"].eq(case_id)],
        group_bill.loc[group_bill["case_id"].eq(PRIVATE_SCENARIO)],
    ]).drop_duplicates("component_id").set_index("component_id")
    frame.insert(0, "component", labels["component"].reindex(frame.index))
    frame.insert(1, "stack_group", labels["stack_group"].reindex(frame.index))
    frame.insert(0, "case_id", case_id)
    frame.index.name = "component_id"
    return frame.reset_index()


def check_components_match_totals(components: pd.DataFrame, bills: dict[str, dict], case_id: str) -> None:
    """Each of the four bills' own lines sums to its summary total (the private lines exclude tariff-only lines)."""
    check_lines_match_totals(components, bills, case_id)


def merge(market_bill: pd.DataFrame, market_summary: dict, market_tariff: pd.DataFrame,
          market_tariff_summary: dict, groups: list[dict]) -> tuple[pd.DataFrame, dict]:
    """``groups``: ``{"name", "bid_case", "by_case", "summary", "tariff", "tariff_summary", "decomposition"}``.

    Every case the group's tariff priced gets its own entry (the bid case and the
    tariff cases that own the same fleet, so bid the same way); the group's
    headline is its bid case's purchase-price-held bid-stack effect.
    """
    frames, cases, headlines = [], {}, {}
    private_market = _bill_total(market_summary, PRIVATE_SCENARIO, "the market group")
    for group in groups:
        name, summary, bid_case = group["name"], group["summary"], group["bid_case"]
        check_comparable(market_summary, summary, name)
        extra = sorted(set(summary.get("cases") or {}) - {PRIVATE_SCENARIO})
        if extra:
            raise ValueError(f"{name}: a tariff balancing group projects private alone on the market bill; got {extra}")
        tariffed = list(group["tariff_summary"].get("cases") or {})
        if bid_case not in tariffed:
            raise ValueError(f"{name}: its tariff must price its bid case {bid_case!r}; it priced {sorted(tariffed)}")
        missing = sorted(set(tariffed) - set(market_tariff_summary.get("cases") or {}))
        if missing:
            raise ValueError(f"{name}: the market group's tariff does not price {missing}, so they have no comparator")
        private_group = _bill_total(summary, PRIVATE_SCENARIO, name)
        check_tariff_replaces_bill(_tariff_case(market_tariff_summary, bid_case, "the market group"), private_market,
                                   "the market group")
        decomposition = (group.get("decomposition") or {}).get("year") or {}
        for case_id in [bid_case] + [c for c in tariffed if c != bid_case]:
            if case_id in cases:
                raise ValueError(f"case {case_id!r} is tariffed by more than one balancing group")
            on_group = _tariff_case(group["tariff_summary"], case_id, name)
            on_market = _tariff_case(market_tariff_summary, case_id, "the market group")
            check_tariffs_comparable(on_market, on_group, case_id, name)
            check_tariff_replaces_bill(on_group, private_group, name)

            components = tariff_components(market_tariff, group["tariff"], market_bill, group["by_case"], case_id)
            check_components_match_totals(components, {
                "case_on_group": on_group["bill"], "case_on_market": on_market["bill"],
                "private_on_group": private_group, "private_on_market": private_market,
            }, case_id)
            components.insert(1, "run_group", name)
            frames.append(components)

            vat_rate = float(on_group["bill"]["vat_gbp"]) / float(on_group["bill"]["ex_vat_gbp"])
            total = float(on_group["bill"]["total_gbp_inc_vat"])
            market_total = float(on_market["bill"]["total_gbp_inc_vat"])
            bid_stack = total - market_total
            price_lines = components.loc[components["purchase_price_line"]]
            price_effect = float(price_lines["bid_stack_effect"].sum()) * (1.0 + vat_rate)
            moved = components.loc[components["bid_stack_effect"].abs() > 1e-9]
            cases[case_id] = {
                "run_group": name,
                "bid_case": bid_case,
                "is_bid_case": case_id == bid_case,
                "forecast_case": summary.get("forecast_case"),
                "institution": on_group.get("institution"),
                "total_gbp_inc_vat": total,
                "tariff_on_market_gbp_inc_vat": market_total,
                "private_on_market_gbp_inc_vat": float(private_market["total_gbp_inc_vat"]),
                "private_on_group_gbp_inc_vat": float(private_group["total_gbp_inc_vat"]),
                # The increment with the purchase held at the market year's price: the figure to present.
                "headline_metric": "bid_stack_effect_purchase_price_held_gbp_inc_vat",
                "bid_stack_effect_purchase_price_held_gbp_inc_vat": bid_stack - price_effect,
                "bid_stack_effect_gbp_inc_vat": bid_stack,
                "purchase_price_line_effect_gbp_inc_vat": price_effect,
                "purchase_price_gbp": {
                    "on_group": {k: on_group["diagnostics"].get(k) for k in PRICE_DIAGNOSTICS},
                    "on_market": {k: on_market["diagnostics"].get(k) for k in PRICE_DIAGNOSTICS},
                },
                # Not a feasible case (private cannot bid the state's stack); a term of the sum only.
                "private_bid_stack_decomposition_term_gbp_inc_vat": (
                    float(private_group["total_gbp_inc_vat"]) - float(private_market["total_gbp_inc_vat"])),
                "ownership_effect_on_stack_gbp_inc_vat": total - float(private_group["total_gbp_inc_vat"]),
                "delta_vs_private_gbp_inc_vat": total - float(private_market["total_gbp_inc_vat"]),
                "bid_stack_effect_by_line_gbp_inc_vat": {
                    str(row.component_id): float(row.bid_stack_effect) * (1.0 + vat_rate)
                    for row in moved.itertuples()},
                "decomposition": {
                    **{key: decomposition.get(key) for key in DECOMPOSITION_FIELDS},
                    "note": ("bid money from the balancing LPs, net of the decision-20 tie steps, shared by every "
                             "case on this stack; the tariff carries no bid line, so this is context and a volume "
                             "check, never converted onto the bill"),
                },
                "levers": on_group.get("levers"),
            }
        headlines[name] = {
            "case_id": bid_case,
            "metric": "bid_stack_effect_purchase_price_held_gbp_inc_vat",
            "gbp_inc_vat": cases[bid_case]["bid_stack_effect_purchase_price_held_gbp_inc_vat"],
            "cases_on_stack": [bid_case] + [c for c in tariffed if c != bid_case],
        }
    components = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return components, {
        "market_forecast_case": market_summary.get("forecast_case"),
        "headline_by_group": headlines,
        "cases": cases,
        "notes": ("Each case = market private + private-on-stack decomposition term + ownership effect on the stack, "
                  "exactly (a telescoping sum of the four bills; the merge checks each bill's lines against its own "
                  "total and each tariff against the private bill it replaced). bid_stack_effect is the tariff on the "
                  "group's months less the same case's tariff on the market months. The purchase-price lines are "
                  "split out because the price is valued on each group's own settlement and is struck on the market "
                  "year; bid_stack_effect_purchase_price_held is the headline, for the group's bid case. The other "
                  "cases own the same fleet (they differ only in purchase price basis, thermal basis or annuity term) "
                  "and so bid the same stack. Never add it to another case's saving: ownership and the balancing "
                  "institution are substitutes."),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("market_by_case")
    parser.add_argument("market_summary")
    parser.add_argument("market_tariff_components")
    parser.add_argument("market_tariff_summary")
    parser.add_argument("out_csv")
    parser.add_argument("out_json")
    parser.add_argument("--group", nargs=7, action="append", required=True,
                        metavar=("NAME", "BID_CASE", "BY_CASE", "BILL_SUMMARY", "TARIFF_COMPONENTS", "TARIFF_SUMMARY",
                                 "DECOMPOSITION"))
    args = parser.parse_args(argv)

    def _read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    groups = [{"name": name, "bid_case": bid_case, "by_case": pd.read_csv(by_case), "summary": _read_json(summary),
               "tariff": pd.read_csv(components), "tariff_summary": _read_json(tariff_summary),
               "decomposition": _read_json(decomposition)}
              for name, bid_case, by_case, summary, components, tariff_summary, decomposition in args.group]
    components, summary = merge(pd.read_csv(args.market_by_case), _read_json(args.market_summary),
                                pd.read_csv(args.market_tariff_components), _read_json(args.market_tariff_summary),
                                groups)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, allow_nan=False, default=float), encoding="utf-8")
    for case_id, entry in summary["cases"].items():
        print(f"{case_id}{' [headline]' if entry['is_bid_case'] else ''} on {entry['run_group']}: "
              "bid-stack increment, purchase price held, "
              f"{entry['bid_stack_effect_purchase_price_held_gbp_inc_vat']:+.2f} GBP/household inc VAT "
              f"(tariff {entry['total_gbp_inc_vat']:.2f} against {entry['tariff_on_market_gbp_inc_vat']:.2f} on the "
              f"market months; purchase-price lines {entry['purchase_price_line_effect_gbp_inc_vat']:+.2f})")


if __name__ == "__main__":
    main()
