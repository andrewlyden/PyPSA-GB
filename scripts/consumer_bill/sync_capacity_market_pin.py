"""Keep the 2030 bill's Capacity Market pin equal to the auction builder's summary (user, 11 Nov 2026).

The bill reads ``sourced_externals.capacity_market.derivation`` in
``data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json``; its model-derived numbers come from
rule ``build_capacity_market`` (``cm_calendar_2030_summary.json``). They were copied by hand, and fell behind the
re-solved fleet (2,578.8 against 2,580.0 GBP m). The assumptions file is the bill's input, so it is not a rule
output: ``--write`` re-pins it from the summary, and ``--check`` (rule ``verify_capacity_market_pin``, in the bill
chain) fails, naming the command, when a pinned number has drifted. Hand-sourced fields (the household factor and
its basis, the notes) are left alone.

    python scripts/consumer_bill/sync_capacity_market_pin.py --summary resources/market/capacity_market/cm_calendar_2030_summary.json --write
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
#: Each delivery year's pinned money keys.
DELIVERY_KEYS = {"2029/2030": ("delivery_year_2029_30_gbp", ("contract_book_gbp", "t1_gbp", "total_gbp")),
                 "2030/2031": ("delivery_year_2030_31_gbp", ("contract_book_gbp", "t4_modelled_gbp", "t1_gbp", "total_gbp"))}
#: The pin holds the summary rounded as written (prices 2 dp, money and MW 1 dp): any other difference is drift.
TOLERANCE = 1e-9


def pinned_values(summary: dict) -> dict:
    """Every model-derived number of the derivation block, keyed by its dotted path in the block."""
    out = {"agreement_cost_gbp": round(float(summary["calendar_2030_cost_gbp"]), 1),
           "t4_2030_31_central_price_gbp_per_kw_year": round(float(summary["t4_2030_31_central"]["price"]), 2),
           "t4_2030_31_central_awarded_mw": round(float(summary["t4_2030_31_central"]["awarded_mw"]), 1)}
    for year, (block, keys) in DELIVERY_KEYS.items():
        for key in keys:
            out[f"{block}.{key}"] = round(float(summary["delivery_years"][year][key]), 1)
    for case, price in summary["t4_2030_31_price_by_case_gbp_per_kw_year"].items():
        out[f"t4_2030_31_price_by_case_gbp_per_kw_year.{case}"] = round(float(price), 2)
    for case, cost in summary["calendar_2030_cost_by_t4_case_gbp"].items():
        out[f"calendar_2030_cost_by_t4_case_gbp.{case}"] = round(float(cost), 1)
        if case != "central":
            out[f"sensitivity_cost_gbp.t4_case_{case}"] = round(float(cost), 1)
    out["sensitivity_cost_gbp.t4_2030_31_at_2029_30_clearing_27_10"] = round(
        float(summary["calendar_2030_cost_t4_at_2029_30_clearing_gbp"]), 1)
    return out


def _get(block: dict, dotted: str):
    for part in dotted.split("."):
        block = block[part]
    return block


def drift(assumptions: dict, summary: dict) -> dict:
    derivation = assumptions["sourced_externals"]["capacity_market"]["derivation"]
    out = {}
    for key, value in pinned_values(summary).items():
        try:
            pinned = float(_get(derivation, key))
        except (KeyError, TypeError):
            out[key] = (None, value)
            continue
        if abs(pinned - value) > TOLERANCE * max(1.0, abs(value)):
            out[key] = (pinned, value)
    return out


def rewrite(text: str, changes: dict) -> str:
    """Replace each drifted number in place (the file's layout and every other byte unchanged)."""
    for key, (old, new) in changes.items():
        if old is None:
            raise ValueError(f"the derivation block has no {key}; add it by hand once")
        leaf = key.split(".")[-1]
        pattern = re.compile(rf'("{re.escape(leaf)}":\s*){re.escape(repr(old) if not float(old).is_integer() else str(old))}\b')
        hits = pattern.findall(text)
        if len(hits) != 1:
            # Fall back on the number as JSON writes it (e.g. 40411.0).
            pattern = re.compile(rf'("{re.escape(leaf)}":\s*){re.escape(json.dumps(old))}\b')
            hits = pattern.findall(text)
        if len(hits) != 1:
            raise ValueError(f"{key}: expected one '{leaf}: {old}' in the file, found {len(hits)}")
        text = pattern.sub(lambda m: f"{m.group(1)}{new}", text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--assumptions", default=str(ASSUMPTIONS))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--stamp", default=None, help="with --check: write a JSON stamp when the pin agrees")
    a = ap.parse_args(argv)
    summary = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    path = Path(a.assumptions)
    text = io.open(path, encoding="utf-8", newline="").read()
    changes = drift(json.loads(text), summary)
    if a.check:
        if changes:
            lines = "\n".join(f"  {k}: pinned {old}, builder {new}" for k, (old, new) in changes.items())
            print(f"{path}: the Capacity Market pin differs from {a.summary}:\n{lines}\nre-pin with:\n"
                  f"  python scripts/consumer_bill/sync_capacity_market_pin.py --summary {a.summary} --write",
                  file=sys.stderr)
            return 1
        if a.stamp:
            Path(a.stamp).write_text(json.dumps({"assumptions": str(path), "summary": a.summary,
                                                 "values": pinned_values(summary)}, indent=1), encoding="utf-8")
        print(f"{path}: the Capacity Market pin agrees with {a.summary}")
        return 0
    if not changes:
        print("nothing to re-pin")
        return 0
    io.open(path, "w", encoding="utf-8", newline="").write(rewrite(text, changes))
    for k, (old, new) in changes.items():
        print(f"re-pinned {k}: {old} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
