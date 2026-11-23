"""Build the cumulative 2030-2050 public ownership page from the pathway summary (user, 10 Nov 2026).

Reads ``pathway_2030_2050_summary.json`` (``scripts/public_ownership/pathway_2030_2050.py``, rule
``build_public_ownership_pathway_2030_2050``) and writes one self-contained page: the cumulative
change by party for every reported case, year by year, with the solved years marked, the yearly
change by mechanism, and the totals undiscounted and at the Green Book rates.

Nothing here is modelled: every figure is the summary's, converted from system GBP to GBP bn. The
solved years come from the summary (``snapshot_years``), so a run with 2040 added marks 2040 as
solved without a change here. Usage::

    python scripts/analysis/public_ownership_pathway_explorer/build_pathway_explorer.py SUMMARY_JSON OUT_HTML
        [--template HTML] [--run-label TEXT]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
DATA_MARKER = "/*__DATA__*/"
PARTIES = ("consumers", "exchequer", "investors", "resource")
MECHANISMS = ("financing", "legacy", "market_and_other")
#: Scenario code and plain name by pathway label (pathway_2030_2050.CASES).
CASE_NAMES = {  # pathway label -> (scenario code, page label, plain name); meeting notes, user 10 Nov 2026
    "P1": ("S1", "S1", "Scenario 1, public control: a 30% state stake in new plant"),
    "P2 (100%)": ("S2", "S2 (100%)", "Scenario 2, public ownership, new build only: the state builds all new plant"),
    "P2 75%": ("S2", "S2 75%", "Scenario 2 with the state building 75% of new plant"),
    "P2 50%": ("S2", "S2 50%", "Scenario 2 with the state building 50% of new plant"),
    "P2 25%": ("S2", "S2 25%", "Scenario 2 with the state building 25% of new plant"),
    "P3 I1": ("S3", "S3 market", "Scenario 3, new build + nationalisation, market dispatch kept"),
    "P3 I3": ("S3", "S3 central", "Scenario 3, new build + nationalisation, dispatched centrally"),
}
#: Where each solved year comes from, as the page names it (any other year: "solved {year}").
SNAPSHOT_NAMES = {2030: "Clean Power 2030, central network case", 2050: "FES 2025 Holistic Transition"}


_P_CODE = re.compile(r"\bP([123])\b")


def scenario_names(obj):
    """P1/P2/P3 in any sentence the page shows written as Scenario 1/2/3 (user, 11 Nov 2026: private is Scenario 0)."""
    if isinstance(obj, str):
        return _P_CODE.sub(lambda m: f"Scenario {int(m.group(1))}", obj)
    if isinstance(obj, list):
        return [scenario_names(v) for v in obj]
    if isinstance(obj, dict):
        return {k: scenario_names(v) for k, v in obj.items()}
    return obj


def bn(value) -> float | None:
    return None if value is None else round(float(value) / 1e9, 4)


def compact(summary: dict, run_label: str = "") -> dict:
    """The page's data: every figure in GBP bn (real 2024), per household in GBP (real 2025)."""
    first, last = (int(y) for y in summary["years"])
    years = list(range(first, last + 1))
    rates = summary.get("discount_rates") or {}
    cases = []
    for case_id, c in summary["cases"].items():
        code, label, name = CASE_NAMES.get(c["label"], (c["label"].split()[0], c["label"], c["label"]))
        annual = {p: {m: [bn((c["annual"][p].get(str(y)) or {}).get(m, 0.0)) for y in years]
                      for m in (*MECHANISMS, "total", "financing_anchor", "legacy_anchor")} for p in PARTIES}
        totals = {p: {k: bn(v) for k, v in c["parties"][p]["total"].items()} for p in PARTIES}
        per_hh = {p: {k: round(float(v), 2) for k, v in c["parties"][p]["per_household_3100kwh_gbp2025"].items()}
                  for p in PARTIES}
        gain = c.get("public_gain_consumers_plus_exchequer") or {}
        beyond = ((c.get("beyond_2050") or {}).get("parties") or {})
        cases.append({
            "id": case_id, "label": label, "code": code, "name": name, "source_2030": c.get("source_2030"),
            "annual": annual, "totals": totals, "per_household": per_hh,
            "gain_totals": {k: bn(v) for k, v in (gain.get("total") or {}).items()},
            "gain_per_household": {k: round(float(v), 2) for k, v in (gain.get("per_household_3100kwh_gbp2025") or {}).items()},
            "beyond_2050": {p: {k: bn(v) for k, v in ((beyond.get(p) or {}).get("total") or {}).items()}
                            for p in PARTIES if p in beyond},
            "beyond_years": (c.get("beyond_2050") or {}).get("years"),
        })
    snaps = [int(y) for y in summary.get("snapshot_years") or (first, last)]
    return {
        "meta": {
            "generated": date.today().isoformat(), "run_label": run_label,
            "first_year": first, "last_year": last, "years": years, "base_year": summary.get("base_year", first),
            "snapshot_years": snaps,
            "snapshot_names": {str(y): SNAPSHOT_NAMES.get(y, f"solved {y}") for y in snaps},
            "discount_rates": rates, "central_rate": summary.get("central_discount_rate", "stpr_3p5"),
            "price_basis": summary.get("price_basis"),
            "anchoring": {k: (summary.get("anchoring") or {}).get(k) for k in ("rule", "all_pass", "worst_difference_gbp")},
            "assumptions": summary.get("assumptions") or [],
            "interpolated_lines": summary.get("interpolated_lines") or {},
            "beyond_rule": summary.get("beyond_2050_rule"),
        },
        "cases": cases,
    }


def build_page(summary_path: Path, template_path: Path, out_path: Path, run_label: str = "") -> int:
    data = compact(json.loads(summary_path.read_text(encoding="utf-8")), run_label)
    raw = json.dumps(scenario_names(data), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template_path.read_text(encoding="utf-8")
    if html.count(DATA_MARKER) != 1:
        raise SystemExit(f"{template_path} must hold exactly one {DATA_MARKER} marker")
    meta = data["meta"]
    html = html.replace("__FIRST_YEAR__", str(meta["first_year"])).replace("__LAST_YEAR__", str(meta["last_year"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace(DATA_MARKER, raw), encoding="utf-8")
    return out_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--template", type=Path, default=HERE / "pathway_explorer_template.html")
    parser.add_argument("--run-label", default="", help="how the page names the run, e.g. 'the 10 November 2026 runs'")
    args = parser.parse_args()
    size = build_page(args.summary, args.template, args.out, args.run_label)
    print(f"wrote {args.out} ({size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
