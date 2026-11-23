"""Start a new long-range FES year: write its study folder and list what it still needs (FES 2040, 11 Nov 2026).

FES 2025 HT 2040 was the first year after 2050. This tool makes the next one
(2045, say) a known checklist instead of a survey:

``--write``  writes ``config/studies/fes<Y>/`` from ``config/studies/fes2050/`` by declared substitutions:
             template names ``ht50`` -> ``ht<yy>``, ids ``FES25_HT_2050`` -> ``FES25_HT_<Y>``, run groups,
             the year's targets, ES1 lines, station schedule, future costs and landing files, the model years,
             and the bill and finance file names. References to the 2050 plan, its decisions and its modules
             stay. Year-specific changes:

             * foreign prices built on TYNDP 2026 draft NT+ directly (the market's override re-prices on NT+
               anyway; re-pricing on the build's own variant reproduces the built links, so this equals the
               2050 build's DE plus override; TYNDP 2024 DE is held for 2050 only);
             * unabated CCGT offers at 1.29 x SRMC, as gas CCS (user, 10 Nov 2026), where the year's ES1 has any;
             * the forecast-error fleet ratio (decision 18): 1.10 x ES1 wind and solar output (with the
               non-networked offshore) / CP30 2030's 273.0 TWh.

             The folder is a draft: its comments still describe 2050 where 2050's facts were written into
             them; read it before solving. Refuses to overwrite an existing folder.
``--check``  (default) lists every input the year needs and whether it is there, with the command or the
             plan item that makes it.

Usage::

    python -m scripts.sources.new_fes_year --year 2045 [--write]
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "config/studies/fes2050"
FILES = ("study.yaml", "families.yaml", "scenarios.yaml", "templates.yaml")
#: References that stay 2050 in a new year's study.
KEEP = ("fes2050_pumped_hydro_sites.csv", "repd_stale_records_2050.csv", "vintage_capital_2050:",
        "build_capacity_market_2050", "contract_book_2050.py", "support_regime_2050.json", "fes_2050_public_ownership_plan",
        "FES 2050 decision", "build_central_dispatch_strike_inputs_2050", "build_central_dispatch_contract_book_2050",
        "support_lines_2050", "capacity_market_2050")
YEAR_PARAMETER_KEYS = ("hydrogen_supply_price_gbp_per_mwh_h2", "hydrogen_end_of_window_value_gbp_per_mwh_h2",
                       "ccs_gas_efficiency", "ccs_gas_variable_om_gbp_per_mwh", "hydrogen_reserve_twh",
                       "performance_factor_wind_offshore", "performance_factor_wind_onshore",
                       "performance_factor_solar_pv")
#: CP30 2030's ES1 wind and solar output, the forecast-error fleet ratio's base (decision 18).
CP30_WIND_SOLAR_TWH = 273.0
FTI_FLEET_RATIO = 1.10


def substitutions(year: int) -> list[tuple[str, str]]:
    y, yy = int(year), f"{int(year) % 100:02d}"
    return [
        (r"\bht50", f"ht{yy}"),
        (r"FES25_HT_2050", f"FES25_HT_{y}"),
        (r"fes_ht_2050", f"fes_ht_{y}"),
        (r"fes2025_holistic_transition_2050", f"fes2025_holistic_transition_{y}"),
        (r"fes2025_future_costs_2050", f"fes2025_future_costs_{y}"),
        (r"fes2050_offshore_landing", f"fes{y}_offshore_landing"),
        (r"(modelled_year|model_year|capability_year|demand_year): 2050\b", rf"\1: {y}"),
        (r"^year: 2050\r?$", f"year: {y}"),
        (r"consumer_bill_benchmarks/2050_", f"consumer_bill_benchmarks/{y}_"),
        (r"consumer_bill_forecast/2050_", f"consumer_bill_forecast/{y}_"),
        (r"(ancillary_services|efw_gate_fees|capture_prices|strike_calibration|cm_fleet|cm_calendar|rav_roll_forward|"
         r"network_guarantee_pass_through)_2050", rf"\1_{y}"),
        (r"^(\s+)2050:(\r?)$", rf"\g<1>{y}:\2"),
        (r"Calendar-2050 snapshot; external lines carry their 2050", f"Calendar-{y} snapshot; external lines carry their {y}"),
        (r"FES 2025 (HT|Holistic Transition) 2050", rf"FES 2025 \1 {y}"),
        (r"pathway in 2050, solved", f"pathway in {y}, solved"),
        (r"ht\d\d_bm2050_bids", f"ht{yy}_bm_bids"),
        (r"TYNDP 2026 draft National Trends\+ 2050", f"TYNDP 2026 draft National Trends+ {y}"),
        # The balancing stage's network case (11 Nov 2026: the boundary outage replay).
        (r"fes_2050_as_built", f"fes_{y}_as_built"),
        (r"The FES 2050 network as built", f"The FES {y} network as built"),
    ]


def convert(text: str, year: int) -> str:
    tokens = {}
    for i, keep in enumerate(KEEP):
        token = f"@@KEEP{i}@@"
        tokens[token] = keep
        text = text.replace(keep, token)
    for pattern, replacement in substitutions(year):
        text = re.sub(pattern, replacement, text, flags=re.M)
    for token, keep in tokens.items():
        text = text.replace(token, keep)
    return text


def targets(year: int) -> dict | None:
    path = ROOT / f"resources/FES/fes2025_holistic_transition_{int(year)}_targets.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fleet_ratio(year: int) -> float | None:
    """Decision 18: 1.10 x the year's ES1 wind and solar output (with non-networked offshore) / 273.0 TWh, over FTI's
    fleet growth across 2030-2040 (user decision, 11 Nov 2026; forecast_error_charge.FTI_FLEET_GROWTH)."""
    from scripts.market.forecast_error_charge import FTI_FLEET_GROWTH

    t = targets(year)
    if t is None:
        return None
    gen = {r["carrier"]: r for r in t["generation"]}
    twh = sum(float(gen[c]["generation_gwh"]) for c in ("wind_offshore", "wind_onshore", "solar_pv") if c in gen) / 1e3
    twh += sum(float(r.get("value_std") or 0.0) for r in (t.get("excluded") or [])
               if isinstance(r, dict) and r.get("carrier") == "wind_offshore" and r.get("quantity") == "generation_gwh") / 1e3
    return FTI_FLEET_RATIO * twh / CP30_WIND_SOLAR_TWH / FTI_FLEET_GROWTH


def unabated_ccgt_mw(year: int) -> float:
    """ES1's unabated CCGT capacity (type Gas, subtype CCGT) in the year's ES1 lines."""
    import pandas as pd

    path = ROOT / f"resources/FES/fes2025_holistic_transition_{int(year)}_es1_lines.csv"
    if not path.exists():
        return 0.0
    lines = pd.read_csv(path)
    rows = lines[lines["type"].eq("Gas") & lines["subtype"].eq("CCGT") & lines["quantity"].eq("capacity_mw")]
    return float(pd.to_numeric(rows["value_std"], errors="coerce").fillna(0.0).sum())


def write(year: int) -> Path:
    y, yy = int(year), f"{int(year) % 100:02d}"
    if y == 2050:
        raise SystemExit("2050 is the template")
    out = ROOT / f"config/studies/fes{y}"
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first to regenerate")
    out.mkdir(parents=True)
    for name in FILES:
        raw = io.open(TEMPLATE / name, encoding="utf-8", newline="").read()
        text = convert(raw, y)
        if name == "templates.yaml":
            nl = "\r\n" if "\r\n" in text else "\n"
            de = f"      tyndp:{nl}        scenario: DE{nl}        route: tyndp_shape{nl}"
            if text.count(de) == 1:
                text = text.replace(de, (
                    f"      # Built on TYNDP 2026 draft NT+ {y} directly: the market's foreign_price_override re-prices every{nl}"
                    f"      # interconnector on NT+, and re-pricing on the build's own variant reproduces the built links, so{nl}"
                    f"      # this equals the 2050 study's DE build plus override.{nl}"
                    f"      tyndp:{nl}        scenario: NTplus2026{nl}        route: tyndp_shape{nl}"))
            # Unabated CCGT offers at 1.6 x SRMC as 2030, in the 2050 study's bid template the draft copies (user,
            # 11 Nov 2026; superseding the 10 Nov 1.29 x insertion here).
            ratio = fleet_ratio(y)
            if ratio is not None:
                text = re.sub(r"^(\s+)fleet_ratio: 1\.714(\r?)$",
                              rf"\g<1># Decision 18 for {y}: 1.10 x ES1 {y} wind and solar / CP30 2030's 273.0 TWh / FTI's 2030-2040 fleet growth 1.354.\2\n"
                              rf"\g<1>fleet_ratio: {ratio:.3f}\2", text, flags=re.M)
        io.open(out / name, "w", encoding="utf-8", newline="").write(text)
    return out


def readiness(year: int) -> list[tuple[str, bool, str]]:
    """Every input a long-range year needs: ``(item, present, how to make it)``."""
    from config.config_loader import year_parameter_table

    y = int(year)
    items: list[tuple[str, bool, str]] = []

    def file(path: str, how: str):
        items.append((path, (ROOT / path).exists(), how))

    file(f"config/studies/fes{y}/templates.yaml", f"python -m scripts.sources.new_fes_year --year {y} --write; add "
                                                  f"fes{y} to config/config.yaml studies")
    for kind in ("targets.json", "es1_lines.csv"):
        file(f"resources/FES/fes2025_holistic_transition_{y}_{kind}", "rule build_fes2025_energy_targets")
    file(f"resources/FES/fes2025_future_costs_{y}.json", "rule build_fes2025_future_costs_source")
    file(f"resources/FES/fes2025_holistic_transition_{y}_station_schedule.csv",
         "rule build_fes_station_schedule; check data/generators/fes/station_list.csv covers the year's Direct blocks")
    table = year_parameter_table()
    for key in YEAR_PARAMETER_KEYS:
        how = (f"python -m scripts.sources.fes_year_inputs performance --year {y} --write (after the year's renewables network is built)"
               if key.startswith("performance_factor") else f"python -m scripts.sources.fes_year_inputs parameters --year {y} --write")
        items.append((f"data/FES/fes_year_parameters.csv: {y} HT {key}", (y, "HT", key) in table, how))
    for suffix in ("", "_projects", "_boundary_check"):
        file(f"data/network/ETYS/fes{y}_offshore_landing{suffix}.csv",
             f"python -m scripts.network_clustering.fes_offshore_landing --year {y} (dated evidence check: plan item 4)")
    from scripts.interconnectors import tyndp_prices as tp
    try:
        for path in tp.tyndp_input_files("NTplus2026", y, "data/interconnectors/tyndp2024").values():
            file(path, "scripts/interconnectors/extract_tyndp_ntplus_mc.py and extract_tyndp_hourly.py (plan item 5)")
    except (KeyError, ValueError) as exc:
        items.append((f"tyndp_prices weather weights for {y}", False,
                      f"add TYNDP2026_WEATHER_WEIGHTS_BY_YEAR[{y}] from TYNDP Annex III ({exc})"))
    for path, how in (
            (f"data/public_ownership/financing/ancillary_services_{y}.csv", f"python -m scripts.sources.fes_year_inputs inputs --year {y}"),
            (f"data/public_ownership/financing/efw_gate_fees_{y}.json", f"python -m scripts.sources.fes_year_inputs inputs --year {y}"),
            (f"data/market/network_cost_sources/rav_roll_forward_{y}.yaml", f"python -m scripts.sources.fes_year_inputs inputs --year {y}"),
            (f"data/market/network_cost_sources/network_guarantee_pass_through_{y}.csv",
             f"python -m scripts.sources.fes_year_inputs inputs --year {y}"),
            (f"data/public_ownership/financing/strike_calibration_{y}.csv", f"python -m scripts.sources.fes_year_inputs seeds --year {y}; re-derived by post_solve_year.py"),
            (f"data/public_ownership/financing/capture_prices_{y}.csv", f"python -m scripts.sources.fes_year_inputs seeds --year {y}; re-derived by post_solve_year.py"),
            (f"resources/market/capacity_market/cm_calendar_{y}_summary.json", f"rule build_capacity_market_year (year {y})"),
            (f"data/market/consumer_bill_benchmarks/{y}_pypsa_informed_assumptions.json",
             f"python -m scripts.consumer_bill.build_fes_bill_inputs --year {y} (needs the CM summary and the RAV inputs)")):
        file(path, how)
    return items


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--write", action="store_true", help="write config/studies/fes<Y>/ from fes2050")
    args = parser.parse_args(argv)
    if args.write:
        print(f"wrote {write(args.year)} (a draft: read its comments before solving)")
    missing = 0
    for item, present, how in readiness(args.year):
        missing += not present
        print(f"[{'ok' if present else '--'}] {item}" + ("" if present else f"\n       -> {how}"))
    print(f"{missing} input(s) missing for {args.year}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
