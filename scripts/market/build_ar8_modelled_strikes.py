"""The modelled AR8 CfD round of the 2030 CP30 chain.

CfD capacity in the 2030 fleet that the vintage schedule (``cfd_vintage_schedule_2030.csv``) cannot place on
a published round used to be labelled ``future`` and settle at a copy of the latest round's clearing price.
It is now the round ``AR8_modelled`` (:data:`scripts.market.cfd_strike_table.MODELLED_ROUND`): plant
generating in calendar 2030 that wins a contract in Allocation Round 8 (results due by March 2027), at a
modelled clearing price.

**Strike.** :func:`scripts.market.cfd_strikes.levelised_strike` on the central basis (economic-life
amortisation, flat F29 v2.0 private rate, the register's insurance rule, EGC 2025 Annex A load factors and
use of system), with the capex calibrated to the last cleared auction for each technology
(``data/market/support/cfd_strike_cost_calibration.csv``: AR7 offshore 91.20 for every model year; AR7a
onshore 72.24, solar 65.23 and tidal stream 265.00 for model year 2030 only). AR8's delivery years are
2028/29-2030/31 (solar, onshore), 2029/30-2031/32 (offshore, floating) and 2029/30-2030/31 (tidal stream)
(DESNZ, CfD AR8 Allocation Framework, 6 Jul 2026). Plant that generates in 2030 under AR8 is built in 2029 or
2030, so the AR8 strike is the mean of the 2029 and 2030 build years' levelised strikes, equally weighted.

Technologies the strike path does not model separately:

* ``offshore_scotland``: AR8 Pot 3 again clears transmission zones 1-12 separately (AR8 Pot and Price Notice);
  the modelled offshore strike times AR7's Scotland-to-rest ratio (64.23 / 65.45, 2012 prices).
* ``floating_offshore``: held at AR7's floating clearing price; no 2030 unit is on this row (the floating
  projects carry their own AR6/AR7 contracts).
* ``nuclear``: the Hinkley Point C strike, a fallback for CfD-labelled nuclear without a round only (HPC carries
  its own round).

**Range** (bill layer only, ``model_linked_rules.cfd.strike_overrides``; no re-solve): low = AR6 clearing
prices (the lowest recent clears) where they are below the modelled strike (AR6 solar and tidal cleared above
it, so the low case holds them at the modelled strike), high = AR8's administrative strike prices (2024 prices, frozen at AR7's;
DESNZ, CfD AR8 Pot and Price Notice, 6 Jul 2026, Table 2). Each is a twin of the central bill assumptions file
(``2030_pypsa_informed_assumptions_cfd_rule_none.json``) that also names the table the BM bid on
(``bm_strike_table``), so the curtailment compensation stays on the solved bids (FES 2050 decision 41's rule).

Usage::

    python scripts/market/build_ar8_modelled_strikes.py            # write the table rows and the twins
    python scripts/market/build_ar8_modelled_strikes.py --check    # fail if the committed files differ
"""

import argparse
import copy
import csv
import io
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.market import cfd_strikes  # noqa: E402
from scripts.market.cfd_strike_table import MODELLED_ROUND  # noqa: E402

STRIKE_TABLE = PROJECT_ROOT / "data" / "market" / "lccc" / "cfd_strike_prices_by_round.csv"
BILL_ASSUMPTIONS = PROJECT_ROOT / "data" / "market" / "consumer_bill_benchmarks" / "2030_pypsa_informed_assumptions_cfd_rule_none.json"
#: The bill-only range twins of :data:`BILL_ASSUMPTIONS` (low: AR6 clears; high: AR8 administrative strikes).
TWINS = {
    "low": BILL_ASSUMPTIONS.with_name("2030_pypsa_informed_assumptions_cfd_rule_none_ar8_low.json"),
    "high": BILL_ASSUMPTIONS.with_name("2030_pypsa_informed_assumptions_cfd_rule_none_ar8_high.json"),
}
#: The round label this script writes, and the labels it replaces (the old declared round).
REPLACED_ROUNDS = (MODELLED_ROUND, "future")
MODEL_YEAR = 2030
#: Build years of AR8 plant generating in calendar 2030 (delivery years 2029/30 and 2030/31), equally weighted.
BUILD_YEARS = (2029, 2030)
#: Strike-table technology -> the strike path's vintage technology.
MODELLED = {"offshore": "Offshore wind", "onshore": "Onshore wind", "solar": "Solar PV", "tidal": "Tidal stream"}
ROW_ORDER = ("offshore", "offshore_scotland", "floating_offshore", "onshore", "solar", "tidal", "nuclear")
DELIVERY_YEARS = {"offshore": "2029-30", "offshore_scotland": "2029-30", "floating_offshore": "2029-30",
                  "onshore": "2029-30", "solar": "2029-30", "tidal": "2029-30", "nuclear": "2026-27"}
#: 2012 -> 2024 prices: the bill's cfd_strike_indexation (ONS CPI 133.9 / 96.1).
CPI_2012_TO_2024 = cfd_strikes.CPI_2012_TO_2024
AR8_FRAMEWORK = ("DESNZ, Contracts for Difference Allocation Round 8: Allocation Framework and "
                 "Pot and Price Notice, 6 Jul 2026")
AR8_PRICE_NOTICE = "https://assets.publishing.service.gov.uk/media/6a4b8dd7732d8e7ce5f53d4d/CfD-AR8-pot-and-price-notice.pdf"
#: AR8 administrative strike prices, 2024 prices (AR8 Pot and Price Notice, Table 2; frozen at AR7's levels).
AR8_ASP_2024 = {"offshore": 113.0, "offshore_scotland": 113.0, "floating_offshore": 271.0, "onshore": 92.0,
                "solar": 75.0, "tidal": 371.0}
#: The low case: each technology at its AR6 clearing price (strike-table row, 2012 prices).
LOW_ROUND = "AR6"
LOW_TECHNOLOGY = {"offshore": "offshore", "offshore_scotland": "offshore", "floating_offshore": "floating_offshore",
                  "onshore": "onshore", "solar": "solar", "tidal": "tidal"}
SCRIPT = "scripts/market/build_ar8_modelled_strikes.py"


def published_rows(path: Path = STRIKE_TABLE) -> pd.DataFrame:
    """The strike table without the modelled round."""
    table = pd.read_csv(path)
    return table.loc[~table["cfd_round"].astype(str).str.strip().isin(REPLACED_ROUNDS)]


def _published(table: pd.DataFrame, cfd_round: str, technology: str) -> float:
    rows = table.loc[table["cfd_round"].eq(cfd_round) & table["technology"].eq(technology)]
    if len(rows) != 1:
        raise ValueError(f"expected one {cfd_round} {technology} row in the strike table, found {len(rows)}")
    return float(rows["strike_price_gbp_per_mwh_2012"].iloc[0])


def modelled_strikes(snapshot=None) -> pd.DataFrame:
    """The levelised strike per modelled technology at each AR8 build year and their mean (2024 prices)."""
    snapshot = cfd_strikes.load_snapshot() if snapshot is None else snapshot
    rows = []
    for technology, vintage in MODELLED.items():
        by_year = {year: cfd_strikes.levelised_strike(vintage, year, snapshot=snapshot, model_year=MODEL_YEAR)
                   for year in BUILD_YEARS}
        strikes = {year: r["strike_gbp_per_mwh_2024"] for year, r in by_year.items()}
        mean = sum(strikes.values()) / len(strikes)
        first = by_year[BUILD_YEARS[0]]
        rows.append({"technology": technology, "vintage_technology": vintage,
                     **{f"strike_{year}_gbp_per_mwh_2024": s for year, s in strikes.items()},
                     "strike_gbp_per_mwh_2024": mean, "strike_gbp_per_mwh_2012": mean / CPI_2012_TO_2024,
                     "capex_calibration_factor": first["capex_calibration_factor"],
                     "private_rate": first["private_rate"], "load_factor": first["load_factor"],
                     "economic_life_years": first["economic_life_years"]})
    return pd.DataFrame(rows).set_index("technology")


def table_rows(strikes: pd.DataFrame, published: pd.DataFrame) -> list[dict]:
    """The ``AR8_modelled`` rows of the strike table (2012 prices), with their derivations."""
    years = " and ".join(str(y) for y in BUILD_YEARS)
    method = (f"Modelled AR8 strike (user decision 27 Oct 2026, F18): the mean of the {years} build years' levelised "
              "strikes (cfd_strikes.levelised_strike: economic-life amortisation at the F29 v2.0 private rate, EGC 2025 "
              "Annex A load factor and use of system, capex calibrated to the last cleared auction in "
              "data/market/support/cfd_strike_cost_calibration.csv). AR8 plant generating in 2030 is built in 2029 or "
              "2030 (delivery years 2029/30 and 2030/31; " + AR8_FRAMEWORK + "). 2012 prices = 2024 prices / "
              f"{CPI_2012_TO_2024:.6f} (ONS CPI 133.9/96.1, the bill's cfd_strike_indexation).")
    rows = {}
    for technology in MODELLED:
        r = strikes.loc[technology]
        parts = ", ".join(f"{y} {r[f'strike_{y}_gbp_per_mwh_2024']:.2f}" for y in BUILD_YEARS)
        rows[technology] = (r["strike_gbp_per_mwh_2012"],
                            f"{method} {r['vintage_technology']}: {parts}; mean {r['strike_gbp_per_mwh_2024']:.2f} "
                            f"GBP/MWh 2024 prices (capex factor {r['capex_calibration_factor']:.4f}, private rate "
                            f"{r['private_rate']:.4f}, load factor {r['load_factor']:.3f}).")
    scot_ratio = _published(published, "AR7", "offshore_scotland") / _published(published, "AR7", "offshore")
    offshore = strikes.loc["offshore", "strike_gbp_per_mwh_2012"]
    rows["offshore_scotland"] = (
        offshore * scot_ratio,
        f"Modelled AR8 Scottish offshore (Pot 3 clears transmission zones 1-12 separately, AR8 Pot and Price Notice): "
        f"the modelled AR8 offshore strike times AR7's Scotland-to-rest ratio {scot_ratio:.6f} (64.23/65.45, 2012 prices).")
    rows["floating_offshore"] = (
        _published(published, "AR7", "floating_offshore"),
        "AR8 floating offshore wind is not modelled: held at the AR7 floating clearing price. No 2030 unit settles on "
        "this row (the floating projects in the fleet carry their own AR6 or AR7 contracts).")
    rows["nuclear"] = (
        _published(published, "HPC", "nuclear"),
        "Not an AR8 technology: the Hinkley Point C strike (89.50 since Sizewell C's financial close), a fallback for "
        "CfD-labelled nuclear without a round only. Hinkley Point C carries its own HPC round.")
    sources = {t: (SCRIPT + "; " + AR8_PRICE_NOTICE) for t in ROW_ORDER}
    sources["nuclear"] = "https://www.nao.org.uk/wp-content/uploads/2017/06/Hinkley-Point-C.pdf"
    return [{"cfd_round": MODELLED_ROUND, "technology": t, "delivery_year": DELIVERY_YEARS[t],
             "strike_price_gbp_per_mwh_2012": f"{rows[t][0]:.4f}", "capacity_mw": "", "price_basis": "2012 prices",
             "source": sources[t], "notes": rows[t][1]} for t in ROW_ORDER]


def render_table(path: Path, rows: list[dict]) -> str:
    """The table's text with the modelled rows in place of the old ones (other lines byte for byte)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in header})
    new = buffer.getvalue().splitlines(keepends=True)
    kept, position = [lines[0]], None
    for line in lines[1:]:
        cfd_round = next(csv.reader([line]))[0].strip() if line.strip() else ""
        if cfd_round in REPLACED_ROUNDS:
            position = len(kept) if position is None else position
            continue
        kept.append(line)
    if position is None:
        position = len(kept)
    return "".join(kept[:position] + new + kept[position:])


def _override(table_row: dict, value_2012: float, source: str) -> dict:
    return {"cfd_round": MODELLED_ROUND, "technology": table_row["technology"],
            "strike_price_gbp_per_mwh_2012": round(float(value_2012), 4), "price_basis": "2012 prices",
            "source": source, "replaces": float(table_row["strike_price_gbp_per_mwh_2012"]),
            "solve_side": "Bill layer only: the BM solves bid on the strike table's modelled AR8 strike, named in "
                          "bm_strike_table, which values the curtailment compensation."}


def twin_assumptions(base: dict, rows: list[dict], published: pd.DataFrame, case: str) -> dict:
    """A twin of the bill assumptions with the AR8 rows overridden to the ``low`` or ``high`` case."""
    out = copy.deepcopy(base)
    cfd = out["model_linked_rules"]["cfd"]
    overrides, held = [], []
    for row in rows:
        technology = row["technology"]
        if technology not in AR8_ASP_2024:
            continue  # nuclear is not an AR8 technology
        if case == "low":
            value = _published(published, LOW_ROUND, LOW_TECHNOLOGY[technology])
            if value >= float(row["strike_price_gbp_per_mwh_2012"]):
                # A low case never raises a strike: AR6 cleared above the modelled AR8 strike here.
                held.append(technology)
                continue
            source = (f"AR8 low case (F18): the AR6 {LOW_TECHNOLOGY[technology]} clearing price "
                      "(DESNZ AR6 results 3 Sep 2024, section (E)), the lowest recent clear.")
        elif case == "high":
            value = AR8_ASP_2024[technology] / CPI_2012_TO_2024
            source = (f"AR8 high case (F18): the AR8 administrative strike price {AR8_ASP_2024[technology]:.0f} GBP/MWh "
                      f"2024 prices ({AR8_PRICE_NOTICE}, Table 2; frozen at AR7's levels), / {CPI_2012_TO_2024:.6f} "
                      "to 2012 prices.")
        else:
            raise ValueError(f"case must be low or high, got {case!r}")
        overrides.append(_override(row, value, source))
    cfd["strike_overrides"] = overrides
    cfd["strike_overrides_note"] = (
        f"AR8 {case} case (F18, user decision 27 Oct 2026): a bill-layer range on the modelled AR8 round, written by "
        f"{SCRIPT}. Low = AR6 clearing prices where below the modelled strike, high = AR8 administrative strike "
        "prices. Difference payments settle at the override; the BM bids and the curtailment compensation stay on "
        "the solved strike (bm_strike_table)."
        + (f" Held at the modelled strike in the low case (AR6 cleared above it): {', '.join(held)}." if held else ""))
    cfd["bm_strike_table"] = cfd["strike_table"]
    cfd["bm_strike_table_note"] = ("The table the solved BM bid on (the central modelled AR8 strike): the curtailment "
                                   "compensation reconciles against those accepted bids (FES 2050 decision 41's rule).")
    return out


def render_json(data: dict) -> str:
    return json.dumps(data, indent=2) + "\n"


def build(snapshot=None, strikes: pd.DataFrame | None = None) -> dict[Path, str]:
    """Every file this script writes, as text (``strikes``: :func:`modelled_strikes`, computed if None)."""
    published = published_rows()
    strikes = modelled_strikes(snapshot) if strikes is None else strikes
    rows = table_rows(strikes, published)
    base = json.loads(BILL_ASSUMPTIONS.read_text(encoding="utf-8"))
    files = {STRIKE_TABLE: render_table(STRIKE_TABLE, rows)}
    for case, path in TWINS.items():
        files[path] = render_json(twin_assumptions(base, rows, published, case))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if a committed file differs from a rebuild")
    args = parser.parse_args()
    strikes = modelled_strikes()
    files = build(strikes=strikes)
    if args.check:
        stale = [str(p.relative_to(PROJECT_ROOT)) for p, text in files.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            raise SystemExit(f"stale AR8 files (rerun {SCRIPT}): {stale}")
        print("AR8 modelled strikes and range twins reproduce.")
        return
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="")
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    print(strikes.round(4).to_string())


if __name__ == "__main__":
    main()
