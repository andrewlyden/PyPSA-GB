"""FES 2050 decision 29: the 2050 market year on TYNDP 2026 NT+ prices against the central (DE) market year.

Ported from the D29 scratch script (``compare_year.py``, 3 Oct 2026) with its arithmetic
unchanged, so the post-solve pipeline (``scripts/public_ownership/post_solve_2050.py``) can run
it from the repository. Read-only on the solved months. For each of the two years it reads the
twelve BM networks, Stage 1 links, prices and dispatch and the constraint costs, and writes

* ``year_by_month.csv``: per run and month, prices, trade, BM cost and shedding, hydrogen;
* ``year_by_country.csv``: imports, exports and Stage 1 trade value by counterparty;
* ``year_compare.json``: the year totals per run, the annual pass, and NT+ minus central.

Usage (from the repository root)::

    python scripts/market/compare_tyndp26_year_2050.py --out-dir <directory>
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
STUDIES = {"central (DE)": "FES25_HT_2050_{m}_market", "NT+": "FES25_HT_2050_{m}_tyndp2026_prices"}
PASSES = (("central (DE)", "FES25_HT_2050"), ("NT+", "FES25_HT_2050_tyndp2026_prices"))
SEM = {"Northern Ireland": "SEM", "Ireland": "SEM", "Isle Of Man": "SEM", "Isle of Man": "SEM"}
TOTAL_KEYS = ("S1_shed_GWh", "S1_import_GWh", "S1_export_GWh", "BM_import_GWh", "BM_export_GWh",
              "S1_trade_value_GBPm", "BM_retrading_GBPm", "BM_cost_excl_shed_GBPm",
              "BM_shed_GWh", "S1_turbine_H2_GWh", "S1_electrolysis_GWh", "BM_turbine_H2_GWh")


def month_rows(label: str, sid: str, market_dir: Path, monthly: list, countries: list) -> None:
    """Append one solved month's row (``monthly``) and its by-country rows (``countries``)."""
    import pypsa
    from scripts.interconnectors.market_coupling import (
        counter_trade_premium_cost, interconnector_trade_cost, link_price_tables)

    mo = sid.split("_")[2]
    n = pypsa.Network(str(market_dir / f"{sid}_balancing.nc"))
    sn = pd.DatetimeIndex(n.snapshots)
    w = n.snapshot_weightings.objective.reindex(n.snapshots).to_numpy()
    links, ask_t = link_price_tables(n)
    ic = links.index[links.get("foreign_market_pricing", pd.Series("", index=links.index)).astype(str)
                     .eq("piecewise_bidirectional")]
    country = links.loc[ic, "counterparty_country"].astype(str).replace(SEM)
    s1_links = pd.read_csv(market_dir / f"{sid}_wholesale_links.csv", index_col=0, parse_dates=True).reindex(index=sn)
    s1 = s1_links.reindex(columns=ic).fillna(0.0)
    bm = n.links_t.p0.reindex(index=sn, columns=ic).fillna(0.0)
    price = pd.read_csv(market_dir / f"{sid}_wholesale_price.csv", index_col=0, parse_dates=True).reindex(sn)
    price = price["wholesale_price"] if "wholesale_price" in price else price.iloc[:, 0]
    gb_loads = n.loads.index[~n.loads.bus.astype(str).str.contains("external|Ext_|EU_", case=False)]
    load = n.loads_t.p_set.reindex(index=sn, columns=gb_loads).fillna(0.0).sum(axis=1)
    disp = pd.read_csv(market_dir / f"{sid}_wholesale_dispatch.csv", index_col=0, parse_dates=True).reindex(sn)
    s1_shed = disp[[c for c in disp.columns if "shed" in c.lower()]].sum().sum() / 1e3
    cc = pd.read_csv(market_dir / f"{sid}_constraint_costs.csv").set_index("carrier").drop(index="TOTAL", errors="ignore")
    tv_s1 = interconnector_trade_cost(s1, links, ask_t).mul(w, axis=0)
    tv_bm = interconnector_trade_cost(bm, links, ask_t).mul(w, axis=0)
    prem = counter_trade_premium_cost(bm, s1, links).mul(w, axis=0)
    lk = n.links
    tur = lk.index[lk.carrier == "H2_turbine"]
    elz = lk.index[lk.carrier == "electrolysis"]
    s1_h2 = s1_links.reindex(columns=lk.index).fillna(0.0)
    bm_h2 = n.links_t.p0.reindex(index=sn, columns=lk.index).fillna(0.0)
    stores = n.stores_t.e
    h2 = stores[[c for c in stores.columns if "H2" in c or "hydrogen" in c.lower()]].sum(axis=1)

    def pos(f):
        return f.clip(lower=0).mul(w, axis=0).sum()

    def neg(f):
        return (-f).clip(lower=0).mul(w, axis=0).sum()

    for stage, frame in (("S1", s1), ("BM", bm)):
        imp, exp = pos(frame).groupby(country).sum() / 1e3, neg(frame).groupby(country).sum() / 1e3
        for c in imp.index:
            countries.append({"run": label, "month": mo, "stage": stage, "country": c,
                              "import_GWh": imp[c], "export_GWh": exp[c]})
    for c, v in (tv_s1.sum().groupby(country).sum() / 1e6).items():
        countries.append({"run": label, "month": mo, "stage": "S1", "country": c, "trade_value_GBPm": v})
    monthly.append({
        "run": label, "month": mo, "hours": float(w.sum()),
        "price_sum": float((price * w).sum()), "price_load": float((price * load * w).sum()),
        "load_MWh": float((load * w).sum()),
        "S1_shed_GWh": float(s1_shed),
        "S1_import_GWh": float(pos(s1).sum() / 1e3), "S1_export_GWh": float(neg(s1).sum() / 1e3),
        "BM_import_GWh": float(pos(bm).sum() / 1e3), "BM_export_GWh": float(neg(bm).sum() / 1e3),
        "S1_trade_value_GBPm": float(tv_s1.to_numpy().sum() / 1e6),
        "BM_retrading_GBPm": float((tv_bm.to_numpy().sum() - tv_s1.to_numpy().sum() + prem.to_numpy().sum()) / 1e6),
        "BM_cost_excl_shed_GBPm": float(cc.drop(index="load_shedding", errors="ignore")["net_cost"].sum() / 1e6),
        "BM_shed_GWh": float(cc.loc["load_shedding", "increase_MWh"] / 1e3) if "load_shedding" in cc.index else 0.0,
        "S1_turbine_H2_GWh": float(s1_h2[tur].clip(lower=0).mul(w, axis=0).sum().sum() / 1e3),
        "S1_electrolysis_GWh": float(s1_h2[elz].clip(lower=0).mul(w, axis=0).sum().sum() / 1e3),
        "BM_turbine_H2_GWh": float(bm_h2[tur].clip(lower=0).mul(w, axis=0).sum().sum() / 1e3),
        "BM_H2_store_end_TWh": float(h2.iloc[-1] / 1e6), "BM_H2_store_min_TWh": float(h2.min() / 1e6),
    })
    del n
    gc.collect()


def summarise(t: pd.DataFrame, market_dir: Path) -> dict:
    """Year totals per run, the annual passes, and NT+ minus central (``delta``)."""
    summary = {}
    for label, g in t.groupby("run", sort=False):
        s = g.drop(columns=["run", "month"]).sum(numeric_only=True)
        summary[label] = {
            "S1_price_mean": s.price_sum / s.hours, "S1_price_load_weighted": s.price_load / s.load_MWh,
            "wholesale_energy_GBPbn": s.price_load / 1e9, "load_TWh": s.load_MWh / 1e6,
            **{k: float(s[k]) for k in TOTAL_KEYS},
            "BM_H2_store_min_TWh": float(g.BM_H2_store_min_TWh.min()),
        }
    for label, sid in PASSES:
        p = json.loads((market_dir / f"{sid}_seasonal_pass.json").read_text())
        tr = pd.read_csv(market_dir / f"{sid}_seasonal_trajectory.csv", index_col=0)
        summary[label]["pass_objective_GBPbn"] = (p.get("objective_gbp") or float("nan")) / 1e9
        summary[label]["pass_store_min_TWh"] = float(tr["e_end_mwh"].min() / 1e6)
        summary[label]["pass_value_mean"] = float(tr["value_gbp_per_mwh_h2"].mean())
    c, v = summary["central (DE)"], summary["NT+"]
    summary["delta"] = {k: v[k] - c[k] for k in c if isinstance(c[k], float)}
    d = summary["delta"]
    d["per_3100kWh_from_S1_price"] = d["S1_price_load_weighted"] * 3.1
    d["per_3100kWh_from_BM_and_retrading"] = ((d["BM_cost_excl_shed_GBPm"] + d["BM_retrading_GBPm"]) * 1e6
                                              / (v["load_TWh"] * 1e6) * 3.1)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--market-dir", default="resources/market")
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)
    out, market_dir = Path(args.out_dir), Path(args.market_dir)
    out.mkdir(parents=True, exist_ok=True)
    monthly, countries = [], []
    for label, pattern in STUDIES.items():
        for mo in MONTHS:
            month_rows(label, pattern.format(m=mo), market_dir, monthly, countries)
            print(label, mo, "done", flush=True)
    t = pd.DataFrame(monthly)
    t["S1_price_mean"] = t.price_sum / t.hours
    t.to_csv(out / "year_by_month.csv", index=False)
    pd.DataFrame(countries).to_csv(out / "year_by_country.csv", index=False)
    summary = summarise(t, market_dir)
    (out / "year_compare.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["delta"], indent=1))
    print(t.pivot(index="month", columns="run", values="S1_price_mean").reindex(MONTHS).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
