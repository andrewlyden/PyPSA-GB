#!/usr/bin/env python3
"""IC1 interconnector diagnostics for solved market months (plan Y8, acceptance item 3).

For each month, both stages (Stage 1 day-ahead from ``_wholesale_links.csv``, the BM from
``_balancing.nc``), per cable, per country and per IC1 market (tranche group):

* **flows:** import and export energy on the GB side (imports delivered, exports given up by
  GB: the B9 convention of ``analyze_market``), net, hours importing and exporting;
* **hours at capacity:** the share of hours a cable runs at 99% or more of its available
  import or export limit (``p_nom x p_max_pu(t)`` / ``p_nom x |p_min_pu(t)|``);
* **price spread:** the foreign import ask (and export bid) less the GB Stage 1 price, hour by
  hour, as a distribution, plus the share of hours inside the counter-trade gate (|spread|
  below 21 GBP/MWh); in the BM, the ask less the nodal price at the GB landing bus;
* **trade value:** what the LP charges for the schedule, valued by
  ``market_coupling.interconnector_trade_cost`` (the one valuation every consumer uses: the
  bid/ask epigraph plus each cable's share of the market's tranche cost), split into imports
  paid and exports received; the BM's re-trading (BM less Stage 1 plus the counter-trade
  premium); and the tranche occupancy of ``diagnose_interconnector_pricing``.

Cross-border energy is priced on the links, never on the foreign balance generators. Every
frame read from the NetCDF is re-indexed to its timestamps (``dataset_snapshots``) before it
meets the timestamped CSVs, and the join is checked.

    python scripts/market/ic1_month_diagnostics.py --study FES25_HT_2050 --months Jan Jul \
        --output-prefix resources/analysis/FES25_HT_2050_ic1_diagnostics
"""


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The repository root: the working directory under Snakemake (which runs a copy of this file),
# this file's grandparent when run as `python scripts/...`.
for _root in (Path.cwd(), Path(__file__).resolve().parents[2]):
    if (_root / "scripts").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.interconnectors.market_coupling import (  # noqa: E402
    _piecewise_links,
    counter_trade_premium_cost,
    dataset_snapshots,
    export_efficiency,
    interconnector_trade_cost,
    link_price_tables_from_dataset,
    piecewise_market_prices,
)
from scripts.market.diagnose_interconnector_pricing import trade_price_summary  # noqa: E402

MARKET_DIR = Path("resources/market")
AT_CAPACITY = 0.99
# The counter-trade gate (AGENTS.md, 27 Sep 2026): hours whose Stage 1 |GB - foreign|
# spread is below this are not counter-traded.
COUNTER_TRADE_GATE_GBP = 21.0
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def _frame(ds, key: str) -> pd.DataFrame:
    if key not in ds:
        return pd.DataFrame()
    frame = ds[key].to_pandas()
    frame.index = dataset_snapshots(ds, frame.index)
    frame.columns = frame.columns.astype(str)
    return frame


def read_month(name: str, market_dir: Path = MARKET_DIR, month: str | None = None) -> dict:
    """Everything one month's diagnostics need (``name`` the month scenario, ``month`` its label), timestamp-indexed."""
    import xarray as xr

    with xr.open_dataset(market_dir / f"{name}_balancing.nc", engine="netcdf4") as ds:
        links, ask_t = link_price_tables_from_dataset(ds)
        bm_flow = _frame(ds, "links_t_p0")
        p_max_t = _frame(ds, "links_t_p_max_pu")
        p_min_t = _frame(ds, "links_t_p_min_pu")
        nodal = _frame(ds, "buses_t_marginal_price")
        weights = pd.Series(ds["snapshots_generators"].values.astype(float), index=bm_flow.index) \
            if "snapshots_generators" in ds else pd.Series(1.0, index=bm_flow.index)
    s1_flow = pd.read_csv(market_dir / f"{name}_wholesale_links.csv", index_col=0, parse_dates=True)
    s1_flow.columns = s1_flow.columns.astype(str)
    gb_price = pd.read_csv(market_dir / f"{name}_wholesale_price.csv", index_col=0, parse_dates=True)["wholesale_price"]
    for label, frame in (("Stage 1 flows", s1_flow), ("GB price", gb_price)):
        missing = bm_flow.index.difference(frame.index)
        if len(missing):
            raise ValueError(f"{name}: {len(missing)} BM hours have no {label}; the indices do not join")
    signed = [link for link in _piecewise_links(links).values() if link in bm_flow.columns]
    return {"name": name, "month": month, "links": links, "ask_t": ask_t, "signed": signed,
            "flows": {"stage1": s1_flow.reindex(index=bm_flow.index, columns=signed).fillna(0.0),
                      "bm": bm_flow[signed].fillna(0.0)},
            "p_max_t": p_max_t, "p_min_t": p_min_t, "nodal": nodal, "gb_price": gb_price.reindex(bm_flow.index),
            "weights": weights}


def _limits(data: dict, link: str, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    links = data["links"]
    p_nom = float(links.at[link, "p_nom"])
    p_max = pd.Series(float(pd.to_numeric(links.get("p_max_pu", pd.Series(1.0, index=links.index)), errors="coerce")
                            .fillna(1.0).get(link, 1.0)), index=index)
    p_min = pd.Series(float(pd.to_numeric(links.get("p_min_pu", pd.Series(-1.0, index=links.index)), errors="coerce")
                            .fillna(-1.0).get(link, -1.0)), index=index)
    if link in data["p_max_t"].columns:
        p_max = data["p_max_t"][link].reindex(index).fillna(p_max)
    if link in data["p_min_t"].columns:
        p_min = data["p_min_t"][link].reindex(index).fillna(p_min)
    return p_nom * p_max.clip(lower=0.0), p_nom * (-p_min).clip(lower=0.0)


def cable_table(data: dict, stage: str) -> pd.DataFrame:
    """One row per cable: flows, hours at capacity, utilisation, trade value."""
    links, flows, w = data["links"], data["flows"][stage], data["weights"]
    cost = interconnector_trade_cost(flows, links, data["ask_t"])
    names = _piecewise_links(links)
    rows = []
    for cable, link in names.items():
        if link not in flows.columns:
            continue
        p = flows[link]
        eff = float(pd.to_numeric(pd.Series([links.at[link, "efficiency"]]), errors="coerce").fillna(1.0).iloc[0])
        imp_lim, exp_lim = _limits(data, link, p.index)
        importing, exporting = p.gt(1e-3), p.lt(-1e-3)
        at_imp = importing & (p >= AT_CAPACITY * imp_lim) & imp_lim.gt(0)
        at_exp = exporting & (-p >= AT_CAPACITY * exp_lim) & exp_lim.gt(0)
        c = cost[link] * w
        imported = (p.clip(lower=0.0) * eff * w).sum()
        exported = ((-p).clip(lower=0.0) * eff * w).sum()
        rows.append({
            "month": data["month"], "stage": stage, "cable": cable, "link": link,
            "country": str(links.at[link, "counterparty_country"]) if "counterparty_country" in links else "",
            "market": str(links.at[link, "market_tranche_group"]) if "market_tranche_group" in links else "",
            "capacity_mw": float(links.at[link, "p_nom"]),
            "import_gwh": imported / 1e3, "export_gwh": exported / 1e3,
            "import_sent_gwh": (p.clip(lower=0.0) * w).sum() / 1e3, "export_sent_gwh": ((-p).clip(lower=0.0) * w).sum() / 1e3, "net_import_gwh": (imported - exported) / 1e3,
            "hours": float(w.sum()), "hours_importing": float(w[importing].sum()), "hours_exporting": float(w[exporting].sum()),
            "share_hours_at_import_capacity": float(w[at_imp].sum() / w.sum()),
            "share_hours_at_export_capacity": float(w[at_exp].sum() / w.sum()),
            "share_hours_at_capacity": float(w[at_imp | at_exp].sum() / w.sum()),
            "mean_utilisation": float((p.abs() / np.maximum(np.where(p >= 0, imp_lim, exp_lim), 1e-9)).clip(upper=1.0).mean()),
            "import_cost_gbp": float(c[p > 0].sum()), "export_income_gbp": float(-c[p < 0].sum()),
            "net_trade_value_paid_abroad_gbp": float(c.sum()),
            "import_price_paid": float(c[p > 0].sum() / (p.clip(lower=0) * w).sum()) if importing.any() else np.nan,
            "export_price_received": float(-c[p < 0].sum() / ((-p).clip(lower=0) * w).sum()) if exporting.any() else np.nan,
        })
    table = pd.DataFrame(rows)
    if stage == "bm" and not table.empty:
        premium = counter_trade_premium_cost(flows, data["flows"]["stage1"], links).mul(w, axis=0).sum()
        s1_cost = interconnector_trade_cost(data["flows"]["stage1"], links, data["ask_t"]).mul(w, axis=0).sum()
        table["counter_trade_premium_gbp"] = table["link"].map(premium).fillna(0.0)
        table["retrading_cost_gbp"] = (table["net_trade_value_paid_abroad_gbp"] - table["link"].map(s1_cost).fillna(0.0)
                                       + table["counter_trade_premium_gbp"])
        table["gross_schedule_adjustment_gwh"] = table["link"].map(
            (flows - data["flows"]["stage1"]).abs().mul(w, axis=0).sum() / 1e3)
    return table


def group_table(cables: pd.DataFrame, key: str) -> pd.DataFrame:
    sums = ["capacity_mw", "import_gwh", "export_gwh", "net_import_gwh", "import_cost_gbp", "export_income_gbp",
            "net_trade_value_paid_abroad_gbp"] + [c for c in ("counter_trade_premium_gbp", "retrading_cost_gbp",
                                                              "gross_schedule_adjustment_gwh") if c in cables]
    out = cables.groupby(["month", "stage", key])[sums].sum()
    cap = cables.assign(w_imp=cables["share_hours_at_import_capacity"] * cables["capacity_mw"],
                        w_exp=cables["share_hours_at_export_capacity"] * cables["capacity_mw"],
                        w_any=cables["share_hours_at_capacity"] * cables["capacity_mw"])
    grouped = cap.groupby(["month", "stage", key])
    out["share_hours_at_import_capacity_capweighted"] = grouped["w_imp"].sum() / out["capacity_mw"]
    out["share_hours_at_export_capacity_capweighted"] = grouped["w_exp"].sum() / out["capacity_mw"]
    out["share_hours_at_capacity_capweighted"] = grouped["w_any"].sum() / out["capacity_mw"]
    out["import_price_paid"] = out["import_cost_gbp"] / (out["import_gwh"] * 1e3).replace(0, np.nan)
    out["export_price_received"] = out["export_income_gbp"] / (out["export_gwh"] * 1e3).replace(0, np.nan)
    return out.reset_index().rename(columns={key: "group"}).assign(grouping=key)


def spread_table(data: dict) -> pd.DataFrame:
    """Hourly spread distributions per cable: import ask and export bid against GB prices."""
    links, gb = data["links"], data["gb_price"]
    rows = []
    for cable, link in _piecewise_links(links).items():
        if link not in data["flows"]["bm"].columns:
            continue
        index = data["flows"]["bm"].index
        ask, bid = piecewise_market_prices(links, data["ask_t"], link, index)
        landing = str(links.at[link, "bus1"])
        nodal = data["nodal"][landing] if landing in data["nodal"].columns else pd.Series(np.nan, index=index)
        eff = export_efficiency(links.at[link, "efficiency"])
        for basis, reference in (("vs_GB_stage1_price", gb), ("vs_BM_landing_nodal_price", nodal)):
            for side, foreign in (("import_ask", ask), ("export_bid", bid * eff)):
                spread = (foreign - reference).dropna()
                if spread.empty:
                    continue
                row = {"month": data["month"], "cable": cable,
                       "market": str(links.at[link, "market_tranche_group"]) if "market_tranche_group" in links else "",
                       "basis": basis, "side": side, "hours": int(len(spread)),
                       "foreign_mean": float(foreign.mean()), "reference_mean": float(reference.mean()),
                       "spread_mean": float(spread.mean()), "spread_std": float(spread.std()),
                       "share_positive": float((spread > 0).mean()),
                       "share_inside_counter_trade_gate": float((spread.abs() < COUNTER_TRADE_GATE_GBP).mean())}
                row.update({f"p{q}": float(np.percentile(spread, q)) for q in PERCENTILES})
                rows.append(row)
    return pd.DataFrame(rows)


def tranche_table(data: dict) -> pd.DataFrame:
    frames = []
    for stage in ("stage1", "bm"):
        summary = trade_price_summary(data["flows"][stage], data["links"], data["ask_t"]).reset_index()
        summary.insert(0, "stage", stage)
        summary.insert(0, "month", data["month"])
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def cross_check(name: str, cables: pd.DataFrame, analysis_dir: Path) -> dict:
    """Against ``analyze_market``'s own ``_interconnector_flows.csv``: its gross flows are Stage 1
    (delivered imports, GB-side exports), its balancing flows the BM's sent side."""
    path = analysis_dir / f"{name}_interconnector_flows.csv"
    if not path.exists():
        return {"file": str(path), "available": False}
    flows = pd.read_csv(path).set_index("interconnector")
    bm = cables[cables["stage"].eq("bm")].set_index("cable")
    s1 = cables[cables["stage"].eq("stage1")].set_index("cable")
    out = {"file": str(path), "available": True}
    if "gross_import_MWh" in flows:
        out["stage1_import_max_abs_diff_mwh"] = float((s1["import_gwh"] * 1e3 - flows["gross_import_MWh"]).abs().max())
    if "gross_export_gb_side_MWh" in flows:
        out["stage1_export_max_abs_diff_mwh"] = float(
            (s1["export_gwh"] * 1e3 - flows["gross_export_gb_side_MWh"]).abs().max())
    if "balancing_import_MWh" in flows:
        out["bm_import_sent_max_abs_diff_mwh"] = float(
            (bm["import_sent_gwh"] * 1e3 - flows["balancing_import_MWh"]).abs().max())
        out["bm_export_sent_max_abs_diff_mwh"] = float(
            (bm["export_sent_gwh"] * 1e3 - flows["balancing_export_MWh"]).abs().max())
    if "day_ahead_trade_cost_GBP" in flows:
        out["stage1_trade_cost_max_abs_diff_gbp"] = float(
            (s1["net_trade_value_paid_abroad_gbp"] - flows["day_ahead_trade_cost_GBP"]).abs().max())
    if "balancing_trade_cost_GBP" in flows:
        out["bm_trade_cost_max_abs_diff_gbp"] = float(
            (bm["net_trade_value_paid_abroad_gbp"] - flows["balancing_trade_cost_GBP"]).abs().max())
    return out


def run(study: str, months: list[str], output_prefix: str, market_dir: Path = MARKET_DIR,
        analysis_dir: Path = Path("resources/analysis"), logger=None, scenarios: list[str] | None = None,
        family: str = "market") -> dict:
    """``scenarios`` names each month's scenario (the rule passes its study's); else ``<study>_<Mon>_<family>``."""
    import logging

    logger = logger or logging.getLogger("ic1_month_diagnostics")
    cables, spreads, tranches, checks = [], [], [], {}
    names = list(scenarios) if scenarios is not None else [f"{study}_{m}_{family}" for m in months]
    for month, name in zip(months, names):
        data = read_month(name, market_dir, month)
        month_cables = pd.concat([cable_table(data, "stage1"), cable_table(data, "bm")], ignore_index=True)
        cables.append(month_cables)
        spreads.append(spread_table(data))
        tranches.append(tranche_table(data))
        checks[month] = cross_check(name, month_cables, analysis_dir)
        logger.info("%s: cross-check %s", month, checks[month])
    cables = pd.concat(cables, ignore_index=True)
    groups = pd.concat([group_table(cables, "country"), group_table(cables, "market")], ignore_index=True)
    spreads = pd.concat(spreads, ignore_index=True)
    tranches = pd.concat(tranches, ignore_index=True)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {"cables": f"{prefix}_cables.csv", "groups": f"{prefix}_groups.csv",
             "spreads": f"{prefix}_spreads.csv", "tranches": f"{prefix}_tranches.csv"}
    cables.to_csv(paths["cables"], index=False)
    groups.to_csv(paths["groups"], index=False)
    spreads.to_csv(paths["spreads"], index=False)
    tranches.to_csv(paths["tranches"], index=False)
    totals = cables.groupby(["month", "stage"])[["import_gwh", "export_gwh", "net_import_gwh", "import_cost_gbp",
                                                "export_income_gbp", "net_trade_value_paid_abroad_gbp"]].sum()
    bm = cables[cables["stage"].eq("bm")].groupby("month")[["counter_trade_premium_gbp", "retrading_cost_gbp"]].sum()
    summary = {
        "study": study, "months": list(months), "outputs": paths,
        "at_capacity_threshold": AT_CAPACITY, "counter_trade_gate_gbp": COUNTER_TRADE_GATE_GBP,
        "totals": {f"{m}/{s}": v for (m, s), v in totals.round(3).to_dict("index").items()},
        "bm_retrading": bm.round(0).to_dict("index"),
        "capacity_weighted_share_hours_at_capacity": {
            f"{m}/{s}": float((g["share_hours_at_capacity"] * g["capacity_mw"]).sum() / g["capacity_mw"].sum())
            for (m, s), g in cables.groupby(["month", "stage"])},
        "cross_check_against_analyze_market": checks,
    }
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


if "snakemake" in globals():
    from scripts.utilities.logging_config import setup_logging

    _out = Path(snakemake.output.summary)
    run(snakemake.params.study, list(snakemake.params.months), str(_out).replace("_summary.json", ""),
        scenarios=list(snakemake.params.scenarios),
        logger=setup_logging(snakemake.log[0] if snakemake.log else "ic1_month_diagnostics"))
elif __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, help="the study id, e.g. FES25_HT_2050")
    ap.add_argument("--months", nargs="+", default=["Jan", "Jul"])
    ap.add_argument("--family", default="market", help="the month family: months are <study>_<Mon>_<family>")
    ap.add_argument("--output-prefix", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.study, a.months, a.output_prefix, family=a.family), indent=2, default=float))
