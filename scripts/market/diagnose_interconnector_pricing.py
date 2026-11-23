"""What GB's cross-border trade actually earned and paid, market by market (item IC1).

The gate for IC1 asks two things of a solved run: that net exports no longer earn one
flat price, and that the import price rises with volume. Both are read here from the
saved schedules and the prices on the saved network's links, valued by
``market_coupling.interconnector_trade_cost``, the same function that reproduces the
LP objective. No solve.

Per market (the volume-tranche groups; cables outside a group are one market each):

- import and export energy, sent-side MWh;
- the volume-weighted price paid for imports and received for exports, tranche cost
  included, against the CP.29 annual levels;
- the spread of the hourly export price (a flat price has none);
- hours in each tranche, by direction, and the average price paid or received in each.

Usage::

    python -m scripts.market.diagnose_interconnector_pricing <scenario> [--stage wholesale|balancing]
        [--out path.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.interconnectors.market_coupling import (  # noqa: E402
    _piecewise_links,
    get_volume_tranche_groups,
    interconnector_trade_cost,
    link_price_tables_from_dataset,
    tranche_boundaries,
    tranche_cost,
)

MARKET_DIR = Path("resources/market")


def market_membership(links: pd.DataFrame) -> dict[str, list[str]]:
    groups = {name: g["links"] for name, g in get_volume_tranche_groups(links).items()}
    grouped = {link for members in groups.values() for link in members}
    for link in _piecewise_links(links).values():
        if link not in grouped:
            groups[f"ungrouped:{link}"] = [link]
    return groups


def trade_price_summary(flows: pd.DataFrame, links: pd.DataFrame, ask_t: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per market: volumes, realised prices and tranche occupancy."""
    signed = [c for c in _piecewise_links(links).values() if c in flows.columns]
    flows = flows[signed].fillna(0.0).astype(float)
    cost = interconnector_trade_cost(flows, links, ask_t)
    groups = get_volume_tranche_groups(links)
    rows = []
    for market, members in market_membership(links).items():
        members = [m for m in members if m in signed]
        if not members:
            continue
        net = flows[members].sum(axis=1)
        hour_cost = cost[members].sum(axis=1)
        imp, exp = net.gt(0.0), net.lt(0.0)
        imported, exported = float(net[imp].sum()), float(-net[exp].sum())
        export_price = (-hour_cost[exp] / -net[exp]) if exp.any() else pd.Series(dtype=float)
        levels = links.loc[members]
        row = {
            "market": market,
            "cables": len(members),
            "capacity_mw": float(pd.to_numeric(levels["p_nom"], errors="coerce").sum()),
            "hours": len(net),
            "import_mwh": imported,
            "export_mwh": exported,
            "net_import_mwh": imported - exported,
            "import_price_paid": float(hour_cost[imp].sum() / imported) if imported else np.nan,
            "export_price_received": float(-hour_cost[exp].sum() / exported) if exported else np.nan,
            "export_price_hourly_std": float(export_price.std()) if len(export_price) > 1 else np.nan,
            "export_price_hourly_distinct": int(export_price.round(4).nunique()),
            "cp29_import_level": float(pd.to_numeric(levels.get("market_import_level", np.nan), errors="coerce").mean()),
            "cp29_export_level": float(pd.to_numeric(levels.get("market_export_level", np.nan), errors="coerce").mean()),
        }
        group = groups.get(market)
        if group is not None:
            edges = [0.0] + tranche_boundaries(group) + [np.inf]
            for direction, mask, volume in (("import", imp, net), ("export", exp, -net)):
                for k in range(group["count"]):
                    in_k = mask & volume.gt(edges[k]) & volume.le(edges[k + 1])
                    row[f"{direction}_hours_tranche_{k + 1}"] = int(in_k.sum())
                    price = hour_cost[in_k] / net[in_k]
                    row[f"{direction}_avg_price_tranche_{k + 1}"] = float(price.mean()) if in_k.any() else np.nan
                    # What volume alone adds: the tranche cost per MWh traded, free
                    # of the hourly shape that the average price above carries.
                    adder = tranche_cost(net[in_k], group) / volume[in_k]
                    row[f"{direction}_tranche_adder_per_mwh_tranche_{k + 1}"] = (
                        float(adder.mean()) if in_k.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("market")


def _flows(scenario: str, stage: str) -> pd.DataFrame:
    if stage == "wholesale":
        return pd.read_csv(MARKET_DIR / f"{scenario}_wholesale_links.csv", index_col=0, parse_dates=True)
    import xarray as xr

    from scripts.interconnectors.market_coupling import dataset_snapshots

    with xr.open_dataset(MARKET_DIR / f"{scenario}_balancing.nc", engine="netcdf4") as ds:
        flow = ds["links_t_p0"].to_pandas()
        flow.index = dataset_snapshots(ds, flow.index)
    flow.columns = flow.columns.astype(str)
    return flow


def diagnose(scenario: str, stage: str = "wholesale") -> pd.DataFrame:
    import xarray as xr

    with xr.open_dataset(MARKET_DIR / f"{scenario}_balancing.nc", engine="netcdf4") as ds:
        links, ask_t = link_price_tables_from_dataset(ds)
    flows = _flows(scenario, stage)
    if not ask_t.empty:
        missing = flows.index.difference(ask_t.index)
        if len(missing):
            raise ValueError(f"{len(missing)} flow hours have no link price; the indices do not join")
    return trade_price_summary(flows, links, ask_t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenario")
    ap.add_argument("--stage", default="wholesale", choices=("wholesale", "balancing"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = diagnose(a.scenario, a.stage)
    pd.set_option("display.width", 200)
    print(out.T.to_string())
    if a.out:
        out.to_csv(a.out)


if __name__ == "__main__":
    main()
