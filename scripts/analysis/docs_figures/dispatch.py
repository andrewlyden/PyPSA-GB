"""Draw the figures of the dispatch chapter (docs/source/_static/figures/dispatch/).

Every figure illustrates a mechanism with the 2030 central case of the CP30 2030 study: the CENTRAL network
case (``no_acceleration``), whose balancing months take their Stage 1 from the ``stage1`` family. The example
week is 14-20 January 2030.

1. stage1_week.png (model/dispatch/wholesale): Stage 1 dispatch by technology over the example week, with
   demand, and the GB wholesale price below. Source: ``CP30_FFR_2030_Jan_stage1_wholesale.nc``.
2. balancing_week.png (model/dispatch/balancing): the balancing stage's moves off Stage 1 in the same week,
   turn-up above zero and turn-down below, by technology. Sources: the January Stage 1 network and
   ``CP30_FFR_2030_Jan_no_acceleration_balancing.nc``.
3. turn_down_bases.png (model/dispatch/balancing): the year's wind turn-down by support scheme, as solved
   (cheapest first) and re-shared pro rata by zone: volume and bid cost. Source: the twelve
   ``CP30_FFR_2030_<Mon>_no_acceleration_pro_rata_turn_down.csv``.
4. wind_bid_stack.png (model/dispatch/bids_offers_support): the wind fleet's balancing bids in January, unit by
   unit, ordered from cheapest to dearest to turn down, by support scheme. Sources:
   ``..._Jan_no_acceleration_pro_rata_turn_down.csv`` (each unit's scheme and pricing rule),
   ``..._contract_value_bids.csv`` (CfD contract-value bids) and the balancing network (ROC banding).
5. cfd_bid_week.png (model/dispatch/bids_offers_support): how a contract-value bid follows the Stage 1 price in
   the example week, for three CfD strikes, a ROC unit and a merchant unit. Sources: the Stage 1 price file and
   the strikes in the bids file.
6. ic_retrading_by_country.png (model/dispatch/interconnector_trading): the year's interconnector moves in the
   balancing stage by country, up (more import or less export) and down. Sources: the twelve Stage 1 and
   balancing networks (``links_t_p0``).
7. ic_week.png (model/dispatch/interconnector_trading): GB net imports in Stage 1 and after balancing over the
   example week. Same sources, January.
8. central_price_map.png (model/dispatch/central_dispatch): the January mean Stage 1 nodal price at each GB
   demand bus under central dispatch. Source: ``CP30_FFR_2030_Jan_no_acceleration_central_dispatch_wholesale.nc``.
9. central_vs_market_redispatch.png (model/dispatch/central_dispatch): monthly redispatch volume, market and
   central dispatch. Sources: the twelve ``..._no_acceleration_redispatch_summary.csv`` and
   ``..._no_acceleration_central_dispatch_redispatch_summary.csv``.
10. bill_sources.png (model/after_the_solve/consumer_bill/index): the 2030 household bill by Ofgem price-cap
    group, each line coloured by where it comes from (modelled, model-informed, published). Sources:
    ``resources/analysis/CP30_2030_no_acceleration_consumer_bill_future_components.csv`` and
    ``data/market/consumer_bill_benchmarks/bill_line_sources.csv``.

Reads resources/ (read only), so run it after those exist:

    python scripts/analysis/docs_figures/dispatch.py [--resources resources] [--only name ...]

Not part of any Snakemake rule: the figures are documentation assets, committed as PNG.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.analysis.docs_figures.common import (  # noqa: E402
    DIVERGING, FIGURES, INK, ROOT, SAGE, SEQUENTIAL, TECH, apply_style, base_map, carrier_group, draw_lines,
    load_land, load_network, map_figure, save,
)

OUT = FIGURES / "dispatch"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEK = ("2030-01-14 00:00", "2030-01-20 23:00")
STAGE1 = "market/CP30_FFR_2030_{month}_stage1_wholesale.nc"
MARKET = "market/CP30_FFR_2030_{month}_no_acceleration"
CENTRAL = "market/CP30_FFR_2030_{month}_no_acceleration_central_dispatch"
FLEX = {"DH_flexibility", "EV_smart_charging", "HP_flexibility", "IC_DSR", "SH_flexibility", "V2G_fleet",
        "residential_DSR"}  # consumer flexibility: demand, not storage
NOT_SHOWN = {"load_shedding", "churn_sink"}
# Two-series and support-scheme colours: the first two validated categorical slots, plus a neutral grey.
PAIR = (DIVERGING["low"], DIVERGING["high"])
SUPPORT = {"CfD": DIVERGING["low"], "ROC": DIVERGING["high"], "Merchant": "#8d9994"}


# ---- reading ----------------------------------------------------------------------------------------
def _static(ds, prefix: str, fields: list[str]) -> pd.DataFrame:
    return pd.DataFrame({f: ds[f"{prefix}_{f}"].values for f in fields if f"{prefix}_{f}" in ds.variables},
                        index=ds[f"{prefix}_i"].values)


def _series(ds, name: str) -> pd.DataFrame:
    frame = ds[name].to_pandas()
    stamps = pd.Series(pd.to_datetime(ds["snapshots_snapshot"].values), index=ds["snapshots"].values)
    frame.index = pd.DatetimeIndex(stamps.reindex(frame.index).values, name="snapshot")
    return frame


def read_run(path: Path, series=("generators_t_p", "storage_units_t_p", "links_t_p0", "loads_t_p_set")) -> dict:
    """Static tables and the named time series of a solved network, read with xarray (no PyPSA objects)."""
    with xr.open_dataset(path) as ds:
        run = {
            "buses": _static(ds, "buses", ["country", "carrier", "x", "y"]),
            "gens": _static(ds, "generators", ["bus", "carrier", "p_nom", "support_type", "ro_banding"]),
            "links": _static(ds, "links", ["bus0", "bus1", "carrier", "efficiency", "counterparty_country"]),
            "stores": _static(ds, "storage_units", ["bus", "carrier"]),
            "loads": _static(ds, "loads", ["bus", "carrier"]),
        }
        for name in series:
            run[name] = _series(ds, name)
    gb = set(run["buses"].index[run["buses"].country == "GB"])
    run["gb"] = gb
    return run


def interconnectors(run: dict) -> pd.DataFrame:
    links = run["links"]
    cc = links.counterparty_country.astype(str)
    return links[(links.carrier == "DC") & ~cc.isin(["", "nan", "None"])]


def gb_import(run: dict) -> pd.DataFrame:
    """Power each interconnector injects into GB (MW; negative = export). The foreign end is bus0."""
    ic = interconnectors(run)
    p0 = run["links_t_p0"][ic.index]
    return p0 * ic.efficiency.astype(float)


def generation_by_group(run: dict) -> pd.DataFrame:
    gens = run["gens"]
    gens = gens[gens.bus.isin(run["gb"]) & ~gens.carrier.isin(NOT_SHOWN)]
    p = run["generators_t_p"][gens.index.intersection(run["generators_t_p"].columns)]
    by = p.T.groupby(gens.carrier.reindex(p.columns).map(carrier_group)).sum().T
    links = run["links"]
    h2 = links[links.carrier == "H2_turbine"]  # hydrogen turbines: gas-group output at their GB end
    by["Gas"] = by.get("Gas", 0) + (run["links_t_p0"][h2.index] * h2.efficiency.astype(float)).sum(axis=1)
    return by


def storage_net(run: dict) -> pd.Series:
    stores = run["stores"]
    grid = stores.index[~stores.carrier.isin(FLEX)]
    return run["storage_units_t_p"][grid].sum(axis=1)


def demand(run: dict) -> pd.Series:
    """GB electricity demand: loads, consumer flexibility's net draw and electrolysis (MW)."""
    loads = run["loads"]
    gb_loads = loads.index[loads.bus.isin(run["gb"]) & (loads.carrier != "H2_demand")]
    stores = run["stores"]
    flex = stores.index[stores.carrier.isin(FLEX)]
    links = run["links"]
    electrolysis = links.index[links.carrier == "electrolysis"]
    return (run["loads_t_p_set"][gb_loads.intersection(run["loads_t_p_set"].columns)].sum(axis=1)
            - run["storage_units_t_p"][flex].sum(axis=1)
            + run["links_t_p0"][electrolysis].sum(axis=1))


def week(frame):
    return frame.loc[WEEK[0]:WEEK[1]]


def day_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set_xlim(pd.Timestamp(WEEK[0]), pd.Timestamp(WEEK[1]) + pd.Timedelta(hours=1))


def hourly_line(frame):
    """Each hourly value at the middle of its hour, held flat to the week's edges, for drawing as a smooth line."""
    middle = frame.copy()
    middle.index = frame.index + pd.Timedelta(minutes=30)
    first, last = frame.iloc[[0]].copy(), frame.iloc[[-1]].copy()
    first.index = pd.DatetimeIndex([pd.Timestamp(WEEK[0])])
    last.index = pd.DatetimeIndex([pd.Timestamp(WEEK[1]) + pd.Timedelta(hours=1)])
    return pd.concat([first, middle, last])


def stack(ax, frame: pd.DataFrame, sign: int = 1) -> None:
    """Stacked areas in TECH order (positive above zero, or the negative parts below), hourly points joined."""
    frame = hourly_line(frame)
    base = np.zeros(len(frame))
    x = frame.index
    for group in TECH:
        if group not in frame or not (frame[group].abs() > 1).any():
            continue
        values = frame[group].to_numpy() / 1e3
        ax.fill_between(x, base, base + sign * values, color=TECH[group], lw=0, alpha=0.92)
        base = base + sign * values


# ---- figures ----------------------------------------------------------------------------------------
def figure_stage1_week(resources: Path, out: Path) -> None:
    run = read_run(resources / STAGE1.format(month="Jan"))
    gen = week(generation_by_group(run))
    store = week(storage_net(run))
    imports = week(gb_import(run).sum(axis=1))
    churn = run["gens"].index[run["gens"].carrier == "churn_sink"]
    churn = week(run["generators_t_p"][churn.intersection(run["generators_t_p"].columns)].sum(axis=1))
    load = week(demand(run))
    up = gen.copy()
    up["Storage"] = store.clip(lower=0)
    up["Imports"] = imports.clip(lower=0)
    down = pd.DataFrame({"Storage": (-store.clip(upper=0) - churn), "Imports": -imports.clip(upper=0)})
    residual = (up.sum(axis=1) - down.sum(axis=1) - load).abs().max()
    print(f"  stage 1 week balance residual {residual:.0f} MW")
    price = week(pd.read_csv(resources / "market/CP30_FFR_2030_Jan_stage1_wholesale_price.csv", index_col=0,
                             parse_dates=True).wholesale_price)

    fig, (ax, axp) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True, height_ratios=[2.6, 1],
                                  gridspec_kw={"hspace": 0.12})
    stack(ax, up)
    stack(ax, down, sign=-1)
    load = hourly_line(load)
    ax.plot(load.index, load / 1e3, color=INK["text"], lw=1.6)
    ax.axhline(0, color=INK["axis"], lw=0.8)
    ax.set_ylabel("GW")
    ax.set_title("Stage 1 dispatch (below zero: storage charging and exports)", fontsize=10)
    handles = [Patch(color=TECH[g], label=g) for g in TECH if g in up and (up[g] > 1).any()]
    handles.append(Line2D([], [], color=INK["text"], lw=1.6, label="Demand"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    line = hourly_line(price)
    axp.plot(line.index, line, color=SAGE["deep"], lw=1.6)
    axp.set_ylabel("GBP/MWh")
    axp.set_title("GB wholesale price", fontsize=10)
    axp.set_ylim(bottom=min(0, price.min() - 5))
    day_axis(axp)
    save(fig, out)


def figure_balancing_week(resources: Path, out: Path) -> None:
    s1 = read_run(resources / STAGE1.format(month="Jan"), series=("generators_t_p", "storage_units_t_p",
                                                                    "links_t_p0"))
    bm = read_run(resources / f"{MARKET.format(month='Jan')}_balancing.nc",
                  series=("generators_t_p", "storage_units_t_p", "links_t_p0"))
    gens = bm["gens"]
    gens = gens[gens.bus.isin(bm["gb"]) & ~gens.carrier.isin(NOT_SHOWN)]
    cols = gens.index.intersection(bm["generators_t_p"].columns).intersection(s1["generators_t_p"].columns)
    delta = week(bm["generators_t_p"][cols]) - week(s1["generators_t_p"][cols])
    group = gens.carrier.reindex(cols).map(carrier_group)
    up = delta.clip(lower=0).T.groupby(group).sum().T
    down = (-delta.clip(upper=0)).T.groupby(group).sum().T
    stores = bm["stores"]
    grid = stores.index[~stores.carrier.isin(FLEX)]
    d_store = week(bm["storage_units_t_p"][grid]) - week(s1["storage_units_t_p"][grid])
    up["Storage"], down["Storage"] = d_store.clip(lower=0).sum(axis=1), (-d_store.clip(upper=0)).sum(axis=1)
    d_ic = week(gb_import(bm)) - week(gb_import(s1))
    up["Imports"], down["Imports"] = d_ic.clip(lower=0).sum(axis=1), (-d_ic.clip(upper=0)).sum(axis=1)
    print(f"  week turn-up {up.sum().sum() / 1e3:.0f} GWh, turn-down {down.sum().sum() / 1e3:.0f} GWh")

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    stack(ax, up)
    stack(ax, down, sign=-1)
    ax.axhline(0, color=INK["text"], lw=0.8)
    ax.set_ylabel("GW moved off the Stage 1 position")
    ax.text(0.005, 0.97, "Turned up", transform=ax.transAxes, va="top", fontsize=9, color=INK["muted"])
    ax.text(0.005, 0.03, "Turned down", transform=ax.transAxes, va="bottom", fontsize=9, color=INK["muted"])
    shown = [g for g in TECH if (g in up and (up[g] > 1).any()) or (g in down and (down[g] > 1).any())]
    ax.legend(handles=[Patch(color=TECH[g], label=g) for g in shown], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=9)
    lim = max(up.sum(axis=1).max(), down.sum(axis=1).max()) / 1e3 * 1.12
    ax.set_ylim(-lim, lim)
    day_axis(ax)
    save(fig, out)


def figure_turn_down_bases(resources: Path, out: Path) -> None:
    frames = [pd.read_csv(resources / f"{MARKET.format(month=m)}_pro_rata_turn_down.csv") for m in MONTHS]
    units = pd.concat(frames)
    units["scheme"] = units.support_type.map({"CfD": "CfD", "ROC": "ROC", "merchant": "Merchant"})
    by = units.groupby("scheme")[["our_MWh", "pro_rata_MWh", "our_cost_GBP", "pro_rata_cost_GBP"]].sum()
    by = by.reindex(["CfD", "ROC", "Merchant"])
    print(by.round(0))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"wspace": 0.32})
    x = np.arange(len(by))
    width = 0.36
    for ax, (cols, scale, unit, title) in zip(axes, [
            (("our_MWh", "pro_rata_MWh"), 1e6, "TWh", "Wind turned down"),
            (("our_cost_GBP", "pro_rata_cost_GBP"), 1e6, "GBP m", "Bid cost of that turn-down")]):
        for i, (col, label) in enumerate(zip(cols, ("Cheapest first (as solved)", "Pro rata by zone"))):
            values = by[col] / scale
            bars = ax.bar(x + (i - 0.5) * width, values, width=width * 0.92, color=PAIR[i], label=label)
            ax.bar_label(bars, fmt="{:,.0f}" if unit == "GBP m" else "{:,.1f}", fontsize=8.5, padding=2,
                         color=INK["text"])
        ax.set_xticks(x, by.index)
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    axes[0].legend(loc="upper right", fontsize=9)
    save(fig, out)


def figure_wind_bid_stack(resources: Path, out: Path) -> None:
    stem = resources / MARKET.format(month="Jan")
    units = pd.read_csv(f"{stem}_pro_rata_turn_down.csv").set_index("unit")
    units = units[units.pricing != "non_participant_penalty"]
    bids = pd.read_csv(f"{stem}_contract_value_bids.csv").set_index("generator")
    with xr.open_dataset(f"{stem}_balancing.nc") as ds:
        gens = _static(ds, "generators", ["carrier", "p_nom", "ro_banding"])
    support = yaml.safe_load((ROOT / "config/defaults/base/support.yaml").read_text())
    buyout = float(support["dispatch_costs"]["roc_buyout_prices"][2030])
    units["p_nom"] = gens.p_nom.reindex(units.index)
    bid = pd.Series(0.0, index=units.index)
    cfd = units.pricing == "cfd_contract_value"
    bid[cfd] = bids.mean_bid_gbp_per_mwh.reindex(units.index[cfd])
    roc = units.pricing == "roc_banding_x_buyout"
    bid[roc] = gens.ro_banding.reindex(units.index[roc]).astype(float) * buyout
    units["bid"] = bid
    units["scheme"] = units.support_type.map({"CfD": "CfD", "ROC": "ROC", "merchant": "Merchant"})
    units = units.dropna(subset=["bid", "p_nom"]).sort_values("bid")
    edges = np.concatenate([[0], units.p_nom.cumsum().to_numpy() / 1e3])
    print(units.groupby("scheme").agg(gw=("p_nom", lambda s: s.sum() / 1e3), bid=("bid", "median")))

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for scheme, colour in SUPPORT.items():
        sel = (units.scheme == scheme).to_numpy()
        ax.bar(edges[:-1][sel], units.bid.to_numpy()[sel], width=np.diff(edges)[sel], align="edge",
               color=colour, lw=0)
    ax.axhline(0, color=INK["text"], lw=0.8)
    merchant = (units.scheme == "Merchant").to_numpy()
    for x0, w in zip(edges[:-1][merchant], np.diff(edges)[merchant]):  # bid of zero: draw as a thick band
        ax.plot([x0, x0 + w], [0, 0], color=SUPPORT["Merchant"], lw=5, solid_capstyle="butt", zorder=3)
    start, end = edges[:-1][merchant].min(), (edges[:-1] + np.diff(edges))[merchant].max()
    ax.annotate("Merchant wind bids zero", xy=((start + end) / 2, 0), xytext=((start + end) / 2, 35),
                ha="center", fontsize=9, color=INK["text"],
                arrowprops={"arrowstyle": "-", "color": INK["muted"], "lw": 0.8})
    ax.set_xlim(0, edges[-1])
    ax.set_xlabel("Wind capacity in the balancing mechanism, cheapest to turn down first (GW)")
    ax.set_ylabel("Mean January bid (GBP/MWh)")
    totals = units.groupby("scheme").p_nom.sum() / 1e3
    ax.legend(handles=[Patch(color=c, label=f"{s} ({totals.get(s, 0):.0f} GW)") for s, c in SUPPORT.items()],
              loc="upper left", fontsize=9)
    save(fig, out)


def figure_cfd_bid_week(resources: Path, out: Path) -> None:
    price = week(pd.read_csv(resources / "market/CP30_FFR_2030_Jan_stage1_wholesale_price.csv", index_col=0,
                             parse_dates=True).wholesale_price)
    bids = pd.read_csv(resources / f"{MARKET.format(month='Jan')}_contract_value_bids.csv")
    wind = bids[bids.carrier == "wind_offshore"]
    strikes = {r: wind.loc[wind.cfd_round == r, "strike_price_gbp_per_mwh"].median()
               for r in ("AR4", "AR7", "Investment Contract")}
    support = yaml.safe_load((ROOT / "config/defaults/base/support.yaml").read_text())
    roc = 2.0 * float(support["dispatch_costs"]["roc_buyout_prices"][2030])
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    series = [
        (f"CfD, offshore AR4 (strike {strikes['AR4']:.0f})", strikes["AR4"] - price, SEQUENTIAL[2], "-"),
        (f"CfD, offshore AR7 (strike {strikes['AR7']:.0f})", strikes["AR7"] - price, SEQUENTIAL[4], "-"),
        (f"CfD, Investment Contract (strike {strikes['Investment Contract']:.0f})",
         strikes["Investment Contract"] - price, SEQUENTIAL[6], "-"),
        ("ROC, offshore at 2 ROCs/MWh", pd.Series(roc, index=price.index), DIVERGING["high"], "-"),
        ("Merchant", pd.Series(0.0, index=price.index), SUPPORT["Merchant"], "--"),
    ]
    for label, values, colour, style in series:
        values = hourly_line(values)
        ax.plot(values.index, values, color=colour, lw=1.8, ls=style, label=label)
    ax.axhline(0, color=INK["axis"], lw=0.8)
    ax.set_ylabel("Bid (GBP/MWh, positive: NESO pays)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    day_axis(ax)
    save(fig, out)


def figure_ic_retrading(resources: Path, out: Path) -> None:
    rows = []
    for month in MONTHS:
        s1 = read_run(resources / STAGE1.format(month=month), series=("links_t_p0",))
        bm = read_run(resources / f"{MARKET.format(month=month)}_balancing.nc", series=("links_t_p0",))
        delta = gb_import(bm) - gb_import(s1)
        country = interconnectors(bm).counterparty_country
        rows.append(pd.DataFrame({"up": delta.clip(lower=0).sum().groupby(country).sum(),
                                  "down": (-delta.clip(upper=0)).sum().groupby(country).sum()}))
    by = pd.concat(rows).groupby(level=0).sum() / 1e6
    by = by.loc[(by.up + by.down).sort_values().index]
    print(by.round(2), by.sum().round(2))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    y = np.arange(len(by))
    ax.barh(y, by.up, height=0.56, color=PAIR[1], label="Up: more import or less export")
    ax.barh(y, -by.down, height=0.56, color=PAIR[0], label="Down: less import or more export")
    for yi, (u, d) in enumerate(zip(by.up, by.down)):
        ax.text(u + 0.1, yi, f"{u:.1f}", va="center", fontsize=8.5, color=INK["text"])
        if d >= 0.05:
            ax.text(-d - 0.1, yi, f"{d:.1f}", va="center", ha="right", fontsize=8.5, color=INK["text"])
    ax.axvline(0, color=INK["text"], lw=0.8)
    ax.set_yticks(y, by.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-max(2.0, by.down.max() * 1.6), by.up.max() * 1.1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}"))
    ax.set_xlabel("Change from the Stage 1 flow over the year (TWh)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, out)


def figure_ic_week(resources: Path, out: Path) -> None:
    s1 = read_run(resources / STAGE1.format(month="Jan"), series=("links_t_p0",))
    bm = read_run(resources / f"{MARKET.format(month='Jan')}_balancing.nc", series=("links_t_p0",))
    a, b = week(gb_import(s1).sum(axis=1)) / 1e3, week(gb_import(bm).sum(axis=1)) / 1e3
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    a, b = hourly_line(a), hourly_line(b)
    ax.fill_between(a.index, a, b, color=SAGE["light"], alpha=0.6, lw=0, label="Re-traded in balancing")
    ax.plot(a.index, a, color=PAIR[0], lw=1.8, label="Stage 1 (day-ahead market)")
    ax.plot(b.index, b, color=PAIR[1], lw=1.8, label="After balancing")
    ax.axhline(0, color=INK["text"], lw=0.8)
    ax.set_ylabel("GB net import (GW; negative: export)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    day_axis(ax)
    save(fig, out)


def figure_central_price_map(resources: Path, out: Path) -> None:
    path = resources / f"{CENTRAL.format(month='Jan')}_wholesale.nc"
    with xr.open_dataset(path) as ds:
        buses = _static(ds, "buses", ["country", "x", "y"])
        loads = _static(ds, "loads", ["bus", "carrier"])
        price = ds["buses_t_marginal_price"].to_pandas()
    demand_buses = pd.Index(loads.bus[loads.carrier != "H2_demand"].unique())
    demand_buses = demand_buses[demand_buses.isin(buses.index[buses.country == "GB"])]
    demand_buses = demand_buses.intersection(price.columns)
    mean = price[demand_buses].mean()
    market = pd.read_csv(resources / "market/CP30_FFR_2030_Jan_stage1_wholesale_price.csv").wholesale_price.mean()
    print(f"  central nodal mean {mean.mean():.1f} (range {mean.min():.1f} to {mean.max():.1f}); "
          f"market {market:.1f}")
    net = load_network(path)
    land = load_land()
    fig, ax, key = map_figure(key_height=1.0)
    base_map(ax, land)
    draw_lines(ax, net, alpha=0.35)
    cmap = LinearSegmentedColormap.from_list("div", [DIVERGING["low"], DIVERGING["mid"], DIVERGING["high"]])
    lo, hi = np.percentile(mean, [1, 99])
    norm = TwoSlopeNorm(vcenter=market, vmin=min(lo, market - 1), vmax=max(hi, market + 1))
    order = mean.sort_values().index
    ax.scatter(buses.loc[order, "x"], buses.loc[order, "y"], c=mean[order].clip(lo, hi), cmap=cmap, norm=norm,
               s=14, linewidths=0.3, edgecolors="white", zorder=4)
    cax = key.inset_axes([0.15, 0.62, 0.7, 0.14])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.5, length=2)
    bar.set_label(f"Mean January price (GBP/MWh); grey = market price ({market:.0f})", fontsize=9,
                  color=INK["text"])
    save(fig, out, dpi=170)


def figure_central_vs_market(resources: Path, out: Path) -> None:
    def gross(pattern: str) -> pd.Series:
        values = {}
        for month in MONTHS:
            rs = pd.read_csv(resources / f"{pattern.format(month=month)}_redispatch_summary.csv")
            rs = rs[rs.carrier != "load_shedding"]
            values[month] = (rs.increase_MWh.sum() + rs.decrease_MWh.sum()) / 1e6
        return pd.Series(values)

    market, central = gross(MARKET), gross(CENTRAL)
    print(f"  redispatch TWh: market {market.sum():.1f}, central {central.sum():.1f}")
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    x = np.arange(12)
    width = 0.36
    ax.bar(x - width / 2, market, width=width * 0.92, color=PAIR[0],
           label=f"Market dispatch ({market.sum():.0f} TWh in the year)")
    ax.bar(x + width / 2, central, width=width * 0.92, color=PAIR[1],
           label=f"Central dispatch ({central.sum():.0f} TWh)")
    ax.set_xticks(x, MONTHS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Redispatch, up plus down (TWh)")
    ax.set_ylim(0, market.max() * 1.22)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    save(fig, out)


def figure_bill_sources(resources: Path, out: Path) -> None:
    lines = pd.read_csv(resources / "analysis/CP30_2030_no_acceleration_consumer_bill_future_components.csv")
    kinds = pd.read_csv(ROOT / "data/market/consumer_bill_benchmarks/bill_line_sources.csv")
    lines["kind"] = lines.component_id.map(kinds.set_index("component_id").source_type)
    missing = lines.loc[lines.kind.isna(), "component_id"].tolist()
    if missing:
        print(f"  lines without a source type: {missing}")
    lines["kind"] = lines.kind.fillna("published")
    table = lines.pivot_table(index="ofgem_group", columns="kind", values="annual_gbp_per_3100kwh_ex_vat",
                              aggfunc="sum").fillna(0.0)
    order = ["modelled", "model-informed", "published"]
    table = table.reindex(columns=order, fill_value=0.0)
    table = table.loc[table.sum(axis=1).sort_values().index]
    print(table.round(1), table.sum().round(1))
    colours = {"modelled": SEQUENTIAL[6], "model-informed": SEQUENTIAL[3], "published": SEQUENTIAL[1]}
    labels = {"modelled": "Modelled (from the solve)", "model-informed": "Model-informed (a rule on a modelled "
              "quantity)", "published": "Published charge"}
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for kind in order:
        values = table[kind].to_numpy()
        ax.barh(y, values, left=left, height=0.56, color=colours[kind], label=labels[kind],
                edgecolor="white", linewidth=1.0)
        left += values
    for yi, total in enumerate(left):
        ax.text(total + 4, yi, f"GBP {total:,.0f}", va="center", fontsize=9, color=INK["text"])
    ax.set_yticks(y, table.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.16)
    ax.set_xlabel("GBP a year for 3,100 kWh, before VAT (real 2025)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, out)


FIGURES_BY_NAME = {
    "stage1_week": figure_stage1_week,
    "balancing_week": figure_balancing_week,
    "turn_down_bases": figure_turn_down_bases,
    "wind_bid_stack": figure_wind_bid_stack,
    "cfd_bid_week": figure_cfd_bid_week,
    "ic_retrading_by_country": figure_ic_retrading,
    "ic_week": figure_ic_week,
    "central_price_map": figure_central_price_map,
    "central_vs_market_redispatch": figure_central_vs_market,
    "bill_sources": figure_bill_sources,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resources", type=Path, default=ROOT / "resources")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES_BY_NAME), default=None)
    args = parser.parse_args()
    apply_style()
    for name, draw in FIGURES_BY_NAME.items():
        if args.only and name not in args.only:
            continue
        print(name)
        draw(args.resources, OUT / f"{name}.png")


if __name__ == "__main__":
    main()
