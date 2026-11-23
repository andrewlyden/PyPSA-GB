"""Draw the figures of the documentation's model pages (docs/source/_static/figures/supply/).

Demand and consumer flexibility (docs/source/model/demand/placement.md, flexibility/consumer.md):

1. demand_winter_week_2030.png: GB electricity demand over an example winter week, fixed load and consumer
   flexibility as dispatched, with the Stage 1 wholesale price underneath. Source: the January Stage 1 solve of the
   2030 study (resources/market/CP30_FFR_2030_Jan_stage1_wholesale.nc, *_wholesale_price.csv).
2. demand_map_2030.png: annual demand at each network node, fixed load plus the consumption of the consumer
   flexibility units. Source: the 2030 build (resources/network/CP30_FFR_2030.nc).
3. consumer_flexibility.png: consumer-flexibility power and energy by type, 2030 and 2050. Source: the 2030 build
   and the 2050 build (resources/network/FES25_HT_2050.nc).

Model overview (docs/source/model/index.md):

4. capacity_2030_2050.png: installed capacity by technology group, 2030 and 2050 builds side by side.

Thermal (generation/thermal.md):

5. merit_order_2030.png: marginal cost against cumulative installed capacity, coloured by technology group (2030
   build; static marginal costs, gas CHP at its annual mean after the heat credit).
6. gas_availability_2030.png: available gas capacity through the year after planned and forced outages (2030 build,
   generators_t.p_max_pu).

Renewables (generation/renewables.md):

7. renewables_named_proxy_2030.png: onshore wind, offshore wind and solar capacity by CP30 region, named projects
   and proxy rows for the residual (2030 build, generator column fleet_origin).
8. renewables_capacity_factors_2030.png: monthly capacity factor and hourly duration curve of available output for
   offshore wind, onshore wind and solar (2030 build, generators_t.p_max_pu, weather year 2013).

Storage (flexibility/storage.md):

9. storage_2030_2050.png: storage power and energy by type, 2030 and 2050 builds.
10. storage_map_2030.png: grid batteries and long-duration storage at their nodes (2030 build).

Hydrogen (flexibility/hydrogen.md):

11. hydrogen_map_2030_2050.png: electrolysers and hydrogen turbines at their nodes, 2030 and 2050 builds.

Interconnectors (interconnectors.md):

12. interconnectors_map_2030.png: the interconnectors at their GB landing, with rating and counterparty (2030 build).
13. foreign_prices_2030_2050.png: the hourly import price by counterparty country, 2030 and 2050 builds (link
    marginal costs of the cross-border cables).

Reads resources/ (the built networks and one solved month), so run it after those exist:

    python scripts/analysis/docs_figures/supply.py [--resources resources] [--out docs/source/_static/figures/supply]

Not part of any Snakemake rule: the figures are documentation assets, committed as PNG. Style, colours and map
helpers come from scripts/analysis/docs_figures/common.py.
"""

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.analysis.docs_figures.common import (  # noqa: E402
    INK, ROOT, SAGE, SEQUENTIAL, TECH, apply_style, base_map, carrier_group, colour_key, dot, draw_lines,
    load_land, load_network, map_figure, save, size_key, size_keys,
)

NET_2030 = "network/CP30_FFR_2030.nc"
NET_2050 = "network/FES25_HT_2050.nc"
WEEK_MONTH = "resources/market/CP30_FFR_2030_Jan_stage1"
HOURS_2030 = pd.date_range("2030-01-01", periods=8760, freq="h")
YEAR_COLOURS = {"2030": SEQUENTIAL[2], "2050": SEQUENTIAL[5]}
YEAR_LABELS = {"2030": "2030 (Clean Power 2030)", "2050": "2050 (FES 2025 Holistic Transition)"}

FLEX_LABELS = {
    "EV_smart_charging": "EV smart charging",
    "V2G_fleet": "Vehicle-to-grid",
    "HP_flexibility": "Heat-pump pre-heating",
    "DH_flexibility": "District-heat pre-heating",
    "SH_flexibility": "Storage heaters",
    "IC_DSR": "Industrial and commercial DSR",
    "residential_DSR": "Residential DSR",
    "DACCS": "Direct air capture",
}
STORAGE_TYPES = ["Battery", "Domestic Battery", "LAES", "CAES", "Pumped Storage Hydroelectricity"]


# ---- reading ----------------------------------------------------------------------------------------
def time_series(path: Path, var: str, columns, chunk: int = 2000) -> pd.DataFrame:
    """Selected columns of a time-varying table, read in chunks to keep memory down."""
    with xr.open_dataset(path) as ds:
        index = f"{var}_i"
        have = pd.Index(ds[index].values)
        cols = [c for c in columns if c in have]
        parts = [ds[var].sel({index: cols[i:i + chunk]}).to_pandas() for i in range(0, len(cols), chunk)]
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


@lru_cache(maxsize=1)
def p_max_pu(path: Path) -> pd.DataFrame:
    """Every generator's hourly p_max_pu, read once in blocks of hours as float32 (column selection on the
    compressed file is slow, a full read is not)."""
    with xr.open_dataset(path) as ds:
        da = ds["generators_t_p_max_pu"]
        hours = da.sizes[da.dims[0]]
        blocks = [da.isel({da.dims[0]: slice(i, i + 1000)}).values.astype("float32") for i in range(0, hours, 1000)]
        columns = ds["generators_t_p_max_pu_i"].values
    return pd.DataFrame(np.vstack(blocks), columns=columns)


def static(path: Path, prefix: str, fields: list[str]) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        return pd.DataFrame({f: ds[f"{prefix}_{f}"].values for f in fields if f"{prefix}_{f}" in ds.variables},
                            index=ds[f"{prefix}_i"].values)


def cross_border(net: dict) -> pd.DataFrame:
    """Cross-border cables: DC links with one end on a foreign bus, with their GB end and country."""
    b, links = net["buses"], net["links"]
    gb = set(b.index[b.country == "GB"])
    dc = links[(links.carrier == "DC") & (links.bus0.isin(gb) != links.bus1.isin(gb))].copy()
    dc["gb_bus"] = np.where(dc.bus0.isin(gb), dc.bus0, dc.bus1)
    dc["foreign_bus"] = np.where(dc.bus0.isin(gb), dc.bus1, dc.bus0)
    dc["country"] = b.loc[dc.foreign_bus, "country"].values
    return dc


def capacity_by_group(net: dict) -> pd.DataFrame:
    """Installed GW by technology group; Gas split into unabated and with CCS; hydrogen turbines on their own."""
    gens = net["gens"]
    gens = gens[gens.carrier.isin(set(gens.carrier) - {"gas_supply"})]
    rows = gens.assign(group=gens.carrier.map(carrier_group))[["group", "carrier", "p_nom"]]
    links = net["links"]
    h2 = links[links.carrier == "H2_turbine"].assign(group="Hydrogen")[["group", "carrier", "p_nom"]]
    stores = net["stores"][net["stores"].carrier.isin(STORAGE_TYPES)].assign(group="Storage")
    imports = cross_border(net).assign(group="Imports", carrier="interconnector")
    rows = pd.concat([rows[rows.carrier != "EU_market"], h2, stores[["group", "carrier", "p_nom"]],
                      imports[["group", "carrier", "p_nom"]]])
    rows = rows[rows.group.notna()]
    rows["part"] = "main"
    rows.loc[rows.carrier.isin(["CCS Gas"]), "part"] = "ccs"
    return rows.groupby(["group", "part"]).p_nom.sum().unstack(fill_value=0) / 1000


# ---- demand -----------------------------------------------------------------------------------------
def figure_winter_week(resources: Path, out: Path) -> None:
    path = resources.parent / f"{WEEK_MONTH}_wholesale.nc"
    loads = static(path, "loads", ["bus", "carrier"])
    su = static(path, "storage_units", ["carrier"])
    elec = loads.index[loads.carrier != "H2_demand"]
    fixed = time_series(path, "loads_t_p_set", elec).sum(axis=1)
    flex = su[su.carrier.isin(FLEX_LABELS)]
    store = time_series(path, "storage_units_t_p_store", flex.index)
    dispatch = time_series(path, "storage_units_t_p_dispatch", flex.index).reindex(columns=store.columns).fillna(0)
    net = (store - dispatch).T.groupby(flex.carrier).sum().T
    hours = pd.date_range("2030-01-01", periods=len(fixed), freq="h")
    fixed.index = net.index = hours
    price = pd.read_csv(resources.parent / f"{WEEK_MONTH}_wholesale_price.csv", index_col=0, parse_dates=True)
    price = price["wholesale_price"]
    total = fixed + net.sum(axis=1)
    peak = total.idxmax()
    start = (peak - pd.Timedelta(days=peak.dayofweek)).normalize()
    start = min(start, hours[-1] - pd.Timedelta(days=7) + pd.Timedelta(hours=1))
    week = slice(start, start + pd.Timedelta(days=7) - pd.Timedelta(hours=1))
    parts = pd.DataFrame({
        "Fixed load, after demand response": fixed + net.get("IC_DSR", 0) + net.get("residential_DSR", 0),
        "Heat (heat pumps, district heat, storage heaters)": net[["HP_flexibility", "DH_flexibility",
                                                                  "SH_flexibility"]].sum(axis=1),
        "Electric vehicles (smart charging, vehicle-to-grid)": net[["EV_smart_charging", "V2G_fleet"]].sum(axis=1),
    }).loc[week] / 1000
    colours = [SEQUENTIAL[1], SEQUENTIAL[3], SEQUENTIAL[5]]
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax.stackplot(parts.index, parts.T.values, colors=colours, labels=parts.columns, linewidth=0)
    ax.plot(parts.index, parts.sum(axis=1), color=INK["text"], lw=1.0, label="Total demand")
    ax.set_ylabel("GW")
    ax.set_ylim(0, parts.sum(axis=1).max() * 1.22)
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, handlelength=1.4, columnspacing=1.2)
    ax.set_title("Demand, excluding electrolysis", fontsize=10)
    axp.plot(price.loc[week].index, price.loc[week].values, color=SAGE["deep"], lw=1.6)
    axp.set_ylabel("GBP/MWh")
    axp.set_title("Stage 1 wholesale price", fontsize=10)
    axp.set_ylim(bottom=min(0, price.loc[week].min() * 1.1))
    axp.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%a %d %b"))
    fig.tight_layout()
    save(fig, out)


def figure_demand_map(net: dict, land, path: Path, out: Path) -> None:
    loads = net["loads"]
    elec = loads[loads.carrier != "H2_demand"]
    energy = time_series(path, "loads_t_p_set", elec.index).sum()
    by_bus = energy.groupby(elec.bus).sum()
    flex = net["stores"][net["stores"].carrier.isin(FLEX_LABELS)]
    inflow = -time_series(path, "storage_units_t_inflow", flex.index).sum()
    flex_bus = inflow.groupby(flex.bus).sum()
    total = by_bus.add(flex_bus, fill_value=0) / 1e6  # TWh
    b = net["buses"]
    total = total[total.index.isin(b.index) & (total > 0.005)].sort_values(ascending=False)
    fig, ax, key = map_figure(key_height=1.1)
    base_map(ax, land)
    draw_lines(ax, net, alpha=0.35)
    per_twh = 60.0
    ax.scatter(b.loc[total.index, "x"], b.loc[total.index, "y"], s=total * per_twh, color=SEQUENTIAL[4],
               alpha=0.7, linewidths=0.4, edgecolors="white", zorder=4)
    colour_key(key, [dot(SEQUENTIAL[4], f"Annual demand at the node (GB {total.sum():.0f} TWh)")], ncol=1)
    size_key(key, size_keys([1, 5, 10], per_twh, "TWh"), top=0.55)
    save(fig, out, dpi=170)


def flex_table(net: dict) -> pd.DataFrame:
    s = net["stores"][net["stores"].carrier.isin(FLEX_LABELS)]
    return s.assign(e=s.p_nom * s.max_hours).groupby("carrier")[["p_nom", "e"]].sum() / 1000


def paired_bars(axes, tables: dict, order: list[str], labels: dict, columns: list[str], units: list[str],
                titles: list[str]) -> None:
    """Horizontal bars, two years per row, one panel per measure."""
    y = np.arange(len(order))
    h = 0.36
    for ax, column, unit, title in zip(axes, columns, units, titles):
        for k, (year, table) in enumerate(tables.items()):
            values = table[column].reindex(order).fillna(0).values
            ax.barh(y + (k - 0.5) * h * 1.08, values, height=h, color=YEAR_COLOURS[year], label=YEAR_LABELS[year])
            for yi, v in zip(y + (k - 0.5) * h * 1.08, values):
                if v > 0:
                    ax.text(v, yi, f" {v:,.0f}" if v >= 10 else f" {v:.1f}", va="center", fontsize=7.5,
                            color=INK["muted"])
        ax.set_yticks(y, [labels.get(c, c) for c in order])
        ax.set_xlabel(unit)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16)
    axes[0].invert_yaxis()  # shared y: invert once


def figure_flexibility(net30: dict, net50: dict, out: Path) -> None:
    tables = {"2030": flex_table(net30), "2050": flex_table(net50)}
    order = [c for c in FLEX_LABELS if any(c in t.index for t in tables.values())]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    paired_bars(axes, tables, order, FLEX_LABELS, ["p_nom", "e"], ["GW", "GWh"],
                ["Power (charging limit)", "Energy buffer"])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.16), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.12)
    save(fig, out)


# ---- supply overview --------------------------------------------------------------------------------
def figure_capacity(net30: dict, net50: dict, out: Path) -> None:
    caps = {"2030": capacity_by_group(net30), "2050": capacity_by_group(net50)}
    order = list(TECH)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, sharex=True)
    hatches = {"main": None, "ccs": "////"}
    for ax, (year, table) in zip(axes, caps.items()):
        table = table.reindex(order).fillna(0)
        for gi, group in enumerate(order):
            left = 0.0
            for part in ("main", "ccs"):
                v = table.at[group, part] if part in table else 0
                if v <= 0:
                    continue
                ax.barh(gi, v, left=left, height=0.58, color=TECH[group], hatch=hatches[part],
                        edgecolor="white" if hatches[part] else TECH[group], linewidth=0)
                left += v
            ax.text(left + 2, gi, f"{left:.0f}", va="center", fontsize=8.5, color=INK["text"])
        ax.set_yticks(y, order)
        ax.set_xlabel("Installed capacity, GW")
        ax.set_title(YEAR_LABELS[year], fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()  # shared y: invert once
    axes[0].set_xlim(0, max(t.sum(axis=1).max() for t in caps.values()) * 1.12)
    grey = INK["muted"]
    handles = [Patch(facecolor=grey, label="Unabated gas"),
               Patch(facecolor=grey, hatch="////", edgecolor="white", label="Gas with CCS")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.04), fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out)


# ---- thermal ----------------------------------------------------------------------------------------
def figure_merit_order(path: Path, out: Path) -> None:
    g = static(path, "generators", ["bus", "carrier", "p_nom", "marginal_cost"])
    g = g[~g.carrier.isin(["load_shedding", "EU_market"])].copy()
    chp = g.index[g.carrier == "CHP"]
    g.loc[chp, "marginal_cost"] = time_series(path, "generators_t_marginal_cost", chp).mean()
    g["group"] = g.carrier.map(carrier_group)
    g = g[g.group.notna() & (g.p_nom > 0)]
    g["mc"] = g.marginal_cost.round(0)
    blocks = g.groupby(["mc", "group"], sort=True).p_nom.sum().reset_index()
    order = {grp: i for i, grp in enumerate(TECH)}
    blocks = blocks.assign(o=blocks.group.map(order)).sort_values(["mc", "o"])
    blocks["left"] = blocks.p_nom.cumsum().shift(fill_value=0) / 1000
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for _, r in blocks.iterrows():  # zero-cost blocks drawn as a thin strip so they stay visible
        ax.bar(r.left, r.mc if abs(r.mc) >= 5 else 5, width=r.p_nom / 1000, align="edge", color=TECH[r.group],
               linewidth=0)
    ax.axhline(0, color=INK["axis"], lw=0.8, zorder=0)
    ax.set_xlabel("Cumulative installed capacity, GW")
    ax.set_ylabel("Marginal cost, GBP/MWh")
    ax.set_xlim(0, blocks.left.iloc[-1] + blocks.p_nom.iloc[-1] / 1000)
    ax.set_ylim(-420, 430)
    shown = [grp for grp in TECH if grp in set(blocks.group)]
    ax.legend(handles=[Patch(color=TECH[grp], label=grp) for grp in shown], loc="upper left", ncol=2, fontsize=9)

    def span(mc_low: float, mc_high: float, groups: set) -> tuple[float, float, float]:
        sel = blocks[(blocks.mc >= mc_low) & (blocks.mc <= mc_high) & blocks.group.isin(groups)]
        return sel.left.min(), (sel.left + sel.p_nom / 1000).max(), sel.mc.max()

    def note(text: str, x: float, y: float, ha: str = "center") -> None:
        ax.text(x, y, text, ha=ha, va="bottom", fontsize=8.5, color=INK["text"])
    lo, hi, top = span(100, 160, {"Gas"})
    note("CCGTs", (lo + hi) / 2, top + 8)
    lo, hi, top = span(165, 200, {"Gas"})
    note("OCGTs and\ngas engines", (lo + hi) / 2, top + 8)
    lo, hi, top = span(300, 400, {"Gas"})
    note("oil", hi, top + 6, ha="right")
    lo, hi, top = span(50, 80, {"Gas"})
    note("gas CHP, after\nits heat credit", lo - 1, top - 10, ha="right")
    lo, hi, top = span(1, 20, {"Nuclear", "Other renewables"})
    note("nuclear and biomass", hi, -38, ha="right")
    lo, hi, _ = span(0, 0, set(TECH))
    note("zero-cost: CfD and merchant renewables, hydro, waste", (lo + hi) / 2, 14)
    lo, hi, _ = span(-400, -1, set(TECH))
    note("ROC-supported renewables:\nnegative cost (the certificates they lose)", hi + 2, -95, ha="left")
    save(fig, out)


def figure_gas_availability(path: Path, out: Path) -> None:
    g = static(path, "generators", ["carrier", "p_nom", "p_max_pu"])
    gas = g[g.carrier.isin(["CCGT", "OCGT", "gas_engine"])]
    pm = p_max_pu(path)
    pm = pm[[c for c in gas.index if c in pm.columns]]
    avail = (pm * gas.loc[pm.columns, "p_nom"]).sum(axis=1)
    static_part = (gas.drop(pm.columns, errors="ignore").p_nom * gas.drop(pm.columns, errors="ignore").p_max_pu)
    avail = (avail + static_part.sum()) / 1000
    avail.index = HOURS_2030[: len(avail)]
    daily = avail.resample("D").mean()
    installed = gas.p_nom.sum() / 1000
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhline(installed, color=INK["muted"], lw=1.0, ls=(0, (4, 3)))
    ax.text(daily.index[3], installed + 0.4, f"installed {installed:.1f} GW", fontsize=8.5, color=INK["muted"])
    ax.plot(daily.index, daily.values, color=TECH["Gas"], lw=1.6, label="Available (daily mean)")
    ax.set_ylabel("Gas capacity, GW")
    ax.set_ylim(installed * 0.7, installed * 1.04)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.legend(loc="lower left")
    save(fig, out)


# ---- renewables -------------------------------------------------------------------------------------
REGIONS = ["North Scotland", "South Scotland", "North England", "Midlands", "Wales", "East England",
           "South West", "South East"]
VRE = {"wind_offshore": "Offshore wind", "wind_onshore": "Onshore wind", "solar_pv": "Solar"}


def figure_named_proxy(path: Path, out: Path) -> None:
    g = static(path, "generators", ["carrier", "p_nom", "fleet_origin", "cp30_region"])
    g = g[g.carrier.isin(VRE)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(REGIONS))
    for ax, (carrier, label) in zip(axes, VRE.items()):
        t = g[g.carrier == carrier].groupby(["cp30_region", "fleet_origin"]).p_nom.sum().unstack(fill_value=0)
        t = t.reindex(REGIONS).fillna(0) / 1000
        named = t.get("named", pd.Series(0, index=t.index))
        proxy = t.get("increment", pd.Series(0, index=t.index))
        colour = TECH[label]
        ax.barh(y, named, height=0.58, color=colour, label="Named projects")
        ax.barh(y, proxy, left=named, height=0.58, color=colour, alpha=0.38, label="Proxy rows for the residual")
        for yi, v in zip(y, named + proxy):
            if v >= 0.05:
                ax.text(v, yi, f" {v:.1f}", va="center", fontsize=7.5, color=INK["muted"])
        ax.set_title(f"{label} ({(named.sum() + proxy.sum()):.0f} GW)", fontsize=10)
        ax.set_xlabel("GW")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.2)
    axes[0].set_yticks(y, REGIONS)
    axes[0].invert_yaxis()
    grey = INK["muted"]
    fig.legend(handles=[Patch(color=grey, label="Named projects (REPD, within their life or expected online)"),
                        Patch(color=grey, alpha=0.38, label="Proxy rows filling the residual to ES5")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out)


def available_cf(path: Path, gens: pd.DataFrame) -> pd.DataFrame:
    """Capacity-weighted hourly p_max_pu per carrier (available output before curtailment)."""
    out = {}
    for carrier in VRE:
        sel = gens[gens.carrier == carrier]
        pm = p_max_pu(path)
        cols = [c for c in sel.index if c in pm.columns]
        total = pm[cols].to_numpy(dtype="float64") @ sel.loc[cols, "p_nom"].to_numpy()
        total = pd.Series(total)
        out[carrier] = total / sel.p_nom.sum()
    df = pd.DataFrame(out)
    df.index = HOURS_2030[: len(df)]
    return df


def figure_capacity_factors(path: Path, out: Path) -> None:
    g = static(path, "generators", ["carrier", "p_nom"])
    cf = available_cf(path, g[g.carrier.isin(VRE)])
    monthly = cf.resample("MS").mean()
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(10, 4.2))
    for carrier, label in VRE.items():
        ax.plot(np.arange(12), monthly[carrier].values * 100, color=TECH[label], marker="o", markersize=3.5,
                label=f"{label} ({cf[carrier].mean() * 100:.0f}% a year)")
        curve = np.sort(cf[carrier].values)[::-1]
        axd.plot(np.arange(len(curve)), curve * 100, color=TECH[label], label=label)
    ax.set_xticks(np.arange(12), [d.strftime("%b")[0] for d in monthly.index])
    ax.set_ylabel("Capacity factor, %")
    ax.set_title("Monthly mean", fontsize=10)
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left", fontsize=8.5)
    axd.set_xlabel("Hours of the year, sorted")
    axd.set_ylabel("Available output, % of capacity")
    axd.set_title("Duration curve", fontsize=10)
    axd.set_xlim(0, 8760)
    axd.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, out)


# ---- storage ----------------------------------------------------------------------------------------
STORAGE_LABELS = {
    "Battery": "Grid batteries",
    "Domestic Battery": "Domestic batteries",
    "Pumped hydro": "Pumped hydro",
    "LDES new": "Long-duration new build (CP30)",
    "LAES": "Liquid air",
    "CAES": "Compressed air",
}


def storage_table(net: dict, cp30: bool) -> pd.DataFrame:
    s = net["stores"][net["stores"].carrier.isin(STORAGE_TYPES)].copy()
    s["type"] = s.carrier.replace({"Pumped Storage Hydroelectricity": "Pumped hydro"})
    if cp30:  # CP30's long-duration category: today's pumped hydro, then new build
        ldes = s.carrier == "LAES"
        s.loc[ldes, "type"] = np.where(s.index[ldes].str.contains("_existing_"), "Pumped hydro", "LDES new")
    return s.assign(e=s.p_nom * s.max_hours).groupby("type")[["p_nom", "e"]].sum() / 1000


def figure_storage(net30: dict, net50: dict, out: Path) -> None:
    tables = {"2030": storage_table(net30, cp30=True), "2050": storage_table(net50, cp30=False)}
    order = [t for t in STORAGE_LABELS if any(t in tb.index for tb in tables.values())]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    paired_bars(axes, tables, order, STORAGE_LABELS, ["p_nom", "e"], ["GW", "GWh"], ["Power", "Energy"])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.12)
    save(fig, out)


def figure_storage_map(net: dict, land, out: Path) -> None:
    s = net["stores"]
    b = net["buses"]
    s = s[s.carrier.isin(["Battery", "LAES"]) & s.bus.isin(b.index)]
    fig, ax, key = map_figure()
    base_map(ax, land)
    draw_lines(ax, net, alpha=0.35)
    per_gw = 260.0
    colours = {"Battery": TECH["Storage"], "LAES": SEQUENTIAL[6]}
    labels = {"Battery": "Grid batteries", "LAES": "Long-duration (pumped hydro and new build)"}
    for carrier in ("Battery", "LAES"):
        by = s[s.carrier == carrier].groupby("bus").p_nom.sum().sort_values(ascending=False) / 1000
        by = by[by > 0.002]
        ax.scatter(b.loc[by.index, "x"], b.loc[by.index, "y"], s=by * per_gw, color=colours[carrier], alpha=0.75,
                   linewidths=0.4, edgecolors="white", zorder=4 if carrier == "Battery" else 5)
    totals = s.groupby("carrier").p_nom.sum() / 1000
    colour_key(key, [dot(colours[c], f"{labels[c]} ({totals[c]:.1f} GW)") for c in ("Battery", "LAES")], ncol=1)
    size_key(key, size_keys([0.1, 0.5, 1.5], per_gw, "GW"), top=0.42)
    save(fig, out, dpi=170)


# ---- hydrogen ---------------------------------------------------------------------------------------
def figure_hydrogen(net30: dict, net50: dict, land, out: Path) -> None:
    fig = plt.figure(figsize=(9.2, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[7.0, 1.0], hspace=0.02, wspace=0.02)
    per_gw = 90.0
    elec_colour, turbine_colour = SEQUENTIAL[5], TECH["Gas"]
    for k, (year, net) in enumerate((("2030", net30), ("2050", net50))):
        ax = fig.add_subplot(grid[0, k])
        base_map(ax, land)
        draw_lines(ax, net, alpha=0.35)
        b, links = net["buses"], net["links"]
        el = links[links.carrier == "electrolysis"].groupby("bus0").p_nom.sum() / 1000
        tu = links[links.carrier == "H2_turbine"].groupby("bus1").p_nom.sum() / 1000
        el, tu = el[el.index.isin(b.index)], tu[tu.index.isin(b.index)]
        tu = tu.sort_values(ascending=False)
        ax.scatter(b.loc[tu.index, "x"], b.loc[tu.index, "y"], s=tu * per_gw, color=turbine_colour, alpha=0.6,
                   linewidths=0.4, edgecolors="white", zorder=4)
        ax.scatter(b.loc[el.index, "x"], b.loc[el.index, "y"], s=el * per_gw, facecolor="none",
                   edgecolors=elec_colour, linewidths=1.6, zorder=5)
        ax.text(0.5, 1.035, YEAR_LABELS[year], ha="center", va="bottom", transform=ax.transAxes, fontsize=9.5,
                fontweight="semibold", color=INK["text"])
        ax.text(0.5, 1.0, f"electrolysers {el.sum():.1f} GW, turbines {tu.sum():.1f} GW", ha="center",
                va="bottom", transform=ax.transAxes, fontsize=8.5, color=INK["muted"])
    key = fig.add_subplot(grid[1, :])
    key.axis("off")
    colour_key(key, [Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor="none",
                            markeredgecolor=elec_colour, markeredgewidth=1.6, label="Electrolysers (demand)"),
                     dot(turbine_colour, "Hydrogen turbines (generation)", size=8)])
    size_key(key, size_keys([0.5, 2, 8], per_gw, "GW"), top=0.5)
    save(fig, out, dpi=160)


# ---- interconnectors --------------------------------------------------------------------------------
CABLE_NAMES = {"IC_Britned": "BritNed", "IC_NS Link": "North Sea Link", "IC_East West Interconnector": "EWIC",
               "IC_NeuConnect Interconnector": "NeuConnect"}
LABELS_AT = {  # label position (km east, km north of the landing) and alignment, by GB landing bus
    "GRAI41": (95, 50, "left"), "RICH41": (100, 4, "left"), "SELL4A": (95, -32, "left"),
    "CHIL41": (45, -45, "center"), "BLYT21": (70, 85, "left"), "BICF41": (85, 55, "left"),
    "AUCH2-": (-72, -45, "right"), "FLIB41": (-110, 52, "right"), "PEMB41": (-90, 48, "right"),
}
STUB_KM = 95


def figure_interconnectors(net: dict, land, out: Path) -> None:
    b = net["buses"]
    ic = cross_border(net)
    ic["x"], ic["y"] = b.loc[ic.gb_bus, "x"].values, b.loc[ic.gb_bus, "y"].values
    ic["fx"], ic["fy"] = b.loc[ic.foreign_bus, "x"].values, b.loc[ic.foreign_bus, "y"].values
    ic["name"] = [CABLE_NAMES.get(i, i.replace("IC_", "")) for i in ic.index]
    # landings within 40 km share one label (EWIC at Deeside and MaresConnect at Bodelwyddan)
    ic["site"] = ic.gb_bus.replace({"BODE41": "FLIB41"})
    fig, ax, key = map_figure(key_height=1.0, width=5.6)
    base_map(ax, land)
    ax.set_xlim(-60_000, 820_000)
    draw_lines(ax, net, alpha=0.3)
    per_gw = 230.0
    colour = TECH["Imports"]
    for _, r in ic.iterrows():
        d = np.array([r.fx - r.x, r.fy - r.y])
        end = np.array([r.x, r.y]) + d / np.linalg.norm(d) * STUB_KM * 1000
        ax.plot([r.x, end[0]], [r.y, end[1]], color=colour, lw=1.4, ls=(0, (3, 2)), zorder=3)
    landing = ic.groupby("gb_bus").agg(p=("p_nom", "sum"), x=("x", "first"), y=("y", "first"))
    ax.scatter(landing.x, landing.y, s=landing.p / 1000 * per_gw, color=colour, alpha=0.75, linewidths=0.5,
               edgecolors="white", zorder=5)
    for site, sel in ic.groupby("site"):
        dx, dy, ha = LABELS_AT.get(site, (STUB_KM + 8, 0, "left"))
        x0, y0 = b.at[site, "x"], b.at[site, "y"]
        same = sel.groupby("country", sort=False).agg(names=("name", " and ".join), p=("p_nom", "sum"))
        text = "\n".join(f"{r.names}, {country} {r.p / 1000:.3g} GW" for country, r in same.iterrows())
        ax.text(x0 + dx * 1000, y0 + dy * 1000, text, ha=ha, va="center", fontsize=7, color=INK["text"],
                linespacing=1.2, zorder=6)
    colour_key(key, [dot(colour, f"GB landing of the interconnectors ({ic.p_nom.sum() / 1000:.2f} GW)"),
                     Line2D([], [], color=colour, lw=1.4, ls=(0, (3, 2)), label="Direction of the far end")],
               ncol=1)
    size_key(key, size_keys([0.5, 1.5, 3], per_gw, "GW"), top=0.45)
    save(fig, out, dpi=170)


def import_prices(path: Path, net: dict) -> pd.DataFrame:
    ic = cross_border(net)
    mc = time_series(path, "links_t_marginal_cost", ic.index)
    return mc.T.groupby(ic.loc[mc.columns, "country"]).mean().T


def figure_foreign_prices(path30: Path, net30: dict, path50: Path, net50: dict, out: Path) -> None:
    prices = {"2030": import_prices(path30, net30), "2050": import_prices(path50, net50)}
    countries = [c for c in ["Norway", "Denmark", "Germany", "Netherlands", "Belgium", "France", "Ireland",
                             "Northern Ireland"] if c in prices["2030"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(countries))
    for k, (year, p) in enumerate(prices.items()):
        yy = y + (k - 0.5) * 0.36
        for yi, c in zip(yy, countries):
            if c not in p:
                continue
            q = np.percentile(p[c], [10, 25, 50, 75, 90])
            ax.plot([q[0], q[4]], [yi, yi], color=YEAR_COLOURS[year], lw=1.2, solid_capstyle="butt")
            ax.plot([q[1], q[3]], [yi, yi], color=YEAR_COLOURS[year], lw=6, solid_capstyle="butt")
            ax.plot(p[c].mean(), yi, marker="|", markersize=11, mew=2, color=INK["text"])
    ax.set_yticks(y, countries)
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Import price, GBP/MWh")
    handles = [Line2D([], [], color=YEAR_COLOURS[yr], lw=6, label=YEAR_LABELS[yr]) for yr in prices]
    handles += [Line2D([], [], color=INK["muted"], lw=6, label="middle half of hours"),
                Line2D([], [], color=INK["muted"], lw=1.2, label="10th to 90th percentile"),
                Line2D([], [], color=INK["text"], marker="|", ls="", markersize=11, mew=2, label="annual mean")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, fontsize=8.5)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resources", type=Path, default=ROOT / "resources")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/source/_static/figures/supply")
    parser.add_argument("--only", nargs="*", default=None, help="figure names to draw (default: all)")
    args = parser.parse_args()
    apply_style()
    land = load_land()
    p30, p50 = args.resources / NET_2030, args.resources / NET_2050
    net30, net50 = load_network(p30), load_network(p50)
    for net, path in ((net30, p30), (net50, p50)):  # load_network does not read max_hours for every file
        net["stores"] = static(path, "storage_units", ["bus", "carrier", "p_nom", "max_hours"])
    figures = {
        "demand_winter_week_2030": lambda o: figure_winter_week(args.resources, o),
        "demand_map_2030": lambda o: figure_demand_map(net30, land, p30, o),
        "consumer_flexibility": lambda o: figure_flexibility(net30, net50, o),
        "capacity_2030_2050": lambda o: figure_capacity(net30, net50, o),
        "merit_order_2030": lambda o: figure_merit_order(p30, o),
        "gas_availability_2030": lambda o: figure_gas_availability(p30, o),
        "renewables_named_proxy_2030": lambda o: figure_named_proxy(p30, o),
        "renewables_capacity_factors_2030": lambda o: figure_capacity_factors(p30, o),
        "storage_2030_2050": lambda o: figure_storage(net30, net50, o),
        "storage_map_2030": lambda o: figure_storage_map(net30, land, o),
        "hydrogen_map_2030_2050": lambda o: figure_hydrogen(net30, net50, land, o),
        "interconnectors_map_2030": lambda o: figure_interconnectors(net30, land, o),
        "foreign_prices_2030_2050": lambda o: figure_foreign_prices(p30, net30, p50, net50, o),
    }
    for name, draw in figures.items():
        if args.only is None or name in args.only:
            draw(args.out / f"{name}.png")


if __name__ == "__main__":
    main()
