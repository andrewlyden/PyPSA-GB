"""Figures for the study pages of the documentation (docs/source/studies/).

Every figure reads the final run's outputs under resources/ (and, for the 2020 validation, the May 2026 research
scorecard the page quotes), the same files the pages cite in their source comments.

Clean Power 2030 (studies/cp30_2030.md):

1. cp30_constraint_cost.png: NESO-comparable redispatch cost by network case on both turn-down bases, against
   NESO's figure, and its composition as solved (accepted offers, accepted bids, interconnector re-trading).
   resources/analysis/CP30_2030_<family>_pro_rata_turn_down_summary.json and _annual_summary.json.
2. cp30_monthly_constraint_cost.png: the same cost month by month, on each case's headline basis.
   resources/analysis/CP30_2030_<family>_monthly_metrics.csv (ordinary redispatch plus re-trading) and
   resources/market/CP30_FFR_2030_<Mon>_<family>_pro_rata_turn_down.json (the pro rata uplift).
3. cp30_wind_turn_down_map.png: wind turned down in the balancing stage, as solved, by bus, for the three cases.
   resources/market/CP30_FFR_2030_<Mon>_<family>_pro_rata_turn_down.csv; network from each case's January
   balancing network.
4. cp30_stage1_price_duration.png: the Stage 1 price duration curve against CP30's CP.32 curve (FFR).
   resources/market/CP30_FFR_2030_<Mon>_stage1_wholesale_price.csv, data/CP30/cp32_price_duration_curve_2030.csv
   and the checked points in resources/analysis/CP30_FFR_2030_stage1_cp30_stage1_validation.json.

FES long-range years (studies/fes_long_range.md):

5. fes_capacity_2040_2050.png: installed capacity by technology in the built networks.
   resources/network/FES25_HT_2040.nc and FES25_HT_2050.nc.
6. fes_cost_and_curtailment.png: constraint and balancing cost by component, and CfD-contracted units turned down by carrier.
   resources/analysis/FES25_HT_<year>_annual_summary.json, _future_balancing_costs_summary.json.
7. fes_wind_turn_down_map.png: wind turned down in the balancing stage, as solved, by zone, 2040 and 2050.
   resources/market/FES25_HT_<year>_<Mon>_market_pro_rata_turn_down.csv; zones from the built networks.

Historical validation 2020 (studies/validation_2020.md):

8. validation_2020_monthly.png: model balancing cost against NESO's total thermal constraint cost, the constraint
   volumes, and the model's wholesale price against ELEXON's MID price, by month.
   resources/analysis/validation_2020_completed_months_scorecard.csv and validation_2020_volume_cost.csv.

Public ownership and household bills (studies/public_ownership_bills.md):

9. po_household_bill.png: the household bill and the household total change against private, for private, S1, S2
   and S3 in 2030 (CENTRAL network), 2040 and 2050.
10. po_three_party_split.png: each scenario's change against private split between consumers, the Exchequer and
    investors, with the resource saving they add up to.
11. po_network_bounds_2030.png: the 2030 bill on the three network cases, and S3's resource saving.
    9-11: resources/public_ownership/<group>_household_totals.csv.

Run from the repository root:

    python scripts/analysis/docs_figures/studies.py [--resources resources]

Not part of any Snakemake rule: the figures are documentation assets, committed as PNG.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.analysis.docs_figures.common import (  # noqa: E402
    FIGURES, INK, ROOT, SAGE, SEQUENTIAL, TECH, XLIM, YLIM, apply_style, carrier_group, dot, gb_dc_links, load_land,
    load_network, save, size_keys,
)

OUT = FIGURES / "studies"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CASES = {  # family: label (the 2030 network-delivery cases, lightest to most constrained)
    "cp30_network": "LOWER\nrecommended network",
    "no_acceleration": "CENTRAL\ntwo projects late",
    "no_upgrades": "UPPER\nno upgrades after 2025",
}
CASE_SHORT = {"cp30_network": "LOWER", "no_acceleration": "CENTRAL", "no_upgrades": "UPPER"}
CASE_TICK = {"cp30_network": "LOWER\nrecommended\nnetwork", "no_acceleration": "CENTRAL\ntwo projects\nlate",
             "no_upgrades": "UPPER\nno upgrades\nafter 2025"}  # short enough to sit side by side on an axis
CASE_COLOUR = {"cp30_network": SEQUENTIAL[2], "no_acceleration": SEQUENTIAL[4], "no_upgrades": SEQUENTIAL[6]}
BASIS_LABEL = {"cheapest_first": "as solved", "pro_rata_zonal": "turn-down shared pro rata"}
COMPONENT = [  # (annual summary key, label, colour: the technology that mostly makes the action)
    ("accepted_offer_cost_GBP", "Plant turned up (offers, mostly gas)", TECH["Gas"]),
    ("accepted_bid_cost_GBP", "Plant turned down (bids, mostly wind)", TECH["Offshore wind"]),
    ("interconnector_retrading_cost_GBP", "Interconnector trades re-traded", TECH["Imports"]),
]
SCENARIOS = [  # (case id in 2030, case id in 2040 and 2050, label, colour)
    ("private", "private", "Private", INK["muted"]),
    ("public_control", "public_control", "S1, public control", SEQUENTIAL[2]),
    ("public_ownership", "public_ownership", "S2, public new build", SEQUENTIAL[4]),
    ("public_ownership@fully_nationalised_market_value_central_dispatch",
     "public_ownership@fully_nationalised_market_value_central_dispatch", "S3, nationalisation", SEQUENTIAL[6]),
]
PO_YEARS = {"2030": "cp30_2030_no_acceleration", "2040": "fes_ht_2040", "2050": "fes_ht_2050"}
PO_COLUMNS = ["case_id", "bill_inc_vat", "bill_delta_vs_private_inc_vat", "household_total_delta_vs_private_gbp_inc_vat",
              "consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp", "investor_change_vs_private_gbp",
              "resource_saving_vs_private_gbp"]
EXTRA_STORAGE = {"Pumped Storage Hydroelectricity": "Storage", "Domestic Battery": "Storage"}


def thousands(ax, axis: str = "y") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def label_bar(ax, x, y, text, dy=0.0, fontsize=8.5, **kw):
    ax.annotate(text, (x, y), xytext=(0, 3 + dy), textcoords="offset points", ha="center", va="bottom",
                fontsize=fontsize, color=INK["text"], **kw)


# ---- Clean Power 2030 ------------------------------------------------------------------------------
def figure_cp30_cost(res: Path, out: Path) -> None:
    turn = {f: read_json(res / f"analysis/CP30_2030_{f}_pro_rata_turn_down_summary.json") for f in CASES}
    annual = {f: read_json(res / f"analysis/CP30_2030_{f}_annual_summary.json")["aggregate"] for f in CASES}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.25})
    x = np.arange(len(CASES))
    width = 0.26
    for i, (basis, colour) in enumerate((("cheapest_first", SEQUENTIAL[2]), ("pro_rata_zonal", SEQUENTIAL[5]))):
        values = [turn[f]["neso_comparable_redispatch_cost_by_basis_GBP"][basis] / 1e6 for f in CASES]
        ax.bar(x + (i - 0.5) * width, values, width=width * 0.92, color=colour, label=f"Model, {BASIS_LABEL[basis]}",
               zorder=3)
    neso = [turn[f]["neso_network_cost_GBP"] / 1e6 for f in CASES]
    for xi, v in zip(x, neso):
        ax.plot([xi - 1.2 * width, xi + 1.2 * width], [v, v], color=INK["text"], lw=2, zorder=5,
                solid_capstyle="butt")
        ax.annotate(f"NESO\n{v:,.0f}", (xi - 1.2 * width, v), xytext=(-3, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK["text"], linespacing=1.0)
    ax.plot([], [], color=INK["text"], lw=2, label="NESO's figure")
    ticks = []
    for f in CASES:
        head = turn[f]["headline_neso_comparable_redispatch_cost_GBP"] / 1e6
        gap = turn[f]["gap_to_neso_by_basis"][turn[f]["headline_basis"]]
        ticks.append(f"{CASE_TICK[f]}\n{head:,.0f} ({gap:+.1%})")
    ax.set_xticks(x, ticks, fontsize=8.5)
    ax.set_xlim(-0.85, 2.45)
    ax.set_ylabel("GBP m a year (2024 prices)")
    ax.set_ylim(0, max(neso) * 1.18)
    ax.set_title("Constraint cost in 2030")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)

    bottom = np.zeros(len(CASES))
    for key, label, colour in COMPONENT:
        values = np.array([annual[f][key] / 1e6 for f in CASES])
        bx.bar(x, values, bottom=bottom, width=0.55, color=colour, label=label, zorder=3, edgecolor="white",
               linewidth=1.0)
        bottom += values
    for xi, f in zip(x, CASES):
        label_bar(bx, xi, bottom[xi], f"{annual[f]['neso_comparable_redispatch_cost_GBP'] / 1e6:,.0f}")
    bx.set_xticks(x, [CASE_TICK[f] for f in CASES], fontsize=8.5)
    bx.set_ylabel("GBP m a year (2024 prices)")
    bx.set_ylim(0, max(neso) * 1.18)
    bx.set_title("What it is made of (as solved)")
    thousands(ax)
    thousands(bx)
    bx.legend(loc="upper left", fontsize=8.5)
    bx.grid(axis="x", visible=False)
    save(fig, out)


def monthly_cost(res: Path, family: str, basis: str) -> pd.Series:
    """NESO-comparable redispatch cost by month (GBP m): ordinary redispatch plus re-trading, plus the pro rata
    uplift on the pro rata basis."""
    mm = pd.read_csv(res / f"analysis/CP30_2030_{family}_monthly_metrics.csv").set_index("scenario")
    cost = mm.ordinary_redispatch_cost_GBP + mm.interconnector_retrading_cost_GBP
    if basis == "pro_rata_zonal":
        uplift = {f"CP30_FFR_2030_{m}_{family}": read_json(
            res / f"market/CP30_FFR_2030_{m}_{family}_pro_rata_turn_down.json")["pro_rata_uplift_GBP"] for m in MONTHS}
        cost = cost + pd.Series(uplift)
    return (cost / 1e6).reindex([f"CP30_FFR_2030_{m}_{family}" for m in MONTHS])


def figure_cp30_monthly(res: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(12)
    for family in CASES:
        summary = read_json(res / f"analysis/CP30_2030_{family}_pro_rata_turn_down_summary.json")
        basis = summary["headline_basis"]
        cost = monthly_cost(res, family, basis)
        total = summary["headline_neso_comparable_redispatch_cost_GBP"] / 1e6
        assert abs(cost.sum() - total) < 1.0, (family, cost.sum(), total)
        ax.plot(x, cost.values, color=CASE_COLOUR[family], marker="o", markersize=4.5,
                label=f"{CASES[family].replace(chr(10), ': ')}, {BASIS_LABEL[basis]} (GBP {total:,.0f} m a year)")
    ax.set_xticks(x, MONTHS)
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylim(0, None)
    ax.set_ylabel("GBP m a month (2024 prices)")
    thousands(ax)
    ax.legend(loc="upper center", fontsize=8.5, ncol=1)
    save(fig, out)


def turn_down_by_bus(res: Path, stems: list[str]) -> pd.DataFrame:
    """Annual as-solved wind turn-down (MWh) by bus and carrier, summed over monthly *_pro_rata_turn_down.csv."""
    parts = [pd.read_csv(res / "market" / f"{s}_pro_rata_turn_down.csv", usecols=["bus", "carrier", "our_MWh"])
             for s in stems]
    return pd.concat(parts).groupby(["bus", "carrier"]).our_MWh.sum().unstack(fill_value=0.0)


def small_multiple_maps(n: int, width: float, key_height: float = 1.0):
    map_h = (width / n) * 7.4 / 5.0
    fig = plt.figure(figsize=(width, map_h + key_height))
    grid = fig.add_gridspec(2, n, height_ratios=[map_h, key_height], hspace=0.02, wspace=0.02)
    axes = [fig.add_subplot(grid[0, i]) for i in range(n)]
    key = fig.add_subplot(grid[1, :])
    key.axis("off")
    return fig, axes, key


def map_panel(ax, land, title: str) -> None:
    land.plot(ax=ax, color=SAGE["land"], edgecolor=SAGE["coast"], linewidth=0.4)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, 1.01, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="semibold",
            color=INK["text"])


def thin_lines(ax, net: dict, min_kv: float = 132.0, alpha: float = 0.5) -> None:
    """Transmission lines above ``min_kv`` (fewer marks than draw_lines, for small maps)."""
    b, lines = net["buses"], net["lines"]
    b = b[b.country == "GB"] if "country" in b else b
    kv = lines["voltage_kv"] if "voltage_kv" in lines else net["buses"].loc[lines.bus0, "v_nom"].values
    for low, high, colour, width in ((min_kv, 275, SAGE["mid"], 0.45), (275, 1000, SAGE["deep"], 0.7)):
        sel = lines[(kv > low) & (kv <= high) & lines.bus0.isin(b.index) & lines.bus1.isin(b.index)]
        xs = np.column_stack([b.loc[sel.bus0, "x"].values, b.loc[sel.bus1, "x"].values, np.full(len(sel), np.nan)])
        ys = np.column_stack([b.loc[sel.bus0, "y"].values, b.loc[sel.bus1, "y"].values, np.full(len(sel), np.nan)])
        ax.plot(xs.ravel(), ys.ravel(), color=colour, lw=width, alpha=alpha, zorder=2)


def wind_dots(ax, down: pd.DataFrame, buses: pd.DataFrame, per_twh: float, threshold_mwh: float) -> None:
    for carrier, group in (("wind_onshore", "Onshore wind"), ("wind_offshore", "Offshore wind")):
        if carrier not in down:
            continue
        s = down[carrier]
        s = s[(s > threshold_mwh) & s.index.isin(buses.index)].sort_values(ascending=False)
        ax.scatter(buses.loc[s.index, "x"], buses.loc[s.index, "y"], s=s / 1e6 * per_twh, color=TECH[group],
                   alpha=0.72, linewidths=0.4, edgecolors="white", zorder=4, clip_on=False)


def figure_cp30_turn_down_map(res: Path, land, out: Path) -> None:
    fig, axes, key = small_multiple_maps(3, 9.6, key_height=0.6)
    per_twh = 60.0
    for ax, family in zip(axes, CASES):
        summary = read_json(res / f"analysis/CP30_2030_{family}_pro_rata_turn_down_summary.json")
        down = turn_down_by_bus(res, [f"CP30_FFR_2030_{m}_{family}" for m in MONTHS])
        total = down.values.sum()
        assert abs(total - summary["turn_down_MWh"]) < 1.0, (family, total, summary["turn_down_MWh"])
        net = load_network(res / f"market/CP30_FFR_2030_Jan_{family}_balancing.nc")
        map_panel(ax, land, f"{CASE_SHORT[family]}: {total / 1e6:.1f} TWh")
        thin_lines(ax, net, alpha=0.45)
        wind_dots(ax, down, net["buses"], per_twh, threshold_mwh=5_000)
    leg = key.legend(handles=[dot(TECH["Offshore wind"], "Offshore wind turned down"),
                              dot(TECH["Onshore wind"], "Onshore wind turned down")],
                     loc="upper center", bbox_to_anchor=(0.3, 0.95), ncol=1, frameon=False, fontsize=9)
    key.add_artist(leg)
    key.legend(handles=size_keys([0.5, 2, 5], per_twh, "TWh"), loc="upper center", bbox_to_anchor=(0.72, 1.0),
               ncol=3, frameon=False, fontsize=9, handletextpad=1.0, columnspacing=2.0, borderpad=0.4)
    save(fig, out, dpi=150)


def figure_cp30_price_duration(res: Path, out: Path) -> None:
    prices = pd.concat([pd.read_csv(res / f"market/CP30_FFR_2030_{m}_stage1_wholesale_price.csv").wholesale_price
                        for m in MONTHS], ignore_index=True)
    check = read_json(res / "analysis/CP30_FFR_2030_stage1_cp30_stage1_validation.json")
    rows = [r for r in check["rows"] if r["component"] == "price" and r["item"].startswith("price exceeded")]
    curve = pd.read_csv(ROOT / "data/CP30/cp32_price_duration_curve_2030.csv")
    assert len(prices) == 8760 and abs(prices.mean() - next(
        r for r in check["rows"] if r["component"] == "price" and r["item"] == "mean")["model"]) < 1e-6
    fig, ax = plt.subplots(figsize=(9, 4.4))
    share = (np.arange(len(prices)) + 0.5) / len(prices)
    ax.plot(share * 100, np.sort(prices.values)[::-1], color=SEQUENTIAL[5], lw=2,
            label=f"PyPSA-GB Stage 1 (mean GBP {prices.mean():.2f}/MWh)")
    ax.plot(curve.share_of_hours * 100, curve.further_flex_and_renewables, color=TECH["Gas"], lw=2,
            label="CP30 CP.32, FFR pathway (mean GBP "
                  f"{next(r for r in check['rows'] if r['component'] == 'price' and r['item'] == 'mean')['cp30']:.2f}"
                  "/MWh)")
    for r in rows:
        s = float(r["item"].split(" ")[3].rstrip("%"))
        model_at = float(np.quantile(prices.values, 1 - s / 100))
        assert abs(model_at - r["model"]) < 1e-6
        ax.scatter([s, s], [r["cp30"], r["model"]], s=22, color="white", edgecolors=INK["text"], linewidths=1.0,
                   zorder=5)
    ax.scatter([], [], s=22, color="white", edgecolors=INK["text"], linewidths=1.0, label="Points checked (1-90%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the year's hours with a price at least this high (%)")
    ax.set_ylabel("Wholesale price, GBP/MWh")
    ax.axhline(0, color=INK["axis"], lw=0.8, zorder=1)
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, out)


# ---- FES long-range years --------------------------------------------------------------------------
def fes_capacity(res: Path, year: int) -> pd.Series:
    """GW by display category from the built network."""
    net = load_network(res / f"network/FES25_HT_{year}.nc")
    gens = net["gens"].assign(group=net["gens"].carrier.map(carrier_group))
    stores = net["stores"].assign(group=net["stores"].carrier.map(lambda c: carrier_group(c) or EXTRA_STORAGE.get(c)))
    gw = {}
    for group in ("Offshore wind", "Onshore wind", "Solar", "Nuclear", "Other renewables", "Gas"):
        gw[group] = gens.loc[gens.group == group, "p_nom"].sum() / 1e3
    links, b = net["links"], net["buses"]
    gw["Hydrogen turbines"] = links.loc[links.carrier == "H2_turbine", "p_nom"].sum() / 1e3
    gw["Storage"] = stores.loc[stores.group == "Storage", "p_nom"].sum() / 1e3
    gb = set(b.index[b.country == "GB"]) if "country" in b else set(b.index[~b.index.str.startswith("External")])
    dc = links[links.carrier == "DC"]
    cross = dc[dc.bus0.isin(gb) != dc.bus1.isin(gb)]
    gw["Interconnectors"] = cross.p_nom.sum() / 1e3
    return pd.Series(gw)


def figure_fes_capacity(res: Path, out: Path) -> None:
    cap = pd.DataFrame({y: fes_capacity(res, y) for y in (2040, 2050)})
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    x = np.arange(len(cap))
    width = 0.3
    for i, (year, colour) in enumerate(((2040, SEQUENTIAL[2]), (2050, SEQUENTIAL[5]))):
        ax.bar(x + (i - 0.5) * width, cap[year], width=width * 0.92, color=colour, label=str(year), zorder=3)
        for xi, v in zip(x, cap[year]):
            label_bar(ax, xi + (i - 0.5) * width, v, f"{v:.0f}", fontsize=7.5)
    labels = [c.replace("Other renewables", "Other\nrenewables").replace("Hydrogen turbines", "Hydrogen\nturbines")
              .replace("Interconnectors", "Inter-\nconnectors").replace("Offshore wind", "Offshore\nwind")
              .replace("Onshore wind", "Onshore\nwind").replace("Gas", "Gas\n(incl. CCS)") for c in cap.index]
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("Installed capacity, GW")
    ax.set_ylim(0, cap.values.max() * 1.1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, out)
    print(cap.round(1).to_string())


def figure_fes_cost_curtailment(res: Path, out: Path) -> None:
    years = (2040, 2050)
    annual = {y: read_json(res / f"analysis/FES25_HT_{y}_annual_summary.json")["aggregate"] for y in years}
    curt = {y: read_json(res / f"analysis/FES25_HT_{y}_future_balancing_costs_summary.json")["annual"]
            ["network_curtailment_twh_by_carrier"] for y in years}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.3})
    x = np.arange(len(years))
    bottom = np.zeros(len(years))
    for key, label, colour in COMPONENT:
        values = np.array([annual[y][key] / 1e6 for y in years])
        ax.bar(x, values, bottom=bottom, width=0.5, color=colour, label=label, zorder=3, edgecolor="white",
               linewidth=1.0)
        bottom += values
    for xi, y in zip(x, years):
        label_bar(ax, xi, bottom[xi], f"{annual[y]['neso_comparable_redispatch_cost_GBP'] / 1e6:,.0f}",
                  fontweight="semibold")
    ax.set_xticks(x, [str(y) for y in years])
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, bottom.max() * 1.45)
    ax.set_ylabel("GBP m a year (2024 prices)")
    ax.set_title("Constraint and balancing cost, as solved")
    thousands(ax)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)

    carriers = [("wind_offshore", "Offshore wind"), ("wind_onshore", "Onshore wind"), ("solar_pv", "Solar")]
    width = 0.24
    for i, (carrier, group) in enumerate(carriers):
        values = [curt[y][carrier] for y in years]
        pos = x + (i - 1) * width
        bx.bar(pos, values, width=width * 0.9, color=TECH[group], label=group, zorder=3)
        for p, v in zip(pos, values):
            label_bar(bx, p, v, f"{v:.1f}")
    bx.set_xticks(x, [str(y) for y in years])
    bx.set_xlim(-0.7, 1.7)
    bx.set_ylabel("TWh a year")
    bx.set_ylim(0, max(max(c.values()) for c in curt.values()) * 1.3)
    bx.set_title("CfD-contracted units turned down")
    bx.legend(loc="upper left", fontsize=8.5)
    bx.grid(axis="x", visible=False)
    save(fig, out)


def figure_fes_turn_down_map(res: Path, land, out: Path) -> None:
    fig, axes, key = small_multiple_maps(2, 7.4, key_height=1.1)
    per_twh = 110.0
    for ax, year in zip(axes, (2040, 2050)):
        summary = read_json(res / f"analysis/FES25_HT_{year}_pro_rata_turn_down_summary.json")
        down = turn_down_by_bus(res, [f"FES25_HT_{year}_{m}_market" for m in MONTHS])
        total = down.values.sum()
        assert abs(total - summary["turn_down_MWh"]) < 1.0, (year, total, summary["turn_down_MWh"])
        net = load_network(res / f"network/FES25_HT_{year}.nc")
        b, lines = net["buses"], net["lines"]
        map_panel(ax, land, f"{year}: {total / 1e6:.1f} TWh")
        for bus0, bus1, s_nom in zip(lines.bus0, lines.bus1, lines.s_nom):
            ax.plot([b.at[bus0, "x"], b.at[bus1, "x"]], [b.at[bus0, "y"], b.at[bus1, "y"]], color=SAGE["deep"],
                    lw=0.4 + s_nom / 12_000, alpha=0.55, solid_capstyle="round", zorder=2)
        for bus0, bus1 in zip(*[gb_dc_links(net)[c] for c in ("bus0", "bus1")]):
            ax.plot([b.at[bus0, "x"], b.at[bus1, "x"]], [b.at[bus0, "y"], b.at[bus1, "y"]], color=SAGE["hvdc"],
                    lw=1.1, ls=(0, (3, 2)), zorder=2)
        zones = b[b.index.str.match(r"^Z\d+$")]
        ax.scatter(zones.x, zones.y, s=6, color=SAGE["deep"], zorder=3)
        wind_dots(ax, down, b, per_twh, threshold_mwh=1_000)
    leg = key.legend(handles=[dot(TECH["Offshore wind"], "Offshore wind turned down"),
                              dot(TECH["Onshore wind"], "Onshore wind turned down"),
                              Line2D([], [], color=SAGE["deep"], lw=1.6, label="AC between zones (width: capacity)"),
                              Line2D([], [], color=SAGE["hvdc"], lw=1.1, ls=(0, (3, 2)), label="HVDC links")],
                     loc="upper center", bbox_to_anchor=(0.27, 1.0), ncol=1, frameon=False, fontsize=8.5)
    key.add_artist(leg)
    key.legend(handles=size_keys([1, 3, 6], per_twh, "TWh"), loc="upper center", bbox_to_anchor=(0.76, 1.0),
               ncol=3, frameon=False, fontsize=8.5, handletextpad=1.5, columnspacing=1.8, borderpad=0.6)
    save(fig, out, dpi=150)


# ---- Validation 2020 -------------------------------------------------------------------------------
def figure_validation_2020(res: Path, out: Path) -> None:
    sc = pd.read_csv(res / "analysis/validation_2020_completed_months_scorecard.csv")
    vc = pd.read_csv(res / "analysis/validation_2020_volume_cost.csv")
    vc = vc[vc.scenario.str.startswith("Validation_")]
    vc = vc.set_index(vc.scenario.str[11:14])
    months = sc.period.str[:3]
    vc = vc.reindex(months)
    fig, (ax, vx, bx) = plt.subplots(3, 1, figsize=(9, 8.6), sharex=True, gridspec_kw={"hspace": 0.22})
    x = np.arange(len(sc))
    width = 0.32
    ax.bar(x - width / 2, sc.bm_cost_gbp / 1e6, width=width * 0.92, color=SEQUENTIAL[4], zorder=3,
           label=f"Model balancing cost (year GBP {sc.bm_cost_gbp.sum() / 1e6:,.0f} m)")
    ax.bar(x + width / 2, sc.neso_breakdown_thermal_cost_gbp / 1e6, width=width * 0.92, color=INK["muted"], zorder=3,
           label=f"NESO thermal constraint cost (year GBP {sc.neso_breakdown_thermal_cost_gbp.sum() / 1e6:,.0f} m)")
    ax.scatter(x + width / 2, sc.neso_thermal_cost_gbp / 1e6, marker="_", s=140, color=INK["text"], zorder=4,
               label=f"of which the six named boundaries (year GBP {sc.neso_thermal_cost_gbp.sum() / 1e6:,.0f} m)")
    ax.set_ylabel("GBP m a month")
    ax.set_title("Constraint cost")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(lab) for lab in labels if not lab.startswith("of which")] +         [labels.index(lab) for lab in labels if lab.startswith("of which")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    vx.bar(x - width / 2, vc.model_volume_gwh.to_numpy(), width=width * 0.92, color=SEQUENTIAL[4], zorder=3,
           label=f"Model turn-down (year {vc.model_volume_gwh.sum() / 1e3:,.1f} TWh)")
    vx.bar(x + width / 2, vc.neso_volume_gwh.to_numpy(), width=width * 0.92, color=INK["muted"], zorder=3,
           label=f"NESO thermal constraint volume (year {vc.neso_volume_gwh.sum() / 1e3:,.1f} TWh)")
    vx.set_ylabel("GWh a month")
    vx.set_title("Constraint volume")
    vx.legend(loc="upper center", fontsize=8.5)
    vx.grid(axis="x", visible=False)
    bx.plot(x, sc.mean_smp_gbp_mwh, color=SEQUENTIAL[4], marker="o", markersize=4.5, label="Model wholesale price")
    bx.plot(x, sc.mean_mid_gbp_mwh, color=INK["muted"], marker="s", markersize=4.5, label="ELEXON MID price")
    bx.set_ylabel("GBP/MWh, monthly mean")
    bx.set_ylim(0, None)
    bx.set_title("Wholesale price")
    bx.legend(loc="lower right", fontsize=8.5)
    bx.set_xticks(x, months)   # the months the scorecard has (a failed month is absent, not mislabelled)
    save(fig, out)


# ---- Public ownership and household bills ----------------------------------------------------------
def po_rows(res: Path, group: str) -> pd.DataFrame:
    d = pd.read_csv(res / f"public_ownership/{group}_household_totals.csv", usecols=PO_COLUMNS, low_memory=False)
    return d.drop_duplicates("case_id").set_index("case_id")


def figure_po_bill(res: Path, out: Path) -> None:
    rows = {y: po_rows(res, g) for y, g in PO_YEARS.items()}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(PO_YEARS))
    width = 0.19
    for i, (case30, case_late, label, colour) in enumerate(SCENARIOS):
        pos = x + (i - 1.5) * width
        bill = [rows[y].at[case30 if y == "2030" else case_late, "bill_inc_vat"] for y in PO_YEARS]
        ax.bar(pos, bill, width=width * 0.9, color=colour, label=label, zorder=3)
        if i > 0:
            total = [rows[y].at[case30 if y == "2030" else case_late, "household_total_delta_vs_private_gbp_inc_vat"]
                     for y in PO_YEARS]
            bx.bar(pos + width / 2, total, width=width * 0.9, color=colour, label=label, zorder=3)
            for p, v in zip(pos + width / 2, total):
                small = abs(v) < 10  # the 2030 S1 and S2 labels would touch: stagger them
                bx.annotate(f"{v:,.2f}" if small else f"{v:,.0f}", (p, v), xytext=(0, -3 - 11 * (small and i == 2)),
                            textcoords="offset points", ha="center", va="top", fontsize=7.5, color=INK["text"])
    for xi, y in zip(x, PO_YEARS):
        label_bar(ax, xi - 1.5 * width, rows[y].at["private", "bill_inc_vat"],
                  f"{rows[y].at['private', 'bill_inc_vat']:,.0f}", fontsize=7.5)
        s3 = SCENARIOS[3][0] if y == "2030" else SCENARIOS[3][1]
        label_bar(ax, xi + 1.5 * width, rows[y].at[s3, "bill_inc_vat"], f"{rows[y].at[s3, 'bill_inc_vat']:,.0f}",
                  fontsize=7.5)
    ax.set_xticks(x, list(PO_YEARS))
    ax.set_ylabel("GBP a year per household, incl. VAT")
    ax.set_ylim(0, 1480)
    ax.set_title("Household electricity bill")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    bx.axhline(0, color=INK["axis"], lw=0.8)
    bx.set_xticks(x, list(PO_YEARS))
    bx.set_ylabel("GBP a year per household, incl. VAT")
    bx.set_ylim(-150, 10)
    bx.set_title("Household total change against private")
    bx.grid(axis="x", visible=False)
    save(fig, out)


def figure_po_three_party(res: Path, out: Path) -> None:
    rows = {y: po_rows(res, g) for y, g in PO_YEARS.items()}
    parties = [("consumer_change_vs_private_gbp", "Consumers", TECH["Offshore wind"]),
               ("exchequer_net_vs_private_gbp", "Exchequer", TECH["Gas"]),
               ("investor_change_vs_private_gbp", "Investors", TECH["Onshore wind"])]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08})
    scen = SCENARIOS[1:]
    y_pos = np.arange(len(scen))[::-1]
    for ax, (year, data) in zip(axes, rows.items()):
        for yi, (case30, case_late, label, _) in zip(y_pos, scen):
            r = data.loc[case30 if year == "2030" else case_late]
            left_pos, left_neg = 0.0, 0.0
            for key, _, colour in parties:
                v = r[key] / 1e9
                start = left_pos if v >= 0 else left_neg + v
                ax.barh(yi, abs(v), left=start, height=0.5, color=colour, edgecolor="white", linewidth=1.0, zorder=3)
                if v >= 0:
                    left_pos += v
                else:
                    left_neg += v
            saving = r["resource_saving_vs_private_gbp"] / 1e9
            ax.scatter([saving], [yi], marker="D", s=40, color=INK["text"], edgecolors="white", linewidths=1.0,
                       zorder=5)
            ax.annotate(f"{saving:.2f}", (max(left_pos, saving), yi), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=INK["text"])
        ax.axvline(0, color=INK["muted"], lw=0.8, zorder=4)
        ax.set_title(year, loc="center")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("GBP bn a year")
    axes[0].set_yticks(y_pos, [s[2] for s in scen])
    axes[0].set_ylim(-0.6, len(scen) - 0.4)
    axes[0].set_xlim(-12, 31)
    handles = [Patch(color=c, label=f"{lab}: gain (+) or cost (-)") for _, lab, c in parties]
    handles.append(Line2D([], [], marker="D", ls="", markersize=6.5, markerfacecolor=INK["text"],
                          markeredgecolor="white", label="Resource saving (the sum)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=8.5)
    save(fig, out)


def figure_po_network_bounds(res: Path, out: Path) -> None:
    groups = {f: po_rows(res, f"cp30_2030_{f}") for f in CASES}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.3})
    x = np.arange(len(CASES))
    width = 0.19
    for i, (case, _, label, colour) in enumerate(SCENARIOS):
        pos = x + (i - 1.5) * width
        bill = [groups[f].at[case, "bill_inc_vat"] for f in CASES]
        ax.bar(pos, bill, width=width * 0.9, color=colour, label=label, zorder=3)
        if i in (0, 3):
            for p, v in zip(pos, bill):
                label_bar(ax, p, v, f"{v:,.0f}", fontsize=7.5)
    ax.set_xticks(x, [CASE_SHORT[f] for f in CASES])
    ax.set_ylabel("GBP a year per household, incl. VAT")
    ax.set_ylim(0, 1480)
    ax.set_title("Household bill, 2030")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    s3 = SCENARIOS[3][0]
    saving = [groups[f].at[s3, "resource_saving_vs_private_gbp"] / 1e9 for f in CASES]
    bx.bar(x, saving, width=0.5, color=SEQUENTIAL[6], zorder=3)
    for xi, v in zip(x, saving):
        label_bar(bx, xi, v, f"{v:.2f}")
    bx.set_xticks(x, [CASE_SHORT[f] for f in CASES])
    bx.set_ylabel("GBP bn a year")
    bx.set_ylim(0, max(saving) * 1.2)
    bx.set_title("S3 resource saving, 2030")
    bx.grid(axis="x", visible=False)
    save(fig, out)


BILL_CASES = [  # (bill case, x label): the private household bill by year and 2030 network case
    ("CP30_2030_cp30_network", "2030\nLOWER"), ("CP30_2030_no_acceleration", "2030\nCENTRAL"),
    ("CP30_2030_no_upgrades", "2030\nUPPER"), ("FES25_HT_2040", "2040"), ("FES25_HT_2050", "2050"),
]
BILL_GROUPS = [  # Ofgem price-cap groups, in the validated categorical order
    ("Wholesale", "#2a78d6"), ("Network costs", "#eb6834"), ("Policy costs", "#1baf7a"),
    ("Operating costs, EBIT and headroom", "#eda100"),
]


def figure_household_bill(res: Path, out: Path) -> None:
    """The household bill by Ofgem price-cap group, 2030 on the three network cases, 2040 and 2050 (inc. VAT)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(BILL_CASES))
    bottom = np.zeros(len(BILL_CASES))
    totals = []
    for case, _ in BILL_CASES:
        totals.append(read_json(res / f"analysis/{case}_consumer_bill_future_summary.json"))
    parts = {case: pd.read_csv(res / f"analysis/{case}_consumer_bill_future_components.csv")
             .groupby("ofgem_group")["annual_gbp_per_3100kwh_ex_vat"].sum() for case, _ in BILL_CASES}
    for group, colour in BILL_GROUPS:
        values = np.array([parts[case].get(group, 0.0) for case, _ in BILL_CASES])
        ax.bar(x, values, bottom=bottom, width=0.62, color=colour, label=group, zorder=3, edgecolor="white", lw=0.6)
        bottom += values
    vat = np.array([t["vat_gbp"] for t in totals])
    ax.bar(x, vat, bottom=bottom, width=0.62, color="#9aa5a0", label="VAT (5%)", zorder=3, edgecolor="white", lw=0.6)
    for xi, total in zip(x, bottom + vat):
        label_bar(ax, xi, total, f"{total:,.0f}", fontsize=9)
    ax.set_xticks(x, [label for _, label in BILL_CASES])
    ax.set_ylabel("GBP a year, 3,100 kWh household")
    ax.set_ylim(0, 1260)
    thousands(ax)
    ax.set_title("Household electricity bill, private ownership")
    ax.legend(loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.17), frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, out)


FIGURES_BY_NAME = {
    "household_bill": lambda r, land: figure_household_bill(r, OUT / "household_bill.png"),
    "cp30_constraint_cost": lambda r, land: figure_cp30_cost(r, OUT / "cp30_constraint_cost.png"),
    "cp30_monthly_constraint_cost": lambda r, land: figure_cp30_monthly(r, OUT / "cp30_monthly_constraint_cost.png"),
    "cp30_wind_turn_down_map": lambda r, land: figure_cp30_turn_down_map(r, land, OUT / "cp30_wind_turn_down_map.png"),
    "cp30_stage1_price_duration": lambda r, land: figure_cp30_price_duration(
        r, OUT / "cp30_stage1_price_duration.png"),
    "fes_capacity_2040_2050": lambda r, land: figure_fes_capacity(r, OUT / "fes_capacity_2040_2050.png"),
    "fes_cost_and_curtailment": lambda r, land: figure_fes_cost_curtailment(r, OUT / "fes_cost_and_curtailment.png"),
    "fes_wind_turn_down_map": lambda r, land: figure_fes_turn_down_map(r, land, OUT / "fes_wind_turn_down_map.png"),
    "validation_2020_monthly": lambda r, land: figure_validation_2020(r, OUT / "validation_2020_monthly.png"),
    "po_household_bill": lambda r, land: figure_po_bill(r, OUT / "po_household_bill.png"),
    "po_three_party_split": lambda r, land: figure_po_three_party(r, OUT / "po_three_party_split.png"),
    "po_network_bounds_2030": lambda r, land: figure_po_network_bounds(r, OUT / "po_network_bounds_2030.png"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resources", type=Path, default=ROOT / "resources")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES_BY_NAME), help="draw only these figures")
    args = parser.parse_args()
    apply_style()
    names = args.only or list(FIGURES_BY_NAME)
    land = load_land() if any(n.endswith("_map") for n in names) else None
    for name in names:
        FIGURES_BY_NAME[name](args.resources, land)


if __name__ == "__main__":
    main()
