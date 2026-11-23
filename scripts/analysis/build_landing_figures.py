"""Draw the three maps on the documentation's landing page (docs/source/_static/landing/).

One map per modelled year, each showing a different side of the model:

- constraints_2030.png: where the network binds in 2030. The balancing mechanism's moves on the Clean Power 2030
  build, central network case (the twelve final monthly no_acceleration solves): wind turned down (pro rata by
  zone, the headline basis) against generation turned up (as solved; imports at each cable's GB landing).
- generation_2040.png: what gets built by 2040. Installed generation by technology in each of NESO's 17 zones,
  FES 2025 Holistic Transition (resources/network/FES25_HT_2040.nc, solved on the 17-zone clustered network).
- flexibility_2050.png: flexibility at net zero. Grid storage, electrolysers and hydrogen turbines in each zone,
  FES 2025 Holistic Transition (resources/network/FES25_HT_2050.nc).

Background lines are each year's full ETYS network. Reads resources/, so run it after those exist:

    python scripts/analysis/build_landing_figures.py [--resources resources] [--out docs/source/_static/landing]

Not part of any Snakemake rule: the figures are documentation assets, committed as PNG. Style, colours and
map helpers come from scripts/analysis/docs_figures/common.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Wedge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.docs_figures.common import (  # noqa: E402
    INK, ROOT, TECH, apply_style, base_map, carrier_group, colour_key, dot, draw_lines, load_land, load_network,
    map_figure, save, size_key, size_keys,
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GENERATION = ["Offshore wind", "Gas", "Onshore wind", "Solar", "Other renewables", "Nuclear"]
GRID_STORAGE = {"Battery", "Domestic Battery", "LAES", "CAES", "Pumped Storage Hydroelectricity", "pumped_hydro"}
METRES_PER_SQRT_GW = 9_500.0        # 2040 pie radius
METRES_PER_GW = 8_500.0             # 2050 bar height
KEY_HEIGHT = 2.5
#: Legend text on the landing maps, which the page shows at about a third of their width (18 Nov 2026: 9 pt was
#: unreadable there).
LEGEND_FS = 12.5


def year_label(ax, year: str, aspect: str, source: str) -> None:
    ax.text(0.03, 0.97, year, transform=ax.transAxes, fontsize=26, fontweight="bold", color=INK["text"], va="top")
    ax.text(0.03, 0.905, aspect, transform=ax.transAxes, fontsize=14, fontweight="semibold", color=INK["text"],
            va="top")
    ax.text(0.03, 0.865, source, transform=ax.transAxes, fontsize=11, color=INK["muted"], va="top")


def zone_nodes(net: dict) -> pd.DataFrame:
    b = net["buses"]
    return b[b.index.str.fullmatch(r"Z\d+")][["x", "y"]]


def to_zone(bus: pd.Series, net: dict, nodes: pd.DataFrame) -> pd.Series:
    """Map buses of the clustered network onto its zone nodes (nearest node for offshore or hub buses)."""
    b = net["buses"]

    def nearest(name):
        if name in nodes.index:
            return name
        if name not in b.index or b.at[name, "country"] != "GB":
            return None
        d = (nodes.x - b.at[name, "x"]) ** 2 + (nodes.y - b.at[name, "y"]) ** 2
        return d.idxmin()

    return bus.map(nearest)


# ---- 2030: where the network binds -------------------------------------------------------------------
def balancing_volumes(net: dict, resources: Path) -> tuple[pd.Series, pd.Series]:
    """Annual MWh by GB bus: wind turned down (pro rata, headline basis) and generation turned up."""
    down, up = [], []
    for month in MONTHS:
        stem = resources / "market" / f"CP30_FFR_2030_{month}_no_acceleration"
        down.append(pd.read_csv(f"{stem}_pro_rata_turn_down.csv").groupby("bus").pro_rata_MWh.sum())
        rs = pd.read_csv(f"{stem}_redispatch_summary.csv")
        up.append(rs[~rs.carrier.isin(["wind_onshore", "wind_offshore", "load_shedding"])]
                  .set_index("component").increase_MWh)
    down = pd.concat(down).groupby(level=0).sum()
    up = pd.concat(up).groupby(level=0).sum()
    up_bus = up.rename(index=pd.concat([net["gens"].bus, net["stores"].bus])).groupby(level=0).sum()
    b, links = net["buses"], net["links"]
    gb = set(b.index[b.country == "GB"])
    dc = links[links.carrier == "DC"]
    landing = {}  # imports sit on foreign buses: place them at the GB end of their cable
    for bus0, bus1 in zip(dc.bus0, dc.bus1):
        if bus0 in gb and bus1 not in gb:
            landing.setdefault(bus1, bus0)
        elif bus1 in gb and bus0 not in gb:
            landing.setdefault(bus0, bus1)
    up_bus = up_bus.rename(index=lambda bus: landing.get(bus, bus)).groupby(level=0).sum()
    return down[down.index.isin(gb)], up_bus[up_bus.index.isin(gb)]


def figure_2030(resources: Path, land, out: Path) -> None:
    net = load_network(resources / "network/CP30_FFR_2030.nc")
    down, up = balancing_volumes(net, resources)
    b = net["buses"]
    fig, ax, key = map_figure(KEY_HEIGHT)
    base_map(ax, land)
    ax.set_xlim(-20_000, 715_000)
    draw_lines(ax, net, alpha=0.4)
    per_twh = 170.0  # points^2 per TWh
    turned_down, turned_up = TECH["Offshore wind"], TECH["Gas"]
    for series, colour in ((up, turned_up), (down, turned_down)):
        s = series[series > 1_000].sort_values(ascending=False)
        ax.scatter(b.loc[s.index, "x"], b.loc[s.index, "y"], s=s / 1e6 * per_twh, color=colour, alpha=0.7,
                   linewidths=0.4, edgecolors="white", zorder=4)
    year_label(ax, "2030", "Where the network binds", "Clean Power 2030, central network case")
    colour_key(key, [dot(turned_down, f"Wind turned down: {down.sum() / 1e6:.1f} TWh"),
                     dot(turned_up, f"Turned up, imports included: {up.sum() / 1e6:.1f} TWh")], ncol=1,
               fontsize=LEGEND_FS)
    size_key(key, size_keys([0.5, 2, 5], per_twh, "TWh"), top=0.42, fontsize=LEGEND_FS)
    save(fig, out, dpi=170)


# ---- 2040: what gets built ---------------------------------------------------------------------------
def draw_pie(ax, x: float, y: float, mix: pd.Series) -> None:
    total = mix.sum()
    if total <= 0.05:
        return
    radius, start = METRES_PER_SQRT_GW * np.sqrt(total), 90.0
    for group in GENERATION:
        share = mix.get(group, 0.0) / total
        if share > 0:
            ax.add_patch(Wedge((x, y), radius, start - 360.0 * share, start, facecolor=TECH[group],
                               edgecolor="white", linewidth=0.6, zorder=5))
            start -= 360.0 * share


def pie_size_key(ax, fig, values_gw):
    fig.canvas.draw()
    pt_per_m = ax.get_window_extent().width * 72.0 / fig.dpi / (ax.get_xlim()[1] - ax.get_xlim()[0])
    return [Line2D([], [], marker="o", ls="", markersize=2 * METRES_PER_SQRT_GW * np.sqrt(v) * pt_per_m,
                   markerfacecolor="none", markeredgecolor=INK["muted"], label=f"{v:g} GW") for v in values_gw]


def figure_2040(resources: Path, land, out: Path) -> None:
    net = load_network(resources / "network/FES25_HT_2040.nc")
    nodes = zone_nodes(net)
    gens = net["gens"].assign(group=net["gens"].carrier.map(carrier_group))
    gens = gens[gens.group.isin(GENERATION)].assign(zone=lambda d: to_zone(d.bus, net, nodes)).dropna(subset=["zone"])
    mix = gens.pivot_table(index="zone", columns="group", values="p_nom", aggfunc="sum", fill_value=0) / 1000
    fig, ax, key = map_figure(KEY_HEIGHT)
    base_map(ax, land)
    ax.set_xlim(-20_000, 715_000)
    draw_lines(ax, load_network(resources / "network/FES25_HT_2040_network.nc"), alpha=0.35)
    for zone, row in mix.iterrows():
        draw_pie(ax, nodes.at[zone, "x"], nodes.at[zone, "y"], row)
    year_label(ax, "2040", "What gets built", "FES 2025 Holistic Transition")
    totals = mix.sum()
    # Six technologies in one column (two columns did not fit), the pie sizes in a row beneath.
    leg = key.legend(handles=[dot(TECH[g], f"{g} ({totals.get(g, 0):.0f} GW)") for g in GENERATION],
                     loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=LEGEND_FS - 1,
                     handletextpad=0.5, labelspacing=0.4)
    key.add_artist(leg)
    key.legend(handles=pie_size_key(ax, fig, (10, 30)), loc="upper center", bbox_to_anchor=(0.5, 0.2), ncol=2,
               frameon=False, fontsize=LEGEND_FS, handletextpad=1.4, columnspacing=3.0)
    save(fig, out, dpi=170)


# ---- 2050: flexibility at net zero -------------------------------------------------------------------
def figure_2050(resources: Path, land, out: Path) -> None:
    net = load_network(resources / "network/FES25_HT_2050.nc")
    nodes = zone_nodes(net)
    links, stores = net["links"], net["stores"]
    elec = links[links.carrier == "electrolysis"]
    turb = links[links.carrier == "H2_turbine"]
    grid = stores[stores.carrier.isin(GRID_STORAGE)]
    series = {
        "storage": grid.groupby(to_zone(grid.bus, net, nodes)).p_nom.sum() / 1000,
        "electrolysers": elec.groupby(to_zone(elec.bus0, net, nodes)).p_nom.sum() / 1000,
        "turbines": turb.groupby(to_zone(turb.bus1, net, nodes)).p_nom.sum() / 1000,
    }
    style = {  # face, edge, hatch
        "storage": (TECH["Storage"], "white", None),
        "electrolysers": ("white", TECH["Hydrogen"], "////"),
        "turbines": (TECH["Hydrogen"], "white", None),
    }
    fig, ax, key = map_figure(KEY_HEIGHT)
    base_map(ax, land)
    ax.set_xlim(-20_000, 715_000)
    draw_lines(ax, load_network(resources / "network/FES25_HT_2050_network.nc"), alpha=0.35)
    width = 17_000.0
    for zone, (x, y) in nodes.iterrows():
        values = [series[k].get(zone, 0.0) for k in style]
        if sum(values) < 0.05:
            continue
        x0 = x - 1.5 * width
        for i, (name, value) in enumerate(zip(style, values)):
            face, edge, hatch = style[name]
            ax.add_patch(Rectangle((x0 + i * width, y), width * 0.92, max(value, 0.02) * METRES_PER_GW,
                                   facecolor=face, edgecolor=edge, hatch=hatch, linewidth=0.9, zorder=5))
        ax.plot([x0 - 2_000, x0 + 3 * width], [y, y], color=INK["muted"], lw=0.8, zorder=6)
    year_label(ax, "2050", "Flexibility at net zero", "FES 2025 Holistic Transition")
    totals = {k: v.sum() for k, v in series.items()}
    colour_key(key, [Patch(facecolor=TECH["Storage"], edgecolor="white",
                           label=f"Grid storage ({totals['storage']:.0f} GW)"),
                     Patch(facecolor="white", edgecolor=TECH["Hydrogen"], hatch="////",
                           label=f"Electrolysers ({totals['electrolysers']:.0f} GW)"),
                     Patch(facecolor=TECH["Hydrogen"], edgecolor="white",
                           label=f"Hydrogen turbines ({totals['turbines']:.0f} GW)")], ncol=1,
               fontsize=LEGEND_FS)
    # a 10 GW scale bar in the empty sea to the south-west
    sx, sy = 40_000.0, 40_000.0
    ax.add_patch(Rectangle((sx, sy), width * 0.92, 10 * METRES_PER_GW, facecolor=INK["axis"], edgecolor="white",
                           zorder=5))
    ax.text(sx + width * 1.2, sy + 5 * METRES_PER_GW, "10 GW", va="center", fontsize=LEGEND_FS, color=INK["text"])
    save(fig, out, dpi=170)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resources", type=Path, default=ROOT / "resources")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/source/_static/landing")
    args = parser.parse_args()
    apply_style()
    land = load_land()
    figure_2030(args.resources, land, args.out / "constraints_2030.png")
    figure_2040(args.resources, land, args.out / "generation_2040.png")
    figure_2050(args.resources, land, args.out / "flexibility_2050.png")
    match_sizes([args.out / "constraints_2030.png", args.out / "generation_2040.png",
                 args.out / "flexibility_2050.png"])


def match_sizes(paths: list[Path]) -> None:
    """Pad each image with white, centred, to the largest width and height, so the three maps show the same size
    side by side (each is saved trimmed to its own contents, and the 2030 legend makes that one wider)."""
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in paths]
    width, height = max(i.width for i in images), max(i.height for i in images)
    for path, image in zip(paths, images):
        if image.size == (width, height):
            continue
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        canvas.save(path, optimize=True)


if __name__ == "__main__":
    main()
