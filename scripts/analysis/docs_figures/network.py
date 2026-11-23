"""Draw the figures of the network pages of the documentation (docs/source/_static/figures/network/).

Topology (docs/source/model/network/topology.md):

1. network_models.png: the three network models side by side: the ETYS network of the Clean Power 2030 build
   (resources/network/CP30_FFR_2030_network.nc) by voltage, the Reduced network (data/network/reduced_network/)
   and the Zonal network (data/network/zonal/).
2. appendix_b_works_2030.png: the ETYS Appendix B works the 2030 build applied, coloured by the year Appendix B
   dates them (resources/network/CP30_FFR_2030_etys_upgrade_audit.csv, drawn on CP30_FFR_2030_network.nc).

Clustering (docs/source/model/network/clustering.md):

3. clustering_2050.png: the full ETYS network of the FES 2050 build (resources/network/FES25_HT_2050_network.nc)
   beside the same network clustered on NESO's 17 economic zones (resources/network/FES25_HT_2050.nc), with
   the zones (data/CP30/validation_references/neso_cp08_zones.geojson).

Constraints (docs/source/model/network/constraints.md):

4. etys_boundaries.png: the ETYS boundaries (data/network/ETYS/etys_boundaries_mar25_osgb36.geojson), those with
   a 2030 secured capability in the CP30 central case marked with it
   (data/network/etys2024_cp30_boundary_capability_noacc_etys2025_ec5_le1.csv), on the 2030 network.
5. seasonal_line_ratings.png: each month's line rating as a share of the winter rating, by voltage, for the lines
   with published ETYS seasonal ratings (CP30_FFR_2030_network.nc, the solve's month calendar).
6. boundary_limit_2030.png: the hourly limit of three boundaries through 2030 in the CP30 central case: with the
   NOA seasonal factors alone and with the historic outage derate the study uses
   (data/network/neso_boundary_outage_generic_2022_25.csv, replay), computed by scripts/solve/secured_boundaries.py.

Run from the repository root after the networks exist:

    python scripts/analysis/docs_figures/network.py [--resources resources] [--out docs/source/_static/figures/network]

Not part of any Snakemake rule: the figures are documentation assets, committed as PNG.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.analysis.docs_figures.common import (  # noqa: E402
    DIVERGING, FIGURES, INK, ROOT, SAGE, SEQUENTIAL, XLIM, YLIM, apply_style, base_map, colour_key, dot,
    draw_lines, gb_dc_links, load_land, load_network, map_figure, save,
)

NETWORK_2030 = "network/CP30_FFR_2030_network.nc"
AUDIT_2030 = "network/CP30_FFR_2030_etys_upgrade_audit.csv"
NETWORK_2050_FULL = "network/FES25_HT_2050_network.nc"
NETWORK_2050_CLUSTERED = "network/FES25_HT_2050.nc"
BOUNDARIES = ROOT / "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson"
CAPABILITY_CENTRAL = ROOT / "data/network/etys2024_cp30_boundary_capability_noacc_etys2025_ec5_le1.csv"
OUTAGE_DERATE = ROOT / "data/network/neso_boundary_outage_generic_2022_25.csv"
NESO_ZONES = ROOT / "data/CP30/validation_references/neso_cp08_zones.geojson"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def hvdc_handle() -> Line2D:
    return Line2D([], [], color=SAGE["hvdc"], lw=1.6, ls=(0, (3, 2)), label="HVDC links")


def draw_segments(ax, b: pd.DataFrame, bus0, bus1, **kw) -> None:
    for u, v in zip(bus0, bus1):
        if u in b.index and v in b.index:
            ax.plot([b.at[u, "x"], b.at[v, "x"]], [b.at[u, "y"], b.at[v, "y"]], **kw)


def draw_internal_hvdc(ax, net: dict, lw: float = 1.6) -> None:
    dc = gb_dc_links(net)
    draw_segments(ax, net["buses"], dc.bus0, dc.bus1, color=SAGE["hvdc"], lw=lw, ls=(0, (3, 2)), zorder=3)


def to_osgb(df: pd.DataFrame, lon: str = "x", lat: str = "y") -> pd.DataFrame:
    pts = gpd.GeoSeries(gpd.points_from_xy(df[lon], df[lat]), crs=4326).to_crs(27700)
    return df.assign(x=pts.x.values, y=pts.y.values)


# ---- topology ---------------------------------------------------------------------------------------
def figure_network_models(net: dict, land, out: Path) -> None:
    fig = plt.figure(figsize=(12.0, 6.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[6.2, 0.4], hspace=0.02, wspace=0.02)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    key = fig.add_subplot(grid[1, :])
    key.axis("off")

    ax = axes[0]
    base_map(ax, land)
    draw_lines(ax, net)
    draw_internal_hvdc(ax, net, lw=1.3)
    gb = net["buses"][net["buses"].country == "GB"]
    ax.set_title(f"ETYS: {len(gb):,} buses", loc="center", y=1.0)

    ax = axes[1]
    base_map(ax, land)
    rb = pd.read_csv(ROOT / "data/network/reduced_network/buses.csv").set_index("name")
    rl = pd.read_csv(ROOT / "data/network/reduced_network/lines.csv")
    for (u, v), s_nom in rl.groupby(["bus0", "bus1"]).s_nom.sum().items():
        ax.plot([rb.at[u, "x"], rb.at[v, "x"]], [rb.at[u, "y"], rb.at[v, "y"]], color=SAGE["deep"],
                lw=0.6 + s_nom / 2500, solid_capstyle="round", zorder=2)
    ax.scatter(rb.x, rb.y, s=16, color=SAGE["deep"], edgecolors="white", linewidths=0.6, zorder=4)
    ax.set_title(f"Reduced: {len(rb)} buses", loc="center", y=1.0)

    ax = axes[2]
    zones = gpd.clip(gpd.read_file(ROOT / "data/network/zonal/zones.geojson").to_crs(27700), land)
    base_map(ax, land)
    zones.boundary.plot(ax=ax, color=INK["axis"], linewidth=0.5, zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    zb = to_osgb(pd.read_csv(ROOT / "data/network/zonal/buses.csv")).set_index("name")
    zl = pd.read_csv(ROOT / "data/network/zonal/links.csv")
    zl = zl[zl.bus0.isin(zb.index) & zl.bus1.isin(zb.index)]
    for u, v, p_nom in zip(zl.bus0, zl.bus1, zl.p_nom):
        ax.plot([zb.at[u, "x"], zb.at[v, "x"]], [zb.at[u, "y"], zb.at[v, "y"]], color=SAGE["deep"],
                lw=0.6 + p_nom / 2500, solid_capstyle="round", zorder=2)
    ax.scatter(zb.x, zb.y, s=16, color=SAGE["deep"], edgecolors="white", linewidths=0.6, zorder=4)
    ax.set_title(f"Zonal: {len(zb)} nodes, {len(zl)} links", loc="center", y=1.0)

    colour_key(key, [Line2D([], [], color=SAGE["deep"], lw=1.2, label="400 kV"),
                     Line2D([], [], color=SAGE["mid"], lw=1.0, label="275 kV"),
                     Line2D([], [], color=SAGE["light"], lw=1.0, label="132 kV and below"),
                     hvdc_handle(),
                     Line2D([], [], color=SAGE["deep"], lw=0.6 + 5000 / 2500,
                            label="Reduced and Zonal: width by rating (5 GW shown)")], ncol=5)
    save(fig, out, dpi=150)


YEAR_GROUPS = [  # (label, test on the Appendix B year, colour)
    ("2025", lambda y, s: (y == 2025) & ~s, SEQUENTIAL[2]),
    ("2026-27", lambda y, s: (y >= 2026) & (y <= 2027) & ~s, SEQUENTIAL[4]),
    ("2028-30", lambda y, s: (y >= 2028) & (y <= 2030) & ~s, SEQUENTIAL[6]),
    ("Later, brought forward", lambda y, s: s, DIVERGING["high"]),
]


def figure_appendix_b_works(net: dict, land, audit_path: Path, out: Path) -> None:
    audit = pd.read_csv(audit_path)
    applied = audit[audit.audit_status.str.startswith("applied") & audit.operation.isin(["addition", "change"])]
    applied = applied.assign(pulled=applied.audit_status.isin(["applied_as_cp30_dated_work",
                                                               "applied_as_hvdc_enabling_works"]))
    b = net["buses"]
    lines, links = net["lines"], net["links"]
    fig, ax, key = map_figure(key_height=1.15)
    base_map(ax, land)
    draw_lines(ax, net, alpha=0.30)
    handles = []
    for label, test, colour in YEAR_GROUPS:
        rows = applied[test(applied.source_year, applied.pulled)]
        names = rows.network_components.str.split("; ").explode().str.split(":", n=1, expand=True)
        names.columns = ["component", "name"]
        ln = lines.loc[lines.index.intersection(names[names.component == "line"].name)]
        lk = links.loc[links.index.intersection(names[names.component == "link"].name)]
        draw_segments(ax, b, ln.bus0, ln.bus1, color=colour, lw=1.5, solid_capstyle="round", zorder=4)
        draw_segments(ax, b, lk.bus0, lk.bus1, color=colour, lw=1.8, ls=(0, (3, 2)), zorder=4)
        tx = names[names.component == "transformer"].name
        sites = {n.split("_")[0] for n in tx}
        sites = [s for s in sites if s in b.index]
        ax.scatter(b.loc[sites, "x"], b.loc[sites, "y"], s=9, color=colour, edgecolors="white", linewidths=0.3,
                   zorder=5)
        handles.append(Line2D([], [], color=colour, lw=2.0, label=f"{label} ({len(rows)})"))
    handles.append(Line2D([], [], color=SAGE["deep"], alpha=0.3, lw=1.0, label="Network in 2030"))
    leg = colour_key(key, handles, ncol=2)
    leg.set_title("Appendix B year (rows applied)", prop={"size": 9})
    save(fig, out, dpi=170)


# ---- clustering -------------------------------------------------------------------------------------
def shared_borders(zones: gpd.GeoDataFrame, tol: float = 300.0) -> gpd.GeoSeries:
    """The borders between neighbouring zones, without their coastlines."""
    parts = []
    geoms = list(zones.geometry)
    for i, gi in enumerate(geoms):
        for gj in geoms[i + 1:]:
            if gi.distance(gj) <= tol:
                edge = gi.boundary.intersection(gj.buffer(tol))
                if not edge.is_empty:
                    parts.append(edge)
    return gpd.GeoSeries(parts, crs=zones.crs)


def figure_clustering(full: dict, clustered: dict, land, out: Path) -> None:
    zones = gpd.read_file(NESO_ZONES).to_crs(27700)
    fig = plt.figure(figsize=(9.6, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[6.6, 0.8], hspace=0.02, wspace=0.02)
    left, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    key = fig.add_subplot(grid[1, :])
    key.axis("off")

    base_map(left, land)
    inner = shared_borders(zones)
    inner.plot(ax=left, color=INK["muted"], linewidth=0.7, linestyle=(0, (3, 1.5)), zorder=3)
    draw_lines(left, full)
    draw_internal_hvdc(left, full, lw=1.2)
    left.set_title(f"Full ETYS: {len(full['buses']):,} buses", loc="center", y=1.0)

    base_map(right, land)
    inner.plot(ax=right, color=INK["axis"], linewidth=0.7, linestyle=(0, (3, 1.5)), zorder=1)
    b = clustered["buses"]
    lines = clustered["lines"]
    per_gw = 0.15
    for u, v, s_nom in zip(lines.bus0, lines.bus1, lines.s_nom):
        right.plot([b.at[u, "x"], b.at[v, "x"]], [b.at[u, "y"], b.at[v, "y"]], color=SAGE["deep"],
                   lw=0.5 + s_nom / 1000 * per_gw, solid_capstyle="round", alpha=0.85, zorder=3)
    draw_internal_hvdc(right, clustered, lw=1.4)
    gb = b[(b.country == "GB") & (b.carrier == "AC")]
    right.scatter(gb.x, gb.y, s=85, color="white", edgecolors=SAGE["deep"], linewidths=1.0, zorder=5)
    for name, r in gb.iterrows():
        right.annotate(name.lstrip("Z"), (r.x, r.y), xytext=(0, 0), textcoords="offset points", ha="center",
                       va="center", fontsize=6.5, color=INK["text"], zorder=6)
    right.set_title(f"Clustered: {len(gb)} zones, {len(lines)} lines", loc="center", y=1.0)

    widths = [Line2D([], [], color=SAGE["deep"], lw=0.5 + gw * per_gw, label=f"{gw} GW") for gw in (5, 20, 40)]
    colour_key(key, [Line2D([], [], color=SAGE["deep"], lw=1.2, label="400 kV"),
                     Line2D([], [], color=SAGE["mid"], lw=1.0, label="275 kV"),
                     Line2D([], [], color=SAGE["light"], lw=1.0, label="132 kV and below"),
                     hvdc_handle(),
                     Line2D([], [], color=INK["muted"], lw=0.8, ls=(0, (3, 1.5)), label="NESO economic zones")],
               ncol=5)
    key.legend(handles=widths, loc="upper center", bbox_to_anchor=(0.5, 0.55), ncol=4, frameon=False, fontsize=9,
               title="Clustered line rating (summed)", title_fontsize=9)
    save(fig, out, dpi=150)


# ---- constraints ------------------------------------------------------------------------------------
LABEL_AT = {  # boundary -> (fraction along the line it points at, label x and y in km; None = on the line)
    "B0": (0.0, 192, 1078), "B1a": (0.0, 40, 724), "B2": (0.0, 40, 684), "B4": (1.0, 448, 752),
    "B5": (1.0, 465, 703), "B6": (0.0, 135, 516), "B7a": (1.0, 625, 503), "B8": (1.0, 615, 448),
    "B9": (0.5, None, None), "NW2": (1.0, 120, 450), "NW1": (0.0, 120, 410), "NW3": (0.0, 120, 330),
    "NW4": (0.0, 120, 290), "SW1": (0.0, 112, 225), "B13": (1.0, 381, 10), "SC1": (0.12, 200, 55),
    "SC1.5": (0.0, 455, 40), "SC2": (1.0, 535, 45), "SC3": (1.0, 620, 60), "LE1": (0.85, 640, 125),
    "EC5": (0.0, 662, 395), "B14": (0.0, 600, 240),
}
LIMIT_TYPE = {"thermal": SEQUENTIAL[5], "voltage": SEQUENTIAL[3], "stability": SEQUENTIAL[1]}


def figure_boundaries(net: dict, land, out: Path) -> None:
    geoms = gpd.read_file(BOUNDARIES)
    cap = pd.read_csv(CAPABILITY_CENTRAL)
    cap = cap[(cap.year == 2030) & (cap.fes_pathway == "HT")].set_index("boundary")
    inactive = geoms[~geoms["boundary"].isin(cap.index)]
    active = geoms[geoms["boundary"].isin(cap.index)].set_index("boundary")

    fig = plt.figure(figsize=(10.5, 8.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[5.0, 4.2], height_ratios=[7.4, 0.5], hspace=0.12, wspace=0.12)
    ax, bars, key = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    key.axis("off")
    base_map(ax, land)
    draw_lines(ax, net, alpha=0.30)
    inactive.plot(ax=ax, color=INK["axis"], linewidth=1.0, linestyle=(0, (2, 2)), zorder=3)
    active.plot(ax=ax, color=DIVERGING["high"], linewidth=1.6, zorder=4)
    for name, r in active.iterrows():
        frac, tx, ty = LABEL_AT.get(name, (0.5, None, None))
        pt = r.geometry.interpolate(frac, normalized=True)
        text = (pt.x, pt.y) if tx is None else (tx * 1000, ty * 1000)
        ax.annotate(name, (pt.x, pt.y), xytext=text, ha="center", va="center", fontsize=7.5, color=INK["text"],
                    zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=None if tx is None else dict(arrowstyle="-", color=INK["muted"], lw=0.6,
                                                            shrinkA=0, shrinkB=1))

    order = active.geometry.centroid.y.sort_values().index  # south at the bottom
    y = np.arange(len(order))
    rows = cap.loc[order]
    bars.barh(y, rows.capability_mw / 1000, height=0.6, color=[LIMIT_TYPE[t] for t in rows.limit_type])
    for yi, v in zip(y, rows.capability_mw / 1000):
        bars.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=8, color=INK["text"])
    bars.set_yticks(y, order)
    bars.set_xlabel("Secured capability in 2030 (GW)")
    bars.grid(axis="y", visible=False)
    bars.set_ylim(-0.6, len(order) - 0.4)
    bars.set_title("North to south", loc="left", fontsize=9, color=INK["muted"], fontweight="normal")
    colour_key(key, [Line2D([], [], color=DIVERGING["high"], lw=1.6, label=f"Secured boundary ({len(active)})"),
                     Line2D([], [], color=INK["axis"], lw=1.0, ls=(0, (2, 2)), label="Other ETYS boundaries"),
                     *[dot(c, f"{t.capitalize()} limit", size=8) for t, c in LIMIT_TYPE.items()]], ncol=5)
    save(fig, out, dpi=150)


def figure_seasonal_ratings(path: Path, out: Path) -> None:
    import xarray as xr

    ds = xr.open_dataset(path)
    cols = ["s_nom_winter", "s_nom_spring", "s_nom_summer", "s_nom_autumn", "v_nom", "seasonal_rating_source"]
    lines = pd.DataFrame({c: ds[f"lines_{c}"].values for c in cols}, index=ds["lines_i"].values)
    ds.close()
    lines = lines[(lines.seasonal_rating_source == "etys") & (lines.s_nom_winter > 0) & (lines.s_nom_winter < 9999)]
    season_of = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "summer", 6: "summer", 7: "summer",
                 8: "summer", 9: "autumn", 10: "autumn", 11: "winter", 12: "winter"}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    classes = [(400, "400 kV", SEQUENTIAL[6]), (275, "275 kV", SEQUENTIAL[4]), (132, "132 kV", SEQUENTIAL[2])]
    x = np.arange(12)
    for kv, label, colour in classes:
        sel = lines[lines.v_nom == kv]
        ratio = {s: sel[f"s_nom_{s}"] / sel.s_nom_winter for s in ("winter", "spring", "summer", "autumn")}
        med = [ratio[season_of[m + 1]].median() for m in x]
        ax.step(np.r_[x, 12] - 0.5, np.r_[med, med[-1]], where="post", color=colour, lw=2,
                label=f"{label} ({len(sel):,} lines)")
    ax.set_xticks(x, MONTHS)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0.8, 1.04)
    ax.set_ylabel("Median rating, share of winter (pu)")
    ax.legend(loc="lower left", ncol=3)
    for m0, m1, name in ((-0.5, 1.5, "winter"), (1.5, 3.5, "spring"), (3.5, 7.5, "summer"), (7.5, 9.5, "autumn"),
                         (9.5, 11.5, "winter")):
        ax.text((m0 + m1) / 2, 1.025, name, ha="center", va="center", fontsize=9, color=INK["muted"])
    save(fig, out)


def boundary_limits(boundaries: list[str]) -> dict:
    """Hourly 2030 limits of the CP30 central case, with and without the historic outage derate."""
    from scripts.solve.secured_boundaries import (
        capability_row, historic_derate_series, seasonal_factor_series, secured_boundary_config,
    )

    snapshots = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")
    cfg = secured_boundary_config({"secured_capability": {"capability_file": str(CAPABILITY_CENTRAL),
                                                          "capability_year": 2030, "fes_pathway": "HT"}},
                                  {"modelled_year": 2030})
    cap_all = pd.read_csv(CAPABILITY_CENTRAL)
    hist = pd.read_csv(OUTAGE_DERATE)
    out = {}
    for b in boundaries:
        row, _ = capability_row(cap_all, b, 2030, "HT")
        noa = seasonal_factor_series(snapshots, row.limit_type, cfg["seasonal_factors"], cfg["month_season"])
        derate = historic_derate_series(snapshots, noa, hist[hist.boundary == b], "replay")
        k = float(row.capability_mw)
        out[b] = (k, noa * k, derate * k)
    return out


def figure_boundary_limits(out: Path) -> None:
    names = {"B4": "B4 (SHE Transmission to SP Transmission)", "B6": "B6 (Scotland to England)",
             "B7a": "B7a (upper north of England)"}
    limits = boundary_limits(list(names))
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 6.0), sharex=True)
    for ax, (b, title) in zip(axes, names.items()):
        k, noa, derate = limits[b]
        ax.plot(noa.index, noa / 1000, color=INK["muted"], lw=1.4, ls=(0, (4, 2)), label="NOA seasonal factor only")
        ax.plot(derate.index, derate / 1000, color=SEQUENTIAL[4], lw=2.0,
                label="With the historic outage derate (as solved)")
        ax.axhline(k / 1000, color=INK["axis"], lw=0.8, ls=(0, (1, 2)))
        ax.text(pd.Timestamp("2030-07-01"), k / 1000, f"winter capability {k / 1000:.1f} GW", ha="center",
                va="bottom", fontsize=8, color=INK["muted"])
        ax.set_title(title)
        ax.set_ylim(0, k / 1000 * 1.12)
        ax.set_ylabel("Limit (GW)")
    axes[0].legend(loc="lower left", ncol=2)
    axes[-1].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    axes[-1].set_xlim(pd.Timestamp("2030-01-01"), pd.Timestamp("2030-12-31 23:00"))
    fig.tight_layout()
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resources", type=Path, default=ROOT / "resources")
    parser.add_argument("--out", type=Path, default=FIGURES / "network")
    parser.add_argument("--only", nargs="*", default=None, help="figure names to draw (default all)")
    args = parser.parse_args()
    apply_style()
    want = set(args.only or [])

    def wanted(name: str) -> bool:
        return not want or name in want

    land = load_land()
    net = load_network(args.resources / NETWORK_2030)
    if wanted("network_models"):
        figure_network_models(net, land, args.out / "network_models.png")
    if wanted("appendix_b_works_2030"):
        figure_appendix_b_works(net, land, args.resources / AUDIT_2030, args.out / "appendix_b_works_2030.png")
    if wanted("clustering_2050"):
        figure_clustering(load_network(args.resources / NETWORK_2050_FULL),
                          load_network(args.resources / NETWORK_2050_CLUSTERED), land,
                          args.out / "clustering_2050.png")
    if wanted("etys_boundaries"):
        figure_boundaries(net, land, args.out / "etys_boundaries.png")
    if wanted("seasonal_line_ratings"):
        figure_seasonal_ratings(args.resources / NETWORK_2030, args.out / "seasonal_line_ratings.png")
    if wanted("boundary_limit_2030"):
        figure_boundary_limits(args.out / "boundary_limit_2030.png")


if __name__ == "__main__":
    main()
