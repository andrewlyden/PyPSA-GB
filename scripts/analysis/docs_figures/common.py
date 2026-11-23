"""Shared style for the documentation figures (docs/source/_static/figures/, _static/landing/).

Every docs figure uses this module, so the site reads as one set:

- ``TECH``: the eight technology groups in a fixed order, each with one colour, then "Imports" in neutral grey.
  The eight are the validated categorical palette (lightness band, chroma floor, colour-vision-deficiency
  separation and normal-vision separation all pass); never reorder, recolour or cycle them, and never add a
  ninth colour (fold it into "Other renewables", draw it grey, or leave it out). Imports are flows, not plant,
  so they take the neutral grey. Light colours (aqua, yellow, magenta) are below 3:1 against white, so a chart
  using them always carries a legend or direct labels.
- ``carrier_group(carrier)``: PyPSA-GB carrier -> technology group.
- ``SEQUENTIAL``: one blue ramp, light to dark, for magnitudes. ``DIVERGING``: blue and orange poles.
- ``INK``: text, axis and grid colours (text never wears a series colour).
- ``apply_style()``: Matplotlib defaults (hairline recessive grid, no top/right spines, DejaVu Sans).
- GB map helpers: ``load_land``, ``load_network``, ``base_map``, ``draw_lines``, ``map_figure``, ``size_keys``.
- ``save(fig, path)``: PNG on white at 150 dpi, warning above 400 kB.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "docs" / "source" / "_static" / "figures"
LAND = ROOT / "data" / "network" / "GSP" / "GSP_regions_27700_20250109.geojson"

# ---- colour -----------------------------------------------------------------------------------------
TECH = {  # fixed order = the validated categorical order
    "Offshore wind": "#2a78d6",
    "Gas": "#eb6834",
    "Onshore wind": "#1baf7a",
    "Solar": "#eda100",
    "Storage": "#e87ba4",
    "Other renewables": "#008300",   # bioenergy, hydro, marine, geothermal
    "Nuclear": "#6250d6",
    "Hydrogen": "#e34948",           # hydrogen turbines (and electrolysers where shown)
    "Imports": "#9aa5a0",            # neutral grey: flows, not a categorical slot
}
_CARRIERS = {
    "Offshore wind": ["wind_offshore", "floating_wind"],
    "Gas": ["CCGT", "OCGT", "CHP", "gas_engine", "CCS Gas", "oil", "coal"],
    "Onshore wind": ["wind_onshore"],
    "Solar": ["solar_pv", "solar"],
    "Storage": ["Battery", "LAES", "pumped_hydro", "CAES", "battery"],
    "Other renewables": ["biomass", "CCS Biomass", "biogas", "waste_to_energy", "large_hydro", "small_hydro",
                         "marine", "tidal_stream", "shoreline_wave", "tidal_lagoon", "geothermal"],
    "Nuclear": ["nuclear"],
    "Hydrogen": ["H2_turbine", "hydrogen", "electrolysis"],
    "Imports": ["EU_market", "interconnector"],
}
CARRIER_GROUP = {c: g for g, cs in _CARRIERS.items() for c in cs}

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
DIVERGING = {"low": "#2a78d6", "mid": "#c8ccc9", "high": "#eb6834"}
INK = {"text": "#2c3a35", "muted": "#66746f", "axis": "#b9c4bf", "grid": "#e6ece9", "surface": "#ffffff"}
SAGE = {"deep": "#3f6f59", "mid": "#7ba591", "light": "#b9c7c1", "land": "#eef4f0", "coast": "#c9dcd1",
        "hvdc": "#e8a317"}


def carrier_group(carrier: str) -> str | None:
    """The technology group of a PyPSA-GB carrier, or None for carriers not shown (load shedding, flexibility)."""
    return CARRIER_GROUP.get(carrier)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK["text"],
        "axes.labelcolor": INK["text"], "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": INK["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelcolor": INK["text"],
        "ytick.labelcolor": INK["text"], "legend.frameon": False, "legend.labelcolor": INK["text"],
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "figure.facecolor": "white",
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    })


def save(fig, path: Path, dpi: int = 150, tight: bool = True) -> Path:
    """Save a figure; ``tight=False`` keeps the whole canvas, so figures drawn on one canvas size match exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None, pad_inches=0.1, facecolor="white")
    plt.close(fig)
    size_kb = path.stat().st_size / 1e3
    note = "  (over 400 kB: simplify)" if size_kb > 400 else ""
    print(f"{path.relative_to(ROOT)}: {size_kb:.0f} kB{note}")
    return path


# ---- GB maps ----------------------------------------------------------------------------------------
XLIM, YLIM = (-20_000, 680_000), (-10_000, 1_225_000)


def load_land() -> gpd.GeoDataFrame:
    return gpd.read_file(LAND).to_crs(27700).dissolve()


def load_network(path: Path) -> dict:
    """The static tables of a PyPSA network file (no time series), read with xarray for speed."""
    ds = xr.open_dataset(path)

    def table(prefix: str, fields: list[str]) -> pd.DataFrame:
        index = f"{prefix}_i"
        if index not in ds.coords and index not in ds.variables:
            return pd.DataFrame(columns=fields)
        cols = {f: ds[f"{prefix}_{f}"].values for f in fields if f"{prefix}_{f}" in ds.variables}
        return pd.DataFrame(cols, index=ds[index].values)

    net = {
        "buses": table("buses", ["x", "y", "v_nom", "country", "carrier"]),
        "lines": table("lines", ["bus0", "bus1", "voltage_kv", "s_nom"]),
        "links": table("links", ["bus0", "bus1", "carrier", "p_nom"]),
        "gens": table("generators", ["bus", "carrier", "p_nom"]),
        "stores": table("storage_units", ["bus", "carrier", "p_nom", "max_hours"]),
        "loads": table("loads", ["bus", "carrier"]),
    }
    ds.close()
    return net


def base_map(ax, land: gpd.GeoDataFrame) -> None:
    land.plot(ax=ax, color=SAGE["land"], edgecolor=SAGE["coast"], linewidth=0.4)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_lines(ax, net: dict, alpha: float = 1.0) -> None:
    """Transmission lines by voltage: 400 kV deep sage, 275 kV sage, lower voltages grey."""
    b, lines = net["buses"], net["lines"]
    kv = lines["voltage_kv"] if "voltage_kv" in lines else net["buses"].loc[lines.bus0, "v_nom"].values
    for low, high, colour, width in ((0, 132, SAGE["light"], 0.35), (132, 275, SAGE["mid"], 0.7),
                                     (275, 1000, SAGE["deep"], 1.0)):
        sel = lines[(kv > low) & (kv <= high) & lines.bus0.isin(b.index) & lines.bus1.isin(b.index)]
        for bus0, bus1 in zip(sel.bus0, sel.bus1):
            ax.plot([b.at[bus0, "x"], b.at[bus1, "x"]], [b.at[bus0, "y"], b.at[bus1, "y"]],
                    color=colour, lw=width, alpha=alpha, solid_capstyle="round", zorder=2)


def gb_dc_links(net: dict) -> pd.DataFrame:
    b, links = net["buses"], net["links"]
    gb = set(b.index[b.country == "GB"])
    dc = links[links.carrier == "DC"]
    return dc[dc.bus0.isin(gb) & dc.bus1.isin(gb)]


def map_figure(key_height: float = 1.55, width: float = 5.0):
    """A GB map with a separate key area underneath, so the key never covers land."""
    fig = plt.figure(figsize=(width, 7.4 * width / 5.0 + key_height))
    grid = fig.add_gridspec(2, 1, height_ratios=[7.4 * width / 5.0, key_height], hspace=0.02)
    ax, key = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    key.axis("off")
    return fig, ax, key


def colour_key(key, handles, ncol: int = 2, top: float = 1.0, fontsize: float = 9):
    leg = key.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, top), ncol=ncol, frameon=False,
                     fontsize=fontsize, handletextpad=0.5, columnspacing=1.2, labelspacing=0.55)
    key.add_artist(leg)
    return leg


def size_key(key, handles, top: float, fontsize: float = 9):
    return key.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, top), ncol=len(handles),
                      frameon=False, fontsize=fontsize, handletextpad=1.3, columnspacing=2.6, borderpad=0.4)


def size_keys(values, per_unit: float, unit: str):
    """Hollow circles for a size legend; ``per_unit`` is the scatter area (points^2) per unit."""
    return [Line2D([], [], marker="o", ls="", markersize=(v * per_unit) ** 0.5, markerfacecolor="none",
                   markeredgecolor=INK["muted"], label=f"{v:g} {unit}") for v in values]


def dot(colour: str, label: str, size: float = 7.5):
    return Line2D([], [], marker="o", ls="", markersize=size, markerfacecolor=colour, markeredgecolor="white",
                  label=label)
