"""Presentation figures for Validation_Jan2020 market validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from matplotlib.patches import Rectangle


SCENARIO = "Validation_Jan2020"
ROOT = Path(".")
OUT_DIR = ROOT / "resources" / "analysis" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "CCGT": "#4C78A8",
    "OCGT": "#72B7B2",
    "coal": "#3B3B3B",
    "nuclear": "#F58518",
    "biomass": "#54A24B",
    "waste_to_energy": "#8CD17D",
    "wind_onshore": "#59A14F",
    "wind_offshore": "#76B7B2",
    "solar_pv": "#EDC948",
    "embedded_wind": "#86BCB6",
    "embedded_solar": "#F1CE63",
    "large_hydro": "#1F77B4",
    "small_hydro": "#9ECAE1",
    "Pumped Storage Hydroelectricity": "#9467BD",
    "Battery": "#B07AA1",
    "EU_import": "#BAB0AC",
    "oil": "#8C564B",
    "other": "#C7C7C7",
    "unknown": "#C7C7C7",
}


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _carrier_color(carrier: str) -> str:
    return COLORS.get(carrier, COLORS["other"])


def merit_order_figure() -> None:
    network = pypsa.Network(ROOT / "resources" / "market" / f"{SCENARIO}_wholesale.nc")
    gen = network.generators.copy()
    gen = gen[(gen["p_nom"] > 0) & (gen["carrier"] != "load_shedding")].copy()

    snapshots = network.snapshots
    pmax = pd.DataFrame(
        np.outer(np.ones(len(snapshots)), gen["p_max_pu"].reindex(gen.index).fillna(1.0).values),
        index=snapshots,
        columns=gen.index,
    )
    if not network.generators_t.p_max_pu.empty:
        tv_cols = [c for c in gen.index if c in network.generators_t.p_max_pu.columns]
        pmax.loc[:, tv_cols] = network.generators_t.p_max_pu.reindex(snapshots)[tv_cols].fillna(1.0)

    gen["available_mw"] = gen["p_nom"] * pmax.mean(axis=0).reindex(gen.index).fillna(1.0)
    gen = gen[gen["available_mw"] > 1.0].copy()
    gen["marginal_cost"] = pd.to_numeric(gen["marginal_cost"], errors="coerce").fillna(0.0)
    gen["carrier_plot"] = gen["carrier"].where(gen["carrier"].isin(COLORS), "other")

    # Aggregate tiny generators with the same carrier and near-identical MC to keep the
    # merit-order figure readable while preserving the cumulative shape.
    gen["mc_bin"] = gen["marginal_cost"].round(1)
    blocks = (
        gen.groupby(["carrier_plot", "mc_bin"], as_index=False)["available_mw"]
        .sum()
        .rename(columns={"carrier_plot": "carrier", "mc_bin": "marginal_cost"})
        .sort_values(["marginal_cost", "carrier"])
    )
    blocks["left_gw"] = blocks["available_mw"].cumsum().shift(fill_value=0) / 1000.0
    blocks["width_gw"] = blocks["available_mw"] / 1000.0

    wholesale = pd.read_csv(
        ROOT / "resources" / "market" / f"{SCENARIO}_wholesale_dispatch.csv",
        index_col=0,
        parse_dates=True,
    )
    served = wholesale.sum(axis=1) / 1000.0
    prices = pd.read_csv(
        ROOT / "resources" / "market" / f"{SCENARIO}_wholesale_price.csv",
        index_col=0,
        parse_dates=True,
    )
    price_col = "wholesale_price" if "wholesale_price" in prices.columns else prices.columns[0]
    mean_price = float(prices[price_col].mean())

    fig, ax = plt.subplots(figsize=(13.3, 7.2))
    y_min, y_max = -120, 220
    for row in blocks.itertuples(index=False):
        mc = float(row.marginal_cost)
        y0 = min(0.0, mc)
        height = abs(mc)
        if height < 1.0:
            # Keep near-zero generators visible without misrepresenting the
            # sign of the block.
            y0 = -0.5 if mc < 0 else 0.0
            height = 1.0
        rect = Rectangle(
            (row.left_gw, y0),
            row.width_gw,
            height,
            facecolor=_carrier_color(row.carrier),
            edgecolor="#FFFFFF",
            linewidth=0.4,
            alpha=0.88,
        )
        ax.add_patch(rect)

    ax.axhline(mean_price, color="#111111", lw=1.8, ls="--", label=f"Mean SMP: GBP {mean_price:.1f}/MWh")
    ax.axvline(served.mean(), color="#D62728", lw=1.8, label=f"Mean served load: {served.mean():.1f} GW")
    ax.axvline(served.max(), color="#D62728", lw=1.8, ls=":", label=f"Peak served load: {served.max():.1f} GW")

    ax.set_xlim(0, blocks["left_gw"].max() + blocks["width_gw"].iloc[-1] + 2)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Cumulative average available capacity (GW)")
    ax.set_ylabel("Marginal cost (GBP/MWh)")
    ax.set_title("Wholesale Merit Order - Validation_Jan2020")
    ax.grid(axis="y", color="#DDDDDD", lw=0.8)

    carriers = [c for c in blocks["carrier"].drop_duplicates() if c != "other"]
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=_carrier_color(c), edgecolor="none", label=c)
        for c in carriers[:14]
    ]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], loc="upper left", ncol=2, frameon=False)
    fig.text(
        0.01,
        0.01,
        "Capacity uses January mean availability from the wholesale network; load-shedding excluded.",
        fontsize=9,
        color="#555555",
    )
    _save(fig, f"{SCENARIO}_fig1_wholesale_merit_order")


def bm_volume_comparison_figure() -> None:
    model = pd.read_csv(ROOT / "resources" / "market" / f"{SCENARIO}_redispatch_summary.csv")
    model_carrier = model.groupby("carrier")[["increase_MWh", "decrease_MWh"]].sum()

    # Latest BOALF fetch failed, so use the cached carrier aggregate produced by
    # the successful deep-dive run. This is ELEXON BOALF all flags for Jan 2020.
    elexon = pd.read_csv(ROOT / "resources" / "analysis" / f"{SCENARIO}_deep_dive_bm_carrier.csv")
    elexon = elexon.rename(columns={"Unnamed: 0": "carrier"}).set_index("carrier")
    elexon = elexon[["elexon_increase_mwh", "elexon_decrease_mwh"]]

    carriers = [
        "CCGT",
        "coal",
        "Pumped Storage Hydroelectricity",
        "large_hydro",
        "biomass",
        "nuclear",
        "wind_onshore",
        "wind_offshore",
        "unknown",
    ]
    rows = []
    for carrier in carriers:
        rows.append(
            {
                "carrier": carrier,
                "model_increase": model_carrier.get("increase_MWh", pd.Series()).get(carrier, 0.0) / 1000.0,
                "model_decrease": model_carrier.get("decrease_MWh", pd.Series()).get(carrier, 0.0) / 1000.0,
                "elexon_increase": elexon.get("elexon_increase_mwh", pd.Series()).get(carrier, 0.0) / 1000.0,
                "elexon_decrease": elexon.get("elexon_decrease_mwh", pd.Series()).get(carrier, 0.0) / 1000.0,
            }
        )
    df = pd.DataFrame(rows).set_index("carrier")

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 7.2), sharey=True)
    y = np.arange(len(df))
    h = 0.36
    labels = [c.replace("Pumped Storage Hydroelectricity", "pumped storage") for c in df.index]

    for ax, side, title in [
        (axes[0], "increase", "BM increase volumes"),
        (axes[1], "decrease", "BM decrease volumes"),
    ]:
        ax.barh(y - h / 2, df[f"model_{side}"], height=h, color="#4C78A8", label="Model")
        ax.barh(y + h / 2, df[f"elexon_{side}"], height=h, color="#F58518", label="ELEXON BOALF")
        ax.set_title(title)
        ax.set_xlabel("Volume (GWh)")
        ax.grid(axis="x", color="#DDDDDD", lw=0.8)
        xmax = max(df[f"model_{side}"].max(), df[f"elexon_{side}"].max())
        ax.set_xlim(0, xmax * 1.12 if xmax > 0 else 1)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Balancing Mechanism Volumes by Carrier - Validation_Jan2020", y=0.98)
    fig.text(
        0.01,
        0.01,
        "ELEXON comparison uses cached BOALF all-flag carrier aggregates from the successful deep-dive run; latest BOALF API refresh failed.",
        fontsize=9,
        color="#555555",
    )
    _save(fig, f"{SCENARIO}_fig2_bm_volumes_vs_elexon")


def _boundary_model_flow(network: pypsa.Network, mapping: dict, boundary: str) -> pd.Series:
    spec = mapping["boundaries"][boundary]
    groups = spec.get("flow_groups", {})
    pos = groups.get("positive", spec.get("lines", [])) or []
    neg = groups.get("negative", []) or []
    flow = pd.Series(0.0, index=network.snapshots)

    for line in pos:
        if line in network.lines_t.p0.columns:
            flow = flow + network.lines_t.p0[line]
    for line in neg:
        if line in network.lines_t.p0.columns:
            flow = flow - network.lines_t.p0[line]

    for link in spec.get("links", []) or []:
        if isinstance(link, dict):
            name = link.get("name")
            sign = float(link.get("sign", 1.0))
        else:
            name = link
            sign = 1.0
        if name in network.links_t.p0.columns:
            flow = flow + sign * network.links_t.p0[name]
    return flow


def neso_outage_comparison_figure() -> None:
    network = pypsa.Network(ROOT / "resources" / "market" / f"{SCENARIO}_balancing.nc")
    with open(ROOT / "data" / "network" / "neso_boundary_mapping.yaml", "r", encoding="utf-8") as fh:
        mapping = yaml.safe_load(fh)

    neso = pd.read_csv(ROOT / "data" / "validation" / "day_ahead_constraint_flows_limits.csv")
    neso["datetime"] = pd.to_datetime(neso["Date (GMT/BST)"], errors="coerce")
    neso = neso[(neso["datetime"] >= "2020-01-01") & (neso["datetime"] < "2020-02-01")]
    neso["hour"] = neso["datetime"].dt.floor("h")

    boundaries = ["SCOTEX", "SSE-SP", "SSHARN"]
    fig, axes = plt.subplots(len(boundaries), 1, figsize=(13.3, 7.5), sharex=True)
    for ax, boundary in zip(axes, boundaries):
        nb = neso[neso["Constraint Group"] == boundary]
        hourly = nb.groupby("hour")[["Limit (MW)", "Flow (MW)"]].mean()
        model_flow = _boundary_model_flow(network, mapping, boundary)
        model_flow = model_flow.reindex(hourly.index).interpolate(limit_direction="both")
        limit_gw = hourly["Limit (MW)"] / 1000.0
        finite_limit_gw = limit_gw.where(limit_gw < 20.0)
        flow_gw = hourly["Flow (MW)"] / 1000.0
        model_gw = model_flow / 1000.0

        ax.fill_between(
            hourly.index,
            0,
            finite_limit_gw,
            color="#D9E8FB",
            alpha=0.8,
            label="NESO DA limit",
        )
        ax.plot(hourly.index, flow_gw, color="#F58518", lw=1.4, label="NESO DA flow")
        ax.plot(model_flow.index, model_gw, color="#1F4E79", lw=1.7, label="Model BM flow")
        ax.set_ylabel(f"{boundary}\nGW")
        ax.grid(axis="y", color="#DDDDDD", lw=0.8)
        ymax = max(
            flow_gw.quantile(0.99),
            model_gw.quantile(0.99),
            finite_limit_gw.max(skipna=True) if finite_limit_gw.notna().any() else 0,
        )
        ax.set_ylim(0, max(1.0, ymax * 1.18))
        constrained = hourly["Limit (MW)"] < 50000
        util = (
            hourly.loc[constrained, "Flow (MW)"]
            / hourly.loc[constrained, "Limit (MW)"]
        ).replace([np.inf, -np.inf], np.nan)
        ax.text(
            0.99,
            0.83,
            f"NESO >=90% limit: {(util >= 0.9).mean() * 100:.0f}% of periods",
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
            color="#333333",
        )

    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[-1].set_xlabel("January 2020")
    fig.suptitle("NESO Outage-Driven Boundary Limits vs Model Boundary Flows", y=0.98)
    fig.text(
        0.01,
        0.01,
        "NESO data: Day-Ahead Constraint Flows and Limits. Model data: balancing network line/link flows aggregated with the NESO boundary mapping.",
        fontsize=9,
        color="#555555",
    )
    _save(fig, f"{SCENARIO}_fig3_neso_outages_vs_model")


def wholesale_price_comparison_figure() -> None:
    model = pd.read_csv(
        ROOT / "resources" / "market" / f"{SCENARIO}_wholesale_price.csv",
        index_col=0,
        parse_dates=True,
    )
    model.index.name = "datetime"
    model_price = model["wholesale_price"].rename("Model SMP")

    elexon = pd.read_csv(
        ROOT / "resources" / "market" / "elexon" / "mid_prices_2020.csv",
        index_col=0,
        parse_dates=True,
    )
    elexon_price = elexon.iloc[:, 0].rename("ELEXON MIP/MID")
    # ELEXON MID is half-hourly; compare to the hourly model using hourly means.
    elexon_hourly = elexon_price.resample("h").mean()

    comparison = pd.concat([model_price, elexon_hourly], axis=1).dropna()
    comparison = comparison.loc["2020-01-01":"2020-01-31 23:00"]
    diff = comparison["Model SMP"] - comparison["ELEXON MIP/MID"]
    corr = comparison["Model SMP"].corr(comparison["ELEXON MIP/MID"])
    mae = diff.abs().mean()
    bias = diff.mean()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.3, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    ax = axes[0]
    ax.plot(
        comparison.index,
        comparison["ELEXON MIP/MID"],
        color="#F58518",
        lw=1.5,
        label=f"ELEXON MIP/MID mean: GBP {comparison['ELEXON MIP/MID'].mean():.1f}/MWh",
    )
    ax.plot(
        comparison.index,
        comparison["Model SMP"],
        color="#1F4E79",
        lw=1.7,
        label=f"Model wholesale SMP mean: GBP {comparison['Model SMP'].mean():.1f}/MWh",
    )
    ax.set_ylabel("GBP/MWh")
    ax.set_title("Wholesale Price: Model SMP vs ELEXON Market Index Price")
    ax.grid(axis="y", color="#DDDDDD", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.92,
        f"r = {corr:.2f}\nMAE = GBP {mae:.1f}/MWh\nBias = GBP {bias:+.1f}/MWh",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.85},
    )

    axes[1].axhline(0, color="#333333", lw=1.0)
    axes[1].fill_between(
        comparison.index,
        0,
        diff,
        where=diff >= 0,
        color="#4C78A8",
        alpha=0.55,
        interpolate=True,
        label="Model above ELEXON",
    )
    axes[1].fill_between(
        comparison.index,
        0,
        diff,
        where=diff < 0,
        color="#E45756",
        alpha=0.55,
        interpolate=True,
        label="Model below ELEXON",
    )
    axes[1].set_ylabel("Delta\nGBP/MWh")
    axes[1].set_xlabel("January 2020")
    axes[1].grid(axis="y", color="#DDDDDD", lw=0.8)

    fig.text(
        0.01,
        0.01,
        "ELEXON series from resources/market/elexon/mid_prices_2020.csv, resampled from half-hourly to hourly means.",
        fontsize=9,
        color="#555555",
    )
    _save(fig, f"{SCENARIO}_fig4_wholesale_price_vs_elexon_mip")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.titlesize": 16,
            "legend.fontsize": 9,
        }
    )
    print("Creating wholesale merit-order figure...")
    merit_order_figure()
    print("Creating BM volume comparison figure...")
    bm_volume_comparison_figure()
    print("Creating NESO outage comparison figure...")
    neso_outage_comparison_figure()
    print("Creating wholesale price comparison figure...")
    wholesale_price_comparison_figure()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
