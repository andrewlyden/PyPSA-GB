"""Build an external-facing notebook for the 2020 validation review."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "validation_2020_external_review.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (pypsa-gb)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    nb.cells = [
        markdown(
            """
            # PyPSA-GB 2020 Market Validation

            External review of the PyPSA-GB rolling day-ahead and balancing-market validation runs for calendar year 2020.

            The notebook is ordered for an external reader:

            1. high-level annual validation results;
            2. wholesale market evidence;
            3. balancing mechanism and network-constraint evidence.

            All charts are generated in this notebook from model outputs and reference datasets.
            """
        ),
        code(
            """
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go
            import plotly.express as px
            import plotly.io as pio
            import pypsa
            import yaml
            from IPython.display import HTML, display
            from plotly.subplots import make_subplots

            ROOT = Path.cwd()
            if not (ROOT / "resources" / "analysis").exists():
                for parent in ROOT.parents:
                    if (parent / "resources" / "analysis").exists():
                        ROOT = parent
                        break

            ANALYSIS = ROOT / "resources" / "analysis"
            MARKET = ROOT / "resources" / "market"
            DATA = ROOT / "data"
            SCORECARD = ANALYSIS / "validation_2020_completed_months_scorecard.csv"
            DISPATCH = ANALYSIS / "validation_2020_completed_months_dispatch_vs_espeni.csv"
            BOUNDARIES = ANALYSIS / "validation_2020_completed_months_neso_boundaries.csv"
            BOAV_MONTHLY = ANALYSIS / "elexon_boav_comparison" / "model_vs_elexon_boav_vs_boalf_2020_by_month_carrier.csv"
            BOAV_ANNUAL = ANALYSIS / "elexon_boav_comparison" / "model_vs_elexon_boav_vs_boalf_2020_annual_by_carrier.csv"

            SCENARIO_YEAR = "Validation_2020"
            MONTH_SCENARIOS = [
                "Validation_Jan2020",
                "Validation_Feb2020",
                "Validation_Mar2020",
                "Validation_Apr2020",
                "Validation_May2020",
                "Validation_Jun2020",
                "Validation_Jul2020",
                "Validation_Aug2020",
                "Validation_Sep2020",
                "Validation_Oct2020",
                "Validation_Nov2020",
                "Validation_Dec2020",
            ]
            MONTH_ORDER = [
                "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020",
                "Jul 2020", "Aug 2020", "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
            ]

            pd.options.display.float_format = "{:,.3f}".format
            pio.renderers.default = "notebook_connected"
            plot_template = "plotly_white"

            CARRIER_COLORS = {
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

            def gbp_m(value):
                return f"GBP {value / 1e6:,.1f}m"

            def ratio(value):
                return f"{value:,.2f}x"

            def style_plot(fig, height=430):
                fig.update_layout(
                    template=plot_template,
                    height=height,
                    margin=dict(l=60, r=30, t=70, b=55),
                    font=dict(size=12),
                    legend_title_text="",
                )
                return fig

            score = pd.read_csv(SCORECARD)
            dispatch = pd.read_csv(DISPATCH)
            boundaries = pd.read_csv(BOUNDARIES)
            boav_monthly = pd.read_csv(BOAV_MONTHLY)
            boav_annual = pd.read_csv(BOAV_ANNUAL)

            score["period"] = pd.Categorical(score["period"], MONTH_ORDER, ordered=True)
            score = score.sort_values("period").reset_index(drop=True)
            dispatch["period"] = pd.Categorical(dispatch["period"], MONTH_ORDER, ordered=True)
            boundaries["period"] = pd.Categorical(boundaries["period"], MONTH_ORDER, ordered=True)
            scenario_to_period = dict(zip(MONTH_SCENARIOS, MONTH_ORDER))
            boav_monthly["period"] = pd.Categorical(
                boav_monthly["scenario"].map(scenario_to_period),
                MONTH_ORDER,
                ordered=True,
            )

            score["bm_cost_gbp_m"] = score["bm_cost_gbp"] / 1e6
            score["neso_thermal_cost_gbp_m"] = score["neso_thermal_cost_gbp"] / 1e6
            score["disbsad_cost_gbp_m"] = score["disbsad_cost_gbp"] / 1e6
            """
        ),
        markdown(
            """
            ## 1. High-Level Validation Results

            This section gives the annual picture first. The reference datasets are observed external benchmarks: ELEXON MID is the observed wholesale market index price, ELEXON BOAV is the real accepted BM bid/offer volume benchmark derived from settlement acceptance data, NESO thermal constraint cost is the published system-operator constraint-cost benchmark, and ESPENI is real metered final physical dispatch and demand by technology.
            """
        ),
        code(
            """
            kpis = {
                "Months covered": f"{len(score):.0f}",
                "Model BM cost": gbp_m(score["bm_cost_gbp"].sum()),
                "NESO thermal constraint cost": gbp_m(score["neso_thermal_cost_gbp"].sum()),
                "Model / NESO cost": ratio(score["bm_cost_gbp"].sum() / score["neso_thermal_cost_gbp"].sum()),
                "Model / ELEXON BOAV increase": ratio(boav_annual["model_increase_mwh"].sum() / boav_annual["boav_increase_mwh"].sum()),
                "Model / ELEXON BOAV decrease": ratio(boav_annual["model_decrease_mwh"].sum() / boav_annual["boav_decrease_mwh"].sum()),
                "Mean SMP / MID": ratio(score["smp_mid_ratio"].mean()),
            }

            cards = "".join(
                f'''
                <div style="border:1px solid #d9dee7;border-radius:6px;padding:14px 16px;background:#fbfcfe">
                  <div style="font-size:12px;color:#5d6778;text-transform:uppercase;letter-spacing:.04em">{name}</div>
                  <div style="font-size:24px;font-weight:650;color:#172033;margin-top:4px">{value}</div>
                </div>
                '''
                for name, value in kpis.items()
            )
            display(HTML(f'<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px">{cards}</div>'))
            """
        ),
        markdown(
            """
            **High-level read.** Wholesale pricing is close to ELEXON MID at annual scale. BM cost is the same order of magnitude as NESO thermal constraint cost, but accepted BOAV volumes are materially under-represented, especially wind bid/decrease volume.
            """
        ),
        code(
            """
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.12,
                subplot_titles=(
                    "Model BM redispatch cost vs NESO real thermal constraint cost",
                    "Model wholesale SMP vs ELEXON real MID",
                ),
            )
            fig.add_bar(
                row=1,
                col=1,
                x=score["period"].astype(str),
                y=score["bm_cost_gbp_m"],
                name="Model BM cost",
                marker_color="#345995",
            )
            fig.add_bar(
                row=1,
                col=1,
                x=score["period"].astype(str),
                y=score["neso_thermal_cost_gbp_m"],
                name="NESO thermal constraint cost",
                marker_color="#E26D5A",
            )
            fig.add_scatter(
                row=2,
                col=1,
                x=score["period"].astype(str),
                y=score["mean_smp_gbp_mwh"],
                mode="lines+markers",
                name="Model SMP",
                line=dict(color="#345995", width=3),
            )
            fig.add_scatter(
                row=2,
                col=1,
                x=score["period"].astype(str),
                y=score["mean_mid_gbp_mwh"],
                mode="lines+markers",
                name="ELEXON MID",
                line=dict(color="#E26D5A", width=3),
            )
            fig.update_layout(title="Annual Validation Scorecard", barmode="group")
            fig.update_yaxes(row=1, col=1, title_text="GBP million")
            fig.update_yaxes(row=2, col=1, title_text="GBP/MWh")
            fig.update_xaxes(row=2, col=1, title_text="")
            style_plot(fig, height=720).show()
            """
        ),
        code(
            """
            volume = (
                boav_monthly
                .groupby("period", as_index=False, observed=True)[
                    ["model_increase_mwh", "boav_increase_mwh", "model_decrease_mwh", "boav_decrease_mwh"]
                ]
                .sum()
            )
            volume = volume.melt("period", var_name="series", value_name="MWh")
            volume["direction"] = np.where(volume["series"].str.contains("increase"), "Increase", "Decrease")
            volume["source"] = np.where(volume["series"].str.startswith("model"), "Model", "ELEXON BOAV")
            volume["GWh"] = volume["MWh"] / 1000

            fig = px.bar(
                volume,
                x=volume["period"].astype(str),
                y="GWh",
                color="source",
                facet_row="direction",
                barmode="group",
                color_discrete_map={"Model": "#345995", "ELEXON BOAV": "#E26D5A"},
                labels={"x": "", "GWh": "GWh", "source": ""},
                title="Model BM Volumes vs ELEXON Real Accepted BOAV Volumes",
                template=plot_template,
                height=650,
            )
            fig.update_yaxes(matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.show()
            """
        ),
        code(
            """
            focus = [
                "CCGT",
                "coal",
                "nuclear",
                "biomass",
                "hydro_non_ps",
                "wind_total",
                "solar_total",
                "pumped_hydro_discharge",
                "model_demand_profile",
            ]
            heat = (
                dispatch[dispatch["comparison"].isin(focus)]
                .pivot(index="comparison", columns="period", values="ratio")
                .reindex(focus)
            )
            heat = heat[MONTH_ORDER]
            fig = px.imshow(
                heat,
                color_continuous_scale="RdBu",
                zmin=0,
                zmax=2,
                aspect="auto",
                labels={"x": "", "y": "", "color": "Model / ESPENI"},
                title="Model Final Physical Dispatch vs ESPENI Real Final Dispatch",
                template=plot_template,
                height=520,
            )
            fig.update_traces(text=np.round(heat.values, 2), texttemplate="%{text}")
            fig.show()
            """
        ),
        markdown(
            """
            ## 2. Method and Data Sources

            The model uses a two-step market representation.

            **Stage 1: wholesale day-ahead dispatch.** The network is relaxed to a copperplate system by setting line and transformer capacities to a very large value. The model solves rolling 24-hour windows with storage state of charge carried between windows. The objective is to meet demand at minimum wholesale marginal cost. The resulting dispatch is the day-ahead position and the demand-bus marginal price is reported as model SMP.

            **Stage 2: balancing mechanism redispatch.** The full transmission network is restored, including NESO day-ahead boundary limits and outage-driven constraints. Generators and storage are anchored to wholesale positions using increase and decrease variables:

            `physical dispatch = wholesale position + BM increase - BM decrease`

            The balancing objective minimises bid/offer redispatch cost. This stage produces final physical dispatch, nodal prices, boundary-flow diagnostics, redispatch volumes, and BM cost.
            """
        ),
        markdown(
            """
            | Model area | Source and use |
            |---|---|
            | Network topology | ETYS network representation, with scenario-specific line rating overrides and NESO boundary mappings. |
            | Transmission outages and limits | NESO Day-Ahead Constraint Flows and Limits. These are real operational day-ahead boundary flow and limit records, used both as model inputs and as a boundary-flow benchmark. |
            | Thermal fleet | DUKES historical generator data, mapped to the ETYS network. |
            | Renewable fleet | REPD site data, aggregated by bus/carrier for tractability. |
            | Demand and final-dispatch benchmark | ESPENI real metered final physical demand and technology output, resampled to hourly where needed. |
            | Renewable availability | Weather-driven renewable profiles, calibrated by scenario performance factors. |
            | Wholesale marginal costs | Historical fuel/carbon cost lookup, generator efficiencies, nuclear override, ELEXON empirical calibration, subsidy-aware renewable costs, and biofuel ROC deductions where configured. |
            | Wholesale price benchmark | ELEXON Market Index Price / MID. This is observed wholesale market price data, resampled to hourly means for comparison with model SMP. |
            | BM bids and offers | ELEXON BOD bid/offer data for mapped BMUs where available; carrier-average or configured fallback prices for unmatched units; absolute carrier overrides where strategic BM behaviour is not well represented by marginal-cost markups. |
            | BM volume benchmark | ELEXON BOAV accepted bid/offer volumes from the current BMRS API. BOAV bid volumes are used as decrease actions and BOAV offer volumes as increase actions. BOALF is retained only for flags/timing diagnostics. |

            The key pricing distinction is that wholesale costs represent economic dispatch costs, while BM bid/offer prices represent the cost of changing position after the wholesale schedule has been set.
            """
        ),
        markdown(
            """
            ## 3. Wholesale Market Evidence

            The wholesale validation checks whether the copperplate day-ahead stage produces a plausible merit order and price signal before network redispatch is considered.
            """
        ),
        markdown(
            """
            ### 3.1 Wholesale Merit Order

            The merit-order view ranks 2020 average available capacity by marginal cost and overlays mean and peak served load.
            """
        ),
        code(
            """
            def plot_merit_order(scenario=SCENARIO_YEAR):
                network = pypsa.Network(MARKET / f"{scenario}_wholesale.nc")
                gen = network.generators.copy()
                gen = gen[(gen["p_nom"] > 0) & (gen["carrier"] != "load_shedding")].copy()

                snapshots = network.snapshots
                static_pmax = gen["p_max_pu"].reindex(gen.index).fillna(1.0).astype(float)
                pmax = pd.DataFrame(
                    np.tile(static_pmax.values, (len(snapshots), 1)),
                    index=snapshots,
                    columns=gen.index,
                )
                if not network.generators_t.p_max_pu.empty:
                    tv_cols = [c for c in gen.index if c in network.generators_t.p_max_pu.columns]
                    pmax.loc[:, tv_cols] = (
                        network.generators_t.p_max_pu.reindex(snapshots, columns=tv_cols)
                        .fillna(1.0)
                    )

                gen["available_mw"] = gen["p_nom"] * pmax.mean(axis=0).reindex(gen.index).fillna(1.0)
                gen = gen[gen["available_mw"] > 1.0].copy()
                gen["marginal_cost"] = pd.to_numeric(gen["marginal_cost"], errors="coerce").fillna(0.0)
                gen["carrier_plot"] = gen["carrier"].where(gen["carrier"].isin(CARRIER_COLORS), "other")
                gen["mc_bin"] = gen["marginal_cost"].round(1)

                blocks = (
                    gen.groupby(["carrier_plot", "mc_bin"], as_index=False)["available_mw"]
                    .sum()
                    .rename(columns={"carrier_plot": "carrier", "mc_bin": "marginal_cost"})
                    .sort_values(["marginal_cost", "carrier"])
                )
                blocks["left_gw"] = blocks["available_mw"].cumsum().shift(fill_value=0) / 1000.0
                blocks["width_gw"] = blocks["available_mw"] / 1000.0
                blocks["center_gw"] = blocks["left_gw"] + blocks["width_gw"] / 2.0
                blocks["base"] = np.minimum(blocks["marginal_cost"], 0.0)
                blocks["height"] = blocks["marginal_cost"].abs().clip(lower=1.0)

                wholesale = pd.read_csv(
                    MARKET / f"{scenario}_wholesale_dispatch.csv",
                    index_col=0,
                    parse_dates=True,
                )
                served = wholesale.sum(axis=1) / 1000.0
                prices = pd.read_csv(
                    MARKET / f"{scenario}_wholesale_price.csv",
                    index_col=0,
                    parse_dates=True,
                )
                price_col = "wholesale_price" if "wholesale_price" in prices.columns else prices.columns[0]
                mean_price = float(prices[price_col].mean())

                fig = go.Figure()
                for carrier, group in blocks.groupby("carrier", sort=False):
                    fig.add_bar(
                        x=group["center_gw"],
                        y=group["height"],
                        width=group["width_gw"],
                        base=group["base"],
                        name=carrier,
                        marker=dict(color=CARRIER_COLORS.get(carrier, CARRIER_COLORS["other"]), line=dict(width=0)),
                        customdata=group["marginal_cost"],
                        hovertemplate=(
                            "Carrier: %{fullData.name}<br>"
                            "Capacity block: %{width:.2f} GW<br>"
                            "Marginal cost: %{customdata:.1f} GBP/MWh<extra></extra>"
                        ),
                    )

                fig.add_hline(
                    y=mean_price,
                    line_dash="dash",
                    line_color="#111111",
                    annotation_text=f"Mean SMP: GBP {mean_price:.1f}/MWh",
                    annotation_position="top right",
                )
                fig.add_vline(
                    x=float(served.mean()),
                    line_color="#D62728",
                    annotation_text=f"Mean served load: {served.mean():.1f} GW",
                    annotation_position="top left",
                )
                fig.add_vline(
                    x=float(served.max()),
                    line_dash="dot",
                    line_color="#D62728",
                    annotation_text=f"Peak served load: {served.max():.1f} GW",
                    annotation_position="top right",
                )
                fig.update_layout(
                    title="Wholesale Merit Order - 2020",
                    xaxis_title="Cumulative average available capacity (GW)",
                    yaxis_title="Marginal cost (GBP/MWh)",
                    barmode="overlay",
                    bargap=0,
                    legend=dict(orientation="h", y=-0.22, x=0),
                    yaxis=dict(range=[-120, 220], zeroline=True, zerolinewidth=1, zerolinecolor="#333333"),
                )
                return style_plot(fig, height=650)
            """
        ),
        code("plot_merit_order().show()"),
        markdown(
            """
            ### 3.2 Annual Wholesale Price Fit

            The benchmark is ELEXON MID, the observed wholesale market index price. The model series is the copperplate day-ahead SMP from the wholesale solve.
            """
        ),
        code(
            """
            fig = go.Figure()
            fig.add_scatter(
                x=score["period"].astype(str),
                y=score["smp_mid_ratio"],
                mode="lines+markers+text",
                text=score["smp_mid_ratio"].map(lambda x: f"{x:.2f}x"),
                textposition="top center",
                name="SMP / MID",
                line=dict(color="#4C9F70", width=3),
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#545b66", annotation_text="Parity with ELEXON MID")
            fig.update_layout(title="Wholesale Price Ratio by Month")
            fig.update_yaxes(title_text="Model SMP / ELEXON MID", range=[0, max(1.25, score["smp_mid_ratio"].max() * 1.15)])
            fig.update_xaxes(title_text="")
            style_plot(fig, height=430).show()
            """
        ),
        markdown(
            """
            ### 3.3 Hourly Price Trace

            The hourly trace compares model SMP with observed ELEXON MID. The lower panel is the model-minus-ELEXON delta.
            """
        ),
        code(
            """
            def plot_wholesale_price_trace(scenario=SCENARIO_YEAR):
                model = pd.read_csv(
                    MARKET / f"{scenario}_wholesale_price.csv",
                    index_col=0,
                    parse_dates=True,
                )
                model_price = model["wholesale_price"].rename("Model SMP")

                elexon = pd.read_csv(
                    MARKET / "elexon" / "mid_prices_2020.csv",
                    index_col=0,
                    parse_dates=True,
                )
                elexon_price = elexon.iloc[:, 0].rename("ELEXON MID").resample("h").mean()

                comparison = pd.concat([model_price, elexon_price], axis=1).dropna()
                comparison = comparison.loc["2020-01-01":"2020-12-31 23:00"]
                comparison["Delta"] = comparison["Model SMP"] - comparison["ELEXON MID"]
                corr = comparison["Model SMP"].corr(comparison["ELEXON MID"])
                mae = comparison["Delta"].abs().mean()
                bias = comparison["Delta"].mean()

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[0.72, 0.28],
                    vertical_spacing=0.08,
                )
                fig.add_scatter(
                    row=1,
                    col=1,
                    x=comparison.index,
                    y=comparison["ELEXON MID"],
                    mode="lines",
                    name=f"ELEXON MID mean: GBP {comparison['ELEXON MID'].mean():.1f}/MWh",
                    line=dict(color="#F58518", width=1.5),
                )
                fig.add_scatter(
                    row=1,
                    col=1,
                    x=comparison.index,
                    y=comparison["Model SMP"],
                    mode="lines",
                    name=f"Model SMP mean: GBP {comparison['Model SMP'].mean():.1f}/MWh",
                    line=dict(color="#1F4E79", width=1.8),
                )
                positive_delta = comparison["Delta"].where(comparison["Delta"] >= 0)
                negative_delta = comparison["Delta"].where(comparison["Delta"] < 0)
                fig.add_scatter(
                    row=2,
                    col=1,
                    x=comparison.index,
                    y=positive_delta,
                    mode="lines",
                    name="Model above ELEXON",
                    line=dict(color="#4C78A8", width=0),
                    fill="tozeroy",
                    fillcolor="rgba(76,120,168,0.55)",
                    connectgaps=False,
                )
                fig.add_scatter(
                    row=2,
                    col=1,
                    x=comparison.index,
                    y=negative_delta,
                    mode="lines",
                    name="Model below ELEXON",
                    line=dict(color="#E45756", width=0),
                    fill="tozeroy",
                    fillcolor="rgba(228,87,86,0.55)",
                    connectgaps=False,
                )
                fig.add_hline(row=2, col=1, y=0, line_color="#333333", line_width=1)
                fig.add_annotation(
                    row=1,
                    col=1,
                    text=f"r = {corr:.2f}<br>MAE = GBP {mae:.1f}/MWh<br>Bias = GBP {bias:+.1f}/MWh",
                    xref="paper",
                    yref="paper",
                    x=0.99,
                    y=0.94,
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#DDDDDD",
                    borderwidth=1,
                )
                fig.update_layout(title="Wholesale Price: Model SMP vs ELEXON MID - 2020")
                fig.update_yaxes(row=1, col=1, title_text="GBP/MWh")
                fig.update_yaxes(row=2, col=1, title_text="Delta<br>GBP/MWh")
                fig.update_xaxes(row=2, col=1, title_text="2020")
                fig.update_xaxes(range=[pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")])
                fig.update_layout(xaxis_type="date", xaxis2_type="date")
                return style_plot(fig, height=650)
            """
        ),
        code("plot_wholesale_price_trace().show()"),
        markdown(
            """
            **Wholesale interpretation.** The annual SMP/MID ratio is close to 1.0, so the average price signal is credible. The hourly trace shows that the model still misses some short-lived volatility and high-error periods, which is expected from a simplified day-ahead LP without full unit commitment or all operational constraints.
            """
        ),
        markdown(
            """
            ## 4. Balancing Mechanism and Network Evidence

            This section keeps all BM material together: monthly BM volumes, carrier-level BM behaviour, BM costs against NESO constraint costs, and outage-driven boundary flows.
            """
        ),
        markdown(
            """
            ### 4.1 BM Volumes by Carrier

            Carrier-level BM volumes compare model redispatch with ELEXON BOAV accepted bid/offer volumes. BOAV bid volumes are the real accepted decrease benchmark, and BOAV offer volumes are the real accepted increase benchmark. The old BOALF endpoint-derived volume estimate is shown only in the wind diagnostic because it is not a reliable volume benchmark.
            """
        ),
        code(
            """
            def plot_bm_carrier_volumes():
                carriers = [
                    "CCGT",
                    "coal",
                    "Pumped Storage Hydroelectricity",
                    "large_hydro",
                    "biomass",
                    "wind",
                    "OCGT",
                    "unknown",
                ]
                rows = []
                for carrier in carriers:
                    row = boav_annual[boav_annual["carrier_compare"] == carrier]
                    values = row.iloc[0].to_dict() if not row.empty else {}
                    rows.append(
                        {
                            "carrier": carrier.replace("Pumped Storage Hydroelectricity", "pumped storage"),
                            "model_increase": values.get("model_increase_mwh", 0.0) / 1000.0,
                            "model_decrease": values.get("model_decrease_mwh", 0.0) / 1000.0,
                            "elexon_increase": values.get("boav_increase_mwh", 0.0) / 1000.0,
                            "elexon_decrease": values.get("boav_decrease_mwh", 0.0) / 1000.0,
                        }
                    )
                df = pd.DataFrame(rows)

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    shared_yaxes=True,
                    horizontal_spacing=0.1,
                    subplot_titles=("BM increase volumes", "BM decrease volumes"),
                )
                y = df["carrier"]
                for col, side in [(1, "increase"), (2, "decrease")]:
                    fig.add_bar(
                        row=1,
                        col=col,
                        y=y,
                        x=df[f"model_{side}"],
                        orientation="h",
                        name="Model" if col == 1 else "Model ",
                        showlegend=col == 1,
                        marker_color="#4C78A8",
                    )
                    fig.add_bar(
                        row=1,
                        col=col,
                        y=y,
                        x=df[f"elexon_{side}"],
                        orientation="h",
                        name="ELEXON BOAV" if col == 1 else "ELEXON BOAV ",
                        showlegend=col == 1,
                        marker_color="#F58518",
                    )
                fig.update_layout(
                    title="Model BM Redispatch vs ELEXON Real Accepted BOAV Volumes by Carrier - 2020",
                    barmode="group",
                )
                fig.update_xaxes(title_text="Volume (GWh)", row=1, col=1)
                fig.update_xaxes(title_text="Volume (GWh)", row=1, col=2)
                fig.update_yaxes(autorange="reversed", row=1, col=1)
                return style_plot(fig, height=620)


            def wind_bm_detail_table():
                out = boav_annual[boav_annual["carrier_compare"] == "wind"].copy()
                out = out.rename(columns={"carrier_compare": "carrier"})
                for col in [c for c in out.columns if c.endswith("_mwh")]:
                    out[col.replace("_mwh", "_gwh")] = out[col] / 1000.0
                return out


            def plot_wind_bm_detail():
                wind = wind_bm_detail_table()
                plot = wind.melt(
                    id_vars="carrier",
                    value_vars=[
                        "model_decrease_gwh",
                        "boav_decrease_gwh",
                        "boalf_endpoint_decrease_gwh",
                        "model_increase_gwh",
                        "boav_increase_gwh",
                        "boalf_endpoint_increase_gwh",
                    ],
                    var_name="series",
                    value_name="GWh",
                )
                plot["direction"] = np.where(plot["series"].str.contains("decrease"), "Decrease / curtailment", "Increase")
                plot["source"] = np.select(
                    [
                        plot["series"].str.startswith("model"),
                        plot["series"].str.startswith("boav"),
                    ],
                    ["Model", "ELEXON BOAV"],
                    default="BOALF endpoint method",
                )
                fig = px.bar(
                    plot,
                    x="carrier",
                    y="GWh",
                    color="source",
                    facet_col="direction",
                    barmode="group",
                    color_discrete_map={"Model": "#4C78A8", "ELEXON BOAV": "#F58518", "BOALF endpoint method": "#9D9DA1"},
                    title="Wind BM Actions Detail - 2020",
                    labels={"carrier": "", "source": ""},
                    template=plot_template,
                    height=390,
                )
                fig.update_yaxes(matches=None)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                return fig
            """
        ),
        code("plot_bm_carrier_volumes().show()"),
        code(
            """
            wind_detail = wind_bm_detail_table()
            display(
                wind_detail[
                    [
                        "carrier",
                        "model_decrease_gwh",
                        "boav_decrease_gwh",
                        "boalf_endpoint_decrease_gwh",
                        "model_increase_gwh",
                        "boav_increase_gwh",
                        "boalf_endpoint_increase_gwh",
                    ]
                ].round(1)
            )
            plot_wind_bm_detail().show()
            """
        ),
        markdown(
            """
            **BM volume interpretation.** Balancing volumes are the largest validation gap. Using ELEXON BOAV, the model captures a meaningful share of real accepted decrease volume, but materially under-represents wind bids/curtailment. The BOALF endpoint method shown in the wind diagnostic is the previous level-delta approximation; it is retained only to show why the old benchmark was misleading.
            """
        ),
        markdown(
            """
            ### 4.2 BM Cost Against NESO Constraint Cost

            The model value is the balancing-stage redispatch cost. The benchmark is NESO real thermal constraint cost.
            """
        ),
        code(
            """
            fig = go.Figure()
            fig.add_bar(
                x=score["period"].astype(str),
                y=score["bm_cost_gbp_m"],
                name="Model BM cost",
                marker_color="#345995",
            )
            fig.add_bar(
                x=score["period"].astype(str),
                y=score["neso_thermal_cost_gbp_m"],
                name="NESO thermal constraint cost",
                marker_color="#E26D5A",
            )
            fig.update_layout(title="Monthly BM Cost Compared With NESO Real Thermal Constraint Cost", barmode="group")
            fig.update_yaxes(title_text="GBP million")
            fig.update_xaxes(title_text="")
            style_plot(fig, height=430).show()
            """
        ),
        markdown(
            """
            ### 4.3 NESO Outages and Boundary Flows

            This view compares NESO real day-ahead boundary limits and flows with model balancing-stage boundary flows for key Scottish export boundaries.
            """
        ),
        code(
            """
            def _boundary_model_flow(network, mapping, boundary):
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


            def _annual_boundary_model_flows(mapping, selected):
                flows = {boundary: [] for boundary in selected}
                for scenario in MONTH_SCENARIOS:
                    path = MARKET / f"{scenario}_balancing.nc"
                    if not path.exists():
                        continue
                    network = pypsa.Network(path)
                    for boundary in selected:
                        flows[boundary].append(_boundary_model_flow(network, mapping, boundary))
                return {
                    boundary: pd.concat(parts).sort_index()
                    for boundary, parts in flows.items()
                    if parts
                }


            def plot_neso_boundary_case_study(selected=("SCOTEX", "SSE-SP", "SSHARN")):
                with open(DATA / "network" / "neso_boundary_mapping.yaml", "r", encoding="utf-8") as fh:
                    mapping = yaml.safe_load(fh)

                neso = pd.read_csv(DATA / "validation" / "day_ahead_constraint_flows_limits.csv")
                neso["datetime"] = pd.to_datetime(neso["Date (GMT/BST)"], errors="coerce")
                neso = neso[(neso["datetime"] >= "2020-01-01") & (neso["datetime"] < "2021-01-01")]
                neso["hour"] = neso["datetime"].dt.floor("h")
                model_flows = _annual_boundary_model_flows(mapping, selected)

                fig = make_subplots(
                    rows=len(selected),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=selected,
                )
                for row, boundary in enumerate(selected, start=1):
                    nb = neso[neso["Constraint Group"] == boundary]
                    hourly = nb.groupby("hour")[["Limit (MW)", "Flow (MW)"]].mean()
                    model_flow = model_flows.get(boundary, pd.Series(dtype=float))
                    model_flow = model_flow.reindex(hourly.index).interpolate(limit_direction="both")
                    limit_gw = (hourly["Limit (MW)"] / 1000.0).where(lambda s: s < 20.0)
                    flow_gw = hourly["Flow (MW)"] / 1000.0
                    model_gw = model_flow / 1000.0

                    fig.add_scatter(
                        row=row,
                        col=1,
                        x=hourly.index,
                        y=limit_gw,
                        mode="lines",
                        name="NESO DA limit",
                        line=dict(color="#9EC5F8", width=0),
                        fill="tozeroy",
                        fillcolor="rgba(158,197,248,0.45)",
                        showlegend=row == 1,
                    )
                    fig.add_scatter(
                        row=row,
                        col=1,
                        x=hourly.index,
                        y=flow_gw,
                        mode="lines",
                        name="NESO DA flow",
                        line=dict(color="#F58518", width=1.4),
                        showlegend=row == 1,
                    )
                    fig.add_scatter(
                        row=row,
                        col=1,
                        x=model_gw.index,
                        y=model_gw,
                        mode="lines",
                        name="Model BM flow",
                        line=dict(color="#1F4E79", width=1.8),
                        showlegend=row == 1,
                    )
                    constrained = hourly["Limit (MW)"] < 50000
                    util = (
                        hourly.loc[constrained, "Flow (MW)"]
                        / hourly.loc[constrained, "Limit (MW)"]
                    ).replace([np.inf, -np.inf], np.nan)
                    fig.add_annotation(
                        row=row,
                        col=1,
                        text=f"NESO >=90% limit: {(util >= 0.9).mean() * 100:.0f}% of constrained periods",
                        xref=f"x{row if row > 1 else ''} domain",
                        yref=f"y{row if row > 1 else ''} domain",
                        x=0.99,
                        y=0.88,
                        showarrow=False,
                        align="right",
                        font=dict(size=10, color="#333333"),
                    )
                    fig.update_yaxes(row=row, col=1, title_text=f"{boundary}<br>GW")

                fig.update_layout(title="NESO Real Boundary Limits and Flows vs Model Boundary Flows - 2020")
                fig.update_xaxes(row=len(selected), col=1, title_text="2020")
                fig.update_xaxes(range=[pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")])
                return style_plot(fig, height=760)
            """
        ),
        code("plot_neso_boundary_case_study().show()"),
        code(
            """
            top_boundaries = boundaries.sort_values("neso_cost_gbp", ascending=False).head(18).copy()
            top_boundaries["label"] = top_boundaries["period"].astype(str) + " - " + top_boundaries["boundary"]
            fig = px.bar(
                top_boundaries.sort_values("neso_cost_gbp"),
                x="neso_cost_gbp",
                y="label",
                color="flow_ratio",
                orientation="h",
                color_continuous_scale="RdBu",
                range_color=[0.5, 1.5],
                labels={
                    "neso_cost_gbp": "NESO boundary cost (GBP)",
                    "label": "",
                    "flow_ratio": "Model / NESO mean flow",
                },
                title="Highest-Cost NESO Real Boundaries and Model Flow Alignment",
                template=plot_template,
                height=560,
            )
            fig.update_layout(coloraxis_colorbar=dict(title="Flow ratio"))
            fig.update_xaxes(tickprefix="GBP ", separatethousands=True)
            fig.show()
            """
        ),
        markdown(
            """
            **Network and BM interpretation.** The model captures the main constrained regions, especially SSHARN, SCOTEX, and SSE-SP. The residual issue is timing and duration of high utilisation: several high-cost SSHARN months show lower model hours above 90% than NESO, consistent with remaining outage-limit or sparse-gate mapping gaps.
            """
        ),
        markdown(
            """
            ## 5. Appendix: Scorecard Table

            The compact table below is suitable for copying into a report appendix.
            """
        ),
        code(
            """
            display_cols = [
                "period",
                "bm_cost_gbp_m",
                "neso_thermal_cost_gbp_m",
                "bm_vs_neso_thermal_ratio",
                "mean_smp_gbp_mwh",
                "mean_mid_gbp_mwh",
                "smp_mid_ratio",
            ]
            boav_score = (
                boav_monthly
                .groupby("period", as_index=False, observed=True)[
                    ["model_increase_mwh", "model_decrease_mwh", "boav_increase_mwh", "boav_decrease_mwh"]
                ]
                .sum()
            )
            boav_score["model_boav_increase_ratio"] = boav_score["model_increase_mwh"] / boav_score["boav_increase_mwh"].replace(0, np.nan)
            boav_score["model_boav_decrease_ratio"] = boav_score["model_decrease_mwh"] / boav_score["boav_decrease_mwh"].replace(0, np.nan)

            table = score[display_cols].merge(
                boav_score[["period", "model_boav_increase_ratio", "model_boav_decrease_ratio"]],
                on="period",
                how="left",
            ).rename(
                columns={
                    "period": "Month",
                    "bm_cost_gbp_m": "Model BM cost (GBPm)",
                    "neso_thermal_cost_gbp_m": "NESO thermal cost (GBPm)",
                    "bm_vs_neso_thermal_ratio": "Model / NESO cost",
                    "mean_smp_gbp_mwh": "Model SMP (GBP/MWh)",
                    "mean_mid_gbp_mwh": "ELEXON MID (GBP/MWh)",
                    "smp_mid_ratio": "SMP / MID",
                    "model_boav_increase_ratio": "Increase vol ratio vs BOAV",
                    "model_boav_decrease_ratio": "Decrease vol ratio vs BOAV",
                }
            )
            display(table.round(2))
            """
        ),
        markdown(
            """
            ## Conclusions

            - The wholesale market price signal is credible at annual scale.
            - Final dispatch is strongest for demand, nuclear, wind, solar, and CCGT.
            - Boundary-cost alignment is directionally useful, but high-cost northern boundaries still need closer outage and limit calibration.
            - BM volumes are materially under-represented versus ELEXON BOAV accepted bid/offer volumes, especially wind bid/decrease volume.
            - Coal and pumped hydro are the clearest technology-level dispatch gaps.

            These results support using the current configuration for directional market and network analysis, with caution around absolute BM volume and technology-level balancing conclusions.
            """
        ),
    ]

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
