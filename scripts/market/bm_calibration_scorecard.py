"""
BM Calibration Scorecard — Per-Carrier Model-vs-ELEXON Price & Volume Audit

Produces a single per-carrier table that compares:
  1. Model effective offer/bid price (from redispatch_summary)
     vs ELEXON median offer/bid (from raw BOD data, matched BMUs only).
  2. Model redispatch volumes (increase/decrease MWh)
     vs BOALF **unflagged** acceptance volumes
     (unflagged = constraint management; flagged includes reserve/response).

Outputs a CSV scorecard and an HTML dashboard. Optionally fails the rule
when any carrier with non-trivial volume drifts outside a configured
tolerance band — set ``market.balancing.calibration.fail_on_regression: true``
in the scenario config to enable.

See also:
  - scripts/market/validate_bm.py — upstream BOALF/B1610 comparison
  - scripts/market/market_utils.py — bid/offer price calculation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

try:
    import pypsa
except ImportError:
    pypsa = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from scripts.utilities.logging_config import setup_logging


# Carriers we watch closely when deciding whether calibration has regressed.
# Drift beyond the tolerance on any of these triggers a warning (or error
# when fail_on_regression is true).
PRIORITY_CARRIERS = (
    "CCGT",
    "OCGT",
    "coal",
    "nuclear",
    "biomass",
    "wind_offshore",
    "wind_onshore",
    "large_hydro",
    "Pumped Storage Hydroelectricity",
    "pumped_hydro",
)


def _load_elexon_prices(
    elexon_offers_path: Path,
    elexon_bids_path: Path,
    bmu_mapping_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load ELEXON offer/bid CSVs and map BMU columns to generator names."""
    offers = pd.read_csv(elexon_offers_path, index_col=0)
    bids = pd.read_csv(elexon_bids_path, index_col=0)
    offers.index = pd.to_datetime(offers.index)
    bids.index = pd.to_datetime(bids.index)

    bmu_map = pd.read_csv(bmu_mapping_path)
    bmu_to_gen = dict(zip(bmu_map["bmu_id"], bmu_map["generator_name"]))

    offers_mapped = offers.rename(columns=bmu_to_gen)
    bids_mapped = bids.rename(columns=bmu_to_gen)
    if offers_mapped.columns.has_duplicates:
        offers_mapped = offers_mapped.T.groupby(level=0).mean().T
    if bids_mapped.columns.has_duplicates:
        bids_mapped = bids_mapped.T.groupby(level=0).mean().T

    return offers_mapped, bids_mapped, bmu_to_gen


def _build_scorecard(
    network,
    redispatch: pd.DataFrame,
    boalf_by_flag: pd.DataFrame,
    offers_mapped: pd.DataFrame,
    bids_mapped: pd.DataFrame,
    mapped_gen_names: set,
) -> pd.DataFrame:
    """Assemble per-carrier comparison table."""
    gens = network.generators

    # Model: per-carrier volumes and costs
    model_by_carrier = (
        redispatch.groupby("carrier")
        .agg(
            increase_MWh=("increase_MWh", "sum"),
            decrease_MWh=("decrease_MWh", "sum"),
            offer_cost=("offer_cost", "sum"),
            bid_cost=("bid_cost", "sum"),
        )
    )
    model_by_carrier["eff_offer_model"] = (
        model_by_carrier["offer_cost"]
        / model_by_carrier["increase_MWh"].replace(0, np.nan)
    )
    model_by_carrier["eff_bid_model"] = (
        model_by_carrier["bid_cost"]
        / model_by_carrier["decrease_MWh"].replace(0, np.nan)
    )

    # BOALF unflagged (constraint management only, reserve/response stripped)
    boalf_uf = boalf_by_flag[
        (boalf_by_flag["scope"] == "carrier")
        & (boalf_by_flag["group"] == "unflagged")
    ].set_index("carrier")
    boalf_inc = boalf_uf["increase_mwh"] if "increase_mwh" in boalf_uf.columns else pd.Series(dtype=float)
    boalf_dec = boalf_uf["decrease_mwh"] if "decrease_mwh" in boalf_uf.columns else pd.Series(dtype=float)

    rows = []
    for carrier in sorted(gens["carrier"].unique()):
        car_gens = gens[gens["carrier"] == carrier]
        matched_gens = car_gens.index[car_gens.index.isin(mapped_gen_names)]

        matched_in_offers = [g for g in matched_gens if g in offers_mapped.columns]
        matched_in_bids = [g for g in matched_gens if g in bids_mapped.columns]

        # ELEXON reference: median across BMUs of per-BMU mean price.
        # Negate bid to ESO-cost convention (positive = costs ESO to decrease).
        if matched_in_offers:
            offer_per_gen = offers_mapped[matched_in_offers].mean(axis=0)
            median_offer_elexon = float(offer_per_gen.median())
        else:
            median_offer_elexon = np.nan
        if matched_in_bids:
            bid_per_gen = -bids_mapped[matched_in_bids].mean(axis=0)
            median_bid_elexon = float(bid_per_gen.median())
        else:
            median_bid_elexon = np.nan

        model_row = model_by_carrier.reindex([carrier]).iloc[0] if carrier in model_by_carrier.index else None
        eff_offer_model = float(model_row["eff_offer_model"]) if model_row is not None else np.nan
        eff_bid_model = float(model_row["eff_bid_model"]) if model_row is not None else np.nan
        inc_mwh_model = float(model_row["increase_MWh"]) if model_row is not None else 0.0
        dec_mwh_model = float(model_row["decrease_MWh"]) if model_row is not None else 0.0

        # BOALF uses BMU-inferred carriers, which match PyPSA names exactly for
        # the carriers in BMU_PREFIX_FUEL. Others (embedded_*, small_hydro,
        # waste_to_energy, etc.) have no BOALF counterpart — left as NaN.
        boalf_inc_mwh = float(boalf_inc.get(carrier, np.nan))
        boalf_dec_mwh = float(boalf_dec.get(carrier, np.nan))

        def _ratio(num, denom):
            if denom is None or not np.isfinite(denom) or denom == 0:
                return np.nan
            if num is None or not np.isfinite(num):
                return np.nan
            return num / denom

        rows.append({
            "carrier": carrier,
            "n_gens": int(len(car_gens)),
            "n_matched": int(len(matched_gens)),
            "capacity_MW": float(car_gens["p_nom"].sum()),
            "capacity_matched_MW": float(car_gens.loc[matched_gens, "p_nom"].sum())
                if len(matched_gens) else 0.0,
            "median_offer_elexon": median_offer_elexon,
            "median_bid_elexon": median_bid_elexon,
            "eff_offer_model": eff_offer_model,
            "eff_bid_model": eff_bid_model,
            "offer_ratio": _ratio(eff_offer_model, median_offer_elexon),
            "bid_ratio": _ratio(eff_bid_model, median_bid_elexon),
            "model_inc_MWh": inc_mwh_model,
            "model_dec_MWh": dec_mwh_model,
            "boalf_uf_inc_MWh": boalf_inc_mwh,
            "boalf_uf_dec_MWh": boalf_dec_mwh,
            "inc_ratio": _ratio(inc_mwh_model, boalf_inc_mwh),
            "dec_ratio": _ratio(dec_mwh_model, boalf_dec_mwh),
        })

    return pd.DataFrame(rows).sort_values("capacity_MW", ascending=False)


def _flag_regressions(
    scorecard: pd.DataFrame,
    tolerance: float,
    min_volume_mwh: float,
    priority: tuple = PRIORITY_CARRIERS,
) -> pd.DataFrame:
    """Return rows that drift outside tolerance on any monitored ratio.

    Only carriers with model volume ≥ ``min_volume_mwh`` on the relevant side
    are checked — a carrier with zero BM activity can't regress meaningfully.
    """
    issues = []
    for _, row in scorecard.iterrows():
        car = row["carrier"]
        if car not in priority:
            continue

        for name, model_mwh_col, ratio_col, side in [
            ("offer", "model_inc_MWh", "offer_ratio", "price"),
            ("bid", "model_dec_MWh", "bid_ratio", "price"),
            ("inc_volume", "model_inc_MWh", "inc_ratio", "volume"),
            ("dec_volume", "model_dec_MWh", "dec_ratio", "volume"),
        ]:
            ratio = row[ratio_col]
            if not np.isfinite(ratio):
                continue
            if row[model_mwh_col] < min_volume_mwh:
                continue
            if abs(ratio - 1.0) > tolerance:
                issues.append({
                    "carrier": car,
                    "metric": name,
                    "side": side,
                    "ratio": ratio,
                    "deviation": ratio - 1.0,
                    "model_volume_MWh": row[model_mwh_col],
                })
    return pd.DataFrame(issues)


def _build_dashboard(
    scorecard: pd.DataFrame,
    issues: pd.DataFrame,
    scenario_id: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Write an HTML dashboard with carrier-level ratio bars."""
    if not HAS_PLOTLY:
        logger.warning("plotly not installed — skipping dashboard HTML")
        output_path.write_text(
            f"<html><body><h1>plotly unavailable — see "
            f"{scenario_id}_bm_calibration.csv</h1></body></html>"
        )
        return

    # Drop carriers with no model volume for readability
    disp = scorecard[
        (scorecard["model_inc_MWh"] > 0) | (scorecard["model_dec_MWh"] > 0)
    ].copy()
    disp = disp.sort_values("capacity_MW", ascending=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Offer price ratio (model / ELEXON median)",
            "Bid price ratio (model / ELEXON median)",
            "Increase volume ratio (model / BOALF unflagged)",
            "Decrease volume ratio (model / BOALF unflagged)",
        ],
        horizontal_spacing=0.14,
        vertical_spacing=0.16,
    )

    def _trace(col, y_col, label, row, col_idx):
        vals = disp[y_col].replace([np.inf, -np.inf], np.nan)
        colors = [
            "#d9534f" if (np.isfinite(v) and abs(v - 1.0) > 0.2)
            else "#f0ad4e" if (np.isfinite(v) and abs(v - 1.0) > 0.1)
            else "#5cb85c"
            for v in vals
        ]
        fig.add_trace(
            go.Bar(
                x=vals,
                y=disp["carrier"],
                orientation="h",
                marker_color=colors,
                name=label,
                showlegend=False,
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ),
            row=row, col=col_idx,
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", row=row, col=col_idx)

    _trace("offer_ratio", "offer_ratio", "offer", 1, 1)
    _trace("bid_ratio", "bid_ratio", "bid", 1, 2)
    _trace("inc_ratio", "inc_ratio", "inc", 2, 1)
    _trace("dec_ratio", "dec_ratio", "dec", 2, 2)

    fig.update_layout(
        title=f"BM Calibration Scorecard — {scenario_id}",
        height=900,
        template="plotly_white",
    )

    issue_html = ""
    if not issues.empty:
        issue_html = "<h2>Regression flags</h2>" + issues.to_html(
            index=False, float_format=lambda v: f"{v:.2f}"
        )

    html = (
        "<html><head><title>BM Calibration — " + scenario_id + "</title></head><body>"
        + fig.to_html(include_plotlyjs="cdn", full_html=False)
        + issue_html
        + "</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")


def run_scorecard(
    scenario_id: str,
    redispatch_csv: Path,
    boalf_by_flag_csv: Path,
    network_path: Path,
    elexon_offers_path: Path,
    elexon_bids_path: Path,
    bmu_mapping_path: Path,
    output_csv: Path,
    output_html: Path,
    tolerance: float,
    min_volume_mwh: float,
    fail_on_regression: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    """End-to-end scorecard build. Raises on regression when requested."""
    if pypsa is None:
        raise ImportError("pypsa is required for the calibration scorecard")

    network = pypsa.Network(str(network_path))
    redispatch = pd.read_csv(redispatch_csv)
    boalf_by_flag = pd.read_csv(boalf_by_flag_csv) if boalf_by_flag_csv.exists() else pd.DataFrame()

    if boalf_by_flag.empty:
        logger.warning(
            f"BOALF-by-flag CSV is empty or missing ({boalf_by_flag_csv}); "
            "volume ratios will be NaN."
        )

    offers_mapped, bids_mapped, bmu_to_gen = _load_elexon_prices(
        elexon_offers_path, elexon_bids_path, bmu_mapping_path
    )
    mapped_gen_names = set(bmu_to_gen.values())

    scorecard = _build_scorecard(
        network, redispatch, boalf_by_flag,
        offers_mapped, bids_mapped, mapped_gen_names,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(output_csv, index=False)
    logger.info(f"Wrote calibration scorecard: {output_csv}")

    issues = _flag_regressions(scorecard, tolerance, min_volume_mwh)
    if not issues.empty:
        logger.warning(
            f"Calibration regressions detected for {issues['carrier'].nunique()} "
            f"priority carrier(s) at tolerance {tolerance:.2f}:"
        )
        for _, issue in issues.iterrows():
            logger.warning(
                f"  {issue['carrier']:30s} {issue['metric']:12s} "
                f"ratio={issue['ratio']:.2f} "
                f"(model {issue['model_volume_MWh']:,.0f} MWh)"
            )
    else:
        logger.info(
            f"No calibration regressions above tolerance {tolerance:.2f} "
            f"(min volume {min_volume_mwh:,.0f} MWh)."
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    _build_dashboard(scorecard, issues, scenario_id, output_html, logger)
    logger.info(f"Wrote calibration dashboard: {output_html}")

    if fail_on_regression and not issues.empty:
        raise RuntimeError(
            f"BM calibration regressed on {issues['carrier'].nunique()} priority "
            f"carrier(s). See {output_csv} and {output_html}."
        )

    return scorecard


if __name__ == "__main__":
    log_path = (
        snakemake.log[0]
        if hasattr(snakemake, "log") and snakemake.log
        else "bm_calibration_scorecard"
    )
    logger = setup_logging(log_path)

    scenario_config = snakemake.params.scenario_config
    scenario_id = snakemake.wildcards.scenario

    calibration_cfg = (
        scenario_config.get("market", {})
        .get("balancing", {})
        .get("calibration", {})
    )
    tolerance = float(calibration_cfg.get("tolerance", 0.20))
    min_volume_mwh = float(calibration_cfg.get("min_volume_mwh", 1000.0))
    fail_on_regression = bool(calibration_cfg.get("fail_on_regression", False))

    run_scorecard(
        scenario_id=scenario_id,
        redispatch_csv=Path(snakemake.input.redispatch_summary_csv),
        boalf_by_flag_csv=Path(snakemake.input.boalf_by_flag_csv),
        network_path=Path(snakemake.input.network),
        elexon_offers_path=Path(snakemake.input.elexon_offers),
        elexon_bids_path=Path(snakemake.input.elexon_bids),
        bmu_mapping_path=Path(snakemake.input.bmu_mapping),
        output_csv=Path(snakemake.output.calibration_csv),
        output_html=Path(snakemake.output.calibration_html),
        tolerance=tolerance,
        min_volume_mwh=min_volume_mwh,
        fail_on_regression=fail_on_regression,
        logger=logger,
    )
