"""Build and apply empirical future balancing bid/offer residuals.

Historic ELEXON prices are not copied into future scenarios.  Instead this
module estimates carrier-level residuals in the model's ESO-cost convention:

    historic bid/offer price - historic marginal cost.

At runtime those residuals are added to the target network's own marginal
costs, which already reflect CP30 or FES commodity and support assumptions.
This makes the method source-agnostic for future fleet data while keeping the
historical evidence traceable.
"""

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pypsa


PARAMETER_COLUMNS = [
    "carrier",
    "n_observations",
    "historical_marginal_cost_median",
    "offer_residual_low",
    "offer_residual_central",
    "offer_residual_high",
    "bid_residual_low",
    "bid_residual_central",
    "bid_residual_high",
]


def _component_marginal_cost(network: pypsa.Network, name: str) -> float:
    """Return a robust historical marginal-cost reference for one generator."""
    static = pd.to_numeric(
        pd.Series([network.generators.at[name, "marginal_cost"]]), errors="coerce"
    ).iloc[0]
    if np.isfinite(static):
        return float(static)

    time_series = getattr(network.generators_t, "marginal_cost", pd.DataFrame())
    if name in time_series.columns:
        values = pd.to_numeric(time_series[name], errors="coerce").dropna()
        if not values.empty:
            return float(values.median())
    return np.nan


def _robust_bmu_price(frame: pd.DataFrame, bmu_id: str, max_abs_price: float) -> float:
    """Return a BMU's robust non-zero price, excluding defensive sentinels."""
    if bmu_id not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[bmu_id], errors="coerce")
    values = values[(values != 0.0) & (values.abs() <= max_abs_price)].dropna()
    return float(values.median()) if not values.empty else np.nan


def _calibration_records(
    network_path: str | Path,
    offers_path: str | Path,
    bids_path: str | Path,
    mapping_path: str | Path,
    max_abs_price: float,
) -> pd.DataFrame:
    """Extract directly mapped historic prices and marginal-cost residuals."""
    network = pypsa.Network(str(network_path))
    offers = pd.read_csv(offers_path)
    bids = pd.read_csv(bids_path)
    mapping = pd.read_csv(mapping_path)
    required = {"bmu_id", "generator_name", "carrier"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"BMU mapping is missing required columns: {sorted(missing)}")

    records = []
    for row in mapping[["bmu_id", "generator_name", "carrier"]].dropna().drop_duplicates().itertuples(index=False):
        bmu_id, generator_name, carrier = map(str, row)
        if generator_name not in network.generators.index:
            continue
        marginal_cost = _component_marginal_cost(network, generator_name)
        if not np.isfinite(marginal_cost):
            continue
        offer = _robust_bmu_price(offers, bmu_id, max_abs_price)
        raw_bid = _robust_bmu_price(bids, bmu_id, max_abs_price)
        # ELEXON BOD bids use the opposite sign to the model's ESO-cost
        # convention.  Convert before calculating an economic residual.
        bid = -raw_bid if np.isfinite(raw_bid) else np.nan
        if not (np.isfinite(offer) or np.isfinite(bid)):
            continue
        records.append(
            {
                "carrier": carrier,
                "generator": generator_name,
                "historical_marginal_cost": marginal_cost,
                "offer_residual": offer - marginal_cost if np.isfinite(offer) else np.nan,
                "bid_residual": bid - marginal_cost if np.isfinite(bid) else np.nan,
            }
        )
    return pd.DataFrame(records)


def calibrate_future_bid_stack(
    calibration_inputs: Iterable[tuple[str | Path, str | Path, str | Path, str | Path]],
    max_abs_price: float = 2_000.0,
    quantiles: tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate carrier residual quantiles from one or more historic periods.

    Returns a compact parameter table and the underlying direct-BMU records for
    audit.  A carrier with no mapped historic evidence is intentionally absent;
    the future solve then retains its normal derived price.
    """
    records = [
        _calibration_records(*inputs, max_abs_price=max_abs_price)
        for inputs in calibration_inputs
    ]
    records = [frame for frame in records if not frame.empty]
    if not records:
        raise ValueError("No usable directly mapped ELEXON records for future bid-stack calibration")
    observations = pd.concat(records, ignore_index=True)

    rows = []
    for carrier, group in observations.groupby("carrier", sort=True):
        offer = group["offer_residual"].dropna()
        bid = group["bid_residual"].dropna()
        n_observations = int(max(len(offer), len(bid)))
        if n_observations == 0:
            continue
        rows.append(
            {
                "carrier": carrier,
                "n_observations": n_observations,
                "historical_marginal_cost_median": float(group["historical_marginal_cost"].median()),
                "offer_residual_low": float(offer.quantile(quantiles[0])) if not offer.empty else np.nan,
                "offer_residual_central": float(offer.quantile(quantiles[1])) if not offer.empty else np.nan,
                "offer_residual_high": float(offer.quantile(quantiles[2])) if not offer.empty else np.nan,
                "bid_residual_low": float(bid.quantile(quantiles[0])) if not bid.empty else np.nan,
                "bid_residual_central": float(bid.quantile(quantiles[1])) if not bid.empty else np.nan,
                "bid_residual_high": float(bid.quantile(quantiles[2])) if not bid.empty else np.nan,
            }
        )
    parameters = pd.DataFrame(rows, columns=PARAMETER_COLUMNS)
    if parameters.empty:
        raise ValueError("Historic ELEXON calibration produced no carrier parameters")
    return parameters, observations


def _target_marginal_cost(components: pd.DataFrame) -> pd.Series:
    """Return static marginal costs aligned to a future component index."""
    return pd.to_numeric(components.get("marginal_cost", pd.Series(index=components.index)), errors="coerce").fillna(0.0)


def apply_future_bid_stack(
    network: pypsa.Network,
    balancing: dict,
    derived_gen_offer: pd.Series,
    derived_gen_bid: pd.Series,
    derived_su_offer: pd.Series,
    derived_su_bid: pd.Series,
    logger: logging.Logger,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Re-anchor empirical residuals to a CP30 or FES future network.

    The configured case selects low/central/high quantiles.  Explicit absolute
    carrier overrides continue to win, and unsupported carriers retain their
    existing derived bid/offer prices.  Storage remains derived in this first
    implementation because its opportunity cost depends on SoC and price
    outlook, not solely a generator-like marginal cost.
    """
    cfg = balancing.get("future_bid_stack", {}) or {}
    parameter_file = Path(cfg.get("parameters_file", ""))
    if not parameter_file.exists():
        raise FileNotFoundError(
            "future_bid_stack requires a calibrated parameters_file; "
            f"not found: {parameter_file}"
        )
    case = str(cfg.get("case", "central")).lower()
    if case not in {"low", "central", "high"}:
        raise ValueError("future_bid_stack.case must be low, central, or high")
    min_samples = int(cfg.get("min_samples", 3))
    offer_floor = float(cfg.get("offer_floor", -1_000.0))
    bid_floor = float(cfg.get("bid_floor", -1_000.0))
    preserve_absolute = bool(cfg.get("preserve_absolute_overrides", True))

    parameters = pd.read_csv(parameter_file).set_index("carrier")
    required = {f"offer_residual_{case}", f"bid_residual_{case}", "n_observations"}
    missing = required.difference(parameters.columns)
    if missing:
        raise ValueError(f"Future bid-stack parameter file missing columns: {sorted(missing)}")

    gen_offer = derived_gen_offer.copy()
    gen_bid = derived_gen_bid.copy()
    marginal_cost = _target_marginal_cost(network.generators)
    overrides = balancing.get("carrier_overrides", {}) or {}
    applied = []
    fallback = []

    for name, component in network.generators.iterrows():
        carrier = str(component.get("carrier", "unknown"))
        override = overrides.get(carrier, {}) or {}
        if preserve_absolute and str(override.get("mode", "")).lower() == "absolute":
            fallback.append(carrier)
            continue
        if carrier not in parameters.index:
            fallback.append(carrier)
            continue
        row = parameters.loc[carrier]
        if int(row["n_observations"]) < min_samples:
            fallback.append(carrier)
            continue
        offer_residual = pd.to_numeric(pd.Series([row[f"offer_residual_{case}"]]), errors="coerce").iloc[0]
        bid_residual = pd.to_numeric(pd.Series([row[f"bid_residual_{case}"]]), errors="coerce").iloc[0]
        if not (np.isfinite(offer_residual) and np.isfinite(bid_residual)):
            fallback.append(carrier)
            continue
        gen_offer.at[name] = max(offer_floor, float(marginal_cost.at[name]) + float(offer_residual))
        gen_bid.at[name] = max(bid_floor, float(marginal_cost.at[name]) + float(bid_residual))
        applied.append(carrier)

    logger.info(
        "Applied empirical future bid stack to %d/%d generators across carriers: %s",
        len(applied),
        len(network.generators),
        ", ".join(sorted(set(applied))) or "none",
    )
    if fallback:
        logger.info(
            "Future bid stack retained derived prices for carriers without sufficient applicable evidence: %s",
            ", ".join(sorted(set(fallback))),
        )
    if len(network.storage_units):
        logger.info("Future bid stack retains derived storage prices (SoC opportunity-cost model pending)")
    return gen_offer, gen_bid, derived_su_offer, derived_su_bid


def main(snakemake) -> None:
    """Snakemake entry point for calibrated parameter and evidence outputs."""
    from scripts.utilities.logging_config import setup_logging

    logger = setup_logging(snakemake.log[0])
    cfg = snakemake.params.future_bid_stack
    inputs = zip(
        snakemake.input.calibration_networks,
        snakemake.input.calibration_offers,
        snakemake.input.calibration_bids,
        snakemake.input.calibration_mappings,
    )
    parameters, observations = calibrate_future_bid_stack(
        list(inputs),
        max_abs_price=float(cfg.get("max_abs_historic_price", 2_000.0)),
        quantiles=tuple(cfg.get("residual_quantiles", [0.25, 0.50, 0.75])),
    )
    parameter_path = Path(snakemake.output.parameters_csv)
    evidence_path = Path(snakemake.output.evidence_csv)
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    parameters.to_csv(parameter_path, index=False)
    observations.to_csv(evidence_path, index=False)
    logger.info("Wrote future bid-stack parameters: %s (%d carriers)", parameter_path, len(parameters))
    logger.info("Wrote future bid-stack evidence: %s (%d observations)", evidence_path, len(observations))


if __name__ == "__main__":
    main(snakemake)