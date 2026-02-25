"""
Normalize forecast fuel price curves for dispatch forecasting.

This script converts user-provided commodity forecast CSV files into a
standardized hourly table consumed by apply_marginal_costs.py.

Input expectations:
- At least one timestamp column
- At least one gas price column

Output format:
- snapshot, gas, coal, oil, biomass
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from scripts.utilities.logging_config import setup_logging


TIMESTAMP_CANDIDATES = [
    "snapshot",
    "timestamp",
    "datetime",
    "time",
    "date",
]

FUEL_COLUMN_CANDIDATES = {
    "gas": [
        "gas",
        "gas_price",
        "gas_price_gbp_per_mwh_thermal",
        "natural_gas",
        "nbp",
    ],
    "coal": [
        "coal",
        "coal_price",
        "coal_price_gbp_per_mwh_thermal",
    ],
    "oil": [
        "oil",
        "oil_price",
        "oil_price_gbp_per_mwh_thermal",
    ],
    "biomass": [
        "biomass",
        "biomass_price",
        "biomass_price_gbp_per_mwh_thermal",
    ],
}


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first matching column by exact or case-insensitive match."""
    if not columns:
        return None

    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _normalize_timestamp_index(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Parse, clean, and normalize timestamp index."""
    timestamp_col = _find_column(list(df.columns), TIMESTAMP_CANDIDATES)
    if not timestamp_col:
        raise ValueError(
            "Could not find timestamp column. Expected one of: "
            f"{TIMESTAMP_CANDIDATES}"
        )

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce", utc=True)
    out = out[out[timestamp_col].notna()].copy()

    if len(out) == 0:
        raise ValueError("No valid timestamps found in forecast fuel curve file")

    out[timestamp_col] = out[timestamp_col].dt.tz_convert("UTC").dt.tz_localize(None)
    out = out.set_index(timestamp_col).sort_index()

    # Duplicate timestamps are averaged.
    if out.index.duplicated().any():
        logger.warning("Duplicate timestamps found in fuel curve. Averaging duplicates.")
        out = out.groupby(level=0).mean(numeric_only=True)

    return out


def normalize_curve(
    df: pd.DataFrame,
    logger,
    snapshots: pd.DatetimeIndex | None = None,
    fallback_fuels: dict | None = None,
) -> pd.DataFrame:
    """Normalize fuel curve table to standard columns and optional snapshot grid."""
    fallback_fuels = fallback_fuels or {}
    normalized = _normalize_timestamp_index(df, logger)

    out = pd.DataFrame(index=normalized.index)

    for fuel, candidates in FUEL_COLUMN_CANDIDATES.items():
        col = _find_column(list(normalized.columns), candidates)
        if col:
            out[fuel] = pd.to_numeric(normalized[col], errors="coerce")

    if "gas" not in out.columns:
        raise ValueError(
            "Gas price column is required. Accepted names include: "
            f"{FUEL_COLUMN_CANDIDATES['gas']}"
        )

    # Optional alignment to hourly snapshots.
    if snapshots is not None and len(snapshots) > 0:
        full_index = out.index.union(snapshots).sort_values()
        out = out.reindex(full_index)
        out = out.interpolate(method="time").ffill().bfill()
        out = out.reindex(snapshots)

    # Fill optional fuels from defaults if missing.
    for fuel in ["coal", "oil", "biomass"]:
        if fuel not in out.columns:
            if fuel in fallback_fuels:
                out[fuel] = float(fallback_fuels[fuel])
        else:
            default_val = fallback_fuels.get(fuel)
            if default_val is not None:
                out[fuel] = out[fuel].fillna(float(default_val))

    out["gas"] = out["gas"].astype(float)

    return out


def _build_snapshots_from_forecast_cfg(forecast_cfg: dict, logger) -> pd.DatetimeIndex | None:
    """Build hourly snapshots from forecast.issue_time_utc + horizon_hours if available."""
    issue_time = forecast_cfg.get("issue_time_utc")
    horizon_hours = int(forecast_cfg.get("horizon_hours", 0) or 0)

    if not issue_time or horizon_hours <= 0:
        return None

    issue = pd.to_datetime(issue_time, utc=True, errors="coerce")
    if pd.isna(issue):
        logger.warning(f"Invalid forecast issue_time_utc: {issue_time}. Skipping snapshot alignment.")
        return None

    issue = issue.tz_convert("UTC").tz_localize(None)
    return pd.date_range(issue, periods=horizon_hours, freq="h")


def main() -> None:
    """Snakemake entrypoint."""
    snk = globals().get("snakemake")
    if snk is None:
        raise RuntimeError("This script must be run via Snakemake")

    logger = setup_logging(snk.log[0] if snk.log else "prepare_forecast_fuel_prices")

    input_path = Path(snk.input.raw_curve)
    output_path = Path(snk.output.forecast_fuel_prices)
    scenario_config = dict(getattr(snk.params, "scenario_config", {}) or {})

    if not input_path.exists():
        raise FileNotFoundError(f"Forecast gas curve file not found: {input_path}")

    mc_cfg = scenario_config.get("marginal_costs", {}) if isinstance(scenario_config, dict) else {}
    fallback_fuels = mc_cfg.get("fuel_prices", {}) if isinstance(mc_cfg, dict) else {}

    forecast_cfg = scenario_config.get("forecast", {}) if isinstance(scenario_config, dict) else {}
    snapshots = _build_snapshots_from_forecast_cfg(forecast_cfg, logger)

    logger.info(f"Loading forecast fuel curve: {input_path}")
    df = pd.read_csv(input_path)

    normalized = normalize_curve(
        df,
        logger=logger,
        snapshots=snapshots,
        fallback_fuels=fallback_fuels,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalized.reset_index().rename(columns={normalized.index.name or "index": "snapshot"})
    normalized.to_csv(output_path, index=False)

    logger.info(f"Saved normalized forecast fuel prices: {output_path}")
    logger.info(f"Rows: {len(normalized)}, Columns: {list(normalized.columns)}")


if __name__ == "__main__":
    main()
