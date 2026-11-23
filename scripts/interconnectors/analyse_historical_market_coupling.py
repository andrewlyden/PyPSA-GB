"""Compare market-coupled historical interconnector flows with ESPENI.

This is intentionally a post-solve diagnostic.  It never feeds observed flows
back into the optimisation: the model selects flows from domestic dispatch,
actual ENTSO-E prices and commissioned cable capacities.  ESPENI is used only
as an independent reference.

``wholesale_links.csv`` uses the signed Link convention: positive ``p0`` is an
import to Great Britain. The output normalises both series to MW, where positive
means an import into Great Britain.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# Names used by the mapped-interconnector resource and the ESPENI reference.
ESPENI_NAME_MAPPING: Dict[str, str] = {
    "Britned": "BritNed",
    "IFA Interconnector": "IFA",
    "East West Interconnector": "EastWest",
    "Nemo Link": "Nemo",
    "NS Link": "NorthSeaLink",
    "North Sea Link": "NorthSeaLink",
    "Viking Link": "VikingLink",
    "ElecLink": "ElecLink",
    "Moyle": "Moyle",
    "Greenlink": "Greenlink",
}


def _hours(index: pd.DatetimeIndex) -> pd.Series:
    """Return snapshot duration in hours, including a robust final interval."""
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    deltas = pd.Series(index, index=index).diff().dt.total_seconds().div(3600)
    fallback = float(deltas.dropna().median()) if deltas.notna().any() else 1.0
    return deltas.shift(-1).fillna(fallback).astype(float)


def _reference_column(name: str, columns: pd.Index) -> str | None:
    """Resolve one mapped cable name to an ESPENI column, case-insensitively."""
    candidate = ESPENI_NAME_MAPPING.get(name, name)
    lookup = {str(column).lower(): str(column) for column in columns}
    return lookup.get(str(candidate).lower())


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return np.nan
    return float(left.corr(right))


def compare_flows(
    wholesale_links: pd.DataFrame,
    historical_flows: pd.DataFrame,
    interconnectors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-cable metrics and aligned model/reference time series."""
    wholesale_links = wholesale_links.copy()
    historical_flows = historical_flows.copy()
    wholesale_links.index = pd.DatetimeIndex(wholesale_links.index).tz_localize(None)
    historical_flows.index = pd.DatetimeIndex(historical_flows.index).tz_localize(None)

    metrics = []
    aligned_parts = []
    for cable in interconnectors.itertuples(index=False):
        name = str(cable.name)
        model_column = f"IC_{name}"
        reference_column = _reference_column(name, historical_flows.columns)
        if model_column not in wholesale_links.columns or reference_column is None:
            metrics.append(
                {
                    "interconnector": name,
                    "status": "missing_model_or_reference_series",
                    "model_column": model_column,
                    "reference_column": reference_column,
                }
            )
            continue

        # Signed direct-Link p0 > 0 is a physical import to GB; ESPENI positive
        # is also an import. The older fixed-flow implementation used a different
        # p_set convention and is not a sign-reference for this comparison.
        frame = pd.DataFrame(
            {
                "modelled_import_mw": pd.to_numeric(
                    wholesale_links[model_column], errors="coerce"
                ),
                "observed_import_mw": pd.to_numeric(
                    historical_flows[reference_column], errors="coerce"
                ),
            }
        ).dropna()
        if frame.empty:
            metrics.append(
                {
                    "interconnector": name,
                    "status": "no_overlapping_observations",
                    "model_column": model_column,
                    "reference_column": reference_column,
                }
            )
            continue

        duration = _hours(frame.index)
        error = frame["modelled_import_mw"] - frame["observed_import_mw"]
        active = (frame[["modelled_import_mw", "observed_import_mw"]].abs().max(axis=1) >= 10.0)
        sign_agreement = pd.Series(np.nan, index=frame.index, dtype=float)
        sign_agreement.loc[active] = (
            np.sign(frame.loc[active, "modelled_import_mw"])
            == np.sign(frame.loc[active, "observed_import_mw"])
        ).astype(float)
        capacity_mw = float(getattr(cable, "capacity_mw", np.nan))
        metrics.append(
            {
                "interconnector": name,
                "status": "compared",
                "model_column": model_column,
                "reference_column": reference_column,
                "hours_compared": len(frame),
                "capacity_mw": capacity_mw,
                "modelled_import_mwh": float((frame["modelled_import_mw"] * duration).sum()),
                "observed_import_mwh": float((frame["observed_import_mw"] * duration).sum()),
                "mean_error_mw": float(error.mean()),
                "mae_mw": float(error.abs().mean()),
                "rmse_mw": float(np.sqrt((error.pow(2)).mean())),
                "correlation": _safe_correlation(
                    frame["modelled_import_mw"], frame["observed_import_mw"]
                ),
                "sign_agreement_share": float(np.nanmean(sign_agreement))
                if np.isfinite(sign_agreement).any()
                else np.nan,
            }
        )
        frame = frame.assign(
            interconnector=name,
            error_mw=error,
            capacity_mw=capacity_mw,
        )
        aligned_parts.append(frame.reset_index(names="snapshot"))

    summary = pd.DataFrame(metrics).sort_values("interconnector")
    timeseries = (
        pd.concat(aligned_parts, ignore_index=True)
        if aligned_parts
        else pd.DataFrame(
            columns=[
                "snapshot",
                "modelled_import_mw",
                "observed_import_mw",
                "interconnector",
                "error_mw",
                "capacity_mw",
            ]
        )
    )
    return summary, timeseries


def main(snakemake) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    wholesale_links = pd.read_csv(
        snakemake.input.wholesale_links, index_col="snapshot", parse_dates=True
    )
    historical_flows = pd.read_csv(
        snakemake.input.historical_flows, index_col=0, parse_dates=True
    )
    interconnectors = pd.read_csv(snakemake.input.interconnectors)
    summary, timeseries = compare_flows(
        wholesale_links, historical_flows, interconnectors
    )

    Path(snakemake.output.summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(snakemake.output.summary, index=False)
    timeseries.to_csv(snakemake.output.timeseries, index=False)
    compared = int((summary["status"] == "compared").sum()) if not summary.empty else 0
    logging.info("Wrote %d cable comparisons to %s", compared, snakemake.output.summary)


if __name__ == "__main__":
    main(snakemake)
