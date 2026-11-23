"""Transparent validation-only anchors for historically committed generation."""

from __future__ import annotations

import pandas as pd


ESPENI_COLUMNS = {
    "coal": "ELEC_POWER_ELEX_COAL[MW](float32)",
}


def apply_historical_dispatch_anchors(network, config: dict, espeni_path: str, logger=None) -> dict:
    """Fix selected carrier totals to metered ESPENI output.

    This bridge is appropriate for backtesting technologies whose historical
    commitment decisions are outside a linear economic-dispatch model.  It is
    deliberately unavailable for future years and its provenance is stored on
    every anchored generator.
    """
    if not config.get("enabled", False):
        return {"anchored_carriers": []}
    data = pd.read_csv(espeni_path, low_memory=False)
    time_col = next(
        column for column in (
            "ELEC_elex_startTime[utc](datetime)",
            "ELEC_startTime[localtime](datetime)",
        ) if column in data.columns
    )
    data.index = pd.to_datetime(data[time_col], errors="coerce").dt.tz_localize(None)
    data = data.loc[data.index.notna()]
    snapshots = pd.DatetimeIndex(network.snapshots).tz_localize(None)
    anchored = []
    for carrier in config.get("carriers", []):
        column = ESPENI_COLUMNS.get(str(carrier))
        gens = network.generators.index[network.generators["carrier"].astype(str).eq(carrier)]
        if not column or column not in data.columns or not len(gens):
            continue
        metered = pd.to_numeric(data[column], errors="coerce").groupby(level=0).mean()
        # ESPENI is half-hourly. Mean resampling preserves MW when the model is hourly.
        resolution = pd.Series(snapshots).diff().dropna().median()
        if resolution >= pd.Timedelta(hours=1):
            metered = metered.resample(resolution).mean()
        metered = metered.reindex(snapshots).interpolate(limit_direction="both").clip(lower=0.0)
        capacity = float(network.generators.loc[gens, "p_nom"].sum())
        pu = (metered / capacity).clip(upper=1.0)
        for name in gens:
            network.generators_t.p_min_pu[name] = pu.values
            network.generators_t.p_max_pu[name] = pu.values
            network.generators.at[name, "historical_dispatch_anchor"] = "ESPENI"
        anchored.append(carrier)
        if logger:
            logger.info(
                "Anchored historical %s dispatch to ESPENI (%.3f TWh over network snapshots)",
                carrier, metered.sum() * (resolution / pd.Timedelta(hours=1)) / 1e6,
            )
    return {"anchored_carriers": anchored}
