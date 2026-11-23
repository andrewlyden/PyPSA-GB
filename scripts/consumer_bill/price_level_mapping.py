"""The model's 2030 hours priced at CP30's own price level, CP.32 (items F12 and F19, 6 Oct 2026).

The model's 2030 Stage 1 price averages about GBP 46/MWh; CP30's own 2030 price for the
same pathway (Clean Power 2030 Data Workbook chart CP.32, Further Flex and Renewables,
Baringa for NESO) averages GBP 69.9/MWh.  Why is item F13's open diagnosis.  This module
re-prices the model's hours on CP30's curve, a reusable static bracket.  Its one use
today is the legacy valuation's post-2030 price path (lever ``legacy_price_path``,
F19 P3 decision 2); the F12 bill sensitivity was dropped (user, 6 Oct 2026) because the
next re-solve puts the model's 2030 prices on CP30's fuel basis.

* **Quantile mapping.** Every market hour keeps its rank in the model's own price
  duration curve and takes the CP.32 price at that rank.  Hour ``i`` covers the share
  of the year ``[a_i, b_i]`` exceeded by higher-priced hours; its mapped price is the
  mean of the CP.32 curve over that interval (the curve is piecewise linear between
  its 1% points), so tied model prices map to one value and the hour-weighted mean of
  the mapped series is the curve's own mean exactly.
* **Static.** Dispatch does not respond: volumes, flows and the balancing stage are
  held as solved.  The mapped series is a bracket on the price level, not a forecast.

Source: ``data/CP30/cp32_price_duration_curve_2030.csv``, extracted from sheet CP.32
of ``data/CP30/Clean Power 2030 Data Workbook_V003.xlsx`` (Annex 4, Figure 11:
"Price duration curve 2030", 101 points at 1% of hours).  The chart states no price
year; Annex 4 section 2 says "All costs have been rebased to 2024 prices", so the
curve is read as 2024 money, the model's price year (a few percent of uncertainty).
Settings live in ``config/defaults/base/consumer_bill.yaml`` under ``consumer_bill.price_level_mapping``; the code
defaults (:data:`DEFAULT_SETTINGS`) must match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: The CP.32 extract's columns: the share of the year's hours at or above a price,
#: then one price column per series of the chart.
SHARE_COLUMN = "share_of_hours"
PATHWAYS = ("actual_2023", "counterfactual", "further_flex_and_renewables", "new_dispatch")

DEFAULT_SETTINGS: dict = {
    "source_file": "data/CP30/cp32_price_duration_curve_2030.csv",
    "source": ("Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), sheet CP.32 'Price duration "
               "curve 2030' (Annex 4 Figure 11)"),
    "pathway": "further_flex_and_renewables",
    # Annex 4 section 2: "All costs have been rebased to 2024 prices"; CP.32 states no year.
    "price_year": 2024,
    # Carries the curve from its price year to the model's (1.0: both 2024).
    "rebasing_to_model_price_year": 1.0,
}


def settings_from_config(block: Mapping | None) -> dict:
    """``price_level_mapping`` over :data:`DEFAULT_SETTINGS`, refusing unknown keys and pathways."""
    block = dict(block or {})
    if unknown := sorted(set(block) - set(DEFAULT_SETTINGS)):
        raise ValueError(f"price_level_mapping has unknown keys {unknown}; supported: {sorted(DEFAULT_SETTINGS)}")
    merged = {**DEFAULT_SETTINGS, **block}
    if merged["pathway"] not in PATHWAYS:
        raise ValueError(f"price_level_mapping.pathway={merged['pathway']!r}; supported: {list(PATHWAYS)}")
    if not float(merged["rebasing_to_model_price_year"]) > 0:
        raise ValueError("price_level_mapping.rebasing_to_model_price_year must be positive")
    return merged


def load_duration_curve(path, pathway: str = DEFAULT_SETTINGS["pathway"], rebasing: float = 1.0) -> pd.Series:
    """The pathway's duration curve: price (GBP/MWh) indexed by the share of hours exceeded, 0 to 1.

    Refuses a curve that does not span 0 to 1 in increasing shares or whose price
    rises with the share (a duration curve falls).
    """
    frame = pd.read_csv(path)
    if pathway not in frame.columns or SHARE_COLUMN not in frame.columns:
        raise ValueError(f"{path} lacks {SHARE_COLUMN!r} or {pathway!r}")
    share = pd.to_numeric(frame[SHARE_COLUMN], errors="raise").to_numpy(dtype=float)
    price = pd.to_numeric(frame[pathway], errors="raise").to_numpy(dtype=float) * float(rebasing)
    if not np.isclose(share[0], 0.0) or not np.isclose(share[-1], 1.0) or np.any(np.diff(share) <= 0):
        raise ValueError(f"{path}: shares must rise strictly from 0 to 1")
    if np.any(np.diff(price) > 1e-9):
        raise ValueError(f"{path}: {pathway} is not a duration curve (price rises with the share of hours)")
    return pd.Series(price, index=pd.Index(share, name=SHARE_COLUMN), name=pathway)


def curve_mean(curve: pd.Series) -> float:
    """The time-average price of a duration curve (the trapezoid over its points)."""
    return float(_cumulative_integral(curve, np.array([1.0]))[0])


def _cumulative_integral(curve: pd.Series, x: np.ndarray) -> np.ndarray:
    """``integral_0^x curve(s) ds`` for each ``x`` (piecewise-linear curve)."""
    s = curve.index.to_numpy(dtype=float)
    p = curve.to_numpy(dtype=float)
    knots = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(s))])
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    k = np.clip(np.searchsorted(s, x, side="right") - 1, 0, len(s) - 2)
    width = x - s[k]
    slope = (p[k + 1] - p[k]) / (s[k + 1] - s[k])
    return knots[k] + p[k] * width + 0.5 * slope * width ** 2


def quantile_map(prices: pd.Series, curve: pd.Series, hours: pd.Series | None = None,
                 zero_band: float | None = None) -> pd.Series:
    """Each hour's price on ``curve`` at the hour's rank in ``prices``.

    Hours are ranked from the highest price down; an hour weighted ``h`` covers ``h /
    sum(h)`` of the year.  Tied prices form one interval, so they map to one value.
    The mapped price of an interval ``[a, b]`` is the curve's mean over it, so the
    hour-weighted mean of the result is :func:`curve_mean` exactly and the order of
    hours is kept (a higher model price never maps lower).

    ``zero_band`` (GBP/MWh, default None: off) treats every price within ``+/- zero_band``
    of zero as exactly zero before ranking (12 Oct 2026). A third of the model's hours
    clear at about zero but differ by solver noise (1e-6), which otherwise orders them and
    spreads their mapped prices across the curve's shoulder; banded, they tie and take the
    curve's mean over their combined interval. Hours below the band keep their rank.
    """
    prices = pd.Series(prices, dtype=float)
    if prices.isna().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("prices to map must be finite")
    if zero_band is not None:
        if not float(zero_band) >= 0.0:
            raise ValueError("zero_band must be non-negative")
        prices = prices.mask(prices.abs() <= float(zero_band), 0.0)
    weights = (pd.Series(1.0, index=prices.index) if hours is None
               else pd.Series(hours, dtype=float).reindex(prices.index))
    if weights.isna().any() or (weights <= 0).any():
        raise ValueError("hour weights must be positive and cover every price")
    frame = pd.DataFrame({"price": prices.to_numpy(), "w": weights.to_numpy()})
    groups = frame.groupby("price", sort=True)["w"].sum().iloc[::-1]  # highest price first
    total = float(groups.sum())
    upper = groups.cumsum().to_numpy() / total
    lower = np.concatenate([[0.0], upper[:-1]])
    upper[-1] = 1.0
    mean = (_cumulative_integral(curve, upper) - _cumulative_integral(curve, lower)) / (upper - lower)
    mapped = pd.Series(mean, index=groups.index)
    return pd.Series(frame["price"].map(mapped).to_numpy(), index=prices.index, name="mapped_price_gbp_per_mwh")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    weights = pd.Series(weights, dtype=float).reindex(values.index)
    return float((values * weights).sum() / weights.sum())


def capture_price(energy_mwh: pd.DataFrame | pd.Series, price: pd.Series) -> pd.Series | float:
    """Energy-weighted price earned by each column of ``energy_mwh`` (hourly MWh)."""
    if isinstance(energy_mwh, pd.Series):
        total = float(energy_mwh.sum())
        return float((energy_mwh * price.reindex(energy_mwh.index)).sum() / total) if total else np.nan
    total = energy_mwh.sum()
    revenue = energy_mwh.mul(price.reindex(energy_mwh.index), axis=0).sum()
    return revenue / total.where(total != 0)


def read_market_year(network_paths: Sequence, price_paths: Sequence, link_paths: Sequence | None = None) -> pd.DataFrame:
    """The market year's hourly price, hours and metered demand, as the bill reads them.

    Each month through ``project_future_bill.read_hourly_inputs`` (the GB demand-bus
    price the bill settles on, item D1's demand basis), concatenated in time order.
    """
    from scripts.consumer_bill.project_future_bill import read_hourly_inputs

    if len(network_paths) != len(price_paths) or not network_paths:
        raise ValueError("network and price paths must be non-empty and aligned")
    links = list(link_paths) if link_paths else [None] * len(network_paths)
    frames = []
    for network, price, link in zip(network_paths, price_paths, links):
        if link is None and str(network).endswith("_wholesale.nc"):
            candidate = Path(str(network)[: -len("_wholesale.nc")] + "_wholesale_links.csv")
            link = candidate if candidate.exists() else None
        frame = read_hourly_inputs(network, price, link)
        frame["price_file"] = str(price)
        frames.append(frame)
    hourly = pd.concat(frames).sort_index()
    if hourly.index.has_duplicates:
        raise ValueError("the market months overlap")
    return hourly


def map_market_year(hourly: pd.DataFrame, curve: pd.Series, zero_band: float | None = None) -> pd.Series:
    """The whole year's hours ranked together (never month by month) and mapped onto ``curve``."""
    return quantile_map(hourly["price_gbp_per_mwh"], curve, hourly["hours"], zero_band=zero_band)


def write_mapped_price_files(hourly: pd.DataFrame, mapped: pd.Series, out_dir=None,
                             targets: Sequence | None = None) -> list[Path]:
    """Copies of each month's price CSV with ``wholesale_price`` on the mapped series.

    The other columns (the spread, the hydrogen price) are held as solved; the model
    price is kept beside it as ``model_wholesale_price``.  Returns the paths in the
    months' order.  Each copy is written to ``out_dir`` under its source's name, or,
    with ``targets`` (one path per month in the months' order; 12 Oct 2026, the
    ownership layer's price-level twins), to that path.
    """
    sources = list(dict.fromkeys(hourly["price_file"]))
    if targets is not None:
        targets = [Path(t) for t in targets]
        if len(targets) != len(sources):
            raise ValueError(f"{len(targets)} target paths for {len(sources)} month price files")
    elif out_dir is None:
        raise ValueError("give out_dir or targets")
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        targets = [Path(out_dir) / Path(source).name for source in sources]
    paths = []
    for source, target in zip(sources, targets):
        frame = pd.read_csv(source, index_col=0, parse_dates=True)
        rows = hourly.index[hourly["price_file"].eq(source)]
        frame["model_wholesale_price"] = frame["wholesale_price"]
        frame.loc[rows, "wholesale_price"] = mapped.loc[rows].to_numpy()
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index_label=frame.index.name or "snapshot")
        paths.append(target)
    return paths


def mapping_summary(hourly: pd.DataFrame, mapped: pd.Series, curve: pd.Series, settings: Mapping) -> dict:
    """The two price levels side by side (time- and demand-weighted, zero-price share, deciles)."""
    price, hours, demand = hourly["price_gbp_per_mwh"], hourly["hours"], hourly["demand_mw"] * hourly["hours"]

    def _level(series):
        order = series.sort_values(ascending=False)
        share = hours.reindex(order.index).cumsum() / hours.sum()
        return {
            "time_weighted_gbp_per_mwh": weighted_mean(series, hours),
            "demand_weighted_gbp_per_mwh": weighted_mean(series, demand),
            "share_of_hours_at_or_below_1_gbp": float(hours[series.le(1.0)].sum() / hours.sum()),
            "price_exceeded_in_share_of_hours": {
                f"{q:.2f}": float(order[share.ge(q)].iloc[0]) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)},
        }

    return {
        "source": settings["source"], "source_file": str(settings["source_file"]), "pathway": settings["pathway"],
        "price_year": int(settings["price_year"]),
        "rebasing_to_model_price_year": float(settings["rebasing_to_model_price_year"]),
        "curve_mean_gbp_per_mwh": curve_mean(curve),
        "hours": float(hours.sum()),
        "model": _level(price), "mapped": _level(mapped),
        "method": ("quantile mapping: each hour keeps its rank in the model's own year and takes the curve's mean "
                   "over the share of hours it covers; static, so volumes and the balancing stage are as solved"),
    }
