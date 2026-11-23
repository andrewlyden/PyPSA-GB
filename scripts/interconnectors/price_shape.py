"""An ENTSO-E hourly price shape conditioned on the GB wind of the weather year (item IC1).

The CP30 months run on 2013 weather, but ENTSO-E day-ahead prices start in 2015, so
the hourly shape has to come from another year. Laid over the wrong year's wind it
loses a real correlation: foreign prices fall when GB wind is high (correlation -0.15
to -0.32 against GB offshore output in 2023 and 2024, about zero when a 2024 shape
is paired with 2013 wind), so a decorrelated shape overpays GB's windy-hour exports.

The conditioning keeps the shape year's prices and swaps only the part GB wind
explains. Per country, with ``z`` the standardised GB offshore output of each year::

    b = cov(price, z_shape) / var(z_shape)            (fitted on the shape year)
    price' = price - b * z_shape + b * z_weather      (weather year on the shape calendar)

Everything the wind term does not explain (the gas-price level, the daily and weekly
profile, events) is left exactly as it was. The level is re-centred to CP.29
downstream, in ``add_to_network``, so only the shape matters here.

Declared simplifications: one linear regressor, GB offshore output, standing in for
north-west European weather; the index is total output across the profile file's
sites, standardised within each year, because the site sets differ between years.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


def _naive(index: pd.Index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(index))
    return index.tz_convert(None) if index.tz is not None else index


def wind_index(profile: pd.DataFrame) -> pd.Series:
    """Total output across sites, standardised within the year (mean 0, sd 1)."""
    total = profile.apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    sd = float(total.std(ddof=0))
    if sd <= 0.0:
        raise ValueError("wind profile has no variation; it cannot condition a price shape")
    out = (total - total.mean()) / sd
    out.index = _naive(out.index)
    return out


def align_by_calendar(source: pd.Series, target_index: pd.Index) -> pd.Series:
    """Values of ``source`` at the same month, day and hour as each target timestamp.

    A leap day with no source counterpart takes 28 February's value.
    """
    src = source.copy()
    src.index = _naive(src.index)
    keys = pd.MultiIndex.from_arrays([src.index.month, src.index.day, src.index.hour])
    lookup = pd.Series(src.to_numpy(), index=keys)
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    target = _naive(target_index)
    source_has_leap_day = bool(((src.index.month == 2) & (src.index.day == 29)).any())
    leap_day = (target.month == 2) & (target.day == 29)
    day = np.where(leap_day & (not source_has_leap_day), 28, target.day)
    wanted = pd.MultiIndex.from_arrays([target.month, day, target.hour])
    values = lookup.reindex(wanted).to_numpy()
    if np.isnan(values).any():
        raise ValueError(f"{int(np.isnan(values).sum())} target hours have no calendar match in the wind series")
    return pd.Series(values, index=pd.Index(target_index))


def condition_prices_on_wind(
    prices: pd.DataFrame, wind_shape_year: pd.Series, wind_weather_year: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Swap the shape year's GB-wind term for the weather year's, per country.

    Returns the conditioned prices on the input index and the fitted slope per
    country (GBP/MWh per standard deviation of GB offshore output).
    """
    z_shape = align_by_calendar(wind_shape_year, prices.index).to_numpy()
    z_weather = align_by_calendar(wind_weather_year, prices.index).to_numpy()
    out = prices.copy()
    betas = {}
    for column in prices.columns:
        p = pd.to_numeric(prices[column], errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(p)
        if ok.sum() < 2:
            raise ValueError(f"{column}: no price observations to condition")
        zs = z_shape[ok]
        b = float(np.cov(p[ok], zs, ddof=0)[0, 1] / np.var(zs))
        betas[column] = b
        out[column] = p - b * z_shape + b * z_weather
    return out, pd.Series(betas, name="gbp_per_mwh_per_sd_gb_offshore")


def build_conditioned_shape(price_file, wind_shape_file, wind_weather_file, out_file, audit_file,
                            logger=_LOG) -> pd.DataFrame:
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    wind_shape = wind_index(pd.read_csv(wind_shape_file, index_col=0, parse_dates=True))
    wind_weather = wind_index(pd.read_csv(wind_weather_file, index_col=0, parse_dates=True))
    shaped, betas = condition_prices_on_wind(prices, wind_shape, wind_weather)
    z_weather = align_by_calendar(wind_weather, prices.index)
    z_shape = align_by_calendar(wind_shape, prices.index)
    audit = {
        "price_file": str(price_file),
        "wind_shape_year_file": str(wind_shape_file),
        "wind_weather_year_file": str(wind_weather_file),
        "slope_gbp_per_mwh_per_sd": betas.round(4).to_dict(),
        "corr_with_shape_year_wind_before": {c: round(float(prices[c].corr(z_shape)), 4) for c in prices},
        "corr_with_weather_year_wind_before": {c: round(float(prices[c].corr(z_weather)), 4) for c in prices},
        "corr_with_weather_year_wind_after": {c: round(float(shaped[c].corr(z_weather)), 4) for c in shaped},
        "mean_before": prices.mean().round(3).to_dict(),
        "mean_after": shaped.mean().round(3).to_dict(),
        "std_before": prices.std().round(3).to_dict(),
        "std_after": shaped.std().round(3).to_dict(),
    }
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    shaped.to_csv(out_file)
    Path(audit_file).write_text(json.dumps(audit, indent=2))
    for country in shaped.columns:
        logger.info("%s: slope %.2f GBP/MWh per sd, corr with weather-year wind %.3f -> %.3f",
                    country, betas[country], audit["corr_with_weather_year_wind_before"][country],
                    audit["corr_with_weather_year_wind_after"][country])
    return shaped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("price_file")
    ap.add_argument("wind_shape_file")
    ap.add_argument("wind_weather_file")
    ap.add_argument("out_file")
    ap.add_argument("audit_file")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_conditioned_shape(a.price_file, a.wind_shape_file, a.wind_weather_file, a.out_file, a.audit_file)


if __name__ == "__main__":
    main()
