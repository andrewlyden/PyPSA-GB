"""Item IC1: the ENTSO-E hourly shape, conditioned on GB wind of the weather year.

The CP30 months use 2013 weather, but ENTSO-E day-ahead prices start in 2015. A
2024 shape laid over 2013 wind loses the negative correlation between GB wind and
foreign prices (measured -0.15 to -0.32 in 2023 and 2024, about zero across years),
which overpays GB's windy-hour exports. The conditioning swaps the part of the
shape GB wind explains in the shape year for the same part driven by the weather
year's wind, keeping everything else in the shape.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.interconnectors.price_shape import (
    align_by_calendar,
    condition_prices_on_wind,
    wind_index,
)


def _hours(year):
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")


def _series(year, seed):
    rng = np.random.default_rng(seed)
    idx = _hours(year)
    return pd.Series(rng.normal(size=len(idx)), index=idx)


def test_wind_index_is_total_output_standardised_within_the_year():
    idx = _hours(2013)[:4]
    profile = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}, index=idx)
    z = wind_index(profile)
    total = profile.sum(axis=1)
    assert z.tolist() == pytest.approx(((total - total.mean()) / total.std(ddof=0)).tolist())
    # Different site sets in different years must not change the index scale.
    assert wind_index(profile * 7.0).tolist() == pytest.approx(z.tolist())


def test_calendar_alignment_maps_the_leap_day_to_28_february():
    source = pd.Series(np.arange(len(_hours(2013)), dtype=float), index=_hours(2013))
    target = _hours(2024)
    aligned = align_by_calendar(source, target)
    assert aligned.index.equals(target)
    assert not aligned.isna().any()
    feb28 = source.loc["2013-02-28 05:00"]
    assert aligned.loc["2024-02-29 05:00"] == feb28
    assert aligned.loc["2024-03-01 00:00"] == source.loc["2013-03-01 00:00"]


def test_conditioning_restores_the_wind_correlation_and_keeps_the_rest():
    wind_shape_year = _series(2024, 1)
    wind_weather_year = _series(2013, 2)
    noise = _series(2024, 3) * 20.0
    prices = pd.DataFrame({
        "France": 60.0 - 15.0 * wind_shape_year + noise,
        "Norway": 40.0 + 0.0 * wind_shape_year + noise,
    })
    out, betas = condition_prices_on_wind(prices, wind_shape_year, wind_weather_year)

    assert out.index.equals(prices.index)
    assert betas["France"] == pytest.approx(-15.0, abs=1.0)
    assert betas["Norway"] == pytest.approx(0.0, abs=1.0)

    target = align_by_calendar(wind_weather_year, prices.index)
    before = prices["France"].corr(wind_shape_year)
    after = out["France"].corr(target)
    assert after == pytest.approx(before, abs=0.03)
    # The residual shape is untouched: what is left after removing each year's
    # wind term is identical.
    residual_in = prices["France"] - betas["France"] * wind_shape_year
    residual_out = out["France"] - betas["France"] * target
    assert np.allclose(residual_in, residual_out)
    # The annual level moves only by the wind term's mean, which is ~0.
    assert out["France"].mean() == pytest.approx(prices["France"].mean(), abs=1.0)


def test_timezone_aware_price_index_is_accepted():
    wind_shape_year = _series(2024, 4)
    idx = wind_shape_year.index.tz_localize("UTC")
    prices = pd.DataFrame({"France": 50.0 - 10.0 * wind_shape_year.to_numpy()}, index=idx)
    out, _ = condition_prices_on_wind(prices, wind_shape_year, _series(2013, 5))
    assert out.index.equals(idx)
