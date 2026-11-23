"""Item G6 follow-up: what a heat pump or an EV costs to serve, against the domestic average.

A single-rate tariff charges these households the same unit rate as everyone else,
so their bill does not show it; the gap between their profile-weighted wholesale
cost and the domestic profile's is the cross-subsidy a flat rate hides.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill.archetype_cost_to_serve import (
    cop, cost_to_serve, ev_profile, heat_pump_profile, weighted_price,
)

IDX = pd.date_range("2030-01-01", periods=48, freq="h")


def test_the_cop_line_passes_through_the_two_trial_figures():
    params = {"spf": 2.80, "spf_mean_temperature_c": 7.0, "cold_cop": 2.44, "cold_temperature_c": 0.0, "min_cop": 1.5}
    assert cop(pd.Series([7.0, 0.0]), params).tolist() == pytest.approx([2.80, 2.44])
    assert cop(pd.Series([-40.0]), params).iloc[0] == 1.5


def test_heat_pump_load_follows_degree_hours_over_the_cop():
    temps = pd.Series([20.0, 10.0, 0.0], index=IDX[:3])
    params = {"base_temperature_c": 15.5, "spf": 2.80, "spf_mean_temperature_c": 7.0,
              "cold_cop": 2.44, "cold_temperature_c": 0.0, "min_cop": 1.5}
    shape = heat_pump_profile(temps, params)
    assert shape.iloc[0] == 0.0
    assert shape.iloc[2] > shape.iloc[1] > 0
    assert shape.sum() == pytest.approx(1.0)


def test_ev_profiles_put_their_energy_in_the_declared_hours():
    evening = ev_profile(IDX, {"hours": [18, 19, 20, 21]})
    assert evening.sum() == pytest.approx(1.0)
    assert set(evening[evening > 0].index.hour) == {18, 19, 20, 21}


def test_weighted_price_and_cost_to_serve():
    price = pd.Series(np.where(IDX.hour >= 18, 100.0, 40.0), index=IDX)
    flat = pd.Series(1.0 / len(IDX), index=IDX)
    evening = ev_profile(IDX, {"hours": [18, 19]})
    assert weighted_price(price, evening) == pytest.approx(100.0)
    result = cost_to_serve(price, {"ev_evening": evening}, flat, added_kwh={"ev_evening": 2000.0})
    row = result.loc["ev_evening"]
    assert row["premium_gbp_per_mwh"] == pytest.approx(100.0 - weighted_price(price, flat))
    assert row["premium_gbp_per_household"] == pytest.approx(row["premium_gbp_per_mwh"] * 2.0)
