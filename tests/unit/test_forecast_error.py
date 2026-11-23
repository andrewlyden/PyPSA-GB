"""Day-ahead forecast error: reproducible, calibrated, and identical across solve periods."""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.forecast_error import (
    apply_day_ahead_forecast_error,
    national_error_series,
    year_index,
)


def _network(start="2030-01-03", periods=24 * 7, n_wind=40, cf=0.5, n_solar=10):
    n = pypsa.Network()
    snaps = pd.date_range(start, periods=periods, freq="h")
    n.set_snapshots(snaps)
    n.add("Bus", "GB1")
    n.add("Bus", "EXT")
    n.buses["country"] = ["GB", "FR"]
    for i in range(n_wind):
        n.add("Generator", f"w{i}", bus="GB1", carrier="wind_onshore", p_nom=100.0,
              p_max_pu=pd.Series(cf, index=snaps))
    for i in range(n_solar):
        n.add("Generator", f"s{i}", bus="GB1", carrier="solar_pv", p_nom=50.0,
              p_max_pu=pd.Series(0.2, index=snaps))
    n.add("Generator", "gas", bus="GB1", carrier="CCGT", p_nom=1000.0)
    n.add("Load", "gb_load", bus="GB1", p_set=pd.Series(3000.0, index=snaps))
    n.add("Load", "ext_load", bus="EXT", p_set=pd.Series(500.0, index=snaps))
    return n


def test_disabled_changes_nothing():
    n = _network()
    before = n.generators_t.p_max_pu.copy()
    out = apply_day_ahead_forecast_error(n, {"enabled": False}, logging.getLogger("t"))
    assert out == {}
    pd.testing.assert_frame_equal(n.generators_t.p_max_pu, before)


def test_series_is_reproducible_and_key_specific():
    a = national_error_series(2030, 0.05, 6.0, 7, "wind_onshore")
    b = national_error_series(2030, 0.05, 6.0, 7, "wind_onshore")
    c = national_error_series(2030, 0.05, 6.0, 7, "wind_offshore")
    pd.testing.assert_series_equal(a, b)
    assert not np.allclose(a.values, c.values)
    assert len(a) == len(year_index(2030)) == 8760


def test_series_has_target_std_and_autocorrelation():
    s = national_error_series(2030, 0.055, 6.0, 1, "x")
    assert s.std() == pytest.approx(0.055, rel=0.10)
    assert s.autocorr(lag=1) == pytest.approx(np.exp(-1 / 6.0), abs=0.03)


def test_week_matches_the_month_that_contains_it():
    cfg = {"enabled": True, "seed": 11, "technologies": {"wind_onshore": 0.05}, "demand_rmse": 0.0}
    week = _network("2030-01-03", 24 * 7)
    month = _network("2030-01-01", 24 * 31)
    apply_day_ahead_forecast_error(week, cfg)
    apply_day_ahead_forecast_error(month, cfg)
    w = week.generators_t.p_max_pu["w0"]
    m = month.generators_t.p_max_pu["w0"].reindex(w.index)
    pd.testing.assert_series_equal(w, m, check_names=False)


def test_realised_national_rmse_matches_target_without_clipping():
    n = _network(periods=24 * 60, cf=0.5)
    cfg = {"enabled": True, "seed": 3, "technologies": {"wind_onshore": 0.05}, "demand_rmse": 0.0}
    out = apply_day_ahead_forecast_error(n, cfg)
    t = out["technologies"]["wind_onshore"]
    assert t["units"] == 40 and t["capacity_mw"] == 4000.0
    assert t["realised_rmse_of_capacity"] == pytest.approx(0.05, rel=0.20)
    # one national series: every unit moved by the same capacity-factor delta
    p = n.generators_t.p_max_pu
    assert np.allclose(p["w0"] - 0.5, p["w1"] - 0.5)


def test_clipped_to_unit_interval_and_solar_untouched_when_not_listed():
    n = _network(cf=0.98)
    cfg = {"enabled": True, "seed": 5, "technologies": {"wind_onshore": 0.30}, "demand_rmse": 0.0}
    apply_day_ahead_forecast_error(n, cfg)
    p = n.generators_t.p_max_pu
    assert p["w0"].max() <= 1.0 and p["w0"].min() >= 0.0
    assert (p["s0"] == 0.2).all()


def test_demand_error_touches_gb_loads_only():
    n = _network()
    cfg = {"enabled": True, "seed": 9, "technologies": {}, "demand_rmse": 0.02}
    out = apply_day_ahead_forecast_error(n, cfg)
    assert out["demand"]["loads"] == 1
    assert (n.loads_t.p_set["ext_load"] == 500.0).all()
    assert not (n.loads_t.p_set["gb_load"] == 3000.0).all()
    assert out["demand"]["realised_rmse_fraction"] == pytest.approx(0.02, rel=0.35)


def test_no_resource_means_no_forecast_and_no_bias():
    n = _network(periods=24 * 30)
    night = pd.Series(0.0, index=n.snapshots)
    for i in range(10):
        n.generators_t.p_max_pu[f"s{i}"] = night
    cfg = {"enabled": True, "seed": 4, "technologies": {"solar_pv": 0.30, "wind_onshore": 0.07}, "demand_rmse": 0.0}
    out = apply_day_ahead_forecast_error(n, cfg)
    assert (n.generators_t.p_max_pu["s0"] == 0.0).all()
    assert out["technologies"]["solar_pv"]["forecast_minus_outturn_gwh"] == 0.0
    w = out["technologies"]["wind_onshore"]
    # symmetric error at half output: the bias is a small fraction of the spread
    assert abs(w["forecast_minus_outturn_gwh"]) < 0.25 * w["realised_rmse_of_capacity"] * w["capacity_mw"] * 24 * 30 / 1e3
