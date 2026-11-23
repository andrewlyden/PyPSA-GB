"""The FES long-range chain for any year.

Renamed keys refused (the aliases are gone since the config restructure), the year-parameter table, leap-year calendars (2040 has 8,784 hours) and
the per-year file templating (TYNDP, offshore landing).
"""

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import config_loader as cl
from scripts.utilities.calendar_year import fill_leap_day, hours_in_year, on_calendar

ROOT = Path(__file__).resolve().parents[2]


# ───────────────────────────── renamed keys ─────────────────────────────

@pytest.mark.parametrize("entry, new_path", [
    ({"fes2050_supply": {"enabled": True}}, "fleet.fes_supply"),
    ({"fes_supply": {"enabled": True}}, "fleet.fes_supply"),
    ({"demand_disaggregation": {"fes2050": {"enabled": True}}}, "demand.disaggregation"),
    ({"hydrogen": {"fes2050": {"enabled": True}}}, "hydrogen.fes"),
])
def test_old_keys_are_refused_with_the_new_path_named(entry, new_path):
    with pytest.raises(ValueError, match=rf"write {new_path.replace('.', '[.]')} "):
        cl.check_renamed_keys(entry, "X")
    assert cl.RENAMED_KEYS["fes2050_supply"] == "fleet.fes_supply"
    assert cl.RENAMED_KEYS["hydrogen.fes2050"] == "hydrogen.fes"
    assert cl.RENAMED_KEYS["demand_disaggregation.fes2050"] == "demand.disaggregation.fes"
    assert not hasattr(cl, "apply_deprecated_aliases") and not hasattr(cl, "DEPRECATED_KEY_ALIASES")


def test_the_new_paths_pass_the_check():
    cl.check_renamed_keys({"fleet": {"fes_supply": {"enabled": True}},
                           "demand": {"disaggregation": {"fes": {"enabled": True}}},
                           "hydrogen": {"fes": {"enabled": True}}}, "X")


def test_demand_and_hydrogen_helpers_read_only_the_new_key():
    from scripts.demand.fes_demand import fes_demand_config
    from scripts.hydrogen.fes_hydrogen import fes_hydrogen_config
    assert fes_demand_config({"fes": {"enabled": True}}) == {"enabled": True}
    assert fes_hydrogen_config({"fes": {"x": 1}}) == {"x": 1}
    assert fes_demand_config({"fes2050": {"enabled": True}}) == {}       # no alias
    assert fes_hydrogen_config({"fes2050": {"x": 1}}) == {}
    assert fes_demand_config(None) == {} and fes_hydrogen_config({}) == {}


# ───────────────────────────── year parameters ─────────────────────────────

def test_year_parameter_table_has_todays_2050_values_with_sources():
    table = cl.year_parameter_table()
    expected = {"hydrogen_supply_price_gbp_per_mwh_h2": 64.0, "hydrogen_end_of_window_value_gbp_per_mwh_h2": 36.75,
                "ccs_gas_variable_om_gbp_per_mwh": 18.09, "ccs_gas_efficiency": 0.47,
                "performance_factor_wind_offshore": 0.9261, "performance_factor_wind_onshore": 0.7464,
                "performance_factor_solar_pv": 0.9968, "hydrogen_reserve_twh": 4.0}
    for key, value in expected.items():
        assert table[(2050, "HT", key)] == value, key
    # The 2040 values (10-11 Nov 2026): decided or derived, each with its source; the performance factors from the
    # 2040 calibration build (ES1 gross over available energy at factor 1.0, 11 Nov 2026).
    expected_2040 = {"hydrogen_supply_price_gbp_per_mwh_h2": 64.0, "hydrogen_end_of_window_value_gbp_per_mwh_h2": 36.49,
                     "ccs_gas_variable_om_gbp_per_mwh": 18.09, "ccs_gas_efficiency": 0.47, "hydrogen_reserve_twh": 2.22}
    for key, value in expected_2040.items():
        assert table[(2040, "HT", key)] == value, key
    factors = {"performance_factor_wind_offshore": 0.9299, "performance_factor_wind_onshore": 0.7441,
               "performance_factor_solar_pv": 0.998}
    for key, value in factors.items():
        assert table[(2040, "HT", key)] == value, key
    frame = pd.read_csv(cl.YEAR_PARAMETERS_FILE)
    assert list(frame.columns) == ["year", "pathway", "key", "value", "unit", "source"]
    assert frame["source"].str.len().gt(20).all()


def test_year_parameters_resolve_by_year_and_pathway():
    table = {(2050, "HT", "h2_price"): 64.0}
    merged = {"modelled_year": 2050, "targets": {"pathway": "HT"},
              "market": {"price": "year_parameter:h2_price", "list": ["year_parameter:h2_price", 1]}}
    cl.resolve_year_parameters(merged, "X", table)
    assert merged["market"] == {"price": 64.0, "list": [64.0, 1]}
    missing = {"modelled_year": 2040, "targets": {"pathway": "HT"}, "market": {"price": "year_parameter:h2_price"}}
    with pytest.raises(ValueError, match="no 'h2_price' for pathway HT year 2040"):
        cl.resolve_year_parameters(missing, "FES25HT_2040", table)
    # A study that writes the number needs no row.
    written = {"modelled_year": 2040, "targets": {"pathway": "HT"}, "market": {"price": 70.0}}
    assert cl.resolve_year_parameters(written, "FES25HT_2040", table)["market"]["price"] == 70.0


# ───────────────────────────── leap years ─────────────────────────────

def test_hours_in_year():
    assert hours_in_year(2040) == 8784 and hours_in_year(2050) == 8760 and hours_in_year(2100) == 8760


def _weather_2013():
    index = pd.date_range("2013-01-01", periods=8760, freq="h")
    return pd.Series(np.arange(8760, dtype=float), index=index)


def test_fill_leap_day_copies_28_february():
    shifted = _weather_2013()
    shifted.index = shifted.index + pd.DateOffset(years=27)        # onto 2040 as the renewable mapping does
    out = fill_leap_day(shifted)
    assert len(out) == 8784 and out.index.is_monotonic_increasing
    feb28 = out[(out.index.month == 2) & (out.index.day == 28)].to_numpy()
    feb29 = out[(out.index.month == 2) & (out.index.day == 29)].to_numpy()
    np.testing.assert_array_equal(feb28, feb29)
    assert out["2040-03-01 00:00"] == shifted["2040-03-01 00:00"]   # 1 March keeps 1 March, not shifted a day
    same = _weather_2013().copy()
    same.index = same.index + pd.DateOffset(years=37)                # 2050: unchanged
    pd.testing.assert_series_equal(fill_leap_day(same), same)


def test_on_calendar_maps_by_date():
    target = pd.date_range("2040-01-01", periods=8784, freq="h")
    out = on_calendar(_weather_2013(), target)
    assert len(out) == 8784
    assert out["2040-02-29 05:00"] == out["2040-02-28 05:00"]
    assert out["2040-12-31 23:00"] == _weather_2013()["2013-12-31 23:00"]


def test_tyndp_hours_on_a_leap_weather_year():
    from scripts.interconnectors.tyndp_prices import HOURS_TYNDP2024, ntplus_to_calendar, tyndp2024_to_calendar
    tyndp = pd.Series(np.arange(HOURS_TYNDP2024, dtype=float))
    common = tyndp2024_to_calendar(tyndp, 2013)
    leap = tyndp2024_to_calendar(tyndp, 2020)
    assert len(common) == 8760 and len(leap) == 8784
    assert leap["2020-02-29 12:00"] == leap["2020-02-28 12:00"]
    assert leap["2020-03-01 12:00"] == common["2013-03-01 12:00"]
    ntplus = ntplus_to_calendar(pd.Series(np.arange(8760, dtype=float)), 2040)
    assert len(ntplus) == 8784 and ntplus["2040-12-31 23:00"] == 8759.0


def test_heat_pump_season_spans_the_leap_year():
    from scripts.demand.fes_demand import heat_pump_profiles, resolve_config
    cfg = resolve_config({})
    for year in (2040, 2050):
        idx = pd.date_range(f"{year}-01-01", periods=hours_in_year(year), freq="h")
        out = heat_pump_profiles(pd.Series(8.0, index=idx), cfg)
        assert out["hot_water_th"].sum() == pytest.approx(cfg["heat_pumps"]["hot_water_share"])
        assert len(out) == hours_in_year(year)


# ───────────────────────────── per-year files ─────────────────────────────

def test_tyndp_files_follow_the_year():
    from scripts.interconnectors.tyndp_prices import tyndp_input_files
    de = tyndp_input_files("DE", 2050, "d")
    assert de == {"tyndp2024_cy_1995": "d/tyndp2024_DE2050CY1995_hourly.csv",
                  "tyndp2024_cy_2008": "d/tyndp2024_DE2050CY2008_hourly.csv",
                  "tyndp2024_cy_2009": "d/tyndp2024_DE2050CY2009_hourly.csv"}
    assert tyndp_input_files("GA_ic1rescale", 2040, "d")["tyndp2024_cy_2009"] == "d/tyndp2024_GA2040CY2009_hourly.csv"
    nt = tyndp_input_files("NTplus2026", 2040, "d")
    assert nt == {"ntplus_prices": "d/tyndp2026_NTplus2040_mc.csv",
                  "ntplus_wind": "d/tyndp2026_NTplus2040_WS071_UK00_wind.csv"}
    for path in tyndp_input_files("NTplus2026", 2050, str(ROOT / "data/interconnectors/tyndp2024")).values():
        assert Path(path).exists(), path                             # the 2050 names are the committed files


def test_offshore_landing_outputs_and_totals_follow_the_year(tmp_path):
    from scripts.network_clustering import fes_offshore_landing as fol
    assert fol.OUTPUTS["out_landing"].format(year=2050) == "data/network/ETYS/fes2050_offshore_landing.csv"
    fes = pd.DataFrame([{"Building Block ID Number": "Gen_BB014", "FES Pathway": "Holistic Transition",
                         "GSP": gsp, "2040": mw, "2050": 2 * mw}
                        for gsp, mw in (("Direct(NGET)", 100.0), ("Direct(SHETL)", 50.0), ("Direct(SPTL)", 25.0))])
    path = tmp_path / "fes.csv"
    fes.to_csv(path, index=False)
    assert fol.fes_direct_offshore(str(path), 2040).to_dict() == {"NGET": 100.0, "SHETL": 50.0, "SPTL": 25.0}
    assert fol.fes_direct_offshore(str(path), 2050)["NGET"] == 200.0
