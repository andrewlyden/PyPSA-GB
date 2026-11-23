"""Task Y3 (FES 2050 plan): European 2050 prices from TYNDP."""
import json

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors import tyndp_prices as tp
from scripts.interconnectors.add_to_network import _apply_price_case, add_interconnector_links

NODES = ("UK00", "UKNI", "IE00", "FR00", "BE00", "NL00", "DE00", "DKW1", "NOS0")


# --------------------------------------------------------------------------- config switch

def test_switch_is_off_unless_the_scenario_sets_it():
    assert not tp.is_tyndp_pricing({})
    assert not tp.is_tyndp_pricing(None)
    assert not tp.is_tyndp_pricing({"foreign_price_source": "entsoe"})
    assert tp.is_tyndp_pricing({"foreign_price_source": "tyndp"})


def test_variant_tokens_and_their_parse():
    assert tp.tyndp_variant({"foreign_price_source": "tyndp"}) == "DE"
    assert tp.tyndp_variant({"tyndp": {"scenario": "GA"}}) == "GA"
    assert tp.tyndp_variant({"tyndp": {"scenario": "DE", "route": "ic1_rescale"}}) == "DE_ic1rescale"
    assert tp.parse_variant("NTplus2026") == ("NTplus2026", "tyndp_shape")
    assert tp.parse_variant("GA_ic1rescale") == ("GA", "ic1_rescale")
    with pytest.raises(ValueError):
        tp.tyndp_settings({"tyndp": {"scenario": "NT"}})
    with pytest.raises(ValueError):
        tp.tyndp_settings({"tyndp": {"shape_climate_year": 1995}})
    with pytest.raises(ValueError):
        tp.parse_variant("XX")


def test_price_base_is_hicp_rebased_then_the_ic1_rate():
    assert tp.eur_to_model_gbp(0.86, 2022) == pytest.approx(0.86 * 126.08 / 116.83)
    assert tp.eur_to_model_gbp(0.86, 2024) == pytest.approx(0.86)


# --------------------------------------------------------------------------- calendar

def test_tyndp_hours_lie_on_2013_by_date_and_31_december_repeats_24_december():
    values = pd.Series(np.arange(tp.HOURS_TYNDP2024, dtype=float))
    out = tp.tyndp2024_to_calendar(values, 2013)
    assert len(out) == 8760 and out.index[0] == pd.Timestamp("2013-01-01 00:00")
    assert out.index[-1] == pd.Timestamp("2013-12-31 23:00")
    assert out["2013-12-30 23:00"] == tp.HOURS_TYNDP2024 - 1
    assert out["2013-12-31"].tolist() == out["2013-12-24"].tolist()
    leap = tp.tyndp2024_to_calendar(values, 2012)   # leap year (stage 5): 29 February takes 28 February's
    assert len(leap) == 8784 and leap["2012-02-29"].tolist() == leap["2012-02-28"].tolist()
    with pytest.raises(ValueError):
        tp.tyndp2024_to_calendar(values.iloc[:-1], 2013)


def test_weekly_low_day_finds_the_sunday():
    demand = np.ones(tp.HOURS_TYNDP2024) * 100.0
    for day in range(3, 364, 7):                       # day positions 3, 10, ... are low
        demand[day * 24:(day + 1) * 24] = 50.0
    assert tp.weekly_low_day(pd.Series(demand)) == 3
    assert tp._first_day_if_sunday_at(3) == "Thursday"


# --------------------------------------------------------------------------- zones, level

def test_sem_is_the_cable_weighted_mean_of_ie00_and_ukni():
    hourly = pd.DataFrame({f"mc_{n}": [10.0, 20.0] for n in NODES})
    hourly["mc_IE00"] = [30.0, 30.0]
    hourly["mc_UKNI"] = [60.0, 60.0]
    zones = tp.zone_prices(hourly, tp.ZONE_WEIGHTS_TYNDP2024)
    assert zones["Ireland"].tolist() == pytest.approx([30 * 1009 / 1509 + 60 * 500 / 1509] * 2)
    assert zones["France"].tolist() == [10.0, 20.0]
    assert list(zones.columns) == ["France", "Belgium", "Netherlands", "Germany", "Denmark", "Norway", "Ireland"]


def test_climate_weighted_level_and_relevel():
    means = {1995: pd.Series({"A": 10.0}), 2008: pd.Series({"A": 20.0}), 2009: pd.Series({"A": 30.0})}
    level = tp.climate_weighted_level(means, tp.TYNDP2024_CLIMATE_WEIGHTS)
    assert level["A"] == pytest.approx(0.23 * 10 + 0.37 * 20 + 0.40 * 30)
    prices = pd.DataFrame({"A": [0.0, 40.0]})
    ratio = tp.relevel(prices, pd.Series({"A": 10.0}), "ratio")
    assert ratio["A"].tolist() == pytest.approx([0.0, 20.0])            # zero hours stay zero
    shift = tp.relevel(prices, pd.Series({"A": 10.0}), "shift")
    assert shift["A"].tolist() == pytest.approx([-10.0, 30.0])
    with pytest.raises(ValueError):
        tp.climate_weighted_level({2009: means[2009]}, tp.TYNDP2024_CLIMATE_WEIGHTS)


# --------------------------------------------------------------------------- end to end

def _write_inputs(tmp_path, rng):
    wind = rng.gamma(2.0, 1.0, tp.HOURS_TYNDP2024)
    files = {}
    for cy, bump in ((1995, 0.0), (2008, 5.0), (2009, 10.0)):
        frame = pd.DataFrame({"time": [f"h{k}" for k in range(tp.HOURS_TYNDP2024)]})
        for node in NODES:
            frame[f"mc_{node}"] = np.clip(50.0 + bump - 6.0 * (wind - wind.mean()) + rng.normal(0, 3, len(wind)), 0, None)
        frame["UK00_wind_offshore_potential_mw"] = wind * 1000.0
        frame["demand_UK00_mw"] = 30000.0
        path = tmp_path / f"cy{cy}.csv"
        frame.to_csv(path, index=False)
        files[cy] = path
    idx = pd.date_range("2013-01-01", periods=8760, freq="h")
    weather = pd.DataFrame({"site": rng.gamma(2.0, 1.0, 8760)}, index=idx)
    wpath = tmp_path / "wind_offshore_2013.csv"
    weather.to_csv(wpath)
    return files, wpath


def test_build_writes_a_shape_at_the_tyndp_level_in_gbp_with_an_audit(tmp_path):
    rng = np.random.default_rng(0)
    files, wpath = _write_inputs(tmp_path, rng)
    out, audit = tmp_path / "shape.csv", tmp_path / "audit.json"
    shaped = tp.build_tyndp_shape("DE", wpath, out, audit, tyndp2024_files=files, eur_to_gbp=0.86)
    report = json.loads(audit.read_text())
    fx = 0.86 * 126.08 / 116.83
    for column in shaped:
        assert shaped[column].mean() == pytest.approx(report["weighted_level_eur"][column] * fx, rel=1e-4)
        assert report["slope_gbp_per_mwh_per_sd"][column] < 0
    assert len(shaped) == 8760 and shaped.index[0] == pd.Timestamp("2013-01-01")
    assert report["price_base"]["gbp_per_tyndp_eur"] == pytest.approx(fx, rel=1e-6)
    assert set(report["climate_year_means_eur"]) == {"1995", "2008", "2009"}
    written = pd.read_csv(out, index_col=0, parse_dates=True)
    assert list(written.columns) == list(shaped.columns)


def test_ic1_rescale_keeps_the_ic1_shape_at_the_tyndp_level(tmp_path):
    rng = np.random.default_rng(1)
    files, wpath = _write_inputs(tmp_path, rng)
    idx = pd.date_range("2024-01-01", periods=8784, freq="h", tz="UTC")
    ic1 = pd.DataFrame({c: 60.0 + rng.normal(0, 20, len(idx)) for c in tp.ZONE_WEIGHTS_TYNDP2024}, index=idx)
    ic1_path = tmp_path / "ic1.csv"
    ic1.to_csv(ic1_path)
    shaped = tp.build_tyndp_shape("DE_ic1rescale", wpath, tmp_path / "s.csv", tmp_path / "a.json",
                                  tyndp2024_files=files, ic1_shape_file=ic1_path)
    report = json.loads((tmp_path / "a.json").read_text())
    assert shaped["France"].corr(ic1["France"].reset_index(drop=True).set_axis(shaped.index)) == pytest.approx(1.0)
    assert shaped["France"].mean() == pytest.approx(report["weighted_level_gbp"]["France"], rel=1e-4)


# --------------------------------------------------------------------------- the network switch

def test_tyndp_reference_is_its_own_level_and_the_price_case_scales_it():
    hourly = pd.Series([0.0, 30.0, 60.0])
    ref, target = tp.tyndp_market_reference(hourly, "France", {}, -500, 500, _apply_price_case)
    assert ref.tolist() == pytest.approx([0.0, 30.0, 60.0]) and target == pytest.approx(30.0)
    high = {"future_price_case": "high", "price_case_multipliers": {"high": 1.2}}
    ref, target = tp.tyndp_market_reference(hourly, "France", high, -500, 500, _apply_price_case)
    assert ref.tolist() == pytest.approx([0.0, 36.0, 72.0]) and target == pytest.approx(36.0)


def _links(pricing):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=2, freq="h"))
    for bus in ("GB", "Ext_FR"):
        n.add("Bus", bus)
    cables = pd.DataFrame([{"name": "FR", "from_bus": "GB", "to_bus": "Ext_FR", "capacity_mw": 300.0,
                            "losses_percent": 2.5, "counterparty_country": "France"}])
    shape = pd.DataFrame({"France": [10.0, 30.0]}, index=pd.date_range("2013-01-01", periods=2, freq="h"))
    add_interconnector_links(n, cables, pd.DataFrame(), pd.DataFrame(), modelled_year=2050,
                             entsoe_prices_df=shape, pricing_config=pricing)
    return n


def test_default_pricing_still_recentres_and_tyndp_does_not():
    base = {"source": "derived", "gb_price": 80.0, "transaction_spread_gbp_per_mwh": 1.0}
    default = _links(base)
    # Default (no foreign_price_source): the shape is re-centred to the 80 GBP target.
    assert default.links.at["IC_FR", "market_import_level"] == pytest.approx(80.0)
    assert default.links_t.marginal_cost["IC_FR"].mean() == pytest.approx(80.5)
    tyndp = _links(dict(base, foreign_price_source="tyndp"))
    # TYNDP: the file's own level (mean 20), no re-centring; it is also the tranche level.
    assert tyndp.links.at["IC_FR", "market_import_level"] == pytest.approx(20.0)
    assert tyndp.links.at["IC_FR", "market_export_level"] == pytest.approx(20.0)
    assert tyndp.links_t.marginal_cost["IC_FR"].tolist() == pytest.approx([10.5, 30.5])


def test_tyndp_on_the_single_basis_uses_a_zero_offset_unless_set():
    markets = {"continent": {"counterparties": ["France"]}}
    pricing = {"source": "derived", "foreign_price_source": "tyndp", "foreign_price_basis": "single",
               "transaction_spread_gbp_per_mwh": 1.0, "volume_tranches": {"markets": markets}}
    n = _links(pricing)
    assert n.links.at["IC_FR", "market_import_level"] == pytest.approx(20.0)
    n = _links(dict(pricing, single_price_level_offsets_gbp_per_mwh={"continent": 2.0}))
    assert n.links.at["IC_FR", "market_import_level"] == pytest.approx(22.0)


# --------------------------------------------------------------------------- tranches

def test_tranche_steps_follow_the_ic1_formula():
    cables = pd.DataFrame({"capacity_mw": [3000.0, 1500.0], "counterparty_country": ["France", "Norway"]})
    config = {"elasticity": 1.0, "tranches": 3, "markets": {
        "continent": {"counterparties": ["France"], "demand_countries": ["France"]},
        "norway": {"counterparties": ["Norway"], "demand_countries": ["Norway"]}}}
    demand = pd.Series({"France": 876.0, "Norway": 219.0})
    table = tp.tranche_steps(cables, {"France": 40.0, "Norway": 30.0}, config, demand)
    # step = e x level / (demand GW) x width GW = 40 / 100 x 1.0
    assert table.at["continent", "import_step"] == pytest.approx(0.4)
    assert table.at["continent", "export_step"] == pytest.approx(0.4)
    assert table.at["norway", "import_step"] == pytest.approx(30.0 / 25.0 * 0.5)


def test_sem_weights_from_a_cable_list():
    table = pd.DataFrame({
        "name": ["Moyle", "East West Interconnector", "Greenlink", "Isle of Man Interconnector", "IFA", "NewCable"],
        "action": ["keep", "keep", "keep", "exclude", "keep", "add"],
        "cp30_mw": [500, 500, 500, 0, 2000, 700],
        "first_year": [None, None, None, None, None, 2031],
        "counterparty_country": [None, None, None, None, None, "Northern Ireland"],
    })
    assert tp.sem_node_weights_from_composition(table, 2030) == {"IE00": 1000.0, "UKNI": 500.0}
    assert tp.sem_node_weights_from_composition(table, 2031) == {"IE00": 1000.0, "UKNI": 1200.0}
    zones = tp.with_sem_weights(tp.ZONE_WEIGHTS_TYNDP2024, {"IE00": 1750.0, "UKNI": 500.0})
    assert zones["Ireland"] == pytest.approx({"IE00": 1750.0 / 2250.0, "UKNI": 500.0 / 2250.0})
    assert zones["France"] == tp.ZONE_WEIGHTS_TYNDP2024["France"]
    # The 2050 route's declared weights are today's register cables, unchanged.
    assert tp.ZONE_WEIGHTS_TYNDP2024["Ireland"] == {"IE00": 1009.0 / 1509.0, "UKNI": 500.0 / 1509.0}
    assert tp.with_sem_weights(tp.ZONE_WEIGHTS_TYNDP2024, None) == tp.ZONE_WEIGHTS_TYNDP2024
