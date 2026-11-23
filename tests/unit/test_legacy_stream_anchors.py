"""Every stream of the legacy valuation on the solved years' path, and a horizon at each plant's life (user, 11 Nov 2026).

``price_anchors --register`` gives each solved year's streams per MW by technology; ``anchor_streams: all`` scales
each plant's own model-year streams by its technology's path (output, wholesale, balancing, fuel and carbon,
Capacity Market), a CfD's top-up following output and price, the RO following output. ``horizon_year: life`` runs
the cash flows to the last plant's retirement.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import legacy_valuation as lv
from scripts.public_ownership.price_anchors import MIN_MW, build, stream_anchor


def _register(scale):
    return pd.DataFrame({
        "case_id": ["private", "private", "private", "public_ownership"],
        "carrier": ["wind_offshore", "wind_offshore", "Pumped Storage Hydroelectricity", "wind_offshore"],
        "capacity_mw": [100.0, 300.0, 2 * MIN_MW, 999.0],
        "wholesale_mwh": [4e5 * scale, 1.2e6 * scale, -1e4, 1.0],
        "wholesale_revenue_gbp": [1e7, 3e7, 2e6, 1.0],
        "balancing_settlement_gbp": [4e6 * scale, 0.0, 0.0, 1.0],
        "fuel_carbon_gbp": [0.0, 0.0, 0.0, 0.0],
        "capacity_market_revenue_gbp": [0.0, 0.0, 1e6, 0.0]})


def test_stream_anchor_is_per_mw_on_the_private_case_with_one_name_per_technology():
    out = stream_anchor(_register(1.0), 2030).set_index("carrier")
    assert out.loc["wind_offshore", "capacity_mw"] == 400.0
    assert out.loc["wind_offshore", "mwh_per_mw"] == pytest.approx(4000.0)
    assert out.loc["wind_offshore", "wholesale_gbp_per_mw"] == pytest.approx(1e5)
    assert out.loc["wind_offshore", "balancing_gbp_per_mw"] == pytest.approx(1e4)
    assert out.loc["pumped_hydro", "capacity_market_gbp_per_mw"] == pytest.approx(1e4)


def test_a_technology_below_the_minimum_capacity_gives_no_stream_anchor():
    reg = _register(1.0)
    reg.loc[2, "capacity_mw"] = MIN_MW / 2
    assert "pumped_hydro" not in set(stream_anchor(reg, 2030)["carrier"])


def test_build_joins_the_stream_anchors_to_the_price_anchors():
    settle = pd.DataFrame({"carrier": ["wind_offshore"], "wholesale_mwh": [1e6], "wholesale_revenue_gbp": [3e7]})
    out = build({2030: settle, 2050: settle}, None, {2030: _register(1.0), 2050: _register(0.5)})
    off = out.loc[out["carrier"].eq("wind_offshore")].set_index("year")
    assert off.loc[2030, "wholesale_gbp_per_mwh"] == pytest.approx(30.0)
    assert off.loc[2050, "mwh_per_mw"] / off.loc[2030, "mwh_per_mw"] == pytest.approx(0.5)
    later_only = build({2030: settle, 2050: settle}, None, {2050: _register(0.5)})
    assert later_only.set_index(["carrier", "year"])["mwh_per_mw"].isna()[("wind_offshore", 2030)]
    with pytest.raises(ValueError, match="no settlement"):
        build({2030: settle}, None, {2050: _register(1.0)})


def test_the_model_years_streams_are_the_valued_registers_own():
    anchors = _anchors()
    anchors.loc[anchors["year"].eq(2050), "mwh_per_mw"] = 1.0           # a stale model-year figure is replaced
    reg = pd.DataFrame({"case_id": "private", "carrier": ["wind_offshore", "Battery"], "capacity_mw": [100.0, 100.0],
                        "wholesale_mwh": [1e5, -5e3], "wholesale_revenue_gbp": [1e4, 1e3],
                        "balancing_settlement_gbp": [1e3, 1e2], "fuel_carbon_gbp": [0.0, 0.0],
                        "capacity_market_revenue_gbp": [0.0, 400.0]})
    settings = _settings("all")
    out = lv.with_model_year_streams(anchors, reg, 2050, settings).set_index(["carrier", "year"])
    assert out.loc[("wind_offshore", 2050), "mwh_per_mw"] == pytest.approx(1000.0)
    assert out.loc[("wind_offshore", 2060), "mwh_per_mw"] == pytest.approx(1100.0)
    assert lv.with_model_year_streams(anchors, reg, 2050, _settings("price")) is anchors


def _anchors():
    return pd.DataFrame({"carrier": ["wind_offshore"] * 2 + ["Battery"] * 2, "year": [2050, 2060] * 2,
                         "wholesale_gbp_per_mwh": [40.0, 20.0, np.nan, np.nan],
                         "mwh_per_mw": [1000.0, 1100.0, -50.0, -60.0],
                         "wholesale_gbp_per_mw": [100.0, 55.0, 10.0, 20.0],
                         "balancing_gbp_per_mw": [10.0, 5.0, 1.0, 1.0],
                         "fuel_carbon_gbp_per_mw": [0.0, 0.0, 0.0, 0.0],
                         "capacity_market_gbp_per_mw": [0.0, 0.0, 4.0, 2.0]})


def test_stream_ratios_run_straight_and_a_non_positive_base_keeps_its_own_figure():
    years = np.arange(2050, 2071)
    r = lv.anchored_stream_ratios(["wind_offshore", "Battery", "CCGT"], years, _anchors(), 2050)
    assert r["mwh_per_mw"][0, 5] == pytest.approx(1.05)
    assert r["wholesale_gbp_per_mw"][0, 10] == pytest.approx(0.55) == r["wholesale_gbp_per_mw"][0, 20]
    assert (r["mwh_per_mw"][1] == 1.0).all()                 # storage: net charging, output ratio held
    assert r["capacity_market_gbp_per_mw"][1, 10] == pytest.approx(0.5)
    assert all((r[c][2] == 1.0).all() for c in lv.STREAM_COLUMNS)   # off the path


def test_the_stream_ratios_need_the_stream_columns():
    with pytest.raises(ValueError, match="price_anchors --register"):
        lv.anchored_stream_ratios(["wind_offshore"], np.arange(2050, 2052),
                                  _anchors()[["carrier", "year", "wholesale_gbp_per_mwh"]], 2050)


def _row(**over):
    row = {"case_id": "private", "asset_id": "W1", "component_type": "Generator", "carrier": "wind_offshore",
           "finance_vintage": "legacy", "support_type": "CfD", "cfd_round": "AR4", "cfd_award_year": 2049.0,
           "capacity_mw": 10.0, "commissioning_year": 2049.0, "market_revenue_gbp": 140.0,
           "wholesale_revenue_gbp": 100.0, "balancing_settlement_gbp": 40.0, "cfd_support_gbp": 50.0,
           "cfd_curtailment_compensation_gbp": 30.0, "roc_support_gbp": 0.0, "fuel_carbon_gbp": 0.0,
           "annual_fixed_om_gbp": 0.0, "wacc": 0.07, "model_year": 2050}
    row.update(over)
    return pd.DataFrame([row])


def _settings(streams, **over):
    s = lv.settings_from_config(None, model_year=2050)
    return {**s, "price_path": lv.PRICE_PATH_ANCHORED, "price_anchors": _anchors(), "anchor_streams": streams,
            "horizon_year": 2070, **over}


def _profile(rows, settings):
    shares = pd.DataFrame(1.0, index=range(2050, 2200), columns=[lv.RO_SHARE_ALL])
    flows, expiry = lv.legacy_cash_flow_profile(rows, settings, shares)
    years = np.arange(2050, int(settings["horizon_year"]) + 1)
    return {k: pd.Series(v[0], index=years) for k, v in flows.items()}, expiry


def test_the_default_is_the_price_only_path():
    assert lv.settings_from_config(None, model_year=2050)["anchor_streams"] == "price"
    _, expiry = _profile(_row(), _settings("price"))
    assert expiry["anchor_streams"].iloc[0] == "price"


def test_a_cfd_plant_earns_strike_times_output_and_its_compensation_follows_balancing():
    flows, expiry = _profile(_row(), _settings("all"))
    assert expiry["anchor_streams"].iloc[0] == "all"
    y, v, w, b = 2055, 1.05, 0.775, 0.75
    assert flows["cfd_top_up"][y] != 0.0
    # Wholesale plus top-up follow output alone (strike x output); compensation and the rest of the balancing
    # settlement follow the balancing path.
    total = flows["net_operating_cash"][y] + flows["cfd_top_up"][y]
    assert total == pytest.approx(v * (100.0 + 50.0) + 40.0 * b)
    assert flows["cfd_top_up"][y] == pytest.approx(v * 50.0 + 100.0 * (v - w))


def test_after_the_contract_the_compensation_is_not_cut_twice():
    flows, _ = _profile(_row(), _settings("all"))
    after = flows["cfd_top_up"].index[(flows["cfd_top_up"] == 0.0) & (flows["net_operating_cash"] != 0.0)]
    assert len(after)
    y = int(after[0])
    w = float(np.interp(y, [2050, 2060], [100.0, 55.0])) / 100.0
    b = float(np.interp(y, [2050, 2060], [10.0, 5.0])) / 10.0
    # A merchant after expiry: its wholesale on the path plus the balancing that was not compensation.
    assert flows["net_operating_cash"][y] == pytest.approx(100.0 * w + 10.0 * b)


def test_storage_keeps_its_output_and_follows_its_revenue_streams():
    row = _row(asset_id="B1", component_type="StorageUnit", carrier="Battery", support_type=None, cfd_round=None,
               cfd_award_year=np.nan, commissioning_year=2045.0, market_revenue_gbp=12.0,
               wholesale_revenue_gbp=10.0, balancing_settlement_gbp=2.0, cfd_support_gbp=0.0,
               cfd_curtailment_compensation_gbp=0.0, capacity_market_revenue_gbp=8.0)
    flows, _ = _profile(row, _settings("all"))
    # 2052: wholesale 1.2 x, balancing flat, Capacity Market 0.9 x (the battery retires before 2060).
    assert flows["net_operating_cash"][2052] == pytest.approx(10.0 * 1.2 + 2.0 + 8.0 * 0.9)


def test_unknown_anchor_streams_and_a_life_horizon_are_checked():
    with pytest.raises(ValueError, match="anchor_streams"):
        lv.settings_from_config({"anchor_streams": "most"}, model_year=2050)
    assert lv.settings_from_config({"horizon_year": lv.HORIZON_LIFE}, model_year=2050)["horizon_year"] == "life"


# ---- today's prices (user, 11 Nov 2026) ----

def test_today_level_holds_the_earliest_anchor_for_every_year():
    anchors = pd.DataFrame({"carrier": ["wind_offshore"] * 3 + ["Battery"], "year": [2026, 2030, 2050, 2030],
                            "wholesale_gbp_per_mwh": [60.0, 30.0, 20.0, np.nan]})
    r = lv.today_price_ratios(["wind_offshore", "Battery", "CCGT"], 5, anchors, 2030)
    assert r.shape == (3, 5)
    assert (r[0] == pytest.approx(2.0))
    assert (r[1] == 1.0).all() and (r[2] == 1.0).all()


def test_today_level_moves_the_price_alone_and_the_cfd_against_it():
    anchors = _anchors().copy()
    anchors.loc[anchors["carrier"].eq("wind_offshore") & anchors["year"].eq(2050), "wholesale_gbp_per_mwh"] = 40.0
    today = pd.DataFrame({"carrier": ["wind_offshore"], "year": [2046], "wholesale_gbp_per_mwh": [80.0]})
    settings = _settings("all", price_path=lv.PRICE_PATH_TODAY, price_anchors=pd.concat([anchors, today]))
    flows, expiry = _profile(_row(), settings)
    assert expiry["price_path"].iloc[0] == lv.PRICE_PATH_TODAY
    assert expiry["anchor_streams"].iloc[0] == "price"
    # Wholesale doubles in every year (80 / 40); the CfD top-up falls by the same amount while it runs.
    y = 2055
    assert flows["cfd_top_up"][y] == pytest.approx(50.0 - 100.0)
    assert flows["net_operating_cash"][y] + flows["cfd_top_up"][y] == pytest.approx(100.0 + 50.0 + 40.0)


def test_today_level_is_a_lever_value_and_a_named_case():
    from scripts.public_ownership import levers

    assert lv.PRICE_PATH_TODAY in levers.LEVERS["legacy_price_path"].values
    assert lv.PRICE_PATH_TODAY in lv.PRICE_PATHS
