"""Task Y11 (FES 2050 plan): the calendar-2050 Capacity Market, with 2030 left as it was."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.market import capacity_market as cm
from scripts.market import capacity_market_2050 as cm50

needs_register = pytest.mark.skipif(not cm.REGISTER_CMU.exists(), reason="NESO Capacity Market Register not downloaded")
ASSET_INPUTS = Path("resources/public_ownership/cp30_2030_cp30_network_asset_inputs.csv")
RATES_2030 = Path("resources/market/capacity_market/cm_revenue_by_technology.csv")


def test_calendar_weights_generalise_the_2030_weights():
    assert cm50.calendar_weights(2030) == cm.CALENDAR_2030_WEIGHTS
    assert cm50.calendar_weights(2050) == cm50.CALENDAR_2050_WEIGHTS == {"2049/2050": 0.5, "2050/2051": 0.5}


def test_stock_flow_vintages_add_up_and_replace_retirements():
    capacity = pd.Series({2024: 10.0, 2025: 10.0, 2026: 12.0, 2027: 12.0})
    vintages = cm50.stock_flow_vintages(capacity, life=2)
    assert vintages.sum() == pytest.approx(12.0)
    assert vintages.index.min() >= 2026                      # a 2-year life has turned the fleet over
    falling = cm50.stock_flow_vintages(pd.Series({2024: 10.0, 2025: 4.0}), life=10)
    assert falling.sum() == pytest.approx(4.0) and falling.index.min() == 2021   # oldest retired first


def test_fleet_matches_fes_2050_capacity_by_carrier():
    trajectory = cm50.read_fleet_trajectory()
    fleet = cm50.fleet_2050(trajectory)
    for carrier, group in fleet.groupby("carrier"):
        target = float(cm50._series(trajectory, "capacity_mw", carrier)[2050])
        assert group["capacity_mw"].sum() == pytest.approx(target, rel=1e-9), carrier
    assert not fleet["carrier"].isin(["gas", "CHP", "oil"]).any()        # no unabated gas in HT 2050
    assert cm50._series(trajectory, "acs_peak_mw", "demand")[2050] == pytest.approx(120050.0)


def test_attach_reads_the_year_labelled_connection_column():
    assets = pd.DataFrame({"carrier": ["Battery", "Battery", "CCS Gas"], "capacity_mw": [100.0, 100.0, 500.0],
                           "support_type": ["merchant", "merchant", "DPA"]})
    rates = pd.DataFrame({"carrier": ["battery", "CCS Gas"], "connection_mw_2049_50": [100.0, 500.0],
                          "gbp_per_kw_connection_year": [10.0, 20.0]})
    out = cm.attach_capacity_market_revenue(assets, rates)
    assert out["capacity_market_participation_share"].tolist() == [0.5, 0.5, 0.0]    # DPA units cannot hold one
    assert out["capacity_market_revenue_gbp"].sum() == pytest.approx(2 * 100 * 1e3 * 10 * 0.5)
    both = rates.assign(connection_mw_2029_30=[200.0, 0.0])
    assert cm.contracted_connection_column(both) == "connection_mw_2029_30"
    with pytest.raises(ValueError):
        cm.contracted_connection_column(rates.drop(columns="connection_mw_2049_50"))


@pytest.mark.skipif(not (ASSET_INPUTS.exists() and RATES_2030.exists()), reason="2030 register inputs not built")
def test_2030_register_attach_is_unchanged_by_the_2050_carriers():
    assets = pd.read_csv(ASSET_INPUTS, low_memory=False)
    rates = pd.read_csv(RATES_2030)
    out = cm.attach_capacity_market_revenue(assets, rates)
    # The carriers added for 2050 have no 2030 rate row, so they earn nothing in 2030.
    added = out["carrier"].isin(["CAES", "CCS Gas", "Hydrogen", "marine"])
    assert out.loc[added, "capacity_market_revenue_gbp"].sum() == 0.0
    assert out.loc[added, "capacity_market_participation_share"].sum() == 0.0


@pytest.mark.skipif(not ASSET_INPUTS.exists(), reason="2030 register inputs not built")
def test_requirement_calibration_reproduces_from_the_2030_register():
    calibration = cm50.read_calibration()
    computed = cm50.non_cm_derated_2030(pd.read_csv(ASSET_INPUTS, low_memory=False), cm50.read_derating())
    assert computed["supported_mw"] == pytest.approx(calibration["non_cm_supported_derated_2030_mw"], rel=1e-9)
    assert computed["merchant_renewable_mw"] == pytest.approx(
        calibration["non_cm_merchant_renewable_derated_2030_mw"], rel=1e-9)


def test_requirement_rules():
    calibration = cm50.read_calibration()
    ratio = cm50.requirement_mw(calibration, 120050.0, "ratio")
    additive = cm50.requirement_mw(calibration, 120050.0, "additive")
    assert ratio["ratio"] == pytest.approx(ratio["held_2030_31_mw"] / 62090.0)
    assert ratio["requirement_mw"] == pytest.approx(ratio["ratio"] * 120050.0)
    assert additive["requirement_mw"] == pytest.approx(120050.0 + ratio["margin_mw"])


def test_dpa_and_cfd_plant_are_outside_the_auction_while_live():
    derating = cm50.read_derating()
    fleet = pd.DataFrame({"carrier": ["CCS Gas", "CCS Gas", "wind_offshore", "Battery"],
                          "build_year": [2040, 2032, 2040, 2049], "capacity_mw": [1000.0, 1000.0, 1000.0, 1000.0],
                          "duration_hours": [np.nan, np.nan, np.nan, 2.0], "energy_mwh": [np.nan] * 4})
    out = cm50.classify_fleet(fleet, derating, cm50.Settings2050())
    assert out["support_live_share"].tolist() == [1.0, 0.0, 1.0, 0.0]         # 2032 CCS: its 15-year DPA ended
    assert out.loc[1, "eligible_derated_mw"] == pytest.approx(1000 * 0.9048)
    assert out["category"].tolist() == ["existing", "existing", "existing", "new_build"]
    assert out.loc[3, "derating_factor"] == pytest.approx(0.2231)


@needs_register
def test_2050_auction_adds_up_and_reports_its_cases():
    from scripts.market.build_capacity_market import auction_model

    model = auction_model(cm.read_register())
    result = cm50.run_2050(model, cm.read_cpi())
    summary, revenue = result["summary"], result["revenue"]
    assert revenue["calendar_2050_revenue_gbp"].sum() == pytest.approx(summary["calendar_2050_cost_gbp"], abs=1.0)
    assert {"connection_mw_2049_50", "revenue_2049_50_gbp", "revenue_2050_51_gbp",
            "gbp_per_kw_connection_year"} <= set(revenue.columns)
    assert summary["contract_book_gbp"] == 0.0
    assert set(summary["cases"]) == set(cm50.CASES)
    assert summary["non_cm_derated_mw_by_carrier"]["CCS Gas"] > 0 and summary["non_cm_derated_mw_by_carrier"]["Hydrogen"] > 0
    # Without V2G counted, the fleet cannot meet the requirement and the auction clears at the cap.
    assert summary["cases"]["v2g_not_counted"]["shortfall"]
    assert summary["cases"]["v2g_not_counted"]["price_gbp_per_kw_year_auction_money"] == pytest.approx(75.0)
    # The 2030 build is untouched by the 2050 path.
    assert model["checks"]["T-4 2029/30"]["actual_price"] == 27.10
