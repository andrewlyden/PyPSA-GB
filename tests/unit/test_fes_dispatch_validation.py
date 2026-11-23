"""Unit tests for the Y8 post-solve validation tooling of the 2050 market year.

- ``scripts/FES/fes_dispatch_validation.py``: interconnector energy measured on the links,
  GB side (B9), per cable and per hour; the WARN-must-be-explained guard.
- ``scripts/market/bm_storage_churn.py``: simultaneous energy, the round-trip loss it burns,
  and the negative-nodal-price split.
- ``scripts/market/ic1_month_diagnostics.py``: hours at capacity and the trade value, which
  must be the LP's own valuation (``interconnector_trade_cost``).
"""

import numpy as np
import pandas as pd
import pytest

from scripts.FES import fes_dispatch_validation as hv
from scripts.market import bm_storage_churn as churn
from scripts.market import ic1_month_diagnostics as ic1

IDX = pd.date_range("2050-01-01", periods=4, freq="h")


def _links():
    return pd.DataFrame({
        "carrier": ["DC", "DC", "electrolysis", "H2_turbine"],
        "bus0": ["External_France", "External_Norway", "GB", "H2"],
        "bus1": ["GB", "GB2", "H2", "GB"],
        "p_nom": [100.0, 50.0, 10.0, 20.0],
        "p_min_pu": [-1.0, -1.0, 0.0, 0.0],
        "p_max_pu": [1.0, 1.0, 1.0, 1.0],
        "marginal_cost": [90.0, 60.0, 0.0, 0.0],
        "efficiency": [0.975, 0.975, 0.8, 0.4],
        "foreign_market_pricing": ["piecewise_bidirectional", "piecewise_bidirectional", "", ""],
        "interconnector_name": ["FR cable", "NO cable", "", ""],
        "counterparty_country": ["France", "Norway", "", ""],
        "market_separate_import_export_prices": [True, True, False, False],
        "market_export_bid_offset": [50.0, 30.0, 0.0, 0.0],
        "market_spread_half": [0.5, 0.5, 0.0, 0.0],
        "market_circulation_epsilon": [0.01, 0.01, 0.0, 0.0],
    }, index=["IC_FR", "IC_NO", "elec_1", "turb_1"])


def test_interconnector_energy_is_gb_side_per_cable_and_netted_per_hour():
    p0 = pd.DataFrame({"IC_FR": [100.0, -40.0, 0.0, 20.0], "IC_NO": [-50.0, 10.0, 0.0, 0.0]}, index=IDX)
    out = hv.interconnector_energy(p0, _links(), np.ones(4))
    assert out["import_mwh"]["IC_FR"] == pytest.approx(120 * 0.975)
    assert out["export_mwh"]["IC_FR"] == pytest.approx(40 * 0.975)
    assert out["import_mwh"]["IC_NO"] == pytest.approx(10 * 0.975)
    assert out["export_mwh"]["IC_NO"] == pytest.approx(50 * 0.975)
    # hour 1: +100 FR, -50 NO -> net +50; hour 2: -40 + 10 -> -30; hour 4: +20
    assert out["net_import_mwh"] == pytest.approx((50 + 20) * 0.975)
    assert out["net_export_mwh"] == pytest.approx(30 * 0.975)


def test_hydrogen_links_count_electricity_in_and_out():
    p0 = pd.DataFrame({"elec_1": [10.0, 0.0, 5.0, 0.0], "turb_1": [0.0, 20.0, 0.0, 10.0]}, index=IDX)
    out = hv._hydrogen_links(p0, _links(), np.ones(4))
    assert out["electrolysis_elec_mwh"] == pytest.approx(15.0)
    assert out["electrolysis_h2_mwh"] == pytest.approx(12.0)
    assert out["h2_turbine_h2_mwh"] == pytest.approx(30.0)
    assert out["h2_turbine_elec_mwh"] == pytest.approx(12.0)


def test_a_warn_without_an_explainer_is_flagged_unexplained():
    row = pd.Series({"component": "something_new", "item": "x", "status": "WARN"})
    text = hv.explain(row, {}, {})
    assert text.startswith("UNEXPLAINED")
    table = pd.DataFrame([{"status": "WARN", "explanation": text}, {"status": "WARN", "explanation": "because"},
                          {"status": "PASS", "explanation": ""}])
    assert len(hv.unexplained(table)) == 1


def _storage(eff_store, eff_dispatch, carriers):
    return pd.DataFrame({"carrier": carriers, "efficiency_store": eff_store, "efficiency_dispatch": eff_dispatch},
                        index=[f"u{i}" for i in range(len(carriers))])


def test_churn_counts_overlap_and_the_round_trip_it_burns():
    attrs = _storage([0.92, 1.0], [0.92, 1.0], ["Battery", "EV_smart_charging"])
    p_dispatch = pd.DataFrame({"u0": [10.0, 0.0, 5.0, 0.0], "u1": [2.0, 0.0, 0.0, 0.0]}, index=IDX)
    p_store = pd.DataFrame({"u0": [4.0, 3.0, 5.0, 0.0], "u1": [2.0, 1.0, 0.0, 0.0]}, index=IDX)
    price = np.array([[-10.0, 5.0], [0.0, 0.0], [20.0, 0.0], [0.0, 0.0]])
    units = churn.churn_frame(p_dispatch, p_store, pd.Series(1.0, index=IDX), attrs, bus_price=price)
    assert units.at["u0", "simultaneous_mwh"] == pytest.approx(4.0 + 5.0)
    assert units.at["u0", "energy_lost_mwh"] == pytest.approx(9.0 * (1 / 0.92 - 0.92))
    assert units.at["u0", "simultaneous_negative_price_mwh"] == pytest.approx(4.0)
    assert units.at["u0", "pair_unit_hours"] == 2
    # Unit efficiencies (consumer flexibility): a pair burns nothing.
    assert units.at["u1", "simultaneous_mwh"] == pytest.approx(2.0)
    assert units.at["u1", "energy_lost_mwh"] == pytest.approx(0.0)
    table = churn.by_carrier(units).set_index("carrier")
    assert table.at["Battery", "simultaneous_share_of_charge"] == pytest.approx(9.0 / 12.0)


def test_ic1_cable_table_measures_capacity_hours_and_values_trade_like_the_lp():
    links = _links().loc[["IC_FR"]]
    flows = pd.DataFrame({"IC_FR": [100.0, 99.5, -50.0, 0.0]}, index=IDX)
    data = {"month": "Jan", "links": links, "ask_t": pd.DataFrame(), "flows": {"bm": flows, "stage1": flows},
            "weights": pd.Series(1.0, index=IDX), "p_max_t": pd.DataFrame(), "p_min_t": pd.DataFrame()}
    row = ic1.cable_table(data, "stage1").iloc[0]
    assert row["share_hours_at_import_capacity"] == pytest.approx(0.5)  # 100 and 99.5 >= 99% of 100
    assert row["share_hours_at_export_capacity"] == pytest.approx(0.0)
    assert row["import_gwh"] * 1e3 == pytest.approx(199.5 * 0.975)
    assert row["import_cost_gbp"] == pytest.approx(199.5 * 90.0)
    # exports earn efficiency^2 x the export bid (ask - offset), as the LP charges (item D2)
    assert row["export_income_gbp"] == pytest.approx(50.0 * 0.975 ** 2 * 40.0)
    assert row["net_trade_value_paid_abroad_gbp"] == pytest.approx(199.5 * 90.0 - 50.0 * 0.975 ** 2 * 40.0)


def test_ic1_bm_retrading_adds_the_counter_trade_premium():
    links = _links().loc[["IC_FR"]].assign(market_counter_trade_premium=100.0)
    s1 = pd.DataFrame({"IC_FR": [100.0, 0.0, 0.0, 0.0]}, index=IDX)
    bm = pd.DataFrame({"IC_FR": [60.0, 0.0, 0.0, 0.0]}, index=IDX)
    data = {"month": "Jan", "links": links, "ask_t": pd.DataFrame(), "flows": {"bm": bm, "stage1": s1},
            "weights": pd.Series(1.0, index=IDX), "p_max_t": pd.DataFrame(), "p_min_t": pd.DataFrame()}
    row = ic1.cable_table(data, "bm").iloc[0]
    assert row["counter_trade_premium_gbp"] == pytest.approx(40.0 * 100.0)
    assert row["retrading_cost_gbp"] == pytest.approx(60 * 90.0 - 100 * 90.0 + 4000.0)
