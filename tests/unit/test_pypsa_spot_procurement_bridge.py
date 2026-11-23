import pandas as pd
import pytest

from scripts.consumer_bill.build_pypsa_spot_procurement_bridge import (
    BASE_COMPONENT,
    DIRECT_ENERGY_COMPONENT,
    PEAK_COMPONENT,
    build_pypsa_spot_procurement_bridge,
)


def _impact(prices):
    return pd.DataFrame({
        "cap_period": ["Q1", "Q2"],
        "modelled_demand_mwh": [100.0, 100.0],
        "modelled_demand_weighted_clearing_price_gbp_per_mwh": prices,
    })


def test_standard_historical_bridge_uses_pypsa_price_and_preserves_base_peak_split():
    register = pd.DataFrame([
        {"component": BASE_COMPONENT, "annual_gbp_per_3100kwh_ex_vat": 100.0},
        {"component": PEAK_COMPONENT, "annual_gbp_per_3100kwh_ex_vat": 100.0},
    ])
    periods = pd.DataFrame({
        "cap_period": ["Q1", "Q2"], "delivery_start": ["2020-01-01", "2020-07-01"],
        "delivery_end": ["2020-06-30", "2020-12-31"], "base_share": [0.5, 0.5],
        "peak_share": [0.5, 0.5], "forward_index_gbp_per_mwh": [100.0, 100.0],
    })
    bridge, diagnostics, summary = build_pypsa_spot_procurement_bridge(register, periods, _impact([20.0, 40.0]))
    assert bridge.set_index("bill_component").loc[BASE_COMPONENT, "candidate_allowance_gbp_per_3100kwh_ex_vat"] == pytest.approx(46.58469945)
    assert bridge.set_index("bill_component").loc[PEAK_COMPONENT, "candidate_allowance_gbp_per_3100kwh_ex_vat"] == pytest.approx(46.58469945)
    assert diagnostics["difference_gbp_per_3100kwh_ex_vat"].tolist() == pytest.approx([-248.0, -186.0])
    assert summary["calibration"]["same_year_fitted"] is False


def test_2024_style_bridge_replaces_only_direct_energy_component():
    register = pd.DataFrame([
        {"component": DIRECT_ENERGY_COMPONENT, "annual_gbp_per_3100kwh_ex_vat": 100.0},
    ])
    periods = pd.DataFrame({
        "cap_period": ["Q1", "Q2"], "delivery_start": ["2024-01-01", "2024-07-01"],
        "delivery_end": ["2024-06-30", "2024-12-31"], "base_share": [1.0, 1.0],
        "peak_share": [0.0, 0.0], "forward_index_gbp_per_mwh": [10.0, 10.0],
        "wholesale_detail_level": ["published_3a_direct_fuel_components"] * 2,
        "direct_fuel_energy_cost_gbp_per_3100kwh": [100.0, 100.0],
    })
    bridge, _, _ = build_pypsa_spot_procurement_bridge(register, periods, _impact([20.0, 40.0]))
    assert bridge.iloc[0].bill_component == DIRECT_ENERGY_COMPONENT
    assert bridge.iloc[0].candidate_allowance_gbp_per_3100kwh_ex_vat == pytest.approx(93.16939891)
