from pathlib import Path

import pandas as pd
import pytest

from scripts.market.ofgem_price_cap import OfgemPriceCapError, calculate_bills, calculate_tariffs, demand_weighted_price, load_snapshot, require_full_calendar_year
from scripts.market.ofgem_hedging import direct_fuel_components, load_hedge_periods

SNAPSHOT = Path(__file__).resolve().parents[2] / "data/market/ofgem_price_cap/2026_Q3"

def test_official_gb_average_reproduces_published_2026_q3_benchmark():
    rates, metadata = load_snapshot(SNAPSHOT)
    tariffs = calculate_tariffs(rates, regions=["GB_average"], payment_methods=["direct_debit"], vat_rate=metadata["vat_rate"])
    bill = calculate_bills(tariffs, [2500]).iloc[0]
    assert bill["consumer_bill_gbp_inc_vat"] == pytest.approx(861.4995)

def test_counterfactual_applies_only_wholesale_delta_to_energy_consumption():
    rates, metadata = load_snapshot(SNAPSHOT)
    base = calculate_bills(calculate_tariffs(rates, regions=["Eastern"], payment_methods=["direct_debit"]), [2500]).iloc[0]
    changed = calculate_bills(calculate_tariffs(rates, regions=["Eastern"], payment_methods=["direct_debit"], mode="counterfactual", modelled_wholesale_price_gbp_per_mwh=111.75160595834542, reference_wholesale_price_gbp_per_mwh=metadata["reference_wholesale_price_gbp_per_mwh"]), [2500]).iloc[0]
    assert changed["consumer_bill_gbp_inc_vat"] - base["consumer_bill_gbp_inc_vat"] == pytest.approx(52.5)

def test_demand_weighted_price_respects_snapshot_duration():
    index = pd.date_range("2026-01-01", periods=2, freq="h")
    assert demand_weighted_price(pd.Series([100.0, 200.0], index=index), pd.Series([10.0, 10.0], index=index), pd.Series([1.0, 3.0], index=index)) == pytest.approx(175.0)

def test_counterfactual_full_year_guard_rejects_partial_period():
    index = pd.date_range("2026-01-01", periods=24, freq="h")
    with pytest.raises(OfgemPriceCapError, match="requires"):
        require_full_calendar_year(index, pd.Series(1.0, index=index))

def test_annualised_2020_snapshot_is_direct_debit_and_reproducible():
    rates, metadata = load_snapshot(Path(__file__).resolve().parents[2] / "data/market/ofgem_price_cap/2020_annual")
    assert set(rates["payment_method"]) == {"direct_debit"}
    bill = calculate_bills(calculate_tariffs(rates, regions=["GB_average"], payment_methods=["direct_debit"], vat_rate=metadata["vat_rate"]), [2500]).iloc[0]
    assert bill["consumer_bill_gbp_inc_vat"] == pytest.approx(530.090546, abs=1e-6)


def test_gb_average_can_use_domestic_meter_weights():
    rates = pd.DataFrame({
        "payment_method": ["direct_debit", "direct_debit"],
        "region": ["Eastern", "Northern"],
        "nil_gbp_per_year_ex_vat": [10.0, 30.0],
        "benchmark_gbp_per_year_ex_vat": [110.0, 230.0],
        "benchmark_kwh": [3100.0, 3100.0],
    })
    weights = pd.DataFrame({
        "ofgem_region": ["Eastern", "Northern"], "domestic_meter_weight": [0.75, 0.25]
    })

    tariff = calculate_tariffs(
        rates, regions=["GB_average"], payment_methods=["direct_debit"], region_weights=weights
    ).iloc[0]

    assert tariff["region"] == "GB average (domestic-meter weighted)"
    assert tariff["nil_gbp_per_year_ex_vat"] == pytest.approx(15.0)
    assert tariff["benchmark_gbp_per_year_ex_vat"] == pytest.approx(140.0)

def test_2020_hedged_wholesale_components_reconcile_to_published_direct_fuel_allowance():
    source = Path(__file__).resolve().parents[2] / "data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv"
    components = direct_fuel_components(load_hedge_periods(source))
    direct_fuel = components.loc[components["component"] == "direct_fuel_allowance"].set_index("cap_period")
    assert direct_fuel.loc["2020_Jan_Mar", "gbp_per_benchmark_customer_ex_vat"] == pytest.approx(200.62671806184008)
    assert direct_fuel.loc["2020_Apr_Sep", "gbp_per_benchmark_customer_ex_vat"] == pytest.approx(184.2102767933549)
    assert direct_fuel.loc["2020_Oct_Dec", "gbp_per_benchmark_customer_ex_vat"] == pytest.approx(154.32526694131108)
