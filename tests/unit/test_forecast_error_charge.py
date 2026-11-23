"""The borrowed forecast-error charge: bases, price basis, the discounting reading, windows."""

import pytest

from scripts.market.forecast_error_charge import (
    BASES,
    DEFAULT_BASIS,
    DEFAULT_SETTINGS,
    MODEL_PRICE_YEAR,
    PRICE_FACTOR_2024_TO_2025,
    adjusted_saving_gbp_bn,
    annuity_factor,
    charge_for_period_gbp_m,
    forecast_error_charge_gbp_bn,
    fti_cost_per_year_gbp_bn,
    gbp_per_household,
    settings_from_config,
)

GROSS, DEMAND = 1.448, 307.9


def test_default_is_renewable_capacity_on_the_npv_reading_in_model_money():
    assert DEFAULT_BASIS == "renewable_capacity"
    assert MODEL_PRICE_YEAR == 2024
    # 7.9 bn (2022) x 8.8/8.2 to 2024 over the 11-year 3.5% annuity, x the fleet ratio 1.10 / 1.354 = 0.812 (our
    # fleet over FTI's discount-weighted 2030-2040 fleet; user decision, 11 Nov 2026).
    assert forecast_error_charge_gbp_bn(GROSS, DEMAND) == pytest.approx(0.765, abs=0.005)
    assert adjusted_saving_gbp_bn(GROSS, DEMAND) == pytest.approx(0.683, abs=0.005)


def test_the_fleet_growth_is_the_tables():
    from scripts.market.forecast_error_charge import FLEET_RATIO, FTI_FLEET_GROWTH, fti_fleet_growth

    assert fti_fleet_growth() == pytest.approx(FTI_FLEET_GROWTH, abs=5e-5)
    assert FLEET_RATIO == round(1.10 / FTI_FLEET_GROWTH, 3) == 0.812


def test_report_year_prices_use_the_bill_rebasing_factor():
    model = forecast_error_charge_gbp_bn(GROSS, DEMAND, price_year=2024)
    report = forecast_error_charge_gbp_bn(GROSS, DEMAND, price_year=2025)
    assert report / model == pytest.approx(PRICE_FACTOR_2024_TO_2025)
    with pytest.raises(ValueError):
        forecast_error_charge_gbp_bn(GROSS, DEMAND, price_year=2023)


def test_simple_total_reading_is_about_18_percent_lower():
    npv = forecast_error_charge_gbp_bn(GROSS, DEMAND)
    total = forecast_error_charge_gbp_bn(GROSS, DEMAND, as_simple_total=True)
    assert total < npv
    assert total / npv == pytest.approx(annuity_factor() / 11, rel=1e-6)
    assert total / npv == pytest.approx(0.818, abs=0.005)


def test_demand_basis_is_ftis_cost_per_mwh_on_the_same_reading():
    # FTI's annual cost over 300 TWh, applied to ours: the capacity charge without the fleet ratio, x demand/300.
    per_twh = fti_cost_per_year_gbp_bn() / 300.0
    assert forecast_error_charge_gbp_bn(GROSS, DEMAND, basis="demand") == pytest.approx(per_twh * DEMAND)
    simple = forecast_error_charge_gbp_bn(GROSS, DEMAND, basis="demand", as_simple_total=True)
    assert simple == pytest.approx(fti_cost_per_year_gbp_bn(as_simple_total=True) / 300.0 * DEMAND)


def test_the_three_bases_are_the_same_order_of_magnitude():
    charges = {b: forecast_error_charge_gbp_bn(GROSS, DEMAND, basis=b) for b in BASES}
    assert charges["gross_share"] == pytest.approx(0.710, abs=0.005)
    # Since 11 Nov 2026 (FTI's 2030-2040 fleet) the capacity basis sits between the other two.
    assert charges["demand"] > charges["renewable_capacity"] > charges["gross_share"]
    assert all(0.5 < c < 1.2 for c in charges.values())


def test_capacity_basis_ignores_demand_and_demand_basis_ignores_the_saving():
    a = forecast_error_charge_gbp_bn(GROSS, 200.0, basis="renewable_capacity")
    b = forecast_error_charge_gbp_bn(GROSS, 400.0, basis="renewable_capacity")
    assert a == b
    c = forecast_error_charge_gbp_bn(0.5, DEMAND, basis="demand")
    d = forecast_error_charge_gbp_bn(5.0, DEMAND, basis="demand")
    assert c == d


def test_gross_share_scales_with_the_saving_and_is_never_negative():
    half = forecast_error_charge_gbp_bn(GROSS / 2, DEMAND, basis="gross_share")
    full = forecast_error_charge_gbp_bn(GROSS, DEMAND, basis="gross_share")
    assert half == pytest.approx(full / 2)
    assert forecast_error_charge_gbp_bn(-1.0, DEMAND, basis="gross_share") == 0.0


def test_fleet_ratio_is_applied_linearly_and_a_bad_basis_is_refused():
    one = forecast_error_charge_gbp_bn(GROSS, DEMAND, fleet_ratio=1.0)
    assert one == pytest.approx(fti_cost_per_year_gbp_bn())
    assert forecast_error_charge_gbp_bn(GROSS, DEMAND, fleet_ratio=1.10) == pytest.approx(one * 1.10)
    with pytest.raises(ValueError):
        forecast_error_charge_gbp_bn(GROSS, DEMAND, basis="vibes")


@pytest.mark.parametrize("basis", BASES)
def test_twelve_windows_add_to_the_annual_charge(basis):
    """The year settlement sums its months, so a window's charge must be additive."""
    hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    demand = [DEMAND * h / 8760 for h in hours]
    gross = [GROSS * 1e3 * h / 8760 for h in hours]
    months = sum(charge_for_period_gbp_m(g, d, h, basis=basis) for g, d, h in zip(gross, demand, hours))
    assert months == pytest.approx(forecast_error_charge_gbp_bn(GROSS, DEMAND, basis=basis) * 1e3)


def test_a_week_on_the_capacity_basis_is_its_share_of_the_year():
    week = charge_for_period_gbp_m(50.0, 6.0, 168)
    assert week == pytest.approx(forecast_error_charge_gbp_bn(0.0, 0.0) * 1e3 * 168 / 8760)


def test_config_settings_are_validated_and_default_to_the_module_constants():
    assert settings_from_config(None) == DEFAULT_SETTINGS
    assert settings_from_config({"basis": "demand"})["basis"] == "demand"
    for bad in ({"basis": "vibes"}, {"fleet_ratio": 0}, {"enabled": "yes"}, {"price_year": 2019}, {"typo": 1}):
        with pytest.raises(ValueError):
            settings_from_config(bad)


def test_per_household_uses_the_bill_recovery_denominator_never_a_household_count():
    """House rule 8: GBP over final demand x rebasing x 3.1 MWh, not over 28 million households."""
    bill = {"final_demand_recovery_denominator_mwh": 318.29e6, "model_rebasing_factor": 1.0384, "benchmark_kwh": 3100.0}
    value = gbp_per_household(1.0e9, bill)
    assert value == pytest.approx(1.0e9 * 1.0384 / 318.29e6 * 3.1)
    assert value != pytest.approx(1.0e9 / 28e6, rel=0.05)
