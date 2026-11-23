import pandas as pd
import pytest

from scripts.consumer_bill.build_network_cost_layers import (
    apply_tariff_to_profile,
    build_tariff_layers,
    build_neso_tnuos_comparator,
    project_allowed_revenue,
)


def _regional():
    return pd.DataFrame({
        "pes_id": [10, 10, 10, 10], "ofgem_region": ["Eastern"] * 4,
        "component": ["TNUoS", "DUoS unit charges", "DUoS fixed charges", "BSUoS"],
        "annual_gbp_per_3100kwh_ex_vat": [31.0, 62.0, 36.6, 15.5],
    })


def test_tariff_layers_separate_unit_standing_and_losses():
    tariffs, summary = build_tariff_layers(
        _regional(), pd.DataFrame({"pes_id": [10], "domestic_meter_weight": [1.0]}), 0.02
    )
    fixed = tariffs.loc[tariffs["component"].eq("DUoS fixed charges")].iloc[0]
    tnuos = tariffs.loc[tariffs["component"].eq("TNUoS")].iloc[0]
    assert fixed["standing_gbp_per_day_ex_vat"] == pytest.approx(0.1)
    assert fixed["unit_gbp_per_kwh_ex_vat"] == 0.0
    assert tnuos["unit_gbp_per_kwh_ex_vat"] == pytest.approx(0.01)
    assert summary["transmission_loss_factor"] == pytest.approx(1 / 0.98)


def test_time_band_engine_accepts_replacement_tariff_rows():
    tariffs, _ = build_tariff_layers(
        _regional(), pd.DataFrame({"pes_id": [10], "domestic_meter_weight": [1.0]}), 0.0
    )
    tariffs = tariffs.loc[tariffs["component"].eq("DUoS unit charges")].copy()
    tariffs["time_band"] = ["peak"]
    tariffs["unit_gbp_per_kwh_ex_vat"] = [0.2]
    profile = pd.DataFrame({
        "pes_id": [10], "tariff_class": ["single_rate"], "time_band": ["peak"],
        "consumption_kwh": [10.0], "customer_days": [0.0],
    })
    priced = apply_tariff_to_profile(tariffs, profile)
    assert priced["charge_gbp_ex_vat"].item() == pytest.approx(2.0)


def test_future_projection_uses_sales_for_unit_and_accounts_for_fixed():
    tariffs, _ = build_tariff_layers(
        _regional(), pd.DataFrame({"pes_id": [10], "domestic_meter_weight": [1.0]}), 0.0
    )
    projected, summary = project_allowed_revenue(tariffs, {
        "target_year": 2030, "tnuos_allowed_revenue_multiplier": 1.5,
        "duos_unit_allowed_revenue_multiplier": 1.2, "duos_fixed_allowed_revenue_multiplier": 1.1,
        "bsuos_recovery_multiplier": 1.0, "domestic_sales_multiplier": 1.25,
        "domestic_accounts_multiplier": 1.1,
    })
    tnuos = projected.loc[projected["component"].eq("TNUoS")].iloc[0]
    fixed = projected.loc[projected["component"].eq("DUoS fixed charges")].iloc[0]
    assert tnuos["projection_multiplier"] == pytest.approx(1.2)
    assert fixed["projection_multiplier"] == pytest.approx(1.0)
    assert summary["target_year"] == 2030


def test_neso_tnuos_comparator_prefers_final_then_latest_forecast():
    source = pd.DataFrame({
        "Publication": ["Forecast", "Final", "Forecast", "Forecast"],
        "Year_FY": [2021, 2021, 2030, 2030],
        "Published_Date": ["01/01/2020", "31/01/2020", "20/06/2024", "12/09/2025"],
        "Zone_No": [9, 9, 6, 6],
        "Zone_Name": ["Eastern", "Eastern", "N Wales & Mersey", "N Wales & Mersey"],
        "NHHTariff(Floored)_p/kWh": [2.0, 3.0, 0.5, 0.7],
    })
    eastern_weights = pd.DataFrame({"pes_id": [10], "ofgem_region": ["Eastern"], "domestic_meter_weight": [1.0]})
    historic, _ = build_neso_tnuos_comparator(source, eastern_weights, 2021)
    assert historic["publication"].item() == "Final"
    assert historic["nhh_tariff_gbp_per_kwh"].item() == pytest.approx(0.03)

    wales_weights = pd.DataFrame({"pes_id": [13], "ofgem_region": ["N Wales and Mersey"], "domestic_meter_weight": [1.0]})
    outlook, _ = build_neso_tnuos_comparator(source, wales_weights, 2030)
    assert outlook["published_date"].item() == pd.Timestamp("2025-09-12")
    assert outlook["nhh_tariff_p_per_kwh"].item() == pytest.approx(0.7)
