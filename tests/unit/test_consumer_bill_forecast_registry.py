import pandas as pd

from scripts.market.build_consumer_bill_forecast_registry import build_registry


def test_registry_classifies_model_and_system_transition_components():
    register = pd.DataFrame([
        {"stack_group": "network", "component": "Ofgem Annex 3 - TNUoS", "annual_gbp_per_3100kwh_ex_vat": 10.0},
        {"stack_group": "policy", "component": "Ofgem Annex 4 - CfD", "annual_gbp_per_3100kwh_ex_vat": 5.0},
        {"stack_group": "hedged_wholesale", "component": "Published seasonal base forward allowance", "annual_gbp_per_3100kwh_ex_vat": 20.0},
        {"stack_group": "vat", "component": "VAT at 5%", "annual_gbp_per_3100kwh_ex_vat": 2.0},
    ])
    result, summary = build_registry(register)

    assert len(result) == 3
    assert result.loc[result["component"].str.contains("TNUoS"), "forecast_driver"].iloc[0] == "system_transition"
    assert result.loc[result["component"].str.contains("CfD"), "forecast_method"].iloc[0] == "modelled_recovery_bridge"
    assert result.loc[result["stack_group"].eq("hedged_wholesale"), "recovery_denominator"].iloc[0] == "supplier_sales_mwh"
    assert summary["by_driver"]["operational_model"] == 2
