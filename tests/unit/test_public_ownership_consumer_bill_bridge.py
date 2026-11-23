import pandas as pd
import pytest

from scripts.public_ownership.build_consumer_bill_financing_bridge import (
    build_consumer_bill_financing_bridge,
)


def test_bridge_assigns_savings_to_generation_finance_not_existing_bill_groups():
    register = pd.DataFrame([
        {"case_id": "private", "asset_id": "solar::new_build", "finance_technology": "Solar PV",
         "finance_vintage": "new_build", "capacity_mw": 1.0, "dispatch_mwh": 1000.0,
         "annual_required_revenue_gbp": 100.0, "annual_capital_recovery_gbp": 80.0, "support_type": "CfD"},
        {"case_id": "public_coordination", "asset_id": "solar::new_build", "finance_technology": "Solar PV",
         "finance_vintage": "new_build", "capacity_mw": 1.0, "dispatch_mwh": 1000.0,
         "annual_required_revenue_gbp": 70.0, "annual_capital_recovery_gbp": 50.0, "support_type": "CfD"},
    ])
    summary = pd.DataFrame([
        {"case_id": "private", "annual_financing_saving_vs_private_gbp": 0.0,
         "consumer_equivalent_saving_gbp_per_mwh_demand": 0.0},
        {"case_id": "public_coordination", "annual_financing_saving_vs_private_gbp": 30.0,
         "consumer_equivalent_saving_gbp_per_mwh_demand": 2.0},
    ])
    technology, bridge, report = build_consumer_bill_financing_bridge(register, summary)
    assert technology.loc[0, "annual_financing_saving_gbp"] == pytest.approx(30.0)
    generation = bridge.loc[
        bridge["consumer_bill_group"].eq("Policy costs - new CfD contracts")
        & bridge["case_id"].eq("public_coordination")
    ].iloc[0]
    assert generation["annual_system_change_gbp"] == pytest.approx(-30.0)
    assert generation["equivalent_gbp_per_3100kwh_ex_vat"] == pytest.approx(-6.2)
    assert set(bridge.loc[bridge["treatment"].eq("held_unchanged"), "annual_system_change_gbp"]) == {0.0}
    assert report["cases"]["public_coordination"]["annual_financing_saving_gbp"] == pytest.approx(30.0)
