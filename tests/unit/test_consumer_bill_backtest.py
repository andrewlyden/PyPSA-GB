import pandas as pd
import pytest

from scripts.consumer_bill.build_2020_backtest import build_backtest


def _forecast(register):
    return pd.DataFrame([
        {"component": component, "forecast_driver": "operational_model", "forecast_method": "forward_procurement_bridge"}
        for component in register.loc[~register["stack_group"].eq("vat"), "component"]
    ])


def _procurement(rows):
    return pd.DataFrame([
        {
            "bill_component": component,
            "candidate_allowance_gbp_per_3100kwh_ex_vat": value,
            "same_year_backtest_allowed": True,
            "recovery_status": "reproduced_from_observed_ofgem_forward_inputs",
            "model_basis": "Observed Ofgem forward procurement",
        }
        for component, value in rows
    ])


def test_backtest_requires_named_procurement_bridge_and_derives_vat():
    register = pd.DataFrame([
        {"stack_group": "hedged_wholesale", "component": "Published seasonal base forward allowance", "annual_gbp_per_3100kwh_ex_vat": 100.0},
        {"stack_group": "hedged_wholesale", "component": "Published seasonal peak forward allowance", "annual_gbp_per_3100kwh_ex_vat": 50.0},
        {"stack_group": "policy", "component": "Ofgem Annex 4 - CfD", "annual_gbp_per_3100kwh_ex_vat": 20.0},
        {"stack_group": "vat", "component": "VAT at 5%", "annual_gbp_per_3100kwh_ex_vat": 8.5},
    ])
    wholesale = pd.DataFrame({"modelled_demand_mwh": [1.0], "modelled_demand_weighted_clearing_price_gbp_per_mwh": [25.0]})
    backtest, summary = build_backtest(
        register, _forecast(register), wholesale,
        forward_procurement_bridge=_procurement([
            ("Published seasonal base forward allowance", 100.0),
            ("Published seasonal peak forward allowance", 50.0),
        ]),
    )
    assert backtest.loc[backtest["is_reconstructed_forward_procurement"], "backtest_gbp_per_3100kwh_ex_vat"].sum() == 150.0
    vat = backtest.loc[backtest["stack_group"].eq("vat"), "backtest_gbp_per_3100kwh_ex_vat"].iloc[0]
    assert vat == pytest.approx((150.0 + 20.0) * 0.05)
    assert summary["wholesale_procurement_bridge_included"] is True


def test_backtest_rejects_missing_procurement_bridge():
    register = pd.DataFrame([
        {"stack_group": "hedged_wholesale", "component": "Published seasonal base forward allowance", "annual_gbp_per_3100kwh_ex_vat": 100.0},
        {"stack_group": "vat", "component": "VAT at 5%", "annual_gbp_per_3100kwh_ex_vat": 5.0},
    ])
    wholesale = pd.DataFrame({"modelled_demand_mwh": [1.0], "modelled_demand_weighted_clearing_price_gbp_per_mwh": [25.0]})
    with pytest.raises(ValueError, match="wholesale-procurement bridge is required"):
        build_backtest(register, _forecast(register), wholesale)


def test_backtest_combines_forward_procurement_and_explicit_network_reconstruction():
    register = pd.DataFrame([
        {"stack_group": "network", "component": "Ofgem Annex 3 - TNUoS", "annual_gbp_per_3100kwh_ex_vat": 10.0},
        {"stack_group": "hedged_wholesale", "component": "Published seasonal base forward allowance", "annual_gbp_per_3100kwh_ex_vat": 70.0},
        {"stack_group": "hedged_wholesale", "component": "Published seasonal peak forward allowance", "annual_gbp_per_3100kwh_ex_vat": 30.0},
        {"stack_group": "vat", "component": "VAT at 5%", "annual_gbp_per_3100kwh_ex_vat": 5.5},
    ])
    wholesale = pd.DataFrame({"modelled_demand_mwh": [1.0], "modelled_demand_weighted_clearing_price_gbp_per_mwh": [30.0]})
    network = pd.DataFrame([{
        "component": "Ofgem Annex 3 - TNUoS", "candidate_allowance_gbp_per_3100kwh_ex_vat": 12.0,
        "same_year_backtest_allowed": True, "recovery_status": "candidate_ready", "model_basis": "delivery",
    }])
    backtest, summary = build_backtest(
        register, _forecast(register), wholesale,
        forward_procurement_bridge=_procurement([
            ("Published seasonal base forward allowance", 70.0),
            ("Published seasonal peak forward allowance", 30.0),
        ]),
        network_bridge=network, include_same_year_network_backtest=True,
    )
    assert backtest.loc[backtest["is_modelled_network_recovery"], "backtest_gbp_per_3100kwh_ex_vat"].sum() == 12.0
    assert summary["wholesale_procurement_historical_coverage_gbp"] == 100.0


def test_cfd_payback_reduces_the_bill_and_vat():
    register = pd.DataFrame([
        ("hedged_wholesale", "Energy", 100), ("policy", "CfD", 20),
        ("policy", "Ofgem Annex 4 - RO", 30), ("vat", "VAT", 7.5),
    ], columns=["stack_group", "component", "annual_gbp_per_3100kwh_ex_vat"])
    support = pd.DataFrame([
        {"scheme": scheme, "candidate_allowance_gbp_per_3100kwh_ex_vat": amount,
         "same_year_backtest_allowed": True, "recovery_status": "candidate", "model_basis": "test recovery"}
        for scheme, amount in [("CfD", -10), ("RO", 30)]
    ])
    wholesale = pd.DataFrame({"modelled_demand_mwh": [1], "modelled_demand_weighted_clearing_price_gbp_per_mwh": [25]})
    rows, summary = build_backtest(register, _forecast(register), wholesale,
        support_bridge=support, include_same_year_support_backtest=True,
        forward_procurement_bridge=_procurement([("Energy", 100)]))
    assert summary["reconstructed_bill_gbp"] == pytest.approx(126)
    assert rows.loc[rows.component.eq("CfD"), "backtest_gbp_per_3100kwh_ex_vat"].iloc[0] == -10
