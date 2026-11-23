import pandas as pd
import pytest

from scripts.market.consumer_bill_breakdown import build_breakdown


def _bill() -> pd.DataFrame:
    return pd.DataFrame([{
        "annual_consumption_kwh": 2500,
        "payment_method": "direct_debit",
        "region": "GB average",
        "vat_rate": 0.05,
        "standing_charge_gbp_ex_vat": 84.0,
        "base_energy_charge_gbp_ex_vat": 420.0,
        "wholesale_price_delta_gbp_ex_vat": -40.0,
        "consumer_bill_gbp_ex_vat": 464.0,
        "consumer_bill_gbp_inc_vat": 487.2,
    }])


def test_breakdown_groups_and_reconciles_published_cost_stack():
    stack = pd.DataFrame([
        {"stack_group": "hedged_wholesale", "component": "Forward allowance", "annual_gbp_per_3100kwh_ex_vat": 150.0},
        {"stack_group": "hedge_uplifts", "component": "Hedging", "annual_gbp_per_3100kwh_ex_vat": 10.0},
        {"stack_group": "system_balancing", "component": "Balancing", "annual_gbp_per_3100kwh_ex_vat": 20.0},
        {"stack_group": "losses", "component": "Losses", "annual_gbp_per_3100kwh_ex_vat": 15.0},
        {"stack_group": "policy", "component": "Policy costs", "annual_gbp_per_3100kwh_ex_vat": 140.0},
        {"stack_group": "network", "component": "Network", "annual_gbp_per_3100kwh_ex_vat": 140.0},
        {"stack_group": "supplier_operating", "component": "Operating", "annual_gbp_per_3100kwh_ex_vat": 80.0},
        {"stack_group": "payment", "component": "Payment", "annual_gbp_per_3100kwh_ex_vat": 5.0},
        {"stack_group": "bad_debt", "component": "Bad debt", "annual_gbp_per_3100kwh_ex_vat": 0.0},
        {"stack_group": "margin", "component": "Margin", "annual_gbp_per_3100kwh_ex_vat": 10.0},
        {"stack_group": "headroom", "component": "Headroom", "annual_gbp_per_3100kwh_ex_vat": 5.0},
        {"stack_group": "vat", "component": "VAT", "annual_gbp_per_3100kwh_ex_vat": 28.75},
        {"stack_group": "total", "component": "Total", "annual_gbp_per_3100kwh_ex_vat": 603.75},
    ])

    components, summary = build_breakdown(_bill(), stack)

    groups = {row["group"]: row["annual_gbp_per_3100kwh"] for row in summary["published_cost_stack"]["groups"]}
    assert groups["Wholesale & hedging"] == pytest.approx(160.0)
    assert groups["Policy costs"] == pytest.approx(140.0)
    assert summary["published_cost_stack"]["total_gbp_inc_vat"] == pytest.approx(603.75)
    assert summary["headline_bill"]["published_tariff_baseline_gbp_inc_vat"] == pytest.approx(529.2)
    assert "not yet been decomposed" in summary["cfd_treatment"]["published_bill"]
    assert len(components) == 12


def test_breakdown_replaces_policy_residual_with_reconciled_annex_4_cfd_row():
    stack = pd.DataFrame([
        {"stack_group": "policy", "component": "Policy costs", "annual_gbp_per_3100kwh_ex_vat": 50.0},
        {"stack_group": "vat", "component": "VAT", "annual_gbp_per_3100kwh_ex_vat": 2.5},
    ])
    policy = pd.DataFrame([
        {"component": "CfD", "component_label": "Contracts for Difference", "annualised_gbp_per_3100kwh_ex_vat": 20.0},
        {"component": "RO", "component_label": "Renewables Obligation", "annualised_gbp_per_3100kwh_ex_vat": 30.0},
    ])

    components, summary = build_breakdown(_bill(), stack, policy)

    assert components["component"].tolist() == [
        "Ofgem Annex 4 — Contracts for Difference",
        "Ofgem Annex 4 — Renewables Obligation",
        "VAT",
    ]
    assert "£20.00" in summary["cfd_treatment"]["published_bill"]
