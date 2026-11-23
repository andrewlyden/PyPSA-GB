import pandas as pd

from scripts.market.audit_consumer_bill_methodology import build_methodology_audit


def test_methodology_audit_disables_unevidenced_cfd_and_redispatch_pass_through():
    stack = pd.DataFrame([
        {"stack_group": "hedged_wholesale", "component": "Forward", "annual_gbp_per_3100kwh_ex_vat": 100.0},
        {"stack_group": "policy", "component": "Policy", "annual_gbp_per_3100kwh_ex_vat": 50.0},
        {"stack_group": "network", "component": "Network", "annual_gbp_per_3100kwh_ex_vat": 60.0},
        {"stack_group": "vat", "component": "VAT", "annual_gbp_per_3100kwh_ex_vat": 10.0},
    ])
    market_summary = {"redispatch": {"total_net_cost": 1234.0}}

    audit, summary = build_methodology_audit(stack, market_summary)

    cfd = audit.loc[audit["audit_area"].eq("CfD Supplier Obligation Levy")].iloc[0]
    redispatch = audit.loc[audit["audit_area"].eq("Balancing / redispatch recovery")].iloc[0]
    assert cfd["consumer_bill_treatment"] == "pass_through_disabled"
    assert pd.isna(cfd["modelled_amount"])
    assert redispatch["modelled_amount"] == 1234.0
    assert redispatch["consumer_bill_treatment"] == "pass_through_disabled"
    assert summary["modelled_component_pass_through_enabled"] == []


def test_methodology_audit_records_published_annex_4_cfd_baseline():
    stack = pd.DataFrame([
        {"stack_group": "policy", "component": "Policy", "annual_gbp_per_3100kwh_ex_vat": 50.0},
    ])
    policy = pd.DataFrame([
        {"component": "CfD", "annualised_gbp_per_3100kwh_ex_vat": 12.34},
    ])

    audit, summary = build_methodology_audit(stack, policy_components=policy)

    cfd = audit.loc[audit["audit_area"].eq("CfD Supplier Obligation Levy")].iloc[0]
    assert cfd["published_amount"] == 12.34
    assert cfd["evidence_status"] == "published_annual_annex_4_allowance"
    assert summary["modelled_results_seen"]["published_cfd_allowance_gbp_per_3100kwh_ex_vat"] == 12.34
