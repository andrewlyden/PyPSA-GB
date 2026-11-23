import pandas as pd
import pytest

import scripts.consumer_bill.build_historical_support_outturn as support_outturn


def test_historical_support_outturn_keeps_proxy_separate_from_bill(monkeypatch):
    monkeypatch.setattr(support_outturn, "lccc_outturn", lambda *args: {
        "available": True, "generation_mwh": 10.0, "payment_gbp": 500.0,
    })
    bill_register = pd.DataFrame([
        {"component": "RO", "annualised_gbp_per_3100kwh_ex_vat": 95.0},
        {"component": "CfD allowance embedded in direct fuel", "annualised_gbp_per_3100kwh_ex_vat": 30.0},
    ])
    table, summary = support_outturn.build_audit(
        {"revenue": {"cfd_total_cost": 900.0, "cfd_total_dispatch_MWh": 8.0,
                      "roc_total_income": 647.3, "roc_total_dispatch_MWh": 12.0}},
        bill_register, "unused.csv", 2024, 64.73,
    )
    assert summary["cfd"]["lccc_outturn"]["payment_gbp"] == 500.0
    assert summary["cfd"]["proxy_cost_to_lccc_outturn_ratio"] == 1.8
    assert summary["roc"]["pypsa_proxy_certificates"] == pytest.approx(10.0)
    assert table.loc[table["measure"].eq("pypsa_support_proxy_cost"), "status"].iloc[0] == "requires_contract_reconciliation"
