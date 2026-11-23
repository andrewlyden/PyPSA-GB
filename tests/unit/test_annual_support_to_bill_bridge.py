import pandas as pd
import pytest

from scripts.consumer_bill.build_annual_support_to_bill_bridge import (
    SupportToBillBridgeError,
    build_bridge,
)


def _inputs():
    policy = pd.DataFrame(
        {
            "component": ["CfD", "RO"],
            "annualised_gbp_per_3100kwh_ex_vat": [10.0, 20.0],
        }
    )
    cfd = {
        "calendar_year": 2020,
        "solved_networks_aggregated": 12,
        "modelled_payment_independent_formula_gbp": 90.0,
        "actual_payment_gbp_mapped_contracts": 100.0,
    }
    cfd_mapping = {
        "calendar_year": 2020,
        "contract_register_complete": True,
        "model_mapping_complete": True,
    }
    roc = {
        "solved_networks_aggregated": 12,
        "ofgem_rocs_issued": 10.0,
    }
    roc_ledger = {"total_station_representation_coverage": 1.0}
    cfd_levy = pd.DataFrame({
        "month": [f"2020-{month:02d}" for month in range(1, 13)],
        "reconciled_eligible_demand_mwh": [10.0] * 12,
        "reconciled_levy_cost_gbp": [10.0] * 12,
    })
    monthly_roc = [{"reconciled_roc_certificates": 1.0, "strict_direct_roc_certificates": 0.5}] * 12
    monthly_starts = [f"2020-{month:02d}-01" for month in range(1, 13)]
    roc_parameters = {"periods": [
        {"scheme_year": "2019-20", "calendar_start": "2020-01-01", "calendar_end": "2020-03-31", "final_notional_roc_value_gbp_per_roc": 50.0},
        {"scheme_year": "2020-21", "calendar_start": "2020-04-01", "calendar_end": "2020-12-31", "final_notional_roc_value_gbp_per_roc": 55.0},
    ]}
    assumptions = {"cfd": {}, "ro": {}}
    return policy, cfd, cfd_mapping, roc, roc_ledger, cfd_levy, monthly_roc, monthly_starts, roc_parameters, assumptions


def test_bridge_uses_final_cfd_levy_demand_and_scales_ro_by_explicit_representation():
    table, summary = build_bridge(*_inputs(), year=2020, expected_months=12)

    assert table.loc[table["scheme"].eq("CfD"), "candidate_allowance_gbp_per_3100kwh_ex_vat"].iloc[0] == 2.325
    assert table.loc[table["scheme"].eq("RO"), "candidate_allowance_gbp_per_3100kwh_ex_vat"].iloc[0] == 24.0
    assert table.loc[table["scheme"].eq("RO"), "direct_candidate_allowance_gbp_per_3100kwh_ex_vat"].iloc[0] == 12.0
    assert table.loc[table["scheme"].eq("RO"), "modelled_scheme_cost_gbp"].iloc[0] == 645.0
    assert not table["pass_through_allowed"].any()
    assert table["same_year_backtest_allowed"].all()
    assert summary["bridge_status"] == "candidates_ready_for_same_year_backtest"
    assert summary["modelled_support_coverage"]["cfd_independent_payment_coverage"] == 0.9


def test_bridge_rejects_incomplete_cfd_mapping():
    policy, cfd, cfd_mapping, roc, roc_ledger, cfd_levy, monthly_roc, monthly_starts, roc_parameters, assumptions = _inputs()
    cfd_mapping["model_mapping_complete"] = False

    with pytest.raises(SupportToBillBridgeError, match="CfD mapping is incomplete"):
        build_bridge(policy, cfd, cfd_mapping, roc, roc_ledger, cfd_levy, monthly_roc, monthly_starts, roc_parameters, assumptions, 2020, 12)


def test_bridge_preserves_a_negative_cfd_levy_credit():
    inputs = list(_inputs())
    inputs[5]["reconciled_levy_cost_gbp"] = -10.0
    inputs[1]["modelled_payment_independent_formula_gbp"] = -90.0
    inputs[1]["actual_payment_gbp_mapped_contracts"] = -100.0

    table, summary = build_bridge(*inputs, year=2020, expected_months=12)

    cfd = table.loc[table["scheme"].eq("CfD")].iloc[0]
    assert cfd["official_recovery_allowance_gbp_per_3100kwh_ex_vat"] == -3.1
    assert cfd["candidate_allowance_gbp_per_3100kwh_ex_vat"] == -2.325
    assert cfd["direct_model_coverage_of_comparator"] == 0.9
    assert summary["cfd_levy_recovery"]["recovered_cost_gbp"] == -120.0
