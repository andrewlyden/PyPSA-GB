"""The must-run support sensitivity (user, 8 Oct 2026): shortfalls per asset, each group on its own recovery line."""
import pandas as pd
import pytest

from scripts.consumer_bill.line_recovery import make_entry
from scripts.consumer_bill.must_run_support_sensitivity import asset_shortfalls, summarise

SPEC = {
    "component_id": "must_run_support_gap", "label": "x", "decided": "test", "case_id": "private",
    "cost_basis": {"description": "going forward",
                   "revenue_columns": ["market_revenue_gbp", "support_revenue_gbp", "capacity_market_revenue_gbp"],
                   "cost_columns": ["fuel_carbon_gbp", "annual_fixed_om_gbp"]},
    "groups": [{"id": "merchant_biomass", "carrier": "biomass", "support_types": ["merchant"], "recovery_line": "cfd"},
               {"id": "merchant_biogas", "carrier": "biogas", "support_types": ["merchant"],
                "recovery_line": "renewables_obligation"}],
    "excluded": [], "scope_note": "",
}


def _register():
    return pd.DataFrame({
        "asset_id": ["LYN", "DRAX", "AD1", "AD2"],
        "carrier": ["biomass", "biomass", "biogas", "biogas"],
        "support_type": ["merchant", "CfD", "merchant", "merchant"],
        "finance_vintage": "legacy", "capacity_mw": [420.0, 1995.0, 10.0, 10.0], "dispatch_mwh": 1.0,
        "market_revenue_gbp": [100.0, 400.0, 50.0, 90.0], "support_revenue_gbp": [0.0, 500.0, 0.0, 0.0],
        "capacity_market_revenue_gbp": [10.0, 0.0, 0.0, 0.0],
        "fuel_carbon_gbp": [200.0, 850.0, 60.0, 60.0], "annual_fixed_om_gbp": [40.0, 100.0, 10.0, 10.0],
    })


def _bill():
    entry = dict(system_price_year=2025, basis="test")
    return {"recovery_basis": {"basis": "loss_inclusive_two_volume"}, "benchmark_kwh": 3100.0, "vat_rate": 0.05,
            "total_gbp_inc_vat": 1000.0,
            "line_recovery": {"cfd": make_entry("cfd_eligible_gsp_volume", 1000.0, household_loss_factor=1.1, **entry),
                              "renewables_obligation": make_entry("ro_eligible_metered_supply", 2000.0, **entry)}}


def test_only_uncontracted_assets_with_a_loss_are_paid():
    assets = asset_shortfalls(_register(), SPEC).set_index("asset_id")
    assert "DRAX" not in assets.index                       # under contract: out of scope
    assert assets.loc["LYN", "shortfall_gbp"] == pytest.approx(130.0)    # 240 cost - 110 revenue
    assert assets.loc["AD1", "shortfall_gbp"] == pytest.approx(20.0)
    assert assets.loc["AD2", "shortfall_gbp"] == pytest.approx(0.0)     # cash positive: nothing


def test_each_group_converts_on_its_own_line_with_vat():
    summary = summarise(asset_shortfalls(_register(), SPEC), _bill(), SPEC)
    biomass, biogas = summary["groups"]["merchant_biomass"], summary["groups"]["merchant_biogas"]
    assert biomass["gbp_per_household_ex_vat"] == pytest.approx(130.0 * 1.1 / 1000.0 * 3.1, abs=0.01)
    assert biogas["gbp_per_household_ex_vat"] == pytest.approx(20.0 / 2000.0 * 3.1, abs=0.01)
    assert summary["sensitivity_bill_gbp_inc_vat"] == pytest.approx(
        1000.0 + summary["gbp_per_household_inc_vat"], abs=0.01)
