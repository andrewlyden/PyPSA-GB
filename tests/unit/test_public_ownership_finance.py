from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.calculate_financing_scenarios import canonical_technology
from scripts.public_ownership.finance import (
    annual_required_revenue,
    capital_recovery_factor,
    fisher_real_rate,
    levelised_cost_gbp_per_mwh,
    state_equity_share,
    wacc_for_case,
)
from scripts.public_ownership.extract_workbook_v5 import extract


#: Item OH1: the state's holding-company cost, switched off in fixtures that
#: pin other mechanics.  See tests/unit/test_public_ownership_state_overhead.py.
NO_OVERHEAD = {"enabled": False}


#: F29 (26 Oct 2026): workbook v2.0 and its Excel-recalculated switch cases.
WORKBOOK = Path("data/public_ownership/GB_Power_Model_v6.2.xlsx")
VARIANTS = Path("data/public_ownership/financing/workbook_v6_2_variants.csv")
#: The snapshot committed for clones without the workbook (the public release does not distribute it).
COMMITTED_SNAPSHOT = Path("data/public_ownership/financing/workbook_v6_2_assumptions.yaml")


def _committed_snapshot() -> dict:
    with COMMITTED_SNAPSHOT.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def workbook_snapshot() -> dict:
    if WORKBOOK.exists():
        return extract(WORKBOOK, VARIANTS)
    return _committed_snapshot()


@pytest.mark.skipif(not WORKBOOK.exists(), reason="the financing workbook is not distributed")
def test_committed_snapshot_equals_a_fresh_extraction(workbook_snapshot):
    committed = _committed_snapshot()
    fresh = {key: value for key, value in workbook_snapshot.items() if key != "extracted_on"}
    assert {key: value for key, value in committed.items() if key != "extracted_on"} == fresh, (
        f"{COMMITTED_SNAPSHOT} is stale: copy the extract_public_ownership_workbook_assumptions output over it")


def _snapshot():
    return {
        "rate_basis": "real_CPI",
        "assumptions": {"public_coordination_state_equity_share": {"value": 0.30}},
        "technologies": [{
            "technology": "Solar PV",
            "wacc": {
                "private": 0.066645,
                "public_coordination": 0.05763,
                "public_ownership_new_build": 0.03245098039215688,
                "public_ownership_legacy": 0.027450980392156765,
            },
        }],
    }


def _costs():
    return pd.DataFrame([{
        "technology": "Solar PV", "capex_power_gbp_per_kw": 800.0,
        "capex_energy_gbp_per_kwh": 0.0, "fixed_om_gbp_per_kw_year": 12.0,
        "variable_om_gbp_per_mwh": 0.0, "economic_life_years": 35.0,
        "price_year": 2024, "real_or_nominal": "real", "status": "reviewed",
    }])


def test_finance_math_matches_workbook_real_public_rate_and_zero_rate_crf():
    assert fisher_real_rate(0.048, 0.02) == pytest.approx(0.027450980392156765)
    assert capital_recovery_factor(0.0, 25) == pytest.approx(1 / 25)
    assert capital_recovery_factor(0.05, 20) == pytest.approx(0.08024258719069129)


def test_workbook_export_reads_all_operational_wacc_rows_with_provenance(workbook_snapshot):
    snapshot = workbook_snapshot
    assert snapshot["schema_version"] == 5
    assert len(snapshot["technologies"]) == 33
    # The base variant reproduces the workbook's cached values, cell for cell.
    assert snapshot["variants_checked_cells"] == 781
    solar = next(row for row in snapshot["technologies"] if row["technology"] == "Solar PV")
    assert solar["wacc"]["private"] == pytest.approx(0.0724268, abs=1e-6)
    assert solar["wacc"]["public_coordination"] == pytest.approx(0.0586207, abs=1e-6)
    # Schema 5 (v6.2): the lead block, at a notional 55% gearing, equals the wacc mapping; Fixed TMR is the
    # workbook's co-equal convention.
    assert solar["wacc_by_basis"]["notional_gearing_shielded"] == {
        "private": solar["wacc"]["private"], "public_coordination": solar["wacc"]["public_coordination"]}
    assert solar["wacc_by_basis"]["fixed_tmr_shielded"]["private"] == pytest.approx(0.0648849, abs=1e-6)
    offshore = next(row for row in snapshot["technologies"] if row["technology"] == "Offshore wind")
    assert offshore["wacc"]["private"] == pytest.approx(0.0847566, abs=1e-6)  # the change note: 8.48% in both versions
    assert solar["wacc_source_cells"]["private"] == "WACC Scenarios!D7"
    assert solar["wacc_source_cells"]["public_coordination"] == "WACC Scenarios!F7"
    assert solar["wacc_source_cells"]["public_ownership_new_build"] == "WACC Scenarios!H7"
    assert snapshot["assumptions"]["public_borrowing_real"]["source_cell"] == "Sc4 Operative!F11"
    # The renamed technologies keep the model's names and record the workbook's.
    ccus = next(row for row in snapshot["technologies"] if row["technology"] == "Gas generation - CCUS - mature")
    assert ccus["workbook_technology"] == "Gas generation - with CCUS - mature"


def test_workbook_parity_every_technology_orders_cases_and_prices_the_states_risk(workbook_snapshot):
    """Sc1 > Sc2 > the cash rate; the cash rate is the real gilt plus the NLF margin; economic = cash + premium."""
    a = workbook_snapshot["assumptions"]
    cash = a["public_borrowing_real"]["value"]
    assert cash == pytest.approx(a["risk_free_real"]["value"] + a["public_lending_margin"]["value"], abs=1e-12)
    margins = workbook_snapshot["public_rate_by_lending_margin"]
    assert margins["nlf"]["value"] == pytest.approx(cash, abs=1e-12)
    assert margins["pwlb_standard"]["value"] == pytest.approx(cash + 0.0089, abs=1e-9)
    assert cash == pytest.approx(0.0275788, abs=1e-7)  # the change note: 2.76% (2.71% in v2.0)
    flat = {"social_flat_19": 0.019, "social_flat_10": 0.010}
    per_beta = {"social_beta_20": 0.02, "market_erp": a["equity_risk_premium"]["value"]}
    for row in workbook_snapshot["technologies"]:
        wacc = row["wacc"]
        assert wacc["private"] > wacc["public_coordination"] > wacc["public_ownership_new_build"], row["technology"]
        assert wacc["public_ownership_new_build"] == pytest.approx(cash, abs=1e-12)
        assert wacc["public_ownership_legacy"] == pytest.approx(cash, abs=1e-12)
        by_premium = row["economic_rate_by_premium"]
        assert by_premium["none"] == pytest.approx(cash, abs=1e-12)
        for key, price in flat.items():
            assert by_premium[key] - by_premium["none"] == pytest.approx(price, abs=1e-9), (row["technology"], key)
        for key, price in per_beta.items():
            assert by_premium[key] - by_premium["none"] == pytest.approx(row["asset_beta"] * price, abs=1e-9), (
                row["technology"], key)


def test_cost_registry_must_match_configured_price_basis():
    assets = pd.DataFrame([{
        "model_year": 2030, "component_type": "Generator", "asset_id": "new-solar",
        "carrier": "solar_pv", "finance_technology": "Solar PV", "finance_vintage": "new_build",
        "capacity_mw": 1.0, "dispatch_mwh": 1_000.0, "evidence_status": "resolved",
    }])
    with pytest.raises(ValueError, match="price_year 2025"):
        calculate_financing_scenarios(assets, _costs(), _snapshot(), price_year=2025)
    nominal_costs = _costs().assign(real_or_nominal="nominal")
    with pytest.raises(ValueError, match="real basis"):
        calculate_financing_scenarios(assets, nominal_costs, _snapshot(), price_year=2024)
    with pytest.raises(ValueError, match="private"):
        calculate_financing_scenarios(assets, _costs(), _snapshot(), cases=("public_ownership",))
    register, _ = calculate_financing_scenarios(assets, _costs(), _snapshot(), price_year=2024)
    assert set(register["case_id"]) == {"private", "public_coordination", "public_ownership"}


def test_case_selection_preserves_legacy_contract_treatment():
    rates = _snapshot()["technologies"][0]
    assert wacc_for_case(rates, "private", "legacy") == pytest.approx(0.066645)
    assert wacc_for_case(rates, "public_coordination", "legacy") == pytest.approx(0.066645)
    assert wacc_for_case(rates, "public_coordination", "new_build") == pytest.approx(0.05763)
    assert wacc_for_case(rates, "public_ownership", "legacy") == pytest.approx(0.066645)
    assert state_equity_share("public_coordination", "new_build") == pytest.approx(0.30)
    assert state_equity_share("public_coordination", "legacy") == 0.0
    assert state_equity_share("public_ownership", "legacy") == 0.0
    assert canonical_technology("Gas generation — unabated") == "Gas generation - unabated"


def test_required_revenue_separates_resource_cost_from_revenue_transfers():
    account = annual_required_revenue(
        capacity_mw=1.0, dispatch_mwh=1000.0, capex_power_gbp_per_kw=800.0,
        fixed_om_gbp_per_kw_year=12.0, variable_om_gbp_per_mwh=3.0,
        wacc=0.0, remaining_life_years=20,
    )
    assert account["annual_capital_recovery_gbp"] == pytest.approx(40_000.0)
    assert account["annual_fixed_om_gbp"] == pytest.approx(12_000.0)
    assert account["annual_variable_om_gbp"] == pytest.approx(3_000.0)
    assert account["annual_required_revenue_gbp"] == pytest.approx(55_000.0)


def test_financing_register_is_case_by_vintage_and_leaves_unresolved_assets_out_of_totals():
    assets = pd.DataFrame([
        {
            "model_year": 2030, "component_type": "Generator", "asset_id": "legacy-solar",
            "carrier": "solar_pv", "finance_technology": "Solar PV", "finance_vintage": "legacy",
            "capacity_mw": 1.0, "dispatch_mwh": 1_000.0, "evidence_status": "resolved",
            "commissioning_year": 2020,
        },
        {
            "model_year": 2030, "component_type": "Generator", "asset_id": "new-solar",
            "carrier": "solar_pv", "finance_technology": "Solar PV", "finance_vintage": "new_build",
            "capacity_mw": 1.0, "dispatch_mwh": 1_000.0, "evidence_status": "resolved",
            "commissioning_year": 2027,
        },
        {
            "model_year": 2030, "component_type": "Generator", "asset_id": "unmapped",
            "carrier": "mystery", "finance_technology": "Missing", "finance_vintage": "new_build",
            "capacity_mw": 1.0, "dispatch_mwh": 1_000.0, "evidence_status": "unmapped_carrier",
        },
    ])
    register, totals = calculate_financing_scenarios(assets, _costs(), _snapshot(),
                                                     state_operating_overhead=NO_OVERHEAD)
    coordination = register.loc[(register.case_id == "public_coordination") & (register.asset_id == "legacy-solar")].iloc[0]
    private = register.loc[(register.case_id == "private") & (register.asset_id == "legacy-solar")].iloc[0]
    assert coordination.wacc == pytest.approx(private.wacc)
    new_public = register.loc[(register.case_id == "public_ownership") & (register.asset_id == "new-solar")].iloc[0]
    assert new_public.wacc == pytest.approx(0.03245098039215688)
    assert new_public.ownership_share == 1.0
    assert totals.set_index("case_id").loc["private", "unresolved_assets"] == 1
    assert pd.isna(totals.set_index("case_id").loc["private", "public_surplus_before_recycling_gbp"])
    assert totals.set_index("case_id").loc["private", "unallocated_consumer_benefit_gbp"] == 0.0
    assert totals.set_index("case_id").loc["public_ownership", "annual_financing_saving_vs_private_gbp"] > 0
