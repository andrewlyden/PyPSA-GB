"""F29 systematic risk in the itemised party flows (fix 8 Nov 2026).

The Exchequer ledger and the resource saving carry the systematic risk the state holds at the workbook's social
price (``socialised_risk_gbp``); the itemised flows had no such flow, so every case with state-financed plant failed
``party_flows.check`` by exactly that amount on the re-solve (GBP 105,065,671.35 for P2 on the central-dispatch
groups, GBP 922,231,051.77 on the 2025 legislation bound).
"""
import pandas as pd
import pytest

from scripts.public_ownership import party_flows


def _register():
    base = {"producer_rent_gbp": 0.0, "annual_capital_recovery_gbp": 100.0, "annual_required_revenue_gbp": 150.0}
    return pd.DataFrame([
        dict(base, case_id="private", asset_id="a", ownership_share=0.0, socialised_risk_gbp=0.0),
        dict(base, case_id="private", asset_id="b", ownership_share=0.0, socialised_risk_gbp=0.0),
        dict(base, case_id="public_ownership", asset_id="a", ownership_share=1.0, socialised_risk_gbp=7.0),
        dict(base, case_id="public_ownership", asset_id="b", ownership_share=0.5, socialised_risk_gbp=3.0),
    ])


def test_the_socialised_risk_is_a_component_against_private():
    components = party_flows.register_components(_register(), "public_ownership")
    assert components["socialised_risk"] == pytest.approx(10.0)


def test_the_socialised_risk_is_an_exchequer_cost_and_a_resource_cost():
    flows = party_flows.Flows("public_ownership", per_household=lambda gbp: 0.0)
    flows.add("socialised_systematic_risk", "systematic risk", "exchequer", "resource", 10.0, "financing")
    net = party_flows.net_by_party(flows.rows)
    assert net["exchequer"] == pytest.approx(-10.0)
    # check() compares -(resource + foreign) with the resource saving, which the ledger lowers by the same amount.
    assert -(net["resource"] + net["foreign"]) == pytest.approx(-10.0)


def test_a_tariff_case_books_the_ledger_part_as_an_exchequer_cost():
    from scripts.public_ownership.investor_ledger import SOCIALISED_RISK

    flows = party_flows.Flows("public_ownership@legacy_acquired", per_household=lambda gbp: 0.0)
    parties = {"bill_items": [], "investor_parts_gbp": {}, "resource_parts_gbp": {SOCIALISED_RISK: -10.0}}
    party_flows.tariff_flows(flows, parties=parties, network={}, levers={})
    net = party_flows.net_by_party(flows.rows)
    assert net["exchequer"] == pytest.approx(-10.0)
    assert -(net["resource"] + net["foreign"]) == pytest.approx(-10.0)


def test_a_register_without_the_column_gives_zero():
    components = party_flows.register_components(_register().drop(columns="socialised_risk_gbp"), "public_ownership")
    assert components["socialised_risk"] == 0.0
