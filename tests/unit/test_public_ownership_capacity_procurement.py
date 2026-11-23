"""F27 item 6 (user, 20 and 24 Oct 2026): capacity without an auction, lever ``capacity_procurement``.

Existing units holding a Capacity Market receipt are paid their going-forward fixed cost less their
own wholesale and balancing margin, floored at zero (capped at the receipt), instead of the clearing
price; new capacity is still auctioned. P2: the Capacity Market line falls by the saving on private
owners' units, a transfer from them (never a resource saving). P3: a lower purchase price.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership import capacity_procurement as cp
from scripts.public_ownership.levers import (
    CAPACITY_AUCTION, CAPACITY_COST_BASED_EXISTING, LEVER_CAPACITY_PROCUREMENT, LEVERS, REGISTER_LEVERS,
)
from config.config_loader import load_defaults

def _po():
    return load_defaults()["public_ownership"]


def _rows():
    # GBP a year. margin = market - fuel and carbon - variable O&M.
    return pd.DataFrame([
        # Existing CCGT, margin 30 below its going-forward cost 100: paid 70 of its 120 receipt.
        {"asset_id": "ccgt", "carrier": "CCGT", "finance_vintage": "legacy", "capacity_market_revenue_gbp": 120.0,
         "annual_fixed_om_gbp": 100.0, "market_revenue_gbp": 200.0, "fuel_carbon_gbp": 160.0,
         "annual_variable_om_gbp": 10.0, "ownership_share": 0.0},
        # Existing hydro, margin above its fixed cost: paid nothing.
        {"asset_id": "hydro", "carrier": "large_hydro", "finance_vintage": "legacy", "capacity_market_revenue_gbp": 50.0,
         "annual_fixed_om_gbp": 40.0, "market_revenue_gbp": 90.0, "fuel_carbon_gbp": 0.0,
         "annual_variable_om_gbp": 5.0, "ownership_share": 0.0},
        # Existing OCGT whose missing money (90) exceeds its receipt (30): capped at the receipt.
        {"asset_id": "ocgt", "carrier": "OCGT", "finance_vintage": "legacy", "capacity_market_revenue_gbp": 30.0,
         "annual_fixed_om_gbp": 100.0, "market_revenue_gbp": 20.0, "fuel_carbon_gbp": 10.0,
         "annual_variable_om_gbp": 0.0, "ownership_share": 0.0},
        # Existing oil without a fixed-cost row: keeps its receipt.
        {"asset_id": "oil", "carrier": "oil", "finance_vintage": "legacy", "capacity_market_revenue_gbp": 5.0,
         "annual_fixed_om_gbp": float("nan"), "market_revenue_gbp": 0.0, "fuel_carbon_gbp": 0.0,
         "annual_variable_om_gbp": 0.0, "ownership_share": 0.0},
        # New build: still auctioned.
        {"asset_id": "battery", "carrier": "Battery", "finance_vintage": "new_build", "capacity_market_revenue_gbp": 60.0,
         "annual_fixed_om_gbp": 10.0, "market_revenue_gbp": 100.0, "fuel_carbon_gbp": 0.0,
         "annual_variable_om_gbp": 0.0, "ownership_share": 1.0},
    ])


def test_the_payment_is_going_forward_cost_less_own_margin_floored_and_capped():
    out = cp.cost_based_payments(_rows()).set_index(_rows()["asset_id"])
    assert out["cost_based_payment_gbp"].to_dict() == pytest.approx(
        {"ccgt": 70.0, "hydro": 0.0, "ocgt": 30.0, "oil": 5.0, "battery": 60.0})
    assert out["saving_gbp"].to_dict() == pytest.approx({"ccgt": 50.0, "hydro": 50.0, "ocgt": 0.0, "oil": 0.0,
                                                         "battery": 0.0})
    assert out["existing"].tolist() == [True, True, True, True, False]
    assert out["costed"].tolist() == [True, True, True, False, False]
    # Without the cap the OCGT would be paid its whole missing money.
    uncapped = cp.cost_based_payments(_rows(), {"cap_at_auction_receipt": False})
    assert uncapped["cost_based_payment_gbp"].iloc[2] == pytest.approx(90.0)


def test_the_receipt_is_scaled_first_so_f9_is_not_taken_twice():
    out = cp.cost_based_payments(_rows(), receipt_scale=0.9)
    # CCGT: receipt 108, payment 70, saving 38; hydro: receipt 45, all saved.
    assert out["saving_gbp"].iloc[:2].tolist() == pytest.approx([38.0, 45.0])
    assert cp.f9_scale({"capacity_market_state_bids": {"calendar_2030_cost_change_share": -0.006}},
                       {"capacity_market_state_bids": {"calendar_2030_cost_change_share": 0.0}}) == pytest.approx(0.994)
    assert cp.f9_scale({}, None) == 1.0


def test_the_bill_saving_is_the_privately_owned_share_only():
    private = _rows()
    case = private.assign(ownership_share=[0.4, 0.0, 0.0, 0.0, 1.0])
    result = cp.account_result(private, case)
    assert result["saving_gbp"] == pytest.approx(100.0)
    assert result["saving_private_owners_gbp"] == pytest.approx(0.6 * 50.0 + 50.0)
    assert result["saving_state_share_kept_gbp"] == pytest.approx(20.0)
    assert result["existing_units"] == 4 and result["existing_units_without_fixed_cost"] == 1
    assert result["by_carrier"]["CCGT"]["saving_gbp"] == pytest.approx(30.0)
    assert "F27 item 6" in cp.method_note(result)


def test_the_valuation_rows_replace_capacity_market_income_on_existing_units_only():
    rows = cp.valuation_rows(_rows())
    assert rows["capacity_market_revenue_gbp"].tolist() == pytest.approx([70.0, 0.0, 30.0, 5.0, 60.0])
    assert _rows()["capacity_market_revenue_gbp"].tolist() == [120.0, 50.0, 30.0, 5.0, 60.0]


def test_parameters_are_checked_and_the_code_defaults_equal_the_config():
    from scripts.public_ownership.calculate_financing_scenarios import LEVER_PARAMETER_DEFAULTS, lever_parameters

    block = _po()["lever_parameters"][cp.PARAMETER_KEY]
    assert block == cp.DEFAULT_PARAMETERS == LEVER_PARAMETER_DEFAULTS[cp.PARAMETER_KEY]
    assert lever_parameters(_po()["lever_parameters"])[cp.PARAMETER_KEY] == cp.DEFAULT_PARAMETERS
    with pytest.raises(ValueError, match="unknown"):
        cp.parameters({"going_forward_cost": 1.0})
    with pytest.raises(ValueError, match="empty"):
        cp.parameters({"existing_finance_vintages": []})


def test_the_lever_is_a_register_lever_on_p2_and_p3_only():
    from scripts.public_ownership.reference_runs import (
        ACQUISITION_TERM_LEVERS, configured_case_specs, expanded_cases, same_fleet_tariff_cases, tariff_cases,
    )

    lever = LEVERS[LEVER_CAPACITY_PROCUREMENT]
    assert lever.values == (CAPACITY_AUCTION, CAPACITY_COST_BASED_EXISTING) == (cp.AUCTION, cp.COST_BASED_EXISTING)
    assert LEVER_CAPACITY_PROCUREMENT in REGISTER_LEVERS and LEVER_CAPACITY_PROCUREMENT in ACQUISITION_TERM_LEVERS
    po = _po()
    ids = expanded_cases(po)
    moved = [c for c in ids if "capacity_cost_based" in c]
    assert moved == ["public_ownership@capacity_cost_based_existing",
                     "public_ownership@fully_nationalised_market_value_capacity_cost_based"]
    specs = {spec.case_id: spec for spec in configured_case_specs(po)}
    p2 = specs["public_ownership@capacity_cost_based_existing"]
    assert p2.market_bill() and p2.meta.get("kind") == "one_at_a_time"
    p3 = specs["public_ownership@fully_nationalised_market_value_capacity_cost_based"]
    assert p3.case_id in tariff_cases(po)
    # It moves only the purchase price, so it owns P3's fleet and bids P3's stack.
    assert p3.case_id in same_fleet_tariff_cases(po, "public_ownership@fully_nationalised_market_value")
    # Not P1: a stake in new build owns no existing plant (the choice would be a comparator change).
    assert not any(c.startswith("public_control@") and "capacity" in c for c in ids)
    assert not lever.applies(CAPACITY_COST_BASED_EXISTING, specs["public_control"].settings)
    assert not lever.applies(CAPACITY_COST_BASED_EXISTING, specs["private"].settings)


def test_the_acquisition_terms_carry_the_lever_into_their_own_plan():
    from scripts.public_ownership.calculate_financing_scenarios import _plan_key, acquisition_terms
    from scripts.public_ownership.levers import DEFAULT_SETTINGS

    base = acquisition_terms({}, DEFAULT_SETTINGS)
    moved = acquisition_terms({}, {**DEFAULT_SETTINGS, LEVER_CAPACITY_PROCUREMENT: CAPACITY_COST_BASED_EXISTING})
    assert base["capacity_procurement"] == CAPACITY_AUCTION
    assert moved["capacity_procurement"] == CAPACITY_COST_BASED_EXISTING
    assert _plan_key(base, 0.03) != _plan_key(moved, 0.03)


def test_the_account_books_only_cases_on_the_lever():
    from scripts.public_ownership.build_public_account import capacity_procurement_accounts

    private = _rows().assign(case_id="private")
    rows = pd.concat([private, private.assign(case_id="p2"), private.assign(case_id="p2_lever")], ignore_index=True)
    summary = {"private": {}, "p2": {}, "p2_lever": {
        "capacity_market_state_bids": {"calendar_2030_cost_change_share": -0.1}}}
    settings = {"private": {}, "p2": {LEVER_CAPACITY_PROCUREMENT: CAPACITY_AUCTION},
                "p2_lever": {LEVER_CAPACITY_PROCUREMENT: CAPACITY_COST_BASED_EXISTING}}
    capacity_procurement_accounts(summary, rows, private, settings, cp.parameters(None))
    assert cp.CONFIG_KEY not in summary["private"] and cp.CONFIG_KEY not in summary["p2"]
    block = summary["p2_lever"][cp.CONFIG_KEY]
    assert block["receipt_scale_f9"] == pytest.approx(0.9)
    # CCGT 108 - 70 = 38 and hydro 45, both wholly private.
    assert block["saving_private_owners_gbp"] == pytest.approx(83.0)


def _entry():
    from scripts.consumer_bill.line_recovery import make_entry

    return make_entry("gsp_charging_volume_mwh", 315e6, household_loss_factor=1.1313, household_allocation_factor=1.348,
                      price_year_factor=1.03, system_price_year=2024, basis="test")


def test_the_bill_row_is_recovered_as_the_capacity_market_line():
    from scripts.consumer_bill.line_recovery import INHERITED_LINE_RECOVERY, household_to_system_gbp
    from scripts.public_ownership import project_public_ownership_consumer_bill as bill

    assert bill.LINE_CM_COST_BASED == cp.LINE and cp.LINE in bill.NEW_LINES
    assert INHERITED_LINE_RECOVERY[cp.LINE] == "capacity_market"
    account = {cp.CONFIG_KEY: cp.account_result(_rows(), _rows())}
    extra = {"line_recovery": {"capacity_market": _entry()}, "two_volume": True, "mwh": 3.1,
             "factor": 1.03, "final_demand_mwh": 290e6}
    [row] = bill.capacity_procurement_row(account, private_extra=extra, cm_factor=1.348, account_source="test")
    assert row["component_id"] == cp.LINE and row["annual_gbp_per_3100kwh_ex_vat"] < 0
    # Back to system money through the Capacity Market line's own entry: exactly the account's saving.
    assert household_to_system_gbp(_entry(), row["annual_gbp_per_3100kwh_ex_vat"], 3100.0) == pytest.approx(-100.0)
    # A D1 bill (2050): final demand over the rebasing factor, times the household peak factor.
    d1 = bill.capacity_procurement_household(100.0, private_extra={**extra, "two_volume": False}, cm_factor=1.348)
    assert d1 == pytest.approx(100.0 * 1.03 / 290e6 * 3.1 * 1.348)
    assert bill.capacity_procurement_row({}, private_extra=extra, cm_factor=1.348, account_source="test") == []


def test_the_saving_is_a_transfer_from_private_owners_in_the_ledger_and_the_flows():
    from scripts.public_ownership.investor_ledger import bill_line_moves, market_case_parties
    from scripts.public_ownership.party_flows import BILL_LINE_PARTY, Flows, market_flows

    saving = 678e6
    bill_case = {"capacity_market_cost_based_existing_gbp_2024": -saving}
    assert bill_line_moves(bill_case)[cp.LINE] == pytest.approx(-saving)
    position = {"private_rent_gbp": 0.0, "capital_recovery_gbp": 0.0, "required_revenue_gbp": 0.0,
                "corporation_tax_forgone_gbp": 0.0, "tax_equivalent_gbp": 0.0}
    parties = market_case_parties(bill_case=bill_case, bill_private={}, account_case={}, position=position,
                                  position_private=position,
                                  network={"premium_gbp": 0.0, "wedge_gbp": 0.0, "excess_return_gbp": 0.0})
    assert parties["bill_change_system_gbp"] == pytest.approx(-saving)
    investors = sum(parties["investor_parts_gbp"].values())
    resource = sum(parties["resource_parts_gbp"].values())
    # Consumers gain what private owners lose; nothing real is saved.
    assert investors == pytest.approx(-saving) and resource == pytest.approx(0.0)
    assert (-parties["bill_change_system_gbp"]) + investors == pytest.approx(resource)
    assert BILL_LINE_PARTY[cp.LINE][0] == "private_investors"
    flows = Flows("p2_lever", lambda gbp: gbp / 1e7)
    components = {k: 0.0 for k in ("state_capital", "private_capital", "state_operating", "private_operating",
                                   "corporation_tax", "carbon_cfd_private", "tax_equivalent_private", "ldes_state",
                                   "ldes_private")}
    market_flows(flows, parties=parties, components=components, account_case={}, network={}, levers={})
    [flow] = [r for r in flows.rows if r["flow_id"] == cp.LINE]
    assert (flow["payer"], flow["payee"], flow["gbp"]) == ("private_investors", "consumers", pytest.approx(saving))
