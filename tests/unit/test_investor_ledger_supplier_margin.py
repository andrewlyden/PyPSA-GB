"""The bid stack moves the supplier line through F18's EBIT and headroom (8 Oct 2026): investors, never resource."""
import pandas as pd
import pytest

from scripts.public_ownership import investor_ledger as investors

COLUMNS = ("producer_rent_gbp", "unresolved_net_cash_gbp", "revenue_gbp", "fuel_carbon_gbp", "required_revenue_gbp",
           "wholesale_revenue_gbp", "balancing_settlement_gbp", "cfd_support_gbp", "roc_support_gbp",
           "ldes_floor_topup_paid_gbp", "battery_revenue_top_up_gbp")


def _position(**values):
    return {column: float(values.get(column, 0.0)) for column in COLUMNS}


def test_the_supplier_margin_move_is_paid_to_suppliers_and_moves_no_resource():
    lines = {"bsuos_constraints": -80.0e6, "supplier_costs_and_margin": -2.3e6}
    out = investors.dispatch_step_parties(line_changes=lines, position_group=_position(), position_market=_position(),
                                          carbon_change_gbp=0.0, retrading_change_gbp=0.0)
    key = "supplier EBIT and headroom on the moved costs (retail suppliers, outside the register)"
    assert out["investor_parts_gbp"][key] == pytest.approx(-2.3e6)
    assert out["bill_change_system_gbp"] == pytest.approx(-82.3e6)
    assert sum(out["resource_parts_gbp"].values()) == pytest.approx(0.0)


def test_a_line_with_no_party_is_still_refused():
    with pytest.raises(ValueError, match="cannot assign"):
        investors.dispatch_step_parties(line_changes={"duos": 1.0}, position_group=_position(),
                                        position_market=_position(), carbon_change_gbp=0.0, retrading_change_gbp=0.0)


def test_line_changes_need_a_rate_for_every_moved_line():
    lines = pd.DataFrame({"component_id": ["supplier_costs_and_margin"], "private_on_group": [197.45],
                          "private_on_market": [197.43]})
    with pytest.raises(ValueError, match="no recovery rate"):
        investors.dispatch_line_changes(lines, {})
    assert investors.dispatch_line_changes(lines, {"supplier_costs_and_margin": 1e-8})["supplier_costs_and_margin"] \
        == pytest.approx(0.02 / 1e-8)


def test_the_battery_top_up_is_paid_to_owners_once():
    """The register books the top-up into rent, not revenue: the stack's move reaches investors once."""
    out = investors.dispatch_step_parties(
        line_changes={"battery_revenue_top_up": 10.0e6},
        position_group=_position(producer_rent_gbp=10.0e6, battery_revenue_top_up_gbp=10.0e6),
        position_market=_position(), carbon_change_gbp=0.0, retrading_change_gbp=0.0)
    assert sum(out["investor_parts_gbp"].values()) == pytest.approx(10.0e6)
    assert sum(out["resource_parts_gbp"].values()) == pytest.approx(0.0)
