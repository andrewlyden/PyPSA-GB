"""Item BM1, step 5: the balancing-bid case beside the market cases, split three ways.

The case's bill is market private, plus what the bid stack does to a bill that
owns nothing (private on the group's months less private on the market months),
plus the ownership effect on that stack. The decomposition cross-check is
converted on the bill's own recovery volumes, bids over final demand and the CfD
offset over CfD-eligible supply.
"""

import copy

import pandas as pd
import pytest

from scripts.public_ownership.merge_balancing_groups import (
    balancing_components, check_comparable, cross_check, merge,
)

CASE = "public_ownership@state_bids_at_cost"
COMMON = {"benchmark_kwh": 3100.0, "price_basis": "GBP real 2025", "cfd_recovery_denominator_mwh": 300e6,
          "final_demand_recovery_denominator_mwh": 310e6, "model_rebasing_factor": 1.0,
          "private_reproduces_forecast_case": True}


def _rows(values):
    rows = []
    for case_id, lines in values.items():
        for component_id, value in lines.items():
            rows.append({"case_id": case_id, "component_id": component_id, "component": component_id.upper(),
                         "stack_group": "g", "annual_gbp_per_3100kwh_ex_vat": value})
    return pd.DataFrame(rows)


def _summary(totals, forecast_case):
    return {**COMMON, "forecast_case": forecast_case,
            "cases": {c: {"total_gbp_inc_vat": t * 1.05, "ex_vat_gbp": t, "vat_gbp": t * 0.05, "levers": {}}
                      for c, t in totals.items()}}


@pytest.fixture()
def bills():
    market = _rows({"private": {"bsuos": 40.0, "cfd": 130.0}, "public_ownership": {"bsuos": 40.0, "cfd": 100.0}})
    group = _rows({"private": {"bsuos": 35.0, "cfd": 132.0}, CASE: {"bsuos": 35.0, "cfd": 101.0}})
    return (market, _summary({"private": 170.0, "public_ownership": 140.0}, "CP30_2030_cp30_network"),
            group, _summary({"private": 167.0, CASE: 136.0}, "CP30_2030_cp30_network_public_bids_p2"))


def _decomposition(private_bids=-5e6, offers=0.0, offset=2e6, to_state=-1e6):
    return {"year": {"payment_to_private_counterparties_gbp": private_bids, "delta_offer_cost_gbp": offers,
                     "offsetting_private_cfd_settlement_gbp": offset, "transfer_to_the_state_gbp": to_state}}


def test_each_line_splits_into_market_private_balancing_and_ownership(bills):
    market, _, group, _ = bills
    lines = balancing_components(market, group, CASE).set_index("component_id")
    assert lines.loc["bsuos", "private_bid_stack_decomposition_term"] == pytest.approx(-5.0)
    assert lines.loc["cfd", "private_bid_stack_decomposition_term"] == pytest.approx(2.0)
    assert lines.loc["cfd", "ownership_effect_on_stack"] == pytest.approx(-31.0)
    assert lines.loc["cfd", "bid_stack_effect"] == pytest.approx(1.0)
    parts = lines["private_on_market"] + lines["private_bid_stack_decomposition_term"] + lines["ownership_effect_on_stack"]
    assert (parts - lines["case_on_group"]).abs().max() < 1e-12


def test_the_totals_add_and_the_bid_stack_is_against_the_same_owner(bills):
    market, market_summary, group, group_summary = bills
    _, summary = merge(market, market_summary, [
        {"name": "p2", "by_case": group, "summary": group_summary, "decomposition": None}])
    entry = summary["cases"][CASE]
    assert entry["private_bid_stack_decomposition_term_gbp_inc_vat"] == pytest.approx(-3.0 * 1.05)
    assert entry["ownership_effect_on_stack_gbp_inc_vat"] == pytest.approx(-31.0 * 1.05)
    assert entry["bid_stack_effect_gbp_inc_vat"] == pytest.approx(-4.0 * 1.05)
    assert entry["base_case"] == "public_ownership" and "cross_check" not in entry
    # Audit 26 Sep 2026: the P2 increment is the headline; private on the state's bid
    # stack is not a feasible case and carries no name that reads as a saving.
    assert entry["headline_metric"] == "bid_stack_effect_gbp_inc_vat"
    assert not any("balancing_effect" in key for key in entry)


def test_bills_on_different_recovery_volumes_are_refused(bills):
    market, market_summary, group, group_summary = bills
    other = copy.deepcopy(group_summary)
    other["final_demand_recovery_denominator_mwh"] = 311e6
    # F19 integration (7 Oct 2026): the recovery volumes are compared through bill_recovery.recovery_signature
    # (each line's entry on the two-volume basis; on D1, as here, the two denominators).
    with pytest.raises(ValueError, match="recovery volumes differ.*final_demand_mwh"):
        check_comparable(market_summary, other, "p2")
    other = copy.deepcopy(group_summary)
    other["private_reproduces_forecast_case"] = False
    with pytest.raises(ValueError, match="does not reproduce"):
        check_comparable(market_summary, other, "p2")


def test_a_bill_with_different_lines_is_refused(bills):
    market, _, group, _ = bills
    extra = pd.concat([group, _rows({CASE: {"new_line": 1.0}})], ignore_index=True)
    with pytest.raises(ValueError, match="new_line"):
        balancing_components(market, extra, CASE)


def test_an_ownership_only_line_is_zero_on_the_private_bills(bills):
    # The Capacity Market public financing adjustment exists only where something is
    # owned; private's bills lack it, which is a zero, not a different bill (25 Sep 2026).
    market, _, group, _ = bills
    line = "capacity_market_public_financing_adjustment"
    market = pd.concat([market, _rows({"public_ownership": {line: -1.9}})], ignore_index=True)
    group = pd.concat([group, _rows({CASE: {line: -1.9}})], ignore_index=True)
    lines = balancing_components(market, group, CASE).set_index("component_id")
    assert lines.loc[line, "private_on_group"] == 0.0 and lines.loc[line, "private_on_market"] == 0.0
    assert lines.loc[line, "ownership_effect_on_stack"] == pytest.approx(-1.9)
    assert lines.loc[line, "bid_stack_effect"] == pytest.approx(0.0)


def test_a_line_only_private_carries_is_refused(bills):
    market, _, group, _ = bills
    market = pd.concat([market, _rows({"private": {"odd": 1.0}})], ignore_index=True)
    group = pd.concat([group, _rows({"private": {"odd": 1.0}})], ignore_index=True)
    with pytest.raises(ValueError, match="odd"):
        balancing_components(market, group, CASE)


def test_the_cross_check_uses_each_line_recovery_volume(bills):
    _, _, _, group_summary = bills
    check = cross_check(_decomposition(), bid_stack_effect_inc_vat=-0.05, summary=group_summary, vat_rate=0.05)
    per_mwh = 3.1 * 1.05
    expected = -5e6 * per_mwh / 310e6 + 2e6 * per_mwh / 300e6
    assert check["decomposition_estimate_gbp_per_household_inc_vat"] == pytest.approx(expected)
    assert check["gap_gbp_per_household_inc_vat"] == pytest.approx(-0.05 - expected)
    # The transfer to the state is reported, never counted: it returns through the public account.
    assert check["decomposition_transfer_to_the_state_gbp"] == -1e6


def test_the_cross_check_follows_the_state_strike_basis(bills):
    """At cost the state's transfer and CfD settlement stay off the bill; on a fixed strike both reach it."""
    _, _, _, group_summary = bills
    decomposition = _decomposition()
    decomposition["year"]["state_cfd_settlement_at_fixed_strike_gbp"] = -4e6
    per_mwh = 3.1 * 1.05
    private_side = -5e6 * per_mwh / 310e6 + 2e6 * per_mwh / 300e6
    at_cost = cross_check(decomposition, -0.05, group_summary, 0.05, {"cfd_strike_basis": "at_cost"})
    assert at_cost["decomposition_estimate_gbp_per_household_inc_vat"] == pytest.approx(private_side)
    fixed = cross_check(decomposition, -0.05, group_summary, 0.05,
                        {"cfd_strike_basis": "auction_clearing", "surplus_recycling": "retained_by_treasury"})
    expected = private_side + (-1e6) * per_mwh / 310e6 + (-4e6) * per_mwh / 300e6
    assert fixed["decomposition_estimate_gbp_per_household_inc_vat"] == pytest.approx(expected)
    assert fixed["state_cfd_strike_basis"] == "auction_clearing"
    # An old decomposition without the state term is flagged, not silently used.
    stale = cross_check(_decomposition(), -0.05, group_summary, 0.05, {"cfd_strike_basis": "auction_clearing"})
    assert stale["decomposition_estimate_gbp_per_household_inc_vat"] == pytest.approx(private_side)
    assert "rerun" in stale["decomposition_basis"]
