"""F19 decision 5 (user, 6 Oct 2026): the battery revenue top-up at each case's own cost.

Until then the top-up was the private case's shortfall, paid to every owner, so a state
battery was paid the private shortfall and the overpayment came back through the Capacity
Market booking.  Now each row is topped up to its own owner's required revenue; the bill
carries the difference on its own row and the battery part of the CM booking is retired.
Synthetic fixtures only: the live top-up sizes move with the ancillary-services data.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import party_flows
from scripts.public_ownership import support_lines_2050 as sl
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    LINE_BATTERY_TOP_UP_ADJ, NEW_LINES, battery_top_up_change, battery_top_up_row,
)


def _register():
    # B1: private, short by 50 before the top-up.  B1 under P2: the state's cheaper capital leaves it
    # short by 20.  B1 under P1: a 30% stake's joint venture short by 40.  B2: a battery earning above its
    # cost.  G1: not a battery.
    frame = pd.DataFrame({
        "case_id": ["private", "public_ownership", "public_control", "public_ownership", "public_ownership"],
        "asset_id": ["B1", "B1", "B1", "B2", "G1"],
        "carrier": ["Battery", "Battery", "Battery", "Battery", "CCGT"],
        "ownership_share": [0.0, 1.0, 0.3, 1.0, 1.0],
        "producer_rent_before_gbp": [-50.0, -20.0, -40.0, 15.0, -5.0],
    })
    frame["battery_revenue_top_up_gbp"] = [50.0, 50.0, 50.0, 0.0, 0.0]        # the private basis, booked
    frame["producer_rent_gbp"] = frame["producer_rent_before_gbp"] + frame["battery_revenue_top_up_gbp"]
    frame["public_surplus_before_recycling_gbp"] = frame["ownership_share"] * frame["producer_rent_gbp"]
    return frame


def test_each_row_is_topped_up_to_its_own_owner_cost():
    out = sl.rebook_battery_top_up_at_owner_cost(_register(), ["Battery"]).set_index(["case_id", "asset_id"])
    top = out["battery_revenue_top_up_gbp"]
    assert top[("private", "B1")] == 50.0                     # the private row is unchanged
    assert top[("public_ownership", "B1")] == 20.0            # the state's own shortfall
    assert top[("public_control", "B1")] == 40.0              # the joint venture's, at its rate
    assert top[("public_ownership", "B2")] == 0.0
    assert top[("public_ownership", "G1")] == 0.0
    # A topped-up row breaks even; the state's surplus moves by its share of the difference.
    assert out.loc[[("public_ownership", "B1"), ("public_control", "B1")], "producer_rent_gbp"].tolist() == [0.0, 0.0]
    # The stake's surplus was 0.3 x 10 of overpaid rent; the 10 the top-up falls takes its share back.
    assert out.loc[("public_control", "B1"), "public_surplus_before_recycling_gbp"] == pytest.approx(0.0)
    assert out.loc[("public_ownership", "G1"), "producer_rent_gbp"] == -5.0
    # The private-basis amount is kept beside it (and marks the basis to the account).
    assert out[sl.COLUMN_BATTERY_TOP_UP_PRIVATE_BASIS].tolist() == [50.0, 50.0, 50.0, 0.0, 0.0]
    assert sl.COLUMN_BATTERY_TOP_UP_PRIVATE_BASIS in sl.MEMO_COLUMNS


def test_the_basis_is_owner_cost_in_2030_and_private_in_a_vintaged_register():
    parsed = sl.settings({"battery_top_up": {"carriers": ["Battery"]}})
    assert sl.owner_cost_top_up(parsed, vintaged=False) == ["Battery"]
    assert sl.owner_cost_top_up(parsed, vintaged=True) is None
    on_2050 = sl.settings({"battery_top_up": {"owner_cost_in_vintaged_registers": True}})
    assert sl.owner_cost_top_up(on_2050, vintaged=True) == ["Battery"]
    assert sl.owner_cost_top_up(sl.settings({"battery_top_up": {"basis": "private"}}), vintaged=False) is None
    with pytest.raises(sl.SupportLinesError, match="basis"):
        sl.settings({"battery_top_up": {"basis": "average"}})


def test_the_bill_row_carries_the_case_top_up_less_the_private_one():
    private = {"battery_top_up_owner_cost": True, "battery_revenue_top_up_gbp": 300.0e6}
    owned = {"battery_top_up_owner_cost": True, "battery_revenue_top_up_gbp": 120.0e6}
    assert battery_top_up_change(owned, private) == pytest.approx(-180.0e6)
    # Off the basis (a register without the marker, e.g. 2050): nothing moves.
    assert battery_top_up_change({"battery_revenue_top_up_gbp": 120.0e6}, private) == 0.0
    rows = battery_top_up_row(owned, private_account=private, factor=1.04, final_demand_mwh=270.0e6, mwh=3.1,
                              account_source="acct")
    assert len(rows) == 1 and rows[0]["component_id"] == LINE_BATTERY_TOP_UP_ADJ in NEW_LINES
    assert rows[0]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-180.0e6 * 1.04 / 270.0e6 * 3.1)
    assert battery_top_up_row(private, private_account=private, factor=1.04, final_demand_mwh=270.0e6, mwh=3.1,
                              account_source="acct") == []


def test_the_flows_split_the_line_between_the_state_and_the_co_owners():
    flows = party_flows.Flows("p1", lambda gbp: gbp / 1e6)
    components = {k: 0.0 for k in ("state_capital", "private_capital", "state_operating", "private_operating",
                                   "corporation_tax", "takeover_positive", "takeover_negative", "carbon_cfd_private",
                                   "tax_equivalent_private", "ldes_state", "ldes_private")}
    parties = {"bill_lines_gbp": {LINE_BATTERY_TOP_UP_ADJ: -100.0e6}}
    party_flows.market_flows(flows, parties=parties, components=components,
                             account_case={"battery_top_up_change_state_gbp": -30.0e6}, network={}, levers={})
    got = {r["flow_id"]: (r["payer"], r["payee"], r["gbp"]) for r in flows.rows}
    assert got["battery_top_up_state"] == ("exchequer", "consumers", pytest.approx(30.0e6))
    assert got["battery_top_up_coowners"] == ("private_investors", "consumers", pytest.approx(70.0e6))


def test_the_takeover_nets_the_beccs_pair_and_splits_by_sign():
    # F19 decision 7: P2-like rows: an R1 row with positive private rent, a loss-making store, and a
    # BECCS row whose private rent includes its carbon CfD.
    register = pd.DataFrame({
        "case_id": ["private"] * 3 + ["public_ownership"] * 3,
        "asset_id": ["R1", "LDES", "BECCS"] * 2,
        "ownership_share": [0.0] * 3 + [1.0] * 3,
        "producer_rent_gbp": [100.0, -60.0, 40.0, 0.0, 0.0, 0.0],
        "carbon_removal_cfd_revenue_gbp": [0.0, 0.0, 30.0, 0.0, 0.0, 30.0],
        "annual_capital_recovery_gbp": [0.0] * 6, "annual_required_revenue_gbp": [0.0] * 6,
    })
    c = party_flows.register_components(register, "public_ownership")
    assert c["takeover_positive"] == pytest.approx(-140.0)    # R1 100 + BECCS 40, carbon CfD included
    assert c["takeover_negative"] == pytest.approx(60.0)      # the store's shortfall relieved
    assert c["carbon_cfd_private"] == pytest.approx(0.0)      # no co-owner: the pair is netted
    assert np.isclose(c["takeover_positive"] + c["takeover_negative"] + c["carbon_cfd_private"], -80.0)


def test_a_calibrated_cohort_is_not_split_into_gross_halves():
    # Two rows of one break-even cohort (+10 and -10) net to nothing; a loss-making store is relief.
    register = pd.DataFrame({
        "case_id": ["private"] * 3 + ["public_ownership"] * 3, "asset_id": ["A", "B", "L"] * 2,
        "carrier": ["wind_onshore", "wind_onshore", "LAES"] * 2, "support_type": ["CfD", "CfD", "merchant"] * 2,
        "cfd_round": ["AR7a", "AR7a", ""] * 2, "finance_vintage": ["new_build"] * 6,
        "ownership_share": [0.0] * 3 + [1.0] * 3, "producer_rent_gbp": [10.0, -10.0, -60.0, 0.0, 0.0, 0.0],
        "carbon_removal_cfd_revenue_gbp": [0.0] * 6, "annual_capital_recovery_gbp": [0.0] * 6,
        "annual_required_revenue_gbp": [0.0] * 6,
    })
    c = party_flows.register_components(register, "public_ownership")
    assert c["takeover_positive"] == pytest.approx(0.0)
    assert c["takeover_negative"] == pytest.approx(60.0)
