"""The account's battery top-up split is measured against private on the same register (14 Oct 2026).

A balancing group holds its support lines at the market group's amounts (``support_from_reference``),
so a row's own private-basis column carries private's move between the two months. The split of
``battery_top_up_change_state_gbp`` / ``_coowners_gbp`` is the case's top-up less private's on the
same register, as ``party_flows.register_components`` and the bill line measure it: private's own
change is zero and the two parts sum to the bill's change.
"""
import pandas as pd
import pytest
import yaml
from pathlib import Path

from scripts.public_ownership.cfd_repricing import per_row_savings, repricing_settings
from scripts.public_ownership.project_public_ownership_consumer_bill import battery_top_up_change
from scripts.public_ownership.public_account import (
    BATTERY_TOP_UP_PRIVATE_BASIS, _private_battery_top_up, account_settings, build_public_account,
)

from tests.unit.test_public_ownership_public_account import _settled_register
from config.config_loader import load_defaults


@pytest.fixture(scope="module")
def defaults() -> dict:
    return load_defaults()


def _balancing_group_register():
    """The fixture register with the battery top-up held at the market group's amounts.

    Private's top-up is 10 m on the group's register; its own private-basis column (sized on the group's
    months) says 6 m, the stale basis that put +4 m into private's co-owner change.
    """
    register, _ = _settled_register()
    register = register.copy()
    register["battery_revenue_top_up_gbp"] = 0.0
    register[BATTERY_TOP_UP_PRIVATE_BASIS] = 0.0
    battery = register["asset_id"].eq("battery")
    private = register["case_id"].eq("private")
    register.loc[battery & private, "battery_revenue_top_up_gbp"] = 10.0e6
    register.loc[battery & ~private, "battery_revenue_top_up_gbp"] = 7.0e6
    register.loc[battery, BATTERY_TOP_UP_PRIVATE_BASIS] = 6.0e6
    return register


def test_private_moves_nothing_and_the_parts_sum_to_the_bill_change(defaults):
    register = _balancing_group_register()
    savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    _, report = build_public_account(register, savings, account_settings(defaults["public_ownership"]))
    private = report["private"]
    assert private["battery_top_up_change_state_gbp"] == pytest.approx(0.0)
    assert private["battery_top_up_change_coowners_gbp"] == pytest.approx(0.0)
    for case_id in ("public_coordination", "public_ownership"):
        case = report[case_id]
        parts = case["battery_top_up_change_state_gbp"] + case["battery_top_up_change_coowners_gbp"]
        assert parts == pytest.approx(battery_top_up_change(case, private))
        assert parts == pytest.approx(-3.0e6)
        share = float(register.loc[register.case_id.eq(case_id) & register.asset_id.eq("battery"),
                                   "ownership_share"].iloc[0])
        assert case["battery_top_up_change_state_gbp"] == pytest.approx(share * -3.0e6)


def test_a_streamed_batch_reads_private_through_rows_for():
    register = _balancing_group_register()
    batch = register.loc[register.case_id.eq("public_ownership")]
    top_up = _private_battery_top_up(batch, rows_for=lambda case_id: register.loc[register.case_id.eq(case_id)])
    assert top_up["battery"] == pytest.approx(10.0e6)
    assert _private_battery_top_up(batch) is None
    assert _private_battery_top_up(register)["battery"] == pytest.approx(10.0e6)
