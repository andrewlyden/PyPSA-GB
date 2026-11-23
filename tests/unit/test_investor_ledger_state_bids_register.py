"""The state-bids case reads its own run group's register, columns and all (14 Oct 2026).

Two faults the 2030 ledger hit (residual GBP -494.65 m on public_ownership@state_bids_at_cost):

* the Exchequer took the case's corporation tax forgone (and tax equivalents, carbon CfD) from the
  market register, settled on the market months, while its bill, its account and the ownership step
  read its own group's register; since F19 decision 3 the two differ (751.5 m against 227.8 m);
* the ledger loaded registers without ``battery_revenue_top_up_gbp``, so the dispatch step saw a zero
  top-up and paid the stack's move to investors twice (+29.06 m).

and, found once the identity closed, the party flows split the battery top-up line with the account's
private-basis amounts, which a balancing group holds at the market group's; the line itself is measured
against the group's own private (+26.53 m to the Exchequer).

A hand-checkable state-bids case: the state owns a wind farm and a battery (P2), its bid stack trims the
private fleet's balancing revenue by 5 m and raises a private battery's top-up by 1 m, the corporation
tax the state forgoes is 12 m on the market months but 4 m on its own, and the state's battery needs
1 m less top-up at its own cost (the account, on the market basis, says 0.5 m).
"""

import pandas as pd
import pytest

from scripts.public_ownership import investor_ledger as inv
from scripts.public_ownership.exchequer_ledger import build, load_register

M = 1.0e6
CONV = {"factor": 1.04, "final_demand_mwh": 300.0e6, "mwh": 3.1, "vat_rate": 0.05, "domestic_mwh": 100.0e6}
CARBON = {"carbon_receipts_gbp": 1.5e9, "carbon_uk_ets_gbp": 1.5e9, "carbon_price_support_gbp": 0.0,
          "chp_heat_credit_carbon_gbp": 0.0}
NETWORK = pd.DataFrame({"treatment": ["acquired"], "sector": ["distribution"], "rav_gbp": [1000 * M],
                        "allowed_wacc_cpih_real": [0.05], "financing_saving_gbp": [17 * M],
                        "tax_allowance_saving_gbp": [5 * M]})
CASE = "p2@state_bids"
RATE = 1.0e-6  # household GBP per system GBP, every moved line alike


def _row(case, asset, *, share, rr, market, support=0.0, balancing=0.0, top_up=0.0, ct=0.0, capital=0.0):
    rent = market + support + top_up - rr
    return {"case_id": case, "asset_id": asset, "finance_vintage": "new_build", "finance_technology": "Offshore wind",
            "ownership_share": share, "state_owned": share > 0, "annual_capital_recovery_gbp": capital,
            "wacc": 0.05, "remaining_life_years": 25.0, "public_return": "public_rate",
            "corporation_tax_forgone_gbp": ct, "annual_required_revenue_gbp": rr, "market_revenue_gbp": market,
            "support_revenue_gbp": support, "capacity_market_revenue_gbp": 0.0, "ldes_floor_topup_paid_gbp": 0.0,
            "fuel_carbon_gbp": 0.0, "balancing_settlement_gbp": balancing, "battery_revenue_top_up_gbp": top_up,
            "producer_rent_gbp": rent, "public_surplus_before_recycling_gbp": share * rent}


def _fleet(case, *, share, wind_market, balancing, top_up, ct):
    rr = 70 * M if share else 100 * M
    # The state's battery: 1 m cheaper capital, so its own shortfall (the top-up) is 1 m, not 2 m.
    store_rr, store_top_up = (9 * M, 1 * M) if share else (10 * M, 2 * M)
    return [_row(case, "wind", share=share, rr=rr, market=wind_market, support=50 * M, balancing=balancing, ct=ct,
                 capital=rr),
            _row(case, "battery", share=0.0, rr=10 * M, market=8 * M, top_up=top_up, capital=6 * M),
            _row(case, "state_battery", share=share, rr=store_rr, market=8 * M, top_up=store_top_up,
                 capital=store_rr - 4 * M)]


def _market_register():
    # The market group registers every case, the state-bids case too, settled on the market months.
    return pd.DataFrame(_fleet("private", share=0.0, wind_market=60 * M, balancing=5 * M, top_up=2 * M, ct=0.0)
                        + _fleet("p2", share=1.0, wind_market=60 * M, balancing=5 * M, top_up=2 * M, ct=12 * M)
                        + _fleet(CASE, share=1.0, wind_market=60 * M, balancing=5 * M, top_up=2 * M, ct=12 * M))


def _group_register():
    # The case's own months: 5 m less balancing revenue, 1 m more top-up, 4 m of corporation tax forgone.
    return pd.DataFrame(_fleet("private", share=0.0, wind_market=55 * M, balancing=0.0, top_up=3 * M, ct=0.0)
                        + _fleet(CASE, share=1.0, wind_market=55 * M, balancing=0.0, top_up=3 * M, ct=4 * M))


def _bill_case(**kw):
    base = {"ex_vat_gbp": 900.0, "total_gbp_inc_vat": 945.0, "financing_saving_gbp_2024": 0.0,
            "levers": {"networks": "private"}, "route_booking_booked_gbp_2024": {},
            "network_adjustments_system_gbp_2024": {}}
    return {**base, **kw}


def _account(retained):
    return {"cases": {"private": {"state_deficit_exchequer_funded_gbp": 0.0, "state_surplus_retained_gbp": 0.0},
                      "p2": {"state_deficit_exchequer_funded_gbp": 0.0, "state_surplus_retained_gbp": 20 * M,
                             "r1_bill_cut_gbp": 20 * M, "r1_cut_borne_by_state_gbp": 20 * M,
                             "battery_top_up_owner_cost": True, "battery_top_up_change_state_gbp": -1 * M},
                      CASE: {"state_deficit_exchequer_funded_gbp": 0.0, "state_surplus_retained_gbp": retained,
                             "r1_bill_cut_gbp": 20 * M, "r1_cut_borne_by_state_gbp": 20 * M}}}


def _build(tmp_path, *, group_register=None):
    if group_register is None:
        path = tmp_path / "grp_asset_finance_register.csv"
        _group_register().to_csv(path, index=False)
        # Loaded as the ledger loads it (investor_inputs): its own column list, no other.
        group_register = load_register(path, [CASE, "private"])
    lines = pd.DataFrame({
        "component_id": ["bsuos_constraints", "battery_revenue_top_up"], "case_id": [CASE, CASE],
        "private_on_market": [40.0, 4.0], "private_on_group": [40.0 - 5 * M * RATE, 4.0 + 1 * M * RATE]})
    battery = {"battery_top_up_owner_cost_gbp_2024": -1 * M}
    market_bill = {"cases": {"private": _bill_case(),
                             "p2": _bill_case(ex_vat_gbp=899.8, total_gbp_inc_vat=899.8 * 1.05,
                                              bill_allocated_r1_gbp_2024=20 * M, **battery)}}
    group_bill = {"cases": {"private": _bill_case(), CASE: _bill_case(bill_allocated_r1_gbp_2024=20 * M, **battery)}}
    balancing = {"market_forecast_case": "market", "cases": {CASE: {
        "run_group": "grp", "base_case": "p2", "total_gbp_inc_vat": 940.0, "levers": {"networks": "private"}}}}
    investor = {"network_public_rate": 0.03, "balancing": {CASE: {
        "register": group_register, "bill": group_bill, "lines": lines,
        "rates": {"bsuos_constraints": RATE, "battery_revenue_top_up": RATE},
        "retrading_market_gbp": 0.0, "retrading_group_gbp": 0.0}}}
    account = _account(retained=0.0)
    group_account = {"cases": {"private": account["cases"]["private"],
                               CASE: {"state_deficit_exchequer_funded_gbp": 0.0, "state_surplus_retained_gbp": 15 * M,
                                      "r1_bill_cut_gbp": 20 * M, "r1_cut_borne_by_state_gbp": 20 * M,
                                      # On the private basis the group holds at the market group's amounts
                                      # (support_from_reference): half the line, as on 14 Oct 2026.
                                      "battery_top_up_owner_cost": True,
                                      "battery_top_up_change_state_gbp": -0.5 * M,
                                      "battery_top_up_change_coowners_gbp": 0.0}}}
    return build(register=_market_register(), bill=market_bill, account=account, network_lines=NETWORK, conv=CONV,
                 wedges={}, carbon={"market": CARBON, "balancing": CARBON}, balancing=balancing,
                 balancing_accounts={CASE: group_account}, investor=investor)


def test_the_ledger_reads_every_instrument_the_dispatch_step_compares():
    instruments = {column for _, column in inv.DISPATCH_INSTRUMENTS.values()} | set(inv.RENT_ONLY_INSTRUMENTS)
    assert instruments <= set(inv.REGISTER_COLUMNS)


def test_a_register_loaded_by_the_ledger_carries_the_battery_top_up(tmp_path):
    path = tmp_path / "grp_asset_finance_register.csv"
    _group_register().to_csv(path, index=False)
    rows = load_register(path, [CASE, "private"]).rows("private")
    # 3 m on the private battery and 2 m on the battery the state owns in P2.
    assert inv.register_position(rows, "private")["battery_revenue_top_up_gbp"] == pytest.approx(5 * M)


def test_the_state_bids_case_closes_on_its_own_groups_register(tmp_path):
    _, summary = _build(tmp_path)
    row = summary["cases"][CASE]
    assert abs(row["three_party_identity_residual_gbp"]) <= inv.IDENTITY_TOLERANCE_GBP
    # The corporation tax the Exchequer forgoes is the case's own months' 4 m, not the market months' 12 m.
    assert row["corporation_tax_forgone_gbp"] == pytest.approx(4 * M)
    assert row["register_run_group"] == "grp"
    # The battery top-up's move reaches investors once: through the rent, nothing left on the line.
    key = "battery top-up line against the top-up the register books: paid to parties the register does not book"
    assert row["investor_change_vs_private_parts_gbp"][key] == pytest.approx(0.0, abs=1.0)
    # Resource: the wind farm's and the battery's capital at the public rate (30 m + 1 m) less the 4 m of
    # tax inside the private rate.
    assert row["resource_saving_vs_private_gbp"] == pytest.approx(27 * M)
    # The party flows closed (build refuses otherwise): the state's battery takes the whole line.
    flows = {f["flow_id"]: f for f in summary["_party_flow_rows"] if f["case_id"] == CASE}
    state = flows["battery_top_up_state"]
    assert (state["payer"], state["payee"], state["gbp_m"]) == ("exchequer", "consumers", pytest.approx(1.0))
    assert "battery_top_up_coowners" not in flows  # a zero flow is not written


def test_a_tax_read_off_another_register_than_the_ownership_steps_is_refused(tmp_path):
    # A group register that carries no corporation tax column falls back to the market register's rows
    # (12 m forgone) while the ownership step reads none: the identity refuses the 12 m, as it refused
    # the 523.7 m of 14 Oct 2026. The check is never loosened to let such a case through.
    group = _group_register().drop(columns=["corporation_tax_forgone_gbp"])
    with pytest.raises(ValueError, match="a transfer is booked to one party"):
        _build(tmp_path, group_register=group)


def test_the_battery_line_is_split_on_the_register_as_the_bill_measures_it():
    from scripts.public_ownership import party_flows as pf

    group = _group_register()
    parts = pf.register_components(group, CASE)
    assert parts["battery_top_up_state"] == pytest.approx(-1 * M)
    assert parts["battery_top_up_private"] == pytest.approx(0.0)
    # Without the column the account's split is kept (registers that predate the top-up).
    assert "battery_top_up_state" not in pf.register_components(group.drop(columns=["battery_revenue_top_up_gbp"]), CASE)
