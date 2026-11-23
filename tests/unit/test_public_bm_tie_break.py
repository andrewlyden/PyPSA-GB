"""Decision 20: P3's balancing tie is broken last in, first off, by commissioning year.

With every CfD unit state-owned, every one bids the same avoided short-run cost and
the balancing LP has nothing to order curtailment by, so which unit is turned down
becomes solver-arbitrary and the CfD and curtailment lines inherit that. The user
chose last in, first off by commissioning date (the principle of access in DNO
active network management schemes): the newest state-owned plant is turned down
first, through small bid steps keyed on the register's commissioning year,

    tie step(g) = step x (newest year - commissioning year(g)),

added to the state-owned part of the bid only, and recorded in the audit so every
money total can be reported without it. The gate is a week solved under LIFO and
under its reverse: the money totals net of the steps must match.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from scripts.market.cfd_bm_bids import TIE_BREAK_RULES, lifo_tie_steps, resolve_tie_break
from scripts.public_ownership.ownership_shares import generator_state_shares, load_commissioning_years

LOG = logging.getLogger("test")


def _register():
    return pd.DataFrame([
        {"case_id": "p3", "asset_id": "OLD::legacy", "capacity_mw": 100.0, "ownership_share": 1.0,
         "commissioning_year": 2015},
        {"case_id": "p3", "asset_id": "NEW::new_build", "capacity_mw": 100.0, "ownership_share": 1.0,
         "commissioning_year": 2029},
        {"case_id": "p3", "asset_id": "MIX::legacy", "capacity_mw": 300.0, "ownership_share": 1.0,
         "commissioning_year": 2020},
        {"case_id": "p3", "asset_id": "MIX::new_build", "capacity_mw": 100.0, "ownership_share": 1.0,
         "commissioning_year": 2028},
        {"case_id": "p3", "asset_id": "PRIV::legacy", "capacity_mw": 50.0, "ownership_share": 0.0,
         "commissioning_year": np.nan},
    ])


def test_the_share_table_carries_a_state_weighted_commissioning_year(tmp_path):
    table = generator_state_shares(_register()).set_index("generator")
    assert table.at["OLD", "state_commissioning_year"] == 2015
    # 300 MW of 2020 and 100 MW of 2028, both state-owned: 2022.
    assert table.at["MIX", "state_commissioning_year"] == pytest.approx(2022.0)
    assert np.isnan(table.at["PRIV", "state_commissioning_year"])
    path = tmp_path / "shares.csv"
    table.reset_index().to_csv(path, index=False)
    years = load_commissioning_years(path, "p3")
    assert years["NEW"] == 2029


def test_lifo_turns_the_newest_plant_down_first():
    years = pd.Series({"OLD": 2015.0, "MIX": 2022.0, "NEW": 2029.0})
    steps = lifo_tie_steps(years, step=0.05, rule="lifo_commissioning")
    assert steps["NEW"] == 0.0            # cheapest to turn down
    assert steps["MIX"] == pytest.approx(0.05 * 7)
    assert steps["OLD"] == pytest.approx(0.05 * 14)


def test_the_reverse_ordering_exists_for_the_gate():
    years = pd.Series({"OLD": 2015.0, "NEW": 2029.0})
    steps = lifo_tie_steps(years, step=0.05, rule="fifo_commissioning")
    assert steps["OLD"] == 0.0 and steps["NEW"] == pytest.approx(0.7)
    assert set(TIE_BREAK_RULES) == {"lifo_commissioning", "fifo_commissioning"}


def test_a_state_owned_unit_without_a_year_is_refused():
    with pytest.raises(ValueError, match="commissioning year"):
        lifo_tie_steps(pd.Series({"A": 2015.0, "B": np.nan}), step=0.05, rule="lifo_commissioning")


@pytest.mark.parametrize("step", [0.0, 0.005, 1.5])
def test_the_step_stays_inside_the_decided_range(step):
    with pytest.raises(ValueError, match="0.01"):
        lifo_tie_steps(pd.Series({"A": 2015.0}), step=step, rule="lifo_commissioning")


def test_the_tie_break_is_off_unless_declared(tmp_path):
    path = tmp_path / "shares.csv"
    generator_state_shares(_register()).to_csv(path, index=False)
    shares = pd.Series({"OLD": 1.0, "NEW": 1.0})
    assert resolve_tie_break(None, {"shares_file": str(path), "case_id": "p3"}, shares, LOG) is None
    assert resolve_tie_break({"enabled": False}, {"shares_file": str(path), "case_id": "p3"}, shares, LOG) is None
    steps = resolve_tie_break({"enabled": True, "rule": "lifo_commissioning", "step_gbp_per_mwh": 0.05},
                              {"shares_file": str(path), "case_id": "p3"},
                              pd.Series({"OLD": 1.0, "NEW": 1.0, "MIX": 1.0, "PRIV": 0.0}), LOG)
    # Only state-owned units are ranked; private plant keeps its contract-value bid.
    assert "PRIV" not in steps.index
    assert steps["NEW"] == 0.0 and steps["OLD"] == pytest.approx(0.7)


# ── the decomposition nets the steps out, and the gate compares two orderings ──

def _write_run(tmp_path, scenario, rows, steps):
    market = tmp_path / "market"
    market.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(market / f"{scenario}_redispatch_summary.csv", index=False)
    if not steps:
        return market
    pd.DataFrame([{"generator": g, "pricing": "cfd_contract_value", "mean_bid_gbp_per_mwh": 0.0,
                   "tie_break_step_gbp_per_mwh": s} for g, s in steps.items()]).to_csv(
        market / f"{scenario}_contract_value_bids.csv", index=False)
    return market


def _unit(name, bid_cost, decrease, carrier="wind_offshore"):
    return {"component": name, "carrier": carrier, "bid_cost": bid_cost, "offer_cost": 0.0,
            "decrease_MWh": decrease, "increase_MWh": 0.0}


def test_the_decomposition_reports_bid_cost_net_of_the_tie_steps(tmp_path, monkeypatch):
    from scripts.market import decompose_public_bm_bids as d

    market = _write_run(tmp_path, "base", [_unit("A", 500.0, 10.0), _unit("B", 500.0, 10.0)], {})
    _write_run(tmp_path, "p3", [_unit("A", 7.0, 10.0), _unit("B", 0.0, 10.0)], {"A": 0.7, "B": 0.0})
    monkeypatch.setattr(d, "RESOURCES", market)
    result, _ = d.decompose("base", "p3", pd.Series({"A": 1.0, "B": 1.0}))
    assert result["tie_break_step_cost_gbp"] == pytest.approx(7.0)
    assert result["public_bid_cost_gbp"] == pytest.approx(0.0)   # net of the steps
    assert result["transfer_to_the_state_gbp"] == pytest.approx(-1000.0)


def test_two_orderings_pass_the_gate_only_when_their_net_totals_match(tmp_path, monkeypatch):
    from scripts.market import decompose_public_bm_bids as d

    # LIFO turns B (new) down, FIFO turns A (old) down: same volume, same net cost.
    market = _write_run(tmp_path, "lifo", [_unit("A", 0.0, 0.0), _unit("B", 0.0, 10.0)], {"A": 0.7, "B": 0.0})
    _write_run(tmp_path, "fifo", [_unit("A", 0.0, 10.0), _unit("B", 0.0, 0.0)], {"A": 0.0, "B": 0.7})
    monkeypatch.setattr(d, "RESOURCES", market)
    gate = d.compare_tie_break_orderings("lifo", "fifo")
    assert gate["passed"], gate
    assert gate["decrease_MWh"] == (10.0, 10.0)
    # Different volumes are a failure: the tie moved something real.
    _write_run(tmp_path, "fifo", [_unit("A", 0.0, 12.0), _unit("B", 0.0, 0.0)], {"A": 0.0, "B": 0.7})
    assert not d.compare_tie_break_orderings("lifo", "fifo")["passed"]


def test_the_gate_nets_renewable_swaps_out_of_the_volume(tmp_path, monkeypatch):
    # 26 Sep 2026: LIFO and FIFO differed by 1.3% gross, all of it wind turned up and
    # down at once at ~0 cost. Net of those swaps they agreed to 0.07%.
    from scripts.market import decompose_public_bm_bids as d

    idx = pd.date_range("2030-01-03", periods=2, freq="h")
    market = tmp_path / "market"
    market.mkdir()
    pd.DataFrame({"A": [100.0, 100.0], "B": [50.0, 50.0], "C": [80.0, 80.0]}, index=idx).to_csv(
        market / "ws_wholesale_dispatch.csv")
    # lifo: A down 30 (needed) plus a 20 MWh A-down/B-up swap each hour; fifo: A down 30 only.
    pd.DataFrame({"A": [50.0, 50.0], "B": [70.0, 70.0], "C": [80.0, 80.0]}, index=idx).to_csv(
        market / "lifo_balancing_dispatch.csv")
    pd.DataFrame({"A": [70.0, 70.0], "B": [50.0, 50.0], "C": [80.0, 80.0]}, index=idx).to_csv(
        market / "fifo_balancing_dispatch.csv")
    rows = lambda down_a, up_b: [  # noqa: E731
        {**_unit("A", 0.0, down_a), "carrier": "wind_offshore"},
        {**_unit("B", 0.0, 0.0), "carrier": "wind_onshore", "increase_MWh": up_b},
        {**_unit("C", 0.0, 0.0), "carrier": "CCGT", "increase_MWh": 60.0},
    ]
    _write_run(tmp_path, "lifo", rows(100.0, 40.0), {"A": 0.0})
    _write_run(tmp_path, "fifo", rows(60.0, 0.0), {"A": 0.0})
    monkeypatch.setattr(d, "RESOURCES", market)
    monkeypatch.setattr(d, "ANALYSIS", tmp_path / "none")
    gross = d.compare_tie_break_orderings("lifo", "fifo")
    assert not gross["passed"] and not gross["swaps_netted"]
    gate = d.compare_tie_break_orderings("lifo", "fifo", wholesale="ws")
    assert gate["swaps_netted"] and gate["renewable_swap_MWh"] == (40.0, 0.0)
    assert gate["decrease_net_of_swaps_MWh"] == (60.0, 60.0)
    assert gate["passed"], gate


def test_the_turn_up_floor_prices_every_state_owned_renewable():
    from scripts.market.cfd_bm_bids import resolve_turn_up_floor

    shares = pd.Series({"WIND": 1.0, "SOLAR": 0.5, "CHP": 1.0, "PRIV": 0.0})
    carriers = pd.Series({"WIND": "wind_offshore", "SOLAR": "solar_pv", "CHP": "CHP", "PRIV": "wind_onshore"})
    tie = {"enabled": True, "turn_up_floor_gbp_per_mwh": 2.0}
    floor = resolve_turn_up_floor(tie, shares, carriers, LOG)
    assert floor.to_dict() == {"WIND": 2.0, "SOLAR": 1.0}      # thermal and private plant untouched
    assert resolve_turn_up_floor({"enabled": True}, shares, carriers, LOG) is None
    with pytest.raises(ValueError, match="at most"):
        resolve_turn_up_floor({"enabled": True, "turn_up_floor_gbp_per_mwh": 50.0}, shares, carriers, LOG)


def test_the_turn_up_floor_never_reaches_a_reported_cost():
    from types import SimpleNamespace

    from scripts.market.solve_balancing import _apply_offer_regulariser, _offer_regulariser_cost

    idx = pd.date_range("2030-01-03", periods=2, freq="h")
    offers = pd.DataFrame({"WIND": [0.02, 0.02], "CCGT": [120.0, 120.0]}, index=idx)
    floor = pd.Series({"WIND": 2.0})
    objective = _apply_offer_regulariser(offers, floor, LOG)
    assert objective["WIND"].tolist() == [2.02, 2.02] and objective["CCGT"].tolist() == [120.0, 120.0]
    assert offers["WIND"].tolist() == [0.02, 0.02]             # the reporting frame is not mutated
    physical = pd.DataFrame({"WIND": [110.0, 90.0], "CCGT": [0.0, 0.0]}, index=idx)
    schedule = pd.DataFrame({"WIND": [100.0, 100.0], "CCGT": [0.0, 0.0]}, index=idx)
    network = SimpleNamespace(snapshot_weightings=pd.DataFrame({"generators": [1.0, 1.0]}, index=idx))
    # Only the 10 MWh turned up carries the floor.
    assert _offer_regulariser_cost(physical, schedule, floor, network) == pytest.approx(20.0)
    assert _apply_offer_regulariser(offers, None, LOG) is offers


def test_the_state_first_step_lands_on_private_merchant_plant_only():
    # P2: state shares and merchant plant both bid 0; the step makes the state turn down first.
    from types import SimpleNamespace

    from scripts.market.cfd_bm_bids import resolve_state_first_step
    from scripts.market.solve_balancing import _offer_regulariser_cost

    shares = pd.Series({"MERCH": 0.0, "MERCH_HALF": 0.5, "CFD_STATE": 1.0})
    block = {"enabled": True, "state_first_step_gbp_per_mwh": 1.0}
    step = resolve_state_first_step(block, shares, pd.Index(["MERCH", "MERCH_HALF", "NEW_MERCH"]), LOG)
    assert step.to_dict() == {"MERCH": 1.0, "MERCH_HALF": 0.5, "NEW_MERCH": 1.0}
    assert resolve_state_first_step({"enabled": True}, shares, pd.Index(["MERCH"]), LOG) is None
    idx = pd.date_range("2030-01-03", periods=2, freq="h")
    physical = pd.DataFrame({"MERCH": [70.0, 100.0]}, index=idx)
    schedule = pd.DataFrame({"MERCH": [100.0, 90.0]}, index=idx)
    network = SimpleNamespace(snapshot_weightings=pd.DataFrame({"generators": [1.0, 1.0]}, index=idx))
    # Only the 30 MWh turned down carries the bid-side step.
    assert _offer_regulariser_cost(physical, schedule, step, network, "down") == pytest.approx(30.0)


def test_an_enabled_tie_break_must_declare_its_rule_and_step(tmp_path):
    path = tmp_path / "shares.csv"
    generator_state_shares(_register()).to_csv(path, index=False)
    with pytest.raises(ValueError, match="step_gbp_per_mwh"):
        resolve_tie_break({"enabled": True, "rule": "lifo_commissioning"},
                          {"shares_file": str(path), "case_id": "p3"}, pd.Series({"OLD": 1.0}), LOG)
