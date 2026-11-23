"""Stage 1 unit commitment with prices (16 Oct 2026, scripts/market/wholesale_commitment.py, variant s1_uc)."""
import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.scarcity_pricing import derated_capacity, margin_members, parse_margin_definition
from scripts.market.solve_wholesale import apply_wholesale_unit_commitment
from scripts.market.wholesale_commitment import (
    CommitmentOptimizer,
    _run_lengths,
    capture_bounds,
    commitment_method,
    configure_commitment,
    priced_commitment,
)

LOG = logging.getLogger("test_wholesale_commitment")
SOLVER = {"solver_name": "highs", "solver_options": {}}


def _uc(method="milp_fixed", **ccgt):
    params = {"enabled": True, "p_min_pu": 0.5, "min_up_time": 1, "min_down_time": 0,
              "ramp_limit_up": 1.0, "ramp_limit_down": 1.0, "start_up_cost_per_mw": 100.0,
              "shut_down_cost_per_mw": 0.0, **ccgt}
    return {"unit_commitment": {"enabled": True, "method": method, "min_p_nom_mw": 0.0, "initial_status": "off",
                                "exclude_carriers": [], "carrier_parameters": {"CCGT": params}}}


def _network(load=(40.0, 120.0, 40.0), ccgt_p_max_pu=None):
    """Base load plant (10), a CCGT (50, 100 MW) and a peaker (100) on one bus."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-04-01", periods=len(load), freq="h"))
    n.add("Bus", "GB")
    n.add("Load", "demand", bus="GB", p_set=pd.Series(load, index=n.snapshots))
    n.add("Generator", "base", bus="GB", carrier="nuclear", p_nom=50.0, marginal_cost=10.0)
    n.add("Generator", "ccgt", bus="GB", carrier="CCGT", p_nom=100.0, marginal_cost=50.0)
    n.add("Generator", "ocgt", bus="GB", carrier="OCGT", p_nom=100.0, marginal_cost=100.0)
    n.add("Generator", "shed", bus="GB", carrier="load_shedding", p_nom=1000.0, marginal_cost=6000.0)
    if ccgt_p_max_pu is not None:
        n.generators_t.p_max_pu["ccgt"] = pd.Series(ccgt_p_max_pu, index=n.snapshots)
    return n


def _plain(n, **kwargs):
    return n.optimize(**kwargs)


def _committed(n, wholesale):
    original = capture_bounds(n)
    assert apply_wholesale_unit_commitment(n, wholesale, LOG)
    plan = configure_commitment(n, wholesale, original, LOG)
    return plan, CommitmentOptimizer(plan, _plain, window_steps=len(n.snapshots), logger=LOG)


def _price(n):
    return n.buses_t.marginal_price["GB"].round(6).tolist()


def test_method_resolution_and_default_is_the_plain_milp():
    assert commitment_method({}) == "milp"
    assert not priced_commitment({"unit_commitment": {"enabled": True}})
    assert not priced_commitment(_uc() | {"unit_commitment": {**_uc()["unit_commitment"], "enabled": False}})
    assert priced_commitment(_uc())
    with pytest.raises(ValueError):
        commitment_method({"method": "rounded"})


def test_commitment_moves_the_price_setting_unit_from_ccgt_to_the_peaker():
    lp = _network()
    lp.optimize(**SOLVER)
    assert _price(lp) == pytest.approx([10.0, 50.0, 10.0])         # LP: the CCGT sets the peak hour
    n = _network()
    plan, opt = _committed(n, _uc())
    status, condition = opt(n, **SOLVER)
    assert status == "ok"
    # A start costs 100 x 100 MW = 10,000 > the 70 MWh x 50 the peaker costs more: the CCGT stays off,
    # so the peaker sets the peak hour.
    assert n.generators_t.p["ccgt"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert _price(n) == pytest.approx([10.0, 100.0, 10.0])
    assert opt.records[-1]["milp_status"] == "ok"
    # The plain LP would run the CCGT: the zero-commitment check reports it but keeps the commitment's answer.
    assert opt.records[-1]["zero_commitment_plain_gas"] and not opt.records[-1]["guard_fallback"]
    assert opt.summary()["zero_commitment_plain_gas_windows"] == 1


def test_minimum_stable_generation_lets_a_cheaper_unit_set_the_price():
    # The CCGT must start for the 140 MW hour; with a 3 h minimum up time it then stays on at its 50 MW
    # floor, so in the 60 MW hours the base plant (10) is marginal where the LP had the CCGT (50).
    load = (140.0, 60.0, 60.0)
    lp = _network(load)
    lp.optimize(**SOLVER)
    assert _price(lp) == pytest.approx([50.0, 50.0, 50.0])
    n = _network(load)
    _, opt = _committed(n, _uc(start_up_cost_per_mw=1.0, min_up_time=3))
    opt(n, **SOLVER)
    assert n.generators_t.p["ccgt"].tolist() == pytest.approx([90.0, 50.0, 50.0])
    assert _price(n) == pytest.approx([50.0, 10.0, 10.0])


def test_the_scarcity_margin_counts_available_not_committed_plant():
    seen = {}

    def probe(tag):
        def extra(net, snapshots):
            definition = parse_margin_definition(None)
            seen.setdefault(tag, []).append(derated_capacity(net, margin_members(net), definition, snapshots).tolist())
        return extra

    lp = _network(ccgt_p_max_pu=[1.0, 0.8, 1.0])
    lp.optimize(extra_functionality=probe("lp"), **SOLVER)
    n = _network(ccgt_p_max_pu=[1.0, 0.8, 1.0])
    _, opt = _committed(n, _uc())
    opt(n, extra_functionality=probe("uc"), **SOLVER)
    assert n.generators_t.p["ccgt"].max() == pytest.approx(0.0)      # the CCGT is off
    assert len(seen["uc"]) == 3        # MILP, the zero-commitment plain-LP check, the pricing LP (gas kept off)
    for margin in seen["uc"]:
        assert margin == pytest.approx(seen["lp"][0])                  # the off CCGT still counts
    assert n.generators_t.p_max_pu["ccgt"].tolist() == pytest.approx([1.0, 0.8, 1.0])


def test_floor_is_per_module_and_relative_to_availability_and_restored():
    n = _network(ccgt_p_max_pu=[1.0, 0.8, 1.0])
    wholesale = _uc(module_mw=40.0)
    plan, _ = _committed(n, wholesale)
    assert plan.units == ["ccgt"]
    assert plan.floor_pu["ccgt"] == pytest.approx(0.5 * 0.4)          # one 40 MW module at 50%
    assert n.generators_t.p_min_pu["ccgt"].tolist() == pytest.approx([0.2, 0.16, 0.2])
    assert n.generators.at["ccgt", "start_up_cost"] == pytest.approx(100.0 * 40.0)
    plan.restore(n, LOG)
    assert not bool(n.generators.at["ccgt", "committable"])
    assert n.generators.at["ccgt", "p_min_pu"] == pytest.approx(0.0)
    assert "ccgt" not in n.generators_t.p_min_pu.columns


def test_must_run_units_are_not_committed():
    n = _network()
    n.generators.loc["ccgt", "p_min_pu"] = 0.2
    original = capture_bounds(n)
    apply_wholesale_unit_commitment(n, _uc(), LOG)
    with pytest.raises(ValueError, match="no unit is committable"):
        configure_commitment(n, _uc(), original, LOG)


def test_commitment_is_carried_to_the_next_window_and_reused_on_a_resolve():
    full = _network((60.0, 140.0, 140.0, 140.0, 60.0, 60.0))
    plan, opt = _committed(full, _uc(start_up_cost_per_mw=1.0, min_up_time=3))
    opt.window_steps = 3
    first = full.copy()
    first.set_snapshots(full.snapshots[:3])
    second = full.copy()
    second.set_snapshots(full.snapshots[3:])
    opt(first, **SOLVER)
    assert first.generators_t.p["ccgt"].tolist() == pytest.approx([0.0, 90.0, 90.0])
    up, down = opt._carry
    assert (up["ccgt"], down["ccgt"]) == (2, 0)                        # on for the last two kept hours
    opt(second, **SOLVER)
    assert second.generators.at["ccgt", "up_time_before"] == 2
    assert second.generators.at["ccgt", "down_time_before"] == 0
    assert opt.records[-1]["milp_status"] == "ok"
    opt(second, **SOLVER)                                              # counterflow re-solve: same window
    assert opt.records[-1]["milp_status"] == "reused"
    assert second.generators.at["ccgt", "up_time_before"] == 2


def test_run_lengths():
    status = pd.DataFrame({"a": [1, 1, 0, 1, 1], "b": [1, 0, 0, 0, 0]}, dtype=float)
    up, down = _run_lengths(status)
    assert (up["a"], down["a"]) == (2, 0)
    assert (up["b"], down["b"]) == (0, 4)


def test_linearized_method_gives_prices_in_one_lp():
    n = _network()
    _, opt = _committed(n, _uc(method="linearized"))
    status, _ = opt(n, **SOLVER)
    assert status == "ok"
    assert np.isfinite(n.buses_t.marginal_price["GB"]).all()


# ── Regression (16 Oct 2026): the real rolling-window path, and the zero-commitment guard ─────────────
def _two_days(look_ahead=0):
    load = [40.0] * 24 + [140.0] * 24        # day 1: no gas needed; day 2: 90 MW of gas every hour
    n = _network(tuple(load))
    wholesale = _uc(start_up_cost_per_mw=1.0, min_up_time=3)
    wholesale.update({"mode": "rolling_day_ahead", "window_hours": 24, "carry_soc": True,
                      "look_ahead_hours": look_ahead, "look_ahead_resolution_hours": 6 if look_ahead else 1})
    return n, wholesale


def _rolling(n, wholesale, opt):
    from scripts.market.solve_wholesale import solve_rolling_day_ahead
    scenario = {"market": {"wholesale": wholesale}, "renewables": {"hydro": {}}}
    return solve_rolling_day_ahead(n, wholesale, "highs", {}, scenario, LOG, optimize=opt)


@pytest.mark.parametrize("look_ahead", [0, 24])
def test_rolling_path_commits_gas_when_it_is_needed(look_ahead):
    n, wholesale = _two_days(look_ahead)
    plan, opt = _committed(n, wholesale)
    opt.window_steps = 24
    gen, _, _, prices, _ = _rolling(n, wholesale, opt)
    last = {r["window_start"][:10]: r for r in opt.records}
    day1, day2 = last["2030-04-01"], last["2030-04-02"]
    assert day1["zero_commitment"] and not day1["guard_fallback"]           # no gas needed, none committed
    assert not day2["zero_commitment"] and day2["committed_available_mw_mean"] > 0
    assert day2["milp_gas_mwh"] == pytest.approx(24 * 90.0)
    assert gen["ccgt"].loc["2030-04-02"].sum() == pytest.approx(24 * 90.0)
    assert prices["GB"].loc["2030-04-02"].round(6).tolist() == pytest.approx([50.0] * 24)
    summary = opt.summary()
    assert summary["guard_fallback_windows"] == 0 and summary["zero_commitment_windows"] == 1


def test_zero_commitment_fallback_flag_takes_the_plain_lp():
    n = _network()
    wholesale = _uc()
    wholesale["unit_commitment"]["zero_commitment_fallback"] = True
    _, opt = _committed(n, wholesale)
    opt(n, **SOLVER)
    assert opt.records[-1]["guard_fallback"]
    assert _price(n) == pytest.approx([10.0, 50.0, 10.0])                 # the plain LP: the CCGT is back


def test_guard_falls_back_to_the_plain_lp_when_the_commitment_is_lost(monkeypatch):
    n, wholesale = _two_days()
    plan, opt = _committed(n, wholesale)
    opt.window_steps = 24
    real = CommitmentOptimizer._milp_pass

    def lost(self, net, kwargs):                                            # the status frame comes back all zero
        commitment, status, condition, seconds = real(self, net, kwargs)
        return commitment * 0.0, status, condition, seconds

    monkeypatch.setattr(CommitmentOptimizer, "_milp_pass", lost)
    gen, _, _, prices, _ = _rolling(n, wholesale, opt)
    last = {r["window_start"][:10]: r for r in opt.records}
    assert last["2030-04-02"]["guard_fallback"]
    assert gen["ccgt"].loc["2030-04-02"].sum() == pytest.approx(24 * 90.0)  # the plain LP runs the gas
    summary = opt.summary()
    assert summary["guard_fallback_windows"] == 1
    assert summary["guard_fallback_window_starts"][0].startswith("2030-04-02")


def test_status_on_an_unavailable_unit_does_not_count_as_committed():
    n = _network(ccgt_p_max_pu=[0.0, 0.0, 0.0])
    plan, opt = _committed(n, _uc())
    opt(n, **SOLVER)
    rec = opt.records[-1]
    assert rec["committed_available_mw_mean"] == pytest.approx(0.0)
    assert rec["zero_commitment"] and not rec["guard_fallback"]
