"""Tie-break cost on |flow| of GB-internal DC links, both stages (user decision, 16 Oct 2026).

GB-internal HVDC links are lossless and free both ways, so beside the AC network
their flow is a degenerate tie (EGL1 ran at full south-to-north for 915 h a year
on the cal_rb1 BM). ``market.internal_dc_tie_break_gbp_per_mwh`` (0.75) prices
|p| through p = fwd - rev in the linopy model. These tests pin: the degeneracy and
its removal on a two-path network, reverse flow costing the same as forward, the
scope (no interconnector, no hydrogen link), and the penalty staying out of the
reported Stage 1 objective and BM cost.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.market.internal_dc_tie_break import (
    CONSTRAINT_NAME,
    DEFAULT_TIE_BREAK_GBP_PER_MWH,
    META_KEY,
    build_internal_dc_tie_break_callback,
    internal_dc_link_names,
    internal_dc_tie_break_cost,
    resolve_internal_dc_tie_break,
    tie_break_summary,
)
from scripts.market.solve_balancing import solve_rolling_balancing
from scripts.market.solve_wholesale import solve_rolling_day_ahead
from scripts.solve.hydro_constraints import combine_extra_functionalities
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_internal_dc_tie_break")
COST = 0.75
HOURS = 3
CHEAP_MC, DEAR_MC = 10.0, 100.0


def _two_path_network(line_mw=1000.0, dc_mw=500.0, gen_at="A", load_at="B", load_mw=300.0):
    """Buses A and B joined by an AC line and a two-way DC link; cheap unit at one end, load at the other."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=HOURS, freq="h"))
    for carrier in ("AC", "DC", "wind_onshore", "CCGT"):
        n.add("Carrier", carrier)
    n.add("Bus", "A", v_nom=400.0, carrier="AC", country="GB")
    n.add("Bus", "B", v_nom=400.0, carrier="AC", country="GB")
    n.add("Line", "A_B", bus0="A", bus1="B", r=0.0, x=0.1, s_nom=line_mw, carrier="AC")
    n.add("Link", "A_B_HVDC", bus0="A", bus1="B", p_nom=dc_mw, p_min_pu=-1.0, p_max_pu=1.0,
          efficiency=1.0, marginal_cost=0.0, carrier="DC")
    other = "B" if gen_at == "A" else "A"
    n.add("Generator", "cheap", bus=gen_at, p_nom=1000.0, marginal_cost=CHEAP_MC, carrier="wind_onshore")
    n.add("Generator", "dear", bus=other, p_nom=1000.0, marginal_cost=DEAR_MC, carrier="CCGT")
    n.add("Load", "load", bus=load_at, p_set=load_mw)
    return n


def _force_dc(value):
    """extra_functionality holding the DC link at ``value`` MW in every hour."""
    def force(n, sns):
        link_p = n.model.variables["Link-p"]
        dim = [d for d in link_p.dims if d != "snapshot"][0]
        n.model.add_constraints(link_p.sel({dim: ["A_B_HVDC"]}) == float(value), name="Link-test_force_dc")
    return force


def _solve(n, cost, *extra):
    callback = build_internal_dc_tie_break_callback(n, cost, LOG)
    status, condition = n.optimize(
        solver_name="highs", extra_functionality=combine_extra_functionalities(*extra, callback))
    assert status == "ok", condition
    return callback


@pytest.mark.unit
class TestTwoPathDegeneracy:

    def test_without_tie_break_dc_flow_is_free(self):
        # Any DC flow, forward or reverse, gives the same objective: the tie.
        objectives = []
        for value in (0.0, 300.0, -400.0, 500.0):
            n = _two_path_network()
            assert _solve(n, 0.0, _force_dc(value)) is None
            objectives.append(float(n.objective))
        assert np.allclose(objectives, CHEAP_MC * 300.0 * HOURS)

    def test_with_tie_break_dc_flow_is_zero_when_not_needed(self):
        n = _two_path_network()
        callback = _solve(n, COST)
        assert callback is not None and callback.links == ("A_B_HVDC",)
        assert np.allclose(n.links_t.p0["A_B_HVDC"], 0.0, atol=1e-6)
        assert np.allclose(n.lines_t.p0["A_B"], 300.0, atol=1e-6)
        gbp, mwh = internal_dc_tie_break_cost(n, callback)
        assert gbp == pytest.approx(0.0, abs=1e-6) and mwh == pytest.approx(0.0, abs=1e-6)

    def test_with_tie_break_dc_carries_only_what_is_needed(self):
        # The line takes 200 MW; the link carries the other 100 and no more.
        n = _two_path_network(line_mw=200.0)
        callback = _solve(n, COST)
        assert np.allclose(n.links_t.p0["A_B_HVDC"], 100.0, atol=1e-6)
        gbp, mwh = internal_dc_tie_break_cost(n, callback)
        assert mwh == pytest.approx(100.0 * HOURS, rel=1e-9, abs=1e-6)
        assert gbp == pytest.approx(COST * 100.0 * HOURS, rel=1e-9, abs=1e-6)
        assert float(n.objective) == pytest.approx(CHEAP_MC * 300.0 * HOURS + gbp, rel=1e-9)


@pytest.mark.unit
class TestReverseFlowCostsTheSame:

    @pytest.mark.parametrize("value", [250.0, -250.0])
    def test_forced_flow_costs_cost_times_abs(self, value):
        n = _two_path_network()
        callback = _solve(n, COST, _force_dc(value))
        gbp, mwh = internal_dc_tie_break_cost(n, callback)
        assert mwh == pytest.approx(abs(value) * HOURS, rel=1e-9)
        assert gbp == pytest.approx(COST * abs(value) * HOURS, rel=1e-9)
        assert float(n.objective) == pytest.approx(CHEAP_MC * 300.0 * HOURS + gbp, rel=1e-9)

    def test_needed_reverse_flow_costs_the_same_as_forward(self):
        forward = _two_path_network(line_mw=200.0, gen_at="A", load_at="B")
        reverse = _two_path_network(line_mw=200.0, gen_at="B", load_at="A")
        cb_f, cb_r = _solve(forward, COST), _solve(reverse, COST)
        assert np.allclose(forward.links_t.p0["A_B_HVDC"], 100.0, atol=1e-6)
        assert np.allclose(reverse.links_t.p0["A_B_HVDC"], -100.0, atol=1e-6)
        assert internal_dc_tie_break_cost(forward, cb_f)[0] == pytest.approx(
            internal_dc_tie_break_cost(reverse, cb_r)[0], rel=1e-9)
        assert float(forward.objective) == pytest.approx(float(reverse.objective), rel=1e-9)

    def test_snapshot_weightings_scale_the_term(self):
        n = _two_path_network(line_mw=200.0)
        n.snapshot_weightings.loc[:, :] = 2.0
        callback = _solve(n, COST)
        gbp, mwh = internal_dc_tie_break_cost(n, callback)
        assert mwh == pytest.approx(2.0 * 100.0 * HOURS, rel=1e-9)
        kept = n.snapshots[:1]
        assert internal_dc_tie_break_cost(n, callback, snapshots=kept)[1] == pytest.approx(200.0, rel=1e-9)


def _scoped_network():
    """The two-path network plus an interconnector, a DC link to a foreign bus and hydrogen links."""
    n = _two_path_network()
    for carrier in ("H2_gas", "electrolysis", "H2_turbine"):
        n.add("Carrier", carrier)
    n.add("Bus", "HVDC_External_France_Calais", v_nom=400.0, carrier="AC", country="France")
    n.add("Bus", "NI_converter", v_nom=400.0, carrier="AC", country="Northern Ireland")
    n.add("Bus", "GB_H2", carrier="H2_gas", country="")
    n.add("Generator", "france", bus="HVDC_External_France_Calais", p_nom=100.0, marginal_cost=50.0,
          carrier="CCGT")
    n.add("Load", "ni_load", bus="NI_converter", p_set=0.0)
    n.add("Link", "IC_IFA", bus0="HVDC_External_France_Calais", bus1="A", p_nom=100.0, p_min_pu=-1.0,
          marginal_cost=0.0, carrier="DC")
    n.add("Link", "IC_Moyle", bus0="NI_converter", bus1="B", p_nom=100.0, p_min_pu=-1.0, carrier="DC")
    n.add("Link", "electrolysis_B", bus0="B", bus1="GB_H2", p_nom=50.0, efficiency=0.7, carrier="electrolysis")
    n.add("Link", "H2_turbine_B", bus0="GB_H2", bus1="B", p_nom=50.0, efficiency=0.5, carrier="H2_turbine")
    n.add("Link", "dc_to_h2_bus", bus0="A", bus1="GB_H2", p_nom=50.0, carrier="DC")
    n.add("Link", "disabled_HVDC", bus0="A", bus1="B", p_nom=0.0, p_min_pu=-1.0, carrier="DC")
    n.add("Store", "GB_H2_store", bus="GB_H2", e_nom=1000.0, e_cyclic=True, carrier="H2_gas")
    return n


@pytest.mark.unit
class TestScope:

    def test_only_internal_dc_links_are_selected(self):
        assert internal_dc_link_names(_scoped_network()) == ["A_B_HVDC"]

    def test_model_term_covers_only_internal_links(self):
        n = _scoped_network()
        callback = _solve(n, COST)
        con = n.model.constraints[CONSTRAINT_NAME]
        dim = [d for d in con.dims if d != "snapshot"][0]
        assert list(map(str, con.coords[dim].values)) == ["A_B_HVDC"]
        # The interconnector, foreign DC and hydrogen links keep PyPSA's own pricing and bounds.
        for name in ("IC_IFA", "IC_Moyle", "electrolysis_B", "H2_turbine_B", "dc_to_h2_bus"):
            assert name not in callback.links
        assert n.links.loc["IC_IFA", "marginal_cost"] == 0.0

    def test_real_network_links_shape(self):
        # The CP30 shape: the nine internal links and none of the twelve interconnectors.
        n = pypsa.Network()
        n.add("Carrier", "AC")
        for bus in ("BRNX4A", "HAWP41", "Central DC Substation", "KERG1J"):
            n.add("Bus", bus, carrier="AC", country="GB")
        n.add("Bus", "HVDC_External_Denmark_Revsing", carrier="AC", country="Denmark")
        n.add("Link", "BRNX4A_HAWP41_HVDC_2029", bus0="BRNX4A", bus1="HAWP41", p_nom=2000.0, p_min_pu=-1.0,
              carrier="DC")
        n.add("Link", "KERG1J_Central DC Substation_0", bus0="KERG1J", bus1="Central DC Substation",
              p_nom=600.0, p_min_pu=-1.0, carrier="DC")
        n.add("Link", "IC_Viking", bus0="HVDC_External_Denmark_Revsing", bus1="HAWP41", p_nom=1400.0,
              p_min_pu=-1.0, carrier="DC")
        assert internal_dc_link_names(n) == ["BRNX4A_HAWP41_HVDC_2029", "KERG1J_Central DC Substation_0"]


@pytest.mark.unit
class TestConfig:

    def test_code_default_matches_defaults_yaml(self):
        defaults = load_defaults()
        assert defaults["market"]["internal_dc_tie_break_gbp_per_mwh"] == DEFAULT_TIE_BREAK_GBP_PER_MWH == 0.75
        assert resolve_internal_dc_tie_break({}) == 0.75

    def test_off_and_validation(self):
        assert resolve_internal_dc_tie_break({"internal_dc_tie_break_gbp_per_mwh": None}) == 0.0
        assert build_internal_dc_tie_break_callback(_two_path_network(), 0.0, LOG) is None
        with pytest.raises(ValueError):
            resolve_internal_dc_tie_break({"internal_dc_tie_break_gbp_per_mwh": -1.0})


@pytest.mark.unit
class TestStage1ReportedObjective:

    def test_rolling_day_ahead_strips_the_penalty(self):
        n = _two_path_network(line_mw=200.0)
        gen, su, links, prices, objective = solve_rolling_day_ahead(
            n, {"window_hours": 2, "look_ahead_hours": 0}, "highs", {}, {"market": {"internal_dc_tie_break_gbp_per_mwh": COST}}, LOG,
        )
        assert np.allclose(links["A_B_HVDC"], 100.0, atol=1e-6)
        assert objective == pytest.approx(CHEAP_MC * 300.0 * HOURS, rel=1e-9)
        record = n.meta[META_KEY]["wholesale"]
        assert record["cost_GBP"] == pytest.approx(COST * 100.0 * HOURS, rel=1e-6)
        assert record["abs_flow_MWh"] == pytest.approx(100.0 * HOURS, rel=1e-6)
        assert record["flow_by_link_MWh"]["A_B_HVDC"] == pytest.approx(100.0 * HOURS, rel=1e-6)

    def test_look_ahead_reports_only_the_kept_hours(self):
        n = _two_path_network(line_mw=200.0)
        wholesale = {"window_hours": 1, "look_ahead_hours": 1}
        _, _, links, _, _ = solve_rolling_day_ahead(
            n, wholesale, "highs", {}, {"market": {"internal_dc_tie_break_gbp_per_mwh": COST}}, LOG,
        )
        assert np.allclose(links["A_B_HVDC"], 100.0, atol=1e-6)
        assert n.meta[META_KEY]["wholesale"]["abs_flow_MWh"] == pytest.approx(100.0 * HOURS, rel=1e-6)


def _bm_inputs(n):
    snaps = n.snapshots
    gen = pd.DataFrame({"cheap": 300.0, "dear": 0.0}, index=snaps)
    su = pd.DataFrame(index=snaps)
    links = pd.DataFrame({"A_B_HVDC": 0.0}, index=snaps)
    offer = pd.Series({"cheap": 12.0, "dear": 110.0})
    bid = pd.Series({"cheap": 9.0, "dear": 90.0})
    return gen, su, links, offer, bid


def _solve_bm(n, cost):
    gen, su, links, offer, bid = _bm_inputs(n)
    return solve_rolling_balancing(
        n, gen, su, links, offer, bid, pd.Series(dtype=float), pd.Series(dtype=float),
        {"window_hours": 1, "fix_interconnectors": True}, "highs", {},
        {"market": {"internal_dc_tie_break_gbp_per_mwh": cost}}, LOG,
    )


@pytest.mark.unit
class TestBalancingCostExcludesPenalty:

    def test_no_redispatch_needed_costs_nothing(self):
        # Line 200 + link 500: the copperplate schedule is feasible once the link carries 100.
        n = _two_path_network(line_mw=200.0)
        physical_gen, _, _, bm_cost = _solve_bm(n, COST)
        assert bm_cost == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(n.links_t.p0["A_B_HVDC"], 100.0, atol=1e-6)
        record = n.meta[META_KEY]["balancing"]
        assert record["cost_GBP"] == pytest.approx(COST * 100.0 * HOURS, rel=1e-6)

    def test_redispatch_cost_is_bids_and_offers_only(self):
        # Line 200 + link 50: 50 MW must be redispatched each hour (cheap down at its bid, dear up at its offer).
        n = _two_path_network(line_mw=200.0, dc_mw=50.0)
        physical_gen, _, _, bm_cost = _solve_bm(n, COST)
        assert np.allclose(physical_gen["dear"], 50.0, atol=1e-6)
        # Bids are on the ESO-cost convention (positive: the ESO pays to turn the unit down).
        assert bm_cost == pytest.approx((110.0 + 9.0) * 50.0 * HOURS, rel=1e-9)
        assert n.meta[META_KEY]["balancing"]["cost_GBP"] == pytest.approx(COST * 50.0 * HOURS, rel=1e-6)
        # The same dispatch and cost with the tie-break off: it is only a tie-break.
        m = _two_path_network(line_mw=200.0, dc_mw=50.0)
        physical_off, _, _, bm_cost_off = _solve_bm(m, 0.0)
        assert bm_cost_off == pytest.approx(bm_cost, rel=1e-9)
        assert META_KEY not in dict(m.meta or {})

    def test_market_summary_reports_it_beside_the_bm_cost(self):
        from scripts.market.analyze_market import create_market_summary

        wholesale = _two_path_network(line_mw=200.0)
        solve_rolling_day_ahead(wholesale, {"window_hours": 3}, "highs", {},
                                {"market": {"internal_dc_tie_break_gbp_per_mwh": COST}}, LOG)
        bm = _two_path_network(line_mw=200.0, dc_mw=50.0)
        _, _, _, bm_cost = _solve_bm(bm, COST)
        bm._objective = bm_cost
        empty = pd.DataFrame()
        summary = create_market_summary(wholesale, bm, empty, empty, empty, empty, "test", LOG)
        assert summary["balancing"]["total_redispatch_cost"] == pytest.approx((110.0 + 9.0) * 50.0 * HOURS)
        assert summary["balancing"]["internal_dc_tie_break_cost_GBP"] == pytest.approx(COST * 50.0 * HOURS, rel=1e-6)
        assert summary["wholesale"]["internal_dc_tie_break_cost_GBP"] == pytest.approx(COST * 100.0 * HOURS, rel=1e-6)
        assert set(summary["internal_dc_tie_break"]) == {"wholesale", "balancing"}

    def test_annual_aggregate_sums_it(self, tmp_path):
        import json

        from scripts.market.aggregate_neso_comparator import aggregate_monthly_comparator_summaries

        fields = {"reported_period_hours": 730.0, "demand_MWh": 1.0e6, "accepted_offer_cost_GBP": 1.0,
                  "accepted_bid_cost_GBP": 1.0, "modelled_redispatch_cost_GBP": 2.0,
                  "ordinary_redispatch_cost_excluding_feasibility_slack_GBP": 2.0,
                  "feasibility_slack_energy_MWh": 0.0, "feasibility_slack_cost_GBP": 0.0,
                  "gross_redispatch_MWh": 10.0}
        paths, names = [], []
        for i in range(12):
            name = f"month{i}"
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps({
                "scenario": name, "accounting": dict(fields, reported_period_hours=8760.0 / 12),
                "balancing": {"internal_dc_tie_break_cost_GBP": 10.0},
                "wholesale": {"internal_dc_tie_break_cost_GBP": 1.0}}), encoding="utf-8")
            paths.append(path)
            names.append(name)
        agg = aggregate_monthly_comparator_summaries(paths, names, tmp_path / "m.csv", tmp_path / "a.json")
        agg = agg.get("aggregate", agg)
        assert agg["internal_dc_tie_break_cost_GBP"] == pytest.approx(120.0)
        assert agg["stage1_internal_dc_tie_break_cost_GBP"] == pytest.approx(12.0)
        assert agg["modelled_redispatch_cost_GBP"] == pytest.approx(24.0)

    def test_summary_block_survives_a_netcdf_round_trip(self, tmp_path):
        n = _two_path_network(line_mw=200.0)
        _solve_bm(n, COST)
        path = tmp_path / "bm.nc"
        n.export_to_netcdf(path)
        loaded = pypsa.Network(path)
        block = tie_break_summary(None, loaded)
        assert block["balancing"]["cost_GBP"] == pytest.approx(COST * 100.0 * HOURS, rel=1e-6)
        assert "wholesale" not in block
