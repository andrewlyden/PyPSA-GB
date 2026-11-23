"""Interconnector counter-trading in the balancing stage (27 Sep 2026).

The balancing stage may move a cross-border cable off its Stage 1 schedule, but
only as NESO can: at an FTI counter-trade premium on top of the trade-value
difference, within LCP's flow limit, and only when the Stage 1 spread clears
LCP's minimum. These pin the LP behaviour on a two-bus toy, the band's
feasibility under an outage, and that every re-trading report carries the
premium the solve paid.

Toy: wind at N (300 MW, bid 50) behind a 100 MW line to S (load 100 MW, gas
offer 77.4), and a 200 MW cable at S to France (ask 50.5, bid 49.5, losses 10%).
Stage 1 ran wind at 300 and exported 200. The line forces 200 MW of wind off;
the BM then either turns gas up 180 (the export lands at 90%: 13,932) or cuts the export 200
(8,910 of lost export revenue, plus 20,000 of premium at 100/MWh).
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa
import pytest
import yaml

from scripts.interconnectors.add_to_network import add_interconnector_links
from scripts.interconnectors.market_coupling import (
    COUNTER_TRADE_PREMIUM_COLUMN,
    COUNTER_TRADE_BID_PREMIUM_COLUMN,
    COUNTER_TRADE_BID_RULE_COLUMN,
    COUNTER_TRADE_OFFER_RULE_COLUMN,
    build_piecewise_interconnector_cost_callback,
    counter_trade_premium_cost,
    marginal_trade_cost,
)
from scripts.market.analyze_market import calculate_interconnector_retrading_metrics
from scripts.market.fti_settlement import interconnector_retrading_cost
from scripts.market.interconnector_counter_trade import (
    COUNTER_TRADE_SHARE_COLUMN,
    DEFAULT_COUNTER_TRADE,
    build_counter_trade_callback,
    counter_trade_bounds,
    counter_trade_links,
    link_shares,
    resolve_counter_trade_config,
    stage1_spread,
    stamp_counter_trade_attributes,
    summarise_counter_trade,
)
from scripts.market.solve_balancing import (
    _build_balancing_extra_functionality,
    prepare_interconnector_counter_trade,
)
from scripts.solve.hydro_constraints import combine_extra_functionalities
from config.config_loader import load_defaults, load_layer

LOGGER = logging.getLogger("test_counter_trade")
ROOT = Path(__file__).resolve().parents[2]
LINK = "IC_TestCable"
WIND_BID, GAS_OFFER = 50.0, 77.4


def _network(hours: int = 1, country: str = "France"):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "N")
    n.add("Bus", "S")
    n.add("Bus", "External_France")
    n.add("Line", "N_S", bus0="N", bus1="S", x=0.1, r=0.01, s_nom=100.0)
    n.add("Generator", "wind", bus="N", p_nom=300.0, marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "gas", bus="S", p_nom=300.0, marginal_cost=60.0, carrier="CCGT")
    n.add("Load", "load", bus="S", p_set=100.0)
    cables = pd.DataFrame([{
        "name": "TestCable", "from_bus": "S", "to_bus": "External_France",
        "capacity_mw": 200.0, "losses_percent": 10.0, "counterparty_country": country,
    }])
    prices = pd.DataFrame([{"country": "france", "year": 2030, "pathway": "Central",
                            "estimated_price_gbp_per_mwh": 50.0}])
    add_interconnector_links(n, cables, pd.DataFrame(), prices, modelled_year=2030,
                             fes_pathway="Central", pricing_config={"transaction_spread_gbp_per_mwh": 1.0})
    return n


def _schedule(n):
    gen = pd.DataFrame({"wind": 300.0, "gas": 0.0}, index=n.snapshots)
    links = pd.DataFrame({LINK: -200.0}, index=n.snapshots)
    return gen, links


def _balancing(**counter_trade):
    return {"interconnector_retrading": True,
            "interconnector_counter_trade": {"enabled": True, **counter_trade}}


def _solve(n, balancing, gb_price=0.0):
    gen, links = _schedule(n)
    prepare_interconnector_counter_trade(n, balancing, links, LOGGER)
    extra = _build_balancing_extra_functionality(
        wholesale_gen=gen, wholesale_su=pd.DataFrame(index=n.snapshots), wholesale_links=links,
        gen_offer_prices=pd.Series({"wind": 0.0, "gas": GAS_OFFER}),
        gen_bid_prices=pd.Series({"wind": WIND_BID, "gas": -54.0}),
        su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
        fix_interconnectors=False, logger=LOGGER,
    )
    price = None if gb_price is None else pd.Series(gb_price, index=n.snapshots)
    counter = build_counter_trade_callback(n, links, price, balancing, LOGGER)
    piecewise = build_piecewise_interconnector_cost_callback(
        n, LOGGER, baseline_dispatch=links, subtract_native_link_cost=False)
    status, condition = n.optimize(
        solver_name="highs",
        extra_functionality=combine_extra_functionalities(extra, counter, piecewise),
    )
    assert status == "ok", condition
    return n, links, counter


@pytest.mark.unit
class TestCounterTradeLP:

    def test_without_counter_trade_pricing_the_bm_unwinds_the_export(self):
        n, _, counter = _solve(_network(), {"interconnector_retrading": True,
                                            "interconnector_counter_trade": {"enabled": False}})
        assert counter is None
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        assert n.generators_t.p.at[n.snapshots[0], "gas"] == pytest.approx(0.0, abs=1e-3)

    def test_the_premium_makes_gas_cheaper_than_unwinding(self):
        n, _, _ = _solve(_network(), _balancing(premium_gbp_per_mwh=100.0, max_deviation_share=None,
                                                max_deviation_share_by_country={}))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(-200.0, abs=1e-3)
        assert n.generators_t.p.at[n.snapshots[0], "gas"] == pytest.approx(180.0, abs=1e-3)
        # Objective = wind bid + gas offer + export revenue kept.
        assert n.objective == pytest.approx(200 * WIND_BID + 180 * GAS_OFFER - 200 * 0.9 ** 2 * 49.5, rel=1e-5)  # export earns eta^2 x bid (item D2)

    def test_flow_limit_caps_the_deviation_at_the_share_of_capacity(self):
        n, _, _ = _solve(_network(), _balancing(premium_gbp_per_mwh=0.0,
                                                max_deviation_share_by_country={"France": 0.25}))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(-150.0, abs=1e-3)
        assert n.generators_t.p.at[n.snapshots[0], "gas"] == pytest.approx(135.0, abs=1e-3)

    def test_spread_gate_still_allows_the_network_to_reduce_the_flow(self):
        # GB 45 against the 49.5 export bid: spread 4.5 < 21, so the hour is gated.
        # The gate stops counter-trades that INCREASE |flow| (27 Sep 2026); cutting
        # the export the network cannot carry is a reduction and stays allowed.
        n, _, counter = _solve(_network(), _balancing(premium_gbp_per_mwh=0.0,
                                                      max_deviation_share_by_country={"France": 2.0}),
                               gb_price=45.0)
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        audit = next(iter(counter.bounds.values()))
        assert bool(audit["gated"].iloc[0][LINK])
        # Up (towards import) only as far as zero; down (more export) shut.
        assert audit["up"].iloc[0][LINK] == pytest.approx(200.0)
        assert audit["down"].iloc[0][LINK] == pytest.approx(0.0)

    def test_spread_gate_blocks_a_move_that_increases_the_flow(self):
        # The band in gated hours: no move away from zero, reductions up to the
        # Stage 1 flow and never beyond the flow limit.
        n = _network(hours=3)
        ws = pd.DataFrame({LINK: [-50.0, 60.0, 0.0]}, index=n.snapshots)
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {
            "enabled": True, "max_deviation_share_by_country": {"France": 0.45}}})
        # GB price at the foreign price every hour: spread < 21, all three gated.
        up, down, audit = counter_trade_bounds(n, [LINK], ws, pd.Series(50.0, index=n.snapshots), cfg)
        assert audit["gated"][LINK].tolist() == [True, True, True]
        # Exporting 50: towards zero (up) up to 50, more export (down) shut.
        # Importing 60: towards zero (down) up to 60, more import (up) shut.
        # Idle: nothing either way. The 0.45 x 200 = 90 MW flow limit still caps.
        assert up[LINK].tolist() == pytest.approx([50.0, 0.0, 0.0])
        assert down[LINK].tolist() == pytest.approx([0.0, 60.0, 0.0])
        ws_big = pd.DataFrame({LINK: [-150.0, 150.0, 0.0]}, index=n.snapshots)
        up, down, _ = counter_trade_bounds(n, [LINK], ws_big, pd.Series(50.0, index=n.snapshots), cfg)
        assert up[LINK].tolist() == pytest.approx([90.0, 0.0, 0.0])
        assert down[LINK].tolist() == pytest.approx([0.0, 90.0, 0.0])

    def test_spread_gate_open_when_stage1_spread_is_wide(self):
        n, _, counter = _solve(_network(), _balancing(premium_gbp_per_mwh=0.0,
                                                      max_deviation_share_by_country={"France": 2.0}),
                               gb_price=0.0)
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        assert not bool(next(iter(counter.bounds.values()))["gated"].iloc[0][LINK])

    def test_outage_forces_movement_beyond_a_zero_limit_and_pays_the_premium(self):
        n = _network()
        # The BM sees the cable out (both directions); Stage 1 exported 200.
        n.links_t.p_min_pu = pd.DataFrame({LINK: 0.0}, index=n.snapshots)
        n.links_t.p_max_pu = pd.DataFrame({LINK: 0.0}, index=n.snapshots)
        n, links, counter = _solve(n, _balancing(premium_gbp_per_mwh=100.0,
                                                 max_deviation_share_by_country={"France": 0.0}))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        assert bool(next(iter(counter.bounds.values()))["forced"].iloc[0][LINK])
        assert n.objective == pytest.approx(200 * WIND_BID + 200 * 100.0, rel=1e-5)

    def _stranded_export(self):
        # No gas and no load at S: only the 100 MW line feeds the landing bus, so
        # the 200 MW Stage 1 export cannot be carried and the cable must move
        # 200 - 100/0.9 = 88.9 MW, beyond a zero flow limit. Nothing else can
        # stand in for the export (the 27 Sep 2026 N-1 + outage infeasibility).
        n = _network()
        n.generators.loc["gas", "p_nom"] = 0.0
        n.loads_t.p_set = pd.DataFrame({"load": 0.0}, index=n.snapshots)
        n.loads.loc["load", "p_set"] = 0.0
        return n

    def test_hard_band_is_infeasible_when_the_network_cannot_carry_the_schedule(self):
        n = self._stranded_export()
        with pytest.raises(AssertionError):
            _solve(n, _balancing(premium_gbp_per_mwh=100.0, beyond_limit_penalty_gbp_per_mwh=None,
                                 max_deviation_share_by_country={"France": 0.0}))

    def test_soft_band_moves_the_cable_at_the_penalty_and_strips_it(self):
        from scripts.market.interconnector_counter_trade import counter_trade_excess_penalty

        n = self._stranded_export()
        n, links, counter = _solve(n, _balancing(premium_gbp_per_mwh=100.0,
                                                 max_deviation_share_by_country={"France": 0.0}))
        moved = n.links_t.p0.at[n.snapshots[0], LINK] - (-200.0)
        assert moved == pytest.approx(200.0 - 100.0 / 0.9, abs=1e-3)
        penalty, mwh = counter_trade_excess_penalty(n, counter)
        assert mwh == pytest.approx(moved, abs=1e-3)
        rate = DEFAULT_COUNTER_TRADE["beyond_limit_penalty_gbp_per_mwh"]
        assert penalty == pytest.approx(rate * moved, rel=1e-6)
        # Objective = wind bid on 200 MW + premium and penalty on the move - export revenue kept.
        kept = 0.9 ** 2 * 49.5 * (200.0 - moved)   # eta^2 x bid (item D2)
        assert n.objective == pytest.approx(200 * WIND_BID + (100.0 + rate) * moved - kept, rel=1e-5)
        summary = summarise_counter_trade(n.links_t.p0, links, n.links, counter.bounds).set_index("link")
        assert summary.at[LINK, "beyond_limit_MWh"] == pytest.approx(moved, abs=1e-3)

    def test_soft_band_penalty_is_not_used_when_the_band_suffices(self):
        from scripts.market.interconnector_counter_trade import counter_trade_excess_penalty

        n, _, counter = _solve(_network(), _balancing(premium_gbp_per_mwh=0.0,
                                                      max_deviation_share_by_country={"France": 0.25}))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(-150.0, abs=1e-3)
        assert counter_trade_excess_penalty(n, counter)[1] == pytest.approx(0.0, abs=1e-6)

    def test_fixed_interconnectors_are_not_counter_traded(self):
        n = _network()
        _, links = _schedule(n)
        names = prepare_interconnector_counter_trade(
            n, {"fix_interconnectors": True, "interconnector_counter_trade": {"enabled": True}}, links, LOGGER)
        assert names == []
        assert COUNTER_TRADE_PREMIUM_COLUMN not in n.links.columns


@pytest.mark.unit
class TestCounterTradeBand:

    def test_band_contains_the_stage1_position_and_the_forced_move(self):
        n = _network(hours=2)
        n.links_t.p_max_pu = pd.DataFrame({LINK: [1.0, 0.5]}, index=n.snapshots)
        n.links_t.p_min_pu = pd.DataFrame({LINK: [-1.0, -0.5]}, index=n.snapshots)
        ws = pd.DataFrame({LINK: [-200.0, -200.0]}, index=n.snapshots)
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {"enabled": True}})
        up, down, audit = counter_trade_bounds(n, [LINK], ws, None, cfg)
        # France 0.45 x 200 = 90 MW each way; hour 1's outage forces +100 MW.
        assert up[LINK].tolist() == pytest.approx([90.0, 100.0])
        assert down[LINK].tolist() == pytest.approx([90.0, 90.0])
        assert audit["forced"][LINK].tolist() == [False, True]

    def test_stage1_spread_uses_the_direction_the_cable_flows(self):
        n = _network(hours=3)
        ws = pd.DataFrame({LINK: [100.0, -100.0, 0.0]}, index=n.snapshots)
        spread = stage1_spread(n, [LINK], ws, pd.Series([80.0, 80.0, 50.2], index=n.snapshots))
        # import -> ask 50.5; export -> bid 49.5; idle -> nearer of the two.
        assert spread[LINK].tolist() == pytest.approx([29.5, 30.5, 0.3])

    def test_country_shares_and_default(self):
        n = _network(country="Norway")
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {"enabled": True}})
        assert link_shares(n, [LINK], cfg)[LINK] == 0.0
        n = _network(country="Atlantis")
        assert link_shares(n, [LINK], cfg)[LINK] == pytest.approx(0.45)

    def test_cross_border_links_are_found_by_pricing_or_external_bus(self):
        n = _network()
        n.add("Bus", "H2")
        n.add("Link", "electrolysis", bus0="S", bus1="H2", p_nom=10.0)
        n.add("Link", "IC_directional", bus0="HVDC_External_X", bus1="S", p_nom=10.0)
        assert counter_trade_links(n) == [LINK, "IC_directional"]

    def test_invalid_config_is_refused(self):
        with pytest.raises(ValueError):
            resolve_counter_trade_config({"interconnector_counter_trade": {"min_spread_rule": "vibes"}})
        with pytest.raises(ValueError):
            resolve_counter_trade_config({"interconnector_counter_trade": {"max_deviation_share": 3.0}})

    def test_premium_floor_rule_lifts_a_small_premium_to_the_spread(self):
        n = _network()
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {
            "enabled": True, "premium_gbp_per_mwh": 5.0, "min_spread_rule": "premium_floor"}})
        stamp_counter_trade_attributes(n, cfg)
        assert n.links.at[LINK, COUNTER_TRADE_PREMIUM_COLUMN] == pytest.approx(21.0)


@pytest.mark.unit
class TestCounterTradeReporting:

    def _solved(self):
        n = _network()
        n.links_t.p_min_pu = pd.DataFrame({LINK: 0.0}, index=n.snapshots)
        n.links_t.p_max_pu = pd.DataFrame({LINK: 0.0}, index=n.snapshots)
        return _solve(n, _balancing(premium_gbp_per_mwh=100.0,
                                    max_deviation_share_by_country={"France": 0.0}))

    def test_premium_is_in_the_retrading_cost_both_reports_use(self):
        n, links, _ = self._solved()
        physical = n.links_t.p0[[LINK]]
        premium = counter_trade_premium_cost(physical, links, n.links)
        assert float(premium.to_numpy().sum()) == pytest.approx(20_000.0, rel=1e-6)
        # analyze_market: lost export revenue 200 x 0.9^2 x 49.5 = 8,019 plus the premium (item D2).
        metrics = calculate_interconnector_retrading_metrics(n, links, physical).set_index("interconnector")
        assert metrics.at["TestCable", "counter_trade_premium_GBP"] == pytest.approx(20_000.0, rel=1e-6)
        assert metrics.at["TestCable", "retrading_cost_GBP"] == pytest.approx(8_019.0 + 20_000.0, rel=1e-6)
        # fti_settlement values re-trading identically.
        assert interconnector_retrading_cost(physical, links, n.links, n.links_t.marginal_cost) == pytest.approx(
            metrics.at["TestCable", "retrading_cost_GBP"], rel=1e-9)

    def test_a_network_solved_without_a_premium_values_it_at_zero(self):
        n = _network()
        _, links = _schedule(n)
        physical = links * 0.0
        assert counter_trade_premium_cost(physical, links, n.links).to_numpy().sum() == 0.0
        metrics = calculate_interconnector_retrading_metrics(n, links, physical).set_index("interconnector")
        assert metrics.at["TestCable", "counter_trade_premium_GBP"] == 0.0

    def test_premium_survives_a_netcdf_round_trip(self, tmp_path):
        n, links, _ = self._solved()
        path = tmp_path / "bm.nc"
        n.export_to_netcdf(path)
        m = pypsa.Network(str(path))
        assert m.links.at[LINK, COUNTER_TRADE_PREMIUM_COLUMN] == pytest.approx(100.0)
        assert m.links.at[LINK, COUNTER_TRADE_SHARE_COLUMN] == pytest.approx(0.0)

    def test_summary_reports_volume_premium_and_bound_hours(self):
        n, links, counter = self._solved()
        summary = summarise_counter_trade(n.links_t.p0, links, n.links, counter.bounds).set_index("link")
        assert summary.at[LINK, "up_MWh"] == pytest.approx(200.0, abs=1e-3)
        assert summary.at[LINK, "premium_GBP"] == pytest.approx(20_000.0, rel=1e-6)
        assert summary.at[LINK, "hours_at_bound"] == 1


@pytest.mark.unit
class TestCounterTradeConfig:

    def _defaults(self):
        return load_defaults()

    def _cp30(self):
        return load_layer("source", "CP30")

    def test_code_defaults_match_defaults_yaml(self):
        block = self._defaults()["market"]["balancing"]["interconnector_counter_trade"]
        # The pricing rules (3 Oct 2026) are documented but not set in
        # config/defaults/base/market/balancing.yaml, so no merged config changes; absent means additive.
        unset = ("offer_pricing", "bid_pricing", "bid_premium_gbp_per_mwh")
        assert block == {k: v for k, v in DEFAULT_COUNTER_TRADE.items() if k not in unset}
        assert all(DEFAULT_COUNTER_TRADE[k] == "additive" for k in unset[:2])
        assert DEFAULT_COUNTER_TRADE["bid_premium_gbp_per_mwh"] is None
        assert resolve_counter_trade_config({"interconnector_counter_trade": block})["offer_pricing"] == "additive"

    def test_cp30_family_prices_offers_all_in_and_bids_at_the_gtma_discount(self):
        # User decision, 3 Oct 2026: offers all-in at 128 (2024 GBP; NESO GTMA 131,
        # FTI 2025 123), bids additive at 40; LCP caps and the 21 GBP gate kept.
        cp30 = self._cp30()["market"]["balancing"]["interconnector_counter_trade"]
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": cp30})
        assert (cfg["offer_pricing"], cfg["bid_pricing"]) == ("all_in", "additive")
        assert cfg["premium_gbp_per_mwh"] == 128.0 and cfg["bid_premium_gbp_per_mwh"] == 40.0
        assert cfg["max_deviation_share_by_country"] == DEFAULT_COUNTER_TRADE["max_deviation_share_by_country"]
        assert (cfg["min_spread_gbp_per_mwh"], cfg["min_spread_rule"]) == (21.0, "stage1_gate")

    def test_cp30_family_turns_it_on_and_prices_gas_offers_at_fti(self):
        cp30 = self._cp30()["market"]["balancing"]
        # on in the base for every scenario (D4); the CP30 source sets its premium and pricing
        assert self._defaults()["market"]["balancing"]["interconnector_counter_trade"]["enabled"] is True
        assert cp30["interconnector_counter_trade"]["premium_gbp_per_mwh"] == 128.0
        for carrier in ("CCGT", "OCGT", "gas_engine", "CCS Gas", "CHP"):
            assert cp30["carrier_overrides"][carrier]["offer_markup"] == pytest.approx(0.29)
            assert "bid_discount" not in cp30["carrier_overrides"][carrier]


@pytest.mark.unit
class TestAllInPricing:
    """The all_in rule (3 Oct 2026): FTI's cost of reversing flow is the whole price of a move.

    On the toy the export bid lands at 0.9^2 x 49.5 = 40.095 per MWh of link flow,
    the import ask is 50.5.
    """

    EXPORT = 0.9 ** 2 * 49.5

    def test_rules_are_validated_and_default_to_additive(self):
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {"enabled": True}})
        assert (cfg["offer_pricing"], cfg["bid_pricing"]) == ("additive", "additive")
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {"offer_pricing": "ALL_IN"}})
        assert cfg["offer_pricing"] == "all_in"
        with pytest.raises(ValueError):
            resolve_counter_trade_config({"interconnector_counter_trade": {"bid_pricing": "vibes"}})

    def test_marginal_trade_cost_follows_the_epigraph_side(self):
        n = _network(hours=3)
        ws = pd.DataFrame({LINK: [-200.0, 100.0, 0.0]}, index=n.snapshots)
        up, down = marginal_trade_cost(ws, n.links, n.links_t.marginal_cost)
        # Exporting: up gives up the export bid, down earns it; importing: the ask;
        # idle: up buys at the ask, down sells at the bid.
        assert up[LINK].tolist() == pytest.approx([self.EXPORT, 50.5, 50.5])
        assert down[LINK].tolist() == pytest.approx([-self.EXPORT, -50.5, -self.EXPORT])

    def test_marginal_trade_cost_includes_the_tranche_slope(self):
        n = _network(hours=2)
        n.links["market_tranche_group"] = ""
        n.links.loc[LINK, "market_tranche_group"] = "fr"
        for column, value in (("market_tranche_width_mw", 100.0), ("market_tranche_count", 3),
                              ("market_tranche_import_step", 2.0), ("market_tranche_export_step", 3.0)):
            n.links[column] = value
        ws = pd.DataFrame({LINK: [-250.0, -200.0]}, index=n.snapshots)
        up, down = marginal_trade_cost(ws, n.links, n.links_t.marginal_cost)
        # -250 is past both export boundaries (100, 200); -200 sits on the second,
        # so a move up leaves only the first behind it and a move down crosses both.
        assert up[LINK].tolist() == pytest.approx([self.EXPORT - 6.0, self.EXPORT - 3.0])
        assert down[LINK].tolist() == pytest.approx([-self.EXPORT + 6.0, -self.EXPORT + 6.0])

    def test_additive_rule_writes_no_rule_columns(self):
        n = _network()
        _, links = _schedule(n)
        prepare_interconnector_counter_trade(n, _balancing(), links, LOGGER)
        assert COUNTER_TRADE_OFFER_RULE_COLUMN not in n.links.columns
        assert COUNTER_TRADE_BID_RULE_COLUMN not in n.links.columns

    def test_all_in_offer_makes_unwinding_cheaper_than_gas(self):
        # Premium 60: additive, unwinding costs 40.1 + 60 = 100.1 per MWh (20,019)
        # against gas at 13,932, so gas runs; all_in, it costs max(60, 40.1) = 60
        # (12,000), so the export is cut instead.
        kw = dict(premium_gbp_per_mwh=60.0, max_deviation_share=None, max_deviation_share_by_country={})
        n, _, _ = _solve(_network(), _balancing(**kw))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(-200.0, abs=1e-3)
        n, links, _ = _solve(_network(), _balancing(offer_pricing="all_in", **kw))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        assert n.generators_t.p.at[n.snapshots[0], "gas"] == pytest.approx(0.0, abs=1e-3)
        # Objective = wind bid + the premium above the lost export revenue (no revenue kept).
        assert n.objective == pytest.approx(200 * WIND_BID + 200 * (60.0 - self.EXPORT), rel=1e-5)
        # Every report values the move at the all-in price: 200 MWh x 60.
        physical = n.links_t.p0[[LINK]]
        premium = counter_trade_premium_cost(physical, links, n.links, n.links_t.marginal_cost)
        assert float(premium.to_numpy().sum()) == pytest.approx(200 * (60.0 - self.EXPORT), rel=1e-6)
        metrics = calculate_interconnector_retrading_metrics(n, links, physical).set_index("interconnector")
        assert metrics.at["TestCable", "retrading_cost_GBP"] == pytest.approx(200 * 60.0, rel=1e-6)
        assert interconnector_retrading_cost(physical, links, n.links, n.links_t.marginal_cost) == pytest.approx(
            200 * 60.0, rel=1e-6)

    def test_all_in_never_prices_a_move_below_the_foreign_market(self):
        # Premium 30 is below the 40.1 export value: the move pays the trade value
        # alone (no premium), never less than the market.
        n = _network()
        _, links = _schedule(n)
        prepare_interconnector_counter_trade(
            n, _balancing(premium_gbp_per_mwh=30.0, offer_pricing="all_in"), links, LOGGER)
        physical = links * 0.0
        premium = counter_trade_premium_cost(physical, links, n.links, n.links_t.marginal_cost)
        assert float(premium.to_numpy().sum()) == pytest.approx(0.0, abs=1e-9)

    def test_all_in_bid_prices_a_reduced_import_at_the_premium(self):
        # Stage 1 imports 100 at the 50.5 ask; cutting it saves 5,050 of trade value,
        # so an all_in bid adds 150.5 per MWh and the move costs 100 x 100 in all.
        n = _network()
        sched = pd.DataFrame({LINK: [100.0]}, index=n.snapshots)
        prepare_interconnector_counter_trade(n, _balancing(bid_pricing="all_in"), sched, LOGGER)
        assert n.links.at[LINK, COUNTER_TRADE_OFFER_RULE_COLUMN] == "additive"
        assert n.links.at[LINK, COUNTER_TRADE_BID_RULE_COLUMN] == "all_in"
        physical = sched * 0.0
        premium = counter_trade_premium_cost(physical, sched, n.links, n.links_t.marginal_cost)
        assert float(premium.to_numpy().sum()) == pytest.approx(100 * 150.5, rel=1e-9)
        metrics = calculate_interconnector_retrading_metrics(n, sched, physical).set_index("interconnector")
        assert metrics.at["TestCable", "retrading_cost_GBP"] == pytest.approx(100 * 100.0, rel=1e-6)

    def test_rule_survives_a_netcdf_round_trip(self, tmp_path):
        kw = dict(premium_gbp_per_mwh=60.0, max_deviation_share=None, max_deviation_share_by_country={})
        n, links, _ = _solve(_network(), _balancing(offer_pricing="all_in", **kw))
        path = tmp_path / "bm.nc"
        n.export_to_netcdf(path)
        m = pypsa.Network(str(path))
        assert m.links.at[LINK, COUNTER_TRADE_OFFER_RULE_COLUMN] == "all_in"
        # The cable ends at zero, which PyPSA does not write; rebuild the flow.
        physical = m.links_t.p0.reindex(index=m.snapshots, columns=[LINK]).fillna(0.0)
        premium = counter_trade_premium_cost(physical, links, m.links, m.links_t.marginal_cost)
        assert float(premium.to_numpy().sum()) == pytest.approx(200 * (60.0 - self.EXPORT), rel=1e-6)

    def test_a_separate_bid_premium_prices_down_moves_only(self):
        # Offers all_in at 128, bids additive at 40 (the GTMA evidence): cutting a
        # 100 MW import saves 5,050 of trade value and pays 40 x 100 of premium.
        n = _network()
        sched = pd.DataFrame({LINK: [100.0]}, index=n.snapshots)
        prepare_interconnector_counter_trade(
            n, _balancing(premium_gbp_per_mwh=128.0, offer_pricing="all_in", bid_premium_gbp_per_mwh=40.0),
            sched, LOGGER)
        assert n.links.at[LINK, COUNTER_TRADE_PREMIUM_COLUMN] == pytest.approx(128.0)
        assert n.links.at[LINK, COUNTER_TRADE_BID_PREMIUM_COLUMN] == pytest.approx(40.0)
        physical = sched * 0.0
        premium = counter_trade_premium_cost(physical, sched, n.links, n.links_t.marginal_cost)
        assert float(premium.to_numpy().sum()) == pytest.approx(100 * 40.0, rel=1e-9)
        metrics = calculate_interconnector_retrading_metrics(n, sched, physical).set_index("interconnector")
        assert metrics.at["TestCable", "retrading_cost_GBP"] == pytest.approx(100 * (40.0 - 50.5), rel=1e-6)
        # An up move on the same link still pays the all-in offer price.
        up = pd.DataFrame({LINK: [-200.0]}, index=n.snapshots)
        premium = counter_trade_premium_cost(up * 0.0, up, n.links, n.links_t.marginal_cost)
        assert float(premium.to_numpy().sum()) == pytest.approx(200 * (128.0 - self.EXPORT), rel=1e-9)

    def test_bid_premium_unset_writes_no_column_and_negative_is_refused(self):
        n = _network()
        _, links = _schedule(n)
        prepare_interconnector_counter_trade(n, _balancing(offer_pricing="all_in"), links, LOGGER)
        assert COUNTER_TRADE_BID_PREMIUM_COLUMN not in n.links.columns
        with pytest.raises(ValueError):
            resolve_counter_trade_config({"interconnector_counter_trade": {"bid_premium_gbp_per_mwh": -1.0}})

    def test_bid_premium_steers_the_lp(self):
        # Additive everywhere, bid premium 0 but offer premium 100: the network still
        # needs the export cut (an up move), so the bid premium changes nothing here
        # and the objective is the offer-side one.
        kw = dict(premium_gbp_per_mwh=60.0, max_deviation_share=None, max_deviation_share_by_country={})
        n, _, _ = _solve(_network(), _balancing(offer_pricing="all_in", bid_premium_gbp_per_mwh=0.0, **kw))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        assert n.objective == pytest.approx(200 * WIND_BID + 200 * (60.0 - self.EXPORT), rel=1e-5)
