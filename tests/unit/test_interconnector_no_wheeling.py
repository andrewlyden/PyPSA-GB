"""No-wheeling rule for interconnector counter-trades (17 Oct 2026).

``interconnector_counter_trade.no_wheeling`` forbids same-hour opposite counter-trades: more export on one cable and
more import on another, a path round a GB boundary through Europe.

Toy (three cables): wind at N (300 MW, bid 50) behind a 100 MW line to S (load 300 MW, gas offer 77.4). Cable A at N
(Norway), cables B and C at S (France, Netherlands), 200 MW each, every foreign market at ask 50.5 / bid 49.5,
losses 10%. Stage 1 (copperplate) ran wind at 300 to serve the load, every cable idle. The line forces 200 MW out
of N:

* without the rule the BM wheels: A exports in full at N and B and C import at S (objective 8,406 over two
  hours), with no gas;
* with it, an hour moves one way only. Exporting at N with gas up at S beats turning wind down (50) and
  importing at S, so the solve keeps A's export and replaces the imports with gas (16,922).
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import add_interconnector_links
from scripts.interconnectors.market_coupling import build_piecewise_interconnector_cost_callback
from scripts.market.interconnector_counter_trade import (
    DEFAULT_COUNTER_TRADE,
    NO_WHEELING_FORMULATIONS,
    build_counter_trade_callback,
    no_wheeling_caps,
    no_wheeling_solver_options,
    resolve_counter_trade_config,
)
from scripts.market.solve_balancing import (
    _build_balancing_extra_functionality,
    prepare_interconnector_counter_trade,
)
from scripts.solve.hydro_constraints import combine_extra_functionalities

LOGGER = logging.getLogger("test_no_wheeling")
CABLES = {"IC_A": ("A", "N", "Norway"), "IC_B": ("B", "S", "France"), "IC_C": ("C", "S", "Netherlands")}
WIND_BID, GAS_OFFER = 50.0, 77.4


def _network(hours: int = 2):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    for bus in ("N", "S", "External_Norway", "External_France", "External_Netherlands"):
        n.add("Bus", bus)
    n.add("Line", "N_S", bus0="N", bus1="S", x=0.1, r=0.01, s_nom=100.0)
    n.add("Generator", "wind", bus="N", p_nom=300.0, marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "gas", bus="S", p_nom=300.0, marginal_cost=60.0, carrier="CCGT")
    n.add("Load", "load", bus="S", p_set=300.0)
    cables = pd.DataFrame([{"name": name, "from_bus": bus, "to_bus": f"External_{country}", "capacity_mw": 200.0,
                            "losses_percent": 10.0, "counterparty_country": country}
                           for name, bus, country in CABLES.values()])
    prices = pd.DataFrame([{"country": c.lower(), "year": 2030, "pathway": "Central",
                            "estimated_price_gbp_per_mwh": 50.0} for _, _, c in CABLES.values()])
    add_interconnector_links(n, cables, pd.DataFrame(), prices, modelled_year=2030,
                             fes_pathway="Central", pricing_config={"transaction_spread_gbp_per_mwh": 1.0})
    return n


def _balancing(**extra):
    return {"interconnector_retrading": True,
            "interconnector_counter_trade": {
                "enabled": True, "premium_gbp_per_mwh": 0.0, "max_deviation_share": 2.0,
                "max_deviation_share_by_country": {}, "min_spread_gbp_per_mwh": 0.0,
                "min_spread_rule": "none", **extra}}


def _solve(n, balancing):
    gen = pd.DataFrame({"wind": 300.0, "gas": 0.0}, index=n.snapshots)
    links = pd.DataFrame({name: 0.0 for name in CABLES}, index=n.snapshots)
    prepare_interconnector_counter_trade(n, balancing, links, LOGGER)
    extra = _build_balancing_extra_functionality(
        wholesale_gen=gen, wholesale_su=pd.DataFrame(index=n.snapshots), wholesale_links=links,
        gen_offer_prices=pd.Series({"wind": 0.0, "gas": GAS_OFFER}),
        gen_bid_prices=pd.Series({"wind": WIND_BID, "gas": -54.0}),
        su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
        fix_interconnectors=False, logger=LOGGER,
    )
    counter = build_counter_trade_callback(n, links, pd.Series(50.0, index=n.snapshots), balancing, LOGGER)
    piecewise = build_piecewise_interconnector_cost_callback(
        n, LOGGER, baseline_dispatch=links, subtract_native_link_cost=False)
    status, condition = n.optimize(
        solver_name="highs", extra_functionality=combine_extra_functionalities(extra, counter, piecewise))
    assert status == "ok", condition
    return n, links


def _moves(n, links):
    dev = n.links_t.p0[list(CABLES)] - links
    return dev.clip(lower=0.0).sum(axis=1), (-dev).clip(lower=0.0).sum(axis=1)


@pytest.mark.unit
class TestNoWheeling:

    def test_config_default_is_off_and_validated(self):
        assert DEFAULT_COUNTER_TRADE["no_wheeling"] is False
        assert DEFAULT_COUNTER_TRADE["no_wheeling_formulation"] == "binary"
        assert set(NO_WHEELING_FORMULATIONS) == {"binary", "hull"}
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {"no_wheeling": True,
                                                                           "no_wheeling_formulation": "HULL"}})
        assert cfg["no_wheeling"] is True and cfg["no_wheeling_formulation"] == "hull"
        with pytest.raises(ValueError, match="no_wheeling_formulation"):
            resolve_counter_trade_config({"interconnector_counter_trade": {"no_wheeling_formulation": "sos"}})

    def test_without_the_rule_the_bm_wheels_through_europe(self):
        n, links = _solve(_network(), _balancing())
        up, down = _moves(n, links)
        assert (np.minimum(up, down) > 150.0).all()
        assert n.links_t.p0["IC_A"].tolist() == pytest.approx([-200.0, -200.0], abs=1e-3)   # A exports in full at N
        assert (n.generators_t.p["gas"].abs() < 1e-3).all()                 # S is served by imports, no gas
        assert n.objective == pytest.approx(8406.44, abs=0.1)
        assert "Link-counter-trade-direction" not in n.model.variables

    def test_binary_rule_moves_each_hour_one_way(self):
        n, links = _solve(_network(), _balancing(no_wheeling=True))
        up, down = _moves(n, links)
        assert (np.minimum(up, down) < 1e-3).all()
        assert "Link-counter-trade-direction" in n.model.variables
        # Export at N and gas up at S (cheaper than wind down + import at S), every hour.
        assert n.links_t.p0["IC_A"].tolist() == pytest.approx([-200.0, -200.0], abs=1e-3)
        assert (n.links_t.p0[["IC_B", "IC_C"]].abs() < 1e-3).all().all()
        assert n.generators_t.p["gas"].tolist() == pytest.approx([200.0, 200.0], abs=1e-2)
        assert n.objective == pytest.approx(16922.0, abs=0.1)

    def test_binary_rule_costs_more_than_wheeling(self):
        free, _ = _solve(_network(), _balancing())
        held, _ = _solve(_network(), _balancing(no_wheeling=True))
        assert held.objective > free.objective + 1.0

    def test_hull_is_a_linear_relaxation(self):
        n, links = _solve(_network(), _balancing(no_wheeling=True, no_wheeling_formulation="hull"))
        assert "Link-counter-trade-direction" not in n.model.variables
        assert "Link-counter-trade-no-wheeling-hull" in n.model.constraints
        up, down = _moves(n, links)
        # Physical headroom from idle: 600 MW each way, so U/600 + D/600 <= 1.
        assert ((up / 600.0 + down / 600.0) <= 1.0 + 1e-6).all()
        free, _ = _solve(_network(), _balancing())
        bin_, _ = _solve(_network(), _balancing(no_wheeling=True))
        assert free.objective - 1e-6 <= n.objective <= bin_.objective + 1e-6

    def test_caps_are_the_physical_headroom_at_stage1(self):
        n = _network(hours=1)
        ws = pd.DataFrame({"IC_A": [-150.0], "IC_B": [100.0], "IC_C": [0.0]}, index=n.snapshots)
        band = pd.DataFrame(400.0, index=n.snapshots, columns=list(CABLES))
        up, down = no_wheeling_caps(n, list(CABLES), ws, band, band)
        assert up.iloc[0].tolist() == pytest.approx([350.0, 100.0, 200.0])
        assert down.iloc[0].tolist() == pytest.approx([50.0, 300.0, 200.0])

    def test_mip_gap_reaches_the_solver_only_for_the_binary_rule(self):
        assert DEFAULT_COUNTER_TRADE["no_wheeling_mip_gap"] is None
        opts = {"method": 2}
        on = {"interconnector_counter_trade": {"enabled": True, "no_wheeling": True, "no_wheeling_mip_gap": 1e-3}}
        assert no_wheeling_solver_options("gurobi", opts, on) == {"method": 2, "MIPGap": 1e-3}
        assert no_wheeling_solver_options("highs", opts, on) == {"method": 2, "mip_rel_gap": 1e-3}
        assert opts == {"method": 2}                                   # the caller's dict is not changed
        hull = {"interconnector_counter_trade": dict(on["interconnector_counter_trade"], no_wheeling_formulation="hull")}
        assert no_wheeling_solver_options("gurobi", opts, hull) == opts
        off = {"interconnector_counter_trade": {"enabled": True, "no_wheeling_mip_gap": 1e-3}}
        assert no_wheeling_solver_options("gurobi", opts, off) == opts
        unset = {"interconnector_counter_trade": {"enabled": True, "no_wheeling": True}}
        assert no_wheeling_solver_options("gurobi", opts, unset) == opts
        with pytest.raises(ValueError, match="no_wheeling_mip_gap"):
            resolve_counter_trade_config({"interconnector_counter_trade": {"no_wheeling_mip_gap": 2.0}})
