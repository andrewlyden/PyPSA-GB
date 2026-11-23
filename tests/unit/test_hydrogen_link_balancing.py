"""Hydrogen links in the balancing stage are anchored and priced at the hydrogen value.

Audit of 26 Sep 2026: every CP30 electrolyser sits on one copperplate GB_H2 bus
and the balancing objective, which replaces PyPSA's, left internal links free,
so Stage 2 moved electrolysis 8.2 TWh a year at no cost (about 14% of GB gross
redispatch). The user decision is to price that movement at the hydrogen value
Stage 1 implies. These tests pin the price formula, the anchor in the LP, and
the redispatch rows the movement now produces.
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import (
    compute_link_redispatch_volumes,
    estimate_hydrogen_value_from_schedule,
    hydrogen_link_prices,
    hydrogen_links,
    resolve_hydrogen_link_config,
    wholesale_hydrogen_price,
)
from scripts.market.solve_balancing import (
    _build_balancing_extra_functionality,
    _prepare_hydrogen_link_pricing,
)
from config.config_loader import load_layer

LOGGER = logging.getLogger("test_hydrogen_links")
ETA = 0.7
CHEAP_OFFER, CHEAP_BID = 12.0, 9.0
DEAR_OFFER, DEAR_BID = 60.0, -54.0


@pytest.fixture
def h2_network():
    """A 100 MW line from A (cheap) to B (dear gas, a 100 MW electrolyser, load 100/0/0 MW).

    Stage 1 ran the cheap unit at 150/50/50 MW for the load plus 50 MW of
    electrolysis at B, overloading the line by 50 MW in hour 0 only. The hydrogen
    store is cyclic over the window, as GB_H2_storage is, so the offtake fixes the
    hydrogen made in the window: electrolysis can move between hours but not in
    net. The balancing stage must turn the cheap unit down 50 MW in hour 0 (bid 9)
    and either turn the dear unit up (offer 60) or shift 50 MWh of electrolysis
    out of hour 0 into hours 1-2, fed by the cheap unit (offer 12).
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=3, freq="h"))
    for carrier in ("AC", "H2_gas", "wind_onshore", "CCGT", "electrolysis", "H2_turbine"):
        n.add("Carrier", carrier)
    n.add("Bus", "A", v_nom=400, carrier="AC")
    n.add("Bus", "B", v_nom=400, carrier="AC")
    n.add("Bus", "GB_H2", carrier="H2_gas")
    n.add("Line", "A_B", bus0="A", bus1="B", r=0.01, x=0.1, s_nom=100.0, carrier="AC")
    n.add("Generator", "cheap", bus="A", p_nom=300.0, carrier="wind_onshore", marginal_cost=10.0)
    n.add("Generator", "dear", bus="B", p_nom=300.0, carrier="CCGT", marginal_cost=100.0)
    n.add("Load", "load", bus="B", p_set=pd.Series([100.0, 0.0, 0.0], index=n.snapshots))
    n.add("Link", "electrolysis_B", bus0="B", bus1="GB_H2", p_nom=100.0,
          efficiency=ETA, carrier="electrolysis", marginal_cost=0.0)
    n.add("Store", "GB_H2_storage", bus="GB_H2", e_nom=1000.0, e_cyclic=True, carrier="H2_gas")
    n.add("Load", "h2_offtake", bus="GB_H2", p_set=35.0)
    return n


def _add_turbine(n):
    n.add("Link", "H2_turbine_B", bus0="GB_H2", bus1="B", p_nom=40.0,
          efficiency=0.5, carrier="H2_turbine", marginal_cost=5.0)
    return n


def _positions(n):
    snaps = n.snapshots
    gen = pd.DataFrame({"cheap": [150.0, 50.0, 50.0], "dear": 0.0}, index=snaps)
    su = pd.DataFrame(index=snaps)
    links = pd.DataFrame({"electrolysis_B": 50.0}, index=snaps)
    return gen, su, links


def _solve(n, hydrogen_value=None):
    gen, su, links = _positions(n)
    kwargs = {}
    if hydrogen_value is not None:
        names = list(hydrogen_links(n, ("electrolysis",), ("H2_gas",)))
        inc, dec = hydrogen_link_prices(
            n, names, pd.Series(hydrogen_value, index=n.snapshots), 0.10, 0.10, ("H2_gas",)
        )
        kwargs = dict(anchored_links=names, link_increase_prices=inc, link_decrease_prices=dec)
    extra = _build_balancing_extra_functionality(
        wholesale_gen=gen, wholesale_su=su, wholesale_links=links,
        gen_offer_prices=pd.Series({"cheap": CHEAP_OFFER, "dear": DEAR_OFFER}),
        gen_bid_prices=pd.Series({"cheap": CHEAP_BID, "dear": DEAR_BID}),
        su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
        fix_interconnectors=False, logger=LOGGER, **kwargs,
    )
    status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
    assert status == "ok", condition
    return n, links


@pytest.mark.unit
class TestHydrogenLinkPrices:

    def test_electrolyser_and_turbine_prices(self, h2_network):
        _add_turbine(h2_network)
        value = pd.Series([100.0, 0.0, 20.0], index=h2_network.snapshots)
        inc, dec = hydrogen_link_prices(
            h2_network, ["electrolysis_B", "H2_turbine_B"], value, 0.10, 0.20, ("H2_gas",)
        )
        # Electrolyser: c = -0.7 x lambda. Drawing more earns NESO 0.9 x 70;
        # drawing less costs NESO 1.2 x 70, the hydrogen no longer made.
        assert inc["electrolysis_B"].tolist() == pytest.approx([-63.0, 0.0, -12.6])
        assert dec["electrolysis_B"].tolist() == pytest.approx([84.0, 0.0, 16.8])
        # Turbine: c = 5 + lambda per MWh of hydrogen burned.
        assert inc["H2_turbine_B"].tolist() == pytest.approx([115.5, 5.5, 27.5])
        assert dec["H2_turbine_B"].tolist() == pytest.approx([-84.0, -4.0, -20.0])

    def test_a_shift_within_the_day_costs_the_spread_not_nothing(self, h2_network):
        value = pd.Series(100.0, index=h2_network.snapshots)
        inc, dec = hydrogen_link_prices(h2_network, ["electrolysis_B"], value, 0.10, 0.10, ("H2_gas",))
        round_trip = inc["electrolysis_B"].iloc[0] + dec["electrolysis_B"].iloc[1]
        assert round_trip == pytest.approx(0.2 * ETA * 100.0)

    def test_config_defaults_follow_the_balancing_markups(self):
        cfg = resolve_hydrogen_link_config({"default_offer_markup": 0.2, "default_bid_discount": 0.3})
        assert cfg["enabled"] is True
        assert cfg["offer_markup"] == 0.2 and cfg["bid_discount"] == 0.3
        assert set(cfg["carriers"]) == {"electrolysis", "H2_turbine"}

    def test_wholesale_hydrogen_price_is_the_hydrogen_bus_dual(self, h2_network):
        prices = pd.DataFrame(
            {"A": [50.0, 60.0, 70.0], "B": [50.0, 60.0, 70.0], "GB_H2": [71.4, 85.7, 100.0]},
            index=h2_network.snapshots,
        )
        value = wholesale_hydrogen_price(prices, h2_network)
        assert value.tolist() == pytest.approx([71.4, 85.7, 100.0])
        assert wholesale_hydrogen_price(prices.drop(columns="GB_H2"), h2_network) is None

    def test_fallback_estimate_is_the_electrolysis_weighted_price_over_efficiency(self, h2_network):
        _add_turbine(h2_network)
        price = pd.Series([10.0, 40.0, 1000.0], index=h2_network.snapshots)
        links = pd.DataFrame({"electrolysis_B": [100.0, 100.0, 0.0]}, index=h2_network.snapshots)
        value = estimate_hydrogen_value_from_schedule(
            price, links, h2_network, ["electrolysis_B", "H2_turbine_B"], 24, LOGGER, ("H2_gas",)
        )
        assert value.tolist() == pytest.approx([25.0 / ETA] * 3)


@pytest.mark.unit
class TestHydrogenLinksInTheBalancingLP:
    CHEAP_DOWN = 50.0 * CHEAP_BID          # unavoidable in hour 0
    SHIFT_FEED = 50.0 * CHEAP_OFFER        # 50 MWh more from A in hours 1-2

    def test_unanchored_electrolysis_is_shifted_for_free(self, h2_network):
        """The defect: without an anchor the BM moves electrolysis out of hour 0 at no cost."""
        n, _ = _solve(h2_network)
        assert n.links_t.p0["electrolysis_B"].iloc[0] == pytest.approx(0.0, abs=1e-4)
        assert n.links_t.p0["electrolysis_B"].sum() == pytest.approx(150.0, abs=1e-4)
        assert n.objective == pytest.approx(self.CHEAP_DOWN + self.SHIFT_FEED, rel=1e-6)

    def test_valuable_hydrogen_keeps_electrolysis_on_schedule(self, h2_network):
        """At 500 GBP/MWh_H2 the shift costs the spread 0.2 x 0.7 x 500 = 70, plus 12, above 60."""
        n, _ = _solve(h2_network, hydrogen_value=500.0)
        assert n.links_t.p0["electrolysis_B"].tolist() == pytest.approx([50.0] * 3, abs=1e-4)
        assert n.generators_t.p["dear"].iloc[0] == pytest.approx(50.0, abs=1e-4)
        assert n.objective == pytest.approx(self.CHEAP_DOWN + 50.0 * DEAR_OFFER, rel=1e-6)

    def test_cheap_hydrogen_is_still_shifted_and_the_shift_is_paid_for(self, h2_network):
        """At 10 GBP/MWh_H2 the shift costs 1.4 + 12, below 60, and its spread is booked."""
        n, links = _solve(h2_network, hydrogen_value=10.0)
        assert n.links_t.p0["electrolysis_B"].iloc[0] == pytest.approx(0.0, abs=1e-4)
        down = 50.0 * ETA * 10.0 * 1.1      # NESO pays for the hydrogen not made in hour 0
        up = -50.0 * ETA * 10.0 * 0.9       # and receives for the extra made in hours 1-2
        assert n.objective == pytest.approx(self.CHEAP_DOWN + self.SHIFT_FEED + down + up, rel=1e-6)

        inc, dec = hydrogen_link_prices(
            n, ["electrolysis_B"], pd.Series(10.0, index=n.snapshots), 0.1, 0.1, ("H2_gas",)
        )
        rows = compute_link_redispatch_volumes(links, n.links_t.p0, inc, dec, n, LOGGER, ("H2_gas",))
        row = rows.set_index("component").loc["electrolysis_B"]
        # Drawing less in hour 0 puts power back on the system (an offer); drawing
        # more later takes it off (a bid, which NESO is paid for).
        assert row["increase_MWh"] == pytest.approx(50.0, abs=1e-3)
        assert row["decrease_MWh"] == pytest.approx(50.0, abs=1e-3)
        assert row["offer_cost"] == pytest.approx(down, rel=1e-6)
        assert row["bid_cost"] == pytest.approx(up, rel=1e-6)
        assert row["type"] == "link"

    def test_prepare_uses_the_stage1_column_and_honours_the_switch(self, h2_network):
        _add_turbine(h2_network)
        _, _, links = _positions(h2_network)
        links["H2_turbine_B"] = 0.0
        price = pd.DataFrame(
            {"wholesale_price": 50.0, "hydrogen_price": [100.0, 110.0, 120.0]},
            index=h2_network.snapshots,
        )
        names, inc, dec, held = _prepare_hydrogen_link_pricing(
            h2_network, {"hydrogen_links": {"bus_carriers": ["H2_gas"]}}, price, links, {}, LOGGER
        )
        # Decision D5: turbines are held at Stage 1 by default, not priced.
        assert names == ["electrolysis_B"] and held == ["H2_turbine_B"]
        assert dec["electrolysis_B"].tolist() == pytest.approx([77.0, 84.7, 92.4])
        names, _, _, held = _prepare_hydrogen_link_pricing(
            h2_network, {"hydrogen_links": {"bus_carriers": ["H2_gas"], "hold_carriers": []}},
            price, links, {}, LOGGER
        )
        assert set(names) == {"electrolysis_B", "H2_turbine_B"} and held == []
        off = _prepare_hydrogen_link_pricing(
            h2_network, {"hydrogen_links": {"enabled": False}}, price, links, {}, LOGGER
        )
        assert off == ([], None, None, [])

    def test_held_turbine_stays_at_its_stage1_position(self, h2_network):
        """Decision D5: a held turbine cannot be turned up, even where it would relieve B."""
        _add_turbine(h2_network)
        gen, su, links = _positions(h2_network)
        links["H2_turbine_B"] = 10.0
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen, wholesale_su=su, wholesale_links=links,
            gen_offer_prices=pd.Series({"cheap": CHEAP_OFFER, "dear": DEAR_OFFER}),
            gen_bid_prices=pd.Series({"cheap": CHEAP_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
            fix_interconnectors=False, logger=LOGGER, fixed_links=["H2_turbine_B"],
        )
        status, condition = h2_network.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        assert h2_network.links_t.p0["H2_turbine_B"].tolist() == pytest.approx([10.0] * 3, abs=1e-6)


@pytest.mark.unit
class TestElectrolysersHeldForCP30:
    """User decision, 3 Oct 2026: CP30 electrolysers out of the BM (FTI 2023 A1.21 keeps demand out)."""

    def _cp30_hold(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        cfg = load_layer("source", "CP30")
        return cfg["market"]["balancing"]["hydrogen_links"]["hold_carriers"]

    def test_cp30_holds_electrolysers_and_turbines(self, h2_network):
        hold = self._cp30_hold()
        assert set(hold) == {"H2_turbine", "electrolysis"}
        _add_turbine(h2_network)
        _, _, links = _positions(h2_network)
        links["H2_turbine_B"] = 0.0
        price = pd.DataFrame({"wholesale_price": 50.0, "hydrogen_price": 100.0}, index=h2_network.snapshots)
        names, inc, dec, held = _prepare_hydrogen_link_pricing(
            h2_network, {"hydrogen_links": {"bus_carriers": ["H2_gas"], "hold_carriers": hold}},
            price, links, {}, LOGGER,
        )
        assert names == [] and inc is None and dec is None
        assert set(held) == {"electrolysis_B", "H2_turbine_B"}

    def test_a_held_electrolyser_is_not_turned_down_and_the_bm_stays_feasible(self, h2_network):
        # Hour 0 overloads the line by 50 MW. With the electrolyser held at its 50 MW,
        # the BM cannot shift electrolysis; it turns the cheap unit down and the dear
        # unit up instead.
        gen, su, links = _positions(h2_network)
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen, wholesale_su=su, wholesale_links=links,
            gen_offer_prices=pd.Series({"cheap": CHEAP_OFFER, "dear": DEAR_OFFER}),
            gen_bid_prices=pd.Series({"cheap": CHEAP_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
            fix_interconnectors=False, logger=LOGGER, fixed_links=["electrolysis_B"],
        )
        status, condition = h2_network.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        assert h2_network.links_t.p0["electrolysis_B"].tolist() == pytest.approx([50.0] * 3, abs=1e-6)
        assert h2_network.generators_t.p.at[h2_network.snapshots[0], "dear"] == pytest.approx(50.0, abs=1e-3)
