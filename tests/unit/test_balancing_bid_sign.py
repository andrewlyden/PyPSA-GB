"""Derived thermal bids hand the fuel saving to NESO, so the BM turns gas down.

Before 26 Sep 2026 the derived bid was +0.9 x marginal cost in ESO-cost
convention, a payment by NESO for turning a gas unit down, while the ELEXON and
future-bid-stack paths negate the raw bid into a receipt. Stage 2 therefore
never turned gas down (CCGT decrease 0.06 MWh in CP30 December 2030) and
curtailed renewables instead. Also pinned here: a time-varying p_min_pu replaces
the static floor in the BM bounds, as it does in PyPSA.
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import calculate_bid_offer_prices
from scripts.market.solve_balancing import _build_balancing_extra_functionality

LOGGER = logging.getLogger("test_bid_sign")


def _export_constrained_network():
    """Wind (100 MW) and gas (100 MW, mc 100) at A behind a 100 MW line; load 150 at B.

    Stage 1 ran wind 100 + gas 50 at A. The BM must take 50 MW off A and add
    50 MW at B (peaker, mc 150). At A it can turn wind down (bid 50, paid by
    NESO) or gas down (the fuel saving, 90, received by NESO).
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for carrier in ("AC", "wind_onshore", "CCGT", "OCGT"):
        n.add("Carrier", carrier)
    n.add("Bus", "A", v_nom=400, carrier="AC")
    n.add("Bus", "B", v_nom=400, carrier="AC")
    n.add("Line", "A_B", bus0="A", bus1="B", r=0.01, x=0.1, s_nom=100.0, carrier="AC")
    n.add("Generator", "wind", bus="A", p_nom=100.0, carrier="wind_onshore", marginal_cost=0.0)
    n.add("Generator", "gas_A", bus="A", p_nom=100.0, carrier="CCGT", marginal_cost=100.0)
    n.add("Generator", "peaker_B", bus="B", p_nom=100.0, carrier="OCGT", marginal_cost=150.0)
    n.add("Load", "load", bus="B", p_set=150.0)
    return n


def _market(convention):
    return {"balancing": {
        "bid_offer_source": "derived", "default_offer_markup": 0.1, "default_bid_discount": 0.1,
        "derived_bid_convention": convention,
        "carrier_overrides": {"wind_onshore": {"mode": "absolute", "offer_price": 0.0, "bid_price": 50.0}},
    }}


def _solve(n, convention):
    offer, bid, _, _ = calculate_bid_offer_prices(n, _market(convention), LOGGER)
    snaps = n.snapshots
    ws = pd.DataFrame({"wind": 100.0, "gas_A": 50.0, "peaker_B": 0.0}, index=snaps)
    extra = _build_balancing_extra_functionality(
        wholesale_gen=ws, wholesale_su=pd.DataFrame(index=snaps), wholesale_links=pd.DataFrame(index=snaps),
        gen_offer_prices=offer, gen_bid_prices=bid,
        su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
        fix_interconnectors=False, logger=LOGGER,
    )
    status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
    assert status == "ok", condition
    return n, bid


@pytest.mark.unit
class TestThermalBidSign:

    def test_gas_is_turned_down_before_wind(self):
        n, bid = _solve(_export_constrained_network(), "net_saving")
        assert bid["gas_A"] == pytest.approx(-90.0)
        assert n.generators_t.p["gas_A"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
        assert n.generators_t.p["wind"].tolist() == pytest.approx([100.0, 100.0], abs=1e-6)
        # Per hour: peaker offer 165 x 50, less the 90 x 50 gas saving NESO receives.
        assert n.objective == pytest.approx(2 * 50.0 * (165.0 - 90.0), rel=1e-6)

    def test_legacy_convention_curtailed_wind_instead(self):
        n, bid = _solve(_export_constrained_network(), "legacy_cost")
        assert bid["gas_A"] == pytest.approx(90.0)
        assert n.generators_t.p["gas_A"].tolist() == pytest.approx([50.0, 50.0], abs=1e-6)
        assert n.generators_t.p["wind"].tolist() == pytest.approx([50.0, 50.0], abs=1e-6)


@pytest.mark.unit
def test_time_varying_floor_replaces_the_static_one_in_the_bm():
    """A heat-led CHP floor of 10% in summer is honoured, not the 30% annual one."""
    n = _export_constrained_network()
    n.add("Generator", "chp_A", bus="A", p_nom=100.0, carrier="CHP", marginal_cost=90.0, p_min_pu=0.3)
    n.generators_t.p_min_pu = pd.DataFrame({"chp_A": 0.1}, index=n.snapshots)
    snaps = n.snapshots
    # Stage 1 (floor 10%): wind 100, chp 25 at A; gas off. Line overloaded by 25.
    ws = pd.DataFrame({"wind": 100.0, "gas_A": 0.0, "peaker_B": 25.0, "chp_A": 25.0}, index=snaps)
    offer = pd.Series({"wind": 0.0, "gas_A": 110.0, "peaker_B": 165.0, "chp_A": 99.0})
    bid = pd.Series({"wind": 50.0, "gas_A": -90.0, "peaker_B": -135.0, "chp_A": -81.0})
    extra = _build_balancing_extra_functionality(
        wholesale_gen=ws, wholesale_su=pd.DataFrame(index=snaps), wholesale_links=pd.DataFrame(index=snaps),
        gen_offer_prices=offer, gen_bid_prices=bid,
        su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
        fix_interconnectors=False, logger=LOGGER,
    )
    status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
    assert status == "ok", condition
    # CHP can come down to its 10 MW summer floor (15 MW of the 25 needed);
    # with the old max(static, varying) floor it could not move at all.
    assert n.generators_t.p["chp_A"].tolist() == pytest.approx([10.0, 10.0], abs=1e-6)
