"""market.hydrogen_supply_price: the reformer's hydrogen at an external long-run cost (2050 price calibration)."""
import logging

import pypsa
import pytest

from scripts.market.market_utils import HYDROGEN_REFORMER_LINK, apply_hydrogen_supply_price

LOG = logging.getLogger(__name__)


def _network(gas=23.64, link_cost=2.0, eta=0.8341):
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "gas")
    n.add("Bus", "h2")
    n.add("Generator", "gas_supply", bus="gas", p_nom=1e4, marginal_cost=gas)
    n.add("Link", HYDROGEN_REFORMER_LINK, bus0="gas", bus1="h2", p_nom=100.0, efficiency=eta, marginal_cost=link_cost)
    return n


def test_off_by_default_leaves_the_reformer_alone():
    n = _network()
    assert apply_hydrogen_supply_price(n, {}, LOG) is None
    assert n.links.at[HYDROGEN_REFORMER_LINK, "marginal_cost"] == 2.0


def test_sets_the_cost_per_mwh_of_hydrogen_out():
    n = _network()
    report = apply_hydrogen_supply_price(n, {"hydrogen_supply_price": {"enabled": True, "gbp_per_mwh_h2": 64.0}}, LOG)
    link = n.links.loc[HYDROGEN_REFORMER_LINK]
    assert (23.64 + link["marginal_cost"]) / link["efficiency"] == pytest.approx(64.0)
    assert report["short_run_gbp_per_mwh_h2"] == pytest.approx((23.64 + 2.0) / 0.8341)


def test_refuses_a_price_below_the_short_run_cost():
    n = _network()
    with pytest.raises(ValueError, match="below the reformer's short-run cost"):
        apply_hydrogen_supply_price(n, {"hydrogen_supply_price": {"enabled": True, "gbp_per_mwh_h2": 20.0}}, LOG)


def test_refuses_a_network_without_the_reformer():
    n = _network()
    n.remove("Link", HYDROGEN_REFORMER_LINK)
    with pytest.raises(ValueError, match="no link"):
        apply_hydrogen_supply_price(n, {"hydrogen_supply_price": {"enabled": True, "gbp_per_mwh_h2": 64.0}}, LOG)
