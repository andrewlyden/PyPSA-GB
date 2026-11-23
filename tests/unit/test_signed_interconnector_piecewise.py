import logging

import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import add_interconnector_links
from scripts.interconnectors.market_coupling import (
    build_piecewise_interconnector_cost_callback,
    get_piecewise_interconnectors,
    get_piecewise_market_prices,
)
from scripts.market.analyze_market import calculate_interconnector_metrics


def _network(hours=2, foreign_price=20.0, entsoe_prices=None):
    network = pypsa.Network()
    snapshots = pd.date_range("2030-01-01", periods=hours, freq="h")
    network.set_snapshots(snapshots)
    network.add("Bus", "GB")
    network.add("Bus", "External_France")
    cables = pd.DataFrame([{
        "name": "TestCable", "from_bus": "GB", "to_bus": "External_France",
        "capacity_mw": 100.0, "losses_percent": 10.0,
        "counterparty_country": "France",
    }])
    prices = pd.DataFrame([{
        "country": "france", "year": 2030, "pathway": "Central",
        "estimated_price_gbp_per_mwh": foreign_price,
    }])
    add_interconnector_links(
        network, cables, pd.DataFrame(), prices,
        modelled_year=2030, fes_pathway="Central",
        entsoe_prices_df=entsoe_prices,
        pricing_config={"transaction_spread_gbp_per_mwh": 1.0},
    )
    return network, snapshots


def test_one_signed_link_has_bidirectional_physical_bounds():
    network, _ = _network()
    assert get_piecewise_interconnectors(network) == {"TestCable": "IC_TestCable"}
    link = network.links.loc["IC_TestCable"]
    assert link.p_min_pu == -1.0
    assert link.p_max_pu == 1.0
    assert link.efficiency == pytest.approx(0.9)
    assert link.foreign_market_pricing == "piecewise_bidirectional"


def test_piecewise_lp_prices_imports_without_counterflow():
    network, snapshots = _network(hours=1)
    network.add("Load", "demand", bus="GB", p_set=45.0)
    network.add("Generator", "domestic", bus="GB", p_nom=100.0, marginal_cost=100.0)
    status, condition = network.optimize(
        solver_name="highs",
        extra_functionality=build_piecewise_interconnector_cost_callback(network),
    )
    assert status == "ok", condition
    assert network.links_t.p0.at[snapshots[0], "IC_TestCable"] == pytest.approx(50.0)


def test_hourly_ask_and_reconstructed_bid_drive_signed_link():
    entsoe = pd.DataFrame({"France": [10.0, 30.0]}, index=pd.date_range("2020-01-01", periods=2, freq="h"))
    network, snapshots = _network(entsoe_prices=entsoe)
    ask, bid = get_piecewise_market_prices(network, "IC_TestCable", snapshots)
    assert ask.tolist() == pytest.approx([10.5, 30.5])
    assert bid.tolist() == pytest.approx([9.5, 29.5])


def test_signed_reporting_uses_flow_sign_for_import_and_export():
    network, snapshots = _network()
    dispatch = pd.DataFrame({"IC_TestCable": [50.0, -40.0]}, index=snapshots)
    metrics = calculate_interconnector_metrics(
        network, dispatch, pd.Series([50.0, 50.0], index=snapshots), logging.getLogger(__name__)
    ).set_index("interconnector")
    cable = metrics.loc["TestCable"]
