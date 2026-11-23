"""Item D2 (single hourly foreign price, export loss) and item C9 (BM cost net of Stage 1 IC value)."""
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import (
    _foreign_bid_ask_prices,
    add_interconnector_links,
    single_price_market_shapes,
    single_price_reference,
)
from scripts.interconnectors.market_coupling import (
    baseline_trade_value,
    build_piecewise_interconnector_cost_callback,
    export_efficiency,
    get_piecewise_market_prices,
    interconnector_trade_cost,
    link_price_tables,
    stage1_interconnector_trade_value,
)

MARKETS = {"continent": {"counterparties": ["France", "Netherlands"]}, "sem": {"counterparties": ["Ireland"]}}


def _pricing(**extra):
    return {"source": "cp30", "foreign_price_basis": "single", "transaction_spread_gbp_per_mwh": 1.0,
            "circulation_epsilon_gbp_per_mwh": 0.01, "cp30_import_price_gbp_per_mwh": 89.43,
            "cp30_export_price_gbp_per_mwh": 40.09, "volume_tranches": {"markets": MARKETS},
            "single_price_level_offsets_gbp_per_mwh": {"continent": 5.0, "sem": -3.0}, **extra}


def _entsoe(hours=2):
    idx = pd.date_range("2024-01-01", periods=hours, freq="h")
    return pd.DataFrame({"France": [10.0, 30.0][:hours], "Netherlands": [40.0, 60.0][:hours],
                         "Ireland": [80.0, 100.0][:hours]}, index=idx)


def _network(hours=2, pricing=None):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    for bus in ("GB", "Ext_FR", "Ext_NL"):
        n.add("Bus", bus)
    cables = pd.DataFrame([
        {"name": "FR", "from_bus": "GB", "to_bus": "Ext_FR", "capacity_mw": 300.0, "losses_percent": 2.5,
         "counterparty_country": "France"},
        {"name": "NL", "from_bus": "GB", "to_bus": "Ext_NL", "capacity_mw": 100.0, "losses_percent": 2.5,
         "counterparty_country": "Netherlands"},
    ])
    add_interconnector_links(n, cables, pd.DataFrame(), pd.DataFrame(), modelled_year=2030,
                             entsoe_prices_df=_entsoe(hours), pricing_config=pricing or _pricing())
    return n


def test_continental_zones_share_one_capacity_weighted_price():
    cables = pd.DataFrame({"counterparty_country": ["France", "Netherlands"], "capacity_mw": [300.0, 100.0]})
    shapes = single_price_market_shapes(_entsoe(), cables, _pricing())
    assert shapes["continent"].tolist() == pytest.approx([17.5, 37.5])


def test_single_reference_is_shape_plus_market_offset_and_price_case_scales_level():
    shape = pd.Series([10.0, 30.0])
    assert single_price_reference(shape, "France", _pricing(), -500, 500).tolist() == pytest.approx([15.0, 35.0])
    high = _pricing(future_price_case="high", price_case_multipliers={"high": 1.5})
    # level (20 + 5) x 1.5 = 37.5; the shape's deviations are kept.
    assert single_price_reference(shape, "France", high, -500, 500).tolist() == pytest.approx([27.5, 47.5])
    with pytest.raises(ValueError):
        single_price_reference(shape, "Norway", _pricing(), -500, 500)


def test_links_trade_at_one_price_with_the_spread_and_keep_cp29_tranche_levels():
    n = _network()
    for link in ("IC_FR", "IC_NL"):
        ask, bid = get_piecewise_market_prices(n, link, n.snapshots)
        assert ask.tolist() == pytest.approx([23.0, 43.0])       # 17.5/37.5 + 5 + 0.5
        assert bid.tolist() == pytest.approx([22.0, 42.0])
        assert not bool(n.links.at[link, "market_separate_import_export_prices"])
        assert n.links.at[link, "market_import_level"] == pytest.approx(89.43)
        assert n.links.at[link, "market_export_level"] == pytest.approx(40.09)


def test_negative_prices_widen_the_spread_so_ask_covers_the_eta_squared_export_slope():
    ref = pd.Series([-200.0, -10.0, 50.0])
    ask, bid = _foreign_bid_ask_prices(ref, 0.975, {"transaction_spread_gbp_per_mwh": 1.0})
    assert (ask >= export_efficiency(0.975) * bid - 1e-9).all()
    assert (ask - bid).iloc[-1] == pytest.approx(1.0)


def test_export_earns_eta_times_bid_per_mwh_gb_sends():
    n = _network(hours=1)
    n.add("Generator", "wind", bus="GB", p_nom=1000.0, marginal_cost=0.0)
    n.add("Load", "demand", bus="GB", p_set=0.0)
    status, _ = n.optimize(solver_name="highs", extra_functionality=build_piecewise_interconnector_cost_callback(n))
    assert status == "ok"
    p = n.links_t.p0.iloc[0][["IC_FR", "IC_NL"]]
    assert (p < 0).all()                                   # both export at the one price
    sent = float(n.links_t.p1.iloc[0][["IC_FR", "IC_NL"]].sum())   # GB withdrawal (p1 = -eta x p0)
    assert sent == pytest.approx(0.975 * 400.0)
    income = -float(interconnector_trade_cost(n.links_t.p0[["IC_FR", "IC_NL"]], *link_price_tables(n)).sum().sum())
    assert income == pytest.approx(0.975 * 22.0 * sent)   # eta x bid on what GB sends


def test_callback_records_the_stage1_trade_value_for_bm_reporting():
    n = _network()
    schedule = pd.DataFrame({"IC_FR": [100.0, -50.0], "IC_NL": [0.0, 20.0]}, index=n.snapshots)
    links, ask_t = link_price_tables(n)
    expected = float(interconnector_trade_cost(schedule, links, ask_t).sum().sum())
    assert baseline_trade_value(schedule, links, ask_t, n.snapshots) == pytest.approx(expected)
    assert expected == pytest.approx(100 * 23.0 - 50 * 0.975 ** 2 * 42.0 + 20 * 43.0)
    n.add("Generator", "gas", bus="GB", p_nom=1000.0, marginal_cost=90.0)
    n.add("Load", "demand", bus="GB", p_set=100.0)
    cb = build_piecewise_interconnector_cost_callback(n, baseline_dispatch=schedule, subtract_native_link_cost=True)
    assert stage1_interconnector_trade_value(n) == 0.0
    status, _ = n.optimize(solver_name="highs", extra_functionality=cb)
    assert status == "ok"
    assert stage1_interconnector_trade_value(n) == pytest.approx(expected)


def test_single_basis_trades_through_its_calibrated_hurdle():
    n = _network(pricing=_pricing(single_price_transaction_spread_gbp_per_mwh=10.0))
    ask, bid = get_piecewise_market_prices(n, "IC_FR", n.snapshots)
    assert ask.tolist() == pytest.approx([27.5, 47.5])        # 17.5/37.5 + 5 + 10/2
    assert bid.tolist() == pytest.approx([17.5, 37.5])
    assert n.links.at["IC_FR", "market_spread_half"] == pytest.approx(5.0)


def test_separate_basis_ignores_the_single_hurdle():
    pricing = _pricing(foreign_price_basis="separate", single_price_transaction_spread_gbp_per_mwh=10.0)
    n = _network(pricing=pricing)
    assert bool(n.links.at["IC_FR", "market_separate_import_export_prices"])
    assert n.links.at["IC_FR", "market_spread_half"] == pytest.approx(0.5)


def test_cp30_basis_per_country_prices_each_country_at_its_own_level():
    """Item F13: own-level file, one price per country (not per coupled market)."""
    n = _network(pricing=_pricing(foreign_price_source="cp30_basis", single_price_per_country=True))
    fr_ask, fr_bid = get_piecewise_market_prices(n, "IC_FR", n.snapshots)
    nl_ask, _ = get_piecewise_market_prices(n, "IC_NL", n.snapshots)
    assert fr_ask.tolist() == pytest.approx([15.5, 35.5])      # France 10/30 + continent offset 5 + 0.5
    assert fr_bid.tolist() == pytest.approx([14.5, 34.5])
    assert nl_ask.tolist() == pytest.approx([45.5, 65.5])      # Netherlands 40/60 + 5 + 0.5
    # Own level: the tranche level is the country's own mean, not CP.29.
    assert n.links.at["IC_FR", "market_import_level"] == pytest.approx(25.0)
    assert n.links.at["IC_NL", "market_import_level"] == pytest.approx(55.0)


def test_cp30_basis_without_per_country_keeps_one_continental_price():
    n = _network(pricing=_pricing(foreign_price_source="cp30_basis"))
    for link in ("IC_FR", "IC_NL"):
        ask, _ = get_piecewise_market_prices(n, link, n.snapshots)
        assert ask.tolist() == pytest.approx([23.0, 43.0])
